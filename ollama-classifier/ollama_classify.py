import os
import pandas as pd
import json
import time
import hashlib
import ollama
from tqdm import tqdm
tqdm.pandas()
from pydantic import BaseModel
import argparse
import sys

# INPUT ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

df_merged = pd.read_excel(args.input)
df_merged.columns = df_merged.columns.str.lower()

# ClASSES for output
class Report(BaseModel):
  seizure_label: bool | None
  SE_label: bool | None
  SE_text: str | None
  SE_text: str | None
  ED_label: bool | None
  ED_text: str | None
  NCSE_label: bool | None
  NCSE_text: str | None
  EMU_label: str | None
  EMU_text: str | None
  StudyNumber: str | None   

class ReportList(BaseModel):
    reports: list[Report]

client = ollama.Client()
# === CONFIG ===
OUTPUT_DIR = args.output_dir
MODEL = args.model    # set to your model identifier
BATCH_SIZE = 5         # tune: 8, 16, 32 ...
CHECKPOINT_EVERY = 50           # checkpoint every N batches
FINAL_XLSX = f"{OUTPUT_DIR}/classified_{MODEL}.xlsx"
OUTPUT_CHECKPOINT = f"{OUTPUT_DIR}/checkpoint_{MODEL}.csv"
TIMEOUT = 120
TEMP = 0.0
REPAIR_TIMEOUT = 30
RETRY = 1
SCHEMA = ReportList.model_json_schema()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for interactive session
is_tty = sys.stderr.isatty()

tqdm_kwargs = dict(
    file=sys.stderr,
    ascii=not is_tty,          # safer in logs
    dynamic_ncols=is_tty,      # only useful in terminals
    leave=not is_tty,          # keep final line in logs
    mininterval=240, 
    miniters=100
)

PROMPT = (
"""You are a strict medical document classifier. For each report below, return exactly one JSON object between <JSON> and </JSON>. 

Schema (exact keys):
- seizure_label: boolean - true/false
- seizure_text: string - short excerpt
- SE_label: boolean - true/false
- SE_text: string - short excerpt
- ED_label: boolean - true/false
- ED_text: string - short excerpt
- NCSE_label: boolean - true/false
- NCSE_text: string - short excerpt
- EMU_label: string - "Phase 1", "Phase 2", or "None"
- EMU_text: string - short excerpt
- StudyNumber: string - study number or "None", typically of format "12-3456" or "None"

Rules:
- seizure_label: true only for explicit mentions of seizures recorded on the EEG in the impressions portion; negations => false. History of seizures => false. Nonepileptic seizures => false.
- SE_label: true only for explicit phrases like "status epilepticus"; negations => false.
- ED_label: true only for explicit mentions of epileptiform spikes, discharges, spike and slow waves, or sharp waves; negations => false.
- NCSE_label: true only for explicit phrases like "nonconvulsive status epilepticus"; negations => false.
- If no seizures, then SE and NCSE must be false.
- *_text: verbatim excerpt (less than 10 words).
- EMU_label: "Phase 1" = EMU scalp EEG monitoring; "Phase 2" = intracranial/pre‑surgical; "None" = all other EEGs.
- StudyNumber: return if present, else "None".

Output format:
<JSON>{...}</JSON>   # classification for REPORT 1
<JSON>{...}</JSON>   # classification for REPORT 2
...

Example:
Report: "Study #12-3456. ICU EEG. no seizures recorded. "
Output:
<JSON>{"seizure_label": false, "seizure_confidence": 1.0, "seizure_text": "no seizures recorded", "SE_label": false, "SE_confidence": 1.0, "SE_text": "no status epilepticus", 
"ED_label": false, "ED_confidence": 1.0, "ED_text": "no epileptiform discharges", "NCSE_label": false, "NCSE_confidence": 1.0, "NCSE_text": "no seizures noted", "EMU_label": "None", "EMU_confidence": 1.0, "EMU_text": "ICU EEG", "StudyNumber": "12-3456"}</JSON>

"""
)  

def classify_batch(reports,note_ids):
    """
    reports: list[str] raw report texts
    returns: list[dict] same length, normalized fields
    """
    if not reports:
        return []

    # Build numbered blocks, safe quoting via json.dumps
    blocks = []
    for i, rpt in enumerate(reports, start=1):
        rpt_json = json.dumps(rpt)  # safely escapes quotes/newlines
        block = f"===REPORT {i}===\nREPORT:\n{rpt_json}\n===END {i}===\n"
        blocks.append(block)
    #collapse blocks into one string prompt
    merged_blocks = "\n".join(blocks)
    #Add prompt to merged_block
    multi_prompt = PROMPT + "\n\nReports:\n" + merged_blocks

    attempt = 0
    while attempt <= RETRY:
        attempt += 1
        try:
            resp = client.generate(prompt=multi_prompt, model=MODEL,format=SCHEMA, options={"temperature": TEMP})
            resp_reports = ReportList.model_validate_json(resp.response)
            return [r.model_dump() for r in resp_reports.reports]
        
        except Exception as e:
            # repair attempt: ask for corrected JSON blocks only
            print(f"  Attempt {attempt} failed: {e}, report note_id {note_ids}")

# === pipeline: batch over DataFrame with progress & checkpointing ===
def row_key(row):
    # prefer stable note_id; else hash note_text
    if "note_id" in row and not pd.isna(row["note_id"]):
        return f"ID:{row['note_id']}"
    h = hashlib.sha1(str(row.get("note_text","")).encode("utf-8")).hexdigest()
    return f"H:{h}"

#df_merged = df_merged.head(1000)  # for testing, limit to first 100 rows
# prepare to process
if os.path.exists(OUTPUT_CHECKPOINT):
    df_checkpoint = pd.read_csv(OUTPUT_CHECKPOINT)
    processed_keys = set(df_checkpoint["note_id"].tolist())
else:
    df_checkpoint = pd.DataFrame()
    processed_keys = set()

rows_to_process = []
for _, row in df_merged.iterrows():
    key = row_key(row)
    if key in processed_keys:
        continue
    rows_to_process.append((key, row.to_dict()))

total = len(rows_to_process)
print(f"Total to process: {total}")

# process batches with progress

# Step 1: Load existing checkpoint if present
if os.path.exists(OUTPUT_CHECKPOINT):
    df_existing = pd.read_csv(OUTPUT_CHECKPOINT)
    processed_ids = set(df_existing["note_id"].unique())
else:
    df_existing = None
    processed_ids = set()


all_rows = df_merged.to_dict(orient="records")
# Step 2: Filter rows to process
rows_to_process = [r for r in all_rows if r["note_id"] not in processed_ids]
total_rows = len(all_rows)          # total rows overall
already_done = len(processed_keys)  # how many already processed
remaining = len(rows_to_process)    # how many left

# Step 3: Progress bar with resume
results_buffer = []
with tqdm(total=total_rows, initial=already_done, desc="Rows", **tqdm_kwargs) as pbar:
    for i in range(0, remaining, BATCH_SIZE):
        batch = rows_to_process[i:i+BATCH_SIZE]
        reports = [r.get("note_text", "") for r in batch]
        note_ids = [r.get("note_id", "") for r in batch]

        batch_start = time.time()
        batch_results = classify_batch(reports, note_ids)
        batch_time = time.time() - batch_start

        # merge results with original row data and buffer
        for orig_row, out in zip(batch, batch_results):
            merged = {**orig_row, **out}
            results_buffer.append(merged)


        # periodic checkpoint flush
        if (i // BATCH_SIZE + 1) % CHECKPOINT_EVERY == 0:
            df_new = pd.DataFrame(results_buffer)
            if df_existing is not None:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined = df_combined.sort_values("note_id").drop_duplicates("note_id", keep="last")
            df_combined.to_csv(OUTPUT_CHECKPOINT, index=False)
            df_existing = df_combined
            results_buffer = []

        pbar.update(len(batch))
        pbar.set_postfix({"batch_time_s": f"{batch_time:.1f}"})

# final flush
if results_buffer:
    df_new = pd.DataFrame(results_buffer)
    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values("note_id").drop_duplicates("note_id", keep="last")
    df_combined.to_csv(OUTPUT_CHECKPOINT, index=False)

# export final Excel
df_final = pd.read_csv(OUTPUT_CHECKPOINT)
df_final.to_excel(FINAL_XLSX, index=False)
print("Done. Results written to", FINAL_XLSX)

