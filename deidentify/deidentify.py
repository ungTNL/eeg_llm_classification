from pathlib import Path
import sys
import pandas as pd
import numpy as np
import subprocess
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default="../data/EEG Notes 12_23_2025.xlsx", help="Input spreadsheet file name")
parser.add_argument("-n","--num_batch", type=int, default=4, help="Input number of batches")
parser.add_argument("--include_PHI",action='store_true', help='Keep PHI. Includes full length of note, and does not run Philer-UCSF')
parser.add_argument("--max_chop",action='store_true', help="Reduce the clinical note to just the impressions section (shortest length possible), if present. Can be run with --include_PHI or default setting that runs Philter-UCSF.")
parser.add_argument("--time_filter",action='store_true', help="Only include records from 2/12/24-2/12/26")
args = parser.parse_args()


#### EXTRACT NOTES WITH PHI
# Read in .xlsx file into a DataFrame
df = pd.read_excel(args.file)
df.columns = df.columns.str.lower()

# update data types
df["note_text"] = df["note_text"].astype(str)
df["note_id"] = df["note_id"].astype(str)

if args.time_filter:
    df["spec_time_loc_dttm"] = pd.to_datetime(df["spec_time_loc_dttm"])
    df = df[df["spec_time_loc_dttm"].between(pd.Timestamp('2024-02-12'), pd.Timestamp('2026-02-12'))]

non_eeg_count  = 0

def process_note(g):
    """
    Main function that collapses multi-line notes and removes the PHI header, 
    or maximally chops down to the "impressions" section, if present.
    """
    global non_eeg_count

    if "line" in df.columns:
        #collapse multi-line notes
        row = g.iloc[0].copy()
        row["line"] = g["line"].max()
        full_text = " ".join(g["note_text"].dropna().astype(str))
    else:
        full_text = g

    # Remove rows that are not EEG records (e.g., lumbar punctures)
    if "EEG" not in full_text:
        non_eeg_count += 1
        return None
    
    # Remove rows that are neonatal EEG's
    if "neonatal" in full_text.lower() or "NICU" in full_text:
        non_eeg_count += 1
        return None

    # check for starter words if max chopping or only removing header
    if args.max_chop:
        WORDS = ["summary",
                 "summary of findings",
                 "impression",
                 "findings",
                 "technique",
                 "recording method",
                 "conditions of the recording"]
        for w in WORDS:
            start = full_text.lower().find(w)
            if start >= 0:
                full_text = full_text[start:]
                break
    elif not args.include_PHI:
        WORDS = ["conditions of the recording", 
                "recording method", 
                "technique",
                "findings",
                "impression",
                "summary of findings",
                "summary"]   
        for w in WORDS:
            start = full_text.lower().find(w)
            if start >= 0:
                full_text = full_text[start:]
                break

    if "line" in df.columns:
        row["note_text"] = full_text.strip()
        return row   
    else:
        return full_text.strip()

if "line" in df.columns:
    df_sorted = df.sort_values(by=["note_id", "line"])
    df_merged = (
    df_sorted
    .groupby("note_id", as_index=False)
    .apply(process_note, include_groups=False)
    .dropna(subset=["note_text"])
    .reset_index(drop=True)
    )
else:
    df_merged = df
    df_merged["note_text"] = df["note_text"].apply(process_note)
    df_merged.dropna(subset=["note_text"]).reset_index(drop=True)   


records = df_merged.shape[0]

print(f"Non-EEG records removed: {non_eeg_count}")
print(f"EEG records recoverd: {records}")



if args.include_PHI:
    # Don't remove header or run Philter
    filename = "EEG_PHI_Included.xlsx"
    df_merged[['note_id','note_text']].to_excel(filename)
    print(f"Saved clinical notes with PHI included as '{filename}'.")
else:
    ####  RUN PHILTER-UCSF
    #### CHECK IF PHILTER-UCSF IS AVAILABLE
    cmd = """
    if [ -d 'philter-ucsf' ]; then
        echo 'philter-ucsf already  exists.'
    else
        echo 'philter-ucsf does not exist. Cloning Repo.'
        git clone https://github.com/BCHSI/philter-ucsf.git
    fi
    """

    subprocess.run(cmd,shell=True)
    # Check Python version
    assert sys.version_info[:2] <= (3, 10), f"Philter incompatible with Python {sys.version}"

    #### CHECK IF NLTK ENG DOWNLOAD IS AVAILABLE
    import nltk
    def is_tagger_downloaded():
        print("Checking for NLTK tagger.")
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            return True
        except LookupError:
            return False

    if is_tagger_downloaded():
        print('NLTK tagger already exists.')
    else: 
        print('NLTK tagger does not exist. Downloading.')
        nltk.download('averaged_perceptron_tagger_eng')

    # Define and Create Directories
    PROJECT_ROOT = Path(__file__).resolve().parent
    PHILTER_ROOT = PROJECT_ROOT / "philter-ucsf"
    CONFIG_PATH = PHILTER_ROOT / "configs" / "philter_delta.json"

    INPUT_ROOT = PROJECT_ROOT / "notes_txt"
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)

    batch_dirs = [
        (INPUT_ROOT/ f"notes_batch_{i+1}").mkdir(exist_ok=True) or
        (INPUT_ROOT / f"notes_batch_{i+1}")
        for i in range(args.num_batch)
    ]

    OUTPUT_ROOT = PROJECT_ROOT / "notes_processed_txt"
    OUTPUT_ROOT.mkdir(exist_ok=True)

    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    # vectorize relevant info
    note_ids = df_merged["note_id"].to_numpy()
    texts = df_merged["note_text"].to_numpy()

    # evenly split into  batches
    batch_indices = np.array_split(np.arange(len(note_ids)), args.num_batch)

    # export as .txt
    for batch_idx, indices in enumerate(batch_indices):
        batch_dir = INPUT_ROOT / f"notes_batch_{batch_idx+1}"

        for i in indices:
            with open(batch_dir / f"{note_ids[i]}.txt", "w", encoding="utf-8") as f:
                f.write(f"{texts[i]}")

    # Launch batch processes            
    processes = []

    def stream_process_output(process, batch_name):
        """Read stdout of a process line by line and print it with batch prefix."""
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"[{batch_name}] {line.strip()}")
        process.stdout.close()

    print(f"Launching {args.num_batch} batch jobs...")    
    for batch_dir in sorted(INPUT_ROOT.glob("notes_batch_*")):
        batch_name = batch_dir.name  # e.g. notes_batch_1
        batch_log_dir = INPUT_ROOT / batch_name
        batch_log_dir.mkdir(parents=True, exist_ok=True)

        assert (INPUT_ROOT / batch_name).exists()

        cmd = [
        "python3",
        "main.py",
        "-i", str(INPUT_ROOT / batch_name),
        "-o", str(OUTPUT_ROOT),
        "-f", str(CONFIG_PATH),
        "--prod=True",
        "--outputformat", "asterisk"
    ]

        p = subprocess.Popen(
            cmd,
            cwd=str(PHILTER_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        t = threading.Thread(target=stream_process_output, args=(p, batch_name))
        t.start()

        processes.append((p, t))

    for p, t in processes:
        p.wait()
        t.join()  # wait for the output thread to finish

    print("Complete! Generating new spreadsheet with PHI removed...")

    #### CREATE NEW DF WITH PHI REMOVED
    from concurrent.futures import ThreadPoolExecutor

    # List out all processed .txt files, put into numpy arrays, convert to df, save as .xlsx
    def read_note(path):
        return path.stem, path.read_text(encoding="utf-8", errors="replace")

    files = sorted(OUTPUT_ROOT.glob("*.txt")) # sorting for reproducibility

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(read_note, files))

    df = pd.DataFrame(rows, columns=["note_id", "note_text"])

    filename = "EEG_PHI_Removed.xlsx"
    df.to_excel(filename)
    print(f"Saved clinical notes with PHI removed as '{filename}'.")


    ### DELETE BATCH FILES
    cmd = """
    rm -rf notes_txt
    rm -rf notes_processed_txt
    """

    subprocess.run(cmd,shell=True)
