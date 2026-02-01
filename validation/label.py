import pandas as pd
import textwrap
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default="../deidentify/EEG_PHI_removed.xlsx", help="Input spreadsheet file name")
args = parser.parse_args()

WRAP_WIDTH = 100
SAMPLE_SIZE = 1000
RANDOM_STATE = 42
CHECKPOINT_PATH = Path("ground_truth_labels_partial.xlsx")

# ------------------------
# Helper for safe input
# ------------------------
def get_binary_label(prompt: str) -> int:
    """
    Prompt user for a binary label (0/1) or quit (q).
    Returns 0 or 1.
    Exits program cleanly on 'q'.
    """
    while True:
        val = input(f"{prompt} (0/1 or q to quit): ").strip().lower()
        if val == "q":
            print("\n[Quitting — progress saved.]\n")
            sys.exit(0)
        if val in {"0", "1"}:
            return int(val)
        print("Invalid input. Please enter 0, 1, or q.")

# ------------------------
# Load data
# ------------------------
df = pd.read_excel(args.file)
print("Unique entries:\n", df.nunique(), "\n")

# Fixed random subset
v_set = (
    df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
      .reset_index(drop=True)
)

# Load checkpoint if present
if CHECKPOINT_PATH.exists():
    labels_df = pd.read_excel(CHECKPOINT_PATH)
    print(f"Loaded checkpoint with {len(labels_df)} labeled rows.\n")
else:
    labels_df = pd.DataFrame(columns=[
        "note_id",
        "seizure",
        "status_epilepticus",
        "epileptiform_discharges",
        "NCSE",
    ])
    print("No checkpoint found. Starting fresh.\n")

labeled_ids = set(labels_df["note_id"])
total = len(v_set)

# ------------------------
# Labeling loop
# ------------------------
for _, row in v_set.iterrows():
    note_id = row["note_id"]

    if note_id in labeled_ids:
        continue

    note_text = row["note_text"]
    wrapped_text = textwrap.fill(note_text, width=WRAP_WIDTH)

    current = len(labeled_ids) + 1
    print(f"\nValidation {current}/{total}")
    print(f"{note_id}:\n{wrapped_text}\n")

    seizure = get_binary_label("Presence of seizure")
    status_epilepticus = get_binary_label("Status epilepticus")
    epileptiform = get_binary_label(
        "Epileptiform spikes, discharges, spike-and-slow waves, or sharp waves"
    )
    ncse = get_binary_label("Nonconvulsive status epilepticus")

    new_row = {
        "note_id": note_id,
        "seizure": bool(seizure),
        "status_epilepticus": bool(status_epilepticus),
        "epileptiform_discharges": bool(epileptiform),
        "NCSE": bool(ncse),
    }

    labels_df = pd.concat(
        [labels_df, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    # Save after every row
    labels_df.to_excel(CHECKPOINT_PATH, index=False)
    labeled_ids.add(note_id)

    print(f"[Checkpoint saved: {len(labeled_ids)} / {total}]")

# ------------------------
# Final merge
# ------------------------
ground_truth_df = v_set.merge(labels_df, on="note_id", how="left")
ground_truth_df.to_excel("ground_truth_labels.xlsx", index=False)

print("\nAll labels complete. Final dataset saved.")

