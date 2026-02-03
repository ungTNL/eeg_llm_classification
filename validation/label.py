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

def get_modifier_label(prompt: str) -> str:
    """
    Prompt user for a modifier label ('mild', 'moderate', or 'severe') or quit (q).
    Returns string
    Exits program cleanly on 'q'.
    """
    while True:
        val = input(f"{prompt} ('mild', 'mild-moderate', 'moderate', 'moderate-severe', 'severe', 'unknown', or q to quit): ").strip().lower()
        if val == "q":
            print("\n[Quitting — progress saved.]\n")
            sys.exit(0)
        if val in {"mild", "moderate", "severe", "mild-moderate", "moderate-severe", "unknown"}:
            return val
        print("Invalid input. Please enter 'mild', 'moderate', 'severe','mild-moderate', 'moderate-severe', 'unknown' or q.")

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
        "SZ",
        "clinical_SZ",
        "nonclinical_SZ",
        "nonconvulsive_SZ",
        "focal_onset_SZ",        
        "SE",
        "NCSE",
        "CSE",
        "epileptiform_discharges",
        "dominant_freq",
        "diffuse_nonspecific_abnormalities",
        "diffuse_nonspecific_abnormalities_modifier",
        "focal_slowing",
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

    # note_text = row["note_text"]
    note_text = "" if pd.isna(row.get("note_text")) else str(row.get("note_text"))
    wrapped_text = textwrap.fill(note_text, width=WRAP_WIDTH)

    current = len(labeled_ids) + 1
    print(f"\nValidation {current}/{total}")
    print(f"{note_id}:\n{wrapped_text}\n")

    SZ = get_binary_label("Presence of seizure")
    if SZ == 0:
        clinical_SZ = \
        nonclinical_SZ = \
        nonconvulsive_SZ = \
        focal_onset_SZ = \
        SE = \
        NCSE = \
        CSE = 0
        print("Marking all seizure-related labels as 0 since no seizure present. \n")
    else:
        clinical_SZ = get_binary_label("Clinical seizure?")
        nonclinical_SZ = get_binary_label("Nonclinical seizure")
        nonconvulsive_SZ = get_binary_label("Nonconvulsive seizure")
        focal_onset_SZ = get_binary_label("Focal onset seizure")
        SE = get_binary_label("Status epilepticus")
        NCSE = get_binary_label("Nonconvulsive status epilepticus")
        CSE = get_binary_label("Convulsive status epileptius")
    epileptiform_discharges = get_binary_label(
        "Epileptiform spikes, discharges, spike-and-slow waves, or sharp waves"
    )
    dominant_freq = input("Dominant Frequencies: ")
    diffuse_nonspecific_abnormalities = get_binary_label("Diffuse nonspecific abnormalities")
    if diffuse_nonspecific_abnormalities == 1:
        diffuse_nonspecific_abnormalities_modifier = get_modifier_label("Diffuse nonspecific abnormalities modifier")
    focal_slowing = get_binary_label("Focal slowing")

    new_row = {
        "note_id": note_id,
        "SZ": bool(SZ),
        "clinical_SZ": bool(clinical_SZ),
        "nonclinical_SZ": bool(nonclinical_SZ),
        "nonconvulsive_SZ": bool(nonconvulsive_SZ),
        "focal_onset_SZ": bool(focal_onset_SZ),
        "SE": bool(SE),
        "NCSE": bool(NCSE),
        "CSE": bool(CSE),
        "epileptiform_discharges": bool(epileptiform_discharges),
        "dominant_freq": str(dominant_freq),
        "diffuse_nonspecific_abnormalities": bool(diffuse_nonspecific_abnormalities),
        "diffuse_nonspecific_abnormalities_modifier": str(diffuse_nonspecific_abnormalities_modifier),
        "focal_slowing": bool(focal_slowing)
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

