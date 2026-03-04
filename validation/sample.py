import argparse
import pandas as pd

ap = argparse.ArgumentParser(
    description= 
    """
    Random sampling tool
    """
)
ap.add_argument("--input", "-i", required=True, help="Input spreadsheet (.xlsx/.xls or .csv)")
ap.add_argument("--output","-o", default="EEG_notes_sample", help="Output spreadsheet name (.xlsx)")
ap.add_argument("--size", "-s", default=1000, help="Sample size. Default=1000")
ap.add_argument("--random_state","-r", default=42, help="Random state number. Default=42")
args = ap.parse_args()


df = pd.read_excel(args.input)
columns = ["note_id", "note_text"]
df2 = df[columns].sample(args.size, random_state=args.random_state)
df2.to_excel(args.output, index=False)