from pathlib import Path
import sys
import pandas as pd
import numpy as np
import subprocess
import threading
import argparse


parser = argparse.ArgumentParser(
    description= 
    """
    Preprocessing tool to help remove PHI. Default settings:
    - Remove clinical notes that are not EEG records
    - Remove all clinical notes' headers that may include patient information
    - Run Philter-UCSF to redact any additional PHI with '*'
    """
    )
parser.add_argument("-i", "--input", type=str, default="../data/EEG Notes ALL time Run 2_13_2026 1.xlsx", help="Input spreadsheet file name/directory")
parser.add_argument("-o", "--output", type=str, default="EEG_Notes_preprocessed.xlsx", help="Output spreadsheet file name/directory")
parser.add_argument("-t","--threads", type=int, default=4, help="Input number of threads if using Philter-UCSF deidentification (default)")
parser.add_argument("--include_PHI",action='store_true', help='Keep PHI. Includes full length of note (no header-removal), and does not run Philer-UCSF')
parser.add_argument("--max_chop",action='store_true', help="Reduce the clinical note to just the shortest length possible (likely the impressions or summary section), if present.")
parser.add_argument("--time_filter",action='store_true', help="Only include records from 2/12/24-2/12/26")
parser.add_argument("--subset", "-s", required=False, help="path to .csv file with specific note_id's to select for")
args = parser.parse_args()


class EEGNotes:

    def __init__(self, filename: str):
        _init_df = pd.read_excel(filename)
        _init_df.columns = _init_df.columns.str.lower()
        _init_df["note_text"] = _init_df["note_text"].astype(str)
        _init_df["note_id"] = _init_df["note_id"].astype(str)

        self._data = _init_df
        self._num_notes = None

    def get_note_count(self) -> int:
        return self._data.shape[0]
    
    def save_to_excel(self, filename: str):        
        self._data.to_excel(filename)
    
    def get_df(self):
        return self._data

    def collapse_note(self):

        def collapse(sub_df):
            _row = sub_df.iloc[0].copy()
            _row["line"] = sub_df["line"].max()
            _full_text = " ".join(sub_df["note_text"].dropna().astype(str))
            _row["note_text"] = _full_text.strip()
            return _row

        _collapsed = (
            self._data
            .sort_values(by=["note_id", "line"])
            .groupby("note_id", as_index=False)
            .apply(collapse, include_groups=False)
            .reset_index(drop=True)
            )

        self._data = _collapsed
        self._num_notes = self._data.shape[0]

    def subset(self):
        note_ids = pd.read_csv(args.subset, header=None).squeeze()
        print(note_ids.shape[0])
        note_ids = note_ids.astype(str)
        self._data = self._data.astype(str)
        self._data = self._data[self._data['note_id'].isin(set(note_ids))]


    def time_filter(self):
        self._data["spec_time_loc_dttm"] = pd.to_datetime(self._data["spec_time_loc_dttm"])
        self._data = self._data[self._data["spec_time_loc_dttm"].between(pd.Timestamp('2024-02-12'), pd.Timestamp('2026-02-12'))]
        self._num_notes = self._data.shape[0]

    def filter_non_eeg(self):
        text = self._data["note_text"].str

        mask = (
            text.contains("EEG", na=False)
            & ~text.contains("neonatal", case=False, na=False)
            & ~text.contains("nicu", case=False, na=False)
            & ~text.contains("interventional neurophysiology service", case=False, na=False)
            & ~text.contains("Electronic Analysis and programming of RNS", case=False, na=False)
        )

        self._data = self._data[mask].reset_index(drop=True)
        self._num_notes = self._data.shape[0]


    def remove_header(self):

        def header(row):
            full_text = row
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
                    full_text = full_text[start:].strip()
                    break
            return full_text
        
        self._data["note_text"] = self._data["note_text"].apply(header)

    def max_chop(self):

        def chop(row):
            full_text = row
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
                    full_text = full_text[start:].strip()
                    break
            return full_text
        
        self._data["note_text"] = self._data["note_text"].apply(chop)

    def philter(self, threads):
        # Check for Philter-UCSF
        cmd = """
        if [ -d 'philter-ucsf' ]; then
            echo 'philter-ucsf already  exists.'
        else
            echo 'philter-ucsf does not exist. Cloning Repo.'
            git clone https://github.com/BCHSI/philter-ucsf.git
        fi
        """
        subprocess.run(cmd,shell=True)

        # Check Python Version
        assert sys.version_info[:2] <= (3, 10), f"Philter incompatible with Python {sys.version}"

        # Check NLTK    
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
            for i in range(threads)
        ]

        OUTPUT_ROOT = PROJECT_ROOT / "notes_processed_txt"
        OUTPUT_ROOT.mkdir(exist_ok=True)

        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

        # vectorize relevant info
        note_ids = self._data["note_id"].to_numpy()
        texts = self._data["note_text"].to_numpy()

        # evenly split into  batches & export as .txt
        batch_indices = np.array_split(np.arange(len(note_ids)), threads)

        for batch_idx, indices in enumerate(batch_indices):
            batch_dir = INPUT_ROOT / f"notes_batch_{batch_idx+1}"

            for i in indices:
                with open(batch_dir / f"{note_ids[i]}.txt", "w", encoding="utf-8") as f:
                    f.write(f"{texts[i]}")

        # Launch threads        
        processes = []

        def stream_process_output(process, batch_name):
            """Read stdout of a process line by line and print it with batch prefix."""
            for line in iter(process.stdout.readline, ""):
                if line:
                    print(f"[{batch_name}] {line.strip()}")
            process.stdout.close()

        print(f"Launching {threads} threads...")    
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

        #### CREATE NEW DF WITH PHI REMOVED
        from concurrent.futures import ThreadPoolExecutor

        # List out all processed .txt files, put into numpy arrays, convert to df, save as .xlsx
        def read_note(path):
            return path.stem, path.read_text(encoding="utf-8", errors="replace")

        files = sorted(OUTPUT_ROOT.glob("*.txt")) # sorting for reproducibility

        with ThreadPoolExecutor(max_workers=threads) as pool:
            rows = list(pool.map(read_note, files))

        self._data = pd.DataFrame(rows, columns=["note_id", "note_text"])

        ### DELETE BATCH FILES
        cmd = """
        rm -rf notes_txt
        rm -rf notes_processed_txt
        """

        subprocess.run(cmd,shell=True)




## PROCESSING
notes = EEGNotes(args.input)

if "line" in notes.get_df().columns:
    notes.collapse_note()
    print("Collapsed detected multi-line notes.")

if args.subset:
    notes.subset()
    print('Filtered for inputed subset of note ids.')
else:
    print(f"Number of ALL records: {notes.get_note_count()}")
    notes.filter_non_eeg()
    print(f"Number of EEG records: {notes.get_note_count()}")

if args.time_filter:
    notes.time_filter()
    print(f"Number of time-filtered EEG records: {notes.get_note_count()}")

if args.max_chop:
    notes.max_chop()
    print('Reduced notes to shortest possible length.')

if not args.include_PHI:
    if not args.max_chop:
        notes.remove_header()
        print("Removed header.")
    print("Preparing Philter-UCSF")
    notes.philter(threads=args.threads)
    print("Finished Running Philter-UCSF")
    

notes.save_to_excel(filename=args.output)
print(f"Saved spreadsheet as {args.output}")