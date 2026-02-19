#!/usr/bin/env python3
"""
Clinical note classifier (single-file)
- Uses ollama Python library (no requests)
- SQLite checkpointing + resume
- One inference per note (extract multiple feature labels in one JSON)
- Dynamic feature set (update FEATURES list any time; run_id changes automatically)
- Outputs ONLY labels (true/false/null). No evidence/confidence.

Example:
  python ollama_classify_labels_only.py \
    --input /path/to/EEG_PHI_removed.xlsx \
    --text_col note_text \
    --id_col note_id \
    --model llama3.2 \
    --output_dir /expanse/lustre/scratch/$USER/temp_project/ollama/results \
    --db_path /expanse/lustre/scratch/$USER/temp_project/ollama/results/results.sqlite \
    --prompt_version v1

Notes:
- Requires an Ollama server reachable via OLLAMA_HOST or default http://127.0.0.1:11434
- Prior to executing this script, start ollama with "export OLLAMA_CONTEXT_LENGTH=16384 ollama serve &"
- If your ollama python package errors on format="json", remove that kwarg; prompt still enforces JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

import ollama


# ----------------------------
# CONFIG: Dynamic features list
# ----------------------------
# Edit this list anytime. Each item must have a stable "key".
FEATURES: List[Dict[str, str]] = [
    {"key": "seizure", "desc": "true only for explicit mentions of seizures recorded on the EEG ('electrographic' or 'epileptic' types) in the impressions portion; negations => false. History of seizures => false. Nonepileptic seizures => false."},
    # {"key": "IIC", "desc": "true only for explicit mentions of ictal-interictal-continuum (IIC) EEG rhythyms, which lie between normal and epileptic."},
    {"key": "status_epilepticus", "desc": "true only for explicit phrases like 'status epilepticus'; negations => false."},
    {"key": "ncse", "desc": "true only for explicit phrases like 'nonconvulsive status epilepticus'; negations => false"},
    {"key": "epileptiform_discharges", "desc": "true only for explicit mentions of epileptiform spikes, discharges, spike and slow waves, or sharp waves; negations => false."},
    {"key": "diffuse_nonspecific_abnormalities", "desc": "true for generalized, widespread patterns of slowed brain waves, usually mentioned as 'generalized background slowing' or 'diffuse nonspecific abnormalities'."},
    {"key": "focal_slowing", "desc": "true only for ecplicit mentions of focal slowing or focal dysfunction."},
]

# Default generation options (tweak as needed)
DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    # "num_ctx": 8192,  # optional, if your model supports
}


# ----------------------------
# Pydantic models (labels only)
# ----------------------------

class NoteLabels(BaseModel):
    # dynamic: map feature_key -> true/false/null
    features: Dict[str, bool]

SCHEMA = NoteLabels.model_json_schema()

# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_run_id(model: str, prompt_version: str, features: List[Dict[str, str]]) -> str:
    payload = {"model": model, "prompt_version": prompt_version, "features": features}
    return stable_hash(payload)[:16]


def init_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def create_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            run_id TEXT NOT NULL,
            note_id TEXT NOT NULL,
            status TEXT NOT NULL,          -- ok | error
            labels_json TEXT,              -- JSON dict {feature_key: true/false/null}
            raw_response TEXT,             -- raw model text (optional)
            error TEXT,                    -- error string (optional)
            elapsed_s REAL,  
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, note_id)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_results_run_status ON results(run_id, status);")
    con.commit()


def get_completed_note_ids(con: sqlite3.Connection, run_id: str) -> set[str]:
    rows = con.execute(
        "SELECT note_id FROM results WHERE run_id = ? AND status='ok';",
        (run_id,),
    ).fetchall()
    return {r[0] for r in rows}


def upsert_result(
    con: sqlite3.Connection,
    run_id: str,
    note_id: str,
    status: str,
    model: str,
    prompt_version: str,
    labels_json: Optional[str] = None,
    raw_response: Optional[str] = None,
    error: Optional[str] = None,
    elapsed_s: Optional[float] = None,
) -> None:
    now = utc_now_iso()
    con.execute(
        """
        INSERT INTO results (run_id, note_id, status, labels_json, raw_response, error, elapsed_s, model, prompt_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, note_id) DO UPDATE SET
            status=excluded.status,
            labels_json=excluded.labels_json,
            raw_response=excluded.raw_response,
            error=excluded.error,
            elapsed_s=excluded.elapsed_s,
            model=excluded.model,
            prompt_version=excluded.prompt_version,
            updated_at=excluded.updated_at;
        """,
        (run_id, note_id, status, labels_json, raw_response, error, elapsed_s, model, prompt_version, now, now),
    )
    con.commit()  # safest checkpointing: commit each note


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# ----------------------------
# Prompting
# ----------------------------

def build_system_prompt(features: List[Dict[str, str]]) -> str:
    feat_lines = "\n".join([f'- "{f["key"]}": {f["desc"]}' for f in features])
    return f"""
Task:
You are a neurologist specializing in epilepsy reading and annotating encephelography (EEG) reports. 
You must return what key findings are present. Use the following annotation guide to understand the terminology:
{feat_lines}

Output rules (STRICT):
- Return ONLY valid JSON (no markdown, no extra text).
- Return only 1 of the following 3 labels: "true", "false" or "null". Do not include additional text.
- If a feature is not mentioned or uncertain, output null.
- Include every requested feature key exactly once in "features".

JSON schema:
{{
  "features": {{
    "<feature_key>": true | false | null,
    ...
  }}
}}
"""


def build_user_prompt(note_id: str, note_text: str, feature_keys: List[str]) -> str:
    keys = ", ".join(feature_keys)
    return f"""note_id: {note_id}

Requested features: {keys}

EEG report to annotate:
\"\"\"{note_text}\"\"\"
"""


# ----------------------------
# Ollama call + retry
# ----------------------------

def call_ollama_chat(
    client: ollama.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict[str, Any],
) -> str:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # If your version errors on this kwarg, remove it:
        format=SCHEMA,
        options=options,
    )
    try:
        return resp["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected ollama response shape: {type(resp)} {resp}") from e


def infer_one_note_with_retries(
    client: ollama.Client,
    model: str,
    note_id: str,
    note_text: str,
    features: List[Dict[str, str]],
    options: Dict[str, Any],
    max_retries: int,
    base_backoff_s: float,
    wallclock_timeout_s: int,
) -> Tuple[NoteLabels, str]:
    system_prompt = build_system_prompt(features)
    feature_keys = [f["key"] for f in features]
    user_prompt = build_user_prompt(note_id, note_text, feature_keys)

    t0 = time.time()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        if (time.time() - t0) > wallclock_timeout_s:
            raise TimeoutError(f"Wallclock timeout exceeded ({wallclock_timeout_s}s) for note_id={note_id}")

        try:
            raw = call_ollama_chat(client, model, system_prompt, user_prompt, options)
            obj = extract_json_object(raw)

            # Force note_id (avoid occasional model drift)
            obj["note_id"] = str(note_id)

            parsed = NoteLabels.model_validate(obj)

            # Ensure all requested keys exist; fill missing with null
            for k in feature_keys:
                if k not in parsed.features:
                    parsed.features[k] = None

            # Optionally drop any extra keys the model invented
            parsed.features = {k: parsed.features.get(k, None) for k in feature_keys}

            # Ensure values are strictly bool/None (guard against "yes"/"no")
            for k, v in parsed.features.items():
                if v is None or isinstance(v, bool):
                    continue
                raise ValidationError(f"Feature {k} must be bool or null; got {type(v)}", NoteLabels)

            return parsed, raw

        except Exception as e:
            last_err = e
            logging.warning("Attempt %d failed for note_id=%s: %s", attempt + 1, note_id, repr(e))
            if attempt >= max_retries:
                break
            time.sleep(base_backoff_s * (2 ** attempt))

    assert last_err is not None
    raise last_err


# ----------------------------
# Output shaping
# ----------------------------

def flatten_labels(run_id: str, rows: List[Tuple[str, str]], feature_keys: List[str]) -> pd.DataFrame:
    """
    rows: (note_id, labels_json_str, elapsed_s) where labels_json_str is dict {feature_key: true/false/null}
    Output columns: run_id, note_id, <feature_key>, elapsed_s
    """
    records: List[Dict[str, Any]] = []
    for note_id, labels_json, elapsed_s in rows:
        try:
            d = json.loads(labels_json) if labels_json else {}
        except Exception:
            d = {}
        rec = {"run_id": run_id, "note_id": note_id, "elapsed_s": elapsed_s}
        for k in feature_keys:
            rec[k] = d.get(k, None)
        records.append(rec)
    return pd.DataFrame(records)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input spreadsheet (.xlsx/.xls or .csv)")
    ap.add_argument("--output_dir", default="results", help="Directory to write outputs")
    ap.add_argument("--db_path", default="results/results.sqlite", help="SQLite checkpoint DB path (e.g., results.sqlite)")
    ap.add_argument("--model", required=True, help="Ollama model name (e.g., llama3.2)")
    ap.add_argument("--prompt_version", default="v1", help="Bump when you change prompts/features behavior")
    ap.add_argument("--id_col", default="note_id", help="Column name for note IDs")
    ap.add_argument("--text_col", default="note_text", help="Column name for note text")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of notes (0 = no limit)")
    ap.add_argument("--verbosity", "-v", action="count", default=0)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--timeout_s", type=int, default=600, help="Per-note wallclock timeout")
    args = ap.parse_args()

    init_logger(args.verbosity)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    db_path = Path(args.db_path)
    con = connect_sqlite(db_path)
    create_tables(con)

    # Load input
    inp = Path(args.input)
    if not inp.exists():
        logging.error("Input not found: %s", inp)
        return 2

    if inp.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(inp)
    elif inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
    else:
        logging.error("Unsupported input extension: %s", inp.suffix)
        return 2

    df.columns = [c.strip() for c in df.columns]
    if args.id_col not in df.columns or args.text_col not in df.columns:
        logging.error("Missing required columns. Have: %s", list(df.columns))
        logging.error("Need id_col=%s and text_col=%s", args.id_col, args.text_col)
        return 2

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    run_id = compute_run_id(args.model, args.prompt_version, FEATURES)
    feature_keys = [f["key"] for f in FEATURES]
    logging.info("run_id=%s model=%s prompt_version=%s", run_id, args.model, args.prompt_version)

    completed = get_completed_note_ids(con, run_id)
    logging.info("Already completed: %d notes for this run_id", len(completed))

    client = ollama.Client()

    total = len(df)
    done = skipped = errors = 0

    for _, row in df.iterrows():
        note_id = str(row[args.id_col])
        note_text = row[args.text_col]

        if note_id in completed:
            skipped += 1
            continue

        if pd.isna(note_text) or not str(note_text).strip():
            errors += 1
            upsert_result(
                con=con,
                run_id=run_id,
                note_id=note_id,
                status="error",
                model=args.model,
                prompt_version=args.prompt_version,
                labels_json=None,
                raw_response=None,
                error="Empty note text",
                elapsed_s=None,
            )
            continue

        try:
            t0 = time.perf_counter()
            parsed, raw = infer_one_note_with_retries(
                client=client,
                model=args.model,
                note_id=note_id,
                note_text=str(note_text),
                features=FEATURES,
                options=DEFAULT_OPTIONS,
                max_retries=args.max_retries,
                base_backoff_s=2.0,
                wallclock_timeout_s=args.timeout_s,
            )
            elapsed_s = time.perf_counter() - t0

            upsert_result(
                con=con,
                run_id=run_id,
                note_id=note_id,
                status="ok",
                model=args.model,
                prompt_version=args.prompt_version,
                labels_json=json.dumps(parsed.features, ensure_ascii=False),
                raw_response=raw,
                error=None,
                elapsed_s=elapsed_s,
            )
            done += 1
            if done % 50 == 0:
                logging.info("Progress: done=%d/%d skipped=%d errors=%d", done, total, skipped, errors)

        except Exception as e:
            errors += 1
            elapsed_s = time.perf_counter() - t0
            logging.error("Failed note_id=%s: %s", note_id, repr(e))
            upsert_result(
                con=con,
                run_id=run_id,
                note_id=note_id,
                status="error",
                model=args.model,
                prompt_version=args.prompt_version,
                labels_json=None,
                raw_response=None,
                error=repr(e),
                elapsed_s=elapsed_s,
            )

    # Export OK results for this run_id
    rows = con.execute(
        "SELECT note_id, labels_json, elapsed_s FROM results WHERE run_id=? AND status='ok' ORDER BY note_id;",
        (run_id,),
    ).fetchall()

    df_out = flatten_labels(run_id, rows, feature_keys)

    xlsx_path = out_dir / f"ollama_labels_{run_id}.xlsx"
    df_out.to_excel(xlsx_path, index=False)
    logging.info("Wrote: %s", xlsx_path)

    # Optional parquet
    parquet_path = out_dir / f"ollama_labels_{run_id}.parquet"
    try:
        df_out.to_parquet(parquet_path, index=False)
        logging.info("Wrote: %s", parquet_path)
    except Exception as e:
        logging.info("Parquet export skipped (missing pyarrow/fastparquet?): %s", repr(e))

    # Metadata
    meta_path = out_dir / f"run_meta_{run_id}.json"
    meta = {
        "run_id": run_id,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "features": FEATURES,
        "options": DEFAULT_OPTIONS,
        "db_path": str(db_path),
        "input": str(inp),
        "export_xlsx": str(xlsx_path),
        "export_parquet": str(parquet_path),
        "finished_at_utc": utc_now_iso(),
        "counts": {"total_rows": total, "new_ok": done, "skipped_ok": skipped, "errors": errors},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote: %s", meta_path)

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
