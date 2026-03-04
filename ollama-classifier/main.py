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
import json
import logging
import time
from pathlib import Path

import pandas as pd
import ollama


# custom modules
from call_retry import infer_one_note_with_retries
from utilities import *
from config import *



# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input spreadsheet (.xlsx/.xls or .csv)")
    ap.add_argument("--output_dir", "-o", default="results", help="Directory to write outputs")
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
