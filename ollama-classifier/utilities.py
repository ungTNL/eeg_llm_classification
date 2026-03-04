from datetime import datetime, timezone
import hashlib
import json
import logging
import sqlite3
import re
import pandas as pd


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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