"""SQLite prediction-history store (Postgres-compatible interface)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import artifact_paths, load_config


def get_db_path(config: dict | None = None) -> Path:
    cfg = config or load_config()
    p = artifact_paths(cfg)["predictions_db"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db(config: dict | None = None) -> Path:
    p = get_db_path(config)
    with sqlite3.connect(p) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              customer_id TEXT, churn_probability REAL,
              created_at TEXT)""")
    return p


def log_prediction(customer_id: str, churn_probability: float, config: dict | None = None) -> None:
    p = init_db(config)
    with sqlite3.connect(p) as con:
        con.execute(
            "INSERT INTO predictions(customer_id, churn_probability, created_at) VALUES(?,?,?)",
            (customer_id, churn_probability, datetime.now(timezone.utc).isoformat()),
        )
