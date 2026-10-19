"""Tiny SQLite persistence layer for generation jobs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    mode          TEXT NOT NULL,          -- text | image | multi_image | upload | remesh | retexture | repair | split
    prompt        TEXT,
    quality       TEXT NOT NULL DEFAULT 'standard',
    status        TEXT NOT NULL,          -- queued | running | succeeded | failed
    progress      INTEGER NOT NULL DEFAULT 0,
    stage         TEXT,
    error         TEXT,
    provider      TEXT NOT NULL,
    input_image   TEXT,                   -- file name of the reference image (uploaded or generated)
    model_file    TEXT,                   -- file name of the resulting .glb
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
"""

# Columns added after the first release; created on startup when missing.
MIGRATIONS = {
    "params": "TEXT NOT NULL DEFAULT '{}'",  # JSON: style, seed, remesh settings, input files...
    "parent_id": "TEXT",
    "name": "TEXT",
    "favorite": "INTEGER NOT NULL DEFAULT 0",
    "thumbnail": "TEXT",
}

UPDATABLE = {"status", "progress", "stage", "error", "input_image", "model_file", "name", "favorite", "thumbnail"}


class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
            for column, decl in MIGRATIONS.items():
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, **fields: Any) -> dict:
        now = time.time()
        fields = {"status": "queued", "progress": 0, "created_at": now, "updated_at": now, **fields}
        fields["params"] = json.dumps(fields.get("params") or {})
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._lock:
            self._conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", tuple(fields.values()))
            self._conn.commit()
        return self.get(fields["id"])  # type: ignore[return-value]

    def update(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"Cannot update fields: {unknown}")
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))
            self._conn.commit()

    def update_params(self, job_id: str, values: dict) -> None:
        """Merge `values` into the job's JSON params."""
        with self._lock:
            row = self._conn.execute("SELECT params FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return
            params = {**json.loads(row["params"] or "{}"), **values}
            self._conn.execute("UPDATE jobs SET params = ?, updated_at = ? WHERE id = ?",
                               (json.dumps(params), time.time(), job_id))
            self._conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _decode(row) if row else None

    def list(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_decode(r) for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def fail_interrupted(self) -> int:
        """Jobs left running/queued by a previous process can never finish; mark them failed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'Interrumpido por reinicio del servidor', updated_at = ?"
                " WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
            self._conn.commit()
        return cur.rowcount


def _decode(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["params"] = json.loads(job.get("params") or "{}")
    job["favorite"] = bool(job.get("favorite"))
    return job
