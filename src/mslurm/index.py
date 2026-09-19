"""Local SQLite index of jobs submitted from this machine.

It holds metadata only: what was submitted, from where, at which commit. Job
state always comes from the cluster, so losing this file loses nothing that
`mslurm queue` cannot show; it just makes bare ids and `mslurm pull` less convenient.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields

from mslurm.config import CACHE_DIR, STATE_DIR

DB_PATH = os.path.join(STATE_DIR, "jobs.db")
_OLD_DB_PATH = os.path.join(CACHE_DIR, "jobs.db")  # where the first version kept it

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    cluster    TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    command    TEXT NOT NULL,
    workdir    TEXT NOT NULL,
    code       TEXT NOT NULL,
    head       TEXT,
    dirty      INTEGER NOT NULL DEFAULT 0,
    local_root TEXT NOT NULL,
    subdir     TEXT NOT NULL DEFAULT '',
    submitted  TEXT NOT NULL,
    PRIMARY KEY (cluster, job_id)
);
"""


@dataclass
class JobRecord:
    cluster: str
    job_id: str
    name: str
    command: str
    workdir: str
    code: str
    head: str | None
    dirty: bool
    local_root: str
    subdir: str = ""
    submitted: str = ""


@contextmanager
def connect(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path == DB_PATH:
        _migrate(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(path: str) -> None:
    """Move a first-version index out of ~/.cache, and step aside from the pre-rewrite tool's database.

    The daemon-based mslurm kept an unrelated `jobs` table at this same path; it is
    renamed rather than reused, so its rows stay recoverable.
    """
    if os.path.exists(path) and not _is_ours(path):
        os.replace(path, path + ".pre-rewrite")
    if not os.path.exists(path) and os.path.exists(_OLD_DB_PATH):
        os.replace(_OLD_DB_PATH, path)


def _is_ours(path: str) -> bool:
    try:
        conn = sqlite3.connect(path)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
    return not columns or "job_id" in columns


def record(job: JobRecord, path: str = DB_PATH) -> None:
    job.submitted = job.submitted or time.strftime("%Y-%m-%dT%H:%M:%S")
    names = [f.name for f in fields(JobRecord)]
    with connect(path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO jobs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
            [getattr(job, n) for n in names],
        )


def get(cluster: str, job_id: str, path: str = DB_PATH) -> JobRecord | None:
    """The record for a job, falling back to the array master for task ids like 123_4."""
    base = job_id.split("_")[0].split("+")[0]
    with connect(path) as conn:
        for candidate in dict.fromkeys([job_id, base]):
            row = conn.execute("SELECT * FROM jobs WHERE cluster = ? AND job_id = ?", (cluster, candidate)).fetchone()
            if row:
                return _to_record(row)
    return None


def clusters_for(job_id: str, path: str = DB_PATH) -> list[str]:
    """Which clusters this machine has submitted a job with this id to."""
    base = job_id.split("_")[0].split("+")[0]
    with connect(path) as conn:
        rows = conn.execute("SELECT DISTINCT cluster FROM jobs WHERE job_id IN (?, ?)", (job_id, base)).fetchall()
    return [r["cluster"] for r in rows]


def recent(limit: int = 20, path: str = DB_PATH) -> list[JobRecord]:
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY submitted DESC LIMIT ?", (limit,)).fetchall()
    return [_to_record(r) for r in rows]


def _to_record(row: sqlite3.Row) -> JobRecord:
    data = dict(row)
    data["dirty"] = bool(data["dirty"])
    return JobRecord(**data)
