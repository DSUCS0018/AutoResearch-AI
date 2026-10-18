"""
app/memory/database.py
----------------------
SQLite database initialization and connection handling.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from app.config import config


def get_db_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """
    Establish and return a SQLite connection with Row factory and PRAGMA settings.
    """
    target = db_path if db_path is not None else config.DATABASE_PATH
    if isinstance(target, Path) and str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str | Path] = None) -> None:
    """
    Create the SQLite tables and indexes if they do not already exist.
    Does NOT wipe or delete existing records.
    """
    conn = get_db_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS research_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            topic_normalized TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            report_path TEXT NOT NULL,
            executive_summary TEXT,
            detailed_analysis TEXT,
            conclusion TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_run_id INTEGER NOT NULL,
            source_identifier TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            FOREIGN KEY (research_run_id) REFERENCES research_runs (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS research_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_run_id INTEGER NOT NULL,
            finding_type TEXT NOT NULL,
            finding_text TEXT NOT NULL,
            FOREIGN KEY (research_run_id) REFERENCES research_runs (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scheduled_research (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            frequency TEXT NOT NULL,
            day_of_week TEXT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            run_at TEXT
        );

        CREATE TABLE IF NOT EXISTS research_jobs (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            topic_normalized TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            schedule_id TEXT,
            research_run_id INTEGER,
            report_path TEXT,
            whats_new_path TEXT,
            error TEXT,
            duration_seconds REAL,
            FOREIGN KEY (research_run_id) REFERENCES research_runs (id) ON DELETE SET NULL,
            FOREIGN KEY (schedule_id) REFERENCES scheduled_research (id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_research_runs_topic ON research_runs(topic_normalized);
        CREATE INDEX IF NOT EXISTS idx_research_jobs_status ON research_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_research_jobs_topic ON research_jobs(topic_normalized);
        CREATE INDEX IF NOT EXISTS idx_research_jobs_created ON research_jobs(created_at DESC);
        """)

        # Safely migrate existing databases that do not yet have run_at
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scheduled_research)")
        columns = [row["name"] for row in cursor.fetchall()]
        if columns and "run_at" not in columns:
            cursor.execute("ALTER TABLE scheduled_research ADD COLUMN run_at TEXT")

    conn.close()
