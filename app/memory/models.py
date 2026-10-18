"""
app/memory/models.py
--------------------
Data models and utility functions for persistent research memory.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Optional
from zoneinfo import ZoneInfo

from app.research.report import ResearchReport, SourceReference

VALID_JOB_STATUSES = {"queued", "running", "completed", "failed"}
VALID_TRIGGER_TYPES = {"manual", "scheduled", "manual_schedule"}


def normalize_topic(topic: str) -> str:
    """
    Normalize a topic string for case-insensitive and whitespace-insensitive matching.

    Examples:
        "  AI   Developments  " → "ai developments"
        "Latest Developments in AI" → "latest developments in ai"
    """
    if not topic:
        return ""
    cleaned = topic.strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def to_display_time(utc_iso: Optional[str], tz_name: str = "Asia/Kolkata") -> Optional[str]:
    """Convert a UTC ISO string to formatted display time in the requested timezone."""
    if not utc_iso:
        return None
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        target_tz = ZoneInfo(tz_name)
        localized = dt.astimezone(target_tz)
        return localized.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return utc_iso


@dataclass
class StoredResearchRun:
    """
    Represents a stored research run database record.
    Wraps ResearchReport with database ID and metadata.
    """
    id: int
    report: ResearchReport
    report_path: str


@dataclass
class ResearchJobRecord:
    """
    Persistent record representing the full lifecycle of a research job.
    """
    id: str
    topic: str
    trigger_type: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    schedule_id: Optional[str] = None
    research_run_id: Optional[int] = None
    report_path: Optional[str] = None
    whats_new_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        self.topic = (self.topic or "").strip()
        if not self.topic:
            raise ValueError("Research job topic must not be empty.")
        if self.status not in VALID_JOB_STATUSES:
            raise ValueError(f"Invalid job status {self.status!r}. Must be one of {sorted(VALID_JOB_STATUSES)}.")
        if self.trigger_type not in VALID_TRIGGER_TYPES:
            raise ValueError(f"Invalid trigger type {self.trigger_type!r}. Must be one of {sorted(VALID_TRIGGER_TYPES)}.")

    def calculate_duration(self) -> Optional[float]:
        """Calculate duration from started_at and completed_at if not explicitly stored."""
        if self.duration_seconds is not None:
            return round(self.duration_seconds, 2)
        if self.started_at and self.completed_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                end = datetime.fromisoformat(self.completed_at)
                return round((end - start).total_seconds(), 2)
            except Exception:
                pass
        return None

    def to_dict(self, tz_name: str = "Asia/Kolkata") -> dict:
        """Convert job record to serializable dictionary with UTC and display timestamps."""
        duration = self.calculate_duration()
        return {
            "id": self.id,
            "topic": self.topic,
            "trigger_type": self.trigger_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "schedule_id": self.schedule_id,
            "research_run_id": self.research_run_id,
            "report_path": self.report_path,
            "whats_new_path": self.whats_new_path,
            "error": self.error,
            "duration_seconds": duration,
            # Localized display fields
            "created_at_display": to_display_time(self.created_at, tz_name),
            "started_at_display": to_display_time(self.started_at, tz_name),
            "completed_at_display": to_display_time(self.completed_at, tz_name),
        }
