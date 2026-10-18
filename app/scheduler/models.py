"""
app/scheduler/models.py
-----------------------
Data models and validation logic for scheduled research jobs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

VALID_FREQUENCIES = {"daily", "weekly", "once"}
VALID_DAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


class ScheduleValidationError(ValueError):
    """Raised when schedule arguments are invalid."""


@dataclass
class ScheduledResearch:
    """
    Model representing a scheduled research job.
    """
    id: str
    topic: str
    frequency: str
    hour: int
    minute: int
    day_of_week: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.topic = (self.topic or "").strip()
        if not self.topic:
            raise ScheduleValidationError("Topic must not be empty.")

        self.frequency = (self.frequency or "").strip().lower()
        if self.frequency not in VALID_FREQUENCIES:
            raise ScheduleValidationError(
                f"Invalid frequency {self.frequency!r}. Must be 'daily', 'weekly', or 'once'."
            )

        if not isinstance(self.hour, int) or not (0 <= self.hour <= 23):
            raise ScheduleValidationError(
                f"Invalid hour {self.hour!r}. Must be an integer between 0 and 23."
            )

        if not isinstance(self.minute, int) or not (0 <= self.minute <= 59):
            raise ScheduleValidationError(
                f"Invalid minute {self.minute!r}. Must be an integer between 0 and 59."
            )

        if self.frequency == "weekly":
            if not self.day_of_week:
                raise ScheduleValidationError("Weekly schedule requires a day_of_week.")
            self.day_of_week = self.day_of_week.strip().lower()
            if self.day_of_week not in VALID_DAYS:
                raise ScheduleValidationError(
                    f"Invalid day_of_week {self.day_of_week!r}. Must be one of {sorted(VALID_DAYS)}."
                )
        else:
            self.day_of_week = None

    def to_dict(self) -> dict:
        """Convert schedule to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "frequency": self.frequency,
            "day_of_week": self.day_of_week,
            "hour": self.hour,
            "minute": self.minute,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "run_at": self.run_at,
        }
