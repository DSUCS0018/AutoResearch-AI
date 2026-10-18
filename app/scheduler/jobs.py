"""
app/scheduler/jobs.py
---------------------
Scheduled research job execution logic and stable job ID generator.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from app.config import config
from app.memory.comparator import ResearchComparator
from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent, _safe_filename
from app.research.report import ResearchReport

logger = logging.getLogger(__name__)


def generate_stable_job_id(
    topic: str, frequency: str, day_of_week: Optional[str] = None
) -> str:
    """
    Generate a stable, deterministic job ID for a scheduled research topic.

    Examples:
        ("AI Trends", "daily") -> "research-ai-trends-daily"
        ("AI Papers", "weekly", "monday") -> "research-ai-papers-weekly-monday"
        ("AI Safety", "once") -> "research-ai-safety-once"
    """
    safe_topic = _safe_filename(topic)
    freq = (frequency or "daily").strip().lower()

    if freq == "weekly" and day_of_week:
        day = day_of_week.strip().lower()
        return f"research-{safe_topic}-weekly-{day}"
    if freq == "once":
        return f"research-{safe_topic}-once"
    return f"research-{safe_topic}-daily"


def execute_scheduled_research_job(
    job_id: str,
    topic: str,
    repository: Optional[ResearchRepository] = None,
    agent: Optional[ResearchAgent] = None,
    trigger_type: str = "scheduled",
) -> None:
    """
    Executes a scheduled research job using the unified ResearchExecutionService.
    Guarantees:
      - Job record persisted before execution.
      - Safe status transitions in SQLite.
      - Duplicate execution protection.
      - Error details persisted on failure.
      - One-time schedules deactivated upon completion.
    """
    from app.services.research_service import (
        DuplicateJobError,
        ResearchExecutionService,
    )

    repo = repository or ResearchRepository()
    service = ResearchExecutionService(repository=repo, agent=agent)

    try:
        job = service.create_and_enqueue_job(
            topic=topic,
            trigger_type=trigger_type,
            schedule_id=job_id,
        )
    except DuplicateJobError as dup_err:
        logger.warning(
            "[Scheduler] Skipping duplicate scheduled job for topic %r (schedule: %s): %s",
            topic,
            job_id,
            dup_err,
        )
        return

    logger.info("[Scheduler] Executing scheduled job %s (Job record: %s)", job_id, job.id)
    service.execute_job(job.id)

