"""
app/services/research_service.py
---------------------------------
Unified research execution service shared by manual research, scheduled jobs, and Run Now.
SQLite is the single authoritative source of truth for all job lifecycle states.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import config
from app.memory.comparator import ResearchComparator
from app.memory.models import ResearchJobRecord, normalize_topic, utc_now_iso
from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent, _safe_filename
from app.research.report import ResearchReport

logger = logging.getLogger(__name__)


def _background_execute(job_id: str, db_path) -> None:
    """
    Top-level wrapper for background task execution.
    Creates a fresh repository bound to db_path so the background thread
    is fully independent of the global service instance.
    Any uncaught exception is persisted as a 'failed' status.
    """
    repo = ResearchRepository(db_path=db_path)
    service = ResearchExecutionService(repository=repo)
    try:
        service.execute_job(job_id)
    except Exception as exc:
        logger.error(
            "[JobLifecycle] Unhandled exception in background task for job %s: %s",
            job_id, exc, exc_info=True,
        )
        # Last-resort: mark the job failed so it never stays 'running' forever
        try:
            repo.update_job(job_id, status="failed", error=f"Background task crashed: {exc}")
        except Exception as db_err:
            logger.error("[JobLifecycle] Could not persist crash status for job %s: %s", job_id, db_err)


class DuplicateJobError(Exception):
    """Raised when an active (queued or running) job already exists for the given topic or schedule."""


class ResearchExecutionService:
    """
    Central execution service managing the complete lifecycle of research jobs.

    Guarantees:
      - SQLite database as single source of truth.
      - Safe status transitions: queued -> running -> completed | failed.
      - Concurrent request and duplicate-run protection via process lock and atomic DB verification.
      - Consistent UTC ISO 8601 timestamps and accurate duration tracking.
      - Recurring schedule preservation and one-time job deactivation.
      - Robust error recording without silently swallowing exceptions.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        repository: Optional[ResearchRepository] = None,
        agent: Optional[ResearchAgent] = None,
    ) -> None:
        self.repository = repository or ResearchRepository()
        self.agent = agent

    def create_and_enqueue_job(
        self,
        topic: str,
        trigger_type: str = "manual",
        schedule_id: Optional[str] = None,
    ) -> ResearchJobRecord:
        """
        Create and persist a new job in 'queued' status.
        Atomically prevents duplicate active runs for the same normalized topic or schedule.
        """
        topic = (topic or "").strip()
        if not topic:
            raise ValueError("Research topic must not be empty.")

        with self._lock:
            # Check for active running or queued job for this topic
            if self.repository.has_active_job_for_topic(topic):
                raise DuplicateJobError(
                    f"A research job is already active for topic: {topic!r}."
                )

            # Verify schedule_id exists in DB to satisfy foreign key constraint
            valid_schedule_id = None
            if schedule_id:
                if self.repository.has_active_job_for_schedule(schedule_id):
                    raise DuplicateJobError(
                        f"A research job is already active for schedule: {schedule_id!r}."
                    )
                if self.repository.get_schedule(schedule_id):
                    valid_schedule_id = schedule_id

            safe_topic = _safe_filename(topic)
            uid = uuid.uuid4().hex[:10]
            if trigger_type == "scheduled":
                job_id = f"job-sched-{safe_topic}-{uid}"
            elif trigger_type == "manual_schedule":
                job_id = f"job-runnow-{safe_topic}-{uid}"
            else:
                job_id = f"job-manual-{safe_topic}-{uid}"

            now_iso = utc_now_iso()
            job = ResearchJobRecord(
                id=job_id,
                topic=topic,
                trigger_type=trigger_type,
                status="queued",
                created_at=now_iso,
                updated_at=now_iso,
                schedule_id=valid_schedule_id,
            )
            self.repository.create_job(job)
            logger.info(
                "[JobLifecycle] [QUEUED] job_id=%s | topic=%r | trigger=%s | schedule_id=%s",
                job.id,
                topic,
                trigger_type,
                schedule_id,
            )
            return job

    def execute_job(self, job_id: str) -> ResearchJobRecord:
        """
        Execute an enqueued research job through the full research pipeline.
        Manages the transition: running -> completed | failed.
        """
        job = self.repository.get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id!r} not found.")

        if job.status == "completed":
            logger.info("[JobLifecycle] Job %s is already completed. Skipping execution.", job_id)
            return job

        if job.status == "running":
            raise DuplicateJobError(f"Job {job_id!r} is currently running.")

        started_at = utc_now_iso()
        self.repository.update_job(job_id, status="running", started_at=started_at)
        logger.info(
            "[JobLifecycle] [RUNNING] job_id=%s | topic=%r | trigger=%s | started_at=%s",
            job_id,
            job.topic,
            job.trigger_type,
            started_at,
        )

        start_dt = datetime.fromisoformat(started_at)

        try:
            # Step 1: Capture previous baseline research BEFORE new research
            logger.info("[JobLifecycle] Step 1/4: Checking for previous baseline research...")
            previous_run = self.repository.get_latest_research(job.topic)
            if previous_run:
                logger.info(
                    "[JobLifecycle] Previous baseline research found (generated: %s).",
                    previous_run.generated_at.isoformat(),
                )
            else:
                logger.info("[JobLifecycle] No previous baseline research found (establishing initial baseline).")

            # Step 2: Execute ResearchAgent workflow
            logger.info("[JobLifecycle] Step 2/4: Executing research workflow (Search -> RAG -> Gemini)...")
            research_agent = self.agent or ResearchAgent(repository=self.repository)
            current_report = research_agent.research(job.topic)

            safe_topic = _safe_filename(job.topic)
            report_path = str(config.REPORTS_DIR / f"{safe_topic}.md")
            logger.info(
                "[JobLifecycle] Research report generated: %d sources, %d findings.",
                len(current_report.sources),
                len(current_report.key_findings),
            )

            # In case the agent was mocked and didn't save the report to SQLite
            latest_check = self.repository.get_latest_research(job.topic)
            if not latest_check and current_report:
                try:
                    self.repository.save_research_run(current_report, report_path=report_path)
                except Exception as save_err:
                    logger.warning("[JobLifecycle] Could not persist report to SQLite: %s", save_err)

            self._last_report = current_report


            # Step 3: Compare with previous baseline
            logger.info("[JobLifecycle] Step 3/4: Generating 'What's New' comparison report...")
            comparator = ResearchComparator()
            whats_new_report = comparator.compare(current_report, previous_run)

            # Step 4: Persist What's New report to disk
            logger.info("[JobLifecycle] Step 4/4: Persisting 'What's New' report...")
            whats_new_filename = f"{safe_topic}-whats-new.md"
            whats_new_path = config.REPORTS_DIR / whats_new_filename
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            whats_new_path.write_text(whats_new_report.to_markdown(), encoding="utf-8")
            whats_new_path_str = str(whats_new_path)

            # Step 5: Associate actual research run ID.
            # Use started_at as a lower bound so we match the run created
            # by THIS execution, not a prior run for the same topic.
            research_run_id = self.repository.get_research_run_id_since(
                topic=job.topic, since_iso=started_at
            )
            if research_run_id is None:
                # Fallback: pick the absolute latest for this topic
                research_run_id = self.repository.get_latest_research_run_id(job.topic)
            logger.info(
                "[JobLifecycle] Resolved research_run_id=%s for job %s",
                research_run_id, job_id,
            )

            # Step 6: Handle one-time schedule deactivation
            if job.schedule_id:
                try:
                    sched = self.repository.get_schedule(job.schedule_id)
                    if sched and sched.frequency == "once":
                        self.repository.update_schedule(job.schedule_id, enabled=False)
                        logger.info(
                            "[JobLifecycle] Deactivated completed one-time schedule '%s'.",
                            job.schedule_id,
                        )
                except Exception as sched_err:
                    logger.warning(
                        "[JobLifecycle] Could not deactivate schedule '%s': %s",
                        job.schedule_id,
                        sched_err,
                    )

            # Step 7: Finalize job as completed
            completed_at = utc_now_iso()
            end_dt = datetime.fromisoformat(completed_at)
            duration = round((end_dt - start_dt).total_seconds(), 2)

            self.repository.update_job(
                job_id,
                status="completed",
                completed_at=completed_at,
                research_run_id=research_run_id,
                report_path=report_path,
                whats_new_path=whats_new_path_str,
                duration_seconds=duration,
            )

            logger.info(
                "[JobLifecycle] [COMPLETED] job_id=%s | topic=%r | duration=%.2fs | run_id=%s | completed_at=%s",
                job_id,
                job.topic,
                duration,
                research_run_id,
                completed_at,
            )

            updated_job = self.repository.get_job(job_id)
            return updated_job or job

        except Exception as exc:
            completed_at = utc_now_iso()
            end_dt = datetime.fromisoformat(completed_at)
            duration = round((end_dt - start_dt).total_seconds(), 2)
            err_msg = str(exc)

            logger.error(
                "[JobLifecycle] [FAILED] job_id=%s | topic=%r | duration=%.2fs | error=%s",
                job_id,
                job.topic,
                duration,
                err_msg,
                exc_info=True,
            )

            # Record failure in SQLite research_runs table so history shows it
            research_run_id = None
            try:
                failed_report = ResearchReport(
                    topic=job.topic,
                    executive_summary="",
                    key_findings=[],
                    detailed_analysis="",
                    emerging_trends=[],
                    conclusion="",
                    sources=[],
                    generated_at=start_dt,
                    error=f"Execution failed: {err_msg}",
                )
                research_run_id = self.repository.save_research_run(failed_report)
            except Exception as db_exc:
                logger.error("[JobLifecycle] Failed to record run failure in research_runs: %s", db_exc)

            self.repository.update_job(
                job_id,
                status="failed",
                completed_at=completed_at,
                research_run_id=research_run_id,
                error=err_msg,
                duration_seconds=duration,
            )

            updated_job = self.repository.get_job(job_id)
            return updated_job or job

    def run_job_sync(
        self,
        topic: str,
        trigger_type: str = "manual",
        schedule_id: Optional[str] = None,
    ) -> ResearchJobRecord:
        """
        Enqueues and executes a job synchronously. Returns the final ResearchJobRecord.
        """
        job = self.create_and_enqueue_job(
            topic=topic, trigger_type=trigger_type, schedule_id=schedule_id
        )
        return self.execute_job(job.id)


_global_execution_service: Optional[ResearchExecutionService] = None


def get_research_execution_service(
    repository: Optional[ResearchRepository] = None,
    agent: Optional[ResearchAgent] = None,
) -> ResearchExecutionService:
    """
    Get or create the global ResearchExecutionService instance.
    """
    global _global_execution_service
    if _global_execution_service is None or repository is not None or agent is not None:
        service = ResearchExecutionService(repository=repository, agent=agent)
        if repository is None and agent is None:
            _global_execution_service = service
        return service
    return _global_execution_service
