"""
app/scheduler/scheduler.py
--------------------------
SchedulerService encapsulating APScheduler lifecycle, job management,
and persistence synchronization.
"""

from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app.config import config
from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent, _safe_filename
from app.scheduler.jobs import execute_scheduled_research_job, generate_stable_job_id
from app.scheduler.models import ScheduleValidationError, ScheduledResearch

logger = logging.getLogger(__name__)


def _parse_time_str(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute) integers."""
    if not time_str or ":" not in time_str:
        raise ScheduleValidationError(
            f"Invalid time format {time_str!r}. Expected HH:MM e.g. '09:00'."
        )
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ScheduleValidationError(
            f"Invalid time format {time_str!r}. Expected HH:MM."
        )
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ScheduleValidationError(
            f"Non-integer hour/minute in {time_str!r}."
        ) from exc

    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ScheduleValidationError(
            f"Time {time_str!r} out of bounds (hour 0-23, minute 0-59)."
        )

    return h, m


class SchedulerService:
    """
    Service managing research schedules and APScheduler instance.
    """

    def __init__(
        self,
        repository: Optional[ResearchRepository] = None,
        agent: Optional[ResearchAgent] = None,
        timezone_name: Optional[str] = None,
    ) -> None:
        self.repository = repository or ResearchRepository()
        self.agent = agent
        self.timezone_name = timezone_name or config.SCHEDULER_TIMEZONE
        self.tz = self._resolve_timezone(self.timezone_name)

        logger.info(
            "[Scheduler] Initialized SchedulerService with timezone '%s' (offset %s)",
            self.timezone_name,
            datetime.now(self.tz).strftime("%z"),
        )

        self._scheduler = BackgroundScheduler(
            timezone=self.tz,
            job_defaults={
                "misfire_grace_time": 300,  # 5-minute grace window prevents missed jobs on minor lags
                "coalesce": True,           # Combine multiple missed triggers into a single run
                "max_instances": 2,         # Allow concurrent runs if a previous run is finishing
            },
        )
        self._setup_event_listeners()

    def _resolve_timezone(self, tz_name: str) -> ZoneInfo:
        """Resolve a timezone string to a valid ZoneInfo object with fallbacks."""
        try:
            return ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, Exception) as exc:
            logger.warning(
                "[Scheduler] Timezone %r could not be loaded directly (%s). Attempting local system timezone fallback.",
                tz_name,
                exc,
            )
            try:
                import tzlocal
                local_tz = tzlocal.get_localzone()
                resolved_key = getattr(local_tz, "key", str(local_tz))
                resolved = ZoneInfo(resolved_key)
                self.timezone_name = resolved_key
                logger.info("[Scheduler] Fallback succeeded: Using system local timezone '%s'", resolved_key)
                return resolved
            except Exception as fb_exc:
                logger.warning("[Scheduler] Local timezone resolution failed (%s). Falling back to UTC.", fb_exc)
                self.timezone_name = "UTC"
                return ZoneInfo("UTC")

    def _setup_event_listeners(self) -> None:
        """Register APScheduler event listeners for execution, error, and misfire logging."""
        def _on_event(event):
            sched_time = getattr(event, "scheduled_run_time", None)
            sched_str = sched_time.strftime("%Y-%m-%d %H:%M:%S %Z") if sched_time else "N/A"
            if event.code == EVENT_JOB_EXECUTED:
                logger.info(
                    "[Scheduler] EVENT: Job '%s' executed successfully (scheduled_time=%s).",
                    event.job_id,
                    sched_str,
                )
            elif event.code == EVENT_JOB_ERROR:
                logger.error(
                    "[Scheduler] EVENT: Job '%s' execution encountered an error (scheduled_time=%s, exception=%s).",
                    event.job_id,
                    sched_str,
                    event.exception,
                    exc_info=event.traceback,
                )
            elif event.code == EVENT_JOB_MISSED:
                logger.warning(
                    "[Scheduler] EVENT: Job '%s' MISSED scheduled run time (%s).",
                    event.job_id,
                    sched_str,
                )

        self._scheduler.add_listener(
            _on_event,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
        )

    def start(self) -> None:
        """Load enabled schedules from DB, register jobs, and start APScheduler."""
        logger.info("[Scheduler] Starting SchedulerService...")
        self.load_and_register_schedules()
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info(
                "[Scheduler] APScheduler background engine started successfully (Timezone: %s).",
                self.timezone_name,
            )
        else:
            logger.info("[Scheduler] APScheduler background engine already running.")

    def stop(self, wait: bool = False) -> None:
        """Shut down the APScheduler background thread."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("[Scheduler] SchedulerService stopped.")

    def load_and_register_schedules(self) -> None:
        """Load enabled schedules from SQLite and register with APScheduler."""
        enabled_schedules = self.repository.list_schedules(enabled_only=True)
        now_tz = datetime.now(self.tz)
        registered_count = 0

        for sched in enabled_schedules:
            # Handle one-time jobs whose scheduled time may have passed while server was stopped
            if sched.frequency == "once" and sched.run_at:
                try:
                    run_dt = datetime.fromisoformat(sched.run_at)
                    if run_dt.tzinfo is None:
                        run_dt = run_dt.replace(tzinfo=self.tz)
                    else:
                        run_dt = run_dt.astimezone(self.tz)

                    if run_dt < now_tz:
                        # Expired while server was offline; mark disabled
                        self.repository.update_schedule(sched.id, enabled=False)
                        logger.info(
                            "[Scheduler] One-time job '%s' scheduled time (%s) passed while server was offline; marked disabled.",
                            sched.id,
                            run_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                        )
                        continue
                except Exception as parse_err:
                    logger.warning("[Scheduler] Error parsing run_at for '%s': %s", sched.id, parse_err)

            self._register_apscheduler_job(sched)
            registered_count += 1

        logger.info(
            "[Scheduler] Registered %d enabled scheduled job(s) with APScheduler (Timezone: %s).",
            registered_count,
            self.timezone_name,
        )

    def add_schedule(
        self,
        topic: str,
        frequency: str,
        time_str: Optional[str] = None,
        day_of_week: Optional[str] = None,
        run_at: Optional[str] = None,
    ) -> ScheduledResearch:
        """
        Add a new scheduled research job.

        Validates inputs, generates stable job ID, persists to SQLite, and
        registers with APScheduler if enabled.
        """
        freq = (frequency or "daily").strip().lower()

        if freq == "once":
            now_tz = datetime.now(self.tz)
            if run_at:
                run_dt = datetime.fromisoformat(run_at)
                if run_dt.tzinfo is None:
                    run_dt = run_dt.replace(tzinfo=self.tz)
                else:
                    run_dt = run_dt.astimezone(self.tz)
            elif time_str:
                h, m = _parse_time_str(time_str)
                run_dt = now_tz.replace(hour=h, minute=m, second=0, microsecond=0)
                if run_dt <= now_tz:
                    run_dt += timedelta(days=1)
            else:
                run_dt = now_tz + timedelta(seconds=60)

            hour = run_dt.hour
            minute = run_dt.minute
            run_at_str = run_dt.isoformat()
            job_id = f"research-{_safe_filename(topic)}-once-{int(now_tz.timestamp())}"
        else:
            if not time_str:
                raise ScheduleValidationError("Time in HH:MM format is required for daily and weekly schedules.")
            hour, minute = _parse_time_str(time_str)
            job_id = generate_stable_job_id(topic, freq, day_of_week)
            run_at_str = None

        scheduled = ScheduledResearch(
            id=job_id,
            topic=topic,
            frequency=freq,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            enabled=True,
            run_at=run_at_str,
        )

        self.repository.save_schedule(scheduled)

        if self._scheduler.running and scheduled.enabled:
            self._register_apscheduler_job(scheduled)

        logger.info(
            "[Scheduler] Added schedule '%s' (topic: %r, frequency: %s, time: %02d:%02d %s).",
            job_id,
            topic,
            freq,
            hour,
            minute,
            self.timezone_name,
        )
        return scheduled

    def add_test_job(
        self,
        topic: str = "Safe Scheduler Test",
        delay_seconds: int = 60,
    ) -> ScheduledResearch:
        """
        Add a safe one-time test job scheduled to run in approximately `delay_seconds`
        from now (default: 60s).
        """
        now_tz = datetime.now(self.tz)
        run_dt = now_tz + timedelta(seconds=max(delay_seconds, 1))
        job_id = f"test-job-{_safe_filename(topic)}-{int(now_tz.timestamp())}"

        scheduled = ScheduledResearch(
            id=job_id,
            topic=topic,
            frequency="once",
            hour=run_dt.hour,
            minute=run_dt.minute,
            enabled=True,
            created_at=now_tz,
            run_at=run_dt.isoformat(),
        )

        self.repository.save_schedule(scheduled)

        if self._scheduler.running and scheduled.enabled:
            self._register_apscheduler_job(scheduled)

        aps_job = self._scheduler.get_job(job_id)
        next_run_dt = getattr(aps_job, "next_run_time", None) if aps_job else None
        next_run_str = (
            next_run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            if next_run_dt
            else run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        )

        logger.info(
            "[Scheduler] Scheduled one-time test job '%s' (Topic: %r) to run at %s (~%d seconds from now).",
            job_id,
            topic,
            next_run_str,
            delay_seconds,
        )
        return scheduled

    def remove_schedule(self, job_id: str) -> bool:
        """Remove a schedule from SQLite and APScheduler."""
        deleted = self.repository.delete_schedule(job_id)
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        logger.info("[Scheduler] Removed schedule '%s'. Deleted from SQLite: %s", job_id, deleted)
        return deleted

    def enable_schedule(self, job_id: str) -> bool:
        """Enable a schedule in SQLite and register it in APScheduler."""
        updated = self.repository.update_schedule(job_id, enabled=True)
        if updated:
            scheduled = self.repository.get_schedule(job_id)
            if scheduled:
                self._register_apscheduler_job(scheduled)
                aps_job = self._scheduler.get_job(job_id)
                next_run_dt = getattr(aps_job, "next_run_time", None) if aps_job else None
                next_run_str = (
                    next_run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                    if next_run_dt
                    else "Pending Start"
                )
                logger.info("[Scheduler] Enabled schedule '%s'. Next run: %s", job_id, next_run_str)
        return updated

    def disable_schedule(self, job_id: str) -> bool:
        """Disable a schedule in SQLite and remove it from APScheduler."""
        updated = self.repository.update_schedule(job_id, enabled=False)
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        logger.info("[Scheduler] Disabled schedule '%s'. Removed from active scheduler execution.", job_id)
        return updated

    def list_schedules(self, enabled_only: bool = False) -> list[dict]:
        """
        List all schedules with metadata and APScheduler next_run_time.
        """
        schedules = self.repository.list_schedules(enabled_only=enabled_only)
        result: list[dict] = []

        for s in schedules:
            job_info = s.to_dict()
            aps_job = self._scheduler.get_job(s.id)
            next_run_dt = getattr(aps_job, "next_run_time", None) if aps_job else None
            if next_run_dt:
                job_info["next_run"] = next_run_dt.strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
            else:
                job_info["next_run"] = "Not Scheduled" if not s.enabled else "Pending Start"
            result.append(job_info)

        return result

    def run_job_now(self, job_id: str) -> None:
        """Immediately trigger execution of a scheduled job."""
        scheduled = self.repository.get_schedule(job_id)
        if not scheduled:
            raise ValueError(f"No schedule found with ID {job_id!r}")

        logger.info("[Scheduler] Manual trigger invoked for job '%s'.", job_id)
        execute_scheduled_research_job(
            job_id=scheduled.id,
            topic=scheduled.topic,
            repository=self.repository,
            agent=self.agent,
        )

    # ── Private helper ────────────────────────────────────────────────────────

    def _register_apscheduler_job(self, scheduled: ScheduledResearch) -> None:
        if scheduled.frequency == "weekly" and scheduled.day_of_week:
            trigger = CronTrigger(
                day_of_week=scheduled.day_of_week[:3],
                hour=scheduled.hour,
                minute=scheduled.minute,
                second=0,
                timezone=self.tz,
            )
        elif scheduled.frequency == "once":
            if scheduled.run_at:
                try:
                    run_dt = datetime.fromisoformat(scheduled.run_at)
                except Exception:
                    run_dt = datetime.now(self.tz) + timedelta(seconds=60)
            else:
                run_dt = datetime.now(self.tz) + timedelta(seconds=60)

            if run_dt.tzinfo is None:
                run_dt = run_dt.replace(tzinfo=self.tz)
            else:
                run_dt = run_dt.astimezone(self.tz)

            trigger = DateTrigger(run_date=run_dt, timezone=self.tz)
        else:
            trigger = CronTrigger(
                hour=scheduled.hour,
                minute=scheduled.minute,
                second=0,
                timezone=self.tz,
            )

        job = self._scheduler.add_job(
            func=execute_scheduled_research_job,
            trigger=trigger,
            id=scheduled.id,
            name=scheduled.topic,
            replace_existing=True,
            kwargs={
                "job_id": scheduled.id,
                "topic": scheduled.topic,
                "repository": self.repository,
                "agent": self.agent,
            },
        )

        next_run_dt = getattr(job, "next_run_time", None)
        next_run_str = (
            next_run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            if next_run_dt
            else "Pending Start"
        )
        logger.info(
            "[Scheduler] Registered job '%s' | Topic: %r | Trigger: %s | Next Run: %s",
            scheduled.id,
            scheduled.topic,
            scheduled.frequency,
            next_run_str,
        )


_global_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """
    Get or instantiate the global application-wide SchedulerService instance.
    """
    global _global_scheduler_service
    if _global_scheduler_service is None:
        _global_scheduler_service = SchedulerService()
    return _global_scheduler_service


def set_scheduler_service(service: Optional[SchedulerService]) -> None:
    """
    Set or override the global SchedulerService instance (useful for testing or custom setup).
    """
    global _global_scheduler_service
    _global_scheduler_service = service
