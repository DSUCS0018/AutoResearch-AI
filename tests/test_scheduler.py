"""
tests/test_scheduler.py
-----------------------
Unit tests for Stage 7: Automation & Scheduling (APScheduler + SQLite persistence).

All tests use mocks and temporary SQLite databases — no real Tavily, Gemini, or network calls.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent
from app.research.report import ResearchReport, SourceReference
from app.scheduler.jobs import execute_scheduled_research_job, generate_stable_job_id
from app.scheduler.models import ScheduleValidationError, ScheduledResearch
from app.scheduler.scheduler import SchedulerService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_report(topic="AI Trends"):
    return ResearchReport(
        topic=topic,
        executive_summary="Summary",
        key_findings=["Finding 1 [S1]"],
        detailed_analysis="Analysis",
        emerging_trends=["Trend 1 [S1]"],
        conclusion="Conclusion",
        sources=[SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/1")],
        generated_at=datetime.now(timezone.utc),
    )


# ── Test 1: Scheduler service initializes ─────────────────────────────────────

def test_scheduler_service_initialization(tmp_path):
    """1. Scheduler service initializes correctly."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)
    assert service is not None
    assert service._scheduler is not None


# ── Test 2: Daily schedule is created ─────────────────────────────────────────

def test_daily_schedule_creation(tmp_path):
    """2. Daily schedule is created."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="AI Daily", frequency="daily", time_str="09:00")

    assert sched.frequency == "daily"
    assert sched.hour == 9
    assert sched.minute == 0
    assert sched.id == "research-ai-daily-daily"


# ── Test 3: Weekly schedule is created ────────────────────────────────────────

def test_weekly_schedule_creation(tmp_path):
    """3. Weekly schedule is created."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(
        topic="AI Weekly", frequency="weekly", time_str="14:30", day_of_week="monday"
    )

    assert sched.frequency == "weekly"
    assert sched.day_of_week == "monday"
    assert sched.hour == 14
    assert sched.minute == 30
    assert sched.id == "research-ai-weekly-weekly-monday"


# ── Test 4: Invalid frequency is rejected ─────────────────────────────────────

def test_invalid_frequency_rejected(tmp_path):
    """4. Invalid frequency is rejected."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    with pytest.raises(ScheduleValidationError, match="Invalid frequency"):
        service.add_schedule(topic="Topic", frequency="hourly", time_str="09:00")


# ── Test 5: Invalid time is rejected ──────────────────────────────────────────

def test_invalid_time_rejected(tmp_path):
    """5. Invalid time is rejected."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    with pytest.raises(ScheduleValidationError, match="out of bounds"):
        service.add_schedule(topic="Topic", frequency="daily", time_str="25:99")


# ── Test 6: Schedule is persisted ─────────────────────────────────────────────

def test_schedule_persisted(tmp_path):
    """6. Schedule is persisted in SQLite."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="Persisted Topic", frequency="daily", time_str="10:00")

    db_sched = repo.get_schedule(sched.id)
    assert db_sched is not None
    assert db_sched.topic == "Persisted Topic"
    assert db_sched.hour == 10


# ── Test 7: Schedule can be retrieved ─────────────────────────────────────────

def test_schedule_retrieval(tmp_path):
    """7. Schedule can be retrieved by job ID."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="Retrieval Topic", frequency="daily", time_str="08:15")
    retrieved = repo.get_schedule(sched.id)

    assert retrieved is not None
    assert retrieved.id == sched.id
    assert retrieved.minute == 15


# ── Test 8: Schedule can be updated ───────────────────────────────────────────

def test_schedule_update(tmp_path):
    """8. Schedule can be updated (e.g. enable / disable)."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="Update Topic", frequency="daily", time_str="12:00")
    assert sched.enabled is True

    service.disable_schedule(sched.id)
    db_sched = repo.get_schedule(sched.id)
    assert db_sched.enabled is False

    service.enable_schedule(sched.id)
    db_sched_enabled = repo.get_schedule(sched.id)
    assert db_sched_enabled.enabled is True


# ── Test 9: Schedule can be deleted ───────────────────────────────────────────

def test_schedule_deletion(tmp_path):
    """9. Schedule can be deleted."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="Delete Topic", frequency="daily", time_str="12:00")
    deleted = service.remove_schedule(sched.id)

    assert deleted is True
    assert repo.get_schedule(sched.id) is None


# ── Test 10: Enabled schedule loads correctly ─────────────────────────────────

def test_enabled_schedule_loads_correctly(tmp_path):
    """10. Enabled schedule loads correctly into APScheduler."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    service.add_schedule(topic="Enabled Topic", frequency="daily", time_str="11:00")

    service.start()
    schedules = service.list_schedules(enabled_only=True)
    assert len(schedules) == 1
    assert schedules[0]["topic"] == "Enabled Topic"
    service.stop()


# ── Test 11: Disabled schedule does not run ───────────────────────────────────

def test_disabled_schedule_does_not_load(tmp_path):
    """11. Disabled schedule does not get registered in APScheduler."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)

    sched = service.add_schedule(topic="Disabled Topic", frequency="daily", time_str="11:00")
    service.disable_schedule(sched.id)

    service.start()
    schedules = service.list_schedules(enabled_only=True)
    assert len(schedules) == 0
    service.stop()


# ── Test 12: Stable job IDs prevent duplicate jobs ────────────────────────────

def test_stable_job_ids_prevent_duplicates(tmp_path):
    """12. Stable job IDs prevent duplicate jobs on restart."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    # Add same schedule twice
    s1 = SchedulerService(repository=repo)
    job1 = s1.add_schedule(topic="Duplicate Test", frequency="daily", time_str="09:00")

    s2 = SchedulerService(repository=repo)
    job2 = s2.add_schedule(topic="Duplicate Test", frequency="daily", time_str="09:00")

    assert job1.id == job2.id
    all_schedules = repo.list_schedules()
    assert len(all_schedules) == 1


# ── Test 13: Scheduled job calls research workflow ───────────────────────────

def test_scheduled_job_calls_research_workflow(tmp_path):
    """13. Scheduled job calls research workflow."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _make_report("Workflow Topic")

    execute_scheduled_research_job(
        job_id="test-job-id",
        topic="Workflow Topic",
        repository=repo,
        agent=agent_mock,
    )

    agent_mock.research.assert_called_once_with("Workflow Topic")


# ── Test 14: Scheduled job captures previous research before new research ─────

def test_scheduled_job_captures_previous_before_new(tmp_path):
    """14. Scheduled job captures previous research before executing new research."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    # Save initial report
    initial_report = _make_report("Order Topic")
    repo.save_research_run(initial_report)

    captured_previous = []

    def mock_research(topic):
        # When research runs, capture what get_latest_research returns
        latest = repo.get_latest_research(topic)
        captured_previous.append(latest)
        return _make_report("Order Topic")

    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.side_effect = mock_research

    execute_scheduled_research_job(
        job_id="test-job-order",
        topic="Order Topic",
        repository=repo,
        agent=agent_mock,
    )

    # Verify initial report was captured before new research run
    assert len(captured_previous) == 1
    assert captured_previous[0] is not None
    assert captured_previous[0].topic == "Order Topic"


# ── Test 15: Failed research job does not crash scheduler ──────────────────────

def test_failed_research_job_does_not_crash_scheduler(tmp_path, caplog):
    """15. Failed research job logs error and does not crash scheduler."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.side_effect = Exception("API connection dropped")

    # Should not raise exception
    execute_scheduled_research_job(
        job_id="failing-job-id",
        topic="Failing Topic",
        repository=repo,
        agent=agent_mock,
    )

    assert any("FAILED" in record.message.upper() for record in caplog.records)


# ── Test 16: Manual schedule trigger works ─────────────────────────────────────

def test_manual_schedule_trigger(tmp_path):
    """16. Manual schedule trigger works."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _make_report("Manual Topic")

    service = SchedulerService(repository=repo, agent=agent_mock)
    sched = service.add_schedule(topic="Manual Topic", frequency="daily", time_str="09:00")

    service.run_job_now(sched.id)
    agent_mock.research.assert_called_once_with("Manual Topic")


# ── Test 17: Timezone configuration works ──────────────────────────────────────

def test_timezone_configuration(tmp_path):
    """17. Timezone configuration works."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo, timezone_name="Asia/Kolkata")

    assert str(service.tz) == "Asia/Kolkata"


# ── Test 18: Failed research job records error in SQLite ──────────────────────

def test_failed_job_persists_error_to_database(tmp_path):
    """18. Failed research job records error in SQLite memory instead of swallowing it."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.side_effect = RuntimeError("Tavily API quota exceeded")

    execute_scheduled_research_job(
        job_id="test-failing-quota",
        topic="Quota Topic",
        repository=repo,
        agent=agent_mock,
    )

    # Verify that the run failure was saved to SQLite
    failed_report = repo.get_latest_research("Quota Topic")
    assert failed_report is not None
    assert failed_report.error is not None
    assert "Tavily API quota exceeded" in failed_report.error


# ── Test 19: Safe one-time test job is created and registered ─────────────────

def test_add_one_time_test_job(tmp_path):
    """19. Safe one-time test job is created with ~60s delay and registered with APScheduler."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = SchedulerService(repository=repo)
    service.start()

    sched = service.add_test_job(topic="Test 60s Execution", delay_seconds=60)

    assert sched.frequency == "once"
    assert sched.enabled is True
    assert sched.run_at is not None

    # Verify in DB
    db_sched = repo.get_schedule(sched.id)
    assert db_sched is not None
    assert db_sched.frequency == "once"

    # Verify registered in APScheduler with DateTrigger
    aps_job = service._scheduler.get_job(sched.id)
    assert aps_job is not None
    assert aps_job.next_run_time is not None
    service.stop()


# ── Test 20: One-time job execution marks schedule disabled ───────────────────

def test_one_time_job_completion_disables_schedule(tmp_path):
    """20. Execution of one-time job automatically marks schedule disabled in SQLite."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _make_report("Once Completed Topic")

    service = SchedulerService(repository=repo, agent=agent_mock)
    sched = service.add_test_job(topic="Once Completed Topic", delay_seconds=60)
    assert sched.enabled is True

    # Execute the job
    execute_scheduled_research_job(
        job_id=sched.id,
        topic=sched.topic,
        repository=repo,
        agent=agent_mock,
    )

    # Check that SQLite schedule is now disabled
    updated_sched = repo.get_schedule(sched.id)
    assert updated_sched is not None
    assert updated_sched.enabled is False


# ── Test 21: Server restart restores schedules and handles expired one-time jobs 

def test_server_restart_restores_active_schedules_and_skips_expired_once(tmp_path):
    """21. Server restart simulation correctly restores active schedules and disables expired one-time jobs."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")

    # 1. Add a daily schedule
    s_daily = ScheduledResearch(
        id="restart-daily-job",
        topic="Daily Topic",
        frequency="daily",
        hour=9,
        minute=0,
        enabled=True,
    )
    repo.save_schedule(s_daily)

    # 2. Add an expired one-time job (scheduled 1 hour ago)
    past_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s_expired = ScheduledResearch(
        id="restart-expired-once",
        topic="Expired Topic",
        frequency="once",
        hour=8,
        minute=0,
        enabled=True,
        run_at=past_iso,
    )
    repo.save_schedule(s_expired)

    # 3. Simulate new server process starting up
    fresh_service = SchedulerService(repository=repo)
    fresh_service.start()

    # Daily job must be registered and running
    daily_job = fresh_service._scheduler.get_job("restart-daily-job")
    assert daily_job is not None
    assert daily_job.next_run_time is not None

    # Expired one-time job must NOT be in scheduler and should be disabled in SQLite
    expired_job = fresh_service._scheduler.get_job("restart-expired-once")
    assert expired_job is None

    db_expired = repo.get_schedule("restart-expired-once")
    assert db_expired.enabled is False

    fresh_service.stop()


# ── Test 22: Scheduler event listeners log properly ───────────────────────────

def test_scheduler_event_listeners_logging(tmp_path, caplog):
    """22. Scheduler event listener logs job execution events."""
    import logging
    caplog.set_level(logging.INFO)

    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _make_report("Event Test Topic")

    service = SchedulerService(repository=repo, agent=agent_mock)
    service.start()

    # Add job that runs in 1 second
    sched = service.add_test_job(topic="Event Test Topic", delay_seconds=1)
    aps_job = service._scheduler.get_job(sched.id)
    assert aps_job is not None

    # Wait briefly for execution or invoke manually
    execute_scheduled_research_job(
        job_id=sched.id,
        topic=sched.topic,
        repository=repo,
        agent=agent_mock,
    )

    assert any(sched.id in record.message for record in caplog.records)
    service.stop()
