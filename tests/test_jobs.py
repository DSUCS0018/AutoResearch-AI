"""
tests/test_jobs.py
------------------
Comprehensive unit tests for Phase 1: Unified Job Lifecycle and Database Reliability.
Covers all 12 implementation test scenarios using mocks — no external API calls.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.memory.database import get_db_connection, init_db
from app.memory.models import (
    ResearchJobRecord,
    normalize_topic,
    to_display_time,
    utc_now_iso,
)
from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent
from app.research.report import ResearchReport, SourceReference
from app.scheduler.jobs import execute_scheduled_research_job
from app.scheduler.models import ScheduledResearch
from app.scheduler.scheduler import SchedulerService
from app.services.research_service import (
    DuplicateJobError,
    ResearchExecutionService,
)

client = TestClient(app)


def _sample_report(topic: str = "AI Testing") -> ResearchReport:
    return ResearchReport(
        topic=topic,
        executive_summary="Summary of research.",
        key_findings=["Key finding A [S1]", "Key finding B [S1]"],
        detailed_analysis="Detailed analysis section.",
        emerging_trends=["Trend 1 [S1]"],
        conclusion="Conclusion text.",
        sources=[SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/1")],
        generated_at=datetime.now(timezone.utc),
    )


# ── 1. Job creation and status transitions ────────────────────────────────────

def test_job_creation_and_status_transitions(tmp_path):
    """1. Job creation and status transitions: queued -> running -> completed."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    now_iso = utc_now_iso()

    job = ResearchJobRecord(
        id="job-test-transitions",
        topic="Neural Networks",
        trigger_type="manual",
        status="queued",
        created_at=now_iso,
        updated_at=now_iso,
    )
    repo.create_job(job)

    # Verify initial queued state
    fetched = repo.get_job("job-test-transitions")
    assert fetched is not None
    assert fetched.status == "queued"
    assert fetched.started_at is None
    assert fetched.completed_at is None

    # Transition to running
    started_at = utc_now_iso()
    repo.update_job("job-test-transitions", status="running", started_at=started_at)
    fetched_running = repo.get_job("job-test-transitions")
    assert fetched_running.status == "running"
    assert fetched_running.started_at == started_at

    # Transition to completed
    completed_at = utc_now_iso()
    repo.update_job(
        "job-test-transitions",
        status="completed",
        completed_at=completed_at,
        duration_seconds=1.25,
    )
    fetched_completed = repo.get_job("job-test-transitions")
    assert fetched_completed.status == "completed"
    assert fetched_completed.completed_at == completed_at
    assert fetched_completed.duration_seconds == 1.25


# ── 2. Successful manual research execution ───────────────────────────────────

def test_successful_manual_research_execution(tmp_path):
    """2. Successful manual research execution via ResearchExecutionService."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _sample_report("Autonomous Agents")

    service = ResearchExecutionService(repository=repo, agent=agent_mock)
    job = service.create_and_enqueue_job(topic="Autonomous Agents", trigger_type="manual")

    assert job.status == "queued"
    completed_job = service.execute_job(job.id)

    assert completed_job.status == "completed"
    assert completed_job.started_at is not None
    assert completed_job.completed_at is not None
    assert completed_job.duration_seconds is not None
    assert completed_job.report_path is not None
    assert completed_job.whats_new_path is not None
    assert completed_job.research_run_id is not None
    agent_mock.research.assert_called_once_with("Autonomous Agents")


# ── 3. Successful scheduled execution ─────────────────────────────────────────

def test_successful_scheduled_execution(tmp_path):
    """3. Successful scheduled execution records full job lifecycle in SQLite."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _sample_report("Daily Robotics")

    # Save a schedule in SQLite
    sched = ScheduledResearch(
        id="sched-robotics-daily",
        topic="Daily Robotics",
        frequency="daily",
        hour=9,
        minute=0,
        enabled=True,
    )
    repo.save_schedule(sched)

    # Execute scheduled job
    execute_scheduled_research_job(
        job_id="sched-robotics-daily",
        topic="Daily Robotics",
        repository=repo,
        agent=agent_mock,
    )

    # Verify persistent research job was created
    jobs = repo.list_jobs(topic="Daily Robotics")
    assert len(jobs) == 1
    sched_job = jobs[0]
    assert sched_job.trigger_type == "scheduled"
    assert sched_job.schedule_id == "sched-robotics-daily"
    assert sched_job.status == "completed"
    assert sched_job.duration_seconds is not None


# ── 4. Run Now execution ──────────────────────────────────────────────────────

def test_run_now_execution(tmp_path):
    """4. Run Now execution creates a persistent job record and executes immediately."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _sample_report("Immediate Run Topic")

    service = SchedulerService(repository=repo, agent=agent_mock)
    sched = service.add_schedule(topic="Immediate Run Topic", frequency="daily", time_str="10:00")

    service.run_job_now(sched.id)

    jobs = repo.list_jobs(topic="Immediate Run Topic")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == "completed"
    assert job.schedule_id == sched.id
    agent_mock.research.assert_called_once_with("Immediate Run Topic")


# ── 5. Research failure and persisted error details ───────────────────────────

def test_research_failure_and_persisted_error_details(tmp_path):
    """5. Research failure persists error details in both research_jobs and research_runs."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.side_effect = RuntimeError("Tavily search rate limit reached")

    service = ResearchExecutionService(repository=repo, agent=agent_mock)
    job = service.create_and_enqueue_job(topic="Failing Topic", trigger_type="manual")
    failed_job = service.execute_job(job.id)

    assert failed_job.status == "failed"
    assert "Tavily search rate limit reached" in failed_job.error
    assert failed_job.duration_seconds is not None

    # Check failure persisted in research_runs memory table
    runs = repo.list_research_runs()
    assert len(runs) >= 1
    assert any("Tavily search rate limit reached" in (r.get("error") or "") for r in runs)


# ── 6. Duplicate-run protection ───────────────────────────────────────────────

def test_duplicate_run_protection(tmp_path):
    """6. Duplicate-run protection prevents duplicate active jobs for the same topic."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = ResearchExecutionService(repository=repo)

    # First enqueue succeeds
    service.create_and_enqueue_job(topic="Quantum Mechanics", trigger_type="manual")

    # Second enqueue for same topic (case/whitespace variant) must raise DuplicateJobError
    with pytest.raises(DuplicateJobError, match="already active"):
        service.create_and_enqueue_job(topic="  quantum   mechanics  ", trigger_type="manual")


# ── 7. Concurrent execution protection ────────────────────────────────────────

def test_concurrent_execution_protection(tmp_path):
    """7. Concurrent execution requests cannot bypass duplicate-run protection."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    service = ResearchExecutionService(repository=repo)

    results = []
    errors = []

    def try_enqueue():
        try:
            job = service.create_and_enqueue_job(topic="Concurrent Topic", trigger_type="manual")
            results.append(job)
        except DuplicateJobError as e:
            errors.append(e)

    threads = [threading.Thread(target=try_enqueue) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread succeeds; the remaining 4 are rejected
    assert len(results) == 1
    assert len(errors) == 4


# ── 8. Report paths and research run ID association ───────────────────────────

def test_report_paths_and_research_run_id_association(tmp_path):
    """8. Report paths and research run ID are properly associated on completion."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    sample = _sample_report("Linked Report Topic")
    agent_mock.research.return_value = sample

    service = ResearchExecutionService(repository=repo, agent=agent_mock)
    job = service.run_job_sync(topic="Linked Report Topic", trigger_type="manual")

    assert job.status == "completed"
    assert job.research_run_id is not None
    assert job.research_run_id > 0
    assert job.report_path is not None
    assert "linked-report-topic.md" in job.report_path
    assert job.whats_new_path is not None
    assert "linked-report-topic-whats-new.md" in job.whats_new_path


# ── 9. Timestamp conversion and duration calculation ──────────────────────────

def test_timestamp_conversion_and_duration_calculation():
    """9. UTC ISO 8601 formatting, duration calculation, and Asia/Kolkata conversion."""
    now_utc = utc_now_iso()
    assert now_utc.endswith("+00:00") or "T" in now_utc

    # Display conversion to Asia/Kolkata
    ist_display = to_display_time("2026-10-12T00:00:00+00:00", tz_name="Asia/Kolkata")
    assert "IST" in ist_display
    assert "05:30:00" in ist_display  # UTC 00:00 = 05:30 IST

    # Duration calculation
    start_iso = "2026-10-12T10:00:00+00:00"
    end_iso = "2026-10-12T10:00:15.500000+00:00"
    job = ResearchJobRecord(
        id="job-duration-test",
        topic="Duration Topic",
        trigger_type="manual",
        status="completed",
        created_at=start_iso,
        updated_at=end_iso,
        started_at=start_iso,
        completed_at=end_iso,
    )
    assert job.calculate_duration() == 15.5


# ── 10. Database migration preserving existing records ─────────────────────────

def test_database_migration_preserving_existing_records(tmp_path):
    """10. SQLite migration preserves existing research runs and schedules without data loss."""
    db_file = tmp_path / "legacy.db"

    # Simulate legacy database schema before research_jobs existed
    conn = sqlite3.connect(str(db_file))
    with conn:
        conn.executescript("""
        CREATE TABLE research_runs (
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
        CREATE TABLE scheduled_research (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            frequency TEXT NOT NULL,
            day_of_week TEXT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        INSERT INTO research_runs (topic, topic_normalized, generated_at, report_path)
        VALUES ('Existing AI', 'existing ai', '2026-10-01T12:00:00Z', 'reports/existing-ai.md');
        INSERT INTO scheduled_research (id, topic, frequency, hour, minute, created_at)
        VALUES ('sched-existing', 'Existing AI', 'daily', 9, 0, '2026-10-01T12:00:00Z');
        """)
    conn.close()

    # Apply migration by calling init_db
    init_db(db_file)

    # Verify existing data is preserved
    repo = ResearchRepository(db_path=db_file)
    existing_run = repo.get_latest_research("Existing AI")
    assert existing_run is not None
    assert existing_run.topic == "Existing AI"

    existing_sched = repo.get_schedule("sched-existing")
    assert existing_sched is not None
    assert existing_sched.topic == "Existing AI"

    # Verify research_jobs table was created and works
    new_job = ResearchJobRecord(
        id="job-migrated-test",
        topic="Migrated Topic",
        trigger_type="manual",
        status="queued",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    repo.create_job(new_job)
    assert repo.get_job("job-migrated-test") is not None


# ── 11. Recurring schedule preservation and one-time job completion ───────────

def test_recurring_schedule_preservation_and_one_time_job_completion(tmp_path):
    """11. Recurring schedules remain enabled, whereas one-time schedules are deactivated."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    agent_mock = MagicMock(spec=ResearchAgent)
    agent_mock.research.return_value = _sample_report("Recurring Test")

    service = ResearchExecutionService(repository=repo, agent=agent_mock)

    # 1. Recurring daily schedule
    sched_daily = ScheduledResearch(
        id="sched-daily-preservation",
        topic="Daily Topic",
        frequency="daily",
        hour=9,
        minute=0,
        enabled=True,
    )
    repo.save_schedule(sched_daily)

    job_daily = service.create_and_enqueue_job(
        topic="Daily Topic", trigger_type="scheduled", schedule_id="sched-daily-preservation"
    )
    service.execute_job(job_daily.id)

    # Daily schedule must remain enabled
    assert repo.get_schedule("sched-daily-preservation").enabled is True

    # 2. One-time schedule
    sched_once = ScheduledResearch(
        id="sched-once-deactivation",
        topic="Once Topic",
        frequency="once",
        hour=10,
        minute=0,
        enabled=True,
    )
    repo.save_schedule(sched_once)

    job_once = service.create_and_enqueue_job(
        topic="Once Topic", trigger_type="scheduled", schedule_id="sched-once-deactivation"
    )
    service.execute_job(job_once.id)

    # One-time schedule must be marked disabled
    assert repo.get_schedule("sched-once-deactivation").enabled is False


# ── 12. API responses for queued, running, completed, and failed jobs ─────────

@patch("app.api.routes.jobs.get_research_execution_service")
@patch("app.api.routes.jobs.ResearchRepository")
def test_api_responses_for_queued_running_completed_failed_jobs(mock_repo_cls, mock_get_svc):
    """12. API endpoints return correct schemas for queued, running, completed, and failed jobs."""
    mock_svc = MagicMock()
    mock_get_svc.return_value = mock_svc

    mock_repo = MagicMock()
    mock_repo_cls.return_value = mock_repo

    now = utc_now_iso()

    # 1. POST /api/jobs -> 202 Accepted with queued job
    mock_job_queued = ResearchJobRecord(
        id="job-manual-api-test",
        topic="API Test Topic",
        trigger_type="manual",
        status="queued",
        created_at=now,
        updated_at=now,
    )
    mock_svc.create_and_enqueue_job.return_value = mock_job_queued

    with patch("app.api.routes.jobs._background_execute"):
        res_post = client.post("/api/jobs", json={"topic": "API Test Topic"})
    assert res_post.status_code == 202
    data_post = res_post.json()
    assert data_post["id"] == "job-manual-api-test"
    assert data_post["status"] == "queued"

    # 2. GET /api/jobs/{id} when running
    mock_job_running = ResearchJobRecord(
        id="job-manual-api-test",
        topic="API Test Topic",
        trigger_type="manual",
        status="running",
        created_at=now,
        started_at=now,
        updated_at=now,
    )
    mock_repo.get_job.return_value = mock_job_running

    res_get = client.get("/api/jobs/job-manual-api-test")
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "running"

    # 3. GET /api/jobs/{id}/status when completed
    mock_job_completed = ResearchJobRecord(
        id="job-manual-api-test",
        topic="API Test Topic",
        trigger_type="manual",
        status="completed",
        created_at=now,
        started_at=now,
        completed_at=now,
        updated_at=now,
        duration_seconds=5.2,
    )
    mock_repo.get_job.return_value = mock_job_completed

    res_status = client.get("/api/jobs/job-manual-api-test/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "completed"
    assert res_status.json()["duration_seconds"] == 5.2

    # 4. GET /api/jobs/{id} when failed
    mock_job_failed = ResearchJobRecord(
        id="job-manual-api-test",
        topic="API Test Topic",
        trigger_type="manual",
        status="failed",
        created_at=now,
        started_at=now,
        completed_at=now,
        updated_at=now,
        error="LLM quota exceeded",
        duration_seconds=2.1,
    )
    mock_repo.get_job.return_value = mock_job_failed

    res_failed = client.get("/api/jobs/job-manual-api-test")
    assert res_failed.status_code == 200
    assert res_failed.json()["status"] == "failed"
    assert "LLM quota exceeded" in res_failed.json()["error"]
