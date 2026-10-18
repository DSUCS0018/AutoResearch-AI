"""
tests/test_api.py
------------------
Unit tests for Stage 8: FastAPI backend endpoints using TestClient.

All tests mock underlying services — no real Tavily, Gemini, or network calls are made.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.research.report import ResearchReport, SourceReference
from app.scheduler.models import ScheduledResearch

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample_report(topic="AI Trends"):
    return ResearchReport(
        topic=topic,
        executive_summary="Summary text.",
        key_findings=["Key finding 1 [S1]"],
        detailed_analysis="Detailed analysis.",
        emerging_trends=["Emerging trend 1 [S1]"],
        conclusion="Conclusion text.",
        sources=[SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/1")],
        generated_at=datetime.now(timezone.utc),
    )


# ── Test 1: GET /health ───────────────────────────────────────────────────────

def test_health_check_endpoint():
    """1. GET /health returns 200 OK and service status."""
    res = client.get("/health")
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["status"] == "ok"
    assert "AutoResearch AI" in data["service"]


# ── Test 2: POST /api/research ────────────────────────────────────────────────

def test_post_research_endpoint(tmp_path):
    """2. POST /api/research calls ResearchAgent and returns report."""
    from app.services.research_service import ResearchExecutionService
    from app.memory.repository import ResearchRepository

    mock_agent = MagicMock()
    sample_report = _make_sample_report("AI Advances")
    mock_agent.research.return_value = sample_report

    # Patch at the route level: inject a service that wraps our mocked agent
    with patch("app.api.routes.research.get_research_execution_service") as mock_svc_factory:
        svc = ResearchExecutionService(
            repository=ResearchRepository(db_path=tmp_path / "test.sqlite"),
            agent=mock_agent,
        )
        mock_svc_factory.return_value = svc

        res = client.post("/api/research", json={"topic": "AI Advances"})

    assert res.status_code == 200, res.json()
    data = res.json()

    assert data["topic"] == "AI Advances"
    assert "executive_summary" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["url"] == "https://example.com/1"
    mock_agent.research.assert_called_once_with("AI Advances")


# ── Test 3: Empty research topic rejected ─────────────────────────────────────

def test_empty_research_topic_rejected():
    """3. Empty research topic is rejected with 400 Bad Request."""
    res = client.post("/api/research", json={"topic": "   "})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


# ── Test 4: GET /api/history/{topic} ──────────────────────────────────────────

@patch("app.api.routes.history.ResearchRepository")
def test_get_history_endpoint(mock_repo_cls):
    """4. GET /api/history/{topic} returns previous research runs."""
    mock_repo = MagicMock()
    mock_repo.get_research_history.return_value = [_make_sample_report("Quantum AI")]
    mock_repo_cls.return_value = mock_repo

    res = client.get("/api/history/Quantum%20AI")
    assert res.status_code == 200, res.json()
    data = res.json()

    assert data["topic"] == "Quantum AI"
    assert len(data["runs"]) == 1
    assert data["runs"][0]["sources_count"] == 1


# ── Test 5: POST /api/whats-new ───────────────────────────────────────────────

@patch("app.api.routes.whats_new.ResearchRepository")
@patch("app.api.routes.whats_new.ResearchAgent")
@patch("app.api.routes.whats_new.ResearchComparator")
def test_post_whats_new_endpoint(mock_comp_cls, mock_agent_cls, mock_repo_cls):
    """5. POST /api/whats-new captures baseline and generates comparison report."""
    mock_repo = MagicMock()
    mock_repo.get_latest_research.return_value = _make_sample_report("AI Trends")
    mock_repo_cls.return_value = mock_repo

    mock_agent = MagicMock()
    mock_agent.research.return_value = _make_sample_report("AI Trends")
    mock_agent_cls.return_value = mock_agent

    mock_comp = MagicMock()
    mock_wn = MagicMock()
    mock_wn.topic = "AI Trends"
    mock_wn.has_previous_research = True
    mock_wn.new_findings = ["New finding"]
    mock_wn.changed_findings = []
    mock_wn.new_sources = []
    mock_wn.unchanged_findings = []
    mock_wn.generated_at = datetime.now(timezone.utc)
    mock_wn.to_markdown.return_value = "# What's New"
    mock_comp.compare.return_value = mock_wn
    mock_comp_cls.return_value = mock_comp

    res = client.post("/api/whats-new", json={"topic": "AI Trends"})
    assert res.status_code == 200, res.json()
    data = res.json()

    assert data["topic"] == "AI Trends"
    assert data["has_previous_research"] is True
    assert "new_findings" in data


# ── Test 6: GET /api/schedules ────────────────────────────────────────────────

# ── Test 6: GET /api/schedules ────────────────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_get_schedules_endpoint(mock_get_service):
    """6. GET /api/schedules returns list of schedules."""
    mock_service = MagicMock()
    mock_service.list_schedules.return_value = [
        {
            "id": "research-ai-daily",
            "topic": "AI",
            "frequency": "daily",
            "day_of_week": None,
            "hour": 9,
            "minute": 0,
            "enabled": True,
            "next_run": "2026-10-09 09:00:00 IST",
            "created_at": "2026-10-08T00:00:00Z",
        }
    ]
    mock_get_service.return_value = mock_service

    res = client.get("/api/schedules")
    assert res.status_code == 200, res.json()
    data = res.json()
    assert len(data) == 1
    assert data[0]["id"] == "research-ai-daily"


# ── Test 7: POST /api/schedules ───────────────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_post_schedules_endpoint(mock_get_service):
    """7. POST /api/schedules creates a new schedule."""
    mock_service = MagicMock()
    mock_sched = ScheduledResearch(
        id="research-ai-daily",
        topic="AI",
        frequency="daily",
        hour=9,
        minute=0,
        enabled=True,
    )
    mock_service.add_schedule.return_value = mock_sched
    mock_service._scheduler.get_job.return_value = None
    mock_get_service.return_value = mock_service

    res = client.post("/api/schedules", json={"topic": "AI", "frequency": "daily", "time": "09:00"})
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == "research-ai-daily"


# ── Test 8: PATCH /api/schedules/{job_id} ──────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_patch_schedule_endpoint(mock_get_service):
    """8. PATCH /api/schedules/{job_id} updates schedule state."""
    mock_service = MagicMock()
    mock_sched = ScheduledResearch(
        id="research-ai-daily",
        topic="AI",
        frequency="daily",
        hour=9,
        minute=0,
        enabled=False,
    )
    mock_service.repository.get_schedule.return_value = mock_sched
    mock_service._scheduler.get_job.return_value = None
    mock_get_service.return_value = mock_service

    res = client.patch("/api/schedules/research-ai-daily", json={"enabled": False})
    assert res.status_code == 200, res.json()
    assert res.json()["enabled"] is False


# ── Test 9: DELETE /api/schedules/{job_id} ─────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_delete_schedule_endpoint(mock_get_service):
    """9. DELETE /api/schedules/{job_id} removes a schedule."""
    mock_service = MagicMock()
    mock_service.remove_schedule.return_value = True
    mock_get_service.return_value = mock_service

    res = client.delete("/api/schedules/research-ai-daily")
    assert res.status_code == 200, res.json()
    assert res.json()["status"] == "ok"


# ── Test 10: POST /api/schedules/{job_id}/run ─────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_post_schedule_run_endpoint(mock_get_service):
    """10. POST /api/schedules/{job_id}/run triggers manual job execution."""
    mock_service = MagicMock()
    mock_get_service.return_value = mock_service

    res = client.post("/api/schedules/research-ai-daily/run")
    assert res.status_code == 200, res.json()
    assert res.json()["status"] == "ok"
    mock_service.run_job_now.assert_called_once_with("research-ai-daily")


# ── Test 11: Enable schedule ──────────────────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_enable_schedule_endpoint(mock_get_service):
    """11. POST /api/schedules/{job_id}/enable enables schedule."""
    mock_service = MagicMock()
    mock_service.enable_schedule.return_value = True
    mock_get_service.return_value = mock_service

    res = client.post("/api/schedules/research-ai-daily/enable")
    assert res.status_code == 200, res.json()
    assert res.json()["status"] == "ok"


# ── Test 12: Disable schedule ─────────────────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_disable_schedule_endpoint(mock_get_service):
    """12. POST /api/schedules/{job_id}/disable disables schedule."""
    mock_service = MagicMock()
    mock_service.disable_schedule.return_value = True
    mock_get_service.return_value = mock_service

    res = client.post("/api/schedules/research-ai-daily/disable")
    assert res.status_code == 200, res.json()
    assert res.json()["status"] == "ok"


# ── Test 13: API errors handled correctly ──────────────────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_api_errors_handled_correctly(mock_get_service):
    """13. API errors (e.g., 404 for non-existent schedule) are handled with JSON detail."""
    mock_service = MagicMock()
    mock_service.remove_schedule.return_value = False
    mock_get_service.return_value = mock_service

    res = client.delete("/api/schedules/non-existent-job")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# ── Test 14: FastAPI Lifespan & Scheduler Lifecycle Integration ───────────────

def test_api_startup_starts_scheduler_and_registers_jobs(tmp_path):
    """
    Verify:
    1. Scheduler starts during API application startup (lifespan).
    2. An enabled persisted schedule in SQLite is registered automatically.
    3. Registered job has a non-null next run time.
    4. Scheduler remains running while API is alive.
    5. Shutdown gracefully stops scheduler.
    """
    from app.memory.repository import ResearchRepository
    from app.scheduler.scheduler import SchedulerService, set_scheduler_service

    test_db = tmp_path / "lifespan_test.db"
    repo = ResearchRepository(db_path=test_db)

    # Create an enabled schedule in SQLite prior to API startup
    test_service = SchedulerService(repository=repo)
    sched = test_service.add_schedule(
        topic="Automated AI Research",
        frequency="daily",
        time_str="09:30",
    )
    assert sched.enabled is True

    # Register this test service as global scheduler service
    set_scheduler_service(test_service)

    try:
        # Before API startup, scheduler is not running
        assert test_service._scheduler.running is False

        # Start API with lifespan context manager
        with TestClient(app) as test_client:
            # 1. Scheduler starts during API application startup
            assert test_service._scheduler.running is True

            # 2. Enabled persisted schedule is registered
            aps_job = test_service._scheduler.get_job(sched.id)
            assert aps_job is not None

            # 3. Registered job has a non-null next_run_time
            assert aps_job.next_run_time is not None

            # 4. Scheduler remains running while API is alive
            res = test_client.get("/api/schedules")
            assert res.status_code == 200, res.json()
            schedules_list = res.json()
            assert len(schedules_list) == 1
            assert schedules_list[0]["id"] == sched.id
            assert schedules_list[0]["next_run"] != "Pending Start"
            assert schedules_list[0]["next_run"] != "N/A"
            assert test_service._scheduler.running is True

        # 5. Shutdown gracefully stops scheduler
        assert test_service._scheduler.running is False

    finally:
        set_scheduler_service(None)


# ── Test 15: POST /api/schedules/test-job creates one-time test job ───────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_post_test_job_endpoint(mock_get_service):
    """15. POST /api/schedules/test-job registers a safe 60s test job."""
    mock_service = MagicMock()
    mock_sched = ScheduledResearch(
        id="test-job-sample-12345",
        topic="Sample Test",
        frequency="once",
        hour=10,
        minute=15,
        enabled=True,
        run_at="2026-10-12T10:15:00+05:30",
    )
    mock_service.add_test_job.return_value = mock_sched
    mock_service._scheduler.get_job.return_value = MagicMock(
        next_run_time=datetime(2026, 10, 12, 10, 15, 0)
    )
    mock_get_service.return_value = mock_service

    res = client.post("/api/schedules/test-job", json={"topic": "Sample Test", "delay_seconds": 60})
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == "test-job-sample-12345"
    assert data["frequency"] == "once"
    assert data["enabled"] is True
    mock_service.add_test_job.assert_called_once_with(topic="Sample Test", delay_seconds=60)


# ── Test 16: GET /api/schedules/status returns engine status & timezone ───────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_get_scheduler_status_endpoint(mock_get_service):
    """16. GET /api/schedules/status returns engine running state and active timezone."""
    from zoneinfo import ZoneInfo
    mock_service = MagicMock()
    mock_service._scheduler.running = True
    mock_service._scheduler.get_jobs.return_value = [MagicMock(), MagicMock()]
    mock_service.timezone_name = "Asia/Kolkata"
    mock_service.tz = ZoneInfo("Asia/Kolkata")
    mock_get_service.return_value = mock_service

    res = client.get("/api/schedules/status")
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["running"] is True
    assert data["timezone"] == "Asia/Kolkata"
    assert data["total_jobs"] == 2
    assert "current_time" in data


# ── Test 17: POST /api/schedules with frequency=once ──────────────────────────

@patch("app.api.routes.schedules.get_scheduler_service")
def test_post_schedules_once_frequency(mock_get_service):
    """17. POST /api/schedules with frequency='once' routes to test job creation."""
    mock_service = MagicMock()
    mock_sched = ScheduledResearch(
        id="test-job-ai-once",
        topic="One Time AI",
        frequency="once",
        hour=9,
        minute=1,
        enabled=True,
    )
    mock_service.add_test_job.return_value = mock_sched
    mock_service._scheduler.get_job.return_value = None
    mock_get_service.return_value = mock_service

    res = client.post(
        "/api/schedules",
        json={"topic": "One Time AI", "frequency": "once", "delay_seconds": 60},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["frequency"] == "once"
    mock_service.add_test_job.assert_called_once_with(topic="One Time AI", delay_seconds=60)

