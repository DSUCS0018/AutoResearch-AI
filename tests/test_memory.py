"""
tests/test_memory.py
--------------------
Unit tests for Stage 6: Persistent Research Memory (SQLite) and ResearchComparator.

All tests use a temporary SQLite database — real data/research.db is never touched.
"""

import sqlite3
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from app.memory.database import get_db_connection, init_db
from app.memory.models import normalize_topic
from app.memory.repository import ResearchRepository
from app.memory.comparator import ResearchComparator, WhatsNewReport
from app.research.report import ResearchReport, SourceReference
from app.research.agent import ResearchAgent, ResearchAgentError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample_report(
    topic="AI Trends",
    summary="AI summary text.",
    findings=None,
    trends=None,
    sources=None,
    generated_at=None,
) -> ResearchReport:
    return ResearchReport(
        topic=topic,
        executive_summary=summary,
        key_findings=findings or ["Key finding 1 [S1]", "Key finding 2 [S2]"],
        detailed_analysis="Detailed analysis text.",
        emerging_trends=trends or ["Emerging trend 1 [S1]"],
        conclusion="Conclusion text.",
        sources=sources or [
            SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/s1"),
            SourceReference(identifier="[S2]", title="Source 2", url="https://example.com/s2"),
        ],
        generated_at=generated_at or datetime.now(timezone.utc),
    )


# ── Test 1: Database initializes correctly ────────────────────────────────────

def test_database_initialization(tmp_path):
    """1. Database initializes correctly."""
    db_file = tmp_path / "test.db"
    assert not db_file.exists()
    init_db(db_file)
    assert db_file.exists()


# ── Test 2: Tables are created ────────────────────────────────────────────────

def test_tables_created(tmp_path):
    """2. Tables are created correctly."""
    db_file = tmp_path / "test.db"
    init_db(db_file)
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    conn.close()

    assert "research_runs" in tables
    assert "research_sources" in tables
    assert "research_findings" in tables


# ── Test 3: Research run can be saved ─────────────────────────────────────────

def test_save_research_run(tmp_path):
    """3. Research run can be saved."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    report = _make_sample_report()
    run_id = repo.save_research_run(report, report_path="reports/test.md")
    assert isinstance(run_id, int)
    assert run_id > 0


# ── Test 4: Research sources are saved ────────────────────────────────────────

def test_save_research_sources(tmp_path):
    """4. Research sources are saved."""
    db_file = tmp_path / "test.db"
    repo = ResearchRepository(db_path=db_file)
    report = _make_sample_report()
    run_id = repo.save_research_run(report, report_path="reports/test.md")

    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM research_sources WHERE research_run_id = ?", (run_id,))
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2
    urls = {r["url"] for r in rows}
    assert "https://example.com/s1" in urls
    assert "https://example.com/s2" in urls


# ── Test 5: Research run can be retrieved ─────────────────────────────────────

def test_get_research_run_by_id(tmp_path):
    """5. Research run can be retrieved by ID."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    report = _make_sample_report(topic="Quantum AI")
    run_id = repo.save_research_run(report, report_path="reports/quantum.md")

    retrieved = repo.get_research_run(run_id)
    assert retrieved is not None
    assert retrieved.topic == "Quantum AI"
    assert len(retrieved.sources) == 2
    assert len(retrieved.key_findings) == 2


# ── Test 6: Latest research for a topic is returned correctly ────────────────

def test_get_latest_research(tmp_path):
    """6. Latest research for a topic is returned correctly."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    report1 = _make_sample_report(topic="AI", summary="Old summary")
    report2 = _make_sample_report(topic="AI", summary="New summary")

    repo.save_research_run(report1)
    repo.save_research_run(report2)

    latest = repo.get_latest_research("AI")
    assert latest is not None
    assert latest.executive_summary == "New summary"


# ── Test 7: Topic matching is case/whitespace normalized ─────────────────────

def test_topic_normalization(tmp_path):
    """7. Topic matching is case/whitespace normalized."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    report = _make_sample_report(topic="  Artificial  Intelligence Trends  ")
    repo.save_research_run(report)

    # Queries with different casing and whitespace should match
    retrieved1 = repo.get_latest_research("artificial intelligence trends")
    retrieved2 = repo.get_latest_research("ARTIFICIAL   INTELLIGENCE   TRENDS")

    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.topic == report.topic
    assert retrieved2.topic == report.topic


# ── Test 8: Unrelated topics are not returned ─────────────────────────────────

def test_unrelated_topics_not_returned(tmp_path):
    """8. Unrelated topics are not returned."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    report = _make_sample_report(topic="Quantum Computing")
    repo.save_research_run(report)

    unrelated = repo.get_latest_research("Space Exploration")
    assert unrelated is None


# ── Test 9: Research history returns multiple runs in correct order ──────────

def test_research_history_order(tmp_path):
    """9. Research history returns multiple runs in correct order (newest first)."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    r1 = _make_sample_report(topic="Robotics", summary="Run 1")
    r2 = _make_sample_report(topic="Robotics", summary="Run 2")
    r3 = _make_sample_report(topic="Robotics", summary="Run 3")

    repo.save_research_run(r1)
    repo.save_research_run(r2)
    repo.save_research_run(r3)

    history = repo.get_research_history("Robotics")
    assert len(history) == 3
    assert history[0].executive_summary == "Run 3"
    assert history[1].executive_summary == "Run 2"
    assert history[2].executive_summary == "Run 1"


# ── Test 10: Duplicate source URLs are handled correctly ─────────────────────

def test_duplicate_source_urls_in_memory(tmp_path):
    """10. Duplicate source URLs are handled correctly."""
    repo = ResearchRepository(db_path=tmp_path / "test.db")
    sources = [
        SourceReference(identifier="[S1]", title="Source A", url="https://example.com/same"),
        SourceReference(identifier="[S2]", title="Source B", url="https://example.com/other"),
    ]
    report = _make_sample_report(sources=sources)
    run_id = repo.save_research_run(report)

    retrieved = repo.get_research_run(run_id)
    assert len(retrieved.sources) == 2


# ── Test 11: New findings are detected ───────────────────────────────────────

def test_comparator_new_findings_detected():
    """11. New findings are detected."""
    prev = _make_sample_report(findings=["Old finding [S1]"])
    curr = _make_sample_report(findings=["Old finding [S1]", "Brand new finding [S2]"])

    comparator = ResearchComparator()
    whats_new = comparator.compare(curr, prev)

    assert "Brand new finding [S2]" in whats_new.new_findings
    assert len(whats_new.new_findings) == 1


# ── Test 12: Existing findings are recognized as unchanged ────────────────────

def test_comparator_existing_findings_unchanged():
    """12. Existing findings are recognized as unchanged."""
    prev = _make_sample_report(findings=["Same finding [S1]"])
    curr = _make_sample_report(findings=["Same finding [S1]"])

    comparator = ResearchComparator()
    whats_new = comparator.compare(curr, prev)

    assert "Same finding [S1]" in whats_new.unchanged_findings
    assert len(whats_new.new_findings) == 0


# ── Test 13: New sources are detected ─────────────────────────────────────────

def test_comparator_new_sources_detected():
    """13. New sources are detected by canonical URL."""
    s1 = SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/1")
    s2 = SourceReference(identifier="[S2]", title="Source 2", url="https://example.com/2")

    prev = _make_sample_report(sources=[s1])
    curr = _make_sample_report(sources=[s1, s2])

    comparator = ResearchComparator()
    whats_new = comparator.compare(curr, prev)

    new_urls = [s.url for s in whats_new.new_sources]
    assert "https://example.com/2" in new_urls
    assert "https://example.com/1" not in new_urls


# ── Test 14: Existing sources are not reported as new ─────────────────────────

def test_comparator_existing_sources_not_new():
    """14. Existing sources are not reported as new."""
    s1 = SourceReference(identifier="[S1]", title="Source 1", url="https://example.com/1")

    prev = _make_sample_report(sources=[s1])
    curr = _make_sample_report(sources=[s1])

    comparator = ResearchComparator()
    whats_new = comparator.compare(curr, prev)

    assert len(whats_new.new_sources) == 0


# ── Test 15: First-time topic correctly reports no previous research ──────────

def test_comparator_first_time_topic():
    """15. First-time topic correctly reports no previous research."""
    curr = _make_sample_report(findings=["Baseline finding [S1]"])

    comparator = ResearchComparator()
    whats_new = comparator.compare(curr, previous=None)

    assert whats_new.has_previous_research is False
    assert len(whats_new.new_findings) == 2   # 1 key + 1 emerging
    assert "No previous research run exists" in whats_new.to_markdown()


# ── Test 16: What's New report is formatted correctly ────────────────────────

def test_whats_new_report_formatting():
    """16. What's New report is formatted correctly into Markdown."""
    s1 = SourceReference(identifier="[S1]", title="New Source", url="https://new.com")
    report = WhatsNewReport(
        topic="AI Research",
        new_findings=["New finding text [S1]"],
        changed_findings=[],
        new_sources=[s1],
        unchanged_findings=["Old finding text [S1]"],
        has_previous_research=True,
    )
    md = report.to_markdown()

    assert "# What's New: AI Research" in md
    assert "## New Findings" in md
    assert "- New finding text [S1]" in md
    assert "## New Sources" in md
    assert "[S1] New Source" in md
    assert "https://new.com" in md
    assert "## Already Known" in md
    assert "- Old finding text [S1]" in md


# ── Test 17: Failed research is not saved as completed history ────────────────

def test_failed_research_not_saved(tmp_path):
    """17. Failed research is not saved as completed history."""
    db_file = tmp_path / "test.db"
    repo = ResearchRepository(db_path=db_file)

    search_mock = MagicMock()
    search_mock.search.side_effect = ResearchAgentError("Search failed")

    agent = ResearchAgent(
        search_client=search_mock,
        collector=MagicMock(),
        rag_service=MagicMock(),
        llm_client=MagicMock(),
        repository=repo,
    )

    with pytest.raises(ResearchAgentError):
        agent.research("Failing Topic")

    # DB should have 0 runs
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM research_runs")
    count = cursor.fetchone()["cnt"]
    conn.close()

    assert count == 0
