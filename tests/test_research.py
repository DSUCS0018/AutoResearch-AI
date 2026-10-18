"""
tests/test_research.py
----------------------
Unit tests for Stage 5: ResearchAgent and report generation.

All tests use mocking — no real Tavily, Gemini, or network calls are made.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from app.search.models import Source, Document
from app.rag.vector_store import SearchResult
from app.research.report import ResearchReport, SourceReference
from app.research.agent import ResearchAgent, ResearchAgentError, ResearchReportError, _safe_filename
from app.research.prompts import build_research_prompt, assign_source_identifiers


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_source(title="Test Source", url="https://example.com", score=0.9):
    return Source(title=title, url=url, snippet="snippet", score=score)


def _make_document(text="Sample text about AI.", url="https://example.com", title="Test"):
    return Document(url=url, title=title, text=text, fetched_at=datetime.now(timezone.utc))


def _make_search_result(text="RAG content", url="https://example.com/rag", title="RAG Article"):
    result = MagicMock(spec=SearchResult)
    result.text = text
    result.metadata = {"source_url": url, "source_title": title}
    result.distance = 0.15
    return result


def _make_valid_gemini_json(
    summary="Summary text [S1].",
    findings=None,
    analysis="Detailed analysis [S1][S2].",
    trends=None,
    conclusion="Conclusion [S1].",
):
    return json.dumps({
        "executive_summary": summary,
        "key_findings": findings or ["Finding one [S1]", "Finding two [S2]"],
        "detailed_analysis": analysis,
        "emerging_trends": trends or ["Trend one [S1]"],
        "conclusion": conclusion,
    })


def _make_agent(
    sources=None, documents=None, retrieved=None, gemini_response=None
) -> ResearchAgent:
    """Build a ResearchAgent with all components mocked."""
    search_mock = MagicMock()
    search_mock.search.return_value = sources or [_make_source()]

    collector_mock = MagicMock()
    collector_mock.collect.return_value = documents or _make_document()

    rag_mock = MagicMock()
    rag_mock.ingest_documents.return_value = {"documents": 1, "chunks": 1, "stored": 1}
    rag_mock.retrieve.return_value = retrieved or [_make_search_result()]

    llm_mock = MagicMock()
    llm_mock.generate.return_value = gemini_response or _make_valid_gemini_json()

    return ResearchAgent(
        search_client=search_mock,
        collector=collector_mock,
        rag_service=rag_mock,
        llm_client=llm_mock,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Input validation
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 1: Empty topic is rejected ──────────────────────────────────────────

def test_empty_topic_raises_agent_error():
    """research() must raise ResearchAgentError for blank topics."""
    agent = _make_agent()
    with pytest.raises(ResearchAgentError, match="empty"):
        agent.research("   ")


def test_none_topic_raises_agent_error():
    """research() must raise ResearchAgentError for None-like topics."""
    agent = _make_agent()
    with pytest.raises(ResearchAgentError, match="empty"):
        agent.research("")


# ── Test 2: Tavily search is called with the topic ───────────────────────────

def test_search_is_called_with_topic(tmp_path, monkeypatch):
    """Tavily search must be called once with the exact topic string."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    agent.research("artificial intelligence trends")

    agent._search.search.assert_called_once_with("artificial intelligence trends")


# ── Test 3: Failed collection doesn't crash the run ──────────────────────────

def test_failed_collection_is_skipped(tmp_path, monkeypatch):
    """A CollectorError on one URL must be skipped; others must proceed."""
    from app.search.collector import CollectorError

    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    sources = [
        _make_source(url="https://good.com"),
        _make_source(url="https://bad.com"),
    ]

    collector_mock = MagicMock()
    collector_mock.collect.side_effect = [
        _make_document(url="https://good.com"),   # first succeeds
        CollectorError("timeout"),                 # second fails
    ]

    rag_mock = MagicMock()
    rag_mock.ingest_documents.return_value = {"documents": 1, "chunks": 1, "stored": 1}
    rag_mock.retrieve.return_value = [_make_search_result()]

    llm_mock = MagicMock()
    llm_mock.generate.return_value = _make_valid_gemini_json()

    agent = ResearchAgent(
        search_client=MagicMock(search=MagicMock(return_value=sources)),
        collector=collector_mock,
        rag_service=rag_mock,
        llm_client=llm_mock,
    )

    # Should succeed despite one failed URL
    report = agent.research("test topic")
    assert isinstance(report, ResearchReport)


# ── Test 4: Collected documents are ingested ─────────────────────────────────

def test_documents_are_ingested(tmp_path, monkeypatch):
    """ingest_documents must be called with the collected documents."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    agent.research("AI research topic")

    agent._rag.ingest_documents.assert_called_once()
    args = agent._rag.ingest_documents.call_args[0][0]
    assert len(args) >= 1
    assert isinstance(args[0], Document)


# ── Test 5: RAG retrieval is called ──────────────────────────────────────────

def test_rag_retrieval_is_called(tmp_path, monkeypatch):
    """retrieve() must be called with the research topic."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    agent.research("quantum computing")

    agent._rag.retrieve.assert_called_once()
    assert "quantum computing" in agent._rag.retrieve.call_args[0][0]


# ── Test 6: Gemini receives a grounded prompt ─────────────────────────────────

def test_gemini_receives_grounded_prompt(tmp_path, monkeypatch):
    """Gemini's generate() must be called once with a non-empty prompt."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    agent.research("machine learning")

    agent._llm.generate.assert_called_once()
    prompt_arg = agent._llm.generate.call_args[0][0]
    assert len(prompt_arg) > 100
    assert "machine learning" in prompt_arg.lower()


# ── Test 7: Source identifiers [S1], [S2] appear in the prompt ───────────────

def test_source_identifiers_in_prompt():
    """The generated prompt must include [S1] style identifiers."""
    results = [
        _make_search_result(url="https://a.com", title="Source A"),
        _make_search_result(url="https://b.com", title="Source B"),
    ]
    prompt = build_research_prompt("AI trends", results)
    assert "[S1]" in prompt
    assert "[S2]" in prompt


# ── Test 8: Valid Gemini JSON is parsed correctly ─────────────────────────────

def test_valid_gemini_json_is_parsed():
    """_parse_gemini_response must return a dict from valid JSON."""
    raw = _make_valid_gemini_json()
    result = ResearchAgent._parse_gemini_response(raw)
    assert isinstance(result, dict)
    assert "executive_summary" in result
    assert "key_findings" in result
    assert isinstance(result["key_findings"], list)


# ── Test 9: Invalid JSON raises ResearchReportError ──────────────────────────

def test_invalid_json_raises_report_error():
    """_parse_gemini_response must raise ResearchReportError for bad JSON."""
    bad_response = "This is not JSON at all, just free text from the model."
    with pytest.raises(ResearchReportError, match="not valid JSON"):
        ResearchAgent._parse_gemini_response(bad_response)


# ── Test 10: Markdown fences are stripped before parsing ─────────────────────

def test_markdown_fences_are_stripped():
    """JSON wrapped in ```json ... ``` should still parse correctly."""
    raw = "```json\n" + _make_valid_gemini_json() + "\n```"
    result = ResearchAgent._parse_gemini_response(raw)
    assert isinstance(result, dict)
    assert "executive_summary" in result


# ── Test 11: ResearchReport is created correctly ──────────────────────────────

def test_research_report_is_created(tmp_path, monkeypatch):
    """research() must return a fully populated ResearchReport."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    report = agent.research("large language models")

    assert isinstance(report, ResearchReport)
    assert report.topic == "large language models"
    assert report.executive_summary != ""
    assert isinstance(report.key_findings, list)
    assert isinstance(report.sources, list)
    assert len(report.sources) >= 1


# ── Test 12: Source URLs come from trusted metadata ───────────────────────────

def test_source_urls_from_trusted_metadata(tmp_path, monkeypatch):
    """Source URLs in the report must come from RAG metadata, not Gemini output."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    trusted_url = "https://trusted-source.com/article"
    retrieved = [_make_search_result(url=trusted_url, title="Trusted Article")]

    agent = _make_agent(retrieved=retrieved)
    report = agent.research("AI safety")

    urls_in_report = [src.url for src in report.sources]
    assert trusted_url in urls_in_report
    # No invented URLs
    for url in urls_in_report:
        assert "trusted-source.com" in url or url == ""


# ── Test 13: Markdown report is saved correctly ───────────────────────────────

def test_markdown_report_is_saved(tmp_path, monkeypatch):
    """research() must save a .md file in the reports/ directory."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    agent = _make_agent()
    agent.research("neural networks overview")

    reports = list((tmp_path / "reports").glob("*.md"))
    assert len(reports) == 1
    content = reports[0].read_text(encoding="utf-8")
    assert "neural networks overview" in content.lower()


# ── Test 14: Safe filename sanitization ──────────────────────────────────────

def test_safe_filename_basic():
    assert _safe_filename("Latest Developments in AI") == "latest-developments-in-ai"


def test_safe_filename_special_chars():
    assert _safe_filename("AI/ML: 2024 trends??") == "ai-ml-2024-trends"


def test_safe_filename_consecutive_hyphens():
    result = _safe_filename("AI   ---   research")
    assert "--" not in result


# ── Test 15: assign_source_identifiers deduplicates by unique URL ────────────

def test_assign_source_identifiers_deduplicates_by_url():
    """assign_source_identifiers must map each unique URL to one identifier [Si]."""
    results = [
        _make_search_result(url="https://same.com", title="Same Source"),
        _make_search_result(url="https://same.com", title="Same Source Chunk 2"),
        _make_search_result(url="https://different.com", title="Different"),
    ]
    registry = assign_source_identifiers(results)
    assert list(registry.keys()) == ["[S1]", "[S2]"]
    assert registry["[S1]"]["url"] == "https://same.com"
    assert registry["[S2]"]["url"] == "https://different.com"


# ── Requirement 9: Duplicate source URLs / chunks tests ───────────────────────

def test_context_block_maps_multiple_chunks_same_url_to_same_id():
    """Multiple chunks from the same URL must map to the same source ID in context block."""
    results = [
        _make_search_result(url="https://cmu.edu", title="CMU", text="Chunk 1 text"),
        _make_search_result(url="https://ibm.com", title="IBM", text="Chunk 2 text"),
        _make_search_result(url="https://cmu.edu", title="CMU", text="Chunk 3 text"),
    ]
    prompt = build_research_prompt("AI topic", results)

    # Both CMU chunks (Chunk 1 & Chunk 3) should be labeled [S1] in the context block
    assert "[S1]\nTitle: CMU\nURL: https://cmu.edu\nContent:\nChunk 1 text" in prompt
    assert "[S1]\nTitle: CMU\nURL: https://cmu.edu\nContent:\nChunk 3 text" in prompt
    assert "[S2]\nTitle: IBM\nURL: https://ibm.com\nContent:\nChunk 2 text" in prompt


def test_duplicate_source_urls_in_report_sources(tmp_path, monkeypatch):
    """The final Sources section must contain each unique URL exactly once."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url="https://cmu.edu", title="CMU"),
        _make_search_result(url="https://ibm.com", title="IBM"),
        _make_search_result(url="https://cmu.edu", title="CMU Chunk 2"),
        _make_search_result(url="https://sciencedaily.com", title="ScienceDaily"),
        _make_search_result(url="https://cmu.edu", title="CMU Chunk 3"),
    ]
    agent = _make_agent(retrieved=results)
    report = agent.research("test topic")

    # Sources section must contain 3 unique entries, not 5
    assert len(report.sources) == 3
    urls_in_report = [s.url for s in report.sources]
    assert len(urls_in_report) == len(set(urls_in_report))
    assert urls_in_report == ["https://cmu.edu", "https://ibm.com", "https://sciencedaily.com"]
    assert [s.identifier for s in report.sources] == ["[S1]", "[S2]", "[S3]"]


# ── Requirement 11 tests ──────────────────────────────────────────────────────

def test_all_valid_citations_have_corresponding_sources(tmp_path, monkeypatch):
    """11.a All valid citations have corresponding source entries."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 3)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url="https://s1.com", title="Title 1"),
        _make_search_result(url="https://s2.com", title="Title 2"),
        _make_search_result(url="https://s3.com", title="Title 3"),
    ]
    gemini_json = _make_valid_gemini_json(
        summary="Summary citing [S1].",
        findings=["Finding [S2]"],
        analysis="Analysis citing [S3].",
    )
    agent = _make_agent(retrieved=results, gemini_response=gemini_json)
    report = agent.research("test topic")

    source_idents = {s.identifier for s in report.sources}
    assert "[S1]" in source_idents
    assert "[S2]" in source_idents
    assert "[S3]" in source_idents


def test_invalid_citation_s99_detected(tmp_path, monkeypatch, caplog):
    """11.b An invalid citation such as [S99] is detected and removed gracefully."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 2)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url="https://s1.com", title="Title 1"),
        _make_search_result(url="https://s2.com", title="Title 2"),
    ]
    gemini_json = _make_valid_gemini_json(
        summary="Summary citing [S1] and invalid [S99].",
        findings=["Finding [S99]"],
    )
    agent = _make_agent(retrieved=results, gemini_response=gemini_json)

    with caplog.at_level("WARNING"):
        report = agent.research("test topic")

    source_idents = {s.identifier for s in report.sources}
    assert "[S99]" not in source_idents
    assert any("Invalid source citation" in record.message for record in caplog.records)
    assert "[S99]" not in report.executive_summary
    assert "[S99]" not in report.key_findings[0]


def test_gemini_cannot_introduce_arbitrary_urls(tmp_path, monkeypatch):
    """11.c Gemini cannot introduce arbitrary URLs into the final Sources section."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 1)
    (tmp_path / "reports").mkdir()

    trusted_url = "https://trusted.com/doc"
    results = [_make_search_result(url=trusted_url, title="Trusted")]
    gemini_json = json.dumps({
        "executive_summary": "Summary mentioning http://fake-gemini-url.org",
        "key_findings": ["Finding"],
        "detailed_analysis": "Analysis",
        "emerging_trends": [],
        "conclusion": "Conclusion",
    })
    agent = _make_agent(retrieved=results, gemini_response=gemini_json)
    report = agent.research("test topic")

    for src in report.sources:
        assert src.url == trusted_url
        assert "fake-gemini-url" not in src.url


def test_source_urls_always_come_from_trusted_metadata(tmp_path, monkeypatch):
    """11.d Source URLs in the final report always come from trusted metadata."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 2)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url="https://metadata-url-1.com", title="Title 1"),
        _make_search_result(url="https://metadata-url-2.com", title="Title 2"),
    ]
    agent = _make_agent(retrieved=results)
    report = agent.research("test topic")

    assert len(report.sources) == 2
    assert report.sources[0].url == "https://metadata-url-1.com"
    assert report.sources[1].url == "https://metadata-url-2.com"


def test_report_referencing_s5_includes_s5_in_sources(tmp_path, monkeypatch):
    """11.e A report referencing [S5] includes [S5] in the final Sources section."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url=f"https://s{i}.com", title=f"Source {i}")
        for i in range(1, 6)
    ]
    gemini_json = _make_valid_gemini_json(findings=["Finding five [S5]"])
    agent = _make_agent(retrieved=results, gemini_response=gemini_json)
    report = agent.research("test topic")

    s5_entry = next((s for s in report.sources if s.identifier == "[S5]"), None)
    assert s5_entry is not None
    assert s5_entry.title == "Source 5"
    assert s5_entry.url == "https://s5.com"


def test_report_valid_if_gemini_uses_only_s1_and_s2(tmp_path, monkeypatch):
    """11.f The report remains valid if Gemini uses only [S1] and [S2]."""
    monkeypatch.setattr("app.research.agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("app.research.agent.config.TOP_K", 5)
    (tmp_path / "reports").mkdir()

    results = [
        _make_search_result(url=f"https://s{i}.com", title=f"Source {i}")
        for i in range(1, 6)
    ]
    gemini_json = _make_valid_gemini_json(
        summary="Overview [S1].",
        findings=["Finding [S2]"],
        analysis="Analysis [S1].",
        trends=["Trend [S2]"],
        conclusion="Conclusion [S1].",
    )
    agent = _make_agent(retrieved=results, gemini_response=gemini_json)
    report = agent.research("test topic")

    assert isinstance(report, ResearchReport)
    assert len(report.sources) == 5
    assert report.sources[0].identifier == "[S1]"
    assert report.sources[1].identifier == "[S2]"

