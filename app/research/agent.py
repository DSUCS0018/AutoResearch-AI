"""
app/research/agent.py
---------------------
ResearchAgent — coordinates search, collection, RAG, and Gemini to
produce a grounded, cited research report.

The agent wires together existing Stage 3 and Stage 4 components.
No new API clients are created here.

Usage:
    from app.research.agent import ResearchAgent

    agent = ResearchAgent()
    report = agent.research("latest developments in artificial intelligence")
    print(report.executive_summary)
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import config
from app.llm.client import GeminiClient, LLMConfigurationError, LLMResponseError
from app.rag.service import RAGService
from app.research.prompts import (
    assign_source_identifiers,
    build_research_prompt,
)
from app.research.report import ResearchReport, SourceReference
from app.search.collector import CollectorError, SourceCollector
from app.search.models import Document
from app.search.tavily_client import (
    SearchAPIError,
    SearchConfigurationError,
    TavilySearchClient,
)

logger = logging.getLogger(__name__)


from app.memory.repository import ResearchRepository


class ResearchAgentError(Exception):
    """Raised when the research workflow cannot be completed."""


class ResearchReportError(Exception):
    """Raised when Gemini's response cannot be parsed into a report."""


class ResearchAgent:
    """
    Orchestrates the full AutoResearch AI pipeline.

    Workflow:
        topic → Tavily search → webpage collection → RAG ingest
              → semantic retrieval → Gemini → ResearchReport → SQLite Memory

    All sub-components are injected so the agent is fully testable
    without real API calls.
    """

    def __init__(
        self,
        search_client: Optional[TavilySearchClient] = None,
        collector: Optional[SourceCollector] = None,
        rag_service: Optional[RAGService] = None,
        llm_client: Optional[GeminiClient] = None,
        repository: Optional[ResearchRepository] = None,
    ) -> None:
        self._search = search_client or TavilySearchClient()
        self._collector = collector or SourceCollector()
        self._rag = rag_service or RAGService()
        self._llm = llm_client or GeminiClient()
        self._repository = repository or ResearchRepository()
        logger.debug("ResearchAgent initialised.")

    # ── Public API ───────────────────────────────────────────────────────────

    def research(self, topic: str) -> ResearchReport:
        """
        Run the full research pipeline for a topic.

        Args:
            topic: The research topic string.

        Returns:
            A ResearchReport with citations and metadata.

        Raises:
            ResearchAgentError: If the pipeline fails unrecoverably.
        """
        topic = (topic or "").strip()
        if not topic:
            raise ResearchAgentError("Research topic must not be empty.")

        logger.info("ResearchAgent: starting research on %r", topic)

        # ── Step 1: Search ────────────────────────────────────────────────
        logger.info("Step 1: Searching Tavily...")
        try:
            sources = self._search.search(topic)
        except (SearchConfigurationError, SearchAPIError) as exc:
            raise ResearchAgentError(f"Search failed: {exc}") from exc

        if not sources:
            raise ResearchAgentError(
                f"Tavily returned no results for topic: {topic!r}"
            )
        logger.info("Found %d source(s) from Tavily.", len(sources))

        # ── Step 2: Collect webpages ──────────────────────────────────────
        logger.info("Step 2: Collecting webpages...")
        documents: list[Document] = []
        for source in sources:
            try:
                doc = self._collector.collect(source.url)
                if doc.text.strip():
                    documents.append(doc)
                    logger.info("Collected: %s", source.url)
                else:
                    logger.warning("Empty content from: %s — skipping.", source.url)
            except CollectorError as exc:
                logger.warning("Collection failed for %s: %s — skipping.", source.url, exc)
                continue

        if not documents:
            raise ResearchAgentError(
                "All source collection attempts failed. Cannot generate report."
            )
        logger.info("Successfully collected %d document(s).", len(documents))

        # ── Step 3: RAG ingest ────────────────────────────────────────────
        logger.info("Step 3: Ingesting documents into RAG...")
        summary = self._rag.ingest_documents(documents)
        logger.info(
            "Ingested: %d docs, %d chunks, %d stored.",
            summary["documents"], summary["chunks"], summary["stored"],
        )

        # ── Step 4: Semantic retrieval ────────────────────────────────────
        logger.info("Step 4: Retrieving relevant chunks...")
        retrieved = self._rag.retrieve(topic, top_k=config.TOP_K)

        if not retrieved:
            raise ResearchAgentError(
                "RAG retrieval returned no chunks. Cannot ground the report."
            )
        logger.info("Retrieved %d chunk(s).", len(retrieved))

        # ── Step 5: Build trusted source registry ─────────────────────────
        source_registry = assign_source_identifiers(retrieved)

        # ── Step 6: Build prompt and call Gemini ──────────────────────────
        logger.info("Step 5: Building prompt and calling Gemini...")
        prompt = build_research_prompt(topic, retrieved, source_registry)

        try:
            raw_response = self._llm.generate(prompt)
        except (LLMConfigurationError, LLMResponseError) as exc:
            raise ResearchAgentError(f"Gemini generation failed: {exc}") from exc

        # ── Step 7: Parse JSON response and validate citations ────────────
        logger.info("Step 6: Parsing Gemini response...")
        parsed = self._parse_gemini_response(raw_response)
        referenced = extract_referenced_citations(parsed)
        logger.info(
            "Extracted %d referenced source citation(s): %s",
            len(referenced),
            sorted(referenced),
        )
        parsed = clean_and_validate_citations(parsed, set(source_registry.keys()))

        # ── Step 8: Build ResearchReport with trusted sources ─────────────
        source_refs = [
            SourceReference(identifier=ident, title=meta["title"], url=meta["url"])
            for ident, meta in source_registry.items()
        ]

        report = ResearchReport(
            topic=topic,
            executive_summary=parsed.get("executive_summary", ""),
            key_findings=parsed.get("key_findings", []),
            detailed_analysis=parsed.get("detailed_analysis", ""),
            emerging_trends=parsed.get("emerging_trends", []),
            conclusion=parsed.get("conclusion", ""),
            sources=source_refs,
            generated_at=datetime.now(timezone.utc),
        )

        # ── Step 9: Save Markdown report ──────────────────────────────────
        report_path = self._save_report(report)
        logger.info("Report saved to: %s", report_path)

        # ── Step 10: Persist to repository memory ─────────────────────────
        try:
            run_id = self._repository.save_research_run(report, report_path=report_path)
            logger.info("Research run persisted to memory DB (ID: %d).", run_id)
        except Exception as exc:
            logger.warning("Failed to persist research run to memory DB: %s", exc)

        return report

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_gemini_response(raw: str) -> dict:
        """
        Parse Gemini's JSON response into a dict.

        Strips any accidental markdown fences that Gemini might add despite
        the strict prompt. Raises ResearchReportError on failure.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ResearchReportError(
                f"Gemini response is not valid JSON. "
                f"Parse error: {exc}. "
                f"Raw response (first 500 chars): {raw[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise ResearchReportError(
                f"Expected a JSON object from Gemini, got {type(data).__name__}."
            )

        return data

    @staticmethod
    def _save_report(report: ResearchReport) -> Path:
        """
        Save the report as a Markdown file under reports/.

        Returns the saved file path.
        """
        reports_dir = config.PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _safe_filename(report.topic)
        path = reports_dir / f"{safe_name}.md"

        path.write_text(report.to_markdown(), encoding="utf-8")
        return path


def _safe_filename(topic: str) -> str:
    """
    Convert a research topic into a safe, lowercase filename.

    Examples:
        "Latest Developments in AI" → "latest-developments-in-ai"
        "AI/ML: 2024 trends??"     → "ai-ml-2024-trends"
    """
    # Lowercase
    name = topic.lower()
    # Replace any non-alphanumeric characters with hyphens
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # Strip leading/trailing hyphens
    name = name.strip("-")
    # Collapse consecutive hyphens
    name = re.sub(r"-{2,}", "-", name)
    # Truncate to 80 characters to keep filenames reasonable
    return name[:80]


def extract_referenced_citations(parsed: dict) -> set[str]:
    """Extract all source identifiers [S1], [S2], etc. referenced in the report text."""
    found: set[str] = set()

    def _scan(val):
        if isinstance(val, str):
            found.update(re.findall(r"\[S\d+\]", val))

    for val in parsed.values():
        if isinstance(val, str):
            _scan(val)
        elif isinstance(val, list):
            for item in val:
                _scan(item)
    return found


def clean_and_validate_citations(parsed: dict, valid_identifiers: set[str]) -> dict:
    """
    Validate all citation identifiers in the parsed report fields.
    If an invalid identifier (e.g., [S99]) is found:
      - Log a warning
      - Remove the invalid citation from the text
      - Do not crash
    Returns the cleaned dictionary.
    """
    cleaned = dict(parsed)

    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text

        def _replace_match(match: re.Match) -> str:
            citation = match.group(0)
            if citation not in valid_identifiers:
                logger.warning("Invalid source citation detected and removed: %s", citation)
                return ""
            return citation

        result = re.sub(r"\[S\d+\]", _replace_match, text)
        result = re.sub(r"\s+([.,;:!?])", r"\1", result)
        result = re.sub(r" {2,}", " ", result)
        return result.strip()

    for key, value in cleaned.items():
        if isinstance(value, str):
            cleaned[key] = _clean_text(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_text(item) if isinstance(item, str) else item for item in value
            ]

    return cleaned
