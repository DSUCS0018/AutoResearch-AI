"""
app/research/report.py
----------------------
Data models for a structured research report.

ResearchReport is a pure dataclass — no LLM logic lives here.
It is serializable to JSON and Markdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SourceReference:
    """
    A trusted source attached to a research report.

    The identifier ([S1], [S2], …) is assigned by the agent, not by Gemini.
    URLs come exclusively from Tavily/Document metadata — Gemini cannot invent them.
    """
    identifier: str   # e.g. "[S1]"
    title: str
    url: str


@dataclass
class ResearchReport:
    """
    A structured research report produced by the ResearchAgent.

    All source URLs are taken from trusted Tavily/Document metadata.
    Gemini only controls the text fields.
    """
    topic: str
    executive_summary: str
    key_findings: list[str]
    detailed_analysis: str
    emerging_trends: list[str]
    conclusion: str
    sources: list[SourceReference]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None          # populated if partial failure

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "topic": self.topic,
            "executive_summary": self.executive_summary,
            "key_findings": self.key_findings,
            "detailed_analysis": self.detailed_analysis,
            "emerging_trends": self.emerging_trends,
            "conclusion": self.conclusion,
            "sources": [
                {"identifier": s.identifier, "title": s.title, "url": s.url}
                for s in self.sources
            ],
            "generated_at": self.generated_at.isoformat(),
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines: list[str] = []

        lines.append(f"# Research Report: {self.topic}")
        lines.append("")
        lines.append(f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self.executive_summary or "_Not available._")
        lines.append("")

        lines.append("## Key Findings")
        lines.append("")
        if self.key_findings:
            for finding in self.key_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("_No key findings extracted._")
        lines.append("")

        lines.append("## Detailed Analysis")
        lines.append("")
        lines.append(self.detailed_analysis or "_Not available._")
        lines.append("")

        lines.append("## Emerging Trends")
        lines.append("")
        if self.emerging_trends:
            for trend in self.emerging_trends:
                lines.append(f"- {trend}")
        else:
            lines.append("_No emerging trends identified._")
        lines.append("")

        lines.append("## Conclusion")
        lines.append("")
        lines.append(self.conclusion or "_Not available._")
        lines.append("")

        lines.append("## Sources")
        lines.append("")
        if self.sources:
            for src in self.sources:
                lines.append(f"{src.identifier} {src.title}")
                lines.append(f"{src.url}")
                lines.append("")
        else:
            lines.append("_No sources recorded._")

        if self.error:
            lines.append("")
            lines.append("> **Warning:** " + self.error)

        return "\n".join(lines)
