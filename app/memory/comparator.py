"""
app/memory/comparator.py
------------------------
Deterministic comparator for current vs previous research reports.

Generates a WhatsNewReport highlighting new findings, changed findings,
new sources, and already known findings without using an LLM.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.research.report import ResearchReport, SourceReference


def _normalize_finding(text: str) -> str:
    """Normalize finding text by lowercasing and stripping whitespace/citations."""
    cleaned = text.strip().lower()
    # Remove citation identifiers like [S1], [S2] for comparison
    cleaned = re.sub(r"\[S\d+\]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class WhatsNewReport:
    """
    Report summarizing changes between a new research run and previous research.
    """
    topic: str
    new_findings: list[str] = field(default_factory=list)
    changed_findings: list[str] = field(default_factory=list)
    new_sources: list[SourceReference] = field(default_factory=list)
    unchanged_findings: list[str] = field(default_factory=list)
    has_previous_research: bool = True
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "topic": self.topic,
            "has_previous_research": self.has_previous_research,
            "new_findings": self.new_findings,
            "changed_findings": self.changed_findings,
            "new_sources": [
                {"identifier": s.identifier, "title": s.title, "url": s.url}
                for s in self.new_sources
            ],
            "unchanged_findings": self.unchanged_findings,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Render the What's New report as a Markdown document."""
        lines: list[str] = []

        lines.append(f"# What's New: {self.topic}")
        lines.append("")
        lines.append(f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        if not self.has_previous_research:
            lines.append("No previous research run exists for this topic.")
            lines.append("")
            lines.append("## New Findings")
            lines.append("")
            if self.new_findings:
                for finding in self.new_findings:
                    lines.append(f"- {finding}")
            else:
                lines.append("_No findings generated._")
            lines.append("")

            lines.append("## New Sources")
            lines.append("")
            if self.new_sources:
                for src in self.new_sources:
                    lines.append(f"{src.identifier} {src.title}")
                    lines.append(f"{src.url}")
                    lines.append("")
            else:
                lines.append("_No sources recorded._")

            return "\n".join(lines)

        lines.append("## New Findings")
        lines.append("")
        if self.new_findings:
            for finding in self.new_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("_No new findings detected._")
        lines.append("")

        lines.append("## Changed Findings")
        lines.append("")
        if self.changed_findings:
            for finding in self.changed_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("_No changed findings detected._")
        lines.append("")

        lines.append("## New Sources")
        lines.append("")
        if self.new_sources:
            for src in self.new_sources:
                lines.append(f"{src.identifier} {src.title}")
                lines.append(f"{src.url}")
                lines.append("")
        else:
            lines.append("_No new sources detected._")

        lines.append("## Already Known")
        lines.append("")
        if self.unchanged_findings:
            for finding in self.unchanged_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("_No matching prior findings._")

        return "\n".join(lines)


class ResearchComparator:
    """
    Compares current ResearchReport with previous ResearchReport deterministically.
    """

    def compare(
        self, current: ResearchReport, previous: Optional[ResearchReport]
    ) -> WhatsNewReport:
        """
        Compare current report against previous report.

        Args:
            current: The newly generated ResearchReport.
            previous: The prior ResearchReport for the topic (or None).

        Returns:
            A WhatsNewReport detailing new findings, new sources, and unchanged items.
        """
        if previous is None:
            return WhatsNewReport(
                topic=current.topic,
                new_findings=list(current.key_findings + current.emerging_trends),
                changed_findings=[],
                new_sources=list(current.sources),
                unchanged_findings=[],
                has_previous_research=False,
                generated_at=datetime.now(timezone.utc),
            )

        # 1. Compare sources by canonical URL
        prev_urls = {s.url.strip() for s in previous.sources if s.url}
        new_sources = [
            src for src in current.sources if src.url.strip() not in prev_urls
        ]

        # 2. Compare findings (key_findings + emerging_trends)
        prev_findings_raw = previous.key_findings + previous.emerging_trends
        prev_norm_map = {
            _normalize_finding(f): f for f in prev_findings_raw if f.strip()
        }

        curr_findings_raw = current.key_findings + current.emerging_trends

        new_findings: list[str] = []
        changed_findings: list[str] = []
        unchanged_findings: list[str] = []

        for finding in curr_findings_raw:
            norm = _normalize_finding(finding)
            if not norm:
                continue

            if norm in prev_norm_map:
                unchanged_findings.append(finding)
            else:
                # Check for near-exact match (e.g. high prefix/substring overlap)
                matched_prev = None
                for p_norm, p_raw in prev_norm_map.items():
                    if len(norm) > 20 and len(p_norm) > 20:
                        if norm in p_norm or p_norm in norm:
                            matched_prev = p_raw
                            break

                if matched_prev:
                    changed_findings.append(finding)
                else:
                    new_findings.append(finding)

        return WhatsNewReport(
            topic=current.topic,
            new_findings=new_findings,
            changed_findings=changed_findings,
            new_sources=new_sources,
            unchanged_findings=unchanged_findings,
            has_previous_research=True,
            generated_at=datetime.now(timezone.utc),
        )
