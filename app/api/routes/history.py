"""
app/api/routes/history.py
-------------------------
FastAPI route for retrieving research history for a topic.
"""

from fastapi import APIRouter, HTTPException, status
from app.api.models import HistoryResponse, HistoryRunSummary
from app.config import config
from app.memory.repository import ResearchRepository
from app.research.agent import _safe_filename

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("/{topic:path}", response_model=HistoryResponse)
def get_history(topic: str):
    """
    Retrieve historical research runs for a topic.
    """
    clean_topic = topic.strip()
    if not clean_topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic parameter must not be empty.",
        )

    repo = ResearchRepository()
    history = repo.get_research_history(clean_topic)

    summaries = []
    for report in history:
        safe_name = _safe_filename(report.topic)
        report_path = str(config.REPORTS_DIR / f"{safe_name}.md")
        preview = report.executive_summary[:200] if report.executive_summary else ""

        summaries.append(
            HistoryRunSummary(
                topic=report.topic,
                generated_at=report.generated_at.isoformat(),
                sources_count=len(report.sources),
                findings_count=len(report.key_findings) + len(report.emerging_trends),
                executive_summary_preview=preview,
                report_path=report_path,
            )
        )

    return HistoryResponse(topic=clean_topic, runs=summaries)
