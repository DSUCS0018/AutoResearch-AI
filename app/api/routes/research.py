"""
app/api/routes/research.py
--------------------------
FastAPI route for running end-to-end research.
Uses the shared ResearchExecutionService so that duplicate-run protection
and lifecycle tracking apply regardless of which entry-point triggered the run.
"""

from fastapi import APIRouter, HTTPException, status
from app.api.models import ResearchRequest, ResearchResponse, SourceModel
from app.config import config
from app.research.agent import ResearchAgent, ResearchAgentError, ResearchReportError, _safe_filename
from app.services.research_service import DuplicateJobError, get_research_execution_service

router = APIRouter(prefix="/api/research", tags=["Research"])


@router.post("", response_model=ResearchResponse, status_code=status.HTTP_200_OK)
def run_research(req: ResearchRequest):
    """
    Run end-to-end research on a given topic using Tavily, RAG, and Gemini.
    Synchronous (blocks until complete) for backward compatibility with the
    Streamlit Research page.  Shares the same duplicate-run guard as POST /api/jobs.
    """
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research topic must not be empty.",
        )

    service = get_research_execution_service()
    try:
        job = service.run_job_sync(topic=topic, trigger_type="manual")
    except DuplicateJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ResearchAgentError, ResearchReportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {exc}",
        ) from exc

    if job.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {job.error}",
        )

    safe_name = _safe_filename(topic)
    report_path = job.report_path or str(config.REPORTS_DIR / f"{safe_name}.md")

    # Use the report cached on the service instance (avoids an extra DB round-trip).
    report = getattr(service, "_last_report", None) or service.repository.get_latest_research(topic)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve generated research report.",
        )

    sources = [
        SourceModel(identifier=src.identifier, title=src.title, url=src.url)
        for src in report.sources
    ]

    return ResearchResponse(
        topic=report.topic,
        executive_summary=report.executive_summary,
        key_findings=report.key_findings,
        detailed_analysis=report.detailed_analysis,
        emerging_trends=report.emerging_trends,
        conclusion=report.conclusion,
        sources=sources,
        report_path=report_path,
        generated_at=report.generated_at.isoformat(),
    )
