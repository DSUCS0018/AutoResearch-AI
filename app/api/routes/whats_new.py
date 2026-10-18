"""
app/api/routes/whats_new.py
---------------------------
FastAPI route for generating and returning "What's New?" comparison reports.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.models import ResearchRequest, SourceModel, WhatsNewResponse
from app.config import config
from app.memory.comparator import ResearchComparator
from app.memory.repository import ResearchRepository
from app.research.agent import ResearchAgent, ResearchAgentError, ResearchReportError, _safe_filename

router = APIRouter(prefix="/api/whats-new", tags=["What's New"])


@router.post("", response_model=WhatsNewResponse, status_code=status.HTTP_200_OK)
def get_whats_new(req: ResearchRequest):
    """
    Generate a fresh research report and compare it against the latest previous research run.
    """
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research topic must not be empty.",
        )

    try:
        repo = ResearchRepository()

        # Step 1: Capture previous research run BEFORE fresh research
        previous_run = repo.get_latest_research(topic)

        # Step 2: Run fresh research cycle
        agent = ResearchAgent(repository=repo)
        current_report = agent.research(topic)

        # Step 3: Compare new report against previous run
        comparator = ResearchComparator()
        whats_new = comparator.compare(current_report, previous_run)

        # Step 4: Save What's New report to disk
        safe_name = _safe_filename(topic)
        filename = f"{safe_name}-whats-new.md"
        report_path = str(config.REPORTS_DIR / filename)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / filename).write_text(whats_new.to_markdown(), encoding="utf-8")

        new_sources = [
            SourceModel(identifier=src.identifier, title=src.title, url=src.url)
            for src in whats_new.new_sources
        ]

        return WhatsNewResponse(
            topic=whats_new.topic,
            has_previous_research=whats_new.has_previous_research,
            new_findings=whats_new.new_findings,
            changed_findings=whats_new.changed_findings,
            new_sources=new_sources,
            unchanged_findings=whats_new.unchanged_findings,
            report_path=report_path,
            generated_at=whats_new.generated_at.isoformat(),
        )

    except (ResearchAgentError, ResearchReportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What's New generation failed: {exc}",
        ) from exc
