"""
app/api/routes/jobs.py
-----------------------
FastAPI endpoints for persistent research job lifecycle management.
Provides non-blocking job enqueueing, status polling, listing, and report retrieval.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.api.models import (
    JobCreateRequest,
    JobListResponse,
    JobReportsResponse,
    JobResponse,
    JobStatusResponse,
)
from app.memory.repository import ResearchRepository
from app.services.research_service import (
    DuplicateJobError,
    _background_execute,
    get_research_execution_service,
)

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(req: JobCreateRequest, background_tasks: BackgroundTasks):
    """
    Create and enqueue a new manual research job.
    Non-blocking: starts execution in background tasks and returns HTTP 202 immediately.
    """
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research topic must not be empty.",
        )

    service = get_research_execution_service()
    try:
        job = service.create_and_enqueue_job(topic=topic, trigger_type="manual")
    except DuplicateJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Enqueue execution in background — use isolated wrapper so the thread
    # is independent of the global service instance and any future replacements.
    background_tasks.add_task(_background_execute, job.id, service.repository.db_path)

    return JobResponse(**job.to_dict())


@router.get("", response_model=JobListResponse)
def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (queued, running, completed, failed)"),
    topic: Optional[str] = Query(None, description="Filter by research topic"),
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
):
    """
    List persistent research jobs with optional topic and status filters.
    """
    repo = ResearchRepository()
    jobs = repo.list_jobs(status=status_filter, topic=topic, limit=limit)
    return JobListResponse(
        jobs=[JobResponse(**j.to_dict()) for j in jobs],
        total=len(jobs),
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """
    Fetch a persistent research job by its unique ID.
    """
    repo = ResearchRepository()
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research job {job_id!r} not found.",
        )
    return JobResponse(**job.to_dict())


@router.get("/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Fetch quick execution status for a job ID.
    """
    repo = ResearchRepository()
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research job {job_id!r} not found.",
        )
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        topic=job.topic,
        trigger_type=job.trigger_type,
        duration_seconds=job.calculate_duration(),
        error=job.error,
    )


@router.get("/{job_id}/reports", response_model=JobReportsResponse)
def get_job_reports(job_id: str):
    """
    Fetch the generated research report and What's New report content for a job.
    """
    repo = ResearchRepository()
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research job {job_id!r} not found.",
        )

    report_content = None
    if job.report_path:
        p = Path(job.report_path)
        if p.exists() and p.is_file():
            try:
                report_content = p.read_text(encoding="utf-8")
            except Exception:
                pass

    whats_new_content = None
    if job.whats_new_path:
        wn = Path(job.whats_new_path)
        if wn.exists() and wn.is_file():
            try:
                whats_new_content = wn.read_text(encoding="utf-8")
            except Exception:
                pass

    return JobReportsResponse(
        job_id=job.id,
        topic=job.topic,
        status=job.status,
        report_path=job.report_path,
        report_content=report_content,
        whats_new_path=job.whats_new_path,
        whats_new_content=whats_new_content,
    )
