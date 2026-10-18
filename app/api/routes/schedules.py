"""
app/api/routes/schedules.py
---------------------------
FastAPI routes for managing research schedules.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

from app.api.models import (
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleUpdateRequest,
    SchedulerStatusResponse,
    TestJobCreateRequest,
)
from app.scheduler.models import ScheduleValidationError
from app.scheduler.scheduler import SchedulerService, get_scheduler_service

router = APIRouter(prefix="/api/schedules", tags=["Schedules"])


@router.get("/status", response_model=SchedulerStatusResponse)
def get_scheduler_status():
    """Return scheduler engine running state, active timezone, and job counts."""
    service = get_scheduler_service()
    now_tz = datetime.now(service.tz)
    return SchedulerStatusResponse(
        running=service._scheduler.running,
        timezone=service.timezone_name,
        timezone_offset=now_tz.strftime("%z"),
        current_time=now_tz.strftime("%Y-%m-%d %H:%M:%S %Z"),
        total_jobs=len(service._scheduler.get_jobs()),
    )


@router.get("", response_model=List[ScheduleResponse])
def list_schedules():
    """List all scheduled research jobs."""
    service = get_scheduler_service()
    schedules = service.list_schedules()
    return [
        ScheduleResponse(
            id=s["id"],
            topic=s["topic"],
            frequency=s["frequency"],
            day_of_week=s.get("day_of_week"),
            hour=s["hour"],
            minute=s["minute"],
            enabled=s["enabled"],
            next_run=s.get("next_run", "N/A"),
            created_at=s["created_at"],
            run_at=s.get("run_at"),
        )
        for s in schedules
    ]


@router.post("", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
def create_schedule(req: ScheduleCreateRequest):
    """Create a new research schedule (daily, weekly, or once)."""
    service = get_scheduler_service()
    try:
        if req.frequency.lower() == "once":
            delay = req.delay_seconds if req.delay_seconds is not None else 60
            scheduled = service.add_test_job(topic=req.topic, delay_seconds=delay)
        else:
            scheduled = service.add_schedule(
                topic=req.topic,
                frequency=req.frequency,
                time_str=req.time,
                day_of_week=req.day,
            )
    except ScheduleValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    s_dict = scheduled.to_dict()
    aps_job = service._scheduler.get_job(scheduled.id)
    if aps_job and aps_job.next_run_time:
        next_run_str = aps_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        next_run_str = "Pending Start" if scheduled.enabled else "Not Scheduled"

    return ScheduleResponse(
        id=s_dict["id"],
        topic=s_dict["topic"],
        frequency=s_dict["frequency"],
        day_of_week=s_dict["day_of_week"],
        hour=s_dict["hour"],
        minute=s_dict["minute"],
        enabled=s_dict["enabled"],
        next_run=next_run_str,
        created_at=s_dict["created_at"],
        run_at=s_dict.get("run_at"),
    )


@router.post("/test-job", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
def create_test_job(req: Optional[TestJobCreateRequest] = None):
    """
    Create a safe one-time test job that runs in approximately 60 seconds.
    Enables immediate real-time verification of automated execution.
    """
    service = get_scheduler_service()
    topic = req.topic if req and req.topic else "Safe Scheduler Test"
    delay = req.delay_seconds if req and req.delay_seconds is not None else 60

    scheduled = service.add_test_job(topic=topic, delay_seconds=delay)
    s_dict = scheduled.to_dict()
    aps_job = service._scheduler.get_job(scheduled.id)
    if aps_job and aps_job.next_run_time:
        next_run_str = aps_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        next_run_str = "Pending Start"

    return ScheduleResponse(
        id=s_dict["id"],
        topic=s_dict["topic"],
        frequency=s_dict["frequency"],
        day_of_week=s_dict["day_of_week"],
        hour=s_dict["hour"],
        minute=s_dict["minute"],
        enabled=s_dict["enabled"],
        next_run=next_run_str,
        created_at=s_dict["created_at"],
        run_at=s_dict.get("run_at"),
    )


@router.patch("/{job_id}", response_model=ScheduleResponse)
def update_schedule(job_id: str, req: ScheduleUpdateRequest):
    """Update a schedule state (enable / disable)."""
    service = get_scheduler_service()
    if req.enabled is True:
        service.enable_schedule(job_id)
    elif req.enabled is False:
        service.disable_schedule(job_id)

    sched = service.repository.get_schedule(job_id)
    if not sched:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {job_id!r} not found.",
        )

    s_dict = sched.to_dict()
    aps_job = service._scheduler.get_job(sched.id)
    if aps_job and aps_job.next_run_time:
        next_run_str = aps_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        next_run_str = "Not Scheduled" if not sched.enabled else "Pending Start"

    return ScheduleResponse(
        id=s_dict["id"],
        topic=s_dict["topic"],
        frequency=s_dict["frequency"],
        day_of_week=s_dict["day_of_week"],
        hour=s_dict["hour"],
        minute=s_dict["minute"],
        enabled=s_dict["enabled"],
        next_run=next_run_str,
        created_at=s_dict["created_at"],
        run_at=s_dict.get("run_at"),
    )


@router.delete("/{job_id}", status_code=status.HTTP_200_OK)
def delete_schedule(job_id: str):
    """Delete a schedule by job ID."""
    service = get_scheduler_service()
    removed = service.remove_schedule(job_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {job_id!r} not found.",
        )
    return {"status": "ok", "message": f"Schedule {job_id!r} deleted."}


@router.post("/{job_id}/run", status_code=status.HTTP_200_OK)
def run_schedule_now(job_id: str):
    """Trigger execution of a scheduled job immediately."""
    service = get_scheduler_service()
    try:
        service.run_job_now(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule run failed: {exc}",
        ) from exc

    return {"status": "ok", "message": f"Scheduled job {job_id!r} executed."}


@router.post("/{job_id}/enable", status_code=status.HTTP_200_OK)
def enable_schedule(job_id: str):
    """Enable a scheduled job."""
    service = get_scheduler_service()
    updated = service.enable_schedule(job_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {job_id!r} not found.",
        )
    return {"status": "ok", "message": f"Schedule {job_id!r} enabled."}


@router.post("/{job_id}/disable", status_code=status.HTTP_200_OK)
def disable_schedule(job_id: str):
    """Disable a scheduled job."""
    service = get_scheduler_service()
    updated = service.disable_schedule(job_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {job_id!r} not found.",
        )
    return {"status": "ok", "message": f"Schedule {job_id!r} disabled."}
