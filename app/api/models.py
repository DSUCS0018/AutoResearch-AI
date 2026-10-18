"""
app/api/models.py
-----------------
Pydantic request and response data models for the AutoResearch AI FastAPI endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="The research topic string.")


class SourceModel(BaseModel):
    identifier: str
    title: str
    url: str


class ResearchResponse(BaseModel):
    topic: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str
    emerging_trends: List[str]
    conclusion: str
    sources: List[SourceModel]
    report_path: str
    generated_at: str


class HistoryRunSummary(BaseModel):
    topic: str
    generated_at: str
    sources_count: int
    findings_count: int
    executive_summary_preview: str
    report_path: str


class HistoryResponse(BaseModel):
    topic: str
    runs: List[HistoryRunSummary]


class WhatsNewResponse(BaseModel):
    topic: str
    has_previous_research: bool
    new_findings: List[str]
    changed_findings: List[str]
    new_sources: List[SourceModel]
    unchanged_findings: List[str]
    report_path: str
    generated_at: str


class ScheduleCreateRequest(BaseModel):
    topic: str
    frequency: str = Field(..., description="'daily', 'weekly', or 'once'")
    time: Optional[str] = Field(None, description="Time string in 'HH:MM' format e.g. '09:00'")
    day: Optional[str] = Field(None, description="Day of week for weekly schedules e.g. 'monday'")
    delay_seconds: Optional[int] = Field(None, description="Delay in seconds for 'once' schedules (default: 60)")


class TestJobCreateRequest(BaseModel):
    topic: Optional[str] = Field("Safe Scheduler Test", description="Research topic for test job")
    delay_seconds: Optional[int] = Field(60, description="Seconds until automatic execution (default: 60)")


class ScheduleUpdateRequest(BaseModel):
    enabled: Optional[bool] = None


class ScheduleResponse(BaseModel):
    id: str
    topic: str
    frequency: str
    day_of_week: Optional[str] = None
    hour: int
    minute: int
    enabled: bool
    next_run: str
    created_at: str
    run_at: Optional[str] = None


class SchedulerStatusResponse(BaseModel):
    running: bool
    timezone: str
    timezone_offset: str
    current_time: str
    total_jobs: int


# ── Job Lifecycle Models ──────────────────────────────────────────────────────

class JobCreateRequest(BaseModel):
    topic: str = Field(..., description="The research topic string.")


class JobResponse(BaseModel):
    id: str
    topic: str
    trigger_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
    schedule_id: Optional[str] = None
    research_run_id: Optional[int] = None
    report_path: Optional[str] = None
    whats_new_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at_display: Optional[str] = None
    started_at_display: Optional[str] = None
    completed_at_display: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


class JobStatusResponse(BaseModel):
    id: str
    status: str
    topic: str
    trigger_type: str
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class JobReportsResponse(BaseModel):
    job_id: str
    topic: str
    status: str
    report_path: Optional[str] = None
    report_content: Optional[str] = None
    whats_new_path: Optional[str] = None
    whats_new_content: Optional[str] = None

