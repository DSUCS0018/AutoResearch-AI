"""
app/api/main.py
---------------
FastAPI application initialization and router configuration.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.models import HealthResponse
from app.api.routes import history, jobs, research, schedules, whats_new
from app.config import config
from app.scheduler.scheduler import get_scheduler_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    FastAPI lifespan context manager.

    Starts the background scheduler on API startup, loading all enabled persisted schedules.
    Stops the background scheduler gracefully on API shutdown.
    """
    logger.info("FastAPI starting up — initializing and starting SchedulerService...")
    scheduler_service = get_scheduler_service()
    scheduler_service.start()
    yield
    logger.info("FastAPI shutting down — stopping SchedulerService...")
    scheduler_service.stop()


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="AutoResearch AI backend API endpoints.",
    lifespan=lifespan,
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(research.router)
app.include_router(jobs.router)
app.include_router(history.router)
app.include_router(whats_new.router)
app.include_router(schedules.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", service=config.APP_NAME)
