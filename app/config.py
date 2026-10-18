"""
app/config.py
-------------
Centralized configuration for AutoResearch AI.
Environment variables are loaded from a .env file via python-dotenv.

Stage 2 adds Google Gemini API credentials.
Stage 3 adds Tavily Search API configuration.
Stage 4 adds RAG (chunking, embeddings, ChromaDB) configuration.
Future stages will extend this module with additional service settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (one level above app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Application configuration loaded from environment variables."""

    # Application identity
    APP_NAME: str = os.getenv("APP_NAME", "AutoResearch AI")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")

    # Runtime environment: development | staging | production
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ── Stage 2: Google Gemini ─────────────────────────────────────────────
    # Loaded from .env — never set a real key here.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model to use for all LLM calls. Override via GEMINI_MODEL in .env.
    # Default: gemini-2.5-flash-lite — fast and available on the free tier.
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # ── Stage 3: Tavily Search ──────────────────────────────────────────────
    # Loaded from .env — never set a real key here.
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Maximum number of search results to request from Tavily per query.
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

    # ── Stage 4: RAG ────────────────────────────────────────────────────────
    # Local sentence-transformers model for embeddings (free, no API key).
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Directory (relative to project root) for ChromaDB persistence.
    CHROMA_PERSIST_DIRECTORY: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", "data/chroma"
    )

    # Target characters per chunk (used by DocumentChunker).
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

    # Overlap characters between consecutive chunks.
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Default number of RAG results to return per query.
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # ── Stage 7: Scheduler ──────────────────────────────────────────────────
    SCHEDULER_TIMEZONE: str = os.getenv("SCHEDULER_TIMEZONE", "Asia/Kolkata")

    # Convenience paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    DATABASE_PATH: Path = DATA_DIR / "research.db"


# Module-level singleton — import this everywhere
config = Config()
