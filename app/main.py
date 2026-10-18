"""
app/main.py
-----------
Application entry point for AutoResearch AI.

Responsibilities:
  - Configure logging
  - Display startup banner
  - Provide a health check / status report
  - Provide manual test commands (--test-llm, --test-search, --test-rag, --research)

No research functionality is implemented here.
Future stages will register agents, services, and routes from this module.
"""

import logging
import sys
from typing import Optional
from app.config import config
from app.llm.client import GeminiClient, LLMConfigurationError, LLMResponseError
from app.search.tavily_client import TavilySearchClient, SearchConfigurationError, SearchAPIError
from app.search.collector import SourceCollector, CollectorError
from app.search.models import Document
from app.rag.service import RAGService
from app.research.agent import ResearchAgent, ResearchAgentError, ResearchReportError


def configure_logging() -> None:
    """Set up the root logger based on the configured log level."""
    numeric_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def print_banner() -> None:
    """Print the application startup banner."""
    width = 40
    print("=" * width)
    title = f"{config.APP_NAME} v{config.APP_VERSION}"
    print(title.center(width))
    print("=" * width)
    print()


def health_check() -> dict:
    """
    Return a basic status dictionary.

    Returns:
        dict: Application health/status information.
    """
    return {
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "environment": config.ENVIRONMENT,
        "status": "READY",
    }


def start() -> None:
    """
    Initialize and start AutoResearch AI.

    This is the main callable for normal startup. It:
      1. Configures logging.
      2. Prints the startup banner.
      3. Logs application metadata.
      4. Runs a health check and prints the result.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()

    logger.info("AutoResearch AI is starting up...")
    logger.info("Version     : %s", config.APP_VERSION)
    logger.info("Environment : %s", config.ENVIRONMENT)
    logger.info("Log Level   : %s", config.LOG_LEVEL)

    status = health_check()

    print(f"Environment : {status['environment']}")
    print(f"Status      : {status['status']}")
    print()

    logger.info("Startup complete. Application status: %s", status["status"])


def test_llm_connection() -> None:
    """
    Manually test the Google Gemini API connection.

    Called when the application is started with the --test-llm flag.
    This function WILL make a real API call.
    Normal startup via start() does NOT call this function.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()

    TEST_PROMPT = "Explain Retrieval-Augmented Generation in one sentence."

    print("Testing Gemini connection...")
    print()

    try:
        client = GeminiClient()
        logger.info("Gemini client initialised.")

        print(f"Prompt:\n{TEST_PROMPT}")
        print()

        response_text = client.generate(TEST_PROMPT)

        print(f"Response:\n{response_text}")
        print()
        print("Gemini connection: SUCCESS")
        logger.info("LLM test completed successfully.")

    except LLMConfigurationError as exc:
        print(f"Configuration error: {exc}")
        logger.error("LLM configuration error: %s", exc)
        sys.exit(1)

    except LLMResponseError as exc:
        print(f"API error: {exc}")
        logger.error("LLM response error: %s", exc)
        sys.exit(1)


def test_search_pipeline() -> None:
    """
    Manually test the Tavily search + source collection pipeline.

    Called when the application is started with the --test-search flag.
    This function WILL make real Tavily API calls and network requests.
    Normal startup via start() does NOT call this function.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()

    TEST_QUERY = "latest developments in artificial intelligence"

    print("Testing Tavily search...")
    print(f"Query: {TEST_QUERY}")
    print()

    # ── Step 1: Search ───────────────────────────────────────────────────────
    try:
        search_client = TavilySearchClient()
        sources = search_client.search(TEST_QUERY)
    except SearchConfigurationError as exc:
        print(f"Search configuration error: {exc}")
        logger.error("Search configuration error: %s", exc)
        sys.exit(1)
    except SearchAPIError as exc:
        print(f"Search API error: {exc}")
        logger.error("Search API error: %s", exc)
        sys.exit(1)

    if not sources:
        print("No search results returned.")
        sys.exit(1)

    print(f"Found {len(sources)} result(s):\n")
    for i, source in enumerate(sources, 1):
        score_str = f"  Score: {source.score:.2f}" if source.score else ""
        print(f"  [{i}] {source.title}")
        print(f"       {source.url}{score_str}")
    print()

    # ── Step 2: Collect first result ─────────────────────────────────────────
    first_url = sources[0].url
    print(f"Fetching first result: {first_url}")
    print()

    try:
        collector = SourceCollector()
        doc = collector.collect(first_url)
    except CollectorError as exc:
        print(f"Collection error: {exc}")
        logger.error("Collection error: %s", exc)
        sys.exit(1)

    print(f"Title   : {doc.title}")
    print(f"URL     : {doc.url}")
    print(f"Length  : {len(doc.text):,} characters")
    print(f"Preview :\n{doc.text[:500]}")
    print()
    print("Search pipeline: SUCCESS")
    logger.info("Search pipeline test completed successfully.")


def test_rag_pipeline() -> None:
    """
    Manually test the local RAG pipeline (chunking + embeddings + ChromaDB).

    Called when the application is started with the --test-rag flag.
    This function downloads the embedding model on first run (~90 MB).
    It does NOT call Gemini or Tavily.
    Normal startup via start() does NOT call this function.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()
    print("RAG test started")
    print()

    # ── Sample documents (no network calls) ──────────────────────────────────
    from datetime import datetime, timezone
    sample_docs = [
        Document(
            url="https://example.com/ai-agents",
            title="AI Agents",
            text=(
                "AI agents are software systems that can reason about tasks, "
                "use tools, observe results, and take actions to achieve goals. "
                "Modern AI agents combine large language models with external tools "
                "such as web search, code execution, and API calls. "
                "They can plan multi-step workflows, maintain state between steps, "
                "and adapt their behavior based on feedback from the environment."
            ),
            fetched_at=datetime.now(timezone.utc),
        ),
        Document(
            url="https://example.com/rag",
            title="Retrieval-Augmented Generation",
            text=(
                "Retrieval-Augmented Generation (RAG) retrieves relevant information "
                "from an external knowledge base before generating a response. "
                "This approach grounds the language model in factual, up-to-date "
                "documents and reduces hallucination. "
                "A typical RAG pipeline embeds documents into a vector database, "
                "then retrieves the most semantically similar chunks for a given query, "
                "and passes them as context to the LLM for synthesis."
            ),
            fetched_at=datetime.now(timezone.utc),
        ),
    ]

    # ── Ingest ───────────────────────────────────────────────────────────────
    try:
        rag = RAGService()
        summary = rag.ingest_documents(sample_docs)
    except Exception as exc:
        print(f"RAG ingest error: {exc}")
        logger.error("RAG ingest error: %s", exc)
        sys.exit(1)

    print(f"Documents ingested : {summary['documents']}")
    print(f"Chunks created     : {summary['chunks']}")
    print(f"Vectors stored     : {summary['stored']}")
    print(f"Vector store count : {rag.count()}")
    print()

    # ── Retrieve ─────────────────────────────────────────────────────────────
    TEST_QUERY = "What is Retrieval-Augmented Generation?"
    print(f"Query: {TEST_QUERY}")
    print()

    try:
        results = rag.retrieve(TEST_QUERY, top_k=3)
    except Exception as exc:
        print(f"RAG retrieve error: {exc}")
        logger.error("RAG retrieve error: %s", exc)
        sys.exit(1)

    if not results:
        print("No results returned from vector store.")
        sys.exit(1)

    print(f"Top {len(results)} result(s):\n")
    for i, result in enumerate(results, 1):
        title = result.metadata.get("source_title", "Unknown")
        url = result.metadata.get("source_url", "")
        dist = result.distance
        preview = result.text[:200].replace("\n", " ")
        print(f"  [{i}] {title}")
        print(f"       {url}")
        print(f"       Distance : {dist:.4f}")
        print(f"       Preview  : {preview}")
        print()

    print("RAG pipeline: SUCCESS")
    logger.info("RAG pipeline test completed successfully.")


def run_research_agent(topic: str) -> None:
    """
    Run the full end-to-end research pipeline for a given topic.

    Called when the application is started with --research "<topic>".
    This function WILL make real Tavily and Gemini API calls.
    Normal startup via start() does NOT call this function.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    width = 40
    print("=" * width)
    print("AutoResearch AI — Research Agent".center(width))
    print("=" * width)
    print()

    print(f"Topic:\n{topic}")
    print()

    print("Searching sources...")
    logger.info("Starting research on: %r", topic)

    try:
        agent = ResearchAgent()
        report = agent.research(topic)
    except ResearchAgentError as exc:
        print(f"Research failed: {exc}")
        logger.error("Research agent error: %s", exc)
        sys.exit(1)
    except ResearchReportError as exc:
        print(f"Report generation failed: {exc}")
        logger.error("Report error: %s", exc)
        sys.exit(1)

    print(f"Collected sources: {len(report.sources)}")
    print(f"Retrieved chunks : {config.TOP_K}")
    print()

    print("Generating report with Gemini...")
    print()
    print("Research completed successfully.")
    print()

    # ── Print report preview ────────────────────────────────────────────────
    print("Executive Summary:")
    print(report.executive_summary[:600] if report.executive_summary else "(none)")
    print()

    if report.key_findings:
        print("Key Findings:")
        for finding in report.key_findings[:5]:
            print(f"  • {finding}")
        print()

    # ── Print sources ───────────────────────────────────────────────────────
    print("Sources:")
    for src in report.sources:
        print(f"  {src.identifier} {src.title}")
        print(f"       {src.url}")
    print()

    # ── Report path ─────────────────────────────────────────────────────────
    from app.research.agent import _safe_filename
    report_filename = _safe_filename(report.topic) + ".md"
    report_path = config.PROJECT_ROOT / "reports" / report_filename
    print(f"Report saved to:\nreports/{report_filename}")
    logger.info("Research pipeline completed successfully. Report: %s", report_path)


def display_research_history(topic: str) -> None:
    """
    Retrieve and display previous research runs for a given topic.

    Called when started with --history "<topic>".
    """
    configure_logging()
    from app.memory.repository import ResearchRepository

    print_banner()
    print(f"Research History for: {topic!r}")
    print("=" * 40)

    repo = ResearchRepository()
    history = repo.get_research_history(topic)

    if not history:
        print("No previous research runs found for this topic.")
        return

    print(f"Found {len(history)} previous research run(s):\n")
    for i, report in enumerate(history, 1):
        dt_str = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        print(f"[{i}] Run Date : {dt_str}")
        print(f"    Topic    : {report.topic}")
        print(f"    Sources  : {len(report.sources)}")
        if report.key_findings:
            print(f"    Findings : {len(report.key_findings)} key finding(s)")
            print(f"    Preview  : {report.key_findings[0][:100]}...")
        print()


def run_whats_new(topic: str) -> None:
    """
    Run a fresh research cycle and compare it against the latest previous research.

    Called when started with --whats-new "<topic>".
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    from app.memory.repository import ResearchRepository
    from app.memory.comparator import ResearchComparator
    from app.research.agent import _safe_filename

    width = 40
    print("=" * width)
    print("AutoResearch AI — What's New?".center(width))
    print("=" * width)
    print()
    print(f"Topic: {topic!r}")
    print()

    # Step 1: Capture latest previous research run BEFORE starting fresh research
    repo = ResearchRepository()
    previous_run = repo.get_latest_research(topic)

    if previous_run:
        print(f"Previous research found: YES ({previous_run.generated_at.strftime('%Y-%m-%d %H:%M UTC')})")
        logger.info("Captured previous research run from: %s", previous_run.generated_at)
    else:
        print("Previous research found: NO")
        logger.info("No previous research run found for topic: %r", topic)

    # Step 2: Perform fresh Tavily + RAG + Gemini research run
    print("Starting fresh research cycle...")
    try:
        agent = ResearchAgent(repository=repo)
        current_report = agent.research(topic)
        print("New research completed: YES")
    except (ResearchAgentError, ResearchReportError) as exc:
        print(f"Research failed: {exc}")
        logger.error("What's New research failed: %s", exc)
        sys.exit(1)

    # Step 3: Compare new report against captured previous run
    comparator = ResearchComparator()
    whats_new_report = comparator.compare(current_report, previous_run)
    print("Comparison completed: YES")

    # Step 4: Save What's New report
    safe_topic = _safe_filename(topic)
    filename = f"{safe_topic}-whats-new.md"
    out_path = config.REPORTS_DIR / filename
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(whats_new_report.to_markdown(), encoding="utf-8")
    print(f"What's New report saved: reports/{filename}")
    print()

    # Step 5: Concise Terminal Summary
    print("=" * width)
    print("WHAT'S NEW SUMMARY".center(width))
    print("=" * width)
    if not whats_new_report.has_previous_research:
        print("No previous research run existed. This report establishes the initial baseline.")
        print(f"Total Findings : {len(whats_new_report.new_findings)}")
        print(f"Total Sources  : {len(whats_new_report.new_sources)}")
    else:
        print(f"New Findings     : {len(whats_new_report.new_findings)}")
        print(f"Changed Findings : {len(whats_new_report.changed_findings)}")
        print(f"Already Known    : {len(whats_new_report.unchanged_findings)}")
        print(f"New Sources      : {len(whats_new_report.new_sources)}")

        if whats_new_report.new_findings:
            print("\nNew Findings Preview:")
            for f in whats_new_report.new_findings[:3]:
                print(f"  • {f}")

        if whats_new_report.new_sources:
            print("\nNew Sources Preview:")
            for s in whats_new_report.new_sources[:3]:
                print(f"  • {s.identifier} {s.title} ({s.url})")


def test_memory_pipeline() -> None:
    """
    Manually test SQLite persistent memory and ResearchComparator without real API calls.
    """
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    from app.memory.repository import ResearchRepository
    from app.memory.comparator import ResearchComparator
    from app.research.report import ResearchReport, SourceReference

    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()
    print("Testing Persistent Research Memory (SQLite) & Comparator...")
    print()

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_research.db"
        repo = ResearchRepository(db_path=test_db)

        # 1. Create sample report 1
        report1 = ResearchReport(
            topic="Artificial Intelligence Trends",
            executive_summary="AI is evolving rapidly with LLMs and agents.",
            key_findings=["AI agents reason and execute tasks.", "LLMs power modern chatbots."],
            detailed_analysis="Detailed analysis of AI trends.",
            emerging_trends=["Multimodal AI"],
            conclusion="AI is reshaping technology.",
            sources=[
                SourceReference(identifier="[S1]", title="AI Trends Article", url="https://example.com/ai-trends")
            ],
            generated_at=datetime.now(timezone.utc),
        )

        # 2. Save report 1
        run1_id = repo.save_research_run(report1, report_path="reports/ai-trends-1.md")
        print(f"Run 1 saved with ID: {run1_id}")

        # 3. Retrieve report 1
        retrieved1 = repo.get_latest_research("Artificial Intelligence Trends")
        assert retrieved1 is not None
        print(f"Retrieved Topic    : {retrieved1.topic}")
        print(f"Retrieved Timestamp: {retrieved1.generated_at}")
        print(f"Retrieved Sources  : {len(retrieved1.sources)} -> {[s.url for s in retrieved1.sources]}")
        print()

        # 4. Create sample report 2 with new findings & new source
        report2 = ResearchReport(
            topic="Artificial Intelligence Trends",
            executive_summary="AI hardware and quantum AI are making breakthroughs.",
            key_findings=[
                "AI agents reason and execute tasks.",
                "Brain-inspired quantum chips running near absolute zero created.",
            ],
            detailed_analysis="Updated analysis.",
            emerging_trends=["Quantum AI integration"],
            conclusion="AI hardware is advancing rapidly.",
            sources=[
                SourceReference(identifier="[S1]", title="AI Trends Article", url="https://example.com/ai-trends"),
                SourceReference(identifier="[S2]", title="Quantum AI News", url="https://example.com/quantum-ai"),
            ],
            generated_at=datetime.now(timezone.utc),
        )

        # 5. Compare report 2 against report 1
        comparator = ResearchComparator()
        whats_new = comparator.compare(current=report2, previous=retrieved1)

        print("What's New Comparison Results:")
        print(f"  Has Previous Research : {whats_new.has_previous_research}")
        print(f"  New Findings Count    : {len(whats_new.new_findings)}")
        print(f"  New Findings          : {whats_new.new_findings}")
        print(f"  Unchanged Findings    : {whats_new.unchanged_findings}")
        print(f"  New Sources Count     : {len(whats_new.new_sources)}")
        print(f"  New Sources           : {[s.url for s in whats_new.new_sources]}")
        print()

        assert len(whats_new.new_findings) >= 1
        assert "https://example.com/quantum-ai" in [s.url for s in whats_new.new_sources]
        assert "https://example.com/ai-trends" not in [s.url for s in whats_new.new_sources]

        print("Memory test: SUCCESS")
        logger.info("Memory test completed successfully.")


def handle_schedule_add(
    topic: str,
    frequency: str,
    time_str: Optional[str] = None,
    day_of_week: Optional[str] = None,
) -> None:
    """Add a new scheduled research job."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService, ScheduleValidationError

    try:
        service = SchedulerService()
        scheduled = service.add_schedule(
            topic=topic,
            frequency=frequency,
            time_str=time_str,
            day_of_week=day_of_week,
        )
        print("Schedule created successfully.")
        print(f"  ID        : {scheduled.id}")
        print(f"  Topic     : {scheduled.topic}")
        print(f"  Frequency : {scheduled.frequency.capitalize()}")
        if scheduled.day_of_week:
            print(f"  Day       : {scheduled.day_of_week.capitalize()}")
        print(f"  Time      : {scheduled.hour:02d}:{scheduled.minute:02d} ({service.timezone_name})")
        print(f"  Enabled   : {scheduled.enabled}")
        if scheduled.run_at:
            print(f"  Run At    : {scheduled.run_at}")
    except ScheduleValidationError as exc:
        print(f"Schedule validation error: {exc}")
        sys.exit(1)


def handle_schedule_test_job(topic: str = "Safe Scheduler Test", delay_seconds: int = 60) -> None:
    """Schedule a safe one-time test job to run in ~60 seconds."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    service = SchedulerService()
    sched = service.add_test_job(topic=topic, delay_seconds=delay_seconds)
    print("One-time test job created successfully.")
    print(f"  ID        : {sched.id}")
    print(f"  Topic     : {sched.topic}")
    print(f"  Frequency : Once (~{delay_seconds}s delay)")
    print(f"  Time      : {sched.hour:02d}:{sched.minute:02d} ({service.timezone_name})")
    print(f"  Run At    : {sched.run_at}")
    print(f"  Enabled   : {sched.enabled}")
    print("\nNote: When running with FastAPI ('python run.py --api') or daemon ('python run.py --scheduler'), this job will trigger automatically.")


def handle_schedule_list() -> None:
    """Display all scheduled research jobs."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    service = SchedulerService()
    schedules = service.list_schedules()

    width = 60
    print("=" * width)
    print("AutoResearch AI — Scheduled Research Jobs".center(width))
    print("=" * width)
    print()

    if not schedules:
        print("No scheduled research jobs found.")
        return

    for i, s in enumerate(schedules, 1):
        freq_str = s["frequency"].capitalize()
        if s["day_of_week"]:
            freq_str += f" ({s['day_of_week'].capitalize()})"
        time_formatted = f"{s['hour']:02d}:{s['minute']:02d}"

        print(f"[{i}] ID        : {s['id']}")
        print(f"    Topic     : {s['topic']}")
        print(f"    Frequency : {freq_str}")
        print(f"    Time      : {time_formatted}")
        print(f"    Enabled   : {s['enabled']}")
        print(f"    Next Run  : {s.get('next_run', 'N/A')}")
        print()


def handle_schedule_remove(job_id: str) -> None:
    """Remove a scheduled research job by ID."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    service = SchedulerService()
    removed = service.remove_schedule(job_id)
    if removed:
        print(f"Schedule {job_id!r} removed successfully.")
    else:
        print(f"Schedule {job_id!r} not found.")


def handle_schedule_enable(job_id: str) -> None:
    """Enable a scheduled research job by ID."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    service = SchedulerService()
    updated = service.enable_schedule(job_id)
    if updated:
        print(f"Schedule {job_id!r} enabled.")
    else:
        print(f"Schedule {job_id!r} not found.")


def handle_schedule_disable(job_id: str) -> None:
    """Disable a scheduled research job by ID."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    service = SchedulerService()
    updated = service.disable_schedule(job_id)
    if updated:
        print(f"Schedule {job_id!r} disabled.")
    else:
        print(f"Schedule {job_id!r} not found.")


def handle_schedule_run(job_id: str) -> None:
    """Immediately trigger a scheduled research job by ID."""
    configure_logging()
    from app.scheduler.scheduler import SchedulerService

    print(f"Triggering scheduled job: {job_id!r}...")
    service = SchedulerService()
    try:
        service.run_job_now(job_id)
        print("Manual schedule trigger completed successfully.")
    except Exception as exc:
        print(f"Manual trigger failed: {exc}")
        sys.exit(1)


def run_scheduler_daemon() -> None:
    """Run the APScheduler background daemon continuously."""
    import time
    configure_logging()
    logger = logging.getLogger(__name__)

    from app.scheduler.scheduler import SchedulerService

    width = 40
    print("=" * width)
    print("AutoResearch AI Scheduler".center(width))
    print("=" * width)
    print()

    service = SchedulerService()
    service.start()

    schedules = service.list_schedules(enabled_only=True)
    print(f"Loaded {len(schedules)} scheduled research job(s).\n")

    for i, s in enumerate(schedules, 1):
        freq_str = s["frequency"].capitalize()
        if s["day_of_week"]:
            freq_str += f" ({s['day_of_week'].capitalize()})"
        time_formatted = f"{s['hour']:02d}:{s['minute']:02d}"

        print(f"[{i}] {s['topic']}")
        print(f"    Frequency : {freq_str}")
        print(f"    Time      : {time_formatted}")
        print(f"    Next run  : {s.get('next_run', 'N/A')}")
        print()

    print("Scheduler running...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        service.stop()
        print("Scheduler stopped.")


def test_scheduler_pipeline() -> None:
    """
    Manually test APScheduler and database schedule persistence without real API calls.
    """
    import tempfile
    from pathlib import Path
    from app.memory.repository import ResearchRepository
    from app.scheduler.scheduler import SchedulerService

    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()
    print("Testing APScheduler & Schedule Persistence...")
    print()

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_scheduler.db"
        repo = ResearchRepository(db_path=test_db)
        service = SchedulerService(repository=repo)

        # 1. Create temporary schedule
        sched = service.add_schedule(
            topic="AI Safety Research",
            frequency="daily",
            time_str="09:00",
        )
        print(f"Added test schedule: {sched.id}")

        # 2. Add to APScheduler
        service.start()
        print("Scheduler started.")

        # 3. Display job
        schedules = service.list_schedules()
        print(f"List schedules returned {len(schedules)} job(s):")
        for s in schedules:
            print(f"  - ID: {s['id']}, Topic: {s['topic']}, Next Run: {s['next_run']}")

        # 4. Verify schedule
        assert len(schedules) == 1
        assert schedules[0]["id"] == sched.id
        assert schedules[0]["topic"] == "AI Safety Research"

        # 5. Remove job
        removed = service.remove_schedule(sched.id)
        assert removed is True
        print(f"Removed schedule: {sched.id}")

        # 6. Confirm scheduler shutdown
        service.stop()
        print("Scheduler stopped successfully.")
        print()
        print("Scheduler test: SUCCESS")
        logger.info("Scheduler test completed successfully.")


def start_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start the FastAPI backend server using uvicorn.

    Called when started with --api flag.
    """
    import uvicorn
    configure_logging()
    logger = logging.getLogger(__name__)

    print_banner()
    logger.info("Starting AutoResearch AI FastAPI server on http://%s:%d", host, port)
    uvicorn.run("app.api.main:app", host=host, port=port, reload=False)



