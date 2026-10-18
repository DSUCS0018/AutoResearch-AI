"""
run.py
------
Root-level entry point for AutoResearch AI.

Usage:
    python run.py                                              # Normal startup
    python run.py --test-llm                                   # Gemini connection test
    python run.py --test-search                                # Tavily search test
    python run.py --test-rag                                   # Local RAG pipeline test
    python run.py --test-memory                                # SQLite memory & comparator test
    python run.py --test-scheduler                             # APScheduler persistence test
    python run.py --research "topic"                           # Full research agent
    python run.py --history "topic"                            # View topic research history
    python run.py --whats-new "topic"                          # Fresh run vs previous research
    python run.py --schedule-add "topic" --daily 09:00         # Add daily schedule
    python run.py --schedule-add "topic" --weekly monday 09:00  # Add weekly schedule
    python run.py --schedule-list                              # List all schedules
    python run.py --schedule-remove <job_id>                   # Remove a schedule
    python run.py --schedule-enable <job_id>                   # Enable a schedule
    python run.py --schedule-disable <job_id>                  # Disable a schedule
    python run.py --schedule-run <job_id>                      # Trigger a scheduled job immediately
    python run.py --scheduler                                  # Start continuous scheduler daemon
"""

import sys
from app.main import (
    start,
    start_api,
    test_llm_connection,
    test_search_pipeline,
    test_rag_pipeline,
    test_memory_pipeline,
    test_scheduler_pipeline,
    run_research_agent,
    display_research_history,
    run_whats_new,
    handle_schedule_add,
    handle_schedule_test_job,
    handle_schedule_list,
    handle_schedule_remove,
    handle_schedule_enable,
    handle_schedule_disable,
    handle_schedule_run,
    run_scheduler_daemon,
)


def _get_arg_value(flag: str) -> str:
    """Extract an argument value following a flag from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip()
    return ""


def main() -> None:
    """Route execution based on command-line arguments."""
    if "--api" in sys.argv:
        start_api()
    elif "--test-llm" in sys.argv:

        test_llm_connection()
    elif "--test-search" in sys.argv:
        test_search_pipeline()
    elif "--test-rag" in sys.argv:
        test_rag_pipeline()
    elif "--test-memory" in sys.argv:
        test_memory_pipeline()
    elif "--test-scheduler" in sys.argv:
        test_scheduler_pipeline()
    elif "--schedule-test-job" in sys.argv:
        topic = _get_arg_value("--schedule-test-job") or "Safe Automated Scheduler Test"
        handle_schedule_test_job(topic=topic, delay_seconds=60)
    elif "--schedule-add" in sys.argv:
        topic = _get_arg_value("--schedule-add")
        if not topic:
            print("Usage: python run.py --schedule-add \"topic\" --daily 09:00 OR --weekly monday 09:00 OR --once")
            sys.exit(1)
        if "--daily" in sys.argv:
            time_str = _get_arg_value("--daily")
            handle_schedule_add(topic, frequency="daily", time_str=time_str)
        elif "--weekly" in sys.argv:
            idx = sys.argv.index("--weekly")
            if idx + 2 < len(sys.argv):
                day_of_week = sys.argv[idx + 1]
                time_str = sys.argv[idx + 2]
                handle_schedule_add(
                    topic, frequency="weekly", time_str=time_str, day_of_week=day_of_week
                )
            else:
                print("Usage: python run.py --schedule-add \"topic\" --weekly monday 09:00")
                sys.exit(1)
        elif "--once" in sys.argv:
            handle_schedule_test_job(topic=topic, delay_seconds=60)
        else:
            print("Usage: Must specify --daily HH:MM, --weekly DAY HH:MM, or --once")
            sys.exit(1)
    elif "--schedule-list" in sys.argv:
        handle_schedule_list()
    elif "--schedule-remove" in sys.argv:
        job_id = _get_arg_value("--schedule-remove")
        if not job_id:
            print("Usage: python run.py --schedule-remove <job_id>")
            sys.exit(1)
        handle_schedule_remove(job_id)
    elif "--schedule-enable" in sys.argv:
        job_id = _get_arg_value("--schedule-enable")
        if not job_id:
            print("Usage: python run.py --schedule-enable <job_id>")
            sys.exit(1)
        handle_schedule_enable(job_id)
    elif "--schedule-disable" in sys.argv:
        job_id = _get_arg_value("--schedule-disable")
        if not job_id:
            print("Usage: python run.py --schedule-disable <job_id>")
            sys.exit(1)
        handle_schedule_disable(job_id)
    elif "--schedule-run" in sys.argv:
        job_id = _get_arg_value("--schedule-run")
        if not job_id:
            print("Usage: python run.py --schedule-run <job_id>")
            sys.exit(1)
        handle_schedule_run(job_id)
    elif "--scheduler" in sys.argv:
        run_scheduler_daemon()
    elif "--research" in sys.argv:
        topic = _get_arg_value("--research")
        if not topic:
            print("Usage: python run.py --research \"your research topic\"")
            sys.exit(1)
        run_research_agent(topic)
    elif "--history" in sys.argv:
        topic = _get_arg_value("--history")
        if not topic:
            print("Usage: python run.py --history \"your research topic\"")
            sys.exit(1)
        display_research_history(topic)
    elif "--whats-new" in sys.argv:
        topic = _get_arg_value("--whats-new")
        if not topic:
            print("Usage: python run.py --whats-new \"your research topic\"")
            sys.exit(1)
        run_whats_new(topic)
    else:
        start()


if __name__ == "__main__":
    main()
