"""
app/ui/streamlit_app.py
-----------------------
Streamlit web interface for AutoResearch AI.
Communicates with the FastAPI backend endpoints.
"""

import time
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AutoResearch AI",
    page_icon="🤖",
    layout="wide",
)

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Status badge helpers
_STATUS_COLOR = {
    "queued":    ("🟡", "orange"),
    "running":   ("🔵", "blue"),
    "completed": ("🟢", "green"),
    "failed":    ("🔴", "red"),
}


def _status_badge(s: str) -> str:
    icon, _ = _STATUS_COLOR.get(s, ("⚪", "gray"))
    return f"{icon} **{s.upper()}**"


def _call_api(method: str, endpoint: str, json_data: dict = None, timeout: int = 120):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            res = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            res = requests.post(url, json=json_data, timeout=timeout)
        elif method.upper() == "PATCH":
            res = requests.patch(url, json=json_data, timeout=timeout)
        elif method.upper() == "DELETE":
            res = requests.delete(url, timeout=timeout)
        else:
            return None, f"Unsupported HTTP method: {method}"

        if res.status_code >= 400:
            try:
                err_detail = res.json().get("detail", res.text)
            except Exception:
                err_detail = res.text
            return None, f"API Error ({res.status_code}): {err_detail}"

        return res.json(), None

    except requests.exceptions.ConnectionError:
        return None, (
            f"FastAPI Backend Unavailable at {API_BASE_URL}.\n"
            "Please start the backend server first: python run.py --api"
        )
    except Exception as exc:
        return None, f"Request failed: {exc}"


# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.title("🤖 AutoResearch AI")
st.sidebar.markdown("Automated AI-powered research and knowledge tracking.")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🔍 Research", "📋 Jobs", "📜 History", "🆕 What's New", "⏰ Schedules"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: `{API_BASE_URL}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Research
# ══════════════════════════════════════════════════════════════════════════════
if nav_choice == "🔍 Research":
    st.title("AutoResearch AI — Research Agent")
    st.caption("Automated AI-powered research, source discovery, and report generation.")

    topic_input = st.text_input(
        "Research Topic",
        value="latest developments in artificial intelligence",
        help="Enter a topic for the research agent to investigate.",
    )

    col_sync, col_async = st.columns(2)
    with col_sync:
        btn_sync = st.button("▶ Run Research (wait here)", type="primary",
                             help="Blocks until complete, then shows the full report.")
    with col_async:
        btn_async = st.button("⚡ Submit Job (background)",
                              help="Submits via POST /api/jobs and returns immediately. "
                                   "Track progress in the Jobs tab.")

    if btn_sync:
        if not topic_input.strip():
            st.warning("Please enter a research topic.")
        else:
            with st.spinner("Searching Tavily, ingesting into RAG, and generating report with Gemini..."):
                data, err = _call_api("POST", "/api/research", {"topic": topic_input})

            if err:
                st.error(err)
            else:
                st.success("Research completed successfully!")
                st.subheader("Executive Summary")
                st.write(data.get("executive_summary", "_None_"))

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Key Findings")
                    for finding in data.get("key_findings", []):
                        st.markdown(f"- {finding}")
                with col2:
                    st.subheader("Emerging Trends")
                    for trend in data.get("emerging_trends", []):
                        st.markdown(f"- {trend}")

                st.subheader("Detailed Analysis")
                st.write(data.get("detailed_analysis", "_None_"))

                st.subheader("Conclusion")
                st.write(data.get("conclusion", "_None_"))

                st.subheader("Sources")
                for src in data.get("sources", []):
                    st.markdown(f"**{src['identifier']}** [{src['title']}]({src['url']})")

                st.info(f"Report saved locally to: `{data.get('report_path')}`")

    if btn_async:
        if not topic_input.strip():
            st.warning("Please enter a research topic.")
        else:
            data, err = _call_api("POST", "/api/jobs", {"topic": topic_input}, timeout=15)
            if err:
                st.error(err)
            else:
                job_id = data.get("id", "")
                st.success(
                    f"✅ Job **`{job_id}`** submitted! "
                    f"Status: {_status_badge(data.get('status', 'queued'))}  \n"
                    "Switch to the **Jobs** tab to track progress."
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Jobs (live tracking dashboard)
# ══════════════════════════════════════════════════════════════════════════════
elif nav_choice == "📋 Jobs":
    st.title("Research Jobs — Live Tracker")
    st.caption("Real-time job status, reports, and execution history. Polls the backend for status.")

    # ── Filter controls ───────────────────────────────────────────────────────
    col_sf, col_tf, col_lim, col_refresh = st.columns([2, 3, 1, 1])
    with col_sf:
        status_filter = st.selectbox(
            "Status filter",
            ["all", "queued", "running", "completed", "failed"],
            index=0,
        )
    with col_tf:
        topic_filter = st.text_input("Topic filter", value="", placeholder="(any)")
    with col_lim:
        limit = st.number_input("Limit", min_value=5, max_value=200, value=20, step=5)
    with col_refresh:
        st.write("")
        manual_refresh = st.button("🔄 Refresh", type="secondary")

    params = f"?limit={limit}"
    if status_filter != "all":
        params += f"&status={status_filter}"
    if topic_filter.strip():
        params += f"&topic={topic_filter.strip()}"

    jobs_data, jobs_err = _call_api("GET", f"/api/jobs{params}", timeout=15)

    if jobs_err:
        st.error(jobs_err)
    else:
        jobs = jobs_data.get("jobs", [])
        total = jobs_data.get("total", 0)

        # Detect any active jobs → trigger auto-refresh
        active_jobs = [j for j in jobs if j.get("status") in ("queued", "running")]
        if active_jobs:
            st.info(
                f"⏳ {len(active_jobs)} job(s) are still active — page will auto-refresh every 5 seconds."
            )
            time.sleep(5)
            st.rerun()

        st.caption(f"Showing {len(jobs)} of {total} job(s)")

        if not jobs:
            st.info("No jobs found. Submit a research job from the Research tab.")
        else:
            for j in jobs:
                jstatus = j.get("status", "unknown")
                icon, color = _STATUS_COLOR.get(jstatus, ("⚪", "gray"))
                job_id = j.get("id", "")

                with st.expander(
                    f"{icon} {jstatus.upper()}  |  {j.get('topic', '')}  |  `{job_id}`",
                    expanded=(jstatus in ("running", "queued")),
                ):
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.markdown(f"**Job ID:** `{job_id}`")
                        st.markdown(f"**Topic:** {j.get('topic', '')}")
                        st.markdown(f"**Trigger:** {j.get('trigger_type', '')}")
                        st.markdown(f"**Status:** {_status_badge(jstatus)}")
                        if j.get("schedule_id"):
                            st.markdown(f"**Schedule ID:** `{j['schedule_id']}`")
                        if j.get("research_run_id"):
                            st.markdown(f"**Research Run ID:** `{j['research_run_id']}`")

                    with col_right:
                        st.markdown(f"**Created:** {j.get('created_at_display') or j.get('created_at', '')}")
                        if j.get("started_at_display"):
                            st.markdown(f"**Started:** {j['started_at_display']}")
                        if j.get("completed_at_display"):
                            st.markdown(f"**Finished:** {j['completed_at_display']}")
                        if j.get("duration_seconds") is not None:
                            dur = j["duration_seconds"]
                            st.markdown(f"**Duration:** {dur:.1f}s")

                    if jstatus == "failed" and j.get("error"):
                        st.error(f"**Error:** {j['error']}")

                    if jstatus == "completed":
                        btn_rpt = st.button("📄 Load Reports", key=f"rpt_{job_id}")
                        if btn_rpt:
                            rpt_data, rpt_err = _call_api("GET", f"/api/jobs/{job_id}/reports", timeout=15)
                            if rpt_err:
                                st.error(rpt_err)
                            else:
                                if rpt_data.get("report_path"):
                                    st.markdown(f"📁 Report: `{rpt_data['report_path']}`")
                                if rpt_data.get("whats_new_path"):
                                    st.markdown(f"📁 What's New: `{rpt_data['whats_new_path']}`")

                                if rpt_data.get("report_content"):
                                    with st.expander("📑 Research Report", expanded=False):
                                        st.markdown(rpt_data["report_content"])
                                if rpt_data.get("whats_new_content"):
                                    with st.expander("🆕 What's New Report", expanded=False):
                                        st.markdown(rpt_data["whats_new_content"])

                    elif jstatus in ("queued", "running"):
                        st.info("Research is in progress — status will update automatically.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — History
# ══════════════════════════════════════════════════════════════════════════════
elif nav_choice == "📜 History":
    st.title("Research History")
    st.caption("View previous research runs stored in SQLite memory.")

    history_topic = st.text_input(
        "Search Topic History",
        value="latest developments in artificial intelligence",
    )

    if st.button("Search History"):
        if not history_topic.strip():
            st.warning("Please enter a topic.")
        else:
            endpoint = f"/api/history/{history_topic.strip()}"
            data, err = _call_api("GET", endpoint)

            if err:
                st.error(err)
            else:
                runs = data.get("runs", [])
                if not runs:
                    st.info("No previous research runs found for this topic.")
                else:
                    st.success(f"Found {len(runs)} previous research run(s).")
                    for i, run in enumerate(runs, 1):
                        with st.expander(f"Run #{i} — {run['generated_at']}"):
                            st.write(f"**Topic:** {run['topic']}")
                            st.write(f"**Sources Count:** {run['sources_count']}")
                            st.write(f"**Findings Count:** {run['findings_count']}")
                            st.write(f"**Executive Summary Preview:** {run['executive_summary_preview']}...")
                            st.caption(f"Report Path: `{run['report_path']}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — What's New
# ══════════════════════════════════════════════════════════════════════════════
elif nav_choice == "🆕 What's New":
    st.title("What's New? — Research Comparison")
    st.caption("Run fresh research and compare against previous baseline runs.")

    wn_topic = st.text_input(
        "Topic for Comparison",
        value="latest developments in artificial intelligence",
    )

    if st.button("Find What's New", type="primary"):
        if not wn_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Capturing baseline, executing research, and performing comparison..."):
                data, err = _call_api("POST", "/api/whats-new", {"topic": wn_topic})

            if err:
                st.error(err)
            else:
                st.success("Comparison completed successfully!")

                if not data.get("has_previous_research"):
                    st.info("No previous research run existed for this topic. This run establishes the baseline.")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🆕 New Findings")
                    new_f = data.get("new_findings", [])
                    if new_f:
                        for f in new_f:
                            st.markdown(f"- {f}")
                    else:
                        st.caption("_No new findings detected._")

                    st.subheader("🔄 Changed Findings")
                    ch_f = data.get("changed_findings", [])
                    if ch_f:
                        for f in ch_f:
                            st.markdown(f"- {f}")
                    else:
                        st.caption("_No changed findings detected._")

                with col2:
                    st.subheader("📚 New Sources")
                    new_s = data.get("new_sources", [])
                    if new_s:
                        for s in new_s:
                            st.markdown(f"**{s['identifier']}** [{s['title']}]({s['url']})")
                    else:
                        st.caption("_No new sources detected._")

                    st.subheader("➖ Already Known")
                    old_f = data.get("unchanged_findings", [])
                    if old_f:
                        for f in old_f:
                            st.markdown(f"- {f}")
                    else:
                        st.caption("_No matching prior findings._")

                st.info(f"What's New report saved to: `{data.get('report_path')}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Schedules
# ══════════════════════════════════════════════════════════════════════════════
elif nav_choice == "⏰ Schedules":
    st.title("Scheduled Research Jobs")
    st.caption("Manage automated daily, weekly, and one-time research schedules.")

    # ── Display Scheduler Engine Status & Timezone ────────────────────────────
    status_data, status_err = _call_api("GET", "/api/schedules/status", timeout=10)
    if not status_err and status_data:
        engine_str = "🟢 Active" if status_data.get("running") else "🔴 Stopped"
        tz_str = status_data.get("timezone", "N/A")
        offset_str = status_data.get("timezone_offset", "")
        time_str = status_data.get("current_time", "N/A")
        st.info(
            f"**Scheduler Engine:** {engine_str} &nbsp;|&nbsp; "
            f"**Timezone:** `{tz_str} ({offset_str})` &nbsp;|&nbsp; "
            f"**Scheduler Clock:** `{time_str}`"
        )

    # ── Quick 60s Test Job Verification Section ───────────────────────────────
    st.subheader("⚡ Quick Verification Test Job")
    st.markdown(
        "Schedule a safe, one-time research job that executes automatically in **approximately 60 seconds** "
        "so you can verify the background scheduler without waiting until tomorrow."
    )
    col_test_topic, col_test_btn = st.columns([3, 1])
    with col_test_topic:
        test_topic_input = st.text_input(
            "Test Topic",
            value="safe automated scheduler verification test",
            key="test_job_topic_input",
            label_visibility="collapsed",
        )
    with col_test_btn:
        btn_create_test = st.button("Schedule 60s Test Job", type="secondary")

    if btn_create_test:
        with st.spinner("Registering one-time test job..."):
            test_res, test_err = _call_api(
                "POST",
                "/api/schedules/test-job",
                {"topic": test_topic_input.strip() or "Safe Scheduler Test", "delay_seconds": 60},
            )
        if test_err:
            st.error(test_err)
        else:
            st.success(
                f"✅ Test job `{test_res['id']}` registered! "
                f"Next Run: **{test_res.get('next_run', 'in ~60s')}**. "
                "The research agent will execute automatically when the time arrives. "
                "Watch your terminal logs or refresh this page in 1 minute!"
            )
            st.rerun()

    st.markdown("---")

    # ── Display Existing Schedules ────────────────────────────────────────────
    st.subheader("Current Schedules")
    schedules, err = _call_api("GET", "/api/schedules", timeout=10)

    if err:
        st.error(err)
    elif not schedules:
        st.info("No scheduled research jobs configured.")
    else:
        for s in schedules:
            col_info, col_actions = st.columns([3, 2])
            with col_info:
                freq_val = s["frequency"].lower()
                if freq_val == "weekly" and s.get("day_of_week"):
                    freq_label = f"Weekly ({s['day_of_week'].capitalize()})"
                elif freq_val == "once":
                    freq_label = "One-Time (~60s test)"
                else:
                    freq_label = "Daily"

                st.write(f"**{s['topic']}** (`{s['id']}`)")
                time_formatted = f"{s['hour']:02d}:{s['minute']:02d}"
                st.caption(
                    f"Frequency: {freq_label} at {time_formatted} | "
                    f"Enabled: {s['enabled']} | Next Run: **{s['next_run']}**"
                )

            with col_actions:
                btn_run = st.button("Run Now", key=f"run_{s['id']}")
                btn_toggle = st.button("Disable" if s["enabled"] else "Enable", key=f"tog_{s['id']}")
                btn_del = st.button("Delete", key=f"del_{s['id']}")

                if btn_run:
                    _, r_err = _call_api("POST", f"/api/schedules/{s['id']}/run")
                    if r_err:
                        st.error(r_err)
                    else:
                        st.success(f"Job {s['id']} triggered. Track it in the **Jobs** tab.")
                        st.rerun()

                if btn_toggle:
                    endpoint = f"/api/schedules/{s['id']}/" + ("disable" if s["enabled"] else "enable")
                    _, t_err = _call_api("POST", endpoint)
                    if t_err:
                        st.error(t_err)
                    else:
                        st.rerun()

                if btn_del:
                    _, d_err = _call_api("DELETE", f"/api/schedules/{s['id']}")
                    if d_err:
                        st.error(d_err)
                    else:
                        st.rerun()

            st.markdown("---")

    # ── Add Schedule Form ─────────────────────────────────────────────────────
    st.subheader("Add New Schedule")
    with st.form("add_schedule_form"):
        new_topic = st.text_input("Topic", value="latest developments in artificial intelligence")
        col_freq, col_time = st.columns(2)

        with col_freq:
            new_freq = st.selectbox("Frequency", ["daily", "weekly", "once (runs in ~60s)"])
            new_day = None
            if new_freq == "weekly":
                new_day = st.selectbox(
                    "Day of Week",
                    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                )

        with col_time:
            new_time = st.text_input(
                "Time (HH:MM in scheduler timezone)",
                value="09:00",
                help="Specified in scheduler timezone. Ignored for 'once'.",
            )

        submitted = st.form_submit_button("Create Schedule", type="primary")

        if submitted:
            if not new_topic.strip():
                st.warning("Please enter a topic.")
            else:
                if "once" in new_freq:
                    payload = {"topic": new_topic.strip(), "delay_seconds": 60}
                    res, s_err = _call_api("POST", "/api/schedules/test-job", payload)
                else:
                    payload = {
                        "topic": new_topic.strip(),
                        "frequency": new_freq,
                        "time": new_time.strip(),
                        "day": new_day,
                    }
                    res, s_err = _call_api("POST", "/api/schedules", payload)
                if s_err:
                    st.error(s_err)
                else:
                    st.success(f"Schedule '{res['id']}' created! Next Run: {res.get('next_run', 'Pending Start')}")
                    st.rerun()
