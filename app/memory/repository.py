"""
app/memory/repository.py
------------------------
Repository pattern for reading and writing research runs in SQLite.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import config
from app.memory.database import get_db_connection, init_db
from app.memory.models import ResearchJobRecord, normalize_topic, utc_now_iso
from app.research.report import ResearchReport, SourceReference


class ResearchRepository:
    """
    Repository abstraction for storing and retrieving research runs.

    All SQL interactions are encapsulated within this class.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = db_path if db_path is not None else config.DATABASE_PATH
        init_db(self.db_path)

    def save_research_run(
        self, report: ResearchReport, report_path: Optional[str | Path] = None
    ) -> int:
        """
        Save a completed ResearchReport and its sources/findings into the database.

        Args:
            report: The ResearchReport dataclass to persist.
            report_path: Optional file path where the report markdown was saved.

        Returns:
            The integer database ID of the inserted research run.
        """
        path_str = str(report_path) if report_path else ""
        norm_topic = normalize_topic(report.topic)
        iso_dt = report.generated_at.isoformat()

        conn = get_db_connection(self.db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO research_runs (
                    topic, topic_normalized, generated_at, report_path,
                    executive_summary, detailed_analysis, conclusion, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.topic,
                    norm_topic,
                    iso_dt,
                    path_str,
                    report.executive_summary,
                    report.detailed_analysis,
                    report.conclusion,
                    report.error,
                ),
            )
            run_id = cursor.lastrowid

            # Save sources
            for src in report.sources:
                cursor.execute(
                    """
                    INSERT INTO research_sources (
                        research_run_id, source_identifier, title, url
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (run_id, src.identifier, src.title, src.url),
                )

            # Save key findings
            for finding in report.key_findings:
                cursor.execute(
                    """
                    INSERT INTO research_findings (
                        research_run_id, finding_type, finding_text
                    ) VALUES (?, ?, ?)
                    """,
                    (run_id, "key_finding", finding),
                )

            # Save emerging trends
            for trend in report.emerging_trends:
                cursor.execute(
                    """
                    INSERT INTO research_findings (
                        research_run_id, finding_type, finding_text
                    ) VALUES (?, ?, ?)
                    """,
                    (run_id, "emerging_trend", trend),
                )

        conn.close()
        return run_id

    def get_research_run(self, run_id: int) -> Optional[ResearchReport]:
        """Retrieve a specific ResearchReport by its database run ID."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        report = self._row_to_report(conn, row)
        conn.close()
        return report

    def get_latest_research(self, topic: str) -> Optional[ResearchReport]:
        """
        Retrieve the most recent ResearchReport for a given topic.

        Matching is case-insensitive and whitespace-normalized.
        """
        norm_topic = normalize_topic(topic)
        if not norm_topic:
            return None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM research_runs
            WHERE topic_normalized = ?
            ORDER BY generated_at DESC, id DESC
            LIMIT 1
            """,
            (norm_topic,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        report = self._row_to_report(conn, row)
        conn.close()
        return report

    def get_latest_research_run_id(self, topic: str) -> Optional[int]:
        """
        Retrieve the latest database run ID (integer) for a given topic.
        """
        norm_topic = normalize_topic(topic)
        if not norm_topic:
            return None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM research_runs
            WHERE topic_normalized = ?
            ORDER BY generated_at DESC, id DESC
            LIMIT 1
            """,
            (norm_topic,),
        )
        row = cursor.fetchone()
        conn.close()
        return row["id"] if row else None

    def get_research_run_id_since(self, topic: str, since_iso: str) -> Optional[int]:
        """
        Retrieve the run ID for the most recent research_run for `topic` whose
        generated_at is >= `since_iso`.  Used to pin a job to the run created
        during *this* execution rather than a prior run for the same topic.
        """
        norm_topic = normalize_topic(topic)
        if not norm_topic or not since_iso:
            return None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM research_runs
            WHERE topic_normalized = ? AND generated_at >= ?
            ORDER BY generated_at DESC, id DESC
            LIMIT 1
            """,
            (norm_topic, since_iso),
        )
        row = cursor.fetchone()
        conn.close()
        return row["id"] if row else None

    def get_research_history(self, topic: str) -> list[ResearchReport]:
        """
        Retrieve all historical ResearchReports for a topic ordered newest first.
        """
        norm_topic = normalize_topic(topic)
        if not norm_topic:
            return []

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM research_runs
            WHERE topic_normalized = ?
            ORDER BY generated_at DESC, id DESC
            """,
            (norm_topic,),
        )
        rows = cursor.fetchall()
        reports = [self._row_to_report(conn, row) for row in rows]
        conn.close()
        return reports

    def list_research_runs(self, limit: int = 20) -> list[dict]:
        """List summary info for recent research runs."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, topic, generated_at, report_path, error
            FROM research_runs
            ORDER BY generated_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result

    # ── Private helper ────────────────────────────────────────────────────────

    def _row_to_report(self, conn, row) -> ResearchReport:
        run_id = row["id"]
        cursor = conn.cursor()

        # Fetch sources
        cursor.execute(
            "SELECT source_identifier, title, url FROM research_sources WHERE research_run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        src_rows = cursor.fetchall()
        sources = [
            SourceReference(
                identifier=s["source_identifier"], title=s["title"], url=s["url"]
            )
            for s in src_rows
        ]

        # Fetch findings
        cursor.execute(
            "SELECT finding_type, finding_text FROM research_findings WHERE research_run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        finding_rows = cursor.fetchall()
        key_findings = [
            f["finding_text"]
            for f in finding_rows
            if f["finding_type"] == "key_finding"
        ]
        emerging_trends = [
            f["finding_text"]
            for f in finding_rows
            if f["finding_type"] == "emerging_trend"
        ]

        dt_str = row["generated_at"]
        try:
            gen_at = datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            gen_at = datetime.now(timezone.utc)

        return ResearchReport(
            topic=row["topic"],
            executive_summary=row["executive_summary"] or "",
            key_findings=key_findings,
            detailed_analysis=row["detailed_analysis"] or "",
            emerging_trends=emerging_trends,
            conclusion=row["conclusion"] or "",
            sources=sources,
            generated_at=gen_at,
            error=row["error"],
        )

    # ── Schedule persistence methods ─────────────────────────────────────────

    def save_schedule(self, scheduled) -> str:
        """
        Save or overwrite a ScheduledResearch object in the database.
        """
        conn = get_db_connection(self.db_path)
        iso_dt = scheduled.created_at.isoformat()
        enabled_int = 1 if scheduled.enabled else 0

        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scheduled_research (
                    id, topic, frequency, day_of_week, hour, minute, enabled, created_at, run_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    topic = excluded.topic,
                    frequency = excluded.frequency,
                    day_of_week = excluded.day_of_week,
                    hour = excluded.hour,
                    minute = excluded.minute,
                    enabled = excluded.enabled,
                    created_at = excluded.created_at,
                    run_at = excluded.run_at
                """,
                (
                    scheduled.id,
                    scheduled.topic,
                    scheduled.frequency,
                    scheduled.day_of_week,
                    scheduled.hour,
                    scheduled.minute,
                    enabled_int,
                    iso_dt,
                    getattr(scheduled, "run_at", None),
                ),
            )
        conn.close()
        return scheduled.id

    def get_schedule(self, job_id: str):
        """Retrieve a ScheduledResearch object by job_id."""
        from app.scheduler.models import ScheduledResearch

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scheduled_research WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_schedule(row)

    def list_schedules(self, enabled_only: bool = False):
        """Retrieve list of ScheduledResearch objects."""
        from app.scheduler.models import ScheduledResearch

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        if enabled_only:
            cursor.execute("SELECT * FROM scheduled_research WHERE enabled = 1 ORDER BY created_at ASC")
        else:
            cursor.execute("SELECT * FROM scheduled_research ORDER BY created_at ASC")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_schedule(row) for row in rows]

    def update_schedule(self, job_id: str, enabled: Optional[bool] = None) -> bool:
        """Update schedule state (e.g. enable/disable)."""
        conn = get_db_connection(self.db_path)
        with conn:
            cursor = conn.cursor()
            if enabled is not None:
                enabled_int = 1 if enabled else 0
                cursor.execute(
                    "UPDATE scheduled_research SET enabled = ? WHERE id = ?",
                    (enabled_int, job_id),
                )
            affected = cursor.rowcount
        conn.close()
        return affected > 0

    def delete_schedule(self, job_id: str) -> bool:
        """Delete a schedule by job_id from SQLite."""
        conn = get_db_connection(self.db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scheduled_research WHERE id = ?", (job_id,))
            affected = cursor.rowcount
        conn.close()
        return affected > 0

    def _row_to_schedule(self, row):
        from app.scheduler.models import ScheduledResearch

        dt_str = row["created_at"]
        try:
            created_at = datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            created_at = datetime.now(timezone.utc)

        run_at = row["run_at"] if "run_at" in row.keys() else None

        return ScheduledResearch(
            id=row["id"],
            topic=row["topic"],
            frequency=row["frequency"],
            day_of_week=row["day_of_week"],
            hour=row["hour"],
            minute=row["minute"],
            enabled=bool(row["enabled"]),
            created_at=created_at,
            run_at=run_at,
        )

    # ── Job lifecycle persistence methods ─────────────────────────────────────

    def create_job(self, job: ResearchJobRecord) -> str:
        """
        Persist a new ResearchJobRecord in the database.
        """
        conn = get_db_connection(self.db_path)
        norm_topic = normalize_topic(job.topic)
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO research_jobs (
                    id, topic, topic_normalized, trigger_type, status,
                    created_at, started_at, completed_at, updated_at,
                    schedule_id, research_run_id, report_path, whats_new_path,
                    error, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.topic,
                    norm_topic,
                    job.trigger_type,
                    job.status,
                    job.created_at,
                    job.started_at,
                    job.completed_at,
                    job.updated_at,
                    job.schedule_id,
                    job.research_run_id,
                    job.report_path,
                    job.whats_new_path,
                    job.error,
                    job.duration_seconds,
                ),
            )
        conn.close()
        return job.id

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        research_run_id: Optional[int] = None,
        report_path: Optional[str] = None,
        whats_new_path: Optional[str] = None,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> bool:
        """
        Update an existing ResearchJobRecord's status, timestamps, paths, or error.
        Automatically updates updated_at.
        """
        conn = get_db_connection(self.db_path)
        updated_at = utc_now_iso()

        fields = ["updated_at = ?"]
        params = [updated_at]

        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if started_at is not None:
            fields.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            params.append(completed_at)
        if research_run_id is not None:
            fields.append("research_run_id = ?")
            params.append(research_run_id)
        if report_path is not None:
            fields.append("report_path = ?")
            params.append(report_path)
        if whats_new_path is not None:
            fields.append("whats_new_path = ?")
            params.append(whats_new_path)
        if error is not None:
            fields.append("error = ?")
            params.append(error)
        if duration_seconds is not None:
            fields.append("duration_seconds = ?")
            params.append(duration_seconds)

        params.append(job_id)
        sql = f"UPDATE research_jobs SET {', '.join(fields)} WHERE id = ?"

        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            affected = cursor.rowcount
        conn.close()
        return affected > 0

    def get_job(self, job_id: str) -> Optional[ResearchJobRecord]:
        """Retrieve a ResearchJobRecord by job_id."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM research_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(
        self,
        status: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 50,
    ) -> list[ResearchJobRecord]:
        """
        List research jobs with optional status and topic filters, ordered newest first.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if topic:
            conditions.append("topic_normalized = ?")
            params.append(normalize_topic(topic))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM research_jobs
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_job(row) for row in rows]

    def has_active_job_for_topic(self, topic: str) -> bool:
        """
        Check if there is currently a queued or running job for the normalized topic.
        """
        norm_topic = normalize_topic(topic)
        if not norm_topic:
            return False
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) AS cnt FROM research_jobs
            WHERE topic_normalized = ? AND status IN ('queued', 'running')
            """,
            (norm_topic,),
        )
        row = cursor.fetchone()
        conn.close()
        return bool(row and row["cnt"] > 0)

    def has_active_job_for_schedule(self, schedule_id: str) -> bool:
        """
        Check if there is currently a queued or running job for the schedule ID.
        """
        if not schedule_id:
            return False
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) AS cnt FROM research_jobs
            WHERE schedule_id = ? AND status IN ('queued', 'running')
            """,
            (schedule_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return bool(row and row["cnt"] > 0)

    def _row_to_job(self, row) -> ResearchJobRecord:
        return ResearchJobRecord(
            id=row["id"],
            topic=row["topic"],
            trigger_type=row["trigger_type"],
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=row["updated_at"],
            schedule_id=row["schedule_id"],
            research_run_id=row["research_run_id"],
            report_path=row["report_path"],
            whats_new_path=row["whats_new_path"],
            error=row["error"],
            duration_seconds=row["duration_seconds"],
        )

