from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ai_eval.models import RUN_EXPORT_FORMAT_VERSION, CaseOutcome, EvalCase, EvalRunSummary, RunExport


class RunStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def default(cls, project_root: Path) -> RunStore:
        return cls(project_root / ".ai-eval" / "runs.db")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  suite_name TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  adapter TEXT NOT NULL,
                  model TEXT,
                  config_hash TEXT NOT NULL,
                  git_sha TEXT,
                  total INTEGER NOT NULL,
                  passed INTEGER NOT NULL,
                  pass_rate REAL NOT NULL,
                  mean_score REAL NOT NULL,
                  metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                  run_id TEXT NOT NULL,
                  case_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY (run_id, case_id),
                  FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS cases (
                  run_id TEXT NOT NULL,
                  case_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY (run_id, case_id)
                );
                """
            )

    def save_run(
        self,
        summary: EvalRunSummary,
        outcomes: list[CaseOutcome],
        cases: list[EvalCase],
    ) -> None:
        case_by_id = {c.id: c for c in cases}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, suite_name, created_at, adapter, model,
                  config_hash, git_sha, total, passed, pass_rate, mean_score, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.run_id,
                    summary.suite_name,
                    summary.created_at.isoformat(),
                    summary.adapter,
                    summary.model,
                    summary.config_hash,
                    summary.git_sha,
                    summary.total,
                    summary.passed,
                    summary.pass_rate,
                    summary.mean_score,
                    json.dumps(summary.metadata),
                ),
            )
            for outcome in outcomes:
                conn.execute(
                    "INSERT INTO outcomes (run_id, case_id, payload_json) VALUES (?, ?, ?)",
                    (summary.run_id, outcome.case_id, outcome.model_dump_json()),
                )
                case = case_by_id.get(outcome.case_id)
                if case:
                    conn.execute(
                        "INSERT INTO cases (run_id, case_id, payload_json) VALUES (?, ?, ?)",
                        (summary.run_id, case.id, case.model_dump_json()),
                    )
            conn.commit()

    def list_runs(self, limit: int = 20) -> list[EvalRunSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_run(self, run_id: str) -> EvalRunSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return self._row_to_summary(row)

    def get_outcomes(self, run_id: str) -> list[CaseOutcome]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM outcomes WHERE run_id = ? ORDER BY case_id",
                (run_id,),
            ).fetchall()
        return [CaseOutcome.model_validate_json(r["payload_json"]) for r in rows]

    def get_cases(self, run_id: str) -> dict[str, EvalCase]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, payload_json FROM cases WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {r["case_id"]: EvalCase.model_validate_json(r["payload_json"]) for r in rows}

    def export_run(self, run_id: str) -> RunExport:
        summary = self.get_run(run_id)
        if not summary:
            raise ValueError(f"Run not found: {run_id}")
        cases = self.get_cases(run_id)
        return RunExport(
            summary=summary,
            outcomes=self.get_outcomes(run_id),
            cases=[cases[k] for k in sorted(cases)],
        )

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> EvalRunSummary:
        from datetime import datetime

        return EvalRunSummary(
            run_id=row["run_id"],
            suite_name=row["suite_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            adapter=row["adapter"],
            model=row["model"],
            config_hash=row["config_hash"],
            git_sha=row["git_sha"],
            total=row["total"],
            passed=row["passed"],
            pass_rate=row["pass_rate"],
            mean_score=row["mean_score"],
            metadata=json.loads(row["metadata_json"]),
        )


def save_run_file(export: RunExport, path: Path) -> Path:
    """Write an exported run as JSON (creating parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run_file(path: Path) -> RunExport:
    """Read a run previously written by `save_run_file`."""
    if not path.is_file():
        raise ValueError(f"Run file not found: {path}")
    try:
        export = RunExport.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"Invalid run file {path}: {e}") from e
    if export.format_version > RUN_EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Run file {path} has format_version {export.format_version}; "
            f"this ai-eval supports up to {RUN_EXPORT_FORMAT_VERSION}"
        )
    return export
