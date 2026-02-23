"""Lightweight in-process background job manager for Streamlit sessions.

This module avoids Streamlit UI calls. It tracks job status in an in-memory
registry so the app can poll progress while work runs on daemon threads.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except Exception:  # pragma: no cover - fallback for non-Streamlit runtimes
    add_script_run_ctx = None


JobWorker = Callable[
    [
        Callable[[int, str], None],   # find progress callback
        Callable[[int, str], None],   # ai progress callback
        Callable[[list[dict[str, Any]], dict[str, Any]], None],  # stream callback
        Callable[[], bool],           # cancel checker
    ],
    None,
]


_JOB_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> float:
    return time.time()


def _job_copy(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    # keep payload small and thread-safe for UI reads
    out.pop("_thread", None)
    return out


def get_job(job_id: str) -> Dict[str, Any]:
    if not job_id:
        return {}
    with _JOB_LOCK:
        job = _JOBS.get(job_id, {})
        return _job_copy(job) if job else {}


def request_cancel(job_id: str) -> bool:
    if not job_id:
        return False
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return False
        job["cancel_requested"] = True
        job["status_text"] = "Cancel requested. Waiting for safe checkpoint..."
        job["updated_ts"] = _now_ts()
        return True


def cleanup_jobs(max_age_sec: float = 1800.0) -> None:
    cutoff = _now_ts() - max(60.0, float(max_age_sec or 1800.0))
    with _JOB_LOCK:
        stale = [
            jid
            for jid, job in _JOBS.items()
            if float(job.get("updated_ts", 0.0) or 0.0) < cutoff
            and str(job.get("status", "")).lower() in {"done", "failed", "canceled"}
        ]
        for jid in stale:
            _JOBS.pop(jid, None)


def start_trade_finder_job(worker: JobWorker, *, script_ctx: Any = None, name: str = "trade_finder") -> str:
    """Start a background Trade Finder job and return job id."""
    job_id = f"{name}_{uuid.uuid4().hex[:10]}"
    created = _now_ts()
    with _JOB_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "name": name,
            "status": "running",
            "status_text": "Queued...",
            "find_pct": 0,
            "find_text": "Find New: queued",
            "ai_pct": 0,
            "ai_text": "AI ranking: queued",
            "processed": 0,
            "total": 0,
            "hard_gate_pass": 0,
            "ai_ranked": 0,
            "preview_rows": [],
            "cancel_requested": False,
            "error": "",
            "created_ts": created,
            "updated_ts": created,
            "done_ts": 0.0,
            "_thread": None,
        }

    def _set_job_fields(**kwargs: Any) -> None:
        with _JOB_LOCK:
            rec = _JOBS.get(job_id)
            if not rec:
                return
            rec.update(kwargs)
            rec["updated_ts"] = _now_ts()

    def _is_cancel_requested() -> bool:
        with _JOB_LOCK:
            rec = _JOBS.get(job_id) or {}
            return bool(rec.get("cancel_requested", False))

    def _find_progress_cb(pct: int, text: str) -> None:
        _set_job_fields(find_pct=max(0, min(100, int(pct))), find_text=str(text or ""))

    def _ai_progress_cb(pct: int, text: str) -> None:
        _set_job_fields(ai_pct=max(0, min(100, int(pct))), ai_text=str(text or ""))

    def _stream_rows_cb(rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        _set_job_fields(
            preview_rows=list(rows or [])[:10],
            processed=int((meta or {}).get("processed", 0) or 0),
            total=int((meta or {}).get("total", 0) or 0),
            hard_gate_pass=int((meta or {}).get("hard_gate_pass", 0) or 0),
            ai_ranked=int((meta or {}).get("ai_ranked", 0) or 0),
        )

    def _runner() -> None:
        try:
            _set_job_fields(status_text="Running Trade Finder workflow...")
            worker(_find_progress_cb, _ai_progress_cb, _stream_rows_cb, _is_cancel_requested)
            if _is_cancel_requested():
                _set_job_fields(
                    status="canceled",
                    status_text="Canceled at checkpoint.",
                    done_ts=_now_ts(),
                )
            else:
                _set_job_fields(
                    status="done",
                    status_text="Completed.",
                    find_pct=100,
                    ai_pct=100,
                    done_ts=_now_ts(),
                )
        except Exception as exc:
            _set_job_fields(
                status="failed",
                status_text="Failed.",
                error=str(exc),
                done_ts=_now_ts(),
            )

    th = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
    if script_ctx is not None and add_script_run_ctx is not None:
        try:
            add_script_run_ctx(th, script_ctx)
        except Exception:
            # Keep running without context attachment if unavailable.
            pass
    with _JOB_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["_thread"] = th
    th.start()
    return job_id

