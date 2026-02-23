"""Trade Finder state/queue/scope policy helpers.

Pure helpers extracted from the Streamlit UI layer so behavior can be tested
without coupling to ``st.session_state`` or background-job side effects.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _clean_sector_list(values: Any) -> list[str]:
    return sorted(
        [
            str(v).strip()
            for v in (values or [])
            if str(v).strip()
        ]
    )


def scope_signature_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical scope-signature snapshot from session-like state mapping."""
    return {
        "max_tickers": int(state.get("find_new_max_tickers", 0) or 0),
        "in_rotation_only": bool(state.get("find_new_in_rotation_only", False)),
        "include_unknown_sector": bool(state.get("find_new_include_unknown_sector", False)),
        "selected_sectors": _clean_sector_list(state.get("find_new_selected_sectors", [])),
    }


def scope_signature_from_report(report: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Canonical scope-signature snapshot from a find-new report payload."""
    _report = dict(report or {})
    _scope = dict(_report.get("scan_scope", {}) or {})
    _sig = dict(_report.get("scope_signature", {}) or {})
    _sig_sectors = _sig.get("selected_sectors", _scope.get("selected_sectors", []))

    return {
        "max_tickers": int(_sig.get("max_tickers", _scope.get("max_tickers", 0)) or 0),
        "in_rotation_only": bool(_sig.get("in_rotation_only", _scope.get("only_in_rotation", False))),
        "include_unknown_sector": bool(
            _sig.get("include_unknown_sector", _scope.get("include_unknown_sector", False))
        ),
        "selected_sectors": _clean_sector_list(_sig_sectors),
    }


def scope_unchanged(state: Mapping[str, Any], report: Mapping[str, Any] | None) -> bool:
    """True when current scope knobs match a cached report scope signature."""
    if not report:
        return False
    return scope_signature_from_state(state) == scope_signature_from_report(report)


def can_auto_rerank_cached(state: Mapping[str, Any], report: Mapping[str, Any] | None) -> bool:
    """Whether cached rows are eligible for no-rescan auto re-rank."""
    _report = dict(report or {})
    if int(_report.get("results_count", 0) or 0) <= 0:
        return False
    return scope_unchanged(state, _report)


def bg_mode_from_job(job: Mapping[str, Any] | None) -> str:
    """Normalized background-job mode: ``full`` / ``rerank`` / ``""``."""
    _name = str((job or {}).get("name", "") or "").strip().lower()
    if "rerank" in _name:
        return "rerank"
    if "full" in _name:
        return "full"
    return ""


def evaluate_active_background_start(
    active_job: Mapping[str, Any] | None,
    *,
    requested_mode: str,
    queue_if_running: bool,
) -> Dict[str, Any]:
    """Policy decision for start requests when a background job may already run."""
    _status = str((active_job or {}).get("status", "") or "").strip().lower()
    _active_mode = bg_mode_from_job(active_job)
    _requested = str(requested_mode or "").strip().lower()

    if _status != "running":
        return {
            "reuse_active": False,
            "queue_rerank": False,
            "message": "",
            "active_mode": _active_mode,
        }

    _queue_rerank = bool(queue_if_running and _requested == "rerank" and _active_mode == "full")
    if _queue_rerank:
        _message = "Background full scan already running. Re-rank queued for auto-start on completion."
    elif queue_if_running and _requested == "full" and _active_mode == "rerank":
        _message = "Background re-rank already running. Start full scan when current job completes."
    else:
        _message = "Background Trade Finder already running; using the current job."

    return {
        "reuse_active": True,
        "queue_rerank": _queue_rerank,
        "message": _message,
        "active_mode": _active_mode,
    }


def should_auto_rerank_after_terminal(
    *,
    status: str,
    mode: str,
    manual_queue: bool,
    auto_enabled: bool,
) -> bool:
    """True when terminal full-run state should trigger automatic rerank."""
    return bool(
        str(status or "").strip().lower() == "done"
        and str(mode or "").strip().lower() == "full"
        and (bool(manual_queue) or bool(auto_enabled))
    )


def adaptive_fetch_batch_size_for_health(
    configured_batch_size: Any,
    fetch_health: Mapping[str, Any] | None,
    *,
    min_batch: int = 10,
    max_batch: int = 250,
) -> int:
    """Throttle fetch batch size based on provider health/governor pressure."""
    _size = int(configured_batch_size or 40)
    _size = max(min_batch, min(max_batch, _size))

    _fh = dict(fetch_health or {})
    _cooldown = int(_fh.get("cooldown_remaining_sec", 0) or 0)
    _hits = int(_fh.get("hits", 0) or 0)
    _rate_limited = bool(_fh.get("rate_limited", False))
    _req_1s = int(_fh.get("requests_last_1s", 0) or 0)

    # Hard pressure: significantly reduce burst size.
    if _rate_limited or _cooldown > 0 or _hits >= 3:
        return max(min_batch, min(_size, 20))

    # Mild pressure or current high request rate.
    if _hits >= 1 or _req_1s >= 8:
        return max(min_batch, min(_size, 30))

    # Soft pressure.
    if _req_1s >= 5:
        return max(min_batch, min(_size, 35))

    return _size
