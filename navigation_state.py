"""Navigation/session-state helpers for Scanner detail view routing.

This module centralizes the fragile session-state contract used by chart/trade
navigation so state keys and lock behavior are explicit and testable.
"""

from __future__ import annotations

import time
from typing import Any, MutableMapping

# Session-state keys
KEY_DEFAULT_DETAIL_TAB = "default_detail_tab"
KEY_DETAIL_TAB_LOCK = "_detail_tab_lock"
KEY_SWITCH_TO_SCANNER_TAB = "_switch_to_scanner_tab"
KEY_SWITCH_TO_SCANNER_TARGET_TAB = "_switch_to_scanner_target_tab"
KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL = "_switch_to_scanner_focus_detail"

# Supported navigation targets -> detail tab index
_TARGET_TO_TAB = {
    "signal": 0,
    "chart": 1,
    "trade": 4,
}


def normalize_nav_target(target: Any, fallback: str = "chart") -> str:
    """Normalize target to one of: signal/chart/trade."""
    _raw = str(target or "").strip().lower()
    if _raw in _TARGET_TO_TAB:
        return _raw
    return str(fallback).strip().lower() if str(fallback).strip().lower() in _TARGET_TO_TAB else "chart"


def detail_tab_for_target(target: Any, fallback: str = "chart") -> int:
    """Resolve detail tab index for a target label."""
    return int(_TARGET_TO_TAB.get(normalize_nav_target(target, fallback=fallback), _TARGET_TO_TAB["chart"]))


def set_detail_tab_lock(
    state: MutableMapping[str, Any],
    *,
    ticker: str,
    tab_index: int,
    lock_runs: int = 3,
    now_ts: float | None = None,
) -> None:
    """Set one-shot detail tab intent and a short-lived lock for rerun stability."""
    _ticker = str(ticker or "").upper().strip()
    _tab = int(tab_index or 0)
    _now = float(time.time() if now_ts is None else now_ts)
    state[KEY_DEFAULT_DETAIL_TAB] = _tab
    state[KEY_DETAIL_TAB_LOCK] = {
        "ticker": _ticker,
        "tab": _tab,
        "remaining": max(0, int(lock_runs)),
        "set_at": _now,
    }


def consume_detail_tab_with_lock(
    state: MutableMapping[str, Any],
    *,
    ticker: str,
    fallback_tab: int = 0,
    max_age_sec: float = 8.0,
    now_ts: float | None = None,
) -> int:
    """Consume requested default tab and apply lock if valid for this ticker."""
    _ticker = str(ticker or "").upper().strip()
    _fallback = int(fallback_tab or 0)
    _default = int(state.pop(KEY_DEFAULT_DETAIL_TAB, _fallback) or _fallback)

    _lock = state.get(KEY_DETAIL_TAB_LOCK)
    if not isinstance(_lock, dict):
        return _default

    _lock_ticker = str(_lock.get("ticker", "") or "").upper().strip()
    _lock_tab = int(_lock.get("tab", _default) or _default)
    _remaining = int(_lock.get("remaining", 0) or 0)
    _age = float(time.time() if now_ts is None else now_ts) - float(_lock.get("set_at", 0) or 0)

    if _lock_ticker == _ticker and _remaining > 0 and _age <= float(max_age_sec):
        _lock["remaining"] = _remaining - 1
        if int(_lock["remaining"]) <= 0:
            state.pop(KEY_DETAIL_TAB_LOCK, None)
        else:
            state[KEY_DETAIL_TAB_LOCK] = _lock
        return _lock_tab

    # Invalid lock (old ticker or stale lock): clear it and fall back.
    if _lock_ticker != _ticker or _age > float(max_age_sec):
        state.pop(KEY_DETAIL_TAB_LOCK, None)
        return _fallback
    return _default


def set_scanner_switch_state(
    state: MutableMapping[str, Any],
    *,
    target: Any,
    focus_detail: bool = True,
) -> None:
    """Set state to request main-tab switch to Scanner and focus detail view."""
    _target = normalize_nav_target(target, fallback="chart")
    state[KEY_SWITCH_TO_SCANNER_TAB] = True
    state[KEY_SWITCH_TO_SCANNER_TARGET_TAB] = _target
    state[KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL] = bool(focus_detail)


def clear_scanner_switch_state(state: MutableMapping[str, Any]) -> None:
    """Clear deferred scanner-tab switching state."""
    state.pop(KEY_SWITCH_TO_SCANNER_TAB, None)
    state.pop(KEY_SWITCH_TO_SCANNER_TARGET_TAB, None)
    state.pop(KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL, None)
