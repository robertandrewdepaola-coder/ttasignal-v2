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
KEY_DETAIL_NAV_INTENT = "_detail_nav_intent"
KEY_SWITCH_TO_SCANNER_TAB = "_switch_to_scanner_tab"
KEY_SWITCH_TO_SCANNER_TARGET_TAB = "_switch_to_scanner_target_tab"
KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL = "_switch_to_scanner_focus_detail"
KEY_DETAIL_TAB_SELECTOR_PREFIX = "detail_view_tab_"
KEY_DETAIL_TAB_SELECTOR_PENDING_PREFIX = "_pending_detail_view_tab_"

# Supported navigation targets -> detail tab index
_TARGET_TO_TAB = {
    "signal": 0,
    "chart": 1,
    "trade": 4,
}

_TARGET_TO_SELECTOR = {
    "signal": "signal",
    "chart": "chart",
    "trade": "trade",
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


def detail_selector_key_for_ticker(ticker: Any) -> str:
    """Canonical per-ticker detail selector key."""
    _ticker = str(ticker or "").upper().strip()
    return f"{KEY_DETAIL_TAB_SELECTOR_PREFIX}{_ticker}"


def detail_selector_pending_key_for_ticker(ticker: Any) -> str:
    """Canonical per-ticker deferred selector key."""
    _ticker = str(ticker or "").upper().strip()
    return f"{KEY_DETAIL_TAB_SELECTOR_PENDING_PREFIX}{_ticker}"


def set_detail_tab_selector_target(
    state: MutableMapping[str, Any],
    *,
    ticker: Any,
    target: Any,
) -> str:
    """Set per-ticker detail selector target (signal/chart/trade) and return value.

    If the selector widget key is already instantiated in the current Streamlit
    pass, direct assignment can raise StreamlitAPIException. In that case we
    store the desired selector in a deferred key and apply it next rerun before
    radio render.
    """
    _target = normalize_nav_target(target, fallback="chart")
    _selector = _TARGET_TO_SELECTOR.get(_target, "chart")
    _selector_key = detail_selector_key_for_ticker(ticker)
    _pending_key = detail_selector_pending_key_for_ticker(ticker)
    try:
        state[_selector_key] = _selector
        state.pop(_pending_key, None)
    except Exception:
        state[_pending_key] = _selector
    return _selector


def consume_detail_tab_selector_pending(
    state: MutableMapping[str, Any],
    *,
    ticker: Any,
) -> str:
    """Consume deferred selector target for a ticker (returns signal/chart/trade or '')."""
    _pending_key = detail_selector_pending_key_for_ticker(ticker)
    _raw = str(state.pop(_pending_key, "") or "").strip().lower()
    return _raw if _raw in _TARGET_TO_SELECTOR.values() else ""


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


def set_detail_nav_intent(
    state: MutableMapping[str, Any],
    *,
    ticker: Any,
    target: Any,
    lock_runs: int = 4,
    now_ts: float | None = None,
) -> None:
    """Set short-lived detail-nav intent (signal/chart/trade) for a ticker."""
    _ticker = str(ticker or "").upper().strip()
    _target = normalize_nav_target(target, fallback="chart")
    _now = float(time.time() if now_ts is None else now_ts)
    state[KEY_DETAIL_NAV_INTENT] = {
        "ticker": _ticker,
        "target": _target,
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


def consume_detail_nav_intent(
    state: MutableMapping[str, Any],
    *,
    ticker: Any,
    max_age_sec: float = 60.0,
    now_ts: float | None = None,
) -> str:
    """Consume per-ticker nav intent and return target label if still valid."""
    _ticker = str(ticker or "").upper().strip()
    _intent = state.get(KEY_DETAIL_NAV_INTENT)
    if not isinstance(_intent, dict):
        return ""

    _intent_ticker = str(_intent.get("ticker", "") or "").upper().strip()
    _target = normalize_nav_target(_intent.get("target"), fallback="chart")
    _remaining = int(_intent.get("remaining", 0) or 0)
    _age = float(time.time() if now_ts is None else now_ts) - float(_intent.get("set_at", 0) or 0)

    if _intent_ticker != _ticker or _age > float(max_age_sec) or _remaining <= 0:
        state.pop(KEY_DETAIL_NAV_INTENT, None)
        return ""

    _intent["remaining"] = _remaining - 1
    if int(_intent["remaining"]) <= 0:
        state.pop(KEY_DETAIL_NAV_INTENT, None)
    else:
        state[KEY_DETAIL_NAV_INTENT] = _intent
    return _target


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
