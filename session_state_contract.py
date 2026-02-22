"""Centralized session-state defaults.

Keeps key contracts explicit and reduces repeated ad-hoc initialization logic
in the Streamlit UI layer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping

Factory = Callable[[], Any]


def _const(value: Any) -> Factory:
    return lambda: value


CORE_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "selected_ticker": _const(None),
    "selected_analysis": _const(None),
    "ticker_data_cache": dict,
    "find_new_max_tickers": _const(250),
    "find_new_in_rotation_only": _const(True),
    "find_new_include_unknown_sector": _const(False),
    "find_new_selected_sectors": list,
    "scan_max_minutes": _const(0.0),
    "find_new_max_minutes": _const(5.0),
    "trade_finder_last_status": dict,
    "trade_finder_ai_top_n": _const(0),
    "trade_finder_ai_budget_sec": _const(30.0),
    "trade_breakout_min_dist_pct": _const(0.2),
    "trade_breakout_max_dist_pct": _const(4.0),
    "trade_monthly_near_macd_pct": _const(0.08),
    "trade_monthly_near_ao_floor": _const(-0.25),
}


TRADE_QUALITY_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "trade_min_rr_threshold": _const(1.2),
    "trade_earnings_block_days": _const(7),
    "trade_require_ready": _const(False),
    "trade_include_watch_only": _const(True),
    "trade_breakout_min_dist_pct": _const(0.2),
    "trade_breakout_max_dist_pct": _const(4.0),
    "trade_monthly_near_macd_pct": _const(0.08),
    "trade_monthly_near_ao_floor": _const(-0.25),
    "trade_apex_primary": _const(True),
    "trade_apex_bear_vix_threshold": _const(20.0),
}


SCANNER_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "wl_version": _const(0),
    "scanner_page": _const(0),
}


PORTFOLIO_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "account_size": _const(100000.0),
}


EXEC_DASHBOARD_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "exec_auto_exit_enabled": _const(False),
}


EXEC_QUEUE_DEFAULT_FACTORIES: Dict[str, Factory] = {
    "_queue_action_confirm": dict,
    "_queue_pending_actions": dict,
    "_queue_action_executed": dict,
    "_queue_recent_actions": list,
    "_queue_last_result_toasts": list,
}


def ensure_state_defaults(
    state: MutableMapping[str, Any],
    factories: Dict[str, Factory],
) -> None:
    """Populate missing keys in session state from factory defaults."""
    for key, factory in factories.items():
        if key not in state:
            state[key] = factory()


def ensure_core_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, CORE_DEFAULT_FACTORIES)


def ensure_trade_quality_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, TRADE_QUALITY_DEFAULT_FACTORIES)


def ensure_scanner_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, SCANNER_DEFAULT_FACTORIES)


def ensure_portfolio_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, PORTFOLIO_DEFAULT_FACTORIES)


def ensure_exec_dashboard_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, EXEC_DASHBOARD_DEFAULT_FACTORIES)


def ensure_exec_queue_defaults(state: MutableMapping[str, Any]) -> None:
    ensure_state_defaults(state, EXEC_QUEUE_DEFAULT_FACTORIES)
