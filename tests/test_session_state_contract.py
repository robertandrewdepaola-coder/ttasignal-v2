from session_state_contract import (
    CORE_DEFAULT_FACTORIES,
    EXEC_DASHBOARD_DEFAULT_FACTORIES,
    EXEC_QUEUE_DEFAULT_FACTORIES,
    PORTFOLIO_DEFAULT_FACTORIES,
    SCANNER_DEFAULT_FACTORIES,
    TRADE_QUALITY_DEFAULT_FACTORIES,
    ensure_core_defaults,
    ensure_exec_dashboard_defaults,
    ensure_exec_queue_defaults,
    ensure_portfolio_defaults,
    ensure_scanner_defaults,
    ensure_trade_quality_defaults,
)


def test_ensure_core_defaults_populates_missing_keys():
    state = {}
    ensure_core_defaults(state)
    for key in CORE_DEFAULT_FACTORIES:
        assert key in state
    assert state["find_new_max_tickers"] == 250
    assert state["trade_finder_ai_top_n"] == 0


def test_ensure_core_defaults_does_not_override_existing_values():
    state = {
        "find_new_max_tickers": 99,
        "find_new_in_rotation_only": False,
        "trade_finder_ai_top_n": 8,
    }
    ensure_core_defaults(state)
    assert state["find_new_max_tickers"] == 99
    assert state["find_new_in_rotation_only"] is False
    assert state["trade_finder_ai_top_n"] == 8


def test_ensure_core_defaults_creates_fresh_mutable_defaults():
    s1 = {}
    s2 = {}
    ensure_core_defaults(s1)
    ensure_core_defaults(s2)
    assert s1["ticker_data_cache"] is not s2["ticker_data_cache"]
    assert s1["find_new_selected_sectors"] is not s2["find_new_selected_sectors"]


def test_ensure_trade_quality_defaults_populates_missing_keys():
    state = {}
    ensure_trade_quality_defaults(state)
    for key in TRADE_QUALITY_DEFAULT_FACTORIES:
        assert key in state
    assert state["trade_min_rr_threshold"] == 1.2
    assert state["trade_apex_primary"] is True


def test_ensure_trade_quality_defaults_does_not_override_existing_values():
    state = {
        "trade_min_rr_threshold": 2.0,
        "trade_require_ready": True,
    }
    ensure_trade_quality_defaults(state)
    assert state["trade_min_rr_threshold"] == 2.0
    assert state["trade_require_ready"] is True


def test_ensure_scanner_defaults_populates_keys():
    state = {}
    ensure_scanner_defaults(state)
    for key in SCANNER_DEFAULT_FACTORIES:
        assert key in state
    assert state["wl_version"] == 0
    assert state["scanner_page"] == 0


def test_ensure_portfolio_defaults_populates_keys():
    state = {}
    ensure_portfolio_defaults(state)
    for key in PORTFOLIO_DEFAULT_FACTORIES:
        assert key in state
    assert state["account_size"] == 100000.0


def test_ensure_exec_dashboard_defaults_populates_keys():
    state = {}
    ensure_exec_dashboard_defaults(state)
    for key in EXEC_DASHBOARD_DEFAULT_FACTORIES:
        assert key in state
    assert state["exec_auto_exit_enabled"] is False


def test_ensure_exec_queue_defaults_populates_and_keeps_existing():
    state = {"_queue_recent_actions": [{"status": "ok"}]}
    ensure_exec_queue_defaults(state)
    for key in EXEC_QUEUE_DEFAULT_FACTORIES:
        assert key in state
    assert state["_queue_recent_actions"] == [{"status": "ok"}]
