from navigation_state import (
    KEY_DEFAULT_DETAIL_TAB,
    KEY_DETAIL_TAB_LOCK,
    KEY_SWITCH_TO_SCANNER_TAB,
    KEY_SWITCH_TO_SCANNER_TARGET_TAB,
    KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL,
    clear_scanner_switch_state,
    consume_detail_tab_with_lock,
    detail_tab_for_target,
    normalize_nav_target,
    set_detail_tab_lock,
    set_scanner_switch_state,
)


def test_normalize_target_and_tab_mapping():
    assert normalize_nav_target("chart") == "chart"
    assert normalize_nav_target("trade") == "trade"
    assert normalize_nav_target("signal") == "signal"
    assert normalize_nav_target("bad-value") == "chart"
    assert detail_tab_for_target("chart") == 1
    assert detail_tab_for_target("trade") == 4
    assert detail_tab_for_target("signal") == 0


def test_lock_consumed_for_same_ticker():
    state = {}
    set_detail_tab_lock(state, ticker="AU", tab_index=1, lock_runs=2, now_ts=100.0)
    assert state[KEY_DEFAULT_DETAIL_TAB] == 1
    assert KEY_DETAIL_TAB_LOCK in state

    first = consume_detail_tab_with_lock(state, ticker="AU", fallback_tab=0, now_ts=101.0)
    second = consume_detail_tab_with_lock(state, ticker="AU", fallback_tab=0, now_ts=102.0)
    third = consume_detail_tab_with_lock(state, ticker="AU", fallback_tab=0, now_ts=103.0)

    assert first == 1
    assert second == 1
    assert third == 0
    assert KEY_DETAIL_TAB_LOCK not in state


def test_lock_clears_on_ticker_mismatch_or_stale():
    state = {}
    set_detail_tab_lock(state, ticker="AU", tab_index=1, lock_runs=3, now_ts=100.0)
    out = consume_detail_tab_with_lock(state, ticker="NVDA", fallback_tab=0, now_ts=101.0)
    assert out == 0
    assert KEY_DETAIL_TAB_LOCK not in state

    set_detail_tab_lock(state, ticker="AU", tab_index=1, lock_runs=3, now_ts=100.0)
    out2 = consume_detail_tab_with_lock(state, ticker="AU", fallback_tab=0, now_ts=200.0)
    assert out2 == 0
    assert KEY_DETAIL_TAB_LOCK not in state


def test_scanner_switch_state_set_and_clear():
    state = {}
    set_scanner_switch_state(state, target="trade", focus_detail=False)
    assert state[KEY_SWITCH_TO_SCANNER_TAB] is True
    assert state[KEY_SWITCH_TO_SCANNER_TARGET_TAB] == "trade"
    assert state[KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL] is False

    clear_scanner_switch_state(state)
    assert KEY_SWITCH_TO_SCANNER_TAB not in state
    assert KEY_SWITCH_TO_SCANNER_TARGET_TAB not in state
    assert KEY_SWITCH_TO_SCANNER_FOCUS_DETAIL not in state

