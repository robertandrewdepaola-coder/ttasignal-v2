from trade_finder_state import (
    adaptive_fetch_batch_size_for_health,
    bg_mode_from_job,
    can_auto_rerank_cached,
    evaluate_active_background_start,
    scope_signature_from_report,
    scope_signature_from_state,
    scope_unchanged,
    should_auto_rerank_after_terminal,
)


def test_scope_signature_from_state_normalizes_values():
    state = {
        "find_new_max_tickers": "250",
        "find_new_in_rotation_only": 1,
        "find_new_include_unknown_sector": 0,
        "find_new_selected_sectors": [" Industrials ", "Technology", ""],
    }
    out = scope_signature_from_state(state)
    assert out == {
        "max_tickers": 250,
        "in_rotation_only": True,
        "include_unknown_sector": False,
        "selected_sectors": ["Industrials", "Technology"],
    }


def test_scope_signature_from_report_uses_signature_then_scan_scope_fallback():
    report = {
        "scan_scope": {
            "max_tickers": 123,
            "only_in_rotation": True,
            "include_unknown_sector": True,
            "selected_sectors": ["Energy"],
        },
        "scope_signature": {
            "max_tickers": 321,
            "selected_sectors": ["Utilities", "Energy"],
        },
    }
    out = scope_signature_from_report(report)
    assert out == {
        "max_tickers": 321,
        "in_rotation_only": True,
        "include_unknown_sector": True,
        "selected_sectors": ["Energy", "Utilities"],
    }


def test_scope_unchanged_and_auto_rerank_guard():
    state = {
        "find_new_max_tickers": 250,
        "find_new_in_rotation_only": True,
        "find_new_include_unknown_sector": False,
        "find_new_selected_sectors": ["Industrials"],
    }
    report = {
        "results_count": 9,
        "scope_signature": {
            "max_tickers": 250,
            "in_rotation_only": True,
            "include_unknown_sector": False,
            "selected_sectors": ["Industrials"],
        },
    }
    assert scope_unchanged(state, report) is True
    assert can_auto_rerank_cached(state, report) is True

    report["scope_signature"]["max_tickers"] = 200
    assert scope_unchanged(state, report) is False
    assert can_auto_rerank_cached(state, report) is False


def test_bg_mode_and_active_start_policy_queue_rerank():
    active_job = {"status": "running", "name": "tf_full_bg"}
    mode = bg_mode_from_job(active_job)
    assert mode == "full"

    decision = evaluate_active_background_start(
        active_job,
        requested_mode="rerank",
        queue_if_running=True,
    )
    assert decision["reuse_active"] is True
    assert decision["queue_rerank"] is True
    assert "queued" in decision["message"].lower()


def test_active_start_policy_when_idle_or_non_running():
    decision = evaluate_active_background_start(
        {"status": "done", "name": "tf_full_bg"},
        requested_mode="full",
        queue_if_running=True,
    )
    assert decision["reuse_active"] is False
    assert decision["queue_rerank"] is False


def test_should_auto_rerank_after_terminal_only_done_full():
    assert should_auto_rerank_after_terminal(
        status="done",
        mode="full",
        manual_queue=False,
        auto_enabled=True,
    ) is True
    assert should_auto_rerank_after_terminal(
        status="done",
        mode="full",
        manual_queue=True,
        auto_enabled=False,
    ) is True
    assert should_auto_rerank_after_terminal(
        status="canceled",
        mode="full",
        manual_queue=True,
        auto_enabled=True,
    ) is False
    assert should_auto_rerank_after_terminal(
        status="done",
        mode="rerank",
        manual_queue=True,
        auto_enabled=True,
    ) is False


def test_adaptive_fetch_batch_size_for_health_throttles_under_pressure():
    assert adaptive_fetch_batch_size_for_health(80, {"cooldown_remaining_sec": 10}) == 20
    assert adaptive_fetch_batch_size_for_health(80, {"hits": 2}) == 30
    assert adaptive_fetch_batch_size_for_health(80, {"requests_last_1s": 6}) == 35
    assert adaptive_fetch_batch_size_for_health(80, {"requests_last_1s": 1, "hits": 0}) == 80
