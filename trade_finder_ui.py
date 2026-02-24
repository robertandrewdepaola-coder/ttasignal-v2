"""UI helpers for Trade Finder top-panel rendering.

This module isolates navigation/state-heavy UI wiring from app.py so behavior
remains consistent while reducing monolith coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Callable, Dict, List, Set

import pandas as pd
import streamlit as st


@dataclass
class TradeFinderTopState:
    alert_tickers: Set[str]
    data_gate: Dict[str, Any]
    bg_job: Dict[str, Any]
    bg_status: str
    bg_mode: str
    bg_full_active: bool
    bg_active: bool
    bg_preview_rows: List[Dict[str, Any]]
    run_active: bool
    cancel_requested: bool


def render_trade_finder_top_panel(
    *,
    can_auto_rerank_cached_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool],
    today_utc_str_fn: Callable[[], str],
    get_fetch_health_status_fn: Callable[[], Dict[str, Any]],
    trade_finder_data_gate_state_fn: Callable[..., Dict[str, Any]],
    get_journal_fn: Callable[[], Any],
    cleanup_background_jobs_fn: Callable[[], None],
    get_trade_finder_background_status_fn: Callable[[], Dict[str, Any]],
    bg_mode_from_job_fn: Callable[[Dict[str, Any]], str],
    start_trade_finder_background_run_fn: Callable[..., str],
    request_trade_finder_background_cancel_fn: Callable[[], bool],
    format_eta_fn: Callable[[float], str],
    fmt_last_update_fn: Callable[[float], str],
    render_green_on_red_sector_finder_fn: Callable[..., None],
) -> TradeFinderTopState:
    """Render Trade Finder top controls/status and return state used by row UI."""
    _pending_gate = st.session_state.pop('trade_gate_pending_apply', None)
    if isinstance(_pending_gate, dict) and _pending_gate:
        _auto_rerank_note = ""
        try:
            _applied_max = float(
                _pending_gate.get('trade_breakout_max_dist_pct', st.session_state.get('trade_breakout_max_dist_pct', 4.0))
            )
            _applied_macd = float(
                _pending_gate.get('trade_monthly_near_macd_pct', st.session_state.get('trade_monthly_near_macd_pct', 0.08))
            )
            _applied_ao = float(
                _pending_gate.get('trade_monthly_near_ao_floor', st.session_state.get('trade_monthly_near_ao_floor', -0.25))
            )
            st.session_state['trade_breakout_max_dist_pct'] = _applied_max
            st.session_state['trade_monthly_near_macd_pct'] = _applied_macd
            st.session_state['trade_monthly_near_ao_floor'] = _applied_ao
            st.session_state['trade_gate_last_applied'] = {
                'trade_breakout_max_dist_pct': _applied_max,
                'trade_monthly_near_macd_pct': _applied_macd,
                'trade_monthly_near_ao_floor': _applied_ao,
                'applied_ts': today_utc_str_fn(),
            }
            _cached_report = st.session_state.get('find_new_trades_report', {}) or {}
            _can_rerank_cached = can_auto_rerank_cached_fn(st.session_state, _cached_report)
            if _can_rerank_cached:
                st.session_state['trade_gate_auto_rerank_requested'] = True
                st.session_state['trade_gate_auto_rerank_requested_ts'] = time.time()
                _auto_rerank_note = " Filters applied. Starting cached re-rank automatically (no rescan)."
            else:
                st.session_state.pop('trade_gate_auto_rerank_requested', None)
                _auto_rerank_note = " Filters applied. Cached rerank skipped (scope changed or no cached analysis rows)."
        except Exception:
            pass
        _pending_reason = str(_pending_gate.get('_reason', '') or '').strip()
        if _auto_rerank_note:
            _pending_reason = f"{_pending_reason}{_auto_rerank_note}".strip()
        if _pending_reason:
            st.session_state['trade_gate_pending_notice'] = _pending_reason

    st.subheader("🧭 Trade Finder")
    st.caption("Runs cross-watchlist scan and ranks model/system trade signals with clear actions.")
    _pending_notice = str(st.session_state.pop('trade_gate_pending_notice', '') or '').strip()
    if _pending_notice:
        st.success(_pending_notice)
    _fh = get_fetch_health_status_fn() or {}
    _data_gate = trade_finder_data_gate_state_fn(rerank_only=False)
    if bool(_fh.get('rate_limited', False)):
        st.warning(
            f"Data provider is rate-limited ({int(_fh.get('cooldown_remaining_sec', 0) or 0)}s cooldown). "
            "Trade Finder is running in partial-data mode using cache/fallbacks."
        )
    if bool(_data_gate.get('hard_block', False)):
        st.error(
            f"Trade Finder data gate: BLOCKED — {_data_gate.get('reason', 'data not ready')}. "
            "Use Fast Refresh / Refresh Mkt, then rerun."
        )
    elif list(_data_gate.get('warnings', []) or []):
        st.info("Trade Finder data gate: " + "; ".join([str(x) for x in (_data_gate.get('warnings', []) or [])]))
    st.caption(
        "Fetch governor: "
        f"interval {float(_data_gate.get('governor_interval_sec', _fh.get('governor_interval_sec', 0.08)) or 0.08):.2f}s | "
        f"req/s {int(_data_gate.get('requests_last_1s', _fh.get('requests_last_1s', 0)) or 0)} | "
        f"429 hits {int(_data_gate.get('hits', _fh.get('hits', 0)) or 0)}"
    )
    jm = get_journal_fn()
    alert_tickers = {
        str(c.get('ticker', '')).upper().strip()
        for c in (jm.get_pending_conditionals() or [])
        if str(c.get('status', 'PENDING')).upper() == "PENDING" and str(c.get('ticker', '')).strip()
    }

    with st.expander("⚙️ Find New Scope", expanded=False):
        st.number_input(
            "Max tickers to scan (0 = all)",
            min_value=0,
            max_value=2000,
            step=25,
            key="find_new_max_tickers",
            help="Cap scan size to reduce runtime and API stress. 0 scans full cross-watchlist universe.",
        )
        st.checkbox(
            "Only in-rotation sectors (Leading/Emerging)",
            key="find_new_in_rotation_only",
            help="Excludes tickers in Fading/Lagging sectors before scan.",
        )
        st.checkbox(
            "Include unknown-sector tickers (slower)",
            key="find_new_include_unknown_sector",
            help="When off, unknown-sector symbols are skipped for faster, stricter in-rotation scans.",
        )
        _rotation = st.session_state.get('sector_rotation', {}) or {}
        _sector_options = sorted([str(s) for s in _rotation.keys() if str(s)])
        if _sector_options:
            st.multiselect(
                "Limit to sectors",
                options=_sector_options,
                key="find_new_selected_sectors",
                help="Optional sector subset. Leave empty for all sectors.",
            )
        st.number_input(
            "Max scan minutes (0 = no limit)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            key="find_new_max_minutes",
            help="Stops long scans once this runtime limit is hit (partial results are kept).",
        )
        adv1, adv2 = st.columns(2)
        with adv1:
            st.number_input(
                "Incremental cache TTL (minutes)",
                min_value=1.0,
                max_value=120.0,
                step=1.0,
                key="find_new_cache_ttl_min",
                help="Recently scanned tickers inside this TTL reuse cached OHLC data instead of refetching.",
            )
        with adv2:
            st.number_input(
                "Fetch batch size",
                min_value=10,
                max_value=250,
                step=10,
                key="find_new_fetch_batch_size",
                help="Controls request burst size. Smaller batches reduce API pressure and improve pacing visibility.",
            )

    _scope_bits = []
    if int(st.session_state.get('find_new_max_tickers', 0) or 0) > 0:
        _scope_bits.append(f"max {int(st.session_state.get('find_new_max_tickers', 0) or 0)}")
    if bool(st.session_state.get('find_new_in_rotation_only', False)):
        _scope_bits.append("in-rotation only")
    if bool(st.session_state.get('find_new_include_unknown_sector', False)):
        _scope_bits.append("unknown sectors included")
    _sel_scope = st.session_state.get('find_new_selected_sectors', []) or []
    if _sel_scope:
        _scope_bits.append(f"{len(_sel_scope)} sectors")
    if float(st.session_state.get('find_new_max_minutes', 0.0) or 0.0) > 0:
        _scope_bits.append(f"{float(st.session_state.get('find_new_max_minutes', 0.0) or 0.0):.1f}m cap")
    _scope_bits.append(f"cache {float(st.session_state.get('find_new_cache_ttl_min', 20.0) or 20.0):.0f}m")
    _scope_bits.append(f"batch {int(st.session_state.get('find_new_fetch_batch_size', 40) or 40)}")
    if _scope_bits:
        st.caption("Active scope: " + " | ".join(_scope_bits))

    st.checkbox(
        "Auto re-rank after background full scan",
        key="trade_finder_bg_auto_rerank",
        help=(
            "When enabled, a background re-rank automatically starts after each completed "
            "background full scan."
        ),
    )
    if bool(st.session_state.get('trade_finder_bg_rerank_queued', False)):
        st.caption("🕒 Queued: one background re-rank will auto-start after the current full scan completes.")

    cleanup_background_jobs_fn()
    _bg_job = get_trade_finder_background_status_fn()
    _bg_status = str((_bg_job or {}).get('status', '') or '').lower()
    _bg_mode = bg_mode_from_job_fn(_bg_job)
    _bg_full_active = bool(_bg_status == "running" and _bg_mode == "full")
    _bg_active = _bg_status == "running"
    _bg_preview_rows = (_bg_job or {}).get('preview_rows', []) or []
    _run_active = bool(st.session_state.get('trade_finder_running', False)) or _bg_active
    _cancel_requested = bool(st.session_state.get('trade_finder_cancel_requested', False))

    if bool(st.session_state.get('trade_gate_auto_rerank_requested', False)):
        _queued_job = start_trade_finder_background_run_fn(rerank_only=True, queue_if_running=True)
        if _queued_job:
            st.session_state.pop('trade_gate_auto_rerank_requested', None)
            st.session_state['trade_finder_last_status'] = {
                'level': 'info',
                'message': "Auto-calibrated filters applied. Cached re-rank started/queued (no rescan).",
                'ts': time.time(),
            }
            st.rerun()

    _rc1, _rc2 = st.columns([4, 1])
    with _rc1:
        if _bg_active:
            _find_pct = int((_bg_job or {}).get('find_pct', 0) or 0)
            _find_txt = str((_bg_job or {}).get('find_text', '') or 'Find New running...')
            _ai_pct = int((_bg_job or {}).get('ai_pct', 0) or 0)
            _ai_txt = str((_bg_job or {}).get('ai_text', '') or 'AI ranking running...')
            _processed = int((_bg_job or {}).get('processed', 0) or 0)
            _total = int((_bg_job or {}).get('total', 0) or 0)
            _pass = int((_bg_job or {}).get('hard_gate_pass', 0) or 0)
            _ranked = int((_bg_job or {}).get('ai_ranked', 0) or 0)
            _phase = str((_bg_job or {}).get('phase', '') or '').strip()
            _scoped_total = int((_bg_job or {}).get('scoped_total', 0) or 0)
            _fetched_done = int((_bg_job or {}).get('fetched_done', 0) or 0)
            _fetched_total = int((_bg_job or {}).get('fetched_total', 0) or 0)
            _analysis_done = int((_bg_job or {}).get('analysis_done', 0) or 0)
            _analysis_total = int((_bg_job or {}).get('analysis_total', 0) or 0)
            _eta_sec = float((_bg_job or {}).get('eta_sec', 0.0) or 0.0)
            _elapsed_sec = float((_bg_job or {}).get('run_elapsed_sec', 0.0) or 0.0)
            _fetch_rate_limited = bool((_bg_job or {}).get('fetch_rate_limited', False))
            _fetch_cooldown = int((_bg_job or {}).get('fetch_cooldown_sec', 0) or 0)
            _fetch_gov = float((_bg_job or {}).get('fetch_governor_interval_sec', 0.08) or 0.08)
            _fetch_req = int((_bg_job or {}).get('fetch_requests_last_1s', 0) or 0)
            _fetch_pct = int(round((100.0 * _fetched_done / max(1, _fetched_total)))) if _fetched_total > 0 else 0
            _analysis_pct = int(round((100.0 * _analysis_done / max(1, _analysis_total)))) if _analysis_total > 0 else 0
            _rank_pct = int(round((100.0 * _processed / max(1, _total)))) if _total > 0 else 0
            _phase_label = {
                'queued': 'Queued',
                'find_scope': 'Scope',
                'find_fetch': 'Fetch',
                'find_analyze': 'Analyze',
                'ai_seed': 'Seed',
                'ai_rank': 'AI Rank',
                'done': 'Done',
                'canceled': 'Canceled',
            }.get(_phase, (_phase or 'running').replace('_', ' ').title())
            _overall_pct = int(round((_find_pct * 0.6) + (_ai_pct * 0.4)))
            st.warning("Background Trade Finder run active. UI remains responsive.")
            st.progress(
                max(0, min(100, _overall_pct)),
                text=(
                    f"Overall {_overall_pct}% | phase {_phase_label} | "
                    f"elapsed {format_eta_fn(_elapsed_sec)} | ETA {format_eta_fn(_eta_sec)}"
                ),
            )
            st.progress(_find_pct, text=_find_txt)
            st.progress(_ai_pct, text=_ai_txt)
            st.progress(_fetch_pct, text=f"Fetch {_fetched_done}/{_fetched_total} ({_fetch_pct}%)")
            st.progress(_analysis_pct, text=f"Analyze {_analysis_done}/{_analysis_total} ({_analysis_pct}%)")
            st.progress(_rank_pct, text=f"Rank {_processed}/{_total} ({_rank_pct}%)")
            st.caption(
                f"Background progress: {_processed}/{_total} processed | "
                f"hard-gate pass {_pass} | AI-ranked {_ranked}"
            )
            st.caption(
                f"Scope {_scoped_total} | fetched {_fetched_done}/{_fetched_total} | "
                f"analyzed {_analysis_done}/{_analysis_total} | "
                f"phase {(_phase or 'n/a')} | elapsed {format_eta_fn(_elapsed_sec)} | ETA {format_eta_fn(_eta_sec)}"
            )
            st.caption(
                "Fetch governor: "
                f"interval {_fetch_gov:.2f}s | req/s {_fetch_req} | "
                f"{'cooldown ' + str(_fetch_cooldown) + 's' if _fetch_rate_limited else 'no cooldown'}"
            )
            if _bg_preview_rows:
                _preview_df = pd.DataFrame(
                    [
                        {
                            "Ticker": str(_r.get("ticker", "")).upper(),
                            "Verdict": str(_r.get("ai_buy_recommendation", "")),
                            "Score": round(float(_r.get("trade_score", 0.0) or 0.0), 2),
                            "R:R": f"{float(_r.get('risk_reward', 0.0) or 0.0):.2f}:1",
                            "Earn(d)": int(_r.get("earn_days", 999) or 999),
                            "Sector": str(_r.get("sector", "") or "-"),
                        }
                        for _r in _bg_preview_rows[:8]
                    ]
                )
                st.dataframe(_preview_df, hide_index=True, width="stretch")
            st.caption("Live progress updates are available. Use Refresh Status to poll without tab flicker.")
        elif _run_active:
            st.warning("Trade Finder run is active. You can request stop; it will halt at the next safe checkpoint.")
        elif _bg_status == "interrupted":
            _bg_upd = float((_bg_job or {}).get('updated_ts', 0.0) or 0.0)
            _when = fmt_last_update_fn(_bg_upd, fallback="unknown")
            st.warning(f"Recovered interrupted background run ({_when}). Start a new run to continue.")
        elif _cancel_requested:
            st.info("A cancel request is queued for the next running workflow.")
    with _rc2:
        if _run_active:
            if st.button("🛑 Cancel Run", key="tf_cancel_run_btn", width="stretch"):
                if _bg_active:
                    request_trade_finder_background_cancel_fn()
                else:
                    st.session_state['trade_finder_cancel_requested'] = True
                    st.session_state['trade_finder_last_status'] = {
                        'level': 'warning',
                        'message': "Cancel requested. Trade Finder will stop at the next batch/checkpoint.",
                        'ts': time.time(),
                    }
                st.rerun()
            if _bg_active and st.button("🔄 Refresh Status", key="tf_refresh_run_btn", width="stretch"):
                st.rerun()

    with st.expander("🧠 Green-on-Red Sector Finder", expanded=False):
        _tf_rows = (st.session_state.get('trade_finder_results', {}) or {}).get('rows', []) or []
        _fn_rows = (st.session_state.get('find_new_trades_report', {}) or {}).get('candidates', []) or []
        _scan_rows = st.session_state.get('scan_results_summary', []) or []
        _source_rows = _tf_rows if _tf_rows else (_fn_rows if _fn_rows else _scan_rows)
        render_green_on_red_sector_finder_fn(_source_rows, key_prefix="tf", include_ai_button=True)

    _cached_report = st.session_state.get('find_new_trades_report', {}) or {}
    _cached_ts = float(st.session_state.get('_find_new_trades_ts', 0.0) or 0.0)
    _cached_age = max(0.0, time.time() - _cached_ts) if _cached_ts else None
    _cached_ttl_sec = float(st.session_state.get('find_new_cache_ttl_min', 20.0) or 20.0) * 60.0
    _cached_is_fresh = bool(_cached_age is not None and _cached_age <= _cached_ttl_sec)
    _cached_analyzed = int(_cached_report.get('results_count', 0) or 0)
    _cached_candidates = int(_cached_report.get('candidate_count', 0) or 0)
    _cached_scope_match = can_auto_rerank_cached_fn(st.session_state, _cached_report) if _cached_report else False
    if _cached_ts:
        st.caption(
            f"Cached Find New snapshot: analyzed={_cached_analyzed}, candidates={_cached_candidates}, "
            f"age={format_eta_fn(_cached_age or 0)} ({'fresh' if _cached_is_fresh else 'stale'}) | "
            f"scope {'match' if _cached_scope_match else 'changed'}."
        )
    else:
        st.caption("Cached Find New snapshot: none yet.")

    _b1, _b2, _b3 = st.columns([3, 2, 2])
    with _b1:
        if st.button(
            "🧭 Find New Trades",
            type="primary",
            width="stretch",
            key="tf_find_btn",
            disabled=(_run_active or bool(_data_gate.get('hard_block', False))),
            help="Starts a non-blocking background full scan with live phase progress and ETA.",
        ):
            _job_id = start_trade_finder_background_run_fn(rerank_only=False)
            if _job_id:
                st.rerun()
            st.session_state['trade_finder_last_status'] = {
                'level': 'warning',
                'message': "Unable to start background Trade Finder run.",
                'ts': time.time(),
            }
    with _b2:
        _rerank_label = "🕒 Queue Re-rank" if _bg_full_active else "⚡ Re-rank Cached"
        _rerank_help = (
            "Queue one rerank to run immediately after the active background full scan completes."
            if _bg_full_active else
            "Re-ranks the existing Find New snapshot without any data fetch."
        )
        if st.button(
            _rerank_label,
            width="stretch",
            key="tf_rerank_cached_btn",
            disabled=(not bool(_cached_report)) or (_run_active and not _bg_full_active),
            help=_rerank_help,
        ):
            if _bg_full_active:
                st.session_state['trade_finder_bg_rerank_queued'] = True
                st.session_state['trade_finder_last_status'] = {
                    'level': 'info',
                    'message': "Queued one background rerank after current full scan.",
                    'ts': time.time(),
                }
                st.rerun()
            else:
                _job_id = start_trade_finder_background_run_fn(rerank_only=True)
                if _job_id:
                    st.rerun()
                st.session_state['trade_finder_last_status'] = {
                    'level': 'warning',
                    'message': "Unable to start background re-rank.",
                    'ts': time.time(),
                }
    with _b3:
        if st.button(
            "⏹ Stop + Clear Queue",
            width="stretch",
            key="tf_stop_clear_queue_btn",
            disabled=(not _run_active),
            help="Requests stop at next safe checkpoint and clears queued reranks/auto-rerank requests.",
        ):
            if _bg_active:
                request_trade_finder_background_cancel_fn()
            else:
                st.session_state['trade_finder_cancel_requested'] = True
                st.session_state['trade_finder_last_status'] = {
                    'level': 'warning',
                    'message': "Stop requested. Foreground workflow will halt at the next checkpoint.",
                    'ts': time.time(),
                }
            st.session_state['trade_finder_bg_rerank_queued'] = False
            st.session_state.pop('trade_gate_auto_rerank_requested', None)
            st.rerun()
        if _bg_active and _bg_preview_rows:
            if st.button(
                "✅ Use Best Current Results Now",
                key="tf_use_best_current_now_btn",
                width="stretch",
                help="Load current background preview into Trade Finder cards/table immediately while run continues.",
            ):
                _curr = st.session_state.get('trade_finder_results', {}) or {}
                st.session_state['trade_finder_results'] = {
                    'run_id': str(_curr.get('run_id', '') or str((_bg_job or {}).get('id', '') or 'bg_live')),
                    'generated_at_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': list(_bg_preview_rows),
                    'provider': str(_curr.get('provider', 'system') or 'system'),
                    'run_mode': 'background_preview',
                    'elapsed_sec': float((_bg_job or {}).get('run_elapsed_sec', 0.0) or 0.0),
                    'input_candidates': int((_bg_job or {}).get('total', 0) or len(_bg_preview_rows)),
                }
                st.session_state['trade_finder_last_status'] = {
                    'level': 'info',
                    'message': (
                        f"Loaded {len(_bg_preview_rows)} live preview row(s). "
                        "Background run is still active and will keep improving results."
                    ),
                    'ts': time.time(),
                }
                st.rerun()

    _last_status = st.session_state.get('trade_finder_last_status', {}) or {}
    _last_level = str(_last_status.get('level', 'info') or 'info').lower()
    _last_message = str(_last_status.get('message', '') or '').strip()
    if _last_message:
        if _last_level == "success":
            st.success(_last_message)
        elif _last_level == "warning":
            st.warning(_last_message)
        elif _last_level == "error":
            st.error(_last_message)
        else:
            st.info(_last_message)

    return TradeFinderTopState(
        alert_tickers=alert_tickers,
        data_gate=dict(_data_gate or {}),
        bg_job=dict(_bg_job or {}),
        bg_status=_bg_status,
        bg_mode=_bg_mode,
        bg_full_active=bool(_bg_full_active),
        bg_active=bool(_bg_active),
        bg_preview_rows=list(_bg_preview_rows or []),
        run_active=bool(_run_active),
        cancel_requested=bool(_cancel_requested),
    )
