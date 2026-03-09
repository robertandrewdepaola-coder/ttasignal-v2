"""
TTA v2 — Executive Dashboard Business Logic
=============================================

Pure business-logic functions extracted from the executive dashboard in app.py.
Zero Streamlit imports — these functions take all context via parameters and
return plain dicts / dataclasses so the UI layer can render however it likes.

Functions extracted:
    build_dashboard_snapshot  — assemble the DashboardSnapshot from raw inputs
    evaluate_trade_gate       — single authority on whether new trades are allowed
    position_posture_summary  — macro guidance for existing positions
    run_auto_exit_engine      — evaluate open positions vs stops/targets, auto-close
    scan_health_snapshot      — compute scan telemetry from audit events

Version: 1.0.0 (2026-03-09)
"""

import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from exec_logic import (
    DashboardSnapshot,
    RiskBudgetPolicy,
    TradeGateDecision,
    infer_exec_regime,
    risk_budget_for_regime,
    normalize_brief_regime,
    stale_stream_status,
)
from journal_manager import JournalManager


# =============================================================================
# 1) DASHBOARD SNAPSHOT BUILDER
# =============================================================================

def build_dashboard_snapshot(
    *,
    jm: JournalManager,
    watchlist_tickers: List[str],
    scan_summary: List[Dict[str, Any]],
    market_filter: Dict[str, Any],
    sector_rotation: Dict[str, Any],
    earnings_flags: Dict[str, Any],
    triggered_alerts: List[Dict[str, Any]],
    scan_run_ts: float = 0.0,
    find_new_ts: float = 0.0,
    trade_finder_iso: str = "",
    market_filter_ts: float = 0.0,
    sector_rotation_ts: float = 0.0,
    position_prices_ts: float = 0.0,
    alert_check_ts: float = 0.0,
    daily_workflow_ts: float = 0.0,
    daily_workflow_sec: float = 0.0,
) -> DashboardSnapshot:
    """
    Build a single source-of-truth DashboardSnapshot from raw inputs.

    All session-state reads are done by the caller (app.py) and passed in here
    so this function stays Streamlit-free.
    """
    now = time.time()

    regime, confidence = infer_exec_regime(market_filter, sector_rotation)
    policy = risk_budget_for_regime(regime)

    # Determine effective scan timestamp (most recent of scan/find-new/trade-finder)
    tf_ts = 0.0
    if trade_finder_iso:
        try:
            tf_ts = datetime.strptime(trade_finder_iso, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            tf_ts = 0.0
    effective_scan_ts = max(scan_run_ts, find_new_ts, tf_ts)

    return DashboardSnapshot(
        generated_at=now,
        generated_at_iso=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
        watchlist_tickers=watchlist_tickers,
        scan_summary=scan_summary,
        open_trades=jm.get_open_trades(),
        pending_alerts=jm.get_pending_conditionals(),
        triggered_alerts=triggered_alerts,
        market_filter=market_filter,
        sector_rotation=sector_rotation,
        earnings_flags=earnings_flags,
        scan_ts=effective_scan_ts,
        market_ts=market_filter_ts,
        sector_ts=sector_rotation_ts,
        pos_ts=position_prices_ts,
        alert_ts=alert_check_ts,
        workflow_ts=daily_workflow_ts,
        workflow_sec=daily_workflow_sec,
        regime=regime,
        regime_confidence=confidence,
        risk_policy=policy,
    )


# =============================================================================
# 2) TRADE GATE DECISION
# =============================================================================

def evaluate_trade_gate(
    snap: DashboardSnapshot,
    *,
    deep_analysis: Optional[Dict[str, Any]] = None,
    deep_analysis_date: str = "",
    morning_narrative: Optional[Dict[str, Any]] = None,
    morning_narrative_date: str = "",
) -> TradeGateDecision:
    """
    Single execution authority for whether new trades should be taken now.

    Parameters
    ----------
    snap : DashboardSnapshot
    deep_analysis : dict or None — today's deep market analysis (from session state)
    deep_analysis_date : str — date string for the deep analysis
    morning_narrative : dict or None — today's morning narrative (from session state)
    morning_narrative_date : str — date string for the morning narrative
    """
    vix = float(snap.market_filter.get("vix_close", 0) or 0)
    spy_ok = bool(snap.market_filter.get("spy_above_200", True))

    stale = stale_stream_status(snap)
    stale_count = int(stale["count"])
    stale_core = (
        bool(stale.get("market", False))
        or (bool(stale.get("scan", False)) and bool(stale.get("sector", False)))
    )

    deep = deep_analysis or {}
    brief = morning_narrative or {}
    today = datetime.now().strftime("%Y-%m-%d")

    has_deep = isinstance(deep, dict) and bool(deep) and deep_analysis_date == today and ("score" in deep)
    has_brief = isinstance(brief, dict) and bool(brief) and morning_narrative_date == today and bool(str(brief.get("regime", "")).strip())

    deep_score = 0
    if has_deep:
        deep_score = int(deep.get("score", 0) or 0)
    brief_regime = normalize_brief_regime(brief.get("regime", "")) if has_brief else "UNKNOWN"

    # Model alignment check
    aligned = 0
    compared = 0
    if has_brief and brief_regime != "UNKNOWN":
        compared += 1
        if brief_regime == snap.regime:
            aligned += 1
    if has_deep:
        compared += 1
        if (deep_score >= 1 and snap.regime == "RISK_ON") or (deep_score <= -1 and snap.regime == "RISK_OFF"):
            aligned += 1
        elif deep_score == 0 and snap.regime in {"TRANSITION", "DEFENSIVE"}:
            aligned += 1

    if compared == 0:
        model_alignment = "UNAVAILABLE"
    elif aligned == compared:
        model_alignment = "ALIGNED"
    elif aligned == 0:
        model_alignment = "DIVERGENT"
    else:
        model_alignment = "PARTIAL"

    # Base decision from unified regime + volatility
    if stale_core:
        status = "NO_TRADE"
        reason = "Core data is stale (market/scan context). Refresh before opening new trades."
    elif snap.regime == "RISK_OFF" or vix >= 25:
        status = "NO_TRADE"
        reason = f"Risk-off environment (VIX {vix:.1f}). Preserve capital."
    elif snap.regime in {"DEFENSIVE", "TRANSITION"} or vix >= 20 or not spy_ok:
        status = "TRADE_LIGHT"
        reason = f"Mixed/cautious regime with elevated risk (VIX {vix:.1f})."
    else:
        status = "FAVOR_TRADING"
        reason = f"Benign risk backdrop and supportive trend (VIX {vix:.1f})."

    # Downgrade one notch when models diverge materially
    if model_alignment == "DIVERGENT":
        if status == "FAVOR_TRADING":
            status = "TRADE_LIGHT"
            reason += " Downgraded due to model divergence."
        elif status == "TRADE_LIGHT":
            status = "NO_TRADE"
            reason += " Downgraded due to model divergence."

    if status == "NO_TRADE":
        stale_gate = reason.startswith("Core data is stale")
        return TradeGateDecision(
            status=status,
            label=("🟠 DATA STALE — REFRESH REQUIRED" if stale_gate else "🛑 TOO RISKY TO TRADE"),
            allow_new_trades=False,
            severity=("warning" if stale_gate else "danger"),
            reason=reason,
            model_alignment=model_alignment,
        )
    if status == "TRADE_LIGHT":
        return TradeGateDecision(
            status=status,
            label="🟡 TRADE LIGHT (SELECTIVE)",
            allow_new_trades=True,
            severity="warning",
            reason=reason,
            model_alignment=model_alignment,
        )
    return TradeGateDecision(
        status=status,
        label="🟢 MARKET FAVORS TRADING",
        allow_new_trades=True,
        severity="success",
        reason=reason,
        model_alignment=model_alignment,
    )


# =============================================================================
# 3) POSITION POSTURE SUMMARY
# =============================================================================

def position_posture_summary(
    snap: DashboardSnapshot,
    gate: TradeGateDecision,
    position_prices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Clear macro guidance for EXISTING positions, separate from new-entry gate.

    Parameters
    ----------
    snap : DashboardSnapshot
    gate : TradeGateDecision
    position_prices : dict  — {TICKER: current_price} from session state
    """
    stop_breaches = 0
    drawdown_count = 0
    for trade in (snap.open_trades or []):
        ticker = str(trade.get("ticker", "")).upper().strip()
        entry = float(trade.get("entry_price", 0) or 0)
        stop = float(trade.get("current_stop", trade.get("initial_stop", 0)) or 0)
        current = float(position_prices.get(ticker) or entry)
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        if stop > 0 and current <= stop:
            stop_breaches += 1
        elif pnl_pct <= -3:
            drawdown_count += 1

    vix = float(snap.market_filter.get("vix_close", 0) or 0)
    severe_macro = snap.regime == "RISK_OFF" or vix >= 30

    if gate.status == "NO_TRADE":
        if severe_macro:
            headline = "🛑 Macro Posture: Defensive hold/reduce risk"
            summary = "No new trades. Keep winners with disciplined stops; reduce weakest positions into strength."
            severity = "danger"
        else:
            headline = "🛡️ Macro Posture: Hold existing positions only"
            summary = "No new trades. Manage open positions with stops; do not exit everything just because entry gate is closed."
            severity = "warning"
    elif gate.status == "TRADE_LIGHT":
        headline = "🟡 Macro Posture: Hold and be selective"
        summary = "Maintain current positions; add risk only on top-quality setups and smaller size."
        severity = "warning"
    else:
        headline = "🟢 Macro Posture: Market supports holding and selective adds"
        summary = "Trend backdrop is supportive. Hold existing positions and allow new entries that meet rules."
        severity = "success"

    if stop_breaches > 0:
        summary += f" Immediate action: {stop_breaches} position(s) at/below stop."
    elif drawdown_count > 0:
        summary += f" Watchlist: {drawdown_count} position(s) in >3% drawdown."

    return {
        "headline": headline,
        "summary": summary,
        "severity": severity,
        "stop_breaches": stop_breaches,
        "drawdowns": drawdown_count,
    }


# =============================================================================
# 4) AUTO EXIT ENGINE (pure logic)
# =============================================================================

def run_auto_exit_engine(
    jm: JournalManager,
    current_prices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Evaluate open positions against stops/targets and auto-close breaches.

    Returns a summary dict — the caller is responsible for session-state updates,
    audit events, and perf metrics.
    """
    if not current_prices:
        return {"checked": 0, "triggered": 0, "closed": 0, "events": [], "stop_hits": 0, "target_hits": 0}

    open_trades = jm.get_open_trades()
    checked = len(open_trades)
    events = jm.check_stops(current_prices, auto_execute=True)
    closed = len(events)
    stop_hits = sum(1 for e in events if str(e.get("trigger", "")) == "stop_loss")
    target_hits = sum(1 for e in events if str(e.get("trigger", "")) == "target_hit")

    return {
        "checked": checked,
        "triggered": closed,
        "closed": closed,
        "events": events,
        "stop_hits": stop_hits,
        "target_hits": target_hits,
    }


# =============================================================================
# 5) SCAN HEALTH SNAPSHOT (pure logic)
# =============================================================================

def scan_health_snapshot(
    audit_events: List[Dict[str, Any]],
    scan_duration_hist: List[float],
) -> Dict[str, float]:
    """
    Compute scan telemetry from audit events + duration history.

    Parameters
    ----------
    audit_events : list — raw audit event list from session state
    scan_duration_hist : list — list of scan durations from session state
    """
    scan_done = [e for e in audit_events if e.get("action") == "SCAN_DONE"]
    scan_start = [e for e in audit_events if e.get("action") == "SCAN_START"]

    last_mode = ""
    last_count = 0
    if scan_done:
        details = str(scan_done[0].get("details", ""))
        m_mode = re.search(r"mode=([a-z_]+)", details)
        m_count = re.search(r"scanned=(\d+)", details)
        if m_mode:
            last_mode = m_mode.group(1)
        if m_count:
            last_count = int(m_count.group(1))

    last_dur = float(scan_duration_hist[-1]) if scan_duration_hist else 0.0
    avg_dur = (sum(scan_duration_hist) / len(scan_duration_hist)) if scan_duration_hist else 0.0

    return {
        "total_scans_logged": float(len(scan_done)),
        "scan_starts_logged": float(len(scan_start)),
        "last_scan_count": float(last_count),
        "last_scan_duration": last_dur,
        "avg_scan_duration": avg_dur,
        "last_scan_mode": last_mode,
    }
