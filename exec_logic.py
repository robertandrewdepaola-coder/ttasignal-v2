"""
TTA v2 — Executive Dashboard & Scoring Logic
===============================================

Pure business logic for regime inference, risk budgets, candidate scoring,
trade gate classification, and dashboard stale-stream detection.

Extracted from app.py to reduce main module size and improve rerun performance.

Version: 2.0.0 (2026-02-28)
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from earnings_sector_logic import is_earnings_data_trusted, calc_rr


# =============================================================================
# DATACLASSES — shared by dashboard, trade gate, and scoring
# =============================================================================

@dataclass
class RiskBudgetPolicy:
    regime: str
    max_new_trades: int
    position_size_multiplier: float
    max_sector_exposure: int
    max_total_open_positions: int
    note: str


@dataclass
class DashboardSnapshot:
    generated_at: float
    generated_at_iso: str
    watchlist_tickers: List[str]
    scan_summary: List[Dict[str, Any]]
    open_trades: List[Dict[str, Any]]
    pending_alerts: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    market_filter: Dict[str, Any]
    sector_rotation: Dict[str, Any]
    earnings_flags: Dict[str, Any]
    scan_ts: float
    market_ts: float
    sector_ts: float
    pos_ts: float
    alert_ts: float
    workflow_ts: float
    workflow_sec: float
    regime: str
    regime_confidence: int
    risk_policy: RiskBudgetPolicy


@dataclass
class TradeGateDecision:
    status: str   # NO_TRADE | TRADE_LIGHT | FAVOR_TRADING
    label: str
    allow_new_trades: bool
    severity: str  # danger | warning | success
    reason: str
    model_alignment: str  # ALIGNED | DIVERGENT | PARTIAL


# =============================================================================
# REGIME & RISK BUDGET
# =============================================================================

def infer_exec_regime(market_filter: Dict[str, Any], sector_rotation: Dict[str, Any]):
    """Unified regime inference for dashboard/risk policy. Returns (regime_str, confidence_score)."""
    spy_ok = bool(market_filter.get('spy_above_200', True))
    vix = float(market_filter.get('vix_close', 0) or 0)

    leading = 0
    emerging = 0
    lagging = 0
    for _name, info in (sector_rotation or {}).items():
        phase = str(info.get('phase', '')).lower()
        if phase == 'leading':
            leading += 1
        elif phase == 'emerging':
            emerging += 1
        elif phase == 'lagging':
            lagging += 1

    score = 0
    score += 35 if spy_ok else 5
    if vix < 20:
        score += 30
    elif vix < 25:
        score += 18
    else:
        score += 5
    score += min(25, leading * 5 + emerging * 3)
    score += max(0, 10 - lagging * 2)
    score = int(max(0, min(100, score)))

    if score >= 70:
        return "RISK_ON", score
    if score >= 50:
        return "TRANSITION", score
    if score >= 35:
        return "DEFENSIVE", score
    return "RISK_OFF", score


def risk_budget_for_regime(regime: str) -> RiskBudgetPolicy:
    """Translate regime into execution guardrails."""
    if regime == "RISK_ON":
        return RiskBudgetPolicy(regime, max_new_trades=6, position_size_multiplier=1.00, max_sector_exposure=4,
                                max_total_open_positions=12, note="Normal risk budget. Prioritize leading sectors.")
    if regime == "TRANSITION":
        return RiskBudgetPolicy(regime, max_new_trades=4, position_size_multiplier=0.75, max_sector_exposure=3,
                                max_total_open_positions=10, note="Selective adds only. Favor strongest setups.")
    if regime == "DEFENSIVE":
        return RiskBudgetPolicy(regime, max_new_trades=2, position_size_multiplier=0.50, max_sector_exposure=2,
                                max_total_open_positions=8, note="Capital preservation mode. Tight stops.")
    return RiskBudgetPolicy(regime, max_new_trades=1, position_size_multiplier=0.35, max_sector_exposure=1,
                            max_total_open_positions=6, note="Risk-off. Avoid new longs unless exceptional.")


def normalize_brief_regime(regime_text: str) -> str:
    r = str(regime_text or "").lower()
    if "risk-off" in r or "bearish" in r:
        return "RISK_OFF"
    if "neutral" in r or "balanced" in r or "caution" in r:
        return "TRANSITION"
    if "risk-on" in r or "bullish" in r:
        return "RISK_ON"
    return "UNKNOWN"


# =============================================================================
# GATE STATUS NORMALIZATION
# =============================================================================

def normalize_gate_status(status: str) -> str:
    """Normalize historical/variant gate strings to canonical status values."""
    s = str(status or "").upper().strip()
    if s in {"FAVOR_TRADING", "FAVORS_TRADING", "MARKET_FAVORS_TRADING"}:
        return "FAVOR_TRADING"
    if s in {"TRADE_LIGHT", "TRADE-LIGHT", "TRADE LIGHT"}:
        return "TRADE_LIGHT"
    if s in {"NO_TRADE", "NO-TRADE", "NO TRADE"}:
        return "NO_TRADE"
    return s or "NO_TRADE"


# =============================================================================
# CANDIDATE SCORING & CLASSIFICATION
# =============================================================================

def recommendation_score(row: Dict) -> float:
    """Simple ranking score for trade candidates."""
    grade_rank = {'A+': 8, 'A': 7, 'A-': 6, 'B+': 5, 'B': 4, 'B-': 3, 'C+': 2, 'C': 1}
    rec = str(row.get('recommendation', '')).upper()
    conv = float(row.get('conviction', 0) or 0)
    vol_ratio = float(row.get('volume_ratio', 0) or 0)
    grade = str(row.get('quality_grade', '')).upper()
    sector_phase = str(row.get('sector_phase', '')).lower()
    earn_days = int(row.get('earn_days', 999) or 999)
    earn_source = str(row.get('earn_source', '') or '').strip()
    earn_confidence = str(row.get('earn_confidence', '') or '').strip().upper()
    earn_date = str(row.get('earn_date', row.get('next_earnings', '')) or '').strip()
    earn_trusted = is_earnings_data_trusted(
        earn_days,
        source=earn_source,
        confidence=earn_confidence,
        next_earnings=earn_date,
    )

    score = conv * 3 + min(vol_ratio, 4) * 2 + grade_rank.get(grade, 0)
    if bool(row.get('apex_buy', False)):
        score += 2.5
    if "RE_ENTRY" in rec or "LATE_ENTRY" in rec:
        score += 1.0
    if "SKIP" in rec or "AVOID" in rec:
        score -= 8.0
    if sector_phase == "leading":
        score += 2.0
    elif sector_phase == "emerging":
        score += 1.0
    elif sector_phase == "lagging":
        score -= 1.5
    if 0 <= earn_days <= 7:
        score -= 2.0
    if not earn_trusted:
        score -= 2.5
    return score


def score_candidate_with_policy(row: Dict, snap: DashboardSnapshot) -> Dict[str, Any]:
    """Rank candidate and capture explainability tags + regime adjustments."""
    score = recommendation_score(row)
    reasons: List[str] = []
    rec = str(row.get('recommendation', '')).upper()
    conv = int(row.get('conviction', 0) or 0)
    phase = str(row.get('sector_phase', '')).lower()
    earn_days = int(row.get('earn_days', 999) or 999)
    earn_source = str(row.get('earn_source', '') or '').strip()
    earn_confidence = str(row.get('earn_confidence', '') or '').strip().upper()
    earn_date = str(row.get('earn_date', row.get('next_earnings', '')) or '').strip()
    earn_trusted = is_earnings_data_trusted(
        earn_days,
        source=earn_source,
        confidence=earn_confidence,
        next_earnings=earn_date,
    )

    if conv >= 8:
        reasons.append("High conviction")
    elif conv >= 6:
        reasons.append("Solid conviction")
    else:
        reasons.append("Lower conviction")

    if phase == 'leading':
        reasons.append("Leading sector")
    elif phase == 'emerging':
        reasons.append("Emerging sector")
    elif phase == 'lagging':
        reasons.append("Lagging sector")

    if not earn_trusted:
        reasons.append("Earnings unverified")
    elif earn_days <= 3:
        reasons.append("Earnings very near")
    elif earn_days <= 7:
        reasons.append("Earnings near")
    else:
        reasons.append("No immediate earnings")

    if snap.regime in {"DEFENSIVE", "RISK_OFF"}:
        score -= 2.5
        reasons.append(f"Regime {snap.regime}: tighter risk budget")
        if conv < 8:
            score -= 2.0
            reasons.append("Filtered by conviction under defensive regime")
    elif snap.regime == "TRANSITION":
        score -= 1.0
        reasons.append("Transition regime: selective entries")
    else:
        reasons.append("Risk-on support")

    blocked = False
    if "SKIP" in rec or "AVOID" in rec:
        blocked = True
        reasons.append("Recommendation is non-entry")
    if snap.regime == "RISK_OFF" and conv < 9:
        blocked = True
        reasons.append("Blocked by risk-off policy")
    if not earn_trusted:
        blocked = True
        reasons.append("Blocked by earnings trust policy")

    return {
        'score': score,
        'reasons': reasons,
        'blocked': blocked,
    }


def classify_trade_candidate_color(
    row: Dict[str, Any],
    gate_status: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Unified green/yellow/red candidate classification.
    settings: quality settings dict (min_rr, earn_block_days). Falls back to defaults.
    """
    cfg = settings or {'min_rr': 1.2, 'earn_block_days': 7}
    min_rr = float(cfg.get('min_rr', 1.2) or 1.2)
    earn_block_days = int(cfg.get('earn_block_days', 7) or 7)

    hard_pass = bool(row.get('hard_gate_pass', True))
    ai_buy_u = str(row.get('ai_buy_recommendation', '') or '').strip().upper()
    entry = float(row.get('suggested_entry', row.get('price', 0)) or 0)
    stop = float(row.get('suggested_stop_loss', 0) or 0)
    target = float(row.get('suggested_target', 0) or 0)
    rr = float(row.get('risk_reward', 0) or 0)
    if rr <= 0:
        rr = calc_rr(entry, stop, target)
    phase = str(row.get('sector_phase', '') or '').upper().strip()
    card = row.get('decision_card', {}) or {}
    readiness = str(card.get('execution_readiness', '') or '').upper().strip()

    earn_days = int(row.get('earn_days', 999) or 999)
    earn_source = str(row.get('earn_source', '') or '').strip()
    earn_confidence = str(row.get('earn_confidence', '') or '').strip().upper()
    earn_date = str(row.get('earn_date', row.get('earnings_date', '')) or '').strip()
    earn_trusted = is_earnings_data_trusted(
        earn_days,
        source=earn_source,
        confidence=earn_confidence,
        next_earnings=earn_date,
    )

    level = "YELLOW"
    reason = "Setup is partially aligned; keep on watch."

    _invalid_levels = (entry <= 0 or stop <= 0 or target <= 0 or not (stop < entry < target))
    _hard_block = (
        (not hard_pass)
        or _invalid_levels
        or (ai_buy_u in {"SKIP", "AVOID"})
        or (not earn_trusted)
        or (0 <= earn_days <= earn_block_days)
        or (readiness == "BLOCKED")
    )
    if _hard_block:
        level = "RED"
        if not hard_pass:
            reason = "Blocked by hard-gate rules."
        elif not earn_trusted:
            reason = "Blocked: earnings data is untrusted/unverified."
        elif 0 <= earn_days <= earn_block_days:
            reason = f"Blocked: earnings within {earn_block_days}d window."
        elif _invalid_levels:
            reason = "Blocked: invalid trade levels."
        else:
            reason = "Blocked by recommendation/readiness."
    else:
        _ready_setup = (
            ai_buy_u in {"STRONG BUY", "BUY"}
            and rr >= min_rr
            and readiness == "READY"
            and phase in {"LEADING", "EMERGING"}
        )
        if _ready_setup:
            level = "GREEN"
            reason = "Trade-ready setup: quality + timing + sector alignment."
        else:
            level = "YELLOW"
            reason = "Watch/conditional: waiting for stronger alignment."

    gate_u = normalize_gate_status(gate_status)
    if gate_u == "NO_TRADE":
        level = "RED"
        reason = "Market gate is NO_TRADE; new entries blocked."
    elif gate_u == "TRADE_LIGHT" and level == "GREEN":
        level = "YELLOW"
        reason = "Downgraded by TRADE_LIGHT market gate."

    if level == "GREEN":
        return {"level": "GREEN", "label": "TRADE-READY", "icon": "🟢", "color": "#22c55e", "reason": reason}
    if level == "RED":
        return {"level": "RED", "label": "BLOCKED", "icon": "🔴", "color": "#ef4444", "reason": reason}
    return {"level": "YELLOW", "label": "WATCH", "icon": "🟡", "color": "#f59e0b", "reason": reason}


# =============================================================================
# STALE STREAM & TIME HELPERS
# =============================================================================

def fmt_last_update(ts: float, fallback: str = "Never") -> str:
    """Format epoch timestamp to local date/time + age."""
    if not ts:
        return fallback
    dt = datetime.fromtimestamp(ts)
    age_sec = max(0, int(time.time() - ts))
    if age_sec < 60:
        age = f"{age_sec}s ago"
    elif age_sec < 3600:
        age = f"{age_sec // 60}m ago"
    elif age_sec < 86400:
        age = f"{age_sec // 3600}h ago"
    else:
        age = f"{age_sec // 86400}d ago"
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} ({age})"


def is_stale(ts: float, max_age_sec: int) -> bool:
    """True if timestamp is missing or older than max_age_sec."""
    if not ts:
        return True
    return (time.time() - ts) > max_age_sec


def stale_stream_status(snap: DashboardSnapshot) -> Dict[str, Any]:
    """
    Compute stale stream status with relevance gating.
    Positions/alerts only count as stale when they are actively in use.
    """
    stale_scan = is_stale(snap.scan_ts, 3 * 3600)
    stale_market = is_stale(snap.market_ts, 30 * 60)
    stale_sector = is_stale(snap.sector_ts, 60 * 60)
    stale_positions = bool(snap.open_trades) and is_stale(snap.pos_ts, 10 * 60)
    stale_alerts = bool(snap.pending_alerts) and is_stale(snap.alert_ts, 5 * 60)

    tags = []
    if stale_scan:
        tags.append("scan")
    if stale_market:
        tags.append("market")
    if stale_sector:
        tags.append("sectors")
    if stale_positions:
        tags.append("positions")
    if stale_alerts:
        tags.append("alerts")

    return {
        "scan": stale_scan,
        "market": stale_market,
        "sector": stale_sector,
        "positions": stale_positions,
        "alerts": stale_alerts,
        "tags": tags,
        "count": len(tags),
    }
