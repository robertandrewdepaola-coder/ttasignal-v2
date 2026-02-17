"""
Trade Decision Card model.

Unifies candidate context across scanner, executive dashboard, and trade finder.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class TradeDecisionCard:
    ticker: str
    source: str
    recommendation: str
    ai_buy_recommendation: str
    conviction: int
    quality_grade: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    rank_score: float
    regime: str
    gate_status: str
    regime_fit_score: float
    earnings_penalty: float
    execution_readiness: str
    reason: str
    ai_rationale: str
    explainability: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _calc_rr(entry: float, stop: float, target: float) -> float:
    if entry <= 0 or stop <= 0 or target <= 0:
        return 0.0
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(reward / risk, 2)


def _regime_fit(regime: str, gate_status: str, sector_phase: str = "") -> float:
    score = 50.0
    regime_u = str(regime or "").upper()
    gate_u = str(gate_status or "").upper()
    phase_u = str(sector_phase or "").upper()
    if regime_u == "RISK_ON":
        score += 20
    elif regime_u in {"TRANSITION", "DEFENSIVE"}:
        score -= 5
    else:
        score -= 20

    if gate_u == "FAVOR_TRADING":
        score += 15
    elif gate_u == "TRADE_LIGHT":
        score -= 10
    elif gate_u == "NO_TRADE":
        score -= 25

    if phase_u == "LEADING":
        score += 10
    elif phase_u == "EMERGING":
        score += 5
    elif phase_u == "LAGGING":
        score -= 10
    return max(0.0, min(100.0, round(score, 1)))


def _execution_readiness(gate_status: str, rr: float, conviction: int) -> str:
    gate_u = str(gate_status or "").upper()
    if gate_u == "NO_TRADE":
        return "BLOCKED"
    if rr >= 2.0 and conviction >= 7:
        return "READY"
    if rr >= 1.2 and conviction >= 5:
        return "WATCH"
    return "LOW_QUALITY"


def build_trade_decision_card(
    *,
    ticker: str,
    source: str,
    recommendation: str,
    ai_buy_recommendation: str,
    conviction: int,
    quality_grade: str,
    entry: float,
    stop: float,
    target: float,
    rank_score: float,
    regime: str,
    gate_status: str,
    reason: str = "",
    ai_rationale: str = "",
    sector_phase: str = "",
    earn_days: int = 999,
    explainability_bits: List[str] = None,
) -> TradeDecisionCard:
    rr = _calc_rr(entry, stop, target)
    earn_pen = 20.0 if 0 <= int(earn_days) <= 3 else (10.0 if int(earn_days) <= 7 else 0.0)
    fit = _regime_fit(regime, gate_status, sector_phase=sector_phase)
    readiness = _execution_readiness(gate_status, rr, conviction)
    explain = " | ".join([x for x in (explainability_bits or []) if x][:5])
    return TradeDecisionCard(
        ticker=str(ticker or "").upper().strip(),
        source=str(source or "scanner"),
        recommendation=str(recommendation or ""),
        ai_buy_recommendation=str(ai_buy_recommendation or ""),
        conviction=_as_int(conviction, 0),
        quality_grade=str(quality_grade or "?"),
        entry=round(_as_float(entry, 0.0), 2),
        stop=round(_as_float(stop, 0.0), 2),
        target=round(_as_float(target, 0.0), 2),
        risk_reward=rr,
        rank_score=round(_as_float(rank_score, 0.0), 2),
        regime=str(regime or "UNKNOWN"),
        gate_status=str(gate_status or ""),
        regime_fit_score=fit,
        earnings_penalty=earn_pen,
        execution_readiness=readiness,
        reason=str(reason or ""),
        ai_rationale=str(ai_rationale or ""),
        explainability=explain,
    )
