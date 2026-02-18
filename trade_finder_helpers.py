"""Helper builders for Trade Finder -> New Trade and staged tickets."""

from __future__ import annotations

from typing import Any, Dict

from journal_manager import PlannedTrade



def build_trade_finder_selection(row: Dict[str, Any], generated_at_iso: str, run_id: str) -> Dict[str, Any]:
    """Normalize selected candidate payload for New Trade prefill."""
    return {
        "ticker": str(row.get("ticker", "") or "").upper().strip(),
        "entry": float(row.get("suggested_entry", row.get("price", 0)) or row.get("price", 0) or 0),
        "stop": float(row.get("suggested_stop_loss", 0) or 0),
        "target": float(row.get("suggested_target", 0) or 0),
        "ai_buy_recommendation": str(row.get("ai_buy_recommendation", "") or ""),
        "risk_reward": float(row.get("risk_reward", 0) or 0),
        "earn_days": int(row.get("earn_days", 999) or 999),
        "reason": str(row.get("reason", "") or ""),
        "ai_rationale": str(row.get("ai_rationale", "") or ""),
        "provider": str(row.get("provider", "system") or "system"),
        "generated_at_iso": str(generated_at_iso or ""),
        "trade_finder_run_id": str(row.get("trade_finder_run_id", "") or run_id or ""),
    }



def build_planned_trade(row: Dict[str, Any], run_id: str) -> PlannedTrade:
    """Build PlannedTrade from Trade Finder candidate row."""
    price = float(row.get("price", 0) or 0)
    rr = float(row.get("risk_reward", 0) or 0)
    return PlannedTrade(
        plan_id="",
        ticker=str(row.get("ticker", "") or "").upper().strip(),
        status="PLANNED",
        source="trade_finder",
        entry=float(row.get("suggested_entry", price) or price or 0),
        stop=float(row.get("suggested_stop_loss", 0) or 0),
        target=float(row.get("suggested_target", 0) or 0),
        risk_reward=rr,
        ai_recommendation=str(row.get("ai_buy_recommendation", "") or ""),
        rank_score=float(row.get("rank_score", 0) or 0),
        trade_finder_run_id=str(row.get("trade_finder_run_id", "") or run_id or ""),
        reason=str(row.get("reason", "") or ""),
        notes=str(row.get("ai_rationale", row.get("scanner_summary", "")) or ""),
    )


def compute_trade_score(row: Dict[str, Any]) -> float:
    """
    Execution-aware score for ranking candidates in Trade Finder.

    Keeps model score but adjusts for practical execution quality:
    - AI verdict quality
    - Risk/reward
    - Decision readiness
    - Earnings proximity
    """
    base = float(row.get("rank_score", 0) or 0)
    verdict = str(row.get("ai_buy_recommendation", "") or "").strip()
    rr = float(row.get("risk_reward", 0) or 0)
    earn_days = int(row.get("earn_days", 999) or 999)
    readiness = str((row.get("decision_card", {}) or {}).get("execution_readiness", "") or "").upper()

    verdict_adj = {
        "Strong Buy": 1.5,
        "Buy": 0.8,
        "Watch Only": -0.4,
        "Skip": -1.2,
    }.get(verdict, 0.0)
    rr_adj = max(-0.5, min(1.5, (rr - 1.5) * 0.4))
    ready_adj = 0.6 if readiness == "READY" else -0.3
    earn_adj = -0.8 if 0 <= earn_days <= 3 else (-0.3 if 4 <= earn_days <= 7 else 0.0)

    return round(base + verdict_adj + rr_adj + ready_adj + earn_adj, 2)
