"""
TTA v2 Position Sizer — Institutional-Grade Risk Engine
=========================================================

Pure logic module. NO UI code. NO data fetching.
Accepts inputs, returns sizing recommendations with full audit trail.

Rules:
- Max 1.5% equity risk per trade
- Max 8% portfolio heat (total risk across all open positions)
- Max 20% capital concentration in one stock
- Dynamic scaling: reduce size on losing streaks

Version: 1.0.0 (2026-02-12)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# =============================================================================
# CONSTANTS — Risk limits (can be overridden per call)
# =============================================================================

DEFAULT_MAX_RISK_PER_TRADE = 1.5    # % of equity
DEFAULT_MAX_PORTFOLIO_HEAT = 8.0    # % total risk across all positions
DEFAULT_MAX_CONCENTRATION = 20.0    # % of equity in one stock
DEFAULT_MAX_POSITIONS = 8

# Losing streak scaling
STREAK_SCALE = {
    0: 1.0,     # No streak — full size
    1: 1.0,     # 1 loss — still full
    2: 0.75,    # 2 losses — scale to 75%
    3: 0.50,    # 3 losses — scale to 50%
    4: 0.25,    # 4+ losses — scale to 25%
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SizingResult:
    """Complete output of the position sizer with full audit trail."""
    ticker: str
    entry_price: float
    stop_price: float
    target_price: float

    # Computed shares from each constraint
    shares_from_risk: int = 0       # Max risk per trade limit
    shares_from_heat: int = 0       # Portfolio heat limit
    shares_from_concentration: int = 0  # Concentration limit
    shares_from_capital: int = 0    # Available capital limit

    # Final recommendation
    recommended_shares: int = 0
    limiting_factor: str = ''       # Which constraint was binding
    position_cost: float = 0
    risk_dollars: float = 0
    risk_pct_of_equity: float = 0

    # Risk metrics
    stop_distance_pct: float = 0
    reward_risk_ratio: float = 0
    potential_profit: float = 0
    potential_loss: float = 0

    # Scaling adjustments
    base_shares: int = 0            # Before any scaling
    scale_factor: float = 1.0
    scale_reason: str = ''          # Why scaled down (if applicable)

    # Portfolio context
    portfolio_heat_before: float = 0
    portfolio_heat_after: float = 0
    concentration_pct: float = 0
    open_positions_count: int = 0

    # Warnings and explanations
    warnings: List[str] = field(default_factory=list)
    explanation: str = ''           # Human-readable summary

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# CORE SIZING LOGIC
# =============================================================================

def calculate_position_size(
    ticker: str,
    entry_price: float,
    stop_price: float,
    target_price: float = 0,
    account_size: float = 100000,
    open_positions: List[Dict] = None,
    recent_win_rate: float = 0.5,
    current_losing_streak: int = 0,
    max_risk_pct: float = DEFAULT_MAX_RISK_PER_TRADE,
    max_heat_pct: float = DEFAULT_MAX_PORTFOLIO_HEAT,
    max_concentration_pct: float = DEFAULT_MAX_CONCENTRATION,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> SizingResult:
    """
    Calculate optimal position size given all risk constraints.

    Parameters:
    -----------
    ticker : str
    entry_price : float
    stop_price : float
    target_price : float (optional, for R:R calculation)
    account_size : float
    open_positions : list of dicts with keys:
        {ticker, entry_price, shares, position_size, current_stop}
    recent_win_rate : float (0-1, from JournalManager)
    current_losing_streak : int (from JournalManager)
    max_risk_pct : float (default 1.5%)
    max_heat_pct : float (default 8%)
    max_concentration_pct : float (default 20%)

    Returns:
    --------
    SizingResult with full audit trail
    """
    open_positions = open_positions or []

    result = SizingResult(
        ticker=ticker,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        open_positions_count=len(open_positions),
    )

    # ── Validate inputs ───────────────────────────────────────────────
    if entry_price <= 0 or stop_price <= 0:
        result.warnings.append("Invalid entry or stop price")
        result.explanation = "Cannot size: invalid prices."
        return result

    if stop_price >= entry_price:
        result.warnings.append("Stop must be below entry for long positions")
        result.explanation = "Cannot size: stop >= entry."
        return result

    # ── Basic risk math ───────────────────────────────────────────────
    risk_per_share = entry_price - stop_price
    result.stop_distance_pct = round(risk_per_share / entry_price * 100, 2)

    if target_price > entry_price:
        reward_per_share = target_price - entry_price
        result.reward_risk_ratio = round(reward_per_share / risk_per_share, 2)

    # ── Constraint 1: Max Risk Per Trade (1.5%) ───────────────────────
    risk_budget = account_size * (max_risk_pct / 100)
    shares_from_risk = int(risk_budget / risk_per_share)
    result.shares_from_risk = shares_from_risk

    # ── Constraint 2: Portfolio Heat (8%) ─────────────────────────────
    current_heat = _calculate_portfolio_heat(open_positions, account_size)
    result.portfolio_heat_before = round(current_heat, 2)

    remaining_heat_budget = max(0, (max_heat_pct / 100) * account_size - current_heat * account_size / 100)
    if remaining_heat_budget > 0 and risk_per_share > 0:
        shares_from_heat = int(remaining_heat_budget / risk_per_share)
    else:
        shares_from_heat = 0
    result.shares_from_heat = shares_from_heat

    # ── Constraint 3: Concentration (20%) ─────────────────────────────
    max_position_dollars = account_size * (max_concentration_pct / 100)
    shares_from_concentration = int(max_position_dollars / entry_price)
    result.shares_from_concentration = shares_from_concentration

    # ── Constraint 4: Available Capital ───────────────────────────────
    deployed = sum(float(p.get('position_size', 0)) for p in open_positions)
    available = account_size - deployed
    shares_from_capital = max(0, int(available / entry_price)) if available > 0 else 0
    result.shares_from_capital = shares_from_capital

    # ── Apply most conservative constraint ────────────────────────────
    constraints = {
        'risk_per_trade': shares_from_risk,
        'portfolio_heat': shares_from_heat,
        'concentration': shares_from_concentration,
        'available_capital': shares_from_capital,
    }

    # Filter out zero (would mean that constraint blocks entirely)
    base_shares = min(constraints.values()) if all(v >= 0 for v in constraints.values()) else 0
    result.base_shares = base_shares

    # Identify limiting factor
    if base_shares <= 0:
        # Find which constraint is blocking
        if shares_from_heat <= 0:
            result.limiting_factor = 'portfolio_heat'
        elif shares_from_capital <= 0:
            result.limiting_factor = 'available_capital'
        else:
            result.limiting_factor = 'unknown'
    else:
        for name, val in constraints.items():
            if val == base_shares:
                result.limiting_factor = name
                break

    # ── Dynamic Scaling: Losing Streak ────────────────────────────────
    scale = _get_streak_scale(current_losing_streak)
    result.scale_factor = scale

    if scale < 1.0:
        result.scale_reason = (
            f"Scaled to {scale:.0%} due to {current_losing_streak}-trade losing streak"
        )

    # Apply scale
    final_shares = max(0, int(base_shares * scale))
    result.recommended_shares = final_shares

    # ── Compute final metrics ─────────────────────────────────────────
    if final_shares > 0:
        result.position_cost = round(final_shares * entry_price, 2)
        result.risk_dollars = round(final_shares * risk_per_share, 2)
        result.risk_pct_of_equity = round(result.risk_dollars / account_size * 100, 2)
        result.potential_loss = round(final_shares * risk_per_share, 2)

        if target_price > entry_price:
            result.potential_profit = round(final_shares * (target_price - entry_price), 2)

        result.concentration_pct = round(result.position_cost / account_size * 100, 2)

        # Portfolio heat after
        new_heat_dollars = current_heat * account_size / 100 + result.risk_dollars
        result.portfolio_heat_after = round(new_heat_dollars / account_size * 100, 2)

    # ── Generate warnings ─────────────────────────────────────────────
    if result.stop_distance_pct > 8:
        result.warnings.append("Stop distance very wide (>8%) — consider tighter stop")
    elif result.stop_distance_pct > 5:
        result.warnings.append("Stop distance wider than typical (>5%)")

    if result.reward_risk_ratio > 0 and result.reward_risk_ratio < 1.5:
        result.warnings.append(f"R:R below 1.5:1 ({result.reward_risk_ratio}:1) — not worth the risk")
    elif result.reward_risk_ratio > 0 and result.reward_risk_ratio < 2.0:
        result.warnings.append(f"R:R below ideal 2:1 ({result.reward_risk_ratio}:1)")

    if result.portfolio_heat_after > max_heat_pct:
        result.warnings.append(f"Portfolio heat would exceed {max_heat_pct}% limit")

    if len(open_positions) >= max_positions:
        result.warnings.append(f"Already at max positions ({max_positions})")

    if recent_win_rate < 0.4 and len(open_positions) > 0:
        result.warnings.append(f"Win rate below 40% ({recent_win_rate:.0%}) — consider reducing exposure")

    # ── Build explanation string ──────────────────────────────────────
    result.explanation = _build_explanation(result, max_risk_pct, max_heat_pct, max_concentration_pct)

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_portfolio_heat(open_positions: List[Dict], account_size: float) -> float:
    """
    Calculate current portfolio heat (total risk across all open positions)
    as a percentage of account size.

    For each position: risk = shares × (entry_price - current_stop)
    """
    if not open_positions or account_size <= 0:
        return 0.0

    total_risk = 0.0
    for pos in open_positions:
        entry = float(pos.get('entry_price', 0))
        stop = float(pos.get('current_stop', pos.get('initial_stop', 0)))
        shares = float(pos.get('shares', 0))

        if entry > 0 and stop > 0 and shares > 0:
            risk = shares * (entry - stop)
            if risk > 0:
                total_risk += risk

    return round(total_risk / account_size * 100, 2)


def _get_streak_scale(losing_streak: int) -> float:
    """Get position size scale factor based on losing streak."""
    if losing_streak <= 0:
        return 1.0
    if losing_streak >= 4:
        return STREAK_SCALE[4]
    return STREAK_SCALE.get(losing_streak, 1.0)


def _build_explanation(result: SizingResult, max_risk_pct, max_heat_pct, max_conc_pct) -> str:
    """Build human-readable explanation of the sizing decision."""
    parts = []

    if result.recommended_shares <= 0:
        if result.limiting_factor == 'portfolio_heat':
            return (f"❌ Cannot open position: Portfolio heat at {result.portfolio_heat_before:.1f}% "
                    f"(max {max_heat_pct}%). Close or tighten existing stops first.")
        elif result.limiting_factor == 'available_capital':
            return "❌ Cannot open position: No available capital."
        return "❌ Cannot size position with current constraints."

    parts.append(f"**{result.recommended_shares:,} shares** @ ${result.entry_price:.2f} "
                 f"= **${result.position_cost:,.0f}**")

    # Show limiting factor
    factor_names = {
        'risk_per_trade': f'{max_risk_pct}% risk-per-trade limit',
        'portfolio_heat': f'{max_heat_pct}% portfolio heat limit',
        'concentration': f'{max_conc_pct}% concentration limit',
        'available_capital': 'available capital',
    }
    parts.append(f"Limited by: {factor_names.get(result.limiting_factor, result.limiting_factor)}")

    # Show scaling
    if result.scale_factor < 1.0:
        parts.append(f"⚠️ Base size was {result.base_shares:,} shares, "
                     f"reduced to {result.recommended_shares:,} — {result.scale_reason}")

    # Risk summary
    parts.append(f"Risk: ${result.risk_dollars:,.0f} ({result.risk_pct_of_equity:.1f}% of equity) "
                 f"| Heat: {result.portfolio_heat_before:.1f}% → {result.portfolio_heat_after:.1f}%")

    return " | ".join(parts)
