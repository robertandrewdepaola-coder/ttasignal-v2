"""
TTA v2 Alert Manager — Centralized Alert & Conditional Entry Logic
===================================================================

ALL alert creation, evaluation, and status logic lives here.
No other module should build ConditionalEntry objects or check trigger states.

This is a PURE LOGIC module:
- NO Streamlit imports
- NO data fetching (prices come in as arguments)
- NO UI rendering

Call sites (app.py, trade_finder_ui, etc.) pass in prices and get back
structured results they can render however they like.

Extracted from app.py functions:
- _set_alert_for_candidate          → create_alert_from_candidate
- _set_breakout_alert_from_resistance → create_breakout_alert
- _render_quick_alert_form logic    → create_alert_from_form
- _run_alert_check_now              → check_alerts_now
- _build_pending_alert_rows         → build_alert_status_rows

Version: 1.0.0 (2026-03-09)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from journal_manager import JournalManager, ConditionalEntry


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_EXPIRY_DAYS = 30
DEFAULT_VOLUME_MULTIPLIER = 1.5
MAX_NOTE_LENGTH = 240


# =============================================================================
# DATA CLASSES — Structured results for callers
# =============================================================================

@dataclass
class AlertStatusRow:
    """One row in the alert status view — ready to render by any UI."""
    ticker: str
    trigger: float
    condition_type: str
    current_price: float
    delta_pct: float
    is_ready: bool
    state_label: str
    created_at: str
    stop_price: float
    target_price: float
    conviction: int
    quality_grade: str
    notes: str
    expires_date: str
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'trigger': self.trigger,
            'condition_type': self.condition_type,
            'current_price': self.current_price,
            'delta_pct': self.delta_pct,
            'is_ready': self.is_ready,
            'state_label': self.state_label,
            'created_at': self.created_at,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'conviction': self.conviction,
            'quality_grade': self.quality_grade,
            'notes': self.notes,
            'expires_date': self.expires_date,
        }


@dataclass
class AlertCheckResult:
    """Result of a batch alert check."""
    pending_count: int
    triggered_count: int
    triggered: List[Dict[str, Any]]
    prices_fetched: int
    errors: List[str]


@dataclass
class AlertSummary:
    """Quick counts for UI badges / headers."""
    total_pending: int
    ready_count: int
    waiting_count: int
    triggered_recent: int

    @property
    def label(self) -> str:
        if self.total_pending > 0:
            return f"🎯 Alerts ({self.total_pending})"
        return "🎯 Alerts"


# =============================================================================
# ALERT CREATION — Build ConditionalEntry from various sources
# =============================================================================

def create_alert_from_candidate(
    jm: JournalManager,
    candidate: Dict[str, Any],
    *,
    price: float = 0.0,
    risk_reward: float = 0.0,
    reason: str = "",
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """
    Create an alert from a Trade Finder candidate row.

    Extracted from app.py _set_alert_for_candidate().
    Returns status message string.
    """
    ticker = str(candidate.get('ticker', '')).upper().strip()
    if not ticker:
        return "Cannot set alert: missing ticker."

    entry_price = float(candidate.get('suggested_entry', price) or price or 0)
    resistance = float(candidate.get('resistance_price', 0) or 0)

    # Trigger priority: resistance > entry + 0.5%
    trigger = resistance if resistance > 0 else 0.0
    if trigger <= 0 and entry_price > 0:
        trigger = round(entry_price * 1.005, 2)
    if trigger <= 0:
        return f"Cannot set alert for {ticker}: missing valid trigger price."

    stop = float(candidate.get('suggested_stop_loss', 0) or 0)
    target = float(candidate.get('suggested_target', 0) or 0)
    conviction = int(candidate.get('conviction', 0) or 0)
    quality = str(candidate.get('quality_grade', '') or '')
    sig_type = str(candidate.get('recommendation', '') or '')
    expires = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')

    note = (
        f"[Trade Finder] trigger={trigger:.2f} rr={float(risk_reward):.2f}:1 "
        f"reason={str(reason or '')[:160]}"
    ).strip()

    entry_obj = ConditionalEntry(
        ticker=ticker,
        condition_type='breakout_above',
        trigger_price=float(trigger),
        volume_multiplier=DEFAULT_VOLUME_MULTIPLIER,
        stop_price=float(stop),
        target_price=float(target),
        conviction=conviction,
        signal_type=sig_type,
        quality_grade=quality,
        expires_date=expires,
        notes=note,
    )
    return jm.add_conditional(entry_obj)


def create_breakout_alert(
    jm: JournalManager,
    *,
    ticker: str,
    trigger_price: float,
    stop_price: float = 0.0,
    target_price: float = 0.0,
    conviction: int = 0,
    quality_grade: str = "?",
    notes: str = "",
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """
    Create a breakout-volume alert at major overhead resistance.

    Extracted from app.py _set_breakout_alert_from_resistance().
    Returns status message string.
    """
    t = str(ticker or '').upper().strip()
    if not t:
        return "Cannot set alert: missing ticker."

    try:
        trig = float(trigger_price or 0)
    except (ValueError, TypeError):
        trig = 0.0
    if trig <= 0:
        return f"Cannot set alert for {t}: missing valid trigger price."

    try:
        stop_val = float(stop_price or 0)
    except (ValueError, TypeError):
        stop_val = 0.0
    try:
        target_val = float(target_price or 0)
    except (ValueError, TypeError):
        target_val = 0.0

    q = str(quality_grade or '?').strip().upper()[:1] if str(quality_grade or '').strip() else '?'
    c = max(0, min(10, int(conviction or 0)))

    entry = ConditionalEntry(
        ticker=t,
        condition_type='breakout_volume',
        trigger_price=round(trig, 2),
        volume_multiplier=DEFAULT_VOLUME_MULTIPLIER,
        stop_price=round(stop_val, 2) if stop_val > 0 else 0.0,
        target_price=round(target_val, 2) if target_val > 0 else 0.0,
        conviction=c,
        quality_grade=q,
        notes=str(notes or '').strip()[:MAX_NOTE_LENGTH],
        expires_date=(datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d'),
    )
    return jm.add_conditional(entry)


def create_alert_from_form(
    jm: JournalManager,
    *,
    ticker: str,
    condition_type: str = 'breakout_above',
    trigger_price: float = 0.0,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
    expires_date: str = "",
    stop_price: float = 0.0,
    target_price: float = 0.0,
    conviction: int = 0,
    quality_grade: str = "",
    notes: str = "",
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """
    Create an alert from user form input (quick alert form).

    Extracted from app.py _render_quick_alert_form() logic.
    Returns status message string.
    """
    t = str(ticker or '').upper().strip()
    if not t:
        return "Cannot set alert: missing ticker."

    try:
        trig = float(trigger_price or 0)
    except (ValueError, TypeError):
        trig = 0.0
    if trig <= 0:
        return f"Cannot set alert for {t}: missing valid trigger price."

    ctype = str(condition_type or 'breakout_above').strip()
    if ctype not in {'breakout_above', 'pullback_to', 'breakout_volume'}:
        ctype = 'breakout_above'

    if not expires_date:
        expires_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')

    entry = ConditionalEntry(
        ticker=t,
        condition_type=ctype,
        trigger_price=round(trig, 2),
        volume_multiplier=float(volume_multiplier or DEFAULT_VOLUME_MULTIPLIER),
        stop_price=round(float(stop_price or 0), 2),
        target_price=round(float(target_price or 0), 2),
        conviction=max(0, min(10, int(conviction or 0))),
        quality_grade=str(quality_grade or '').strip(),
        expires_date=expires_date,
        notes=str(notes or '').strip()[:MAX_NOTE_LENGTH],
    )
    return jm.add_conditional(entry)


# =============================================================================
# ALERT EVALUATION — Check alerts against live prices
# =============================================================================

def check_alerts_now(
    jm: JournalManager,
    fetch_price_fn: Callable[[str], Optional[float]],
) -> AlertCheckResult:
    """
    Evaluate all pending conditionals against current prices.

    Extracted from app.py _run_alert_check_now().

    Parameters:
    -----------
    jm : JournalManager
    fetch_price_fn : callable(ticker) -> float or None
        Caller provides the price-fetching function so this module
        stays free of data_fetcher imports.

    Returns AlertCheckResult with triggered list and diagnostics.
    """
    conditionals = jm.get_pending_conditionals()
    if not conditionals:
        return AlertCheckResult(
            pending_count=0,
            triggered_count=0,
            triggered=[],
            prices_fetched=0,
            errors=[],
        )

    current_prices: Dict[str, float] = {}
    errors: List[str] = []

    for cond in conditionals:
        ticker = str(cond.get('ticker', '')).upper().strip()
        if not ticker or ticker in current_prices:
            continue
        try:
            price = fetch_price_fn(ticker)
            if price and price > 0:
                current_prices[ticker] = float(price)
        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:100]}")

    triggered = jm.check_conditionals(current_prices, volume_ratios={})

    return AlertCheckResult(
        pending_count=len(conditionals),
        triggered_count=len(triggered),
        triggered=triggered,
        prices_fetched=len(current_prices),
        errors=errors,
    )


# =============================================================================
# ALERT STATUS — Build display-ready rows with live state
# =============================================================================

def build_alert_status_rows(
    jm: JournalManager,
    fetch_price_fn: Callable[[str], Optional[float]],
) -> List[AlertStatusRow]:
    """
    Build alert status rows with live readiness state.

    Extracted from app.py _build_pending_alert_rows().
    Sorted: ready alerts first (by delta%), then waiting (by proximity).
    """
    conditionals = jm.get_pending_conditionals()
    if not conditionals:
        return []

    rows: List[AlertStatusRow] = []

    for cond in conditionals:
        ticker = str(cond.get('ticker', '')).upper().strip()
        if not ticker:
            continue

        trigger = float(cond.get('trigger_price', 0) or 0)
        cond_type = str(cond.get('condition_type', 'breakout_above') or 'breakout_above')

        # Fetch current price via caller-provided function
        current = 0.0
        try:
            price = fetch_price_fn(ticker)
            current = float(price) if price and price > 0 else 0.0
        except Exception:
            pass

        # Determine readiness
        is_ready = False
        if cond_type in {'breakout_above', 'breakout_volume'} and trigger > 0 and current >= trigger:
            is_ready = True
        elif cond_type == 'pullback_to' and trigger > 0 and current <= trigger:
            is_ready = True

        delta_pct = ((current - trigger) / trigger * 100) if trigger > 0 else 0.0

        if is_ready:
            state_label = "🟢 READY TO TRADE"
        elif cond_type == 'pullback_to':
            state_label = "🟡 ABOVE ALERT (waiting)"
        else:
            state_label = "🟡 BELOW ALERT (waiting)"

        rows.append(AlertStatusRow(
            ticker=ticker,
            trigger=trigger,
            condition_type=cond_type,
            current_price=current,
            delta_pct=round(delta_pct, 2),
            is_ready=is_ready,
            state_label=state_label,
            created_at=str(cond.get('created_at') or cond.get('created_date') or ''),
            stop_price=float(cond.get('stop_price', 0) or 0),
            target_price=float(cond.get('target_price', 0) or 0),
            conviction=int(cond.get('conviction', 0) or 0),
            quality_grade=str(cond.get('quality_grade', '') or ''),
            notes=str(cond.get('notes', '') or ''),
            expires_date=str(cond.get('expires_date', '') or ''),
            raw=cond,
        ))

    # Sort: ready first (biggest delta), then waiting (closest to trigger)
    ready = sorted(
        [r for r in rows if r.is_ready],
        key=lambda r: abs(r.delta_pct),
        reverse=True,
    )
    waiting = sorted(
        [r for r in rows if not r.is_ready],
        key=lambda r: abs(r.delta_pct),
    )
    return ready + waiting


def get_alert_summary(
    jm: JournalManager,
    fetch_price_fn: Optional[Callable[[str], Optional[float]]] = None,
    triggered_limit: int = 20,
) -> AlertSummary:
    """
    Quick summary counts for UI badges.

    If fetch_price_fn is provided, determines ready vs waiting.
    Otherwise just counts pending.
    """
    pending = jm.get_pending_conditionals()
    triggered = jm.get_triggered_conditionals(limit=triggered_limit)

    if fetch_price_fn and pending:
        rows = build_alert_status_rows(jm, fetch_price_fn)
        ready_count = sum(1 for r in rows if r.is_ready)
        waiting_count = len(rows) - ready_count
    else:
        ready_count = 0
        waiting_count = len(pending)

    return AlertSummary(
        total_pending=len(pending),
        ready_count=ready_count,
        waiting_count=waiting_count,
        triggered_recent=len(triggered),
    )


def get_alert_tickers(jm: JournalManager) -> Set[str]:
    """Return set of tickers with pending alerts — for badge display in scan tables."""
    return {
        str(c.get('ticker', '')).upper().strip()
        for c in jm.get_pending_conditionals()
        if str(c.get('ticker', '')).strip()
    }


# =============================================================================
# ALERT HELPERS — Trigger price derivation
# =============================================================================

def derive_trigger_price(
    *,
    resistance_price: float = 0.0,
    entry_price: float = 0.0,
    current_price: float = 0.0,
    buffer_pct: float = 0.5,
) -> float:
    """
    Derive a trigger price from available data.

    Priority:
    1. Resistance price (if available)
    2. Entry price + buffer
    3. Current price + buffer
    4. 0.0 (caller should handle)
    """
    if resistance_price > 0:
        return round(resistance_price, 2)
    base = entry_price if entry_price > 0 else current_price
    if base > 0:
        return round(base * (1 + buffer_pct / 100), 2)
    return 0.0


def extract_resistance_info(overhead_resistance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract trigger price and description from signal overhead_resistance data.

    Extracted from app.py inline logic used in multiple places.
    """
    if not overhead_resistance:
        return {'trigger': 0.0, 'distance_pct': 0.0, 'volume_needed': 0.0,
                'description': '', 'assessment': ''}

    critical = overhead_resistance.get('critical_level', {}) or {}
    trigger = 0.0
    try:
        trigger = float(critical.get('price', 0) or 0)
    except (ValueError, TypeError):
        pass

    dist_pct = 0.0
    try:
        dist_pct = float(critical.get('distance_pct', 0) or 0)
    except (ValueError, TypeError):
        pass

    vol_need = 0.0
    try:
        vol_need = float(overhead_resistance.get('breakout_volume_needed', 0) or 0)
    except (ValueError, TypeError):
        pass

    return {
        'trigger': trigger,
        'distance_pct': dist_pct,
        'volume_needed': vol_need,
        'description': str(critical.get('description', '') or ''),
        'assessment': str(overhead_resistance.get('assessment', '') or ''),
    }


def format_alert_timestamp(raw: Any) -> str:
    """Format an alert timestamp for display. Handles ISO and date-only formats."""
    s = str(raw or "").strip()
    if not s:
        return "n/a"
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s[:19]
