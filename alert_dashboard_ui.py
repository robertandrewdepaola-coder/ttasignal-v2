"""
TTA v2 Alert Dashboard UI — Dedicated Alerts Panel
=====================================================

Streamlit UI module for the 🎯 Alerts tab.
Renders pending, waiting, and triggered alert cards using
data from alert_manager (pure logic) + journal_manager (persistence).

This module ONLY does rendering. All business logic is in alert_manager.py.

Requires two navigation callbacks from app.py:
- open_chart_fn(ticker) -> bool
- open_trade_fn(ticker) -> bool

Version: 1.0.0 (2026-03-09)
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from journal_manager import JournalManager
from alert_manager import (
    build_alert_status_rows,
    format_alert_timestamp,
    get_alert_summary,
    AlertStatusRow,
    AlertSummary,
)


# =============================================================================
# MAIN ENTRY POINT — called by app.py as a @st.fragment
# =============================================================================

def render_alerts_dashboard(
    jm: JournalManager,
    fetch_price_fn: Callable[[str], Optional[float]],
    *,
    open_chart_fn: Optional[Callable[[str], bool]] = None,
    open_trade_fn: Optional[Callable[[str], bool]] = None,
    triggered_limit: int = 20,
):
    """
    Render the complete alerts dashboard.

    Parameters:
    -----------
    jm : JournalManager — for data access
    fetch_price_fn : callable(ticker) -> float — for live prices
    open_chart_fn : callable(ticker) -> bool — navigate to chart (optional)
    open_trade_fn : callable(ticker) -> bool — navigate to trade tab (optional)
    triggered_limit : int — how many triggered alerts to show in history
    """
    # Build alert status rows (sorted: ready first, then by proximity)
    status_rows = build_alert_status_rows(jm, fetch_price_fn)
    ready_rows = [r for r in status_rows if r.is_ready]
    waiting_rows = [r for r in status_rows if not r.is_ready]
    triggered_hist = jm.get_triggered_conditionals(limit=triggered_limit) or []

    # ── Empty state ──────────────────────────────────────────────────
    if not status_rows and not triggered_hist:
        st.info("No alerts yet. Set alerts from the trade tab when analyzing a ticker.")
        return

    # ── Summary metrics ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Active", len(status_rows))
    with m2:
        st.metric("Ready Now", len(ready_rows))
    with m3:
        st.metric("Waiting", len(waiting_rows))
    with m4:
        st.metric("Triggered (recent)", len(triggered_hist))

    st.caption("🟢 READY = trigger price reached | 🟡 WAITING = approaching trigger | Sorted by proximity")

    # ── Ready to trade section ───────────────────────────────────────
    if ready_rows:
        st.success(f"**{len(ready_rows)} alert(s) ready to trade now**")
        for row in ready_rows:
            _render_alert_card(
                row, section="ready", jm=jm,
                open_chart_fn=open_chart_fn,
                open_trade_fn=open_trade_fn,
            )
    else:
        st.info("No alerts are trade-ready right now.")

    # ── Waiting section ──────────────────────────────────────────────
    if waiting_rows:
        with st.expander(f"🗂️ Waiting Alerts ({len(waiting_rows)})", expanded=True):
            for row in waiting_rows:
                _render_alert_card(
                    row, section="wait", jm=jm,
                    open_chart_fn=open_chart_fn,
                    open_trade_fn=open_trade_fn,
                )
    else:
        st.caption("No waiting alerts.")

    # ── Triggered history ────────────────────────────────────────────
    st.divider()
    with st.expander(f"✅ Triggered Alerts (recent {len(triggered_hist)})", expanded=False):
        if not triggered_hist:
            st.caption("No triggered alerts recorded yet.")
        else:
            for idx, cond in enumerate(triggered_hist):
                _render_triggered_card(
                    cond, idx=idx,
                    open_chart_fn=open_chart_fn,
                    open_trade_fn=open_trade_fn,
                )


# =============================================================================
# CARD RENDERERS — Individual alert cards
# =============================================================================

def _render_alert_card(
    row: AlertStatusRow,
    *,
    section: str,
    jm: JournalManager,
    open_chart_fn: Optional[Callable] = None,
    open_trade_fn: Optional[Callable] = None,
):
    """Render one pending/waiting alert card."""
    ticker = row.ticker
    trigger = row.trigger
    cond_type = row.condition_type
    current = row.current_price
    delta_pct = row.delta_pct
    placed_text = format_alert_timestamp(row.created_at)

    # Expiry info
    expires_text = ""
    days_left = None
    if row.expires_date:
        try:
            exp_dt = datetime.strptime(row.expires_date[:10], "%Y-%m-%d")
            days_left = (exp_dt - datetime.now()).days
            if days_left <= 3:
                expires_text = f"⚠️ Expires in {days_left}d"
            elif days_left <= 7:
                expires_text = f"Expires in {days_left}d"
            else:
                expires_text = f"Expires {row.expires_date[:10]}"
        except Exception:
            expires_text = f"Expires {row.expires_date[:10]}"

    # Type label
    type_labels = {
        'breakout_above': '📈 Breakout Above',
        'breakout_volume': '📊 Breakout + Volume',
        'pullback_to': '📉 Pullback To',
    }
    type_label = type_labels.get(cond_type, cond_type)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1.5, 1.8, 0.8])

        with c1:
            status_line = row.state_label
            st.markdown(f"**{ticker}**  \n{status_line}")
            # Show thesis/notes if present
            if row.notes:
                _clean_notes = row.notes.strip()
                if len(_clean_notes) > 120:
                    _clean_notes = _clean_notes[:120] + "..."
                st.caption(_clean_notes)

        with c2:
            st.caption(f"Trigger: **${trigger:.2f}**")
            st.caption(f"Type: {type_label}")
            if row.stop_price > 0 or row.target_price > 0:
                parts = []
                if row.stop_price > 0:
                    parts.append(f"Stop: ${row.stop_price:.2f}")
                if row.target_price > 0:
                    parts.append(f"Target: ${row.target_price:.2f}")
                st.caption(" | ".join(parts))

        with c3:
            st.caption(f"Current: **${current:.2f}** ({delta_pct:+.1f}%)")
            st.caption(f"Placed: {placed_text}")
            if expires_text:
                st.caption(expires_text)
            if row.conviction > 0 or row.quality_grade:
                meta_parts = []
                if row.conviction > 0:
                    meta_parts.append(f"Conv: {row.conviction}/10")
                if row.quality_grade:
                    meta_parts.append(f"Quality: {row.quality_grade}")
                st.caption(" | ".join(meta_parts))

        with c4:
            if st.button("📊", key=f"alert_view_{section}_{ticker}", help="View chart", width="stretch"):
                if open_chart_fn:
                    if not open_chart_fn(ticker):
                        st.warning(f"Unable to load {ticker}.")
                    else:
                        st.rerun()

            trade_disabled = (section == "wait")
            if st.button("✅", key=f"alert_trade_{section}_{ticker}", help="Open trade tab", width="stretch", disabled=trade_disabled):
                if open_trade_fn:
                    if not open_trade_fn(ticker):
                        st.warning(f"Unable to load {ticker}.")
                    else:
                        st.rerun()

            if st.button("✕", key=f"rm_alert_{section}_{ticker}", help="Remove alert", width="stretch"):
                jm.remove_conditional(ticker)
                st.rerun()


def _render_triggered_card(
    cond: Dict[str, Any],
    *,
    idx: int,
    open_chart_fn: Optional[Callable] = None,
    open_trade_fn: Optional[Callable] = None,
):
    """Render one triggered alert history card."""
    ticker = str(cond.get('ticker', '')).upper().strip()
    trigger = float(cond.get('trigger_price', 0) or 0)
    triggered_price = float(cond.get('triggered_price', 0) or 0)
    placed_text = format_alert_timestamp(cond.get('created_at') or cond.get('created_date'))
    triggered_text = format_alert_timestamp(cond.get('triggered_at') or cond.get('triggered_date'))
    move_pct = ((triggered_price - trigger) / trigger * 100) if trigger > 0 and triggered_price > 0 else 0.0

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 2.0, 1.0])

        with c1:
            st.markdown(f"**{ticker}**  \n🟢 TRIGGERED — eligible to take trade")
            notes = str(cond.get('notes', '') or '').strip()
            if notes:
                st.caption(notes[:140])

        with c2:
            st.caption(f"Trigger: ${trigger:.2f} | Triggered: ${triggered_price:.2f} ({move_pct:+.1f}%)")
            st.caption(f"Placed: {placed_text} | Triggered: {triggered_text}")

        with c3:
            if st.button("📊", key=f"trig_view_{ticker}_{idx}", help="View chart", width="stretch"):
                if open_chart_fn:
                    if not open_chart_fn(ticker):
                        st.warning(f"Unable to load {ticker}.")
                    else:
                        st.rerun()

            if st.button("✅", key=f"trig_trade_{ticker}_{idx}", help="Open trade tab", width="stretch"):
                if open_trade_fn:
                    if not open_trade_fn(ticker):
                        st.warning(f"Unable to load {ticker}.")
                    else:
                        st.rerun()
