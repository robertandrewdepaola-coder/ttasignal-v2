"""
TTA v2 — Main Streamlit UI
============================

Single-mode interface. No Analysis/Journal toggle.
Flow: Watchlist → Scan → Click Ticker → Tabs (Signal, Chart, AI Intel, Trade Mgmt)

This is a THIN LAYER. All logic lives in the backend modules:
- signal_engine: calculations
- data_fetcher: yfinance calls
- scanner_engine: analysis & recommendations
- ai_analysis: AI-enhanced insights
- chart_engine: Plotly charts
- journal_manager: trade CRUD & P&L

Version: 2.0.0 (2026-02-08)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

# Backend imports
from signal_engine import EntrySignal
from data_fetcher import (
    fetch_all_ticker_data, fetch_scan_data, fetch_market_filter,
    fetch_current_price, fetch_daily, fetch_weekly, fetch_monthly,
    clear_cache,
)
from scanner_engine import analyze_ticker, scan_watchlist, TickerAnalysis
from ai_analysis import analyze as run_ai_analysis
from chart_engine import render_tv_chart, render_mtf_chart
from journal_manager import JournalManager, WatchlistItem, Trade, ConditionalEntry
from apex_signals import detect_apex_signals, get_apex_markers, get_apex_summary

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="TTA v2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize journal
if 'journal' not in st.session_state:
    st.session_state['journal'] = JournalManager(data_dir=".")

# Initialize scan results — restore from disk if available
if 'scan_results' not in st.session_state:
    jm_init = st.session_state['journal']
    saved_scan = jm_init.load_scan_results()
    if saved_scan and saved_scan.get('results'):
        # Restore minimal scan summary for display (not full TickerAnalysis objects)
        st.session_state['scan_results'] = []
        st.session_state['scan_results_summary'] = saved_scan['results']
        st.session_state['scan_timestamp'] = saved_scan.get('timestamp', '')
    else:
        st.session_state['scan_results'] = []
        st.session_state['scan_results_summary'] = []
        st.session_state['scan_timestamp'] = ''

if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = None

if 'selected_analysis' not in st.session_state:
    st.session_state['selected_analysis'] = None

if 'ticker_data_cache' not in st.session_state:
    st.session_state['ticker_data_cache'] = {}


def get_journal() -> JournalManager:
    return st.session_state['journal']


# =============================================================================
# =============================================================================
# SIDEBAR — Slim: Scan controls, Open Positions, Alerts, Market
# =============================================================================

def render_sidebar():
    jm = get_journal()

    st.sidebar.title("📊 TTA v2")
    st.sidebar.caption("Technical Trading Assistant")

    # ── Scan Controls ─────────────────────────────────────────────────
    watchlist_tickers = jm.get_watchlist_tickers()
    ticker_count = len(watchlist_tickers)

    # Count unscanned
    existing_summary = st.session_state.get('scan_results_summary', [])
    scanned = {s.get('ticker', '') for s in existing_summary}
    new_count = len([t for t in watchlist_tickers if t not in scanned])

    st.sidebar.caption(f"📋 {ticker_count} tickers in watchlist")

    scan_col1, scan_col2 = st.sidebar.columns(2)
    with scan_col1:
        if st.button("🔍 Scan All", use_container_width=True, type="primary",
                     disabled=(ticker_count == 0)):
            _run_scan(mode='all')
    with scan_col2:
        btn_label = f"⚡ New ({new_count})" if new_count > 0 else "⚡ New"
        if st.button(btn_label, use_container_width=True,
                     disabled=(new_count == 0)):
            _run_scan(mode='new_only')

    # ── Open Positions (clickable) ────────────────────────────────────
    open_trades = jm.get_open_trades()
    if open_trades:
        st.sidebar.divider()
        st.sidebar.subheader(f"📈 Open ({len(open_trades)})")
        for trade in open_trades:
            ticker = trade['ticker']
            entry = trade.get('entry_price', 0)
            current = fetch_current_price(ticker) or entry
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            icon = "🟢" if pnl_pct >= 0 else "🔴"

            if st.sidebar.button(
                f"{icon} {ticker}  ${current:.2f}  ({pnl_pct:+.1f}%)",
                key=f"sidebar_pos_{ticker}",
                use_container_width=True,
            ):
                _load_ticker_for_view(ticker)

    # ── Conditional Alerts ────────────────────────────────────────────
    conditionals = jm.get_pending_conditionals()
    if conditionals:
        st.sidebar.divider()
        st.sidebar.subheader(f"🎯 Alerts ({len(conditionals)})")
        for cond in conditionals:
            ticker = cond['ticker']
            trigger = cond.get('trigger_price', 0)
            current = fetch_current_price(ticker) or 0
            dist_pct = ((trigger - current) / current * 100) if current > 0 else 0

            label = f"⏳ {ticker}  ${trigger:.2f} ({dist_pct:+.1f}%)"
            col_a, col_b = st.sidebar.columns([3, 1])
            with col_a:
                if st.button(label, key=f"sidebar_cond_{ticker}", use_container_width=True):
                    _load_ticker_for_view(ticker)
            with col_b:
                if st.button("✕", key=f"rm_cond_{ticker}"):
                    jm.remove_conditional(ticker)
                    st.rerun()

    # ── Market Filter ─────────────────────────────────────────────────
    st.sidebar.divider()
    mkt = fetch_market_filter()
    spy_ok = mkt.get('spy_above_200', True)
    vix_ok = mkt.get('vix_below_30', True)
    st.sidebar.markdown(
        f"**Market:** SPY {'✅' if spy_ok else '❌'} ${mkt.get('spy_close', '?')} | "
        f"VIX {'✅' if vix_ok else '❌'} {mkt.get('vix_close', '?')}"
    )


def _load_ticker_for_view(ticker: str):
    """Load a ticker for the detail view — works for ANY ticker (open positions, conditionals, etc.)."""
    ticker = ticker.upper().strip()

    # Check if we already have analysis from a scan
    results = st.session_state.get('scan_results', [])
    for r in results:
        if r.ticker == ticker:
            st.session_state['selected_ticker'] = ticker
            st.session_state['selected_analysis'] = r
            st.rerun()
            return

    # Fetch fresh data and analyze on-the-fly
    try:
        data = fetch_all_ticker_data(ticker)
        if data.get('daily') is not None:
            analysis = analyze_ticker(data)
            st.session_state['selected_ticker'] = ticker
            st.session_state['selected_analysis'] = analysis
            # Cache the data
            cache = st.session_state.get('ticker_data_cache', {})
            cache[ticker] = data
            st.session_state['ticker_data_cache'] = cache
            st.rerun()
        else:
            st.sidebar.error(f"No data for {ticker}")
    except Exception as e:
        st.sidebar.error(f"Error loading {ticker}: {e}")


def _run_scan(mode='all'):
    """
    Execute watchlist scan with persistence and conditional checking.

    mode='all': Rescan everything (daily refresh, full watchlist)
    mode='new_only': Only scan tickers not in current results
    """
    jm = get_journal()
    all_watchlist = jm.get_watchlist_tickers()

    # Also include conditional alert tickers and open positions
    conditional_tickers = [c['ticker'] for c in jm.get_pending_conditionals()]
    open_tickers = jm.get_open_tickers()

    full_list = list(set(all_watchlist + conditional_tickers + open_tickers))

    if not full_list:
        st.sidebar.warning("Add tickers to watchlist first")
        return

    # Determine which tickers to scan
    if mode == 'new_only':
        existing_summary = st.session_state.get('scan_results_summary', [])
        already_scanned = {s.get('ticker', '') for s in existing_summary}
        tickers_to_scan = [t for t in full_list if t not in already_scanned]
        if not tickers_to_scan:
            st.sidebar.info("All tickers already scanned")
            return
    else:
        tickers_to_scan = full_list

    with st.spinner(f"Scanning {len(tickers_to_scan)} tickers..."):
        all_data = fetch_scan_data(tickers_to_scan)
        new_results = scan_watchlist(all_data)

        # Merge with existing results if new_only mode
        if mode == 'new_only':
            existing_results = st.session_state.get('scan_results', [])
            existing_tickers = {r.ticker for r in new_results}
            # Keep old results that aren't being rescanned
            merged_results = [r for r in existing_results if r.ticker not in existing_tickers]
            merged_results.extend(new_results)
            st.session_state['scan_results'] = merged_results

            # Merge data cache
            existing_cache = st.session_state.get('ticker_data_cache', {})
            existing_cache.update(all_data)
            st.session_state['ticker_data_cache'] = existing_cache
        else:
            st.session_state['scan_results'] = new_results
            st.session_state['ticker_data_cache'] = all_data

        st.session_state['selected_ticker'] = None
        st.session_state['selected_analysis'] = None

        # Build full summary (merge if new_only)
        results_for_summary = st.session_state['scan_results']

        summary = []
        for r in results_for_summary:
            rec = r.recommendation or {}
            q = r.quality or {}
            sig = r.signal
            summary.append({
                'ticker': r.ticker,
                'recommendation': rec.get('recommendation', 'SKIP'),
                'conviction': rec.get('conviction', 0),
                'quality_grade': q.get('quality_grade', '?'),
                'price': r.current_price,
                'summary': rec.get('summary', ''),
                'macd_bullish': sig.macd.get('bullish', False) if sig else False,
                'ao_positive': sig.ao.get('positive', False) if sig else False,
                'weekly_bullish': sig.weekly_macd.get('bullish', False) if sig else False,
                'monthly_bullish': sig.monthly_macd.get('bullish', False) if sig else False,
                'is_open_position': r.ticker in open_tickers,
            })
        jm.save_scan_results(summary)
        st.session_state['scan_results_summary'] = summary
        st.session_state['scan_timestamp'] = datetime.now().isoformat()

        # Check conditional alerts
        # Use all cached data (both old and new)
        full_cache = st.session_state.get('ticker_data_cache', {})
        current_prices = {}
        volume_ratios = {}
        for ticker, data in full_cache.items():
            daily = data.get('daily')
            if daily is not None and len(daily) > 0:
                current_prices[ticker] = float(daily['Close'].iloc[-1])
                if len(daily) > 50:
                    avg_vol = daily['Volume'].tail(50).mean()
                    if avg_vol > 0:
                        volume_ratios[ticker] = float(daily['Volume'].iloc[-1] / avg_vol)

        triggered = jm.check_conditionals(current_prices, volume_ratios)
        if triggered:
            st.session_state['triggered_alerts'] = triggered

    st.rerun()


# =============================================================================
# MAIN CONTENT — Scanner Results Table
# =============================================================================

def render_scanner_table():
    """Render scan results with watchlist management, filters, and click-to-view."""
    results = st.session_state.get('scan_results', [])
    summary = st.session_state.get('scan_results_summary', [])
    timestamp = st.session_state.get('scan_timestamp', '')
    jm = get_journal()

    # ══════════════════════════════════════════════════════════════════
    # TRIGGERED ALERTS BANNER
    # ══════════════════════════════════════════════════════════════════
    triggered = st.session_state.get('triggered_alerts', [])
    if triggered:
        for t in triggered:
            st.success(
                f"🎯 **BREAKOUT TRIGGERED: {t['ticker']}** — "
                f"Price ${t.get('triggered_price', 0):.2f} broke above "
                f"${t.get('trigger_price', 0):.2f} "
                f"(Volume: {t.get('triggered_volume_ratio', 0):.1f}x avg)"
            )
        st.session_state['triggered_alerts'] = []

    # ══════════════════════════════════════════════════════════════════
    # WATCHLIST EDITOR (collapsible)
    # ══════════════════════════════════════════════════════════════════
    watchlist_tickers = jm.get_watchlist_tickers()

    with st.expander(f"📋 Watchlist ({len(watchlist_tickers)} tickers) — click to edit",
                     expanded=(len(watchlist_tickers) == 0)):
        st.caption("Paste tickers separated by commas, spaces, or new lines. "
                   "Example: AAPL, NVDA, META, TSLA")

        # Bulk text area
        default_text = ", ".join(watchlist_tickers) if watchlist_tickers else ""
        new_text = st.text_area(
            "Tickers",
            value=default_text,
            height=100 if len(watchlist_tickers) > 20 else 68,
            label_visibility="collapsed",
            key="watchlist_editor",
        )

        wl_col1, wl_col2, wl_col3 = st.columns([1, 1, 2])
        with wl_col1:
            if st.button("💾 Save Watchlist", use_container_width=True, type="primary"):
                # Parse: handle commas, spaces, newlines, tabs
                import re
                raw = re.split(r'[,\s\n\t]+', new_text)
                tickers = [t.strip().upper() for t in raw if t.strip()]
                # Remove duplicates preserving order
                seen = set()
                unique = []
                for t in tickers:
                    if t not in seen:
                        seen.add(t)
                        unique.append(t)

                jm.clear_watchlist()
                jm.set_watchlist_tickers(unique)
                st.success(f"✅ Saved {len(unique)} tickers")
                st.rerun()

        with wl_col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                jm.clear_watchlist()
                st.session_state.pop('scan_results', None)
                st.session_state.pop('scan_results_summary', None)
                st.session_state.pop('ticker_data_cache', None)
                st.rerun()

        with wl_col3:
            st.caption(f"{len(watchlist_tickers)} saved"
                       + (f" • {len(summary)} scanned" if summary else ""))

    # ══════════════════════════════════════════════════════════════════
    # SCAN RESULTS
    # ══════════════════════════════════════════════════════════════════

    # No results at all
    if not results and not summary:
        st.info("👆 Add tickers to your watchlist (above) and click **Scan All** in the sidebar.")
        return

    # Build table from live scan results if available, else from persisted summary
    if results:
        rows = _build_rows_from_analysis(results, jm)
        source = "live"
    else:
        rows = _build_rows_from_summary(summary, jm)
        source = "restored"
        if timestamp:
            try:
                ts = datetime.fromisoformat(timestamp)
                age = datetime.now() - ts
                age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days > 0 else f"{age.seconds // 3600}h ago"
                st.caption(f"📌 Last scan: {ts.strftime('%Y-%m-%d %H:%M')} ({age_str}) — Scan All for fresh data")
            except Exception:
                st.caption("📌 Showing last saved scan results — Scan All for fresh data")

    if not rows:
        return

    # ── Filter Bar ────────────────────────────────────────────────────
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns([2, 2, 2, 2])

    with filt_col1:
        rec_filter = st.selectbox("Filter", [
            "All", "Signals Only", "BUY+", "Quality A-B", "Open Positions"
        ], key="scan_filter", label_visibility="collapsed")

    with filt_col2:
        sort_by = st.selectbox("Sort", [
            "Default", "Conviction ↓", "Quality ↓", "Price ↓", "Price ↑"
        ], key="scan_sort", label_visibility="collapsed")

    with filt_col3:
        search = st.text_input("Search", placeholder="Filter by ticker...",
                                key="scan_search", label_visibility="collapsed")

    with filt_col4:
        st.caption(f"**{len(rows)}** total results")

    # Apply filters
    filtered = rows.copy()

    if search:
        search_upper = search.upper()
        filtered = [r for r in filtered if search_upper in r['Ticker'].upper()]

    if rec_filter == "Signals Only":
        filtered = [r for r in filtered if r['Rec'] != 'SKIP']
    elif rec_filter == "BUY+":
        filtered = [r for r in filtered if r['Rec'] in ('BUY NOW', 'RE-ENTRY', 'FRESH ENTRY')]
    elif rec_filter == "Quality A-B":
        filtered = [r for r in filtered if r['Quality'] in ('A', 'B')]
    elif rec_filter == "Open Positions":
        filtered = [r for r in filtered if 'Open' in r.get('Status', '')]

    # Apply sort
    if sort_by == "Conviction ↓":
        filtered.sort(key=lambda r: int(r['Conv'].split('/')[0]) if '/' in r['Conv'] else 0, reverse=True)
    elif sort_by == "Quality ↓":
        q_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1, '?': 0}
        filtered.sort(key=lambda r: q_order.get(r.get('Quality', '?'), 0), reverse=True)
    elif sort_by == "Price ↓":
        filtered.sort(key=lambda r: float(r['Price'].replace('$', '').replace(',', '') or '0'), reverse=True)
    elif sort_by == "Price ↑":
        filtered.sort(key=lambda r: float(r['Price'].replace('$', '').replace(',', '') or '0'))

    showing = len(filtered)
    if showing != len(rows):
        st.caption(f"Showing {showing} of {len(rows)}")

    # ── Results as clickable ticker buttons ────────────────────────────
    if not filtered:
        st.info("No tickers match the current filter.")
        return

    # Table header
    hdr_cols = st.columns([1.2, 1, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 3])
    headers = ['Ticker', 'Rec', 'Conv', 'MACD', 'AO', 'Wkly', 'Mthly', 'Qlty', 'Price', 'Summary']
    for col, h in zip(hdr_cols, headers):
        col.markdown(f"**{h}**")

    # Table rows — each ticker is a button
    for idx, row in enumerate(filtered):
        cols = st.columns([1.2, 1, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 3])

        # Ticker as clickable button
        with cols[0]:
            ticker_label = row['Ticker']
            status = row.get('Status', '')
            if 'Open' in status:
                ticker_label = f"📈 {ticker_label}"
            elif 'Alert' in status:
                ticker_label = f"🎯 {ticker_label}"

            if st.button(ticker_label, key=f"row_{row['Ticker']}_{idx}",
                        use_container_width=True):
                _load_ticker_for_view(row['Ticker'])

        # Recommendation with color
        rec_val = row.get('Rec', 'SKIP')
        rec_colors = {
            'BUY NOW': '🟢', 'FRESH ENTRY': '🟢', 'RE-ENTRY': '🔵',
            'SKIP': '⚪', 'WAIT': '🟡',
        }
        rec_icon = rec_colors.get(rec_val, '⚪')
        cols[1].caption(f"{rec_icon} {rec_val}")
        cols[2].caption(row.get('Conv', '0/10'))
        cols[3].caption(row.get('MACD', '❌'))
        cols[4].caption(row.get('AO', '❌'))
        cols[5].caption(row.get('Wkly', '❌'))
        cols[6].caption(row.get('Mthly', '❌'))

        # Quality with color
        q = row.get('Quality', '?')
        q_colors = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🔴', 'F': '🔴'}
        cols[7].caption(f"{q_colors.get(q, '⚪')} {q}")

        cols[8].caption(row.get('Price', '?'))
        cols[9].caption(row.get('Summary', '')[:80])

    # ── Quick Actions for selected ticker ─────────────────────────────
    st.divider()

    # Inline quick-add ticker
    qa_col1, qa_col2 = st.columns([3, 1])
    with qa_col1:
        quick_ticker = st.text_input("Quick add ticker", placeholder="Type ticker and press Enter",
                                      key="quick_add_main", label_visibility="collapsed")
    with qa_col2:
        if st.button("➕ Add & Scan", use_container_width=True):
            if quick_ticker:
                ticker_clean = quick_ticker.strip().upper()
                wl = jm.get_watchlist_tickers()
                if ticker_clean not in wl:
                    jm.add_to_watchlist(WatchlistItem(ticker=ticker_clean))
                _load_ticker_for_view(ticker_clean)

    # Alert form (if requested from detail view)
    alert_ticker = st.session_state.get('show_alert_form')
    if alert_ticker:
        _render_quick_alert_form(alert_ticker, jm)


def _build_rows_from_analysis(results, jm) -> list:
    """Build table rows from live TickerAnalysis objects."""
    open_tickers = jm.get_open_tickers()
    conditional_tickers = [c['ticker'] for c in jm.get_pending_conditionals()]
    rows = []
    for r in results:
        rec = r.recommendation or {}
        q = r.quality or {}
        sig = r.signal

        # Status column
        if r.ticker in open_tickers:
            status = "📈 Open"
        elif r.ticker in conditional_tickers:
            status = "🎯 Alert"
        else:
            status = "👀"

        rows.append({
            'Ticker': r.ticker,
            'Status': status,
            'Rec': rec.get('recommendation', 'SKIP'),
            'Conv': f"{rec.get('conviction', 0)}/10",
            'MACD': "✅" if sig and sig.macd.get('bullish') else "❌",
            'AO': "✅" if sig and sig.ao.get('positive') else "❌",
            'Wkly': "✅" if sig and sig.weekly_macd.get('bullish') else "❌",
            'Mthly': "✅" if sig and sig.monthly_macd.get('bullish') else "❌",
            'Quality': q.get('quality_grade', '?'),
            'Price': f"${r.current_price:.2f}" if r.current_price else "?",
            'Summary': rec.get('summary', ''),
        })
    return rows


def _build_rows_from_summary(summary, jm) -> list:
    """Build table rows from persisted scan summary (cross-session)."""
    open_tickers = jm.get_open_tickers()
    conditional_tickers = [c['ticker'] for c in jm.get_pending_conditionals()]
    rows = []
    for s in summary:
        ticker = s.get('ticker', '?')
        if ticker in open_tickers:
            status = "📈 Open"
        elif ticker in conditional_tickers:
            status = "🎯 Alert"
        else:
            status = "👀"

        rows.append({
            'Ticker': ticker,
            'Status': status,
            'Rec': s.get('recommendation', 'SKIP'),
            'Conv': f"{s.get('conviction', 0)}/10",
            'MACD': "✅" if s.get('macd_bullish') else "❌",
            'AO': "✅" if s.get('ao_positive') else "❌",
            'Wkly': "✅" if s.get('weekly_bullish') else "❌",
            'Mthly': "✅" if s.get('monthly_bullish') else "❌",
            'Quality': s.get('quality_grade', '?'),
            'Price': f"${s.get('price', 0):.2f}" if s.get('price') else "?",
            'Summary': s.get('summary', ''),
        })
    return rows


def _render_quick_alert_form(ticker: str, jm: JournalManager):
    """Inline form to set a breakout/pullback alert."""
    st.markdown(f"### 🎯 Set Alert for {ticker}")

    current = fetch_current_price(ticker) or 0

    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        cond_type = st.selectbox("Type", ['breakout_above', 'pullback_to', 'breakout_volume'],
                                  key=f"alert_type_{ticker}")
    with ca2:
        # Default trigger: overhead resistance if available, else current + 3%
        default_trigger = current * 1.03 if current > 0 else 0
        # Try to get resistance from analysis
        analysis = st.session_state.get('selected_analysis')
        if analysis and analysis.signal and analysis.signal.overhead_resistance:
            ores = analysis.signal.overhead_resistance
            if ores.get('critical_level'):
                default_trigger = float(ores['critical_level']['price'])

        trigger = st.number_input("Trigger Price", value=default_trigger,
                                   step=0.01, format="%.2f", key=f"alert_trigger_{ticker}")
    with ca3:
        vol_mult = st.number_input("Volume (x avg)", value=1.5,
                                    min_value=1.0, max_value=5.0, step=0.1,
                                    key=f"alert_vol_{ticker}")
    with ca4:
        expires = st.date_input("Expires",
                                 value=datetime.now() + timedelta(days=30),
                                 key=f"alert_exp_{ticker}")

    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("✅ Set Alert", type="primary", key=f"set_alert_{ticker}"):
            entry = ConditionalEntry(
                ticker=ticker,
                condition_type=cond_type,
                trigger_price=trigger,
                volume_multiplier=vol_mult,
                expires_date=expires.strftime('%Y-%m-%d'),
                notes=f"Current: ${current:.2f}",
            )
            result = jm.add_conditional(entry)
            st.success(result)
            st.session_state.pop('show_alert_form', None)
            st.rerun()
    with ac2:
        if st.button("Cancel", key=f"cancel_alert_{ticker}"):
            st.session_state.pop('show_alert_form', None)
            st.rerun()


# =============================================================================
# DETAIL VIEW — Tabbed analysis for selected ticker
# =============================================================================

def render_detail_view():
    """Render detailed analysis for selected ticker."""
    analysis: TickerAnalysis = st.session_state.get('selected_analysis')
    if not analysis:
        return

    ticker = analysis.ticker
    signal = analysis.signal
    rec = analysis.recommendation or {}

    st.header(f"{ticker} — {rec.get('recommendation', 'SKIP')}")
    st.caption(rec.get('summary', ''))

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_signal, tab_chart, tab_ai, tab_trade = st.tabs([
        "📊 Signal", "📈 Chart", "🤖 AI Intel", "💼 Trade"
    ])

    # ── Signal Tab ────────────────────────────────────────────────────
    with tab_signal:
        _render_signal_tab(signal, analysis)

    # ── Chart Tab ─────────────────────────────────────────────────────
    with tab_chart:
        _render_chart_tab(ticker, signal)

    # ── AI Intel Tab ──────────────────────────────────────────────────
    with tab_ai:
        _render_ai_tab(ticker, signal, rec, analysis)

    # ── Trade Tab ─────────────────────────────────────────────────────
    with tab_trade:
        _render_trade_tab(ticker, signal, analysis)


def _render_signal_tab(signal: EntrySignal, analysis: TickerAnalysis):
    """Signal details and multi-timeframe status."""
    if not signal:
        st.warning("No signal data")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Daily")
        m = signal.macd
        st.metric("MACD", "Bullish ✅" if m.get('bullish') else "Bearish ❌",
                  f"Hist: {m.get('histogram', 0):+.4f}")
        if m.get('weakening'):
            st.warning("⚠ MACD histogram weakening")
        if m.get('near_cross'):
            st.warning("⚠ Near crossover")

        a = signal.ao
        st.metric("AO", "Positive ✅" if a.get('positive') else "Negative ❌",
                  f"Value: {a.get('value', 0):+.4f}")
        st.caption(f"Trend: {a.get('trend', '?')}")

    with col2:
        st.subheader("Weekly")
        w = signal.weekly_macd
        if w:
            st.metric("MACD", "Bullish ✅" if w.get('bullish') else "Bearish ❌",
                      f"Hist: {w.get('histogram', 0):+.4f}")
        else:
            st.info("No weekly data")

        st.subheader("Monthly")
        mo = signal.monthly_macd
        if mo:
            st.metric("MACD", "Bullish ✅" if mo.get('bullish') else "Bearish ❌",
                      f"Hist: {mo.get('histogram', 0):+.4f}")
        else:
            st.info("No monthly data")

    with col3:
        st.subheader("Context")
        ws = signal.weinstein
        st.metric("Weinstein", ws.get('label', '?'), ws.get('trend_maturity', ''))

        q = analysis.quality or {}
        st.metric("Quality", q.get('quality_grade', '?'),
                  f"Score: {q.get('quality_score', 0)}/100")
        st.caption(f"Win rate: {q.get('win_rate', 0):.0f}% | "
                   f"Signals: {q.get('signals_found', 0)}")

    # Overhead Resistance
    ores = signal.overhead_resistance
    if ores and ores.get('levels'):
        st.divider()
        st.subheader("Overhead Resistance")
        st.caption(ores.get('assessment', ''))
        for lev in ores['levels']:
            st.text(f"  {lev['description']}")

    # Key Levels
    kl = signal.key_levels
    if kl and kl.get('price'):
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("50 SMA", f"${kl.get('sma50', '?')}", kl.get('price_vs_sma50', ''))
        c2.metric("200 SMA", f"${kl.get('sma200', '?')}", kl.get('price_vs_sma200', ''))
        c3.metric("Golden Cross", "Yes ✅" if kl.get('golden_cross') else "No ❌")


def _render_chart_tab(ticker: str, signal: EntrySignal):
    """Interactive TradingView-style chart with APEX MTF signal overlay."""
    data_cache = st.session_state.get('ticker_data_cache', {})
    ticker_data = data_cache.get(ticker, {})
    daily = ticker_data.get('daily')

    if daily is None:
        st.warning("No chart data available")
        return

    from chart_engine import render_tv_chart, render_mtf_chart

    weekly = ticker_data.get('weekly')
    monthly = ticker_data.get('monthly')

    # ── APEX Signal Detection ─────────────────────────────────────
    show_apex = st.checkbox("🎯 Show APEX Signals", value=True, key=f"apex_{ticker}")

    apex_markers = []
    apex_signals_list = []
    apex_summary = {}

    if show_apex and weekly is not None and monthly is not None:
        try:
            # Fetch SPY/VIX for bear filter (cache across tickers)
            if 'apex_spy_data' not in st.session_state:
                from data_fetcher import fetch_daily
                st.session_state['apex_spy_data'] = fetch_daily("SPY")
                st.session_state['apex_vix_data'] = fetch_daily("^VIX")

            spy_df = st.session_state.get('apex_spy_data')
            vix_df = st.session_state.get('apex_vix_data')

            apex_signals_list = detect_apex_signals(
                ticker=ticker,
                daily_data=daily,
                weekly_data=weekly,
                monthly_data=monthly,
                spy_data=spy_df,
                vix_data=vix_df,
            )

            apex_markers = get_apex_markers(apex_signals_list)
            apex_summary = get_apex_summary(apex_signals_list)

            # Store for trade tab health monitoring
            st.session_state[f'apex_signals_{ticker}'] = apex_signals_list

        except Exception as e:
            st.warning(f"APEX detection error: {e}")

    # ── Render Chart ──────────────────────────────────────────────
    render_tv_chart(daily, ticker, signal=signal, height=750,
                    zoom_level=200, extra_markers=apex_markers,
                    key=f"tv_{ticker}")

    # ── APEX Signal Panel ─────────────────────────────────────────
    if apex_signals_list:
        st.markdown("---")
        st.markdown("### 🎯 APEX MTF Signals")

        cols = st.columns(5)
        with cols[0]:
            st.metric("Signals", apex_summary.get('total', 0))
        with cols[1]:
            wr = apex_summary.get('win_rate', 0)
            st.metric("Win Rate", f"{wr:.0f}%")
        with cols[2]:
            avg_r = apex_summary.get('avg_return', 0)
            st.metric("Avg Return", f"{avg_r:+.1f}%")
        with cols[3]:
            st.metric("Best", f"{apex_summary.get('best_trade', 0):+.1f}%")
        with cols[4]:
            st.metric("Active", apex_summary.get('active', 0))

        # Active trade banner
        if 'active_trade' in apex_summary:
            at = apex_summary['active_trade']
            trail_status = '🟢 ATR Trail ON' if at['atr_trail_active'] else '⏳ Pre-trail'
            st.success(
                f"**ACTIVE** | Entry: {at['entry_date']} @ ${at['entry_price']:.2f} | "
                f"Return: {at['current_return']:+.1f}% | "
                f"{at['tier'].replace('_', ' ')} | {at['regime'].replace('Monthly_', '')} | "
                f"Stop: {at['stop']}% | {trail_status}"
            )

        # Signal history table
        with st.expander("📋 Signal History", expanded=False):
            history_data = []
            for sig in reversed(apex_signals_list):
                history_data.append({
                    'Entry': sig.entry_date.strftime('%Y-%m-%d'),
                    'Exit': sig.exit_date.strftime('%Y-%m-%d') if sig.exit_date else '🔵 ACTIVE',
                    'Tier': sig.signal_tier.replace('Tier_', 'T'),
                    'Regime': sig.monthly_regime.replace('Monthly_', ''),
                    'Return': f"{sig.return_pct:+.1f}%" if sig.return_pct is not None else '-',
                    'Exit Type': (sig.exit_reason or 'Active').replace('_', ' '),
                    'Weeks': f"{sig.hold_weeks:.1f}" if sig.hold_weeks else '-',
                })
            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True,
            )

    # ── Chart Legend ──────────────────────────────────────────────
    with st.expander("📊 Chart Legend", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Price Panel:**
- 🟢🔴 **Candles** — Green = bullish close, Red = bearish close
- 🟠 **150d SMA** (dotted) — 30-week trend proxy. Price above = uptrend
- 🔵 **50 SMA** (dashed) — Medium-term trend direction
- 🟣 **200 SMA** (dashed) — Long-term trend. Price above = bull market
- 🔴 **R $xxx** — Overhead resistance levels (★ = critical resistance)

**TTA Signals (MACD zero-cross):**
- 🟢 **BUY** — MACD crossed above zero + AO confirmed positive
- 🔴 **SELL** — MACD crossed below zero + AO confirmed negative
- 🔶 **W5(div)** — Bearish divergence (AO wave method)
- 🟢 **W3** — Wave 3 momentum peak
""")
        with c2:
            st.markdown("""
**APEX MTF Signals (multi-timeframe system):**
- 🟢 **APEX T1** — Daily + Weekly confirmed + Monthly bullish
- 🟢 **APEX T2** — Daily + Weekly confirmed + Monthly curling
- 🟢 **APEX T3** — Daily + Weekly early + Monthly bullish
- 🔴 **EXIT** (red) — Stop loss hit
- 🟡 **EXIT** (yellow) — Weekly MACD crossed down
- 🟠 **EXIT** (orange) — ATR trailing stop (profit protection)
- 🔵 **ACTIVE** — Trade still open with current return %

**Indicator Panels:**
- **Volume** — Green/red bars
- **AO** — Momentum histogram (green = bullish)
- **MACD (12/26/9)** — Blue = MACD, Orange = Signal, Histogram = diff
""")

    # ── MTF chart ─────────────────────────────────────────────────
    if weekly is not None and monthly is not None:
        with st.expander("Multi-Timeframe View"):
            render_mtf_chart(daily, weekly, monthly, ticker, height=400)


def _render_ai_tab(ticker: str, signal: EntrySignal,
                   rec: Dict, analysis: TickerAnalysis):
    """AI-enhanced analysis with fundamental profile, TV-TA, news, and breakout guidance."""
    quality = analysis.quality or {}
    jm = get_journal()

    # Check for AI providers: Groq (primary) → Gemini (fallback)
    gemini = None
    openai_client = None  # Used for Groq (same API format)

    # Groq — primary (free, fast, generous limits)
    groq_error = None
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        if groq_key:
            from openai import OpenAI
            openai_client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            groq_error = "No GROQ_API_KEY in secrets"
    except ImportError:
        groq_error = "openai package not installed — add to requirements.txt"
    except Exception as e:
        groq_error = str(e)[:200]

    # Gemini — fallback
    gemini_error = None
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            gemini = genai.GenerativeModel('gemini-2.0-flash')
        else:
            gemini_error = "No GEMINI_API_KEY in secrets"
    except ImportError:
        gemini_error = "google-generativeai not installed"
    except Exception as e:
        gemini_error = str(e)[:200]

    if not openai_client and not gemini:
        errors = []
        if groq_error:
            errors.append(f"Groq: {groq_error}")
        if gemini_error:
            errors.append(f"Gemini: {gemini_error}")
        st.caption(f"⚠️ {' | '.join(errors)}")

    if st.button("🤖 Run AI Analysis", type="primary"):
        with st.spinner("Fetching fundamentals, TradingView, news & analyzing..."):
            fundamentals = {}
            fundamental_profile = {}
            tradingview_data = {}
            news_data = {}

            try:
                from data_fetcher import (
                    fetch_ticker_info, fetch_options_data,
                    fetch_insider_transactions, fetch_institutional_holders,
                    fetch_earnings_date, fetch_fundamental_profile,
                    fetch_earnings_history,
                    fetch_tradingview_mtf, fetch_finnhub_news,
                )
                fundamentals = {
                    'info': fetch_ticker_info(ticker),
                    'options': fetch_options_data(ticker),
                    'insider': fetch_insider_transactions(ticker),
                    'institutional': fetch_institutional_holders(ticker),
                    'earnings': fetch_earnings_date(ticker),
                }
                fundamental_profile = fetch_fundamental_profile(ticker)
            except Exception as e:
                st.caption(f"Fundamentals error: {e}")

            # Earnings history
            earnings_history = {}
            try:
                earnings_history = fetch_earnings_history(ticker)
            except Exception:
                pass

            # TradingView-TA (optional — no API key needed)
            try:
                tradingview_data = fetch_tradingview_mtf(ticker)
            except Exception:
                pass

            # Finnhub news (optional — needs API key)
            try:
                finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
                if finnhub_key:
                    news_data = fetch_finnhub_news(ticker, api_key=finnhub_key)
            except Exception:
                pass

            # Market intelligence — analysts, insiders, social
            market_intel = {}
            try:
                from data_fetcher import fetch_market_intelligence
                finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
                market_intel = fetch_market_intelligence(ticker, finnhub_key=finnhub_key)
            except Exception as e:
                st.caption(f"Market intel error: {e}")

            result = run_ai_analysis(
                ticker=ticker,
                signal=signal,
                recommendation=rec,
                quality=quality,
                fundamentals=fundamentals,
                fundamental_profile=fundamental_profile,
                tradingview_data=tradingview_data,
                news_data=news_data,
                market_intel=market_intel,
                gemini_model=gemini,
                openai_client=openai_client,
            )

            # Attach extra data for UI display
            result['earnings_history'] = earnings_history
            result['market_intel'] = market_intel

            st.session_state[f'ai_result_{ticker}'] = result

    # Display result
    ai_result = st.session_state.get(f'ai_result_{ticker}')
    if not ai_result:
        return

    provider = ai_result.get('provider', 'unknown')
    st.caption(f"Provider: {provider} | {ai_result.get('note', '')}")

    # Show AI errors if present
    if ai_result.get('groq_error'):
        st.warning(f"⚠️ Groq error: {ai_result['groq_error']}")
    if ai_result.get('gemini_error'):
        st.warning(f"⚠️ Gemini error: {ai_result['gemini_error']}")
    if ai_result.get('openai_error'):
        st.warning(f"⚠️ OpenAI error: {ai_result['openai_error']}")
    if ai_result.get('error'):
        st.warning(f"⚠️ Error: {ai_result['error']}")

    # ═══════════════════════════════════════════════════════════════
    # TOP ROW: Action + Conviction + Sizing
    # ═══════════════════════════════════════════════════════════════
    conv = ai_result.get('conviction', 0)
    action = ai_result.get('action', ai_result.get('timing', '?'))

    # Action banner
    action_upper = action.upper() if action else ''
    if 'BUY NOW' in action_upper:
        st.success(f"🟢 **{action}**")
    elif 'WAIT' in action_upper:
        st.warning(f"🟡 **{action}**")
    elif 'SKIP' in action_upper:
        st.error(f"🔴 **{action}**")
    else:
        st.info(f"📊 **{action}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        conv_icon = "🟢" if conv >= 7 else ("🟡" if conv >= 4 else "🔴")
        st.metric("Conviction", f"{conv_icon} {conv}/10")
    with c2:
        sizing = ai_result.get('position_sizing', '?')
        st.metric("Position Size", sizing.split('—')[0].strip() if '—' in sizing else sizing)
    with c3:
        fq = ai_result.get('fundamental_quality', '?')
        grade = fq[0] if fq and fq[0] in 'ABCD' else '?'
        grade_icon = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🔴'}.get(grade, '⚪')
        st.metric("Business Quality", f"{grade_icon} {grade}")

    # ═══════════════════════════════════════════════════════════════
    # RESISTANCE VERDICT + BREAKOUT ALERT BUTTON
    # ═══════════════════════════════════════════════════════════════
    resistance = ai_result.get('resistance_verdict', '')
    if resistance:
        is_wait = any(w in resistance.lower() for w in ['wait', 'breakout', 'stall', 'failed'])
        if is_wait:
            st.warning(f"🚧 **Resistance:** {resistance}")

            # Extract trigger price from resistance verdict or overhead data
            trigger_price = None
            ores = signal.overhead_resistance if signal else None
            if ores and ores.get('critical_level'):
                trigger_price = float(ores['critical_level']['price'])

            # Show "Set Breakout Alert" button
            if trigger_price:
                ba_col1, ba_col2 = st.columns([3, 1])
                with ba_col1:
                    st.markdown(f"**Set alert for breakout above ${trigger_price:.2f}?**")
                with ba_col2:
                    if st.button("🎯 Set Breakout Alert", key=f"ba_{ticker}",
                                 type="primary"):
                        from journal_manager import ConditionalEntry
                        entry = ConditionalEntry(
                            ticker=ticker,
                            condition_type='breakout_volume',
                            trigger_price=trigger_price,
                            volume_multiplier=1.5,
                            stop_price=signal.stops.get('stop', 0) if signal.stops else 0,
                            target_price=signal.stops.get('target', 0) if signal.stops else 0,
                            conviction=conv,
                            quality_grade=fq[0] if fq and fq[0] in 'ABCD' else '?',
                            notes=f"AI: {resistance[:100]}",
                            expires_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                        )
                        msg = jm.add_conditional(entry)
                        st.success(msg)
                        st.rerun()
        else:
            st.success(f"✅ **Resistance:** {resistance}")

    # ═══════════════════════════════════════════════════════════════
    # TRADINGVIEW CONFIRMATION
    # ═══════════════════════════════════════════════════════════════
    tv_data = ai_result.get('tradingview_data', {})
    if tv_data:
        _render_tv_confirmation(tv_data)

    # ═══════════════════════════════════════════════════════════════
    # WHY IT'S MOVING + FUNDAMENTAL QUALITY
    # ═══════════════════════════════════════════════════════════════
    col_l, col_r = st.columns(2)

    with col_l:
        why = ai_result.get('why_moving', '')
        if why:
            st.markdown(f"**📰 Why it's moving:**")
            st.info(why)

        fq_detail = ai_result.get('fundamental_quality', '')
        if fq_detail:
            st.markdown(f"**💼 Fundamental quality:**")
            st.info(fq_detail)

    with col_r:
        bull = ai_result.get('bull_case', '')
        if bull:
            st.markdown("**🐂 Bull case:**")
            st.success(bull)

        bear = ai_result.get('bear_case', '')
        if bear:
            st.markdown("**🐻 Bear case:**")
            st.error(bear)

    # ═══════════════════════════════════════════════════════════════
    # SMART MONEY (AI-synthesized analyst + insider view)
    # ═══════════════════════════════════════════════════════════════
    smart = ai_result.get('smart_money', '')
    if smart:
        st.markdown("**🏦 Smart Money:**")
        st.info(smart)

    # ═══════════════════════════════════════════════════════════════
    # RED FLAGS
    # ═══════════════════════════════════════════════════════════════
    flags = ai_result.get('red_flags', '')
    if flags and flags.lower() != 'none':
        st.warning(f"🚩 **Red flags:** {flags}")
    else:
        st.success("🚩 **Red flags:** None — clean setup")

    # ═══════════════════════════════════════════════════════════════
    # MARKET INTELLIGENCE PANEL
    # ═══════════════════════════════════════════════════════════════
    mi = ai_result.get('market_intel', {})
    if mi and not mi.get('error'):
        _render_market_intelligence(mi)

    # ═══════════════════════════════════════════════════════════════
    # EARNINGS SECTION
    # ═══════════════════════════════════════════════════════════════
    earnings = ai_result.get('earnings_history', {})
    if earnings and not earnings.get('error'):
        _render_earnings_section(earnings)

    # ═══════════════════════════════════════════════════════════════
    # NEWS HEADLINES
    # ═══════════════════════════════════════════════════════════════
    news = ai_result.get('news_data', {})
    if news and news.get('headlines'):
        with st.expander(f"📰 Recent News ({news.get('count', 0)} articles)", expanded=False):
            for h in news['headlines'][:5]:
                st.markdown(f"**{h.get('datetime', '')}** — {h.get('headline', '')} "
                            f"*({h.get('source', '')})*")

    # ═══════════════════════════════════════════════════════════════
    # FUNDAMENTAL SNAPSHOT TABLE
    # ═══════════════════════════════════════════════════════════════
    fp = ai_result.get('fundamental_profile', {})
    if fp and not fp.get('error'):
        with st.expander("📊 Fundamental Snapshot", expanded=False):
            _render_fundamental_snapshot(fp)

    # ═══════════════════════════════════════════════════════════════
    # FULL AI RESPONSE
    # ═══════════════════════════════════════════════════════════════
    with st.expander("📝 Full AI Response"):
        st.text(ai_result.get('raw_text', ''))


def _render_market_intelligence(intel: Dict):
    """Render market intelligence panel — analysts, insiders, social."""
    st.markdown("### 🏦 Market Intelligence")

    # ── Analyst Consensus + Price Targets ─────────────────────────
    col_analyst, col_targets = st.columns(2)

    with col_analyst:
        consensus = intel.get('analyst_consensus')
        total = intel.get('analyst_count', 0)

        if consensus and total:
            # Color-coded consensus
            c_map = {
                'Strong Buy': ('success', '🟢🟢'),
                'Buy': ('success', '🟢'),
                'Hold': ('warning', '🟡'),
                'Sell': ('error', '🔴'),
                'Strong Sell': ('error', '🔴🔴'),
            }
            method, icon = c_map.get(consensus, ('info', '⚪'))
            getattr(st, method)(f"{icon} **Analyst Consensus: {consensus}** ({total} analysts)")

            # Breakdown bar
            sb = intel.get('analyst_strong_buy', 0)
            b = intel.get('analyst_buy', 0)
            h = intel.get('analyst_hold', 0)
            s = intel.get('analyst_sell', 0)
            ss = intel.get('analyst_strong_sell', 0)

            st.caption(f"Strong Buy: {sb} | Buy: {b} | Hold: {h} | Sell: {s} | Strong Sell: {ss}")
        else:
            st.caption("No analyst data available")

    with col_targets:
        target = intel.get('target_mean')
        if target:
            high = intel.get('target_high')
            low = intel.get('target_low')
            upside = intel.get('target_upside_pct')

            if upside is not None:
                if upside > 15:
                    st.success(f"🎯 **Target: ${target:.2f}** ({upside:+.1f}% upside)")
                elif upside > 0:
                    st.info(f"🎯 **Target: ${target:.2f}** ({upside:+.1f}% upside)")
                else:
                    st.error(f"🎯 **Target: ${target:.2f}** ({upside:+.1f}% — below current)")

            if high and low:
                st.caption(f"Range: ${low:.2f} (bear) → ${high:.2f} (bull)")
        else:
            st.caption("No price targets available")

    # ── Recent Rating Changes ─────────────────────────────────────
    changes = intel.get('recent_changes', [])
    if changes:
        with st.expander(f"📋 Recent Upgrades/Downgrades ({len(changes)})", expanded=False):
            rows = []
            for c in changes[:8]:
                action = c.get('action', '?')
                # Color code
                if 'upgrade' in action.lower() or 'initiated' in action.lower():
                    action_str = f"⬆️ {action}"
                elif 'downgrade' in action.lower():
                    action_str = f"⬇️ {action}"
                else:
                    action_str = f"➡️ {action}"

                from_g = f" (from {c.get('from_grade', '')})" if c.get('from_grade') else ""
                rows.append({
                    'Date': c.get('date', '?'),
                    'Firm': c.get('firm', '?'),
                    'Action': action_str,
                    'Rating': f"{c.get('to_grade', '?')}{from_g}",
                })

            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Insider Activity + Social ─────────────────────────────────
    col_insider, col_social = st.columns(2)

    with col_insider:
        buys = intel.get('insider_buys_90d', 0)
        sells = intel.get('insider_sells_90d', 0)

        if buys > 0 or sells > 0:
            net = intel.get('insider_net_shares', 0)
            if net > 0:
                st.success(f"👔 **Insiders (90d): {buys} buys, {sells} sells — NET BUYING**")
            elif net < 0:
                st.warning(f"👔 **Insiders (90d): {buys} buys, {sells} sells — NET SELLING**")
            else:
                st.info(f"👔 **Insiders (90d): {buys} buys, {sells} sells — Neutral**")

            # Show top transactions
            txns = intel.get('insider_transactions', [])
            if txns:
                with st.expander("Insider Transactions", expanded=False):
                    rows = []
                    for t in txns[:8]:
                        val = t.get('value', 0)
                        val_str = f"${val:,.0f}" if val else "—"
                        rows.append({
                            'Date': t.get('date', '?'),
                            'Name': t.get('name', '?'),
                            'Type': t.get('type', '?'),
                            'Shares': f"{t.get('shares', 0):,}",
                            'Value': val_str,
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No insider transactions found")

    with col_social:
        social = intel.get('social_score')
        reddit = intel.get('social_reddit_mentions')
        twitter = intel.get('social_twitter_mentions')

        if social:
            s_map = {'High buzz': ('success', '🔥'), 'Moderate': ('info', '📊'), 'Low': ('warning', '😴')}
            method, icon = s_map.get(social, ('info', '📊'))
            getattr(st, method)(f"{icon} **Social: {social}**")
            parts = []
            if reddit is not None:
                parts.append(f"Reddit: {reddit}")
            if twitter is not None:
                parts.append(f"Twitter: {twitter}")
            if parts:
                st.caption(f"7-day mentions — {' | '.join(parts)}")
        else:
            st.caption("Social sentiment not available (needs Finnhub)")


def _render_tv_confirmation(tv_data: Dict):
    """Render TradingView-TA conviction with overall verdict and timeframe breakdown."""
    if not tv_data:
        return

    # Calculate overall conviction from all timeframes
    rec_scores = {'STRONG_BUY': 2, 'BUY': 1, 'NEUTRAL': 0, 'SELL': -1, 'STRONG_SELL': -2}
    rec_labels = {2: 'STRONG BUY', 1: 'BUY', 0: 'NEUTRAL', -1: 'SELL', -2: 'STRONG SELL'}
    rec_icons = {
        'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'NEUTRAL': '🟡',
        'SELL': '🔴', 'STRONG_SELL': '🔴🔴',
    }

    scores = []
    valid_data = {}
    for interval in ['1h', '4h', '1d', '1W']:
        data = tv_data.get(interval, {})
        if data.get('error') or not data.get('recommendation'):
            continue
        valid_data[interval] = data
        rec = data['recommendation']
        if rec in rec_scores:
            scores.append(rec_scores[rec])

    if not scores:
        # Check for specific errors
        first_err = None
        for data in tv_data.values():
            err = data.get('error')
            if err:
                first_err = err
                break
        if first_err and 'not installed' in first_err:
            st.caption("TradingView-TA: Not available (pip install tradingview_ta)")
        elif first_err:
            st.caption(f"TradingView-TA: {first_err}")
        else:
            st.caption("TradingView-TA: No data returned")
        return

    # Overall verdict
    avg_score = sum(scores) / len(scores)
    if avg_score >= 1.5:
        overall = 'STRONG BUY'
        overall_color = 'success'
        icon = '🟢🟢'
    elif avg_score >= 0.5:
        overall = 'BUY'
        overall_color = 'success'
        icon = '🟢'
    elif avg_score >= -0.5:
        overall = 'NEUTRAL'
        overall_color = 'warning'
        icon = '🟡'
    elif avg_score >= -1.5:
        overall = 'SELL'
        overall_color = 'error'
        icon = '🔴'
    else:
        overall = 'STRONG SELL'
        overall_color = 'error'
        icon = '🔴🔴'

    # Display overall verdict
    getattr(st, overall_color)(f"{icon} **TradingView Conviction: {overall}** "
                                f"(avg score: {avg_score:+.1f})")

    # Timeframe breakdown
    labels = {'1h': '1 Hour', '4h': '4 Hour', '1d': 'Daily', '1W': 'Weekly'}
    cols = st.columns(len(valid_data))

    for i, (interval, data) in enumerate(valid_data.items()):
        rec = data.get('recommendation', '')
        buy = data.get('buy', 0)
        sell = data.get('sell', 0)
        neutral = data.get('neutral', 0)
        total = buy + sell + neutral
        label = labels.get(interval, interval)
        ri = rec_icons.get(rec, '⚪')

        with cols[i]:
            st.markdown(f"**{label}**")
            st.markdown(f"{ri} **{rec.replace('_', ' ')}**")
            if total > 0:
                st.caption(f"Buy: {buy} | Neutral: {neutral} | Sell: {sell}")

    # Key indicators from daily
    daily = valid_data.get('1d', {})
    indicator_parts = []
    if daily.get('rsi') is not None:
        rsi = daily['rsi']
        rsi_label = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
        indicator_parts.append(f"RSI: {rsi:.1f} ({rsi_label})")
    if daily.get('adx') is not None:
        adx = daily['adx']
        trend_str = "strong trend" if adx > 25 else "weak trend"
        indicator_parts.append(f"ADX: {adx:.1f} ({trend_str})")
    if daily.get('cci') is not None:
        indicator_parts.append(f"CCI: {daily['cci']:.0f}")

    if indicator_parts:
        st.caption(" | ".join(indicator_parts))


def _render_earnings_section(earnings: Dict):
    """Render earnings history with next date, streak, and quarterly results."""
    next_date = earnings.get('next_earnings')
    days_until = earnings.get('days_until_earnings')
    quarters = earnings.get('quarters', [])
    streak = earnings.get('streak', 0)

    if not next_date and not quarters:
        return

    st.markdown("### 📅 Earnings")

    # ── Next Earnings Banner ──────────────────────────────────────
    if next_date:
        next_eps = earnings.get('next_eps_estimate')

        if days_until is not None and days_until <= 14:
            # Imminent — warning
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.error(f"⚠️ **Next Earnings: {next_date} ({days_until} days)**{eps_str} — "
                     f"Consider waiting or reducing size")
        elif days_until is not None and days_until <= 30:
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.warning(f"📅 **Next Earnings: {next_date} ({days_until} days)**{eps_str}")
        else:
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.info(f"📅 **Next Earnings: {next_date}"
                    f"{f' ({days_until} days)' if days_until else ''}**{eps_str}")

    # ── Streak + Summary ──────────────────────────────────────────
    if quarters:
        col_streak, col_avg = st.columns(2)

        with col_streak:
            if streak > 0:
                st.success(f"🔥 **{streak} consecutive beat{'s' if streak > 1 else ''}**")
            elif streak < 0:
                st.error(f"📉 **{abs(streak)} consecutive miss{'es' if abs(streak) > 1 else ''}**")
            else:
                st.info("➡️ **Mixed results**")

        with col_avg:
            avg = earnings.get('avg_surprise_pct')
            if avg is not None:
                if avg > 0:
                    st.success(f"📊 **Avg surprise: +{avg:.1f}%**")
                else:
                    st.error(f"📊 **Avg surprise: {avg:.1f}%**")

    # ── Quarterly Table ───────────────────────────────────────────
    if quarters:
        rows = []
        for q in quarters:
            date = q.get('date', '?')
            eps_est = q.get('eps_estimate')
            eps_act = q.get('eps_actual')
            surprise = q.get('surprise_pct')
            beat = q.get('beat')

            est_str = f"${eps_est:.2f}" if eps_est is not None else "—"
            act_str = f"${eps_act:.2f}" if eps_act is not None else "—"
            surp_str = f"{surprise:+.1f}%" if surprise is not None else "—"

            if beat is True:
                verdict = "✅ Beat"
            elif beat is False:
                verdict = "❌ Miss"
            else:
                verdict = "—"

            rows.append({
                'Date': date,
                'EPS Est': est_str,
                'EPS Act': act_str,
                'Surprise': surp_str,
                'Result': verdict,
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                      column_config={
                          'Date': st.column_config.TextColumn(width="medium"),
                          'EPS Est': st.column_config.TextColumn("Estimate", width="small"),
                          'EPS Act': st.column_config.TextColumn("Actual", width="small"),
                          'Surprise': st.column_config.TextColumn(width="small"),
                          'Result': st.column_config.TextColumn(width="small"),
                      })


def _render_fundamental_snapshot(fp: Dict):
    """Render compact fundamental data table from profile."""

    def _fmt_money(val):
        if val is None:
            return "—"
        if abs(val) >= 1e9:
            return f"${val/1e9:.1f}B"
        if abs(val) >= 1e6:
            return f"${val/1e6:.0f}M"
        return f"${val:,.0f}"

    def _fmt_pct(val):
        if val is None:
            return "—"
        return f"{val*100:.1f}%"

    def _fmt_num(val, decimals=1):
        if val is None:
            return "—"
        return f"{val:.{decimals}f}"

    # Company header
    name = fp.get('name', '?')
    sector = fp.get('sector', '?')
    industry = fp.get('industry', '?')
    st.markdown(f"**{name}** — {sector} / {industry}")

    if fp.get('business_summary'):
        st.caption(fp['business_summary'][:300] + "..." if len(fp.get('business_summary', '')) > 300 else fp['business_summary'])

    # Three column layout
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Growth & Scale**")
        rows = [
            ("Market Cap", _fmt_money(fp.get('market_cap'))),
            ("Revenue", _fmt_money(fp.get('total_revenue'))),
            ("Revenue Growth", _fmt_pct(fp.get('revenue_growth_yoy'))),
            ("Earnings Growth", _fmt_pct(fp.get('earnings_growth_yoy'))),
            ("EBITDA", _fmt_money(fp.get('ebitda'))),
        ]
        for label, val in rows:
            if val != "—":
                st.caption(f"{label}: **{val}**")

    with c2:
        st.markdown("**Profitability**")
        rows = [
            ("Gross Margin", _fmt_pct(fp.get('gross_margin'))),
            ("Operating Margin", _fmt_pct(fp.get('operating_margin'))),
            ("Net Margin", _fmt_pct(fp.get('profit_margin'))),
            ("ROE", _fmt_pct(fp.get('return_on_equity'))),
            ("FCF", _fmt_money(fp.get('free_cash_flow'))),
        ]
        for label, val in rows:
            if val != "—":
                st.caption(f"{label}: **{val}**")

    with c3:
        st.markdown("**Valuation & Health**")
        rows = [
            ("P/E (Fwd)", _fmt_num(fp.get('forward_pe'))),
            ("EV/EBITDA", _fmt_num(fp.get('ev_to_ebitda'))),
            ("P/Sales", _fmt_num(fp.get('price_to_sales'))),
            ("Debt/Equity", _fmt_num(fp.get('debt_to_equity'), 0)),
            ("Short % Float", _fmt_pct(fp.get('short_pct_float'))),
        ]
        for label, val in rows:
            if val != "—":
                st.caption(f"{label}: **{val}**")

    # Ownership bar
    insider = fp.get('insider_pct')
    inst = fp.get('institution_pct')
    if insider is not None or inst is not None:
        parts = []
        if insider is not None:
            parts.append(f"Insider: {insider*100:.1f}%")
        if inst is not None:
            parts.append(f"Institutional: {inst*100:.1f}%")
        if fp.get('next_earnings'):
            parts.append(f"Next Earnings: {fp['next_earnings']}")
        if fp.get('last_earnings_surprise_pct') is not None:
            parts.append(f"Last Surprise: {fp['last_earnings_surprise_pct']:+.1f}%")
        st.caption(" | ".join(parts))


def _render_trade_tab(ticker: str, signal: EntrySignal,
                      analysis: TickerAnalysis):
    """Enhanced trade management: position calculator, portfolio dashboard, health monitoring."""
    jm = get_journal()
    rec = analysis.recommendation or {}
    stops = signal.stops if signal else {}

    # ── Portfolio Capital Summary (always visible) ────────────────
    _render_capital_overview(jm)

    st.divider()

    # Check if already in a position
    open_tickers = jm.get_open_tickers()
    if ticker in open_tickers:
        _render_position_management(ticker, jm)
        return

    # ── Position Calculator & Entry ───────────────────────────────
    _render_position_calculator(ticker, signal, analysis, jm, rec, stops)


def _render_capital_overview(jm: JournalManager):
    """
    Always-visible capital bar: total account, deployed, available,
    per-ticker breakdown with health status.
    """
    open_trades = jm.get_open_trades()

    # Account size (persist across session)
    if 'account_size' not in st.session_state:
        st.session_state['account_size'] = 100000.0

    account_size = st.session_state['account_size']

    if not open_trades:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            new_acct = st.number_input(
                "💰 Account Size", value=account_size,
                step=10000.0, format="%.0f", key="global_acct",
                label_visibility="collapsed",
            )
            if new_acct != account_size:
                st.session_state['account_size'] = new_acct
        with col_b:
            st.info(f"💰 **${account_size:,.0f}** available — no open positions")
        return

    # Fetch live prices
    from data_fetcher import fetch_current_price

    current_prices = {}
    for trade in open_trades:
        t = trade['ticker']
        price = fetch_current_price(t)
        if price:
            current_prices[t] = price

    # Calculate totals
    total_deployed = 0
    total_current_value = 0
    total_pnl = 0
    position_rows = []

    for trade in open_trades:
        t = trade['ticker']
        entry = float(trade.get('entry_price', 0))
        shares = float(trade.get('shares', 0))
        pos_size = float(trade.get('position_size', 0))
        stop = float(trade.get('current_stop', trade.get('initial_stop', 0)))
        current = current_prices.get(t, entry)

        total_deployed += pos_size
        current_value = current * shares
        total_current_value += current_value
        pnl_dollars = (current - entry) * shares
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        total_pnl += pnl_dollars

        # Distance to stop
        stop_dist = ((current - stop) / current * 100) if current > 0 and stop > 0 else 999

        # Health traffic light
        if (stop > 0 and current <= stop) or pnl_pct < -10:
            health = "🔴"
            action = "CLOSE NOW"
        elif stop_dist < 2 or pnl_pct < -5:
            health = "🔴"
            action = "EXIT SOON"
        elif stop_dist < 5 or pnl_pct < -3:
            health = "🟡"
            action = "WATCH"
        elif pnl_pct >= 15:
            health = "🟢"
            action = "TRAIL STOP"
        else:
            health = "🟢"
            action = "HOLD"

        # Days held
        try:
            days_held = (datetime.now() - datetime.strptime(trade.get('entry_date', ''), '%Y-%m-%d')).days
        except Exception:
            days_held = 0

        position_rows.append({
            'health': health,
            'action': action,
            'ticker': t,
            'shares': shares,
            'entry': entry,
            'current': current,
            'cost': pos_size,
            'value': current_value,
            'pnl_dollars': pnl_dollars,
            'pnl_pct': pnl_pct,
            'stop': stop,
            'stop_dist': stop_dist,
            'days': days_held,
        })

    available = account_size - total_deployed
    deployed_pct = (total_deployed / account_size * 100) if account_size > 0 else 0

    # ── Capital Bar ───────────────────────────────────────────────
    st.markdown("### 💼 Portfolio Capital")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Account", f"${account_size:,.0f}")
    with c2:
        st.metric("Deployed", f"${total_deployed:,.0f}",
                  f"{deployed_pct:.0f}%")
    with c3:
        avail_color = "normal" if available > 0 else "inverse"
        st.metric("Available", f"${available:,.0f}",
                  f"{100 - deployed_pct:.0f}%")
    with c4:
        st.metric("Current Value", f"${total_current_value:,.0f}",
                  f"{total_pnl:+,.0f}")
    with c5:
        total_pnl_pct = (total_pnl / total_deployed * 100) if total_deployed > 0 else 0
        st.metric("Total P&L", f"{total_pnl_pct:+.1f}%",
                  f"${total_pnl:+,.0f}")

    # Deployment progress bar
    bar_pct = min(deployed_pct / 100, 1.0)
    if deployed_pct >= 90:
        st.progress(bar_pct, text=f"⚠️ {deployed_pct:.0f}% deployed — near full allocation")
    elif deployed_pct >= 70:
        st.progress(bar_pct, text=f"🟡 {deployed_pct:.0f}% deployed")
    else:
        st.progress(bar_pct, text=f"🟢 {deployed_pct:.0f}% deployed")

    # ── Positions Table with Health Lights ─────────────────────────
    # Alert banner for red positions
    reds = [r for r in position_rows if r['health'] == '🔴']
    if reds:
        tickers_at_risk = ", ".join(f"**{r['ticker']}** ({r['action']})" for r in reds)
        st.error(f"🚨 ACTION REQUIRED: {tickers_at_risk}")

    # Table
    table_rows = []
    for r in position_rows:
        table_rows.append({
            '': r['health'],
            'Ticker': r['ticker'],
            'Action': r['action'],
            'Shares': f"{r['shares']:.0f}",
            'Entry': f"${r['entry']:.2f}",
            'Now': f"${r['current']:.2f}",
            'P&L': f"{r['pnl_pct']:+.1f}%",
            'P&L $': f"${r['pnl_dollars']:+,.0f}",
            'Cost': f"${r['cost']:,.0f}",
            'Value': f"${r['value']:,.0f}",
            'Stop': f"${r['stop']:.2f}" if r['stop'] > 0 else "—",
            'To Stop': f"{r['stop_dist']:.1f}%" if r['stop_dist'] < 999 else "—",
            'Days': r['days'],
        })

    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )

    # Account size editor (collapsed)
    with st.expander("⚙️ Account Settings"):
        new_acct = st.number_input(
            "Account Size ($)", value=account_size,
            step=10000.0, format="%.0f", key="global_acct_edit",
        )
        if new_acct != account_size:
            st.session_state['account_size'] = new_acct
            st.rerun()


def _render_position_calculator(ticker, signal, analysis, jm, rec, stops):
    """
    Two-mode position calculator:
    1. Quick mode — enter $ amount → get shares
    2. Full mode — risk-based sizing with stop/target
    """
    current_price = analysis.current_price or 0
    entry_default = float(stops.get('entry', current_price or 0))
    stop_default = float(stops.get('stop', 0))
    target_default = float(stops.get('target', 0))

    account_size = st.session_state.get('account_size', 100000.0)
    open_trades = jm.get_open_trades()
    deployed = sum(float(t.get('position_size', 0)) for t in open_trades)
    available = account_size - deployed
    max_positions = 7  # APEX V4

    st.subheader(f"📐 Position Calculator — {ticker}")

    # ══════════════════════════════════════════════════════════════
    # QUICK CALCULATOR — just enter $ amount
    # ══════════════════════════════════════════════════════════════
    st.markdown("**Quick Size** — enter the amount you want to invest:")

    qc1, qc2, qc3 = st.columns([2, 1, 1])
    with qc1:
        invest_amount = st.number_input(
            "💵 Amount to Invest ($)",
            value=min(float(int(account_size / max_positions)), available),
            step=1000.0, format="%.0f",
            key=f"quick_amt_{ticker}",
        )
    with qc2:
        buy_price = st.number_input(
            "Buy Price ($)",
            value=entry_default if entry_default > 0 else current_price,
            step=0.01, format="%.2f",
            key=f"quick_price_{ticker}",
        )
    with qc3:
        if buy_price > 0:
            quick_shares = int(invest_amount / buy_price)
            actual_cost = quick_shares * buy_price
            st.metric(
                "Shares to Buy",
                f"{quick_shares:,}",
                f"${actual_cost:,.0f}",
            )
        else:
            st.metric("Shares to Buy", "—", "Enter price")

    # Quick validation
    if buy_price > 0 and invest_amount > 0:
        quick_shares = int(invest_amount / buy_price)
        actual_cost = quick_shares * buy_price
        pct_of_account = (actual_cost / account_size * 100) if account_size > 0 else 0
        pct_of_available = (actual_cost / available * 100) if available > 0 else 999

        info_parts = [
            f"**{quick_shares:,} shares** × ${buy_price:.2f} = **${actual_cost:,.0f}**",
            f"({pct_of_account:.1f}% of account)",
        ]

        if actual_cost > available:
            st.error(f"⚠️ Exceeds available capital! Need ${actual_cost:,.0f} but only ${available:,.0f} free")
        elif pct_of_account > 15:
            st.warning(f"{'  |  '.join(info_parts)}  —  ⚠️ Over 15% concentration")
        else:
            st.success(f"{'  |  '.join(info_parts)}")

    # ══════════════════════════════════════════════════════════════
    # FULL RISK-BASED CALCULATOR
    # ══════════════════════════════════════════════════════════════
    with st.expander("🎯 Full Risk-Based Calculator", expanded=False):
        st.caption("Sizes position using your risk tolerance and stop distance")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            risk_per_trade = st.number_input(
                "⚡ Max Risk per Trade (%)", value=2.0,
                min_value=0.5, max_value=10.0, step=0.5, format="%.1f",
                key=f"risk_pct_{ticker}",
                help="Maximum portfolio % you're willing to lose on this trade"
            )
        with col_b:
            max_position_pct = st.number_input(
                "📊 Max Position Size (%)", value=12.5,
                min_value=1.0, max_value=25.0, step=0.5, format="%.1f",
                key=f"max_pos_{ticker}",
                help="Maximum % of portfolio in one position (APEX V4 uses 12.5%)"
            )
        with col_c:
            st.metric("Available Capital", f"${available:,.0f}",
                      f"{len(open_trades)}/{max_positions} slots used")

        col1, col2, col3 = st.columns(3)
        with col1:
            entry_price = st.number_input("Entry Price",
                                           value=entry_default,
                                           step=0.01, format="%.2f",
                                           key=f"entry_{ticker}")
        with col2:
            stop_price = st.number_input("Stop Loss",
                                          value=stop_default,
                                          step=0.01, format="%.2f",
                                          key=f"stop_{ticker}")
        with col3:
            target_price = st.number_input("Target",
                                            value=target_default,
                                            step=0.01, format="%.2f",
                                            key=f"target_{ticker}")

        if entry_price > 0 and stop_price > 0 and entry_price > stop_price:
            risk_per_share = entry_price - stop_price
            risk_pct_trade = risk_per_share / entry_price * 100

            max_position_dollars = account_size * (max_position_pct / 100)
            risk_budget = account_size * (risk_per_trade / 100)

            shares_from_risk = int(risk_budget / risk_per_share)
            shares_from_max = int(max_position_dollars / entry_price)
            shares_from_capital = int(available / entry_price) if available > 0 else 0

            # Most conservative
            shares = min(shares_from_risk, shares_from_max, shares_from_capital)
            position_cost = shares * entry_price
            actual_risk_dollars = shares * risk_per_share
            actual_risk_pct = actual_risk_dollars / account_size * 100

            reward_per_share = target_price - entry_price if target_price > entry_price else 0
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            potential_profit = shares * reward_per_share if reward_per_share > 0 else 0

            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Risk-Based", f"{shares_from_risk} shares",
                          f"${shares_from_risk * entry_price:,.0f}")
            with c2:
                st.metric("Max Position", f"{shares_from_max} shares",
                          f"${shares_from_max * entry_price:,.0f}")
            with c3:
                limiting = "risk" if shares == shares_from_risk else (
                    "max position" if shares == shares_from_max else "available capital"
                )
                st.metric("✅ Final", f"{shares} shares",
                          f"${position_cost:,.0f} — limited by {limiting}")

            c4, c5, c6, c7 = st.columns(4)
            with c4:
                st.metric("💸 Risk", f"${actual_risk_dollars:,.0f}",
                          f"{actual_risk_pct:.1f}% of account")
            with c5:
                st.metric("📉 Stop Distance", f"{risk_pct_trade:.1f}%",
                          f"${risk_per_share:.2f}/share")
            with c6:
                if rr_ratio > 0:
                    st.metric("🎯 R/R Ratio", f"{rr_ratio:.1f}:1",
                              "Good ✅" if rr_ratio >= 2 else "Low ⚠️")
                else:
                    st.metric("🎯 R/R Ratio", "—", "Set target")
            with c7:
                if potential_profit > 0:
                    st.metric("💰 Potential", f"${potential_profit:,.0f}",
                              f"+{reward_per_share / entry_price * 100:.1f}%")

            # Warnings
            warnings = []
            if risk_pct_trade > 8:
                warnings.append("🔴 Stop distance very wide — consider tighter stop")
            elif risk_pct_trade > 5:
                warnings.append("🟡 Stop distance wider than typical (3-5%)")
            if rr_ratio > 0 and rr_ratio < 1.5:
                warnings.append("🔴 R/R below 1.5:1 — not worth the risk")
            elif rr_ratio > 0 and rr_ratio < 2.0:
                warnings.append("🟡 R/R below ideal 2:1")
            if actual_risk_pct > 3:
                warnings.append("🔴 Portfolio risk exceeds 3% — reduce position")
            if rec.get('conviction', 0) < 5:
                warnings.append(f"🟡 Low conviction ({rec.get('conviction', 0)}/10)")
            if shares_from_capital < shares_from_risk:
                warnings.append("🟡 Position limited by available capital, not risk parameters")

            for w in warnings:
                st.warning(w)
            if not warnings:
                st.success("✅ Position sizing within all risk parameters")

        elif entry_price > 0 and stop_price > 0 and stop_price >= entry_price:
            st.error("Stop price must be below entry price")

    # ══════════════════════════════════════════════════════════════
    # ENTER TRADE
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### ✅ Enter Trade")

    # Determine shares — use quick calculator values as defaults
    final_price = buy_price if buy_price > 0 else entry_default
    final_shares = int(invest_amount / final_price) if (final_price > 0 and invest_amount > 0) else 0
    final_stop = stop_default
    final_target = target_default

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        confirm_shares = st.number_input("Shares", value=final_shares,
                                          min_value=0, step=1,
                                          key=f"confirm_shares_{ticker}")
    with ec2:
        confirm_entry = st.number_input("Entry $", value=float(final_price),
                                         step=0.01, format="%.2f",
                                         key=f"confirm_entry_{ticker}")
    with ec3:
        confirm_stop = st.number_input("Stop $", value=float(final_stop),
                                        step=0.01, format="%.2f",
                                        key=f"confirm_stop_{ticker}")
    with ec4:
        confirm_target = st.number_input("Target $", value=float(final_target),
                                          step=0.01, format="%.2f",
                                          key=f"confirm_target_{ticker}")

    if confirm_shares > 0 and confirm_entry > 0:
        total_cost = confirm_shares * confirm_entry
        st.caption(
            f"**{confirm_shares} shares × ${confirm_entry:.2f} = ${total_cost:,.0f}** "
            f"({total_cost / account_size * 100:.1f}% of account)"
        )

    notes = st.text_input("Notes", value=rec.get('summary', ''), key=f"notes_{ticker}")

    if st.button("✅ Enter Trade", type="primary", key=f"enter_{ticker}"):
        if confirm_entry <= 0 or confirm_shares <= 0:
            st.error("Set entry price and shares first")
        elif confirm_stop <= 0:
            st.error("Set a stop loss — never trade without a stop")
        elif confirm_stop >= confirm_entry:
            st.error("Stop must be below entry price")
        else:
            pos_size = confirm_shares * confirm_entry
            trade = Trade(
                trade_id='',
                ticker=ticker,
                entry_price=confirm_entry,
                initial_stop=confirm_stop,
                target=confirm_target,
                position_size=pos_size,
                shares=confirm_shares,
                signal_type=rec.get('signal_type', ''),
                quality_grade=analysis.quality.get('quality_grade', '') if analysis.quality else '',
                conviction_at_entry=rec.get('conviction', 0),
                weekly_bullish_at_entry=signal.weekly_macd.get('bullish', False) if signal else False,
                monthly_bullish_at_entry=signal.monthly_macd.get('bullish', False) if signal else False,
                weinstein_stage_at_entry=signal.weinstein.get('stage', 0) if signal else 0,
                risk_per_share=confirm_entry - confirm_stop,
                risk_pct=((confirm_entry - confirm_stop) / confirm_entry * 100) if confirm_entry > 0 else 0,
                notes=notes,
            )
            result = jm.enter_trade(trade)
            st.success(result)
            st.rerun()


def _render_portfolio_dashboard(jm: JournalManager):
    """Compatibility wrapper — now redirects to capital overview."""
    _render_capital_overview(jm)


def _render_position_management(ticker: str, jm: JournalManager):
    """Manage an existing open position with health monitoring and APEX context."""
    trades = jm.get_open_trades()
    trade = next((t for t in trades if t['ticker'] == ticker), None)
    if not trade:
        return

    from data_fetcher import fetch_current_price
    current = fetch_current_price(ticker) or 0

    entry = float(trade.get('entry_price', 0))
    shares = float(trade.get('shares', 0))
    stop = float(trade.get('current_stop', trade.get('initial_stop', 0)))
    target = float(trade.get('target', 0))
    pos_size = float(trade.get('position_size', 0))

    pnl_pct = ((current - entry) / entry * 100) if entry > 0 and current > 0 else 0
    pnl_dollars = (current - entry) * shares
    current_value = current * shares if current > 0 else pos_size
    stop_distance = ((current - stop) / current * 100) if current > 0 and stop > 0 else 999
    target_distance = ((target - current) / current * 100) if current > 0 and target > 0 else 0

    try:
        days_held = (datetime.now() - datetime.strptime(trade.get('entry_date', ''), '%Y-%m-%d')).days
    except Exception:
        days_held = 0

    # ── Health Assessment ─────────────────────────────────────────
    if stop > 0 and current > 0 and current <= stop:
        health_icon = "🔴"
        health_msg = "STOP HIT — Close this position NOW"
        health_level = "error"
    elif stop_distance < 2:
        health_icon = "🔴"
        health_msg = f"Only {stop_distance:.1f}% above stop — prepare to exit"
        health_level = "error"
    elif pnl_pct < -10:
        health_icon = "🔴"
        health_msg = f"Down {pnl_pct:.1f}% — significant loss, review immediately"
        health_level = "error"
    elif pnl_pct < -5:
        health_icon = "🔴"
        health_msg = f"Down {pnl_pct:.1f}% — approaching max pain, decide: hold or cut"
        health_level = "error"
    elif stop_distance < 5:
        health_icon = "🟡"
        health_msg = f"{stop_distance:.1f}% buffer to stop — monitor closely"
        health_level = "warning"
    elif pnl_pct < -3:
        health_icon = "🟡"
        health_msg = f"Small drawdown {pnl_pct:.1f}% — within normal range but watchful"
        health_level = "warning"
    elif days_held > 60 and pnl_pct < 3:
        health_icon = "🟡"
        health_msg = f"Held {days_held}d with only {pnl_pct:+.1f}% gain — dead money?"
        health_level = "warning"
    elif pnl_pct >= 20:
        health_icon = "🟢"
        health_msg = f"Excellent +{pnl_pct:.1f}% — trail stop to protect profits!"
        health_level = "success"
    elif pnl_pct >= 15:
        health_icon = "🟢"
        health_msg = f"Strong +{pnl_pct:.1f}% — ATR trailing stop should be active"
        health_level = "success"
    elif pnl_pct >= 5:
        health_icon = "🟢"
        health_msg = f"Healthy +{pnl_pct:.1f}% — trend intact"
        health_level = "success"
    else:
        health_icon = "🟢"
        health_msg = "Within normal parameters"
        health_level = "success"

    # Header
    st.subheader(f"{health_icon} {ticker} — Open Position")

    if health_level == "error":
        st.error(health_msg)
    elif health_level == "warning":
        st.warning(health_msg)
    else:
        st.success(health_msg)

    # ── Key Metrics ───────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Entry", f"${entry:.2f}", f"{shares:.0f} shares")
    with col2:
        st.metric("Current", f"${current:.2f}" if current > 0 else "—",
                  f"{pnl_pct:+.1f}%" if current > 0 else "")
    with col3:
        st.metric("P&L", f"${pnl_dollars:+,.0f}" if current > 0 else "—",
                  f"${pos_size:,.0f} → ${current_value:,.0f}" if current > 0 else "")
    with col4:
        st.metric("Stop", f"${stop:.2f}",
                  f"{stop_distance:.1f}% away" if stop_distance < 999 else "")
    with col5:
        st.metric("Target", f"${target:.2f}" if target > 0 else "—",
                  f"{target_distance:.1f}% to go" if target > 0 and current > 0 else "")

    st.caption(
        f"Signal: {trade.get('signal_type', '?')} | "
        f"Quality: {trade.get('quality_grade', '?')} | "
        f"Conviction: {trade.get('conviction_at_entry', '?')}/10 | "
        f"Opened: {trade.get('entry_date', '?')} ({days_held}d ago)"
    )

    # ── APEX Signal Context ───────────────────────────────────────
    # Check if APEX signals are available in session state
    apex_signals_key = f'apex_signals_{ticker}'
    if apex_signals_key in st.session_state:
        apex_sigs = st.session_state[apex_signals_key]
        active_apex = [s for s in apex_sigs if s.is_active]
        if active_apex:
            a = active_apex[-1]
            trail_status = '🟢 ATR Trail ON' if a.atr_trail_active else '⏳ Pre-trail'
            st.info(
                f"📡 **APEX Signal Active** — {a.signal_tier.replace('_', ' ')} | "
                f"{a.monthly_regime.replace('Monthly_', '')} regime | "
                f"Stop: {a.stop_level}% | {trail_status} | "
                f"Highest: ${a.highest_price:.2f}"
            )
        elif apex_sigs:
            last = apex_sigs[-1]
            st.warning(
                f"📡 **Last APEX signal closed** — {last.exit_reason} on "
                f"{last.exit_date.strftime('%Y-%m-%d') if last.exit_date else '?'} "
                f"({last.return_pct:+.1f}%)"
            )

    # ── Trail Stop Suggestions ────────────────────────────────────
    if pnl_pct >= 15 and current > 0:
        st.divider()
        st.markdown("**💡 Trail Stop Suggestions:**")

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            # Breakeven + buffer
            be_stop = entry * 1.02
            if be_stop > stop:
                locked = ((be_stop - entry) / entry * 100)
                st.metric("🔒 Breakeven +2%", f"${be_stop:.2f}",
                          f"Locks {locked:.1f}% profit")
        with tc2:
            # 50% profit lock
            half_profit_stop = entry + (current - entry) * 0.5
            if half_profit_stop > stop:
                locked = ((half_profit_stop - entry) / entry * 100)
                st.metric("🔐 Lock 50% Profit", f"${half_profit_stop:.2f}",
                          f"Locks {locked:.1f}% profit")
        with tc3:
            # ATR trail (approximate)
            atr_trail = current * 0.92  # ~8% from current
            if atr_trail > stop:
                locked = ((atr_trail - entry) / entry * 100)
                st.metric("📊 ATR Trail (~8%)", f"${atr_trail:.2f}",
                          f"Locks {locked:.1f}% profit")

    # ── Position Scaling ──────────────────────────────────────────
    if pnl_pct >= 10 and current > 0:
        with st.expander("📈 Add to Winner?"):
            account_size = st.session_state.get('account_size', 100000.0)
            max_pos = account_size * 0.125  # 12.5%
            room = max_pos - current_value

            if room > current:
                add_shares = int(room / current)
                st.info(
                    f"Room to add **{add_shares} shares** (${add_shares * current:,.0f}) "
                    f"before hitting 12.5% max position. "
                    f"Current position: ${current_value:,.0f} / ${max_pos:,.0f}"
                )
            else:
                st.caption("Position near or above max size — no room to add")

    # ── Actions ───────────────────────────────────────────────────
    st.divider()
    act1, act2 = st.columns(2)

    with act1:
        st.markdown("**📈 Trail Stop**")
        new_stop = st.number_input("New Stop", value=float(stop),
                                    step=0.50, format="%.2f",
                                    key=f"trail_{ticker}")
        if st.button("📈 Update Stop", key=f"trail_btn_{ticker}"):
            if new_stop > stop:
                result = jm.update_stop(ticker, new_stop)
                st.info(result)
                st.rerun()
            else:
                st.warning(f"New stop must be higher than current ${stop:.2f}")

    with act2:
        st.markdown("**🔴 Close Position**")
        exit_price = st.number_input("Exit Price",
                                      value=float(current) if current > 0 else 0.0,
                                      step=0.01, format="%.2f",
                                      key=f"exit_{ticker}")
        exit_reason = st.selectbox("Exit Reason",
                                    ['manual', 'stop_loss', 'target_hit',
                                     'weekly_cross', 'time_exit'],
                                    key=f"exit_reason_{ticker}")

        if st.button("🔴 Close Position", key=f"close_{ticker}"):
            if exit_price > 0:
                result = jm.close_trade(ticker, exit_price, exit_reason)
                st.success(result)
                st.rerun()
            else:
                st.warning("Enter exit price")


# =============================================================================
# PERFORMANCE VIEW
# =============================================================================

def render_performance():
    """Trade history and performance stats."""
    jm = get_journal()
    stats = jm.get_performance_stats()

    if stats['total_trades'] == 0:
        st.info("No closed trades yet.")
        return

    st.subheader("Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", stats['total_trades'])
    col2.metric("Win Rate", f"{stats['win_rate']:.0f}%")
    col3.metric("Total P&L", f"${stats['total_pnl']:+,.2f}")
    col4.metric("Avg P&L", f"{stats['avg_pnl_pct']:+.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Best Trade", f"{stats['best_trade']:+.1f}%")
    col6.metric("Worst Trade", f"{stats['worst_trade']:+.1f}%")
    col7.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    col8.metric("Avg Hold", f"{stats['avg_days_held']:.0f}d")

    # By signal type
    if stats['by_signal_type']:
        st.divider()
        st.subheader("By Signal Type")
        for st_name, st_data in stats['by_signal_type'].items():
            st.text(f"  {st_name}: {st_data['count']} trades, "
                    f"Win rate: {st_data['win_rate']:.0f}%, "
                    f"Avg: {st_data['avg_pnl_pct']:+.1f}%")

    # Trade history table
    history = jm.get_trade_history(last_n=20)
    if history:
        st.divider()
        st.subheader("Recent Trades")
        rows = []
        for t in history:
            pnl = t.get('realized_pnl_pct', 0)
            rows.append({
                'Ticker': t['ticker'],
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}",
                'P&L': f"{pnl:+.1f}%",
                'Days': t.get('days_held', 0),
                'Reason': t.get('exit_reason', '?'),
                'Signal': t.get('signal_type', '?'),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# =============================================================================
# APP MAIN
# =============================================================================

def main():
    render_sidebar()

    # Main content area
    tab_scanner, tab_perf = st.tabs(["🔍 Scanner", "📊 Performance"])

    with tab_scanner:
        render_scanner_table()

        if st.session_state.get('selected_analysis'):
            st.divider()
            render_detail_view()

    with tab_perf:
        render_performance()


if __name__ == "__main__":
    main()
