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
from datetime import datetime
from typing import Dict

# Backend imports
from signal_engine import EntrySignal
from data_fetcher import (
    fetch_all_ticker_data, fetch_scan_data, fetch_market_filter,
    clear_cache,
)
from scanner_engine import analyze_ticker, scan_watchlist, TickerAnalysis
from ai_analysis import analyze as run_ai_analysis
from chart_engine import render_tv_chart, render_mtf_chart
from journal_manager import JournalManager, WatchlistItem, Trade

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

# Initialize scan results
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = []

if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = None

if 'selected_analysis' not in st.session_state:
    st.session_state['selected_analysis'] = None

if 'ticker_data_cache' not in st.session_state:
    st.session_state['ticker_data_cache'] = {}


def get_journal() -> JournalManager:
    return st.session_state['journal']


# =============================================================================
# SIDEBAR — Watchlist & Controls
# =============================================================================

def render_sidebar():
    jm = get_journal()

    st.sidebar.title("📊 TTA v2")
    st.sidebar.caption("Technical Trading Assistant")

    # ── Watchlist Input ───────────────────────────────────────────────
    st.sidebar.subheader("Watchlist")

    watchlist_tickers = jm.get_watchlist_tickers()
    default_text = ", ".join(watchlist_tickers) if watchlist_tickers else ""

    new_tickers = st.sidebar.text_area(
        "Tickers (comma separated)",
        value=default_text,
        height=80,
        help="Enter tickers separated by commas. Example: AAPL, NVDA, META",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            tickers = [t.strip().upper() for t in new_tickers.split(",") if t.strip()]
            jm.clear_watchlist()
            jm.set_watchlist_tickers(tickers)
            st.sidebar.success(f"Saved {len(tickers)} tickers")
            st.rerun()
    with col2:
        if st.button("🔍 Scan", use_container_width=True, type="primary"):
            _run_scan()

    # ── Open Positions ────────────────────────────────────────────────
    open_trades = jm.get_open_trades()
    if open_trades:
        st.sidebar.divider()
        st.sidebar.subheader(f"Open Positions ({len(open_trades)})")
        for trade in open_trades:
            ticker = trade['ticker']
            entry = trade.get('entry_price', 0)
            stop = trade.get('current_stop', trade.get('initial_stop', 0))
            st.sidebar.text(f"{ticker}: ${entry:.2f} | Stop: ${stop:.2f}")

    # ── Market Filter ─────────────────────────────────────────────────
    st.sidebar.divider()
    mkt = fetch_market_filter()
    spy_ok = mkt.get('spy_above_200', True)
    vix_ok = mkt.get('vix_below_30', True)
    st.sidebar.markdown(
        f"**Market:** SPY {'✅' if spy_ok else '❌'} ${mkt.get('spy_close', '?')} | "
        f"VIX {'✅' if vix_ok else '❌'} {mkt.get('vix_close', '?')}"
    )


def _run_scan():
    """Execute watchlist scan."""
    jm = get_journal()
    tickers = jm.get_watchlist_tickers()

    if not tickers:
        st.sidebar.warning("Add tickers to watchlist first")
        return

    with st.spinner(f"Scanning {len(tickers)} tickers..."):
        all_data = fetch_scan_data(tickers)
        results = scan_watchlist(all_data)

        st.session_state['scan_results'] = results
        st.session_state['ticker_data_cache'] = all_data
        st.session_state['selected_ticker'] = None
        st.session_state['selected_analysis'] = None

    st.rerun()


# =============================================================================
# MAIN CONTENT — Scanner Results Table
# =============================================================================

def render_scanner_table():
    """Render the scan results as an interactive table."""
    results = st.session_state.get('scan_results', [])

    if not results:
        st.info("👆 Add tickers to your watchlist and click **Scan** to begin.")
        return

    st.subheader(f"Scan Results — {len(results)} tickers")

    # Build table data
    rows = []
    for r in results:
        rec = r.recommendation or {}
        q = r.quality or {}
        sig = r.signal

        macd_status = "✅" if sig and sig.macd.get('bullish') else "❌"
        ao_status = "✅" if sig and sig.ao.get('positive') else "❌"
        weekly_status = "✅" if sig and sig.weekly_macd.get('bullish') else "❌"
        monthly_status = "✅" if sig and sig.monthly_macd.get('bullish') else "❌"

        rows.append({
            'Ticker': r.ticker,
            'Rec': rec.get('recommendation', 'SKIP'),
            'Conv': f"{rec.get('conviction', 0)}/10",
            'MACD': macd_status,
            'AO': ao_status,
            'Wkly': weekly_status,
            'Mthly': monthly_status,
            'Quality': q.get('quality_grade', '?'),
            'Price': f"${r.current_price:.2f}" if r.current_price else "?",
            'Summary': rec.get('summary', ''),
        })

    df = pd.DataFrame(rows)

    # Color-code recommendation
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Ticker': st.column_config.TextColumn(width="small"),
            'Rec': st.column_config.TextColumn("Recommendation", width="medium"),
            'Conv': st.column_config.TextColumn("Conviction", width="small"),
            'MACD': st.column_config.TextColumn(width="small"),
            'AO': st.column_config.TextColumn(width="small"),
            'Wkly': st.column_config.TextColumn("Weekly", width="small"),
            'Mthly': st.column_config.TextColumn("Monthly", width="small"),
            'Quality': st.column_config.TextColumn(width="small"),
            'Price': st.column_config.TextColumn(width="small"),
            'Summary': st.column_config.TextColumn(width="large"),
        },
    )

    # Ticker selector
    st.divider()
    ticker_options = [r.ticker for r in results]
    selected = st.selectbox("Select ticker for detailed analysis", ticker_options,
                            index=0 if ticker_options else None)

    if selected and selected != st.session_state.get('selected_ticker'):
        st.session_state['selected_ticker'] = selected
        # Find the analysis
        for r in results:
            if r.ticker == selected:
                st.session_state['selected_analysis'] = r
                break
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
    """Interactive TradingView-style chart."""
    data_cache = st.session_state.get('ticker_data_cache', {})
    ticker_data = data_cache.get(ticker, {})
    daily = ticker_data.get('daily')

    if daily is None:
        st.warning("No chart data available")
        return

    from chart_engine import render_tv_chart, render_mtf_chart

    # Render TradingView chart — ALL data, LWC handles zoom natively
    # zoom_level=200 shows ~200 bars initially, user scrolls to zoom
    render_tv_chart(daily, ticker, signal=signal, height=750,
                    zoom_level=200, key=f"tv_{ticker}")

    # MTF chart
    weekly = ticker_data.get('weekly')
    monthly = ticker_data.get('monthly')
    if weekly is not None and monthly is not None:
        with st.expander("Multi-Timeframe View"):
            render_mtf_chart(daily, weekly, monthly, ticker, height=400,
                             key=f"mtf_{ticker}")


def _render_ai_tab(ticker: str, signal: EntrySignal,
                   rec: Dict, analysis: TickerAnalysis):
    """AI-enhanced analysis."""
    quality = analysis.quality or {}

    # Check for Gemini/OpenAI config
    gemini = None
    openai_client = None

    # Try to load Gemini
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            gemini = genai.GenerativeModel('gemini-2.0-flash')
    except Exception:
        pass

    if st.button("🤖 Run AI Analysis", type="primary"):
        with st.spinner("Analyzing..."):
            # Fetch fundamentals for AI context
            data_cache = st.session_state.get('ticker_data_cache', {})
            ticker_data = data_cache.get(ticker, {})

            # Get fundamentals (if not already fetched)
            fundamentals = {}
            try:
                from data_fetcher import (
                    fetch_ticker_info, fetch_options_data,
                    fetch_insider_transactions, fetch_institutional_holders,
                    fetch_earnings_date,
                )
                fundamentals = {
                    'info': fetch_ticker_info(ticker),
                    'options': fetch_options_data(ticker),
                    'insider': fetch_insider_transactions(ticker),
                    'institutional': fetch_institutional_holders(ticker),
                    'earnings': fetch_earnings_date(ticker),
                }
            except Exception as e:
                st.caption(f"Fundamentals fetch error: {e}")

            result = run_ai_analysis(
                ticker=ticker,
                signal=signal,
                recommendation=rec,
                quality=quality,
                fundamentals=fundamentals,
                gemini_model=gemini,
                openai_client=openai_client,
            )

            st.session_state[f'ai_result_{ticker}'] = result

    # Display result
    ai_result = st.session_state.get(f'ai_result_{ticker}')
    if ai_result:
        provider = ai_result.get('provider', 'unknown')
        st.caption(f"Provider: {provider} | {ai_result.get('note', '')}")

        col1, col2 = st.columns([1, 3])
        with col1:
            conv = ai_result.get('conviction', 0)
            color = "🟢" if conv >= 7 else ("🟡" if conv >= 4 else "🔴")
            st.metric("Conviction", f"{color} {conv}/10")
            st.metric("Timing", ai_result.get('timing', '?'))
            st.metric("Sizing", ai_result.get('position_sizing', '?'))

        with col2:
            st.markdown("**What the scanner misses:**")
            st.info(ai_result.get('scanner_misses', 'N/A'))
            st.markdown("**Red flags:**")
            flags = ai_result.get('red_flags', 'None')
            if flags and flags != 'None':
                st.warning(flags)
            else:
                st.success("No red flags")

        with st.expander("Full AI Response"):
            st.text(ai_result.get('raw_text', ''))


def _render_trade_tab(ticker: str, signal: EntrySignal,
                      analysis: TickerAnalysis):
    """Trade entry form and position management."""
    jm = get_journal()
    rec = analysis.recommendation or {}
    stops = signal.stops if signal else {}

    # Check if already in a position
    open_tickers = jm.get_open_tickers()
    if ticker in open_tickers:
        _render_position_management(ticker, jm)
        return

    # ── Entry Form ────────────────────────────────────────────────────
    st.subheader("Entry Setup")

    col1, col2, col3 = st.columns(3)
    with col1:
        entry_price = st.number_input("Entry Price",
                                       value=float(stops.get('entry', 0)),
                                       step=0.01, format="%.2f")
    with col2:
        stop_price = st.number_input("Stop Loss",
                                      value=float(stops.get('stop', 0)),
                                      step=0.01, format="%.2f")
    with col3:
        target_price = st.number_input("Target",
                                        value=float(stops.get('target', 0)),
                                        step=0.01, format="%.2f")

    col4, col5 = st.columns(2)
    with col4:
        position_size = st.number_input("Position Size ($)",
                                         value=10000.0, step=1000.0, format="%.0f")
    with col5:
        if entry_price > 0 and stop_price > 0:
            risk_pct = (entry_price - stop_price) / entry_price * 100
            shares = int(position_size / entry_price)
            st.metric("Shares", shares)
            st.caption(f"Risk: {risk_pct:.1f}% | ${(entry_price - stop_price) * shares:.0f}")

    notes = st.text_input("Notes", value=rec.get('summary', ''))

    if st.button("✅ Enter Trade", type="primary"):
        trade = Trade(
            trade_id='',
            ticker=ticker,
            entry_price=entry_price,
            initial_stop=stop_price,
            target=target_price,
            position_size=position_size,
            signal_type=rec.get('signal_type', ''),
            quality_grade=analysis.quality.get('quality_grade', '') if analysis.quality else '',
            conviction_at_entry=rec.get('conviction', 0),
            weekly_bullish_at_entry=signal.weekly_macd.get('bullish', False) if signal else False,
            monthly_bullish_at_entry=signal.monthly_macd.get('bullish', False) if signal else False,
            weinstein_stage_at_entry=signal.weinstein.get('stage', 0) if signal else 0,
            notes=notes,
        )
        result = jm.enter_trade(trade)
        st.success(result)
        st.rerun()


def _render_position_management(ticker: str, jm: JournalManager):
    """Manage an existing open position."""
    trades = jm.get_open_trades()
    trade = next((t for t in trades if t['ticker'] == ticker), None)
    if not trade:
        return

    st.subheader(f"Open Position — {ticker}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Entry", f"${trade.get('entry_price', 0):.2f}")
    col2.metric("Stop", f"${trade.get('current_stop', trade.get('initial_stop', 0)):.2f}")
    col3.metric("Target", f"${trade.get('target', 0):.2f}")

    st.caption(f"Shares: {trade.get('shares', 0):.0f} | "
               f"Signal: {trade.get('signal_type', '?')} | "
               f"Opened: {trade.get('entry_date', '?')}")

    # Trail stop
    st.divider()
    new_stop = st.number_input("New Stop", value=float(trade.get('current_stop', 0)),
                                step=0.50, format="%.2f")
    if st.button("📈 Trail Stop"):
        result = jm.update_stop(ticker, new_stop)
        st.info(result)
        st.rerun()

    # Close position
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        exit_price = st.number_input("Exit Price", value=0.0, step=0.01, format="%.2f")
    with col2:
        exit_reason = st.selectbox("Exit Reason",
                                    ['manual', 'stop_loss', 'target_hit',
                                     'weekly_cross', 'time_exit'])

    if st.button("🔴 Close Position"):
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
