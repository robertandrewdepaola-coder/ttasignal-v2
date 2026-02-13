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
import re
from datetime import datetime, timedelta
from typing import Dict, List

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
# AI TEXT CLEANUP — fix garbled formatting from LLM outputs
# =============================================================================

def clean_ai_formatting(text: str) -> str:
    """Fix common AI output formatting issues with currency, percentages, and spacing.
    
    Handles:
    - Missing spaces after dollar amounts ($184.54Buy → $184.54 Buy)
    - Missing spaces before dollar amounts (target$210 → target $210)
    - Letter-number concatenation (gained27% → gained 27%)
    - Percentage concatenation (27%gains → 27% gains)
    - Punctuation spacing (end.Start → end. Start)
    - Em-dash spacing (word—word → word — word)
    - Number-letter concatenation (27times → 27 times)
    - Preserves: 200d, 50d, 1x, $15.2K, $3.2B, Q3, 1st/2nd/3rd
    """
    if not text:
        return text

    # Fix dollar amounts followed by words: $184.54Buy → $184.54 Buy
    text = re.sub(r'(\$\d+[\d,.]*[KMBkmb]?)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'(\$\d+\.?\d{2})([a-z])', r'\1 \2', text)

    # Fix missing spaces before dollar amounts
    text = re.sub(r'([a-zA-Z])(\$\d)', r'\1 \2', text)

    # Fix letter-number concatenation (gained27 → gained 27)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)

    # Fix percentage spacing
    text = re.sub(r'(\d+\.?\d*)%([a-zA-Z])', r'\1% \2', text)

    # Fix missing spaces after sentence-ending punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text)

    # Fix em-dash spacing
    text = re.sub(r'([a-zA-Z])—([a-zA-Z])', r'\1 — \2', text)

    # Fix number-letter concatenation (27times → 27 times)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Restore common abbreviations that should NOT have spaces
    text = re.sub(r'(\d+) ([dxwDXW])\b', r'\1\2', text)   # 200d, 2.5x, 52w
    text = re.sub(r'(\d+) ([KMBkmb])\b', r'\1\2', text)   # $15.2K, $3.2B
    text = re.sub(r'(\d) (st|nd|rd|th)\b', r'\1\2', text)  # 1st, 2nd, 3rd
    text = re.sub(r'\bQ (\d)\b', r'Q\1', text)             # Q1, Q2, Q3, Q4

    # Clean multiple spaces (preserve markdown indentation)
    text = re.sub(r'(?<!\n) {2,}', ' ', text)

    return text.strip()

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="TTA v2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# AI PROVIDER AUTO-DETECTION — supports Groq (gsk_) and xAI/Grok (xai-)
# =============================================================================

def _detect_ai_provider(api_key: str) -> Dict:
    """
    Auto-detect AI provider from API key prefix.
    Returns config dict with base_url, models, and provider name.
    
    Supported:
      - Groq (groq.com): keys start with 'gsk_', endpoint api.groq.com
      - xAI/Grok (x.ai): keys start with 'xai-', endpoint api.x.ai
    """
    key = (api_key or "").strip().strip('"').strip("'").strip()
    
    if key.startswith("gsk_"):
        return {
            'provider': 'groq',
            'base_url': 'https://api.groq.com/openai/v1',
            'model': 'llama-3.3-70b-versatile',
            'fallback_model': 'llama-3.1-8b-instant',
            'key': key,
            'display': f'Groq (gsk_...{key[-4:]})',
        }
    elif key.startswith("xai-") or key.startswith("xai_"):
        return {
            'provider': 'xai',
            'base_url': 'https://api.x.ai/v1',
            'model': 'grok-3-fast',
            'fallback_model': 'grok-3-mini-fast',
            'key': key,
            'display': f'xAI/Grok (xai-...{key[-4:]})',
        }
    elif key:
        # Unknown prefix — try Groq format as default
        return {
            'provider': 'unknown',
            'base_url': 'https://api.groq.com/openai/v1',
            'model': 'llama-3.3-70b-versatile',
            'fallback_model': 'llama-3.1-8b-instant',
            'key': key,
            'display': f'Unknown ({key[:6]}...{key[-4:]})',
        }
    else:
        return {
            'provider': 'none',
            'base_url': '',
            'model': '',
            'fallback_model': '',
            'key': '',
            'display': 'Not configured',
        }

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

# Fetch sector rotation on startup if not already loaded (critical for sector colors)
if 'sector_rotation' not in st.session_state:
    try:
        from data_fetcher import fetch_sector_rotation
        st.session_state['sector_rotation'] = fetch_sector_rotation()
    except Exception:
        st.session_state['sector_rotation'] = {}

# Pre-fetch SPY/VIX for APEX chart signals (avoids mid-render fetch on chart tab)
if 'apex_spy_data' not in st.session_state:
    try:
        from data_fetcher import fetch_daily
        st.session_state['apex_spy_data'] = fetch_daily("SPY")
        st.session_state['apex_vix_data'] = fetch_daily("^VIX")
    except Exception:
        st.session_state['apex_spy_data'] = None
        st.session_state['apex_vix_data'] = None

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
# MORNING BRIEFING — AI Market Narrative (sidebar)
# =============================================================================

def _render_morning_briefing():
    """
    DUAL ANALYSIS SYSTEM:
    Part B — Deep Structural Analysis with Sector ETF Rotation ("Juan's Market Filter")
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # ══════════════════════════════════════════════════════════════════
    # DEEP STRUCTURAL ANALYSIS — Sector Rotation + 5-Factor Score
    # ══════════════════════════════════════════════════════════════════
    deep_data = st.session_state.get('deep_market_analysis')
    deep_date = st.session_state.get('deep_analysis_date', '')

    with st.sidebar.expander("🏛️ Market Structure", expanded=True):
        if deep_data and deep_date == today:
            score = deep_data.get('score', 0)
            label = deep_data.get('score_label', 'Neutral')
            factors = deep_data.get('factors', {})
            phases = deep_data.get('sectors_by_phase', {})

            # ── Score Banner ──────────────────────────────────────────
            if score >= 2:
                st.success(f"**{score:+d}/5 {label}**")
            elif score <= -2:
                st.error(f"**{score:+d}/5 {label}**")
            elif score >= 1:
                st.info(f"**{score:+d}/5 {label}**")
            elif score <= -1:
                st.warning(f"**{score:+d}/5 {label}**")
            else:
                st.info(f"**{score:+d}/5 {label}**")

            # ── 5-Factor Scores (compact) ─────────────────────────────
            factor_labels = {
                'sp500': 'S&P 500',
                'vix': 'VIX/Commercials',
                'dollar': 'US Dollar',
                'cost_of_money': 'Cost of Money',
                'rotation': 'Rotation/Breadth',
            }
            for key, display_name in factor_labels.items():
                val = factors.get(key, '')
                if val:
                    st.markdown(f"**{display_name}:** {val}")

            # ── Sector ETF Rotation Table ─────────────────────────────
            st.divider()
            st.markdown("**📊 Sector ETF Rotation**")

            def _show_phase(emoji, phase_name, items):
                if items:
                    for s in sorted(items, key=lambda x: x.get('vs_spy_20d', 0), reverse=True):
                        vs5 = s.get('vs_spy_5d', 0)
                        vs20 = s.get('vs_spy_20d', 0)
                        st.caption(
                            f"{emoji} **{s['etf']}** {s['short']}  "
                            f"5d:{vs5:+.1f}% 20d:{vs20:+.1f}%"
                        )

            leading = phases.get('LEADING', [])
            emerging = phases.get('EMERGING', [])
            fading = phases.get('FADING', [])
            lagging = phases.get('LAGGING', [])

            if leading:
                st.markdown("🟢 **Leading** — trade these")
                _show_phase("🟢", "LEADING", leading)
            if emerging:
                st.markdown("🔵 **Emerging** — watch for entries")
                _show_phase("🔵", "EMERGING", emerging)
            if fading:
                st.markdown("🟡 **Fading** — tighten stops")
                _show_phase("🟡", "FADING", fading)
            if lagging:
                st.markdown("🔴 **Lagging** — avoid")
                _show_phase("🔴", "LAGGING", lagging)

            # ── AI Narrative (expandable) ─────────────────────────────
            analysis_text = deep_data.get('analysis', '')
            if analysis_text:
                with st.expander("📝 Full Analysis"):
                    # Extract narrative sections
                    for section_header in ['SECTOR ROTATION NARRATIVE:', 'WHAT TO TRADE:', 'WHAT TO AVOID:',
                                           'STRUCTURAL READ:', 'ACTIONABLE GUIDANCE:']:
                        if section_header in analysis_text:
                            idx = analysis_text.index(section_header)
                            remaining = analysis_text[idx + len(section_header):]
                            # Find end of section
                            end = len(remaining)
                            for marker in ['FACTOR SCORES:', 'SECTOR ROTATION NARRATIVE:',
                                           'WHAT TO TRADE:', 'WHAT TO AVOID:', 'STRUCTURAL READ:',
                                           'ACTIONABLE GUIDANCE:', 'SCORE:', 'LABEL:']:
                                if marker in remaining and remaining.index(marker) > 0:
                                    end = min(end, remaining.index(marker))
                            section_text = remaining[:end].strip()
                            if section_text:
                                st.markdown(f"**{section_header.replace(':', '')}**")
                                st.caption(section_text)

                    # Provider info
                    provider = deep_data.get('provider', '?')
                    st.caption(f"_via {provider}_")
        else:
            st.caption("Click refresh to generate sector rotation analysis")

        # Refresh button
        if st.button("🔄 Refresh Analysis", use_container_width=True,
                     key="refresh_deep_analysis"):
            _run_deep_analysis()


def _run_deep_analysis():
    """Run the deep structural market analysis with sector ETF rotation."""
    with st.spinner("Analyzing market structure & sector rotation..."):
        try:
            from data_fetcher import fetch_macro_narrative_data, fetch_market_filter, fetch_sector_rotation
            from ai_analysis import generate_deep_market_analysis

            macro_data = fetch_macro_narrative_data()
            market_filter = fetch_market_filter()
            sector_rotation = fetch_sector_rotation()

            gemini_model = st.session_state.get('gemini_model')
            openai_client = st.session_state.get('openai_client')

            result = generate_deep_market_analysis(
                macro_data,
                market_filter=market_filter,
                sector_rotation=sector_rotation,
                gemini_model=gemini_model,
                openai_client=openai_client,
                ai_model=st.session_state.get('_ai_config', {}).get('model', 'llama-3.3-70b-versatile'),
            )

            result['macro_data'] = macro_data
            st.session_state['deep_market_analysis'] = result
            st.session_state['deep_analysis_date'] = datetime.now().strftime('%Y-%m-%d')
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Analysis error: {e}")


def _render_factual_market_brief():
    """
    PART A: Factual market brief — replaces the old green/yellow/red market health rectangle.
    Shows indices, VIX, breadth data + AI narrative summary.
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # Cache market filter in session_state (survives reruns, refreshes via data_fetcher's 5min TTL)
    if 'market_filter_data' not in st.session_state:
        st.session_state['market_filter_data'] = fetch_market_filter()
    mkt = st.session_state['market_filter_data']
    spy_ok = mkt.get('spy_above_200', True)
    vix_close = mkt.get('vix_close', 0) or 0

    # ── Market Status Line (compact) ──────────────────────────────────
    spy_str = f"SPY {'✅' if spy_ok else '❌'} ${mkt.get('spy_close', '?')}"

    if vix_close < 15:
        vix_icon = "🟢"
    elif vix_close < 20:
        vix_icon = "🟡"
    elif vix_close < 25:
        vix_icon = "🟠"
    elif vix_close < 30:
        vix_icon = "🔴"
    else:
        vix_icon = "🔴🔴"
    vix_str = f"VIX {vix_icon} {vix_close}"

    if spy_ok and vix_close < 20:
        st.sidebar.success(f"**{spy_str} | {vix_str}**")
    elif spy_ok and vix_close < 30:
        st.sidebar.warning(f"**{spy_str} | {vix_str}**")
    else:
        st.sidebar.error(f"**{spy_str} | {vix_str}**")

    # ── AI Narrative (if generated) ───────────────────────────────────
    narrative_data = st.session_state.get('morning_narrative')
    narrative_date = st.session_state.get('morning_narrative_date', '')

    if narrative_data and narrative_date == today:
        regime = narrative_data.get('regime', 'Neutral')
        regime_colors = {
            'Risk-On': '🟢', 'Bullish': '🟢', 'Cautiously Bullish': '🟢',
            'Neutral': '🟡', 'Balanced': '🟡',
            'Caution': '🟠', 'Rotation to Safety': '🟠',
            'Risk-Off': '🔴', 'Bearish': '🔴',
        }
        icon = regime_colors.get(regime, '🟡')

        with st.sidebar.expander(f"{icon} Market Brief — {regime}"):
            st.caption(narrative_data.get('narrative', '')[:400])

            # Raw data — multi-timeframe momentum analysis
            macro = narrative_data.get('macro_data', {})
            if macro:
                st.divider()
                st.caption("**Index Momentum**")
                for name, info in macro.get('indices', {}).items():
                    d1 = info.get('1d', 0)
                    d5 = info.get('5d', 0)
                    d20 = info.get('20d', 0)
                    price = info.get('price', '?')

                    # Color based on multi-timeframe health:
                    # Green = strong (positive on all), Yellow = mixed, Red = weak
                    pos_count = sum(1 for x in [d1, d5, d20] if x > 0)
                    if pos_count == 3 and d5 > 1.0:
                        ic = '🟢'  # Strong uptrend
                    elif pos_count >= 2:
                        ic = '🟡'  # Mixed — momentum fading or just starting
                    elif pos_count == 1:
                        ic = '🟠'  # Mostly negative — caution
                    else:
                        ic = '🔴'  # Downtrend on all timeframes

                    # Show directional arrows for each timeframe
                    d1_arrow = '↑' if d1 > 0.3 else ('↓' if d1 < -0.3 else '→')
                    d5_arrow = '↑' if d5 > 0.5 else ('↓' if d5 < -0.5 else '→')
                    d20_arrow = '↑' if d20 > 1.0 else ('↓' if d20 < -1.0 else '→')

                    st.caption(
                        f"{ic} **{name}**: ${price} — "
                        f"1d:{d1_arrow}{d1:+.1f}% | 5d:{d5_arrow}{d5:+.1f}% | 20d:{d20_arrow}{d20:+.1f}%"
                    )

                vix_data = macro.get('vix', {})
                if vix_data:
                    vix_lvl = vix_data.get('level', 0)
                    vix_chg = vix_data.get('change_5d', 0)
                    vix_regime = vix_data.get('regime', '')
                    vix_ic = '🟢' if vix_lvl < 15 else ('🟡' if vix_lvl < 20 else ('🟠' if vix_lvl < 25 else '🔴'))
                    chg_arrow = '↑' if vix_chg > 1 else ('↓' if vix_chg < -1 else '→')
                    st.caption(f"{vix_ic} **VIX**: {vix_lvl} ({vix_regime}) | 5d:{chg_arrow}{vix_chg:+.1f}")

                sectors = macro.get('sectors', {})
                if sectors:
                    spread = sectors.get('spread', 0)
                    regime_str = sectors.get('regime', '')
                    sec_ic = '🟢' if spread > 2 else ('🔴' if spread < -2 else '🟡')
                    st.caption(f"{sec_ic} **Rotation**: {regime_str} (Off-Def: {spread:+.1f}%)")

                breadth = macro.get('breadth', {})
                if breadth:
                    br_spread = breadth.get('spread', 0)
                    br_regime = breadth.get('regime', '')
                    br_ic = '🟢' if br_spread > 1 else ('🔴' if br_spread < -1 else '🟡')
                    st.caption(f"{br_ic} **Breadth**: {br_regime} (RSP-SPY: {br_spread:+.1f}%)")

        # Refresh button
        if st.sidebar.button("🔄 Refresh Brief", use_container_width=True, key="refresh_brief"):
            _run_factual_brief()
    else:
        if st.sidebar.button("📊 Generate Market Brief", use_container_width=True, key="gen_brief"):
            _run_factual_brief()


def _run_factual_brief():
    """Generate the factual morning brief."""
    with st.spinner("Generating market brief..."):
        try:
            from data_fetcher import fetch_macro_narrative_data
            from ai_analysis import generate_market_narrative

            macro_data = fetch_macro_narrative_data()
            gemini_model = st.session_state.get('gemini_model')
            openai_client = st.session_state.get('openai_client')

            narrative_result = generate_market_narrative(
                macro_data,
                gemini_model=gemini_model,
                openai_client=openai_client,
                ai_model=st.session_state.get('_ai_config', {}).get('model', 'llama-3.3-70b-versatile'),
            )

            narrative_result['macro_data'] = macro_data
            st.session_state['morning_narrative'] = narrative_result
            st.session_state['morning_narrative_date'] = datetime.now().strftime('%Y-%m-%d')
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Brief error: {e}")


# =============================================================================
# SIDEBAR — Slim: Scan controls, Open Positions, Alerts, Market
# =============================================================================

def render_sidebar():
    jm = get_journal()

    st.sidebar.title("📊 TTA v2")
    st.sidebar.caption("Technical Trading Assistant")

    # ══════════════════════════════════════════════════════════════════
    # PART B: Deep Structural Analysis ("Juan's Market Filter")
    # ══════════════════════════════════════════════════════════════════
    _render_morning_briefing()

    st.sidebar.divider()

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

        # Cache position prices — refresh every 60 seconds, not every rerun
        _pos_cache = st.session_state.get('_position_prices', {})
        _pos_ts = st.session_state.get('_position_prices_ts', 0)
        _now_ts = datetime.now().timestamp()
        _stale = (_now_ts - _pos_ts) > 60  # 60-second TTL

        if _stale:
            _new_prices = {}
            for trade in open_trades:
                t = trade['ticker']
                _new_prices[t] = fetch_current_price(t) or trade.get('entry_price', 0)
            st.session_state['_position_prices'] = _new_prices
            st.session_state['_position_prices_ts'] = _now_ts
            _pos_cache = _new_prices

        for trade in open_trades:
            ticker = trade['ticker']
            entry = trade.get('entry_price', 0)
            current = _pos_cache.get(ticker, entry)
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            icon = "🟢" if pnl_pct >= 0 else "🔴"

            if st.sidebar.button(
                f"{icon} {ticker}  ${current:.2f}  ({pnl_pct:+.1f}%)",
                key=f"sidebar_pos_{ticker}",
                use_container_width=True,
            ):
                _load_ticker_for_view(ticker)

    # ── Market Brief (replaces old green/yellow/red rectangle) ──────
    st.sidebar.divider()
    _render_factual_market_brief()

    # ── Settings (bottom of sidebar) ──────────────────────────────────
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            if st.button("🔑 Reset API", key="reset_api_cache",
                         help="Clear cached API key status after updating secrets"):
                for k in list(st.session_state.keys()):
                    if k.startswith(('_groq_key', '_groq_validated', 'ai_result_', 'chat_')):
                        st.session_state.pop(k, None)
                st.toast("✅ API cache cleared. Click a ticker to re-run.")
        with s2:
            if st.button("📊 Refresh Mkt", key="refresh_market_data",
                         help="Re-fetch SPY, VIX, sector rotation"):
                st.session_state.pop('market_filter_data', None)
                st.session_state.pop('sector_rotation', None)
                st.session_state.pop('_research_market_cache', None)
                st.session_state.pop('_research_market_cache_ts', None)
                st.session_state.pop('_position_prices', None)
                st.session_state.pop('_position_prices_ts', None)
                # Clear APEX caches (SPY/VIX data + per-ticker detection)
                st.session_state.pop('apex_spy_data', None)
                st.session_state.pop('apex_vix_data', None)
                for k in [k for k in st.session_state if k.startswith('_apex_cache_')]:
                    st.session_state.pop(k, None)
                st.toast("✅ Market data will refresh on next load.")

        # Key diagnostic
        try:
            _diag_key = st.secrets.get("GROQ_API_KEY", "")
            if _diag_key:
                _diag_cfg = _detect_ai_provider(_diag_key)
                _status = st.session_state.get('_groq_key_status', 'not tested')
                st.caption(f"🔑 API key: `{_diag_cfg['key'][:8]}...{_diag_cfg['key'][-4:]}` "
                           f"({len(_diag_cfg['key'])} chars)")
                st.caption(f"   Provider: **{_diag_cfg['provider'].upper()}** → {_diag_cfg['base_url']} | "
                           f"Model: {_diag_cfg['model']} | Status: {_status}")
                if _diag_cfg['provider'] == 'unknown':
                    st.warning("⚠️ Key prefix not recognized. Expected `gsk_` (Groq) or `xai-` (xAI/Grok).")
            else:
                st.caption("🔑 API key: **not set** — add GROQ_API_KEY to secrets")
            _gem_key = st.secrets.get("GEMINI_API_KEY", "")
            st.caption(f"🤖 Gemini key: {'set (' + _gem_key[:6] + '...)' if _gem_key else '**not set** (optional fallback)'}")
        except Exception:
            st.caption("🔑 Could not read secrets")


def _load_ticker_for_view(ticker: str):
    """Load a ticker for the detail view — works for ANY ticker (open positions, conditionals, etc.).
    
    NOTE: No st.rerun() needed here. Button clicks already trigger a Streamlit rerun.
    Setting session state is enough — render_detail_view() runs later in the same pass
    and picks up the new state. Avoiding rerun eliminates a full double-repaint of the
    200+ ticker scanner table.
    """
    ticker = ticker.upper().strip()

    # Check if we already have analysis from a scan (instant — no API call)
    results = st.session_state.get('scan_results', [])
    for r in results:
        if r.ticker == ticker:
            st.session_state['selected_ticker'] = ticker
            st.session_state['selected_analysis'] = r
            st.session_state['scroll_to_detail'] = True
            return  # No rerun — current pass will render detail view

    # Fetch fresh data and analyze on-the-fly (only for tickers not in scan)
    try:
        data = fetch_all_ticker_data(ticker)
        if data.get('daily') is not None:
            analysis = analyze_ticker(data)
            st.session_state['selected_ticker'] = ticker
            st.session_state['selected_analysis'] = analysis
            st.session_state['scroll_to_detail'] = True
            # Cache the data
            cache = st.session_state.get('ticker_data_cache', {})
            cache[ticker] = data
            st.session_state['ticker_data_cache'] = cache
            # Clear stale APEX cache (new data = needs re-detection)
            st.session_state.pop(f'_apex_cache_{ticker}', None)
            # No rerun — current pass will render detail view
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

        # Fetch sector rotation (independent — failure doesn't block earnings)
        try:
            from data_fetcher import fetch_sector_rotation
            sector_rotation = fetch_sector_rotation()
            st.session_state['sector_rotation'] = sector_rotation
        except Exception as e:
            print(f"Sector rotation error: {e}")

        # Fetch earnings flags (independent — failure doesn't block sectors)
        try:
            from data_fetcher import fetch_batch_earnings_flags
            all_scan_tickers = [r.ticker for r in new_results]
            earnings_flags = fetch_batch_earnings_flags(all_scan_tickers, days_ahead=14)
            if earnings_flags:  # Only update if we got results
                if mode == 'new_only':
                    existing_flags = st.session_state.get('earnings_flags', {})
                    existing_flags.update(earnings_flags)
                    st.session_state['earnings_flags'] = existing_flags
                else:
                    st.session_state['earnings_flags'] = earnings_flags
        except Exception as e:
            print(f"Earnings flags error: {e}")

        # Fetch sectors for scanned tickers (independent)
        try:
            from data_fetcher import get_ticker_sector
            ticker_sectors = st.session_state.get('ticker_sectors', {})
            for r in new_results:
                if r.ticker not in ticker_sectors:
                    sector = get_ticker_sector(r.ticker)
                    if sector:
                        ticker_sectors[r.ticker] = sector
            st.session_state['ticker_sectors'] = ticker_sectors
        except Exception as e:
            print(f"Sector assignment error: {e}")

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

        ticker_sectors = st.session_state.get('ticker_sectors', {})

        summary = []
        for r in results_for_summary:
            rec = r.recommendation or {}
            q = r.quality or {}
            sig = r.signal

            # Volume string for persistence
            vol = r.volume or 0
            avg_vol = r.avg_volume_50d or 0
            vol_ratio = r.volume_ratio or 0
            if vol >= 1_000_000:
                vol_str = f"{vol/1_000_000:.1f}M"
            elif vol >= 1_000:
                vol_str = f"{vol/1_000:.0f}K"
            else:
                vol_str = str(int(vol)) if vol else ""
            if vol_ratio >= 2.0:
                vol_str = f"🔥{vol_str}"
            elif vol_ratio >= 1.5:
                vol_str = f"📈{vol_str}"

            # Earnings data for persistence
            earn = earnings_flags.get(r.ticker, {})
            earn_date = earn.get('next_earnings', '')
            earn_days = earn.get('days_until', 999)

            # Re-entry recency (bars ago)
            reentry_bars_ago = 0
            if r.reentry and r.reentry.get('is_valid'):
                reentry_bars_ago = r.reentry.get('macd_cross_bars_ago', 0)
            elif r.late_entry and r.late_entry.get('is_valid'):
                reentry_bars_ago = r.late_entry.get('days_since_cross', 0)

            # Sector phase for filtering persistence
            sector_name = ticker_sectors.get(r.ticker, '')
            sector_info = sector_rotation.get(sector_name, {})
            sector_phase = sector_info.get('phase', '')

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
                'sector': sector_name,
                'sector_phase': sector_phase,
                'ao_divergence_active': r.ao_divergence_active,
                'apex_buy': r.apex_buy,
                'volume_str': vol_str,
                'volume_ratio': vol_ratio,
                'earn_date': earn_date,
                'earn_days': earn_days,
                'reentry_bars_ago': reentry_bars_ago,
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
    favorite_tickers = set(jm.get_favorite_tickers())

    # Watchlist version counter — used to force text_area reset when watchlist changes
    if 'wl_version' not in st.session_state:
        st.session_state['wl_version'] = 0

    with st.expander(f"📋 Watchlist ({len(watchlist_tickers)} tickers) — click to edit",
                     expanded=(len(watchlist_tickers) == 0)):

        # ── Quick Add (single ticker) ─────────────────────────────────
        qa1, qa2 = st.columns([3, 1])
        with qa1:
            new_ticker = st.text_input("Add ticker", placeholder="e.g. AAPL",
                                       key="wl_add_input", label_visibility="collapsed")
        with qa2:
            if st.button("➕ Add", key="wl_add_btn", use_container_width=True):
                if new_ticker:
                    ticker_clean = new_ticker.strip().upper()
                    if ticker_clean and ticker_clean not in watchlist_tickers:
                        jm.add_to_watchlist(WatchlistItem(ticker=ticker_clean))
                        st.session_state['wl_version'] += 1  # Force text_area refresh
                        st.rerun()

        # ── Sort Controls ─────────────────────────────────────────────
        if watchlist_tickers:
            sort_col1, sort_col2 = st.columns([1, 1])
            with sort_col1:
                wl_sort = st.selectbox(
                    "Sort", ["⭐ Favorites first", "A-Z", "Z-A", "Date added"],
                    key="wl_sort", label_visibility="collapsed",
                )
            with sort_col2:
                st.caption(f"{len(watchlist_tickers)} tickers"
                           + (f" | ⭐{len(favorite_tickers)}" if favorite_tickers else ""))

        # ── Interactive List with Favorite/Delete ─────────────────────
        if watchlist_tickers:
            # Apply sort
            if wl_sort == "A-Z":
                sorted_tickers = sorted(watchlist_tickers)
            elif wl_sort == "Z-A":
                sorted_tickers = sorted(watchlist_tickers, reverse=True)
            elif wl_sort == "Date added":
                sorted_tickers = list(watchlist_tickers)  # Original order = date added
            else:  # Favorites first (default)
                sorted_tickers = sorted(
                    watchlist_tickers,
                    key=lambda t: (0 if t in favorite_tickers else 1, t)
                )

            # Show in compact rows
            for t in sorted_tickers:
                is_fav = t in favorite_tickers
                fav_icon = "⭐" if is_fav else "☆"

                tc1, tc2, tc3, tc4 = st.columns([0.5, 2.5, 0.5, 0.5])
                with tc1:
                    if st.button(fav_icon, key=f"fav_{t}",
                                 help="Toggle favorite"):
                        jm.toggle_favorite(t)
                        st.rerun()
                with tc2:
                    st.caption(f"{'⭐ ' if is_fav else ''}{t}")
                with tc3:
                    if st.button("📈", key=f"chart_{t}",
                                 help="Open chart"):
                        st.session_state['default_detail_tab'] = 1  # Chart tab index
                        _load_ticker_for_view(t)
                with tc4:
                    if st.button("🗑️", key=f"del_{t}",
                                 help="Delete"):
                        jm.delete_single_ticker(t)
                        st.session_state['wl_version'] += 1  # Force text_area refresh
                        # Also remove from scan results
                        if 'scan_results' in st.session_state:
                            st.session_state['scan_results'] = [
                                r for r in st.session_state['scan_results']
                                if r.ticker != t
                            ]
                        if 'scan_results_summary' in st.session_state:
                            st.session_state['scan_results_summary'] = [
                                s for s in st.session_state['scan_results_summary']
                                if s.get('ticker') != t
                            ]
                        st.rerun()

        # ── Bulk Editor (for pasting 200 tickers) ─────────────────────
        with st.expander("📝 Bulk Edit (paste tickers)"):
            st.caption("Paste tickers separated by commas, spaces, or new lines.")

            # PERSISTENCE FIX: Use dynamic key that resets when watchlist changes
            # This prevents stale text_area values from overwriting new additions
            wl_ver = st.session_state.get('wl_version', 0)
            default_text = ", ".join(sorted(watchlist_tickers)) if watchlist_tickers else ""
            new_text = st.text_area(
                "Tickers",
                value=default_text,
                height=100 if len(watchlist_tickers) > 20 else 68,
                label_visibility="collapsed",
                key=f"watchlist_editor_v{wl_ver}",  # Dynamic key forces fresh value
            )

            wl_col1, wl_col2, wl_col3 = st.columns([1, 1, 2])
            with wl_col1:
                if st.button("💾 Save", use_container_width=True, type="primary",
                             key="wl_save"):
                    import re
                    raw = re.split(r'[,\s\n\t]+', new_text)
                    tickers = [t.strip().upper() for t in raw if t.strip()]
                    seen = set()
                    unique = []
                    for t in tickers:
                        if t not in seen:
                            seen.add(t)
                            unique.append(t)

                    # Preserve favorites across bulk save
                    old_favorites = set(jm.get_favorite_tickers())
                    jm.clear_watchlist()
                    jm.set_watchlist_tickers(unique)
                    # Re-apply favorites
                    for fav in old_favorites:
                        if fav in unique:
                            jm.toggle_favorite(fav)

                    st.session_state['wl_version'] += 1
                    st.success(f"✅ Saved {len(unique)} tickers")
                    st.rerun()
            with wl_col2:
                if st.button("🗑️ Clear All", use_container_width=True, key="wl_clear"):
                    jm.clear_watchlist()
                    st.session_state['wl_version'] += 1
                    st.session_state.pop('scan_results', None)
                    st.session_state.pop('scan_results_summary', None)
                    st.session_state.pop('ticker_data_cache', None)
                    st.rerun()
            with wl_col3:
                st.caption(f"{len(watchlist_tickers)} saved"
                           + (f" | ⭐ {len(favorite_tickers)} favorites" if favorite_tickers else ""))

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

    # ── Sector Rotation Strip ─────────────────────────────────────────
    sector_rotation = st.session_state.get('sector_rotation', {})
    if sector_rotation:
        # Build compact sector strip using phase classification
        leading = []
        emerging = []
        fading = []
        lagging = []
        for sector, info in sector_rotation.items():
            short = info.get('short_name', sector[:4])
            vs = info.get('vs_spy_20d', 0)
            phase = info.get('phase', '')
            label = f"{short} ({vs:+.1f}%)"
            if phase == 'LEADING':
                leading.append(label)
            elif phase == 'EMERGING':
                emerging.append(label)
            elif phase == 'FADING':
                fading.append(label)
            elif phase == 'LAGGING':
                lagging.append(label)

        parts = []
        if leading:
            parts.append(f"🟢 **Leading:** {', '.join(leading)}")
        if emerging:
            parts.append(f"🔵 **Emerging:** {', '.join(emerging)}")
        if lagging:
            parts.append(f"🔴 **Lagging:** {', '.join(lagging)}")
        if parts:
            st.caption(" | ".join(parts))

    # ── Filter Bar ────────────────────────────────────────────────────
    filt_col1, filt_col2, filt_col3, filt_col4, filt_col5 = st.columns([2, 1.5, 1.5, 2, 1])

    with filt_col1:
        rec_filter = st.selectbox("Filter", [
            "All", "Signals Only", "BUY+", "STRONG BUY", "Quality A-B",
            "🟢 Focus", "🟡 Focus", "🔴 Focus", "🔵 Focus", "Any Focus",
            "Open Positions", "⚡ Earnings Soon"
        ], key="scan_filter", label_visibility="collapsed")

    with filt_col2:
        sector_filter = st.selectbox("Sector", [
            "All Sectors", "🟢 Leading", "🔵 Emerging", "🟡 Fading", "🔴 Lagging"
        ], key="sector_filter", label_visibility="collapsed")

    with filt_col3:
        sort_by = st.selectbox("Sort", [
            "Signal Strength ↓", "Conviction ↓", "Name A-Z", "Name Z-A",
            "Quality ↓", "Price ↓", "Price ↑", "Default"
        ], key="scan_sort", label_visibility="collapsed")

    with filt_col4:
        search = st.text_input("Search", placeholder="Filter by ticker...",
                                key="scan_search", label_visibility="collapsed")

    with filt_col5:
        st.caption(f"**{len(rows)}** total")

    # Reset pagination when filter/sort/search changes
    _filter_sig = f"{rec_filter}|{sector_filter}|{sort_by}|{search}"
    if st.session_state.get('_last_filter_sig') != _filter_sig:
        st.session_state['scanner_page'] = 0
        st.session_state['_last_filter_sig'] = _filter_sig

    # ── Build focus label lookup ──────────────────────────────────────
    focus_labels = jm.get_focus_labels()

    # Apply filters
    filtered = rows.copy()

    if search:
        search_upper = search.upper()
        filtered = [r for r in filtered if search_upper in r['Ticker'].upper()]

    if rec_filter == "Signals Only":
        filtered = [r for r in filtered if r['Rec'] != 'SKIP']
    elif rec_filter == "BUY+":
        # Match ALL buy-type recommendations (both old and new names)
        filtered = [r for r in filtered
                    if any(kw in r['Rec'].upper() for kw in
                           ['STRONG BUY', 'BUY', 'RE-ENTRY', 'LATE ENTRY', 'FRESH', 'AO'])
                    and 'SKIP' not in r['Rec'] and 'WATCH' not in r['Rec']
                    and 'WAIT' not in r['Rec']]
    elif rec_filter == "STRONG BUY":
        filtered = [r for r in filtered if r['Rec'] == 'STRONG BUY']
    elif rec_filter == "Quality A-B":
        filtered = [r for r in filtered if r['Quality'] in ('A', 'B')]
    elif rec_filter == "Open Positions":
        filtered = [r for r in filtered if 'Open' in r.get('Status', '')]
    elif rec_filter == "⚡ Earnings Soon":
        filtered = [r for r in filtered if r.get('Earn', '')]
    elif rec_filter == "🟢 Focus":
        filtered = [r for r in filtered if focus_labels.get(r['Ticker']) == 'green']
    elif rec_filter == "🟡 Focus":
        filtered = [r for r in filtered if focus_labels.get(r['Ticker']) == 'yellow']
    elif rec_filter == "🔴 Focus":
        filtered = [r for r in filtered if focus_labels.get(r['Ticker']) == 'red']
    elif rec_filter == "🔵 Focus":
        filtered = [r for r in filtered if focus_labels.get(r['Ticker']) == 'blue']
    elif rec_filter == "Any Focus":
        filtered = [r for r in filtered if focus_labels.get(r['Ticker'], '') != '']

    # Apply sector phase filter
    _sector_phase_map = {
        "🟢 Leading": "LEADING", "🔵 Emerging": "EMERGING",
        "🟡 Fading": "FADING", "🔴 Lagging": "LAGGING",
    }
    if sector_filter != "All Sectors":
        target_phase = _sector_phase_map.get(sector_filter, '')
        if target_phase:
            filtered = [r for r in filtered if r.get('SectorPhase', '') == target_phase]

    # Always sort favorites to top first
    fav_tickers = set(jm.get_favorite_tickers())

    # Signal strength hierarchy for sorting (handles both old and new names)
    _rec_rank = {
        'STRONG BUY': 10,
        'BUY': 8,
        'BUY (AO)': 7, 'BUY (AO CONFIRM)': 7, 'BUY (CAUTION)': 7,
        'RE-ENTRY': 6, 'RE-ENTRY (CAUTIOUS)': 5,
        'WATCH (AO)': 4, 'WATCH (AO CONFIRM)': 4,
        'WATCH': 3, 'WATCH (RE-ENTRY)': 3, 'WATCH (LATE)': 3,
        'WAIT': 2, 'WAIT (D)': 2,
        'SKIP': 0,
    }

    # Apply sort
    if sort_by == "Signal Strength ↓":
        filtered.sort(key=lambda r: (
            0 if r['Ticker'] in fav_tickers else 1,
            -_rec_rank.get(r['Rec'].split(' (+')[0], 5 if 'LATE ENTRY' in r['Rec'] else 0),
            -(int(r['Conv'].split('/')[0]) if '/' in r['Conv'] else 0),
        ))
    elif sort_by == "Conviction ↓":
        filtered.sort(key=lambda r: (
            0 if r['Ticker'] in fav_tickers else 1,
            -(int(r['Conv'].split('/')[0]) if '/' in r['Conv'] else 0),
        ))
    elif sort_by == "Name A-Z":
        filtered.sort(key=lambda r: (0 if r['Ticker'] in fav_tickers else 1, r['Ticker']))
    elif sort_by == "Name Z-A":
        # Use negative ord values to reverse alpha while keeping favorites first
        filtered.sort(key=lambda r: (
            0 if r['Ticker'] in fav_tickers else 1,
            [-ord(c) for c in r['Ticker']],
        ))
    elif sort_by == "Quality ↓":
        q_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1, '?': 0}
        filtered.sort(key=lambda r: (
            0 if r['Ticker'] in fav_tickers else 1,
            -q_order.get(r.get('Quality', '?'), 0),
        ))
    elif sort_by == "Price ↓":
        filtered.sort(key=lambda r: -float(r['Price'].replace('$', '').replace(',', '') or '0'))
    elif sort_by == "Price ↑":
        filtered.sort(key=lambda r: float(r['Price'].replace('$', '').replace(',', '') or '0'))
    else:
        # Default: favorites first, then by conviction
        filtered.sort(key=lambda r: (
            0 if r['Ticker'] in fav_tickers else 1,
            -(int(r['Conv'].split('/')[0]) if '/' in r['Conv'] else 0),
        ))

    showing = len(filtered)
    if showing != len(rows):
        st.caption(f"Showing {showing} of {len(rows)}")

    # ── Results as clickable ticker buttons ────────────────────────────
    if not filtered:
        st.info("No tickers match the current filter.")
        return

    # ── Pagination — render 25 rows at a time to keep UI fast ─────────
    PAGE_SIZE = 25
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)

    # Track page in session state
    if 'scanner_page' not in st.session_state:
        st.session_state['scanner_page'] = 0
    current_page = st.session_state['scanner_page']
    # Clamp to valid range
    if current_page >= total_pages:
        current_page = total_pages - 1
        st.session_state['scanner_page'] = current_page

    start_idx = current_page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(filtered))
    page_rows = filtered[start_idx:end_idx]

    # Pagination controls (top)
    if total_pages > 1:
        pg_col1, pg_col2, pg_col3, pg_col4, pg_col5 = st.columns([1, 1, 3, 1, 1])
        with pg_col1:
            if st.button("⏮", key="page_first", disabled=current_page == 0):
                st.session_state['scanner_page'] = 0
                st.rerun()
        with pg_col2:
            if st.button("◀", key="page_prev", disabled=current_page == 0):
                st.session_state['scanner_page'] = current_page - 1
                st.rerun()
        with pg_col3:
            st.caption(f"Page {current_page + 1} of {total_pages}  ·  Rows {start_idx + 1}–{end_idx} of {len(filtered)}")
        with pg_col4:
            if st.button("▶", key="page_next", disabled=current_page >= total_pages - 1):
                st.session_state['scanner_page'] = current_page + 1
                st.rerun()
        with pg_col5:
            if st.button("⏭", key="page_last", disabled=current_page >= total_pages - 1):
                st.session_state['scanner_page'] = total_pages - 1
                st.rerun()

    # Table header — added Focus, Earnings Date, Volume, Apex columns
    hdr_cols = st.columns([0.9, 0.3, 0.4, 0.9, 0.4, 0.7, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.7, 0.5, 1.8])
    headers = ['Ticker', '📈', '🏷️', 'Rec', 'Conv', 'Sector', '🎯', 'MACD', 'AO', 'Wkly', 'Mthly', 'Qlty', 'Price', 'Vol', 'Earn', 'Summary']
    for col, h in zip(hdr_cols, headers):
        col.markdown(f"**{h}**")

    # Focus label icons
    _focus_icons = {
        'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'blue': '🔵', '': '⚪'
    }
    _focus_cycle = ['', 'green', 'yellow', 'red', 'blue']  # Click to cycle

    # Table rows — ONLY render current page (25 rows max = fast)
    for idx, row in enumerate(page_rows):
        # Use global index for unique keys
        global_idx = start_idx + idx
        cols = st.columns([0.9, 0.3, 0.4, 0.9, 0.4, 0.7, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.7, 0.5, 1.8])

        # Ticker as clickable button with earnings flag
        with cols[0]:
            ticker_label = row['Ticker']
            status = row.get('Status', '')
            earn = row.get('Earn', '')
            is_fav = row['Ticker'] in fav_tickers
            if is_fav:
                ticker_label = f"⭐ {ticker_label}"
            elif 'Open' in status:
                ticker_label = f"📈 {ticker_label}"
            elif 'Alert' in status:
                ticker_label = f"🎯 {ticker_label}"
            if earn:
                ticker_label = f"{ticker_label} ⚡"

            if st.button(ticker_label, key=f"row_{row['Ticker']}_{global_idx}",
                        use_container_width=True):
                st.session_state['default_detail_tab'] = 0  # Signal tab
                st.session_state['scroll_to_detail'] = True
                _load_ticker_for_view(row['Ticker'])

        # Chart button — opens directly to chart tab
        with cols[1]:
            if st.button("📈", key=f"chart_row_{row['Ticker']}_{global_idx}"):
                st.session_state['default_detail_tab'] = 1  # Chart tab
                st.session_state['scroll_to_detail'] = True
                _load_ticker_for_view(row['Ticker'])

        # Focus label — click to cycle through colors
        with cols[2]:
            curr_label = focus_labels.get(row['Ticker'], '')
            curr_icon = _focus_icons.get(curr_label, '⚪')
            if st.button(curr_icon, key=f"focus_{row['Ticker']}_{global_idx}",
                        help="Click to cycle: ⚪→🟢→🟡→🔴→🔵"):
                curr_idx = _focus_cycle.index(curr_label) if curr_label in _focus_cycle else 0
                next_label = _focus_cycle[(curr_idx + 1) % len(_focus_cycle)]
                jm.set_focus_label(row['Ticker'], next_label)
                st.rerun()

        # Recommendation with color + divergence flag + re-entry recency
        rec_val = row.get('Rec', 'SKIP')
        div_flag = row.get('DivFlag', '')
        rec_colors = {
            'STRONG BUY': '🟢', 'BUY': '🟢', 'BUY (CAUTION)': '🟢',
            'BUY (AO)': '🔵', 'BUY (AO CONFIRM)': '🔵',
            'RE-ENTRY': '🔵', 'RE-ENTRY (CAUTIOUS)': '🔵',
            'WATCH (AO)': '🟡', 'WATCH (AO CONFIRM)': '🟡',
            'WATCH': '🟡', 'WATCH (RE-ENTRY)': '🟡', 'WATCH (LATE)': '🟡',
            'WAIT': '🟡', 'WAIT (D)': '🟠', 'SKIP': '⚪',
        }
        rec_icon = rec_colors.get(rec_val.split(' (+')[0], '⚪')
        if 'LATE ENTRY' in rec_val:
            rec_icon = '🕐'

        # RE-ENTRY / LATE ENTRY recency color coding
        reentry_age = row.get('ReentryAge', 0)
        age_tag = ""
        if ('RE-ENTRY' in rec_val or 'LATE ENTRY' in rec_val):
            if reentry_age <= 3:
                age_tag = f" 🟢{reentry_age}d"      # Fresh — ideal entry
                rec_icon = '🟢'
            elif reentry_age <= 7:
                age_tag = f" 🟡{reentry_age}d"      # Acceptable — move quickly
                rec_icon = '🟡'
            elif reentry_age > 7:
                age_tag = f" 🔴{reentry_age}d"      # Stale — higher risk
                rec_icon = '🔴'

        cols[3].caption(f"{rec_icon}{rec_val}{div_flag}{age_tag}")
        cols[4].caption(row.get('Conv', '0/10'))
        cols[5].caption(row.get('Sector', ''))

        # Apex buy indicator
        cols[6].caption(row.get('ApexFlag', ''))

        cols[7].caption(row.get('MACD', '❌'))
        cols[8].caption(row.get('AO', '❌'))
        cols[9].caption(row.get('Wkly', '❌'))
        cols[10].caption(row.get('Mthly', '❌'))

        # Quality with color
        q = row.get('Quality', '?')
        q_colors = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🔴', 'F': '🔴'}
        cols[11].caption(f"{q_colors.get(q, '⚪')}{q}")

        cols[12].caption(row.get('Price', '?'))
        cols[13].caption(row.get('Volume', ''))

        # Earnings date with highlight
        cols[14].caption(row.get('EarnDate', ''))
        cols[15].caption(row.get('Summary', '')[:45])

    # ── Bottom pagination ─────────────────────────────────────────────
    if total_pages > 1:
        bpg1, bpg2, bpg3, bpg4, bpg5 = st.columns([1, 1, 3, 1, 1])
        with bpg1:
            if st.button("⏮", key="bpage_first", disabled=current_page == 0):
                st.session_state['scanner_page'] = 0
                st.rerun()
        with bpg2:
            if st.button("◀", key="bpage_prev", disabled=current_page == 0):
                st.session_state['scanner_page'] = current_page - 1
                st.rerun()
        with bpg3:
            st.caption(f"Page {current_page + 1}/{total_pages}")
        with bpg4:
            if st.button("▶", key="bpage_next", disabled=current_page >= total_pages - 1):
                st.session_state['scanner_page'] = current_page + 1
                st.rerun()
        with bpg5:
            if st.button("⏭", key="bpage_last", disabled=current_page >= total_pages - 1):
                st.session_state['scanner_page'] = total_pages - 1
                st.rerun()

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
    sector_rotation = st.session_state.get('sector_rotation', {})
    ticker_sectors = st.session_state.get('ticker_sectors', {})
    earnings_flags = st.session_state.get('earnings_flags', {})

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

        # Sector rotation — use phase for color (LEADING/EMERGING/FADING/LAGGING)
        sector = ticker_sectors.get(r.ticker, '')
        sector_info = sector_rotation.get(sector, {})
        sector_phase = sector_info.get('phase', '')
        sector_short = sector_info.get('short_name', sector[:4] if sector else '')
        if sector_phase == 'LEADING':
            sector_dot = f"🟢 {sector_short}"
        elif sector_phase == 'EMERGING':
            sector_dot = f"🔵 {sector_short}"
        elif sector_phase == 'FADING':
            sector_dot = f"🟡 {sector_short}"
        elif sector_phase == 'LAGGING':
            sector_dot = f"🔴 {sector_short}"
        elif sector_short:
            sector_dot = f"⚪ {sector_short}"
        else:
            sector_dot = ""

        # Earnings flag
        earn = earnings_flags.get(r.ticker)
        earn_flag = f"⚡{earn['days_until']}d" if earn else ""

        # Volume
        vol = r.volume or 0
        avg_vol = r.avg_volume_50d or 0
        vol_ratio = r.volume_ratio or 0
        if vol >= 1_000_000:
            vol_str = f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol/1_000:.0f}K"
        else:
            vol_str = str(int(vol)) if vol else ""

        # Volume ratio indicator
        if vol_ratio >= 2.0:
            vol_str = f"🔥{vol_str}"  # 2x+ above average
        elif vol_ratio >= 1.5:
            vol_str = f"📈{vol_str}"  # 1.5x+ above average

        # Earnings date
        earn_date_str = ""
        if earn:
            days = earn['days_until']
            earn_date_str = earn.get('next_earnings', '')
            if days <= 7:
                earn_date_str = f"⚡{earn_date_str}"
            elif days <= 14:
                earn_date_str = f"⏰{earn_date_str}"

        # AO divergence + Apex indicators
        div_flag = " (D)" if r.ao_divergence_active else ""
        apex_flag = "🎯" if r.apex_buy else ""

        # Re-entry recency (for color coding)
        reentry_bars_ago = 0
        if r.reentry and r.reentry.get('is_valid'):
            reentry_bars_ago = r.reentry.get('macd_cross_bars_ago', 0)
        elif r.late_entry and r.late_entry.get('is_valid'):
            reentry_bars_ago = r.late_entry.get('days_since_cross', 0)

        rows.append({
            'Ticker': r.ticker,
            'Status': status,
            'Sector': sector_dot,
            'SectorPhase': sector_phase,  # For filtering
            'Earn': earn_flag,
            'EarnDate': earn_date_str,
            'Rec': rec.get('recommendation', 'SKIP'),
            'Conv': f"{rec.get('conviction', 0)}/10",
            'MACD': "✅" if sig and sig.macd.get('bullish') else "❌",
            'AO': "✅" if sig and sig.ao.get('positive') else "❌",
            'Wkly': "✅" if sig and sig.weekly_macd.get('bullish') else "❌",
            'Mthly': "✅" if sig and sig.monthly_macd.get('bullish') else "❌",
            'Quality': q.get('quality_grade', '?'),
            'Price': f"${r.current_price:.2f}" if r.current_price else "?",
            'Volume': vol_str,
            'DivFlag': div_flag,
            'ApexFlag': apex_flag,
            'ReentryAge': reentry_bars_ago,
            'Summary': rec.get('summary', ''),
        })
    return rows


def _build_rows_from_summary(summary, jm) -> list:
    """Build table rows from persisted scan summary (cross-session)."""
    open_tickers = jm.get_open_tickers()
    conditional_tickers = [c['ticker'] for c in jm.get_pending_conditionals()]
    sector_rotation = st.session_state.get('sector_rotation', {})
    earnings_flags = st.session_state.get('earnings_flags', {})

    rows = []
    for s in summary:
        ticker = s.get('ticker', '?')
        if ticker in open_tickers:
            status = "📈 Open"
        elif ticker in conditional_tickers:
            status = "🎯 Alert"
        else:
            status = "👀"

        # Sector rotation — prefer persisted phase, fallback to runtime lookup
        sector = s.get('sector', '')
        sector_phase = s.get('sector_phase', '')  # Persisted from scan
        if not sector_phase:
            # Fallback to runtime sector_rotation (may be empty after page refresh)
            sector_info = sector_rotation.get(sector, {})
            sector_phase = sector_info.get('phase', '')
        sector_info = sector_rotation.get(sector, {})
        sector_short = sector_info.get('short_name', sector[:4] if sector else '')
        if sector_phase == 'LEADING':
            sector_dot = f"🟢 {sector_short}"
        elif sector_phase == 'EMERGING':
            sector_dot = f"🔵 {sector_short}"
        elif sector_phase == 'FADING':
            sector_dot = f"🟡 {sector_short}"
        elif sector_phase == 'LAGGING':
            sector_dot = f"🔴 {sector_short}"
        elif sector_short:
            sector_dot = f"⚪ {sector_short}"
        else:
            sector_dot = ""

        # Earnings — prefer persisted data from summary, fallback to session state
        earn = earnings_flags.get(ticker)
        earn_date = s.get('earn_date', '')
        earn_days = s.get('earn_days', 999)

        # If persisted data exists, use it; otherwise try session state
        if earn_date:
            earn_flag = f"⚡{earn_days}d" if earn_days <= 14 else ""
            earn_date_str = earn_date
            if earn_days <= 7:
                earn_date_str = f"⚡{earn_date}"
            elif earn_days <= 14:
                earn_date_str = f"⏰{earn_date}"
        elif earn:
            earn_flag = f"⚡{earn['days_until']}d"
            earn_date_str = earn.get('next_earnings', '')
            if earn['days_until'] <= 7:
                earn_date_str = f"⚡{earn_date_str}"
            elif earn['days_until'] <= 14:
                earn_date_str = f"⏰{earn_date_str}"
        else:
            earn_flag = ""
            earn_date_str = ""

        # Volume from persisted data
        vol_str = s.get('volume_str', '')
        div_flag = " (D)" if s.get('ao_divergence_active') else ""
        apex_flag = "🎯" if s.get('apex_buy') else ""
        reentry_bars_ago = s.get('reentry_bars_ago', 0)

        rows.append({
            'Ticker': ticker,
            'Status': status,
            'Sector': sector_dot,
            'SectorPhase': sector_phase,
            'Earn': earn_flag,
            'EarnDate': earn_date_str,
            'Rec': s.get('recommendation', 'SKIP'),
            'Conv': f"{s.get('conviction', 0)}/10",
            'MACD': "✅" if s.get('macd_bullish') else "❌",
            'AO': "✅" if s.get('ao_positive') else "❌",
            'Wkly': "✅" if s.get('weekly_bullish') else "❌",
            'Mthly': "✅" if s.get('monthly_bullish') else "❌",
            'Quality': s.get('quality_grade', '?'),
            'Price': f"${s.get('price', 0):.2f}" if s.get('price') else "?",
            'Volume': vol_str,
            'DivFlag': div_flag,
            'ApexFlag': apex_flag,
            'ReentryAge': reentry_bars_ago,
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

    # Auto-scroll anchor — when a ticker is clicked, scroll here
    st.markdown('<div id="detail-anchor"></div>', unsafe_allow_html=True)

    # Auto-scroll JavaScript — fires once when a new ticker is selected
    if st.session_state.pop('scroll_to_detail', False):
        import streamlit.components.v1 as components
        components.html(
            """<script>
            const el = window.parent.document.getElementById('detail-anchor');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            </script>""",
            height=0,
        )

    # Header with scroll-to-top button
    hdr_col1, hdr_col2 = st.columns([8, 1])
    with hdr_col1:
        st.header(f"{ticker} — {rec.get('recommendation', 'SKIP')}")
    with hdr_col2:
        if st.button("⬆️ Top", key="scroll_top", help="Scroll to top"):
            st.session_state['_do_scroll_top'] = True
            st.rerun()

    # Scroll-to-top JS — fires on next render after button click
    if st.session_state.pop('_do_scroll_top', False):
        import streamlit.components.v1 as components
        components.html(
            """<script>
            setTimeout(function() {
                var doc = window.parent.document;
                // Try iframe parent scroll
                doc.querySelectorAll('[data-testid="stAppViewContainer"], section.main, .main, [data-testid="stMain"]').forEach(function(el) {
                    el.scrollTop = 0;
                });
                window.parent.scrollTo(0, 0);
                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;
            }, 100);
            </script>""",
            height=0,
        )

    st.caption(rec.get('summary', ''))

    # ── Tabs (with optional default tab from chart-first navigation) ──
    tab_names = ["📊 Signal", "📈 Chart", "🤖 AI Intel", "💬 Ask AI", "💼 Trade"]
    default_tab = st.session_state.pop('default_detail_tab', 0)

    # Streamlit tabs don't support programmatic selection directly,
    # so we reorder tabs to put the desired one first, then reorder back
    # Actually, we can't reorder. We use a workaround with session state key.
    tab_signal, tab_chart, tab_ai, tab_chat, tab_trade = st.tabs(tab_names)

    # ── Signal Tab ────────────────────────────────────────────────────
    with tab_signal:
        _render_signal_tab(signal, analysis)

    # ── Chart Tab ─────────────────────────────────────────────────────
    with tab_chart:
        _render_chart_tab(ticker, signal)

    # ── AI Intel Tab ──────────────────────────────────────────────────
    with tab_ai:
        _render_ai_tab(ticker, signal, rec, analysis)

    # ── Ask AI Chat Tab ───────────────────────────────────────────────
    with tab_chat:
        _render_chat_tab(ticker, signal, rec, analysis)

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

    # ── APEX Signal Detection (cached per ticker) ──────────────────
    show_apex = st.checkbox("🎯 Show APEX Signals", value=True, key=f"apex_{ticker}")

    apex_markers = []
    apex_signals_list = []
    apex_summary = {}

    _apex_cache_key = f'_apex_cache_{ticker}'

    if show_apex and weekly is not None and monthly is not None:
        # Use cached APEX results if available
        _cached_apex = st.session_state.get(_apex_cache_key)
        if _cached_apex:
            apex_signals_list = _cached_apex['signals']
            apex_markers = _cached_apex['markers']
            apex_summary = _cached_apex['summary']
        else:
            try:
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

                # Cache for subsequent reruns (avoids re-detection on tab switches)
                st.session_state[_apex_cache_key] = {
                    'signals': apex_signals_list,
                    'markers': apex_markers,
                    'summary': apex_summary,
                }

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

    # Check for AI providers: auto-detect Groq vs xAI/Grok from key prefix
    gemini = None
    openai_client = None
    ai_config = {'model': '', 'fallback_model': '', 'provider': 'none'}

    # Primary provider — auto-detect from GROQ_API_KEY (supports Groq gsk_ and xAI xai-)
    primary_error = None
    try:
        raw_key = st.secrets.get("GROQ_API_KEY", "")
        ai_config = _detect_ai_provider(raw_key)
        api_key = ai_config['key']

        if api_key and ai_config['provider'] != 'none':
            # Check cached validation
            cached_status = st.session_state.get('_groq_key_status')
            cached_key = st.session_state.get('_groq_key_cached', '')
            if cached_status == 'invalid' and cached_key == api_key:
                primary_error = f"Key previously failed (401). Click 🔑 Reset API after updating. [{ai_config['display']}]"
            else:
                from openai import OpenAI
                openai_client = OpenAI(
                    api_key=api_key,
                    base_url=ai_config['base_url'],
                )
                # Store config in session for other tabs
                st.session_state['_ai_config'] = ai_config
                # Pre-flight validation (once per key)
                if f'_ai_validated_{api_key[:8]}' not in st.session_state:
                    try:
                        _test = openai_client.chat.completions.create(
                            model=ai_config['model'],
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=1,
                        )
                        st.session_state[f'_ai_validated_{api_key[:8]}'] = True
                    except Exception as val_err:
                        val_str = str(val_err)
                        if 'Invalid API Key' in val_str or '401' in val_str or 'Unauthorized' in val_str:
                            st.session_state['_groq_key_status'] = 'invalid'
                            st.session_state['_groq_key_cached'] = api_key
                            openai_client = None
                            primary_error = f"Key validation failed (401). {ai_config['display']}"
                        else:
                            # Non-auth error (rate limit etc) — key itself probably fine
                            st.session_state[f'_ai_validated_{api_key[:8]}'] = True
        else:
            primary_error = "No GROQ_API_KEY in secrets (supports Groq gsk_ or xAI xai- keys)"
    except ImportError:
        primary_error = "openai package not installed — add to requirements.txt"
    except Exception as e:
        primary_error = str(e)[:200]

    # Show provider info
    if openai_client:
        st.caption(f"Provider: {ai_config['display']} | Model: {ai_config['model']}")

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
        if primary_error:
            errors.append(primary_error)
        if gemini_error:
            errors.append(f"Gemini: {gemini_error}")
        st.warning(f"⚠️ AI providers unavailable: {' | '.join(errors)}")
        st.caption("💡 After updating API keys, click 🔑 **Reset API** in ⚙️ Settings sidebar.")

    # Auto-run on first view for this ticker, or manual re-run
    has_cached = st.session_state.get(f'ai_result_{ticker}') is not None
    should_run = False
    keys_available = bool(openai_client or gemini)

    if has_cached:
        # Show refresh button only if already have results
        if st.button("🔄 Re-run AI Analysis", type="secondary"):
            should_run = True
    elif keys_available:
        # Show prominent run button — don't auto-fire (saves ~10 API calls per ticker switch)
        st.info("🤖 **AI analysis ready.** Click below to fetch fundamentals, market intel & AI recommendation.")
        if st.button("▶️ Run AI Analysis", type="primary", use_container_width=True):
            should_run = True
    else:
        # No providers + no cached results — show manual button (to fetch data at least)
        st.caption("AI providers unavailable. Data-only analysis available.")
        if st.button("📊 Fetch Fundamentals (data only)", type="secondary"):
            should_run = True

    if should_run:
        with st.spinner("Fetching fundamentals, market intel & analyzing..."):
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
                finnhub_key = ""
                try:
                    finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
                except Exception:
                    pass
                if finnhub_key:
                    news_data = fetch_finnhub_news(ticker, api_key=finnhub_key)
            except Exception:
                pass

            # Market intelligence — ALWAYS fetch (analysts, insiders, social proxy)
            market_intel = {}
            try:
                from data_fetcher import fetch_market_intelligence
                # Safe secrets access — st.secrets.get() throws if no secrets.toml
                finnhub_key = ""
                try:
                    finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
                except Exception:
                    pass
                market_intel = fetch_market_intelligence(ticker, finnhub_key=finnhub_key)
            except Exception as e:
                st.caption(f"Market intel error: {e}")

            # AI analysis — only if provider available
            if openai_client or gemini:
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
                    ai_model=ai_config.get('model', 'llama-3.3-70b-versatile'),
                )
            else:
                # No AI provider — build result from data only
                result = {
                    'provider': 'none',
                    'note': 'No AI provider configured. Showing data only.',
                    'conviction': 0,
                    'action': rec.get('recommendation', 'N/A'),
                    'position_sizing': 'N/A',
                    'fundamental_quality': 'N/A',
                    'analysis': 'Configure GROQ_API_KEY or GEMINI_API_KEY for AI analysis.',
                }

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
    groq_err = ai_result.get('groq_error', '')
    gemini_err = ai_result.get('gemini_error', '')
    _cfg = st.session_state.get('_ai_config', {})
    _provider_name = _cfg.get('display', 'AI provider')

    if groq_err:
        if 'Invalid API Key' in groq_err or '401' in groq_err or 'Unauthorized' in groq_err:
            st.error(f"🔑 **{_provider_name}: API key invalid (401).** Click 🔑 Reset API in ⚙️ Settings sidebar after fixing.")
        else:
            st.warning(f"⚠️ {_provider_name} error: {groq_err}")
    if gemini_err:
        if '429' in gemini_err or 'quota' in gemini_err.lower():
            st.warning("⚠️ Gemini fallback: quota exceeded. Fix your primary API key to avoid this.")
        else:
            st.warning(f"⚠️ Gemini error: {gemini_err}")
    if ai_result.get('openai_error'):
        st.warning(f"⚠️ API error: {ai_result['openai_error']}")
    if ai_result.get('error'):
        if 'All AI providers failed' in str(ai_result.get('error', '')):
            st.error(f"❌ All AI providers failed. Check your API key in secrets, then click 🔑 Reset API.")
        else:
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
    # TRADE LEVELS (Entry / Target / Stop / R:R)
    # ═══════════════════════════════════════════════════════════════
    stops = signal.stops if signal else {}
    _entry = stops.get('entry', 0)
    _stop = stops.get('stop', 0)
    _target = stops.get('target', 0)
    _price = analysis.current_price if analysis else 0

    if _entry and _stop and _target and _entry > _stop:
        risk = _entry - _stop
        reward = _target - _entry
        rr_ratio = reward / risk if risk > 0 else 0

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Entry", f"${_entry:.2f}")
        with t2:
            pct_target = ((_target - _entry) / _entry * 100) if _entry else 0
            st.metric("Target", f"${_target:.2f}", delta=f"+{pct_target:.1f}%")
        with t3:
            pct_stop = ((_stop - _entry) / _entry * 100) if _entry else 0
            st.metric("Stop Loss", f"${_stop:.2f}", delta=f"{pct_stop:.1f}%")
        with t4:
            rr_icon = "🟢" if rr_ratio >= 2.5 else ("🟡" if rr_ratio >= 2.0 else "🔴")
            st.metric("R:R Ratio", f"{rr_icon} {rr_ratio:.1f}:1")

    # ═══════════════════════════════════════════════════════════════
    # SECTOR ROTATION POSITION
    # ═══════════════════════════════════════════════════════════════
    _rotation = st.session_state.get('sector_rotation', {})
    _fp = ai_result.get('fundamental_profile', {})
    _stock_sector = _fp.get('sector', '') if _fp else ''
    _phase = ''  # Initialize — may be set below
    _vs_spy = 0

    if _rotation and _stock_sector:
        _sector_info = _rotation.get(_stock_sector, {})
        _phase = _sector_info.get('phase', '')
        _vs_spy = _sector_info.get('vs_spy_20d', 0)
        _etf = _sector_info.get('etf', '')
        _perf_20d = _sector_info.get('perf_20d', 0)

        if _phase == 'LEADING':
            st.success(f"🚀 **MOMENTUM TAILWIND** — {_stock_sector} ({_etf}) is **LEADING** the market ({_vs_spy:+.1f}% vs SPY, {_perf_20d:+.1f}% 20d)")
            st.caption("**Trading Guidance:** 2 of 3 timeframes aligned = valid setup. Sector momentum supports the trade.")
        elif _phase == 'EMERGING':
            st.info(f"📈 **REQUIRES STRONGER CONFLUENCE** — {_stock_sector} ({_etf}) is **EMERGING** ({_vs_spy:+.1f}% vs SPY, {_perf_20d:+.1f}% 20d)")
            st.caption("**Trading Guidance:** All 3 timeframes must align. Sector is building momentum but not yet confirmed.")
        elif _phase == 'FADING':
            st.warning(f"⚠️ **TIGHTEN STOPS** — {_stock_sector} ({_etf}) is **FADING** ({_vs_spy:+.1f}% vs SPY, {_perf_20d:+.1f}% 20d)")
            st.caption("**Trading Guidance:** Consider taking profits, reduce position size. Watch for sector breakdown.")
        elif _phase == 'LAGGING':
            st.error(f"🔴 **SECTOR HEADWIND** — {_stock_sector} ({_etf}) is **LAGGING** ({_vs_spy:+.1f}% vs SPY, {_perf_20d:+.1f}% 20d)")
            st.caption("**Trading Guidance:** Perfect confluence + volume surge required. Sector is working against you.")

    # ═══════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME CONFLUENCE
    # ═══════════════════════════════════════════════════════════════
    _macd = signal.macd if signal else {}
    _ao = signal.ao if signal else {}
    _weinstein = signal.weinstein if signal else {}
    _ws_stage = _weinstein.get('stage', 0) if _weinstein else 0

    # Daily: MACD above signal line = bullish
    _daily_bullish = _macd.get('bullish', False) if _macd else False
    _daily_weakening = _macd.get('weakening', False) if _macd else False
    _daily_cross_recent = _macd.get('cross_recent', False) if _macd else False
    _daily_hist = _macd.get('histogram', 0) if _macd else 0

    # Weekly: Weinstein Stage 2 (advancing) = bullish structure
    _weekly_bullish = _ws_stage == 2

    # Momentum: AO positive = bullish momentum (trend adds nuance)
    _ao_positive = _ao.get('positive', False) if _ao else False
    _ao_trend = _ao.get('trend', 'flat') if _ao else 'flat'
    _ao_value = _ao.get('value', 0) if _ao else 0
    _momentum_bullish = _ao_positive

    _aligned_count = sum([_daily_bullish, _weekly_bullish, _momentum_bullish])
    _confluence_score = _aligned_count / 3.0

    st.markdown("---")
    tf1, tf2, tf3 = st.columns(3)
    with tf1:
        if _daily_bullish:
            _d_icon = "✅ Bullish"
            if _daily_weakening:
                _d_icon = "⚠️ Weakening"
        else:
            _d_icon = "❌ Bearish"
        _d_detail = f"Hist: {_daily_hist:+.2f}" + (" | Recent cross" if _daily_cross_recent else "")
        st.metric("Daily (MACD)", _d_icon)
        st.caption(_d_detail)
    with tf2:
        _w_icon = "✅ Stage 2" if _weekly_bullish else ("⚠️ Stage " + str(_ws_stage) if _ws_stage else "❌ N/A")
        _w_label = _weinstein.get('label', '')[:25] if _weinstein else ''
        st.metric("Weekly (Weinstein)", _w_icon)
        st.caption(_w_label if _w_label else "Structure")
    with tf3:
        if _ao_positive:
            _m_icon = "✅ Positive" if _ao_trend != 'falling' else "⚠️ Fading"
        else:
            _m_icon = "❌ Negative"
        st.metric("Momentum (AO)", _m_icon)
        st.caption(f"AO: {_ao_value:+.1f} ({_ao_trend})")

    # Confluence bar + sector-adjusted guidance
    _required = 3 if _phase == 'LAGGING' else (3 if _phase == 'EMERGING' else 2)
    _meets_req = _aligned_count >= _required
    _bar_text = f"Confluence: {_aligned_count}/3 aligned"
    if _phase:
        _bar_text += f" ({'✅ meets' if _meets_req else '❌ below'} {_phase.lower()} sector requirement of {_required}/3)"
    st.progress(_confluence_score, text=_bar_text)

    # ═══════════════════════════════════════════════════════════════
    # RISK ASSESSMENT (earnings proximity + volatility)
    # ═══════════════════════════════════════════════════════════════
    resistance = ai_result.get('resistance_verdict', '')  # Used here and in resistance section below

    _earnings_data = ai_result.get('earnings_history', {})
    _next_earnings = _earnings_data.get('next_date') if _earnings_data else None
    _days_to_earn = None
    if _next_earnings:
        try:
            _earn_dt = datetime.strptime(str(_next_earnings), '%Y-%m-%d')
            _days_to_earn = (_earn_dt - datetime.now()).days
        except Exception:
            pass

    _risk_factors = []
    _risk_score = 0

    # Earnings risk
    if _days_to_earn is not None and _days_to_earn <= 30:
        _risk_score += 40 if _days_to_earn <= 7 else (30 if _days_to_earn <= 14 else 20)
        _risk_factors.append(f"Earnings in {_days_to_earn}d")

    # Sector headwind
    if _phase in ('LAGGING', 'FADING'):
        _risk_score += 20
        _risk_factors.append(f"Sector {_phase.lower()}")

    # Low conviction
    if conv <= 3:
        _risk_score += 20
        _risk_factors.append("Low conviction")

    # Resistance overhead
    if resistance and any(w in resistance.lower() for w in ['wait', 'stall', 'failed']):
        _risk_score += 15
        _risk_factors.append("Resistance overhead")

    # Low confluence
    if _aligned_count <= 1:
        _risk_score += 15
        _risk_factors.append(f"Low confluence ({_aligned_count}/3)")

    _risk_score = min(_risk_score, 100)

    if _risk_score > 0:
        r1, r2 = st.columns([3, 1])
        with r1:
            _risk_label = "LOW" if _risk_score < 30 else ("MODERATE" if _risk_score < 50 else ("HIGH" if _risk_score < 75 else "EXTREME"))
            _risk_color = "success" if _risk_score < 30 else ("info" if _risk_score < 50 else ("warning" if _risk_score < 75 else "error"))
            getattr(st, _risk_color)(f"⚠️ **Risk: {_risk_label}** ({_risk_score}/100) — {', '.join(_risk_factors)}")
        with r2:
            st.progress(_risk_score / 100, text=f"{_risk_score}%")

    # ═══════════════════════════════════════════════════════════════
    # TECHNICAL SNAPSHOT TABLE (reuses _macd, _ao, _weinstein from confluence)
    # ═══════════════════════════════════════════════════════════════
    if signal and _price:
        _vol_ratio = analysis.volume_ratio if analysis else 0
        _ores = signal.overhead_resistance if signal else {}

        _tech_rows = []

        # Price
        _tech_rows.append(("Price", f"${_price:.2f}", "—"))

        # Weinstein Stage
        _ts_icon = "✅" if _ws_stage == 2 else ("⚠️" if _ws_stage in (1, 3) else "❌")
        _ts_label = _weinstein.get('label', '')[:30] if _weinstein else ''
        _tech_rows.append(("Weinstein Stage", f"Stage {_ws_stage}" + (f" — {_ts_label}" if _ts_label else ""), _ts_icon))

        # MACD Signal
        _macd_bullish = _macd.get('bullish', False) if _macd else False
        _macd_weak = _macd.get('weakening', False) if _macd else False
        _macd_hist = _macd.get('histogram', 0) if _macd else 0
        if _macd_bullish:
            _ms_label = "Bullish" + (" (weakening)" if _macd_weak else "")
            _ms_icon = "⚠️" if _macd_weak else "✅"
        else:
            _ms_label = "Bearish"
            _ms_icon = "❌"
        _tech_rows.append(("MACD Signal", f"{_ms_label} (Hist: {_macd_hist:+.2f})", _ms_icon))

        # AO Momentum
        _ao_pos = _ao.get('positive', False) if _ao else False
        _ao_trn = _ao.get('trend', 'flat') if _ao else 'flat'
        _ao_val = _ao.get('value', 0) if _ao else 0
        if _ao_pos:
            _ao_label = f"Positive ({_ao_trn})"
            _ao_icon = "✅" if _ao_trn != 'falling' else "⚠️"
        else:
            _ao_label = f"Negative ({_ao_trn})"
            _ao_icon = "❌"
        _tech_rows.append(("AO Momentum", f"{_ao_label} ({_ao_val:+.1f})", _ao_icon))

        # Volume vs Average
        if _vol_ratio:
            _vol_status = "✅ High" if _vol_ratio >= 1.5 else ("⚠️ Normal" if _vol_ratio >= 0.8 else "❌ Low")
            _tech_rows.append(("Volume vs Avg", f"{_vol_ratio:.1f}x", _vol_status))

        # Risk per Share
        if _entry and _stop:
            _risk_pct = abs((_entry - _stop) / _entry * 100) if _entry else 0
            _tech_rows.append(("Risk per Share", f"${abs(_entry - _stop):.2f}", f"{_risk_pct:.1f}%"))

        # Overhead Resistance Density
        if _ores:
            _density = _ores.get('density_pct', 0) if isinstance(_ores, dict) else 0
            _crit = _ores.get('critical_level', {}) if isinstance(_ores, dict) else {}
            _crit_price = _crit.get('price', 0) if isinstance(_crit, dict) else 0
            _ores_val = f"{_density:.0f}%" + (f" (critical: ${_crit_price:.2f})" if _crit_price else "")
            _ores_icon = "✅ Clear" if _density < 15 else ("⚠️ Moderate" if _density < 40 else "❌ Heavy")
            _tech_rows.append(("Overhead Resistance", _ores_val, _ores_icon))

        # Distance to 200 SMA
        _sma200 = _weinstein.get('sma_200', 0) if _weinstein else 0
        if not _sma200:
            _sma200 = _weinstein.get('ma_200', 0) if _weinstein else 0
        if _sma200 and _price:
            _dist_200 = ((_price / _sma200) - 1) * 100
            _dist_icon = "✅ Above" if _dist_200 > 0 else "❌ Below"
            _tech_rows.append(("Distance to 200 SMA", f"{_dist_200:+.1f}%" + (f" (${_sma200:.2f})" if _sma200 else ""), _dist_icon))

        if _tech_rows:
            with st.expander("📈 Technical Snapshot", expanded=False):
                _df = pd.DataFrame(_tech_rows, columns=["Metric", "Value", "Status"])
                st.dataframe(_df, hide_index=True, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # RESISTANCE VERDICT + BREAKOUT ALERT BUTTON
    # ═══════════════════════════════════════════════════════════════
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
            st.info(clean_ai_formatting(why))

        fq_detail = ai_result.get('fundamental_quality', '')
        if fq_detail:
            st.markdown(f"**💼 Fundamental quality:**")
            st.info(clean_ai_formatting(fq_detail))

    with col_r:
        bull = ai_result.get('bull_case', '')
        if bull:
            st.markdown("**🐂 Bull case:**")
            st.success(clean_ai_formatting(bull))

        bear = ai_result.get('bear_case', '')
        if bear:
            st.markdown("**🐻 Bear case:**")
            st.error(clean_ai_formatting(bear))

    # ═══════════════════════════════════════════════════════════════
    # SMART MONEY (AI-synthesized analyst + insider view)
    # ═══════════════════════════════════════════════════════════════
    smart = ai_result.get('smart_money', '')
    if smart:
        st.markdown("**🏦 Smart Money:**")
        st.info(clean_ai_formatting(smart))

    # ═══════════════════════════════════════════════════════════════
    # RED FLAGS
    # ═══════════════════════════════════════════════════════════════
    flags = ai_result.get('red_flags', '')
    if flags and flags.lower() != 'none':
        st.warning(f"🚩 **Red flags:** {clean_ai_formatting(flags)}")
    else:
        st.success("🚩 **Red flags:** None — clean setup")

    # ═══════════════════════════════════════════════════════════════
    # MARKET INTELLIGENCE PANEL
    # ═══════════════════════════════════════════════════════════════
    mi = ai_result.get('market_intel', {})
    if mi:
        _render_market_intelligence(mi)
    else:
        st.caption("Market intelligence not available — re-run analysis to fetch")

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
                url = h.get('url', '')
                headline = h.get('headline', '')
                source = h.get('source', '')
                dt = h.get('datetime', '')
                if url:
                    st.markdown(f"**{dt}** — [{headline}]({url}) *({source})*")
                else:
                    st.markdown(f"**{dt}** — {headline} *({source})*")

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
        st.markdown(clean_ai_formatting(ai_result.get('raw_text', '')))


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
        st.markdown("**👔 Insider Transactions (90d)**")
        buys = intel.get('insider_buys_90d', 0)
        sells = intel.get('insider_sells_90d', 0)

        if buys > 0 or sells > 0:
            net = intel.get('insider_net_shares', 0)
            if net > 0:
                st.success(f"**{buys} buys, {sells} sells — NET BUYING**")
            elif net < 0:
                st.warning(f"**{buys} buys, {sells} sells — NET SELLING**")
            else:
                st.info(f"**{buys} buys, {sells} sells — Neutral**")

            # Show top transactions
            txns = intel.get('insider_transactions', [])
            if txns:
                with st.expander("Transaction Details", expanded=False):
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
            st.caption("No insider transactions found in last 90 days")

    with col_social:
        st.markdown("**📊 Social / Volume Buzz**")
        social = intel.get('social_score')
        reddit = intel.get('social_reddit_mentions')
        twitter = intel.get('social_twitter_mentions')
        social_source = intel.get('social_source', '')
        social_error = intel.get('social_error', '')
        vol_surge = intel.get('volume_surge_ratio')

        if social:
            s_map = {
                'High buzz': ('success', '🔥'), 'Moderate': ('info', '📊'), 'Low': ('warning', '😴'),
                'High volume surge': ('success', '🔥'), 'Elevated volume': ('info', '📈'),
                'Above avg volume': ('info', '📊'), 'Normal volume': ('warning', '😴'),
                'No volume data': ('warning', '📊'), 'Volume check failed': ('warning', '⚠️'),
            }
            method, icon = s_map.get(social, ('info', '📊'))
            getattr(st, method)(f"{icon} **{social}**")

            parts = []
            if reddit is not None and reddit > 0:
                parts.append(f"Reddit: {reddit}")
            if twitter is not None and twitter > 0:
                parts.append(f"Twitter: {twitter}")
            if vol_surge is not None and social_source == 'volume_proxy':
                parts.append(f"Vol ratio: {vol_surge:.1f}x avg")
            if parts:
                source_label = "7-day mentions" if social_source != 'volume_proxy' else "5-day vs 50-day avg"
                st.caption(f"{source_label} — {' | '.join(parts)}")

            # Show source info
            if social_source == 'volume_proxy':
                if social_error == 'Finnhub premium required':
                    st.caption("ℹ️ Finnhub social requires premium plan — using volume as proxy")
                else:
                    st.caption("ℹ️ Using volume surge as social proxy")
            elif social_source == 'unavailable':
                st.caption("Volume data unavailable for this ticker")
        else:
            if social_error:
                st.caption(f"📊 {social_error}")
            else:
                st.caption("📊 Social data not available")


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


# =============================================================================
# ASK AI — Research Analyst + Interactive Chat
# =============================================================================

def _build_sector_rotation_context() -> str:
    """
    Build dynamic sector rotation context from live cached data.
    Pulls from session state (populated by fetch_sector_rotation at startup).
    Returns formatted text for injection into the AI system prompt.
    """
    rotation = st.session_state.get('sector_rotation')
    if not rotation:
        return "SECTOR ROTATION: Data not yet loaded. Ask user to refresh market data."

    # Group sectors by phase
    phases = {'LEADING': [], 'EMERGING': [], 'FADING': [], 'LAGGING': []}
    seen_etfs = set()
    for sector, info in rotation.items():
        etf = info.get('etf', '')
        if etf in seen_etfs:
            continue
        seen_etfs.add(etf)
        phase = info.get('phase', 'LAGGING')
        vs_spy = info.get('vs_spy_20d', 0)
        perf = info.get('perf_20d', 0)
        short_name = info.get('short_name', sector[:4])
        phases[phase].append({
            'name': sector,
            'short': short_name,
            'etf': etf,
            'perf_20d': perf,
            'vs_spy_20d': vs_spy,
        })

    lines = [f"═══ CURRENT SECTOR ROTATION STATUS (Updated: {datetime.now().strftime('%Y-%m-%d')}) ═══\n"]

    def _fmt_sector_list(sectors):
        if not sectors:
            return "  (none currently)"
        return "\n".join(f"  - {s['name']} ({s['etf']}): {s['perf_20d']:+.1f}% (vs SPY: {s['vs_spy_20d']:+.1f}%)"
                         for s in sorted(sectors, key=lambda x: x['vs_spy_20d'], reverse=True))

    lines.append("LEADING Sectors (momentum tailwind — 2 of 3 timeframes aligned = valid setup):")
    lines.append(_fmt_sector_list(phases['LEADING']))
    lines.append("")
    lines.append("EMERGING Sectors (building momentum — all 3 timeframes must align):")
    lines.append(_fmt_sector_list(phases['EMERGING']))
    lines.append("")
    lines.append("FADING Sectors (losing momentum — tighten stops, reduce position size):")
    lines.append(_fmt_sector_list(phases['FADING']))
    lines.append("")
    lines.append("LAGGING Sectors (headwind — perfect confluence + volume breakout required):")
    lines.append(_fmt_sector_list(phases['LAGGING']))
    lines.append("")
    lines.append("""MOMENTUM TRADING GUIDANCE:
- Stocks in LEADING sectors: 2 of 3 timeframes aligned = valid setup. Sector is a tailwind.
- Stocks in EMERGING sectors: All 3 timeframes must align. Sector is neutral-to-positive.
- Stocks in FADING sectors: Tighten stops, reduce sizing. Sector momentum is weakening.
- Stocks in LAGGING sectors: Perfect confluence + volume breakout required. Sector is a headwind.
- ALWAYS state which sector rotation category the ticker belongs to in your analysis.
- Adjust conviction level up/down based on whether sector has momentum tailwinds or headwinds.
- If the stock's actual business doesn't match its assigned sector, note which sector's data is more relevant.""")

    return "\n".join(lines)


def _build_internal_context(ticker: str, signal: EntrySignal, rec: Dict,
                            analysis: TickerAnalysis) -> str:
    """Build internal app data context — signals, quality, AI results.
    This is what the AI can SEE from the app. It should interpret, not repeat."""
    lines = [f"═══ IN-APP DATA FOR {ticker} (visible to user — DO NOT repeat, only interpret) ═══\n"]

    # Price & recommendation
    if analysis.current_price:
        lines.append(f"PRICE: ${analysis.current_price:.2f}")
    lines.append(f"APP RECOMMENDATION: {rec.get('recommendation', 'N/A')} | Conviction: {rec.get('conviction', 0)}/10")

    # Quality
    q = analysis.quality or {}
    if q:
        lines.append(f"QUALITY GRADE: {q.get('quality_grade', '?')}")

    # Signals
    if signal:
        m = signal.macd
        ao = signal.ao
        wm = signal.weekly_macd
        mm = signal.monthly_macd
        lines.append(f"\nTECHNICAL SIGNALS (user can see these on Signal tab):")
        lines.append(f"  Daily MACD: {'Bullish' if m.get('bullish') else 'Bearish'} | Hist: {m.get('histogram', 0):+.4f}"
                     f"{' | NEAR CROSS' if m.get('near_cross') else ''}"
                     f"{' | WEAKENING' if m.get('weakening') else ''}")
        lines.append(f"  AO: {'Positive' if ao.get('positive') else 'Negative'} | {ao.get('value', 0):+.4f}"
                     f"{' | SAUCER' if ao.get('saucer') else ''}")
        lines.append(f"  Weekly MACD: {'Bullish' if wm.get('bullish') else 'Bearish'}")
        lines.append(f"  Monthly MACD: {'Bullish' if mm.get('bullish') else 'Bearish'}")

        if analysis.volume_ratio:
            lines.append(f"  Volume: {analysis.volume_ratio:.1f}x average")

    # Special signals
    if analysis.reentry and analysis.reentry.get('is_valid'):
        re = analysis.reentry
        lines.append(f"  RE-ENTRY: MACD cross {re.get('macd_cross_bars_ago', '?')} bars ago, AO confirm: {re.get('ao_confirmed')}")
    if analysis.apex_buy:
        lines.append(f"  🎯 APEX BUY SIGNAL ACTIVE")
    if analysis.ao_divergence_active:
        lines.append(f"  ⚡ AO BULLISH DIVERGENCE DETECTED")

    # Previous AI analysis
    ai_result = st.session_state.get(f'ai_result_{ticker}')
    if ai_result:
        lines.append(f"\nPREVIOUS AI ANALYSIS (AI Intel tab):")
        for key in ['action', 'conviction', 'resistance_verdict', 'why_moving',
                     'fundamental_quality', 'smart_money', 'bull_case', 'bear_case',
                     'red_flags', 'position_sizing']:
            val = ai_result.get(key)
            if val:
                lines.append(f"  {key.replace('_', ' ').title()}: {val}")

        # Market intel summary
        mi = ai_result.get('market_intel', {})
        if mi:
            ac = mi.get('analyst_consensus')
            if ac:
                lines.append(f"  Analyst Consensus: {ac} ({mi.get('analyst_count', 0)} analysts)")
            target = mi.get('target_mean')
            if target:
                lines.append(f"  Mean Price Target: ${target:.2f} ({mi.get('target_upside_pct', 0):+.1f}%)")
            buys = mi.get('insider_buys_90d', 0)
            sells = mi.get('insider_sells_90d', 0)
            if buys > 0 or sells > 0:
                lines.append(f"  Insider Transactions: {buys} buys, {sells} sells (90d)")
            else:
                lines.append(f"  Insider Transactions: None in 90 days")

    # ═══ SECTOR (from app session — MANDATORY for AI to discuss) ═══
    sector = st.session_state.get('ticker_sectors', {}).get(ticker)
    if sector:
        rotation = st.session_state.get('sector_rotation', {}).get(sector, {})
        if rotation:
            lines.append(f"\n═══ SECTOR CONTEXT (MANDATORY — you MUST discuss this) ═══")
            lines.append(f"  Sector: {sector}")
            lines.append(f"  Sector ETF: {rotation.get('etf', '?')}")
            lines.append(f"  Phase: {rotation.get('phase', '?')}")
            lines.append(f"  Sector vs SPY (20-day): {rotation.get('vs_spy_20d', 0):+.1f}%")
            lines.append(f"  Sector vs SPY (5-day): {rotation.get('vs_spy_5d', 0):+.1f}%")
            lines.append(f"  Sector 5d perf: {rotation.get('perf_5d', 0):+.1f}%")
            lines.append(f"  Sector 20d perf: {rotation.get('perf_20d', 0):+.1f}%")
        else:
            lines.append(f"\n═══ SECTOR: {sector} (no rotation data — check external research) ═══")
    else:
        lines.append(f"\n═══ SECTOR: Unknown (check Yahoo data in external research for sector) ═══")

    # ═══ EARNINGS (from app session — MANDATORY for AI to discuss) ═══
    earn = st.session_state.get('earnings_flags', {}).get(ticker)
    if earn:
        days = earn.get('days_until', 999)
        lines.append(f"\n═══ EARNINGS (MANDATORY — you MUST address this in your analysis) ═══")
        lines.append(f"  Next Earnings Date: {earn.get('next_earnings', '?')}")
        lines.append(f"  Days Until: {days}")
        if days <= 7:
            lines.append(f"  ⚠️ CRITICAL: EARNINGS IN {days} DAYS — EXTREME RISK")
        elif days <= 14:
            lines.append(f"  ⚠️ WARNING: EARNINGS IN {days} DAYS — HIGH RISK, limited trading window")
        elif days <= 30:
            lines.append(f"  ⚠️ CAUTION: EARNINGS WITHIN 30 DAYS — affects hold duration")
        else:
            lines.append(f"  ✅ Clear runway: {days} days before earnings")
    else:
        lines.append(f"\n═══ EARNINGS: Date not available from app — CHECK Yahoo data in external research ═══")

    return "\n".join(lines)


def _fetch_external_research(ticker: str) -> str:
    """Fetch comprehensive external data — this is the AI's UNIQUE VALUE.
    Includes: market conditions, sector rotation, earnings, news, fundamentals, social."""
    lines = [f"\n═══ EXTERNAL RESEARCH FOR {ticker} (freshly fetched) ═══\n"]

    # ══════════════════════════════════════════════════════════════════
    # SECTIONS 1 & 2: MARKET CONDITIONS + SECTOR ROTATION
    # These are TICKER-INDEPENDENT — cache in session state (5-min TTL)
    # ══════════════════════════════════════════════════════════════════
    _mkt_cache = st.session_state.get('_research_market_cache', '')
    _mkt_ts = st.session_state.get('_research_market_cache_ts', 0)
    _now = datetime.now().timestamp()

    if _mkt_cache and (_now - _mkt_ts) < 300:
        # Use cached market context (same for all tickers)
        lines.append(_mkt_cache)
    else:
        # Build fresh market context
        _mkt_lines = []
        _mkt_lines.append("🌍 OVERALL MARKET CONDITIONS:")
        try:
            from data_fetcher import fetch_daily, fetch_market_filter
            market = fetch_market_filter()

            spy_close = market.get('spy_close')
            spy_sma200 = market.get('spy_sma200')
            spy_above = market.get('spy_above_200', True)
            vix_close = market.get('vix_close')
            vix_below = market.get('vix_below_30', True)

            if spy_close:
                _mkt_lines.append(f"  SPY: ${spy_close:.2f} | 200-day SMA: ${spy_sma200:.2f} | {'ABOVE ✅' if spy_above else 'BELOW ❌'}")
            if vix_close:
                _mkt_lines.append(f"  VIX: {vix_close:.1f} | {'LOW fear ✅' if vix_close < 20 else 'ELEVATED ⚠️' if vix_close < 30 else 'HIGH FEAR ❌'}")

            # SPY recent performance (risk-on vs risk-off)
            spy_df = fetch_daily("SPY")
            if spy_df is not None and len(spy_df) >= 50:
                spy_5d = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-5] - 1) * 100
                spy_20d = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-20] - 1) * 100
                spy_50d = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-50] - 1) * 100
                spy_sma50 = float(spy_df['Close'].rolling(50).mean().iloc[-1])
                spy_high52 = float(spy_df['Close'].tail(252).max()) if len(spy_df) >= 252 else float(spy_df['Close'].max())
                pct_from_high = (spy_df['Close'].iloc[-1] / spy_high52 - 1) * 100

                _mkt_lines.append(f"  SPY Returns: 5d {spy_5d:+.1f}% | 20d {spy_20d:+.1f}% | 50d {spy_50d:+.1f}%")
                _mkt_lines.append(f"  SPY 50-day SMA: ${spy_sma50:.2f} | {'Above' if spy_df['Close'].iloc[-1] > spy_sma50 else 'Below'}")
                _mkt_lines.append(f"  SPY vs 52-week high: {pct_from_high:+.1f}%")

                # Overall market assessment
                if spy_above and vix_close and vix_close < 20 and spy_5d > 0:
                    _mkt_lines.append(f"  ASSESSMENT: RISK-ON environment — market bullish, low fear, new positions supported")
                elif spy_above and vix_close and vix_close < 25:
                    _mkt_lines.append(f"  ASSESSMENT: CAUTIOUSLY BULLISH — market above key support, moderate fear")
                elif not spy_above:
                    _mkt_lines.append(f"  ASSESSMENT: RISK-OFF — SPY below 200-day SMA, defensive posture recommended")
                elif vix_close and vix_close >= 30:
                    _mkt_lines.append(f"  ASSESSMENT: HIGH VOLATILITY — elevated VIX, reduce position sizes")
                else:
                    _mkt_lines.append(f"  ASSESSMENT: NEUTRAL — mixed signals, selective stock-picking environment")

                # Breadth proxy: RSP (equal-weight SPY) vs SPY
                try:
                    rsp_df = fetch_daily("RSP")
                    if rsp_df is not None and len(rsp_df) >= 20:
                        rsp_20d = (rsp_df['Close'].iloc[-1] / rsp_df['Close'].iloc[-20] - 1) * 100
                        spread = rsp_20d - spy_20d
                        if spread > 1.0:
                            breadth = "BROAD — equal-weight outperforming (healthy breadth)"
                        elif spread < -1.0:
                            breadth = "NARROW — cap-weighted leading (top-heavy, fragile)"
                        else:
                            breadth = "BALANCED — similar performance"
                        _mkt_lines.append(f"  Market Breadth: RSP 20d {rsp_20d:+.1f}% vs SPY {spy_20d:+.1f}% → {breadth}")
                except Exception:
                    pass
        except Exception as e:
            _mkt_lines.append(f"  Error fetching market data: {str(e)[:100]}")

        # Sector rotation (already cached in session state from startup)
        _mkt_lines.append(f"\n📊 SECTOR ROTATION (all sectors vs SPY):")
        try:
            all_sectors = st.session_state.get('sector_rotation')
            if not all_sectors:
                from data_fetcher import fetch_sector_rotation
                all_sectors = fetch_sector_rotation()
            if all_sectors:
                sorted_sectors = sorted(all_sectors.items(),
                                        key=lambda x: x[1].get('vs_spy_20d', 0), reverse=True)
                for sector_name, data in sorted_sectors:
                    phase = data.get('phase', '?')
                    vs_spy = data.get('vs_spy_20d', 0)
                    perf_20d = data.get('perf_20d', 0)
                    etf = data.get('etf', '?')
                    icon = "🟢" if phase == 'LEADING' else "🟡" if phase == 'WEAKENING' else "🔴" if phase == 'LAGGING' else "⚪"
                    _mkt_lines.append(f"  {icon} {sector_name} ({etf}): {phase} | 20d: {perf_20d:+.1f}% | vs SPY: {vs_spy:+.1f}%")
            else:
                _mkt_lines.append(f"  Sector data unavailable")
        except Exception as e:
            _mkt_lines.append(f"  Error: {str(e)[:100]}")

        # Cache the combined market context
        _mkt_text = "\n".join(_mkt_lines)
        st.session_state['_research_market_cache'] = _mkt_text
        st.session_state['_research_market_cache_ts'] = _now
        lines.append(_mkt_text)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: EARNINGS DATE (MANDATORY — try multiple sources)
    # ══════════════════════════════════════════════════════════════════
    lines.append(f"\n📅 EARNINGS DATE (MANDATORY — you MUST state this):")
    earnings_found = False
    earnings_date_str = None
    earnings_days = None

    # Source 1: App session (batch scanner)
    earn_flag = st.session_state.get('earnings_flags', {}).get(ticker)
    if earn_flag:
        earnings_date_str = earn_flag.get('next_earnings')
        earnings_days = earn_flag.get('days_until')
        earnings_found = True
        lines.append(f"  Source: TTA Scanner")
        lines.append(f"  Next Earnings: {earnings_date_str}")
        lines.append(f"  Days Until: {earnings_days}")

    # Source 2: yfinance (fresh fetch if scanner didn't have it)
    if not earnings_found:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)

            # Try .calendar
            try:
                cal = stock.calendar
                if cal is not None:
                    raw_date = None
                    if isinstance(cal, dict):
                        raw_date = cal.get('Earnings Date')
                        if isinstance(raw_date, list) and raw_date:
                            raw_date = raw_date[0]
                    if raw_date is not None:
                        if hasattr(raw_date, 'date'):
                            earn_dt = raw_date.date()
                        elif isinstance(raw_date, str):
                            earn_dt = datetime.strptime(raw_date[:10], '%Y-%m-%d').date()
                        else:
                            earn_dt = None
                        if earn_dt:
                            earnings_days = (earn_dt - datetime.now().date()).days
                            earnings_date_str = earn_dt.strftime('%Y-%m-%d')
                            earnings_found = True
                            lines.append(f"  Source: Yahoo Finance (.calendar)")
                            lines.append(f"  Next Earnings: {earnings_date_str}")
                            lines.append(f"  Days Until: {earnings_days}")
            except Exception:
                pass

            # Try info dict
            if not earnings_found:
                try:
                    info = stock.info or {}
                    for key in ('earningsTimestamp', 'earningsTimestampStart'):
                        ts = info.get(key)
                        if ts and isinstance(ts, (int, float)) and ts > 0:
                            from datetime import timezone
                            earn_dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                            earnings_days = (earn_dt - datetime.now().date()).days
                            if earnings_days >= -7:
                                earnings_date_str = earn_dt.strftime('%Y-%m-%d')
                                earnings_found = True
                                lines.append(f"  Source: Yahoo Finance (info)")
                                lines.append(f"  Next Earnings: {earnings_date_str}")
                                lines.append(f"  Days Until: {earnings_days}")
                                break
                except Exception:
                    pass

            # Try .earnings_dates
            if not earnings_found:
                try:
                    edates = stock.earnings_dates
                    if edates is not None and len(edates) > 0:
                        today = datetime.now().date()
                        for dt_idx in sorted(edates.index):
                            try:
                                d = dt_idx.date() if hasattr(dt_idx, 'date') else dt_idx.to_pydatetime().date()
                                if (d - today).days >= -7:
                                    earnings_date_str = d.strftime('%Y-%m-%d')
                                    earnings_days = (d - today).days
                                    earnings_found = True
                                    lines.append(f"  Source: Yahoo Finance (.earnings_dates)")
                                    lines.append(f"  Next Earnings: {earnings_date_str}")
                                    lines.append(f"  Days Until: {earnings_days}")
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
        except Exception as e:
            lines.append(f"  Yahoo earnings fetch error: {str(e)[:100]}")

    if not earnings_found:
        lines.append(f"  ⚠️ EARNINGS DATE NOT FOUND — tell user this could not be determined")
        lines.append(f"  (Some stocks, especially small-caps or foreign ADRs, may not have scheduled dates)")

    # Earnings risk assessment
    if earnings_found and earnings_days is not None:
        if earnings_days <= 7:
            lines.append(f"  🚨 CRITICAL RISK: Earnings in {earnings_days} days — binary event imminent")
            lines.append(f"  → Any position recommendation MUST account for earnings gap risk")
            lines.append(f"  → Consider: wait until after earnings, or use options to define risk")
        elif earnings_days <= 14:
            lines.append(f"  ⚠️ HIGH RISK: Earnings in {earnings_days} days — limited trading window")
            lines.append(f"  → A 3-6 month hold recommendation does NOT make sense here")
            lines.append(f"  → Must be an earnings play or wait until after report")
        elif earnings_days <= 30:
            lines.append(f"  ⚠️ CAUTION: Earnings within 30 days — adjust hold duration")
            lines.append(f"  → Any swing trade must have exit plan BEFORE earnings")
        elif earnings_days <= 60:
            lines.append(f"  ℹ️ Earnings approaching in {earnings_days} days — factor into hold duration")
        else:
            lines.append(f"  ✅ Clear runway: {earnings_days} days before next earnings report")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: NEWS
    # ══════════════════════════════════════════════════════════════════
    try:
        finnhub_key = ""
        try:
            finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
        except Exception:
            pass

        if finnhub_key:
            from data_fetcher import fetch_finnhub_news
            news = fetch_finnhub_news(ticker, api_key=finnhub_key)
            if news and news.get('articles'):
                lines.append(f"\n📰 RECENT NEWS (last 7 days):")
                for article in news['articles'][:10]:
                    headline = article.get('headline', article.get('title', '?'))
                    source = article.get('source', '?')
                    date = article.get('datetime', '')
                    if isinstance(date, (int, float)) and date > 0:
                        from datetime import datetime as dt
                        try:
                            date = dt.fromtimestamp(date).strftime('%Y-%m-%d')
                        except Exception:
                            date = ''
                    summary = article.get('summary', '')[:300]
                    lines.append(f"  [{date}] {headline} — {source}")
                    if summary:
                        lines.append(f"    Summary: {summary}")
            else:
                lines.append(f"\n📰 NEWS: No articles found on Finnhub for last 7 days")
        else:
            lines.append(f"\n📰 NEWS: No Finnhub API key — limited news data")
    except Exception as e:
        lines.append(f"\n📰 NEWS ERROR: {str(e)[:100]}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5: YAHOO FINANCE FUNDAMENTALS + ANALYST DATA
    # ══════════════════════════════════════════════════════════════════
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        if info:
            lines.append(f"\n📊 YAHOO FINANCE DATA:")
            # Company basics
            for key, label in [
                ('shortName', 'Company'), ('industry', 'Industry'), ('sector', 'Sector'),
            ]:
                val = info.get(key)
                if val:
                    lines.append(f"  {label}: {val}")

            # Business description (critical for sector misclassification detection)
            biz_summary = info.get('longBusinessSummary', '')
            if biz_summary:
                # First 400 chars is enough to identify actual business
                lines.append(f"  Business: {biz_summary[:400]}")

            # Valuation
            lines.append(f"  --- Valuation ---")
            mc = info.get('marketCap')
            if mc:
                if mc >= 1e12: mc_str = f"${mc/1e12:.1f}T"
                elif mc >= 1e9: mc_str = f"${mc/1e9:.1f}B"
                else: mc_str = f"${mc/1e6:.0f}M"
                lines.append(f"  Market Cap: {mc_str}")
            for key, label in [
                ('trailingPE', 'Trailing P/E'), ('forwardPE', 'Forward P/E'),
                ('priceToBook', 'P/B'), ('enterpriseToRevenue', 'EV/Revenue'),
                ('trailingEps', 'EPS'), ('dividendYield', 'Dividend Yield'),
            ]:
                val = info.get(key)
                if val is not None:
                    if 'Yield' in label:
                        lines.append(f"  {label}: {val*100:.2f}%")
                    else:
                        lines.append(f"  {label}: {val:.2f}")

            # Growth & profitability
            lines.append(f"  --- Growth & Profitability ---")
            for key, label in [
                ('revenueGrowth', 'Revenue Growth'), ('earningsGrowth', 'Earnings Growth'),
                ('profitMargins', 'Profit Margin'), ('grossMargins', 'Gross Margin'),
                ('operatingMargins', 'Operating Margin'), ('returnOnEquity', 'ROE'),
            ]:
                val = info.get(key)
                if val is not None:
                    lines.append(f"  {label}: {val*100:.1f}%")

            # Analyst data
            lines.append(f"  --- Analyst Consensus ---")
            rec_key = info.get('recommendationKey', '')
            rec_mean = info.get('recommendationMean')
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            num_analysts = info.get('numberOfAnalystOpinions')
            current = info.get('currentPrice') or info.get('regularMarketPrice')

            if rec_key:
                lines.append(f"  Yahoo Recommendation: {rec_key.upper()}")
            if rec_mean:
                lines.append(f"  Recommendation Score: {rec_mean:.1f} (1=Strong Buy, 5=Strong Sell)")
            if target_mean and current:
                upside = (target_mean - current) / current * 100
                lines.append(f"  Target: ${target_low:.2f} — ${target_mean:.2f} — ${target_high:.2f} ({upside:+.1f}% to mean)")
            if num_analysts:
                lines.append(f"  Analysts Covering: {num_analysts}")

            # Short interest
            lines.append(f"  --- Short Interest & Risk ---")
            for key, label in [
                ('shortPercentOfFloat', 'Short % of Float'),
                ('shortRatio', 'Short Ratio (days to cover)'),
                ('beta', 'Beta'),
            ]:
                val = info.get(key)
                if val is not None:
                    if 'Percent' in label:
                        lines.append(f"  {label}: {val*100:.1f}%")
                    else:
                        lines.append(f"  {label}: {val:.2f}")

            # 52-week range
            high52 = info.get('fiftyTwoWeekHigh')
            low52 = info.get('fiftyTwoWeekLow')
            if high52 and low52 and current:
                range_pct = (current - low52) / (high52 - low52) * 100 if high52 != low52 else 50
                lines.append(f"  52-Week: ${low52:.2f} — ${high52:.2f} (currently at {range_pct:.0f}% of range)")

            # Ownership
            insider_pct = info.get('heldPercentInsiders')
            inst_pct = info.get('heldPercentInstitutions')
            if insider_pct is not None:
                lines.append(f"  Insider Ownership: {insider_pct*100:.1f}% (static stake, NOT selling)")
            if inst_pct is not None:
                lines.append(f"  Institutional Ownership: {inst_pct*100:.1f}%")

        # Yahoo news
        try:
            news_items = stock.news
            if news_items:
                lines.append(f"\n📰 YAHOO NEWS:")
                for item in news_items[:8]:
                    title = item.get('title', '?')
                    publisher = item.get('publisher', '?')
                    lines.append(f"  • {title} ({publisher})")
        except Exception:
            pass

    except Exception as e:
        lines.append(f"\nYAHOO ERROR: {str(e)[:100]}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 6: SOCIAL / VOLUME SENTIMENT
    # ══════════════════════════════════════════════════════════════════
    try:
        from data_fetcher import fetch_daily
        daily = fetch_daily(ticker, period='3mo')
        if daily is not None and len(daily) >= 20 and 'Volume' in daily.columns:
            recent_vol = float(daily['Volume'].iloc[-5:].mean())
            avg_vol = float(daily['Volume'].tail(50).mean())
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

            lines.append(f"\n📱 SOCIAL/VOLUME SENTIMENT PROXY:")
            lines.append(f"  5-day avg volume vs 50-day avg: {vol_ratio:.1f}x")
            if vol_ratio >= 3.0:
                lines.append(f"  Signal: 🔥 EXTREME volume surge — major institutional activity or news-driven")
            elif vol_ratio >= 2.0:
                lines.append(f"  Signal: 📈 ELEVATED volume — increased interest, possible accumulation")
            elif vol_ratio >= 1.5:
                lines.append(f"  Signal: 📊 Above average — moderate interest")
            elif vol_ratio <= 0.5:
                lines.append(f"  Signal: 😴 Very LOW volume — lack of interest, thin liquidity risk")
            else:
                lines.append(f"  Signal: Normal trading volume")

            # Recent price action context
            if len(daily) >= 5:
                last_5_return = (daily['Close'].iloc[-1] / daily['Close'].iloc[-5] - 1) * 100
                last_20_return = (daily['Close'].iloc[-1] / daily['Close'].iloc[-20] - 1) * 100 if len(daily) >= 20 else None
                lines.append(f"  5-day return: {last_5_return:+.1f}%")
                if last_20_return is not None:
                    lines.append(f"  20-day return: {last_20_return:+.1f}%")
    except Exception:
        pass

    # Finnhub Social Sentiment (if premium key)
    try:
        finnhub_key = ""
        try:
            finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
        except Exception:
            pass

        if finnhub_key:
            import requests as req
            import pandas as pd
            from_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={ticker}&from={from_date}&token={finnhub_key}"
            resp = req.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                reddit = data.get('reddit', [])
                twitter = data.get('twitter', [])
                if reddit or twitter:
                    r_mentions = sum(r.get('mention', 0) for r in reddit[-7:]) if reddit else 0
                    t_mentions = sum(t.get('mention', 0) for t in twitter[-7:]) if twitter else 0
                    lines.append(f"\n📱 FINNHUB SOCIAL SENTIMENT (7 days):")
                    lines.append(f"  Reddit mentions: {r_mentions}")
                    lines.append(f"  Twitter mentions: {t_mentions}")
                    total = r_mentions + t_mentions
                    if total > 100:
                        lines.append(f"  Assessment: HIGH social buzz — stock is being actively discussed")
                    elif total > 20:
                        lines.append(f"  Assessment: Moderate social interest")
                    elif total > 0:
                        lines.append(f"  Assessment: Low social mentions")
                    else:
                        lines.append(f"  Assessment: Minimal social presence")
            elif resp.status_code in (401, 403):
                lines.append(f"\n📱 FINNHUB SOCIAL: Premium subscription required for social sentiment data")
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════
    # SECTION 7: CORRELATED ASSET PRICE ACTION
    # (Auto-detects crypto miners, oil/gas, gold miners, etc.)
    # ══════════════════════════════════════════════════════════════════
    try:
        # Re-use Yahoo info if available, otherwise fetch
        try:
            _yf_info = stock.info if 'stock' in dir() else yf.Ticker(ticker).info
        except Exception:
            import yfinance as yf
            _yf_info = yf.Ticker(ticker).info or {}

        biz = (_yf_info.get('longBusinessSummary', '') + ' ' +
               _yf_info.get('industry', '') + ' ' +
               _yf_info.get('sector', '')).lower()

        # Map business keywords to correlated assets
        CORRELATED_ASSETS = {
            'BTC-USD': {
                'name': 'Bitcoin',
                'keywords': ['bitcoin', 'crypto', 'blockchain', 'mining', 'btc',
                             'digital asset', 'digital currency', 'hash rate'],
            },
            'CL=F': {
                'name': 'Crude Oil (WTI)',
                'keywords': ['oil', 'petroleum', 'crude', 'drilling', 'upstream',
                             'exploration and production', 'e&p'],
            },
            'GC=F': {
                'name': 'Gold',
                'keywords': ['gold mining', 'gold miner', 'precious metal',
                             'gold exploration', 'gold production'],
            },
            'SI=F': {
                'name': 'Silver',
                'keywords': ['silver mining', 'silver miner'],
            },
            'NG=F': {
                'name': 'Natural Gas',
                'keywords': ['natural gas', 'lng', 'gas producer'],
            },
            'ETH-USD': {
                'name': 'Ethereum',
                'keywords': ['ethereum', 'defi', 'smart contract', 'eth'],
            },
        }

        matched_assets = []
        for asset_ticker, asset_info in CORRELATED_ASSETS.items():
            if any(kw in biz for kw in asset_info['keywords']):
                matched_assets.append((asset_ticker, asset_info['name']))

        if matched_assets:
            from data_fetcher import fetch_daily
            lines.append(f"\n🔗 CORRELATED ASSET PRICE ACTION:")
            for asset_ticker, asset_name in matched_assets:
                try:
                    asset_df = fetch_daily(asset_ticker, period='3mo')
                    if asset_df is not None and len(asset_df) >= 20:
                        current = float(asset_df['Close'].iloc[-1])
                        ret_5d = (asset_df['Close'].iloc[-1] / asset_df['Close'].iloc[-5] - 1) * 100
                        ret_20d = (asset_df['Close'].iloc[-1] / asset_df['Close'].iloc[-20] - 1) * 100
                        high_3mo = float(asset_df['Close'].max())
                        low_3mo = float(asset_df['Close'].min())
                        pct_from_high = (current / high_3mo - 1) * 100

                        # Format price based on asset
                        if current >= 1000:
                            price_str = f"${current:,.0f}"
                        elif current >= 1:
                            price_str = f"${current:,.2f}"
                        else:
                            price_str = f"${current:.4f}"

                        lines.append(f"  {asset_name} ({asset_ticker}): {price_str}")
                        lines.append(f"    5d: {ret_5d:+.1f}% | 20d: {ret_20d:+.1f}% | vs 3mo high: {pct_from_high:+.1f}%")
                        lines.append(f"    3mo range: ${low_3mo:,.0f} — ${high_3mo:,.0f}")

                        # Directional assessment
                        if ret_5d > 3 and ret_20d > 5:
                            lines.append(f"    → {asset_name} in STRONG UPTREND — supports bullish thesis")
                        elif ret_5d > 0 and ret_20d > 0:
                            lines.append(f"    → {asset_name} trending UP — mildly supportive")
                        elif ret_5d < -3 and ret_20d < -5:
                            lines.append(f"    → {asset_name} in DOWNTREND — HEADWIND for {ticker}")
                        elif ret_5d < 0:
                            lines.append(f"    → {asset_name} pulling back — near-term caution")
                        else:
                            lines.append(f"    → {asset_name} FLAT — neutral for {ticker}")
                except Exception:
                    lines.append(f"  {asset_name}: Data unavailable")
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════
    # SECTION 8: EARNINGS VOLATILITY ESTIMATE
    # (For stocks with imminent earnings + high short interest)
    # ══════════════════════════════════════════════════════════════════
    try:
        # Only generate if earnings within 30 days
        if earnings_found and earnings_days is not None and earnings_days <= 30:
            try:
                _yf_info2 = stock.info if 'stock' in dir() else {}
            except Exception:
                _yf_info2 = {}

            short_pct = _yf_info2.get('shortPercentOfFloat', 0) or 0
            beta_val = _yf_info2.get('beta', 1.0) or 1.0

            # Historical earnings moves (from earnings_dates if available)
            hist_moves = []
            try:
                import yfinance as yf
                _stock = yf.Ticker(ticker)
                edates = _stock.earnings_dates
                if edates is not None and len(edates) >= 2:
                    from data_fetcher import fetch_daily
                    hist_df = fetch_daily(ticker, period='2y')
                    if hist_df is not None and len(hist_df) > 50:
                        for dt_idx in edates.index:
                            try:
                                d = dt_idx.date() if hasattr(dt_idx, 'date') else dt_idx.to_pydatetime().date()
                                # Find the trading day before and after earnings
                                mask_before = hist_df.index.date <= d
                                mask_after = hist_df.index.date >= d
                                if mask_before.any() and mask_after.any():
                                    close_before = float(hist_df.loc[mask_before, 'Close'].iloc[-1])
                                    # Day after (or same day if reported pre-market)
                                    after_df = hist_df.loc[mask_after]
                                    if len(after_df) >= 2:
                                        close_after = float(after_df['Close'].iloc[1])
                                        move_pct = (close_after / close_before - 1) * 100
                                        hist_moves.append(round(move_pct, 1))
                            except Exception:
                                continue
            except Exception:
                pass

            lines.append(f"\n📊 EARNINGS VOLATILITY ESTIMATE:")
            lines.append(f"  Days to earnings: {earnings_days}")
            lines.append(f"  Short % of float: {short_pct*100:.1f}%" if short_pct else "  Short %: N/A")
            lines.append(f"  Beta: {beta_val:.2f}")

            if hist_moves and len(hist_moves) >= 2:
                avg_move = sum(abs(m) for m in hist_moves) / len(hist_moves)
                max_move = max(abs(m) for m in hist_moves)
                lines.append(f"  Historical earnings moves (last {len(hist_moves)}): {', '.join(f'{m:+.1f}%' for m in hist_moves[:6])}")
                lines.append(f"  Average absolute move: ±{avg_move:.1f}%")
                lines.append(f"  Largest move: ±{max_move:.1f}%")
            else:
                # Estimate from beta and short interest
                base_move = 8.0  # Average S&P stock earnings move
                beta_adj = base_move * beta_val
                short_adj = beta_adj * (1 + short_pct * 2) if short_pct else beta_adj
                lines.append(f"  No historical earnings data — estimating from beta + short interest:")
                lines.append(f"  Estimated earnings move: ±{short_adj:.0f}–{short_adj*1.5:.0f}%")

            # Short squeeze potential
            if short_pct and short_pct > 0.15:
                lines.append(f"  ⚠️ SHORT SQUEEZE RISK: {short_pct*100:.1f}% short interest")
                lines.append(f"  → If earnings beat: potential for {short_pct*100:.0f}–{short_pct*200:.0f}%+ upside squeeze")
                lines.append(f"  → If earnings miss: shorts will pile on, amplifying downside")
                lines.append(f"  → Standard stop losses may be INEFFECTIVE on gap moves — stock could gap past your stop")
            elif short_pct and short_pct > 0.10:
                lines.append(f"  ℹ️ Elevated short interest ({short_pct*100:.1f}%) — earnings moves likely amplified")
    except Exception:
        pass

    return "\n".join(lines)


def _render_chat_tab(ticker: str, signal: EntrySignal, rec: Dict,
                     analysis: TickerAnalysis):
    """AI Research Analyst — auto-runs external research + interactive follow-up chat."""

    # ── Initialize AI client (auto-detect Groq vs xAI/Grok) ─────────
    openai_client = None
    ai_config = st.session_state.get('_ai_config')  # From AI Intel pre-flight

    if not ai_config:
        # First time — detect from key
        try:
            raw_key = st.secrets.get("GROQ_API_KEY", "")
            ai_config = _detect_ai_provider(raw_key)
            st.session_state['_ai_config'] = ai_config
        except Exception:
            ai_config = {'provider': 'none', 'key': '', 'model': '', 'fallback_model': '', 'base_url': '', 'display': 'Error'}

    if ai_config['key'] and ai_config['provider'] != 'none':
        # Check cached validation
        cached_status = st.session_state.get('_groq_key_status')
        cached_key = st.session_state.get('_groq_key_cached', '')
        if cached_status == 'invalid' and cached_key == ai_config['key']:
            pass  # Skip — known bad key
        else:
            try:
                from openai import OpenAI
                openai_client = OpenAI(
                    api_key=ai_config['key'],
                    base_url=ai_config['base_url'],
                )
            except Exception:
                pass

    if not openai_client:
        st.warning(f"🔑 **API key missing or invalid.** Current: {ai_config.get('display', 'none')}")
        st.caption("Add your API key in Settings → Secrets as `GROQ_API_KEY`. "
                   "Supports **Groq** (`gsk_...`) or **xAI/Grok** (`xai-...`) keys. "
                   "After updating, click 🔑 **Reset API** in ⚙️ Settings sidebar.")
        return

    # ── Chat state management (per ticker) ────────────────────────────
    chat_key = f'chat_history_{ticker}'
    research_key = f'chat_research_{ticker}'
    autorun_key = f'chat_autorun_{ticker}'

    # Reset when switching tickers
    if st.session_state.get('chat_active_ticker') != ticker:
        st.session_state[chat_key] = []
        st.session_state['chat_active_ticker'] = ticker
        st.session_state.pop(research_key, None)
        st.session_state.pop(autorun_key, None)

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # ── Build context lazily (only when needed for API calls) ─────────
    def _get_system_prompt():
        """Build system prompt on demand — avoids fetching external research on every render."""
        internal_context = _build_internal_context(ticker, signal, rec, analysis)

        # Fetch external research (cached per ticker within session)
        if research_key not in st.session_state:
            with st.spinner(f"🔍 Researching {ticker} — fetching news, analyst data, sentiment..."):
                external_research = _fetch_external_research(ticker)
                st.session_state[research_key] = external_research
        external_research = st.session_state.get(research_key, "External research not yet loaded.")

        return f"""You are a senior equity research analyst integrated into a stock trading application called TTA (Technical Trading Assistant).

═══ YOUR DATA SOURCES ═══

1. IN-APP DATA (the user can already see this on their screen — DO NOT list it back):
{internal_context}

2. EXTERNAL RESEARCH (you just gathered this — this is YOUR unique value):
{external_research}

3. {_build_sector_rotation_context()}

═══ MANDATORY ANALYSIS STRUCTURE ═══
Your response MUST include ALL sections below in this order. Omitting any section is a FAILURE.

**1. MARKET & SECTOR CONTEXT** (ALWAYS INCLUDE FIRST)
- Current overall equity market conditions (use SPY, VIX, breadth data provided)
- Identify the stock's SECTOR and INDUSTRY by name
- SECTOR CLASSIFICATION CHECK: Read the company's Business description. Does the Yahoo-assigned sector accurately reflect what this company actually does? Many companies are misclassified (e.g., crypto miners in "Financial Services", SaaS companies in "Industrials", EV companies in "Consumer Cyclical"). If the actual business doesn't match the assigned sector, SAY SO and explain which sector's rotation data is more relevant. Example: "WULF is classified as Financial Services but is actually a Bitcoin mining company — crypto/tech sector rotation is more relevant than traditional financials."
- Is the RELEVANT sector in rotation vs S&P 500? LEADING, LAGGING, or WEAKENING?
- Cite current sector performance data (20d vs SPY)
- CORRELATED ASSET: If your research data includes a "CORRELATED ASSET PRICE ACTION" section (e.g., Bitcoin for crypto miners, oil for E&P companies, gold for gold miners), you MUST mention it. State the correlated asset's current trend and whether it supports or contradicts the bullish thesis. This is critical context — a Bitcoin miner in a BTC downtrend faces a massive headwind regardless of its own technicals.
- If sector rotation data unavailable, explicitly state this limitation

**2. EARNINGS & TRADE TIMING** (CRITICAL — ALWAYS INCLUDE)
- State the next earnings date and exact days remaining
- If earnings <30 days away: flag as HIGH RISK and assess viability
- VOLATILITY ESTIMATE: If your research data includes an "EARNINGS VOLATILITY ESTIMATE" section, you MUST reference it. State the estimated or historical earnings move range (e.g., "±15-25%"). If short interest is >15%, explicitly warn that a gap move could blow past any stop loss — the stock could open 20-30% lower/higher than the prior close. This affects whether a stop loss is even a viable risk management tool.
- Is there a viable trade window before the next binary event?
- Note any other upcoming catalysts from news data
- If earnings date is missing, state: "Unable to fully assess trade timing — earnings date not found"

**3. TECHNICAL INTERPRETATION** (from in-app data)
- Reference but DO NOT LIST the technical indicators the user can already see on screen
- INTERPRET what the signals mean collectively — what's the story?
- Identify the most critical patterns, levels, and momentum state
- Volume and momentum assessment in context

**4. EXTERNAL INTELLIGENCE** (from your research data above)
- Latest analyst ratings, price targets, and any recent changes
- Social media / volume sentiment assessment
- Recent news highlights (past 7 days) — only material items
- Institutional/insider transaction activity (ownership % is NOT selling)
- Only cite data actually present in your research — do NOT hallucinate

**5. SYNTHESIZED RECOMMENDATION**
- **BUY / HOLD / PASS** with confidence level (High/Medium/Low)
- Entry price or zone (if BUY)
- Stop loss level (MUST include)
- Target price with upside %
- **RISK/REWARD RATIO** (MUST calculate explicitly): Upside % to target ÷ Downside % to stop. State the ratio (e.g., "1.75:1"). Assess whether it's adequate given the setup: 2:1+ is standard for swing trades; earnings plays with high short interest need 2.5:1+ to justify the binary risk. If R:R is below 1.5:1, flag it as inadequate.
- Hold duration — MUST be appropriate to earnings calendar
- Position sizing: Full (100%) / Reduced (75%) / Small (50%) / Skip — with reason

**6. POST-EARNINGS SCENARIOS** (REQUIRED if earnings are within 30 days)
If the next earnings report falls within the recommended hold period OR within 30 days, you MUST provide:
- **If earnings BEAT and stock gaps up:** Take profits at what level? Or reassess for a longer hold? What would confirm a continuation vs a "sell the news" fade?
- **If earnings MISS and stock gaps down:** Exit immediately regardless of stop loss? Or is there a lower support level worth holding to?
- **Recommended exit strategy:** Should this be a day-after-earnings exit regardless of direction? Or hold through? What's the plan for a flat/neutral reaction?
- **Pre-earnings positioning:** Should the full position be entered now, or scale in? Should part be hedged with options?
If earnings are 60+ days away, skip this section.

═══ CRITICAL RULES ═══
- Never recommend a multi-month hold if earnings are <30 days away without explicitly acknowledging the risk
- Your value is SYNTHESIS + EXTERNAL CONTEXT, not repeating in-app data
- When discussing insider activity, ONLY report actual BUY/SELL transactions — ownership % is NOT selling
- Cite sources for external data (e.g. "Yahoo analysts", "Finnhub news", "volume data")
- Do NOT say "Based on the app data..." or list signals back — INTERPRET them
- ALWAYS calculate and state the risk/reward ratio — never present a trade plan without it
- Keep under 600 words. Be decisive, not exhaustive. Every sentence must add value.

═══ FOR FOLLOW-UP QUESTIONS (CONVERSATIONAL Q&A MODE) ═══

When the user asks follow-up questions, switch to conversational mode:

YOUR AVAILABLE DATA SOURCES:
✅ In-app technical indicators (MACD, AO, RSI, volume, moving averages, Weinstein stages)
✅ Yahoo Finance (fundamentals, analyst ratings, insider transactions, earnings dates, options)
✅ Finnhub basic tier (news, company profile, social sentiment proxy)
✅ Market conditions (SPY, VIX, sector rotation, breadth)
✅ Correlated asset data (BTC, oil, gold for relevant stocks)
✅ Your previous analysis in this conversation — reference it freely

NOT AVAILABLE (be transparent):
❌ Real-time social media (Twitter/X, Reddit, StockTwits) — requires premium subscription
❌ Elliott Wave analysis — not in the system, suggest TradingView
❌ Level 2 / order book data — not available
❌ Advanced options flow (unusual activity) — only basic Yahoo options chain
❌ Dark pool / institutional flow data — not available
❌ Proprietary scoring models — only what's in the app

RESPONSE GUIDELINES:
- Be conversational, helpful, and direct — cite specific prices, percentages, dates
- Reference your previous analysis sections when relevant ("As I noted in Section 2...")
- When users ask about unavailable data:
  * Explain what you DO have access to as an alternative
  * Suggest external tools when appropriate (TradingView, Finviz, Unusual Whales, etc.)
  * Offer to analyze available proxy data instead
- When asked "why did you recommend X?": Give detailed reasoning referencing specific data points
- When asked hypotheticals ("what if BTC drops?"): Use available data to model scenarios
- If asked about something not in your data, say so honestly and suggest clicking "Refresh Research"
- You can discuss entry strategy, position sizing, risk management, catalysts, sector trends

EXAMPLE RESPONSES:
Q: "What's the sentiment on Twitter?"
→ "I don't have real-time Twitter/X access. However, I can see that [institutional ownership, insider activity, analyst consensus, social mention counts from Finnhub] — these 'smart money' signals often matter more than social buzz."

Q: "Can you do Elliott Wave?"
→ "Elliott Wave isn't part of my analysis tools. I work with MACD, AO, Weinstein stages, and support/resistance. For EW, try TradingView. I can help with trend structure using multi-timeframe momentum — want me to break that down?"

Q: "Why PASS when analysts say $23?"
→ [Reference specific sections from initial analysis with data points]"""

    # ── Header with controls ──────────────────────────────────────────
    hdr1, hdr2, hdr3 = st.columns([5, 2, 1])
    with hdr1:
        st.markdown(f"**🔬 AI Research Analyst — {ticker}**")
    with hdr2:
        if st.button("🔄 Refresh Research", key="chat_refresh_research",
                     help="Re-fetch latest news, analyst data & sentiment"):
            st.session_state.pop(research_key, None)
            st.session_state.pop(autorun_key, None)
            st.session_state[chat_key] = []
            st.rerun()
    with hdr3:
        if st.button("🗑️", key="chat_clear", help="Clear conversation"):
            st.session_state[chat_key] = []
            st.session_state.pop(autorun_key, None)
            st.rerun()

    # ── Auto-run initial analysis ─────────────────────────────────────
    history = st.session_state[chat_key]

    if not st.session_state.get(autorun_key):
        if not history:
            # Show run button — don't auto-fire (saves ~15 API calls per ticker switch)
            st.info("💬 **AI Research Analyst ready.** Click below to fetch research data & generate analysis.")
            if st.button("▶️ Run Research Analysis", type="primary", use_container_width=True,
                         key=f"chat_run_{ticker}"):
                pass  # Fall through to run the analysis below
            else:
                # Show data sources info even before running
                with st.expander("ℹ️ About Data Sources", expanded=False):
                    st.markdown("""**Available in this analysis:**
✅ Technical indicators (MACD, AO, Weinstein stages, volume, support/resistance)
✅ Yahoo Finance (fundamentals, analyst ratings, insider transactions, earnings)
✅ Finnhub (news, company profile, social sentiment proxy)
✅ Market context (SPY, VIX, sector rotation, breadth)
✅ Correlated assets (BTC for crypto miners, oil for E&P, gold for miners)

**Not available (will suggest alternatives):**
❌ Real-time social media (Twitter/X, Reddit) · ❌ Elliott Wave · ❌ Level 2 / dark pool · ❌ Advanced options flow""")
                return  # Don't run yet

        # Auto-populate and send the initial analysis request
        initial_query = f"Analyze {ticker} and provide a BUY/HOLD/PASS recommendation."

        history.append({'role': 'user', 'content': initial_query})

        messages = [
            {'role': 'system', 'content': _get_system_prompt()},
            {'role': 'user', 'content': initial_query},
        ]

        try:
            with st.spinner(f"🧠 Analyzing {ticker} — synthesizing signals + research..."):
                response = openai_client.chat.completions.create(
                    model=ai_config['model'],
                    messages=messages,
                    max_tokens=1800,
                    temperature=0.3,
                )
                reply = response.choices[0].message.content
                history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            err_str = str(e)
            # Cache 401 invalid key so we don't keep retrying
            if 'Invalid API Key' in err_str or '401' in err_str or 'Unauthorized' in err_str:
                try:
                    st.session_state['_groq_key_status'] = 'invalid'
                    st.session_state['_groq_key_cached'] = ai_config.get('key', '')
                except Exception:
                    pass
                history.append({'role': 'assistant',
                                'content': f"🔑 **API key invalid ({ai_config.get('display', 'unknown')}).** Update GROQ_API_KEY in secrets, then click 🔑 Reset API in sidebar."})
            else:
                # Fallback model
                try:
                    response = openai_client.chat.completions.create(
                        model=ai_config['fallback_model'],
                        messages=messages,
                        max_tokens=1400,
                        temperature=0.3,
                    )
                    reply = response.choices[0].message.content
                    history.append({'role': 'assistant', 'content': reply})
                except Exception as e2:
                    err2_str = str(e2)
                    if 'Invalid API Key' in err2_str or '401' in err2_str or 'Unauthorized' in err2_str:
                        try:
                            st.session_state['_groq_key_status'] = 'invalid'
                            st.session_state['_groq_key_cached'] = ai_config.get('key', '')
                        except Exception:
                            pass
                        history.append({'role': 'assistant',
                                        'content': f"🔑 **API key invalid.** Update GROQ_API_KEY in secrets, then click 🔑 Reset API."})
                    else:
                        history.append({'role': 'assistant',
                                        'content': f"⚠️ Analysis failed: {err_str[:200]}\nFallback: {err2_str[:200]}"})

        st.session_state[chat_key] = history
        st.session_state[autorun_key] = True

    # ── Display conversation ──────────────────────────────────────────
    for msg in history:
        with st.chat_message(msg['role'], avatar="🧑‍💼" if msg['role'] == 'user' else "🔬"):
            st.markdown(clean_ai_formatting(msg['content']) if msg['role'] == 'assistant' else msg['content'])

    # ── Suggested follow-ups (after initial analysis) ─────────────────
    if len(history) == 2:  # Just the auto-run Q&A
        st.caption("💡 **Ask follow-ups:** *\"What's the biggest risk here?\"* · "
                   "*\"Where exactly should I enter?\"* · "
                   "*\"What catalyst could move this 20%?\"* · "
                   "*\"Is smart money buying or selling?\"* · "
                   "*\"Compare bull and bear case\"* · "
                   "*\"How does this sector look right now?\"*")

    # ── Data sources transparency ─────────────────────────────────────
    with st.expander("ℹ️ About Data Sources", expanded=False):
        st.markdown("""
**Available in this analysis:**
- ✅ Technical indicators — MACD, AO, RSI, volume, moving averages, Weinstein stages
- ✅ Yahoo Finance — fundamentals, analyst ratings, insider transactions, earnings dates
- ✅ Finnhub (basic) — news, company profile, social mention counts
- ✅ Market context — SPY/VIX, sector rotation (all 11 GICS sectors), breadth
- ✅ Correlated assets — BTC, oil, gold (auto-detected for relevant stocks)
- ✅ Earnings volatility — historical moves, short squeeze risk assessment

**Not available (ask the AI for workarounds):**
- ❌ Real-time social media (Twitter/X, Reddit) — premium subscription required
- ❌ Elliott Wave analysis — use TradingView for this
- ❌ Level 2 / order book / dark pool data
- ❌ Advanced options flow (unusual activity beyond Yahoo chain)

The AI is transparent about limitations and will suggest alternatives using available data.
""")

    # ── Chat input for follow-ups ─────────────────────────────────────
    user_input = st.chat_input(f"Ask about {ticker}...", key=f"chat_input_{ticker}")

    if user_input:
        history.append({'role': 'user', 'content': user_input})

        with st.chat_message('user', avatar="🧑‍💼"):
            st.markdown(user_input)

        # Build full message chain (system + last 20 messages)
        messages = [{'role': 'system', 'content': _get_system_prompt()}]
        for msg in history[-20:]:
            messages.append({'role': msg['role'], 'content': msg['content']})

        with st.chat_message('assistant', avatar="🔬"):
            try:
                with st.spinner("Thinking..."):
                    response = openai_client.chat.completions.create(
                        model=ai_config['model'],
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.4,
                    )
                    reply = response.choices[0].message.content

                st.markdown(clean_ai_formatting(reply))
                history.append({'role': 'assistant', 'content': reply})
                st.session_state[chat_key] = history

            except Exception as e:
                error_msg = str(e)[:300]
                st.error(f"AI Error: {error_msg}")
                try:
                    with st.spinner("Retrying..."):
                        response = openai_client.chat.completions.create(
                            model=ai_config['fallback_model'],
                            messages=messages,
                            max_tokens=800,
                            temperature=0.4,
                        )
                        reply = response.choices[0].message.content
                    st.markdown(clean_ai_formatting(reply))
                    history.append({'role': 'assistant', 'content': reply})
                    st.session_state[chat_key] = history
                except Exception as e2:
                    st.error(f"Fallback failed: {str(e2)[:200]}")


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
    Institutional-grade position calculator using position_sizer.py.
    Shows full audit trail of sizing decision.
    """
    from position_sizer import calculate_position_size

    current_price = analysis.current_price or 0
    entry_default = float(stops.get('entry', current_price or 0))
    stop_default = float(stops.get('stop', 0))
    target_default = float(stops.get('target', 0))

    account_size = st.session_state.get('account_size', 100000.0)
    open_trades = jm.get_open_trades()

    # Get performance context
    win_rate = jm.get_recent_win_rate(last_n=20)
    losing_streak = jm.get_current_losing_streak()

    st.subheader(f"📐 Position Sizer — {ticker}")

    # ── Performance Context ───────────────────────────────────────────
    if losing_streak >= 2:
        st.warning(f"⚠️ On a {losing_streak}-trade losing streak — position sizes auto-reduced")
    if win_rate < 0.4 and jm.get_trade_history(last_n=5):
        st.warning(f"⚠️ Recent win rate: {win_rate:.0%} — consider reducing exposure")

    # ── Input Row ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        entry_price = st.number_input("Entry Price",
                                       value=entry_default if entry_default > 0 else current_price,
                                       step=0.01, format="%.2f",
                                       key=f"sizer_entry_{ticker}")
    with col2:
        stop_price = st.number_input("Stop Loss",
                                      value=stop_default,
                                      step=0.01, format="%.2f",
                                      key=f"sizer_stop_{ticker}")
    with col3:
        target_price = st.number_input("Target",
                                        value=target_default,
                                        step=0.01, format="%.2f",
                                        key=f"sizer_target_{ticker}")
    with col4:
        max_risk = st.number_input("Max Risk %", value=1.5,
                                    min_value=0.5, max_value=5.0, step=0.25, format="%.2f",
                                    key=f"sizer_risk_{ticker}")

    # ── Run Sizer ─────────────────────────────────────────────────────
    if entry_price > 0 and stop_price > 0 and entry_price > stop_price:
        result = calculate_position_size(
            ticker=ticker,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            account_size=account_size,
            open_positions=open_trades,
            recent_win_rate=win_rate,
            current_losing_streak=losing_streak,
            max_risk_pct=max_risk,
        )

        # ── Results Display ───────────────────────────────────────────
        st.markdown("---")

        # Main recommendation
        if result.recommended_shares > 0:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("✅ Recommended", f"{result.recommended_shares:,} shares",
                      f"${result.position_cost:,.0f}")
            r2.metric("💸 Risk", f"${result.risk_dollars:,.0f}",
                      f"{result.risk_pct_of_equity:.1f}% of equity")
            r3.metric("🔥 Heat", f"{result.portfolio_heat_before:.1f}% → {result.portfolio_heat_after:.1f}%",
                      f"{'✅' if result.portfolio_heat_after < 8 else '⚠️'}")
            if result.reward_risk_ratio > 0:
                r4.metric("🎯 R:R", f"{result.reward_risk_ratio:.1f}:1",
                          "Good ✅" if result.reward_risk_ratio >= 2 else "Low ⚠️")
            else:
                r4.metric("📊 Concentration", f"{result.concentration_pct:.1f}%",
                          f"{'✅' if result.concentration_pct < 20 else '⚠️'}")

            # Show constraint breakdown
            with st.expander("📊 Sizing Breakdown"):
                st.caption(f"**Risk limit (1.5%):** {result.shares_from_risk:,} shares")
                st.caption(f"**Heat limit (8%):** {result.shares_from_heat:,} shares")
                st.caption(f"**Concentration (20%):** {result.shares_from_concentration:,} shares")
                st.caption(f"**Available capital:** {result.shares_from_capital:,} shares")
                st.caption(f"**Binding constraint:** {result.limiting_factor.replace('_', ' ').title()}")

                if result.scale_factor < 1.0:
                    st.warning(f"⚠️ Base size: {result.base_shares:,} shares → "
                               f"Reduced to {result.recommended_shares:,} — {result.scale_reason}")

                st.caption(f"Win rate (last 20): {win_rate:.0%} | Losing streak: {losing_streak}")

            # Warnings
            for w in result.warnings:
                st.warning(w)
            if not result.warnings:
                st.success("✅ Position sizing within all risk parameters")

            # Store for trade entry
            st.session_state[f'sizer_result_{ticker}'] = result.recommended_shares

        else:
            st.error(result.explanation)

    elif entry_price > 0 and stop_price > 0 and stop_price >= entry_price:
        st.error("Stop price must be below entry price")

    # ══════════════════════════════════════════════════════════════
    # ENTER TRADE
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### ✅ Enter Trade")

    # Use sizer recommendation as default
    sizer_shares = st.session_state.get(f'sizer_result_{ticker}', 0)
    final_shares = sizer_shares if sizer_shares > 0 else (
        int(account_size * 0.125 / entry_price) if entry_price > 0 else 0
    )

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        confirm_shares = st.number_input("Shares", value=final_shares,
                                          min_value=0, step=1,
                                          key=f"confirm_shares_{ticker}")
    with ec2:
        confirm_entry = st.number_input("Entry $", value=float(entry_price) if entry_price > 0 else float(entry_default),
                                         step=0.01, format="%.2f",
                                         key=f"confirm_entry_{ticker}")
    with ec3:
        confirm_stop = st.number_input("Stop $", value=float(stop_default),
                                        step=0.01, format="%.2f",
                                        key=f"confirm_stop_{ticker}")
    with ec4:
        confirm_target = st.number_input("Target $", value=float(target_default),
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
    # Check if APEX signals are available in session state (from chart tab cache)
    _apex_cache = st.session_state.get(f'_apex_cache_{ticker}', {})
    apex_sigs = _apex_cache.get('signals', []) if _apex_cache else []
    if apex_sigs:
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
        elif apex_sigs:  # signals exist but none active
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
# POSITION MANAGER — Exit Advisor Tab
# =============================================================================

def render_position_manager():
    """Position Manager: AI-driven exit analysis for all open positions."""
    jm = get_journal()
    open_trades = jm.get_open_trades()

    st.subheader(f"🏦 Position Manager ({len(open_trades)} open)")

    if not open_trades:
        st.info("No open positions. Enter trades from the Trade tab to use the Position Manager.")
        return

    # ── Portfolio Summary ─────────────────────────────────────────────
    from data_fetcher import fetch_current_price

    total_deployed = 0
    total_unrealized = 0
    position_rows = []

    for trade in open_trades:
        ticker = trade['ticker']
        entry = float(trade.get('entry_price', 0))
        shares = float(trade.get('shares', 0))
        stop = float(trade.get('current_stop', trade.get('initial_stop', 0)))
        current = fetch_current_price(ticker) or entry

        pos_size = shares * entry
        total_deployed += pos_size
        pnl = (current - entry) * shares
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        total_unrealized += pnl

        # Days held
        days_held = 0
        entry_date = trade.get('entry_date', '')
        if entry_date:
            try:
                days_held = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
            except Exception:
                pass

        # Get last AI advice if available
        last_advice = st.session_state.get(f'exit_advice_{ticker}', {})

        position_rows.append({
            'ticker': ticker,
            'entry': entry,
            'current': current,
            'shares': shares,
            'stop': stop,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'pos_size': pos_size,
            'advice': last_advice,
        })

    # Summary metrics
    account_size = st.session_state.get('account_size', 100000.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed", f"${total_deployed:,.0f}",
              f"{total_deployed / account_size * 100:.0f}% of account")
    c2.metric("Unrealized P&L", f"${total_unrealized:+,.0f}",
              f"{total_unrealized / total_deployed * 100:+.1f}%" if total_deployed > 0 else "")
    c3.metric("Positions", f"{len(open_trades)}")

    # Portfolio heat
    risk_summary = jm.get_portfolio_risk_summary()
    heat_pct = (risk_summary['total_risk_dollars'] / account_size * 100) if account_size > 0 else 0
    heat_color = "normal" if heat_pct < 6 else ("off" if heat_pct < 8 else "inverse")
    c4.metric("Portfolio Heat", f"{heat_pct:.1f}%",
              f"${risk_summary['total_risk_dollars']:,.0f} at risk",
              delta_color=heat_color)

    st.divider()

    # ── Position Table with AI Advice ─────────────────────────────────
    for pos in position_rows:
        ticker = pos['ticker']
        pnl_icon = "🟢" if pos['pnl'] >= 0 else "🔴"
        advice = pos.get('advice', {})
        action = advice.get('action', '')

        action_icons = {
            'HOLD': '🟢 HOLD', 'TAKE_PARTIAL': '🟡 TAKE PARTIAL',
            'CLOSE': '🔴 CLOSE', 'TIGHTEN_STOP': '🔵 TIGHTEN STOP',
        }
        action_display = action_icons.get(action, '⚪ Not analyzed')

        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns([1.2, 1, 1, 1, 1.5, 1.5])
        pc1.markdown(f"**{pnl_icon} {ticker}**")
        pc2.caption(f"${pos['current']:.2f} ({pos['pnl_pct']:+.1f}%)")
        pc3.caption(f"${pos['pnl']:+,.0f}")
        pc4.caption(f"{pos['days_held']}d | Stop: ${pos['stop']:.2f}")
        pc5.markdown(f"**{action_display}**")

        with pc6:
            if st.button("📈 Chart", key=f"pm_chart_{ticker}"):
                st.session_state['default_detail_tab'] = 1
                _load_ticker_for_view(ticker)

        # Show advice details if available
        if advice.get('reasoning'):
            with st.expander(f"💡 {ticker} — {advice.get('reasoning', '')[:80]}"):
                st.caption(f"**Reasoning:** {advice.get('reasoning', '')}")
                st.caption(f"**Confidence:** {advice.get('confidence', 0)}/10 | Provider: {advice.get('provider', '')}")
                if advice.get('risk_note'):
                    st.caption(f"**Risk:** {advice.get('risk_note', '')}")
                if action == 'TIGHTEN_STOP' and advice.get('suggested_stop', 0) > 0:
                    st.caption(f"**Suggested Stop:** ${advice['suggested_stop']:.2f}")
                if action == 'TAKE_PARTIAL' and advice.get('partial_pct', 0) > 0:
                    st.caption(f"**Sell:** {advice['partial_pct']}% of position")

    # ── Analyze All Button ────────────────────────────────────────────
    st.divider()
    an_col1, an_col2, an_col3 = st.columns([1, 1, 2])

    with an_col1:
        if st.button("🤖 Analyze All Positions", type="primary", use_container_width=True):
            _run_exit_analysis(open_trades)

    with an_col2:
        if st.button("📧 Analyze + Email Report", use_container_width=True):
            _run_exit_analysis(open_trades, send_email=True)

    with an_col3:
        st.caption("AI will analyze each position and recommend: HOLD, TAKE PARTIAL, CLOSE, or TIGHTEN STOP")

    # ── Audit Log ─────────────────────────────────────────────────────
    with st.expander("📋 Audit Log (last 20 analyses)"):
        try:
            from exit_advisor import get_audit_history
            history = get_audit_history(last_n=20)
            if history:
                for h in history:
                    action_icon = {'HOLD': '🟢', 'TAKE_PARTIAL': '🟡',
                                   'CLOSE': '🔴', 'TIGHTEN_STOP': '🔵'}.get(h.get('action', ''), '⚪')
                    st.caption(
                        f"{h.get('analyzed_at', '')[:16]} | {action_icon} {h.get('ticker', '')} "
                        f"→ {h.get('action', '')} ({h.get('confidence', 0)}/10) "
                        f"| P&L: {h.get('unrealized_pnl_pct', 0):+.1f}% | {h.get('provider', '')}"
                    )
            else:
                st.caption("No audit history yet.")
        except Exception:
            st.caption("Audit log unavailable.")


def _run_exit_analysis(open_trades: List, send_email: bool = False):
    """Execute exit analysis for all open positions."""
    with st.spinner(f"Analyzing {len(open_trades)} positions..."):
        try:
            from exit_advisor import analyze_all_positions, save_audit_batch, send_email_report
            from data_fetcher import fetch_current_price, fetch_signal_for_exit

            gemini_model = st.session_state.get('gemini_model')
            openai_client = st.session_state.get('openai_client')

            advices = analyze_all_positions(
                open_trades,
                fetch_price_fn=fetch_current_price,
                fetch_signal_fn=fetch_signal_for_exit,
                gemini_model=gemini_model,
                openai_client=openai_client,
            )

            # Store results in session state for display
            for advice in advices:
                st.session_state[f'exit_advice_{advice.ticker}'] = advice.to_dict()

            # Save to audit log
            save_audit_batch(advices)

            # Send email if requested
            if send_email:
                import os
                smtp_email = os.environ.get('SMTP_EMAIL', '')
                smtp_password = os.environ.get('SMTP_PASSWORD', '')
                recipient = os.environ.get('ALERT_EMAIL', smtp_email)

                if smtp_email and smtp_password:
                    sent = send_email_report(advices, smtp_email, smtp_password, recipient)
                    if sent:
                        st.success(f"✅ Analyzed {len(advices)} positions + email sent to {recipient}")
                    else:
                        st.warning(f"✅ Analyzed {len(advices)} positions but email failed")
                else:
                    st.warning("Email not configured. Set SMTP_EMAIL, SMTP_PASSWORD, ALERT_EMAIL in Streamlit secrets.")
            else:
                st.success(f"✅ Analyzed {len(advices)} positions")

            st.rerun()

        except Exception as e:
            st.error(f"Exit analysis error: {e}")


# =============================================================================
# APP MAIN
# =============================================================================

def main():
    render_sidebar()

    jm = get_journal()

    # Main content area — added Alerts tab
    conditionals = jm.get_pending_conditionals()
    alerts_label = f"🎯 Alerts ({len(conditionals)})" if conditionals else "🎯 Alerts"
    tab_scanner, tab_alerts, tab_positions, tab_perf = st.tabs([
        "🔍 Scanner", alerts_label, "🏦 Position Manager", "📊 Performance"
    ])

    with tab_scanner:
        render_scanner_table()

        if st.session_state.get('selected_analysis'):
            st.divider()
            render_detail_view()

    with tab_alerts:
        _render_alerts_panel()

    with tab_positions:
        render_position_manager()

    with tab_perf:
        render_performance()


def _render_alerts_panel():
    """Dedicated alerts panel — moved from sidebar."""
    from data_fetcher import fetch_current_price
    jm = get_journal()

    conditionals = jm.get_pending_conditionals()
    if not conditionals:
        st.info("No active alerts. Set alerts from the trade tab when analyzing a ticker.")
        return

    st.subheader(f"🎯 Active Alerts ({len(conditionals)})")

    for cond in conditionals:
        ticker = cond['ticker']
        trigger = cond.get('trigger_price', 0)
        current = fetch_current_price(ticker) or 0
        dist_pct = ((trigger - current) / current * 100) if current > 0 else 0

        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 0.5])
        with col1:
            if st.button(f"📊 {ticker}", key=f"alert_view_{ticker}", use_container_width=True):
                _load_ticker_for_view(ticker)
        with col2:
            st.caption(f"Trigger: **${trigger:.2f}**")
        with col3:
            color = "🟢" if abs(dist_pct) < 3 else "🟡"
            st.caption(f"Current: ${current:.2f} ({color}{dist_pct:+.1f}%)")
        with col4:
            if st.button("✕", key=f"rm_alert_{ticker}"):
                jm.remove_conditional(ticker)
                st.rerun()


if __name__ == "__main__":
    main()
