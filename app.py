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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Backend imports
from signal_engine import EntrySignal
try:
    from data_fetcher import (
        fetch_all_ticker_data, fetch_scan_data, fetch_market_filter,
        fetch_current_price, fetch_daily, fetch_weekly, fetch_monthly,
        clear_cache,
    )
except KeyError:
    # Streamlit Cloud can race module loading during hot-reload after git pulls.
    # Retry via importlib to recover from transient sys.modules inconsistencies.
    import importlib as _importlib
    _df = _importlib.import_module("data_fetcher")
    fetch_all_ticker_data = _df.fetch_all_ticker_data
    fetch_scan_data = _df.fetch_scan_data
    fetch_market_filter = _df.fetch_market_filter
    fetch_current_price = _df.fetch_current_price
    fetch_daily = _df.fetch_daily
    fetch_weekly = _df.fetch_weekly
    fetch_monthly = _df.fetch_monthly
    clear_cache = _df.clear_cache
from scanner_engine import analyze_ticker, scan_watchlist, TickerAnalysis
from ai_analysis import analyze as run_ai_analysis
from chart_engine import render_tv_chart, render_mtf_chart
from journal_manager import JournalManager, WatchlistItem, Trade, ConditionalEntry, PlannedTrade
from watchlist_manager import WatchlistManager
from watchlist_bridge import WatchlistBridge
from apex_signals import detect_apex_signals, get_apex_markers, get_apex_summary
from scan_utils import resolve_tickers_to_scan
from trade_decision import build_trade_decision_card
from backup_health import get_backup_health_status, run_backup_now
from system_self_test import run_system_self_test
from trade_finder_helpers import build_planned_trade, build_trade_finder_selection, compute_trade_score


# =============================================================================
# AI TEXT CLEANUP — fix garbled formatting from LLM outputs
# =============================================================================

def clean_ai_formatting(text: str) -> str:
    """Fix common AI output formatting issues with currency, percentages, spacing, and markdown.
    
    Handles:
    - Markdown bold/italic stripping (***text***, **text**, *text* → text with proper spacing)
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

    # ── Strip markdown bold/italic markers with space preservation ──
    # Handle ***bold italic*** first (most greedy)
    text = re.sub(r'(\w)\*{3}(\w)', r'\1 \2', text)   # word***word → word word
    text = re.sub(r'\*{3}', '', text)                    # remaining ***

    # Handle **bold**
    text = re.sub(r'(\w)\*{2}(\w)', r'\1 \2', text)    # word**word → word word
    text = re.sub(r'\*{2}', '', text)                    # remaining **

    # Handle *italic* — careful not to hit multiplication
    # Pattern: *word(s)* where content has letters
    text = re.sub(r'(\w)\*([a-zA-Z])', r'\1 \2', text)  # word*text → word text
    text = re.sub(r'([a-zA-Z])\*(\w)', r'\1 \2', text)  # text*word → text word
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)          # remaining lone *

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

    # Fix camelCase word boundaries from AI word-smashing (e.g. "dataWithout" → "data Without")
    # But preserve intentional camelCase (single uppercase after lowercase is a word boundary)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)

    # Fix em-dash spacing
    text = re.sub(r'([a-zA-Z])—([a-zA-Z])', r'\1 — \2', text)

    # Fix number-letter concatenation (27times → 27 times)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Restore common abbreviations that should NOT have spaces
    text = re.sub(r'(\d+) ([dxwDXW])\b', r'\1\2', text)   # 200d, 2.5x, 52w
    text = re.sub(r'(\d+) ([KMBkmb])\b', r'\1\2', text)   # $15.2K, $3.2B
    text = re.sub(r'(\d) (st|nd|rd|th)\b', r'\1\2', text)  # 1st, 2nd, 3rd
    text = re.sub(r'\bQ (\d)\b', r'Q\1', text)             # Q1, Q2, Q3, Q4
    text = re.sub(r'\bR: R\b', r'R:R', text)               # R:R ratio
    text = re.sub(r'(\d) : (\d)', r'\1:\2', text)          # 5.9:1

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


# ═══════════════════════════════════════════════════════════════════════
# CENTRALIZED AI CLIENT — initialized once, cached in session_state
# Eliminates ~500ms of repeated imports + client creation on every rerun
# ═══════════════════════════════════════════════════════════════════════

def _get_ai_clients() -> Dict:
    """
    Get cached AI clients (OpenAI-compatible + Gemini).
    Creates clients once per session, validates once per key.
    Returns dict: {openai_client, gemini, ai_config, primary_error, gemini_error}
    """
    # Return cached if available and key hasn't changed and not invalidated
    cached = st.session_state.get('_ai_clients_cache')
    if cached:
        # Check if key changed (user updated secrets) or key was invalidated
        current_key = ""
        try:
            current_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
        key_invalidated = st.session_state.get('_groq_key_status') == 'invalid'
        if cached.get('_raw_key') == current_key and not key_invalidated:
            return cached

    result = {
        'openai_client': None,
        'gemini': None,
        'ai_config': {'model': '', 'fallback_model': '', 'provider': 'none', 'display': 'Not configured'},
        'primary_error': None,
        'gemini_error': None,
        '_raw_key': '',
    }

    # ── Primary provider (OpenAI-compatible: Groq or xAI) ─────────────
    try:
        raw_key = st.secrets.get("GROQ_API_KEY", "")
        result['_raw_key'] = raw_key
        ai_config = _detect_ai_provider(raw_key)
        result['ai_config'] = ai_config
        api_key = ai_config['key']

        if api_key and ai_config['provider'] != 'none':
            cached_status = st.session_state.get('_groq_key_status')
            cached_key_val = st.session_state.get('_groq_key_cached', '')
            if cached_status == 'invalid' and cached_key_val == api_key:
                result['primary_error'] = f"Key previously failed (401). Click 🔑 Reset API after updating. [{ai_config['display']}]"
            else:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=ai_config['base_url'])
                # Pre-flight validation (once per key)
                validation_key = f'_ai_validated_{api_key[:8]}'
                if validation_key not in st.session_state:
                    try:
                        client.chat.completions.create(
                            model=ai_config['model'],
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=1,
                        )
                        st.session_state[validation_key] = True
                    except Exception as val_err:
                        val_str = str(val_err)
                        if 'Invalid API Key' in val_str or '401' in val_str or 'Unauthorized' in val_str:
                            st.session_state['_groq_key_status'] = 'invalid'
                            st.session_state['_groq_key_cached'] = api_key
                            client = None
                            result['primary_error'] = f"Key validation failed (401). {ai_config['display']}"
                        else:
                            st.session_state[validation_key] = True
                result['openai_client'] = client
                st.session_state['_ai_config'] = ai_config
        else:
            result['primary_error'] = "No GROQ_API_KEY in secrets (supports Groq gsk_ or xAI xai- keys)"
    except ImportError:
        result['primary_error'] = "openai package not installed — add to requirements.txt"
    except Exception as e:
        result['primary_error'] = str(e)[:200]

    # ── Gemini fallback ───────────────────────────────────────────────
    try:
        import warnings
        # Suppress the deprecation warning from google-generativeai package
        # TODO: Migrate to google.genai when ready (google-generativeai is deprecated)
        warnings.filterwarnings("ignore", category=FutureWarning,
                                 module="google.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                 module="google.*")
        import google.generativeai as genai
        gkey = st.secrets.get("GEMINI_API_KEY", "")
        if gkey:
            genai.configure(api_key=gkey)
            result['gemini'] = genai.GenerativeModel('gemini-2.0-flash')
        else:
            result['gemini_error'] = "No GEMINI_API_KEY in secrets"
    except ImportError:
        result['gemini_error'] = "google-generativeai not installed"
    except Exception as e:
        result['gemini_error'] = str(e)[:200]

    # Cache for all subsequent reruns
    # Keep legacy session keys in sync for older call sites.
    st.session_state['openai_client'] = result.get('openai_client')
    st.session_state['gemini_model'] = result.get('gemini')
    st.session_state['_ai_clients_cache'] = result
    return result

# ── Restore data files from GitHub backup (before any data managers load) ──
if '_github_backup_restored' not in st.session_state:
    try:
        import github_backup
        restored = github_backup.restore_all()
        if restored:
            files_restored = [k for k, v in restored.items() if v]
            if files_restored:
                print(f"[startup] Restored from GitHub: {', '.join(files_restored)}")
    except Exception as e:
        print(f"[startup] GitHub restore skipped: {e}")
    st.session_state['_github_backup_restored'] = True

# Initialize journal
if 'journal' not in st.session_state:
    st.session_state['journal'] = JournalManager(data_dir=".")

# Initialize watchlist bridge (routes jm.get_watchlist_tickers() → new JSON system)
if 'watchlist_bridge' not in st.session_state:
    _wm = WatchlistManager()
    st.session_state['watchlist_bridge'] = WatchlistBridge(_wm, st.session_state['journal'])

# =============================================================================
# PER-WATCHLIST SCAN CACHE — persist scan results per watchlist
# =============================================================================
import json as _json
from pathlib import Path as _Path

_SCAN_CACHE_FILE = _Path("v2_scan_cache.json")
_AUDIT_FILE = _Path("v2_watchlist_audit.json")
_PERF_FILE = _Path("v2_perf_metrics.jsonl")
_AUDIT_MAX_ENTRIES = 500

def _load_scan_cache_file() -> dict:
    """Load the per-watchlist scan cache from disk."""
    if _SCAN_CACHE_FILE.exists():
        try:
            with open(_SCAN_CACHE_FILE, 'r') as f:
                return _json.load(f)
        except Exception:
            return {}
    return {}

def _save_scan_cache_file(cache: dict):
    """Save the per-watchlist scan cache to disk (atomic write)."""
    tmp = str(_SCAN_CACHE_FILE) + ".tmp"
    try:
        with open(tmp, 'w') as f:
            _json.dump(cache, f)
        import os
        os.replace(tmp, str(_SCAN_CACHE_FILE))
        # Queue for GitHub backup
        try:
            import github_backup
            github_backup.mark_dirty(_SCAN_CACHE_FILE.name)
        except ImportError:
            pass
    except Exception:
        pass


def _load_audit_file() -> list:
    """Load persistent watchlist/scan audit events."""
    if _AUDIT_FILE.exists():
        try:
            with open(_AUDIT_FILE, 'r') as f:
                data = _json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_audit_file(events: list):
    """Persist audit events to disk (atomic write)."""
    tmp = str(_AUDIT_FILE) + ".tmp"
    try:
        with open(tmp, 'w') as f:
            _json.dump(events, f)
        import os
        os.replace(tmp, str(_AUDIT_FILE))
        try:
            import github_backup
            github_backup.mark_dirty(_AUDIT_FILE.name)
        except ImportError:
            pass
    except Exception:
        pass


def _get_audit_events() -> list:
    """Get cached audit events, loading from disk once per session."""
    if '_watchlist_audit_events' not in st.session_state:
        st.session_state['_watchlist_audit_events'] = _load_audit_file()
    return st.session_state['_watchlist_audit_events']


def _append_audit_event(action: str, details: str = '', source: str = 'ui'):
    """Append a watchlist/scan activity event."""
    wl_id = ''
    wl_name = ''
    try:
        active = st.session_state['watchlist_bridge'].manager.get_active_watchlist()
        wl_id = active.get('id', '')
        wl_name = active.get('name', '')
    except Exception:
        pass

    events = _get_audit_events()
    events.insert(0, {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'action': action,
        'source': source,
        'wl_id': wl_id,
        'wl_name': wl_name,
        'details': details[:300] if details else '',
    })
    if len(events) > _AUDIT_MAX_ENTRIES:
        del events[_AUDIT_MAX_ENTRIES:]
    st.session_state['_watchlist_audit_events'] = events
    _save_audit_file(events)


def _clear_audit_events():
    """Clear all audit events."""
    st.session_state['_watchlist_audit_events'] = []
    _save_audit_file([])


def _count_today_trade_entries() -> int:
    """Count today's entered trades from audit trail."""
    today = datetime.now().strftime('%Y-%m-%d')
    return sum(
        1 for e in _get_audit_events()
        if e.get('action') == 'ENTER_TRADE' and str(e.get('ts', '')).startswith(today)
    )


def _append_perf_metric(metric: Dict[str, Any]):
    """Append performance telemetry row as JSONL."""
    try:
        row = {
            "ts": datetime.now().isoformat(timespec='seconds'),
            **metric,
        }
        with open(_PERF_FILE, "a") as f:
            f.write(_json.dumps(row) + "\n")
        try:
            import github_backup
            github_backup.mark_dirty(_PERF_FILE.name)
        except Exception:
            pass
    except Exception:
        pass


def save_scan_for_watchlist(wl_id: str):
    """Save current session scan results to per-watchlist cache (session + disk)."""
    scan_data = {
        'results': st.session_state.get('scan_results_summary', []),
        'timestamp': st.session_state.get('scan_timestamp', ''),
    }
    # Session state cache (instant switching)
    st.session_state[f'_scan_cache_{wl_id}'] = scan_data
    # Disk cache (survives restarts)
    disk_cache = _load_scan_cache_file()
    disk_cache[wl_id] = scan_data
    _save_scan_cache_file(disk_cache)

def load_scan_for_watchlist(wl_id: str):
    """Load scan results for a watchlist into session state."""
    # Try session state cache first (fast)
    cached = st.session_state.get(f'_scan_cache_{wl_id}')
    if not cached:
        # Try disk cache (restart recovery)
        disk_cache = _load_scan_cache_file()
        cached = disk_cache.get(wl_id)

    if cached and cached.get('results'):
        st.session_state['scan_results'] = []  # No live TickerAnalysis objects
        st.session_state['scan_results_summary'] = cached['results']
        st.session_state['scan_timestamp'] = cached.get('timestamp', '')
    else:
        st.session_state['scan_results'] = []
        st.session_state['scan_results_summary'] = []
        st.session_state['scan_timestamp'] = ''

# Initialize scan results — restore from per-watchlist disk cache
if 'scan_results' not in st.session_state:
    _active_wl_init = st.session_state['watchlist_bridge'].manager.get_active_watchlist()
    load_scan_for_watchlist(_active_wl_init["id"])

# Fetch sector rotation on startup if not already loaded (critical for sector colors)
if 'sector_rotation' not in st.session_state:
    try:
        from data_fetcher import fetch_sector_rotation
        st.session_state['sector_rotation'] = fetch_sector_rotation()
        st.session_state['_sector_rotation_ts'] = time.time()
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

def get_bridge() -> WatchlistBridge:
    return st.session_state['watchlist_bridge']


if 'trade_finder_results' not in st.session_state:
    try:
        _tf_snapshot_payload = get_journal().load_trade_finder_snapshot() or {}
        _tf_latest = _tf_snapshot_payload.get('latest', {}) if isinstance(_tf_snapshot_payload, dict) else {}
        if isinstance(_tf_latest, dict) and _tf_latest:
            _rows = _tf_latest.get('rows', []) or []
            st.session_state['trade_finder_results'] = {
                'run_id': _tf_latest.get('run_id', ''),
                'generated_at_iso': _tf_latest.get('generated_at_iso', ''),
                'rows': _rows if isinstance(_rows, list) else [],
                'provider': _tf_latest.get('provider', 'system'),
                'elapsed_sec': float(_tf_latest.get('elapsed_sec', 0) or 0),
                'input_candidates': int(_tf_latest.get('input_candidates', 0) or 0),
            }
            _find_new = _tf_latest.get('find_new_report', {}) or {}
            if isinstance(_find_new, dict) and _find_new:
                st.session_state['find_new_trades_report'] = _find_new
    except Exception:
        pass


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
        if st.button("🔄 Refresh Analysis", width="stretch",
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

            ai_clients = _get_ai_clients()
            gemini_model = ai_clients.get('gemini')
            openai_client = ai_clients.get('openai_client')
            ai_cfg = ai_clients.get('ai_config', {}) or {}
            ai_model = ai_cfg.get('model', 'llama-3.3-70b-versatile')
            fallback_model = ai_cfg.get('fallback_model', '')

            result = generate_deep_market_analysis(
                macro_data,
                market_filter=market_filter,
                sector_rotation=sector_rotation,
                gemini_model=gemini_model,
                openai_client=openai_client,
                ai_model=ai_model,
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
        st.session_state['_market_filter_ts'] = time.time()
    mkt = st.session_state['market_filter_data']
    _sector_ctx = st.session_state.get('sector_rotation', {}) or {}
    try:
        _regime_u, _reg_conf = _infer_exec_regime(mkt, _sector_ctx)
    except Exception:
        _regime_u, _reg_conf = "UNKNOWN", 0
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
    st.sidebar.caption(f"Unified Regime: **{_regime_u}** ({_reg_conf}%)")

    # Single execution authority: should we be trading at all?
    try:
        gate = _evaluate_trade_gate(_build_dashboard_snapshot())
        if gate.severity == "danger":
            st.sidebar.error(f"**{gate.label}**")
        elif gate.severity == "warning":
            st.sidebar.warning(f"**{gate.label}**")
        else:
            st.sidebar.success(f"**{gate.label}**")
        st.sidebar.caption(f"Reason: {gate.reason}")
        st.sidebar.caption(f"Model alignment: {gate.model_alignment}")
    except Exception:
        pass

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
        if st.sidebar.button("🔄 Refresh Brief", width="stretch", key="refresh_brief"):
            _run_factual_brief()
    else:
        if st.sidebar.button("📊 Generate Market Brief", width="stretch", key="gen_brief"):
            _run_factual_brief()


def _run_factual_brief():
    """Generate the factual morning brief."""
    with st.spinner("Generating market brief..."):
        try:
            from data_fetcher import fetch_macro_narrative_data
            from ai_analysis import generate_market_narrative

            macro_data = fetch_macro_narrative_data()
            ai_clients = _get_ai_clients()
            gemini_model = ai_clients.get('gemini')
            openai_client = ai_clients.get('openai_client')

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


def _compute_system_health(force: bool = False) -> Dict[str, Any]:
    """Compute cached system health snapshot (AI + price feed + heartbeat)."""
    now = time.time()
    ttl_sec = 60
    cached = st.session_state.get('_system_health_snapshot')
    if cached and not force and (now - float(cached.get('ts_epoch', 0.0))) < ttl_sec:
        return cached

    ai_clients = _get_ai_clients()
    ai_cfg = ai_clients.get('ai_config', {}) or {}
    has_openai_compat = ai_clients.get('openai_client') is not None
    has_gemini = ai_clients.get('gemini') is not None
    ai_key = str(ai_cfg.get('key', '') or '')
    ai_validation_ok = False
    if ai_key:
        ai_validation_ok = bool(st.session_state.get(f"_ai_validated_{ai_key[:8]}"))
    ai_ok = bool((has_openai_compat and ai_validation_ok) or has_gemini)

    spy_px = fetch_current_price("SPY")
    price_feed_ok = bool(spy_px is not None and float(spy_px) > 0)

    ts_local = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ts_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open = (
        now_et.weekday() < 5
        and ((now_et.hour > 9) or (now_et.hour == 9 and now_et.minute >= 30))
        and (now_et.hour < 16)
    )
    market_session = "OPEN" if market_open else "CLOSED"
    market_et_time = now_et.strftime('%Y-%m-%d %H:%M:%S ET')

    issues = []
    if not ai_ok:
        issues.append("AI connection unavailable")
    if not price_feed_ok:
        issues.append("Price feed unavailable")

    overall_ok = len(issues) == 0
    snapshot = {
        'ts_epoch': now,
        'ts_local': ts_local,
        'ts_utc': ts_utc,
        'overall_ok': overall_ok,
        'issues': issues,
        'ai_ok': ai_ok,
        'ai_validation_ok': ai_validation_ok,
        'provider': ai_cfg.get('provider', 'none'),
        'model': ai_cfg.get('model', ''),
        'has_openai_compat': has_openai_compat,
        'has_gemini': has_gemini,
        'primary_error': ai_clients.get('primary_error'),
        'gemini_error': ai_clients.get('gemini_error'),
        'price_feed_ok': price_feed_ok,
        'spy_price': float(spy_px) if price_feed_ok else None,
        'market_session': market_session,
        'market_et_time': market_et_time,
    }
    st.session_state['_system_health_snapshot'] = snapshot

    last_log = float(st.session_state.get('_last_system_health_log_ts', 0.0) or 0.0)
    if force or (now - last_log) > 300:
        st.session_state['_last_system_health_log_ts'] = now
        _append_perf_metric({
            "kind": "system_health",
            "ok": overall_ok,
            "ai_ok": ai_ok,
            "price_feed_ok": price_feed_ok,
            "provider": snapshot['provider'],
            "model": snapshot['model'],
        })

    return snapshot


def _render_system_status_panel():
    """Persistent sidebar system status panel with clear health indicator."""
    st.sidebar.divider()
    health = _compute_system_health(force=False)

    with st.sidebar.container(border=True):
        if health.get('overall_ok'):
            st.sidebar.success("🟢 System Status: ALL SYSTEMS OPERATIONAL")
        else:
            st.sidebar.error("🔴 System Status: DEGRADED")
            st.sidebar.error("Alert: " + "; ".join(health.get('issues', [])))
        prev_ok = st.session_state.get('_system_health_last_ok')
        now_ok = bool(health.get('overall_ok'))
        if prev_ok is None:
            st.session_state['_system_health_last_ok'] = now_ok
        elif prev_ok != now_ok:
            if now_ok:
                st.toast("🟢 System recovered: feeds and AI are operational.")
            else:
                st.toast("🔴 System degraded: check AI/feed status.")
            st.session_state['_system_health_last_ok'] = now_ok

        c1, c2 = st.sidebar.columns([2, 1])
        with c1:
            st.sidebar.caption(
                f"AI: {'OK' if health.get('ai_ok') else 'FAIL'} "
                f"({health.get('provider', 'none')} / {health.get('model', '') or 'n/a'})"
            )
            st.sidebar.caption(
                f"AI validation: {'tested' if health.get('ai_validation_ok') else 'not tested'}"
            )
            st.sidebar.caption(
                f"Price Feed: {'OK' if health.get('price_feed_ok') else 'FAIL'} "
                + (f"(SPY ${health.get('spy_price', 0):.2f})" if health.get('price_feed_ok') else "")
            )
            st.sidebar.caption(
                f"Market: {health.get('market_session', 'n/a')} | {health.get('market_et_time', '')}"
            )
            st.sidebar.caption(f"Heartbeat: {health.get('ts_local', '')} | {health.get('ts_utc', '')}")
        with c2:
            if st.sidebar.button("↻ Check", key="system_health_recheck"):
                _compute_system_health(force=True)
                st.rerun()


# =============================================================================
# SIDEBAR — Slim: Scan controls, Open Positions, Alerts, Market
# =============================================================================

def render_sidebar():
    jm = get_journal()
    bridge = get_bridge()

    st.sidebar.title("📊 TTA v2")
    st.sidebar.caption("Technical Trading Assistant")

    # ══════════════════════════════════════════════════════════════════
    # PART B: Deep Structural Analysis ("Juan's Market Filter")
    # ══════════════════════════════════════════════════════════════════
    _render_morning_briefing()

    st.sidebar.divider()

    # ── Watchlist Selector ─────────────────────────────────────────────
    _wm = st.session_state['watchlist_bridge'].manager
    all_wls = _wm.get_all_watchlists()
    active_wl = _wm.get_active_watchlist()

    if len(all_wls) > 1:
        # Multiple watchlists — show selector
        st.sidebar.caption("**Active Watchlist**")
        wl_labels = []
        wl_id_map = {}
        for wl in all_wls:
            icon = "🔒" if wl.get("is_system") else ("🔄" if wl.get("type") == "auto" else "✏️")
            label = f"{icon} {wl['name']} ({len(wl.get('tickers', []))})"
            wl_labels.append(label)
            wl_id_map[label] = wl["id"]

        active_label = None
        for label, wl_id in wl_id_map.items():
            if wl_id == active_wl["id"]:
                active_label = label
                break
        current_idx = wl_labels.index(active_label) if active_label in wl_labels else 0

        sel_col1, sel_col2 = st.sidebar.columns([4, 1])
        with sel_col1:
            selected = st.selectbox(
                "Watchlist", wl_labels, index=current_idx,
                key="sidebar_wl_selector", label_visibility="collapsed",
            )
        with sel_col2:
            if st.button("➕", key="sidebar_wl_create", help="Create new watchlist"):
                st.session_state['show_wl_create'] = True
                st.rerun()

        selected_id = wl_id_map.get(selected)
        if selected_id and selected_id != active_wl["id"]:
            # Save current watchlist's scan results before switching
            save_scan_for_watchlist(active_wl["id"])
            # Switch active watchlist
            _wm.set_active_watchlist(selected_id)
            # Load target watchlist's scan results
            load_scan_for_watchlist(selected_id)
            st.session_state['ticker_data_cache'] = {}
            st.session_state['wl_version'] = st.session_state.get('wl_version', 0) + 1
            st.rerun()
    else:
        # Single watchlist — just show create button
        wl_col1, wl_col2 = st.sidebar.columns([3, 1])
        with wl_col1:
            st.sidebar.caption(f"📋 {active_wl['name']}")
        with wl_col2:
            if st.button("➕", key="sidebar_wl_create", help="Create new watchlist"):
                st.session_state['show_wl_create'] = True
                st.rerun()

    # ── Create Watchlist Dialog (inline sidebar) ───────────────────────
    if st.session_state.get('show_wl_create'):
        from scraping_bridge import ETF_SHORTCUTS
        with st.sidebar.container(border=True):
            st.markdown("**New Watchlist**")
            wl_name = st.text_input("Name", placeholder="e.g. ARKK Holdings", key="create_wl_name")
            wl_type = st.radio("Type", ["Manual", "Auto (ETF)"], horizontal=True, key="create_wl_type")

            source_type = None
            source = None
            if wl_type == "Auto (ETF)":
                shortcuts = list(ETF_SHORTCUTS.keys())
                choice = st.selectbox("Select ETF", shortcuts, key="create_wl_etf")
                source_type = "etf_shortcut"
                source = ETF_SHORTCUTS.get(choice, "")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create", type="primary", key="create_wl_submit", width="stretch"):
                    # Auto-fill name from ETF selection if left blank
                    final_name = wl_name.strip() if wl_name and wl_name.strip() else ""
                    if not final_name and wl_type == "Auto (ETF)" and choice:
                        final_name = choice.split(" - ")[0].strip()  # e.g. "ARKK"
                    
                    if final_name:
                        try:
                            actual_type = "manual" if wl_type == "Manual" else "auto"
                            new_id = _wm.create_watchlist(
                                name=final_name, wl_type=actual_type,
                                source_type=source_type, source=source,
                            )
                            if new_id:
                                # Save current watchlist's scan before switching
                                save_scan_for_watchlist(active_wl["id"])
                                _wm.set_active_watchlist(new_id)
                                st.session_state['wl_version'] = st.session_state.get('wl_version', 0) + 1  # FIX: Force text_area refresh
                                st.session_state['show_wl_create'] = False
                                st.session_state['scan_results'] = []
                                st.session_state['scan_results_summary'] = []
                                st.session_state['scan_timestamp'] = ''

                                # Auto-fetch tickers for auto watchlists
                                if actual_type == "auto" and source_type:
                                    from scraping_bridge import ScrapingBridge
                                    _scraper = st.session_state.get('scraping_bridge')
                                    if not _scraper:
                                        _scraper = ScrapingBridge()
                                        st.session_state['scraping_bridge'] = _scraper
                                    new_wl = _wm.get_watchlist(new_id)
                                    if new_wl:
                                        _wm.record_refresh_request(new_id)
                                        success, msg, tickers = _scraper.fetch_tickers(new_wl)
                                        if success and tickers:
                                            _wm.update_tickers(new_id, tickers, backup_old=False)
                                            _wm.record_refresh_success(new_id)
                                            st.toast(f"✅ Created '{final_name}' with {len(tickers)} tickers")
                                        else:
                                            _wm.record_refresh_failure(new_id, msg)
                                            st.toast(f"⚠️ Created '{final_name}' but fetch failed: {msg}")
                                else:
                                    st.toast(f"✅ Created '{final_name}'")

                                _append_audit_event("CREATE_WATCHLIST", f"name={final_name} type={actual_type}", source="sidebar_create")
                                st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.error("Enter a name")
            with c2:
                if st.button("Cancel", key="create_wl_cancel", width="stretch"):
                    st.session_state['show_wl_create'] = False
                    st.rerun()

    # ── Auto-Refresh Controls (for auto watchlists) ────────────────────
    if active_wl.get("type") == "auto":
        from scraping_bridge import ScrapingBridge, ETF_CSV_URLS

        # Show source info
        src = active_wl.get("source", "")
        url_override = active_wl.get("url_override", "")
        if active_wl.get("source_type") == "etf_shortcut" and src:
            configured_url = ETF_CSV_URLS.get(src.lower(), "")
            display_url = url_override or configured_url
            st.sidebar.caption(f"Source: ARK ETF `{src.upper()}` (auto-populated)")
            if url_override:
                st.sidebar.caption(f"🔗 Using custom URL override")

        # Show last error if any
        stats = active_wl.get("scraping_stats", {})
        last_error = stats.get("last_error")
        if last_error:
            st.sidebar.warning(f"⚠️ Last fetch: {last_error[:120]}")

        # Skip cooldown if last attempt failed (let user retry immediately)
        can_refresh, remaining = _wm.can_refresh_auto(active_wl["id"])
        if last_error:
            can_refresh = True  # Allow retry after failure

        if can_refresh:
            if st.sidebar.button("🔄 Refresh Tickers", key="sidebar_refresh_wl"):
                _wm.record_refresh_request(active_wl["id"])
                try:
                    _bridge_scraper = st.session_state.get('scraping_bridge')
                    if not _bridge_scraper:
                        _bridge_scraper = ScrapingBridge()
                        st.session_state['scraping_bridge'] = _bridge_scraper
                    with st.sidebar:
                        with st.spinner("Validating URL and fetching tickers..."):
                            success, msg, tickers = _bridge_scraper.fetch_tickers(active_wl)
                    if success and tickers:
                        ok, update_msg, cleaned = _wm.update_tickers(active_wl["id"], tickers, backup_old=True)
                        if ok:
                            _wm.record_refresh_success(active_wl["id"])
                            st.sidebar.success(f"✅ {update_msg}")
                        else:
                            _wm.record_refresh_failure(active_wl["id"], update_msg)
                            st.sidebar.error(update_msg)
                    else:
                        _wm.record_refresh_failure(active_wl["id"], msg)
                        st.sidebar.error(f"✗ {msg}")
                except Exception as e:
                    _wm.record_refresh_failure(active_wl["id"], str(e))
                    st.sidebar.error(f"Fetch failed: {str(e)[:100]}")
                st.rerun()
        else:
            st.sidebar.caption(f"⏳ Retry in {remaining}s")

        # ── URL Fix / CSV Upload Fallback (shown when last fetch failed) ──
        if last_error and ("URL" in last_error or "404" in last_error or "403" in last_error
                          or "unreachable" in last_error or "timed out" in last_error):
            with st.sidebar.expander("🔧 Fix Source", expanded=True):
                st.caption("The auto-fetch URL may have changed. Choose a fix:")

                # Show current URL for reference
                _src_key = active_wl.get("source", "").lower()
                _current_url = url_override or ETF_CSV_URLS.get(_src_key, "unknown")
                st.code(_current_url, language=None)

                # Option 1: Custom URL override
                st.markdown("**Option 1: Paste corrected URL**")
                new_url = st.text_input(
                    "CSV URL", value=url_override or "",
                    placeholder="https://assets.ark-funds.com/...",
                    key="url_override_input", label_visibility="collapsed",
                )
                if new_url and new_url.strip() != url_override:
                    if st.button("✅ Save URL & Fetch", key="save_url_override"):
                        _bridge_scraper = st.session_state.get('scraping_bridge')
                        if not _bridge_scraper:
                            _bridge_scraper = ScrapingBridge()
                            st.session_state['scraping_bridge'] = _bridge_scraper
                        # Validate first
                        ok, check_msg, status = _bridge_scraper.validate_url(new_url.strip())
                        if ok:
                            _wm.update_watchlist_metadata(active_wl["id"], url_override=new_url.strip())
                            # Fetch with override
                            active_wl["url_override"] = new_url.strip()
                            success, msg, tickers = _bridge_scraper.fetch_tickers(active_wl)
                            if success and tickers:
                                _wm.update_tickers(active_wl["id"], tickers, backup_old=True)
                                _wm.record_refresh_success(active_wl["id"])
                                st.toast(f"✅ Fetched {len(tickers)} tickers from new URL")
                            else:
                                st.error(f"URL reachable but no tickers found: {msg}")
                        else:
                            st.error(f"❌ {check_msg}")
                        st.rerun()

                # Clear override if one exists
                if url_override:
                    if st.button("🔄 Reset to default URL", key="clear_url_override"):
                        _wm.update_watchlist_metadata(active_wl["id"], url_override="")
                        st.toast("Reset to default URL")
                        st.rerun()

                st.divider()

                # Option 2: Upload CSV file
                st.markdown("**Option 2: Upload CSV file**")
                st.caption("Download holdings CSV from ark-funds.com, then upload here.")
                csv_file = st.file_uploader(
                    "Upload CSV", type=["csv"], key="csv_fallback_upload",
                    label_visibility="collapsed",
                )
                if csv_file is not None:
                    _bridge_scraper = st.session_state.get('scraping_bridge')
                    if not _bridge_scraper:
                        _bridge_scraper = ScrapingBridge()
                        st.session_state['scraping_bridge'] = _bridge_scraper
                    csv_bytes = csv_file.read()
                    success, msg, tickers = _bridge_scraper.fetch_from_csv_content(csv_bytes)
                    if success and tickers:
                        _wm.update_tickers(active_wl["id"], tickers, backup_old=True)
                        _wm.record_refresh_success(active_wl["id"])
                        st.toast(f"✅ Loaded {len(tickers)} tickers from CSV")
                        st.rerun()
                    else:
                        st.error(f"CSV parse failed: {msg}")

        # Rollback option
        backup = active_wl.get("last_backup")
        if backup and backup.get("tickers"):
            if st.sidebar.button(f"↩️ Rollback ({len(backup['tickers'])} tickers)", key="sidebar_rollback_wl"):
                _wm.rollback_tickers(active_wl["id"])
                st.rerun()

    # ── Manage Watchlist (rename, delete) ─────────────────────────────
    if not active_wl.get("is_system"):
        with st.sidebar.expander("⚙️ Manage Watchlist", expanded=False):
            # Rename
            new_name = st.text_input("Rename", value=active_wl["name"], key="wl_rename_input")
            if new_name and new_name.strip() != active_wl["name"]:
                if st.button("✏️ Rename", key="wl_rename_btn"):
                    _wm.update_watchlist_metadata(active_wl["id"], name=new_name.strip())
                    st.toast(f"✅ Renamed to '{new_name.strip()}'")
                    st.rerun()

            st.divider()

            # Delete
            st.caption(f"⚠️ Delete **{active_wl['name']}** permanently")
            if st.button("🗑️ Delete This Watchlist", key="wl_delete_sidebar",
                         type="secondary", width="stretch"):
                st.session_state['confirm_delete_wl'] = True
                st.rerun()
            if st.session_state.get('confirm_delete_wl'):
                st.warning("Are you sure? This cannot be undone.")
                d1, d2 = st.columns(2)
                with d1:
                    if st.button("Yes, Delete", key="wl_confirm_del", type="primary"):
                        ok, msg = _wm.delete_watchlist(active_wl["id"])
                        if ok:
                            _append_audit_event("DELETE_WATCHLIST", msg, source="sidebar_delete")
                            st.session_state['confirm_delete_wl'] = False
                            st.session_state['scan_results'] = []
                            st.session_state['scan_results_summary'] = []
                            st.session_state['scan_timestamp'] = ''
                            st.session_state['ticker_data_cache'] = {}
                            st.toast(f"✅ {msg}")
                            st.rerun()
                with d2:
                    if st.button("Cancel", key="wl_cancel_del"):
                        st.session_state['confirm_delete_wl'] = False
                        st.rerun()

    # ── Scan Controls ─────────────────────────────────────────────────
    watchlist_tickers = bridge.get_watchlist_tickers()
    ticker_count = len(watchlist_tickers)

    # Count unscanned
    existing_summary = st.session_state.get('scan_results_summary', [])
    scanned = {s.get('ticker', '') for s in existing_summary}
    new_count = len([t for t in watchlist_tickers if t not in scanned])

    st.sidebar.caption(f"📋 **{active_wl['name']}** — {ticker_count} tickers")

    scan_col1, scan_col2 = st.sidebar.columns(2)
    with scan_col1:
        if st.button("🔍 Scan All", width="stretch", type="primary",
                     disabled=(ticker_count == 0)):
            _run_scan(mode='all')
    with scan_col2:
        btn_label = f"⚡ New ({new_count})" if new_count > 0 else "⚡ New"
        if st.button(btn_label, width="stretch",
                     disabled=(new_count == 0)):
            _run_scan(mode='new_only')

    all_watchlists = bridge.manager.get_all_watchlists() or []
    all_watch_tickers = set()
    for _wl in all_watchlists:
        for _t in (_wl.get('tickers', []) or []):
            if isinstance(_t, str) and _t.strip():
                all_watch_tickers.add(_t.upper().strip())
    if st.sidebar.button(
        f"🧭 Find New Trades ({len(all_watch_tickers)})",
        width="stretch",
        help="Scan ALL watchlists and return a ranked list of new trade candidates.",
        disabled=(len(all_watch_tickers) == 0),
    ):
        _run_find_new_trades()

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
                width="stretch",
            ):
                _load_ticker_for_view(ticker)

    # ── Market Brief (replaces old green/yellow/red rectangle) ──────
    st.sidebar.divider()
    _render_factual_market_brief()
    _render_system_status_panel()

    # ── Settings (bottom of sidebar) ──────────────────────────────────
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        current_beta = bool(st.session_state.get('exec_dashboard_beta_enabled', _is_exec_dashboard_enabled()))
        beta_enabled = st.checkbox(
            "Executive Dashboard Beta",
            value=current_beta,
            key="exec_dashboard_beta_toggle",
            help="Toggle phased rollout of Executive Dashboard tab.",
        )
        st.session_state['exec_dashboard_beta_enabled'] = beta_enabled

        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("🔑 Reset API", key="reset_api_cache",
                         help="Clear cached API key status after updating secrets"):
                for k in list(st.session_state.keys()):
                    if k.startswith(('_groq_key', '_groq_validated', '_ai_validated', '_ai_clients',
                                     'ai_result_', 'chat_')):
                        st.session_state.pop(k, None)
                st.session_state.pop('_ai_config', None)
                st.toast("✅ API cache cleared. Click a ticker to re-run.")
        with s2:
            if st.button("📊 Refresh Mkt", key="refresh_market_data",
                         help="Re-fetch SPY, VIX, sector rotation"):
                st.session_state.pop('market_filter_data', None)
                st.session_state.pop('_market_filter_ts', None)
                st.session_state.pop('sector_rotation', None)
                st.session_state.pop('_sector_rotation_ts', None)
                st.session_state.pop('_research_market_cache', None)
                st.session_state.pop('_research_market_cache_ts', None)
                st.session_state.pop('_position_prices', None)
                st.session_state.pop('_position_prices_ts', None)
                # Clear APEX caches (SPY/VIX data + per-ticker detection)
                st.session_state.pop('apex_spy_data', None)
                st.session_state.pop('apex_vix_data', None)
                for k in [k for k in list(st.session_state.keys()) if k.startswith('_apex_cache_')]:
                    st.session_state.pop(k, None)
                st.toast("✅ Market data will refresh on next load.")
        with s3:
            if st.button("☁ Restore WL", key="restore_watchlists_from_backup",
                         help="Force-restore watchlist files from GitHub backup branch."):
                try:
                    import os
                    import github_backup
                    # Force restore by removing local watchlist files first.
                    for fn in ["v2_multi_watchlist.json", "v2_watchlist.json"]:
                        try:
                            if os.path.exists(fn):
                                os.remove(fn)
                        except Exception:
                            pass
                    restored = github_backup.restore_all() or {}

                    # Reinitialize watchlist manager/bridge in current session.
                    _wm = WatchlistManager()
                    st.session_state['watchlist_bridge'] = WatchlistBridge(_wm, get_journal())
                    _active = _wm.get_active_watchlist()
                    if _active and _active.get("id"):
                        load_scan_for_watchlist(_active["id"])

                    restored_count = sum(1 for v in restored.values() if v)
                    st.toast(f"✅ Restore complete ({restored_count} file(s) restored).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Restore failed: {str(e)[:180]}")

        # Backup health (watchlist persistence confidence)
        try:
            _b = get_backup_health_status()
            if not _b.get("available", False):
                raise RuntimeError(str(_b.get("last_error", "backup status unavailable")))
            _enabled = bool(_b.get("enabled", False))
            _pending = int(_b.get("pending_count", 0) or 0)
            _last_ok = float(_b.get("last_success_epoch", 0.0) or 0.0)
            _branch = str(_b.get("branch", "data-backup") or "data-backup")
            _berr = str(_b.get("last_error", "") or "")
            _berr_code = int(_b.get("last_error_code", 0) or 0)
            _last_txt = datetime.fromtimestamp(_last_ok).strftime('%Y-%m-%d %H:%M:%S') if _last_ok > 0 else "never"
            st.caption(
                f"☁ Backup: {'enabled' if _enabled else 'disabled'} | "
                f"Branch: {_branch} | Pending: {_pending} | Last success: {_last_txt}"
            )
            if _enabled and _berr:
                code_txt = f"{_berr_code} " if _berr_code else ""
                st.caption(f"☁ Backup error: {code_txt}{_berr[:140]}")
            if _enabled:
                if st.button("☁ Backup Now", key="force_backup_now", width="stretch"):
                    try:
                        pushed = int(run_backup_now() or 0)
                        st.toast(f"✅ Backup complete ({pushed} file(s) pushed).")
                        st.rerun()
                    except Exception as _be:
                        st.error(f"Backup flush failed: {str(_be)[:180]}")
        except Exception:
            st.caption("☁ Backup status unavailable")

        if st.button("🧪 Run System Self-Test", key="run_system_self_test", width="stretch"):
            _health = st.session_state.get('_system_health_snapshot') or _compute_system_health(force=True)
            _backup = get_backup_health_status()
            _report = run_system_self_test(_health, _backup)
            st.session_state['_latest_system_self_test'] = _report
            if _report.get('overall_ok'):
                st.toast("🟢 System self-test passed.")
            else:
                st.toast(f"🔴 System self-test failed ({len(_report.get('failures', []))} issues).")
            st.rerun()
        _last_self_test = st.session_state.get('_latest_system_self_test') or {}
        if _last_self_test:
            _summary = str(_last_self_test.get('summary', '') or '').strip()
            st.caption(f"🧪 Self-test: {_summary}")
            for _f in (_last_self_test.get('failures', []) or [])[:2]:
                st.caption(f"• {str(_f.get('name', 'check'))}: {str(_f.get('detail', ''))[:110]}")

        # Key diagnostic
        try:
            _diag_key = st.secrets.get("GROQ_API_KEY", "")
            if _diag_key:
                _diag_cfg = _detect_ai_provider(_diag_key)
                _diag_clients = _get_ai_clients()
                _validated = bool(st.session_state.get(f"_ai_validated_{_diag_cfg['key'][:8]}")) if _diag_cfg.get('key') else False
                if _diag_clients.get('openai_client') is not None:
                    _status = 'connected/tested' if _validated else 'connected/not-tested'
                elif _diag_clients.get('primary_error'):
                    _status = f"error: {_diag_clients.get('primary_error')}"
                else:
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


def _resolve_tickers_to_scan(full_list: List[str], existing_summary: List[Dict], mode: str) -> List[str]:
    """Deterministic scan universe resolver used by scan-all and scan-new paths."""
    return resolve_tickers_to_scan(full_list, existing_summary, mode)


def _run_scan(mode='all'):
    """
    Execute watchlist scan with persistence and conditional checking.

    mode='all': Rescan everything (daily refresh, full watchlist)
    mode='new_only': Only scan tickers not in current results
    """
    jm = get_journal()
    bridge = get_bridge()
    all_watchlist = bridge.get_watchlist_tickers()

    # Scan universe is strictly the active watchlist.
    # This avoids re-introducing stale/non-watchlist tickers into scanner results.
    open_tickers = jm.get_open_tickers()
    full_list = list(set(all_watchlist))
    
    # ── Validate tickers — reject corrupt entries before they hit yfinance ──
    import re as _re
    valid_list = []
    rejected = []
    for t in full_list:
        # Valid ticker: 1-5 uppercase alpha, class shares (e.g. BRK.B), or indices (^VIX).
        if not isinstance(t, str):
            rejected.append(t)
            continue
        t = t.upper().strip()

        if _re.match(r'^[A-Z]{1,5}$', t) or t.startswith('^'):
            valid_list.append(t)
            continue

        class_match = _re.match(r'^[A-Z]{1,5}\.([A-Z])$', t)
        if class_match and class_match.group(1) in {'A', 'B', 'C', 'D'}:
            valid_list.append(t)
        else:
            rejected.append(t)
    if rejected:
        print(f"[scan] Rejected {len(rejected)} invalid tickers: {rejected[:10]}")
    full_list = valid_list

    if not full_list:
        _append_audit_event("SCAN_SKIPPED", "empty watchlist", source=f"scan:{mode}")
        st.sidebar.warning("Add tickers to watchlist first")
        return

    # Determine which tickers to scan
    existing_summary = st.session_state.get('scan_results_summary', [])
    tickers_to_scan = _resolve_tickers_to_scan(full_list, existing_summary, mode)
    if mode == 'new_only' and not tickers_to_scan:
        st.sidebar.info("All tickers already scanned")
        return

    _scan_started = time.time()
    with st.spinner(f"Scanning {len(tickers_to_scan)} tickers..."):
        _timing = {
            'mode': mode,
            'universe': len(full_list),
            'to_scan': len(tickers_to_scan),
            'fetch_scan_data_sec': 0.0,
            'scan_watchlist_sec': 0.0,
            'sector_refresh_sec': 0.0,
            'earnings_refresh_sec': 0.0,
            'sector_assign_sec': 0.0,
            'summary_build_sec': 0.0,
            'alerts_check_sec': 0.0,
            'total_sec': 0.0,
        }
        _append_audit_event(
            "SCAN_START",
            f"mode={mode} universe={len(full_list)} scan_count={len(tickers_to_scan)}",
            source=f"scan:{mode}",
        )
        _t = time.time()
        all_data = fetch_scan_data(tickers_to_scan)
        _timing['fetch_scan_data_sec'] = time.time() - _t

        _t = time.time()
        new_results = scan_watchlist(all_data)
        _timing['scan_watchlist_sec'] = time.time() - _t

        # Defaults keep scan flow resilient if auxiliary fetches fail.
        sector_rotation = st.session_state.get('sector_rotation', {}) or {}
        earnings_flags = st.session_state.get('earnings_flags', {}) or {}

        # Fetch sector rotation (independent — failure doesn't block earnings)
        try:
            _t = time.time()
            from data_fetcher import fetch_sector_rotation
            sector_rotation = fetch_sector_rotation()
            st.session_state['sector_rotation'] = sector_rotation
            st.session_state['_sector_rotation_ts'] = time.time()
            _timing['sector_refresh_sec'] = time.time() - _t
        except Exception as e:
            print(f"Sector rotation error: {e}")

        # Fetch earnings flags (independent — failure doesn't block sectors)
        try:
            _t = time.time()
            from data_fetcher import fetch_batch_earnings_flags
            all_scan_tickers = [r.ticker for r in new_results]
            earnings_flags = fetch_batch_earnings_flags(all_scan_tickers, days_ahead=60)
            if earnings_flags:  # Only update if we got results
                if mode == 'new_only':
                    existing_flags = st.session_state.get('earnings_flags', {})
                    existing_flags.update(earnings_flags)
                    st.session_state['earnings_flags'] = existing_flags
                else:
                    st.session_state['earnings_flags'] = earnings_flags
            _timing['earnings_refresh_sec'] = time.time() - _t
        except Exception as e:
            print(f"Earnings flags error: {e}")

        # Fetch sectors for scanned tickers (independent)
        try:
            _t = time.time()
            from data_fetcher import get_ticker_sector
            ticker_sectors = st.session_state.get('ticker_sectors', {})
            for r in new_results:
                if r.ticker not in ticker_sectors:
                    sector = get_ticker_sector(r.ticker)
                    if sector:
                        ticker_sectors[r.ticker] = sector
            st.session_state['ticker_sectors'] = ticker_sectors
            _timing['sector_assign_sec'] = time.time() - _t
        except Exception as e:
            print(f"Sector assignment error: {e}")

        # Merge with existing results if new_only mode
        if mode == 'new_only':
            existing_results = st.session_state.get('scan_results', [])
            # Drop stale rows that are no longer in the current scan universe.
            existing_results = [r for r in existing_results if r.ticker in full_list]
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

        _t = time.time()
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
        _timing['summary_build_sec'] = time.time() - _t
        jm.save_scan_results(summary)
        st.session_state['scan_results_summary'] = summary
        st.session_state['scan_timestamp'] = datetime.now().isoformat()
        st.session_state['_scan_run_ts'] = time.time()
        # Save per-watchlist scan cache (for switching + restart)
        _active_wl_id = bridge.manager.get_active_watchlist().get("id", "")
        if _active_wl_id:
            save_scan_for_watchlist(_active_wl_id)

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

        _t = time.time()
        triggered = jm.check_conditionals(current_prices, volume_ratios)
        _timing['alerts_check_sec'] = time.time() - _t
        if triggered:
            st.session_state['triggered_alerts'] = triggered

        _append_audit_event(
            "SCAN_DONE",
            f"mode={mode} scanned={len(tickers_to_scan)} results={len(summary)} triggered={len(triggered)}",
            source=f"scan:{mode}",
        )

        # Telemetry: runtime history + daily workflow finalize
        _scan_elapsed = max(0.0, time.time() - _scan_started)
        _timing['total_sec'] = _scan_elapsed
        st.session_state['_timing_last_scan'] = _timing
        _append_perf_metric({
            "kind": "scan_run",
            "mode": mode,
            "universe": _timing.get("universe", 0),
            "to_scan": _timing.get("to_scan", 0),
            "total_sec": round(_timing.get("total_sec", 0.0), 3),
            "fetch_sec": round(_timing.get("fetch_scan_data_sec", 0.0), 3),
            "analyze_sec": round(_timing.get("scan_watchlist_sec", 0.0), 3),
            "sectors_sec": round(_timing.get("sector_refresh_sec", 0.0), 3),
            "earnings_sec": round(_timing.get("earnings_refresh_sec", 0.0), 3),
            "alerts_sec": round(_timing.get("alerts_check_sec", 0.0), 3),
        })
        hist = st.session_state.get('_scan_duration_hist', [])
        hist.append(_scan_elapsed)
        st.session_state['_scan_duration_hist'] = hist[-30:]
        st.session_state['_last_scan_duration_sec'] = _scan_elapsed
        st.session_state['_last_scan_mode'] = mode

        if st.session_state.get('_daily_workflow_in_progress'):
            wf_start = float(st.session_state.get('_daily_workflow_start_ts', _scan_started) or _scan_started)
            wf_sec = max(0.0, time.time() - wf_start)
            st.session_state['_daily_workflow_sec'] = wf_sec
            st.session_state['_daily_workflow_ts'] = time.time()
            st.session_state['_daily_workflow_in_progress'] = False
            _append_audit_event(
                "DAILY_WORKFLOW_DONE",
                f"mode={mode} scan_sec={_scan_elapsed:.1f} total_sec={wf_sec:.1f}",
                source=f"scan:{mode}",
            )

    st.rerun()


def _run_find_new_trades():
    """
    Scan the union of ALL watchlists and build a consolidated "new trades" report.
    This does not replace active-watchlist scanner results.
    """
    jm = get_journal()
    bridge = get_bridge()
    all_watchlists = bridge.manager.get_all_watchlists() or []

    # Build ticker -> source watchlists map.
    ticker_sources: Dict[str, List[str]] = {}
    for wl in all_watchlists:
        wl_name = str(wl.get('name', 'Watchlist'))
        for t in (wl.get('tickers', []) or []):
            if not isinstance(t, str):
                continue
            ticker = t.upper().strip()
            if not ticker:
                continue
            ticker_sources.setdefault(ticker, [])
            if wl_name not in ticker_sources[ticker]:
                ticker_sources[ticker].append(wl_name)

    # Validate symbols before data fetch.
    import re as _re
    universe = []
    rejected = []
    for t in sorted(ticker_sources.keys()):
        is_base = bool(_re.match(r'^[A-Z]{1,5}$', t))
        class_match = _re.match(r'^[A-Z]{1,5}\.([A-Z])$', t)
        is_class = bool(class_match and class_match.group(1) in {'A', 'B', 'C', 'D'})
        is_index = t.startswith('^')
        if is_base or is_class or is_index:
            universe.append(t)
        else:
            rejected.append(t)

    if rejected:
        print(f"[find_new] Rejected {len(rejected)} invalid tickers: {rejected[:10]}")

    if not universe:
        st.sidebar.warning("No tickers found across watchlists.")
        return

    _start = time.time()
    with st.spinner(f"Finding new trades across {len(universe)} tickers..."):
        _append_audit_event(
            "FIND_NEW_START",
            f"watchlists={len(all_watchlists)} universe={len(universe)}",
            source="find_new",
        )

        all_data = fetch_scan_data(universe)
        results = scan_watchlist(all_data)

        # Pull current supporting context.
        sector_rotation = st.session_state.get('sector_rotation', {}) or {}
        try:
            from data_fetcher import fetch_sector_rotation
            sector_rotation = fetch_sector_rotation()
            st.session_state['sector_rotation'] = sector_rotation
            st.session_state['_sector_rotation_ts'] = time.time()
        except Exception:
            pass

        earnings_flags = {}
        try:
            from data_fetcher import fetch_batch_earnings_flags
            earnings_flags = fetch_batch_earnings_flags([r.ticker for r in results], days_ahead=60) or {}
        except Exception:
            earnings_flags = {}

        ticker_sectors = st.session_state.get('ticker_sectors', {}) or {}
        try:
            from data_fetcher import get_ticker_sector
            for r in results:
                if r.ticker not in ticker_sectors:
                    sec = get_ticker_sector(r.ticker)
                    if sec:
                        ticker_sectors[r.ticker] = sec
            st.session_state['ticker_sectors'] = ticker_sectors
        except Exception:
            pass

        open_tickers = set(jm.get_open_tickers())
        rows = []
        candidates = []
        for r in results:
            rec = r.recommendation or {}
            q = r.quality or {}
            rec_text = str(rec.get('recommendation', 'SKIP'))
            earn = earnings_flags.get(r.ticker, {}) or {}
            sector_name = ticker_sectors.get(r.ticker, '')
            sector_info = sector_rotation.get(sector_name, {}) if sector_name else {}

            row = {
                'ticker': r.ticker,
                'recommendation': rec_text,
                'conviction': int(rec.get('conviction', 0) or 0),
                'quality_grade': q.get('quality_grade', '?'),
                'price': float(r.current_price or 0),
                'summary': rec.get('summary', ''),
                'sector': sector_name,
                'sector_phase': sector_info.get('phase', ''),
                'earn_date': earn.get('next_earnings', ''),
                'earn_days': int(earn.get('days_until', 999) or 999),
                'is_open_position': r.ticker in open_tickers,
                'watchlists': ", ".join(ticker_sources.get(r.ticker, [])),
                'source_watchlists': ticker_sources.get(r.ticker, []),
            }
            rows.append(row)

            rec_upper = rec_text.upper()
            is_entry = ('BUY' in rec_upper or 'ENTRY' in rec_upper) and 'SKIP' not in rec_upper and 'AVOID' not in rec_upper
            if is_entry and not row['is_open_position']:
                score = _recommendation_score(row)
                candidates.append({
                    'ticker': row['ticker'],
                    'recommendation': row['recommendation'],
                    'conviction': row['conviction'],
                    'quality_grade': row['quality_grade'],
                    'price': row['price'],
                    'summary': row.get('summary', ''),
                    'sector': row['sector'],
                    'sector_phase': row['sector_phase'],
                    'earn_date': row['earn_date'],
                    'earn_days': row['earn_days'],
                    'watchlists': row['watchlists'],
                    'score': round(score, 2),
                })

        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

    elapsed = max(0.0, time.time() - _start)
    report = {
        'generated_at': time.time(),
        'generated_at_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'watchlists_count': len(all_watchlists),
        'scan_universe': len(universe),
        'results_count': len(rows),
        'candidate_count': len(candidates),
        'candidates': candidates,
        'rows': rows,
        'elapsed_sec': elapsed,
    }
    st.session_state['find_new_trades_report'] = report
    st.session_state['_find_new_trades_ts'] = time.time()
    try:
        _existing_rows = (st.session_state.get('trade_finder_results', {}) or {}).get('rows', []) or []
        get_journal().save_trade_finder_snapshot({
            'generated_at_iso': report.get('generated_at_iso', ''),
            'run_id': (st.session_state.get('trade_finder_results', {}) or {}).get('run_id', ''),
            'provider': (st.session_state.get('trade_finder_results', {}) or {}).get('provider', 'system'),
            'elapsed_sec': float(report.get('elapsed_sec', 0) or 0),
            'input_candidates': int(report.get('candidate_count', 0) or 0),
            'rows': _existing_rows,
            'find_new_report': report,
        })
    except Exception:
        pass
    _append_perf_metric({
        'kind': 'find_new_trades',
        'watchlists': len(all_watchlists),
        'universe': len(universe),
        'results': len(rows),
        'candidates': len(candidates),
        'sec': round(elapsed, 3),
    })
    _append_audit_event(
        "FIND_NEW_DONE",
        f"watchlists={len(all_watchlists)} universe={len(universe)} candidates={len(candidates)} sec={elapsed:.1f}",
        source="find_new",
    )
    st.success(f"Found {len(candidates)} candidate(s) from {len(universe)} tickers across {len(all_watchlists)} watchlists.")


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extractor for AI responses."""
    if not raw:
        return {}
    txt = str(raw).strip()
    try:
        return _json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return {}
    try:
        return _json.loads(m.group(0))
    except Exception:
        return {}

def _extract_earn_days_hint(text: str) -> Optional[int]:
    """Best-effort extract of 'earnings in X days' from freeform model text."""
    if not text:
        return None
    m = re.search(r"earnings?\s+(?:in|within)\s+(\d{1,3})\s*days?", str(text).lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _calc_rr(entry: float, stop: float, target: float) -> float:
    """Compute reward/risk ratio with guardrails."""
    if entry <= 0 or stop <= 0 or target <= 0:
        return 0.0
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(reward / risk, 2)


def _extract_rr_from_text(text: str) -> Optional[float]:
    """Extract first R:R-like token (e.g. 1.8:1) from model text."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1\b", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _lookup_company_name_for_trade_finder(ticker: str) -> str:
    """
    Best-effort company-name lookup for Trade Finder rows.

    Uses cached fundamental profile by ticker. Never infers names from ticker string.
    """
    t = str(ticker or '').upper().strip()
    if not t:
        return "Unknown company"

    cache = st.session_state.get('_tf_company_name_cache')
    if not isinstance(cache, dict):
        cache = {}
        st.session_state['_tf_company_name_cache'] = cache
    if t in cache:
        return str(cache.get(t, "Unknown company") or "Unknown company")

    name = ""
    try:
        from data_fetcher import fetch_fundamental_profile
        profile = fetch_fundamental_profile(t) or {}
        name = str(profile.get('name', '') or '').strip()
    except Exception:
        name = ""

    if not name:
        name = "Unknown company"
    cache[t] = name
    st.session_state['_tf_company_name_cache'] = cache
    return name


def _normalize_gold_ai_contract(
    ticker: str,
    row: Dict[str, Any],
    *,
    fallback_ai_buy: str,
    fallback_rank_score: float,
) -> Dict[str, Any]:
    """
    Unified AI recommendation contract.
    Gold source: Ask AI cached analysis (ai_result_<ticker>) when available.
    """
    entry = float(row.get('entry', row.get('price', 0)) or 0)
    stop = float(row.get('stop', 0) or 0)
    target = float(row.get('target', 0) or 0)
    level_rr = _calc_rr(entry, stop, target)

    ai_result = st.session_state.get(f'ai_result_{ticker}', {}) or {}
    action_text = str(ai_result.get('action', ai_result.get('timing', '')) or '').upper()
    raw_text = str(ai_result.get('raw_text', '') or '')
    conv = int(ai_result.get('conviction', 0) or 0)
    pos_size = str(ai_result.get('position_sizing', '') or '').lower()
    provider = str(ai_result.get('provider', 'system') or 'system')

    verdict = fallback_ai_buy if fallback_ai_buy in {"Strong Buy", "Buy", "Watch Only", "Skip"} else "Watch Only"
    if action_text:
        if any(k in action_text for k in ["PASS", "SKIP"]):
            verdict = "Skip"
        elif "HOLD" in action_text:
            verdict = "Watch Only"
        elif "BUY" in action_text:
            verdict = "Strong Buy" if conv >= 8 else "Buy"

    rr_from_text = _extract_rr_from_text(raw_text)
    rr = rr_from_text if rr_from_text and rr_from_text > 0 else level_rr

    size_adj = 0.0
    if "full" in pos_size:
        size_adj = 1.0
    elif "reduced" in pos_size:
        size_adj = 0.4
    elif "small" in pos_size:
        size_adj = 0.15
    elif "skip" in pos_size:
        size_adj = -0.6

    verdict_adj = {"Strong Buy": 1.1, "Buy": 0.5, "Watch Only": -0.2, "Skip": -1.0}.get(verdict, 0.0)
    conv_adj = max(-0.5, min(0.9, (conv - 5) * 0.1))
    rr_adj = max(-0.5, min(1.2, (rr - 1.5) * 0.35))
    unified_score = round(float(fallback_rank_score or 0) + verdict_adj + conv_adj + rr_adj + size_adj, 2)

    note = str(ai_result.get('why_moving', '') or ai_result.get('smart_money', '') or '')[:180]
    return {
        'verdict': verdict,
        'confidence': conv,
        'risk_reward': round(rr, 2),
        'provider': provider,
        'position_sizing': ai_result.get('position_sizing', ''),
        'unified_rank_score': unified_score,
        'note': note,
        'is_gold_source': bool(ai_result),
    }


def _trade_quality_settings() -> Dict[str, Any]:
    """Global quality gates used by Trade Finder, Exec candidates, and New Trade entry checks."""
    if 'trade_min_rr_threshold' not in st.session_state:
        st.session_state['trade_min_rr_threshold'] = 1.2
    if 'trade_earnings_block_days' not in st.session_state:
        st.session_state['trade_earnings_block_days'] = 3
    if 'trade_require_ready' not in st.session_state:
        st.session_state['trade_require_ready'] = False
    if 'trade_include_watch_only' not in st.session_state:
        st.session_state['trade_include_watch_only'] = True
    return {
        'min_rr': float(st.session_state.get('trade_min_rr_threshold', 1.2) or 1.2),
        'earn_block_days': int(st.session_state.get('trade_earnings_block_days', 3) or 3),
        'require_ready': bool(st.session_state.get('trade_require_ready', False)),
        'include_watch_only': bool(st.session_state.get('trade_include_watch_only', True)),
    }


def _trade_candidate_is_qualified(row: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    """Shared candidate quality filter."""
    min_rr = float(settings.get('min_rr', 2.0) or 2.0)
    earn_block_days = int(settings.get('earn_block_days', 7) or 7)
    require_ready = bool(settings.get('require_ready', True))
    include_watch_only = bool(settings.get('include_watch_only', False))

    entry = float(row.get('suggested_entry', row.get('price', 0)) or 0)
    stop = float(row.get('suggested_stop_loss', 0) or 0)
    target = float(row.get('suggested_target', 0) or 0)
    rr = float(row.get('risk_reward', 0) or 0)
    if rr <= 0:
        rr = _calc_rr(entry, stop, target)
    if entry <= 0 or stop <= 0 or target <= 0:
        return False
    if not (stop < entry < target):
        return False
    if rr < min_rr:
        return False
    ai_buy = str(row.get('ai_buy_recommendation', '') or '').strip()
    allowed_ai = {"Strong Buy", "Buy"} | ({"Watch Only"} if include_watch_only else set())
    if ai_buy not in allowed_ai:
        return False

    earn_days = int(row.get('earn_days', 999) or 999)
    if 0 <= earn_days <= earn_block_days:
        return False

    if require_ready:
        card = row.get('decision_card', {}) or {}
        if str(card.get('execution_readiness', '')).upper() != "READY":
            return False
    return True


def _heuristic_trade_finder_ai(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback AI recommendation when Grok is unavailable."""
    price = float(candidate.get('price', 0) or 0)
    conv = int(candidate.get('conviction', 0) or 0)
    rec = str(candidate.get('recommendation', '')).upper()
    summary = str(candidate.get('summary', '') or '')
    quality = str(candidate.get('quality_grade', '?') or '?')

    entry = price
    if conv >= 8:
        stop = round(entry * 0.95, 2) if entry > 0 else 0.0
        target = round(entry * 1.12, 2) if entry > 0 else 0.0
        ai_buy = "Strong Buy"
    elif "BUY" in rec or "ENTRY" in rec:
        stop = round(entry * 0.94, 2) if entry > 0 else 0.0
        target = round(entry * 1.10, 2) if entry > 0 else 0.0
        ai_buy = "Buy"
    else:
        stop = round(entry * 0.93, 2) if entry > 0 else 0.0
        target = round(entry * 1.08, 2) if entry > 0 else 0.0
        ai_buy = "Watch Only"
    rr = _calc_rr(entry, stop, target)
    rank_score = round(conv * 1.5 + rr * 3 + (2 if quality.startswith('A') else 0), 2)
    return {
        'ai_buy_recommendation': ai_buy if entry > 0 else "Skip",
        'suggested_entry': entry,
        'suggested_stop_loss': stop,
        'suggested_target': target,
        'risk_reward': rr,
        'rank_score': rank_score,
        'rationale': summary or "Heuristic fallback used (AI unavailable).",
        'provider': 'system',
    }


def _grok_trade_finder_assessment(candidate: Dict[str, Any], ai_clients: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask Grok/OpenAI-compatible provider for trade-finder scoring and risk levels.
    Returns normalized fields used by the Trade Finder table.
    """
    openai_client = ai_clients.get('openai_client')
    ai_cfg = ai_clients.get('ai_config', {}) or {}
    model = ai_cfg.get('model', 'grok-3-fast')
    fallback_model = ai_cfg.get('fallback_model', '')

    if openai_client is None:
        return _heuristic_trade_finder_ai(candidate)

    payload = {
        "ticker": candidate.get('ticker', ''),
        "recommendation": candidate.get('recommendation', ''),
        "conviction": int(candidate.get('conviction', 0) or 0),
        "quality_grade": candidate.get('quality_grade', '?'),
        "price": float(candidate.get('price', 0) or 0),
        "summary": candidate.get('summary', ''),
        "sector": candidate.get('sector', ''),
        "sector_phase": candidate.get('sector_phase', ''),
        "earn_days": int(candidate.get('earn_days', 999) or 999),
        "watchlists": candidate.get('watchlists', ''),
    }
    prompt = (
        "You are scoring swing-trade candidates. Return ONLY valid JSON with keys: "
        "ai_buy_recommendation, suggested_entry, suggested_stop_loss, suggested_target, "
        "risk_reward, rank_score, rationale. "
        "ai_buy_recommendation must be one of: Strong Buy, Buy, Watch Only, Skip. "
        "Do not change ticker identity. Do not infer or mention company names. "
        "Do not invent earnings timing; use earn_days exactly as provided. "
        "Set stop below entry, target above entry. Keep rationale <= 180 chars.\n\n"
        f"Candidate:\n{_json.dumps(payload)}"
    )

    models = [m for m in [model, fallback_model] if m]
    raw_text = None
    provider = None
    for m in models:
        try:
            resp = openai_client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=260,
                temperature=0.2,
            )
            raw_text = resp.choices[0].message.content
            provider = m
            if raw_text:
                break
        except Exception:
            continue

    if not raw_text:
        out = _heuristic_trade_finder_ai(candidate)
        out['provider'] = 'system'
        return out

    parsed = _extract_first_json_object(raw_text)
    if not parsed:
        out = _heuristic_trade_finder_ai(candidate)
        out['provider'] = provider or 'system'
        return out

    entry = float(parsed.get('suggested_entry', payload['price']) or payload['price'] or 0)
    stop = float(parsed.get('suggested_stop_loss', 0) or 0)
    target = float(parsed.get('suggested_target', 0) or 0)
    if entry > 0 and (stop <= 0 or stop >= entry):
        stop = round(entry * 0.94, 2)
    if entry > 0 and (target <= entry):
        target = round(entry * 1.10, 2)

    rr = float(parsed.get('risk_reward', 0) or 0)
    if rr <= 0:
        rr = _calc_rr(entry, stop, target)
    rank_score = float(parsed.get('rank_score', 0) or 0)
    if rank_score <= 0:
        rank_score = round(payload['conviction'] * 1.5 + rr * 3, 2)

    ai_buy = str(parsed.get('ai_buy_recommendation', '') or '').strip() or "Watch Only"
    if ai_buy not in {"Strong Buy", "Buy", "Watch Only", "Skip"}:
        ai_buy = "Watch Only"
    rationale = str(parsed.get('rationale', '') or '').strip()[:180]
    if not rationale:
        rationale = payload.get('summary') or "AI-ranked candidate."

    return {
        'ai_buy_recommendation': ai_buy,
        'suggested_entry': round(entry, 2),
        'suggested_stop_loss': round(stop, 2),
        'suggested_target': round(target, 2),
        'risk_reward': round(rr, 2),
        'rank_score': round(rank_score, 2),
        'rationale': rationale,
        'provider': provider or 'system',
    }


def _run_trade_finder_workflow() -> None:
    """
    Executes scanner candidate discovery + Grok scoring.
    Reuses existing find-new logic, then enriches/ranks candidates.
    """
    _run_find_new_trades()
    report = st.session_state.get('find_new_trades_report', {}) or {}
    base_candidates = report.get('candidates', []) or []
    if not base_candidates:
        st.session_state['trade_finder_results'] = {
            'generated_at_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rows': [],
            'provider': 'system',
            'elapsed_sec': 0.0,
        }
        return

    ai_clients = _get_ai_clients()
    snap = _build_dashboard_snapshot()
    gate = _evaluate_trade_gate(snap)
    run_id = datetime.now().strftime("TF_%Y%m%d_%H%M%S")
    rows = []
    _t0 = time.time()
    for c in base_candidates:
        ticker = str(c.get('ticker', '')).upper().strip()
        company_name = str(c.get('company_name', '') or '').strip()
        if not company_name:
            company_name = _lookup_company_name_for_trade_finder(ticker)

        ai_rec = _grok_trade_finder_assessment(c, ai_clients)
        entry = float(ai_rec.get('suggested_entry', c.get('price', 0)) or c.get('price', 0) or 0)
        stop = float(ai_rec.get('suggested_stop_loss', 0) or 0)
        target = float(ai_rec.get('suggested_target', 0) or 0)
        ai_rr = float(ai_rec.get('risk_reward', 0) or 0)
        # Canonical R:R is always derived from entry/stop/target levels.
        rr = _calc_rr(entry, stop, target)
        earn_days = int(c.get('earn_days', 999) or 999)
        rationale = str(ai_rec.get('rationale', '') or '')
        warnings = []
        if ai_rr > 0 and abs(ai_rr - rr) > 0.15:
            warnings.append(f"AI R:R {ai_rr:.2f} differs from level-based R:R {rr:.2f}")
        hinted_earn_days = _extract_earn_days_hint(rationale)
        if hinted_earn_days is not None and abs(hinted_earn_days - earn_days) > 1:
            warnings.append(f"Model note earnings hint ({hinted_earn_days}d) conflicts with signal ({earn_days}d)")
        gold = _normalize_gold_ai_contract(
            ticker,
            {
                'entry': entry,
                'stop': stop,
                'target': target,
                'price': float(c.get('price', 0) or 0),
            },
            fallback_ai_buy=str(ai_rec.get('ai_buy_recommendation', 'Watch Only') or 'Watch Only'),
            fallback_rank_score=float(ai_rec.get('rank_score', 0) or 0),
        )
        rows.append({
            'ticker': ticker,
            'trade_finder_run_id': run_id,
            'company_name': company_name,
            'price': float(c.get('price', 0) or 0),
            'earn_days': earn_days,
            'reason': str(c.get('recommendation', '')) + f" | Conviction {int(c.get('conviction', 0) or 0)}/10",
            'scanner_summary': str(c.get('summary', '') or ''),
            'watchlists': str(c.get('watchlists', '') or ''),
            'ai_buy_recommendation': gold.get('verdict', ai_rec.get('ai_buy_recommendation', 'Watch Only')),
            'suggested_entry': round(entry, 2),
            'suggested_stop_loss': round(stop, 2),
            'suggested_target': round(target, 2),
            'risk_reward': round(float(gold.get('risk_reward', rr) or rr), 2),
            'ai_reported_rr': round(ai_rr, 2),
            'rank_score': float(gold.get('unified_rank_score', ai_rec.get('rank_score', 0) or 0) or 0),
            'ai_rationale': rationale,
            'ai_confidence': int(gold.get('confidence', 0) or 0),
            'ai_position_sizing': str(gold.get('position_sizing', '') or ''),
            'gold_source': bool(gold.get('is_gold_source', False)),
            'consistency_warnings': warnings,
            'provider': gold.get('provider', ai_rec.get('provider', 'system')),
            'decision_card': build_trade_decision_card(
                ticker=ticker,
                source="trade_finder",
                recommendation=str(c.get('recommendation', '') or ''),
                ai_buy_recommendation=str(gold.get('verdict', ai_rec.get('ai_buy_recommendation', 'Watch Only')) or 'Watch Only'),
                conviction=max(int(c.get('conviction', 0) or 0), int(gold.get('confidence', 0) or 0)),
                quality_grade=str(c.get('quality_grade', '?') or '?'),
                entry=entry,
                stop=stop,
                target=target,
                rank_score=float(gold.get('unified_rank_score', ai_rec.get('rank_score', 0) or 0) or 0),
                regime=snap.regime,
                gate_status=gate.status,
                reason=str(c.get('summary', '') or str(c.get('recommendation', '') or '')),
                ai_rationale=str(gold.get('note', '') or ai_rec.get('rationale', '') or ''),
                sector_phase=str(c.get('sector_phase', '') or ''),
                earn_days=earn_days,
                explainability_bits=[
                    f"rec={str(c.get('recommendation', '') or '')}",
                    f"conv={int(c.get('conviction', 0) or 0)}",
                    f"phase={str(c.get('sector_phase', '') or '')}",
                    f"rr={float(gold.get('risk_reward', rr) or rr):.2f}",
                ],
            ).to_dict(),
        })
        rows[-1]['trade_score'] = compute_trade_score(rows[-1])

    rows = sorted(
        rows,
        key=lambda x: (float(x.get('trade_score', 0) or 0), float(x.get('rank_score', 0) or 0)),
        reverse=True,
    )
    elapsed = max(0.0, time.time() - _t0)
    st.session_state['trade_finder_results'] = {
        'run_id': run_id,
        'generated_at_iso': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rows': rows,
        'provider': rows[0].get('provider', 'system') if rows else 'system',
        'elapsed_sec': elapsed,
        'input_candidates': len(base_candidates),
    }
    try:
        get_journal().save_trade_finder_snapshot({
            'generated_at_iso': st.session_state['trade_finder_results'].get('generated_at_iso', ''),
            'run_id': run_id,
            'provider': st.session_state['trade_finder_results'].get('provider', 'system'),
            'elapsed_sec': float(elapsed),
            'input_candidates': int(len(base_candidates)),
            'rows': rows,
            'find_new_report': report,
        })
    except Exception:
        pass
    _append_perf_metric({
        'kind': 'trade_finder',
        'candidates_in': len(base_candidates),
        'candidates_out': len(rows),
        'sec': round(elapsed, 3),
        'provider': st.session_state['trade_finder_results'].get('provider', 'system'),
    })


def render_trade_finder_tab():
    """Top-level Trade Finder workflow with Grok ranking and click-through to Trade tab."""
    st.subheader("🧭 Trade Finder")
    st.caption("Runs cross-watchlist scan and ranks model/system trade signals with clear actions.")
    jm = get_journal()

    if st.button("🧭 Find New Trades", type="primary", width="stretch", key="tf_find_btn"):
        with st.spinner("Running trade finder and AI ranking..."):
            _run_trade_finder_workflow()
        st.rerun()

    results = st.session_state.get('trade_finder_results', {}) or {}
    rows = results.get('rows', []) or []
    if not rows:
        st.info("Click Find New Trades to generate ranked candidates.")
        return

    settings = _trade_quality_settings()
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.number_input(
            "Min R:R",
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            format="%.1f",
            key="trade_min_rr_threshold",
            help="Candidates below this risk/reward are excluded from Ready-to-Enter list.",
        )
    with fc2:
        st.number_input(
            "Block if earnings <= days",
            min_value=0,
            max_value=30,
            step=1,
            key="trade_earnings_block_days",
            help="Candidates with earnings inside this window are excluded by default.",
        )
    with fc3:
        st.checkbox(
            "Require READY",
            key="trade_require_ready",
            help="Only include decision-card READY candidates.",
        )
    st.checkbox(
        "Include Watch Only ideas",
        key="trade_include_watch_only",
        help="When enabled, Watch Only AI ratings are included if other gates pass.",
    )
    settings = _trade_quality_settings()
    qualified_rows = [r for r in rows if _trade_candidate_is_qualified(r, settings)]

    st.caption(
        f"Generated: {results.get('generated_at_iso', '')} | "
        f"Run: {results.get('run_id', 'n/a') or 'n/a'} | "
        f"Candidates: {len(rows)} | Ready: {len(qualified_rows)} | Runtime: {float(results.get('elapsed_sec', 0) or 0):.1f}s | "
        f"Provider: {results.get('provider', 'system')}"
    )
    st.caption("Snapshot is saved daily and restored after app restart.")

    if not qualified_rows:
        st.warning("No candidates meet current quality gates. Relax filters or run Find New Trades again.")
        fail = {
            'ai_rating_filtered': 0,
            'missing_levels': 0,
            'invalid_level_order': 0,
            'rr_below_threshold': 0,
            'earnings_blocked': 0,
            'not_ready': 0,
        }
        for r in rows:
            entry = float(r.get('suggested_entry', r.get('price', 0)) or 0)
            stop = float(r.get('suggested_stop_loss', 0) or 0)
            target = float(r.get('suggested_target', 0) or 0)
            rr = float(r.get('risk_reward', 0) or 0)
            if rr <= 0:
                rr = _calc_rr(entry, stop, target)
            ai_buy = str(r.get('ai_buy_recommendation', '') or '').strip()
            allowed_ai = {"Strong Buy", "Buy"} | ({"Watch Only"} if settings.get('include_watch_only') else set())
            if ai_buy not in allowed_ai:
                fail['ai_rating_filtered'] += 1
            if not (entry > 0 and stop > 0 and target > 0):
                fail['missing_levels'] += 1
            if not (stop < entry < target):
                fail['invalid_level_order'] += 1
            if rr < float(settings.get('min_rr', 2.0)):
                fail['rr_below_threshold'] += 1
            earn_days = int(r.get('earn_days', 999) or 999)
            if 0 <= earn_days <= int(settings.get('earn_block_days', 7)):
                fail['earnings_blocked'] += 1
            if settings.get('require_ready'):
                card = r.get('decision_card', {}) or {}
                if str(card.get('execution_readiness', '')).upper() != "READY":
                    fail['not_ready'] += 1
        st.caption(
            "Filter diagnostics: "
            + ", ".join([f"{k}={v}" for k, v in fail.items() if v > 0])
            if any(v > 0 for v in fail.values())
            else "Filter diagnostics: no hard failures detected (check data freshness)."
        )
    else:
        st.markdown("### Qualified Signals")
        st.caption("Signal Verdict uses model/system scoring. Use actions to review chart or place trade.")

        def _open_candidate_for_action(_r: Dict[str, Any], detail_tab: int):
            _ticker = str(_r.get('ticker', '')).upper().strip()
            st.session_state['trade_finder_selected_trade'] = build_trade_finder_selection(
                _r,
                generated_at_iso=str(results.get('generated_at_iso', '') or ''),
                run_id=str(results.get('run_id', '') or ''),
            )
            st.session_state['default_detail_tab'] = detail_tab
            st.session_state['_switch_to_scanner_tab'] = True
            _load_ticker_for_view(_ticker)
            st.rerun()

        primary_rows = [r for r in qualified_rows if str(r.get('ai_buy_recommendation', '')).strip() in {"Strong Buy", "Buy"}]
        watch_rows = [r for r in qualified_rows if str(r.get('ai_buy_recommendation', '')).strip() not in {"Strong Buy", "Buy"}]
        view_mode = st.radio(
            "View Mode",
            ["Cards", "Table"],
            horizontal=True,
            key="tf_view_mode",
        )

        def _stage_candidate(_r: Dict[str, Any], _price: float, _rr: float, _rank: float, _reason: str, _notes: str):
            # Canonical builder keeps staging payload consistent with New Trade prefill.
            _tmp = dict(_r)
            _tmp['price'] = _price
            _tmp['risk_reward'] = _rr
            _tmp['rank_score'] = _rank
            _tmp['reason'] = _reason
            _tmp['ai_rationale'] = _notes
            plan = build_planned_trade(_tmp, run_id=str(results.get('run_id', '') or ''))
            st.success(jm.add_planned_trade(plan))
            st.rerun()

        for group_label, group_rows, expanded in [
            ("🚀 Actionable Signals", primary_rows, True),
            ("👀 Watchlist Signals", watch_rows, False),
        ]:
            with st.expander(f"{group_label} ({len(group_rows)})", expanded=expanded):
                if not group_rows:
                    st.caption("None in this group.")
                if view_mode == "Table":
                    h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([1.0, 2.2, 1.0, 1.4, 0.9, 0.8, 0.9, 2.4])
                    h1.caption("Ticker")
                    h2.caption("Company")
                    h3.caption("Price")
                    h4.caption("Verdict")
                    h5.caption("R:R")
                    h6.caption("Score")
                    h7.caption("Earn")
                    h8.caption("Actions")
                for i, r in enumerate(group_rows[:40]):
                    ticker = str(r.get('ticker', '')).upper().strip()
                    company = str(r.get('company_name', '') or '').strip() or "Unknown company"
                    ai_buy = str(r.get('ai_buy_recommendation', '') or '')
                    rr = float(r.get('risk_reward', 0) or 0)
                    rank = float(r.get('rank_score', 0) or 0)
                    trade_score = float(r.get('trade_score', rank) or rank)
                    price = float(r.get('price', 0) or 0)
                    earn_days = int(r.get('earn_days', 999) or 999)
                    signal_reason = str(r.get('reason', '') or '')
                    rationale = str(r.get('ai_rationale', '') or str(r.get('scanner_summary', '') or '')).strip()
                    warnings = list(r.get('consistency_warnings', []) or [])

                    if view_mode == "Table":
                        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.0, 2.2, 1.0, 1.4, 0.9, 0.8, 0.9, 2.4])
                        c1.write(ticker)
                        c2.write(company)
                        c3.write(f"${price:.2f}")
                        c4.write(ai_buy or "N/A")
                        c5.write(f"{rr:.2f}:1")
                        c6.write(f"{float(r.get('trade_score', rank) or rank):.2f}")
                        c7.write(f"{earn_days}d")
                        a1, a2, a3 = c8.columns(3)
                        with a1:
                            if st.button("📈", key=f"tf_tbl_chart_{group_label}_{i}_{ticker}", help="View Chart", width="stretch"):
                                _open_candidate_for_action(r, detail_tab=1)
                        with a2:
                            if st.button("✅", key=f"tf_tbl_trade_{group_label}_{i}_{ticker}", help="Place Trade", width="stretch"):
                                _open_candidate_for_action(r, detail_tab=4)
                        with a3:
                            if st.button("🗂️", key=f"tf_tbl_stage_{group_label}_{i}_{ticker}", help="Stage Ticket", width="stretch"):
                                _stage_candidate(r, price, rr, rank, signal_reason, rationale)
                        if signal_reason:
                            st.caption(f"{ticker}: {signal_reason}")
                        for w in warnings[:2]:
                            st.caption(f"⚠ {w}")
                    else:
                        with st.container(border=True):
                            st.markdown(f"**{ticker} — {company}**")
                            st.caption(
                                f"Price ${price:.2f} | Signal Verdict: {ai_buy} | "
                                f"Trade Score: {trade_score:.2f} (Model {rank:.2f}) | Earnings: {earn_days}d | "
                                f"Source: {'Ask AI Gold' if bool(r.get('gold_source', False)) else 'Model'}"
                            )
                            st.caption(
                                f"Entry ${float(r.get('suggested_entry', price) or price):.2f} | "
                                f"Stop ${float(r.get('suggested_stop_loss', 0) or 0):.2f} | "
                                f"Target ${float(r.get('suggested_target', 0) or 0):.2f} | "
                                f"R:R {rr:.2f}:1"
                            )
                            st.caption(f"Signal Context: {signal_reason}")
                            if rationale:
                                st.caption(f"Analysis Note: {rationale}")
                            for w in warnings[:3]:
                                st.warning(f"Consistency warning: {w}")
                            a1, a2, a3 = st.columns(3)
                            with a1:
                                if st.button("📈 View Chart", key=f"tf_chart_{group_label}_{i}_{ticker}", width="stretch"):
                                    _open_candidate_for_action(r, detail_tab=1)
                            with a2:
                                if st.button("✅ Place Trade", key=f"tf_trade_{group_label}_{i}_{ticker}", width="stretch"):
                                    _open_candidate_for_action(r, detail_tab=4)
                            with a3:
                                if st.button("🗂️ Stage Ticket", key=f"tf_stage_{group_label}_{i}_{ticker}", width="stretch"):
                                    _stage_candidate(r, price, rr, rank, signal_reason, rationale)

    st.markdown("### Open In New Trade")
    sb1, sb2 = st.columns([1, 1])
    with sb1:
        stage_n = st.number_input("Stage Best N", min_value=1, max_value=20, value=5, step=1, key="tf_stage_best_n")
    with sb2:
        if st.button("🗂️ Stage Top Qualified", key="tf_stage_top_qualified", width="stretch"):
            staged = 0
            for r in qualified_rows[:int(stage_n)]:
                plan = build_planned_trade(r, run_id=str(results.get('run_id', '') or ''))
                jm.add_planned_trade(plan)
                staged += 1
            st.success(f"Staged {staged} qualified trade(s).")
            st.rerun()

    planned = jm.get_planned_trades()
    if planned:
        st.divider()
        st.markdown(f"### 🗂️ Planned Trade Tickets ({len(planned)})")
        for p in planned[:30]:
            pid = str(p.get('plan_id', ''))
            pt = str(p.get('ticker', ''))
            pstatus = str(p.get('status', 'PLANNED'))
            plabel = (
                f"{pt} | {pstatus} | Entry ${float(p.get('entry', 0) or 0):.2f} "
                f"| Stop ${float(p.get('stop', 0) or 0):.2f} | Target ${float(p.get('target', 0) or 0):.2f}"
            )
            st.caption(plabel)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                if st.button("▶ Open", key=f"plan_open_{pid}", width="stretch"):
                    st.session_state['trade_finder_selected_trade'] = {
                        'plan_id': pid,
                        'ticker': pt,
                        'entry': float(p.get('entry', 0) or 0),
                        'stop': float(p.get('stop', 0) or 0),
                        'target': float(p.get('target', 0) or 0),
                        'ai_buy_recommendation': str(p.get('ai_recommendation', '') or ''),
                        'risk_reward': float(p.get('risk_reward', 0) or 0),
                        'trade_finder_run_id': str(p.get('trade_finder_run_id', '') or ''),
                        'reason': str(p.get('reason', '') or ''),
                        'ai_rationale': str(p.get('notes', '') or ''),
                        'provider': str(p.get('source', 'planned') or 'planned'),
                    }
                    st.session_state['default_detail_tab'] = 4
                    st.session_state['_switch_to_scanner_tab'] = True
                    _load_ticker_for_view(pt)
                    st.rerun()
            with a2:
                if st.button("⚡ Trigger", key=f"plan_trigger_{pid}", width="stretch"):
                    st.info(jm.update_planned_trade_status(pid, "TRIGGERED"))
                    st.rerun()
            with a3:
                if st.button("✖ Cancel", key=f"plan_cancel_{pid}", width="stretch"):
                    st.info(jm.update_planned_trade_status(pid, "CANCELLED"))
                    st.rerun()
            with a4:
                if st.button("🗑️ Remove", key=f"plan_remove_{pid}", width="stretch"):
                    st.info(jm.remove_planned_trade(pid))
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
    bridge = get_bridge()

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
    # CROSS-WATCHLIST "FIND NEW TRADES" REPORT
    # ══════════════════════════════════════════════════════════════════
    find_new = st.session_state.get('find_new_trades_report', {}) or {}
    if find_new:
        cands = find_new.get('candidates', []) or []
        generated = find_new.get('generated_at_iso', '')
        wl_count = int(find_new.get('watchlists_count', 0) or 0)
        universe = int(find_new.get('scan_universe', 0) or 0)
        elapsed = float(find_new.get('elapsed_sec', 0) or 0)
        with st.expander(
            f"🧭 Find New Trades Report — {len(cands)} candidate(s) from {universe} tickers ({wl_count} watchlists)",
            expanded=True,
        ):
            st.caption(f"Generated: {generated} | Runtime: {elapsed:.1f}s")
            if not cands:
                st.info("No new buy candidates found in this run.")
            else:
                view_rows = []
                for c in cands[:50]:
                    view_rows.append({
                        'Ticker': c.get('ticker', ''),
                        'Rec': c.get('recommendation', ''),
                        'Conv': c.get('conviction', 0),
                        'Grade': c.get('quality_grade', ''),
                        'Score': c.get('score', 0),
                        'Price': f"${float(c.get('price', 0) or 0):.2f}",
                        'Sector': c.get('sector', ''),
                        'Phase': c.get('sector_phase', ''),
                        'Earnings': c.get('earn_date', ''),
                        'Watchlists': c.get('watchlists', ''),
                    })
                st.dataframe(pd.DataFrame(view_rows), hide_index=True, width='stretch')

    # ══════════════════════════════════════════════════════════════════
    # WATCHLIST EDITOR (collapsible)
    # ══════════════════════════════════════════════════════════════════
    watchlist_tickers = bridge.get_watchlist_tickers()
    favorite_tickers = set(bridge.get_favorite_tickers())

    # Get active watchlist name for display
    _active_wl_name = bridge.manager.get_active_watchlist().get("name", "Watchlist")

    # Watchlist version counter — used to force text_area reset when watchlist changes
    if 'wl_version' not in st.session_state:
        st.session_state['wl_version'] = 0

    with st.expander(f"📋 {_active_wl_name} ({len(watchlist_tickers)} tickers) — click to edit",
                     expanded=(len(watchlist_tickers) == 0)):

        # ── Quick Add (single ticker) ─────────────────────────────────
        qa1, qa2 = st.columns([3, 1])
        with qa1:
            new_ticker = st.text_input("Add ticker", placeholder="e.g. AAPL",
                                       key="wl_add_input", label_visibility="collapsed")
        with qa2:
            if st.button("➕ Add", key="wl_add_btn", width="stretch"):
                if new_ticker:
                    import re as _re
                    ticker_clean = new_ticker.strip().upper()
                    # Validate: base symbols (AAPL), class shares (BRK.B), or indices (^VIX).
                    is_base = bool(_re.match(r'^[A-Z]{1,5}$', ticker_clean))
                    class_match = _re.match(r'^[A-Z]{1,5}\.([A-Z])$', ticker_clean)
                    is_class = bool(class_match and class_match.group(1) in {'A', 'B', 'C', 'D'})
                    is_index = ticker_clean.startswith('^')
                    if ticker_clean and (is_base or is_class or is_index):
                        if ticker_clean not in watchlist_tickers:
                            bridge.add_to_watchlist(WatchlistItem(ticker=ticker_clean))
                            _append_audit_event("ADD_TICKER", ticker_clean, source="quick_add")
                            st.session_state['wl_version'] += 1
                            st.rerun()
                        else:
                            st.toast(f"⚠️ {ticker_clean} already in watchlist")
                    else:
                        st.toast(f"⚠️ Invalid ticker: {ticker_clean[:20]}")

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
                        bridge.toggle_favorite(t)
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
                                 help="Remove from watchlist"):
                        bridge.remove_from_watchlist(t)
                        # Also remove from JM for metadata sync
                        try:
                            jm.delete_single_ticker(t)
                        except Exception:
                            pass
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
                        _append_audit_event("REMOVE_TICKER", t, source="row_delete")
                        st.rerun()

        # ── Bulk Editor (for pasting 200 tickers) ─────────────────────
        with st.expander("📝 Bulk Edit (paste tickers)"):
            st.caption("Paste tickers separated by commas, spaces, or new lines.")
            append_only = st.checkbox(
                "Add/Merge only (recommended)",
                value=True,
                key="wl_bulk_append_only",
                help="On: adds new tickers and keeps existing ones. Off: replaces watchlist with pasted list.",
            )

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
                if st.button("💾 Save", width="stretch", type="primary",
                             key="wl_save"):
                    import re
                    raw = re.split(r'[,\s\n\t]+', new_text)
                    tickers = [t.strip().upper() for t in raw if t.strip()]
                    
                    # Validate: real tickers are 1-5 uppercase alpha chars
                    # Reject comma-strings, numbers-only, special chars
                    seen = set()
                    unique = []
                    rejected = []
                    for t in tickers:
                        if t in seen:
                            continue
                        is_base = bool(re.match(r'^[A-Z]{1,5}$', t))
                        class_match = re.match(r'^[A-Z]{1,5}\.([A-Z])$', t)
                        is_class = bool(class_match and class_match.group(1) in {'A', 'B', 'C', 'D'})
                        is_index = t.startswith('^')
                        if not (is_base or is_class or is_index):
                            rejected.append(t)
                            continue
                        seen.add(t)
                        unique.append(t)

                    existing_set = set(watchlist_tickers)
                    if append_only:
                        # Merge mode (default): incremental adds via the same path
                        # as quick-add, so auto/manual watchlist rules stay consistent.
                        added_count = 0
                        for t in unique:
                            if t not in existing_set:
                                bridge.add_to_watchlist(WatchlistItem(ticker=t))
                                existing_set.add(t)
                                added_count += 1
                        st.session_state.pop('wl_bulk_pending_unique', None)
                        st.session_state.pop('wl_bulk_pending_rejected', None)
                        st.session_state.pop('wl_bulk_pending_removed', None)
                        _append_audit_event(
                            "BULK_ADD",
                            f"added={added_count} total={len(existing_set)} rejected={len(rejected)}",
                            source="bulk_merge",
                        )
                        st.session_state['wl_version'] += 1
                        msg = f"✅ Added {added_count} ticker(s) | Total: {len(existing_set)}"
                        if rejected:
                            msg += f" | ⚠️ Rejected {len(rejected)}: {', '.join(rejected[:5])}"
                        st.success(msg)
                        st.rerun()

                    new_set = set(unique)
                    removed = sorted(existing_set - new_set)

                    # Guardrail: require explicit confirmation before removing existing tickers.
                    if removed:
                        st.session_state['wl_bulk_pending_unique'] = unique
                        st.session_state['wl_bulk_pending_rejected'] = rejected
                        st.session_state['wl_bulk_pending_removed'] = removed
                        st.rerun()

                    # Preserve favorites across bulk save (non-destructive path).
                    old_favorites = set(bridge.get_favorite_tickers())
                    bridge.set_watchlist_tickers(unique)
                    for fav in old_favorites:
                        if fav in unique and not bridge.is_favorite(fav):
                            bridge.toggle_favorite(fav)

                    _append_audit_event(
                        "BULK_REPLACE",
                        f"saved={len(unique)} removed={len(removed)} rejected={len(rejected)}",
                        source="bulk_replace",
                    )
                    st.session_state['wl_version'] += 1
                    msg = f"✅ Saved {len(unique)} tickers"
                    if rejected:
                        msg += f" | ⚠️ Rejected {len(rejected)}: {', '.join(rejected[:5])}"
                    st.success(msg)
                    st.rerun()
            with wl_col2:
                if st.button("🗑️ Clear All", width="stretch", key="wl_clear"):
                    bridge.clear_watchlist()
                    st.session_state.pop('wl_bulk_pending_unique', None)
                    st.session_state.pop('wl_bulk_pending_rejected', None)
                    st.session_state.pop('wl_bulk_pending_removed', None)
                    st.session_state['wl_version'] += 1
                    st.session_state['scan_results'] = []
                    st.session_state['scan_results_summary'] = []
                    st.session_state['scan_timestamp'] = ''
                    st.session_state['ticker_data_cache'] = {}
                    # Clear per-watchlist scan cache
                    _clear_wl_id = bridge.manager.get_active_watchlist().get("id", "")
                    if _clear_wl_id:
                        save_scan_for_watchlist(_clear_wl_id)
                    _append_audit_event("CLEAR_WATCHLIST", "all tickers removed", source="bulk_clear")
                    st.rerun()
            with wl_col3:
                st.caption(f"{len(watchlist_tickers)} saved"
                           + (f" | ⭐ {len(favorite_tickers)}" if favorite_tickers else ""))

            pending_unique = st.session_state.get('wl_bulk_pending_unique')
            pending_rejected = st.session_state.get('wl_bulk_pending_rejected', [])
            pending_removed = st.session_state.get('wl_bulk_pending_removed', [])
            if pending_unique is not None:
                st.warning(
                    f"This save will remove {len(pending_removed)} ticker(s): "
                    f"{', '.join(pending_removed[:10])}"
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Confirm Save", key="wl_bulk_confirm_save", type="primary", width="stretch"):
                        old_favorites = set(bridge.get_favorite_tickers())
                        bridge.set_watchlist_tickers(pending_unique)
                        for fav in old_favorites:
                            if fav in pending_unique and not bridge.is_favorite(fav):
                                bridge.toggle_favorite(fav)
                        st.session_state.pop('wl_bulk_pending_unique', None)
                        st.session_state.pop('wl_bulk_pending_rejected', None)
                        st.session_state.pop('wl_bulk_pending_removed', None)
                        _append_audit_event(
                            "BULK_REPLACE_CONFIRM",
                            f"saved={len(pending_unique)} removed={len(pending_removed)} rejected={len(pending_rejected)}",
                            source="bulk_replace_confirm",
                        )
                        st.session_state['wl_version'] += 1
                        msg = f"✅ Saved {len(pending_unique)} tickers"
                        if pending_rejected:
                            msg += f" | ⚠️ Rejected {len(pending_rejected)}: {', '.join(pending_rejected[:5])}"
                        st.success(msg)
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="wl_bulk_confirm_cancel", width="stretch"):
                        st.session_state.pop('wl_bulk_pending_unique', None)
                        st.session_state.pop('wl_bulk_pending_rejected', None)
                        st.session_state.pop('wl_bulk_pending_removed', None)
                        _append_audit_event("BULK_REPLACE_CANCEL", "user canceled replacement", source="bulk_replace_confirm")
                        st.info("Bulk save canceled.")

            with st.expander("🧾 Activity Log", expanded=False):
                current_wl_id = bridge.manager.get_active_watchlist().get("id", "")
                show_all = st.checkbox("Show all watchlists", value=False, key="wl_audit_show_all")
                events = _get_audit_events()
                if not show_all:
                    events = [e for e in events if e.get('wl_id') == current_wl_id]

                if not events:
                    st.caption("No activity logged yet.")
                else:
                    for e in events[:30]:
                        ts = e.get('ts', '')
                        action = e.get('action', '')
                        src = e.get('source', '')
                        details = e.get('details', '')
                        st.caption(f"{ts} | {action} | {src} | {details}")

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("Clear Log", key="wl_audit_clear", width="stretch"):
                        _clear_audit_events()
                        st.rerun()
                with a2:
                    st.caption(f"{len(events)} shown")

    # ══════════════════════════════════════════════════════════════════
    # WATCHLIST NOTES (collapsible accordion)
    # ══════════════════════════════════════════════════════════════════
    _active_wl = bridge.manager.get_active_watchlist()
    _active_wl_id = _active_wl.get("id", "")
    _existing_notes = _active_wl.get("notes", "")
    _notes_edit_key = f"_notes_editing_{_active_wl_id}"
    _is_editing = st.session_state.get(_notes_edit_key, False)

    # Only show if notes exist OR user is editing
    _has_notes = bool(_existing_notes and _existing_notes.strip())

    # Build label
    if _has_notes:
        # Show preview of first line (truncated)
        _first_line = _existing_notes.strip().split('\n')[0][:60]
        _notes_label = f"📝 Notes — {_first_line}{'…' if len(_first_line) >= 60 else ''}"
    else:
        _notes_label = "📝 Notes — click to add"

    with st.expander(_notes_label, expanded=_is_editing):
        if _is_editing:
            # ── Edit mode ──
            _edited = st.text_area(
                "Watchlist notes (Markdown supported)",
                value=_existing_notes,
                height=250,
                key=f"notes_textarea_{_active_wl_id}",
                placeholder="Add notes, rationales, strategy context...\n\n"
                            "Supports **bold**, *italic*, - bullet lists, 1. numbered lists",
                label_visibility="collapsed",
            )
            _nc1, _nc2, _nc3 = st.columns([1, 1, 3])
            with _nc1:
                if st.button("💾 Save", key="notes_save", type="primary", width="stretch"):
                    bridge.manager.update_watchlist_metadata(_active_wl_id, notes=_edited)
                    st.session_state[_notes_edit_key] = False
                    st.toast("Notes saved ✓")
                    st.rerun()
            with _nc2:
                if st.button("Cancel", key="notes_cancel", width="stretch"):
                    st.session_state[_notes_edit_key] = False
                    st.rerun()
        else:
            # ── View mode ──
            if _has_notes:
                st.markdown(_existing_notes)
            else:
                st.caption("No notes yet.")
            if st.button("✏️ Edit Notes", key="notes_edit_btn"):
                st.session_state[_notes_edit_key] = True
                st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # SCAN RESULTS
    # ══════════════════════════════════════════════════════════════════

    # No results at all
    if not results and not summary:
        st.info(f"👆 Add tickers to **{_active_wl_name}** (above) and click **Scan All** in the sidebar.")
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
    fav_tickers = set(bridge.get_favorite_tickers())

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
                        width="stretch"):
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
        if st.button("➕ Add & Scan", width="stretch"):
            if quick_ticker:
                ticker_clean = quick_ticker.strip().upper()
                wl = bridge.get_watchlist_tickers()
                if ticker_clean not in wl:
                    bridge.add_to_watchlist(WatchlistItem(ticker=ticker_clean))
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

    def _resolve_sector(_ticker: str) -> str:
        t = str(_ticker or '').upper().strip()
        if not t:
            return ''
        sec = str(ticker_sectors.get(t, '') or '').strip()
        if sec:
            return sec
        # Lazy backfill when scan-time sector assignment missed this ticker.
        try:
            from data_fetcher import get_ticker_sector
            sec = str(get_ticker_sector(t) or '').strip()
            if sec:
                ticker_sectors[t] = sec
                st.session_state['ticker_sectors'] = ticker_sectors
                return sec
        except Exception:
            pass
        return ''

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
        sector = _resolve_sector(r.ticker)
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
        if earn:
            _ed = earn['days_until']
            if _ed <= 7:
                earn_flag = f"⚡{_ed}d"
            elif _ed <= 14:
                earn_flag = f"⏰{_ed}d"
            elif _ed <= 30:
                earn_flag = f"📅{_ed}d"
            else:
                earn_flag = f"{_ed}d"
        else:
            earn_flag = ""

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
    ticker_sectors = st.session_state.get('ticker_sectors', {})
    earnings_flags = st.session_state.get('earnings_flags', {})

    def _resolve_sector(_ticker: str, _persisted_sector: str) -> str:
        sec = str(_persisted_sector or '').strip()
        if sec:
            return sec
        t = str(_ticker or '').upper().strip()
        if not t:
            return ''
        sec = str(ticker_sectors.get(t, '') or '').strip()
        if sec:
            return sec
        try:
            from data_fetcher import get_ticker_sector
            sec = str(get_ticker_sector(t) or '').strip()
            if sec:
                ticker_sectors[t] = sec
                st.session_state['ticker_sectors'] = ticker_sectors
                return sec
        except Exception:
            pass
        return ''

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
        sector = _resolve_sector(ticker, s.get('sector', ''))
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
        default_trigger = float(current * 1.03) if current > 0 else 0.0
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

@st.fragment
def render_detail_view():
    """Render detailed analysis for selected ticker.
    
    Decorated with @st.fragment — interactions within this view (button clicks,
    tab switches, checkbox toggles) only re-render this fragment, NOT the entire
    scanner table + sidebar. This eliminates ~1-2s of lag on every interaction.
    """
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

    # ── Tabs (supports programmatic default by ordering selected tab first) ──
    tab_defs = [
        ("📊 Signal", "signal"),
        ("📈 Chart", "chart"),
        ("🤖 AI Intel", "ai"),
        ("💬 Ask AI", "chat"),
        ("💼 Trade", "trade"),
    ]
    default_tab = int(st.session_state.pop('default_detail_tab', 0) or 0)
    if default_tab < 0 or default_tab >= len(tab_defs):
        default_tab = 0
    ordered_defs = tab_defs[default_tab:] + tab_defs[:default_tab]
    tabs = st.tabs([name for name, _ in ordered_defs])

    for tab, (_name, key) in zip(tabs, ordered_defs):
        with tab:
            if key == "signal":
                _render_signal_tab(signal, analysis)
            elif key == "chart":
                _render_chart_tab(ticker, signal)
            elif key == "ai":
                _render_ai_tab(ticker, signal, rec, analysis)
            elif key == "chat":
                _render_chat_tab(ticker, signal, rec, analysis)
            elif key == "trade":
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
                width="stretch",
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

    # ── Get cached AI clients (no re-import/re-create on every rerun) ──
    ai_clients = _get_ai_clients()
    openai_client = ai_clients['openai_client']
    gemini = ai_clients['gemini']
    ai_config = ai_clients['ai_config']
    primary_error = ai_clients['primary_error']
    gemini_error = ai_clients['gemini_error']

    # Show provider info
    if openai_client:
        st.caption(f"Provider: {ai_config['display']} | Model: {ai_config['model']}")

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
        if st.button("▶️ Run AI Analysis", type="primary", width="stretch"):
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
            result['fundamentals'] = fundamentals  # For earnings fallback in risk assessment
            result['fundamental_profile'] = fundamental_profile

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
    # EARNINGS DATE BANNER (always visible — multiple source fallback)
    # ═══════════════════════════════════════════════════════════════
    _earn_hist = ai_result.get('earnings_history', {})
    _earn_cal = ai_result.get('fundamentals', {}).get('earnings', {}) if ai_result.get('fundamentals') else {}

    # Try earnings_history first (has quarters + streak), then calendar fallback
    _e_date = _earn_hist.get('next_earnings') or _earn_cal.get('next_earnings')
    _e_days = _earn_hist.get('days_until_earnings') or _earn_cal.get('days_until_earnings')
    _e_eps = _earn_hist.get('next_eps_estimate') or _earn_cal.get('next_eps_estimate')
    _e_conf = _earn_hist.get('confidence') or _earn_cal.get('confidence', '')
    _e_src = _earn_hist.get('source') or _earn_cal.get('source', '')
    _e_streak = _earn_hist.get('streak', 0)
    _e_last = _earn_hist.get('last_earnings') or _earn_cal.get('last_earnings', '')

    if _e_date:
        _eps_str = f" | EPS Est: ${_e_eps:.2f}" if _e_eps else ""
        _streak_str = f" | {'🔥' if _e_streak > 0 else '📉'} {abs(_e_streak)} {'beat' if _e_streak > 0 else 'miss'}{'s' if abs(_e_streak) > 1 else ''}" if _e_streak != 0 else ""
        _conf_str = f" ({_e_conf})" if _e_conf else ""

        if _e_days is not None and _e_days <= 7:
            st.error(f"⚡ **Earnings: {_e_date} ({_e_days}d)**{_eps_str}{_streak_str} — Binary event imminent{_conf_str}")
        elif _e_days is not None and _e_days <= 14:
            st.warning(f"⏰ **Earnings: {_e_date} ({_e_days}d)**{_eps_str}{_streak_str} — Plan exit strategy{_conf_str}")
        elif _e_days is not None and _e_days <= 30:
            st.info(f"📅 **Earnings: {_e_date} ({_e_days}d)**{_eps_str}{_streak_str}{_conf_str}")
        else:
            _days_str = f" ({_e_days}d)" if _e_days else ""
            st.caption(f"📅 Next Earnings: {_e_date}{_days_str}{_eps_str}{_streak_str}{_conf_str}")
    elif _e_last:
        st.caption(f"📅 Earnings: Not yet announced (Last: {_e_last})")
    # If no earnings data at all, don't clutter the header

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

    # Earnings date — try earnings_history first, then fundamentals.earnings as fallback
    _earnings_data = ai_result.get('earnings_history', {})
    _next_earnings = _earnings_data.get('next_earnings') if _earnings_data else None
    _days_to_earn = _earnings_data.get('days_until_earnings') if _earnings_data else None
    _earn_confidence = _earnings_data.get('confidence', '') if _earnings_data else ''

    # Fallback: fetch_earnings_date result (stored in fundamentals dict during AI Intel)
    if not _next_earnings:
        _fund_earnings = ai_result.get('fundamentals', {}).get('earnings', {}) if ai_result.get('fundamentals') else {}
        if _fund_earnings:
            _next_earnings = _fund_earnings.get('next_earnings')
            _days_to_earn = _fund_earnings.get('days_until_earnings')
            _earn_confidence = _fund_earnings.get('confidence', '')

    # Parse days if we have a date string but no days count
    if _next_earnings and _days_to_earn is None:
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
                st.dataframe(_df, hide_index=True, width='stretch')

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
    earnings = ai_result.get('earnings_history', {}) or {}

    # Merge in fallback date from fetch_earnings_date() if history didn't get it
    if not earnings.get('next_earnings'):
        _fallback_earn = ai_result.get('fundamentals', {}).get('earnings', {}) if ai_result.get('fundamentals') else {}
        if _fallback_earn and _fallback_earn.get('next_earnings'):
            earnings['next_earnings'] = _fallback_earn['next_earnings']
            earnings['days_until_earnings'] = _fallback_earn.get('days_until_earnings')
            if not earnings.get('next_eps_estimate'):
                earnings['next_eps_estimate'] = _fallback_earn.get('next_eps_estimate')
            earnings['confidence'] = _fallback_earn.get('confidence', '')
            earnings['source'] = _fallback_earn.get('source', '')

    if earnings and (earnings.get('next_earnings') or earnings.get('quarters')):
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
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

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
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
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
    confidence = earnings.get('confidence', '')
    source = earnings.get('source', '')

    if not next_date and not quarters:
        return

    st.markdown("### 📅 Earnings")

    # ── Next Earnings Banner ──────────────────────────────────────
    if next_date:
        next_eps = earnings.get('next_eps_estimate')
        conf_str = f" [{confidence}]" if confidence and confidence != 'HIGH' else ""

        if days_until is not None and days_until <= 14:
            # Imminent — warning
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.error(f"⚠️ **Next Earnings: {next_date} ({days_until} days)**{eps_str}{conf_str} — "
                     f"Consider waiting or reducing size")
        elif days_until is not None and days_until <= 30:
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.warning(f"📅 **Next Earnings: {next_date} ({days_until} days)**{eps_str}{conf_str}")
        else:
            eps_str = f" | Consensus EPS: ${next_eps:.2f}" if next_eps else ""
            st.info(f"📅 **Next Earnings: {next_date}"
                    f"{f' ({days_until} days)' if days_until else ''}**{eps_str}{conf_str}")

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
        st.dataframe(df, width='stretch', hide_index=True,
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
    # SECTION 3: EARNINGS DATE (MANDATORY — 4-method cascade via fetch_earnings_date)
    # ══════════════════════════════════════════════════════════════════
    lines.append(f"\n📅 EARNINGS DATE (MANDATORY — you MUST state this):")
    earnings_found = False
    earnings_date_str = None
    earnings_days = None

    # Source 1: App session (batch scanner — instant, no API call)
    earn_flag = st.session_state.get('earnings_flags', {}).get(ticker)
    if earn_flag:
        earnings_date_str = earn_flag.get('next_earnings')
        earnings_days = earn_flag.get('days_until')
        earnings_found = True
        lines.append(f"  Source: TTA Scanner")
        lines.append(f"  Next Earnings: {earnings_date_str}")
        lines.append(f"  Days Until: {earnings_days}")

    # Source 2: fetch_earnings_date() — 4-method cascade (calendar → info → earnings_dates → pattern)
    if not earnings_found:
        try:
            from data_fetcher import fetch_earnings_date
            earn_result = fetch_earnings_date(ticker)
            if earn_result.get('next_earnings'):
                earnings_date_str = earn_result['next_earnings']
                earnings_days = earn_result.get('days_until_earnings')
                earnings_found = True
                _src = earn_result.get('source', 'Yahoo Finance')
                _conf = earn_result.get('confidence', 'MEDIUM')
                lines.append(f"  Source: {_src} (confidence: {_conf})")
                lines.append(f"  Next Earnings: {earnings_date_str}")
                lines.append(f"  Days Until: {earnings_days}")
                if earn_result.get('next_eps_estimate'):
                    lines.append(f"  EPS Estimate: ${earn_result['next_eps_estimate']:.2f}")
                if earn_result.get('last_earnings'):
                    lines.append(f"  Last Earnings: {earn_result['last_earnings']}")
        except Exception as e:
            lines.append(f"  Earnings fetch error: {str(e)[:100]}")

    if not earnings_found:
        lines.append(f"  ⚠️ EARNINGS DATE NOT FOUND after 4-method cascade")
        lines.append(f"  → Tell user: 'Earnings date could not be determined from available sources'")
        lines.append(f"  → Still provide risk framework based on typical quarterly patterns")

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

    # ── Get cached AI client (no re-import/re-create on every rerun) ──
    ai_clients = _get_ai_clients()
    openai_client = ai_clients['openai_client']
    ai_config = ai_clients['ai_config']

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
CRITICAL FORMATTING RULE: NEVER use markdown formatting in your response — no *italic*, **bold**, or ***bold italic*** markers. Use PLAIN TEXT only with proper spacing between ALL words. Section headers should be plain numbered labels like "1. MARKET & SECTOR CONTEXT".

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
- FORMATTING: Do NOT use markdown formatting in your response body — no *italic*, **bold**, or ***bold italic*** markers anywhere. Use PLAIN TEXT with proper spacing between all words. Section headers should be plain numbered labels (e.g., "1. MARKET & SECTOR CONTEXT") without markdown.

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
            if st.button("▶️ Run Research Analysis", type="primary", width="stretch",
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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
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
            width="stretch",
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


def _extract_ai_trade_levels(
    ai_result: Dict,
    technical_entry: float,
    technical_stop: float,
    technical_target: float,
    current_price: float,
) -> Dict[str, float]:
    """Parse AI text for entry/stop/target levels with technical fallback."""
    action = str(ai_result.get('action', '') or '')
    fields = [
        action,
        str(ai_result.get('resistance_verdict', '') or ''),
        str(ai_result.get('analysis', '') or ''),
        str(ai_result.get('raw_text', '') or ''),
        str(ai_result.get('bull_case', '') or ''),
        str(ai_result.get('bear_case', '') or ''),
    ]
    blob = "\n".join(fields)

    def _find_labeled_price(labels: List[str]) -> float:
        for label in labels:
            m = re.search(rf"{label}\s*[:=]?\s*\$?\s*(\d+(?:\.\d+)?)", blob, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
        return 0.0

    ai_entry = _find_labeled_price([r'entry', r'buy(?:\s+zone)?', r'breakout(?:\s+above)?', r'entry zone'])
    ai_stop = _find_labeled_price([r'stop(?:\s+loss)?', r'invalidat(?:ion|e)\s+below'])
    ai_target = _find_labeled_price([r'target', r'price target', r'upside target'])

    if not ai_entry:
        m_break = re.search(r'WAIT\s+FOR\s+BREAKOUT\s+above\s+\$?(\d+(?:\.\d+)?)', action, flags=re.IGNORECASE)
        if m_break:
            ai_entry = float(m_break.group(1))

    if not ai_entry and "BUY NOW" in action.upper() and current_price > 0:
        ai_entry = float(current_price)

    entry = ai_entry if ai_entry > 0 else float(technical_entry or current_price or 0.0)
    stop = ai_stop if (ai_stop > 0 and ai_stop < entry) else float(technical_stop or 0.0)
    target = ai_target if (ai_target > entry) else float(technical_target or 0.0)

    using_ai = bool(ai_entry or ai_stop or ai_target)
    return {
        'entry': float(entry),
        'stop': float(stop),
        'target': float(target),
        'using_ai': using_ai,
        'parsed_entry': float(ai_entry),
        'parsed_stop': float(ai_stop),
        'parsed_target': float(ai_target),
    }


def _build_trade_ticket_note(
    ticker: str,
    ai_result: Dict,
    rec: Dict,
    signal: EntrySignal,
    source_mode: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> str:
    """Build timestamped trade-ticket note attached to the trade."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    action = str(ai_result.get('action', ai_result.get('timing', '')) or '')
    conviction = ai_result.get('conviction', rec.get('conviction', 0))
    pos_size_txt = str(ai_result.get('position_sizing', '') or '')
    resistance = str(ai_result.get('resistance_verdict', '') or '')
    why = str(ai_result.get('why_moving', '') or '')
    red_flags = str(ai_result.get('red_flags', '') or '')

    lines = [
        f"[TRADE TICKET | {ts}]",
        f"Ticker: {ticker}",
        f"Default Source: {source_mode}",
        f"Action: {action or rec.get('recommendation', 'N/A')}",
        f"Conviction: {conviction}/10",
        f"Position Sizing Guidance: {pos_size_txt or 'N/A'}",
        f"Planned Entry: ${entry_price:.2f} | Stop: ${stop_price:.2f} | Target: ${target_price:.2f}",
    ]
    if signal and signal.stops:
        rr = signal.stops.get('reward_risk')
        if rr:
            lines.append(f"Technical Reward:Risk: {rr}")
    if resistance:
        lines.append(f"Resistance Verdict: {resistance[:220]}")
    if why:
        lines.append(f"Narrative: {why[:220]}")
    if red_flags:
        lines.append(f"Red Flags: {red_flags[:220]}")

    return "\n".join(lines)


def _render_position_calculator(ticker, signal, analysis, jm, rec, stops):
    """
    Institutional-grade position calculator using position_sizer.py.
    Uses AI-recommended defaults when available, with technical fallback.
    """
    from position_sizer import calculate_position_size

    current_price = float(analysis.current_price or 0.0)
    tech_entry = float(stops.get('entry', current_price or 0.0))
    tech_stop = float(stops.get('stop', 0.0))
    tech_target = float(stops.get('target', 0.0))
    ai_result = st.session_state.get(f'ai_result_{ticker}', {}) or {}
    ai_levels = _extract_ai_trade_levels(ai_result, tech_entry, tech_stop, tech_target, current_price)
    finder_trade = st.session_state.get('trade_finder_selected_trade', {}) or {}
    finder_for_ticker = finder_trade if str(finder_trade.get('ticker', '')).upper() == str(ticker).upper() else {}
    if finder_for_ticker:
        ai_levels['entry'] = float(finder_for_ticker.get('entry', ai_levels.get('entry', current_price)) or ai_levels.get('entry', current_price) or current_price)
        ai_levels['stop'] = float(finder_for_ticker.get('stop', ai_levels.get('stop', 0)) or ai_levels.get('stop', 0) or 0)
        ai_levels['target'] = float(finder_for_ticker.get('target', ai_levels.get('target', 0)) or ai_levels.get('target', 0) or 0)
        ai_levels['using_ai'] = True

    account_size = float(st.session_state.get('account_size', 100000.0))
    open_trades = jm.get_open_trades()
    snap = _build_dashboard_snapshot()
    policy = snap.risk_policy
    gate = _evaluate_trade_gate(snap)
    win_rate = jm.get_recent_win_rate(last_n=20)
    losing_streak = jm.get_current_losing_streak()
    stale_scan = _is_stale(snap.scan_ts, 3 * 3600)
    stale_market = _is_stale(snap.market_ts, 30 * 60)
    stale_sector = _is_stale(snap.sector_ts, 60 * 60)
    stale_positions = _is_stale(snap.pos_ts, 10 * 60)
    stale_alerts = _is_stale(snap.alert_ts, 5 * 60)
    stale_tags = []
    if stale_scan:
        stale_tags.append("scan")
    if stale_market:
        stale_tags.append("market")
    if stale_sector:
        stale_tags.append("sectors")
    if stale_positions:
        stale_tags.append("positions")
    if stale_alerts:
        stale_tags.append("alerts")
    stale_count = len(stale_tags)
    hard_block_stale = stale_count >= 3

    st.subheader(f"📐 Position Sizer — {ticker}")

    if losing_streak >= 2:
        st.warning(f"⚠️ On a {losing_streak}-trade losing streak — position sizes auto-reduced")
    if win_rate < 0.4 and jm.get_trade_history(last_n=5):
        st.warning(f"⚠️ Recent win rate: {win_rate:.0%} — consider reducing exposure")
    st.caption(
        f"Regime policy: {policy.regime} | "
        f"Max new/day {policy.max_new_trades} | "
        f"Size x{policy.position_size_multiplier:.2f} | "
        f"Max open {policy.max_total_open_positions}"
    )
    if gate.severity == "danger":
        st.error(f"{gate.label} — {gate.reason}")
    elif gate.severity == "warning":
        st.warning(f"{gate.label} — {gate.reason}")
    else:
        st.success(f"{gate.label} — {gate.reason}")
    if hard_block_stale:
        st.error(
            "Entry hard-blocked: critical stale data streams detected "
            f"({', '.join(stale_tags)}). Run Fast Refresh before entering new trades."
        )
    if len(open_trades) >= policy.max_total_open_positions:
        st.error(
            f"New entries blocked by policy: open positions {len(open_trades)} >= "
            f"max {policy.max_total_open_positions} for regime {policy.regime}."
        )

    source_options = ["AI Recommended", "Technical"]
    default_source_key = f"sizer_default_source_{ticker}"
    prev_source_key = f"sizer_default_source_prev_{ticker}"
    if default_source_key not in st.session_state:
        st.session_state[default_source_key] = "AI Recommended" if ai_levels.get('using_ai') else "Technical"
    selected_source = st.selectbox(
        "Default Price Source",
        source_options,
        index=source_options.index(st.session_state.get(default_source_key, "Technical")),
        key=default_source_key,
    )

    selected_entry = ai_levels['entry'] if selected_source == "AI Recommended" else tech_entry
    selected_stop = ai_levels['stop'] if selected_source == "AI Recommended" else tech_stop
    selected_target = ai_levels['target'] if selected_source == "AI Recommended" else tech_target
    recommended_stop_ref = float(ai_levels.get('stop', 0) or 0)
    recommended_stop_src = "Grok/AI" if ai_levels.get('using_ai') else "System"

    if selected_source == "AI Recommended":
        if ai_levels.get('using_ai'):
            st.caption("AI defaults loaded (with technical fallback for missing fields).")
        else:
            st.caption("AI defaults not fully available for this ticker; using technical fallback.")
    else:
        st.caption("Technical defaults loaded from signal model.")

    entry_key = f"sizer_entry_{ticker}"
    stop_key = f"sizer_stop_{ticker}"
    target_key = f"sizer_target_{ticker}"
    stop_dist_key = f"sizer_stop_dist_pct_{ticker}"
    stop_dist_prev_key = f"sizer_stop_dist_pct_prev_{ticker}"
    stop_prev_key = f"sizer_stop_prev_{ticker}"
    confirm_entry_key = f"confirm_entry_{ticker}"
    confirm_stop_key = f"confirm_stop_{ticker}"
    confirm_target_key = f"confirm_target_{ticker}"

    if entry_key not in st.session_state:
        st.session_state[entry_key] = float(selected_entry if selected_entry > 0 else current_price)
    if stop_key not in st.session_state:
        st.session_state[stop_key] = float(selected_stop)
    if target_key not in st.session_state:
        st.session_state[target_key] = float(selected_target)
    if stop_dist_key not in st.session_state:
        _entry0 = float(st.session_state.get(entry_key, selected_entry) or 0)
        _stop0 = float(st.session_state.get(stop_key, selected_stop) or 0)
        _dist0 = ((_entry0 - _stop0) / _entry0 * 100) if _entry0 > 0 and _stop0 > 0 and _entry0 > _stop0 else 0.0
        st.session_state[stop_dist_key] = round(max(0.0, _dist0), 2)
    if stop_dist_prev_key not in st.session_state:
        st.session_state[stop_dist_prev_key] = float(st.session_state.get(stop_dist_key, 0.0) or 0.0)
    if stop_prev_key not in st.session_state:
        st.session_state[stop_prev_key] = float(st.session_state.get(stop_key, selected_stop) or selected_stop or 0.0)

    prev_source = st.session_state.get(prev_source_key)
    if prev_source != selected_source:
        st.session_state[entry_key] = float(selected_entry if selected_entry > 0 else current_price)
        st.session_state[stop_key] = float(selected_stop)
        st.session_state[target_key] = float(selected_target)
        st.session_state[confirm_entry_key] = float(selected_entry if selected_entry > 0 else current_price)
        st.session_state[confirm_stop_key] = float(selected_stop)
        st.session_state[confirm_target_key] = float(selected_target)
        _entry = float(st.session_state.get(entry_key, 0) or 0)
        _stop = float(st.session_state.get(stop_key, 0) or 0)
        _dist = ((_entry - _stop) / _entry * 100) if _entry > 0 and _stop > 0 and _entry > _stop else 0.0
        st.session_state[stop_dist_key] = round(max(0.0, _dist), 2)
        st.session_state[stop_dist_prev_key] = float(st.session_state[stop_dist_key])
        st.session_state[stop_prev_key] = float(_stop)
        st.session_state[prev_source_key] = selected_source

    if st.button("↺ Apply Selected Defaults", key=f"apply_defaults_{ticker}"):
        st.session_state[entry_key] = float(selected_entry if selected_entry > 0 else current_price)
        st.session_state[stop_key] = float(selected_stop)
        st.session_state[target_key] = float(selected_target)
        st.session_state[confirm_entry_key] = float(selected_entry if selected_entry > 0 else current_price)
        st.session_state[confirm_stop_key] = float(selected_stop)
        st.session_state[confirm_target_key] = float(selected_target)
        _entry = float(st.session_state.get(entry_key, 0) or 0)
        _stop = float(st.session_state.get(stop_key, 0) or 0)
        _dist = ((_entry - _stop) / _entry * 100) if _entry > 0 and _stop > 0 and _entry > _stop else 0.0
        st.session_state[stop_dist_key] = round(max(0.0, _dist), 2)
        st.session_state[stop_dist_prev_key] = float(st.session_state[stop_dist_key])
        st.session_state[stop_prev_key] = float(_stop)
        st.rerun()

    if recommended_stop_ref > 0:
        st.caption(f"Recommended Stop Loss ({recommended_stop_src}): ${recommended_stop_ref:.2f}")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        entry_price = st.number_input("Entry Price", step=0.01, format="%.2f", key=entry_key)
    with col2:
        stop_price = st.number_input("Stop Loss", step=0.01, format="%.2f", key=stop_key)
    with col3:
        target_price = st.number_input("Target", step=0.01, format="%.2f", key=target_key)
    with col4:
        stop_dist_pct = st.number_input(
            "Stop Distance %",
            min_value=0.10,
            max_value=50.00,
            step=0.10,
            format="%.2f",
            key=stop_dist_key,
            help="Percent distance from Entry. Changing this recalculates Stop Loss.",
        )
    with col5:
        max_risk = st.number_input(
            "Max Risk %",
            value=max(0.5, min(5.0, round(1.5 * policy.position_size_multiplier, 2))),
            min_value=0.5,
            max_value=5.0,
            step=0.25,
            format="%.2f",
            key=f"sizer_risk_{ticker}",
        )

    prev_stop_dist_pct = float(st.session_state.get(stop_dist_prev_key, stop_dist_pct) or 0.0)
    if abs(stop_dist_pct - prev_stop_dist_pct) > 1e-9 and entry_price > 0:
        new_stop_from_pct = max(0.01, entry_price * (1 - (stop_dist_pct / 100.0)))
        st.session_state[stop_key] = round(new_stop_from_pct, 2)
        st.session_state[confirm_stop_key] = round(new_stop_from_pct, 2)
        st.session_state[stop_prev_key] = float(st.session_state[stop_key])
        st.session_state[stop_dist_prev_key] = float(stop_dist_pct)
        st.rerun()

    prev_stop_price = float(st.session_state.get(stop_prev_key, stop_price) or 0.0)
    if abs(stop_price - prev_stop_price) > 1e-9 and entry_price > 0:
        derived_stop_dist_pct = ((entry_price - stop_price) / entry_price * 100.0) if stop_price > 0 else 0.0
        derived_stop_dist_pct = round(max(0.0, derived_stop_dist_pct), 2)
        st.session_state[stop_dist_key] = derived_stop_dist_pct
        st.session_state[stop_dist_prev_key] = float(derived_stop_dist_pct)
        st.session_state[stop_prev_key] = float(stop_price)

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

        st.markdown("---")
        if result.recommended_shares > 0:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("✅ Recommended", f"{result.recommended_shares:,} shares", f"${result.position_cost:,.0f}")
            r2.metric("💸 Risk", f"${result.risk_dollars:,.0f}", f"{result.risk_pct_of_equity:.1f}% of equity")
            r3.metric(
                "🔥 Heat",
                f"{result.portfolio_heat_before:.1f}% → {result.portfolio_heat_after:.1f}%",
                f"{'✅' if result.portfolio_heat_after < 8 else '⚠️'}",
            )
            if result.reward_risk_ratio > 0:
                r4.metric("🎯 R:R", f"{result.reward_risk_ratio:.1f}:1", "Good ✅" if result.reward_risk_ratio >= 2 else "Low ⚠️")
            else:
                r4.metric("📊 Concentration", f"{result.concentration_pct:.1f}%", f"{'✅' if result.concentration_pct < 20 else '⚠️'}")

            with st.expander("📊 Sizing Breakdown"):
                st.caption(f"**Risk limit (1.5%):** {result.shares_from_risk:,} shares")
                st.caption(f"**Heat limit (8%):** {result.shares_from_heat:,} shares")
                st.caption(f"**Concentration (20%):** {result.shares_from_concentration:,} shares")
                st.caption(f"**Available capital:** {result.shares_from_capital:,} shares")
                st.caption(f"**Binding constraint:** {result.limiting_factor.replace('_', ' ').title()}")
                if result.scale_factor < 1.0:
                    st.warning(
                        f"⚠️ Base size: {result.base_shares:,} shares → "
                        f"Reduced to {result.recommended_shares:,} — {result.scale_reason}"
                    )
                st.caption(f"Win rate (last 20): {win_rate:.0%} | Losing streak: {losing_streak}")

            for w in result.warnings:
                st.warning(w)
            if not result.warnings:
                st.success("✅ Position sizing within all risk parameters")
            st.session_state[f'sizer_result_{ticker}'] = result.recommended_shares
        else:
            st.error(result.explanation)
    elif entry_price > 0 and stop_price > 0 and stop_price >= entry_price:
        st.error("Stop price must be below entry price")

    st.divider()
    st.markdown("### ✅ Enter Trade")

    sizer_shares = st.session_state.get(f'sizer_result_{ticker}', 0)
    final_shares = sizer_shares if sizer_shares > 0 else (int(account_size * 0.125 / entry_price) if entry_price > 0 else 0)

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        confirm_shares = st.number_input("Shares", value=final_shares, min_value=0, step=1, key=f"confirm_shares_{ticker}")
    with ec2:
        if confirm_entry_key not in st.session_state:
            st.session_state[confirm_entry_key] = float(entry_price if entry_price > 0 else selected_entry)
        confirm_entry = st.number_input("Entry $", step=0.01, format="%.2f", key=confirm_entry_key)
    with ec3:
        if confirm_stop_key not in st.session_state:
            st.session_state[confirm_stop_key] = float(stop_price if stop_price > 0 else selected_stop)
        confirm_stop = st.number_input("Stop $", step=0.01, format="%.2f", key=confirm_stop_key)
    with ec4:
        if confirm_target_key not in st.session_state:
            st.session_state[confirm_target_key] = float(target_price if target_price > 0 else selected_target)
        confirm_target = st.number_input("Target $", step=0.01, format="%.2f", key=confirm_target_key)

    if confirm_shares > 0 and confirm_entry > 0:
        total_cost = confirm_shares * confirm_entry
        st.caption(
            f"**{confirm_shares} shares × ${confirm_entry:.2f} = ${total_cost:,.0f}** "
            f"({total_cost / account_size * 100:.1f}% of account)"
        )

    notes_key = f"notes_{ticker}"
    finder_note_applied_key = f"_tf_note_applied_{ticker}"
    tf_run_id = str(finder_for_ticker.get('trade_finder_run_id', '') or '').strip()
    if finder_for_ticker and not st.session_state.get(finder_note_applied_key):
        tf_reason = str(finder_for_ticker.get('ai_rationale', '') or finder_for_ticker.get('reason', '') or '').strip()
        tf_rr = float(finder_for_ticker.get('risk_reward', 0) or 0)
        tf_rec = str(finder_for_ticker.get('ai_buy_recommendation', '') or '').strip()
        seed = rec.get('summary', '') or ''
        tf_line = f"[Trade Finder {tf_run_id or 'run:unknown'}] {tf_rec} | R:R {tf_rr:.2f}:1 | {tf_reason}".strip()
        st.session_state[notes_key] = f"{seed}\n{tf_line}".strip() if seed else tf_line
        st.session_state[finder_note_applied_key] = True
    notes = st.text_area("Notes", value=rec.get('summary', ''), height=90, key=notes_key)
    attach_ticket = st.checkbox("Attach AI trade ticket to notes", value=True, key=f"attach_trade_ticket_{ticker}")
    quality_settings = _trade_quality_settings()
    min_rr_required = float(quality_settings.get('min_rr', 2.0) or 2.0)
    earnings_block_days = int(quality_settings.get('earn_block_days', 7) or 7)
    confirm_rr = _calc_rr(float(confirm_entry or 0), float(confirm_stop or 0), float(confirm_target or 0))

    ticker_earn_days = 999
    try:
        ef = st.session_state.get('earnings_flags', {}).get(ticker, {}) or {}
        ticker_earn_days = int(ef.get('days_until', ef.get('days_until_earnings', 999)) or 999)
    except Exception:
        ticker_earn_days = 999
    finder_earn_days = int(finder_for_ticker.get('earn_days', 999) or 999) if finder_for_ticker else 999
    effective_earn_days = min(ticker_earn_days, finder_earn_days)

    st.caption(
        f"Entry guards: min R:R {min_rr_required:.1f} | block earnings <= {earnings_block_days}d | "
        f"current R:R {confirm_rr:.2f}"
    )
    earnings_override = st.checkbox(
        f"Override earnings window block (earnings <= {earnings_block_days}d)",
        value=False,
        key=f"earnings_override_{ticker}",
        help="Allows entry despite imminent earnings. Use only for intentional earnings plays.",
    )

    if st.button("✅ Enter Trade", type="primary", key=f"enter_{ticker}", disabled=hard_block_stale):
        trades_today = _count_today_trade_entries()
        if hard_block_stale:
            st.error(
                "Blocked: critical stale data streams. Run Fast Refresh and recheck before entering."
            )
        elif confirm_stop <= 0 or confirm_stop >= confirm_entry:
            st.error("Blocked: stop must be below entry and greater than 0.")
        elif confirm_target <= confirm_entry:
            st.error("Blocked: target must be above entry.")
        elif confirm_rr < min_rr_required:
            st.error(f"Blocked: R:R {confirm_rr:.2f} is below minimum {min_rr_required:.1f}.")
        elif 0 <= effective_earn_days <= earnings_block_days and not earnings_override:
            st.error(
                f"Blocked: earnings in {effective_earn_days} day(s). "
                "Enable override only if this is an intentional earnings trade."
            )
        elif not gate.allow_new_trades:
            st.error(f"Blocked: {gate.label}. {gate.reason}")
        elif len(open_trades) >= policy.max_total_open_positions:
            st.error(f"Blocked by regime policy: max open positions reached ({policy.max_total_open_positions}).")
        elif trades_today >= policy.max_new_trades:
            st.error(f"Blocked by regime policy: max new trades today reached ({policy.max_new_trades}).")
        elif confirm_entry <= 0 or confirm_shares <= 0:
            st.error("Set entry price and shares first")
        elif confirm_stop <= 0:
            st.error("Set a stop loss — never trade without a stop")
        elif confirm_stop >= confirm_entry:
            st.error("Stop must be below entry price")
        else:
            pos_size = confirm_shares * confirm_entry
            final_notes = notes.strip()
            if attach_ticket:
                ticket = _build_trade_ticket_note(
                    ticker=ticker,
                    ai_result=ai_result,
                    rec=rec,
                    signal=signal,
                    source_mode=selected_source,
                    entry_price=float(confirm_entry),
                    stop_price=float(confirm_stop),
                    target_price=float(confirm_target),
                )
                final_notes = f"{final_notes}\n\n{ticket}".strip() if final_notes else ticket

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
                    regime_at_entry=str(snap.regime or ''),
                    gate_status_at_entry=str(gate.status or ''),
                    vix_at_entry=float(snap.market_filter.get('vix_close', 0) or 0),
                    entry_source='trade_finder' if finder_for_ticker else 'scanner',
                    trade_finder_run_id=tf_run_id,
                    risk_per_share=confirm_entry - confirm_stop,
                    risk_pct=((confirm_entry - confirm_stop) / confirm_entry * 100) if confirm_entry > 0 else 0,
                    notes=final_notes,
                )
            result = jm.enter_trade(trade)
            st.success(result)
            try:
                plan_id = str((finder_for_ticker or {}).get('plan_id', '')).strip()
                if plan_id:
                    jm.update_planned_trade_status(plan_id, "ENTERED", notes=f"Entered {ticker} @ {confirm_entry:.2f}")
            except Exception:
                pass
                _append_audit_event(
                    "ENTER_TRADE",
                    (
                        f"{ticker} entry={confirm_entry:.2f} stop={confirm_stop:.2f} "
                        f"target={confirm_target:.2f} shares={confirm_shares} "
                        f"regime={policy.regime} source={'trade_finder' if finder_for_ticker else 'scanner'} "
                        f"tf_run={tf_run_id or 'n/a'}"
                    ),
                    source="trade_entry",
                )
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

    def _compute_spy_alpha(details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-trade SPY benchmark returns and alpha in UI layer."""
        if not details:
            return {'details': [], 'avg_alpha_pct': 0.0, 'total_alpha_pct': 0.0, 'beat_rate': 0.0}
        try:
            from data_fetcher import fetch_historical_data
        except Exception:
            return {'details': details, 'avg_alpha_pct': 0.0, 'total_alpha_pct': 0.0, 'beat_rate': 0.0}

        min_entry = None
        max_exit = None
        for t in details:
            e = str(t.get('entry_date', '') or '')
            x = str(t.get('exit_date', '') or '')
            if not e or not x:
                continue
            if min_entry is None or e < min_entry:
                min_entry = e
            if max_exit is None or x > max_exit:
                max_exit = x
        if not min_entry or not max_exit:
            return {'details': details, 'avg_alpha_pct': 0.0, 'total_alpha_pct': 0.0, 'beat_rate': 0.0}

        spy_map = {}
        try:
            spy_df = fetch_historical_data("SPY", min_entry, max_exit, interval='1d')
            if isinstance(spy_df, pd.DataFrame) and not spy_df.empty and 'Close' in spy_df.columns:
                for idx, row in spy_df.iterrows():
                    try:
                        spy_map[idx.strftime('%Y-%m-%d')] = float(row['Close'])
                    except Exception:
                        pass
        except Exception:
            spy_map = {}

        def _nearest(date_str: str, forward: bool = True):
            if date_str in spy_map:
                return spy_map.get(date_str)
            if not spy_map:
                return None
            keys = sorted(spy_map.keys())
            if forward:
                for k in keys:
                    if k >= date_str:
                        return spy_map.get(k)
            else:
                for k in reversed(keys):
                    if k <= date_str:
                        return spy_map.get(k)
            return None

        out = []
        alphas = []
        beats = 0
        for t in details:
            tr = float(t.get('realized_pnl_pct', 0) or 0)
            e = str(t.get('entry_date', '') or '')
            x = str(t.get('exit_date', '') or '')
            se = _nearest(e, forward=True) if e else None
            sx = _nearest(x, forward=False) if x else None
            spy_ret = 0.0
            if se and sx and se > 0:
                spy_ret = round((sx - se) / se * 100.0, 2)
            alpha = round(tr - spy_ret, 2)
            if alpha > 0:
                beats += 1
            alphas.append(alpha)
            row = dict(t)
            row['spy_return_pct'] = spy_ret
            row['alpha_pct'] = alpha
            out.append(row)
        return {
            'details': out,
            'avg_alpha_pct': round(sum(alphas) / len(alphas), 2) if alphas else 0.0,
            'total_alpha_pct': round(sum(alphas), 2) if alphas else 0.0,
            'beat_rate': round(beats / len(out) * 100, 1) if out else 0.0,
        }

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

    enriched = _compute_spy_alpha(stats.get('trade_details', []) or [])
    stats['avg_alpha_pct'] = enriched.get('avg_alpha_pct', 0.0)
    stats['total_alpha_pct'] = enriched.get('total_alpha_pct', 0.0)
    stats['beat_benchmark_rate'] = enriched.get('beat_rate', 0.0)
    details = enriched.get('details', []) or stats.get('trade_details', []) or []

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Expectancy", f"{stats.get('expectancy_pct', 0):+.2f}%")
    col10.metric("Payoff Ratio", f"{stats.get('payoff_ratio', 0):.2f}")
    col11.metric("Avg Win / Loss", f"{stats.get('avg_win_pct', 0):+.1f}% / {stats.get('avg_loss_pct', 0):+.1f}%")
    col12.metric("Avg Alpha vs SPY", f"{stats.get('avg_alpha_pct', 0):+.2f}%", f"Beat rate {stats.get('beat_benchmark_rate', 0):.1f}%")

    # By signal type
    if stats['by_signal_type']:
        st.divider()
        st.subheader("By Signal Type")
        for st_name, st_data in stats['by_signal_type'].items():
            st.text(f"  {st_name}: {st_data['count']} trades, "
                    f"Win rate: {st_data['win_rate']:.0f}%, "
                    f"Avg: {st_data['avg_pnl_pct']:+.1f}%")

    # By ticker
    by_ticker = stats.get('by_ticker', {}) or {}
    if by_ticker:
        st.divider()
        st.subheader("By Ticker")
        rows_tk = []
        for tk, d in by_ticker.items():
            rows_tk.append({
                'Ticker': tk,
                'Trades': d.get('count', 0),
                'Win Rate %': d.get('win_rate', 0),
                'Avg P&L %': d.get('avg_pnl_pct', 0),
            })
        rows_tk = sorted(rows_tk, key=lambda r: (r['Avg P&L %'], r['Win Rate %']), reverse=True)
        st.dataframe(pd.DataFrame(rows_tk), hide_index=True, width='stretch')

    by_regime = stats.get('by_regime', {}) or {}
    if by_regime:
        st.divider()
        st.subheader("By Regime At Entry")
        rows_rg = []
        for rg, d in by_regime.items():
            rows_rg.append({
                'Regime': rg,
                'Trades': d.get('count', 0),
                'Win Rate %': d.get('win_rate', 0),
                'Avg P&L %': d.get('avg_pnl_pct', 0),
            })
        rows_rg = sorted(rows_rg, key=lambda r: (r['Avg P&L %'], r['Win Rate %']), reverse=True)
        st.dataframe(pd.DataFrame(rows_rg), hide_index=True, width='stretch')

    by_src = stats.get('by_entry_source', {}) or {}
    if by_src:
        st.divider()
        st.subheader("By Entry Source")
        rows_src = []
        for src, d in by_src.items():
            rows_src.append({
                'Source': src,
                'Trades': d.get('count', 0),
                'Win Rate %': d.get('win_rate', 0),
                'Avg P&L %': d.get('avg_pnl_pct', 0),
            })
        rows_src = sorted(rows_src, key=lambda r: (r['Avg P&L %'], r['Win Rate %']), reverse=True)
        st.dataframe(pd.DataFrame(rows_src), hide_index=True, width='stretch')

    by_tf_run = stats.get('by_trade_finder_run', {}) or {}
    if by_tf_run:
        st.divider()
        st.subheader("Trade Finder Attribution (By Run)")
        rows_run = []
        for run_id, d in by_tf_run.items():
            rows_run.append({
                'Run ID': run_id,
                'Trades': d.get('count', 0),
                'Win Rate %': d.get('win_rate', 0),
                'Avg P&L %': d.get('avg_pnl_pct', 0),
            })
        rows_run = sorted(rows_run, key=lambda r: (r['Avg P&L %'], r['Win Rate %']), reverse=True)
        st.dataframe(pd.DataFrame(rows_run), hide_index=True, width='stretch')

    # Underperforming buckets
    under = stats.get('underperforming_buckets', []) or []
    if under:
        st.divider()
        st.subheader("What To Stop Doing")
        st.warning("Buckets below have negative expectancy over meaningful sample size (>=5). Reduce size or pause them.")
        under_rows = []
        for u in under:
            under_rows.append({
                'Bucket': u.get('bucket', ''),
                'Trades': u.get('count', 0),
                'Avg P&L %': u.get('avg_pnl_pct', 0),
                'Win Rate %': u.get('win_rate', '—') if u.get('win_rate', None) is not None else '—',
            })
        st.dataframe(pd.DataFrame(under_rows), hide_index=True, width='stretch')

    # Trade history table
    history = details[:20] if details else jm.get_trade_history(last_n=20)
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
                'SPY': f"{float(t.get('spy_return_pct', 0) or 0):+.1f}%" if 'spy_return_pct' in t else "—",
                'Alpha': f"{float(t.get('alpha_pct', 0) or 0):+.1f}%" if 'alpha_pct' in t else "—",
                'Days': t.get('days_held', 0),
                'Reason': t.get('exit_reason', '?'),
                'Signal': t.get('signal_type', '?'),
                'Source': t.get('entry_source', t.get('source', '?')),
                'TF Run': t.get('trade_finder_run_id', ''),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


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

    # Macro posture for managing EXISTING positions
    try:
        snap = _build_dashboard_snapshot()
        gate = _evaluate_trade_gate(snap)
        posture = _position_posture_summary(snap, gate)
        if posture['severity'] == "danger":
            st.error(f"{posture['headline']} — {posture['summary']}")
        elif posture['severity'] == "warning":
            st.warning(f"{posture['headline']} — {posture['summary']}")
        else:
            st.success(f"{posture['headline']} — {posture['summary']}")
        st.caption(f"Derived from Unified Regime {snap.regime} ({snap.regime_confidence}%) and Trade Gate {gate.status}.")
    except Exception:
        pass
    _last_exit_summary = st.session_state.get('_last_exit_analysis_summary', {}) or {}
    if _last_exit_summary:
        _cts = _last_exit_summary.get('action_counts', {}) or {}
        _order = ['HOLD', 'TIGHTEN_STOP', 'TAKE_PARTIAL', 'CLOSE']
        _parts = [f"{k}:{int(_cts.get(k, 0))}" for k in _order if int(_cts.get(k, 0)) > 0]
        _parts_txt = " | ".join(_parts) if _parts else "No actions"
        st.caption(
            f"Last exit analysis {_last_exit_summary.get('ts', '')}: "
            f"analyzed {int(_last_exit_summary.get('analyzed', 0))}/{int(_last_exit_summary.get('requested', 0))} "
            f"| {_parts_txt} | system fallback {int(_last_exit_summary.get('fallback_count', 0))}"
        )
        _skipped = list(_last_exit_summary.get('skipped_tickers', []) or [])
        if _skipped:
            st.caption(f"Skipped (no live price): {', '.join(_skipped[:8])}")

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
        if st.button("🤖 Analyze All Positions", type="primary", width="stretch"):
            _run_exit_analysis(open_trades)

    with an_col2:
        if st.button("📧 Analyze + Email Report", width="stretch"):
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

            ai_clients = _get_ai_clients()
            gemini_model = ai_clients.get('gemini')
            openai_client = ai_clients.get('openai_client')
            ai_cfg = ai_clients.get('ai_config', {}) or {}
            ai_model = ai_cfg.get('model', 'llama-3.3-70b-versatile')
            fallback_model = ai_cfg.get('fallback_model', '')

            if not openai_client and not gemini_model:
                primary_err = ai_clients.get('primary_error') or "primary unavailable"
                gemini_err = ai_clients.get('gemini_error') or "gemini unavailable"
                st.warning(
                    "AI providers unavailable for exit analysis. "
                    f"Primary: {primary_err}. Fallback: {gemini_err}. "
                    "System HOLD fallback will be used."
                )

            advices = analyze_all_positions(
                open_trades,
                fetch_price_fn=fetch_current_price,
                fetch_signal_fn=fetch_signal_for_exit,
                gemini_model=gemini_model,
                openai_client=openai_client,
                ai_model=ai_model,
                fallback_model=fallback_model,
            )
            analyzed_tickers = {str(a.ticker or '').upper().strip() for a in advices}
            requested_tickers = [str(t.get('ticker', '')).upper().strip() for t in open_trades if str(t.get('ticker', '')).strip()]
            skipped_tickers = [t for t in requested_tickers if t and t not in analyzed_tickers]

            # Store results in session state for display
            for advice in advices:
                st.session_state[f'exit_advice_{advice.ticker}'] = advice.to_dict()

            # Store quick summary for Position Manager visibility.
            action_counts: Dict[str, int] = {}
            fallback_count = 0
            for advice in advices:
                a = str(getattr(advice, 'action', '') or 'UNKNOWN')
                action_counts[a] = int(action_counts.get(a, 0) + 1)
                if str(getattr(advice, 'provider', '') or '').lower() == 'system':
                    fallback_count += 1
            st.session_state['_last_exit_analysis_summary'] = {
                'requested': len(requested_tickers),
                'analyzed': len(advices),
                'skipped_tickers': skipped_tickers,
                'action_counts': action_counts,
                'fallback_count': fallback_count,
                'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

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
            if skipped_tickers:
                st.warning(
                    f"Skipped {len(skipped_tickers)} position(s) due to missing live price data: "
                    f"{', '.join(skipped_tickers[:8])}"
                )

            st.rerun()

        except Exception as e:
            st.error(f"Exit analysis error: {e}")


# =============================================================================
# EXECUTIVE DASHBOARD
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
    status: str  # NO_TRADE | TRADE_LIGHT | FAVOR_TRADING
    label: str
    allow_new_trades: bool
    severity: str  # danger | warning | success
    reason: str
    model_alignment: str  # ALIGNED | DIVERGENT | PARTIAL


def _is_exec_dashboard_enabled() -> bool:
    """
    Feature flag for phased rollout.
    Priority: session override > secrets > enabled.
    """
    if 'exec_dashboard_beta_enabled' in st.session_state:
        return bool(st.session_state['exec_dashboard_beta_enabled'])
    try:
        raw = str(st.secrets.get('EXEC_DASHBOARD_BETA', 'true')).strip().lower()
        return raw in {'1', 'true', 'yes', 'on'}
    except Exception:
        return True


def _infer_exec_regime(market_filter: Dict[str, Any], sector_rotation: Dict[str, Any]) -> (str, int):
    """Unified regime inference for dashboard/risk policy."""
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


def _risk_budget_for_regime(regime: str) -> RiskBudgetPolicy:
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


def _build_dashboard_snapshot() -> DashboardSnapshot:
    """Create single dashboard source-of-truth snapshot."""
    jm = get_journal()
    bridge = get_bridge()
    now = time.time()
    scan_summary = st.session_state.get('scan_results_summary', []) or []
    market_filter = st.session_state.get('market_filter_data', {}) or {}
    sector_rotation = st.session_state.get('sector_rotation', {}) or {}
    earnings_flags = st.session_state.get('earnings_flags', {}) or {}

    regime, confidence = _infer_exec_regime(market_filter, sector_rotation)
    policy = _risk_budget_for_regime(regime)

    return DashboardSnapshot(
        generated_at=now,
        generated_at_iso=datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
        watchlist_tickers=bridge.get_watchlist_tickers(),
        scan_summary=scan_summary,
        open_trades=jm.get_open_trades(),
        pending_alerts=jm.get_pending_conditionals(),
        triggered_alerts=st.session_state.get('triggered_alerts', []) or [],
        market_filter=market_filter,
        sector_rotation=sector_rotation,
        earnings_flags=earnings_flags,
        scan_ts=float(st.session_state.get('_scan_run_ts', 0.0) or 0.0),
        market_ts=float(st.session_state.get('_market_filter_ts', 0.0) or 0.0),
        sector_ts=float(st.session_state.get('_sector_rotation_ts', 0.0) or 0.0),
        pos_ts=float(st.session_state.get('_position_prices_ts', 0.0) or 0.0),
        alert_ts=float(st.session_state.get('_alert_check_ts', 0.0) or 0.0),
        workflow_ts=float(st.session_state.get('_daily_workflow_ts', 0.0) or 0.0),
        workflow_sec=float(st.session_state.get('_daily_workflow_sec', 0.0) or 0.0),
        regime=regime,
        regime_confidence=confidence,
        risk_policy=policy,
    )


def _normalize_brief_regime(regime_text: str) -> str:
    r = str(regime_text or "").lower()
    if "risk-off" in r or "bearish" in r:
        return "RISK_OFF"
    if "neutral" in r or "balanced" in r or "caution" in r:
        return "TRANSITION"
    if "risk-on" in r or "bullish" in r:
        return "RISK_ON"
    return "UNKNOWN"


def _evaluate_trade_gate(snap: DashboardSnapshot) -> TradeGateDecision:
    """
    Single execution authority for whether new trades should be taken now.
    """
    vix = float(snap.market_filter.get('vix_close', 0) or 0)
    spy_ok = bool(snap.market_filter.get('spy_above_200', True))

    stale_scan = _is_stale(snap.scan_ts, 3 * 3600)
    stale_market = _is_stale(snap.market_ts, 30 * 60)
    stale_sector = _is_stale(snap.sector_ts, 60 * 60)
    stale_positions = _is_stale(snap.pos_ts, 10 * 60)
    stale_alerts = _is_stale(snap.alert_ts, 5 * 60)
    stale_count = sum([stale_scan, stale_market, stale_sector, stale_positions, stale_alerts])

    deep_score = 0
    deep = st.session_state.get('deep_market_analysis') or {}
    if isinstance(deep, dict):
        deep_score = int(deep.get('score', 0) or 0)
    brief = st.session_state.get('morning_narrative') or {}
    brief_regime = _normalize_brief_regime(brief.get('regime', ''))

    aligned = 0
    if brief_regime == "UNKNOWN":
        aligned += 0
    elif brief_regime == snap.regime:
        aligned += 1
    if (deep_score >= 1 and snap.regime == "RISK_ON") or (deep_score <= -1 and snap.regime == "RISK_OFF"):
        aligned += 1
    elif deep_score == 0 and snap.regime in {"TRANSITION", "DEFENSIVE"}:
        aligned += 1

    if aligned >= 2:
        model_alignment = "ALIGNED"
    elif aligned == 1:
        model_alignment = "PARTIAL"
    else:
        model_alignment = "DIVERGENT"

    # Base decision from unified regime + volatility
    if stale_count >= 3:
        status = "NO_TRADE"
        reason = "Core data is stale. Refresh before opening new trades."
    elif snap.regime == "RISK_OFF" or vix >= 25:
        status = "NO_TRADE"
        reason = f"Risk-off environment (VIX {vix:.1f}). Preserve capital."
    elif snap.regime in {"DEFENSIVE", "TRANSITION"} or vix >= 20 or not spy_ok:
        status = "TRADE_LIGHT"
        reason = f"Mixed/cautious regime with elevated risk (VIX {vix:.1f})."
    else:
        status = "FAVOR_TRADING"
        reason = f"Benign risk backdrop and supportive trend (VIX {vix:.1f})."

    # Downgrade one notch when models diverge materially.
    if model_alignment == "DIVERGENT":
        if status == "FAVOR_TRADING":
            status = "TRADE_LIGHT"
            reason += " Downgraded due to model divergence."
        elif status == "TRADE_LIGHT":
            status = "NO_TRADE"
            reason += " Downgraded due to model divergence."

    if status == "NO_TRADE":
        return TradeGateDecision(
            status=status,
            label="🛑 TOO RISKY TO TRADE",
            allow_new_trades=False,
            severity="danger",
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


def _position_posture_summary(snap: DashboardSnapshot, gate: TradeGateDecision) -> Dict[str, Any]:
    """
    Clear macro guidance for EXISTING positions, separate from new-entry gate status.
    """
    stop_breaches = 0
    drawdown_count = 0
    for trade in (snap.open_trades or []):
        ticker = str(trade.get('ticker', '')).upper().strip()
        entry = float(trade.get('entry_price', 0) or 0)
        stop = float(trade.get('current_stop', trade.get('initial_stop', 0)) or 0)
        current = float(st.session_state.get('_position_prices', {}).get(ticker) or entry)
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        if stop > 0 and current <= stop:
            stop_breaches += 1
        elif pnl_pct <= -3:
            drawdown_count += 1

    vix = float(snap.market_filter.get('vix_close', 0) or 0)
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


def _fmt_last_update(ts: float, fallback: str = "Never") -> str:
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


def _is_stale(ts: float, max_age_sec: int) -> bool:
    """True if timestamp is missing or older than max_age_sec."""
    if not ts:
        return True
    return (time.time() - ts) > max_age_sec


def _scan_health_snapshot() -> Dict[str, float]:
    """Compute scan telemetry from audit events + session metrics."""
    events = _get_audit_events()
    scan_done = [e for e in events if e.get('action') == 'SCAN_DONE']
    scan_start = [e for e in events if e.get('action') == 'SCAN_START']

    last_mode = ""
    last_count = 0
    if scan_done:
        details = str(scan_done[0].get('details', ''))
        m_mode = re.search(r'mode=([a-z_]+)', details)
        m_count = re.search(r'scanned=(\d+)', details)
        if m_mode:
            last_mode = m_mode.group(1)
        if m_count:
            last_count = int(m_count.group(1))

    dur_hist = st.session_state.get('_scan_duration_hist', [])
    last_dur = float(dur_hist[-1]) if dur_hist else 0.0
    avg_dur = (sum(dur_hist) / len(dur_hist)) if dur_hist else 0.0

    return {
        'total_scans_logged': float(len(scan_done)),
        'scan_starts_logged': float(len(scan_start)),
        'last_scan_count': float(last_count),
        'last_scan_duration': last_dur,
        'avg_scan_duration': avg_dur,
        'last_scan_mode': last_mode,
    }


def _recommendation_score(row: Dict) -> float:
    """Simple ranking score for trade candidates."""
    grade_rank = {'A+': 8, 'A': 7, 'A-': 6, 'B+': 5, 'B': 4, 'B-': 3, 'C+': 2, 'C': 1}
    rec = str(row.get('recommendation', '')).upper()
    conv = float(row.get('conviction', 0) or 0)
    vol_ratio = float(row.get('volume_ratio', 0) or 0)
    grade = str(row.get('quality_grade', '')).upper()
    sector_phase = str(row.get('sector_phase', '')).lower()
    earn_days = int(row.get('earn_days', 999) or 999)

    score = conv * 3 + min(vol_ratio, 4) * 2 + grade_rank.get(grade, 0)
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
    return score


def _score_candidate_with_policy(row: Dict, snap: DashboardSnapshot) -> Dict[str, Any]:
    """Rank candidate and capture explainability tags + regime adjustments."""
    score = _recommendation_score(row)
    reasons: List[str] = []
    rec = str(row.get('recommendation', '')).upper()
    conv = int(row.get('conviction', 0) or 0)
    phase = str(row.get('sector_phase', '')).lower()
    earn_days = int(row.get('earn_days', 999) or 999)

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

    if earn_days <= 3:
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

    return {
        'score': score,
        'reasons': reasons,
        'blocked': blocked,
    }


def _run_alert_check_now(jm: JournalManager) -> int:
    """Evaluate pending conditionals immediately using current prices."""
    conditionals = jm.get_pending_conditionals()
    if not conditionals:
        return 0

    current_prices = {}
    for cond in conditionals:
        ticker = cond.get('ticker', '').upper().strip()
        if ticker and ticker not in current_prices:
            price = fetch_current_price(ticker)
            if price and price > 0:
                current_prices[ticker] = float(price)

    triggered = jm.check_conditionals(current_prices, volume_ratios={})
    if triggered:
        st.session_state['triggered_alerts'] = triggered
    st.session_state['_alert_check_ts'] = time.time()
    _append_audit_event("ALERT_CHECK", f"pending={len(conditionals)} triggered={len(triggered)}", source="exec_dashboard")
    return len(triggered)


def _run_auto_exit_engine(jm: JournalManager, current_prices: Dict[str, float], source: str = "manual") -> Dict[str, Any]:
    """
    Evaluate open positions against stop/target and auto-close breaches.
    Returns summary for UI + telemetry.
    """
    if not current_prices:
        return {'checked': 0, 'triggered': 0, 'closed': 0, 'events': []}

    open_trades = jm.get_open_trades()
    checked = len(open_trades)
    events = jm.check_stops(current_prices, auto_execute=True)
    closed = len(events)
    stop_hits = sum(1 for e in events if str(e.get('trigger', '')) == 'stop_loss')
    target_hits = sum(1 for e in events if str(e.get('trigger', '')) == 'target_hit')

    st.session_state['_auto_exit_last_ts'] = time.time()
    st.session_state['_auto_exit_last_count'] = closed

    if closed > 0:
        tickers = ",".join(sorted({str(e.get('ticker', '')).upper() for e in events if e.get('ticker')}))
        _append_audit_event(
            "AUTO_EXIT",
            f"source={source} checked={checked} closed={closed} stops={stop_hits} targets={target_hits} tickers={tickers}",
            source="auto_exit",
        )
    else:
        _append_audit_event(
            "AUTO_EXIT",
            f"source={source} checked={checked} closed=0",
            source="auto_exit",
        )

    _append_perf_metric({
        "kind": "auto_exit",
        "source": source,
        "checked": checked,
        "closed": closed,
        "stop_hits": stop_hits,
        "target_hits": target_hits,
    })
    return {'checked': checked, 'triggered': closed, 'closed': closed, 'events': events}


def _fast_refresh_dashboard() -> None:
    """Fast refresh for dashboard metrics without running a full scan."""
    _t0 = time.time()
    from data_fetcher import fetch_sector_rotation

    st.session_state['market_filter_data'] = fetch_market_filter()
    st.session_state['_market_filter_ts'] = time.time()

    st.session_state['sector_rotation'] = fetch_sector_rotation()
    st.session_state['_sector_rotation_ts'] = time.time()

    jm = get_journal()
    open_trades = jm.get_open_trades()
    prices = {}
    for trade in open_trades:
        t = trade.get('ticker', '').upper().strip()
        if t:
            prices[t] = fetch_current_price(t) or float(trade.get('entry_price', 0) or 0)
    st.session_state['_position_prices'] = prices
    st.session_state['_position_prices_ts'] = time.time()

    auto_exit_enabled = bool(st.session_state.get('exec_auto_exit_enabled', False))
    auto_exit = {'closed': 0}
    if auto_exit_enabled:
        auto_exit = _run_auto_exit_engine(jm, prices, source="fast_refresh")
        # refresh prices map after closes to keep panel consistent
        refreshed_open = jm.get_open_trades()
        refreshed_prices = {}
        for trade in refreshed_open:
            t = trade.get('ticker', '').upper().strip()
            if t:
                refreshed_prices[t] = fetch_current_price(t) or float(trade.get('entry_price', 0) or 0)
        st.session_state['_position_prices'] = refreshed_prices
        st.session_state['_position_prices_ts'] = time.time()

    trig_count = _run_alert_check_now(jm)
    st.session_state['_dashboard_last_refresh'] = time.time()
    _append_audit_event(
        "FAST_REFRESH",
        f"positions={len(open_trades)} alerts_triggered={trig_count} auto_closed={int(auto_exit.get('closed', 0) or 0)}",
        source="exec_dashboard",
    )
    _append_perf_metric({
        "kind": "fast_refresh",
        "sec": round(time.time() - _t0, 3),
        "positions": len(open_trades),
        "alerts_triggered": trig_count,
        "auto_closed": int(auto_exit.get('closed', 0) or 0),
    })
    st.success(
        f"Fast refresh complete. Triggered alerts: {trig_count}. "
        f"Auto-closed: {int(auto_exit.get('closed', 0) or 0)}."
    )


def _run_daily_workflow() -> Dict[str, float]:
    """
    Execute practical daily cycle:
    1) Fast refresh macro/sector/positions/alerts
    2) Scan only new tickers for speed
    """
    t0 = time.time()
    st.session_state['_daily_workflow_start_ts'] = t0
    st.session_state['_daily_workflow_in_progress'] = True
    _append_audit_event("DAILY_WORKFLOW_START", "fast_refresh + scan_new_only", source="exec_dashboard")
    _fast_refresh_dashboard()
    _run_scan(mode='new_only')
    # Note: _run_scan triggers rerun; completion metrics are finalized in _run_scan.
    return {'elapsed': 0.0}


def render_executive_dashboard():
    """Daily command center for actionable overview and refresh controls."""
    _render_started = time.time()
    jm = get_journal()
    snap = _build_dashboard_snapshot()
    gate = _evaluate_trade_gate(snap)
    dash_ts = float(st.session_state.get('_dashboard_last_refresh', 0.0) or 0.0)

    st.subheader("Executive Dashboard")
    st.caption(f"Now: {snap.generated_at_iso}")
    if 'exec_auto_exit_enabled' not in st.session_state:
        st.session_state['exec_auto_exit_enabled'] = False
    st.checkbox(
        "Enable Auto Exit on Fast Refresh / Daily Workflow",
        key="exec_auto_exit_enabled",
        help="When enabled, stop-loss and target-hit rules auto-close positions during refresh runs.",
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        if st.button("⚡ Fast Refresh", key="exec_fast_refresh", type="primary", width="stretch"):
            _fast_refresh_dashboard()
            st.rerun()
    with c2:
        if st.button("🔍 Full Refresh (Scan All)", key="exec_full_refresh", width="stretch"):
            _run_scan(mode='all')
    with c3:
        if st.button("🎯 Check Alerts Now", key="exec_alert_refresh", width="stretch"):
            trig_count = _run_alert_check_now(jm)
            st.session_state['_alert_check_ts'] = time.time()
            st.success(f"Alert check complete. Triggered alerts: {trig_count}")
            st.rerun()
    with c4:
        if st.button("✅ Run Daily Workflow", key="exec_daily_workflow", width="stretch"):
            _run_daily_workflow()
    with c5:
        if st.button("🛑 Auto-Manage Now", key="exec_auto_manage_now", width="stretch"):
            open_trades = jm.get_open_trades()
            prices = {}
            for trade in open_trades:
                t = trade.get('ticker', '').upper().strip()
                if t:
                    prices[t] = fetch_current_price(t) or float(trade.get('entry_price', 0) or 0)
            res = _run_auto_exit_engine(jm, prices, source="manual_button")
            st.success(f"Auto-manage complete. Checked {res.get('checked', 0)} positions; auto-closed {res.get('closed', 0)}.")
            st.rerun()
    with c6:
        if st.button("🧭 Find New Trades", key="exec_find_new_trades", width="stretch"):
            with st.spinner("Scanning all watchlists and ranking candidates..."):
                _run_trade_finder_workflow()
            st.success("Trade Finder updated from all watchlists.")
            st.rerun()

    last_auto_ts = float(st.session_state.get('_auto_exit_last_ts', 0.0) or 0.0)
    last_auto_count = int(st.session_state.get('_auto_exit_last_count', 0) or 0)
    st.caption(
        f"Auto Exit: {'ON' if st.session_state.get('exec_auto_exit_enabled') else 'OFF'} | "
        f"Last run: {_fmt_last_update(last_auto_ts)} | Last closed: {last_auto_count}"
    )
    tf_last = st.session_state.get('trade_finder_results', {}) or {}
    st.caption(
        f"Trade Finder: {len(tf_last.get('rows', []) or [])} ranked candidate(s) | "
        f"Updated: {tf_last.get('generated_at_iso', 'never') or 'never'}"
    )

    st.caption(
        f"Last fast refresh: {_fmt_last_update(dash_ts)} | "
        f"Scan: {_fmt_last_update(snap.scan_ts)} | "
        f"Market: {_fmt_last_update(snap.market_ts)} | "
        f"Sectors: {_fmt_last_update(snap.sector_ts)} | "
        f"Position prices: {_fmt_last_update(snap.pos_ts)} | "
        f"Alerts check: {_fmt_last_update(snap.alert_ts)} | "
        f"Workflow: {_fmt_last_update(snap.workflow_ts)}"
    )
    if snap.workflow_sec > 0:
        st.caption(f"Last workflow runtime: {snap.workflow_sec:.1f}s")

    candidate_rows = []
    if gate.allow_new_trades:
        find_new = st.session_state.get('find_new_trades_report', {}) or {}
        find_new_cands = find_new.get('candidates', []) or []
        if find_new_cands:
            for c in find_new_cands:
                entry = float(c.get('price', 0) or 0)
                stop = float(c.get('price', 0) or 0) * 0.94 if entry > 0 else 0.0
                target = float(c.get('price', 0) or 0) * 1.10 if entry > 0 else 0.0
                card = build_trade_decision_card(
                    ticker=str(c.get('ticker', '') or ''),
                    source="find_new",
                    recommendation=str(c.get('recommendation', '') or ''),
                    ai_buy_recommendation=str(c.get('recommendation', '') or ''),
                    conviction=int(c.get('conviction', 0) or 0),
                    quality_grade=str(c.get('quality_grade', '?') or '?'),
                    entry=entry,
                    stop=stop,
                    target=target,
                    rank_score=float(c.get('score', 0) or 0),
                    regime=snap.regime,
                    gate_status=gate.status,
                    reason=str(c.get('summary', '') or ''),
                    ai_rationale="",
                    sector_phase=str(c.get('sector_phase', '') or ''),
                    earn_days=int(c.get('earn_days', 999) or 999),
                    explainability_bits=[
                        "source=find_new",
                        f"conv={int(c.get('conviction', 0) or 0)}",
                        f"phase={str(c.get('sector_phase', '') or '')}",
                    ],
                ).to_dict()
                candidate_rows.append({
                    'row': {
                        'ticker': c.get('ticker', ''),
                        'recommendation': c.get('recommendation', ''),
                        'conviction': c.get('conviction', 0),
                    },
                    'score': float(c.get('score', 0) or 0),
                    'reasons': [
                        "Cross-watchlist candidate",
                        f"Quality {c.get('quality_grade', '?')}",
                        f"Sector {c.get('sector_phase', '')}".strip(),
                    ],
                    'card': card,
                })
        else:
            for row in snap.scan_summary:
                rec = str(row.get('recommendation', '')).upper()
                if ('BUY' in rec or 'ENTRY' in rec) and 'SKIP' not in rec and 'AVOID' not in rec:
                    scored = _score_candidate_with_policy(row, snap)
                    if not scored['blocked']:
                        entry = float(row.get('price', 0) or 0)
                        stop = entry * 0.94 if entry > 0 else 0.0
                        target = entry * 1.10 if entry > 0 else 0.0
                        card = build_trade_decision_card(
                            ticker=str(row.get('ticker', '') or ''),
                            source="scanner",
                            recommendation=str(row.get('recommendation', '') or ''),
                            ai_buy_recommendation=str(row.get('recommendation', '') or ''),
                            conviction=int(row.get('conviction', 0) or 0),
                            quality_grade=str(row.get('quality_grade', '?') or '?'),
                            entry=entry,
                            stop=stop,
                            target=target,
                            rank_score=float(scored['score'] or 0),
                            regime=snap.regime,
                            gate_status=gate.status,
                            reason=str(row.get('summary', '') or ''),
                            ai_rationale="",
                            sector_phase=str(row.get('sector_phase', '') or ''),
                            earn_days=int(row.get('earn_days', 999) or 999),
                            explainability_bits=list(scored['reasons'][:3]),
                        ).to_dict()
                        candidate_rows.append({
                            'row': row,
                            'score': scored['score'],
                            'reasons': scored['reasons'],
                            'card': card,
                        })
    candidate_rows = sorted(candidate_rows, key=lambda x: x['score'], reverse=True)
    actionable = candidate_rows[: max(0, snap.risk_policy.max_new_trades)]

    # SLO freshness checks
    stale_scan = _is_stale(snap.scan_ts, 3 * 3600)      # target: <3h old
    stale_market = _is_stale(snap.market_ts, 30 * 60)   # target: <30m old
    stale_sector = _is_stale(snap.sector_ts, 60 * 60)   # target: <60m old
    stale_positions = _is_stale(snap.pos_ts, 10 * 60)   # target: <10m old
    stale_alerts = _is_stale(snap.alert_ts, 5 * 60)     # target: <5m old
    stale_count = sum([stale_scan, stale_market, stale_sector, stale_positions, stale_alerts])

    if stale_count > 0:
        stale_tags = []
        if stale_scan:
            stale_tags.append("scan")
        if stale_market:
            stale_tags.append("market")
        if stale_sector:
            stale_tags.append("sectors")
        if stale_positions:
            stale_tags.append("positions")
        if stale_alerts:
            stale_tags.append("alerts")
        st.warning(f"Stale data detected: {', '.join(stale_tags)}")
    else:
        st.success("All core data streams are fresh.")

    health = _scan_health_snapshot()
    last_timing = st.session_state.get('_timing_last_scan', {}) or {}

    regime_col1, regime_col2 = st.columns(2)
    regime_col1.metric("Regime", f"{snap.regime}", f"{snap.regime_confidence}% confidence")
    regime_col2.caption(
        f"Risk Budget: max new {snap.risk_policy.max_new_trades}, "
        f"size x{snap.risk_policy.position_size_multiplier:.2f}, "
        f"max sector {snap.risk_policy.max_sector_exposure} | {snap.risk_policy.note}"
    )
    if gate.severity == "danger":
        st.error(f"{gate.label} — {gate.reason}")
    elif gate.severity == "warning":
        st.warning(f"{gate.label} — {gate.reason}")
    else:
        st.success(f"{gate.label} — {gate.reason}")
    st.caption(f"Execution Authority: Unified Trade Gate | Model alignment: {gate.model_alignment}")
    posture = _position_posture_summary(snap, gate)
    st.caption(f"Open-position posture: {posture['summary']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Watchlist", len(snap.watchlist_tickers))
    m2.metric("Scanned", len(snap.scan_summary))
    m3.metric("Actionable", len(actionable))
    m4.metric("Open Positions", len(snap.open_trades))
    m5.metric("Alerts", len(snap.pending_alerts), delta=f"{len(snap.triggered_alerts)} triggered")
    m6.metric("Stale Streams", stale_count, delta=f"Last scan {health['last_scan_mode'] or 'n/a'}")

    t1, t2, t3 = st.columns(3)
    t1.metric("Last Scan Count", int(health['last_scan_count']))
    t2.metric("Last Scan Runtime", f"{health['last_scan_duration']:.1f}s")
    t3.metric("Avg Scan Runtime", f"{health['avg_scan_duration']:.1f}s")
    if last_timing:
        st.caption(
            f"Scan timings (sec): data={last_timing.get('fetch_scan_data_sec', 0.0):.1f}, "
            f"analyze={last_timing.get('scan_watchlist_sec', 0.0):.1f}, "
            f"sectors={last_timing.get('sector_refresh_sec', 0.0):.1f}, "
            f"earnings={last_timing.get('earnings_refresh_sec', 0.0):.1f}, "
            f"summary={last_timing.get('summary_build_sec', 0.0):.1f}, "
            f"alerts={last_timing.get('alerts_check_sec', 0.0):.1f}"
        )

    st.divider()
    q1, q2, q3 = st.columns(3)

    must_act = []
    action_queue = []
    planned_trades = jm.get_planned_trades()
    for p in planned_trades:
        plan_id = str(p.get('plan_id', '')).strip()
        pstatus = str(p.get('status', '')).upper()
        pticker = str(p.get('ticker', '')).upper().strip()
        if not pticker:
            continue
        if pstatus == "TRIGGERED":
            must_act.append((92, f"🗂️ Planned triggered: {pticker}", pticker))
            action_queue.append({
                'priority': 92, 'category': 'planned', 'ticker': pticker, 'plan_id': plan_id,
                'message': f"Planned triggered: {pticker}", 'action': 'planned_triggered'
            })
        elif pstatus == "PLANNED":
            must_act.append((55, f"🗂️ Planned queued: {pticker}", pticker))
            action_queue.append({
                'priority': 55, 'category': 'planned', 'ticker': pticker, 'plan_id': plan_id,
                'message': f"Planned queued: {pticker}", 'action': 'planned_queued'
            })
    for t in snap.triggered_alerts:
        ticker = t.get('ticker', '')
        must_act.append((100, f"🎯 Alert triggered: {ticker}", ticker))
        action_queue.append({'priority': 100, 'category': 'alert', 'ticker': ticker, 'message': f"Alert triggered: {ticker}", 'action': 'open_trade'})
    for trade in snap.open_trades:
        ticker = trade.get('ticker', '').upper().strip()
        entry = float(trade.get('entry_price', 0) or 0)
        stop = float(trade.get('current_stop', trade.get('initial_stop', 0)) or 0)
        current = float(st.session_state.get('_position_prices', {}).get(ticker) or entry)
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        if stop > 0 and current <= stop:
            must_act.append((95, f"🔴 Stop breached: {ticker}", ticker))
            action_queue.append({
                'priority': 95, 'category': 'risk', 'ticker': ticker, 'message': f"Stop breached: {ticker}",
                'action': 'risk_manage', 'current': current, 'stop': stop
            })
        elif pnl_pct <= -3:
            must_act.append((75, f"🟠 Drawdown >3%: {ticker} ({pnl_pct:+.1f}%)", ticker))
            action_queue.append({
                'priority': 75, 'category': 'risk', 'ticker': ticker, 'message': f"Drawdown >3%: {ticker} ({pnl_pct:+.1f}%)",
                'action': 'risk_manage', 'current': current, 'stop': stop
            })
    for row in snap.scan_summary:
        earn_days = int(row.get('earn_days', 999) or 999)
        if 0 <= earn_days <= 3:
            ticker = row.get('ticker', '')
            must_act.append((70 - earn_days, f"🗓️ Earnings soon ({earn_days}d): {ticker}", ticker))
            action_queue.append({'priority': 70 - earn_days, 'category': 'earnings', 'ticker': ticker, 'message': f"Earnings soon ({earn_days}d): {ticker}", 'action': 'open_trade'})
    if stale_count >= 3:
        action_queue.append({'priority': 98, 'category': 'system', 'ticker': '', 'message': "Critical stale data streams — refresh required", 'action': 'refresh'})
    must_act.sort(key=lambda x: x[0], reverse=True)
    action_queue = sorted(action_queue, key=lambda x: x['priority'], reverse=True)

    with q1:
        st.markdown("**Must Act Now**")
        if not must_act:
            st.caption("No urgent actions right now.")
        for i, (_, msg, ticker) in enumerate(must_act[:12]):
            if st.button(msg, key=f"exec_urgent_{i}_{ticker}", width="stretch"):
                if ticker:
                    _load_ticker_for_view(ticker)

    with q2:
        st.markdown("**Top Trade Candidates**")
        settings = _trade_quality_settings()
        tf_ranked_all = (st.session_state.get('trade_finder_results', {}) or {}).get('rows', []) or []
        tf_ranked = [r for r in tf_ranked_all if _trade_candidate_is_qualified(r, settings)]
        _find_new = st.session_state.get('find_new_trades_report', {}) or {}
        if _find_new.get('candidates'):
            st.caption(
                f"Source: Find New Trades ({int(_find_new.get('scan_universe', 0) or 0)} tickers, "
                f"{int(_find_new.get('watchlists_count', 0) or 0)} watchlists)"
            )
        if tf_ranked_all:
            st.caption(
                f"Source: Trade Finder ranked list (qualified {len(tf_ranked)}/{len(tf_ranked_all)} "
                f"| min R:R {settings['min_rr']:.1f} | earnings>{settings['earn_block_days']}d)"
            )
        if not gate.allow_new_trades:
            st.caption("New trade entries are blocked by current market gate.")
        elif not actionable and not tf_ranked:
            st.caption("No actionable entries in current scan.")
        if tf_ranked:
            for idx, tr in enumerate(tf_ranked[:12]):
                ticker = str(tr.get('ticker', '?') or '?')
                ai_rec = str(tr.get('ai_buy_recommendation', 'Watch Only') or 'Watch Only')
                score = float(tr.get('rank_score', 0) or 0)
                rr = float(tr.get('risk_reward', 0) or 0)
                if st.button(
                    f"{ticker} | {ai_rec} | R:R {rr:.2f}:1 | Score {score:.1f}",
                    key=f"exec_pick_tf_{idx}_{ticker}",
                    width="stretch",
                ):
                    st.session_state['trade_finder_selected_trade'] = {
                        'ticker': ticker,
                        'entry': float(tr.get('suggested_entry', tr.get('price', 0)) or tr.get('price', 0) or 0),
                        'stop': float(tr.get('suggested_stop_loss', 0) or 0),
                        'target': float(tr.get('suggested_target', 0) or 0),
                        'ai_buy_recommendation': ai_rec,
                        'risk_reward': rr,
                        'earn_days': int(tr.get('earn_days', 999) or 999),
                        'reason': str(tr.get('reason', '') or ''),
                        'ai_rationale': str(tr.get('ai_rationale', '') or ''),
                        'provider': str(tr.get('provider', 'system') or 'system'),
                        'generated_at_iso': str((st.session_state.get('trade_finder_results', {}) or {}).get('generated_at_iso', '')),
                        'trade_finder_run_id': str(tr.get('trade_finder_run_id', '') or (st.session_state.get('trade_finder_results', {}) or {}).get('run_id', '') or ''),
                    }
                    st.session_state['default_detail_tab'] = 4
                    st.session_state['_switch_to_scanner_tab'] = True
                    _load_ticker_for_view(ticker)
                    st.rerun()
                st.caption(f"Why: {str(tr.get('ai_rationale', '') or str(tr.get('scanner_summary', '') or ''))[:180]}")
        else:
            for cand in actionable[:12]:
                row = cand['row']
                card = cand.get('card', {}) or {}
                ticker = row.get('ticker', '?')
                rec = row.get('recommendation', '?')
                conv = row.get('conviction', 0)
                score = cand['score']
                why = card.get('explainability', '') or "; ".join(cand['reasons'][:3])
                readiness = card.get('execution_readiness', '')
                if st.button(f"{ticker} | {rec} | Conviction {conv} | Score {score:.1f}",
                             key=f"exec_pick_{ticker}", width="stretch"):
                    _load_ticker_for_view(ticker)
                st.caption(f"Why: {why}")
                if readiness:
                    st.caption(f"Decision: {readiness} | Regime fit: {card.get('regime_fit_score', 0)}")

    with q3:
        st.markdown("**Open Positions Needing Attention**")
        if not snap.open_trades:
            st.caption("No open positions.")
        else:
            pos_cache = st.session_state.get('_position_prices', {})
            for trade in snap.open_trades:
                ticker = trade.get('ticker', '').upper().strip()
                entry = float(trade.get('entry_price', 0) or 0)
                stop = float(trade.get('current_stop', trade.get('initial_stop', 0)) or 0)
                current = float(pos_cache.get(ticker) or fetch_current_price(ticker) or entry)
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                at_stop = stop > 0 and current <= stop
                if pnl_pct <= -2 or at_stop:
                    icon = "🔴" if at_stop else "🟠"
                    text = f"{icon} {ticker}  ${current:.2f} ({pnl_pct:+.1f}%)"
                    if st.button(text, key=f"exec_risk_{ticker}", width="stretch"):
                        _load_ticker_for_view(ticker)

    if '_queue_action_confirm' not in st.session_state:
        st.session_state['_queue_action_confirm'] = {}
    if '_queue_pending_actions' not in st.session_state:
        st.session_state['_queue_pending_actions'] = {}
    if '_queue_action_executed' not in st.session_state:
        st.session_state['_queue_action_executed'] = {}
    if '_queue_recent_actions' not in st.session_state:
        st.session_state['_queue_recent_actions'] = []
    if '_queue_last_result_toasts' not in st.session_state:
        st.session_state['_queue_last_result_toasts'] = []

    undo_window_s = 12
    confirm_ttl_s = 20

    now_ts = time.time()
    # Expire stale confirmation prompts so accidental stale confirms are not executed.
    for ckey, ts in list(st.session_state['_queue_action_confirm'].items()):
        if (now_ts - float(ts or 0)) > confirm_ttl_s:
            st.session_state['_queue_action_confirm'].pop(ckey, None)

    def _record_queue_recent(status: str, detail: str):
        recent = st.session_state.get('_queue_recent_actions', [])
        recent.insert(0, {
            'ts': datetime.now().isoformat(timespec='seconds'),
            'status': status,
            'detail': (detail or '')[:220],
        })
        st.session_state['_queue_recent_actions'] = recent[:30]

    def _open_trade_for_ticker(tkr: str) -> Optional[dict]:
        for tr in jm.open_trades:
            if str(tr.get('ticker', '')).upper().strip() == tkr and str(tr.get('status', '')).upper() == 'OPEN':
                return tr
        return None

    def _planned_trade_for_id(pid: str) -> Optional[dict]:
        for tr in jm.get_planned_trades():
            if str(tr.get('plan_id', '')).strip() == pid:
                return tr
        return None

    def _queue_log(op: str, status: str, details: str):
        msg = f"op={op} status={status} {details}".strip()
        _append_audit_event("QUEUE_ACTION", msg, source="exec_queue")
        _record_queue_recent(status, msg)

    def _confirm_remaining(confirm_key: str) -> int:
        ts = float(st.session_state.get('_queue_action_confirm', {}).get(confirm_key, 0) or 0)
        if ts <= 0:
            return 0
        left = max(0, int(confirm_ttl_s - (time.time() - ts)))
        if left <= 0:
            st.session_state['_queue_action_confirm'].pop(confirm_key, None)
            return 0
        return left

    def _queue_schedule(action_key: str, payload: Dict[str, Any]) -> str:
        pending = st.session_state.get('_queue_pending_actions', {})
        executed = st.session_state.get('_queue_action_executed', {})
        if action_key in executed:
            return "Action already executed"
        if action_key in pending:
            return f"Action already pending ({max(0, int(pending[action_key].get('execute_at', 0) - time.time()))}s)"
        now_ts = time.time()
        payload['created_at'] = now_ts
        payload['execute_at'] = now_ts + undo_window_s
        pending[action_key] = payload
        st.session_state['_queue_pending_actions'] = pending
        _queue_log(payload.get('kind', 'unknown'), "QUEUED", f"action_key={action_key}")
        return f"Queued. Undo available for {undo_window_s}s"

    def _process_pending_queue_actions() -> bool:
        pending = st.session_state.get('_queue_pending_actions', {})
        executed = st.session_state.get('_queue_action_executed', {})
        now_ts = time.time()
        did_any = False
        for action_key, payload in list(pending.items()):
            if now_ts < float(payload.get('execute_at', 0) or 0):
                continue

            kind = str(payload.get('kind', '')).strip()
            ticker = str(payload.get('ticker', '')).upper().strip()
            plan_id = str(payload.get('plan_id', '')).strip()
            status = "SKIPPED"
            detail = f"action_key={action_key} ticker={ticker} plan_id={plan_id}"

            if action_key in executed:
                detail += " reason=already_executed"
            elif kind == 'trigger_plan':
                p = _planned_trade_for_id(plan_id)
                p_status = str((p or {}).get('status', '')).upper()
                if p and p_status == 'PLANNED':
                    jm.update_planned_trade_status(plan_id, "TRIGGERED", notes="triggered from action queue")
                    status = "EXECUTED"
                else:
                    detail += f" reason=invalid_status({p_status or 'missing'})"
            elif kind == 'entered_plan':
                p = _planned_trade_for_id(plan_id)
                p_status = str((p or {}).get('status', '')).upper()
                if p and p_status == 'TRIGGERED':
                    jm.update_planned_trade_status(plan_id, "ENTERED", notes="entered from action queue")
                    status = "EXECUTED"
                else:
                    detail += f" reason=invalid_status({p_status or 'missing'})"
            elif kind == 'cancel_plan':
                p = _planned_trade_for_id(plan_id)
                p_status = str((p or {}).get('status', '')).upper()
                if p and p_status in {'PLANNED', 'TRIGGERED'}:
                    jm.update_planned_trade_status(plan_id, "CANCELLED", notes="cancelled from action queue")
                    status = "EXECUTED"
                else:
                    detail += f" reason=invalid_status({p_status or 'missing'})"
            elif kind == 'tighten_stop':
                tr = _open_trade_for_ticker(ticker)
                new_stop = float(payload.get('new_stop', 0) or 0)
                old_stop = float(payload.get('old_stop', 0) or 0)
                if tr and new_stop > float(tr.get('current_stop', tr.get('initial_stop', 0)) or 0):
                    jm.update_stop(ticker, new_stop)
                    status = "EXECUTED"
                    detail += f" old_stop={old_stop:.2f} new_stop={new_stop:.2f}"
                else:
                    detail += " reason=not_tightening_or_missing"
            elif kind == 'close_now':
                tr = _open_trade_for_ticker(ticker)
                close_px = float(payload.get('close_price', 0) or 0)
                if tr and close_px > 0:
                    jm.close_trade(ticker, close_px, exit_reason='manual', notes='Closed from action queue')
                    status = "EXECUTED"
                    detail += f" price={close_px:.2f}"
                else:
                    detail += " reason=no_open_position"
            else:
                detail += " reason=unknown_kind"

            executed[action_key] = now_ts
            pending.pop(action_key, None)
            _queue_log(kind or 'unknown', status, detail)
            st.session_state['_queue_last_result_toasts'].append({
                'status': status,
                'detail': f"{kind} {ticker}".strip(),
            })
            did_any = True

        st.session_state['_queue_pending_actions'] = pending
        st.session_state['_queue_action_executed'] = executed
        return did_any

    if _process_pending_queue_actions():
        st.rerun()

    with st.expander(f"⚡ Unified Action Queue ({len(action_queue)})", expanded=False):
        if not action_queue:
            st.caption("No queued actions.")
        toast_rows = st.session_state.get('_queue_last_result_toasts', [])
        for row in toast_rows[:5]:
            st.toast(f"{row.get('status', 'INFO')}: {row.get('detail', '')}")
        if toast_rows:
            st.session_state['_queue_last_result_toasts'] = []
        pending_actions = st.session_state.get('_queue_pending_actions', {})
        if pending_actions:
            st.markdown("**Pending Queue Actions (undo window)**")
            for pkey, payload in sorted(pending_actions.items(), key=lambda x: x[1].get('execute_at', 0)):
                secs_left = max(0, int(float(payload.get('execute_at', 0) or 0) - time.time()))
                p_kind = str(payload.get('kind', 'action'))
                p_ticker = str(payload.get('ticker', '')).upper().strip()
                pc1, pc2 = st.columns([4, 1])
                with pc1:
                    st.caption(f"⏳ {p_kind} {p_ticker} executing in {secs_left}s")
                with pc2:
                    if st.button("Undo", key=f"aq_undo_{pkey}", width="stretch"):
                        st.session_state['_queue_pending_actions'].pop(pkey, None)
                        _queue_log(p_kind, "UNDONE", f"action_key={pkey} ticker={p_ticker}")
                        st.rerun()
            st.divider()
        for i, item in enumerate(action_queue[:30]):
            pri = int(item.get('priority', 0))
            msg = str(item.get('message', ''))
            ticker = str(item.get('ticker', '') or '').upper().strip()
            plan_id = str(item.get('plan_id', '') or '').strip()
            action = str(item.get('action', 'open_trade'))
            current = float(item.get('current', 0) or 0)
            stop = float(item.get('stop', 0) or 0)
            cmsg, c1, c2, c3 = st.columns([4, 1, 1, 1])
            row_badges = []
            if action == 'planned_queued' and plan_id:
                _k = f"trigger_plan:{plan_id}"
                if _k in pending_actions:
                    _left = max(0, int(float(pending_actions[_k].get('execute_at', 0) or 0) - time.time()))
                    row_badges.append(f"⏳ pending {_left}s")
                if _k in st.session_state.get('_queue_action_executed', {}):
                    row_badges.append("✅ executed")
            if action == 'planned_triggered' and plan_id:
                _ek = f"entered_plan:{plan_id}"
                _ck = f"cancel_plan:{plan_id}"
                if _ek in pending_actions or _ck in pending_actions:
                    _lefts = []
                    if _ek in pending_actions:
                        _lefts.append(int(float(pending_actions[_ek].get('execute_at', 0) or 0) - time.time()))
                    if _ck in pending_actions:
                        _lefts.append(int(float(pending_actions[_ck].get('execute_at', 0) or 0) - time.time()))
                    row_badges.append(f"⏳ pending {max(0, min(_lefts))}s")
                if _ek in st.session_state.get('_queue_action_executed', {}) or _ck in st.session_state.get('_queue_action_executed', {}):
                    row_badges.append("✅ executed")
            if action == 'risk_manage' and ticker:
                _ck = f"close_now:{ticker}"
                if _ck in pending_actions:
                    _left = max(0, int(float(pending_actions[_ck].get('execute_at', 0) or 0) - time.time()))
                    row_badges.append(f"⏳ pending {_left}s")
                if _ck in st.session_state.get('_queue_action_executed', {}):
                    row_badges.append("✅ executed")
            with cmsg:
                badge_txt = (" | " + " | ".join(row_badges)) if row_badges else ""
                st.caption(f"P{pri} | {msg}{badge_txt}")
            with c1:
                if action == 'refresh':
                    if st.button("Refresh", key=f"aq_refresh_{i}", width="stretch"):
                        _fast_refresh_dashboard()
                        st.rerun()
                else:
                    if st.button("Open", key=f"aq_open_{i}_{ticker}", width="stretch"):
                        if ticker:
                            _load_ticker_for_view(ticker)
                            st.session_state['default_detail_tab'] = 4
                            st.session_state['_switch_to_scanner_tab'] = True
                            st.rerun()
            with c2:
                if action == 'planned_queued' and plan_id:
                    if st.button("Trigger", key=f"aq_trigger_{i}_{plan_id}", width="stretch"):
                        action_key = f"trigger_plan:{plan_id}"
                        status_msg = _queue_schedule(action_key, {
                            'kind': 'trigger_plan',
                            'plan_id': plan_id,
                            'ticker': ticker,
                        })
                        st.info(status_msg)
                        st.rerun()
                elif action == 'planned_triggered' and plan_id:
                    confirm_key = f"entered_{plan_id}"
                    if st.button("Entered", key=f"aq_entered_{i}_{plan_id}", width="stretch"):
                        st.session_state['_queue_action_confirm'][confirm_key] = time.time()
                    left_s = _confirm_remaining(confirm_key)
                    if left_s > 0:
                        st.caption(f"confirm {left_s}s")
                        if st.button("Confirm Entered", key=f"aq_entered_confirm_{i}_{plan_id}", width="stretch"):
                            action_key = f"entered_plan:{plan_id}"
                            status_msg = _queue_schedule(action_key, {
                                'kind': 'entered_plan',
                                'plan_id': plan_id,
                                'ticker': ticker,
                            })
                            st.info(status_msg)
                            st.session_state['_queue_action_confirm'].pop(confirm_key, None)
                            st.rerun()
                elif action == 'risk_manage' and ticker and current > 0:
                    if st.button("Tighten +1%", key=f"aq_tighten_{i}_{ticker}", width="stretch"):
                        # quick risk action: move stop to 1% below current if this tightens risk
                        new_stop = round(max(stop, current * 0.99), 2) if stop > 0 else round(current * 0.99, 2)
                        action_key = f"tighten_stop:{ticker}:{new_stop:.2f}"
                        status_msg = _queue_schedule(action_key, {
                            'kind': 'tighten_stop',
                            'ticker': ticker,
                            'old_stop': stop,
                            'new_stop': new_stop,
                        })
                        st.info(status_msg)
                        st.rerun()
            with c3:
                if action in {'planned_queued', 'planned_triggered'} and plan_id:
                    confirm_key = f"cancel_{plan_id}"
                    if st.button("Cancel", key=f"aq_cancel_{i}_{plan_id}", width="stretch"):
                        st.session_state['_queue_action_confirm'][confirm_key] = time.time()
                    left_s = _confirm_remaining(confirm_key)
                    if left_s > 0:
                        st.caption(f"confirm {left_s}s")
                        if st.button("Confirm Cancel", key=f"aq_cancel_confirm_{i}_{plan_id}", width="stretch"):
                            action_key = f"cancel_plan:{plan_id}"
                            status_msg = _queue_schedule(action_key, {
                                'kind': 'cancel_plan',
                                'plan_id': plan_id,
                                'ticker': ticker,
                            })
                            st.info(status_msg)
                            st.session_state['_queue_action_confirm'].pop(confirm_key, None)
                            st.rerun()
                elif action == 'risk_manage' and ticker and current > 0:
                    confirm_key = f"close_{ticker}_{i}"
                    if st.button("Close Now", key=f"aq_close_{i}_{ticker}", width="stretch"):
                        st.session_state['_queue_action_confirm'][confirm_key] = time.time()
                    left_s = _confirm_remaining(confirm_key)
                    if left_s > 0:
                        st.caption(f"confirm {left_s}s")
                        if st.button("Confirm Close", key=f"aq_close_confirm_{i}_{ticker}", width="stretch"):
                            action_key = f"close_now:{ticker}"
                            status_msg = _queue_schedule(action_key, {
                                'kind': 'close_now',
                                'ticker': ticker,
                                'close_price': current,
                            })
                            st.info(status_msg)
                            st.session_state['_queue_action_confirm'].pop(confirm_key, None)
                            st.rerun()

        with st.expander("🧾 Recent Queue Actions", expanded=False):
            recent_queue_audit = []
            for evt in _get_audit_events():
                if evt.get('action') == 'QUEUE_ACTION' and evt.get('source') == 'exec_queue':
                    recent_queue_audit.append(evt)
                if len(recent_queue_audit) >= 12:
                    break
            if not recent_queue_audit:
                st.caption("No queue actions yet.")
            else:
                for evt in recent_queue_audit:
                    ts = str(evt.get('ts', ''))
                    det = str(evt.get('details', ''))
                    st.caption(f"{ts} | {det}")

    st.divider()
    with st.expander("🗂️ Planned Trades Board", expanded=False):
        all_plans = jm.get_planned_trades()
        status_filter = st.selectbox(
            "Status Filter",
            ["ALL", "PLANNED", "TRIGGERED", "CANCELLED", "ENTERED"],
            key="exec_plan_status_filter",
        )
        only_today = st.checkbox("Today only", value=False, key="exec_plan_today_only")
        stale_days = st.number_input(
            "Stale threshold (days)",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key="exec_plan_stale_days",
        )

        def _parse_dt(s: str):
            if not s:
                return None
            try:
                return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            except Exception:
                return None

        filtered = []
        now_dt = datetime.now()
        for p in all_plans:
            p_status = str(p.get('status', '')).upper()
            if status_filter != "ALL" and p_status != status_filter:
                continue
            created = _parse_dt(str(p.get('created_at', '') or ''))
            if only_today and created is not None and created.date() != now_dt.date():
                continue
            filtered.append(p)

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("✖ Cancel All Planned", key="exec_cancel_all_planned", width="stretch"):
                cnt = 0
                for p in jm.get_planned_trades(status="PLANNED"):
                    pid = str(p.get('plan_id', ''))
                    if pid:
                        jm.update_planned_trade_status(pid, "CANCELLED", notes="bulk-cancelled from executive board")
                        cnt += 1
                st.success(f"Cancelled {cnt} planned trade(s).")
                st.rerun()
        with bc2:
            if st.button("🧹 Cancel Stale Planned", key="exec_cancel_stale_planned", width="stretch"):
                cnt = 0
                for p in jm.get_planned_trades(status="PLANNED"):
                    pid = str(p.get('plan_id', ''))
                    created = _parse_dt(str(p.get('created_at', '') or ''))
                    if pid and created is not None and (now_dt - created.replace(tzinfo=None) if created.tzinfo else now_dt - created).days >= int(stale_days):
                        jm.update_planned_trade_status(pid, "CANCELLED", notes=f"stale >={int(stale_days)}d")
                        cnt += 1
                st.success(f"Cancelled {cnt} stale planned trade(s).")
                st.rerun()
        with bc3:
            if st.button("✅ Mark Triggered Entered", key="exec_mark_triggered_entered", width="stretch"):
                cnt = 0
                for p in jm.get_planned_trades(status="TRIGGERED"):
                    pid = str(p.get('plan_id', ''))
                    if pid:
                        jm.update_planned_trade_status(pid, "ENTERED", notes="manual-entered from executive board")
                        cnt += 1
                st.success(f"Marked {cnt} triggered trade(s) as entered.")
                st.rerun()

        st.caption(f"Showing {len(filtered)} / {len(all_plans)} planned trades")
        if not filtered:
            st.caption("No planned trades match filters.")
        for p in filtered[:50]:
            pid = str(p.get('plan_id', ''))
            pticker = str(p.get('ticker', '')).upper().strip()
            pstatus = str(p.get('status', 'PLANNED')).upper()
            pentry = float(p.get('entry', 0) or 0)
            pstop = float(p.get('stop', 0) or 0)
            ptarget = float(p.get('target', 0) or 0)
            prr = float(p.get('risk_reward', 0) or 0)
            pscore = float(p.get('rank_score', 0) or 0)
            st.caption(
                f"{pticker} | {pstatus} | Entry ${pentry:.2f} Stop ${pstop:.2f} Target ${ptarget:.2f} "
                f"| R:R {prr:.2f}:1 | Score {pscore:.2f} | Run {str(p.get('trade_finder_run_id', '') or 'n/a')}"
            )
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                if st.button("▶ Open", key=f"exec_plan_open_{pid}", width="stretch"):
                    st.session_state['trade_finder_selected_trade'] = {
                        'plan_id': pid,
                        'ticker': pticker,
                        'entry': pentry,
                        'stop': pstop,
                        'target': ptarget,
                        'risk_reward': prr,
                        'trade_finder_run_id': str(p.get('trade_finder_run_id', '') or ''),
                        'reason': str(p.get('reason', '') or ''),
                        'ai_rationale': str(p.get('notes', '') or ''),
                        'provider': str(p.get('source', 'planned') or 'planned'),
                    }
                    st.session_state['default_detail_tab'] = 4
                    st.session_state['_switch_to_scanner_tab'] = True
                    _load_ticker_for_view(pticker)
                    st.rerun()
            with r2:
                if st.button("⚡ Trigger", key=f"exec_plan_trigger_{pid}", width="stretch"):
                    st.info(jm.update_planned_trade_status(pid, "TRIGGERED"))
                    st.rerun()
            with r3:
                if st.button("✖ Cancel", key=f"exec_plan_cancel_{pid}", width="stretch"):
                    st.info(jm.update_planned_trade_status(pid, "CANCELLED"))
                    st.rerun()
            with r4:
                if st.button("✅ Entered", key=f"exec_plan_entered_{pid}", width="stretch"):
                    st.info(jm.update_planned_trade_status(pid, "ENTERED", notes="manual-entered from executive row"))
                    st.rerun()

    # Render telemetry (throttled)
    now_ts = time.time()
    if (now_ts - float(st.session_state.get('_last_exec_render_log_ts', 0.0))) > 60:
        st.session_state['_last_exec_render_log_ts'] = now_ts
        _append_perf_metric({
            "kind": "dashboard_render",
            "sec": round(now_ts - _render_started, 3),
            "stale_streams": stale_count,
            "regime": snap.regime,
            "actionable": len(actionable),
            "open_positions": len(snap.open_trades),
            "alerts": len(snap.pending_alerts),
        })


# =============================================================================
# APP MAIN
# =============================================================================

def main():
    render_sidebar()

    jm = get_journal()

    # Main content area
    conditionals = jm.get_pending_conditionals()
    alerts_label = f"🎯 Alerts ({len(conditionals)})" if conditionals else "🎯 Alerts"
    if _is_exec_dashboard_enabled():
        tab_exec, tab_finder, tab_scanner, tab_alerts, tab_positions, tab_perf = st.tabs([
            "📌 Executive Dashboard", "🧭 Trade Finder", "🔍 Scanner", alerts_label, "🏦 Position Manager", "📊 Performance"
        ])

        with tab_exec:
            render_executive_dashboard()

        with tab_finder:
            render_trade_finder_tab()

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
    else:
        tab_finder, tab_scanner, tab_alerts, tab_positions, tab_perf = st.tabs([
            "🧭 Trade Finder", "🔍 Scanner", alerts_label, "🏦 Position Manager", "📊 Performance"
        ])
        with tab_finder:
            render_trade_finder_tab()
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

    if st.session_state.pop('_switch_to_scanner_tab', False):
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            setTimeout(function() {
              const doc = window.parent.document;
              const buttons = Array.from(doc.querySelectorAll('button[role="tab"], [data-baseweb="tab"] button'));
              const scanner = buttons.find(b => (b.innerText || '').includes('Scanner'));
              if (scanner) scanner.click();
            }, 80);
            </script>
            """,
            height=0,
        )


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
            if st.button(f"📊 {ticker}", key=f"alert_view_{ticker}", width="stretch"):
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
    # Flush any queued GitHub backups (debounced — won't push on every rerun)
    try:
        import github_backup
        github_backup.flush()
    except Exception:
        pass
