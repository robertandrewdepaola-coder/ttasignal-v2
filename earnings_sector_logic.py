"""
TTA v2 — Earnings & Sector Logic
==================================

Pure business logic for earnings date handling, sector name canonicalization,
sector phase inference, and recommendation adjustments.

Extracted from app.py to reduce main module size and improve rerun performance.

Version: 2.0.0 (2026-02-28)
"""

import json as _json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# JSON / TEXT EXTRACTION HELPERS
# =============================================================================

def extract_first_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extractor for AI responses."""
    if not raw:
        return {}
    txt = str(raw).strip()
    try:
        return _json.loads(txt)
    except Exception:
        pass
    decoder = _json.JSONDecoder()
    for i, ch in enumerate(txt):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(txt[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def extract_earn_days_hint(text: str) -> Optional[int]:
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


# =============================================================================
# EARNINGS DATE PARSING & VALIDATION
# =============================================================================

def parse_earnings_date(value: Any) -> Optional[datetime]:
    """Parse YYYY-MM-DD earnings date safely. Returns date object or None."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except Exception:
        return None


def normalize_earnings_days(next_earnings: str, raw_days: Any) -> Optional[int]:
    """Prefer a fresh date-derived days-until value over stale cached counters."""
    d = parse_earnings_date(next_earnings)
    if d is not None:
        days = (d - datetime.now().date()).days
        if days < 0:
            return None
        return int(days)
    try:
        return max(0, int(raw_days))
    except Exception:
        return None


def earnings_confidence_rank(confidence: str) -> int:
    c = str(confidence or '').upper().strip()
    if c in {"HIGH", "MEDIUM-HIGH", "MEDIUM_HIGH"}:
        return 3
    if c == "MEDIUM":
        return 2
    if c == "LOW":
        return 1
    return 0


def is_estimated_earnings_source(source: str) -> bool:
    s = str(source or '').lower()
    return "estimated" in s or "historical + 91d" in s or "last + 91d" in s


def is_earnings_data_trusted(
    earn_days: Any,
    source: str = "",
    confidence: str = "",
    next_earnings: str = "",
) -> bool:
    try:
        d = int(earn_days)
    except Exception:
        return False
    if d < 0 or d >= 999 or d > 400:
        return False
    if next_earnings:
        dt = parse_earnings_date(next_earnings)
        if dt is None:
            return False
        if (dt - datetime.now().date()).days < 0:
            return False
    if is_estimated_earnings_source(source):
        return False
    conf_rank = earnings_confidence_rank(confidence)
    if conf_rank >= 2:
        return True
    if conf_rank == 1:
        return False
    src = str(source or '').strip().lower()
    if not src:
        return False
    weak_src_tokens = ("fundamental profile", "earnings history", "unknown", "n/a")
    if any(tok in src for tok in weak_src_tokens):
        return False
    return True


def earnings_badge(
    earn_days: int,
    source: str = "",
    confidence: str = "",
    earn_date: str = "",
) -> Dict[str, str]:
    """Color-coded earnings horizon badge for Trade Finder rows."""
    trusted = is_earnings_data_trusted(
        earn_days,
        source=source,
        confidence=confidence,
        next_earnings=earn_date,
    )
    d = int(earn_days or 999)
    if not trusted:
        if 0 <= d < 999:
            return {"text": f"{d}d?", "color": "#ef4444", "icon": "🔴"}
        return {"text": "unverified", "color": "#ef4444", "icon": "🔴"}
    if d <= 7:
        return {"text": f"{d}d", "color": "#ef4444", "icon": "🔴"}
    if d <= 14:
        return {"text": f"{d}d", "color": "#f59e0b", "icon": "🟡"}
    return {"text": f"{d}d", "color": "#22c55e", "icon": "🟢"}


# =============================================================================
# SECTOR NAME & PHASE LOGIC
# =============================================================================

def canonicalize_sector_name(sector_name: str, rotation_ctx: Optional[Dict[str, Any]] = None) -> str:
    """Normalize sector labels to canonical names used by rotation context."""
    s = str(sector_name or "").strip()
    if not s:
        return ""
    aliases = {
        "information technology": "Technology",
        "info tech": "Technology",
        "health care": "Healthcare",
        "healthcare": "Healthcare",
        "financials": "Financial Services",
        "financial": "Financial Services",
        "communications": "Communication Services",
        "communication": "Communication Services",
        "consumer staples": "Consumer Defensive",
        "consumer discretionary": "Consumer Cyclical",
        "basic materials": "Materials",
    }
    s_alias = aliases.get(s.lower(), s)

    if not isinstance(rotation_ctx, dict) or not rotation_ctx:
        return s_alias
    if s_alias in rotation_ctx:
        return s_alias

    lower_map = {str(k).lower(): str(k) for k in rotation_ctx.keys()}
    if s_alias.lower() in lower_map:
        return lower_map[s_alias.lower()]
    if s.lower() in lower_map:
        return lower_map[s.lower()]
    return s_alias


def infer_sector_from_text(text: str) -> str:
    """Infer canonical sector label from model/scanner text when direct sector lookup is unavailable."""
    txt = str(text or "").lower()
    if not txt:
        return ""
    sector_aliases = [
        ("Industrials", ["industrials", "industrial"]),
        ("Technology", ["technology", "tech"]),
        ("Financial Services", ["financial services", "financials", "finance"]),
        ("Energy", ["energy"]),
        ("Utilities", ["utilities", "utility"]),
        ("Healthcare", ["health care", "healthcare", "health"]),
        ("Materials", ["basic materials", "materials"]),
        ("Real Estate", ["real estate"]),
        ("Communication Services", ["communication services", "communications"]),
        ("Consumer Defensive", ["consumer defensive", "consumer staples", "constap"]),
        ("Consumer Cyclical", ["consumer cyclical", "consumer discretionary", "condisc"]),
    ]
    for canonical, aliases in sector_aliases:
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", txt):
                return canonical
    return ""


def infer_sector_phase_from_text(text: str) -> str:
    """Infer rotation phase from model/scanner text when explicit phase is missing."""
    txt = str(text or "").lower()
    if not txt:
        return ""
    if "out of rotation" in txt or re.search(r"\blagging\b", txt):
        return "LAGGING"
    if re.search(r"\bfading\b", txt) or "transition" in txt:
        return "FADING"
    if "in rotation" in txt and re.search(r"\bleading\b", txt):
        return "LEADING"
    if "in rotation" in txt and re.search(r"\bemerging\b", txt):
        return "EMERGING"
    if re.search(r"\bleading\b", txt):
        return "LEADING"
    if re.search(r"\bemerging\b", txt):
        return "EMERGING"
    return ""


def sector_phase_display(sector_phase: str) -> Dict[str, str]:
    """Map sector phase to trader-facing rotation badge."""
    phase_u = str(sector_phase or "").upper().strip()
    if phase_u in {"LEADING", "EMERGING"}:
        return {"label": "IN ROTATION", "icon": "🟢", "color": "#22c55e"}
    if phase_u == "FADING":
        return {"label": "TRANSITION", "icon": "🟡", "color": "#f59e0b"}
    if phase_u == "LAGGING":
        return {"label": "OUT OF ROTATION", "icon": "🔴", "color": "#ef4444"}
    return {"label": "UNCLASSIFIED", "icon": "⚪", "color": "#94a3b8"}


def adjust_recommendation_for_sector(
    recommendation: str,
    conviction: int,
    sector_phase: str,
) -> Dict[str, Any]:
    """
    Reconcile raw signal recommendation with current sector rotation regime.
    Keeps signal engine output intact but applies UI-level execution context.
    """
    rec_raw = str(recommendation or "").strip()
    rec_u = rec_raw.upper()
    phase_u = str(sector_phase or "").upper()
    conv = int(conviction or 0)
    adjusted = rec_raw
    reason = ""

    if phase_u == "LAGGING":
        if rec_u == "STRONG BUY":
            adjusted = "BUY (SECTOR HEADWIND)"
            conv = max(1, conv - 2)
            reason = "Sector lagging"
        elif ("BUY" in rec_u or "ENTRY" in rec_u) and "SKIP" not in rec_u:
            adjusted = "WATCH (SECTOR)"
            conv = max(1, conv - 2)
            reason = "Sector lagging"
    elif phase_u == "FADING":
        if rec_u == "STRONG BUY":
            adjusted = "BUY (CAUTION)"
            conv = max(1, conv - 1)
            reason = "Sector fading"
        elif rec_u == "BUY":
            adjusted = "BUY (CAUTION)"
            conv = max(1, conv - 1)
            reason = "Sector fading"

    return {
        "recommendation": adjusted,
        "conviction": conv,
        "adjusted": adjusted != rec_raw,
        "reason": reason,
    }


# =============================================================================
# TRADE MATH HELPERS
# =============================================================================

def calc_rr(entry: float, stop: float, target: float) -> float:
    """Compute reward/risk ratio with guardrails."""
    if entry <= 0 or stop <= 0 or target <= 0:
        return 0.0
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return 0.0
    return round(reward / risk, 2)


def extract_rr_from_text(text: str) -> Optional[float]:
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


def format_eta(seconds: float) -> str:
    """Format remaining seconds into human-friendly ETA string."""
    if seconds is None or seconds <= 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m = s // 60
    rem = s % 60
    if m < 60:
        return f"{m}m{rem:02d}s" if rem else f"{m}m"
    h = m // 60
    rm = m % 60
    return f"{h}h{rm:02d}m"
