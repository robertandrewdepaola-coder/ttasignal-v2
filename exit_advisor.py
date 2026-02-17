"""
TTA v2 Exit Advisor — Position Management & Exit Analysis Engine
==================================================================

Analyzes open positions and provides AI-driven exit recommendations:
HOLD, TAKE PARTIAL, CLOSE, or TIGHTEN STOP.

Follows Separation of Concerns:
- Reads positions from JournalManager
- Fetches data from data_fetcher
- Uses signal_engine for technical state
- Calls AI via call_ai patterns (Groq/Gemini)
- Logs results to SQLite audit trail
- Sends email notifications via Gmail SMTP

NO UI code. Called by app.py.

Version: 1.0.0 (2026-02-12)
"""

import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExitAdvice:
    """AI-generated exit recommendation for one position."""
    ticker: str
    action: str = ''            # HOLD, TAKE_PARTIAL, CLOSE, TIGHTEN_STOP
    confidence: int = 0         # 1-10
    reasoning: str = ''
    suggested_stop: float = 0   # New stop if TIGHTEN_STOP
    partial_pct: int = 0        # % to sell if TAKE_PARTIAL (e.g. 50)
    risk_note: str = ''
    provider: str = ''          # groq, gemini, system
    analyzed_at: str = ''

    # Position context at time of analysis
    entry_price: float = 0
    current_price: float = 0
    current_stop: float = 0
    unrealized_pnl_pct: float = 0
    days_held: int = 0

    # Technical context
    macd_bullish: bool = False
    weekly_bullish: bool = False
    ao_positive: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# EXIT ANALYSIS — builds prompt, calls AI, parses result
# =============================================================================

def build_exit_prompt(ticker: str, trade: Dict, current_price: float,
                      signal_data: Dict = None) -> str:
    """
    Build AI prompt for exit analysis of one position.

    trade: dict from JournalManager.get_open_trades()
    signal_data: dict with macd, ao, weekly, monthly technical state
    """
    entry = float(trade.get('entry_price', 0))
    shares = float(trade.get('shares', 0))
    stop = float(trade.get('current_stop', trade.get('initial_stop', 0)))
    target = float(trade.get('target', 0))
    entry_date = trade.get('entry_date', '')
    signal_type = trade.get('signal_type', '')

    # P&L
    pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
    pnl_dollars = (current_price - entry) * shares

    # Days held
    days_held = 0
    if entry_date:
        try:
            days_held = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
        except Exception:
            pass

    # Risk from current price to stop
    risk_to_stop_pct = ((current_price - stop) / current_price * 100) if stop > 0 and current_price > 0 else 0

    # Distance to target
    target_distance_pct = ((target - current_price) / current_price * 100) if target > 0 and current_price > 0 else 0

    prompt = f"""POSITION EXIT ANALYSIS for {ticker}

POSITION:
- Entry: ${entry:.2f} on {entry_date} ({days_held} days ago)
- Shares: {shares:.0f} (${shares * entry:,.0f} position)
- Current price: ${current_price:.2f}
- P&L: {pnl_pct:+.1f}% (${pnl_dollars:+,.0f})
- Current stop: ${stop:.2f} (risk: {risk_to_stop_pct:.1f}% from current price)
- Target: ${target:.2f} ({target_distance_pct:+.1f}% away)
- Signal type: {signal_type}
"""

    if signal_data:
        macd = signal_data.get('macd', {})
        ao = signal_data.get('ao', {})
        weekly = signal_data.get('weekly', {})
        monthly = signal_data.get('monthly', {})

        prompt += f"""
CURRENT TECHNICAL STATE:
- Daily MACD: {'Bullish' if macd.get('bullish') else 'Bearish'} | Histogram: {macd.get('histogram', 0):+.4f}{'  ⚠️ WEAKENING' if macd.get('weakening') else ''}
- Daily AO: {'Positive' if ao.get('positive') else 'Negative'} | Value: {ao.get('value', 0):+.4f}
- Weekly MACD: {'Bullish' if weekly.get('bullish') else 'Bearish'}
- Monthly MACD: {'Bullish' if monthly.get('bullish') else 'Bearish'}
"""

    prompt += """
YOUR TASK: As a senior portfolio manager, analyze this position and provide a SPECIFIC recommendation.

Respond in EXACTLY this format:
ACTION: [HOLD / TAKE_PARTIAL / CLOSE / TIGHTEN_STOP]
CONFIDENCE: [1-10]
REASONING: [2-3 sentences explaining why]
SUGGESTED_STOP: [new stop price if TIGHTEN_STOP, otherwise current stop]
PARTIAL_PCT: [% to sell if TAKE_PARTIAL, e.g. 50, otherwise 0]
RISK_NOTE: [one sentence on key risk to watch]

Rules:
- HOLD: technicals still aligned, let it run
- TAKE_PARTIAL: large unrealized gain (>15%) or approaching resistance, lock in some profit
- CLOSE: technicals deteriorating (weekly MACD bearish + daily MACD bearish), or stop about to be hit
- TIGHTEN_STOP: profitable trade, trail stop up to protect gains
- Always prioritize capital preservation
- If P&L is negative and technicals are weakening, recommend CLOSE
- If profitable >10% with weekly MACD turning, recommend TIGHTEN_STOP or TAKE_PARTIAL
"""

    return prompt


def parse_exit_response(text: str, ticker: str) -> ExitAdvice:
    """Parse AI response into ExitAdvice dataclass."""
    advice = ExitAdvice(
        ticker=ticker,
        analyzed_at=datetime.now().isoformat(),
    )

    if not text:
        advice.action = 'HOLD'
        advice.reasoning = 'AI analysis unavailable — defaulting to hold.'
        advice.provider = 'system'
        return advice

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue

        key, _, value = line.partition(':')
        key = key.strip().upper().replace(' ', '_')
        value = value.strip()

        if key == 'ACTION':
            clean = value.upper().strip()
            if clean in ('HOLD', 'TAKE_PARTIAL', 'CLOSE', 'TIGHTEN_STOP'):
                advice.action = clean
        elif key == 'CONFIDENCE':
            try:
                advice.confidence = min(10, max(1, int(value.split('/')[0].strip())))
            except Exception:
                pass
        elif key == 'REASONING':
            advice.reasoning = value
        elif key == 'SUGGESTED_STOP':
            try:
                advice.suggested_stop = float(value.replace('$', '').replace(',', '').strip())
            except Exception:
                pass
        elif key == 'PARTIAL_PCT':
            try:
                advice.partial_pct = int(value.replace('%', '').strip())
            except Exception:
                pass
        elif key == 'RISK_NOTE':
            advice.risk_note = value

    # Default action if parsing failed
    if not advice.action:
        advice.action = 'HOLD'
        advice.reasoning = advice.reasoning or 'Unable to parse AI recommendation — defaulting to hold.'

    return advice


def analyze_position(ticker: str, trade: Dict, current_price: float,
                     signal_data: Dict = None,
                     gemini_model=None, openai_client=None,
                     ai_model: str = "llama-3.3-70b-versatile",
                     fallback_model: str = "") -> ExitAdvice:
    """
    Full analysis pipeline for one position:
    1. Build prompt
    2. Call AI
    3. Parse response
    4. Enrich with position context
    """
    prompt = build_exit_prompt(ticker, trade, current_price, signal_data)

    # Call AI (reuse pattern from ai_analysis.py)
    raw_text = None
    provider = 'system'

    # Try OpenAI-compatible provider first (Groq or xAI)
    if openai_client is not None:
        models = [m for m in [ai_model, fallback_model] if m]
        if not models:
            models = ["llama-3.3-70b-versatile"]
        for model_name in models:
            try:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a senior portfolio manager focused on risk management and exit timing."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.4
                )
                raw_text = response.choices[0].message.content
                provider = 'openai_compat'
                break
            except Exception:
                continue

    # Gemini fallback
    if raw_text is None and gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            raw_text = response.text
            provider = 'gemini'
        except Exception:
            pass

    # Parse
    advice = parse_exit_response(raw_text or '', ticker)
    advice.provider = provider

    # Enrich with position context
    entry = float(trade.get('entry_price', 0))
    advice.entry_price = entry
    advice.current_price = current_price
    advice.current_stop = float(trade.get('current_stop', trade.get('initial_stop', 0)))
    advice.unrealized_pnl_pct = round(((current_price - entry) / entry * 100), 1) if entry > 0 else 0

    entry_date = trade.get('entry_date', '')
    if entry_date:
        try:
            advice.days_held = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
        except Exception:
            pass

    if signal_data:
        advice.macd_bullish = signal_data.get('macd', {}).get('bullish', False)
        advice.weekly_bullish = signal_data.get('weekly', {}).get('bullish', False)
        advice.ao_positive = signal_data.get('ao', {}).get('positive', False)

    return advice


def analyze_all_positions(open_trades: List[Dict],
                          fetch_price_fn=None,
                          fetch_signal_fn=None,
                          gemini_model=None,
                          openai_client=None,
                          ai_model: str = "llama-3.3-70b-versatile",
                          fallback_model: str = "") -> List[ExitAdvice]:
    """
    Analyze all open positions and return list of ExitAdvice.

    fetch_price_fn: callable(ticker) -> float
    fetch_signal_fn: callable(ticker) -> dict with macd/ao/weekly/monthly
    """
    results = []

    for trade in open_trades:
        ticker = trade.get('ticker', '')
        if not ticker:
            continue

        # Fetch current price
        current_price = 0
        if fetch_price_fn:
            current_price = fetch_price_fn(ticker) or 0

        if current_price <= 0:
            continue

        # Fetch signal data
        signal_data = None
        if fetch_signal_fn:
            try:
                signal_data = fetch_signal_fn(ticker)
            except Exception:
                pass

        # Analyze
        advice = analyze_position(
            ticker, trade, current_price, signal_data,
            gemini_model=gemini_model,
            openai_client=openai_client,
            ai_model=ai_model,
            fallback_model=fallback_model,
        )
        results.append(advice)

    return results


# =============================================================================
# AUDIT LOG — SQLite database for tracking AI performance
# =============================================================================

def _get_db(data_dir: str = '.') -> sqlite3.Connection:
    """Get or create SQLite database for exit audit log."""
    db_path = Path(data_dir) / "v2_exit_audit.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exit_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyzed_at TEXT,
            ticker TEXT,
            action TEXT,
            confidence INTEGER,
            reasoning TEXT,
            suggested_stop REAL,
            partial_pct INTEGER,
            risk_note TEXT,
            provider TEXT,
            entry_price REAL,
            current_price REAL,
            current_stop REAL,
            unrealized_pnl_pct REAL,
            days_held INTEGER,
            macd_bullish INTEGER,
            weekly_bullish INTEGER,
            ao_positive INTEGER
        )
    """)
    conn.commit()
    return conn


def save_audit(advice: ExitAdvice, data_dir: str = '.'):
    """Save one ExitAdvice to the audit log."""
    try:
        conn = _get_db(data_dir)
        conn.execute("""
            INSERT INTO exit_audit
            (analyzed_at, ticker, action, confidence, reasoning,
             suggested_stop, partial_pct, risk_note, provider,
             entry_price, current_price, current_stop, unrealized_pnl_pct,
             days_held, macd_bullish, weekly_bullish, ao_positive)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            advice.analyzed_at, advice.ticker, advice.action, advice.confidence,
            advice.reasoning, advice.suggested_stop, advice.partial_pct,
            advice.risk_note, advice.provider,
            advice.entry_price, advice.current_price, advice.current_stop,
            advice.unrealized_pnl_pct, advice.days_held,
            int(advice.macd_bullish), int(advice.weekly_bullish), int(advice.ao_positive),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[exit_advisor] Audit log error: {e}")


def save_audit_batch(advices: List[ExitAdvice], data_dir: str = '.'):
    """Save a batch of exit analyses to audit log."""
    for a in advices:
        save_audit(a, data_dir)


def get_audit_history(ticker: str = None, last_n: int = 50,
                      data_dir: str = '.') -> List[Dict]:
    """Retrieve audit log entries."""
    try:
        conn = _get_db(data_dir)
        if ticker:
            cursor = conn.execute(
                "SELECT * FROM exit_audit WHERE ticker=? ORDER BY analyzed_at DESC LIMIT ?",
                (ticker, last_n)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM exit_audit ORDER BY analyzed_at DESC LIMIT ?",
                (last_n,)
            )
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[exit_advisor] Audit read error: {e}")
        return []


# =============================================================================
# EMAIL NOTIFICATIONS — Gmail SMTP
# =============================================================================

def send_email_report(advices: List[ExitAdvice],
                      smtp_email: str = '',
                      smtp_password: str = '',
                      recipient: str = '') -> bool:
    """
    Send exit analysis summary via Gmail SMTP.

    Requires:
    - smtp_email: Gmail address
    - smtp_password: App-specific password (not regular password)
    - recipient: email to send to (can be same as smtp_email)

    Returns True if sent successfully.
    """
    if not smtp_email or not smtp_password or not recipient:
        return False

    if not advices:
        return False

    # Build report
    subject = f"TTA Exit Advisor Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    body_parts = [
        "=" * 60,
        "TTA v2 — EXIT ADVISOR REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Positions Analyzed: {len(advices)}",
        "=" * 60,
        "",
    ]

    # Summary counts
    actions = {}
    for a in advices:
        actions[a.action] = actions.get(a.action, 0) + 1
    body_parts.append("SUMMARY: " + " | ".join(f"{k}: {v}" for k, v in actions.items()))
    body_parts.append("")

    # Per-position details
    for a in advices:
        icon = {
            'HOLD': '🟢', 'TAKE_PARTIAL': '🟡',
            'CLOSE': '🔴', 'TIGHTEN_STOP': '🔵',
        }.get(a.action, '⚪')

        body_parts.extend([
            f"--- {a.ticker} ---",
            f"  Action: {icon} {a.action} (Confidence: {a.confidence}/10)",
            f"  P&L: {a.unrealized_pnl_pct:+.1f}% | Price: ${a.current_price:.2f} | Stop: ${a.current_stop:.2f}",
            f"  Days Held: {a.days_held}",
            f"  Reasoning: {a.reasoning}",
            f"  Risk Note: {a.risk_note}",
        ])
        if a.action == 'TIGHTEN_STOP' and a.suggested_stop > 0:
            body_parts.append(f"  Suggested Stop: ${a.suggested_stop:.2f}")
        if a.action == 'TAKE_PARTIAL' and a.partial_pct > 0:
            body_parts.append(f"  Sell: {a.partial_pct}% of position")
        body_parts.append("")

    body = "\n".join(body_parts)

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(smtp_email, smtp_password)
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"[exit_advisor] Email error: {e}")
        return False
