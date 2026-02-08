"""
TTA v2 AI Analysis — Enhanced Trade Intelligence
==================================================

Builds rich context from signal_engine computations + data_fetcher fundamentals,
then sends to Gemini for synthesis that adds genuine edge beyond mechanical signals.

The AI receives PRE-COMPUTED data (Weinstein stage, overhead resistance, volume
profile, relative strength, options sentiment, insider activity) and its job is
to SYNTHESIZE — what do these data points mean TOGETHER that the scanner can't see.

NO yfinance calls. NO indicator calculations. Pure prompt construction + AI call.

Version: 2.0.0 (2026-02-08)
"""

from typing import Dict, Any, Optional
from datetime import datetime

from signal_engine import EntrySignal


# =============================================================================
# CONTEXT FORMATTERS — Turn computed dicts into readable prompt sections
# =============================================================================

def _format_signal_status(signal: EntrySignal, rec: Dict) -> str:
    """Format the mechanical scanner result."""
    lines = []
    lines.append(f"Signal Type: {rec.get('signal_type', 'None')}")
    lines.append(f"Recommendation: {rec.get('recommendation', 'SKIP')}")
    lines.append(f"Conviction: {rec.get('conviction', 0)}/10")

    m = signal.macd
    lines.append(f"Daily MACD: {'BULLISH' if m.get('bullish') else 'BEARISH'} "
                 f"(hist={m.get('histogram', 0):+.4f})")
    if m.get('cross_recent'):
        lines.append(f"  Cross: {m.get('cross_bars_ago', '?')} bars ago")
    if m.get('weakening'):
        lines.append("  ⚠ Histogram weakening")
    if m.get('near_cross'):
        lines.append("  ⚠ Near crossover")

    a = signal.ao
    lines.append(f"Daily AO: {'POSITIVE' if a.get('positive') else 'NEGATIVE'} "
                 f"(value={a.get('value', 0):+.4f}, trend={a.get('trend', '?')})")

    return "\n".join(lines)


def _format_timeframes(signal: EntrySignal) -> str:
    """Format multi-timeframe alignment."""
    lines = []

    w = signal.weekly_macd
    if w:
        lines.append(f"Weekly MACD: {'BULLISH ✅' if w.get('bullish') else 'BEARISH ❌'} "
                     f"(hist={w.get('histogram', 0):+.4f})")
        if w.get('cross_up'):
            lines.append("  ↗ Just crossed bullish")
        if w.get('cross_down'):
            lines.append("  ↘ Just crossed bearish")

    mo = signal.monthly_macd
    mo_ao = signal.monthly_ao
    if mo:
        lines.append(f"Monthly MACD: {'BULLISH ✅' if mo.get('bullish') else 'BEARISH ❌'} "
                     f"(hist={mo.get('histogram', 0):+.4f})")
    if mo_ao:
        lines.append(f"Monthly AO: {'POSITIVE ✅' if mo_ao.get('positive') else 'NEGATIVE ❌'}")

    # Alignment assessment
    d_bull = signal.macd.get('bullish', False)
    w_bull = w.get('bullish', False) if w else False
    m_bull = mo.get('bullish', False) if mo else False

    if d_bull and w_bull and m_bull:
        lines.append("ALIGNMENT: All timeframes bullish ✅✅✅")
    elif d_bull and w_bull:
        lines.append("ALIGNMENT: Daily + Weekly aligned, Monthly divergent")
    elif d_bull:
        lines.append("ALIGNMENT: Only daily bullish — higher timeframes opposing")
    else:
        lines.append("ALIGNMENT: Daily not bullish")

    return "\n".join(lines)


def _format_chart_structure(signal: EntrySignal) -> str:
    """Format Weinstein stage and trend context."""
    w = signal.weinstein
    if not w or w.get('stage', 0) == 0:
        return "Insufficient data for chart structure analysis."

    lines = []
    lines.append(f"Weinstein: {w.get('label', '?')} — Maturity: {w.get('trend_maturity', '?')}")
    lines.append(f"150-day SMA: {w.get('sma150_slope', '?')} (ROC: {w.get('sma150_roc_pct', 0):+.2f}%)")
    lines.append(f"Price vs 150-SMA: {w.get('price_vs_sma150', '?')} ({w.get('pct_from_sma150', 0):+.1f}%)")
    lines.append(f"52-week position: {w.get('pct_from_52w_high', 0):+.1f}% from high, "
                 f"{w.get('pct_from_52w_low', 0):+.1f}% from low")

    return "\n".join(lines)


def _format_overhead_resistance(signal: EntrySignal) -> str:
    """Format overhead resistance levels."""
    r = signal.overhead_resistance
    if not r or not r.get('levels'):
        return "No significant overhead resistance detected."

    lines = []
    for lev in r['levels']:
        lines.append(f"  {lev['description']}")

    if r.get('critical_level'):
        c = r['critical_level']
        lines.append(f"\nCRITICAL LEVEL: ${c['price']:.2f} ({c['type']})")
        lines.append(f"  Distance: {r.get('distance_to_critical_pct', '?')}% above current price")
        lines.append(f"  Breakout volume needed: {r.get('breakout_volume_needed', '?')}x average")

    lines.append(f"\n{r.get('assessment', '')}")

    return "\n".join(lines)


def _format_volume(signal: EntrySignal) -> str:
    """Format volume analysis."""
    v = signal.volume
    if not v:
        return "No volume data."

    lines = []
    lines.append(f"50-day avg volume: {v.get('avg_volume_50d', 0):,.0f}")
    if v.get('cross_volume_ratio'):
        lines.append(f"Volume at MACD cross: {v['cross_volume_ratio']}x average")
    lines.append(f"Accumulation/Distribution: {v.get('accum_dist_trend', '?')}")
    lines.append(f"Volume trend (20d): {v.get('volume_trend_20d', '?')}")
    if v.get('big_volume_days_20d', 0) > 0:
        lines.append(f"Big volume days (20d): {v['big_volume_days_20d']}")

    return "\n".join(lines)


def _format_key_levels(signal: EntrySignal) -> str:
    """Format price vs key levels."""
    k = signal.key_levels
    if not k or k.get('price') is None:
        return "No key levels data."

    lines = []
    lines.append(f"Price: ${k['price']:.2f}")
    if k.get('sma50'):
        lines.append(f"50 SMA: ${k['sma50']:.2f} ({k.get('price_vs_sma50', '?')}, "
                     f"{k.get('pct_from_sma50', 0):+.1f}%)")
    if k.get('sma200'):
        lines.append(f"200 SMA: ${k['sma200']:.2f} ({k.get('price_vs_sma200', '?')}, "
                     f"{k.get('pct_from_sma200', 0):+.1f}%)")
    lines.append(f"Golden cross: {'Yes' if k.get('golden_cross') else 'No'}")
    if k.get('nearest_support'):
        lines.append(f"Nearest support: ${k['nearest_support']:.2f}")
    if k.get('nearest_resistance'):
        lines.append(f"Nearest resistance: ${k['nearest_resistance']:.2f}")
    if k.get('at_key_level'):
        lines.append("⚡ Price is AT a key level")

    return "\n".join(lines)


def _format_relative_strength(signal: EntrySignal) -> str:
    """Format relative strength vs SPY."""
    rs = signal.relative_strength
    if not rs or rs.get('rs_1mo') is None:
        return "No relative strength data."

    lines = []
    lines.append(f"RS vs SPY 1 month: {rs.get('rs_1mo', 0):+.1f}%")
    lines.append(f"RS vs SPY 3 month: {rs.get('rs_3mo', 0):+.1f}%")
    lines.append(f"RS vs SPY 6 month: {rs.get('rs_6mo', 0):+.1f}%")
    lines.append(f"RS vs SPY 12 month: {rs.get('rs_12mo', 0):+.1f}%")
    lines.append(f"RS trend: {rs.get('rs_trend', '?')}")

    return "\n".join(lines)


def _format_stops(signal: EntrySignal) -> str:
    """Format trade setup."""
    s = signal.stops
    if not s:
        return "No stop data."

    lines = []
    lines.append(f"Entry: ${s.get('entry', 0):.2f}")
    lines.append(f"Stop: ${s.get('stop', 0):.2f} ({s.get('stop_method', '?')})")
    lines.append(f"Target: ${s.get('target', 0):.2f}")
    lines.append(f"Risk: {s.get('risk_pct', 0):.1f}% (${s.get('risk_per_share', 0):.2f}/share)")
    lines.append(f"Reward:Risk = {s.get('reward_risk', '?')}")

    return "\n".join(lines)


def _format_fundamentals(fundamentals: Dict) -> str:
    """Format fundamental data from data_fetcher."""
    if not fundamentals:
        return "No fundamental data available."

    lines = []

    # Options
    opt = fundamentals.get('options', {})
    if opt and not opt.get('error'):
        pc = opt.get('put_call_ratio')
        lines.append(f"OPTIONS (nearest expiry: {opt.get('nearest_expiry', '?')}):")
        if pc is not None:
            sentiment = 'bullish' if pc < 0.7 else ('bearish' if pc > 1.3 else 'neutral')
            lines.append(f"  Put/Call volume ratio: {pc:.2f} ({sentiment})")
        pc_oi = opt.get('put_call_oi_ratio')
        if pc_oi is not None:
            lines.append(f"  Put/Call OI ratio: {pc_oi:.2f}")
        if opt.get('unusual_activity'):
            lines.append("  ⚠ UNUSUAL OPTIONS ACTIVITY detected")
        if opt.get('max_pain'):
            lines.append(f"  Max pain: ${opt['max_pain']:.2f}")

    # Insider activity
    ins = fundamentals.get('insider', {})
    if ins and not ins.get('error'):
        lines.append(f"\nINSIDER ACTIVITY (90 days):")
        lines.append(f"  Buys: {ins.get('buys_90d', 0)}, Sells: {ins.get('sells_90d', 0)}")
        lines.append(f"  Net: {ins.get('net_activity', '?')}")
        if ins.get('total_buy_value', 0) > 0:
            lines.append(f"  Buy value: ${ins['total_buy_value']:,.0f}")
        if ins.get('total_sell_value', 0) > 0:
            lines.append(f"  Sell value: ${ins['total_sell_value']:,.0f}")

    # Institutional
    inst = fundamentals.get('institutional', {})
    if inst and not inst.get('error'):
        holders = inst.get('top_holders', [])
        if holders:
            lines.append(f"\nINSTITUTIONAL HOLDERS (top {len(holders)}):")
            for h in holders[:3]:
                lines.append(f"  {h.get('name', '?')}: {h.get('pct_out', 0):.1f}%")

    # Earnings
    earn = fundamentals.get('earnings', {})
    if earn and not earn.get('error'):
        if earn.get('next_earnings'):
            days = earn.get('days_until_earnings', '?')
            lines.append(f"\nEARNINGS: {earn['next_earnings']} ({days} days away)")
            if isinstance(days, (int, float)) and days <= 14:
                lines.append("  ⚠ EARNINGS IMMINENT — consider waiting or sizing down")

    # Ticker info
    info = fundamentals.get('info', {})
    if info and not info.get('error'):
        if info.get('sector'):
            lines.append(f"\nSector: {info['sector']} / {info.get('industry', '?')}")
        if info.get('short_pct_float') is not None:
            short_pct = info['short_pct_float'] * 100
            lines.append(f"Short interest: {short_pct:.1f}% of float")
            if short_pct > 10:
                lines.append("  ⚠ High short interest — short squeeze potential")

    return "\n".join(lines) if lines else "No fundamental data available."


def _format_quality(quality: Dict) -> str:
    """Format quality score from mini-backtest."""
    if not quality or quality.get('error'):
        return f"Quality: N/A ({quality.get('error', 'no data')})"

    lines = []
    lines.append(f"Grade: {quality.get('quality_grade', '?')} "
                 f"({quality.get('quality_score', 0)}/100)")
    lines.append(f"Win rate: {quality.get('win_rate', 0):.0f}%")
    lines.append(f"Avg return: {quality.get('avg_return', 0):+.1f}%")
    lines.append(f"Signals found (3yr): {quality.get('signals_found', 0)}")
    lines.append(f"Best: {quality.get('best_return', 0):+.1f}% / "
                 f"Worst: {quality.get('worst_return', 0):+.1f}%")

    return "\n".join(lines)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_ai_prompt(ticker: str,
                    signal: EntrySignal,
                    recommendation: Dict,
                    quality: Dict,
                    fundamentals: Dict = None) -> str:
    """
    Build the enhanced AI analysis prompt.

    This is the core of v2 AI — instead of echoing scanner data,
    we provide pre-computed analysis modules and ask the AI to SYNTHESIZE.
    """
    prompt = f"""You are a senior technical analyst reviewing {ticker} for a swing trade entry.
Your job is to add INSIGHT the mechanical scanner cannot — synthesize the data below
and tell the trader something they don't already know.

══════════════════════════════════════════════════
SCANNER SIGNAL (mechanical — this is what the trader already sees)
══════════════════════════════════════════════════
{_format_signal_status(signal, recommendation)}

══════════════════════════════════════════════════
MULTI-TIMEFRAME ALIGNMENT
══════════════════════════════════════════════════
{_format_timeframes(signal)}

══════════════════════════════════════════════════
CHART STRUCTURE (Weinstein Stage Analysis)
══════════════════════════════════════════════════
{_format_chart_structure(signal)}

══════════════════════════════════════════════════
OVERHEAD RESISTANCE — What must break for a sustained trend
══════════════════════════════════════════════════
{_format_overhead_resistance(signal)}

══════════════════════════════════════════════════
VOLUME ANALYSIS
══════════════════════════════════════════════════
{_format_volume(signal)}

══════════════════════════════════════════════════
KEY LEVELS
══════════════════════════════════════════════════
{_format_key_levels(signal)}

══════════════════════════════════════════════════
RELATIVE STRENGTH vs SPY
══════════════════════════════════════════════════
{_format_relative_strength(signal)}

══════════════════════════════════════════════════
TRADE SETUP
══════════════════════════════════════════════════
{_format_stops(signal)}

══════════════════════════════════════════════════
QUALITY SCORE (Historical Backtest on this ticker)
══════════════════════════════════════════════════
{_format_quality(quality)}

══════════════════════════════════════════════════
FUNDAMENTAL & SENTIMENT DATA
══════════════════════════════════════════════════
{_format_fundamentals(fundamentals)}

══════════════════════════════════════════════════
YOUR ANALYSIS — Be concise, be honest, add edge
══════════════════════════════════════════════════

Respond with EXACTLY this format:

CONVICTION: [1-10] — How confident is this setup overall?

WHAT THE SCANNER MISSES: [1-2 sentences — the key insight the mechanical signal doesn't capture. Consider the overhead resistance levels, volume patterns, relative strength trend, Weinstein stage maturity, and any fundamental red flags.]

TIMING: [Enter now / Wait for pullback to $X / Wait for breakout above $X / Skip]

RED FLAGS: [1-3 specific concerns, or "None" if clean setup]

POSITION SIZING: [Full size / Reduced (75%) / Small (50%) / Skip] — with reason

Keep total response under 8 sentences. No fluff. Trader wants actionable edge."""

    return prompt


# =============================================================================
# AI CALL — Gemini preferred, OpenAI fallback
# =============================================================================

def call_ai(prompt: str,
            gemini_model=None,
            openai_client=None) -> Dict[str, Any]:
    """
    Send prompt to AI and parse response.

    Tries Gemini first (preferred), falls back to OpenAI.
    Returns raw text and parsed fields.
    """
    result = {
        'raw_text': '',
        'conviction': 0,
        'scanner_misses': '',
        'timing': '',
        'red_flags': '',
        'position_sizing': '',
        'provider': None,
        'success': False,
        'error': None,
    }

    narrative = None

    # Try Gemini
    if gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            narrative = response.text
            result['provider'] = 'gemini'
        except Exception as e:
            result['gemini_error'] = str(e)[:200]

    # Try OpenAI fallback
    if narrative is None and openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a senior technical analyst. Be concise, honest, actionable."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.5
            )
            narrative = response.choices[0].message.content
            result['provider'] = 'openai'
        except Exception as e:
            result['openai_error'] = str(e)[:200]

    if narrative:
        result['raw_text'] = narrative
        result['success'] = True
        result.update(_parse_ai_response(narrative))
    else:
        result['error'] = 'Both AI providers failed'

    return result


def _parse_ai_response(text: str) -> Dict[str, Any]:
    """Parse structured fields from AI response text."""
    parsed = {
        'conviction': 0,
        'scanner_misses': '',
        'timing': '',
        'red_flags': '',
        'position_sizing': '',
    }

    lines = text.strip().split('\n')

    for line in lines:
        line_upper = line.strip().upper()
        line_clean = line.strip()

        if line_upper.startswith('CONVICTION:'):
            value = line_clean.split(':', 1)[1].strip()
            # Extract number
            for char in value:
                if char.isdigit():
                    parsed['conviction'] = int(char)
                    break
            # Check for "10"
            if '10' in value[:5]:
                parsed['conviction'] = 10

        elif line_upper.startswith('WHAT THE SCANNER MISSES:'):
            parsed['scanner_misses'] = line_clean.split(':', 1)[1].strip()

        elif line_upper.startswith('TIMING:'):
            parsed['timing'] = line_clean.split(':', 1)[1].strip()

        elif line_upper.startswith('RED FLAGS:'):
            parsed['red_flags'] = line_clean.split(':', 1)[1].strip()

        elif line_upper.startswith('POSITION SIZING:'):
            parsed['position_sizing'] = line_clean.split(':', 1)[1].strip()

    return parsed


# =============================================================================
# FALLBACK — System analysis when no AI available
# =============================================================================

def generate_system_analysis(ticker: str,
                             signal: EntrySignal,
                             recommendation: Dict,
                             quality: Dict) -> Dict[str, Any]:
    """
    Generate analysis without AI, using pure logic on computed data.
    Used when Gemini/OpenAI are unavailable.
    """
    result = {
        'raw_text': '',
        'conviction': recommendation.get('conviction', 0),
        'scanner_misses': '',
        'timing': '',
        'red_flags': '',
        'position_sizing': '',
        'provider': 'system',
        'success': True,
        'error': None,
    }

    # Scanner misses — derive from computed data
    insights = []

    # Weinstein stage insight
    ws = signal.weinstein
    if ws.get('stage') == 2 and ws.get('trend_maturity') == 'late':
        insights.append("Late Stage 2 — trend may be extended, watch for Stage 3 transition")
    elif ws.get('stage') == 2 and ws.get('trend_maturity') == 'early':
        insights.append("Early Stage 2 — strongest part of the advance")
    elif ws.get('stage') == 3:
        insights.append("Stage 3 topping pattern — avoid new entries")
    elif ws.get('stage') == 4:
        insights.append("Stage 4 decline — avoid longs")
    elif ws.get('stage') == 1:
        insights.append("Stage 1 base building — not yet ready for trend trades")

    # Overhead resistance insight
    ores = signal.overhead_resistance
    if ores and ores.get('critical_level'):
        dist = ores.get('distance_to_critical_pct', 0)
        if dist and dist < 3:
            insights.append(f"Critical resistance at ${ores['critical_level']['price']:.2f} "
                          f"just {dist:.1f}% above — breakout needed")
        elif dist and dist < 8:
            insights.append(f"Room to ${ores['critical_level']['price']:.2f} "
                          f"({dist:.1f}% above) before major resistance")

    # RS insight
    rs = signal.relative_strength
    if rs and rs.get('rs_trend') == 'deteriorating':
        insights.append("Relative strength vs SPY deteriorating — weakening leadership")
    elif rs and rs.get('rs_trend') == 'improving':
        insights.append("Relative strength improving — gaining market leadership")

    # Volume insight
    vol = signal.volume
    if vol and vol.get('accum_dist_trend') == 'distributing':
        insights.append("Distribution detected — smart money may be selling into strength")
    elif vol and vol.get('accum_dist_trend') == 'accumulating':
        insights.append("Accumulation pattern — volume confirms buying interest")

    result['scanner_misses'] = '. '.join(insights[:2]) if insights else 'No notable divergence from scanner signal.'

    # Timing
    if signal.is_valid and recommendation.get('conviction', 0) >= 7:
        result['timing'] = 'Enter now — signal fresh and aligned'
    elif signal.is_valid:
        result['timing'] = 'Enter with caution — some concerns noted'
    elif signal.macd.get('bullish') and signal.ao.get('positive'):
        result['timing'] = 'Wait for fresh MACD cross confirmation'
    else:
        result['timing'] = 'Skip — conditions not met'

    # Red flags
    flags = []
    if ws.get('stage') in [3, 4]:
        flags.append(f"Weinstein {ws.get('label', 'Stage ?')} — unfavorable trend structure")
    if not signal.weekly_macd.get('bullish', False):
        flags.append("Weekly MACD bearish — higher timeframe headwind")
    if not signal.monthly_macd.get('bullish', True):
        flags.append("Monthly MACD bearish — macro headwind")
    if ores and ores.get('distance_to_critical_pct') and ores['distance_to_critical_pct'] < 3:
        flags.append(f"Immediate resistance at ${ores['critical_level']['price']:.2f}")
    if vol and vol.get('accum_dist_trend') == 'distributing':
        flags.append("Distribution volume pattern")
    if rs and rs.get('rs_1mo') is not None and rs['rs_1mo'] < -5:
        flags.append(f"Weak vs SPY ({rs['rs_1mo']:+.1f}% relative performance)")

    result['red_flags'] = '; '.join(flags[:3]) if flags else 'None'

    # Position sizing
    conv = recommendation.get('conviction', 0)
    if conv >= 8:
        result['position_sizing'] = 'Full size — high conviction setup'
    elif conv >= 6:
        result['position_sizing'] = 'Reduced (75%) — moderate conviction'
    elif conv >= 4:
        result['position_sizing'] = 'Small (50%) — low conviction, tight stop'
    else:
        result['position_sizing'] = 'Skip — insufficient conviction'

    # Build narrative text
    parts = [
        f"CONVICTION: {result['conviction']}/10",
        f"WHAT THE SCANNER MISSES: {result['scanner_misses']}",
        f"TIMING: {result['timing']}",
        f"RED FLAGS: {result['red_flags']}",
        f"POSITION SIZING: {result['position_sizing']}",
    ]
    result['raw_text'] = '\n'.join(parts)

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def analyze(ticker: str,
            signal: EntrySignal,
            recommendation: Dict,
            quality: Dict,
            fundamentals: Dict = None,
            gemini_model=None,
            openai_client=None) -> Dict[str, Any]:
    """
    Run enhanced AI analysis on a ticker.

    This is the main entry point. Call with pre-computed data from
    signal_engine.validate_entry() and scanner_engine.analyze_ticker().

    Args:
        ticker: Stock symbol
        signal: EntrySignal from signal_engine.validate_entry()
        recommendation: Dict from scanner_engine.generate_recommendation()
        quality: Dict from scanner_engine.calculate_quality_score()
        fundamentals: Dict from data_fetcher.fetch_all_ticker_data(include_fundamentals=True)
        gemini_model: Google Gemini model instance
        openai_client: OpenAI client instance

    Returns:
        Dict with conviction, scanner_misses, timing, red_flags, position_sizing,
        raw_text, provider, success
    """
    # Build prompt
    prompt = build_ai_prompt(ticker, signal, recommendation, quality, fundamentals)

    # Try AI first
    if gemini_model is not None or openai_client is not None:
        result = call_ai(prompt, gemini_model=gemini_model, openai_client=openai_client)

        # If AI failed, use system fallback
        if not result['success']:
            result = generate_system_analysis(ticker, signal, recommendation, quality)
            result['note'] = 'AI unavailable — using system analysis'
    else:
        # No AI configured — use system analysis
        result = generate_system_analysis(ticker, signal, recommendation, quality)
        result['note'] = 'No AI configured — using system analysis'

    result['ticker'] = ticker
    result['timestamp'] = datetime.now().isoformat()

    return result
