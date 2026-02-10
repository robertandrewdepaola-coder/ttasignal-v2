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


def _format_fundamental_profile(profile: Dict) -> str:
    """Format banker-grade fundamental profile for AI prompt."""
    if not profile or profile.get('error'):
        return "No fundamental profile available."

    lines = []

    # Business identity
    name = profile.get('name', '?')
    sector = profile.get('sector', '?')
    industry = profile.get('industry', '?')
    mcap = profile.get('market_cap')
    mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap >= 1e9 else (
        f"${mcap/1e6:.0f}M" if mcap else "?"
    )
    lines.append(f"COMPANY: {name} | {sector} / {industry} | Mkt Cap: {mcap_str}")

    if profile.get('business_summary'):
        lines.append(f"BUSINESS: {profile['business_summary'][:300]}")

    # Revenue & Growth
    lines.append("\nGROWTH:")
    rev = profile.get('total_revenue')
    if rev:
        rev_str = f"${rev/1e9:.2f}B" if rev >= 1e9 else f"${rev/1e6:.0f}M"
        lines.append(f"  Revenue: {rev_str}")
    if profile.get('revenue_growth_yoy') is not None:
        lines.append(f"  Revenue Growth (YoY): {profile['revenue_growth_yoy']*100:.1f}%")
    if profile.get('revenue_growth_quarterly') is not None:
        lines.append(f"  Revenue Growth (QoQ): {profile['revenue_growth_quarterly']*100:.1f}%")
    if profile.get('earnings_growth_yoy') is not None:
        lines.append(f"  Earnings Growth (YoY): {profile['earnings_growth_yoy']*100:.1f}%")

    # Profitability
    lines.append("\nPROFITABILITY:")
    for key, label in [
        ('gross_margin', 'Gross Margin'),
        ('operating_margin', 'Operating Margin'),
        ('profit_margin', 'Net Margin'),
        ('ebitda_margin', 'EBITDA Margin'),
        ('return_on_equity', 'ROE'),
        ('return_on_assets', 'ROA'),
    ]:
        val = profile.get(key)
        if val is not None:
            lines.append(f"  {label}: {val*100:.1f}%")

    ebitda = profile.get('ebitda')
    if ebitda:
        ebitda_str = f"${ebitda/1e9:.2f}B" if abs(ebitda) >= 1e9 else f"${ebitda/1e6:.0f}M"
        lines.append(f"  EBITDA: {ebitda_str}")

    # Cash Flow
    fcf = profile.get('free_cash_flow')
    ocf = profile.get('operating_cash_flow')
    if fcf or ocf:
        lines.append("\nCASH FLOW:")
        if ocf:
            lines.append(f"  Operating CF: ${ocf/1e6:.0f}M")
        if fcf:
            lines.append(f"  Free CF: ${fcf/1e6:.0f}M")
            if rev and rev > 0:
                lines.append(f"  FCF Yield: {fcf/rev*100:.1f}% of revenue")

    # Balance Sheet
    lines.append("\nBALANCE SHEET:")
    debt = profile.get('total_debt')
    cash = profile.get('total_cash')
    if debt:
        lines.append(f"  Total Debt: ${debt/1e6:.0f}M")
    if cash:
        lines.append(f"  Cash: ${cash/1e6:.0f}M")
    if debt and cash:
        lines.append(f"  Net Debt: ${(debt-cash)/1e6:.0f}M")
    if profile.get('debt_to_equity') is not None:
        lines.append(f"  Debt/Equity: {profile['debt_to_equity']:.1f}")
    if profile.get('current_ratio') is not None:
        lines.append(f"  Current Ratio: {profile['current_ratio']:.2f}")

    # Valuation
    lines.append("\nVALUATION MULTIPLES:")
    for key, label in [
        ('trailing_pe', 'Trailing P/E'),
        ('forward_pe', 'Forward P/E'),
        ('peg_ratio', 'PEG Ratio'),
        ('price_to_book', 'P/Book'),
        ('price_to_sales', 'P/Sales'),
        ('ev_to_ebitda', 'EV/EBITDA'),
        ('ev_to_revenue', 'EV/Revenue'),
    ]:
        val = profile.get(key)
        if val is not None:
            lines.append(f"  {label}: {val:.2f}")

    # Ownership & Short Interest
    lines.append("\nOWNERSHIP:")
    if profile.get('insider_pct') is not None:
        lines.append(f"  Insider: {profile['insider_pct']*100:.1f}%")
    if profile.get('institution_pct') is not None:
        lines.append(f"  Institutional: {profile['institution_pct']*100:.1f}%")
    if profile.get('short_pct_float') is not None:
        lines.append(f"  Short % Float: {profile['short_pct_float']*100:.1f}%")
    if profile.get('short_ratio') is not None:
        lines.append(f"  Short Ratio (days to cover): {profile['short_ratio']:.1f}")

    # Earnings
    if profile.get('next_earnings'):
        lines.append(f"\nNEXT EARNINGS: {profile['next_earnings']}")
    if profile.get('last_earnings_surprise_pct') is not None:
        lines.append(f"  Last Surprise: {profile['last_earnings_surprise_pct']:+.1f}%")

    return "\n".join(lines)


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
                    fundamentals: Dict = None,
                    fundamental_profile: Dict = None) -> str:
    """
    Build enhanced AI analysis prompt with fundamental profile
    and actionable breakout/resistance guidance.
    """
    prompt = f"""You are a senior equity analyst at a top investment bank reviewing {ticker} for a swing/position trade.
Your job is to synthesize technical signals with fundamental reality to give an actionable decision.
The trader sees the technical scanner output — you must add the WHY behind the price action.

══════════════════════════════════════════════════
TECHNICAL SIGNAL (what the scanner says)
══════════════════════════════════════════════════
{_format_signal_status(signal, recommendation)}

══════════════════════════════════════════════════
MULTI-TIMEFRAME ALIGNMENT
══════════════════════════════════════════════════
{_format_timeframes(signal)}

══════════════════════════════════════════════════
CHART STRUCTURE (Weinstein Stage)
══════════════════════════════════════════════════
{_format_chart_structure(signal)}

══════════════════════════════════════════════════
OVERHEAD RESISTANCE
══════════════════════════════════════════════════
{_format_overhead_resistance(signal)}

══════════════════════════════════════════════════
VOLUME & RELATIVE STRENGTH
══════════════════════════════════════════════════
{_format_volume(signal)}
{_format_relative_strength(signal)}

══════════════════════════════════════════════════
KEY LEVELS & TRADE SETUP
══════════════════════════════════════════════════
{_format_key_levels(signal)}
{_format_stops(signal)}

══════════════════════════════════════════════════
QUALITY SCORE (Historical Backtest)
══════════════════════════════════════════════════
{_format_quality(quality)}

══════════════════════════════════════════════════
FUNDAMENTAL PROFILE
══════════════════════════════════════════════════
{_format_fundamental_profile(fundamental_profile) if fundamental_profile else 'No fundamental data.'}

══════════════════════════════════════════════════
MARKET SENTIMENT DATA
══════════════════════════════════════════════════
{_format_fundamentals(fundamentals)}

══════════════════════════════════════════════════
YOUR ANALYSIS — Respond with EXACTLY this format:
══════════════════════════════════════════════════

CONVICTION: [1-10]

ACTION: [BUY NOW / WAIT FOR BREAKOUT / WAIT FOR PULLBACK / SKIP]

RESISTANCE VERDICT: [Is overhead resistance a problem? Should trader wait for breakout? Specify the exact price level that must break, what volume confirms it (e.g. "2x avg"), and what a failed breakout looks like. If no significant resistance, say "Clear path — enter on signal." 2-3 sentences max.]

WHY IT'S MOVING: [What fundamental catalyst explains the current price action? Recent earnings beat/miss? Sector rotation? New product? M&A? Macro tailwind? If nothing obvious, say "Technical momentum — no fundamental catalyst visible." 1-2 sentences.]

FUNDAMENTAL QUALITY: [Rate the business A/B/C/D using this framework:
A = Strong moat, growing revenue, profitable, clean balance sheet
B = Decent business, some growth, acceptable margins
C = Weak fundamentals, declining margins, or high debt
D = Speculative, no profits, or deteriorating rapidly
Include 1 sentence explaining the rating with specific numbers.]

BULL CASE: [Best realistic outcome in 3-6 months, with a price target and the catalyst that gets it there. 1-2 sentences.]

BEAR CASE: [What kills this trade? Specific risk with price level. 1-2 sentences.]

RED FLAGS: [1-3 specific concerns, or "None" if clean]

POSITION SIZING: [Full (100%) / Reduced (75%) / Small (50%) / Skip — with 1 reason]

Keep total response under 250 words. No fluff. Be specific with prices and percentages."""

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
        'action': '',
        'resistance_verdict': '',
        'why_moving': '',
        'fundamental_quality': '',
        'bull_case': '',
        'bear_case': '',
        'red_flags': '',
        'position_sizing': '',
        # Legacy compat
        'scanner_misses': '',
        'timing': '',
    }

    lines = text.strip().split('\n')
    current_field = None
    current_value = []

    field_map = {
        'CONVICTION': 'conviction',
        'ACTION': 'action',
        'RESISTANCE VERDICT': 'resistance_verdict',
        'WHY IT\'S MOVING': 'why_moving',
        'WHY ITS MOVING': 'why_moving',
        'FUNDAMENTAL QUALITY': 'fundamental_quality',
        'BULL CASE': 'bull_case',
        'BEAR CASE': 'bear_case',
        'RED FLAGS': 'red_flags',
        'POSITION SIZING': 'position_sizing',
        # Legacy
        'WHAT THE SCANNER MISSES': 'scanner_misses',
        'TIMING': 'timing',
    }

    def _flush():
        nonlocal current_field, current_value
        if current_field and current_value:
            val = ' '.join(current_value).strip()
            if current_field == 'conviction':
                for char in val[:10]:
                    if char.isdigit():
                        parsed['conviction'] = int(char)
                        break
                if '10' in val[:5]:
                    parsed['conviction'] = 10
            else:
                parsed[current_field] = val
        current_field = None
        current_value = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check if this line starts a new field
        matched = False
        for prefix, field_key in field_map.items():
            if line_clean.upper().startswith(prefix + ':'):
                _flush()
                current_field = field_key
                remainder = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ''
                current_value = [remainder] if remainder else []
                matched = True
                break

        if not matched and current_field:
            current_value.append(line_clean)

    _flush()

    # Map action → timing for backward compat
    if parsed['action'] and not parsed['timing']:
        parsed['timing'] = parsed['action']
    if parsed['resistance_verdict'] and not parsed['scanner_misses']:
        parsed['scanner_misses'] = parsed['resistance_verdict']

    return parsed


# =============================================================================
# FALLBACK — System analysis when no AI available
# =============================================================================

def generate_system_analysis(ticker: str,
                             signal: EntrySignal,
                             recommendation: Dict,
                             quality: Dict,
                             fundamental_profile: Dict = None) -> Dict[str, Any]:
    """
    Generate analysis without AI, using pure logic on computed data.
    Used when Gemini/OpenAI are unavailable.
    """
    result = {
        'raw_text': '',
        'conviction': recommendation.get('conviction', 0),
        'action': '',
        'resistance_verdict': '',
        'why_moving': '',
        'fundamental_quality': '',
        'bull_case': '',
        'bear_case': '',
        'red_flags': '',
        'position_sizing': '',
        'scanner_misses': '',
        'timing': '',
        'provider': 'system',
        'success': True,
        'error': None,
    }

    # ── Action / Timing ───────────────────────────────────────────
    ores = signal.overhead_resistance
    critical_near = False
    if ores and ores.get('distance_to_critical_pct') and ores['distance_to_critical_pct'] < 3:
        critical_near = True

    if signal.is_valid and recommendation.get('conviction', 0) >= 7 and not critical_near:
        result['action'] = 'BUY NOW'
        result['timing'] = 'Enter now — signal fresh and aligned'
    elif signal.is_valid and critical_near:
        crit_price = ores['critical_level']['price']
        result['action'] = f'WAIT FOR BREAKOUT above ${crit_price:.2f}'
        result['timing'] = f'Wait for breakout above ${crit_price:.2f} on 2x volume'
    elif signal.is_valid:
        result['action'] = 'BUY NOW'
        result['timing'] = 'Enter with caution — some concerns noted'
    elif signal.macd.get('bullish') and signal.ao.get('positive'):
        result['action'] = 'WAIT FOR PULLBACK'
        result['timing'] = 'Wait for fresh MACD cross confirmation'
    else:
        result['action'] = 'SKIP'
        result['timing'] = 'Skip — conditions not met'

    # ── Resistance Verdict ────────────────────────────────────────
    if ores and ores.get('critical_level'):
        dist = ores.get('distance_to_critical_pct', 0)
        crit = ores['critical_level']
        vol_needed = ores.get('breakout_volume_needed', '1.5-2')
        if dist and dist < 3:
            result['resistance_verdict'] = (
                f"Critical resistance at ${crit['price']:.2f} ({dist:.1f}% above). "
                f"Wait for a daily close above ${crit['price']:.2f} on {vol_needed}x average volume. "
                f"Failed breakout = close back below on next day → exit immediately."
            )
        elif dist and dist < 8:
            result['resistance_verdict'] = (
                f"Resistance at ${crit['price']:.2f} ({dist:.1f}% above) — room to run but "
                f"watch for stall. Clear path if volume confirms above that level."
            )
        else:
            result['resistance_verdict'] = "No immediate overhead resistance — clear path for entry."
    else:
        result['resistance_verdict'] = "No significant overhead resistance detected — clear path."

    # ── Why It's Moving ───────────────────────────────────────────
    insights = []
    rs = signal.relative_strength
    if rs and rs.get('rs_trend') == 'improving':
        insights.append("Relative strength improving — gaining market leadership")
    elif rs and rs.get('rs_trend') == 'deteriorating':
        insights.append("Relative strength weakening vs SPY")

    vol = signal.volume
    if vol and vol.get('accum_dist_trend') == 'accumulating':
        insights.append("Accumulation volume — institutional buying")
    elif vol and vol.get('accum_dist_trend') == 'distributing':
        insights.append("Distribution pattern — smart money selling")

    ws = signal.weinstein
    if ws.get('stage') == 2 and ws.get('trend_maturity') == 'early':
        insights.append("Early Stage 2 advance — strongest trend phase")
    elif ws.get('stage') == 2 and ws.get('trend_maturity') == 'late':
        insights.append("Late Stage 2 — trend extended")

    result['why_moving'] = '. '.join(insights[:2]) if insights else (
        "Technical momentum — no fundamental catalyst visible from available data."
    )
    result['scanner_misses'] = result['why_moving']

    # ── Fundamental Quality ───────────────────────────────────────
    if fundamental_profile and not fundamental_profile.get('error'):
        fp = fundamental_profile
        grade_score = 0
        reasons = []

        # Revenue growth
        rg = fp.get('revenue_growth_yoy')
        if rg is not None:
            if rg > 0.20:
                grade_score += 3
                reasons.append(f"Strong revenue growth {rg*100:.0f}%")
            elif rg > 0.05:
                grade_score += 2
            elif rg > 0:
                grade_score += 1
            else:
                reasons.append(f"Revenue declining {rg*100:.0f}%")

        # Profitability
        pm = fp.get('profit_margin')
        if pm is not None:
            if pm > 0.15:
                grade_score += 2
            elif pm > 0.05:
                grade_score += 1
            elif pm < 0:
                reasons.append("Not profitable")

        # FCF
        fcf = fp.get('free_cash_flow')
        if fcf is not None and fcf > 0:
            grade_score += 1
        elif fcf is not None and fcf < 0:
            reasons.append("Negative FCF")

        # Balance sheet
        dte = fp.get('debt_to_equity')
        if dte is not None:
            if dte < 50:
                grade_score += 1
            elif dte > 200:
                reasons.append(f"High leverage D/E={dte:.0f}")

        # ROE
        roe = fp.get('return_on_equity')
        if roe is not None and roe > 0.15:
            grade_score += 1

        if grade_score >= 7:
            grade = 'A'
        elif grade_score >= 5:
            grade = 'B'
        elif grade_score >= 3:
            grade = 'C'
        else:
            grade = 'D'

        reason_str = reasons[0] if reasons else "Based on available metrics"
        mcap = fp.get('market_cap')
        mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap >= 1e9 else (
            f"${mcap/1e6:.0f}M" if mcap else ""
        )

        result['fundamental_quality'] = f"{grade} — {reason_str}. {mcap_str} market cap."
    else:
        result['fundamental_quality'] = "N/A — Fundamental data unavailable"

    # ── Bull/Bear Cases ───────────────────────────────────────────
    current_price = signal.key_levels.get('price', 0) if signal.key_levels else 0
    target = signal.stops.get('target', 0) if signal.stops else 0

    if target and current_price:
        upside = (target - current_price) / current_price * 100
        result['bull_case'] = (
            f"Technical target ${target:.2f} (+{upside:.0f}%). "
            f"Higher timeframe alignment and momentum favor continuation."
        )
    else:
        result['bull_case'] = "Momentum continuation on timeframe alignment."

    stop = signal.stops.get('stop', 0) if signal.stops else 0
    if stop and current_price:
        downside = (current_price - stop) / current_price * 100
        result['bear_case'] = (
            f"Break below ${stop:.2f} ({downside:.1f}% risk) invalidates the setup. "
        )
        if not signal.weekly_macd.get('bullish', True):
            result['bear_case'] += "Weekly MACD bearish adds headwind risk."
    else:
        result['bear_case'] = "Higher timeframe reversal or macro deterioration."

    # ── Red Flags ─────────────────────────────────────────────────
    flags = []
    if ws.get('stage') in [3, 4]:
        flags.append(f"Weinstein {ws.get('label', 'Stage ?')} — unfavorable structure")
    if not signal.weekly_macd.get('bullish', False):
        flags.append("Weekly MACD bearish")
    if not signal.monthly_macd.get('bullish', True):
        flags.append("Monthly MACD bearish")
    if critical_near:
        flags.append(f"Resistance at ${ores['critical_level']['price']:.2f}")
    if vol and vol.get('accum_dist_trend') == 'distributing':
        flags.append("Distribution volume")
    if rs and rs.get('rs_1mo') is not None and rs['rs_1mo'] < -5:
        flags.append(f"Weak vs SPY ({rs['rs_1mo']:+.1f}%)")

    result['red_flags'] = '; '.join(flags[:3]) if flags else 'None'

    # ── Position Sizing ───────────────────────────────────────────
    conv = recommendation.get('conviction', 0)
    if conv >= 8:
        result['position_sizing'] = 'Full (100%) — high conviction setup'
    elif conv >= 6:
        result['position_sizing'] = 'Reduced (75%) — moderate conviction'
    elif conv >= 4:
        result['position_sizing'] = 'Small (50%) — low conviction, tight stop'
    else:
        result['position_sizing'] = 'Skip — insufficient conviction'

    # Build narrative text
    parts = [
        f"CONVICTION: {result['conviction']}/10",
        f"ACTION: {result['action']}",
        f"RESISTANCE VERDICT: {result['resistance_verdict']}",
        f"WHY IT'S MOVING: {result['why_moving']}",
        f"FUNDAMENTAL QUALITY: {result['fundamental_quality']}",
        f"BULL CASE: {result['bull_case']}",
        f"BEAR CASE: {result['bear_case']}",
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
            fundamental_profile: Dict = None,
            gemini_model=None,
            openai_client=None) -> Dict[str, Any]:
    """
    Run enhanced AI analysis on a ticker.

    Args:
        ticker: Stock symbol
        signal: EntrySignal from signal_engine.validate_entry()
        recommendation: Dict from scanner_engine.generate_recommendation()
        quality: Dict from scanner_engine.calculate_quality_score()
        fundamentals: Dict from data_fetcher (options, insider, etc.)
        fundamental_profile: Dict from data_fetcher.fetch_fundamental_profile()
        gemini_model: Google Gemini model instance
        openai_client: OpenAI client instance
    """
    # Build prompt
    prompt = build_ai_prompt(ticker, signal, recommendation, quality,
                             fundamentals, fundamental_profile)

    # Try AI first
    if gemini_model is not None or openai_client is not None:
        result = call_ai(prompt, gemini_model=gemini_model, openai_client=openai_client)

        # If AI failed, use system fallback
        if not result['success']:
            result = generate_system_analysis(ticker, signal, recommendation,
                                              quality, fundamental_profile)
            result['note'] = 'AI unavailable — using system analysis'
    else:
        # No AI configured — use system analysis
        result = generate_system_analysis(ticker, signal, recommendation,
                                          quality, fundamental_profile)
        result['note'] = 'No AI configured — using system analysis'

    result['ticker'] = ticker
    result['timestamp'] = datetime.now().isoformat()
    # Attach fundamental profile for UI display
    result['fundamental_profile'] = fundamental_profile

    return result
