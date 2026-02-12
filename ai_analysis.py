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

    # Insider TRANSACTIONS (actual buys/sells, NOT ownership %)
    ins = fundamentals.get('insider', {})
    if ins and not ins.get('error'):
        buys = ins.get('buys_90d', 0)
        sells = ins.get('sells_90d', 0)
        if buys > 0 or sells > 0:
            lines.append(f"\nINSIDER TRANSACTIONS (90 days):")
            lines.append(f"  Buys: {buys}, Sells: {sells}")
            lines.append(f"  Net: {ins.get('net_activity', '?')}")
            if ins.get('total_buy_value', 0) > 0:
                lines.append(f"  Buy value: ${ins['total_buy_value']:,.0f}")
            if ins.get('total_sell_value', 0) > 0:
                lines.append(f"  Sell value: ${ins['total_sell_value']:,.0f}")
        else:
            lines.append(f"\nINSIDER TRANSACTIONS (90 days): None found — no recent insider buying or selling")

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
    lines.append("\nOWNERSHIP (static stake, NOT recent trading activity):")
    if profile.get('insider_pct') is not None:
        lines.append(f"  Insider Ownership: {profile['insider_pct']*100:.1f}% (this is ownership stake, NOT selling)")
    if profile.get('institution_pct') is not None:
        lines.append(f"  Institutional Ownership: {profile['institution_pct']*100:.1f}%")
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

def _format_tradingview(tv_data: Dict) -> str:
    """Format TradingView-TA multi-timeframe summary for prompt."""
    if not tv_data:
        return "TradingView-TA not available."

    lines = []
    for interval, data in tv_data.items():
        if data.get('error'):
            continue
        rec = data.get('recommendation', '?')
        buy = data.get('buy', 0)
        sell = data.get('sell', 0)
        neutral = data.get('neutral', 0)
        ma_rec = data.get('ma_recommendation', '?')
        osc_rec = data.get('osc_recommendation', '?')
        lines.append(f"  {interval}: {rec} (Buy:{buy} Sell:{sell} Neutral:{neutral}) "
                     f"| MAs: {ma_rec} | Oscillators: {osc_rec}")

    if not lines:
        return "TradingView-TA not available."

    # Add key indicators from daily
    daily = tv_data.get('1d', {})
    extras = []
    if daily.get('rsi') is not None:
        extras.append(f"RSI: {daily['rsi']:.1f}")
    if daily.get('adx') is not None:
        extras.append(f"ADX: {daily['adx']:.1f}")
    if daily.get('cci') is not None:
        extras.append(f"CCI: {daily['cci']:.1f}")
    if extras:
        lines.append(f"  Indicators: {', '.join(extras)}")

    return "\n".join(lines)


def _format_news(news_data: Dict) -> str:
    """Format Finnhub news headlines for prompt."""
    if not news_data or news_data.get('error'):
        return "No recent news available."

    headlines = news_data.get('headlines', [])
    if not headlines:
        return "No recent news found."

    lines = []
    for h in headlines[:5]:
        lines.append(f"  [{h.get('datetime', '?')}] {h.get('headline', '?')} ({h.get('source', '?')})")
    return "\n".join(lines)


def _format_market_intelligence(intel: Dict) -> str:
    """Format market intelligence for AI prompt."""
    if not intel or intel.get('error'):
        return "No market intelligence available."

    lines = []

    # Analyst consensus
    consensus = intel.get('analyst_consensus')
    if consensus:
        sb = intel.get('analyst_strong_buy', 0)
        b = intel.get('analyst_buy', 0)
        h = intel.get('analyst_hold', 0)
        s = intel.get('analyst_sell', 0)
        ss = intel.get('analyst_strong_sell', 0)
        total = intel.get('analyst_count', 0)
        lines.append(f"  ANALYST CONSENSUS: {consensus} ({total} analysts)")
        lines.append(f"    Strong Buy: {sb} | Buy: {b} | Hold: {h} | Sell: {s} | Strong Sell: {ss}")

    # Price targets
    target = intel.get('target_mean')
    if target:
        high = intel.get('target_high')
        low = intel.get('target_low')
        upside = intel.get('target_upside_pct')
        lines.append(f"  PRICE TARGETS: Mean ${target:.2f} | High ${high:.2f} | Low ${low:.2f}"
                     f" | Upside: {upside:+.1f}%" if upside else "")

    # Recent upgrades/downgrades
    changes = intel.get('recent_changes', [])
    if changes:
        lines.append(f"  RECENT RATING CHANGES ({len(changes)}):")
        for c in changes[:5]:
            from_g = f" from {c['from_grade']}" if c.get('from_grade') else ""
            lines.append(f"    {c.get('date', '?')} — {c.get('firm', '?')}: {c.get('action', '?')} → {c.get('to_grade', '?')}{from_g}")

    # Insider activity — ALWAYS show status (prevents AI hallucinating "insider selling" when no data)
    buys = intel.get('insider_buys_90d', 0)
    sells = intel.get('insider_sells_90d', 0)
    if buys > 0 or sells > 0:
        net = intel.get('insider_net_shares', 0)
        signal_str = "NET BUYING ✅" if net > 0 else ("NET SELLING ⚠️" if net < 0 else "NEUTRAL")
        lines.append(f"  INSIDER TRANSACTIONS (90 days): {buys} buys, {sells} sells — {signal_str}")
        # Show top transactions
        txns = intel.get('insider_transactions', [])
        for t in txns[:3]:
            val_str = f" (${t['value']:,.0f})" if t.get('value') else ""
            lines.append(f"    {t.get('date', '?')} — {t.get('name', '?')}: {t.get('type', '?')} "
                         f"{t.get('shares', 0):,} shares{val_str}")
    else:
        lines.append("  INSIDER TRANSACTIONS (90 days): None found — no insider buying or selling detected")

    # Social sentiment — always show status
    social = intel.get('social_score')
    if social:
        reddit = intel.get('social_reddit_mentions', 0)
        twitter = intel.get('social_twitter_mentions', 0)
        source = intel.get('social_source', '')
        if source == 'volume_proxy':
            vol_surge = intel.get('volume_surge_ratio', 1.0)
            lines.append(f"  SOCIAL SENTIMENT: {social} (volume proxy: {vol_surge:.1f}x avg — Finnhub social requires premium)")
        else:
            lines.append(f"  SOCIAL SENTIMENT: {social} (Reddit: {reddit} mentions | Twitter: {twitter} mentions, last 7 days)")
    else:
        error = intel.get('social_error', '')
        lines.append(f"  SOCIAL SENTIMENT: Not available ({error or 'no data source'})")

    if not lines:
        return "No market intelligence available."

    return "\n".join(lines)


def build_ai_prompt(ticker: str,
                    signal: EntrySignal,
                    recommendation: Dict,
                    quality: Dict,
                    fundamentals: Dict = None,
                    fundamental_profile: Dict = None,
                    tradingview_data: Dict = None,
                    news_data: Dict = None,
                    market_intel: Dict = None) -> str:
    """
    Build enhanced AI analysis prompt with fundamental profile,
    TradingView confirmation, news, and actionable breakout guidance.
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
TRADINGVIEW TECHNICAL SUMMARY (independent confirmation)
══════════════════════════════════════════════════
{_format_tradingview(tradingview_data)}

══════════════════════════════════════════════════
RECENT NEWS & CATALYSTS
══════════════════════════════════════════════════
{_format_news(news_data)}

══════════════════════════════════════════════════
MARKET INTELLIGENCE — Analysts, Insiders, Social
══════════════════════════════════════════════════
{_format_market_intelligence(market_intel) if market_intel else 'No market intelligence available.'}

══════════════════════════════════════════════════
MARKET SENTIMENT DATA
══════════════════════════════════════════════════
{_format_fundamentals(fundamentals)}

══════════════════════════════════════════════════
YOUR ANALYSIS — Respond with EXACTLY this format:
══════════════════════════════════════════════════

CONVICTION: [1-10, weigh: technical alignment (30%), analyst consensus (20%), fundamental quality (20%), insider activity (15%), risk factors (15%). Be specific about what raised or lowered conviction.]

ACTION: [BUY NOW / WAIT FOR BREAKOUT / WAIT FOR PULLBACK / SKIP]

RESISTANCE VERDICT: [Is overhead resistance a problem? Should trader wait for breakout? Specify the exact price level that must break, what volume confirms it (e.g. "2x avg"), and what a failed breakout looks like. If no significant resistance, say "Clear path — enter on signal." 2-3 sentences max.]

WHY IT'S MOVING: [Synthesize from news headlines, analyst actions, insider activity, and social buzz. What is the NARRATIVE driving this stock right now? Recent earnings beat/miss? Analyst upgrades? Insider buying? Reddit buzz? Sector rotation? If nothing, say "Technical momentum — no catalyst visible." 2-3 sentences.]

FUNDAMENTAL QUALITY: [Rate the business A/B/C/D using this framework:
A = Strong moat, growing revenue, profitable, clean balance sheet
B = Decent business, some growth, acceptable margins
C = Weak fundamentals, declining margins, or high debt
D = Speculative, no profits, or deteriorating rapidly
Include 1 sentence explaining the rating with specific numbers.]

SMART MONEY: [What are analysts and insiders telling us? Summarize: analyst consensus rating, mean price target vs current (upside %), any recent upgrades/downgrades, and insider buying/selling pattern. CRITICAL: Only report insider SELLING or BUYING if there are actual insider TRANSACTIONS shown in the data. Insider OWNERSHIP % is NOT selling — it is a static stake. If no insider transactions found, say "No recent insider transactions." Is smart money aligned with the technical signal? 2-3 sentences.]

BULL CASE: [Best realistic outcome in 3-6 months, with a price target and the catalyst that gets it there. Use analyst targets if available. 1-2 sentences.]

BEAR CASE: [What kills this trade? Specific risk with price level. 1-2 sentences.]

RED FLAGS: [1-3 specific concerns, or "None" if clean. CRITICAL: Do NOT flag "insider selling" unless there are actual insider SELL TRANSACTIONS in the data. Insider ownership % is a static stake, NOT evidence of selling. Consider ONLY actual evidence: insider sell transactions, analyst downgrades, high short interest, earnings risk, declining fundamentals.]

POSITION SIZING: [Full (100%) / Reduced (75%) / Small (50%) / Skip — with 1 reason]

Keep total response under 300 words. No fluff. Be specific with prices and percentages."""

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

    # Try Groq/OpenAI first (primary — free, fast, generous limits)
    if openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system",
                     "content": "You are a senior technical analyst. Be concise, honest, actionable."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.5
            )
            narrative = response.choices[0].message.content
            result['provider'] = 'groq'
        except Exception as e:
            result['groq_error'] = str(e)[:200]

    # Try Gemini fallback
    if narrative is None and gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            narrative = response.text
            result['provider'] = 'gemini'
        except Exception as e:
            result['gemini_error'] = str(e)[:200]

    if narrative:
        result['raw_text'] = narrative
        result['success'] = True
        result.update(_parse_ai_response(narrative))
    else:
        result['error'] = 'All AI providers failed'

    return result


def _parse_ai_response(text: str) -> Dict[str, Any]:
    """Parse structured fields from AI response text."""
    parsed = {
        'conviction': 0,
        'action': '',
        'resistance_verdict': '',
        'why_moving': '',
        'fundamental_quality': '',
        'smart_money': '',
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
        'SMART MONEY': 'smart_money',
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
            tradingview_data: Dict = None,
            news_data: Dict = None,
            market_intel: Dict = None,
            gemini_model=None,
            openai_client=None) -> Dict[str, Any]:
    """
    Run enhanced AI analysis on a ticker.
    """
    # Build prompt
    prompt = build_ai_prompt(ticker, signal, recommendation, quality,
                             fundamentals, fundamental_profile,
                             tradingview_data, news_data, market_intel)

    # Try AI first
    if gemini_model is not None or openai_client is not None:
        result = call_ai(prompt, gemini_model=gemini_model, openai_client=openai_client)

        # If AI failed, use system fallback but preserve error info
        if not result['success']:
            ai_errors = {
                'groq_error': result.get('groq_error'),
                'gemini_error': result.get('gemini_error'),
                'openai_error': result.get('openai_error'),
                'error': result.get('error'),
            }
            result = generate_system_analysis(ticker, signal, recommendation,
                                              quality, fundamental_profile)
            result['note'] = 'AI unavailable — using system analysis'
            result.update({k: v for k, v in ai_errors.items() if v})
    else:
        # No AI configured — use system analysis
        result = generate_system_analysis(ticker, signal, recommendation,
                                          quality, fundamental_profile)
        result['note'] = 'No AI configured — using system analysis'

    result['ticker'] = ticker
    result['timestamp'] = datetime.now().isoformat()
    # Attach extra data for UI display
    result['fundamental_profile'] = fundamental_profile
    result['tradingview_data'] = tradingview_data
    result['news_data'] = news_data
    result['market_intel'] = market_intel

    return result


# =============================================================================
# MORNING BRIEFING — Market Narrative Generator
# =============================================================================

def generate_market_narrative(macro_data: Dict,
                              gemini_model=None,
                              openai_client=None) -> Dict[str, Any]:
    """
    Generate a concise market narrative from macro data.

    Returns: {narrative: str, regime: str, provider: str, success: bool}
    """
    result = {
        'narrative': '',
        'regime': '',
        'provider': None,
        'success': False,
        'error': None,
    }

    # Build prompt from macro data
    prompt = _build_narrative_prompt(macro_data)

    # Call AI
    narrative_text = None

    if openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system",
                     "content": "You are a senior macro strategist giving a morning briefing to a swing trader."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            narrative_text = response.choices[0].message.content
            result['provider'] = 'groq'
        except Exception as e:
            result['groq_error'] = str(e)[:200]

    if narrative_text is None and gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            narrative_text = response.text
            result['provider'] = 'gemini'
        except Exception as e:
            result['gemini_error'] = str(e)[:200]

    if narrative_text:
        result['narrative'] = narrative_text.strip()
        result['success'] = True

        # Extract regime from text
        for regime in ['Risk-On', 'Risk-Off', 'Caution', 'Rotation to Safety',
                       'Balanced', 'Bullish', 'Bearish', 'Neutral']:
            if regime.lower() in narrative_text.lower():
                result['regime'] = regime
                break
        if not result['regime']:
            result['regime'] = 'Neutral'
    else:
        result['narrative'] = _generate_system_narrative(macro_data)
        result['regime'] = _infer_regime(macro_data)
        result['provider'] = 'system'
        result['success'] = True

    return result


def _build_narrative_prompt(data: Dict) -> str:
    """Build the prompt for narrative generation."""
    parts = ["DAILY MARKET DATA (generate a morning briefing):\n"]

    # Indices
    indices = data.get('indices', {})
    if indices:
        parts.append("MAJOR INDICES (20-day performance):")
        for name, info in indices.items():
            price = info.get('price', '?')
            d1 = info.get('1d', '?')
            d5 = info.get('5d', '?')
            d20 = info.get('20d', '?')
            parts.append(f"  {name}: ${price} | 1d: {d1:+.1f}% | 5d: {d5:+.1f}% | 20d: {d20:+.1f}%")

    # Breadth
    breadth = data.get('breadth', {})
    if breadth:
        parts.append(f"\nBREADTH: RSP 20d: {breadth.get('rsp_20d', '?')}% vs SPY 20d: {breadth.get('spy_20d', '?')}%")
        parts.append(f"  Spread: {breadth.get('spread', '?')}% → {breadth.get('regime', '?')}")

    # VIX
    vix = data.get('vix', {})
    if vix:
        parts.append(f"\nVOLATILITY: VIX {vix.get('level', '?')} (5d change: {vix.get('change_5d', '?'):+.1f})")
        parts.append(f"  Regime: {vix.get('regime', '?')}")

    # Sectors
    sectors = data.get('sectors', {})
    if sectors:
        parts.append(f"\nSECTOR ROTATION: Offensive avg 20d: {sectors.get('offensive_avg_20d', '?')}% | "
                     f"Defensive avg 20d: {sectors.get('defensive_avg_20d', '?')}%")
        parts.append(f"  Spread: {sectors.get('spread', '?')}% → {sectors.get('regime', '?')}")

    # Macro
    macro = data.get('macro', {})
    if macro:
        parts.append("\nMACRO:")
        for name, info in macro.items():
            price = info.get('price', '?')
            d20 = info.get('20d', '?')
            parts.append(f"  {name}: ${price} (20d: {d20:+.1f}%)")

    parts.append("""
TASK: Write a concise 100-word morning briefing paragraph for a swing trader.
- Describe the current market regime (Risk-On, Risk-Off, Caution, Rotation, etc.)
- Highlight which sectors are leading/lagging
- Note any divergences (e.g. narrow breadth, VIX rising while market up)
- End with ONE sentence starting with "Net Read:" summarizing the trading bias
- Be direct and actionable, not academic
""")

    return "\n".join(parts)


def _generate_system_narrative(data: Dict) -> str:
    """Fallback narrative when AI is unavailable."""
    parts = []

    indices = data.get('indices', {})
    spy = indices.get('S&P 500', {})
    if spy:
        direction = "higher" if spy.get('20d', 0) > 0 else "lower"
        parts.append(f"S&P 500 trending {direction} over 20 days ({spy.get('20d', 0):+.1f}%).")

    vix = data.get('vix', {})
    if vix:
        parts.append(f"VIX at {vix.get('level', '?')} ({vix.get('regime', 'unknown')}).")

    sectors = data.get('sectors', {})
    if sectors:
        parts.append(f"Sector rotation: {sectors.get('regime', 'Balanced')}.")

    breadth = data.get('breadth', {})
    if breadth:
        parts.append(f"Breadth: {breadth.get('regime', 'Neutral')}.")

    regime = _infer_regime(data)
    parts.append(f"Net Read: Market posture is {regime} — size positions accordingly.")

    return " ".join(parts)


def _infer_regime(data: Dict) -> str:
    """Infer market regime from data without AI."""
    score = 0

    indices = data.get('indices', {})
    spy = indices.get('S&P 500', {})
    if spy.get('20d', 0) > 3:
        score += 2
    elif spy.get('20d', 0) > 0:
        score += 1
    elif spy.get('20d', 0) < -3:
        score -= 2
    else:
        score -= 1

    vix = data.get('vix', {})
    if vix.get('level', 20) < 15:
        score += 1
    elif vix.get('level', 20) > 25:
        score -= 2

    sectors = data.get('sectors', {})
    if sectors.get('regime') == 'Risk-On':
        score += 1
    elif sectors.get('regime') == 'Risk-Off':
        score -= 1

    if score >= 3:
        return 'Risk-On'
    elif score >= 1:
        return 'Cautiously Bullish'
    elif score <= -3:
        return 'Risk-Off'
    elif score <= -1:
        return 'Caution'
    return 'Neutral'


# =============================================================================
# DEEP MARKET STRUCTURE ANALYSIS — "Juan's Market Filter" style
# =============================================================================

def generate_deep_market_analysis(macro_data: Dict,
                                   market_filter: Dict = None,
                                   sector_rotation: Dict = None,
                                   gemini_model=None,
                                   openai_client=None) -> Dict[str, Any]:
    """
    Generate a deep, interpretive market structure analysis centered on
    sector ETF rotation — which sectors are LEADING, EMERGING, FADING, LAGGING,
    and the narrative WHY money is flowing where it is.

    Returns: {analysis: str, score: int, score_label: str, factors: dict,
              sectors_by_phase: dict, provider: str, success: bool}
    """
    result = {
        'analysis': '',
        'score': 0,
        'score_label': '',
        'factors': {},
        'sectors_by_phase': {},
        'provider': None,
        'success': False,
        'error': None,
    }

    # Classify sectors by momentum phase
    phases = {'LEADING': [], 'EMERGING': [], 'FADING': [], 'LAGGING': []}
    if sector_rotation:
        seen_etfs = set()
        for sector, info in sector_rotation.items():
            etf = info.get('etf', '')
            if etf in seen_etfs:
                continue  # Skip duplicate ETFs (Financials / Financial Services)
            seen_etfs.add(etf)
            phase = info.get('phase', 'LAGGING')
            phases[phase].append({
                'sector': sector,
                'etf': etf,
                'short': info.get('short_name', sector[:4]),
                'perf_1d': info.get('perf_1d', 0),
                'perf_5d': info.get('perf_5d', 0),
                'perf_20d': info.get('perf_20d', 0),
                'vs_spy_20d': info.get('vs_spy_20d', 0),
                'vs_spy_5d': info.get('vs_spy_5d', 0),
            })
    result['sectors_by_phase'] = phases

    prompt = _build_deep_analysis_prompt(macro_data, market_filter, sector_rotation, phases)

    raw_text = None

    # Try Groq first
    if openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system",
                     "content": (
                         "You are an elite institutional macro strategist and sector rotation analyst. "
                         "Your specialty is reading where institutional money is flowing by analyzing "
                         "sector ETF performance, then explaining WHY money is moving there and what "
                         "stocks/sectors a swing trader should target. You think like Juan Maldonado's "
                         "5-Factor Institutional Flow model — scoring the market environment and giving "
                         "a clear directional bias. Be direct, opinionated, and actionable. No hedging."
                     )},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6
            )
            raw_text = response.choices[0].message.content
            result['provider'] = 'groq'
        except Exception as e:
            result['groq_error'] = str(e)[:200]

    # Gemini fallback
    if raw_text is None and gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            raw_text = response.text
            result['provider'] = 'gemini'
        except Exception:
            pass

    if raw_text:
        result['analysis'] = raw_text.strip()
        result['success'] = True
        parsed = _parse_deep_analysis(raw_text)
        result.update(parsed)
        result['sectors_by_phase'] = phases  # Keep phases even after update
    else:
        # System fallback
        result.update(_generate_system_deep_analysis(macro_data, market_filter, phases))
        result['provider'] = 'system'
        result['success'] = True

    return result


def _build_deep_analysis_prompt(macro_data: Dict,
                                 market_filter: Dict = None,
                                 sector_rotation: Dict = None,
                                 phases: Dict = None) -> str:
    """Build sector-rotation-centric deep analysis prompt."""

    parts = ["""MARKET STRUCTURE & SECTOR ROTATION ANALYSIS

You are scoring the market using Juan Maldonado's 5-Factor Institutional Flow model.
The CORE of your analysis is SECTOR ETF ROTATION — where is institutional money flowing
and WHY? What does the rotation pattern TELL US about the market regime?

═══════════════════════════════════════════════
SECTOR ETF ROTATION DATA (sorted by momentum phase)
═══════════════════════════════════════════════
"""]

    def _fmt_sector(s):
        return (f"  {s['etf']} ({s['sector']}): "
                f"1d: {s['perf_1d']:+.1f}% | 5d: {s['perf_5d']:+.1f}% | 20d: {s['perf_20d']:+.1f}% | "
                f"vs SPY 5d: {s['vs_spy_5d']:+.1f}% | vs SPY 20d: {s['vs_spy_20d']:+.1f}%")

    if phases:
        leading = phases.get('LEADING', [])
        emerging = phases.get('EMERGING', [])
        fading = phases.get('FADING', [])
        lagging = phases.get('LAGGING', [])

        parts.append("🟢 LEADING (outperforming SPY on both 5d + 20d — dominant sectors, trade these):")
        if leading:
            for s in sorted(leading, key=lambda x: x['vs_spy_20d'], reverse=True):
                parts.append(_fmt_sector(s))
        else:
            parts.append("  (none — no sector clearly dominant)")

        parts.append("\n🔵 EMERGING (5d accelerating but 20d still catching up — money rotating IN, watch for entries):")
        if emerging:
            for s in sorted(emerging, key=lambda x: x['vs_spy_5d'], reverse=True):
                parts.append(_fmt_sector(s))
        else:
            parts.append("  (none)")

        parts.append("\n🟡 FADING (20d still positive but 5d losing momentum — money rotating OUT, tighten stops):")
        if fading:
            for s in sorted(fading, key=lambda x: x['vs_spy_5d']):
                parts.append(_fmt_sector(s))
        else:
            parts.append("  (none)")

        parts.append("\n🔴 LAGGING (underperforming SPY on both timeframes — avoid, no institutional interest):")
        if lagging:
            for s in sorted(lagging, key=lambda x: x['vs_spy_20d']):
                parts.append(_fmt_sector(s))
        else:
            parts.append("  (none)")

    parts.append("\n═══════════════════════════════════════════════")
    parts.append("5-FACTOR MACRO CONTEXT")
    parts.append("═══════════════════════════════════════════════\n")

    # Factor 1: S&P 500 Trend
    indices = macro_data.get('indices', {})
    spy = indices.get('S&P 500', {})
    if spy:
        parts.append(f"FACTOR 1 — S&P 500 TREND:")
        parts.append(f"  ${spy.get('price', '?')} | 1d: {spy.get('1d', 0):+.1f}% | 5d: {spy.get('5d', 0):+.1f}% | 20d: {spy.get('20d', 0):+.1f}%")
    if market_filter:
        above200 = market_filter.get('spy_above_200', True)
        parts.append(f"  200 SMA: ${market_filter.get('spy_sma200', '?')} — Price {'ABOVE' if above200 else 'BELOW'}")

    # Factor 2: VIX
    vix = macro_data.get('vix', {})
    if vix:
        parts.append(f"\nFACTOR 2 — VIX / VOLATILITY / COMMERCIALS:")
        parts.append(f"  VIX: {vix.get('level', '?')} (5d change: {vix.get('change_5d', 0):+.1f}) — {vix.get('regime', '?')}")

    # Factor 3: US Dollar
    macro = macro_data.get('macro', {})
    dollar = macro.get('Dollar', {})
    if dollar:
        parts.append(f"\nFACTOR 3 — US DOLLAR:")
        parts.append(f"  UUP: ${dollar.get('price', '?')} | 20d: {dollar.get('20d', 0):+.1f}%")
        parts.append(f"  (Strong dollar = risk-off headwind for equities & commodities)")

    # Factor 4: Cost of Money
    bonds = macro.get('20Y Bond', {})
    if bonds:
        parts.append(f"\nFACTOR 4 — COST OF MONEY (bonds/yields):")
        parts.append(f"  TLT: ${bonds.get('price', '?')} | 20d: {bonds.get('20d', 0):+.1f}%")
        parts.append(f"  (TLT falling = yields rising = headwind for growth/duration)")

    # Factor 5: Breadth
    breadth = macro_data.get('breadth', {})
    if breadth:
        parts.append(f"\nFACTOR 5 — MARKET BREADTH:")
        parts.append(f"  RSP vs SPY spread: {breadth.get('spread', 0):+.1f}% → {breadth.get('regime', '?')}")
        parts.append(f"  (Positive = broad participation, negative = narrow mega-cap leadership)")

    # Additional context
    qqq = indices.get('Nasdaq 100', {})
    iwm = indices.get('Russell 2000', {})
    if qqq or iwm:
        parts.append(f"\nINDEX DIVERGENCES:")
    if qqq:
        parts.append(f"  QQQ: ${qqq.get('price', '?')} (5d: {qqq.get('5d', 0):+.1f}%, 20d: {qqq.get('20d', 0):+.1f}%)")
    if iwm:
        parts.append(f"  IWM: ${iwm.get('price', '?')} (5d: {iwm.get('5d', 0):+.1f}%, 20d: {iwm.get('20d', 0):+.1f}%)")
        if spy:
            spread = (iwm.get('20d', 0) or 0) - (spy.get('20d', 0) or 0)
            parts.append(f"  Small vs Large cap 20d spread: {spread:+.1f}%")

    parts.append("""

═══════════════════════════════════════════════
YOUR ANALYSIS — provide in EXACTLY this format:
═══════════════════════════════════════════════

SCORE: [integer -5 to +5]
LABEL: [e.g. "BEARISH ENVIRONMENT" or "BULLISH — OFFENSIVE ROTATION" or "CAUTIOUS — DEFENSIVE SHIFT"]

FACTOR SCORES:
S&P 500: [🟢 or 🟡 or 🔴] [one-line, e.g. "Above 200MA, momentum positive" — use 🟡 if above 200MA but momentum weak]
VIX/Commercials: [🟢 or 🟡 or 🔴] [one-line — 🟢 below 15, 🟡 15-20, 🔴 above 20]
US Dollar: [🟢 or 🟡 or 🔴] [one-line — 🟢 weakening >1%, 🟡 flat, 🔴 strengthening >1%]
Cost of Money: [🟢 or 🟡 or 🔴] [one-line — 🟢 yields falling >1.5%, 🟡 flat, 🔴 yields rising >1.5%]
Rotation/Breadth: [🟢 or 🟡 or 🔴] [one-line — 🟢 broad leadership, 🟡 mixed, 🔴 narrow/defensive]

SECTOR ROTATION NARRATIVE:
[3-4 sentences: This is the MOST IMPORTANT section. Explain WHERE institutional money is flowing
based on the ETF data above. WHY are the leading sectors leading? What macro/fundamental story
explains the rotation? Are emerging sectors signaling a regime change? What does the fading of
certain sectors tell us? Connect the dots between sector flows and the broader macro picture.
Name specific ETFs (XLK, XLE, etc.) and their sectors.]

WHAT TO TRADE:
[2-3 sentences: Based on the rotation analysis, which SPECIFIC sectors should a swing trader
focus on RIGHT NOW? Name the ETFs. Why those sectors? What type of stocks within those sectors
(growth, value, mega-cap, mid-cap)? What setups to look for?]

WHAT TO AVOID:
[1-2 sentences: Which sectors to stay away from and why. Which fading/lagging sectors are traps?]
""")

    return "\n".join(parts)


def _parse_deep_analysis(text: str) -> Dict:
    """Parse structured fields from deep analysis response."""
    result = {
        'score': 0,
        'score_label': '',
        'factors': {},
    }

    lines = text.strip().split('\n')

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Parse SCORE
        if line_stripped.upper().startswith('SCORE:'):
            try:
                score_text = line_stripped.split(':', 1)[1].strip()
                score_num = score_text.split('/')[0].strip().replace('+', '')
                result['score'] = max(-5, min(5, int(score_num)))
            except Exception:
                pass

        # Parse LABEL
        elif line_stripped.upper().startswith('LABEL:'):
            result['score_label'] = line_stripped.split(':', 1)[1].strip().strip('"\'')

        # Parse factor scores — match any line with factor name + any signal emoji
        elif 'S&P 500' in line_stripped and any(e in line_stripped for e in ['🟢', '🔴', '🟡', '🟠', '+']):
            result['factors']['sp500'] = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped
        elif 'VIX' in line_stripped and any(e in line_stripped for e in ['🟢', '🔴', '🟡', '🟠', '+']):
            result['factors']['vix'] = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped
        elif 'Dollar' in line_stripped and any(e in line_stripped for e in ['🟢', '🔴', '🟡', '🟠', '+']):
            result['factors']['dollar'] = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped
        elif 'Cost' in line_stripped and 'Money' in line_stripped and any(e in line_stripped for e in ['🟢', '🔴', '🟡', '🟠', '+']):
            result['factors']['cost_of_money'] = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped
        elif ('Rotation' in line_stripped or 'Breadth' in line_stripped) and any(e in line_stripped for e in ['🟢', '🔴', '🟡', '🟠', '+']):
            result['factors']['rotation'] = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped

    return result


def _generate_system_deep_analysis(macro_data: Dict,
                                    market_filter: Dict = None,
                                    phases: Dict = None) -> Dict:
    """Fallback deep analysis when AI is unavailable — with nuanced scoring."""
    score = 0
    factors = {}

    # Factor 1: S&P — above 200MA AND positive momentum = green, above 200 but weak = neutral
    indices = macro_data.get('indices', {})
    spy = indices.get('S&P 500', {})
    spy_above = market_filter.get('spy_above_200', True) if market_filter else True
    spy_20d = spy.get('20d', 0) or 0
    spy_5d = spy.get('5d', 0) or 0

    if spy_above and spy_20d > 1.0:
        factors['sp500'] = f"🟢 Above 200MA, strong momentum (20d: {spy_20d:+.1f}%)"
        score += 1
    elif spy_above and spy_20d > -1.0:
        factors['sp500'] = f"🟡 Above 200MA but momentum flat (20d: {spy_20d:+.1f}%, 5d: {spy_5d:+.1f}%)"
        # Neutral — no score change
    elif not spy_above:
        factors['sp500'] = f"🔴 Below 200MA ({spy_20d:+.1f}%) — structural damage"
        score -= 1
    else:
        factors['sp500'] = f"🔴 Negative momentum (20d: {spy_20d:+.1f}%)"
        score -= 1

    # Factor 2: VIX — levels matter more than just above/below 20
    vix = macro_data.get('vix', {})
    vix_level = vix.get('level', 20) or 20
    vix_change = vix.get('change_5d', 0) or 0

    if vix_level < 15:
        factors['vix'] = f"🟢 Low fear (VIX={vix_level:.1f}) — complacency zone"
        score += 1
    elif vix_level < 20:
        if vix_change > 2:
            factors['vix'] = f"🟡 Moderate but rising (VIX={vix_level:.1f}, +{vix_change:.1f} 5d)"
        else:
            factors['vix'] = f"🟡 Moderate (VIX={vix_level:.1f}) — normal range"
        # Neutral — no score change
    elif vix_level < 25:
        factors['vix'] = f"🟠 Elevated (VIX={vix_level:.1f}) — caution"
        score -= 1
    else:
        factors['vix'] = f"🔴 High fear (VIX={vix_level:.1f}) — risk-off"
        score -= 1

    # Factor 3: Dollar — needs meaningful move, not noise
    macro = macro_data.get('macro', {})
    dollar = macro.get('Dollar', {})
    dollar_20d = dollar.get('20d', 0) or 0

    if dollar_20d < -1.0:
        factors['dollar'] = f"🟢 Weakening ({dollar_20d:+.1f}%) — risk-on tailwind"
        score += 1
    elif dollar_20d > 1.0:
        factors['dollar'] = f"🔴 Strengthening ({dollar_20d:+.1f}%) — risk-off headwind"
        score -= 1
    else:
        factors['dollar'] = f"🟡 Flat ({dollar_20d:+.1f}%) — no strong signal"
        # Neutral — no score change

    # Factor 4: Cost of money — TLT direction matters
    bonds = macro.get('20Y Bond', {})
    bonds_20d = bonds.get('20d', 0) or 0

    if bonds_20d > 1.5:
        factors['cost_of_money'] = f"🟢 Yields falling (TLT {bonds_20d:+.1f}%) — growth tailwind"
        score += 1
    elif bonds_20d < -1.5:
        factors['cost_of_money'] = f"🔴 Yields rising (TLT {bonds_20d:+.1f}%) — growth headwind"
        score -= 1
    else:
        factors['cost_of_money'] = f"🟡 Yields flat (TLT {bonds_20d:+.1f}%) — no clear direction"
        # Neutral — no score change

    # Factor 5: Rotation/Breadth — sector leadership matters
    if phases:
        n_leading = len(phases.get('LEADING', []))
        n_emerging = len(phases.get('EMERGING', []))
        n_fading = len(phases.get('FADING', []))
        n_lagging = len(phases.get('LAGGING', []))

        if n_leading >= 4:
            factors['rotation'] = f"🟢 {n_leading} sectors leading — broad offensive rotation"
            score += 1
        elif n_leading >= 2 and n_emerging >= 2:
            factors['rotation'] = f"🟢 {n_leading} leading + {n_emerging} emerging — healthy rotation"
            score += 1
        elif n_lagging >= 5:
            factors['rotation'] = f"🔴 {n_lagging} sectors lagging — narrow/defensive market"
            score -= 1
        elif n_fading >= 3 and n_leading <= 1:
            factors['rotation'] = f"🟠 {n_fading} fading, only {n_leading} leading — rotation deteriorating"
            score -= 1
        else:
            factors['rotation'] = f"🟡 Mixed ({n_leading}L / {n_emerging}E / {n_fading}F / {n_lagging}X)"
            # Neutral — no score change
    else:
        sectors = macro_data.get('sectors', {})
        breadth = macro_data.get('breadth', {})
        spread = breadth.get('spread', 0) or 0
        if spread > 1.0:
            factors['rotation'] = f"🟢 Broad participation (RSP-SPY: {spread:+.1f}%)"
            score += 1
        elif spread < -2.0:
            factors['rotation'] = f"🔴 Narrow leadership (RSP-SPY: {spread:+.1f}%)"
            score -= 1
        else:
            factors['rotation'] = f"🟡 Neutral breadth (RSP-SPY: {spread:+.1f}%)"

    # Score label — maps to 11-point scale (-5 to +5)
    if score >= 4:
        label = "STRONGLY BULLISH"
    elif score >= 2:
        label = "BULLISH ENVIRONMENT"
    elif score >= 1:
        label = "CAUTIOUSLY BULLISH"
    elif score <= -4:
        label = "STRONGLY BEARISH"
    elif score <= -2:
        label = "BEARISH ENVIRONMENT"
    elif score <= -1:
        label = "CAUTION — HEADWINDS"
    else:
        label = "MIXED / NEUTRAL"

    # Build sector narrative from phases
    narrative_parts = [f"{score}/5 {label}."]
    if phases:
        leading = phases.get('LEADING', [])
        emerging = phases.get('EMERGING', [])
        fading = phases.get('FADING', [])
        if leading:
            names = ', '.join(f"{s['etf']} ({s['short']})" for s in leading)
            narrative_parts.append(f"LEADING sectors: {names}.")
        if emerging:
            names = ', '.join(f"{s['etf']} ({s['short']})" for s in emerging)
            narrative_parts.append(f"EMERGING (rotation in): {names}.")
        if fading:
            names = ', '.join(f"{s['etf']} ({s['short']})" for s in fading)
            narrative_parts.append(f"FADING (rotation out): {names}.")
        narrative_parts.append("AI unavailable for deep interpretation — refresh when online.")

    return {
        'analysis': ' '.join(narrative_parts),
        'score': score,
        'score_label': label,
        'factors': factors,
    }
