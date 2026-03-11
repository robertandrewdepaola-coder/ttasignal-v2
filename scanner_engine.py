"""
TTA v2 Scanner Engine — Analysis & Recommendations
====================================================

Orchestrates signal_engine + data_fetcher to produce actionable results.
Contains: quality scoring (mini-backtest), signal classification
(primary/AO confirm/re-entry/late entry), and recommendation logic.

NO yfinance calls. NO Streamlit code. Pure analysis logic.

Version: 2.0.0 (2026-02-07)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from signal_engine import (
    ENTRY_WINDOW, MACD_CROSS_LOOKBACK,
    AO_CONFIRM_MACD_LOOKBACK, AO_CONFIRM_MAX_PREMIUM,
    LATE_ENTRY_MAX_DAYS, LATE_ENTRY_MAX_PREMIUM,
    normalize_columns, calculate_macd, calculate_ao,
    detect_macd_cross, detect_ao_state,
    check_timeframe_macd, check_timeframe_ao, classify_macd_zone,
    validate_entry, EntrySignal,
    MACD_PROFILE_LEGACY, MACD_PROFILE_HPOTTER_ZONE, MACD_PROFILE_SHADOW,
    get_active_macd_profile,
)


# =============================================================================
# CONSTANTS
# =============================================================================

RE_ENTRY_MACD_LOOKBACK = 10  # Bars to look back for re-entry MACD cross


def _resolve_macd_profile(profile: Optional[str]) -> str:
    p = str(profile or get_active_macd_profile() or MACD_PROFILE_LEGACY).strip().lower()
    if p not in {MACD_PROFILE_LEGACY, MACD_PROFILE_HPOTTER_ZONE, MACD_PROFILE_SHADOW}:
        return MACD_PROFILE_LEGACY
    return p


# =============================================================================
# QUALITY SCORING — Mini-Backtest
# =============================================================================

def calculate_quality_score(daily_df: pd.DataFrame,
                            weekly_df: pd.DataFrame = None,
                            ticker: str = '',
                            lookback_years: int = 3,
                            macd_profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Run mini-backtest on historical data to calculate quality score.

    Simulates TTA strategy entry/exit on past signals:
    - Entry: MACD cross up + AO > 0 + AO zero-cross in prior window
    - Exit: 15% stop loss OR weekly MACD cross down OR 60-day hold

    Returns quality grade (A/B/C/F), win rate, avg return, signal count.
    """
    result = {
        'ticker': ticker,
        'quality_grade': 'N/A',
        'quality_score': 0,
        'signals_found': 0,
        'win_rate': 0,
        'avg_return': 0,
        'best_return': 0,
        'worst_return': 0,
        'weekly_confirmed_pct': 0,
        'error': None,
        'details': []
    }

    if daily_df is None or len(daily_df) < 100:
        result['error'] = "Insufficient daily data"
        return result

    profile = _resolve_macd_profile(macd_profile)
    profile_for_primary = MACD_PROFILE_LEGACY if profile == MACD_PROFILE_SHADOW else profile

    daily = normalize_columns(daily_df).copy()
    daily = calculate_macd(daily, profile=profile_for_primary)
    daily = calculate_ao(daily)

    # Weekly data for exit signals
    weekly = None
    if weekly_df is not None and len(weekly_df) >= 30:
        weekly = normalize_columns(weekly_df).copy()
        weekly = calculate_macd(weekly, profile=profile_for_primary)

    PROTECTIVE_STOP_PCT = -15.0
    signals = []

    for i in range(ENTRY_WINDOW + 30, len(daily)):
        # --- Check MACD crossover ---
        macd_now = daily['MACD'].iloc[i]
        sig_now = daily['MACD_Signal'].iloc[i]
        macd_prev = daily['MACD'].iloc[i - 1]
        sig_prev = daily['MACD_Signal'].iloc[i - 1]

        if pd.isna(macd_now) or pd.isna(sig_now) or pd.isna(macd_prev) or pd.isna(sig_prev):
            continue

        macd_cross = (macd_now > sig_now) and (macd_prev <= sig_prev)
        if not macd_cross:
            continue

        # --- Check AO positive ---
        ao_val = daily['AO'].iloc[i]
        if pd.isna(ao_val) or ao_val <= 0:
            continue

        # --- Check AO zero-cross in lookback ---
        ao_cross_found = False
        for j in range(1, ENTRY_WINDOW + 1):
            pi = i - j
            if pi < 1:
                break
            ao_before = daily['AO'].iloc[pi - 1]
            ao_after = daily['AO'].iloc[pi]
            if not (pd.isna(ao_before) or pd.isna(ao_after)):
                if ao_before <= 0 and ao_after > 0:
                    ao_cross_found = True
                    break

        if not ao_cross_found:
            continue

        # --- Entry signal found — simulate trade ---
        entry_price = float(daily['Close'].iloc[i])
        entry_date = daily.index[i]

        # Weekly confirmation at entry
        weekly_confirmed = False
        weekly_idx = None
        if weekly is not None:
            weekly_dates = weekly.index[weekly.index <= entry_date]
            if len(weekly_dates) > 0:
                weekly_idx = len(weekly_dates) - 1
                if weekly_idx >= 26:
                    w_m = weekly['MACD'].iloc[weekly_idx]
                    w_s = weekly['MACD_Signal'].iloc[weekly_idx]
                    if not (pd.isna(w_m) or pd.isna(w_s)):
                        weekly_confirmed = w_m > w_s

        # Simulate exit
        exit_price = entry_price
        exit_reason = "End_Data"

        for fd in range(i + 1, min(i + 60, len(daily))):
            future_price = float(daily['Close'].iloc[fd])
            current_return = ((future_price - entry_price) / entry_price) * 100

            # Stop loss
            if current_return <= PROTECTIVE_STOP_PCT:
                exit_price = future_price
                exit_reason = "Stop_Loss"
                break

            # Weekly MACD cross down
            if weekly is not None and weekly_idx is not None:
                future_date = daily.index[fd]
                fw_dates = weekly.index[weekly.index <= future_date]
                if len(fw_dates) > 0:
                    fw_idx = len(fw_dates) - 1
                    if fw_idx > weekly_idx and fw_idx >= 26:
                        cw_m = weekly['MACD'].iloc[fw_idx]
                        cw_s = weekly['MACD_Signal'].iloc[fw_idx]
                        pw_m = weekly['MACD'].iloc[fw_idx - 1]
                        pw_s = weekly['MACD_Signal'].iloc[fw_idx - 1]

                        if not any(pd.isna(x) for x in [cw_m, cw_s, pw_m, pw_s]):
                            if cw_m < cw_s and pw_m >= pw_s:
                                exit_price = future_price
                                exit_reason = "Weekly_Cross_Down"
                                break
        else:
            exit_price = float(daily['Close'].iloc[min(i + 59, len(daily) - 1)])

        return_pct = ((exit_price - entry_price) / entry_price) * 100

        signals.append({
            'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(return_pct, 1),
            'win': return_pct > 0,
            'weekly_confirmed': weekly_confirmed,
            'exit_reason': exit_reason,
        })

    # --- Calculate metrics ---
    if len(signals) == 0:
        result['error'] = "No signals found in backtest"
        return result

    df = pd.DataFrame(signals)

    result['signals_found'] = len(signals)
    result['win_rate'] = round((df['win'].sum() / len(df)) * 100, 1)
    result['avg_return'] = round(df['return_pct'].mean(), 1)
    result['best_return'] = round(df['return_pct'].max(), 1)
    result['worst_return'] = round(df['return_pct'].min(), 1)
    result['weekly_confirmed_pct'] = round((df['weekly_confirmed'].sum() / len(df)) * 100, 1)
    result['details'] = signals[-5:]

    # Compute expectancy: (win_rate × avg_win) - (loss_rate × avg_loss)
    wins = df[df['win']]
    losses = df[~df['win']]
    avg_win = float(wins['return_pct'].mean()) if len(wins) > 0 else 0.0
    avg_loss = abs(float(losses['return_pct'].mean())) if len(losses) > 0 else 0.0
    wr_frac = result['win_rate'] / 100.0
    expectancy = (wr_frac * avg_win) - ((1 - wr_frac) * avg_loss)
    result['avg_win'] = round(avg_win, 1)
    result['avg_loss'] = round(avg_loss, 1)
    result['expectancy'] = round(expectancy, 2)

    # Quality score (0-100) — expectancy-weighted
    # Expectancy is the single best predictor of whether a strategy makes money.
    # A 40% WR with +20% avg win / -5% avg loss = +5% expectancy (profitable)
    # An 80% WR with +1.5% avg win / -8% avg loss = -0.4% expectancy (unprofitable)
    expect_score = min(max(expectancy, 0), 10) / 10 * 30    # 30 pts max, caps at 10% expectancy
    wr_score = min(result['win_rate'], 80) / 80 * 25         # 25 pts max
    ret_score = min(max(result['avg_return'], 0), 15) / 15 * 20  # 20 pts max, caps at 15% avg
    sig_score = min(result['signals_found'], 20) / 20 * 15   # 15 pts max
    loss_score = max(0, 10 + result['worst_return'] / 3)     # 10 pts max

    result['quality_score'] = int(expect_score + wr_score + ret_score + sig_score + loss_score)

    if result['quality_score'] >= 60 and result['win_rate'] >= 55 and expectancy >= 2.0:
        result['quality_grade'] = 'A'
    elif result['quality_score'] >= 42 and result['win_rate'] >= 45 and expectancy >= 0.5:
        result['quality_grade'] = 'B'
    elif result['quality_score'] >= 25 and result['win_rate'] >= 35:
        result['quality_grade'] = 'C'
    else:
        result['quality_grade'] = 'F'

    return result


# =============================================================================
# SECONDARY SIGNAL TYPES
# =============================================================================

def check_ao_confirmation(daily_df: pd.DataFrame,
                          market_filter: Dict = None,
                          macd_lookback: int = AO_CONFIRM_MACD_LOOKBACK,
                          macd_profile: Optional[str] = None) -> Dict[str, Any]:
    """
    AO Confirmation signal: MACD crossed up first, AO confirms by crossing zero later.

    Conditions:
    1. MACD crossed up within last macd_lookback days
    2. AO just crossed from ≤0 to >0 (today or last 1-2 days)
    3. MACD still bullish
    4. Market filter passes
    """
    result = {
        'is_valid': False,
        'signal_type': 'AO_CONFIRMATION',
        'macd_cross_date': None,
        'macd_cross_price': 0,
        'macd_cross_days_ago': 0,
        'ao_at_macd_cross': 0,
        'ao_cross_date': None,
        'ao_cross_days_ago': 0,
        'current_price': 0,
        'entry_premium_pct': 0,
        'quality': '❌ No Signal',
        'quality_score': 0,
        'reason': '',
    }

    if daily_df is None or len(daily_df) < 50:
        result['reason'] = 'Insufficient data'
        return result

    profile = _resolve_macd_profile(macd_profile)
    profile_for_primary = MACD_PROFILE_LEGACY if profile == MACD_PROFILE_SHADOW else profile

    df = normalize_columns(daily_df).copy()
    df = calculate_macd(df, profile=profile_for_primary)
    df = calculate_ao(df)

    i = len(df) - 1
    current_price = float(df['Close'].iloc[i])
    result['current_price'] = round(current_price, 2)

    macd_now = float(df['MACD'].iloc[i])
    sig_now = float(df['MACD_Signal'].iloc[i])
    ao_now = float(df['AO'].iloc[i])

    if pd.isna(macd_now) or pd.isna(sig_now) or pd.isna(ao_now):
        result['reason'] = 'NaN in indicators'
        return result

    # Must be bullish and AO positive
    if macd_now <= sig_now:
        result['reason'] = 'MACD not bullish'
        return result

    if ao_now <= 0:
        result['reason'] = 'AO not positive'
        return result

    # Find AO zero-cross in last 3 days
    ao_cross_found = False
    for j in range(0, 3):
        ci = i - j
        if ci < 1:
            break
        ao_before = float(df['AO'].iloc[ci - 1])
        ao_after = float(df['AO'].iloc[ci])
        if not (pd.isna(ao_before) or pd.isna(ao_after)):
            if ao_before <= 0 and ao_after > 0:
                ao_cross_found = True
                result['ao_cross_days_ago'] = j
                idx = df.index[ci]
                result['ao_cross_date'] = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                break

    if not ao_cross_found:
        result['reason'] = 'No recent AO zero-cross (last 3 days)'
        return result

    # Find MACD cross in lookback period
    macd_cross_found = False
    for j in range(0, macd_lookback):
        ci = i - j
        if ci < 1:
            break
        cm = float(df['MACD'].iloc[ci])
        cs = float(df['MACD_Signal'].iloc[ci])
        pm = float(df['MACD'].iloc[ci - 1])
        ps = float(df['MACD_Signal'].iloc[ci - 1])

        if pd.isna(cm) or pd.isna(cs) or pd.isna(pm) or pd.isna(ps):
            continue

        if cm > cs and pm <= ps:
            macd_cross_found = True
            result['macd_cross_days_ago'] = j
            result['macd_cross_price'] = round(float(df['Close'].iloc[ci]), 2)
            result['ao_at_macd_cross'] = round(float(df['AO'].iloc[ci]), 4)
            idx = df.index[ci]
            result['macd_cross_date'] = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            break

    if not macd_cross_found:
        result['reason'] = f'No MACD cross in last {macd_lookback} days'
        return result

    # Entry premium check
    if result['macd_cross_price'] > 0:
        premium = (current_price - result['macd_cross_price']) / result['macd_cross_price'] * 100
        result['entry_premium_pct'] = round(premium, 1)
        if premium > AO_CONFIRM_MAX_PREMIUM:
            result['reason'] = f'Entry premium too high ({premium:.1f}% > {AO_CONFIRM_MAX_PREMIUM}%)'
            return result

    # Market filter — conservative: reject if missing or bearish
    _mf = market_filter or {}
    if not _mf.get('spy_above_200', False):
        result['reason'] = 'Market filter: SPY below 200 SMA (or data missing)'
        return result
    if not _mf.get('vix_below_30', False):
        result['reason'] = 'Market filter: VIX above 30 (or data missing)'
        return result

    # VALID
    result['is_valid'] = True

    # Quality rating
    days_ago = result['macd_cross_days_ago']
    premium = result['entry_premium_pct']

    if days_ago <= 3 and premium <= 2:
        result['quality'] = '🟢 Fresh AO Confirm'
        result['quality_score'] = 85
    elif days_ago <= 5 and premium <= 4:
        result['quality'] = '🟡 Recent AO Confirm'
        result['quality_score'] = 70
    else:
        result['quality'] = '🟠 Late AO Confirm'
        result['quality_score'] = 55

    return result


def check_reentry_signal(daily_df: pd.DataFrame,
                         market_filter: Dict = None,
                         macd_profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Re-entry signal: MACD crosses up while AO is already positive (established trend).

    This fills the gap where:
    - Primary fails (no fresh AO zero-cross)
    - AO Confirmation fails (AO was already positive at MACD cross)

    Conditions:
    1. MACD crossed up within last RE_ENTRY_MACD_LOOKBACK bars
    2. MACD still bullish
    3. AO positive
    4. AO has been positive for a while (no recent zero-cross = established momentum)
    5. Market filter passes
    """
    result = {
        'is_valid': False,
        'signal_type': 'RE_ENTRY',
        'macd_cross_date': None,
        'macd_cross_price': 0,
        'macd_cross_bars_ago': 0,
        'ao_value': 0,
        'current_price': 0,
        'reason': '',
        'quality': '',
        'quality_score': 0,
    }

    if daily_df is None or len(daily_df) < 50:
        result['reason'] = 'Insufficient data'
        return result

    profile = _resolve_macd_profile(macd_profile)
    profile_for_primary = MACD_PROFILE_LEGACY if profile == MACD_PROFILE_SHADOW else profile

    df = normalize_columns(daily_df).copy()
    df = calculate_macd(df, profile=profile_for_primary)
    df = calculate_ao(df)

    i = len(df) - 1
    current_price = float(df['Close'].iloc[i])
    result['current_price'] = round(current_price, 2)

    macd_now = float(df['MACD'].iloc[i])
    sig_now = float(df['MACD_Signal'].iloc[i])
    ao_now = float(df['AO'].iloc[i])

    if pd.isna(macd_now) or pd.isna(sig_now) or pd.isna(ao_now):
        result['reason'] = 'NaN in indicators'
        return result

    result['ao_value'] = round(ao_now, 2)

    # MACD must be bullish
    if macd_now <= sig_now:
        result['reason'] = 'MACD not bullish'
        return result

    # AO must be positive
    if ao_now <= 0:
        result['reason'] = 'AO not positive'
        return result

    # Find MACD cross in lookback
    macd_cross_found = False
    for j in range(RE_ENTRY_MACD_LOOKBACK):
        ci = i - j
        if ci < 1:
            break
        cm = float(df['MACD'].iloc[ci])
        cs = float(df['MACD_Signal'].iloc[ci])
        pm = float(df['MACD'].iloc[ci - 1])
        ps = float(df['MACD_Signal'].iloc[ci - 1])

        if pd.isna(cm) or pd.isna(cs) or pd.isna(pm) or pd.isna(ps):
            continue

        if cm > cs and pm <= ps:
            macd_cross_found = True
            result['macd_cross_bars_ago'] = j
            result['macd_cross_price'] = round(float(df['Close'].iloc[ci]), 2)
            idx = df.index[ci]
            result['macd_cross_date'] = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            break

    if not macd_cross_found:
        result['reason'] = f'No MACD cross in last {RE_ENTRY_MACD_LOOKBACK} bars'
        return result

    # AO must NOT have a recent zero-cross (that would be primary signal territory)
    for j in range(1, ENTRY_WINDOW + 1):
        pi = i - j
        if pi < 1:
            break
        ao_before = float(df['AO'].iloc[pi - 1])
        ao_after = float(df['AO'].iloc[pi])
        if not (pd.isna(ao_before) or pd.isna(ao_after)):
            if ao_before <= 0 and ao_after > 0:
                result['reason'] = 'AO had recent zero-cross — primary signal applies'
                return result

    # Market filter — conservative: reject if missing or bearish
    _mf = market_filter or {}
    if not _mf.get('spy_above_200', False):
        result['reason'] = 'Market filter: SPY below 200 SMA (or data missing)'
        return result
    if not _mf.get('vix_below_30', False):
        result['reason'] = 'Market filter: VIX above 30 (or data missing)'
        return result

    # VALID
    result['is_valid'] = True

    bars_ago = result['macd_cross_bars_ago']
    if bars_ago <= 2:
        result['quality'] = '🟢 Fresh Re-Entry'
        result['quality_score'] = 75
    elif bars_ago <= 5:
        result['quality'] = '🟡 Recent Re-Entry'
        result['quality_score'] = 65
    else:
        result['quality'] = '🟠 Late Re-Entry'
        result['quality_score'] = 55

    result['reason'] = f"MACD crossed up {bars_ago}d ago, AO already positive ({ao_now:.1f})"
    return result


def check_late_entry(daily_df: pd.DataFrame,
                     market_filter: Dict = None,
                     macd_profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Late entry: valid signal occurred 1-5 days ago, still actionable if not too extended.

    Finds the most recent primary signal in the last LATE_ENTRY_MAX_DAYS
    and checks if the price hasn't run too far.
    """
    result = {
        'is_valid': False,
        'signal_type': 'LATE_ENTRY',
        'cross_date': None,
        'cross_price': 0,
        'days_since_cross': 0,
        'entry_premium_pct': 0,
        'current_price': 0,
        'reason': '',
    }

    if daily_df is None or len(daily_df) < 50:
        result['reason'] = 'Insufficient data'
        return result

    profile = _resolve_macd_profile(macd_profile)
    profile_for_primary = MACD_PROFILE_LEGACY if profile == MACD_PROFILE_SHADOW else profile

    df = normalize_columns(daily_df).copy()
    df = calculate_macd(df, profile=profile_for_primary)
    df = calculate_ao(df)

    i = len(df) - 1
    current_price = float(df['Close'].iloc[i])
    result['current_price'] = round(current_price, 2)

    # Look back for a valid primary signal in last LATE_ENTRY_MAX_DAYS
    for lookback in range(1, LATE_ENTRY_MAX_DAYS + 1):
        ci = i - lookback
        if ci < ENTRY_WINDOW + 1:
            break

        # Check MACD cross at that bar
        cm = float(df['MACD'].iloc[ci])
        cs = float(df['MACD_Signal'].iloc[ci])
        pm = float(df['MACD'].iloc[ci - 1])
        ps = float(df['MACD_Signal'].iloc[ci - 1])

        if pd.isna(cm) or pd.isna(cs) or pd.isna(pm) or pd.isna(ps):
            continue

        if not (cm > cs and pm <= ps):
            continue

        # AO positive at that bar
        ao_val = float(df['AO'].iloc[ci])
        if pd.isna(ao_val) or ao_val <= 0:
            continue

        # AO zero-cross in prior window
        ao_cross = False
        for j in range(1, ENTRY_WINDOW + 1):
            pi = ci - j
            if pi < 1:
                break
            ab = float(df['AO'].iloc[pi - 1])
            aa = float(df['AO'].iloc[pi])
            if not (pd.isna(ab) or pd.isna(aa)):
                if ab <= 0 and aa > 0:
                    ao_cross = True
                    break

        if not ao_cross:
            continue

        # Found a valid signal — check if still actionable
        cross_price = float(df['Close'].iloc[ci])
        premium = (current_price - cross_price) / cross_price * 100

        result['cross_date'] = (df.index[ci].strftime('%Y-%m-%d')
                                if hasattr(df.index[ci], 'strftime') else str(df.index[ci]))
        result['cross_price'] = round(cross_price, 2)
        result['days_since_cross'] = lookback
        result['entry_premium_pct'] = round(premium, 1)

        if premium > LATE_ENTRY_MAX_PREMIUM:
            result['reason'] = f'Price extended {premium:.1f}% above signal (max {LATE_ENTRY_MAX_PREMIUM}%)'
            return result

        # MACD must still be bullish
        if float(df['MACD'].iloc[i]) <= float(df['MACD_Signal'].iloc[i]):
            result['reason'] = 'MACD no longer bullish'
            return result

        # Market filter — conservative: reject if missing or bearish
        _mf = market_filter or {}
        if not _mf.get('spy_above_200', False) or not _mf.get('vix_below_30', False):
            result['reason'] = 'Market filter failed (or data missing)'
            return result

        result['is_valid'] = True
        result['reason'] = f'Signal {lookback}d ago at ${cross_price:.2f}, premium {premium:.1f}%'
        return result

    result['reason'] = f'No valid signal in last {LATE_ENTRY_MAX_DAYS} days'
    return result


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

def generate_recommendation(signal: EntrySignal,
                            quality: Dict,
                            ao_confirm: Dict = None,
                            reentry: Dict = None,
                            late_entry: Dict = None) -> Dict[str, Any]:
    """
    Generate final recommendation based on all signal types and context.

    Priority: PRIMARY > AO_CONFIRMATION > RE_ENTRY > LATE_ENTRY > MTF_PULLBACK > WATCH > SKIP

    Returns dict with: signal_type, recommendation, summary, conviction (1-10)

    Conviction scoring uses quality_score (0-100) for granularity within grades:
      STRONG BUY: base 7-10 (A=8-10, B=7-8), refined by quality_score
      BUY:        base 5-7
      RE-ENTRY:   base 4-6
      WATCH:      base 2-4
      SKIP:       0-1
    """
    grade = quality.get('quality_grade', 'N/A')
    q_score = quality.get('quality_score', 0)  # 0-100 for fine-tuning
    weekly_bullish = signal.weekly_macd.get('bullish', False)
    # Weekly "healthy" = bullish AND histogram not deeply negative/declining.
    # A ticker where MACD line is barely above signal but histogram is negative
    # (e.g. META: MACD=0.64, signal=-7.70, hist=-8.33) is technically bullish
    # but the weekly trend is rolling over — NOT a healthy pullback base.
    _wk_hist = float(signal.weekly_macd.get('histogram', 0) or 0)
    _wk_weakening = bool(signal.weekly_macd.get('weakening', False))
    _wk_bearish_cross = bool(signal.weekly_macd.get('bearish_cross', False))
    weekly_healthy = weekly_bullish and not _wk_bearish_cross and _wk_hist >= 0
    # Even if hist >= 0, if it's weakening rapidly, flag it
    if weekly_healthy and _wk_weakening and _wk_hist < 1.0:
        weekly_healthy = False
    # Conservative: missing monthly data = unconfirmed, not assumed bullish.
    _has_monthly = bool(signal.monthly_macd) and signal.monthly_macd.get('error') is None
    monthly_macd_bullish = signal.monthly_macd.get('bullish', False) if _has_monthly else False
    monthly_ao_positive = signal.monthly_ao.get('positive', False) if bool(signal.monthly_ao) else False
    monthly_bullish = bool(monthly_macd_bullish and monthly_ao_positive)
    monthly_warning = ""
    if not monthly_macd_bullish:
        monthly_warning += " ⚠️ Monthly MACD bearish headwind!"
    if not monthly_ao_positive:
        monthly_warning += " ⚠️ Monthly AO negative headwind!"

    result = {
        'signal_type': None,
        'recommendation': 'SKIP',
        'summary': '',
        'conviction': 0,
    }

    # Helper: scale quality_score into a range
    # e.g. _q_refine(7, 10, q_score) → 7.0 to 10.0 based on quality_score 0-100
    def _q_refine(low, high, qs):
        return int(round(low + (high - low) * min(qs, 100) / 100))

    def _apply_monthly_ao_guard(out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hard guardrail: if monthly AO is negative, do not allow bullish entry recommendations.
        This enforces consistency with multi-timeframe momentum rules.
        """
        if bool(monthly_ao_positive):
            return out
        rec_u = str(out.get('recommendation', '') or '').upper()
        is_bullish_entry = (
            ('BUY' in rec_u or 'ENTRY' in rec_u)
            and 'WATCH' not in rec_u
            and 'WAIT' not in rec_u
            and 'SKIP' not in rec_u
            and 'AVOID' not in rec_u
        )
        if is_bullish_entry:
            base = str(out.get('summary', '') or '').strip()
            out['recommendation'] = 'WAIT (M-AO)'
            out['summary'] = (
                (base + " " if base else "")
                + "⚠️ Monthly AO negative headwind — no new bullish entries."
            )
            out['conviction'] = min(max(int(out.get('conviction', 0) or 0), 1), 4)
            out['monthly_ao_downgrade'] = True
            out['monthly_ao_value'] = signal.monthly_ao.get('value')
        return out

    def _apply_mtf_zone_guard(out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hard guardrail: prevent bullish entry recommendations when MTF MACD zone check rejects.
        Only applies to HPotter Zone profile where zone approval is the primary timing mechanism.
        Legacy profile uses MACD cross + context modifiers instead.
        """
        # Legacy profile: zone check is informational, not a gate.
        # Context modifiers already penalize extended/declining conditions.
        _profile = str(signal.macd_profile or '').lower()
        if _profile not in (MACD_PROFILE_HPOTTER_ZONE,):
            return out

        zone = signal.mtf_zone_check or {}
        buy_approved = bool(zone.get('buy_approved', False))
        if buy_approved:
            return out

        rec_u = str(out.get('recommendation', '') or '').upper()
        is_bullish_entry = (
            ('BUY' in rec_u or 'ENTRY' in rec_u)
            and 'WATCH' not in rec_u
            and 'WAIT' not in rec_u
            and 'SKIP' not in rec_u
            and 'AVOID' not in rec_u
        )
        if not is_bullish_entry:
            return out

        reject_reason = str(zone.get('reject_reason', '') or '').strip()
        d_zone = str((signal.daily_macd_zone or {}).get('zone', '') or '')
        w_zone = str((signal.weekly_macd_zone or {}).get('zone', '') or '')
        m_zone = str((signal.monthly_macd_zone or {}).get('zone', '') or '')
        reason_txt = reject_reason or f"D:{d_zone} W:{w_zone} M:{m_zone}"
        rr = reason_txt.lower()

        if ('extended' in rr) or ('bearish' in rr):
            out['recommendation'] = 'SKIP'
            out['summary'] = f"⚠️ MACD zone reject ({reason_txt}) — setup extended/bearish, skip new entry."
            out['conviction'] = min(max(int(out.get('conviction', 0) or 0), 0), 1)
        else:
            out['recommendation'] = 'WAIT (ZONE)'
            out['summary'] = f"⚠️ MACD zone timing not approved ({reason_txt}) — wait for valid just-cross entry."
            out['conviction'] = min(max(int(out.get('conviction', 0) or 0), 1), 4)

        out['mtf_zone_downgrade'] = True
        out['mtf_zone_reject_reason'] = reason_txt
        return out

    def _apply_context_modifiers(out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust conviction using contextual data the signal already carries.
        Wires volume, relative strength, VCP, Weinstein stage, and overhead
        resistance into the decision path instead of display-only.
        """
        if out.get('conviction', 0) <= 0:
            return out  # Don't modify SKIP/no-signal

        modifiers = []
        delta = 0

        # ── 1. Weinstein Stage guard ─────────────────────────────────
        w = signal.weinstein or {}
        stage = w.get('stage', 0)
        maturity = w.get('trend_maturity', '')

        if stage == 4:
            delta -= 3
            modifiers.append("Stage 4 decline ⚠️")
            rec_u = str(out.get('recommendation', '')).upper()
            if 'BUY' in rec_u or 'ENTRY' in rec_u:
                if 'WATCH' not in rec_u and 'WAIT' not in rec_u and 'SKIP' not in rec_u:
                    out['recommendation'] = 'WATCH (STAGE 4)'
                    out['stage4_downgrade'] = True
        elif stage == 3:
            delta -= 1
            modifiers.append("Stage 3 topping")
        elif stage == 2:
            if maturity == 'early':
                delta += 1
                modifiers.append("Early Stage 2 advance ✅")
            elif maturity == 'extended':
                delta -= 1
                modifiers.append("Stage 2 extended")

        # ── 2. Volume confirmation ───────────────────────────────────
        vol = signal.volume or {}
        cross_vol = vol.get('cross_volume_ratio')
        ad_trend = vol.get('accum_dist_trend', 'unknown')

        if cross_vol is not None:
            if cross_vol >= 2.0:
                delta += 1
                modifiers.append(f"Strong breakout volume ({cross_vol:.1f}x) ✅")
            elif cross_vol < 0.7:
                delta -= 1
                modifiers.append(f"Weak volume on cross ({cross_vol:.1f}x) ⚠️")

        if ad_trend == 'distributing':
            delta -= 1
            modifiers.append("Distribution pattern ⚠️")
        elif ad_trend == 'accumulating':
            delta += 1
            modifiers.append("Accumulation ✅")

        # ── 3. Relative strength vs SPY ──────────────────────────────
        rs = signal.relative_strength or {}
        rs_1mo = rs.get('rs_1mo')
        rs_3mo = rs.get('rs_3mo')
        rs_trend = rs.get('rs_trend', 'unknown')

        if rs_1mo is not None and rs_3mo is not None:
            if rs_1mo > 5 and rs_3mo > 5:
                delta += 1
                modifiers.append(f"Strong RS vs SPY ✅")
            elif rs_1mo < -5 and rs_3mo < -5:
                delta -= 1
                modifiers.append(f"Weak RS vs SPY ⚠️")

        if rs_trend == 'deteriorating':
            delta -= 1
            modifiers.append("RS deteriorating ⚠️")

        # ── 4. VCP detection bonus ───────────────────────────────────
        vcp = signal.vcp or {}
        if vcp.get('vcp_detected'):
            delta += 1
            vcp_score = vcp.get('vcp_score', 0)
            modifiers.append(f"VCP pattern ({vcp_score:.0f}) ✅")

        # ── 5. Overhead resistance proximity ─────────────────────────
        ores = signal.overhead_resistance or {}
        dist = ores.get('distance_to_critical_pct')
        if dist is not None and 0 < dist < 2.0:
            delta -= 1
            modifiers.append(f"Near resistance ({dist:.1f}%) ⚠️")

        # ── 6. MACD overbought / momentum fading guard ──────────────
        # When daily MACD is elevated above zero AND the histogram is
        # shrinking (momentum fading), this is NOT a fresh entry — it's
        # an extended move losing steam.  Downgrade buy-type signals.
        _d_macd = signal.macd or {}
        _d_macd_val = float(_d_macd.get('macd', 0) or 0)
        _d_hist = float(_d_macd.get('histogram', 0) or 0)
        # 'weakening' is True when histogram is shrinking while MACD still bullish
        _momentum_fading = bool(_d_macd.get('weakening', False))

        # Overbought threshold: MACD well above zero + momentum fading
        if _d_macd_val > 1.5 and _momentum_fading:
            delta -= 2
            modifiers.append(f"MACD overbought ({_d_macd_val:.1f}), momentum fading ⚠️")
            rec_u = str(out.get('recommendation', '')).upper()
            # Hard downgrade: BUY/ENTRY signals become WATCH when overbought
            is_buy = (
                ('BUY' in rec_u or 'ENTRY' in rec_u)
                and 'WATCH' not in rec_u
                and 'WAIT' not in rec_u
                and 'SKIP' not in rec_u
            )
            if is_buy:
                out['recommendation'] = f"WATCH ({out.get('recommendation', 'SIGNAL')})"
                out['overbought_downgrade'] = True
        elif _d_macd_val > 2.0:
            # MACD very elevated even if histogram not yet shrinking — caution
            delta -= 1
            modifiers.append(f"MACD extended ({_d_macd_val:.1f}) ⚠️")

        # ── Apply with bounds ────────────────────────────────────────
        if delta != 0 or modifiers:
            original = out['conviction']
            out['conviction'] = max(0, min(10, out['conviction'] + delta))
            out['context_modifiers'] = modifiers
            out['conviction_delta'] = delta
            # Append key modifiers to summary
            flags = [m for m in modifiers if '✅' in m or '⚠️' in m][:3]
            if flags:
                out['summary'] += ' | ' + ', '.join(flags)

        return out

    def _apply_signal_guards(out: Dict[str, Any]) -> Dict[str, Any]:
        out = _apply_monthly_ao_guard(out)
        out = _apply_mtf_zone_guard(out)
        out = _apply_context_modifiers(out)
        return out

    # --- PRIMARY SIGNAL ---
    if signal.is_valid:
        result['signal_type'] = 'PRIMARY'

        # Volume context for gating STRONG BUY
        _vol = signal.volume or {}
        _cross_vol = _vol.get('cross_volume_ratio')  # None if no cross bar
        _ad_trend = _vol.get('accum_dist_trend', 'unknown')
        _vol_ok = (
            _ad_trend != 'distributing'
            and (_cross_vol is None or _cross_vol >= 0.8)
        )
        _vol_strong = _cross_vol is not None and _cross_vol >= 1.5
        _vol_tag = ""
        if not _vol_ok:
            _vol_tag = " ⚠️ Volume concern"
        elif _vol_strong:
            _vol_tag = " 📊 Strong volume"

        if grade in ['A', 'B'] and weekly_healthy and monthly_bullish and _vol_ok:
            result['recommendation'] = 'STRONG BUY'
            result['summary'] = f"✅ Entry signal valid, Weekly + Monthly bullish, Quality {grade}{_vol_tag}"
            if grade == 'A':
                result['conviction'] = _q_refine(8, 10, q_score)
            else:
                result['conviction'] = _q_refine(7, 9, q_score)
            # Bonus for strong breakout volume
            if _vol_strong:
                result['conviction'] = min(10, result['conviction'] + 1)
        elif grade in ['A', 'B'] and weekly_bullish and not weekly_healthy and monthly_bullish and _vol_ok:
            # Weekly technically bullish but histogram deteriorating — cap at BUY
            result['recommendation'] = 'BUY'
            result['summary'] = f"✅ Entry valid, Weekly bullish but momentum fading, Monthly bullish, Quality {grade}{_vol_tag}"
            result['conviction'] = _q_refine(6, 8, q_score)
        elif grade in ['A', 'B'] and weekly_bullish and monthly_bullish and not _vol_ok:
            # All timeframes aligned but volume is distributing or weak — downgrade to BUY
            result['recommendation'] = 'BUY'
            result['summary'] = f"✅ Entry valid, all TFs bullish but volume weak, Quality {grade}{_vol_tag}"
            result['conviction'] = _q_refine(6, 8, q_score)
        elif grade in ['A', 'B'] and weekly_bullish:
            result['recommendation'] = 'BUY'
            result['summary'] = f"✅ Entry valid, Weekly bullish, Quality {grade}{monthly_warning}{_vol_tag}"
            result['conviction'] = _q_refine(6, 8, q_score)
        elif grade in ['A', 'B']:
            # No weekly confirmation — WATCH not BUY
            result['recommendation'] = 'WATCH'
            result['summary'] = f"🟡 Entry valid, Quality {grade}, awaiting Weekly confirmation{monthly_warning}{_vol_tag}"
            result['conviction'] = _q_refine(4, 6, q_score)
        elif grade == 'C':
            result['recommendation'] = 'WAIT'
            result['summary'] = f"⚠️ Entry valid but Quality {grade}{monthly_warning}"
            result['conviction'] = _q_refine(3, 5, q_score)
        else:
            result['recommendation'] = 'SKIP'
            result['summary'] = f"❌ Entry valid but Quality {grade}"
            result['conviction'] = 1
        return _apply_signal_guards(result)

    # --- AO CONFIRMATION ---
    if ao_confirm and ao_confirm.get('is_valid'):
        result['signal_type'] = 'AO_CONFIRMATION'
        ao_q = ao_confirm.get('quality_score', 0)

        if grade in ['A', 'B'] and ao_q >= 80 and weekly_bullish:
            result['recommendation'] = 'BUY (AO)'
            result['summary'] = f"🔄 AO Confirmation, Weekly bullish, Quality {grade}{monthly_warning}"
            result['conviction'] = _q_refine(6, 8, q_score)
        elif grade in ['A', 'B'] and ao_q >= 65:
            result['recommendation'] = 'WATCH (AO)'
            result['summary'] = f"🟡 AO Confirmation, Quality {grade}{monthly_warning}"
            result['conviction'] = _q_refine(4, 6, q_score)
        else:
            result['recommendation'] = 'SKIP'
            result['summary'] = f"⚠️ AO Confirmation but weak"
            result['conviction'] = 2
        return _apply_signal_guards(result)

    # --- RE-ENTRY ---
    if reentry and reentry.get('is_valid'):
        result['signal_type'] = 'RE_ENTRY'
        bars_ago = reentry.get('macd_cross_bars_ago', 0)

        if grade in ['A', 'B'] and weekly_bullish:
            result['recommendation'] = 'RE-ENTRY'
            result['summary'] = f"🔁 Re-Entry ({bars_ago}d ago), Weekly bullish, Quality {grade}{monthly_warning}"
            result['conviction'] = _q_refine(5, 7, q_score)
        elif grade in ['A', 'B']:
            # No weekly confirmation — WATCH not green RE-ENTRY.
            # Re-entering an established trend without weekly backing is risky.
            result['recommendation'] = 'WATCH (RE-ENTRY)'
            result['summary'] = f"🟡 Re-Entry ({bars_ago}d ago), Quality {grade}, no Weekly confirmation{monthly_warning}"
            result['conviction'] = _q_refine(3, 5, q_score)
        else:
            result['recommendation'] = 'WATCH'
            result['summary'] = f"🟡 Re-Entry but Quality {grade}"
            result['conviction'] = _q_refine(2, 4, q_score)
        return _apply_signal_guards(result)

    # --- LATE ENTRY ---
    if late_entry and late_entry.get('is_valid'):
        result['signal_type'] = 'LATE_ENTRY'
        days = late_entry.get('days_since_cross', 0)
        premium = late_entry.get('entry_premium_pct', 0)

        if grade in ['A', 'B'] and premium <= 3 and weekly_bullish:
            result['recommendation'] = f'LATE ENTRY (+{days}d)'
            result['summary'] = f"🕐 Signal {days}d ago, +{premium:.1f}% premium, Weekly bullish, Quality {grade}{monthly_warning}"
            result['conviction'] = _q_refine(4, 6, q_score)
        elif grade in ['A', 'B'] and premium <= 3:
            # No weekly — downgrade to WATCH
            result['recommendation'] = f'WATCH (LATE +{days}d)'
            result['summary'] = f"🕐 Signal {days}d ago, +{premium:.1f}% premium, Quality {grade}, no Weekly confirmation{monthly_warning}"
            result['conviction'] = _q_refine(3, 4, q_score)
        else:
            result['recommendation'] = 'WATCH'
            result['summary'] = f"🕐 Signal {days}d ago, +{premium:.1f}% premium"
            result['conviction'] = _q_refine(2, 3, q_score)
        return _apply_signal_guards(result)

    # --- MTF PULLBACK ENTRY ---
    # Daily MACD just crossed bullish from oversold (below zero line) while
    # weekly + monthly MACD are both positive.  This is a high-quality pullback
    # buy in an established multi-timeframe uptrend.  AO confirmation is NOT
    # required — the higher-timeframe trend alignment substitutes for it.
    _daily_cross = bool(signal.macd.get('cross_recent', False))
    _daily_macd_val = float(signal.macd.get('macd', 0) or 0)
    _daily_hist = float(signal.macd.get('histogram', 0) or 0)
    _cross_bars = int(signal.macd.get('cross_bars_ago', 0) or 0)
    # "From oversold" = MACD line is still near zero (just crossed up from negative)
    # or histogram is small-positive (early days of the cross).
    _from_oversold = _daily_cross and (_daily_macd_val <= 0.5 or _daily_hist < 1.0)
    # Also accept when MACD crossed recently and is still below its own signal line
    # by a tiny amount — histogram barely positive.
    if not _from_oversold and _daily_cross and _cross_bars <= 3:
        _from_oversold = True  # Any fresh cross within 3 bars qualifies

    if _from_oversold and weekly_healthy and monthly_bullish:
        # Best case: weekly trend is healthy (positive histogram, not weakening)
        # AND monthly is bullish — this is a genuine pullback in a strong uptrend
        result['signal_type'] = 'MTF_PULLBACK'
        _mtf_note = f"Daily MACD cross from oversold ({_cross_bars}d ago, MACD={_daily_macd_val:.2f})"

        if grade in ['A', 'B']:
            result['recommendation'] = 'BUY (PULLBACK)'
            result['summary'] = (
                f"🔄 {_mtf_note}, Weekly + Monthly bullish, Quality {grade}"
            )
            result['conviction'] = _q_refine(6, 8, q_score)
        elif grade in ['C']:
            result['recommendation'] = 'WATCH (PULLBACK)'
            result['summary'] = (
                f"🔄 {_mtf_note}, Weekly + Monthly bullish, Quality {grade}"
            )
            result['conviction'] = _q_refine(4, 6, q_score)
        else:
            result['recommendation'] = 'WATCH'
            result['summary'] = f"🔄 {_mtf_note}, Quality {grade}"
            result['conviction'] = _q_refine(2, 4, q_score)
        return _apply_signal_guards(result)

    if _from_oversold and weekly_bullish and not weekly_healthy and monthly_bullish:
        # Weekly is technically bullish (MACD > signal) but histogram is negative
        # or weakening — the weekly uptrend is rolling over.  This is NOT a
        # high-quality pullback; it's a bounce inside a deteriorating trend.
        # Downgrade to WATCH regardless of quality grade.
        result['signal_type'] = 'MTF_PULLBACK'
        _wk_warn = []
        if _wk_hist < 0:
            _wk_warn.append(f"wkly hist={_wk_hist:.1f}")
        if _wk_weakening:
            _wk_warn.append("wkly momentum fading")
        if _wk_bearish_cross:
            _wk_warn.append("wkly bearish cross")
        _wk_tag = ", ".join(_wk_warn) if _wk_warn else "wkly trend deteriorating"
        result['recommendation'] = 'WATCH (PULLBACK)'
        result['summary'] = (
            f"🔄 Daily MACD cross from oversold ({_cross_bars}d ago), "
            f"Weekly technically bullish but {_wk_tag}, Monthly bullish, Quality {grade}"
        )
        result['conviction'] = _q_refine(3, 4, q_score) if grade in ['A', 'B'] else _q_refine(2, 3, q_score)
        return _apply_signal_guards(result)

    # Also catch: daily MACD just crossed from oversold + weekly bullish (no monthly)
    # — still worth a WATCH, not a SKIP.
    if _from_oversold and weekly_bullish and not monthly_bullish:
        result['signal_type'] = 'MTF_PULLBACK'
        _wk_health_tag = "" if weekly_healthy else " (wkly trend weakening)"
        result['recommendation'] = 'WATCH (PULLBACK)'
        result['summary'] = (
            f"🔄 Daily MACD cross from oversold, Weekly bullish{_wk_health_tag} but Monthly not confirmed, "
            f"Quality {grade}{monthly_warning}"
        )
        result['conviction'] = _q_refine(3, 5, q_score) if grade in ['A', 'B'] and weekly_healthy else _q_refine(2, 3, q_score)
        return _apply_signal_guards(result)

    # --- NO SIGNAL ---
    if signal.is_valid_relaxed:
        result['recommendation'] = 'WATCH'
        result['summary'] = f"🟡 MACD bullish but no fresh cross, Quality {grade}"
        result['conviction'] = 2
    else:
        result['recommendation'] = 'SKIP'
        failed = []
        if not signal.macd.get('bullish', False):
            failed.append("MACD bearish")
        if not signal.ao.get('positive', False):
            failed.append("AO negative")
        if not signal.ao.get('zero_cross_found', False):
            failed.append("No AO zero-cross")
        result['summary'] = f"❌ {', '.join(failed)}" if failed else "❌ No signal"
        result['conviction'] = 0

    return _apply_signal_guards(result)


# =============================================================================
# FULL TICKER ANALYSIS — Orchestrator
# =============================================================================

@dataclass
class TickerAnalysis:
    """Complete analysis result for a single ticker."""
    ticker: str
    timestamp: str = ''

    # Core signal
    signal: EntrySignal = None

    # Secondary signals
    ao_confirmation: Dict = field(default_factory=dict)
    reentry: Dict = field(default_factory=dict)
    late_entry: Dict = field(default_factory=dict)

    # Quality
    quality: Dict = field(default_factory=dict)

    # Recommendation
    recommendation: Dict = field(default_factory=dict)

    # Current price
    current_price: float = None

    # AO divergence (bearish warning)
    ao_divergence_active: bool = False

    # Volume data
    volume: float = 0
    avg_volume_50d: float = 0
    volume_ratio: float = 0  # today_vol / avg_50d

    # Apex signal
    apex_buy: bool = False
    apex_signal_days_ago: int = 999
    apex_signal_tier: str = ''
    apex_signal_regime: str = ''

    # Error
    error: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'current_price': self.current_price,
            'signal': self.signal.to_dict() if self.signal else {},
            'ao_confirmation': self.ao_confirmation,
            'reentry': self.reentry,
            'late_entry': self.late_entry,
            'quality': self.quality,
            'recommendation': self.recommendation,
            'ao_divergence_active': self.ao_divergence_active,
            'volume': self.volume,
            'avg_volume_50d': self.avg_volume_50d,
            'volume_ratio': self.volume_ratio,
            'apex_buy': self.apex_buy,
            'apex_signal_days_ago': self.apex_signal_days_ago,
            'apex_signal_tier': self.apex_signal_tier,
            'apex_signal_regime': self.apex_signal_regime,
            'error': self.error,
        }


def analyze_ticker(ticker_data: Dict[str, Any], macd_profile: Optional[str] = None) -> TickerAnalysis:
    """
    Full analysis of a single ticker using pre-fetched data.

    ticker_data should come from data_fetcher.fetch_all_ticker_data()
    with keys: ticker, daily, weekly, monthly, spy_daily, market_filter

    Returns TickerAnalysis with signal, quality, recommendation.
    """
    ticker = ticker_data.get('ticker', '???')
    result = TickerAnalysis(
        ticker=ticker,
        timestamp=datetime.now().isoformat(),
        current_price=ticker_data.get('current_price'),
    )

    daily = ticker_data.get('daily')
    weekly = ticker_data.get('weekly')
    monthly = ticker_data.get('monthly')
    spy = ticker_data.get('spy_daily')
    mkt = ticker_data.get('market_filter', {})

    if daily is None or len(daily) < 50:
        result.error = 'Insufficient daily data'
        result.recommendation = {'signal_type': None, 'recommendation': 'SKIP',
                                 'summary': '❌ Insufficient data', 'conviction': 0}
        return result

    profile_req = _resolve_macd_profile(macd_profile)
    profile_for_primary = MACD_PROFILE_LEGACY if profile_req == MACD_PROFILE_SHADOW else profile_req

    # --- Primary entry signal ---
    signal = validate_entry(
        daily_df=daily,
        weekly_df=weekly,
        monthly_df=monthly,
        spy_df=spy,
        ticker=ticker,
        macd_profile=profile_req,
    )

    # Inject market filter (VIX comes from data_fetcher)
    signal.market_filter = mkt
    
    # Re-evaluate validity with full market filter
    # Conservative: missing market data = unconfirmed
    _has_mkt = bool(mkt) and mkt.get('spy_close') is not None
    spy_ok = mkt.get('spy_above_200', False) if _has_mkt else False
    vix_ok = mkt.get('vix_below_30', False) if _has_mkt else False
    
    legacy_valid = all([
        signal.macd['cross_recent'],
        signal.ao['positive'],
        signal.ao['zero_cross_found'],
        spy_ok, vix_ok,
    ])
    legacy_relaxed = all([
        signal.macd['bullish'],
        signal.ao['positive'],
        signal.ao['zero_cross_found'],
        spy_ok, vix_ok,
    ])
    if profile_for_primary == MACD_PROFILE_HPOTTER_ZONE:
        d_zone = str((signal.daily_macd_zone or {}).get('zone', 'neutral'))
        w_zone = str((signal.weekly_macd_zone or {}).get('zone', 'neutral'))
        m_zone = str((signal.monthly_macd_zone or {}).get('zone', 'neutral'))
        d_recent = bool((signal.daily_macd_zone or {}).get('recent_cross', False))
        d_hist_pct = float((signal.daily_macd_zone or {}).get('hist_pct', 1.0) or 1.0)
        daily_relaxed_ok = (d_zone != 'bearish') and not (
            d_zone == 'extended' and (not d_recent or d_hist_pct > 0.95)
        )
        signal.is_valid = all([
            bool((signal.mtf_zone_check or {}).get('buy_approved', False)),
            signal.ao['positive'],
            signal.ao['zero_cross_found'],
            spy_ok,
            vix_ok,
        ])
        signal.is_valid_relaxed = all([
            daily_relaxed_ok,
            w_zone not in {'bearish'},
            m_zone not in {'bearish'},
            signal.ao['positive'],
            spy_ok,
            vix_ok,
        ])
    else:
        signal.is_valid = legacy_valid
        signal.is_valid_relaxed = legacy_relaxed
    if profile_req == MACD_PROFILE_SHADOW:
        zone_ok = bool((signal.mtf_zone_check or {}).get('buy_approved', False))
        signal.mtf_zone_check['shadow_diff'] = bool(legacy_valid != zone_ok)
        signal.mtf_zone_check['legacy_is_valid'] = bool(legacy_valid)
        signal.mtf_zone_check['zone_is_valid'] = bool(zone_ok)

    result.signal = signal

    # --- Secondary signals (only if primary fails) ---
    if not signal.is_valid:
        result.ao_confirmation = check_ao_confirmation(daily, market_filter=mkt, macd_profile=profile_for_primary)

        if not result.ao_confirmation.get('is_valid'):
            result.reentry = check_reentry_signal(daily, market_filter=mkt, macd_profile=profile_for_primary)

            if not result.reentry.get('is_valid'):
                result.late_entry = check_late_entry(daily, market_filter=mkt, macd_profile=profile_for_primary)

    # --- Quality score ---
    # Use longer history for backtest
    backtest_daily = ticker_data.get('daily')  # Already 1y
    backtest_weekly = ticker_data.get('weekly')  # Already 2y
    result.quality = calculate_quality_score(
        backtest_daily,
        backtest_weekly,
        ticker=ticker,
        macd_profile=profile_for_primary,
    )

    # --- AO Bearish Divergence Detection ---
    try:
        from signal_engine import detect_bearish_divergence
        div_df = detect_bearish_divergence(daily)
        if div_df is not None and 'bearish_div_active' in div_df.columns:
            # Check if divergence is currently active (last bar)
            result.ao_divergence_active = bool(div_df['bearish_div_active'].iloc[-1])
    except Exception:
        pass

    # --- Volume Data ---
    try:
        if daily is not None and len(daily) > 50 and 'Volume' in daily.columns:
            result.volume = float(daily['Volume'].iloc[-1])
            result.avg_volume_50d = float(daily['Volume'].tail(50).mean())
            if result.avg_volume_50d > 0:
                result.volume_ratio = round(result.volume / result.avg_volume_50d, 2)
    except Exception:
        pass

    # --- Apex Buy Signal Detection ---
    try:
        from apex_signals import detect_apex_signals
        if weekly is not None and monthly is not None:
            spy = ticker_data.get('spy_daily')
            apex_signals = detect_apex_signals(
                ticker=ticker,
                daily_data=daily,
                weekly_data=weekly,
                monthly_data=monthly,
                spy_data=spy,
                vix_data=ticker_data.get('vix_daily'),
            )
            if apex_signals:
                last_date = daily.index[-1] if len(daily) > 0 else None
                active = [s for s in apex_signals if bool(getattr(s, 'is_active', False))]
                ref_sig = active[-1] if active else apex_signals[-1]
                sig_date = getattr(ref_sig, 'entry_date', None)
                sig_days = 999
                if sig_date is not None and last_date is not None:
                    try:
                        sig_days = max(0, int((last_date - sig_date).days))
                    except Exception:
                        sig_days = 999
                result.apex_signal_days_ago = sig_days
                result.apex_signal_tier = str(getattr(ref_sig, 'signal_tier', '') or '')
                result.apex_signal_regime = str(getattr(ref_sig, 'monthly_regime', '') or '')
                # APEX buy if a signal is still active or fired very recently.
                result.apex_buy = bool(active) or sig_days <= 7
    except Exception:
        pass

    # --- Recommendation ---
    result.recommendation = generate_recommendation(
        signal=signal,
        quality=result.quality,
        ao_confirm=result.ao_confirmation,
        reentry=result.reentry,
        late_entry=result.late_entry,
    )

    # --- AO Divergence Downgrade ---
    # If bearish divergence is active, downgrade any bullish entry signals.
    # Divergence means price making new highs but AO making lower highs —
    # momentum is failing even if other indicators look green.
    if result.ao_divergence_active:
        rec = result.recommendation
        original_rec = rec.get('recommendation', '')
        rec_upper = original_rec.upper()
        # Catch all buy-type signals: STRONG BUY, BUY, BUY (AO), RE-ENTRY,
        # LATE ENTRY, BUY (PULLBACK)
        _is_bullish = (
            ('BUY' in rec_upper or 'ENTRY' in rec_upper)
            and 'WATCH' not in rec_upper
            and 'WAIT' not in rec_upper
            and 'SKIP' not in rec_upper
        )
        if _is_bullish:
            rec['recommendation'] = 'WAIT (D)'
            rec['summary'] = f"⚠️ {rec.get('summary', '')} — AO divergence active, momentum failing"
            rec['conviction'] = max(1, rec.get('conviction', 0) - 2)
            rec['ao_divergence_downgrade'] = True
            rec['original_recommendation'] = original_rec

    return result


def scan_watchlist(all_ticker_data: Dict[str, Dict], macd_profile: Optional[str] = None) -> List[TickerAnalysis]:
    """
    Scan an entire watchlist.

    all_ticker_data: dict of ticker -> data from data_fetcher.fetch_scan_data()

    Returns list of TickerAnalysis sorted by conviction (highest first).
    """
    results = []
    for ticker, data in all_ticker_data.items():
        try:
            analysis = analyze_ticker(data, macd_profile=macd_profile)
            results.append(analysis)
        except Exception as e:
            err = TickerAnalysis(
                ticker=ticker,
                timestamp=datetime.now().isoformat(),
                error=str(e),
                recommendation={'signal_type': None, 'recommendation': 'ERROR',
                                'summary': f'❌ {str(e)}', 'conviction': 0}
            )
            results.append(err)

    # Sort by conviction descending, then by quality grade
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'F': 3, 'N/A': 4}
    results.sort(key=lambda r: (
        -r.recommendation.get('conviction', 0),
        grade_order.get(r.quality.get('quality_grade', 'N/A'), 4),
    ))

    return results
