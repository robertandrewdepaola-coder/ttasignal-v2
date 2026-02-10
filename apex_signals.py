"""
APEX MTF Wave 3 Signal Detection
=================================
Adapted from backtest V4 for live chart overlay in ttasignal-v2.

Uses signal_engine's MACD/AO calculations (SMA signal line, AO/2)
so signals are consistent with what the chart displays.

Usage:
    from apex_signals import detect_apex_signals, get_apex_markers, get_apex_summary

Version: 1.0.0 (2026-02-10)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from signal_engine import (
    normalize_columns, calculate_macd, calculate_ao, calculate_atr,
    ATR_PERIOD,
)

# ============================================================
# CONFIGURATION (matches backtest V4)
# ============================================================

ENTRY_WINDOW = 20           # AO zero cross lookback
ATR_MULTIPLIER = 2.0
PROTECTIVE_STOP_PCT = -12.0  # Standard stop for Monthly_Bullish
CURLING_STOP_PCT = -8.0      # Tighter stop for Monthly_Curling
PROFIT_TRAIL_THRESHOLD = 15.0
MIN_HOLD_WEEKS = 3           # Conditional min hold


# ============================================================
# DATACLASS
# ============================================================

@dataclass
class ApexSignal:
    """Single APEX MTF entry signal with exit tracking."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    win: Optional[bool] = None
    weekly_confirmed: bool = False
    monthly_regime: str = ""       # Monthly_Bullish or Monthly_Curling
    signal_tier: str = ""          # Tier_1, Tier_2, Tier_3
    exit_reason: Optional[str] = None
    hold_weeks: Optional[float] = None
    stop_level: float = -12.0
    atr_trail_active: bool = False
    highest_price: float = 0.0
    is_active: bool = False


# ============================================================
# BEAR MARKET FILTER
# ============================================================

def _is_bear_market(date, spy_data, vix_data):
    """Bear market = SPY below 200-day SMA AND VIX above 20."""
    if spy_data is None or vix_data is None:
        return False
    try:
        spy_slice = spy_data.loc[:date]
        if len(spy_slice) < 200:
            return False
        spy_sma200 = spy_slice['Close'].rolling(200).mean().iloc[-1]
        spy_close = spy_slice['Close'].iloc[-1]

        vix_slice = vix_data.loc[:date]
        if len(vix_slice) == 0:
            return False
        vix_close = vix_slice['Close'].iloc[-1]

        return spy_close < spy_sma200 and vix_close > 20
    except Exception:
        return False


# ============================================================
# MAIN DETECTION
# ============================================================

def detect_apex_signals(
    ticker: str,
    daily_data: pd.DataFrame,
    weekly_data: pd.DataFrame,
    monthly_data: pd.DataFrame,
    spy_data: pd.DataFrame = None,
    vix_data: pd.DataFrame = None,
) -> List[ApexSignal]:
    """
    Detect APEX MTF Wave 3 signals on historical data.

    Uses signal_engine's MACD (SMA signal line) and AO (/2)
    for consistency with the chart display.

    Returns list of ApexSignal objects — both completed and active trades.
    """
    # Normalize and add indicators using signal_engine functions
    daily = normalize_columns(daily_data.copy())
    weekly = normalize_columns(weekly_data.copy())
    monthly = normalize_columns(monthly_data.copy())

    daily = calculate_macd(daily)
    daily = calculate_ao(daily)
    daily = calculate_atr(daily, period=ATR_PERIOD)
    weekly = calculate_macd(weekly)
    monthly = calculate_macd(monthly)

    # Normalize SPY/VIX if provided
    if spy_data is not None:
        spy_data = normalize_columns(spy_data.copy())
    if vix_data is not None:
        vix_data = normalize_columns(vix_data.copy())

    recent_high = daily['Close'].rolling(window=ENTRY_WINDOW).max()
    min_hold_days = MIN_HOLD_WEEKS * 5

    signals = []
    current_exit_date = None

    for i in range(2, len(daily)):
        current_date = daily.index[i]

        # Skip if still in a previous trade
        if current_exit_date is not None and current_date <= current_exit_date:
            continue

        # === FILTER 1: Daily MACD Entry ===
        macd_now = daily['MACD'].iloc[i]
        macd_prev1 = daily['MACD'].iloc[i - 1]
        macd_prev2 = daily['MACD'].iloc[i - 2]
        sig_now = daily['MACD_Signal'].iloc[i]
        sig_prev1 = daily['MACD_Signal'].iloc[i - 1]

        if pd.isna(macd_now) or pd.isna(sig_now) or pd.isna(sig_prev1):
            continue

        macd_cross_late = (macd_now > sig_now) and (macd_prev1 <= sig_prev1)
        macd_higher_low = (macd_prev1 < sig_prev1) and (macd_prev1 > macd_prev2)
        macd_turning_up = macd_now > macd_prev1
        macd_curl_early = macd_higher_low and macd_turning_up

        if not (macd_curl_early or macd_cross_late):
            continue

        # === FILTER 2: AO Positive + Recent Zero Cross ===
        daily_ao = daily['AO'].iloc[i]
        if pd.isna(daily_ao) or daily_ao <= 0:
            continue

        ao_cross_found = False
        lookback = min(i, ENTRY_WINDOW)
        for j in range(1, lookback):
            prev_ao = daily['AO'].iloc[i - j]
            next_ao = daily['AO'].iloc[i - j + 1]
            if not pd.isna(prev_ao) and not pd.isna(next_ao):
                if prev_ao <= 0 and next_ao > 0:
                    ao_cross_found = True
                    break
        if not ao_cross_found:
            continue

        # === FILTER 3: Price Strength (within 5% of recent high) ===
        px = daily['Close'].iloc[i]
        rh = recent_high.iloc[i]
        if pd.isna(rh) or px < 0.95 * rh:
            continue

        # === FILTER 4: Bear Market Filter ===
        if _is_bear_market(current_date, spy_data, vix_data):
            continue

        # === FILTER 5: Weekly MACD ===
        weekly_dates = weekly.index[weekly.index <= current_date]
        if len(weekly_dates) == 0:
            continue

        weekly_idx = len(weekly_dates) - 1
        if weekly_idx < 26:
            continue

        weekly_macd = weekly['MACD'].iloc[weekly_idx]
        weekly_signal = weekly['MACD_Signal'].iloc[weekly_idx]
        weekly_macd_prev = weekly['MACD'].iloc[weekly_idx - 1]
        weekly_signal_prev = weekly['MACD_Signal'].iloc[weekly_idx - 1]

        if pd.isna(weekly_macd) or pd.isna(weekly_signal):
            continue

        if weekly_macd > weekly_signal:
            weekly_macd_confirmed = True
        elif (weekly_macd > weekly_macd_prev) and \
             (weekly_macd - weekly_signal) > (weekly_macd_prev - weekly_signal_prev):
            weekly_macd_confirmed = False  # Curling but not confirmed
        else:
            continue  # Weekly bearish — skip

        # === FILTER 6: Monthly MACD Regime ===
        monthly_dates = monthly.index[monthly.index <= current_date]
        if len(monthly_dates) < 3:
            continue

        monthly_idx = len(monthly_dates) - 1
        m_macd = monthly['MACD'].iloc[monthly_idx]
        m_signal = monthly['MACD_Signal'].iloc[monthly_idx]
        m_macd_prev1 = monthly['MACD'].iloc[monthly_idx - 1]
        m_signal_prev1 = monthly['MACD_Signal'].iloc[monthly_idx - 1]
        m_macd_prev2 = monthly['MACD'].iloc[monthly_idx - 2]
        m_signal_prev2 = monthly['MACD_Signal'].iloc[monthly_idx - 2]

        if pd.isna(m_macd) or pd.isna(m_signal) or pd.isna(m_macd_prev2):
            continue

        m_hist_now = m_macd - m_signal
        m_hist_prev1 = m_macd_prev1 - m_signal_prev1
        m_hist_prev2 = m_macd_prev2 - m_signal_prev2

        if m_macd > m_signal:
            monthly_regime = "Monthly_Bullish"
        elif m_hist_now > m_hist_prev1 and m_hist_prev1 > m_hist_prev2:
            monthly_regime = "Monthly_Curling"
            if not weekly_macd_confirmed:
                continue  # Curling regime needs confirmed weekly
        else:
            continue  # Monthly bearish — skip

        # === SIGNAL TIER ===
        if weekly_macd_confirmed and monthly_regime == "Monthly_Bullish":
            signal_tier = "Tier_1"
        elif weekly_macd_confirmed and monthly_regime == "Monthly_Curling":
            signal_tier = "Tier_2"
        else:
            signal_tier = "Tier_3"

        # === STOP LEVEL ===
        active_stop = CURLING_STOP_PCT if monthly_regime == "Monthly_Curling" else PROTECTIVE_STOP_PCT

        # === EXIT SCAN ===
        entry_price = daily['Close'].iloc[i]
        entry_atr = daily['ATR'].iloc[i]

        if pd.isna(entry_atr):
            continue

        highest_price = entry_price
        exit_idx = None
        exit_price = None
        exit_reason = None
        atr_trail_active = False

        for future_day in range(i + 1, len(daily)):
            future_price = daily['Close'].iloc[future_day]
            future_date = daily.index[future_day]
            days_held = future_day - i

            if future_price > highest_price:
                highest_price = future_price

            current_return = (future_price - entry_price) / entry_price * 100.0

            if current_return >= PROFIT_TRAIL_THRESHOLD:
                atr_trail_active = True

            # EXIT 1: Protective stop
            if current_return <= active_stop:
                exit_idx = future_day
                exit_price = future_price
                exit_reason = "Stop_Loss"
                break

            # EXIT 2: Weekly cross (conditional min hold)
            allow_weekly_exit = True
            if days_held < min_hold_days and current_return > 0:
                allow_weekly_exit = False

            if allow_weekly_exit:
                future_weekly_dates = weekly.index[weekly.index <= future_date]
                if len(future_weekly_dates) > 0:
                    fw_idx = len(future_weekly_dates) - 1
                    if fw_idx > weekly_idx and fw_idx >= 1:
                        curr_w_macd = weekly['MACD'].iloc[fw_idx]
                        curr_w_signal = weekly['MACD_Signal'].iloc[fw_idx]
                        prev_w_macd = weekly['MACD'].iloc[fw_idx - 1]
                        prev_w_signal = weekly['MACD_Signal'].iloc[fw_idx - 1]

                        if not pd.isna(curr_w_macd) and not pd.isna(prev_w_macd):
                            if curr_w_macd < curr_w_signal and prev_w_macd >= prev_w_signal:
                                exit_idx = future_day
                                exit_price = future_price
                                exit_reason = "Weekly_Cross_Down"
                                break

            # EXIT 3: ATR trailing stop
            if atr_trail_active:
                future_atr = daily['ATR'].iloc[future_day]
                if not pd.isna(future_atr):
                    trail_level = highest_price - ATR_MULTIPLIER * future_atr
                    if future_price < trail_level:
                        exit_idx = future_day
                        exit_price = future_price
                        exit_reason = "ATR_Trail"
                        break

        # Determine if trade is still active
        is_active = exit_idx is None

        if is_active:
            exit_date_val = None
            exit_price_val = daily['Close'].iloc[-1]
            return_pct = (exit_price_val - entry_price) / entry_price * 100.0
            hold_days = (daily.index[-1] - current_date).days
            current_exit_date = None
        else:
            exit_date_val = daily.index[exit_idx]
            exit_price_val = exit_price
            return_pct = (exit_price - entry_price) / entry_price * 100.0
            hold_days = (daily.index[exit_idx] - current_date).days
            current_exit_date = daily.index[exit_idx]

        signals.append(ApexSignal(
            ticker=ticker,
            entry_date=current_date,
            entry_price=entry_price,
            exit_date=exit_date_val,
            exit_price=exit_price_val,
            return_pct=return_pct,
            win=return_pct > 0 if not is_active else None,
            weekly_confirmed=weekly_macd_confirmed,
            monthly_regime=monthly_regime,
            signal_tier=signal_tier,
            exit_reason=exit_reason if not is_active else "Active",
            hold_weeks=hold_days / 7,
            stop_level=active_stop,
            atr_trail_active=atr_trail_active,
            highest_price=highest_price,
            is_active=is_active,
        ))

    return signals


# ============================================================
# CHART MARKER GENERATION
# ============================================================

def get_apex_markers(signals: List[ApexSignal]) -> List[Dict]:
    """
    Convert APEX signals to LWC markers.
    Format matches chart_engine.py: {time, position, color, shape, text}
    """
    markers = []

    for sig in signals:
        tier_short = sig.signal_tier.replace("Tier_", "T")
        regime_short = "Bull" if sig.monthly_regime == "Monthly_Bullish" else "Curl"

        # --- ENTRY ---
        markers.append({
            "time": sig.entry_date.strftime("%Y-%m-%d"),
            "position": "belowBar",
            "color": "#00e676",
            "shape": "arrowUp",
            "text": f"APEX {tier_short}",
        })

        # --- EXIT (completed trades) ---
        if not sig.is_active and sig.exit_date is not None:
            color_map = {
                "Stop_Loss": "#ff1744",
                "ATR_Trail": "#ff9100",
                "Weekly_Cross_Down": "#ffea00",
            }
            exit_color = color_map.get(sig.exit_reason, "#ffffff")
            ret_str = f"{sig.return_pct:+.1f}%"

            markers.append({
                "time": sig.exit_date.strftime("%Y-%m-%d"),
                "position": "aboveBar",
                "color": exit_color,
                "shape": "arrowDown",
                "text": f"EXIT {ret_str}",
            })

        # --- ACTIVE TRADE ---
        elif sig.is_active:
            ret_str = f"{sig.return_pct:+.1f}%"
            markers.append({
                "time": sig.entry_date.strftime("%Y-%m-%d"),
                "position": "aboveBar",
                "color": "#00e5ff",
                "shape": "circle",
                "text": f"ACTIVE {ret_str}",
            })

    return markers


def get_apex_summary(signals: List[ApexSignal]) -> Dict:
    """Quick summary stats for display in the UI."""
    if not signals:
        return {"total": 0}

    completed = [s for s in signals if not s.is_active]
    active = [s for s in signals if s.is_active]
    wins = [s for s in completed if s.win]

    summary = {
        "total": len(signals),
        "completed": len(completed),
        "active": len(active),
        "wins": len(wins),
        "losses": len(completed) - len(wins),
        "win_rate": (len(wins) / len(completed) * 100) if completed else 0,
        "avg_return": np.mean([s.return_pct for s in completed]) if completed else 0,
        "best_trade": max([s.return_pct for s in completed]) if completed else 0,
        "worst_trade": min([s.return_pct for s in completed]) if completed else 0,
    }

    if active:
        a = active[-1]
        summary["active_trade"] = {
            "entry_date": a.entry_date.strftime("%Y-%m-%d"),
            "entry_price": a.entry_price,
            "current_return": a.return_pct,
            "tier": a.signal_tier,
            "regime": a.monthly_regime,
            "stop": a.stop_level,
            "atr_trail_active": a.atr_trail_active,
            "highest_price": a.highest_price,
        }

    return summary
