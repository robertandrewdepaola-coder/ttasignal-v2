"""
TTA v2 Signal Engine — Single Source of Truth
==============================================

ALL indicator calculations and signal detection logic lives here.
No other module should calculate MACD, AO, ATR, or detect signals.

Matches TradingView Pine Script indicators exactly:
- MACD: EMA(12,26) with SMA(9) signal line
- AO: (SMA(hl2,5) - SMA(hl2,34)) / 2
- Signal line uses SMA, NOT EMA

Version: 2.0.0 (2026-02-07)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# CONSTANTS — Define once, use everywhere
# =============================================================================

# MACD parameters (matches TradingView AO+MACD overlay)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Awesome Oscillator parameters
AO_FAST = 5
AO_SLOW = 34

# ATR
ATR_PERIOD = 14

# Entry window: how far back to look for AO zero-cross
ENTRY_WINDOW = 20

# MACD cross lookback: how many bars back a cross is still actionable
MACD_CROSS_LOOKBACK = 10

# Late entry
LATE_ENTRY_MAX_DAYS = 5
LATE_ENTRY_MAX_PREMIUM = 5.0  # % above crossover price

# AO Confirmation signal
AO_CONFIRM_MACD_LOOKBACK = 7
AO_CONFIRM_MAX_PREMIUM = 8.0  # % above MACD cross price

# Stop loss
STOP_LOSS_PCT = 0.15  # 15% default
PROFIT_TARGET_MULT = 2.0  # 2x risk

# Market filter
MARKET_FILTER_SPY_SMA = 200
MARKET_FILTER_VIX_MAX = 30

# Data periods (standardized)
DAILY_PERIOD = '5y'      # Full history for chart viewing + indicator warmup
WEEKLY_PERIOD = '5y'     # ~260 bars
MONTHLY_PERIOD = '10y'   # ~120 bars


# =============================================================================
# DATA NORMALIZATION — Handle yfinance quirks in ONE place
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names from yfinance.
    Handles both MultiIndex and case variations.
    Returns DataFrame with standard column names: Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Handle MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names to Title Case
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == 'open':
            col_map[col] = 'Open'
        elif lower == 'high':
            col_map[col] = 'High'
        elif lower == 'low':
            col_map[col] = 'Low'
        elif lower == 'close':
            col_map[col] = 'Close'
        elif lower == 'volume':
            col_map[col] = 'Volume'
        elif lower == 'adj close':
            col_map[col] = 'Adj Close'

    if col_map:
        df = df.rename(columns=col_map)

    return df


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_macd(df: pd.DataFrame,
                   fast: int = MACD_FAST,
                   slow: int = MACD_SLOW,
                   signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """
    Calculate MACD indicator.

    Matches TradingView AO+MACD overlay Pine Script:
        fastMA = ema(src, fastLength)
        slowMA = ema(src, slowLength)
        macd = fastMA - slowMA
        signal = sma(macd, signalLength)   <-- SMA, not EMA!

    Adds columns: MACD, MACD_Signal, MACD_Hist
    """
    df = df.copy()
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].rolling(window=signal).mean()  # SMA
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_ao(df: pd.DataFrame,
                 fast: int = AO_FAST,
                 slow: int = AO_SLOW) -> pd.DataFrame:
    """
    Calculate Awesome Oscillator.

    Matches TradingView AO+MACD overlay Pine Script:
        ao = (sma(hl2, 5) - sma(hl2, 34)) / 2

    Adds column: AO
    """
    df = df.copy()
    median_price = (df['High'] + df['Low']) / 2
    df['AO'] = (median_price.rolling(window=fast).mean()
                - median_price.rolling(window=slow).mean()) / 2
    return df


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Calculate Average True Range.

    Adds column: ATR
    """
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(period).mean()
    return df


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all standard indicators to a DataFrame in one call.
    Adds: MACD, MACD_Signal, MACD_Hist, AO, ATR, SMA_50, SMA_200, SMA_150
    """
    df = normalize_columns(df)
    df = calculate_macd(df)
    df = calculate_ao(df)
    df = calculate_atr(df)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['SMA_200'] = calculate_sma(df['Close'], 200)
    df['SMA_150'] = calculate_sma(df['Close'], 150)  # ~30 week SMA for Weinstein
    return df


# =============================================================================
# MACD SIGNAL DETECTION
# =============================================================================

def detect_macd_cross(df: pd.DataFrame, bar_index: int = -1) -> Dict[str, Any]:
    """
    Detect MACD crossover state at a given bar.

    Returns dict with:
        - cross_today: bool — exact crossover on this bar
        - cross_recent: bool — crossover within MACD_CROSS_LOOKBACK bars AND still bullish
        - cross_bars_ago: int — how many bars since the cross (0 = today)
        - bullish: bool — MACD > Signal right now
        - bearish_cross: bool — MACD just crossed below signal
        - histogram: float — current histogram value
        - weakening: bool — histogram shrinking while still bullish
        - near_cross: bool — histogram tiny relative to MACD
    """
    if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
        df = calculate_macd(df)

    i = len(df) + bar_index if bar_index < 0 else bar_index

    if i < 2:
        return _empty_macd_result()

    macd = float(df['MACD'].iloc[i])
    signal = float(df['MACD_Signal'].iloc[i])
    prev_macd = float(df['MACD'].iloc[i - 1])
    prev_signal = float(df['MACD_Signal'].iloc[i - 1])
    hist = macd - signal

    # Skip if NaN (insufficient data for SMA warmup)
    if pd.isna(macd) or pd.isna(signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
        return _empty_macd_result()

    # Cross today
    cross_today = (macd > signal) and (prev_macd <= prev_signal)
    bearish_cross = (macd < signal) and (prev_macd >= prev_signal)

    # Recent cross (within lookback, must still be bullish now)
    cross_recent = False
    cross_bars_ago = 0

    for lookback in range(min(MACD_CROSS_LOOKBACK, i)):
        ci = i - lookback
        if ci < 1:
            break
        cm = float(df['MACD'].iloc[ci])
        cs = float(df['MACD_Signal'].iloc[ci])
        pm = float(df['MACD'].iloc[ci - 1])
        ps = float(df['MACD_Signal'].iloc[ci - 1])

        if pd.isna(cm) or pd.isna(cs) or pd.isna(pm) or pd.isna(ps):
            continue

        if (cm > cs) and (pm <= ps):
            cross_recent = True
            cross_bars_ago = lookback
            break

    # Must still be bullish for recent cross to count
    bullish = macd > signal
    cross_recent = cross_recent and bullish

    # Histogram trend (weakening detection)
    weakening = False
    near_cross = False

    if i >= 3:
        h1 = float(df['MACD'].iloc[i - 1]) - float(df['MACD_Signal'].iloc[i - 1])
        h2 = float(df['MACD'].iloc[i - 2]) - float(df['MACD_Signal'].iloc[i - 2])

        if not (pd.isna(h1) or pd.isna(h2)):
            weakening = bullish and (hist < h1) and (h1 < h2)
            near_cross = abs(hist) < abs(macd) * 0.1 if macd != 0 else False

    return {
        'cross_today': cross_today,
        'cross_recent': cross_recent,
        'cross_bars_ago': cross_bars_ago,
        'bullish': bullish,
        'bearish_cross': bearish_cross,
        'macd': round(macd, 4),
        'signal': round(signal, 4),
        'histogram': round(hist, 4),
        'weakening': weakening,
        'near_cross': near_cross,
    }


def _empty_macd_result() -> Dict[str, Any]:
    """Default empty MACD result when data insufficient."""
    return {
        'cross_today': False,
        'cross_recent': False,
        'cross_bars_ago': 0,
        'bullish': False,
        'bearish_cross': False,
        'macd': 0.0,
        'signal': 0.0,
        'histogram': 0.0,
        'weakening': False,
        'near_cross': False,
    }


# =============================================================================
# AO SIGNAL DETECTION
# =============================================================================

def detect_ao_state(df: pd.DataFrame, bar_index: int = -1,
                    entry_window: int = ENTRY_WINDOW) -> Dict[str, Any]:
    """
    Detect AO state at a given bar.

    Returns dict with:
        - positive: bool — AO > 0
        - value: float — current AO value
        - zero_cross_found: bool — AO crossed from ≤0 to >0 in prior entry_window bars
        - zero_cross_date: str — date of that cross
        - zero_cross_days_ago: int — bars since that cross
        - cross_today: bool — AO crossed zero today
        - trend: str — 'rising', 'falling', 'flat'
    """
    if 'AO' not in df.columns:
        df = calculate_ao(df)

    i = len(df) + bar_index if bar_index < 0 else bar_index

    if i < 2:
        return _empty_ao_result()

    ao = float(df['AO'].iloc[i])
    ao_prev = float(df['AO'].iloc[i - 1])

    if pd.isna(ao):
        return _empty_ao_result()

    positive = ao > 0
    cross_today = (ao > 0) and (ao_prev <= 0)

    # Look backwards for zero-cross in entry window
    zero_cross_found = False
    zero_cross_date = None
    zero_cross_days_ago = None

    for j in range(1, min(entry_window + 1, i)):
        past_idx = i - j
        if past_idx < 1:
            break
        ao_before = float(df['AO'].iloc[past_idx - 1])
        ao_after = float(df['AO'].iloc[past_idx])

        if pd.isna(ao_before) or pd.isna(ao_after):
            continue

        if ao_before <= 0 and ao_after > 0:
            zero_cross_found = True
            idx_val = df.index[past_idx]
            zero_cross_date = (idx_val.strftime('%Y-%m-%d')
                               if hasattr(idx_val, 'strftime') else str(idx_val))
            zero_cross_days_ago = j
            break

    # Trend
    if i >= 3:
        ao_prev2 = float(df['AO'].iloc[i - 2])
        if not pd.isna(ao_prev2):
            if ao > ao_prev > ao_prev2:
                trend = 'rising'
            elif ao < ao_prev < ao_prev2:
                trend = 'falling'
            else:
                trend = 'flat'
        else:
            trend = 'flat'
    else:
        trend = 'flat'

    return {
        'positive': positive,
        'value': round(ao, 4),
        'zero_cross_found': zero_cross_found,
        'zero_cross_date': zero_cross_date,
        'zero_cross_days_ago': zero_cross_days_ago,
        'cross_today': cross_today,
        'trend': trend,
    }


def _empty_ao_result() -> Dict[str, Any]:
    """Default empty AO result."""
    return {
        'positive': False,
        'value': 0.0,
        'zero_cross_found': False,
        'zero_cross_date': None,
        'zero_cross_days_ago': None,
        'cross_today': False,
        'trend': 'flat',
    }


# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================

def detect_bearish_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect bearish divergence using AO wave structure.

    The pattern:
    1. AO is positive (Wave 3 momentum building)
    2. AO crosses below zero (Wave 3 correction / Wave 4)
    3. AO returns positive again (Wave 5 begins)
    4. This new positive AO peak is SMALLER than the previous positive peak
       while price is HIGHER → bearish divergence
    5. Mark divergence at the highest price during the smaller AO block
    6. Divergence confirmed once AO drops below zero again (correction starts)

    Adds columns: bearish_div_detected, bearish_div_active
    Also stores divergence line data in df.attrs['divergence_lines'] for chart drawing.
    """
    if df is None or len(df) < 50:
        df = df.copy() if df is not None else pd.DataFrame()
        df['bearish_div_detected'] = False
        df['bearish_div_active'] = False
        df.attrs['divergence_lines'] = []
        return df

    df = df.copy()

    if 'AO' not in df.columns:
        df = calculate_ao(df)

    df['bearish_div_detected'] = False
    df['bearish_div_active'] = False

    ao = df['AO'].values
    highs = df['High'].values

    # Step 1: Identify AO positive blocks (continuous runs above zero)
    # Each block = (start_idx, end_idx, peak_ao, peak_ao_idx, highest_price, highest_price_idx)
    positive_blocks = []
    in_block = False
    block_start = 0
    block_peak_ao = 0.0
    block_peak_ao_idx = 0
    block_high_price = 0.0
    block_high_idx = 0

    for i in range(len(df)):
        if pd.isna(ao[i]):
            if in_block:
                if block_peak_ao > 0:
                    positive_blocks.append((block_start, i - 1, block_peak_ao,
                                            block_peak_ao_idx, block_high_price, block_high_idx))
                in_block = False
            continue

        if ao[i] > 0:
            if not in_block:
                in_block = True
                block_start = i
                block_peak_ao = ao[i]
                block_peak_ao_idx = i
                block_high_price = highs[i]
                block_high_idx = i
            else:
                if ao[i] > block_peak_ao:
                    block_peak_ao = ao[i]
                    block_peak_ao_idx = i
                if highs[i] > block_high_price:
                    block_high_price = highs[i]
                    block_high_idx = i
        else:
            if in_block:
                if block_peak_ao > 0:
                    positive_blocks.append((block_start, i - 1, block_peak_ao,
                                            block_peak_ao_idx, block_high_price, block_high_idx))
                in_block = False

    # Capture final block if still in one
    if in_block and block_peak_ao > 0:
        positive_blocks.append((block_start, len(df) - 1, block_peak_ao,
                                block_peak_ao_idx, block_high_price, block_high_idx))

    # Step 2: Compare consecutive positive blocks separated by negative zone
    divergence_lines = []

    for b_idx in range(1, len(positive_blocks)):
        prev_start, prev_end, prev_peak_ao, prev_ao_idx, prev_high_price, prev_high_idx = positive_blocks[b_idx - 1]
        curr_start, curr_end, curr_peak_ao, curr_ao_idx, curr_high_price, curr_high_idx = positive_blocks[b_idx]

        # Verify there's a negative zone between them (Wave 4)
        gap_has_negative = False
        for j in range(prev_end + 1, curr_start):
            if not pd.isna(ao[j]) and ao[j] < 0:
                gap_has_negative = True
                break

        if not gap_has_negative:
            continue

        # Bearish divergence: price higher high BUT AO smaller peak
        if curr_high_price > prev_high_price and curr_peak_ao < prev_peak_ao:
            # Mark at the highest price bar in the smaller AO block
            df.iloc[curr_high_idx, df.columns.get_loc('bearish_div_detected')] = True

            # Store line coordinates for chart drawing
            # Line 1: Price panel — connects W3 high to W5 high (rising)
            # Line 2: AO panel — connects W3 AO peak to W5 AO peak (falling)
            divergence_lines.append({
                "price_line": {
                    "x0": df.index[prev_high_idx].strftime('%Y-%m-%d'),
                    "y0": round(float(prev_high_price), 2),
                    "x1": df.index[curr_high_idx].strftime('%Y-%m-%d'),
                    "y1": round(float(curr_high_price), 2),
                },
                "ao_line": {
                    "x0": df.index[prev_ao_idx].strftime('%Y-%m-%d'),
                    "y0": round(float(prev_peak_ao), 4),
                    "x1": df.index[curr_ao_idx].strftime('%Y-%m-%d'),
                    "y1": round(float(curr_peak_ao), 4),
                },
                "w3_label": {
                    "date": df.index[prev_high_idx].strftime('%Y-%m-%d'),
                    "price": round(float(prev_high_price), 2),
                },
                "w5_label": {
                    "date": df.index[curr_high_idx].strftime('%Y-%m-%d'),
                    "price": round(float(curr_high_price), 2),
                },
            })

    df.attrs['divergence_lines'] = divergence_lines

    # Step 3: Set active flag — active from detection until AO goes negative again
    div_active = False
    for i in range(len(df)):
        if df.iloc[i]['bearish_div_detected']:
            div_active = True

        if div_active and not pd.isna(ao[i]) and ao[i] < 0:
            div_active = False

        df.iloc[i, df.columns.get_loc('bearish_div_active')] = div_active

    return df


# =============================================================================
# MULTI-TIMEFRAME CONFIRMATION
# =============================================================================

def check_timeframe_macd(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check MACD status on a given timeframe DataFrame (weekly or monthly).

    Returns dict with:
        - bullish: bool — MACD > Signal
        - macd: float
        - signal: float
        - histogram: float
        - cross_down: bool — just crossed bearish
        - cross_up: bool — just crossed bullish
    """
    if df is None or df.empty or len(df) < 30:
        return {
            'bullish': False, 'macd': None, 'signal': None,
            'histogram': None, 'cross_down': False, 'cross_up': False,
            'error': 'Insufficient data'
        }

    df = normalize_columns(df)
    df = calculate_macd(df)

    macd_val = float(df['MACD'].iloc[-1])
    signal_val = float(df['MACD_Signal'].iloc[-1])

    if pd.isna(macd_val) or pd.isna(signal_val):
        return {
            'bullish': False, 'macd': None, 'signal': None,
            'histogram': None, 'cross_down': False, 'cross_up': False,
            'error': 'NaN in MACD values'
        }

    bullish = macd_val > signal_val
    hist = macd_val - signal_val

    # Check for recent cross
    cross_down = False
    cross_up = False
    if len(df) >= 2:
        prev_m = float(df['MACD'].iloc[-2])
        prev_s = float(df['MACD_Signal'].iloc[-2])
        if not (pd.isna(prev_m) or pd.isna(prev_s)):
            cross_down = (macd_val < signal_val) and (prev_m >= prev_s)
            cross_up = (macd_val > signal_val) and (prev_m <= prev_s)

    return {
        'bullish': bullish,
        'macd': round(macd_val, 4),
        'signal': round(signal_val, 4),
        'histogram': round(hist, 4),
        'cross_down': cross_down,
        'cross_up': cross_up,
        'error': None
    }


def check_timeframe_ao(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check AO status on a given timeframe DataFrame (weekly or monthly).
    """
    if df is None or df.empty or len(df) < 35:
        return {'positive': False, 'value': None, 'error': 'Insufficient data'}

    df = normalize_columns(df)
    df = calculate_ao(df)

    ao_val = float(df['AO'].iloc[-1])

    if pd.isna(ao_val):
        return {'positive': False, 'value': None, 'error': 'NaN in AO'}

    return {
        'positive': ao_val > 0,
        'value': round(ao_val, 4),
        'error': None
    }


# =============================================================================
# WEINSTEIN STAGE DETECTION
# =============================================================================

def detect_weinstein_stage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Classify stock into Weinstein Stage (1-4) using 150-day SMA (~30 weeks).

    Stage 1 (Base): SMA flattening, price consolidating near SMA
    Stage 2 (Advance): SMA rising, price above SMA
    Stage 3 (Top): SMA flattening after rise, price rolling over
    Stage 4 (Decline): SMA falling, price below SMA

    Returns dict with stage number, label, SMA slope, and trend maturity.
    """
    if df is None or len(df) < 200:
        return {
            'stage': 0, 'label': 'Insufficient data',
            'sma150_slope': 'unknown', 'price_vs_sma150': 'unknown',
            'trend_maturity': 'unknown'
        }

    df = df.copy()
    if 'SMA_150' not in df.columns:
        df['SMA_150'] = calculate_sma(df['Close'], 150)

    price = float(df['Close'].iloc[-1])
    sma_now = float(df['SMA_150'].iloc[-1])
    sma_20ago = float(df['SMA_150'].iloc[-20]) if len(df) >= 20 else sma_now
    sma_50ago = float(df['SMA_150'].iloc[-50]) if len(df) >= 50 else sma_now

    if pd.isna(sma_now) or pd.isna(sma_20ago):
        return {
            'stage': 0, 'label': 'Insufficient data',
            'sma150_slope': 'unknown', 'price_vs_sma150': 'unknown',
            'trend_maturity': 'unknown'
        }

    # SMA slope (20-bar rate of change)
    sma_roc = (sma_now - sma_20ago) / sma_20ago * 100  # percentage change
    above_sma = price > sma_now
    pct_from_sma = (price - sma_now) / sma_now * 100

    # Classify slope
    if sma_roc > 1.0:
        slope = 'rising'
    elif sma_roc < -1.0:
        slope = 'falling'
    else:
        slope = 'flat'

    # 52-week position
    high_52w = float(df['High'].tail(252).max())
    low_52w = float(df['Low'].tail(252).min())
    range_52w = high_52w - low_52w
    pct_from_high = (price - high_52w) / high_52w * 100
    pct_from_low = (price - low_52w) / low_52w * 100

    # Stage classification
    if slope == 'rising' and above_sma:
        stage = 2
        label = 'Stage 2 — Advance'
        # Maturity: early if just broke above, late if extended
        if pct_from_sma < 5:
            maturity = 'early'
        elif pct_from_sma < 15:
            maturity = 'mid'
        elif pct_from_sma < 30:
            maturity = 'late'
        else:
            maturity = 'extended'
    elif slope == 'falling' and not above_sma:
        stage = 4
        label = 'Stage 4 — Decline'
        maturity = 'active'
    elif slope == 'flat' and abs(pct_from_sma) < 5:
        # Flat SMA, price near it — could be Stage 1 or 3
        # Differentiate: was it rising before (Stage 3) or falling before (Stage 1)?
        sma_longer_roc = (sma_now - sma_50ago) / sma_50ago * 100 if not pd.isna(sma_50ago) else 0
        if sma_longer_roc > 5:
            stage = 3
            label = 'Stage 3 — Top/Distribution'
            maturity = 'caution'
        else:
            stage = 1
            label = 'Stage 1 — Base/Accumulation'
            maturity = 'building'
    elif slope == 'rising' and not above_sma:
        # SMA still rising but price dipped below — pullback in Stage 2
        stage = 2
        label = 'Stage 2 — Pullback'
        maturity = 'pullback'
    elif slope == 'falling' and above_sma:
        # SMA falling but price above — rally in Stage 4 (bear market rally)
        stage = 4
        label = 'Stage 4 — Bear Rally'
        maturity = 'counter-trend'
    elif slope == 'flat' and above_sma:
        stage = 3
        label = 'Stage 3 — Topping'
        maturity = 'distribution'
    elif slope == 'flat' and not above_sma:
        stage = 1
        label = 'Stage 1 — Basing'
        maturity = 'accumulation'
    else:
        stage = 0
        label = 'Indeterminate'
        maturity = 'unknown'

    return {
        'stage': stage,
        'label': label,
        'sma150_slope': slope,
        'sma150_roc_pct': round(sma_roc, 2),
        'price_vs_sma150': 'above' if above_sma else 'below',
        'pct_from_sma150': round(pct_from_sma, 1),
        'pct_from_52w_high': round(pct_from_high, 1),
        'pct_from_52w_low': round(pct_from_low, 1),
        'trend_maturity': maturity,
    }


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def analyze_volume(df: pd.DataFrame, macd_cross_bar: int = None) -> Dict[str, Any]:
    """
    Analyze volume patterns.

    Returns:
        - avg_volume_50d: average daily volume over 50 days
        - cross_volume_ratio: volume on cross day vs 50d average (if cross provided)
        - accum_dist_trend: 'accumulating', 'distributing', or 'neutral'
        - volume_trend_20d: 'increasing', 'decreasing', or 'flat'
        - big_volume_days_20d: count of 2x+ average volume days in last 20
    """
    if df is None or len(df) < 50 or 'Volume' not in df.columns:
        return {
            'avg_volume_50d': 0, 'cross_volume_ratio': None,
            'accum_dist_trend': 'unknown', 'volume_trend_20d': 'unknown',
            'big_volume_days_20d': 0
        }

    vol = df['Volume']
    avg_50 = float(vol.tail(50).mean())

    # Cross volume ratio
    cross_vol_ratio = None
    if macd_cross_bar is not None and 0 <= macd_cross_bar < len(df):
        cross_vol = float(vol.iloc[macd_cross_bar])
        cross_vol_ratio = round(cross_vol / avg_50, 2) if avg_50 > 0 else None

    # Accumulation/Distribution: compare up-day volume vs down-day volume (last 20 bars)
    recent = df.tail(20)
    up_days = recent[recent['Close'] >= recent['Close'].shift(1)]
    down_days = recent[recent['Close'] < recent['Close'].shift(1)]
    up_vol = float(up_days['Volume'].sum()) if len(up_days) > 0 else 0
    down_vol = float(down_days['Volume'].sum()) if len(down_days) > 0 else 0

    if up_vol > down_vol * 1.3:
        ad_trend = 'accumulating'
    elif down_vol > up_vol * 1.3:
        ad_trend = 'distributing'
    else:
        ad_trend = 'neutral'

    # Volume trend (is volume increasing or drying up?)
    vol_first_10 = float(vol.tail(20).head(10).mean())
    vol_last_10 = float(vol.tail(10).mean())

    if vol_last_10 > vol_first_10 * 1.2:
        vol_trend = 'increasing'
    elif vol_last_10 < vol_first_10 * 0.8:
        vol_trend = 'decreasing'
    else:
        vol_trend = 'flat'

    # Big volume days
    big_days = int((vol.tail(20) > avg_50 * 2).sum())

    return {
        'avg_volume_50d': int(avg_50),
        'cross_volume_ratio': cross_vol_ratio,
        'accum_dist_trend': ad_trend,
        'volume_trend_20d': vol_trend,
        'big_volume_days_20d': big_days,
    }


# =============================================================================
# KEY LEVELS ANALYSIS
# =============================================================================

def analyze_key_levels(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze price relative to key technical levels.

    Returns price vs 50/200 SMA, golden/death cross status,
    and simple pivot-based support/resistance.
    """
    if df is None or len(df) < 200:
        return {
            'price': None, 'sma50': None, 'sma200': None,
            'price_vs_sma50': 'unknown', 'price_vs_sma200': 'unknown',
            'golden_cross': None, 'nearest_support': None,
            'nearest_resistance': None, 'at_key_level': False
        }

    if 'SMA_50' not in df.columns:
        df = df.copy()
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['SMA_200'] = calculate_sma(df['Close'], 200)

    price = float(df['Close'].iloc[-1])
    sma50 = float(df['SMA_50'].iloc[-1])
    sma200 = float(df['SMA_200'].iloc[-1])

    if pd.isna(sma50) or pd.isna(sma200):
        return {
            'price': price, 'sma50': None, 'sma200': None,
            'price_vs_sma50': 'unknown', 'price_vs_sma200': 'unknown',
            'golden_cross': None, 'nearest_support': None,
            'nearest_resistance': None, 'at_key_level': False
        }

    golden_cross = sma50 > sma200

    # Simple support/resistance from recent 60-day lows/highs
    recent = df.tail(60)
    swing_lows = recent['Low'].rolling(5, center=True).min()
    swing_highs = recent['High'].rolling(5, center=True).max()

    # Support: highest swing low below current price
    supports = swing_lows[swing_lows < price].dropna()
    nearest_support = float(supports.iloc[-1]) if len(supports) > 0 else None

    # Resistance: lowest swing high above current price
    resistances = swing_highs[swing_highs > price].dropna()
    nearest_resistance = float(resistances.iloc[-1]) if len(resistances) > 0 else None

    # Near a key level? (within 1% of SMA50, SMA200, support, or resistance)
    at_key = False
    for level in [sma50, sma200, nearest_support, nearest_resistance]:
        if level and abs(price - level) / price < 0.01:
            at_key = True
            break

    return {
        'price': round(price, 2),
        'sma50': round(sma50, 2),
        'sma200': round(sma200, 2),
        'price_vs_sma50': 'above' if price > sma50 else 'below',
        'price_vs_sma200': 'above' if price > sma200 else 'below',
        'pct_from_sma50': round((price - sma50) / sma50 * 100, 1),
        'pct_from_sma200': round((price - sma200) / sma200 * 100, 1),
        'golden_cross': golden_cross,
        'nearest_support': round(nearest_support, 2) if nearest_support else None,
        'nearest_resistance': round(nearest_resistance, 2) if nearest_resistance else None,
        'at_key_level': at_key,
    }


# =============================================================================
# OVERHEAD RESISTANCE ANALYSIS
# =============================================================================

def find_overhead_resistance(df: pd.DataFrame, num_levels: int = 5) -> Dict[str, Any]:
    """
    Identify major overhead resistance levels the stock must break through
    for a sustained uptrend.

    Combines four methods:
    1. Volume Profile — price zones where heaviest trading occurred above current price
       (overhead supply = trapped buyers who want to sell at breakeven)
    2. Prior Swing Highs — significant peaks where rallies previously failed
    3. Declining SMAs — moving averages sitting above price act as dynamic resistance
    4. Psychological levels — 52-week high, round numbers

    Returns dict with:
        - levels: list of resistance levels, each with price, type, strength, description
        - critical_level: the single most important level to break
        - breakout_volume_needed: estimated volume multiple needed for breakout
        - distance_to_critical: % from current price to critical level
        - assessment: plain text summary for AI context
    """
    if df is None or len(df) < 100:
        return {
            'levels': [],
            'critical_level': None,
            'breakout_volume_needed': None,
            'distance_to_critical': None,
            'assessment': 'Insufficient data for resistance analysis',
        }

    df = df.copy()
    price = float(df['Close'].iloc[-1])
    levels = []

    # ── 1. VOLUME PROFILE — find price zones with heavy volume above current price ──
    # Bucket the last 200 bars into price bins, sum volume in each bin
    lookback = min(200, len(df))
    recent = df.tail(lookback)

    price_range = float(recent['High'].max() - recent['Low'].min())
    if price_range > 0:
        num_bins = 30  # 30 price zones
        bin_size = price_range / num_bins
        bin_low = float(recent['Low'].min())

        vol_profile = {}
        for _, row in recent.iterrows():
            bar_mid = (float(row['High']) + float(row['Low'])) / 2
            bin_idx = int((bar_mid - bin_low) / bin_size)
            bin_idx = min(bin_idx, num_bins - 1)
            bin_price = bin_low + (bin_idx + 0.5) * bin_size
            vol_profile[round(bin_price, 2)] = (
                vol_profile.get(round(bin_price, 2), 0) + float(row['Volume'])
            )

        # Find volume nodes ABOVE current price
        avg_bin_vol = np.mean(list(vol_profile.values())) if vol_profile else 0
        for vp_price, vol in sorted(vol_profile.items()):
            if vp_price > price * 1.005:  # At least 0.5% above
                strength = vol / avg_bin_vol if avg_bin_vol > 0 else 0
                if strength > 1.3:  # Must be above average to be significant
                    strength_label = 'strong' if strength > 2.0 else 'moderate'
                    total_vol_m = vol / 1_000_000
                    levels.append({
                        'price': vp_price,
                        'type': 'volume_node',
                        'strength': round(strength, 1),
                        'strength_label': strength_label,
                        'description': f"${vp_price:.0f} — Volume node ({total_vol_m:.0f}M shares, {strength:.1f}x avg)",
                    })

    # ── 2. PRIOR SWING HIGHS — peaks where price reversed ──
    # Use a 10-bar window to find local maxima in the last 120 bars
    swing_lookback = min(120, len(df))
    swing_data = df.tail(swing_lookback)

    if len(swing_data) >= 10:
        highs = swing_data['High']
        for i in range(5, len(highs) - 5):
            window = highs.iloc[i - 5:i + 6]
            peak = float(highs.iloc[i])
            if peak == float(window.max()) and peak > price * 1.005:
                # Count how many times price tested this level (within 1%)
                tests = 0
                for j in range(len(swing_data)):
                    if abs(float(swing_data['High'].iloc[j]) - peak) / peak < 0.01:
                        tests += 1

                if tests >= 2:  # Tested at least twice = real resistance
                    strength = min(tests, 5)
                    # Check if this level is near an existing level (merge within 1.5%)
                    merged = False
                    for existing in levels:
                        if abs(existing['price'] - peak) / peak < 0.015:
                            # Merge — take the stronger one
                            if strength > existing.get('strength', 0):
                                existing['price'] = peak
                                existing['strength'] = strength
                                existing['description'] = f"${peak:.2f} — Swing high (tested {tests}x)"
                            existing['type'] = 'confluence'  # Volume + swing = confluence
                            merged = True
                            break
                    if not merged:
                        levels.append({
                            'price': round(peak, 2),
                            'type': 'swing_high',
                            'strength': strength,
                            'strength_label': 'strong' if tests >= 3 else 'moderate',
                            'description': f"${peak:.2f} — Swing high (tested {tests}x)",
                        })

    # ── 3. DECLINING SMAs OVERHEAD — dynamic resistance ──
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['SMA_200'] = calculate_sma(df['Close'], 200)
    if 'SMA_150' not in df.columns:
        df['SMA_150'] = calculate_sma(df['Close'], 150)

    for sma_name, col in [('50-day SMA', 'SMA_50'), ('150-day SMA', 'SMA_150'),
                           ('200-day SMA', 'SMA_200')]:
        sma_val = float(df[col].iloc[-1])
        if pd.isna(sma_val):
            continue
        if sma_val > price * 1.005:
            # Check if SMA is declining (acting as resistance, not support)
            sma_prev = float(df[col].iloc[-20]) if len(df) >= 20 else sma_val
            declining = sma_val < sma_prev if not pd.isna(sma_prev) else False
            flat_or_declining = sma_val <= sma_prev * 1.005 if not pd.isna(sma_prev) else False

            if flat_or_declining:
                pct_above = (sma_val - price) / price * 100
                strength = 3.0 if declining else 2.0  # Declining SMA = stronger resistance
                levels.append({
                    'price': round(sma_val, 2),
                    'type': 'declining_sma',
                    'strength': strength,
                    'strength_label': 'strong' if declining else 'moderate',
                    'description': f"${sma_val:.2f} — {sma_name} {'declining ↘' if declining else 'flat →'} ({pct_above:.1f}% above)",
                })

    # ── 4. PSYCHOLOGICAL LEVELS — 52-week high, round numbers ──
    high_52w = float(df['High'].tail(252).max()) if len(df) >= 252 else float(df['High'].max())

    if high_52w > price * 1.005:
        pct_below = (high_52w - price) / price * 100
        levels.append({
            'price': round(high_52w, 2),
            'type': '52w_high',
            'strength': 4.0,  # Always significant
            'strength_label': 'strong',
            'description': f"${high_52w:.2f} — 52-week high ({pct_below:.1f}% above)",
        })

    # Round numbers (nearest $10, $25, $50, $100 above price)
    for increment in [10, 25, 50, 100]:
        round_level = float(np.ceil(price / increment) * increment)
        if round_level > price * 1.005 and round_level < price * 1.30:
            # Only add if not too close to existing level
            too_close = any(abs(l['price'] - round_level) / round_level < 0.01 for l in levels)
            if not too_close:
                levels.append({
                    'price': round_level,
                    'type': 'round_number',
                    'strength': 1.5,
                    'strength_label': 'moderate',
                    'description': f"${round_level:.0f} — Round number psychological level",
                })

    # ── SORT & RANK — closest first, with strength weighting ──
    # Remove duplicates within 1% of each other, keeping strongest
    filtered = []
    for level in sorted(levels, key=lambda x: x['price']):
        merged = False
        for existing in filtered:
            if abs(existing['price'] - level['price']) / level['price'] < 0.01:
                if level['strength'] > existing['strength']:
                    existing.update(level)
                    existing['type'] = 'confluence'
                    existing['strength'] = max(existing['strength'], level['strength'])
                merged = True
                break
        if not merged:
            filtered.append(level)

    # Sort by proximity to current price
    filtered.sort(key=lambda x: x['price'])

    # Take top N levels
    top_levels = filtered[:num_levels]

    # ── CRITICAL LEVEL — the most important one to break ──
    # Prioritize: closest + highest strength
    critical = None
    if top_levels:
        # Score: strength * proximity_bonus (closer = more important)
        for lev in top_levels:
            dist = (lev['price'] - price) / price * 100
            # Closer levels weighted more heavily, but very close ones (< 1%) less so
            proximity_score = max(0, 10 - dist) if dist < 10 else 1
            lev['_score'] = lev['strength'] * proximity_score

        critical = max(top_levels, key=lambda x: x.get('_score', 0))

        # Clean up internal score
        for lev in top_levels:
            lev.pop('_score', None)

    # ── BREAKOUT VOLUME ESTIMATE ──
    # If critical level is a volume node, need proportionally more volume to absorb supply
    breakout_vol = None
    if critical:
        if critical['type'] in ['volume_node', 'confluence']:
            breakout_vol = round(max(1.5, critical['strength'] * 0.8), 1)
        elif critical['type'] == 'declining_sma':
            breakout_vol = 1.5
        else:
            breakout_vol = 1.3  # Default: need at least 1.3x avg volume

    # ── DISTANCE TO CRITICAL ──
    dist_pct = None
    if critical:
        dist_pct = round((critical['price'] - price) / price * 100, 1)

    # ── ASSESSMENT TEXT ──
    if not top_levels:
        assessment = "No significant overhead resistance detected — clear path higher."
    elif critical and dist_pct is not None:
        num_above = len(top_levels)
        if dist_pct < 3:
            assessment = (f"Critical resistance at ${critical['price']:.2f} ({dist_pct:.1f}% above) — "
                         f"{critical['description']}. "
                         f"{num_above} resistance level{'s' if num_above > 1 else ''} overhead. "
                         f"Breakout needs {breakout_vol:.1f}x average volume to stick.")
        elif dist_pct < 8:
            assessment = (f"Room to run before hitting resistance at ${critical['price']:.2f} "
                         f"({dist_pct:.1f}% above). {num_above} level{'s' if num_above > 1 else ''} overhead.")
        else:
            assessment = (f"Significant resistance at ${critical['price']:.2f} ({dist_pct:.1f}% above) "
                         f"but has room to move. {num_above} level{'s' if num_above > 1 else ''} total overhead.")
    else:
        assessment = f"{len(top_levels)} resistance level{'s' if len(top_levels) > 1 else ''} identified overhead."

    return {
        'levels': top_levels,
        'critical_level': {
            'price': critical['price'],
            'type': critical['type'],
            'description': critical['description'],
        } if critical else None,
        'breakout_volume_needed': breakout_vol,
        'distance_to_critical_pct': dist_pct,
        'assessment': assessment,
    }


# =============================================================================
# RELATIVE STRENGTH
# =============================================================================

def analyze_relative_strength(df: pd.DataFrame, spy_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate relative strength vs SPY over multiple periods.

    Both DataFrames should be daily with 'Close' column.
    """
    if df is None or spy_df is None or len(df) < 252 or len(spy_df) < 252:
        return {
            'rs_1mo': None, 'rs_3mo': None, 'rs_6mo': None, 'rs_12mo': None,
            'rs_trend': 'unknown'
        }

    price = df['Close']
    spy = spy_df['Close']

    # Align on dates
    common = price.index.intersection(spy.index)
    if len(common) < 252:
        return {
            'rs_1mo': None, 'rs_3mo': None, 'rs_6mo': None, 'rs_12mo': None,
            'rs_trend': 'unknown'
        }

    price = price.reindex(common)
    spy = spy.reindex(common)

    def period_rs(n_bars):
        if len(price) < n_bars:
            return None
        stock_ret = (float(price.iloc[-1]) / float(price.iloc[-n_bars]) - 1) * 100
        spy_ret = (float(spy.iloc[-1]) / float(spy.iloc[-n_bars]) - 1) * 100
        return round(stock_ret - spy_ret, 1)

    rs_1mo = period_rs(21)
    rs_3mo = period_rs(63)
    rs_6mo = period_rs(126)
    rs_12mo = period_rs(252)

    # RS trend: is short-term RS improving or deteriorating vs longer-term?
    if rs_1mo is not None and rs_3mo is not None:
        if rs_1mo > rs_3mo:
            trend = 'improving'
        elif rs_1mo < rs_3mo - 5:
            trend = 'deteriorating'
        else:
            trend = 'stable'
    else:
        trend = 'unknown'

    return {
        'rs_1mo': rs_1mo,
        'rs_3mo': rs_3mo,
        'rs_6mo': rs_6mo,
        'rs_12mo': rs_12mo,
        'rs_trend': trend,
    }


# =============================================================================
# STOP LOSS & POSITION SIZING
# =============================================================================

def calculate_stops(entry_price: float, atr: float = None) -> Dict[str, Any]:
    """
    Calculate stop loss and profit target.

    Uses 15% stop loss or 2x ATR, whichever is tighter.
    Profit target = 2x risk (reward:risk = 2:1).
    """
    # Percentage stop
    pct_stop = entry_price * (1 - STOP_LOSS_PCT)

    # ATR stop (2x ATR below entry)
    atr_stop = entry_price - (2 * atr) if atr else None

    # Use the tighter (higher) stop
    if atr_stop and atr_stop > pct_stop:
        stop = round(atr_stop, 2)
        stop_method = f"ATR-based ({2}x ATR)"
    else:
        stop = round(pct_stop, 2)
        stop_method = f"Percentage ({STOP_LOSS_PCT:.0%})"

    risk = entry_price - stop
    target = round(entry_price + (risk * PROFIT_TARGET_MULT), 2)
    risk_pct = round(risk / entry_price * 100, 1)
    rr_ratio = round(PROFIT_TARGET_MULT, 1)

    return {
        'entry': round(entry_price, 2),
        'stop': stop,
        'stop_method': stop_method,
        'target': target,
        'risk_per_share': round(risk, 2),
        'risk_pct': risk_pct,
        'reward_risk': f"1:{rr_ratio}",
    }


def calculate_position_size(entry_price: float, stop_price: float,
                            account_size: float = 100000,
                            risk_pct: float = 1.0) -> Dict[str, Any]:
    """
    Calculate position size based on account risk.

    Default: risk 1% of $100k account = $1,000 max loss.
    """
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return {'shares': 0, 'position_value': 0, 'dollar_risk': 0}

    dollar_risk = account_size * (risk_pct / 100)
    shares = int(dollar_risk / risk_per_share)
    position_value = shares * entry_price

    return {
        'shares': shares,
        'position_value': round(position_value, 2),
        'dollar_risk': round(dollar_risk, 2),
        'pct_of_portfolio': round(position_value / account_size * 100, 1),
    }


# =============================================================================
# FULL ENTRY VALIDATION — Combines all checks
# =============================================================================

@dataclass
class EntrySignal:
    """Complete entry signal assessment."""
    ticker: str
    is_valid: bool = False
    is_valid_relaxed: bool = False

    # MACD state
    macd: Dict = field(default_factory=dict)

    # AO state
    ao: Dict = field(default_factory=dict)

    # Market filter
    market_filter: Dict = field(default_factory=dict)

    # Multi-timeframe
    weekly_macd: Dict = field(default_factory=dict)
    monthly_macd: Dict = field(default_factory=dict)
    monthly_ao: Dict = field(default_factory=dict)

    # Context (for AI analysis)
    weinstein: Dict = field(default_factory=dict)
    volume: Dict = field(default_factory=dict)
    key_levels: Dict = field(default_factory=dict)
    overhead_resistance: Dict = field(default_factory=dict)
    relative_strength: Dict = field(default_factory=dict)

    # Stops
    stops: Dict = field(default_factory=dict)

    # Debug
    data_bars: int = 0
    last_bar_date: str = ''
    error: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility."""
        return {
            'ticker': self.ticker,
            'is_valid': self.is_valid,
            'is_valid_relaxed': self.is_valid_relaxed,
            'macd': self.macd,
            'ao': self.ao,
            'market_filter': self.market_filter,
            'weekly_macd': self.weekly_macd,
            'monthly_macd': self.monthly_macd,
            'monthly_ao': self.monthly_ao,
            'weinstein': self.weinstein,
            'volume': self.volume,
            'key_levels': self.key_levels,
            'overhead_resistance': self.overhead_resistance,
            'relative_strength': self.relative_strength,
            'stops': self.stops,
            'data_bars': self.data_bars,
            'last_bar_date': self.last_bar_date,
            'error': self.error,
        }


def validate_entry(daily_df: pd.DataFrame,
                   weekly_df: pd.DataFrame = None,
                   monthly_df: pd.DataFrame = None,
                   spy_df: pd.DataFrame = None,
                   ticker: str = '') -> EntrySignal:
    """
    Complete entry validation against TTA strategy rules.

    Takes pre-fetched DataFrames (data fetching is NOT this module's job).

    Checks:
    1. MACD cross (recent, within lookback)
    2. AO positive
    3. AO zero-cross in prior entry window
    4. Market filter (SPY > 200 SMA, VIX < 30) — requires spy_df
    5. Weekly MACD confirmation
    6. Monthly MACD top-down context
    """
    signal = EntrySignal(ticker=ticker)

    if daily_df is None or len(daily_df) < 50:
        signal.error = 'Insufficient daily data'
        return signal

    # Normalize and add indicators
    daily_df = add_all_indicators(daily_df)
    signal.data_bars = len(daily_df)

    last_idx = daily_df.index[-1]
    signal.last_bar_date = (last_idx.strftime('%Y-%m-%d')
                            if hasattr(last_idx, 'strftime') else str(last_idx))

    # ── MACD ──────────────────────────────────────────────────────────
    signal.macd = detect_macd_cross(daily_df)

    # ── AO ────────────────────────────────────────────────────────────
    signal.ao = detect_ao_state(daily_df)

    # ── Volume ────────────────────────────────────────────────────────
    cross_bar = None
    if signal.macd['cross_recent']:
        cross_bar = len(daily_df) - 1 - signal.macd['cross_bars_ago']
    signal.volume = analyze_volume(daily_df, macd_cross_bar=cross_bar)

    # ── Key Levels ────────────────────────────────────────────────────
    signal.key_levels = analyze_key_levels(daily_df)

    # ── Overhead Resistance ───────────────────────────────────────────
    signal.overhead_resistance = find_overhead_resistance(daily_df)

    # ── Weinstein Stage ───────────────────────────────────────────────
    signal.weinstein = detect_weinstein_stage(daily_df)

    # ── Relative Strength ─────────────────────────────────────────────
    if spy_df is not None:
        spy_norm = normalize_columns(spy_df)
        signal.relative_strength = analyze_relative_strength(daily_df, spy_norm)

    # ── Stops ─────────────────────────────────────────────────────────
    price = float(daily_df['Close'].iloc[-1])
    atr = float(daily_df['ATR'].iloc[-1]) if not pd.isna(daily_df['ATR'].iloc[-1]) else None
    signal.stops = calculate_stops(price, atr)

    # ── Weekly MACD ───────────────────────────────────────────────────
    if weekly_df is not None and len(weekly_df) >= 30:
        signal.weekly_macd = check_timeframe_macd(weekly_df)

    # ── Monthly MACD + AO ─────────────────────────────────────────────
    if monthly_df is not None and len(monthly_df) >= 30:
        signal.monthly_macd = check_timeframe_macd(monthly_df)
        signal.monthly_ao = check_timeframe_ao(monthly_df)

    # ── Market Filter ─────────────────────────────────────────────────
    if spy_df is not None:
        spy_norm = normalize_columns(spy_df)
        spy_close = float(spy_norm['Close'].iloc[-1])
        spy_sma200 = float(calculate_sma(spy_norm['Close'], 200).iloc[-1])
        spy_above = spy_close > spy_sma200 if not pd.isna(spy_sma200) else True

        signal.market_filter = {
            'spy_above_200': spy_above,
            'spy_close': round(spy_close, 2),
            'spy_sma200': round(spy_sma200, 2) if not pd.isna(spy_sma200) else None,
            # VIX needs separate fetch — handled by data_fetcher
            'vix_below_30': signal.market_filter.get('vix_below_30', True),
            'vix_close': signal.market_filter.get('vix_close', None),
        }

    # ── FINAL DETERMINATION ───────────────────────────────────────────
    spy_ok = signal.market_filter.get('spy_above_200', True)
    vix_ok = signal.market_filter.get('vix_below_30', True)

    signal.is_valid = all([
        signal.macd['cross_recent'],
        signal.ao['positive'],
        signal.ao['zero_cross_found'],
        spy_ok,
        vix_ok,
    ])

    signal.is_valid_relaxed = all([
        signal.macd['bullish'],
        signal.ao['positive'],
        signal.ao['zero_cross_found'],
        spy_ok,
        vix_ok,
    ])

    return signal
