"""
TTA v2 Chart Engine — TradingView Lightweight Charts v5
=========================================================

Multi-pane financial charts with native TradingView behavior:
zoom, pan, crosshair, auto-scaling y-axis.

v3.2: Price marker, divergence markers, tight margins,
      MTF with LWC, timeframe support, proper labels.

Version: 3.2.0 (2026-02-08)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from signal_engine import (
    normalize_columns, calculate_macd, calculate_ao,
    add_all_indicators, detect_bearish_divergence,
    EntrySignal, get_macd_indicator_label,
)

# =============================================================================
# COLORS
# =============================================================================

COLOR_BULL = 'rgba(38,166,154,0.9)'
COLOR_BEAR = 'rgba(239,83,80,0.9)'
COLOR_BULL_LIGHT = 'rgba(38,166,154,0.3)'
COLOR_BEAR_LIGHT = 'rgba(239,83,80,0.3)'

# Dark theme shared across all panes
def _theme():
    return {
        "layout": {
            "background": {"color": "#131722"},
            "textColor": "#d1d4dc",
            "fontSize": 12,
            "panes": {
                "separatorColor": "rgba(42, 46, 57, 0.8)",
                "separatorHoverColor": "rgba(100, 100, 100, 0.5)",
                "enableResize": True,
            },
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.3)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.3)"},
        },
        "crosshair": {"mode": 0},
        "rightPriceScale": {
            "borderColor": "rgba(197, 203, 206, 0.3)",
            "scaleMargins": {"top": 0.02, "bottom": 0.02},
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.3)",
            "timeVisible": False,
            "barSpacing": 6,
            "minBarSpacing": 2,
            "rightOffset": 5,
        },
    }


# =============================================================================
# DATA FORMATTERS
# =============================================================================

def _candles(df):
    return [{'time': idx.strftime('%Y-%m-%d'),
             'open': round(float(r['Open']), 2),
             'high': round(float(r['High']), 2),
             'low': round(float(r['Low']), 2),
             'close': round(float(r['Close']), 2)}
            for idx, r in df.iterrows()]


def _line(df, col):
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'), 'value': round(float(v), 4)}
            for idx, v in s.items()]


def _hist(df, col, pos=COLOR_BULL, neg=COLOR_BEAR):
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'),
             'value': round(float(v), 4),
             'color': pos if v >= 0 else neg}
            for idx, v in s.items()]


def _vol(df):
    return [{'time': idx.strftime('%Y-%m-%d'),
             'value': float(r['Volume']),
             'color': COLOR_BULL_LIGHT if r['Close'] >= r['Open'] else COLOR_BEAR_LIGHT}
            for idx, r in df.iterrows()]


# =============================================================================
# DIVERGENCE + SIGNAL MARKERS — shown on candlestick series
# =============================================================================

def _divergence_markers(df):
    """Build LWC marker objects for bearish divergence points (W5 labels)."""
    if 'bearish_div_detected' not in df.columns:
        return []
    divs = df[df['bearish_div_detected'] == True]
    markers = []
    for idx, row in divs.iterrows():
        markers.append({
            "time": idx.strftime('%Y-%m-%d'),
            "position": "aboveBar",
            "color": "#f87171",
            "shape": "arrowDown",
            "text": "W5(div)",
        })
    return markers


def _wave_markers(df):
    """Build wave label markers (W3) from divergence line data."""
    div_lines = df.attrs.get('divergence_lines', [])
    markers = []
    for dl in div_lines:
        w3 = dl.get('w3_label', {})
        if w3.get('date'):
            markers.append({
                "time": w3['date'],
                "position": "aboveBar",
                "color": "#4ade80",
                "shape": "arrowDown",
                "text": "W3",
            })
    return markers


def _signal_markers(df):
    """
    Build QUALIFIED buy/sell markers on price chart.

    BUY: MACD crosses from below zero to above zero + AO confirms positive
    SELL: MACD crosses from above zero to below zero + AO confirms negative

    Filters:
    - MACD must have reached meaningful depth below/above zero (not just tickle it)
    - Minimum 15 bars between signals to avoid chop clusters
    """
    if 'MACD' not in df.columns:
        return []

    macd_vals = df['MACD'].values
    ao = df['AO'].values if 'AO' in df.columns else [0] * len(df)
    close = df['Close'].values

    # Compute a dynamic threshold: ignore MACD zero-crosses if MACD never
    # went more than 10% of its recent range below/above zero
    # This filters out the tiny oscillations around zero in choppy markets
    valid_macd = [v for v in macd_vals if not pd.isna(v)]
    if len(valid_macd) < 30:
        return []
    macd_range = max(abs(max(valid_macd)), abs(min(valid_macd)))
    min_depth = macd_range * 0.08  # must reach at least 8% of range beyond zero

    markers = []
    last_signal_bar = -30  # cooldown tracker

    # ── Scan for zero-cross events ──
    for i in range(1, len(df)):
        if pd.isna(macd_vals[i]) or pd.isna(macd_vals[i-1]):
            continue

        # Cooldown: skip if too close to last signal
        if i - last_signal_bar < 15:
            continue

        date_str = df.index[i].strftime('%Y-%m-%d')
        price = float(close[i])

        # BUY: MACD crosses from negative to positive
        if macd_vals[i] > 0 and macd_vals[i-1] <= 0:
            # Check MACD reached meaningful depth below zero before this cross
            min_macd_before = 0.0
            for j in range(max(0, i - 60), i):
                if not pd.isna(macd_vals[j]) and macd_vals[j] < min_macd_before:
                    min_macd_before = macd_vals[j]
                # Stop scanning once we hit the previous positive zone
                if not pd.isna(macd_vals[j]) and macd_vals[j] > 0 and j < i - 3:
                    min_macd_before = 0.0  # reset — only count depth from last negative stretch

            if abs(min_macd_before) < min_depth:
                continue  # Too shallow — skip

            # Check AO confirmation
            ao_ok = False
            for j in range(max(0, i - 3), min(len(df), i + 15)):
                if not pd.isna(ao[j]) and ao[j] > 0:
                    ao_ok = True
                    break
            if ao_ok:
                markers.append({
                    "time": date_str,
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "BUY",
                    "_price": price,
                })
                last_signal_bar = i

        # SELL: MACD crosses from positive to negative
        elif macd_vals[i] < 0 and macd_vals[i-1] >= 0:
            # Check MACD reached meaningful height above zero before this cross
            max_macd_before = 0.0
            for j in range(max(0, i - 60), i):
                if not pd.isna(macd_vals[j]) and macd_vals[j] > max_macd_before:
                    max_macd_before = macd_vals[j]
                if not pd.isna(macd_vals[j]) and macd_vals[j] < 0 and j < i - 3:
                    max_macd_before = 0.0

            if abs(max_macd_before) < min_depth:
                continue  # Too shallow

            # Check AO confirmation
            ao_ok = False
            for j in range(max(0, i - 3), min(len(df), i + 15)):
                if not pd.isna(ao[j]) and ao[j] < 0:
                    ao_ok = True
                    break
            if ao_ok:
                markers.append({
                    "time": date_str,
                    "position": "aboveBar",
                    "color": "#ef5350",
                    "shape": "arrowDown",
                    "text": "SELL",
                    "_price": price,
                })
                last_signal_bar = i

    # Add price to the most recent buy and sell only
    for idx in range(len(markers) - 1, -1, -1):
        if markers[idx]['text'] == 'BUY':
            markers[idx]['text'] = f"BUY ${markers[idx].pop('_price', 0):.0f}"
            break
    for idx in range(len(markers) - 1, -1, -1):
        if markers[idx]['text'] == 'SELL':
            markers[idx]['text'] = f"SELL ${markers[idx].pop('_price', 0):.0f}"
            break

    # Clean up _price from all markers
    for m in markers:
        m.pop('_price', None)

    return markers


def _crossover_dots(df):
    """
    Build green/red dot data for MACD crossover points on the MACD pane.
    Returns two lists: bullish dots and bearish dots (as line series data).
    """
    if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
        return [], []

    bull_dots = []
    bear_dots = []
    macd = df['MACD'].values
    sig = df['MACD_Signal'].values

    for i in range(1, len(df)):
        if pd.isna(macd[i]) or pd.isna(sig[i]) or pd.isna(macd[i-1]) or pd.isna(sig[i-1]):
            continue

        date_str = df.index[i].strftime('%Y-%m-%d')
        val = round(float(macd[i]), 4)

        if macd[i] > sig[i] and macd[i-1] <= sig[i-1]:
            bull_dots.append({"time": date_str, "value": val})
        elif macd[i] < sig[i] and macd[i-1] >= sig[i-1]:
            bear_dots.append({"time": date_str, "value": val})

    return bull_dots, bear_dots


def _divergence_line_series(df):
    """
    Build LWC line series for divergence lines on price and AO panes.
    Returns dict with 'price_lines' and 'ao_lines' lists of series configs.
    """
    div_lines = df.attrs.get('divergence_lines', [])
    price_line_series = []
    ao_line_series = []

    for i, dl in enumerate(div_lines):
        pl = dl.get('price_line', {})
        al = dl.get('ao_line', {})

        if pl.get('x0') and pl.get('x1'):
            price_line_series.append({
                "type": "Line",
                "data": [
                    {"time": pl['x0'], "value": pl['y0']},
                    {"time": pl['x1'], "value": pl['y1']},
                ],
                "options": {
                    "color": "#ef4444",
                    "lineWidth": 3,
                    "lineStyle": 0,
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                    "pointMarkersVisible": False,
                },
                "label": f"Div (price)" if i == 0 else "",
            })

        if al.get('x0') and al.get('x1'):
            ao_line_series.append({
                "type": "Line",
                "data": [
                    {"time": al['x0'], "value": al['y0']},
                    {"time": al['x1'], "value": al['y1']},
                ],
                "options": {
                    "color": "#ef4444",
                    "lineWidth": 3,
                    "lineStyle": 0,
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                    "pointMarkersVisible": False,
                },
                "label": f"Div (AO)" if i == 0 else "",
            })

    return {"price_lines": price_line_series, "ao_lines": ao_line_series}


# =============================================================================
# TRIGGER LEVELS (Buy/Sell price lines)
# =============================================================================

def _compute_trigger_levels(df, signal=None):
    """
    Compute Buy/Sell trigger levels from signal data or recent swing highs/lows.
    Returns dict with 'buy_level', 'sell_level', 'status_text', 'status_color'.
    """
    current_price = float(df['Close'].iloc[-1])

    # Try to get from signal's overhead resistance
    buy_level = None
    sell_level = None

    if signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        if levels:
            # First resistance above current price = buy level
            above = [l['price'] for l in levels if l['price'] > current_price]
            if above:
                buy_level = min(above)
            # Nearest below = sell level
            below = [l['price'] for l in levels if l['price'] < current_price]
            if below:
                sell_level = max(below)

    # Fallback: use recent 20-bar swing high/low
    if buy_level is None:
        buy_level = float(df['High'].tail(20).max())
    if sell_level is None:
        sell_level = float(df['Low'].tail(20).min())

    # Status text
    if current_price > buy_level:
        status = f"▲ ${current_price:.2f} > ${buy_level:.2f} = BUY"
        color = "#26a69a"
    elif current_price < sell_level:
        status = f"▼ ${current_price:.2f} < ${sell_level:.2f} = SELL"
        color = "#ef5350"
    else:
        status = f"${current_price:.2f} in range — WAIT"
        color = "#ffa726"

    return {
        "buy_level": buy_level,
        "sell_level": sell_level,
        "status_text": status,
        "status_color": color,
        "current_price": current_price,
    }


# =============================================================================
# BUILD MULTI-PANE CHART CONFIG
# =============================================================================

def build_lwc_charts(
    df: pd.DataFrame,
    ticker: str,
    signal: EntrySignal = None,
    show_volume: bool = True,
    show_resistance: bool = True,
    show_divergence: bool = True,
    total_height: int = 800,
    extra_markers: list = None,
    macd_profile: Optional[str] = None,
) -> list:
    """
    Build `charts` list for LWC v5.
    Each pane = separate dict with {chart, series, height, title}.
    Includes: divergence lines, MACD crossover dots, buy/sell markers,
    wave labels, and price status.
    """
    df = normalize_columns(df).copy()
    if macd_profile is not None or 'MACD' not in df.columns:
        df = add_all_indicators(df, macd_profile=macd_profile)
    if show_divergence:
        df = detect_bearish_divergence(df)

    has_vol = show_volume and 'Volume' in df.columns
    panes = []
    current_price = round(float(df['Close'].iloc[-1]), 2)

    # Compute trigger levels and status
    triggers = _compute_trigger_levels(df, signal)

    # Get divergence line series for price and AO panes
    div_series = _divergence_line_series(df) if show_divergence else {"price_lines": [], "ao_lines": []}

    # ── PANE 0: PRICE ────────────────────────────────────────────────
    price_series = []

    # Candlestick with ALL markers: divergence, wave labels, buy/sell signals
    candle_config = {
        "type": "Candlestick",
        "data": _candles(df),
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR,
            "lastValueVisible": True,
            "priceLineVisible": True,
            "priceLineColor": "#787b86",
            "priceLineStyle": 2,
        },
    }

    # Combine all markers on the candlestick
    all_markers = []
    if show_divergence:
        all_markers.extend(_divergence_markers(df))
        all_markers.extend(_wave_markers(df))
    # APEX markers (from apex_signals.py) replace the old TTA BUY/SELL
    if extra_markers:
        all_markers.extend(extra_markers)
    if all_markers:
        candle_config["markers"] = all_markers

    price_series.append(candle_config)

    # Moving averages
    sma_configs = [
        ('SMA_150', '#ff9800', 2, 1, '150d SMA'),
        ('SMA_50', '#42a5f5', 1, 2, '50 SMA'),
        ('SMA_200', '#ab47bc', 1, 2, '200 SMA'),
    ]

    for col, color, width, style, name in sma_configs:
        if col in df.columns:
            data = _line(df, col)
            if data:
                val = df[col].dropna()
                label_text = f"{name} ${float(val.iloc[-1]):.2f}" if len(val) > 0 else name
                price_series.append({
                    "type": "Line",
                    "data": data,
                    "options": {
                        "color": color,
                        "lineWidth": width,
                        "lineStyle": style,
                        "crosshairMarkerVisible": False,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                    },
                    "label": label_text,
                })

    # Resistance levels
    if show_resistance and signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        critical = signal.overhead_resistance.get('critical_level', {})
        for lev in levels:
            price = lev['price']
            is_crit = critical and abs(price - critical.get('price', 0)) < 0.01
            color = '#ff1744' if is_crit else 'rgba(239, 83, 80, 0.35)'
            res_data = [
                {'time': df.index[0].strftime('%Y-%m-%d'), 'value': round(price, 2)},
                {'time': df.index[-1].strftime('%Y-%m-%d'), 'value': round(price, 2)},
            ]
            price_series.append({
                "type": "Line",
                "data": res_data,
                "options": {
                    "color": color,
                    "lineWidth": 2 if is_crit else 1,
                    "lineStyle": 0 if is_crit else 2,
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": True,
                    "priceLineVisible": False,
                },
                "label": f"R ${price:.0f}" + (" ★" if is_crit else ""),
            })

    # Buy/Sell trigger level lines
    if triggers['buy_level']:
        bl = triggers['buy_level']
        price_series.append({
            "type": "Line",
            "data": [
                {'time': df.index[0].strftime('%Y-%m-%d'), 'value': round(bl, 2)},
                {'time': df.index[-1].strftime('%Y-%m-%d'), 'value': round(bl, 2)},
            ],
            "options": {
                "color": "rgba(38, 166, 154, 0.5)",
                "lineWidth": 1, "lineStyle": 2,
                "crosshairMarkerVisible": False,
                "lastValueVisible": True,
                "priceLineVisible": False,
            },
            "label": f"▲ BUY ${bl:.2f}",
        })

    if triggers['sell_level']:
        sl = triggers['sell_level']
        price_series.append({
            "type": "Line",
            "data": [
                {'time': df.index[0].strftime('%Y-%m-%d'), 'value': round(sl, 2)},
                {'time': df.index[-1].strftime('%Y-%m-%d'), 'value': round(sl, 2)},
            ],
            "options": {
                "color": "rgba(239, 83, 80, 0.5)",
                "lineWidth": 1, "lineStyle": 2,
                "crosshairMarkerVisible": False,
                "lastValueVisible": True,
                "priceLineVisible": False,
            },
            "label": f"▼ SELL ${sl:.2f}",
        })

    # Divergence lines on price pane
    for dls in div_series['price_lines']:
        price_series.append(dls)

    # Price pane
    price_theme = _theme()
    price_theme["rightPriceScale"]["scaleMargins"] = {"top": 0.02, "bottom": 0.02}

    # Title includes status
    status = triggers['status_text']
    ph = int(total_height * 0.55) if has_vol else int(total_height * 0.50)
    panes.append({
        "chart": price_theme,
        "series": price_series,
        "height": ph,
        "title": f"{ticker}  ${current_price}   |   {status}",
    })

    # ── PANE 1: VOLUME ───────────────────────────────────────────────
    if has_vol:
        panes.append({
            "chart": _theme(),
            "series": [{
                "type": "Histogram",
                "data": _vol(df),
                "options": {"priceFormat": {"type": "volume"}},
            }],
            "height": int(total_height * 0.10),
            "title": "Volume",
        })

    # ── PANE 2: AO ───────────────────────────────────────────────────
    if 'AO' in df.columns:
        ao_data = _hist(df, 'AO')
        if ao_data:
            ao_series_list = [{"type": "Histogram", "data": ao_data, "options": {}}]
            # Add divergence lines on AO pane
            for als in div_series['ao_lines']:
                ao_series_list.append(als)
            panes.append({
                "chart": _theme(),
                "series": ao_series_list,
                "height": int(total_height * 0.15),
                "title": "Awesome Oscillator",
            })

    # ── PANE 3: MACD ─────────────────────────────────────────────────
    macd_series = []

    if 'MACD_Hist' in df.columns:
        h = _hist(df, 'MACD_Hist')
        if h:
            macd_series.append({
                "type": "Histogram",
                "data": h,
                "options": {},
            })

    if 'MACD' in df.columns:
        d = _line(df, 'MACD')
        if d:
            macd_series.append({
                "type": "Line",
                "data": d,
                "options": {
                    "color": "#2962ff",
                    "lineWidth": 2,
                    "crosshairMarkerVisible": False,
                },
                "label": "MACD",
            })

    if 'MACD_Signal' in df.columns:
        d = _line(df, 'MACD_Signal')
        if d:
            macd_series.append({
                "type": "Line",
                "data": d,
                "options": {
                    "color": "#ff6d00",
                    "lineWidth": 2,
                    "crosshairMarkerVisible": False,
                },
                "label": "Signal",
            })

    # Crossover dots on MACD pane
    bull_dots, bear_dots = _crossover_dots(df)
    if bull_dots:
        macd_series.append({
            "type": "Line",
            "data": bull_dots,
            "options": {
                "color": "#00E676",
                "lineWidth": 0,
                "lineVisible": False,
                "pointMarkersVisible": True,
                "pointMarkersRadius": 5,
                "crosshairMarkerVisible": False,
                "lastValueVisible": False,
                "priceLineVisible": False,
            },
            "label": "Bullish Cross",
        })
    if bear_dots:
        macd_series.append({
            "type": "Line",
            "data": bear_dots,
            "options": {
                "color": "#FF1744",
                "lineWidth": 0,
                "lineVisible": False,
                "pointMarkersVisible": True,
                "pointMarkersRadius": 5,
                "crosshairMarkerVisible": False,
                "lastValueVisible": False,
                "priceLineVisible": False,
            },
            "label": "Bearish Cross",
        })

    if macd_series:
        panes.append({
            "chart": _theme(),
            "series": macd_series,
            "height": int(total_height * 0.20),
            "title": get_macd_indicator_label(macd_profile),
        })

    return panes


# =============================================================================
# RENDER — Called from app.py
# =============================================================================

def render_tv_chart(df: pd.DataFrame, ticker: str,
                    signal: EntrySignal = None,
                    show_volume: bool = True,
                    show_resistance: bool = True,
                    show_divergence: bool = True,
                    height: int = 800,
                    zoom_level: int = 200,
                    extra_markers: list = None,
                    key: str = None,
                    macd_profile: Optional[str] = None):
    """Render TradingView chart in Streamlit."""
    from lightweight_charts_v5 import lightweight_charts_v5_component

    charts = build_lwc_charts(
        df, ticker, signal=signal,
        show_volume=show_volume,
        show_resistance=show_resistance,
        show_divergence=show_divergence,
        total_height=height,
        extra_markers=extra_markers,
        macd_profile=macd_profile,
    )

    lightweight_charts_v5_component(
        name=f"{ticker} — ${float(df['Close'].iloc[-1]):.2f}",
        charts=charts,
        height=height,
        zoom_level=zoom_level,
        key=key or f"tv_{ticker}",
    )


# =============================================================================
# MTF CHART — Also TradingView LWC v5
# =============================================================================

def render_mtf_chart(daily_df, weekly_df, monthly_df, ticker, height=350, key_prefix: str = "", macd_profile: Optional[str] = None):
    """
    Multi-timeframe view: 3 separate LWC charts side by side
    using st.columns, each with candlestick + MACD + AO.
    """
    import streamlit as st
    from lightweight_charts_v5 import lightweight_charts_v5_component

    panels = [
        ('Daily', daily_df, 120),     # Show last 120 bars initially, but ALL data available
        ('Weekly', weekly_df, 104),    # ~2 years of weeks
        ('Monthly', monthly_df, 60),   # 5 years of months
    ]

    cols = st.columns(3)

    for col_idx, (label, raw_df, zoom) in enumerate(panels):
        if raw_df is None or raw_df.empty:
            continue

        d = normalize_columns(raw_df).copy()
        d = add_all_indicators(d, macd_profile=macd_profile)
        # Pass ALL data — let LWC zoom_level handle initial view
        if d.empty:
            continue

        # Build panes for this timeframe
        panes = []

        # Price pane
        panes.append({
            "chart": _theme(),
            "series": [{
                "type": "Candlestick",
                "data": _candles(d),
                "options": {
                    "upColor": COLOR_BULL,
                    "downColor": COLOR_BEAR,
                    "borderVisible": False,
                    "wickUpColor": COLOR_BULL,
                    "wickDownColor": COLOR_BEAR,
                    "lastValueVisible": True,
                    "priceLineVisible": True,
                },
            }],
            "height": int(height * 0.50),
            "title": f"{ticker} {label}",
        })

        # AO pane
        if 'AO' in d.columns:
            ao_data = _hist(d, 'AO')
            if ao_data:
                panes.append({
                    "chart": _theme(),
                    "series": [{"type": "Histogram", "data": ao_data, "options": {}}],
                    "height": int(height * 0.25),
                    "title": "AO",
                })

        # MACD pane
        macd_s = []
        if 'MACD_Hist' in d.columns:
            h = _hist(d, 'MACD_Hist')
            if h:
                macd_s.append({"type": "Histogram", "data": h, "options": {}})
        if 'MACD' in d.columns:
            md = _line(d, 'MACD')
            if md:
                macd_s.append({"type": "Line", "data": md,
                               "options": {"color": "#2962ff", "lineWidth": 1,
                                           "crosshairMarkerVisible": False},
                               "label": "MACD"})
        if 'MACD_Signal' in d.columns:
            sd = _line(d, 'MACD_Signal')
            if sd:
                macd_s.append({"type": "Line", "data": sd,
                               "options": {"color": "#ff6d00", "lineWidth": 1,
                                           "crosshairMarkerVisible": False},
                               "label": "Signal"})
        if macd_s:
            panes.append({
                "chart": _theme(),
                "series": macd_s,
                "height": int(height * 0.25),
                "title": get_macd_indicator_label(macd_profile),
            })

        with cols[col_idx]:
            lightweight_charts_v5_component(
                name=f"{ticker} {label}",
                charts=panes,
                height=height,
                zoom_level=zoom,
                key=f"{key_prefix}mtf_{ticker}_{label}",
            )


# =============================================================================
# EXPORT (for AI analysis)
# =============================================================================

def chart_to_base64(fig, width=1200, height=700):
    import base64
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        return base64.b64encode(img_bytes).decode('utf-8')
    except:
        return None

def chart_to_html(fig):
    return fig.to_html(include_plotlyjs='cdn', full_html=False)
