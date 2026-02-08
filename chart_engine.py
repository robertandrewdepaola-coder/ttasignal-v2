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
    EntrySignal,
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
# DIVERGENCE MARKERS — shown on candlestick series
# =============================================================================

def _divergence_markers(df):
    """Build LWC marker objects for bearish divergence points."""
    if 'bearish_div_detected' not in df.columns:
        return []
    divs = df[df['bearish_div_detected'] == True]
    markers = []
    for idx, row in divs.iterrows():
        markers.append({
            "time": idx.strftime('%Y-%m-%d'),
            "position": "aboveBar",
            "color": "#ffa726",
            "shape": "arrowDown",
            "text": "Div ⚠",
        })
    return markers


def _signal_markers(df):
    """
    Build LWC marker objects for MACD buy/sell crossover signals.
    Buy = MACD crosses above Signal while AO > 0
    Sell = MACD crosses below Signal
    """
    if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
        return []

    markers = []
    macd = df['MACD'].values
    sig = df['MACD_Signal'].values
    ao = df['AO'].values if 'AO' in df.columns else [0] * len(df)

    for i in range(1, len(df)):
        if pd.isna(macd[i]) or pd.isna(sig[i]) or pd.isna(macd[i-1]) or pd.isna(sig[i-1]):
            continue

        # Bullish cross: MACD crosses above Signal
        if macd[i] > sig[i] and macd[i-1] <= sig[i-1]:
            ao_val = ao[i] if not pd.isna(ao[i]) else 0
            if ao_val > 0:
                # Strong buy: AO positive
                markers.append({
                    "time": df.index[i].strftime('%Y-%m-%d'),
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "Buy",
                })
            else:
                # Weak buy: AO negative
                markers.append({
                    "time": df.index[i].strftime('%Y-%m-%d'),
                    "position": "belowBar",
                    "color": "#66bb6a",
                    "shape": "arrowUp",
                    "text": "Buy?",
                })

        # Bearish cross: MACD crosses below Signal
        elif macd[i] < sig[i] and macd[i-1] >= sig[i-1]:
            markers.append({
                "time": df.index[i].strftime('%Y-%m-%d'),
                "position": "aboveBar",
                "color": "#ef5350",
                "shape": "arrowDown",
                "text": "Sell",
            })

    return markers


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
) -> list:
    """
    Build `charts` list for LWC v5.
    Each pane = separate dict with {chart, series, height, title}.
    """
    df = normalize_columns(df).copy()
    if 'MACD' not in df.columns:
        df = add_all_indicators(df)
    if show_divergence:
        df = detect_bearish_divergence(df)

    has_vol = show_volume and 'Volume' in df.columns
    panes = []
    current_price = round(float(df['Close'].iloc[-1]), 2)

    # ── PANE 0: PRICE ────────────────────────────────────────────────
    price_series = []

    # Candlestick with divergence markers
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

    # Add divergence + buy/sell markers to candlestick
    all_markers = []
    if show_divergence:
        all_markers.extend(_divergence_markers(df))
    all_markers.extend(_signal_markers(df))
    if all_markers:
        candle_config["markers"] = all_markers

    price_series.append(candle_config)

    # Moving averages — crosshairMarkerVisible=False so dots don't snap to them
    sma_configs = [
        ('SMA_150', '#ff9800', 2, 1, '150d SMA'),  # orange, dotted
        ('SMA_50', '#42a5f5', 1, 2, '50 SMA'),      # blue, dashed
        ('SMA_200', '#ab47bc', 1, 2, '200 SMA'),     # purple, dashed
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

    # Price pane — tight scaleMargins to reduce dead space
    price_theme = _theme()
    price_theme["rightPriceScale"]["scaleMargins"] = {"top": 0.02, "bottom": 0.02}

    ph = int(total_height * 0.55) if has_vol else int(total_height * 0.50)
    panes.append({
        "chart": price_theme,
        "series": price_series,
        "height": ph,
        "title": f"{ticker}  ${current_price}",
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
            panes.append({
                "chart": _theme(),
                "series": [{
                    "type": "Histogram",
                    "data": ao_data,
                    "options": {},
                }],
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

    if macd_series:
        panes.append({
            "chart": _theme(),
            "series": macd_series,
            "height": int(total_height * 0.20),
            "title": "MACD (12/26/9)",
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
                    key: str = None):
    """Render TradingView chart in Streamlit."""
    from lightweight_charts_v5 import lightweight_charts_v5_component

    charts = build_lwc_charts(
        df, ticker, signal=signal,
        show_volume=show_volume,
        show_resistance=show_resistance,
        show_divergence=show_divergence,
        total_height=height,
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

def render_mtf_chart(daily_df, weekly_df, monthly_df, ticker, height=350):
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
        d = add_all_indicators(d)
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
                "title": "MACD",
            })

        with cols[col_idx]:
            lightweight_charts_v5_component(
                name=f"{ticker} {label}",
                charts=panes,
                height=height,
                zoom_level=zoom,
                key=f"mtf_{ticker}_{label}",
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
