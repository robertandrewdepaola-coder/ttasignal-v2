"""
TTA v2 Chart Engine — TradingView Lightweight Charts v5
=========================================================

Uses streamlit-lightweight-charts-v5 for native TradingView experience.

v5 multi-pane: each pane is a separate dict in the `charts` array,
each with its own height and series list.

Version: 3.1.0 (2026-02-08)
"""

import pandas as pd
import numpy as np
import json
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

# Dark theme config shared across all panes
DARK_THEME = {
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

def _df_to_candles(df: pd.DataFrame) -> list:
    """OHLCV -> LWC candlestick records."""
    records = []
    for idx, row in df.iterrows():
        records.append({
            'time': idx.strftime('%Y-%m-%d'),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
        })
    return records


def _col_to_line(df: pd.DataFrame, col: str) -> list:
    """Column -> LWC line records (no color per point)."""
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'), 'value': round(float(v), 4)}
            for idx, v in s.items()]


def _col_to_colored_hist(df: pd.DataFrame, col: str,
                         pos_color: str = COLOR_BULL,
                         neg_color: str = COLOR_BEAR) -> list:
    """Column -> LWC histogram records with pos/neg colors."""
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'),
             'value': round(float(v), 4),
             'color': pos_color if v >= 0 else neg_color}
            for idx, v in s.items()]


def _volume_hist(df: pd.DataFrame) -> list:
    """Volume -> LWC histogram with bull/bear colors."""
    records = []
    for idx, row in df.iterrows():
        c = COLOR_BULL_LIGHT if row['Close'] >= row['Open'] else COLOR_BEAR_LIGHT
        records.append({
            'time': idx.strftime('%Y-%m-%d'),
            'value': float(row['Volume']),
            'color': c,
        })
    return records


# =============================================================================
# BUILD MULTI-PANE CHART CONFIG
# =============================================================================

def build_lwc_charts(
    df: pd.DataFrame,
    ticker: str,
    signal: EntrySignal = None,
    show_volume: bool = True,
    show_resistance: bool = True,
    total_height: int = 800,
) -> list:
    """
    Build the `charts` list for lightweight_charts_v5_component.

    Each pane is a separate dict: {chart, series, height, title}

    Pane 0: Price (candlestick + SMAs + resistance) — 55% height
    Pane 1: Volume histogram — 10% height
    Pane 2: AO histogram — 15% height
    Pane 3: MACD (histogram + lines) — 20% height
    """
    df = normalize_columns(df).copy()
    if 'MACD' not in df.columns:
        df = add_all_indicators(df)

    has_vol = show_volume and 'Volume' in df.columns
    panes = []

    # ── PANE 0: PRICE ────────────────────────────────────────────────
    price_series = []

    # Candlestick
    price_series.append({
        "type": "Candlestick",
        "data": _df_to_candles(df),
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR,
        },
    })

    # 150d SMA
    if 'SMA_150' in df.columns:
        data = _col_to_line(df, 'SMA_150')
        if data:
            price_series.append({
                "type": "Line",
                "data": data,
                "options": {
                    "color": "#ff9800",
                    "lineWidth": 2,
                    "lineStyle": 1,
                },
                "label": f"150 SMA ${float(df['SMA_150'].dropna().iloc[-1]):.2f}" if len(df['SMA_150'].dropna()) > 0 else "150 SMA",
            })

    # 50d SMA
    if 'SMA_50' in df.columns:
        data = _col_to_line(df, 'SMA_50')
        if data:
            price_series.append({
                "type": "Line",
                "data": data,
                "options": {
                    "color": "#42a5f5",
                    "lineWidth": 1,
                    "lineStyle": 2,
                },
                "label": f"50 SMA ${float(df['SMA_50'].dropna().iloc[-1]):.2f}" if len(df['SMA_50'].dropna()) > 0 else "50 SMA",
            })

    # 200d SMA
    if 'SMA_200' in df.columns:
        data = _col_to_line(df, 'SMA_200')
        if data:
            price_series.append({
                "type": "Line",
                "data": data,
                "options": {
                    "color": "#ab47bc",
                    "lineWidth": 1,
                    "lineStyle": 2,
                },
                "label": f"200 SMA ${float(df['SMA_200'].dropna().iloc[-1]):.2f}" if len(df['SMA_200'].dropna()) > 0 else "200 SMA",
            })

    # Resistance levels as flat line series
    if show_resistance and signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        critical = signal.overhead_resistance.get('critical_level', {})
        for lev in levels:
            price = lev['price']
            is_crit = critical and abs(price - critical.get('price', 0)) < 0.01
            color = '#ff1744' if is_crit else 'rgba(239, 83, 80, 0.4)'
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

    price_height = int(total_height * 0.55) if has_vol else int(total_height * 0.50)
    panes.append({
        "chart": DARK_THEME,
        "series": price_series,
        "height": price_height,
        "title": ticker,
    })

    # ── PANE 1: VOLUME ───────────────────────────────────────────────
    if has_vol:
        vol_data = _volume_hist(df)
        panes.append({
            "chart": DARK_THEME,
            "series": [{
                "type": "Histogram",
                "data": vol_data,
                "options": {
                    "priceFormat": {"type": "volume"},
                },
            }],
            "height": int(total_height * 0.10),
            "title": "Volume",
        })

    # ── PANE 2: AWESOME OSCILLATOR ───────────────────────────────────
    if 'AO' in df.columns:
        ao_data = _col_to_colored_hist(df, 'AO')
        if ao_data:
            panes.append({
                "chart": DARK_THEME,
                "series": [{
                    "type": "Histogram",
                    "data": ao_data,
                    "options": {},
                }],
                "height": int(total_height * 0.15),
                "title": "AO",
            })

    # ── PANE 3: MACD ─────────────────────────────────────────────────
    macd_series = []

    if 'MACD_Hist' in df.columns:
        hist_data = _col_to_colored_hist(df, 'MACD_Hist')
        if hist_data:
            macd_series.append({
                "type": "Histogram",
                "data": hist_data,
                "options": {},
            })

    if 'MACD' in df.columns:
        macd_data = _col_to_line(df, 'MACD')
        if macd_data:
            macd_series.append({
                "type": "Line",
                "data": macd_data,
                "options": {
                    "color": "#2962ff",
                    "lineWidth": 2,
                },
                "label": "MACD",
            })

    if 'MACD_Signal' in df.columns:
        sig_data = _col_to_line(df, 'MACD_Signal')
        if sig_data:
            macd_series.append({
                "type": "Line",
                "data": sig_data,
                "options": {
                    "color": "#ff6d00",
                    "lineWidth": 2,
                },
                "label": "Signal",
            })

    if macd_series:
        panes.append({
            "chart": DARK_THEME,
            "series": macd_series,
            "height": int(total_height * 0.20),
            "title": "MACD",
        })

    return panes


# =============================================================================
# RENDER — Called from app.py
# =============================================================================

def render_tv_chart(df: pd.DataFrame, ticker: str,
                    signal: EntrySignal = None,
                    show_volume: bool = True,
                    show_resistance: bool = True,
                    height: int = 800,
                    key: str = None):
    """Render TradingView chart in Streamlit."""
    from lightweight_charts_v5 import lightweight_charts_v5_component

    charts = build_lwc_charts(
        df, ticker, signal=signal,
        show_volume=show_volume,
        show_resistance=show_resistance,
        total_height=height,
    )

    lightweight_charts_v5_component(
        name=f"{ticker} — ${float(df['Close'].iloc[-1]):.2f}",
        charts=charts,
        height=height,
        zoom_level=200,
        key=key or f"tv_chart_{ticker}",
    )


# =============================================================================
# MTF CHART (Plotly for side-by-side)
# =============================================================================

def create_mtf_chart(daily_df, weekly_df, monthly_df, ticker, height=400):
    """Multi-timeframe chart using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, shared_yaxes=False,
                        horizontal_spacing=0.05,
                        subplot_titles=('Daily', 'Weekly', 'Monthly'))

    panels = [('Daily', daily_df, 60), ('Weekly', weekly_df, 52), ('Monthly', monthly_df, 24)]

    for idx, (label, raw_df, n) in enumerate(panels, start=1):
        if raw_df is None or raw_df.empty:
            continue
        d = normalize_columns(raw_df).copy()
        d = calculate_macd(d)
        r = d.tail(n)
        if r.empty:
            continue

        fig.add_trace(go.Candlestick(
            x=r.index, open=r['Open'], high=r['High'],
            low=r['Low'], close=r['Close'],
            increasing=dict(line=dict(color='#26a69a'), fillcolor='rgba(38,166,154,0.8)'),
            decreasing=dict(line=dict(color='#ef5350'), fillcolor='rgba(239,83,80,0.8)'),
            name=label, showlegend=False,
        ), row=1, col=idx)

        if 'MACD' in r.columns and 'MACD_Signal' in r.columns:
            m, s = float(r['MACD'].iloc[-1]), float(r['MACD_Signal'].iloc[-1])
            if not (pd.isna(m) or pd.isna(s)):
                bullish = m > s
                fig.add_annotation(
                    text="MACD ✅" if bullish else "MACD ❌",
                    x=r.index[len(r)//2], y=float(r['Low'].min()) * 0.97,
                    xref=f'x{idx}', yref=f'y{idx}',
                    showarrow=False,
                    font=dict(size=12, color='#26a69a' if bullish else '#ef5350'),
                )

    fig.update_layout(
        height=height, template='plotly_dark',
        paper_bgcolor='#131722', plot_bgcolor='#131722',
        showlegend=False, margin=dict(l=40, r=40, t=40, b=20),
    )
    for i in range(1, 4):
        fig.update_xaxes(rangeslider_visible=False, showgrid=False,
                         tickformat='%b\'%y', tickfont=dict(size=9, color='#787b86'),
                         row=1, col=i)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(42,46,57,0.6)',
                         side='right', tickfont=dict(size=9, color='#787b86'),
                         row=1, col=i)
    return fig


# =============================================================================
# EXPORT
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
