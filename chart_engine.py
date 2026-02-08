"""
TTA v2 Chart Engine — TradingView Lightweight Charts v5
=========================================================

Uses streamlit-lightweight-charts-v5 for native TradingView experience:
- Mouse wheel zoom (bunch up / spread out candles)
- Auto-scaling y-axis as you zoom
- Crosshair cursor synced across panes
- Multi-pane: Price + Volume + AO + MACD
- Pan, zoom, scroll — all native TradingView behavior

Version: 3.0.0 (2026-02-08)
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

COLOR_BULL = 'rgba(38,166,154,0.9)'   # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'    # #ef5350
COLOR_BULL_LIGHT = 'rgba(38,166,154,0.3)'
COLOR_BEAR_LIGHT = 'rgba(239,83,80,0.3)'


# =============================================================================
# DATA FORMATTERS — Convert DataFrames to LWC JSON format
# =============================================================================

def _df_to_candles(df: pd.DataFrame) -> list:
    """Convert OHLCV DataFrame to LWC candlestick JSON."""
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


def _series_to_line(df: pd.DataFrame, col: str, color: str = 'blue') -> list:
    """Convert a single column to LWC line series JSON."""
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'), 'value': round(float(v), 4), 'color': color}
            for idx, v in s.items()]


def _series_to_histogram(df: pd.DataFrame, col: str,
                         pos_color: str = COLOR_BULL,
                         neg_color: str = COLOR_BEAR) -> list:
    """Convert a column to LWC histogram JSON with pos/neg colors."""
    s = df[col].dropna()
    return [{'time': idx.strftime('%Y-%m-%d'),
             'value': round(float(v), 4),
             'color': pos_color if v >= 0 else neg_color}
            for idx, v in s.items()]


def _volume_to_histogram(df: pd.DataFrame) -> list:
    """Convert volume to LWC histogram with bull/bear colors."""
    records = []
    for idx, row in df.iterrows():
        color = COLOR_BULL_LIGHT if row['Close'] >= row['Open'] else COLOR_BEAR_LIGHT
        records.append({
            'time': idx.strftime('%Y-%m-%d'),
            'value': float(row['Volume']),
            'color': color,
        })
    return records


# =============================================================================
# MAIN CHART — Multi-pane TradingView chart config
# =============================================================================

def build_lwc_chart_config(
    df: pd.DataFrame,
    ticker: str,
    signal: EntrySignal = None,
    show_volume: bool = True,
    show_resistance: bool = True,
    height: int = 800,
) -> dict:
    """
    Build the charts config dict for lightweight_charts_v5_component.

    Returns a single chart config with multi-pane series using the
    v5 component's chart/series format.

    Panes:
    - Pane 0: Candlestick + SMAs + resistance rectangles
    - Pane 1: Volume histogram
    - Pane 2: AO histogram
    - Pane 3: MACD (histogram + signal + MACD lines)
    """
    df = normalize_columns(df).copy()

    # Ensure indicators
    if 'MACD' not in df.columns:
        df = add_all_indicators(df)

    # ── Chart options (TradingView dark theme) ────────────────────────
    chart_options = {
        "height": height,
        "layout": {
            "background": {"type": "solid", "color": "#131722"},
            "textColor": "#d1d4dc",
            "fontSize": 12,
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.4)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.4)"},
        },
        "crosshair": {
            "mode": 0,  # Normal crosshair
        },
        "rightPriceScale": {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "scaleMargins": {"top": 0.05, "bottom": 0.05},
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "timeVisible": False,
            "barSpacing": 6,
            "minBarSpacing": 2,
            "rightOffset": 5,
        },
    }

    # ── Build series list ─────────────────────────────────────────────
    series = []

    # --- Pane 0: Candlestick ---
    candles = _df_to_candles(df)
    series.append({
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR,
            "title": ticker,
            "pane": 0,
        },
    })

    # --- Pane 0: Moving Averages ---
    if 'SMA_150' in df.columns:
        sma = _series_to_line(df, 'SMA_150', '#ff9800')
        if sma:
            series.append({
                "type": "Line",
                "data": sma,
                "options": {
                    "color": "#ff9800",
                    "lineWidth": 2,
                    "lineStyle": 1,  # Dotted
                    "title": "150d SMA",
                    "pane": 0,
                },
            })

    if 'SMA_50' in df.columns:
        sma = _series_to_line(df, 'SMA_50', '#42a5f5')
        if sma:
            series.append({
                "type": "Line",
                "data": sma,
                "options": {
                    "color": "#42a5f5",
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed
                    "title": "50 SMA",
                    "pane": 0,
                },
            })

    if 'SMA_200' in df.columns:
        sma = _series_to_line(df, 'SMA_200', '#ab47bc')
        if sma:
            series.append({
                "type": "Line",
                "data": sma,
                "options": {
                    "color": "#ab47bc",
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed
                    "title": "200 SMA",
                    "pane": 0,
                },
            })

    # --- Pane 0: Resistance levels as horizontal lines ---
    if show_resistance and signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        critical = signal.overhead_resistance.get('critical_level', {})

        for lev in levels:
            price = lev['price']
            is_crit = critical and abs(price - critical.get('price', 0)) < 0.01

            # Create a flat line series at the resistance price
            res_data = [
                {'time': df.index[0].strftime('%Y-%m-%d'), 'value': round(price, 2)},
                {'time': df.index[-1].strftime('%Y-%m-%d'), 'value': round(price, 2)},
            ]
            color = '#ff1744' if is_crit else 'rgba(239, 83, 80, 0.5)'
            series.append({
                "type": "Line",
                "data": res_data,
                "options": {
                    "color": color,
                    "lineWidth": 2 if is_crit else 1,
                    "lineStyle": 0 if is_crit else 2,  # Solid if critical, dashed otherwise
                    "title": f"R ${price:.0f}" + (" ★" if is_crit else ""),
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": True,
                    "priceLineVisible": False,
                    "pane": 0,
                },
            })

    # --- Pane 1: Volume ---
    if show_volume and 'Volume' in df.columns:
        vol_data = _volume_to_histogram(df)
        series.append({
            "type": "Histogram",
            "data": vol_data,
            "options": {
                "priceFormat": {"type": "volume"},
                "title": "Volume",
                "pane": 1,
            },
        })

    # --- Pane 2: Awesome Oscillator ---
    if 'AO' in df.columns:
        ao_data = _series_to_histogram(df, 'AO')
        series.append({
            "type": "Histogram",
            "data": ao_data,
            "options": {
                "title": "AO",
                "pane": 2,
            },
        })

    # --- Pane 3: MACD ---
    if 'MACD_Hist' in df.columns:
        macd_hist = _series_to_histogram(df, 'MACD_Hist')
        series.append({
            "type": "Histogram",
            "data": macd_hist,
            "options": {
                "title": "MACD Hist",
                "pane": 3,
            },
        })

    if 'MACD' in df.columns:
        macd_line = _series_to_line(df, 'MACD', '#2962ff')
        series.append({
            "type": "Line",
            "data": macd_line,
            "options": {
                "color": "#2962ff",
                "lineWidth": 2,
                "title": "MACD",
                "pane": 3,
            },
        })

    if 'MACD_Signal' in df.columns:
        sig_line = _series_to_line(df, 'MACD_Signal', '#ff6d00')
        series.append({
            "type": "Line",
            "data": sig_line,
            "options": {
                "color": "#ff6d00",
                "lineWidth": 2,
                "title": "Signal",
                "pane": 3,
            },
        })

    return {
        "chart": chart_options,
        "series": series,
    }


# =============================================================================
# RENDER FUNCTION — Called from app.py
# =============================================================================

def render_tv_chart(df: pd.DataFrame, ticker: str,
                    signal: EntrySignal = None,
                    show_volume: bool = True,
                    show_resistance: bool = True,
                    height: int = 800,
                    key: str = None):
    """
    Render TradingView-style chart in Streamlit using lightweight-charts-v5.

    Call this from app.py instead of st.plotly_chart().
    """
    from lightweight_charts_v5 import lightweight_charts_v5_component

    config = build_lwc_chart_config(
        df, ticker, signal=signal,
        show_volume=show_volume,
        show_resistance=show_resistance,
        height=height,
    )

    lightweight_charts_v5_component(
        name=f"{ticker} — ${float(df['Close'].iloc[-1]):.2f}",
        charts=[config],
        height=height,
        zoom_level=200,  # Initial visible bars
        key=key or f"tv_chart_{ticker}",
    )


# =============================================================================
# MULTI-TIMEFRAME (kept simple with Plotly for now)
# =============================================================================

def create_mtf_chart(daily_df, weekly_df, monthly_df, ticker, height=400):
    """MTF chart using Plotly (LWC v5 doesn't support side-by-side easily)."""
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
# EXPORT (kept for AI analysis screenshots)
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
