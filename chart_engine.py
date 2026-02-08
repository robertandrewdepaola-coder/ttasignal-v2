"""
TTA v2 Chart Engine — Professional Plotly Financial Charts
============================================================

Built with Graph Objects + make_subplots for granular control.
TradingView-inspired design: clean, interactive, narrative.

Design principles:
- Multi-pane vertical layout with proper row weighting
- Spikelines (crosshairs) that snap to data points
- Right-side price axis (TradingView convention)
- Clean aesthetic: minimal gridlines, dark theme, data pops
- Narrative annotations on resistance levels
- All data loaded, visible window controlled by x-range

Version: 2.1.0 (2026-02-08)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

from signal_engine import (
    normalize_columns, calculate_macd, calculate_ao, calculate_atr,
    calculate_sma, add_all_indicators, detect_bearish_divergence,
    find_overhead_resistance, EntrySignal,
)


# =============================================================================
# COLOR PALETTE — TradingView-inspired dark theme
# =============================================================================

C = {
    # Candles
    'up': '#26a69a',
    'down': '#ef5350',
    'up_fill': 'rgba(38, 166, 154, 0.8)',
    'down_fill': 'rgba(239, 83, 80, 0.8)',

    # Moving averages
    'sma150': '#ff9800',      # Orange — 30-week / 150-day
    'sma50': '#42a5f5',       # Blue
    'sma200': '#ab47bc',      # Purple

    # Volume
    'vol_up': 'rgba(38, 166, 154, 0.4)',
    'vol_down': 'rgba(239, 83, 80, 0.4)',

    # Oscillators
    'ao_up': '#26a69a',
    'ao_down': '#ef5350',
    'macd_line': '#2962ff',
    'macd_signal': '#ff6d00',
    'macd_hist_up': 'rgba(38, 166, 154, 0.6)',
    'macd_hist_dn': 'rgba(239, 83, 80, 0.6)',

    # Overlays
    'resistance': 'rgba(239, 83, 80, 0.5)',
    'resistance_crit': '#ff1744',
    'divergence': '#ffa726',
    'entry': '#00e676',
    'exit_win': '#00e676',
    'exit_loss': '#ff1744',

    # Theme
    'bg': '#131722',
    'panel_bg': '#131722',
    'grid': 'rgba(42, 46, 57, 0.6)',
    'text': '#d1d4dc',
    'text_dim': '#787b86',
    'spike': '#9598a1',
    'zeroline': 'rgba(120, 123, 134, 0.3)',
}


# =============================================================================
# MAIN CHART — Professional multi-pane financial chart
# =============================================================================

def create_analysis_chart(
    df: pd.DataFrame,
    ticker: str,
    weekly_sma: pd.Series = None,
    signal: EntrySignal = None,
    show_volume: bool = True,
    show_resistance: bool = True,
    show_divergence: bool = True,
    trade_markers: List[Dict] = None,
    visible_bars: int = 130,
    height: int = 1000,
) -> go.Figure:
    """
    Professional 4-pane chart: Price | Volume | AO | MACD

    All data is loaded into the figure. The visible_bars parameter
    controls the initial x-axis window. User can pan, scroll-zoom,
    and use timeframe buttons to change the view.

    The y-axis auto-fits to the initial visible window.
    """
    df = normalize_columns(df).copy()

    # Ensure indicators exist
    if 'MACD' not in df.columns:
        df = add_all_indicators(df)
    if show_divergence:
        df = detect_bearish_divergence(df)

    # ─── SUBPLOT LAYOUT ──────────────────────────────────────────────
    has_volume = show_volume and 'Volume' in df.columns

    if has_volume:
        rows = 4
        heights = [0.55, 0.08, 0.17, 0.20]
        titles = ('', '', 'Awesome Oscillator', 'MACD (12/26/9)')
    else:
        rows = 3
        heights = [0.55, 0.22, 0.23]
        titles = ('', 'Awesome Oscillator', 'MACD (12/26/9)')

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=heights,
        subplot_titles=titles,
    )

    # Row assignments
    price_row = 1
    vol_row = 2 if has_volume else None
    ao_row = 3 if has_volume else 2
    macd_row = 4 if has_volume else 3

    # ─── ROW 1: CANDLESTICK ──────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color=C['up'], width=1), fillcolor=C['up_fill']),
            decreasing=dict(line=dict(color=C['down'], width=1), fillcolor=C['down_fill']),
            whiskerwidth=0.5,
        ),
        row=price_row, col=1,
    )

    # ─── MOVING AVERAGES (on price panel) ────────────────────────────
    if weekly_sma is not None and len(weekly_sma) > 0:
        filtered = weekly_sma[
            (weekly_sma.index >= df.index.min()) & (weekly_sma.index <= df.index.max())
        ]
        if len(filtered) > 0:
            fig.add_trace(go.Scatter(
                x=filtered.index, y=filtered,
                mode='lines', name=f'30W SMA ${filtered.iloc[-1]:.2f}',
                line=dict(color=C['sma150'], width=2),
            ), row=price_row, col=1)
    elif 'SMA_150' in df.columns:
        sma = df['SMA_150'].dropna()
        if len(sma) > 0:
            fig.add_trace(go.Scatter(
                x=sma.index, y=sma,
                mode='lines', name=f'150d SMA ${sma.iloc[-1]:.2f}',
                line=dict(color=C['sma150'], width=2, dash='dot'),
            ), row=price_row, col=1)

    if 'SMA_50' in df.columns:
        sma = df['SMA_50'].dropna()
        if len(sma) > 0:
            fig.add_trace(go.Scatter(
                x=sma.index, y=sma,
                mode='lines', name=f'50 SMA ${sma.iloc[-1]:.2f}',
                line=dict(color=C['sma50'], width=1.5, dash='dash'),
            ), row=price_row, col=1)

    if 'SMA_200' in df.columns:
        sma = df['SMA_200'].dropna()
        if len(sma) > 0:
            fig.add_trace(go.Scatter(
                x=sma.index, y=sma,
                mode='lines', name=f'200 SMA ${sma.iloc[-1]:.2f}',
                line=dict(color=C['sma200'], width=1.5, dash='dash'),
            ), row=price_row, col=1)

    # ─── OVERHEAD RESISTANCE (shapes + annotations) ──────────────────
    if show_resistance and signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        critical = signal.overhead_resistance.get('critical_level', {})

        for lev in levels:
            price = lev['price']
            is_crit = critical and abs(price - critical.get('price', 0)) < 0.01
            color = C['resistance_crit'] if is_crit else C['resistance']
            width = 2.5 if is_crit else 1
            dash = 'solid' if is_crit else 'dot'

            # Shape instead of hline (doesn't expand y-axis)
            fig.add_shape(
                type='line',
                x0=df.index[0], x1=df.index[-1],
                y0=price, y1=price,
                line=dict(color=color, width=width, dash=dash),
                xref='x', yref='y',
            )
            fig.add_annotation(
                x=df.index[-1], y=price,
                text=f"R ${price:.0f}" + (" ★" if is_crit else ""),
                font=dict(size=9, color=color),
                showarrow=False,
                xanchor='left', xshift=5,
            )

    # ─── DIVERGENCE MARKERS ──────────────────────────────────────────
    if show_divergence and 'bearish_div_detected' in df.columns:
        divs = df[df['bearish_div_detected'] == True]
        if len(divs) > 0:
            fig.add_trace(go.Scatter(
                x=divs.index, y=divs['High'] * 1.005,
                mode='markers', name='Bearish Divergence',
                marker=dict(symbol='diamond', size=10, color=C['divergence'],
                            line=dict(width=1, color='white')),
            ), row=price_row, col=1)

    # ─── TRADE MARKERS ───────────────────────────────────────────────
    if trade_markers:
        for m in trade_markers:
            color = C['entry'] if m.get('type') == 'entry' else (
                C['exit_win'] if m.get('win', True) else C['exit_loss'])
            symbol = 'triangle-up' if m.get('type') == 'entry' else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['price']],
                mode='markers+text', name=m.get('label', ''),
                marker=dict(symbol=symbol, size=12, color=color),
                text=[m.get('label', '')],
                textposition='top center' if m.get('type') == 'entry' else 'bottom center',
                textfont=dict(size=9, color=color),
                showlegend=False,
            ), row=price_row, col=1)

    # ─── ROW 2: VOLUME ───────────────────────────────────────────────
    if has_volume and vol_row:
        vol_colors = [
            C['vol_up'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
            else C['vol_down']
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume', marker_color=vol_colors, showlegend=False,
        ), row=vol_row, col=1)

    # ─── ROW 3: AWESOME OSCILLATOR ───────────────────────────────────
    if 'AO' in df.columns:
        ao = df['AO'].dropna()
        ao_colors = [C['ao_up'] if v >= 0 else C['ao_down'] for v in ao]
        fig.add_trace(go.Bar(
            x=ao.index, y=ao, name='AO',
            marker_color=ao_colors, showlegend=False,
        ), row=ao_row, col=1)

    # ─── ROW 4: MACD ─────────────────────────────────────────────────
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        hist = df['MACD_Hist'].dropna()
        hist_colors = [C['macd_hist_up'] if v >= 0 else C['macd_hist_dn'] for v in hist]

        fig.add_trace(go.Bar(
            x=hist.index, y=hist, name='MACD Hist',
            marker_color=hist_colors, showlegend=False,
        ), row=macd_row, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color=C['macd_line'], width=1.5),
        ), row=macd_row, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color=C['macd_signal'], width=1.5),
        ), row=macd_row, col=1)

    # ═══════════════════════════════════════════════════════════════════
    # VISIBLE WINDOW + AXIS CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    n_vis = min(visible_bars, len(df))
    vis = df.tail(n_vis)

    x_min = vis.index[0]
    x_max = df.index[-1]

    # Tight y-range from visible candles
    y_lo = float(vis['Low'].min())
    y_hi = float(vis['High'].max())
    pad = (y_hi - y_lo) * 0.05 if (y_hi - y_lo) > 1 else y_hi * 0.05
    y_min = y_lo - pad
    y_max = y_hi + pad

    # ─── GLOBAL LAYOUT ───────────────────────────────────────────────
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor=C['bg'],
        plot_bgcolor=C['panel_bg'],
        font=dict(family='Inter, sans-serif', color=C['text']),
        title=dict(
            text=f"<b>{ticker}</b> — ${float(df['Close'].iloc[-1]):.2f}",
            font=dict(size=18, color=C['text']),
            x=0.01, xanchor='left',
        ),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
            font=dict(size=10, color=C['text_dim']),
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant',
        ),
        margin=dict(l=10, r=70, t=60, b=30),
        dragmode='pan',
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=False)),
    )

    # ─── PER-AXIS CONFIGURATION ──────────────────────────────────────
    for i in range(1, rows + 1):
        ax = '' if i == 1 else str(i)

        fig.update_layout(**{
            f'xaxis{ax}': dict(
                range=[x_min, x_max],
                showgrid=False,
                zeroline=False,
                showticklabels=(i == rows),
                tickformat='%b\'%y',
                tickfont=dict(size=10, color=C['text_dim']),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor=C['spike'],
                spikethickness=0.5,
                spikedash='solid',
            ),
            f'yaxis{ax}': dict(
                side='right',
                showgrid=True,
                gridcolor=C['grid'],
                gridwidth=0.5,
                zeroline=False,
                tickfont=dict(size=10, color=C['text_dim']),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor=C['spike'],
                spikethickness=0.5,
                spikedash='solid',
            ),
        })

    # ─── PRICE Y-AXIS: TIGHT (last to override everything) ──────────
    fig.update_layout(yaxis=dict(
        range=[y_min, y_max],
        autorange=False,
        fixedrange=False,
    ))

    # Oscillator zero lines
    ao_ax = f'yaxis{ao_row}' if ao_row > 1 else 'yaxis'
    macd_ax = f'yaxis{macd_row}' if macd_row > 1 else 'yaxis'
    fig.update_layout(**{
        ao_ax: dict(zeroline=True, zerolinecolor=C['zeroline'], zerolinewidth=1),
        macd_ax: dict(zeroline=True, zerolinecolor=C['zeroline'], zerolinewidth=1),
    })

    # Volume: hide y labels
    if has_volume:
        fig.update_layout(yaxis2=dict(showticklabels=False))

    return fig


# =============================================================================
# MINI CHART — Sparkline for scanner table
# =============================================================================

def create_mini_chart(df: pd.DataFrame, ticker: str,
                      width: int = 300, height: int = 150) -> go.Figure:
    df = normalize_columns(df).copy()
    recent = df.tail(60)
    up = recent['Close'].iloc[-1] >= recent['Close'].iloc[0]
    color = C['up'] if up else C['down']
    fill = 'rgba(38,166,154,0.1)' if up else 'rgba(239,83,80,0.1)'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['Close'],
        mode='lines', line=dict(color=color, width=1.5),
        fill='tozeroy', fillcolor=fill,
    ))
    fig.update_layout(
        height=height, width=width,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# =============================================================================
# MULTI-TIMEFRAME CHART — Daily | Weekly | Monthly
# =============================================================================

def create_mtf_chart(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    ticker: str,
    height: int = 400,
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.05,
        subplot_titles=('Daily', 'Weekly', 'Monthly'),
    )

    panels = [
        ('Daily', daily_df, 60),
        ('Weekly', weekly_df, 52),
        ('Monthly', monthly_df, 24),
    ]

    for idx, (label, raw_df, n_bars) in enumerate(panels, start=1):
        if raw_df is None or raw_df.empty:
            continue

        d = normalize_columns(raw_df).copy()
        d = calculate_macd(d)
        recent = d.tail(n_bars)
        if recent.empty:
            continue

        fig.add_trace(go.Candlestick(
            x=recent.index,
            open=recent['Open'], high=recent['High'],
            low=recent['Low'], close=recent['Close'],
            increasing=dict(line=dict(color=C['up']), fillcolor=C['up_fill']),
            decreasing=dict(line=dict(color=C['down']), fillcolor=C['down_fill']),
            name=label, showlegend=False,
        ), row=1, col=idx)

        # MACD status annotation
        if 'MACD' in recent.columns and 'MACD_Signal' in recent.columns:
            m = float(recent['MACD'].iloc[-1])
            s = float(recent['MACD_Signal'].iloc[-1])
            if not (pd.isna(m) or pd.isna(s)):
                bullish = m > s
                status = "MACD ✅" if bullish else "MACD ❌"
                color = C['up'] if bullish else C['down']
                fig.add_annotation(
                    text=status,
                    x=recent.index[len(recent) // 2],
                    y=float(recent['Low'].min()) * 0.97,
                    xref=f'x{idx}', yref=f'y{idx}',
                    showarrow=False,
                    font=dict(size=12, color=color),
                )

    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor=C['bg'], plot_bgcolor=C['panel_bg'],
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=20),
    )

    for i in range(1, 4):
        fig.update_xaxes(
            rangeslider_visible=False, showgrid=False,
            tickformat='%b\'%y', tickfont=dict(size=9, color=C['text_dim']),
            row=1, col=i,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=C['grid'], gridwidth=0.5,
            side='right', tickfont=dict(size=9, color=C['text_dim']),
            row=1, col=i,
        )

    return fig


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def chart_to_base64(fig: go.Figure, width: int = 1200, height: int = 700) -> Optional[str]:
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"[chart_engine] Export error: {e}")
        return None


def chart_to_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs='cdn', full_html=False)
