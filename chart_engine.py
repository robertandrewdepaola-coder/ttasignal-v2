"""
TTA v2 Chart Engine — Plotly Chart Generation
===============================================

Generates interactive candlestick charts with indicator overlays.
Uses signal_engine for all calculations. No Streamlit dependencies.

Charts:
- Main panel: Candlestick + SMA overlays + resistance levels
- Sub panel: AO histogram + MACD histogram + signal line
- Optional: Volume bars, divergence markers, entry/exit markers

Version: 2.0.0 (2026-02-08)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io

from signal_engine import (
    normalize_columns, calculate_macd, calculate_ao, calculate_atr,
    calculate_sma, add_all_indicators, detect_bearish_divergence,
    find_overhead_resistance, EntrySignal,
)


# =============================================================================
# COLOR SCHEME
# =============================================================================

COLORS = {
    'candle_up': '#26a69a',
    'candle_down': '#ef5350',
    'sma_30w': '#ff9800',     # Orange - 30-week / 150-day SMA
    'sma_50': '#2196f3',      # Blue - 50 SMA
    'sma_200': '#9c27b0',     # Purple - 200 SMA
    'ao_positive': '#26a69a',
    'ao_negative': '#ef5350',
    'macd_line': '#2962ff',
    'macd_signal': '#ff6d00',
    'macd_hist_pos': 'rgba(38, 166, 154, 0.5)',
    'macd_hist_neg': 'rgba(239, 83, 80, 0.5)',
    'volume_up': 'rgba(38, 166, 154, 0.3)',
    'volume_down': 'rgba(239, 83, 80, 0.3)',
    'resistance': 'rgba(239, 83, 80, 0.6)',
    'support': 'rgba(38, 166, 154, 0.6)',
    'entry_marker': '#00e676',
    'exit_win': '#00e676',
    'exit_loss': '#ff1744',
    'divergence': '#ff9800',
    'bg_dark': '#131722',
    'grid': '#1e222d',
    'text': '#d1d4dc',
}


# =============================================================================
# MAIN CHART BUILDER
# =============================================================================

def create_analysis_chart(df: pd.DataFrame,
                          ticker: str,
                          weekly_sma: pd.Series = None,
                          signal: EntrySignal = None,
                          show_volume: bool = True,
                          show_resistance: bool = True,
                          show_divergence: bool = True,
                          trade_markers: List[Dict] = None,
                          height: int = 700) -> go.Figure:
    """
    Create the main analysis chart.

    Layout:
    - Row 1 (60%): Candlestick + SMAs + resistance levels
    - Row 2 (20%): AO histogram
    - Row 3 (20%): MACD histogram + signal line

    If show_volume=True, volume bars are overlaid on row 1 with secondary y-axis.

    Args:
        df: Daily OHLCV DataFrame (should already have indicators from add_all_indicators)
        ticker: Symbol for title
        weekly_sma: 30-week SMA series (optional, for Weinstein overlay)
        signal: EntrySignal with pre-computed analysis (optional, for annotations)
        show_volume: Show volume bars
        show_resistance: Show overhead resistance levels
        show_divergence: Show bearish divergence markers
        trade_markers: List of dicts with entry/exit markers
        height: Chart height in pixels
    """
    df = normalize_columns(df).copy()

    # Ensure indicators exist
    if 'MACD' not in df.columns:
        df = add_all_indicators(df)

    # Detect divergence if requested
    if show_divergence:
        df = detect_bearish_divergence(df)

    # ── Build subplots ────────────────────────────────────────────────
    row_heights = [0.55, 0.22, 0.23]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=('', 'Awesome Oscillator', 'MACD (12/26/9 SMA)'),
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Price',
            increasing_line_color=COLORS['candle_up'],
            decreasing_line_color=COLORS['candle_down'],
        ),
        row=1, col=1
    )

    # ── Volume bars (overlay on row 1) ────────────────────────────────
    if show_volume and 'Volume' in df.columns:
        vol_colors = [COLORS['volume_up'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
                      else COLORS['volume_down'] for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'],
                name='Volume',
                marker_color=vol_colors,
                opacity=0.3,
                yaxis='y2',
            ),
            row=1, col=1
        )

    # ── 30-Week SMA (Weinstein) ───────────────────────────────────────
    if weekly_sma is not None and len(weekly_sma) > 0:
        chart_start = df.index.min()
        chart_end = df.index.max()
        filtered_sma = weekly_sma[(weekly_sma.index >= chart_start) &
                                   (weekly_sma.index <= chart_end)]
        if len(filtered_sma) > 0:
            sma_val = filtered_sma.iloc[-1]
            slope = "↗" if len(filtered_sma) >= 5 and filtered_sma.iloc[-1] > filtered_sma.iloc[-5] else "↘"
            fig.add_trace(
                go.Scatter(
                    x=filtered_sma.index, y=filtered_sma,
                    mode='lines',
                    name=f'30W SMA ${sma_val:.2f} {slope}',
                    line=dict(color=COLORS['sma_30w'], width=2.5),
                ),
                row=1, col=1
            )
    elif 'SMA_150' in df.columns:
        # Fallback: use 150-day SMA as proxy for 30-week
        sma150 = df['SMA_150'].dropna()
        if len(sma150) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sma150.index, y=sma150,
                    mode='lines',
                    name=f'150d SMA ${sma150.iloc[-1]:.2f}',
                    line=dict(color=COLORS['sma_30w'], width=2, dash='dot'),
                ),
                row=1, col=1
            )

    # ── 50 & 200 SMA ─────────────────────────────────────────────────
    if 'SMA_50' in df.columns:
        sma50 = df['SMA_50'].dropna()
        if len(sma50) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sma50.index, y=sma50,
                    mode='lines',
                    name=f'50 SMA ${sma50.iloc[-1]:.2f}',
                    line=dict(color=COLORS['sma_50'], width=1.5, dash='dash'),
                ),
                row=1, col=1
            )

    if 'SMA_200' in df.columns:
        sma200 = df['SMA_200'].dropna()
        if len(sma200) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sma200.index, y=sma200,
                    mode='lines',
                    name=f'200 SMA ${sma200.iloc[-1]:.2f}',
                    line=dict(color=COLORS['sma_200'], width=1.5, dash='dash'),
                ),
                row=1, col=1
            )

    # ── Overhead Resistance Levels ────────────────────────────────────
    if show_resistance and signal and signal.overhead_resistance:
        levels = signal.overhead_resistance.get('levels', [])
        critical = signal.overhead_resistance.get('critical_level', {})

        for lev in levels:
            price = lev['price']
            is_critical = critical and abs(price - critical.get('price', 0)) < 0.01
            color = '#ff1744' if is_critical else COLORS['resistance']
            width = 2 if is_critical else 1
            dash = 'solid' if is_critical else 'dot'

            fig.add_hline(
                y=price, row=1, col=1,
                line_dash=dash, line_color=color, line_width=width,
                annotation_text=f"R ${price:.0f}" + (" ★" if is_critical else ""),
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color=color,
            )

    # ── Trade Markers (entry/exit) ────────────────────────────────────
    if trade_markers:
        for marker in trade_markers:
            date = marker.get('date')
            price = marker.get('price')
            mtype = marker.get('type', 'entry')  # 'entry' or 'exit'
            label = marker.get('label', '')
            win = marker.get('win', True)

            if mtype == 'entry':
                color = COLORS['entry_marker']
                symbol = 'triangle-up'
            else:
                color = COLORS['exit_win'] if win else COLORS['exit_loss']
                symbol = 'triangle-down'

            fig.add_trace(
                go.Scatter(
                    x=[date], y=[price],
                    mode='markers+text',
                    name=label,
                    marker=dict(symbol=symbol, size=12, color=color),
                    text=[label],
                    textposition='top center' if mtype == 'entry' else 'bottom center',
                    textfont=dict(size=9, color=color),
                    showlegend=False,
                ),
                row=1, col=1
            )

    # ── Divergence Markers ────────────────────────────────────────────
    if show_divergence and 'bearish_div_detected' in df.columns:
        div_bars = df[df['bearish_div_detected'] == True]
        if len(div_bars) > 0:
            fig.add_trace(
                go.Scatter(
                    x=div_bars.index,
                    y=div_bars['High'] * 1.005,
                    mode='markers',
                    name='Bearish Divergence',
                    marker=dict(symbol='diamond', size=10,
                                color=COLORS['divergence'], line=dict(width=1, color='white')),
                    showlegend=True,
                ),
                row=1, col=1
            )

    # ── Row 2: Awesome Oscillator ─────────────────────────────────────
    if 'AO' in df.columns:
        ao = df['AO'].dropna()
        ao_colors = [COLORS['ao_positive'] if v >= 0 else COLORS['ao_negative']
                     for v in ao]
        fig.add_trace(
            go.Bar(
                x=ao.index, y=ao,
                name='AO',
                marker_color=ao_colors,
            ),
            row=2, col=1
        )
        # Zero line
        fig.add_hline(y=0, row=2, col=1, line_dash='solid',
                      line_color='rgba(255,255,255,0.2)', line_width=0.5)

    # ── Row 3: MACD ───────────────────────────────────────────────────
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = df['MACD'].dropna()
        signal_line = df['MACD_Signal'].dropna()
        hist = df['MACD_Hist'].dropna()

        # Histogram
        hist_colors = [COLORS['macd_hist_pos'] if v >= 0 else COLORS['macd_hist_neg']
                       for v in hist]
        fig.add_trace(
            go.Bar(x=hist.index, y=hist, name='MACD Hist', marker_color=hist_colors),
            row=3, col=1
        )

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=macd.index, y=macd, mode='lines',
                name='MACD', line=dict(color=COLORS['macd_line'], width=1.5),
            ),
            row=3, col=1
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=signal_line.index, y=signal_line, mode='lines',
                name='Signal (SMA 9)', line=dict(color=COLORS['macd_signal'], width=1.5),
            ),
            row=3, col=1
        )

        # Zero line
        fig.add_hline(y=0, row=3, col=1, line_dash='solid',
                      line_color='rgba(255,255,255,0.2)', line_width=0.5)

    # ── Layout ────────────────────────────────────────────────────────
    current_price = float(df['Close'].iloc[-1])
    fig.update_layout(
        title=dict(
            text=f"{ticker} — ${current_price:.2f}",
            font=dict(size=16, color=COLORS['text']),
        ),
        height=height,
        template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'],
        plot_bgcolor=COLORS['bg_dark'],
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=10, color=COLORS['text']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=60, r=20, t=60, b=20),
    )

    # Grid styling
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor=COLORS['grid'], zeroline=False, row=i, col=1,
            showticklabels=(i == 3),  # Only show x-axis labels on bottom
        )
        fig.update_yaxes(
            gridcolor=COLORS['grid'], zeroline=False, row=i, col=1,
        )

    # Volume y-axis (secondary on row 1)
    if show_volume:
        fig.update_layout(
            yaxis2=dict(
                overlaying='y', side='right',
                showgrid=False, showticklabels=False,
                range=[0, df['Volume'].max() * 4] if 'Volume' in df.columns else [0, 1],
            )
        )

    return fig


# =============================================================================
# MINI CHART — For scanner table (sparkline-style)
# =============================================================================

def create_mini_chart(df: pd.DataFrame, ticker: str,
                      width: int = 300, height: int = 150) -> go.Figure:
    """
    Create a small sparkline chart for scanner table display.
    Shows last 60 days of price + 50 SMA.
    """
    df = normalize_columns(df).copy()
    recent = df.tail(60)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index, y=recent['Close'],
            mode='lines', name='Price',
            line=dict(color=COLORS['candle_up'] if recent['Close'].iloc[-1] >= recent['Close'].iloc[0]
                      else COLORS['candle_down'], width=1.5),
            fill='tozeroy',
            fillcolor='rgba(38, 166, 154, 0.1)' if recent['Close'].iloc[-1] >= recent['Close'].iloc[0]
            else 'rgba(239, 83, 80, 0.1)',
        )
    )

    fig.update_layout(
        height=height, width=width,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


# =============================================================================
# MULTI-TIMEFRAME CHART GRID
# =============================================================================

def create_mtf_chart(daily_df: pd.DataFrame,
                     weekly_df: pd.DataFrame,
                     monthly_df: pd.DataFrame,
                     ticker: str,
                     height: int = 500) -> go.Figure:
    """
    Create a 3-panel multi-timeframe view: Daily | Weekly | Monthly.
    Each panel shows candlestick + MACD status.
    """
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.03,
        subplot_titles=('Daily', 'Weekly', 'Monthly'),
    )

    for idx, (label, df) in enumerate([(
        'Daily', daily_df), ('Weekly', weekly_df), ('Monthly', monthly_df)
    ], start=1):
        if df is None or df.empty:
            continue

        df = normalize_columns(df).copy()
        df = calculate_macd(df)
        recent = df.tail(60 if label == 'Daily' else (26 if label == 'Weekly' else 24))

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=recent.index, open=recent['Open'], high=recent['High'],
                low=recent['Low'], close=recent['Close'],
                name=label,
                increasing_line_color=COLORS['candle_up'],
                decreasing_line_color=COLORS['candle_down'],
                showlegend=False,
            ),
            row=1, col=idx
        )

        # MACD status annotation
        if 'MACD' in recent.columns and 'MACD_Signal' in recent.columns:
            m = float(recent['MACD'].iloc[-1])
            s = float(recent['MACD_Signal'].iloc[-1])
            if not (pd.isna(m) or pd.isna(s)):
                bullish = m > s
                status = "MACD ✅" if bullish else "MACD ❌"
                color = COLORS['candle_up'] if bullish else COLORS['candle_down']
                fig.add_annotation(
                    text=status, xref=f'x{idx}', yref=f'y{idx}',
                    x=0.5, y=1.05, xanchor='center', yanchor='bottom',
                    showarrow=False, font=dict(size=11, color=color),
                )

    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'],
        plot_bgcolor=COLORS['bg_dark'],
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=20),
    )

    for i in range(1, 4):
        fig.update_xaxes(rangeslider_visible=False, gridcolor=COLORS['grid'], row=1, col=i)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=1, col=i)

    return fig


# =============================================================================
# CHART EXPORT
# =============================================================================

def chart_to_base64(fig: go.Figure, width: int = 1200, height: int = 700) -> Optional[str]:
    """
    Export chart as base64 PNG string (for AI analysis or PDF reports).
    Requires kaleido package.
    """
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"[chart_engine] Error exporting chart: {e}")
        return None


def chart_to_html(fig: go.Figure) -> str:
    """Export chart as standalone HTML string."""
    return fig.to_html(include_plotlyjs='cdn', full_html=False)
