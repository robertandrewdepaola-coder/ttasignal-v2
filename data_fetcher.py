"""
TTA v2 Data Fetcher — Centralized Data Access
===============================================

ALL yfinance calls go through this module. No other module touches yfinance.

Features:
- Session-level caching (avoid redundant API calls within a scan)
- Standardized periods (defined in signal_engine constants)
- Column normalization (handled once at fetch time)
- New data sources: options, insider, institutional, earnings
- Market filter data (SPY, VIX)

Version: 2.0.0 (2026-02-07)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import time

from signal_engine import (
    DAILY_PERIOD, WEEKLY_PERIOD, MONTHLY_PERIOD,
    MARKET_FILTER_SPY_SMA, MARKET_FILTER_VIX_MAX,
    normalize_columns,
)


# =============================================================================
# CACHE — Avoid redundant API calls within a session
# =============================================================================

class DataCache:
    """
    Simple in-memory cache for yfinance data.
    
    Keyed by (ticker, interval, period). Expires after `ttl` seconds.
    Call cache.clear() at the start of each scan to ensure fresh data.
    """
    
    def __init__(self, ttl: int = 300):
        """ttl: cache lifetime in seconds (default 5 minutes)."""
        self._store: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            age = time.time() - self._timestamps.get(key, 0)
            if age < self.ttl:
                return self._store[key]
            else:
                # Expired
                del self._store[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        self._store[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached data. Call at start of new scan."""
        self._store.clear()
        self._timestamps.clear()
    
    def stats(self) -> Dict[str, int]:
        return {
            'entries': len(self._store),
            'expired': sum(1 for k in self._timestamps
                          if time.time() - self._timestamps[k] >= self.ttl)
        }


# Module-level cache instance
_cache = DataCache(ttl=300)


def clear_cache():
    """Clear the data cache. Call at start of each scan."""
    _cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return _cache.stats()


# =============================================================================
# CORE PRICE DATA — Daily, Weekly, Monthly
# =============================================================================

def fetch_daily(ticker: str, period: str = DAILY_PERIOD) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data for a ticker.
    
    Returns normalized DataFrame with columns:
    Open, High, Low, Close, Volume
    
    Returns None on error.
    """
    cache_key = f"{ticker}:daily:{period}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1d')
        
        if df is None or df.empty:
            return None
        
        df = normalize_columns(df)
        _cache.set(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[data_fetcher] Error fetching daily {ticker}: {e}")
        return None


def fetch_weekly(ticker: str, period: str = WEEKLY_PERIOD) -> Optional[pd.DataFrame]:
    """Fetch weekly OHLCV data."""
    cache_key = f"{ticker}:weekly:{period}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1wk')
        
        if df is None or df.empty:
            return None
        
        df = normalize_columns(df)
        _cache.set(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[data_fetcher] Error fetching weekly {ticker}: {e}")
        return None


def fetch_monthly(ticker: str, period: str = MONTHLY_PERIOD) -> Optional[pd.DataFrame]:
    """Fetch monthly OHLCV data."""
    cache_key = f"{ticker}:monthly:{period}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1mo')
        
        if df is None or df.empty:
            return None
        
        df = normalize_columns(df)
        _cache.set(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[data_fetcher] Error fetching monthly {ticker}: {e}")
        return None


def fetch_intraday(ticker: str, interval: str = '1h',
                   period: str = '5d') -> Optional[pd.DataFrame]:
    """Fetch intraday data (for 4h divergence checks etc)."""
    cache_key = f"{ticker}:intraday:{interval}:{period}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df is None or df.empty:
            return None
        
        df = normalize_columns(df)
        _cache.set(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[data_fetcher] Error fetching intraday {ticker}: {e}")
        return None


def fetch_history(ticker: str, start: str = None, end: str = None,
                  interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Fetch historical data with explicit start/end dates.
    Used for backtesting with specific date ranges.
    """
    cache_key = f"{ticker}:hist:{start}:{end}:{interval}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        
        if df is None or df.empty:
            return None
        
        df = normalize_columns(df)
        _cache.set(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[data_fetcher] Error fetching history {ticker}: {e}")
        return None


# =============================================================================
# CURRENT PRICE — Single value, fast
# =============================================================================

def fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch current/last close price. Fast single-value fetch."""
    # Try to get from cached daily data first
    cache_key = f"{ticker}:daily:{DAILY_PERIOD}"
    cached = _cache.get(cache_key)
    if cached is not None and len(cached) > 0:
        return float(cached['Close'].iloc[-1])
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2d')
        if hist is not None and not hist.empty:
            hist = normalize_columns(hist)
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"[data_fetcher] Error fetching price {ticker}: {e}")
        return None


# =============================================================================
# MARKET FILTER — SPY + VIX
# =============================================================================

def fetch_market_filter() -> Dict[str, Any]:
    """
    Fetch market filter data: SPY vs 200 SMA, VIX level.
    
    Returns dict with:
        spy_above_200, spy_close, spy_sma200,
        vix_below_30, vix_close
    """
    cache_key = "MARKET_FILTER"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'spy_above_200': True,  # Default to passing on error
        'spy_close': None,
        'spy_sma200': None,
        'vix_below_30': True,
        'vix_close': None,
    }
    
    try:
        # SPY
        spy_df = fetch_daily("SPY")
        if spy_df is not None and len(spy_df) >= 200:
            spy_close = float(spy_df['Close'].iloc[-1])
            spy_sma200 = float(spy_df['Close'].rolling(MARKET_FILTER_SPY_SMA).mean().iloc[-1])
            result['spy_close'] = round(spy_close, 2)
            result['spy_sma200'] = round(spy_sma200, 2)
            result['spy_above_200'] = spy_close > spy_sma200
        
        # VIX
        vix_df = fetch_daily("^VIX", period='5d')
        if vix_df is not None and not vix_df.empty:
            vix_close = float(vix_df['Close'].iloc[-1])
            result['vix_close'] = round(vix_close, 2)
            result['vix_below_30'] = vix_close < MARKET_FILTER_VIX_MAX
    
    except Exception as e:
        print(f"[data_fetcher] Error fetching market filter: {e}")
    
    _cache.set(cache_key, result)
    return result


def fetch_spy_daily() -> Optional[pd.DataFrame]:
    """Fetch SPY daily data (used for relative strength calculations)."""
    return fetch_daily("SPY")


# =============================================================================
# SECTOR ROTATION — Sector ETF performance vs SPY
# =============================================================================

# Map sectors to ETFs
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
    'Basic Materials': 'XLB',
}

# Short sector labels for display
SECTOR_SHORT = {
    'Technology': 'Tech',
    'Healthcare': 'Health',
    'Financial Services': 'Fin',
    'Financials': 'Fin',
    'Consumer Cyclical': 'ConDisc',
    'Consumer Defensive': 'ConStap',
    'Industrials': 'Indust',
    'Energy': 'Energy',
    'Materials': 'Mater',
    'Real Estate': 'RE',
    'Utilities': 'Util',
    'Communication Services': 'Comm',
    'Basic Materials': 'Mater',
}


def fetch_sector_rotation() -> Dict[str, Dict]:
    """
    Fetch sector rotation data — which sectors are leading/lagging vs SPY.

    Returns dict keyed by sector name:
        {sector: {etf, perf_1d, perf_5d, perf_20d, vs_spy_20d, vs_spy_5d,
                  status, phase, short_name, spy_perf_20d}}

    Status: 'leading' (green), 'neutral' (yellow), 'lagging' (red)
    Phase:  'LEADING'  — outperforming on both 5d and 20d
            'EMERGING' — 5d accelerating, 20d still catching up (rotation IN)
            'FADING'   — 20d still up, but 5d momentum dying (rotation OUT)
            'LAGGING'  — underperforming on both timeframes
    """
    cache_key = "SECTOR_ROTATION"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    rotation = {}

    try:
        # Fetch SPY performance across timeframes
        spy_df = fetch_daily("SPY", period='3mo')
        if spy_df is None or len(spy_df) < 20:
            return rotation

        spy_perf_1 = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-2] - 1) * 100 if len(spy_df) >= 2 else 0
        spy_perf_5 = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-5] - 1) * 100
        spy_perf_20 = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-20] - 1) * 100

        # Fetch each sector ETF
        unique_etfs = set(SECTOR_ETF_MAP.values())
        etf_perf = {}

        for etf in unique_etfs:
            try:
                df = fetch_daily(etf, period='3mo')
                if df is not None and len(df) >= 20:
                    perf_1 = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
                    perf_5 = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                    perf_20 = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                    etf_perf[etf] = {
                        'perf_1d': perf_1,
                        'perf_5d': perf_5,
                        'perf_20d': perf_20,
                    }
            except Exception:
                pass

        # Build rotation map with momentum phase classification
        # Use RELATIVE ranking so there's always a distribution even in broad rallies
        sector_scores = []
        for sector, etf in SECTOR_ETF_MAP.items():
            if etf in etf_perf:
                p = etf_perf[etf]
                vs_spy_20 = p['perf_20d'] - spy_perf_20
                vs_spy_5 = p['perf_5d'] - spy_perf_5

                # Legacy status (for backward compat)
                if vs_spy_20 > 2.0 and p['perf_5d'] > 0:
                    status = 'leading'
                elif vs_spy_20 < -2.0 and p['perf_5d'] < 0:
                    status = 'lagging'
                else:
                    status = 'neutral'

                # Composite score for ranking: weight 5d momentum slightly more
                composite = vs_spy_5 * 0.6 + vs_spy_20 * 0.4

                sector_scores.append({
                    'sector': sector, 'etf': etf,
                    'perf_1d': round(p['perf_1d'], 2),
                    'perf_5d': round(p['perf_5d'], 2),
                    'perf_20d': round(p['perf_20d'], 1),
                    'vs_spy_20d': round(vs_spy_20, 1),
                    'vs_spy_5d': round(vs_spy_5, 2),
                    'spy_perf_20d': round(spy_perf_20, 1),
                    'spy_perf_5d': round(spy_perf_5, 2),
                    'status': status,
                    'composite': composite,
                    'short_name': SECTOR_SHORT.get(sector, sector[:4]),
                })

        # De-duplicate by ETF (keep first occurrence)
        seen_etfs = set()
        unique_sectors = []
        for s in sector_scores:
            if s['etf'] not in seen_etfs:
                seen_etfs.add(s['etf'])
                unique_sectors.append(s)

        # Rank by composite score — split into quartiles
        unique_sectors.sort(key=lambda x: x['composite'], reverse=True)
        n = len(unique_sectors)
        for i, s in enumerate(unique_sectors):
            pct = i / max(n - 1, 1)  # 0 = top, 1 = bottom
            if pct <= 0.25:
                phase = 'LEADING'    # Top ~25% — trade these NOW
            elif pct <= 0.50:
                phase = 'EMERGING'   # 25-50% — money rotating IN, watch
            elif pct <= 0.75:
                phase = 'FADING'     # 50-75% — money rotating OUT, tighten
            else:
                phase = 'LAGGING'    # Bottom ~25% — avoid

            s['phase'] = phase
            rotation[s['sector']] = s

        # Add aliases so both yfinance sector names resolve
        # (e.g., yfinance returns 'Financials' but SECTOR_ETF_MAP has 'Financial Services')
        alias_map = {}
        for sector, etf in SECTOR_ETF_MAP.items():
            if sector not in rotation:
                # Find the canonical entry for this ETF
                for canon_sector, info in rotation.items():
                    if info.get('etf') == etf:
                        alias_map[sector] = canon_sector
                        break
        for alias, canonical in alias_map.items():
            rotation[alias] = rotation[canonical]

    except Exception as e:
        print(f"[data_fetcher] Sector rotation error: {e}")

    _cache.set(cache_key, rotation)
    return rotation


def fetch_batch_earnings_flags(tickers: list, days_ahead: int = 14) -> Dict[str, Dict]:
    """
    Check earnings dates for a batch of tickers using parallel threading.

    Returns dict keyed by ticker:
        {ticker: {next_earnings: date_str, days_until: int, within_window: bool}}

    Uses ThreadPoolExecutor with 5 workers to balance speed vs rate-limiting.
    Three strategies per ticker: info dict → calendar → earnings_dates.
    """
    cache_key = "BATCH_EARNINGS_FLAGS"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    from concurrent.futures import ThreadPoolExecutor, as_completed

    today = datetime.now().date()

    def _fetch_one_earnings(ticker: str):
        """Fetch earnings date for a single ticker with 3 fallback strategies."""
        try:
            stock = yf.Ticker(ticker)
            earn_dt = None

            # Strategy 1: info dict (often has earningsTimestamp — fast, same API call)
            try:
                info = stock.info
                if info:
                    # Try multiple possible keys
                    for key in ('earningsTimestamp', 'earningsTimestampStart', 'mostRecentQuarter'):
                        ts = info.get(key)
                        if ts and isinstance(ts, (int, float)) and ts > 0:
                            from datetime import timezone
                            candidate = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                            # earningsTimestamp is usually future; mostRecentQuarter is past
                            if key == 'mostRecentQuarter':
                                continue  # Skip past dates from this key
                            if (candidate - today).days >= -7:
                                earn_dt = candidate
                                break
                    # Also check string date format
                    if earn_dt is None:
                        for key in ('earningsDate',):
                            val = info.get(key)
                            if val:
                                if isinstance(val, list) and val:
                                    val = val[0]
                                if hasattr(val, 'date') and callable(getattr(val, 'date', None)):
                                    candidate = val.date()
                                    if (candidate - today).days >= -7:
                                        earn_dt = candidate
            except Exception:
                pass

            # Strategy 2: calendar property
            if earn_dt is None:
                try:
                    cal = stock.calendar
                    if cal is not None:
                        raw_date = None
                        if isinstance(cal, dict):
                            raw_date = cal.get('Earnings Date')
                            if isinstance(raw_date, list) and raw_date:
                                raw_date = raw_date[0]
                        elif isinstance(cal, pd.DataFrame) and not cal.empty:
                            try:
                                raw_date = cal.iloc[0].get('Earnings Date')
                            except Exception:
                                pass

                        if raw_date is not None:
                            if hasattr(raw_date, 'date') and callable(getattr(raw_date, 'date', None)):
                                earn_dt = raw_date.date()
                            elif isinstance(raw_date, str) and len(raw_date) >= 10:
                                earn_dt = datetime.strptime(raw_date[:10], '%Y-%m-%d').date()
                except Exception:
                    pass

            # Strategy 3: earnings_dates (most reliable but slowest)
            if earn_dt is None:
                try:
                    edates = stock.earnings_dates
                    if edates is not None and len(edates) > 0:
                        for dt_idx in sorted(edates.index):
                            try:
                                if hasattr(dt_idx, 'date') and callable(getattr(dt_idx, 'date')):
                                    d = dt_idx.date()
                                elif hasattr(dt_idx, 'to_pydatetime'):
                                    d = dt_idx.to_pydatetime().date()
                                else:
                                    continue
                                if (d - today).days >= -7:
                                    earn_dt = d
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass

            if earn_dt is not None:
                days_until = (earn_dt - today).days
                if days_until >= -7:
                    return (ticker, {
                        'next_earnings': earn_dt.strftime('%Y-%m-%d'),
                        'days_until': days_until,
                        'within_window': 0 <= days_until <= days_ahead,
                    })
        except Exception:
            pass
        return (ticker, None)

    # Run in parallel — 3 workers (reduced to avoid rate limiting), 8s per ticker
    flags = {}
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_fetch_one_earnings, t): t for t in tickers}
            for future in as_completed(futures, timeout=90):
                try:
                    ticker, result = future.result(timeout=8)
                    if result:
                        flags[ticker] = result
                except Exception:
                    pass
    except Exception as e:
        print(f"[data_fetcher] Batch earnings error: {e}")

    # If batch got very few results (possible rate-limiting), retry missing ones sequentially
    if len(flags) < len(tickers) * 0.3 and len(tickers) > 1:
        import time
        time.sleep(2)  # Back off before retry
        missing = [t for t in tickers if t not in flags]
        for t in missing[:10]:  # Cap at 10 retries
            try:
                _, result = _fetch_one_earnings(t)
                if result:
                    flags[t] = result
                time.sleep(0.5)  # Stagger
            except Exception:
                pass

    # Only cache if we got meaningful results (don't cache empty on failure)
    if flags:
        _cache.set(cache_key, flags)
    return flags


def get_ticker_sector(ticker: str) -> Optional[str]:
    """Get sector for a ticker — with static fallback when yfinance is rate-limited."""
    # Try yfinance first
    info = fetch_ticker_info(ticker)
    sector = info.get('sector')
    if sector:
        return sector

    # Fallback: static sector map for common tickers
    return _SECTOR_FALLBACK.get(ticker.upper())


# Static sector map — covers top ~150 most traded US tickers
# Used when yfinance is rate-limited or returns no data
_SECTOR_FALLBACK = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
    'GOOG': 'Technology', 'META': 'Technology', 'AVGO': 'Technology', 'ORCL': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CSCO': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology',
    'LRCX': 'Technology', 'KLAC': 'Technology', 'MU': 'Technology', 'MRVL': 'Technology',
    'ADI': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology', 'NXPI': 'Technology',
    'MCHP': 'Technology', 'ON': 'Technology', 'SWKS': 'Technology', 'MPWR': 'Technology',
    'NOW': 'Technology', 'PANW': 'Technology', 'CRWD': 'Technology', 'SNOW': 'Technology',
    'PLTR': 'Technology', 'NET': 'Technology', 'DDOG': 'Technology', 'ZS': 'Technology',
    'FTNT': 'Technology', 'WDAY': 'Technology', 'TEAM': 'Technology', 'HUBS': 'Technology',
    'SHOP': 'Technology', 'SQ': 'Technology', 'UBER': 'Technology', 'DASH': 'Technology',
    'COIN': 'Technology', 'MSTR': 'Technology', 'DELL': 'Technology', 'HPE': 'Technology',
    'IBM': 'Technology', 'SMCI': 'Technology', 'ARM': 'Technology', 'TSM': 'Technology',
    'ASML': 'Technology', 'SAP': 'Technology',
    # Communication Services
    'NFLX': 'Communication Services', 'DIS': 'Communication Services',
    'CMCSA': 'Communication Services', 'T': 'Communication Services',
    'VZ': 'Communication Services', 'TMUS': 'Communication Services',
    'SPOT': 'Communication Services', 'ROKU': 'Communication Services',
    'SNAP': 'Communication Services', 'PINS': 'Communication Services',
    'RBLX': 'Communication Services', 'TTWO': 'Communication Services',
    'EA': 'Communication Services', 'WBD': 'Communication Services',
    # Consumer Discretionary
    'AMZN': 'Consumer Cyclical', 'TSLA': 'Consumer Cyclical', 'HD': 'Consumer Cyclical',
    'MCD': 'Consumer Cyclical', 'NKE': 'Consumer Cyclical', 'SBUX': 'Consumer Cyclical',
    'LOW': 'Consumer Cyclical', 'TJX': 'Consumer Cyclical', 'BKNG': 'Consumer Cyclical',
    'CMG': 'Consumer Cyclical', 'ABNB': 'Consumer Cyclical', 'RCL': 'Consumer Cyclical',
    'GM': 'Consumer Cyclical', 'F': 'Consumer Cyclical', 'RIVN': 'Consumer Cyclical',
    'LCID': 'Consumer Cyclical', 'LULU': 'Consumer Cyclical', 'ROST': 'Consumer Cyclical',
    # Consumer Staples
    'WMT': 'Consumer Defensive', 'PG': 'Consumer Defensive', 'COST': 'Consumer Defensive',
    'KO': 'Consumer Defensive', 'PEP': 'Consumer Defensive', 'PM': 'Consumer Defensive',
    'MO': 'Consumer Defensive', 'CL': 'Consumer Defensive', 'KMB': 'Consumer Defensive',
    'MDLZ': 'Consumer Defensive', 'GIS': 'Consumer Defensive', 'KHC': 'Consumer Defensive',
    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare',
    'VRTX': 'Healthcare', 'BMY': 'Healthcare', 'MDT': 'Healthcare', 'SYK': 'Healthcare',
    'REGN': 'Healthcare', 'ZTS': 'Healthcare', 'MRNA': 'Healthcare', 'BIIB': 'Healthcare',
    # Financials
    'BRK.B': 'Financial Services', 'JPM': 'Financial Services', 'V': 'Financial Services',
    'MA': 'Financial Services', 'BAC': 'Financial Services', 'WFC': 'Financial Services',
    'GS': 'Financial Services', 'MS': 'Financial Services', 'BLK': 'Financial Services',
    'SCHW': 'Financial Services', 'AXP': 'Financial Services', 'C': 'Financial Services',
    'USB': 'Financial Services', 'PNC': 'Financial Services', 'PYPL': 'Financial Services',
    'SOFI': 'Financial Services', 'HOOD': 'Financial Services',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
    'SLB': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy', 'FANG': 'Energy',
    # Industrials
    'GE': 'Industrials', 'CAT': 'Industrials', 'RTX': 'Industrials', 'HON': 'Industrials',
    'UNP': 'Industrials', 'BA': 'Industrials', 'DE': 'Industrials', 'LMT': 'Industrials',
    'GD': 'Industrials', 'NOC': 'Industrials', 'MMM': 'Industrials', 'WM': 'Industrials',
    'UPS': 'Industrials', 'FDX': 'Industrials', 'STRL': 'Industrials',
    # Materials
    'LIN': 'Basic Materials', 'APD': 'Basic Materials', 'SHW': 'Basic Materials',
    'FCX': 'Basic Materials', 'NEM': 'Basic Materials', 'NUE': 'Basic Materials',
    'GOLD': 'Basic Materials', 'X': 'Basic Materials', 'CLF': 'Basic Materials',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'SRE': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities',
    'CEG': 'Utilities', 'VST': 'Utilities',
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate',
    'PSA': 'Real Estate', 'DLR': 'Real Estate',
}


# =============================================================================
# MACRO NARRATIVE DATA — Morning Briefing
# =============================================================================

def fetch_macro_narrative_data() -> Dict[str, Any]:
    """
    Fetch broad market data for AI-generated morning narrative.

    Returns dict with:
    - indices: SPY, QQQ, IWM performance (1d, 5d, 20d)
    - breadth: RSP vs SPY (equal-weight vs cap-weight)
    - vix: level, 5-day change, regime
    - sectors: offensive vs defensive rotation
    - macro: 10Y yield proxy (TLT), dollar index (UUP)
    """
    cache_key = "MACRO_NARRATIVE"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = {
        'indices': {},
        'breadth': {},
        'vix': {},
        'sectors': {},
        'macro': {},
        'timestamp': datetime.now().isoformat(),
    }

    def _perf(df, periods):
        """Calculate returns for multiple periods."""
        if df is None or df.empty:
            return {}
        close = df['Close']
        out = {'price': round(float(close.iloc[-1]), 2)}
        for label, n in periods:
            if len(close) >= n:
                out[label] = round((close.iloc[-1] / close.iloc[-n] - 1) * 100, 2)
        return out

    periods = [('1d', 2), ('5d', 5), ('20d', 20)]

    # ── Major Indices ─────────────────────────────────────────────────
    for sym, name in [('SPY', 'S&P 500'), ('QQQ', 'Nasdaq 100'), ('IWM', 'Russell 2000')]:
        try:
            df = fetch_daily(sym, period='3mo')
            result['indices'][name] = _perf(df, periods)
        except Exception:
            pass

    # ── Market Breadth: RSP (equal weight S&P) vs SPY ─────────────────
    try:
        rsp_df = fetch_daily('RSP', period='3mo')
        spy_df = fetch_daily('SPY', period='3mo')
        if rsp_df is not None and spy_df is not None and len(rsp_df) >= 20:
            rsp_20d = (rsp_df['Close'].iloc[-1] / rsp_df['Close'].iloc[-20] - 1) * 100
            spy_20d = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-20] - 1) * 100
            spread = rsp_20d - spy_20d
            result['breadth'] = {
                'rsp_20d': round(rsp_20d, 2),
                'spy_20d': round(spy_20d, 2),
                'spread': round(spread, 2),
                'regime': 'Broad participation' if spread > 1 else (
                    'Narrow leadership' if spread < -1 else 'Neutral'),
            }
    except Exception:
        pass

    # ── VIX ───────────────────────────────────────────────────────────
    try:
        vix_df = fetch_daily('^VIX', period='1mo')
        if vix_df is not None and len(vix_df) >= 5:
            vix_now = float(vix_df['Close'].iloc[-1])
            vix_5d_ago = float(vix_df['Close'].iloc[-5])
            vix_chg = vix_now - vix_5d_ago

            if vix_now < 15:
                regime = 'Low volatility (complacency)'
            elif vix_now < 20:
                regime = 'Normal'
            elif vix_now < 25:
                regime = 'Elevated'
            elif vix_now < 30:
                regime = 'High fear'
            else:
                regime = 'Extreme fear'

            result['vix'] = {
                'level': round(vix_now, 2),
                'change_5d': round(vix_chg, 2),
                'regime': regime,
            }
    except Exception:
        pass

    # ── Sector Rotation: Offensive vs Defensive ───────────────────────
    try:
        offensive = ['XLK', 'XLY', 'XLC']  # Tech, Consumer Disc, Comm
        defensive = ['XLU', 'XLP', 'XLV']  # Utilities, Staples, Healthcare

        off_perf = []
        def_perf = []

        for etf in offensive:
            df = fetch_daily(etf, period='3mo')
            if df is not None and len(df) >= 20:
                p = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                off_perf.append(p)

        for etf in defensive:
            df = fetch_daily(etf, period='3mo')
            if df is not None and len(df) >= 20:
                p = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                def_perf.append(p)

        if off_perf and def_perf:
            avg_off = sum(off_perf) / len(off_perf)
            avg_def = sum(def_perf) / len(def_perf)
            spread = avg_off - avg_def

            result['sectors'] = {
                'offensive_avg_20d': round(avg_off, 2),
                'defensive_avg_20d': round(avg_def, 2),
                'spread': round(spread, 2),
                'regime': 'Risk-On' if spread > 2 else (
                    'Risk-Off' if spread < -2 else 'Balanced'),
            }
    except Exception:
        pass

    # ── Macro Gauges: Bond proxy (TLT) and Dollar (UUP) ──────────────
    try:
        for sym, name in [('TLT', '20Y Bond'), ('UUP', 'Dollar')]:
            df = fetch_daily(sym, period='3mo')
            if df is not None and len(df) >= 20:
                result['macro'][name] = _perf(df, periods)
    except Exception:
        pass

    _cache.set(cache_key, result)
    return result


def fetch_signal_for_exit(ticker: str) -> Optional[Dict]:
    """
    Fetch minimal signal data for exit advisor.
    Returns dict with macd, ao, weekly, monthly state.
    """
    try:
        daily = fetch_daily(ticker)
        weekly = fetch_weekly(ticker)
        monthly = fetch_monthly(ticker)

        if daily is None or len(daily) < 50:
            return None

        # Import signal engine functions
        from signal_engine import detect_macd_cross, detect_ao_state

        macd_state = detect_macd_cross(daily)
        ao_state = detect_ao_state(daily)

        weekly_state = {}
        if weekly is not None and len(weekly) >= 26:
            weekly_state = detect_macd_cross(weekly)

        monthly_state = {}
        if monthly is not None and len(monthly) >= 26:
            monthly_state = detect_macd_cross(monthly)

        return {
            'macd': macd_state,
            'ao': ao_state,
            'weekly': weekly_state,
            'monthly': monthly_state,
        }
    except Exception as e:
        print(f"[data_fetcher] Signal for exit error ({ticker}): {e}")
        return None


# =============================================================================
# FUNDAMENTAL DATA — Ticker info, sector, earnings
# =============================================================================

def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch ticker fundamental info from yfinance.
    
    Returns dict with sector, industry, market cap, short interest,
    52-week range, earnings date, etc.
    """
    cache_key = f"{ticker}:info"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'sector': None,
        'industry': None,
        'market_cap': None,
        'short_pct_float': None,
        'fifty_two_week_high': None,
        'fifty_two_week_low': None,
        'avg_volume': None,
        'earnings_date': None,
        'forward_pe': None,
        'trailing_pe': None,
        'error': None,
    }
    
    try:
        stock = yf.Ticker(ticker)
        info = None
        
        # Try up to 2 times with delay if rate-limited
        for _attempt in range(2):
            try:
                info = stock.info
                if info and info.get('sector'):
                    break  # Got real data
                if _attempt == 0:
                    import time
                    time.sleep(1.5)
            except Exception:
                if _attempt == 0:
                    import time
                    time.sleep(1.5)
                break
        
        if info:
            result['sector'] = info.get('sector')
            result['industry'] = info.get('industry')
            result['market_cap'] = info.get('marketCap')
            result['short_pct_float'] = info.get('shortPercentOfFloat')
            result['fifty_two_week_high'] = info.get('fiftyTwoWeekHigh')
            result['fifty_two_week_low'] = info.get('fiftyTwoWeekLow')
            result['avg_volume'] = info.get('averageVolume')
            result['forward_pe'] = info.get('forwardPE')
            result['trailing_pe'] = info.get('trailingPE')
            
            # Earnings date (can be a list or single value)
            earnings = info.get('earningsTimestamp')
            if earnings:
                result['earnings_date'] = datetime.fromtimestamp(earnings).strftime('%Y-%m-%d')
    
    except Exception as e:
        result['error'] = str(e)
    
    _cache.set(cache_key, result)
    return result


def fetch_fundamental_profile(ticker: str) -> Dict[str, Any]:
    """
    Fetch banker-grade fundamental profile from yfinance.

    Returns revenue growth, margins, cash flow, debt, valuation multiples,
    business description, and peer context for AI synthesis.
    """
    cache_key = f"{ticker}:fundamentals"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    profile = {
        'error': None,
        # Identity
        'name': None, 'sector': None, 'industry': None,
        'business_summary': None, 'market_cap': None,
        'employees': None, 'country': None,
        # Valuation
        'trailing_pe': None, 'forward_pe': None,
        'peg_ratio': None, 'price_to_book': None,
        'price_to_sales': None, 'ev_to_ebitda': None,
        'ev_to_revenue': None,
        # Growth
        'revenue_growth_yoy': None, 'earnings_growth_yoy': None,
        'revenue_growth_quarterly': None, 'earnings_growth_quarterly': None,
        # Profitability
        'gross_margin': None, 'operating_margin': None,
        'profit_margin': None, 'ebitda_margin': None,
        'return_on_equity': None, 'return_on_assets': None,
        # Financial Health
        'total_revenue': None, 'ebitda': None, 'net_income': None,
        'total_debt': None, 'total_cash': None,
        'debt_to_equity': None, 'current_ratio': None,
        'free_cash_flow': None, 'operating_cash_flow': None,
        # Shareholder
        'dividend_yield': None, 'payout_ratio': None,
        'shares_outstanding': None, 'float_shares': None,
        'insider_pct': None, 'institution_pct': None,
        'short_pct_float': None, 'short_ratio': None,
        # Earnings
        'next_earnings': None,
        'last_earnings_surprise_pct': None,
        # Per-share
        'revenue_per_share': None, 'book_value': None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Identity
        profile['name'] = info.get('longName') or info.get('shortName')
        profile['sector'] = info.get('sector')
        profile['industry'] = info.get('industry')
        profile['country'] = info.get('country')
        profile['employees'] = info.get('fullTimeEmployees')
        summary = info.get('longBusinessSummary', '')
        profile['business_summary'] = summary[:500] if summary else None
        profile['market_cap'] = info.get('marketCap')

        # Valuation
        profile['trailing_pe'] = info.get('trailingPE')
        profile['forward_pe'] = info.get('forwardPE')
        profile['peg_ratio'] = info.get('pegRatio')
        profile['price_to_book'] = info.get('priceToBook')
        profile['price_to_sales'] = info.get('priceToSalesTrailing12Months')
        profile['ev_to_ebitda'] = info.get('enterpriseToEbitda')
        profile['ev_to_revenue'] = info.get('enterpriseToRevenue')

        # Growth
        profile['revenue_growth_yoy'] = info.get('revenueGrowth')
        profile['earnings_growth_yoy'] = info.get('earningsGrowth')
        profile['revenue_growth_quarterly'] = info.get('revenueQuarterlyGrowth')
        profile['earnings_growth_quarterly'] = info.get('earningsQuarterlyGrowth')

        # Profitability
        profile['gross_margin'] = info.get('grossMargins')
        profile['operating_margin'] = info.get('operatingMargins')
        profile['profit_margin'] = info.get('profitMargins')
        profile['return_on_equity'] = info.get('returnOnEquity')
        profile['return_on_assets'] = info.get('returnOnAssets')

        # Financials
        profile['total_revenue'] = info.get('totalRevenue')
        profile['ebitda'] = info.get('ebitda')
        if profile['total_revenue'] and profile['ebitda']:
            profile['ebitda_margin'] = profile['ebitda'] / profile['total_revenue']
        profile['net_income'] = info.get('netIncomeToCommon')
        profile['total_debt'] = info.get('totalDebt')
        profile['total_cash'] = info.get('totalCash')
        profile['debt_to_equity'] = info.get('debtToEquity')
        profile['current_ratio'] = info.get('currentRatio')
        profile['free_cash_flow'] = info.get('freeCashflow')
        profile['operating_cash_flow'] = info.get('operatingCashflow')

        # Shareholder
        profile['dividend_yield'] = info.get('dividendYield')
        profile['payout_ratio'] = info.get('payoutRatio')
        profile['shares_outstanding'] = info.get('sharesOutstanding')
        profile['float_shares'] = info.get('floatShares')
        profile['insider_pct'] = info.get('heldPercentInsiders')
        profile['institution_pct'] = info.get('heldPercentInstitutions')
        profile['short_pct_float'] = info.get('shortPercentOfFloat')
        profile['short_ratio'] = info.get('shortRatio')

        # Per-share
        profile['revenue_per_share'] = info.get('revenuePerShare')
        profile['book_value'] = info.get('bookValue')

        # Earnings
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get('Earnings Date')
                    if ed:
                        profile['next_earnings'] = str(ed[0]) if isinstance(ed, list) else str(ed)
        except Exception:
            pass

        # Last earnings surprise
        try:
            earnings_hist = stock.earnings_dates
            if earnings_hist is not None and len(earnings_hist) > 0:
                surprise = earnings_hist.iloc[0].get('Surprise(%)')
                if surprise is not None:
                    profile['last_earnings_surprise_pct'] = float(surprise)
        except Exception:
            pass

    except Exception as e:
        profile['error'] = str(e)[:200]

    _cache.set(cache_key, profile)
    return profile


def fetch_earnings_date(ticker: str) -> Dict[str, Any]:
    """
    Fetch next earnings date using 4-method cascade for maximum reliability.
    
    Method 1: stock.calendar (dict or DataFrame)
    Method 2: stock.info timestamps (earningsTimestamp, earningsTimestampStart)
    Method 3: stock.earnings_dates (sorted, find next upcoming)
    Method 4: Historical pattern estimation (last earnings + ~90 days)
    
    Includes retry with delay for yfinance rate limiting.
    Returns dict with next_earnings, days_until_earnings, confidence, source.
    """
    cache_key = f"{ticker}:calendar"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'next_earnings': None,
        'days_until_earnings': None,
        'last_earnings': None,
        'next_eps_estimate': None,
        'confidence': None,        # HIGH / MEDIUM / LOW
        'source': None,
        'error': None,
    }
    
    today = datetime.now().date()
    
    # Try up to 2 attempts (initial + 1 retry after delay if rate-limited)
    for attempt in range(2):
        try:
            stock = yf.Ticker(ticker)
            
            # ── METHOD 1: stock.calendar ──────────────────────────────
            try:
                cal = stock.calendar
                if cal is not None:
                    raw_date = None
                    if isinstance(cal, dict):
                        raw_date = cal.get('Earnings Date')
                        if isinstance(raw_date, list) and len(raw_date) > 0:
                            raw_date = raw_date[0]
                    elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.columns:
                        raw_date = cal['Earnings Date'].iloc[0]
                    
                    if raw_date is not None:
                        earn_dt = None
                        if hasattr(raw_date, 'date'):
                            earn_dt = raw_date.date() if callable(getattr(raw_date, 'date', None)) else raw_date
                        elif hasattr(raw_date, 'strftime'):
                            earn_dt = raw_date
                        elif isinstance(raw_date, str) and len(raw_date) >= 10:
                            try:
                                earn_dt = datetime.strptime(raw_date[:10], '%Y-%m-%d').date()
                            except ValueError:
                                pass
                        
                        if earn_dt:
                            days = (earn_dt - today).days
                            if days >= -7:  # Accept recent past (just reported)
                                result['next_earnings'] = earn_dt.strftime('%Y-%m-%d')
                                result['days_until_earnings'] = max(0, days)
                                result['confidence'] = 'HIGH'
                                result['source'] = 'Yahoo (.calendar)'
            except Exception:
                pass
            
            # ── METHOD 2: stock.info timestamps ───────────────────────
            if not result['next_earnings']:
                try:
                    info = stock.info or {}
                    for ts_key in ('earningsTimestamp', 'earningsTimestampStart',
                                   'mostRecentQuarter'):
                        ts = info.get(ts_key)
                        if ts and isinstance(ts, (int, float)) and ts > 0:
                            from datetime import timezone
                            earn_dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                            days = (earn_dt - today).days
                            if days >= -7:
                                result['next_earnings'] = earn_dt.strftime('%Y-%m-%d')
                                result['days_until_earnings'] = max(0, days)
                                result['confidence'] = 'MEDIUM-HIGH'
                                result['source'] = f'Yahoo (.info:{ts_key})'
                                break
                except Exception:
                    pass
            
            # ── METHOD 3: stock.earnings_dates (most reliable for upcoming) ──
            if not result['next_earnings']:
                try:
                    edates = stock.earnings_dates
                    if edates is not None and len(edates) > 0:
                        # Sort and find the next date >= today - 7
                        for dt_idx in sorted(edates.index):
                            try:
                                d = dt_idx.date() if hasattr(dt_idx, 'date') else dt_idx.to_pydatetime().date()
                                days = (d - today).days
                                if days >= -7:
                                    result['next_earnings'] = d.strftime('%Y-%m-%d')
                                    result['days_until_earnings'] = max(0, days)
                                    result['confidence'] = 'MEDIUM-HIGH'
                                    result['source'] = 'Yahoo (.earnings_dates)'
                                    # Grab EPS estimate if available
                                    row = edates.loc[dt_idx]
                                    eps_est = row.get('EPS Estimate') if hasattr(row, 'get') else None
                                    if eps_est is not None and not (isinstance(eps_est, float) and pd.isna(eps_est)):
                                        result['next_eps_estimate'] = float(eps_est)
                                    break
                            except Exception:
                                continue
                        
                        # Also grab last earnings date (most recent past)
                        for dt_idx in sorted(edates.index, reverse=True):
                            try:
                                d = dt_idx.date() if hasattr(dt_idx, 'date') else dt_idx.to_pydatetime().date()
                                days = (d - today).days
                                if days < -7:
                                    result['last_earnings'] = d.strftime('%Y-%m-%d')
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
            
            # If we got a result from any method, break out of retry loop
            if result['next_earnings']:
                break
            
            # If first attempt got nothing, check if rate-limited and retry
            if attempt == 0:
                # If stock.info returned empty/error, likely rate-limited
                try:
                    _test = stock.info
                    if not _test or _test.get('trailingPegRatio') is None:
                        # Possibly rate-limited — wait and retry
                        import time
                        time.sleep(2)
                        continue
                except Exception:
                    import time
                    time.sleep(2)
                    continue
                break  # Not rate-limited, just no data available
            
        except Exception as e:
            err_str = str(e).lower()
            if attempt == 0 and ('429' in err_str or 'rate' in err_str or 'too many' in err_str):
                import time
                time.sleep(2)
                continue
            result['error'] = str(e)[:200]
            break
    
    # ── METHOD 4: Historical pattern estimation ───────────────
    if not result['next_earnings'] and result.get('last_earnings'):
        try:
            last_dt = datetime.strptime(result['last_earnings'], '%Y-%m-%d').date()
            estimated = last_dt + timedelta(days=91)
            while (estimated - today).days < -7:
                estimated += timedelta(days=91)
            result['next_earnings'] = estimated.strftime('%Y-%m-%d')
            result['days_until_earnings'] = max(0, (estimated - today).days)
            result['confidence'] = 'LOW'
            result['source'] = 'Estimated (last + 91d pattern)'
        except Exception:
            pass
    
    # ── METHOD 4b: If no last_earnings either, try earnings_dates for any past date ──
    if not result['next_earnings'] and not result.get('last_earnings'):
        try:
            stock = yf.Ticker(ticker)
            edates = stock.earnings_dates
            if edates is not None and len(edates) > 0:
                for dt_idx in sorted(edates.index, reverse=True):
                    try:
                        d = dt_idx.date() if hasattr(dt_idx, 'date') else dt_idx.to_pydatetime().date()
                        result['last_earnings'] = d.strftime('%Y-%m-%d')
                        estimated = d + timedelta(days=91)
                        while (estimated - today).days < -7:
                            estimated += timedelta(days=91)
                        result['next_earnings'] = estimated.strftime('%Y-%m-%d')
                        result['days_until_earnings'] = max(0, (estimated - today).days)
                        result['confidence'] = 'LOW'
                        result['source'] = 'Estimated (historical + 91d)'
                        break
                    except Exception:
                        continue
        except Exception:
            pass
    
    _cache.set(cache_key, result)
    return result


def fetch_earnings_history(ticker: str) -> Dict[str, Any]:
    """
    Fetch earnings history — last 4 quarters of EPS estimates vs actuals,
    plus next earnings date and consensus estimate.
    """
    cache_key = f"{ticker}:earnings_hist"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = {
        'next_earnings': None,
        'days_until_earnings': None,
        'next_eps_estimate': None,
        'quarters': [],       # Last 4 quarters: {date, eps_estimate, eps_actual, surprise_pct, beat}
        'streak': 0,          # Consecutive beats (positive) or misses (negative)
        'avg_surprise_pct': None,
        'error': None,
    }

    try:
        stock = yf.Ticker(ticker)

        # Next earnings date — use the robust 4-method cascade
        try:
            earn_date_result = fetch_earnings_date(ticker)
            if earn_date_result.get('next_earnings'):
                result['next_earnings'] = earn_date_result['next_earnings']
                result['days_until_earnings'] = earn_date_result.get('days_until_earnings')
                result['next_eps_estimate'] = earn_date_result.get('next_eps_estimate')
                result['last_earnings'] = earn_date_result.get('last_earnings')
                result['confidence'] = earn_date_result.get('confidence', 'MEDIUM')
                result['source'] = earn_date_result.get('source', '?')
        except Exception:
            pass

        # Earnings history — EPS estimates vs actuals
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                quarters = []
                for idx, row in earnings_dates.head(8).iterrows():
                    eps_est = row.get('EPS Estimate')
                    eps_act = row.get('Reported EPS')
                    surprise = row.get('Surprise(%)')

                    # Skip future dates with no actual
                    if eps_act is None or (isinstance(eps_act, float) and pd.isna(eps_act)):
                        # This might be the next upcoming — grab estimate
                        if eps_est is not None and not (isinstance(eps_est, float) and pd.isna(eps_est)):
                            result['next_eps_estimate'] = float(eps_est)
                        continue

                    q = {
                        'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'eps_estimate': float(eps_est) if eps_est is not None and not pd.isna(eps_est) else None,
                        'eps_actual': float(eps_act) if eps_act is not None and not pd.isna(eps_act) else None,
                        'surprise_pct': float(surprise) if surprise is not None and not pd.isna(surprise) else None,
                    }

                    # Determine beat/miss
                    if q['eps_estimate'] is not None and q['eps_actual'] is not None:
                        q['beat'] = q['eps_actual'] > q['eps_estimate']
                    elif q['surprise_pct'] is not None:
                        q['beat'] = q['surprise_pct'] > 0
                    else:
                        q['beat'] = None

                    quarters.append(q)
                    if len(quarters) >= 4:
                        break

                result['quarters'] = quarters

                # Calculate streak
                streak = 0
                for q in quarters:
                    if q.get('beat') is True:
                        streak += 1
                    elif q.get('beat') is False:
                        if streak == 0:
                            streak -= 1
                        else:
                            break
                    else:
                        break
                result['streak'] = streak

                # Average surprise
                surprises = [q['surprise_pct'] for q in quarters if q.get('surprise_pct') is not None]
                if surprises:
                    result['avg_surprise_pct'] = sum(surprises) / len(surprises)

        except Exception:
            pass

    except Exception as e:
        result['error'] = str(e)[:200]

    _cache.set(cache_key, result)
    return result

def fetch_options_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch options data for put/call ratio analysis.
    
    Uses nearest expiry chain. Returns P/C ratio, total OI,
    unusual activity flag, and max pain estimate.
    """
    cache_key = f"{ticker}:options"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'put_call_ratio': None,
        'put_call_oi_ratio': None,
        'total_call_oi': None,
        'total_put_oi': None,
        'total_call_volume': None,
        'total_put_volume': None,
        'unusual_activity': False,
        'nearest_expiry': None,
        'max_pain': None,
        'error': None,
    }
    
    try:
        stock = yf.Ticker(ticker)
        expiry_dates = stock.options
        
        if not expiry_dates or len(expiry_dates) == 0:
            result['error'] = 'No options data available'
            _cache.set(cache_key, result)
            return result
        
        # Use nearest expiry
        nearest = expiry_dates[0]
        result['nearest_expiry'] = nearest
        
        chain = stock.option_chain(nearest)
        calls = chain.calls
        puts = chain.puts
        
        if calls is not None and puts is not None:
            # Volume-based P/C ratio
            call_vol = float(calls['volume'].sum()) if 'volume' in calls.columns else 0
            put_vol = float(puts['volume'].sum()) if 'volume' in puts.columns else 0
            
            if call_vol > 0:
                result['put_call_ratio'] = round(put_vol / call_vol, 2)
            
            result['total_call_volume'] = int(call_vol)
            result['total_put_volume'] = int(put_vol)
            
            # Open Interest based P/C ratio
            call_oi = float(calls['openInterest'].sum()) if 'openInterest' in calls.columns else 0
            put_oi = float(puts['openInterest'].sum()) if 'openInterest' in puts.columns else 0
            
            if call_oi > 0:
                result['put_call_oi_ratio'] = round(put_oi / call_oi, 2)
            
            result['total_call_oi'] = int(call_oi)
            result['total_put_oi'] = int(put_oi)
            
            # Unusual activity: total volume > 2x total OI
            total_vol = call_vol + put_vol
            total_oi = call_oi + put_oi
            if total_oi > 0:
                result['unusual_activity'] = total_vol > (total_oi * 2)
            
            # Max Pain: strike where most options expire worthless
            # Simplified: strike with highest total OI
            if 'openInterest' in calls.columns and 'openInterest' in puts.columns:
                all_strikes = pd.concat([
                    calls[['strike', 'openInterest']].rename(columns={'openInterest': 'call_oi'}),
                    puts[['strike', 'openInterest']].rename(columns={'openInterest': 'put_oi'})
                ]).groupby('strike').sum()
                
                if not all_strikes.empty:
                    all_strikes['total_oi'] = all_strikes.get('call_oi', 0) + all_strikes.get('put_oi', 0)
                    max_pain_strike = all_strikes['total_oi'].idxmax()
                    result['max_pain'] = float(max_pain_strike)
    
    except Exception as e:
        result['error'] = str(e)
    
    _cache.set(cache_key, result)
    return result


# =============================================================================
# INSTITUTIONAL & INSIDER DATA
# =============================================================================

def fetch_institutional_holders(ticker: str) -> Dict[str, Any]:
    """
    Fetch institutional holder data.
    
    Returns top holders and institutional ownership percentage.
    """
    cache_key = f"{ticker}:institutional"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'top_holders': [],
        'total_shares_held': None,
        'holder_count': 0,
        'error': None,
    }
    
    try:
        stock = yf.Ticker(ticker)
        
        # Institutional holders
        inst = stock.institutional_holders
        if inst is not None and not inst.empty:
            result['holder_count'] = len(inst)
            
            # Top 5 holders
            top = inst.head(5)
            holders = []
            for _, row in top.iterrows():
                holder = {
                    'name': str(row.get('Holder', '')),
                    'shares': int(row.get('Shares', 0)) if pd.notna(row.get('Shares')) else 0,
                    'pct_out': float(row.get('% Out', 0)) if pd.notna(row.get('% Out')) else 0,
                    'value': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else 0,
                }
                holders.append(holder)
            result['top_holders'] = holders
        
        # Major holders (ownership breakdown)
        major = stock.major_holders
        if major is not None and not major.empty:
            # major_holders format varies; try to extract key percentages
            result['major_holders_raw'] = major.to_dict()
    
    except Exception as e:
        result['error'] = str(e)
    
    _cache.set(cache_key, result)
    return result


def fetch_insider_transactions(ticker: str) -> Dict[str, Any]:
    """
    Fetch insider trading activity.
    
    Returns recent buys/sells, net activity.
    """
    cache_key = f"{ticker}:insider"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'transactions': [],
        'buys_90d': 0,
        'sells_90d': 0,
        'net_activity': 'unknown',
        'total_buy_value': 0,
        'total_sell_value': 0,
        'error': None,
    }
    
    try:
        stock = yf.Ticker(ticker)
        
        # Insider transactions
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            # Filter to last 90 days
            cutoff = datetime.now() - timedelta(days=90)
            
            transactions = []
            buys = 0
            sells = 0
            buy_value = 0
            sell_value = 0
            
            for _, row in insider.iterrows():
                txn_date = row.get('Start Date', row.get('Date', None))
                
                # Parse date
                if txn_date is not None:
                    if isinstance(txn_date, str):
                        try:
                            txn_date = pd.to_datetime(txn_date)
                        except Exception:
                            txn_date = None
                
                text = str(row.get('Text', row.get('Transaction', '')))
                shares = abs(int(row.get('Shares', 0))) if pd.notna(row.get('Shares')) else 0
                value = abs(float(row.get('Value', 0))) if pd.notna(row.get('Value')) else 0
                
                is_buy = any(w in text.lower() for w in ['purchase', 'buy', 'acquisition'])
                is_sell = any(w in text.lower() for w in ['sale', 'sell', 'disposition'])
                
                txn = {
                    'date': txn_date.strftime('%Y-%m-%d') if hasattr(txn_date, 'strftime') else str(txn_date),
                    'insider': str(row.get('Insider', row.get('Name', 'Unknown'))),
                    'type': 'buy' if is_buy else ('sell' if is_sell else 'other'),
                    'shares': shares,
                    'value': value,
                }
                transactions.append(txn)
                
                # Count 90-day activity
                if txn_date is not None and hasattr(txn_date, 'timestamp'):
                    if txn_date >= cutoff:
                        if is_buy:
                            buys += 1
                            buy_value += value
                        elif is_sell:
                            sells += 1
                            sell_value += value
            
            result['transactions'] = transactions[:10]  # Last 10
            result['buys_90d'] = buys
            result['sells_90d'] = sells
            result['total_buy_value'] = round(buy_value, 2)
            result['total_sell_value'] = round(sell_value, 2)
            
            if buys > sells:
                result['net_activity'] = 'buying'
            elif sells > buys:
                result['net_activity'] = 'selling'
            else:
                result['net_activity'] = 'neutral'
    
    except Exception as e:
        result['error'] = str(e)
    
    _cache.set(cache_key, result)
    return result


# =============================================================================
# BATCH FETCH — Get all data for a ticker in one shot
# =============================================================================

def fetch_all_ticker_data(ticker: str, include_fundamentals: bool = False) -> Dict[str, Any]:
    """
    Fetch all price data for a ticker: daily, weekly, monthly.
    
    Optionally include fundamentals (slower — use for detailed single-ticker view only).
    
    Returns dict with all DataFrames and metadata.
    """
    data = {
        'ticker': ticker,
        'daily': fetch_daily(ticker),
        'weekly': fetch_weekly(ticker),
        'monthly': fetch_monthly(ticker),
        'spy_daily': fetch_spy_daily(),
        'market_filter': fetch_market_filter(),
    }
    
    # Current price (from daily data if available)
    if data['daily'] is not None and len(data['daily']) > 0:
        data['current_price'] = float(data['daily']['Close'].iloc[-1])
    else:
        data['current_price'] = fetch_current_price(ticker)
    
    if include_fundamentals:
        data['info'] = fetch_ticker_info(ticker)
        data['earnings'] = fetch_earnings_date(ticker)
        data['options'] = fetch_options_data(ticker)
        data['institutional'] = fetch_institutional_holders(ticker)
        data['insider'] = fetch_insider_transactions(ticker)
    
    return data


def fetch_scan_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Batch fetch data for multiple tickers (scan mode).
    
    Fetches only price data (daily/weekly/monthly), not fundamentals.
    SPY and market filter are fetched once and shared.
    
    Returns dict of ticker -> data dict.
    """
    clear_cache()  # Fresh data for each scan
    
    # Pre-fetch shared data once
    spy_daily = fetch_spy_daily()
    market_filter = fetch_market_filter()
    
    results = {}
    for ticker in tickers:
        data = {
            'ticker': ticker,
            'daily': fetch_daily(ticker),
            'weekly': fetch_weekly(ticker),
            'monthly': fetch_monthly(ticker),
            'spy_daily': spy_daily,
            'market_filter': market_filter,
        }
        
        if data['daily'] is not None and len(data['daily']) > 0:
            data['current_price'] = float(data['daily']['Close'].iloc[-1])
        else:
            data['current_price'] = None
        
        results[ticker] = data
    
    return results


# =============================================================================
# MARKET INTELLIGENCE — Analysts, Insiders, Social Sentiment
# =============================================================================

def fetch_market_intelligence(ticker: str, finnhub_key: str = None) -> Dict[str, Any]:
    """
    Comprehensive market intelligence aggregator.
    Pulls analyst ratings, price targets, insider activity, and social sentiment.

    Returns structured dict for AI prompt and UI display.
    """
    cache_key = f"{ticker}:market_intel"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    intel = {
        'error': None,
        # Analyst consensus
        'analyst_count': None,
        'analyst_buy': 0, 'analyst_hold': 0, 'analyst_sell': 0,
        'analyst_strong_buy': 0, 'analyst_strong_sell': 0,
        'analyst_consensus': None,  # Strong Buy / Buy / Hold / Sell / Strong Sell
        # Price targets
        'target_mean': None, 'target_median': None,
        'target_high': None, 'target_low': None,
        'target_count': None,
        'target_upside_pct': None,  # vs current price
        # Recent upgrades/downgrades
        'recent_changes': [],  # [{firm, date, action, from_grade, to_grade}]
        # Insider activity
        'insider_buys_90d': 0, 'insider_sells_90d': 0,
        'insider_net_shares': 0,
        'insider_transactions': [],  # [{name, title, date, shares, value, type}]
        # Social sentiment (Finnhub)
        'social_reddit_mentions': None,
        'social_twitter_mentions': None,
        'social_score': None,
        # Finnhub analyst consensus
        'finnhub_buy': 0, 'finnhub_hold': 0, 'finnhub_sell': 0,
        'finnhub_strong_buy': 0, 'finnhub_strong_sell': 0,
        'finnhub_target': None,
    }

    stock = None
    try:
        stock = yf.Ticker(ticker)
    except Exception:
        intel['error'] = 'Failed to create yfinance ticker'
        _cache.set(cache_key, intel)
        return intel

    # ── Analyst Recommendations ───────────────────────────────────
    try:
        recs = stock.recommendations
        if recs is not None and len(recs) > 0:
            # Latest row contains aggregated counts
            latest = recs.iloc[-1] if len(recs) > 0 else None
            if latest is not None:
                sb = int(latest.get('strongBuy', 0) or 0)
                b = int(latest.get('buy', 0) or 0)
                h = int(latest.get('hold', 0) or 0)
                s = int(latest.get('sell', 0) or 0)
                ss = int(latest.get('strongSell', 0) or 0)
                total = sb + b + h + s + ss

                intel['analyst_strong_buy'] = sb
                intel['analyst_buy'] = b
                intel['analyst_hold'] = h
                intel['analyst_sell'] = s
                intel['analyst_strong_sell'] = ss
                intel['analyst_count'] = total

                if total > 0:
                    score = (sb * 5 + b * 4 + h * 3 + s * 2 + ss * 1) / total
                    if score >= 4.5:
                        intel['analyst_consensus'] = 'Strong Buy'
                    elif score >= 3.5:
                        intel['analyst_consensus'] = 'Buy'
                    elif score >= 2.5:
                        intel['analyst_consensus'] = 'Hold'
                    elif score >= 1.5:
                        intel['analyst_consensus'] = 'Sell'
                    else:
                        intel['analyst_consensus'] = 'Strong Sell'
    except Exception:
        pass

    # ── Price Targets ─────────────────────────────────────────────
    try:
        targets = stock.analyst_price_targets
        if targets is not None:
            if isinstance(targets, dict):
                intel['target_mean'] = targets.get('mean')
                intel['target_median'] = targets.get('median')
                intel['target_high'] = targets.get('high')
                intel['target_low'] = targets.get('low')
                intel['target_count'] = targets.get('numberOfAnalystOpinions')

                # Calculate upside
                current = targets.get('current')
                if current and intel['target_mean']:
                    intel['target_upside_pct'] = round(
                        (intel['target_mean'] - current) / current * 100, 1)
    except Exception:
        pass

    # ── Recent Upgrades/Downgrades ────────────────────────────────
    try:
        upgrades = stock.upgrades_downgrades
        if upgrades is not None and len(upgrades) > 0:
            recent = upgrades.head(10)
            changes = []
            for idx, row in recent.iterrows():
                changes.append({
                    'firm': row.get('Firm', '?'),
                    'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10],
                    'action': row.get('Action', '?'),
                    'to_grade': row.get('ToGrade', '?'),
                    'from_grade': row.get('FromGrade', ''),
                })
            intel['recent_changes'] = changes
    except Exception:
        pass

    # ── Insider Transactions ──────────────────────────────────────
    try:
        insiders = stock.insider_transactions
        if insiders is not None and len(insiders) > 0:
            buys_90d = 0
            sells_90d = 0
            net_shares = 0
            transactions = []
            cutoff = datetime.now() - pd.Timedelta(days=90)

            for _, row in insiders.head(20).iterrows():
                tx_date = row.get('Start Date')
                shares = row.get('Shares', 0) or 0
                value = row.get('Value', 0) or 0
                tx_type = str(row.get('Transaction', '')).lower()
                text = row.get('Text', '')

                is_recent = True
                if tx_date and hasattr(tx_date, 'timestamp'):
                    try:
                        is_recent = tx_date >= cutoff
                    except Exception:
                        pass

                is_buy = 'purchase' in tx_type or 'buy' in tx_type or 'acquisition' in text.lower()
                is_sell = 'sale' in tx_type or 'sell' in tx_type or 'disposition' in text.lower()

                if is_recent:
                    if is_buy:
                        buys_90d += 1
                        net_shares += abs(shares)
                    elif is_sell:
                        sells_90d += 1
                        net_shares -= abs(shares)

                transactions.append({
                    'name': row.get('Insider', '?'),
                    'title': row.get('Position', ''),
                    'date': tx_date.strftime('%Y-%m-%d') if hasattr(tx_date, 'strftime') else str(tx_date)[:10] if tx_date else '?',
                    'shares': int(shares),
                    'value': float(value),
                    'type': 'Buy' if is_buy else ('Sell' if is_sell else 'Other'),
                })

            intel['insider_buys_90d'] = buys_90d
            intel['insider_sells_90d'] = sells_90d
            intel['insider_net_shares'] = net_shares
            intel['insider_transactions'] = transactions[:10]
    except Exception:
        pass

    # ── Finnhub Data (if API key available) ───────────────────────
    import os
    fh_key = finnhub_key or os.environ.get('FINNHUB_API_KEY', '')

    if fh_key:
        import requests as req

        # Analyst consensus from Finnhub
        try:
            url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={fh_key}"
            resp = req.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    latest = data[0]
                    intel['finnhub_strong_buy'] = latest.get('strongBuy', 0)
                    intel['finnhub_buy'] = latest.get('buy', 0)
                    intel['finnhub_hold'] = latest.get('hold', 0)
                    intel['finnhub_sell'] = latest.get('sell', 0)
                    intel['finnhub_strong_sell'] = latest.get('strongSell', 0)
        except Exception:
            pass

        # Price target from Finnhub
        try:
            url = f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker}&token={fh_key}"
            resp = req.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    intel['finnhub_target'] = data.get('targetMean')
                    if not intel['target_mean']:
                        intel['target_mean'] = data.get('targetMean')
                        intel['target_high'] = data.get('targetHigh')
                        intel['target_low'] = data.get('targetLow')
                        intel['target_median'] = data.get('targetMedian')
        except Exception:
            pass

        # Social sentiment from Finnhub (requires premium on many tickers)
        try:
            from_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={ticker}&from={from_date}&token={fh_key}"
            resp = req.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                reddit = data.get('reddit', [])
                twitter = data.get('twitter', [])

                if reddit:
                    total_mentions = sum(r.get('mention', 0) for r in reddit[-7:])
                    intel['social_reddit_mentions'] = total_mentions

                if twitter:
                    total_mentions = sum(t.get('mention', 0) for t in twitter[-7:])
                    intel['social_twitter_mentions'] = total_mentions

                # Simple social score: mentions in last week
                r_count = intel['social_reddit_mentions'] or 0
                t_count = intel['social_twitter_mentions'] or 0
                if r_count + t_count > 0:
                    if r_count + t_count > 100:
                        intel['social_score'] = 'High buzz'
                    elif r_count + t_count > 20:
                        intel['social_score'] = 'Moderate'
                    else:
                        intel['social_score'] = 'Low'
            elif resp.status_code in (401, 403):
                intel['social_error'] = 'Finnhub premium required'
            else:
                intel['social_error'] = f'Finnhub {resp.status_code}'
        except Exception as e:
            intel['social_error'] = str(e)[:100]

    # ── Volume-Based Social Proxy (fallback) ──────────────────────
    # If Finnhub social sentiment unavailable, use volume surge as a proxy
    if not intel.get('social_score'):
        try:
            vol_data = None

            # Prefer cached daily data (already fetched during scan) to avoid extra API call
            try:
                daily = fetch_daily(ticker, period='3mo')
                if daily is not None and len(daily) >= 20:
                    vol_col = 'Volume' if 'Volume' in daily.columns else (
                        'volume' if 'volume' in daily.columns else None)
                    if vol_col:
                        vol_data = daily.rename(columns={vol_col: 'Volume'}) if vol_col != 'Volume' else daily
            except Exception:
                pass

            # Fallback: direct yfinance call
            if vol_data is None and stock is not None:
                try:
                    hist = stock.history(period='3mo')
                    if hist is not None and len(hist) >= 20:
                        vol_col = 'Volume' if 'Volume' in hist.columns else (
                            'volume' if 'volume' in hist.columns else None)
                        if vol_col:
                            vol_data = hist.rename(columns={vol_col: 'Volume'}) if vol_col != 'Volume' else hist
                except Exception:
                    pass

            if vol_data is not None and len(vol_data) >= 20 and 'Volume' in vol_data.columns:
                recent_vol = float(vol_data['Volume'].iloc[-5:].mean())
                avg_vol_50 = float(vol_data['Volume'].tail(min(50, len(vol_data))).mean())

                vol_surge = recent_vol / avg_vol_50 if avg_vol_50 > 0 else 1.0
                intel['volume_surge_ratio'] = round(vol_surge, 2)

                if vol_surge >= 3.0:
                    intel['social_score'] = 'High volume surge'
                elif vol_surge >= 2.0:
                    intel['social_score'] = 'Elevated volume'
                elif vol_surge >= 1.5:
                    intel['social_score'] = 'Above avg volume'
                else:
                    intel['social_score'] = 'Normal volume'
                intel['social_source'] = 'volume_proxy'
            else:
                intel['social_score'] = 'No volume data'
                intel['social_source'] = 'unavailable'
        except Exception as e:
            intel['social_score'] = 'Volume check failed'
            intel['social_source'] = 'error'
            intel['social_error'] = str(e)[:100]

    _cache.set(cache_key, intel)
    return intel




def fetch_tradingview_summary(ticker: str, interval: str = '1d') -> Dict[str, Any]:
    """
    Fetch TradingView technical analysis summary via tradingview_ta library.

    Returns summary recommendation (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL),
    oscillator and moving average recommendations, and key indicator values.

    interval: '1m', '5m', '15m', '1h', '4h', '1d', '1W', '1M'
    """
    cache_key = f"{ticker}:tv:{interval}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = {
        'recommendation': None,
        'buy': 0, 'sell': 0, 'neutral': 0,
        'ma_recommendation': None,
        'ma_buy': 0, 'ma_sell': 0, 'ma_neutral': 0,
        'osc_recommendation': None,
        'osc_buy': 0, 'osc_sell': 0, 'osc_neutral': 0,
        'rsi': None, 'stoch_k': None, 'cci': None,
        'macd_signal': None, 'adx': None,
        'interval': interval,
        'error': None,
    }

    try:
        from tradingview_ta import TA_Handler, Interval

        interval_map = {
            '1m': Interval.INTERVAL_1_MINUTE,
            '5m': Interval.INTERVAL_5_MINUTES,
            '15m': Interval.INTERVAL_15_MINUTES,
            '1h': Interval.INTERVAL_1_HOUR,
            '4h': Interval.INTERVAL_4_HOURS,
            '1d': Interval.INTERVAL_1_DAY,
            '1W': Interval.INTERVAL_1_WEEK,
            '1M': Interval.INTERVAL_1_MONTH,
        }

        # Try common US exchanges in order
        exchanges = ["NASDAQ", "NYSE", "AMEX"]
        analysis = None

        for exchange in exchanges:
            try:
                handler = TA_Handler(
                    symbol=ticker,
                    screener="america",
                    exchange=exchange,
                    interval=interval_map.get(interval, Interval.INTERVAL_1_DAY),
                )
                analysis = handler.get_analysis()
                if analysis:
                    break
            except Exception:
                continue

        if analysis:
            summary = analysis.summary or {}
            result['recommendation'] = summary.get('RECOMMENDATION')
            result['buy'] = summary.get('BUY', 0)
            result['sell'] = summary.get('SELL', 0)
            result['neutral'] = summary.get('NEUTRAL', 0)

            ma = analysis.moving_averages or {}
            result['ma_recommendation'] = ma.get('RECOMMENDATION')
            result['ma_buy'] = ma.get('BUY', 0)
            result['ma_sell'] = ma.get('SELL', 0)
            result['ma_neutral'] = ma.get('NEUTRAL', 0)

            osc = analysis.oscillators or {}
            result['osc_recommendation'] = osc.get('RECOMMENDATION')
            result['osc_buy'] = osc.get('BUY', 0)
            result['osc_sell'] = osc.get('SELL', 0)
            result['osc_neutral'] = osc.get('NEUTRAL', 0)

            indicators = analysis.indicators or {}
            result['rsi'] = indicators.get('RSI')
            result['stoch_k'] = indicators.get('Stoch.K')
            result['cci'] = indicators.get('CCI20')
            result['macd_signal'] = indicators.get('MACD.signal')
            result['adx'] = indicators.get('ADX')

    except ImportError:
        result['error'] = 'tradingview_ta not installed (pip install tradingview_ta)'
    except Exception as e:
        result['error'] = str(e)[:200]

    _cache.set(cache_key, result)
    return result


def fetch_tradingview_mtf(ticker: str) -> Dict[str, Dict]:
    """
    Fetch TradingView summaries for multiple timeframes.
    Returns: {'1h': {...}, '4h': {...}, '1d': {...}, '1W': {...}}
    """
    mtf = {}
    for interval in ['1h', '4h', '1d', '1W']:
        mtf[interval] = fetch_tradingview_summary(ticker, interval)
    return mtf


# =============================================================================
# FINNHUB — News & Events (requires free API key)
# =============================================================================

def fetch_finnhub_news(ticker: str, api_key: str = None,
                       days_back: int = 7) -> Dict[str, Any]:
    """
    Fetch recent news for a ticker from Finnhub.

    Returns list of headlines with source, datetime, summary, and sentiment.
    Free tier: 60 calls/minute, no auth for basic endpoints.

    api_key: Finnhub API key. If None, tries env var FINNHUB_API_KEY.
    """
    import os
    key = api_key or os.environ.get('FINNHUB_API_KEY', '')

    result = {
        'headlines': [],
        'count': 0,
        'error': None,
    }

    if not key:
        result['error'] = 'No Finnhub API key (set FINNHUB_API_KEY or pass api_key)'
        return result

    cache_key = f"{ticker}:finnhub_news"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import requests
        from_date = (datetime.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={ticker}&from={from_date}&to={to_date}&token={key}"
        )

        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            articles = resp.json()
            headlines = []
            for a in articles[:10]:  # Top 10
                headlines.append({
                    'headline': a.get('headline', ''),
                    'source': a.get('source', ''),
                    'datetime': datetime.fromtimestamp(a.get('datetime', 0)).strftime('%Y-%m-%d %H:%M'),
                    'summary': (a.get('summary', '') or '')[:200],
                    'url': a.get('url', ''),
                    'category': a.get('category', ''),
                })
            result['headlines'] = headlines
            result['count'] = len(headlines)
        else:
            result['error'] = f"Finnhub returned {resp.status_code}"

    except ImportError:
        result['error'] = 'requests not installed'
    except Exception as e:
        result['error'] = str(e)[:200]

    _cache.set(cache_key, result)
    return result
