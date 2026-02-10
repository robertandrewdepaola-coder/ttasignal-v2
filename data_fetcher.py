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
        info = stock.info
        
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
    Fetch next earnings date and related calendar info.
    
    Returns dict with next_earnings, days_until_earnings.
    """
    cache_key = f"{ticker}:calendar"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = {
        'next_earnings': None,
        'days_until_earnings': None,
        'error': None,
    }
    
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        
        if cal is not None:
            # calendar can be a dict or DataFrame depending on yfinance version
            if isinstance(cal, dict):
                earnings = cal.get('Earnings Date')
                if earnings:
                    # Can be a list of dates
                    if isinstance(earnings, list) and len(earnings) > 0:
                        next_date = earnings[0]
                    else:
                        next_date = earnings
                    
                    if hasattr(next_date, 'strftime'):
                        result['next_earnings'] = next_date.strftime('%Y-%m-%d')
                        delta = (next_date - datetime.now()).days
                        result['days_until_earnings'] = max(0, delta)
                    elif isinstance(next_date, str):
                        result['next_earnings'] = next_date
                        
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.columns:
                    next_date = cal['Earnings Date'].iloc[0]
                    if hasattr(next_date, 'strftime'):
                        result['next_earnings'] = next_date.strftime('%Y-%m-%d')
                        delta = (next_date - datetime.now()).days
                        result['days_until_earnings'] = max(0, delta)
    
    except Exception as e:
        result['error'] = str(e)
    
    _cache.set(cache_key, result)
    return result


# =============================================================================
# OPTIONS DATA — Put/Call ratio, open interest, max pain
# =============================================================================

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
