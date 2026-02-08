"""
TTA v2 Journal Manager — Trade Lifecycle Management
=====================================================

Handles: watchlist persistence, trade entry/exit, position tracking,
P&L calculation, and exit categorization.

NO yfinance calls. NO UI code. Pure data + business logic.
Price updates come from data_fetcher via the UI layer.

Persistence: JSON files (upgradeable to SQLite later).

Version: 2.0.0 (2026-02-08)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_DIR = "."

EXIT_REASONS = [
    'stop_loss',        # Hit initial or trailing stop
    'target_hit',       # Hit profit target
    'weekly_cross',     # Weekly MACD crossed bearish
    'time_exit',        # Held too long without progress
    'manual',           # Trader decided to exit
    'batch_close',      # Closed all positions
]

SIGNAL_TYPES = ['PRIMARY', 'AO_CONFIRMATION', 'RE_ENTRY', 'LATE_ENTRY']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WatchlistItem:
    ticker: str
    added_date: str = ''
    signal_type: str = ''
    quality_grade: str = ''
    recommendation: str = ''
    conviction: int = 0
    entry_zone_low: float = 0
    entry_zone_high: float = 0
    stop_price: float = 0
    notes: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Trade:
    trade_id: str
    ticker: str
    status: str = 'OPEN'  # OPEN, CLOSED

    # Entry
    entry_date: str = ''
    entry_price: float = 0
    shares: float = 0
    position_size: float = 0
    signal_type: str = ''       # PRIMARY, AO_CONFIRMATION, RE_ENTRY, LATE_ENTRY
    signal_age_days: int = 0    # How old was the signal at execution
    quality_grade: str = ''
    conviction_at_entry: int = 0

    # Risk
    initial_stop: float = 0
    current_stop: float = 0     # May be trailed up
    stop_method: str = ''
    target: float = 0
    risk_per_share: float = 0
    risk_pct: float = 0

    # Exit
    exit_date: str = ''
    exit_price: float = 0
    exit_reason: str = ''       # One of EXIT_REASONS

    # P&L
    realized_pnl: float = 0
    realized_pnl_pct: float = 0
    slippage_entry: float = 0   # Difference between signal price and actual fill
    days_held: int = 0

    # Context
    weekly_bullish_at_entry: bool = False
    monthly_bullish_at_entry: bool = False
    weinstein_stage_at_entry: int = 0
    notes: str = ''
    opened_at: str = ''
    closed_at: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# JOURNAL MANAGER
# =============================================================================

class JournalManager:
    """
    Manages watchlist, open trades, and trade history with JSON persistence.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.watchlist_file = self.data_dir / "v2_watchlist.json"
        self.open_trades_file = self.data_dir / "v2_open_trades.json"
        self.history_file = self.data_dir / "v2_trade_history.json"

        self.watchlist: List[Dict] = self._load(self.watchlist_file, [])
        self.open_trades: List[Dict] = self._load(self.open_trades_file, [])
        self.trade_history: List[Dict] = self._load(self.history_file, [])

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[journal] Error loading {path}: {e}")
        return default

    def _save(self, path: Path, data: Any):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[journal] Error saving {path}: {e}")

    def _save_all(self):
        self._save(self.watchlist_file, self.watchlist)
        self._save(self.open_trades_file, self.open_trades)
        self._save(self.history_file, self.trade_history)

    # ── Watchlist ─────────────────────────────────────────────────────

    def add_to_watchlist(self, item: WatchlistItem) -> str:
        ticker = item.ticker.upper().strip()
        item.ticker = ticker
        item.added_date = item.added_date or datetime.now().strftime('%Y-%m-%d')

        if any(w['ticker'] == ticker for w in self.watchlist):
            # Update existing
            self.watchlist = [w if w['ticker'] != ticker else item.to_dict()
                              for w in self.watchlist]
            self._save(self.watchlist_file, self.watchlist)
            return f"Updated {ticker} in watchlist"

        self.watchlist.append(item.to_dict())
        self._save(self.watchlist_file, self.watchlist)
        return f"Added {ticker} to watchlist"

    def remove_from_watchlist(self, ticker: str) -> str:
        ticker = ticker.upper().strip()
        before = len(self.watchlist)
        self.watchlist = [w for w in self.watchlist if w['ticker'] != ticker]
        if len(self.watchlist) < before:
            self._save(self.watchlist_file, self.watchlist)
            return f"Removed {ticker}"
        return f"{ticker} not in watchlist"

    def get_watchlist(self) -> List[Dict]:
        return self.watchlist

    def get_watchlist_tickers(self) -> List[str]:
        return [w['ticker'] for w in self.watchlist]

    def clear_watchlist(self) -> str:
        self.watchlist = []
        self._save(self.watchlist_file, self.watchlist)
        return "Watchlist cleared"

    def set_watchlist_tickers(self, tickers: List[str]):
        """Bulk set watchlist from a list of ticker strings."""
        existing = {w['ticker'] for w in self.watchlist}
        for t in tickers:
            t = t.upper().strip()
            if t and t not in existing:
                item = WatchlistItem(ticker=t, added_date=datetime.now().strftime('%Y-%m-%d'))
                self.watchlist.append(item.to_dict())
                existing.add(t)
        self._save(self.watchlist_file, self.watchlist)

    # ── Trade Entry ───────────────────────────────────────────────────

    def enter_trade(self, trade: Trade) -> str:
        ticker = trade.ticker.upper().strip()
        trade.ticker = ticker

        # Prevent duplicate open positions
        if any(t['ticker'] == ticker and t['status'] == 'OPEN' for t in self.open_trades):
            return f"Already have open position in {ticker}"

        # Auto-fill fields
        trade.trade_id = trade.trade_id or f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trade.entry_date = trade.entry_date or datetime.now().strftime('%Y-%m-%d')
        trade.opened_at = datetime.now().isoformat()
        trade.status = 'OPEN'

        # Calculate derived fields
        if trade.entry_price > 0 and trade.initial_stop > 0:
            trade.risk_per_share = round(trade.entry_price - trade.initial_stop, 2)
            trade.risk_pct = round(trade.risk_per_share / trade.entry_price * 100, 1)

        if trade.entry_price > 0 and trade.shares == 0 and trade.position_size > 0:
            trade.shares = round(trade.position_size / trade.entry_price, 2)
        elif trade.shares > 0 and trade.position_size == 0:
            trade.position_size = round(trade.shares * trade.entry_price, 2)

        if trade.current_stop == 0:
            trade.current_stop = trade.initial_stop

        self.open_trades.append(trade.to_dict())
        self._save(self.open_trades_file, self.open_trades)

        # Remove from watchlist
        self.remove_from_watchlist(ticker)

        return (f"Opened {ticker}: {trade.shares:.0f} shares @ ${trade.entry_price:.2f} "
                f"| Stop: ${trade.initial_stop:.2f} | Risk: {trade.risk_pct:.1f}%")

    # ── Trade Exit ────────────────────────────────────────────────────

    def close_trade(self, ticker: str, exit_price: float,
                    exit_reason: str = 'manual',
                    exit_date: str = None,
                    notes: str = '') -> str:
        ticker = ticker.upper().strip()
        trade = None
        trade_idx = None

        for i, t in enumerate(self.open_trades):
            if t['ticker'] == ticker and t['status'] == 'OPEN':
                trade = t
                trade_idx = i
                break

        if trade is None:
            return f"No open position in {ticker}"

        # Fill exit fields
        trade['exit_price'] = float(exit_price)
        trade['exit_date'] = exit_date or datetime.now().strftime('%Y-%m-%d')
        trade['exit_reason'] = exit_reason
        trade['closed_at'] = datetime.now().isoformat()
        trade['status'] = 'CLOSED'

        if notes:
            trade['notes'] = (trade.get('notes', '') + ' | EXIT: ' + notes).strip()

        # Calculate P&L
        entry = float(trade.get('entry_price', 0))
        shares = float(trade.get('shares', 0))
        if entry > 0:
            trade['realized_pnl'] = round((exit_price - entry) * shares, 2)
            trade['realized_pnl_pct'] = round((exit_price - entry) / entry * 100, 1)

        # Days held
        try:
            entry_dt = datetime.strptime(trade.get('entry_date', ''), '%Y-%m-%d')
            exit_dt = datetime.strptime(trade['exit_date'], '%Y-%m-%d')
            trade['days_held'] = (exit_dt - entry_dt).days
        except Exception:
            trade['days_held'] = 0

        # Slippage (if signal price was recorded)
        # TODO: calculate when we have signal price in context

        # Move to history
        self.trade_history.append(trade)
        self.open_trades.pop(trade_idx)

        self._save(self.open_trades_file, self.open_trades)
        self._save(self.history_file, self.trade_history)

        pnl = trade['realized_pnl']
        pnl_pct = trade['realized_pnl_pct']
        emoji = '🟢' if pnl >= 0 else '🔴'
        return (f"{emoji} Closed {ticker} @ ${exit_price:.2f} | "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%) | "
                f"Held: {trade['days_held']}d | Reason: {exit_reason}")

    def close_all_trades(self, exit_prices: Dict[str, float],
                         exit_reason: str = 'batch_close') -> List[str]:
        """Close all open positions. exit_prices: {ticker: price}"""
        results = []
        tickers = [t['ticker'] for t in self.open_trades if t['status'] == 'OPEN']
        for ticker in tickers:
            price = exit_prices.get(ticker, 0)
            if price > 0:
                results.append(self.close_trade(ticker, price, exit_reason))
        return results

    # ── Position Updates ──────────────────────────────────────────────

    def update_stop(self, ticker: str, new_stop: float) -> str:
        """Trail stop up for an open position."""
        ticker = ticker.upper().strip()
        for trade in self.open_trades:
            if trade['ticker'] == ticker and trade['status'] == 'OPEN':
                old_stop = trade.get('current_stop', 0)
                if new_stop > old_stop:
                    trade['current_stop'] = float(new_stop)
                    self._save(self.open_trades_file, self.open_trades)
                    return f"Trailed stop {ticker}: ${old_stop:.2f} → ${new_stop:.2f}"
                else:
                    return f"New stop ${new_stop:.2f} not higher than current ${old_stop:.2f}"
        return f"No open position in {ticker}"

    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check all open positions against stop levels.
        Returns list of triggered stops.
        
        current_prices: {ticker: current_price} from data_fetcher
        """
        triggered = []
        for trade in self.open_trades:
            if trade['status'] != 'OPEN':
                continue
            ticker = trade['ticker']
            price = current_prices.get(ticker)
            if price is None:
                continue

            stop = trade.get('current_stop', trade.get('initial_stop', 0))
            target = trade.get('target', 0)

            if stop > 0 and price <= stop:
                triggered.append({
                    'ticker': ticker,
                    'trigger': 'stop_loss',
                    'price': price,
                    'level': stop,
                    'entry_price': trade.get('entry_price', 0),
                })
            elif target > 0 and price >= target:
                triggered.append({
                    'ticker': ticker,
                    'trigger': 'target_hit',
                    'price': price,
                    'level': target,
                    'entry_price': trade.get('entry_price', 0),
                })

        return triggered

    # ── Queries ───────────────────────────────────────────────────────

    def get_open_trades(self) -> List[Dict]:
        return [t for t in self.open_trades if t.get('status') == 'OPEN']

    def get_open_tickers(self) -> List[str]:
        return [t['ticker'] for t in self.open_trades if t.get('status') == 'OPEN']

    def get_trade_history(self, last_n: int = None) -> List[Dict]:
        trades = sorted(self.trade_history,
                        key=lambda t: t.get('closed_at', ''), reverse=True)
        return trades[:last_n] if last_n else trades

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict]:
        for t in self.open_trades + self.trade_history:
            if t.get('trade_id') == trade_id:
                return t
        return None

    def get_position_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        Summary of all open positions with current P&L.
        
        current_prices: {ticker: price} — if provided, calculates unrealized P&L.
        """
        open_trades = self.get_open_trades()
        if not open_trades:
            return {
                'count': 0, 'total_exposure': 0, 'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0, 'positions': [],
            }

        total_exposure = 0
        unrealized_pnl = 0
        positions = []

        for trade in open_trades:
            entry = float(trade.get('entry_price', 0))
            shares = float(trade.get('shares', 0))
            pos_size = float(trade.get('position_size', 0))
            total_exposure += pos_size

            current = current_prices.get(trade['ticker']) if current_prices else None

            pos = {
                'ticker': trade['ticker'],
                'entry_price': entry,
                'shares': shares,
                'position_size': pos_size,
                'stop': trade.get('current_stop', trade.get('initial_stop', 0)),
                'target': trade.get('target', 0),
                'entry_date': trade.get('entry_date', ''),
                'signal_type': trade.get('signal_type', ''),
            }

            if current and entry > 0:
                pnl = (current - entry) * shares
                pnl_pct = (current - entry) / entry * 100
                unrealized_pnl += pnl
                pos['current_price'] = current
                pos['unrealized_pnl'] = round(pnl, 2)
                pos['unrealized_pnl_pct'] = round(pnl_pct, 1)

            positions.append(pos)

        return {
            'count': len(positions),
            'total_exposure': round(total_exposure, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'unrealized_pnl_pct': round(unrealized_pnl / total_exposure * 100, 1)
            if total_exposure > 0 else 0,
            'positions': positions,
        }

    # ── Performance Analytics ─────────────────────────────────────────

    def get_performance_stats(self) -> Dict:
        """
        Calculate performance statistics from closed trades.
        """
        trades = self.trade_history
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'avg_pnl_pct': 0,
                'total_pnl': 0, 'best_trade': 0, 'worst_trade': 0,
                'avg_days_held': 0, 'by_signal_type': {}, 'by_exit_reason': {},
            }

        pnls = [t.get('realized_pnl_pct', 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        dollar_pnls = [t.get('realized_pnl', 0) for t in trades]
        days = [t.get('days_held', 0) for t in trades]

        # By signal type
        by_signal = {}
        for t in trades:
            st = t.get('signal_type', 'Unknown')
            if st not in by_signal:
                by_signal[st] = {'count': 0, 'wins': 0, 'total_pnl_pct': 0}
            by_signal[st]['count'] += 1
            if t.get('realized_pnl_pct', 0) > 0:
                by_signal[st]['wins'] += 1
            by_signal[st]['total_pnl_pct'] += t.get('realized_pnl_pct', 0)

        for st in by_signal:
            c = by_signal[st]['count']
            by_signal[st]['win_rate'] = round(by_signal[st]['wins'] / c * 100, 1) if c > 0 else 0
            by_signal[st]['avg_pnl_pct'] = round(by_signal[st]['total_pnl_pct'] / c, 1) if c > 0 else 0

        # By exit reason
        by_exit = {}
        for t in trades:
            er = t.get('exit_reason', 'unknown')
            if er not in by_exit:
                by_exit[er] = {'count': 0, 'avg_pnl_pct': 0, 'total_pnl_pct': 0}
            by_exit[er]['count'] += 1
            by_exit[er]['total_pnl_pct'] += t.get('realized_pnl_pct', 0)

        for er in by_exit:
            c = by_exit[er]['count']
            by_exit[er]['avg_pnl_pct'] = round(by_exit[er]['total_pnl_pct'] / c, 1) if c > 0 else 0

        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
            'avg_pnl_pct': round(sum(pnls) / len(pnls), 1) if pnls else 0,
            'total_pnl': round(sum(dollar_pnls), 2),
            'best_trade': round(max(pnls), 1) if pnls else 0,
            'worst_trade': round(min(pnls), 1) if pnls else 0,
            'avg_days_held': round(sum(days) / len(days), 0) if days else 0,
            'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0,
            'by_signal_type': by_signal,
            'by_exit_reason': by_exit,
        }

    # ── Sector Exposure ───────────────────────────────────────────────

    def get_sector_exposure(self, ticker_sectors: Dict[str, str]) -> Dict[str, int]:
        """
        Count open positions by sector.
        ticker_sectors: {ticker: sector} from data_fetcher.fetch_ticker_info()
        """
        exposure = {}
        for trade in self.get_open_trades():
            sector = ticker_sectors.get(trade['ticker'], 'Unknown')
            exposure[sector] = exposure.get(sector, 0) + 1
        return exposure
