"""
TTA v2 Journal Manager - Trade Lifecycle Management
=====================================================

Handles: watchlist persistence, trade entry/exit, position tracking,
P&L calculation, and exit categorization.

NO yfinance calls. NO UI code. Pure data + business logic.
Price updates come from data_fetcher via the UI layer.

Persistence: JSON files (upgradeable to SQLite later).

Version: 2.2.0 (2026-02-16) - Phase 2: File locking, Decimal precision, timezones
"""

import json
import uuid
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
import shutil

# Cross-platform file locking
try:
    if sys.platform == 'win32':
        import msvcrt
    else:
        import fcntl
    FILE_LOCKING_AVAILABLE = True
except ImportError:
    FILE_LOCKING_AVAILABLE = False
    print("[journal] WARNING: File locking not available on this platform")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_DIR = "."
MAX_WATCHLIST_SIZE = 500
FILE_LOCK_TIMEOUT = 5.0

EXIT_REASONS = [
    'stop_loss',
    'target_hit',
    'weekly_cross',
    'time_exit',
    'manual',
    'batch_close',
]

SIGNAL_TYPES = ['PRIMARY', 'AO_CONFIRMATION', 'RE_ENTRY', 'LATE_ENTRY']

CONDITION_TYPES = [
    'breakout_above',
    'pullback_to',
    'breakout_volume',
]


# =============================================================================
# DECIMAL HELPERS
# =============================================================================

def to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """Convert value to Decimal with proper precision."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def decimal_to_float(value: Decimal, places: int = 2) -> float:
    """Convert Decimal to float with specified decimal places."""
    quantize_str = '0.' + '0' * places
    return float(value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))


def now_utc() -> datetime:
    """Get current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def now_utc_str() -> str:
    """Get current UTC timestamp as ISO string."""
    return now_utc().isoformat()


def today_utc_str() -> str:
    """Get today's date in UTC as YYYY-MM-DD string."""
    return now_utc().strftime('%Y-%m-%d')


# =============================================================================
# FILE LOCKING CONTEXT MANAGER
# =============================================================================

class FileLock:
    """Cross-platform file locking context manager."""

    def __init__(self, file_path: Path, timeout: float = FILE_LOCK_TIMEOUT):
        self.file_path = file_path
        self.timeout = timeout
        self.lock_file = None
        self.locked = False

    def __enter__(self):
        if not FILE_LOCKING_AVAILABLE:
            return self

        lock_path = self.file_path.with_suffix(self.file_path.suffix + '.lock')
        start_time = time.time()

        while True:
            try:
                self.lock_file = open(lock_path, 'w')

                if sys.platform == 'win32':
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                self.locked = True
                return self

            except (IOError, OSError):
                if time.time() - start_time > self.timeout:
                    if self.lock_file:
                        self.lock_file.close()
                    raise TimeoutError(f"Could not acquire lock for {self.file_path} within {self.timeout}s")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not FILE_LOCKING_AVAILABLE or not self.locked:
            return

        try:
            if sys.platform == 'win32':
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)

            self.lock_file.close()

            lock_path = self.file_path.with_suffix(self.file_path.suffix + '.lock')
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            print(f"[journal] Error releasing lock: {e}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConditionalEntry:
    """A pending conditional order - triggers when price/volume conditions are met."""
    ticker: str
    condition_type: str = 'breakout_above'
    trigger_price: float = 0
    volume_multiplier: float = 1.5
    stop_price: float = 0
    target_price: float = 0
    position_size_pct: float = 12.5
    notes: str = ''
    conviction: int = 0
    signal_type: str = ''
    quality_grade: str = ''
    created_date: str = ''
    expires_date: str = ''
    status: str = 'PENDING'

    def to_dict(self) -> Dict:
        return asdict(self)


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
    is_favorite: bool = False
    focus_label: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Trade:
    trade_id: str
    ticker: str
    status: str = 'OPEN'
    entry_date: str = ''
    entry_price: float = 0
    shares: float = 0
    position_size: float = 0
    signal_type: str = ''
    signal_age_days: int = 0
    quality_grade: str = ''
    conviction_at_entry: int = 0
    initial_stop: float = 0
    current_stop: float = 0
    stop_method: str = ''
    target: float = 0
    risk_per_share: float = 0
    risk_pct: float = 0
    exit_date: str = ''
    exit_price: float = 0
    exit_reason: str = ''
    realized_pnl: float = 0
    realized_pnl_pct: float = 0
    slippage_entry: float = 0
    days_held: int = 0
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
    """Manages watchlist, open trades, and trade history with JSON persistence."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.watchlist_file = self.data_dir / "v2_watchlist.json"
        self.open_trades_file = self.data_dir / "v2_open_trades.json"
        self.history_file = self.data_dir / "v2_trade_history.json"
        self.conditionals_file = self.data_dir / "v2_conditionals.json"
        self.scan_results_file = self.data_dir / "v2_last_scan.json"

        self.watchlist: List[Dict] = self._load(self.watchlist_file, [])
        if not isinstance(self.watchlist, list):
            print(f"[journal] WARNING: watchlist file contained {type(self.watchlist).__name__}, resetting to []")
            self.watchlist = []
            self._save(self.watchlist_file, self.watchlist)
        elif self.watchlist and any(not isinstance(w, dict) for w in self.watchlist):
            self.watchlist = [w for w in self.watchlist if isinstance(w, dict) and 'ticker' in w]
            self._save(self.watchlist_file, self.watchlist)
        self.open_trades: List[Dict] = self._load(self.open_trades_file, [])
        self.trade_history: List[Dict] = self._load(self.history_file, [])
        self.conditionals: List[Dict] = self._load(self.conditionals_file, [])

    def _load(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                with FileLock(path):
                    with open(path, 'r') as f:
                        return json.load(f)
            except TimeoutError:
                print(f"[journal] WARNING: Timeout acquiring lock for {path}, using default")
            except Exception as e:
                print(f"[journal] Error loading {path}: {e}")
        return default

    def _save(self, path: Path, data: Any):
        try:
            with FileLock(path):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            try:
                import github_backup
                github_backup.mark_dirty(path.name)
                print(f"[journal] Queued {path.name} for GitHub backup")
            except ImportError:
                print(f"[journal] WARNING: github_backup module not available - no remote backup")
            except Exception as e:
                print(f"[journal] ERROR: GitHub backup failed for {path.name}: {e}")
        except TimeoutError:
            print(f"[journal] ERROR: Timeout acquiring lock for {path}, save failed")
        except Exception as e:
            print(f"[journal] Error saving {path}: {e}")

    def _save_atomic(self, path: Path, data: Any):
        """Atomic save: write to temp file, then rename. Prevents corruption on crash."""
        temp_path = path.with_suffix('.tmp')
        try:
            with FileLock(path):
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                shutil.move(str(temp_path), str(path))
            # Keep backup behavior consistent with _save().
            try:
                import github_backup
                github_backup.mark_dirty(path.name)
                print(f"[journal] Queued {path.name} for GitHub backup")
            except ImportError:
                print(f"[journal] WARNING: github_backup module not available - no remote backup")
            except Exception as e:
                print(f"[journal] ERROR: GitHub backup failed for {path.name}: {e}")
        except TimeoutError:
            print(f"[journal] ERROR: Timeout acquiring lock for {path}, atomic save failed")
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            print(f"[journal] Error in atomic save {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _save_all(self):
        self._save(self.watchlist_file, self.watchlist)
        self._save(self.open_trades_file, self.open_trades)
        self._save(self.history_file, self.trade_history)

    def add_to_watchlist(self, item: WatchlistItem) -> str:
        ticker = item.ticker.upper().strip()
        item.ticker = ticker
        item.added_date = item.added_date or today_utc_str()

        if len(self.watchlist) >= MAX_WATCHLIST_SIZE:
            existing_idx = next((i for i, w in enumerate(self.watchlist) 
                               if isinstance(w, dict) and w.get('ticker') == ticker), None)
            if existing_idx is None:
                return f"Watchlist full ({MAX_WATCHLIST_SIZE} tickers). Remove tickers before adding new ones."

        if any(isinstance(w, dict) and w.get('ticker') == ticker for w in self.watchlist):
            self.watchlist = [(item.to_dict() if isinstance(w, dict) and w.get('ticker') == ticker else w)
                              for w in self.watchlist if isinstance(w, dict)]
            self._save(self.watchlist_file, self.watchlist)
            return f"Updated {ticker} in watchlist"

        self.watchlist.append(item.to_dict())
        self._save(self.watchlist_file, self.watchlist)
        return f"Added {ticker} to watchlist"

    def remove_from_watchlist(self, ticker: str) -> str:
        ticker = ticker.upper().strip()
        before = len(self.watchlist)
        self.watchlist = [w for w in self.watchlist if isinstance(w, dict) and w.get('ticker') != ticker]
        if len(self.watchlist) < before:
            self._save(self.watchlist_file, self.watchlist)
            return f"Removed {ticker}"
        return f"{ticker} not in watchlist"

    def get_watchlist(self) -> List[Dict]:
        return self.watchlist

    def get_watchlist_tickers(self) -> List[str]:
        return [w['ticker'] for w in self.watchlist if isinstance(w, dict) and 'ticker' in w]

    def clear_watchlist(self) -> str:
        self.watchlist = []
        self._save(self.watchlist_file, self.watchlist)
        return "Watchlist cleared"

    def set_watchlist_tickers(self, tickers: List[str]):
        """Bulk set watchlist from a list of ticker strings."""
        existing = {w['ticker'] for w in self.watchlist if isinstance(w, dict) and 'ticker' in w}
        added = 0
        for t in tickers:
            t = t.upper().strip()
            if t and t not in existing:
                if len(self.watchlist) >= MAX_WATCHLIST_SIZE:
                    print(f"[journal] WARNING: Watchlist limit reached ({MAX_WATCHLIST_SIZE}), skipping remaining tickers")
                    break
                item = WatchlistItem(ticker=t, added_date=today_utc_str())
                self.watchlist.append(item.to_dict())
                existing.add(t)
                added += 1
        if added > 0:
            self._save(self.watchlist_file, self.watchlist)

    def toggle_favorite(self, ticker: str) -> bool:
        """Toggle is_favorite for a ticker. Returns new favorite state."""
        ticker = ticker.upper().strip()
        for w in self.watchlist:
            if w['ticker'] == ticker:
                current = w.get('is_favorite', False)
                w['is_favorite'] = not current
                self._save(self.watchlist_file, self.watchlist)
                return w['is_favorite']
        return False

    def is_favorite(self, ticker: str) -> bool:
        """Check if a ticker is favorited."""
        ticker = ticker.upper().strip()
        for w in self.watchlist:
            if w['ticker'] == ticker:
                return w.get('is_favorite', False)
        return False

    def get_favorite_tickers(self) -> List[str]:
        """Get list of favorited tickers."""
        return [w['ticker'] for w in self.watchlist if isinstance(w, dict) and w.get('is_favorite', False)]

    def set_focus_label(self, ticker: str, label: str) -> str:
        """Set focus label for a ticker. Auto-adds ticker to watchlist if not already present."""
        ticker = ticker.upper().strip()
        for w in self.watchlist:
            if w['ticker'] == ticker:
                w['focus_label'] = label
                self._save(self.watchlist_file, self.watchlist)
                return label

        if label:
            if len(self.watchlist) >= MAX_WATCHLIST_SIZE:
                return f"ERROR: Watchlist full ({MAX_WATCHLIST_SIZE} tickers)"
            new_item = WatchlistItem(
                ticker=ticker,
                added_date=today_utc_str(),
                focus_label=label,
            )
            self.watchlist.append(new_item.to_dict())
            self._save(self.watchlist_file, self.watchlist)
            return label
        return ''

    def get_focus_label(self, ticker: str) -> str:
        """Get focus label for a ticker."""
        ticker = ticker.upper().strip()
        for w in self.watchlist:
            if w['ticker'] == ticker:
                return w.get('focus_label', '')
        return ''

    def get_focus_labels(self) -> Dict[str, str]:
        """Get all focus labels as {ticker: label} for non-empty labels."""
        return {w['ticker']: w['focus_label'] for w in self.watchlist
                if isinstance(w, dict) and w.get('focus_label', '')}

    def delete_single_ticker(self, ticker: str) -> str:
        """Delete a single ticker from watchlist with immediate persistence."""
        return self.remove_from_watchlist(ticker)

    def enter_trade(self, trade: Trade) -> str:
        ticker = trade.ticker.upper().strip()
        trade.ticker = ticker

        if any(t['ticker'] == ticker and t['status'] == 'OPEN' for t in self.open_trades):
            return f"Already have open position in {ticker}"

        trade.trade_id = trade.trade_id or f"{ticker}_{uuid.uuid4().hex[:12]}"
        trade.entry_date = trade.entry_date or today_utc_str()
        trade.opened_at = now_utc_str()
        trade.status = 'OPEN'

        if trade.entry_price > 0 and trade.initial_stop > 0:
            entry_d = to_decimal(trade.entry_price)
            stop_d = to_decimal(trade.initial_stop)
            risk_d = entry_d - stop_d
            trade.risk_per_share = decimal_to_float(risk_d, 2)
            trade.risk_pct = decimal_to_float((risk_d / entry_d) * 100, 1)

        if trade.entry_price > 0 and trade.shares > 0 and trade.position_size > 0:
            shares_d = to_decimal(trade.shares)
            entry_d = to_decimal(trade.entry_price)
            calculated_d = shares_d * entry_d
            calculated_size = decimal_to_float(calculated_d, 2)
            
            if abs(calculated_size - trade.position_size) > 1.0:
                print(f"[journal] WARNING: Position size mismatch for {ticker}. "
                      f"Provided: ${trade.position_size:.2f}, Calculated: ${calculated_size:.2f}")
                trade.position_size = calculated_size
        elif trade.entry_price > 0 and trade.shares == 0 and trade.position_size > 0:
            pos_d = to_decimal(trade.position_size)
            entry_d = to_decimal(trade.entry_price)
            trade.shares = decimal_to_float(pos_d / entry_d, 2)
        elif trade.shares > 0 and trade.position_size == 0:
            shares_d = to_decimal(trade.shares)
            entry_d = to_decimal(trade.entry_price)
            trade.position_size = decimal_to_float(shares_d * entry_d, 2)

        if trade.current_stop == 0:
            trade.current_stop = trade.initial_stop

        self.open_trades.append(trade.to_dict())
        self._save(self.open_trades_file, self.open_trades)
        self.remove_from_watchlist(ticker)

        return (f"Opened {ticker}: {trade.shares:.0f} shares @ ${trade.entry_price:.2f} "
                f"| Stop: ${trade.initial_stop:.2f} | Risk: {trade.risk_pct:.1f}%")

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

        trade_id = trade.get('trade_id', '')
        if any(h.get('trade_id') == trade_id for h in self.trade_history):
            print(f"[journal] WARNING: Trade {trade_id} already in history, skipping duplicate close")
            return f"Trade {ticker} already closed"

        trade['exit_price'] = float(exit_price)
        trade['exit_date'] = exit_date or today_utc_str()
        trade['exit_reason'] = exit_reason
        trade['closed_at'] = now_utc_str()
        trade['status'] = 'CLOSED'

        if notes:
            trade['notes'] = (trade.get('notes', '') + ' | EXIT: ' + notes).strip()

        entry = trade.get('entry_price', 0)
        shares = trade.get('shares', 0)
        if entry > 0 and shares > 0:
            entry_d = to_decimal(entry)
            exit_d = to_decimal(exit_price)
            shares_d = to_decimal(shares)
            
            pnl_d = (exit_d - entry_d) * shares_d
            pnl_pct_d = ((exit_d - entry_d) / entry_d) * 100
            
            trade['realized_pnl'] = decimal_to_float(pnl_d, 2)
            trade['realized_pnl_pct'] = decimal_to_float(pnl_pct_d, 1)

        try:
            entry_dt = datetime.strptime(trade.get('entry_date', ''), '%Y-%m-%d')
            exit_dt = datetime.strptime(trade['exit_date'], '%Y-%m-%d')
            trade['days_held'] = (exit_dt - entry_dt).days
        except Exception:
            trade['days_held'] = 0

        self.trade_history.append(trade)
        self._save_atomic(self.history_file, self.trade_history)
        
        self.open_trades.pop(trade_idx)
        self._save_atomic(self.open_trades_file, self.open_trades)

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

    def check_stops(self, current_prices: Dict[str, float], auto_execute: bool = False) -> List[Dict]:
        """Check all open positions against stop levels. Returns list of triggered stops."""
        triggered = []
        # Iterate over a snapshot to avoid skipping items when auto_execute closes trades.
        for trade in list(self.open_trades):
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
                if auto_execute:
                    self.close_trade(ticker, price, 'stop_loss')
            elif target > 0 and price >= target:
                triggered.append({
                    'ticker': ticker,
                    'trigger': 'target_hit',
                    'price': price,
                    'level': target,
                    'entry_price': trade.get('entry_price', 0),
                })
                if auto_execute:
                    self.close_trade(ticker, price, 'target_hit')

        return triggered

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
        """Summary of all open positions with current P&L."""
        open_trades = self.get_open_trades()
        if not open_trades:
            return {
                'count': 0, 'total_exposure': 0, 'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0, 'positions': [],
            }

        total_exposure_d = Decimal('0')
        unrealized_pnl_d = Decimal('0')
        positions = []

        for trade in open_trades:
            entry = trade.get('entry_price', 0)
            shares = trade.get('shares', 0)
            pos_size = trade.get('position_size', 0)
            
            entry_d = to_decimal(entry)
            shares_d = to_decimal(shares)
            pos_size_d = to_decimal(pos_size)
            total_exposure_d += pos_size_d

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
                current_d = to_decimal(current)
                pnl_d = (current_d - entry_d) * shares_d
                pnl_pct_d = ((current_d - entry_d) / entry_d) * 100
                unrealized_pnl_d += pnl_d
                
                pos['current_price'] = current
                pos['unrealized_pnl'] = decimal_to_float(pnl_d, 2)
                pos['unrealized_pnl_pct'] = decimal_to_float(pnl_pct_d, 1)

            positions.append(pos)

        total_exposure_f = decimal_to_float(total_exposure_d, 2)
        unrealized_pnl_f = decimal_to_float(unrealized_pnl_d, 2)
        
        return {
            'count': len(positions),
            'total_exposure': total_exposure_f,
            'unrealized_pnl': unrealized_pnl_f,
            'unrealized_pnl_pct': decimal_to_float((unrealized_pnl_d / total_exposure_d) * 100, 1)
            if total_exposure_d > 0 else 0,
            'positions': positions,
        }

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics from closed trades."""
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

    def get_recent_win_rate(self, last_n: int = 20) -> float:
        """Win rate over last N closed trades. Returns 0-1."""
        trades = self.get_trade_history(last_n=last_n)
        if not trades:
            return 0.5
        wins = sum(1 for t in trades if t.get('realized_pnl_pct', 0) > 0)
        return round(wins / len(trades), 2)

    def get_current_losing_streak(self) -> int:
        """Count consecutive losses from most recent trade backwards."""
        trades = self.get_trade_history()
        streak = 0
        for t in trades:
            if t.get('realized_pnl_pct', 0) <= 0:
                streak += 1
            else:
                break
        return streak

    def get_portfolio_risk_summary(self) -> Dict:
        """Get risk metrics for all open positions."""
        open_trades = self.get_open_trades()
        positions = []
        total_risk_d = Decimal('0')

        for trade in open_trades:
            entry = trade.get('entry_price', 0)
            stop = trade.get('current_stop', trade.get('initial_stop', 0))
            shares = trade.get('shares', 0)

            if entry > 0 and stop > 0 and shares > 0:
                entry_d = to_decimal(entry)
                stop_d = to_decimal(stop)
                shares_d = to_decimal(shares)
                risk_d = shares_d * (entry_d - stop_d)
                total_risk_d += risk_d
                risk_f = decimal_to_float(risk_d, 2)
            else:
                risk_f = 0

            positions.append({
                'ticker': trade['ticker'],
                'risk_dollars': max(0, risk_f),
                'entry': entry,
                'stop': stop,
                'shares': shares,
            })

        return {
            'total_risk_dollars': decimal_to_float(total_risk_d, 2),
            'positions': positions,
        }

    def get_sector_exposure(self, ticker_sectors: Dict[str, str]) -> Dict[str, int]:
        """Count open positions by sector."""
        exposure = {}
        for trade in self.get_open_trades():
            sector = ticker_sectors.get(trade['ticker'], 'Unknown')
            exposure[sector] = exposure.get(sector, 0) + 1
        return exposure

    def add_conditional(self, entry: ConditionalEntry) -> str:
        """Add a conditional entry (breakout alert)."""
        ticker = entry.ticker.upper().strip()
        entry.ticker = ticker
        entry.created_date = entry.created_date or today_utc_str()
        entry.status = 'PENDING'

        existing = [c for c in self.conditionals
                    if c['ticker'] == ticker and c['status'] == 'PENDING']
        if existing:
            for i, c in enumerate(self.conditionals):
                if c['ticker'] == ticker and c['status'] == 'PENDING':
                    self.conditionals[i] = entry.to_dict()
                    break
            self._save(self.conditionals_file, self.conditionals)
            return f"Updated conditional for {ticker}: {entry.condition_type} @ ${entry.trigger_price:.2f}"

        self.conditionals.append(entry.to_dict())
        self._save(self.conditionals_file, self.conditionals)
        return f"Set conditional for {ticker}: {entry.condition_type} @ ${entry.trigger_price:.2f}"

    def remove_conditional(self, ticker: str) -> str:
        ticker = ticker.upper().strip()
        before = len(self.conditionals)
        self.conditionals = [c for c in self.conditionals
                             if not (c['ticker'] == ticker and c['status'] == 'PENDING')]
        if len(self.conditionals) < before:
            self._save(self.conditionals_file, self.conditionals)
            return f"Removed conditional for {ticker}"
        return f"No pending conditional for {ticker}"

    def get_pending_conditionals(self) -> List[Dict]:
        return [c for c in self.conditionals if c.get('status') == 'PENDING']

    def get_conditional(self, ticker: str) -> Optional[Dict]:
        ticker = ticker.upper().strip()
        for c in self.conditionals:
            if c['ticker'] == ticker and c['status'] == 'PENDING':
                return c
        return None

    def check_conditionals(self, current_prices: Dict[str, float],
                           volume_ratios: Dict[str, float] = None) -> List[Dict]:
        """Check all pending conditionals against current prices."""
        volume_ratios = volume_ratios or {}
        triggered = []
        changed = False

        for cond in self.conditionals:
            if cond.get('status') != 'PENDING':
                continue

            ticker = cond['ticker']
            price = current_prices.get(ticker)
            if price is None:
                continue

            trigger = float(cond.get('trigger_price', 0))
            vol_required = float(cond.get('volume_multiplier', 1.5))
            vol_actual = volume_ratios.get(ticker, 0)
            cond_type = cond.get('condition_type', '')

            expires = cond.get('expires_date', '')
            if expires:
                try:
                    expires_dt = datetime.strptime(expires, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if expires_dt < now_utc():
                        cond['status'] = 'EXPIRED'
                        changed = True
                        continue
                except Exception:
                    pass

            is_triggered = False
            if cond_type == 'breakout_above' and trigger > 0 and price > trigger:
                is_triggered = True
            elif cond_type == 'breakout_volume' and trigger > 0 and price > trigger:
                if vol_actual >= vol_required:
                    is_triggered = True
            elif cond_type == 'pullback_to' and trigger > 0 and price <= trigger:
                is_triggered = True

            if is_triggered:
                cond['status'] = 'TRIGGERED'
                cond['triggered_price'] = price
                cond['triggered_date'] = today_utc_str()
                cond['triggered_volume_ratio'] = vol_actual
                triggered.append(cond)
                changed = True

        if changed:
            self._save(self.conditionals_file, self.conditionals)

        return triggered

    def expire_old_conditionals(self, days: int = 30):
        """Expire conditionals older than N days."""
        cutoff = now_utc()
        changed = False
        for cond in self.conditionals:
            if cond.get('status') != 'PENDING':
                continue
            created = cond.get('created_date', '')
            try:
                created_dt = datetime.strptime(created, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if (cutoff - created_dt).days > days:
                    cond['status'] = 'EXPIRED'
                    changed = True
            except Exception:
                pass
        if changed:
            self._save(self.conditionals_file, self.conditionals)

    def save_scan_results(self, results: List[Dict]):
        """Save scan results summary to JSON for cross-session persistence."""
        payload = {
            'timestamp': now_utc_str(),
            'count': len(results),
            'results': results,
        }
        self._save(self.scan_results_file, payload)

    def load_scan_results(self) -> Optional[Dict]:
        """Load last scan results. Returns {timestamp, count, results} or None."""
        data = self._load(self.scan_results_file, None)
        return data

    def get_all_tracked_tickers(self) -> Dict[str, str]:
        """Get all tickers being tracked with their status."""
        result = {}
        for w in self.watchlist:
            if isinstance(w, dict) and 'ticker' in w:
                result[w['ticker']] = 'watchlist'
        for c in self.conditionals:
            if c.get('status') == 'PENDING':
                result[c['ticker']] = 'conditional'
        for t in self.open_trades:
            if t.get('status') == 'OPEN':
                result[t['ticker']] = 'open'
        return result
