"""
TTA v2 Watchlist Bridge — Compatibility Shim
==============================================

Provides the same interface as JournalManager's watchlist methods,
but routes to the new WatchlistManager (JSON) and ScrapingBridge.

This allows app.py to switch from jm.get_watchlist_tickers() to
bridge.get_watchlist_tickers() with zero logic changes.

Also handles one-time migration from JournalManager to v2 JSON.

Usage:
    from watchlist_bridge import WatchlistBridge
    bridge = WatchlistBridge(watchlist_manager, journal_manager)

    # Drop-in replacements for jm.get_watchlist_tickers(), etc.
    tickers = bridge.get_watchlist_tickers()
    bridge.add_to_watchlist(WatchlistItem(ticker="AAPL"))
    bridge.toggle_favorite("AAPL")

Version: 1.0.0 (2026-02-14)
"""

from typing import Dict, List, Optional, Tuple
from watchlist_manager import WatchlistManager, MASTER_ID


class WatchlistBridge:
    """
    Compatibility bridge between old JournalManager calls and new WatchlistManager.

    Maps old methods to new system, keeping favorites/focus in JournalManager
    and ticker lists in WatchlistManager.

    Integration: Replace `jm.get_watchlist_tickers()` with `bridge.get_watchlist_tickers()`
    at ~8 call sites in app.py. No other logic changes needed.
    """

    def __init__(self, manager: WatchlistManager, journal_manager):
        """
        Args:
            manager: New WatchlistManager (JSON-based).
            journal_manager: Existing JournalManager (SQLite/JSON).
        """
        self.manager = manager
        self.jm = journal_manager
        self._migration_checked = False

    def ensure_migration(self) -> None:
        """
        Run one-time migration from JournalManager to WatchlistManager.

        Safe to call multiple times — checks metadata flag internally.
        Called automatically on first get_watchlist_tickers() call.
        """
        if self._migration_checked:
            return
        self._migration_checked = True
        try:
            ok, msg = self.manager.migrate_from_journal(self.jm)
            if ok:
                print(f"[watchlist_bridge] Migration: {msg}")
            # "Already migrated" is fine — not an error
        except Exception as e:
            print(f"[watchlist_bridge] Migration error (non-fatal): {e}")

    # --- Drop-in Replacements ------------------------------------------------
    # These methods match the exact signatures used in app.py

    def get_watchlist_tickers(self) -> List[str]:
        """
        Returns tickers from the active watchlist.
        Drop-in replacement for jm.get_watchlist_tickers().
        """
        self.ensure_migration()
        return self.manager.get_active_watchlist().get("tickers", [])

    def add_to_watchlist(self, item) -> str:
        """
        Add a ticker to the active watchlist.
        Accepts WatchlistItem (for compatibility) or string.
        Also forwards to JournalManager to preserve metadata (favorites, notes).
        """
        ticker = item.ticker if hasattr(item, "ticker") else str(item)
        active = self.manager.get_active_watchlist()

        ok, msg = self.manager.add_manual_ticker(active["id"], ticker)

        # Also add to JournalManager for metadata preservation
        try:
            self.jm.add_to_watchlist(item)
        except Exception:
            pass  # Non-fatal — JM is secondary now

        return msg

    def remove_from_watchlist(self, ticker: str) -> str:
        """Remove ticker from active watchlist."""
        active = self.manager.get_active_watchlist()
        ok, msg = self.manager.remove_manual_ticker(active["id"], ticker)
        return msg

    def clear_watchlist(self) -> str:
        """Clear all tickers from the active watchlist."""
        active = self.manager.get_active_watchlist()
        wl_id = active["id"]

        self.manager.update_tickers(wl_id, [], backup_old=True)
        return "Watchlist cleared"

    def set_watchlist_tickers(self, tickers: List[str]) -> None:
        """
        Bulk replace tickers on the active watchlist.
        Drop-in replacement for jm.set_watchlist_tickers().
        """
        active = self.manager.get_active_watchlist()
        self.manager.update_tickers(active["id"], tickers, backup_old=True)

    def import_tickers(self, tickers: List[str]) -> str:
        """Import tickers (merge, don't replace). Used by Bulk Save flow."""
        active = self.manager.get_active_watchlist()
        existing = set(active.get("tickers", []))
        merged = sorted(existing | set(t.upper().strip() for t in tickers if t.strip()))
        self.manager.update_tickers(active["id"], merged, backup_old=True)
        return f"Imported {len(merged) - len(existing)} new tickers"

    def export_tickers(self) -> List[str]:
        """Export all tickers from active watchlist."""
        return self.get_watchlist_tickers()

    # --- Favorites (delegated to JournalManager) ----------------------------
    # Favorites live in JM because they carry rich metadata (notes, focus labels)

    def get_favorite_tickers(self) -> List[str]:
        """Get favorites from JournalManager."""
        try:
            return self.jm.get_favorite_tickers()
        except Exception:
            return []

    def toggle_favorite(self, ticker: str) -> bool:
        """Toggle favorite status in JournalManager."""
        try:
            return self.jm.toggle_favorite(ticker)
        except Exception:
            return False

    def is_favorite(self, ticker: str) -> bool:
        """Check favorite status in JournalManager."""
        try:
            return self.jm.is_favorite(ticker)
        except Exception:
            return False


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    # Mock JournalManager for testing
    class MockJM:
        def __init__(self):
            self._watchlist = ["AAPL", "NVDA", "TSLA", "PLTR"]
            self._favorites = {"AAPL", "TSLA"}

        def get_watchlist_tickers(self):
            return list(self._watchlist)

        def get_favorite_tickers(self):
            return [t for t in self._watchlist if t in self._favorites]

        def add_to_watchlist(self, item):
            self._watchlist.append(item.ticker)

        def toggle_favorite(self, ticker):
            if ticker in self._favorites:
                self._favorites.discard(ticker)
                return False
            self._favorites.add(ticker)
            return True

        def is_favorite(self, ticker):
            return ticker in self._favorites

    test_path = os.path.join(tempfile.gettempdir(), "test_bridge.json")
    if os.path.exists(test_path):
        os.remove(test_path)

    mgr = WatchlistManager(test_path)
    mock_jm = MockJM()
    bridge = WatchlistBridge(mgr, mock_jm)

    # Test 1: Migration on first call
    tickers = bridge.get_watchlist_tickers()
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "TSLA" in tickers
    assert "PLTR" in tickers
    print("  1. Migration on first call (4 tickers)")

    # Test 2: Add ticker
    from journal_manager import WatchlistItem
    msg = bridge.add_to_watchlist(WatchlistItem(ticker="MSFT"))
    assert "MSFT" in bridge.get_watchlist_tickers()
    print(f"  2. Add ticker: {msg}")

    # Test 3: Remove ticker
    msg = bridge.remove_from_watchlist("PLTR")
    assert "PLTR" not in bridge.get_watchlist_tickers()
    print(f"  3. Remove ticker: {msg}")

    # Test 4: Favorites delegate to JM
    favs = bridge.get_favorite_tickers()
    assert "AAPL" in favs
    print(f"  4. Favorites from JM: {favs}")

    # Test 5: Clear with backup
    bridge.clear_watchlist()
    assert len(bridge.get_watchlist_tickers()) == 0
    # Rollback available
    wl = mgr.get_active_watchlist()
    assert wl.get("last_backup") is not None
    print("  5. Clear with backup")

    # Test 6: Re-migration blocked
    ok, msg = mgr.migrate_from_journal(mock_jm)
    assert not ok
    assert "Already migrated" in msg
    print(f"  6. Re-migration blocked: {msg}")

    os.remove(test_path)
    print("\n All 6 tests passed")
