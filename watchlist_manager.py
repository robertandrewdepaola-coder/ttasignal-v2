"""
TTA v2 Watchlist Manager — Multi-Watchlist System
===================================================

Core CRUD operations for managing multiple watchlists.
Stores data in v2_watchlist.json with atomic writes.

Features:
- Multiple named watchlists (manual + auto-populated)
- Protected Master watchlist (cannot be deleted)
- Ticker validation (US markets, class shares)
- Manual overrides on auto-populated lists
- Backup/rollback for auto-refresh safety
- One-time migration from JournalManager

Phase 1: ETF shortcuts, TradingView CSV, custom URL
Phase 2: EODHD/FMP screener APIs

Version: 1.0.0 (2026-02-14)
"""

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

REFRESH_COOLDOWN = 60       # seconds between auto-refreshes
MIN_TICKERS = 5             # warning threshold (still saves)
MAX_TICKERS = 500           # warning threshold (still saves)

MASTER_ID = "system-master-00000"

DEFAULT_METADATA = {
    "migrated_from_journal": False,
    "migration_date": None,
    "schema_version": "2.0",
}

DEFAULT_MASTER = {
    "id": MASTER_ID,
    "name": "Master Watchlist",
    "type": "manual",
    "is_system": True,
    "tickers": [],
    "created_at": None,
    "last_updated": None,
}

# Ticker extraction false positives — common words that match [A-Z]{1,5}
FALSE_POSITIVE_TICKERS = frozenset({
    "I", "A", "TO", "OR", "AND", "THE", "FOR", "CAN", "US", "OK",
    "URL", "API", "NYSE", "NASD", "ETF", "USD", "HTTP", "HTML",
    "JSON", "CSV", "YES", "NO", "FUND", "TOTAL", "CASH", "NAME",
    "DATE", "WEIGHT", "SHARE", "HTTPS", "NASDAQ", "TABLE", "INDEX",
    "CLOSE", "OPEN", "HIGH", "LOW", "VOL", "PCT", "AVG", "DAY",
})

# Foreign exchange suffixes to reject
FOREIGN_SUFFIXES = frozenset({"F", "TO", "L", "SW", "V", "O", "DE", "PA", "AS", "HK"})

# Valid US class-share suffixes
VALID_CLASS_SHARES = frozenset({"A", "B", "C", "D"})


# =============================================================================
# WATCHLIST MANAGER
# =============================================================================

class WatchlistManager:
    """
    Manages multiple watchlists stored in a JSON file.

    Each watchlist has a UUID, name, type (manual/auto), and ticker list.
    The system Master watchlist cannot be deleted.
    """

    def __init__(self, json_path: str = "v2_watchlist.json"):
        """Load watchlist data from JSON, or create fresh if missing/corrupted."""
        self.json_path = json_path
        self.data = self._load_or_create()

    # --- Persistence ---------------------------------------------------------

    def _load_or_create(self) -> Dict:
        """Load JSON file, recreate if missing or corrupted."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                if "watchlists" in data and isinstance(data["watchlists"], list):
                    data = self._ensure_metadata(data)
                    data = self._ensure_master(data)
                    return data
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[watchlist_manager] Corrupted JSON, recreating: {e}")

        now = datetime.now().isoformat()
        master = {**DEFAULT_MASTER, "created_at": now, "last_updated": now}
        return {
            "metadata": {**DEFAULT_METADATA},
            "active_watchlist_id": MASTER_ID,
            "watchlists": [master],
        }

    def _ensure_metadata(self, data: Dict) -> Dict:
        """Ensure metadata block exists with defaults."""
        if "metadata" not in data:
            data["metadata"] = {**DEFAULT_METADATA}
        else:
            for key, val in DEFAULT_METADATA.items():
                if key not in data["metadata"]:
                    data["metadata"][key] = val
        return data

    def _ensure_master(self, data: Dict) -> Dict:
        """Ensure system Master watchlist exists."""
        ids = {w["id"] for w in data["watchlists"]}
        if MASTER_ID not in ids:
            now = datetime.now().isoformat()
            master = {**DEFAULT_MASTER, "created_at": now, "last_updated": now}
            data["watchlists"].insert(0, master)
        return data

    def save(self) -> None:
        """Atomic write: write to .tmp then os.replace to prevent corruption."""
        temp_path = f"{self.json_path}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(temp_path, self.json_path)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to save watchlist data: {e}") from e

    # --- CRUD ----------------------------------------------------------------

    def create_watchlist(
        self,
        name: str,
        wl_type: str,
        source_type: Optional[str] = None,
        source: Optional[str] = None,
        provider: str = "direct",
    ) -> Optional[str]:
        """
        Create a new watchlist.

        Args:
            name: Display name (must be unique, non-empty).
            wl_type: 'manual' or 'auto'.
            source_type: For auto: 'etf_shortcut', 'etf_url', 'tradingview', 'custom_url'.
            source: The ETF key, URL, or file path.
            provider: 'direct' (Phase 1).

        Returns:
            Watchlist UUID string, or None on error.
        """
        if not name or not name.strip():
            raise ValueError("Watchlist name cannot be empty")

        name = name.strip()
        if any(w["name"].lower() == name.lower() for w in self.data["watchlists"]):
            raise ValueError(f"Watchlist '{name}' already exists")

        if wl_type == "auto" and not source_type:
            raise ValueError("Auto watchlists require a source_type")

        now = datetime.now().isoformat()
        wl_id = str(uuid.uuid4())

        watchlist = {
            "id": wl_id,
            "name": name,
            "type": wl_type,
            "is_system": False,
            "tickers": [],
            "created_at": now,
            "last_updated": now,
        }

        if wl_type == "auto":
            watchlist.update({
                "source_type": source_type,
                "source": source or "",
                "provider": provider,
                "last_refresh": None,
                "refresh_schedule": "manual",
                "last_backup": None,
                "scraping_stats": {
                    "total_refreshes": 0,
                    "successful_refreshes": 0,
                    "failed_refreshes": 0,
                    "last_error": None,
                    "last_error_date": None,
                },
                "manual_additions": [],
                "manual_exclusions": [],
            })

        self.data["watchlists"].append(watchlist)
        self.save()
        return wl_id

    def delete_watchlist(self, wl_id: str) -> Tuple[bool, str]:
        """
        Delete a watchlist by ID.

        Cannot delete system watchlist or the last remaining watchlist.
        If deleting the active watchlist, switches to Master.
        """
        wl = self._find(wl_id)
        if not wl:
            return False, f"Watchlist not found: {wl_id}"

        if wl.get("is_system"):
            return False, "Cannot delete the Master watchlist"

        if len(self.data["watchlists"]) <= 1:
            return False, "Cannot delete the last watchlist"

        self.data["watchlists"] = [w for w in self.data["watchlists"] if w["id"] != wl_id]

        if self.data["active_watchlist_id"] == wl_id:
            self.data["active_watchlist_id"] = MASTER_ID

        self.save()
        return True, f"Deleted '{wl['name']}'"

    def update_watchlist_metadata(self, wl_id: str, **kwargs) -> Tuple[bool, str]:
        """Update name, source, provider, or other metadata fields."""
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found"

        allowed = {"name", "source", "source_type", "provider", "refresh_schedule"}
        for key, val in kwargs.items():
            if key in allowed:
                wl[key] = val

        wl["last_updated"] = datetime.now().isoformat()
        self.save()
        return True, "Metadata updated"

    # --- Ticker Management ---------------------------------------------------

    def update_tickers(
        self, wl_id: str, tickers: List[str], backup_old: bool = True
    ) -> Tuple[bool, str, List[str]]:
        """
        Replace a watchlist's tickers with a new cleaned list.

        Backs up old tickers before replacing (for rollback).
        Applies manual overrides if auto watchlist.
        Returns (success, message, cleaned_tickers).
        """
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found", []

        if backup_old and wl["tickers"]:
            wl["last_backup"] = {
                "tickers": list(wl["tickers"]),
                "timestamp": datetime.now().isoformat(),
            }

        cleaned = self._clean_ticker_list(tickers)

        if wl.get("type") == "auto":
            additions = wl.get("manual_additions", [])
            exclusions = wl.get("manual_exclusions", [])
            cleaned = self._apply_manual_overrides(cleaned, additions, exclusions)

        wl["tickers"] = cleaned
        wl["last_updated"] = datetime.now().isoformat()

        msg = f"Updated with {len(cleaned)} tickers"
        if len(cleaned) < MIN_TICKERS:
            msg += f" (below {MIN_TICKERS} minimum)"
        elif len(cleaned) > MAX_TICKERS:
            msg += f" (above {MAX_TICKERS} maximum)"

        self.save()
        return True, msg, cleaned

    def add_manual_ticker(self, wl_id: str, ticker: str) -> Tuple[bool, str]:
        """Add a single ticker to a watchlist."""
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found"

        ticker = ticker.upper().strip()
        if not self._validate_us_ticker(ticker):
            return False, f"Invalid ticker: {ticker}"

        if ticker in wl["tickers"]:
            return False, f"{ticker} already in watchlist"

        if wl.get("type") == "auto":
            additions = wl.get("manual_additions", [])
            if ticker not in additions:
                additions.append(ticker)
                wl["manual_additions"] = additions
            exclusions = wl.get("manual_exclusions", [])
            if ticker in exclusions:
                exclusions.remove(ticker)
                wl["manual_exclusions"] = exclusions

        wl["tickers"] = sorted(set(wl["tickers"] + [ticker]))
        wl["last_updated"] = datetime.now().isoformat()
        self.save()
        return True, f"Added {ticker}"

    def remove_manual_ticker(self, wl_id: str, ticker: str) -> Tuple[bool, str]:
        """Remove a single ticker from a watchlist."""
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found"

        ticker = ticker.upper().strip()
        if ticker not in wl["tickers"]:
            return False, f"{ticker} not in watchlist"

        if wl.get("type") == "auto":
            exclusions = wl.get("manual_exclusions", [])
            if ticker not in exclusions:
                exclusions.append(ticker)
                wl["manual_exclusions"] = exclusions
            additions = wl.get("manual_additions", [])
            if ticker in additions:
                additions.remove(ticker)
                wl["manual_additions"] = additions

        wl["tickers"] = [t for t in wl["tickers"] if t != ticker]
        wl["last_updated"] = datetime.now().isoformat()
        self.save()
        return True, f"Removed {ticker}"

    def rollback_tickers(self, wl_id: str) -> Tuple[bool, str]:
        """Restore tickers from last_backup. Does NOT clear backup."""
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found"

        backup = wl.get("last_backup")
        if not backup or not backup.get("tickers"):
            return False, "No backup available"

        wl["tickers"] = list(backup["tickers"])
        wl["last_updated"] = datetime.now().isoformat()
        self.save()
        return True, f"Rolled back to {len(backup['tickers'])} tickers (from {backup.get('timestamp', 'unknown')})"

    # --- Getters -------------------------------------------------------------

    def get_watchlist(self, wl_id: str) -> Optional[Dict]:
        """Get a single watchlist by ID."""
        return self._find(wl_id)

    def get_all_watchlists(self) -> List[Dict]:
        """Get all watchlists."""
        return self.data["watchlists"]

    def get_active_watchlist(self) -> Dict:
        """Get the currently active watchlist, falling back to Master."""
        active_id = self.data.get("active_watchlist_id", MASTER_ID)
        wl = self._find(active_id)
        if wl:
            return wl
        self.data["active_watchlist_id"] = MASTER_ID
        master = self._find(MASTER_ID)
        if master:
            return master
        return {**DEFAULT_MASTER, "tickers": []}

    def set_active_watchlist(self, wl_id: str) -> Tuple[bool, str]:
        """Set the active watchlist by ID."""
        wl = self._find(wl_id)
        if not wl:
            return False, "Watchlist not found"
        self.data["active_watchlist_id"] = wl_id
        self.save()
        return True, f"Switched to '{wl['name']}'"

    # --- Auto-Refresh Controls -----------------------------------------------

    def can_refresh_auto(self, wl_id: str) -> Tuple[bool, int]:
        """Check if auto-refresh is allowed. Returns (can_refresh, seconds_remaining)."""
        wl = self._find(wl_id)
        if not wl:
            return False, 0

        last_refresh = wl.get("last_refresh")
        if not last_refresh:
            return True, 0

        try:
            last_dt = datetime.fromisoformat(last_refresh)
            elapsed = (datetime.now() - last_dt).total_seconds()
            remaining = max(0, REFRESH_COOLDOWN - elapsed)
            return remaining <= 0, int(remaining)
        except (ValueError, TypeError):
            return True, 0

    def record_refresh_request(self, wl_id: str) -> None:
        """Mark that a refresh was started."""
        wl = self._find(wl_id)
        if wl:
            wl["last_refresh"] = datetime.now().isoformat()
            stats = wl.get("scraping_stats", {})
            stats["total_refreshes"] = stats.get("total_refreshes", 0) + 1
            wl["scraping_stats"] = stats
            self.save()

    def record_refresh_success(self, wl_id: str) -> None:
        """Record successful refresh."""
        wl = self._find(wl_id)
        if wl:
            stats = wl.get("scraping_stats", {})
            stats["successful_refreshes"] = stats.get("successful_refreshes", 0) + 1
            stats["last_error"] = None
            wl["scraping_stats"] = stats
            self.save()

    def record_refresh_failure(self, wl_id: str, error: str) -> None:
        """Record failed refresh."""
        wl = self._find(wl_id)
        if wl:
            stats = wl.get("scraping_stats", {})
            stats["failed_refreshes"] = stats.get("failed_refreshes", 0) + 1
            stats["last_error"] = error[:200]
            stats["last_error_date"] = datetime.now().isoformat()
            wl["scraping_stats"] = stats
            self.save()

    # --- Migration -----------------------------------------------------------

    def migrate_from_journal(self, journal_manager) -> Tuple[bool, str]:
        """
        One-time migration from existing JournalManager watchlist.

        Checks metadata flag to prevent re-migration.
        Copies tickers to Master watchlist, deduplicates and uppercases.
        """
        meta = self.data.get("metadata", {})
        if meta.get("migrated_from_journal"):
            return False, "Already migrated"

        try:
            old_tickers = journal_manager.get_watchlist_tickers()
            if not old_tickers:
                self.data["metadata"]["migrated_from_journal"] = True
                self.data["metadata"]["migration_date"] = datetime.now().isoformat()
                self.save()
                return False, "No tickers to migrate"

            cleaned = list(set(t.upper().strip() for t in old_tickers if t and t.strip()))
            cleaned = [t for t in cleaned if self._validate_us_ticker(t)]
            cleaned = sorted(cleaned)

            master = self._find(MASTER_ID)
            if master:
                existing = set(master["tickers"])
                master["tickers"] = sorted(existing | set(cleaned))
                master["last_updated"] = datetime.now().isoformat()

            self.data["metadata"]["migrated_from_journal"] = True
            self.data["metadata"]["migration_date"] = datetime.now().isoformat()
            self.save()
            return True, f"Migrated {len(cleaned)} tickers to Master Watchlist"

        except Exception as e:
            return False, f"Migration failed: {str(e)}"

    # --- Validation Helpers (Private) ----------------------------------------

    def _validate_us_ticker(self, ticker: str) -> bool:
        """
        Validate a US market ticker symbol.

        Allows: AAPL, NVDA, BRK.B, GOOG.A (1-5 chars, optional .A-.D)
        Rejects: foreign suffixes (.F, .TO, .L, etc.), false positives
        """
        if not ticker or ticker in FALSE_POSITIVE_TICKERS:
            return False

        if not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
            return False

        if '.' in ticker:
            suffix = ticker.split('.')[1]
            if suffix in FOREIGN_SUFFIXES:
                return False
            if suffix not in VALID_CLASS_SHARES:
                return False

        return True

    def _clean_ticker_list(self, tickers: List[str]) -> List[str]:
        """Clean, validate, deduplicate, and sort a ticker list."""
        seen = set()
        cleaned = []
        for t in tickers:
            if not t or not isinstance(t, str):
                continue
            t = t.upper().strip()
            if t and t not in seen and self._validate_us_ticker(t):
                seen.add(t)
                cleaned.append(t)
        return sorted(cleaned)

    def _apply_manual_overrides(
        self, base: List[str], additions: List[str], exclusions: List[str]
    ) -> List[str]:
        """
        Apply manual overrides to auto-populated ticker list.

        Logic: (base - exclusions) + (additions - base - exclusions)
        Exclusions override everything.
        """
        base_set = {t.upper() for t in base}
        additions_set = {t.upper() for t in additions}
        exclusions_set = {t.upper() for t in exclusions}

        result = base_set - exclusions_set
        new_additions = additions_set - base_set - exclusions_set
        result.update(new_additions)
        return sorted(result)

    def _find(self, wl_id: str) -> Optional[Dict]:
        """Find a watchlist by ID. Returns the dict reference (mutable)."""
        for wl in self.data["watchlists"]:
            if wl["id"] == wl_id:
                return wl
        return None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    test_path = os.path.join(tempfile.gettempdir(), "test_watchlist.json")

    if os.path.exists(test_path):
        os.remove(test_path)

    mgr = WatchlistManager(test_path)

    # Test 1: Master exists
    master = mgr.get_active_watchlist()
    assert master["id"] == MASTER_ID and master["is_system"] is True
    print("  1. Master watchlist created")

    # Test 2: Add tickers
    ok, _ = mgr.add_manual_ticker(MASTER_ID, "AAPL")
    assert ok
    ok, _ = mgr.add_manual_ticker(MASTER_ID, "BRK.B")
    assert ok
    assert len(mgr.get_active_watchlist()["tickers"]) == 2
    print("  2. Manual ticker add (including BRK.B)")

    # Test 3: Reject invalid
    ok, _ = mgr.add_manual_ticker(MASTER_ID, "AAPL.F")
    assert not ok
    ok, _ = mgr.add_manual_ticker(MASTER_ID, "ABC.Z")
    assert not ok
    ok, _ = mgr.add_manual_ticker(MASTER_ID, "AND")
    assert not ok
    print("  3. Invalid ticker rejection")

    # Test 4: Cannot delete Master
    ok, _ = mgr.delete_watchlist(MASTER_ID)
    assert not ok
    print("  4. Master deletion prevented")

    # Test 5: Create auto watchlist + overrides
    auto_id = mgr.create_watchlist("ARKK", "auto", source_type="etf_shortcut", source="arkk")
    mgr.update_tickers(auto_id, ["TSLA", "CRSP", "TEM"])
    mgr.add_manual_ticker(auto_id, "AAPL")
    mgr.remove_manual_ticker(auto_id, "TEM")
    wl = mgr.get_watchlist(auto_id)
    assert "AAPL" in wl["manual_additions"]
    assert "TEM" in wl["manual_exclusions"]
    print("  5. Auto watchlist + manual overrides")

    # Test 6: Rollback
    mgr.update_tickers(auto_id, ["TSLA", "CRSP", "TEM", "ROKU", "SHOP"])
    ok, _ = mgr.rollback_tickers(auto_id)
    assert ok
    wl = mgr.get_watchlist(auto_id)
    assert wl["last_backup"] is not None  # Preserved
    print("  6. Rollback (backup preserved)")

    # Test 7: Delete active -> switch to Master
    mgr.set_active_watchlist(auto_id)
    mgr.delete_watchlist(auto_id)
    assert mgr.get_active_watchlist()["id"] == MASTER_ID
    print("  7. Delete active -> auto-switch to Master")

    # Test 8: Atomic save
    with open(test_path) as f:
        saved = json.load(f)
    assert saved["metadata"]["schema_version"] == "2.0"
    print("  8. Atomic save verified")

    os.remove(test_path)
    print("\n All 8 tests passed")
