"""
TTA v2 Emergency Rollback Script
==================================

If the new multi-watchlist system causes ANY issues with the working app,
run this script to instantly revert to the old behavior.

What it does:
1. Disables the new watchlist system by renaming files
2. Restores original app.py behavior (jm.get_watchlist_tickers() works as before)
3. Preserves the new files as .disabled (not deleted) for debugging

Usage:
    python rollback_watchlist.py

To re-enable after fixing:
    python rollback_watchlist.py --reenable

Version: 1.0.0 (2026-02-14)
"""

import os
import sys
import shutil
from datetime import datetime

# Files that are part of the new watchlist system
NEW_FILES = [
    "watchlist_manager.py",
    "scraping_bridge.py",
    "app_watchlist_ui.py",
    "watchlist_bridge.py",
]

# Data file
DATA_FILE = "v2_watchlist.json"


def rollback():
    """Disable the new watchlist system by renaming files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  TTA v2 EMERGENCY ROLLBACK — {timestamp}")
    print(f"{'='*60}\n")

    disabled = []
    for fname in NEW_FILES:
        if os.path.exists(fname):
            disabled_name = f"{fname}.disabled_{timestamp}"
            os.rename(fname, disabled_name)
            disabled.append(f"  {fname} -> {disabled_name}")
        else:
            disabled.append(f"  {fname} (not found, skipping)")

    # Preserve data file
    if os.path.exists(DATA_FILE):
        backup_name = f"{DATA_FILE}.backup_{timestamp}"
        shutil.copy2(DATA_FILE, backup_name)
        disabled.append(f"  {DATA_FILE} backed up to {backup_name}")

    print("Files disabled:")
    for line in disabled:
        print(line)

    print(f"\n{'='*60}")
    print("  ROLLBACK COMPLETE")
    print("  The app will now use jm.get_watchlist_tickers() as before.")
    print("  Restart the Streamlit app to take effect.")
    print(f"{'='*60}\n")
    print("  To re-enable: python rollback_watchlist.py --reenable")


def reenable():
    """Re-enable the most recently disabled watchlist files."""
    print(f"\n{'='*60}")
    print(f"  TTA v2 RE-ENABLE WATCHLIST SYSTEM")
    print(f"{'='*60}\n")

    restored = []
    for fname in NEW_FILES:
        # Find the most recent .disabled_ version
        candidates = sorted(
            [f for f in os.listdir('.') if f.startswith(f"{fname}.disabled_")],
            reverse=True,
        )
        if candidates:
            latest = candidates[0]
            if os.path.exists(fname):
                os.remove(fname)  # Remove any partial copy
            os.rename(latest, fname)
            restored.append(f"  {latest} -> {fname}")
        else:
            restored.append(f"  {fname} (no disabled version found)")

    print("Files restored:")
    for line in restored:
        print(line)

    print(f"\n  Restart the Streamlit app to take effect.\n")


if __name__ == "__main__":
    if "--reenable" in sys.argv:
        reenable()
    else:
        rollback()
