#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] compiling core modules..."
python3 -m py_compile app.py data_fetcher.py journal_manager.py watchlist_manager.py scan_utils.py

echo "[smoke] running regression tests..."
python3 -m unittest discover -s tests -p "test_*.py"

echo "[smoke] OK"

