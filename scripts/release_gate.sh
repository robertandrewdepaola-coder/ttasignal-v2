#!/usr/bin/env bash
set -euo pipefail

echo "[gate] running smoke tests..."
scripts/run_smoke_tests.sh

echo "[gate] running SLO checks..."
python3 scripts/check_slos.py

echo "[gate] PASS"

