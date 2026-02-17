# Executive Dashboard Rollout Plan

## Feature Flag
- Runtime toggle in sidebar: `Executive Dashboard Beta`
- Secrets override: `EXEC_DASHBOARD_BETA=true|false`

## Measurement Window
- Run for 3-5 trading days with Beta enabled.
- Keep watchlist/scan/alert flows unchanged outside dashboard usage.

## KPIs
- Time-to-decision proxy:
  - First `ENTER_TRADE` timestamp minus first `FAST_REFRESH`/`SCAN_DONE` of day.
- Operational load:
  - `scan_all` count
  - `scan_new` count
  - `fast_refresh` count
- Reliability:
  - alert checks/day
  - stale streams observed
  - SLO breaches (`scripts/check_slos.py`)

## Daily Report Command
```bash
python3 scripts/executive_kpi_report.py
```

## Promotion Criteria
- No critical watchlist/scan regressions.
- SLO checks pass for representative sample.
- Reduced manual triage (fewer full scans, more fast refresh + ranked actions).

