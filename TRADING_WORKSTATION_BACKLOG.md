# Trading Workstation Backlog

Owner: discretionary-system trader workflow  
Date: 2026-02-17

## P0 (Immediate)

### P0-1 Unified Decision Card (in progress)
- Goal: One canonical decision object used across Scanner, Trade Finder, Executive Dashboard, and Trade entry.
- Build:
  - `trade_decision.py` with `TradeDecisionCard`.
  - Normalize recommendation/risk/reward/regime-fit/readiness/explainability into one payload.
  - Replace ad-hoc candidate dictionaries where possible.
- Files:
  - `trade_decision.py`
  - `app.py` (`_run_trade_finder_workflow`, `render_trade_finder_tab`, `render_executive_dashboard`)
- Acceptance:
  - Candidate shown in Trade Finder and Exec includes identical readiness/risk metadata.
  - Clicking candidate to trade preserves decision context.

### P0-2 Auto Exit Engine
- Goal: Enforce stops/targets reliably without depending on manual tab checks.
- Build:
  - Add optional scheduled/manual “Auto Manage Now” command in Executive Dashboard.
  - Call `JournalManager.check_stops(..., auto_execute=True)` on refreshed price map.
  - Log each auto close with reason and timestamp.
- Files:
  - `app.py` (dashboard controls and workflow)
  - `journal_manager.py` (idempotent close safety checks)
- Acceptance:
  - Stop breach during refresh closes trade automatically and appears in history.
  - No duplicate close records.

### P0-3 Planned Trades Board
- Goal: Stage high-quality ideas before committing capital.
- Build:
  - New persistence file: `v2_planned_trades.json`.
  - CRUD: add from Trade Finder/Scanner, mark `PLANNED|TRIGGERED|CANCELLED|ENTERED`.
  - “Promote to Entry” action opens prefilled Trade tab.
- Files:
  - `journal_manager.py`
  - `app.py` (new panel/tab)
- Acceptance:
  - User can queue, prioritize, and execute from plan list without rescanning.

## P1 (Next)

### P1-1 Performance Intelligence (alpha-focused)
- Goal: Determine what actually compounds.
- Build:
  - Add expectancy, payoff ratio, MAE/MFE approximation, regime-bucket stats.
  - Compare realized results vs SPY benchmark over same hold windows.
- Files:
  - `journal_manager.py` (`get_performance_stats` extension)
  - `app.py` (`render_performance` visuals)
- Acceptance:
  - Performance view answers: “Which setups beat benchmark and under what conditions?”

### P1-2 Intraday Command Center
- Goal: One-screen action routing.
- Build:
  - Consolidate triggered alerts, stop-near breaches, stale-risk data, and top entries.
  - Add “action queue” ordering by urgency and P&L risk.
- Files:
  - `app.py` (`render_executive_dashboard`)
- Acceptance:
  - Trader can process urgent items without tab hopping.

## P2 (Stabilization / Quality)

### P2-1 Explainability + Audit
- Goal: Post-trade decision quality review.
- Build:
  - Persist “why entered/why exited/why overridden”.
  - Link each trade to originating decision card snapshot.
- Files:
  - `journal_manager.py`
  - `app.py`
- Acceptance:
  - Each closed trade has machine-readable decision provenance.

### P2-2 UX friction cleanup
- Goal: Reduce cognitive overhead.
- Build:
  - Eliminate duplicate/conflicting labels.
  - Standardize recommendation vocabulary and color semantics.
- Files:
  - `app.py` UI sections
- Acceptance:
  - No contradictory risk messages for the same timestamped state.
