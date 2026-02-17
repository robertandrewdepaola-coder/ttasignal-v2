# Release Freeze Checklist

Use this checklist before merging dashboard/runtime changes to `main`.

## Freeze
- [ ] Freeze new feature PRs.
- [ ] Scope only bug fixes, SLO fixes, and regression fixes.
- [ ] Announce freeze window and rollback commit hash.

## Gates
- [ ] Run `scripts/run_smoke_tests.sh`
- [ ] Run `python3 scripts/check_slos.py --strict`
- [ ] Run `scripts/release_gate.sh` (non-strict baseline)

## Runtime Validation
- [ ] Verify `Executive Dashboard Beta` flag behavior (on/off).
- [ ] Verify `Scan New` only scans new universe members.
- [ ] Verify earnings dates roll forward after past report.
- [ ] Verify alert trigger state persists after reload.

## Rollback
- [ ] Record release commit hash.
- [ ] Record previous known-good hash.
- [ ] If needed: `git revert <release_hash>` and redeploy.

