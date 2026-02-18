"""System self-test helpers used by the sidebar diagnostics panel."""

from __future__ import annotations

from typing import Any, Dict, List



def run_system_self_test(health: Dict[str, Any], backup: Dict[str, Any]) -> Dict[str, Any]:
    """Run deterministic health checks and return a compact report."""
    checks: List[Dict[str, Any]] = []

    ai_ok = bool(health.get("ai_ok", False))
    checks.append({
        "name": "AI connectivity",
        "ok": ai_ok,
        "detail": "AI provider reachable" if ai_ok else "AI provider unavailable",
    })

    feed_ok = bool(health.get("price_feed_ok", False))
    checks.append({
        "name": "Price feed",
        "ok": feed_ok,
        "detail": "Live prices available" if feed_ok else "Price feed unavailable",
    })

    market_time_ok = bool(str(health.get("market_et_time", "") or "").strip())
    checks.append({
        "name": "Market clock",
        "ok": market_time_ok,
        "detail": str(health.get("market_et_time", "missing")),
    })

    backup_enabled = bool(backup.get("enabled", False))
    backup_last_success = float(backup.get("last_success_epoch", 0.0) or 0.0) > 0
    backup_error = str(backup.get("last_error", "") or "")
    backup_ok = backup_enabled and backup_last_success and not backup_error
    checks.append({
        "name": "Backup persistence",
        "ok": backup_ok,
        "detail": (
            "Backup is healthy"
            if backup_ok
            else f"enabled={backup_enabled}, last_success={backup_last_success}, error={backup_error or 'none'}"
        ),
    })

    overall_ok = all(bool(c.get("ok")) for c in checks)
    failures = [c for c in checks if not c.get("ok")]

    return {
        "overall_ok": overall_ok,
        "checks": checks,
        "failures": failures,
        "summary": "PASS" if overall_ok else f"FAIL ({len(failures)} issue(s))",
    }
