"""Backup health helpers for sidebar diagnostics and actions."""

from __future__ import annotations

from typing import Any, Dict


def get_backup_health_status() -> Dict[str, Any]:
    """Return normalized backup status payload for UI."""
    try:
        import github_backup

        raw = github_backup.status() or {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "pending_count": int(raw.get("pending_count", 0) or 0),
            "last_success_epoch": float(raw.get("last_success_epoch", 0.0) or 0.0),
            "branch": str(raw.get("branch", "data-backup") or "data-backup"),
            "last_error": str(raw.get("last_error", "") or ""),
            "last_error_code": int(raw.get("last_error_code", 0) or 0),
            "available": True,
        }
    except Exception as exc:
        return {
            "enabled": False,
            "pending_count": 0,
            "last_success_epoch": 0.0,
            "branch": "data-backup",
            "last_error": str(exc),
            "last_error_code": 0,
            "available": False,
        }


def run_backup_now() -> int:
    """Run backup push now, falling back to force push when queue is empty."""
    import github_backup

    pushed = int(github_backup.flush() or 0)
    if pushed <= 0:
        pushed = int(github_backup.force_backup_all() or 0)
    return pushed
