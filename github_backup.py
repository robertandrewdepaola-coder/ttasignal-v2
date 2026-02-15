"""
GitHub Auto-Backup — Data persistence across Streamlit Cloud reboots
====================================================================

Streamlit Cloud destroys the container on reboot, wiping all data files.
This module pushes data files to a `data-backup` branch in the GitHub repo
so they can be restored on next startup.

Uses the `data-backup` branch specifically to avoid triggering Streamlit
redeployments (which only watch `main`).

Setup:
    1. Create a GitHub Personal Access Token (classic) with `repo` scope
    2. Add to Streamlit secrets as GITHUB_TOKEN
    3. Add GITHUB_REPO as "owner/repo" (e.g., "robertdepaola/ttasignal-v2")

Usage (automatic — hooked into app.py):
    - On startup: restore_all() pulls missing files from GitHub
    - After data changes: mark_dirty(filename) queues a backup
    - Periodically: flush_dirty() pushes queued files (debounced)

Version: 1.0.0
"""

import os
import json
import base64
import time
from typing import Optional, Dict, Set
from pathlib import Path

# ── Files to back up ────────────────────────────────────────────────────────
BACKUP_FILES = [
    "v2_watchlist.json",
    "v2_open_trades.json",
    "v2_trade_history.json",
    "v2_conditionals.json",
    "v2_multi_watchlist.json",
    "v2_scan_cache.json",
]

BACKUP_BRANCH = "data-backup"
DEBOUNCE_SECONDS = 30  # Minimum seconds between pushes per file
GITHUB_API = "https://api.github.com"


class GitHubBackup:
    """Manages backup/restore of data files to GitHub."""

    def __init__(self, token: str, repo: str, data_dir: str = "."):
        """
        Args:
            token: GitHub Personal Access Token with repo scope
            repo: "owner/repo" format (e.g., "robertdepaola/ttasignal-v2")
            data_dir: Local directory where data files live
        """
        self.token = token
        self.repo = repo
        self.data_dir = Path(data_dir)
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TTA-Signal-v2",
        }
        # Track file SHAs for update operations (GitHub requires SHA to update)
        self._sha_cache: Dict[str, str] = {}
        # Track last push time per file for debouncing
        self._last_push: Dict[str, float] = {}
        # Dirty files waiting to be pushed
        self._dirty: Set[str] = set()
        # Whether the data-backup branch has been verified/created
        self._branch_verified = False
        self._enabled = bool(token and repo)

        if not self._enabled:
            print("[backup] GitHub backup disabled — missing GITHUB_TOKEN or GITHUB_REPO")

    # ── Public API ──────────────────────────────────────────────────────

    def restore_all(self) -> Dict[str, bool]:
        """
        Restore all data files from GitHub if they don't exist locally.

        Called once on startup. Only downloads files that are missing locally
        to avoid overwriting in-session changes.

        Returns dict of {filename: restored_bool}.
        """
        if not self._enabled:
            return {}

        results = {}
        for filename in BACKUP_FILES:
            local_path = self.data_dir / filename
            if local_path.exists() and local_path.stat().st_size > 2:
                # File already exists locally — don't overwrite
                results[filename] = False
                continue
            try:
                restored = self._restore_file(filename)
                results[filename] = restored
                if restored:
                    print(f"[backup] Restored {filename} from GitHub")
            except Exception as e:
                print(f"[backup] Failed to restore {filename}: {e}")
                results[filename] = False

        restored_count = sum(1 for v in results.values() if v)
        if restored_count > 0:
            print(f"[backup] Restored {restored_count}/{len(BACKUP_FILES)} data files from GitHub")
        return results

    def mark_dirty(self, filename: str):
        """Mark a file as needing backup. Call after any data save."""
        if self._enabled and filename in BACKUP_FILES:
            self._dirty.add(filename)

    def flush_dirty(self) -> int:
        """
        Push all dirty files that have passed the debounce window.

        Returns number of files pushed.
        """
        if not self._enabled or not self._dirty:
            return 0

        pushed = 0
        now = time.time()
        still_dirty = set()

        for filename in self._dirty:
            last = self._last_push.get(filename, 0)
            if now - last < DEBOUNCE_SECONDS:
                still_dirty.add(filename)  # Not yet — keep in queue
                continue
            try:
                if self._push_file(filename):
                    pushed += 1
                    self._last_push[filename] = now
            except Exception as e:
                print(f"[backup] Push failed for {filename}: {e}")
                still_dirty.add(filename)  # Retry next flush

        self._dirty = still_dirty
        if pushed > 0:
            print(f"[backup] Pushed {pushed} file(s) to GitHub")
        return pushed

    def force_backup_all(self) -> int:
        """Force push all data files immediately (ignores debounce)."""
        if not self._enabled:
            return 0

        pushed = 0
        for filename in BACKUP_FILES:
            local_path = self.data_dir / filename
            if local_path.exists() and local_path.stat().st_size > 2:
                try:
                    if self._push_file(filename):
                        pushed += 1
                        self._last_push[filename] = time.time()
                except Exception as e:
                    print(f"[backup] Force push failed for {filename}: {e}")
        return pushed

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def pending_count(self) -> int:
        return len(self._dirty)

    # ── Internal Methods ────────────────────────────────────────────────

    def _ensure_branch(self):
        """Create data-backup branch if it doesn't exist."""
        if self._branch_verified:
            return

        import requests

        # Check if branch exists
        url = f"{GITHUB_API}/repos/{self.repo}/branches/{BACKUP_BRANCH}"
        resp = requests.get(url, headers=self.headers, timeout=10)

        if resp.status_code == 200:
            self._branch_verified = True
            return

        if resp.status_code == 404:
            # Create branch from main
            print(f"[backup] Creating '{BACKUP_BRANCH}' branch...")
            # Get main branch SHA
            main_url = f"{GITHUB_API}/repos/{self.repo}/git/refs/heads/main"
            main_resp = requests.get(main_url, headers=self.headers, timeout=10)
            if main_resp.status_code != 200:
                print(f"[backup] Can't find main branch: {main_resp.status_code}")
                return

            main_sha = main_resp.json()["object"]["sha"]

            # Create new branch ref
            create_url = f"{GITHUB_API}/repos/{self.repo}/git/refs"
            create_data = {
                "ref": f"refs/heads/{BACKUP_BRANCH}",
                "sha": main_sha,
            }
            create_resp = requests.post(
                create_url, headers=self.headers,
                json=create_data, timeout=10
            )
            if create_resp.status_code in (200, 201):
                print(f"[backup] Created '{BACKUP_BRANCH}' branch ✓")
                self._branch_verified = True
            else:
                print(f"[backup] Failed to create branch: {create_resp.status_code}")

    def _push_file(self, filename: str) -> bool:
        """Push a single file to GitHub."""
        import requests

        local_path = self.data_dir / filename
        if not local_path.exists():
            return False

        self._ensure_branch()

        # Read local file
        with open(local_path, 'rb') as f:
            content = f.read()

        encoded = base64.b64encode(content).decode('utf-8')
        github_path = f"data/{filename}"

        # Get existing file SHA (needed for updates)
        sha = self._get_remote_sha(github_path)

        url = f"{GITHUB_API}/repos/{self.repo}/contents/{github_path}"
        payload = {
            "message": f"Auto-backup: {filename}",
            "content": encoded,
            "branch": BACKUP_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=self.headers, json=payload, timeout=15)

        if resp.status_code in (200, 201):
            # Update SHA cache
            new_sha = resp.json().get("content", {}).get("sha")
            if new_sha:
                self._sha_cache[github_path] = new_sha
            return True
        else:
            error_msg = resp.json().get("message", "unknown error")
            print(f"[backup] GitHub push error for {filename}: {resp.status_code} — {error_msg}")
            # If SHA mismatch (409), clear cache and retry once
            if resp.status_code == 409:
                self._sha_cache.pop(github_path, None)
            return False

    def _restore_file(self, filename: str) -> bool:
        """Pull a single file from GitHub and write locally."""
        import requests

        self._ensure_branch()

        github_path = f"data/{filename}"
        url = f"{GITHUB_API}/repos/{self.repo}/contents/{github_path}"
        params = {"ref": BACKUP_BRANCH}

        resp = requests.get(url, headers=self.headers, params=params, timeout=10)

        if resp.status_code != 200:
            return False

        data = resp.json()
        if data.get("encoding") != "base64" or not data.get("content"):
            return False

        # Decode and write
        content = base64.b64decode(data["content"])

        # Validate it's proper JSON before writing
        try:
            json.loads(content)
        except (json.JSONDecodeError, ValueError):
            print(f"[backup] Corrupt backup for {filename} — skipping")
            return False

        local_path = self.data_dir / filename
        with open(local_path, 'wb') as f:
            f.write(content)

        # Cache the SHA for future updates
        if data.get("sha"):
            self._sha_cache[github_path] = data["sha"]

        return True

    def _get_remote_sha(self, github_path: str) -> Optional[str]:
        """Get the SHA of a file on GitHub (needed for updates)."""
        # Check cache first
        if github_path in self._sha_cache:
            return self._sha_cache[github_path]

        import requests

        url = f"{GITHUB_API}/repos/{self.repo}/contents/{github_path}"
        params = {"ref": BACKUP_BRANCH}
        resp = requests.get(url, headers=self.headers, params=params, timeout=10)

        if resp.status_code == 200:
            sha = resp.json().get("sha")
            if sha:
                self._sha_cache[github_path] = sha
            return sha
        return None  # File doesn't exist yet — will create


# =============================================================================
# SINGLETON — one instance shared across Streamlit reruns
# =============================================================================

_backup_instance: Optional[GitHubBackup] = None


def get_backup(force_new: bool = False) -> Optional[GitHubBackup]:
    """
    Get or create the GitHubBackup singleton.

    Reads GITHUB_TOKEN and GITHUB_REPO from Streamlit secrets.
    Returns None if secrets aren't configured.
    """
    global _backup_instance

    if _backup_instance is not None and not force_new:
        return _backup_instance

    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")
        repo = os.environ.get("GITHUB_REPO", "")

    if not token or not repo:
        return None

    _backup_instance = GitHubBackup(token=token, repo=repo)
    return _backup_instance


def mark_dirty(filename: str):
    """Convenience wrapper — marks a file for backup."""
    backup = get_backup()
    if backup:
        backup.mark_dirty(filename)


def flush():
    """Convenience wrapper — pushes dirty files past debounce window."""
    backup = get_backup()
    if backup:
        return backup.flush_dirty()
    return 0


def restore_all():
    """Convenience wrapper — restores missing files on startup."""
    backup = get_backup()
    if backup:
        return backup.restore_all()
    return {}
