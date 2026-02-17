"""
Scan utilities shared between app runtime and regression tests.
"""

from typing import Dict, List


def resolve_tickers_to_scan(full_list: List[str], existing_summary: List[Dict], mode: str) -> List[str]:
    """Deterministic scan universe resolver for scan-all vs scan-new-only."""
    if mode != "new_only":
        return list(full_list)
    already_scanned = {str(s.get("ticker", "")).upper().strip() for s in existing_summary}
    return [t for t in full_list if t not in already_scanned]

