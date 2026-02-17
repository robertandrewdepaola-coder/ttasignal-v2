"""
Pure earnings helpers for rollover/date selection logic.
"""

from datetime import date
from typing import Dict, List, Optional


def select_earnings_dates(candidates: List, today: date) -> Dict[str, Optional[date]]:
    """
    Select nearest future earnings date and most recent past date from candidates.
    Returns: {'next': date|None, 'recent_past': date|None, 'latest_any': date|None}
    """
    parsed: List[date] = []
    for c in candidates:
        try:
            if hasattr(c, 'date') and callable(getattr(c, 'date', None)):
                d = c.date()
            elif hasattr(c, 'to_pydatetime'):
                d = c.to_pydatetime().date()
            else:
                continue
            parsed.append(d)
        except Exception:
            continue

    parsed = sorted(set(parsed))
    next_dt = next((d for d in parsed if d >= today), None)
    recent_past = next((d for d in reversed(parsed) if (today - d).days <= 30 and d < today), None)
    latest_any = parsed[-1] if parsed else None
    return {'next': next_dt, 'recent_past': recent_past, 'latest_any': latest_any}

