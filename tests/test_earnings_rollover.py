import unittest
from datetime import datetime

from earnings_utils import select_earnings_dates


class EarningsRolloverTests(unittest.TestCase):
    def test_selects_future_and_recent_past(self):
        today = datetime(2026, 2, 17).date()
        candidates = [
            datetime(2026, 1, 30),
            datetime(2026, 2, 14),
            datetime(2026, 3, 1),
            datetime(2026, 5, 2),
        ]
        picks = select_earnings_dates(candidates, today)
        self.assertEqual(str(picks["next"]), "2026-03-01")
        self.assertEqual(str(picks["recent_past"]), "2026-02-14")
        self.assertEqual(str(picks["latest_any"]), "2026-05-02")

    def test_handles_no_future_dates(self):
        today = datetime(2026, 2, 17).date()
        candidates = [datetime(2025, 10, 1), datetime(2026, 1, 20)]
        picks = select_earnings_dates(candidates, today)
        self.assertIsNone(picks["next"])
        self.assertEqual(str(picks["latest_any"]), "2026-01-20")


if __name__ == "__main__":
    unittest.main()
