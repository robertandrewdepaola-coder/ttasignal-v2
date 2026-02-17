import tempfile
import unittest
from pathlib import Path

from watchlist_manager import MASTER_ID, WatchlistManager


class WatchlistManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.json_path = str(Path(self.tmp.name) / "watchlists.json")
        self.mgr = WatchlistManager(self.json_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_remove_bulk(self):
        ok, _ = self.mgr.add_manual_ticker(MASTER_ID, "AAPL")
        self.assertTrue(ok)
        ok, _ = self.mgr.add_manual_ticker(MASTER_ID, "BRK.B")
        self.assertTrue(ok)

        ok, _ = self.mgr.add_manual_ticker(MASTER_ID, "ABC.Z")
        self.assertFalse(ok)

        ok, msg, cleaned = self.mgr.update_tickers(MASTER_ID, ["TSLA", "AAPL", "TSLA", "NVDA"])
        self.assertTrue(ok)
        self.assertIn("Updated with", msg)
        self.assertEqual(cleaned, ["AAPL", "NVDA", "TSLA"])

        ok, _ = self.mgr.remove_manual_ticker(MASTER_ID, "NVDA")
        self.assertTrue(ok)
        wl = self.mgr.get_watchlist(MASTER_ID)
        self.assertEqual(sorted(wl["tickers"]), ["AAPL", "TSLA"])


if __name__ == "__main__":
    unittest.main()

