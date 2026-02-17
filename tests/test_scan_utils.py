import unittest

from scan_utils import resolve_tickers_to_scan


class ScanUtilsTests(unittest.TestCase):
    def test_scan_new_only_filters_previously_scanned(self):
        full = ["AAPL", "MSFT", "NVDA", "BRK.B"]
        existing = [{"ticker": "AAPL"}, {"ticker": "NVDA"}]
        result = resolve_tickers_to_scan(full, existing, mode="new_only")
        self.assertEqual(result, ["MSFT", "BRK.B"])

    def test_scan_all_keeps_full_universe(self):
        full = ["AAPL", "MSFT"]
        existing = [{"ticker": "AAPL"}]
        result = resolve_tickers_to_scan(full, existing, mode="all")
        self.assertEqual(result, full)


if __name__ == "__main__":
    unittest.main()

