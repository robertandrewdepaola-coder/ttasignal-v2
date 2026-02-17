import tempfile
import unittest
import types
import sys

if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.SimpleNamespace(DataFrame=object)

from journal_manager import ConditionalEntry, JournalManager


class AlertPersistenceTests(unittest.TestCase):
    def test_trigger_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            jm = JournalManager(data_dir=tmp)
            entry = ConditionalEntry(
                ticker="AAPL",
                condition_type="breakout_above",
                trigger_price=100.0,
                status="PENDING",
            )
            jm.add_conditional(entry)

            triggered = jm.check_conditionals({"AAPL": 101.0}, volume_ratios={})
            self.assertEqual(len(triggered), 1)
            self.assertEqual(triggered[0]["status"], "TRIGGERED")

            jm_reloaded = JournalManager(data_dir=tmp)
            pending = jm_reloaded.get_pending_conditionals()
            self.assertEqual(len(pending), 0)
            all_rows = [c for c in jm_reloaded.conditionals if c.get("ticker") == "AAPL"]
            self.assertTrue(all_rows)
            self.assertEqual(all_rows[0].get("status"), "TRIGGERED")


if __name__ == "__main__":
    unittest.main()
