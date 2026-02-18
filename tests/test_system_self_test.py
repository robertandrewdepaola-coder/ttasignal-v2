import unittest

from system_self_test import run_system_self_test


class SystemSelfTestTests(unittest.TestCase):
    def test_pass_when_all_core_checks_are_ok(self):
        health = {
            "ai_ok": True,
            "price_feed_ok": True,
            "market_et_time": "2026-02-18 09:30:00 ET",
        }
        backup = {
            "enabled": True,
            "last_success_epoch": 1700000000.0,
            "last_error": "",
        }

        report = run_system_self_test(health, backup)

        self.assertTrue(report["overall_ok"])
        self.assertEqual(report["summary"], "PASS")
        self.assertEqual(len(report["failures"]), 0)

    def test_fail_when_backup_or_feed_is_unhealthy(self):
        health = {
            "ai_ok": True,
            "price_feed_ok": False,
            "market_et_time": "",
        }
        backup = {
            "enabled": True,
            "last_success_epoch": 0.0,
            "last_error": "401 Bad credentials",
        }

        report = run_system_self_test(health, backup)

        self.assertFalse(report["overall_ok"])
        self.assertIn("FAIL", report["summary"])
        names = {f["name"] for f in report["failures"]}
        self.assertIn("Price feed", names)
        self.assertIn("Market clock", names)
        self.assertIn("Backup persistence", names)


if __name__ == "__main__":
    unittest.main()
