import unittest
from unittest.mock import patch

import backup_health


class BackupHealthTests(unittest.TestCase):
    @patch("github_backup.status")
    def test_get_backup_health_status_normalizes_fields(self, mock_status):
        mock_status.return_value = {
            "enabled": True,
            "pending_count": 2,
            "last_success_epoch": 123.4,
            "branch": "data-backup",
            "last_error": "",
            "last_error_code": 0,
        }

        out = backup_health.get_backup_health_status()

        self.assertTrue(out["enabled"])
        self.assertEqual(out["pending_count"], 2)
        self.assertEqual(out["branch"], "data-backup")
        self.assertTrue(out["available"])

    @patch("github_backup.force_backup_all")
    @patch("github_backup.flush")
    def test_run_backup_now_fallbacks_to_force(self, mock_flush, mock_force):
        mock_flush.return_value = 0
        mock_force.return_value = 3

        pushed = backup_health.run_backup_now()

        self.assertEqual(pushed, 3)


if __name__ == "__main__":
    unittest.main()
