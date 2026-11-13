import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import track_hf_downloads


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous_directory = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous_directory)
        Path("README.md").write_text(
            "Before\n<!-- download-stats:start -->\nOld stats\n"
            "<!-- download-stats:end -->\nAfter\n",
            encoding="utf-8",
        )

    @patch("track_hf_downloads.requests.get")
    def test_updates_readme_and_appends_tracking_data(self, get):
        get.return_value.json.return_value = {"downloads": 31492, "downloadsAllTime": 55363}
        Path("hf_download_tracking.csv").write_text(
            "time,downloads_last_month\n2026-01-01 00:00:00,42\n", encoding="utf-8"
        )

        track_hf_downloads.main()

        get.assert_called_once_with(
            "https://huggingface.co/api/datasets/anglixue/TenK10K_multiome",
            params=[("expand[]", "downloads"), ("expand[]", "downloadsAllTime")],
            timeout=30,
        )
        get.return_value.raise_for_status.assert_called_once()
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("Downloads in the last 30 days: **31,492**", readme)
        self.assertIn("Total downloads: **55,363**", readme)
        self.assertIn(" UTC.", readme)
        self.assertTrue(readme.startswith("Before\n"))
        self.assertTrue(readme.endswith("\nAfter\n"))
        rows = Path("hf_download_tracking.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], "2026-01-01 00:00:00,42")
        self.assertTrue(rows[2].endswith(",31492"))

    @patch("track_hf_downloads.requests.get")
    def test_invalid_api_response_does_not_update_files(self, get):
        get.return_value.json.return_value = {"downloads": 31492}
        original_readme = Path("README.md").read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "valid download counts"):
            track_hf_downloads.main()

        self.assertEqual(Path("README.md").read_text(encoding="utf-8"), original_readme)
        self.assertFalse(Path("hf_download_tracking.csv").exists())

    @patch("track_hf_downloads.requests.get")
    def test_missing_markers_do_not_update_tracking_data(self, get):
        get.return_value.json.return_value = {"downloads": 31492, "downloadsAllTime": 55363}
        Path("README.md").write_text("No markers", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "markers"):
            track_hf_downloads.main()

        self.assertFalse(Path("hf_download_tracking.csv").exists())


if __name__ == "__main__":
    unittest.main()
