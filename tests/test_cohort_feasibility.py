"""Check repeated reviews, share denominators and complete cohort tables."""

import csv
import gzip
from hashlib import blake2b
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cohort_feasibility import aggregate, publish, scan_text
from user_genre_profiles import write_csv


class CohortFeasibilityTests(unittest.TestCase):
    def test_text_deduplication_denominators_and_empty_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = blake2b(b"reader", digest_size=16).hexdigest()
            profiles = root / "profiles.csv"
            write_csv(profiles, [{
                "user_key": key, "genre": "Philosophy", "distinct_books": 3,
                "share_mapped_books": 1, "share_all_books": .375,
                "mapping_coverage": .375, "active_quarters": 3,
                "span_days": 200, "dated_books": 3,
            }])
            mapping = root / "mapping.csv"
            write_csv(mapping, [{"parent_asin": "p1", "genre": "Philosophy"}])
            reviews = root / "reviews.gz"
            records = [
                ("reader", "p1", 600), ("reader", "p1", 600),
                ("reader", "p1", 500), ("reader", "p2", 400),
                ("reader", "p3", 400), ("reader", "p4", 400),
                ("other", "p1", 1000),
            ]
            with gzip.open(reviews, "wt", encoding="utf-8") as handle:
                for user, parent, n in records:
                    handle.write(json.dumps({"user_id": user, "parent_asin": parent,
                                             "text": "word " * n}) + "\n")
            output = root / "text"
            output.mkdir()
            text_path, state = scan_text(reviews, profiles, mapping, output, len(records))
            with text_path.open(encoding="utf-8", newline="") as handle:
                rows = {row["genre"]: row for row in csv.DictReader(handle)}
            self.assertEqual(int(rows["ALL"]["tokens"]), 1800)
            self.assertEqual(int(rows["ALL"]["texts_at_least_20_tokens"]), 4)
            self.assertEqual(int(rows["Philosophy"]["tokens"]), 600)
            self.assertEqual(state["candidate_users"], 1)
            stages, grid = aggregate(profiles, text_path)
            common = ("Philosophy", "mapped_books", 3, .5, 0, 2, 180)
            self.assertEqual(grid[(*common, "outside_target_genre", 1000)], 1)
            self.assertEqual(grid[(*common, "all_history", 2000)], 0)
            self.assertEqual(grid[("Philosophy", "all_books", 3, .5, 0, 2, 180, "none", 0)], 0)
            self.assertEqual(stages["Philosophy"]["plus_coverage_50"], 0)
            folder = root / "reports"
            publish(stages, grid, state, folder)
            with (folder / "cohort_stages.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 7)
            with (folder / "cohort_sensitivity.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9408)
            self.assertTrue(any(row["users"] == "0" for row in rows))
            # A completed cache must produce the same measurements without rescanning.
            cached_path, cached_state = scan_text(reviews, profiles, mapping, output, len(records))
            self.assertEqual((cached_path, cached_state), (text_path, state))


if __name__ == "__main__":
    unittest.main()
