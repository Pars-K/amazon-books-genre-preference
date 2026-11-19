"""Exercise snapshot selection, deduplication and user-isolated partitions."""

import csv
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeze_analysis_data import assign_split, export_snapshot, scan_reviews, select_review
from hashlib import blake2b


class FreezeDataTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.rules = json.loads((root / "reports/phase2/data_v1_rules.json").read_text(encoding="utf-8"))

    def test_preserves_style_and_does_not_impute_metadata(self):
        record = {"parent_asin": "book", "text": "Hello! " * 20,
                  "timestamp": -1, "rating": 7, "helpful_vote": -1}
        selected = select_review(record, 10, {}, self.rules)
        self.assertEqual(selected[2]["text"], record["text"])
        self.assertEqual(selected[2]["tokens"], 20)
        self.assertIsNone(selected[2]["timestamp"])
        self.assertIsNone(selected[2]["rating"])
        self.assertIsNone(selected[2]["book_genre"])
        self.assertIsNone(select_review({"parent_asin": "book", "text": "short"}, 1, {}, self.rules))

    def test_scan_export_deduplicates_and_rechecks_eligible_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            users = {user: blake2b(user.encode(), digest_size=16).hexdigest() for user in ("keep", "drop")}
            memberships = {}
            for user, key in users.items():
                memberships[key] = [{"genre": "Fantasy", "distinct_books": "3",
                                     "share_all_books": "0.5", "share_mapped_books": "1",
                                     "mapping_coverage": "0.5", "dated_books": "3",
                                     "active_quarters": "2", "span_days": "180"}]
            # A second membership must not duplicate any review or change the user's split.
            memberships[users["keep"]].append({**memberships[users["keep"]][0], "genre": "Mystery"})
            records = []
            for user in users:
                for i in range(3):
                    records.append({"user_id": user, "parent_asin": str(i),
                                    "text": "WORD! " * (400 if user == "keep" else 300)})
            records.append(records[0])
            records.append({**records[0], "text": "short"})
            source = output / "raw.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            rules = {**self.rules, "expected_source_rows": len(records)}
            connection, state = scan_reviews(source, memberships, {"0": "Fantasy"}, rules,
                                             output / "staging.sqlite", output / "state.json", {})
            summary = export_snapshot(connection, memberships, rules, output)
            connection.close()
            self.assertEqual(summary["retained_users"], 1)
            self.assertEqual(summary["removed_after_short_text_filter"], 1)
            self.assertEqual(summary["overlapping_memberships"], 1)
            self.assertEqual(summary["genres"]["Fantasy"]["target_genre_reviews"], 1)
            with gzip.open(output / "reviews.jsonl.gz", "rt", encoding="utf-8") as handle:
                exported = [json.loads(line) for line in handle]
            self.assertEqual(len(exported), 3)
            self.assertEqual({r["tokens"] for r in exported}, {400})
            split = assign_split(users["keep"], rules)
            self.assertEqual({r["split"] for r in exported}, {split})
            with (output / "memberships.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual({r["split"] for r in csv.DictReader(handle)}, {split})
            # A complete scan can be reused and still exports exactly one record per book.
            connection, cached = scan_reviews(source, memberships, {}, rules,
                                              output / "staging.sqlite", output / "state.json", {})
            self.assertEqual(cached, state)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 6)
            connection.close()


if __name__ == "__main__":
    unittest.main()
