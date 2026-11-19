"""Check deduplication, denominators, time evidence, and complete user histories."""

import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from user_genre_profiles import Book, profile, run
from report_user_genres import analyze_shares, validate


def stamp(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1000)


def record(user, parent, rating, date):
    return {"user_id": user, "parent_asin": parent, "rating": rating,
            "timestamp": stamp(date), "text": "A review."}


class ProfileTests(unittest.TestCase):
    def test_earliest_book_event_latest_rating_and_order_independence(self):
        events = [record("u", "a", 5, "2020-01-01"), record("u", "a", 2, "2021-01-01"),
                  record("u", "a", 4, "2021-01-01")]
        for sequence in (events, events[::-1], events + events):
            book = Book()
            for item in sequence:
                book.update(item)
            self.assertEqual(book.first_stamp, stamp("2020-01-01"))
            self.assertEqual(book.rating, 2)
            self.assertTrue(book.nonblank_text)

    def test_unmapped_denominator_and_repeated_books(self):
        books = {}
        for item in [record("u", "a", 5, "2020-01-01"), record("u", "a", 2, "2021-01-01"),
                     record("u", "b", 3, "2020-07-01"), record("u", "c", 1, "2020-01-01")]:
            books.setdefault(item["parent_asin"], Book()).update(item)
        user, rows = profile("u", books, {"a": "Fantasy", "b": "Fantasy"}, 4)
        row = rows[0]
        self.assertEqual(user["distinct_books"], 3)
        self.assertEqual(user["mapped_books"], 2)
        self.assertAlmostEqual(user["mapping_coverage"], 2 / 3)
        self.assertEqual(row["distinct_books"], 2)
        self.assertEqual(row["share_mapped_books"], 1)
        self.assertAlmostEqual(row["share_all_books"], 2 / 3)
        self.assertEqual(row["active_quarters"], 2)
        self.assertEqual(row["active_years"], 1)
        self.assertEqual(row["mean_rating"], 2.5)
        self.assertEqual(row["mean_rating_minus_other_books"], 1.5)
        self.assertNotEqual(user["user_key"], "u")

    def test_invalid_rating_and_timestamp_do_not_create_evidence(self):
        book = Book()
        book.update({"rating": True, "timestamp": True, "text": None})
        user, rows = profile("u", {"a": book}, {"a": "Fantasy"}, 1)
        self.assertEqual(user["rated_books"], 0)
        self.assertIsNone(rows[0]["mean_rating"])
        self.assertEqual(rows[0]["active_quarters"], 0)
        self.assertIsNone(rows[0]["span_days"])

    def fixture(self, folder, records):
        reviews = folder / "reviews.gz"
        with gzip.open(reviews, "wt", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item) + "\n")
        mapping = folder / "mapping.csv"
        mapping.write_text("parent_asin,genre\na,Fantasy\nb,Science\n", encoding="utf-8")
        return reviews, mapping

    def test_end_to_end_full_histories_and_rating_inclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = [record("u", "a", 1, "2020-01-01"), record("u", "a", 1, "2020-01-01"),
                       record("u", "b", 5, "2020-02-01"), record("u", "z", 2, "2020-01-01"),
                       record("v", "z", 4, "2020-01-01")]
            reviews, mapping = self.fixture(folder, records)
            result = run(reviews, mapping, folder / "data", folder / "report", 5)
            self.assertEqual(result["users"]["users"], 2)
            self.assertEqual(result["users"]["users_with_mapped_books"], 1)
            self.assertEqual(result["users"]["distinct_user_book_pairs"], 4)
            self.assertEqual(result["users"]["extra_rows_beyond_distinct_books"], 1)
            self.assertEqual(result["genres"]["Fantasy"]["distinct_books"], 1)
            self.assertEqual(result["genres"]["Fantasy"]["book_weighted_mean_rating"], 1)
            self.assertEqual(result["genres"]["Fantasy"]["users_at_least_2_books"], 0)
            with (folder / "data" / "user_genre_profiles.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_noncontiguous_user_histories_are_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reviews, mapping = self.fixture(folder, [record("u", "a", 5, "2020-01-01"),
                                                      record("v", "a", 4, "2020-01-01"),
                                                      record("u", "a", 2, "2021-01-01"),
                                                      record("u", "b", 3, "2020-07-01")])
            result = run(reviews, mapping, folder / "data", folder / "report", partitions=2)
            self.assertEqual(result["users"]["users"], 2)
            self.assertEqual(result["users"]["distinct_user_book_pairs"], 3)
            self.assertEqual(result["genres"]["Fantasy"]["book_weighted_mean_rating"], 3)
            self.assertEqual(result["users"]["users_with_multiple_mapped_genres"], 1)

    def test_expected_row_count_mismatch_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reviews, mapping = self.fixture(folder, [record("u", "a", 4, "2020-01-01")])
            with self.assertRaisesRegex(ValueError, "Expected 2"):
                run(reviews, mapping, folder / "data", folder / "report", 2)
            self.assertFalse((folder / "data" / "user_profiles.csv").exists())

    def test_share_comparisons_preserve_denominators_and_coverage_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = [record("u", "a", 5, "2020-01-01"), record("u", "b", 3, "2020-02-01")]
            records += [record("u", f"unmapped-{i}", 2, "2020-01-01") for i in range(8)]
            reviews, mapping = self.fixture(folder, records)
            result = run(reviews, mapping, folder / "data", folder / "report", partitions=2)
            validate(result)
            rows = analyze_shares(folder / "data" / "user_genre_profiles.csv", result)
            fantasy = [r for r in rows if r["genre"] == "Fantasy" and r["book_count_bin"] == "1"]
            self.assertEqual(len(fantasy), 1)
            self.assertEqual(fantasy[0]["coverage_filter"], "all")
            self.assertEqual(fantasy[0]["share_mapped_bin"], "[50%,75%)")
            self.assertEqual(fantasy[0]["user_weighted_mean_rating"], 5)
            self.assertAlmostEqual(fantasy[0]["mean_rating_minus_other_books"], 5 - 19 / 9)

    def test_empty_input_has_consistent_zero_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reviews, mapping = self.fixture(folder, [])
            result = run(reviews, mapping, folder / "data", folder / "report", 0, partitions=2)
            validate(result)
            self.assertEqual(result["users"]["users"], 0)

    def test_resume_reuses_complete_partitions_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reviews, mapping = self.fixture(folder, [record("u", "a", 5, "2020-01-01"),
                                                      record("v", "a", 4, "2020-01-01"),
                                                      record("u", "b", 2, "2020-07-01")])
            with patch("user_genre_profiles.profile", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run(reviews, mapping, folder / "data", folder / "report", partitions=2)
            with patch("user_genre_profiles.gzip.open", side_effect=AssertionError("source reread")):
                result = run(reviews, mapping, folder / "data", folder / "report", partitions=2, resume=True)
            validate(result)
            self.assertEqual(result["users"]["users"], 2)
            self.assertEqual(result["users"]["distinct_user_book_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
