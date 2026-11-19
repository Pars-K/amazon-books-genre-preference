"""Describe genre engagement and satisfaction without assigning preference labels.

Run from the repository root. Hash partitions bring complete user histories
together even when a user's reviews occur in separate parts of the input.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b, sha256
import gzip
import json
import os
from pathlib import Path
import time

from audit import EARLIEST, LATEST, GENRE_RULES


GENRES = tuple(GENRE_RULES)
COUNT_BINS = ("1", "2", "3-4", "5-9", "10+")
USER_FIELDS = (
    "user_key", "review_rows", "distinct_books", "mapped_books", "mapping_coverage",
    "mapped_genres", "rated_books", "mean_rating", "books_with_nonblank_text",
    "top_mapped_genre_share", "top_share_tie_count",
)
GENRE_FIELDS = (
    "user_key", "genre", "distinct_books", "share_mapped_books", "share_all_books",
    "mapping_coverage", "all_user_books", "rated_books", "mean_rating",
    "mean_rating_minus_user_mean", "mean_rating_minus_other_books",
    "positive_rating_share", "low_rating_share", "rating_1", "rating_2", "rating_3",
    "rating_4", "rating_5", "dated_books", "first_book_review", "last_book_review",
    "span_days", "active_years", "active_quarters", "books_with_nonblank_text",
)


@dataclass
class Book:
    first_stamp: int | None = None
    rating_stamp: int = -1
    rating: int | None = None
    nonblank_text: bool = False

    def update(self, record):
        stamp = record.get("timestamp")
        valid_stamp = type(stamp) is int and EARLIEST <= stamp < LATEST
        rating = record.get("rating")
        valid_rating = type(rating) in (int, float) and rating in (1, 2, 3, 4, 5)
        text = record.get("text")
        self.observe(stamp if valid_stamp else None, int(rating) if valid_rating else None,
                     isinstance(text, str) and bool(text.strip()))

    def observe(self, stamp, rating, nonblank_text):
        if stamp is not None:
            self.first_stamp = stamp if self.first_stamp is None else min(self.first_stamp, stamp)
        if rating is not None:
            event_stamp = stamp if stamp is not None else -1
            if (self.rating is None or event_stamp > self.rating_stamp
                    or (event_stamp == self.rating_stamp and rating < self.rating)):
                self.rating_stamp, self.rating = event_stamp, int(rating)
        self.nonblank_text |= nonblank_text


def mean(values):
    count = sum(values.values())
    return sum(rating * n for rating, n in values.items()) / count if count else None


def count_bin(n):
    return "1" if n == 1 else "2" if n == 2 else "3-4" if n < 5 else "5-9" if n < 10 else "10+"


def coverage_bin(value):
    return "<25%" if value < .25 else "25-49%" if value < .5 else "50-74%" if value < .75 else "75-100%"


def profile(user, books, mapping, review_rows, user_key=None):
    key = user_key or blake2b(user.encode("utf-8", errors="surrogatepass"), digest_size=16).hexdigest()
    ratings = Counter(book.rating for book in books.values() if book.rating is not None)
    overall_mean = mean(ratings)
    groups = defaultdict(list)
    for parent, book in books.items():
        genre = mapping.get(parent)
        if genre is not None:
            groups[genre].append(book)
    n_all = len(books)
    n_mapped = sum(map(len, groups.values()))
    coverage = n_mapped / n_all if n_all else 0
    max_count = max(map(len, groups.values()), default=0)
    user_row = dict(zip(USER_FIELDS, (
        key, review_rows, n_all, n_mapped, coverage, len(groups), sum(ratings.values()),
        overall_mean, sum(book.nonblank_text for book in books.values()),
        max_count / n_mapped if n_mapped else None,
        sum(len(group) == max_count for group in groups.values()),
    )))
    genre_rows = []
    for genre in GENRES:
        group = groups.get(genre)
        if not group:
            continue
        n = len(group)
        hist = Counter(book.rating for book in group if book.rating is not None)
        n_rated = sum(hist.values())
        average = mean(hist)
        other = ratings - hist
        other_mean = mean(other)
        dates = [datetime.fromtimestamp(book.first_stamp / 1000, timezone.utc)
                 for book in group if book.first_stamp is not None]
        first, last = (min(dates), max(dates)) if dates else (None, None)
        genre_rows.append(dict(zip(GENRE_FIELDS, (
            key, genre, n, n / n_mapped, n / n_all, coverage, n_all, n_rated, average,
            average - overall_mean if average is not None else None,
            average - other_mean if average is not None and other_mean is not None else None,
            (hist[4] + hist[5]) / n_rated if n_rated else None,
            (hist[1] + hist[2]) / n_rated if n_rated else None,
            *(hist[r] for r in range(1, 6)), len(dates),
            first.date().isoformat() if first else None,
            last.date().isoformat() if last else None,
            (last - first).total_seconds() / 86400 if dates else None,
            len({date.year for date in dates}),
            len({(date.year, (date.month - 1) // 3) for date in dates}),
            sum(book.nonblank_text for book in group),
        ))))
    return user_row, genre_rows


def add_relation(counter, row):
    counter["user_genre_pairs"] += 1
    counter["distinct_books"] += row["distinct_books"]
    counter["rated_books"] += row["rated_books"]
    counter["rating_sum"] += sum(r * row[f"rating_{r}"] for r in range(1, 6))
    counter["positive_books"] += row["rating_4"] + row["rating_5"]
    if row["mean_rating"] is not None:
        counter["users_with_ratings"] += 1
        counter["sum_user_mean_rating"] += row["mean_rating"]
    for field in ("mean_rating_minus_user_mean", "mean_rating_minus_other_books"):
        if row[field] is not None:
            counter[f"n_{field}"] += 1
            counter[f"sum_{field}"] += row[field]


def relation_result(counter):
    result = {key: counter[key] for key in (
        "user_genre_pairs", "distinct_books", "rated_books", "rating_sum",
        "positive_books", "users_with_ratings", "sum_user_mean_rating")}
    result.update(counter)
    for output, numerator, denominator in (
        ("book_weighted_mean_rating", "rating_sum", "rated_books"),
        ("user_weighted_mean_rating", "sum_user_mean_rating", "users_with_ratings"),
        ("positive_rating_share", "positive_books", "rated_books"),
        *((field, f"sum_{field}", f"n_{field}") for field in
          ("mean_rating_minus_user_mean", "mean_rating_minus_other_books")),
    ):
        result[output] = counter[numerator] / counter[denominator] if counter[denominator] else None
    return result


class Aggregates:
    def __init__(self):
        self.users = Counter()
        self.activity = Counter()
        self.mapped_activity = Counter()
        self.coverage = Counter()
        self.genres = defaultdict(Counter)
        self.relations = defaultdict(Counter)
        self.grid = Counter()

    def add(self, user, rows):
        self.users["users"] += 1
        self.users["distinct_user_book_pairs"] += user["distinct_books"]
        self.users["valid_user_rows"] += user["review_rows"]
        self.activity[user["distinct_books"]] += 1
        if not rows:
            return
        self.users["users_with_mapped_books"] += 1
        self.users["mapped_distinct_user_book_pairs"] += user["mapped_books"]
        self.users["users_with_multiple_mapped_genres"] += user["mapped_genres"] >= 2
        self.mapped_activity[user["mapped_books"]] += 1
        self.coverage[coverage_bin(user["mapping_coverage"])] += 1
        for row in rows:
            genre, n = row["genre"], row["distinct_books"]
            stats = self.genres[genre]
            add_relation(stats, row)
            for k in (1, 2, 3, 5, 10):
                stats[f"users_at_least_{k}_books"] += n >= k
            stats["single_book_positive_rating"] += n == 1 and row["positive_rating_share"] == 1
            repeated = n >= 3
            enough_ratings = row["rated_books"] >= 3
            stats["repeated_with_three_ratings"] += repeated and enough_ratings
            if repeated:
                stats["repeated_multi_quarter"] += row["active_quarters"] >= 2
                stats["repeated_span_180_days"] += row["dated_books"] >= 3 and row["span_days"] >= 180
                stats["repeated_span_365_days"] += row["dated_books"] >= 3 and row["span_days"] >= 365
                if enough_ratings:
                    stats["repeated_mean_rating_at_most_3"] += row["mean_rating"] <= 3
                    stats["repeated_mean_rating_below_4"] += row["mean_rating"] < 4
                    stats["repeated_mean_rating_at_least_4"] += row["mean_rating"] >= 4
            add_relation(self.relations[(genre, count_bin(n), "all")], row)
            add_relation(self.relations[(genre, count_bin(n), coverage_bin(row["mapping_coverage"]))], row)
            for k in (2, 3, 5, 10):
                if n < k:
                    continue
                for share in (.25, .5, .75):
                    if row["share_mapped_books"] < share:
                        continue
                    for quarters in (1, 2, 4):
                        if row["active_quarters"] >= quarters:
                            self.grid[(genre, k, share, quarters, "all")] += 1
                            if row["mapping_coverage"] >= .5:
                                self.grid[(genre, k, share, quarters, "coverage>=50%")] += 1

    def result(self):
        users = {key: self.users[key] for key in (
            "users", "distinct_user_book_pairs", "valid_user_rows", "users_with_mapped_books",
            "mapped_distinct_user_book_pairs", "users_with_multiple_mapped_genres")}
        users["extra_rows_beyond_distinct_books"] = (
            users["valid_user_rows"] - users["distinct_user_book_pairs"])
        mapped_total = users.get("mapped_distinct_user_book_pairs", 0)
        return {
            "users": users,
            "all_user_distinct_book_histogram": dict(sorted(self.activity.items())),
            "mapped_user_mapped_book_histogram": dict(sorted(self.mapped_activity.items())),
            "mapping_coverage_user_bins": dict(self.coverage),
            "genres": {g: {**relation_result(self.genres[g]),
                           "share_of_all_mapped_user_book_pairs":
                           self.genres[g]["distinct_books"] / mapped_total if mapped_total else None}
                       for g in GENRES},
            "rating_by_count": [
                {"genre": genre, "book_count_bin": bucket, "mapping_coverage_bin": coverage,
                 **relation_result(value)}
                for (genre, bucket, coverage), value in sorted(self.relations.items())
            ],
            "diagnostic_threshold_grid": [
                {"genre": genre, "minimum_books": k, "minimum_share_mapped": share,
                 "minimum_active_quarters": quarters, "coverage_filter": coverage,
                 "users": self.grid[(genre, k, share, quarters, coverage)]}
                for genre in GENRES for k in (2, 3, 5, 10) for share in (.25, .5, .75)
                for quarters in (1, 2, 4) for coverage in ("all", "coverage>=50%")
            ],
        }


def load_mapping(path):
    with path.open(encoding="utf-8", newline="") as handle:
        mapping = {}
        for row in csv.DictReader(handle):
            if row["genre"] not in GENRES or row["parent_asin"] in mapping:
                raise ValueError("Mapping must contain one known genre per unique parent_asin")
            mapping[row["parent_asin"]] = row["genre"]
    return mapping


def write_csv(path, records, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(reviews, mapping_path, output_dir, report_dir, expected_rows=None, partitions=128, resume=False):
    if not 1 <= partitions <= 256:
        raise ValueError("partitions must be between 1 and 256")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in ("user_profiles.csv", "user_genre_profiles.csv")]
    partition_dir = output_dir / "partitions"
    checkpoint_path = output_dir / "checkpoint.json"
    if any(path.exists() for path in paths) or (not resume and (partition_dir.exists() or any(
            path.with_suffix(".csv.partial").exists() for path in paths))):
        raise FileExistsError("Output or partial files exist; choose a new --output-dir")
    mapping = load_mapping(mapping_path)
    started = time.monotonic()
    aggregates, quality = Aggregates(), Counter()
    fingerprint = {"review_bytes": reviews.stat().st_size, "review_mtime_ns": reviews.stat().st_mtime_ns,
                   "mapping_sha256": sha256(mapping_path.read_bytes()).hexdigest(), "partitions": partitions}
    checkpoint = None
    if resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["input"] != fingerprint:
            raise ValueError("Checkpoint inputs or partition count do not match this run")
        quality.update(checkpoint["quality"])
    else:
        partition_dir.mkdir()
    partition_paths = [partition_dir / f"part-{i:03}.csv.partial" for i in range(partitions)]
    handles = []

    def save_checkpoint(offset, complete=False):
        sizes = []
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            sizes.append(handle.buffer.tell())
        payload = {"input": fingerprint, "quality": dict(quality), "uncompressed_offset": offset,
                   "partition_sizes": sizes, "partition_complete": complete}
        partial = checkpoint_path.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        partial.replace(checkpoint_path)

    if checkpoint is not None:
        for path, size in zip(partition_paths, checkpoint["partition_sizes"]):
            if path.stat().st_size < size:
                raise ValueError("Checkpoint partition is incomplete")
            with path.open("r+b") as handle:
                handle.truncate(size)
    try:
        complete = checkpoint is not None and checkpoint["partition_complete"]
        handles = [path.open("a" if resume else "w", encoding="utf-8", newline="") for path in partition_paths]
        writers = [csv.writer(handle) for handle in handles]
        if not complete:
          with gzip.open(reviews, "rb") as handle:
            if checkpoint is not None:
                print(f"Resuming after {quality['raw_rows']:,} source rows", flush=True)
                handle.seek(checkpoint["uncompressed_offset"])
            else:
                save_checkpoint(0)
            for index, line in enumerate(handle, quality["raw_rows"] + 1):
                quality["raw_rows"] += 1
                if index % 1_000_000 == 0:
                    print(f"Partitioning | {index:,} rows | {time.monotonic() - started:.0f}s", flush=True)
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    quality["invalid_json"] += 1
                    continue
                if not isinstance(record, dict):
                    quality["non_object_rows"] += 1
                    continue
                user = record.get("user_id")
                if not isinstance(user, str) or not user.strip():
                    quality["invalid_user_id"] += 1
                    continue
                key = blake2b(user.encode("utf-8", errors="surrogatepass"), digest_size=16).digest()
                parent = record.get("parent_asin")
                if not isinstance(parent, str) or not parent.strip():
                    quality["invalid_parent_asin"] += 1
                    writers[key[0] % partitions].writerow((key.hex(), "", "", "", 0))
                    continue
                rating, stamp = record.get("rating"), record.get("timestamp")
                valid_rating = type(rating) in (int, float) and rating in (1, 2, 3, 4, 5)
                valid_stamp = type(stamp) is int and EARLIEST <= stamp < LATEST
                quality["invalid_rating_rows"] += not valid_rating
                quality["invalid_timestamp_rows"] += not valid_stamp
                body = record.get("text")
                writers[key[0] % partitions].writerow((
                    key.hex(), parent, stamp if valid_stamp else "",
                    int(rating) if valid_rating else "",
                    int(isinstance(body, str) and bool(body.strip())),
                ))
                if index % 1_000_000 == 0:
                    save_checkpoint(handle.tell())
            save_checkpoint(handle.tell(), complete=True)
    finally:
        for handle in handles:
            handle.close()
    if expected_rows is not None and quality["raw_rows"] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; observed {quality['raw_rows']}")
    print(f"Partitioning complete | {quality['raw_rows']:,} rows", flush=True)
    with paths[0].with_suffix(".csv.partial").open("w", encoding="utf-8", newline="") as users_file, \
            paths[1].with_suffix(".csv.partial").open("w", encoding="utf-8", newline="") as genres_file:
        user_writer = csv.DictWriter(users_file, fieldnames=USER_FIELDS)
        genre_writer = csv.DictWriter(genres_file, fieldnames=GENRE_FIELDS)
        user_writer.writeheader()
        genre_writer.writeheader()
        projected_rows = 0
        for number, path in enumerate(partition_paths, 1):
            users = {}
            with path.open(encoding="utf-8", newline="") as handle:
                for key, parent, stamp, rating, text in csv.reader(handle):
                    projected_rows += 1
                    if key not in users:
                        users[key] = [0, {}]
                    item = users[key]
                    item[0] += 1
                    if not parent:
                        continue
                    books = item[1]
                    if parent not in books:
                        books[parent] = Book()
                    books[parent].observe(int(stamp) if stamp else None,
                                          int(rating) if rating else None, text == "1")
            for key, (n_rows, books) in users.items():
                user_row, genre_rows = profile(key, books, mapping, n_rows, user_key=key)
                aggregates.add(user_row, genre_rows)
                if genre_rows:
                    user_writer.writerow(user_row)
                    genre_writer.writerows(genre_rows)
            users.clear()
            if number % 8 == 0 or number == partitions:
                print(f"Profiles | {number}/{partitions} partitions | "
                      f"{aggregates.users['users']:,} users | {time.monotonic() - started:.0f}s", flush=True)
    if projected_rows != aggregates.users["valid_user_rows"]:
        raise AssertionError("Projected review-row count disagrees with aggregation")
    if projected_rows != quality["raw_rows"] - quality["invalid_json"] - quality["non_object_rows"] - quality["invalid_user_id"]:
        raise AssertionError("Projection does not account for every valid-user source row")
    result = aggregates.result()
    result["quality"] = dict(quality)
    result["run"] = {
        "review_file": reviews.name, "compressed_review_bytes": reviews.stat().st_size,
        "mapping_file": mapping_path.name, "mapping_sha256": sha256(mapping_path.read_bytes()).hexdigest(),
        "mapped_book_ids": len(mapping), "sampling": "none; all rows and users",
        "partitions": partitions, "user_key": "blake2b-128, pseudonymous; not anonymous",
        "history_grouping": "hash partitions merge all rows for each user, including noncontiguous rows",
        "elapsed_seconds_current_invocation": time.monotonic() - started, "resumed": resume,
        "timestamp_valid_range": "1996-01-01 <= timestamp < 2024-01-01, UTC",
        "label_status": "descriptive profiles only; no preference labels assigned",
        "scope": "entire observed archive; not a train/test split or future-label cohort",
    }
    for path in paths:
        path.with_suffix(".csv.partial").replace(path)
    summary = report_dir / "summary.json"
    partial = summary.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    partial.replace(summary)
    write_csv(report_dir / "rating_by_count.csv", result["rating_by_count"])
    write_csv(report_dir / "diagnostic_threshold_grid.csv", result["diagnostic_threshold_grid"])
    for path in partition_paths:
        if path.resolve().parent != partition_dir.resolve():
            raise ValueError("Unexpected temporary partition path")
        path.unlink()
    partition_dir.rmdir()
    checkpoint_path.unlink()
    print(f"Complete | {aggregates.users['users']:,} users | {summary}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/Books.jsonl.gz"))
    parser.add_argument("--mapping", type=Path, default=Path("data/genre_coverage/book_genres.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/user_genre_profiles"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/genre_preference"))
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--partitions", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.reviews, args.mapping, args.output_dir, args.report_dir, args.expected_rows, args.partitions, args.resume)


if __name__ == "__main__":
    main()
