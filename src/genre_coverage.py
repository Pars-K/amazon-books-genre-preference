"""Count genre coverage from the local Books metadata and reviews."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

from audit import GENRE_RULES, RATINGS, SCIENCE_BRANCH, genre_matches, rows


ADDED = ("Philosophy", "Science")


def write_json(path, result):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(result, indent=2), encoding="utf-8")
    partial.replace(path)


def build_mapping(path, output_dir):
    quality = Counter()
    matches_by_book = {}
    science_categories = Counter()
    for _, record in rows(path, quality):
        categories = record.get("categories")
        parent = record.get("parent_asin")
        if not isinstance(categories, list) or not isinstance(parent, str) or not parent.strip():
            continue
        matches = genre_matches(categories)
        if SCIENCE_BRANCH in categories:
            science_categories.update({c for c in categories if isinstance(c, str)})
        if matches:
            if parent in matches_by_book:
                matches_by_book[parent].update(matches)
            else:
                matches_by_book[parent] = matches

    counts = Counter()
    overlaps = Counter()
    before = Counter()
    ambiguous = 0
    output_path = output_dir / "book_genres.csv"
    partial = output_path.with_suffix(".tmp")
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("parent_asin", "genre"))
        for parent, matches in matches_by_book.items():
            original = matches.difference(ADDED)
            if len(original) == 1:
                before[next(iter(original))] += 1
            if len(matches) == 1:
                genre = next(iter(matches))
                writer.writerow((parent, genre))
                counts[genre] += 1
            else:
                ambiguous += 1
                overlaps.update(" + ".join(pair) for pair in combinations(sorted(matches), 2))
    partial.replace(output_path)
    result = {
        "quality": quality,
        "rules": {genre: sorted(values) for genre, values in GENRE_RULES.items()},
        "science_requires_branch": SCIENCE_BRANCH,
        "science_branch_category_counts": dict(science_categories.most_common(25)),
        "exclusive_book_ids_before_expansion": before,
        "exclusive_book_ids": counts,
        "ambiguous_book_ids": ambiguous,
        "pair_overlaps": overlaps,
    }
    write_json(output_dir / "metadata_summary.json", result)
    print(f"Metadata complete | {dict(counts)}", flush=True)


def review_coverage(path, output_dir):
    with (output_dir / "book_genres.csv").open(encoding="utf-8", newline="") as file:
        mapping = {row["parent_asin"]: row["genre"] for row in csv.DictReader(file)}
    quality = Counter()
    reviews = Counter()
    positive = Counter()
    with_text = Counter()
    reviewed_books = set()
    users = {genre: set() for genre in ADDED}
    distinct_positive = {genre: defaultdict(set) for genre in ADDED}
    for _, record in rows(path, quality):
        parent = record.get("parent_asin")
        genre = mapping.get(parent) if isinstance(parent, str) else None
        if genre is None:
            continue
        reviews[genre] += 1
        reviewed_books.add(parent)
        body = record.get("text")
        if isinstance(body, str) and body.strip():
            with_text[genre] += 1
        rating = record.get("rating")
        is_positive = type(rating) in (int, float) and rating in RATINGS and rating >= 4
        if is_positive:
            positive[genre] += 1
        if genre in users:
            user = record.get("user_id")
            if isinstance(user, str) and user.strip():
                users[genre].add(user)
                if is_positive:
                    distinct_positive[genre][user].add(parent)

    result = {
        "quality": quality,
        "reviews": reviews,
        "positive_reviews": positive,
        "reviews_with_nonblank_text": with_text,
        "reviewed_book_ids": Counter(mapping[parent] for parent in reviewed_books),
        "added_genre_unique_reviewers": {g: len(accounts) for g, accounts in users.items()},
        "added_genre_users_by_minimum_distinct_positive_books": {
            genre: {
                k: sum(len(books) >= k for books in accounts.values())
                for k in (1, 2, 3, 5)
            }
            for genre, accounts in distinct_positive.items()
        },
    }
    write_json(output_dir / "reviews_summary.json", result)
    print(f"Reviews complete | {dict(reviews)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/genre_coverage"))
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--metadata-only", action="store_true")
    stage.add_argument("--reviews-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.reviews_only:
        build_mapping(args.data_dir / "meta_Books.jsonl.gz", args.output_dir)
    if not args.metadata_only:
        review_coverage(args.data_dir / "Books.jsonl.gz", args.output_dir)


if __name__ == "__main__":
    main()
