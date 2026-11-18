"""Stream the two Amazon Books gzip files and publish aggregate Phase 1 evidence."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import blake2b
import json
from pathlib import Path
import random
import re
import unicodedata
from zlib import crc32


GENRE_RULES = {
    "Fantasy": {"Fantasy", "Dark Fantasy"},
    "Science Fiction": {"Science Fiction", "Hard Science Fiction"},
    "Romance": {"Romance", "LGBTQ+ Romance"},
    "Mystery": {"Mystery"},
    "Thriller": {"Thrillers & Suspense", "Psychological Thrillers", "Technothrillers"},
    "Philosophy": {"Philosophy"},
    "Science": {
        "Agricultural Sciences", "Astronomy & Space Science", "Biological Sciences",
        "Chemistry", "Earth Sciences", "Nature & Ecology", "Physics",
    },
}
CATEGORY_GENRE = {category: genre for genre, categories in GENRE_RULES.items()
                  if genre != "Science" for category in categories}
SCIENCE_BRANCH = "Science & Math"
UNMAPPED = "Other or unmapped"
AMBIGUOUS = "Ambiguous"
NO_METADATA = "No usable metadata"
META_FIELDS = ("parent_asin", "title", "categories", "price", "images")
RATINGS = (1, 2, 3, 4, 5)
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def date_ms(year):
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


EARLIEST = date_ms(1996)
STYLE_START = date_ms(2015)
CUTOFF = date_ms(2020)
LABEL_END = date_ms(2023)
LATEST = date_ms(2024)


def rows(path, quality):
    with gzip.open(path, "rb") as handle:
        for index, line in enumerate(handle, 1):
            if index % 1_000_000 == 0:
                print(f"{path.name} | {index:,} rows", flush=True)
            quality["rows"] += 1
            try:
                obj = json.loads(line)
            except (ValueError, UnicodeError):
                quality["invalid_json"] += 1
                continue
            if not isinstance(obj, dict):
                quality["non_object"] += 1
                continue
            yield index, obj


def state(value, kind):
    if value is None:
        return "missing"
    if type(value) not in kind:
        return "wrong_type"
    if isinstance(value, str):
        return "available" if value.strip() else "blank"
    if isinstance(value, list):
        return "available" if value else "blank"
    return "available"


def field_states(obj, kinds):
    return {key: state(obj.get(key), kind) for key, kind in kinds.items()}


def schema_update(schema, obj):
    schema["objects"] += 1
    for key, value in obj.items():
        schema["fields"][key][type(value).__name__] += 1


def schema_counter():
    return {"objects": 0, "fields": defaultdict(Counter)}


def genre_matches(categories):
    categories = {category for category in categories if isinstance(category, str)}
    matches = {CATEGORY_GENRE[category] for category in categories & CATEGORY_GENRE.keys()}
    if SCIENCE_BRANCH in categories and categories & GENRE_RULES["Science"]:
        matches.add("Science")
    return matches


def audit_catalog(path):
    quality = Counter()
    catalog = {}
    conflict = set()
    groups = defaultdict(Counter)
    schema = schema_counter()
    kinds = dict(zip(META_FIELDS, [(str,), (str,), (list,), (int, float), (list,)]))
    for index, obj in rows(path, quality):
        if index % 101 == 0:
            schema_update(schema, obj)
        parent = obj.get("parent_asin")
        states = field_states(obj, kinds)
        categories = obj.get("categories") if states["categories"] == "available" else []
        matches = genre_matches(categories)
        group = next(iter(matches)) if len(matches) == 1 else (
            AMBIGUOUS if matches else UNMAPPED
        )
        price = obj.get("price")
        if states["price"] == "available" and not (0 < price < float("inf")):
            states["price"] = "out_of_range"
        counts = groups[group]
        counts["rows"] += 1
        for key, value in states.items():
            counts[f"{key}:{value}"] += 1
        if states["parent_asin"] != "available":
            continue
        # Keep one record per parent ID. Exclude IDs with inconsistent audit fields.
        entry = (group, states["price"], states["images"])
        previous = catalog.setdefault(parent, entry)
        if previous is not entry:
            quality["repeated_parent_rows"] += 1
            if previous != entry:
                conflict.add(parent)
    for parent in conflict:
        del catalog[parent]
    quality["conflicting_parent_ids"] = len(conflict)
    quality["usable_parent_ids"] = len(catalog)
    return catalog, {"quality": quality, "groups": groups, "schema_sample": schema,
                     "unique_genre_ids": Counter(entry[0] for entry in catalog.values())}


def text_audit(text):
    words = WORD.findall(text)
    # A writing-system screen. Latin script is shared by many languages.
    non_ascii = any(ord(char) > 127 for char in text)
    non_latin = non_ascii and any(
        char.isalpha() and ord(char) > 127
        and "LATIN" not in unicodedata.name(char, "") for char in text
    )
    return len(words), non_ascii, non_latin


def cohort_row(cohort, user, obj, parent, genre, stamp, rating, text_ok):
    if not isinstance(parent, str) or not parent.strip() or not STYLE_START <= stamp < LABEL_END:
        return
    history, later = cohort[user]
    if stamp < CUTOFF:
        length = len(WORD.findall(obj["text"])) if text_ok else 0
        # One review per parent. Use the earliest qualifying writing sample.
        prev = history.get(parent, (CUTOFF, 0))
        if length >= 20 and (stamp, length) < prev:
            history[parent] = (stamp, length)
        elif parent not in history:
            history[parent] = (CUTOFF, 0)
    elif genre in GENRE_RULES and rating:
        prev = later.get(parent, (0, 0, genre))
        # Retain the latest rating; use the lower rating on a tied timestamp.
        if (stamp, -rating) > (prev[0], -prev[1]):
            later[parent] = (stamp, rating, genre)


def cohort_summary(cohort):
    stages = Counter()
    label_counts = {str(k): Counter() for k in (1, 2, 3)}
    label_sizes = {str(k): Counter() for k in (1, 2, 3)}
    stages["sampled_users"] = len(cohort)
    for history, later in cohort.values():
        lengths = [n for _, n in history.values() if n]
        adequate = len(lengths) >= 3 and sum(lengths) >= 150
        books = {book: row for book, row in later.items() if book not in history}
        if adequate:
            stages["adequate_style"] += 1
        if len(books) >= 3:
            stages["adequate_label_exposure"] += 1
        if not adequate or len(books) < 3:
            continue
        stages["eligible_users"] += 1
        positives = Counter(genre for _, rating, genre in books.values() if rating >= 4)
        for k in label_counts:
            labels = [genre for genre, n in positives.items() if n >= int(k)]
            label_counts[k].update(labels)
            label_sizes[k][len(labels)] += 1
    return {"stages": stages, "genre_counts": label_counts, "labels_per_user": label_sizes}


def audit_reviews(path, catalog, seed):
    quality = Counter()
    availability = defaultdict(Counter)
    rating_counts = Counter()
    month_counts = Counter()
    ratings_by_group = defaultdict(Counter)
    words_by_rating = defaultdict(Counter)
    text_flags = Counter()
    schema = schema_counter()
    user_frequency = Counter()
    cohort = defaultdict(lambda: ({}, {}))
    sample_seen = set()
    timestamps = [LATEST, EARLIEST]
    rng = random.Random(seed)
    kinds = {
        "user_id": (str,), "asin": (str,), "parent_asin": (str,), "rating": (int, float),
        "title": (str,), "text": (str,), "images": (list,), "timestamp": (int,),
        "helpful_vote": (int,), "verified_purchase": (bool,),
    }
    for _, obj in rows(path, quality):
        user = obj.get("user_id")
        parent = obj.get("parent_asin")
        meta = catalog.get(parent) if isinstance(parent, str) else None
        group, price, product_image = meta or (NO_METADATA, "unlinked", "unlinked")
        availability[group]["rows"] += 1
        availability[group][f"price:{price}"] += 1
        availability[group][f"product_image:{product_image}"] += 1
        states = field_states(obj, kinds)
        for key, value in states.items():
            availability[group][f"{key}:{value}"] += 1
        rating = obj.get("rating")
        valid_rating = states["rating"] == "available" and rating in RATINGS
        stamp = obj.get("timestamp")
        valid_stamp = states["timestamp"] == "available" and EARLIEST <= stamp < LATEST
        if valid_stamp:
            timestamps = [min(timestamps[0], stamp), max(timestamps[1], stamp)]
        elif states["timestamp"] == "available":
            quality["timestamp_out_of_range"] += 1
        if valid_rating:
            rating_counts[int(rating)] += 1
            ratings_by_group[group][int(rating)] += 1
        elif states["rating"] == "available":
            quality["rating_out_of_range"] += 1
        if states["helpful_vote"] == "available" and obj["helpful_vote"] < 0:
            quality["negative_helpful_votes"] += 1

        if rng.random() < 0.01:
            schema_update(schema, obj)
            if valid_stamp:
                month = datetime.fromtimestamp(stamp / 1000, tz=timezone.utc).strftime("%Y-%m")
                month_counts[month] += 1
            if states["text"] == "available":
                n, non_ascii, non_latin = text_audit(obj["text"])
                text_flags["nonblank"] += 1
                text_flags["non_ascii"] += non_ascii
                text_flags["contains_non_latin_letter"] += non_latin
                text_flags["contains_replacement_character"] += "\ufffd" in obj["text"]
                words_by_rating[int(rating) if valid_rating else "invalid"][n] += 1

        if states["user_id"] == "available" and crc32(user.encode("utf-8")) % 64 == seed % 64:
            user_frequency[user] += 1
            record_hash = blake2b(
                json.dumps(obj, sort_keys=True, ensure_ascii=True).encode(), digest_size=16
            ).digest()
            if record_hash in sample_seen:
                quality["exact_duplicates_in_user_sample"] += 1
                continue
            sample_seen.add(record_hash)
            if valid_stamp:
                cohort_row(cohort, user, obj, parent, group, stamp,
                           int(rating) if valid_rating else 0, states["text"] == "available")

    cohort_result = cohort_summary(cohort)
    cohort_result["stages"]["sampled_users"] = len(user_frequency)
    return {
        "quality": quality, "availability": availability, "rating_counts": rating_counts,
        "ratings_by_group": ratings_by_group, "schema_sample": schema,
        "sample_month_counts": month_counts, "sample_words_by_rating": words_by_rating,
        "sample_text_flags": text_flags, "valid_timestamp_range_ms": timestamps,
        "sample_user_activity": Counter(user_frequency.values()),
        "cohort": cohort_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/phase1_audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    review_file = args.data_dir / "Books.jsonl.gz"
    metadata_file = args.data_dir / "meta_Books.jsonl.gz"
    catalog, metadata = audit_catalog(metadata_file)
    print(f"Catalog complete | {len(catalog):,} usable parent IDs", flush=True)
    reviews = audit_reviews(review_file, catalog, args.seed)
    output = {
        "seed": args.seed,
        "compressed_bytes": {p.name: p.stat().st_size for p in (review_file, metadata_file)},
        "metadata": metadata,
        "reviews": reviews,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".tmp")
    partial.write_text(json.dumps(output, indent=2), encoding="utf-8")
    partial.replace(args.output)
    print(f"Audit complete | {args.output}", flush=True)


if __name__ == "__main__":
    main()
