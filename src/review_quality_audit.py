"""Audit Books metadata, review fields, and duplicate review records."""

import argparse
from collections import Counter, defaultdict
from hashlib import blake2b
import json
from pathlib import Path
import random
from zlib import crc32

from audit import EARLIEST, LATEST, NO_METADATA, RATINGS, audit_catalog, field_states, rows
from genre_coverage import write_json


FIELD_TYPES = {
    "user_id": (str,),
    "asin": (str,),
    "parent_asin": (str,),
    "rating": (int, float),
    "text": (str,),
    "images": (list,),
    "timestamp": (int,),
    "helpful_vote": (int,),
    "verified_purchase": (bool,),
}
USER_BUCKETS = 64


def audit_review_fields(record):
    states = field_states(record, FIELD_TYPES)
    if states["rating"] == "available" and record["rating"] not in RATINGS:
        states["rating"] = "out_of_range"
    if states["timestamp"] == "available" and not EARLIEST <= record["timestamp"] < LATEST:
        states["timestamp"] = "out_of_range"
    if states["helpful_vote"] == "available" and record["helpful_vote"] < 0:
        states["helpful_vote"] = "negative"
    return states


def audit_reviews(path, catalog, rate, seed):
    quality = Counter()
    sample = defaultdict(Counter)
    duplicate_counts = Counter()
    seen_users = set()
    seen_records = set()
    seen_pairs = set()
    repeated_pairs = set()
    min_stamp = max_stamp = None
    rng = random.Random(seed)
    for _, record in rows(path, quality):
        parent = record.get("parent_asin")
        if rng.random() < rate:
            item = catalog.get(parent) if isinstance(parent, str) else None
            group, price, product_images = item or (NO_METADATA, "unlinked", "unlinked")
            counts = sample[group]
            counts["rows"] += 1
            counts[f"price:{price}"] += 1
            counts[f"product_images:{product_images}"] += 1
            states = audit_review_fields(record)
            for key, value in states.items():
                counts[f"{key}:{value}"] += 1
            if states["timestamp"] == "available":
                stamp = record["timestamp"]
                min_stamp = stamp if min_stamp is None else min(min_stamp, stamp)
                max_stamp = stamp if max_stamp is None else max(max_stamp, stamp)

        user = record.get("user_id")
        if not isinstance(user, str) or not user.strip():
            continue
        if crc32(user.encode("utf-8", errors="surrogatepass")) % USER_BUCKETS != seed % USER_BUCKETS:
            continue
        duplicate_counts["sampled_review_rows"] += 1
        seen_users.add(user)
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=True)
        digest = blake2b(canonical.encode("ascii"), digest_size=16).digest()
        if digest in seen_records:
            duplicate_counts["extra_identical_rows"] += 1
            continue
        seen_records.add(digest)
        if isinstance(parent, str) and parent.strip():
            pair = (user, parent)
            if pair in seen_pairs:
                duplicate_counts["extra_distinct_rows_for_existing_user_book_pair"] += 1
                repeated_pairs.add(pair)
            else:
                seen_pairs.add(pair)
    duplicate_counts["sampled_users"] = len(seen_users)
    duplicate_counts["distinct_user_book_pairs"] = len(seen_pairs)
    duplicate_counts["user_book_pairs_with_multiple_distinct_rows"] = len(repeated_pairs)
    return {
        "quality": quality,
        "row_sample_probability": rate,
        "seed": seed,
        "row_sample_by_group": sample,
        "valid_timestamp_range_in_sample_ms": [min_stamp, max_stamp],
        "user_sample_buckets": USER_BUCKETS,
        "duplicate_audit": duplicate_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/genre_coverage"))
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.rate <= 1:
        parser.error("--rate must be greater than zero and at most one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog, metadata = audit_catalog(args.data_dir / "meta_Books.jsonl.gz")
    write_json(args.output_dir / "catalog_quality.json", metadata)
    print(f"Catalog quality complete | {len(catalog):,} usable IDs", flush=True)
    reviews = audit_reviews(args.data_dir / "Books.jsonl.gz", catalog, args.rate, args.seed)
    write_json(args.output_dir / "review_quality.json", reviews)
    print("Review quality complete", flush=True)


if __name__ == "__main__":
    main()
