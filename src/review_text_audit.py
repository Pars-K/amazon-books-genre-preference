"""Summarize review text in a reproducible sample of the Books file."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from math import ceil
from pathlib import Path
import random

from audit import RATINGS, text_audit
from genre_coverage import write_json


UNMAPPED = "Outside the seven mapped groups"


def summarize_length(counts):
    total = sum(counts.values())
    if not total:
        return {"texts": 0}
    quantiles = {}
    cumulative = 0
    for length, n in sorted(counts.items()):
        cumulative += n
        for percentile in (10, 25, 50, 75, 90, 95, 99):
            if percentile not in quantiles and cumulative >= ceil(total * percentile / 100):
                quantiles[percentile] = length
    return {
        "texts": total,
        "mean_tokens": sum(length * n for length, n in counts.items()) / total,
        "percentile_tokens": quantiles,
        "token_bins": {
            "0": counts[0],
            "1–4": sum(n for length, n in counts.items() if 1 <= length <= 4),
            "5–19": sum(n for length, n in counts.items() if 5 <= length <= 19),
            "20–49": sum(n for length, n in counts.items() if 20 <= length <= 49),
            "50–99": sum(n for length, n in counts.items() if 50 <= length <= 99),
            "100–499": sum(n for length, n in counts.items() if 100 <= length <= 499),
            "500+": sum(n for length, n in counts.items() if length >= 500),
        },
    }


def audit_text(path, mapping, rate, seed):
    rng = random.Random(seed)
    scan = Counter()
    quality = defaultdict(Counter)
    lengths = defaultdict(Counter)
    by_rating = defaultdict(Counter)
    traversed = 0
    with gzip.open(path, "rb") as file:
        for traversed, line in enumerate(file, 1):
            if traversed % 5_000_000 == 0:
                print(f"{traversed:,} review rows traversed", flush=True)
            if rng.random() >= rate:
                continue
            scan["sampled_rows"] += 1
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                scan["invalid_json_in_sample"] += 1
                continue
            if not isinstance(record, dict):
                scan["non_object_in_sample"] += 1
                continue
            parent = record.get("parent_asin")
            group = mapping.get(parent, UNMAPPED) if isinstance(parent, str) else UNMAPPED
            counts = quality[group]
            counts["sampled_reviews"] += 1
            body = record.get("text")
            if body is None:
                counts["missing_text"] += 1
            elif not isinstance(body, str):
                counts["wrong_text_type"] += 1
            elif not body.strip():
                counts["blank_text"] += 1
            else:
                length, non_ascii, non_latin = text_audit(body)
                lengths[group][length] += 1
                counts["non_ascii"] += non_ascii
                counts["contains_non_latin_letter"] += non_latin
                counts["contains_replacement_character"] += "\ufffd" in body
                rating = record.get("rating")
                if type(rating) in (int, float) and rating in RATINGS:
                    by_rating[int(rating)][length] += 1

    scan["traversed_rows"] = traversed
    all_lengths = Counter()
    for histogram in lengths.values():
        all_lengths.update(histogram)
    return {
        "seed": seed,
        "sampling_probability": rate,
        "scan": scan,
        "all_lengths": summarize_length(all_lengths),
        "quality_by_group": quality,
        "length_by_group": {g: summarize_length(lengths[g]) for g in quality},
        "length_by_rating": {rating: summarize_length(hist) for rating, hist in by_rating.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/Books.jsonl.gz"))
    parser.add_argument("--mapping", type=Path, default=Path("data/genre_coverage/book_genres.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/genre_coverage/text_summary.json"))
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.rate <= 1:
        parser.error("--rate must be greater than zero and at most one")
    with args.mapping.open(encoding="utf-8", newline="") as file:
        mapping = {row["parent_asin"]: row["genre"] for row in csv.DictReader(file)}
    result = audit_text(args.reviews, mapping, args.rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Text audit complete | {args.output}", flush=True)


if __name__ == "__main__":
    main()
