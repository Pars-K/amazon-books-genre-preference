"""Measure review-level background distributions using a seeded sample of the full archive."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import random
from statistics import median, quantiles
from tempfile import NamedTemporaryFile

from audit import WORD
from style_indicators import METRICS, counts_from_text, metrics


def run(raw, snapshot, reports):
    groups = defaultdict(lambda: defaultdict(list))
    sizes, scanned = Counter(), Counter()
    def append(scope, values):
        sizes[scope] += 1
        for k, v in values.items():
            if v is not None:
                groups[scope][k].append(v)

    rng = random.Random(20261119)
    with gzip.open(raw, "rb") as f:
        for i, line in enumerate(f, 1):
            if i % 5_000_000 == 0:
                print(f"Reference: {i:,} archive rows", flush=True)
            if rng.random() >= .001:
                continue
            scanned["sampled_raw"] += 1
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            values = metrics(counts_from_text(text))
            append("archive_nonblank", values)
            if len(WORD.findall(text)) >= 20:
                append("archive_min20", values)
    scanned["archive_rows"] = i
    rng = random.Random(20261120)
    with gzip.open(snapshot / "reviews.jsonl.gz", "rb") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "train" and rng.random() < 1 / 16:
                append("eligible_development", metrics(counts_from_text(row["text"])))
    reports.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=reports, delete=False) as f:
        out = csv.writer(f)
        out.writerow(["scope","sample_reviews","metric","valid_reviews","p25","median","p75"])
        for scope, columns in groups.items():
            for col in METRICS:
                v = columns[col]
                q = quantiles(v, n=4, method="inclusive")
                out.writerow([scope, sizes[scope], col, len(v), round(q[0],6), round(median(v),6), round(q[2],6)])
    Path(f.name).replace(reports / "review_reference.csv")
    print(json.dumps({**scanned, **sizes}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/Books.jsonl.gz"))
    p.add_argument("--snapshot", type=Path, default=Path("data/analysis/eda-v1"))
    p.add_argument("--reports", type=Path, default=Path("reports/phase2/style-v1"))
    args = p.parse_args()
    run(args.raw, args.snapshot, args.reports)
