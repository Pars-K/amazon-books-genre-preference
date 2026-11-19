"""Measure the frozen review corpus; export private review/user tables and descriptive summaries."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import gzip
from hashlib import file_digest
from itertools import batched, groupby
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from style_indicators import COUNTS, METRICS, counts_from_text, metrics


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def measure_batch(lines):
    result = []
    for line in lines:
        row = json.loads(line)
        result.append((row["user_key"], row["parent_asin"], row["book_genre"] or "",
                       row["split"], row["tokens"], counts_from_text(row["text"])))
    return result


def make_writer(stack, path, columns):
    if path.suffix == ".gz":
        f = stack.enter_context(gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1))
    else:
        f = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.writer(f)
    writer.writerow(columns)
    return writer


def metric_row(values):
    return [round(values[k], 6) if values[k] is not None else None for k in METRICS]


def summarize(path, groups, users, reviews):
    def quantile(ordered, p):
        pos = (len(ordered) - 1) * p
        lower = int(pos)
        return round(ordered[lower] + (pos - lower) * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]), 6)

    with ExitStack() as stack:
        writer = make_writer(stack, path, ("scope", "preference_group", "users", "reviews",
                                          "metric", "valid_users", "p25", "median", "p75"))
        for key, columns in sorted(groups.items()):
            for name in METRICS:
                v = sorted(columns[name])
                writer.writerow([*key, users[key], reviews[key], name, len(v),
                                 *([quantile(v, p) for p in (.25, .5, .75)] if v else [None] * 3)])


def run(snapshot, private, reports, workers):
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    inputs = {}
    for name in ("users.csv", "memberships.csv", "reviews.jsonl.gz"):
        expected = next(v["sha256"] for k, v in manifest["outputs"].items()
                        if k.replace("\\", "/").split("/")[-1] == name)
        inputs[name] = digest(snapshot / name)
        if inputs[name] != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    with (snapshot / "users.csv").open(encoding="utf-8", newline="") as f:
        registry = {r["user_key"]: r for r in csv.DictReader(f)}
    members = defaultdict(list)
    with (snapshot / "memberships.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            members[r["user_key"]].append(r["genre"])
    private.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    group_users, group_reviews, totals = Counter(), Counter(), Counter()
    def collect(scope, labels, counts, values):
        for label in ["ALL", *labels]:
            key = (scope, label)
            group_users[key] += 1
            group_reviews[key] += counts["reviews"]
            for name, value in values.items():
                if value is not None:
                    groups[key][name].append(value)

    with TemporaryDirectory(dir=private) as temp:
        stage = Path(temp)
        with ExitStack() as stack:
            review_out = make_writer(stack, stage / "review_style.csv.gz",
                                     ["user_key", "parent_asin", "book_genre", "split", *COUNTS, *METRICS])
            user_out = make_writer(stack, stage / "user_style.csv",
                                   ["user_key", "split", "genres", *COUNTS, *METRICS])
            by_book_out = make_writer(stack, stage / "user_book_genre_style.csv.gz",
                                      ["user_key", "book_genre", "split", *COUNTS, *METRICS])
            source = stack.enter_context(gzip.open(snapshot / "reviews.jsonl.gz", "rb"))
            batches = batched(source, 192)
            if workers > 1:
                pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
                measured = pool.map(measure_batch, batches, buffersize=workers * 2)
            else:
                measured = map(measure_batch, batches)
            rows = (r for batch in measured for r in batch)
            seen = set()
            for user, records in groupby(rows, key=lambda r: r[0]):
                entry = registry[user]
                if user in seen or sorted(members[user]) != sorted(entry["genres"].split("|")):
                    raise ValueError("User grouping or membership mismatch")
                seen.add(user)
                full, screened = Counter(), Counter()
                by_book = defaultdict(Counter)
                previous = ""
                raw_tokens = 0
                split = entry["split"]
                for _, book, genre, row_split, n, counts in records:
                    if book <= previous or row_split != split:
                        raise ValueError("Review keys or split mismatch")
                    previous = book
                    raw_tokens += n
                    full.update(counts)
                    by_book[genre].update(counts)
                    if not counts["quote_reviews"] and not counts["no_english_anchor_reviews"]:
                        screened.update(counts)
                    review_out.writerow([user, book, genre, split, *[counts[k] for k in COUNTS],
                                         *metric_row(metrics(counts))])
                if full["reviews"] != int(entry["eligible_reviews"]) or raw_tokens != int(entry["eligible_tokens"]):
                    raise ValueError("Per-user input counts mismatch")
                totals.update(full)
                overall = metrics(full)
                user_out.writerow([user, split, entry["genres"], *[full[k] for k in COUNTS], *metric_row(overall)])
                development = split == "train"
                if development:
                    collect("all_reviews", members[user], full, overall)
                    if screened["reviews"]:
                        collect("no_quote_with_english_anchor", members[user], screened, metrics(screened))
                for genre, counts in by_book.items():
                    values = metrics(counts)
                    by_book_out.writerow([user, genre, split, *[counts[k] for k in COUNTS], *metric_row(values)])
                    if development:
                        collect("book=" + (genre or "UNMAPPED"), members[user], counts, values)
                if len(seen) % 2000 == 0:
                    print(f"{len(seen):,} users | {totals['reviews']:,} reviews", flush=True)
            if seen != registry.keys():
                raise ValueError("Missing users")
        summarize(stage / "group_style.csv", groups, group_users, group_reviews)
        results = {
            "snapshot": manifest["version"], "inputs_sha256": inputs,
            "users": len(registry), "memberships": sum(map(len, members.values())),
            "counts": dict(totals), "metrics": list(METRICS), "group_statistics_split": "train",
            "source_sha256": {n: digest(Path(__file__).with_name(n)) for n in
                              ("measure_corpus_style.py", "style_indicators.py", "writing_style.py", "audit.py")},
            "output_sha256": {p.name: digest(p) for p in sorted(stage.iterdir())},
        }
        (stage / "measurement.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        for p in sorted(stage.iterdir()):
            destination = reports if p.name in {"group_style.csv", "measurement.json"} else private
            p.replace(destination / p.name)
    print(f"Complete: {len(registry):,} users | {totals['reviews']:,} reviews", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/analysis/eda-v1"))
    parser.add_argument("--private", type=Path, default=Path("data/analysis/style-v1"))
    parser.add_argument("--reports", type=Path, default=Path("reports/phase2/style-v1"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.snapshot, args.private, args.reports, args.workers)
