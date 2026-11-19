"""Select a genre-balanced development sample and compare fixed markers to hand labels."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path

from audit import WORD
from style_indicators import sentence_markers
from writing_style import clean_body, sentence_parts


ROLES = ("reading_experience", "recommendation", "plot_description", "evaluation")


def rank(stage, key):
    return sha256(f"da-style-audit-v1|{stage}|{key}".encode()).hexdigest()


def sample(snapshot):
    groups = defaultdict(list)
    with (snapshot / "users.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train" and "|" not in row["genres"]:
                groups[row["genres"]].append(row["user_key"])
    selected = {u for users in groups.values() for u in sorted(users, key=lambda v: rank("user", v))[:8]}
    best = {}
    with gzip.open(snapshot / "reviews.jsonl.gz", "rb") as f:
        for line in f:
            row = json.loads(line)
            key = row["user_key"]
            if key not in selected:
                continue
            order = rank("review", key + "|" + row["parent_asin"])
            if key not in best or order < best[key][0]:
                best[key] = order, row["text"]
    items = []
    for user in sorted(selected, key=lambda v: rank("display", v)):
        review_key, text = best[user]
        candidates = [part.strip() for part, marks in sentence_parts(clean_body(text)[0]) if WORD.search(part)]
        i = min(range(len(candidates)), key=lambda j: rank("sentence", f"{review_key}|{j}"))
        items.append((review_key[:12], candidates[i]))
    return items


def compare(items, labels):
    with labels.open(encoding="utf-8-sig", newline="") as f:
        annotations = {row["item"]: row for row in csv.DictReader(f)}
    if len(items) != len(annotations) or set(annotations) != {key for key, _ in items}:
        raise ValueError("The annotation IDs do not match the selected sentences")
    totals = Counter()
    roles = {name: Counter() for name in (*ROLES, "subordination")}
    for key, text in items:
        row = annotations[key]
        words = [w.casefold() for w in WORD.findall(text)]
        totals["words"] += len(words)
        for name in ("adjectives", "adverbs", "finite_subordinate_clauses"):
            totals[name] += int(row[name])
        markers = sentence_markers(words)
        for name, counts in roles.items():
            gold = int(row["finite_subordinate_clauses"] if name == "subordination" else row[name]) > 0
            predicted = name in markers
            counts["annotated_positive"] += gold
            counts["marker_positive"] += predicted
            counts["both_positive"] += gold and predicted
        totals["sentences"] += 1
    return {"counts": dict(totals), "marker_comparison": roles}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=Path("data/analysis/eda-v1"))
    p.add_argument("--labels", type=Path)
    opts = p.parse_args()
    selected = sample(opts.snapshot)
    if opts.labels:
        print(json.dumps(compare(selected, opts.labels), indent=2))
    else:
        for key, text in selected:
            print(key, json.dumps(text, ensure_ascii=True))
