"""Development-only review selection, item metadata and independent preference evidence."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
from hashlib import blake2b, sha256
import json
import math
from pathlib import Path
import re

from audit import EARLIEST, LATEST
from user_genre_profiles import load_mapping
from writing_style import clean_body

BASE = Path("data/analysis/eda-v1")
OUT = Path("data/analysis/phase2-final")
PUB = Path("reports/phase2/patterns")
LABELS = Path("reports/phase2/data_v1_rules.json")
STAMP = datetime(2023, 9, 11, tzinfo=timezone.utc).date()


def write_rows(path, rows):
    rows = iter(rows)
    first = next(rows)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
    tmp.replace(path)


def read_sample():
    return [json.loads(line) for line in (OUT / "sample.jsonl").read_text(encoding="utf-8").splitlines()]


def sample():
    choices = {}
    with gzip.open(BASE / "reviews.jsonl.gz", "rb") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            user = row["user_key"]
            order = sha256(f"phase2-one-v1|{user}|{row['parent_asin']}".encode()).digest()
            if user not in choices or order < choices[user][0]:
                choices[user] = order, row
    seen = set()
    with (BASE / "users.csv").open(encoding="utf-8") as f:
        users = {r["user_key"]: r for r in csv.DictReader(f) if r["split"] == "train"}
    if choices.keys() != users.keys():
        raise ValueError("Development roster mismatch")
    tmp = OUT / "sample.partial"
    with tmp.open("w", encoding="utf-8") as f:
        for user in sorted(choices):
            row = choices[user][1]
            clean = " ".join(clean_body(row["text"])[0].casefold().split())
            h = sha256(clean.encode()).hexdigest()
            row.update(genres=users[user]["genres"], eligible_reviews=int(users[user]["eligible_reviews"]),
                       duplicate_sample_body=h in seen)
            seen.add(h)
            f.write(json.dumps(row) + "\n")
    tmp.replace(OUT / "sample.jsonl")
    print("Development reviews", len(users), "unique normalized bodies", len(seen))


def metadata():
    records = read_sample()
    books = {r["parent_asin"] for r in records}
    prices = defaultdict(set)
    catalog = set()
    with gzip.open("data/raw/meta_Books.jsonl.gz", "rb") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            b = r.get("parent_asin")
            if b in books:
                catalog.add(b)
                try:
                    p = float(r.get("price"))
                    if math.isfinite(p) and p > 0:
                        prices[b].add(p)
                except (ValueError, TypeError):
                    pass
            if i % 1000000 == 0:
                print("Metadata", i, flush=True)
    def structured(r):
        b = r["parent_asin"]
        values = prices[b]
        price = next(iter(values)) if len(values) == 1 else None
        when = datetime.fromtimestamp(r["timestamp"]/1000, timezone.utc).date() if r["timestamp"] is not None else None
        return dict(user_key=r["user_key"], parent_asin=b, book_genre=r["book_genre"] or "UNMAPPED",
                    rating=r["rating"], review_age_days=(STAMP-when).days if when else None,
                    review_year=when.year if when else None, verified=r["verified_purchase"],
                    helpful_votes=r["helpful_vote"], log_helpful=math.log1p(r["helpful_vote"]) if r["helpful_vote"] is not None else None,
                    review_tokens=r["tokens"], eligible_reviews=r["eligible_reviews"], price=price,
                    log_price=math.log1p(price) if price is not None else None, price_missing=int(price is None),
                    meta_found=int(b in catalog), price_conflict=int(len(values)>1))
    write_rows(OUT / "context.csv", (structured(r) for r in records))
    print("Catalog records", i, "matched titles", len(catalog))


def qualifying(books, mapping, rules):
    by_genre = defaultdict(list)
    for b, t in books.items():
        if b in mapping:
            by_genre[mapping[b]].append(t)
    n = len(books)
    if not n or sum(map(len, by_genre.values()))/n < rules["minimum_mapping_coverage"]:
        return []
    result = []
    for g, times in by_genre.items():
        dated = [t for t in times if t is not None]
        if (len(times) < rules["minimum_genre_books"] or len(times)/n < rules["minimum_share_all_books"]
                or len(dated) < rules["minimum_dated_genre_books"]):
            continue
        dates = [datetime.fromtimestamp(t/1000, timezone.utc) for t in dated]
        quarters = {(d.year, (d.month-1)//3) for d in dates}
        if (len(quarters) >= rules["minimum_active_quarters"]
                and (max(dated)-min(dated))/86400000 >= rules["minimum_span_days"]):
            result.append(g)
    return sorted(result)


def independent():
    records = read_sample()
    histories = {r["user_key"]: {} for r in records}
    mapping = load_mapping(Path("data/genre_coverage/book_genres.csv"))
    rules = json.loads(LABELS.read_text(encoding="utf-8"))
    # Raw user IDs are ASCII in this export. Fall back to JSON decoding for other encodings.
    uid = re.compile(rb'(?<!\\)"user_id"\s*:\s*"([A-Za-z0-9_-]+)"')
    with gzip.open("data/raw/Books.jsonl.gz", "rb") as f:
        for i, line in enumerate(f, 1):
            if i % 5000000 == 0:
                print("History", i, flush=True)
            m = uid.search(line)
            r = None
            if m:
                raw = m[1]
            else:
                r = json.loads(line)
                raw_user = r.get("user_id")
                if not isinstance(raw_user, str):
                    continue
                raw = raw_user.encode("utf-8", errors="surrogatepass")
            key = blake2b(raw, digest_size=16).hexdigest()
            if key not in histories:
                continue
            r = r if r is not None else json.loads(line)
            b, t = r.get("parent_asin"), r.get("timestamp")
            if not isinstance(b,str) or not b.strip():
                continue
            hist = histories[key]
            t = t if type(t) is int and EARLIEST <= t < LATEST else None
            if b not in hist or (t is not None and (hist[b] is None or t < hist[b])):
                hist[b] = t
    if i != rules["expected_source_rows"]:
        raise ValueError("Incomplete archive scan")
    result = []
    for r in records:
        hist = histories[r["user_key"]]
        before = qualifying(hist, mapping, rules)
        if before != sorted(r["genres"].split("|")):
            raise ValueError("Whole-history reconstruction differs from frozen membership")
        del hist[r["parent_asin"]]
        after = qualifying(hist, mapping, rules)
        result.append(dict(user_key=r["user_key"], original="|".join(before), independent="|".join(after),
                           remaining_books=len(hist)))
    write_rows(OUT / "independent.csv", result)
    print("Verified original labels and recomputed independent evidence for", len(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("sample","metadata","independent"))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    {"sample":sample,"metadata":metadata,"independent":independent}[args.step]()
