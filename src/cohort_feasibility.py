"""Count exploratory genre cohorts and their available writing, without labeling interest.

Run from the repository root. Private per-user measurements remain under data/.
Only aggregate counts are published under reports/.
"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import blake2b, sha256
import gzip
import itertools
import json
from pathlib import Path
import time

from audit import WORD, GENRE_RULES
from user_genre_profiles import load_mapping, write_csv


GENRES = tuple(GENRE_RULES)
PERSIAN = dict(zip(GENRES, ("فانتزی", "علمی‌تخیلی", "عاشقانه", "معمایی", "تریلر", "فلسفی", "علمی")))
COUNTS = (2, 3, 5, 10)
SHARES = (.25, .5, .75)
TOKENS = (500, 1000, 2000)


def source_signature(paths):
    return {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths}


def scan_text(reviews, profiles, mapping_path, output, expected_rows):
    """Keep the longest body per user/book to avoid inflating writing with repeats."""
    candidates = set()
    with profiles.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["distinct_books"]) >= min(COUNTS):
                candidates.add(bytes.fromhex(row["user_key"]))
    fingerprint = source_signature((reviews, profiles, mapping_path))
    fingerprint["rules"] = {"minimum_genre_books": min(COUNTS), "minimum_review_tokens": 20,
                            "book_text": "longest_body", "word_regex": WORD.pattern,
                            "mapping_sha256": sha256(mapping_path.read_bytes()).hexdigest()}
    cache = output / "text_scan.json"
    complete = output / "user_text.csv"
    if cache.exists():
        state = json.loads(cache.read_text(encoding="utf-8"))
        if state["input"] != fingerprint:
            raise ValueError("Cached inputs differ; choose a new --data-dir")
        if state["rows"] != expected_rows:
            raise ValueError("Cached source row count differs from --expected-rows")
        if complete.exists():
            return complete, state
    else:
        # A failed scan can be repeated. Its unfinished partition files contain no results.
        paths = [output / f"text-{i:03}.partial" for i in range(128)]
        handles = [path.open("w", encoding="utf-8", newline="") for path in paths]
        writers = [csv.writer(handle) for handle in handles]
        started = time.monotonic()
        matched = 0
        rows = 0
        try:
            with gzip.open(reviews, "rb") as handle:
                for rows, line in enumerate(handle, 1):
                    record = json.loads(line)
                    user, parent = record.get("user_id"), record.get("parent_asin")
                    if isinstance(user, str) and isinstance(parent, str) and parent.strip():
                        key = blake2b(user.encode("utf-8", errors="surrogatepass"), digest_size=16).digest()
                        if key in candidates:
                            body = record.get("text")
                            length = len(WORD.findall(body)) if isinstance(body, str) else 0
                            writers[key[0] % 128].writerow((key.hex(), parent, length))
                            matched += 1
                    if rows % 1_000_000 == 0:
                        print(f"Text scan | {rows:,} rows | {time.monotonic() - started:.0f}s", flush=True)
        finally:
            for handle in handles:
                handle.close()
        if rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows; observed {rows}")
        state = {"input": fingerprint, "rows": rows, "candidate_users": len(candidates),
                 "candidate_review_rows": matched}
        cache.write_text(json.dumps(state, indent=2), encoding="utf-8")
    del candidates
    mapping = load_mapping(mapping_path)
    fields = ("user_key", "genre", "tokens", "texts_at_least_20_tokens")
    partial = complete.with_suffix(".csv.partial")
    with partial.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        for index in range(128):
            books = {}
            with (output / f"text-{index:03}.partial").open(encoding="utf-8", newline="") as handle:
                for key, parent, value in csv.reader(handle):
                    pair = (key, parent)
                    books[pair] = max(books.get(pair, 0), int(value))
            totals = defaultdict(lambda: [0, 0])
            for (key, parent), length in books.items():
                for genre in ("ALL", mapping.get(parent, "UNMAPPED")):
                    aggregate = totals[(key, genre)]
                    aggregate[0] += length
                    aggregate[1] += length >= 20
            for (key, genre), (tokens, texts) in totals.items():
                writer.writerow(dict(zip(fields, (key, genre, tokens, texts))))
            if (index + 1) % 16 == 0:
                print(f"Text reduction | {index + 1}/128 partitions", flush=True)
    partial.replace(complete)
    return complete, state


def aggregate(profiles, text_path):
    writing = {}
    with text_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = writing.setdefault(row["user_key"], {})
            values[row["genre"]] = (int(row["tokens"]), int(row["texts_at_least_20_tokens"]))
    grid = Counter()
    stages = defaultdict(Counter)
    for genre in GENRES:
        stages[genre]  # Preserve empty groups.
    with profiles.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            genre, n = row["genre"], int(row["distinct_books"])
            stages[genre]["any_observed_book"] += 1
            if n < min(COUNTS):
                continue
            user_text = writing.get(row["user_key"])
            if user_text is None:
                raise ValueError("Missing candidate text measurement")
            all_tokens, all_texts = user_text["ALL"]
            own_tokens, own_texts = user_text.get(genre, (0, 0))
            outside_tokens, outside_texts = all_tokens - own_tokens, all_texts - own_texts
            mapped_share = float(row["share_mapped_books"])
            all_share = float(row["share_all_books"])
            coverage = float(row["mapping_coverage"])
            quarters = int(row["active_quarters"])
            span = float(row["span_days"]) if row["span_days"] else 0
            dated = int(row["dated_books"])
            count_ok = n >= 3
            share_ok = count_ok and mapped_share >= .5
            coverage_ok = share_ok and coverage >= .5
            continuity_ok = coverage_ok and quarters >= 2 and dated >= 3 and span >= 180
            for field, passes in (("at_least_3_books", count_ok), ("plus_share_mapped_50", share_ok),
                                  ("plus_coverage_50", coverage_ok), ("plus_continuity", continuity_ok)):
                stages[genre][field] += passes
            for threshold in TOKENS:
                stages[genre][f"plus_all_text_{threshold}"] += (
                    continuity_ok and all_tokens >= threshold and all_texts >= 3)
                stages[genre][f"plus_outside_text_{threshold}"] += (
                    continuity_ok and outside_tokens >= threshold and outside_texts >= 3)
            for denominator, share in (("mapped_books", mapped_share), ("all_books", all_share)):
                for minimum, minimum_share, minimum_coverage, minimum_quarters, minimum_span in itertools.product(
                        COUNTS, SHARES, (0, .5), (1, 2), (0, 180)):
                    if (n < minimum or share < minimum_share or coverage < minimum_coverage
                            or quarters < minimum_quarters or span < minimum_span
                            or (minimum_span > 0 and dated < minimum)):
                        continue
                    key = (genre, denominator, minimum, minimum_share, minimum_coverage,
                           minimum_quarters, minimum_span)
                    grid[(*key, "none", 0)] += 1
                    for scope, tokens, texts in (("all_history", all_tokens, all_texts),
                                                 ("outside_target_genre", outside_tokens, outside_texts)):
                        for threshold in TOKENS:
                            if tokens >= threshold and texts >= 3:
                                grid[(*key, scope, threshold)] += 1
    return stages, grid


def publish(stages, grid, state, folder):
    folder.mkdir(parents=True, exist_ok=True)
    stage_fields = ("any_observed_book", "at_least_3_books", "plus_share_mapped_50",
                    "plus_coverage_50", "plus_continuity",
                    *(f"plus_all_text_{n}" for n in TOKENS),
                    *(f"plus_outside_text_{n}" for n in TOKENS))
    write_csv(folder / "cohort_stages.csv", [
        {"genre": g, **{field: stages[g][field] for field in stage_fields}} for g in GENRES])
    grid_fields = ("genre", "share_denominator", "minimum_books", "minimum_share", "minimum_coverage",
                   "minimum_active_quarters", "minimum_span_days", "text_scope", "minimum_tokens", "users")
    rows = []
    for base in itertools.product(GENRES, ("mapped_books", "all_books"), COUNTS, SHARES, (0, .5), (1, 2), (0, 180)):
        for scope, threshold in (("none", 0), *itertools.product(("all_history", "outside_target_genre"), TOKENS)):
            key = (*base, scope, threshold)
            rows.append(dict(zip(grid_fields, (*key, grid[key]))))
    write_csv(folder / "cohort_sensitivity.csv", rows)
    report = ["# امکان‌سنجی گروه‌های گرایش ژانری و حجم متن", "",
              "این جدول برای مقایسهٔ قواعد آزمایشی گروه‌بندی کاربران است. آستانه‌ها تعریف نهایی علاقه نیستند.", "",
              f"کل آرشیو با {state['rows']:,} ردیف پیمایش شد. حجم متن برای {state['candidate_users']:,} کاربر دارای حداقل دو کتاب در یکی از ژانرها محاسبه شد.", "",
              "## شمارش مرحله‌ای", "",
              "شرط‌های ستون‌ها تجمعی‌اند. سهم در این جدول نسبت به کتاب‌های قابل‌نگاشت است. پوشش حداقل ۵۰٪ و تداوم شامل دست‌کم سه کتاب تاریخ‌دار، دو سه‌ماههٔ فعال و فاصلهٔ حداقل ۱۸۰ روز است.", "",
              "| ژانر | ≥۱ کتاب | ≥۳ کتاب | + سهم ≥۵۰٪ | + پوشش ≥۵۰٪ | + تداوم | + ۵۰۰ توکن | + ۱۰۰۰ توکن | + ۲۰۰۰ توکن | + ۱۰۰۰ توکن خارج ژانر |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    display = (*stage_fields[:5], *(f"plus_all_text_{n}" for n in TOKENS), "plus_outside_text_1000")
    for genre in GENRES:
        report.append("| " + PERSIAN[genre] + " | " + " | ".join(f"{stages[genre][k]:,}" for k in display) + " |")
    report += ["", "ستون‌های حجم متن هرکدام از ستون تداوم شروع می‌شوند. شرط حجم متن علاوه بر مجموع توکن، حداقل سه متن از سه کتاب متمایز با طول دست‌کم ۲۰ توکن دارد. ستون خارج ژانر همین شرط را فقط روی کتاب‌هایی اعمال می‌کند که در نگاشت به ژانر هدف نسبت داده نشده‌اند.", "",
               "## اثر سهم ژانر و انتخاب مخرج", "",
               "در جدول زیر حداقل سه کتاب در ژانر، پوشش ۵۰٪، دو سه‌ماهه، سه کتاب تاریخ‌دار با فاصلهٔ ۱۸۰ روز، حداقل ۱۰۰۰ توکن در کل سابقه و سه متن حداقل ۲۰ توکنی ثابت‌اند. فقط سهم و مخرج آن تغییر می‌کنند.", "",
               "| ژانر | سهم ≥۲۵٪ از قابل‌نگاشت | سهم ≥۵۰٪ از قابل‌نگاشت | سهم ≥۷۵٪ از قابل‌نگاشت | سهم ≥۲۵٪ از کل سابقه | سهم ≥۵۰٪ از کل سابقه | سهم ≥۷۵٪ از کل سابقه |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for genre in GENRES:
        values = [grid[(genre, denominator, 3, share, .5, 2, 180, "all_history", 1000)]
                  for denominator in ("mapped_books", "all_books") for share in SHARES]
        report.append("| " + PERSIAN[genre] + " | " + " | ".join(f"{value:,}" for value in values) + " |")
    report += ["",
               "## سهم، پوشش و حساسیت", "",
               "در cohort_sensitivity.csv هر دو مخرج سهم بررسی شده‌اند: تمام کتاب‌های متمایز کاربر و فقط کتاب‌های قابل‌نگاشت. شبکه شامل حداقل ۲، ۳، ۵ یا ۱۰ کتاب، سهم ۲۵٪، ۵۰٪ یا ۷۵٪، پوشش صفر یا ۵۰٪، یک یا دو سه‌ماهه و فاصلهٔ صفر یا ۱۸۰ روز است. برای شرط فاصله، حداقل تعداد انتخابی کتاب تاریخ‌دار لازم است. ردیف‌های صفر هم ذخیره شده‌اند.", "",
               "سهم از کل سابقه شامل کتاب‌های مبهم و بیرون از نگاشت در مخرج است. سهم بالا در دامنهٔ هفت‌گانه باید همراه پوشش نگاشت تفسیر شود. هر کاربر ممکن است در چند گروه حضور داشته باشد؛ جمع سطرها تعداد کاربران یکتا نیست.", "",
               "## سنجش متن و حدود تفسیر", "",
               "توکن با همان الگوی حروف و اعداد در ممیزی متن پروژه شمرده شده است. از بدنهٔ نقد استفاده می‌شود. برای هر جفت کاربر و parent_asin فقط طولانی‌ترین بدنه در کل آرشیو شمرده می‌شود؛ بنابراین رکوردهای تکراری و نسخه‌های دیگر نقد یک کتاب حجم متن را چندبرابر نمی‌کنند. parent_asin تضمین یکسان‌سازی تمام نسخه‌های یک اثر نیست.", "",
               "حجم متنِ کل سابقه یک سنجش اولیهٔ دسترس‌پذیری است. در این شمارش، نقدهای مبنای گرایش از متن تحلیل جدا نشده‌اند. پیش از مقایسهٔ زبانی باید بخش مستقلی برای تعیین گرایش و بخش دیگری برای سنجش زبان تعریف و کفایت دوباره محاسبه شود.", "",
               "خارج ژانر شامل کتاب‌های بدون نگاشت معتبر نیز هست؛ بنابراین معادل نقدهای تأییدشده دربارهٔ یک ژانر مشترک نیست. جدول حاضر دسترس‌پذیری متن مستقل زمانی یا مقایسهٔ هم‌ژانر را تأیید نمی‌کند.", "",
               "این داده کتاب‌های نقدشده را ثبت می‌کند. زبان متن‌ها شناسایی نشده و نویسنده یا مجموعه و طول فرصت مشاهده هم در این جدول کنترل نشده‌اند. آستانه‌های حجم متن آزمایشی‌اند و اعتبار اندازه‌گیری سبک زبانی را تضمین نمی‌کنند.", "",
               "## بازاجرا", "", "```powershell", "python src/cohort_feasibility.py", "```", "",
               "ورودی‌ها و قواعد سنجش در data/cohort_feasibility/text_scan.json ثبت شده‌اند. سنجش فردی و فایل‌های موقت در data می‌مانند. پس از پایان اسکن کامل، اجرای دوباره از سنجش ذخیره‌شده استفاده می‌کند. فایل‌های فردی شامل کلید مستعار، ژانر و شمارش متن‌اند و متن خام ذخیره نمی‌کنند."]
    (folder / "report-fa.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/Books.jsonl.gz"))
    parser.add_argument("--profiles", type=Path, default=Path("data/user_genre_profiles/user_genre_profiles.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("data/genre_coverage/book_genres.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/cohort_feasibility"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/cohort_feasibility"))
    parser.add_argument("--expected-rows", type=int, default=29475453)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    text, state = scan_text(args.reviews, args.profiles, args.mapping, args.data_dir, args.expected_rows)
    print("Aggregating cohort scenarios", flush=True)
    stages, grid = aggregate(args.profiles, text)
    publish(stages, grid, state, args.report_dir)
    print(f"Complete | {args.report_dir / 'report-fa.md'}", flush=True)


if __name__ == "__main__":
    main()
