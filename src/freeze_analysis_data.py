"""Build a versioned, private review snapshot and publish aggregate provenance.

Whole-history engagement defines exploratory groups. These memberships are not
prospective prediction labels. Review text and metadata remain distinct fields.
"""

import argparse
from collections import Counter
import csv
from hashlib import blake2b, sha256
import gzip
import json
from pathlib import Path
import sqlite3
import time

from audit import EARLIEST, LATEST, WORD
from user_genre_profiles import GENRES, load_mapping, write_csv


PERSIAN = dict(zip(GENRES, ("فانتزی", "علمی‌تخیلی", "عاشقانه", "معمایی", "تریلر", "فلسفی", "علمی")))
def file_hash(path):
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(paths):
    return {str(path): {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths}


def assign_split(key, rules):
    value = int.from_bytes(sha256((rules["split_seed"] + key).encode()).digest()[:8], "big") / 2**64
    if value < rules["train_fraction"]:
        return "train"
    if value < rules["train_fraction"] + rules["validation_fraction"]:
        return "validation"
    return "test"


def candidate_memberships(profile_path, text_path, rules):
    candidates = {}
    with profile_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (int(row["distinct_books"]) >= rules["minimum_genre_books"]
                    and float(row["share_all_books"]) >= rules["minimum_share_all_books"]
                    and float(row["mapping_coverage"]) >= rules["minimum_mapping_coverage"]
                    and int(row["dated_books"]) >= rules["minimum_dated_genre_books"]
                    and int(row["active_quarters"]) >= rules["minimum_active_quarters"]
                    and float(row["span_days"] or 0) >= rules["minimum_span_days"]):
                candidates.setdefault(row["user_key"], []).append(row)
    accepted = set()
    with text_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["genre"] == "ALL" and row["user_key"] in candidates
                    and int(row["tokens"]) >= rules["minimum_user_tokens"]
                    and int(row["texts_at_least_20_tokens"]) >= rules["minimum_user_reviews"]):
                accepted.add(row["user_key"])
    return {key: candidates[key] for key in sorted(accepted)}


def select_review(record, source_row, mapping, rules):
    parent, text = record.get("parent_asin"), record.get("text")
    if not isinstance(parent, str) or not parent.strip() or not isinstance(text, str):
        return None
    tokens = len(WORD.findall(text))
    if tokens < rules["minimum_review_tokens"]:
        return None
    stamp, rating = record.get("timestamp"), record.get("rating")
    stamp = stamp if type(stamp) is int and EARLIEST <= stamp < LATEST else None
    rating = rating if type(rating) in (int, float) and rating in (1, 2, 3, 4, 5) else None
    votes = record.get("helpful_vote")
    votes = votes if type(votes) is int and votes >= 0 else None
    verified = record.get("verified_purchase")
    verified = verified if type(verified) is bool else None
    # A hash gives deterministic tie-breaking independent of input row order.
    payload = {"parent_asin": parent, "book_genre": mapping.get(parent), "text": text,
               "tokens": tokens, "timestamp": stamp, "rating": rating,
               "verified_purchase": verified, "helpful_vote": votes}
    digest = sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode()).hexdigest()
    payload["source_row"] = source_row
    return tokens, digest, payload


def scan_reviews(reviews, memberships, mapping, rules, db_path, state_path, inputs):
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS reviews (user_key TEXT, parent TEXT, tokens INTEGER, "
                       "digest TEXT, payload TEXT, PRIMARY KEY(user_key, parent)) WITHOUT ROWID")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["inputs"] != inputs:
            raise ValueError("Snapshot inputs changed; choose a new --output-dir")
        if state.get("scan_complete"):
            return connection, state
    state = {"inputs": inputs, "scan_complete": False}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    keys = {bytes.fromhex(key) for key in memberships}
    started = time.monotonic()
    counts = Counter()
    statement = ("INSERT INTO reviews VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_key,parent) DO UPDATE SET "
                 "tokens=excluded.tokens,digest=excluded.digest,payload=excluded.payload "
                 "WHERE excluded.tokens>reviews.tokens OR "
                 "(excluded.tokens=reviews.tokens AND excluded.digest<reviews.digest)")
    with gzip.open(reviews, "rb") as handle:
        for source_row, line in enumerate(handle, 1):
            counts["source_rows"] += 1
            record = json.loads(line)
            user = record.get("user_id")
            if isinstance(user, str):
                key = blake2b(user.encode("utf-8", errors="surrogatepass"), digest_size=16).digest()
                if key in keys:
                    counts["candidate_source_rows"] += 1
                    selected = select_review(record, source_row, mapping, rules)
                    if selected is not None:
                        tokens, digest, payload = selected
                        connection.execute(statement, (key.hex(), payload["parent_asin"], tokens,
                                                       digest, json.dumps(payload, ensure_ascii=True)))
                        counts["eligible_source_rows"] += 1
            if source_row % 1_000_000 == 0:
                connection.commit()
                print(f"Snapshot scan | {source_row:,} rows | {time.monotonic() - started:.0f}s", flush=True)
    connection.commit()
    if counts["source_rows"] != rules["expected_source_rows"]:
        raise ValueError("Source row count differs from configured expectation")
    state.update(scan_complete=True, counts=dict(counts))
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return connection, state


def export_snapshot(connection, memberships, rules, output):
    totals = {key: (n, tokens) for key, n, tokens in connection.execute(
        "SELECT user_key,COUNT(*),SUM(tokens) FROM reviews GROUP BY user_key")
        if n >= rules["minimum_user_reviews"] and tokens >= rules["minimum_user_tokens"]}
    summaries = {g: Counter() for g in GENRES}
    user_rows, membership_rows = [], []
    splits = {}
    for key in sorted(totals):
        n, tokens = totals[key]
        split = assign_split(key, rules)
        splits[key] = split
        rows = memberships[key]
        user_rows.append({"user_key": key, "split": split, "eligible_reviews": n,
                          "eligible_tokens": tokens, "genres": "|".join(sorted(r["genre"] for r in rows))})
        for row in rows:
            genre = row["genre"]
            summaries[genre]["users"] += 1
            summaries[genre][f"{split}_users"] += 1
            membership_rows.append({"user_key": key, "split": split, "genre": genre,
                                    **{field: row[field] for field in (
                                        "distinct_books", "share_all_books", "share_mapped_books",
                                        "mapping_coverage", "dated_books", "active_quarters", "span_days")}})
    if not user_rows:
        raise ValueError("No users satisfy the snapshot criteria")
    write_csv(output / "users.csv", user_rows)
    write_csv(output / "memberships.csv", membership_rows)
    split_counts = {split: Counter() for split in ("train", "validation", "test")}
    by_user_genres = {key: {row["genre"] for row in memberships[key]} for key in totals}
    reviews_path = output / "reviews.jsonl.gz"
    with reviews_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as dest:
            for key, payload in connection.execute("SELECT user_key,payload FROM reviews ORDER BY user_key,parent"):
                if key not in totals:
                    continue
                record = json.loads(payload)
                record = {"user_key": key, "split": splits[key], **record}
                dest.write((json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n").encode())
                split_counts[splits[key]]["reviews"] += 1
                split_counts[splits[key]]["tokens"] += record["tokens"]
                for genre in by_user_genres[key]:
                    summaries[genre]["all_genre_reviews"] += 1
                    summaries[genre]["all_genre_tokens"] += record["tokens"]
                    if record["book_genre"] == genre:
                        summaries[genre]["target_genre_reviews"] += 1
                        summaries[genre]["target_genre_tokens"] += record["tokens"]
    for split in split_counts:
        split_counts[split]["users"] = sum(value == split for value in splits.values())
    return {"candidate_users": len(memberships), "retained_users": len(totals),
            "removed_after_short_text_filter": len(memberships) - len(totals),
            "overlapping_memberships": sum(len(memberships[key]) > 1 for key in totals),
            "splits": {key: dict(value) for key, value in split_counts.items()},
            "genres": {g: dict(summaries[g]) for g in GENRES}}


def publish_report(manifest, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = manifest["summary"]
    rows = [{"genre": g, **summary["genres"][g]} for g in GENRES]
    write_csv(report_dir / "data_v1_counts.csv", rows)
    (report_dir / "data_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = ["# تثبیت دادهٔ اولیهٔ فاز دوم", "",
            "## هدف و وضعیت", "",
            "هدف اصلی، شناسایی و مقایسهٔ ویژگی‌های زبانیِ نقدنویسان دارای گرایش به ژانرهای مختلف، با در نظر گرفتن ژانر کتاب نقدشده است. واحد اصلی تحلیل کاربر، واحد استخراج ویژگی نقد و واحد عضویت جفت کاربر–ژانر است.", "",
            "نسخهٔ eda-v1 برای بررسی اکتشافی و شروع مهندسی ویژگی تثبیت شد. عضویت گروه بر اساس کل سابقهٔ نقدنویسی محاسبه شده است. این عضویت یک شاخص رفتاری مشاهده‌شده است. اعتبارسنجی علاقهٔ واقعی یا پیش‌بینی آینده نیازمند طراحی جداگانه است.", "",
            "## قواعد ورود", "",
            "حداقل سه کتاب متمایز در ژانر، سهم حداقل ۵۰٪ از تمام کتاب‌های متمایز کاربر، حداقل سه کتاب تاریخ‌دار در ژانر، حضور در دو سه‌ماهه و فاصلهٔ حداقل ۱۸۰ روز بین اولین و آخرین کتاب لازم است. پوشش نگاشت حداقل ۵۰٪ هم ثبت شده است؛ با تعریف سهم از کل سابقه این شرط به‌طور منطقی تأمین می‌شود.", "",
            "برای متن، از هر جفت کاربر و parent_asin طولانی‌ترین بدنه نگه داشته می‌شود. در تساوی طول، هش محتوای رکورد تعیین‌کننده است. متن‌های کمتر از ۲۰ توکن کنار گذاشته می‌شوند و پس از این حذف، هر کاربر باید حداقل سه متن و حداقل ۱۰۰۰ توکن داشته باشد. حجم متن شامل تمام ژانرها و کتاب‌های بدون نگاشت نیز هست.", "",
            "هیچ شرط امتیاز مثبت، وزن‌دهی، بازنمونه‌گیری، سقف تعداد نقد یا حذف ژانر اعمال نشده است. تمام هفت ژانر از جمله فلسفی باقی‌اند. تعداد اندک فلسفی در گزارش مشخص شده و تصمیم دربارهٔ استفادهٔ مدل‌سازی آن به مرحلهٔ ارزیابی موکول می‌شود.", "",
            f"تعداد کاربران نامزد {summary['candidate_users']:,} و کاربران باقی‌مانده پس از فیلتر متن {summary['retained_users']:,} است. {summary['removed_after_short_text_filter']:,} کاربر پس از محاسبهٔ مجدد حجم متن کنار رفتند. تعداد کاربران با عضویت در چند گروه {summary['overlapping_memberships']:,} است.", "",
            "| ژانر | کاربران | آموزش | اعتبارسنجی | آزمون | نقدهای همان ژانر پس از فیلتر |", "|---|---:|---:|---:|---:|---:|"]
    for genre in GENRES:
        s = summary["genres"][genre]
        text.append("| " + PERSIAN[genre] + " | " + " | ".join(
            f"{s.get(k, 0):,}" for k in ("users", "train_users", "validation_users", "test_users", "target_genre_reviews")) + " |")
    text += ["", "## ساختار داده و پیشگیری از نشت", "",
             "users.csv یک سطر برای هر کاربر و memberships.csv یک سطر برای هر عضویت ژانری دارد. reviews.jsonl.gz یک سطر برای هر جفت کاربر و کتاب منتخب دارد و عضویت چندگانه باعث تکثیر نقد نمی‌شود. کلید کاربر مستعار است. ژانر کتاب و ژانر گرایش کاربر در دو جدول جدا نگهداری می‌شوند.", "",
             "تقسیم اولیه با هش ثابت در سطح کاربر و نسبت‌های انتظاری ۷۰٪ آموزش، ۱۵٪ اعتبارسنجی و ۱۵٪ آزمون انجام شده است. همهٔ نقدها و عضویت‌های یک فرد در یک بخش قرار دارند. این تقسیم طبقه‌بندی‌شده نیست و تعدادهای واقعی برای هر ژانر گزارش شده‌اند. پارامترهای TF-IDF، مقیاس‌بندی، کاهش بُعد، انتخاب ویژگی و خوشه‌بندی باید بر بخش آموزش برازش شوند. در این مرحله فقط تعدادهای تجمیعی بخش‌های دیگر بررسی شده‌اند.", "",
             "تکرار یک کتاب بین کاربران مختلف ممکن است؛ ارزیابی تعمیم به کتاب جدید نیازمند بررسی همپوشانی کتاب‌ها و تقسیم مناسب در مرحلهٔ مدل‌سازی است. نقدهای سازندهٔ عضویت و نقدهای تحلیل زبان در این نسخه از کل سابقه آمده‌اند. مقایسهٔ نقدهای ژانر مشترک و استفاده از شواهد جدا برای تعیین گرایش باید به‌عنوان بررسی استحکام طراحی شوند.", "",
             "شناسهٔ کتاب و کاربر، شمارهٔ سطر منبع و عضویت ژانری ورودیِ ویژگی زبانی نیستند. امتیاز، زمان و شاخص‌های تعامل برای ممیزی نگه داشته شده‌اند و ورودشان به مدل باید با سؤال تحلیل و زمان دسترس‌پذیری سازگار باشد. تصویر، قیمت و فرادادهٔ محصول در این نسخه معیار ورود نیستند.", "",
             "## کیفیت متن و محدودیت‌ها", "",
             "بدنهٔ متن با حروف بزرگ و کوچک، نشانه‌گذاری و قالب اصلی نگهداری شده تا ویژگی‌های سبکی از بین نروند. توکن با همان الگوی حروف و اعداد ممیزی قبلی شمرده می‌شود. شناسایی زبان، پاک‌سازی HTML و بررسی متن‌های یکسان در حساب‌های متفاوت هنوز انجام نشده‌اند. این مراحل باید پیش از برازش بازنمایی متن تعیین تکلیف شوند. متن‌های پرتِ بلند حذف نشده‌اند و اثرشان باید در EDA بخش آموزش بررسی شود.", "",
             "هر کتاب با parent_asin شناخته می‌شود و نسخه‌های مختلف یک اثر ممکن است جدا شمرده شوند. زمان نقد معادل زمان قطعی مطالعه نیست. پوشش نگاشت و تفاوت فرصت مشاهده محدودیت دارند. سهم ژانر رفتار نقدنویسی را توصیف می‌کند. وزن و تکرار نقدهای هر کاربر در این نسخه حفظ شده و روش تجمیع یا وزن‌دهی در تحلیل بعدی ارزیابی می‌شود.", "",
             "## تطبیق با الزامات فاز دوم", "",
             "| الزام | وضعیت فعلی |", "|---|---|",
             "| آماده‌سازی قابل‌بازاجرا و ثبت قواعد | کد ساخت، قواعد نسخه، منشأ و هش فایل‌ها و شمارش ریزش ثبت شدند |",
             "| مدیریت مقادیر گمشده | شناسه یا متن نامعتبر برای انتخاب نقد پذیرفته نمی‌شود؛ امتیاز و زمان نامعتبر تهی نگه داشته می‌شوند؛ مقدارگذاری جایگزین انجام نشده است |",
             "| مهندسی ویژگی با منطق دامنه‌ای | تعداد، سهم و تداوم برای عضویت ثبت شده‌اند؛ ویژگی‌های زبانی هنوز باید ساخته شوند |",
             "| TF-IDF و ویژگی‌های ساخت‌یافته | ساخت بازنمایی و تصمیم دربارهٔ ویژگی‌های ورودی باقی است |",
             "| کاهش بُعد یا بازنمایی | باقی است |",
             "| خوشه‌بندی نقد یا محصول و تفسیر | باقی است |",
             "| انتخاب پارامتر و پایداری | باقی است |",
             "| گزارش نهایی و تگ phase-2 | پس از تکمیل تحلیل‌ها تهیه می‌شود |", "",
             "## بازتولید و نسخه‌بندی", "", "```powershell", "python src/freeze_analysis_data.py", "```", "",
             "قواعد در data_v1_rules.json و منشأ، هش‌ها و شمارش‌ها در data_v1_manifest.json ثبت‌اند. داده‌های فردی و متن در data/analysis/eda-v1 قرار دارند. خروجی‌های کامل بازنویسی نمی‌شوند؛ تغییر قواعد نیازمند پوشهٔ خروجی تازه است. اجرای ناتمام از اسکن کامل ذخیره‌شده استفاده می‌کند یا اسکن را با درج تکرارپذیر دوباره انجام می‌دهد. برای بازتولید byte-identical باید نسخهٔ Python و zlib ثبت‌شده در manifest نیز یکسان باشد."]
    (report_dir / "data-v1-fa.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=Path("reports/phase2/data_v1_rules.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/eda-v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    output = args.output_dir
    if (output / "manifest.json").exists():
        raise FileExistsError("Snapshot is sealed; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    reviews = Path("data/raw/Books.jsonl.gz")
    profiles = Path("data/user_genre_profiles/user_genre_profiles.csv")
    texts = Path("data/cohort_feasibility/user_text.csv")
    mapping_path = Path("data/genre_coverage/book_genres.csv")
    print("Fingerprinting source files", flush=True)
    sources = (reviews, profiles, texts, mapping_path, args.rules, Path(__file__),
               Path(__file__).with_name("audit.py"), Path(__file__).with_name("user_genre_profiles.py"))
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources}
    inputs = fingerprint(sources)
    memberships = candidate_memberships(profiles, texts, rules)
    print(f"Candidate users | {len(memberships):,}", flush=True)
    connection, state = scan_reviews(reviews, memberships, load_mapping(mapping_path), rules,
                                     output / "staging.sqlite", output / "scan_state.json", inputs)
    summary = export_snapshot(connection, memberships, rules, output)
    connection.close()
    for path, signature in before.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
            raise ValueError(f"Input changed while building snapshot: {path}")
    import platform
    import zlib
    manifest = {"version": rules["version"], "rules": rules, "inputs": inputs,
                "runtime": {"python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION},
                "source_scan": state["counts"], "summary": summary,
                "outputs": fingerprint(tuple(output / name for name in
                                              ("users.csv", "memberships.csv", "reviews.jsonl.gz")))}
    publish_report(manifest, args.report_dir)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Sealed | {output} | {summary['retained_users']:,} users", flush=True)


if __name__ == "__main__":
    main()
