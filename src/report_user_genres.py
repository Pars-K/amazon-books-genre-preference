"""Render aggregate-only figures and a Persian account of the profiling study."""

import argparse
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from user_genre_profiles import COUNT_BINS, GENRES, add_relation, relation_result, count_bin, write_csv


PERSIAN = dict(zip(GENRES, ("فانتزی", "علمی‌تخیلی", "عاشقانه", "معمایی", "تریلر", "فلسفی", "علمی")))
COLORS = ("#2878a0", "#8956a5", "#cc6680", "#b47835", "#386c57", "#5274bd", "#5a8d89")
SHARE_BINS = ("(0,25%)", "[25%,50%)", "[50%,75%)", "[75%,100%]")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#253348", "text.color": "#253348",
    "axes.titleweight": "bold", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def pct(n, denominator):
    return f"{100 * n / denominator:.1f}%" if denominator else "—"


def number(value, decimals=2):
    return f"{value:.{decimals}f}" if value is not None else "—"


def analyze_shares(path, result):
    groups = defaultdict(Counter)
    seen = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            seen += 1
            integers = ("distinct_books", "rated_books", "rating_1", "rating_2", "rating_3", "rating_4", "rating_5")
            row = {key: int(raw[key]) for key in integers}
            for field in ("mean_rating", "mean_rating_minus_user_mean", "mean_rating_minus_other_books"):
                row[field] = float(raw[field]) if raw[field] else None
            share = float(raw["share_mapped_books"])
            share_bucket = SHARE_BINS[0 if share < .25 else 1 if share < .5 else 2 if share < .75 else 3]
            for n_bucket in (count_bin(row["distinct_books"]), "all"):
                add_relation(groups[(raw["genre"], n_bucket, share_bucket, "all")], row)
                if float(raw["mapping_coverage"]) >= .5:
                    add_relation(groups[(raw["genre"], n_bucket, share_bucket, "coverage>=50%")], row)
    expected = sum(st["user_genre_pairs"] for st in result["genres"].values())
    if seen != expected:
        raise ValueError(f"User/genre CSV contains {seen} rows; aggregate summary expects {expected}")
    return [
        {"genre": genre, "book_count_bin": n_bucket, "share_mapped_bin": share_bucket,
         "coverage_filter": coverage, **relation_result(counter)}
        for (genre, n_bucket, share_bucket, coverage), counter in sorted(groups.items())
    ]


def render_figures(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    stats = result["genres"]
    fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
    y = list(range(len(GENRES)))
    for offset, minimum, color in ((-.24, 1, "#a7c6d4"), (0, 3, "#397c9a"), (.24, 5, "#244559")):
        values = [stats[g].get(f"users_at_least_{minimum}_books", 0) for g in GENRES]
        ax.barh([v + offset for v in y], values, height=.22, color=color, label=f"At least {minimum} distinct books")
    ax.set_yticks(y, GENRES)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Users with recorded reviews (log scale)")
    ax.set_title("How much genre-specific evidence exists per user?")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3, frameon=False, fontsize=8)
    fig.savefig(folder / "genre_support.png", dpi=180)
    plt.close(fig)

    rows = {(r["genre"], r["book_count_bin"]): r for r in result["rating_by_count"]
            if r["mapping_coverage_bin"] == "all"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for g, color in zip(GENRES, COLORS):
        for ax, field in zip(axes, ("user_weighted_mean_rating", "mean_rating_minus_other_books")):
            values = [rows.get((g, b), {}).get(field) for b in COUNT_BINS]
            ax.plot(COUNT_BINS, [v if v is not None else float("nan") for v in values],
                    marker="o", linewidth=1.7, markersize=4, label=g, color=color)
    axes[0].set_title("Satisfaction by repeated genre engagement")
    axes[0].set_ylabel("Mean rating per user, then averaged across users")
    axes[1].set_title("Comparison with the user's other books")
    axes[1].set_ylabel("Genre mean rating minus mean rating of other books")
    axes[1].axhline(0, color="#6c7785", linewidth=.8)
    for ax in axes:
        ax.set_xlabel("Distinct reviewed books in the genre")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, loc="best", fontsize=8)
    fig.savefig(folder / "ratings_and_repetition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    lows, middles, highs, continuity = [], [], [], []
    for g in GENRES:
        st = stats[g]
        denominator = st.get("repeated_with_three_ratings", 0)
        low = st.get("repeated_mean_rating_at_most_3", 0)
        high = st.get("repeated_mean_rating_at_least_4", 0)
        lows.append(100 * low / denominator if denominator else 0)
        middles.append(100 * (denominator - low - high) / denominator if denominator else 0)
        highs.append(100 * high / denominator if denominator else 0)
        repeated = st.get("users_at_least_3_books", 0)
        continuity.append(100 * st.get("repeated_span_180_days", 0) / repeated if repeated else 0)
    axes[0].barh(y, lows, color="#c37764", label="Mean <= 3")
    axes[0].barh(y, middles, left=lows, color="#d7b66f", label="3 < mean < 4")
    axes[0].barh(y, highs, left=[a + b for a, b in zip(lows, middles)], color="#609b89", label="Mean >= 4")
    axes[0].set_title("Repeated engagement can have different satisfaction")
    axes[0].set_xlabel("Users with at least 3 rated genre books (%)")
    axes[0].legend(frameon=False, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, -.25))
    axes[1].barh(y, continuity, color="#397c9a")
    axes[1].set_title("Does repeated engagement extend over time?")
    axes[1].set_xlabel("Users with >=3 genre books whose reviews span >=180 days (%)")
    for ax in axes:
        ax.set_yticks(y, GENRES)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=.15)
    fig.savefig(folder / "satisfaction_and_continuity.png", dpi=180)
    plt.close(fig)

    share_rows = {(r["genre"], r["book_count_bin"], r["share_mapped_bin"]): r
                  for r in result["rating_by_share"] if r["coverage_filter"] == "all"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, n_bucket in zip(axes, ("3-4", "5-9")):
        for genre, color in zip(GENRES, COLORS):
            values = [share_rows.get((genre, n_bucket, b), {}).get("mean_rating_minus_other_books")
                      for b in SHARE_BINS]
            ax.plot(range(4), [v if v is not None else float("nan") for v in values],
                    marker="o", linewidth=1.7, markersize=4, color=color, label=genre)
        ax.set_title(f"Users with {n_bucket} distinct books in the genre")
        ax.set_xticks(range(4), ("<25%", "25-49%", "50-74%", "75-100%"))
        ax.set_xlabel("Genre share of the user's mapped books")
        ax.set_ylabel("Genre mean rating minus mean rating of other books")
        ax.axhline(0, color="#6c7785", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Genre concentration and satisfaction, within book-count bands", fontsize=13)
    fig.savefig(folder / "genre_share_and_ratings.png", dpi=180)
    plt.close(fig)


def render_report(result, folder):
    users, quality, stats = result["users"], result["quality"], result["genres"]
    mapped_users = users["users_with_mapped_books"]
    all_users = users["users"]
    one_mapped = result["mapped_user_mapped_book_histogram"].get("1", 0)
    grouped_rows = {(r["genre"], r["book_count_bin"]): r for r in result["rating_by_count"]
                    if r["mapping_coverage_bin"] == "all"}
    differences = {}
    for genre in GENRES:
        first = grouped_rows.get((genre, "1"), {}).get("user_weighted_mean_rating")
        repeated = grouped_rows.get((genre, "10+"), {}).get("user_weighted_mean_rating")
        if first is not None and repeated is not None:
            differences[genre] = repeated - first
    findings = [
        f"- **{pct(one_mapped, mapped_users)}** از کاربران دارای نگاشت، فقط یک کتاب نگاشت‌شده دارند؛ سهم بالا بدون کفایت تعداد مشاهدات قابل تفسیر نیست.",
        f"- در مجموع **{sum(st.get('repeated_mean_rating_at_most_3', 0) for st in stats.values()):,} جفت کاربر و ژانر** با حداقل سه کتاب و سه امتیاز معتبر، میانگین امتیاز حداکثر ۳ دارند. یک فرد می‌تواند در چند ژانر شمرده شود. تکرار و رضایت باید جدا بررسی شوند.",
    ]
    if differences:
        rising = sum(value > 0 for value in differences.values())
        falling = sum(value < 0 for value in differences.values())
        findings.append(f"- میانگین امتیاز گروه دارای ۱۰ کتاب یا بیشتر، در **{rising} ژانر بالاتر** و در **{falling} ژانر پایین‌تر** از گروه تک‌کتاب است. مقایسه بین گروه‌های متفاوت کاربران است و تغییر امتیاز یک فرد در طول زمان را اندازه نمی‌گیرد.")
        for genre in (max(differences, key=differences.get), min(differences, key=differences.get)):
            first = grouped_rows[(genre, "1")]["user_weighted_mean_rating"]
            repeated = grouped_rows[(genre, "10+")]["user_weighted_mean_rating"]
            findings.append(f"- مثال {PERSIAN[genre]}: میانگین گروه تک‌کتاب **{first:.2f}** و گروه دارای حداقل ۱۰ کتاب **{repeated:.2f}** است؛ اندازهٔ گروه‌ها در فایل تجمیعی ثبت شده است.")
    lines = [
        "# بررسی اکتشافی گرایش ژانری و رضایت کاربران", "",
        "این بررسی برای انتخاب تعریف مناسب ترجیح ژانر انجام شده است. هیچ برچسب «علاقه‌مند» یا «بی‌علاقه» تولید نشده است.", "",
        "## یافته‌های اصلی", "", *findings, "",
        "## دامنه و روش", "",
        f"- تمام **{quality['raw_rows']:,} سطر** آرشیو نقدها پردازش شد؛ نمونه‌گیری انجام نشد.",
        f"- **{all_users:,} کاربر** در کل آرشیو و **{mapped_users:,} کاربر** با دست‌کم یک کتاب دارای نگاشت معتبر مشاهده شد.",
        f"- **{users['distinct_user_book_pairs']:,} جفت متمایز کاربر و کتاب** وجود داشت؛ **{users['mapped_distinct_user_book_pairs']:,} جفت** به هفت گروه منتخب نگاشت شدند.",
        f"- فاصلهٔ تعداد سطرهای دارای شناسهٔ معتبر کاربر تا جفت‌های متمایز، **{users['extra_rows_beyond_distinct_books']:,} سطر** است. این اختلاف، رکوردهای یکسان و نقدهای متفاوت از یک جفت را با هم شامل می‌شود؛ شمارش صرفاً تکرار کامل نیست.",
        "- نقدهای هر کاربر با پارتیشن‌بندی هش در کل فایل تجمیع شدند؛ پیوسته‌بودن سطرهای یک کاربر فرض نشده است.",
        "- شناسهٔ کتاب همان `parent_asin` است. این شناسه تضمین نمی‌کند نسخه‌های مختلف یک اثر ادبی حتماً یک کتاب شمرده شوند.",
        "- از هر جفت کاربر و کتاب، یک کتاب شمرده شد. برای امتیاز، آخرین امتیاز معتبر با زمان معتبر انتخاب شد؛ در تساوی زمان، امتیاز پایین‌تر و در نبود زمان معتبر، امتیاز موجود با همان قاعدهٔ رفع تساوی نگه داشته شد.",
        "- برای تداوم، زمان اولین نقد معتبر هر کتاب مبناست. نقد مجدد همان کتاب در فصل دیگری، شواهد دنبال‌کردن کتاب تازه ایجاد نمی‌کند. فصل‌ها سه‌ماههٔ تقویمی میلادی و زمان‌ها UTC هستند.",
        "- سهم ژانر هم نسبت به کل کتاب‌های دارای نگاشت معتبر و هم نسبت به تمام کتاب‌های کاربر ثبت شد. کتاب‌های مبهم یا بیرون از نگاشت از مخرج کل سابقه حذف نشده‌اند.",
        "- رضایت با میانگین امتیاز، سهم امتیازهای ۴ و ۵ و سهم امتیازهای ۱ و ۲ توصیف شد. امتیاز مثبت شرط شمارش گرایش نیست.",
        "- برای مقایسهٔ رضایت، اختلاف میانگین امتیاز ژانر با میانگین سایر کتاب‌های همان کاربر نیز محاسبه شد. کاربرانی که کتاب امتیازدار دیگری ندارند از این مقایسه کنار گذاشته می‌شوند.", "",
        "## مقدار شواهد موجود", "",
        "| ژانر | کاربران با ≥۱ کتاب | با ≥۳ کتاب | با ≥۵ کتاب | سهم ژانر از جفت‌های دارای نگاشت |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in GENRES:
        st = stats[g]
        lines.append(f"| {PERSIAN[g]} | {st.get('users_at_least_1_books', 0):,} | {st.get('users_at_least_3_books', 0):,} | {st.get('users_at_least_5_books', 0):,} | {100 * (st['share_of_all_mapped_user_book_pairs'] or 0):.1f}% |")
    lines.extend([
        "", "![مقدار شواهد ژانری](genre_support.png)", "",
        f"از کاربران دارای نگاشت، **{one_mapped:,} نفر ({pct(one_mapped, mapped_users)})** فقط یک کتاب دارای نگاشت دارند. سهم ۱۰۰٪ یک ژانر برای این کاربران، به‌تنهایی شواهد کافی ترجیح پایدار نیست.",
        f"**{users['users_with_multiple_mapped_genres']:,} کاربر** در بیش از یک گروه منتخب نقد کتاب دارند؛ این عدد، تعداد کاربران با چند علاقهٔ تأییدشده نیست.", "",
        "پوشش نگاشت در سابقهٔ کاربران دارای دست‌کم یک کتاب نگاشت‌شده:", "",
        "| سهم کتاب‌های قابل‌نگاشت از تمام کتاب‌های کاربر | تعداد کاربران | سهم کاربران دارای نگاشت |",
        "|---|---:|---:|",
    ])
    for bucket in ("<25%", "25-49%", "50-74%", "75-100%"):
        value = result["mapping_coverage_user_bins"].get(bucket, 0)
        lines.append(f"| {bucket} | {value:,} | {pct(value, mapped_users)} |")
    lines.extend([
        "", "## رابطهٔ تکرار و رضایت", "",
        "در جدول زیر، ابتدا میانگین امتیاز هر کاربر در ژانر محاسبه و سپس این میانگین‌ها با وزن برابر بین کاربران میانگین‌گیری شده‌اند. همهٔ ژانرها با گروه‌های فعالیت یکسان مقایسه نشده‌اند؛ جدول توصیفی است و اثر علّی تکرار بر رضایت را نشان نمی‌دهد.", "",
        "| ژانر | یک کتاب | دو کتاب | ۳ تا ۴ کتاب | ۵ تا ۹ کتاب | ۱۰ کتاب یا بیشتر |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for g in GENRES:
        values = [number(grouped_rows.get((g, b), {}).get("user_weighted_mean_rating")) for b in COUNT_BINS]
        lines.append("| " + " | ".join((PERSIAN[g], *values)) + " |")
    lines.extend([
        "", "![امتیاز و تکرار](ratings_and_repetition.png)", "",
        "پنل راست، تفاوت امتیاز ژانر با سایر کتاب‌های همان فرد را نشان می‌دهد. این مقایسه، بخشی از تفاوت افراد در نمره‌دهی را لحاظ می‌کند، اما کنترل کامل تفاوت کاربران، محبوبیت، مواجهه یا انتخاب کتاب‌ها نیست. مخرج هر گروه در `rating_by_count.csv` ثبت شده است.", "",
        "| ژانر | یک کتاب با امتیاز ۴ یا ۵ | ≥۳ کتاب با ≥۳ امتیاز معتبر | از این گروه، میانگین ≤۳ | از این گروه، میانگین ≥۴ |",
        "|---|---:|---:|---:|---:|",
    ])
    for g in GENRES:
        st = stats[g]
        denominator = st.get("repeated_with_three_ratings", 0)
        low, high = st.get("repeated_mean_rating_at_most_3", 0), st.get("repeated_mean_rating_at_least_4", 0)
        lines.append(f"| {PERSIAN[g]} | {st.get('single_book_positive_rating', 0):,} | {denominator:,} | {low:,} ({pct(low, denominator)}) | {high:,} ({pct(high, denominator)}) |")
    lines.extend([
        "", "ستون اول نمونه‌هایی از رضایت ثبت‌شده با تنها یک کتاب در ژانر را می‌شمارد. ستون‌های بعدی نشان می‌دهند تکرار می‌تواند با سطوح متفاوت رضایت همراه باشد. این گروه‌ها به‌عنوان حقیقت مبنای علاقه یا بی‌علاقگی اعتبارسنجی نشده‌اند.", "",
        "## رابطهٔ سهم ژانر و رضایت", "",
        "![سهم ژانر و رضایت](genre_share_and_ratings.png)", "",
        "در این شکل، رابطهٔ سهم ژانر با امتیازدهی نسبی در دو بازهٔ تعداد کتاب بررسی شده است تا مقایسه صرفاً میان تک‌مشاهده‌ها و کاربران بسیار فعال نباشد. مخرج سهم، کتاب‌های دارای نگاشت معتبر است و امتیاز نسبی با تمام کتاب‌های امتیازدار دیگر فرد مقایسه می‌شود. تعداد کاربران و مقایسهٔ حساسیت با شرط پوشش نگاشت حداقل ۵۰٪ در `rating_by_share.csv` آمده است. تفاوت ترکیب کاربران و فرصت مواجهه همچنان باقی است.", "",
        "## تداوم زمانی و حساسیت تعریف", "",
        "| ژانر | کاربران با ≥۳ کتاب | از این گروه، ≥۲ فصل فعال | از این گروه، ≥۳ کتاب تاریخ‌دار با فاصلهٔ ≥۱۸۰ روز |",
        "|---|---:|---:|---:|",
    ])
    for g in GENRES:
        st = stats[g]
        denominator = st.get("users_at_least_3_books", 0)
        multi, span = st.get("repeated_multi_quarter", 0), st.get("repeated_span_180_days", 0)
        lines.append(f"| {PERSIAN[g]} | {denominator:,} | {multi:,} ({pct(multi, denominator)}) | {span:,} ({pct(span, denominator)}) |")
    lines.extend([
        "", "![رضایت و تداوم](satisfaction_and_continuity.png)", "",
        "حضور در دو فصل می‌تواند با فاصلهٔ یک روز در مرز فصل‌ها رخ دهد؛ بنابراین تعداد فصل و فاصلهٔ زمانی جدا گزارش شده‌اند.",
        "شبکهٔ تشخیصی شامل حداقل ۲، ۳، ۵ یا ۱۰ کتاب، سهم حداقل ۲۵٪، ۵۰٪ یا ۷۵٪ از کتاب‌های قابل‌نگاشت و حضور در حداقل ۱، ۲ یا ۴ فصل است. شمارش‌ها یک‌بار برای همهٔ کاربران دارای نگاشت و یک‌بار با شرط پوشش نگاشت حداقل ۵۰٪ ثبت شده‌اند. این آستانه‌ها سناریوهای توصیفی‌اند و هیچ‌کدام تعریف نهایی ترجیح نیستند.", "",
        "## نتیجه برای طراحی پروژه", "",
        "- تعداد کتاب، سهم ژانر، تداوم زمانی و رضایت باید به‌صورت ابعاد جدا باقی بمانند تا رابطهٔ آن‌ها قابل بررسی باشد.",
        "- سهم زیاد با یک مشاهده یا پوشش ضعیف نگاشت باید همراه با مقدار شواهد گزارش شود.",
        "- میانگین امتیاز بالا برای یک کتاب و مراجعهٔ مکرر به یک ژانر، دو الگوی متفاوت مشاهده‌شده‌اند. هیچ‌کدام به‌تنهایی علاقهٔ واقعی را تأیید نمی‌کند.",
        "- پیش از انتخاب برچسب، لازم است کفایت متن ورودی و شواهد ژانری در بازهٔ مستقل بعدی هم‌زمان بررسی شود. این مطالعه، شمارش نمونهٔ نهایی پیش‌بینی نیست.", "",
        "## محدودیت‌ها و حدود استفاده", "",
        "این آرشیو همهٔ خریدها، مطالعه‌ها یا مواجهه‌های کاربر را ثبت نمی‌کند. زمان نقد هم لزوماً زمان مطالعه نیست. نبود نقد از یک ژانر شاهد بی‌علاقگی نیست. شواهدی از علاقهٔ واقعی برای سنجش اعتبار معیارهای پیشنهادی نداریم؛ این تحلیل روابط میان رفتار نقدنویسی و رضایت ثبت‌شده را توصیف می‌کند.", "",
        "سهم جمعیتی ژانرها، توصیف نقدهای موجود است و احتمال مواجههٔ هر فرد با ژانر را مشخص نمی‌کند. نویسنده و مجموعه در این اجرای اولیه تحلیل نشده‌اند؛ تمرکز روی یک مجموعه می‌تواند بخشی از تکرار را توضیح دهد.", "",
        "طول فرصت مشاهده بین کاربران برابر نیست. کاربران تازه‌وارد یا افرادی که نقدهایشان نزدیک انتهای آرشیو ثبت شده است، فرصت کمتری برای ایجاد سابقهٔ طولانی دارند. جمع‌بندی کل سابقه می‌تواند تغییر گرایش فرد در طول زمان را نیز پنهان کند؛ بررسی بازه‌های زمانی مستقل در مرحلهٔ بعد لازم است.", "",
        "این بررسی کل بازهٔ آرشیو را پوشش می‌دهد. برای مدل پیش‌بینی، باید بازهٔ ویژگی و بازهٔ هدف و کاربران آموزش/اعتبارسنجی/آزمون مستقل تعریف شوند؛ قواعد و پارامترها پیش از ارزیابی نهایی تثبیت شوند. نمایهٔ کل‌سابقه نباید مستقیماً به‌عنوان ویژگی پیش‌بینی آینده استفاده شود.", "",
        "هر کاربر ممکن است در چند ردیف ژانری حضور داشته باشد؛ ردیف‌های ژانری مستقل نیستند. آزمون معناداری، ادعای علّی یا برآورد اعتبار تشخیص علاقه انجام نشده است.", "",
        "## بازاجرا و فایل‌ها", "",
        "```powershell", "python src/user_genre_profiles.py --expected-rows 29475453",
        "python src/report_user_genres.py", "```", "",
        "اجرای اول از مسیر خروجی خالی استفاده می‌کند و فایل‌های موجود را بازنویسی نمی‌کند. برای اجرای مجدد، `--output-dir` و `--report-dir` تازه تعیین کنید و مسیر خلاصه را با `--summary` به اسکریپت گزارش بدهید.", "",
        "- `data/user_genre_profiles/user_profiles.csv`: نمایهٔ کاربران دارای حداقل یک کتاب نگاشت‌شده، شامل اندازهٔ کل سابقه و پوشش نگاشت.",
        "- `data/user_genre_profiles/user_genre_profiles.csv`: تعداد، سهم، رضایت و تداوم برای هر جفت کاربر و ژانر مشاهده‌شده.",
        "- شناسه‌ها با هش بدون نمک جایگزین شده‌اند؛ مستعارند و تضمین ناشناس‌بودن ندارند. فایل‌های فردی در پوشهٔ محلیِ نادیده‌گرفته‌شدهٔ `data/` باقی می‌مانند.",
        "- `summary.json`، `rating_by_count.csv`، `rating_by_share.csv` و `diagnostic_threshold_grid.csv`: فقط نتایج تجمیعی، همراه با شکل‌های همین گزارش.",
        "- متن خام نقدها در خروجی فردی یا گزارش ذخیره نشده است. تعداد کتاب‌های دارای متن غیرخالی، سنجش کفایت حجم متن نیست.", "",
        "خطاهای ثبت‌شدهٔ ورودی:", "", "```json", json.dumps(quality, indent=2), "```", "",
        "تفسیر فراوانی به‌عنوان میزان شواهد، با الهام از [Hu, Koren & Volinsky (2008)](https://yifanhu.net/PUB/cf.pdf) است. تعریف ژانری این پروژه، مستقیماً از آن مقاله اخذ یا اعتبارسنجی نشده است.", "",
    ])
    (folder / "report-fa.md").write_text("\n".join(lines), encoding="utf-8")


def validate(result):
    users = result["users"]
    all_hist = result["all_user_distinct_book_histogram"]
    mapped_hist = result["mapped_user_mapped_book_histogram"]
    assert sum(all_hist.values()) == users["users"]
    assert sum(int(k) * n for k, n in all_hist.items()) == users["distinct_user_book_pairs"]
    assert sum(mapped_hist.values()) == users["users_with_mapped_books"]
    assert sum(int(k) * n for k, n in mapped_hist.items()) == users["mapped_distinct_user_book_pairs"]
    assert sum(g["distinct_books"] for g in result["genres"].values()) == users["mapped_distinct_user_book_pairs"]
    assert sum(result["mapping_coverage_user_bins"].values()) == users["users_with_mapped_books"]
    for stats in result["genres"].values():
        assert stats.get("users_at_least_5_books", 0) <= stats.get("users_at_least_3_books", 0) <= stats.get("users_at_least_1_books", 0)
        assert stats.get("repeated_mean_rating_at_most_3", 0) + stats.get("repeated_mean_rating_at_least_4", 0) <= stats.get("repeated_with_three_ratings", 0)
    grid = {(r["genre"], r["minimum_books"], r["minimum_share_mapped"],
             r["minimum_active_quarters"], r["coverage_filter"]): r["users"]
            for r in result["diagnostic_threshold_grid"]}
    for (genre, n, share, quarters, coverage), count in grid.items():
        assert count <= result["genres"][genre].get(f"users_at_least_{n}_books", 0)
        assert count <= grid[(genre, n, share, quarters, "all")]
        for stricter_n in (2, 3, 5, 10):
            if stricter_n > n:
                assert grid[(genre, stricter_n, share, quarters, coverage)] <= count
        for stricter_share in (.25, .5, .75):
            if stricter_share > share:
                assert grid[(genre, n, stricter_share, quarters, coverage)] <= count
        for stricter_quarters in (1, 2, 4):
            if stricter_quarters > quarters:
                assert grid[(genre, n, share, stricter_quarters, coverage)] <= count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/genre_preference/summary.json"))
    parser.add_argument("--profiles", type=Path, default=Path("data/user_genre_profiles/user_genre_profiles.csv"))
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    validate(result)
    result["rating_by_share"] = analyze_shares(args.profiles, result)
    write_csv(args.summary.parent / "rating_by_share.csv", result["rating_by_share"])
    partial = args.summary.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    partial.replace(args.summary)
    render_figures(result, args.summary.parent)
    render_report(result, args.summary.parent)
    print(f"Validated aggregates; report and four figures written to {args.summary.parent}")


if __name__ == "__main__":
    main()
