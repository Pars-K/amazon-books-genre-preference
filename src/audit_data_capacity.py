"""Summarize existing cohort capacity without changing eligibility or reading raw text."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "Fantasy": "فانتزی",
    "Science Fiction": "علمی‌تخیلی",
    "Romance": "عاشقانه",
    "Mystery": "معمایی",
    "Thriller": "تریلر",
    "Philosophy": "فلسفه",
    "Science": "علمی",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    sensitivity = ROOT / "reports/cohort_feasibility/cohort_sensitivity.csv"
    frozen = ROOT / "reports/phase2/data_v1_counts.csv"
    output = ROOT / "reports/phase2/adequacy"
    output.mkdir(parents=True, exist_ok=True)
    current = {row["genre"]: int(row["users"]) for row in read_csv(frozen)}
    capacity = {}
    for row in read_csv(sensitivity):
        if (
            row["share_denominator"] == "all_books"
            and int(row["minimum_books"]) == 3
            and float(row["minimum_share"]) == 0.5
            and float(row["minimum_coverage"]) == 0.5
            and int(row["minimum_active_quarters"]) == 2
            and int(row["minimum_span_days"]) == 180
            and row["text_scope"] in {"none", "all_history"}
        ):
            key = (row["genre"], row["text_scope"], int(row["minimum_tokens"]))
            if key in capacity:
                raise ValueError(f"Duplicate capacity cell: {key}")
            capacity[key] = int(row["users"])
    records = []
    for genre in NAMES:
        values = [capacity[(genre, "all_history", n)] for n in (500, 1000, 2000)]
        no_text = capacity[(genre, "none", 0)]
        if not no_text >= values[0] >= values[1] >= values[2]:
            raise ValueError(f"Inconsistent nested thresholds for {genre}")
        if current[genre] > values[1]:
            raise ValueError(f"Frozen count exceeds preliminary candidates for {genre}")
        records.append({
            "genre": genre,
            "frozen_users": current[genre],
            "behavior_only_candidates": no_text,
            "preliminary_500_token_candidates": values[0],
            "preliminary_1000_token_candidates": values[1],
            "preliminary_2000_token_candidates": values[2],
        })
    with (output / "capacity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# کفایت داده و ظرفیت نمونه‌های موجود",
        "",
        "این بررسی در ۱۹ نوامبر ۲۰۲۶ صرفاً از جدول‌های موجود استفاده کرد. هیچ کاربر، شرط انتخاب، تقسیم داده یا نتیجهٔ قبلی تغییر نکرد.",
        "",
        "## نتیجهٔ تحلیلی",
        "",
        "داده برای ادامهٔ تحلیل اکتشافی و آزمودن چند فرضیهٔ مشخص وجود دارد. کفایت داده برای همهٔ ژانرها، شاخص‌ها و مقایسه‌ها یکسان نیست. تعداد زیاد نقد به معنای تعداد زیاد نویسندهٔ مستقل نیست. ناکافی‌بودن شواهد نیز به‌تنهایی کمبود حجم نمونه را اثبات نمی‌کند. اثر ممکن است کوچک یا ناموجود باشد، اندازه‌گیری ممکن است پرخطا باشد و موضوع کتاب ممکن است رابطهٔ ظاهری ایجاد کند.",
        "",
        "در مقایسهٔ محدودِ نقد کتاب‌های تریلر در کاربران جداگانه، ۱۳۵ عاشقانه‌خوان و ۲۲ علمی‌تخیلی‌خوان حضور داشتند. این محدودیت متعلق به همان مقایسه است. نسخهٔ تثبیت‌شده در مجموع ۱۷٬۰۲۲ کاربر دارد. هم‌پوشانی اندک عضویت ژانری باعث می‌شود جمع سطرهای جدول ۱۷٬۰۲۸ باشد.",
        "",
        "## ظرفیت با تعریف رفتاری ثابت",
        "",
        "در تمام ستون‌های نامزدها، حداقل سه کتاب ژانر، سهم ژانر از کل کتاب‌های نقدشده حداقل ۵۰ درصد، پوشش دسته‌بندی حداقل ۵۰ درصد، حداقل سه کتاب ژانر با تاریخ معتبر، حضور در حداقل دو فصل تقویمی و فاصلهٔ حداقل ۱۸۰ روز ثابت است. این سهم برآورد گرایش از سابقهٔ نقد است و تمام کتاب‌های خوانده‌شدهٔ شخص را پوشش نمی‌دهد.",
        "",
        "| ژانر | کاربران نسخهٔ فعلی | نامزد با حداقل ۵۰۰ توکن | نامزد با حداقل ۱۰۰۰ توکن | پیش از شرط مقدار متن |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in records:
        values = [row[k] for k in (
            "frozen_users", "preliminary_500_token_candidates",
            "preliminary_1000_token_candidates", "behavior_only_candidates",
        )]
        lines.append("| " + NAMES[row["genre"]] + " | " + " | ".join(f"{n:,}" for n in values) + " |")
    lines += [
        "",
        "توکن در این جدول شمارشِ واژه‌مانندِ روش اولیه است. نامزدهای ۵۰۰ و ۱۰۰۰ توکن علاوه بر مجموع متن، حداقل سه نقدِ ۲۰توکنی دارند. در جدول اولیه مجموع متن شامل نقدهای کوتاه‌تر نیز بود. نسخهٔ تثبیت‌شده ابتدا نقدهای زیر ۲۰ توکن را کنار می‌گذارد و سپس حداقل ۱۰۰۰ توکن را دوباره کنترل می‌کند. بنابراین ستون‌های نامزد، برآورد ظرفیت پیش از کنترل نهایی‌اند و با کاربران نهایی یا تعداد نقدهای بلندِ مناسب برای هر شاخص برابر نیستند. ستون پیش از شرط مقدار متن تضمین کافی‌بودن متن ندارد. این شمارها در کل آرشیو هستند و به معنای قابل‌استفاده‌بودن بخش آزمون نهایی برای توسعه نیستند.",
        "",
        "## اقدام بعدی و معیار تصمیم",
        "",
        "۱. ابتدا رابطهٔ مقدار متن و ثبات هر شاخص را بررسی کنیم. بازه‌های مقدار متن و معیار ثبات پیش از مقایسهٔ ژانرها تعیین شوند. نرخ‌های مبتنی بر شمارش با مقدار متن و فراوانی رخداد گزارش شوند. ویژگی‌های کم‌وقوع یا تفسیری صرفاً با زیادکردن تعداد کاربران معتبر نمی‌شوند.",
        "",
        "۲. امکان گسترش نمونه از همین آرشیو بررسی شود. شرط ۱۰۰۰ توکن یک انتخاب اجرایی برای تحلیل قبلی بوده و آستانهٔ اثبات‌شدهٔ کفایت همهٔ ویژگی‌ها نیست. آستانهٔ ۵۰۰ توکن فعلاً سناریوی امکان‌سنجی است. تغییر آن باید بر مبنای ثبات اندازه‌گیری و تناسب با هدفِ استنباط از متن کاربر جدید باشد. تعریف گرایش برای رسیدن به نتیجهٔ معنادار تغییر نکند. در تحلیل گسترش‌یافته نیز کتابِ متن ورودی از شواهد تعیین گرایش حذف شود.",
        "",
        "۳. مقایسه‌های اصلی محدود و از پیش مشخص شوند. هنگام امکان‌پذیری، طول متن و ژانر کتاب کنترل شوند و وابستگی نقدهای یک نویسنده یا یک کتاب لحاظ شود. اندازهٔ اثر و بازهٔ عدم‌قطعیت گزارش شوند. فهرست‌های واژگانی اکتشافی در نویسندگان جداگانه سنجیده شوند. استفاده از واژه‌های موضوعی به‌تنهایی شاهد سبک نویسنده نیست.",
        "",
        "۴. برای هر مقایسهٔ اصلی، کوچک‌ترین اختلافِ مفید یا دقت موردنیاز از قبل تعیین شود و سپس حجم نمونهٔ لازم برآورد شود. افزایش نمونه تا عبور از آستانهٔ معناداری معیار توقف مناسبی نیست. نبود تفاوت معنادار هم به‌تنهایی یکسان‌بودن گروه‌ها را ثابت نمی‌کند.",
        "",
        "۵. فلسفه با ۱۹ کاربر فعلی و ۷۹ نامزدِ صرفاً رفتاری، محدودیت جدی دارد. تا زمان دستیابی به شواهد کافی، نتیجه‌های آن توصیفی بمانند. اگر ظرفیت پس از کنترل کیفیت و همسان‌سازی کافی نبود، گردآوری دادهٔ سازگار یا محدودکردن دامنهٔ ادعا بررسی شود. مبنای این تصمیم تعداد نقد خام نیست.",
        "",
        "۶. جدایی ضعیف خوشه‌ها جداگانه گزارش شود. تعداد بیشتر مشاهده، وجود گروه‌های طبیعی یا تعداد برابر با ژانرها را تضمین نمی‌کند. برای شاخص‌های معناییِ مبتنی بر واژه‌نامه نیز ارزیابی مستقلِ درستی تشخیص لازم است.",
        "",
        "بخش اعتبارسنجی در ممیزی قبلی برای توسعهٔ تحلیل مصرف شده است. بخش آزمون نهایی برای این تصمیم‌ها باز نمی‌شود. این بررسی به معنی بازطراحی یا آموزش مدلِ فاز سوم نیست.",
        "",
        "## بازتولید",
        "",
        "از ریشهٔ مخزن: `python src/audit_data_capacity.py`. ورودی‌ها، خروجی و کد در `manifest.json` هش شده‌اند. این برنامه فقط جدول‌های تجمیعی قبلی را می‌خواند.",
        "",
    ]
    (output / "report-fa.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [sensitivity, frozen, Path(__file__), output / "capacity.csv", output / "report-fa.md"]
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved capacity for {len(records)} genres, frozen unique users: 17022")


if __name__ == "__main__":
    main()
