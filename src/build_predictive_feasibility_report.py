"""Build the decision report and figures without selecting additional models."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from predictive_feasibility import LABELS, OUT, PUB, ROOT, read_csv


def main():
    rows = read_csv(PUB / "metrics.csv")
    table = {(r["scheme"], r["model"]): r for r in rows}
    cv = {r["model"]: r for r in read_csv(PUB / "common_cv_metrics.csv")}
    decision = json.loads((PUB / "decision.json").read_text())
    support = json.loads((PUB / "common_support.json").read_text())
    similarity = json.loads((PUB / "common_similarity.json").read_text())
    comparisons = read_csv(PUB / "paired_intervals.csv")
    names = {"majority": "حدس گروه پرتکرار", "prior_random": "حدس تصادفی متناسب با فراوانی",
             "style": "شاخص‌های نوشتاری", "words": "واژگان", "combined": "ترکیب نوشتار و واژگان",
             "context": "ژانر کتاب و طول متن", "context_combined": "زمینهٔ کتاب همراه متن",
             "naive_bayes": "مدل واژگانی نایو بیز"}
    order = ["majority", "prior_random", "style", "words", "combined", "naive_bayes", "context", "context_combined"]
    lines = [
        "# آیا پیش‌بینی گرایش ژانری امکان‌پذیر است؟",
        "",
        "تاریخ بررسی: ۱۹ نوامبر ۲۰۲۶. طرح اصلی پیش از برازش مدل‌ها در کامیت `88f4301` ثبت شد. این بررسی برای تصمیم دربارهٔ مسیر پروژه است و ارزیابی نهاییِ محصول یا تکمیل فاز سوم نیست.",
        "",
        "یادداشتِ مرحلهٔ بعد: این گزارش تصمیمِ بررسی اولیه را ثبت می‌کند. پس از پیشنهاد کاربر و [بررسی سه‌ژانری](../feasibility-three/report-fa.md)، دامنهٔ جاری به معمایی، تریلر و عاشقانه تغییر کرد. [هدف جاری](../../CURRENT_GOAL.md) مرجع کار بعدی است. عددهای این گزارش تغییری نکرده‌اند.",
        "",
        "## نتیجهٔ تصمیم",
        "",
        "نتیجهٔ بررسی کاملاً منفی نبود. مدل اصلی برای هدف کلیِ هفت‌ژانری از معیار ازپیش‌ثبت‌شده عبور نکرد. افزودن متن به اطلاعات ژانر کتاب و طول متن نیز افزایشِ موردنیاز را در آن آزمایش ایجاد نکرد. با این حال یک مدل واژگانیِ ازقبل‌مشخص‌شده عملکرد بهتری داشت و در مقایسهٔ محدودِ دو گرایش با ژانر کتاب مشترک، نشانهٔ پیش‌بینی روشن و قابل تکرار مشاهده شد. بنابراین نتیجهٔ «هیچ رابطه‌ای وجود ندارد» با داده سازگار نیست.",
        "",
        "در پایان این بررسی اولیه، تمایز گرایش عاشقانه و علمی‌تخیلی از متن نقد یک کتاب تریلر به‌عنوان دامنهٔ محدودِ قابل‌دفاع انتخاب شد. موفقیت در این دامنه، هدفِ تشخیص همهٔ گرایش‌ها از نوشتهٔ هر کاربر دلخواه را تأیید نمی‌کند. [اصلاحیهٔ تاریخیِ دوگروهی](../../OBJECTIVE_AMENDMENT.md) همان تصمیم را ثبت می‌کند. پیشنهاد بعدی کاربر برای سه ژانر جداگانه ارزیابی شده و دامنهٔ جاری در سندِ هدف جاری آمده است. سؤال‌های اصلی و PDF فاز اول دست‌نخورده باقی می‌مانند.",
        "",
        "## داده و جلوگیری از آمیختگی",
        "",
        "ارزیابی کلی شامل ۱۰٬۹۶۸ نویسنده با یک نقد از هر فرد و گرایش تک‌گروهی است. گرایش پس از حذف تمام سابقهٔ کتاب ورودی از شواهد رفتاری دوباره محاسبه شده بود. سه بخش ارزیابی اصلی عنوان کتاب مشترک ندارند و نویسندگان نیز مشترک نیستند. در بررسی حساسیت، سه بخش با جدایی نویسنده و توزیع تقریباً متوازنِ گرایش‌ها ساخته شد. برازش واژگان، IDF، جایگزینی مقدار ناموجود و مقیاس‌بندی فقط از آموزش هر بخش استفاده کرد. انتخاب ضریب منظم‌سازی نیز در یک بخش داخلی با عنوان کتاب جدا انجام شد.",
        "",
        f"در مقایسهٔ دوگروهی، از ۷۶۸ نویسندهٔ توسعه، پس از حذف عنوان‌ها و متن‌های مشترک با ارزیابی، {support['retained_train']} نفر برای آموزش باقی ماندند. ارزیابی جدا شامل ۱۳۵ عاشقانه‌خوان و ۲۲ علمی‌تخیلی‌خوان روی ۱۴۷ عنوان کتاب بود. همهٔ نقدها بر اساس دسته‌بندی ثبت‌شده دربارهٔ کتاب تریلرند. وجود ژانرهای فرعی یا ناقص‌بودن دسته‌بندی آمازون همچنان ممکن است.",
        "",
        "وزن‌دهی به ژانرها، تکرار مصنوعی نمونه و حذف ژانر ضعیف برای بهبود نتیجه انجام نشد. اطلاعات امتیاز، شناسهٔ کاربر، شناسهٔ کتاب و تاریخچهٔ مطالعه ورودی مدل زبانی نبودند. ژانر کتاب و طول متن فقط در مدل تشخیصیِ زمینه استفاده شدند. شاخص‌های واژه‌نامه‌ای با همان تعریف قبلی وارد شدند و اعتبار معنایی تازه‌ای برای آن‌ها فرض نشده است.",
        "",
        "## معیارها چه می‌گویند؟",
        "",
        "Macro-F1 میانگین F1 گروه‌هاست و به هر گروه سهم برابر در سنجش می‌دهد. Balanced accuracy میانگین سهم افراد درست‌شناسایی‌شده در هر گروه است. این برابری در ارزیابی با وزن‌دادن به نمونه‌ها هنگام آموزش فرق دارد. دقت کلی در دادهٔ نامتوازن می‌تواند گمراه‌کننده باشد: در مسئلهٔ هفت‌گروهی، حدس همیشگی تریلر حدود ۷۵ درصد دقت کلی دارد.",
        "",
        "## نتیجهٔ هفت‌گروهی، نویسنده و عنوان کتاب جدا",
        "",
        "| روش | Macro-F1 | میانگین بازیابی گروه‌ها | دقت کلی |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        r = table[("book", name)]
        lines.append(f"| {names[name]} | {float(r['macro_f1']):.3f} | {float(r['balanced_accuracy']):.3f} | {float(r['accuracy']):.3f} |")
    r = table[("book", "combined")]
    six = table[("book_six_supported", "combined")]
    lines += [
        "",
        f"بازهٔ ۹۵ درصدی Macro-F1 مدل ترکیبی از {float(r['macro_f1_low']):.3f} تا {float(r['macro_f1_high']):.3f} بود. حتی در گزارش تکمیلیِ شش گروهِ دارای نمونهٔ بیشتر، این مدل Macro-F1 برابر {float(six['macro_f1']):.3f} و میانگین بازیابی {float(six['balanced_accuracy']):.3f} داشت. این محاسبه با کنارگذاشتن کاربران فلسفی از ارزیابیِ همان پیش‌بینی‌ها انجام شد و یک مدل شش‌کلاسهٔ تازه نیست.",
        "",
        "مدل نایو بیز از ابتدا به‌عنوان الگوریتم ثانویه ثبت شده بود و عملکرد آن را نیز کامل گزارش کردیم. بهترشدن آن نسبت به مدل اصلی نشان می‌دهد نتیجه به نمایش متن و روش یادگیری وابسته است. این نتیجهٔ ثانویه جایگزین معیار اصلیِ ثبت‌شده نمی‌شود و مجوز نتیجه‌گیری دربارهٔ نبود تمام روابط هم نیست. گزارش هر گروه در `per_genre.csv` و تمام تقسیم‌ها در `fold_metrics.csv` موجود است.",
        "",
        "## نتیجهٔ دوگروهی با ژانر کتاب مشترک",
        "",
        "| روش | Macro-F1 | میانگین بازیابی دو گروه | دقت کلی |",
        "|---|---:|---:|---:|",
    ]
    for name in ["majority", "prior_random", "style", "words", "combined", "naive_bayes"]:
        r = table[("common", name)]
        lines.append(f"| {names[name]} | {float(r['macro_f1']):.3f} | {float(r['balanced_accuracy']):.3f} | {float(r['accuracy']):.3f} |")
    r = table[("common", "combined")]
    lines += [
        "",
        f"برای مدل ترکیبی، بازهٔ Macro-F1 از {float(r['macro_f1_low']):.3f} تا {float(r['macro_f1_high']):.3f} و بازهٔ میانگین بازیابی از {float(r['balanced_accuracy_low']):.3f} تا {float(r['balanced_accuracy_high']):.3f} بود. از ۲۲ علمی‌تخیلی‌خوان، ۱۴ نفر و از ۱۳۵ عاشقانه‌خوان، ۱۳۲ نفر درست شناسایی شدند. بنابراین عدد بالای دقت صرفاً حاصل حدس همیشگی گروه بزرگ‌تر نیست. با این حال تعداد علمی‌تخیلی‌خوان‌ها برای تخمین دقیق خطا همچنان کم است.",
        "",
        "مدل صرفاً مبتنی بر طول متن همان نتیجهٔ حدس گروه بزرگ‌تر را داشت. افزایش مدل ترکیبی نسبت به آن و نسبت به حدس گروه بزرگ‌تر، در هر دو معیار بیش از پنج واحد درصد بود و بازهٔ اختلاف کاملاً مثبت بود. جدول کامل اختلاف‌ها در `paired_intervals.csv` ثبت است.",
        "",
        "## بررسی تکمیلیِ نتیجهٔ محدود",
        "",
        "پس از مشاهدهٔ نتیجهٔ اولیه، بررسی تکمیلی با طرح جداگانه ثبت شد. این مرحله تأییدِ ازپیش‌ثبت‌شدهٔ مستقلی محسوب نمی‌شود. سه بخش با عنوان کتاب جدا در همان ۷۶۸ نویسندهٔ توسعه ساخته شد و هیچ تنظیم یا ویژگی تازه‌ای به مدل اضافه نشد.",
        "",
        "| روش | Macro-F1 در سه بخش | میانگین بازیابی دو گروه |",
        "|---|---:|---:|",
    ]
    for name in ["majority", "style", "words", "combined", "naive_bayes"]:
        r = cv[name]
        lines.append(f"| {names[name]} | {float(r['macro_f1']):.3f} | {float(r['balanced_accuracy']):.3f} |")
    lines += [
        "",
        f"حداکثر شباهت کسینوسی متن‌های ارزیابی به متن‌های آموزش در نمایش واژگانی {similarity['maximum_cosine']:.3f} بود. هیچ متن ارزیابی از آستانه‌های ۰٫۶ یا ۰٫۸ عبور نکرد. در این سنجش، نتیجه با نسخه‌های تقریباً یکسانِ متن توضیح داده نمی‌شود. این بررسی تمام صورت‌های بازنویسی یا وابستگیِ مجموعه‌کتاب‌ها و نویسندگان کتاب را رد نمی‌کند.",
        "",
        "## این الگو چه معنایی دارد؟",
        "",
        "شواهد فعلی عمدتاً واژگانی‌اند. شاخص‌های ساختار نوشتار به‌تنهایی قدرت اندکی داشتند. در توضیح توصیفی مدل، واژه‌هایی مثل love، romance و suspense بیشتر به سوی گروه عاشقانه و science، fiction و human بیشتر به سوی علمی‌تخیلی سهم داشتند. این فهرست، توضیح محاسبهٔ مدل است و نباید به‌عنوان واژه‌نامهٔ ادبیِ تأییدشده یا ویژگی شخصیتی افراد خوانده شود. بخشی از تفاوت می‌تواند از ژانرهای فرعی، اشارهٔ صریح به ژانر و سبکِ نقدنویسیِ گروه‌های خاص مانند دریافت‌کنندگان نسخهٔ پیش از انتشار ناشی شود.",
        "",
        "## تصمیم و محدودیت",
        "",
        f"وضع معیار ثبت‌شده: معیار کلی = {decision['main_gate']}، افزایش پس از کنترل زمینه = {decision['context_gate']}، مقایسهٔ ژانر مشترک = {decision['common_genre_gate']}. بنابراین تصمیم اولیهٔ ادامهٔ هدف گسترده همان `False` باقی مانده و برای بهترکردن نتیجه تغییر نکرده است. مسیر محدودِ دوگروهی به‌عنوان اصلاح دامنه انتخاب می‌شود و نیازمند ارزیابی نهاییِ جداگانه است.",
        "",
        "کاربران ارزیابیِ دوگروهی پیش‌تر در EDA و ممیزی واژه‌ها بررسی شده‌اند. در نتیجه این دادهٔ توسعه است و یک آزمون نهاییِ دست‌نخورده نیست. متن‌های بخش آزمون نهایی در این کار باز نشدند. بازه‌ها با ۲۰۰۰ بازنمونه‌گیری عنوان کتاب و ثابت نگه‌داشتن پیش‌بینی‌ها محاسبه شده‌اند و تمام عدم‌قطعیت ناشی از آموزش دوبارهٔ مدل یا انتخاب دامنه را پوشش نمی‌دهند. کاربرد روی کاربر با گرایش دیگر، متن غیرنقد، زبان دیگر یا ژانر دیگرِ کتاب هنوز ارزیابی نشده است.",
        "",
        "برای فاز بعد، نمونه و معیارهای همین سؤال محدود باید تثبیت شوند، سهم واژگان در برابر ساختار و اشارهٔ مستقیم به ژانر سنجیده شود، و ارزیابی نهایی با نویسندگان و کتاب‌های جدا انجام شود. این مرحله مدل نهایی یا نتیجهٔ نهایی پروژه نیست.",
        "",
        "![نتیجهٔ ارزیابی و دامنهٔ قابل دفاع](feasibility.png)",
        "",
        "## بازتولید و خروجی‌ها",
        "",
        "```powershell",
        "python -m unittest discover -s tests -p test_predictive_feasibility.py -v",
        "python src/predictive_feasibility.py all",
        "python src/audit_common_prediction.py",
        "python src/build_predictive_feasibility_report.py",
        "```",
        "",
        "برای محدودکردن مصرف منابع می‌توان پیش از اجرا `OPENBLAS_NUM_THREADS=1` و `OMP_NUM_THREADS=1` تنظیم کرد. همهٔ ضرایب انتخاب‌شده، تعداد تکرار بهینه‌ساز و نتیجهٔ همگرایی در `*_fit_diagnostics.csv` ثبت‌اند. پیش‌بینی‌های فردی و نقاط بازیابی اجرا در `data/analysis/predictive-feasibility-v1/` محلی می‌مانند. تغییر کد یا ورودی‌ها مستلزم اجرای تازه در پوشهٔ خروجی تازه است و نباید با نقاط بازیابیِ قدیمی ترکیب شود.",
        "",
    ]
    (PUB / "report-fa.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    short = ["Majority", "Prior random", "Style", "Words", "Combined", "Naive Bayes"]
    model_order = ["majority", "prior_random", "style", "words", "combined", "naive_bayes"]
    for ax, scope, title in [(axes[0, 0], "book", "Seven preferences: unseen authors and book titles"),
                             (axes[0, 1], "common", "Two preferences: thriller reviews, separate authors/titles")]:
        values = [float(table[(scope, m)]["macro_f1"]) for m in model_order]
        lows = [v - float(table[(scope, m)]["macro_f1_low"]) for v, m in zip(values, model_order)]
        highs = [float(table[(scope, m)]["macro_f1_high"]) - v for v, m in zip(values, model_order)]
        ax.bar(short, values, yerr=[lows, highs], color=["#9aa7b0"] * 2 + ["#27838b"] * 4, capsize=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1")
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(title, fontsize=11)
    ax = axes[1, 0]
    positions = np.arange(len(LABELS))
    for j, (model, color) in enumerate([("combined", "#27838b"), ("naive_bayes", "#b4602c")]):
        r = [r for r in read_csv(PUB / "per_genre.csv") if r["scheme"] == "book" and r["model"] == model]
        ax.bar(positions + (j - .5) * .36, [float(next(v["recall"] for v in r if v["genre"] == g)) for g in LABELS],
               width=.36, label=model, color=color)
    ax.set_xticks(positions, LABELS)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel("Recall per preference group")
    ax.set_title("Overall performance hides weak preference groups", fontsize=11)
    ax.legend()
    ax = axes[1, 1]
    x = np.arange(3)
    for model, color in [("combined", "#27838b"), ("words", "#6385b6"), ("naive_bayes", "#b4602c"), ("style", "#9aa7b0")]:
        r = [r for r in read_csv(PUB / "common_cv_fold_metrics.csv") if r["model"] == model]
        ax.plot(x, [float(next(v["macro_f1"] for v in r if int(v["fold"]) == i)) for i in x], "o-", label=model, color=color)
    ax.set_xticks(x, ["Book fold 1", "Book fold 2", "Book fold 3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Follow-up: two-preference result across book splits", fontsize=11)
    ax.legend(ncol=2)
    fig.suptitle("Feasibility evidence supports a restricted question", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(PUB / "feasibility.png", dpi=160)
    plt.close(fig)

    paths = [p for p in PUB.glob("*") if p.is_file() and p.name != "manifest.json"]
    paths += list(OUT.glob("*_predictions.csv"))
    paths += [ROOT / "src/predictive_feasibility.py", ROOT / "src/audit_common_prediction.py", Path(__file__),
              ROOT / "tests/test_predictive_feasibility.py", ROOT / "src/phase2_patterns.py", ROOT / "src/style_indicators.py",
              ROOT / "src/writing_style.py", ROOT / "src/audit.py", ROOT / "OBJECTIVE_AMENDMENT.md",
              ROOT / "README.md", ROOT / "PROJECT_CARD.md", ROOT / "reports/phase1.pdf",
              ROOT / "data/analysis/phase2-final/sample.jsonl", ROOT / "data/analysis/phase2-final/independent.csv",
              ROOT / "data/analysis/phase2-final/first_review_style.csv.gz", ROOT / "data/analysis/phase2-final/common_genre_sample.jsonl",
              ROOT / "data/analysis/validity-v1/validation_common.jsonl"]
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest() for p in paths}
    (PUB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved decision report, figure, and {len(manifest)} hashes.")


if __name__ == "__main__":
    main()
