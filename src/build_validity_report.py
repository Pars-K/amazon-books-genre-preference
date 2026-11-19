"""Publish the statistical evidence audit with distributions and conservative conclusions."""

import json
from hashlib import file_digest,sha256
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from audit_phase2_validity import FROZEN, PRIVATE, PUBLIC
from phase2_sample import OUT
from style_indicators import METRICS


NAMES = {
    "words_per_review":"طول نقد", "sentences_per_review":"تعداد جمله در نقد",
    "paragraphs_per_review":"تعداد پاراگراف در نقد", "words_per_paragraph":"طول پاراگراف",
    "mean_sentence_words":"طول متوسط جمله", "sd_sentence_words":"پراکندگی طول جمله",
    "vocabulary_diversity_20_pct":"تنوع واژه در قطعهٔ ۲۰واژه‌ای",
    "first_singular_per_1000_words":"اول‌شخص مفرد", "first_plural_per_1000_words":"اول‌شخص جمع",
    "second_person_per_1000_words":"دوم‌شخص", "contractions_per_1000_words":"صورت‌های کوتاه",
    "exclamations_per_1000_words":"علامت تعجب", "ellipses_per_1000_words":"سه‌نقطه",
    "colons_per_1000_words":"دونقطه", "semicolons_per_1000_words":"نقطه‌ویرگول",
    "upper_tokens_per_1000_words":"واژهٔ تماماً بزرگ", "emphatic_runs_per_1000_words":"رشتهٔ نشانه‌های تأکید",
    "colloquial_tokens_per_1000_words":"صورت‌های محاوره‌ای فهرست‌شده", "question_ending_pct":"پایان پرسشی",
    "subordination_sentence_pct":"عبارتِ وابستگی جمله", "hedge_sentence_pct":"عبارتِ تردید",
    "certainty_sentence_pct":"عبارتِ قطعیت", "causal_sentence_pct":"عبارتِ علت یا نتیجه",
    "contrast_sentence_pct":"عبارتِ تقابل", "example_sentence_pct":"عبارتِ مثال",
    "reading_experience_sentence_pct":"عبارتِ تجربهٔ خواندن", "recommendation_sentence_pct":"عبارتِ توصیه",
    "plot_description_sentence_pct":"عبارتِ توصیف داستان", "evaluation_sentence_pct":"عبارتِ ارزیابی",
}
GENRE_NAMES = dict(zip(
    ["Fantasy","Science Fiction","Romance","Mystery","Thriller","Philosophy","Science"],
    ["فانتزی","علمی‌تخیلی","عاشقانه","معمایی","تریلر","فلسفی","علمی"]))


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|",
                      *["| "+" | ".join(map(str,r))+" |" for r in rows]])


def figures():
    discovery = pd.read_csv(OUT/"common_genre_style.csv.gz")
    validation = pd.read_csv(PRIVATE/"validation_common_style.csv")
    colors = {"Romance":"#237C83","Science Fiction":"#BA632F"}
    fig,axes = plt.subplots(2,2,figsize=(10.5,8))
    for ax,data,title in zip(axes[0],[discovery,validation],["Discovery authors","Separate validation authors"]):
        for group,color in colors.items():
            values = np.sort(data.loc[data.preference==group,"first_singular_per_1000_words"])
            ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=color,
                    label=f"{group} (n={len(values)})")
        full_max = max(discovery.first_singular_per_1000_words.max(),
                       validation.first_singular_per_1000_words.max())
        ax.set_xlim(0,1.03*full_max)
        ax.set_ylim(0,1.02)
        ax.set_xlabel("First-person singular per 1,000 words")
        ax.set_ylabel("Fraction of authors")
        ax.set_title(title,loc="left")
        ax.legend(frameon=False,fontsize=9)
    words = pd.read_csv(PUBLIC/"word_replication.csv").set_index("term")
    for ax,term in zip(axes[1],["love","fiction"]):
        row = words.loc[term]
        for pos,(group,color) in enumerate(colors.items()):
            a,n = (int(row.romance_using),int(row.romance_users)) if group=="Romance" else (int(row.sf_using),int(row.sf_users))
            ci = binomtest(a,n).proportion_ci(method="exact")
            point = 100*a/n
            ax.errorbar(point,pos,xerr=[[point-100*ci.low],[100*ci.high-point]],fmt="o",color=color,capsize=4)
            ax.text(point,pos+.23,f"{a}/{n}",ha="center",fontsize=9)
        ax.set_yticks([0,1],list(colors))
        ax.set_ylim(-.4,1.6)
        ax.set_xlim(0,70)
        ax.set_title(f'Presence of "{term}" in validation reviews',loc="left",fontsize=11)
        ax.set_xlabel("Authors using the word (%)")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle("Observed patterns in reviews of thriller books",x=.03,ha="left",fontsize=17,weight="bold")
    fig.text(.03,.015,"One review per author. Lower panels show exact 95% intervals for each proportion.",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(PUBLIC/"observed_patterns.png",dpi=165,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def run():
    replication = pd.read_csv(PUBLIC/"common_style_evidence.csv")
    primary = replication[(replication.scope=="validation") & replication.p_holm_primary_2.notna()].set_index("metric")
    words = pd.read_csv(PUBLIC/"word_replication.csv")
    summary = pd.read_csv(PUBLIC/"discovery_summary.csv")
    repeat = pd.read_csv(FROZEN/"review_repeatability.csv").query("scope=='all'").set_index("metric")
    support = pd.read_csv(FROZEN/"style_support.csv").set_index("metric")
    cluster = json.loads((PUBLIC/"cluster_evidence.json").read_text())
    evidence = []
    val = replication[replication.scope=="validation"].set_index("metric")
    for metric in METRICS:
        measurement = "نشانگر واژه‌نامه‌ای با نیاز به اعتبارسنجی معنا" if metric.endswith("_sentence_pct") else "شمارش با تعریف عملیاتی"
        evidence.append(dict(metric=metric,name=NAMES[metric],measurement=measurement,
                             repeat_spearman=repeat.loc[metric,"repeat_spearman"],
                             single_review_zero_pct=100*support.loc[metric,"single_review_zero_share"],
                             validation_cliff_delta=val.loc[metric,"cliff_delta"],
                             validation_q_by_29=val.loc[metric,"q_by_29"]))
    pd.DataFrame(evidence).to_csv(PUBLIC/"indicator_evidence.csv",index=False)
    first = primary.loc["first_singular_per_1000_words"]
    length = primary.loc["mean_sentence_words"]
    paragraphs = [
        "# ممیزی شواهد آماری فاز دوم","",
        "این ممیزی پس از پرسش کاربر دربارهٔ اتکاپذیری الگوها انجام شد. نسخهٔ مرجعِ تحلیل اکتشافی کامیت dfd8b18 است. تعریف ویژگی، تکرارپذیری اندازه‌گیری، تفاوت گروه‌ها و تعمیم به کاربر جداگانه، هرکدام شواهد مخصوص خود را می‌خواهند.","",
        "## نتیجهٔ اصلی","",
        "دو خوشهٔ واژگانی در اجراهای مختلف تکرارپذیرند، ولی جدایی هندسی آن‌ها کم است و ارتباطشان با طول و موضوع نقد بیشتر از گرایش مرجع نویسنده است. چند واژه تفاوت مشاهده‌شدهٔ قابل تکرار دارند. برای بسیاری از واژه‌ها و شاخص‌های دیگر، تعمیم هنوز پشتیبانی کافی ندارد.","",
        "## طرح بررسی تکمیلی","",
        "پیش از محاسبهٔ تفاوت‌های کاربران جداگانه، فهرست ۲۴ واژهٔ قبلی و هر ۲۹ شاخص ثابت شدند. دو فرضیهٔ اصلیِ سبک، اول‌شخص مفرد و طول جمله، از گزارش قبلی آمده‌اند. تنظیمات در protocol.json ثبت شدند. واژگان، وزن‌ها، مؤلفه‌ها و مرکزهای خوشه از بخش توسعهٔ قبلی استفاده شدند.","",
        "از بخش اعتبارسنجیِ کاربران، از هر نفر یک نقد با همان قاعدهٔ قبلی انتخاب شد. ۲٬۵۷۰ نویسنده در این بخش حضور دارند. مقایسهٔ ژانر مشترک شامل ۱۳۵ عاشقانه‌خوان و ۲۲ علمی‌تخیلی‌خوان است که هر کدام دربارهٔ یک کتاب تریلر نوشته‌اند. گرایش اصلی آن‌ها پس از حذف این کتابِ خارج از ژانرِ گرایش همچنان حفظ می‌شود. نویسندگان این مقایسه با نمونهٔ اکتشافی مشترک نیستند. متن تکراریِ دقیق پس از پاک‌سازی هم در برابر نمونهٔ اکتشافی یافت نشد.","",
        "این داده از همان آرشیو آمازون است و تکرار در منبع بیرونی به شمار نمی‌آید. بخش اعتبارسنجی از این نقطه در توسعهٔ تحلیل استفاده شده است. برای ارزیابی نهایی مدل باید نقش آن روشن باشد. مجموعهٔ آزمون نهایی در این ممیزی به کار نرفته است.","",
        "## خوشه‌ها چقدر قابل اتکا هستند؟","",
        f"در تحلیل اولیه، سیلوئت حدود ۰٫۰۴۳ و توافق شروع‌های مختلف حدود ۰٫۹۹ بود. در نویسندگان جداگانه، با همان مرکزهای قبلی، سیلوئت {cluster['validation_silhouette']:.3f} و اندازهٔ دو گروه {cluster['validation_cluster_sizes'][0]} و {cluster['validation_cluster_sizes'][1]} شد. برای یک نقد، واژهٔ شناخته‌شده‌ای در بازنمایی باقی نماند. این عددهای نزدیک به صفر، هم‌پوشانی زیاد گروه‌ها را نشان می‌دهند.","",
        f"برای تشخیص اندازهٔ این جدایی، ۳۹ مجموعه از یک توزیع نرمال با میانگین و کوواریانس همان فضا شبیه‌سازی و مانند داده نرمال شدند. روی نمونه‌های ۱٬۸۰۰نفری، میانهٔ سیلوئت مرجع {cluster['single_gaussian_silhouette_median']:.3f} و صدک‌های ۵ تا ۹۵ آن {cluster['single_gaussian_p05']:.3f} تا {cluster['single_gaussian_p95']:.3f} بود. در نمونهٔ واقعیِ هم‌اندازه، سیلوئت {cluster['actual_sample_silhouette']:.3f} بود. تفاوت اندکی دیده می‌شود. این یک بررسی تشخیصیِ وابسته به مدل مرجع است.","",
        "اثر حذف نرمال‌سازی و تغییر تعداد مؤلفه‌ها در گزارش قبلی ثبت شده است. ارتباط خام خوشه با گرایش حدود ۰٫۰۴۳ بیت و افزودهٔ آن نسبت به خط مبنا پس از لحاظ‌کردن ژانر کتاب و پنجک طول حدود ۰٫۰۰۳۶ بیت بود. در مقایسهٔ عنوان یکسان، شواهد فراتر از خط مبنا به دست نیامد. این خروجی برای ادعای کشف گروه‌های طبیعیِ علاقه‌مندان ژانرها کفایت نمی‌کند.","",
        "## آیا واژگان فقط رتبه گرفته بودند؟","",
        "فهرست اولیه از شمار واقعیِ نویسندگانِ استفاده‌کننده از واژه و نسبت شانس ساخته شده بود. با این حال، رتبهٔ بالا به‌تنهایی تعمیم را تأیید نمی‌کند. در ممیزی جدید، برای ۱۲هزار ویژگی در هفت گروه، ۸۴هزار مقایسهٔ دقیق فیشر انجام شد. اصلاح بنجامینی–یکوتیلی در کل این خانواده اعمال شد تا وابستگی بین واژه‌ها هم لحاظ شود.","",
        "برای پایداری رتبه، نویسندگان هر طرف ۲۰۰ بار بازنمونه‌گیری و فهرست ۱۵تایی دوباره ساخته شد. جدول زیر دو جنبه را جدا گزارش می‌کند. ستون آخر نشان می‌دهد چند واژه در دست‌کم ۸۰٪ این بازنمونه‌گیری‌ها در میان ۱۵ مورد نخست مانده‌اند. این ۸۰٪ یک توصیف پایداری رتبه است.","",
        table(["گروه","کاربران","واژهٔ گزارش‌شده","مقایسه با p اصلاح‌شده زیر ۰٫۰۵","حفظ رتبه در ۸۰٪ اجراها"],
              [[GENRE_NAMES[r.genre],r.users,r.reported_words,r.q05_words,r.selected_in_80pct_resamples] for r in summary.itertuples()]),"",
        "تفاوتِ فراوانی در همین مجموعه برای بسیاری از واژه‌ها شواهد آماری دارد، اما ترتیب دقیق واژه‌های برجسته می‌تواند تغییر کند. در گروه فلسفی فقط ۱۰ نفر حضور دارند و اتکای فهرست به چند نفر زیاد است. موضوع کتاب و انتخاب عنوان نیز می‌توانند تفاوت واقعی در واژگان ایجاد کنند.","",
        "## واژگان در نویسندگان جداگانه چه شدند؟","",
        f"از ۲۴ واژهٔ از پیش انتخاب‌شده در مقایسهٔ کتاب‌های تریلر، {int(words.direction_replicated.sum())} مورد همان جهت تفاوت را در کاربران جداگانه داشتند. پس از آزمون دقیق فیشر و اصلاح بنجامینی–یکوتیلی برای هر ۲۴ مورد، {int((words.q_by_24<.05).sum())} مورد شواهد کافی در آستانهٔ ۰٫۰۵ داشتند.","",
        table(["واژه","عاشقانه‌خوان‌ها","علمی‌تخیلی‌خوان‌ها","p اصلاح‌شده"],
              [[r.term,f"{r.romance_using}/{r.romance_users} ({r.romance_pct:.1f}٪)",
                f"{r.sf_using}/{r.sf_users} ({r.sf_pct:.1f}٪)",f"{r.q_by_24:.4f}"] for r in words[words.q_by_24<.05].itertuples()]),"",
        "در ۱۵۷ نقد این مقایسه، ۱۴۷ عنوان کتاب متمایز وجود داشت. استفاده از love در ۶۳ عنوانِ مربوط به عاشقانه‌خوان‌ها دیده شد. با بازنمونه‌گیری عنوان‌ها به‌جای افراد، بازهٔ اختلاف فراوانی love همچنان مثبت و برای fiction همچنان منفی بود. این شواهد به همین دو گروه و نقد کتاب‌های تریلر مربوط است.","",
        "برای ۲۲ واژهٔ دیگر، اندازه و جهت اثر و تعداد استفاده‌کنندگان در word_replication.csv آمده است. کوچک‌بودن نمونهٔ علمی‌تخیلی‌خوان‌ها عدم‌قطعیت را زیاد می‌کند. برای تعمیم کل فهرست هفت ژانر به کاربران تازه به بررسی جداگانهٔ بیشتری نیاز داریم.","",
        "![توزیع واقعیِ شاخص و فراوانی واژه در کاربران اکتشافی و جداگانه](observed_patterns.png)","",
        "## دیگر شاخص‌های نوشتاری","",
        "شمارش علامت تعجب، واژه یا صورت ضمیر، تحت قواعد اعلام‌شده قابل تکرار است. تفسیر آن به‌عنوان هیجان، پیچیدگی، استدلال یا ویژگی ثابتِ شخص نیازمند شواهد بیشتری است. برای نمونه، در دو نقد جدا از هر نویسنده، همبستگی طول جمله حدود ۰٫۴۶، اول‌شخص مفرد ۰٫۳۱ و تنوع واژگان ۰٫۰۸ است. مقایسه‌های هم‌ژانر و متن‌های بلند هم در گزارش تکرارپذیری ثبت‌اند.","",
        f"در کاربران جداگانه، جهت اختلاف اول‌شخص مفرد با گزارش اولیه هم‌سو بود. آزمون تفاوت توزیع‌ها، پس از اصلاح برای دو فرضیهٔ اصلی، p برابر {first.p_holm_primary_2:.4f} داشت. اختلاف میانه {first.difference:.1f} مورد در هزار واژه و بازهٔ بوت‌استرپ آن {first.p025:.1f} تا {first.p975:.1f} بود. بازه شامل صفر است، بنابراین مقدار اختلاف میانه هنوز با عدم‌قطعیت زیادی همراه است. آزمون توزیع و بازهٔ اختلاف میانه دو پرسش آماری متفاوت دارند.","",
        f"برای طول جمله در کاربران جداگانه، p اصلاح‌شدهٔ دو فرضیهٔ اصلی {length.p_holm_primary_2:.4f} و بازهٔ اختلاف میانه {length.p025:.1f} تا {length.p975:.1f} واژه بود. برای تفاوت این شاخص در این نمونه شواهد کافی به دست نیامد.","",
        "وقتی هر ۲۹ شاخص را هم‌زمان با اصلاح محافظه‌کارانه بررسی کردیم، در بخش اعتبارسنجی هیچ موردی از آستانهٔ ۰٫۰۵ عبور نکرد. این نتیجه، دامنهٔ ادعا را محدود می‌کند. دو فرضیهٔ اصلی از قبل در گزارش قبلی مشخص بودند و نتیجهٔ آن‌ها جداگانه نیز گزارش شد.","",
        "به‌علت صفرهای زیاد و مقادیر مساوی در برخی شاخص‌ها، احتمال‌های آزمونِ رتبه‌ها با ۹٬۹۹۹ جابه‌جاییِ عضویت گروه محاسبه شد. تقریب نرمال برای شاخص‌های بسیار کم‌وقوع کنار گذاشته شد. این اصلاحِ روش محاسبه در جریان ممیزی انجام شد و گزینش شاخص‌ها و گروه‌ها ثابت ماند.","",
        "برای شاخص‌های مبتنی بر عبارتِ توصیه، توصیف داستان و تجربهٔ خواندن، بازخوانی قبلی توسط دستیار در ۵۶ جمله پوشش ضعیف واژه‌نامه را نشان داده بود. مثلاً از ۱۲ جملهٔ توصیف داستان، هیچ‌کدام عبارتِ واژه‌نامه را نداشتند. ارزیاب انسانیِ مستقل در آن بازخوانی حضور نداشت. این ستون‌ها باید با عنوان حضور همان عبارت‌ها خوانده شوند.","",
        "## جدول شواهد هر ۲۹ شاخص","",
        table(["شاخص","نوع اندازه‌گیری","تکرارپذیری دو نقد","درصد صفر در نقد منتخب","p اصلاح‌شدهٔ ۲۹تایی در کاربران جداگانه"],
              [[r["name"],r["measurement"],f"{r['repeat_spearman']:.2f}",f"{r['single_review_zero_pct']:.1f}",f"{r['validation_q_by_29']:.3f}"] for r in evidence]),"",
        "## محدودیت و جایگاه نتیجه","",
        "مقایسه‌ها بر دادهٔ مشاهده‌ایِ کاربران واجد شرایط در هفت گروه آمازون انجام شده‌اند. امکان تفاوت در انتخاب عنوان، ژانرهای فرعی کتاب، امتیاز، زمان و فرصت مواجهه وجود دارد. آزمون آماریِ رابطه، توان پیش‌بینیِ کاربر تازه یا رابطهٔ علت‌ومعلولی را تعیین نمی‌کند. مجموعهٔ آزمون نهایی برای آن ارزیابی لازم است.","",
        "بخشی از ویژگی‌های قبلی به‌عنوان نامزد اکتشافی ساخته شده بودند. اکنون برای بعضی واژه‌ها، شمار واقعی، اندازهٔ تفاوت، پایداری رتبه و تکرار در کاربران جداگانه داریم. برای خوشه‌ها و بسیاری از شاخص‌های دیگر، نتیجه با اندازهٔ اثر و عدم‌قطعیت محدود گزارش می‌شود.","",
        "## بازتولید و منابع","",
        "فرمان‌ها از ریشهٔ مخزن اجرا می‌شوند. خروجی فردی زیر data/analysis/validity-v1 قرار می‌گیرد.","",
        "```console","python src/audit_phase2_validity.py freeze","python src/audit_phase2_validity.py words",
        "python src/audit_phase2_validity.py replicate","python src/audit_phase2_validity.py clusters",
        "python src/audit_phase2_validity.py books","python src/build_validity_report.py","```","",
        "- [جدول کامل راستی‌آزمایی واژه‌ها](word_replication.csv)",
        "- [جدول شواهد شاخص‌ها](indicator_evidence.csv)",
        "- [مقایسه‌های سبکی با اندازهٔ اثر و بازه](common_style_evidence.csv)",
        "- [حساسیت به عنوان کتاب](book_sensitivity.csv)","",
        "- [تعریف و تفسیر سیلوئت](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)",
        "- [آزمون دقیق فیشر](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)",
        "- [اصلاح آزمون‌های متعدد و وابسته](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)",
        "- [آزمون رتبه‌ای و مقادیر مساوی](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)","",
    ]
    (PUBLIC/"report-fa.md").write_text("\n".join(paragraphs),encoding="utf-8")
    figures()
    def fingerprint(path):
        with path.open("rb") as f:
            return file_digest(f,"sha256").hexdigest()
    inputs = [
        Path("reports/phase2/completion_manifest.json"),FROZEN/"targeted_common_terms.csv",
        FROZEN/"distinctive_terms.csv",FROZEN/"review_repeatability.csv",FROZEN/"style_support.csv",
        OUT/"lexical_fit.npz",OUT/"tfidf.npz",OUT/"common_genre_style.csv.gz",
        Path("data/analysis/eda-v1/users.csv"),Path("data/analysis/eda-v1/reviews.jsonl.gz"),
    ]
    sources = [Path("src/audit_phase2_validity.py"),Path("src/build_validity_report.py"),Path("tests/test_phase2_validity.py")]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),baseline_analysis_commit="dfd8b18",
        code_review_baseline="123c505",discovery_users=10968,
        comparison_validation_authors=dict(Romance=135,Science_Fiction=22),
        global_word_test_family=84000,fixed_replication_words=24,fixed_style_hypotheses=2,
        additional_style_hypotheses=29,word_rank_bootstraps=200,rank_permutations=9999,
        median_bootstraps=2000,book_bootstraps=2000,normalized_gaussian_repeats=39,
        inputs={str(p):fingerprint(p) for p in inputs},
        source_sha256_lf={str(p):sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in sources},
        outputs={str(p):fingerprint(p) for p in [*sorted(PUBLIC.iterdir()),*sorted(PRIVATE.iterdir())]
                 if p.is_file() and p.name!="manifest.json"},
    )
    (PUBLIC/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Published statistical evidence audit")


if __name__ == "__main__":
    run()
