"""Report paired changes from balanced training, including their costs."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from balanced_genre_feasibility import GROUPS, OUT, PUB
from predictive_feasibility import ROOT, read_csv


def main():
    scores = {r['model']: r for r in read_csv(PUB / 'all_metrics.csv')}
    group = {(r['model'], r['genre']): r for r in read_csv(PUB / 'all_per_genre.csv')}
    common = {r['model']: r for r in read_csv(PUB / 'thriller_input_metrics.csv')}
    names = {'style':'نوشتار، بدون وزن', 'style_balanced':'نوشتار، وزن‌دار',
             'words':'TF-IDF، بدون وزن', 'words_balanced':'TF-IDF، وزن‌دار',
             'combined':'ترکیب نوشتار و TF-IDF، بدون وزن', 'combined_balanced':'ترکیب نوشتار و TF-IDF، وزن‌دار',
             'naive_bayes':'نایو بیز، بدون وزن', 'naive_bayes_equal_prior':'نایو بیز، فقط پیش‌فرضِ برابر گروه‌ها',
             'naive_bayes_balanced':'نایو بیز، شمارهای وزن‌دار و پیش‌فرض برابر'}
    fa = {'Mystery':'معمایی', 'Thriller':'تریلر', 'Romance':'عاشقانه'}
    text = [
        '# مقایسهٔ آموزش وزن‌دار و بدون وزن', '',
        'تاریخ: ۱۹ نوامبر ۲۰۲۶. این بررسی پس از تذکر کاربر دربارهٔ غالب‌شدن تریلر انجام شد. طرح قبل از اجرا در کامیت `720606a` ثبت شد. مدل‌های بدون وزن و نتیجه‌های قبلی حفظ شده‌اند. این بررسی توسعه‌ای است و آزمون نهایی نیست.', '',
        '## چرا این بررسی لازم بود؟', '',
        'معیارهای Macro-F1 و میانگین بازیابی به هر گروه سهم برابر در ارزیابی می‌دهند، اما مدل بدون وزن همچنان از مجموع خطای همهٔ کاربران یاد می‌گیرد. تعداد زیاد تریلرخوان‌ها می‌تواند آن مجموع را غالب کند. بنابراین استفاده از معیار متوازن به‌تنهایی معادل آموزش متوازن نیست و شکست روش بدون وزن در گروه‌های کوچک‌تر برای قضاوت نهایی کافی نبود.', '',
        'از هر کاربر یک نقد داریم. وزن به گرایش رفتاریِ نویسنده داده می‌شود و ژانر کتاب ورودی مبنای آن نیست. آموزش وزن‌دار، اهمیت مجموع خطای هر سه گروه را برابر می‌کند. تعداد پیش‌بینی‌های نهایی الزاماً برابر نمی‌شود.', '',
        '## وزن چگونه تعیین شد؟', '',
        'در هر بخش آموزش، وزن هر شخص از گروه g برابر است با تعداد تمام کاربران آموزش تقسیم بر سه برابر تعداد کاربرانِ همان گروه. میانگین وزن یک می‌ماند و مجموع وزن هر گروه یک‌سوم مجموع وزن‌ها می‌شود. وزن‌ها فقط از آموزش همان بخش محاسبه می‌شوند؛ بخش داخلیِ انتخاب تنظیم نیز وزن‌های خودش را دارد.', '',
        '| گرایش | کاربرانِ کل نمونهٔ توسعه | وزنِ صرفاً توضیحی بر اساس کل نمونه |', '|---|---:|---:|',
    ]
    for genre in GROUPS:
        n = int(group[('combined', genre)]['users'])
        text.append(f'| {fa[genre]} | {n:,} | {10281 / (3 * n):.3f} |')
    text += ['',
        'اعداد ستون آخر صرفاً برای توضیح‌اند؛ برازش‌ها از شمار آموزش همان بخش استفاده کردند. جدول تمام وزن‌های واقعیِ هر بخش در `*_training_weights.csv` ثبت است. هیچ نمونهٔ ارزیابی حذف، تکرار یا وزن‌دهی نشد. تمام کاربران، متن‌ها، عنوان‌ها و تقسیم‌ها با بررسی بدون وزن یکسان‌اند و تطبیقِ شناسه، کتاب، گروه واقعی و بخش برای هر پیش‌بینی کنترل شد.', '',
        '## نتیجهٔ کلی روی همان ۱۰٬۲۸۱ نویسنده', '',
        '| روش | Macro-F1 | میانگین بازیابی گروه‌ها | دقت کلی |', '|---|---:|---:|---:|',
    ]
    for key, label in names.items():
        row = scores[key]
        text.append(f"| {label} | {float(row['macro_f1']):.3f} | {float(row['balanced_accuracy']):.3f} | {float(row['accuracy']):.3f} |")
    text += ['',
        'در مدل ترکیبیِ لجستیک، Macro-F1 از حدود ۰٫۵۷۶ به ۰٫۶۴۹ و میانگین بازیابی از ۰٫۵۲۳ به ۰٫۶۸۸ رسید. بازهٔ ۹۵ درصدی افزایش Macro-F1 حدود ۰٫۰۵۸ تا ۰٫۰۸۹ و افزایش میانگین بازیابی حدود ۰٫۱۵۲ تا ۰٫۱۸۰ بود. دقت کلی از حدود ۰٫۸۴۲ به ۰٫۸۰۵ کاهش یافت. معیارهای متوازن و معیار تعداد کل پاسخ درست، هدف یکسانی ندارند.', '',
        'در نایو بیز، وزن‌دهی میانگین بازیابی را از حدود ۰٫۶۷۳ به ۰٫۷۰۴ رساند، اما Macro-F1 از حدود ۰٫۶۶۳ به ۰٫۶۲۳ و دقت کلی از حدود ۰٫۸۳۲ به ۰٫۷۶۵ کاهش یافت. بنابراین وزن‌دهی برای همهٔ مدل‌ها تمام معیارها را بهتر نکرد. نتیجه‌های کامل و بازهٔ تغییرها در `all_paired_metrics.csv` آمده‌اند.', '',
        '## مقایسهٔ هر گروه در همان مدل واژگانیِ قبلی', '',
        'این جدول نایو بیز را با خودش مقایسه می‌کند. اعداد شناسایی گروه‌ها در گزارش قبلی نیز از همین مدل بودند.', '',
        '| گرایش | شناسایی بدون وزن | شناسایی وزن‌دار | دقت نسبت‌دادن بدون وزن | دقت نسبت‌دادن وزن‌دار |', '|---|---:|---:|---:|---:|',
    ]
    for genre in GROUPS:
        a, b = group[('naive_bayes', genre)], group[('naive_bayes_balanced', genre)]
        text.append(f"| {fa[genre]} | {100 * float(a['recall']):.1f}٪ | {100 * float(b['recall']):.1f}٪ | {100 * float(a['precision']):.1f}٪ | {100 * float(b['precision']):.1f}٪ |")
    text += ['',
        'شناسایی یعنی سهم افراد واقعی یک گروه که پیدا شده‌اند. دقت نسبت‌دادن یعنی سهم پیش‌بینی‌های آن گروه که درست‌اند. افزایش شناسایی معمایی و عاشقانه با نسبت‌دادن اشتباهِ افراد بیشتری به این گروه‌ها همراه بود. این مبادله باید در انتخاب مدل نهایی دیده شود.', '',
        '## همان مقایسه برای مدل ترکیبی', '',
        '| گرایش | شناسایی بدون وزن | شناسایی وزن‌دار | دقت نسبت‌دادن بدون وزن | دقت نسبت‌دادن وزن‌دار |', '|---|---:|---:|---:|---:|',
    ]
    for genre in GROUPS:
        a, b = group[('combined', genre)], group[('combined_balanced', genre)]
        text.append(f"| {fa[genre]} | {100 * float(a['recall']):.1f}٪ | {100 * float(b['recall']):.1f}٪ | {100 * float(a['precision']):.1f}٪ | {100 * float(b['precision']):.1f}٪ |")
    a, b = group[('combined', 'Thriller')], group[('combined_balanced', 'Thriller')]
    text += ['',
        f"مدل ترکیبیِ بدون وزن {a['predicted_users']} نفر از ۱۰٬۲۸۱ نفر را تریلر تشخیص می‌داد. در نسخهٔ وزن‌دار این تعداد {b['predicted_users']} نفر شد. تعداد واقعیِ تریلرخوان‌ها ۸۲۲۹ نفر است. برابرکردن اهمیت گروه‌ها در یادگیری، به معنی وادارکردن مدل به خروجیِ یک‌سوم برای هر گروه نیست.", '',
        '## بررسی ژانر کتاب مشترک', '',
        'در زیرنمونهٔ نقدهای تریلر، همان ۵۵۷۹ تریلرخوان، ۸۰ معمایی‌خوان و ۶۸ عاشقانه‌خوان حفظ شدند. برای مدل ترکیبی، میانگین بازیابی از ۰٫۳۳۳ به حدود ۰٫۴۳۳ و Macro-F1 از ۰٫۳۲۹ به حدود ۰٫۳۶۷ رسید. بنابراین نتیجهٔ قبلی که مدل همه را تریلر می‌گفت، به سیاست آموزش وابسته بود و شاهد کافی برای نبود تفاوت زبانی نبود.', '',
        'با این حال نتیجهٔ وزن‌دار هم قوی نیست: مدل ترکیبی فقط ۳ نفر از ۸۰ معمایی‌خوان و ۲۳ نفر از ۶۸ عاشقانه‌خوان را پیدا کرد. دقت نسبت‌دادن به این گروه‌ها به‌ترتیب حدود ۱٫۷ و ۸٫۰ درصد بود. وزن‌دهی اهمیت داده‌های موجود را تغییر می‌دهد و نمونهٔ مستقل تازه تولید نمی‌کند. برای ادعای سبک مستقل از موضوع کتاب هنوز شواهد کافی نداریم. این زیرنمونه تمام ظرفیت آرشیو برای کاربران دارای نقد تریلر نیست.', '',
        '## نایو بیز: دو تغییر متفاوت', '',
        'در نسخهٔ وزن‌دارِ نایو بیز، شمار واژه‌های هر گروه در وزن همان گروه ضرب شد، هموارسازی یک واحد باقی ماند و احتمال اولیهٔ سه گروه برابر شد. در نسخهٔ دیگر، فقط احتمال اولیهٔ گروه‌ها برابر شد و شمارهای واژگانیِ قبلی حفظ شدند. چون وزن ثابتِ هر گروه در فراوانی نسبی واژه‌ها تا حد زیادی خنثی می‌شود، تفاوت این دو نسخه به هموارسازی نیز مربوط است. این دو روش جدا گزارش شده‌اند و با وزن‌دهی زیان در رگرسیون لجستیک یکسان تلقی نشده‌اند.', '',
        '## تصمیم جاری', '',
        'آموزش متوازن باید در مقایسه‌های فاز بعد حضور داشته باشد و روش بدون وزن خط مبنا بماند. تصمیم نهایی صرفاً با دقت کلی یا افزایش شناسایی اقلیت گرفته نمی‌شود؛ Macro-F1، میانگین بازیابی و دقت نسبت‌دادن هر گروه کنار هم سنجیده می‌شوند. با هدف اهمیت برابرِ سه گرایش، نسخهٔ وزن‌دار مبنای ادامهٔ بررسی است و این انتخاب توزیع واقعیِ کاربران یا احتمال‌های جمعیتی را برابر فرض نمی‌کند.', '',
        'قدرت فعلی بیشتر واژگانی است. نمونه و جدایی کتاب/نویسنده حفظ شده‌اند، اما دادهٔ ارزیابی قبلاً در توسعه استفاده شده است. بخش آزمون نهایی خوانده نشد. بازه‌ها حاصل ۲۰۰۰ بازنمونه‌گیریِ عنوان کتاب با پیش‌بینی‌های ثابت‌اند و تمام عدم‌قطعیتِ آموزش مجدد یا انتخاب روش را در بر نمی‌گیرند.', '',
        '![اثر وزن‌دهی بر شناسایی و دقت نسبت‌دادن هر گروه](balance_effect.png)', '',
        '## کنترل محاسبات و بازتولید', '',
        'چهار آزمون تازه، برابری سهم گروه‌ها، گرادیان عددیِ تابع وزن‌دار، یکسان‌بودن تابع وزن‌دار با نسخهٔ اصلی برای وزن‌های یک، و حذف اثر صرفِ فراوانی بر گرادیانِ ثابت مدل را کنترل کردند. تمام ۳۶ آزمون پروژه گذشتند. ۱۶۸ برازش برای تنظیم داخلی و مدل هر بخش انجام شد و همگی همگرا شدند.', '',
        '```powershell', 'python -m unittest discover -s tests -q',
        'python src/balanced_genre_feasibility.py', 'python src/report_balanced_feasibility.py', '```', '',
        'پیش‌بینی‌های فردی و نقاط بازیابی در `data/analysis/balanced-genre-feasibility-v1/` محلی‌اند. برای تغییر کد، ورودی یا طرح، باید پوشهٔ اجرای تازه ساخت و نتیجهٔ تازه را با نقاط بازیابی قبلی ترکیب نکرد. همهٔ مدل‌ها، وزن‌های هر بخش، تغییر معیارها و تغییر شناسایی و دقت نسبت‌دادن هر گروه در جدول‌های این پوشه ثبت‌اند.', '',
    ]
    (PUB / 'report-fa.md').write_text('\n'.join(text), encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(3)
    for ax, metric, title in zip(axes, ['recall', 'precision'],
                                ['How many actual members does the model find?', 'How often is a predicted group correct?']):
        for j, (model, label, color) in enumerate([('combined','Unweighted','#8193a0'),
                                                  ('combined_balanced','Balanced training','#267e86')]):
            values = [float(group[(model, g)][metric]) for g in GROUPS]
            bars = ax.bar(x + (j - .5) * .36, values, width=.36, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
        ax.set_xticks(x, GROUPS)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.capitalize())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Same combined model, same held-out authors and books', fontsize=14, weight='bold')
    fig.tight_layout(rect=(0,.07,1,.93))
    fig.savefig(PUB / 'balance_effect.png', dpi=170)
    plt.close(fig)

    paths = [p for p in PUB.glob('*') if p.is_file() and p.name != 'manifest.json']
    paths += list(OUT.glob('*_predictions.csv'))
    paths += [ROOT / 'src/balanced_genre_feasibility.py', Path(__file__), ROOT / 'tests/test_balanced_genre_feasibility.py',
              ROOT / 'src/assess_three_genres.py', ROOT / 'src/predictive_feasibility.py',
              ROOT / 'src/phase2_patterns.py', ROOT / 'src/style_indicators.py', ROOT / 'src/writing_style.py',
              ROOT / 'data/analysis/phase2-final/sample.jsonl', ROOT / 'data/analysis/phase2-final/independent.csv',
              ROOT / 'data/analysis/phase2-final/first_review_style.csv.gz',
              ROOT / 'data/analysis/tri-genre-feasibility-v1/all_predictions.csv',
              ROOT / 'data/analysis/tri-genre-feasibility-v1/thriller_input_predictions.csv']
    manifest = {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p.read_bytes()).hexdigest() for p in paths}
    (PUB / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Saved balance report, figure and {len(manifest)} hashes.')


if __name__ == '__main__':
    main()
