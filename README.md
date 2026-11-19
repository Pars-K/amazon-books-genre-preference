# پیش‌بینی گرایش ژانری از متن نقد کتاب

پروژهٔ درس تحلیل داده با مجموعهٔ Amazon Reviews’23 Books. سؤال ۱۰ می‌پرسد از یک نقد تا چه حد می‌توان گرایش نویسنده را از روی نوشتارش پیش‌بینی کرد. گرایش مرجع را از سابقهٔ نقد کتاب‌های دیگر همان فرد برآورد می‌کنیم.

| بخش | گزارش و کد |
|---|---|
| فاز ۱، شناخت داده و انتخاب سؤال | [گزارش فاز ۱](reports/phase1.pdf) · [ده سؤال پروژه](PROJECT_CARD.md) · [کد ممیزی داده](src/audit.py) |
| فاز ۲، استخراج ویژگی و کشف الگو | [گزارش فاز ۲](reports/phase2.pdf) · [روش اجرا، شکل‌ها و جدول‌ها](reports/phase2/README.md) · [کد اجرای تحلیل](src/run_phase2.py) |
| شواهد تکمیلی برای تعیین دامنه | [اعتبار الگوها](reports/phase2/validity/report-fa.md) · [کفایت داده](reports/phase2/adequacy/report-fa.md) · [آزمایش هفت و دو گروه](reports/feasibility/report-fa.md) |
| دامنهٔ منتخب برای ادامه | [هدف سه‌گروهی](CURRENT_GOAL.md) · [ارزیابی سه‌ژانری](reports/feasibility-three/report-fa.md) · [اثر وزن‌دهی](reports/feasibility-balanced/report-fa.md) |

نتایج تکمیلی از دادهٔ توسعه به دست آمده‌اند. برای بازتولید، پس از دریافت داده و نصب وابستگی‌ها با `python -m pip install -r requirements.txt`، مراحل آماده‌سازی زیر و سپس [دستورهای فاز ۲](reports/phase2/README.md) را اجرا کنید. استفاده از ابزار هوش مصنوعی در [توضیحات روش کار](AI_USAGE.md) ثبت شده است.

## Set up the local data

Use Python 3.14. The audit scripts use the Python standard library.

The local input archives in `data/raw/` are `Books.jsonl.gz` (6,216,820,644 bytes) and `meta_Books.jsonl.gz` (4,942,125,770 bytes). To set up another copy of the workspace, download both files from the Books row and place them in that directory. [Official dataset page](https://amazon-reviews-2023.github.io/main.html)

## Reproduce the expanded genre coverage counts

Run this command from the project folder to count coverage for the seven categories. Philosophy uses the exact `Philosophy` category. Science requires the `Science & Math` branch and an accepted natural or applied science subcategory. The code counts mapped books, linked reviews, and users with distinct positively rated books in the two added categories.

```powershell
python src/genre_coverage.py
```

The results are in `data/genre_coverage/`. The command writes the book-level mapping and aggregate metadata and review summaries. The five-category mapping in `data/feasibility/` belongs to the original feasibility run.

## Reproduce the review-text audit

After generating the seven-category mapping, run `python src/review_text_audit.py`. It samples review rows across the entire Books archive with a fixed seed and saves text-quality and length statistics to `data/genre_coverage/text_summary.json`.

## Reproduce the structural quality audit

Run `python src/review_quality_audit.py` from the project folder. It scans the complete metadata archive, uses the same 2% review-row sample as the text audit for field quality and modality availability, and uses all review rows from a fixed user bucket for duplicate checks. The aggregate outputs are `data/genre_coverage/catalog_quality.json` and `data/genre_coverage/review_quality.json`.

## Explore genre engagement and satisfaction per user

The working definition now separates repeated genre engagement from satisfaction with individual books. The following commands process complete user histories across the entire review archive, collapse repeated user/book pairs, and describe distinct-book counts, genre shares, review-date continuity, and ratings. They do not assign preference labels.

The earlier positive-rating counts and the cohort logic in `src/audit.py` document the original feasibility definition. They do not implement the revised preference definition or the descriptive profiling protocol below.

```powershell
python src/user_genre_profiles.py --expected-rows 29475453
python src/report_user_genres.py
```

User-level and user/genre CSVs stay locally in the ignored `data/user_genre_profiles/` directory with pseudonymous user keys. Only aggregate results and figures are written to `reports/genre_preference/`. Hash partitioning merges noncontiguous histories without loading all users into memory. Temporary partitions are removed after successful completion. Use a fresh `--output-dir` and `--report-dir` to rerun; then pass that report directory's `summary.json` to `src/report_user_genres.py --summary`.

The profiling scan saves a durable checkpoint every million source rows. If interrupted, rerun the same profiling command with `--resume`. It verifies the source size/modification time and mapping hash, truncates only its temporary partitions to their checkpointed sizes, and continues. A completed partitioning stage can be reused without rescanning the source. Pass a custom user/genre CSV to the report script with `--profiles` when using a different output directory.

Each `parent_asin` counts once per user. Its earliest valid review date supplies the event date; its latest valid rating supplies satisfaction, with the lower rating retained on a tied timestamp. A repeated review of the same book does not count as a new book or a later return to the genre. Shares include both the seven-genre denominator and the user's full observed book history. These whole-history profiles are exploratory and must not be used directly as features for predicting future preference.

Run the data-processing checks with `python -m unittest discover -s tests -v`.

## Count feasible genre cohorts and available writing

After the full user profiling run, execute `python src/cohort_feasibility.py`. It
measures body-text volume for users with at least two distinct books in a mapped
genre, counting the longest body once per user/book. This prevents repeated
records and revisions of the same book review from inflating available writing.
Word counts reuse the text-audit tokenizer.

The aggregate Persian report is `reports/cohort_feasibility/report-fa.md`.
`cohort_stages.csv` contains cumulative screening counts for one illustrative
scenario. `cohort_sensitivity.csv` varies book count, genre share, mapping
coverage, continuity, and text volume. It includes both share denominators
(all books and mapped books) and both text scopes (all history and books outside
the target genre). The latter includes unmapped books. These exploratory
thresholds do not assign validated interest labels or establish that independent
writing samples or reviews of a common genre are available.

Private text measurements and temporary partitions stay under
`data/cohort_feasibility/`. A completed scan is reused after input verification.
Use a fresh `--data-dir` for changed inputs. The run uses all observed history;
separate evidence for cohort assignment and linguistic comparison must be
constructed before the final analysis.

## Reproduce the initial Phase 2 data snapshot

Run `python src/freeze_analysis_data.py` after the profiling and text-feasibility
runs. The script reads `reports/phase2/data_v1_rules.json`, scans the full review
archive and writes `users.csv`, `memberships.csv`, and `reviews.jsonl.gz` under
`data/analysis/eda-v1/`. Every selected user/book pair contributes its longest
body, with deterministic content-based tie-breaking. It drops bodies shorter
than 20 tokens and then rechecks that users have at least three eligible reviews
and 1,000 tokens. The cohort criteria use a genre share of at least 50% of all
distinct reviewed books, at least three genre books, and documented continuity.

The snapshot preserves original body text for later stylistic features. Ratings
and timestamps that fail validation are stored as null. All seven genres remain
available, and multiple memberships use the same user key and split. A seeded
hash assigns complete users to train, validation and test with expected ratios
70/15/15. Fit learned preprocessing on train. Language identification and HTML
handling are documented follow-up preparation tasks.

The aggregate report and manifest are in `reports/phase2/`. The manifest records
input and output SHA-256 hashes, rules, source counts and runtime versions. A
sealed snapshot rejects overwriting. Choose a new `--output-dir` and
`--report-dir` for a revised version. An unfinished run can be retried with the
same command and matching inputs. It reuses a completed scan or repeats an
interrupted scan with idempotent upserts.

## Measure writing style in a review

Run `python src/writing_style.py --text "I'm glad. Are you?"` to obtain counts
and length-normalized features from a single English review. For the aggregate
measurement pilot, run `python src/writing_style.py --pilot data/analysis/eda-v1/reviews.jsonl.gz`.
The [measurement definitions and pilot results](reports/phase2/writing-style-fa.md)
describe the treatment of sentence boundaries, quotes, and English contractions.
