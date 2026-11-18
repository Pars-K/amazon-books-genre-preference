# Amazon Books Genre Preference

Phase 1 of the Data Analysis course project, Route B (Amazon Reviews’23). The project asks how well stylistic, lexical, and semantic features from earlier reviews predict later book-genre preferences at the user level. Phase 1 scopes the problem, audits the source data and measures candidate cohort size. It uses observational review data.

## Project files

- [Project card and initial research questions](PROJECT_CARD.md)
- [Current Phase 1 report in Persian](reports/phase1.pdf)
- [Structural quality audit](src/review_quality_audit.py)

## Local feasibility archive

The original executed notebook is preserved locally at `notebooks/feasibility_exploration.ipynb`. It records the exploratory counts and contains the original execution outputs. The two downloaded archives are in `data/raw/`. The user-count, review-distribution, and genre-mapping CSVs are in `data/feasibility/`. Git ignores the entire `data/` directory, including generated summaries, and the archived notebook. Use the commands below to regenerate the summaries locally.

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
