"""Build the Phase 1 PDF from aggregate outputs of src/audit.py."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from audit import AMBIGUOUS, GENRE_RULES, NO_METADATA, UNMAPPED


INK = "#17283a"
MUTED = "#465e71"
BLUE = "#216575"
PALE = "#eaf2f3"
GRID = "#d8e4e8"
GENRES = list(GENRE_RULES)
GROUPS = GENRES + [AMBIGUOUS, UNMAPPED, NO_METADATA]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": MUTED, "axes.edgecolor": GRID, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
})


def pct(count, total):
    return f"{100 * count / total:.2f}%" if total else "—"


def quantile(histogram, fraction):
    if not histogram:
        return 0
    rank = max(1, ceil(sum(histogram.values()) * fraction))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= rank:
            return value
    raise ValueError("Invalid histogram")


def histogram(raw):
    return Counter({int(k): v for k, v in raw.items()})


def timestamp(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class Page:
    def __init__(self, pdf, number, section, title):
        self.pdf = pdf
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        self.fig.text(0.08, 0.955, f"DATA ANALYSIS  /  PHASE 1  /  {section.upper()}",
                      fontsize=9, color=BLUE, weight="bold")
        self.fig.text(0.08, 0.917, title, fontsize=21, weight="bold")
        self.fig.text(0.08, 0.037, "Amazon Reviews’23  /  Books", fontsize=8, color=MUTED)
        self.fig.text(0.92, 0.037, f"{number:02d}", ha="right", fontsize=8, color=MUTED)
        self.y = 0.885

    def heading(self, title):
        self.fig.text(0.08, self.y, title, va="top", fontsize=12, weight="bold", color=BLUE)
        self.y -= 0.027

    def paragraph(self, body, *, small=False):
        size = 8.5 if small else 10
        line_step = 0.016 if small else 0.019
        # Conservative line width for the report font and page margins.
        lines = textwrap.wrap(body, width=109 if small else 91,
                              break_long_words=False, break_on_hyphens=False)
        for line in lines:
            self.fig.text(0.08, self.y, line, va="top", fontsize=size)
            self.y -= line_step
        self.y -= 0.014

    def table(self, columns, rows, widths=None):
        height = (len(rows) + 1) * 0.025
        self.y -= height
        ax = self.fig.add_axes([0.08, self.y, 0.84, height])
        ax.axis("off")
        grid = ax.table(cellText=rows, colLabels=columns, cellLoc="left",
                        colLoc="left", bbox=[0, 0, 1, 1], colWidths=widths)
        grid.auto_set_font_size(False)
        grid.set_fontsize(8.4)
        for (row, _), cell in grid.get_celld().items():
            cell.set_edgecolor("white")
            cell.PAD = 0.11
            cell.set_facecolor(BLUE if row == 0 else (PALE if row % 2 else "#f6f8f9"))
            cell.set_text_props(color="white" if row == 0 else INK,
                                weight="bold" if row == 0 else "normal")
        self.y -= 0.025

    def chart(self, title, height=0.20):
        top = self.y - 0.03
        ax = self.fig.add_axes([0.16, top - height, 0.74, height])
        ax.set_title(title, loc="left", fontsize=10, weight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, alpha=0.8)
        ax.set_axisbelow(True)
        self.y = top - height - 0.07
        return ax

    def close(self):
        if self.y < 0.060:
            raise ValueError(f"Report content exceeds page height ({self.y:.3f})")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def validate(audit):
    review = audit["reviews"]
    meta = audit["metadata"]
    assert sum(g["rows"] for g in review["availability"].values()) == (
        review["quality"]["rows"] - review["quality"].get("invalid_json", 0)
        - review["quality"].get("non_object", 0)
    )
    assert sum(g["rows"] for g in meta["groups"].values()) == (
        meta["quality"]["rows"] - meta["quality"].get("invalid_json", 0)
        - meta["quality"].get("non_object", 0)
    )
    assert sum(meta["unique_genre_ids"].values()) == meta["quality"]["usable_parent_ids"]
    assert sum(sum(group.values()) for group in review["ratings_by_group"].values()) == (
        sum(review["rating_counts"].values())
    )
    assert sum(review["sample_user_activity"].values()) == (
        review["cohort"]["stages"]["sampled_users"]
    )
    for threshold, distribution in review["cohort"]["labels_per_user"].items():
        assert sum(distribution.values()) == review["cohort"]["stages"].get("eligible_users", 0)
        assert sum(int(k) * v for k, v in distribution.items()) == (
            sum(review["cohort"]["genre_counts"][threshold].values())
        )


def overview(pdf, review, meta):
    p = Page(pdf, 1, "Overview", "Predicting Preferred Book Genres")
    p.paragraph("Can style, vocabulary and meaning help predict users’ preferred book genres?")
    p.heading("1. Research question and decision")
    p.paragraph(
        "The research question concerns stylistic, lexical and semantic features of Books reviews. "
        "Phase 1 checks whether the data contain enough usable writing and separate rating evidence "
        "to study genre preference at the account level."
    )
    p.heading("2. Alternative questions considered")
    p.table(["Candidate", "Reason for the Phase 1 decision"], [
        ["Changes in positive ratings over time", "Feasible descriptively. It studies time and rating mix."],
        ["Writing style across product categories", "Feasible with topic controls. It describes group differences."],
        ["Style, vocabulary and semantics for preference", "Selected. Earlier reviews provide the input text."],
    ], [0.43, 0.57])
    p.paragraph(
        "The project card records ten initial ideas. Question 10 uses the repeated reviews and "
        "genre linkage established in the feasibility study. The current data check is to find users "
        "who have both sufficient review text and separate ratings for genre-linked books."
    )
    p.heading("3. Empirical scope")
    review_rows = review["quality"]["rows"]
    matched = sum(review["availability"].get(g, {}).get("rows", 0) for g in GENRES)
    eligible = review["cohort"]["stages"].get("eligible_users", 0)
    p.table(["Evidence from this audit", "Count"], [
        ["Review-file rows", f"{review_rows:,}"],
        ["Metadata-file rows", f"{meta['quality']['rows']:,}"],
        ["Reviews linked to the mapped genres", f"{matched:,} ({pct(matched, review_rows)})"],
        ["Eligible users in the separated-window pilot", f"{eligible:,}"],
    ], [0.74, 0.26])
    p.paragraph(
        "This phase measures data quality, modality availability and the number of users "
        "with both forms of evidence."
    )
    p.close()


def data_page(pdf, review, meta):
    p = Page(pdf, 2, "Data and protocol", "Data Sources and Linkage")
    p.heading("4. Provenance and observational unit")
    start, end = map(timestamp, review["valid_timestamp_range_ms"])
    p.paragraph(
        "McAuley Lab assembled Amazon Reviews’23 from Amazon reviews and item metadata. The "
        "Books subset contains self-selected reviews and associated catalog records. A review row "
        f"records an account–product interaction. Valid review dates span {start} to {end} UTC. "
        "The data describe who reviewed, what was reviewed, and the posted rating. The date is the "
        "review timestamp. Catalog price is an item snapshot and may differ from transaction price."
    )
    rs = review["schema_sample"]
    ms = meta["schema_sample"]
    p.table(["Source", "Keys and main fields", "Types in this audit"], [
        ["Books.jsonl.gz", "user_id, asin, parent_asin, title, text", "String"],
        ["Review interactions", "rating, helpful_vote, timestamp", "Numeric, integer, UTC ms"],
        ["Review interactions", "verified_purchase, images", "Boolean, list"],
        ["meta_Books.jsonl.gz", "parent_asin, title, categories", "String, string, list"],
        ["Catalog metadata", "price, images, description", "Numeric or null, list, list"],
    ], [0.26, 0.48, 0.26])
    p.paragraph(
        f"The sampled schemas contain {len(rs['fields'])} review fields across {rs['objects']:,} "
        f"objects and {len(ms['fields'])} metadata fields across {ms['objects']:,} objects. Full "
        "critical-field counts and the sampled type inventories are stored in phase1_audit.json.",
        small=True,
    )
    p.heading("5. From raw records to cohort")
    p.table(["Input", "Join and screening", "User-level output"], [
        ["Review rows", "parent_asin → catalog lookup", "Prior writing and later ratings"],
        ["Metadata rows", "Exact category rules → genre", "Five weak preference labels"],
    ], [0.25, 0.39, 0.36])
    p.paragraph(
        "Seven groups use exact Amazon category strings. Philosophy uses the Philosophy category. "
        "Science requires Science & Math plus a specified scientific subcategory, such as Physics "
        "or Biological Sciences. Science Fiction uses its own categories. Multiple matched groups "
        "stay ambiguous. The coverage of the rules and their agreement with literary genre "
        "are separate questions."
    )
    p.heading("6. Sampling and reproduction")
    p.paragraph(
        "The code streams both archives to EOF. Ratings and modality counts use full scans. A seeded "
        "1% review-row sample supports text and monthly EDA. A separate fixed CRC32 bucket "
        "(1/64 of the hash space) retains all rows for sampled users and supports duplicate checks "
        "and cohort construction. Both are audit samples, with the review file order held fixed."
    )
    p.close()


def quality_page(pdf, review, meta):
    p = Page(pdf, 3, "Quality and modalities", "Data Quality and Coverage")
    q = review["quality"]
    mq = meta["quality"]
    groups = review["availability"]
    all_fields = Counter()
    for counts in groups.values():
        all_fields.update(counts)
    sample_rows = sum(int(k) * n for k, n in review["sample_user_activity"].items())
    bad_states = ("missing", "blank", "wrong_type")
    missing = lambda key: sum(all_fields[f"{key}:{state}"] for state in bad_states)
    p.heading("7. Full quality audit and decisions")
    p.table(["Check", "Count", "Treatment"], [
        ["Invalid JSON or non-object rows",
         f"{q.get('invalid_json', 0) + q.get('non_object', 0):,} reviews",
         "Skip malformed records"],
        ["Missing, blank or mistyped user / parent IDs",
         f"{missing('user_id'):,} / {missing('parent_asin'):,}", "Exclude from cohort when unusable"],
        ["Unusable rating / timestamp",
         f"{missing('rating') + q.get('rating_out_of_range', 0):,} / "
         f"{missing('timestamp') + q.get('timestamp_out_of_range', 0):,}", "Skip in affected analysis"],
        ["Negative helpful votes",
         f"{q.get('negative_helpful_votes', 0):,}", "Flag for future preprocessing"],
        ["Non-usable review text",
         f"{missing('text'):,}", "Require nonblank body for writing"],
        ["Repeated metadata parent rows / conflicting IDs",
         f"{mq.get('repeated_parent_rows', 0):,} / {mq.get('conflicting_parent_ids', 0):,}",
         "Exclude conflicting audit entries"],
        ["Exact repeated review objects (user sample)",
         f"{q.get('exact_duplicates_in_user_sample', 0):,} / {sample_rows:,}",
         "Deduplicate for cohort analysis"],
    ], [0.49, 0.19, 0.32])
    p.paragraph(
        "An exact duplicate has an identical normalized JSON representation. Metadata conflicts are differences "
        "in mapped group, price availability or product-image availability for a shared parent ID. "
        "The audit retains only integer star levels 1–5 and UTC timestamps from 1996–2023.",
        small=True,
    )
    p.heading("8. Available modalities by mapped genre")
    rows = []
    for group in GROUPS:
        counts = groups.get(group, {})
        total = counts.get("rows", 0)
        rows.append([group, f"{total:,}"] + [
            pct(counts.get(f"{field}:available", 0), total)
            for field in ("text", "images", "product_image", "price")
        ])
    p.table(["Group", "Review rows", "Text", "Review img.", "Product img.", "Price"],
            rows, [0.25, 0.19, 0.13, 0.14, 0.15, 0.14])
    price_shares = [
        (g, groups[g].get("price:available", 0) / groups[g]["rows"])
        for g in GENRES if g in groups
    ]
    if len(price_shares) > 1:
        low = min(price_shares, key=lambda entry: entry[1])
        high = max(price_shares, key=lambda entry: entry[1])
        p.paragraph(
            f"Catalog price coverage among mapped genres ranges from {low[1]:.1%} "
            f"({low[0]}) to {high[1]:.1%} ({high[0]}).",
            small=True,
        )
    p.paragraph(
        "Denominators are review rows in each group. Text must be nonblank, image lists must "
        "be nonempty, and prices must be positive finite numbers. The audit measures listed image "
        "availability. Unlinked rows have no catalog price or image to measure. Comparing mapped "
        "groups shows whether missingness differs with genre and the join. Group differences "
        "are descriptive. The reasons for missingness remain unresolved."
    )
    p.paragraph(
        "Other or unmapped includes out-of-scope categories and insufficiently specific tags. "
        "Ambiguous means multiple genre rules matched. Missing data are recorded separately "
        "from those labeling decisions.",
        small=True,
    )
    p.close()


def eda_page(pdf, review):
    p = Page(pdf, 4, "Exploratory analysis", "Rating, Activity and Time")
    ratings = histogram(review["rating_counts"])
    n = sum(ratings.values())
    p.heading("9. Rating imbalance")
    p.paragraph(
        f"Across {n:,} valid ratings, the mean is {sum(k*v for k,v in ratings.items()) / n:.2f} "
        f"and the median is {quantile(ratings, .5)} stars. Ratings 4–5 account for "
        f"{pct(ratings[4] + ratings[5], n)} of valid ratings. Positive genre evidence should "
        "therefore be interpreted against genre and account activity rather than as direct preference."
    )
    ax = p.chart("Full scan · percentage of valid reviews by star rating", 0.17)
    bars = ax.bar(range(1, 6), [100 * ratings[i] / n for i in range(1, 6)], color=BLUE)
    ax.set_xticks(range(1, 6))
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, max(b.get_height() for b in bars) * 1.18)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    p.heading("10. Activity and changes over time")
    activity = histogram(review["sample_user_activity"])
    total = sum(activity.values())
    repeat = sum(v for k, v in activity.items() if k >= 10)
    p.paragraph(
        f"In the user sample, {pct(activity[1], total)} have one review and "
        f"{pct(repeat, total)} have at least ten. Median reviews per user are "
        f"{quantile(activity, .5)} (IQR {quantile(activity, .25)}–{quantile(activity, .75)}). "
        "The writing requirement concentrates the study on unusually active accounts."
    )
    months = review["sample_month_counts"]
    years = Counter()
    for month, count in months.items():
        years[int(month[:4])] += count
    complete = list(range(min(years) + 1, max(years)))
    ax = p.chart("Review-row sample · yearly volume across complete calendar years", 0.17)
    ax.plot(complete, [years[i] for i in complete], color=BLUE, linewidth=2)
    ax.set_ylabel("Sample reviews")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
    p.paragraph(
        "Endpoint years are omitted because dataset coverage can be partial. The chart shows "
        "how much review activity is observable in each complete year.",
        small=True,
    )
    p.close()


def text_page(pdf, review, meta):
    p = Page(pdf, 5, "Exploratory analysis", "Text Length and Genre Signals")
    flags = review["sample_text_flags"]
    text_counts = review["sample_words_by_rating"]
    combined = Counter()
    for values in text_counts.values():
        combined.update(histogram(values))
    total = sum(combined.values())
    short = sum(v for k, v in combined.items() if k < 20)
    long = sum(v for k, v in combined.items() if k >= 500)
    p.heading("11. Writing sample quality")
    p.paragraph(
        f"Among {total:,} sampled nonblank bodies, the median length is "
        f"{quantile(combined, .5)} tokens (IQR {quantile(combined, .25)}–"
        f"{quantile(combined, .75)}). {pct(short, total)} contain fewer than 20 tokens "
        f"and {pct(long, total)} contain at least 500. {pct(flags['non_ascii'], total)} contain "
        f"non-ASCII characters and {pct(flags['contains_non_latin_letter'], total)} contain "
        f"a non-Latin letter. {flags['contains_replacement_character']:,} contain a Unicode "
        "replacement character. The writing-system screen motivates a later language audit."
    )
    ax = p.chart("Review-row sample · median review length and interquartile range by rating", 0.17)
    per_star = [histogram(text_counts.get(str(star), {})) for star in range(1, 6)]
    medians = [quantile(h, .5) for h in per_star]
    lows = [m - quantile(h, .25) for m, h in zip(medians, per_star)]
    highs = [quantile(h, .75) - m for m, h in zip(medians, per_star)]
    ax.errorbar(range(1, 6), medians, yerr=[lows, highs], fmt="o", capsize=6, color=BLUE)
    ax.set_xticks(range(1, 6))
    ax.set_ylabel("Body tokens")
    p.paragraph(
        f"The rating-group medians range from {min(medians)} to {max(medians)} tokens. "
        "The interquartile ranges show how much length varies within each rating group.",
        small=True,
    )

    p.heading("12. Genre coverage and ratings")
    per_group = review["ratings_by_group"]
    rows = []
    for genre in GENRES:
        h = histogram(per_group.get(genre, {}))
        rows.append([
            genre, f"{meta['unique_genre_ids'].get(genre, 0):,}",
            f"{sum(h.values()):,}", pct(h[4] + h[5], sum(h.values())),
        ])
    p.table(["Genre", "Mapped parent IDs", "Valid reviews", "4–5 star share"],
            rows, [0.30, 0.25, 0.25, 0.20])
    p.paragraph(
        "Genre popularity, positivity and modality availability are measured on different "
        "denominators. The table makes differences in coverage and positive ratings visible. "
        "Extreme text lengths can be inspected for quotations or copied content."
    )
    p.close()


def cohort_page(pdf, review):
    p = Page(pdf, 6, "Data feasibility", "User Coverage")
    cohort = review["cohort"]
    stages = cohort["stages"]
    p.heading("13. Candidate cohort on sampled accounts")
    p.paragraph(
        "The counts below use the exploratory audit’s settings. It counts writing from 2015–2019 "
        "and rating evidence from 2020–2022. The writing check requires three distinct books with "
        "at least 20 tokens each and 150 tokens in total. The rating check requires three distinct, "
        "genre-linked books with valid ratings that were absent from the earlier writing window. "
        "For repeated ratings, the audit keeps the latest one in the rating window."
    )
    p.table(["User-sample condition", "Count"], [
        [label, f"{stages.get(key, 0):,}"] for key, label in [
            ("sampled_users", "All sampled users"),
            ("adequate_style", "Enough earlier writing"),
            ("adequate_label_exposure", "Enough later mapped books"),
            ("eligible_users", "Both conditions"),
        ]
    ], [0.76, 0.24])
    labels = cohort["genre_counts"]
    p.table(["Positive label", "≥1 book", "≥2 books", "≥3 books"], [
        [genre] + [f"{labels[k].get(genre, 0):,}" for k in ("1", "2", "3")]
        for genre in GENRES
    ], [0.34, 0.22, 0.22, 0.22])
    label_sizes = histogram(cohort["labels_per_user"]["2"])
    p.paragraph(
        f"At the two-book threshold, {label_sizes[0]:,} users have zero observed positive "
        f"labels and {sum(v for k, v in label_sizes.items() if k > 1):,} have multiple. "
        "The table counts distinct positively rated books within each genre and shows why "
        "threshold choice affects class support. These are sample counts."
    )
    p.heading("14. Separation of evidence")
    p.paragraph(
        "A review used to establish genre preference should stay separate from the text used "
        "to describe that user’s writing. Reviews from the same account should stay together "
        "whenever the data are divided. The share of users with positive evidence in each genre "
        "is a simple reference for describing the resulting cohort."
    )
    p.close()


def limitations_page(pdf):
    p = Page(pdf, 7, "Limitations and references", "Interpretation of the Audit")
    p.heading("15. What the Phase 1 data support")
    p.paragraph(
        "The raw files provide repeated writing and linked ratings. The exploratory count shows "
        "how many sampled users meet both evidence requirements under the settings used. "
        "Preference is inferred from self-selected reviews, conditional "
        "on observed follow-up. Genre, language, level of activity and platform behavior constrain "
        "how widely the results can generalize."
    )
    p.heading("16. Open data checks")
    p.paragraph(
        "Use the Amazon categories and the documented mapping rules as the source of genre "
        "labels. Quantify ambiguous and unmapped records. Audit language, boilerplate and "
        "quotations. Check how the number "
        "of usable users in each genre changes with the minimum amount of text and rating evidence."
    )
    p.heading("17. Responsible use")
    p.paragraph(
        "Use account identifiers only to link and split records. Share aggregate statistics and "
        "code with the report. Document missingness, review-selection bias and weak labels. "
        "The audit describes observed reviewing behavior. Catalog attributes were collected after "
        "some review events. "
        "The audit uses them for outcome labels and coverage checks."
    )
    p.heading("References and reproducibility")
    p.paragraph(
        "McAuley Lab. Amazon Reviews’23, dataset documentation and Books downloads. "
        "https://amazon-reviews-2023.github.io/main.html",
        small=True,
    )
    p.paragraph(
        "Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. (2024). Bridging Language "
        "and Items for Retrieval and Recommendation. https://arxiv.org/abs/2403.03952v1",
        small=True,
    )
    p.paragraph(
        "Course project brief, Data Analysis, Fall 1405, sections 2, 4 and 5. Audit sources "
        "and commands are in the project README. The complete aggregate evidence is in "
        "reports/phase1_audit.json.",
        small=True,
    )
    p.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/phase1_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase1.pdf"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    validate(data)
    review, meta = data["reviews"], data["metadata"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".tmp")
    with PdfPages(partial, metadata={"Title": "Amazon Books Genre Preference | Phase 1"}) as pdf:
        overview(pdf, review, meta)
        data_page(pdf, review, meta)
        quality_page(pdf, review, meta)
        eda_page(pdf, review)
        text_page(pdf, review, meta)
        cohort_page(pdf, review)
        limitations_page(pdf)
    partial.replace(args.output)


if __name__ == "__main__":
    main()
