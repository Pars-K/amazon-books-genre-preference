"""Calculate descriptive Spearman correlations and plot user-level quartiles."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from audit import GENRE_RULES
from style_indicators import METRICS


def correlations(private, reports, history):
    users = pd.read_csv(private / "user_style.csv", dtype={"user_key": str})
    users = users[users.split == "train"].set_index("user_key")
    shares = pd.DataFrame(0.0, index=users.index, columns=list(GENRE_RULES))
    observed = set()
    rows = []
    with history.open(encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            u = item["user_key"]
            if u in shares.index:
                rows.append((u, item["genre"], float(item["share_all_books"])))
                observed.add(u)
    if observed != set(shares.index):
        raise ValueError("Missing genre histories for one or more development users")
    for u, genre, value in rows:
        shares.at[u, genre] = value
    common = pd.read_csv(private / "user_book_genre_style.csv.gz", dtype={"user_key": str})
    common = common[common.split == "train"]
    scopes = [("all_reviews", users)]
    scopes.extend(("book=" + str(g), df.set_index("user_key")) for g, df in common.groupby("book_genre"))
    with (reports / "style_correlations.csv").open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["scope", "genre_share", "metric", "valid_users", "nonzero_share_users", "spearman_rho"])
        for scope, df in scopes:
            for metric in METRICS:
                y = df[metric].dropna()
                ranks = y.rank()
                active = shares.loc[y.index]
                # Pearson correlation of average ranks handles tied rates and zero shares.
                variable = active.loc[:, active.nunique() > 1]
                coeffs = variable.rank().corrwith(ranks) if ranks.nunique() > 1 else {}
                for genre in shares:
                    val = coeffs.get(genre)
                    out.writerow([scope, genre, metric, len(ranks), int((active[genre] > 0).sum()),
                                  round(val, 6) if pd.notna(val) else ""])


def chart(reports):
    data = pd.read_csv(reports / "group_style.csv")
    data = data[data.scope == "all_reviews"].set_index(["preference_group", "metric"])
    labels = list(GENRE_RULES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    fig.subplots_adjust(left=.22, right=.97, top=.77, bottom=.15, wspace=.19)
    for ax, name, title, color in zip(axes, ["mean_sentence_words", "first_singular_per_1000_words"],
                                    ["Words per sentence", "First-person singular per 1,000 words"],
                                    ["#176A72", "#9D4C2B"]):
        for i, g in enumerate(labels):
            r = data.loc[(g, name), :]
            ax.plot([r.p25, r.p75], [i, i], lw=3.2, color=color, alpha=.7, solid_capstyle="round")
            ax.scatter(r["median"], i, s=45, color=color, edgecolor="white", linewidth=.7, zorder=3)
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.grid(axis="x", color="#E1E5E8", lw=.8)
        ax.set_axisbelow(True)
    ns = data.xs("mean_sentence_words", level="metric")["users"]
    axes[0].set_yticks(range(len(labels)), [f"{g}  ({int(ns[g]):,})" for g in labels])
    axes[0].set_ylim(len(labels)-.55, -.45)
    fig.text(.03,.93,"How does review style vary across preference groups?",fontsize=16,weight="bold")
    fig.text(.03,.865,"Development users · Dot = median user · Line = middle 50% of users",fontsize=10,color="#475569")
    fig.savefig(reports / "user_style_intervals.png", dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private", type=Path, default=Path("data/analysis/style-v1"))
    p.add_argument("--reports", type=Path, default=Path("reports/phase2/style-v1"))
    p.add_argument("--history", type=Path, default=Path("data/user_genre_profiles/user_genre_profiles.csv"))
    args = p.parse_args()
    correlations(args.private, args.reports, args.history)
    chart(args.reports)
