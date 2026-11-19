"""Create question-driven figures from the completed development analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit import GENRE_RULES
from phase2_interpret import load
from phase2_sample import OUT, PUB

INK, BLUE, ORANGE = "#173142", "#237C83", "#BA632F"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False,"axes.titlelocation":"left",
                     "axes.labelcolor":INK,"text.color":INK,"axes.titlesize":12,
                     "savefig.facecolor":"white"})


def save(fig, name):
    fig.savefig(PUB/(name+".png"), dpi=165, bbox_inches="tight")
    plt.close(fig)


def repeatability():
    data = pd.read_csv(PUB/"review_repeatability.csv")
    data = data[data.scope=="all"].set_index("metric")
    names = {
        "words_per_review":"Review length", "mean_sentence_words":"Sentence length",
        "sd_sentence_words":"Variation of sentence length", "first_singular_per_1000_words":"First-person singular",
        "contractions_per_1000_words":"Contractions", "exclamations_per_1000_words":"Exclamation marks",
        "vocabulary_diversity_20_pct":"20-word vocabulary diversity",
        "reading_experience_sentence_pct":"Reading experience phrases",
        "plot_description_sentence_pct":"Plot description phrases",
    }
    fig, ax = plt.subplots(figsize=(9,4.9))
    fig.subplots_adjust(left=.34,bottom=.15,top=.79)
    values = data.loc[list(names),"repeat_spearman"]
    ax.barh(range(len(values)),values,color=[BLUE]*6+[ORANGE]*3,height=.63)
    ax.set_yticks(range(len(values)),list(names.values()))
    ax.invert_yaxis()
    ax.set_xlim(0,.7)
    ax.set_xlabel("Spearman correlation between two separate reviews by the same author")
    for i,v in enumerate(values):
        ax.text(v+.012,i,f"{v:.2f}",va="center",fontsize=9)
    fig.suptitle("Which measures repeat across an author's reviews?",x=.03,ha="left",fontsize=15,weight="bold")
    save(fig,"repeatability")


def selection():
    d = pd.read_csv(PUB/"cluster_selection.csv")
    fig, axes = plt.subplots(1,2,figsize=(9,3.7))
    for ax,key,title in zip(axes,["silhouette","seed_ari"],["Separation on a fixed review sample","Agreement across random seeds"]):
        ax.plot(d.k,d[key],marker="o",color=BLUE)
        ax.set_xlabel("Number of clusters")
        ax.set_title(title)
        ax.set_xticks(d.k)
    axes[0].set_ylabel("Median silhouette")
    axes[1].set_ylabel("Median adjusted Rand index")
    fig.tight_layout()
    save(fig,"cluster_selection")


def map_and_length(df):
    coordinates = np.load(OUT/"lexical_fit.npz")["coordinates"]
    rng = np.random.default_rng(440)
    take = np.sort(rng.choice(len(df),min(3000,len(df)),replace=False))
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for g,color in [(1,BLUE),(2,ORANGE)]:
        ix = take[df.cluster.iloc[take].to_numpy()==g]
        axes[0].scatter(coordinates[ix,0],coordinates[ix,1],s=7,alpha=.3,c=color,label=f"Cluster {g}",rasterized=True)
        vals = np.sort(df.loc[df.cluster==g,"review_tokens"])
        axes[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),color=color,label=f"Cluster {g}")
    axes[0].set_title("Two-dimensional lexical projection")
    axes[0].set_xlabel("SVD component 1")
    axes[0].set_ylabel("SVD component 2")
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].set_title("How strongly do the clusters track length?")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Review length in raw word tokens")
    axes[1].set_ylabel("Fraction of reviews")
    fig.text(.03,.01,"Projection shows 3.2% of TF-IDF squared norm. Clustering uses 48 dimensions.",fontsize=9,color="#4A6070")
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig,"clusters_and_length")


def label_retention():
    d = pd.read_csv(PUB/"label_sensitivity.csv").set_index("genre").loc[list(GENRE_RULES)]
    fig, ax = plt.subplots(figsize=(8,4))
    vals = 100*d.retained/d.original
    ax.barh(d.index,vals,color=BLUE)
    ax.invert_yaxis()
    ax.set_xlim(0,115)
    for i,v in enumerate(vals):
        ax.text(v+1,i,f"{v:.1f}%  ({d.retained.iloc[i]}/{d.original.iloc[i]})",va="center",fontsize=9)
    ax.set_xlabel("Original genre memberships retained (%)")
    ax.set_title("Does the preference label survive removing the input book?")
    fig.tight_layout()
    save(fig,"label_retention")


def contrasts():
    d = pd.read_csv(PUB/"targeted_common_contrasts.csv")
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    labels = ["All selected thriller reviews","Without quote markers","At least 100 words"]
    for ax,metric,title in zip(axes,["first_singular_per_1000_words","mean_sentence_words"],
                               ["First-person singular per 1,000 words","Mean words per sentence"]):
        data = d[d.metric==metric]
        values = data.difference.to_numpy()
        ax.errorbar(values,np.arange(len(data)),xerr=[values-data.p025.to_numpy(),data.p975.to_numpy()-values],
                    fmt="o",color=BLUE,capsize=4)
        ax.axvline(0,color="#A3AFB5",lw=1)
        ax.set_yticks(range(3),labels if ax==axes[0] else [""]*3)
        ax.invert_yaxis()
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("Romance minus science-fiction median")
    fig.text(.02,.015,"One thriller review per author; other books support preference. Lines show exploratory 95% bootstrap intervals.",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    save(fig,"common_genre_contrasts")


def association():
    d = pd.read_csv(PUB/"cluster_association.csv")
    d = d[d.target=="independent_preference"]
    labels = ["No conditioning","Within input book genre","Within genre and length quintile","Within the same book title"]
    fig, ax = plt.subplots(figsize=(8,3.5))
    pos = np.arange(len(d))
    ax.barh(pos-.15,d.information_bits,height=.28,color=BLUE,label="Observed")
    ax.barh(pos+.15,d.null_median_bits,height=.28,color="#B7C7CE",label="Median under label permutation")
    ax.set_yticks(pos,labels[:len(d)])
    ax.invert_yaxis()
    ax.set_xlabel("Information about the independent preference label (bits)")
    ax.set_title("How much preference information remains after conditioning?")
    ax.legend(frameon=False,fontsize=8)
    fig.text(.01,.015,"The same-title subset contains 163 authors. Each observed value is compared with its own permutation baseline.",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    save(fig,"conditional_association")


def additional_tables(df):
    context = ["rating","review_age_days","log_helpful","review_tokens","eligible_reviews","log_price"]
    main = ["mean_sentence_words","first_singular_per_1000_words","contractions_per_1000_words",
            "exclamations_per_1000_words","vocabulary_diversity_20_pct"]
    rows = []
    for a in context:
        for b in main:
            good = df[[a,b]].dropna()
            rows.append(dict(context=a,style=b,users=len(good),spearman=good[a].rank().corr(good[b].rank())))
    pd.DataFrame(rows).to_csv(PUB/"context_correlations.csv",index=False)
    df.groupby("rating").agg(reviews=("cluster","size"),median_tokens=("review_tokens","median"),
                            median_sentence_words=("mean_sentence_words","median")).to_csv(PUB/"rating_and_length.csv")
    t = df.copy()
    t["length_quintile"] = pd.qcut(df.review_tokens,5,duplicates="drop").astype(str)
    pd.crosstab(t.length_quintile,t.cluster,normalize="index").to_csv(PUB/"cluster_by_length.csv")


if __name__ == "__main__":
    data = load()
    additional_tables(data)
    repeatability()
    selection()
    map_and_length(data)
    label_retention()
    contrasts()
    association()
