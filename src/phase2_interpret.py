"""Interpret clusters and independent reviewer-group associations on the development sample."""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from audit import GENRE_RULES
from phase2_patterns import ari, choose
from phase2_sample import OUT, PUB, write_rows
from style_indicators import METRICS


def load():
    a = pd.read_csv(OUT/"assignments.csv",dtype={"user_key":str})
    for name in ("context.csv","independent.csv","first_review_style.csv.gz"):
        b = pd.read_csv(OUT/name,dtype={"user_key":str},keep_default_na=False)
        if name.startswith("first_"):
            b = b[["user_key",*METRICS,"quote_reviews","no_english_anchor_reviews"]]
        a = a.merge(b,on="user_key",validate="one_to_one",how="left",suffixes=("","_context"),sort=False)
    if not a.parent_asin.equals(a.parent_asin_context):
        raise ValueError("Context attached to a different book")
    for name in [*METRICS,"rating","review_age_days","helpful_votes","log_helpful","log_price","price",
                 "review_tokens","eligible_reviews","quote_reviews","no_english_anchor_reviews"]:
        a[name] = pd.to_numeric(a[name],errors="coerce")
    a["single"] = a.independent.where(a.independent.isin(GENRE_RULES),"")
    return a


def structured(df):
    # Scale for optional unsupervised comparisons. Raw nullable data remain in context.csv.
    numeric = ["rating","review_age_days","log_helpful","log_price","review_tokens","eligible_reviews"]
    values = df[numeric].copy()
    for col in ("review_tokens","eligible_reviews"):
        values[col] = np.log1p(values[col])
    frame, params = {}, {}
    for c in numeric:
        med = values[c].median()
        iqr = values[c].quantile(.75)-values[c].quantile(.25)
        scale = float(iqr) if iqr>0 else 1.0
        frame[c] = (values[c].fillna(med)-med)/scale
        frame[c+"_missing"] = values[c].isna().astype(int)
        params[c] = dict(median=float(med),scale=scale,missing=int(values[c].isna().sum()))
    verified = df.verified.astype(str).str.lower().map({"true":1,"false":0})
    frame["verified"] = verified.fillna(0)
    frame["verified_missing"] = verified.isna().astype(int)
    for g in [*GENRE_RULES,"UNMAPPED"]:
        frame["book="+g] = (df.book_genre==g).astype(int)
    pd.DataFrame(frame,index=df.index).set_index(df.user_key).to_csv(OUT/"structured_features.csv.gz")
    (PUB/"structured_parameters.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    summary = df.groupby("book_genre").agg(reviews=("cluster","size"),price_observed=("price","count"),
            median_observed_price=("price","median"),median_rating=("rating","median"),
            median_review_age_days=("review_age_days","median"),median_tokens=("review_tokens","median"))
    summary["price_missing_pct"] = 100*(1-summary.price_observed/summary.reviews)
    summary.to_csv(PUB/"metadata_coverage.csv")


def conditional_info(labels, cluster, strata, repeats=499):
    """Conditional mutual information in bits, with label permutations restricted to strata."""
    y, g = pd.factorize(labels)[0],pd.factorize(cluster)[0]
    strata = pd.factorize(strata)[0]
    ng,ny,ns = g.max()+1,y.max()+1,strata.max()+1
    base = (strata*ng+g)*ny
    def stat(v):
        cell = np.bincount(base+v,minlength=ns*ng*ny).reshape(ns,ng,ny)
        expected_num = cell.sum(1)[:,None,:]*cell.sum(2)[:,:,None]
        numerator = cell * cell.sum((1,2))[:,None,None]
        ok = cell>0
        return float(np.sum(cell[ok]*np.log2(numerator[ok]/expected_num[ok]))/len(v))
    actual = stat(y)
    rng = np.random.default_rng(1405)
    index = [np.flatnonzero(strata==i) for i in range(ns)]
    null = []
    for _ in range(repeats):
        shuffled = y.copy()
        for ix in index:
            shuffled[ix] = rng.permutation(y[ix])
        null.append(stat(shuffled))
    return dict(users=len(y),strata=ns,information_bits=actual,null_median_bits=float(np.median(null)),
                excess_bits=actual-float(np.median(null)),null_p95_bits=float(np.quantile(null,.95)),
                permutation_p=(1+sum(n>=actual for n in null))/(1+repeats))


def lexicon(df):
    x = sparse.load_npz(OUT/"tfidf.npz")
    x.data[:] = 1
    vocab = np.load(OUT/"lexical_fit.npz",allow_pickle=False)["vocabulary"]
    rows, coverage = [], []
    for scope,scope_mask in (("all",np.ones(len(df),bool)),("book=Thriller",df.book_genre=="Thriller")):
        valid = scope_mask & df.single.ne("")
        for g in GENRE_RULES:
            pos,other = valid & (df.single==g),valid & (df.single!=g)
            n,m = int(pos.sum()),int(other.sum())
            if n == 0 or m == 0:
                coverage.append(dict(scope=scope,genre=g,users=n,other_users=m,terms_eligible=0))
                continue
            a,b = np.asarray(x[np.asarray(pos)].sum(0)).ravel(),np.asarray(x[np.asarray(other)].sum(0)).ravel()
            # Document presence counts each reviewer once. Frequency floor limits single-title accidents.
            keep = (a>=5)&(b>=5)&(a+b>=25)
            logor = np.log((a+.5)/(n-a+.5))-np.log((b+.5)/(m-b+.5))
            se = np.sqrt(1/(a+.5)+1/(n-a+.5)+1/(b+.5)+1/(m-b+.5))
            coverage.append(dict(scope=scope,genre=g,users=n,other_users=m,terms_eligible=int(keep.sum())))
            ix = np.flatnonzero(keep & (logor>0))
            for j in ix[np.argsort(-(logor/se)[ix])[:15]]:
                rows.append(dict(scope=scope,genre=g,term=vocab[j],users=n,others=m,using_term=int(a[j]),
                                 others_using=int(b[j]),pct=100*a[j]/n,other_pct=100*b[j]/m,
                                 log_odds=logor[j],ranking_score=logor[j]/se[j]))
    write_rows(PUB/"distinctive_terms.csv",rows)
    write_rows(PUB/"lexical_support.csv",coverage)


def compare(df):
    # A contrast motivated in earlier EDA. Bootstrap persons, not individual words.
    rng = np.random.default_rng(372)
    output = []
    for scope, subset in (("all",df),("book=Thriller",df[df.book_genre=="Thriller"]),
                           ("book=Thriller_no_quotes",df[(df.book_genre=="Thriller") & (df.quote_reviews==0)])):
        for metric in ("first_singular_per_1000_words","mean_sentence_words"):
            v,w = (subset.loc[subset.single==g,metric].dropna().to_numpy() for g in ("Romance","Science Fiction"))
            if not len(v) or not len(w):
                continue
            deltas = [np.median(rng.choice(v,len(v)))-np.median(rng.choice(w,len(w))) for _ in range(2000)]
            output.append(dict(scope=scope,metric=metric,romance_users=len(v),sf_users=len(w),
                               romance_median=np.median(v),sf_median=np.median(w),
                               difference=np.median(v)-np.median(w),p025=np.quantile(deltas,.025),p975=np.quantile(deltas,.975)))
    write_rows(PUB/"user_contrasts.csv",output)


def run():
    df = load()
    structured(df)
    lexicon(df)
    compare(df)
    rows = []
    for g in GENRE_RULES:
        before = df.original.str.split("|",regex=False).apply(lambda x:g in x)
        after = df.independent.str.split("|",regex=False).apply(lambda x:g in x)
        rows.append(dict(genre=g,original=int(before.sum()),retained=int((before&after).sum()),
                         lost=int((before&~after).sum()),gained=int((~before&after).sum()),after=int(after.sum())))
    write_rows(PUB/"label_sensitivity.csv",rows)
    clean = df[df.single!=""].copy()
    size = pd.qcut(clean.review_tokens,5,duplicates="drop").astype(str)
    results = []
    for name,strata in (("none",np.zeros(len(clean))),("book_genre",clean.book_genre),
                        ("book_genre_and_length_quintile",clean.book_genre+"|"+size)):
        results.append(dict(target="independent_preference",condition=name,
                            **conditional_info(clean.single,clean.cluster,strata)))
    varied_titles = clean.groupby("parent_asin").single.nunique()
    repeated = clean[clean.parent_asin.isin(varied_titles[varied_titles > 1].index)]
    if len(repeated):
        results.append(dict(target="independent_preference",condition="same_title_varied_preferences",
                            **conditional_info(repeated.single,repeated.cluster,repeated.parent_asin)))
    for name,cat in (("book_genre",df.book_genre),("rating",df.rating.fillna(-1)),("length_quintile",pd.qcut(df.review_tokens,5,duplicates="drop"))):
        results.append(dict(target=name,condition="none",**conditional_info(cat,df.cluster,np.zeros(len(df)))))
    order = np.argsort([r["permutation_p"] for r in results])
    q = 1.0
    for rank in range(len(results)-1,-1,-1):
        i = order[rank]
        q = min(q,results[i]["permutation_p"]*len(results)/(rank+1))
        results[i]["bh_q"] = q
    write_rows(PUB/"cluster_association.csv",results)
    for name,field in (("book_genre","book_genre"),("preference","independent")):
        pd.crosstab(df.cluster,df[field].replace("","INSUFFICIENT")).to_csv(PUB/f"cluster_by_{name}.csv")
    agg = df.groupby("cluster").agg(users=("cluster","size"),median_words=("review_tokens","median"),
             median_sentence_words=("mean_sentence_words","median"),median_person=("first_singular_per_1000_words","median"),
             median_rating=("rating","median"),mean_rating=("rating","mean"),median_helpful=("helpful_votes","median"),
             median_eligible_reviews=("eligible_reviews","median"),median_review_age_days=("review_age_days","median"),
             mean_price_missing=("price_missing","mean"))
    agg.to_csv(PUB/"cluster_profiles.csv")
    # Style-only view: compress heavy positive tails, impute development medians and use robust scaling.
    style = np.log1p(df[list(METRICS)]).copy()
    med = style.median()
    iqr = style.quantile(.75)-style.quantile(.25)
    scale = iqr.where(iqr>0,style.std()).replace(0,1)
    scaled = ((style.fillna(med)-med)/scale).clip(-5,5)
    scaled.set_index(df.user_key).to_csv(OUT/"scaled_style.csv.gz")
    pd.DataFrame(dict(log_median=med,scale=scale,missing=style.isna().sum())).to_csv(PUB/"style_parameters.csv")
    u,s,axes = np.linalg.svd(scaled-scaled.mean(),full_matrices=False)
    np.savez_compressed(OUT/"style_pca.npz",coordinates=u[:,:2]*s[:2],components=axes,
                        explained_fraction=s**2/sum(s**2),mean=scaled.mean().to_numpy())
    k,(_,other,_),scores,qualified = choose(scaled.to_numpy(),range(2,7))
    write_rows(PUB/"style_cluster_selection.csv",scores)
    descriptor = df.assign(style_cluster=other+1).groupby("style_cluster")[list(METRICS)].median()
    descriptor.to_csv(PUB/"style_cluster_profiles.csv")
    stats = dict(users=len(df),independent_single=len(clean),independent_none=int((df.independent=="").sum()),
                 independent_multi=int(df.independent.str.contains(r"\|").sum()),style_k=k,
                 style_qualified=qualified,style_text_ari=ari(df.cluster,other),distinct_books=df.parent_asin.nunique(),
                 rating_missing=int(df.rating.isna().sum()),unmapped=int((df.book_genre=="UNMAPPED").sum()),
                 no_english_anchors=int(df.no_english_anchor_reviews.sum()),price_conflicts=int(df.price_conflict.sum()))
    (PUB/"interpretation.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    run()
