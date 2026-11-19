"""Check repeatability across two disjoint reviews per development user; audit style redundancy."""
import csv
import gzip
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from phase2_sample import OUT, PUB, read_sample, write_rows
from style_indicators import METRICS


def run():
    records = read_sample()
    keys = {r["user_key"]:r["parent_asin"] for r in records}
    first, second = {}, {}
    with gzip.open("data/analysis/style-v1/review_style.csv.gz","rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            u,b = row["user_key"],row["parent_asin"]
            if keys[u] == b:
                first[u] = row
                continue
            order = sha256(f"phase2-repeat|{u}|{b}".encode()).digest()
            if u not in second or order < second[u][0]:
                second[u] = order,row
    if keys.keys()!=first.keys() or keys.keys()!=second.keys():
        raise ValueError("Unpaired development users")
    a = pd.DataFrame(first.values()).set_index("user_key").sort_index()
    b = pd.DataFrame(v[1] for v in second.values()).set_index("user_key").sort_index()
    a.to_csv(OUT/"first_review_style.csv.gz")
    ma = a[list(METRICS)].apply(pd.to_numeric,errors="coerce")
    mb = b[list(METRICS)].apply(pd.to_numeric,errors="coerce")
    scopes = {"all":np.ones(len(a),bool),"same_mapped_book_genre":(a.book_genre==b.book_genre)&(a.book_genre!=""),
              "both_at_least_100_words":(a.words.astype(int)>=100)&(b.words.astype(int)>=100)}
    rows = []
    for name, mask in scopes.items():
        for col in METRICS:
            pairs = pd.concat([ma.loc[mask,col],mb.loc[mask,col]],axis=1).dropna()
            rho = pairs.iloc[:,0].rank().corr(pairs.iloc[:,1].rank()) if pairs.nunique().min()>1 else None
            rows.append(dict(scope=name,metric=col,users=len(pairs),repeat_spearman=rho,
                             median_abs_difference=(pairs.iloc[:,0]-pairs.iloc[:,1]).abs().median(),
                             both_zero_fraction=(pairs==0).all(axis=1).mean()))
    write_rows(PUB/"review_repeatability.csv",rows)
    user_df = pd.read_csv("data/analysis/style-v1/user_style.csv")
    values = user_df.loc[user_df.split=="train",list(METRICS)]
    corr = values.corr(method="spearman")
    write_rows(PUB/"style_redundancy.csv",
               (dict(metric_a=c,metric_b=d,spearman=corr.loc[c,d]) for i,c in enumerate(METRICS) for d in METRICS[i+1:]))
    write_rows(PUB/"style_support.csv",
               (dict(metric=c,valid_users=values[c].count(),zero_share=(values[c].dropna()==0).mean(),
                     p05=values[c].quantile(.05),median=values[c].median(),p95=values[c].quantile(.95),
                     valid_single_reviews=ma[c].count(),single_review_zero_share=(ma[c].dropna()==0).mean()) for c in METRICS))
    print("Paired",len(a),"development authors")


if __name__ == "__main__":
    run()
