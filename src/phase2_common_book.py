"""Expand the common-input-genre comparison using one thriller review per eligible author."""

import csv
import gzip
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from phase2_patterns import terms, vectorize
from phase2_sample import BASE, OUT, PUB, write_rows
from style_indicators import METRICS, counts_from_text, metrics


def run():
    with (BASE / "users.csv").open(encoding="utf-8") as f:
        users = {r["user_key"]:r["genres"] for r in csv.DictReader(f)
                 if r["split"] == "train" and r["genres"] in ("Romance","Science Fiction")}
    selected = {}
    with gzip.open(BASE / "reviews.jsonl.gz","rb") as f:
        for line in f:
            r = json.loads(line)
            u = r["user_key"]
            if u not in users or r["book_genre"] != "Thriller":
                continue
            order = sha256(f"phase2-common-v1|{u}|{r['parent_asin']}".encode()).digest()
            if u not in selected or order < selected[u][0]:
                selected[u] = order,r
    rows, docs = [], []
    with (OUT / "common_genre_sample.jsonl").open("w",encoding="utf-8") as f:
        for u in sorted(selected):
            r = selected[u][1]
            r["preference"] = users[u]
            f.write(json.dumps(r)+"\n")
            counts = counts_from_text(r["text"])
            rows.append(dict(user_key=u,preference=users[u],parent_asin=r["parent_asin"],
                             quote=counts["quote_reviews"],words=counts["words"],**metrics(counts)))
            docs.append(terms(r["text"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "common_genre_style.csv.gz",index=False)
    rng = np.random.default_rng(393)
    results = []
    subsets = (("all_targeted",df),("no_quotes",df[df.quote==0]),("at_least_100_words",df[df.words>=100]))
    for scope, part in subsets:
        for metric in ("first_singular_per_1000_words","mean_sentence_words"):
            a,b = (part.loc[part.preference==g,metric].dropna().to_numpy() for g in ("Romance","Science Fiction"))
            boot = [np.median(rng.choice(a,len(a)))-np.median(rng.choice(b,len(b))) for _ in range(2000)]
            results.append(dict(scope=scope,metric=metric,romance_users=len(a),sf_users=len(b),
                                romance_median=np.median(a),sf_median=np.median(b),
                                difference=np.median(a)-np.median(b),
                                p025=np.quantile(boot,.025),p975=np.quantile(boot,.975)))
    write_rows(PUB / "targeted_common_contrasts.csv",results)
    summary = []
    for g in ("Romance","Science Fiction"):
        part = df[df.preference==g]
        for c in METRICS:
            summary.append(dict(genre=g,metric=c,users=len(part),valid_users=part[c].count(),
                                p25=part[c].quantile(.25),median=part[c].median(),p75=part[c].quantile(.75)))
    write_rows(PUB / "targeted_common_style.csv",summary)
    fitted = np.load(OUT / "lexical_fit.npz",allow_pickle=False)
    x = vectorize(docs,list(fitted["vocabulary"]),fitted["idf"])
    x.data[:] = 1
    p = df.preference.eq("Romance").to_numpy()
    a,b = np.asarray(x[p].sum(0)).ravel(),np.asarray(x[~p].sum(0)).ravel()
    n,m = p.sum(),(~p).sum()
    logor = np.log((a+.5)/(n-a+.5))-np.log((b+.5)/(m-b+.5))
    se = np.sqrt(1/(a+.5)+1/(n-a+.5)+1/(b+.5)+1/(m-b+.5))
    support = (a>=5)&(b>=5)&(a+b>=25)
    words = []
    for name,sign in (("Romance",1),("Science Fiction",-1)):
        ix = np.flatnonzero(support & (sign*logor>0))
        for j in ix[np.argsort(-(sign*logor/se)[ix])[:12]]:
            words.append(dict(group=name,term=str(fitted["vocabulary"][j]),romance_users=int(n),sf_users=int(m),
                              romance_using=int(a[j]),sf_using=int(b[j]),
                              romance_pct=100*a[j]/n,sf_pct=100*b[j]/m,romance_log_odds=logor[j]))
    write_rows(PUB / "targeted_common_terms.csv",words)
    print("Targeted common input genre",df.preference.value_counts().to_dict())


if __name__ == "__main__":
    run()
