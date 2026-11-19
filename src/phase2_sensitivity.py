"""Check whether lexical clusters persist after stopword changes and stronger compression capacity."""

import json

import numpy as np
import pandas as pd
from scipy import sparse

from phase2_patterns import SEED, ari, fit_clusters, fit_vocabulary, svd_fit, terms, unit, vectorize
from phase2_sample import OUT, PUB, read_sample, write_rows

ADDITIONAL_FUNCTION_WORDS = set("""who whom whose which what when where why how there here out up down
into onto over under about only even just still much many more most less least all any some each every
such both either neither other another one two get got gets getting make made makes making""".split())


def best(z, k):
    return min((fit_clusters(z, k, SEED+i) for i in range(3)), key=lambda fit: fit[2])


def run():
    main = pd.read_csv(OUT / "assignments.csv", dtype={"user_key":str})
    original = main.cluster.to_numpy()
    k = len(np.unique(original))
    sample = read_sample()
    if [r["user_key"] for r in sample] != list(main.user_key):
        raise ValueError("Sensitivity sample is out of order")
    x = sparse.load_npz(OUT / "tfidf.npz")
    rows = []
    _, axes = svd_fit(x, rank=96)
    z = unit(x @ axes.T)
    _, alt, _ = best(z,k)
    rows.append(dict(change="96_components", users=len(main), ari=ari(original,alt)))
    docs = []
    for r in sample:
        doc = terms(r["text"])
        docs.append({t:c for t,c in doc.items() if not ADDITIONAL_FUNCTION_WORDS.intersection(t.split())})
    vocab, idf = fit_vocabulary(docs)
    filtered = vectorize(docs,vocab,idf)
    ok = filtered.getnnz(axis=1)>0
    _, axes = svd_fit(filtered[ok],rank=48)
    _, alt, _ = best(unit(filtered[ok] @ axes.T),k)
    rows.append(dict(change="expanded_function_word_removal",users=int(ok.sum()),ari=ari(original[ok],alt)))
    new_names = {r["change"] for r in rows}
    old_rows = pd.read_csv(PUB/"cluster_stability.csv").to_dict("records")
    all_rows = [r for r in old_rows if r["change"] not in new_names]
    for r in rows:
        r["seed"] = SEED
        all_rows.append(r)
    write_rows(PUB/"cluster_stability.csv",all_rows)
    print(rows)


if __name__ == "__main__":
    run()
