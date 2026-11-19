"""Fit development-only lexical representations and evaluate unsupervised review clusters."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from audit import WORD
from phase2_sample import OUT, PUB, read_sample, write_rows
from writing_style import clean_body, sentence_parts


SEED = 412
MAX_TERMS = 12000
STOP = set("""a an the to of in on at by with from as and or but so i me my mine we us our you your
he him his she her it its they them their this that these those be been being is am are was were
do does did have has had will would could can should may might for if then than also very not no""".split())
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)


def terms(body):
    text = unicodedata.normalize("NFKC", clean_body(body)[0]).casefold()
    text = URL.sub(" ", text)
    counts = Counter()
    for part, _ in sentence_parts(text):
        previous = None
        for w in WORD.findall(part):
            if not w.isalpha() or w in STOP or len(w) < 2 or len(w) > 40:
                previous = None
                continue
            counts[w] += 1
            if previous:
                counts[previous + " " + w] += 1
            previous = w
    return counts


def fit_vocabulary(docs):
    df = Counter()
    for doc in docs:
        df.update(doc.keys())
    vocab = sorted((t for t in df if 5 <= df[t] <= .65 * len(docs)), key=lambda t: (-df[t], t))[:MAX_TERMS]
    idf = 1 + np.log((1 + len(docs)) / (1 + np.array([df[t] for t in vocab])))
    return vocab, idf


def vectorize(docs, vocab, idf):
    index = {term: j for j, term in enumerate(vocab)}
    cols, vals, starts = [], [], [0]
    for doc in docs:
        for term, count in doc.items():
            j = index.get(term)
            if j is not None:
                cols.append(j)
                vals.append((1 + np.log(count)) * idf[j])
        starts.append(len(cols))
    x = sparse.csr_matrix((vals, cols, starts), shape=(len(docs), len(vocab)), dtype=np.float64)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    return x.multiply((1 / np.maximum(norms, 1e-12))[:, None]).tocsr()


def svd_fit(x, rank=48, seed=SEED):
    rng = np.random.default_rng(seed)
    trial = x @ rng.normal(size=(x.shape[1], rank + 12))
    for _ in range(3):
        q = np.linalg.qr(trial, mode="reduced")[0]
        trial = x @ np.linalg.qr(x.T @ q, mode="reduced")[0]
    q = np.linalg.qr(trial, mode="reduced")[0]
    _, s, vt = np.linalg.svd((x.T @ q).T, full_matrices=False)
    return s[:rank], vt[:rank]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def ari(a, b):
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    nb = int(ib.max()) + 1
    table = np.bincount(ia * nb + ib, minlength=(int(ia.max())+1)*nb).reshape(-1, nb)
    def pairs(n):
        return np.sum(n * (n-1) / 2)
    ra, cb = pairs(table.sum(1)), pairs(table.sum(0))
    exp = ra * cb / pairs(np.array(len(a)))
    denom = (ra+cb)/2 - exp
    return float((pairs(table)-exp)/denom) if denom else 1.0


def fit_clusters(z, k, seed):
    centers, _ = kmeans2(z, k, iter=55, minit="++", seed=seed, missing="raise")
    d = cdist(z, centers, "sqeuclidean")
    labels = d.argmin(1)
    return centers, labels, float(d.min(1).sum())


def silhouette(distance, labels):
    s = np.zeros(len(labels))
    clusters = np.unique(labels)
    for g in clusters:
        mask = labels == g
        if mask.sum() < 2 or len(clusters) == 1:
            continue
        a = distance[np.ix_(mask,mask)].sum(1)/(mask.sum()-1)
        b = np.min([distance[np.ix_(mask,labels == h)].mean(1) for h in clusters if h != g],axis=0)
        s[mask] = (b-a)/np.maximum(np.maximum(a,b),1e-12)
    return float(s.mean())


def choose(z, ks, seed=SEED):
    sample = np.sort(np.random.default_rng(seed).choice(len(z), min(1800,len(z)),replace=False))
    distances = cdist(z[sample],z[sample])
    candidates, models = [], {}
    for k in ks:
        fits = [fit_clusters(z,k,s) for s in (seed,seed+1,seed+2,seed+3)]
        best = min(fits,key=lambda f:f[2])
        pair_ari = [ari(a[1],b[1]) for i,a in enumerate(fits) for b in fits[i+1:]]
        sil = [silhouette(distances,f[1][sample]) for f in fits]
        freq = np.bincount(best[1],minlength=k)
        candidates.append(dict(k=k,silhouette=float(np.median(sil)),seed_ari=float(np.median(pair_ari)),
                               min_cluster_share=float(freq.min()/len(z)),inertia=best[2]))
        models[k] = best
    adequate = [r for r in candidates if r["seed_ari"] >= .7 and r["min_cluster_share"] >= .02]
    pool = adequate or [r for r in candidates if r["min_cluster_share"] >= .02] or candidates
    winner = max(pool,key=lambda r:(r["silhouette"], -r["k"]))["k"]
    return winner, models[winner], candidates, bool(adequate)


def discover():
    records = read_sample()
    docs = [terms(r["text"]) for r in records]
    vocab, idf = fit_vocabulary(docs)
    x = vectorize(docs,vocab,idf)
    usable = np.array([not r["duplicate_sample_body"] for r in records]) & (x.getnnz(axis=1)>0)
    x = x[usable]
    records = [r for r, ok in zip(records,usable) if ok]
    docs = [d for d, ok in zip(docs,usable) if ok]
    # The frozen sample currently has no normalized duplicates. Refit if this changes.
    if not usable.all():
        vocab, idf = fit_vocabulary(docs)
        x = vectorize(docs,vocab,idf)
        if np.any(x.getnnz(axis=1)==0):
            raise ValueError("Vocabulary leaves empty documents after filtering")
    s, vt = svd_fit(x)
    raw = x @ vt.T
    z = unit(raw)
    k,(centers,labels,inertia),options,qualified = choose(z,range(2,9))
    write_rows(PUB/"cluster_selection.csv",options)
    print("Chosen k",k,"usable",len(records),flush=True)
    stability = []
    for seed in (413,414,415):
        take = np.sort(np.random.default_rng(seed).choice(len(z),int(.8*len(z)),replace=False))
        subset = [docs[i] for i in take]
        v, weights = fit_vocabulary(subset)
        xt = vectorize(subset,v,weights)
        _, components = svd_fit(xt,seed=seed)
        proj = unit(xt @ components.T)
        _, sublabels, _ = min((fit_clusters(proj,k,seed+j) for j in range(3)),key=lambda f:f[2])
        stability.append(dict(change="refit_on_80pct",seed=seed,users=len(take),ari=ari(labels[take],sublabels)))
    for name, alternative in (("without_row_normalization",raw),("24_components",unit(raw[:,:24]))):
        _, other, _ = min((fit_clusters(alternative,k,SEED+j) for j in range(3)),key=lambda f:f[2])
        stability.append(dict(change=name,seed=SEED,users=len(z),ari=ari(labels,other)))
    write_rows(PUB/"cluster_stability.csv",stability)
    descriptors = []
    total = np.asarray(x.sum(0)).ravel()
    for g in range(k):
        mask = labels == g
        summed = np.asarray(x[mask].sum(0)).ravel()
        diff = summed/mask.sum() - (total-summed)/(len(z)-mask.sum())
        top = np.argsort(-diff)[:12]
        descriptors.append(dict(cluster=int(g)+1,users=int(mask.sum()),terms=" | ".join(vocab[j] for j in top)))
    write_rows(PUB/"cluster_terms.csv",descriptors)
    write_rows(OUT/"assignments.csv",(dict(user_key=r["user_key"],parent_asin=r["parent_asin"],
                                            cluster=int(g)+1,svd1=a[0],svd2=a[1]) for r,g,a in zip(records,labels,raw)))
    sparse.save_npz(OUT/"tfidf.npz",x)
    np.savez_compressed(OUT/"lexical_fit.npz",vocabulary=np.asarray(vocab),idf=idf,components=vt,
                        singular_values=s,centers=centers,coordinates=raw)
    stats = dict(input_reviews=len(usable),usable=len(records),terms=len(vocab),nonzero=x.nnz,
                 energy_2=float(sum(s[:2]**2)/len(records)),energy_24=float(sum(s[:24]**2)/len(records)),
                 energy_48=float(sum(s**2)/len(records)),k=k,qualified=qualified,inertia=inertia)
    (PUB/"representation.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(stats,flush=True)


if __name__ == "__main__":
    discover()
