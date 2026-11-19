"""Audit lexical discovery stability and replicate fixed contrasts on separate validation authors."""

import argparse
from collections import defaultdict
import csv
from hashlib import file_digest, sha256
import json
from pathlib import Path
import gzip

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import binomtest, false_discovery_control, fisher_exact, mannwhitneyu, rankdata

from audit import GENRE_RULES
from phase2_patterns import fit_clusters, silhouette, terms, unit, vectorize
from phase2_sample import BASE, OUT
from style_indicators import METRICS, counts_from_text, metrics


PUBLIC = Path("reports/phase2/validity")
PRIVATE = Path("data/analysis/validity-v1")
FROZEN = Path("reports/phase2/patterns")
BOOTSTRAPS = 200


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def protocol():
    definition = dict(
        analysis_reference="dfd8b18", purpose="Evidence audit after exploratory discovery",
        vocabulary_candidates_sha256=sha(FROZEN/"targeted_common_terms.csv"),
        style_metrics=list(METRICS), primary_style_metrics=["first_singular_per_1000_words","mean_sentence_words"],
        replication_split="validation", common_input_genre="Thriller",
        preference_groups=["Romance","Science Fiction"],
        selection="minimum sha256(phase2-common-v1|user_key|parent_asin) among thriller reviews",
        primary_style_family_size=2, additional_style_family_size=29,
        word_family="all 24 candidate terms in the frozen targeted comparison, with fixed expected directions",
        correction_primary="Holm for 2 fixed style hypotheses",
        correction_additional="Benjamini-Yekutieli for dependent word and style families",
        bootstrap_repeats=2000, word_rank_resamples=BOOTSTRAPS,
        limitations="Exploratory training estimates already observed. Validation is used for statistical replication and becomes part of analysis development.",
    )
    path = PUBLIC/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != definition:
            raise ValueError("Audit protocol differs from the recorded definition")
    else:
        path.write_text(json.dumps(definition,indent=2)+"\n",encoding="utf-8")


def validation_sample():
    with (BASE/"users.csv").open(encoding="utf-8") as f:
        roster = {r["user_key"]:r["genres"] for r in csv.DictReader(f) if r["split"]=="validation"}
    any_review, common = {}, {}
    with gzip.open(BASE/"reviews.jsonl.gz","rb") as f:
        for line in f:
            row = json.loads(line)
            user, book = row["user_key"], row["parent_asin"]
            if user not in roster:
                continue
            order = sha256(f"phase2-one-v1|{user}|{book}".encode()).digest()
            if user not in any_review or order < any_review[user][0]:
                any_review[user] = order, row
            if roster[user] in ("Romance","Science Fiction") and row["book_genre"]=="Thriller":
                order = sha256(f"phase2-common-v1|{user}|{book}".encode()).digest()
                if user not in common or order < common[user][0]:
                    row["preference"] = roster[user]
                    common[user] = order, row
    if any_review.keys()!=roster.keys():
        raise ValueError("The separate-author sample is incomplete")
    def save(name, selected):
        with (PRIVATE/name).open("w",encoding="utf-8") as f:
            for key in sorted(selected):
                f.write(json.dumps(selected[key][1])+"\n")
    save("validation_random.jsonl",any_review)
    save("validation_common.jsonl",common)
    selected = [common[key][1] for key in sorted(common)]
    original = pd.read_csv(OUT/"common_genre_style.csv.gz")
    if set(original.user_key) & set(common):
        raise ValueError("Replication shares authors with the discovery comparison")
    print("Separate validation authors",len(roster),"common input genre",len(common),flush=True)
    return selected


def discovery_words():
    fitted = np.load(OUT/"lexical_fit.npz",allow_pickle=False)
    vocabulary = list(fitted["vocabulary"])
    assignment = pd.read_csv(OUT/"assignments.csv",dtype={"user_key":str})
    evidence = pd.read_csv(OUT/"independent.csv",dtype={"user_key":str},keep_default_na=False)
    merged = assignment.merge(evidence,on="user_key",validate="one_to_one")
    valid = merged.independent.isin(GENRE_RULES).to_numpy()
    x = sparse.load_npz(OUT/"tfidf.npz")[valid].astype(float)
    x.data[:] = 1
    labels = merged.loc[valid,"independent"].to_numpy()
    total = np.asarray(x.sum(0)).ravel().astype(int)
    tests, groups = [], {}
    rng = np.random.default_rng(516)
    for group in GENRE_RULES:
        positive = labels==group
        xp, xo = x[positive],x[~positive]
        n,m = xp.shape[0],xo.shape[0]
        a = np.asarray(xp.sum(0)).ravel().astype(int)
        b = total-a
        cache = {}
        ps = []
        for u,v in zip(a,b):
            key = (int(u),int(v))
            if key not in cache:
                cache[key] = fisher_exact([[u,n-u],[v,m-v]],alternative="two-sided").pvalue
            ps.append(cache[key])
        tests.extend(ps)
        frequency = np.zeros(len(vocabulary),int)
        for _ in range(BOOTSTRAPS):
            wa = rng.multinomial(n,np.full(n,1/n))
            wb = rng.multinomial(m,np.full(m,1/m))
            ra,rb = np.asarray(xp.T@wa).ravel(),np.asarray(xo.T@wb).ravel()
            logor = np.log((ra+.5)/(n-ra+.5))-np.log((rb+.5)/(m-rb+.5))
            error = np.sqrt(1/(ra+.5)+1/(n-ra+.5)+1/(rb+.5)+1/(m-rb+.5))
            eligible = (ra>=5)&(rb>=5)&(ra+rb>=25)&(logor>0)
            indices = np.flatnonzero(eligible)
            top = indices[np.argsort(-(logor/error)[indices])[:15]]
            frequency[top] += 1
        groups[group] = dict(p=ps,frequency=frequency,a=a,b=b,n=n,m=m)
        print("Word audit",group,"users",n,flush=True)
    adjusted = false_discovery_control(np.asarray(tests),method="by").reshape(len(GENRE_RULES),-1)
    candidates = pd.read_csv(FROZEN/"distinctive_terms.csv")
    candidates = candidates[candidates.scope=="all"]
    lookup = {term:i for i,term in enumerate(vocabulary)}
    rows, summary = [], []
    for gi,g in enumerate(GENRE_RULES):
        item = groups[g]
        chosen = candidates[candidates.genre==g]
        for record in chosen.to_dict("records"):
            j = lookup[record["term"]]
            rows.append(dict(genre=g,term=record["term"],users=item["n"],others=item["m"],
                             using=int(item["a"][j]),others_using=int(item["b"][j]),
                             p_fisher=item["p"][j],q_by_global=adjusted[gi,j],
                             top15_bootstrap_fraction=item["frequency"][j]/BOOTSTRAPS))
        indices = [lookup[t] for t in chosen.term]
        summary.append(dict(genre=g,users=item["n"],reported_words=len(indices),
                            q05_words=int(sum(adjusted[gi,j]<.05 for j in indices)),
                            selected_in_80pct_resamples=int(sum(item["frequency"][j]>=.8*BOOTSTRAPS for j in indices))))
    write_csv(PUBLIC/"discovery_words.csv",rows)
    write_csv(PUBLIC/"discovery_summary.csv",summary)
    np.savez_compressed(PRIVATE/"all_word_tests.npz",pvalues=np.array(tests).reshape(len(GENRE_RULES),-1),
                        q_by=adjusted,vocabulary=np.asarray(vocabulary),groups=np.asarray(list(GENRE_RULES)))


def rank_permutation_p(a,b,rng,repeats=9999):
    """Two-sided randomization calibration of the average-rank sum, preserving ties."""
    ranks = rankdata(np.concatenate([a,b]))
    expected = len(b)*(len(ranks)+1)/2
    observed = abs(ranks[len(a):].sum()-expected)
    extreme = 0
    for _ in range(repeats):
        selected = rng.choice(len(ranks),len(b),replace=False)
        extreme += abs(ranks[selected].sum()-expected) >= observed-1e-9
    return float((1+extreme)/(repeats+1))


def differences(a,b,rng):
    a,b = np.asarray(a),np.asarray(b)
    test = mannwhitneyu(a,b,alternative="two-sided",method="asymptotic")
    permutation_p = rank_permutation_p(a,b,rng)
    boots = np.array([np.median(rng.choice(a,len(a)))-np.median(rng.choice(b,len(b))) for _ in range(2000)])
    return dict(group_a=len(a),group_b=len(b),median_a=float(np.median(a)),median_b=float(np.median(b)),
                difference=float(np.median(a)-np.median(b)),p025=float(np.quantile(boots,.025)),
                p975=float(np.quantile(boots,.975)),p_rank_permutation=permutation_p,
                cliff_delta=float(2*test.statistic/(len(a)*len(b))-1))


def replication(reviews):
    fitted = np.load(OUT/"lexical_fit.npz",allow_pickle=False)
    vocab = list(fitted["vocabulary"])
    x = vectorize([terms(r["text"]) for r in reviews],vocab,fitted["idf"])
    x.data[:] = 1
    rows = []
    for r in reviews:
        counts = counts_from_text(r["text"])
        rows.append(dict(user_key=r["user_key"],preference=r["preference"],**metrics(counts)))
    frame = pd.DataFrame(rows)
    frame.to_csv(PRIVATE/"validation_common_style.csv",index=False)
    rng = np.random.default_rng(938)
    style_rows = []
    for scope,data in (("discovery",pd.read_csv(OUT/"common_genre_style.csv.gz")),("validation",frame)):
        for c in METRICS:
            a,b = (data.loc[data.preference==g,c].dropna().to_numpy() for g in ("Romance","Science Fiction"))
            style_rows.append(dict(scope=scope,metric=c,**differences(a,b,rng)))
    styles = pd.DataFrame(style_rows)
    for scope in ("discovery","validation"):
        take = styles.scope==scope
        styles.loc[take,"q_by_29"] = false_discovery_control(styles.loc[take,"p_rank_permutation"],method="by")
        primary = take & styles.metric.isin(["first_singular_per_1000_words","mean_sentence_words"])
        indices = styles.loc[primary,"p_rank_permutation"].sort_values().index
        values = styles.loc[indices,"p_rank_permutation"].to_numpy()*np.array([2,1])
        styles.loc[indices,"p_holm_primary_2"] = np.minimum(1,np.maximum.accumulate(values))
    styles.to_csv(PUBLIC/"common_style_evidence.csv",index=False)
    words = []
    is_romance = frame.preference.eq("Romance").to_numpy()
    a,b = np.asarray(x[is_romance].sum(0)).ravel(),np.asarray(x[~is_romance].sum(0)).ravel()
    n,m = int(is_romance.sum()),int((~is_romance).sum())
    lookup = {w:i for i,w in enumerate(vocab)}
    for item in pd.read_csv(FROZEN/"targeted_common_terms.csv").to_dict("records"):
        j = lookup[item["term"]]
        u,v = int(a[j]),int(b[j])
        lower_a,upper_a = binomtest(u,n).proportion_ci(confidence_level=.975,method="exact")
        lower_b,upper_b = binomtest(v,m).proportion_ci(confidence_level=.975,method="exact")
        words.append(dict(expected_group=item["group"],term=item["term"],romance_users=n,sf_users=m,
                          romance_using=u,sf_using=v,romance_pct=100*u/n,sf_pct=100*v/m,
                          percentage_point_difference=100*(u/n-v/m),
                          conservative_ci_low=100*(lower_a-upper_b),conservative_ci_high=100*(upper_a-lower_b),
                          direction_replicated=bool((u/n-v/m)*item["romance_log_odds"]>0),
                          p_fisher=fisher_exact([[u,n-u],[v,m-v]],alternative="two-sided").pvalue))
    wd = pd.DataFrame(words)
    wd["q_by_24"] = false_discovery_control(wd.p_fisher,method="by")
    wd.to_csv(PUBLIC/"word_replication.csv",index=False)


def cluster_check():
    fitted = np.load(OUT/"lexical_fit.npz",allow_pickle=False)
    z = unit(fitted["coordinates"])
    rng = np.random.default_rng(193)
    take = np.sort(rng.choice(len(z),min(1800,len(z)),replace=False))
    obs = z[take]
    distances = cdist(obs,obs)
    _,observed,_ = min((fit_clusters(obs,2,193+i) for i in range(4)),key=lambda f:f[2])
    actual = silhouette(distances,observed)
    mean,cov = z.mean(0),np.cov(z,rowvar=False)
    eigen,axes = np.linalg.eigh(cov)
    root = axes * np.sqrt(np.maximum(eigen,0))
    null = []
    for i in range(39):
        gaussian = unit(rng.normal(size=obs.shape)@root.T+mean)
        _,labels,_ = min((fit_clusters(gaussian,2,193+j) for j in range(4)),key=lambda f:f[2])
        null.append(silhouette(cdist(gaussian,gaussian),labels))
    reviews = [json.loads(l) for l in (PRIVATE/"validation_random.jsonl").read_text(encoding="utf-8").splitlines()]
    matrix = vectorize([terms(r["text"]) for r in reviews],list(fitted["vocabulary"]),fitted["idf"])
    usable = matrix.getnnz(axis=1)>0
    coords = unit(matrix[usable]@fitted["components"].T)
    labels = cdist(coords,fitted["centers"]).argmin(1)
    result = dict(actual_sample_users=len(obs),actual_sample_silhouette=actual,simulation_repeats=len(null),
                  single_gaussian_silhouette_median=float(np.median(null)),
                  single_gaussian_p05=float(np.quantile(null,.05)),single_gaussian_p95=float(np.quantile(null,.95)),
                  diagnostic_exceedance_fraction=float(np.mean(np.array(null)>=actual)),
                  validation_users=int(usable.sum()),validation_empty_vectors=int((~usable).sum()),
                  validation_silhouette=silhouette(cdist(coords,coords),labels),
                  validation_cluster_sizes=np.bincount(labels,minlength=2).tolist())
    (PUBLIC/"cluster_evidence.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    np.save(PRIVATE/"single_gaussian_silhouettes.npy",np.array(null))
    print("Cluster evidence",result,flush=True)


def book_sensitivity():
    reviews = [json.loads(l) for l in (PRIVATE/"validation_common.jsonl").read_text(encoding="utf-8").splitlines()]
    frame = pd.read_csv(PRIVATE/"validation_common_style.csv")
    if list(frame.user_key)!=[r["user_key"] for r in reviews]:
        raise ValueError("Book sensitivity authors are out of order")
    romance = frame.preference.eq("Romance").to_numpy()
    codes, titles = pd.factorize(np.array([r["parent_asin"] for r in reviews]))
    candidates = list(pd.read_csv(FROZEN/"targeted_common_terms.csv").term)
    docs = [terms(r["text"]) for r in reviews]
    table = np.column_stack([romance,~romance,*[
        mask & np.array([word in doc for doc in docs]) for word in candidates for mask in (romance,~romance)
    ]]).astype(int)
    by_book = np.zeros((len(titles),table.shape[1]),int)
    np.add.at(by_book,codes,table)
    rng = np.random.default_rng(946)
    weights = rng.multinomial(len(titles),np.full(len(titles),1/len(titles)),size=2000)
    aggregate = weights@by_book
    good = (aggregate[:,0]>0)&(aggregate[:,1]>0)
    aggregate = aggregate[good]
    rows = []
    for i,word in enumerate(candidates):
        delta = 100*(aggregate[:,2+2*i]/aggregate[:,0]-aggregate[:,3+2*i]/aggregate[:,1])
        rows.append(dict(kind="word",feature=word,books=len(titles),
                         romance_positive_books=int(np.sum(by_book[:,2+2*i]>0)),
                         sf_positive_books=int(np.sum(by_book[:,3+2*i]>0)),
                         ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975))))
    for metric in ("first_singular_per_1000_words","mean_sentence_words"):
        values = frame[metric].to_numpy()
        boot = []
        for repeat in np.flatnonzero(good):
            ids = np.repeat(np.arange(len(reviews)),weights[repeat,codes])
            boot.append(np.median(values[ids[romance[ids]]])-np.median(values[ids[~romance[ids]]]))
        rows.append(dict(kind="style",feature=metric,books=len(titles),romance_positive_books=None,sf_positive_books=None,
                         ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    write_csv(PUBLIC/"book_sensitivity.csv",rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","words","replicate","clusters","books"))
    opt = parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    if opt.stage=="freeze":
        protocol()
    elif opt.stage=="words":
        discovery_words()
    elif opt.stage=="replicate":
        protocol()
        replication(validation_sample())
    elif opt.stage=="clusters":
        cluster_check()
    else:
        book_sensitivity()
