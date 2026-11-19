"""Transparent follow-up of the already specified common-input-genre comparison."""

from collections import Counter
import json

import numpy as np

from predictive_feasibility import (Dataset, OUT, PUB, ROOT, body_key, confusion, fit_split,
                                    folds, read_csv, read_jsonl, score_matrix, scores, write_csv)
from phase2_patterns import fit_vocabulary, vectorize


def main():
    train = read_jsonl(ROOT / "data/analysis/phase2-final/common_genre_sample.jsonl")
    test = read_jsonl(ROOT / "data/analysis/validity-v1/validation_common.jsonl")
    data = Dataset(train, ["Romance", "Science Fiction"])
    assignment = folds(data, "book")
    rows, diag = [], []
    for fold in range(3):
        a, b = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        assert not (set(data.books[a]) & set(data.books[b]))
        assert not (set(data.users[a]) & set(data.users[b]))
        checkpoint = OUT / f"common_cv_fold{fold}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            r, d = saved["rows"], saved["diagnostics"]
        else:
            r, d = fit_split(data, a, b, "common_book_cv", fold)
            checkpoint.write_text(json.dumps(dict(rows=r, diagnostics=d)), encoding="utf-8")
        rows.extend(r)
        diag.extend(d)
    write_csv(OUT / "common_cv_predictions.csv", rows)
    write_csv(PUB / "common_cv_fit_diagnostics.csv", diag)
    models = [c for c in rows[0] if c not in {"user_key", "parent_asin", "truth", "fold"}]
    y = np.array([r["truth"] for r in rows])
    predictions = {m: np.array([r[m] for r in rows]) for m in models}
    books, inv = np.unique([r["parent_asin"] for r in rows], return_inverse=True)
    rng = np.random.default_rng(1431)
    boots = {m: [] for m in models}
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(books), size=len(books)), minlength=len(books))
        for model, pred in predictions.items():
            s = score_matrix(confusion(y, pred, 2, counts[inv]))
            boots[model].append([s["macro_f1"], s["balanced_accuracy"]])
    summaries = []
    for model, pred in predictions.items():
        ci = np.quantile(boots[model], [.025, .975], axis=0)
        summaries.append(dict(model=model, users=len(y), books=len(books), **scores(y, pred, 2),
                              macro_f1_low=ci[0, 0], macro_f1_high=ci[1, 0],
                              balanced_accuracy_low=ci[0, 1], balanced_accuracy_high=ci[1, 1]))
    write_csv(PUB / "common_cv_metrics.csv", summaries)
    fold_rows = []
    for fold in range(3):
        mask = np.array([r["fold"] == fold for r in rows])
        for model, pred in predictions.items():
            fold_rows.append(dict(fold=fold, model=model, users=int(mask.sum()),
                                  sf_users=int(sum(y[mask] == 1)), **scores(y[mask], pred[mask], 2)))
    write_csv(PUB / "common_cv_fold_metrics.csv", fold_rows)

    # Same exclusions as the registered separate-author comparison, no refitting
    # or model selection is performed in response to these similarity values.
    eval_books = {r["parent_asin"] for r in test}
    eval_bodies = {body_key(r["text"]) for r in test}
    kept = [r for r in train if r["parent_asin"] not in eval_books and body_key(r["text"]) not in eval_bodies]
    combined = Dataset(kept + test, data.labels)
    td, vd = combined.docs[:len(kept)], combined.docs[len(kept):]
    vocab, idf = fit_vocabulary(td)
    xt, xv = vectorize(td, vocab, idf), vectorize(vd, vocab, idf)
    similarities = (xv @ xt.T).toarray()
    nearest = similarities.max(1)
    held = read_csv(OUT / "common_predictions.csv")
    lookup = {r["user_key"]: r for r in held}
    selected = [lookup[r["user_key"]] for r in test]
    yt = np.array([int(r["truth"]) for r in selected])
    sensitivity = []
    for cutoff in (.8, .6):
        mask = nearest < cutoff
        for name in ("combined", "words", "style", "naive_bayes", "majority"):
            pred = np.array([int(r[name]) for r in selected])
            sensitivity.append(dict(maximum_cosine_below=cutoff, model=name, users=int(mask.sum()),
                                    sf_users=int(sum(yt[mask] == 1)), **scores(yt[mask], pred[mask], 2)))
    write_csv(PUB / "common_similarity_sensitivity.csv", sensitivity)
    info = dict(training_users=len(kept), evaluation_users=len(test),
                maximum_cosine=float(nearest.max()), median_cosine=float(np.median(nearest)),
                cosine_p95=float(np.quantile(nearest, .95)),
                at_least_point8=int(sum(nearest >= .8)), at_least_point6=int(sum(nearest >= .6)),
                shared_titles=0, shared_authors=0, identical_normalized_bodies=0,
                heldout_rating_by_preference={g: dict(Counter(str(r["rating"]) for r in test if r["preference"] == g)) for g in data.labels})
    (PUB / "common_similarity.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # Descriptive contribution of a term to the fixed NB model's held-out score.
    from predictive_feasibility import count_matrix
    x, xv_counts = count_matrix(td, vocab), count_matrix(vd, vocab)
    lab = combined.y[:len(kept)]
    term_sums = np.vstack([np.asarray(x[lab == i].sum(0)).ravel() for i in range(2)]) + 1
    logp = np.log(term_sums) - np.log(term_sums.sum(1, keepdims=True))
    diff = logp[1] - logp[0]
    presence = xv_counts.copy()
    presence.data[:] = 1
    descriptions = []
    for label, sign in ((0, -1), (1, 1)):
        total = np.asarray(xv_counts[yt == label].sum(0)).ravel()
        contribution = sign * diff * total
        for i in np.argsort(-contribution)[:20]:
            descriptions.append(dict(direction=data.labels[label], term=vocab[i],
                                     log_ratio_sf_vs_romance=float(diff[i]),
                                     evaluation_users_using=int(presence[yt == label, i].sum()),
                                     descriptive_score_contribution=float(contribution[i])))
    write_csv(PUB / "common_model_terms.csv", descriptions)
    print(json.dumps(info), flush=True)
    print("Common CV", [(r["model"], r["macro_f1"]) for r in summaries], flush=True)


if __name__ == "__main__":
    main()
