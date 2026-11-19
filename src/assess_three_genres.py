"""Assess the user's proposed three-group scope with the existing fixed pipeline."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from predictive_feasibility import (Dataset, ROOT, confusion, fit_split, folds, read_csv,
                                    read_jsonl, score_matrix, scores, write_csv)

PUB = ROOT / "reports/feasibility-three"
OUT = ROOT / "data/analysis/tri-genre-feasibility-v1"
GROUPS = ["Mystery", "Thriller", "Romance"]


def load_data(common):
    base = ROOT / "data/analysis/phase2-final"
    independent = {r["user_key"]: r["independent"] for r in read_csv(base / "independent.csv")}
    style = {r["user_key"]: r for r in read_csv(base / "first_review_style.csv.gz")}
    records = [dict(r, preference=independent[r["user_key"]]) for r in read_jsonl(base / "sample.jsonl")
               if independent[r["user_key"]] in GROUPS and (not common or r["book_genre"] == "Thriller")]
    return Dataset(records, GROUPS, style)


def run_scheme(scheme, common):
    data = load_data(common)
    assigned = folds(data, "book")
    rows, diagnostic, support = [], [], []
    for fold in range(3):
        train, test = np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold)
        assert not (set(data.books[train]) & set(data.books[test]))
        assert not (set(data.users[train]) & set(data.users[test]))
        checkpoint = OUT / f"{scheme}_fold{fold}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            predictions, diag = saved["rows"], saved["diagnostics"]
        else:
            predictions, diag = fit_split(data, train, test, scheme, fold)
            checkpoint.write_text(json.dumps(dict(rows=predictions, diagnostics=diag)), encoding="utf-8")
        rows.extend(predictions)
        diagnostic.extend(diag)
        for j, name in enumerate(GROUPS):
            support.append(dict(scheme=scheme, fold=fold, genre=name,
                                train=int(sum(data.y[train] == j)), test=int(sum(data.y[test] == j))))
    assert len(rows) == len(data.y) and len({r["user_key"] for r in rows}) == len(data.y)
    write_csv(OUT / f"{scheme}_predictions.csv", rows)
    write_csv(PUB / f"{scheme}_fit_diagnostics.csv", diagnostic)
    write_csv(PUB / f"{scheme}_support.csv", support)
    models = [c for c in rows[0] if c not in {"user_key", "parent_asin", "truth", "fold"}]
    truth = np.array([r["truth"] for r in rows])
    pred = {m: np.array([r[m] for r in rows]) for m in models}
    titles, inverse = np.unique([r["parent_asin"] for r in rows], return_inverse=True)
    rng = np.random.default_rng(1962)
    bootstrap = {m: [] for m in models}
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(titles), size=len(titles)), minlength=len(titles))
        for name, predicted in pred.items():
            result = score_matrix(confusion(truth, predicted, 3, counts[inverse]))
            bootstrap[name].append([result["macro_f1"], result["balanced_accuracy"]])
    bootstrap = {m: np.asarray(v) for m, v in bootstrap.items()}
    summaries, per_class, fold_rows, intervals = [], [], [], []
    for name, predicted in pred.items():
        result = scores(truth, predicted, 3)
        ci = np.quantile(bootstrap[name], [.025, .975], axis=0)
        summaries.append(dict(scheme=scheme, model=name, users=len(truth), books=len(titles), **result,
                              macro_f1_low=ci[0, 0], macro_f1_high=ci[1, 0],
                              balanced_accuracy_low=ci[0, 1], balanced_accuracy_high=ci[1, 1]))
        c = confusion(truth, predicted, 3)
        for j, group in enumerate(GROUPS):
            tp, n, p = c[j, j], c[j].sum(), c[:, j].sum()
            per_class.append(dict(scheme=scheme, model=name, genre=group, users=int(n),
                                  precision=float(tp / p) if p else 0,
                                  recall=float(tp / n) if n else 0,
                                  f1=float(2 * tp / (n + p)) if n + p else 0))
        fold = np.array([r["fold"] for r in rows])
        for i in range(3):
            mask = fold == i
            fold_rows.append(dict(scheme=scheme, model=name, fold=i, **scores(truth[mask], predicted[mask], 3)))
    for a, b in [("naive_bayes", "majority"), ("naive_bayes", "prior_random"), ("context_combined", "context")]:
        sa, sb = scores(truth, pred[a], 3), scores(truth, pred[b], 3)
        for j, metric in enumerate(["macro_f1", "balanced_accuracy"]):
            lo, hi = np.quantile(bootstrap[a][:, j] - bootstrap[b][:, j], [.025, .975])
            intervals.append(dict(scheme=scheme, model=a, baseline=b, metric=metric,
                                  difference=sa[metric] - sb[metric], low=float(lo), high=float(hi)))
    for name, content in [("metrics", summaries), ("per_genre", per_class), ("fold_metrics", fold_rows), ("paired_intervals", intervals)]:
        write_csv(PUB / f"{scheme}_{name}.csv", content)
    info = dict(users=len(truth), books=len(titles), groups=dict(Counter(r["preference"] for r in data.records)))
    (PUB / f"{scheme}_population.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", scheme, [(r["model"], r["macro_f1"], r["balanced_accuracy"]) for r in summaries], flush=True)


def decision():
    metric = {r["model"]: r for r in read_csv(PUB / "all_metrics.csv")}
    rows = read_csv(PUB / "all_per_genre.csv")
    recall_ok = all(float(r["recall"]) >= .30 for r in rows if r["model"] == "naive_bayes")
    comparisons = [r for r in read_csv(PUB / "all_paired_intervals.csv") if r["model"] == "naive_bayes"]
    improvement = all(float(r["difference"]) >= .1 and float(r["low"]) > 0 for r in comparisons)
    main = metric["naive_bayes"]
    result = dict(three_group_predictive_criterion=bool(float(main["macro_f1"]) >= .5 and
                  float(main["balanced_accuracy"]) >= .5 and recall_ok and improvement),
                  final_test_opened=False, assessment_is_development=True,
                  current_objective_automatically_replaced=False)
    (PUB / "decision.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    files = [p for p in PUB.glob('*') if p.is_file() and p.name != 'manifest.json']
    files += list(OUT.glob('*_predictions.csv')) + [Path(__file__), ROOT / 'src/predictive_feasibility.py',
              ROOT / 'src/phase2_patterns.py', ROOT / 'src/style_indicators.py',
              ROOT / 'data/analysis/phase2-final/sample.jsonl', ROOT / 'data/analysis/phase2-final/independent.csv',
              ROOT / 'data/analysis/phase2-final/first_review_style.csv.gz']
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p.read_bytes()).hexdigest() for p in files}
    (PUB / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run_scheme("all", False)
    run_scheme("thriller_input", True)
    decision()
