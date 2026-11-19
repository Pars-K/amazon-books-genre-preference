"""Registered development-only predictive feasibility experiment.

Uses SciPy's optimizer, with explicit fold-local preprocessing and book grouping.
No final-test files are read. Individual predictions stay under ignored data/.
"""

import argparse
from collections import Counter
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

from phase2_patterns import fit_vocabulary, terms, vectorize
from style_indicators import METRICS, counts_from_text, metrics
from writing_style import clean_body


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "reports/feasibility"
OUT = ROOT / "data/analysis/predictive-feasibility-v1"
LABELS = ["Fantasy", "Science Fiction", "Romance", "Mystery", "Thriller", "Philosophy", "Science"]
ALPHAS = (0.0001, 0.001, 0.01)
FEATURES = ("style", "words", "combined", "context", "context_style", "context_words", "context_combined")


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def body_key(text):
    return sha256(" ".join(clean_body(text)[0].casefold().split()).encode()).hexdigest()


class Dataset:
    def __init__(self, records, labels, styles=None):
        self.records = records
        self.labels = list(labels)
        self.y = np.array([self.labels.index(r["preference"]) for r in records])
        self.books = np.array([r["parent_asin"] for r in records])
        self.users = np.array([r["user_key"] for r in records])
        self.docs = [terms(r["text"]) for r in records]
        values = []
        for row in records:
            measured = styles[row["user_key"]] if styles is not None else metrics(counts_from_text(row["text"]))
            values.append([float(measured[m]) if measured[m] not in (None, "") else np.nan for m in METRICS])
        self.raw_style = np.asarray(values, dtype=float)
        if len(set(self.users)) != len(records):
            raise ValueError("Dataset has repeated authors")
        if len({body_key(r["text"]) for r in records}) != len(records):
            raise ValueError("Dataset has duplicate normalized bodies")


def main_data():
    base = ROOT / "data/analysis/phase2-final"
    independent = {r["user_key"]: r["independent"] for r in read_csv(base / "independent.csv")}
    style = {r["user_key"]: r for r in read_csv(base / "first_review_style.csv.gz")}
    records = []
    for row in read_jsonl(base / "sample.jsonl"):
        if independent[row["user_key"]] in LABELS:
            if style[row["user_key"]]["parent_asin"] != row["parent_asin"]:
                raise ValueError("Style belongs to a different book")
            records.append(dict(row, preference=independent[row["user_key"]]))
    return Dataset(records, LABELS, style)


def objective(theta, x, y, k, alpha):
    w = theta.reshape(x.shape[1], k)
    scores = x @ w
    logp = scores - logsumexp(scores, axis=1, keepdims=True)
    loss = -logp[np.arange(len(y)), y].mean() + 0.5 * alpha * np.sum(w[:-1] ** 2)
    error = np.exp(logp)
    error[np.arange(len(y)), y] -= 1
    grad = np.asarray(x.T @ error) / len(y)
    grad[:-1] += alpha * w[:-1]
    return float(loss), grad.ravel()


def logistic(x, y, xt, k, alpha):
    x = sparse.hstack([x, np.ones((x.shape[0], 1))], format="csr")
    xt = sparse.hstack([xt, np.ones((xt.shape[0], 1))], format="csr")
    freq = np.bincount(y, minlength=k)
    if np.any(freq == 0):
        raise ValueError("A training class has no observations")
    initial = np.zeros((x.shape[1], k))
    initial[-1] = np.log(freq / freq.sum())
    result = minimize(objective, initial.ravel(), args=(x, y, k, alpha), jac=True,
                      method="L-BFGS-B", options={"maxiter": 350, "gtol": 1e-6, "ftol": 1e-10, "maxcor": 10})
    if not result.success:
        raise RuntimeError(f"Optimizer did not converge: {result.message}; {result.nit}")
    pred = np.asarray(xt @ result.x.reshape(x.shape[1], k)).argmax(1)
    return pred, dict(iterations=int(result.nit), loss=float(result.fun), gradient_max=float(np.abs(result.jac).max()))


def confusion(y, pred, k, weights=None):
    return np.bincount(y * k + pred, weights=weights, minlength=k * k).reshape(k, k)


def score_matrix(c, selected=None):
    tp = np.diag(c)
    supports = c.sum(1)
    pred_supports = c.sum(0)
    recall = np.divide(tp, supports, out=np.zeros_like(tp, dtype=float), where=supports > 0)
    f1 = np.divide(2 * tp, supports + pred_supports, out=np.zeros_like(tp, dtype=float), where=(supports + pred_supports) > 0)
    ix = np.arange(len(c)) if selected is None else selected
    return dict(macro_f1=float(f1[ix].mean()), balanced_accuracy=float(recall[ix].mean()),
                accuracy=float(tp.sum() / max(c.sum(), 1)))


def scores(y, pred, k):
    return score_matrix(confusion(y, pred, k))


def folds(data, scheme):
    if scheme == "book":
        return np.array([int(sha256(("feasibility-v1-book|" + b).encode()).hexdigest()[:12], 16) % 3 for b in data.books])
    assignments = np.zeros(len(data.y), dtype=int)
    rng = np.random.default_rng(601)
    for label in range(len(data.labels)):
        indices = np.flatnonzero(data.y == label)
        rng.shuffle(indices)
        assignments[indices] = np.arange(len(indices)) % 3
    return assignments


def inner_split(data, train):
    held = np.array([int(sha256(("feasibility-v1-inner|" + data.books[i]).encode()).hexdigest()[:12], 16) % 5 == 0 for i in train])
    a, b = train[~held], train[held]
    if set(data.books[a]) & set(data.books[b]):
        raise ValueError("Inner book leakage")
    return a, b


def numeric_transform(train, test, robust):
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    if robust:
        q = np.nanquantile(train, [.25, .75], axis=0)
        scale = q[1] - q[0]
    else:
        scale = np.nanstd(train, axis=0)
    sd = np.nanstd(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, sd)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1)
    def apply(values):
        return np.clip((np.where(np.isfinite(values), values, med) - med) / scale, -5, 5)
    return apply(train), apply(test)


def count_matrix(docs, vocab):
    lookup = {w: i for i, w in enumerate(vocab)}
    cols, values, starts = [], [], [0]
    for doc in docs:
        for w, count in doc.items():
            if w in lookup:
                cols.append(lookup[w])
                values.append(count)
        starts.append(len(cols))
    return sparse.csr_matrix((values, cols, starts), shape=(len(docs), len(vocab)), dtype=float)


def make_features(data, train, test):
    td, vd = [data.docs[i] for i in train], [data.docs[i] for i in test]
    vocab, idf = fit_vocabulary(td)
    word = (vectorize(td, vocab, idf), vectorize(vd, vocab, idf))
    raw = np.log1p(data.raw_style)
    a, b = numeric_transform(raw[train], raw[test], robust=True)
    style = (sparse.csr_matrix(a / np.sqrt(len(METRICS))), sparse.csr_matrix(b / np.sqrt(len(METRICS))))
    length = np.log1p(np.array([r["tokens"] for r in data.records], dtype=float))
    lengths = np.column_stack([length, length ** 2])
    a, b = numeric_transform(lengths[train], lengths[test], robust=False)
    if len(data.labels) > 2:
        book = np.array([[float(r["book_genre"] == g) for g in LABELS + ["UNMAPPED"]] for r in data.records])
        a = np.column_stack([book[train], a])
        b = np.column_stack([book[test], b])
    context = (sparse.csr_matrix(a), sparse.csr_matrix(b))
    features = dict(style=style, words=word, context=context)
    for name, parts in {
        "combined": (style, word), "context_style": (context, style),
        "context_words": (context, word), "context_combined": (context, style, word),
    }.items():
        features[name] = tuple(sparse.hstack([part[j] for part in parts], format="csr") for j in (0, 1))
    counts = (count_matrix(td, vocab), count_matrix(vd, vocab))
    return features, counts, len(vocab)


def bayes(counts, y, k):
    x, xt = counts
    sums = np.vstack([np.asarray(x[y == j].sum(0)).ravel() for j in range(k)]) + 1
    lp = np.log(sums) - np.log(sums.sum(1, keepdims=True))
    prior = np.log(np.bincount(y, minlength=k) / len(y))
    return np.asarray(xt @ lp.T + prior).argmax(1)


def fit_split(data, train, test, scheme, fold):
    k = len(data.labels)
    inner_train, inner_valid = inner_split(data, train)
    print(json.dumps(dict(stage="features", scheme=scheme, fold=fold, train=len(train), test=len(test))), flush=True)
    inner, _, inner_terms = make_features(data, inner_train, inner_valid)
    outer, counts, outer_terms = make_features(data, train, test)
    predictions = {}
    diagnostics = []
    for name in FEATURES:
        candidates = []
        for alpha in ALPHAS:
            pred, diag = logistic(inner[name][0], data.y[inner_train], inner[name][1], k, alpha)
            quality = scores(data.y[inner_valid], pred, k)["macro_f1"]
            candidates.append((quality, alpha))
            diagnostics.append(dict(scheme=scheme, fold=fold, model=name, stage="inner", alpha=alpha,
                                    macro_f1=quality, vocabulary_terms=inner_terms, **diag))
        quality, alpha = max(candidates)
        pred, diag = logistic(outer[name][0], data.y[train], outer[name][1], k, alpha)
        predictions[name] = pred
        diagnostics.append(dict(scheme=scheme, fold=fold, model=name, stage="outer", alpha=alpha,
                                macro_f1=scores(data.y[test], pred, k)["macro_f1"], vocabulary_terms=outer_terms, **diag))
        print(json.dumps(diagnostics[-1]), flush=True)
    predictions["naive_bayes"] = bayes(counts, data.y[train], k)
    freq = np.bincount(data.y[train], minlength=k)
    predictions["majority"] = np.full(len(test), freq.argmax())
    predictions["prior_random"] = np.random.default_rng(801 + fold).choice(k, size=len(test), p=freq / freq.sum())
    rows = [dict(user_key=data.users[i], parent_asin=data.books[i], truth=int(data.y[i]), fold=fold,
                 **{name: int(pred[j]) for name, pred in predictions.items()}) for j, i in enumerate(test)]
    return rows, diagnostics


def fit_main(scheme):
    target = OUT / f"{scheme}_predictions.csv"
    if target.exists():
        print(f"Completed predictions already exist: {target.name}", flush=True)
        return
    data = main_data()
    assignments = folds(data, scheme)
    all_rows, all_diag, support = [], [], []
    for fold in range(3):
        train, test = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
        if set(data.users[train]) & set(data.users[test]):
            raise ValueError("Author leakage")
        if scheme == "book" and set(data.books[train]) & set(data.books[test]):
            raise ValueError("Book leakage")
        checkpoint = OUT / f"{scheme}_fold{fold}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            rows, diag = saved["rows"], saved["diagnostics"]
        else:
            rows, diag = fit_split(data, train, test, scheme, fold)
            checkpoint.write_text(json.dumps(dict(rows=rows, diagnostics=diag)), encoding="utf-8")
        all_rows.extend(rows)
        all_diag.extend(diag)
        for j, name in enumerate(data.labels):
            support.append(dict(scheme=scheme, fold=fold, genre=name, train=int(sum(data.y[train] == j)), test=int(sum(data.y[test] == j))))
    if len(all_rows) != len(data.y) or len({r["user_key"] for r in all_rows}) != len(data.y):
        raise ValueError("Incomplete or duplicate out-of-fold predictions")
    write_csv(target, all_rows)
    write_csv(PUB / f"{scheme}_fit_diagnostics.csv", all_diag)
    write_csv(PUB / f"{scheme}_support.csv", support)


def fit_common():
    if (OUT / "common_predictions.csv").exists():
        return
    train = read_jsonl(ROOT / "data/analysis/phase2-final/common_genre_sample.jsonl")
    test = read_jsonl(ROOT / "data/analysis/validity-v1/validation_common.jsonl")
    books = {r["parent_asin"] for r in test}
    bodies = {body_key(r["text"]) for r in test}
    kept = [r for r in train if r["parent_asin"] not in books and body_key(r["text"]) not in bodies]
    if {r["user_key"] for r in kept} & {r["user_key"] for r in test}:
        raise ValueError("Common-genre author overlap")
    data = Dataset(kept + test, ["Romance", "Science Fiction"])
    rows, diag = fit_split(data, np.arange(len(kept)), np.arange(len(kept), len(data.y)), "common", 0)
    write_csv(OUT / "common_predictions.csv", rows)
    write_csv(PUB / "common_fit_diagnostics.csv", diag)
    info = dict(original_train=len(train), retained_train=len(kept), removed_train=len(train) - len(kept),
                evaluation=len(test), train_books=len({r["parent_asin"] for r in kept}),
                evaluation_books=len(books), train_by_genre=dict(Counter(r["preference"] for r in kept)),
                evaluation_by_genre=dict(Counter(r["preference"] for r in test)),
                book_overlap=0, author_overlap=0, exact_body_overlap=0)
    (PUB / "common_support.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def summarize():
    summaries, per_class, confusion_rows, intervals, fold_scores = [], [], [], [], []
    for scheme in ("book", "user", "common"):
        path = OUT / f"{scheme}_predictions.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        labels = ["Romance", "Science Fiction"] if scheme == "common" else LABELS
        k = len(labels)
        y = np.array([int(r["truth"]) for r in rows])
        models = [c for c in rows[0] if c not in {"user_key", "parent_asin", "truth", "fold"}]
        predictions = {name: np.array([int(r[name]) for r in rows]) for name in models}
        pairs = [("combined", "majority"), ("combined", "prior_random"), ("context_combined", "context")]
        if scheme == "common":
            pairs.append(("combined", "context"))
        books, inverse = np.unique([r["parent_asin"] for r in rows], return_inverse=True)
        rng = np.random.default_rng(907)
        bootstrap = {name: [] for name in models}
        for _ in range(2000):
            multiplicity = np.bincount(rng.integers(len(books), size=len(books)), minlength=len(books))
            weights = multiplicity[inverse]
            for name, pred in predictions.items():
                result = score_matrix(confusion(y, pred, k, weights))
                bootstrap[name].append([result["macro_f1"], result["balanced_accuracy"]])
        bootstrap = {name: np.asarray(values) for name, values in bootstrap.items()}
        for name, pred in predictions.items():
            c = confusion(y, pred, k)
            result = scores(y, pred, k)
            ci = np.quantile(bootstrap[name], [.025, .975], axis=0)
            summaries.append(dict(scheme=scheme, model=name, users=len(y), books=len(books), **result,
                                  macro_f1_low=ci[0, 0], macro_f1_high=ci[1, 0],
                                  balanced_accuracy_low=ci[0, 1], balanced_accuracy_high=ci[1, 1]))
            if k == 7:
                mask = y != LABELS.index("Philosophy")
                selected = [i for i in range(k) if LABELS[i] != "Philosophy"]
                supplemental = score_matrix(confusion(y[mask], pred[mask], k), selected)
                summaries.append(dict(scheme=scheme + "_six_supported", model=name, users=int(mask.sum()),
                                      books=len(set(np.array([r["parent_asin"] for r in rows])[mask])), **supplemental,
                                      macro_f1_low="", macro_f1_high="", balanced_accuracy_low="", balanced_accuracy_high=""))
            for j, label in enumerate(labels):
                tp, n, p = c[j, j], c[j].sum(), c[:, j].sum()
                per_class.append(dict(scheme=scheme, model=name, genre=label, users=int(n),
                                      precision=float(tp / p) if p else 0, recall=float(tp / n) if n else 0,
                                      f1=float(2 * tp / (n + p)) if n + p else 0))
                for l, predicted in enumerate(labels):
                    confusion_rows.append(dict(scheme=scheme, model=name, actual=label, predicted=predicted, users=int(c[j, l])))
            f = np.array([int(r["fold"]) for r in rows])
            for fold in sorted(set(f)):
                ix = f == fold
                fold_scores.append(dict(scheme=scheme, model=name, fold=int(fold), users=int(ix.sum()), **scores(y[ix], pred[ix], k)))
        for a, b in pairs:
            ca, cb = scores(y, predictions[a], k), scores(y, predictions[b], k)
            for j, metric in enumerate(("macro_f1", "balanced_accuracy")):
                lo, hi = np.quantile(bootstrap[a][:, j] - bootstrap[b][:, j], [.025, .975])
                intervals.append(dict(scheme=scheme, model=a, baseline=b, metric=metric,
                                      difference=ca[metric] - cb[metric], low=float(lo), high=float(hi)))
    for name, rows in [("metrics", summaries), ("per_genre", per_class), ("confusion", confusion_rows),
                       ("paired_intervals", intervals), ("fold_metrics", fold_scores)]:
        write_csv(PUB / f"{name}.csv", rows)
    table = {(r["scheme"], r["model"]): r for r in summaries}
    interval = {(r["scheme"], r["model"], r["baseline"], r["metric"]): r for r in intervals}
    def beats(scheme, a, b):
        return all(interval[(scheme, a, b, m)]["difference"] >= .05 and
                   interval[(scheme, a, b, m)]["low"] > 0 for m in ("macro_f1", "balanced_accuracy"))
    main = table[("book", "combined")]
    gate_main = (main["macro_f1"] >= .40 and main["balanced_accuracy"] >= .40 and
                 beats("book", "combined", "majority") and beats("book", "combined", "prior_random"))
    gate_context = beats("book", "context_combined", "context")
    gate_common = beats("common", "combined", "majority") and beats("common", "combined", "context")
    decision = dict(main_gate=bool(gate_main), context_gate=bool(gate_context), common_genre_gate=bool(gate_common),
                    continue_preference_objective=bool(gate_main and (gate_context or gate_common)),
                    philosophy_adequate=False, thresholds_registered_before_fit=True,
                    final_test_opened=False, validation_is_development=True)
    (PUB / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision), flush=True)


def seal():
    inputs = [ROOT / "reports/feasibility/protocol.json", ROOT / "data/analysis/phase2-final/sample.jsonl",
              ROOT / "data/analysis/phase2-final/independent.csv", ROOT / "data/analysis/phase2-final/first_review_style.csv.gz",
              ROOT / "data/analysis/phase2-final/common_genre_sample.jsonl", ROOT / "data/analysis/validity-v1/validation_common.jsonl"]
    files = inputs + [Path(__file__), ROOT / "src/phase2_patterns.py", ROOT / "src/style_indicators.py"]
    files += list(PUB.glob("*.csv")) + [PUB / "decision.json"] + list(OUT.glob("*_predictions.csv"))
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest() for p in files}
    (PUB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["book", "user", "common", "summarize", "seal", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    if args.stage in {"book", "all"}:
        fit_main("book")
    if args.stage in {"user", "all"}:
        fit_main("user")
    if args.stage in {"common", "all"}:
        fit_common()
    if args.stage in {"summarize", "all"}:
        summarize()
    if args.stage in {"seal", "all"}:
        seal()
