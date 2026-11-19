"""Class-balanced training on exactly the previous three-group folds.

Prior unweighted code and predictions are left intact for a paired comparison.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

from assess_three_genres import GROUPS, load_data
from predictive_feasibility import (ALPHAS, FEATURES, ROOT, confusion, folds, inner_split,
                                    make_features, read_csv, score_matrix, scores, write_csv)

PUB = ROOT / 'reports/feasibility-balanced'
OUT = ROOT / 'data/analysis/balanced-genre-feasibility-v1'
OLD = ROOT / 'data/analysis/tri-genre-feasibility-v1'


def group_weights(y, k):
    counts = np.bincount(y, minlength=k)
    if np.any(counts == 0):
        raise ValueError('Cannot balance a training group with no observations')
    weights = len(y) / (k * counts)
    return weights, weights[y]


def weighted_objective(theta, x, y, k, alpha, weights):
    w = theta.reshape(x.shape[1], k)
    scores_ = x @ w
    logp = scores_ - logsumexp(scores_, axis=1, keepdims=True)
    normalizer = float(weights.sum())
    loss = -np.dot(weights, logp[np.arange(len(y)), y]) / normalizer
    loss += .5 * alpha * np.sum(w[:-1] ** 2)
    error = np.exp(logp)
    error[np.arange(len(y)), y] -= 1
    error *= (weights / normalizer)[:, None]
    grad = np.asarray(x.T @ error)
    grad[:-1] += alpha * w[:-1]
    return float(loss), grad.ravel()


def weighted_logistic(x, y, xt, k, alpha):
    _, weights = group_weights(y, k)
    x = sparse.hstack([x, np.ones((x.shape[0], 1))], format='csr')
    xt = sparse.hstack([xt, np.ones((xt.shape[0], 1))], format='csr')
    initial = np.zeros((x.shape[1], k))
    freq = np.bincount(y, weights=weights, minlength=k)
    initial[-1] = np.log(freq / freq.sum())
    result = minimize(weighted_objective, initial.ravel(), args=(x, y, k, alpha, weights), jac=True,
                      method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-10, 'maxcor': 10})
    if not result.success:
        raise RuntimeError(f'Weighted optimizer did not converge: {result.message}')
    pred = np.asarray(xt @ result.x.reshape(x.shape[1], k)).argmax(1)
    return pred, dict(iterations=int(result.nit), loss=float(result.fun), gradient_max=float(np.abs(result.jac).max()))


def balanced_bayes(counts, y, k):
    x, xt = counts
    class_weights, _ = group_weights(y, k)
    sums = np.vstack([np.asarray(x[y == j].sum(0)).ravel() for j in range(k)])
    result = {}
    for name, factors in [('naive_bayes_balanced', class_weights), ('naive_bayes_equal_prior', np.ones(k))]:
        smoothed = sums * factors[:, None] + 1
        logp = np.log(smoothed) - np.log(smoothed.sum(1, keepdims=True))
        # Effective weighted class totals are equal. The common log-prior cancels.
        result[name] = np.asarray(xt @ logp.T).argmax(1)
    return result


def fit_fold(data, train, test, scheme, fold):
    it, iv = inner_split(data, train)
    inner, _, inner_terms = make_features(data, it, iv)
    outer, counts, terms = make_features(data, train, test)
    weights_rows = []
    for stage, ids in [('inner', it), ('outer', train)]:
        weights, examples = group_weights(data.y[ids], 3)
        for j, genre in enumerate(GROUPS):
            mass = examples[data.y[ids] == j].sum()
            assert np.isclose(mass, len(ids) / 3)
            weights_rows.append(dict(scheme=scheme, fold=fold, stage=stage, genre=genre,
                                     training_users=int(sum(data.y[ids] == j)), weight=float(weights[j]),
                                     total_training_weight=float(mass)))
    diagnostic, predictions = [], {}
    print(json.dumps(dict(stage='weighted features', scheme=scheme, fold=fold, train=len(train), test=len(test))), flush=True)
    for name in FEATURES:
        candidates = []
        for alpha in ALPHAS:
            pred, diag = weighted_logistic(inner[name][0], data.y[it], inner[name][1], 3, alpha)
            quality = scores(data.y[iv], pred, 3)['macro_f1']
            candidates.append((quality, alpha))
            diagnostic.append(dict(scheme=scheme, fold=fold, model=name + '_balanced', stage='inner',
                                   alpha=alpha, macro_f1=quality, vocabulary_terms=inner_terms, **diag))
        _, alpha = max(candidates)
        pred, diag = weighted_logistic(outer[name][0], data.y[train], outer[name][1], 3, alpha)
        predictions[name + '_balanced'] = pred
        diagnostic.append(dict(scheme=scheme, fold=fold, model=name + '_balanced', stage='outer',
                               alpha=alpha, macro_f1=scores(data.y[test], pred, 3)['macro_f1'],
                               vocabulary_terms=terms, **diag))
        print(json.dumps(diagnostic[-1]), flush=True)
    predictions.update(balanced_bayes(counts, data.y[train], 3))
    records = [dict(user_key=data.users[i], parent_asin=data.books[i], truth=int(data.y[i]), fold=fold,
                    **{name: int(pred[j]) for name, pred in predictions.items()}) for j, i in enumerate(test)]
    return dict(rows=records, diagnostics=diagnostic, weights=weights_rows)


def detailed(c):
    basic = score_matrix(c)
    tp, support, predicted = np.diag(c), c.sum(1), c.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    return np.r_[basic['macro_f1'], basic['balanced_accuracy'], basic['accuracy'], precision, recall, f1]


def summarize(scheme, rows):
    names = [c for c in rows[0] if c not in {'user_key', 'parent_asin', 'truth', 'fold'}]
    truth = np.array([int(r['truth']) for r in rows])
    pred = {m: np.array([int(r[m]) for r in rows]) for m in names}
    titles, inverse = np.unique([r['parent_asin'] for r in rows], return_inverse=True)
    rng = np.random.default_rng(2311)
    boot = {m: [] for m in names}
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(titles), size=len(titles)), minlength=len(titles))
        sample_weight = counts[inverse]
        for name, predictions in pred.items():
            boot[name].append(detailed(confusion(truth, predictions, 3, sample_weight)))
    boot = {m: np.asarray(v) for m, v in boot.items()}
    metrics, groups, matrices, fold_rows = [], [], [], []
    point = {}
    for name, predicted in pred.items():
        c = confusion(truth, predicted, 3)
        point[name] = detailed(c)
        lo, hi = np.quantile(boot[name], [.025, .975], axis=0)
        metrics.append(dict(scheme=scheme, model=name, users=len(truth), books=len(titles),
                            macro_f1=point[name][0], balanced_accuracy=point[name][1], accuracy=point[name][2],
                            macro_f1_low=lo[0], macro_f1_high=hi[0],
                            balanced_accuracy_low=lo[1], balanced_accuracy_high=hi[1]))
        for j, genre in enumerate(GROUPS):
            groups.append(dict(scheme=scheme, model=name, genre=genre, users=int(c[j].sum()),
                               predicted_users=int(c[:, j].sum()), precision=point[name][3 + j],
                               recall=point[name][6 + j], f1=point[name][9 + j]))
            for l, predicted_genre in enumerate(GROUPS):
                matrices.append(dict(scheme=scheme, model=name, actual=genre, predicted=predicted_genre, users=int(c[j, l])))
        folds_ = np.array([int(r['fold']) for r in rows])
        for fold in range(3):
            mask = folds_ == fold
            fold_rows.append(dict(scheme=scheme, model=name, fold=fold, **scores(truth[mask], predicted[mask], 3)))
    pairs = [(m + '_balanced', m) for m in FEATURES]
    pairs += [('naive_bayes_balanced', 'naive_bayes'), ('naive_bayes_equal_prior', 'naive_bayes')]
    pairs += [('context_combined_balanced', 'context_balanced')]
    changes, group_changes = [], []
    for a, b in pairs:
        delta = point[a] - point[b]
        lo, hi = np.quantile(boot[a] - boot[b], [.025, .975], axis=0)
        for j, metric in enumerate(['macro_f1', 'balanced_accuracy', 'accuracy']):
            changes.append(dict(scheme=scheme, model=a, comparator=b, metric=metric,
                                difference=delta[j], low=lo[j], high=hi[j]))
        for j, genre in enumerate(GROUPS):
            for metric, offset in [('precision', 3), ('recall', 6), ('f1', 9)]:
                group_changes.append(dict(scheme=scheme, model=a, comparator=b, genre=genre, metric=metric,
                                          difference=delta[offset + j], low=lo[offset + j], high=hi[offset + j]))
    for name, records in [('metrics', metrics), ('per_genre', groups), ('confusion', matrices),
                          ('fold_metrics', fold_rows), ('paired_metrics', changes), ('paired_genre', group_changes)]:
        write_csv(PUB / f'{scheme}_{name}.csv', records)
    print('RESULT', scheme, [(m['model'], round(m['macro_f1'], 4), round(m['balanced_accuracy'], 4))
                            for m in metrics if m['model'] in ['combined', 'combined_balanced', 'naive_bayes', 'naive_bayes_balanced']], flush=True)


def run(scheme, common):
    data = load_data(common)
    assigned = folds(data, 'book')
    rows, diagnostics, weights = [], [], []
    for fold in range(3):
        train, test = np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold)
        assert not (set(data.books[train]) & set(data.books[test]))
        assert not (set(data.users[train]) & set(data.users[test]))
        checkpoint = OUT / f'{scheme}_fold{fold}.json'
        if checkpoint.exists():
            result = json.loads(checkpoint.read_text())
        else:
            result = fit_fold(data, train, test, scheme, fold)
            checkpoint.write_text(json.dumps(result), encoding='utf-8')
        rows.extend(result['rows'])
        diagnostics.extend(result['diagnostics'])
        weights.extend(result['weights'])
    old = {r['user_key']: r for r in read_csv(OLD / f'{scheme}_predictions.csv')}
    merged = []
    assert len(rows) == len(old) == len(data.y)
    for row in rows:
        before = old[row['user_key']]
        for field in ['parent_asin', 'truth', 'fold']:
            assert str(row[field]) == str(before[field]), (row['user_key'], field)
        merged.append(dict(before, **row))
    assert len({r['user_key'] for r in merged}) == len(merged)
    write_csv(OUT / f'{scheme}_predictions.csv', merged)
    write_csv(PUB / f'{scheme}_fit_diagnostics.csv', diagnostics)
    write_csv(PUB / f'{scheme}_training_weights.csv', weights)
    summarize(scheme, merged)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    run('all', False)
    run('thriller_input', True)
