"""Module 1, active learning: quality vs. number of labelled groups (10, 20, ..., 100).

For every outer fold (test = that fold, pool = the other four) and every repeat (5 seeds):
  * start: 10 random groups from the pool (redrawn until both classes are present);
  * each round: train LR, add 10 groups chosen by the strategy, retrain;
  * strategies: random; uncertainty = smallest |p(1) - 0.5|.
    For two classes entropy, margin and least-confidence give the same ranking, so they are one strategy.
Features: TF-IDF (fitted on the pool texts only) or frozen embeddings (e5-base, rubert-tiny2).
Reported: test accuracy / macro-F1 / AUC at each budget, mean ± std over 5 folds x 5 repeats.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from jise.common import TEXT_VARIANTS, RES, PROC, N_FOLDS, load_groups, group_folds, run_info, save_json
from jise.tfidf_lr import tfidf_features

BUDGETS = list(range(10, 101, 10))
STEP = 10
REPEATS = [0, 1, 2, 3, 4]


def initial_set(y_pool, rng, n=10):
    for _ in range(1000):
        s = rng.choice(len(y_pool), n, replace=False)
        if len(np.unique(y_pool[s])) == 2:
            return list(s)
    raise RuntimeError('no two-class start set')


def clf():
    return LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced')


def run(feat, text):
    g = load_groups(text)
    fold = group_folds(g)
    y = g['label'].values
    E = None if feat == 'tfidf' else np.load(PROC / f'emb_{feat}_{text}.npy')
    rows = []
    t0 = time.time()
    for k in range(N_FOLDS):
        te = np.where(fold == k)[0]
        pool = np.where(fold != k)[0]
        if feat == 'tfidf':
            Xp, (Xt,) = tfidf_features(g['text'].values[pool], [g['text'].values[te]])
        else:
            Xp, Xt = E[pool], E[te]
        yp, yt = y[pool], y[te]
        full = clf().fit(Xp, yp).predict_proba(Xt)
        rows.append(dict(feat=feat, text=text, fold=k, repeat=-1, strategy='full_pool', n_labeled=len(pool),
                         accuracy=accuracy_score(yt, full.argmax(1)),
                         macro_f1=f1_score(yt, full.argmax(1), average='macro'),
                         auc=roc_auc_score(yt, full[:, 1])))
        for rep in REPEATS:
            start = initial_set(yp, np.random.default_rng(1000 * k + rep))
            for strat in ['random', 'uncertainty']:
                rng = np.random.default_rng(10_000 + 1000 * k + rep)
                lab = list(start)
                while True:
                    m = clf().fit(Xp[lab], yp[lab])
                    pt = m.predict_proba(Xt)
                    rows.append(dict(feat=feat, text=text, fold=k, repeat=rep, strategy=strat, n_labeled=len(lab),
                                     accuracy=accuracy_score(yt, pt.argmax(1)),
                                     macro_f1=f1_score(yt, pt.argmax(1), average='macro'),
                                     auc=roc_auc_score(yt, pt[:, 1]),
                                     pos_share_labeled=float(yp[lab].mean())))
                    if len(lab) >= BUDGETS[-1]:
                        break
                    rest = np.setdiff1d(np.arange(len(yp)), lab)
                    if strat == 'random':
                        new = rng.choice(rest, STEP, replace=False)
                    else:
                        pp = m.predict_proba(Xp[rest])[:, 1]
                        new = rest[np.argsort(np.abs(pp - 0.5), kind='stable')[:STEP]]
                    lab += list(new)
        print(f'AL {feat}/{text} fold {k} done, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    out = RES / 'groups' / 'active_learning'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'al_raw_{feat}_{text}.csv', index=False)
    save_json(run_info({'feat': feat, 'text': text, 'budgets': BUDGETS, 'step': STEP, 'repeats': REPEATS,
                        'sec': time.time() - t0}), out / f'al_run_info_{feat}_{text}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='tfidf')
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    a = ap.parse_args()
    for f in a.feats.split(','):
        run(f, a.text)
