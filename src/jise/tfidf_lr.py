"""Baselines on the same folds: TF-IDF + logistic regression, frozen sentence embeddings + LR."""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jise.common import TEXT_VARIANTS, RES, PROC, N_FOLDS, SPLIT_SEED, load_groups, group_folds, inner_val_split, run_info, save_json

C_GRID = [0.25, 1, 4, 16]


def tfidf_features(tr_text, other):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                        max_features=200000)
    Xtr = hstack([w.fit_transform(tr_text), c.fit_transform(tr_text)]).tocsr()
    return Xtr, [hstack([w.transform(t), c.transform(t)]).tocsr() for t in other]


def lr(C, seed=0):
    return LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=seed)


def main(feats, text):
    g = load_groups(text)
    fold = group_folds(g)
    y = g['label'].values
    out = RES / 'groups' / 'preds' / text / 'classic'
    out.mkdir(parents=True, exist_ok=True)
    for feat in feats:
        E = None if feat == 'tfidf' else np.load(PROC / f'emb_{feat}_{text}.npy')
        for k in range(N_FOLDS):
            te = np.where(fold == k)[0]
            trf = np.where(fold != k)[0]
            tr_rel, va_rel = inner_val_split(g.iloc[trf], SPLIT_SEED + k)
            tr, va = trf[tr_rel], trf[va_rel]
            t0 = time.time()
            if feat == 'tfidf':
                Xtr, (Xva, Xte) = tfidf_features(g['text'].values[tr], [g['text'].values[va], g['text'].values[te]])
            else:
                Xtr, Xva, Xte = E[tr], E[va], E[te]
            best = max(C_GRID, key=lambda C: (lr(C).fit(Xtr, y[tr]).predict(Xva) == y[va]).mean())
            m = lr(best).fit(Xtr, y[tr])
            p_te, p_va = m.predict_proba(Xte), m.predict_proba(Xva)
            np.savez_compressed(out / f'{feat}_lr_f{k}_s0.npz', te_idx=te, va_idx=va, y_te=y[te], y_va=y[va],
                                p_te=p_te, p_va=p_va)
            save_json({'model': f'{feat}_lr', 'fold': k, 'C': best, 'fit_sec': time.time() - t0,
                       'test_acc': float((p_te.argmax(1) == y[te]).mean())}, out / f'{feat}_lr_f{k}_s0.json')
            print(f'{feat}_lr fold {k}: C={best} acc={(p_te.argmax(1) == y[te]).mean():.4f}', flush=True)
    save_json(run_info({'feats': feats, 'text': text}), out / f'run_info_{int(time.time())}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='tfidf')
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    a = ap.parse_args()
    main(a.feats.split(','), a.text)
