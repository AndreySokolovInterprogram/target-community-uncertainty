"""Post-hoc statistics on existing out-of-fold predictions (query-masked groups); no training.

* Paired comparison of the best models on all 11,612 groups: exact McNemar test (Holm-corrected),
  cluster bootstrap over duplicate groups for the accuracy difference, per-fold differences.
* Temperature scaling for the LLM: for test fold k the temperature is fitted on the OOF P(1) of the inner
  validation part of the training folds (the same split as for the other models); ECE / NLL before and after.
Outputs: results/jise/groups/stats/.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from jise.common import RES, SPLIT_SEED, load_groups, group_folds, inner_val_split, run_info, save_json
from jise.label_check import MODELS as _M, oof_p1, PRED
from jise.metrics import ece, fit_temperature, apply_temperature
from sklearn.metrics import log_loss

OUT = RES / 'groups' / 'stats'
MODELS = dict(_M)
MODELS['rubert-tiny2 fine-tuned, ens5'] = ('transformers', 'ft_rubert_tiny2', 5)
PAIRS_FIRST = 'e5-base fine-tuned, ens5'
B = 2000


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_groups('title_desc_masked')
    y = g['label'].values
    fold = group_folds(g)
    clusters = g['dup_group'].values
    P = {m: oof_p1(*v[:2], v[2], len(g)) for m, v in MODELS.items()}
    C = {m: ((p >= 0.5).astype(int) == y) for m, p in P.items()}

    pairs = [(PAIRS_FIRST, m) for m in MODELS if m != PAIRS_FIRST]
    pairs += [('e5-base frozen + LR', 'TF-IDF + LR'), ('e5-base frozen + LR', 'CNN, ens5'),
              ('TF-IDF + LR', 'CNN, ens5'), ('qwen2.5-7b, 4-shot', 'qwen3-8b, 4-shot')]
    uc, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(0)
    rows = []
    for a, b in pairs:
        ca, cb = C[a], C[b]
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        d = ca.astype(float) - cb.astype(float)
        # cluster bootstrap: resample duplicate groups
        s_d = np.bincount(inv, weights=d, minlength=len(uc))
        s_n = np.bincount(inv, minlength=len(uc)).astype(float)
        boots = []
        for _ in range(B):
            i = rng.integers(0, len(uc), len(uc))
            boots.append(s_d[i].sum() / s_n[i].sum())
        fd = [d[fold == k].mean() for k in range(5)]
        rows.append({'A': a, 'B': b, 'acc_A': ca.mean(), 'acc_B': cb.mean(), 'diff': d.mean(),
                     'diff_ci_lo': np.percentile(boots, 2.5), 'diff_ci_hi': np.percentile(boots, 97.5),
                     'A_right_B_wrong': n01, 'A_wrong_B_right': n10, 'mcnemar_p': p,
                     'fold_diff_mean': np.mean(fd), 'fold_diff_std': np.std(fd, ddof=1),
                     'folds_A_better': int(sum(x > 0 for x in fd))})
    t = pd.DataFrame(rows)
    t['mcnemar_p_holm'] = holm(t['mcnemar_p'].values)
    t.to_csv(OUT / 'paired_tests.csv', index=False)

    # LLM temperature scaling with inner validation
    trows = []
    for tag in ['qwen2.5-7b-instruct_0shot', 'qwen2.5-7b-instruct_4shot', 'qwen3-8b_0shot', 'qwen3-8b_4shot']:
        p1 = oof_p1('llm', tag, 0, len(g))
        logit = np.log(np.clip(p1, 1e-12, 1)) - np.log(np.clip(1 - p1, 1e-12, 1))
        L = np.stack([np.zeros_like(logit), np.clip(logit, -30, 30)], 1)
        for k in range(5):
            te = np.where(fold == k)[0]
            trf = np.where(fold != k)[0]
            _, va_rel = inner_val_split(g.iloc[trf], SPLIT_SEED + k)
            va = trf[va_rel]
            T = fit_temperature(L[va], y[va])
            p_before = apply_temperature(L[te], 1.0)
            p_after = apply_temperature(L[te], T)
            trows.append({'model': tag, 'fold': k, 'T': T, 'ece_before': ece(p_before, y[te]),
                          'ece_after': ece(p_after, y[te]),
                          'nll_before': log_loss(y[te], np.clip(p_before, 1e-7, 1 - 1e-7), labels=[0, 1]),
                          'nll_after': log_loss(y[te], np.clip(p_after, 1e-7, 1 - 1e-7), labels=[0, 1])})
    tt = pd.DataFrame(trows)
    tt.to_csv(OUT / 'llm_temperature_by_fold.csv', index=False)
    ts = tt.groupby('model').agg(['mean', 'std']).drop(columns='fold')
    ts.columns = [f'{a}_{b}' for a, b in ts.columns]
    ts = ts.reset_index()
    ts.to_csv(OUT / 'llm_temperature.csv', index=False)

    with open(OUT / 'stats.md', 'w', encoding='utf-8') as f:
        f.write('Query-masked groups, OOF predictions on all 11,612 groups.\n\n## Paired comparisons\n\n'
                'diff = acc(A) − acc(B); CI = cluster bootstrap over duplicate groups (B=2000); McNemar exact, '
                'Holm-adjusted over all pairs; folds_A_better = of 5 folds.\n\n')
        show = t[['A', 'B', 'acc_A', 'acc_B', 'diff', 'diff_ci_lo', 'diff_ci_hi', 'A_right_B_wrong', 'A_wrong_B_right',
                  'mcnemar_p_holm', 'folds_A_better']]
        f.write(show.to_markdown(index=False, floatfmt='.4g') + '\n\n## LLM temperature scaling '
                '(T fitted on inner validation of the training folds)\n\n')
        f.write(ts.to_markdown(index=False, floatfmt='.4f') + '\n\nNote: for 4-shot the validation predictions '
                'were produced with the few-shot examples of their own fold; for 0-shot the prompt is identical '
                'across folds.\n')
    save_json(run_info({'B': B}), OUT / 'info.json')
    print(open(OUT / 'stats.md', encoding='utf-8').read())


if __name__ == '__main__':
    main()
