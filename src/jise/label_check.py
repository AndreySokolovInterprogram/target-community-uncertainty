"""Manual label check (one annotator, final version; 300 groups = 150 'delivery' + 150 'other', seed 20261020).

1. Label noise per class and for the whole data set (weighted by the real class shares), Wilson intervals.
2. Out-of-fold model predictions (query-masked variant) vs. the original and the corrected label,
   weighted back to the real class shares; what models do with the falsely labelled 'delivery' groups.
Outputs: results/jise/groups/label_check/.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from jise.common import ROOT, RES, load_groups, run_info, save_json

ANN = ROOT / 'annotation' / 'manual_label_check.xlsx'
OUT = RES / 'groups' / 'label_check'
COL = 'label_correct'
TEXT = 'title_desc_masked'
PRED = RES / 'groups' / 'preds' / TEXT
# Category of each wrong label: assigned by us from title + description (the annotator gave only yes/no).
CATEGORY = {
    9: 'recipes / food blog / mukbang', 47: 'recipes / food blog / mukbang', 205: 'recipes / food blog / mukbang',
    218: 'recipes / food blog / mukbang', 234: 'recipes / food blog / mukbang',
    89: 'promo codes / aggregator', 119: 'promo codes / aggregator', 92: 'promo codes / aggregator',
    96: 'courier jobs',
    7: 'not food (shop, rental, flowers, etc.)', 148: 'not food (shop, rental, flowers, etc.)',
    273: 'not food (shop, rental, flowers, etc.)', 285: 'not food (shop, rental, flowers, etc.)',
    291: 'not food (shop, rental, flowers, etc.)',
    29: 'food-related, not a ready-food service', 157: 'food-related, not a ready-food service',
    230: 'food-related, not a ready-food service',
    19: 'delivery-like by text, judged not a ready-food delivery service', 111: 'delivery-like by text, judged not a ready-food delivery service',
    177: 'delivery-like by text, judged not a ready-food delivery service', 271: 'delivery-like by text, judged not a ready-food delivery service',
    279: 'delivery-like by text, judged not a ready-food delivery service',
}
MODELS = {   # name -> loader returning OOF P(1) for all rows
    'e5-base fine-tuned, ens5': ('transformers', 'ft_e5base', 5),
    'e5-base frozen + LR': ('classic', 'e5base_lr', 1),
    'TF-IDF + LR': ('classic', 'tfidf_lr', 1),
    'CNN, ens5': ('keras', 'cnn', 5),
    'qwen2.5-7b, 4-shot': ('llm', 'qwen2.5-7b-instruct_4shot', 0),
    'qwen3-8b, 4-shot': ('llm', 'qwen3-8b_4shot', 0),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def oof_p1(sub, name, n_seeds, n_rows):
    p = np.full(n_rows, np.nan)
    for k in range(5):
        if sub == 'llm':
            z = np.load(PRED / sub / f'{name}_f{k}.npz')
            p[z['te_idx']] = z['p1']
        else:
            zs = [np.load(PRED / sub / f'{name}_f{k}_s{s}.npz') for s in range(n_seeds)]
            p[zs[0]['te_idx']] = np.mean([z['p_te'][:, 1] for z in zs], 0)
    assert np.isfinite(p).all()
    return p


def weighted_acc(correct, strata, w):
    return float(np.sum(w * correct) / np.sum(w))


def strat_boot(correct, strata, w, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = [np.where(strata == s)[0] for s in np.unique(strata)]
    vals = []
    for _ in range(B):
        b = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx])
        vals.append(np.sum(w[b] * correct[b]) / np.sum(w[b]))
    return np.percentile(vals, [2.5, 97.5])


def main(dp_as='yes'):
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_excel(ANN, 'annotation')
    raw_vals = a[COL].value_counts(dropna=False).to_dict()
    a['verdict'] = a[COL].replace({'unclear': dp_as})
    assert set(a['verdict']) <= {'yes', 'no'}
    a['y_orig'] = (a['label'] == 'delivery').astype(int)
    a['wrong'] = (a['verdict'] == 'no').astype(int)
    a['y_true'] = np.where(a['wrong'] == 1, 1 - a['y_orig'], a['y_orig'])

    g = load_groups(TEXT)
    N = len(g)
    share = {1: (g['label'] == 1).mean(), 0: (g['label'] == 0).mean()}
    pos = pd.Series(np.arange(N), index=g['group_id'].values)
    a['row'] = pos.loc[a['vk_id'].values].values
    assert (g['label'].values[a['row']] == a['y_orig']).all()

    # 1. noise
    rows = []
    for c, name in [(1, 'delivery'), (0, 'other')]:
        s = a[a['y_orig'] == c]
        k, n = int(s['wrong'].sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({'class': name, 'n_checked': n, 'wrong': k, 'share_wrong': k / n, 'wilson_lo': lo,
                     'wilson_hi': hi, 'class_share_in_data': share[c]})
    t = pd.DataFrame(rows)
    p_all = float(sum(t['share_wrong'] * t['class_share_in_data']))
    lo_all = float(sum(t['wilson_lo'] * t['class_share_in_data']))
    hi_all = float(sum(t['wilson_hi'] * t['class_share_in_data']))
    t.loc[len(t)] = {'class': 'whole data set (weighted)', 'n_checked': len(a), 'wrong': int(a['wrong'].sum()),
                     'share_wrong': p_all, 'wilson_lo': lo_all, 'wilson_hi': hi_all, 'class_share_in_data': 1.0}
    t.to_csv(OUT / 'label_noise.csv', index=False)
    wrong = a[a['wrong'] == 1][['n', 'vk_id', 'title', 'label']].copy()
    wrong['category'] = wrong['n'].map(CATEGORY)
    assert wrong['category'].notna().all(), 'category missing'
    wrong.to_csv(OUT / 'wrong_labels.csv', index=False)
    cats = wrong['category'].value_counts().rename_axis('category').reset_index(name='n')

    # 2. models
    w = np.where(a['y_orig'] == 1, share[1] / 0.5, share[0] / 0.5)   # 150+150 -> real class shares
    strata = a['y_orig'].values
    mrows, frows = [], []
    false_del = a['wrong'].values == 1
    true_del = (a['y_orig'].values == 1) & ~false_del
    for name, (sub, key, ns) in MODELS.items():
        p = oof_p1(sub, key, ns, N)
        conf = np.maximum(p, 1 - p)
        thr10 = np.quantile(conf, 0.10)                     # 10 % least confident over all 11,612 groups
        ps = p[a['row']]
        yhat = (ps >= 0.5).astype(int)
        r = {'model': name}
        for lab, col in [('orig', 'y_orig'), ('corrected', 'y_true')]:
            corr = (yhat == a[col].values).astype(float)
            r[f'acc_{lab}'] = weighted_acc(corr, strata, w)
            r[f'acc_{lab}_lo'], r[f'acc_{lab}_hi'] = strat_boot(corr, strata, w)
            r[f'acc_{lab}_unweighted'] = corr.mean()
        r['acc_full_data_orig_label'] = float(((p >= 0.5).astype(int) == g['label'].values).mean())
        mrows.append(r)
        unc = conf[a['row']] <= thr10
        frows.append({'model': name, 'false_delivery_n': int(false_del.sum()),
                      'predicted_other': int((yhat[false_del] == 0).sum()),
                      'in_10pct_least_confident': int(unc[false_del].sum()),
                      'predicted_other_or_uncertain': int(((yhat == 0) | unc)[false_del].sum()),
                      'true_delivery_n': int(true_del.sum()),
                      'true_delivery_predicted_other': int((yhat[true_del] == 0).sum()),
                      'true_delivery_in_10pct_least_confident': int(unc[true_del].sum())})
        a[f'p1 | {name}'] = ps
    m = pd.DataFrame(mrows)
    f = pd.DataFrame(frows)
    m.to_csv(OUT / 'model_accuracy_manual_labels.csv', index=False)
    f.to_csv(OUT / 'false_delivery_handling.csv', index=False)
    a.drop(columns=['description']).to_csv(OUT / 'annotation_with_oof.csv', index=False)

    with open(OUT / 'label_check.md', 'w', encoding='utf-8') as fh:
        fh.write(f'Manual check (one annotator, final version), 300 groups (150 delivery + 150 other, seed 20261020). Raw verdicts: {raw_vals}; '
                 f"'unclear' treated as '{dp_as}'.\n\n## Label noise (Wilson 95 %; overall = class-share weighted, "
                 'interval = weighted Wilson bounds, conservative)\n\n')
        fh.write(t.to_markdown(index=False, floatfmt='.4f') + '\n\n## Wrong labels by category (our categories)\n\n')
        fh.write(cats.to_markdown(index=False) + '\n\n' + wrong.to_markdown(index=False) + '\n\n')
        fh.write('## OOF predictions (query-masked) on the 300 groups, accuracy weighted to real class shares; '
                 '95 % CI = stratified bootstrap\n\n')
        mm = m[['model']].copy()
        for lab in ['orig', 'corrected']:
            mm[f'acc vs {lab} label'] = [f'{x:.3f} [{lo:.3f}, {hi:.3f}]' for x, lo, hi in
                                         zip(m[f'acc_{lab}'], m[f'acc_{lab}_lo'], m[f'acc_{lab}_hi'])]
        mm['acc on all 11,612 (orig label)'] = m['acc_full_data_orig_label'].map('{:.4f}'.format)
        fh.write(mm.to_markdown(index=False) + '\n\n## Falsely labelled "delivery" groups\n\n')
        fh.write(f.to_markdown(index=False) + '\n')
    save_json(run_info({'dp_as': dp_as, 'raw_verdicts': raw_vals, 'weights': {'delivery': share[1] / 0.5,
                                                                              'other': share[0] / 0.5}}),
              OUT / 'info.json')
    print(open(OUT / 'label_check.md', encoding='utf-8').read())
    return t, m, f


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'yes')
