"""Aggregate group-classification predictions into tables and figures (results/jise/groups/).

Per model family (files <model>_f<fold>_s<seed>.npz):
  single      - one model; fold value = mean over seeds; reported mean ± std over the 5 folds
  single+TS   - same, temperature fitted on the inner validation of that run
  ens5        - mean probability of the 5 seeds (deep ensemble); mean ± std over folds
  ens5+TS     - ensemble with temperature fitted on the ensemble's validation probabilities
  last_epoch  - single model scored with last-epoch weights (what the CNN-weights bug did to LSTM)
MC-dropout files (<model>_mc_f<fold>_s<seed>.npz with p_te_mc of shape (T, n, 2)) are used when present.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jise.common import RES, TEXT_VARIANTS, run_info, save_json
from jise.metrics import (all_metrics, entropy, fit_temperature, apply_temperature, to_logits,
                          risk_coverage, reliability_bins)

G = RES / 'groups'
PRED = G / 'preds'
NAMES = {'tfidf_lr': 'TF-IDF + LR', 'rubert_tiny2_lr': 'rubert-tiny2 (frozen) + LR',
         'e5base_lr': 'e5-base (frozen) + LR', 'cnn': 'CNN', 'lstm': 'LSTM', 'gru': 'GRU',
         'ft_rubert_tiny2': 'rubert-tiny2 (fine-tuned)', 'ft_e5base': 'e5-base (fine-tuned)'}
SHOW = ['accuracy', 'macro_f1', 'f1_class0', 'f1_class1', 'auc', 'ece', 'nll', 'brier', 'aurc',
        'acc@cov90', 'acc@cov80']


def load(pdir):
    runs = defaultdict(dict)
    for f in sorted(pdir.glob('*.npz')):
        stem = f.stem
        model, rest = stem.rsplit('_f', 1)
        k, s = rest.split('_s')
        runs[model][(int(k), int(s))] = dict(np.load(f))
    return runs


def mi_score(P):
    """Ensemble/MC mutual information (epistemic part); returned as confidence = -MI."""
    pm = P.mean(0)
    return -(entropy(pm) - np.mean([entropy(p) for p in P], 0))


def summarize(model, runs):
    folds = sorted({k for k, _ in runs})
    seeds = sorted({s for _, s in runs})
    rec = defaultdict(list)       # variant -> list of per-fold metric dicts
    pooled = defaultdict(lambda: {'p': [], 'y': [], 'score': []})
    for k in folds:
        rs = [runs[(k, s)] for s in seeds if (k, s) in runs]
        y = rs[0]['y_te']
        per_seed = defaultdict(list)
        for r in rs:
            per_seed['single'].append(all_metrics(r['p_te'], y))
            if 'p_va' in r and r['p_va'] is not None and r['p_va'].ndim == 2:
                T = fit_temperature(to_logits(r['p_va']), r['y_va'])
                per_seed['single+TS'].append(all_metrics(apply_temperature(to_logits(r['p_te']), T), y))
            if 'p_last_te' in r:
                per_seed['last_epoch'].append(all_metrics(r['p_last_te'], y))
        for v, lst in per_seed.items():
            rec[v].append(pd.DataFrame(lst).mean().to_dict())
        r0 = rs[0]
        pooled['single (seed 0)']['p'].append(r0['p_te']); pooled['single (seed 0)']['y'].append(y)
        pooled['single (seed 0)']['score'].append(r0['p_te'].max(1))
        if len(rs) > 1:
            P = np.stack([r['p_te'] for r in rs])
            pe = P.mean(0)
            rec['ens5'].append(all_metrics(pe, y))
            rec['ens5 (MI score)'].append(all_metrics(pe, y, score=mi_score(P)))
            if 'p_va' in r0 and r0['p_va'].ndim == 2:
                Pv = np.stack([r['p_va'] for r in rs]).mean(0)
                T = fit_temperature(to_logits(Pv), r0['y_va'])
                pts = apply_temperature(to_logits(pe), T)
                rec['ens5+TS'].append(all_metrics(pts, y))
                pooled['ens5+TS']['p'].append(pts); pooled['ens5+TS']['y'].append(y)
                pooled['ens5+TS']['score'].append(pts.max(1))
            pooled['ens5']['p'].append(pe); pooled['ens5']['y'].append(y); pooled['ens5']['score'].append(pe.max(1))
            pooled['ens5 (MI score)']['p'].append(pe); pooled['ens5 (MI score)']['y'].append(y)
            pooled['ens5 (MI score)']['score'].append(mi_score(P))
        if 'p_te_mc' in r0:
            Pmc = r0['p_te_mc']
            pmc = Pmc.mean(0)
            rec['mc_dropout'].append(all_metrics(pmc, y))
            rec['mc_dropout (MI score)'].append(all_metrics(pmc, y, score=mi_score(Pmc)))
            pooled['mc_dropout']['p'].append(pmc); pooled['mc_dropout']['y'].append(y)
            pooled['mc_dropout']['score'].append(pmc.max(1))
    rows = []
    for v, lst in rec.items():
        df = pd.DataFrame(lst)
        row = {'model': NAMES.get(model, model), 'key': model, 'variant': v, 'n_folds': len(df),
               'n_seeds': len(seeds)}
        for m in SHOW:
            row[m] = df[m].mean()
            row[m + '_std'] = df[m].std(ddof=1) if len(df) > 1 else np.nan
        rows.append(row)
    pooled = {v: {kk: np.concatenate(vv) for kk, vv in d.items()} for v, d in pooled.items()}
    return rows, pooled


def fmt_table(df, cols):
    out = df[['model', 'variant']].copy()
    for c in cols:
        out[c] = [f'{m:.4f} ± {s:.4f}' if np.isfinite(s) else f'{m:.4f}' for m, s in zip(df[c], df[c + '_std'])]
    return out


def evaluate(text):
    out = G / text
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_pooled = [], {}
    for sub in ['classic', 'keras', 'transformers']:
        if not (PRED / text / sub).exists():
            continue
        for model, runs in load(PRED / text / sub).items():
            rows, pooled = summarize(model, runs)
            all_rows += rows
            all_pooled[model] = pooled
    df = pd.DataFrame(all_rows)
    df.to_csv(out / 'table_all_variants.csv', index=False)
    main_v = df[df['variant'].isin(['single', 'ens5', 'ens5+TS', 'mc_dropout'])]
    with open(out / 'table_main.md', 'w', encoding='utf-8') as f:
        f.write(f'Groups, text = {text}, 5-fold StratifiedGroupKFold (duplicates in one fold). mean ± std over folds.\n\n')
        f.write(fmt_table(main_v, ['accuracy', 'macro_f1', 'f1_class0', 'f1_class1', 'auc']).to_markdown(index=False))
        f.write('\n\nCalibration and selective prediction\n\n')
        f.write(fmt_table(df[~df['variant'].isin(['last_epoch'])],
                          ['ece', 'nll', 'brier', 'aurc', 'acc@cov90', 'acc@cov80']).to_markdown(index=False))
        le = df[df['variant'].isin(['single', 'last_epoch'])]
        if len(le[le['variant'] == 'last_epoch']):
            f.write('\n\nBest checkpoint vs last epoch (effect of loading the wrong weights)\n\n')
            f.write(fmt_table(le[le['key'].isin(['cnn', 'lstm', 'gru'])], ['accuracy', 'macro_f1', 'auc'])
                    .to_markdown(index=False))

    # risk-coverage and reliability (pooled over folds: each group is in test exactly once)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rc_rows = []
    for model, pooled in all_pooled.items():
        for v in ['single (seed 0)', 'ens5', 'ens5 (MI score)', 'mc_dropout']:
            if v not in pooled:
                continue
            cov, acc, aurc = risk_coverage(pooled[v]['p'], pooled[v]['y'], pooled[v]['score'])
            for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
                rc_rows.append(dict(model=NAMES.get(model, model), variant=v, coverage=c,
                                    accuracy=acc[max(1, int(np.ceil(c * len(acc)))) - 1], aurc=aurc))
            if v in ('ens5', 'single (seed 0)') and (v == 'ens5' or 'ens5' not in pooled):
                ax[0].plot(cov, acc, label=f'{NAMES.get(model, model)} ({v})', lw=1.4)
    ax[0].set_xlabel('coverage'); ax[0].set_ylabel('accuracy on retained groups'); ax[0].set_xlim(0.3, 1)
    ax[0].set_ylim(0.85, 1.0); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7); ax[0].set_title(f'Risk–coverage (groups, {text})')
    for model, pooled in all_pooled.items():
        v = 'ens5' if 'ens5' in pooled else 'single (seed 0)'
        b = reliability_bins(pooled[v]['p'], pooled[v]['y'], 10)
        xs = [r[4] for r in b if r[2] > 0]; ys = [r[3] for r in b if r[2] > 0]
        ax[1].plot(xs, ys, marker='o', ms=3, lw=1.2, label=f'{NAMES.get(model, model)} ({v})')
    ax[1].plot([0.5, 1], [0.5, 1], 'k--', lw=.8)
    ax[1].set_xlabel('mean confidence'); ax[1].set_ylabel('accuracy'); ax[1].set_title(f'Reliability (groups, {text})')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'fig_risk_coverage_reliability.png', dpi=200); plt.close(fig)
    pd.DataFrame(rc_rows).to_csv(out / 'table_risk_coverage.csv', index=False)

    print(text); print(fmt_table(main_v, ['accuracy', 'macro_f1', 'auc', ]).to_string(index=False))


def orig_ablation():
    """Original-protocol ablation (Embedding bug, last-epoch)."""
    orig = []
    for f in sorted((PRED / 'keras_orig').glob('*.json')):
        if f.name.startswith('run_info'):
            continue
        orig.append(json.loads(f.read_text(encoding='utf-8')))
    if orig:
        o = pd.DataFrame(orig)
        t = o.groupby('model').agg(n=('seed', 'size'), acc_best=('test_acc', 'mean'), acc_best_std=('test_acc', 'std'),
                                   acc_last=('test_acc_last_epoch', 'mean'),
                                   acc_last_std=('test_acc_last_epoch', 'std'),
                                   best_epoch=('best_epoch', 'mean')).reset_index()
        t.to_csv(G / 'table_orig_protocol_ablation.csv', index=False)
        with open(G / 'table_orig_protocol_ablation.md', 'w', encoding='utf-8') as f:
            f.write('Notebook protocol (dropna, description only, random 70/30, val = last 10 % of train), '
                    'mean ± std over seeds\n\n' + t.to_markdown(index=False, floatfmt='.4f'))
        print(t.to_string(index=False))


def llm_table(text):
    """Ollama zero-/few-shot: hard answers + P(1) from token log-probs, mean ± std over folds."""
    d = PRED / text / 'llm'
    if not d.exists():
        return
    tags = sorted({f.stem.rsplit('_f', 1)[0] for f in d.glob('*_f*.npz')})
    rows = []
    for tag in tags:
        per = []
        for k in range(5):
            f = d / f'{tag}_f{k}.npz'
            if not f.exists():
                continue
            z = np.load(f)
            y, hard, p1 = z['y_te'], z['hard'], z['p1']
            row = {'accuracy': float((hard == y).mean()), 'invalid': int((hard < 0).sum()),
                   'with_prob': float(np.isfinite(p1).mean())}
            from sklearn.metrics import f1_score
            row['macro_f1'] = f1_score(y, np.where(hard < 0, 0, hard), average='macro')
            if np.isfinite(p1).all():
                m = all_metrics(np.stack([1 - p1, p1], 1), y)
                row.update({kk: m[kk] for kk in ['auc', 'ece', 'nll', 'aurc', 'acc@cov90', 'acc@cov80']})
            per.append(row)
        if not per:
            continue
        df = pd.DataFrame(per)
        r = {'model': tag, 'n_folds': len(df)}
        for c in df.columns:
            r[c] = df[c].mean()
            r[c + '_std'] = df[c].std(ddof=1) if len(df) > 1 else np.nan
        rows.append(r)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(G / text / 'table_llm.csv', index=False)
        cols = [c for c in ['accuracy', 'macro_f1', 'auc', 'ece', 'aurc', 'acc@cov90', 'acc@cov80'] if c in t]
        out = t[['model', 'n_folds']].copy()
        for c in cols:
            out[c] = [f'{m:.4f} ± {s:.4f}' if np.isfinite(s) else f'{m:.4f}' for m, s in zip(t[c], t[c + '_std'])]
        out['invalid'] = t['invalid']; out['with_prob'] = t['with_prob']
        with open(G / text / 'table_llm.md', 'w', encoding='utf-8') as f:
            f.write(f'Local LLM via Ollama, no fine-tuning, text = {text}. mean ± std over folds; '
                    'P(1) from token log-probabilities.\n\n' + out.to_markdown(index=False))
        print(out.to_string(index=False))


def main():
    for text in TEXT_VARIANTS:
        if (PRED / text).exists():
            evaluate(text)
            llm_table(text)
    orig_ablation()
    save_json(run_info(), G / 'evaluate_run_info.json')


if __name__ == '__main__':
    main()
