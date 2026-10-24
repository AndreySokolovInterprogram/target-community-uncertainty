"""Publication figures (English labels, PNG 300 dpi + PDF) for the query-masked group task.

fig_risk_coverage   - accuracy on retained groups vs coverage (abstain on least confident), pooled OOF
fig_active_learning - accuracy vs number of labelled groups, random vs uncertainty sampling
fig_reliability     - reliability diagram before / after temperature scaling
Data behind each figure is saved next to it as CSV. Palette: validated categorical palette (dataviz skill),
secondary encoding by line style / marker because three hues are below 3:1 contrast on white.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jise.common import RES, SPLIT_SEED, load_groups, group_folds, inner_val_split
from jise.label_check import oof_p1, PRED
from jise.metrics import fit_temperature, apply_temperature, risk_coverage, reliability_bins

OUT = RES / 'figures'
INK, MUTED, GRID = '#2b2b2b', '#6b6b6b', '#e6e6e6'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
SERIES = [  # label, (sub, key, seeds), color, linestyle, marker
    ('e5-base, fine-tuned (ensemble of 5)', ('transformers', 'ft_e5base', 5), PAL[0], '-', 'o'),
    ('e5-base, frozen + LR', ('classic', 'e5base_lr', 1), PAL[1], '--', 's'),
    ('TF-IDF + LR', ('classic', 'tfidf_lr', 1), PAL[2], '-.', '^'),
    ('CNN (ensemble of 5)', ('keras', 'cnn', 5), PAL[3], ':', 'D'),
    ('Qwen2.5-7B, 4-shot', ('llm', 'qwen2.5-7b-instruct_4shot', 0), PAL[4], (0, (5, 1.5)), 'v'),
    ('Qwen3-8B, 4-shot', ('llm', 'qwen3-8b_4shot', 0), PAL[5], (0, (1.5, 1.5)), 'P'),
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.6, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=300)
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def calibrated(p1, sub, key, n_seeds, g, y, fold):
    """Per-fold temperature on inner validation; LLM: OOF P(1) of validation rows (see stats_tests)."""
    out = np.empty_like(p1)
    L = np.stack([np.zeros_like(p1), np.log(np.clip(p1, 1e-12, 1)) - np.log(np.clip(1 - p1, 1e-12, 1))], 1)
    L = np.clip(L, -30, 30)
    for k in range(5):
        te = np.where(fold == k)[0]
        if sub == 'llm':
            trf = np.where(fold != k)[0]
            _, va_rel = inner_val_split(g.iloc[trf], SPLIT_SEED + k)
            va = trf[va_rel]
            T = fit_temperature(L[va], y[va])
        else:
            zs = [np.load(PRED / sub / f'{key}_f{k}_s{s}.npz') for s in range(n_seeds)]
            pv = np.mean([z['p_va'][:, 1] for z in zs], 0)
            Lv = np.stack([np.zeros_like(pv), np.log(np.clip(pv, 1e-12, 1)) - np.log(np.clip(1 - pv, 1e-12, 1))], 1)
            T = fit_temperature(np.clip(Lv, -30, 30), zs[0]['y_va'])
        out[te] = apply_temperature(L[te], T)[:, 1]
    return out


def main():
    g = load_groups('title_desc_masked')
    y = g['label'].values
    fold = group_folds(g)
    probs = {lab: oof_p1(*spec[:2], spec[2], len(g)) for lab, spec, *_ in SERIES}

    # 1. risk-coverage
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    rows = []
    for lab, spec, col, ls, mk in SERIES:
        p1 = probs[lab]
        P = np.stack([1 - p1, p1], 1)
        cov, acc, aurc = risk_coverage(P, y)
        ax.plot(cov, acc, color=col, ls=ls, lw=1.5, label=f'{lab} (AURC {aurc:.4f}, pooled)')
        for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
            rows.append({'model': lab, 'coverage': c, 'accuracy': float(acc[int(np.ceil(c * len(acc))) - 1]),
                         'aurc': aurc})
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.92, 1.0)
    ax.set_xlabel('Coverage (share of groups classified automatically)')
    ax.set_ylabel('Accuracy on retained groups')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2), frameon=False, ncol=1, handlelength=2.6)
    save(fig, 'fig_risk_coverage')
    pd.DataFrame(rows).to_csv(OUT / 'fig_risk_coverage.csv', index=False)

    # 2. active learning
    s = pd.read_csv(RES / 'groups' / 'active_learning' / 'al_summary.csv')
    full = pd.read_csv(RES / 'groups' / 'active_learning' / 'al_full_pool.csv')
    s = s[s['text'] == 'title_desc_masked']
    feats = [('e5base', 'e5-base embeddings + LR', PAL[0], 'o'), ('rubert_tiny2', 'rubert-tiny2 embeddings + LR',
                                                                 PAL[1], 's'),
             ('tfidf', 'TF-IDF + LR', PAL[2], '^')]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for f, lab, col, mk in feats:
        for strat, ls, mfc, name in [('uncertainty', '-', col, 'uncertainty'), ('random', '--', 'white', 'random')]:
            t = s[(s['feat'] == f) & (s['strategy'] == strat)].sort_values('n_labeled')
            ax.plot(t['n_labeled'], t['acc'], color=col, ls=ls, marker=mk, ms=4, mfc=mfc, mew=1.1,
                    label=f'{lab}, {name}')
            ax.fill_between(t['n_labeled'], t['acc'] - t['acc_std'], t['acc'] + t['acc_std'], color=col,
                            alpha=0.10 if strat == 'uncertainty' else 0.05, lw=0)
        fa = full[(full['text'] == 'title_desc_masked') & (full['feat'] == f)]['full_acc'].iloc[0]
        ax.axhline(fa, color=col, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel('Number of labelled groups'); ax.set_ylabel('Test accuracy')
    ax.set_xticks(range(10, 101, 10)); ax.set_ylim(0.5, 1.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2), frameon=False, ncol=1, handlelength=2.6)
    ax.text(101, 0.985, 'dotted: all labels', ha='right', va='bottom', fontsize=6.5, color=MUTED)
    save(fig, 'fig_active_learning')
    s.to_csv(OUT / 'fig_active_learning.csv', index=False)

    # 3. reliability, before / after temperature scaling
    pick = [SERIES[0], SERIES[1], SERIES[3], SERIES[5]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    rows = []
    for ax, title, cal in [(axes[0], 'Before calibration', False), (axes[1], 'After temperature scaling', True)]:
        ax.plot([0.5, 1], [0.5, 1], color=MUTED, lw=0.8, ls='--')
        for lab, spec, col, ls, mk in pick:
            p1 = probs[lab]
            if cal:
                p1 = calibrated(p1, spec[0], spec[1], spec[2], g, y, fold)
            P = np.stack([1 - p1, p1], 1)
            b = [r for r in reliability_bins(P, y, 10) if r[2] >= 20]
            from jise.metrics import ece
            e = ece(P, y)
            ax.plot([r[4] for r in b], [r[3] for r in b], color=col, ls=ls, marker=mk, ms=4, label=f'{lab} (ECE {e:.3f})')
            rows += [{'model': lab, 'calibrated': cal, 'bin_lo': r[0], 'bin_hi': r[1], 'n': r[2], 'accuracy': r[3],
                      'confidence': r[4], 'ece_pooled': e} for r in b]
        ax.set_title(title); ax.set_xlim(0.5, 1.0); ax.set_ylim(0.3, 1.0)
        ax.set_xlabel('Mean predicted confidence')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=1, handlelength=2.6)
    axes[0].set_ylabel('Observed accuracy')
    save(fig, 'fig_reliability')
    pd.DataFrame(rows).to_csv(OUT / 'fig_reliability.csv', index=False)
    print('saved', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
