"""Summary table + learning curves for active learning (results/jise/groups/active_learning/)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from jise.common import RES

D = RES / 'groups' / 'active_learning'
FEAT = {'tfidf': 'TF-IDF', 'rubert_tiny2': 'rubert-tiny2 emb.', 'e5base': 'e5-base emb.'}


def main():
    raw = pd.concat([pd.read_csv(f) for f in sorted(D.glob('al_raw_*.csv'))], ignore_index=True)
    al = raw[raw['strategy'] != 'full_pool']
    s = (al.groupby(['text', 'feat', 'strategy', 'n_labeled'])
           .agg(acc=('accuracy', 'mean'), acc_std=('accuracy', 'std'), f1=('macro_f1', 'mean'),
                f1_std=('macro_f1', 'std'), auc=('auc', 'mean'), auc_std=('auc', 'std'), runs=('accuracy', 'size'))
           .reset_index())
    full = (raw[raw['strategy'] == 'full_pool'].groupby(['text', 'feat'])
              .agg(full_acc=('accuracy', 'mean'), full_n=('n_labeled', 'mean')).reset_index())
    s.to_csv(D / 'al_summary.csv', index=False)
    full.to_csv(D / 'al_full_pool.csv', index=False)
    # paired difference uncertainty - random at each budget (same fold, repeat, start set)
    piv = al.pivot_table(index=['text', 'feat', 'fold', 'repeat', 'n_labeled'], columns='strategy',
                         values='accuracy').reset_index()
    piv['diff'] = piv['uncertainty'] - piv['random']
    d = (piv.groupby(['text', 'feat', 'n_labeled'])['diff']
            .agg(['mean', 'std', lambda x: (x > 0).mean()]).reset_index())
    d.columns = ['text', 'feat', 'n_labeled', 'diff_mean', 'diff_std', 'share_runs_uncertainty_better']
    d.to_csv(D / 'al_paired_diff.csv', index=False)
    with open(D / 'al_summary.md', 'w', encoding='utf-8') as f:
        for text in s['text'].unique():
            t = s[s['text'] == text].copy()
            t['acc'] = [f'{m:.3f} ± {v:.3f}' for m, v in zip(t['acc'], t['acc_std'])]
            w = t.pivot_table(index='n_labeled', columns=['feat', 'strategy'], values='acc', aggfunc='first')
            f.write(f'## text = {text}: accuracy, mean ± std over 5 folds x 5 repeats\n\n')
            f.write(w.to_markdown() + '\n\n')
            fp = full[full['text'] == text]
            f.write('Full pool (all ~9.3k labels): ' + ', '.join(
                f'{FEAT.get(a, a)} {b:.4f}' for a, b in zip(fp['feat'], fp['full_acc'])) + '\n\n')
            dd = d[d['text'] == text]
            f.write('Paired difference uncertainty − random (accuracy):\n\n')
            f.write(dd.pivot_table(index='n_labeled', columns='feat', values='diff_mean').to_markdown(floatfmt='+.4f'))
            f.write('\n\n')
    texts = list(s['text'].unique())
    fig, axes = plt.subplots(1, len(texts), figsize=(5.5 * len(texts), 4), squeeze=False)
    for ax, text in zip(axes[0], texts):
        for feat in s['feat'].unique():
            for strat, ls in [('random', '--'), ('uncertainty', '-')]:
                t = s[(s['text'] == text) & (s['feat'] == feat) & (s['strategy'] == strat)]
                if len(t):
                    ax.errorbar(t['n_labeled'], t['acc'], yerr=t['acc_std'], ls=ls, marker='o', ms=3, capsize=2,
                                label=f'{FEAT.get(feat, feat)}, {strat}')
        ax.set_xlabel('labelled groups'); ax.set_ylabel('test accuracy'); ax.grid(alpha=.3)
        ax.set_title(f'Active learning, groups ({text})'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(D / 'fig_active_learning.png', dpi=200); plt.close(fig)
    print(open(D / 'al_summary.md', encoding='utf-8').read())


if __name__ == '__main__':
    main()
