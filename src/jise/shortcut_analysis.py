"""Does the label coincide with the search-query words? (results/jise/groups/shortcut/)

Labels were produced by groups.search: 1 = found by a food query, 0 = found by a non-food query
('Get data.ipynb'). Here: coverage of query words, a no-learning keyword rule, errors of the masked models
on texts that became empty after masking, and a heuristic count of possibly mislabelled 'delivery' groups.
"""
import numpy as np
import pandas as pd

from jise.common import RES, load_groups, group_folds, save_json, run_info
from jise.query_terms import POS_STEMS, NEG_STEMS, NEG_WORDS, has_terms, POS_QUERIES, NEG_QUERIES

OUT = RES / 'groups' / 'shortcut'
POS_W = ['еда', 'еды', 'еду', 'едой', 'еде']   # 'дом'/'на' not counted as evidence of the query


def coverage():
    g = load_groups('title_desc')
    rows = []
    for col in ['title', 'description', 'text']:
        pos = g[col].map(lambda s: has_terms(s, POS_STEMS, POS_W))
        neg = g[col].map(lambda s: has_terms(s, NEG_STEMS, NEG_WORDS))
        pred = pos.astype(int)              # rule: food query word present -> 1, otherwise 0
        rows.append({'field': col,
                     'label1_with_food_query_word': pos[g.label == 1].mean(),
                     'label0_with_nonfood_query_word': neg[g.label == 0].mean(),
                     'label1_with_nonfood_word_only': (neg & ~pos)[g.label == 1].mean(),
                     'label0_with_food_word_only': (pos & ~neg)[g.label == 0].mean(),
                     'rule_accuracy': (pred == g.label).mean()})
    return pd.DataFrame(rows)


def masked_errors():
    rows = []
    for text, models in [('title_desc_masked', ['classic/e5base_lr', 'classic/tfidf_lr', 'keras/cnn']),
                         ('desc_masked', ['classic/e5base_lr', 'classic/tfidf_lr'])]:
        g = load_groups(text)
        empty = (g['dup_key'] == '').values
        for m in models:
            sub, name = m.split('/')
            P, Y, I = [], [], []
            for k in range(5):
                if sub == 'keras':
                    p = np.mean([np.load(RES / 'groups' / 'preds' / text / sub / f'{name}_f{k}_s{s}.npz')['p_te']
                                 for s in range(5)], 0)
                    z = np.load(RES / 'groups' / 'preds' / text / sub / f'{name}_f{k}_s0.npz')
                else:
                    z = np.load(RES / 'groups' / 'preds' / text / sub / f'{name}_f{k}_s0.npz')
                    p = z['p_te']
                P.append(p); Y.append(z['y_te']); I.append(z['te_idx'])
            P, Y, I = np.concatenate(P), np.concatenate(Y), np.concatenate(I)
            err = P.argmax(1) != Y
            e = empty[I]
            rows.append({'text': text, 'model': name + (' ens5' if sub == 'keras' else ''), 'n': len(Y),
                         'n_empty_after_mask': int(e.sum()), 'acc_all': 1 - err.mean(),
                         'acc_nonempty': 1 - err[~e].mean(), 'acc_empty': 1 - err[e].mean() if e.any() else np.nan,
                         'share_of_errors_on_empty': err[e].sum() / max(1, err.sum())})
    return pd.DataFrame(rows)


def label_noise():
    g = load_groups('title_desc')
    t = g.loc[g.label == 1, 'text'].str.lower()
    recipe = t.str.contains('рецепт|мукбанг|кулинарн|готовим дома')
    order = t.str.contains('достав|заказ|самовывоз|меню')
    return {'label1': int(len(t)), 'label1_without_delivery_or_order_words': int((~order).sum()),
            'label1_recipe_like_without_order_words': int((recipe & ~order).sum()),
            'note': 'heuristic upper/lower bounds, not manually verified'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c, m, n = coverage(), masked_errors(), label_noise()
    c.to_csv(OUT / 'query_word_coverage.csv', index=False)
    m.to_csv(OUT / 'masked_errors.csv', index=False)
    with open(OUT / 'shortcut.md', 'w', encoding='utf-8') as f:
        f.write('Food queries (label 1): ' + ', '.join(POS_QUERIES[:7]) + '; non-food (label 0): '
                + ', '.join(NEG_QUERIES[:6]) + ' (later items of the lists were never reached).\n\n')
        f.write(c.to_markdown(index=False, floatfmt='.4f') + '\n\n')
        f.write('Masked variants: where the errors are\n\n' + m.to_markdown(index=False, floatfmt='.4f') + '\n\n')
        f.write('Possible label noise among label 1: ' + str(n) + '\n')
    save_json(run_info({'label_noise': n}), OUT / 'info.json')
    print(open(OUT / 'shortcut.md', encoding='utf-8').read())


if __name__ == '__main__':
    main()
