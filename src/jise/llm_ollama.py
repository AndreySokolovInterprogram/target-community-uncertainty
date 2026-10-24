"""Local LLM via Ollama, no fine-tuning: zero-shot and few-shot classification of groups (GPU stage).

Every group is scored once, inside its outer test fold; few-shot examples (k per class) are drawn from the
training part of that fold only. Deterministic decoding (temperature 0, fixed seed), one output token.
P(class 1) is taken from the token log-probabilities of '0'/'1' when the server returns them; otherwise the
hard answer is stored and probability-based metrics (AUC, ECE, risk-coverage) are not reported for the LLM.
"""
import argparse
import json
import math
import time
import urllib.request

import numpy as np

from jise.common import TEXT_VARIANTS, RES, N_FOLDS, load_groups, group_folds, run_info, save_json

URL = 'http://localhost:11434/api/chat'
SYSTEM = ('Ты классифицируешь сообщества ВКонтакте. Ответь одной цифрой: 1 — если сообщество является службой '
          'доставки готовой еды (суши, пицца, обеды, кафе или ресторан с доставкой и т.п.), 0 — если нет. '
          'Без пояснений.')


def ask(model, messages, seed=0):
    body = {'model': model, 'messages': messages, 'stream': False, 'think': False, 'logprobs': True,
            'top_logprobs': 10, 'options': {'temperature': 0, 'seed': seed, 'num_predict': 1}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    ans = d['message']['content'].strip()
    p1 = None
    lp = d.get('logprobs')
    if lp:
        top = {t['token'].strip(): t['logprob'] for t in lp[0].get('top_logprobs', [])}
        if '0' in top or '1' in top:
            a, b = math.exp(top.get('0', -50)), math.exp(top.get('1', -50))
            p1 = b / (a + b)
    return ans, p1


def user_msg(t):
    return f'Сообщество:\n{t[:1500]}\n\nОтвет (0 или 1):'


def run(model, text, shots, limit=None):
    g = load_groups(text)
    fold = group_folds(g)
    y = g['label'].values
    out = RES / 'groups' / 'preds' / text / 'llm'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{model.replace(":", "-")}_{shots}shot'
    t0 = time.time()
    for k in range(N_FOLDS):
        f = out / f'{tag}_f{k}.npz'
        if f.exists():
            continue
        te = np.where(fold == k)[0][:limit]
        tr = np.where(fold != k)[0]
        rng = np.random.default_rng(100 + k)
        ex = []
        for c in (0, 1):
            ex += list(rng.choice(tr[y[tr] == c], shots, replace=False)) if shots else []
        rng.shuffle(ex)
        msgs = [{'role': 'system', 'content': SYSTEM}]
        for i in ex:
            msgs += [{'role': 'user', 'content': user_msg(g['text'][i])},
                     {'role': 'assistant', 'content': str(y[i])}]
        ans, p1 = [], []
        for i in te:
            a, p = ask(model, msgs + [{'role': 'user', 'content': user_msg(g['text'][i])}])
            ans.append(a); p1.append(np.nan if p is None else p)
        hard = np.array([1 if a.startswith('1') else 0 if a.startswith('0') else -1 for a in ans])
        np.savez_compressed(f, te_idx=te, y_te=y[te], hard=hard, p1=np.array(p1), shots_idx=np.array(ex))
        acc = float((hard == y[te]).mean())
        print(f'{tag}/{text} fold {k}: acc={acc:.4f} invalid={int((hard < 0).sum())} '
              f'with_prob={int(np.isfinite(p1).sum())}/{len(te)} {time.time() - t0:.0f}s', flush=True)
    save_json(run_info({'ollama_model': model, 'shots_per_class': shots, 'text': text, 'system': SYSTEM}),
              out / f'run_info_{tag}_{int(time.time())}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='qwen2.5:7b-instruct,qwen3:8b')
    ap.add_argument('--shots', default='0,4')
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    for m in a.models.split(','):
        for s in a.shots.split(','):
            run(m, a.text, int(s), a.limit)
