"""Trial of the Ollama classifier on 20 groups (fold-0 test, 10 + 10, seed 7) before the full LLM run."""
import json
import time

import numpy as np

from jise.common import RES, load_groups, group_folds, run_info, save_json
from jise.llm_ollama import SYSTEM, ask, user_msg

g = load_groups('title_desc_masked')
fold = group_folds(g)
y = g['label'].values
te, tr = np.where(fold == 0)[0], np.where(fold != 0)[0]
rng = np.random.default_rng(7)
sample = np.concatenate([rng.choice(te[y[te] == c], 10, replace=False) for c in (0, 1)])
out = RES / 'groups' / 'llm_trial'
out.mkdir(parents=True, exist_ok=True)
rows = []
for model in ['qwen2.5:7b-instruct', 'qwen3:8b']:
    for shots in (0, 4):
        r2 = np.random.default_rng(100)            # same rule as llm_ollama for fold 0
        ex = []
        for c in (0, 1):
            ex += list(r2.choice(tr[y[tr] == c], shots, replace=False)) if shots else []
        r2.shuffle(ex)
        msgs = [{'role': 'system', 'content': SYSTEM}]
        for i in ex:
            msgs += [{'role': 'user', 'content': user_msg(g['text'][i])}, {'role': 'assistant', 'content': str(y[i])}]
        for i in sample:
            t0 = time.time()
            a, p = ask(model, msgs + [{'role': 'user', 'content': user_msg(g['text'][i])}])
            rows.append(dict(model=model, shots=shots, idx=int(i), y=int(y[i]), answer=a, p1=p,
                             sec=round(time.time() - t0, 3)))
    import urllib.request
    urllib.request.urlopen(urllib.request.Request('http://localhost:11434/api/generate',
                           data=json.dumps({'model': model, 'keep_alive': 0}).encode()), timeout=60).read()
save_json(rows, out / 'trial_rows.json')
summ = {}
for model in ['qwen2.5:7b-instruct', 'qwen3:8b']:
    for shots in (0, 4):
        rs = [r for r in rows if r['model'] == model and r['shots'] == shots]
        valid = [r for r in rs if r['answer'] in ('0', '1')]
        summ[f'{model} {shots}-shot'] = dict(
            n=len(rs), parsed=len(valid), with_prob=sum(r['p1'] is not None for r in rs),
            accuracy=round(sum(int(r['answer'] == str(r['y'])) for r in rs) / len(rs), 3),
            answers=sorted({r['answer'] for r in rs}), median_sec=float(np.median([r['sec'] for r in rs])))
save_json(run_info({'trial': summ}), out / 'trial_summary.json')
for k, v in summ.items():
    print(k, v)
