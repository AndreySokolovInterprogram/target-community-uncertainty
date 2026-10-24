"""Fine-tuning rubert-tiny2 / multilingual-e5-base on groups (GPU stage), same folds as everything else.

Per fold: inner validation (shared by seeds) -> best epoch by val accuracy; 5 seeds = deep ensemble.
Seed 0 additionally stores MC-dropout predictions (dropout kept on at test time, T passes).
Output files are read by evaluate_groups (preds/<text>/transformers/ft_<name>_f<k>_s<seed>.npz).
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from jise.common import (TEXT_VARIANTS, RES, MODEL_SEEDS, N_FOLDS, SPLIT_SEED, load_groups, group_folds, inner_val_split,
                         run_info, save_json)

CFG = {
    'rubert_tiny2': dict(hf='cointegrated/rubert-tiny2', prefix='', lr=5e-5, epochs=4, bs=32),
    'e5base': dict(hf='intfloat/multilingual-e5-base', prefix='query: ', lr=2e-5, epochs=3, bs=16),
}
MAX_LEN = 256
MC_T = 20


def batches(tok, texts, labels, bs, shuffle, seed):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
    items = [{'input_ids': enc['input_ids'][i], 'attention_mask': enc['attention_mask'][i],
              'labels': int(labels[i])} for i in range(len(texts))]
    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, generator=g,
                      collate_fn=lambda b: tok.pad(b, return_tensors='pt'))


@torch.no_grad()
def predict(model, dl, dev, mc=False):
    model.train(mc)
    out = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items() if k != 'labels'}
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            out.append(torch.softmax(model(**b).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def run(name, text, seeds, folds, dev):
    c = CFG[name]
    g = load_groups(text)
    fold = group_folds(g)
    y = g['label'].values
    texts = np.array([c['prefix'] + t for t in g['text']])
    tok = AutoTokenizer.from_pretrained(c['hf'])
    out = RES / 'groups' / 'preds' / text / 'transformers'
    out.mkdir(parents=True, exist_ok=True)
    for k in folds:
        te = np.where(fold == k)[0]
        trf = np.where(fold != k)[0]
        tr_rel, va_rel = inner_val_split(g.iloc[trf], SPLIT_SEED + k)
        tr, va = trf[tr_rel], trf[va_rel]
        dl_va = batches(tok, texts[va], y[va], 64, False, 0)
        dl_te = batches(tok, texts[te], y[te], 64, False, 0)
        for seed in seeds:
            f = out / f'ft_{name}_f{k}_s{seed}.npz'
            if f.exists():
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            model = AutoModelForSequenceClassification.from_pretrained(c['hf'], num_labels=2).to(dev)
            dl_tr = batches(tok, texts[tr], y[tr], c['bs'], True, seed)
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=0.01)
            steps = len(dl_tr) * c['epochs']
            sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
            scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
            w = torch.tensor(len(tr) / (2 * np.bincount(y[tr])), dtype=torch.float, device=dev)
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)
            best, best_state, best_ep, hist = -1, None, -1, []
            t0 = time.time()
            for ep in range(c['epochs']):
                model.train()
                for b in dl_tr:
                    b = {kk: v.to(dev) for kk, v in b.items()}
                    with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                        logits = model(input_ids=b['input_ids'], attention_mask=b['attention_mask']).logits
                    loss = loss_fn(logits.float(), b['labels'])
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
                acc = float((predict(model, dl_va, dev).argmax(1) == y[va]).mean())
                hist.append(acc)
                if acc > best:
                    best, best_ep = acc, ep
                    best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            res = dict(te_idx=te, va_idx=va, y_te=y[te], y_va=y[va],
                       p_te=predict(model, dl_te, dev), p_va=predict(model, dl_va, dev))
            if seed == 0:   # MC-dropout for seed 0 only
                torch.manual_seed(12345)
                res['p_te_mc'] = np.stack([predict(model, dl_te, dev, mc=True) for _ in range(MC_T)])
            np.savez_compressed(f, **res)
            accte = float((res['p_te'].argmax(1) == y[te]).mean())
            save_json(dict(model=f'ft_{name}', fold=k, seed=seed, best_epoch=best_ep, val_acc=hist,
                           test_acc=accte, sec=time.time() - t0), f.with_suffix('.json'))
            print(f'ft_{name}/{text} fold {k} seed {seed}: acc={accte:.4f} best_ep={best_ep} {time.time() - t0:.0f}s',
                  flush=True)
            del model; torch.cuda.empty_cache()
    save_json(run_info({'model': c['hf'], 'cfg': c, 'max_len': MAX_LEN, 'mc_T': MC_T, 'text': text,
                        'device': dev, 'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None}),
              out / f'run_info_ft_{name}_{int(time.time())}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='rubert_tiny2,e5base')
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    ap.add_argument('--seeds', default=','.join(map(str, MODEL_SEEDS)))
    ap.add_argument('--folds', default=','.join(map(str, range(N_FOLDS))))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    for n in a.models.split(','):
        run(n, a.text, [int(s) for s in a.seeds.split(',')], [int(s) for s in a.folds.split(',')], a.device)
