"""Frozen sentence embeddings of all groups (no fine-tuning, no labels used)."""
import argparse
import time

import numpy as np

from jise.common import TEXT_VARIANTS, PROC, load_groups, run_info, save_json

MODELS = {
    'e5base': ('intfloat/multilingual-e5-base', 'query: '),   # e5 needs a prefix; 'query:' for classification
    'rubert_tiny2': ('cointegrated/rubert-tiny2', ''),
}


def main(names, device, max_len, text):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(0)
    g = load_groups(text)
    for n in names:
        hf, prefix = MODELS[n]
        m = SentenceTransformer(hf, device=device)
        m.max_seq_length = max_len
        t0 = time.time()
        E = m.encode([prefix + t for t in g['text']], batch_size=32, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
        np.save(PROC / f'emb_{n}_{text}.npy', E.astype(np.float32))
        rev = getattr(m[0].auto_model.config, '_commit_hash', None)
        save_json(run_info({'model': hf, 'hf_revision': rev, 'prefix': prefix, 'max_seq_length': max_len,
                            'device': device, 'text': text, 'n': len(g), 'dim': int(E.shape[1]),
                            'sec': time.time() - t0}), PROC / f'emb_{n}_{text}.json')
        print(n, E.shape, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='rubert_tiny2,e5base')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    a = ap.parse_args()
    main(a.models.split(','), a.device, a.max_len, a.text)
