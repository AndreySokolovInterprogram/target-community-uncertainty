"""Shared helpers: paths, data loading, leakage-safe splits, run metadata."""
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results' / 'jise'

SPLIT_SEED = 2026          # fixed seed of the outer CV folds
N_FOLDS = 5
MODEL_SEEDS = [0, 1, 2, 3, 4]  # ensemble members / repeats


def norm_text(s):
    """Lower-case, mask digits, drop punctuation/emoji, squeeze spaces. Used only to find near-duplicates."""
    s = str(s).lower().replace('\r', ' ')
    s = re.sub(r'\d+', '#', s)
    s = re.sub(r'[^\w#]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


TEXT_VARIANTS = ('title_desc', 'desc', 'title_desc_masked', 'desc_masked')


def load_groups(text='title_desc'):
    """All groups (11,613 rows) minus the single duplicated group ID.

    text='title_desc': text = title + newline + description (missing parts -> empty string), 11,612 rows.
    text='desc': description only (as in the WorldCIST notebook), rows with an empty description dropped.
    dup_key = normalized text; rows with the same key always fall into the same fold.
    """
    assert text in TEXT_VARIANTS
    g = pd.read_excel(RAW / 'vk_groups_dataset.xlsx')
    g = g.rename(columns={'ID_группы': 'group_id', 'Название': 'title', 'ID_города': 'city_id',
                          'Город': 'city', 'Описание': 'description', 'Признак': 'label'})
    n0 = len(g)
    g = g.drop_duplicates('group_id', keep='first').reset_index(drop=True)
    g['title'] = g['title'].fillna('').astype(str)
    g['description'] = g['description'].fillna('').astype(str)
    if text.startswith('title_desc'):
        g['text'] = (g['title'] + '\n' + g['description']).str.strip()
    else:
        g = g[g['description'].str.strip() != ''].reset_index(drop=True)
        g['text'] = g['description'].str.strip()
    if text.endswith('_masked'):
        from jise.query_terms import mask_text
        g['text'] = g['text'].map(mask_text).str.strip()
    g['label'] = g['label'].astype(int)
    g['dup_key'] = g['text'].map(norm_text)
    g['dup_group'] = pd.factorize(g['dup_key'])[0]
    empty = g['dup_key'] == ''
    if text.endswith('_masked') and empty.any():
        # texts emptied by masking carry no information: no need to keep them in one fold
        # (only for masked variants, so the folds of title_desc / desc stay exactly as computed)
        g.loc[empty, 'dup_group'] = g['dup_group'].max() + 1 + np.arange(int(empty.sum()))
    g.attrs['n_raw'] = n0
    g.attrs['text'] = text
    return g


def group_folds(df, y_col='label', group_col='dup_group', n_splits=N_FOLDS, seed=SPLIT_SEED):
    """Outer CV fold id per row: stratified by label, duplicates kept together."""
    fold = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(df, df[y_col], df[group_col])):
        fold[te] = k
    return fold


def inner_val_split(df_tr, seed, frac=0.1, y_col='label', group_col='dup_group'):
    """Validation part inside a training fold (for early stopping / checkpoint / temperature)."""
    n = int(round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(df_tr, df_tr[y_col], df_tr[group_col]))
    return tr, va


def run_info(extra=None):
    """Software/hardware/time metadata stored next to every result file."""
    info = {
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    try:
        info['cpu'] = subprocess.run(['bash', '-c', "lscpu | grep 'Model name' | cut -d: -f2"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    from importlib.metadata import version   # read versions without importing (TF + torch in one process crash)
    for mod in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'tensorflow-cpu', 'keras', 'torch', 'transformers',
                'sentence-transformers']:
        try:
            info[mod] = version(mod)
        except Exception:
            pass
    try:
        info['git_commit'] = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    if extra:
        info.update(extra)
    return info


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
