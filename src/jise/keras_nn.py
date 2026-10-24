"""CNN / LSTM / GRU for group classification (module 1), configs from the WorldCIST notebook.

Two settings:
  main  - leakage-safe: all groups, title+description, 5-fold StratifiedGroupKFold (duplicates in one fold),
          inner validation inside the training fold, class weights, 5 seeds per fold.
  orig  - the notebook protocol, for the bug ablation: dropna() rows, description only, random 70/30 split,
          validation = last 10 % of train (Keras validation_split), one split per seed.

Fixes vs. the notebook (documented in the report):
  * CNN Embedding(input_dim=100) while token ids go up to ~30k. On GPU tf.gather returns zeros for
    out-of-range ids, so the CNN saw only the ~99 most frequent words. 'cnn_bug' emulates exactly that
    (ids >= 100 -> zero vector); 'cnn' uses input_dim=num_words.
  * Each model is evaluated with its own best checkpoint (posts notebooks loaded CNN weights before
    evaluating LSTM, i.e. LSTM was scored at the last epoch). We store both best-checkpoint and
    last-epoch predictions to measure that effect.
  * binary_crossentropy on a 2-unit softmax equals categorical cross-entropy; we use the latter.
"""
import argparse
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

from jise.common import (TEXT_VARIANTS, RAW, RES, MODEL_SEEDS, N_FOLDS, SPLIT_SEED, load_groups, group_folds,
                         inner_val_split, run_info, save_json)

NUM_WORDS = 100000
EPOCHS = {'cnn': 15, 'cnn_bug': 15, 'lstm': 5, 'gru': 5}   # as in the notebook
BATCH = 128


class GPUOutOfRangeEmbedding(layers.Layer):
    """Embedding(input_dim) that returns a zero vector for ids >= input_dim (tf.gather on GPU)."""

    def __init__(self, input_dim, output_dim, **kw):
        super().__init__(**kw)
        self.input_dim = input_dim
        self.emb = layers.Embedding(input_dim, output_dim)

    def call(self, x):
        mask = tf.cast(x < self.input_dim, self.compute_dtype)
        return self.emb(tf.minimum(x, self.input_dim - 1)) * mask[..., None]


def build(name, maxlen):
    inp = keras.Input(shape=(maxlen,), dtype='int32')
    if name == 'cnn':
        x = layers.Embedding(NUM_WORDS, 32)(inp)
    elif name == 'cnn_bug':
        x = GPUOutOfRangeEmbedding(100, 32)(inp)
    else:
        x = layers.Embedding(NUM_WORDS, 32)(inp)
    if name.startswith('cnn'):
        x = layers.Conv1D(350, 8, padding='valid', activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
    elif name == 'lstm':
        x = layers.LSTM(16)(x)
    elif name == 'gru':
        x = layers.GRU(16)(x)
    out = layers.Dense(2, activation='softmax')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


class BestAndLast(keras.callbacks.Callback):
    """ModelCheckpoint(monitor='val_accuracy', save_best_only=True) kept in memory."""

    def __init__(self):
        super().__init__()
        self.best, self.best_w, self.best_epoch = -1, None, -1

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] > self.best:
            self.best, self.best_epoch = logs['val_accuracy'], epoch
            self.best_w = self.model.get_weights()


def encode(train_texts, other_texts):
    tok = Tokenizer(num_words=NUM_WORDS)
    tok.fit_on_texts(train_texts)
    s_tr = tok.texts_to_sequences(train_texts)
    maxlen = max(1, max(len(s) for s in s_tr))
    x_tr = pad_sequences(s_tr, maxlen=maxlen)
    xs = [pad_sequences(tok.texts_to_sequences(t), maxlen=maxlen) for t in other_texts]
    return x_tr, xs, maxlen, len(tok.word_index)


def fit_predict(name, seed, x_tr, y_tr, x_va, y_va, x_te, class_weight, val_split=None):
    keras.utils.set_random_seed(seed)
    model = build(name, x_tr.shape[1])
    cb = BestAndLast()
    t0 = time.time()
    kw = dict(epochs=EPOCHS[name], batch_size=BATCH, verbose=0, callbacks=[cb], class_weight=class_weight)
    if val_split is not None:          # notebook: last 10 % of train, no shuffle
        hist = model.fit(x_tr, keras.utils.to_categorical(y_tr, 2), validation_split=val_split, **kw)
    else:
        hist = model.fit(x_tr, keras.utils.to_categorical(y_tr, 2),
                         validation_data=(x_va, keras.utils.to_categorical(y_va, 2)), **kw)
    t_fit = time.time() - t0
    p_last_te = model.predict(x_te, batch_size=512, verbose=0)
    p_last_va = model.predict(x_va, batch_size=512, verbose=0) if x_va is not None else None
    model.set_weights(cb.best_w)
    p_te = model.predict(x_te, batch_size=512, verbose=0)
    p_va = model.predict(x_va, batch_size=512, verbose=0) if x_va is not None else None
    return dict(p_te=p_te, p_last_te=p_last_te, p_va=p_va, p_last_va=p_last_va,
                best_epoch=cb.best_epoch, best_val_acc=cb.best, fit_sec=t_fit,
                history={k: [float(v) for v in vs] for k, vs in hist.history.items()})


def run_main(models, seeds, folds, text):
    g = load_groups(text)
    fold = group_folds(g)
    out = RES / 'groups' / 'preds' / text / 'keras'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'group_id': g['group_id'], 'label': g['label'], 'dup_group': g['dup_group'],
                  'fold': fold}).to_csv(RES / 'groups' / f'folds_{text}.csv', index=False)
    log = []
    for k in folds:
        te_idx = np.where(fold == k)[0]
        trf_idx = np.where(fold != k)[0]
        # one inner validation per fold, shared by all seeds (ensemble is calibrated on it)
        tr_rel, va_rel = inner_val_split(g.iloc[trf_idx], SPLIT_SEED + k)
        tr_idx, va_idx = trf_idx[tr_rel], trf_idx[va_rel]
        x_tr, (x_va, x_te), maxlen, vocab = encode(g['text'].values[tr_idx],
                                                   [g['text'].values[va_idx], g['text'].values[te_idx]])
        y_tr, y_va, y_te = (g['label'].values[i] for i in (tr_idx, va_idx, te_idx))
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
        for seed in seeds:
            for name in models:
                f = out / f'{name}_f{k}_s{seed}.npz'
                if f.exists():
                    continue
                r = fit_predict(name, seed, x_tr, y_tr, x_va, y_va, x_te, {0: cw[0], 1: cw[1]})
                np.savez_compressed(f, te_idx=te_idx, va_idx=va_idx, y_te=y_te, y_va=y_va,
                                    p_te=r['p_te'], p_last_te=r['p_last_te'], p_va=r['p_va'],
                                    p_last_va=r['p_last_va'])
                acc = float((r['p_te'].argmax(1) == y_te).mean())
                row = dict(model=name, fold=k, seed=seed, n_tr=len(tr_idx), n_va=len(va_idx),
                           n_te=len(te_idx), maxlen=maxlen, vocab=vocab, best_epoch=r['best_epoch'],
                           best_val_acc=r['best_val_acc'], test_acc=acc, fit_sec=r['fit_sec'],
                           history=r['history'])
                log.append(row)
                save_json(row, f.with_suffix('.json'))
                print(f'{name:8s} fold {k} seed {seed}: acc={acc:.4f} best_ep={r["best_epoch"]} '
                      f'maxlen={maxlen} {r["fit_sec"]:.0f}s', flush=True)
    save_json(run_info({'setting': 'main', 'text': text, 'models': models, 'seeds': seeds, 'folds': folds}),
              out / f'run_info_{int(time.time())}.json')


def run_orig(models, seeds):
    """Notebook protocol (for the Embedding bug ablation)."""
    raw = pd.read_excel(RAW / 'vk_groups_dataset.xlsx').dropna().reset_index(drop=True)
    texts, y = raw['Описание'].astype(str).values, raw['Признак'].astype(int).values
    out = RES / 'groups' / 'preds' / 'keras_orig'
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        idx = np.random.default_rng(seed).permutation(len(raw))
        n_te = int(len(raw) * 0.3)
        tr_idx, te_idx = idx[:len(raw) - n_te], idx[len(raw) - n_te:]
        x_tr, (x_te,), maxlen, vocab = encode(texts[tr_idx], [texts[te_idx]])
        for name in models:
            f = out / f'{name}_s{seed}.npz'
            if f.exists():
                continue
            r = fit_predict(name, seed, x_tr, y[tr_idx], None, None, x_te, None, val_split=0.1)
            np.savez_compressed(f, te_idx=te_idx, y_te=y[te_idx], p_te=r['p_te'], p_last_te=r['p_last_te'])
            acc = float((r['p_te'].argmax(1) == y[te_idx]).mean())
            acc_last = float((r['p_last_te'].argmax(1) == y[te_idx]).mean())
            row = dict(model=name, seed=seed, n_tr=len(tr_idx), n_te=len(te_idx), maxlen=maxlen, vocab=vocab,
                       best_epoch=r['best_epoch'], best_val_acc=r['best_val_acc'], test_acc=acc,
                       test_acc_last_epoch=acc_last, fit_sec=r['fit_sec'], history=r['history'])
            save_json(row, f.with_suffix('.json'))
            print(f'[orig] {name:8s} seed {seed}: acc={acc:.4f} last={acc_last:.4f} best_ep={r["best_epoch"]} '
                  f'{r["fit_sec"]:.0f}s', flush=True)
    save_json(run_info({'setting': 'orig', 'models': models, 'seeds': seeds}),
              out / f'run_info_{int(time.time())}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=['main', 'orig'], default='main')
    ap.add_argument('--models', default='cnn,lstm,gru')
    ap.add_argument('--seeds', default=','.join(map(str, MODEL_SEEDS)))
    ap.add_argument('--folds', default=','.join(map(str, range(N_FOLDS))))
    ap.add_argument('--text', default='title_desc', choices=TEXT_VARIANTS)
    a = ap.parse_args()
    tf.config.experimental.enable_op_determinism()
    models = a.models.split(',')
    seeds = [int(s) for s in a.seeds.split(',')]
    if a.setting == 'main':
        run_main(models, seeds, [int(s) for s in a.folds.split(',')], a.text)
    else:
        run_orig(models, seeds)
