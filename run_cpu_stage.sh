#!/usr/bin/env bash
# CPU stage of JISE experiments (groups). GPU is hidden on purpose.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
PY=${PY:-python3}
step() { echo "=== $(date '+%F %T') $*"; }
step tfidf desc;             $PY -m jise.tfidf_lr --feats tfidf --text desc
step keras orig ablation;    $PY -m jise.keras_nn --setting orig --models cnn_bug,cnn,lstm,gru
for t in title_desc desc; do
  step embeddings $t;        $PY -m jise.embed --models rubert_tiny2,e5base --device cpu --text $t
  step emb+LR $t;            $PY -m jise.tfidf_lr --feats rubert_tiny2,e5base --text $t
  step active learning $t;   $PY -m jise.active_learning --feats tfidf,rubert_tiny2,e5base --text $t
done
step keras main title_desc;  $PY -m jise.keras_nn --setting main --models cnn,lstm,gru --text title_desc
step keras main desc;        $PY -m jise.keras_nn --setting main --models cnn,lstm,gru --text desc
step evaluate;               $PY -m jise.evaluate_groups
step done
