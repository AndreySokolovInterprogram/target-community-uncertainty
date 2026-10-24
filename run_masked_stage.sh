#!/usr/bin/env bash
# CPU stage, query-masked variants (search-query words removed). Runs after run_cpu_stage.sh. GPU hidden.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
PY=${PY:-python3}
step() { echo "=== $(date '+%F %T') $*"; }
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
step start masked
for t in title_desc_masked desc_masked; do
  step embeddings $t;        $PY -m jise.embed --models rubert_tiny2,e5base --device cpu --text $t
  step emb+LR $t;            $PY -m jise.tfidf_lr --feats rubert_tiny2,e5base --text $t
  step active learning $t;   $PY -m jise.active_learning --feats tfidf,rubert_tiny2,e5base --text $t
done
step keras main title_desc_masked; $PY -m jise.keras_nn --setting main --models cnn,lstm,gru --text title_desc_masked
step evaluate;               $PY -m jise.evaluate_groups
step al_summary;             $PY -m jise.al_summary
step done
