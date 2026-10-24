#!/usr/bin/env bash
# GPU stage (groups, variant without query words). Create file STOP_GPU in this folder to stop after the current run.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== $(date '+%F %T') (re)start, existing runs are skipped"
export PYTHONPATH=src PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
PY=${PY:-python3}
T=title_desc_masked
step() { echo "=== $(date '+%F %T') $*"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
stop_check() { if [ -f STOP_GPU ]; then echo "=== $(date '+%F %T') STOP_GPU found, exiting"; unload; exit 0; fi; }
unload() { for m in qwen2.5:7b-instruct qwen3:8b; do curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true; done; }
for m in rubert_tiny2 e5base; do
  for k in 0 1 2 3 4; do
    for s in 0 1 2 3 4; do
      stop_check
      $PY -m jise.finetune --models $m --text $T --folds $k --seeds $s 2>&1 | grep -E "acc=|Error|Traceback|out of memory" || true
    done
  done
done
step finetune done
if [ "${RUN_LLM:-0}" != 1 ]; then $PY -m jise.evaluate_groups > /dev/null; step done-finetune-only; exit 0; fi
for m in qwen2.5:7b-instruct qwen3:8b; do
  for sh in 0 4; do
    stop_check
    step llm $m ${sh}-shot
    $PY -m jise.llm_ollama --models $m --shots $sh --text $T
  done
  unload
done
step llm done
$PY -m jise.evaluate_groups > /dev/null
step done
