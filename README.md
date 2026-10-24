# target-community-uncertainty

Code and results for the paper

**Uncertainty-aware search for target communities…** <!-- TODO: full title --> (submitted to the Journal of
Information Science and Engineering, JISE).

Extended version of the WorldCIST 2026 paper.

Authors: A. V. Sokolov, A. Shutkin, E. V. Isaeva, Igor <SURNAME>.

The task: classify VK (VKontakte) communities as ready-food delivery services vs. other communities, with labels
taken from the search query that returned each group, and study how uncertainty estimates (calibration,
selective prediction, active learning) behave under such noisy, query-derived labels.

## Data

| File | Source | sha256 |
|---|---|---|
| `data/raw/vk_groups_dataset.xlsx` | [Mitigating_uncertainty_in_data_driven_marketing](https://github.com/AndreySokolovInterprogram/Mitigating_uncertainty_in_data_driven_marketing) @ `5c78b9c` | see `data/raw/SHA256SUMS` |

The raw file is not included here. Download it from the repository above at commit `5c78b9c`, put it into
`data/raw/` and check it:

```bash
cd data/raw && sha256sum -c SHA256SUMS
```

No other data are used. Derived files (embeddings) are written to `data/processed/`; per-run predictions are
written to `results/jise/groups/preds/` (both are not in git).

## Protocol

* 11,613 rows → 11,612 after removing one duplicated group ID. Text = title + description (missing → empty).
  Text variants: `title_desc`, `desc` (description only, rows with an empty description dropped) and their
  query-masked versions `*_masked` (all tokens starting with a search-query stem are removed, see
  `src/jise/query_terms.py`).
* Near-duplicates: normalized text (lower case, digits masked, punctuation removed). All rows with the same
  normalized text are in the same fold.
* Outer evaluation: 5-fold `StratifiedGroupKFold(shuffle=True, random_state=2026)`; fold ids in
  `results/jise/groups/folds_<variant>.csv`.
* Inside each training fold: 10 % inner validation (also group-aware) for checkpoint selection and temperature
  scaling. Class weights computed on the training part only. Tokenizers / TF-IDF fitted on the training part only.
* Metrics: accuracy, macro-F1, per-class F1, ROC AUC, ECE (15 bins), NLL, Brier, AURC, accuracy at 90/80 % coverage —
  mean ± std over folds.
* Seeds: model seeds 0–4 (ensemble members); split seed 2026.

## Run

Python 3.12; package versions are pinned in `requirements.txt`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PY=$PWD/.venv/bin/python
```

**CPU stage** (TF-IDF / frozen embeddings + LR, CNN / LSTM / GRU, active learning; GPU hidden on purpose):

```bash
./run_cpu_stage.sh       # original text variants
./run_masked_stage.sh    # query-masked variants
```

**GPU stage** (fine-tuning of rubert-tiny2 / multilingual-e5-base, 5 folds × 5 seeds, query-masked text):

```bash
./run_gpu_stage.sh
```

**LLM stage** (local LLM, zero-shot and 4-shot, no fine-tuning). Requires [Ollama](https://ollama.com) running on
`localhost:11434` with the models pulled:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen3:8b
RUN_LLM=1 ./run_gpu_stage.sh      # fine-tuning (finished runs are skipped), then LLM
```

Prompts: `prompts/llm_prompts.md`.

**Analyses on the saved predictions** (no training; need the stages above):

```bash
export PYTHONPATH=src
$PY -m jise.evaluate_groups       # tables and figures per text variant
$PY -m jise.shortcut_analysis     # label vs. search-query words
$PY -m jise.label_check           # manual label check
$PY -m jise.stats_tests           # paired tests
$PY -m jise.paper_figures         # figures of the paper
```

## Manual label check

`annotation/manual_label_check.xlsx` — 300 groups (stratified random sample, 150 "delivery" + 150 "other",
seed 20261020), checked by one annotator. Columns: `n`, `vk_id`, `link`, `title`, `description` (original
language, Russian), `label` (query-derived label: `delivery` / `other`), `label_correct` (verdict: `yes` / `no` /
`unclear`), `comment`. Result: 278 yes, 22 no. Class definition and instructions:
`annotation/class_definition.md`. The sample was drawn by `src/jise/make_annotation.py`; the analysis is
`src/jise/label_check.py` → `results/jise/groups/label_check/`.

Note: the column names of some committed result tables in `results/jise/groups/label_check/` are still in Russian
(they were produced before the annotation file was translated); rerunning `jise.label_check` gives English names.
The `git_commit` fields in run metadata refer to the internal development history.

## Layout

```
src/jise/          code
annotation/        manual label check
prompts/           LLM prompts
data/raw/          input file (not in git) + SHA256SUMS
results/jise/      tables, figures, run metadata (per-run predictions not in git)
```

## Citation

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {Sokolov, A. V. and Shutkin, A. and Isaeva, E. V. and <SURNAME>, Igor},
  journal = {Journal of Information Science and Engineering},
  year    = {TODO}
}
```

## License

MIT, see `LICENSE`.
