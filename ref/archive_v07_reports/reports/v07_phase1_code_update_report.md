# v0.7 Phase 1 Code Update Report

## Status

Status: passed for local code checks and dry-run validation.

Date: 2026-06-14

## Implemented

- Added v0.7 graph artifact materialization and sha256 verification in `scripts/v07_artifacts.py`.
- Added unified experiment entrypoint `scripts/run_experiment.py`.
- Updated copy runner to derive `copy_source_length=511` from `N_total=1024`, validate `B=32,q=32,d=8`, copy canonical graph artifacts into each run directory, and write `raw_config_snapshot.json` plus `resolved_config_snapshot.json`.
- Added v0.7 result fields for graph sha, canonical graph metadata, `rho_zigzag_bound`, runtime timings, LR scheduler, and random/zigzag budget diagnostics.
- Implemented per-query `random_regular` K alignment to zigzag actual post-causal K.
- Added `zigzag_certified_cosine` method semantics: same graph/mask as `zigzag_certified`, only LR scheduler differs.
- Added v0.7 WikiText data/tokenization script and wrapper entrypoint, plus v0.7 WikiText training path that reads Phase 4 tokenized artifacts.
- Updated WikiText evaluation to use the Phase 4 tokenizer pad/eos ids during test batches and to honor method-level cosine LR overrides.
- Kept per-query K arrays in memory for alignment, but archived only compact aggregate budget diagnostics.
- Added v0.7 configs under `configs/`.

## Local Validation

Commands:

```bash
python -m py_compile scripts/*.py scripts/synthetic_mvp_core/*.py
python scripts/graph_diagnostics.py --config configs/graph_v07_n1024_q32_B32_d8.json --output-dir outputs/v07_graph_n1024_q32_B32_d8 2>&1 | tee logs/graph_v07_n1024_q32_B32_d8_20260614T110346Z.log
python scripts/run_experiment.py --config configs/copy_v07_smoke_n1024_q32_B32_d8.json --output-dir outputs/copy_v07_phase1_dryrun_n1024_q32_B32_d8 --methods local,zigzag_certified,random_regular --steps 1 --batch-size 1 --eval-batches 1 --d-model 16 --layers 1 --heads 1 --ffn-dim 32 --dropout 0 --device cpu --skip-tests
```

Results:

- `py_compile`: passed.
- Copy dry-run status: `ok`.
- Dry-run methods: `local`, `zigzag_certified`, `random_regular`.
- `N_total=1024`, `copy_source_length=511`, `T=1024`, `B=32`, `q=32`, `d=8`.
- Each dry-run method copied `artifacts/graph/selected_graph.json`, `graph_certificate.json`, and `graph_generation.json` into its own output directory.
- Run graph sha matched canonical graph sha for all dry-run rows.
- `random_regular` per-query K alignment error max: `0`.
- `metrics.jsonl` contains `timestamp_utc`, `elapsed_sec_total`, `seconds_since_prev_log`, `learning_rate`, `lr_scheduler`, `tokens_per_sec`, and memory fields.
- `training_curves.png` generated for all dry-run methods.
- `summary.json` and budget JSON files are compact and do not archive per-query K arrays.
- `zigzag_certified_cosine` schedule resolution checked for copy and WikiText configs.

## Local Environment Note

Local environment:

```text
Python 3.13.13
torch 2.12.0
cuda False
device cpu
tokenizers unavailable
```

Because `tokenizers` is not installed locally, full WikiText Phase 4 tokenizer training must run in the remote experiment environment or an environment with `tokenizers` installed. The v0.7 script now fails explicitly if `tokenizers` is missing rather than silently falling back to GPT-2 or byte tokenizer.
