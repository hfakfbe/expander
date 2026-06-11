# Phase 6 Strict Implementation Status

Date: 2026-06-11

## Implemented

- Restored the full `guy-dar/lra-benchmarks` source into `code/lra-benchmarks/` while preserving the existing generated ListOps data.
- Preserved the selected base commit reference:

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

- Fixed `code/lra-benchmarks/fetch_data.py` so its dataset table is indexed by the selected task, downloads fail on HTTP errors, extraction uses the requested destination directory, and the ListOps branch writes `lra_release.gz`.
- Added strict data readiness tooling:

```text
scripts/phase6_data_readiness.py
outputs/phase6_data_readiness/readiness.json
```

- Added a Phase-6 runner and result table writer:

```text
scripts/phase6_runner.py
configs/phase6_strict_plan.json
outputs/phase6_runner_smoke/phase6_results.csv
outputs/phase6_runner_smoke/phase6_results.jsonl
```

- Added base BERT attention replacement smoke tooling:

```text
scripts/base_attention_smoke.py
scripts/phase6_static_checks.py
outputs/phase6_static_checks.json
```

- Extended `scripts/listops_base_smoke.py` with `--task listops|imdb`, so the same base smoke path can validate both official ListOps and LRA Text/IMDB once data is ready.

## Current Data Gate

Current gate status is blocked, as required by the strict manual.

Generated ListOps is present but remains pipeline-only:

```text
basic_train.tsv rows: 1024
basic_val.tsv rows: 256
basic_test.tsv rows: 256
source_scope: pipeline_only_generated_or_incomplete
ready_for_official_claims: false
```

Official ListOps download attempt:

```text
https://storage.googleapis.com/long-range-arena/lra_release.gz
curl: (56) The requested URL returned error: 403
```

IMDB/Text is not ready locally:

```text
datasets/aclImdb/train: missing
datasets/aclImdb/test: missing
```

An incomplete local IMDB archive was detected and rejected:

```text
datasets/_archives/aclImdb_v1.tar.gz
gzip_error: Compressed file ended before the end-of-stream marker was reached
```

## Verification Run

Passed:

```bash
python -m py_compile \
  scripts/phase6_data_readiness.py \
  scripts/phase6_runner.py \
  scripts/base_attention_smoke.py \
  scripts/phase6_static_checks.py \
  scripts/listops_base_smoke.py \
  code/lra-benchmarks/fetch_data.py
```

Passed:

```bash
python scripts/phase6_static_checks.py
```

Passed and correctly blocked official claims:

```bash
python scripts/phase6_data_readiness.py \
  --repo-dir code/lra-benchmarks \
  --output-json outputs/phase6_data_readiness/readiness.json
```

Passed synthetic Phase-6 table smoke:

```bash
python scripts/phase6_runner.py \
  --task copy_first \
  --methods dense,local \
  --seq-len 64 \
  --block-size 16 \
  --degree 2 \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/phase6_runner_smoke
```

The smoke output uses the required Phase-6 schema fields.

## Remaining Strict-Mode Blockers

Phase 6 official experiments must not start until:

1. Official ListOps data is acquired from a verified source and passes integrity, row-count, label, and max-length checks.
2. IMDB/Text data is fully downloaded and extracted under `code/lra-benchmarks/datasets/aclImdb/`.
3. Dense base smoke passes for both `listops` and `imdb`.
4. `scripts/base_attention_smoke.py` passes in the server environment with `pandas`, `torch`, and `transformers` available.

Recommended remote commands after syncing this workspace:

```bash
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base

python scripts/phase6_data_readiness.py \
  --repo-dir code/lra-benchmarks \
  --output-json outputs/phase6_data_readiness/readiness_download_attempt.json \
  --download \
  --timeout 120

python scripts/listops_base_smoke.py \
  --repo-dir code/lra-benchmarks \
  --task imdb \
  --steps 10 \
  --batch-size 2 \
  --max-length 128 \
  --hidden-size 64 \
  --layers 1 \
  --heads 4 \
  --output-json outputs/phase6_imdb_base_smoke/summary.json

python scripts/base_attention_smoke.py \
  --repo-dir code/lra-benchmarks \
  --task listops \
  --method zigzag \
  --backend split \
  --seq-len 64 \
  --block-size 16 \
  --degree 2 \
  --hidden-size 64 \
  --layers 1 \
  --heads 4 \
  --batch-size 2 \
  --output-json outputs/base_attention_smoke/listops_zigzag_split.json
```
