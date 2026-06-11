# Phase 2 Task Preparation

Date: 2026-06-10

## Status

Phase 2 is complete for the current MVP gate.

Completed:

- Fixed a first synthetic task: Associative Recall.
- Added a reproducible MVP config: `configs/synthetic_mvp.json`.
- Added a runnable synthetic experiment script: `scripts/synthetic_mvp.py`.
- Implemented dense, local-only, local + random same-budget, and local + zig-zag masks for the MVP.
- Implemented mask correctness checks for `N = 64, 128`, `B = 8, 16`, `d = 2, 3`.
- Verified dense-mask attention and neighbor-list attention agree on small cases.
- Produced metrics/log artifacts under `outputs/synthetic_mvp_cpu/` and `logs/synthetic_mvp_cpu.log`.
- Prepared a generated ListOps-compatible dataset using a project-local generator based on the official LRA ListOps rules.
- Verified the base repository can load the generated ListOps TSV files through its original `ListOpsDataset`.

Not yet complete:

- The original downloaded LRA release is still not available. The documented Google Storage URL failed from `huiwei`, and Hugging Face probing timed out.
- LRA Text/Retrieval/AAN are not prepared.
- Full first-grid GPU configs are not complete; only smoke/MVP GPU runs are complete.

## Fixed Synthetic Task

Task:

```text
Associative Recall
```

Current MVP generator:

- Sequence contains key/value pairs.
- Last two positions contain a query marker and a queried key.
- Target is the value associated with that key.
- Metric is classification accuracy over value classes.

Current task constants:

```text
num_keys: 64
num_values: 10
query_token: 75
pad_token: 0
```

MVP sequence length:

```text
N = 128
```

This is deliberately smaller than the manual's first full grid. It is used only to validate the experiment harness.

## Generated ListOps-Compatible Data

Because the upstream `lra_release.gz` URL was not usable from `huiwei`, a generated ListOps-compatible split was prepared with:

```text
scripts/prepare_generated_listops.py
```

Remote output:

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks/datasets/lra_release/listops-1000/
```

Local copy:

```text
code/lra-benchmarks/datasets/lra_release/listops-1000/
```

Split sizes:

```text
basic_train.tsv: 1024 examples
basic_val.tsv:   256 examples
basic_test.tsv:  256 examples
```

Generation settings:

```text
max_depth: 10
max_args: 10
min_tokens: 24
max_tokens: 512
seed: 2
mean_tokens: 221.64
```

Important boundary: this is generated from the ListOps grammar and label rules, but it is not the original released LRA split.

## Fixed Mask / Baseline Methods

| Method | Mask | Raw K for MVP |
| --- | --- | --- |
| dense | Full sequence attention | 128 |
| local | Block-local complete attention | 16 |
| random | Local complete + random nonlocal same-budget edges | 20 |
| zigzag | Local complete + cyclic-G/cycle-H zig-zag edges | 20 |

MVP graph parameters:

```text
B = 16
d = 2
raw K for random/zigzag = B + d^2 = 20
```

## Mask Correctness Tests

The current script checks:

- Rot_G reverse consistency for cyclic G.
- H degree equals configured `d`.
- No empty attention rows.
- Dense-mask attention and neighbor-list attention produce close outputs.
- Raw K, effective K, pair count, duplicate-rate estimate, and self-loop rate are recorded.

Result:

```text
mask_tests: ok
cases: 8
max dense-vs-neighbor error: < 5e-7
```

Artifact:

```text
outputs/synthetic_mvp_cpu/mask_tests.json
```

## Run Command

CPU MVP run used because GPUs were occupied:

```bash
cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES="" \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python synthetic_mvp.py \
  --methods dense,local,random,zigzag \
  --seq-len 128 \
  --block-size 16 \
  --degree 2 \
  --steps 60 \
  --eval-batches 5 \
  --batch-size 16 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --ffn-dim 128 \
  --log-every 20 \
  --output-dir ../../outputs/synthetic_mvp_cpu
```

## Phase-2 Gate

For the manual's strict pipeline, the MVP version of Phase 2 is now passable because:

- Synthetic Associative Recall is fixed and runnable.
- Generated ListOps-compatible data is fixed and runnable.
- Base forward/backward/eval was verified on generated ListOps with A100.

Remaining caveat: do not report generated ListOps numbers as official LRA benchmark results.
