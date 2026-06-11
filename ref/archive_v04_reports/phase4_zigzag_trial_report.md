# Phase 4 Zig-Zag Trial Report

Date: 2026-06-10

## Status

Phase 4 is MVP-passable as a trainability trial, but not complete as a final new-model evaluation.

The current evidence validates correctness and trainability of the local + zig-zag mask path on one synthetic long-range task. It still does not claim sparse-kernel memory or speed improvements, because the current implementation uses dense score matrices with masks.

## Correctness Checks

Implemented in:

```text
scripts/synthetic_mvp.py
```

Checks performed:

- `N = 64, 128`
- `B = 8, 16`
- `d = 2, 3`
- cyclic `Rot_G` reverse consistency
- `H` degree correctness
- no empty attention rows
- no illegal indices
- dense-mask attention vs neighbor-list attention numeric agreement

Result:

```text
mask_tests: ok
cases: 8
max dense-vs-neighbor error: < 5e-7
```

Artifacts:

```text
outputs/phase4_zigzag_trial/mask_tests.json
```

## A100 Trial Configuration

Command:

```bash
cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES=0 \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python synthetic_mvp.py \
  --methods local,zigzag \
  --seq-len 512 \
  --block-size 16 \
  --degree 2 \
  --steps 400 \
  --eval-batches 20 \
  --batch-size 16 \
  --d-model 128 \
  --layers 4 \
  --heads 4 \
  --ffn-dim 512 \
  --num-keys 32 \
  --num-values 4 \
  --learning-rate 0.001 \
  --log-every 100 \
  --output-dir ../../outputs/phase4_zigzag_trial
```

Task:

```text
Associative Recall
```

This is an easier synthetic setting used only to verify no-NaN training and loss movement.

## Results

### Earlier Associative-Recall Trial

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | local | 512 | 16 | 2 | 16 | 16.0 | 8192 | 1.386313 | 0.2719 | 1.035156 |
| associative_recall | zigzag | 512 | 16 | 2 | 20 | 20.0 | 10240 | 1.386421 | 0.2687 | 1.035156 |

Training trace:

| method | step 1 valid_loss | step 400 valid_loss | step 400 logged accuracy |
| --- | --- | --- | --- |
| local | 1.893273 | 1.379536 | 0.28125 |
| zigzag | 1.858559 | 1.379474 | 0.28125 |

This run shows no NaN and loss movement, but it is not convincing task learning because the final loss is near the four-class random entropy.

### Long-Range Copy Trial

The stronger Phase-4 evidence is `copy_first` at `N=256`, `B=16`, `d=2`, `layers=8`, seed 0. The source token is at position 0 and the classifier reads from the final token representation, so local-only cannot directly solve it.

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | local | 256 | 16 | 2 | 16 | 16.0 | 4096 | 1.388920 | 0.2445 | 2.195312 |
| copy_first | zigzag | 256 | 16 | 2 | 20 | 20.0 | 5120 | 0.00001560 | 1.0000 | 2.195312 |
| copy_first | random | 256 | 16 | 2 | 20 | 20.0 | 5120 | 0.00007693 | 1.0000 | 2.195312 |

Artifacts:

- `outputs/copy_first_convergence_n256/summary.json`
- `outputs/copy_first_convergence_n256/results.csv`
- `outputs/copy_first_convergence_n256/*_metrics.jsonl`
- `logs/copy_first_convergence_n256.log`

Artifacts:

- `outputs/phase4_zigzag_trial/summary.json`
- `outputs/phase4_zigzag_trial/results.csv`
- `outputs/phase4_zigzag_trial/*_metrics.jsonl`
- `logs/phase4_zigzag_trial.log`

## Interpretation

The zig-zag path passes smoke-level checks:

- Unit/mask tests pass.
- Dense-mask and neighbor-list attention agree on small cases.
- Zig-zag training completes on A100.
- No NaN occurred.
- Validation loss moves down from the initial high value to near the random-classification entropy for four labels.
- Raw/effective K and pair counts are recorded.

This is enough to treat Phase 4 as an MVP trainability pass: zig-zag passes unit checks, trains without NaN, solves one long-range synthetic task, and is clearly better than local-only on that task.

It is not enough for final Phase-4 claims. Same-budget random also solves the task, so the current result does not show a zig-zag advantage over random. Before moving to performance tuning or full experiments, the run should be repeated at larger `N` and additional seeds.

### Neighbor-Backend Follow-Up At N=512

`scripts/synthetic_mvp.py` now supports `--attention-backend auto`, where dense uses the original debug mask path and sparse methods use precomputed neighbor tables. The follow-up run used GPU 3:

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | neighbor | 512 | 16 | 1.393645 | 0.2703 | 3.4473 |
| copy_first | random | neighbor | 512 | 20 | 1.393900 | 0.2703 | 6.2598 |
| copy_first | zigzag | neighbor | 512 | 20 | 0.000202 | 1.0000 | 6.2598 |

Artifacts:

- `outputs/neighbor_copy_first_n512_gpu3/summary.json`
- `outputs/neighbor_copy_first_n512_gpu3/results.csv`
- `outputs/neighbor_copy_first_n512_gpu3/command.sh`

This strengthens the trainability evidence for zig-zag on a longer sequence. It also shows the current naive neighbor gather is not a finished performance implementation: random/zig-zag reserve more memory than dense in this configuration. The next implementation step remains local/cross split with unified softmax and then block-pair batching.

### Local/Cross Split Follow-Up

The local/cross split backend is now implemented as `--attention-backend auto_split`. It computes local block attention and cross edges separately, but applies one softmax over the combined logits.

Clean GPU3 result:

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | split | 512 | 16 | 1.393645 | 0.2703 | 1.0332 |
| copy_first | random | split | 512 | 20 | 1.393946 | 0.2719 | 1.7969 |
| copy_first | zigzag | split | 512 | 20 | 0.000202 | 1.0000 | 1.7969 |

Artifacts:

- `outputs/split_copy_first_n512_gpu3_clean/summary.json`
- `outputs/split_copy_first_n512_gpu3_clean/results.csv`
- `outputs/split_copy_first_n512_gpu3_clean/command.sh`

This turns the earlier memory caveat into a concrete improvement: split zig-zag keeps the long-range task result and uses much less peak reserved memory than dense in this diagnostic. This paragraph records a single training-run diagnostic; the later repeated warmup/measure profile is summarized in `phase5_performance_tuning_report.md`.

### N=1024 Split Follow-Up

The split result was extended to `N=1024`:

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 1024 | 1024 | 0.020978 | 1.0000 | 5.9180 |
| copy_first | local | split | 1024 | 16 | 1.377474 | 0.3000 | 1.0488 |
| copy_first | random | split | 1024 | 20 | 1.377595 | 0.3000 | 1.8086 |
| copy_first | zigzag | split | 1024 | 20 | 0.000280 | 1.0000 | 1.8086 |

Artifacts:

- `outputs/split_copy_first_n1024_gpu3/summary.json`
- `outputs/split_copy_first_n1024_gpu3/results.csv`
- `outputs/split_copy_first_n1024_gpu3/command.sh`

This strengthens the Phase-4 trainability evidence: zig-zag still solves the long-range synthetic task at `N=1024` under the split backend.
