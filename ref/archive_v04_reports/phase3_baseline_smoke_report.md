# Phase 3 Baseline Smoke Report

Date: 2026-06-10

## Status

Phase 3 is partially complete for the MVP gate, but not complete for final baseline claims.

The runs below prove the pipeline runs and include meaningful synthetic long-range diagnostics. They should still not be used as final baseline comparisons because official LRA data is missing, the neighbor backend is still a naive gather implementation, and only seed 0 has been run.

What has been completed:

- Base repository GPU smoke test on `huiwei` passed with `guy-dar/lra-benchmarks`.
- A project-local synthetic MVP baseline script ran successfully.
- Dense, local-only, random same-budget, and zig-zag mask variants all completed short training without NaN.
- Metrics include loss, accuracy, tokens/sec, raw K, effective K, pair count, duplicate-rate estimate, and self-loop rate.
- A100 synthetic baseline smoke completed at `N = 1024`.
- Base dense BERT ListOps smoke completed on generated ListOps-compatible data.
- A long-range `copy_first` synthetic convergence run completed at `N = 256`, where local-only fails and dense/random/zig-zag solve the task.

What is still required by the manual before moving to new-model full training:

- Longer baseline runs beyond smoke length.
- Original downloaded LRA data, if official benchmark comparability is required.
- Broader `N = 2048, 4096` synthetic baseline grid.
- Replacement of the naive neighbor gather with a memory-efficient local/cross or block-pair layout for performance claims.

## A100 Long-Range Copy Diagnostic

This run is more informative than `copy_visible`: the target is the first token, while classification happens from the final token representation. Local-only cannot directly access the source token.

| task | method | N | B | d | raw_K | effective_K_mean | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | 256 | 16 | 2 | 256 | 256.0 | 0.00002081 | 1.0000 | 2.195312 |
| copy_first | local | 256 | 16 | 2 | 16 | 16.0 | 1.388920 | 0.2445 | 2.195312 |
| copy_first | random | 256 | 16 | 2 | 20 | 20.0 | 0.00007693 | 1.0000 | 2.195312 |
| copy_first | zigzag | 256 | 16 | 2 | 20 | 20.0 | 0.00001560 | 1.0000 | 2.195312 |

Artifacts:

- `outputs/copy_first_convergence_n256/summary.json`
- `outputs/copy_first_convergence_n256/results.csv`
- `outputs/copy_first_convergence_n256/*_metrics.jsonl`
- `logs/copy_first_convergence_n256.log`

Interpretation:

- This confirms that cross-block edges matter on at least one synthetic task.
- Same-budget random also solves the task, so this is not evidence that zig-zag is better than random.
- Identical memory in this earlier run is expected because that run used the dense-score debug path for all methods.

## GPU3 Neighbor-Backend Diagnostic

The follow-up `N=512` run used `--attention-backend auto`: dense used the debug dense-mask path, while local/random/zig-zag used precomputed neighbor tables.

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

This strengthens the synthetic baseline story: local fails, zig-zag succeeds, and this particular random seed/config does not solve the task within 1000 steps. It is still not a sparse performance win because the naive gather path reserves too much memory.

## GPU3 Local/Cross Split Diagnostic

The follow-up `N=512` clean run used `--attention-backend auto_split`, which keeps dense as the reference path and uses local/cross split for local/random/zig-zag.

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

This is the strongest current synthetic baseline evidence: local fails, random fails for this seed/config, zig-zag succeeds, and split sparse attention uses less memory than dense in this run.

## GPU3 N=1024 Split Diagnostic

The same `auto_split` diagnostic was extended to `N=1024` with batch size 16.

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

This extends the synthetic baseline evidence to `N=1024`: local-only and same-budget random remain near chance, while dense and zig-zag solve the task.

## CPU Synthetic MVP Results

These are smoke results only. They validate the harness but are not quality conclusions.

| task | method | N | B | d | raw_K | effective_K_mean | final_valid_loss | final_valid_accuracy | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | dense | 128 | 16 | 2 | 128 | 128.0 | 2.327524 | 0.1000 | cpu |
| associative_recall | local | 128 | 16 | 2 | 16 | 16.0 | 2.328650 | 0.1000 | cpu |
| associative_recall | random | 128 | 16 | 2 | 20 | 20.0 | 2.330900 | 0.0750 | cpu |
| associative_recall | zigzag | 128 | 16 | 2 | 20 | 20.0 | 2.326729 | 0.0875 | cpu |

Artifacts:

- `outputs/synthetic_mvp_cpu/summary.json`
- `outputs/synthetic_mvp_cpu/*_metrics.jsonl`
- `outputs/synthetic_mvp_cpu/mask_tests.json`
- `logs/synthetic_mvp_cpu.log`

## A100 Synthetic MVP Results

This run uses the same dense-score debug implementation for all masks, so memory is not expected to scale with the theoretical pair count yet. It is a GPU training and mask-fairness baseline, not a sparse-kernel benchmark.

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | valid_loss | valid_accuracy | tokens/sec | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | dense | 1024 | 16 | 2 | 1024 | 1024.0 | 1048576 | 2.355018 | 0.0750 | 173454.807 | 1.783203 |
| associative_recall | local | 1024 | 16 | 2 | 16 | 16.0 | 16384 | 2.355216 | 0.0750 | 208076.216 | 1.783203 |
| associative_recall | random | 1024 | 16 | 2 | 20 | 20.0 | 20480 | 2.355375 | 0.0750 | 206008.660 | 1.783203 |
| associative_recall | zigzag | 1024 | 16 | 2 | 20 | 20.0 | 20480 | 2.351962 | 0.0875 | 208438.076 | 1.783203 |

Artifacts:

- `outputs/synthetic_mvp_gpu_n1024/summary.json`
- `outputs/synthetic_mvp_gpu_n1024/results.csv`
- `logs/synthetic_mvp_gpu_n1024.log`

## Generated ListOps Base GPU Smoke

Generated ListOps-compatible split:

```text
train: 1024
eval: 256
test: 256
max_length: 512
```

Base dense BERT smoke result:

| task | method | steps | batch | eval_loss | eval_accuracy | tokens/sec | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_listops | base_dense_bert | 100 | 4 | 2.273257 | 0.1750 | 121409.895 | 0.087891 |

Artifacts:

- `outputs/listops_base_gpu/summary.json`
- `outputs/listops_base_gpu/results.csv`
- `logs/listops_base_gpu.log`

## GPU Availability Note

Earlier, before the CPU fallback, all four A100 GPUs showed existing user processes:

```text
GPU 0: modded-nanogpt process, about 1496 MiB
GPU 1: modded-nanogpt process, about 1500 MiB
GPU 2: modded-nanogpt process, about 1500 MiB
GPU 3: modded-nanogpt process, about 1500 MiB
```

After the user reported the server was free, all four A100 GPUs were rechecked as idle, and GPU smoke runs were launched on GPU 0 only.

## Base Repository GPU Smoke

The base repository itself did run a 100-step GPU smoke test earlier:

```text
repo: /home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
commit: afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
steps: 100
device: cuda
gpu: NVIDIA A100-SXM4-80GB
final_loss: 1.0127418041229248
tokens_per_sec: 18974.068654582854
peak_allocated_gb: 0.0232696533203125
```

That run used generated tiny ListOps-format TSV files and is recorded in:

```text
smoke_test_log.txt
outputs/base_listops_smoke.json
```

## Required Convergence Runs

Before Phase 3 can be marked complete in the manual's full benchmark sense:

1. Add seeds `1` and `2` for dense/local/random/zig-zag on the selected synthetic gate.
2. Add local/random/zig-zag baselines for generated ListOps or obtain the official LRA split.
3. Keep generated ListOps separate from official LRA ListOps in all tables.

Phase-5 synthetic MVP profiling has now been run with the manual's warmup/measure/repeat protocol; see `phase5_performance_tuning_report.md`.
