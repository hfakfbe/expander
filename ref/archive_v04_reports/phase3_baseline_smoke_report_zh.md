# Phase 3 Baseline Smoke 报告

日期：2026-06-10

## 状态

对于 MVP gate，Phase 3 已部分完成，但还不足以支撑最终 baseline claims。

下面的运行证明 pipeline 可以运行，并包含有意义的 synthetic long-range diagnostics。但它们仍不应被用作最终 baseline comparisons，因为 official LRA data 缺失，neighbor backend 仍是 naive gather implementation，并且只运行了 seed 0。

已完成内容：

- `huiwei` 上的 base repository GPU smoke test 通过，使用 `guy-dar/lra-benchmarks`。
- Project-local synthetic MVP baseline script 成功运行。
- Dense、local-only、random same-budget 和 zig-zag mask variants 都完成了 short training，且没有 NaN。
- Metrics 包含 loss、accuracy、tokens/sec、raw K、effective K、pair count、duplicate-rate estimate 和 self-loop rate。
- A100 synthetic baseline smoke 已在 `N = 1024` 完成。
- Base dense BERT ListOps smoke 已在 generated ListOps-compatible data 上完成。
- `N = 256` 的 long-range `copy_first` synthetic convergence run 已完成，其中 local-only 失败，dense/random/zig-zag 解决任务。

根据手册，在进入 new-model full training 前仍需要：

- 超过 smoke length 的更长 baseline runs。
- 如果需要 official benchmark comparability，则需要原始下载的 LRA data。
- 更广的 `N = 2048, 4096` synthetic baseline grid。
- 对于 performance claims，需要用 memory-efficient local/cross 或 block-pair layout 替代 naive neighbor gather。

## A100 Long-Range Copy Diagnostic

这次运行比 `copy_visible` 信息量更大：target 是第一个 token，而 classification 从 final token representation 读取。Local-only 无法直接访问 source token。

| task | method | N | B | d | raw_K | effective_K_mean | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | 256 | 16 | 2 | 256 | 256.0 | 0.00002081 | 1.0000 | 2.195312 |
| copy_first | local | 256 | 16 | 2 | 16 | 16.0 | 1.388920 | 0.2445 | 2.195312 |
| copy_first | random | 256 | 16 | 2 | 20 | 20.0 | 0.00007693 | 1.0000 | 2.195312 |
| copy_first | zigzag | 256 | 16 | 2 | 20 | 20.0 | 0.00001560 | 1.0000 | 2.195312 |

产物：

- `outputs/copy_first_convergence_n256/summary.json`
- `outputs/copy_first_convergence_n256/results.csv`
- `outputs/copy_first_convergence_n256/*_metrics.jsonl`
- `logs/copy_first_convergence_n256.log`

解读：

- 这确认 cross-block edges 至少在一个 synthetic task 上很重要。
- Same-budget random 也解决了任务，因此这不是 zig-zag 优于 random 的证据。
- 这次早期运行中的 memory 相同是预期行为，因为该运行对所有 methods 都使用 dense-score debug path。

## GPU3 Neighbor-Backend Diagnostic

Follow-up `N=512` run 使用 `--attention-backend auto`：dense 使用 debug dense-mask path，而 local/random/zig-zag 使用预计算 neighbor tables。

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | neighbor | 512 | 16 | 1.393645 | 0.2703 | 3.4473 |
| copy_first | random | neighbor | 512 | 20 | 1.393900 | 0.2703 | 6.2598 |
| copy_first | zigzag | neighbor | 512 | 20 | 0.000202 | 1.0000 | 6.2598 |

产物：

- `outputs/neighbor_copy_first_n512_gpu3/summary.json`
- `outputs/neighbor_copy_first_n512_gpu3/results.csv`
- `outputs/neighbor_copy_first_n512_gpu3/command.sh`

这强化了 synthetic baseline story：local 失败，zig-zag 成功，而这个特定 random seed/config 在 1000 steps 内没有解决任务。它仍不是 sparse performance win，因为 naive gather path 保留了过多 memory。

## GPU3 Local/Cross Split Diagnostic

Follow-up `N=512` clean run 使用 `--attention-backend auto_split`，它将 dense 保持为 reference path，并对 local/random/zig-zag 使用 local/cross split。

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | split | 512 | 16 | 1.393645 | 0.2703 | 1.0332 |
| copy_first | random | split | 512 | 20 | 1.393946 | 0.2719 | 1.7969 |
| copy_first | zigzag | split | 512 | 20 | 0.000202 | 1.0000 | 1.7969 |

产物：

- `outputs/split_copy_first_n512_gpu3_clean/summary.json`
- `outputs/split_copy_first_n512_gpu3_clean/results.csv`
- `outputs/split_copy_first_n512_gpu3_clean/command.sh`

这是当前最强的 synthetic baseline evidence：local 失败，这个 seed/config 下 random 失败，zig-zag 成功，并且 split sparse attention 在这次运行中使用的 memory 少于 dense。

## GPU3 N=1024 Split Diagnostic

相同的 `auto_split` diagnostic 被扩展到 `N=1024`，batch size 为 16。

| task | method | backend | N | raw_K | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 1024 | 1024 | 0.020978 | 1.0000 | 5.9180 |
| copy_first | local | split | 1024 | 16 | 1.377474 | 0.3000 | 1.0488 |
| copy_first | random | split | 1024 | 20 | 1.377595 | 0.3000 | 1.8086 |
| copy_first | zigzag | split | 1024 | 20 | 0.000280 | 1.0000 | 1.8086 |

产物：

- `outputs/split_copy_first_n1024_gpu3/summary.json`
- `outputs/split_copy_first_n1024_gpu3/results.csv`
- `outputs/split_copy_first_n1024_gpu3/command.sh`

这将 synthetic baseline evidence 扩展到 `N=1024`：local-only 和 same-budget random 保持在 chance 附近，而 dense 和 zig-zag 解决任务。

## CPU Synthetic MVP Results

这些只是 smoke results。它们验证 harness，但不是质量结论。

| task | method | N | B | d | raw_K | effective_K_mean | final_valid_loss | final_valid_accuracy | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | dense | 128 | 16 | 2 | 128 | 128.0 | 2.327524 | 0.1000 | cpu |
| associative_recall | local | 128 | 16 | 2 | 16 | 16.0 | 2.328650 | 0.1000 | cpu |
| associative_recall | random | 128 | 16 | 2 | 20 | 20.0 | 2.330900 | 0.0750 | cpu |
| associative_recall | zigzag | 128 | 16 | 2 | 20 | 20.0 | 2.326729 | 0.0875 | cpu |

产物：

- `outputs/synthetic_mvp_cpu/summary.json`
- `outputs/synthetic_mvp_cpu/*_metrics.jsonl`
- `outputs/synthetic_mvp_cpu/mask_tests.json`
- `logs/synthetic_mvp_cpu.log`

## A100 Synthetic MVP Results

这次运行对所有 masks 都使用相同的 dense-score debug implementation，因此 memory 预计尚不会随理论 pair count 缩放。它是 GPU training 和 mask-fairness baseline，不是 sparse-kernel benchmark。

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | valid_loss | valid_accuracy | tokens/sec | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | dense | 1024 | 16 | 2 | 1024 | 1024.0 | 1048576 | 2.355018 | 0.0750 | 173454.807 | 1.783203 |
| associative_recall | local | 1024 | 16 | 2 | 16 | 16.0 | 16384 | 2.355216 | 0.0750 | 208076.216 | 1.783203 |
| associative_recall | random | 1024 | 16 | 2 | 20 | 20.0 | 20480 | 2.355375 | 0.0750 | 206008.660 | 1.783203 |
| associative_recall | zigzag | 1024 | 16 | 2 | 20 | 20.0 | 20480 | 2.351962 | 0.0875 | 208438.076 | 1.783203 |

产物：

- `outputs/synthetic_mvp_gpu_n1024/summary.json`
- `outputs/synthetic_mvp_gpu_n1024/results.csv`
- `logs/synthetic_mvp_gpu_n1024.log`

## Generated ListOps Base GPU Smoke

Generated ListOps-compatible split：

```text
train: 1024
eval: 256
test: 256
max_length: 512
```

Base dense BERT smoke result：

| task | method | steps | batch | eval_loss | eval_accuracy | tokens/sec | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_listops | base_dense_bert | 100 | 4 | 2.273257 | 0.1750 | 121409.895 | 0.087891 |

产物：

- `outputs/listops_base_gpu/summary.json`
- `outputs/listops_base_gpu/results.csv`
- `logs/listops_base_gpu.log`

## GPU 可用性说明

早些时候，在 CPU fallback 之前，四张 A100 GPU 上都显示已有用户进程：

```text
GPU 0: modded-nanogpt process, about 1496 MiB
GPU 1: modded-nanogpt process, about 1500 MiB
GPU 2: modded-nanogpt process, about 1500 MiB
GPU 3: modded-nanogpt process, about 1500 MiB
```

用户报告服务器空闲后，重新检查四张 A100 GPU 均为空闲，并且 GPU smoke runs 只在 GPU 0 上启动。

## Base Repository GPU Smoke

基础仓库本身早先确实运行过 100-step GPU smoke test：

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

该运行使用 generated tiny ListOps-format TSV files，并记录于：

```text
smoke_test_log.txt
outputs/base_listops_smoke.json
```

## 必需的 Convergence Runs

在手册的 full benchmark 意义上标记 Phase 3 完成之前：

1. 为 selected synthetic gate 上的 dense/local/random/zig-zag 添加 seeds `1` 和 `2`。
2. 为 generated ListOps 添加 local/random/zig-zag baselines，或获取 official LRA split。
3. 在所有表格中保持 generated ListOps 与 official LRA ListOps 分离。

Phase-5 synthetic MVP profiling 现已使用手册的 warmup/measure/repeat protocol 运行；见 `phase5_performance_tuning_report.md`。
