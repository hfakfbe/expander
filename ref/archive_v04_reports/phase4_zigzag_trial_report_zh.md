# Phase 4 Zig-Zag Trial 报告

日期：2026-06-10

## 状态

作为 trainability trial，Phase 4 达到 MVP 可通过状态；但作为最终 new-model evaluation 尚未完成。

当前证据验证了 local + zig-zag mask path 在一个 synthetic long-range task 上的正确性和可训练性。它仍不声称 sparse-kernel memory 或 speed improvements，因为当前实现使用带 mask 的 dense score matrices。

## Correctness Checks

实现位置：

```text
scripts/synthetic_mvp.py
```

执行的检查：

- `N = 64, 128`
- `B = 8, 16`
- `d = 2, 3`
- cyclic `Rot_G` reverse consistency
- `H` degree correctness
- no empty attention rows
- no illegal indices
- dense-mask attention vs neighbor-list attention numeric agreement

结果：

```text
mask_tests: ok
cases: 8
max dense-vs-neighbor error: < 5e-7
```

产物：

```text
outputs/phase4_zigzag_trial/mask_tests.json
```

## A100 Trial 配置

命令：

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

任务：

```text
Associative Recall
```

这是一个更容易的 synthetic setting，仅用于验证 no-NaN training 和 loss movement。

## 结果

### 早期 Associative-Recall Trial

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| associative_recall | local | 512 | 16 | 2 | 16 | 16.0 | 8192 | 1.386313 | 0.2719 | 1.035156 |
| associative_recall | zigzag | 512 | 16 | 2 | 20 | 20.0 | 10240 | 1.386421 | 0.2687 | 1.035156 |

Training trace：

| method | step 1 valid_loss | step 400 valid_loss | step 400 logged accuracy |
| --- | --- | --- | --- |
| local | 1.893273 | 1.379536 | 0.28125 |
| zigzag | 1.858559 | 1.379474 | 0.28125 |

这次运行显示没有 NaN 且 loss 有变化，但它不是令人信服的 task learning，因为 final loss 接近四分类 random entropy。

### Long-Range Copy Trial

更强的 Phase-4 证据是 seed 0 下的 `copy_first`，设置为 `N=256`、`B=16`、`d=2`、`layers=8`。Source token 位于 position 0，classifier 从 final token representation 读取，因此 local-only 无法直接解决。

| task | method | N | B | d | raw_K | effective_K_mean | pair_count | final_valid_loss | final_valid_accuracy | peak_reserved_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | local | 256 | 16 | 2 | 16 | 16.0 | 4096 | 1.388920 | 0.2445 | 2.195312 |
| copy_first | zigzag | 256 | 16 | 2 | 20 | 20.0 | 5120 | 0.00001560 | 1.0000 | 2.195312 |
| copy_first | random | 256 | 16 | 2 | 20 | 20.0 | 5120 | 0.00007693 | 1.0000 | 2.195312 |

产物：

- `outputs/copy_first_convergence_n256/summary.json`
- `outputs/copy_first_convergence_n256/results.csv`
- `outputs/copy_first_convergence_n256/*_metrics.jsonl`
- `logs/copy_first_convergence_n256.log`

产物：

- `outputs/phase4_zigzag_trial/summary.json`
- `outputs/phase4_zigzag_trial/results.csv`
- `outputs/phase4_zigzag_trial/*_metrics.jsonl`
- `logs/phase4_zigzag_trial.log`

## 解读

Zig-zag path 通过 smoke-level checks：

- Unit/mask tests 通过。
- Dense-mask 和 neighbor-list attention 在小 cases 上一致。
- Zig-zag training 在 A100 上完成。
- 没有出现 NaN。
- Validation loss 从初始高值下降到接近四标签 random-classification entropy。
- Raw/effective K 和 pair counts 已记录。

这足以将 Phase 4 视为 MVP trainability pass：zig-zag 通过 unit checks，无 NaN 训练，解决一个 long-range synthetic task，并且在该任务上明显优于 local-only。

但这不足以支撑最终 Phase-4 claims。Same-budget random 也解决了该任务，因此当前结果没有显示 zig-zag 相对 random 的优势。在进入 performance tuning 或 full experiments 之前，应在更大的 `N` 和更多 seeds 上重复该运行。

### Neighbor-Backend Follow-Up At N=512

`scripts/synthetic_mvp.py` 现在支持 `--attention-backend auto`，其中 dense 使用原始 debug mask path，sparse methods 使用预计算 neighbor tables。Follow-up run 使用 GPU 3：

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

这强化了 zig-zag 在更长 sequence 上的 trainability evidence。它也显示当前 naive neighbor gather 不是完成版 performance implementation：在此配置下 random/zig-zag 比 dense 保留更多 memory。下一步实现仍是带 unified softmax 的 local/cross split，然后是 block-pair batching。

### Local/Cross Split Follow-Up

Local/cross split backend 现已实现为 `--attention-backend auto_split`。它分别计算 local block attention 和 cross edges，但在 combined logits 上应用一个 softmax。

Clean GPU3 结果：

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

这将早先的 memory caveat 变成了具体改进：split zig-zag 保持 long-range task 结果，并且在该 diagnostic 中使用的 peak reserved memory 明显少于 dense。本段记录的是单次 training-run diagnostic；后续重复 warmup/measure profile 总结在 `phase5_performance_tuning_report.md`。

### N=1024 Split Follow-Up

Split 结果扩展到 `N=1024`：

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

这强化了 Phase-4 trainability evidence：在 split backend 下，zig-zag 仍能在 `N=1024` 解决 long-range synthetic task。
