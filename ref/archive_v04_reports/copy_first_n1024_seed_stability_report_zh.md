# Copy-First N=1024 Seed 稳定性报告

日期：2026-06-11

## 目的

本报告记录在越过 synthetic MVP gate 之前要求进行的 seed-stability follow-up。它使用与 Phase-5 diagnostic 相同的 `copy_first`、`N=1024`、`B=16`、`d=2`、`auto_split` 设置，并使用 seeds `0`、`1` 和 `2`。

已遵循 GPU 策略：每次新运行前检查 GPU3，并且仅在利用率低于 `10%` 时使用。

## 产物

```text
outputs/split_copy_first_n1024_gpu3/
outputs/split_copy_first_n1024_seed1_gpu3/
outputs/split_copy_first_n1024_seed2_gpu3/
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## 最终结果

| seed | method | backend | final valid loss | final valid accuracy | peak reserved GB |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | dense | dense_mask | 0.020978 | 1.0000 | 5.9180 |
| 0 | local | split | 1.377474 | 0.3000 | 1.0488 |
| 0 | random | split | 1.377595 | 0.3000 | 1.8086 |
| 0 | zigzag | split | 0.000280 | 1.0000 | 1.8086 |
| 1 | dense | dense_mask | 0.323924 | 0.7812 | 5.9180 |
| 1 | local | split | 1.389727 | 0.2344 | 1.0488 |
| 1 | random | split | 1.390445 | 0.2344 | 1.8086 |
| 1 | zigzag | split | 0.000200 | 1.0000 | 1.8086 |
| 2 | dense | dense_mask | 0.001413 | 1.0000 | 5.9180 |
| 2 | local | split | 1.385352 | 0.2781 | 1.0488 |
| 2 | random | split | 0.001622 | 1.0000 | 1.8086 |
| 2 | zigzag | split | 0.000250 | 1.0000 | 1.8086 |

## 收敛时间

记录步数处的 validation accuracy：

| seed | method | step 1 | step 250 | step 500 | step 750 | step 1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | dense | 0.2344 | 0.2875 | 0.2500 | 0.5375 | 1.0000 |
| 0 | local | 0.2500 | 0.2406 | 0.2437 | 0.2594 | 0.2406 |
| 0 | random | 0.2500 | 0.2344 | 0.2437 | 0.2594 | 0.2406 |
| 0 | zigzag | 0.2469 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | dense | 0.2313 | 0.2781 | 0.2469 | 0.7281 | 0.7625 |
| 1 | local | 0.2562 | 0.2281 | 0.2469 | 0.2469 | 0.2656 |
| 1 | random | 0.2562 | 0.2281 | 0.2313 | 0.2406 | 0.2656 |
| 1 | zigzag | 0.2687 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | dense | 0.2500 | 0.2375 | 0.2313 | 0.2625 | 1.0000 |
| 2 | local | 0.2812 | 0.2375 | 0.2781 | 0.2531 | 0.2531 |
| 2 | random | 0.2812 | 0.2375 | 0.2781 | 1.0000 | 1.0000 |
| 2 | zigzag | 0.2812 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 解读

可靠结论：

- Zig-zag 在三个 seeds 中都解决了 `copy_first`，并且每次都在 step 250 前达到 `1.0` validation accuracy。
- Local-only 在三个 seeds 中都失败，这符合该 long-range gate 的预期。
- Dense 解决了 seeds `0` 和 `2`，但 seed `1` 到 1000 steps 时 accuracy 只有 `0.7812`。
- 同预算 random 对 seed 敏感：它在 seeds `0` 和 `1` 中失败，但在 seed `2` 中到 step 750 时解决任务。

边界：

- 这些结果强化了 zig-zag 稳定性的 synthetic long-range gate 证据。
- 它们不能证明 zig-zag 相对 same-budget random 具有类别性优势，因为 random 解决了 seed `2`。
- 它们确实表明，在这个特定 `copy_first N=1024` 设置中，zig-zag 的收敛比 dense/random 更早且更稳定。
