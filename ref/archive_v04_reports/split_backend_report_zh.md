# Local/Cross Split Backend 报告

日期：2026-06-10

## 状态

`scripts/synthetic_mvp.py` 现在包含 `split` attention backend 和 `auto_split` selector：

- `dense_mask`：dense debug/reference path。
- `neighbor`：之前的 union-neighbor gather path。
- `split`：block-local dense attention 加 cross-neighbor gather，并在 local 与 cross logits 上使用一个 unified softmax。
- `auto_split`：dense 使用 `dense_mask`；local/random/zigzag 使用 `split`。

这遵循手册在 neighbor-list attention 之后的实现顺序：local dense 和 cross sparse 分开计算，但 softmax 在它们的 union 上执行。

## Correctness

Mask tests 现在比较三个 small-`N` paths：

- Dense masked attention。
- Union neighbor-list attention。
- Local/cross split attention。

Split-vs-dense max error 在 mask test artifacts 中记录为 `dense_split_max_error`。在 GPU3 clean run 中，检查到的 errors 低于 `1e-5`。

## GPU3 Clean Diagnostic

命令：

```text
outputs/split_copy_first_n512_gpu3_clean/command.sh
```

结果：

| task | method | backend | N | raw_K | cross shape | valid loss | valid accuracy | tokens/sec | peak allocated GB | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | - | 0.000331 | 1.0000 | 293064.22 | 3.2469 | 3.4453 |
| copy_first | local | split | 512 | 16 | `[512, 0]` | 1.393645 | 0.2703 | 445470.03 | 0.9443 | 1.0332 |
| copy_first | random | split | 512 | 20 | `[512, 4]` | 1.393946 | 0.2719 | 373449.26 | 1.6591 | 1.7969 |
| copy_first | zigzag | split | 512 | 20 | `[512, 4]` | 0.000202 | 1.0000 | 373654.43 | 1.6591 | 1.7969 |

解读：

- Zig-zag 仍然在 `N=512` 解决 long-range `copy_first` task。
- Local-only 按预期保持在 chance 附近。
- Same-budget random 在 1000 steps 内没有解决这个 seed/config。
- 与之前的 naive neighbor backend 相比，split 将 random/zigzag peak reserved memory 从约 `6.26GB` 降低到 `1.80GB`。
- 与 dense 相比，split zig-zag 使用约 `52%` 的 peak reserved memory，并在本次运行中更快。

## Caveats

本节记录原始单次 training-run timing。它不应单独作为最终 performance protocol 使用。

此外，split backend 仍在 token level gather cross edges。Block-pair grouping/sorting 后来已作为单独的 `blockpair` backend 实现，但当前 PyTorch prototype 不是最终 block-sparse kernel。

## Phase-5 Follow-Up

已经添加 graph cache generation、warmup/measure/repeat profiling entry point、`N=1024` GPU3 diagnostic，以及重复的 `N=512/N=1024` GPU profiles。见 `phase5_performance_tuning_report.md`。
