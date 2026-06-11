# Neighbor Backend 报告

日期：2026-06-10

## 状态

已经向 `scripts/synthetic_mvp.py` 添加了真正的 neighbor-list attention backend。

可用 backends：

- `dense_mask`：原始 debug path，物化 dense `N x N` scores 并应用 mask。
- `neighbor`：预计算 `[N, K]` neighbor tables，并对 gathered keys/values 计算 attention。
- `auto`：对 `dense` 使用 `dense_mask`，对 `local`、`random` 和 `zigzag` 使用 `neighbor`。

这在功能层面满足了手册的下一步实现要求：sparse methods 在训练期间不再需要 dense score mask path。但它尚未满足手册的 performance-tuning 目标，因为当前 gather 实现仍会创建大型中间张量，并且在 `N=512` 时 memory behavior 很差。

## 验证

Smoke run：

```text
outputs/neighbor_backend_smoke/
```

结果：

- Mask tests 通过。
- `auto` backend 完成了 forward、backward、eval 和 metric logging。
- Sparse methods 记录了 `neighbor_shape`，例如 random/zigzag 的 `[128, 20]`。

## GPU3 `N=512` Diagnostic

主 diagnostic 按要求在最后一张 A100 上运行：

```text
CUDA_VISIBLE_DEVICES=3
outputs/neighbor_copy_first_n512_gpu3/
```

| task | method | backend | N | raw_K | effective_K | valid loss | valid accuracy | tokens/sec | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | 512.0 | 0.000331 | 1.0000 | 302767.51 | 3.4453 |
| copy_first | local | neighbor | 512 | 16 | 16.0 | 1.393645 | 0.2703 | 343838.24 | 3.4473 |
| copy_first | random | neighbor | 512 | 20 | 20.0 | 1.393900 | 0.2703 | 293208.41 | 6.2598 |
| copy_first | zigzag | neighbor | 512 | 20 | 20.0 | 0.000202 | 1.0000 | 277966.38 | 6.2598 |

解读：

- Dense 确认该任务可学习。
- Local-only 保持在四分类 chance 附近，符合 long-range copy task 的预期。
- Zig-zag 在 `N=512`、`K=20` 下解决该任务。
- Same-budget random 在 1000 steps 内没有解决这个 seed/config，不同于早先的 `N=256` run。
- 当前 neighbor backend 不具 memory efficiency：random/zigzag 的 peak reserved memory 高于 dense。这很可能来自 advanced-index gather intermediates，因此下一步实现应避免 batch/head-expanded gathers，并转向 local/cross split 或 block-pair batching。

## 产物

- `outputs/neighbor_copy_first_n512_gpu3/summary.json`
- `outputs/neighbor_copy_first_n512_gpu3/results.csv`
- `outputs/neighbor_copy_first_n512_gpu3/*_metrics.jsonl`
- `outputs/neighbor_copy_first_n512_gpu3/command.sh`

## 后续

本报告中的下一步现已作为 `split` / `auto_split` backend 实现。GPU3 clean diagnostic 见 `split_backend_report.md`。
