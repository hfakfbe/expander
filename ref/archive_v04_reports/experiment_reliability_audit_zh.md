# 实验可靠性审计

日期：2026-06-11

## 结论摘要

当前实验状态在 synthetic-MVP Phase-5 gate 以内是可靠的，但仍不是对 `ref/zigzag_experiment_execution_manual_v04.md` 的完整执行，因为 official LRA/full experiments 还没有开始。

足以作为证据使用的内容：

- Phase 1 base-repository 选择和 smoke test。
- small `N` 上 dense-mask 与 neighbor-list 等价性的 mask correctness checks。
- `N=256` 的 long-range synthetic `copy_first` convergence diagnostic，其中 local-only 失败，而 dense/random/zigzag 成功。
- GPU3 `N=512` neighbor-backend diagnostic，其中 zig-zag 解决了 `copy_first`，local/random 在 1000 steps 内没有解决。
- GPU3 `N=512` local/cross split diagnostic，其中 zig-zag 保持 100% accuracy，且 peak reserved memory 降到 dense 以下。
- GPU3 `N=1024` local/cross split diagnostic，其中 zig-zag 再次以低于 dense 的 peak memory 解决了 `copy_first`。
- CPU 生成的 graph caches，包含 `N=512,B=16,d=2` 的 neighbor、edge、cross-edge 和 slot-aware block-pair metadata。
- split 与 block-pair backends 在 `N=512` 和 `N=1024` 上的多次 GPU3 warmup/measure/profile runs。
- 三个 seed 的 `copy_first N=1024` follow-up，其中 zig-zag 在 step 250 前解决所有 seeds，local-only 在所有 seeds 中失败。
- Generated-ListOps base smoke/convergence 证据，可作为 pipeline check。

尚不足以支撑最终结论的内容：

- Official LRA benchmark quality，因为原始 LRA release 不可用，generated ListOps 不是 released split。
- Zig-zag 相对 same-budget random 的类别性优势。`N=1024` 三 seed 运行在稳定性和收敛速度上有利于 zig-zag，但 random 解决了 seed `2`。
- 最终 block-pair kernel performance。当前 block-pair backend 验证了 sorted edge layout，但仍是 PyTorch scatter prototype。
- 完整 ablation、official benchmark 或 scaling-law 结论。

## 阶段状态

| Phase | Manual gate | Current status | Reliability verdict |
| --- | --- | --- | --- |
| 1. Repository evaluation | 可复用 base repo smoke test 和决策产物。 | 完成：选择 `guy-dar/lra-benchmarks`，A100 smoke 通过。 | 在已记录 caveats 下可靠。 |
| 2. Task preparation | Synthetic + ListOps-like data load、configs、forward path。 | MVP 完成：synthetic tasks 和 generated ListOps 可加载。Official LRA data 缺失。 | 对 pipeline 可靠；不是 official LRA。 |
| 3. Base training/eval | Dense/local/random baselines 和至少一个 LRA task。 | 部分完成：synthetic dense/local/random 已完成；generated ListOps dense base 已完成。 | 足够用于 MVP diagnostics，不是最终 baseline table。 |
| 4. New model trial | Mask tests 通过；zig-zag 可训练；无 NaN；至少一个 task 上不差于 local。 | MVP 通过于 `copy_first`；新增三 seed `N=1024` follow-up；早期 associative-recall run 不具结论性。 | 对 synthetic long-range gate 可靠，不是最终 benchmark result。 |
| 5. Performance tuning | 真 sparse layout、memory/speed protocol。 | 对 synthetic MVP 完成：实现 split/block-pair backends，生成 graph caches，并在 N=512/N=1024 上完成重复 GPU profiles。 | 对 synthetic Phase-5 gate 可靠；不是最终 block-sparse kernel 声明。 |
| 6. Full experiments | Main tables、ablations、seeds。 | 未开始。 | 不允许最终质量结论。 |
| 7. Scaling law | N/P/K/compute trends。 | 未开始。 | 不允许 scaling 结论。 |

## 关键可靠性发现

### 1. `copy_visible` 不是 Long-Range Test

`copy_visible` 将 source token 放在 `seq_len - 2`，而 classifier 从 final token representation 读取。对于 `B=16`，local-only attention 可以直接看到 source。`outputs/copy_visible_convergence/` 中的 100% accuracy 证明 training loop 可以收敛，但它没有测试跨 block 通信。

### 2. `copy_first` 是有用的 Synthetic Gate

`outputs/copy_first_convergence_n256/` 中同步后的远程运行是更强的 diagnostic：

| task | method | N | raw_K | effective_K | final valid loss | final valid accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | 256 | 256 | 256.0 | 0.00002081 | 1.0000 |
| copy_first | local | 256 | 16 | 16.0 | 1.388920 | 0.2445 |
| copy_first | random | 256 | 20 | 20.0 | 0.00007693 | 1.0000 |
| copy_first | zigzag | 256 | 20 | 20.0 | 0.00001560 | 1.0000 |

解读：

- Local-only 保持在四分类 chance 附近，符合预期。
- Zig-zag 能够在与 random 相同 raw budget 下通过 8 层传递 long-range information。
- Random 也解决了该任务，因此这不是 zig-zag 优于 random 的证据。

### 3. Associative Recall 仍不具结论性

`phase4_zigzag_trial` associative-recall run 达到接近四标签 random-class entropy 的水平。它显示没有 NaN 且 loss 有一些变化，但没有证明有意义的 task learning。

### 4. Memory/Speed 数字是 Debug-Mode 数字

Dense、local、random 和 zig-zag 在 synthetic runs 中都报告相同的 peak memory，因为 `MaskedSelfAttention` 在应用 mask 前计算 dense `N x N` scores。这些 logs 对 pair-count bookkeeping 和 trainability 有用，但不能支撑 sparse efficiency claims。

### 5. Neighbor Backend 改变了正确性叙事，但还没有改变效率叙事

`scripts/synthetic_mvp.py` 现在支持 `--attention-backend auto`，它对 sparse methods 使用预计算 neighbor tables。GPU3 `N=512` diagnostic 位于 `outputs/neighbor_copy_first_n512_gpu3/`，显示：

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | neighbor | 512 | 1.393645 | 0.2703 | 3.4473 |
| copy_first | random | neighbor | 512 | 1.393900 | 0.2703 | 6.2598 |
| copy_first | zigzag | neighbor | 512 | 0.000202 | 1.0000 | 6.2598 |

这是 trainability 和 long-range communication 的有用证据，但也对当前 naive gather layout 作为 performance implementation 给出了负面证据。

### 6. Local/Cross Split 修复了第一个 Layout 问题

`scripts/synthetic_mvp.py` 现在支持 `--attention-backend auto_split`，它将 block-local dense attention 与 `d^2` cross edges 分开计算，并使用一个 unified softmax。`outputs/split_copy_first_n512_gpu3_clean/` 中的 clean GPU3 `N=512` run 显示：

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | split | 512 | 1.393645 | 0.2703 | 1.0332 |
| copy_first | random | split | 512 | 1.393946 | 0.2719 | 1.7969 |
| copy_first | zigzag | split | 512 | 0.000202 | 1.0000 | 1.7969 |

这是 sparse layout 能够在保留 zig-zag long-range behavior 的同时降低测得 memory 的第一份证据。

### 7. N=1024 扩展了 Split 结果

GPU3 `N=1024` run 位于 `outputs/split_copy_first_n1024_gpu3/`，显示：

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 1024 | 0.020978 | 1.0000 | 5.9180 |
| copy_first | local | split | 1024 | 1.377474 | 0.3000 | 1.0488 |
| copy_first | random | split | 1024 | 1.377595 | 0.3000 | 1.8086 |
| copy_first | zigzag | split | 1024 | 0.000280 | 1.0000 | 1.8086 |

这将当前 trainability claim 从 `N=512` diagnostic 加强到 `N=1024` diagnostic。重复 performance protocol 总结如下。

### 8. Phase-5 Repeated Profile 已完成 Synthetic Gate

手册的 warmup/measure/repeat protocol 已在 GPU3 上运行，使用 `warmup=20`、`measure=100` 和 `repeats=3`。

在 `N=1024`，最强结果是 split zig-zag backend：

| backend | method | tokens/sec mean | peak reserved GB |
| --- | --- | ---: | ---: |
| dense | dense | 189398.39 | 6.0299 |
| split | zigzag | 332173.36 | 1.8066 |
| blockpair | zigzag | 372037.00 | 2.1113 |

解读：

- Split zig-zag 达到约 `1.75x` dense throughput，且在 `N=1024` 仅使用约 `30%` 的 dense peak reserved memory。
- 在修复 non-contiguous accumulation 后，block-pair backend 数值等价；本次运行中它在 `N=1024` 快于 split zig-zag。
- 当前 block-pair prototype 比 split 使用更多 memory，因此应将其视为 layout evidence，而不是最终优化 backend。

### 9. Three-Seed Copy-First Follow-Up

`copy_first N=1024` split-backend diagnostic 对 seeds `0`、`1` 和 `2` 进行了重复。

| method | seed 0 acc | seed 1 acc | seed 2 acc |
| --- | ---: | ---: | ---: |
| dense | 1.0000 | 0.7812 | 1.0000 |
| local | 0.3000 | 0.2344 | 0.2781 |
| random | 0.3000 | 0.2344 | 1.0000 |
| zigzag | 1.0000 | 1.0000 | 1.0000 |

Convergence logs 显示 zig-zag 对所有三个 seeds 都在 step 250 前达到 `1.0` validation accuracy。Random 在 seeds `0` 和 `1` 中失败，但在 seed `2` 到 step 750 时解决。Dense 解决了 seeds `0` 和 `2`，但 seed `1` 到 step 1000 尚未完全收敛。

产物：

```text
copy_first_n1024_seed_stability_report.md
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## 推荐的下一个 Gate

在从 synthetic MVP 进入更强 Phase-6 claims 之前：

1. 在 released split 可用之前，将 generated ListOps 与 official LRA 保持分离。
2. 只在清楚说明当前 caveats 的前提下启动 Phase 6。
3. 除非被 true block-sparse backend 替代，否则将 block-pair results 视为 prototype。
4. 在提出 benchmark-level claims 之前，增加更广泛的 task evidence。
