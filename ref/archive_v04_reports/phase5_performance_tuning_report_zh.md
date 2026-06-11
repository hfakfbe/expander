# Phase 5 Performance Tuning 报告

日期：2026-06-11

## 状态

对于当前 synthetic MVP gate，Phase 5 已完成。

已完成：

- 实现了带 unified softmax 的 local/cross split attention。
- 实现了 block-pair-indexed cross-edge backend 和 `auto_blockpair` selector。
- 验证 split 和 block-pair attention 在 CPU 和 GPU 上数值等价。
- 添加了 graph cache generation，包含 neighbors、edge index、cross edges 和 slot-aware block-pair metadata。
- 添加并运行了手册中的 warmup/measure/repeat profiling protocol。
- 在 `N=512` 和 `N=1024` 上为 `auto_split` 和 `auto_blockpair` 运行了重复 GPU3 profiling。

剩余 caveat：

- Block-pair backend 是一个使用 sorted edge lists 和 scatter 的 PyTorch prototype。它验证了 layout path，但不是最终的 block-sparse CUDA 或 xFormers-style kernel。

## Correctness Checks

Block-pair cache 现在存储七列：

```text
source_block
target_block
source_port
target_port
source_token
target_token
cross_neighbor_slot
```

`cross_neighbor_slot` 列用于将每条 sorted cross edge 放回 split backend 所使用的同一个 unified-softmax slot。

验证：

- `run_mask_tests` 比较 dense masked attention、neighbor attention、split attention 和 block-pair attention。
- Direct split-vs-blockpair checks 在 CPU 上通过，测试 cases 的误差为 zero。
- Direct model-level split-vs-blockpair checks 在 GPU 上通过，max error 约 `1.2e-7`。

早先一次 block-pair profile 暴露了 `cross_out` accumulation 中的 non-contiguous tensor bug。该无效运行已隔离在：

```text
outputs/profile_blockpair_n512_gpu3_invalid_noncontiguous_bug_20260611/
```

它仅作为 debugging record 保留，不用于结论。

## Graph Cache Artifacts

在 CPU 上生成，未使用 GPU：

```text
cached_graphs/copy_first_n512_B16_d2_seed0_local/
cached_graphs/copy_first_n512_B16_d2_seed0_random/
cached_graphs/copy_first_n512_B16_d2_seed0_zigzag/
```

命令：

```text
cached_graphs/copy_first_n512_B16_d2_seed0_command.sh
```

摘要：

| method | N | B | d | cross pairs | neighbor shape | cross neighbor shape | block-pair index shape | block-pair count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local | 512 | 16 | 2 | 0 | `[512, 16]` | `[512, 0]` | `[0, 7]` | 0 |
| random | 512 | 16 | 2 | 2048 | `[512, 20]` | `[512, 4]` | `[2048, 7]` | 861 |
| zigzag | 512 | 16 | 2 | 2048 | `[512, 20]` | `[512, 4]` | `[2048, 7]` | 512 |

解读：

- Random 将相同数量的 cross edges 分散到更多 block pairs 上。
- Zig-zag 有更结构化的 block-pair reuse，这对未来 block-sparse kernels 有用。

## Profiling Protocol

下面所有正式 profiles 都使用：

```text
warmup steps: 20
measure steps: 100
repeats: 3
GPU: CUDA_VISIBLE_DEVICES=3
```

每次运行都会调用 `torch.cuda.synchronize()`，在 measured section 前重置 peak memory stats，并写入 per-run 与 aggregate CSV/JSON artifacts。

产物：

```text
outputs/profile_split_n512_gpu3/
outputs/profile_blockpair_n512_gpu3/
outputs/profile_split_n1024_gpu3/
outputs/profile_blockpair_n1024_gpu3/
```

这些运行遵循的 GPU policy：每次正式 profile 前检查 GPU3，并且仅在利用率低于 `10%` 时使用。

## 正式 Profile 结果

### N=512, Batch Size 32

| backend | method | tokens/sec mean | tokens/sec std | peak reserved GB |
| --- | --- | ---: | ---: | ---: |
| split | dense | 317736.82 | 1383.64 | 3.4531 |
| split | local | 441534.28 | 3240.55 | 1.0332 |
| split | random | 339933.18 | 36953.06 | 1.7969 |
| split | zigzag | 340511.29 | 33979.81 | 1.7969 |
| blockpair | dense | 314642.16 | 1281.71 | 3.4531 |
| blockpair | local | 417181.03 | 2317.77 | 1.0332 |
| blockpair | random | 332378.92 | 4315.21 | 2.1094 |
| blockpair | zigzag | 333566.75 | 2060.32 | 2.1094 |

### N=1024, Batch Size 16

| backend | method | tokens/sec mean | tokens/sec std | peak reserved GB |
| --- | --- | ---: | ---: | ---: |
| split | dense | 189398.39 | 778.05 | 6.0299 |
| split | local | 441810.44 | 3334.67 | 1.0469 |
| split | random | 348983.54 | 31375.10 | 1.8066 |
| split | zigzag | 332173.36 | 27272.90 | 1.8066 |
| blockpair | dense | 188838.02 | 648.27 | 6.0299 |
| blockpair | local | 442611.19 | 4109.30 | 1.0469 |
| blockpair | random | 341435.94 | 3030.14 | 2.1113 |
| blockpair | zigzag | 372037.00 | 29012.72 | 2.1113 |

## 解读

最强的 Phase-5 结果是 `N=1024` 下的 split backend：

- Split zig-zag 的 throughput 约为 dense 的 `1.75x`。
- Split zig-zag 使用约 `30%` 的 dense peak reserved memory。
- 早先的 `N=1024` trainability diagnostics 显示 split zig-zag 解决了 `copy_first`，而 local 和 same-budget random seed/config 保持在 chance 附近。

Block-pair prototype 验证了 sorted edge layout，并且在 non-contiguous accumulation fix 后保持数值等价。不过：

- 在 `N=512`，block-pair zig-zag 的 throughput 约为 split zig-zag 的 `0.98x`，并使用约 `1.17x` 的 split peak reserved memory。
- 在 `N=1024`，block-pair zig-zag 的 throughput 约为 split zig-zag 的 `1.12x`，但仍使用约 `1.17x` 的 split peak reserved memory。

因此，当前 block-pair PyTorch prototype 是有用的 layout evidence，而不是最终 memory optimization。在提出更强 block-pair performance claims 之前，需要真正的 block-sparse kernel。

## Phase-5 Gate

对于 synthetic MVP gate，Phase 5 在 split backend 上通过：在 `N=1024`，它相比 dense 给出了明确的 memory reduction 和更高 throughput，同时保留了 diagnostic training run 所显示的 zig-zag long-range behavior。

Phase 6 仍应谨慎限定范围：official LRA data caveat 仍然存在，并且在有 true block-sparse implementation 支撑前，不应将 block-pair performance 宣传为最终结果。
