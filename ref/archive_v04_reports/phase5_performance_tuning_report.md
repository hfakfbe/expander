# Phase 5 Performance Tuning Report

Date: 2026-06-11

## Status

Phase 5 is complete for the current synthetic MVP gate.

Completed:

- Implemented local/cross split attention with unified softmax.
- Implemented a block-pair-indexed cross-edge backend and `auto_blockpair` selector.
- Verified split and block-pair attention are numerically equivalent on CPU and GPU.
- Added graph cache generation for neighbors, edge index, cross edges, and slot-aware block-pair metadata.
- Added and ran the manual's warmup/measure/repeat profiling protocol.
- Ran repeated GPU3 profiling at `N=512` and `N=1024` for `auto_split` and `auto_blockpair`.

Remaining caveat:

- The block-pair backend is a PyTorch prototype using sorted edge lists and scatter. It validates the layout path, but it is not a final block-sparse CUDA or xFormers-style kernel.

## Correctness Checks

The block-pair cache now stores seven columns:

```text
source_block
target_block
source_port
target_port
source_token
target_token
cross_neighbor_slot
```

The `cross_neighbor_slot` column is needed to place each sorted cross edge back into the same unified-softmax slot used by the split backend.

Validation:

- `run_mask_tests` compares dense masked attention, neighbor attention, split attention, and block-pair attention.
- Direct split-vs-blockpair checks pass on CPU with zero error for tested cases.
- Direct model-level split-vs-blockpair checks pass on GPU with max error about `1.2e-7`.

An earlier block-pair profile exposed a non-contiguous tensor bug in `cross_out` accumulation. That invalid run was isolated at:

```text
outputs/profile_blockpair_n512_gpu3_invalid_noncontiguous_bug_20260611/
```

It is retained only as a debugging record and is not used for conclusions.

## Graph Cache Artifacts

Generated on CPU with no GPU use:

```text
cached_graphs/copy_first_n512_B16_d2_seed0_local/
cached_graphs/copy_first_n512_B16_d2_seed0_random/
cached_graphs/copy_first_n512_B16_d2_seed0_zigzag/
```

Command:

```text
cached_graphs/copy_first_n512_B16_d2_seed0_command.sh
```

Summary:

| method | N | B | d | cross pairs | neighbor shape | cross neighbor shape | block-pair index shape | block-pair count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local | 512 | 16 | 2 | 0 | `[512, 16]` | `[512, 0]` | `[0, 7]` | 0 |
| random | 512 | 16 | 2 | 2048 | `[512, 20]` | `[512, 4]` | `[2048, 7]` | 861 |
| zigzag | 512 | 16 | 2 | 2048 | `[512, 20]` | `[512, 4]` | `[2048, 7]` | 512 |

Interpretation:

- Random spreads the same number of cross edges over more block pairs.
- Zig-zag has more structured block-pair reuse, which is useful for future block-sparse kernels.

## Profiling Protocol

All formal profiles below used:

```text
warmup steps: 20
measure steps: 100
repeats: 3
GPU: CUDA_VISIBLE_DEVICES=3
```

Each run calls `torch.cuda.synchronize()`, resets peak memory stats before the measured section, and writes per-run plus aggregate CSV/JSON artifacts.

Artifacts:

```text
outputs/profile_split_n512_gpu3/
outputs/profile_blockpair_n512_gpu3/
outputs/profile_split_n1024_gpu3/
outputs/profile_blockpair_n1024_gpu3/
```

GPU policy followed for these runs: GPU3 was checked before each formal profile and only used when utilization was below `10%`.

## Formal Profile Results

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

## Interpretation

The strongest Phase-5 result is the split backend at `N=1024`:

- Split zig-zag is about `1.75x` dense throughput.
- Split zig-zag uses about `30%` of dense peak reserved memory.
- Earlier `N=1024` trainability diagnostics show split zig-zag solves `copy_first`, while local and the same-budget random seed/config remain near chance.

The block-pair prototype validates the sorted edge layout and remains numerically equivalent after the non-contiguous accumulation fix. However:

- At `N=512`, block-pair zig-zag is about `0.98x` split zig-zag throughput and uses about `1.17x` split peak reserved memory.
- At `N=1024`, block-pair zig-zag is about `1.12x` split zig-zag throughput but still uses about `1.17x` split peak reserved memory.

So the current block-pair PyTorch prototype is useful layout evidence, not a final memory optimization. A real block-sparse kernel would be needed before making stronger block-pair performance claims.

## Phase-5 Gate

For the synthetic MVP gate, Phase 5 passes on the split backend: at `N=1024`, it gives a clear memory reduction and higher throughput than dense while preserving the zig-zag long-range behavior shown by the diagnostic training run.

Phase 6 should still be scoped carefully: the official LRA data caveat remains, and block-pair performance should not be advertised as final until backed by a true block-sparse implementation.
