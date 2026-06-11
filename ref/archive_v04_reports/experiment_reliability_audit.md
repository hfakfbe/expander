# Experiment Reliability Audit

Date: 2026-06-11

## Bottom Line

The current experiment state is reliable through the synthetic-MVP Phase-5 gate, but it is still not a complete execution of `ref/zigzag_experiment_execution_manual_v04.md` because official LRA/full experiments have not started.

Reliable enough to use as evidence:

- Phase 1 base-repository selection and smoke test.
- Mask correctness checks for dense-mask and neighbor-list equivalence on small `N`.
- A long-range synthetic `copy_first` convergence diagnostic at `N=256`, where local-only fails and dense/random/zigzag succeed.
- A GPU3 `N=512` neighbor-backend diagnostic where zig-zag solves `copy_first` and local/random do not in 1000 steps.
- A GPU3 `N=512` local/cross split diagnostic where zig-zag keeps 100% accuracy and peak reserved memory drops below dense.
- A GPU3 `N=1024` local/cross split diagnostic where zig-zag again solves `copy_first` with lower peak memory than dense.
- CPU-generated graph caches with neighbor, edge, cross-edge, and slot-aware block-pair metadata for `N=512,B=16,d=2`.
- Repeated GPU3 warmup/measure/profile runs at `N=512` and `N=1024` for split and block-pair backends.
- A three-seed `copy_first N=1024` follow-up where zig-zag solves all seeds by step 250 and local-only fails all seeds.
- Generated-ListOps base smoke/convergence evidence as a pipeline check.

Not reliable enough for final claims:

- Official LRA benchmark quality, because the original LRA release was not available and generated ListOps is not the released split.
- Zig-zag categorical superiority over same-budget random. The `N=1024` three-seed run favors zig-zag on stability and convergence speed, but random solves seed `2`.
- Final block-pair kernel performance. The current block-pair backend validates the sorted edge layout, but it is still a PyTorch scatter prototype.
- Complete ablation, official benchmark, or scaling-law conclusions.

## Phase Status

| Phase | Manual gate | Current status | Reliability verdict |
| --- | --- | --- | --- |
| 1. Repository evaluation | Reusable base repo smoke test and decision artifacts. | Complete: `guy-dar/lra-benchmarks` selected, A100 smoke passed. | Reliable with documented caveats. |
| 2. Task preparation | Synthetic + ListOps-like data load, configs, forward path. | MVP complete: synthetic tasks and generated ListOps load. Official LRA data missing. | Reliable for pipeline; not official LRA. |
| 3. Base training/eval | Dense/local/random baselines and at least one LRA task. | Partially complete: synthetic dense/local/random done; generated ListOps dense base done. | Enough for MVP diagnostics, not final baseline table. |
| 4. New model trial | Mask tests pass; zig-zag trains; no NaN; not worse than local on at least one task. | MVP pass on `copy_first`; three-seed `N=1024` follow-up added; earlier associative-recall run was inconclusive. | Reliable for synthetic long-range gate, not final benchmark result. |
| 5. Performance tuning | True sparse layout, memory/speed protocol. | Complete for synthetic MVP: split/block-pair backends implemented, graph caches generated, and repeated GPU profiles done at N=512/N=1024. | Reliable for synthetic Phase-5 gate; not a final block-sparse kernel claim. |
| 6. Full experiments | Main tables, ablations, seeds. | Not started. | No final quality claims allowed. |
| 7. Scaling law | N/P/K/compute trends. | Not started. | No scaling claims allowed. |

## Key Reliability Findings

### 1. `copy_visible` Is Not a Long-Range Test

`copy_visible` places the source token at `seq_len - 2`, while the classifier reads from the final token representation. With `B=16`, local-only attention can see the source directly. The 100% accuracy in `outputs/copy_visible_convergence/` proves the training loop can converge, but it does not test cross-block communication.

### 2. `copy_first` Is the Useful Synthetic Gate

The synced remote run in `outputs/copy_first_convergence_n256/` is a stronger diagnostic:

| task | method | N | raw_K | effective_K | final valid loss | final valid accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | 256 | 256 | 256.0 | 0.00002081 | 1.0000 |
| copy_first | local | 256 | 16 | 16.0 | 1.388920 | 0.2445 |
| copy_first | random | 256 | 20 | 20.0 | 0.00007693 | 1.0000 |
| copy_first | zigzag | 256 | 20 | 20.0 | 0.00001560 | 1.0000 |

Interpretation:

- Local-only stays near four-class chance, as expected.
- Zig-zag can carry long-range information through 8 layers at same raw budget as random.
- Random also solves the task, so this is not evidence that zig-zag is better than random.

### 3. Associative Recall Is Still Inconclusive

The `phase4_zigzag_trial` associative-recall run reaches roughly random-class entropy for four labels. It shows no NaN and some loss movement, but it does not demonstrate meaningful task learning.

### 4. Memory/Speed Numbers Are Debug-Mode Numbers

Dense, local, random, and zig-zag all report the same peak memory in the synthetic runs because `MaskedSelfAttention` computes dense `N x N` scores before applying the mask. These logs are useful for pair-count bookkeeping and trainability, but they cannot support sparse efficiency claims.

### 5. Neighbor Backend Changes The Correctness Story, Not Yet The Efficiency Story

`scripts/synthetic_mvp.py` now supports `--attention-backend auto`, which uses precomputed neighbor tables for sparse methods. The GPU3 `N=512` diagnostic in `outputs/neighbor_copy_first_n512_gpu3/` shows:

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | neighbor | 512 | 1.393645 | 0.2703 | 3.4473 |
| copy_first | random | neighbor | 512 | 1.393900 | 0.2703 | 6.2598 |
| copy_first | zigzag | neighbor | 512 | 0.000202 | 1.0000 | 6.2598 |

This is useful evidence for trainability and long-range communication, but it is negative evidence for the current naive gather layout as a performance implementation.

### 6. Local/Cross Split Fixes The First Layout Problem

`scripts/synthetic_mvp.py` now supports `--attention-backend auto_split`, which computes block-local dense attention separately from `d^2` cross edges and uses one unified softmax. The clean GPU3 `N=512` run in `outputs/split_copy_first_n512_gpu3_clean/` shows:

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 0.000331 | 1.0000 | 3.4453 |
| copy_first | local | split | 512 | 1.393645 | 0.2703 | 1.0332 |
| copy_first | random | split | 512 | 1.393946 | 0.2719 | 1.7969 |
| copy_first | zigzag | split | 512 | 0.000202 | 1.0000 | 1.7969 |

This is the first evidence that the sparse layout can reduce measured memory while preserving zig-zag's long-range behavior.

### 7. N=1024 Extends The Split Result

The GPU3 `N=1024` run in `outputs/split_copy_first_n1024_gpu3/` shows:

| task | method | backend | N | valid loss | valid accuracy | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 1024 | 0.020978 | 1.0000 | 5.9180 |
| copy_first | local | split | 1024 | 1.377474 | 0.3000 | 1.0488 |
| copy_first | random | split | 1024 | 1.377595 | 0.3000 | 1.8086 |
| copy_first | zigzag | split | 1024 | 0.000280 | 1.0000 | 1.8086 |

This strengthens the current trainability claim from an `N=512` diagnostic to an `N=1024` diagnostic. The repeated performance protocol is summarized next.

### 8. Phase-5 Repeated Profile Is Complete For The Synthetic Gate

The manual's warmup/measure/repeat protocol was run on GPU3 with `warmup=20`, `measure=100`, and `repeats=3`.

At `N=1024`, the strongest result is the split zig-zag backend:

| backend | method | tokens/sec mean | peak reserved GB |
| --- | --- | ---: | ---: |
| dense | dense | 189398.39 | 6.0299 |
| split | zigzag | 332173.36 | 1.8066 |
| blockpair | zigzag | 372037.00 | 2.1113 |

Interpretation:

- Split zig-zag reaches about `1.75x` dense throughput and about `30%` of dense peak reserved memory at `N=1024`.
- The block-pair backend is numerically equivalent after a non-contiguous accumulation fix, and it is faster than split zig-zag at `N=1024` in this run.
- The current block-pair prototype uses more memory than split, so it should be treated as layout evidence rather than a final optimized backend.

### 9. Three-Seed Copy-First Follow-Up

The `copy_first N=1024` split-backend diagnostic was repeated for seeds `0`, `1`, and `2`.

| method | seed 0 acc | seed 1 acc | seed 2 acc |
| --- | ---: | ---: | ---: |
| dense | 1.0000 | 0.7812 | 1.0000 |
| local | 0.3000 | 0.2344 | 0.2781 |
| random | 0.3000 | 0.2344 | 1.0000 |
| zigzag | 1.0000 | 1.0000 | 1.0000 |

The convergence logs show zig-zag reaches `1.0` validation accuracy by step 250 for all three seeds. Random fails seeds `0` and `1`, but solves seed `2` by step 750. Dense solves seeds `0` and `2`, but seed `1` has not fully converged by step 1000.

Artifacts:

```text
copy_first_n1024_seed_stability_report.md
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## Recommended Next Gate

Before moving beyond the synthetic MVP into stronger Phase-6 claims:

1. Keep generated ListOps separate from official LRA until the released split is available.
2. Start Phase 6 only with the current caveats clearly stated.
3. Treat block-pair results as a prototype unless replaced by a true block-sparse backend.
4. Add broader task evidence before making benchmark-level claims.
