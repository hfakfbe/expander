# Neighbor Backend Report

Date: 2026-06-10

## Status

A real neighbor-list attention backend has been added to `scripts/synthetic_mvp.py`.

Available backends:

- `dense_mask`: original debug path, materializes dense `N x N` scores and applies the mask.
- `neighbor`: precomputes `[N, K]` neighbor tables and computes attention over gathered keys/values.
- `auto`: uses `dense_mask` for `dense`, and `neighbor` for `local`, `random`, and `zigzag`.

This satisfies the manual's next implementation step at a functional level: sparse methods no longer need the dense score mask path during training. It does not yet satisfy the manual's performance-tuning goal, because the current gather implementation still creates large intermediate tensors and has poor memory behavior at `N=512`.

## Verification

Smoke run:

```text
outputs/neighbor_backend_smoke/
```

Result:

- Mask tests passed.
- `auto` backend completed forward, backward, eval, and metric logging.
- Sparse methods recorded `neighbor_shape`, for example `[128, 20]` for random/zigzag.

## GPU3 `N=512` Diagnostic

The main diagnostic was run on the last A100 as requested:

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

Interpretation:

- Dense confirms the task is learnable.
- Local-only stays near four-class chance, as expected for a long-range copy task.
- Zig-zag solves the task at `N=512` with `K=20`.
- Random same-budget does not solve this seed/config within 1000 steps, unlike the earlier `N=256` run.
- The current neighbor backend is not memory efficient: random/zigzag peak reserved memory is higher than dense. This is likely from advanced-index gather intermediates, so the next implementation step should avoid batch/head-expanded gathers and move toward local/cross split or block-pair batching.

## Artifacts

- `outputs/neighbor_copy_first_n512_gpu3/summary.json`
- `outputs/neighbor_copy_first_n512_gpu3/results.csv`
- `outputs/neighbor_copy_first_n512_gpu3/*_metrics.jsonl`
- `outputs/neighbor_copy_first_n512_gpu3/command.sh`

## Follow-Up

The next step in this report has now been implemented as the `split` / `auto_split` backend. See `split_backend_report.md` for the GPU3 clean diagnostic.
