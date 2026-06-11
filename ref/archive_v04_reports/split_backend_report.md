# Local/Cross Split Backend Report

Date: 2026-06-10

## Status

`scripts/synthetic_mvp.py` now includes a `split` attention backend and an `auto_split` selector:

- `dense_mask`: dense debug/reference path.
- `neighbor`: previous union-neighbor gather path.
- `split`: block-local dense attention plus cross-neighbor gather, with one unified softmax over local and cross logits.
- `auto_split`: dense uses `dense_mask`; local/random/zigzag use `split`.

This follows the manual's implementation order after neighbor-list attention: local dense and cross sparse are computed separately, but the softmax is over their union.

## Correctness

Mask tests now compare all three small-`N` paths:

- Dense masked attention.
- Union neighbor-list attention.
- Local/cross split attention.

The split-vs-dense max error is recorded as `dense_split_max_error` in mask test artifacts. In the GPU3 clean run, the checked errors are below `1e-5`.

## GPU3 Clean Diagnostic

Command:

```text
outputs/split_copy_first_n512_gpu3_clean/command.sh
```

Results:

| task | method | backend | N | raw_K | cross shape | valid loss | valid accuracy | tokens/sec | peak allocated GB | peak reserved GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copy_first | dense | dense_mask | 512 | 512 | - | 0.000331 | 1.0000 | 293064.22 | 3.2469 | 3.4453 |
| copy_first | local | split | 512 | 16 | `[512, 0]` | 1.393645 | 0.2703 | 445470.03 | 0.9443 | 1.0332 |
| copy_first | random | split | 512 | 20 | `[512, 4]` | 1.393946 | 0.2719 | 373449.26 | 1.6591 | 1.7969 |
| copy_first | zigzag | split | 512 | 20 | `[512, 4]` | 0.000202 | 1.0000 | 373654.43 | 1.6591 | 1.7969 |

Interpretation:

- Zig-zag still solves the long-range `copy_first` task at `N=512`.
- Local-only stays near chance, as expected.
- Same-budget random does not solve this seed/config within 1000 steps.
- Compared with the previous naive neighbor backend, split reduces random/zigzag peak reserved memory from about `6.26GB` to `1.80GB`.
- Compared with dense, split zig-zag uses about `52%` of peak reserved memory and is faster in this run.

## Caveats

This section records the original single training-run timing. It should not be used alone as the final performance protocol.

Also, the split backend still gathers cross edges at token level. Block-pair grouping/sorting has since been implemented as a separate `blockpair` backend, but the current PyTorch prototype is not a final block-sparse kernel.

## Phase-5 Follow-Up

Graph cache generation, a warmup/measure/repeat profiling entry point, an `N=1024` GPU3 diagnostic, and repeated `N=512/N=1024` GPU profiles have been added. See `phase5_performance_tuning_report.md`.
