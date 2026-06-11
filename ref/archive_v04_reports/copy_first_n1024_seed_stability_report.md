# Copy-First N=1024 Seed Stability Report

Date: 2026-06-11

## Purpose

This report records the seed-stability follow-up requested before moving beyond the synthetic MVP gate. It uses the same `copy_first`, `N=1024`, `B=16`, `d=2`, `auto_split` setup as the Phase-5 diagnostic, with seeds `0`, `1`, and `2`.

GPU policy followed: GPU3 was checked before each new run and used only when utilization was below `10%`.

## Artifacts

```text
outputs/split_copy_first_n1024_gpu3/
outputs/split_copy_first_n1024_seed1_gpu3/
outputs/split_copy_first_n1024_seed2_gpu3/
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## Final Results

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

## Convergence Timing

Validation accuracy at logged steps:

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

## Interpretation

Reliable conclusions:

- Zig-zag solves `copy_first` in all three seeds and reaches `1.0` validation accuracy by step 250 every time.
- Local-only fails in all three seeds, as expected for this long-range gate.
- Dense solves seeds `0` and `2`, but seed `1` is only at `0.7812` accuracy by 1000 steps.
- Same-budget random is seed-sensitive: it fails in seeds `0` and `1`, but solves seed `2` by step 750.

Boundary:

- These results strengthen the synthetic long-range gate evidence for zig-zag stability.
- They do not prove zig-zag is categorically superior to same-budget random, because random solves seed `2`.
- They do show zig-zag's convergence is earlier and more stable than dense/random in this specific `copy_first N=1024` setup.
