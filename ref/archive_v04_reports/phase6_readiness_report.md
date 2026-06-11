# Phase 6 Readiness Report

Date: 2026-06-11

## Status

Phase 6 is not started.

The synthetic MVP gates through Phase 5 are now complete, and a three-seed `copy_first N=1024` follow-up has been added. However, the manual's Phase-6 main experiment requires stable tasks beyond synthetic data, including LRA ListOps and one text/retrieval-style task. Those official benchmark inputs are not currently ready.

## Synthetic Gate Readiness

Ready:

- `copy_first N=1024` has seed `0/1/2` results.
- Zig-zag solves all three seeds by step 250.
- Local-only fails all three seeds.
- Same-budget random is seed-sensitive: it fails seeds `0/1` and solves seed `2`.
- Dense is also seed-sensitive under the 1000-step budget: seeds `0/2` solve, seed `1` reaches `0.7812`.

Artifacts:

```text
copy_first_n1024_seed_stability_report.md
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## LRA / ListOps Readiness

Not ready for official benchmark claims.

Remote check:

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks/datasets/lra_release.gz
```

This file exists but is empty/corrupt:

```text
file: empty
gzip -t: unexpected end of file
```

The only usable ListOps files currently present are generated-compatible local splits:

```text
datasets/lra_release/listops-1000/basic_train.tsv
datasets/lra_release/listops-1000/basic_val.tsv
datasets/lra_release/listops-1000/basic_test.tsv
```

These are generated from the ListOps grammar and useful for pipeline testing, but they are not the released LRA split.

Official source probe:

```text
https://storage.googleapis.com/long-range-arena/lra_release.gz
```

The remote server did not receive headers within a 30-second `curl -I` probe, and a ranged GET also stalled without receiving bytes. This matches the earlier data-access caveat.

## Phase-6 Gate Decision

Do not start the full Phase-6 main table yet.

Allowed next steps under the manual:

1. Continue data-readiness work for official LRA ListOps and one text/retrieval task.
2. If official data remains inaccessible, clearly label generated ListOps as pipeline-only and do not mix it with official benchmark tables.
3. Prepare Phase-6 configs and result schemas without running full benchmark claims.
4. Optionally run generated-ListOps local/random/zig-zag smoke experiments as engineering checks only.

Prepared schema:

```text
configs/phase6_schema.json
```

This schema records the method list, task readiness state, required ablations, result fields, and guardrails for keeping generated ListOps separate from official LRA results.

Not allowed yet:

- Reporting generated ListOps as official LRA.
- Starting scaling-law experiments.
- Claiming complete Phase-6 benchmark results.
