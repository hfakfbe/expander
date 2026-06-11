# Phase 6 Readiness 报告

日期：2026-06-11

## 状态

Phase 6 尚未开始。

截至 Phase 5 的 synthetic MVP gates 现已完成，并且新增了三 seed `copy_first N=1024` follow-up。不过，手册的 Phase-6 main experiment 要求 synthetic data 之外的稳定任务，包括 LRA ListOps 和一个 text/retrieval-style task。这些 official benchmark inputs 当前尚未就绪。

## Synthetic Gate Readiness

已就绪：

- `copy_first N=1024` 有 seed `0/1/2` 结果。
- Zig-zag 在 step 250 前解决全部三个 seeds。
- Local-only 在全部三个 seeds 中失败。
- Same-budget random 对 seed 敏感：它在 seeds `0/1` 中失败，并解决 seed `2`。
- Dense 在 1000-step budget 下也对 seed 敏感：seeds `0/2` 解决，seed `1` 达到 `0.7812`。

产物：

```text
copy_first_n1024_seed_stability_report.md
outputs/split_copy_first_n1024_seeds_gpu3_summary.csv
```

## LRA / ListOps Readiness

尚未准备好用于 official benchmark claims。

远程检查：

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks/datasets/lra_release.gz
```

该文件存在但为空/损坏：

```text
file: empty
gzip -t: unexpected end of file
```

当前唯一可用的 ListOps files 是 generated-compatible local splits：

```text
datasets/lra_release/listops-1000/basic_train.tsv
datasets/lra_release/listops-1000/basic_val.tsv
datasets/lra_release/listops-1000/basic_test.tsv
```

这些文件从 ListOps grammar 生成，对 pipeline testing 有用，但不是 released LRA split。

Official source probe：

```text
https://storage.googleapis.com/long-range-arena/lra_release.gz
```

远程服务器在 30 秒 `curl -I` probe 内未收到 headers，ranged GET 也停滞且没有收到 bytes。这与早先的 data-access caveat 一致。

## Phase-6 Gate Decision

现在不要启动完整 Phase-6 main table。

手册允许的下一步：

1. 继续 official LRA ListOps 和一个 text/retrieval task 的 data-readiness 工作。
2. 如果 official data 仍不可访问，清楚标注 generated ListOps 为 pipeline-only，并且不要将其与 official benchmark tables 混合。
3. 准备 Phase-6 configs 和 result schemas，但不运行 full benchmark claims。
4. 可选地运行 generated-ListOps local/random/zig-zag smoke experiments，仅作为 engineering checks。

已准备的 schema：

```text
configs/phase6_schema.json
```

该 schema 记录 method list、task readiness state、required ablations、result fields，以及保持 generated ListOps 与 official LRA results 分离的 guardrails。

尚不允许：

- 将 generated ListOps 报告为 official LRA。
- 启动 scaling-law experiments。
- 声称完整 Phase-6 benchmark results。
