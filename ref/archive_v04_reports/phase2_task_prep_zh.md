# Phase 2 任务准备

日期：2026-06-10

## 状态

对于当前 MVP gate，Phase 2 已完成。

已完成：

- 固定了第一个 synthetic task：Associative Recall。
- 添加了可复现的 MVP config：`configs/synthetic_mvp.json`。
- 添加了可运行的 synthetic experiment script：`scripts/synthetic_mvp.py`。
- 为 MVP 实现了 dense、local-only、local + random same-budget 和 local + zig-zag masks。
- 为 `N = 64, 128`、`B = 8, 16`、`d = 2, 3` 实现了 mask correctness checks。
- 验证 dense-mask attention 与 neighbor-list attention 在小规模 cases 上一致。
- 在 `outputs/synthetic_mvp_cpu/` 和 `logs/synthetic_mvp_cpu.log` 下产生了 metrics/log artifacts。
- 使用基于 official LRA ListOps rules 的 project-local generator 准备了 generated ListOps-compatible dataset。
- 验证基础仓库可以通过原始 `ListOpsDataset` 加载 generated ListOps TSV files。

尚未完成：

- 原始下载的 LRA release 仍不可用。文档中的 Google Storage URL 从 `huiwei` 访问失败，Hugging Face probing 也超时。
- LRA Text/Retrieval/AAN 尚未准备。
- 完整 first-grid GPU configs 尚未完成；仅完成 smoke/MVP GPU runs。

## 固定的 Synthetic Task

任务：

```text
Associative Recall
```

当前 MVP generator：

- Sequence 包含 key/value pairs。
- 最后两个位置包含 query marker 和 queried key。
- Target 是与该 key 关联的 value。
- Metric 是 value classes 上的 classification accuracy。

当前任务常量：

```text
num_keys: 64
num_values: 10
query_token: 75
pad_token: 0
```

MVP sequence length：

```text
N = 128
```

这有意小于手册中的第一个 full grid。它仅用于验证 experiment harness。

## Generated ListOps-Compatible Data

由于 upstream `lra_release.gz` URL 无法从 `huiwei` 使用，因此用以下脚本准备了 generated ListOps-compatible split：

```text
scripts/prepare_generated_listops.py
```

远程输出：

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks/datasets/lra_release/listops-1000/
```

本地副本：

```text
code/lra-benchmarks/datasets/lra_release/listops-1000/
```

Split sizes：

```text
basic_train.tsv: 1024 examples
basic_val.tsv:   256 examples
basic_test.tsv:  256 examples
```

生成设置：

```text
max_depth: 10
max_args: 10
min_tokens: 24
max_tokens: 512
seed: 2
mean_tokens: 221.64
```

重要边界：这是从 ListOps grammar 和 label rules 生成的，但不是原始 released LRA split。

## 固定的 Mask / Baseline Methods

| Method | Mask | Raw K for MVP |
| --- | --- | --- |
| dense | Full sequence attention | 128 |
| local | Block-local complete attention | 16 |
| random | Local complete + random nonlocal same-budget edges | 20 |
| zigzag | Local complete + cyclic-G/cycle-H zig-zag edges | 20 |

MVP graph parameters：

```text
B = 16
d = 2
raw K for random/zigzag = B + d^2 = 20
```

## Mask Correctness Tests

当前脚本检查：

- cyclic G 的 Rot_G reverse consistency。
- H degree 等于配置的 `d`。
- 没有 empty attention rows。
- Dense-mask attention 与 neighbor-list attention 产生接近的输出。
- 记录 raw K、effective K、pair count、duplicate-rate estimate 和 self-loop rate。

结果：

```text
mask_tests: ok
cases: 8
max dense-vs-neighbor error: < 5e-7
```

产物：

```text
outputs/synthetic_mvp_cpu/mask_tests.json
```

## 运行命令

因为 GPUs 被占用，使用了 CPU MVP run：

```bash
cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES="" \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python synthetic_mvp.py \
  --methods dense,local,random,zigzag \
  --seq-len 128 \
  --block-size 16 \
  --degree 2 \
  --steps 60 \
  --eval-batches 5 \
  --batch-size 16 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --ffn-dim 128 \
  --log-every 20 \
  --output-dir ../../outputs/synthetic_mvp_cpu
```

## Phase-2 Gate

对于手册中的 strict pipeline，Phase 2 的 MVP 版本现在可通过，因为：

- Synthetic Associative Recall 已固定且可运行。
- Generated ListOps-compatible data 已固定且可运行。
- Base forward/backward/eval 已在 generated ListOps 上用 A100 验证。

剩余 caveat：不要将 generated ListOps 数字报告为 official LRA benchmark results。
