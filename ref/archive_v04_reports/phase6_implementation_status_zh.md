# Phase 6 Strict Implementation 状态

日期：2026-06-11

## 已实现

- 将完整 `guy-dar/lra-benchmarks` 源码恢复到 `code/lra-benchmarks/`，同时保留现有 generated ListOps data。
- 保留选定的 base commit reference：

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

- 修复 `code/lra-benchmarks/fetch_data.py`，使其 dataset table 按 selected task 索引，下载在 HTTP errors 时失败，解压使用 requested destination directory，并且 ListOps branch 写入 `lra_release.gz`。
- 添加 strict data readiness tooling：

```text
scripts/phase6_data_readiness.py
outputs/phase6_data_readiness/readiness.json
```

- 添加 Phase-6 runner 和 result table writer：

```text
scripts/phase6_runner.py
configs/phase6_strict_plan.json
outputs/phase6_runner_smoke/phase6_results.csv
outputs/phase6_runner_smoke/phase6_results.jsonl
```

- 添加 base BERT attention replacement smoke tooling：

```text
scripts/base_attention_smoke.py
scripts/phase6_static_checks.py
outputs/phase6_static_checks.json
```

- 扩展 `scripts/listops_base_smoke.py`，加入 `--task listops|imdb`，因此同一个 base smoke path 可以在 data ready 后验证 official ListOps 和 LRA Text/IMDB。

## 当前 Data Gate

当前 gate status 按 strict manual 要求处于 blocked。

Generated ListOps 存在，但仍仅限 pipeline-only：

```text
basic_train.tsv rows: 1024
basic_val.tsv rows: 256
basic_test.tsv rows: 256
source_scope: pipeline_only_generated_or_incomplete
ready_for_official_claims: false
```

Official ListOps download attempt：

```text
https://storage.googleapis.com/long-range-arena/lra_release.gz
curl: (56) The requested URL returned error: 403
```

IMDB/Text 本地尚未就绪：

```text
datasets/aclImdb/train: missing
datasets/aclImdb/test: missing
```

检测到并拒绝了一个不完整的 local IMDB archive：

```text
datasets/_archives/aclImdb_v1.tar.gz
gzip_error: Compressed file ended before the end-of-stream marker was reached
```

## Verification Run

通过：

```bash
python -m py_compile \
  scripts/phase6_data_readiness.py \
  scripts/phase6_runner.py \
  scripts/base_attention_smoke.py \
  scripts/phase6_static_checks.py \
  scripts/listops_base_smoke.py \
  code/lra-benchmarks/fetch_data.py
```

通过：

```bash
python scripts/phase6_static_checks.py
```

通过并正确阻止 official claims：

```bash
python scripts/phase6_data_readiness.py \
  --repo-dir code/lra-benchmarks \
  --output-json outputs/phase6_data_readiness/readiness.json
```

通过 synthetic Phase-6 table smoke：

```bash
python scripts/phase6_runner.py \
  --task copy_first \
  --methods dense,local \
  --seq-len 64 \
  --block-size 16 \
  --degree 2 \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/phase6_runner_smoke
```

Smoke output 使用所需 Phase-6 schema fields。

## 剩余 Strict-Mode Blockers

在满足以下条件前，不得启动 Phase 6 official experiments：

1. Official ListOps data 从 verified source 获取，并通过 integrity、row-count、label 和 max-length checks。
2. IMDB/Text data 完整下载并解压到 `code/lra-benchmarks/datasets/aclImdb/` 下。
3. Dense base smoke 对 `listops` 和 `imdb` 都通过。
4. `scripts/base_attention_smoke.py` 在具备 `pandas`、`torch` 和 `transformers` 的 server environment 中通过。

同步此 workspace 后推荐的远程命令：

```bash
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base

python scripts/phase6_data_readiness.py \
  --repo-dir code/lra-benchmarks \
  --output-json outputs/phase6_data_readiness/readiness_download_attempt.json \
  --download \
  --timeout 120

python scripts/listops_base_smoke.py \
  --repo-dir code/lra-benchmarks \
  --task imdb \
  --steps 10 \
  --batch-size 2 \
  --max-length 128 \
  --hidden-size 64 \
  --layers 1 \
  --heads 4 \
  --output-json outputs/phase6_imdb_base_smoke/summary.json

python scripts/base_attention_smoke.py \
  --repo-dir code/lra-benchmarks \
  --task listops \
  --method zigzag \
  --backend split \
  --seq-len 64 \
  --block-size 16 \
  --degree 2 \
  --hidden-size 64 \
  --layers 1 \
  --heads 4 \
  --batch-size 2 \
  --output-json outputs/base_attention_smoke/listops_zigzag_split.json
```
