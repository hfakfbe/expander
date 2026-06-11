# 仓库调研

日期：2026-06-10

执行上下文：

- 本地 workspace：`/Users/sxye/Documents/expander`
- 远程服务器：`huiwei`
- 远程项目目录：`/home/huiwei/ysx/zigzag_attention`
- Conda executable：`/home/huiwei/miniconda3/bin/conda`
- 远程环境：`ysx_base`

## 候选摘要

| Priority | Repository / Resource | Status | Notes |
| --- | --- | --- | --- |
| A | `guy-dar/lra-benchmarks` | 选为基础 | PyTorch/HuggingFace 风格，代码面小，`run_model.py`、dataset classes 和 BERT attention path 清晰。远程 clone 和 GPU smoke test 成功。 |
| B | `google-research/long-range-arena` | 仅作参考 | Official LRA repository，但偏 JAX/Flax，不太适合本项目快速修改 attention-module。HEAD 可通过 `git ls-remote` 访问。 |
| C | Hugging Face BigBird | 未来 baseline/reference | 对 BigBird-style sparse baseline 和 block sparse design 有用，但不是第一基础，因为仍需要 task/training framework。 |
| D | `google-research/bigbird` | 仅作参考 | 对 sparse attention structure 有用，但未选作 primary sequence experiment base。 |
| E | `facebookresearch/xformers` | 未来 performance reference | 在 correctness 和 MVP quality experiments 之后有用，尤其用于 optimized attention backends。 |
| F | `hamed1375/Exphormer` | 仅作参考 | 相关的 expander design ideas，但 graph Transformer 语境与 sequence LRA base 不够直接匹配。 |
| G | OpenAI sparse_attention | 仅作参考 | 历史 fixed/strided sparse patterns，未选作基础。 |
| H | Longformer | reference/baseline | 有用的 local/global baseline idea，未选作 primary base。 |

## 已执行检查

### 远程服务器

`ssh huiwei` 成功。初次 GPU query 显示四张 NVIDIA A100-SXM4-80GB devices 可见。之后 GPUs 上出现了其他用户进程，因此未来运行必须在启动 training 前立即重新检查 `nvidia-smi`。

`ysx_base` 通过 `/home/huiwei/miniconda3/bin/conda` 解析，而不是默认 shell PATH。关键 runtime facts：

- Python: 3.10.0
- PyTorch: 2.10.0+cu128
- CUDA visible from PyTorch: yes
- GPU count: 4
- Device name: NVIDIA A100-SXM4-80GB

### `guy-dar/lra-benchmarks`

远程 clone：

```bash
cd /home/huiwei/ysx/zigzag_attention/code
git clone --depth 1 https://github.com/guy-dar/lra-benchmarks.git
cd lra-benchmarks
git rev-parse HEAD
```

Commit：

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

该仓库代码小且可读：

- `run_model.py`：model construction、training loop、task registry。
- `lra_config.py`：task/model configs 和 tokenizers。
- `lra_datasets.py`：ListOps、CIFAR-10、IMDB dataset loaders。
- `train_utils.py`：learning-rate schedules。

Smoke testing 后，基础仓库保持 clean：

```bash
cd /home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
git status --short
```

没有在 base checkout 内修改 source。

## 发现的问题

1. `ysx_base` 不包含 `ml_collections`。

   为避免修改公共环境，smoke test 使用 project-local compatibility shim，位置为：

   ```text
   /home/huiwei/ysx/zigzag_attention/code/smoke_scripts/smoke_support/ml_collections
   ```

2. `fetch_data.py` 的 ListOps branch 有 bug。

   它在设置 `task = args.task` 后索引 `task["lra_release"]["url"]`，因此 `task` 是 string，脚本抛出：

   ```text
   TypeError: string indices must be integers
   ```

3. 文档中的 LRA data URL 无法从服务器可靠使用。

   尝试下载 `https://storage.googleapis.com/long-range-arena/lra_release.gz` 时返回 `403 Forbidden` 或停滞。服务器上的 Hugging Face dataset probing 也超时。因此，phase-1 smoke test 使用 generated tiny ListOps-format TSV 来验证 repository trainability，而不声称 full LRA data 已准备好。

4. 本地机器 GitHub clone 间歇性出现 TLS errors。

   `huiwei` 上的远程 clone 成功，并且是本阶段 authoritative base checkout。

## Smoke Test 结果

Smoke test command：

```bash
cd /home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../smoke_scripts/smoke_support:/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python ../smoke_scripts/base_smoke_listops.py \
  --repo-dir . \
  --steps 100 \
  --rows 256 \
  --batch-size 4 \
  --max-length 128 \
  --hidden-size 64 \
  --layers 2 \
  --heads 4 \
  --output-json ../../outputs/smoke/base_listops_smoke.json \
2>&1 | tee ../../logs/base_listops_smoke.txt
```

结果：

- Status: OK
- Steps: 100
- Device: CUDA
- GPU: NVIDIA A100-SXM4-80GB
- Final loss: 1.0127418041229248
- Mean loss: 1.465013245344162
- Tokens/sec: 18974.068654582854
- Peak allocated memory: 0.0232696533203125 GB
- Peak reserved memory: 0.02734375 GB

产物：

- `smoke_test_log.txt`
- `outputs/base_listops_smoke.json`
- `env_snapshot.yaml`
- `envs/requirements_snapshot.txt`

## Phase-1 决策

`guy-dar/lra-benchmarks` 作为可复用基础仓库满足 phase-1 smoke-test 要求，但有 caveats：

- Full LRA data 尚未准备，因为 upstream data path 当前不可靠。
- Base 需要在私有环境中安装 `ml_collections`，或为 smoke/debug scripts 保留 project-local shim。
- `fetch_data.py` 应在 phase 2 中 patch 或绕过。

下一阶段应准备稳定 datasets 和 task configs，先从 synthetic Associative Recall / Delayed Copy 开始，然后通过可靠 mirror 或 generated-compatible local preprocessing 解决 ListOps。
