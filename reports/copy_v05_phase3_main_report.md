# Copy v0.5 Phase 3 Main Report

## 目标

在 v0.5 手册定义的 online synthetic causal full-copy task 上，完成 `N_train=256/512/1024`、`N_eval=256/512/1024/2048`、`methods=dense/local/random/zigzag`、`seeds=[0,1,2]` 的主实验矩阵。

## 配置

- Config: `configs/copy_v05_main.json`
- Task: `copy`, `data=online`, `mode=full_copy`, `num_values=4`
- Special tokens: `PAD=0`, `SEP=5`, `EOS=6`
- Train source lengths: `256`, `512`, `1024`
- Eval source lengths: `256`, `512`, `1024`, `2048`
- Methods: `dense`, `local`, `random`, `zigzag`
- Seeds: `0`, `1`, `2`
- Attention: `causal=true`, `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Architecture: `transformer`, `layers=8`, `d_model=128`, `heads=4`, `ffn_dim=256`, `dropout=0.1`
- Train: `steps=1000`, `batch_size=16`, `eval_batches=20`, `learning_rate=0.001`, `optimizer=adamw`, `log_every=250`
- Commit recorded in artifacts: `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU: `CUDA_VISIBLE_DEVICES=1`, `NVIDIA A100-SXM4-80GB`
- Run mode: `tmux` session `copy_v05_main_753`
- Log: `logs/copy_v05_main_20260612_131510.log`
- Remote plotting dependency: project-local `.deps/` with `matplotlib`; `PYTHONPATH` is recorded in `command.sh`.

## 命令

```bash
COPY_V05_GIT_COMMIT=75330e074f67e3f181dc3e4cab2d941eb54dbf2d \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/huiwei/ysx/zigzag_attention/.deps \
/home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/synthetic_mvp.py \
  --config configs/copy_v05_main.json \
  --output-dir outputs/copy_v05_main
```

## 结果文件

- `outputs/copy_v05_main/phase3_results.csv`
- `outputs/copy_v05_main/phase3_results.jsonl`
- `outputs/copy_v05_main/results.csv`
- `outputs/copy_v05_main/results.jsonl`
- `outputs/copy_v05_main/summary.json`
- `outputs/copy_v05_main/metrics.jsonl`
- 36 run directories under `outputs/copy_v05_main/train_N*_seed*_*`

Completeness checks:

- Eval rows: `144`
- Training runs: `36`
- Run directories with required files: `36/36`
- Training curve images: `36/36`
- Status values: `ok` only
- Non-finite loss/accuracy: none observed
- Every run evaluated at `N_eval=2048`: yes

## 结果表

Mean eval token accuracy across seeds. Values are from `phase3_results.csv`.

| N_eval | N_train | dense | local | random | zigzag |
|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 0.997 | 0.253 | 0.367 | 0.253 |
| 256 | 512 | 0.278 | 0.250 | 0.251 | 0.250 |
| 256 | 1024 | 0.265 | 0.250 | 0.249 | 0.250 |
| 512 | 256 | 0.252 | 0.248 | 0.250 | 0.248 |
| 512 | 512 | 0.281 | 0.250 | 0.302 | 0.250 |
| 512 | 1024 | 0.259 | 0.250 | 0.250 | 0.250 |
| 1024 | 256 | 0.251 | 0.249 | 0.249 | 0.249 |
| 1024 | 512 | 0.261 | 0.250 | 0.250 | 0.249 |
| 1024 | 1024 | 0.257 | 0.250 | 0.250 | 0.250 |
| 2048 | 256 | 0.251 | 0.250 | 0.250 | 0.250 |
| 2048 | 512 | 0.257 | 0.250 | 0.249 | 0.250 |
| 2048 | 1024 | 0.253 | 0.250 | 0.250 | 0.250 |

`N_eval=2048` extrapolation token accuracy mean +/- population std across seeds:

| N_train | dense | local | random | zigzag |
|---:|---:|---:|---:|---:|
| 256 | 0.2508 +/- 0.0003 | 0.2500 +/- 0.0003 | 0.2498 +/- 0.0002 | 0.2500 +/- 0.0003 |
| 512 | 0.2569 +/- 0.0010 | 0.2498 +/- 0.0004 | 0.2495 +/- 0.0005 | 0.2497 +/- 0.0005 |
| 1024 | 0.2531 +/- 0.0009 | 0.2496 +/- 0.0004 | 0.2498 +/- 0.0005 | 0.2497 +/- 0.0005 |

Mean sequence accuracy across seeds:

| Setting | dense | local | random | zigzag |
|---|---:|---:|---:|---:|
| N_train=256, N_eval=256 | 0.327 | 0.000 | 0.000 | 0.000 |
| All other train/eval settings | 0.000 | 0.000 | 0.000 | 0.000 |

Mean EOS accuracy at `N_eval=2048` is `0.000` for all methods and all training lengths.

Performance summary:

| method | mean tokens/sec | max peak_reserved_gb |
|---|---:|---:|
| dense | 147796.8 | 22.129 |
| local | 405908.5 | 2.244 |
| random | 279887.7 | 3.699 |
| zigzag | 291779.0 | 3.699 |

## 通过/失败项

- `N_train=256/512/1024`: pass
- `dense/local/random/zigzag`: pass
- Seeds `0/1/2`: pass
- `N_eval=2048` for every run: pass
- Main CSV/JSONL complete: pass
- Per-run `training_curves.png`: pass, 36/36
- Results/logs synced back to local: pass
- Git commit completed before final runs: pass, `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 解释

在 causal full-copy task 上，在当前 `B=16,d=2,G=cyclic,H=cycle` 配置和 seeds `[0,1,2]` 内，结果不支持“zigzag 优于 local 或 same-budget random”的结论。除 `dense` 在最短同长度设置 `N_train=N_eval=256` 上能学到完整复制外，其余方法和多数更长设置的 token accuracy 基本接近随机水平 `0.25`，sequence accuracy 为 `0`。

这与旧的 copy 分类实验结果不同；旧结果来自已删除的错误任务定义，不能作为 v0.5 causal full-copy 证据。以上结论只限于本报告的 online synthetic causal full-copy task、当前配置和 seeds，不涉及 official LRA、真实文本任务、大模型 scaling law 或最终 block-sparse kernel 性能。

## 下一步

若继续研究 zig-zag 是否能帮助 full-copy，需要新建 config，优先测试更长训练步数、更强 positional/copy inductive bias、不同 `B/d`、或更适合 full-copy 的 curriculum；不得覆盖本次 config 或复用旧错误结果。

