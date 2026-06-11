# Copy v0.5 Phase 3 Main Report

## 目标

在 v0.5 手册定义的 online synthetic copy task 上，完成 `N_train=256/512/1024`、`N_eval=256/512/1024/2048`、`methods=dense/local/random/zigzag`、`seeds=[0,1,2]` 的主实验矩阵。

## 配置

- Config: `configs/copy_v05_main.json`
- Task: `copy`, `data=online`, `num_values=4`
- Train lengths: `256`, `512`, `1024`
- Eval lengths: `256`, `512`, `1024`, `2048`
- Methods: `dense`, `local`, `random`, `zigzag`
- Seeds: `0`, `1`, `2`
- Attention: `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Architecture: `tiny_transformer`, `layers=8`, `d_model=128`, `heads=4`, `ffn_dim=256`, `dropout=0.1`
- Train: `steps=1000`, `batch_size=16`, `eval_batches=20`, `learning_rate=0.001`, `optimizer=adamw`
- Commit recorded in runs: `ea296322d724ce578573f8f0e8e613107a3d090c`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU precheck: GPU 0 `NVIDIA A100-SXM4-80GB`, utilization `0%`, memory `2331/81920 MiB`
- Remote run GPU: `CUDA_VISIBLE_DEVICES=0`
- Run mode: `tmux` session `copy_v05_main`
- Log: `logs/copy_v05_main_20260612_023053.log`

## 命令

```bash
COPY_V05_GIT_COMMIT=ea296322d724ce578573f8f0e8e613107a3d090c \
CUDA_VISIBLE_DEVICES=0 python scripts/synthetic_mvp.py \
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
- Status values: `ok` only
- Non-finite loss/accuracy: none observed
- Every run evaluated at `N_eval=2048`: yes

## 结果表

Mean eval accuracy across seeds. Values are from `phase3_results.csv`.

| N_eval | N_train | dense | local | random | zigzag |
|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 1.000 | 0.239 | 0.572 | 1.000 |
| 256 | 512 | 0.520 | 0.259 | 0.261 | 0.953 |
| 256 | 1024 | 0.746 | 0.253 | 0.319 | 0.983 |
| 512 | 256 | 0.567 | 0.249 | 0.255 | 0.978 |
| 512 | 512 | 1.000 | 0.270 | 0.504 | 1.000 |
| 512 | 1024 | 0.761 | 0.274 | 0.476 | 1.000 |
| 1024 | 256 | 0.674 | 0.237 | 0.236 | 0.964 |
| 1024 | 512 | 0.528 | 0.258 | 0.447 | 0.994 |
| 1024 | 1024 | 1.000 | 0.255 | 0.509 | 1.000 |
| 2048 | 256 | 0.627 | 0.252 | 0.237 | 0.964 |
| 2048 | 512 | 0.512 | 0.261 | 0.263 | 0.926 |
| 2048 | 1024 | 0.830 | 0.242 | 0.249 | 0.999 |

`N_eval=2048` extrapolation mean +/- population std across seeds:

| N_train | dense | local | random | zigzag |
|---:|---:|---:|---:|---:|
| 256 | 0.627 +/- 0.269 | 0.252 +/- 0.006 | 0.237 +/- 0.016 | 0.964 +/- 0.049 |
| 512 | 0.512 +/- 0.297 | 0.261 +/- 0.021 | 0.263 +/- 0.020 | 0.926 +/- 0.090 |
| 1024 | 0.830 +/- 0.147 | 0.242 +/- 0.025 | 0.249 +/- 0.025 | 0.999 +/- 0.001 |

Performance summary:

| method | mean tokens/sec | max peak_reserved_gb |
|---|---:|---:|
| dense | 174181.4 | 5.920 |
| local | 247692.4 | 1.045 |
| random | 206170.7 | 1.805 |
| zigzag | 206510.4 | 1.805 |

## 通过/失败项

- `N_train=256/512/1024`: pass
- `dense/local/random/zigzag`: pass
- Seeds `0/1/2`: pass
- `N_eval=2048` for every run: pass
- Main CSV/JSONL complete: pass
- Results/logs synced back to local: pass
- Git commit completed before run: pass, `ea296322d724ce578573f8f0e8e613107a3d090c`

## 解释

在 copy online synthetic task 上，在当前 `B=16,d=2,G=cyclic,H=cycle` 配置下，`zigzag` 在 `N_eval=2048` 的三组训练长度中均明显高于 `local` 和 same-budget `random`。`local` 在所有 extrapolation 设置中接近随机猜测水平，符合最后 token 无法直接看到 `x0` 的预期。`random` 在部分 train/eval 同长度或较短 eval 下可以学习到信号，但在 `N_eval=2048` 上平均仍接近随机猜测。`dense` 是小 N 质量参考，不是同预算 sparse baseline；它在 extrapolation 上有明显 seed 方差。

以上结论只限于本报告的 online synthetic copy task、当前配置和 seeds，不涉及 official LRA、真实文本任务、大模型 scaling law 或最终 block-sparse kernel 性能。

## 下一步

可在不覆盖本次 config 的前提下新建配置，进一步测试不同 `B/d`、layer-wise graph、或更长 `N_eval` 的稳健性。
