# Copy v0.5 Phase 2 Smoke Report

## 目标

验证 online causal full-copy 数据生成、训练、反向传播、评估、结构化日志和 matplotlib 曲线生成在本地与远端 GPU 上都可运行。该阶段只证明 pipeline 可用，不解释方法优劣。

## 配置

- Config: `configs/copy_v05_smoke.json`
- Task: `copy`, `data=online`, `mode=full_copy`, `num_values=4`
- Special tokens: `PAD=0`, `SEP=5`, `EOS=6`
- Train/Eval source length: `N=128`, `T_raw=258`, `T=272`
- Attention: `causal=true`, `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Methods: `dense`, `local`, `random`, `zigzag`
- Steps: `20`
- Batch size: `4`
- Eval batches: `2`
- Seed: `0`
- Commit recorded in artifacts: `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU: `CUDA_VISIBLE_DEVICES=1`, `NVIDIA A100-SXM4-80GB`
- Remote plotting dependency: project-local `.deps/` with `matplotlib`; `PYTHONPATH` is recorded in `command.sh`.

## 命令

Local:

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_local
```

Remote:

```bash
COPY_V05_GIT_COMMIT=75330e074f67e3f181dc3e4cab2d941eb54dbf2d \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/huiwei/ysx/zigzag_attention/.deps \
/home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_gpu
```

## 结果表

Remote synced artifacts:

- `outputs/copy_v05_smoke_gpu/summary.json`
- `outputs/copy_v05_smoke_gpu/results.csv`
- `outputs/copy_v05_smoke_gpu/metrics.jsonl`
- `outputs/copy_v05_smoke_gpu/command.sh`
- `logs/copy_v05_smoke_gpu_20260612_131435.log`

Remote synced result rows:

| method | backend | final_train_loss | eval_loss | eval_token_accuracy | eval_sequence_accuracy | eval_eos_accuracy | status |
|---|---|---:|---:|---:|---:|---:|---|
| dense | dense_mask | 1.4386 | 1.4418 | 0.2703 | 0.0000 | 0.0000 | ok |
| local | split | 1.4339 | 1.4344 | 0.2665 | 0.0000 | 0.0000 | ok |
| random | split | 1.4362 | 1.4413 | 0.2703 | 0.0000 | 0.0000 | ok |
| zigzag | split | 1.4365 | 1.4418 | 0.2752 | 0.0000 | 0.0000 | ok |

## 通过/失败项

- 本地 smoke: pass
- 远端 GPU smoke: pass
- Four methods forward/backward/eval: pass
- Non-finite loss/metrics: none observed
- `summary.json`, `results.csv`, `results.jsonl`, `metrics.jsonl`, `command.sh`: pass
- Per-run `training_curves.png`: pass, 4/4
- Results/logs synced back to local: pass
- Git commit completed before final runs: pass, `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 解释

Smoke accuracy is near random and is not a convergence claim. The smoke run confirms that online generated train/eval streams, causal full-copy loss positions, token/sequence/EOS metrics, after-causal mask metrics, and direct matplotlib curve generation all work end to end.

