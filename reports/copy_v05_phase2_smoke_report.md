# Copy v0.5 Phase 2 Smoke Report

## 目标

验证 online copy 数据生成、训练、反向传播和评估流程在本地与远端 GPU 上均可运行。该阶段只证明 pipeline 可用，不解释方法优劣。

## 配置

- Config: `configs/copy_v05_smoke.json`
- Task: `copy`, `data=online`, `num_values=4`
- Train/Eval length: `N=128`
- Attention: `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Methods: `dense`, `local`, `random`, `zigzag`
- Steps: `20`
- Batch size: `4`
- Eval batches: `2`
- Seed: `0`
- Commit recorded in runs: `75077e0288be387b394f5058237e79c36f8feaff`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU precheck: all 4 A100 GPUs at `0%` utilization; selected GPU 0 with `2331/81920 MiB`
- Remote run GPU: `CUDA_VISIBLE_DEVICES=0`

## 命令

Local:

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_local
```

Remote:

```bash
COPY_V05_GIT_COMMIT=75077e0288be387b394f5058237e79c36f8feaff \
CUDA_VISIBLE_DEVICES=0 python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_gpu
```

## 结果表

Local artifacts:

- `outputs/copy_v05_smoke_local/summary.json`
- `outputs/copy_v05_smoke_local/results.csv`
- `outputs/copy_v05_smoke_local/metrics.jsonl`
- `outputs/copy_v05_smoke_local/command.sh`

Remote synced artifacts:

- `outputs/copy_v05_smoke_gpu/summary.json`
- `outputs/copy_v05_smoke_gpu/results.csv`
- `outputs/copy_v05_smoke_gpu/metrics.jsonl`
- `outputs/copy_v05_smoke_gpu/command.sh`
- `logs/copy_v05_smoke_gpu_20260612_022826.log`

Remote synced result rows:

| method | backend | final_train_loss | eval_loss | eval_accuracy | status | finite |
|---|---|---:|---:|---:|---|---|
| dense | dense_mask | 1.053965 | 1.647702 | 0.0 | ok | yes |
| local | split | 1.054509 | 1.651361 | 0.0 | ok | yes |
| random | split | 1.030897 | 1.649288 | 0.0 | ok | yes |
| zigzag | split | 1.033706 | 1.625570 | 0.0 | ok | yes |

## 通过/失败项

- 本地 smoke: pass
- 远端 GPU smoke: pass
- 4 methods forward/backward/eval: pass
- Non-finite loss/metrics: none observed
- `summary.json`, `results.csv`, `metrics.jsonl`, `command.sh`: pass
- Results/logs synced back to local: pass
- Git commit completed before run: pass, `75077e0288be387b394f5058237e79c36f8feaff`

## 解释

The online copy batches are generated from deterministic seed-derived streams for train and eval. Eval does not reuse train batches. Smoke accuracy is not interpreted as convergence because this run is only 20 steps.

## 下一步

Proceed to Phase 3 main copy experiment with `configs/copy_v05_main.json`.
