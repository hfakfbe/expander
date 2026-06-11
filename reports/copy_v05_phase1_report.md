# Copy v0.5 Phase 1 Report

## 目标

在 v0.5 手册约束下完成配置化与 G/H 图结构解耦，并验证 `--config` 能驱动 online copy smoke run。

## 配置

- Config: `configs/copy_v05_smoke.json`
- Task: `copy`, `data=online`, `num_values=4`
- Train/Eval length: `N=128`
- Attention: `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Methods: `dense`, `local`, `random`, `zigzag`
- Smoke overrides: `steps=2`, `batch_size=2`, `eval_batches=1`
- Implementation commit: `c8e5a8b019e0586caa506ec3eb0fbe09ea197e25`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU precheck: GPU 0 `NVIDIA A100-SXM4-80GB`, utilization `0%`, memory `2331/81920 MiB`
- Remote run GPU: `CUDA_VISIBLE_DEVICES=0`

## 命令

Local:

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --methods dense,local,random,zigzag \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/copy_v05_phase1_smoke
```

Remote:

```bash
COPY_V05_GIT_COMMIT=c8e5a8b019e0586caa506ec3eb0fbe09ea197e25 \
CUDA_VISIBLE_DEVICES=0 python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --methods dense,local,random,zigzag \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/copy_v05_phase1_smoke
```

## 结果表

Artifacts:

- `outputs/copy_v05_phase1_smoke/summary.json`
- `outputs/copy_v05_phase1_smoke/results.csv`
- `outputs/copy_v05_phase1_smoke/results.jsonl`
- `outputs/copy_v05_phase1_smoke/metrics.jsonl`
- `outputs/copy_v05_phase1_smoke/mask_tests.json`
- `logs/copy_v05_phase1_smoke_20260612_022656.log`

Remote synced result rows:

| method | backend | raw_K | effective_K_mean | attention_pair_count | eval_accuracy | status |
|---|---|---:|---:|---:|---:|---|
| dense | dense_mask | 128 | 128.0 | 16384 | 0.5 | ok |
| local | split | 16 | 16.0 | 2048 | 0.5 | ok |
| random | split | 20 | 20.0 | 2560 | 0.5 | ok |
| zigzag | split | 20 | 20.0 | 2560 | 0.5 | ok |

## 通过/失败项

- `configs/copy_v05_smoke.json` exists: pass
- `configs/copy_v05_main.json` exists: pass
- `scripts/graph_structures.py` exists: pass
- `synthetic_mvp.py --config` drives smoke run: pass
- 4 methods complete forward/backward/eval: pass
- Mask tests: pass, 8 cases
- Output contains command and config snapshot: pass
- Remote result synced back to local: pass
- Git commit completed: pass, `c8e5a8b019e0586caa506ec3eb0fbe09ea197e25`

## 解释

Phase 1 only validates configurability, graph decoupling, artifact recording, and basic execution. The smoke accuracy values are not convergence claims and are not used as main experiment evidence.

## 下一步

Proceed to Phase 2 online copy smoke with the committed v0.5 config path and remote GPU run protocol.
