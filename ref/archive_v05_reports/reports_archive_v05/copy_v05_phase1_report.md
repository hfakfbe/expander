# Copy v0.5 Phase 1 Report

## 目标

在 v0.5 手册约束下完成配置化、G/H 图结构解耦、causal full-copy 任务定义，并验证 `--config` 能驱动四种 attention 方法的最小 smoke run。

## 配置

- Config: `configs/copy_v05_smoke.json`
- Task: `copy`, `data=online`, `mode=full_copy`, `num_values=4`
- Special tokens: `PAD=0`, `SEP=5`, `EOS=6`
- Train/Eval source length: `N=128`, `T_raw=258`, `T=272`
- Attention: `causal=true`, `B=16`, `d=2`, `G.type=cyclic`, `H.type=cycle`
- Methods: `dense`, `local`, `random`, `zigzag`
- Phase 1 overrides: `steps=2`, `batch_size=2`, `eval_batches=1`
- Implementation commit recorded in artifacts: `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 运行环境

- Local root: `/Users/sxye/Documents/expander`
- Remote root: `/home/huiwei/ysx/zigzag_attention`
- Remote env: `ysx_base`, Python `3.10.0`, Torch `2.10.0+cu128`
- Remote GPU: `CUDA_VISIBLE_DEVICES=1`, `NVIDIA A100-SXM4-80GB`
- Remote plotting dependency: `matplotlib` installed under project-local `.deps/`; run commands record `PYTHONPATH=/home/huiwei/ysx/zigzag_attention/.deps`.

## 命令

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --methods dense,local,random,zigzag \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/copy_v05_phase1_smoke
```

Remote command is recorded in `outputs/copy_v05_phase1_smoke/command.sh` and each run subdirectory's `command.sh`.

## 结果表

Artifacts:

- `outputs/copy_v05_phase1_smoke/summary.json`
- `outputs/copy_v05_phase1_smoke/results.csv`
- `outputs/copy_v05_phase1_smoke/results.jsonl`
- `outputs/copy_v05_phase1_smoke/metrics.jsonl`
- `outputs/copy_v05_phase1_smoke/mask_tests.json`
- `logs/copy_v05_phase1_smoke_20260612_141458.log`

Remote synced result rows:

| method | backend | raw_K | effective_K_mean_after_causal | attention_pair_count_after_causal | eval_token_accuracy | status |
|---|---|---:|---:|---:|---:|---|
| dense | dense_mask | 272 | 136.500 | 37128 | 0.2287 | ok |
| local | split | 16 | 8.500 | 2312 | 0.2093 | ok |
| random | split | 20 | 10.460 | 2845 | 0.2171 | ok |
| zigzag | split | 20 | 10.500 | 2856 | 0.2209 | ok |

## 通过/失败项

- Config files exist: pass
- `scripts/graph_structures.py` exists and G/H config is recorded: pass
- `synthetic_mvp.py --config` drives the run: pass
- Dense/local/random/zigzag complete forward/backward/eval: pass
- Mask tests: pass, 8 cases
- `command.sh`, `config_snapshot.json`, CSV/JSONL, metrics JSONL: pass
- Per-run `training_curves.png`: pass, 4/4
- Remote results synced back to local: pass
- Git commit completed before final runs: pass, `75330e074f67e3f181dc3e4cab2d941eb54dbf2d`

## 解释

Phase 1 only validates implementation structure and artifact completeness. Accuracy is not interpreted as convergence. The previous single-token copy path and old "Tiny" model naming were removed; the script now implements causal full-copy LM loss over copy output positions and EOS.

