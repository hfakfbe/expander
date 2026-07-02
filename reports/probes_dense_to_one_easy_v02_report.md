# Probes Dense-to-One Easy v02 Report

## Scope

This phase follows `ref/experiment_environment_and_version_control.md` and runs:

- `dense` on `selective_copy`, `induction_associative_recall`, and `lra_listops`, after adjusting the suite until the dense baseline reaches 1.0 on the required final metrics.
- `random_regular` and `random_memory` on the same three tasks with pure random actual mask density target `0.1`.

The v02 suite is a finite-language calibration suite generated after the earlier v01 attempt failed to make the dense baseline reach 1.0 on all tasks. It is therefore evidence for the requested dense-to-one calibration target, not a broad generalization benchmark.

## Configuration

- Configs:
  - `configs/probes_dense_to_one_easy_v02_dense.json`
  - `configs/probes_dense_to_one_easy_v02_random_density10.json`
  - `configs/probes_dense_to_one_easy_v02_task_parameters.json`
- Data root: `datasets/probes_dense_to_one_easy_v02/`
- Output root: `outputs/probes_dense_to_one_easy_v02/runs/`
- Formal run host: `huiwei:/home/huiwei/ysx/zigzag_attention`
- Environment snapshot: `envs/probes_dense_to_one_easy_v02_requirements_snapshot.txt`
- Code/config baseline before formal remote run: `3dcfd0c` (`probes-dense-to-one-easy-v02-calibration`)

Shared main-profile training settings:

| Field | Value |
|---|---:|
| Layers | 8 |
| d_model | 128 |
| Heads | 4 |
| FFN dim | 512 |
| Dropout | 0.0 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Weight decay | 0.0 |
| Grad clip | 1.0 |
| LR scheduler | cosine |
| Base LR | 3e-4 |
| Min LR | 3e-5 |
| Warmup ratio | 0.0 |
| Position encoding | RoPE, Q/K-only |

The learning-rate schedule is identical across `dense`, `random_regular`, and `random_memory`: cosine from `3e-4` to `3e-5` with no warmup. This is the corrected probe runner's scheduled LR path, not the older constant-`1e-3` sanity style.

Random settings:

| Field | Value |
|---|---:|
| Configured actual mask density | 0.1 |
| Actual pair count | 410 |
| Actual density | 0.10009765625 |
| Sequence length | 64 |
| Block-local exclusion | enabled |
| `use_log_m` | false |
| Relative attention bias | not enabled; `ProbeTransformer` default is false |
| Attention top-k | not enabled; `ProbeTransformer` default is 0 |

`random_memory` settings:

| Field | Value |
|---|---:|
| Source | input |
| Update | lazy |
| Lazy alpha | 0.5 |
| Scale | 2.0 |
| Steps | 1 |
| Head merge | mean |
| Weight mode | soft |
| Edge scope | all |

Note: this v02 run uses single-hop generic attention-rollout memory (`steps=1`). It does not run the earlier weighted multiscale `0.9 * one-hop + 0.1 * two-hop` variant.

## Dense Results

Aggregate file: `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_dense/results_train.csv`

| Task | Method | Steps | Primary metric | Token metric | Sequence / class metric |
|---|---|---:|---:|---:|---:|
| selective_copy | dense | 1920 | 1.0 | token acc 1.0 | sequence acc 1.0 |
| induction_associative_recall | dense | 1920 | 1.0 | retrieval token acc 1.0 | retrieval exact match 1.0 |
| lra_listops | dense | 3000 | 1.0 | n/a | listops accuracy 1.0, macro 1.0 |

Dense final-eval paths:

- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_dense/selective_copy/dense/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_dense/induction_associative_recall/dense/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_dense/lra_listops/dense/seed0/final_eval.json`

## Random Results

Aggregate file: `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/results_train.csv`

| Task | Method | Steps | Primary metric | Secondary metric | Density | Pair count |
|---|---|---:|---:|---:|---:|---:|
| selective_copy | random_regular | 1920 | token acc 1.0 | sequence acc 1.0 | 0.10009765625 | 410 |
| selective_copy | random_memory | 1920 | token acc 1.0 | sequence acc 1.0 | 0.10009765625 | 410 |
| induction_associative_recall | random_regular | 1920 | exact match 1.0 | token acc 1.0 | 0.10009765625 | 410 |
| induction_associative_recall | random_memory | 1920 | exact match 1.0 | token acc 1.0 | 0.10009765625 | 410 |
| lra_listops | random_regular | 3000 | accuracy 1.0 | macro accuracy 1.0 | 0.10009765625 | 410 |
| lra_listops | random_memory | 3000 | accuracy 0.8225 | macro accuracy 0.8225 | 0.10009765625 | 410 |

The listops result is the main negative finding: under this v02 setting, generic single-hop `random_memory` is worse than `random_regular` on the final/test split, despite using the same mask density and training profile.

Random final-eval paths:

- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/selective_copy/random_regular/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/selective_copy/random_memory/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/induction_associative_recall/random_regular/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/induction_associative_recall/random_memory/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/lra_listops/random_regular/seed0/final_eval.json`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/lra_listops/random_memory/seed0/final_eval.json`

## Commands And Logs

Dense formal run timestamp: `20260702_025759`

- GPU state: `logs/probes_dense_to_one_easy_v02_dense_gpu_state_20260702_025759.txt`
- Train logs:
  - `logs/probes_dense_to_one_easy_v02_dense_selective_copy_train_20260702_025759.log`
  - `logs/probes_dense_to_one_easy_v02_dense_induction_train_20260702_025759.log`
  - `logs/probes_dense_to_one_easy_v02_dense_lra_listops_train_20260702_025759.log`
- Final-eval logs:
  - `logs/probes_dense_to_one_easy_v02_dense_selective_copy_final_20260702_025759.log`
  - `logs/probes_dense_to_one_easy_v02_dense_induction_final_20260702_025759.log`
  - `logs/probes_dense_to_one_easy_v02_dense_lra_listops_final_20260702_025759.log`

Random formal run timestamp: `20260702_030333`

- GPU state: `logs/probes_dense_to_one_easy_v02_random_gpu_state_20260702_030333.txt`
- Train logs:
  - `logs/probes_dense_to_one_easy_v02_random_selective_copy_train_20260702_030333.log`
  - `logs/probes_dense_to_one_easy_v02_random_induction_train_20260702_030333.log`
  - `logs/probes_dense_to_one_easy_v02_random_lra_listops_train_20260702_030333.log`
- Final-eval logs:
  - `logs/probes_dense_to_one_easy_v02_random_selective_copy_final_20260702_030333.log`
  - `logs/probes_dense_to_one_easy_v02_random_induction_final_20260702_030333.log`
  - `logs/probes_dense_to_one_easy_v02_random_lra_listops_final_20260702_030333.log`

Each formal run directory contains `command.sh`, `raw_config_snapshot.json`, `resolved_config_snapshot.json`, `run_identity.json`, `summary.json`, `metrics.jsonl`, `final_eval.json`, `final_eval.csv`, and `training_curves.png`. Checkpoint tensor files are intentionally excluded from git.

## Verification

Completed local checks:

```bash
python scripts/run_probes_corrected.py --config configs/probes_dense_to_one_easy_v02_dense.json --mode aggregate --aggregate-mode train
python scripts/run_probes_corrected.py --config configs/probes_dense_to_one_easy_v02_random_density10.json --mode aggregate --aggregate-mode train
python -m py_compile scripts/*.py
```

Both aggregate commands returned `status: ok`; dense completed `3/3` expected runs and random completed `6/6` expected runs. `python -m py_compile scripts/*.py` passed locally before the results commit.
