# Copy v06 Experiment Report

Date: 2026-06-13

This report records the repaired v0.6 copy result under
`ref/zigzag_experiment_execution_manual_v06.md` and
`ref/experiment_environment_and_version_control.md`.

## Scope

- Task: online causal full-copy.
- Main train/eval length: `N_train=512`, `N_eval=512`.
- Main seed: `0`.
- Main methods: `dense`, `local`, `random_regular`, `zigzag_certified`, `zigzag_boolean`.
- Main result root: `outputs/copy_v06_main_n512`.
- Historical extra run dirs for seeds `1` and `2` are preserved but excluded from the repaired v0.6 main table.
- Remote run metadata records `CUDA_VISIBLE_DEVICES=3`, `NVIDIA A100-SXM4-80GB`, and git commit `525791fb5f020146bbd8aee3400c3ac7fb998521`.

## Structural Gate

Selected graph: `v06_B16_d8_s2_c41f9afbd3840ed1`.

| Metric | Value |
|---|---:|
| graph seed | 2 |
| B | 16 |
| d | 8 |
| q | 65 |
| lambda_G | 0.4430783779678246 |
| mu_H | 0.5491481939947735 |
| rho_bound | 0.860373989287474 |
| rho_exact | 0.5999831601197602 |
| certified graph | true |

Artifacts:

- `outputs/copy_v06_graph_search/graph_certificates.csv`
- `outputs/copy_v06_graph_search/graph_certificates.jsonl`
- `outputs/copy_v06_graph_search/selected_graph.json`
- `outputs/copy_v06_graph_search/selected_graph_certificate.json`

## Repair Pass

The v0.6 manual section 16 required repairing result semantics without rerunning copy training. Completed fixes:

- Recomputed `shortcut_diagnostics.csv/jsonl` with `target_in_Lhop_rate = dist <= layers`.
- Rewrote root `results.csv/jsonl`, `phase5_results.*`, and `summary.json` to include only configured main seed `0`.
- Added `graph_certified`, `implementation_certified`, and `theory_aligned_method`.
- Kept `certified` only as a backward-compatible alias for `theory_aligned_method`.
- Added resolved `config_snapshot.json` and preserved original `raw_config_snapshot.json`.

## Main Results

| Method | eval loss | token acc | seq acc | 1-hop | 2-hop | L-hop | unreachable | theory aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.00004525 | 0.99999756 | 0.99875 | 0.99805068 | 0.99805068 | 0.99805068 | 0.00194932 | false |
| local | 1.38366863 | 0.25192008 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | false |
| random_regular | 0.22453421 | 0.87857943 | 0.0 | 0.04288499 | 0.95126706 | 0.99805068 | 0.00194932 | false |
| zigzag_boolean | 0.69184633 | 0.62804337 | 0.0 | 0.04483431 | 0.39571150 | 0.98635478 | 0.01364522 | false |
| zigzag_certified | 0.69764033 | 0.62485136 | 0.0 | 0.04483431 | 0.39571150 | 0.98635478 | 0.01364522 | true |

Interpretation limits:

- `zigzag_certified` is above local but below random regular on this copy setup.
- Random regular has much stronger 2-hop shortcut coverage, so its advantage is not a general sparse-attention conclusion.
- `zigzag_boolean` uses the certified graph but is not implementation-certified and is only an ablation.
- No `N > 512` extrapolation, multi-seed stability claim, official benchmark claim, or general sparse superiority claim is made.

## Validation

- `python -m py_compile scripts/*.py`: pass.
- Wrapper compatibility check: `python scripts/synthetic_mvp.py --config configs/copy_v06_smoke.json --output-dir outputs/copy_v06_wrapper_check --methods local --steps 1 --eval-batches 1 --batch-size 1 --device cpu --skip-tests`: pass.
- Repair command: `python scripts/repair_copy_v06_outputs.py --output-dir outputs/copy_v06_main_n512 --config configs/copy_v06_main_n512.json --graph-artifact outputs/copy_v06_graph_search/selected_graph.json --main-seeds 0`.
