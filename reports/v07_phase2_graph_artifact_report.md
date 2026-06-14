# v0.7 Phase 2 Canonical Graph Artifact Report

## Status

Status: passed.

Date: 2026-06-14

## Command

```bash
python scripts/graph_diagnostics.py --config configs/graph_v07_n1024_q32_B32_d8.json --output-dir outputs/v07_graph_n1024_q32_B32_d8 2>&1 | tee logs/graph_v07_n1024_q32_B32_d8_20260614T110346Z.log
```

## Canonical Artifact

- Directory: `outputs/v07_graph_n1024_q32_B32_d8/`
- Graph artifact: `outputs/v07_graph_n1024_q32_B32_d8/selected_graph.json`
- Certificate: `outputs/v07_graph_n1024_q32_B32_d8/graph_certificate.json`
- Generation metadata: `outputs/v07_graph_n1024_q32_B32_d8/graph_generation.json`
- Artifact sha file: `outputs/v07_graph_n1024_q32_B32_d8/graph_artifact.sha256`

## Fixed Parameters

- `N_total=1024`
- `T=1024`
- `q=32`
- `B=32`
- `d=8`
- `graph_seed=0`
- `graph_generation_algorithm=zigzag_v07_fixed_N1024_q32_B32_d8`
- `allow_multiedges=true`
- `preserve_multiplicity=true`

## Certificate Summary

- `graph_id=v07_B32_d8_s0_619e9b962fc61bfd`
- `lambda_G=0.0625`
- `mu_H=0.6436248213046418`
- `rho_zigzag_bound=0.9114789606824917`
- `rho_zigzag_exact=0.46386654287979273`
- `rho_zigzag_certified=true`
- `rot_g_is_bijection=true`
- `P_G_row_stochastic_error=0.0`
- `P_G_col_stochastic_error=0.0`
- `P_H_row_stochastic_error=0.0`
- `P_H_col_stochastic_error=0.0`

## SHA256

```text
canonical_graph_artifact_sha256=53ae37a6584833a1d20d51162a01a03d18bf7eeb9fd0efd2ebf3b8a482427a48
```

Dry-run verification copied this artifact into a method run directory and confirmed byte-identical sha256:

```text
53ae37a6584833a1d20d51162a01a03d18bf7eeb9fd0efd2ebf3b8a482427a48  outputs/v07_graph_n1024_q32_B32_d8/selected_graph.json
53ae37a6584833a1d20d51162a01a03d18bf7eeb9fd0efd2ebf3b8a482427a48  outputs/copy_v07_phase1_dryrun_n1024_q32_B32_d8/train_N511_seed0_zigzag_certified/artifacts/graph/selected_graph.json
```
