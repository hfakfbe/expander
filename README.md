# Expander Attention Experiments

This repository contains a neutral runtime for the current sparse-attention experiments.

Main tasks:

- `copy`
- `selective_copy`
- `induction_associative_recall`
- `lra_listops`

Main attention methods:

- `dense`
- `local`
- `random_regular`
- `zigzag_logm`
- `zigzag_boolean`

Memory rollout is an independent option configured through `memory_rollout`.
For example, `configs/runs/copy_random_memory.json` uses
`attention.method=random_regular` with `memory_rollout.enabled=true`, and
`configs/runs/copy_zigzag_logm_memory.json` enables the same rollout on
`zigzag_logm`.

Model normalization is selected with `model.norm_type`: `layernorm`,
`rmsnorm`, or `none`.

Core code lives under `src/`. The `scripts/` directory only contains thin CLI wrappers.

## Common Commands

Audit data:

```bash
python scripts/audit_data.py --task copy --config configs/runs/copy_dense.json
```

Prepare a resolved config and graph artifacts:

```bash
python scripts/prepare_data.py --task copy --config configs/runs/copy_zigzag_logm.json
```

Check one configured run:

```bash
python scripts/run_task.py --task copy --config configs/runs/copy_dense.json --mode check
```

Train:

```bash
python scripts/run_task.py --task copy --config configs/runs/copy_random_memory.json --mode train
```

Final evaluation:

```bash
python scripts/run_task.py --task copy --config configs/runs/copy_random_memory.json --mode final-eval
```

## Outputs

Each run writes:

- `resolved_config.json`
- `metrics.jsonl` for training
- `final_metrics.json`
- `run_manifest.json`
- checkpoints when enabled
- graph artifacts when enabled

The manifest records the resolved config, command, git state, dataset hashes, and graph artifact hashes.

## Validation

Run the neutral test suite:

```bash
python -m unittest discover -s tests
```

Run static compilation:

```bash
python -m compileall -q src scripts tests
```
