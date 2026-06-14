# v06 Artifact Archive

This directory archives v06 experiment artifacts before starting the v07 plan.

Archived on: 2026-06-14
Checkpoint archive moved on: 2026-06-15

Original locations:

```text
outputs/*v06*        -> ref/archive_v06_reports/outputs/
configs/*v06*        -> ref/archive_v06_reports/configs/
reports/*v06*        -> ref/archive_v06_reports/reports/
logs/*v06*           -> ref/archive_v06_reports/logs/
datasets/wikitext2*  -> ref/archive_v06_reports/datasets/
envs/*v06*           -> ref/archive_v06_reports/envs/
scripts/*v06* helper -> ref/archive_v06_reports/scripts/
```

Large checkpoint/tensor artifacts are intentionally not committed, following
`ref/experiment_environment_and_version_control.md`. The v06 checkpoint files
were moved out of the git worktree to:

```text
/Users/sxye/Documents/expander_external_artifacts/v06_checkpoints_20260615
```

Moved checkpoint files:

```text
150 files, about 2.0G total
```

The committed archive keeps the lightweight reproducibility artifacts:
commands, config snapshots, metrics, summaries, diagnostics, reports, logs,
graphs, and training curves.

The original v06-specific helper is archived here for reproducibility. A generic copy-output repair utility is retained at:

```text
scripts/repair_copy_outputs.py
```

The v06 execution manual itself remains at:

```text
ref/zigzag_experiment_execution_manual_v06.md
```

New v07 work should use versioned paths such as:

```text
outputs/copy_v07_main_n1024_q32_B32_d8/
outputs/wikitext_v07_main_n1024_q32_B32_d8/
configs/copy_v07_main_n1024_q32_B32_d8.json
configs/wikitext_v07_main_n1024_q32_B32_d8.json
```
