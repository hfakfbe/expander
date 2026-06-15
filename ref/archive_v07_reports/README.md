# v07 Artifact Archive

This directory archives v07 experiment artifacts before starting the v08 probe-data plan.

Archived on: 2026-06-15

Original locations:

```text
configs/*v07*                 -> ref/archive_v07_reports/configs/
env_snapshot.yaml and envs/env_snapshot.yaml -> ref/archive_v07_reports/envs/
envs/*v07*                    -> ref/archive_v07_reports/envs/
envs/requirements_snapshot.txt -> ref/archive_v07_reports/envs/
logs/*v07*                    -> ref/archive_v07_reports/logs/
outputs/*v07*                 -> ref/archive_v07_reports/outputs/
reports/*v07*                 -> ref/archive_v07_reports/reports/
reports/user_experiment_report_2.docx -> ref/archive_v07_reports/reports/
reports/user_experiment_report_3.*    -> ref/archive_v07_reports/reports/
scripts/v07_artifacts.py      -> ref/archive_v07_reports/scripts/
```

Root-level duplicate v05 logs were removed because the same files already exist
under `ref/archive_v05_reports/logs_archive_v05/`.

Ignored tensor/cache artifacts are intentionally kept out of the git worktree.
Old v04 `.pt` tensor files were moved to:

```text
/Users/sxye/Documents/expander_external_artifacts/v04_ignored_tensors_20260615
```

Moved tensor files:

```text
56 files, about 2.3M total
```

The committed v07 archive keeps the lightweight reproducibility artifacts:
commands, configs, environment snapshots, logs, graph artifacts, tokenizer
artifacts, metrics, summaries, diagnostics, training curves, and reports.

The v07 execution manual itself remains at:

```text
ref/zigzag_experiment_execution_manual_v07.md
```

New v08 work should use fresh versioned paths such as:

```text
configs/probes_v08_smoke.json
configs/probes_v08_main.json
outputs/probes_v08_smoke/
outputs/probes_v08_main/
reports/v08_probe_main_eval_report.md
```
