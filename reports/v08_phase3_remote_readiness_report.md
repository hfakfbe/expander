# v08 Phase 3 远端数据与环境就绪报告

## 结论

Phase 3 远端就绪检查通过。六个 v08 主实验允许的数据集在本地与远端之间的 split 行数、核心文件数量和 SHA256 校验和一致，远端训练环境 `ysx_base` 可用，A100 GPU 在检查时均为空闲。

## 参数说明

- `remote_host`: `huiwei`
- `remote_data_root`: `/home/huiwei/ysx/expander_bench/data/probes`
- `remote_env`: `ysx_base`
- `attention_contract`: `non_causal`
- `causal`: `false`
- `graph_directionality`: `directed`
- `selected_tasks`: `copy`, `selective_copy`, `induction_associative_recall`, `niah_kv_retrieval`, `ruler`, `lra_listops`
- `forbidden_main_eval_tasks`: `lra_pathfinder`, `lra_pathx`
- `readiness_command`: `python scripts/probe_remote_readiness.py`

`lra_pathfinder` 与 `lra_pathx` 原始数据目录随完整数据树同步到远端，但不在 v08 主实验任务清单中，后续参数配置和训练入口均只读取上述六个允许任务。

## 数据同步与校验

| 任务 | train | validation | test | 必需文件数 | 状态 |
|---|---:|---:|---:|---:|---|
| copy | 10000 | 1000 | 1000 | 10 | ok |
| selective_copy | 10000 | 1000 | 1000 | 10 | ok |
| induction_associative_recall | 180000 | 3000 | 4000 | 10 | ok |
| niah_kv_retrieval | 10000 | 500 | 500 | 10 | ok |
| ruler | 12000 | 3000 | 3000 | 10 | ok |
| lra_listops | 96000 | 2000 | 2000 | 10 | ok |

每个任务均验证了 `train.jsonl`、`validation.jsonl`、`test.jsonl`、`dataset_card.json` 的远端 SHA256 与本地一致。完整记录见 `outputs/probes_v08_remote_readiness/remote_checksums_verification.json`。

## 远端环境快照

- Python: `Python 3.10.0`
- PyTorch/CUDA: `2.10.0+cu128`, CUDA 可用
- GPU: 4 张 `NVIDIA A100-SXM4-80GB`
- 检查时 GPU 状态: GPU 0/1/2/3 利用率均为 `0 %`，显存占用均为 `10 MiB / 81920 MiB`

`pip freeze` 已保存到 `outputs/probes_v08_remote_readiness/requirements_snapshot.txt`，环境和 GPU 快照已保存到 `outputs/probes_v08_remote_readiness/env_snapshot.txt`。

## 审计结论

- 本地与远端六个选定数据版本完全一致，满足 Phase 3 的可复现实验输入要求。
- 远端训练入口将使用 `/home/huiwei/ysx/zigzag_attention` 作为代码根目录，数据从 `/home/huiwei/ysx/expander_bench/data/probes` 读取。
- 远端 GPU 优先级按手册采用 `3 -> 2 -> 1 -> 0`，检查时全部低于 20% 利用率。
- 远端代码目录通过 rsync 同步时排除了 `.git`，因此后续同步会写入 `.deployed_git_commit_v08` 标记文件，训练结果将记录部署 commit。

## 产物

- `outputs/probes_v08_remote_readiness/summary.json`
- `outputs/probes_v08_remote_readiness/remote_file_counts.csv`
- `outputs/probes_v08_remote_readiness/remote_checksums_verification.json`
- `outputs/probes_v08_remote_readiness/env_snapshot.txt`
- `outputs/probes_v08_remote_readiness/requirements_snapshot.txt`
- `outputs/probes_v08_remote_readiness/command.sh`
