# v08 Phase 3 远端数据与环境就绪报告

## 结论

Phase 3 远端就绪检查通过。六个 v08 主实验允许的数据集在本地与远端之间的 split 行数、核心文件数量和 SHA256 校验和一致，远端训练环境 `ysx_base` 可用，A100 GPU 在检查时均为空闲。

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| `remote_host` | 远端登录主机 | `huiwei` | Phase 3 同步和检查命令 | 标识数据和训练将部署到哪台机器 | 无 |
| `remote_data_root` | 远端 probe 数据根目录 | `/home/huiwei/ysx/expander_bench/data/probes` | Phase 3 readiness 输出 | 复现实验数据读取路径 | 无 |
| `remote_env` | 远端 conda 环境 | `ysx_base` | Phase 3 readiness 输出 | 标识 Python/PyTorch 运行环境 | 无 |
| `attention_contract` | 注意力合同 | `non_causal` | v08 手册和 readiness 检查 | 防止误用 causal LM 设置 | 不满足则阻止进入 main |
| `causal` | 是否使用 causal mask | `false` | v08 手册和 readiness 检查 | v08 主实验必须为 non-causal | 不满足则阻止进入 main |
| `graph_directionality` | 图方向性 | `directed` | v08 手册 | 保证后续图工件按 directed contract 生成 | 不满足则阻止进入 main |
| `selected_tasks` | 允许进入 v08 主实验的任务集合 | 6 个 task 名 | Phase 1 数据审计和 v08 手册 | 防止未 validated 任务混入 | 无 |
| `forbidden_main_eval_tasks` | 禁止进入主评估的任务集合 | `lra_pathfinder`, `lra_pathx` | v08 手册和 Phase 1 审计 | 明确同步但不训练的目录 | 若出现于 main config 则失败 |
| `readiness_command` | 远端 readiness 检查命令 | `python scripts/probe_remote_readiness.py` | Phase 3 执行记录 | 复现环境和数据检查 | 无 |
| `SHA256` | 文件内容哈希 | 64 位十六进制摘要 | checksums 和 readiness 输出 | 验证本地/远端数据一致 | 校验失败则阻止后续 phase |
| `A100` | GPU 型号缩写 | `NVIDIA A100-SXM4-80GB` | 远端 GPU 快照 | 记录训练硬件 | 无 |
| `CUDA` | CUDA 运行时 | `2.10.0+cu128` 对应 PyTorch CUDA 构建 | 远端环境快照 | 记录 GPU 软件栈 | CUDA 不可用则阻止训练 |

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

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 12 |
| unexplained_parameters | [] |
| english_only_sections | [] |
