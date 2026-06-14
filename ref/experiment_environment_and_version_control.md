# 实验环境与版本控制规范

## 0. 文档定位

本文档保存所有实验版本共享的环境、远端服务器、GPU 使用、同步和 git 版本控制规则。任务手册只描述具体实验目标、phase、配置和结果字段；环境要求统一引用本文档，不再复制到每个大版本手册中。

适用范围：

```text
ref/zigzag_experiment_execution_manual_v06.md
后续 v07/v08/... 实验手册
所有 scripts/configs/ref 代码与文档更新
所有远端 GPU 实验
```

若任务手册与本文档冲突，以本文档为准，除非用户在当前对话中明确指定临时例外。

## 1. Git 版本控制

每完成一次代码大更新，必须使用 git 提交版本控制。代码大更新包括：

```text
新增或重构核心脚本；
新增实验 phase；
新增或修改 config schema；
新增数据下载/预处理/eval pipeline；
修改 attention、graph、mask、training loop；
完成一个 phase 并产出阶段报告；
修复影响结果解释的 bug。
```

实验产物、文档和日志也是实验可追溯性的一部分。除 checkpoint/tensor 大文件外，以下内容也必须进入 git：

```text
ref/ 下的实验文档、报告、归档说明；
configs/ 下实际使用的配置；
reports/ 下阶段报告和最终报告；
logs/ 下正式运行日志；
outputs/ 下结果表、summary、metrics、diagnostics、training_curves.png、command.sh、config snapshot、graph artifact 副本；
datasets/ 下 data_readiness、dataset_info、tokenized_smoke、数据源说明等轻量元数据；
envs/ 下环境快照。
```

不进入 git 的例外：

```text
*.pt
*.pth
checkpoint*.pt
checkpoint*.pth
大型原始数据文件或 cache 文件；
大型 tensor dump；
临时 __pycache__、.DS_Store。
```

如果某个产物太大但不是 checkpoint，先移动到明确的外部归档位置，并在对应报告或 README 中记录路径、sha256、生成命令和不提交原因。

提交前必须检查：

```bash
git status --short
python -m py_compile scripts/*.py
```

如果涉及新增脚本，还必须运行该脚本的 smoke test 或 dry run，并在阶段报告中记录命令和结果。

提交规范：

```text
1. 提交代码、配置、文档、日志和轻量实验产物；
2. 不提交 checkpoint/tensor 大文件，尤其是 *.pt、*.pth；
3. 不回退用户已有改动；
4. 若工作区已有无关改动，只提交本次明确相关文件；
5. commit message 使用可追溯前缀，例如 v06-graph、v06-mask、v06-wikitext-data；
6. 每个 phase 完成后至少有一个对应 commit。
```

远端若发生小修复，必须同步回本地并进入 git；远端不作为长期编辑主目录。

## 2. 本地与远端目录

本地唯一编辑主目录：

```text
/Users/sxye/Documents/expander
```

远端运行主目录：

```text
/home/huiwei/ysx/zigzag_attention
```

代码、配置和文档修改优先在本地完成，通过 rsync 同步到远端。远端只用于运行、查看日志和必要的小修复。

推荐代码同步命令：

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'cached_graphs/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  ./ huiwei:/home/huiwei/ysx/zigzag_attention/
```

结果同步使用任务手册中的具体 outputs/datasets/logs 路径。同步回本地后，报告中的所有数值必须能追溯到本地已保存且已进入 git 的产物；checkpoint 例外，但必须保留生成命令和路径记录。

## 3. 服务器规范

远端实验默认在公用 A100 服务器上运行：

```text
host alias: huiwei
remote root: /home/huiwei/ysx/zigzag_attention
conda env: ysx_base
GPU: 4 x NVIDIA A100-SXM4-80GB
```

远端默认环境：

```bash
conda activate ysx_base
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

每个大版本至少保存一次环境快照：

```bash
mkdir -p envs
python -V
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pip freeze > envs/requirements_snapshot.txt
```

如果环境中缺少包，不直接污染公共环境；优先使用项目内兼容 shim 或新建私有环境，并把原因写入 phase 报告。

## 4. GPU 选择规则

每次远端运行前必须检查 GPU 使用情况：

```bash
ssh huiwei
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

启动任务规则：

```text
1. 优先按 GPU3 -> GPU2 -> GPU1 -> GPU0 的顺序考虑；
2. 目标 GPU 当前 utilization.gpu < 20% 时，可以启动任务；
3. 若 GPU3/2/1/0 的利用率都 >= 20%，或显存被未知进程大量占用，不启动新任务；
4. 可以同时使用最多 4 张 GPU，以提升整体实验效率；
5. 不同 GPU 可以执行不同任务，例如 copy smoke、copy real、wikitext smoke、wikitext real 或不同 method；
6. 单个训练进程默认只绑定一张 GPU，除非某个脚本明确实现并记录了多 GPU 训练；
7. 正式记录必须写明每个任务的 CUDA_VISIBLE_DEVICES、启动前 GPU 状态和实际 GPU 名称。
```

若多张 GPU 都低于 `20%`，按 `3 -> 2 -> 1 -> 0` 优先级分配；若需要并行启动多个任务，按该顺序逐个分配空闲 GPU。若优先级相同的人工判断场景出现，再选择显存占用更低的 GPU。

并行任务推荐使用多个 tmux window/session，每个任务设置一个单独的 `CUDA_VISIBLE_DEVICES=<gpu_id>`。只有后续明确加入多 GPU 训练时，才允许单个任务同时设置多个 GPU id。

## 5. 运行方式与日志

短 smoke 可以直接运行。超过数分钟的图搜索、数据处理或正式训练必须使用 `tmux` 或 `screen`：

```bash
tmux new -s <session_name>
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
```

所有远端正式命令必须同时写入 log：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=<gpu_id> python <script.py> ... \
  2>&1 | tee logs/<run_name>_$(date +%Y%m%d_%H%M%S).log
```

每个 run 输出目录必须保存：

```text
command.sh
config_snapshot.json
summary.json 或 error.log
```

## 6. 远端运行前检查清单

每次正式运行前检查：

```text
git status 本地干净或仅含本次明确改动；
当前 commit hash 已记录；
本地代码已 rsync 到远端；
远端 configs/scripts/ref 文件存在；
任务所需 graph/data artifact 已存在；
nvidia-smi 按 GPU3 -> GPU2 -> GPU1 -> GPU0 检查，确认目标 GPU 利用率低于 20%；
输出目录不会覆盖已有有效结果；
日志路径已设置；
config snapshot 会写入输出目录；
command.sh 会写入输出目录。
```

## 7. 断点、失败与归档

正式训练必须支持或至少满足以下恢复策略：

```text
每个 method / seed 或 run_id 独立输出目录；
已完成 run 不重复覆盖；
失败 run 保留 error.log；
合并结果时跳过未完成 run 并标记 status=failed；
重新运行只补缺失项。
```

失败不是删除重跑。每次失败必须保留：

```text
error.log
failed command
config snapshot
失败发生的 step 或 batch
是否占用 GPU
初步原因判断
是否需要改代码
```

旧版本 outputs、reports、logs、configs 和轻量 dataset/env 元数据应归档到 `ref/archive_<version>_reports/` 并进入 git；checkpoint/tensor 大文件不进入 git，但归档 README 必须记录外部保存路径或删除原因。旧版本产物不得混入新版本主结果表。
