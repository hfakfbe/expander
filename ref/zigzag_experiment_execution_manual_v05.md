# Zig-Zag Sparse Attention Copy 实验执行手册 v0.5

## 0. 文档定位

本文档取代 `zigzag_experiment_execution_manual_v04.md` 中的旧 phase 设计。v0.5 不再围绕 LRA、base 仓库复用、Phase 6 或 scaling law 展开，而是聚焦一个更窄、更可控的目标：

```text
在当前代码基础上，完成可配置、可复现、可同步、可版本控制的 synthetic copy 实验。
```

当前实验只考虑在线生成的 copy 数据，不再考虑 LRA/ListOps/Text/Retrieval 等任务。旧阶段报告已归档到：

```text
ref/archive_v04_reports/
```

v0.5 的实验入口是现有 synthetic attention 代码，而不是重新选择外部仓库。所有后续结论只允许来自本手册定义的任务、配置、日志和结果表。

## 1. 全局执行规范

### 1.1 严格顺序

实验重新划分为 3 个 phase。必须按顺序推进，前一个 phase 没有通过，不进入下一个 phase。

| Phase | 名称 | 目标 | 通过条件 |
|---|---|---|---|
| 1 | 配置化与图结构解耦 | 所有关键参数可通过 config 控制，G/H 与训练代码解耦 | config 能驱动 smoke run，G/H 单元测试通过 |
| 2 | Online Copy Smoke Test | 用随机种子在线生成 copy 数据，跑通训练与评估 | 4 种方法均能完成小步数 forward/backward/eval，无 NaN，日志完整 |
| 3 | Copy 主实验 | 在 N=256/512/1024 训练，评测到 N=2048 | 4 种方法、指定 seeds、训练表和 extrapolation 评测表完整 |

禁止跳步。例如：没有完成 config 化，不允许直接跑主实验；没有完成 smoke test，不允许提交长训练；没有本地和远端代码同步，不允许报告远端结果。

### 1.2 本地和远端同步

所有代码、配置、脚本和文档必须保持本地与远端同步。默认路径如下：

```text
local:  /Users/sxye/Documents/expander
remote: /home/huiwei/ysx/zigzag_attention
```

每次进入远端运行前，必须执行同步检查：

```bash
# local
rsync -av --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  ./ huiwei:/home/huiwei/ysx/zigzag_attention/

# remote
cd /home/huiwei/ysx/zigzag_attention
find scripts configs ref -type f | sort
```

远端产生的新结果必须同步回本地：

```bash
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/ ./outputs/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/logs/ ./logs/
```

任何报告中的数值必须能追溯到本地已同步的 `outputs/` 或 `logs/`。

### 1.3 Git 版本控制

本项目从 v0.5 开始要求使用 git 管理大版本。当前目录若尚不是 git 仓库，必须先初始化或绑定远端仓库：

```bash
cd /Users/sxye/Documents/expander
git init
git status --short
```

每完成一个大版本修改必须提交一次。建议提交粒度：

```text
v05-doc: 归档旧报告并新增 v05 实验手册
v05-phase1: 完成 config 化与 G/H 解耦
v05-phase2: 完成 online copy smoke test
v05-phase3: 完成 copy 主实验与评测表
```

提交前必须检查：

```bash
git status --short
python -m py_compile scripts/*.py
```

若工作区中存在用户未说明的改动，不得覆盖或回退；必须先确认这些改动是否属于当前 phase。

### 1.4 结果记录规范

每次运行必须保存：

```text
config path
command
git commit hash
local/remote run location
GPU id
seed
task
method
N_train
N_eval
B
d
graph_G
graph_H
architecture
steps
batch_size
eval_batches
learning_rate
final train loss
valid/test accuracy
tokens/sec
peak allocated memory
peak reserved memory
```

所有结果表保存为 CSV 和 JSONL，不在正文中手工维护超宽表。

## 2. 研究对象和方法定义

### 2.1 Attention Pattern

长度为 `N` 的 token 序列被划分为 `q=N/B` 个 block，每个 block 包含 `B` 个 token。

4 种方法固定如下：

| 方法 | Attention 范围 | 用途 |
|---|---|---|
| dense | 所有 token | 小 N 上的质量参考 |
| local | 同一 block 内 token | 检查跨 block 边是否必要 |
| random | local + same-budget random cross edges | 检查 zig-zag 是否优于随机跨块边 |
| zigzag | local + zig-zag cross edges | 当前目标方法 |

当 `B=16,d=2` 时：

```text
local K = 16
random / zigzag raw K = B + d^2 = 20
dense K = N
```

### 2.2 G 和 H

从 v0.5 起，G/H 图结构必须与训练代码解耦，单独写在代码文件中。推荐新增：

```text
scripts/graph_structures.py
```

该文件至少提供：

```text
build_h_graph(...)
rot_g(...)
build_zigzag_cross_edges(...)
build_random_cross_edges(...)
validate_graph_config(...)
```

训练脚本不得在主训练逻辑中硬编码 G/H。所有 G/H 选择必须来自 config。

当前第一版配置仍可使用：

```text
G: cyclic rotation map
H: cycle-neighbor
```

但 config 必须预留后续扩展入口，例如：

```text
G.type: cyclic
H.type: cycle
```

后续若加入 random regular、permutation 或 layer-wise graph，只能通过新增 graph module 和 config 字段实现，不能散落在训练脚本里。

## 3. Phase 1：配置化与图结构解耦

### 3.1 目标

把当前实验从脚本参数堆叠改为任务 config 驱动。所有关键参数必须可调，包括：

```text
N
B
d
steps
batch_size
eval_batches
num_values
learning_rate
seeds
methods
attention_backend
G graph structure
H graph structure
architecture
optimizer
output path
```

### 3.2 目录要求

创建或整理 config 目录。后续统一使用：

```text
configs/
  copy_v05_smoke.json
  copy_v05_main.json
```

如果已经存在 `configs/`，沿用该目录，但 v0.5 的配置不要混入旧 LRA/phase6 配置语义。

### 3.3 推荐 config 字段

`configs/copy_v05_main.json` 至少包含：

```json
{
  "task": {
    "name": "copy",
    "data": "online",
    "num_values": 4,
    "train_lengths": [256, 512, 1024],
    "eval_lengths": [256, 512, 1024, 2048]
  },
  "model": {
    "architecture": "tiny_transformer",
    "layers": 8,
    "d_model": 128,
    "heads": 4,
    "ffn_dim": 256,
    "dropout": 0.1,
    "attention_backend": "auto_split"
  },
  "attention": {
    "methods": ["dense", "local", "random", "zigzag"],
    "block_size": 16,
    "degree": 2,
    "graph": {
      "G": {"type": "cyclic"},
      "H": {"type": "cycle"}
    }
  },
  "train": {
    "steps": 1000,
    "batch_size": 16,
    "eval_batches": 20,
    "learning_rate": 0.001,
    "seeds": [0, 1, 2]
  },
  "output": {
    "root": "outputs/copy_v05_main"
  }
}
```

### 3.4 实现要求

本 phase 必须完成：

1. 新增 config 文件。
2. 训练脚本支持 `--config <path>`。
3. 命令行参数只允许覆盖 config 中的少量运行字段，例如 `--output-dir` 或 `--device`。
4. G/H 图结构移入单独文件。
5. 旧 `synthetic_mvp.py` 中与 G/H 相关的函数迁移或包装到 `graph_structures.py`。
6. mask metrics 必须继续记录 raw K、effective K、pair count、duplicate rate、self-loop rate。

### 3.5 单元测试 / Smoke

本 phase 的最小测试：

```bash
python -m py_compile scripts/*.py

python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --methods dense,local,random,zigzag \
  --steps 2 \
  --batch-size 2 \
  --eval-batches 1 \
  --output-dir outputs/copy_v05_phase1_smoke
```

必须检查：

```text
1. config 被正确读取；
2. G/H 类型写入 summary；
3. dense/local/random/zigzag 都有结果；
4. mask tests 通过；
5. 输出包含 command/config snapshot；
6. 本地和远端代码一致。
```

### 3.6 Phase 1 通过条件

满足以下条件才进入 Phase 2：

```text
configs/copy_v05_smoke.json 存在
configs/copy_v05_main.json 存在
scripts/graph_structures.py 存在
--config 能驱动 smoke run
4 种方法都能完成最小运行
结果同步回本地
git commit 完成
```

## 4. Phase 2：Online Copy Smoke Test

### 4.1 目标

不再依赖任何下载数据。copy 数据根据随机种子在线生成，并用于训练、验证和评估。

任务定义：

```text
输入:  x = [x0, x1, ..., xN-1]
x_i ∈ {1, 2, 3, 4}
输出:  y = x0
读取:  最后一个 token 的 hidden state 接分类头
```

随机猜测准确率约为 `1/num_values`。当 `N` 足够大且 `B=16` 时，local-only 无法直接看到 `x0`。

### 4.2 数据生成规范

在线数据生成必须满足：

```text
同一个 seed 下可复现
train/eval 使用同一生成规则
eval batch 不复用训练 batch
num_values 可配置
N 可配置
不读取外部数据文件
```

建议生成函数只依赖：

```text
seed
batch index / step
N
num_values
device
```

避免隐式全局随机状态导致本地/远端结果不可复现。

### 4.3 Smoke 配置

Smoke test 使用低成本设置即可：

```text
N: 128 或 256
B: 16
d: 2
steps: 10 到 50
batch_size: 4 或 8
eval_batches: 2
seeds: [0]
methods: dense, local, random, zigzag
```

Smoke test 的目标不是收敛，而是验证：

```text
数据在线生成
forward/backward 正常
eval 正常
loss/accuracy 可记录
无 NaN
4 种方法产物完整
```

### 4.4 运行命令模板

本地快速检查：

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_local
```

远端 GPU smoke：

```bash
ssh huiwei
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/synthetic_mvp.py \
  --config configs/copy_v05_smoke.json \
  --output-dir outputs/copy_v05_smoke_gpu
```

### 4.5 Phase 2 通过条件

必须满足：

```text
本地 smoke 通过
远端 GPU smoke 通过
4 种方法均无 NaN
summary.json / results.csv / metrics.jsonl 存在
command.sh 或等价命令记录存在
结果同步回本地
git commit 完成
```

## 5. Phase 3：Copy 主实验

### 5.1 目标

进行 copy 实际训练，并测试从训练长度到更长长度的泛化。

训练长度：

```text
N_train = 256, 512, 1024
```

评测长度：

```text
N_eval = 256, 512, 1024, 2048
```

必须包含 4 种方法：

```text
dense
local
random
zigzag
```

### 5.2 主实验固定参数

第一轮主实验固定：

```text
B = 16
d = 2
steps = 1000
batch_size = 16
eval_batches = 20
num_values = 4
learning_rate = 0.001
seeds = config 指定，默认 [0, 1, 2]
architecture = tiny_transformer
layers = 8
d_model = 128
heads = 4
ffn_dim = 256
dropout = 0.1
attention_backend = auto_split
```

如需改动以上参数，必须新建 config，不得覆盖已完成实验的 config。

### 5.3 实验矩阵

主表最小矩阵：

| N_train | method | seeds | eval N |
---|---|---|---|
| 256 | dense/local/random/zigzag | config seeds | 256/512/1024/2048 |
| 512 | dense/local/random/zigzag | config seeds | 256/512/1024/2048 |
| 1024 | dense/local/random/zigzag | config seeds | 256/512/1024/2048 |

注意：

```text
训练在 N_train 上进行。
评测必须覆盖 N_eval=2048。
当 N_eval > N_train 时，报告为 extrapolation，不得与训练长度内评测混淆。
```

### 5.4 输出目录

推荐输出结构：

```text
outputs/copy_v05_main/
  config_snapshot.json
  phase3_results.csv
  phase3_results.jsonl
  train_N256_seed0/
  train_N256_seed1/
  train_N256_seed2/
  train_N512_seed0/
  ...
```

每个 run 子目录必须包含：

```text
summary.json
results.csv
*_metrics.jsonl
command.sh
config_snapshot.json
```

### 5.5 结果字段

主结果表必须至少包含：

```text
task
method
seed
N_train
N_eval
B
d
G_type
H_type
architecture
layers
d_model
heads
ffn_dim
steps
batch_size
eval_batches
learning_rate
raw_K
effective_K_mean
attention_pair_count
final_train_loss
eval_loss
eval_accuracy
tokens_per_sec
peak_allocated_gb
peak_reserved_gb
artifact_dir
git_commit
```

### 5.6 通过条件

Phase 3 完成标准：

```text
1. N_train=256/512/1024 全部完成；
2. dense/local/random/zigzag 全部完成；
3. 每个训练 run 都评测到 N_eval=2048；
4. 所有 seeds 都按 config 完成，或失败原因明确记录；
5. 主结果 CSV/JSONL 完整；
6. 本地和远端结果同步；
7. git commit 完成。
```

## 6. 分析口径

### 6.1 允许的结论

完成 Phase 3 后，允许回答：

```text
copy task 上，zig-zag 是否比 local-only 更能传递长程信息；
同预算下 zig-zag 与 random 的稳定性差异；
训练长度 N_train 对 N_eval=2048 泛化的影响；
不同方法在同一配置下的显存和速度差异。
```

### 6.2 不允许的结论

在本手册范围内，不允许声称：

```text
official LRA benchmark 结果；
真实文本任务效果；
大模型 scaling law；
通用 sparse attention superiority；
最终 CUDA/block-sparse kernel 性能。
```

若报告中提到 blockpair 或其它后端，只能写为当前实现细节或 prototype，不得写成最终系统优化结论。

## 7. 推荐执行清单

### 7.1 v05-doc

```text
归档旧报告到 ref/archive_v04_reports/
新增 ref/zigzag_experiment_execution_manual_v05.md
确认当前目录 git 状态
提交 git commit
```

### 7.2 v05-phase1

```text
创建 configs/copy_v05_smoke.json
创建 configs/copy_v05_main.json
创建 scripts/graph_structures.py
让 synthetic_mvp.py 支持 --config
本地 smoke
远端同步和 smoke
同步结果回本地
git commit
```

### 7.3 v05-phase2

```text
确认 copy 在线生成只依赖 seed/config
运行 4 方法 smoke
检查 outputs/copy_v05_smoke_gpu/
同步结果
git commit
```

### 7.4 v05-phase3

```text
运行 N_train=256
评测 N_eval=256/512/1024/2048
运行 N_train=512
评测 N_eval=256/512/1024/2048
运行 N_train=1024
评测 N_eval=256/512/1024/2048
合并 phase3_results.csv/jsonl
写阶段性实验总结
git commit
```

## 8. 当前已知注意事项

1. 当前根目录在检查时不是 git 仓库；开始 v0.5 实验前必须初始化或绑定 git 仓库。
2. 旧 LRA 相关脚本可以暂时保留，但不得作为 v0.5 结果来源。
3. 旧 `outputs/` 中的结果只能作为历史参考，不得混入 v0.5 主结果表。
4. `reports/user_experiment_report_1.docx` 是阶段性人工订正报告，不属于旧 phase 报告归档范围，除非用户明确要求移动。
5. 每个 phase 的实际执行记录应新建独立报告，不要继续覆盖旧 v04 报告。
