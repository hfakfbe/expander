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
eval token accuracy
eval sequence accuracy
eval EOS accuracy
tokens/sec
peak allocated memory
peak reserved memory
training curve image path
```

所有结果表保存为 CSV 和 JSONL，不在正文中手工维护超宽表。

## 2. 研究对象和方法定义

### 2.1 Attention Pattern

attention 实际作用在长度为 `T` 的 token 序列上。该序列被划分为 `q=T/B` 个 block，每个 block 包含 `B` 个 token。对于 full-copy 任务，`N` 表示源序列长度，实际 attention 长度是包含 `SEP`、copy 输出段、`EOS` 和 padding 后的 `T`。

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
dense K = T
```

当前 v0.5 实验明确使用 causal 自回归 attention。若用户或报告中写到 casual，应统一理解为 causal mask 设定；本手册当前固定为：

```text
causal = true
```

causal 规则是：位置 `t` 的 query 只能看 `key_position <= t` 的 token，不能看未来 token。对 sparse 方法，先构造 dense/local/random/zigzag 的结构 mask，再与 causal lower-triangular mask 相交。所有方法必须报告：

```text
raw K before causal filtering
effective K after causal filtering
attention pair count after causal filtering
```

因此，copy 输出区间中的 token 可以看见前面的源序列和分隔符，但不能看见未来尚未预测的目标 token。

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

### 2.3 Copy 实验输入输出

v0.5 使用 causal full-copy 任务，不再使用“最后一个 token 预测 `x0`”的分类任务。

每个样本先在线生成源序列：

```text
s = [s0, s1, ..., sN-1]
s_i ∈ {1, 2, ..., num_values}
```

第一轮主实验固定：

```text
num_values = 4
s_i ∈ {1, 2, 3, 4}
```

使用特殊 token：

```text
PAD = 0
SEP = num_values + 1
EOS = num_values + 2
```

模型输入序列为：

```text
z = [s0, s1, ..., sN-1, SEP, s0, s1, ..., sN-1, EOS]
```

这里 `N` 表示需要复制的源序列长度，不是实际送入模型的总长度。实际序列长度为：

```text
T_raw = 2N + 2
```

为了兼容 block attention，实际 attention 长度必须 pad 到 `B` 的倍数：

```text
T = ceil(T_raw / B) * B
```

padding 只用于对齐，不参与 loss 和 accuracy。

训练使用 teacher forcing 的 causal LM 目标：位置 `t` 的 logits 预测 `z[t+1]`。loss 只计算 copy 输出区间和 EOS：

```text
loss_positions = [N, N+1, ..., 2N]
targets        = [s0, s1, ..., sN-1, EOS]
```

也就是说，模型在看到完整源序列和 `SEP` 后，需要自回归地输出完整源序列，并在末尾输出 `EOS`。

评估必须至少报告：

```text
token accuracy: copy 输出区间和 EOS 上的逐 token accuracy
sequence accuracy: 整个复制序列和 EOS 全部正确的样本比例
eos accuracy: EOS 位置 accuracy
loss: copy 输出区间和 EOS 上的 cross entropy
```

训练和评估都在线生成数据，不保存静态数据集。训练 copy 长度用 `N_train` 表示，评估 copy 长度用 `N_eval` 表示。`N_eval > N_train` 的结果必须标注为 extrapolation。

### 2.4 训练动态曲线

v0.5 不只记录最终 accuracy，还必须记录训练动态。每个 method / seed / N_train 至少输出：

```text
train loss vs step
eval loss vs step
eval token accuracy vs step
eval sequence accuracy vs step
eval EOS accuracy vs step
tokens/sec vs step 或 measured interval
```

曲线数据来自训练过程中的结构化日志，不允许从终端文本手工摘抄。推荐每次在以下 step 记录一次：

```text
step = 1
step % log_every == 0
step = final step
```

第一轮主实验推荐：

```text
log_every = 250
```

训练动态曲线必须直接通过 `matplotlib` 生成图片，不以“保存 CSV 后人工画图”作为通过条件。每个 run 至少保存：

```text
training_curves.png
```

`training_curves.png` 至少包含：

```text
train loss vs step
eval loss vs step
eval token accuracy vs step
eval sequence accuracy vs step
```

服务器无显示环境时必须使用非交互 backend，例如：

```python
import matplotlib
matplotlib.use("Agg")
```

结构化 metrics 仍然可以写入 JSONL 以便复查，但 matplotlib 生成的图片是 phase 通过条件。

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
    "mode": "full_copy",
    "num_values": 4,
    "special_tokens": {
      "pad": 0,
      "sep": 5,
      "eos": 6
    },
    "train_lengths": [256, 512, 1024],
    "eval_lengths": [256, 512, 1024, 2048]
  },
  "model": {
    "architecture": "transformer",
    "layers": 8,
    "d_model": 128,
    "heads": 4,
    "ffn_dim": 256,
    "dropout": 0.1,
    "attention_backend": "auto_split"
  },
  "attention": {
    "methods": ["dense", "local", "random", "zigzag"],
    "causal": true,
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
    "log_every": 250,
    "seeds": [0, 1, 2]
  },
  "output": {
    "root": "outputs/copy_v05_main",
    "plot_curves": true,
    "curve_format": "png"
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
7. attention config 必须显式写入 `causal=true`，并在 summary 中记录。
8. metrics 日志必须足以通过 matplotlib 直接生成训练动态曲线图片。

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
源序列:  s = [s0, s1, ..., sN-1]
取值:    s_i ∈ {1, 2, 3, 4}
输入:    z = [s, SEP, s, EOS]
目标:    causal LM next-token prediction
loss:    只计算第二段 s 和 EOS
causal: true
```

随机逐 token accuracy 约为 `1/(num_values+2)`。核心评估对象不是单 token 分类，而是完整复制段的 token accuracy、sequence accuracy 和 EOS accuracy。

### 4.2 数据生成规范

在线数据生成必须满足：

```text
同一个 seed 下可复现
train/eval 使用同一生成规则
eval batch 不复用训练 batch
num_values 可配置
N 可配置，表示源序列长度
SEP/EOS/PAD token 固定或由 config 明确给出
不读取外部数据文件
```

建议生成函数只依赖：

```text
seed
batch index / step
N
num_values
special tokens
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
loss、token accuracy、sequence accuracy、EOS accuracy 可记录
matplotlib 能生成训练动态曲线图片
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
log_every = 250
seeds = config 指定，默认 [0, 1, 2]
architecture = transformer
layers = 8
d_model = 128
heads = 4
ffn_dim = 256
dropout = 0.1
attention_backend = auto_split
causal = true
plot_curves = true
```

如需改动以上参数，必须新建 config，不得覆盖已完成实验的 config。

### 5.3 实验矩阵

主表最小矩阵：

| N_train | method | seeds | eval N |
|---|---|---|---|
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
training_curves.png
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
T_raw
T
B
d
G_type
H_type
causal
copy_mode
architecture
layers
d_model
heads
ffn_dim
steps
batch_size
eval_batches
learning_rate
log_every
raw_K
effective_K_mean_after_causal
attention_pair_count_after_causal
final_train_loss
eval_loss
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
training_curves_path
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
6. 每个 run 都直接生成 matplotlib 训练动态曲线图片；
7. 本地和远端结果同步；
8. git commit 完成。
```

## 6. 分析口径

### 6.1 允许的结论

完成 Phase 3 后，允许回答：

```text
causal full-copy task 上，zig-zag 是否比 local-only 更能复制长序列；
同预算下 zig-zag 与 random 的稳定性差异；
训练长度 N_train 对 N_eval=2048 泛化的影响；
不同方法在同一配置下的显存和速度差异。
causal full-copy 中 token accuracy、sequence accuracy、EOS accuracy 的差异。
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

## 7. 实验环境与服务器规范

### 7.1 服务器

远端实验默认在公用 A100 服务器上运行：

```text
host alias: huiwei
remote root: /home/huiwei/ysx/zigzag_attention
conda env: ysx_base
GPU: 4 x NVIDIA A100-SXM4-80GB
```

每次远端运行前必须检查 GPU 使用情况：

```bash
ssh huiwei
nvidia-smi
```

4 张 A100 都可以使用。启动任务的规则是：目标 GPU 当前利用率低于 `5%` 时，可以在该 GPU 上启动任务。若目标 GPU 利用率大于等于 `5%`，或显存被未知进程大量占用，或有未知训练进程，不启动实验。正式记录中必须写明 `CUDA_VISIBLE_DEVICES`、启动前 `nvidia-smi` 状态和实际 GPU 名称。

推荐选择 GPU 的检查命令：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

若多张 GPU 都低于 `5%`，优先选择显存占用更低的 GPU。单个实验默认只占用一张 GPU；只有后续明确加入多 GPU 训练时，才允许同时设置多个 `CUDA_VISIBLE_DEVICES`。

### 7.2 本地与远端目录

本地是唯一的编辑主目录：

```text
/Users/sxye/Documents/expander
```

远端是运行主目录：

```text
/home/huiwei/ysx/zigzag_attention
```

代码修改优先在本地完成，通过 rsync 同步到远端。远端只用于运行、查看日志和必要的小修复；若远端发生小修复，必须同步回本地并进入 git。

推荐同步命令：

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude 'cached_graphs/' \
  ./ huiwei:/home/huiwei/ysx/zigzag_attention/
```

结果同步命令：

```bash
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/ ./outputs/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/logs/ ./logs/
```

注意：`outputs/`、`logs/` 和 `cached_graphs/` 不纳入 git 主提交，但必须保存在本地磁盘，作为报告数值来源。

### 7.3 Conda 环境

远端默认使用：

```bash
conda activate ysx_base
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

每个大版本至少保存一次环境快照：

```bash
python -V
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pip freeze > envs/requirements_snapshot.txt
```

如果环境中缺少包，不直接污染公共环境；优先使用项目内兼容 shim 或新建私有环境，并把原因写入 phase 报告。

### 7.4 运行方式

短 smoke 可以直接运行。超过数分钟的正式实验必须使用 `tmux` 或 `screen`：

```bash
tmux new -s copy_v05
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/synthetic_mvp.py \
  --config configs/copy_v05_main.json \
  --output-dir outputs/copy_v05_main
```

日志必须同时写入文件：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/synthetic_mvp.py \
  --config configs/copy_v05_main.json \
  --output-dir outputs/copy_v05_main \
  2>&1 | tee logs/copy_v05_main_$(date +%Y%m%d_%H%M%S).log
```

### 7.5 断点与恢复

Phase 1/2 smoke 不要求 checkpoint resume。Phase 3 主实验若单个 run 时间较长，必须支持或至少规划以下恢复策略：

```text
每个 N_train / seed / method 独立输出目录；
已完成 run 不重复覆盖；
失败 run 记录 error.log；
合并结果时跳过未完成 run 并标记 status=failed；
重新运行只补缺失项。
```

如实现 checkpoint，checkpoint 必须包含：

```text
model state
optimizer state
step
seed / RNG state
config snapshot
method
N_train
```

### 7.6 远端运行前检查清单

每次正式运行前检查：

```text
git status 本地干净或仅含本次明确改动；
当前 commit hash 已记录；
本地代码已 rsync 到远端；
远端 configs/scripts/ref 文件存在；
nvidia-smi 确认目标 GPU 利用率低于 5%；
输出目录不会覆盖已有有效结果；
日志路径已设置；
config snapshot 会写入输出目录。
```

## 8. 实验记录规范

### 8.1 文件组织

v0.5 新实验记录放在：

```text
reports/
  copy_v05_phase1_report.md
  copy_v05_phase2_smoke_report.md
  copy_v05_phase3_main_report.md
```

运行产物放在：

```text
outputs/copy_v05_phase1_smoke/
outputs/copy_v05_smoke_gpu/
outputs/copy_v05_main/
logs/
```

旧 v04 报告只保存在：

```text
ref/archive_v04_reports/
```

不得把旧结果复制到 v05 主表中。

### 8.2 每次运行必须记录

每次运行至少记录以下字段：

```text
run_id
timestamp
host
local_or_remote
git_commit
config_path
config_sha256
command
output_dir
log_path
CUDA_VISIBLE_DEVICES
gpu_name
torch_version
task
data_mode
num_values
copy_mode
sep_token
eos_token
pad_token
method
attention_backend
N_train
N_eval
T_raw
T
B
d
G_type
H_type
causal
seed
architecture
layers
d_model
heads
ffn_dim
dropout
optimizer
learning_rate
log_every
steps
batch_size
eval_batches
raw_K
effective_K_mean_after_causal
effective_K_min_after_causal
effective_K_max_after_causal
duplicate_rate
self_loop_rate
attention_pair_count_after_causal
final_train_loss
eval_loss
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
training_curves_path
tokens_per_sec
elapsed_sec
peak_allocated_gb
peak_reserved_gb
status
failure_reason
```

### 8.3 Config Snapshot

每个输出目录必须保存运行时 config：

```text
config_snapshot.json
```

如果命令行覆盖了 config 中的字段，必须在 snapshot 中体现最终实际值。报告中引用的参数以 snapshot 为准，不以记忆或命令草稿为准。

### 8.4 Command 记录

每个 run 子目录必须保存：

```text
command.sh
```

`command.sh` 必须可以直接复现本次运行，至少包含：

```bash
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/synthetic_mvp.py ...
```

### 8.5 结果表

Phase 3 必须产出：

```text
outputs/copy_v05_main/phase3_results.csv
outputs/copy_v05_main/phase3_results.jsonl
```

CSV 用于人工检查和报告制表；JSONL 用于后续脚本读取。两者必须来自同一份 run records，不允许手工分别维护。训练动态曲线必须由脚本使用 matplotlib 直接生成 `training_curves.png`，并在结果表中记录图片路径。

### 8.6 报告写法

阶段报告必须先写事实，再写解释。推荐结构：

```text
目标
配置
运行环境
命令
结果表
通过/失败项
解释
下一步
```

报告中的结论必须使用限定语：

```text
在 copy online synthetic task 上
在当前 B=16,d=2,G=cyclic,H=cycle 配置下
在 causal full-copy 设定下
在 seeds=[...] 内
在 N_train -> N_eval 的设定下
```

不得把 smoke test 写成主实验结论，也不得把单 seed 结果写成稳定性结论。

### 8.7 失败记录

失败不是删除重跑。每次失败必须保留：

```text
error.log
failed command
config snapshot
失败发生的 step
是否占用 GPU
初步原因判断
是否需要改代码
```

失败结果在总表中保留 `status=failed`，除非确认是启动命令写错且没有进入训练。

## 9. 推荐执行清单

### 9.1 v05-doc

```text
归档旧报告到 ref/archive_v04_reports/
新增 ref/zigzag_experiment_execution_manual_v05.md
确认当前目录 git 状态
提交 git commit
```

### 9.2 v05-phase1

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

### 9.3 v05-phase2

```text
确认 copy 在线生成只依赖 seed/config
运行 4 方法 smoke
检查 outputs/copy_v05_smoke_gpu/
同步结果
git commit
```

### 9.4 v05-phase3

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

## 10. 当前已知注意事项

1. 当前根目录已经初始化为 git 仓库，v05 文档归档版本提交为 `4cbb945`。
2. 旧 LRA 相关脚本可以暂时保留，但不得作为 v0.5 结果来源。
3. 旧 `outputs/` 中的结果只能作为历史参考，不得混入 v0.5 主结果表。
4. `reports/user_experiment_report_1.docx` 和对应 PDF 已归档到 `ref/archive_v04_reports/user_report/`。
5. 每个 phase 的实际执行记录应新建独立报告，不要继续覆盖旧 v04 报告。
