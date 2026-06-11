# Expander/Zig-Zag Sparse Attention 实验执行手册 v0.4

## 0. 文档定位

本文档只服务于一个目标：**指导在 4 张 A100 公用服务器条件下，按严格顺序完成 expander / zig-zag sparse attention 的实验验证**。它不是综述，不记录文档修改历史，不展开暂时无法执行的大规模、多模态或生产级 CUDA 优化计划。

本项目的实验入口不是“直接写新模型”，而是：

1. 先确认是否存在可以复用的仓库；
2. 选一个最适合作为 base 的仓库；
3. 在 base 上跑通评测任务和 baseline；
4. 再实现本文提出的 local dense block + zig-zag / expander cross-block sparse attention；
5. 先做小规模试训练；
6. 再做针对新模型稀疏结构的 batch / layout / graph 级调优；
7. 最后做完整训练、评测、调参、消融和小规模 scaling law 测试。

## 1. 总体原则：严格流水线，不跳步

整个实验按 7 个阶段推进。每个阶段都有准入条件、操作、输出物和停止条件。没有完成当前阶段，不进入下一阶段。

| 阶段 | 名称 | 目标 | 通过条件 |
|---|---|---|---|
| 1 | 仓库可用性评估 | 找到可复用 base 仓库 | 至少 1 个候选仓库能在服务器跑通 smoke test |
| 2 | 准备评测任务 | 固定任务、数据、指标 | 数据能加载，baseline 训练脚本能启动 |
| 3 | base 训练评测 | 建立可复现 baseline | base 结果、显存、速度全部记录 |
| 4 | 新模型试训练评测 | 验证 zig-zag attention 正确且能训练 | 小规模 loss/accuracy 正常下降 |
| 5 | 新模型性能调优 | 针对稀疏边结构优化 batch/layout | 在长序列上显存或速度有可观改善 |
| 6 | 完整实验与消融 | 系统比较方法和参数 | 主表、消融表、效率表完成 |
| 7 | scaling law 测试 | 在小算力下研究扩展趋势 | 得到随 N/P/K/compute 的趋势图 |

禁止跳步。例如：没有完成 base 的训练评测，不做新模型主实验；没有完成新模型试训练，不做 scaling law；没有确定一个可复用 base，不开始大规模重写。

## 2. 研究对象最小解释

### 2.1 dense attention 的问题

标准 self-attention 可以看作 token 间的完全图。若序列长度为 `N`，每层 attention pair 数约为：

```text
N^2
```

当 `N` 增大到 4096、8192、16384 时，显存和计算会迅速增长。因此实验目标是构造一个 sparse attention mask，让每个 token 只看少量关键 token，同时尽量保持长程通信能力。

### 2.2 当前模型思路

把长度为 `N` 的 token 序列分成 `q` 个局部 block，每个 block 有：

```text
B = m * n
```

个 token。这里 `m*n` 只是表示局部块大小；对文本序列可直接理解为 `B = block_size`，不需要真的使用二维图像结构。

每层 mask 由两部分组成：

```text
local dense edges: 每个 block 内全连接
zig-zag cross-block edges: block 之间的稀疏连接
```

若 zig-zag 小图 `H` 的度数为 `d`，则每个 token 的跨块邻居数为：

```text
d^2
```

所以每个 token 大约看：

```text
K = B + d^2
```

个 key。理论 attention pair 数从 `N^2` 降为：

```text
N * (B + d^2)
```

当 `B` 和 `d` 固定时，这是关于 `N` 的线性复杂度。

### 2.3 G 和 H 的实验含义

在 zig-zag product 中：

- `G` 是 block-level 大图；一个顶点代表一个 block。
- `H` 是 port-level 小图；一个顶点代表 block 内的一个端口/token 位置。
- product 后的顶点 `(v, i)` 才是真正的 token，其中 `v` 是 block id，`i` 是 block 内 token/port id。

注意：

```text
local dense block 不是 H。
H 只用于 zig-zag 的端口扰动。
```

`G` 的边必须明确给出，不能只说“每个点度数为 B”。真正的连边由旋转映射决定：

```text
Rot_G(v, i) = (w, j)
```

它表示：从 block `v` 的 port `i` 跳到 block `w` 的 port `j`。

## 3. 阶段 1：仓库可用性评估

### 3.1 目标

先判断能否复用已有仓库，尽量减少从零实现训练框架、数据管线、日志系统和 baseline 的工作量。

本阶段不写新模型，只做仓库调查和 smoke test。

### 3.2 候选仓库优先级

优先考虑 PyTorch 仓库，因为后续修改 attention 模块更直接，也更符合现有服务器实验习惯。

| 优先级 | 候选仓库/资源 | 用途 | 初步判断 |
|---|---|---|---|
| A | `guy-dar/lra-benchmarks` | PyTorch/HuggingFace 风格 LRA 任务框架 | 最优先尝试作为 base |
| B | `google-research/long-range-arena` | 官方 LRA 参考实现 | JAX/Flax，适合作任务定义参考，不优先作为 base |
| C | Hugging Face BigBird | BigBird baseline 和 block sparse 思路 | 可作为 baseline 参考 |
| D | `google-research/bigbird` | 官方 BigBird 代码 | 参考 sparse attention 结构，不一定作为主 base |
| E | `facebookresearch/xformers` | memory-efficient / sparse / block-sparse attention 后端 | 后期性能调优参考 |
| F | `hamed1375/Exphormer` | expander sparse Transformer 思路 | 参考 expander 设计，不作为 sequence base |
| G | OpenAI sparse_attention | fixed/strided sparse attention 与 block 化经验 | 参考历史实现，不作为主 base |
| H | Longformer | sliding window + global attention baseline | 参考 local/global sparse pattern |

资料依据：LRA 官方仓库说明其目标是系统评测 efficient Transformer 的泛化能力、计算效率和显存占用；`guy-dar/lra-benchmarks` 说明其基于 Google LRA 数据并使用 PyTorch/HuggingFace 以便扩展；BigBird 采用 local/random/global sparse pattern；xFormers 提供 PyTorch optimized Transformer components；Exphormer 使用 expander graphs、actual graph edges 和 universal connectors 作为 graph sparse Transformer 的组成部分。

### 3.3 仓库评估标准

每个候选仓库按以下标准打分。

| 维度 | 要求 | 不通过条件 |
|---|---|---|
| 安装成本 | 1 天内能在服务器装好 | 依赖过旧、无法编译、CUDA 冲突严重 |
| 任务可用 | 至少能跑 1 个目标任务 | 数据管线不可用或下载困难 |
| 模型可改 | attention 模块位置清晰 | 模型结构高度封装，难以替换 attention |
| 训练可控 | 能指定 batch、seq len、seed、device | 配置混乱，无法复现实验 |
| 资源适配 | 单卡 A100 可跑 smoke test | 默认必须多机或占满 4 卡 |
| 日志完整 | 能记录 loss、accuracy、速度 | 需要大量手工补日志 |

### 3.4 Smoke test 要求

对每个候选 base，执行最小命令：

```bash
ssh huiwei
cd ~/ysx/
conda activate ysx_base
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_gpu_id> python <train_or_eval_script> --small_config
```

Smoke test 只要求：

```text
1. 程序能启动；
2. 数据能加载；
3. forward/backward 能完成；
4. 训练至少 100 step 不崩；
5. 日志能保存；
6. 不占用别人正在使用的 GPU。
```

### 3.5 阶段输出物

本阶段结束时必须产出：

```text
repo_survey.md
base_repo_decision.md
env_snapshot.yaml
smoke_test_log.txt
```

`base_repo_decision.md` 必须明确写出：

```text
选择哪个仓库作为 base；
为什么选择它；
哪些仓库被排除；
后续修改 attention 模块的位置；
base commit hash；
smoke test 命令和结果。
```

### 3.6 停止条件

若没有任何候选仓库能在 2 个工作日内跑通 smoke test，则进入 fallback：基于 PyTorch 写最小训练框架，但只支持 synthetic task 和一个简单文本任务。fallback 不是首选，只有仓库复用失败才启动。

## 4. 阶段 2：准备评测任务

### 4.1 任务范围

受限于 4 张 A100 且是公用服务器，本项目不做多模态、不做高分辨率视觉、不做大规模语言模型预训练。LRA 中的 image/pathfinder 类任务暂不作为主线。

主线任务只保留：

```text
合成序列任务 + 文本/符号长序列任务
```

### 4.2 第一批任务

| 优先级 | 任务 | 类型 | 目的 | 是否进入第一轮 |
|---|---|---|---|---|
| A | Associative Recall | synthetic | 测试长程 key-value 检索 | 是 |
| A | Delayed Copy / Copy | synthetic | 测试远程复制和信息保持 | 是 |
| A | LRA ListOps | 符号序列 | 测试层级结构和长序列处理 | 是 |
| B | LRA Text | 文本分类 | 测试文本场景泛化 | base 跑通后加入 |
| B | LRA Retrieval / AAN | 文本匹配 | 测试跨段匹配 | base 稳定后加入 |
| C | LRA Image / Pathfinder | 图像/视觉式任务 | 非主线 | 暂不做 |

### 4.3 任务准备原则

每个任务必须固定：

```text
数据版本
train/valid/test split
最大序列长度 N
padding/truncation 规则
metric
batch size 规则
seed
```

不要在同一阶段频繁换任务。先用 1 个 synthetic + 1 个 LRA 任务跑通完整流程，再扩展任务数量。

### 4.4 任务验收

任务准备阶段完成的标准：

```text
1. 数据能自动加载；
2. 能打印样本 shape；
3. 能跑 base model 的 forward；
4. 评价脚本能输出指标；
5. 所有任务配置写入 YAML/JSON；
6. 本地和远程代码一致。
```

## 5. 阶段 3：base 训练与评测

### 5.1 目标

在不引入新模型的情况下，先建立可信 baseline。此阶段回答：

```text
base 仓库能否复现基本训练；
当前服务器上 dense / local / BigBird-style 等方法的速度和显存是多少；
后续新模型至少要超过哪些基准。
```

### 5.2 base 方法

优先训练以下 baseline：

| 方法 | 作用 | 是否必须 |
|---|---|---|
| Dense Transformer | 质量上界，小 N 参考 | 必须，若 N 太大可只做小 N |
| Local-only attention | 检查跨块边是否有用 | 必须 |
| Local + random same-budget | 检查 zig-zag 是否优于随机边 | 必须 |
| BigBird-style | local/random/global 稀疏基线 | 仓库支持则必须 |
| Longformer-style | sliding window + global | 可选 |

### 5.3 第一轮 base 配置

第一轮不要追求大规模。建议固定：

```text
N: 1024, 2048, 4096
model_dim: 256
num_layers: 4
num_heads: 4
ffn_dim: 1024
dropout: 0.1
optimizer: AdamW
learning_rate: 3e-4
batch_size: 按显存调整，但同任务内保持规则一致
seeds: 0, 1, 2 中先跑 seed 0，稳定后补全
```

### 5.4 记录指标

base 每次训练必须记录：

```text
final train loss
best valid metric
test metric, if available
step time
tokens/sec
peak allocated memory
peak reserved memory
effective batch size
GPU id
commit hash
config file path
log file path
```

### 5.5 阶段通过标准

满足以下条件才进入新模型试训练：

```text
1. 至少 1 个 synthetic 任务和 1 个 LRA 任务有 base 结果；
2. local-only 和 random same-budget 结果可用；
3. 性能日志完整；
4. 所有实验命令可复现；
5. base 仓库修改保持最小。
```

## 6. 阶段 4：新模型试训练与评测

### 6.1 目标

本阶段只验证新模型是否正确、是否能训练，不追求最终最优性能。

新模型为：

```text
local dense block attention + zig-zag / expander cross-block sparse attention
```

### 6.2 实现顺序

新模型实现也必须按顺序：

1. `dense mask debug`：直接构造完整 `N x N` mask，只用于小 N 正确性验证；
2. `neighbor list attention`：构造 `[N, K]` 邻居表，开始实现真实稀疏计算；
3. `local + cross split`：local dense 与 cross sparse 分开计算，但 softmax 必须等价于 union mask；
4. `cached graph`：固定 G/H，预计算并缓存邻居表；
5. `layer-wise graph variant`：后续再考虑不同层使用不同 G/H。

不要一开始写自定义 CUDA kernel。

### 6.3 mask 正确性单元测试

至少测试：

```text
N = 64, 128
B = 8, 16
d = 2, 3
```

必须检查：

```text
1. 每个 token 的 raw K 是否为 B + d^2；
2. 去重后 effective K；
3. 是否有非法 index；
4. 是否有跨样本连接；
5. G 的 Rot_G 是否满足反向一致性；
6. H 的 degree 是否正确；
7. dense mask 版本和 neighbor list 版本在小 N 下输出接近。
```

### 6.4 第一批新模型参数

试训练只跑小网格：

```text
B = 16
d = 2, 3, 4
N = 1024, 2048, 4096
G = cyclic, random permutation
H = cycle, random regular
layers = 4
```

第一阶段不跑 `B=64`、不跑 `N=16384`、不跑全任务。

### 6.5 公平性规则

所有 sparse 方法必须报告：

```text
raw K
effective K
duplicate rate
self-loop rate
attention pair count
```

如果 causal mask 后续加入，还要报告 causal filtering 后的 effective K。本阶段默认 non-causal encoder-style attention，不做 causal LM。

### 6.6 阶段通过标准

进入性能调优前，新模型必须满足：

```text
1. 小 N 单元测试通过；
2. 至少一个任务 loss 正常下降；
3. 训练没有 NaN；
4. 同预算下不明显弱于 local-only；
5. attention pair count 与理论一致；
6. 训练日志和配置完整。
```

若新模型不如 local-only，先暂停调优，回到 mask 和任务诊断。

## 7. 阶段 5：新模型性能调优

### 7.1 本阶段调优范围

这里的“性能调优”不是优先写 CUDA，也不是盲目微调 learning rate，而是针对新模型的稀疏边结构做计算组织优化。

核心问题是：

```text
新模型边很稀疏，很多 block pair 没有边。
如果仍按 dense block 或随机 gather 计算，会浪费大量计算或造成 GPU 访存效率差。
需要找到更适合 sparse edge pattern 的 batching/layout。
```

### 7.2 相关工作的可借鉴点

| 相关工作/实现 | 可借鉴点 | 对本项目的启发 |
|---|---|---|
| Sparse Transformer | fixed/strided pattern 可按 block slicing 计算 | 尽量把稀疏边整理成规则 block group |
| Longformer | local window + global attention，线性复杂度 | local 部分应走高效块计算，不要随机 gather |
| BigBird | local + random + global 的 block sparse pattern | baseline 和 same-budget random 设计参考 |
| FlashAttention | IO-aware tiling，避免显式 materialize 全 score | 先用 tiling 思维组织 local block attention |
| xFormers | optimized ops 和 attention bias 抽象 | 作为后端或工程参考，不一定强依赖 |
| Exphormer | expander edges + local neighborhoods + connectors | expander 稀疏图的实验设计参考 |

### 7.3 调优方向一：local 和 cross 分离，但保持统一 softmax

local 部分：

```text
X -> [batch, q, B, dim]
对每个 block 内 B x B 做 dense attention
```

cross 部分：

```text
每个 token gather d^2 个跨块 key/value
```

注意：不能分别对 local 和 cross 做 softmax 后相加。正确做法是对两部分 logits 做 log-sum-exp 合并，使结果等价于在 `B + d^2` 个 key 上一次 softmax。

### 7.4 调优方向二：按 block pair 聚合 cross edges

随机 gather 的问题是访存不连续。改进方式：

```text
把 cross edges 从 token-level 整理为 block-pair-level。
例如 source block u 到 target block v 之间有若干条 token 边，统一成一个小 batch 计算。
```

记录每层中实际存在的 block pair：

```text
(source_block, target_block) -> list[(source_token, target_token)]
```

这样可以：

```text
减少无边 block pair 的计算；
让同一 target block 的 K/V 尽量连续读取；
便于未来接 block-sparse backend。
```

### 7.5 调优方向三：边排序与 bucket

对 cross edges 排序：

```text
按 source block 排序；
再按 target block 排序；
再按 source token 排序。
```

如果不同 token 的 effective K 不一致，则按 effective K 分 bucket，避免为了少数 token padding 大量无效边。

### 7.6 调优方向四：图缓存

默认固定 G/H 并缓存：

```text
neighbors.pt
edge_index.pt
block_pair_index.pt
metadata.json
```

每层复用同一图，先减少工程变量。只有当固定图表现不佳时，再测试 layer-wise resampling。

### 7.7 调优方向五：token 重排

若 zig-zag 边导致大量随机访存，可以尝试对 block 或 token 重新编号，使高频连接的 target block 在内存上更接近。

本阶段只做简单版本：

```text
按 G 的 BFS/order 或 cyclic order 重排 block id；
比较重排前后的 tokens/sec 和 cache behavior。
```

### 7.8 性能测试协议

每个性能实验必须：

```text
warmup: 20 steps
measure: 100 steps
每次计时前后 torch.cuda.synchronize()
显存: torch.cuda.reset_peak_memory_stats(), max_memory_allocated(), max_memory_reserved()
记录 batch size、seq len、dtype、GPU id
至少重复 3 次，报告 mean/std
```

### 7.9 调优通过标准

进入完整实验前，至少满足一个条件：

```text
1. 在 N >= 4096 时显存明显低于 dense；
2. 在 N >= 4096 时 tokens/sec 高于 dense 或接近 dense 但支持更大 N；
3. 在同等 pair budget 下速度不明显差于 random sparse；
4. 若速度暂时不优，必须有明确 profile 证明瓶颈在哪里。
```

## 8. 阶段 6：调优后完整训练、评测、调参、消融

### 8.1 主实验方法

完整实验比较：

```text
Dense Transformer
Local-only
Local + random same-budget
BigBird-style, if available
Zig-zag sparse attention
Tuned zig-zag sparse attention
```

### 8.2 主实验任务

主表只包含已稳定的任务：

```text
Associative Recall
Delayed Copy
LRA ListOps
LRA Text 或 Retrieval/AAN（二选一，视 base 稳定性）
```

不加入图像、Pathfinder、多模态任务。

### 8.3 调参顺序

调参仍按顺序，不全网格暴力搜索：

1. 固定 `B=16, layers=4`，扫 `d=2,3,4`；
2. 固定最优 `d`，扫 `B=8,16,32`；
3. 固定 `B,d`，扫 `G`；
4. 固定 `G`，扫 `H`；
5. 固定结构后，调学习率和 dropout；
6. 最后补 3 个 seed。

### 8.4 消融实验

必须消融：

| 消融项 | 对比 |
|---|---|
| 跨块边是否有用 | local-only vs zig-zag |
| zig-zag 是否优于随机 | random same-budget vs zig-zag |
| G 的影响 | cyclic G vs random permutation G vs random regular G |
| H 的影响 | cycle H vs random regular H vs complete H 反例 |
| degree 的影响 | d=2/3/4 |
| block size 的影响 | B=8/16/32 |
| 固定图 vs 层间变化 | fixed G/H vs layer-wise resampled G/H |
| 调优是否有效 | naive zig-zag vs tuned zig-zag |

### 8.5 结果表模板

结果表建议保存为 CSV/JSONL，不建议在正文中维护超宽表。字段如下：

```text
task, method, N, B, d, K_eff, params, valid_metric, test_metric, memory_gb, tokens_per_sec
ListOps, Local, 2048, 16, -, 16, ..., ..., ..., ..., ...
ListOps, Random, 2048, 16, 3, 25, ..., ..., ..., ..., ...
ListOps, Zig-zag, 2048, 16, 3, 25, ..., ..., ..., ..., ...
```

### 8.6 阶段通过标准

完整实验完成标准：

```text
1. 每个主任务至少有 3 个核心方法结果；
2. 每个主方法至少有 1 个 seed，最终模型补 3 seeds；
3. 主表、消融表、效率表完整；
4. 所有 checkpoint、config、log 可追溯；
5. 能明确回答新模型相对 local/random/base 的优势和不足。
```

## 9. 阶段 7：scaling law 测试

### 9.1 本项目中的 scaling law 定义

这里不做大模型 scaling law，也不声称能复现 Kaplan 或 Chinchilla 级别规律。受限于 4 张 A100，本项目只做小规模经验扩展趋势：

```text
当 sequence length、model size、attention budget 或训练 compute 增大时，新模型和 baseline 的 loss/accuracy/efficiency 如何变化。
```

相关 scaling law 工作通常研究 loss 随 model size、data size、compute 的幂律变化；Chinchilla 进一步研究固定 compute 下模型大小和训练 token 数的分配。本文只借鉴其实验形式：多尺度训练、记录 compute、拟合趋势，不扩大到不可承受的规模。

### 9.2 scaling 轴

只测试 4 个轴：

| 轴 | 取值 | 目的 |
|---|---|---|
| 序列长度 N | 1024, 2048, 4096, 8192, 可选 16384 | 看长序列扩展能力 |
| 模型规模 P | tiny, small, base | 看参数规模扩展 |
| attention budget K | B+d^2 的不同组合 | 看稀疏预算对效果影响 |
| 训练 compute C | 固定 step / 固定 token / 固定时间三种视角 | 看 compute-normalized 性能 |

### 9.3 主要图表

必须画：

```text
validation loss vs sequence length
validation loss vs parameter count
validation loss vs effective attention pair count
validation loss vs measured training FLOPs 或 tokens processed
peak memory vs sequence length
tokens/sec vs sequence length
```

### 9.4 拟合方法

若任务输出是 loss，可尝试拟合：

```text
L(x) = L_inf + a * x^(-alpha)
```

其中 `x` 可以是 `N`、`P`、`C` 或 `K_eff`。若任务指标是 accuracy，则优先转换为 error rate：

```text
error = 1 - accuracy
```

再观察 log-log 关系。若数据点太少，不强行声称 scaling law，只报告 trend。

### 9.5 scaling law 阶段通过标准

```text
1. 至少一个任务上有 N-scaling 曲线；
2. 至少一个任务上有 P-scaling 曲线；
3. 至少一个任务上有 K-scaling 曲线；
4. 所有曲线同时报告质量和效率；
5. 不做超出数据支持的结论。
```

## 10. 实验环境与服务器规范

### 10.1 服务器

远程访问：

```bash
ssh huiwei
```

服务器是 A100 x 4 公用服务器。使用规则：

```text
如果某张卡已经被占用，不要使用该卡。
默认只用 1 张空闲卡。
只有确认多张卡空闲时，才允许使用多卡。
不要默认占满 4 张卡。
```

检查 GPU：

```bash
nvidia-smi
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python train.py ...
```

### 10.2 远程目录

所有远程工作必须在：

```bash
~/ysx/
```

建议结构：

```text
~/ysx/zigzag_attention/
  code/
  data/
  logs/
  checkpoints/
  outputs/
  envs/
  cached_graphs/
```

不要动其他人的目录、数据、环境、checkpoint。

### 10.3 Conda 环境

默认使用：

```bash
conda activate ysx_base
```

如果需要新环境：

```bash
conda create -n ysx_zigzag python=3.10
conda activate ysx_zigzag
```

不要修改公共环境中的关键包，除非确认不会影响他人。新依赖优先装在个人新环境中。

环境记录：

```bash
conda env export > env_snapshot.yaml
pip freeze > requirements_snapshot.txt
```

### 10.4 本地开发，远程测试

访问服务器网络不稳定，所以代码开发以本地为主。远程只做：

```text
smoke test
短训练
正式训练
性能 profiling
```

远程 hotfix 必须同步回本地。正式实验必须记录 git commit hash。

### 10.5 tmux / screen

远程训练必须用：

```bash
tmux new -s zigzag
```

或：

```bash
screen -S zigzag
```

避免网络断开导致训练终止。

### 10.6 checkpoint resume

所有训练脚本必须支持：

```bash
--resume <checkpoint_path>
```

并定期保存：

```text
last.ckpt
best_valid.ckpt
step_<N>.ckpt
```

日志中必须写入 checkpoint 路径。

## 11. 实验记录规范

每次实验创建一个目录：

```text
outputs/<date>_<task>_<method>_<short_config>/
```

目录内必须包含：

```text
config.yaml
command.sh
git_commit.txt
env_snapshot.yaml
train.log
metrics.jsonl
profile.json
checkpoint_path.txt
```

`metrics.jsonl` 每条记录至少包含：

```json
{"step": 100, "train_loss": 1.23, "valid_metric": 0.45, "tokens_per_sec": 12000, "memory_allocated_gb": 12.3}
```

## 12. 失败诊断流程

### 12.1 新模型不如 local-only

检查：

```text
1. 任务是否真的需要远程依赖；
2. zig-zag 跨块边是否生成成功；
3. effective K 是否太小；
4. 多层可达率是否不足；
5. G/H 是否退化或 disconnected；
6. softmax 是否错误地分开归一化。
```

### 12.2 新模型不如 random same-budget

检查：

```text
1. G 的直径和谱间隙；
2. H 是否太弱，例如 cycle H 可能混合慢；
3. duplicate rate 是否过高；
4. 是否固定同一图导致层间覆盖不足；
5. 尝试 random regular H 或 layer-wise resampling。
```

### 12.3 显存降了但速度不升

检查：

```text
1. gather/scatter 是否成为瓶颈；
2. cross edges 是否未排序；
3. 是否仍 materialize 大 mask；
4. local 与 cross 是否重复读取 K/V；
5. 是否 batch size 太小导致 GPU 利用率低；
6. 是否应该按 block pair 做 batching。
```

## 13. 最终交付物

项目最终至少交付：

```text
1. base repo 可用性报告；
2. base 训练评测结果；
3. zig-zag attention 实现；
4. 新模型试训练报告；
5. 性能调优报告；
6. 完整主实验表；
7. 消融实验表；
8. scaling trend 图；
9. 可复现实验命令和配置；
10. 结论：新模型在哪些条件下有效，在哪些条件下无效。
```

## 14. 附录 A：推荐第一周任务清单

第一周只做这些：

```text
Day 1: 调查候选仓库，优先 guy-dar/lra-benchmarks。
Day 2: 在 huiwei 上跑通 base 仓库 smoke test。
Day 3: 准备 synthetic Associative Recall 和 LRA ListOps。
Day 4: 跑 dense/local/random baseline 的小配置。
Day 5: 写 Rot_G、Rot_H、zig-zag neighbor builder 的单元测试。
Day 6: dense mask debug 版本接入 base。
Day 7: 新模型小 N 试训练 1000 steps，并记录初步结果。
```

不要在第一周做：

```text
多任务全量训练；
大规模 scaling law；
自定义 CUDA；
多模态任务；
复杂消融；
占满 4 张 A100 的训练。
```

## 15. 附录 B：参考资料

- Long Range Arena official repository: https://github.com/google-research/long-range-arena
- Long Range Arena paper: https://arxiv.org/abs/2011.04006
- PyTorch LRA benchmark repository: https://github.com/guy-dar/lra-benchmarks
- BigBird paper: https://arxiv.org/abs/2007.14062
- BigBird official repository: https://github.com/google-research/bigbird
- Hugging Face BigBird docs: https://huggingface.co/docs/transformers/en/model_doc/big_bird
- xFormers documentation: https://facebookresearch.github.io/xformers/
- xFormers optimized ops: https://facebookresearch.github.io/xformers/components/ops.html
- Exphormer repository: https://github.com/hamed1375/Exphormer
- Exphormer paper: https://arxiv.org/abs/2303.06147
- Longformer paper: https://arxiv.org/abs/2004.05150
- Longformer repository: https://github.com/allenai/longformer
- Sparse Attention repository: https://github.com/openai/sparse_attention
- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention repository: https://github.com/dao-ailab/flash-attention
- Scaling Laws for Neural Language Models: https://arxiv.org/abs/2001.08361
- Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
