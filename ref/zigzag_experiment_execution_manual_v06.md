# Zig-Zag Sparse Attention Copy 实验执行手册 v0.6

## 0. 文档定位

本文档在 `ref/zigzag_experiment_execution_manual_v05.md` 基础上更新，吸收
`ref/Method_and_Experiment_Design_v1.pdf` 的方法设计要求。v0.6 的核心变化是：

```text
先验证 H/G 图结构和 mask 是否与 directed block zig-zag 理论对象对齐，
再运行 copy 训练。
```

v0.6 暂时保留 copy 任务，作为工程闭环和长程通信诊断任务；同时新增 WikiText2 作为与 copy 并行的第二个任务，用于真实文本 causal LM pipeline 的下载、smoke test 和轻量评测。copy 任务不是最终理论证据，尤其不能把固定位置捷径或单个 seed 的成功解释成 expander-aligned zig-zag 的普遍优势。

本版本明确修正 v0.5 的两个问题：

```text
小问题:
  log_every=250 太稀，难以观察收敛动态；
  steps=1000 在单 seed 下也不足以稳定观察收敛；
  v0.6 copy 主实验只训练 N_train=512，并只评估 N_eval=512；
  多 seed 要求移除，默认只跑 seed=0。

大问题:
  H/G 不能继续用 cyclic G + cycle H 当主方法；
  必须先生成 directed regular G/H，计算图结构证书和诊断指标；
  未通过图证书的结构只能作为 sparse baseline，不能称为 expander-aligned zig-zag。
```

旧 v04/v05 结果可以作为历史参考，不得混入 v0.6 主结果表。v05 的 outputs、reports、logs 和 configs 已归档到：

```text
ref/archive_v05_reports/
```

v0.6 的新代码、配置、图证书、训练结果、WikiText2 数据产物和报告必须使用独立版本名和输出目录。

## 1. 可行实验步骤

v0.6 重新划分为共享基础 phase 和两条任务分支。共享 phase 必须按顺序推进；共享 gate 通过后，copy 分支和 WikiText2 分支可以并行推进。

| Phase | 名称 | 目标 | 通过条件 |
|---|---|---|---|
| 0 | Scope Freeze | 固定 v0.6 范围：copy `N_train=N_eval=512`、WikiText2 轻量评测、主问题转为 H/G 图结构生成与诊断 | v06 文档、配置草案、输出目录命名确认 |
| 1 | Code Modularization and Task Interface | 拆分过长的 `scripts/synthetic_mvp.py`，建立 copy/WikiText2 共用的任务接口、runner 和日志组件 | 旧入口兼容，新模块可导入，copy smoke 仍可跑 |
| 2 | H/G Graph Generation | 实现 directed regular G/H 生成器，替代 cyclic/cycle 主配置 | `Rot_G` 双射、`P_G/P_H` doubly stochastic、谱诊断和重采样逻辑通过 |
| 3 | Mask and Multiplicity | 实现 labelled/multigraph/multiplicity-preserving mask，保留 boolean 版本作消融 | 小规模 reference test 通过，`unique key + log M_xy` 与重复边 softmax 等价 |
| 4 | Graph Certificate Gate | 对候选 B,d,graph_seed 生成图证书，选择可训练主配置 | 输出 `graph_certificates.csv/jsonl` 和 selected graph artifact |
| 5C | Copy Smoke | 用选定图结构跑 N=512 短训练，验证 copy 数据、日志、评估和曲线产物 | 无 NaN，日志步距足够密，4 类方法产物完整 |
| 6C | Copy Main Run | 只运行 `N_train=512`、`N_eval=512` copy 主训练 | 主结果表、训练曲线、图诊断、失败记录和阶段报告完整 |
| 5W | WikiText2 Download and Smoke | 下载 WikiText2 raw 数据，验证 splits、schema、tokenizer 和短 LM batch | data readiness 和 smoke 产物完整 |
| 6W | WikiText2 Evaluation | 对 WikiText2 做轻量 LM 评测，报告 loss/perplexity 和速度/显存 | 评测表、命令、数据快照和失败记录完整 |

禁止跳步。例如：没有模块化任务接口，不允许把 WikiText2 逻辑塞进 `synthetic_mvp.py`；没有图证书，不允许跑主训练；没有 multiplicity-preserving reference test，不允许把 zig-zag 结果写成理论对齐；没有 copy smoke，不允许提交 5000-step copy 主训练；没有 WikiText2 data smoke，不允许报告 WikiText2 评测。

## 2. v0.6 固定范围

### 2.1 任务范围

v0.6 有两个并行任务：在线 causal full-copy 和 WikiText2 causal LM。二者共享 graph/mask/attention/model/training 基础设施，但任务数据、loss/metrics 和输出目录必须分离。

copy 任务定义：

```text
task: copy
data: online
mode: full_copy
N_train: 512
N_eval: 512
num_values: 4
causal: true
```

每个样本在线生成源序列：

```text
s = [s0, s1, ..., sN-1]
s_i in {1, 2, 3, 4}
```

特殊 token：

```text
PAD = 0
SEP = num_values + 1 = 5
EOS = num_values + 2 = 6
```

输入序列：

```text
z = [s0, s1, ..., sN-1, SEP, s0, s1, ..., sN-1, EOS]
```

`N=512` 表示需要复制的源序列长度，不是 attention token 总长度。实际长度为：

```text
T_raw = 2N + 2 = 1026
T = ceil(T_raw / B) * B
```

训练使用 causal LM teacher forcing。位置 `t` 的 logits 预测 `z[t+1]`。loss 只计算第二段 copy 输出和 EOS：

```text
loss_positions = [N, N+1, ..., 2N]
targets        = [s0, s1, ..., sN-1, EOS]
```

评估只做：

```text
N_eval = 512
```

copy 部分不做 `N_eval > N_train` extrapolation。

WikiText2 任务定义：

```text
dataset: WikiText2
variant: wikitext-2-raw-v1
preferred source: Salesforce/wikitext
fallback source: carlosejimenez/wikitext__wikitext-2-raw-v1
task: causal language modeling
seed: 0
```

WikiText2 阶段只做轻量语言模型 smoke 和评测，用来检查真实文本 pipeline、tokenizer、padding、causal LM loss、perplexity、速度和显存。它不是 official LRA benchmark，也不是大规模语言模型训练。

### 2.2 方法范围

v0.6 至少包含以下方法：

| 方法 | 作用 | 备注 |
|---|---|---|
| dense | 完整 causal self-attention | 质量参考 |
| local | block 内 complete attention | 检查跨 block 边是否必要 |
| random_regular | local + same-budget random regular directed cross edges | 关键公平 baseline |
| zigzag_certified | local + certified directed zig-zag multiplicity mask | 主方法 |
| zigzag_boolean | local + zig-zag pure boolean union | 消融，不作为理论对齐主方法 |
| zigzag_cycle | v05 cyclic G + cycle H | 历史 baseline，不得称为主方法 |

如果成本需要压缩，copy 分支的 Phase 5C smoke 可以只跑 `dense/local/random_regular/zigzag_certified`。Phase 6C 主实验应至少保留这 4 类方法。

### 2.3 代码模块化与任务接口

当前 `scripts/synthetic_mvp.py` 已经超过 2000 行，混合了 attention backend、model、copy 数据生成、训练循环、评估、绘图、config、CLI 和文件写入逻辑。v0.6 不允许继续把 WikiText2 逻辑追加到这个单文件里。WikiText2 是与 copy 并行的任务，因此必须先做适度模块化，再接入新任务。

推荐拆分为：

```text
scripts/synthetic_mvp.py
  仅保留向后兼容 CLI wrapper，调用 shared runner。

scripts/run_experiment.py
  通用入口：读取 config、展开 methods/seeds、调度 task、写 summary。

scripts/config_io.py
  config merge、CLI override、sha256、git commit、command.sh。

scripts/artifact_io.py
  write_json/write_jsonl/write_csv、config snapshot、failure record。

scripts/plotting.py
  training_curves.png 和后续 WikiText2 曲线。

scripts/attention_backends.py
  dense/neighbor/split/blockpair attention、neighbor table、block-pair index。

scripts/modeling.py
  Transformer、Block、MaskedSelfAttention。

scripts/training_loop.py
  train/evaluate/checkpoint/cuda timing，任务无关。

scripts/tasks/base.py
  TaskSpec、Batch、Task interface。

scripts/tasks/copy_task.py
  full-copy batch generation、copy loss、token/sequence/EOS metrics。

scripts/tasks/wikitext2_task.py
  WikiText2 loading/tokenization/blocking、LM loss、perplexity metrics。

scripts/graph_structures.py
  H/G graph、mask、multiplicity 结构。

scripts/graph_diagnostics.py
  graph certificate、shortcut diagnostics。
```

任务接口至少包含：

```python
class Task:
    name: str
    def build_batch(self, split: str, batch_index: int, device): ...
    def loss_and_metrics(self, logits, batch): ...
    def eval_splits(self) -> list[str]: ...
    def result_fields(self) -> list[str]: ...
```

拆分要求：

```text
1. copy 和 WikiText2 不能互相 import 任务私有逻辑；
2. training_loop 只依赖 Task interface，不依赖具体任务名；
3. attention/model/graph 模块任务无关；
4. synthetic_mvp.py 作为兼容入口保留，但新增逻辑不得继续堆在其中；
5. 每次拆分后必须跑 copy smoke，证明旧路径未破坏；
6. WikiText2 只能在模块化后接入。
```

Phase 1 的目标不是一次性重构成复杂框架，而是把长文件切开到可维护边界，让 copy 和 WikiText2 作为并行任务共享同一套 runner。

## 3. 理论对齐对象

### 3.1 Token 和 block

设 attention 总长度为 `T`，block size 为 `B`，block 数为：

```text
q = T / B
```

每个 token 写成：

```text
x = (v, i)
v in [0, q-1]       # block id
i in [0, B-1]       # port or token id inside block
idx(v, i) = vB + i
```

v0.6 的图证书在 padding 后的 `T` 上计算。copy 的 `N=512` 只用于数据任务，图结构诊断使用 `T`、`B`、`q`。

### 3.2 Local complete block

对于 query token `x=(v,i)`，local keys 为：

```text
N_loc(v, i) = {(v, j): j in [0, B-1]}
```

因此 local 部分每个 query 有 `B` 条 labelled key path。

### 3.3 小端口图 H

小图 `H` 定义在端口集合 `[B]` 上。它必须是 `d-in/d-out regular directed graph`。用 `d` 个端口置换构造：

```text
sigma_0, sigma_1, ..., sigma_{d-1}: permutations of [B]
N_H^+(i) = {sigma_a(i): a in [0, d-1]}
```

归一化转移矩阵：

```text
P_H = (1/d) * sum_a S_{sigma_a}
```

要求：

```text
P_H 1_B = 1_B
1_B^T P_H = 1_B^T
```

诊断量：

```text
J_B = (1/B) 1_B 1_B^T
mu_H = ||P_H - J_B||_2
```

`cycle-like H` 不能作为主方法。对于 `B=16,d=2`，cycle random walk 的第二奇异值接近 `cos(2*pi/16) ~= 0.924`，通常无法满足 directed singular-value expansion 条件。因此 `H.type=cycle` 在 v0.6 中只能叫 baseline 或 regression case。

### 3.4 大图 G 和旋转映射

大图 `G` 定义在 block 集合 `[q]` 上。每个 block 有 `B` 个出端口和 `B` 个入端口。旋转映射：

```text
R_G: [q] x [B] -> [q] x [B]
```

若：

```text
R_G(v, p) = (w, r)
```

表示从 block `v` 的出端口 `p` 跳到 block `w`，到达端口 `r`。严格要求 `R_G` 是 `[q] x [B]` 上的双射。

推荐用 `B` 个 block-level permutation 构造：

```text
pi_0, pi_1, ..., pi_{B-1}: permutations of [q]
R_G(v, p) = (pi_p(v), p)
```

为减少自环，优先采样 derangement：

```text
pi_p(v) != v, for all v
```

为减少平行边，应记录并限制同一 `v -> w` 的重复次数。

大图的 block-level transition：

```text
(P_G)_{w,v} = (1/B) * count{p: R_G(v,p) = (w,r) for some r}
```

要求：

```text
P_G 1_q = 1_q
1_q^T P_G = 1_q^T
```

诊断量：

```text
J_q = (1/q) 1_q 1_q^T
lambda_G = ||P_G - J_q||_2
```

### 3.5 Directed zig-zag remote paths

对 query token：

```text
x = (v, i)
```

一次 directed zig-zag remote path 为：

```text
(v, i) --H--> (v, p) --G--> (w, r) --H--> (w, j)
```

其中：

```text
p in N_H^+(i)
(w, r) = R_G(v, p)
j in N_H^+(r)
```

每个 query 有 `d^2` 条 labelled remote paths。定义 multiplicity：

```text
C_xy = count{(a,b) in [d] x [d]: ZZ(x,a,b) = y}
```

labelled remote transition：

```text
P_zz,xy = C_xy / d^2
```

### 3.6 Expansion certificate

构造：

```text
P_zz = (I_q x P_H) R_G (I_q x P_H)
J_qB = J_q x J_B
```

必须报告：

```text
lambda_G = ||P_G - J_q||_2
mu_H     = ||P_H - J_B||_2
rho_bound = sqrt(lambda_G^2 + 2*mu_H^2 - lambda_G^2*mu_H^2)
```

主方法候选必须满足：

```text
rho_bound < 1
```

更强的简单充分条件：

```text
lambda_G^2 + 2*mu_H^2 < 1
```

对中小规模 `T`，还要直接构造 `P_zz` 并报告：

```text
rho_exact = ||P_zz - J_qB||_2
```

并检查：

```text
rho_exact <= rho_bound + eps
```

若上述条件不满足，该配置只能称为 sparse pattern 或 non-certified baseline，不能称为 expander-aligned zig-zag。

## 4. H/G 生成与诊断要求

### 4.1 新增或更新文件

v0.6 推荐新增或扩展以下图结构文件。任务接口和通用 runner 的拆分见第 2.3 节，不应继续堆进 `scripts/synthetic_mvp.py`。

```text
scripts/graph_structures.py
scripts/graph_diagnostics.py
configs/copy_v06_graph_search.json
configs/copy_v06_smoke.json
configs/copy_v06_main_n512.json
```

`scripts/graph_structures.py` 至少提供：

```text
build_permutation_regular_g(...)
build_permutation_regular_h(...)
build_rot_g(...)
build_zigzag_multiplicity(...)
build_random_regular_cross_edges(...)
build_boolean_ablation_mask(...)
load_graph_artifact(...)
validate_graph_config(...)
```

`scripts/graph_diagnostics.py` 至少提供：

```text
check_rot_g_bijection(...)
check_doubly_stochastic(...)
compute_lambda_g(...)
compute_mu_h(...)
compute_rho_bound(...)
compute_rho_exact(...)
compute_collision_overlap_stats(...)
compute_boolean_ablation_stats(...)
compute_shortcut_stats(...)
write_graph_certificate(...)
```

训练脚本不得在主训练逻辑中硬编码 H/G。所有图结构必须来自 config 或 graph artifact。

### 4.2 图生成 config

推荐图搜索配置：

```json
{
  "version": "v06",
  "N_task": 512,
  "T_raw": 1026,
  "candidate_block_sizes": [16, 32, 64],
  "candidate_degrees": [4, 6, 8],
  "graph_seeds": [0],
  "G": {
    "type": "permutation_regular",
    "require_derangement": true,
    "max_parallel_edges_per_block_pair": 2
  },
  "H": {
    "type": "permutation_regular",
    "allow_self_port": false
  },
  "acceptance": {
    "rho_bound_lt": 1.0,
    "prefer_simple_sufficient_condition": true,
    "max_remote_local_overlap_mean": 0.25
  },
  "output": {
    "root": "outputs/copy_v06_graph_search"
  }
}
```

说明：

```text
B,d 不在文档中强行固定。
Phase 4 先用 graph_seed=0 搜索并选择 certified graph。
Phase 6C 和 Phase 6W 使用 selected graph artifact。
多 graph_seed 不再是默认要求；只有 graph_seed=0 无法生成 certified graph 时，才扩展 graph_seeds。
```

### 4.3 必须报告的图证书字段

每个候选图必须输出一行 certificate：

```text
version
graph_id
graph_seed
T_raw
T
N_task
B
d
q
G_type
H_type
rot_g_is_bijection
P_G_row_stochastic_error
P_G_col_stochastic_error
P_H_row_stochastic_error
P_H_col_stochastic_error
lambda_G
mu_H
rho_bound
rho_exact
simple_condition_lhs
certified
resample_reason
remote_labelled_K
local_labelled_K
raw_K
remote_unique_K_min
remote_unique_K_mean
remote_unique_K_max
collision_count_min
collision_count_mean
collision_count_max
remote_local_overlap_min
remote_local_overlap_mean
remote_local_overlap_max
unique_total_K_min
unique_total_K_mean
unique_total_K_max
row_degree_min_boolean
row_degree_max_boolean
col_degree_min_boolean
col_degree_max_boolean
stationary_l2_to_uniform_boolean
```

保存路径：

```text
outputs/copy_v06_graph_search/graph_certificates.csv
outputs/copy_v06_graph_search/graph_certificates.jsonl
outputs/copy_v06_graph_search/graphs/<graph_id>.json
```

### 4.4 选择主图的规则

主图选择顺序：

```text
1. certified = true
2. rho_bound 最小
3. simple_condition_lhs 更小
4. remote/local overlap 更小
5. unique_total_K_mean 更接近 raw_K
6. B,d 成本更低
```

若 graph_seed=0 没有任何候选图通过 `rho_bound < 1`，不得进入主训练。可以扩大搜索：

```text
B in [16, 32, 64, 128]
d in [4, 6, 8]
graph_seeds 增加到 [0..19]
```

若仍失败，Phase 4 报告负结果，v0.6 停在图结构问题，不用训练掩盖该问题。

## 5. Multiplicity-Preserving Attention

### 5.1 总 multiplicity

local complete block 和 remote zig-zag 合并时，定义总 multiplicity：

```text
M_xy = 1[y in N_loc(x)] + C_xy
```

每个 query 的 labelled edge 总数：

```text
K_raw = B + d^2
```

若 remote path 落回 local block，不应直接丢弃。理论对象是 labelled/multigraph transition，必须把 multiplicity 相加。

### 5.2 Unique key + log multiplicity bias

工程实现不需要真的重复 key/value。对每个 query `x` 存储：

```text
N(x) = {y: M_xy > 0}
logM_xy = log(M_xy)
```

标准 attention score：

```text
s_xy = q_x^T k_y / sqrt(d_head)
```

加入 multiplicity bias：

```text
s_tilde_xy = s_xy + logM_xy
```

softmax：

```text
alpha_xy = exp(s_tilde_xy) / sum_{z in N(x)} exp(s_tilde_xz)
o_x = sum_{y in N(x)} alpha_xy v_y
```

这与真实重复 `M_xy` 次 key/value 的 softmax 等价，因为：

```text
exp(s_xy + log M_xy) = M_xy * exp(s_xy)
```

### 5.3 Reference tests

Phase 3 必须在小规模上做 reference test：

```text
T in {64, 128}
B in {16, 32}
d in {4}
batch_size = 2
heads = 2
d_model = 32
```

检查：

```text
1. labelled repeated-edge attention 与 unique+logM attention 输出误差 < 1e-5；
2. unique+logM 与 pure boolean 输出不同，且差异被记录；
3. causal filtering 前后的 K 和 pair count 都被记录；
4. dense masked reference、neighbor/split/blockpair backend 输出一致；
5. logM 不参与梯度，不随训练变化。
```

如果当前 backend 暂时不支持 logM，允许先保留 boolean 训练，但报告必须写为：

```text
zigzag_boolean: non-certified implementation ablation
```

不能写为：

```text
expander-aligned zig-zag main result
```

## 6. Copy 任务的捷径诊断

PDF 明确指出固定位置 `copy_first` 容易被稀疏 mask 的直接捷径解决。v0.6 暂时保留 full-copy，但必须附带 shortcut 诊断，避免把一跳可见性误认为多层全局混合。

对每个训练配置，设 copy 输出位置为 query `t`，目标源位置为 `p`。必须在 structural mask 和 causal-effective mask 上分别计算：

```text
target_in_1hop_rate
target_in_2hop_rate
target_in_Lhop_rate
average_shortest_path
unreachable_rate
```

建议 `L = model.layers`。

对于 full-copy，目标集合为：

```text
query positions:  [N, N+1, ..., 2N]
target positions: [0, 1, ..., N-1] plus EOS target
```

如果 `target_in_1hop_rate` 过高，训练成功只能说明 mask 里存在直接 copy path，不足以说明多层 expander mixing。

shortcut 诊断保存到：

```text
outputs/copy_v06_main_n512/shortcut_diagnostics.csv
outputs/copy_v06_main_n512/shortcut_diagnostics.jsonl
```

## 7. WikiText2 数据下载、Smoke 与评测

WikiText2 是与 copy 并行的任务分支，不是 copy 主实验完成后的附属评测。WikiText2 数据下载可以在 Phase 1 模块化完成后启动；WikiText2 smoke/eval 中涉及 zig-zag 方法的部分必须等待 Phase 4 selected graph artifact 存在。WikiText2 分支不依赖 Phase 5C/6C 的 copy 训练结果。

### 7.1 数据源选择

WikiText2 首选 Hugging Face `Salesforce/wikitext` 的 `wikitext-2-raw-v1` 配置。若该源在远端下载失败、cache 异常或数据脚本不可用，可以使用用户提供的 fallback：

```text
https://huggingface.co/datasets/carlosejimenez/wikitext__wikitext-2-raw-v1
```

数据下载阶段必须先验证数据源，不允许默认信任镜像。至少检查：

```text
splits: train, validation, test
field: text
train rows > 0
validation rows > 0
test rows > 0
空字符串比例
最长文本长度
tokenized length 分布
dataset source id
dataset revision 或 parquet file hash
```

首选下载方式：

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
```

fallback 下载方式：

```python
from datasets import load_dataset
ds = load_dataset("carlosejimenez/wikitext__wikitext-2-raw-v1")
```

若 `Salesforce/wikitext` 在当前环境中无法解析，也允许尝试：

```python
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
```

但报告必须写明实际使用的数据源和失败原因。

### 7.2 数据产物

推荐新增：

```text
scripts/prepare_wikitext2.py
scripts/wikitext2_smoke.py
scripts/wikitext2_eval.py
configs/wikitext2_v06_data.json
configs/wikitext2_v06_smoke.json
configs/wikitext2_v06_eval.json
```

下载和 smoke 产物保存到：

```text
datasets/wikitext2_raw_v1/
  dataset_info.json
  data_readiness.json
  tokenized_smoke.json

outputs/wikitext2_v06_smoke/
  summary.json
  batch_examples.jsonl
  smoke_metrics.jsonl
  command.sh
```

`datasets/` 是否纳入 git、如何同步，统一按 `ref/experiment_environment_and_version_control.md` 执行；data readiness JSON 和 smoke summary 必须同步回本地。

### 7.3 WikiText2 Smoke

WikiText2 smoke 目标不是质量，而是验证真实文本路径：

```text
加载数据集；
过滤空行或仅空白行；
tokenizer 可运行；
能构造 fixed length causal LM block；
能生成 attention mask；
dense/local/random_regular/zigzag_certified 至少完成 forward；
至少一个方法完成 backward；
loss 非 NaN；
能记录 tokens/sec 和 peak memory。
```

建议 smoke 参数：

```text
sequence_length = 512
max_train_batches = 2
max_eval_batches = 2
batch_size = 2
seed = 0
methods = dense, local, random_regular, zigzag_certified
```

如果 tokenizer 尚未固定，第一版可以使用 byte-level 或 GPT-2 tokenizer，但必须在 config snapshot 中写明 tokenizer 名称、vocab size、EOS/PAD 处理方式。

### 7.4 WikiText2 评测

WikiText2 评测只要求单 seed：

```text
seed = 0
```

默认做轻量 causal LM 训练加 validation/test 评测。如果代码暂时只能做无训练 eval，则必须标记为 `pipeline_only=true`，不得解释为模型质量。

推荐第一版评测参数：

```text
sequence_length = 512
train_steps = 1000
log_every = 50
eval_every = 100
batch_size = 8
eval_batches = 50
methods = dense, local, random_regular, zigzag_certified
```

必须报告：

```text
validation loss
validation perplexity
test loss
test perplexity
tokens/sec
peak allocated GB
peak reserved GB
number of non-empty train/validation/test rows
tokenizer
dataset source
dataset revision or hash
```

评测产物保存到：

```text
outputs/wikitext2_v06_eval/
  summary.json
  results.csv
  results.jsonl
  metrics.jsonl
  command.sh
  config_snapshot.json
  data_readiness.json
```

### 7.5 WikiText2 结论限制

WikiText2 v0.6 只允许作为真实文本 pipeline check 和轻量 LM sanity result。它不允许被写成：

```text
大规模语言模型结果；
official benchmark result；
多 seed 稳定性结论；
最终 sparse attention 优势结论。
```

## 8. 训练与日志规范

### 8.1 主训练固定参数

v0.6 主训练只使用：

```text
N_train = 512
N_eval = 512
steps = 5000
log_every = 50
eval_every = 50
batch_size = 16
eval_batches = 50
num_values = 4
learning_rate = 0.001
seeds = [0]
attention_backend = auto_split 或支持 logM 的等价 backend
causal = true
plot_curves = true
```

如果 5000 steps 成本过高，允许先跑 copy 分支 Phase 5C smoke：

```text
steps = 200
log_every = 10
eval_every = 10
batch_size = 8
eval_batches = 5
seeds = [0]
```

但 smoke 结果不能写成主结论。

### 8.2 日志密度

必须记录：

```text
step = 1
step % log_every == 0
step = final step
```

主训练中：

```text
log_every = 50
```

这比 v0.5 的 `log_every=250` 更密，足以观察早期收敛、平台期和单 seed 失败模式。

每次 log 至少记录：

```text
train_loss
eval_loss
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
tokens_per_sec
effective_K_mean_after_causal
attention_pair_count_after_causal
```

训练曲线必须由脚本直接生成：

```text
training_curves.png
```

至少包含：

```text
train loss vs step
eval loss vs step
eval token accuracy vs step
eval sequence accuracy vs step
eval EOS accuracy vs step
```

### 8.3 Checkpoint 和失败记录

主训练每 500 steps 保存一次 checkpoint：

```text
checkpoint_step_0500.pt
checkpoint_step_1000.pt
...
checkpoint_step_5000.pt
```

checkpoint 至少包含：

```text
model state
optimizer state
step
training seed
graph id
config snapshot
RNG state
```

失败不能删除重跑。每次失败必须保留：

```text
error.log
failed command
config snapshot
failure step
GPU state
failure_reason
needs_code_change
```

## 9. v0.6 Config 模板

### 9.1 Copy Smoke config

```json
{
  "version": "v06",
  "task": {
    "name": "copy",
    "data": "online",
    "mode": "full_copy",
    "num_values": 4,
    "train_lengths": [512],
    "eval_lengths": [512]
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
    "methods": ["dense", "local", "random_regular", "zigzag_certified"],
    "causal": true,
    "graph_artifact": "outputs/copy_v06_graph_search/graphs/<selected_graph_id>.json",
    "multiplicity": {
      "mode": "unique_log_m",
      "boolean_ablation": true
    }
  },
  "train": {
    "steps": 200,
    "batch_size": 8,
    "eval_batches": 5,
    "learning_rate": 0.001,
    "log_every": 10,
    "eval_every": 10,
    "seeds": [0]
  },
  "output": {
    "root": "outputs/copy_v06_smoke",
    "plot_curves": true,
    "curve_format": "png"
  }
}
```

### 9.2 Copy Main config

```json
{
  "version": "v06",
  "task": {
    "name": "copy",
    "data": "online",
    "mode": "full_copy",
    "num_values": 4,
    "train_lengths": [512],
    "eval_lengths": [512]
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
    "methods": ["dense", "local", "random_regular", "zigzag_certified", "zigzag_boolean"],
    "causal": true,
    "graph_artifact": "outputs/copy_v06_graph_search/graphs/<selected_graph_id>.json",
    "multiplicity": {
      "mode": "unique_log_m",
      "boolean_ablation": true
    }
  },
  "train": {
    "steps": 5000,
    "batch_size": 16,
    "eval_batches": 50,
    "learning_rate": 0.001,
    "log_every": 50,
    "eval_every": 50,
    "checkpoint_every": 500,
    "seeds": [0]
  },
  "output": {
    "root": "outputs/copy_v06_main_n512",
    "plot_curves": true,
    "curve_format": "png"
  }
}
```

### 9.3 WikiText2 Data config

```json
{
  "version": "v06",
  "dataset": {
    "name": "wikitext2",
    "preferred_source": {
      "path": "Salesforce/wikitext",
      "name": "wikitext-2-raw-v1"
    },
    "fallback_source": {
      "path": "carlosejimenez/wikitext__wikitext-2-raw-v1"
    },
    "text_field": "text",
    "cache_dir": "datasets/hf_cache",
    "output_dir": "datasets/wikitext2_raw_v1"
  },
  "validation": {
    "required_splits": ["train", "validation", "test"],
    "min_rows": {
      "train": 1,
      "validation": 1,
      "test": 1
    },
    "max_empty_line_rate": 0.5
  }
}
```

### 9.4 WikiText2 Eval config

```json
{
  "version": "v06",
  "task": {
    "name": "wikitext2",
    "mode": "causal_lm",
    "dataset_dir": "datasets/wikitext2_raw_v1",
    "sequence_length": 512
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
    "methods": ["dense", "local", "random_regular", "zigzag_certified"],
    "causal": true,
    "graph_artifact": "outputs/copy_v06_graph_search/graphs/<selected_graph_id>.json",
    "multiplicity": {
      "mode": "unique_log_m",
      "boolean_ablation": false
    }
  },
  "tokenizer": {
    "name": "gpt2",
    "pad_token": "eos"
  },
  "train": {
    "steps": 1000,
    "batch_size": 8,
    "eval_batches": 50,
    "learning_rate": 0.001,
    "log_every": 50,
    "eval_every": 100,
    "seeds": [0]
  },
  "output": {
    "root": "outputs/wikitext2_v06_eval",
    "plot_curves": true,
    "curve_format": "png"
  }
}
```

## 10. 输出目录和结果字段

### 10.1 输出目录

```text
outputs/copy_v06_graph_search/
  graph_certificates.csv
  graph_certificates.jsonl
  graphs/<graph_id>.json

outputs/copy_v06_smoke/
  summary.json
  results.csv
  results.jsonl
  mask_tests.json
  shortcut_diagnostics.csv
  train_N512_seed0_<method>/

outputs/copy_v06_main_n512/
  summary.json
  phase5_results.csv
  phase5_results.jsonl
  graph_diagnostics.csv
  shortcut_diagnostics.csv
  train_N512_seed0_<method>/

datasets/wikitext2_raw_v1/
  dataset_info.json
  data_readiness.json
  tokenized_smoke.json

outputs/wikitext2_v06_smoke/
  summary.json
  batch_examples.jsonl
  smoke_metrics.jsonl
  command.sh

outputs/wikitext2_v06_eval/
  summary.json
  results.csv
  results.jsonl
  metrics.jsonl
  command.sh
  config_snapshot.json
  data_readiness.json
```

每个 run 子目录必须包含：

```text
summary.json
results.csv
results.jsonl
metrics.jsonl
training_curves.png
command.sh
config_snapshot.json
graph_certificate.json
```

### 10.2 主结果字段

主结果表至少包含：

```text
version
run_id
status
failure_reason
task
copy_mode
method
seed
graph_id
graph_seed
N_train
N_eval
T_raw
T
B
d
q
G_type
H_type
causal
multiplicity_mode
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
eval_every
raw_K
unique_K_mean
effective_K_min_after_causal
effective_K_mean_after_causal
effective_K_max_after_causal
attention_pair_count_after_causal
lambda_G
mu_H
rho_bound
rho_exact
certified
remote_local_overlap_mean
target_in_1hop_rate
target_in_2hop_rate
target_in_Lhop_rate
average_shortest_path
unreachable_rate
final_train_loss
eval_loss
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
tokens_per_sec
elapsed_sec
peak_allocated_gb
peak_reserved_gb
training_curves_path
artifact_dir
git_commit
config_sha256
```

### 10.3 WikiText2 结果字段

WikiText2 评测结果表至少包含：

```text
version
run_id
status
failure_reason
dataset
dataset_source
dataset_revision_or_hash
tokenizer
vocab_size
sequence_length
method
seed
graph_id
graph_seed
B
d
q
G_type
H_type
causal
multiplicity_mode
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
eval_every
train_nonempty_rows
validation_nonempty_rows
test_nonempty_rows
validation_loss
validation_perplexity
test_loss
test_perplexity
tokens_per_sec
elapsed_sec
peak_allocated_gb
peak_reserved_gb
pipeline_only
artifact_dir
git_commit
config_sha256
```

## 11. 共享环境与版本控制规范

所有实验版本共享的环境、服务器、GPU、同步、日志和 git 版本控制要求统一见：

```text
ref/experiment_environment_and_version_control.md
```

v0.6 任务手册不再复制这些规则。执行任意 phase 前，必须先按共享环境文档检查：

```text
本地/远端目录；
rsync 同步规则；
GPU3 -> GPU2 -> GPU1、utilization.gpu < 50% 启动规则；
tmux/log 运行方式；
每次代码大更新后的 git commit 要求；
checkpoint 与 outputs/logs/datasets 不进入主提交。
```

## 12. 运行命令模板

### 12.1 本地图搜索

```bash
python scripts/graph_diagnostics.py \
  --config configs/copy_v06_graph_search.json \
  --output-dir outputs/copy_v06_graph_search
```

### 12.2 本地 copy smoke

```bash
python scripts/synthetic_mvp.py \
  --config configs/copy_v06_smoke.json \
  --output-dir outputs/copy_v06_smoke
```

### 12.3 远端同步

按 `ref/experiment_environment_and_version_control.md` 第 2 节执行代码同步。v0.6 手册只列任务相关运行命令和结果同步路径。

### 12.4 远端 copy smoke

```bash
ssh huiwei
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/synthetic_mvp.py \
  --config configs/copy_v06_smoke.json \
  --output-dir outputs/copy_v06_smoke \
  2>&1 | tee logs/copy_v06_smoke_$(date +%Y%m%d_%H%M%S).log
```

### 12.5 远端 copy 主训练

```bash
ssh huiwei
tmux new -s copy_v06_n512
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/synthetic_mvp.py \
  --config configs/copy_v06_main_n512.json \
  --output-dir outputs/copy_v06_main_n512 \
  2>&1 | tee logs/copy_v06_main_n512_$(date +%Y%m%d_%H%M%S).log
```

### 12.6 WikiText2 下载和 smoke

```bash
python scripts/prepare_wikitext2.py \
  --config configs/wikitext2_v06_data.json \
  --output-dir datasets/wikitext2_raw_v1

python scripts/wikitext2_smoke.py \
  --config configs/wikitext2_v06_smoke.json \
  --output-dir outputs/wikitext2_v06_smoke
```

### 12.7 WikiText2 评测

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/wikitext2_eval.py \
  --config configs/wikitext2_v06_eval.json \
  --output-dir outputs/wikitext2_v06_eval \
  2>&1 | tee logs/wikitext2_v06_eval_$(date +%Y%m%d_%H%M%S).log
```

### 12.8 结果同步

```bash
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/copy_v06_graph_search/ ./outputs/copy_v06_graph_search/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/copy_v06_smoke/ ./outputs/copy_v06_smoke/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/copy_v06_main_n512/ ./outputs/copy_v06_main_n512/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/datasets/wikitext2_raw_v1/ ./datasets/wikitext2_raw_v1/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/wikitext2_v06_smoke/ ./outputs/wikitext2_v06_smoke/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/wikitext2_v06_eval/ ./outputs/wikitext2_v06_eval/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/logs/ ./logs/
```

## 13. Phase 通过条件

### 13.1 Phase 0: Scope Freeze

通过条件：

```text
ref/zigzag_experiment_execution_manual_v06.md 存在；
v06 明确 copy 分支只做 N_train=512, N_eval=512；
v06 明确 WikiText2 是与 copy 并行的任务分支；
v06 明确默认只要求 seed=0；
v06 明确 copy 只是临时工程任务；
v06 明确包含 WikiText2 下载、smoke 和轻量评测；
v06 明确 H/G 图证书是训练前置条件；
v06 引用共享实验环境与版本控制规范。
```

### 13.2 Phase 1: Code Modularization and Task Interface

通过条件：

```text
synthetic_mvp.py 只保留兼容入口或显著瘦身；
attention/model/training/config/io/task 逻辑已拆分；
copy_task.py 与 wikitext2_task.py 使用同一 Task interface；
training_loop 不依赖具体任务名；
旧 copy smoke 命令仍可运行；
新增模块均可 import；
python -m py_compile scripts/*.py 通过。
```

### 13.3 Phase 2: H/G Graph Generation

通过条件：

```text
permutation_regular G/H 可生成；
Rot_G bijection 检查通过；
P_G/P_H doubly stochastic 检查通过；
cycle/cyclic 配置被降级为 baseline；
图生成完全由 config/graph artifact 控制。
```

### 13.4 Phase 3: Mask and Multiplicity

通过条件：

```text
labelled repeated-edge reference attention 实现；
unique key + logM attention 实现；
二者 small test 误差 < 1e-5；
pure boolean ablation 可单独运行；
pre-causal 和 post-causal 的 K/pair count 都被记录。
```

### 13.5 Phase 4: Graph Certificate Gate

通过条件：

```text
候选 B,d,graph_seed 搜索完成；
graph_certificates.csv/jsonl 存在；
至少一个 selected graph 满足 rho_bound < 1；
selected graph artifact 存在；
如果没有 certified graph，则停止并写负结果报告。
```

### 13.6 Phase 5C: Copy Smoke

通过条件：

```text
N_train=512；
N_eval=512；
steps=200；
log_every=10；
dense/local/random_regular/zigzag_certified 至少完成 seed 0；
无 NaN；
summary/results/metrics/training_curves 都存在；
shortcut diagnostics 存在。
```

### 13.7 Phase 6C: Copy Main Run

通过条件：

```text
N_train=512；
N_eval=512；
steps=5000；
log_every=50；
seeds=[0]；
dense/local/random_regular/zigzag_certified 完成；
zigzag_boolean 作为消融完成或失败原因明确；
主结果 CSV/JSONL 完整；
每个 run 生成 training_curves.png；
每个 run 绑定 graph_certificate.json；
本地和远端结果同步；
阶段报告完成。
```

### 13.8 Phase 5W: WikiText2 Download and Smoke

通过条件：

```text
WikiText2 数据下载完成或 fallback 数据源成功；
data_readiness.json 存在；
train/validation/test split 均存在；
text 字段存在；
空行比例、行数、token length 统计已记录；
tokenizer smoke 通过；
至少一个 batch 完成 forward/backward；
loss 非 NaN；
smoke summary 和 command.sh 存在。
```

### 13.9 Phase 6W: WikiText2 Evaluation

通过条件：

```text
seed=0；
dense/local/random_regular/zigzag_certified 完成或失败原因明确；
validation/test loss 和 perplexity 已记录；
tokens/sec 和 peak memory 已记录；
results.csv/results.jsonl 完整；
data_readiness.json 复制到评测输出目录；
command.sh 和 config_snapshot.json 存在；
阶段报告完成。
```

## 14. 分析口径

### 14.1 允许的结论

v0.6 完成后，允许回答：

```text
在 online causal full-copy N=512 -> N=512 上，zigzag_certified 是否能训练；
在同一 N=512、同一模型、同一训练预算下，zigzag_certified 是否优于 local-only；
在 same-budget 条件下，zigzag_certified 与 random_regular 的单 seed 收敛和质量差异；
unique+logM 与 pure boolean zig-zag 的差异；
图证书指标 lambda_G/mu_H/rho 与训练结果是否一致；
copy 任务中是否存在一跳或少跳 shortcut；
WikiText2 raw 数据集 pipeline 是否可用；
WikiText2 轻量 causal LM eval 的 validation/test loss 和 perplexity。
```

### 14.2 不允许的结论

v0.6 不允许声称：

```text
official LRA benchmark 结果；
真实文本任务的通用效果；
N > 512 extrapolation；
大模型 scaling law；
多 seed 稳定性；
通用 sparse attention superiority；
最终 block-sparse CUDA/Triton kernel 性能；
cycle/cyclic H/G 是理论对齐的主 zig-zag 方法。
```

### 14.3 负结果解释

若 `zigzag_certified < local`：

```text
检查 causal filtering 后 effective K 是否过低；
检查 target shortcut / unreachable rate；
检查 selected graph 是否虽然 certified 但 remote/local overlap 太高；
检查 logM 是否实际进入 attention logits；
检查训练步数、学习率和模型容量。
```

若 `zigzag_certified < random_regular`：

```text
确认 random_regular 与 zig-zag 的 raw_K 和 effective_K 相同；
比较 lambda_G, mu_H, rho_bound, rho_exact；
比较 collision/overlap 和 boolean stationary distribution；
尝试不同 graph_seed 或 layer-wise graph resampling；
不要直接写成 zig-zag 理论失败。
```

若 memory 降低但速度不升：

```text
拆分 local/cross/QKV/FFN time；
检查 gather/scatter 是否是瓶颈；
检查 cross edges 是否能按 block pair 排序；
把 blockpair 结果写成 prototype evidence，不写成最终 kernel 结果。
```

## 15. 运行记录规范

每次运行必须记录：

```text
config path
config sha256
graph artifact path
graph certificate path
command
git commit hash
local/remote run location
GPU id
seed
method
N_train
N_eval
T_raw
T
B
d
graph_id
G_type
H_type
multiplicity mode
causal
steps
log_every
eval_every
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
shortcut diagnostics path
dataset source
dataset revision or hash
tokenizer
validation loss/perplexity
test loss/perplexity
pipeline_only flag
```

git 提交、同步、checkpoint 排除和归档规则统一见：

```text
ref/experiment_environment_and_version_control.md
```

## 16. 当前 Copy 结果复查与修复事项

### 16.1 暂时不重跑 Copy 实验

当前阶段暂时不需要重跑 `copy_v06_main_n512` 训练。已完成的 copy 训练结果仍可作为调试和任务闭环参考，但在下列代码/结果语义问题修复前，不应把 shortcut diagnostics 或 `certified=True` 字段作为最终解释依据。

修复优先级：

```text
1. 先修统计和结果字段语义；
2. 尽量基于已有 graph artifact、config snapshot、results/metrics 重新生成诊断表或结果汇总；
3. 只有当修复影响训练 forward/loss/attention 行为时，才考虑重跑 copy；
4. 当前列出的三个问题本身不要求重跑 copy 训练。
```

### 16.2 `target_in_Lhop_rate` 统计语义偏差

当前 `target_in_Lhop_rate` 统计疑似有实现偏差。代码在 `depth == layers` 时才累计 L-hop，并且可能提前 `break`；这会把“L 层内可达”误算成“恰好第 L 层仍被访问”。

相关位置：

```text
scripts/graph_diagnostics.py:493
outputs/copy_v06_main_n512/shortcut_diagnostics.csv:2
```

已观察到的异常现象：

```text
dense 的 causal target_in_1hop_rate = 0.998；
但 target_in_Lhop_rate = 0.0；
这明显不符合“L-hop 内可达”的语义。
```

修复建议：

```text
1. 对每个 query-target pair 计算最短距离 dist；
2. target_in_1hop_rate 使用 dist <= 1；
3. target_in_2hop_rate 使用 dist <= 2；
4. target_in_Lhop_rate 使用 dist <= layers；
5. unreachable_rate 使用 dist is None；
6. BFS 可以在找到目标后停止，但不能因此跳过 dist <= L 的累计；
7. 字段名保留 target_in_Lhop_rate，但语义固定为 within L hops。
```

修复后只需要重算：

```text
outputs/copy_v06_main_n512/shortcut_diagnostics.csv
outputs/copy_v06_main_n512/shortcut_diagnostics.jsonl
results.csv/results.jsonl 中的 shortcut 汇总字段
```

不需要重跑 copy training。

### 16.3 `certified=True` 字段语义会误导

当前结果表中的 `certified=True` 容易误导。`zigzag_boolean` 行也写了 `certified=True`，但 PDF 明确要求 pure boolean 只能作为消融，不能称为 theory-aligned 主实现。

相关位置：

```text
outputs/copy_v06_main_n512/results.csv:6
scripts/synthetic_mvp.py:1489
```

根因判断：

```text
训练记录把全局 selected graph certificate 直接复制到所有 method 行；
这混淆了 graph 是否 certified 和当前 method implementation 是否 theory-aligned。
```

修复建议：将结果字段拆成至少两个语义不同的字段：

```text
graph_certified:
  selected graph artifact 是否满足 rho_bound < 1 等图证书条件。

implementation_certified:
  当前 method 的实现是否保留 PDF 要求的 labelled/multigraph/multiplicity 语义。

theory_aligned_method:
  当前 method 是否可以作为 theory-aligned directed zig-zag 主方法解释。
```

字段建议：

| method | graph_certified | implementation_certified | theory_aligned_method |
|---|---:|---:|---:|
| dense | NA | NA | false |
| local | NA | NA | false |
| random_regular | NA 或 false | false | false |
| zigzag_certified | true | true | true |
| zigzag_boolean | true | false | false |
| zigzag_cycle | false | false | false |

保留旧字段 `certified` 时，必须在报告中弃用或只作为 backward-compatible alias；新分析不得再使用单个 `certified=True` 判断理论对齐。

修复后只需要重写：

```text
results.csv
results.jsonl
summary.json 中相关字段
阶段报告中的字段解释
```

不需要重跑 copy training。

### 16.4 `config_snapshot` 可复现性语义不够干净

当前主配置可能不显式写 `B,d`，运行时从 graph artifact 解析成 `B=16,d=8`，结果行里的 `d=8` 是对的；但保存的 `config_snapshot` 仍可能保留默认 `degree=4`，容易误读。

相关位置：

```text
scripts/synthetic_mvp.py:1776
```

问题性质：

```text
这不一定影响训练结果；
但会影响复现实验时对“实际 config”和“用户输入 config”的理解。
```

修复建议：

```text
1. 保存 raw_config_snapshot，表示用户提供的原始 config + CLI overrides；
2. 保存 resolved_config_snapshot，表示运行时实际使用的最终配置；
3. resolved_config_snapshot 必须把 graph artifact 解析出的 B、d、q、graph_id、graph_seed 写回 attention/runtime 字段；
4. results.csv/results.jsonl 只使用 resolved config 中的实际参数；
5. command.sh 和 summary.json 明确记录 graph_artifact path 和 graph_certificate path。
```

可选兼容策略：

```text
如果暂时只保留 config_snapshot.json，则它必须改为 resolved config；
原始输入 config 可另存为 raw_config_snapshot.json。
```

修复后不需要重跑 copy training，但建议重写每个 run 目录中的 snapshot 文件，或在阶段报告中明确旧 snapshot 的 caveat。
