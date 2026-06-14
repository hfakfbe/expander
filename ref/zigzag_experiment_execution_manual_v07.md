# Zig-Zag Sparse Attention 实验执行手册 v0.7

## 0. 文档定位

v0.7 假定当前代码和当前需求是起点，不再把 v0.6 已完成或已过时的图搜索、证书搜索、归档等工作列为实验 phase。v0.6 文档保持为历史版本；v0.6 产物已归档到：

```text
ref/archive_v06_reports/
```

v0.7 只执行五个阶段：

| Phase | 名称 | 目标 | 通过条件 |
|---|---|---|---|
| 1 | Code Update for Copy and WikiText | 修改当前代码，使 copy 和 wikitext 两个任务共用结构参数、v07 graph artifact 读取/复制/校验、random/zigzag budget 对齐、日志和结果字段 | 代码检查、配置解析、graph 生成与 graph 复制/sha 校验 dry-run 可运行 |
| 2 | Canonical Graph Artifact Generation | 用固定 `graph_seed` 和固定生成算法只生成一次 v07 canonical graph artifact | canonical graph、certificate、generation metadata 和 sha256 完整 |
| 3 | Copy Smoke and Real Run | 先跑 copy smoke，再跑 copy real run | copy 结果表、训练动态 PNG、总耗时、v07 graph artifact sha256 与 canonical 完全一致 |
| 4 | WikiText Data Preparation and Tokenization | 下载/校验 WikiText-103，训练本实验 tokenizer，并 tokenize train/test | data readiness、tokenizer、tokenized train/test、sha256 和总耗时完整 |
| 5 | WikiText Smoke and Real Run | 读取 Phase 4 的固定数据/tokenizer 产物，先跑 wikitext smoke，再跑 wikitext real run | wikitext 结果表、训练动态 PNG、test PPL、总耗时、v07 graph artifact sha256 与 canonical 完全一致 |

禁止把 v0.6 的旧 phase 原样搬进 v0.7 报告。v0.7 的主线是“当前代码修改 -> 生成唯一 graph -> copy -> 准备并 tokenize wikitext -> wikitext 训练/评测”。

共享环境、GPU 选择、git 提交、checkpoint 忽略和远端同步规则统一见：

```text
ref/experiment_environment_and_version_control.md
```

## 1. 固定实验参数

v0.7 锁死结构参数：

```text
N = N_total = 1024
T = 1024
B = 32
q = 32
d = 8
causal = true
seed = 0
```

本文档中的 `N` 表示整体 token 序列长度，包括 copy 任务中的 `SEP` 和 `EOS`，不是 copy 原文长度。

copy 任务：

```text
copy_source_length = (N_total - 2) / 2 = 511

sequence =
  [s_0, ..., s_510, SEP, s_0, ..., s_510, EOS]

total length =
  511 + 1 + 511 + 1 = 1024
```

位置约定：

```text
source positions: 0..510
SEP position: 511
copy output positions: 512..1022
EOS position: 1023
loss query positions: 511..1022
loss target positions: 512..1023
```

wikitext 任务：

```text
sequence_length = 1024
train split = train
test split = test
```

默认数据源仍可使用 WikiText-103 raw：

```text
preferred: Salesforce/wikitext, wikitext-103-raw-v1
fallback/reference: iohadrubin/wikitext-103-raw-v1
```

如果实际使用其他 wikitext 数据源，必须在 `data_readiness.json`、`summary.json` 和最终报告中写明。

wikitext 默认不使用 GPT-2 tokenizer。主实验必须只基于 wikitext 的 train split 训练本实验自己的 tokenizer，然后用同一个 tokenizer 编码 train/test。

默认 tokenizer 策略：

```text
tokenizer_algorithm = byte_level_bpe
tokenizer_train_split = train
tokenizer_vocab_size = 32000
tokenizer_min_frequency = 2
special_tokens = <pad>, <eos>, <unk>
pad_token = <pad>
eos_token = <eos>
unk_token = <unk>
```

tokenizer 训练不得使用 test split。GPT-2 tokenizer 只允许作为临时 debug fallback，并且不得写入 v0.7 主结果。

## 2. v07 图生成与 Artifact 规则

v0.7 必须自己生成 graph artifact，不使用之前跑出的 graph artifact。graph 生成必须独立成 Phase 2，并且只生成一次 canonical graph。四个训练 run：

```text
copy smoke
copy real run
wikitext smoke
wikitext real run
```

必须使用同一个 `graph_seed`、同一个图生成算法和同一个 canonical graph artifact。四个 run 的 `graph_artifact_sha256` 必须完全一致。

canonical graph 输出目录固定为：

```text
outputs/v07_graph_n1024_q32_B32_d8/
```

canonical graph 固定参数：

```text
graph_seed = 0
graph_generation_algorithm = zigzag_v07_fixed_N1024_q32_B32_d8
N_total = 1024
T = 1024
q = 32
B = 32
d = 8
allow_multiedges = true
preserve_multiplicity = true
```

`graph_generation_algorithm` 是实现侧需要写入 metadata 的稳定算法标识；如果代码中使用更精确的名字，必须保证四个 run 记录完全一致。

Phase 2 输出目录必须包含：

```text
selected_graph.json
graph_certificate.json
graph_generation.json
graph_artifact.sha256
```

如果暂时沿用旧的 `scripts/graph_diagnostics.py`，必须在 Phase 1 或 Phase 2 中把旧输出标准化为以上文件名。例如旧的 `selected_graph_certificate.json` 必须复制或改名为 `graph_certificate.json`，并补写 `graph_generation.json` 与 `graph_artifact.sha256`。

训练 run 不得重新独立生成 graph。每个 run 启动时必须把 Phase 2 的 canonical graph 复制到自己的输出目录，并从本 run 输出目录内的副本读取：

```text
artifacts/graph/selected_graph.json
artifacts/graph/graph_certificate.json
artifacts/graph/graph_generation.json
```

其中：

```text
selected_graph.json:
  从 Phase 2 canonical graph 复制到本 run 输出目录、并实际使用的完整 graph artifact。

graph_certificate.json:
  从 Phase 2 canonical graph 复制到本 run 输出目录、并实际使用的证书文件。

graph_generation.json:
  记录 canonical_source_path、copy_timestamp_utc、graph_seed、graph_generation_algorithm、
  sha256、是否允许重边，以及是否通过 canonical sha256 校验。
```

`raw_config_snapshot.json` 记录用户提供的 graph 来源和期望 sha256；`resolved_config_snapshot.json` 必须把运行时 graph path 写成本次输出目录内副本路径，例如：

```text
outputs/copy_v07_main_n1024_q32_B32_d8/artifacts/graph/selected_graph.json
```

训练、诊断、结果汇总和报告都必须引用本 run 输出目录内的 graph artifact 副本，不能引用 v06 archive 或其他旧输出目录作为运行时依赖。Phase 2 canonical graph 不是旧产物；它是 v0.7 本次实验自己生成的唯一 graph 来源。

若任一 run 的 `graph_artifact_sha256` 不等于 Phase 2 的 `canonical_graph_artifact_sha256`，该 run 必须标记为 `status=failed`，不得进入主结果比较。

## 3. 谱字段和 Random 对齐

谱字段使用明确名字：

```text
lambda_G
mu_H
rho_zigzag_bound
rho_zigzag_certified
```

定义：

```text
rho_zigzag_bound =
  sqrt(lambda_G^2 + 2 * mu_H^2 - lambda_G^2 * mu_H^2)

rho_zigzag_certified =
  rho_zigzag_bound < 1
```

报告、CSV、JSONL 不使用含糊的 rhs/bound 代称。

random baseline 必须对齐 zigzag 的实际 post-causal attention budget。若训练前无法知道 zigzag 的实际 K，可以改变方法顺序，先执行 zigzag 的 mask dry-run 或 artifact-only pass，再生成 random mask。

至少记录：

```text
zigzag_actual_k_min_after_causal
zigzag_actual_k_mean_after_causal
zigzag_actual_k_max_after_causal
zigzag_attention_pair_count_after_causal
random_target_k_source = "zigzag_actual_post_causal"
random_alignment_mode = "per_query" or "global_pair_count"
random_actual_k_min_after_causal
random_actual_k_mean_after_causal
random_actual_k_max_after_causal
random_attention_pair_count_after_causal
random_k_aligned_to_zigzag
```

优先级：

```text
1. per-query K 对齐；
2. 如果实现成本过高，允许 global pair count 对齐；
3. 无论哪种方式，都必须在结果字段中写明 random_alignment_mode。
```

### 3.1 对比方法

v0.7 的对比方法固定为同一组，copy 和 wikitext 必须使用一致的 method 语义。

| method | 角色 | 运行要求 | 解释口径 |
|---|---|---|---|
| dense | dense reference | smoke 和 real run 都应包含 | 质量上界/完整 causal attention 参考 |
| local | sparse lower baseline | smoke 和 real run 都应包含 | 只看 block 内 complete attention，检查跨 block 边是否必要 |
| zigzag_certified | 主方法 | smoke 和 real run 都应包含 | 使用 Phase 2 v07 canonical graph、multiplicity-preserving mask 和 `rho_zigzag_bound` 证书 |
| zigzag_certified_cosine | 主方法调度对比 | real run 必须包含；smoke 可选 | 与 `zigzag_certified` 完全同结构，只把学习率从 constant 改为 cosine schedule |
| random_regular | 关键公平 baseline | smoke 和 real run 都应包含 | 必须用 zigzag actual post-causal K 对齐后再训练 |
| zigzag_boolean | 消融 | real run 必须包含；smoke 可选 | pure boolean union，不得称为 theory-aligned 主实现 |

smoke 阶段至少运行：

```text
dense
local
zigzag_certified
random_regular
```

real run 阶段运行：

```text
dense
local
zigzag_certified
zigzag_certified_cosine
random_regular
zigzag_boolean
```

公平性约束：

```text
1. 同一任务内所有 method 使用同一 N/B/q/d、同一模型宽度、同一训练预算和同一 seed；
2. zigzag_certified、zigzag_certified_cosine 和 zigzag_boolean 使用同一 Phase 2 canonical graph；
3. copy smoke、copy real run、wikitext smoke、wikitext real run 的 `graph_artifact_sha256` 完全一致；
4. zigzag_certified_cosine 与 zigzag_certified 的唯一差别是 lr_scheduler；
5. random_regular 不允许用配置 d 直接近似预算，必须按 zigzag actual post-causal K 对齐；
6. 如果某个 method 失败，不删除结果；写入 status=failed、failure_reason、error.log 和 summary。
```

学习率调度对比：

```text
default lr_scheduler = constant
only method zigzag_certified_cosine:
  lr_scheduler = cosine
  base_learning_rate = 与 zigzag_certified 相同
  warmup_ratio = 0.03
  min_lr_ratio = 0.1
  cosine_total_steps = 当前任务完整训练步数
```

## 4. Phase 1: 修改代码并适配两个任务

Phase 1 目标是修改当前代码，而不是重新做 v0.6 的 phase。完成后，copy 和 wikitext 应共享同一套结构参数解析、canonical graph artifact 读取/复制/sha 校验、budget 诊断、日志字段和结果字段。

必须完成：

```text
1. copy 支持 N_total=1024，并从 N_total 推导 copy_source_length=511；
2. wikitext 支持 sequence_length=1024；
3. 统一 B=32,q=32,d=8 的 runtime config 校验；
4. 支持按 v07 固定参数生成唯一 canonical graph，并写入 outputs/v07_graph_n1024_q32_B32_d8/；
5. copy/wikitext run 启动前把 canonical graph 复制到当前 output_dir/artifacts/graph/；
6. resolved_config_snapshot.json 引用当前 output_dir/artifacts/graph/ 内的 graph artifact 副本；
7. zigzag 实际 post-causal K 可在训练前或训练开始前 dry-run 得到；
8. random_regular 使用 zigzag 实际 K 对齐；
9. wikitext 数据准备、tokenizer 训练、tokenize train/test 可作为独立 Phase 4 执行；
10. wikitext 训练入口只读取 Phase 4 的固定数据/tokenizer 产物，不重新下载、不重新训练 tokenizer、不重新 tokenize；
11. copy 和 wikitext 都写训练动态 metrics、训练曲线 PNG、总耗时；
12. copy 和 wikitext 都写 raw_config_snapshot.json 与 resolved_config_snapshot.json；
13. 每次代码大更新后按环境文档提交 git。
```

建议代码入口：

```text
scripts/run_experiment.py
  统一 copy/wikitext 训练入口。

scripts/graph_diagnostics.py 或 scripts/generate_graph_artifact.py
  生成 v07 canonical graph artifact，并写出 sha256 与 certificate。

scripts/prepare_wikitext.py
  wikitext 数据下载、data readiness、tokenizer 训练和 train/test tokenize。

scripts/repair_copy_outputs.py
  copy 训练完成后的离线结果修复工具，只修产物，不重跑训练。
```

如果暂时继续使用现有入口，也必须保证输出字段和目录结构与本手册一致。

Phase 1 smoke 通过条件：

```text
python -m py_compile scripts/*.py scripts/synthetic_mvp_core/*.py 通过；
copy 配置解析和最小 dry-run 可跑；
wikitext 配置解析和最小 dry-run 可跑；
Phase 2 可生成 canonical graph artifact；
Phase 4 可独立生成 wikitext 数据/tokenizer/tokenized blocks；
run 启动时可把 canonical graph 复制到本次 output_dir/artifacts/graph/；
resolved config 指向本次 output_dir 内的 graph artifact 副本；
metrics.jsonl 包含时间字段。
```

## 5. Phase 2: 生成唯一 v07 Graph Artifact

Phase 2 只负责生成并冻结 v0.7 的 canonical graph，不训练模型。

输出目录：

```text
outputs/v07_graph_n1024_q32_B32_d8/
```

必须执行：

```text
1. 使用固定 graph_seed=0 和固定 graph_generation_algorithm 生成 graph；
2. 使用 N_total=1024,T=1024,q=32,B=32,d=8；
3. 允许 G 出现重边，并保留 multiplicity；
4. 写出 selected_graph.json、graph_certificate.json、graph_generation.json；
5. 计算 selected_graph.json 的 canonical_graph_artifact_sha256；
6. 在 graph_generation.json 和 summary.json 中写入生成命令、代码版本、生成算法标识、固定参数和 sha256；
7. 不启动任何训练，直到 canonical graph sha256 已经稳定记录。
```

Phase 2 通过条件：

```text
selected_graph.json 存在；
graph_certificate.json 存在；
graph_generation.json 存在；
summary.json 存在；
graph_artifact.sha256 存在；
graph_seed = 0；
graph_generation_algorithm 非空；
canonical_graph_artifact_sha256 非空；
rho_zigzag_bound = sqrt(lambda_G^2 + 2 * mu_H^2 - lambda_G^2 * mu_H^2)；
rho_zigzag_certified 字段存在；
N_total/T/q/B/d 与 v0.7 固定参数一致；
allow_multiedges=true；
preserve_multiplicity=true。
```

Phase 3 和 Phase 5 的四个训练 run 必须把这里生成的 graph 复制到各自输出目录，并验证：

```text
graph_seed == canonical_graph_seed
graph_generation_algorithm == canonical_graph_generation_algorithm
graph_artifact_sha256 == canonical_graph_artifact_sha256
```

## 6. 训练动态和耗时字段

每一行训练动态日志都必须记录当前行时间，以及相对上一行报告的耗时。

`metrics.jsonl` 每行至少包含：

```text
timestamp_utc
step
epoch
elapsed_sec_total
seconds_since_prev_log
train_loss
eval_loss 或 running_eval_loss
tokens_per_sec
learning_rate
lr_scheduler
peak_allocated_gb
peak_reserved_gb
```

copy 额外记录：

```text
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
```

wikitext 额外记录：

```text
learning_rate
running_train_perplexity
test_loss 或 latest_eval_loss
test_perplexity 或 latest_eval_perplexity
```

`training_curves.png` 必须从 `metrics.jsonl` 生成。copy 至少包含：

```text
train loss vs step
eval loss vs step
learning rate vs step
eval token accuracy vs step
eval sequence accuracy vs step
eval EOS accuracy vs step
seconds_since_prev_log vs step
tokens/sec vs step
```

wikitext 至少包含：

```text
train loss vs step
running train perplexity vs step
learning rate vs step
seconds_since_prev_log vs step
train tokens/sec vs step
test loss/perplexity final marker
```

最终 `summary.json`、`results.csv` 和 `results.jsonl` 必须包含总时间：

```text
total_wall_time_sec
train_wall_time_sec
eval_wall_time_sec
data_prep_wall_time_sec
```

如果某项不适用，填空字符串或 `0`，但 `total_wall_time_sec` 必须存在。

## 7. Phase 3: Copy Smoke + Real Run

copy 输出目录：

```text
outputs/copy_v07_smoke_n1024_q32_B32_d8/
outputs/copy_v07_main_n1024_q32_B32_d8/
```

copy smoke 参数：

```text
steps = 200
batch_size = 32
eval.batch_size = 32
eval.eval_batches = 5
log_every = 10
eval_every = 10
methods = dense, local, zigzag_certified, random_regular
optional_methods = zigzag_certified_cosine
```

copy real run 参数：

```text
steps = 5000
batch_size = 64
eval.batch_size = 64
eval.eval_batches = 50
learning_rate = 0.001
default_lr_scheduler = constant
zigzag_certified_cosine_lr_scheduler = cosine
zigzag_certified_cosine_warmup_ratio = 0.03
zigzag_certified_cosine_min_lr_ratio = 0.1
log_every = 50
eval_every = 50
checkpoint_every = 500
methods = dense, local, zigzag_certified, zigzag_certified_cosine, random_regular, zigzag_boolean
```

如果 copy 出现 OOM，允许先把 `train.batch_size/eval.batch_size` 从 `64` 降到 `32`；仍 OOM 再降到 `16`。同一次对比中的所有 method 必须使用相同 batch 设置，并在 results 中记录实际 batch。

推荐模型参数：

```text
layers = 8
d_model = 128
heads = 4
ffn_dim = 256
dropout = 0.1
```

copy 每个 run 必须包含：

```text
summary.json
results.csv
results.jsonl
metrics.jsonl
training_curves.png
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
artifacts/graph/selected_graph.json
artifacts/graph/graph_certificate.json
artifacts/graph/graph_generation.json
zigzag_budget.json
random_budget.json
shortcut_diagnostics.csv
shortcut_diagnostics.jsonl
```

copy 通过条件：

```text
smoke 无 NaN；
real run 所有要求方法完成或失败原因明确；
random_regular 已对齐 zigzag actual K；
training_curves.png 存在且包含 seconds_since_prev_log；
summary/results 中 total_wall_time_sec 存在；
graph artifact 已从 Phase 2 canonical graph 复制到本次输出目录；
graph_artifact_sha256 与 canonical_graph_artifact_sha256 完全一致；
shortcut diagnostics 使用 within-L-hop 语义。
```

## 8. Phase 4: WikiText Data Preparation and Tokenization

Phase 4 在 copy real run 完成后启动，只准备 wikitext 数据和 tokenizer，不训练模型。完成后，Phase 5 的 smoke 和 real run 都必须读取本 phase 产物。

Phase 4 输出目录：

```text
datasets/wikitext_v07_raw/
outputs/wikitext_v07_data_tokenize_n1024/
```

必须执行：

```text
1. 下载或加载 WikiText-103 raw train/test split；
2. 写出原始数据来源、revision/hash、行数、非空行数和本地缓存路径；
3. 只使用 train split 训练 byte_level_bpe tokenizer；
4. 使用同一个 tokenizer 编码 train/test；
5. 按 sequence_length=1024 构造 train/test blocks；
6. 写出 tokenized train/test 的路径、block 数、token 数和 sha256；
7. 写出 tokenizer.json、tokenizer_config.json、tokenizer_training.json；
8. 写出 data_readiness.json、tokenization_summary.json、summary.json 和 command.sh；
9. 写出总耗时与分段耗时。
```

Phase 4 产物至少包含：

```text
summary.json
data_readiness.json
tokenization_summary.json
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
artifacts/tokenizer/tokenizer.json
artifacts/tokenizer/tokenizer_config.json
artifacts/tokenizer/tokenizer_training.json
tokenized/train_blocks.*
tokenized/test_blocks.*
```

Phase 4 必须报告：

```text
dataset
dataset_source
dataset_revision_or_hash
dataset_cache_or_local_path
train_nonempty_rows
test_nonempty_rows
tokenizer_algorithm
tokenizer_train_split
tokenizer_vocab_size
tokenizer_min_frequency
tokenizer_sha256
tokenized_train_path
tokenized_train_sha256
tokenized_test_path
tokenized_test_sha256
train_token_count
train_block_count
test_token_count
test_block_count
data_download_wall_time_sec
tokenizer_train_wall_time_sec
tokenization_wall_time_sec
total_wall_time_sec
```

Phase 4 通过条件：

```text
train/test split 均非空；
tokenizer 训练只使用 train split；
tokenizer_sha256 非空；
tokenized train/test 文件存在；
tokenized_train_sha256 和 tokenized_test_sha256 非空；
sequence_length = 1024；
train_block_count 和 test_block_count 大于 0；
summary.json 中 total_wall_time_sec 存在。
```

## 9. Phase 5: WikiText Smoke + Real Run

wikitext 训练必须在 Phase 4 通过后启动。训练脚本不得重新下载数据、不得重新训练 tokenizer、不得重新 tokenize；只能读取 Phase 4 产物，并把必要 metadata/tokenizer 文件复制到自己的输出目录。

wikitext 输出目录：

```text
outputs/wikitext_v07_smoke_n1024_q32_B32_d8/
outputs/wikitext_v07_main_n1024_q32_B32_d8/
```

wikitext 固定输入：

```text
wikitext_data_phase_dir = outputs/wikitext_v07_data_tokenize_n1024/
tokenizer_path = outputs/wikitext_v07_data_tokenize_n1024/artifacts/tokenizer/tokenizer.json
tokenized_train_path = outputs/wikitext_v07_data_tokenize_n1024/tokenized/train_blocks.*
tokenized_test_path = outputs/wikitext_v07_data_tokenize_n1024/tokenized/test_blocks.*
```

wikitext smoke 参数：

```text
sequence_length = 1024
max_train_batches = 2
max_test_batches = 2
batch_size = 4
eval.batch_size = 4
eval.eval_batches = 2
methods = dense, local, zigzag_certified, random_regular
```

wikitext real run 参数：

```text
sequence_length = 1024
train_epochs = 1
batch_size = 16
gradient_accumulation_steps = 2
effective_batch_size = 32
eval.batch_size = 16
learning_rate = 0.0003
default_lr_scheduler = constant
zigzag_certified_cosine_lr_scheduler = cosine
zigzag_certified_cosine_warmup_ratio = 0.03
zigzag_certified_cosine_min_lr_ratio = 0.1
weight_decay = 0.01
grad_clip_norm = 1.0
log_every = 100
eval.split = test
eval.eval_batches = all
data_phase_dir = outputs/wikitext_v07_data_tokenize_n1024/
tokenizer_source = phase4
require_tokenizer_sha256_match = true
require_tokenized_sha256_match = true
methods = dense, local, zigzag_certified, zigzag_certified_cosine, random_regular, zigzag_boolean
```

如果 wikitext 出现 OOM，允许先把 `train.batch_size/eval.batch_size` 从 `16` 降到 `8`，并把 `gradient_accumulation_steps` 提高到 `4` 以维持 `effective_batch_size=32`；仍 OOM 再降到 `train.batch_size=4, gradient_accumulation_steps=8`。同一次对比中的所有 method 必须使用相同 batch 和 accumulation 设置，并在 results 中记录实际值。

wikitext 每个 run 必须包含：

```text
summary.json
results.csv
results.jsonl
metrics.jsonl
training_curves.png
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
data_readiness.json
tokenization_summary.json
artifacts/tokenizer/tokenizer.json
artifacts/tokenizer/tokenizer_config.json
artifacts/tokenizer/tokenizer_training.json
phase4_data_artifact_manifest.json
artifacts/graph/selected_graph.json
artifacts/graph/graph_certificate.json
artifacts/graph/graph_generation.json
zigzag_budget.json
random_budget.json
```

wikitext 必须报告：

```text
train_epoch_count
train_token_count
train_block_count
final_train_loss
test_token_count
test_block_count
test_loss
test_perplexity
train_tokens_per_sec
test_tokens_per_sec
total_wall_time_sec
peak_allocated_gb
peak_reserved_gb
dataset_source
dataset_revision_or_hash
tokenizer
tokenizer_sha256
tokenized_train_sha256
tokenized_test_sha256
```

wikitext 通过条件：

```text
smoke 无 NaN；
real run 完成 1 epoch train；
使用 Phase 4 的 tokenizer 和 tokenized train/test；
tokenizer_sha256、tokenized_train_sha256、tokenized_test_sha256 与 Phase 4 记录一致；
完整 test split 已评测，或明确记录无法完整评测的原因；
training_curves.png 存在且包含 seconds_since_prev_log；
summary/results 中 total_wall_time_sec 存在；
graph artifact 已从 Phase 2 canonical graph 复制到本次输出目录；
graph_artifact_sha256 与 canonical_graph_artifact_sha256 完全一致；
test_loss 和 test_perplexity 已记录。
```

## 10. Config 模板

配置 schema 中 copy 和 wikitext 都使用独立的 `eval` 对象；结果表里的统一字段 `eval_batch_size` 从 `eval.batch_size` 派生。

### 10.1 Graph Artifact

```json
{
  "version": "v07",
  "task": {
    "name": "graph_only",
    "N_total": 1024,
    "T": 1024
  },
  "structure": {
    "B": 32,
    "q": 32,
    "d": 8,
    "allow_multiedges": true,
    "preserve_multiplicity": true,
    "rho_field": "rho_zigzag_bound"
  },
  "graph": {
    "generate": true,
    "graph_seed": 0,
    "graph_generation_algorithm": "zigzag_v07_fixed_N1024_q32_B32_d8",
    "allow_multiedges": true,
    "preserve_multiplicity": true,
    "write_sha256": true
  },
  "output": {
    "root": "outputs/v07_graph_n1024_q32_B32_d8"
  }
}
```

### 10.2 Copy Main

```json
{
  "version": "v07",
  "task": {
    "name": "copy",
    "N_total": 1024,
    "copy_source_length": 511,
    "num_values": 4
  },
  "structure": {
    "B": 32,
    "q": 32,
    "d": 8,
    "allow_multiedges": true,
    "preserve_multiplicity": true,
    "rho_field": "rho_zigzag_bound"
  },
  "graph": {
    "generate": false,
    "source_dir": "outputs/v07_graph_n1024_q32_B32_d8",
    "copy_to_subdir": "artifacts/graph",
    "graph_seed": 0,
    "graph_generation_algorithm": "zigzag_v07_fixed_N1024_q32_B32_d8",
    "expected_graph_artifact_sha256": "<filled_after_phase_2>",
    "require_sha256_match": true,
    "allow_multiedges": true
  },
  "attention": {
    "methods": ["dense", "local", "zigzag_certified", "zigzag_certified_cosine", "random_regular", "zigzag_boolean"],
    "causal": true,
    "random_alignment_mode": "per_query",
    "random_target_k_source": "zigzag_actual_post_causal"
  },
  "method_overrides": {
    "zigzag_certified_cosine": {
      "lr_scheduler": "cosine",
      "warmup_ratio": 0.03,
      "min_lr_ratio": 0.1,
      "cosine_total_steps": 5000
    }
  },
  "model": {
    "layers": 8,
    "d_model": 128,
    "heads": 4,
    "ffn_dim": 256,
    "dropout": 0.1
  },
  "train": {
    "steps": 5000,
    "batch_size": 64,
    "learning_rate": 0.001,
    "log_every": 50,
    "eval_every": 50,
    "checkpoint_every": 500,
    "seeds": [0]
  },
  "eval": {
    "batch_size": 64,
    "eval_batches": 50
  },
  "output": {
    "root": "outputs/copy_v07_main_n1024_q32_B32_d8",
    "plot_curves": true,
    "curve_format": "png"
  }
}
```

### 10.3 WikiText Data and Tokenize

```json
{
  "version": "v07",
  "task": {
    "name": "wikitext_data_tokenize",
    "dataset": "wikitext-103-raw-v1",
    "preferred_source": "Salesforce/wikitext",
    "fallback_source": "iohadrubin/wikitext-103-raw-v1",
    "raw_dataset_dir": "datasets/wikitext_v07_raw",
    "train_split": "train",
    "test_split": "test",
    "sequence_length": 1024
  },
  "tokenizer": {
    "algorithm": "byte_level_bpe",
    "train_from_split": "train",
    "vocab_size": 32000,
    "min_frequency": 2,
    "special_tokens": ["<pad>", "<eos>", "<unk>"],
    "pad_token": "<pad>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "output_subdir": "artifacts/tokenizer"
  },
  "tokenize": {
    "output_subdir": "tokenized",
    "train_output": "train_blocks",
    "test_output": "test_blocks",
    "append_eos": true,
    "drop_last_incomplete_block": true,
    "write_sha256": true
  },
  "output": {
    "root": "outputs/wikitext_v07_data_tokenize_n1024"
  }
}
```

### 10.4 WikiText Main

```json
{
  "version": "v07",
  "task": {
    "name": "wikitext",
    "data_phase_dir": "outputs/wikitext_v07_data_tokenize_n1024",
    "train_split": "train",
    "test_split": "test",
    "sequence_length": 1024
  },
  "structure": {
    "B": 32,
    "q": 32,
    "d": 8,
    "allow_multiedges": true,
    "preserve_multiplicity": true,
    "rho_field": "rho_zigzag_bound"
  },
  "graph": {
    "generate": false,
    "source_dir": "outputs/v07_graph_n1024_q32_B32_d8",
    "copy_to_subdir": "artifacts/graph",
    "graph_seed": 0,
    "graph_generation_algorithm": "zigzag_v07_fixed_N1024_q32_B32_d8",
    "expected_graph_artifact_sha256": "<filled_after_phase_2>",
    "require_sha256_match": true,
    "allow_multiedges": true
  },
  "attention": {
    "methods": ["dense", "local", "zigzag_certified", "zigzag_certified_cosine", "random_regular", "zigzag_boolean"],
    "causal": true,
    "random_alignment_mode": "per_query",
    "random_target_k_source": "zigzag_actual_post_causal"
  },
  "method_overrides": {
    "zigzag_certified_cosine": {
      "lr_scheduler": "cosine",
      "warmup_ratio": 0.03,
      "min_lr_ratio": 0.1,
      "cosine_total_steps": "train_total_steps"
    }
  },
  "data": {
    "source_dir": "outputs/wikitext_v07_data_tokenize_n1024",
    "data_readiness_path": "outputs/wikitext_v07_data_tokenize_n1024/data_readiness.json",
    "tokenization_summary_path": "outputs/wikitext_v07_data_tokenize_n1024/tokenization_summary.json",
    "tokenized_train_path": "outputs/wikitext_v07_data_tokenize_n1024/tokenized/train_blocks.*",
    "tokenized_test_path": "outputs/wikitext_v07_data_tokenize_n1024/tokenized/test_blocks.*",
    "expected_tokenized_train_sha256": "<filled_after_phase_4>",
    "expected_tokenized_test_sha256": "<filled_after_phase_4>",
    "require_sha256_match": true
  },
  "tokenizer": {
    "train": false,
    "source_dir": "outputs/wikitext_v07_data_tokenize_n1024/artifacts/tokenizer",
    "path": "outputs/wikitext_v07_data_tokenize_n1024/artifacts/tokenizer/tokenizer.json",
    "expected_tokenizer_sha256": "<filled_after_phase_4>",
    "require_sha256_match": true
  },
  "model": {
    "layers": 8,
    "d_model": 128,
    "heads": 4,
    "ffn_dim": 256,
    "dropout": 0.1
  },
  "train": {
    "epochs": 1,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "log_every": 100,
    "seeds": [0]
  },
  "eval": {
    "split": "test",
    "batch_size": 16,
    "eval_batches": "all"
  },
  "output": {
    "root": "outputs/wikitext_v07_main_n1024_q32_B32_d8",
    "plot_curves": true,
    "curve_format": "png"
  }
}
```

## 11. 结果字段

copy 和 wikitext 共享字段：

```text
version
task
run_id
status
failure_reason
timestamp_utc
host
local_or_remote
command
log_path
CUDA_VISIBLE_DEVICES
gpu_name
python_version
torch_version
method
seed
N_total
B
q
d
causal
graph_id
graph_seed
graph_generation_algorithm
canonical_graph_dir
canonical_graph_artifact_path
canonical_graph_artifact_sha256
canonical_graph_seed
canonical_graph_generation_algorithm
graph_generation_status
graph_generation_attempts
graph_artifact_path
graph_generation_path
graph_certificate_path
graph_artifact_sha256
graph_artifact_sha256_matches_canonical
graph_certificate_sha256
G_type
H_type
allow_multiedges
multiplicity_mode
lambda_G
mu_H
rho_zigzag_bound
rho_zigzag_certified
rho_zigzag_exact
rot_g_is_bijection
P_G_row_stochastic_error
P_G_col_stochastic_error
P_H_row_stochastic_error
P_H_col_stochastic_error
graph_certified
implementation_certified
theory_aligned_method
duplicate_rate
self_loop_rate
remote_local_overlap_mean
collision_count_mean
zigzag_actual_k_min_after_causal
zigzag_actual_k_mean_after_causal
zigzag_actual_k_max_after_causal
zigzag_attention_pair_count_after_causal
random_target_k_source
random_actual_k_min_after_causal
random_actual_k_mean_after_causal
random_actual_k_max_after_causal
random_attention_pair_count_after_causal
random_k_alignment_error_mean
random_k_alignment_error_max
random_alignment_mode
random_k_aligned_to_zigzag
attention_pair_count_after_causal
layers
d_model
heads
ffn_dim
dropout
optimizer
steps
train_epochs
batch_size
gradient_accumulation_steps
effective_batch_size
eval_batch_size
eval_batches
learning_rate
base_learning_rate
lr_scheduler
warmup_ratio
warmup_steps
min_lr_ratio
min_learning_rate
cosine_total_steps
weight_decay
grad_clip_norm
log_every
eval_every
checkpoint_every
training_curves_path
total_wall_time_sec
train_wall_time_sec
eval_wall_time_sec
data_prep_wall_time_sec
peak_allocated_gb
peak_reserved_gb
artifact_dir
metrics_path
summary_path
raw_config_snapshot_path
resolved_config_snapshot_path
git_commit
config_sha256
```

copy 额外字段：

```text
copy_source_length
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
target_in_1hop_rate
target_in_2hop_rate
target_in_Lhop_rate
average_shortest_path
unreachable_rate
```

wikitext 额外字段：

```text
dataset
dataset_source
dataset_revision_or_hash
dataset_cache_or_local_path
wikitext_data_phase_dir
data_readiness_path
data_readiness_sha256
tokenization_summary_path
tokenization_summary_sha256
tokenizer
tokenizer_algorithm
tokenizer_train_split
tokenizer_path
tokenizer_sha256
tokenizer_min_frequency
tokenizer_special_tokens
pad_token
eos_token
unk_token
vocab_size
train_nonempty_rows
test_nonempty_rows
tokenized_train_path
tokenized_train_sha256
tokenized_test_path
tokenized_test_sha256
train_steps
train_token_count
train_block_count
test_token_count
test_block_count
final_train_loss
test_loss
test_perplexity
train_tokens_per_sec
test_tokens_per_sec
```

## 12. 命令模板

命令模板只表达 v0.7 的执行主线。具体脚本名可在 Phase 1 代码修改后确定，但输出目录和字段必须保持一致。

### 12.1 Graph Artifact

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/graph_diagnostics.py \
  --config configs/graph_v07_n1024_q32_B32_d8.json \
  --output-dir outputs/v07_graph_n1024_q32_B32_d8 \
  2>&1 | tee logs/graph_v07_n1024_q32_B32_d8_$(date +%Y%m%d_%H%M%S).log
```

### 12.2 Copy

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/run_experiment.py \
  --config configs/copy_v07_smoke_n1024_q32_B32_d8.json \
  --output-dir outputs/copy_v07_smoke_n1024_q32_B32_d8 \
  2>&1 | tee logs/copy_v07_smoke_n1024_q32_B32_d8_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/run_experiment.py \
  --config configs/copy_v07_main_n1024_q32_B32_d8.json \
  --output-dir outputs/copy_v07_main_n1024_q32_B32_d8 \
  2>&1 | tee logs/copy_v07_main_n1024_q32_B32_d8_$(date +%Y%m%d_%H%M%S).log
```

### 12.3 WikiText Data and Tokenize

```bash
python scripts/prepare_wikitext.py \
  --config configs/wikitext_v07_data_tokenize_n1024.json \
  --output-dir outputs/wikitext_v07_data_tokenize_n1024 \
  2>&1 | tee logs/wikitext_v07_data_tokenize_n1024_$(date +%Y%m%d_%H%M%S).log
```

### 12.4 WikiText Smoke and Real Run

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/run_experiment.py \
  --config configs/wikitext_v07_smoke_n1024_q32_B32_d8.json \
  --output-dir outputs/wikitext_v07_smoke_n1024_q32_B32_d8 \
  2>&1 | tee logs/wikitext_v07_smoke_n1024_q32_B32_d8_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=<free_gpu_id> python scripts/run_experiment.py \
  --config configs/wikitext_v07_main_n1024_q32_B32_d8.json \
  --output-dir outputs/wikitext_v07_main_n1024_q32_B32_d8 \
  2>&1 | tee logs/wikitext_v07_main_n1024_q32_B32_d8_$(date +%Y%m%d_%H%M%S).log
```

## 13. 结论限制

v0.7 允许回答：

```text
固定 N_total=1024,q=32,B=32,d=8 时 copy 是否能训练；
zigzag_certified 与 random_regular 在实际 K 对齐后的差异；
zigzag_certified constant LR 与 zigzag_certified_cosine 的收敛速度、最终质量和稳定性差异；
rho_zigzag_bound 是否与训练表现一致；
copy 中是否仍存在 shortcut；
wikitext 数据准备、tokenizer 训练和 tokenized train/test 的可复现性；
wikitext 在相同结构下的训练动态和 test PPL；
copy 与 wikitext 的总耗时和每段训练动态。
```

v0.7 不允许声称：

```text
多 seed 稳定性；
official WikiText benchmark；
大规模语言模型质量；
未对齐 random budget 下的 sparse attention 优势；
boolean zigzag 是 theory-aligned 主实现。
```
