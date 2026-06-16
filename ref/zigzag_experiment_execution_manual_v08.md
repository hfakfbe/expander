# Zig-Zag Sparse Attention Probe 实验执行手册 v0.8

## 0. 文档定位

v0.8 接在 v0.7 之后，不再继续 copy/WikiText 两任务主线。v0.7 的配置、输出、日志、报告和轻量环境快照已归档到：

```text
ref/archive_v07_reports/
```

v0.8 的唯一目标是：根据 `../expander_bench/data/probes/DEPLOYMENT_SUMMARY.md` 中已经验证通过的 6 个 probe 数据集，完成当前 expander/zigzag 代码对任务格式的适配、数据上传、每任务 smoke test、全量训练和评测。

重要前提：当前 expander 理论对齐只覆盖 `non-causal` 的有向图设定，probe 数据也是按 `causal=false` / `non-causal` contract 部署的。因此 v0.8 主实验默认且必须使用：

```text
attention_contract = non_causal
causal = false
graph_directionality = directed
```

本文档中若出现用户口头写法 `non-casual`，统一按 `non-causal` 理解。v0.8 不沿用 v0.7 的 causal LM mask 作为主实验设定；任何临时 causal debug run 都必须标记为 `debug_only`，不得进入主结果表或理论对齐结论。

共享环境、GPU 选择、git 提交、checkpoint 忽略和远端同步规则统一见：

```text
ref/experiment_environment_and_version_control.md
```

## 1. v08 数据范围

只允许读取 `status=validated` 且 `can_enter_main_eval=true` 的数据版本。以下目录来自 2026-06-15 的 probe 部署总结。这 6 个数据版本的部署合同是 `non-causal`，主评测不得把它们改写成 causal next-token 任务：

| task | selected version directory | train / validation / test |
|---|---|---|
| copy | `../expander_bench/data/probes/copy/s4_copying_length_extrapolation/copy_s4_l0_m1024_a64_full_v2/` | 10000 / 1000 / 1000 |
| selective_copy | `../expander_bench/data/probes/selective_copy/s4_variable_copy_regenerated/selective_copy_s4_l4096_m16_a16_full_v1/` | 10000 / 1000 / 1000 |
| induction_associative_recall | `../expander_bench/data/probes/induction_associative_recall/zoology_mqar_regenerated/mqar_vocab8192_len64_128_256_512_1024_full_v2/` | 180000 / 3000 / 4000 |
| niah_kv_retrieval | `../expander_bench/data/probes/niah_kv_retrieval/ruler_niah_single_1/niah_ruler_noise_4k_full_v2/` | 10000 / 500 / 500 |
| ruler | `../expander_bench/data/probes/ruler/ruler_official_nonqa_synthetic_suite/ruler_nonqa_suite_4k_full_v2/` | 12000 / 3000 / 3000 |
| lra_listops | `../expander_bench/data/probes/lra_listops/lra_official_generator_regenerated/lra_listops_regenerated_len500_2000_96k_2k_2k_full_v1/` | 96000 / 2000 / 2000 |

以下数据不得进入 v08 主评测：

```text
lra_pathfinder
lra_pathx
任何 status!=validated 的 version
任何 can_enter_main_eval!=true 的 version
```

每个 version 目录必须至少包含：

```text
README.md
deployment_report.md
dataset_card.json
config.yaml
source.lock
train.jsonl
validation.jsonl
test.jsonl
checksums.sha256
deployment_status.yaml
```

## 2. 总体 Phase

v0.8 分为 6 个 phase，必须按顺序推进：

| Phase | 名称 | 目标 | 通过条件 |
|---|---|---|---|
| 1 | Probe Data Contract Audit | 读取 6 个数据版本，确认 schema、长度、词表/标签、metric、non-causal 标志和 checksum | `outputs/probes_v08_data_audit/summary.json` 完整，6 个 task 均 validated |
| 2 | Code Adaptation | 修改训练/评测代码，支持 6 个 probe task 的 JSONL 输入、任务头、metric 和统一结果字段 | py_compile 通过，每个 task 可跑 tiny dry-run |
| 3 | Data Upload and Remote Readiness | 把代码和 6 个 probe 数据上传到远端，保存远端路径、sha256、行数和环境快照 | 远端 readiness 检查通过，6 个 task 文件数/sha 与本地一致 |
| 4 | Task Parameter Selection | 按 task 决定长度、encoder、模型尺寸、图参数、batch、训练预算、评测预算和 checkpoint 策略 | `configs/probes_v08_task_parameters.json` 与参数选择报告完整 |
| 5 | Per-Task Smoke Test | 读取 Phase 4 参数，对每个 task 跑 smoke train + eval | 6 个 task smoke 均无 NaN，能写 results/metrics/summary |
| 6 | Full Train + Eval | 读取 Phase 4 冻结参数，对 6 个 task 跑全量训练和测试评测 | 主结果表、每任务报告、失败审计和最终总报告完整 |

v08 不要求沿用 v07 的 `N=1024,B=32,q=32,d=8` 固定任务长度，也不沿用 v07 的 causal attention contract。每个 probe 的序列长度、词表大小、目标格式、metric、模型容量和训练预算必须由 Phase 1 audit、Phase 3 远端资源检查和 Phase 4 参数选择共同决定，再写入 resolved config。

但是 v08 必须继承 v07 的字段严谨度。v07 已经强制记录过的运行参数、图证书、budget、timing、环境、sha256、失败原因和产物路径，v08 不得因为“参数由 Phase 4 决定”而省略。数值可以不在本文档中预先规定，但字段必须存在，且必须填入 Phase 4 和运行时解析得到的 resolved value。

字段继承规则：

```text
1. v07 字段若在 v08 仍有同义含义，必须保留原字段名或给出明确 v08 replacement field；
2. v07 字段若因任务变化确实不适用，结果中必须写 not_applicable，并提供 not_applicable_reason，不得留空；
3. Phase 4 manifest、resolved_config_snapshot.json、summary.json、results.csv 和 results.jsonl 必须能相互追溯；
4. 每个 main result row 必须能单独重建该 run 的数据版本、参数选择、模型配置、图 artifact、budget 对齐、训练预算、评测预算、环境和代码版本；
5. 缺字段、空字段或只写 policy 不写 resolved value，均视为该 run 的 reporting failure。
```

## 3. 任务适配要求

所有 task 都从标准化 JSONL 读取样本。每行包含：

```text
id
task
variant
input
target
metadata
```

Phase 1 必须自动扫描并记录：

```text
task
version_path
dataset_card.status
dataset_card.can_enter_main_eval
split row counts
input type and representative shape
target type and representative shape
min/mean/max input length
min/mean/max target length
token/value vocabulary estimate
metadata keys
non-causal contract and causal=false verification
recommended metric
sha256 verification status
```

训练入口必须支持至少三类输出：

```text
sequence_generation:
  copy, selective_copy

key_value_or_token_retrieval:
  induction_associative_recall, niah_kv_retrieval, ruler

classification:
  lra_listops
```

如果 `ruler` 的 6 个子任务在 `metadata` 中可区分，结果必须同时写 task-level 和 subtask-level metrics。

## 4. 模型和 Method

v08 的第一版目标是把 6 个 probe 跑完整、跑可复现、跑可比较。候选方法池如下，具体每个 task 的 required/optional method 集合由 Phase 4 参数选择决定：

```text
dense
local
zigzag_certified
random_regular
```

方法语义固定在 `non-causal directed graph` 上解释：

```text
dense:
  non-causal dense attention reference。

local:
  non-causal block-local sparse baseline。

zigzag_certified:
  使用有向 expander / zigzag graph；这是当前唯一可写作 theory-aligned 的主方法。

random_regular:
  在同一 non-causal sparse budget 下的随机有向图 baseline。
```

可选扩展方法：

```text
zigzag_certified_cosine
zigzag_boolean
```

公平性约束：

```text
1. 同一 task 内所有 method 使用同一模型宽度、层数、训练步数、batch、seed 和数据 split；
2. sparse method 的 attention budget 必须记录 min/mean/max K 和 pair count；
3. random_regular 必须按 zigzag actual non-causal sparse K 或同任务实际 sparse budget 对齐；
4. 所有主结果必须显式记录 attention_contract=non_causal、causal=false、graph_directionality=directed；
5. 若临时只能先跑 dense/local 基线，主结果必须标记为 partial，不得冒充完整 v08。
```

参数选择优化准则：

```text
1. Phase 4 的目标不是寻找最低成本能跑通配置，而是在 non-causal directed contract、公平性、远端资源和可复现成本约束下，让主方法 zigzag_certified 的验证/测试效果尽可能好；
2. 长度覆盖、模型容量、sparse budget、训练预算和 eval budget 应优先服务主方法效果，只有在有明确 OOM、吞吐或总成本证据时才下调；
3. 若资源不足，优先缩减 optional methods、额外 seeds 或非关键诊断频率，再考虑降低会影响主方法效果的长度、容量或训练预算；
4. 对 required methods 的参数仍必须公平同步；如果为了主方法效果采用更强配置，dense/local/random_regular 也必须使用同一 task 级配置，除非该结果明确标记为 partial/diagnostic；
5. 每个 task 的 selection_reason 必须说明为什么该配置有利于主方法效果，而不只是说明它能跑通。
```

模型、长度和训练配置不从 v07 继承默认值。Phase 4 必须按任务逐项选择并冻结：

```text
input_length_policy
target_length_policy
token_or_value_encoder
label_space
loss_type
model_family
model_depth_width_heads
graph_block_or_node_policy
graph_degree_or_budget_policy
effective_batch_policy
optimizer_and_lr_policy
train_budget_policy
validation_eval_policy
test_eval_policy
checkpoint_policy
```

任何参数变更都必须保持同一 task 内 method 间公平。若因为 OOM 或吞吐问题调整某 task 的 batch、长度、图预算或训练预算，必须对该 task 的所有 required methods 使用同一新参数，并在 Phase 4 参数清单和后续结果中记录原因。

手册不预先规定 `max_steps`、`batch_size`、`eval_every`、`checkpoint_every` 或具体长度截断值；这些都属于 Phase 4 的产物。上一个 v07 版本的经验只能作为背景参考，不能直接作为 v08 默认配置。

## 5. Phase 1: Probe Data Contract Audit

新增或复用脚本：

```text
scripts/probe_data_audit.py
```

输出目录：

```text
outputs/probes_v08_data_audit/
```

必须输出：

```text
summary.json
task_audit.csv
task_audit.jsonl
checksums_verification.json
sample_preview.jsonl
command.sh
```

通过条件：

```text
6 个 task 都能读取 train/validation/test；
部署状态均为 validated/can_enter_main_eval=true；
checksums.sha256 校验通过，或记录可解释的非内容性差异；
每个 task 的 input/target schema 已明确；
每个 task 的 metric 已明确；
每个 task 的 non-causal contract 已确认，且 resolved config 中 causal=false；
失败任务 lra_pathfinder/lra_pathx 未被纳入 main plan。
```

## 6. Phase 2: Code Adaptation

训练和评测入口可以复用 `scripts/run_experiment.py`，也可以新增 probe 专用入口：

```text
scripts/run_probe_experiment.py
scripts/probe_tasks.py
scripts/probe_metrics.py
```

必须完成：

```text
1. 从 config 读取 task name、version path、split paths 和 metric；
2. 支持 train/validation/test JSONL streaming 或 mmap-friendly 读取；
3. 支持不同 task 的 tokenizer/value encoder/label encoder；
4. 支持 sequence generation、retrieval、classification 三类 loss；
5. 支持 validation early smoke eval 和 test final eval；
6. 写 raw_config_snapshot.json 与 resolved_config_snapshot.json；
7. 写 command.sh、metrics.jsonl、results.csv、results.jsonl、summary.json；
8. 失败时写 error.log 和 status=failed summary；
9. 默认 attention mask 为 non-causal；任何 causal mask 只能作为 debug-only 配置；
10. directed graph artifact、mask diagnostics 和 budget 字段必须能证明使用的是有向 non-causal 设定；
11. 写 training_curves.png，且曲线只能从 metrics.jsonl 生成；
12. 写 zigzag_budget.json、random_budget.json、attention_diagnostics.json 或等价 task/method 诊断文件；
13. 写 result_field_audit.json，检查 v08 必需字段是否存在、是否为空、是否可追溯到 Phase 4；
14. checkpoint/tensor 文件不进入 git，但 checkpoint_manifest.json 必须记录外部 checkpoint 路径、sha256、step 和保留策略。
```

Phase 2 interface dry-run：

```text
使用最小化 fixture 或少量样本只验证代码路径；
不在这里决定 task 长度、batch、模型尺寸或训练预算；
确认 forward/backward/eval 和结果写出即可。
```

通过条件：

```bash
python -m py_compile scripts/*.py scripts/synthetic_mvp_core/*.py
```

并且 6 个 task interface dry-run 均能完成或给出明确代码待修项。

## 7. Phase 3: Data Upload and Remote Readiness

默认远端路径：

```text
remote code root: /home/huiwei/ysx/zigzag_attention
remote probe data root: /home/huiwei/ysx/expander_bench/data/probes
```

同步命令建议：

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude '.deps/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  ./ huiwei:/home/huiwei/ysx/zigzag_attention/

rsync -av ../expander_bench/data/probes/ \
  huiwei:/home/huiwei/ysx/expander_bench/data/probes/
```

远端 readiness 输出：

```text
outputs/probes_v08_remote_readiness/
  summary.json
  remote_file_counts.csv
  remote_checksums_verification.json
  env_snapshot.txt
  requirements_snapshot.txt
  command.sh
```

通过条件：

```text
远端 6 个 selected version path 存在；
train/validation/test 行数与本地 audit 一致；
关键文件 sha256 与本地一致；
远端 Python、torch、CUDA 可用；
目标 GPU 空闲度符合环境文档。
```

## 8. Phase 4: Task Parameter Selection

Phase 4 专门决定每个任务的完整实验参数。它必须发生在数据审计、代码适配和远端资源检查之后，不能在手册里提前用固定数值替代。

输出文件：

```text
configs/probes_v08_task_parameters.json
configs/probes_v08_smoke.json
configs/probes_v08_main.json
configs/probes_v08_result_field_contract.json
reports/v08_phase4_task_parameter_selection_report.md
reports/v08_parameter_glossary.md
outputs/probes_v08_parameter_selection/summary.json
outputs/probes_v08_parameter_selection/task_parameters.csv
outputs/probes_v08_parameter_selection/task_parameters.jsonl
```

每个 task 必须独立确定并记录：

```text
task
version_path
attention_contract
causal
graph_directionality
input_schema
target_schema
primary_metric
secondary_metrics
input_length_min_mean_max
target_length_min_mean_max
chosen_train_length_policy
chosen_eval_length_policy
encoder_or_tokenizer
label_or_value_space
loss_type
model_family
model_capacity
graph_block_policy
graph_degree_or_budget_policy
required_methods
optional_methods
seed_policy
train_split_policy
validation_split_policy
test_split_policy
effective_batch_policy
optimizer_policy
lr_schedule_policy
train_budget_policy
validation_eval_policy
test_eval_policy
logging_policy
checkpoint_policy
oom_or_runtime_fallback_policy
selection_reason
```

Phase 4 不得只写 policy 名称，还必须写 resolved values。每个 task 的参数记录至少包含以下 resolved 字段；字段值由 Phase 4 决定，不由本文档预设：

```text
resolved_input_length_policy
resolved_target_length_policy
resolved_sequence_length_min
resolved_sequence_length_mean
resolved_sequence_length_p95
resolved_sequence_length_max
resolved_train_examples
resolved_validation_examples
resolved_test_examples
resolved_train_split_sha256
resolved_validation_split_sha256
resolved_test_split_sha256
resolved_encoder_type
resolved_tokenizer_or_encoder_path
resolved_tokenizer_or_encoder_sha256
resolved_vocab_or_value_space_size
resolved_label_space
resolved_loss_type
resolved_model_family
resolved_layers
resolved_d_model
resolved_heads
resolved_ffn_dim
resolved_dropout
resolved_parameter_count
resolved_attention_backend
resolved_graph_id
resolved_graph_seed
resolved_graph_generation_algorithm
resolved_graph_block_size
resolved_graph_num_blocks_or_nodes
resolved_graph_degree_or_budget
resolved_B_alias_if_applicable
resolved_q_alias_if_applicable
resolved_d_alias_if_applicable
resolved_required_methods
resolved_optional_methods
resolved_seeds
resolved_optimizer
resolved_learning_rate
resolved_base_learning_rate
resolved_lr_scheduler
resolved_warmup_ratio
resolved_warmup_steps
resolved_min_lr_ratio
resolved_min_learning_rate
resolved_weight_decay
resolved_grad_clip_norm
resolved_batch_size
resolved_gradient_accumulation_steps
resolved_effective_batch_size
resolved_eval_batch_size
resolved_train_budget_unit
resolved_train_budget_value
resolved_steps_planned_if_step_budget
resolved_epochs_planned_if_epoch_budget
resolved_log_every
resolved_log_step_policy
resolved_min_logged_train_step_count
resolved_planned_logged_train_step_count
resolved_log_coverage_ratio_min
resolved_eval_every
resolved_checkpoint_every
resolved_checkpoint_policy
resolved_validation_eval_budget
resolved_test_eval_budget
resolved_oom_fallback_sequence
```

logging policy 硬约束：

```text
1. Phase 4 必须把每个 task/method/seed 的 planned_train_step_count 解析成明确整数；若训练预算按 epoch、example 或 token 表达，也必须先换算出用于 logging gate 的 planned train steps；
2. 若 planned_train_step_count < 100，metrics.jsonl 必须记录每一个训练 step；
3. 若 planned_train_step_count >= 100，实际记录的训练 step 数不得少于 ceil(planned_train_step_count * 0.01)；
4. final train step 必须记录，即使它不落在 log_every 间隔上；
5. 若使用 log_every 作为间隔，resolved_log_every 必须保证 planned_logged_train_step_count 满足上面的下限；
6. logging gate 对 smoke 和 main 都生效，不能因为 smoke 短、main 长或训练很稳而放宽。
```

如果某个 v07 兼容字段无法直接映射，例如 `N_total/B/q/d`，必须写入 replacement 字段和原因：

```text
v07_field_name
v08_replacement_field
replacement_reason
resolved_value
```

参数选择依据必须至少包括：

```text
Phase 1 的长度/schema/metric/data size audit；
Phase 2 的代码能力和 dry-run 限制；
Phase 3 的远端 GPU/内存/吞吐可用性；
non-causal directed expander 的理论对齐要求；
主方法 zigzag_certified 的效果最大化目标；
同一 task 内 method 间公平性；
结果可复现性和总运行成本。
```

禁止事项：

```text
不在手册中硬编码 max_steps、batch_size、eval_every、checkpoint_every；
不把 v07 的 copy/WikiText 经验直接当作 v08 默认参数；
不把“smoke 能跑通”当作 main 参数选择充分理由；
不把 log_every 设到导致记录 step 数少于总 train step 1%；
不在 smoke 或 main 阶段临时悄悄改变已冻结参数；
不为某个 method 单独放宽长度、batch、训练预算或 eval budget。
```

通过条件：

```text
6 个 task 都有完整参数记录；
smoke 和 main config 都由同一份 task parameter manifest 派生；
result field contract 已生成，且覆盖本文档第 11 节的全部字段；
每个参数选择都有来源、理由和对主方法效果的预期影响；
主结果字段所需的参数均可从 manifest 追溯；
logging policy 已证明每个 run 的 planned logged train steps 满足 1% gate；
参数说明已写入 `reports/v08_phase4_task_parameter_selection_report.md` 和 `reports/v08_parameter_glossary.md`；
attention_contract=non_causal、causal=false、graph_directionality=directed 已冻结。
```

## 9. Phase 5: Per-Task Smoke Test

smoke 输出目录：

```text
outputs/probes_v08_smoke/<task>/<method>/
```

smoke 必须读取 Phase 4 产物：

```text
configs/probes_v08_task_parameters.json
configs/probes_v08_smoke.json
```

smoke 的样本数、步数、batch、eval/log 频率和 method 集合都由 Phase 4 决定。Smoke 只验证参数组合能否真实跑通，不重新选择参数。

每个 smoke run 必须写：

```text
summary.json
results.csv
results.jsonl
metrics.jsonl
training_curves.png
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
phase4_task_parameter_record.json
result_field_audit.json
checkpoint_manifest.json
artifacts/graph/selected_graph.json
artifacts/graph/graph_certificate.json
artifacts/graph/graph_generation.json
artifacts/graph/graph_artifact.sha256
zigzag_budget.json
random_budget.json
attention_diagnostics.json
error.log if failed
```

通过条件：

```text
6 个 task 的所有 required methods 无 NaN；
loss 能下降或至少保持有限值；
validation/test metric 可计算；
每个 task 的 input/target decode 检查通过；
random_regular non-causal budget 对齐字段存在；
summary/resolved config 明确写入 attention_contract=non_causal、causal=false；
result_field_audit.json 显示缺字段数量为 0；
actual_logged_train_step_count 满足 logging gate；若总训练步数不足 100，则每一步都已记录；
失败 run 保留 error.log，不删除。
```

如果 smoke 暴露参数不可行，例如 OOM、训练入口 schema 错误或 metric 无法计算，必须回到 Phase 4 修改参数 manifest，记录变更原因，并重新生成 smoke/main config。不能只改命令行临时绕过。

## 10. Phase 6: Full Train + Eval

main 输出目录：

```text
outputs/probes_v08_main/<task>/<method>/
```

main 必须读取 Phase 4 产物：

```text
configs/probes_v08_task_parameters.json
configs/probes_v08_main.json
```

main 的训练预算、validation/test 评测预算、checkpoint 策略和所有运行参数都由 Phase 4 冻结。若 smoke 之后需要改变任何主实验参数，必须更新 Phase 4 参数选择报告并重新提交对应 config。

主结果必须输出：

```text
outputs/probes_v08_main/results_all.csv
outputs/probes_v08_main/results_all.jsonl
outputs/probes_v08_main/summary.json
outputs/probes_v08_main/result_field_audit.json
reports/v08_probe_main_eval_report.md
```

每个 main run 目录必须包含：

```text
summary.json
results.csv
results.jsonl
metrics.jsonl
training_curves.png
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
phase4_task_parameter_record.json
result_field_audit.json
checkpoint_manifest.json
artifacts/graph/selected_graph.json
artifacts/graph/graph_certificate.json
artifacts/graph/graph_generation.json
artifacts/graph/graph_artifact.sha256
zigzag_budget.json
random_budget.json
attention_diagnostics.json
error.log if failed
```

每行结果至少包含第 11 节的完整字段集合，不得只写简化表。

主结果中 `causal` 必须为 `false`，`attention_contract` 必须为 `non_causal`，`graph_directionality` 必须为 `directed`。若某行不满足这三项，只能进入 debug/failed/partial 表，不能进入 theory-aligned main comparison。

main comparison 额外 gate：

```text
result_field_audit.json status=passed；
actual_logged_train_step_count >= ceil(logging_reference_train_steps * 0.01)；
若 logging_reference_train_steps < 100，则 actual_logged_train_step_count = logging_reference_train_steps；
final train step 已记录在 metrics.jsonl；
training_curves.png 覆盖 metrics.jsonl 中的全部记录点。
```

## 11. 输出字段和反偷懒审计

v08 的输出字段继承 v07 的强约束，并扩展到 6 个 probe task。以下字段是 `summary.json`、`results.csv`、`results.jsonl` 和合并后的 `results_all.*` 的最低字段集合。字段值必须来自 Phase 1 audit、Phase 4 manifest、resolved config、运行时测量或 artifact sha256，不能由报告阶段手填猜测。

### 11.1 通用 provenance 字段

```text
version
experiment_version
phase
task
subtask
variant
run_id
status
failure_reason
timestamp_utc
host
remote_host
local_or_remote
command
command_sha256
log_path
error_log
CUDA_VISIBLE_DEVICES
gpu_id
gpu_name
python_version
torch_version
cuda_version
git_commit
git_dirty
config_sha256
phase4_manifest_path
phase4_manifest_sha256
phase4_task_parameter_record_path
phase4_task_parameter_record_sha256
artifact_dir
external_artifact_manifest_path
```

### 11.2 数据和任务字段

```text
version_path
dataset
dataset_source
dataset_revision_or_hash
dataset_cache_or_local_path
dataset_card_path
dataset_card_sha256
deployment_status_path
deployment_status_sha256
source_lock_path
source_lock_sha256
checksums_path
checksums_sha256
train_path
train_sha256
validation_path
validation_sha256
test_path
test_sha256
train_examples
validation_examples
test_examples
train_examples_used
validation_examples_used
test_examples_used
train_split_policy
validation_split_policy
test_split_policy
input_schema
target_schema
metadata_keys
encoder_or_tokenizer
data_readiness_path
data_readiness_sha256
tokenization_summary_path
tokenization_summary_sha256
wikitext_data_phase_dir
tokenized_train_path
tokenized_train_sha256
tokenized_test_path
tokenized_test_sha256
tokenizer_or_encoder_path
tokenizer_or_encoder_sha256
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
train_token_count
train_block_count
test_token_count
test_block_count
label_or_value_space
loss_type
primary_metric_name
secondary_metric_names
sequence_length_min
sequence_length_mean
sequence_length_p95
sequence_length_max
target_length_min
target_length_mean
target_length_p95
target_length_max
```

### 11.3 模型、训练和评测字段

```text
method
method_role
required_or_optional_method
seed
model_family
layers
d_model
heads
ffn_dim
dropout
parameter_count
optimizer
steps
steps_planned
steps_completed
train_epochs
train_epochs_planned
train_epochs_completed
train_steps
train_budget_policy
train_budget_unit
train_budget_value
completed_train_units
train_examples_seen
train_tokens_seen
batch_size
gradient_accumulation_steps
effective_batch_size
eval_batch_size
validation_eval_budget
test_eval_budget
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
log_step_policy
logging_reference_train_steps
min_logged_train_step_count
planned_logged_train_step_count
actual_logged_train_step_count
log_coverage_ratio
log_policy_satisfied
eval_every
checkpoint_every
checkpoint_policy
checkpoint_manifest_path
```

### 11.4 Attention、graph 和 budget 字段

```text
attention_contract
causal
graph_directionality
attention_backend
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
selected_graph_sha256
graph_artifact_sha256
graph_artifact_sha256_matches_canonical
graph_certificate_sha256
graph_block_policy
graph_degree_or_budget_policy
graph_block_size
graph_num_blocks_or_nodes
graph_degree
N_total
B
q
d
N_total_v07_alias
B_v07_alias
q_v07_alias
d_v07_alias
v07_alias_replacement_reason
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
zigzag_actual_k_min_noncausal
zigzag_actual_k_mean_noncausal
zigzag_actual_k_max_noncausal
zigzag_attention_pair_count_noncausal
random_target_k_source
random_actual_k_min_after_causal
random_actual_k_mean_after_causal
random_actual_k_max_after_causal
random_attention_pair_count_after_causal
random_actual_k_min_noncausal
random_actual_k_mean_noncausal
random_actual_k_max_noncausal
random_attention_pair_count_noncausal
random_k_alignment_error_mean
random_k_alignment_error_max
random_alignment_mode
random_k_aligned_to_zigzag
attention_pair_count_after_causal
attention_k_min
attention_k_mean
attention_k_max
attention_pair_count
attention_diagnostics_path
zigzag_budget_path
random_budget_path
```

字段名必须使用 `noncausal` 或 `non_causal`，并在 result field audit 中检查无 `noncaual/non-casual` 拼写残留。上表语义对应的实际字段名必须在 `configs/probes_v08_result_field_contract.json` 中固定。

### 11.5 指标、时间和资源字段

```text
train_loss_final
validation_loss_final
test_loss
primary_metric_value
secondary_metrics_json
task_metrics_json
final_train_loss
test_perplexity
validation_perplexity_if_applicable
test_perplexity_if_applicable
train_tokens_per_sec
validation_tokens_per_sec
test_tokens_per_sec
train_examples_per_sec
validation_examples_per_sec
test_examples_per_sec
total_wall_time_sec
train_wall_time_sec
eval_wall_time_sec
data_prep_wall_time_sec
seconds_since_prev_log_mean
seconds_since_prev_log_max
peak_allocated_gb
peak_reserved_gb
oom_fallback_applied
oom_fallback_reason
training_curves_path
metrics_path
summary_path
raw_config_snapshot_path
resolved_config_snapshot_path
result_field_audit_path
```

### 11.6 Task-specific 字段

```text
copy_token_accuracy
copy_sequence_accuracy
copy_eos_accuracy
copy_source_length
eval_token_accuracy
eval_sequence_accuracy
eval_eos_accuracy
target_in_1hop_rate
target_in_2hop_rate
target_in_Lhop_rate
average_shortest_path
unreachable_rate
selective_copy_token_accuracy
selective_copy_sequence_accuracy
retrieval_exact_match
retrieval_token_accuracy
retrieval_answer_format
ruler_subtask
ruler_subtask_exact_match
ruler_subtask_token_accuracy
listops_accuracy
listops_macro_accuracy
listops_class_count
```

不相关任务的 task-specific 字段和 v07 compatibility 字段仍必须存在，值写 `not_applicable`，并由 `task_metrics_json` 或 `v07_alias_replacement_reason` 写明原因。不得因为任务不同而生成互不兼容的结果表。

### 11.7 metrics.jsonl 字段

`metrics.jsonl` 每行至少包含：

```text
run_id
task
subtask
method
seed
timestamp_utc
step
epoch
elapsed_sec_total
seconds_since_prev_log
split
phase
train_loss
running_train_loss
eval_loss
running_eval_loss
primary_metric_name
primary_metric_value
secondary_metrics_json
learning_rate
lr_scheduler
grad_norm
tokens_per_sec
examples_per_sec
peak_allocated_gb
peak_reserved_gb
nonfinite_loss_detected
nan_detected
```

若 `split=train`，metrics 行数必须满足 Phase 4 的 logging policy。字段审计必须用 `metrics.jsonl` 实际行数计算 `actual_logged_train_step_count`，不能只信 config 中的 `log_every`。

`training_curves.png` 必须从 `metrics.jsonl` 生成，至少包含：

```text
train loss vs step
validation/test loss vs step or final marker
primary metric vs step
learning rate vs step
seconds_since_prev_log vs step
tokens/sec or examples/sec vs step
peak memory vs step
```

### 11.8 字段审计 gate

每个 smoke/main run 结束后必须生成 `result_field_audit.json`：

```text
expected_field_count
present_field_count
missing_fields
empty_fields
not_applicable_fields
not_applicable_reasons
phase4_trace_missing_fields
resolved_config_trace_missing_fields
forbidden_spelling_hits
manual_only_policy_fields_without_resolved_value
logging_reference_train_steps
min_logged_train_step_count
actual_logged_train_step_count
log_coverage_ratio
log_policy_satisfied
final_train_step_logged
status
```

通过条件：

```text
missing_fields = []
phase4_trace_missing_fields = []
resolved_config_trace_missing_fields = []
manual_only_policy_fields_without_resolved_value = []
forbidden_spelling_hits = []
log_policy_satisfied = true
final_train_step_logged = true
status = passed
```

任何字段审计未通过的 run 可以保留为 failed/diagnostic，但不得进入 main comparison。

## 12. Metrics

以下只是 metric 候选口径，不能跳过 Phase 1 audit 和 Phase 4 参数选择。最终每个 task 的 `primary_metric`、`secondary_metrics` 和 subtask aggregation 必须写入 `configs/probes_v08_task_parameters.json`：

| task | primary metric | notes |
|---|---|---|
| copy | token_accuracy | also sequence_accuracy, eos_accuracy |
| selective_copy | token_accuracy | also sequence_accuracy |
| induction_associative_recall | exact_match or token_accuracy | choose by target schema after audit |
| niah_kv_retrieval | exact_match | also token_accuracy if generated target has tokens |
| ruler | exact_match | additionally subtask-level exact_match |
| lra_listops | accuracy | 10-class classification |

如果 audit 发现上表与数据 schema 不匹配，Phase 1 报告必须修正候选 metric；Phase 4 必须把最终 metric 选择和理由冻结到参数 manifest。

## 13. 报告与提交

每个 phase 完成后至少写一份报告：

```text
reports/v08_phase1_probe_data_audit_report.md
reports/v08_phase2_code_adaptation_report.md
reports/v08_phase3_remote_readiness_report.md
reports/v08_phase4_task_parameter_selection_report.md
reports/v08_phase5_smoke_report.md
reports/v08_probe_main_eval_report.md
```

报告语言和参数解释 gate：

```text
1. 所有 v08 报告必须使用中文撰写；英文参数名、method 名、metric 名可以保留原文，但解释必须是中文；
2. 每份报告必须包含“参数说明”或“字段说明”小节；
3. 报告中正文、表格、图注、命令摘要、结果摘要里出现的每个参数、字段、缩写、metric 和 resolved value，都必须在本报告的说明小节中简要解释；
4. 参数说明至少包含：参数名、中文含义、单位或取值范围、来源、为什么要记录；
5. 如果参数来自第 11 节字段契约，可以复用 `reports/v08_parameter_glossary.md` 的解释，但当前报告仍必须列出自己实际使用到的参数子集；
6. 不允许只贴英文表头、JSON 字段或 CSV 字段而不解释含义；
7. 不适用字段仍要解释为什么不适用，不能只写 not_applicable；
8. 每份报告末尾必须有报告审计小节，列出 report_language=zh、explained_parameter_count、unexplained_parameters、english_only_sections。
```

每份报告的参数说明表建议使用以下列：

```text
参数名
中文含义
单位或取值
来源
记录原因
不适用时的处理
```

通过条件：

```text
report_language = zh；
unexplained_parameters = []；
english_only_sections = []；
报告中出现的所有表格列名都能在参数说明表中找到；
关键参数如 attention_contract、causal、graph_directionality、train_budget_policy、log_every、actual_logged_train_step_count、primary_metric_value 均有中文解释。
```

提交建议：

```text
v08-doc-archive-v07
v08-probe-data-audit
v08-probe-code-adaptation
v08-probe-task-parameters
v08-probe-smoke
v08-probe-main-eval
```

提交前检查：

```bash
git status --short
python -m py_compile scripts/*.py scripts/synthetic_mvp_core/*.py
git ls-files -o --ignored --exclude-standard | rg '\\.(pt|pth|ckpt|safetensors)$' || true
```

当前版本不得把 checkpoint、tensor cache、大型原始缓存或 `.deps/` 提交到 git。若需要保留大型外部产物，在对应 archive README 或 phase report 中记录外部路径、文件数量、大小和原因。
