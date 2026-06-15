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
11. checkpoint/tensor 文件不进入 git。
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
reports/v08_phase4_task_parameter_selection_report.md
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
不在 smoke 或 main 阶段临时悄悄改变已冻结参数；
不为某个 method 单独放宽长度、batch、训练预算或 eval budget。
```

通过条件：

```text
6 个 task 都有完整参数记录；
smoke 和 main config 都由同一份 task parameter manifest 派生；
每个参数选择都有来源、理由和对主方法效果的预期影响；
主结果字段所需的参数均可从 manifest 追溯；
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
command.sh
raw_config_snapshot.json
resolved_config_snapshot.json
```

通过条件：

```text
6 个 task 的所有 required methods 无 NaN；
loss 能下降或至少保持有限值；
validation/test metric 可计算；
每个 task 的 input/target decode 检查通过；
random_regular non-causal budget 对齐字段存在；
summary/resolved config 明确写入 attention_contract=non_causal、causal=false；
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
reports/v08_probe_main_eval_report.md
```

每行结果至少包含：

```text
run_id
task
subtask
method
seed
status
version_path
train_examples
validation_examples
test_examples
train_budget_policy
train_budget_value
completed_train_units
effective_batch_policy
effective_batch_size
learning_rate
lr_scheduler
attention_contract
causal
graph_directionality
sequence_length_min
sequence_length_mean
sequence_length_max
attention_k_min
attention_k_mean
attention_k_max
attention_pair_count
random_alignment_mode
train_loss_final
validation_loss_final
test_loss
primary_metric_name
primary_metric_value
secondary_metrics_json
total_wall_time_sec
train_wall_time_sec
eval_wall_time_sec
peak_allocated_gb
peak_reserved_gb
git_commit
remote_host
gpu_id
error_log
```

主结果中 `causal` 必须为 `false`，`attention_contract` 必须为 `non_causal`，`graph_directionality` 必须为 `directed`。若某行不满足这三项，只能进入 debug/failed/partial 表，不能进入 theory-aligned main comparison。

## 11. Metrics

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

## 12. 报告与提交

每个 phase 完成后至少写一份报告：

```text
reports/v08_phase1_probe_data_audit_report.md
reports/v08_phase2_code_adaptation_report.md
reports/v08_phase3_remote_readiness_report.md
reports/v08_phase4_task_parameter_selection_report.md
reports/v08_phase5_smoke_report.md
reports/v08_probe_main_eval_report.md
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
