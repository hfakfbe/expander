# 代码结构重构与维护方案

## 0. 文档定位

本文档定义当前仓库下一轮结构重构、代码减枝和维护收敛方案。目标不是继续在旧目录和旧命名上打补丁，而是建立一套清晰、可维护、可验证的主线代码结构。

本轮重构完成后，可以在 git commit message、tag 或 release 说明中标记为 `V10`。但是版本号不得进入正式文件名、目录名、模块名、类名、函数名或配置路径。

本文档只描述方案，不执行实际整理。

当前仓库已经完成一轮历史资产清理。本文档以清理后的状态为基线：

```text
configs/ 当前无正式配置文件；
outputs/ 当前无需要保留的运行产物；
reports/ 当前无需要保留的报告文件；
datasets/ 当前保留 copy、selective_copy、induction_associative_recall、lra_listops 四个已物化任务数据目录；
wikitext 尚未物化为当前 datasets/ 下的任务数据目录；
scripts/ 仍保留大量历史命名入口和核心逻辑，是下一步重构重点。
```

## 1. 核心原则

### 1.1 版本号不得进入路径命名

禁止在正式文件和目录中使用任何版本号标记，例如：

```text
v01
v02
v08
v10
V10
```

不允许出现：

```text
scripts/v10_run_task.py
configs/v10/
outputs/copy_corrected_v01/
datasets/probes_dense_to_one_easy_v02/
ref/*_v10.md
```

允许版本号出现的位置仅限：

```text
git commit message
git tag
release note
manifest 内的 release 字段
历史报告正文
迁移说明正文
```

### 1.2 过程词不得成为核心领域名

以下词汇是历史过程词，不应作为核心代码命名：

```text
probe
corrected
phase
dryrun
smoke
readiness
legacy
old
archive
dense_to_one_easy
```

这些词可以出现在历史报告正文中，但不得出现在主线模块、目录、正式配置文件名、类名、函数名或运行产物路径中。

替换原则：

| 历史词 | 主线替代 |
|---|---|
| probe | task / task suite / 具体任务名 |
| corrected | dataset policy / split policy / canonical data |
| phase | migration step / check / release milestone |
| smoke | quick check / test profile |
| dryrun | interface check |
| readiness | environment check / data check |
| legacy | compatibility adapter / migration tool |
| old | source / previous split / deprecated input |

### 1.3 `src/` 不再嵌套项目名

目标结构直接使用 `src/` 作为代码根，不再使用 `src/expander/` 这类额外项目名层级。

原因：

```text
当前项目规模不大，额外包名层级不会提升边界清晰度；
用户明确要求避免多余嵌套；
任务、图、模型、训练、IO 已经足够表达模块职责。
```

### 1.4 五个任务并列

主线任务为五个并列任务：

```text
copy
wikitext
selective_copy
induction_associative_recall
lra_listops
```

它们都放在 `src/tasks/` 下，不能把 `copy`、`wikitext` 单独放在一套体系，把其他三个称为 `probes`。

## 2. 当前历史包袱

### 2.1 历史命名已经侵入核心代码

当前 `scripts/` 中大量核心类、函数、字段仍使用历史过程词：

```text
scripts/probe_common.py
scripts/probe_tasks.py
scripts/probe_metrics.py
scripts/run_probe_experiment.py
scripts/run_copy_corrected.py
scripts/materialize_probes_corrected.py
scripts/prepare_probes_corrected.py
```

代表性问题：

```text
EXPERIMENT_VERSION = "v08"
SELECTED_PROBES
ProbeEncoder
ProbeBatch
ProbeTransformer
copy_corrected_v01_l8_log5
probes_corrected_valid_as_test_l8_log5
```

这些命名会让新代码继续背负旧实验叙事，必须在重构中替换为任务、数据契约、训练运行时等中性概念。

### 2.2 结果契约绑定历史阶段

当前结果字段中存在：

```text
phase4_manifest_path
phase4_manifest_sha256
phase4_task_parameter_record_path
N_total_v07_alias
B_v07_alias
q_v07_alias
d_v07_alias
v07_alias_replacement_reason
```

问题：

```text
结果 schema 绑定 v07/v08/phase4；
字段含义是当前运行契约的一部分，不是历史记录；
后续任何训练结果都会继续输出旧版本语义。
```

处理原则：

```text
将 phase4_manifest_* 改为 task_manifest_*；
将 phase4_task_parameter_record_* 改为 task_spec_record_*；
删除 v07_alias_* 字段；
N_total/B/q/d 若仍有理论意义，应改为 graph/node/block 语义字段；
历史兼容字段只允许在迁移读取工具中出现。
```

### 2.3 v06/v07 兼容逻辑仍在主路径

当前代码中仍有：

```text
scripts/v07_artifacts.py
scripts/synthetic_mvp_core/config.py 默认 version = v06
scripts/synthetic_mvp_core/config.py 对 version == v07 的特殊校验
scripts/graph_structures.py 默认 version = v06
outputs/copy_v06_graph_search 引用
```

问题：

```text
旧版本不是被隔离为迁移工具，而是影响主配置和主 artifact 生成；
新任务运行时会被迫理解 v06/v07 的特殊规则；
代码读者无法判断哪些是当前逻辑，哪些是历史兼容。
```

处理原则：

```text
图 artifact 相关能力迁入 src/graph/；
v07_artifacts.py 的通用函数改为中性命名；
旧 artifact 读取如果仍必须支持，放入 tools/migrate_* 或 tools/inspect_*；
主训练路径不得根据 version == v06/v07 分支。
```

### 2.4 个人路径和远程机器路径硬编码

当前代码仍包含：

```text
/Users/sxye/Documents/expander_bench
/home/huiwei/ysx/expander_bench
/home/huiwei/ysx/zigzag_attention
```

问题：

```text
脚本不可复用；
CI 和新机器无法运行；
路径判断被当成 local/remote 语义；
远端脚本依赖个人 conda 路径。
```

处理原则：

```text
所有数据根目录通过 CLI 参数、环境变量或配置传入；
不再根据 cwd 前缀判断 local/remote；
remote/local 只作为运行环境元数据，由显式参数或 host 信息解析；
删除硬编码远程 shell 脚本，必要时改成参数化工具。
```

### 2.5 一次性实验代码仍残留在脚本层

历史配置、数据、输出和报告中的 dense-to-one 实验资产已经清理；当前仍残留的是脚本层入口和生成逻辑：

```text
scripts/create_probe_dense_to_one_easy.py
scripts/plot_probe_dense_to_one_easy_v02_loss_curves.py
```

问题：

```text
文件名含历史过程词和实验版本；
生成器固定输出路径和固定 manifest；
不是可复用工具；
若继续保留，会重新生成已清理掉的历史资产。
```

处理原则：

```text
若仅为一次性校准实验，删除脚本；
若需要作为测试 fixture，重命名为中性 fixture，并放入 tests/fixtures/；
报告可保留在 ref/ 或 reports/，但不得作为主线入口引用。
```

### 2.6 任务范围仍残留旧评估集合

当前代码中存在：

```text
niah_kv_retrieval
ruler
lra_pathfinder
lra_pathx
BLOCKED_TASKS
ensure_no_forbidden_probe
```

问题：

```text
主线目标是五个任务；
旧候选任务的 blocked/forbidden 逻辑污染当前运行时；
任务集合边界不清晰。
```

处理原则：

```text
主代码只注册五个任务；
旧候选任务写入历史报告或迁移说明；
不得在主运行时代码中保留 blocked historical task 分支。
```

## 3. 目标目录结构

目标结构如下：

```text
src/
  cli/
    run_task.py
    prepare_data.py
    materialize_data.py
    audit_data.py

  tasks/
    copy/
      spec.py
      data.py
      encoder.py
      batch.py
      metrics.py

    wikitext/
      spec.py
      data.py
      tokenizer.py
      batch.py
      metrics.py

    selective_copy/
      spec.py
      data.py
      encoder.py
      batch.py
      metrics.py

    induction_associative_recall/
      spec.py
      data.py
      encoder.py
      batch.py
      metrics.py

    lra_listops/
      spec.py
      data.py
      encoder.py
      batch.py
      metrics.py

  graph/
    structures.py
    generation.py
    diagnostics.py
    certificate.py
    reachability.py
    artifacts.py

  model/
    attention.py
    backends.py
    rotary.py
    transformer.py

  training/
    runner.py
    evaluation.py
    schedule.py
    checkpoints.py
    results.py

  config/
    loading.py
    schema.py
    validation.py

  io/
    json.py
    csv.py
    png.py
    hashing.py
    git.py
    manifest.py
    paths.py

scripts/
  run_task.py
  prepare_data.py
  materialize_data.py
  audit_data.py

configs/
  tasks/
  runs/
  contracts/

datasets/
  copy/
  wikitext/
  selective_copy/
  induction_associative_recall/
  lra_listops/

outputs/
  artifacts/

tests/
  test_task_contracts.py
  test_task_data.py
  test_graph_artifacts.py
  test_attention_backends.py
  test_training_runtime.py
```

说明：

```text
scripts/ 只保留薄 CLI wrapper；
src/ 保存全部核心逻辑；
configs/ 只按职责分层，不按版本分层；
outputs/ 不保存训练 run、checkpoint、日志堆积，只保存必要静态 artifact；
tests/ 接管原 scripts/test_*；
ref/ 保存方案、历史说明和人工文档；
reports/ 仅保存当前需要的人类报告，不作为运行入口。
```

## 4. 任务接口规范

五个任务必须实现同一套接口。每个 `src/tasks/<task>/spec.py` 至少暴露：

```python
TASK_NAME: str

def load_task_spec(path):
    ...

def validate_task_spec(spec):
    ...

def load_split(spec, split):
    ...

def build_encoder(spec):
    ...

def make_batch(rows, spec, encoder, device):
    ...

def compute_metrics(outputs, batch, spec):
    ...

def result_fields(spec):
    ...
```

训练运行时只依赖这些任务接口，不再知道历史上哪个任务来自 copy、wikitext 或 probe。

任务命名规则：

| 当前目标任务 | 目录名 | 说明 |
|---|---|---|
| copy | `copy` | 序列复制任务 |
| WikiText | `wikitext` | 语言建模或文本序列任务 |
| selective copy | `selective_copy` | 选择性复制任务 |
| induction associative recall | `induction_associative_recall` | 归纳/关联回忆任务 |
| LRA ListOps | `lra_listops` | ListOps 分类任务 |

任务目录名应与当前 `datasets/` 下的中性数据目录保持一致，避免在数据层和代码层之间再制造一套别名。

## 5. 配置结构规范

配置文件不带版本号，按职责命名：

```text
configs/tasks/copy.json
configs/tasks/wikitext.json
configs/tasks/selective_copy.json
configs/tasks/induction_associative_recall.json
configs/tasks/lra_listops.json

configs/runs/copy_dense.json
configs/runs/copy_zigzag.json
configs/runs/wikitext_dense.json
configs/runs/selective_copy_zigzag.json
configs/runs/induction_associative_recall_zigzag.json
configs/runs/lra_listops_zigzag.json

configs/contracts/task_result_fields.json
configs/contracts/graph_artifact.json
configs/contracts/run_manifest.json
```

配置中允许记录 release 信息，但字段不得影响路径：

```json
{
  "release": "V10",
  "task": "copy",
  "data": {
    "split_policy": "validation_as_test"
  }
}
```

禁止配置默认值写死个人路径。数据根目录优先级：

```text
CLI 参数
环境变量
配置相对路径
明确报错
```

## 6. 脚本保留标准

`scripts/` 下文件必须满足：

```text
文件名不含版本号；
文件名不含 probe/corrected/phase/smoke/dryrun；
只负责 argparse 和调用 src/；
可重复执行；
支持参数或配置；
不依赖个人路径；
有清晰 help 文本；
不写死实验输出目录。
```

目标脚本：

```text
scripts/run_task.py
scripts/prepare_data.py
scripts/materialize_data.py
scripts/audit_data.py
```

旧入口处理原则：

```text
不长期保留旧入口；
短期迁移时可保留兼容 wrapper，但 wrapper 文件名不能进入最终提交；
README 和报告中的命令全部更新为新入口；
最终主分支不保留 run_copy_corrected.py、run_probe_experiment.py 等历史命名入口。
```

## 7. 现有文件迁移映射

### 7.1 公共工具

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/probe_common.py` | `src/io/`, `src/config/`, `src/training/results.py` | 拆分，删除 probe 命名 |
| `scripts/png_utils.py` | `src/io/png.py` | 迁移 |
| `scripts/runtime_common.py` | `src/training/checkpoints.py` 和 `src/model/` | 拆分 |
| `scripts/sweep_summary_common.py` | 删除或 `src/training/results.py` | 仅保留通用汇总 |

### 7.2 图与 artifact

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/graph_structures.py` | `src/graph/structures.py` | 迁移，去掉 v06 默认 |
| `scripts/graph_diagnostics.py` | `src/graph/diagnostics.py` | 迁移，去掉 v07 分支 |
| `scripts/v07_artifacts.py` | `src/graph/artifacts.py` 或迁移工具 | 主路径不得保留 v07 命名 |
| `scripts/synthetic_mvp_core/artifacts.py` | `src/graph/artifacts.py` 和 `src/training/results.py` | 拆分 |

### 7.3 模型与训练

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/synthetic_mvp_core/attention.py` | `src/model/attention.py` | 迁移 |
| `scripts/synthetic_mvp_core/model.py` | `src/model/transformer.py` | 迁移 |
| `scripts/synthetic_mvp_core/config.py` | `src/config/loading.py` 和 `src/config/validation.py` | 拆分，删除版本分支 |
| `scripts/synthetic_mvp_core/training.py` | `src/training/runner.py` 和 `src/training/evaluation.py` | 拆分 |
| `scripts/synthetic_mvp.py` | 删除 | 历史兼容入口 |

### 7.4 任务代码

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/probe_tasks.py` | `src/tasks/*/` 和 `src/model/transformer.py` | 按任务拆分，删除 Probe 前缀 |
| `scripts/probe_metrics.py` | `src/tasks/*/metrics.py` 和 `src/training/results.py` | 拆分 |
| `scripts/wikitext2_utils.py` | `src/tasks/wikitext/` | 迁移，去掉数字命名 |
| `scripts/wikitext2_eval.py` | `src/tasks/wikitext/` 和 `src/cli/run_task.py` | 拆分 |
| `scripts/wikitext2_smoke.py` | `tests/` 或删除 | 不作为主入口 |

### 7.5 CLI 入口

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/run_copy_corrected.py` | `scripts/run_task.py` + `src/cli/run_task.py` | 替换 |
| `scripts/run_probe_experiment.py` | `scripts/run_task.py` + `src/cli/run_task.py` | 替换 |
| `scripts/run_probes_corrected.py` | 删除或并入新入口 | 删除历史命名 |
| `scripts/run_experiment.py` | `scripts/run_task.py` | 若仍有外部使用，短期迁移后删除 |

### 7.6 数据准备与审计

| 当前文件 | 目标位置 | 处理 |
|---|---|---|
| `scripts/materialize_copy_corrected.py` | `scripts/materialize_data.py` + `src/tasks/copy/data.py` | 替换 corrected 命名 |
| `scripts/materialize_probes_corrected.py` | `scripts/materialize_data.py` + `src/tasks/*/data.py` | 替换 probes/corrected 命名 |
| `scripts/prepare_copy_corrected.py` | `scripts/prepare_data.py` | 替换 |
| `scripts/prepare_probes_corrected.py` | `scripts/prepare_data.py` | 替换 |
| `scripts/probe_data_audit.py` | `scripts/audit_data.py` | 仅保留通用审计 |
| `scripts/probe_remote_readiness.py` | 删除或环境检查工具 | 不保留 probe/remote 硬编码 |

### 7.7 一次性实验

| 当前文件 | 目标处理 |
|---|---|
| `scripts/create_probe_dense_to_one_easy.py` | 删除，或改成 `tests/fixtures/` 生成器 |
| `scripts/plot_probe_dense_to_one_easy_v02_loss_curves.py` | 删除，或改成通用结果绘图工具 |
| `scripts/probe_phase2_dryrun.py` | 删除 |
| `scripts/probe_parameter_selection.py` | 删除，或将仍有效的规则迁入 task spec |
| `scripts/repair_copy_outputs.py` | 若修复完成则删除；若仍需，改成迁移工具并加参数 |
| `scripts/run_copy_random_density_sweep_remote.sh` | 删除 |
| `scripts/run_copy_random_layerwise_density_sweep_remote.sh` | 删除 |

## 8. 迁移顺序

### 8.1 建立新骨架

创建目标目录：

```text
src/
src/cli/
src/tasks/
src/graph/
src/model/
src/training/
src/config/
src/io/
scripts/
tests/
configs/tasks/
configs/runs/
configs/contracts/
```

先放空 `__init__.py` 和接口文件，不迁行为。

通过检查：

```bash
python -m compileall -q src
```

### 8.2 迁移 IO 和公共工具

先迁移无业务含义的公共函数：

```text
JSON/JSONL/CSV 读写
sha256
git commit/dirty
command string
PNG 写图
路径解析
manifest 写入
```

禁止把 `probe_common.py` 整文件搬过去。必须按职责拆分。

通过检查：

```bash
python -m compileall -q src scripts
```

### 8.3 迁移 graph 能力

迁移：

```text
graph 结构生成
graph artifact 读写
certificate
reachability
diagnostics
method canonicalization
```

同时删除主路径中的：

```text
v06 默认值
v07 特殊校验
v07_artifacts 命名
copy_v06_graph_search 默认路径
```

旧 artifact 如果必须读取，放入显式迁移工具。

### 8.4 迁移 model 和 training

迁移：

```text
attention backend
RoPE
Transformer block
checkpoint manifest
训练循环
评估循环
结果聚合
学习率 schedule
```

拆分 `run_copy_corrected.py` 和 `run_probe_experiment.py` 中混杂的训练逻辑。训练运行时不得包含具体任务分支，任务差异由 `src/tasks/*` 接口提供。

### 8.5 拆分五个任务

按以下顺序迁移：

```text
copy
selective_copy
induction_associative_recall
lra_listops
wikitext
```

每迁移一个任务，就补齐：

```text
task spec
data loading
encoder/tokenizer
batch
metrics
contract test
```

### 8.6 合并 CLI

统一入口：

```bash
python scripts/materialize_data.py --task copy --config configs/tasks/copy.json
python scripts/prepare_data.py --task copy --config configs/tasks/copy.json
python scripts/run_task.py --task copy --config configs/runs/copy_zigzag.json
python scripts/audit_data.py --task copy --config configs/tasks/copy.json
```

旧入口只在迁移过程中用于对比，最终删除。

### 8.7 清理历史资产

删除或迁移：

```text
旧命名 scripts
旧命名 configs
旧命名 datasets
旧命名 outputs
历史 run/log/checkpoint
一次性报告生成器
硬编码远端 shell 脚本
```

保留历史说明时，写入 ref 文档正文，不保留历史目录堆积。

## 9. 删除与保留标准

### 9.1 必删

满足任一条件即删除：

```text
文件名含版本号；
文件名含 probe/corrected/phase/dryrun/smoke/readiness；
脚本写死个人路径；
脚本固定生成某个历史实验目录；
脚本只是一次性报告或一次性修复；
代码只服务旧候选任务；
旧版本兼容逻辑已经不在主流程需要；
大段注释掉的旧实现；
if False 或永远不可达分支。
```

### 9.2 可改造后保留

满足以下条件才允许保留：

```text
能服务五个主线任务之一；
支持参数化输入输出；
无个人路径；
无版本号路径；
无历史过程词命名；
有测试覆盖；
可通过统一 CLI 调用；
职责单一。
```

### 9.3 删除前需确认

以下类型删除前需要确认外部使用：

```text
README 中公开的 CLI；
外部脚本可能调用的入口；
CI/CD 或远端任务入口；
配置中仍引用的 artifact；
测试依赖的 fixture；
报告中作为可复现实验证据引用的轻量文件。
```

确认方式：

```bash
rg "scripts/<file>" README.md ref reports configs
rg "<config-name>" README.md ref reports scripts
rg "<dataset-or-output-path>" README.md ref reports configs scripts
```

## 10. 测试与验证

### 10.1 静态检查

```bash
python -m compileall -q src scripts tests
```

### 10.2 命名检查

正式代码和配置不得命中：

```bash
rg -n "v[0-9]{1,2}|V[0-9]{1,2}|probe|corrected|phase[0-9]|dense_to_one|dryrun|readiness|legacy|old|archive" \
  src scripts configs tests
```

允许例外：

```text
历史迁移说明正文；
测试中明确验证禁用词的字符串；
manifest release 字段；
第三方数据源原始名称，且只能出现在 manifest 的 source 字段中。
```

### 10.3 路径检查

```bash
rg -n "/Users/|/home/|expander_bench" src scripts configs tests README.md
```

正式代码中不得出现个人绝对路径。`expander_bench` 只能作为文档中的外部数据源说明，不能作为默认路径。

### 10.4 功能检查

每个任务至少通过：

```bash
python scripts/audit_data.py --task <task> --config configs/tasks/<task>.json
python scripts/run_task.py --task <task> --config configs/runs/<task>_dense.json --mode check
```

如果任务训练成本高，允许 `--mode check` 只跑小样本，但必须覆盖：

```text
加载数据；
构造 batch；
forward；
loss；
metric；
result manifest；
graph artifact 读取；
attention backend 初始化。
```

### 10.5 结果契约检查

必须验证：

```text
五个任务结果字段一致；
任务特有指标通过 task metrics 扩展；
没有 phase/v07/v08/corrected/probe 字段；
run manifest 可追溯 config、task spec、git commit、command、artifact sha256。
```

## 11. 最终验收标准

重构完成后，仓库应满足：

```text
scripts/ 只保留少量薄 CLI；
scripts/ 下无版本号文件；
src/ 下无项目名额外嵌套；
五个任务并列放在 src/tasks/；
主代码无 probe/corrected/phase/v06/v07/v08 命名；
配置不按版本号分目录；
无个人绝对路径；
无硬编码远端机器路径；
无一次性实验脚本；
无 old/backup/archive 堆积目录；
测试迁入 tests/；
核心入口统一；
所有保留脚本可重复执行；
核心测试通过；
配置引用无断链；
outputs/ 不保留 run/checkpoint/log 堆积。
```

## 12. 最终提交要求

完成实际整理后，做一个整体提交。提交信息可以使用：

```text
V10: restructure task runtime and remove legacy experiment layout
```

注意：

```text
V10 只出现在 commit message、tag 或 release 说明；
不得因此创建带 V10/v10 的文件或目录；
提交前必须运行静态检查、命名检查、路径检查和核心任务检查；
不得通过删除测试来让检查通过；
不得把废弃代码移动到 old/archive/backup。
```

## 13. 推荐执行切片

建议实际执行时分成以下提交或工作切片：

1. 新建中性目录骨架和接口，不改变行为。
2. 拆分 IO、hash、git、PNG、manifest 公共工具。
3. 迁移 graph 和 artifact，移除 v06/v07 主路径分支。
4. 迁移 model 和 training runtime。
5. 拆分五个任务接口。
6. 合并 CLI 入口。
7. 迁移测试到 `tests/`。
8. 清理一次性脚本、旧配置、旧输出和个人路径。
9. 更新 README 和 ref 说明。
10. 运行完整验证并提交 release 边界。

每个切片都应保持项目可编译，且不回退用户已有改动。
