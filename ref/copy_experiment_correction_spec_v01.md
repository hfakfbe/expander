# Copy 实验更正规格 v0.2

## 0. 文档状态与适用优先级

本文档是对当前 Copy 实验实现的纠错规格，不是实现报告，也不授权直接开始正式训练。

本文档只约束新的 corrected Copy 实验。它不回写、不删除旧 v08 配置、日志和结果；旧产物保留为错误实现的历史证据。对于 Copy task，本文件的约束优先于：

- `ref/zigzag_experiment_execution_manual_v08.md` 中与 Copy 数据、长度、readout、位置编码和评测有关的条款；
- `reports/v08_dense_copy_1epoch_tuning_plan.md`；
- `reports/v08_dense_copy_1epoch_tuning_report.md`；
- 现有 `configs/probes_v08*.json` 中的 Copy record。

环境、远端、GPU、日志和 Git 规则仍遵守 `ref/experiment_environment_and_version_control.md`。若本文件与该环境规范冲突，以环境规范为准，除非用户在当前对话中明确修改。

本轮只产出规格。任何代码、数据、配置、图 artifact 或实验结果的修改，必须在后续实现轮次逐项对照本文档并通过验收门槛后进行。本文档及后续 corrected Copy 修改只提交到专用分支，不提交到 `main`。

## 0.1 Git 分支隔离与生命周期

### 0.1.1 分支定位

corrected Copy 是从主线切出的封闭实验分支，不是 v09 的前置重构。冻结约定：

```text
base_branch = main
branch_point_commit = 64845eb22b149fa5496dacecdfcdb610fbdc1cbb
experiment_branch = codex/copy-corrected-v01
merge_back_to_main = false
main_next_version = v09
```

如果真正建分支前 `main` 已前进，必须在 branch manifest 中记录实际 branch point；不得继续保留上面的旧 hash 却从另一个 commit 开始。第一次正式实验开始后冻结 branch point，不在实验中途 rebase、merge `main` 或更换基础代码。

### 0.1.2 默认采用独立 Git worktree

推荐把分支放在独立 worktree，而不是在 `/Users/sxye/Documents/expander` 中反复切换：

```text
main worktree:
/Users/sxye/Documents/expander

corrected Copy worktree:
/Users/sxye/Documents/expander-copy-corrected-v01
```

这样分支内的 `datasets/copy/`、outputs、logs、临时 checkpoint 和未跟踪大文件不会残留在 v09 主线工作目录。进入 corrected Copy 实现阶段后，所有编辑、测试、数据 materialization、远端 rsync 和结果回收都必须从 corrected Copy worktree 执行；主 worktree 保持 `main`，可以继续准备 v09。

建议的建分支语义如下；命令只作为后续执行参考，本轮不执行：

```bash
cd /Users/sxye/Documents/expander
git branch codex/copy-corrected-v01 64845eb22b149fa5496dacecdfcdb610fbdc1cbb
git worktree add /Users/sxye/Documents/expander-copy-corrected-v01 codex/copy-corrected-v01
```

本文档当前尚未提交。建立 worktree 后，必须把本文档放到 corrected Copy worktree 并作为该分支的第一批 commit；不得先把它提交到 `main` 再声称主线未受影响。

### 0.1.3 分支中必须提交的内容

corrected Copy 分支至少保存并提交：

- 本更正规格及后续阶段报告；
- 实际实现代码和自动化测试；
- config、manifest、field contract；
- 数据 materialization 脚本、README、dataset card、source lock、checksums；
- graph 生成配置、selected graph、certificate 和 reachability；
- 正式命令、轻量日志、metrics、summary、results 和诊断；
- 环境快照；
- `branch_manifest.json`。

原始 JSONL 约 130 MiB（旧 train 约 118 MiB，旧 validation 约 12 MiB），属于大型原始数据，不应仅为了“分支完整”而强塞进 Git。分支必须提交可重复 materialize 它们的脚本、来源 hash 和生成后 hash；实际 JSONL 位于 corrected worktree 的 `datasets/copy/`，按环境规范作为未跟踪/忽略的大文件处理。

### 0.1.4 branch manifest

分支根产物必须记录：

```text
branch_name
branch_point_commit
branch_head_commit
base_branch
merge_back_to_main
worktree_path
remote_run_root
dataset_materialization_script_sha256
train/test content sha256
final_report_path
final_status
created_at
completed_at
```

每次正式 run 还必须记录 `branch_name` 和 `branch_head_commit`。远端 rsync 不带 `.git` 时，使用部署 commit marker，不能让结果继续写 `git_commit=unknown`。

### 0.1.5 完成与回归 main

corrected Copy 完成后的正确动作是“封存分支并回到 main”，不是在 main 上执行反向 patch，也不是 `git reset --hard`：

1. corrected Copy 分支所有代码、文档和轻量产物已 commit；
2. 工作区无尚未归档的必要改动；
3. 记录 final branch head，并创建不可歧义的 final tag，例如 `copy-corrected-v01-final`；
4. 大型 JSONL/checkpoint 已按 hash 归档，或确认可由脚本重建；
5. main 不 merge、不 squash、不 cherry-pick corrected Copy 的实现 commit；
6. 返回 `/Users/sxye/Documents/expander` 的 main worktree，确认 HEAD 和预期 main commit 一致；
7. 运行主线状态审计，确认 corrected Copy 的脚本、config、`datasets/copy/`、outputs 和 logs 没有作为未跟踪残留混进 main worktree；
8. v09 从 main 的独立 commit 开始。

如果未来需要让 v09 引用 corrected Copy 的结论，只允许在 main 新写一份简短的“外部分支结果索引”，记录 branch/tag/final report/结果 hash。不得为了引用结论而把整条实验实现分支 merge 回 main。

### 0.1.6 单 worktree 的备用规则

只有无法使用 `git worktree` 时，才允许在当前目录直接切分支。此时必须先保护用户已有未跟踪文件；完成后用普通 `git switch main` 回归，不使用破坏性 reset。由于被 `.gitignore` 忽略的大型 `datasets/copy/*.jsonl` 不会随切分支自动消失，回 main 前必须先按 sha256 归档，并显式检查和清理 corrected Copy 残留。未完成该检查时不得开始 v09。

---

## 1. 最终冻结的纠正结论

新的 Copy 实验必须同时满足以下四项；缺少任意一项都不得启动正式训练：

1. **位置编码改为不可学习的 RoPE。** 删除 Copy 模型中的 learned absolute position embedding；RoPE 只作用于每层 attention 的 Q、K，不作用于 V，不作为可训练参数。
2. **训练和测试使用同一长度、同一格式。** 新数据直接放在仓库根目录的 `datasets/copy/`。保留原 train 数据；彻底弃用原 OOD test；把原 validation 的样本内容转换为新 test。
3. **直接在 marker positions 上预测。** 模型输入始终是 1024 个 source token 加 1024 个 marker token，总长度固定为 2048。target 只作为 label，不追加到输入，不在额外 PAD/readout slots 上预测。
4. **旧 Copy 结果全部失去结论效力。** 所有使用 T=6144、`readout_start=4096`、learned absolute position embedding、原 4096 长度 test 或旧有放回“epoch”采样的 Copy 结果，不得与 corrected 结果混合，不得继续调参后冒充修复结果。

更正后的任务语义不再是 `length extrapolation`。它是固定 `M=1024`、固定输入长度 2048、train/test 同分布的 Copy 学习与 attention-structure 对比。报告、run id、variant 和目录名不得继续声称测试了 2× length extrapolation。

---

## 2. 已确认的现有实现错误与风险

### 2.1 核心语义错误

| 编号 | 现状 | 后果 | 纠正要求 |
|---|---|---|---|
| C01 | `_task_lengths()` 对 sequence loss 使用 `max_input + max_target`，得到 raw T=6144 | 凭空创建外部 readout 区域，改变 S4 Copy 任务接口 | Copy raw T 固定为 2048 |
| C02 | `make_probe_batch()` 使用固定 `readout_start=4096` | train/validation 的预测位置是不存在于原输入中的 PAD slots | target positions 必须为 1024–2047，即原 marker positions |
| C03 | train/validation 被补到 6144，包含 3072 个纯无用 PAD 和 1024 个 PAD readout slots | dense attention 计算量膨胀，稀疏图路由被改变，大量无意义 token 参与 attention | corrected Copy 不允许任何 tensor padding，shape 必须为 `[B, 2048]` |
| C04 | `ProbeTransformer` 使用 `nn.Embedding(seq_len, d_model)` 的 learned absolute position embedding | 每个位置映射趋向被独立记忆；缺少 Copy 所需的相对位移归纳偏置 | 使用不可学习 RoPE，删除 Copy learned position table |
| C05 | 旧 test 为输入 4096、target 2048，而 train 为 2048/1024 | 训练与测试同时改变输入长度、目标长度、图规模和绝对位置，无法只判断是否学会 Copy | 丢弃旧 test；新 test 来自原 validation，保持 2048/1024 |
| C06 | dataset card 声称仅做 JSONL 转换，但 runtime 改成外部 PAD readout | 数据卡的 fidelity 声明与实际模型接口不一致 | 新 dataset card 必须明确 marker readout，并记录与旧 v08 的差异 |

### 2.2 训练公平性与可复现性错误

| 编号 | 现状 | 后果 | 纠正要求 |
|---|---|---|---|
| R01 | runner 没有在建模前调用完整随机种子初始化 | 同一个 `seed` 不能复现模型初始化和 dropout | Python、NumPy、Torch CPU/CUDA 全部显式设 seed；记录 deterministic policy |
| R02 | 数据采样 stream 包含 method：`f"{task}:{method}:{profile}"` | 不同 attention method 看到不同样本序列 | sampler permutation 只能依赖 data seed 和 epoch，不能依赖 method |
| R03 | 当前“1 epoch”是 10,000 次有放回 draw | 实测只有 6,333 个 unique train examples，3,667 个完全没见过 | epoch 必须是对 10,000 行的一次无放回全遍历 |
| R04 | `run_dir = output_root/task/method`，目录不含 seed | 多 seed 会覆盖或错误跳过 | 路径必须包含 `seed{seed}`，run id 也必须包含 seed |
| R05 | 发现已有 `summary.json` 就跳过，未核对 config/data/code hash | 旧配置或旧数据结果可能被当成当前结果 | 仅当完整 run identity hash 一致且产物审计通过时才允许 resume/skip |
| R06 | `config_sha256` 和 manifest path 在结果中硬编码为旧主配置 | dense tuning 等结果记录了错误的 config provenance | 必须记录命令实际传入的 config 和 manifest 路径及 sha256 |
| R07 | config 中 `eval_every` 没有控制评测；实际每个 log step 都评测 | 配置和行为不一致，浪费大量时间，也容易反复窥视 test | `log_every` 与 `eval_every` 分离；`copy_corrected_v01` 禁止训练中读 test |
| R08 | 没有 tensor checkpoint，仅写空 manifest | 长实验中断无法从 optimizer/sampler/RNG 状态恢复 | 正式 run 必须保存可恢复 checkpoint，或在运行前明确获得用户豁免 |

### 2.3 数据编码、模型头和指标错误

| 编号 | 现状 | 后果 | 纠正要求 |
|---|---|---|---|
| M01 | Copy 使用 `integer_shift`，把原 token 全部加 1，vocab 变成 65 | 偏离上游 A=64 的输入/输出空间；EOS/UNK 与本任务无关 | Copy 保留原始 ID，`vocab_size=64`、`token_output_size=64` |
| M02 | target 实际只出现 1–62，但旧报告用 `1/65` 描述随机基线 | 随机/边际基线解释错误 | 同时报告 uniform-64、有效 target-62 和 empirical marginal baseline |
| M03 | Copy 实例仍创建未使用的 `class_head` | 参数量包含不参与 loss 的参数，模型规格不干净 | corrected Copy 模型只创建需要的 token head |
| M04 | `copy_eos_accuracy` 被列为 secondary metric，但 Copy 没有 EOS label | 字段名制造并不存在的评测 | 删除该 metric，或固定为 `not_applicable` 并写明“数据无 EOS” |
| M05 | evaluate 把一个 batch 的平均 loss 重复赋给每个 sample，再二次加权 | 对可变目标长度任务会产生错误总体 loss | 聚合原始 loss sum 和有效 token count；不得从 batch mean 反推 sample loss |
| M06 | 通用聚合器用 token 数给 exact/sequence accuracy 加权 | 可变 target length 时 sequence accuracy 不是按样本平均 | token metric 按 token 聚合；sequence/exact metric 按 example 聚合 |
| M07 | `train_loss_final` 只是最后一个 optimizer step 的 micro-batch 平均 | 不能代表完整 epoch 训练损失 | 分开记录 last-step loss、epoch mean loss 和移动平均 |

M05、M06 对固定 target length=1024 的新 Copy 数值影响较小，但它们仍属于实验代码缺陷，必须修复并测试，不能因为当前数据等长而继续保留。

### 2.4 attention mask 与图诊断缺陷

| 编号 | 现状 | 后果 | 纠正要求 |
|---|---|---|---|
| A01 | `pad_mask` 只用于 classification pooling，没有进入 self-attention | 旧 6144 实现让纯 PAD 作为可见 key/value | corrected Copy 不含 padding；batch audit 必须断言 2048 个位置全部有效 |
| A02 | 旧图按 T=6144、q=96 生成 | 与 corrected T=2048 完全不是同一图 | 所有 Copy graph artifacts 必须重建为 T=2048、B=64、q=32 |
| A03 | 结果中 target hop 字段长期为 N/A | 未证明 marker query 能在给定层数内到达对应 source | 正式训练前必须生成 exact source-to-marker reachability 诊断 |
| A04 | local 方法的 block 永不跨 block | marker 与对应 source 相隔 1024，local 在任意层数都不可达 | local 只能作为结构性负对照；不得把 local 失败解释为训练代码失败 |

使用当前图生成器、T=2048、B=64、d=8、graph seed=0 的预审计结果如下。这里的“可达”方向是 **marker query row 到对应 source key column**：

| method | 1 hop | 2 hops | 3 hops | 结论 |
|---|---:|---:|---:|---|
| local | 0% | 0% | 0% | 永久不可达，预期失败 |
| zigzag_certified | 2.7344% | 82.9102% | 100% | 至少 3 层后结构上全覆盖 |
| random_regular（按 zigzag unique K 对齐） | 3.3203% | 99.5117% | 100% | 至少 3 层后结构上全覆盖 |
| dense | 100% | 100% | 100% | 1 层全覆盖 |

这些百分比是预审计，不可直接复制到正式结果。实现后必须针对最终 selected graph、最终 seed 和最终层数重新计算并保存 artifact。

---

## 3. 新数据合同

### 3.1 唯一允许的数据目录

新数据的唯一 canonical path 为：

```text
datasets/copy/
```

目录不得再嵌套 `copy/s4_copying_length_extrapolation/copy_s4_...` 等多层版本前缀。建议至少包含：

```text
datasets/copy/
├── README.md
├── dataset_card.json
├── source.lock
├── config.yaml
├── checksums.sha256
├── train.jsonl
└── test.jsonl
```

`copy_corrected_v01` **不创建 `validation.jsonl`**。不能把新 test 再复制一份命名为 validation；那会造成完全重复的 split 泄漏。

### 3.2 split 来源

| 新 split | 唯一来源 | 原始 sha256 | 行数 | 使用规则 |
|---|---|---|---:|---|
| train | 旧 `train.jsonl` | `a5a4aa651a5bdec25075930d1f59b7d0358e29dcca0fdd8f8dc897d55ee3de1c` | 10,000 | 用于训练 |
| test | 旧 `validation.jsonl` | `e5f48fc67dd3b4c41c39a224b3a01c5f01de7023425fe951875c029c04d82abd` | 1,000 | 只在配置冻结后的最终评测读取 |
| discarded | 旧 `test.jsonl` | `50de40e9b6f7c53af8a912cf0967ae1129e84028bcc7f90c14a94620d0760fac` | 1,000 | 不复制、不读取、不作为 fallback |

旧数据来源目录仅用于一次性、可审计的 materialization：

```text
/Users/sxye/Documents/expander_bench/data/probes/copy/
  s4_copying_length_extrapolation/
  copy_s4_l0_m1024_a64_full_v2/
```

materialization 后，runner 不得依赖该外部目录。若 `datasets/copy/` 缺失或 checksum 不匹配，程序必须 fail closed，不能自动退回旧 probe 路径。

### 3.3 允许的数据转换

`input` 和 `target` 数组必须逐元素保持不变。为了防止新 test 仍显示为 validation，允许且要求只做以下元数据归一化：

1. 新 test 的 `id` 中 split 名从 `.validation.` 改为 `.test.`；
2. train/test 的 `variant` 改为不含 `length_extrapolation` 的新名称，例如 `s4_copying_iid_m1024_marker_readout`；
3. dataset-level provenance 记录原文件路径、原 sha256、转换脚本、转换时间和转换后 sha256；
4. 不更改 `input`、`target`、`l_memorize`、`l_noise`、`n_tokens`、`seed`、`input_length`、`target_length`。

必须额外生成忽略 `id/variant` 后的 content digest，证明新 train/test 的 `(input, target, numeric metadata)` 分别与旧 train/validation 一一对应。

### 3.4 每行数据的硬不变量

train 和 test 的每一行都必须满足：

```text
len(input) == 2048
len(target) == 1024
input[0:1024] == target
input[1024:2048] == [63] * 1024
metadata.input_length == 2048
metadata.target_length == 1024
metadata.l_memorize == 1024
metadata.l_noise == 0
metadata.n_tokens == 64
target values are all in [1, 62]
input values are all in [1, 63]
```

split-level 还必须满足：

- train 有 10,000 行且 id 唯一；
- test 有 1,000 行且 id 唯一；
- train/test id 不相交；
- train/test 的 1024-token target 序列不重复、不交叉；
- 两个 split 的 input/target 长度分布完全一致；
- `checksums.sha256` 覆盖目录内所有受控文件；
- 旧 OOD test 的 sha256 不得出现在新 config 的 train/test 字段中。

### 3.5 test 使用纪律

因为 `copy_corrected_v01` 没有 validation split：

- test 不得用于 early stopping；
- test 不得在每个 log/eval interval 运行；
- test 不得用于选择 learning rate、模型尺寸或训练步数；
- test 不得出现在 tiny-overfit gate 中；
- 同一冻结 config/method/seed 的 final test 原则上只运行一次；失败重跑必须记录原因。

训练中允许记录 train loss、train token accuracy 和固定 train diagnostic batch 的指标，但必须明确标为 `train_diagnostic`，不能写成 validation。

如果后续确实需要调参 validation，必须另建新的数据版本，从旧 train 的 10,000 行中按固定索引切出 validation，并把这些行从 train 删除；不得使用或复制 test。该动作需要新的文档和用户确认，不属于 `copy_corrected_v01`。

---

## 4. token、vocab 与模型输出合同

### 4.1 禁止 integer shift

Copy 直接使用上游 S4 的原始 token ID：

| ID | 含义 |
|---:|---|
| 0 | 上游 noise/zero token；当前 L=0 数据中不出现，也不代表本 batch 的 tensor padding |
| 1–62 | 可复制的 source/target value |
| 63 | marker token |

冻结值：

```text
encoder_type = identity_integer
vocab_size = 64
token_output_size = 64
marker_token_id = 63
target_value_min = 1
target_value_max = 62
eos_token = not_applicable
unk_token = not_applicable
tensor_padding = none
```

不得继续使用 `integer_shift`，不得把 marker 从 63 改成 64，不得创建 vocab size 65。也不得为了缩小 head 而把 target 1–62 另行重映射为 0–61；上游 S4 `Copying.d_output` 为 `n_tokens=64`，corrected 实验保持该接口。

### 4.2 task-specific head

Copy 模型只需要：

```text
token_embedding: 64 -> d_model
token_head: d_model -> 64
```

不得实例化不参与 Copy loss 的 `class_head`。参数量统计只包含实际 forward/loss 路径中的参数。

---

## 5. marker readout 与 loss 合同

### 5.1 正确的数据流

对每个样本：

```text
model tokens:
[source_0, source_1, ..., source_1023,
 marker_0, marker_1, ..., marker_1023]

token positions:
[0, 1, ..., 1023, 1024, 1025, ..., 2047]

labels（不送入模型）:
[target_0, target_1, ..., target_1023]

loss positions:
[1024, 1025, ..., 2047]
```

其中 `marker_i` 的 token ID 全部为 63，`target_i == source_i`。模型在 position `1024+i` 的 hidden state 上预测 `target_i`。

target 不能追加到 tokens 后面，不能覆盖 marker，不能构造 external query/PAD slots，也不能做 teacher-forced shifted-token LM。

### 5.2 batch 张量

正确 shape：

```text
tokens.shape           == [batch, 2048]
targets.shape          == [batch, 1024]
target_positions.shape == [batch, 1024]
target_mask.shape      == [batch, 1024]
valid_token_mask.shape == [batch, 2048]
```

并满足：

```text
target_positions[b] == arange(1024, 2048)
target_mask.all() == true
valid_token_mask.all() == true
tokens[b, :1024] == targets[b]
tokens[b, 1024:] == 63
```

`resolved_readout_start` 这类容易重新引入固定空白区的通用字段，在 corrected Copy 中应删除或明确设置为 1024，并由逐样本公式验证。推荐直接由 `input_length - target_length` 推导 marker start，同时断言结果为 1024，避免手写 4096 再次混入。

### 5.3 loss

唯一主 loss：

```text
selected_logits = token_logits[:, 1024:2048, :]
loss_sum = cross_entropy(selected_logits, targets, reduction="sum")
loss = loss_sum / 1_024_target_tokens_per_example / batch_size
```

允许用 gather，但 gather positions 必须等于 1024–2047。直接 slice 更简单，优先使用 slice，减少位置张量错配的机会。

loss 审计必须证明：

- 修改 position 0–1023 的 output logits 不直接改变当前 batch loss；
- 修改 position 1024–2047 的 output logits会改变 loss；
- 每个样本恰好有 1024 个监督 token；
- 没有 PAD ignore-index；
- 没有 EOS loss；
- target 不作为额外输入 token 出现。

---

## 6. RoPE 位置编码合同

### 6.1 冻结配置

```text
position_encoding = rope
rope_learnable = false
rope_theta = 10000.0
rope_scaling = none
rope_dim = head_dim
apply_to = q_and_k_only
position_ids = 0..2047
absolute_position_embedding = none
```

`head_dim = d_model / num_heads` 必须为偶数；不满足时在模型初始化阶段直接报错，不允许静默只旋转一部分奇数维。

### 6.2 参考实现

仓库相邻、已锁定 upstream 中可参考的主动实现位于：

```text
/Users/sxye/Documents/expander_bench/.cache/upstreams/zoology/
  zoology/mixers/ttt.py
```

参考其中：

- `rotate_half`；
- `apply_rotary_pos_emb`；
- `RotaryEmbedding`；
- `inv_freq` 用 `register_buffer(..., persistent=False)` 注册；
- trig 计算强制 float32，再 cast 回 Q/K dtype。

`/Users/sxye/Documents/bishe/n_exp/n_exp3_new/model.py` 中也有 `_rope_cache()` 和 `_apply_rope()` 的数学形状示例，但该文件当前真正调用 RoPE 的三行是注释状态。它不能作为“已启用 RoPE”的证据，不能原样照搬后声称完成。

### 6.3 放置位置

RoPE 必须在每一层 attention 内、Q/K 完成线性投影并 reshape 为 `[B, H, T, head_dim]` 后应用：

```text
h -> qkv projection -> split q,k,v -> reshape/transpose
  -> apply RoPE to q,k
  -> dense / neighbor / split / blockpair attention
  -> output projection
```

它必须位于 attention backend 分支之前，以保证 dense、local、zigzag、random 使用完全相同的 Q/K 旋转。不得只给 dense 加 RoPE，也不得在 token embedding 上相加一个所谓“RoPE embedding”。

### 6.4 禁止项

- 禁止 `nn.Embedding(seq_len, d_model)` 作为 Copy 位置编码；
- 禁止 learned RoPE frequency；
- 禁止把 `rope_theta` 定义为 `nn.Parameter`；
- 禁止对 V 旋转；
- 禁止给不同 method 使用不同 RoPE；
- 禁止在某些 backend 中绕过 RoPE；
- 禁止以“RoPE 已存在于参考文件但调用被注释”为通过依据；
- 禁止保留 learned absolute embedding 再叠加 RoPE，除非另立 ablation，不得进入 corrected main。

### 6.5 RoPE 为什么适合本任务

对应的 source/marker 位置满足：

```text
marker_position_i - source_position_i = (1024 + i) - i = 1024
```

所有 1024 个复制对共享同一个相对位移。RoPE 让 Q/K 内积自然依赖相对位置差，而不是要求模型分别学习 `P_1024 -> P_0`、`P_1025 -> P_1` 等 1024 个独立绝对位置映射。这是本次使用 RoPE 的核心归纳偏置理由。

---

## 7. 序列长度、padding 与 attention 合同

### 7.1 冻结长度

```text
source_length = 1024
marker_length = 1024
target_length = 1024
raw_sequence_length = 2048
padded_sequence_length = 2048
tensor_padding_tokens = 0
graph_block_size = 64
graph_num_blocks = 32
```

2048 可被 64 整除，因此 graph/block backend 不需要补齐。任何出现 3072、4096、5120、6144 的 Copy runtime position 都应立即触发失败。

相比旧 dense mask：

```text
old: 6144^2 = 37,748,736 attention pairs
new: 2048^2 =  4,194,304 attention pairs
ratio: old/new = 9
```

### 7.2 padding mask

corrected Copy 数据没有 tensor padding，所有 2048 个位置都是实际任务 token。`valid_token_mask` 必须全 true。

通用 probe 框架中“pad_mask 不进入 attention”的问题不能被掩盖，但对 corrected Copy 的处理不是再造复杂 padding 逻辑，而是硬断言无 padding。未来其他可变长任务需要单独修复 per-example attention mask，不得把该改动未经测试地混入本次 Copy 修正。

### 7.3 non-causal

attention 仍为 directed non-causal：

```text
causal = false
is_causal = false
KV-cache incremental decoding = forbidden
teacher-forced next-token LM = forbidden
```

mask 矩阵语义必须在代码和诊断中统一为：

```text
mask[query_position, key_position] == true
```

source-to-marker reachability 的检查必须按 marker query 到 source key 的方向计算，不能把图边方向反过来。

---

## 8. graph 与 method 公平性合同

### 8.1 图必须重建

旧 T=6144 graph artifact 全部作废。新图至少冻结并记录：

```text
T = 2048
B = 64
q = 32
d = 8（若后续变更，必须重新生成并重新诊断）
graph_seed
graph_generation_algorithm
selected_graph_sha256
certificate_sha256
```

task manifest 中不得同时保存一份可能过期的 inline graph 和另一份 file graph 而不做等价校验。推荐 manifest 只保存 canonical path、sha256 和必要摘要；若保留 inline artifact，运行前必须逐字段/hash 证明它与 canonical file 完全相同。

### 8.2 必做 reachability artifact

每个 method、graph seed、最终 layer count 必须输出：

```text
target_in_1hop_rate
target_in_2hop_rate
...
target_in_Lhop_rate
average_shortest_path
unreachable_rate
per_target_shortest_path_histogram
direction_definition
```

验收门槛：

- dense：`target_in_1hop_rate == 1.0`；
- zigzag_certified：`target_in_Lhop_rate == 1.0`，否则不得训练；
- random_regular：`target_in_Lhop_rate == 1.0`，否则不得作为公平对照；
- local：允许且预期 `unreachable_rate == 1.0`，报告必须注明它是结构性负对照。

### 8.3 random 对齐

random_regular 继续按 zigzag 每 query 的 non-causal unique K 对齐。必须满足：

```text
random_k_alignment_error_max == 0
random_k_alignment_error_mean == 0
```

同时分别记录 unique K 和 multiplicity。zigzag 的 `unique_log_m` 与 random 的 multiplicity=1 不是同一个对象，报告不能只写 K 相同就声称所有 attention bias 完全相同。

### 8.4 同一模型与数据

同一 trial 中各 method 必须使用：

- 相同 model architecture；
- 相同初始 model state hash；
- 相同 train permutation；
- 相同 optimizer 和 scheduler；
- 相同 micro-batch、gradient accumulation 和 optimizer steps；
- 相同 RoPE 配置；
- 相同 train/test 文件 sha256；
- 相同 metric 代码；
- 独立且不覆盖的输出目录。

只有 attention structure/artifact 允许不同。

---

## 9. 训练 iterator、seed 与恢复合同

### 9.1 真正的 epoch

对 10,000 行 train：

1. 每个 epoch 用 `data_seed + epoch` 生成一个包含 0–9999 的确定性 permutation；
2. 每个 index 在该 epoch 恰好出现一次；
3. 不使用有放回 `randrange()`；
4. method 名不得进入 permutation seed；
5. effective batch=16 时，1 epoch 恰好 625 个 optimizer steps；
6. 记录 `draw_count=10000`、`unique_count=10000`、`never_seen=0`、`max_repeat_count=1`。

若使用 gradient accumulation，micro-batch 必须按 permutation 连续消费，不能每个 accumulation slot 独立随机采样。

### 9.2 seed 分离

至少分开记录：

```text
model_seed
data_seed
graph_seed
dropout_seed_or_policy
```

建模前必须设置：

```text
random.seed(model_seed)
numpy.random.seed(model_seed)
torch.manual_seed(model_seed)
torch.cuda.manual_seed_all(model_seed)
```

并记录 `torch.backends` / deterministic algorithm 设置。正式报告不得只写一个未真正控制模型初始化的 `seed`。

### 9.3 checkpoint/resume

正式 run 的 checkpoint 至少保存：

- model state；
- optimizer state；
- scheduler/当前 learning rate state；
- epoch、optimizer step、micro-step；
- sampler permutation 与当前位置，或可无歧义重建它的状态；
- Python/NumPy/Torch CPU/CUDA RNG states；
- config/data/graph/code identity hashes。

resume 后的下一 batch、learning rate 和未中断运行必须一致。checkpoint 文件不进 Git，但 checkpoint manifest、路径、sha256、step 和保留策略必须进入产物。

---

## 10. 评测与指标合同

### 10.1 主指标

```text
primary_metric = copy_token_accuracy
secondary_metric = copy_sequence_accuracy
```

定义：

```text
copy_token_accuracy = total_correct_target_tokens / total_target_tokens
copy_sequence_accuracy = exactly_correct_examples / total_examples
test_loss = total_cross_entropy_sum / total_target_tokens
```

1,000 行 test 应有：

```text
total_examples = 1000
total_target_tokens = 1,024,000
```

不得对 per-batch mean 简单平均，除非严格按有效 target token 数加权；推荐直接累计 loss sum、correct count、token count、exact count、example count。

### 10.2 baseline

每次正式 test 前必须从 train 统计并保存 baseline：

- uniform over 64 output classes：accuracy 期望 `1/64`，NLL `log(64)`；
- target support 实际为 62 个值：仅用于解释数据分布，不改变 64-class head；
- global mode token accuracy；
- empirical train-marginal NLL；
- position-wise mode token accuracy/NLL。

旧结果 loss 约 4.127、token accuracy 约 0.016 对应的是 62 类近均匀边际分布附近，而不是已经学会 Copy。新报告必须以实际统计值为准，不再用 `1/65`。

### 10.3 final test 隔离

runner 需要两个明确模式：

```text
train          # 不读取 test
final-eval     # 读取 frozen checkpoint 和 test，只评测
```

正式训练命令不应打开 `test.jsonl`。final-eval 产物记录 checkpoint sha256；这样才能证明测试没有参与模型选择。

---

## 11. 当前代码的逐文件修改范围（后续实现清单）

本节只定义影响范围，不在本轮修改代码。

| 文件/区域 | 必须修改或审计的内容 |
|---|---|
| `datasets/copy/` | materialize 新 train/test、metadata、source lock、checksums；不创建 validation |
| `scripts/probe_common.py` | Copy canonical path 改为 `datasets/copy`；Copy required files 改为 train/test；禁止旧路径 fallback；结果字段增加 RoPE/data identity |
| `scripts/probe_data_audit.py` | 支持 Copy 两 split；加入 2048/1024、source==target、marker==63、split provenance 和 discarded-test 禁用检查 |
| `scripts/probe_parameter_selection.py` | Copy T=2048、marker readout start=1024、vocab64、identity encoder、q32；重建 graph；不从跨 split max 推导外部 readout |
| `scripts/probe_phase2_dryrun.py` | dry-run 使用 marker positions，不使用 toy external readout；增加 RoPE enabled 证据 |
| `scripts/probe_tasks.py` | Copy identity encoder；batch 不补 6144；target positions=marker；Copy 模型删除 learned pos 和 unused class head |
| `scripts/synthetic_mvp_core/model.py` | 为 Probe attention 显式加入 nonlearnable RoPE；在 backend 分支前旋转 Q/K；避免无意改变历史 v07 runner |
| `scripts/synthetic_mvp_core/attention.py` | 验证 dense/neighbor/split/blockpair 在 RoPE 后保持同一 attention 语义；必要时仅加测试，不重复实现 RoPE |
| `scripts/synthetic_mvp_core/artifacts.py` | T2048 artifact；输出 source-to-marker reachability；验证方向和 random K alignment |
| `scripts/run_probe_experiment.py` | train/test-only Copy 流程、无放回 iterator、完整 seeds、run dir 含 seed、真实 config hashes、train/final-eval 分离、resume identity |
| `scripts/probe_metrics.py` | 用 count/sum 聚合 loss/token/exact；去除 Copy EOS 伪指标 |
| `configs/` | 新建 corrected Copy 独立 config/manifest；不得覆写旧 v08 manifest 后继续复用旧输出 |
| `outputs/`、`logs/`、`reports/` | 使用新版本根目录；旧结果保留并标记 invalid-for-corrected-copy |

推荐新实验版本名：

```text
copy_corrected_v01
```

该版本名只存在于 `codex/copy-corrected-v01` 分支。不得把同名 config/output 直接生成在 main worktree 后再靠人工删除恢复主线。

推荐输出结构：

```text
outputs/copy_corrected_v01/<trial_id>/<method>/seed<seed>/
logs/copy_corrected_v01_<trial_id>_<method>_seed<seed>_<timestamp>.log
outputs/copy_corrected_v01/branch_manifest.json
```

---

## 12. 必须新增的自动化测试

### 12.1 数据测试

1. train/test 行数、sha256、唯一 id；
2. 每行 2048/1024 长度；
3. source prefix 等于 target；
4. marker suffix 全为 63；
5. token 范围正确；
6. train/test 序列无交叉；
7. content digest 与旧 train/validation 对齐；
8. 旧 test sha/path 不在任何 corrected config 中；
9. `datasets/copy/validation.jsonl` 不存在；
10. runner 缺新数据时 fail closed，不 fallback。

### 12.2 batch/readout 测试

1. `tokens.shape == [B, 2048]`；
2. 没有额外 PAD positions；
3. `target_positions == 1024..2047`；
4. `targets == tokens[:, :1024]`；
5. `tokens[:, 1024:] == 63`；
6. 每样本 target mask sum=1024；
7. loss 只读 marker logits；
8. target 没有被 append 或写进 marker tokens；
9. train/test batch contract 完全一致。

### 12.3 RoPE 测试

1. model parameter names 中不存在 learned position embedding；
2. `inv_freq` 是 buffer，`requires_grad=false`；
3. head_dim 为奇数时初始化失败；
4. 与锁定参考实现在固定 Q/K 上数值一致；
5. Q/K 被旋转、V 未旋转；
6. 每一层都调用 RoPE；
7. dense/local/zigzag/random 都经过同一 RoPE 路径；
8. 同时平移 Q/K position IDs 时，RoPE dot product 保持只依赖相对位移；
9. float32 trig + bf16/fp16 cast 无 NaN/Inf；
10. state dict 不含 2048×d_model 的 learned pos table。

### 12.4 attention backend 一致性测试

dropout=0、相同权重、相同 mask 和输入时：

1. dense masked reference 与 neighbor 输出在容差内一致；
2. dense masked reference 与 split 输出在容差内一致；
3. dense masked reference 与 blockpair 输出在容差内一致；
4. multiplicity/log_m 在各 backend 中一致；
5. mask row/column 方向测试明确 query→key；
6. target reachability 的方向与 forward attention 相同。

### 12.5 sampler/reproducibility 测试

1. 一个 epoch 10,000 draws、10,000 unique、0 missing、最大重复1；
2. 相同 data seed 的各 method index 顺序完全一致；
3. 不同 data seed permutation 不同；
4. 相同 model seed 的初始 state hash 一致；
5. 同 seed 同 method 重跑首 N step loss 完全一致或在已声明容差内；
6. resume 与 uninterrupted run 的后续 batch/lr/loss 对齐；
7. 多 seed 输出目录互不覆盖；
8. stale summary 的 identity hash 不一致时不得 skip。

### 12.6 metric 测试

使用手工构造 logits 验证：

1. token accuracy 按 token count 聚合；
2. sequence accuracy 按 example count 聚合；
3. loss 是全体有效 target token 的总 NLL/总 token；
4. 不同 eval batch size 得到相同 loss 和 metrics；
5. final partial batch 不改变聚合口径；
6. Copy 没有 EOS metric；
7. primary metric 从 test count 计算，不从 per-sample rounded value平均。

---

## 13. 分阶段验收与停止规则

### Gate 0：静态合同

必须全部通过：

- 当前 branch/worktree 是 `codex/copy-corrected-v01`，不是 `main`；
- branch point、branch head、worktree path 已写入 branch manifest；
- 新数据目录和 checksum；
- T=2048，无 padding；
- marker readout positions 正确；
- vocab/output=64，无 integer shift；
- learned pos 参数不存在；
- RoPE 参数不可学习；
- graph T=2048/q=32；
- corrected config 不含旧 test hash和旧 readout 4096。

失败则停止，不运行 GPU。

### Gate 1：CPU 单元测试与 backend 对齐

运行本文件第 12 节全部相关测试。任何 backend 数值不一致、方向错误或 metric batch-size dependent 都停止。

### Gate 2：单 batch overfit

只用 train 中固定的少量样本，dense、dropout=0。目标是证明模型、RoPE、marker readout、loss 和 optimizer 能形成闭环。

最低门槛建议冻结为：

```text
train token accuracy >= 0.999
train sequence accuracy >= 0.99
train loss <= 0.01
```

该 gate 不读取 test。若失败，禁止扩大模型、加 epoch 或运行 sparse methods；先定位实现错误。

### Gate 3：full-length train diagnostic

使用完整 T=2048 和无放回 iterator，先跑 dense 的受控短训练，检查：

- loss 明显低于 empirical marginal baseline；
- token accuracy 持续高于 global-mode baseline；
- attention/QKV/token head 梯度非零且有限；
- 每个 epoch 覆盖全部 10,000 行；
- 训练吞吐和显存符合 T=2048，而不是暗中仍为 6144。

若仍停在边际基线，先检查模型容量、学习率和优化阈值，但不得重新引入 PAD readout、learned absolute positions、旧 OOD test 或有放回采样。

### Gate 4：sparse structure

只在 dense Gate 2/3 通过后运行 zigzag/random。训练前先通过 reachability 和 K-alignment gate。local 可并行作为负对照，但它失败是结构预期。

### Gate 5：冻结后 final test

只有 config、checkpoint selection rule 和 method 列表冻结后才运行一次新 test。final report 必须列出：

- 新 train/test sha256；
- checkpoint sha256；
- code commit；
- model/data/graph seeds；
- RoPE 配置与 position parameter count=0；
- target positions 1024–2047；
- T=2048、padding=0；
- reachability；
- baseline 和 corrected metrics；
- test 首次读取时间或 eval run id。

---

## 14. 禁止通过的“表面修复”

以下做法即使能让 accuracy 上升，也不得验收：

1. 只把 `resolved_padded_sequence_length` 从 6144 改成 2048，但 target positions 仍指向外部 slots；
2. 把 target token 写进 marker positions 再让模型复述，造成 label leakage；
3. 把 marker 改成 1024 个可学习 query embeddings；
4. 保留 learned absolute position embedding，再声称“也加了 RoPE”；
5. 只在 dense backend 应用 RoPE；
6. 复制旧 validation 为 test，同时训练中继续每隔若干 step 评测它；
7. 新数据缺失时 fallback 到旧 4096-length test；
8. 把 test 复制成 validation；
9. 继续使用有放回 draw，却把 10,000 draws 写成完整 epoch；
10. 不设置 model seed，只记录 sampler seed；
11. 多 method 使用不同训练样本顺序；
12. 复用 T=6144 graph artifact；
13. 用 local 的失败证明整个模型失败；
14. 只看 loss 从 4.3 降到约 4.127，就声称学会 Copy；
15. 复用旧 summary/outputs 而不校验 identity hash；
16. 把参考文件中被注释的 RoPE 代码当作已启用证据；
17. 修改数据 arrays 以降低任务难度；
18. 用 test 指标挑 learning rate 或模型大小后，再把同一 test 当最终无偏结果。

---

## 15. 旧产物的解释与归档

以下旧值是错误实现的识别标志：

```text
resolved_padded_sequence_length = 6144
resolved_runtime_input_length = 4096
resolved_runtime_target_length = 2048
resolved_readout_start = 4096
target positions = 4096..5119 (train/validation)
target positions = 4096..6143 (old test)
position embedding = learned absolute nn.Embedding
vocab size = 65
train sampling = with replacement
```

匹配任一核心标志的旧 Copy run：

- 可保留用于说明失败原因；
- 不得进入 corrected results.csv；
- 不得与 corrected trial 画在同一主比较图中而不显著标注 invalid implementation；
- 不得用于选择 corrected 超参数；
- 不得被新 runner 因 `summary.json status=ok` 自动跳过并复用。

建议在旧 Copy 报告顶部追加显著声明，而不是删除历史文件：

```text
INVALID FOR CORRECTED COPY CONCLUSIONS:
this run used external PAD readout positions, learned absolute positions,
T=6144 and/or the discarded OOD test split.
```

---

## 16. 最终验收清单

只有以下项目全部为 true，才可以宣布“Copy 实现已纠正”并进入正式实验：

```text
[ ] current worktree is the corrected Copy worktree, not main
[ ] branch is codex/copy-corrected-v01 and branch point is recorded
[ ] merge_back_to_main is false
[ ] branch_manifest.json records branch head and final status
[ ] canonical data path is datasets/copy/
[ ] train source is old train, 10,000 rows
[ ] test source is old validation, 1,000 rows
[ ] old 4096/2048 test is absent and blocked
[ ] no validation split is duplicated from test
[ ] train/test are both input 2048, target 1024
[ ] input prefix equals target; suffix is marker 63
[ ] identity integer encoding; vocab/output size is 64
[ ] tokens tensor is exactly [B, 2048]
[ ] there are zero tensor padding positions
[ ] loss is read from marker positions 1024..2047
[ ] target is not appended, injected or teacher-forced
[ ] learned absolute position embedding is absent
[ ] nonlearnable RoPE rotates Q/K in every layer/backend
[ ] RoPE theta=10000, no scaling, head_dim even
[ ] graph is regenerated for T=2048, B=64, q=32
[ ] zigzag/random target_in_Lhop_rate is 1.0
[ ] local is documented as unreachable negative control
[ ] sampler is without replacement and method-independent
[ ] model/data/graph seeds are actually applied and recorded
[ ] run directory includes trial, method and seed
[ ] actual config/manifest/data/graph/code hashes are recorded
[ ] stale outputs cannot be silently reused
[ ] test is not read during training or tuning
[ ] loss/token/sequence metrics use correct count aggregation
[ ] single-batch overfit gate passes before full runs
[ ] checkpoint/resume policy is tested or explicitly waived
[ ] old v08 Copy results are marked invalid for corrected conclusions
[ ] py_compile, unit tests, smoke and artifact audits all pass
[ ] implementation and resulting artifacts are committed separately
[ ] final branch commit/tag is recorded before returning to main
[ ] raw JSONL/checkpoints are archived or reproducible by committed scripts
[ ] main worktree has no corrected Copy untracked residue
[ ] v09 starts from main without merging corrected Copy implementation commits
```

任何一项失败，都必须停止并回到对应 gate；不得通过增加模型、延长训练或换 GPU 绕过。

---

## 17. 本文档所依据的代码与数据证据

本轮已逐段检查：

- `scripts/probe_common.py`；
- `scripts/probe_data_audit.py`；
- `scripts/probe_parameter_selection.py`；
- `scripts/probe_phase2_dryrun.py`；
- `scripts/probe_tasks.py`；
- `scripts/probe_metrics.py`；
- `scripts/run_probe_experiment.py`；
- `scripts/graph_structures.py`；
- `scripts/graph_diagnostics.py`；
- `scripts/synthetic_mvp_core/model.py`；
- `scripts/synthetic_mvp_core/attention.py`；
- `scripts/synthetic_mvp_core/artifacts.py`；
- 当前 Copy configs、tuning plan、tuning report 和 v08 manual；
- 上游 S4 `copying.py`、`synthetic.py`、`SequenceDecoder`；
- 旧 train/validation/test 的全量行级结构不变量与 sha256；
- T=2048 图的 source-to-marker 多跳可达性；
- 相邻 upstream 中的 active RoPE 实现。

上游 S4 的关键接口是：A=64，输入末尾 M 个位置为 marker，`l_output=M`，默认 `SequenceDecoder(mode="last")` 直接读取输入序列最后 M 个 hidden states。也就是说，marker readout 本身已经避免 target leakage；额外 PAD readout slots 从来不是完成该任务所必需的。
