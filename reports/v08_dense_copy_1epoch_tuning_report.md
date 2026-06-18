# v08 dense-copy 1 epoch 调参实验报告

## 1. 结论

本轮严格按 `reports/v08_dense_copy_1epoch_tuning_plan.md` 先执行两个 sanity trial。`dcopy_1ep_M_lr1em4` 与 `dcopy_1ep_L_lr1em4` 均在第 1 个训练 step 的首次前向传播中 OOM，没有产生任何有效训练指标。按计划中的停止规则，后续主 grid 和可选 trial 均未启动。

当前固定实验合同在现有 fp32 dense attention 实现上不可运行，因此本轮没有可推荐的 dense-copy 模型尺寸或学习率。不能根据这两次失败比较 M/L 学习能力，也不能报告 train/validation/test 指标。

OOM 的直接原因可精确复现：`batch=16`、`heads=8`、`padded_sequence_length=6144` 时，每层 attention score 张量形状为 `[16, 8, 6144, 6144]`，单个 fp32 张量大小恰好为 `18.0 GiB`。该数值与两次异常中的“尝试分配 18.00 GiB”完全一致。

## 2. 实验环境与执行约束

- 代码 commit：`f146357`
- 远端主机：`hhpc`
- Conda 环境：`ysx_base`
- Python：`3.10.0`
- PyTorch：`2.10.0+cu128`
- GPU：`NVIDIA A100-SXM4-80GB`
- task / method：`copy` / `dense`
- 计划训练预算：625 steps，1 train-equivalent epoch
- 固定有效 batch：16（batch 16，gradient accumulation 1）
- scheduler：真实 `const` 分支；warmup 字段仅记录，不参与恒定学习率调度
- checkpoint：仅 manifest，不写 tensor checkpoint

正式运行前已完成本地编译、JSON 解析、rsync、远端环境检查和 GPU 利用率检查。M 使用物理 GPU 3，L 使用物理 GPU 2；两张卡启动时利用率均为 0%，显存占用均为 10 MiB。

## 3. Sanity trial 结果

| trial | model | learning rate | 参数量 | batch | 计划/完成 steps | GPU | wall time | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `dcopy_1ep_M_lr1em4` | M: 6L, d384, h8, ffn1536 | `1e-4` | 13,081,858 | 16 | 625 / 0 | 3 | 14 s | OOM |
| `dcopy_1ep_L_lr1em4` | L: 8L, d512, h8, ffn2048 | `1e-4` | 28,465,794 | 16 | 625 / 0 | 2 | 14 s | OOM |

M 的 OOM 快照为 PyTorch allocated `59.73 GiB`、reserved but unallocated `13.89 GiB`、进程总占用 `74.10 GiB`，随后还需要分配 `18.00 GiB`。L 对应为 allocated `61.61 GiB`、reserved but unallocated `12.81 GiB`、进程总占用 `74.90 GiB`，随后还需要分配 `18.00 GiB`。

由于两个 trial 都在首次前向传播失败，下列字段均为不可用：

- `final train/validation/test loss`：N/A
- `copy_token_accuracy`：N/A
- `copy_sequence_accuracy`：N/A
- `peak_allocated_gb` / `peak_reserved_gb`：未生成结构化结果；上表后的数值仅为 OOM 异常快照
- `training_curves.png`：未生成，尺寸 N/A

正式 trial 产物：

| trial | log_path | output_root | error_log |
|---|---|---|---|
| M | `logs/v08_dense_copy_1epoch_dcopy_1ep_M_lr1em4_20260618_122430.log` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_M_lr1em4` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_M_lr1em4/copy/dense/error.log` |
| L | `logs/v08_dense_copy_1epoch_dcopy_1ep_L_lr1em4_20260618_122432.log` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_L_lr1em4` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_L_lr1em4/copy/dense/error.log` |

每个正式 output root 均保留了 `command.sh`、config snapshots、attention diagnostics、budget manifests 和 `error.log`，未写 checkpoint/tensor 文件。

## 4. 失败审计

### 4.1 数据与 readout

对完整 split 和首样本 batch 的结构审计结果如下：

| split | 样本数 | input 长度 | target 长度 | 有效 target positions |
|---|---:|---:|---:|---|
| train | 10,000 | 2048 | 1024 | 4096–5119，连续 |
| validation | 1,000 | 2048 | 1024 | 4096–5119，连续 |
| test | 1,000 | 4096 | 2048 | 4096–6143，连续 |

`target_mask` 仅覆盖真实 target；train/validation 的 1024 个有效 target 与数据 metadata 一致，test 的 2048 个有效 target 覆盖完整预留区间。position embedding 长度覆盖 `T=6144`。未发现 target positions、loss mask 或 readout 错位。

### 4.2 Dense mask 与显存

dense mask 形状为 `[6144, 6144]`，全部为 true、对称且 non-causal。当前实现显式执行：

```text
scores = q @ k.transpose(-1, -2)
```

因此 score 内存下界为：

```text
16 * 8 * 6144 * 6144 * 4 bytes = 18.0 GiB
```

训练阶段还需要 softmax、每层激活、反向图和 FFN 激活。M/L 在单卡 80GB 上均无法完成一次前向传播。模型参数量不是主要内存项；只减小 d_model 或学习率不能消除这个由 batch、head 数、序列长度和 fp32 决定的 18 GiB score 张量。

## 5. 无效环境尝试说明

第一次 tmux 包装错误地落入 base Python 3.13，未使用指定的 `ysx_base`，因此对应运行不计入实验结果。失败现场没有覆盖，已移动到：

- `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_M_lr1em4_invalid_base_env_20260618_1221`
- `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_L_lr1em4_invalid_base_env_20260618_1221`
- `logs/v08_dense_copy_1epoch_dcopy_1ep_M_lr1em4_20260618_122156.log`
- `logs/v08_dense_copy_1epoch_dcopy_1ep_L_lr1em4_20260618_122159.log`

修正后通过独立 tmux 探针确认 Python、Torch 及其安装路径，再重新执行正式 sanity trial。

## 6. 推荐与下一步

本轮最终推荐：**无可用 dense-copy 参数组合**。在维持 `batch=16`、`T=6144`、fp32 和当前显式 dense score 实现时，不应继续运行剩余学习率或更大模型。

下一轮必须建立新的 trial id 和实验文档，不能复用本轮失败目录。优先建议保持 effective batch 16，但把 micro-batch 降为 1 并用 gradient accumulation 16；同时单独评估 bf16/AMP、activation checkpoint，以及 PyTorch scaled-dot-product attention。任何一项都属于新的受控变量，不能回填到本轮结果中。

## 7. 报告审计

```text
report_language = zh
experiment_scope = copy + dense only
train_budget = 1 train-equivalent epoch planned; 0 steps completed due OOM
method_count = 1
task_count = 1
primary_tuned_parameters = model size, learning rate
fixed_parameters = data split, sequence length, encoder, loss, effective batch, optimizer family
unexplained_parameters = []
grid_status = stopped_after_two_sanity_trials
final_recommendation = none_under_current_memory_contract
```
