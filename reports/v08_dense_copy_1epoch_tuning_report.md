# v08 dense-copy 1 epoch 调参实验报告

## 1. 结论

本轮按 `reports/v08_dense_copy_1epoch_tuning_plan.md` 先完成两个 sanity trial：`dcopy_1ep_M_lr1em4` 和 `dcopy_1ep_L_lr1em4`。两者均使用 `batch_size=2`、`gradient_accumulation_steps=8`，跑满 625 optimizer steps 和 10,000 个 train-example draws，无 OOM、无 NaN，恒定学习率 `1e-4` 生效。

两个 trial 均未达到最低学习信号标准：M/L 的全量 validation `copy_token_accuracy` 分别为 `0.0162168` 和 `0.0160723`，接近随机基线 `1/65=0.0153846`，且远低于方案要求的 `0.03`。全量 test token accuracy 也分别只有 `0.0160815` 和 `0.0160190`，sequence accuracy 均为 0。

因此触发方案中的 sanity stop rule，后续 M/L 其他学习率、S grid 和 optional XL 均未启动。本轮没有可推荐的 dense-copy 模型尺寸或学习率；不能把 loss 从约 4.35/4.32 降到 4.128 解释为已经学会 copy。

## 2. 实验环境与固定合同

- 本地 base commit：`cd501bc`；远端按规范 rsync 不包含 `.git`，运行结果中的 `git_commit` 为 `unknown`，config/result sha256 仍完整保存。
- 远端主机：`hhpc`（SSH alias `huiwei`）。
- Conda 环境：`ysx_base`。
- Python：`3.10.0`。
- PyTorch：`2.10.0+cu128`。
- GPU：`NVIDIA A100-SXM4-80GB`。
- task / method：`copy` / `dense`。
- attention：`non_causal`、`causal=false`、`dense_mask`。
- train budget：625 optimizer steps，effective batch 16，10,000 train-example draws。
- micro-batch / accumulation：2 / 8。
- eval batch：2。
- scheduler：真实 `const` 分支，learning rate `1e-4`。
- checkpoint：`manifest_only_no_tensor_checkpoint`。

启动前 GPU 3 和 GPU 2 均为 0% utilization、10 MiB 占用；M 绑定物理 GPU 3，L 绑定物理 GPU 2。两个 run 使用独立 tmux session、独立 output root 和独立 log。

## 3. Sanity trial 结果

| trial | model | 参数量 | train loss | validation loss / token acc | test loss / token acc | sequence acc | peak allocated / reserved | wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dcopy_1ep_M_lr1em4` | 6L, d384, h8, ffn1536 | 13,081,858 | 4.127928 | 4.128277 / 0.016217 | 4.128365 / 0.016082 | 0 | 22.295 / 23.422 GiB | 3483.05 s |
| `dcopy_1ep_L_lr1em4` | 8L, d512, h8, ffn2048 | 28,465,794 | 4.127847 | 4.128654 / 0.016072 | 4.128701 / 0.016019 | 0 | 28.350 / 30.217 GiB | 4932.85 s |

M 的首个 log 为 train loss `4.350762`、validation loss `4.250243`、token accuracy `0.017609`；第 625 step 的小 validation log 为 `4.128350` / `0.014893`。L 对应从 `4.319885` / `4.235443` / `0.015808` 到 `4.127847` / `4.128916` / `0.017761`。loss 早期快速下降后进入平台，但 token accuracy 始终在随机波动区间；扩大到 L 没有带来验证指标改善。

成功判据审计：

- OOM / NaN：通过。
- `train_loss_final < 4.0`：M/L 均未通过。
- `copy_token_accuracy > 0.03`：M/L 均未通过。
- 强信号 `train_loss_final < 3.5`、`copy_token_accuracy > 0.05`：均未通过。

## 4. 停止规则与未运行 trial

由于两个 sanity trial 都没有有效学习信号，按计划停止完整 grid。以下 trial 未启动，因此没有结果或失败产物：

```text
dcopy_1ep_M_lr3em4
dcopy_1ep_M_lr3em5
dcopy_1ep_L_lr3em4
dcopy_1ep_L_lr3em5
dcopy_1ep_S_lr3em4
dcopy_1ep_S_lr1em4
dcopy_1ep_S_lr3em5
```

本轮已运行 trial 均成功结束，没有 OOM/error log。上一轮 batch 16 的 OOM 产物不属于本轮结果。

## 5. 失败分流审计

结构化审计保存于 `outputs/probes_v08_dense_copy_1epoch/sanity_stop_audit.json`。

### 5.1 数据、target positions 与 loss mask

抽查每个 split 的首尾样本并通过当前 `make_probe_batch()` 路径解析：

| split | rows | input / target length | 有效 target positions | 结论 |
|---|---:|---:|---:|---|
| train | 10,000 | 2048 / 1024 | 4096–5119 | 连续，mask 数量 1024，target 等于 JSON value + 1 |
| validation | 1,000 | 2048 / 1024 | 4096–5119 | 连续，mask 数量 1024，target 等于 JSON value + 1 |
| test | 1,000 | 4096 / 2048 | 4096–6143 | 连续，mask 数量 2048，target 等于 JSON value + 1 |

loss 只在 `target_mask=true` 的 readout positions 上计算；`integer_shift` encoder 与 vocab size 65 对齐。未发现 input/target、loss mask 或 readout position 的错位。

### 5.2 Dense mask 与 non-causal contract

实际 dense mask 形状为 `[6144, 6144]`，37,748,736 个元素全部为 true，矩阵对称，backend 为 `dense_mask`，符合 `non_causal` / `causal=false` 合同。position embedding 覆盖完整 6144 positions。micro-batch 2 后显存峰值降到 22.3/28.4 GiB，上一轮 OOM 已解除。

### 5.3 训练采样覆盖

runner 的训练采样是 deterministic random sampling with replacement。复现本轮 seed、stream、625 steps、8 次 accumulation、micro-batch 2 后：

- draws：10,000；
- unique train examples：6,333 / 10,000（63.33%）；
- never seen：3,667；
- seen exactly once：3,707；
- max repeat count：6。

这不属于“反复训练极少数样本”，但也不是无放回地完整遍历一个 dataset epoch。因此本报告使用 `1 train-equivalent epoch`，不把它描述成严格覆盖全数据的一轮。该采样策略是后续实验必须控制的方法学风险。

### 5.4 解释边界

数据位置、target mask、dense mask 和 non-causal contract 均未发现实现错误。当前证据只支持：M/L 在该 readout 设计、完整长度、1 train-equivalent epoch 和 `1e-4` 下没有学到 copy 映射。loss 接近 `4.128` 而 accuracy 保持随机，更像是模型学到了边际 token 分布，而非位置对应关系；这是基于曲线的推断，不是已证明的根因。

## 6. 产物

| trial | log | output root | training curve |
|---|---|---|---|
| M | `logs/v08_dense_copy_1epoch_dcopy_1ep_M_lr1em4_20260618_124029.log` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_M_lr1em4` | `copy/dense/training_curves.png`，1080×960，15,823 bytes |
| L | `logs/v08_dense_copy_1epoch_dcopy_1ep_L_lr1em4_20260618_124029.log` | `outputs/probes_v08_dense_copy_1epoch/dcopy_1ep_L_lr1em4` | `copy/dense/training_curves.png`，1080×960，14,513 bytes |

两个 run 均保存了 `command.sh`、raw/resolved config snapshot、phase4 task record、summary、metrics、results、result-field audit、attention diagnostics、budget manifests 和 training curve。两份 result-field audit 均为 `passed`。

## 7. 最终推荐

本轮最终推荐：**不从当前 grid 选择 dense-copy 参数，也不继续盲跑剩余学习率或更大模型。**

下一轮应先建立新的受控诊断文档，优先验证两个问题：小样本是否能被模型完全过拟合，以及训练 iterator 改为无放回完整 epoch 后是否仍然停在随机水平。若 tiny-overfit 仍失败，再检查 readout query 表示与 position-to-position copy 机制；这些都属于新实验变量，不回填到本轮结果。

## 8. 报告审计

```text
report_language = zh
experiment_scope = copy + dense only
train_budget = 1 train-equivalent epoch
method_count = 1
task_count = 1
completed_trials = 2
grid_status = stopped_after_two_sanity_trials_without_learning_signal
primary_tuned_parameters = model size, learning rate
fixed_parameters = data split, sequence length, encoder, loss, effective batch, optimizer family
unexplained_parameters = []
final_recommendation = none; run a separately documented tiny-overfit and no-replacement audit next
```
