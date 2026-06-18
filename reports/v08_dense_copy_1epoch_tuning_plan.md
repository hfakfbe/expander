# v08 dense-copy 1 epoch 调参方案

## 0. 实验定位

本文档只针对 `copy` task 和 `dense` method，目标是先把 dense 模型训练信号跑起来。该实验是 capacity / learning-rate diagnostic，不是完整 v08 主实验，也不替代 multi-method comparison。

约束来自 `ref/experiment_environment_and_version_control.md`：

- 本地唯一编辑目录：`/Users/sxye/Documents/expander`
- 远端运行目录：`/home/huiwei/ysx/zigzag_attention`
- 远端环境：`conda activate ysx_base`
- GPU 默认：4 x NVIDIA A100-SXM4-80GB
- 正式运行前必须检查 `git status --short`、编译脚本、rsync 到远端、检查 GPU、用 tmux/screen 运行并写 log
- 每个正式 run 必须保留 `command.sh`、config snapshot、summary 或 error log
- 不得覆盖已有有效输出；失败必须保留错误现场

本轮只训练 1 个 train-equivalent epoch，用来回答两个问题：

1. 当前 copy 训不起来是否主要因为模型容量不足。
2. 在 dense attention 下，学习率和模型尺寸的可行区间在哪里。

## 1. 固定实验条件

| 参数 | resolved value | 说明 |
|---|---:|---|
| task | `copy` | 只跑 copy |
| method | `dense` | 只跑 dense |
| required_methods | `["dense"]` | 本诊断实验中 dense 是唯一 required method |
| optional_methods | `[]` | 不跑 local / zigzag / random |
| attention_contract | `non_causal` | 沿用 v08 copy 的 non-causal contract |
| attention_backend | `dense_mask` | dense 会解析到 full dense mask |
| graph_directionality | `not_applicable_for_dense` | dense 不使用稀疏图比较 |
| dataset_source | `https://github.com/state-spaces/s4` | 沿用 v08 copy 数据 |
| dataset_revision_or_hash | `e757cef57d89e448c413de7325ed5601aceaac13` | 沿用 v08 copy 数据版本 |
| train_examples | `10000` | 沿用 v08 copy train split |
| validation_examples | `1000` | 沿用 v08 copy validation split |
| test_examples | `1000` | 沿用 v08 copy test split |
| sequence_length_max | `4096` | v08 copy 最大输入长度 |
| runtime_target_length | `2048` | copy 目标长度 |
| padded_sequence_length | `6144` | 当前实现用于 position embedding / dense mask 的长度 |
| vocab_size | `65` | integer-shift encoder |
| encoder_or_tokenizer | `integer_shift` | 沿用 v08 copy encoder |
| loss_type | `copy_sequence_cross_entropy` | 沿用 copy loss |
| primary_metric | `copy_token_accuracy` | 先看 token accuracy 是否脱离随机 |
| secondary_metrics | `copy_sequence_accuracy, copy_eos_accuracy` | sequence accuracy 1 epoch 后可以仍为 0 |

当前已知问题背景：上一轮 copy 在 5 个 train-equivalent epochs 后仍接近随机，`copy_token_accuracy≈0.016`，接近 `1/65≈0.0154`，loss 接近 `log(65)≈4.17`。因此本轮优先扩大模型容量，同时保留完整长度，不先裁短任务。

## 2. 训练预算

只训 1 个 train-equivalent epoch：

```text
effective_batch_size = batch_size * gradient_accumulation_steps = 2 * 8 = 16
strict_one_epoch_steps = ceil(train_examples / effective_batch_size) = ceil(10000 / 16) = 625
manual_budget_steps = 625
train_examples_seen_planned = 625 * 16 = 10000
train_equivalent_epochs_actual = 10000 / 10000 = 1.0
```

统一训练参数：

| 参数 | resolved value | 说明 |
|---|---:|---|
| train_budget_unit | `steps` | 与现有 runner 对齐 |
| train_budget_value | `625` | 严格 1 epoch |
| steps_planned_if_step_budget | `625` | logging gate 使用 |
| train_equivalent_epochs | `1` | 严格 1 epoch |
| batch_size | `2` | dense 6144 长度下使用小 micro-batch 控制显存 |
| gradient_accumulation_steps | `8` | 累积 8 次以保持 effective batch 不变 |
| effective_batch_size | `16` | `2 * 8` |
| eval_batch_size | `2` | 与 dense micro-batch 对齐，降低评测显存峰值 |
| optimizer | `adamw` | 沿用 v08 |
| lr_scheduler | `const` | 目标语义为恒定学习率；注意当前 runner 还需要代码支持，见下方实现审计 |
| warmup_ratio | `0.1` | 若 runner 未修改，实际仍会按 warmup+cosine 使用该值 |
| warmup_steps | `62` | `round(625 * 0.1)`；仅在当前 runner 的 warmup+cosine 实现下生效 |
| min_lr_ratio | `0.1` | 沿用 v08 |
| weight_decay | `0.01` | 先不调 weight decay |
| grad_clip_norm | `1.0` | 保留梯度裁剪 |
| dropout | `0.0` | 先降低学习噪声，目标是确认能否学起来 |
| log_every | `5` |  |
| log_eval_examples | `32` | 每次 log 只做小 validation eval，控制 dense 成本 |
| eval_every | `25` | 记录在 config；当前 runner 主要由 log_every 触发中途 eval |
| validation_eval_budget | `1000` | final validation 全量 |
| test_eval_budget | `1000` | final test 全量 |
| checkpoint_policy | `manifest_only_no_tensor_checkpoint` | 不写 tensor checkpoint |
| checkpoint_every | `0` | 避免大文件进入 git |

如果某个 trial OOM，不允许在同一 run 里静默改参数继续写同一目录。必须保留失败输出，并新建 trial id 后再调整。

实现审计备注：截至当前代码，`scripts/run_probe_experiment.py::schedule_lr()` 不读取 `resolved_lr_scheduler` 分支，训练循环会无条件执行 warmup + cosine。也就是说，如果只在 manifest 中写 `resolved_lr_scheduler="const"`，结果字段会显示 const，但实际学习率不是恒定值。要真正执行 const，需要修改训练脚本的 scheduler 分支，或把本文档中的 `lr_scheduler` 改回当前实现实际使用的 `cosine`。

## 3. 模型尺寸网格

参数量按当前 `scripts/probe_tasks.py::ProbeTransformer` 结构估算，`padded_sequence_length=6144`，`vocab_size=65`。

| model_id | layers | d_model | heads | head_dim | ffn_dim | dropout | 估算参数量 | 角色 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B0` | 4 | 128 | 4 | 32 | 512 | 0.0 | 1.60M | 可选 anchor；接近旧模型 |
| `S` | 4 | 256 | 8 | 32 | 1024 | 0.0 | 4.78M | 小容量扩展 |
| `M` | 6 | 384 | 8 | 48 | 1536 | 0.0 | 13.08M | 主力搜索下界 |
| `L` | 8 | 512 | 8 | 64 | 2048 | 0.0 | 28.47M | 推荐主力 |
| `XL` | 10 | 768 | 12 | 64 | 3072 | 0.0 | 75.75M | 可选升级；只在 L 不 OOM 后跑 |

不建议第一轮直接使用 100M+ 模型。原因是 dense full copy 的 `padded_sequence_length=6144`，attention 内存按 `L^2` 增长；当前训练脚本没有明确的 AMP 或 activation checkpoint 保护。第一轮应先建立“有学习信号”的容量区间。

## 4. 学习率网格

主学习率集合：

```text
[3e-4, 1e-4, 3e-5]
```

解释：

- `3e-4`：沿用旧 v08 学习率，判断放大模型后是否立刻可学。
- `1e-4`：较稳健的中等学习率，是 `M/L` 的首跑值。
- `3e-5`：较大模型或 loss 抖动时的保守学习率。

`XL` 只跑较稳的学习率：

```text
[1e-4, 3e-5]
```

## 5. Trial 矩阵与运行顺序

优先先跑两个 sanity trial：

| 优先级 | trial_id | model_id | learning_rate | 目的 |
|---:|---|---|---:|---|
| 1 | `dcopy_1ep_M_lr1em4` | `M` | `1e-4` | 中等容量是否有学习信号 |
| 2 | `dcopy_1ep_L_lr1em4` | `L` | `1e-4` | 推荐容量是否可跑且更快脱离随机 |

如果两个 sanity trial 都完全无学习信号，应暂停完整 grid，先审计数据、target positions、loss mask、dense mask 和 readout 逻辑。不要继续盲目加大模型。

主 grid：

| trial_id | model_id | layers | d_model | heads | ffn_dim | learning_rate | steps | effective_batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dcopy_1ep_S_lr3em4` | `S` | 4 | 256 | 8 | 1024 | `3e-4` | 625 | 16 |
| `dcopy_1ep_S_lr1em4` | `S` | 4 | 256 | 8 | 1024 | `1e-4` | 625 | 16 |
| `dcopy_1ep_S_lr3em5` | `S` | 4 | 256 | 8 | 1024 | `3e-5` | 625 | 16 |
| `dcopy_1ep_M_lr3em4` | `M` | 6 | 384 | 8 | 1536 | `3e-4` | 625 | 16 |
| `dcopy_1ep_M_lr1em4` | `M` | 6 | 384 | 8 | 1536 | `1e-4` | 625 | 16 |
| `dcopy_1ep_M_lr3em5` | `M` | 6 | 384 | 8 | 1536 | `3e-5` | 625 | 16 |
| `dcopy_1ep_L_lr3em4` | `L` | 8 | 512 | 8 | 2048 | `3e-4` | 625 | 16 |
| `dcopy_1ep_L_lr1em4` | `L` | 8 | 512 | 8 | 2048 | `1e-4` | 625 | 16 |
| `dcopy_1ep_L_lr3em5` | `L` | 8 | 512 | 8 | 2048 | `3e-5` | 625 | 16 |

可选 trial：

| trial_id | model_id | layers | d_model | heads | ffn_dim | learning_rate | 运行条件 |
|---|---|---:|---:|---:|---:|---:|---|
| `dcopy_1ep_B0_lr3em4` | `B0` | 4 | 128 | 4 | 512 | `3e-4` | 只作为旧模型 anchor |
| `dcopy_1ep_XL_lr1em4` | `XL` | 10 | 768 | 12 | 3072 | `1e-4` | `L` 显存和速度可接受后再跑 |
| `dcopy_1ep_XL_lr3em5` | `XL` | 10 | 768 | 12 | 3072 | `3e-5` | `XL_lr1em4` 不发散但偏慢时跑 |

建议运行顺序：

```text
M_lr1em4 -> L_lr1em4 -> M_lr3em4 -> M_lr3em5 -> L_lr3em4 -> L_lr3em5 -> S grid -> optional XL
```

## 6. Config / manifest 组织方式

当前 `scripts/run_probe_experiment.py` 的输出目录为：

```text
run_dir = output_root / task / method
```

因此同一个 `output_root` 下不能同时放多个 dense-copy trial，否则会发生目录碰撞，且已完成的 `summary.json` 会导致后续 trial 被跳过。

每个 trial 必须使用独立 config、独立 manifest、独立 output_root：

```text
configs/probes_v08_dense_copy_1epoch/<trial_id>.json
configs/probes_v08_dense_copy_1epoch/<trial_id>_task_parameters.json
outputs/probes_v08_dense_copy_1epoch/<trial_id>/copy/dense/
logs/v08_dense_copy_1epoch_<trial_id>_<timestamp>.log
```

每个 trial config 模板：

```json
{
  "version": "v08_dense_copy_1epoch",
  "phase": "phase_dense_copy_1epoch_tuning",
  "profile": "main",
  "tasks": ["copy"],
  "methods": ["dense"],
  "seeds": [0],
  "task_parameter_manifest": "configs/probes_v08_dense_copy_1epoch/<trial_id>_task_parameters.json",
  "output_root": "outputs/probes_v08_dense_copy_1epoch/<trial_id>"
}
```

每个 trial manifest 从 `configs/probes_v08_task_parameters.json` 的 copy record 派生，但必须更新以下字段：

```text
top-level required_methods = ["dense"]
top-level optional_methods = []
copy.resolved_required_methods = ["dense"]
copy.resolved_optional_methods = []
copy.resolved_attention_backend = "dense_mask"
copy.resolved_layers = <grid value>
copy.resolved_d_model = <grid value>
copy.resolved_heads = <grid value>
copy.resolved_ffn_dim = <grid value>
copy.resolved_dropout = 0.0
copy.resolved_parameter_count = <estimated or runtime-confirmed value>
copy.resolved_batch_size = 2
copy.resolved_gradient_accumulation_steps = 8
copy.resolved_effective_batch_size = 16
copy.resolved_eval_batch_size = 2
copy.resolved_train_budget_unit = "steps"
copy.resolved_train_budget_value = 625
copy.resolved_steps_planned_if_step_budget = 625
copy.resolved_train_equivalent_epochs = 1
copy.resolved_train_examples_seen_planned = 10000
copy.resolved_base_learning_rate = <grid value>
copy.resolved_learning_rate = <grid value>
copy.resolved_lr_scheduler = "const"
copy.resolved_warmup_ratio = 0.1
copy.resolved_warmup_steps = 62
copy.resolved_min_lr_ratio = 0.1
copy.resolved_min_learning_rate = <learning_rate * 0.1>
copy.resolved_log_every = 5
copy.resolved_eval_every = 25
copy.resolved_validation_eval_budget = 1000
copy.resolved_test_eval_budget = 1000
copy.main.steps = 625
copy.main.log_every = 5
copy.main.eval_every = 25
copy.main.train_examples = 10000
copy.main.validation_examples = 1000
copy.main.test_examples = 1000
copy.main.log_eval_examples = 32
copy.main.train_equivalent_epochs = 1
copy.main.train_examples_seen_planned = 10000
```

Manifest 的 `selection_reason` 必须说明这是 dense-copy 1 epoch capacity diagnostic，且只改变模型容量和学习率，不改变数据长度、训练样本数、encoder、loss 和 primary metric。

## 7. 远端运行流程

本地生成 config / manifest 后：

```bash
git status --short
python -m py_compile scripts/*.py
```

同步到远端：

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'cached_graphs/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  ./ huiwei:/home/huiwei/ysx/zigzag_attention/
```

远端检查：

```bash
ssh huiwei
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

GPU 选择仍按 `GPU3 -> GPU2 -> GPU1 -> GPU0`，目标 GPU utilization 低于 20% 才启动。

正式运行模板：

```bash
tmux new -s v08_dense_copy_1epoch
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base
mkdir -p logs

TRIAL=dcopy_1ep_M_lr1em4
GPU_ID=3
LOG=logs/v08_dense_copy_1epoch_${TRIAL}_$(date +%Y%m%d_%H%M%S).log

PROBE_V08_LOG_PATH="$LOG" CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/run_probe_experiment.py \
  --config configs/probes_v08_dense_copy_1epoch/${TRIAL}.json \
  --task copy \
  --method dense \
  --seed 0 \
  2>&1 | tee "$LOG"
```

多个 trial 可并行跑在不同 GPU 上，但每个训练进程只绑定一张 GPU。并行时每个 trial 使用独立 tmux window、独立 `CUDA_VISIBLE_DEVICES`、独立 log。

## 8. 成功判据与选参规则

1 epoch 后不要只看 sequence accuracy。copy sequence accuracy 在短训练下可能仍为 0，第一轮重点看 loss 和 token accuracy 是否脱离随机。

随机基线参考：

```text
random token accuracy ≈ 1 / 65 = 0.0154
random cross entropy ≈ log(65) = 4.174
```

单个 trial 通过最低标准：

- 无 OOM
- 无 NaN / non-finite loss
- `train_loss_final < 4.0`
- `validation_loss_final` 相比初始 log 有下降趋势
- `copy_token_accuracy > 0.03`，或训练曲线显示持续上升

强信号标准：

- `train_loss_final < 3.5`
- `copy_token_accuracy > 0.05`
- validation loss 和 train loss 同向下降
- grad norm 没有长期爆炸

候选选择顺序：

1. 先按 validation `copy_token_accuracy` 排序。
2. 若 accuracy 接近，选 validation loss 更低者。
3. 若指标接近，选更小模型。
4. 若大模型 only train loss 好、validation 不动，先怀疑过拟合或长度泛化问题，不直接继续加大模型。

## 9. 失败分流

若所有 `M/L` trial 的 loss 都贴近 `4.17`，不要继续扩大模型，先检查：

- copy 数据中 input / target 是否和当前 loss positions 对齐
- `target_positions` 是否覆盖正确的 2048 个 copy 位置
- `target_mask` 是否只在 copy target 上计算 loss
- dense mask 是否确实全连接且 non-causal
- position embedding 长度是否覆盖 `padded_sequence_length=6144`
- 训练采样是否真的覆盖 train split，而不是重复极少数样本

若 `S/M` 有学习信号而 `L/XL` OOM：

- 保留 OOM trial 的 error log
- 不覆盖原输出目录
- 下一轮再考虑 AMP、activation checkpoint 或降低 `d_model/layers`

若 train loss 下降但 validation/test 不动：

- 保留 best train trial
- 检查 train/validation/test 的长度分布差异
- 下一轮考虑 curriculum 或 length-bucket tuning，但本轮不改数据长度

## 10. 结果同步、报告和 git

远端结束后同步轻量产物回本地：

```bash
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/outputs/probes_v08_dense_copy_1epoch/ outputs/probes_v08_dense_copy_1epoch/
rsync -av huiwei:/home/huiwei/ysx/zigzag_attention/logs/v08_dense_copy_1epoch_*.log logs/
```

本地必须检查：

```bash
git status --short
python -m py_compile scripts/*.py
```

调参完成后新增报告：

```text
reports/v08_dense_copy_1epoch_tuning_report.md
```

报告至少包含：

- 每个 trial 的 model_id、learning_rate、参数量、batch、steps、GPU、log_path、output_root
- final train/validation/test loss
- copy_token_accuracy、copy_sequence_accuracy
- peak_allocated_gb、peak_reserved_gb、total_wall_time_sec
- training_curves.png 尺寸和路径
- OOM/失败 trial 的 error log 路径和失败原因
- 最终推荐的 dense-copy 参数

按照环境规范，config、manifest、报告、logs、summary、metrics、results、training_curves.png、command.sh、config snapshots 等轻量产物应进入 git；checkpoint/tensor 大文件不进入 git。

## 11. 报告审计

```text
report_language = zh
experiment_scope = copy + dense only
train_budget = 1 train-equivalent epoch
method_count = 1
task_count = 1
primary_tuned_parameters = model size, learning rate
fixed_parameters = data split, sequence length, encoder, loss, effective batch, optimizer family
unexplained_parameters = []
```
