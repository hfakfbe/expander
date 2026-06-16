# v08 Phase 6 Per-Task 报告

## 总览

本报告按 task 汇总 Phase 6 main eval。所有 task 均使用同一个 task 内冻结配置比较 `local`、`zigzag_certified`、`random_regular`，并使用 seed 0。每个 task 的 `random_regular` 均逐 query 对齐 `zigzag_certified` 的 non-causal K，`random_k_alignment_error_max=0`。

## copy

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.016165 | 4.127336 | 6250 | 50000 | 2.60 |
| zigzag_certified | 0.016072 | 4.127310 | 6250 | 50000 | 18.60 |
| random_regular | 0.016214 | 4.127317 | 6250 | 50000 | 18.62 |

copy 的主指标 `copy_token_accuracy` 接近随机水平。5 个 train-equivalent epochs 后三种方法差异很小，不能声称任何方法学起 copy。

## selective_copy

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.070500 | 2.641094 | 6250 | 50000 | 2.26 |
| zigzag_certified | 0.068500 | 2.640435 | 6250 | 50000 | 14.97 |
| random_regular | 0.068875 | 2.640640 | 6250 | 50000 | 15.01 |

selective_copy 的主指标 `selective_copy_token_accuracy` 仍低，三种方法在同一预算下接近。该任务需要继续检查优化和读出设置。

## induction_associative_recall

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.000000 | 8.334947 | 56250 | 900000 | 3.34 |
| zigzag_certified | 0.000000 | 9.691010 | 56250 | 900000 | 10.55 |
| random_regular | 0.000000 | 8.822923 | 56250 | 900000 | 10.54 |

induction_associative_recall 的主指标 `retrieval_exact_match` 仍为 0。虽然训练量已从一轮扩展到 5 个 train-equivalent epochs，但 exact match 未改善到可用水平。

## niah_kv_retrieval

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.122000 | 2.196365 | 6250 | 50000 | 1.21 |
| zigzag_certified | 0.116000 | 2.200174 | 6250 | 50000 | 7.92 |
| random_regular | 0.104000 | 2.198175 | 6250 | 50000 | 7.93 |

niah_kv_retrieval 有低水平 retrieval signal。local 在 seed 0 下略高于 zigzag_certified 和 random_regular，但本轮只有单 seed，不能做统计显著结论。

## ruler

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.000000 | 3.664145 | 7500 | 60000 | 1.21 |
| zigzag_certified | 0.000000 | 2.832849 | 7500 | 60000 | 7.92 |
| random_regular | 0.000000 | 2.893758 | 7500 | 60000 | 7.93 |

ruler 的主指标 `retrieval_exact_match` 为 0。zigzag_certified 和 random_regular 的 test loss 低于 local，但 exact match 没有起色，报告中只记作 loss 改善，不记作任务成功。

## lra_listops

| method | primary_metric_value | test_loss | train_steps | train_examples_seen | peak_reserved_gb |
|---|---:|---:|---:|---:|---:|
| local | 0.368000 | 1.599512 | 60000 | 480000 | 2.50 |
| zigzag_certified | 0.383000 | 1.581749 | 60000 | 480000 | 18.23 |
| random_regular | 0.366500 | 1.596271 | 60000 | 480000 | 18.24 |

lra_listops 是本轮最清楚的学习信号。zigzag_certified 在 seed 0 下取得最高 `listops_accuracy=0.383`，但仍需多 seed 复核稳定性。

## 跨任务审计

| gate | 结果 |
|---|---|
| train_epochs | main 全部为 5 |
| attention_contract | 全部为 `non_causal` |
| causal | 全部为 `false` |
| graph_directionality | 全部为 `directed` |
| random_k_alignment_error_max | 全部为 0 |
| training_curves.png | 18 张均为 1080 x 960 PNG |
| result_field_audit | 18 个 run 全部 passed |
| old_bad_outputs | 一轮训练和 1 x 1 曲线图旧产物已归档到 `ref/archive_v08_one_epoch_bad_curves/` |

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| task | probe 任务名 | 6 个任务之一 | Phase 4 manifest | 按任务拆分结论 | 无 |
| method | attention 方法 | `local`、`zigzag_certified`、`random_regular` | Phase 4 manifest | 比较同任务不同 attention 方法 | 无 |
| primary_metric_value | 主指标数值 | 0 到 1 或任务定义范围 | test eval | 任务成功与否的主判断 | 无 |
| copy_token_accuracy | copy token 准确率 | 0 到 1 | copy metric | copy 的主指标 | 其他 task 写 `not_applicable` |
| selective_copy_token_accuracy | selective_copy token 准确率 | 0 到 1 | selective_copy metric | selective_copy 的主指标 | 其他 task 写 `not_applicable` |
| retrieval_exact_match | 检索 exact match | 0 到 1 | retrieval metric | induction、niah、ruler 的主指标 | 非检索任务写 `not_applicable` |
| listops_accuracy | ListOps 分类准确率 | 0 到 1 | listops metric | lra_listops 的主指标 | 其他 task 写 `not_applicable` |
| test_loss | test split loss | loss 值 | eval 过程 | 辅助解释主指标 | 无 |
| train_steps | 训练更新步数 | step 数 | Phase 4 resolved value | 验证训练预算 | 无 |
| train_examples_seen | 实际训练样本数 | example 数 | 运行时统计 | 验证 5 个 train-equivalent epochs | 无 |
| train_epochs | 训练等效 epoch | 本轮为 5 | Phase 4 resolved value | 防止把一轮 sweep 当成 full train | 无 |
| peak_reserved_gb | GPU 峰值 reserved 显存 | GB | PyTorch CUDA 统计 | 解释吞吐和资源使用 | 无 |
| attention_contract | attention 合约 | `non_causal` | config 与 diagnostics | 确认非 causal 设定 | 无 |
| causal | 是否 causal mask | `false` | attention diagnostics | 与合约互相校验 | 无 |
| graph_directionality | 图方向性 | `directed` | graph artifact | 确认证书图为有向图 | 无 |
| random_k_alignment_error_max | random 与 zigzag 每 query K 最大差 | K 的绝对差 | random budget | 验证 random_regular 公平对齐 | 无 |
| training_curves.png | 训练曲线图 | PNG 文件 | metrics 脚本 | 检查训练轨迹存在且非空白 | 无 |
| result_field_audit | 结果字段审计 | passed/failed | audit 脚本 | 防止缺字段、空字段和坏曲线图 | 无 |
| seed | 随机种子 | `0` | main config | 复现实验 | 多 seed 未执行时写为限制 |
| train-equivalent epochs | 训练集等效覆盖量 | epoch 等效数 | Phase 4 预算 | 说明预算不是单轮 sweep | 若固定 step smoke 则不适用 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 20 |
| unexplained_parameters | [] |
| english_only_sections | [] |
