# v08 Phase 2 Code Adaptation 报告

## 结论

Phase 2 已新增 probe 专用审计、参数、任务、指标、远端 readiness 和训练入口，并用 tiny interface dry-run 验证 6 个 task 的 JSONL 读取、编码、non-causal attention、forward、loss 和 backward 路径。dry-run 参数只用于接口验证，不进入 Phase 4 主参数。

| task | 状态 | dry-run loss | batch shape |
|---|---|---:|---|
| copy | ok | 4.3217 | [2, 80] |
| selective_copy | ok | 3.7917 | [2, 80] |
| induction_associative_recall | ok | 9.2343 | [2, 64] |
| niah_kv_retrieval | ok | 5.8187 | [2, 144] |
| ruler | ok | 5.9088 | [2, 144] |
| lra_listops | ok | 2.4314 | [2, 128] |

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| attention_contract | 注意力合同 | non_causal | v08 手册 | 确认 dry-run 不走 causal LM | 不满足则失败 |
| causal | 是否 causal mask | false | v08 手册 | 防止任务格式错误 | 不满足则失败 |
| graph_directionality | 图方向性 | directed | graph artifact | 验证 directed graph 路径 | 无 |
| input_batch_shape | 输入 batch 张量形状 | [batch, T] | dry-run | 验证不同 schema 可进入模型 | 无 |
| target_positions_shape | 目标 readout/position 形状 | [batch, target_len] 或 classification | dry-run | 验证三类 loss | classification 写 classification |
| loss | dry-run 训练损失 | 标量 | forward/backward | 验证 loss 有限 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 6 |
| unexplained_parameters | [] |
| english_only_sections | [] |
