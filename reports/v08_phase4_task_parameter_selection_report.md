# v08 Phase 4 Task Parameter Selection 报告

## 结论

Phase 4 已在 Phase 1 数据审计基础上冻结 6 个 probe task 的参数，并生成 smoke/main 配置、字段契约、编码器和每任务 directed non-causal graph artifact。required methods 为 `local, zigzag_certified, random_regular`；`dense` 暂列 optional，原因是 4k/8k 长上下文 dense reference 会显著挤占本轮主方法验证预算，v08 第一轮优先完成 theory-aligned zigzag_certified 与同预算 local/random_regular 对照。

| task | padded T | B/d | 模型 | effective batch | main steps | primary metric |
|---|---:|---:|---:|---:|---:|---|
| copy | 6144 | 64/8 | 4x128 | 4 | 80 | copy_token_accuracy |
| selective_copy | 8256 | 64/8 | 3x96 | 4 | 80 | selective_copy_token_accuracy |
| induction_associative_recall | 1024 | 64/8 | 4x128 | 16 | 120 | retrieval_exact_match |
| niah_kv_retrieval | 4160 | 64/8 | 3x96 | 4 | 80 | retrieval_exact_match |
| ruler | 4160 | 64/8 | 3x96 | 4 | 80 | retrieval_exact_match |
| lra_listops | 6016 | 64/8 | 4x128 | 4 | 80 | listops_accuracy |

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| resolved_padded_sequence_length | 图和位置编码使用的补齐后长度 | token 数 | Phase 1 长度与 B 补齐 | 决定 attention mask 尺寸 | 无 |
| resolved_graph_block_size | zigzag/local 的 block 大小 | token 数 | Phase 4 选择 | 控制局部预算和 q | 无 |
| resolved_graph_degree_or_budget | zigzag H 图 degree | 整数 | Phase 4 选择 | 控制稀疏远程边预算 | 无 |
| resolved_graph_max_parallel_edges_per_block_pair | G 图 block-pair 并行边上限 | not_capped | Phase 4 选择 | 避免对小 q 大 B 任务施加数学上不可满足的人为上限；实际重复率由 graph certificate 记录 | 无 |
| resolved_layers | Transformer 层数 | 层 | Phase 4 选择 | 记录模型容量 | 无 |
| resolved_d_model | hidden 维度 | 维度 | Phase 4 选择 | 记录模型容量 | 无 |
| resolved_effective_batch_size | 梯度累积后的有效 batch | 样本数 | batch_size * gradient_accumulation_steps | 保证 method 间公平 | 无 |
| resolved_train_budget_value | main 训练步数 | steps | Phase 4 选择 | 复现实验预算 | smoke 使用 smoke.steps 并记录 |
| resolved_log_every | 训练日志间隔 | steps | logging gate | 满足 1% 日志覆盖 | 无 |
| attention_contract | 注意力合同 | non_causal | v08 手册 | 保证理论对齐 | 不满足则不得进入主结果 |
| causal | 是否 causal mask | false | v08 手册 | 防止 LM 化 | 不满足则不得进入主结果 |
| graph_directionality | 图方向性 | directed | v08 手册和 graph artifact | 对齐 directed expander | 无 |
| primary_metric | 每个 task 主指标 | metric 名称 | Phase 1 schema 与手册 | 主比较字段 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 13 |
| unexplained_parameters | [] |
| english_only_sections | [] |
