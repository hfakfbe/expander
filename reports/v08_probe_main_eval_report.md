# v08 Phase 6 Full Train + Eval 主评估报告

## 结论

Phase 6 已按 v08 手册重新完成 Full Train + Eval。此前 80/120 steps 的短扫产物已归档到 `ref/archive_v08_short_sweep_audit_failure/`，本报告只基于新的全量训练结果。

- 配置文件: `configs/probes_v08_main.json`
- 参数清单: `configs/probes_v08_task_parameters.json`
- 输出目录: `outputs/probes_v08_main/`
- 汇总文件: `outputs/probes_v08_main/summary.json`
- 汇总状态: `status=ok`, `completed_runs=18`, `expected_runs=18`
- 字段审计: `result_field_audits_passed=18`, `result_field_audits_total=18`
- 必跑方法: `local`, `zigzag_certified`, `random_regular`
- 种子: `seed=0`
- 远端环境: `host=hhpc`, `gpu_name=NVIDIA A100-SXM4-80GB`, `torch_version=2.10.0+cu128`, `python_version=3.10.0`
- 运行代码指纹: `git_commit=11ac24dec1564bc38bff8a5360cffb7a9fe5fb47`, `git_dirty=false`

## Full Train 预算

训练步数不再使用 smoke 的 80/120 steps。每个任务的 main budget 解析为 `ceil(train_examples / effective_batch_size)`，即至少覆盖一轮 full train split。

| task | train_examples | batch | accum | effective_batch | main_steps | test_examples | log_every | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 10000 | 8 | 1 | 8 | 1250 | 1000 | 96 | 15/13 |
| selective_copy | 10000 | 8 | 1 | 8 | 1250 | 1000 | 96 | 15/13 |
| induction_associative_recall | 180000 | 16 | 1 | 16 | 11250 | 4000 | 99 | 115/113 |
| niah_kv_retrieval | 10000 | 8 | 1 | 8 | 1250 | 500 | 96 | 15/13 |
| ruler | 12000 | 8 | 1 | 8 | 1500 | 3000 | 100 | 16/15 |
| lra_listops | 96000 | 8 | 1 | 8 | 12000 | 2000 | 100 | 121/120 |

## 主结果

| task | method | steps | primary_metric | value | test_loss | final_train_loss | logged/min | random_align_err_max |
|---|---|---:|---|---:|---:|---:|---:|---:|
| copy | local | 1250 | copy_token_accuracy | 0.0162354 | 4.12795 | 4.12797 | 15/13 | n/a |
| copy | zigzag_certified | 1250 | copy_token_accuracy | 0.0160576 | 4.12787 | 4.1287 | 15/13 | n/a |
| copy | random_regular | 1250 | copy_token_accuracy | 0.0160093 | 4.12782 | 4.12829 | 15/13 | 0 |
| selective_copy | local | 1250 | selective_copy_token_accuracy | 0.07 | 2.64183 | 2.64483 | 15/13 | n/a |
| selective_copy | zigzag_certified | 1250 | selective_copy_token_accuracy | 0.069 | 2.64122 | 2.64364 | 15/13 | n/a |
| selective_copy | random_regular | 1250 | selective_copy_token_accuracy | 0.0746875 | 2.64113 | 2.64797 | 15/13 | 0 |
| induction_associative_recall | local | 11250 | retrieval_exact_match | 0 | 8.32111 | 8.31978 | 115/113 | n/a |
| induction_associative_recall | zigzag_certified | 11250 | retrieval_exact_match | 0 | 8.3209 | 8.31736 | 115/113 | n/a |
| induction_associative_recall | random_regular | 11250 | retrieval_exact_match | 0 | 8.32127 | 8.32397 | 115/113 | 0 |
| niah_kv_retrieval | local | 1250 | retrieval_exact_match | 0.11 | 2.204 | 2.1666 | 15/13 | n/a |
| niah_kv_retrieval | zigzag_certified | 1250 | retrieval_exact_match | 0.13 | 2.20392 | 2.24809 | 15/13 | n/a |
| niah_kv_retrieval | random_regular | 1250 | retrieval_exact_match | 0.106 | 2.20789 | 2.25163 | 15/13 | 0 |
| ruler | local | 1500 | retrieval_exact_match | 0 | 3.69104 | 3.58209 | 16/15 | n/a |
| ruler | zigzag_certified | 1500 | retrieval_exact_match | 0 | 2.68691 | 2.53056 | 16/15 | n/a |
| ruler | random_regular | 1500 | retrieval_exact_match | 0 | 2.66503 | 2.65037 | 16/15 | 0 |
| lra_listops | local | 12000 | listops_accuracy | 0.3315 | 1.8855 | 1.53766 | 121/120 | n/a |
| lra_listops | zigzag_certified | 12000 | listops_accuracy | 0.367 | 1.71408 | 1.71986 | 121/120 | n/a |
| lra_listops | random_regular | 12000 | listops_accuracy | 0.355 | 1.70251 | 1.61176 | 121/120 | 0 |

单 seed 结论需要保守解释。`zigzag_certified` 在 `niah_kv_retrieval` 和 `lra_listops` 的主指标高于两个 baseline；`ruler` 和 `induction_associative_recall` 的 exact match 仍为 0，但 loss 有可复核记录；`copy` 和 `selective_copy` 指标较低，不应被解释为已收敛到任务解。

## Random Budget 对齐

`random_regular` 已按 `zigzag_actual_noncausal_per_query_unique_k` 对齐。对齐单位是每个 query 的 non-causal unique K，而不是只对齐全局平均值；因此 `random_k_alignment_error_max=0` 是严格逐 query 检查。

| task | zigzag_k_mean | random_k_mean | zigzag_pairs | random_pairs | err_mean | err_max |
|---|---:|---:|---:|---:|---:|---:|
| copy | 120.578125 | 120.578125 | 740832 | 740832 | 0.0 | 0 |
| selective_copy | 120.578125 | 120.578125 | 995493 | 995493 | 0.0 | 0 |
| induction_associative_recall | 119.781250 | 119.781250 | 122656 | 122656 | 0.0 | 0 |
| niah_kv_retrieval | 120.578125 | 120.578125 | 501605 | 501605 | 0.0 | 0 |
| ruler | 120.578125 | 120.578125 | 501605 | 501605 | 0.0 | 0 |
| lra_listops | 120.578117 | 120.578117 | 725398 | 725398 | 0.0 | 0 |

## Gate 检查

| gate | resolved value | 结论 |
|---|---|---|
| required run count | 6 tasks x 3 methods x 1 seed = 18 | 通过 |
| completed run count | 18/18 | 通过 |
| result field audit | 18/18 | 通过 |
| attention_contract | `non_causal` for all runs | 通过 |
| causal | `false` for all runs | 通过 |
| graph_directionality | `directed` for all runs | 通过 |
| full train budget | main steps 覆盖每个任务一轮 train split | 通过 |
| log_step_policy | `step_1_every_log_every_and_final_step` | 通过 |
| log coverage | 每个 run 的 `actual_logged_train_step_count >= min_logged_train_step_count` | 通过 |
| random alignment | 每个 `random_regular` run 的 `random_k_alignment_error_max=0` | 通过 |
| checkpoint policy | `manifest_only_no_tensor_checkpoint`，本地输出中无 `.pt/.pth/.ckpt/.safetensors` | 通过 |

## 日志和产物

- Smoke 日志: `logs/probes_v08_smoke_fulltrain_fix_20260616T171057Z.log`
- Main copy 日志: `logs/probes_v08_main_copy_fulltrain_fix_20260616T171545Z.log`
- Main selective_copy 日志: `logs/probes_v08_main_selective_copy_fulltrain_fix_20260616T171545Z.log`
- Main niah 日志: `logs/probes_v08_main_niah_kv_retrieval_fulltrain_fix_20260616T171545Z.log`
- Main ruler 日志: `logs/probes_v08_main_ruler_fulltrain_fix_20260616T171545Z.log`
- Main induction 日志: `logs/probes_v08_main_induction_associative_recall_fulltrain_fix_20260616T172633Z.log`
- Main listops 日志: `logs/probes_v08_main_lra_listops_fulltrain_fix_20260616T172651Z.log`
- Phase 6 每任务报告: `reports/v08_phase6_per_task_reports.md`
- 主结果 JSONL: `outputs/probes_v08_main/results_all.jsonl`
- 主结果 CSV: `outputs/probes_v08_main/results_all.csv`
- 总 summary: `outputs/probes_v08_main/summary.json`
- 每 run 字段审计: `outputs/probes_v08_main/<task>/<method>/result_field_audit.json`

## 参数和字段说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| `version` | 实验版本 | `v08` | `configs/probes_v08_main.json` 和 run 输出 | 区分不同实验版本 | 无 |
| `phase` | 实验阶段 | `phase6_main` | `configs/probes_v08_main.json` | 标记 Full Train + Eval | 无 |
| `profile` | 运行档位 | `main` | `configs/probes_v08_main.json` | 防止与 smoke 混表 | 无 |
| `config` | main 配置路径 | `configs/probes_v08_main.json` | 本轮运行命令 | 复现实验入口 | 无 |
| `task_parameter_manifest` | Phase 4 参数清单 | `configs/probes_v08_task_parameters.json` | config 字段 | 证明 main 读取冻结参数 | 无 |
| `output_root` | main 输出根目录 | `outputs/probes_v08_main/` | config 字段 | 定位主评估产物 | 无 |
| `summary.json` | 阶段汇总文件 | JSON 文件 | 输出目录 | 汇总 run 数、状态和审计数 | 无 |
| `status` | 阶段运行状态 | `ok` | `summary.json` | 判断 Phase 6 是否完成 | 失败时写失败状态和原因 |
| `completed_runs` | 已完成 run 数 | `18` | `summary.json` | 与 expected 对比检查遗漏 | 无 |
| `expected_runs` | 预期 run 数 | `18` | `summary.json` | 由 6 tasks x 3 methods x 1 seed 得到 | 无 |
| `result_field_audits_passed` | 字段审计通过数 | `18` | `summary.json` | 检查每个 run 字段完整性 | 无 |
| `result_field_audits_total` | 字段审计总数 | `18` | `summary.json` | 给通过率提供分母 | 无 |
| `required_methods` | 必跑方法集合 | `local`, `zigzag_certified`, `random_regular` | Phase 4 参数清单 | 定义主比较范围 | 无 |
| `seed` | 随机种子 | `0` | config `seeds` 列表 | 支持复现实验 | 无 |
| `task` | probe 任务名 | 6 个任务名 | config `tasks` 列表 | 标识数据和 metric 口径 | 无 |
| `method` | attention 方法 | `local`, `zigzag_certified`, `random_regular` | config `methods` 列表 | 定义比较组 | 无 |
| `train_examples` | 训练样本数 | examples | Phase 4 `main.train_examples` | 计算 full-train sweep 步数 | 无 |
| `batch` | 单次前向 batch size | examples | Phase 4 `resolved_batch_size` | 记录吞吐和显存设置 | 无 |
| `accum` | 梯度累积步数 | steps | Phase 4 `resolved_gradient_accumulation_steps` | 计算 effective batch | 无 |
| `effective_batch` | 有效 batch size | examples | `batch x accum` | 计算训练步数和公平预算 | 无 |
| `main_steps` | Phase 4 冻结主训练步数 | optimizer steps | Phase 4 `main.steps` | 证明不是短扫 | 无 |
| `steps` | run 实际训练步数 | optimizer steps | run 输出 | 检查实际执行是否等于 main budget | 无 |
| `Full Train` | 全量训练口径 | 至少一轮 train split | Phase 4 预算公式 | 区分 smoke 与 main | 无 |
| `Eval` | final validation/test 评测 | examples | Phase 4 eval budget 和 run 输出 | 记录主结果评测范围 | 无 |
| `test_examples` | final test 使用样本数 | examples | Phase 4 `main.test_examples` 和 run 输出 | 说明 test metric 分母 | 无 |
| `log_every` | 训练日志间隔 | steps | Phase 4 `main.log_every` | 满足 1% logging gate | 无 |
| `logged/min` | 实际日志条数/最低要求条数 | 行数 | `actual_logged_train_step_count` 和 `min_logged_train_step_count` | 验证 logging gate | 无 |
| `actual_logged_train_step_count` | 实际训练日志条数 | 行数 | `metrics.jsonl` 审计 | 证明日志覆盖真实存在 | 无 |
| `min_logged_train_step_count` | 最低训练日志条数 | 行数 | v08 1% gate | 给日志覆盖提供门槛 | 无 |
| `primary_metric` | 主指标名称 | task-specific metric 名 | Phase 4 `primary_metric` 和 run 输出 `primary_metric_name` | 说明 `value` 的含义 | 无 |
| `value` | 主指标值 | 0 到 1 的 accuracy/exact match | run 输出 `primary_metric_value` | 主比较结果 | 无 |
| `test_loss` | final test loss | 交叉熵类 loss | run 输出 | 补充主指标之外的效果判断 | 无 |
| `final_train_loss` | 最后训练 step loss | 交叉熵类 loss | run 输出 | 检查训练末态 | 无 |
| `zigzag_k_mean` | zigzag non-causal 每 query unique K 平均值 | key 数 | `zigzag_actual_k_mean_noncausal` | 审计 sparse budget | 无 |
| `random_k_mean` | random non-causal 每 query unique K 平均值 | key 数 | `random_actual_k_mean_noncausal` | 与 zigzag 对齐比较 | 仅 random 对齐表使用 |
| `zigzag_pairs` | zigzag non-causal attention pair 总数 | pair 数 | `zigzag_attention_pair_count_noncausal` | 审计总稀疏预算 | 无 |
| `random_pairs` | random non-causal attention pair 总数 | pair 数 | `random_attention_pair_count_noncausal` | 验证 random 总预算 | 仅 random 对齐表使用 |
| `err_mean` | random 与 zigzag 每 query K 平均绝对误差 | key 数 | `random_k_alignment_error_mean` | 验证平均预算对齐 | 非 random 主表写 `n/a` |
| `err_max` | random 与 zigzag 每 query K 最大绝对误差 | key 数 | `random_k_alignment_error_max` | 验证逐 query 严格对齐 | 非 random 主表写 `n/a` |
| `random_align_err_max` | 主结果表中的 `err_max` 简写 | key 数 | `random_k_alignment_error_max` | 表格节省宽度 | 非 random 写 `n/a` |
| `random_target_k_source` | random 对齐目标来源 | `zigzag_actual_noncausal_per_query_unique_k` | run 输出 | 说明对齐口径 | 非 random 仅作诊断字段 |
| `random_alignment_mode` | random 对齐模式 | `per_query_noncausal_unique_k` | run 输出 | 说明不是只对齐均值 | 非 random 仅作诊断字段 |
| `attention_contract` | 注意力合同 | `non_causal` | Phase 4 和 run 输出 | v08 理论对齐要求 | 不满足则不得进入主结果 |
| `causal` | 是否使用 causal mask | `false` | Phase 4 和 run 输出 | 防止混入 next-token LM 口径 | 不满足则失败 |
| `graph_directionality` | 图方向性 | `directed` | Phase 4 和 graph artifact | 对齐 directed zigzag 设定 | 不满足则失败 |
| `result_field_audit` | 字段完整性审计 | JSON 审计文件 | 每 run `result_field_audit.json` 和总 summary | 检查缺字段、空字段和日志 gate | 无 |
| `checkpoint_policy` | checkpoint 策略 | `manifest_only_no_tensor_checkpoint` | run 输出 | 说明没有提交 tensor 权重 | 无 |
| `not_applicable` | JSON 中不适用字段占位 | 字符串 | run 输出 | 保留字段契约但说明无值原因 | 非本任务 metric、非 causal after-causal 字段使用 |
| `n/a` | 报告表格中的不适用占位 | 字符串 | 本报告 | 表示该表格单元无适用值 | 非 random 的 random alignment 单元使用 |
| `host` | 远端主机名 | `hhpc` | run 输出 | 记录实际执行机器 | 无 |
| `gpu_name` | GPU 型号 | `NVIDIA A100-SXM4-80GB` | run 输出 | 记录硬件环境 | 无 |
| `A100` | GPU 型号缩写 | `NVIDIA A100-SXM4-80GB` | `gpu_name` | 便于报告正文简写 | 无 |
| `torch_version` | PyTorch 版本 | `2.10.0+cu128` | run 输出 | 记录深度学习运行时 | 无 |
| `python_version` | Python 版本 | `3.10.0` | run 输出 | 记录解释器环境 | 无 |
| `CUDA` | CUDA 运行时 | `12.8` | run 输出 `cuda_version` | 记录 GPU 软件栈 | 无 |
| `git_commit` | 运行代码 commit | `11ac24dec1564bc38bff8a5360cffb7a9fe5fb47` | run 输出 | 追溯代码版本 | 无 |
| `git_dirty` | 运行时工作树是否有未提交改动 | `false` | run 输出 | 判断代码状态是否干净 | 无 |
| `log_path` | 运行日志路径 | 7 个本地 log 文件 | run 输出和本地同步 | 定位 stdout/stderr 证据 | 无 |
| `SHA256` | 文件内容哈希 | 64 位十六进制摘要 | 数据、配置、命令、图工件字段 | 检查输入和工件未被替换 | 无 |
| `JSONL` | 每行一个 JSON 对象的记录格式 | 文本文件格式 | `results_all.jsonl`, `metrics.jsonl` | 支持逐 run 和逐 step 追溯 | 无 |
| `CSV` | 逗号分隔表格格式 | 文本表格格式 | `results_all.csv` | 便于表格软件读取 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 61 |
| unexplained_parameters | [] |
| english_only_sections | [] |

本报告明确区分 Phase 5 smoke 和 Phase 6 Full Train + Eval。未在本报告逐项展开的全量 run 字段由 `outputs/probes_v08_main/*/*/result_field_audit.json` 审计，字段总数为 298，missing fields 和 empty fields 均为空。主要剩余限制是本轮为 `seed=0` 的 first complete sweep，不能替代多 seed 置信区间分析。
