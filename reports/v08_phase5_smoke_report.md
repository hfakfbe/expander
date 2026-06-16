# v08 Phase 5 Smoke 报告

## 结论

Phase 5 smoke 已重新执行并通过。该阶段只验证远端执行链路、日志覆盖、字段审计、attention contract 和 random budget 对齐，不作为主效果评估。完整主评估见 `reports/v08_probe_main_eval_report.md`。

- 配置文件: `configs/probes_v08_smoke.json`
- 参数清单: `configs/probes_v08_task_parameters.json`
- 输出目录: `outputs/probes_v08_smoke/`
- 汇总文件: `outputs/probes_v08_smoke/summary.json`
- 汇总状态: `status=ok`, `completed_runs=18`, `expected_runs=18`
- 字段审计: `result_field_audits_passed=18`, `result_field_audits_total=18`
- 远端环境: `host=hhpc`, `gpu_name=NVIDIA A100-SXM4-80GB`, `torch_version=2.10.0+cu128`, `python_version=3.10.0`
- 运行代码指纹: `git_commit=11ac24dec1564bc38bff8a5360cffb7a9fe5fb47`, `git_dirty=false`
- 运行日志: `logs/probes_v08_smoke_fulltrain_fix_20260616T171057Z.log`

## Smoke 结果

| task | method | steps | primary_metric | value | test_loss | final_train_loss | logged/min | random_err_max |
|---|---|---:|---|---:|---:|---:|---:|---:|
| copy | local | 3 | copy_token_accuracy | 0.0163574 | 4.25873 | 4.25452 | 3/3 | 0 |
| copy | zigzag_certified | 3 | copy_token_accuracy | 0.0185547 | 4.28139 | 4.28031 | 3/3 | 0 |
| copy | random_regular | 3 | copy_token_accuracy | 0.0161133 | 4.27217 | 4.26595 | 3/3 | 0 |
| selective_copy | local | 3 | selective_copy_token_accuracy | 0.109375 | 2.94301 | 2.76116 | 3/3 | 0 |
| selective_copy | zigzag_certified | 3 | selective_copy_token_accuracy | 0.015625 | 2.90357 | 2.87729 | 3/3 | 0 |
| selective_copy | random_regular | 3 | selective_copy_token_accuracy | 0.109375 | 2.83823 | 2.85409 | 3/3 | 0 |
| induction_associative_recall | local | 3 | retrieval_exact_match | 0 | 9.16736 | 9.00608 | 3/3 | 0 |
| induction_associative_recall | zigzag_certified | 3 | retrieval_exact_match | 0 | 9.13343 | 9.11915 | 3/3 | 0 |
| induction_associative_recall | random_regular | 3 | retrieval_exact_match | 0 | 9.15853 | 9.15271 | 3/3 | 0 |
| niah_kv_retrieval | local | 3 | retrieval_exact_match | 0 | 5.40313 | 5.77551 | 3/3 | 0 |
| niah_kv_retrieval | zigzag_certified | 3 | retrieval_exact_match | 0 | 5.41628 | 5.59103 | 3/3 | 0 |
| niah_kv_retrieval | random_regular | 3 | retrieval_exact_match | 0 | 5.67137 | 5.58226 | 3/3 | 0 |
| ruler | local | 3 | retrieval_exact_match | 0 | 5.58393 | 5.61738 | 3/3 | 0 |
| ruler | zigzag_certified | 3 | retrieval_exact_match | 0 | 5.3056 | 5.26495 | 3/3 | 0 |
| ruler | random_regular | 3 | retrieval_exact_match | 0 | 5.48698 | 5.55732 | 3/3 | 0 |
| lra_listops | local | 3 | listops_accuracy | 0.125 | 2.44918 | 2.3917 | 3/3 | 0 |
| lra_listops | zigzag_certified | 3 | listops_accuracy | 0 | 2.52299 | 2.14982 | 3/3 | 0 |
| lra_listops | random_regular | 3 | listops_accuracy | 0 | 2.40535 | 2.32908 | 3/3 | 0 |

## Gate 检查

| gate | resolved value | 结论 |
|---|---|---|
| required run count | 6 tasks x 3 methods x 1 seed = 18 | 通过 |
| completed run count | 18/18 | 通过 |
| result field audit | 18/18 | 通过 |
| attention_contract | `non_causal` for all runs | 通过 |
| causal | `false` for all runs | 通过 |
| graph_directionality | `directed` for all runs | 通过 |
| log_step_policy | `step_1_every_log_every_and_final_step` | 通过 |
| actual_logged_train_step_count | 3 for every run | 通过 |
| min_logged_train_step_count | 3 for every run | 通过 |
| random alignment | `random_k_alignment_error_max=0` for every run | 通过 |

## 参数和字段说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| `version` | 实验版本 | `v08` | `configs/probes_v08_smoke.json` 和 run 输出 | 区分不同实验版本 | 无 |
| `phase` | 实验阶段 | `phase5_smoke` | `configs/probes_v08_smoke.json` | 防止 smoke 与 main 混表 | 无 |
| `profile` | 运行档位 | `smoke` | `configs/probes_v08_smoke.json` | 标明本报告只验证链路 | 无 |
| `config` | smoke 配置路径 | `configs/probes_v08_smoke.json` | 本轮运行命令 | 复现实验入口 | 无 |
| `task_parameter_manifest` | Phase 4 参数清单 | `configs/probes_v08_task_parameters.json` | config 字段 | 证明 smoke 读取冻结参数 | 无 |
| `output_root` | smoke 输出根目录 | `outputs/probes_v08_smoke/` | config 字段 | 定位本阶段所有产物 | 无 |
| `summary.json` | 阶段汇总文件 | JSON 文件 | 输出目录 | 汇总 run 数、状态和审计数 | 无 |
| `status` | 阶段运行状态 | `ok` | `summary.json` | 判断阶段是否完成 | 失败时写失败状态和原因 |
| `completed_runs` | 已完成 run 数 | `18` | `summary.json` | 与 expected 对比检查遗漏 | 无 |
| `expected_runs` | 预期 run 数 | `18` | `summary.json` | 由 6 tasks x 3 methods x 1 seed 得到 | 无 |
| `result_field_audits_passed` | 字段审计通过数 | `18` | `summary.json` | 检查每个 run 字段完整性 | 无 |
| `result_field_audits_total` | 字段审计总数 | `18` | `summary.json` | 给通过率提供分母 | 无 |
| `host` | 远端主机名 | `hhpc` | run 输出 | 记录实际执行机器 | 无 |
| `gpu_name` | GPU 型号 | `NVIDIA A100-SXM4-80GB` | run 输出 | 记录硬件环境 | 无 |
| `A100` | GPU 型号缩写 | `NVIDIA A100-SXM4-80GB` | `gpu_name` | 便于报告正文简写 | 无 |
| `torch_version` | PyTorch 版本 | `2.10.0+cu128` | run 输出 | 记录深度学习运行时 | 无 |
| `python_version` | Python 版本 | `3.10.0` | run 输出 | 记录解释器环境 | 无 |
| `CUDA` | CUDA 运行时 | `12.8` | run 输出 `cuda_version` | 记录 GPU 软件栈 | 无 |
| `git_commit` | 运行代码 commit | `11ac24dec1564bc38bff8a5360cffb7a9fe5fb47` | run 输出 | 追溯代码版本 | 无 |
| `git_dirty` | 运行时工作树是否有未提交改动 | `false` | run 输出 | 判断代码状态是否干净 | 无 |
| `log_path` | 运行日志路径 | `logs/probes_v08_smoke_fulltrain_fix_20260616T171057Z.log` | run 输出和本地同步 | 定位 stdout/stderr 证据 | 无 |
| `task` | probe 任务名 | 6 个任务名 | config `tasks` 列表 | 标识数据和 metric 口径 | 无 |
| `method` | attention 方法 | `local`, `zigzag_certified`, `random_regular` | config `methods` 列表 | 定义比较组 | 无 |
| `seed` | 随机种子 | `0` | config `seeds` 列表 | 支持复现实验 | 无 |
| `steps` | 训练 optimizer step 数 | smoke 每个 run 为 `3` steps | Phase 4 `smoke.steps` 和 run 输出 | 验证链路，不作为主效果 | 无 |
| `primary_metric` | 主指标名称 | task-specific metric 名 | Phase 4 `primary_metric` 和 run 输出 `primary_metric_name` | 说明 `value` 的含义 | 无 |
| `value` | 主指标值 | 0 到 1 的 accuracy/exact match | run 输出 `primary_metric_value` | smoke 表格的结果列 | 无 |
| `test_loss` | test split loss | 交叉熵类 loss | run 输出 | 检查 eval 是否正常执行 | 无 |
| `final_train_loss` | 最后训练 step loss | 交叉熵类 loss | run 输出 | 检查 train 是否正常执行 | 无 |
| `logged/min` | 实际训练日志条数/最低要求条数 | smoke 每个 run 为 `3/3` | `actual_logged_train_step_count` 和 `min_logged_train_step_count` | 验证 logging gate | 无 |
| `random_err_max` | random 与 zigzag 每 query K 最大误差 | key 数；本轮为 `0` | `random_k_alignment_error_max` | 验证 random 同预算 | 非 random 方法只作诊断记录 |
| `attention_contract` | 注意力合同 | `non_causal` | Phase 4 和 run 输出 | v08 理论对齐要求 | 不满足则不得进入主结果 |
| `causal` | 是否使用 causal mask | `false` | Phase 4 和 run 输出 | 防止混入 next-token LM 口径 | 不满足则失败 |
| `graph_directionality` | 图方向性 | `directed` | Phase 4 和 graph artifact | 对齐 directed zigzag 设定 | 不满足则失败 |
| `result_field_audit` | 每 run 字段完整性审计 | JSON 审计文件 | `result_field_audit.json` | 检查缺字段、空字段和日志 gate | 无 |
| `not_applicable` | 不适用占位值 | 字符串 `not_applicable` | run 输出和审计文件 | 明确字段为何无值 | 非本任务 metric、非 causal after-causal 字段使用该值 |
| `SHA256` | 文件内容哈希 | 64 位十六进制摘要 | 数据、配置、命令、图工件字段 | 检查输入和工件未被替换 | 无 |
| `JSONL` | 每行一个 JSON 对象的日志格式 | 文本文件格式 | `results_all.jsonl`, `metrics.jsonl` | 支持逐 run 和逐 step 追溯 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 38 |
| unexplained_parameters | [] |
| english_only_sections | [] |

本报告没有把 smoke 当作主评估；`steps=3` 仅用于链路验证。完整 run 字段审计保存在 `outputs/probes_v08_smoke/*/*/result_field_audit.json`，总审计保存在 `outputs/probes_v08_smoke/result_field_audit.json`。
