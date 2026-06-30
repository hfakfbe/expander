# Random sparse copy 多跳改造尝试报告

## 2026-06-30 严格复验：pure random no-local，去掉 relative bias/top-k

这次复验按新的约束重跑：`random_regular` 不再带 block-local 边，`density=0.1` 直接按 pure random non-local candidates 采样；同时关闭 `relative_attention_bias` 和 `top-k sparse attention`，只比较 plain random、generic attention-rollout memory、以及 weighted multiscale rollout `[1,2]` 权重 `[0.9,0.1]`。

**当前结论：**

- plain pure random no-local 仍明显不够：final/test token acc `0.7201103515625`，seq acc `0.0`。
- single-hop generic rollout memory 有显著增益：final/test token acc `0.9977763671875`，seq acc `0.0`。
- weighted multiscale `[0.9,0.1]` 是负优化：final/test token acc `0.9844814453125`，低于 single-hop 的 `0.9977763671875`。
- 三条新实验均确认 `local_attention_pair_count=0`，`random_attention_top_k=0`，`random_relative_attention_bias=false`，实际 density `0.09999990463256836`。

### 复验设置

| 路线 | config | rollout | multiscale | local 边 | relative bias | top-k | LR/schedule |
|---|---|---:|---|---:|---:|---:|---|
| plain no-local random | `configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_pure_no_local_plain_density10_lr1e3_e4_b4a2.json` | 否 | - | 0 | false | 0 | const `1e-3`, 5000 steps |
| single-hop rollout | `configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_pure_no_local_rollout_singlehop_density10_lr1e3_e4_b4a2.json` | 是 | `[1]`, `[1.0]` | 0 | false | 0 | const `1e-3`, 5000 steps |
| weighted multiscale rollout | `configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_pure_no_local_rollout_multiscale12_w0p9_0p1_density10_lr1e3_e4_b4a2.json` | 是 | `[1,2]`, `[0.9,0.1]` | 0 | false | 0 | const `1e-3`, 5000 steps |

代码/配置版本：

- pure no-local 支持与主 multiscale config：`12dead9a61aee3f6ee45e3da51f0b17a0f9dbfbe`
- ablation configs：`7dbc145f0f2edb65f878cd56e40e00e616c2aab7`

注意：multiscale 训练开始时 checkpoint identity 绑定 `12dead9a61aee3f6ee45e3da51f0b17a0f9dbfbe`；中途为了启动 ablation，远端 marker 曾切到 `7dbc145f0f2edb65f878cd56e40e00e616c2aab7`，所以 multiscale `summary.json` 顶层 `git_commit` 显示 `7dbc145...`，但 `identity.branch_head_commit` 和 final eval identity 均为 `12dead9...`。

### 最终结果

| 路线 | train diagnostic token/seq | train full token/seq | final/test token acc | final/test seq acc | final/test loss | checkpoint sha256 |
|---|---:|---:|---:|---:|---:|---|
| plain no-local random | `0.71978759765625 / 0.0` | `0.72022783203125 / 0.0` | `0.7201103515625` | `0.0` | `1.1717212433815003` | `4f7b405e1a531acd462a3f877af8e11f8b0b2db3c90e03c87ab7487a033228f7` |
| single-hop rollout | `0.997802734375 / 0.0` | `0.9978169921875 / 0.0006` | `0.9977763671875` | `0.0` | `0.009312204461544751` | `df04913a3643dcddb9585ef51974d3b614760ca4daf50e95334405336ab52797` |
| weighted multiscale `[0.9,0.1]` | `0.98468017578125 / 0.0` | `0.98452353515625 / 0.0` | `0.9844814453125` | `0.0` | `0.06501547843962908` | `a2e382c620a2e69479097673e276bb5e376168ba8310c5bb6ed34a82de677066` |

### 与旧 xlsx 的差异原因

`../copy_all_results_summary_to_date_filled_density.xlsx` 中最接近的旧记录是：

```text
copy_noncausal / random / layer=8 / 每层单独随机=是 / density=0.099999905
test loss = 2.731628
test token acc = 0.360513
```

它对应外部归档：

```text
../expander_external_artifacts/expander-probes-corrected-valid-as-test-l8-log5/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/random_layerwise_density_sweep_results.csv
```

以及 run：

```text
q32_B64_d32_l8_log5_random_layerwise_density10/random_regular/seed0
checkpoint = train_final_step625.pt
commit = d4958fafe8c30d87b36ba905cf39bbf5bde6df8d
```

旧表和本次 plain 不可直接横向比较，原因有三点：

1. **训练步数不同**：旧表 density10 只训练 `625` optimizer steps；本次训练 `5000` steps。本次 plain 在 step625 的 train diagnostic token acc 已经是 `0.56170654296875`，到 step5000 才到 `0.71978759765625`。
2. **学习率调度不同**：旧表默认来自 task manifest：`base_lr=3e-4`、cosine decay 到 `min_lr=3e-5`、effective batch `16`。旧 metrics 显示 step625 时 LR 已降到 `3e-05`。本次三条新实验均是 const `1e-3`、effective batch `8`。
3. **density 口径不同**：旧 `random_regular` 实现无条件先加 block-local 边，再补 random cross edges；旧 density10 的 `0.1` 是 local+remote 总边密度。对 `T=2048, B=64`，local 边约 `131072` 条，密度 `0.03125`；旧 density10 中真正 remote random 边约 `288358` 条，remote density `0.0687499`。本次 no-local density10 是 `419430` 条全 remote random 边，相当于旧 remote random 边数的约 `1.45x`。

因此，旧 xlsx 的 `0.360513` 更适合作为“旧 local+remote density 口径、短训练、cosine 低 LR”的历史 baseline；本次 `0.720110` 是“pure remote no-local、长训练、const 1e-3”的新 baseline。

### 解释

在严格去掉 local、relative bias、top-k 后，generic rollout memory 的潜力仍然很强：single-hop 从 plain 的 `0.7201` 提升到 `0.9978` token acc。但它还没有解决 exact-match 序列级错误，final/test 1000 条没有完整全对样本。

weighted multiscale 这次没有提升，反而把 single-hop 的结果从 `0.9978` 拉低到 `0.9845`。当前最合理解释是，两跳 rollout 在没有 relative bias/top-k 的情况下引入了额外混合噪声；90% one-hop + 10% two-hop 并没有稳定补足多跳，反而破坏了已经学到的 one-hop memory channel。后续如果继续看 multiscale，应先做更小 two-hop 权重、late-start 或 learnable gate，而不是把 `[0.9,0.1]` 当成正结果。

---

日期：2026-06-27  
远端训练机器：`huiwei`  
远端项目目录：`/home/huiwei/ysx/zigzag_attention_heads_trial`  
本地整理文档：`/Users/sxye/Documents/expander/reports/random_copy_multihop_attempts_report.md`

> 以下为 2026-06-27 的历史探索记录，包含 relative bias/top-k/local-edge 口径下的旧结论；严格 no-local、no-relative、no-top-k 的当前结论以上方 2026-06-30 复验为准。

## 1. 一句话结论

在 corrected copy 任务上，`random_regular` 的 plain sparse attention 不够；真正有效的 clean 路线是：

```text
random layerwise independent sparse mask
+ Q/K-only RoPE
+ relative attention bias
+ top-k sparse attention
+ generic attention-rollout memory
+ weighted multiscale rollout: 90% one-hop memory + 10% two-hop memory
```

其中最强的 clean final/test 结果是：

| 路线 | train 口径 | final/test token acc | final/test seq acc | 作弊风险 |
|---|---|---:|---:|---|
| weighted multiscale rollout `[1,2]`、权重 `[0.9,0.1]` + relative bias + topk8，不带 history readout | full train 1/1；train-final checkpoint final-eval | `0.999998046875` | `0.998` | 低到中 |
| rollout memory + relative bias + topk8，不带 history readout | full train 接近 1/1；best checkpoint final-eval | `0.9999970703125` | `0.997` | 低到中 |
| rollout memory + logit history readout + topk8 | full train 1/1；final-eval | `0.9999951171875` | `0.995` | 低到中 |
| margin continuation topk16 | full train 1/1；final-eval | `0.9999951171875` | `0.995` | 低到中，训练公平性需说明 |
| cross-bias +1 topk8 | full train 1/1；final-eval | `0.9999951171875` | `0.995` | 低到中 |
| learned-gate topk16 | full train 1/1；final-eval | `0.9999931640625` | `0.993` | 低到中 |

如果按“接近 1 就可以”的口径，最新 clean 结果已经达标；如果严格要求 final/test `seq acc=1.0`，还没达成。

注意：最新 `seq acc=0.998` 是 test 1000 条中 998 条完全正确；`token acc=0.999998046875` 是 1,024,000 个 test target token 中只错 2 个。

## 2. 实验合规边界

这份报告只把“不明显任务特化”的结构当作可申报结果。根据 `ref/copy_experiment_correction_spec_v01.md`，corrected copy 的关键合同是：

- 输入固定长度 `2048`：前 1024 个 source token，后 1024 个 marker token；
- 只在 marker positions `1024..2047` 上预测 target；
- target 不追加进输入，不做 teacher forcing，不创建额外 PAD/readout slots；
- RoPE 只作用在 attention 的 Q/K，不作用在 V；
- 训练中不读 test，final/test 只做最后验收；
- random 的 source-to-marker reachability 必须按 `mask[marker_query, source_key]` 方向记录。

我用下面这些红线判断作弊嫌疑：

- 显式写入 `source j -> marker 1024+j`：高风险；
- 直接传 token id、one-hot 或答案概率：高风险；
- 给 V 注入位置编码：高风险；
- 只在 random 可见边上做通用 hidden/memory 传播：低到中风险；
- 用 relative bias、top-k、history readout 这类强归纳偏置：不是直接作弊，但报告必须明说。

## 3. plain random 为什么不够

plain `random_regular` 的失败不是因为图完全不可达。最终 random mask 的结构诊断显示：

```text
copy_target_in_1hop_rate = 0.0634765625
copy_target_in_2hop_rate = 1.0
copy_target_in_8hop_rate = 1.0
copy_average_shortest_path = 1.9365234375
```

也就是说，大多数 marker 到对应 source 不是一跳直接可见，但两跳已经全可达。问题在于普通 Transformer block 没有一条稳定的“独立记忆通道”保存多跳传播过来的内容。

举个直观例子，marker position `1536` 要复制 source position `512`：

```text
position 1536 --第1层 sparse attention--> 中间位置 a
position a    --第2层 sparse attention--> source position 512
```

图上有路，不等于模型会稳定使用这条路。普通 hidden 每层都会和 residual/FFN/其它邻居混在一起；copy 又要求每个 token 都对，少数位置传播不稳，sequence accuracy 就会掉到 0。plain random layerwise density 0.1 早期约 `token≈0.873, seq=0`，说明“有可达路径”远远不够。

## 4. 路线 A：任务特化 staged multihop route

### 改进到底是什么

这条路线尝试显式构造 8-hop 路由，让 marker 沿多层中转逐步拿到对应 source：

```text
layer 1: marker 1536 走到某个中转点
layer 2: 中转点继续走
...
layer 8: 把 source 512 的信息送到 marker 1536
```

相关开关包括：

- `random_multihop_copy_route`
- `random_route_layerwise_staged`
- `random_route_transport`
- `random_route_transport_mode`
- `memory_replace` / `memory_residual`

### 为什么可能有效

它把 copy 最难的对应关系直接写进图结构：`marker 1024+j` 最终能沿人为安排的路线找到 `source j`。这样模型不用自己从随机图中学习稳定多跳路由。

### 结果

这条路线没有作为正式 clean 结果推进。它适合做 sanity check：证明“如果路线给对，模型能 copy”。但它不能证明 random sparse 架构自己学会了通用多跳信息。

### 作弊嫌疑

高。只要路由构造围绕 `source j -> marker 1024+j` 设计，即使拆成 8 hop，本质还是把 copy offset 写进架构。最终报告不应使用这条路线作为主结果。

## 5. 路线 B：Value RoPE / positional side-channel

### 改进到底是什么

尝试把相对位置信息也放进 value 表示，即让 V 不只是内容，还携带“我来自哪个位置”的编码。直觉上，copy 任务需要稳定识别 `marker 1024+j` 和 `source j` 的固定相对距离；如果 V 自带位置，读出就会容易很多。

### 结果

若只看 gate-overfit，小样本可以很快到 1/1；例如若干 `value_rope_relative_bias` 变体在几十步内过了 2-example gate。但这不是正式可申报结果。

### 作弊嫌疑

高。corrected spec 明确说 RoPE 只作用 Q/K，不作用 V。给 V 加位置编码会形成强位置 side-channel，即使没有直接写 copy offset，也偏离了任务合同。

## 6. 路线 C：token rollout memory

### 改进到底是什么

这条路线不是传播 hidden state，而是传播 token identity 或 token logits/probabilities：

```text
token one-hot / token logits
  -- sparse attention weights -->
下一层 token distribution
  -- sparse attention weights -->
更远处 token distribution
```

### 为什么可能有效

copy 的答案就是 token id。直接传播 token distribution 可以绕开“先把 token 编成 hidden，再从 hidden 解码”的难题。

### 结果

小样本 gate-overfit 能通过，但没有作为 clean 正式路线推进。

### 作弊嫌疑

高。它不显式写 copy offset，但直接传答案空间的信息，像答案通道。若目标是证明 sparse attention 学会 hidden-level 多跳传播，这条不合格。

## 7. 路线 D：learned edge bias / learned edge transport

### 改进到底是什么

这条路线给随机可见边增加可学习参数，让每条边不仅是“能看见”，还可以学习“这条边更应该传信息”：

- `random_learned_attention_edge_bias`
- `random_learned_edge_memory_transport_mode`
- learned local/neighbor edge log bias

### 为什么可能有效

随机图中存在正确 source 的多跳路径，但很多边是噪声。边上可学习 bias 可以把概率质量集中到更有用的边上。

### 结果

没有进入最终 clean 候选。主要原因不是它一定无效，而是固定长度 copy 上，per-edge 参数很容易学成固定位置图的记忆。

### 作弊嫌疑

中到高。它没有直接写 `j -> 1024+j`，但固定位置边可学习，在固定长度任务上可能近似记住一批 copy 图路径。若要申报，至少需要跨 mask、跨 seed、跨长度验证；当前不建议作为主结果。

## 8. 路线 E：generic attention-rollout memory

这是最核心、最可辩护的路线。

### 改进到底是什么

普通 sparse Transformer 每层只更新 hidden：

```text
h_l = TransformerBlock(h_{l-1})
```

rollout memory 增加一条独立 memory state：

```text
rollout_state_0 = token embedding

每一层：
  1. 用当前层 Q/K 在同一张 random sparse mask 上算 attention 权重
  2. 用这些权重把 rollout_state 沿 sparse 边传播一步
  3. 用 lazy update 保留一部分旧 memory
  4. 把 memory 注入 hidden，帮助后续层和最终读出
```

核心更新：

```text
propagated = AttentionWeights_l @ rollout_state_{l-1}
rollout_state_l = 0.5 * rollout_state_{l-1} + 0.5 * propagated
h_l = h_l + 2.0 * rollout_state_l
```

关键点是：它没有新增 copy 专用边，只沿模型原本可见的 random sparse edges 传播 memory。换句话说，它不是“给答案修路”，而是“让已有随机路上的信息别在 hidden 混合中丢掉”。

### 为什么它能学多跳

如果某个 marker 需要两跳拿到 source：

```text
marker -> intermediate -> source
```

第 1 层可以把 source 信息推到 intermediate 附近，第 2 层再把它推到 marker。因为 rollout_state 是独立 memory，不完全等同于普通 hidden，它更像一个“沿图传递的包裹”，不会每层都被 FFN 和其它 residual 彻底冲散。

### 结果

先前最强 clean 结果来自不带 history readout 的 rollout memory + relative bias + topk8：

```text
config:
configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_rollout_lazy0p5_memoryscale2_relative_bias_scale2p0_topk8_density10_lr3e5_finetune_from_ft5000_final_b8.json

remote run:
/home/huiwei/ysx/zigzag_attention_heads_trial/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5_random_layerwise_density10_lr3e5_finetune_from_ft5000_final/random_regular/seed0
```

训练和 final/test：

| 口径 | token acc | seq acc | 备注 |
|---|---:|---:|---|
| summary 最后 train diagnostic | `1.0` | `1.0` | 训练过程中未读 test |
| full train eval | `0.99999990234375` | `0.9999` | 10000 条 train 中 1 条不全对 |
| final/test best checkpoint | `0.9999970703125` | `0.997` | test 1000 条中 997 条全对 |

这个 final/test 用的是该 run 的 `checkpoint_step100.pt`，不是最后的 `train_final_step1000.pt`：

```text
checkpoint_path = outputs/.../checkpoints/checkpoint_step100.pt
checkpoint_sha256 = e9b1d35ee29998dfd1ef603bd8a92dccfe50e9729c863a51cfa091127838c437
first_test_read_at = 2026-06-24T01:05:25.489421+00:00
position_parameter_count = 0
padding_positions = 0
```

必须如实写成“best checkpoint final-eval”，不能偷换成“最后一步模型 final-eval”。

后来我在这条 clean 路线之上做了一个更稳的通用多跳改动：weighted multiscale rollout memory。它仍然不带 history readout，不传 token id，不加 value-position side-channel，也不改 random mask。

具体说，原来的 rollout memory 每层只做一种传播：

```text
one_hop = AttentionWeights_l @ rollout_state
rollout_state_l = 0.5 * rollout_state + 0.5 * one_hop
```

weighted multiscale 同时看两个尺度：

```text
one_hop = A_l @ rollout_state
two_hop = A_l @ (A_l @ rollout_state)

propagated = 0.9 * one_hop + 0.1 * two_hop
rollout_state_l = 0.5 * rollout_state + 0.5 * propagated
```

可以把它理解成：每层主要相信“走一步”的稳定传播，同时给一点点“朋友的朋友”信息。这样 rare position 如果一跳上拿不到足够干净的信息，还有 10% 的两跳 lookahead；但 90% 仍保留原本成功的一跳逐层传播，避免两跳信息把短路、局部信息和已经正确的 memory 冲散。

我也试过更激进的 unweighted multiscale `[1,1,2]`，也就是三份里两份一跳、一份两跳。它在小 batch diagnostic 上能到 1/1，但 full train 没过：

| multiscale 版本 | train token | train seq | final/test |
|---|---:|---:|---|
| `[1,1,2]`，等权 | `0.9999994140625` | `0.9994` | 未做；train 已有 6 条错序列 |
| `[1,2]`，权重 `[0.9,0.1]` | `1.0` | `1.0` | token `0.999998046875`，seq `0.998` |

最新最好 run：

```text
config:
configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_rollout_multiscale12_w0p9_0p1_lazy0p5_memoryscale2_relative_bias_scale2p0_topk8_density10_lr3e5_resume_from_nohistory_final_b4a2.json

remote run:
/home/huiwei/ysx/zigzag_attention_heads_trial/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5_random_layerwise_density10_multiscale12_w0p9_0p1_lr3e5_resume_from_nohistory_final/random_regular/seed0
```

训练和 final/test：

| 口径 | token acc | seq acc | margin min | 备注 |
|---|---:|---:|---:|---|
| full train eval | `1.0` | `1.0` | `2.132284164428711` | 训练集全对，未读 test |
| final/test | `0.999998046875` | `0.998` | `-0.5164852142333984` | test 1000 条中 998 条全对 |

checkpoint：

```text
checkpoint_path = outputs/.../checkpoints/train_final_step1000.pt
checkpoint_sha256 = 2c5e8b89a9030cfe7d2dc2c6bd40af506bb09b20c84f5288ffd154dcc9a3f744
first_test_read_at = 2026-06-27T09:01:28.901295+00:00
random_rollout_memory_multiscale_steps = [1, 2]
random_rollout_memory_multiscale_weights = [0.9, 0.1]
random_history_output_logits = false
```

这条比前一个 `seq=0.997` 更好，而且不是用 `checkpoint_step100` 做 test 选择，而是用该 continuation 的 train-final checkpoint 做 final-eval。

### 作弊嫌疑

低到中。

低风险点：

- `random_multihop_copy_route=false`；
- `random_token_rollout_memory=false`；
- `random_value_position_encoding=none`；
- `random_learned_attention_edge_bias=false`；
- `random_learned_edge_memory_transport_mode=null`；
- `random_history_output_logits=false`；
- RoPE 只进 Q/K；
- 只沿 random sparse mask 的真实可见边传播；
- 训练 summary 记录 `test_read_during_training=false`。

中风险点：

- `relative_attention_bias` 是强相对位置归纳偏置；copy 的固定 offset 正好非常吃这个偏置；
- `topk8` 会强化路径选择能力；
- weighted multiscale 的两跳分支更直接服务“多跳可达”问题；它是通用图传播，不写 copy offset，但需要在报告里明确说明；
- 历史上的 `seq=0.997` 版本 final 选了 checkpoint step100，需要说明 checkpoint selection 风险。最新 `seq=0.998` 版本用 train-final checkpoint，风险小一些。

我的判断：weighted multiscale rollout memory 是当前最干净、最强的可申报路线。

## 9. 路线 F：history readout / weighted-sum / logit-weighted readout

### 改进到底是什么

普通模型只读最后一层 hidden。history readout 会把多层 hidden 或 rollout state 也用于预测。它解决的问题是：某个位置可能第 4 层已经拿到正确 source token，第 8 层反而被后续传播稀释。

尝试过：

1. `weighted_sum`：先把多层 state 混成一个 state，再过 token head；
2. `logit_weighted_sum`：每层 state 各自出 logits，再按全局权重混 logits；
3. `confidence_logit_weighted_sum`：每个位置动态偏向置信度高的层。

### 结果

| 变体 | train 结果 | final/test 结果 |
|---|---|---|
| weighted history rollout topk8 | 多个 continuation 达到 train 1/1 | final token `0.999994140625`, seq `0.994` |
| logit-weighted rollout topk8 | full train 1/1 | final token `0.9999951171875`, seq `0.995` |
| logit-weighted rollout topk16 | full train 1/1 | final token `0.999994140625`, seq `0.994` |
| confidence logit-weighted topk8 | train summary 1/1 | final token `0.9999931640625`, seq `0.993` |

代表 run：

```text
topk8 logit history:
/home/huiwei/ysx/zigzag_attention_heads_trial/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5_random_layerwise_density10_lr1e6_history_logitweighted_rollout_lazy0p5_memoryscale2_relative_bias_scale2p0_topk8_b8_resume_from_logitweighted_step503_fulltrain_cleanup/random_regular/seed0

confidence topk8:
/home/huiwei/ysx/zigzag_attention_heads_trial/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5_random_layerwise_density10_confidence_lr3e6_history_confidence_logitweighted_rollout_lazy0p5_memoryscale2_relative_bias_scale2p0_topk8_b8_resume_from_lr1e6_step1253/random_regular/seed0
```

### 作弊嫌疑

低到中。它不写 copy offset，也不传 token side-channel；但它确实增强了读出能力，让模型可以从“最早已经正确的层”拿答案。若使用这条路线，报告里必须说是“multi-layer memory readout”，不能说成原始 random attention。

## 10. 路线 G：rollout steps=2 / hard rollout / multi-step 替换

### 改进到底是什么

既然 copy 信息常常需要两跳，尝试让单层 rollout 内部直接做两步传播，或者用 hard/top route 替代 soft 权重，希望更像显式多跳 walk。

这里和最新 weighted multiscale 的差别很关键：

- `rollout_steps=2` 是“把每层传播替换成两跳”，一跳通道被削弱；
- weighted multiscale 是“保留 90% 一跳，再加 10% 两跳”，主干仍然是已经验证有效的一跳逐层传播。

### 结果

效果反而差：

| 变体 | train token | train seq | loss/现象 |
|---|---:|---:|---|
| history weighted rollout steps2b | `0.9981689453125` | `0.25` | 不稳定 |
| logit history rollout steps2 lr=1e-5 resume | `0.9925765991210938` | `0.0` | 明显退化 |
| weighted rollout steps2 lr=3e-6 resume | `0.9842681884765625` | `0.0` | 更差 |
| hard rollout continuation | 未进入最终候选 | hard 选择太早离散化，训练不稳 |

### 为什么会差

“两跳”不是简单把每层都强制走两步。现有成功模型依赖的是每层逐步传播 + lazy memory 保留；强行把一步替换成两步，会让短路径和一跳直接可见的信息被过度扩散，反而稀释。

### 作弊嫌疑

低。它仍然沿 random 边传播，没有任务特化；只是效果差。

## 11. 路线 H：margin continuation

### 改进到底是什么

普通 cross entropy 只要求正确 token logit 最大。margin continuation 额外要求正确 logit 比 runner-up 高：

```text
margin = true_logit - max_wrong_logit
loss += weight * relu(target_margin - margin)
```

目的是让低 margin 位置更稳，降低 final/test 中“只错几个 token”的概率。

### 结果

从 clean topk16 checkpoint 继续：

| 配置 | train token | train seq | train margin min | train p01 | final token | final seq |
|---|---:|---:|---:|---:|---:|---:|
| margin m1/w0.05 | `1.0` | `1.0` | `3.4098` | `8.7653` | 未正式作为主结果 | 未正式作为主结果 |
| margin m2/w0.02 | `1.0` | `1.0` | `3.4098` | `8.7653` | `0.9999951171875` | `0.995` |

代表 final-eval：

```text
/home/huiwei/ysx/zigzag_attention_heads_trial/logs/final_eval_margin_m2_topk16_legacy_20260626_172119.log
checkpoint_sha256 = d73d3a49e0ba07a096794ba0874a5f02157944b177d2273a9969460696ac1675
identity_compatibility = legacy_added_default_fields_match
```

### 作弊嫌疑

低到中。它只用 train label，不读 test，不写 route；不是数据泄漏。但如果 dense baseline 没用同样 objective，公平性会被质疑。因此它适合作为 robustness continuation，不建议作为主结果。

## 12. 路线 I：learned rollout update / learned rollout scale

### 改进到底是什么

固定 rollout memory 用：

```text
new_memory = 0.5 * old_memory + 0.5 * propagated_memory
h = h + 2.0 * new_memory
```

learned-gate 把 `0.5` 和 `2.0` 改成每层可学习参数：

```text
alpha_l = sigmoid(parameter_l)
scale_l = softplus(parameter_l)

new_memory_l = alpha_l * old_memory + (1-alpha_l) * propagated_memory
h_l = h_l + scale_l * new_memory_l
```

初始值等价于旧行为，所以它是“给每层调节传播/保留比例”，不是另起炉灶。

### 结果

| 变体 | train token | train seq | train margin min | final token | final seq |
|---|---:|---:|---:|---:|---:|
| learned-gate topk16 lr=1e-5 | `1.0` | `1.0` | `3.6682` | 未做 final | 未做 final |
| learned-gate topk16 lr=3e-6 | `1.0` | `1.0` | `2.8935` | `0.9999931640625` | `0.993` |
| learned-gate topk0 lr=3e-6 | `1.0` | `1.0` | `3.5176` | 未做 final | 未做 final |

代表 final-eval：

```text
/home/huiwei/ysx/zigzag_attention_heads_trial/logs/final_eval_learnedgate_topk16_lr3e6_20260626_165754.log
checkpoint_sha256 = 99d0eab3fd9ef0c94302a4538fad32d7b42fb4286783014e8e542fc7b7ac33fa
```

### 作弊嫌疑

低到中。它是通用层级 gate，不知道 copy offset；但它增加了模型自由度，而且有些 checkpoint 是从旧结构 non-strict 加载新增参数继续训练。最终报告必须说明加载方式和缺失参数。

## 13. 路线 J：top-k / cross-only / cross-bias

### top-k 是什么

top-k 在每个 query 的可见 sparse keys 里只保留权重最高的 k 条路径：

- `topk8`：路线更尖锐，噪声少；
- `topk16`：覆盖更宽，可能更稳；
- `topk0`：不裁剪，所有可见边参与。

结果上，clean topk8 最好，topk16 略低。

### cross-only 是什么

cross-only 只允许 rollout memory 沿跨 block/远程边走，不走 local 边。动机是 copy 需要远距离信息，local 边可能稀释远程 memory。

结果很差，刚开始就破坏已学表示：

```text
step 1 diagnostic token acc ≈ 0.7396
step 50 diagnostic token acc ≈ 0.8184
step 100 diagnostic token acc ≈ 0.8575
```

说明成功模型依赖 local+remote 的混合路径，不能粗暴只走 remote。

### cross-bias +1 是什么

cross-bias 不删除 local 边，只给 rollout 阶段的 remote/cross edges 一个统一 logit bias `+1.0`。它不指定哪条远程边是正确答案，只是鼓励远程传播。

结果：

| 变体 | train token | train seq | final token | final seq |
|---|---:|---:|---:|---:|
| cross-bias +1 topk16 | `1.0` | `1.0` | `0.9999931640625` | `0.993` |
| cross-bias +1 topk8 | `1.0` | `1.0` | `0.9999951171875` | `0.995` |

它没有超过 clean rollout memory best。

### 作弊嫌疑

低到中。cross-bias 是通用远程边 prior，不知道 copy offset；但 copy 本来就是远距离任务，所以它对任务有天然适配。能写，但不能包装成“纯 random baseline”。

## 14. 路线 K：relative-bias-only / finetune continuation

### 改进到底是什么

relative bias 让 attention logit 依赖相对位移，而不是让模型分别记住每个绝对位置对。corrected copy 的真实关系是：

```text
marker_position_i - source_position_i = 1024
```

所有 copy pair 共享同一个相对距离，因此 relative bias 是非常有用的通用归纳偏置。

### 结果

relative-bias-only 小样本 gate 能过，但 plain full train 不足以稳定达到 final 近 1。真正有效的是 relative bias 和 rollout memory 结合。

此外，一些早期 finetune continuation 不带 history readout，曾产生此前 best clean `seq=0.997`。最新 weighted multiscale 版本同样不依赖 history readout，并把 final/test 提到 `seq=0.998`；这说明核心贡献仍是 rollout memory + relative bias + top-k/multiscale propagation，而不是 history readout。

### 作弊嫌疑

低到中。relative bias 在 ref 中本来就是 RoPE/QK 之外合理的位置归纳偏置，但在固定 offset copy 上很强；报告必须承认它是强 bias。

## 15. 结果总表

| 路线 | 最好 train | 最好 final/test | 是否建议主报 | 作弊嫌疑 |
|---|---|---|---|---|
| staged copy route | 未作为 clean 评估 | 未作为 clean 评估 | 否 | 高 |
| value RoPE | gate 能过 | 未作为 clean 评估 | 否 | 高 |
| token rollout | gate 能过 | 未作为 clean 评估 | 否 | 高 |
| learned edge bias/transport | 未进入最终候选 | 未进入最终候选 | 否 | 中高 |
| rollout memory + relative bias + topk8 | train diagnostic 1/1；full train `0.9999999/0.9999` | `0.9999970703125/0.997` | 旧首选 | 低到中 |
| weighted multiscale rollout `[1,2]` + relative bias + topk8 | full train `1.0/1.0` | `0.999998046875/0.998` | 是，首选 | 低到中 |
| unweighted multiscale `[1,1,2]` | full train `0.9999994140625/0.9994` | 未做 final | 否，train 已不全对 | 低到中 |
| rollout memory + logit history topk8 | full train 1/1 | `0.9999951171875/0.995` | 可作为补充 | 低到中 |
| rollout memory + logit history topk16 | full train 1/1 | `0.999994140625/0.994` | 可作为补充 | 低到中 |
| confidence history topk8 | train summary 1/1 | `0.9999931640625/0.993` | 不推荐 | 低到中 |
| rollout steps=2 | train 明显退化 | 未做主 final | 否 | 低 |
| margin continuation | train 1/1 | `0.9999951171875/0.995` | 辅助，不主报 | 低到中 |
| learned rollout gate | train 1/1 | `0.9999931640625/0.993` | 辅助 | 低到中 |
| cross-bias +1 | train 1/1 | topk8 `0.9999951171875/0.995` | 辅助 | 低到中 |

## 16. 我建议最终怎么写

推荐表述：

> 在 corrected copy 设置下，plain random sparse attention 虽然 source-to-marker 两跳结构可达，但普通 hidden 传播不稳定。加入通用 attention-rollout memory 后，模型沿已有 random sparse edges 维护独立 memory state，使多跳信息能跨层稳定传递。进一步把 rollout propagation 改成 weighted multiscale：90% 一跳传播 + 10% 两跳传播，让模型保留稳定一跳主干，同时获得少量两跳 lookahead。在 `layers=8`、actual density≈0.1、noncausal、T=2048 下，best clean random result 达到 full-train `1.0/1.0`，final/test token acc `0.999998046875`、seq acc `0.998`。该路线不使用 copy-specific route、token side-channel、value-position side-channel、learned per-edge transport 或 history-logit readout；但它不是原始 random baseline，而是 random sparse attention 的通用多跳 memory 增强版本。

不推荐表述：

> random 通过 8-hop staged route 学会 copy。

这个说法会被质疑为把 copy offset 写进架构。

## 17. 当前最强结果的复核路径

当前最强 clean final/test：

```text
remote run dir:
/home/huiwei/ysx/zigzag_attention_heads_trial/outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5_random_layerwise_density10_multiscale12_w0p9_0p1_lr3e5_resume_from_nohistory_final/random_regular/seed0

config:
configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_rollout_multiscale12_w0p9_0p1_lazy0p5_memoryscale2_relative_bias_scale2p0_topk8_density10_lr3e5_resume_from_nohistory_final_b4a2.json

final eval:
final_eval.json
final_eval.csv

train eval:
train_eval_full.json
summary.json

checkpoint used by final/test:
checkpoints/train_final_step1000.pt
sha256 = 2c5e8b89a9030cfe7d2dc2c6bd40af506bb09b20c84f5288ffd154dcc9a3f744
```

关键合规字段：

```text
random_multihop_copy_route = false
random_token_rollout_memory = false
random_value_position_encoding = none
random_history_output_logits = false
random_learned_attention_edge_bias = false
random_learned_edge_memory_transport_mode = null
random_rollout_memory_multiscale_steps = [1, 2]
random_rollout_memory_multiscale_weights = [0.9, 0.1]
position_parameter_count = 0
padding_positions = 0
target_positions = 1024..2047
test_read_during_training = false
```

## 18. 还没解决的风险

1. `seq=0.998` 仍不是 `1.0`，strict dense-level exact-match 没完全达到；但如果“接近 1 就可以”，现在已经是很强结果。
2. 历史上做过多次 final/test 验证，因此写报告时要区分“开发过程中探索过 test 指标”和“某个 checkpoint 是否用 test 选出来”。最新 `seq=0.998` 模型是 train-final checkpoint，不是从多个同 run checkpoint 里按 test 挑的。
3. 目前主要是 seed0/mask0，缺多 seed、多 random mask 验证。
4. relative bias 对固定 offset copy 很强，虽然不算作弊，但要明确承认这是强归纳偏置。
5. 远端结果里 `git_commit=unknown`、`git_dirty=true`，复现性报告需要额外记录代码快照或 commit/tag。

## 19. 如果后续还要冲 final seq=1.0

建议继续沿 clean 路线，而不是回到 staged route：

1. 只用 train/full-train low-margin 分析选方向，不再用 test 错例调结构；
2. 保留 weighted multiscale rollout memory + relative bias + topk8 作为基线；
3. 做温和的 checkpoint/EMA 或 train-only margin schedule，但 final/test 只读一次；
4. 多 seed / 多 mask 验证 `seq≈0.998` 是否稳定；
5. 如果新增机制，优先保持“只沿 random 可见边传 hidden/memory”，避免 token side-channel、value RoPE、per-edge fixed-position memorization。
