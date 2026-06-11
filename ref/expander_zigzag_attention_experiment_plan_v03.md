# Expander / Zig-Zag Sparse Attention 实验执行手册

_从理论基础、方法定义到可复现实验与工程落地_
_版本：v0.3；日期：2026-06-10_

> **说明：** 使用说明：本文档假设读者没有 expander、zig-zag product 或稀疏 attention 的前置知识。阅读顺序建议为：第 2-5 节理解理论和方法，第 6-9 节确定实验，第 10-13 节实现和记录，第 14 节处理风险和负结果。

# 目录
- 1. 文档定位、修订说明与一页路线图
  - 1.1 文档目标
  - 1.2 v0.3 相对上一版的修正
  - 1.3 一页版实验路线图
- 2. 从零开始：attention mask 的图视角
  - 2.1 普通全注意力为什么是 N×N
  - 2.2 局部 block、B=mn 与 q 的含义
  - 2.3 mask 与 causal mask 的关系
- 3. Expander 图理论基础
  - 3.1 稀疏性：常数度给出线性边数
  - 3.2 扩张性：稀疏但没有窄瓶颈
  - 3.3 谱扩张、混合与多层信息传播
  - 3.4 理论概念如何转化为实验指标
- 4. Zig-Zag Product：G、H、Rot 与连边
  - 4.1 大图 G 与小图 H
  - 4.2 旋转映射 Rot_G 与 Rot_H
  - 4.3 zig、jump、zag 三步
  - 4.4 G/H 具体生成规范
- 5. 当前方法：local complete block + zig-zag cross edges
  - 5.1 最终 mask 的组成
  - 5.2 复杂度公式与节省比例
  - 5.3 常见误解和变体边界
- 6. 实验假设、成功标准与公平性约束
  - 6.1 核心实验假设
  - 6.2 same-budget 原则
  - 6.3 raw K 与 effective K
  - 6.4 负结果诊断原则
- 7. 实验任务与评测标准
  - 7.1 任务分层
  - 7.2 质量指标
  - 7.3 图结构指标
  - 7.4 工程性能指标
- 8. 最小可执行实验 MVP
  - 8.1 第一轮固定配置
  - 8.2 第一轮对比方法
  - 8.3 第一轮验收标准
- 9. 完整实验矩阵与参数扫描策略
  - 9.1 G/H 图结构消融
  - 9.2 B、d、N、L 参数消融
  - 9.3 避免组合爆炸的冻结策略
- 10. 工程实现路线
  - 10.1 v0 dense mask 原型
  - 10.2 v1 neighbor-list attention
  - 10.3 v2 local dense + zig-zag sparse
  - 10.4 v3 block-sparse / Triton / xFormers
  - 10.5 核心代码接口与单元测试
- 11. 性能测试、复现与远程服务器规范
  - 11.1 性能测试协议
  - 11.2 随机性与复现
  - 11.3 远程服务器使用规范
  - 11.4 本地开发与远程测试流程
  - 11.5 环境锁定、日志和 checkpoint
- 12. 可复用代码与已有研究
  - 12.1 Long Range Arena
  - 12.2 BigBird
  - 12.3 xFormers
  - 12.4 Exphormer / GraphGPS
  - 12.5 PyTorch 官方工具链
- 13. 结果表格模板、画图与分析
  - 13.1 主结果表
  - 13.2 图结构相关性分析
  - 13.3 性能扩展曲线
- 14. 风险清单、决策树与里程碑
  - 14.1 风险清单
  - 14.2 负结果诊断决策树
  - 14.3 里程碑与交付物
- 15. 参考资料与资料来源
- 16. 文档自检与剩余边界

# 1. 文档定位、修订说明与一页路线图

## 1.1 文档目标
本文档的目标是把“固定 m×n token 完全图 + zig-zag product 跨块稀疏边”的想法整理为一套可以直接执行的实验方案。文档既解释理论基础，也规定工程实现、评测标准、服务器使用方式和负结果诊断流程。
核心研究问题是：在相同 attention pair budget 下，local complete block + zig-zag / expander 跨块边，是否能优于 local-only、random sparse、BigBird-style sparse 等基线，并且能否在真实 GPU 上兑现接近线性的显存和速度。

## 1.2 v0.3 相对上一版的修正

| 类别 | 已修正或新增内容 |
| --- | --- |
| 结构修正 | 服务器规范已移动到工程与复现章节内部，不再错误地出现在目录之后、正文之前。 |
| 可执行性 | 新增 MVP 固定配置、验收标准、图生成规则、性能测试协议和单元测试清单。 |
| 公平性 | 新增 raw K / effective K、重复边、self-loop、causal 过滤后的实际邻居数记录规则。 |
| 工程规范 | 新增本地开发-远程测试流程、GPU 选择规则、环境锁定、checkpoint resume、日志目录和 git commit 记录。 |
| 资料补充 | 补充 LRA、BigBird、xFormers、Exphormer、PyTorch CUDA memory、PyTorch reproducibility、tmux/screen、conda 等来源。 |
| 自检 | 新增文档自检与剩余边界章节，明确哪些问题已解决，哪些必须通过真实实验回答。 |

## 1.3 一页版实验路线图

| 阶段 | 目标 | 输入 | 输出 / 验收标准 |
| --- | --- | --- | --- |
| 0. 理论与接口冻结 | 统一符号、mask 定义、Rot_G/Rot_H 接口。 | 两份用户文档、本文档第 2-5 节。 | 完成接口文档；能手算一个小图例子。 |
| 1. Dense mask 原型 | 先做正确，不追求省显存。 | N≤1024；B=8/16；小型 G/H。 | dense mask 与手写邻接结果一致；无 NaN。 |
| 2. Neighbor-list attention | 实现 O(NK) 逻辑复杂度。 | neighbors[N,K]；K=B+d²。 | 输出与 dense mask 版本在小 N 上数值一致。 |
| 3. Synthetic task | 验证跨块边是否有建模价值。 | Associative Recall / Copy。 | zig-zag 优于 local-only，或给出明确负结果诊断。 |
| 4. LRA 单任务 | 引入标准长序列 benchmark。 | ListOps 或 Text；固定模型规模。 | 完成 full/local/random/zig-zag 对比。 |
| 5. 图结构和参数消融 | 分析 G、H、B、d、N、L。 | 冻结策略逐步扫描。 | 得到质量-图指标-成本三方关系。 |
| 6. 工程优化 | 兑现实际显存与速度。 | profile 数据；local/cross 分解。 | peak memory 近似线性；速度瓶颈可解释。 |

# 2. 从零开始：attention mask 的图视角

## 2.1 普通全注意力为什么是 N×N
给定长度为 N 的 token 序列，标准 self-attention 会为每个 query token 和每个 key token 计算一个注意力分数，因此 attention score 矩阵是 N×N。若不考虑投影层和 FFN，仅 score 与 value mixing 的主要规模可以粗略看成 O(N² d_model)。
mask 的作用是规定哪些 query-key pair 允许计算。图视角下，token 是顶点，允许 attention 的 pair 是有向边。全注意力对应 token 完全图；稀疏 attention 对应只保留部分边。

## 2.2 局部 block、B=mn 与 q 的含义
本文方法把 N 个 token 分成 q 个局部 block，每个 block 有 B 个 token。原始文档把 B 写作 B=mn，是因为每个局部 block 可以被看成 m×n 的二维小窗口；但对纯文本序列，完全可以令 m=1、n=B。关键复杂度由 B 决定，而不是 m、n 分别决定。

```text
N = T = qB = qmn
q = N / B
一个 token 可以写成 (v, i)，其中 v 是 block id，i 是 block 内 token/port id。
```

## 2.3 mask 与 causal mask 的关系
本文主实验默认 non-causal / bidirectional attention。此时 block 内 complete graph 可以表示为 B×B 全 1 的局部 mask。若用于 GPT-style causal LM，则最终 mask 必须取 graph mask 与 lower-triangular causal mask 的交集。

```text
M_final[i, j] = M_graph[i, j] AND (j <= i)
```

> **说明：** 阶段性决策：第一阶段不优先做 causal LM。causal 版本会改变 effective K，尤其跨 block 边会被未来位置过滤，因此必须另设公平性规则。

# 3. Expander 图理论基础

## 3.1 稀疏性：常数度给出线性边数
一个 n 点 d-正则无向图中，每个顶点度数为 d。根据握手引理，无向边数为 dn/2。如果 d 是常数，则边数是 O(n)，远小于完全图的 n(n-1)/2。对 attention 来说，这意味着每个 token 只和常数个远程 token 通信，理论上可以把全局部分从 O(N²) 降到 O(N)。

## 3.2 扩张性：稀疏但没有窄瓶颈
Expander 的直觉是：虽然边数很少，但任意不太大的点集都能连向相当多的外部点。它不像链式或窗口图那样有长距离瓶颈。对 Transformer 来说，expander mask 的价值不是一层内复原所有 N² 个 pair，而是在多层网络中提供低直径、快速混合、无窄瓶颈的全局通信通道。

## 3.3 谱扩张、混合与多层信息传播
在 d-正则图中，归一化随机游走矩阵 P=A/d 的非平凡特征值上界 λ 控制混合速度。若 λ<1，则 P^L 会以 λ^L 的速度接近全局平均算子。这给出一个实验直觉：如果跨块图具有较好的谱间隙，多层 attention 的可达性和信息扩散会更快。

## 3.4 理论概念如何转化为实验指标

| 理论概念 | 实验中记录什么 | 解释作用 |
| --- | --- | --- |
| 度数 | raw K、effective K、每 token 可见 key 数。 | 决定 attention pair budget。 |
| 边数 | N×K 或 qB(B+d²)。 | 决定理论复杂度。 |
| 连通性 | connected components。 | 防止 mask 把 token 分裂成孤岛。 |
| 直径 | approximate diameter。 | 衡量多层通信路径长度。 |
| 谱间隙 | 归一化邻接矩阵的非平凡特征值。 | 解释混合速度和跨区域通信能力。 |
| L 层覆盖率 | L 次邻接可达 token 比例。 | 直接对应深层 Transformer 的信息传播范围。 |

# 4. Zig-Zag Product：G、H、Rot 与连边

## 4.1 大图 G 与小图 H
标准 zig-zag product 使用两个图：大图 G 和小图 H。G 是 Δ-正则图，H 有 Δ 个顶点并且是 d-正则图。product 后的顶点集合是 V(G)×[Δ]，度数是 d²。

| 对象 | 数学含义 | attention 语义 | 顶点数 | 度数 | 无向边数 |
| --- | --- | --- | --- | --- | --- |
| G | Δ-正则大图 | block-level 图；一个顶点是一个 block | q | Δ=B | qB/2 |
| H | d-正则小图 | port-level 图；一个顶点是一个端口编号 | Δ=B | d | Bd/2 |
| G ∘z H | zig-zag product 图 | token-level 跨块稀疏图 | qB=N | d² | qBd²/2 |

## 4.2 旋转映射 Rot_G 与 Rot_H
只说“G 每个点度数为 B”并不能决定边怎么连。真正的连边规则是旋转映射 Rot_G(v,i)=(w,j)：从 block v 的第 i 个端口出发，会跳到 block w，并落在 w 的第 j 个端口。无向图要求反向一致：Rot_G(w,j)=(v,i)。

```text
Rot_G(v, i) = (w, j)
含义：block v 的 port i 与 block w 的 port j 配对。

Rot_H(i, a) = (i_prime, a_reverse)
含义：在端口图 H 中，从端口 i 沿第 a 条小图边走到 i_prime。
```

## 4.3 zig、jump、zag 三步
从 product 图顶点 (v,i) 出发，选择两个小图边标号 a,b∈[d]。先在 H 中从 i 走到 i′，再用 Rot_G(v,i′) 跳到另一个 block 的端口 j′，最后在 H 中从 j′ 走到 j。最终得到跨块边 (v,i) -> (w,j)。

```text
i_prime = H_neighbor(i, a)          # zig
w, j_prime = Rot_G(v, i_prime)    # jump
j = H_neighbor(j_prime, b)        # zag
add_edge((v, i), (w, j))
```

## 4.4 G/H 具体生成规范

| 图结构 | 生成规则 | 实验用途 |
| --- | --- | --- |
| Cyclic G | B 为偶数；给定 offsets=[s1,...,s_{B/2}]；port 2k 连 +s_k，port 2k+1 连 -s_k，并设置反向端口。 | 最简单确定性结构，便于调试 Rot_G。 |
| Random permutation G | B=2r；采样 r 个置换 π_k；每个置换贡献一对正反向端口。 | 接近随机正则图，生成快，适合作主力 G。 |
| Random regular G | 直接采样 B-正则图或多重图，并为 incident edges 分配端口编号。 | 更标准的 random regular baseline。 |
| Explicit expander G | 使用已知显式 expander 或近 Ramanujan 构造；保留邻居 oracle。 | 理论性质强，但实现成本高。 |
| Cycle H | H 是 B 个端口上的 cycle，d=2。 | 最小可用小图，调试简单。 |
| Random d-regular H | 在 B 个端口上采样 d-正则图。 | 主力 H 候选。 |
| Complete H=K_B | d=B-1。 | 反例；验证复杂度膨胀，不推荐作为主方法。 |

> **说明：** 强制记录：所有 G/H 生成器必须记录 self-loop 数、multi-edge 数、重复 zig-zag neighbor 数、raw K、effective K 和随机种子。

# 5. 当前方法：local complete block + zig-zag cross edges

## 5.1 最终 mask 的组成
最终 attention mask 分为两部分：block 内 local complete graph 与跨 block zig-zag sparse edges。local complete 保证每个局部窗口内 token 充分交互；zig-zag 边负责跨 block 通信。

```text
E_mask = E_local_complete ∪ E_zigzag
每个 token 的理论可见 key 数：K = B + d²（non-causal、未去重时）。
```

## 5.2 复杂度公式与节省比例
设 N=T=qB，且 Δ=B=mn。block 内 dense pair 数为 qB²；zig-zag cross pair 数为 qBd²；总 pair 数为 qB(B+d²)=N(B+d²)。因此 mask 相关 attention FLOPs 约为 O(N(B+d²)d_model)。当 B 和 d 固定时，它对总 token 数 N 近似线性。

| 方法 | 每 token 可见 key 数 | 总 pair 数 | 相对 full attention |
| --- | --- | --- | --- |
| Full attention | N | N² | 质量上界；复杂度最高。 |
| Local-only | B | NB | 无跨块通信；作为必要下界。 |
| Local + zig-zag | B+d² | N(B+d²) | 目标方法。 |
| Local + random | B+d² | N(B+d²) | same-budget 随机基线。 |

## 5.3 常见误解和变体边界

| 问题 | 澄清 |
| --- | --- |
| H 是否就是 block 内 dense attention？ | 不是。block 内 dense 是最终 mask 的一部分；H 是 zig-zag product 的端口扰动小图。 |
| 知道 B 是否就知道 G 怎么连？ | 不是。B 只规定每个 block 有多少端口；必须额外给出 Rot_G。 |
| 是否任意两个 block 都做 zig-zag？ | 主方案不是。G 是 B-正则 block-level 图，每个 block 只通过 B 个端口连到有限邻居。 |
| 可以用 H=K_B 吗？ | 可以作为反例，但 d=B-1 会使 zig-zag 项膨胀到 qB³，不推荐。 |

# 6. 实验假设、成功标准与公平性约束

## 6.1 核心实验假设

| 编号 | 假设 | 验证方式 |
| --- | --- | --- |
| H1 | local + zig-zag 在需要长程通信的任务上优于 local-only。 | Synthetic task 和 LRA 单任务上比较 validation score。 |
| H2 | 在 same-budget 条件下，expander/zig-zag 结构比简单 random 更稳定或更可解释。 | 多 seed 均值/方差、图指标相关性。 |
| H3 | 当 N 增大时，neighbor-list 或 block-sparse 实现的 peak memory 近似线性。 | N 扫描：1K/2K/4K/8K/16K。 |
| H4 | 真实速度收益取决于 kernel layout，而不仅取决于 pair 数。 | 拆分 local/cross/QKV/FFN profile。 |

## 6.2 same-budget 原则
所有稀疏方法必须尽量保持相同的每 token 可见 key 数。例如，local + random baseline 应与 local + zig-zag 使用同样的理论 K=B+d²。若某方法有 global tokens 或边数不同，必须单独报告 pair budget，不能直接和 same-budget 结果混在一起。

## 6.3 raw K 与 effective K

| 指标 | 定义 | 为什么必须记录 |
| --- | --- | --- |
| raw K | 理论生成的邻居数，例如 B+d²。 | 反映方法设计预算。 |
| effective K | 去重、去 self-loop、去非法边、causal 过滤后的实际邻居数。 | 反映真实计算量和公平性。 |
| duplicate rate | 重复邻居数 / raw K。 | 检测 zig-zag 或 random 采样是否浪费边。 |
| future-filter rate | causal 场景中被过滤的边比例。 | 解释 causal 任务中理论 K 与实际 K 的差异。 |

## 6.4 负结果诊断原则
负结果不应直接解释为方法失败。若 zig-zag 不如 random，需要检查 G/H 谱间隙、层数、effective K、任务是否真正需要跨块通信、同一 G 是否在所有层复用导致覆盖不足。若 memory 降但速度不升，需要检查 gather/scatter 是否成为瓶颈。

# 7. 实验任务与评测标准

## 7.1 任务分层

| 阶段 | 任务 | 目的 | 优先级 |
| --- | --- | --- | --- |
| Synthetic | Copy / Delayed Copy / Associative Recall / chunk parity。 | 便宜、可控，验证远程通信。 | 最高 |
| LRA 单任务 | ListOps 或 Text。 | 引入标准 efficient Transformer 长序列评测。 | 高 |
| LRA 扩展 | Retrieval / Image / Pathfinder。 | 跨模态和更复杂依赖。 | 中 |
| 图任务 | Exphormer / GraphGPS 相关 benchmark。 | 扩展到 graph Transformer 语境。 | 中低 |
| Causal LM | 长上下文语言模型。 | 真实应用价值高但实现复杂。 | 后续 |

> **说明：** 任务选择边界：第一阶段不优先做大规模语言模型和高分辨率视觉任务，因为训练成本和 pipeline 复杂度会掩盖 mask 结构本身的问题。

## 7.2 质量指标

| 任务类型 | 主指标 | 辅助指标 |
| --- | --- | --- |
| 分类 | accuracy / macro-F1 | validation loss、收敛 step、seed 方差。 |
| 检索 / copy | exact match | 长度外推曲线、错误位置分布。 |
| 语言建模 | perplexity / validation loss | tokens/sec、context length 扩展。 |
| 图任务 | MAE / RMSE / accuracy | 节点数扩展、显存。 |

## 7.3 图结构指标
对每个 G/H 配置记录 spectral gap、approximate diameter、connected components、L-layer coverage、effective K 和 duplicate rate。后续分析要画出这些指标与 validation score 的相关性。

## 7.4 工程性能指标

| 指标 | 记录方式 |
| --- | --- |
| peak GPU memory | torch.cuda.reset_peak_memory_stats() 后运行，再读 max_memory_allocated / max_memory_reserved。 |
| tokens/sec | 正式计时区间内 processed_tokens / elapsed_time。 |
| forward time | 仅前向。 |
| forward+backward time | 训练 step 总耗时。 |
| attention time breakdown | local attention、cross attention、QKV/output、FFN 分开记录。 |
| mask construction time | 构造 dense mask、edge_index、neighbors 的时间；区分预计算与每层重采样。 |

# 8. 最小可执行实验 MVP

## 8.1 第一轮固定配置

| 项目 | 固定值 |
| --- | --- |
| 任务 | Associative Recall；可加 Delayed Copy 作为 sanity check。 |
| N | 1024、2048、4096。 |
| B | 16。 |
| d | 2、3、4。 |
| 层数 L | 4。 |
| d_model | 256。 |
| heads | 4。 |
| d_ff | 1024。 |
| optimizer | AdamW。 |
| learning rate | 3e-4 起步，必要时扫 1e-4 / 3e-4 / 1e-3。 |
| seeds | 0、1、2。 |
| batch | 优先固定 tokens/batch；显存不足时自动降 batch 并记录。 |

## 8.2 第一轮对比方法

| 方法 | 说明 | 预算 |
| --- | --- | --- |
| Full attention | 小 N 上的质量上界。 | K=N。 |
| Local-only | 只保留 block 内 complete graph。 | K=B。 |
| Local + random | 每个 token 随机采样 d² 个跨块 token。 | K=B+d²。 |
| Local + cyclic G + cycle H | 最简单 zig-zag。 | K≈B+d²。 |
| Local + random permutation G + random H | 主力 zig-zag 候选。 | K≈B+d²。 |

## 8.3 第一轮验收标准

| 验收项 | 通过标准 |
| --- | --- |
| mask 正确性 | 小 N 下 dense mask 与 neighbor-list attention 输出数值一致。 |
| 建模价值 | zig-zag 至少在一个 synthetic 任务上明显优于 local-only，或有清晰负结果解释。 |
| 公平性 | 所有稀疏方法报告 raw K、effective K、duplicate rate。 |
| 性能记录 | 所有方法记录 peak memory、tokens/sec、attention time。 |
| 可复现 | 每个结果包含 seed、git commit、环境快照、GPU id。 |

# 9. 完整实验矩阵与参数扫描策略

## 9.1 G/H 图结构消融

| 实验组 | G | H | 目的 |
| --- | --- | --- | --- |
| A | cyclic offsets | cycle | 最简单可解释结构。 |
| B | random permutation | cycle | 只随机化 block-level。 |
| C | random permutation | random d-regular | 主力候选。 |
| D | random regular | random d-regular | 标准随机正则对照。 |
| E | explicit expander | small expander | 理论性质强的构造。 |
| F | same-budget random edges | 无 H | 非结构化随机 baseline。 |
| G | random permutation | complete K_B | 复杂度膨胀反例。 |

## 9.2 B、d、N、L 参数消融

| 参数 | 候选值 | 观察目标 |
| --- | --- | --- |
| B | 8、16、32、64 | 局部 dense 常数与性能之间的 trade-off。 |
| d | 2、3、4、6、8 | 跨块边数量 d² 与效果之间的 trade-off。 |
| N | 1K、2K、4K、8K、16K | 扩展性曲线。 |
| L | 2、4、6、8、12 | 多层可达性和性能。 |
| layer-wise graph | 固定 G / 每层重采样 G | 覆盖率和稳定性。 |

## 9.3 避免组合爆炸的冻结策略

```text
第一轮：固定 B=16, L=4, N=4096，扫 d。
第二轮：固定最优 d，扫 B。
第三轮：固定 B,d，扫 N。
第四轮：固定 B,d,N，扫 G/H。
第五轮：只对前两名配置做多 seed 和完整 LRA。
```

# 10. 工程实现路线

## 10.1 v0 dense mask 原型
v0 的目标是正确性，不是速度。直接构造 N×N boolean mask，用 PyTorch dense attention 计算。该版本用于调试 mask、Rot_G、Rot_H、causal 过滤和小 N 数值对齐。

```text
scores = Q @ K.transpose(-1, -2) / sqrt(head_dim)
scores = scores.masked_fill(~mask, -inf)
out = softmax(scores, dim=-1) @ V
```

## 10.2 v1 neighbor-list attention
v1 构造 neighbors[N,K]，每个 token 只 gather 允许的 K/V。理论计算量为 O(NK)。这是验证线性复杂度的第一版实现。

```text
neighbors: LongTensor [N, K]
K_gathered: [batch, heads, N, K, head_dim]
V_gathered: [batch, heads, N, K, head_dim]
scores: [batch, heads, N, K]
```

## 10.3 v2 local dense + zig-zag sparse
v2 将 block 内 local dense 和跨块 zig-zag 分开实现：local 部分 reshape 为 [batch*q, B, dim] 调用 dense/flash attention；cross 部分使用 gather。注意不能把两个 softmax 后的输出直接相加，必须用 log-sum-exp 合并 local 与 cross 的 numerator / denominator，才能等价于在 union mask 上做一次 softmax。

## 10.4 v3 block-sparse / Triton / xFormers
v3 的目标是工程落地。优先复用 xFormers 或 PyTorch SDPA/FlashAttention 处理 local block；zig-zag 边若能按目标 block 分组，则尝试 block-sparse layout。只有在 v1/v2 的质量结果明确后，才投入 Triton 或自定义 kernel。

## 10.5 核心代码接口与单元测试

```python
class RotG:
    def __call__(self, v: int, port: int) -> tuple[int, int]: ...

class RotH:
    def neighbors(self, port: int) -> list[int]: ...

class SparsePattern:
    def build_neighbors(self, N: int, B: int, device) -> Tensor: ...
```

| 单元测试 | 通过标准 |
| --- | --- |
| Rot_G 反向一致性 | Rot_G(w,j)==(v,i)。 |
| H 度数检查 | 每个端口恰有 d 个邻居；记录重复/自环。 |
| edge count 检查 | raw zig-zag 数接近 N d²。 |
| dense vs neighbor 对齐 | 小 N 上 max_abs_error < 1e-5。 |
| causal 过滤检查 | 若启用 causal，保证没有 j>i 的 key。 |
| NaN 检查 | 所有 query 至少有一个有效 key；softmax 无 NaN。 |

# 11. 性能测试、复现与远程服务器规范

## 11.1 性能测试协议

```python
for _ in range(20):
    run_one_step()       # warmup
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
for _ in range(100):
    run_one_step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_alloc = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

| 规则 | 说明 |
| --- | --- |
| warmup | 每个配置至少 20 step warmup。 |
| 正式计时 | 至少 100 step；报告 mean/std。 |
| 同步 | 计时前后必须 torch.cuda.synchronize()。 |
| 显存 | 使用 reset_peak_memory_stats 与 max_memory_allocated / max_memory_reserved。 |
| batch 规则 | 优先固定 tokens/batch；若变更 batch，必须记录。 |
| profile 拆分 | local、cross、QKV/output、FFN、mask construction 分开测。 |

## 11.2 随机性与复现
每次实验必须记录 Python、NumPy、PyTorch、CUDA、cuDNN、GPU 型号、git commit、conda 环境、随机种子。若需要严格可复现，可启用 PyTorch deterministic 选项，但要记录其可能降低性能。

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 可选：torch.use_deterministic_algorithms(True)
```

## 11.3 远程服务器使用规范
远程服务器通过 ssh huiwei 访问。这是一台 A100×4 公用服务器。由于是公用资源，如果某个 GPU 已经被占用，就不要使用该卡；不要默认占用所有 GPU；单卡实验必须只暴露一张空闲卡。远程工作目录限定为 ~/ysx/，默认使用 ysx_base conda 环境，不要动其他人的目录、环境、数据、日志或 checkpoint。必要时可以在 ~/ysx/ 下新建独立环境。

```bash
ssh huiwei
cd ~/ysx/
nvidia-smi
conda activate ysx_base
CUDA_VISIBLE_DEVICES=<空闲卡编号> python train.py ...
```

> **说明：** 硬性规则：不要在未确认空闲的情况下使用 GPU；不要使用别人的目录；不要修改 base 环境或他人 conda 环境；DDP/多卡实验只有在确认 4 张卡都空闲且必要时才运行。

## 11.4 本地开发与远程测试流程
访问服务器的网络不稳定，因此代码开发、重构、单元测试优先在本地完成。远程服务器只用于 GPU smoke test、性能测试和正式训练。远程修改只允许 hotfix；任何 hotfix 必须同步回本地并提交 git。

```text
~/ysx/zigzag_attention/
  code/
  data/
  logs/
  checkpoints/
  outputs/
  envs/
```

```bash
# 建议传输方式
rsync -av --exclude .git --exclude __pycache__ ./zigzag_attention/ huiwei:~/ysx/zigzag_attention/code/

# 网络不稳定时
tmux new -s zigzag
# detach: Ctrl-b, d
tmux attach -t zigzag
```

## 11.5 环境锁定、日志和 checkpoint

```bash
conda activate ysx_base
conda env export > env_ysx_base_snapshot.yaml
pip freeze > requirements_snapshot.txt

# 新环境示例
conda create -n ysx_zigzag python=3.10
conda activate ysx_zigzag
pip install -r requirements.txt
conda env export > environment.yaml
```

| 对象 | 规则 |
| --- | --- |
| checkpoint | 训练脚本必须支持 --resume；每 N steps 保存；记录 latest checkpoint。 |
| 日志 | 每个 run 保存 config.yaml、metrics.jsonl、stdout.log、stderr.log。 |
| 结果 | 输出表格和图放 outputs/，不要散落在代码目录。 |
| 环境 | 每个正式 run 保存 environment.yaml 或 requirements_snapshot.txt。 |
| 代码版本 | 每个正式 run 记录 git commit hash；dirty working tree 的实验标记为不可复现。 |

# 12. 可复用代码与已有研究

## 12.1 Long Range Arena
LRA 是系统评测 efficient Transformer 的长序列 benchmark，适合用作第二阶段标准任务。官方实现基于 JAX/Flax，若主代码用 PyTorch，可以参考 PyTorch/HuggingFace 风格的复现实现，但要核对数据处理和任务设置。

## 12.2 BigBird
BigBird 是 sparse attention baseline 的重要参考，结构上包含 local、random、global 三类连接。它适合作为“local + random + global”的强基线，但注意 BigBird 的 block sparse 设计并不等同于本文 zig-zag product。

## 12.3 xFormers
xFormers 提供 PyTorch 生态的 optimized Transformer components，包括 memory-efficient attention、sparse attention、block-sparse attention 和 fused softmax。本文建议先用 xFormers 做工程 baseline，再评估是否需要自写 Triton。

## 12.4 Exphormer / GraphGPS
Exphormer 在 GraphGPS 框架中组合 actual graph edges、expander graphs 和 universal connectors。它适合复用 expander graph 生成、图任务训练框架和稀疏图注意力思路，但它的语境是 graph Transformer，不是本文的 sequence block zig-zag attention，因此不能直接作为最终实现。

## 12.5 PyTorch 官方工具链
性能和复现相关规范应优先参考 PyTorch 官方文档。显存统计使用 torch.cuda.reset_peak_memory_stats、max_memory_allocated、max_memory_reserved；随机性控制参考 PyTorch reproducibility note。

# 13. 结果表格模板、画图与分析

## 13.1 主结果表

| 方法 | K/effective K | Accuracy / Loss | Peak Mem | Tokens/s | Attention Time | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| Full | N / N |  |  |  |  | 小 N 质量上界。 |
| Local-only | B / ? |  |  |  |  | 无跨块通信。 |
| Local+Random | B+d² / ? |  |  |  |  | same-budget 随机。 |
| Local+ZigZag | B+d² / ? |  |  |  |  | 主方法。 |
| BigBird-style | matched / ? |  |  |  |  | local+random+global。 |

## 13.2 图结构相关性分析
每个 G/H 配置保存 graph_metrics.csv，其中至少包含 spectral_gap、diameter、L_layer_coverage、effective_K、duplicate_rate、validation_score。分析时画出 spectral_gap vs score、coverage vs score、effective_K vs score。

## 13.3 性能扩展曲线
必须画 N 扫描曲线：x 轴为 N，y 轴分别为 peak memory、tokens/sec、attention time、effective pair count。曲线至少包含 dense、local-only、local+random、local+zig-zag。

# 14. 风险清单、决策树与里程碑

## 14.1 风险清单

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 理论线性但 GPU 不快 | memory 降低，tokens/sec 不升。 | profile gather/scatter；优化 layout；尝试 block-sparse。 |
| zig-zag 不如 random | same-budget random 更强。 | 检查 G/H 指标、effective K、层间重采样。 |
| local block 太小 | accuracy 低于 full 很多。 | 扫 B；增加 global token；增加层数。 |
| 任务不需要远程通信 | local-only 已经很强。 | 换 synthetic 或更长序列任务。 |
| 远程训练中断 | ssh 断开或进程丢失。 | 使用 tmux；开启 checkpoint resume。 |

## 14.2 负结果诊断决策树

```text
若 zig-zag < local-only:
    检查跨块边是否真的加入；检查 L-layer coverage；换需要远程依赖的任务。

若 zig-zag < random:
    检查 effective K 是否相同；检查 G/H spectral gap；尝试 layer-wise resampling。

若 memory 降但速度不升:
    拆分 local/cross/QKV/FFN time；检查 gather/scatter；尝试按 block 排序边。

若结果方差很大:
    增加 seed；固定数据顺序；记录 graph seed 与 model seed。
```

## 14.3 里程碑与交付物

| 里程碑 | 交付物 | 验收标准 |
| --- | --- | --- |
| M0 | Rot_G/Rot_H 和 mask 单元测试。 | 所有小图测试通过。 |
| M1 | Dense mask Transformer 原型。 | 能跑 synthetic 小任务。 |
| M2 | Neighbor-list attention。 | 小 N 输出与 dense 版本一致。 |
| M3 | MVP 实验结果。 | full/local/random/zig-zag 四类方法完整记录。 |
| M4 | LRA 单任务结果。 | 至少一个标准任务可复现实验。 |
| M5 | 参数与图结构消融。 | 得到推荐配置和负结果解释。 |
| M6 | 工程 profile 报告。 | 明确是否需要 xFormers/Triton/block-sparse。 |

# 15. 参考资料与资料来源
资料分为用户文档、理论背景、benchmark、baseline、工程实现和复现规范。外部资料于 2026-06-10 检索。
[D1] 用户上传文档：《固定 m×n Token 完全图与 Zig-Zag 乘法 Block 的复杂度》。
[D2] 用户上传文档：《Expander 图简介》。
[R1] Tay et al., “Long Range Arena: A Benchmark for Efficient Transformers”, arXiv:2011.04006。
[R2] google-research/long-range-arena, GitHub repository。
[R3] Zaheer et al., “Big Bird: Transformers for Longer Sequences”, arXiv:2007.14062。
[R4] google-research/bigbird, GitHub repository。
[R5] Google Research Blog: Constructing Transformers For Longer Sequences with Sparse Attention Methods。
[R6] facebookresearch/xformers, GitHub repository and documentation。
[R7] Shirzad et al., “Exphormer: Sparse Transformers for Graphs”。
[R8] hamed1375/Exphormer, GitHub repository。
[R9] PyTorch documentation: torch.cuda memory statistics。
[R10] PyTorch documentation: Reproducibility note。
[R11] Conda documentation: environment management。
[R12] tmux/screen documentation for unstable remote sessions。

# 16. 文档自检与剩余边界
本节记录针对上一版缺陷的自检结果。结论是：作为实验执行手册，当前版本已覆盖理论背景、方法定义、G/H 生成、MVP 配置、公平性、工程实现、远程环境、性能测试、复现和负结果诊断。没有发现会阻止开工的结构性缺陷。

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 章节结构 | 通过 | 服务器规范位于第 11 节，不再出现在目录之后。 |
| 目录一致性 | 通过 | 目录由当前 outline 生成，标题编号一致。 |
| 实验可执行性 | 通过 | MVP 配置、baseline、验收标准已固定。 |
| G/H 连边规则 | 通过 | Rot_G、Rot_H、cyclic/random permutation/random regular 规则已写明。 |
| 公平性 | 通过 | same-budget、raw K、effective K、duplicate rate 已明确。 |
| 性能测试 | 通过 | warmup、计时、CUDA synchronize、显存统计已规定。 |
| 远程规范 | 通过 | ssh huiwei、~/ysx/、ysx_base、GPU 占用和 tmux 已明确。 |
| 复现 | 通过 | seed、git commit、environment.yaml、checkpoint resume 已明确。 |
| 负结果 | 通过 | 提供了诊断决策树。 |

> **说明：** 剩余边界：文档无法替代真实实验结果。是否优于 random、是否在 A100 上获得实际速度收益、xFormers/Triton 是否值得接入，都必须由后续实验数据决定。
