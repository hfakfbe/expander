# Expander-First Causal Memory 架构备忘录

日期：2026-06-15

本文整理一个比 CEMA 更激进的版本：**Expander-First Causal Memory**，简称 **EFCM**。目标是让 expander 成为长程信息流的主角，而不是 memory 后面的辅助 mixer。

核心原则：

\[
\boxed{
\text{All long-range information must pass through expander edges.}
}
\]

也就是说：

- token 不能 dense-read 全体 memory；
- token 只能读 expander graph 上的局部邻域；
- token 只能写入口 slot 或入口邻域；
- 远程传播必须靠 expander propagation；
- attention 只是在 expander 给出的边上学习权重。

## 1. 为什么需要 Expander-First

上一版 CEMA 的 memory read 是：

\[
r_t=\operatorname{Attn}(h_t,M_{t-1}).
\]

这会让 token 直接读所有 memory slots。即使 memory 内部用了 expander mixing，真正的信息读取仍然是 dense cross-attention。因此 expander 更像正则器或平滑器，不是架构骨架。

EFCM 改成：

\[
r_t=\operatorname{Attn}(h_t,M_{\mathcal{R}_t}),
\qquad
|\mathcal{R}_t|\ll m.
\]

其中 \(\mathcal{R}_t\) 必须由 expander graph 的邻域给出。这样远程能力不来自 dense attention，而来自多层或多步 expander walk。

## 2. 基本对象

设 memory graph 为

\[
G_M=(V_M,E_M),
\qquad
|V_M|=m.
\]

每个节点是一个 memory slot：

\[
M_t=
\begin{bmatrix}
M_{t,1}\\
\cdots\\
M_{t,m}
\end{bmatrix}
\in\mathbb{R}^{m\times d_m}.
\]

设 \(G_M\) 是 \(D\)-in/\(D\)-out regular directed expander。令 row-normalized transition matrix 为

\[
P\in\mathbb{R}^{m\times m}.
\]

本文约定：

\[
P_{ij}>0
\]

表示 slot \(j\) 的信息可以传到 slot \(i\)。因此一次 expander propagation 的线性部分是：

\[
(PM)_i=\sum_{j:P_{ij}>0}P_{ij}M_j.
\]

理想情况下：

\[
P\mathbf{1}=\mathbf{1},
\qquad
\mathbf{1}^{\top}P=\mathbf{1}^{\top},
\qquad
\|P-J_m\|_2\le \rho<1.
\]

## 3. Token 到 Memory 的入口路由

每个 token 或 chunk 被分配到一个入口 slot：

\[
i_t=\pi(t)\in [m].
\]

最简单的非学习路由：

\[
\pi(t)=t\bmod m.
\]

更稳一点可以按 chunk 路由：

\[
\pi(t)=\left\lfloor \frac{t}{C_\pi}\right\rfloor \bmod m.
\]

也可以使用 learned router：

\[
\pi(t)=\operatorname{Top1Router}(h_t),
\]

但必须保证 router 不依赖未来 token。对于严格 NTP，位置路由或 causal hidden 路由是安全的。

## 4. Expander 邻域

定义 \(i\) 的 \(r\)-hop 入邻域：

\[
\Gamma_{\text{in}}^{\le r}(i)
=
\left\{
j:
\text{there exists a directed path } j\to\cdots\to i
\text{ of length }\le r
\right\}.
\]

定义 \(i\) 的 \(r\)-hop 出邻域：

\[
\Gamma_{\text{out}}^{\le r}(i)
=
\left\{
j:
\text{there exists a directed path } i\to\cdots\to j
\text{ of length }\le r
\right\}.
\]

读邻域：

\[
\mathcal{R}_t=\Gamma_{\text{in}}^{\le r_R}(i_t).
\]

写邻域：

\[
\mathcal{W}_t=\Gamma_{\text{out}}^{\le r_W}(i_t).
\]

如果图出入度都是 \(D\)，则

\[
|\mathcal{R}_t|
\le
1+D+\cdots+D^{r_R}
=
O(D^{r_R}),
\]

\[
|\mathcal{W}_t|
\le
1+D+\cdots+D^{r_W}
=
O(D^{r_W}).
\]

常用极简设定：

\[
r_R=1,
\qquad
r_W=0.
\]

也就是 token 读入口 slot 的一跳入邻居，但只写自己的入口 slot。

## 5. Read：Expander-Restricted Memory Read

token hidden 为 \(h_t\)。它只能读取 \(\mathcal{R}_t\) 中的 slots：

\[
M_{\mathcal{R}_t}
=
\{M_i:i\in\mathcal{R}_t\}.
\]

读 attention：

\[
q_t=h_tW_Q^R,
\]

\[
K_t^R=M_{\mathcal{R}_t}W_K^R,
\qquad
V_t^R=M_{\mathcal{R}_t}W_V^R.
\]

\[
r_t
=
\operatorname{softmax}
\left(
\frac{q_t(K_t^R)^\top}{\sqrt{d_r}}
\right)
V_t^R.
\]

读复杂度：

\[
T_{\text{read/token}}
=
O(|\mathcal{R}_t|d_m)
=
O(D^{r_R}d_m).
\]

这一步是 EFCM 和普通 memory attention 的关键区别：复杂度不含 \(m\)。

## 6. Write：Expander-Restricted Memory Write

token 只能写 \(\mathcal{W}_t\)：

\[
M_{\mathcal{W}_t}
=
\{M_i:i\in\mathcal{W}_t\}.
\]

slot-wise write weights：

\[
q_i^W=M_iW_Q^W,
\qquad
k_t^W=h_tW_K^W,
\qquad
v_t^W=h_tW_V^W.
\]

\[
a_{t,i}
=
\operatorname{softmax}_{i\in\mathcal{W}_t}
\left(
\frac{q_i^W(k_t^W)^\top}{\sqrt{d_w}}
\right).
\]

写入增量：

\[
\Delta M_{t,i}
=
a_{t,i}v_t^W,
\qquad
i\in\mathcal{W}_t.
\]

提交可以用 GRU-style update：

\[
M_{t,i}^{(0)}
=
\operatorname{GRUCell}
\left(
M_{t-1,i},\Delta M_{t,i}
\right),
\qquad
i\in\mathcal{W}_t.
\]

未被写入的 slot：

\[
M_{t,i}^{(0)}=M_{t-1,i},
\qquad
i\notin\mathcal{W}_t.
\]

写复杂度：

\[
T_{\text{write/token}}
=
O(|\mathcal{W}_t|d_m)
=
O(D^{r_W}d_m).
\]

当 \(r_W=0\) 时，每个 token 只写一个 slot：

\[
T_{\text{write/token}}=O(d_m).
\]

## 7. Propagate：Expander 是远程传播主干

写入之后，远程传播必须沿 expander edge 发生。做 \(K\) 轮 propagation：

\[
M^{(k+1)}_i
=
\operatorname{LN}
\left(
(1-\alpha)M_i^{(k)}
+
\alpha\,
\phi
\left(
\sum_{j:P_{ij}>0}P_{ij}M_j^{(k)}W_E^{(k)}
\right)
\right),
\]

\[
k=0,\ldots,K-1.
\]

矩阵写法：

\[
M^{(k+1)}
=
\operatorname{LN}
\left(
(1-\alpha)M^{(k)}
+
\alpha\,\phi(PM^{(k)}W_E^{(k)})
\right).
\]

最终：

\[
M_t=M^{(K)}.
\]

每轮 propagation 复杂度：

\[
O(mDd_m).
\]

\(K\) 轮：

\[
T_{\text{propagate}}
=
O(KmDd_m).
\]

这就是 EFCM 的“主角成本”。远程信息不是靠读全体 memory，而是靠这个 expander propagation 扩散。

## 8. 单 Token Decode 流程

每个 decode step：

1. 计算入口 slot：

\[
i_t=\pi(t).
\]

2. 读 local KV window：

\[
c_t=\operatorname{LocalCausalAttn}(h_t,\operatorname{KV}_{t-W:t}).
\]

3. 读 expander 邻域：

\[
r_t=\operatorname{Read}_{G_M}(h_t,M_{t-1},\mathcal{R}_t).
\]

4. 融合 hidden：

\[
\hat h_t=\operatorname{Block}(h_t,c_t,r_t).
\]

5. 写入口邻域：

\[
M_t^{(0)}
=
\operatorname{Write}_{G_M}(M_{t-1},\hat h_t,\mathcal{W}_t).
\]

6. 做 expander propagation：

\[
M_t
=
\operatorname{Propagate}_{G_M}^{K}(M_t^{(0)}).
\]

7. 输出 logits：

\[
\operatorname{logits}_{t+1}
=
\hat h_tW_{\text{vocab}}.
\]

单层单 token decode 复杂度：

\[
O\left(
Wd
+D^{r_R}d_m
+D^{r_W}d_m
+KmDd_m
\right).
\]

若 \(W,m,D,K,r_R,r_W\) 都固定，则相对历史长度 \(t\) 是常数复杂度。

## 9. Chunk Prefill 流程

把 prompt 分成 chunk：

\[
X_j=x_{s_j:e_j},
\qquad
|X_j|\le C.
\]

为保证因果性，chunk \(j\) 中所有 token 只能读旧 memory：

\[
M_{j-1}.
\]

不能读同一 chunk 聚合后的 \(M_j\)。

### 9.1 Chunk Read

chunk 内 local causal attention：

\[
C_j=\operatorname{LocalCausalAttn}(X_j).
\]

对每个 token \(t\in X_j\)，计算入口：

\[
i_t=\pi(t),
\qquad
\mathcal{R}_t=\Gamma_{\text{in}}^{\le r_R}(i_t).
\]

restricted memory read：

\[
r_t
=
\operatorname{Read}_{G_M}(h_t,M_{j-1},\mathcal{R}_t).
\]

得到 chunk hidden：

\[
H_j=\operatorname{Block}(X_j,C_j,R_j).
\]

### 9.2 Chunk Write

每个 token 产生局部写入。对每个 slot \(i\)，聚合所有写到它的 token：

\[
\Delta M_{j,i}
=
\sum_{t\in X_j:\,i\in\mathcal{W}_t}
a_{t,i}h_tW_V^W.
\]

提交：

\[
M_j^{(0)}
=
\operatorname{Commit}(M_{j-1},\Delta M_j).
\]

### 9.3 Chunk Propagation

chunk 结束后做 \(K\) 轮 expander propagation：

\[
M_j^{(k+1)}
=
\operatorname{Propagate}_{G_M}(M_j^{(k)}),
\qquad
k=0,\ldots,K-1.
\]

\[
M_j=M_j^{(K)}.
\]

\(M_j\) 只能被后续 chunk \(j+1,j+2,\ldots\) 使用。

### 9.4 Prefill 复杂度

若 chunk 内 local attention 使用 window \(W\)，则单层 prefill 复杂度为：

\[
O\left(
NWd
+ND^{r_R}d_m
+ND^{r_W}d_m
+\frac{N}{C}KmDd_m
\right).
\]

如果 chunk 内使用 full causal block attention，则 local 项变为：

\[
O(NCd).
\]

因此：

\[
O\left(
NCd
+ND^{r_R}d_m
+ND^{r_W}d_m
+\frac{N}{C}KmDd_m
\right).
\]

## 10. Decode 中的 Amortized Propagation 变体

逐 token 做全图 propagation：

\[
O(KmDd_m)
\]

可能偏贵。可以每 \(C_d\) 个 decode token 才跑一次 full propagation。

每个 token 仍然做：

\[
O(Wd+D^{r_R}d_m+D^{r_W}d_m).
\]

每 \(C_d\) 个 token 做一次：

\[
O(KmDd_m).
\]

amortized decode 复杂度：

\[
O\left(
Wd
+D^{r_R}d_m
+D^{r_W}d_m
+\frac{KmDd_m}{C_d}
\right).
\]

代价是远程传播延迟增加：

\[
\text{propagation delay}\le C_d.
\]

## 11. 相对 Full Attention 的加速比

这一节把变量压缩到一个有效每 token 预算，看清楚本质。

设 full causal attention 在长度 \(N\) 的序列上，每层 attention pair 数为

\[
E_{\text{full}}
\approx
\frac{N^2}{2}.
\]

忽略常数和 hidden width 后，full attention 的每 token 历史读取预算是：

\[
B_{\text{full/token}}\approx N.
\]

EFCM 的每 token 预算可以写成：

\[
B_{\text{eff}}
=
B_{\text{local}}
+B_{\text{read}}
+B_{\text{write}}
+B_{\text{prop}}.
\]

其中：

\[
B_{\text{local}}=W,
\]

\[
B_{\text{read}}\approx D^{r_R},
\]

\[
B_{\text{write}}\approx D^{r_W},
\]

\[
B_{\text{prop}}=\frac{KmD}{C_p}.
\]

\(C_p\) 是 propagation amortization interval：

- prefill 时通常 \(C_p=C\)，即每个 chunk 跑一次 propagation；
- decode 若每 token 都 propagation，则 \(C_p=1\)；
- decode 若每 \(C_d\) 个 token propagation 一次，则 \(C_p=C_d\)。

因此：

\[
B_{\text{eff}}
=
W+D^{r_R}+D^{r_W}+\frac{KmD}{C_p}.
\]

相对 full attention 的 pair-count 级加速比近似为：

\[
\operatorname{Speedup}_{\text{pair}}
\approx
\frac{N}{B_{\text{eff}}}
=
\frac{N}{
W+D^{r_R}+D^{r_W}+\frac{KmD}{C_p}
}.
\]

这就是最核心的式子。

### 11.1 本质解释

Full attention 的问题是：

\[
B_{\text{full/token}}\sim N.
\]

EFCM 的目标是让：

\[
B_{\text{eff}}\ll N.
\]

只要

\[
W+D^{r_R}+D^{r_W}+\frac{KmD}{C_p}
=O(1)
\quad\text{or}\quad
o(N),
\]

就能在长上下文上获得随 \(N\) 增长的理论加速。

这里最重要的不是变量多，而是四类成本的角色不同：

| 项 | 含义 | 是否随 \(N\) 增长 | 直觉 |
|---|---:|---:|---|
| \(W\) | 近期局部上下文 | 否 | 保留短程语言建模 |
| \(D^{r_R}\) | memory 读邻域 | 否 | token 只能看 expander 局部 |
| \(D^{r_W}\) | memory 写邻域 | 否 | token 只写少量 slots |
| \(KmD/C_p\) | 摊销后的 expander 传播 | 通常否 | 远程通信主成本 |

### 11.2 最小推荐配置下的加速比

取推荐参数：

\[
D=4,\quad r_R=1,\quad r_W=0,\quad K=1,\quad m=64,\quad W=512.
\]

则：

\[
D^{r_R}=4,
\qquad
D^{r_W}=1.
\]

如果 prefill chunk size 为

\[
C_p=C=256,
\]

则：

\[
\frac{KmD}{C_p}
=
\frac{1\cdot64\cdot4}{256}
=1.
\]

所以：

\[
B_{\text{eff}}
\approx
512+4+1+1
=518.
\]

相对 full attention：

\[
\operatorname{Speedup}_{\text{pair}}
\approx
\frac{N}{518}.
\]

例如：

| \(N\) | 近似加速比 |
|---:|---:|
| 4,096 | \(7.9\times\) |
| 16,384 | \(31.6\times\) |
| 65,536 | \(126.5\times\) |
| 262,144 | \(506.1\times\) |

这说明：如果 \(W=512\)，短程窗口其实是主要成本；expander 读写和传播在这个配置下很小。

### 11.3 Decode 的关键瓶颈

decode 如果每 token 都跑 full propagation，则

\[
C_p=1,
\]

\[
B_{\text{prop}}=KmD.
\]

在同样参数下：

\[
B_{\text{prop}}=1\cdot64\cdot4=256.
\]

\[
B_{\text{eff}}
=512+4+1+256
=773.
\]

decode pair-count 级加速：

\[
\operatorname{Speedup}_{\text{decode}}
\approx
\frac{N}{773}.
\]

如果每

\[
C_d=16
\]

个 token 才 propagation 一次，则：

\[
B_{\text{prop}}=\frac{256}{16}=16,
\]

\[
B_{\text{eff}}
=512+4+1+16
=533.
\]

这接近 prefill 的预算，但代价是远程 memory 扩散最多延迟 \(16\) 个 decode steps。

### 11.4 什么时候 expander 成本会变成主瓶颈

expander propagation 成本主导当且仅当：

\[
\frac{KmD}{C_p}
\gg
W.
\]

也就是：

\[
m
\gg
\frac{WC_p}{KD}.
\]

例如 \(W=512,C_p=256,K=1,D=4\)，阈值是：

\[
m\gg \frac{512\cdot256}{4}=32768.
\]

所以在 \(m=64\) 或 \(128\) 时，prefill 的 expander propagation 不是瓶颈。

decode 若 \(C_p=1\)，阈值变成：

\[
m\gg \frac{W}{KD}.
\]

同样 \(W=512,K=1,D=4\)，阈值是：

\[
m\gg128.
\]

这说明 decode 里逐 token propagation 对 \(m\) 更敏感。若 \(m\ge256\)，更应该使用 amortized propagation。

### 11.5 透过符号看设计原则

从

\[
B_{\text{eff}}
=
W+D^{r_R}+D^{r_W}+\frac{KmD}{C_p}
\]

可以直接读出设计原则：

1. \(W\) 决定短程语言建模成本，通常是最大常数项。
2. \(r_R,r_W\) 不能大，否则 \(D^r\) 指数增长。
3. \(D\) 既影响扩张质量，也线性影响 propagation 成本。
4. \(m\) 增大提高 memory capacity，但 decode propagation 成本线性增加。
5. \(C_p\) 是速度和延迟的旋钮：越大越快，但远程扩散越慢。
6. 真正的长上下文收益来自让 \(B_{\text{eff}}\) 固定，而让 \(N\) 增长。

简化一句话：

\[
\boxed{
\text{Full attention pays } N \text{ per token; EFCM pays } W+\text{small expander budget}.
}
\]

## 12. 与 CEMA 的区别

| 项目 | CEMA | EFCM |
|---|---|---|
| token 读 memory | 读全部 \(m\) slots | 只读 \(\mathcal{R}_t\) |
| token 写 memory | 可写全部或 dense slot attention | 只写 \(\mathcal{W}_t\) |
| expander 角色 | memory mixer / regularizer | 远程信息主干 |
| 长程通信来源 | memory dense read + mixer | expander walk |
| 读复杂度 | \(O(md_m)\) | \(O(D^{r_R}d_m)\) |
| 写复杂度 | \(O(md_m)\) 或较大 | \(O(D^{r_W}d_m)\) |
| 主成本 | memory attention | \(K\)-step propagation |

EFCM 的设计目标是让下面这句话成立：

\[
\text{If the expander graph is removed, long-range memory communication is removed.}
\]

这比“模型里有 expander”更强。

## 13. 推荐最小配置

第一版建议：

\[
m\in\{64,128\},
\qquad
D=4,
\qquad
K\in\{1,2\},
\]

\[
r_R=1,
\qquad
r_W=0,
\qquad
W=512,
\qquad
C=256.
\]

此时：

\[
|\mathcal{R}_t|\le 1+D=5,
\qquad
|\mathcal{W}_t|=1.
\]

每 token 只读约 5 个 memory slots，只写 1 个 slot。远程信息必须靠 propagation：

\[
M\leftarrow \operatorname{Propagate}_{G_M}(M).
\]

对照组：

1. \(P=I\)：无远程传播；
2. \(P=\) cycle：弱扩张；
3. \(P=\) random \(D\)-regular digraph：高概率 expander；
4. \(P=\) certified expander：带谱证书。

## 14. 需要报告的指标

### 14.1 图指标

\[
\rho(P)=\|P-J_m\|_2.
\]

多步混合：

\[
\|P^K-J_m\|_2\le \rho(P)^K.
\]

### 14.2 读写负载

slot 写入次数：

\[
\operatorname{load}_i
=
\sum_t \mathbf{1}[i\in\mathcal{W}_t].
\]

报告：

\[
\max_i\operatorname{load}_i,
\qquad
\operatorname{var}_i(\operatorname{load}_i).
\]

### 14.3 多步可达性

从入口 slot \(i_t\) 出发，经过 \(K\) 次 propagation 后能影响的 slots：

\[
\Gamma_{\text{out}}^{\le K}(i_t).
\]

报告平均覆盖率：

\[
\frac{1}{m}\left|\Gamma_{\text{out}}^{\le K}(i_t)\right|.
\]

### 14.4 任务指标

建议任务：

1. passkey retrieval；
2. multi-needle retrieval；
3. associative recall；
4. long-context LM；
5. code completion with long-range definitions。

## 15. 风险

1. **Propagation 成本**：如果 \(m\) 和 \(K\) 偏大，decode 做全图 propagation 会贵。
2. **入口路由冲突**：很多 token 写同一 slot，可能导致 memory overwrite。
3. **过度混合**：过强 propagation 可能让 slot 内容平均化。
4. **读邻域太小**：\(r_R=0\) 或 \(1\) 可能读不到刚扩散到合适位置的信息。
5. **训练难度**：模型必须学会把内容写入正确入口，并依赖 expander 多步传播。

## 16. 一句话定义

EFCM 是：

\[
\boxed{
\text{A causal recurrent memory architecture where read, write, and propagation are all constrained by an expander graph.}
}
\]

在这个版本中，attention 不再决定全局可见性；expander graph 决定全局可达性，attention 只决定已有边上的权重。
