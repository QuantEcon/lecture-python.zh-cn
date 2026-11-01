---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 观点的融合：布莱克韦尔-杜宾斯定理
  headings:
    Overview: 概述
    Probability measures on sequence spaces: 序列空间上的概率测度
    Probability measures on sequence spaces::The sequence space and its filtration: 序列空间及其滤流
    Probability measures on sequence spaces::Absolute continuity: 绝对连续性
    Probability measures on sequence spaces::Total variation distance: 全变差距离
    Probability measures on sequence spaces::The merging question: 融合问题
    The likelihood-ratio martingale: 似然比鞅
    The likelihood-ratio martingale::The likelihood ratio: 似然比
    The likelihood-ratio martingale::Connecting conditional measures to the likelihood ratio: 将条件测度与似然比联系起来
    The Blackwell–Dubins theorem: 布莱克韦尔-杜宾斯定理
    The Beta–Bernoulli model: 贝塔-伯努利模型
    The Beta–Bernoulli model::Model: 模型
    The Beta–Bernoulli model::The marginal likelihood and likelihood ratio: 边缘似然与似然比
    The Beta–Bernoulli model::The exact Blackwell–Dubins distance: 精确的布莱克韦尔-杜宾斯距离
    The Beta–Bernoulli model::Simulation: 模拟
    The Beta–Bernoulli model::Almost-sure convergence across many paths: 跨多条路径的几乎必然收敛
    The Beta–Bernoulli model::Is $d_n$ a supermartingale?: $d_n$ 是上鞅吗？
    'Failure of merging: mutual singularity': 融合的失败：相互奇异
    'Failure of merging: mutual singularity::Point-mass priors': 点质量先验
    'Kakutani''s theorem: when does merging hold?': 角谷定理：融合何时成立？
    'Kakutani''s theorem: when does merging hold?::Hellinger affinities': 海林格亲和度
    'Kakutani''s theorem: when does merging hold?::Kakutani''s dichotomy': 角谷二分法
    'Kakutani''s theorem: when does merging hold?::Implication for merging': 对融合的蕴含
    'Kakutani''s theorem: when does merging hold?::A Gaussian product-measure example': 一个高斯乘积测度例子
    Extension to continuous time: 推广到连续时间
    Extension to continuous time::Girsanov's theorem and the likelihood-ratio process: 吉尔萨诺夫定理与似然比过程
    Extension to continuous time::The dichotomy at infinity: 无穷远处的二分法
    Applications: 应用
    Applications::Bayesian learning: 贝叶斯学习
    Applications::Rational expectations and heterogeneous priors: 理性预期与异质先验
    Applications::Ergodic Markov chains: 遍历马尔可夫链
    The rate of merging: 融合的速率
    Summary and extensions: 总结与推广
    Summary and extensions::Applications in economics: 经济学中的应用
    Exercises: 练习
---

(merging_of_opinions)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 观点的融合：布莱克韦尔-杜宾斯定理

```{contents} Contents
:depth: 2
```

## 概述

本讲座研究 {cite:t}`blackwell1962` 的观点融合定理。

该定理提出了一个简单的问题：

> 如果两个主体对某个随机过程持有不同的先验信念，但无限期地观察相同的数据流，他们的概率评估最终会收敛吗？

在绝对连续性条件下，答案是肯定的。

如果 $Q \ll P$（即 $P$ 支配 $Q$），那么在 $P$ 和 $Q$ 下关于整个未来路径的条件分布会在全变差意义下融合，$Q$-几乎必然。

如果此外还有 $P \ll Q$（因而 $P \sim Q$），那么同样的结论在两个主体的概率下都成立。

这个结果与其他若干思想相关联：

- 贝叶斯一致性：当先验位于正确的绝对连续性类中时，后验预测趋近于真值（{doc}`likelihood_bayes`）。
- 一致性结果：{cite:t}`aumann1976` 研究了具有共同先验和私人信息的主体；布莱克韦尔-杜宾斯则研究了具有不同先验和共同信息的主体。
- 角谷二分法：对于乘积测度，等价性与奇异性可以从一个海林格准则中读出。

我们在离散时间中展开理论，然后勾勒连续时间的类比。

在整个讨论中，我们使用贝塔-伯努利模型作为贯穿始终的例子。

两个主体观察相同的抛硬币结果流，但从关于硬币偏差的不同先验出发。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, binom, betabinom, norm
from scipy.special import betaln, beta as beta_fn, polygamma
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
mpl.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
```

## 序列空间上的概率测度


### 序列空间及其滤流

设 $(S, \mathscr{S})$ 是一个标准博雷尔空间（即与完备可分度量空间的一个博雷尔子集同构的可测空间），称为信号空间。

标准博雷尔假设保证了正则条件分布的存在性，这正是该定理所要求的。

令 $\Omega = S^{\mathbb{N}}$，即所有无限序列
$\omega = (x_1, x_2, \ldots)$（其中 $x_n \in S$）的集合，配备乘积
$\sigma$-代数 $\mathscr{F} = \mathscr{S}^{\otimes \mathbb{N}}$。

对于每个 $n \geq 1$，定义**有限视界** $\sigma$-代数

$$
\mathscr{F}_n = \sigma(x_1, \ldots, x_n),
$$

并设 $\mathscr{F}_0 = \{\emptyset, \Omega\}$，因此 $\mathscr{F}_0 \subseteq \mathscr{F}_1 \subseteq \mathscr{F}_2 \subseteq \cdots \subseteq \mathscr{F}$。

定义 $\mathscr{F}_\infty = \sigma\!\left(\bigcup_{n \geq 1} \mathscr{F}_n\right)$，即最终能够学到的事件所构成的 $\sigma$-代数。

柱集 $\{x_1 \in A_1, \ldots, x_n \in A_n\}$ 生成了乘积 $\sigma$-代数，并且属于 $\bigcup_n \mathscr{F}_n$，因此 $\mathscr{F}_\infty = \mathscr{F}$。

$\sigma$-代数 $\mathscr{F}_\infty$ 不同于**尾 $\sigma$-代数** $\mathscr{T} = \bigcap_{n \geq 1} \sigma(x_n, x_{n+1}, \ldots)$，后者的事件（例如 $\{x_n \in A \text{ 对无穷多个 } n \text{ 成立}\}$）不依赖于任何有限长度的数据。

集合 $\{\mathscr{F}_n\}_{n \geq 1}$ 是由观察过程生成的**自然滤流**；$\mathscr{F}_n$ 编码了从前 $n$ 个数据点能够学到的一切。

设 $P$ 和 $Q$ 表示 $(\Omega, \mathscr{F})$ 上的两个概率测度。

记 $P_n = P|_{\mathscr{F}_n}$ 和 $Q_n = Q|_{\mathscr{F}_n}$ 表示它们
限制到时刻 $n$ 之前的历史。

### 绝对连续性

```{prf:definition} 绝对连续性
:label: absolute_continuity

$P$ 关于 $Q$ **绝对连续**，记为 $P \ll Q$，如果对于每个 $A \in \mathscr{F}$，
$Q(A) = 0$ 蕴含 $P(A) = 0$。

它们**相互绝对连续**，或**等价**，记为 $P \sim Q$，
如果 $P \ll Q$ 和 $Q \ll P$ 都成立。

$P$ 关于 $Q$ **局部绝对连续**，如果对于每个 $n \geq 1$，$P_n \ll Q_n$。
```

全局绝对连续性 $P \ll Q$ 蕴含局部绝对连续性，但
反之不然。

相互绝对连续意味着两个主体在哪些事件是*可能的*这一点上意见一致。

他们可以对概率有不同看法，但都不会排除对方认为可能的事件。

### 全变差距离

```{prf:definition} 全变差距离
:label: total_variation_distance

对于 $(E, \mathscr{E})$ 上的两个概率测度 $\mu$ 和 $\nu$，

$$
\|\mu - \nu\|_{\mathrm{TV}}
= \sup_{A \in \mathscr{E}} |\mu(A) - \nu(A)|
= \frac{1}{2} \int_E \left|\frac{d\mu}{d\lambda} - \frac{d\nu}{d\lambda}\right| d\lambda,
$$

其中 $\lambda$ 是任意**支配测度**，即 $\mu \ll \lambda$ 且 $\nu \ll \lambda$（例如 $\lambda = \mu + \nu$）。

距离 $\|\mu - \nu\|_{\mathrm{TV}}$ 位于 $[0,1]$ 之中；它等于 0 当且仅当 $\mu = \nu$，等于 1 当且仅当 $\mu \perp \nu$（相互奇异）。
```

当 $\mu \ll \nu$ 且 $f = d\mu/d\nu$ 时，

$$
\|\mu - \nu\|_{\mathrm{TV}} = \mathbb{E}_\nu[(f-1)^+] = 1 - \mathbb{E}_\nu[\min(f,1)].
$$

```{exercise}
:label: tv_derivation

证明上述恒等式。

*提示：*从 $\|\mu - \nu\|_{\mathrm{TV}} = \tfrac{1}{2}\,\mathbb{E}_\nu[|f - 1|]$（这由取 $\nu$ 作为支配测度得出）出发，并利用 $\mathbb{E}_\nu[f] = 1$ 这一事实。
```

```{solution} tv_derivation
:class: dropdown

由于 $\mu \ll \nu$，我们可以取 $\nu$ 作为支配测度，因此 $d\mu/d\nu = f$ 且 $d\nu/d\nu = 1$，给出

$$
\|\mu - \nu\|_{\mathrm{TV}} = \tfrac{1}{2}\,\mathbb{E}_\nu[|f - 1|].
$$

写出 $|f-1| = (f-1)^+ + (1-f)^+$。

由于 $\mu$ 是概率测度，$\mathbb{E}_\nu[f] = 1$，因此两部分贡献相等：$\mathbb{E}_\nu[(f-1)^+] = \mathbb{E}_\nu[(1-f)^+]$。

因此 $\tfrac{1}{2}\,\mathbb{E}_\nu[|f-1|] = \mathbb{E}_\nu[(f-1)^+]$。

接下来，注意 $(f-1)^+ = f - \min(f,1)$，所以 $\mathbb{E}_\nu[(f-1)^+] = \mathbb{E}_\nu[f] - \mathbb{E}_\nu[\min(f,1)] = 1 - \mathbb{E}_\nu[\min(f,1)]$。
```

全变差是概率测度之间距离的最强的标准概念之一。

如果两个测度在全变差意义下接近，那么它们对每个事件的概率都接近。

### 融合问题

布莱克韦尔-杜宾斯定理研究给定*过去*时*未来*的条件分布。

在时刻 $n$，观察到 $(x_1,\ldots,x_n)$ 之后，每个主体对所有未来事件形成一个条件分布：

$$
P(\,\cdot\,|\,\mathscr{F}_n)(\omega), \qquad
Q(\,\cdot\,|\,\mathscr{F}_n)(\omega).
$$

这些是关于整个未来路径的概率测度，而不仅仅是下一个观察。

融合问题询问是否

$$
d_n \;:=\; \bigl\|P(\,\cdot\,|\,\mathscr{F}_n) - Q(\,\cdot\,|\,\mathscr{F}_n)\bigr\|_{\mathrm{TV}}
\;\longrightarrow\; 0
$$

当 $n \to \infty$ 时几乎必然成立。

在 $n = 0$ 时，条件化是平凡的，此时 $d_0 = \|P - Q\|_{\mathrm{TV}}$。

## 似然比鞅

我们的主要工具是拉东-尼科迪姆导数过程。

### 似然比

由于对于每个 $n$，$Q \ll P$ 蕴含 $Q_n \ll P_n$，拉东-尼科迪姆
定理保证了似然比的存在性

$$
Z_n = \frac{dQ_n}{dP_n}, \qquad Z_n \geq 0 \;\; P\text{-a.s.},
\qquad \mathbb{E}_P[Z_n] = 1.
$$

关键的结构性质是，全局绝对连续性 $Q \ll P$
蕴含在整个 $(\Omega, \mathscr{F})$ 上存在总体拉东-尼科迪姆导数 $Z = dQ/dP$，
并且

$$
Z_n = \mathbb{E}_P[Z \,|\, \mathscr{F}_n] \qquad P\text{-a.s.}
$$

也就是说，$\{Z_n, \mathscr{F}_n\}_{n \geq 0}$ 是一个非负、一致
可积的 $P$-鞅，且 $Z_0 = 1$。

在本讲座中，$Z$ 和 $Z_n$ 均表示 $Q$ 相对于 $P$ 的密度。

后验概率是同一类型的鞅；{doc}`blackwell_kihlstrom` 利用这一性质来比较实验。

```{prf:lemma} 鞅收敛
:label: martingale_convergence

似然比过程 $\{Z_n\}$ 满足：

1. 当 $n \to \infty$ 时，$Z_n \to Z_\infty$ $P$-几乎必然。
2. $Z_\infty = \mathbb{E}_P[Z \,|\, \mathscr{F}_\infty] = Z$ $P$-a.s.
3. $Z_n \to Z_\infty$ 在 $L^1(P)$ 中：$\;\mathbb{E}_P[|Z_n - Z_\infty|] \to 0$。

*证明概要。*非负性和鞅性质给出了在 $L^1(P)$ 中的
有界性。

然后几乎必然收敛由杜布鞅
收敛定理 {cite:p}`doob1953` 得出。

一致可积性（它通过条件延森不等式
由 $Z \in L^1(P)$ 得出）将其提升到
$L^1(P)$ 收敛。

列维上升定理确定该极限为 $\mathbb{E}_P[Z \,|\, \mathscr{F}_\infty]$，而由于 $\mathscr{F}_\infty = \mathscr{F}$，此极限即为 $Z$。$\square$
```

从此以后，我们用 $Z_\infty$ 表示 $Z = dQ/dP$。

### 将条件测度与似然比联系起来

下面的恒等式将似然比与条件分布联系起来。

在集合 $\{Z_n > 0\}$ 上，$Q(\,\cdot\,|\,\mathscr{F}_n)$ 关于 $P(\,\cdot\,|\,\mathscr{F}_n)$ 的
拉东-尼科迪姆导数为

$$
\frac{d\,Q(\,\cdot\,|\,\mathscr{F}_n)}{d\,P(\,\cdot\,|\,\mathscr{F}_n)}
= \frac{Z_\infty}{Z_n}
\qquad P\text{-a.s. on } \{Z_n > 0\}.
$$

这就是抽象贝叶斯公式 $Q(A \,|\, \mathscr{F}_n) = \mathbb{E}_P[Z_\infty \mathbf{1}_A \,|\, \mathscr{F}_n] / \mathbb{E}_P[Z_\infty \,|\, \mathscr{F}_n]$。

集合 $\{Z_n > 0\}$ 的 $Q$-概率为一，因为 $Q(Z_n = 0) = \mathbb{E}_P[Z_n \mathbf{1}\{Z_n = 0\}] = 0$。

于是应用带 $f = Z_\infty / Z_n$ 的全变差公式给出

$$
d_n
= \mathbb{E}_{P(\cdot|\mathscr{F}_n)}\!\left[\left(\frac{Z_\infty}{Z_n} - 1\right)^{\!+}\right]
= 1 - \mathbb{E}_{P(\cdot|\mathscr{F}_n)}\!\left[\min\!\left(\frac{Z_\infty}{Z_n},\,1\right)\right].
$$

由于 $Z_n$ 是 $\mathscr{F}_n$-可测的，该量也可写成条件期望之比：

$$
d_n
= \frac{\mathbb{E}_P[(Z_\infty - Z_n)^+ \,|\, \mathscr{F}_n]}{Z_n}
= \frac{\mathbb{E}_P[|Z_\infty - Z_n| \,|\, \mathscr{F}_n]}{2 Z_n}
\qquad Q\text{-a.s.}
$$ (eq_dn_conditional)

第二个等式成立是因为 $\mathbb{E}_P[Z_\infty - Z_n \,|\, \mathscr{F}_n] = 0$，所以 $Z_\infty - Z_n$ 的正部与负部具有相同的条件均值。

两边乘以 $Z_n$ 并取 $P$-期望（然后对非负的 $\mathscr{F}_n$-可测的 $g$ 使用 $\mathbb{E}_P[Z_n \, g] = \mathbb{E}_Q[g]$）：

$$
2\,\mathbb{E}_Q[d_n] \;=\; \mathbb{E}_P[|Z_\infty - Z_n|].
$$

因此鞅的 $L^1(P)$ 收敛给出 $\mathbb{E}_Q[d_n] \to 0$，且其收敛速度决定了 $\mathbb{E}_Q[d_n]$ 趋于零的快慢。

## 布莱克韦尔-杜宾斯定理

```{prf:theorem} 布莱克韦尔-杜宾斯（1962）
:label: blackwell_dubins

设 $P$ 和 $Q$ 是 $(\Omega, \mathscr{F})$ 上的概率测度，满足
$Q \ll P$。

定义

$$
d_n = \bigl\|P(\,\cdot\,|\,\mathscr{F}_n) - Q(\,\cdot\,|\,\mathscr{F}_n)\bigr\|_{\mathrm{TV}}.
$$

那么 $d_n \to 0$ $Q$-几乎必然。
```

恒等式 $\mathbb{E}_Q[d_n] = \tfrac{1}{2}\,\mathbb{E}_P[|Z_\infty - Z_n|] \to 0$ 给出了 $Q$-概率意义下的收敛。

几乎必然收敛的证明有三个步骤。

步骤 1. $d_n$ 的一个公式。

方程 {eq}`eq_dn_conditional` 将 $d_n$ 写为 $\mathbb{E}_P[|Z_\infty - Z_n| \,|\, \mathscr{F}_n]$ 与 $2 Z_n$ 的比值，$Q$-a.s.

步骤 2. 分母收敛到一个正的极限。

由 {prf:ref}`martingale_convergence`，$Z_n \to Z_\infty$ $P$-a.s.，从而由 $Q \ll P$ 得 $Q$-a.s. 成立。

该极限 $Q$-几乎必然为正，因为 $Q(Z_\infty = 0) = \mathbb{E}_P[Z_\infty \mathbf{1}\{Z_\infty = 0\}] = 0$。

步骤 3. 分子收敛到零。

固定 $M > 0$，将 $Z_\infty = U + V$ 拆分，其中 $U = \min(Z_\infty, M)$，$V = (Z_\infty - M)^+$。

令 $U_n = \mathbb{E}_P[U \,|\, \mathscr{F}_n]$ 且 $V_n = \mathbb{E}_P[V \,|\, \mathscr{F}_n]$，于是 $Z_n = U_n + V_n$。

由三角不等式得 $|Z_\infty - Z_n| \leq |U - U_n| + V + V_n$，对 $\mathscr{F}_n$ 取条件期望得到

$$
\mathbb{E}_P[|Z_\infty - Z_n| \,|\, \mathscr{F}_n]
\leq \mathbb{E}_P[|U - U_n| \,|\, \mathscr{F}_n] + 2 V_n .
$$

由列维向上定理，$V_n \to V$ $P$-a.s.

对于有界部分，固定 $m$，令 $W_m = \sup_{k \geq m} |U - U_k| \leq 2M$。

对于 $n \geq m$，有 $\mathbb{E}_P[|U - U_n| \,|\, \mathscr{F}_n] \leq \mathbb{E}_P[W_m \,|\, \mathscr{F}_n]$，而列维向上定理使右边当 $n \to \infty$ 时趋于 $W_m$。

列维向上定理同样给出 $U_k \to U$ $P$-a.s.，因此当 $m \to \infty$ 时 $W_m \to 0$。

综合这些极限，

$$
\limsup_{n \to \infty} \mathbb{E}_P[|Z_\infty - Z_n| \,|\, \mathscr{F}_n] \leq 2\,(Z_\infty - M)^+
\qquad P\text{-a.s.}
$$

对每个 $M$ 都成立。

令 $M$ 沿整数趋于无穷，右边趋于零，因为 $Z_\infty$ 是有限的。

所以分子 $P$-几乎必然收敛到零，从而 $Q$-几乎必然也收敛到零。

结合步骤 1–3，得到 $d_n \to 0$ $Q$-a.s. $\square$

关于有界部分 $U$ 的论证即为亨特引理（Hunt's lemma）。

该证明没有用到 $\{d_n\}$ 的单调性；{ref}`后面一节 <mo_supermartingale>` 表明 $\{d_n\}$ 不一定是上鞅。

```{prf:remark} 单边绝对连续性 vs. 相互绝对连续性
:label: one_sided_vs_mutual

该定理只要求 $Q \ll P$，而不要求 $P \ll Q$。

单边绝对连续性 $Q \ll P$ 给出 $Q$-几乎必然的融合。

由于 $Q \ll P$ 意味着每个 $P$-零集也是 $Q$-零集，$Q$-a.s. 收敛并不*自动*蕴含 $P$-a.s. 收敛。

要得出在*两个*主体的测度下 $d_n \to 0$，需要相互绝对连续性 $P \sim Q$。

在加上 $P \ll Q$ 之后，可以将 $P$ 和 $Q$ 的角色互换运行该证明，从而也得到 $d_n \to 0$ $P$-a.s.。
```

```{prf:remark} 尖锐性
:label: sharpness

绝对连续性很重要。

当 $P$ 和 $Q$ 奇异时，融合可能完全失败。

假设 $P \perp Q$，且对每个 $n$ 都有 $P_n \sim Q_n$。

取 $S \in \mathscr{F}$，满足 $P(S) = 1$ 且 $Q(S) = 0$。

那么 $P(S \,|\, \mathscr{F}_n) = 1$ $P$-a.s.，而 $Q(S \,|\, \mathscr{F}_n) = 0$ $Q$-a.s.

事件 $\{P(S \,|\, \mathscr{F}_n) = 1\}$ 与 $\{Q(S \,|\, \mathscr{F}_n) = 0\}$ 都属于 $\mathscr{F}_n$，因此局部等价性使得它们各自在 $P$-a.s. 和 $Q$-a.s. 意义下都成立。

因此对每个 $n$，在两种测度下都有 $d_n \geq |P(S \,|\, \mathscr{F}_n) - Q(S \,|\, \mathscr{F}_n)| = 1$。

下面的点质量例子对每个 $n$ 都有 $d_n = 1$。

对于乘积测度，后面的角谷定理给出了一个尖锐的等价性与奇异性二分法。
```

## 贝塔-伯努利模型

在转向 Python 之前，我们介绍贯穿所有
模拟的主要例子。

### 模型

假设数据流 $(x_1, x_2, \ldots)$ 由独立同分布的伯努利
抽样构成，未知概率为 $p^* \in (0,1)$。

主体 $i$ 有一个贝塔先验：

$$
p \sim \mathrm{Beta}(\alpha_i, \beta_i), \qquad i = 1, 2.
$$

在观察到 $n$ 次抽样中有 $k$ 次成功之后，贝叶斯法则给出
后验

$$
p \,|\, x^n \;\sim\; \mathrm{Beta}(\alpha_i + k,\; \beta_i + n - k),
$$

而单步向前预测概率为

$$
\hat{p}_i^n = \mathbb{E}[p\,|\,x^n] = \frac{\alpha_i + k}{\alpha_i + \beta_i + n}.
$$

由强大数定律，$k/n \to p^*$ 几乎必然，因此无论
主体的初始先验 $(\alpha_i, \beta_i)$ 如何，$\hat{p}_1^n$ 和 $\hat{p}_2^n$ 都收敛到 $p^*$。

### 边缘似然与似然比

对于每个固定的值 $p \in (0,1)$，令 $P_p$ 表示无限序列上的独立同分布伯努利$(p)$
概率律。

主体 $i$ 不知道 $p$。

相反，主体 $i$ 对 $p$ 放置先验密度 $\pi_i$，这通过下式在数据序列上诱导出
一个概率测度 $P_i$

$$
P_i(A) = \int_0^1 P_p(A)\,\pi_i(p)\,dp
\qquad \text{对于每个事件 } A.
$$

因此 $P_i$ 是主体在对 $p$ 的不确定性取平均之后关于
历史的边缘概率测度。

特别地，如果 $x^n$ 是一个精确的观察历史，有 $k$ 次成功，那么
$P_i(x^n)$ 表示主体 $i$ 在这个混合测度下赋予该历史的概率。

要计算它，从贝塔密度出发

$$
\pi_i(p)
= \frac{p^{\alpha_i - 1} (1-p)^{\beta_i - 1}}{B(\alpha_i, \beta_i)},
\qquad 0 < p < 1.
$$

给定 $p$，该有序历史的概率为 $p^k (1-p)^{n-k}$。

因此

$$
\begin{aligned}
P_i(x^n)
&= \int_0^1 p^k (1-p)^{n-k} \pi_i(p)\, dp \\
&= \frac{1}{B(\alpha_i, \beta_i)}
\int_0^1 p^{\alpha_i + k - 1} (1-p)^{\beta_i + n - k - 1}\, dp \\
&= \frac{B(\alpha_i + k,\; \beta_i + n - k)}{B(\alpha_i,\, \beta_i)}.
\end{aligned}
$$

其中 $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$ 是贝塔函数。

这个表达式是有序历史 $x^n$ 的概率。

它仅通过计数 $k$ 依赖于数据，因此成功次数相同的历史获得相同的概率。

因此时刻 $n$ 处的似然比为

$$
Z_n = \frac{P_{1,n}(x^n)}{P_{2,n}(x^n)}
= \frac{B(\alpha_2,\, \beta_2)}{B(\alpha_1,\, \beta_1)}
\cdot
\frac{B(\alpha_1 + k,\, \beta_1 + n - k)}{B(\alpha_2 + k,\, \beta_2 + n - k)}.
$$

按照定理中的记号，$Q = P_1$，$P = P_2$。

过程 $\{Z_n\}$ 是 $P_2$（主体 2 的概率）下的一个鞅，它几乎必然
收敛到一个有限正极限 $Z_\infty$，反映了对于任何具有正参数的贝塔先验，
$P_1 \sim P_2$ 这一事实。

该极限有一个闭式表达。

令 $\bar p = \lim_n k_n/n$，这在 $P_i$ 意义下几乎必然存在，因为在给定 $p$ 的条件下，强大数定律适用。

那么

$$
Z_\infty = \frac{dP_1}{dP_2} = \frac{\pi_1(\bar p)}{\pi_2(\bar p)}.
$$

{ref}`mo_ex_zinf` 要求给出该式的推导。

对于下面使用的先验 $\mathrm{Beta}(1, 8)$ 和 $\mathrm{Beta}(8, 1)$，有 $\pi_1(p) = 8(1-p)^7$ 和 $\pi_2(p) = 8p^7$，因此

$$
\log Z_\infty = 7 \log \frac{1 - \bar p}{\bar p}.
$$

### 精确的布莱克韦尔-杜宾斯距离

对于贝塔-伯努利模型，存在一个 $d_n$ 的简洁公式。

根据构造，给定 $x^n$ 时主体 $i$ 对未来序列
$(x_{n+1}, x_{n+2}, \ldots)$ 的条件分布
是独立同分布伯努利$(p)$
过程的混合，其中 $p$ 从主体 $i$ 的后验贝塔分布抽取。

{doc}`exchangeable` 描述了德菲内蒂定理，该定理为每个可交换过程给出了这样一种混合；在此处，这种混合是由构造直接得到的。

不同 $p$ 的伯努利$(p)^{\infty}$ 测度相互
奇异，且极限频率 $\bar p$ 可恢复出 $p$。

映射 $\omega \mapsto \bar p$ 将每个主体对未来的条件分布送到该主体的后验分布，因此条件分布之间的全变差距离至少等于后验分布之间的全变差距离。

每个条件分布都是后验分布在马尔可夫核 $p \mapsto \mathrm{Bernoulli}(p)^{\infty}$ 下的像，因此条件分布之间的全变差距离至多等于后验分布之间的全变差距离。

因此

$$
d_n
= \bigl\|\mathrm{Beta}(\alpha_1 + k_n,\,\beta_1 + n - k_n)
- \mathrm{Beta}(\alpha_2 + k_n,\,\beta_2 + n - k_n)\bigr\|_{\mathrm{TV}}.
$$

两个后验都集中于 $k_n/n$ 附近，标准差为 $n^{-1/2}$ 阶。

仅凭集中性并不能得出 $d_n \to 0$。

两个标准差均为 $s_n$ 的分布，其全变差距离趋于零，仅当它们的均值之差为 $o(s_n)$ 时才成立。

当 $\alpha_1 + \beta_1 = \alpha_2 + \beta_2 = 9$（如下文情形）时，后验均值之差为 $\delta_n = (\alpha_2 - \alpha_1)/(n + 9) = 7/(n+9)$，其为 $O(1/n)$ 阶。

两个后验标准差都近似为 $s_n = \sqrt{\hat p_n (1 - \hat p_n)/n}$，其中 $\hat p_n = k_n/n$。

用具有公共标准差 $s_n$、均值相差 $\delta_n$ 的正态分布来近似两个后验，可得

$$
d_n \approx 2\,\Phi\!\left(\frac{\delta_n}{2 s_n}\right) - 1
\approx \frac{7}{\sqrt{2 \pi n\, \hat p_n (1 - \hat p_n)}},
$$

其中 $\Phi$ 是标准正态累积分布函数。

当 $\hat p_n = 0.65$ 时，右边为 $5.85/\sqrt{n}$。

关于融合速率的小节将检验这个近似。

当 $\alpha_1 + \beta_1 = \alpha_2 + \beta_2$ 时，两个后验密度之比正比于 $((1-p)/p)^{\beta_1 - \beta_2}$，它关于 $p$ 是单调的。

两条密度曲线仅在一点 $x^*$ 处相交，且 $d_n = |F_1(x^*) - F_2(x^*)|$，其中 $F_i$ 是主体 $i$ 的后验累积分布函数。

下面的代码实现了上述贝塔-伯努利更新、预测概率、全变差距离和似然比的计算。

```{code-cell} ipython3
def beta_bernoulli_update(data, a0, b0):
    """
    序贯贝塔-伯努利贝叶斯更新。
    """
    n = len(data)
    cum_k = np.concatenate([[0], np.cumsum(data)])   # 累积成功次数
    ns    = np.arange(n + 1)                          # 0, 1, ..., n
    a_post = a0 + cum_k
    b_post = b0 + (ns - cum_k)
    return a_post, b_post


def predictive_prob(a_post, b_post):
    """单步向前预测概率 P(X=1 | data)。"""
    return a_post / (a_post + b_post)


def tv_distance_beta(a1, b1, a2, b2):
    """
    当 a1 + b1 = a2 + b2 且 a1 != a2 时，
    Beta(a1, b1) 和 Beta(a2, b2) 之间的精确全变差距离。

    两条密度曲线仅在一点 x_star 处相交，其中
    log(x_star / (1 - x_star))
        = (betaln(a2, b2) - betaln(a1, b1)) / (b1 - b2)。
    可按元素作用于数组。
    """
    logit_x = (betaln(a2, b2) - betaln(a1, b1)) / (b1 - b2)
    x_star = 1 / (1 + np.exp(-logit_x))
    return np.abs(beta_dist.cdf(x_star, a1, b1)
                  - beta_dist.cdf(x_star, a2, b2))


def log_likelihood_ratio(data, a1, b1, a2, b2):
    """
    计算对数似然比 log Z_n = log P1_n(data) - log P2_n(data)
    对 `data` 的每个前缀。

    返回一个长度为 len(data) + 1 的数组，从 0 开始（数据之前）。
    """
    a1p, b1p = beta_bernoulli_update(data, a1, b1)
    a2p, b2p = beta_bernoulli_update(data, a2, b2)
    log_P1 = betaln(a1p, b1p) - betaln(a1, b1)
    log_P2 = betaln(a2p, b2p) - betaln(a2, b2)
    return log_P1 - log_P2


def run_simulation(p_true, a1, b1, a2, b2, n_steps, seed=0):
    """
    模拟融合实验的一次实现。

    返回一个字典，包含长度为 n_steps + 1 的数组（索引 0 = 先验）。
    """
    rng  = np.random.default_rng(seed)
    data = rng.binomial(1, p_true, n_steps)

    a1p, b1p = beta_bernoulli_update(data, a1, b1)
    a2p, b2p = beta_bernoulli_update(data, a2, b2)

    pred1    = predictive_prob(a1p, b1p)
    pred2    = predictive_prob(a2p, b2p)
    tv_1step = np.abs(pred1 - pred2)

    # 后验贝塔之间的全变差；在此模型中它等于 d_n
    tv_beta = tv_distance_beta(a1p, b1p, a2p, b2p)

    log_Z = log_likelihood_ratio(data, a1, b1, a2, b2)

    return dict(data=data, pred1=pred1, pred2=pred2,
                tv_1step=tv_1step, tv_beta=tv_beta, log_Z=log_Z)
```

### 模拟

我们选择两个对一枚硬币偏差持有非常不同信念的主体。

- 主体 1（怀疑者）：先验 $\mathrm{Beta}(1, 8)$，因此
  $\hat{p}_1^0 = 1/9 \approx 0.11$。
- 主体 2（乐观者）：先验 $\mathrm{Beta}(8, 1)$，因此
  $\hat{p}_2^0 = 8/9 \approx 0.89$。

两个先验都支撑在整个 $(0,1)$ 上，因此 $P_1 \sim P_2$。

布莱克韦尔-杜宾斯保证在 $P_1$ 几乎每条路径上以及 $P_2$ 几乎每条路径上都发生融合。

下图跟踪由一枚正面朝上概率固定为 $p^* = 0.65$ 的硬币生成的一条路径。

这样一条路径的分布律 $P_{0.65}$ 相对于 $P_1$ 和 $P_2$ 是奇异的：事件 $\{k_n/n \to 0.65\}$ 在 $P_{0.65}$ 下的概率为一，而在 $P_i$ 下的概率为零。

因此布莱克韦尔-杜宾斯定理对这条路径没有任何说明。

在固定 $p^*$ 下的融合源于伯恩斯坦-冯·米塞斯定理。

在 $P_{p^*}$ 下，每个主体的后验在全变差意义下接近均值为 $k_n/n$、方差为 $p^*(1-p^*)/n$ 的正态分布，因此两个后验彼此接近。

上面的正态近似将此说得更具体。

下一小节在每条路径上从主体 1 的先验中抽取 $p$，这样我们就处于定理所设定的情形中。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      贝塔-伯努利例子中的融合。
      四个面板显示后验预测均值、全变差距离 $d_n$、对数似然比 $\log Z_n$，以及选定视界处的后验密度。
    name: fig-merging-of-opinions-beta-bernoulli
---
p_true = 0.65
a1, b1 = 1.0, 8.0    # 怀疑者
a2, b2 = 8.0, 1.0    # 乐观者
n_steps = 600

sim = run_simulation(p_true, a1, b1, a2, b2, n_steps, seed=7)
steps = np.arange(n_steps + 1)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
ax = axes[0, 0]
ax.plot(steps, sim['pred1'], color='steelblue', lw=2,
        label=r'主体 1 $\hat p_1^n$（先验：怀疑者）')
ax.plot(steps, sim['pred2'], color='firebrick', lw=2,
        label=r'主体 2 $\hat p_2^n$（先验：乐观者）')
ax.axhline(p_true, color='black', lw=1.0, ls='--',
           label=f'真值 $p^*={p_true}$')
ax.set_xlabel('观察数 $n$')
ax.set_ylabel('预测概率')
ax.set_title('(a) 后验预测均值')
ax.legend(fontsize=8)
ax.set_ylim(0, 1)

ax = axes[0, 1]
ax.semilogy(steps, sim['tv_beta'] + 1e-10, color='mediumpurple', lw=2)
ax.set_xlabel('观察数 $n$')
ax.set_ylabel(
    r'$d_n = \|P(\cdot|\mathscr{F}_n)'
    r' - Q(\cdot|\mathscr{F}_n)\|_{\mathrm{TV}}$'
)
ax.set_title(r'(b) 全变差距离 $d_n$')
ax.set_ylim(0.1, 1.1)

ax = axes[1, 0]
log_Z_limit = 7 * np.log((1 - p_true) / p_true)
ax.plot(steps, sim['log_Z'], color='darkorange', lw=2,
        label=r'$\log Z_n$')
ax.axhline(log_Z_limit, color='black', lw=1.0, ls='--',
           label=r'$7 \log((1 - p^*)/p^*)$')
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel('观察数 $n$')
ax.set_ylabel(r'$\log Z_n$')
ax.set_title(r'(c) 对数似然比')
ax.legend(fontsize=8)

ax = axes[1, 1]
xs = np.linspace(0.01, 0.99, 500)
epochs = [0, 20, 100, n_steps]
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(epochs)))

for epoch, col in zip(epochs, colors):
    k_e = int(np.sum(sim['data'][:epoch]))
    pdf1 = beta_dist.pdf(xs, a1 + k_e, b1 + epoch - k_e)
    pdf2 = beta_dist.pdf(xs, a2 + k_e, b2 + epoch - k_e)
    ax.plot(xs, pdf1, color=col, lw=2, ls='-')
    ax.plot(xs, pdf2, color=col, lw=2, ls='--')

ax.axvline(p_true, color='black', lw=1.0, ls=':', label=f'$p^*={p_true}$')
ax.set_xlabel('$p$')
ax.set_ylabel('后验密度')
ax.set_title('(d) 后验密度')

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color='black', lw=2, label='主体 1'),
    Line2D([0], [0], color='black', lw=2, ls='--', label='主体 2'),
]
for epoch, col in zip(epochs, colors):
    handles.append(Line2D([0], [0], color=col, lw=2, label=f'$n={epoch}$'))
handles.append(
    Line2D([0], [0], color='black', lw=1.0, ls=':', label=f'$p^*={p_true}$')
)
ax.legend(handles=handles, fontsize=8)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()

p_hat = sim['data'].sum() / n_steps
print(f"n = {n_steps} 时的 k_n/n: {p_hat:.3f}")
print(f"n = {n_steps} 时的 d_n: {sim['tv_beta'][-1]:.3f}")
print(f"n = {n_steps} 时的 log Z_n: {sim['log_Z'][-1]:.2f}")
print(f"7 log((1 - k_n/n) / (k_n/n)): "
      f"{7 * np.log((1 - p_hat) / p_hat):.2f}")
print(f"7 log((1 - p*) / p*): {log_Z_limit:.2f}")
```

四个面板显示：

- 面板 (a)：从 $\hat{p}_1^0 \approx 0.11$ 和
  $\hat{p}_2^0 \approx 0.89$ 出发，两个主体的预测概率都
  收敛到 $p^* = 0.65$。
- 面板 (b)：全变差距离 $d_n$ 从 $0.99$ 降到 $n = 600$ 时的 $0.23$。
- 面板 (c)：对数似然比 $\log Z_n$ 稳定在 $7 \log((1-p^*)/p^*) = -4.33$ 附近，即 $\bar p = p^*$ 时的闭式极限。
  在 $n = 600$ 处，该路径的 $k_n/n = 0.643$，$\log Z_n = -4.07$，而 $7 \log((1 - k_n/n)/(k_n/n)) = -4.13$。
- 面板 (d)：两个主体的后验贝塔密度起初相距
  甚远（一个接近 0，一个接近 1），并逐渐集中到以真值为中心的相同
  分布。

### 跨多条路径的几乎必然收敛

为了说明该定理的几乎必然特性，我们运行许多独立重复。

每次重复都从主体 1 的先验 $\pi_1 = \mathrm{Beta}(1, 8)$ 中抽取 $p$，然后抽取独立同分布的伯努利$(p)$ 序列 $x_1, x_2, \ldots$。

于是每条路径都是 $P_1$ 的一个抽样。

取 $Q = P_1$、$P = P_2$，定理表明 $d_n \to 0$ 在 $Q$ 几乎每条路径上成立。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      从 $P_1$ 抽取的多条样本路径的几乎必然融合。
      左面板绘制全变差距离，右面板绘制对数似然比 $\log Z_n$。
      线条颜色记录了每条路径上抽取的 $p$ 值；散点标出极限值 $7 \log((1-p)/p)$。
    name: fig-merging-of-opinions-many-paths
---
N_paths = 80
n_steps = 2000

rng_p = np.random.default_rng(1234)
p_draws = rng_p.beta(a1, b1, N_paths)    # p 从主体 1 的先验中抽取

tv_all   = np.empty((N_paths, n_steps + 1))
logZ_all = np.empty((N_paths, n_steps + 1))
steps    = np.arange(n_steps + 1)

for i in range(N_paths):
    s = run_simulation(p_draws[i], a1, b1, a2, b2, n_steps, seed=i)
    tv_all[i]   = s['tv_beta']
    logZ_all[i] = s['log_Z']

log_Z_limits = 7 * np.log((1 - p_draws) / p_draws)
norm_p = plt.Normalize(0, p_draws.max())
cmap = plt.cm.viridis

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax_tv, ax_log = axes

for i in range(N_paths):
    col = cmap(norm_p(p_draws[i]))
    ax_tv.semilogy(steps, tv_all[i], color=col, lw=0.8, alpha=0.5)
    ax_log.plot(steps, logZ_all[i], color=col, lw=0.8, alpha=0.5)
ax_tv.semilogy(steps, tv_all.mean(axis=0),
               color='black', lw=2, label='跨路径均值')
ax_tv.set_xlabel('观察数 $n$')
ax_tv.set_ylabel(r'$d_n$（对数刻度）')
ax_tv.legend()

ax_log.scatter(np.full(N_paths, n_steps), log_Z_limits,
               c=p_draws, cmap=cmap, norm=norm_p, s=12, zorder=3,
               label=r'极限 $7 \log((1-p)/p)$')
ax_log.axhline(0, color='gray', lw=0.8, ls=':')
ax_log.set_xlabel('观察数 $n$')
ax_log.set_ylabel(r'$\log Z_n$')
ax_log.legend()

fig.colorbar(plt.cm.ScalarMappable(norm=norm_p, cmap=cmap),
             ax=axes, label='从 $\\pi_1$ 抽取的 $p$')
plt.show()

for n in (500, n_steps):
    print(f"n = {n}: d_n < 0.30 的路径比例 = "
          f"{np.mean(tv_all[:, n] < 0.30):.2f}, "
          f"平均 d_n = {tv_all[:, n].mean():.3f}")
slow = np.argsort(tv_all[:, -1])[-5:]
print(f"五条最慢路径上的 p: {np.round(p_draws[slow], 4)}")
```

在每条路径上，$d_{2000}$ 都低于 $d_0$，但在许多路径上 $d_n$ 下降得很慢。

在 $n = 500$ 时，14% 的路径满足 $d_n < 0.30$，$d_n$ 的均值为 $0.489$；在 $n = 2000$ 时，这两个数字分别为 68% 和 $0.299$。

最慢的路径上 $p$ 低于 $0.004$。

主体 2 的先验密度 $8p^7$ 在 $p = 0$ 附近极小，因此主体 2 需要大量观察才能将后验质量移到那里。

这些有限视界的数值与定理一致，因为几乎必然收敛是一个渐近陈述。

(mo_supermartingale)=
### $d_n$ 是上鞅吗？

该定理的证明并未用到 $\{d_n\}$ 的任何单调性。

一般而言，$\{d_n\}$ 既不是 $P$-上鞅，也不是 $Q$-上鞅。

一个五点例子说明了这一点。

令 $\Omega = \{1, 2, 3, 4, 5\}$，令 $P$ 为均匀分布，并令 $Z_\infty = dQ/dP = (0, 1, 1, 1, 2)$。

第一次观察揭示 $\omega \in \{1, 2\}$ 还是 $\omega \in \{3, 4, 5\}$，第二次观察则揭示出 $\omega$。

```{code-cell} ipython3
P_pts = np.full(5, 0.2)
Q_pts = np.array([0, 1, 1, 1, 2]) * P_pts
blocks = [[0, 1], [2, 3, 4]]      # 第一次观察所揭示的单元格

d_0 = 0.5 * np.abs(P_pts - Q_pts).sum()
E_Q_d1 = E_P_d1 = 0.0
for b in blocks:
    P_b, Q_b = P_pts[b].sum(), Q_pts[b].sum()
    d_1 = 0.5 * np.abs(P_pts[b] / P_b - Q_pts[b] / Q_b).sum()
    E_Q_d1 += Q_b * d_1
    E_P_d1 += P_b * d_1

print(f"d_0 = {d_0:.3f}, E_Q[d_1] = {E_Q_d1:.3f}, E_P[d_1] = {E_P_d1:.3f}")
```

此处 $d_0 = 0.200$，$\mathbb{E}_Q[d_1] = 0.233$，$\mathbb{E}_P[d_1] = 0.300$。

第一次观察提高了两种测度下的期望距离。

第二次观察揭示出 $\omega$，因此 $d_2 = 0$。

恒等式 $2\,\mathbb{E}_Q[d_n] = \mathbb{E}_P[|Z_\infty - Z_n|]$ 指明了问题的根源：$Z_\infty$ 与其关于 $\mathscr{F}_n$ 的条件期望之间的 $L^1$ 距离不必随 $n$ 递减。

当 $\mathbb{E}_P[Z_\infty^2] < \infty$ 时，$L^2$ 距离确实递减，这是因为 $\mathbb{E}_P[(Z_\infty - Z_n)^2] = \mathbb{E}_P[Z_\infty^2] - \mathbb{E}_P[Z_n^2]$，而 $\mathbb{E}_P[Z_n^2]$ 关于 $n$ 不减。

在使用上述先验的贝塔-伯努利模型中，不等式 $\mathbb{E}[d_{n+1} \,|\, \mathscr{F}_n] \leq d_n$ 在我们检验的每个状态上都成立。

下一个单元格对每个状态 $(n, k_n)$（$n < 400$）精确计算 $\mathbb{E}_Q[d_{n+1} \,|\, \mathscr{F}_n] - d_n$ 和 $\mathbb{E}_P[d_{n+1} \,|\, \mathscr{F}_n] - d_n$，使用的是每个主体的单步预测概率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      贝塔-伯努利模型中 $d_n$ 的精确单步条件增量。
      左面板针对若干 $n$ 绘制 $\mathbb{E}_Q[d_{n+1} | \mathscr{F}_n] - d_n$ 对 $k_n/n$ 的曲线。
      右面板绘制每个 $n$ 下所有 $k_n$ 中的最大增量。
    name: fig-merging-of-opinions-supermartingale
---
def conditional_increments(n, a1, b1, a2, b2):
    """
    E_Q[d_{n+1} | n, k] - d_n 和 E_P[d_{n+1} | n, k] - d_n
    对 k = 0, ..., n，其中 Q = P_1，P = P_2。
    """
    k = np.arange(n + 1)
    d_now  = tv_distance_beta(a1 + k, b1 + n - k, a2 + k, b2 + n - k)
    d_head = tv_distance_beta(a1 + k + 1, b1 + n - k,
                              a2 + k + 1, b2 + n - k)
    d_tail = tv_distance_beta(a1 + k, b1 + n - k + 1,
                              a2 + k, b2 + n - k + 1)
    q1 = (a1 + k) / (a1 + b1 + n)        # 主体 1 的预测概率
    q2 = (a2 + k) / (a2 + b2 + n)        # 主体 2 的预测概率
    inc_Q = q1 * d_head + (1 - q1) * d_tail - d_now
    inc_P = q2 * d_head + (1 - q2) * d_tail - d_now
    return inc_Q, inc_P

n_check = 400
max_inc_Q = np.empty(n_check)
max_inc_P = np.empty(n_check)
for n in range(n_check):
    inc_Q, inc_P = conditional_increments(n, a1, b1, a2, b2)
    max_inc_Q[n], max_inc_P[n] = inc_Q.max(), inc_P.max()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
for n, col in zip([20, 100, 399], ['steelblue', 'darkorange', 'firebrick']):
    inc_Q, _ = conditional_increments(n, a1, b1, a2, b2)
    ax.plot(np.arange(n + 1) / n, inc_Q, color=col, lw=2, label=f'$n={n}$')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.set_xlabel('$k_n / n$')
ax.set_ylabel(r'$\mathbb{E}_Q[d_{n+1} | \mathscr{F}_n] - d_n$')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(max_inc_Q, color='purple', lw=2, label='在 $Q = P_1$ 下')
ax.plot(max_inc_P, color='gray', lw=2, ls='--', label='在 $P = P_2$ 下')
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel('观察数 $n$')
ax.set_ylabel(r'对 $k_n$ 的最大增量')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"Q 下所有状态中的最大增量: {max_inc_Q.max():.2e}")
print(f"P 下所有状态中的最大增量: {max_inc_P.max():.2e}")
```

每个增量都为负，所以在这个模型中，$\{d_n\}$ 在我们检验的状态范围内，在 $P_1$ 和 $P_2$ 下都表现得像一个上鞅。

右面板中的两条曲线重合，这是因为 $\pi_2(p) = \pi_1(1-p)$：交换正面与反面即交换了两个主体。

上鞅性质是该模型所特有的；五点例子表明它在一般情形下并不成立。

## 融合的失败：相互奇异

当假设 $Q \ll P$ 失败时会发生什么？

奇异情形是最简洁的反例。

### 点质量先验

假设两个主体都持有退化（点质量）先验：

- 主体 P：确信 $p = p_P = 0.30$。
- 主体 Q：确信 $p = p_Q = 0.75$。

由于 $P$ 仅对经验频率收敛到 $0.30$ 的序列赋值，而 $Q$ 仅对经验频率收敛到 $0.75$ 的序列赋值，这两个测度相互奇异：$P \perp Q$。

条件分布不更新，因为两个主体都已经确信各自的模型。

对于定理的对象，即整个未来路径的条件律，

$$
\|P(\,\cdot\,|\,\mathscr{F}_n) - Q(\,\cdot\,|\,\mathscr{F}_n)\|_{\mathrm{TV}}
= \|P - Q\|_{\mathrm{TV}} = 1
\quad \text{对所有 } n.
$$

这个等式成立是因为具有不同成功概率的无限乘积伯努利测度是奇异的。

如果我们只向前看一步，预测距离为 $|p_P - p_Q| = 0.45$。

这小于一，但它不是出现在布莱克韦尔-杜宾斯中的量。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      奇异先验下融合的失败。
      整个未来路径距离保持在一，
      而单步预测差距保持
      在 $|p_P - p_Q|$。
    name: fig-merging-of-opinions-singular-priors
---
p_P = 0.30
p_Q = 0.75
n_steps = 500

tv_singular_full = np.ones(n_steps + 1)
tv_singular_1step = np.full(n_steps + 1, np.abs(p_P - p_Q))

sim_abs_cont = run_simulation(
    p_Q, 1.0, 8.0, 8.0, 1.0, n_steps, seed=1
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(n_steps + 1), tv_singular_full,
        color='firebrick', lw=2,
        label=r'奇异：整路径 $d_n = 1$')
ax.plot(np.arange(n_steps + 1), tv_singular_1step,
        color='gray', lw=2, ls=':',
        label=r'单步差距 $= |p_P - p_Q|$')
ax.plot(np.arange(n_steps + 1),
        sim_abs_cont['tv_beta'],
        color='steelblue', lw=2,
        label=(r'$\mathrm{Beta}(1,8)$ vs'
               r' $\mathrm{Beta}(8,1)$'))
ax.set_xlabel('观察数 $n$')
ax.set_ylabel(r'$d_n$')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
```

对比是鲜明的。

对于相互绝对连续的先验，在由 $p = 0.75$ 的硬币生成的路径上，$d_n$ 衰减到零。

对于奇异的点质量先验，整个未来路径距离永远保持在一。

更多的数据无法调和这两个主体，因为每个主体都排除了对方赋予正概率的路径。

## 角谷定理：融合何时成立？

一个自然的问题是：对于哪些乘积测度，布莱克韦尔-杜宾斯
假设 $Q \ll P$ 成立？

对于无限乘积测度，答案由
{cite:t}`kakutani1948` 的经典结果给出。

### 海林格亲和度

在本节中，$P^{(m)}$ 和 $Q^{(m)}$ 表示信号空间 $(S, \mathscr{S})$ 上的概率测度，即乘积测度的因子。

符号 $P_n$ 和 $Q_n$ 保持其早先的含义：$P$ 和 $Q$ 限制在 $\mathscr{F}_n$ 上。

```{prf:definition} 海林格亲和度
:label: hellinger_affinity

对于 $(S, \mathscr{S})$ 上具有共同
支配测度 $\lambda$ 的概率测度 $P^{(m)}$ 和 $Q^{(m)}$，**海林格亲和度**为

$$
\rho_m = \int_S \sqrt{\frac{dP^{(m)}}{d\lambda} \cdot \frac{dQ^{(m)}}{d\lambda}}\,d\lambda
\;\in\; [0, 1].
$$

$\rho_m = 1$ 当且仅当 $P^{(m)} = Q^{(m)}$；$\rho_m = 0$ 当且仅当 $P^{(m)} \perp Q^{(m)}$。
```

海林格亲和度就是巴氏系数，即 {doc}`divergence_measures` 中 $\phi = 1/2$ 处的切尔诺夫系数。

对于两个特定的一维族：

- 高斯：$P^{(m)} = \mathcal{N}(0, 1)$ vs $Q^{(m)} = \mathcal{N}(\mu_m,1)$：

$$
\rho_m^{\text{Gauss}} = \exp\!\left(-\frac{\mu_m^2}{8}\right).
$$

- 伯努利：$P^{(m)} = \mathrm{Bernoulli}(p)$ vs $Q^{(m)} = \mathrm{Bernoulli}(q)$：

$$
\rho_m^{\text{Bern}} = \sqrt{pq} + \sqrt{(1-p)(1-q)}.
$$

### 角谷二分法

```{prf:theorem} 角谷（1948）
:label: kakutani_dichotomy

设 $P = \bigotimes_{m=1}^\infty P^{(m)}$ 和 $Q = \bigotimes_{m=1}^\infty Q^{(m)}$
是无限乘积测度，其因子逐对等价：对于每个 $m$，$P^{(m)} \sim Q^{(m)}$。

那么要么 $P \sim Q$，要么 $P \perp Q$；不存在
中间情形。

具体地，

$$
P \sim Q
\quad \iff \quad
\prod_{m=1}^\infty \rho_m > 0
\quad \iff \quad
\sum_{m=1}^\infty (1 - \rho_m) < \infty.
$$

如果 $\prod_{m=1}^\infty \rho_m = 0$，那么 $P \perp Q$。

*证明思路。*
一个标准证明研究似然比鞅
$Z_N = \prod_{m=1}^N (dQ^{(m)}/dP^{(m)})$ 连同恒等式
$\mathbb{E}_P[\sqrt{Z_N}] = \prod_{m=1}^N \rho_m$。

如果乘积为正，则 $\{\sqrt{Z_N}\}$ 在 $L^2(P)$ 中是柯西序列，因此 $Z_N$ 在 $L^1(P)$ 中收敛，且 $Q \ll P$；将 $P$ 和 $Q$ 互换，同样的论证给出 $P \ll Q$。

如果乘积为零，由法图引理得 $\mathbb{E}_P[\sqrt{Z_\infty}] = 0$，因此 $Z_\infty = 0$ $P$-a.s. 且 $P \perp Q$。

$\square$
```

### 对融合的蕴含

对于独立同分布类型的序列，角谷定理给出以下图景：

| 情形 | $\sum_m (1-\rho_m)$ | 结论 | 融合？ |
|---|---|---|---|
| 对所有 $m$，$P^{(m)} = Q^{(m)}$ | $0$ | $P = Q$ | 平凡地是 |
| $P^{(m)} \ne Q^{(m)}$ 且 $\sum_m (1-\rho_m) < \infty$ | 有限 | $P \sim Q$ | 是；布莱克韦尔-杜宾斯适用 |
| 对所有 $m$，$P^{(m)} = f \ne g = Q^{(m)}$ | $\infty$ | $P \perp Q$ | 否 |

具有不同固定边缘分布的独立同分布情形是标准的不融合例子。

如果两个主体对每个观察永久地赋予不同的分布，他们最终处于互不相交的概率世界中。

{doc}`likelihood_ratio_process` 展示了当自然永久地从两个固定密度之一抽取时，似然比过程收敛到零。

对于乘积测度，给定 $\mathscr{F}_n$ 条件下未来的条件分布不依赖于过去。

因此 $d_n = \bigl\|\bigotimes_{m>n} P^{(m)} - \bigotimes_{m>n} Q^{(m)}\bigr\|_{\mathrm{TV}}$ 是确定性的。

每个 $\rho_m$ 都为正，因为各因子是等价的，因此去掉有限多个因子不会改变 $\prod_m \rho_m$ 是否为正。

当 $P \perp Q$ 时，$m > n$ 上的乘积是奇异的，且对每个 $n$ 都有 $d_n = 1$。

当 $P \sim Q$ 时，由布莱克韦尔-杜宾斯定理知 $d_n \to 0$。

{ref}`mo_ex_gauss` 为高斯乘积精确计算了 $d_n$。

### 一个高斯乘积测度例子

我们用高斯乘积测度来说明角谷二分法。

取 $P = \mathcal{N}(0,1)^{\otimes\mathbb{N}}$ 作为参考测度，取 $Q = \bigotimes_m \mathcal{N}(\mu_m,1)$ 作为备择。

那么 $\log Z_N = \sum_{m=1}^N (\mu_m x_m - \mu_m^2/2)$，下面的代码从 $\mathcal{N}(0, 1)$，即在 $P$ 下抽取 $x_m$。

$\mu_m$ 的三种选择：

1. $\mu_m = \mu > 0$ 常数（$\sum (1-\rho_m) = \infty$）$\Rightarrow P \perp Q$。
2. $\mu_m = c/\!\sqrt{m}$（$\sum (1-\rho_m) \approx \sum c^2/(8m) = \infty$）$\Rightarrow P \perp Q$。
3. $\mu_m = c/m$（$\sum (1-\rho_m) \approx \sum c^2/(8m^2) < \infty$）$\Rightarrow P \sim Q$。

```{code-cell} ipython3
N_max  = 2000
ms     = np.arange(1, N_max + 1)
c      = 2.0
N_plot = 400
rng    = np.random.default_rng(0)

cases = [
    (r'$\mu_m = c$（常数）', np.full(N_max, c)),
    (r'$\mu_m = c/\sqrt{m}$', c / np.sqrt(ms)),
    (r'$\mu_m = c/m$', c / ms),
]
```

在常数漂移下，$\log Z_N$ 在 $P$ 下漂移到 $-\infty$，因此 $Z_N \to 0$ $P$-a.s. 且 $P \perp Q$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      常数漂移 $\mu_m = c$：
      似然比坍缩（$P \perp Q$）。
    name: fig-kakutani-constant
---
label, μ_seq = cases[0]
x = rng.standard_normal(N_plot)
log_Z_inc = μ_seq[:N_plot] * x - μ_seq[:N_plot]**2 / 2
log_Z = np.concatenate([[0], np.cumsum(log_Z_inc)])

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(N_plot + 1), log_Z,
        color='darkorange', lw=2, label=label)
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel('视界 $N$')
ax.set_ylabel(r'$P$ 下的 $\log Z_N$')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

$\mu_m = c/\sqrt{m}$ 的情形显示出相同的定性图景：尽管漂移消失，但消失得太慢。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      漂移 $\mu_m = c/\sqrt{m}$：仍然
      奇异（$P \perp Q$）。
    name: fig-kakutani-sqrt
---
label, μ_seq = cases[1]
x = rng.standard_normal(N_plot)
log_Z_inc = μ_seq[:N_plot] * x - μ_seq[:N_plot]**2 / 2
log_Z = np.concatenate([[0], np.cumsum(log_Z_inc)])

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(N_plot + 1), log_Z,
        color='purple', lw=2, label=label)
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel('视界 $N$')
ax.set_ylabel(r'$P$ 下的 $\log Z_N$')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

只有在 $\mu_m = c/m$ 时 $\sum (1-\rho_m) < \infty$ 才成立，因此似然比保持非退化且 $P \sim Q$。

布莱克韦尔-杜宾斯仅在此情形下适用。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      漂移 $\mu_m = c/m$：似然
      比稳定（$P \sim Q$）。
    name: fig-kakutani-inv-n
---
label, μ_seq = cases[2]
x = rng.standard_normal(N_plot)
log_Z_inc = μ_seq[:N_plot] * x - μ_seq[:N_plot]**2 / 2
log_Z = np.concatenate([[0], np.cumsum(log_Z_inc)])

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(N_plot + 1), log_Z,
        color='steelblue', lw=2, label=label)
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_xlabel('视界 $N$')
ax.set_ylabel(r'$P$ 下的 $\log Z_N$')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## 推广到连续时间

同样的逻辑推广到连续时间。

### 吉尔萨诺夫定理与似然比过程

在典范维纳空间上，$P$ 为维纳测度（标准
布朗运动 $W$），假设主体 $Q$ 相信该过程有一个
额外漂移 $\theta = \{\theta_s\}_{s \geq 0}$：

$$
W_t = \widetilde{W}_t + \int_0^t \theta_s\, ds,
$$

其中 $\widetilde{W}$ 是一个 $Q$-布朗运动。

吉尔萨诺夫-卡梅隆-马丁定理 {cite:p}`girsanov1960` 将
似然比过程 $Z_t = dQ_t/dP_t$ 给出为随机指数

$$
Z_t
= \exp\!\left(\int_0^t \theta_s\, dW_s - \frac{1}{2}\int_0^t \theta_s^2\, ds\right).
$$

$Z_t$ 始终是一个非负的 $P$-局部鞅；它是一个真鞅
当且仅当对所有 $t$，$\mathbb{E}_P[Z_t] = 1$。

诺维科夫条件 {cite:p}`novikov1972`，
对所有 $T$，$\mathbb{E}_P\!\left[\exp\!\left(\tfrac{1}{2}\int_0^T \theta_s^2\,ds\right)\right] < \infty$，
是充分的。

### 无穷远处的二分法

$[0,+\infty)$ 上的一个关键微妙之处是局部绝对连续性*不*蕴含 $\mathscr{F}_\infty$ 上的全局绝对连续性。

```{prf:remark} 无限视界的微妙之处
:label: dichotomy_at_infinity

假设 $Z_t$ 对每个有限视界都是一个真 $P$-鞅，并令 $Z_t \to Z_\infty$ $P$-a.s.

如果 $\{Z_t\}$ 在 $[0,\infty)$ 上一致可积，那么在 $\mathscr{F}_\infty$ 上 $Q \ll P$，且 $dQ/dP = Z_\infty$。

这正是布莱克韦尔-杜宾斯的假设条件，因此融合在 $Q$-a.s. 意义下成立。

如果进一步地 $Z_\infty > 0$ $P$-a.s.，那么 $P \ll Q$ 也成立，融合在 $P$-a.s. 意义下也成立。

$P$-a.s. 意义下的融合仅需要 $P \ll Q$，因为 $d_t$ 在 $P$ 和 $Q$ 之间是对称的，该定理在它们的角色互换后同样适用。

如果一致可积性失败，那么 $\mathscr{F}_\infty$ 上的全局绝对连续性可能失败。

在许多标准例子中，包括一个非零常数漂移，这些测度实际上在 $\mathscr{F}_\infty$ 上是奇异的。
```

对于确定性漂移 $\theta$，**能量条件**

$$
\int_0^\infty \theta_s^2\,ds < \infty
$$

区分了这两种情形。

如果该条件成立，$\mathbb{E}_P[Z_t^2] = \exp\!\left(\int_0^t \theta_s^2\, ds\right)$ 关于 $t$ 有界，$\{Z_t\}$ 一致可积，$Z_\infty > 0$，且 $\mathscr{F}_\infty$ 上 $P \sim Q$。

如果该条件不成立，$\log Z_t \to -\infty$ $P$-a.s.，且 $\mathscr{F}_\infty$ 上 $P \perp Q$。

非形式地说，$\int_0^\infty \theta_s^2\,ds$ 度量了在无限视界上区分两个测度的信息总量。

在能量条件下，布莱克韦尔-杜宾斯适用，融合在两个测度下都成立。

当 $\theta$ 是非零常数时，该条件失败，这些测度在 $\mathscr{F}_\infty$ 上奇异，融合不会发生。

{doc}`survival_recursive_preferences` 研究了信念与真实情况相差这种常数漂移的主体。

当 $\theta$ 依赖于 $W$ 的路径时，$\int_0^\infty \theta_s^2\,ds$ 是一个随机变量。

此时，$\mathscr{F}_\infty$ 上 $P \sim Q$ 当且仅当 $\int_0^\infty \theta_s^2\,ds < \infty$ 在 $P$-a.s. 和 $Q$-a.s. 意义下都成立。

在 $P$ 下，$\log Z_t = \int_0^t \theta_s\,dW_s - \tfrac{1}{2}\int_0^t \theta_s^2\,ds$ 在事件 $\{\int_0^\infty \theta_s^2\,ds = \infty\}$ 上趋于 $-\infty$。

在 $Q$ 下，$\log Z_t = \int_0^t \theta_s\,d\widetilde{W}_s + \tfrac{1}{2}\int_0^t \theta_s^2\,ds$ 在同一事件上趋于 $+\infty$。

在互补事件上，$\log Z_t$ 在两个测度下都收敛到一个有限极限。

由于 $\mathscr{F}_\infty$ 上 $Q \ll P$ 当且仅当 $Z_\infty < \infty$ $Q$-a.s.，而 $P \ll Q$ 当且仅当 $Z_\infty > 0$ $P$-a.s.，该结论由此得出。

一旦在 $\mathscr{F}_\infty$ 上建立了 $Q \ll P$，离散时间证明
即可直接套用。

条件距离满足 $d_t = \mathbb{E}_P[|Z_\infty - Z_t| \,|\, \mathscr{F}_t]/(2 Z_t)$，$Z_t \to Z_\infty > 0$ $Q$-a.s.，而第三步中的截断论证使分子趋于零。

恒等式 $\mathbb{E}_Q[d_t] = \tfrac{1}{2}\mathbb{E}_P[|Z_t - Z_\infty|]$ 也同样适用。

## 应用


### 贝叶斯学习

最直接的应用是贝叶斯推断。

假设数据 $(x_1, x_2, \ldots)$ 从真实测度 $Q^*$ 抽取。

一个主体在族 $\{Q_\theta : \theta \in \Theta\}$ 上持有先验 $\pi$，诱导出边缘 $P = \int Q_\theta\,\pi(d\theta)$。

如果 $Q^* \ll P$（即主体的边缘模型支配真值），那么布莱克韦尔-杜宾斯给出

$$
\bigl\|P(\,\cdot\,|\,x_1,\ldots,x_n) - Q^*(\,\cdot\,|\,x_1,\ldots,x_n)\bigr\|_{\mathrm{TV}}
\to 0 \quad Q^*\text{-a.s.}
$$

这是贝叶斯一致性的一种强形式：主体的预测在真实测度下与真值融合。

对真实参数邻域赋予正质量的先验通常保证对每个有限视界 $n$ 的*局部*绝对连续性 $Q^*_n \ll P_n$，但不保证布莱克韦尔-杜宾斯所要求的 $\mathscr{F}_\infty$ 上的全局条件 $Q^* \ll P$。

例如，在具有非原子先验 $\pi$ 的贝塔-伯努利模型中，混合 $P = \int \mathrm{Bernoulli}(p)^{\infty}\,\pi(dp)$ 对每个 $n$ 满足 $Q^*_n \ll P_n$，但全局上 $Q^* \not\ll P$，因为集合 $\{\lim k_n/n = p^*\}$ 的 $Q^*$-测度为一，但 $P$-测度为零（不同的伯努利乘积测度相互奇异）。

在一个可识别的独立同分布族中，不同 $\theta$ 对应的测度 $Q_\theta$ 相互奇异，因此 $Q_{\theta^*} \ll P$ 当且仅当 $\pi(\{\theta^*\}) > 0$。

因此，当 $\pi$ 在 $\theta^*$ 处有原子时，全局绝对连续性成立，例如当 $\Theta$ 为有限集时。

{doc}`likelihood_bayes` 研究了一个主体，其先验对两个模型中的每一个都赋予正概率，其中一个模型为真；在那里 $Q^* \ll P$。

在非原子先验下，主体对整个未来的预测永远不会与真值融合。

在贝塔-伯努利模型中，给出 $d_n$ 闭合形式的论证也给出 $\|P_i(\,\cdot\,|\,\mathscr{F}_n) - P_{p^*}(\,\cdot\,|\,\mathscr{F}_n)\|_{\mathrm{TV}} = 1$ 对每个 $n$ 成立，因为主体 $i$ 的后验在 $p^*$ 处没有原子。

每个主体的一步超前预测仍然收敛到 $p^*$，并且两个主体的预测彼此融合。

杜布一致性定理 {cite:p}`doob1949` 指出，对于 $\pi$-几乎每个 $\theta$，后验都集中于真实的 $\theta$。

{cite:t}`DiaconisFreedman1986` 研究在固定参数值处的一致性，这是杜布定理未涉及的情形。

他们表明，在无限维模型中，例外的 $\pi$-零测集可能很重要：在一个误差分布具有狄利克雷过程先验的位置问题中，后验在先验支撑中的某些参数值处可能不一致。

在 {doc}`mix_model` 中，自然从密度 $f$ 和 $g$ 的混合 $h$ 中独立同分布地抽取，而主体却相信自然是一劳永逸地从 $f$ 或 $g$ 中抽取的。

在那里 $P \perp Q^*$，主体的预测始终是错误的，并且主体的后验集中于在库尔贝克-莱布勒散度意义下更接近 $h$ 的模型。

### 理性预期与异质先验

在宏观经济学中，理性预期模型通常施加共同先验。

布莱克韦尔-杜宾斯为较弱的初始一致性提供了动态论证。

如果两个主体从等价先验出发并观察相同的历史，他们的条件预测最终在每个事件上达成一致。

{cite:t}`aumann1976` 的一致性定理是对这一结果的补充。

奥曼的主体共享共同先验，但看到不同的私有信息；如果他们对某个事件的后验概率是共同知识，那么这些后验概率必须相等。

布莱克韦尔和杜宾斯的主体持有不同的先验，但看到相同的信息。

当融合失败时，分歧会持续存在并产生经济后果。

{doc}`likelihood_ratio_process_2` 研究财富如何在持有不同信念的主体之间转移，其中包括一个先验排除了真实模型的主体。

{doc}`survival_recursive_preferences` 研究当信念永久存在分歧时，哪些主体能够生存下来。

### 遍历马尔可夫链

对于具有转移核 $\Pi$ 和两个初始
分布 $\mu$ 和 $\nu$ 的马尔可夫链，$n$-步分布为 $\mu\Pi^n$
和 $\nu\Pi^n$。

如果 $\Pi$ 是遍历的，具有唯一平稳分布
$\pi$，两者都收敛到 $\pi$，因此

$$
\|\mu\Pi^n - \nu\Pi^n\|_{\mathrm{TV}}
\leq \|\mu\Pi^n - \pi\|_{\mathrm{TV}} + \|\nu\Pi^n - \pi\|_{\mathrm{TV}}
\to 0.
$$

这种收敛*不*要求绝对连续性：$\mu$ 和 $\nu$ 可以是不同状态处的点质量。

它涉及的对象与布莱克韦尔-杜宾斯不同。

测度 $\mu\Pi^n$ 和 $\nu\Pi^n$ 是日期 $n$ 处状态的无条件分布，它们的收敛表明遍历链会遗忘其初始条件。

布莱克韦尔-杜宾斯涉及的是在共同观测历史给定条件下，对整个未来的条件分布。

对 $\Pi$ 持相同看法并观察当前状态的两个主体，会对未来做出完全相同的条件预测，无论 $\Pi$ 是否遍历。

## 融合的速率

布莱克韦尔-杜宾斯是定性的。

它告诉我们 $d_n \to 0$，但没有告诉我们速度有多快。

恒等式

$$
\mathbb{E}_Q[d_n] = \tfrac{1}{2}\,\mathbb{E}_P[|Z_n - Z_\infty|]
$$

表明 $Q$ 均值意义下的融合速率由似然比鞅的 $L^1(P)$ 收敛速率所支配。

在正则参数例子中，人们常常看到 $n^{-1/2}$ 型的行为。

在贝塔-伯努利模型中，前面推导出的正态近似给出 $d_n \approx 7/\sqrt{2 \pi n\, \hat p_n (1 - \hat p_n)}$。

在固定硬币 $P_{p^*}$ 下，$\hat p_n \to p^*$，因此 $\mathbb{E}_{P_{p^*}}[d_n] \approx C(p^*)/\sqrt{n}$，其中 $C(p^*) = 7/\sqrt{2\pi p^*(1-p^*)}$，$C(0.65) = 5.85$。

在 $Q = P_1$ 下，硬币 $p$ 是从 $\pi_1$ 中抽取的，对 $p$ 取平均给出 $\mathbb{E}_Q[d_n] \approx C_Q/\sqrt{n}$，其中

$$
C_Q = \frac{7}{\sqrt{2\pi}}\, \mathbb{E}_{\pi_1}\!\left[(p(1-p))^{-1/2}\right]
= \frac{7}{\sqrt{2\pi}}\, \frac{B(1/2,\, 15/2)}{B(1,\, 8)} = 14.7 。
$$

下面的单元格通过对成功次数 $k_n$ 求和来精确计算这两个期望：$k_n$ 在 $P_{p^*}$ 下服从二项分布，在 $Q = P_1$ 下服从贝塔-二项分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: |
      贝塔-伯努利模型中，在固定硬币 $P_{0.65}$ 下和在 $Q = P_1$ 下 $d_n$ 的精确均值。
      虚线为正态近似 $C/\sqrt{n}$。
    name: fig-merging-of-opinions-rate
---
def expected_d(n):
    """精确的 E_{P_{p*}}[d_n] 和 E_Q[d_n]，其中 Q = P_1。"""
    k = np.arange(n + 1)
    d = tv_distance_beta(a1 + k, b1 + n - k, a2 + k, b2 + n - k)
    return binom.pmf(k, n, p_true) @ d, betabinom.pmf(k, n, a1, b1) @ d

n_grid = np.unique(np.logspace(0, 4, 60).astype(int))
mean_fixed, mean_Q = np.array([expected_d(n) for n in n_grid]).T

C_fixed = 7 / np.sqrt(2 * np.pi * p_true * (1 - p_true))
C_Q = 7 / np.sqrt(2 * np.pi) * beta_fn(0.5, 7.5) / beta_fn(a1, b1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(n_grid, mean_fixed, color='steelblue', lw=2,
          label=r'$\mathbb{E}_{P_{0.65}}[d_n]$')
ax.loglog(n_grid, C_fixed / np.sqrt(n_grid), color='steelblue',
          lw=1.5, ls='--', label=rf'${C_fixed:.2f}/\sqrt{{n}}$')
ax.loglog(n_grid, mean_Q, color='firebrick', lw=2,
          label=r'$\mathbb{E}_Q[d_n]$, $Q = P_1$')
ax.loglog(n_grid, C_Q / np.sqrt(n_grid), color='firebrick',
          lw=1.5, ls='--', label=rf'${C_Q:.1f}/\sqrt{{n}}$')
ax.set_ylim(1e-2, 1.5)
ax.set_xlabel('样本大小 $n$')
ax.set_ylabel('$d_n$ 的均值')
ax.legend()
plt.tight_layout()
plt.show()

for n in (100, 500, 1000, 10_000):
    e_fixed, e_Q = expected_d(n)
    δ, s = 7 / (n + 9), np.sqrt(p_true * (1 - p_true) / n)
    print(f"n = {n:>6}: E_P0.65[d_n] = {e_fixed:.3f}, "
          f"normal approx = {2 * norm.cdf(δ / (2 * s)) - 1:.3f}, "
          f"C/sqrt(n) = {C_fixed / np.sqrt(n):.3f} | "
          f"E_Q[d_n] / (C_Q/sqrt(n)) = {e_Q / (C_Q / np.sqrt(n)):.2f}")

late = n_grid >= 1000
for name, m in (("P_0.65", mean_fixed), ("Q", mean_Q)):
    slope = np.polyfit(np.log(n_grid[late]), np.log(m[late]), 1)[0]
    print(f"log-log slope under {name}, 1000 <= n <= 10^4: {slope:.2f}")
```

在 $P_{0.65}$ 下，精确均值从下方逼近 $5.85/\sqrt{n}$。

在 $n = 500$ 时，精确均值为 $0.255$，在 $\hat p_n = 0.65$ 处的正态近似 $2\Phi(\delta_n/(2 s_n)) - 1$ 为 $0.253$，而 $5.85/\sqrt{n}$ 为 $0.262$。

在 $1000 \leq n \leq 10^4$ 范围内拟合的对数-对数斜率为 $-0.50$。

在 $Q$ 下，向 $14.7/\sqrt{n}$ 的逼近较慢。

在 $n = 500$ 时，$\mathbb{E}_Q[d_n]$ 与 $14.7/\sqrt{n}$ 之比为 $0.70$，在 $n = 10^4$ 时为 $0.92$，而在 $1000 \leq n \leq 10^4$ 范围内拟合的斜率为 $-0.43$。

先验 $\pi_1$ 将大部分质量集中在 $p = 0$ 附近，而正态近似要求 $n p$ 较大。

## 总结与推广

布莱克韦尔-杜宾斯定理背后的逻辑流程是：

$$
Q \ll P
\;\Longrightarrow\;
Z = \frac{dQ}{dP} \in L^1(P)
\;\Longrightarrow\;
Z_n = \mathbb{E}_P[Z \,|\, \mathscr{F}_n] \to Z
\text{ 且 }
\mathbb{E}_P[|Z - Z_n| \,|\, \mathscr{F}_n] \to 0
\;\; P\text{-几乎必然}
\;\Longrightarrow\;
d_n = \frac{\mathbb{E}_P[|Z - Z_n| \,|\, \mathscr{F}_n]}{2 Z_n} \xrightarrow{Q\text{-a.s.}} 0.
$$

要点：

1. 单边绝对连续性 $Q \ll P$ 给出 $Q$-几乎必然的融合。对于在*两个*测度下的融合，需要相互绝对连续性 $P \sim Q$。

2. 似然比鞅 $Z_n = \mathbb{E}_P[Z|\mathscr{F}_n]$ 驱动了这个结果：在 $d_n = \mathbb{E}_P[|Z - Z_n| \,|\, \mathscr{F}_n]/(2 Z_n)$ 中，分子收敛到零，而 $Z_n$ 以 $Q$-几乎必然的方式收敛到 $Z > 0$。

3. 更多的数据只能增加区分两个假设的能力：在 $\mathscr{F}_n$ 上的 $\|P_n - Q_n\|_{\mathrm{TV}}$ 是关于 $n$ 非递减的。条件距离 $d_n$ 未必递减，即使在期望意义上也是如此。当 $Q \ll P$ 时，它收敛到零。

4. 对于无限乘积测度，角谷定理给出一个尖锐的等价性与奇异性二分法：要么 $P \sim Q$（当 $\sum_m (1 - \rho_m) < \infty$ 时），要么 $P \perp Q$（当该和发散时），不存在中间情形。

5. 当 $P \sim Q$ 时，布莱克韦尔-杜宾斯适用，融合在两个测度下都发生；当 $P \perp Q$ 且对于每个 $n$ 都有 $P_n \sim Q_n$ 时，对于每个 $n$ 都有 $d_n = 1$，分歧永远持续。

### 经济学中的应用

一些有影响力的应用和推广是：

- {cite}`KalaiLehrer1993Nash`：当先验关于真值绝对连续时，重复博弈学习驱动策略趋向纳什行为。
- {cite}`KalaiLehrer1993Subjective`：在相同条件下，主观均衡和客观均衡渐近地重合。
- {cite}`KalaiLehrer1994Merging`：为完全全变差收敛过强的环境引入了弱和强融合概念。
- {cite}`KalaiLehrerSmorodinsky1999`：将融合与校准预测联系起来。
- {cite}`JacksonKalaiSmorodinsky1999`：将德菲内蒂式的表示与贝叶斯学习和后验收敛联系起来。
- {cite}`JacksonKalai1999`：社会学习削弱了依赖于跨群体持续分歧的声誉效应。
- {cite}`Sandroni1998Nash`：表明近似绝对连续性条件足以在重复博弈中实现纳什型收敛。
- {cite}`MillerSanchirico1999`：给出了一个替代证明，并从相互有利的赌注角度对持续分歧作出了经济学解释。
- {cite}`LehrerSmorodinsky1996Compatible`：研究超越布莱克韦尔-杜宾斯绝对连续性的更广泛的兼容性概念。
- {cite}`LehrerSmorodinsky1996Learning`：综述重复策略环境中的融合与学习。
- {cite}`Nyarko1994`：将绝对连续性下的贝叶斯学习与趋向相关均衡的收敛联系起来。
- {cite}`PomattoAlNajjarSandroni2014`：将定理推广到有限可加概率，并将融合与检验可操纵性联系起来。
- {cite}`AcemogluChernozhukovYildiz2016`：表明当主体对信号结构本身不确定时分歧如何持续。

## 练习

```{exercise}
:label: mo_ex_one_sided

设智能体 1 的先验为 $\pi_1 = \mathrm{Unif}(0, \tfrac12)$，智能体 2 的先验为 $\pi_2 = \mathrm{Unif}(0, 1)$，并令 $P_i = \int_0^1 P_p\, \pi_i(p)\, dp$，如贝塔-伯努利小节所示。

1. 证明 $P_1 \ll P_2$，但 $P_2 \ll P_1$ 不成立。
2. 证明 $d_n = \Pr_2(p > \tfrac12 \,|\, x^n)$，即智能体 2 关于 $p > \tfrac12$ 的后验概率。
3. 从 $P_2$ 中抽取路径：从 $\pi_2$ 中抽取 $p$，然后抽取独立同分布的伯努利 $(p)$ 变量 $x_1, x_2, \ldots$。
   沿每条路径计算并绘制 $d_n$，验证在 $p < \tfrac12$ 的路径上 $d_n \to 0$，在 $p > \tfrac12$ 的路径上 $d_n \to 1$。
4. 将第 3 部分的结果与布莱克韦尔-杜宾斯定理进行协调。

*提示：* 对于第 2 部分，智能体 1 的后验是智能体 2 的后验在 $p < \tfrac12$ 条件下的结果。
```

```{solution-start} mo_ex_one_sided
:class: dropdown
```

第 1 部分。

$\pi_1$ 的密度为 $2\cdot\mathbf{1}\{p < \tfrac12\} \leq 2\pi_2(p)$，因此对每个事件 $A$ 都有 $P_1(A) \leq 2 P_2(A)$，即 $P_1 \ll P_2$。

设 $\bar p = \limsup_n k_n/n$。

事件 $\{\bar p > \tfrac12\}$ 在 $P_2$ 下的概率为 $\tfrac12$，在 $P_1$ 下的概率为零，因此 $P_2 \ll P_1$ 不成立。

{ref}`mo_ex_zinf` 给出 $dP_1/dP_2 = \pi_1(\bar p)/\pi_2(\bar p) = 2\cdot\mathbf{1}\{\bar p < \tfrac12\}$。

第 2 部分。

智能体 2 在观察到 $x^n$ 后的后验为 $\mu_n = \mathrm{Beta}(k_n + 1, n - k_n + 1)$。

智能体 1 的后验密度正比于 $p^{k_n}(1-p)^{n-k_n}\mathbf{1}\{p < \tfrac12\}$，因此它等于 $\mu_n(\,\cdot\,|\,A)$，其中 $A = (0, \tfrac12)$。

对于概率测度 $\mu$、满足 $\mu(A) > 0$ 的事件 $A$，以及任意事件 $B$，

$$
\mu(B \,|\, A) - \mu(B) = \mu(B \cap A)\left(\frac{1}{\mu(A)} - 1\right) - \mu(B \cap A^c),
$$

该式位于 $-\mu(A^c)$ 和 $\mu(A^c)$ 之间，上界在 $B = A$ 时取得。

因此 $\|\mu(\,\cdot\,|\,A) - \mu\|_{\mathrm{TV}} = \mu(A^c)$。

贝塔-伯努利小节中的论证将 $d_n$ 等同于两个智能体后验之间的全变差距离，因此

$$
d_n = \mu_n\bigl((\tfrac12, 1)\bigr) = 1 - F_{\mathrm{Beta}(k_n+1,\, n-k_n+1)}(\tfrac12)。
$$

由于 $\bar p = p$ 在 $P_2$ 下几乎必然成立，故 $d_n = P_2(\bar p > \tfrac12 \,|\, \mathscr{F}_n)$。

因此 $\{d_n\}$ 是一个 $P_2$-鞅，由列维上升定理可得 $d_n \to \mathbf{1}\{\bar p > \tfrac12\}$（$P_2$ 几乎必然成立）。

第 3 部分。

```{code-cell} ipython3
n_steps_ex = 3000
N_paths_ex = 40
rng_ex = np.random.default_rng(0)
p_paths = rng_ex.uniform(0, 1, N_paths_ex)     # p drawn from π_2
steps_ex = np.arange(n_steps_ex + 1)

fig, ax = plt.subplots(figsize=(8, 4))
d_final = np.empty(N_paths_ex)
for i, p in enumerate(p_paths):
    x = rng_ex.binomial(1, p, n_steps_ex)
    k = np.concatenate([[0], np.cumsum(x)])
    d = beta_dist.sf(0.5, k + 1, steps_ex - k + 1)   # Pr_2(p > 1/2 | x^n)
    d_final[i] = d[-1]
    ax.semilogx(steps_ex[1:], d[1:], lw=0.8, alpha=0.6,
                color='firebrick' if p > 0.5 else 'steelblue')
ax.set_xlabel('observations $n$')
ax.set_ylabel('$d_n$')
ax.legend(handles=[
    Line2D([0], [0], color='steelblue', lw=2, label=r'$p < 1/2$'),
    Line2D([0], [0], color='firebrick', lw=2, label=r'$p > 1/2$'),
])
plt.tight_layout()
plt.show()

err = np.abs(d_final - (p_paths > 0.5))
print(f"largest |d_n - 1{{p > 1/2}}| at n = {n_steps_ex}: {err.max():.4f}, "
      f"on the path with p = {p_paths[err.argmax()]:.3f}")
```

每条路径最终都接近 $\mathbf{1}\{p > \tfrac12\}$；收敛较慢的路径对应的 $p$ 接近 $\tfrac12$。

第 4 部分。

令 $Q = P_1$，$P = P_2$，假设 $Q \ll P$ 成立，则该定理给出 $d_n \to 0$（$P_1$ 几乎必然成立）。

由于 $dP_1/dP_2 = 2\cdot\mathbf{1}\{\bar p < \tfrac12\}$，$P_1$ 是 $P_2$ 在 $\{\bar p < \tfrac12\}$ 条件下的结果。

因此该定理的结论为：在 $\{\bar p < \tfrac12\}$ 上 $d_n \to 0$（$P_2$ 几乎必然成立）。

该定理对 $\{\bar p > \tfrac12\}$ 不作说明，该事件在 $P_2$ 下的概率为 $\tfrac12$，在 $P_1$ 下的概率为零。

在该事件上 $d_n \to 1$：智能体 1 的先验排除了 $p > \tfrac12$ 的可能性，因此无论观察到何种数据都不会使智能体 1 的后验质量超过 $\tfrac12$。

```{solution-end}
```

```{exercise}
:label: mo_ex_gauss

设 $P = \mathcal{N}(0,1)^{\otimes \mathbb{N}}$，$Q = \bigotimes_m \mathcal{N}(\mu_m, 1)$，如高斯乘积测度示例所示，并令 $\sigma_n^2 = \sum_{m > n} \mu_m^2$。

1. 证明 $d_n = 2\Phi(\sigma_n/2) - 1$，当 $\sigma_n = \infty$ 时 $d_n = 1$。
2. 对于 $\mu_m = c/m$，证明在 $n$ 较大时 $d_n \approx c/\sqrt{2\pi n}$。
   在 $n = 100$、$c = 2$ 时比较精确值与近似值。
3. 对于 $\mu_m = c/\sqrt{m}$，证明对每个 $n$ 都有 $d_n = 1$，而两个智能体一步预测之间的距离 $\|\mathcal{N}(0,1) - \mathcal{N}(\mu_{n+1}, 1)\|_{\mathrm{TV}}$ 收敛于零。

{cite:t}`KalaiLehrer1994Merging` 将每个固定有限视界上预测的收敛称为**弱融合**，而将 $d_n$ 的收敛称为**强融合**。

第 3 部分给出了弱融合而非强融合的例子。
```

```{solution-start} mo_ex_gauss
:class: dropdown
```

第 1 部分。

对于乘积测度，未来独立于过去，因此 $d_n$ 是 $\bigotimes_{m>n} \mathcal{N}(0,1)$ 与 $\bigotimes_{m>n} \mathcal{N}(\mu_m,1)$ 之间的全变差距离。

假设 $\sigma_n < \infty$。

则 $\sum_{m>n} (1 - \rho_m) \leq \sum_{m>n} \mu_m^2/8 < \infty$，因此这两个尾部乘积是等价的，它们的似然比为

$$
L = \exp\!\left(S - \tfrac12 \sigma_n^2\right),
\qquad S = \sum_{m>n} \mu_m x_m 。
$$

在 $P$ 下，$S \sim \mathcal{N}(0, \sigma_n^2)$；在 $Q$ 下，$S \sim \mathcal{N}(\sigma_n^2, \sigma_n^2)$。

由于 $L$ 是 $S$ 的函数，尾部乘积之间的全变差距离等于 $S$ 的两个分布之间的全变差距离。

这两个正态密度在 $\sigma_n^2/2$ 处相交，因此

$$
d_n = \Phi\!\left(\frac{\sigma_n}{2}\right) - \Phi\!\left(-\frac{\sigma_n}{2}\right) = 2\Phi\!\left(\frac{\sigma_n}{2}\right) - 1。
$$

假设 $\sigma_n = \infty$。

若 $\mu_m \not\to 0$，则项 $1 - \rho_m = 1 - e^{-\mu_m^2/8}$ 不趋于零；若 $\mu_m \to 0$，则对较大的 $m$ 有 $1 - \rho_m \geq \mu_m^2/16$。

无论哪种情况，$\sum_{m>n}(1-\rho_m) = \infty$，由角谷定理可知尾部乘积是奇异的，且 $d_n = 1$。

第 2 部分。

对于 $\mu_m = c/m$，$\sigma_n^2 = c^2 \sum_{m>n} m^{-2}$，且 $1/(n+1) \leq \sum_{m>n} m^{-2} \leq 1/n$。

因此 $\sigma_n \approx c/\sqrt{n}$，由 $2\Phi(x) - 1 \approx x\sqrt{2/\pi}$（当 $x$ 较小时）可得 $d_n \approx c/\sqrt{2\pi n}$。

尾部和 $\sum_{m>n} m^{-2}$ 是三伽玛函数 $\psi'(n+1)$，可由 `polygamma(1, n + 1)` 计算。

第 3 部分。

对于 $\mu_m = c/\sqrt{m}$，对每个 $n$ 都有 $\sum_{m>n} c^2/m = \infty$，因此 $d_n = 1$。

一步预测距离为 $2\Phi(c/(2\sqrt{n+1})) - 1 \to 0$。

对任意固定的 $k$，两个智能体对 $(x_{n+1}, \ldots, x_{n+k})$ 预测之间的距离为 $2\Phi\bigl(\tfrac12 (\sum_{m=n+1}^{n+k} c^2/m)^{1/2}\bigr) - 1 \leq 2\Phi\bigl(c\sqrt{k}/(2\sqrt{n+1})\bigr) - 1 \to 0$。

智能体对任意固定数量的未来观测值的预测会融合，但它们对整个未来的预测永远不会融合。

```{code-cell} ipython3
c = 2.0
n_vals = np.arange(1, 1001)

σ_inv_m = c * np.sqrt(polygamma(1, n_vals + 1))     # μ_m = c/m
d_inv_m = 2 * norm.cdf(σ_inv_m / 2) - 1
d_inv_m_approx = c / np.sqrt(2 * np.pi * n_vals)
one_step_sqrt = 2 * norm.cdf(c / (2 * np.sqrt(n_vals + 1))) - 1   # μ_m = c/√m

fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(n_vals, d_inv_m, color='steelblue', lw=2,
          label=r'$d_n$, $\mu_m = c/m$')
ax.loglog(n_vals, d_inv_m_approx, color='steelblue', lw=1.5, ls='--',
          label=r'$c/\sqrt{2\pi n}$')
ax.loglog(n_vals, np.ones_like(n_vals, dtype=float), color='purple',
          lw=2, label=r'$d_n$, $\mu_m = c/\sqrt{m}$')
ax.loglog(n_vals, one_step_sqrt, color='purple', lw=1.5, ls=':',
          label=r'one-step distance, $\mu_m = c/\sqrt{m}$')
ax.set_xlabel('$n$')
ax.set_ylabel('total-variation distance')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

n = 100
print(f"n = {n}: exact d_n = {d_inv_m[n - 1]:.4f}, "
      f"approximation = {d_inv_m_approx[n - 1]:.4f}")
```

在 $n = 100$、$c = 2$ 时，精确距离为 $0.0795$，近似值为 $0.0798$。

对于 $\mu_m = c/\sqrt{m}$ 的一步预测距离几乎与 $\mu_m = c/m$ 的 $d_n$ 重合，因为 $\mu_{n+1}^2 = c^2/(n+1)$ 与 $c^2 \sum_{m>n} m^{-2}$ 都接近于 $c^2/n$。

```{solution-end}
```

```{exercise}
:label: mo_ex_zinf

设 $\pi_1$ 和 $\pi_2$ 是 $(0,1)$ 上的先验密度，满足只要 $\pi_1 > 0$ 则 $\pi_2 > 0$，设 $P_i = \int_0^1 P_p\,\pi_i(p)\,dp$，并令 $\bar p = \limsup_n k_n/n$。

1. 证明在 $\mathscr{F}$ 上 $dP_1/dP_2 = \pi_1(\bar p)/\pi_2(\bar p)$，因此 $Z_\infty = \pi_1(\bar p)/\pi_2(\bar p)$。
2. 对于先验 $\mathrm{Beta}(1,8)$ 和 $\mathrm{Beta}(8,1)$，从 $P_2$ 中模拟若干条路径。
   将大 $n$ 时的 $\log Z_n$ 与 $7\log((1-\hat p_n)/\hat p_n)$ 进行比较，其中 $\hat p_n = k_n/n$；并与 $7\log((1-p)/p)$ 进行比较，其中 $p$ 是该路径上抽取的硬币参数。
```

```{solution-start} mo_ex_zinf
:class: dropdown
```

第 1 部分。

在 $P_p$ 下，强大数定律给出 $\bar p = p$ 几乎必然成立。

设在 $\pi_2 = 0$ 处 $\pi_1/\pi_2 = 0$。

对每个 $A \in \mathscr{F}$，

$$
\begin{aligned}
P_1(A)
&= \int_0^1 P_p(A)\, \pi_1(p)\, dp
= \int_0^1 \mathbb{E}_{P_p}[\mathbf{1}_A]\, \frac{\pi_1(p)}{\pi_2(p)}\, \pi_2(p)\, dp \\
&= \int_0^1 \mathbb{E}_{P_p}\!\left[\mathbf{1}_A\, \frac{\pi_1(\bar p)}{\pi_2(\bar p)}\right] \pi_2(p)\, dp
= \mathbb{E}_{P_2}\!\left[\mathbf{1}_A\, \frac{\pi_1(\bar p)}{\pi_2(\bar p)}\right]。
\end{aligned}
$$

第二个等式使用了只要 $\pi_1 > 0$ 则 $\pi_2 > 0$ 的条件。

第三个等式将 $p$ 替换为 $\bar p$，两者在 $P_p$ 下几乎必然相等。

因此 $\pi_1(\bar p)/\pi_2(\bar p)$ 是 $P_1$ 关于 $P_2$ 在 $\mathscr{F}$ 上的一个密度，由 {prf:ref}`martingale_convergence` 可得 $Z_\infty = \pi_1(\bar p)/\pi_2(\bar p)$。

拉普拉斯近似解释了 $Z_n$ 的收敛性。

比值 $Z_n = \int_0^1 p^{k_n}(1-p)^{n-k_n}\pi_1(p)\,dp \,/\, \int_0^1 p^{k_n}(1-p)^{n-k_n}\pi_2(p)\,dp$ 对两个先验都用一个集中于 $\hat p_n$ 附近的似然进行加权，因此 $Z_n \approx \pi_1(\hat p_n)/\pi_2(\hat p_n)$。

第 2 部分。

```{code-cell} ipython3
n_ex = 20_000
rng_z = np.random.default_rng(5)

print(f"{'p':>6} {'p_hat':>7} {'log Z_n':>9} "
      f"{'gap to p_hat formula':>21} {'gap to p formula':>17}")
for j in range(6):
    p = rng_z.beta(a2, b2)                     # p drawn from π_2
    data = rng_z.binomial(1, p, n_ex)
    log_Z_n = log_likelihood_ratio(data, a1, b1, a2, b2)[-1]
    p_hat = data.mean()
    gap_hat = log_Z_n - 7 * np.log((1 - p_hat) / p_hat)
    gap_true = log_Z_n - 7 * np.log((1 - p) / p)
    print(f"{p:6.3f} {p_hat:7.4f} {log_Z_n:9.3f} "
          f"{gap_hat:21.3f} {gap_true:17.3f}")
```

在 $n = 20{,}000$ 时，这些路径上 $\log Z_n$ 与 $7\log((1-\hat p_n)/\hat p_n)$ 之间的差值至多为 $0.10$。

该差值是拉普拉斯近似的 $O(1/n)$ 误差，在 $p$ 最接近 1 的路径上该差值最大，因为此时 $\log(\pi_1/\pi_2)$ 的导数较大。

$\log Z_n$ 与 $7\log((1-p)/p)$ 之间的差值可达 $0.25$，这是因为 $\hat p_n$ 与 $p$ 之间相差 $O(n^{-1/2})$；随着 $\hat p_n \to p$，该差值趋于消失。

在该讲座第一张图所示的路径上，$\log Z_{600} = -4.07$，而 $7\log((1-\hat p_{600})/\hat p_{600}) = -4.13$。

```{solution-end}
```
