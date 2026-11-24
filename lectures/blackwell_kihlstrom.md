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
  title: 布莱克韦尔的实验比较定理
  headings:
    Overview: 概述
    Experiments and stochastic transformations: 实验与随机变换
    Experiments and stochastic transformations::The state space and experiments: 状态空间与实验
    Experiments and stochastic transformations::Stochastic transformations: 随机变换
    Three equivalent criteria: 三个等价标准
    'Three equivalent criteria::Criterion 1: the economic criterion': 标准 1：经济学标准
    'Three equivalent criteria::Criterion 2: the sufficiency criterion': 标准 2：充分性标准
    'Three equivalent criteria::Criterion 3: the uncertainty criterion': 标准 3：不确定性标准
    The main theorem: 主定理
    Kihlstrom's Bayesian interpretation: 基尔斯特罗姆的贝叶斯解释
    Kihlstrom's Bayesian interpretation::Posteriors and standard experiments: 后验与标准实验
    Kihlstrom's Bayesian interpretation::Mean-preserving spreads and Blackwell's order: 均值保持展开与布莱克韦尔序
    Kihlstrom's Bayesian interpretation::Convex order and second-order stochastic dominance: 凸序与二阶随机占优
    Simulating the Blackwell order with many states: 用多状态模拟布莱克韦尔序
    The DeGroot uncertainty function: 德格鲁特不确定性函数
    The DeGroot uncertainty function::Concave uncertainty functions and the value of information: 凹不确定性函数与信息的价值
    The DeGroot uncertainty function::Shannon entropy as a special case: 作为特例的香农熵
    The DeGroot uncertainty function::Value of information as a function of experiment quality: 作为实验质量函数的信息价值
    'Application 1: demand for information about product quality': 应用1：对产品质量信息的需求
    'Application 2: sequential experimental design': 应用 2：序贯实验设计
    The Data Processing Inequality and Coarse-Graining: 数据处理不等式与粗粒化
    The Data Processing Inequality and Coarse-Graining::The DPI for f-divergences: f-散度的 DPI
    The Data Processing Inequality and Coarse-Graining::Connection to Blackwell's sufficiency condition: 与布莱克韦尔充分性条件的联系
    'The Data Processing Inequality and Coarse-Graining::Information geometry: Chentsov''s theorem': 信息几何：陈氏定理
    The Data Processing Inequality and Coarse-Graining::The information bottleneck in machine learning: 机器学习中的信息瓶颈
    The Data Processing Inequality and Coarse-Graining::Summary of the DPI–Blackwell correspondence: DPI-布莱克韦尔对应的小结
    Relation to Bayesian likelihood-ratio learning: 与贝叶斯似然比学习的关系
    Relation to Bayesian likelihood-ratio learning::Summary table: 小结表
    Summary: 小结
    Exercises: 练习
---

(blackwell_kihlstrom)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 布莱克韦尔的实验比较定理

```{contents} Contents
:depth: 2
```

## 概述



本讲座研究 *布莱克韦尔定理*  {cite}`blackwell1951,blackwell1953` ，该定理用于对统计实验进行排序。

我们的介绍引入了 {cite:t}`kihlstrom1984` 对布莱克韦尔定理的贝叶斯解释所得到的发现。

布莱克韦尔和基尔斯特罗姆研究的统计模型选择问题，与本 QuantEcon 讲座 {doc}`likelihood_bayes` 中遇到的问题密切相关。

为了理解其中的关联，理解布莱克韦尔的 **实验** 概念与 {doc}`likelihood_bayes` 中出现的"概率分布"或"参数化统计模型"概念之间的关系是很有帮助的。

布莱克韦尔研究的情形是：一个决策者想要知道位于空间 $S$ 中的状态 $s$ 的值。

对布莱克韦尔而言，**实验** 是一个 **条件概率模型**  $\{\mu(\cdot \mid s) : s \in S\}$ ，即一族由同一状态 $s \in S$ 条件化的概率分布。

我们可以自由地将"状态"解释为"参数"或"参数向量"。

在两状态情形 $S = \{s_1, s_2\}$ 下，两个条件密度 $f(\cdot) = \mu(\cdot \mid s_1)$ 和 $g(\cdot) = \mu(\cdot \mid s_2)$ 正是我们在本 QuantEcon 讲座 {doc}`likelihood_bayes` 以及本系列 QuantEcon 讲座中其他若干讲座中，反复用于研究经典假设检验和贝叶斯推断的那两个密度。

{cite:t}`kihlstrom1984` 将 *哪个实验更具信息量？* 这一问题解释为：询问哪个条件概率模型能让一个对 $\{s_1, s_2\}$ 持有先验的贝叶斯决策者获得更高的期望效用。

我们将把"信号"和"实验"作为同义词使用。

因此，假设两个信号 $\tilde{x}_\mu$ 和 $\tilde{x}_\nu$ 都对未知状态 $\tilde{s}$ 具有信息量。

如果每个贝叶斯决策者用 $\mu$ 都能获得弱高于用 $\nu$ 的期望效用，则称信号 $\mu$ **至少与** 信号 $\nu$ **一样具有信息量**。

这个经济学标准等价于两个统计学标准：

- *充分性*（布莱克韦尔）： $\tilde{x}_\nu$ 可以通过对 $\tilde{x}_\mu$ 进行额外的随机化来生成。
- *不确定性降低*（ {cite:t}`degroot1962` ）：对每个凹的不确定性函数而言， $\tilde{x}_\mu$ 至少与 $\tilde{x}_\nu$ 一样降低了期望不确定性。

基尔斯特罗姆的表述聚焦于 *后验分布*。

更具信息量的实验会生成在凸序意义下更为分散的后验分布。

在两状态情形下，这就变成了在 $[0, 1]$ 上的均值保持展开比较，可以用检验二阶随机占优的积分 CDF 检验来验证。

{doc}`information_market_equilibrium` 和 {doc}`pricing_information` 以本讲座的结果为基础。

本讲座的进行方式如下。

1. 我们将实验定义为马尔可夫矩阵，并将信息扭曲（garbling）定义为随机变换。
2. 我们陈述对实验进行排序的三个等价标准以及布莱克韦尔定理。
3. 我们通过标准实验、均值保持展开和凸序，发展基尔斯特罗姆的贝叶斯解释。
4. 我们研究德格鲁特的不确定性函数及其两个应用。
5. 我们将布莱克韦尔序与数据处理不等式以及 {doc}`likelihood_bayes` 中的似然比学习联系起来。

Python 模拟将阐释每一个步骤。

我们先进行一些导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
mpl.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
```

## 实验与随机变换


### 状态空间与实验

设 $S = \{s_1, \ldots, s_N\}$ 为世界可能状态的有限集合。

一个 **实验** 由观测信号 $\tilde{x}$ 在给定状态 $\tilde{s}$ 下的条件分布来描述。

当信号空间也是有限的，比如 $X = \{x_1, \ldots, x_M\}$ 时，一个实验就归结为一个 $N \times M$ 的 *马尔可夫矩阵*

$$
\mu = [\mu_{ij}], \qquad
\mu_{ij} = \Pr(\tilde{x}_\mu = x_j \mid \tilde{s} = s_i) \geq 0,
\quad \sum_{j=1}^{M} \mu_{ij} = 1 \;\forall\, i.
$$

每一行 $i$ 给出了当真实状态为 $s_i$ 时的信号分布。


```{code-cell} ipython3
μ = np.array([[0.6, 0.3, 0.1],
              [0.1, 0.3, 0.6]])

Q = np.array([[1.0, 0.0],
              [0.5, 0.5],
              [0.0, 1.0]])

ν = μ @ Q

print("实验 μ（3 个信号，各行之和为 1）：")
print(μ)
print("\n随机变换 Q（3 × 2）：")
print(Q)
print("\n实验 ν = μ @ Q（2 个信号）：")
print(ν)
print("\nμ 的行和：", μ.sum(axis=1))
print("ν 的行和：", ν.sum(axis=1))
```

### 随机变换

**随机变换**  $Q$ 通过进一步的随机化将一个实验的信号映射为另一个实验的信号。

在具有 $M$ 个输入信号和 $K$ 个输出信号的离散设定下， $Q$ 是一个 $M \times K$ 的马尔可夫矩阵：对每一行 $l$ 有 $q_{lk} \geq 0$ 且 $\sum_k q_{lk} = 1$ 。

```{prf:definition} 充分性
:label: def-sufficiency

如果存在一个随机变换 $Q$ （一个 $M \times K$ 的马尔可夫矩阵）使得

$$
\nu = \mu \, Q,
$$

则称实验 $\mu$  *对*  $\nu$ *充分*，这意味着一个观测到 $\tilde{x}_\mu$ 的观察者可以通过将其信号传递给 $Q$ 来生成 $\tilde{x}_\nu$ 的分布。

对于一般的信号空间， $Q$ 是一个马尔可夫核，此时该条件可写为：对 $\nu$ 的信号空间中的每个可测集合 $E$ 以及每个状态 $s_i$ ，都有 $\nu_i(E) = \int Q(x, E) \, \mu_i(dx)$ 。
```

当 $\mu$ 对 $\nu$ 充分时，我们记作 $\mu \geq \nu$ ，并称 $\nu$ 为 $\mu$ 的 **粗化**（garbling）。

如果你观测到更具信息量的信号 $\tilde{x}_\mu$ ，那么你总是可以 *丢弃* 信息来重现一个信息量较少的信号。

反过来通常行不通：粗化无法恢复其所舍弃的信息。

我们可以用上面定义的两个实验 $\mu$ 和 $\nu$ 来进行数值验证。

下面的函数搜索一个使 $\|\nu - \mu \, Q\|$ 最小化的随机变换 $Q$ 。

如果存在一个精确的 $Q$ ，残差将接近于零；否则残差会很大。

```{code-cell} ipython3
def find_stochastic_transform(μ, ν, tol=1e-8):
    """
    寻找一个行随机矩阵 Q，使 ||ν - μ @ Q|| 最小化。
    """
    _, M = μ.shape
    _, K = ν.shape

    def unpack(q_flat):
        return q_flat.reshape(M, K)

    def objective(q_flat):
        Q = unpack(q_flat)
        return np.linalg.norm(ν - μ @ Q)**2

    constraints = [
        {"type": "eq", "fun": lambda q_flat, 
        row=i: unpack(q_flat)[row].sum() - 1.0}
        for i in range(M)
    ]
    bounds = [(0.0, 1.0)] * (M * K)
    Q0 = np.full((M, K), 1 / K).ravel()

    result = minimize(
        objective,
        Q0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1_000},
    )

    Q = unpack(result.x)
    residual = np.linalg.norm(ν - μ @ Q)
    return Q, residual

# 正向：寻找使 ν = μ @ Q 的 Q（应当成功）
Q_fwd, res_fwd = find_stochastic_transform(μ, ν)
print("正向（μ 到 ν）：")
print(f"  残差 = {res_fwd:.2e}")
print(f"  存在精确变换：{res_fwd < 1e-6}")

# 反向：寻找使 μ = ν @ Q' 的 Q'（应当失败）
Q_rev, res_rev = find_stochastic_transform(ν, μ)
print("\n反向（ν 到 μ）：")
print(f"  残差 = {res_rev:.2e}")
print(f"  存在精确变换：{res_rev < 1e-6}")
```

正向残差接近于零：存在一个从 $\mu$ 到 $\nu$ 的随机变换，这证实了 $\mu$ 对 $\nu$ 充分。

反向残差很大：没有任何随机变换能从 $\nu$ 恢复出 $\mu$ 。

{ref}`bk_ex1` 证明了不存在能将 $\nu$ 映射到 $\mu$ 的随机变换。

粗化只能将后验信念拉向先验，而 $\mu$ 所生成的后验比 $\nu$ 所能达到的后验离先验更远。

## 三个等价标准

布莱克韦尔定理确立了比较实验的三种不同方式最终都是等价的。

### 标准 1：经济学标准

第一个标准通过实验对决策者的价值来比较实验。

设 $A$ 为一个紧的行动集合， $u: A \times S \to \mathbb{R}$ 为一个有界连续效用函数。

决策者观测到 $x \in X$ ，然后选择一个行动，该行动可能是随机的。

**决策规则** $\delta$ 为每个信号 $x$ 指定一个行动上的概率分布 $\delta(\cdot \mid x)$ 。

设 $p = (p_1, \ldots, p_N)$ 为状态上的先验，并记

$$
P = \bigl\{(p_1, \ldots, p_N) : p_i \geq 0,\; \textstyle\sum_i p_i = 1\bigr\}
$$

为概率单纯形。

对固定的 $A$ 和 $u$ ，在实验 $\mu$ 下的 *可达期望效用向量集合* 为

$$
B(\mu, A, u) = \Bigl\{v \in \mathbb{R}^N :
  v_i = \textstyle\int_X \int_A u(a, s_i)\, \delta(da \mid x)\,\mu_i(dx)
  \text{ for some decision rule } \delta \Bigr\}.
$$

每个 $v \in B(\mu, A, u)$ 都列出了某个决策规则在每个状态下所能带来的期望效用。

随机化规则使得 $B(\mu, A, u)$ 是凸集。

它们也使得下面的复制论证变得精确：一个看到 $\tilde{x}_\mu$ 的决策者可以生成一个合成的 $\tilde{x}_\nu$ ，然后遵循为 $\nu$ 设计的任何规则。

```{prf:definition} 经济学标准
:label: def-economic-criterion

如果对每个紧行动集 $A$ 和每个有界连续效用函数 $u: A \times S \to \mathbb{R}$ 都有

$$
B(\mu, A, u) \supseteq B(\nu, A, u)
$$

则称 $\mu$ 在经济学意义上 **至少与** $\nu$ **一样具有信息量**。
```

这个标准说的是：如果决策者在看到 $\nu$ 后所能达到的任何结果，在看到 $\mu$ 后也都能达到，那么实验 $\mu$ 就优于实验 $\nu$ 。

其原因在于，一个更具信息量的实验允许决策者通过 *忽略* 或 *混淆* 部分额外信息来模仿一个信息量较少的实验。

但反过来则未必可能。

因此 $B(\mu, A, u) \supseteq B(\nu, A, u)$ 意味着 $\mu$ 给决策者提供的可行期望效用结果至少与 $\nu$ 一样多。

下面的布莱克韦尔定理意味着一个等价的价值比较。

对每个先验 $p \in P$ 、每个 $A$ 和每个 $u$ ，在 $\mu$ 下最大化的期望效用 $\max_{v \in B(\mu, A, u)} \sum_i p_i v_i$ 至少与在 $\nu$ 下一样大。

贝叶斯决策者通过一个确定性规则（该规则在每个后验下都选择最优行动）来达到该最大值；随机化只在集合的比较中起作用。

### 标准 2：充分性标准

第二个标准就是 {prf:ref}`def-sufficiency` 中的充分性关系：如果存在某个随机变换 $Q$ 使得 $\nu = \mu Q$ ，则 $\mu \geq \nu$ 。

### 标准 3：不确定性标准

第三个标准通过实验对状态不确定性的降低程度来比较实验。

{cite:t}`degroot1962` 把任何凹函数 $U: P \to \mathbb{R}$ 称为 **不确定性函数**。

典型的例子是香农熵：

$$
U(p) = -\sum_{i=1}^{N} p_i \log p_i.
$$

```{prf:definition} 德格鲁特不确定性标准
:label: def-degroot-uncertainty

如果对每个先验 $p \in P$ 和每个凹函数 $U: P \to \mathbb{R}$ 都有

$$
\int_P U(q)\,\hat\mu^p(dq)
\;\leq\;
\int_P U(q)\,\hat\nu^p(dq),
$$

其中 $\hat\mu^p$ 是实验 $\mu$ 在先验 $p$ 下诱导的后验信念分布，则称 $\mu$ **至少与** $\nu$ **一样地降低了期望不确定性**。
```

任何实验都会弱地降低期望不确定性。

为看清这一点，设 $\tilde{\pi} = p^\mu(\tilde{x}_\mu)$ 表示由实验 $\mu$ 诱导的随机后验。

那么 $\tilde{\pi}$ 服从分布 $\hat\mu^p$ ，因此

$$
\mathbb{E}[U(\tilde{\pi})] = \int_P U(q)\,\hat\mu^p(dq).
$$

后验的均值为 $p$ （见下面的 {prf:ref}`prop-mean-preservation`），因此詹森不等式给出

$$
\mathbb{E}[U(\tilde{\pi})] \leq U(\mathbb{E}[\tilde{\pi}]) = U(p).
$$

因此

$$
\int_P U(q)\,\hat\mu^p(dq) \leq U(p),
$$

所以任何实验都会弱地降低期望不确定性。

标准 3 对每个先验 $p$ 都进行了量化。

布莱克韦尔定理表明，一个具有全支撑的先验就足够了：如果该不等式在某一个这样的先验下对每个凹函数 $U$ 都成立，那么它在每个先验下都成立。

下面基尔斯特罗姆的构造正是利用了这一事实，使用一个固定的先验 $c$ ，通常取均匀先验 $(1/N, \ldots, 1/N)$ 。

## 主定理

```{prf:theorem} 布莱克韦尔定理
:label: thm-blackwell

以下三个条件等价：

(i) 经济学标准：对每个紧集 $A$ 和每个有界连续效用函数 $u$ 都有 $B(\mu, A, u) \supseteq B(\nu, A, u)$ 。

(ii) 充分性标准：存在一个从 $\mu$ 的信号空间到 $\nu$ 的信号空间的随机变换 $Q$ ，使得 $\nu = \mu Q$ 。

(iii) 不确定性标准：对每个凹函数 $U$ ，在每个先验 $p \in P$ 处，或者等价地在一个具有完全支持的先验处，都有 $\int_P U(q)\,\hat\mu^p(dq) \leq \int_P U(q)\,\hat\nu^p(dq)$ 。
```

另见 {cite:t}`blackwell1951` 、 {cite:t}`bonnenblust1949` 和 {cite:t}`degroot1962` 。

最困难的部分是经济学标准与充分性标准之间的等价性。

*概要（ii $\Rightarrow$ i）：* 如果 $\nu = \mu Q$ ，那么任何基于 $\tilde{x}_\nu$ 的决策规则都可以通过如下方式复现：先观测 $\tilde{x}_\mu$ ，再从 $Q$ 中抽取一个合成的 $\tilde{x}_\nu$ ，然后应用同样的规则。

*概要（i $\Rightarrow$ ii）：* 由于对每个 $A$ 和 $u$ 都有 $B(\mu, A, u) \supseteq B(\nu, A, u)$ ，一个分离超平面（对偶）论证意味着 $\mu$ 的后验分布是 $\nu$ 的后验分布的均值保持展开，而斯特拉森定理 {cite}`strassen1965` 随后给出一个后验空间中的均值保持核 $D$ ，将 $\nu$ 的标准实验送入 $\mu$ 的标准实验。再从这些后验律回到原始信号空间就得到所需的混淆 $Q$ ，满足 $\nu = \mu Q$ 。因此 $D$ 是后验信念上的中间随机化，而非字面意义上的信号空间核 $Q$ 。

*概要（ii $\Rightarrow$ iii）：* 在混淆之下，较粗实验的后验是较细实验后验的条件期望，因此詹森不等式对每个凹函数 $U$ 都给出该结果。

*概要（iii $\Rightarrow$ ii）：* 其逆命题，即对所有凹函数 $U$ 的不等式迫使 $Q$ 存在，在 {cite}`blackwell1953` 中得到证明。基尔斯特罗姆基于后验的表示使这一几何变得透明。

## 基尔斯特罗姆的贝叶斯解释


### 后验与标准实验

基尔斯特罗姆分析中的关键对象是 *后验信念向量*。

当先验 $p$ 成立且实验 $\mu$ 产生信号 $x$ 时，贝叶斯规则给出

$$
p_i^\mu(x) = \Pr(\tilde{s} = s_i \mid \tilde{x}_\mu = x)
= \frac{\mu_{ix} \, p_i}{\sum_j \mu_{jx}\, p_j}, \qquad i = 1, \ldots, N.
$$

后验 $p^\mu(x) \in P$ 是单纯形中的一个随机点。

```{prf:property} 均值保持
:label: prop-mean-preservation

先验 $p$ 是后验的期望：

$$
\mathbb{E}[p^\mu] = \sum_x \Pr(\tilde{x}_\mu = x)\, p^\mu(x) = p.
$$

这有时被称为 *信念的迭代期望定律*。
```

对固定的先验 $c$ ，基尔斯特罗姆的 **标准实验** 用 $\mu$ 所生成的后验信念来替换其原始信号。

设 $\hat\mu^c$ 表示由 $\mu$ 在先验 $c$ 下诱导的后验分布。
均值保持意味着 $\int_P q \, \hat\mu^c(dq) = c$ 。

当两个实验诱导相同的后验分布时，称它们 **信息等价**。

标准实验剥离了信号除后验以外的所有细节，因此它提供了用于比较实验的一个规范的贝叶斯表示。

后验信念上的随机核存在于单纯形 $P$ 上，而布莱克韦尔混淆 $Q$ 存在于原始信号空间上。基尔斯特罗姆的构造使用前者来研究凸序，然后在过渡到标准实验之后恢复后者。

任何两个生成相同后验分布的实验，都会导致每个贝叶斯决策者做出完全相同的决策，无论它们的原始信号空间看起来多么不同。

### 均值保持展开与布莱克韦尔序

基尔斯特罗姆的关键重述如下。

```{prf:theorem} 基尔斯特罗姆的重述
:label: thm-kihlstrom

$\mu \geq \nu$ 在布莱克韦尔意义下成立，当且仅当 $\hat\mu^c$ 是 $\hat\nu^c$ 的一个
**均值保持展开**；即，对每个凸函数 $g: P \to \mathbb{R}$ 都有

$$
\int_P g(p)\,\hat\mu^c(dp) \;\geq\; \int_P g(p)\,\hat\nu^c(dp)
$$

。
```

等价地， $\hat\mu^c$ 在凸序意义下大于 $\hat\nu^c$ 。

先验 $c$ 可以是任何具有完全支撑的先验；比较结果并不依赖于具体选择哪一个。

一个更好的实验会将后验信念推向离先验更远的地方，同时保持它们的均值不变。

为了具体地看到这一点，我们为两状态情形定义两个实验并计算它们的后验。

```{code-cell} ipython3
def compute_posteriors(μ, prior, tol=1e-14):
    """
    为每个信号实现计算后验分布。
    """
    N, M = μ.shape
    signal_probs = μ.T @ prior
    numerators = μ.T * prior
    posteriors = np.zeros((M, N))
    np.divide(
        numerators,
        signal_probs[:, None],
        out=posteriors,
        where=signal_probs[:, None] > tol,
    )
    return posteriors, signal_probs


def check_mean_preservation(posteriors, signal_probs, prior):
    """验证 E[后验] == 先验。"""
    expected_posterior = (posteriors * signal_probs[:, None]).sum(axis=0)
    return expected_posterior, np.allclose(expected_posterior, prior)


N = 2
prior = np.array([0.5, 0.5])

μ_info = np.array([[0.8, 0.2],
                   [0.2, 0.8]])

ν_info = np.array([[0.6, 0.4],
                   [0.4, 0.6]])

post_μ, probs_μ = compute_posteriors(μ_info, prior)
post_ν, probs_ν = compute_posteriors(ν_info, prior)

print("实验 μ（更具信息量）：\n")
print("信号概率：", probs_μ.round(3))
print("后验（行 = 信号，列 = 状态）：")
print(post_μ.round(3))
mean_μ, ok_μ = check_mean_preservation(post_μ, probs_μ, prior)
print(f"E[后验] = {mean_μ.round(4)}  （等于先验：{ok_μ}）")

print("\n 实验 ν（信息量较少）：\n")
print("信号概率：", probs_ν.round(3))
print("后验：")
print(post_ν.round(3))
mean_ν, ok_ν = check_mean_preservation(post_ν, probs_ν, prior)
print(f"E[后验] = {mean_ν.round(4)}  （等于先验：{ok_ν}）")
```

对 $N = 2$ 个状态，单纯形 $P$ 是单位区间 $[0, 1]$ （状态 $s_1$ 的概率）。

我们可以直接绘制在实验 $\mu$ 和 $\nu$ 下的后验分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 两状态情形下的后验分布
    name: fig-blackwell-two-state-posteriors
---
def plot_posterior_distributions(μ_matrix, ν_matrix, prior,
                                 labels=("μ（更具信息量）",
                                         "ν（信息量较少）")):
    """
    对两状态实验，绘制 [0,1] 上的后验分布
    （即标准实验分布）。
    """
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    prior_val = prior[0]

    for ax, posts, probs, label in zip(
        axes, [posts_μ, posts_ν], [probs_μ, probs_ν], labels):
        p_s1 = posts[:, 0]
        ax.vlines(p_s1, 0, probs, linewidth=6, color="steelblue", alpha=0.7)
        ax.axvline(prior_val, color="tomato", linestyle="--", linewidth=2,
                   label=f"先验 = {prior_val:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"后验 $p(s_1 \mid x)$", fontsize=12)
        ax.set_ylabel("概率质量", fontsize=12)
        mean_post = (p_s1 * probs).sum()
        ax.axvline(mean_post, color="green", linestyle=":", linewidth=2,
                   label=f"E[后验] = {mean_post:.2f}")
        ax.text(0.03, 0.94, label, transform=ax.transAxes, va="top")
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_posterior_distributions(μ_info, ν_info, prior)
```

这就是均值保持展开的实际体现：两个分布具有相同的均值（等于先验），但更具信息量的实验 $\mu$ 将其后验展开得更远。

我们可以数值地验证均值保持展开条件。

关键事实在于，至多相差一个仿射项，任何凸函数都可以表示为"看涨期权"收益 $g_t(p) = \max(p - t, 0)$ 的混合。

由于被比较的两个后验分布具有相同的均值，那个仿射项在比较中会相互抵消。

因此只需对所有阈值 $t \in [0, 1]$ 检验 $E[g_t(p^\mu)] \geq E[g_t(p^\nu)]$ 即可。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 两状态情形下的凸序检验
    name: fig-blackwell-convex-order-check
---
def check_mps_convex_functions(μ_matrix, ν_matrix, prior, n_functions=200):
    """
    使用凸函数 g(p) = max(p - t, 0) 验证
    均值保持展开条件。
    """
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    p_μ = posts_μ[:, 0]
    p_ν = posts_ν[:, 0]

    thresholds = np.linspace(0, 1, n_functions)
    diffs = []
    for t in thresholds:
        Eg_μ = (np.maximum(p_μ - t, 0) * probs_μ).sum()
        Eg_ν = (np.maximum(p_ν - t, 0) * probs_ν).sum()
        diffs.append(Eg_μ - Eg_ν)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, diffs, color="steelblue", linewidth=2)
    ax.axhline(0, color="tomato", linestyle="--", linewidth=2)
    ax.fill_between(thresholds, diffs, 0,
                    where=np.array(diffs) >= 0,
                    alpha=0.25, color="steelblue",
                    label="$E[g(p^μ)] - E[g(p^ν)] \\geq 0$")
    ax.set_xlabel("阈值 $t$", fontsize=12)
    ax.set_ylabel(r"$E[\max(p-t,0)]$ 差值", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    all_non_negative = all(d >= -1e-10 for d in diffs)
    print(f"μ 是 ν 的均值保持展开：{all_non_negative}")
    return diffs

_ = check_mps_convex_functions(μ_info, ν_info, prior)
```

差值 $E[g_t(p^\mu)] - E[g_t(p^\nu)]$ 对每个阈值 $t$ 都是非负的，这证实了 $\hat\mu^c$ 是 $\hat\nu^c$ 的均值保持展开，因此在布莱克韦尔序中 $\mu \geq \nu$ 。

### 凸序与二阶随机占优

如果对每个凹函数 $u$ 都有 $E[u(X)] \geq E[u(Y)]$ ，则称随机变量 $X$ 在 **凹序** 意义下占优于 $Y$ 。

取 $u(x) = x$ 与 $u(x) = -x$ 表明两者的均值必须相等。

$X$ 在凹序意义下占优于 $Y$ ，当且仅当 $Y$ 是 $X$ 的均值保持展开。

**二阶随机占优** 仅使用递增凹函数 $u$ ，并允许均值不同；当均值相等时，它与凹序一致。

由于不确定性函数是凹函数，条件

$$
\mathbb{E}[U(p^\mu)] \leq \mathbb{E}[U(p^\nu)] \quad \text{对所有凹函数 } U
$$

表明 $\nu$ 下的后验在凹序意义下占优于 $\mu$ 下的后验，因此 $\hat\mu^c$ 在 $P$ 上的凸序意义下占优于 $\hat\nu^c$ 。

当 $N = 2$ 时，后验信念是 $[0, 1]$ 中的标量，此时比较可归结为经典的累积分布函数积分检验。

在均值相等的情况下， $\hat\mu^c$ 是 $\hat\nu^c$ 的均值保持展开，当且仅当对所有 $t \in [0,1]$ 都有 $\int_0^t F_\mu(s)\,ds \geq \int_0^t F_\nu(s)\,ds$ ，其中 $F_\mu$ 和 $F_\nu$ 分别是每个实验下状态 $s_1$ 上后验的累积分布函数。

两个后验分布的均值均等于先验，因此该检验适用。

我们对上面的两状态示例进行验证。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 两状态情形下的累积分布函数积分检验
    name: fig-blackwell-integrated-cdf
---
def cdf_data_1d(weights, values):
    """对离散分布的支撑点排序并计算累积质量。"""
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_wts = weights[idx]
    cum_mass = np.cumsum(sorted_wts)
    return sorted_vals, cum_mass


def plot_sosd_posteriors(μ_matrix, ν_matrix, prior):
    """绘制状态 s1 上后验分布的累积分布函数及其积分。"""
    posts_μ, probs_μ = compute_posteriors(μ_matrix, prior)
    posts_ν, probs_ν = compute_posteriors(ν_matrix, prior)

    p_μ = posts_μ[:, 0]
    p_ν = posts_ν[:, 0]

    sv_μ, cm_μ = cdf_data_1d(probs_μ, p_μ)
    sv_ν, cm_ν = cdf_data_1d(probs_ν, p_ν)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for sv, cm, lbl, c in [(sv_μ, cm_μ, "μ", "steelblue"),
                           (sv_ν, cm_ν, "ν", "darkorange")]:
        xs = np.concatenate([[0], sv, [1]])
        ys = np.concatenate([[0], cm, [1]])
        ax.step(xs, ys, where="post", label=lbl, color=c, linewidth=2)
    ax.axvline(prior[0], linestyle="--", color="gray", alpha=0.6, linewidth=2,
               label="先验")
    ax.set_xlabel(r"后验 $p(s_1 \mid x)$", fontsize=12)
    ax.set_ylabel("累积概率", fontsize=12)
    ax.text(0.03, 0.94, "累积分布函数", transform=ax.transAxes, va="top")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    grid = np.linspace(0, 1, 200)

    def integrated_cdf(sorted_vals, cum_mass, grid):
        cdf = np.array([cum_mass[sorted_vals <= t].max()
                        if np.any(sorted_vals <= t) else 0.0
                        for t in grid])
        return np.cumsum(cdf) * (grid[1] - grid[0])

    int_μ = integrated_cdf(sv_μ, cm_μ, grid)
    int_ν = integrated_cdf(sv_ν, cm_ν, grid)

    ax2.plot(grid, int_μ, label=r"$\int F_\mu$", color="steelblue", linewidth=2)
    ax2.plot(grid, int_ν, color="darkorange",
             label=r"$\int F_\nu$", linewidth=2)
    ax2.fill_between(grid, int_ν, int_μ,
                     where=int_μ >= int_ν,
                     alpha=0.2, color="steelblue",
                     label=(r"$\int F_\mu \geq \int F_\nu$"
                            r"（$\mu$ 是 $\nu$ 的均值保持展开）"))
    ax2.set_xlabel(r"$t$", fontsize=12)
    ax2.set_ylabel("累积分布函数的积分", fontsize=12)
    ax2.text(0.03, 0.94, "累积分布函数的积分", transform=ax2.transAxes, va="top")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

plot_sosd_posteriors(μ_info, ν_info, prior)
```

## 用多状态模拟布莱克韦尔序

我们现在转到一个三状态的例子。

实验 $\mu$ 与状态强相关，而实验 $\nu$ 是 $\mu$ 的一个混淆。

```{code-cell} ipython3
N3 = 3
prior3 = np.array([1/3, 1/3, 1/3])

μ3 = np.array([[0.7, 0.2, 0.1],
               [0.1, 0.7, 0.2],
               [0.2, 0.1, 0.7]])

Q3 = np.array([[0.9, 0.05, 0.05],
               [0.05, 0.8, 0.15],
               [0.05, 0.15, 0.8]])

ν3 = μ3 @ Q3

print("μ (3×3)：")
print(np.round(μ3, 2))
print("\nQ（混淆）：")
print(np.round(Q3, 2))
print("\nν = μ @ Q：")
print(np.round(ν3, 3))
```


对三个状态，后验信念存在于一个 2-单纯形中。

让我们绘制 $\mu$ 和 $\nu$ 下的后验原子，标记面积与概率成正比

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 2-单纯形上的后验原子
    name: fig-blackwell-simplex-clouds
---
def simplex_to_cart(pts):
    """将 2-单纯形上的重心坐标转换为二维笛卡尔坐标。"""
    corners = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3)/2]])
    return pts @ corners


def plot_simplex_posteriors(μ_matrix, ν_matrix, prior3):
    """绘制 2-单纯形上的后验原子，面积与概率成正比。"""
    corners = simplex_to_cart(np.eye(3))
    prior_cart = simplex_to_cart(prior3[None, :])[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel_labels = ["μ（更具信息量）", "ν（被混淆）"]
    colors = ["steelblue", "darkorange"]
    labels = ["$s_1$", "$s_2$", "$s_3$"]
    offsets = [(-0.07, -0.05), (0.02, -0.05), (-0.02, 0.03)]

    for ax, m, c, panel_label in zip(axes, [μ_matrix, ν_matrix],
                                     colors, panel_labels):
        posts, probs = compute_posteriors(m, prior3)
        cart = simplex_to_cart(posts)
        tri = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(tri)
        ax.scatter(cart[:, 0], cart[:, 1], s=3000 * probs, alpha=0.6, color=c)
        ax.scatter(*prior_cart, s=120, color="red", zorder=5,
                   label="先验", marker="*")
        for i, (lbl, off) in enumerate(zip(labels, offsets)):
            ax.text(corners[i][0] + off[0], corners[i][1] + off[1],
                    lbl, fontsize=13)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.94, panel_label, transform=ax.transAxes, va="top")
        ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.show()

plot_simplex_posteriors(μ3, ν3, prior3)
```

在 $\mu$ 下，后验原子位于更靠近顶点的位置。

在被混淆的实验 $\nu$ 下，它们则更靠近先验。

看涨期权检验无法推广到单纯形的情形。

混淆矩阵 $Q_3$ 本身就证明了 $\mu_3 \geq \nu_3$，下一节将在这个例子中检验若干凹不确定性函数。

## 德格鲁特不确定性函数


### 凹不确定性函数与信息的价值

{cite}`degroot1962` 通过 **不确定性函数**  $U: P \to \mathbb{R}$ 将信息的价值形式化。

德格鲁特的定义只要求凹性，使得观测任何信号都弱地降低期望不确定性。

常见的例子，例如香农熵，还在 $p$ 的各分量上是对称的，并在均匀先验处取最大值。

布莱克韦尔等价性使用每个凹函数 $U$，无论其是否对称；{ref}`bk_ex2` 说明了为何非对称函数也很重要。

*在给定先验 $p$ 下实验 $\mu$ 的价值* 为

$$
I(\tilde{x}_\mu;\, \tilde{s};\, U)
= U(p) - \mathbb{E}[U(p^\mu)],
$$

这个量是不确定性的期望降低量。

布莱克韦尔序等价于以下陈述：对 *每个* 凹函数 $U$ 都有 $I(\tilde{x}_\mu; \tilde{s}; U) \geq I(\tilde{x}_\nu; \tilde{s}; U)$ 。

### 作为特例的香农熵

规范的不确定性函数是香农熵

$$
U_H(p) = -\sum_{i=1}^{N} p_i \log p_i.
$$

对任意先验 $p$，德格鲁特的价值公式变为

$$
I(\tilde{x}_\mu, p;\, U_H)
= H(\tilde{s}) - H(\tilde{s} \mid \tilde{x}_\mu),
$$

其中 $H(\tilde{s}) = U_H(p)$ 是先验的熵，$H(\tilde{s} \mid \tilde{x}_\mu)$ 是给定信号时状态的条件熵。

在均匀先验下，$H(\tilde{s}) = \log N$ 。

要看清原因，将 $H(\tilde{s} \mid \tilde{x}_\mu) = \sum_x \Pr(\tilde{x}_\mu = x) \, H(\tilde{s} \mid \tilde{x}_\mu = x)$ 写出，其中每个条件熵项等于 $-\sum_i p_i^\mu(x) \log p_i^\mu(x) = U_H(p^\mu(x))$ 。

代入德格鲁特公式得到 $I = U_H(p) - \mathbb{E}[U_H(p^\mu)] = H(\tilde{s}) - H(\tilde{s} \mid \tilde{x}_\mu)$ ，这正是 $\tilde{x}_\mu$ 与 $\tilde{s}$ 之间的 *互信息*。

```{note}
布莱克韦尔序蕴含基于熵的不等式。

由于熵是严格凹的，严格的布莱克韦尔改进会严格提升互信息。

但其逆命题不成立：按互信息排序并不意味着按布莱克韦尔序排序。

两个实验在布莱克韦尔序下可能不可比，而其中一个具有更高的互信息，或者两者具有相同的互信息，如 {ref}`bk_ex2` 所示。

完整的布莱克韦尔序要求不等式对 *每个* 凹函数 $U$ 都成立，而不仅仅是香农熵。
```

```{code-cell} ipython3
def entropy(p, ε=1e-12):
    """概率向量的香农熵。"""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, ε, 1.0)
    return -np.sum(p * np.log(p))


def degroot_value(μ_matrix, prior, U_func):
    """
    计算德格鲁特信息价值 I = U(先验) - E[U(后验)]。
    """
    posts, probs = compute_posteriors(μ_matrix, prior)
    prior_uncertainty = U_func(prior)
    expected_post_uncertainty = sum(
        probs[j] * U_func(posts[j]) for j in range(len(probs)))
    return prior_uncertainty - expected_post_uncertainty


def gini_impurity(p):
    """基尼不纯度：1 - sum(p_i^2)。"""
    return 1.0 - np.sum(np.asarray(p)**2)


def tsallis_entropy(p, q=2):
    """q 阶查理斯熵（q>1 时为凹）。"""
    p = np.clip(p, 1e-12, 1.0)
    return (1 - np.sum(p**q)) / (q - 1)


def tsallis_q15(p):
    """q=1.5 的查理斯熵，用于独立的凹性检验。"""
    return tsallis_entropy(p, q=1.5)


def sqrt_index(p):
    """基于 sum(sqrt(p_i)) 的凹不确定性指数。"""
    p = np.clip(np.asarray(p), 0.0, 1.0)
    return np.sum(np.sqrt(p)) - 1.0

uncertainty_functions = {
    "Shannon entropy": entropy,
    "Gini impurity": gini_impurity,
    "Tsallis (q=1.5)": tsallis_q15,
    "Square-root index": sqrt_index,
}

header = (f"{'Uncertainty function':<22}  "
          f"{'I(μ)':<10}  {'I(ν)':<10}  "
          f"{'I(μ)>=I(ν)?'}")
print(header)
print("-" * 58)
for name, U in uncertainty_functions.items():
    I_μ = degroot_value(μ_info, prior, U)
    I_ν = degroot_value(ν_info, prior, U)
    print(f"{name:<22}  {I_μ:<10.4f}  {I_ν:<10.4f}  {I_μ >= I_ν - 1e-10}")

print("\n三状态示例（μ3、ν3，均匀先验）：")
print(header)
print("-" * 58)
for name, U in uncertainty_functions.items():
    I_μ = degroot_value(μ3, prior3, U)
    I_ν = degroot_value(ν3, prior3, U)
    print(f"{name:<22}  {I_μ:<10.4f}  {I_ν:<10.4f}  {I_μ >= I_ν - 1e-10}")
```

正如定理所预测的那样，在这两个例子中对每个凹不确定性函数都有 $I(\mu) \geq I(\nu)$ ，因为 $\mu \geq \nu$ 在布莱克韦尔意义下成立。

### 作为实验质量函数的信息价值

我们现在将介于无信息量和完全有信息量之间的实验连续统参数化。

对 $N = 2$ 个状态，一个自然的族是

$$
\mu(\theta) = (1 - \theta) \cdot \tfrac{1}{2}\mathbf{1}\mathbf{1}^\top
             + \theta \cdot I_2,
\quad \theta \in [0, 1],
$$

第一项是完全混合的矩阵， $I_2$ 是单位矩阵。

这个族按布莱克韦尔准则排序：对 $\theta' < \theta$ ，有 $\mu(\theta') = \mu(\theta) \, \mu(\theta' / \theta)$ ，因此质量较低的实验是质量较高实验的混淆。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 信息价值与实验质量
    name: fig-blackwell-value-by-quality
---
def make_experiment(θ, N=2):
    """参数化实验：θ=0 无信息量，θ=1 完全有信息量。"""
    return (1 - θ) * np.ones((N, N)) / N + θ * np.eye(N)


θs = np.linspace(0, 1, 100)
prior2 = np.array([0.5, 0.5])

fig, ax = plt.subplots(figsize=(9, 4))
for name, U in uncertainty_functions.items():
    values = [degroot_value(make_experiment(θ), prior2, U) for θ in θs]
    vmin, vmax = values[0], values[-1]
    normed = (np.array(values) - vmin) / (vmax - vmin + 1e-15)
    ax.plot(θs, normed, label=name, linewidth=2)

ax.set_xlabel("实验质量 θ（0 = 无信息量，1 = 完全有信息量）",
              fontsize=11)
ax.set_ylabel("归一化信息价值 I(μ(θ))", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

每个凹不确定性函数都会给更具信息量的实验赋予弱更高的价值。

## 应用1：对产品质量信息的需求

{cite:t}`kihlstrom1974a` 将布莱克韦尔定理应用于消费者对产品质量信息的需求。

- 未知状态 $\tilde{s}$ 是产品的质量。
- 消费者可以花费 $c \lambda^2$ 的成本购买 $\lambda$ 单位的信息。
- 随着 $\lambda$ 增大，实验在布莱克韦尔意义上变得更具信息性。

在没有成本的情况下，布莱克韦尔序表明，对于每个期望效用最大化者来说，更多的信息总是更好的。

在存在成本的情况下，消费者选择 $\lambda$ 以最大化信息的净价值。

我们让实验的准确度随 $\phi(\lambda) = 1 - e^{-a \lambda}$ 递增但呈边际递减，并将其代入上述族 $\mu(\phi)$ 中。

我们用德格鲁特熵值来衡量信息的总价值，它在 {cite:t}`kihlstrom1974a` 中代表消费者期望效用的增益。

{doc}`pricing_information` 研究了向对信息价值评估不同的买家出售信息的问题。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Demand for information with a quadratic cost
    name: fig-blackwell-information-demand
---
def gross_value(λ, prior2, U=entropy, rate=2):
    """Gross value of buying λ units of information."""
    accuracy = 1 - np.exp(-rate * λ)
    return degroot_value(make_experiment(accuracy), prior2, U)


λ_grid = np.linspace(0, 1, 200)
c = 0.6

gross_vals = np.array([gross_value(λ, prior2) for λ in λ_grid])
cost_vals = c * λ_grid**2
net_vals = gross_vals - cost_vals
marginal_vals = np.gradient(gross_vals, λ_grid)
marginal_cost = 2 * c * λ_grid
opt_idx = int(np.argmax(net_vals))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(λ_grid, gross_vals, label="Gross value I(λ)",
        color="steelblue", linewidth=2)
ax.plot(λ_grid, cost_vals, label=r"Cost $c\lambda^2$",
        color="tomato", linestyle="--", linewidth=2)
ax.plot(λ_grid, net_vals, label="Net value", color="green", linewidth=2)
ax.axvline(λ_grid[opt_idx], color="green", linestyle=":", linewidth=2,
           label=f"λ* ≈ {λ_grid[opt_idx]:.2f}")
ax.set_xlabel("information purchased λ", fontsize=11)
ax.set_ylabel("value (entropy units)", fontsize=11)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.plot(λ_grid, marginal_vals, label="Marginal value I'(λ)",
         color="steelblue", linewidth=2)
ax2.plot(λ_grid, marginal_cost, label=r"Marginal cost $2c\lambda$",
         color="tomato", linestyle="--", linewidth=2)
ax2.axvline(λ_grid[opt_idx], color="green", linestyle=":", linewidth=2,
            label=f"λ* ≈ {λ_grid[opt_idx]:.2f}")
ax2.set_xlabel("information purchased λ", fontsize=11)
ax2.set_ylabel("marginal value / cost", fontsize=11)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

最优购买量 $\lambda^*$ 出现在边际价值等于边际成本的地方。

边际价值先上升后下降。

在 $\lambda = 0$ 附近，信息的价值是凸的：它仅随实验准确度呈二次方增长。

{cite:t}`radner_stiglitz1984` 表明，这种信息价值的非凹性是普遍存在的，因此随着价格下降，对信息的需求可能从零跳跃到一个正值。

由于准确度存在边际递减效应，边际价值最终会低于不断上升的边际成本，这在此处产生了一个内部最优解。

提高 $c$ 会使边际成本曲线上移，从而降低 $\lambda^*$，而更不对称的先验分布会使边际价值曲线发生变化，从而改变最优解。

## 应用 2：序贯实验设计

{cite:t}`degroot1962` 将不确定性函数框架应用于 *序贯实验设计*。

每一期一个统计学家观测一次抽取并更新后验。

问题是哪一个实验序列最小化累积期望不确定性。

如果一个实验在每一阶段都比另一个更具信息量，那么布莱克韦尔序倾向于在每一时期都使用更好的实验。

{doc}`wald_friedman` 和 {doc}`wald_friedman_2` 研究了一个互补的问题：何时停止从一个固定实验中抽样。

我们现在为不同质量的实验模拟序贯信念更新。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 不同实验质量下的序贯后验路径
    name: fig-blackwell-sequential-paths
---
def sequential_update(μ_matrix, prior, T=20, seed=0):
    """在实验 μ 下模拟 T 次序贯信念更新。"""
    rng = np.random.default_rng(seed)
    N, M = μ_matrix.shape
    beliefs = np.zeros((T + 1, N))
    beliefs[0] = prior.copy()

    true_state = rng.choice(N, p=prior)

    for t in range(T):
        p = beliefs[t]
        signal = rng.choice(M, p=μ_matrix[true_state])
        unnorm = μ_matrix[:, signal] * p
        beliefs[t + 1] = unnorm / unnorm.sum()

    return beliefs, true_state


def plot_sequential_beliefs(θs_compare, prior2, T=25):
    fig, axes = plt.subplots(1, len(θs_compare), figsize=(14, 4), sharey=True)

    for ax, θ in zip(axes, θs_compare):
        μ_t = make_experiment(θ, N=2)
        for seed in range(15):
            beliefs, ts = sequential_update(μ_t, prior2, T=T, seed=seed)
            c = "steelblue" if ts == 0 else "darkorange"
            ax.plot(beliefs[:, 0], alpha=0.35, color=c, linewidth=2)
        ax.axhline(prior2[0], linestyle="--", color="gray", linewidth=2,
                   label="先验")
        ax.axhline(1.0, linestyle=":", color="steelblue", linewidth=2)
        ax.axhline(0.0, linestyle=":", color="darkorange", linewidth=2)
        ax.set_xlabel(r"时期 $t$", fontsize=11)
        if θ == θs_compare[0]:
            ax.set_ylabel(r"后验 $p(s_1 \mid x^t)$", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.03, 0.94, f"θ = {θ}", transform=ax.transAxes, va="top")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

plot_sequential_beliefs([0.2, 0.5, 0.9], prior2, T=30)
```

更具信息量的实验使信念更快地收敛到真相。

{doc}`likelihood_ratio_process` 表明，对数似然比以一个等于实验两行之间库尔贝克-莱布勒散度的速率漂移，并且该散度随 $\theta$ 的增大而上升。

在正确的先验下，后验过程是一个鞅。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 后验鞅性质的无条件蕴含
    name: fig-blackwell-martingale-mean
---
def check_martingale_mean(μ_matrix, prior, T=15, n_paths=2000, seed=0):
    """
    模拟许多信念路径并检验 E[p_t] = p_0。
    """
    rng = np.random.default_rng(seed)
    N, M = μ_matrix.shape
    all_paths = np.zeros((n_paths, T + 1, N))

    for k in range(n_paths):
        true_state = rng.choice(N, p=prior)
        p = prior.copy()
        all_paths[k, 0] = p
        for t in range(T):
            signal = rng.choice(M, p=μ_matrix[true_state])
            unnorm = μ_matrix[:, signal] * p
            p = unnorm / unnorm.sum()
            all_paths[k, t + 1] = p

    mean_path = all_paths[:, :, 0].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_path, color="steelblue", linewidth=2,
            label=r"$\bar p_t(s_1)$（路径均值）")
    ax.axhline(prior[0], linestyle="--", color="tomato", linewidth=2,
               label=fr"先验 $p_0 = {prior[0]:.2f}$")
    ax.set_xlabel(r"时期 $t$", fontsize=12)
    ax.set_ylabel(r"$E[p_t(s_1)]$", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    print(f"先验 = {prior[0]:.4f}")
    print(f"各时期的平均信念均值：{mean_path.mean():.4f}")

check_martingale_mean(μ_info, prior, T=20, n_paths=5000)
```

模拟的横截面均值在每一时期都保持接近先验。

这是后验鞅性质的无条件蕴含。

{doc}`merging_of_opinions` 研究了此类信念鞅的几乎必然收敛性，以及关于意见融合的布莱克韦尔-杜宾斯定理。

(bk_data_processing)=
## 数据处理不等式与粗粒化

布莱克韦尔的条件——对某个马尔可夫核 $Q$ 有 $\nu = \mu Q$ ——与信息论、信息几何和机器学习中支撑 **数据处理不等式**（DPI）和 **粗粒化定理** 的数学操作是同一个。

### f-散度的 DPI

有限空间 $\Omega$ 上两个概率分布 $P$ 和 $Q$ 之间的 **f-散度** 为

$$
D_f(P \| Q) = \sum_{\omega \in \Omega} q_\omega \, f\!\left(\frac{p_\omega}{q_\omega}\right),
$$

其中 $f : (0,\infty) \to \mathbb{R}$ 是满足 $f(1) = 0$ 的凸函数。

特例包括：

| 散度 | 生成元 $f(t)$ |
|:---|:---|
| KL 散度 | $t \log t$ |
| 平方 Hellinger $H^2$ | $(\sqrt{t} - 1)^2 / 2$ |
| 全变差 TV | $\lvert t - 1 \rvert / 2$ |
| 卡方 $\chi^2$ | $(t-1)^2$ |

f-散度这一类别是由 {cite:t}`ali1966` 、 {cite:t}`csiszar1963` 和 {cite:t}`morimoto1963` 独立引入的；另见 {cite:t}`liese2012` 。

{doc}`divergence_measures` 针对连续分布研究了其中若干种散度。

```{prf:theorem} 数据处理不等式
:label: thm-data-processing

对任何 f-散度 $D_f$ 和任何马尔可夫核（随机变换）
$\kappa$ ，以 $P \kappa$ 表示 $P$ 在 $\kappa$ 下的像，我们有

$$
D_f(P \| Q) \geq D_f(P\kappa \| Q\kappa).
$$

如果 $\kappa$ 由关于对 $\{P, Q\}$ 的充分统计量诱导，则等式成立。

这种形式的逆命题需要额外的假设；下面给出一个清晰的二元模型刻画。
```

该证明来自将詹森不等式应用于凸函数 $f$ ，并利用 $\kappa$ 是随机矩阵这一事实 {cite}`csiszar1963` 。

### 与布莱克韦尔充分性条件的联系

在布莱克韦尔的框架中， $\mu$ 和 $\nu$ 是在同一状态空间 $S = \{s_1, \ldots, s_N\}$ 上的实验。

对两个状态，每个实验有两行： $\mu_1 = \mu(s_1, \cdot)$ 和 $\mu_2 = \mu(s_2, \cdot)$ 。

如果 $\nu = \mu Q$ （即 $\nu$ 是 $\mu$ 的混淆），那么对 $(\nu_1, \nu_2) = (\mu_1 Q, \mu_2 Q)$ 就是通过将马尔可夫核 $Q$ 应用于对 $(\mu_1, \mu_2)$ 得到的。

粗粒化定理由此立即蕴含：

$$
D_f(\mu_1 \| \mu_2) \geq D_f(\nu_1 \| \nu_2)
\quad \text{for every f-divergence } D_f,
$$

只要 $\mu \geq \nu$ 在布莱克韦尔序中成立。

对两个状态而言，逆命题也成立（采用关于零概率的通常约定）：如果对每个凸函数 $f$ 都有 $D_f(\mu_1 \| \mu_2) \geq D_f(\nu_1 \| \nu_2)$ ，那么 $\mu \geq \nu$ 。

当状态数多于两个时，各行之间的散度并不能刻画布莱克韦尔序。

因此一个更具信息量的实验总是会产生 *更分离的* 条件信号分布，在每个 f-散度的意义下同时如此。

由此，DPI 是关于状态 *可区分性* 的一个陈述：混淆一个实验会使得在每种统计可分性度量下，状态都更难被区分。

对二元实验，等式条件将 DPI 直接联系回布莱克韦尔：对某个严格凸的 $f$ 有 $D_f(\mu_1 Q \| \mu_2 Q) = D_f(\mu_1 \| \mu_2)$ ，当且仅当 $Q$ 是关于 $(\mu_1, \mu_2)$ 的充分统计量。

一旦充分性成立，则对每个凸函数 $f$ 等式都成立 {cite}`liese2012` 。

### 信息几何：陈氏定理

DPI 有一个无穷小的、微分几何的伴侣。

**陈氏定理**  {cite}`chentsov1981` 断言， **Fisher 信息度量** 至多相差一个常数因子，是概率单纯形上在马尔可夫嵌入（由充分统计量诱导的变换）下不变的 *唯一* 黎曼度量。

Fisher 信息矩阵 $I_F(\theta)$ 还在每个马尔可夫核下收缩：

$$
I_F(\theta;\, \mu) \succeq I_F(\theta;\, \mu\kappa)
\quad \text{for every differentiable family } \{\mu_\theta\} \text{ and every Markov kernel } \kappa.
$$

等式成立当且仅当 $\kappa$ 是关于 $\theta$ 的充分统计量。

唯一性条款说明，在所有黎曼度量中，唯有 Fisher 度量在充分统计量下保持不变。

在 {doc}`gallant_tauchen` 中，当辅助模型嵌套结构模型时，结构模型的 Fisher 信息设定了得分匹配所能达到的效率界限。

关于信息几何及其与充分性联系的详尽处理，参见 {cite:t}`amari_nagaoka2000` 。

### 机器学习中的信息瓶颈

{cite:t}`tishby_pereira_bialek1999` 的 **信息瓶颈** 方法提供了 DPI 在机器学习中一个突出的应用。

给定输入 $X$ 和目标 $Y$ 上的联合分布 $p(X, Y)$ ，目标是找到一个由随机映射 $p(T \mid X)$ 形成的压缩表示 $T$ ，使其在用尽可能少的比特描述 $X$ 的同时，尽可能多地保留关于 $Y$ 的信息。

该方法最小化拉格朗日量

$$
\mathcal{L}[p(T \mid X)] = I(X;\, T) - \beta \, I(T;\, Y),
$$

其中 $I(\cdot\,;\,\cdot)$ 表示互信息， $\beta \geq 0$ 支配压缩-相关性权衡。

由于 $Y - X - T$ 构成一个马尔可夫链（T 仅从 X 导出），DPI 蕴含

$$
I(T;\, Y) \leq I(X;\, Y),
$$

等式成立当且仅当 $T$ 是 $X$ 关于 $Y$ 的 **充分统计量**，也就是说，如果 $Y - T - X$ 也构成一个马尔可夫链。

布莱克韦尔序解释了为什么对 $X$ 的任何确定性或随机后处理都无法增加与 $Y$ 的互信息：应用于 $X$ 的任何马尔可夫核都是布莱克韦尔意义下的混淆，而 DPI 是粗粒化定理的互信息形式。

用机器学习的语言来说，信息瓶颈在 $X$ 的所有混淆中搜索那个在压缩预算约束下最好地保留关于 $Y$ 相关信息的混淆。

在一个具有输入 $X$ 、目标 $Y$ 以及层 $X \to T_1 \to T_2 \to \cdots \to T_L \to \hat{Y}$ 的深度神经网络中，每一层的表示都是前一层的混淆。

于是 DPI 蕴含如下不等式链

$$
I(X;\, Y) \geq I(T_1;\, Y) \geq I(T_2;\, Y) \geq \cdots \geq I(T_L;\, Y),
$$

因此后继各层关于 $Y$ 的信息只会损失，绝不会增加。

这一观察被 {cite}`shwartz_ziv_tishby2017` 置于研究深度网络所学内容的中心。

{numref}`fig-blackwell-value-by-quality` 已经阐释了这一点：随着实验质量 $\theta$ 增加，每一种信息量度量都单调上升。

DPI 反过来说的是同一件事：混淆（降低 $\theta$ ）只会收缩这些度量。

### DPI-布莱克韦尔对应的小结

下表汇集了布莱克韦尔框架与数据处理和粗粒化文献之间的精确对应。

| 布莱克韦尔 / 德格鲁特 | 数据处理 / 粗粒化 |
|:---|:---|
| 混淆 $\nu = \mu Q$ | 将马尔可夫核 $\kappa$ 应用于对 $(P, Q) = (\mu_1, \mu_2)$ |
| $\mu \geq \nu$ 在布莱克韦尔序中 | 对每个 f-散度 $D_f(\mu_1 \| \mu_2) \geq D_f(\nu_1 \| \nu_2)$ |
| 充分性（ $Q$ 不丢弃任何东西） | DPI 中的等式；在二元模型中，一个严格凸的 $f$ 就已刻画充分性 |
| 德格鲁特价值 $I(\mu; U_H)$ | 互信息 $I(\tilde{x}_\mu;\, \tilde{s})$ （香农 DPI） |
| $\mu$ 与 $\nu$ 下的后验展开 | $\mu$ 下各行之间的 $D_f$ 更大 |
| 两状态下的布莱克韦尔定理 | 对所有 $f$ 的 $D_f$ 占优 $\Leftrightarrow$ 单个马尔可夫核见证占优 |
| 陈氏唯一性定理 | Fisher 信息是在马尔可夫嵌入（充分统计量）下唯一不变的度量；它也在马尔可夫核下收缩 |
| 信息瓶颈 $I(T;Y) \leq I(X;Y)$ | 应用于马尔可夫链 $Y{-}X{-}T$ 的互信息 DPI |


(bk_lr_learning)=
## 与贝叶斯似然比学习的关系

讲座 {doc}`likelihood_bayes` 是这里所发展框架的一个动态的两状态特例。

设 $S = \{s_1, s_2\}$ ，其中 $s_1 \leftrightarrow f$ 且 $s_2 \leftrightarrow g$ ，其中 $f$ 和 $g$ 是两个候选的数据生成密度。

那么单个观测是一个具有行 $f(\cdot)$ 和 $g(\cdot)$ 的布莱克韦尔实验，而历史 $w^t = (w_1, \ldots, w_t)$ 定义了一个更丰富的实验 $\mu_t$ 。

这里我们将第一次抽取的时间标为 $t = 1$ ，并把先验写为 $\pi_0$ ，与 {doc}`likelihood_ratio_process` 中一致；而 {doc}`likelihood_bayes` 将第一次抽取的时间标为 $t = 0$ ，并把先验写为 $\pi_{-1}$ 。

由于总是可以丢弃最后 $t-s$ 个观测，对每个 $t > s$ ， $\mu_t$ 都布莱克韦尔占优 $\mu_s$ 。

似然比过程

$$
L(w^t) = \prod_{i=1}^t \frac{f(w_i)}{g(w_i)}
$$

是 $\mu_t$ 的一个充分统计量，而后验

$$
\pi_t = \Pr(s_1 \mid w^t)
= \frac{\pi_0 L(w^t)}{\pi_0 L(w^t) + 1 - \pi_0}
$$

是基尔斯特罗姆在这个两状态设定下的标准实验。

它的鞅性质 $E[\pi_t] = \pi_0$ 正是上面为后验分布证明的均值保持结果。

同样地， $\mu_t \geq \mu_s$ 意味着 $\pi_t$ 的分布是 $\pi_s$ 分布的一个均值保持展开，因此额外的数据会把信念推向离 $0$ 和 $1$ 更远的地方，同时在每个凹不确定性函数下降低期望不确定性。

### 小结表

下表在不重复前面论证的情况下记录了两个讲座之间的对照词典。

| {doc}`likelihood_bayes` 中的概念 | 本讲座中的概念 |
|---|---|
| 状态 $\{f, g\}$ | 状态空间 $S = \{s_1, s_2\}$ |
| 密度 $f(\cdot)$ 、 $g(\cdot)$ | 实验矩阵 $\mu$ 的各行 |
| 单次抽取 $w_t$ | 具有连续信号空间的布莱克韦尔实验 |
| $t$ 次 IID 抽取的历史 $w^t$ | 布莱克韦尔占优 $\mu_s$（ $s < t$ ）的更丰富实验 $\mu_t$ |
| 似然比 $L(w^t)$ | $\mu_t$ 的充分统计量 |
| 先验 $\pi_0$ | 1-单纯形 $[0,1]$ 上的先验 $p \in P$ |
| 后验 $\pi_t$ | $P = [0,1]$ 上的后验（基尔斯特罗姆的标准实验） |
| $\pi_t$ 在各历史上的分布 | $\hat{\mu}^c$ （基尔斯特罗姆的后验分布） |
| 鞅性质 $E[\pi_t] = \pi_0$ | $\hat{\mu}^c$ 的均值保持 |
| $\pi_t \to 0$ 或 $1$ 几乎必然 | 后验向顶点展开（极限中的 MPS） |
| $\pi_t$ 的熵的期望下降 | 德格鲁特价值 $I(\mu_t; U_H)$ ，即互信息 |
| 更多抽取 $\Rightarrow$ 对所有决策者更好 | 布莱克韦尔序 $\mu_t \geq \mu_s$ |
| 混淆（丢弃最后 $t - s$ 次抽取） | 满足 $\mu_s = \mu_t Q$ 的随机变换 $Q$ |

## 小结

布莱克韦尔定理在统计实验上确定了一个 *偏序*，具有三个等价的刻画：

| 标准 | 条件 |
|-----------|-----------|
| 经济学 | 每个决策者弱地偏好 $\mu$ 甚于 $\nu$ ： $B(\mu, A, u) \supseteq B(\nu, A, u)$ |
| 充分性 | $\nu$ 是 $\mu$ 的混淆：对某个马尔可夫矩阵 $Q$ 有 $\nu = \mu Q$ |
| 不确定性 | 对每个先验 $p$ 和每个凹函数 $U$ ， $\mu$ 更多地降低期望不确定性 |

基尔斯特罗姆的贝叶斯阐述将 *后验分布* 置于中心。

一个更具信息量的实验会生成一个更为分散、但均值与先验相同的后验分布。

恰当的概率语言是后验信念单纯形上的凸序。

在两状态情形下，这简化为 $[0, 1]$ 上熟悉的 SOSD / 积分 CDF 检验。

德格鲁特的贡献在于将比较从特定效用函数扩展到全部凹不确定性函数类。

布莱克韦尔序蕴含了每个 f-散度的数据处理不等式，而 {doc}`likelihood_bayes` 中的似然比学习则是该框架的一个动态特例。

## 练习

```{exercise}
:label: bk_ex1

第一节发现没有随机变换能将 $\nu = \mu Q$ 映射回 $\mu$，但该节只提供了一个数值残差作为证据。

本练习提供一个证明。

假设 $\mu = \nu Q'$ 对于某个马尔可夫矩阵 $Q'$ 成立，即 $\tilde{x}_\mu$ 是通过一个不依赖于 $\tilde{s}$ 的随机化从 $\tilde{x}_\nu$ 生成的。

1. 证明 $\mu$ 下的每个后验都是 $\nu$ 下后验的加权平均：

$$
p^\mu(x) = \sum_{y} \Pr(\tilde{x}_\nu = y \mid \tilde{x}_\mu = x) \, p^\nu(y) .
$$

2. 在均匀先验下，计算第一节中 $\mu$ 和 $\nu$ 生成的 $s_1$ 的后验概率。

3. 得出 $\mu$ 不是 $\nu$ 的模糊化这一结论。
```

```{solution-start} bk_ex1
:class: dropdown
```

对于第 1 部分，状态和两个信号构成一个马尔可夫链 $\tilde{s} \to \tilde{x}_\nu \to \tilde{x}_\mu$，因为 $Q'$ 不依赖于状态。

因此 $\Pr(\tilde{s} = s_i \mid \tilde{x}_\nu = y, \tilde{x}_\mu = x) = \Pr(\tilde{s} = s_i \mid \tilde{x}_\nu = y)$，并且

$$
p_i^\mu(x)
= \sum_{y} \Pr(\tilde{x}_\nu = y \mid \tilde{x}_\mu = x) \, \Pr(\tilde{s} = s_i \mid \tilde{x}_\nu = y)
= \sum_{y} \Pr(\tilde{x}_\nu = y \mid \tilde{x}_\mu = x) \, p_i^\nu(y) .
$$

因此，被模糊化的实验的每个后验都落在原始实验后验的凸包内。

```{code-cell} ipython3
c2 = np.array([0.5, 0.5])
post_μ1, _ = compute_posteriors(μ, c2)
post_ν1, _ = compute_posteriors(ν, c2)

lo, hi = post_ν1[:, 0].min(), post_ν1[:, 0].max()
outside = (post_μ1[:, 0] < lo - 1e-12) | (post_μ1[:, 0] > hi + 1e-12)

print("posteriors on s1 under μ:", post_μ1[:, 0].round(4))
print("posteriors on s1 under ν:", post_ν1[:, 0].round(4))
print(f"interval spanned by ν's posteriors: [{lo:.4f}, {hi:.4f}]")
print("μ's posteriors outside that interval:", post_μ1[outside, 0].round(4))
```

在 $\nu$ 下，$s_1$ 的后验为 $3/4$ 和 $1/4$。

在 $\mu$ 下，它们为 $6/7$、$1/2$ 和 $1/7$。

后验 $6/7$ 和 $1/7$ 落在 $[1/4, 3/4]$ 之外，因此没有马尔可夫矩阵 $Q'$ 能从 $\nu$ 产生 $\mu$。

这一论证解释了第一节中出现的较大残差：模糊化只能将后验拉向先验，而 $\mu$ 使后验远离先验的程度超出了 $\nu$ 所能达到的范围。

```{solution-end}
```

```{exercise}
:label: bk_ex2

布莱克韦尔的序是一个偏序，{doc}`pricing_information` 正是建立在这一事实之上。

考虑两个具有两个状态和两个信号的实验，

$$
\mu_a = \begin{bmatrix} 1 & 0 \\ 0.5 & 0.5 \end{bmatrix} ,
\qquad
\mu_b = \begin{bmatrix} 0.5 & 0.5 \\ 0 & 1 \end{bmatrix} ,
$$

以及均匀先验。

1. 计算每个实验下 $s_1$ 后验概率的分布，以及信号与状态之间的互信息。

2. 使用看涨期权检验 $\mathbb{E}[\max(p - t, 0)]$ 证明这两个后验分布均不是对方的均值保持展开。

3. 对于凹的不确定性函数 $U_t(p) = -\max(p_1 - t, 0)$，取 $t = 0.2$ 和 $t = 0.6$，计算每个实验的德格鲁特价值，并解释结果。
```

```{solution-start} bk_ex2
:class: dropdown
```

在 $\mu_a$ 下，信号 $x_2$ 只在状态 $s_2$ 中出现，因此它揭示了 $s_2$。

在 $\mu_b$ 下，信号 $x_1$ 只在状态 $s_1$ 中出现，因此它揭示了 $s_1$。

```{code-cell} ipython3
μ_a = np.array([[1.0, 0.0],
                [0.5, 0.5]])
μ_b = np.array([[0.5, 0.5],
                [0.0, 1.0]])

for name, m in [("μ_a", μ_a), ("μ_b", μ_b)]:
    posts, probs = compute_posteriors(m, c2)
    print(f"{name}: posteriors on s1 = {posts[:, 0].round(4)}, "
          f"probabilities = {probs.round(4)}, "
          f"mutual information = {degroot_value(m, c2, entropy):.4f}")
```

这两个后验分布互为镜像。

重新标记状态和信号可以将一个实验映射为另一个实验，因此两者具有相同的互信息。

```{code-cell} ipython3
def call_values(m, prior, ts):
    "E[max(p_1 - t, 0)] under experiment m for each threshold t."
    posts, probs = compute_posteriors(m, prior)
    return np.array([(np.maximum(posts[:, 0] - t, 0) * probs).sum() for t in ts])

ts = np.linspace(0, 1, 201)
diff_ab = call_values(μ_a, c2, ts) - call_values(μ_b, c2, ts)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, diff_ab, linewidth=2)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("threshold $t$")
ax.set_ylabel(r"$E[\max(p - t, 0)]$ under $\mu_a$ minus under $\mu_b$")
plt.show()
```

该差值出现了符号变化。

这两个后验分布都不是对方的均值保持展开，因此根据基尔斯特罗姆定理，这两个实验互相都不是布莱克韦尔充分的。

```{code-cell} ipython3
def U_call(t):
    "The concave uncertainty function -max(p_1 - t, 0)."
    return lambda p: -max(p[0] - t, 0.0)

for t in (0.2, 0.6):
    print(f"t = {t}:  I(μ_a) = {degroot_value(μ_a, c2, U_call(t)):.4f},  "
          f"I(μ_b) = {degroot_value(μ_b, c2, U_call(t)):.4f}")
```

当 $t = 0.2$ 时，$\mu_a$ 的价值为 $0.05$，$\mu_b$ 的价值为零。

当 $t = 0.6$ 时，排名发生了反转。

除符号和一个仿射项外，$U_t$ 是一个在两种行动间选择、并在信念为 $t$ 处切换行动的决策者的价值函数。

在 $t = 0.2$ 处切换的决策者只看重那些将后验推低于 $0.2$ 的信号，而 $\mu_b$ 从未做到这一点。

在 $t = 0.6$ 处切换的决策者认为 $\mu_b$ 中揭示 $s_1$ 的信号比 $\mu_a$ 中将后验提高到 $2/3$ 的信号更有价值。

互信息对两个实验的排名相同，但决策者对它们的看法却存在分歧。

单一的不确定性函数无法确定布莱克韦尔比较的结果。

```{solution-end}
```

```{exercise}
:label: bk_ex3

本练习将布莱克韦尔的序与 {doc}`likelihood_bayes` 中的学习动态联系起来。

设单次抽取来自实验 `make_experiment(θ)`，其中 $\theta = 0.3$，因此信号与状态匹配的概率为 $(1 + \theta) / 2$。

观察 $t$ 次独立抽取。

1. 解释为什么等于 $x_1$ 的抽取次数 $k$ 是一个充分统计量，并写出 $k$ 对应的 $2 \times (t + 1)$ 实验矩阵 $\mu_t$。

2. 构造丢弃 $t + 1$ 次抽取中最后一次的马尔可夫矩阵 $Q_t$，并通过数值方法验证 $\mu_t = \mu_{t+1} Q_t$。

3. 在均匀先验下，使用看涨期权检验来验证对于 $t = 1, \ldots, 29$，$t + 1$ 次抽取后的后验分布是 $t$ 次抽取后的后验分布的均值保持展开。

4. 绘制 $k$ 与状态之间的互信息随 $t$ 变化的图像，并将其与上界 $\log 2$ 进行比较。
```

```{solution-start} bk_ex3
:class: dropdown
```

记 $a = (1 + \theta)/2$ 以及 $b = (1 - \theta)/2$。

在状态 $s_1$ 中，$t$ 次抽取中含有 $k$ 个 $x_1$ 的似然为 $a^k b^{t-k}$，在状态 $s_2$ 中为 $b^k a^{t-k}$。

似然比仅通过 $k$ 依赖于数据，因此 $k$ 是充分的。

$\mu_t$ 的各行是二项分布 $\text{Bin}(t, a)$ 和 $\text{Bin}(t, b)$。

这些抽取是可交换的，正如 {doc}`exchangeable` 中所讨论的那样。

给定 $t + 1$ 次抽取中有 $k'$ 个 $x_1$，无论在哪种状态下，最后一次抽取等于 $x_1$ 的概率均为 $k' / (t + 1)$。

丢弃它会以该概率将 $k'$ 移动到 $k' - 1$，否则 $k'$ 保持不变。

该核不依赖于状态，因此它是一个模糊化。

```{code-cell} ipython3
from scipy.stats import binom

def draws_experiment(θ, t):
    "Experiment matrix for the number of x_1 signals in t independent draws."
    a, b = (1 + θ) / 2, (1 - θ) / 2
    k = np.arange(t + 1)
    return np.vstack([binom.pmf(k, t, a), binom.pmf(k, t, b)])

def discard_last(t):
    "Markov matrix from k' in {0, ..., t+1} to k in {0, ..., t}."
    Q_t = np.zeros((t + 2, t + 1))
    for kp in range(t + 2):
        if kp >= 1:
            Q_t[kp, kp - 1] = kp / (t + 1)
        if kp <= t:
            Q_t[kp, kp] = 1 - kp / (t + 1)
    return Q_t

θ_ex = 0.3
for t in (1, 4, 9):
    err = np.abs(draws_experiment(θ_ex, t + 1) @ discard_last(t)
                 - draws_experiment(θ_ex, t)).max()
    print(f"t = {t}:  max |μ_(t+1) Q_t - μ_t| = {err:.1e}")

is_mps = all(
    (call_values(draws_experiment(θ_ex, t + 1), c2, ts)
     - call_values(draws_experiment(θ_ex, t), c2, ts)).min() >= -1e-12
    for t in range(1, 30))
print(f"\nposteriors after t + 1 draws are an MPS of those after t draws: {is_mps}")
```

```{code-cell} ipython3
t_grid = np.arange(1, 81)
mi = [degroot_value(draws_experiment(θ_ex, t), c2, entropy) for t in t_grid]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_grid, mi, linewidth=2, label="mutual information")
ax.axhline(np.log(2), color="black", linestyle="--", linewidth=1, label=r"$\log 2$")
ax.set_xlabel("number of draws $t$")
ax.set_ylabel("mutual information")
ax.legend()
plt.show()
```

每多一次抽取就会使后验分布更加分散，而其均值始终保持在先验处，这正是 {doc}`likelihood_bayes` 中研究的鞅性质。

随着后验集中于真实状态，互信息逐渐上升，趋向于均匀先验的熵 $\log 2$。

```{solution-end}
```
