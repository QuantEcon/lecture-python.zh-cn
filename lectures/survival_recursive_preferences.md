---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: 递归偏好下的生存与长期动态
  headings:
    Overview: 概述
    Environment: 环境
    Environment::Aggregate endowment: 总禀赋
    Environment::Heterogeneous beliefs: 异质信念
    Environment::Recursive preferences: 递归偏好
    Planner's problem: 规划者问题
    Planner's problem::Modified discount factors: 修正贴现因子
    'Planner''s problem::State variable: Pareto share': 状态变量：帕累托份额
    Planner's problem::HJB equation: HJB 方程
    Planner's problem::HJB equation::From discrete to continuous time: 从离散时间到连续时间
    Planner's problem::HJB equation::Exact reduced ODE: 精确的约化 ODE
    Survival conditions: 生存条件
    Wealth dynamics decomposition: 财富动态分解
    Wealth dynamics decomposition::Portfolio returns: 组合收益
    Wealth dynamics decomposition::Consumption-wealth ratios: 消费-财富比
    Wealth dynamics decomposition::Two comparative statics: 两个比较静态分析
    Wealth dynamics decomposition::Risk aversion versus the IES: 风险厌恶与 IES 之比较
    Survival regions: 生存区域
    Three survival channels: 三个生存渠道
    Varying the IES: 改变 IES
    Asymptotic results: 渐近结果
    The separable case: 可分情形
    Asset pricing implications: 资产定价含义
    Optimistic and pessimistic distortions: 乐观和悲观扭曲
    Long-run consumption distribution: 长期消费分布
    Summary: 总结
---

(survival_recursive_preferences)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 递归偏好下的生存与长期动态

```{index} single: Survival; Recursive Preferences
```

```{contents} Contents
:depth: 2
```

## 概述

本讲座研究 {cite:t}`Borovicka2020` 中关于长期生存的理论。

经典的**市场选择假说**认为，信念较不准确的主体最终会被市场淘汰。

{cite:t}`Sandroni2000Markets` 和 {cite:t}`Blume_Easley2006` 针对具有可分期望效用偏好、且总禀赋有上界并远离零的经济体证明了这一结果。

Borovička 表明，在 Epstein-Zin 递归偏好下，这一结论可能不成立。

在递归偏好下，具有扭曲信念的主体可以生存，甚至可以占据主导地位。

关键机制在于递归偏好将风险厌恶与跨期替代弹性(IES)分离开来。

有三个渠道对生存至关重要：

1. *风险溢价渠道*奖励更乐观的主体持有更多的风险资产。
1. *投机波动率渠道*通过对数收益率波动率惩罚激进的头寸。
1. *储蓄渠道*在 IES 不等于 1 时改变消费和储蓄决策。

在可分偏好下，风险厌恶等于 IES 的倒数，这一约束将三个渠道联系在一起。

三者之和此时仅取决于两个主体信念的相对熵，信念更准确的主体将占据主导地位。

在递归偏好下，风险厌恶 $\gamma$ 与 IES 倒数 $\rho$ 之间的比较决定了可能出现的长期结果。

只有当 $\gamma > \rho$ 时，两个主体才可能都生存下来，这意味着主体偏好不确定性的早期解决。

只有当 $\gamma < \rho$ 时，每个主体才可能以正概率占据主导地位。

IES 大于 1 会增强乐观主体的储蓄渠道，但共存并不要求这一条件成立。

```{note}
本文建立在 {cite:t}`Duffie_Epstein1992a` 的连续时间递归效用表述之上，
并采用 {cite:t}`Dumas_Uppal_Wang2000` 的规划者问题方法。

市场选择假说的重要基础由
{cite:t}`DeLong_etal1991` 和 {cite:t}`Blume_Easley1992` 奠定。
```

我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt
from scipy.optimize import brentq
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
mpl.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
```

## 环境

该经济体包含两个无限期存活的主体，用 $n \in \{1, 2\}$ 索引。

这两个主体具有相同的递归偏好，但对总禀赋增长持有不同的信念。

我们将 Borovička 的信念扭曲 $u^n$ 记作 $\omega^n$。

### 总禀赋

在真实概率测度 $P$ 下，总禀赋满足

$$
d \log Y_t = \mu_Y dt + \sigma_Y dW_t, \quad Y_0 > 0
$$ (eq:srp_endowment)

其中 $W$ 是标准布朗运动，$\mu_Y$ 是漂移，$\sigma_Y > 0$ 是波动率。

### 异质信念

主体 $n$ 认为漂移是 $\mu_Y + \omega^n \sigma_Y$ 而非 $\mu_Y$。

参数 $\omega^n$ 在 $\omega^n > 0$ 时衡量乐观程度，在 $\omega^n < 0$ 时衡量悲观程度。

主体 $n$ 的主观概率测度 $Q^n$ 由 Radon–Nikodym 导数定义

$$
M_t^n = \frac{dQ^n}{dP}\bigg|_t = \exp\left(-\frac{1}{2} |\omega^n|^2 t + \omega^n W_t\right)
$$ (eq:radon_nikodym)

$M^n$ 是一个似然比过程，是 {doc}`likelihood_ratio_process` 中研究的过程在连续时间中的对应物。

在 $Q^n$ 下，过程 $W_t^n = W_t - \omega^n t$ 是一个布朗运动，主体 $n$ 感知到

$$
d \log Y_t = (\mu_Y + \omega^n \sigma_Y) dt + \sigma_Y dW_t^n .
$$

$\omega^n > 0$ 的主体对禀赋增长持乐观态度，而 $\omega^n < 0$ 的主体则持悲观态度。

信念是教条式的：主体 $n$ 在观测到 $Y$ 之后，永远不会修正 $\omega^n$。

对于每个有限的 $t$，$Q^n$ 和 $P$ 在 $\mathcal{F}_t$ 上是等价的。

在 $P$ 下，$\log M_t^n$ 的漂移是常数 $-\frac{1}{2}(\omega^n)^2$，因此当 $\omega^n \neq 0$ 时，$M_t^n \to 0$ 在 $P$ 几乎必然成立，从而 $Q^n$ 和 $P$ 在 $\mathcal{F}_\infty$ 上是互为奇异的。

{doc}`merging_of_opinions` 中的 Blackwell–Dubins 定理要求在 $\mathcal{F}_\infty$ 上绝对连续，因此该定理在此不适用。

$\omega^n \neq 0$ 的主体永远不会与 $P$ 达成一致，而 $\omega^1 \neq \omega^2$ 的主体之间也永远不会彼此达成一致。

市场选择取代了意见趋同：悬而未决的问题是，市场是否会淘汰信念错误的主体。

### 递归偏好

两个主体都具有 Epstein-Zin 递归偏好。

我们用 $\gamma > 0$ 表示相对风险厌恶，用 $\rho > 0$ 表示 IES 的倒数，用 $\beta > 0$ 表示时间偏好率。

Duffie-Epstein-Zin 幸福函数为

$$
F(C, \nu)
= \beta \frac{C^{1-\gamma}}{1-\gamma}
\cdot
\left(\frac{(1-\gamma) - (1-\rho)\nu / \beta}{\rho - \gamma}\right)^{(\gamma - \rho)/(1-\rho)}
$$ (eq:felicity)

其中 $\nu$ 是内生的贴现率。

```{note}
在离散时间中，Epstein-Zin 偏好通过 CES 聚合器将当前消费与未来效用的确定性等价物聚合在一起(参见 {doc}`advanced:doubts_or_variability`)。

在连续时间中不存在"下一期 $V_{t+1}$"，因此 {cite:t}`Duffie_Epstein1992a` 将该递归重新表述为一个依赖于主体自身延续值率 $\nu$ 的幸福函数 $F(C,\nu)$。

这两种表述编码了相同的风险厌恶 $\gamma$ 与 IES 倒数 $\rho$ 的分离。

当 $\gamma = \rho$ 时，偏好退化为标准的可分 CRRA 情形。
```

## 规划者问题

遵循 {cite:t}`Dumas_Uppal_Wang2000`，我们通过社会规划者问题来研究均衡配置。

规划者为两个主体选择消费份额 $z^1$ 和 $z^2 = 1 - z^1$ 以及贴现率过程 $\nu^n$。

### 修正贴现因子

将信念扭曲吸收进修正贴现因子 $\tilde{\lambda}^n = \lambda^n M^n$ 是很方便的，其中 $M^n$ 是 Radon-Nikodym 导数 {eq}`eq:radon_nikodym`。

这些过程满足

$$
d \log \tilde{\lambda}_t^n
= -\left(\nu_t^n + \frac{1}{2} (\omega^n)^2\right) dt + \omega^n dW_t .
$$ (eq:modified_discount)

```{exercise}
:label: ex_modified_discount

推导 {eq}`eq:modified_discount`。

*提示：* 使用 $\log \tilde{\lambda}^n = \log \lambda^n + \log M^n$。帕累托权重 $\lambda^n$ 按 $d\log \lambda_t^n = -\nu_t^n \, dt$ 演化，而 $\log M_t^n$ 由 {eq}`eq:radon_nikodym` 给出。
```

```{solution-start} ex_modified_discount
:class: dropdown
```

由定义 $\tilde{\lambda}^n = \lambda^n M^n$，我们有

$$
\log \tilde{\lambda}_t^n = \log \lambda_t^n + \log M_t^n.
$$

帕累托权重满足 $d\log \lambda_t^n = -\nu_t^n \, dt$。

由 {eq}`eq:radon_nikodym`，$\log M_t^n = -\frac{1}{2}|\omega^n|^2 t + \omega^n W_t$，因此

$$
d \log M_t^n = -\tfrac{1}{2}(\omega^n)^2 \, dt + \omega^n \, dW_t.
$$

将两者相加：

$$
d \log \tilde{\lambda}_t^n = -\nu_t^n \, dt - \tfrac{1}{2}(\omega^n)^2 \, dt + \omega^n \, dW_t = -\left(\nu_t^n + \tfrac{1}{2}(\omega^n)^2\right) dt + \omega^n \, dW_t.
$$

```{solution-end}
```

### 状态变量：帕累托份额

关键的状态变量是主体 1 的帕累托份额：

$$
\upsilon = \frac{\tilde{\lambda}^1}{\tilde{\lambda}^1 + \tilde{\lambda}^2} \in (0, 1)
$$ (eq:pareto_share)

它刻画了主体 1 在规划者配置中的相对权重。

定义对数几率比 $\vartheta = \log(\upsilon / (1 - \upsilon))$。

其动态为

$$
d\vartheta_t = \underbrace{\left[\nu_t^2 + \frac{1}{2}(\omega^2)^2 - \nu_t^1 - \frac{1}{2}(\omega^1)^2\right]}_{m_{\vartheta}(\upsilon_t)} dt + (\omega^1 - \omega^2) dW_t
$$ (eq:log_odds)

漂移 $m_\vartheta(\upsilon)$ 决定了帕累托份额的长期行为。

```{exercise}
:label: ex_log_odds

从 {eq}`eq:modified_discount` 和定义 $\vartheta = \log(\upsilon/(1-\upsilon))$ 推导 {eq}`eq:log_odds`。

*提示：* 首先证明 $\vartheta = \log \tilde{\lambda}^1 - \log \tilde{\lambda}^2$，然后将两个 SDE 相减。
```

```{solution-start} ex_log_odds
:class: dropdown
```

由于 $\upsilon = \tilde{\lambda}^1 / (\tilde{\lambda}^1 + \tilde{\lambda}^2)$，我们有 $1 - \upsilon = \tilde{\lambda}^2 / (\tilde{\lambda}^1 + \tilde{\lambda}^2)$，因此

$$
\vartheta = \log\frac{\upsilon}{1-\upsilon} = \log \tilde{\lambda}^1 - \log \tilde{\lambda}^2.
$$

由 {eq}`eq:modified_discount`，两个对数贴现因子的 SDE 为

$$
d\log \tilde{\lambda}^1_t = -\left(\nu_t^1 + \tfrac{1}{2}(\omega^1)^2\right)dt + \omega^1 dW_t,
$$

$$
d\log \tilde{\lambda}^2_t = -\left(\nu_t^2 + \tfrac{1}{2}(\omega^2)^2\right)dt + \omega^2 dW_t.
$$

用第一个减去第二个：

$$
d\vartheta_t = \left[\nu_t^2 + \tfrac{1}{2}(\omega^2)^2 - \nu_t^1 - \tfrac{1}{2}(\omega^1)^2\right]dt + (\omega^1 - \omega^2)dW_t.
$$

```{solution-end}
```

### HJB 方程

齐次性将规划者问题简化为关于单一状态变量 $\upsilon$ 的非线性 ODE。

因为每个主体的效用关于消费是 $1-\gamma$ 次齐次的，规划者的值函数关于权重 $(\tilde{\lambda}^1, \tilde{\lambda}^2)$ 是一次齐次的，关于 $Y$ 是 $1-\gamma$ 次齐次的。

它可以分解为 $J(\tilde{\lambda}, Y) = (\tilde{\lambda}^1 + \tilde{\lambda}^2) Y^{1-\gamma} \tilde{J}(\upsilon)$，从而消除了 $Y$ 作为状态变量。

#### 从离散时间到连续时间

在离散时间中，规划者通过在每个日期选择配置来最大化主体效用的加权和。

贝尔曼方程为

$$
\tilde{J}(\upsilon) = \max_{z^1, z^2} \left\{ \upsilon \, u(z^1) + (1-\upsilon) \, u(z^2) + \delta \, \mathbb{E}\left[\tilde{J}(\upsilon')\right] \right\}.
$$

这里 $\delta \in (0, 1)$ 是单期贴现因子；本讲座其他地方的 $\beta$ 则是连续时间下的时间偏好率。

在连续时间中，时期长度缩小为 $dt$。

$[t, t+dt)$ 上的"流量收益"变为 $\left[\upsilon F(z^1, \nu^1) + (1-\upsilon)F(z^2, \nu^2)\right] dt$，其中 $F$ 是 Duffie-Epstein-Zin 幸福函数 {eq}`eq:felicity`。

值函数在 $dt$ 上的期望变化由**无穷小生成元** $\mathcal{L}$ 捕获。

对于扩散过程 $d\upsilon = m \, dt + s \, dW$，伊藤引理给出

$$
\mathcal{L}\tilde{J}(\upsilon) = m(\upsilon)\,\tilde{J}'(\upsilon) + \tfrac{1}{2} s(\upsilon)^2 \, \tilde{J}''(\upsilon),
$$

其中 $m$ 和 $s$ 是帕累托份额的漂移和扩散。

这是 $\delta \, \mathbb{E}[\tilde{J}(\upsilon')] - \tilde{J}(\upsilon)$ 的连续时间类比：它衡量随着 $\upsilon$ 演化，值函数如何漂移和波动。

令流量收益加上期望资本收益等于零，得到示意性的 HJB 方程：

$$
0 = \sup_{(z^1,z^2,\nu^1,\nu^2)} \left\{ \upsilon F(z^1, \nu^1) + (1-\upsilon) F(z^2, \nu^2) + \mathcal{L} \tilde{J}(\upsilon) \right\}
$$ (eq:hjb_sketch)

满足约束 $z^1 + z^2 \leq 1$。

#### 精确的约化 ODE

{cite:t}`Borovicka2020` 的命题 2.3 在代入齐次性约化 $J(\tilde{\lambda}, Y) = (\tilde{\lambda}^1 + \tilde{\lambda}^2) Y^{1-\gamma} \tilde{J}(\upsilon)$ 以及 $\upsilon$ 和 $Y$ 的动态后，给出了精确的 HJB 方程：

$$
0 = \sup_{(z^1, z^2, \nu^1, \nu^2)} \;
\upsilon \, F(z^1, \nu^1) + (1 - \upsilon) \, F(z^2, \nu^2)
$$ (eq:hjb)

$$
+ \left[
-\upsilon \nu^1 - (1-\upsilon)\nu^2
+ \bigl(\upsilon \omega^1 + (1-\upsilon)\omega^2\bigr)(1-\gamma)\sigma_Y
+ (1-\gamma)\mu_Y
+ \tfrac{1}{2}(1-\gamma)^2 \sigma_Y^2
\right] \tilde{J}(\upsilon)
$$

$$
+ \upsilon(1-\upsilon)
\left[\nu^2 - \nu^1 + (\omega^1 - \omega^2)(1-\gamma)\sigma_Y\right]
\tilde{J}'(\upsilon)
$$

$$
+ \tfrac{1}{2}\upsilon^2(1-\upsilon)^2 (\omega^1 - \omega^2)^2 \,
\tilde{J}''(\upsilon)
$$

满足约束 $z^1 + z^2 \leq 1$。

第一行是来自两个主体幸福函数的流量收益。

第二行将 $\tilde{J}(\upsilon)$ 乘以一个结合了主体贴现率、信念加权的禀赋漂移以及方差修正的项——这些来自于通过伊藤引理吸收 $Y^{1-\gamma}$ 因子。

第三行将 $\tilde{J}'(\upsilon)$ 乘以帕累托份额的漂移，它取决于贴现率的差异以及对禀赋风险的信念加权响应。

第四行将 $\tilde{J}''(\upsilon)$ 乘以帕累托份额扩散的平方。

边界条件为 $\tilde{J}(0) = \tilde{V}^2$ 和 $\tilde{J}(1) = \tilde{V}^1$，其中 $\tilde{V}^n$ 是仅由主体 $n$ 单独构成的齐次经济体中的延续值。

这是 {cite:t}`Blume_Easley2006` 中离散时间规划者问题的连续时间对应（另见 {doc}`likelihood_ratio_process_2`）。

## 生存条件

核心结果通过 $m_\vartheta(\upsilon)$ 的边界行为来刻画生存。

```{prf:proposition}
:label: survival_conditions

定义以下排斥条件 (i) 和 (ii) 及其吸引对应条件 (i') 和 (ii')：

$$
\text{(i)} \lim_{\upsilon \searrow 0} m_\vartheta(\upsilon) > 0, \qquad
\text{(i')} \lim_{\upsilon \searrow 0} m_\vartheta(\upsilon) < 0
$$

$$
\text{(ii)} \lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon) < 0, \qquad
\text{(ii')} \lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon) > 0
$$

那么：

*(a)* 若 (i) 和 (ii) 成立，则两个主体在 $P$ 下都生存。

*(b)* 若 (i) 和 (ii') 成立，则主体 1 在 $P$ 下长期占据主导地位。

*(c)* 若 (i') 和 (ii) 成立，则主体 2 在 $P$ 下长期占据主导地位。

*(d)* 若 (i') 和 (ii') 成立，则每个主体都以严格正概率占据主导地位。
```

该证明使用了扩散过程边界行为的 Feller 分类，如 {cite:t}`Karlin_Taylor1981` 所述。

条件 (i) 表明，当主体 1 接近消亡时，存在一种将其份额推回上升的力量。

条件 (ii) 表明，当主体 1 接近吸收整个经济体时，存在一种将其份额推回下降的力量。

当两种力量都存在时，帕累托份额是常返的，两个主体都生存。

## 财富动态分解

我们现在用均衡财富动态重写 {prf:ref}`survival_conditions` 中的生存条件。

当且仅当主体 1 在其规模微不足道时财富增长快于主体 2，主体 1 才能在接近消亡时生存下来。

当 $\upsilon \searrow 0$ 时，价格完全由主体 2 决定，仿佛经济体是齐次的。

主体 1 是主体 2 经济体中的价格接受者。

令 $m_A^n(\upsilon)$ 表示主体 $n$ 的财富 $A^n$ 的期望对数增长率，即 $d \log A_t^n = m_A^n(\upsilon_t) dt + \sigma_A^n(\upsilon_t) dW_t$。

{cite:t}`Borovicka2020` 的命题 3.3 将对数比值的边界漂移与这些增长率联系起来：

$$
\lim_{\upsilon \to \bar{\upsilon}} \gamma^{-1} m_\vartheta(\upsilon)
= \lim_{\upsilon \to \bar{\upsilon}} [m_A^1(\upsilon) - m_A^2(\upsilon)],
\qquad \bar{\upsilon} \in \{0, 1\} .
$$ (eq:drift_wealth)

令 $R^n$ 表示主体 $n$ 投资组合的累积收益指数，其中 $d \log R_t^n = m_R^n(\upsilon_t) dt + \sigma_R^n(\upsilon_t) dW_t$。

因此 $m_R^n$ 是主体 $n$ 投资组合的期望*对数*收益，而 $m_R^n + \frac{1}{2}(\sigma_R^n)^2$ 是其期望算术收益。

令 $(y^n)^{-1}$ 表示主体 $n$ 的消费-财富比。

财富增长率之差分解为两个渠道：

$$
\lim_{\upsilon \searrow 0} [m_A^1(\upsilon) - m_A^2(\upsilon)]
= \underbrace{\lim_{\upsilon \searrow 0} [m_R^1(\upsilon) - m_R^2(\upsilon)]}_{\text{组合收益}}
+ \underbrace{\lim_{\upsilon \searrow 0} [(y^2(\upsilon))^{-1} - (y^1(\upsilon))^{-1}]}_{\text{消费-财富比}}
$$ (eq:wealth_decomp)

第一项衡量主体 1 的投资组合增长快多少。

第二项衡量主体 1 从财富中消费少多少——较低的消费-财富比意味着更多的储蓄和更快的财富积累。

当此总差异为正时，条件 (i) 成立，主体 1 在接近消亡时生存。

当其为负时，边界 $\upsilon = 0$ 具有吸引性。

如果条件 (ii) 也成立(情形 (c))，主体 1 将几乎必然消亡；如果条件 (ii') 成立(情形 (d))，则以严格正的概率消亡。

```{exercise}
:label: ex_wealth_decomp

推导 {eq}`eq:wealth_decomp`。

令 $A^n$ 表示主体 $n$ 的财富，$C^n$ 表示她的消费。

令 $R^n$ 为她投资组合的累积收益指数，其中 $d\log R^n = m_R^n \, dt + \sigma_R^n \, dW$，因此 $m_R^n$ 是她的期望对数投资组合收益。

预算约束为 $dA^n = A^n \, dR^n / R^n - C^n dt$。

定义消费-财富比 $(y^n)^{-1} = C^n / A^n$。

证明 $d\log A^n = \left(m_R^n - (y^n)^{-1}\right) dt + \sigma_R^n \, dW$，因此期望对数财富增长之差为 $m_A^1 - m_A^2 = (m_R^1 - m_R^2) + [(y^2)^{-1} - (y^1)^{-1}]$。
```

```{solution-start} ex_wealth_decomp
:class: dropdown
```

将伊藤引理应用于 $R^n = \exp(\log R^n)$，瞬时收益为

$$
\frac{dR^n}{R^n} = \left(m_R^n + \tfrac{1}{2}(\sigma_R^n)^2\right) dt + \sigma_R^n \, dW.
$$

将预算约束除以 $A^n$：

$$
\frac{dA^n}{A^n} = \left(m_R^n + \tfrac{1}{2}(\sigma_R^n)^2 - (y^n)^{-1}\right) dt + \sigma_R^n \, dW.
$$

由伊藤引理，$d\log A^n = \frac{dA^n}{A^n} - \frac{1}{2}\left(\frac{dA^n}{A^n}\right)^2$，且 $\left(dA^n/A^n\right)^2 = (\sigma_R^n)^2 dt$。

那么

$$
d\log A^n = \left(m_R^n - (y^n)^{-1}\right) dt + \sigma_R^n \, dW,
$$

因此 $m_A^n = m_R^n - (y^n)^{-1}$。

对主体 1 和 2 取差得到

$$
m_A^1 - m_A^2 = (m_R^1 - m_R^2) + \left[(y^2)^{-1} - (y^1)^{-1}\right],
$$

取极限 $\upsilon \searrow 0$ 即得 {eq}`eq:wealth_decomp`。

```{solution-end}
```

### 组合收益

在边界 $\upsilon \searrow 0$ 处，期望对数投资组合收益之差为

$$
\lim_{\upsilon \searrow 0} [m_R^1 - m_R^2]
= \underbrace{\frac{\omega^1 - \omega^2}{\gamma \sigma_Y}}_{\text{风险份额之差}}
\cdot \underbrace{(\gamma \sigma_Y^2 - \omega^2 \sigma_Y)}_{\text{风险溢价}}
- \underbrace{\frac{\omega^1 - \omega^2}{\gamma}
\left(\sigma_Y + \frac{\omega^1 - \omega^2}{2\gamma}\right)}_{\text{波动率项}}
$$ (eq:portfolio_returns)

乐观的主体($\omega^1 > \omega^2$)相对于主体 2 将风险资产超配 $(\omega^1 - \omega^2)/(\gamma \sigma_Y)$，并从这一额外敞口中赚取股权风险溢价。

被减去的*波动率惩罚*反映了持有更极端投资组合的成本：对数收益率的较高方差拖累了期望对数财富增长。

此项依赖于风险厌恶 $\gamma$，但不依赖于 IES，因为投资组合选择仅由风险厌恶决定。

```{exercise}
:label: ex_portfolio_returns

推导 {eq}`eq:portfolio_returns`。

在边界 $\upsilon \searrow 0$ 处，主体 $n$ 的最优风险资产份额为 $\pi^n = 1 + (\omega^n - \omega^2)/(\gamma \sigma_Y)$(参见 {eq}`eq:portfolio`)。

令 $r$ 表示无风险利率，$\bar{\mu}_R = r + \gamma \sigma_Y^2 - \omega^2 \sigma_Y$ 表示 $P$ 下风险资产的期望算术收益。

(利用 {eq}`eq:riskfree` 和 {eq}`eq:wc_ratio`，$\bar{\mu}_R$ 等于期望资本利得 $\mu_Y + \frac{1}{2}\sigma_Y^2$ 加上股息收益率 $(y(0))^{-1}$。)

连续再平衡的投资组合具有期望对数收益 $m_R^n = r + \pi^n(\bar{\mu}_R - r) - \frac{1}{2}(\pi^n)^2 \sigma_Y^2$。

计算 $m_R^1 - m_R^2$ 并化简。
```

```{solution-start} ex_portfolio_returns
:class: dropdown
```

使用 $m_R^n = r + \pi^n(\bar{\mu}_R - r) - \frac{1}{2}(\pi^n)^2 \sigma_Y^2$，差异为

$$
m_R^1 - m_R^2 = (\pi^1 - \pi^2)(\bar{\mu}_R - r) - \tfrac{1}{2}[(\pi^1)^2 - (\pi^2)^2]\sigma_Y^2.
$$

风险份额之差为 $\pi^1 - \pi^2 = (\omega^1 - \omega^2)/(\gamma \sigma_Y)$。

算术股权溢价为 $\bar{\mu}_R - r = \gamma \sigma_Y^2 - \omega^2 \sigma_Y$，因此：

$$
(\pi^1 - \pi^2)(\bar{\mu}_R - r) = \frac{\omega^1 - \omega^2}{\gamma \sigma_Y} \cdot (\gamma \sigma_Y^2 - \omega^2 \sigma_Y).
$$

对于波动率项，写出 $(\pi^1)^2 - (\pi^2)^2 = (\pi^1 - \pi^2)(\pi^1 + \pi^2)$ 并使用 $\pi^2 = 1$，因此 $\pi^1 + \pi^2 = 2 + (\omega^1 - \omega^2)/(\gamma \sigma_Y)$。

化简后：

$$
\tfrac{1}{2}[(\pi^1)^2 - (\pi^2)^2]\sigma_Y^2 = \frac{\omega^1 - \omega^2}{\gamma}\left(\sigma_Y + \frac{\omega^1 - \omega^2}{2\gamma}\right).
$$

将两部分合并即得 {eq}`eq:portfolio_returns`。

```{solution-end}
```

### 消费-财富比

在边界处消费-财富比之差为

$$
\lim_{\upsilon \searrow 0} [(y^2)^{-1} - (y^1)^{-1}]
= \frac{1-\rho}{\rho} \left[(\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma}\right]
$$ (eq:consumption_rates)

括号中的项是两位主体预期从各自最优投资组合中获得的*主观*确定性等价收益之间的差距。

在她自己的信念 $Q^n$ 下，主体 $n$ 感知到风险资产的夏普比率为 $\lambda^n = \gamma \sigma_Y + \omega^n - \omega^2$。

她的最优投资组合获得确定性等价收益 $r + (\lambda^n)^2/(2\gamma)$，括号中的项等于 $[(\lambda^1)^2 - (\lambda^2)^2]/(2\gamma)$。

该括号项也等于两位主体主观期望算术投资组合收益之差的一半。

因子 $(1-\rho)/\rho$ 将这种感知到的收益优势转化为储蓄响应。

- 当 IES $> 1$($\rho < 1$)时，因子为正：更高的感知收益使主体储蓄更多，因为替代效应主导了收入效应。
- 当 IES $< 1$($\rho > 1$)时，因子为负：收入效应占主导，主体储蓄更少，不利于生存。
- 当 IES $= 1$($\rho = 1$)时，两种效应相互抵消，储蓄渠道完全消失。

这就是递归偏好通过将 $\gamma$ 与 $\rho$ 分离来改变生存结果的渠道。

```{exercise}
:label: ex_consumption_wealth

推导 {eq}`eq:consumption_rates`。

在边界 $\upsilon \searrow 0$ 处，两位主体都将无风险利率 $r$ 和在 $P$ 下波动率为 $\sigma_Y$、期望超额收益为 $\gamma \sigma_Y^2 - \omega^2 \sigma_Y$ 的风险资产视为既定。

1. 证明在 $Q^n$ 下主体 $n$ 感知到的夏普比率为 $\lambda^n = \gamma\sigma_Y + \omega^n - \omega^2$。
   风险份额为 $\pi$ 的投资组合的主观确定性等价收益为 $r + \pi \sigma_Y \lambda^n - \frac{1}{2}\gamma \pi^2 \sigma_Y^2$。
   证明最优份额重现了 {eq}`eq:portfolio`，且最大化的确定性等价收益为 $\mathrm{CE}^n = r + (\lambda^n)^2/(2\gamma)$。

2. 令 $c^n = (y^n)^{-1}$，令 $\mu_V^n = \mathrm{CE}^n - c^n$ 为主体 $n$ 财富的主观确定性等价增长率，它也是其消费的增长率。
   利用面对固定投资机会的 Epstein-Zin 投资者的 Hamilton-Jacobi-Bellman 方程，证明 $c^n = \beta - (1-\rho)\mu_V^n$，从而 $\mu_V^n = (\mathrm{CE}^n - \beta)/\rho$。

3. 证明 $\mu_V^2 = \mu_Y + \omega^2\sigma_Y + \frac{1}{2}(1-\gamma)\sigma_Y^2$，从而 $c^2$ 重现了 {eq}`eq:wc_ratio`。

4. 推导出 {eq}`eq:consumption_rates`。

*提示：* 对于第 2 步，使用归一化的 Duffie-Epstein 聚合器

$$
f(C, V) = \frac{\beta}{1-\rho} (1-\gamma) V \left[\left(\frac{C}{((1-\gamma)V)^{1/(1-\gamma)}}\right)^{1-\rho} - 1\right]
$$

并猜测 $V(A) = (\kappa A)^{1-\gamma}/(1-\gamma)$，其中 $\kappa > 0$ 为常数。
```

```{solution-start} ex_consumption_wealth
:class: dropdown
```

*第 1 步。*
在 $Q^n$ 下，$dW = dW^n + \omega^n dt$。

风险资产的瞬时收益 $(r + \gamma\sigma_Y^2 - \omega^2\sigma_Y)dt + \sigma_Y dW$ 变为 $(r + \gamma\sigma_Y^2 - \omega^2\sigma_Y + \omega^n\sigma_Y)dt + \sigma_Y dW^n$。

因此其主观夏普比率为 $\lambda^n = \gamma\sigma_Y + \omega^n - \omega^2$。

确定性等价收益 $r + \pi\sigma_Y\lambda^n - \frac{1}{2}\gamma\pi^2\sigma_Y^2$ 在 $\pi^n = \lambda^n/(\gamma\sigma_Y) = 1 + (\omega^n - \omega^2)/(\gamma\sigma_Y)$ 处最大化，这正是 {eq}`eq:portfolio`。

最大化值为 $\mathrm{CE}^n = r + (\lambda^n)^2/(2\gamma)$。

*第 2 步。*
在 $Q^n$ 下，财富满足 $dA = A\left[(r + \pi\sigma_Y\lambda^n - c)dt + \pi\sigma_Y dW^n\right]$。

当 $V(A) = (\kappa A)^{1-\gamma}/(1-\gamma)$ 时，有 $((1-\gamma)V)^{1/(1-\gamma)} = \kappa A$，HJB 方程为

$$
0 = \max_{c, \pi} \left\{ f(cA, V(A)) + V'(A) A (r + \pi\sigma_Y\lambda^n - c) + \tfrac{1}{2} V''(A) A^2 \pi^2 \sigma_Y^2 \right\}.
$$

除以 $(\kappa A)^{1-\gamma}$ 得到

$$
0 = \max_{c, \pi} \left\{ \frac{\beta}{1-\rho}\left[\left(\frac{c}{\kappa}\right)^{1-\rho} - 1\right] + r + \pi\sigma_Y\lambda^n - c - \tfrac{1}{2}\gamma\pi^2\sigma_Y^2 \right\}.
$$

关于 $c$ 的一阶条件为 $\beta (c/\kappa)^{-\rho}/\kappa = 1$，这意味着 $(c/\kappa)^{1-\rho} = c/\beta$。

将其与最优 $\pi$ 代入 HJB 方程得到

$$
0 = \frac{c^n - \beta}{1-\rho} + \mathrm{CE}^n - c^n ,
$$

整理得 $c^n = \beta - (1-\rho)(\mathrm{CE}^n - c^n) = \beta - (1-\rho)\mu_V^n$。

对同一方程求解 $c^n$ 得到 $\rho c^n = \beta - (1-\rho)\mathrm{CE}^n$，因此 $\mu_V^n = \mathrm{CE}^n - c^n = (\mathrm{CE}^n - \beta)/\rho$。

量 $\mu_V^n$ 是财富的主观确定性等价增长率。

在 $Q^n$ 下，$d\log A$ 的漂移为 $r + \pi\sigma_Y\lambda^n - c - \frac{1}{2}\pi^2\sigma_Y^2$，波动率为 $\pi\sigma_Y$。

加上风险调整项 $\frac{1}{2}(1-\gamma)\pi^2\sigma_Y^2$ 并在 $\pi^n$ 处求值得到 $\mathrm{CE}^n - c^n$。

*第 3 步。*
对于主体 2，$\lambda^2 = \gamma\sigma_Y$，因此 $\mathrm{CE}^2 = r + \frac{1}{2}\gamma\sigma_Y^2$。

代入 {eq}`eq:riskfree` 得到 $\mathrm{CE}^2 = \beta + \rho\left(\mu_Y + \omega^2\sigma_Y + \frac{1}{2}(1-\gamma)\sigma_Y^2\right)$，因此 $\mu_V^2 = \mu_Y + \omega^2\sigma_Y + \frac{1}{2}(1-\gamma)\sigma_Y^2$。

那么 $c^2 = \beta - (1-\rho)\mu_V^2$ 即为 {eq}`eq:wc_ratio`。

*第 4 步。*
取差，

$$
(y^2)^{-1} - (y^1)^{-1} = c^2 - c^1 = (1-\rho)(\mu_V^1 - \mu_V^2) = \frac{1-\rho}{\rho}\left(\mathrm{CE}^1 - \mathrm{CE}^2\right).
$$

由于 $\lambda^1 - \lambda^2 = \omega^1 - \omega^2$ 且 $\lambda^1 + \lambda^2 = 2\gamma\sigma_Y + \omega^1 - \omega^2$，

$$
\mathrm{CE}^1 - \mathrm{CE}^2 = \frac{(\lambda^1)^2 - (\lambda^2)^2}{2\gamma}
= (\omega^1 - \omega^2)\sigma_Y + \frac{(\omega^1 - \omega^2)^2}{2\gamma},
$$

由此得到 {eq}`eq:consumption_rates`。

因子 $1/\rho$ 反映了消费对财富增长的反馈：$c^n$ 依赖于 $\mu_V^n$，而 $\mu_V^n = \mathrm{CE}^n - c^n$ 又依赖于 $c^n$。

下面是在 $\gamma = 5$、$\rho = 0.67$、$\omega^1 = 0.25$、$\omega^2 = 0$、$\sigma_Y = 0.02$、$\mu_Y = 0.02$、$\beta = 0.05$ 下的数值检验。

```{code-cell} ipython3
γ, ρ, β, μ_y, σ_y = 5.0, 0.67, 0.05, 0.02, 0.02
ω_1, ω_2 = 0.25, 0.0

r = β + ρ * (μ_y + ω_2 * σ_y + 0.5 * (1 - γ) * σ_y**2) - 0.5 * γ * σ_y**2

def consumption_rate(ω_n):
    λ_n = γ * σ_y + ω_n - ω_2
    CE_n = r + λ_n**2 / (2 * γ)
    return (β - (1 - ρ) * CE_n) / ρ

c_1, c_2 = consumption_rate(ω_1), consumption_rate(ω_2)
y0_inv = β - (1 - ρ) * (μ_y + ω_2 * σ_y + 0.5 * (1 - γ) * σ_y**2)
Δω = ω_1 - ω_2
formula = (1 - ρ) / ρ * (Δω * σ_y + Δω**2 / (2 * γ))

print(f"c^2 from the HJB solution:     {c_2:.6f}")
print(f"(y(0))^(-1) from eq:wc_ratio:  {y0_inv:.6f}")
print(f"c^2 - c^1 from the HJB:        {c_2 - c_1:.6f}")
print(f"eq:consumption_rates:          {formula:.6f}")
```

```{solution-end}
```

### 两个比较静态分析

生存取决于 $\gamma$、$\rho$ 以及信噪比 $\omega^1 / \sigma_Y$ 和 $\omega^2 / \sigma_Y$，而非分别取决于 $\omega^1$、$\omega^2$ 和 $\sigma_Y$。

生存条件不依赖于 $\beta$ 或 $\mu_Y$，它们影响消费和价格的水平，但不影响边界处的相对财富动态。

```{code-cell} ipython3
def portfolio_return_diff(ω_1, ω_2, γ, σ_y):
    """
    边界处期望对数投资组合收益之差。
    """
    Δω = ω_1 - ω_2
    risky_share_diff = Δω / (γ * σ_y)
    risk_premium = γ * σ_y**2 - ω_2 * σ_y
    volatility_term = (Δω / γ) * (σ_y + 0.5 * Δω / γ)
    return risky_share_diff * risk_premium - volatility_term


def saving_channel(ω_1, ω_2, γ, ρ, σ_y):
    """
    边界处消费-财富比之差。
    """
    Δω = ω_1 - ω_2
    subjective_return_diff = Δω * σ_y + Δω**2 / (2 * γ)
    return (1 - ρ) / ρ * subjective_return_diff


def boundary_drift(ω_1, ω_2, γ, ρ, σ_y):
    """
    当主体 1 变得微不足道时的边界漂移 m_ϑ(0+)。

    根据 eq:drift_wealth，这等于 eq:wealth_decomp
    中财富增长差异乘以 γ。
    正漂移意味着主体 1 生存(排斥边界)。
    """
    return γ * (
        portfolio_return_diff(ω_1, ω_2, γ, σ_y)
        + saving_channel(ω_1, ω_2, γ, ρ, σ_y)
    )
```

情形 $\upsilon \nearrow 1$ 是对称的。

交换两位主体的标签会将 $\vartheta$ 变为 $-\vartheta$，因此 $\lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon)$ 等于 `-boundary_drift(ω_2, ω_1, γ, ρ, σ_y)`。

### 风险厌恶与 IES 之比较

两个边界漂移满足一个简单的恒等式。

```{prf:proposition}
:label: boundary_drift_gap

$$
\lim_{\upsilon \searrow 0} m_\vartheta(\upsilon) - \lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon)
= (\omega^1 - \omega^2)^2 \left(\frac{1}{\rho} - \frac{1}{\gamma}\right) .
$$
```

为了推导它，令 $\Delta = \omega^1 - \omega^2$。

结合 {eq}`eq:drift_wealth`、{eq}`eq:wealth_decomp`、{eq}`eq:portfolio_returns` 和 {eq}`eq:consumption_rates` 得到

$$
\lim_{\upsilon \searrow 0} m_\vartheta(\upsilon)
= \Delta(\gamma\sigma_Y - \omega^2) - \Delta\sigma_Y - \frac{\Delta^2}{2\gamma}
+ \frac{1-\rho}{\rho}\left(\gamma\Delta\sigma_Y + \frac{\Delta^2}{2}\right) .
$$

在 $\upsilon \nearrow 1$ 处，两位主体的角色互换，这将 $\Delta$ 替换为 $-\Delta$，$\omega^2$ 替换为 $\omega^1$，$m_\vartheta$ 替换为 $-m_\vartheta$：

$$
\lim_{\upsilon \nearrow 1} m_\vartheta(\upsilon)
= -\left[-\Delta(\gamma\sigma_Y - \omega^1) + \Delta\sigma_Y - \frac{\Delta^2}{2\gamma}
+ \frac{1-\rho}{\rho}\left(-\gamma\Delta\sigma_Y + \frac{\Delta^2}{2}\right)\right] .
$$

在差值中，与 $\sigma_Y$ 成线性的项相互抵消。

风险溢价项相加得到 $\Delta(\omega^1 - \omega^2) = \Delta^2$，波动率项相加得到 $-\Delta^2/\gamma$，储蓄项相加得到 $\Delta^2 (1-\rho)/\rho$。

总和为 $\Delta^2(1/\rho - 1/\gamma)$。

该恒等式对 $(\gamma, \rho)$ 平面进行了划分：

- 结果 (a)，即共存，要求 $\lim_{\upsilon \searrow 0} m_\vartheta > 0 > \lim_{\upsilon \nearrow 1} m_\vartheta$，因此要求 $\gamma > \rho$。
- 结果 (d)，即每个主体都以正概率占优，要求相反的不等式，因此要求 $\gamma < \rho$。
- 在对角线 $\gamma = \rho$ 上，两个边界漂移相等，因此当 $|\omega^1| \neq |\omega^2|$ 时，恰好有一个主体占优。
- 结果 (b) 和 (c) 可能发生在对角线的任一侧。

当 $\gamma > \rho$ 时，Epstein-Zin 主体偏好不确定性的早期解决 {cite}`Kreps_Porteus1978,Epstein_Zin1989`。

只有在由具有这种偏好的主体构成的经济体中，两位主体才可能都生存。

{cite:t}`Borovicka2020` 指出，他的图 2 左上面板中的共存区域位于 $\gamma > \rho$ 处；{prf:ref}`boundary_drift_gap` 表明，这对所有信念扭曲和所有 $\sigma_Y$ 都成立。

单凭 IES 或单凭风险厌恶都无法解决这个问题。

IES 大于 1 并不足以保证共存，因为 $\gamma < \rho < 1$ 排除了这种可能性。

IES 大于 1 也并非必要条件，下一节将说明这一点。

```{exercise}
:label: ex_drift_gap

1. 从 {eq}`eq:portfolio_returns` 和 {eq}`eq:consumption_rates` 出发，完成 {prf:ref}`boundary_drift_gap` 背后的代数运算。
2. 证明当 $\gamma = \rho$ 时，两个边界漂移都等于 $\frac{1}{2}\left[(\omega^2)^2 - (\omega^1)^2\right]$。
3. 使用 `boundary_drift` 在 10,000 个随机抽取的参数向量 $(\omega^1, \omega^2, \gamma, \rho, \sigma_Y)$ 上验证该恒等式。
```

```{solution-start} ex_drift_gap
:class: dropdown
```

*第 1 部分。*
将 {eq}`eq:portfolio_returns` 乘以 $\gamma$ 得到

$$
\gamma \lim_{\upsilon \searrow 0}[m_R^1 - m_R^2]
= \frac{\Delta}{\sigma_Y}(\gamma\sigma_Y^2 - \omega^2\sigma_Y) - \Delta\left(\sigma_Y + \frac{\Delta}{2\gamma}\right)
= \Delta(\gamma\sigma_Y - \omega^2) - \Delta\sigma_Y - \frac{\Delta^2}{2\gamma} .
$$

将 {eq}`eq:consumption_rates` 乘以 $\gamma$ 得到 $\frac{1-\rho}{\rho}\left(\gamma\Delta\sigma_Y + \frac{\Delta^2}{2}\right)$。

由 {eq}`eq:drift_wealth` 和 {eq}`eq:wealth_decomp`，两者之和即为 $\lim_{\upsilon \searrow 0} m_\vartheta$。

在 $\upsilon \nearrow 1$ 处，主体 2 是微不足道的主体，主体 1 决定价格。

对交换标签后的情形应用相同的公式得到 $\gamma \lim_{\upsilon \nearrow 1}[m_A^2 - m_A^1]$，它等于 $-\lim_{\upsilon \nearrow 1} m_\vartheta$。

交换标签将 $\Delta$ 替换为 $-\Delta$，$\omega^2$ 替换为 $\omega^1$，因此

$$
-\lim_{\upsilon \nearrow 1} m_\vartheta
= -\Delta(\gamma\sigma_Y - \omega^1) + \Delta\sigma_Y - \frac{\Delta^2}{2\gamma}
+ \frac{1-\rho}{\rho}\left(-\gamma\Delta\sigma_Y + \frac{\Delta^2}{2}\right) .
$$

将两个表达式逐项相加：

$$
\Delta(\omega^1 - \omega^2) - \frac{\Delta^2}{\gamma} + \frac{1-\rho}{\rho}\Delta^2
= \Delta^2\left(1 - \frac{1}{\gamma} + \frac{1}{\rho} - 1\right)
= \Delta^2\left(\frac{1}{\rho} - \frac{1}{\gamma}\right) .
$$

*第 2 部分。*
令 $\rho = \gamma$，储蓄项变为 $(1-\gamma)\Delta\sigma_Y + (1-\gamma)\Delta^2/(2\gamma)$。

将其加到投资组合项上，含 $\sigma_Y$ 的项和含 $\Delta^2/(2\gamma)$ 的项相互抵消，剩下

$$
\lim_{\upsilon \searrow 0} m_\vartheta = -\Delta\omega^2 - \frac{\Delta^2}{2}
= -\frac{(\omega^1 - \omega^2)(\omega^1 + \omega^2)}{2}
= \frac{1}{2}\left[(\omega^2)^2 - (\omega^1)^2\right] .
$$

由第 1 部分可知，$\upsilon \nearrow 1$ 处的漂移相同。

当 $\gamma = \rho$ 时，除非 $\rho = 1$，否则储蓄项不为零；这三个渠道净得一个不依赖于 $\gamma$ 或 $\sigma_Y$ 的常数。

*第 3 部分。*

```{code-cell} ipython3
rng = np.random.default_rng(0)
n_draws = 10_000
max_err = 0.0
for _ in range(n_draws):
    ω_1, ω_2 = rng.normal(0.0, 0.3, size=2)
    γ = rng.uniform(0.05, 30.0)
    ρ = rng.uniform(0.05, 5.0)
    σ_y = rng.uniform(0.005, 0.2)
    m_0 = boundary_drift(ω_1, ω_2, γ, ρ, σ_y)
    m_1 = -boundary_drift(ω_2, ω_1, γ, ρ, σ_y)
    gap = (ω_1 - ω_2)**2 * (1 / ρ - 1 / γ)
    max_err = max(max_err, abs(m_0 - m_1 - gap))

print(f"largest error over {n_draws} draws: {max_err:.2e}")

for γ in [0.5, 2.0, 7.0]:
    m_sep = boundary_drift(0.25, -0.1, γ, γ, 0.02)
    print(f"γ = ρ = {γ}: m_ϑ(0+) = {m_sep:.6f},  "
          f"[(ω^2)^2 - (ω^1)^2] / 2 = {0.5 * (0.1**2 - 0.25**2):.6f}")
```

```{solution-end}
```

## 生存区域

{cite:t}`Borovicka2020` 的一个核心贡献是刻画了 $(\gamma, \rho)$ 平面中的生存区域。

在可分偏好下，$\gamma = \rho$，信念更准确的主体总是占据主导地位。

在递归偏好下，{prf:ref}`survival_conditions` 中的所有四种结果都可能出现。

文中的图 2 研究了主体 2 具有正确信念的情形，即 $\omega^2 = 0$。

下一个单元遵循该图。

对于极限面板 $|\omega^1|/\sigma_Y \to \infty$，我们使用 $m_\vartheta(0+)/(\omega^1)^2$ 和 $m_\vartheta(1-)/(\omega^1)^2$ 的极限，这些极限由 {ref}`ex_limit_boundaries` 推导得出。

```{code-cell} ipython3
def limit_boundary_drifts(γ, ρ):
    """
    当 |ω^1| / σ_y -> ∞ 且 ω^2 = 0 时，
    m_ϑ(0+) / (ω^1)^2 和 m_ϑ(1-) / (ω^1)^2 的极限。
    """
    m_0 = (γ - (1 + γ) * ρ) / (2 * γ * ρ)
    m_1 = ((1 - γ) * ρ - γ) / (2 * γ * ρ)
    return m_0, m_1
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: 对应于 Borovička (2020) 图 2 的生存区域
    name: fig-survival-regions
---
σ_y = 0.02
γ_vals = np.linspace(0.01, 6.0, 500)
ρ_vals = np.linspace(0.01, 2.0, 400)
G, R = np.meshgrid(γ_vals, ρ_vals)

panel_specs = [
    ("finite", 0.10, r"$\omega^1 = 0.10$"),
    ("finite", 0.20, r"$\omega^1 = 0.20$"),
    ("limit", None, r"$|\omega^1| / \sigma_Y \to \infty$"),
    ("finite", -0.25, r"$\omega^1 = -0.25$"),
]
region_masks = {}

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

for idx, (case, value, label) in enumerate(panel_specs):
    ax = axes.flat[idx]

    if case == "limit":
        m0, m1 = limit_boundary_drifts(G, R)
    else:
        m0 = boundary_drift(value, 0.0, G, R, σ_y)
        m1 = -boundary_drift(0.0, value, G, R, σ_y)

    # 对所有四个区域进行分类
    both = (m0 > 0) & (m1 < 0)
    ag1_dom = (m0 > 0) & (m1 > 0)
    ag2_dom = (m0 < 0) & (m1 < 0)
    either = (m0 < 0) & (m1 > 0)
    region_masks[idx] = (both, ag1_dom, ag2_dom, either)

    # 对共存区域着色
    ax.contourf(G, R, both.astype(float), levels=[0.5, 1.5],
                colors=["C2"], alpha=0.18)
    if idx == 0:
        ax.fill_between([], [], color="C2", alpha=0.18,
                        label="两者都生存")

    # 绘制边界曲线
    ax.contour(G, R, m0, levels=[0], colors=["C0"],
               linestyles="--", linewidths=2)
    ax.contour(G, R, m1, levels=[0], colors=["C3"],
               linestyles="-", linewidths=2)
    if idx == 0:
        ax.plot([], [], "--", color="C0", lw=2, label="主体 1 边界")
        ax.plot([], [], "-", color="C3", lw=2, label="主体 2 边界")

    ax.plot(
        γ_vals, γ_vals, ":", color="black", lw=2,
        label=r"$\gamma = \rho$" if idx == 0 else None
    )

    tkw = dict(ha="center", va="center", style="italic", color="0.15")
    if case == "finite" and value == 0.10:
        ax.text(0.31, 1.05, "任一主体占主导", rotation=90,
                fontsize=10, **tkw)
        ax.text(2.6, 1.6, "主体 2\n占主导", fontsize=11, **tkw)
        ax.text(3.5, 0.75, "两者\n都生存", fontsize=11, **tkw)
        if ag1_dom.any():
            ax.text(5.0, 0.25, "主体 1\n占主导", fontsize=11, **tkw)
    elif case == "finite" and value == 0.20:
        ax.text(0.31, 1.05, "任一主体占主导", rotation=90,
                fontsize=10, **tkw)
        ax.text(2.5, 1.55, "主体 2\n占主导", fontsize=11, **tkw)
        ax.text(3.8, 0.55, "两者\n都生存", fontsize=11, **tkw)
        if ag1_dom.any():
            ax.annotate("主体 1 占主导", xy=(5.85, 0.06),
                        xytext=(5.0, 0.3), fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="0.15"),
                        **tkw)
    elif case == "limit":
        ax.text(0.31, 1.05, "任一主体占主导", rotation=90,
                fontsize=10, **tkw)
        ax.text(3.0, 1.40, "主体 2\n占主导", fontsize=11, **tkw)
        ax.text(3.5, 0.30, "两者\n都生存", fontsize=11, **tkw)
    elif case == "finite" and value == -0.25:
        ax.text(0.31, 1.05, "任一主体占主导", rotation=90,
                fontsize=10, **tkw)
        ax.text(3.5, 1.20, "主体 2\n占主导", fontsize=11, **tkw)
        ax.text(2.5, 0.18, "两者\n都生存", fontsize=11, **tkw)

    ax.set_title(label, fontsize=12)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\rho$")

axes[0, 0].legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()
```

每个面板针对主体 1 的信念扭曲 $\omega^1$ 的不同取值（主体 2 具有正确信念，$\omega^2 = 0$）绘制 $(\gamma, \rho)$ 平面中的两条曲线。

- 虚线（蓝色）是 $\upsilon = 0$ 处边界漂移等于零的地方——{prf:ref}`survival_conditions` 中的条件 (i)。
- 实线（红色）是 $\upsilon = 1$ 处边界漂移等于零的地方——条件 (ii)。
- 阴影区域是两个主体都生存的地方。
- 虚线对角线 $\gamma = \rho$ 是可分 CRRA 情形，沿此线信念更准确的主体总是占据主导地位。

下一个单元报告每种结果在绘图窗口中所占的比例、共存区域中 $\rho > 1$（IES $< 1$）所占的比例，以及共存区域中最大的 $\rho$。

```{code-cell} ipython3
names = ["ω¹ = 0.10", "ω¹ = 0.20", "|ω¹|/σ_y → ∞", "ω¹ = -0.25"]
print(f"{'panel':<15}{'both':>7}{'agent 1':>9}{'agent 2':>9}{'either':>8}"
      f"{'share ρ > 1':>13}{'max ρ':>8}")
n_wrong_side = 0
for idx, name in enumerate(names):
    both, ag1_dom, ag2_dom, either = region_masks[idx]
    print(f"{name:<15}{both.mean():7.3f}{ag1_dom.mean():9.3f}"
          f"{ag2_dom.mean():9.3f}{either.mean():8.3f}"
          f"{(both & (R > 1)).sum() / both.sum():13.3f}{R[both].max():8.3f}")
    n_wrong_side += (both & (R >= G)).sum() + (either & (R <= G)).sum()

print(f"\ngrid points with coexistence and ρ ≥ γ, "
      f"or with outcome (d) and ρ ≤ γ: {n_wrong_side}")
```

每个阴影区域都位于对角线下方，而每个任一主体都可能占主导的区域都位于对角线上方，这正是 {prf:ref}`boundary_drift_gap` 所要求的。

适度乐观（$\omega^1 = 0.10$）产生了一个宽阔的共存区域，横跨大部分 $\gamma$ 范围。

共存并不要求 IES $> 1$：该面板中有 42% 的阴影区域满足 $\rho > 1$，其中包括点 $\gamma = 5$，$\rho = 1.5$。

较强的乐观（$\omega^1 = 0.20$）使两条曲线都顺时针旋转。

主体 1 的曲线下降，因此主体 2 在其上方占主导的区域变大。

主体 2 曲线的下支在 $\gamma = \omega^1/(2\sigma_Y)$ 处与轴 $\rho = 0$ 相交，当 $\omega^1 = 0.10$ 时该值为 2.5，当 $\omega^1 = 0.20$ 时为 5。

在该 $\gamma$ 值处，两个主体在边界 $\upsilon = 1$ 处的主观确定性等价回报相等，储蓄渠道在此处改变符号。

因此在 $\omega^1 = 0.20$ 的面板中，主体 1 仅在右下角的一小块区域内占主导，而共存区域从绘图窗口的 44% 增长到 49%。

在极限 $|\omega^1|/\sigma_Y \to \infty$（左下）中，边界曲线为 $\rho = \gamma/(1+\gamma)$，以及当 $\gamma < 1$ 时的 $\rho = \gamma/(1-\gamma)$（参见 {ref}`ex_limit_boundaries`）。

当且仅当 $\rho < \gamma/(1+\gamma)$ 时，即在主体 1 曲线下方的区域，两个主体都生存。

该曲线随 $\gamma$ 增大而趋向 $\rho = 1$，因此在此极限下共存要求 IES $> 1$。

主体 2 曲线将结果 (c) 与结果 (d) 分开，主体 1 永远不会占主导。

悲观的扭曲（$\omega^1 = -0.25$，右下）也可以生存，但仅在参数空间中窄得多的部分：占绘图窗口的 17%，且全部满足 $\rho < 0.51$。

共存区域在 $\gamma = |\omega^1|/(2\sigma_Y) = 6.25$ 处结束，刚好超出绘图范围。

对于更大的 $\gamma$，她的空头头寸过小，$\pi^1 > -1$，不足以使她的主观确定性等价回报超过主体 2。

当 IES $> 1$ 时，储蓄渠道这时对她不利；当 IES $< 1$ 时，该渠道又太弱，无法抵消她的投资组合损失。

```{exercise}
:label: ex_limit_boundaries

设 $\omega^2 = 0$。

1. 证明在固定 $\gamma$ 和 $\rho$ 的情况下，$m_\vartheta(0+)/(\omega^1)^2$ 和 $m_\vartheta(1-)/(\omega^1)^2$ 仅通过 $\sigma_Y/\omega^1$ 依赖于 $\omega^1$ 和 $\sigma_Y$，并且当 $\sigma_Y/\omega^1 \to 0$ 时它们收敛于 `limit_boundary_drifts` 中的表达式。
2. 由此推导出极限下的边界曲线为 $\rho = \gamma/(1+\gamma)$，以及当 $\gamma < 1$ 时的 $\rho = \gamma/(1-\gamma)$。
   对极限下的四种结果进行分类，并证明主体 1 永远不会占主导。
3. 通过数值方法验证，当 $\omega^1/\sigma_Y$ 增大时，`boundary_drift` 在 $\upsilon = 0$ 处的零点曲线收敛于 $\rho = \gamma/(1+\gamma)$。

*提示：*$m_\vartheta(0+)$ 关于 $1/\rho$ 是仿射的，因此对于每个 $\gamma$，其零点都可以用闭式表达式计算得出。
```

```{solution-start} ex_limit_boundaries
:class: dropdown
```

*第 1 部分。*
设 $\Delta = \omega^1$，$s = \sigma_Y/\omega^1$。

根据 {prf:ref}`boundary_drift_gap` 推导过程中 $\lim_{\upsilon \searrow 0} m_\vartheta$ 的表达式，在 $\omega^2 = 0$ 的情况下，

$$
\frac{m_\vartheta(0+)}{\Delta^2}
= (\gamma - 1)s - \frac{1}{2\gamma} + \frac{1-\rho}{\rho}\left(\gamma s + \frac{1}{2}\right)
\;\longrightarrow\; \frac{\gamma - (1+\gamma)\rho}{2\gamma\rho} .
$$

类似地，

$$
\frac{m_\vartheta(1-)}{\Delta^2}
= (\gamma - 1)s - 1 + \frac{1}{2\gamma} + \frac{1-\rho}{\rho}\left(\gamma s - \frac{1}{2}\right)
\;\longrightarrow\; \frac{(1-\gamma)\rho - \gamma}{2\gamma\rho} .
$$

两个极限之差为 $(\gamma - \rho)/(\gamma\rho) = 1/\rho - 1/\gamma$，这正是 {prf:ref}`boundary_drift_gap` 所要求的。

*第 2 部分。*
第一个极限为正当且仅当 $\rho < \gamma/(1+\gamma)$。

第二个极限为负当且仅当 $(1-\gamma)\rho < \gamma$。

当 $\gamma \geq 1$ 时该不等式对所有 $\rho > 0$ 都成立；当 $\gamma < 1$ 时，该不等式当且仅当 $\rho < \gamma/(1-\gamma)$ 时成立。

由于当 $\gamma \in (0, 1)$ 时 $\gamma/(1+\gamma) < \gamma/(1-\gamma)$，条件 (i) 蕴含条件 (ii)。

因此：

- (a) 当且仅当 $\rho < \gamma/(1+\gamma)$ 时两者都生存；
- (b) 主体 1 永远不会占主导，因为条件 (i) 和条件 (ii') 不能同时成立；
- (c) 若 $\rho > \gamma/(1+\gamma)$ 且 $\gamma \geq 1$ 或 $\rho < \gamma/(1-\gamma)$，则主体 2 占主导；
- (d) 若 $\gamma < 1$ 且 $\rho > \gamma/(1-\gamma)$，则任一主体都可能占主导。

*第 3 部分。*
将 $m_\vartheta(0+)$ 写成 $\gamma\left[P + (1/\rho - 1) S\right]$，其中 $P$ 是投资组合回报之差，$S$ 是 {eq}`eq:consumption_rates` 中的括号项。

该漂移在 $1/\rho = 1 - P/S$ 处为零，因此 $\rho = S/(S - P)$。

```{code-cell} ipython3
def lower_zero_curve(ω_1, ω_2, σ_y, γ):
    """
    利用 m_ϑ(0+) 关于 1/ρ 为仿射的性质，计算
    m_ϑ(0+) = 0 处的 ρ。
    """
    Δω = ω_1 - ω_2
    P = portfolio_return_diff(ω_1, ω_2, γ, σ_y)
    S = Δω * σ_y + Δω**2 / (2 * γ)
    return S / (S - P)


γ_check = np.array([0.5, 1.0, 2.0, 5.0, 20.0])
σ_y = 0.02
print("γ:              ", γ_check)
for ratio in [5, 50, 500, 5000]:
    ρ_zero = lower_zero_curve(ratio * σ_y, 0.0, σ_y, γ_check)
    print(f"ω¹/σ_y = {ratio:>5}: ", np.round(ρ_zero, 4))
print("γ / (1 + γ):    ", np.round(γ_check / (1 + γ_check), 4))
```

```{solution-end}
```

## 三个生存渠道

上述分解可以直接可视化。

```{code-cell} ipython3
def decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y):
    """
    分解命题 3.4 中的财富增长差异。

    总计等于 lim (m_A^1 - m_A^2) = m_ϑ(0+) / γ，当 υ -> 0 时。
    """
    Δω = ω_1 - ω_2
    risk_premium_term = Δω * (γ_grid * σ_y - ω_2) / γ_grid
    volatility_term = -(Δω / γ_grid) * (σ_y + 0.5 * Δω / γ_grid)
    saving_term = (1 - ρ) / ρ * (Δω * σ_y + Δω**2 / (2 * γ_grid))
    total = risk_premium_term + volatility_term + saving_term
    return risk_premium_term, volatility_term, saving_term, total


ω_1 = 0.25
ω_2 = 0.0
ρ = 0.67
σ_y = 0.02
γ_grid = np.linspace(0.5, 15.0, 300)

risk_term, vol_term, save_term, total = decompose_survival(
    ω_1, ω_2, γ_grid, ρ, σ_y
)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(γ_grid, risk_term, color="C0", lw=2, label="风险溢价项")
ax.plot(γ_grid, vol_term, "--", color="C3", lw=2, label="波动率项")
ax.plot(γ_grid, save_term, "-.", color="C2", lw=2, label="储蓄项")
ax.plot(γ_grid, total, color="black", lw=2,
        label=r"总计 $= m_\vartheta(0+)/\gamma$")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel(r"风险厌恶 $\gamma$")
ax.set_ylabel("对财富增长差异的贡献")
ax.legend()
plt.tight_layout()
plt.show()

γ_star = brentq(
    lambda γ: decompose_survival(ω_1, ω_2, γ, ρ, σ_y)[3], 0.5, 15.0
)
print(f"the total crosses zero at γ = {γ_star:.3f}")
```

此图将财富增长差异 $\lim_{\upsilon \searrow 0}[m_A^1 - m_A^2] = \lim_{\upsilon \searrow 0} m_\vartheta / \gamma$ 分解为三项，针对一个乐观主体(ω^1 = {0.25}，$\omega^2 = 0$)，其 IES $= 1/\rho \approx 1.49$，$\sigma_Y = 0.02$。

- 风险溢价项(蓝色)始终为正，因为乐观主体超配了风险资产并赚取了股权溢价。
- 波动率项(红色虚线)为负且在低 $\gamma$ 时较大，反映了持有波动投资组合的成本。
- 储蓄项(绿色点划线)在 IES $> 1$ 时为正，因为乐观主体感知到财富的高收益并更激进地储蓄。
- 总计(黑色)在 $\gamma \approx 1.47$ 处穿过零点；低于此值时波动率惩罚占主导，条件(i)不成立，边界 $\upsilon = 0$ 具有吸引力。

## 改变 IES

储蓄项的符号由 IES 确定。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 不同 IES 值的边界分解
    name: fig-survival-ies-panels
---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

ω_1 = 0.25
ω_2 = 0.0
σ_y = 0.02
γ_grid = np.linspace(0.5, 25.0, 300)

ies_values = [0.5, 1.0, 1.5]

for idx, ies in enumerate(ies_values):
    ρ = 1.0 / ies
    risk_term, vol_term, save_term, total = decompose_survival(
        ω_1, ω_2, γ_grid, ρ, σ_y
    )

    ax = axes[idx]
    ax.plot(γ_grid, risk_term, color="C0", lw=2, label="风险溢价")
    ax.plot(γ_grid, vol_term, "--", color="C3", lw=2, label="波动率")
    ax.plot(γ_grid, save_term, "-.", color="C2", lw=2, label="储蓄")
    ax.plot(γ_grid, total, color="black", lw=2, label="总计")
    ax.axhline(0, color="gray", lw=1)
    ax.set_title(f"IES = {ies:.1f}", fontsize=12)
    ax.set_xlabel(r"风险厌恶 $\gamma$")
    ax.set_ylabel("贡献")

axes[0].legend(fontsize=9)
plt.tight_layout()
plt.show()

for ies in ies_values:
    γ_star = brentq(
        lambda γ: decompose_survival(ω_1, ω_2, γ, 1 / ies, σ_y)[3], 0.5, 25.0
    )
    print(f"IES = {ies:.1f}: the total crosses zero at γ = {γ_star:.2f}")
```

每个面板显示与前一图相同的三项分解，但现在针对三种不同的 IES 值($\omega^1 = 0.25$，$\omega^2 = 0$，$\sigma_Y = 0.02$)。

- 左面板(IES $= 0.5$)：储蓄项为负，因此乐观主体实际上储蓄更少，不利于生存。风险溢价项仍使总和在 $\gamma > 9.56$ 时为正，因此在 IES 低于 1 的情况下条件(i)依然成立。
- 中面板(IES $= 1.0$)：储蓄项完全消失，因此只剩下投资组合收益和波动率渠道。

    - 这消除了储蓄渠道，但本身并不能重现完整的可分 CRRA 基准，后者要求 $\gamma = \rho$(即 IES $= 1/\gamma$)，而不仅仅是 $\rho = 1$。
- 右面板(IES $= 1.5$)：储蓄项为正并将总漂移向上移动，扩大了乐观主体生存的 $\gamma$ 值范围。

随着 IES 从 0.5 升至 1 再升至 1.5，零点穿越位置从 $\gamma = 9.56$ 移动到 $3.05$，再移动到 $1.45$。

## 渐近结果

Borovička 推导了几个有用的渐近结果。

1. 当 $\gamma \searrow 0$ 时，每个主体都以严格正概率占据主导地位。
1. 当 $\gamma \nearrow \infty$ 时，相对更乐观的主体占据主导地位。
1. 当 $\rho \searrow 0$ 时，相对更乐观的主体总是生存。
   - 当风险厌恶足够低时，相对更悲观的主体也可以生存。
1. 当 $\rho \nearrow \infty$ 时，非退化的长期均衡不可能存在。

下一个图通过绘制 $\gamma$ 变小时的两个边界漂移来说明第一个结果。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 小风险厌恶下的边界漂移
    name: fig-boundary-drifts-small-gamma
---
ω_1 = 0.25
ω_2 = 0.0
ρ = 0.67
σ_y = 0.02
γ_grid = np.linspace(0.05, 5.0, 300)

drift_at_0 = np.array([boundary_drift(ω_1, ω_2, γ, ρ, σ_y) for γ in γ_grid])
drift_at_1 = np.array([-boundary_drift(ω_2, ω_1, γ, ρ, σ_y) for γ in γ_grid])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(γ_grid, drift_at_0, color="C0", lw=2, label=r"$\upsilon \to 0$")
ax.plot(γ_grid, drift_at_1, "--", color="C3", lw=2, label=r"$\upsilon \to 1$")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel(r"风险厌恶 $\gamma$")
ax.set_ylabel("边界漂移")
ax.legend()
plt.tight_layout()
plt.show()
```

此图将两个边界漂移绘制为 $\gamma$ 的函数（$\omega^1 = 0.25$，$\omega^2 = 0$，IES $\approx 1.49$）。

- 实线蓝色曲线是 $\upsilon \to 0$ 处的漂移 $m_\vartheta$（主体 1 接近消亡）；共存要求这为正（条件 (i)）。
- 虚线红色曲线是 $\upsilon \to 1$ 处的漂移 $m_\vartheta$（主体 2 接近消亡）；共存要求这为负（条件 (ii)）。

此图说明了渐近结果 1。

对于小 $\gamma$，蓝色曲线为负，红色曲线为正。

两个边界都是吸引的：在 $\upsilon = 0$ 附近负漂移将 $\upsilon$ 拉向 0，在 $\upsilon = 1$ 附近正漂移将 $\upsilon$ 推向 1。

这是 {prf:ref}`survival_conditions` 中的结果 (d)：两个边界都不是排斥的，因此无论哪个主体碰巧早期领先都将占据主导地位，每个主体都有严格正概率占据主导地位，这取决于实现的布朗路径。

红色曲线首先在 $\gamma \approx 0.39$ 处穿过零点，而此时蓝色曲线仍为负。

对于大约在 0.39 到 1.47 之间的 $\gamma$，边界 $\upsilon = 1$ 排斥而边界 $\upsilon = 0$ 吸引，因此主体 2 占据主导地位——结果 (c)。

正如 {prf:ref}`boundary_drift_gap` 所要求的，两条曲线在 $\gamma = \rho = 0.67$ 处相交；它们在该点的共同值为 $-\frac{1}{2}(\omega^1)^2 = -0.03125$。

蓝色曲线在 $\gamma \approx 1.47$ 处穿过零点。

超过该点后，两个边界都排斥，两个主体都能生存——结果 (a)。

在所绘范围之外，红色曲线在 $\gamma \approx 10.7$ 处再次穿过零点，对于更大的 $\gamma$，主体 1 占据主导地位——结果 (b)，这与渐近结果 2 一致。

```{exercise}
:label: ex_thresholds

采用 {numref}`fig-boundary-drifts-small-gamma` 中的参数：$\omega^1 = 0.25$，$\omega^2 = 0$，$\rho = 0.67$，$\sigma_Y = 0.02$。

1. 证明 $2\gamma \lim_{\upsilon \searrow 0} m_\vartheta$ 和 $2\gamma \lim_{\upsilon \nearrow 1} m_\vartheta$ 是关于 $\gamma$ 的二次多项式，并利用它们找出使边界漂移变号的所有 $\gamma > 0$。
2. 用 `boundary_drift` 确认这些根。
3. 在相邻根之间的每个区间上标注 {prf:ref}`survival_conditions` 的结果。
```

```{solution-start} ex_thresholds
:class: dropdown
```

设 $\Delta = \omega^1$，$\omega^2 = 0$，$k = (1-\rho)/\rho$。

将 {prf:ref}`boundary_drift_gap` 推导中的两个边界漂移乘以 $2\gamma$ 得到

$$
2\gamma \lim_{\upsilon \searrow 0} m_\vartheta
= \frac{2\Delta\sigma_Y}{\rho}\gamma^2 + \left(k\Delta^2 - 2\Delta\sigma_Y\right)\gamma - \Delta^2,
$$

$$
2\gamma \lim_{\upsilon \nearrow 1} m_\vartheta
= \frac{2\Delta\sigma_Y}{\rho}\gamma^2 - \left(2\Delta\sigma_Y + \frac{1+\rho}{\rho}\Delta^2\right)\gamma + \Delta^2 .
$$

当 $\Delta > 0$ 时，第一个二次多项式有一个正根，因为其常数项为负。

第二个二次多项式在判别式为正时有两个正根，因为其常数项和首项系数都为正。

```{code-cell} ipython3
ω_1, ω_2, ρ, σ_y = 0.25, 0.0, 0.67, 0.02
Δ, k = ω_1 - ω_2, (1 - ρ) / ρ

roots_0 = np.roots([2 * Δ * σ_y / ρ, k * Δ**2 - 2 * Δ * σ_y, -Δ**2])
roots_1 = np.roots([2 * Δ * σ_y / ρ,
                    -(2 * Δ * σ_y + (1 + ρ) / ρ * Δ**2), Δ**2])
roots_0 = np.sort(roots_0[roots_0 > 0])
roots_1 = np.sort(roots_1[roots_1 > 0])

m_0 = lambda γ: boundary_drift(ω_1, ω_2, γ, ρ, σ_y)
m_1 = lambda γ: -boundary_drift(ω_2, ω_1, γ, ρ, σ_y)

print("roots of m_ϑ(0+):", roots_0, " check:", [m_0(g) for g in roots_0])
print("roots of m_ϑ(1-):", roots_1, " check:", [m_1(g) for g in roots_1])

outcome = {(True, True): "(a) both survive",
           (True, False): "(b) agent 1 dominates",
           (False, True): "(c) agent 2 dominates",
           (False, False): "(d) either agent dominates"}

cuts = np.concatenate(([0.0], np.sort(np.concatenate((roots_0, roots_1))),
                       [np.inf]))
for lo, hi in zip(cuts[:-1], cuts[1:]):
    γ_mid = 0.5 * (lo + hi) if np.isfinite(hi) else 2 * lo
    print(f"γ in ({lo:6.3f}, {hi:6.3f}):  "
          f"{outcome[(m_0(γ_mid) > 0, m_1(γ_mid) < 0)]}")
```

随着 $\gamma$ 的增加，结果从 (d) 到 (c) 再到 (a) 再到 (b)，这与渐近结果 1 和 2 一致。

```{solution-end}
```

## 可分情形

当 $\gamma = \rho$ 时，模型退化为可分 CRRA 基准。

{cite:t}`Blume_Easley2006` 在离散时间中研究了这一基准；{doc}`likelihood_ratio_process_2` 介绍了他们的分析。

此时两个主体的贴现率 $\nu^1$ 和 $\nu^2$ 相等，对数几率过程 {eq}`eq:log_odds` 变为

$$
d\vartheta_t = \frac{1}{2}\left[(\omega^2)^2 - (\omega^1)^2\right] dt + (\omega^1 - \omega^2) dW_t .
$$

当 $\gamma = \rho$ 时，其漂移是 {prf:ref}`boundary_drift_gap` 中两个边界漂移的公共值(参见 {ref}`ex_drift_gap`)。

漂移是常数，且仅取决于两个信念扭曲的相对熵。

在 $P$ 下，$P$ 相对于 $Q^n$ 在 $\mathcal{F}_t$ 上的相对熵为 $E^P[-\log M_t^n] = \frac{1}{2}(\omega^n)^2 t$，即 {ref}`rel_entropy` 中描述的 Kullback–Leibler 散度。

$\vartheta$ 的漂移是两个主体相对熵增长率之差。

$|\omega^n|$ 较小的主体在 $P$ 下占据主导地位。

如果两个主体的信念扭曲幅度相等，则几乎必然没有一个主体会消亡，但不存在非退化的平稳财富分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 可分基准中的帕累托份额路径
    name: fig-crra-pareto-paths
---
def simulate_crra_pareto(ω_1, ω_2, T, dt, n_paths, seed=42):
    """
    模拟可分基准中的帕累托份额动态。
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    drift = 0.5 * (ω_2**2 - ω_1**2)
    volatility = ω_1 - ω_2

    θ = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    for t in range(n_steps):
        θ[:, t + 1] = θ[:, t] + drift * dt + volatility * dW[:, t]

    υ_paths = 1.0 / (1.0 + np.exp(-θ))
    return t_grid, υ_paths


ω_1 = 0.10
ω_2 = 0.0
T, dt = 2000, 0.05
t_grid, υ_paths = simulate_crra_pareto(ω_1, ω_2, T=T, dt=dt, n_paths=50)

fig, ax = plt.subplots(figsize=(11, 5))

for i in range(20):
    ax.plot(t_grid, υ_paths[i], color="C0", alpha=0.25, lw=1)

ax.axhline(0.5, color="gray", linestyle=":", lw=1)
ax.set_xlabel("时间")
ax.set_ylabel(r"帕累托份额 $\upsilon_t$")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

drift = 0.5 * (ω_2**2 - ω_1**2)
vol = abs(ω_1 - ω_2)
print(f"{'t':>6}{'share of paths with υ_t < 1/2':>32}{'P(υ_t < 1/2)':>16}")
for t in [200, 500, 1000, 2000]:
    k = int(round(t / dt))
    share = np.mean(υ_paths[:, k] < 0.5)
    prob = 0.5 * (1 + erf(-drift * sqrt(t) / vol / sqrt(2)))
    print(f"{t:>6}{share:>32.2f}{prob:>16.3f}")
```

此图模拟了在可分 CRRA 偏好($\gamma = \rho$)下帕累托份额 $\upsilon_t$ 的 50 条样本路径，其中 $\omega^1 = 0.10$ 且 $\omega^2 = 0$。

主体 2 具有正确信念，因此 $\vartheta$ 的漂移为 $-\frac{1}{2}(\omega^1)^2 = -0.005$，且 $\upsilon_t \to 0$ 几乎必然成立。

主体 1 被驱向消亡——这是 {cite:t}`Blume_Easley2006` 的经典市场选择结果。

选择过程是缓慢的。

$\vartheta$ 的波动率为 $0.1$，因此在 $t = 200$ 时，$\vartheta_t$ 的均值 $-1$ 在量级上小于其标准差 $1.41$。

下表将 50 条模拟路径中 $\upsilon_t < 1/2$ 的比例与概率 $P(\vartheta_t < 0) = \Phi(0.05\sqrt{t})$ 进行了比较，其中 $\Phi$ 是标准正态分布函数。

在 $t = 200$ 时，约有四分之三的路径满足 $\upsilon_t < 1/2$；到 $t = 2000$ 时，几乎所有路径都满足这一条件。

## 资产定价含义

随着一个主体变得微不足道，当前价格收敛于由大主体构成的齐次经济体的价格。

当主体 2 是大主体时，{cite:t}`Borovicka2020` 中的命题 5.1 意味着

$$
\lim_{\upsilon \searrow 0} r(\upsilon)
= \beta + \rho \left(\mu_Y + \omega^2 \sigma_Y
+ \frac{1}{2} (1 - \gamma) \sigma_Y^2\right)
- \frac{1}{2} \gamma \sigma_Y^2
$$ (eq:riskfree)

以及

$$
\lim_{\upsilon \searrow 0} y(\upsilon)
= \left[
\beta - (1 - \rho)
\left(
\mu_Y + \omega^2 \sigma_Y + \frac{1}{2} (1 - \gamma) \sigma_Y^2
\right)
\right]^{-1} .
$$ (eq:wc_ratio)

总财富动态也收敛于齐次经济体的动态：

$$
\lim_{\upsilon \searrow 0} m_A(\upsilon) = \mu_Y,
\qquad
\lim_{\upsilon \searrow 0} \sigma_A(\upsilon) = \sigma_Y .
$$

命题 5.3 随后给出了微不足道主体自己的消费-储蓄和投资组合选择。

她的消费-财富比收敛于

$$
\lim_{\upsilon \searrow 0} (y^1(\upsilon))^{-1}
= (y(0))^{-1}
- \frac{1-\rho}{\rho}
\left[
(\omega^1 - \omega^2)\sigma_Y
+ \frac{(\omega^1 - \omega^2)^2}{2 \gamma}
\right] .
$$

小主体的风险资产份额收敛于

$$
\lim_{\upsilon \searrow 0} \pi^1(\upsilon)
= 1 + \frac{\omega^1 - \omega^2}{\gamma \sigma_Y} .
$$ (eq:portfolio)

因此乐观意味着杠杆，而足够强的悲观意味着做空。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 小主体的极限风险资产份额
    name: fig-limiting-portfolio-shares
---
ω_2 = 0.0
σ_y = 0.02
ω_grid = np.linspace(-0.5, 1.0, 300)

fig, ax = plt.subplots(figsize=(10, 5))

for γ in [2, 5, 10, 20]:
    π_1 = 1 + (ω_grid - ω_2) / (γ * σ_y)
    ax.plot(ω_grid, π_1, lw=2, label=rf"$\gamma = {γ}$")

ax.axhline(1.0, color="gray", linestyle=":", lw=1)
ax.axhline(0.0, color="gray", linestyle=":", lw=1)
ax.axvline(0.0, color="gray", linestyle=":", lw=1)
ax.set_xlabel(r"信念扭曲 $\omega^1$")
ax.set_ylabel(r"风险份额 $\pi^1$")
ax.legend()
plt.tight_layout()
plt.show()
```

此图将微不足道主体的极限风险资产份额 $\pi^1$ 绘制为她的信念扭曲 $\omega^1$ 的函数($\omega^2 = 0$，$\sigma_Y = 0.02$)，针对四个风险厌恶水平。

在 $\omega^1 = 0$ 处，主体与主体 2 意见一致并持有市场投资组合($\pi^1 = 1$)。

乐观($\omega^1 > 0$)导致杠杆($\pi^1 > 1$)，而足够的悲观($\omega^1 < 0$)导致做空($\pi^1 < 0$)。

较高的风险厌恶将这些偏离压缩至接近 1。

## 乐观和悲观扭曲

乐观和悲观信念对生存的影响是不对称的。

乐观主体从风险溢价项中获益，并且当 IES $> 1$ 时，也从储蓄项中获益。

悲观主体持有的风险资产少于持有市场组合的主体 2，即 $\pi^2 = 1$。

当 $\omega^2 = 0$ 时，在 $P$ 下使期望对数收益最大化的投资组合的风险份额为 $\gamma$。

当 $\gamma \geq 1$ 时，较小的风险份额使悲观主体偏离该组合，因此她获得的期望对数收益低于主体 2。

此时她只有在储蓄效应足够强以抵消该损失时才能生存。

当 IES $> 1$ 时，储蓄项只有在她持有较大的空头头寸（$\pi^1 < -1$）时才能帮助她；当 IES $< 1$ 时，储蓄项只有在 $-1 < \pi^1 < 1$ 时才能帮助她。

当 $\gamma < 1$ 时，市场组合比使期望对数收益最大化的组合杠杆更高。

此时一个具有 $2\gamma - 1 < \pi^1 < 1$ 的轻度悲观主体，无需借助储蓄渠道，就能获得高于主体 2 的期望对数收益。

例如，当 $\rho = 1$、$\gamma = 0.5$、$\sigma_Y = 0.02$ 且 $\omega^1 = -0.005$ 时，两个边界漂移都为正，悲观主体占主导地位。

```{code-cell} ipython3
print(f"m_ϑ(0+) = {boundary_drift(-0.005, 0.0, 0.5, 1.0, 0.02):.2e}, "
      f"m_ϑ(1-) = {-boundary_drift(0.0, -0.005, 0.5, 1.0, 0.02):.2e}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 乐观和悲观扭曲在边界处的财富增长差
    name: fig-optimistic-pessimistic-drifts
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

σ_y = 0.02
ω_2 = 0.0
ρ = 0.67
γ_grid = np.linspace(0.5, 25.0, 300)

ax = axes[0]
for ω_1 in [0.1, 0.25, 0.5, 1.0]:
    _, _, _, total = decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y)
    ax.plot(γ_grid, total, lw=2, label=rf"$\omega^1 = {ω_1}$")
ax.axhline(0, color="gray", lw=1)
ax.set_title("乐观", fontsize=12)
ax.set_xlabel(r"风险厌恶 $\gamma$")
ax.set_ylabel(r"$m_\vartheta(0+)/\gamma$")
ax.set_ylim(-0.03, 0.06)
ax.legend(fontsize=9)

ax = axes[1]
for ω_1 in [-0.1, -0.25, -0.5, -1.0]:
    _, _, _, total = decompose_survival(ω_1, ω_2, γ_grid, ρ, σ_y)
    ax.plot(γ_grid, total, lw=2, label=rf"$\omega^1 = {ω_1}$")
ax.axhline(0, color="gray", lw=1)
ax.set_title("悲观", fontsize=12)
ax.set_xlabel(r"风险厌恶 $\gamma$")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

f = lambda γ: decompose_survival(-1.0, ω_2, γ, ρ, σ_y)[3]
total = f(γ_grid)
idx = np.where(np.diff(np.sign(total)) != 0)[0]
roots = [brentq(f, γ_grid[i], γ_grid[i + 1]) for i in idx]
print("ω¹ = -1.0: the curve changes sign at γ =", np.round(roots, 2))
```

两个面板都将财富增长差 $\lim_{\upsilon \searrow 0}[m_A^1 - m_A^2] = \lim_{\upsilon \searrow 0} m_\vartheta/\gamma$ 绘制为 $\gamma$ 的函数（IES $\approx 1.49$，$\omega^2 = 0$）。

我们对纵轴进行了截断；当 $\gamma \to 0$ 时，每条曲线都会急剧下降。

曲线为正的地方，主体 1 在接近消亡时生存。

- 左面板（乐观主体）：较大的 $\omega^1$ 意味着对风险资产更大的押注，因此波动率惩罚在低 $\gamma$ 时占主导，但一旦 $\gamma$ 足够大漂移转为正。
- 右面板（悲观主体）：$\omega^1 = -0.1, -0.25, -0.5$ 的曲线始终为负。只有最悲观的主体（$\omega^1 = -1$）在接近消亡时得以生存，对应的 $\gamma$ 范围大约在 2.69 到 6.23 之间。在该区间内，她大量做空风险资产（$\pi^1 < -1$），预期其头寸能获得高回报，并储蓄足够多以抵消投资组合损失。

## 长期消费分布

当两个主体都生存时，帕累托份额持续在整个区间 $(0, 1)$ 中移动。

下一个模拟只是一个玩具近似。

它在两个边界值之间插值漂移，因此在不求解完整均衡 ODE 的情况下说明了常返逻辑。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 一个玩具平稳帕累托份额模拟
    name: fig-toy-stationary-pareto-share
---
def simulate_pareto_share_toy(ω_1, ω_2, γ, ρ, σ_y, T, dt, n_paths=20, seed=42):
    """
    通过插值边界漂移模拟玩具帕累托份额过程。
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)

    volatility = ω_1 - ω_2
    m_0 = boundary_drift(ω_1, ω_2, γ, ρ, σ_y)
    m_1 = -boundary_drift(ω_2, ω_1, γ, ρ, σ_y)

    θ = np.zeros((n_paths, n_steps + 1))
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    for t in range(n_steps):
        υ = 1.0 / (1.0 + np.exp(-θ[:, t]))
        drift = m_0 * (1 - υ) + m_1 * υ
        θ[:, t + 1] = θ[:, t] + drift * dt + volatility * dW[:, t]

    υ_paths = 1.0 / (1.0 + np.exp(-θ))
    return t_grid, υ_paths


ω_1 = 0.25
ω_2 = 0.0
γ = 5.0
ρ = 0.67
σ_y = 0.02

t_grid, υ_paths = simulate_pareto_share_toy(
    ω_1, ω_2, γ, ρ, σ_y, T=500, dt=0.05, n_paths=50, seed=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i in range(20):
    ax.plot(t_grid, υ_paths[i], color="C0", alpha=0.25, lw=1)
ax.axhline(0.5, color="gray", linestyle=":", lw=1)
ax.set_title("样本路径", fontsize=12)
ax.set_xlabel("时间")
ax.set_ylabel(r"帕累托份额 $\upsilon_t$")
ax.set_ylim(0, 1)

ax = axes[1]
_, υ_long = simulate_pareto_share_toy(
    ω_1, ω_2, γ, ρ, σ_y, T=2000, dt=0.05, n_paths=5, seed=123
)
υ_stationary = υ_long[:, υ_long.shape[1] // 2:].ravel()
ax.hist(υ_stationary, bins=80, density=True, color="steelblue",
        edgecolor="white", alpha=0.7)
ax.set_title("近似平稳密度", fontsize=12)
ax.set_xlabel(r"帕累托份额 $\upsilon$")
ax.set_ylabel("密度")
ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()
```

左面板显示了在共存区域内的参数下帕累托份额 $\upsilon_t$ 的 50 条样本路径($\omega^1 = 0.25$，$\omega^2 = 0$，$\gamma = 5$，IES $\approx 1.49$)。

与 {numref}`fig-crra-pareto-paths` 中的可分情形不同，这些路径不会漂向零——它们反复访问广泛的取值范围，在两个排斥边界之间来回弹跳。

右面板通过汇集较长模拟的后半部分来近似平稳密度。

内部模态与两个主体都不被驱向消亡相一致。

然而，这个玩具插值仅仅说明了常返逻辑；它并不重现 {cite:t}`Borovicka2020` 图 4 中定量的平稳消费份额密度，后者需要求解完整的内部均衡 ODE。

## 总结

递归偏好削弱了经典的市场选择结果。

投资组合收益渠道仍然奖励更乐观的信念。

波动率渠道仍然惩罚激进的头寸。

当 IES $> 1$ 时，储蓄渠道会奖励预期其投资组合收益较高的主体，而当 IES $< 1$ 时则会惩罚该主体。

两个边界漂移项相差 $(\omega^1 - \omega^2)^2(1/\rho - 1/\gamma)$。

只有当 $\gamma > \rho$ 时，两个主体才都能存活，这意味着主体偏好不确定性的早期解决。

只有当 $\gamma < \rho$ 时，每个主体才有可能以正概率占据主导地位。

共存并不要求 IES $> 1$：当 $\omega^1 = 0.10$、$\omega^2 = 0$ 且 $\sigma_Y = 0.02$ 时，在 $\gamma = 5$、$\rho = 1.5$ 的情形下两个主体都能存活。

当 $\gamma > \rho$ 时，递归偏好经济体可以支持在信念和投资组合头寸方面具有持续异质性的平稳长期财富分布。
