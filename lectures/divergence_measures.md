---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 统计散度度量
  headings:
    Overview: 概述
    Primer on entropy, cross-entropy, KL divergence: 熵、交叉熵、KL散度入门
    'Two Beta distributions: running example': 两个Beta分布：运行示例
    Kullback–Leibler divergence: Kullback–Leibler散度
    Jensen-Shannon divergence: Jensen-Shannon散度
    Chernoff entropy: Chernoff 熵
    Comparing divergence measures: 比较散度度量
    KL divergence and maximum-likelihood estimation: KL散度和最大似然估计
    Related lectures: 相关讲座
    Exercises: 练习
---

(divergence_measures)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 统计散度度量

```{contents} 目录
:depth: 2
```

## 概述

统计散度用于量化两个不同概率分布之间的差异，这些分布可能难以区分，原因如下：

  * 在一个分布下具有正概率的每个事件在另一个分布下也具有正概率

  * 这意味着没有"确凿证据"事件的发生能让统计学家确定数据一定服从其中某一个概率分布

统计散度是一个将两个概率分布映射到 $[0, +\infty]$ 中某个数的**函数**。

当且仅当两个分布相等时，该数值为零。

统计散度函数在统计学、信息论和现在许多人称之为"机器学习"的领域中发挥着重要作用。

本讲座描述了三种散度度量：

* **库尔贝克-莱布勒(KL)散度**

* **Jensen-Shannon (JS) 散度**
* **切尔诺夫熵**

这些概念将在多个 quantecon 课程中出现。

让我们首先导入必要的 Python 工具。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC', 'DejaVu Sans']
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import betaln, digamma
from scipy.stats import beta as beta_dist
import pandas as pd
from IPython.display import display, Math
```

## 熵、交叉熵、KL散度入门

在深入之前，我们先介绍一些有用的基本概念。

我们暂时假设 $f$ 和 $g$ 是离散随机变量在状态空间 $I = \{1, 2, \ldots, n\}$ 上的两个概率质量函数，满足 $f_i \geq 0, \sum_{i} f_i =1, g_i \geq 0, \sum_{i} g_i =1$。

我们遵循一些统计学家和信息论学家的做法，将从分布 $f$ 中观察到单次抽样 $x = i$ 所关联的**惊奇度**或**惊奇量**定义为

$$
\log\left(\frac{1}{f_i}\right)
$$

他们进一步将从单次实现中预期获得的**信息量**定义为期望惊奇度

$$
H(f) = \sum_i f_i \log\left(\frac{1}{f_i}\right).  
$$

Claude Shannon {cite}`shannon1948mathematical` 将 $H(f)$ 称为分布 $f$ 的**熵**。

```{note}
通过对 $\{f_1, f_2, \ldots, f_n\}$ 在约束 $\sum_i f_i = 1$ 下最大化 $H(f)$，我们可以验证使熵最大化的分布是均匀分布
$
f_i = \frac{1}{n} .
$
均匀分布的熵 $H(f)$ 等于 $\log n$。
```

{cite:t}`kullback1951information` 将单次抽样 $x = i$ 用于区分 $f$ 和 $g$ 所提供的信息量定义为对数似然比

$$
\log \frac{f_i}{g_i}
$$

以下两个概念被广泛用于比较两个分布 $f$ 和 $g$。

**交叉熵:**

\begin{equation}
H(f,g) = -\sum_{i} f_i \log g_i
\end{equation}



**Kullback-Leibler (KL) 散度：** 
\begin{equation}
D_{KL}(f \parallel g) = \sum_{i} f_i \log\left[\frac{f_i}{g_i}\right]
\end{equation}

这些概念通过以下等式相关联。

$$
D_{KL}(f \parallel g) = H(f,g) - H(f)
$$ (eq:KLcross)

要证明{eq}`eq:KLcross`，注意到


\begin{align}
D_{KL}(f \parallel g) &= \sum_{i} f_i \log\left[\frac{f_i}{g_i}\right] \\
&= \sum_{i} f_i \left[\log f_i - \log g_i\right] \\
&= \sum_{i} f_i \log f_i - \sum_{i} f_i \log g_i \\
&= -H(f) + H(f,g) \\
&= H(f,g) - H(f)
\end{align}

记住$H(f)$是从$f$中抽取$x$时的预期惊异度。

交叉熵 $H(f,g) = \sum_i f_i \log(1/g_i)$ 是当$x$从$f$中抽取，但用$g$来计算惊异度时的预期惊异度。

那么上述等式告诉我们，KL散度是当预期$x$是从$g$中抽取而实际上是从$f$中抽取时产生的预期"额外惊异度"。

## 两个Beta分布：运行示例

我们将广泛使用Beta分布来说明概念。

Beta分布特别方便，因为它定义在$[0,1]$上，并且通过适当选择其两个参数可以呈现多样的形状。

具有参数$a$和$b$的Beta分布的密度函数为

$$
f(z; a, b) = \frac{\Gamma(a+b) z^{a-1} (1-z)^{b-1}}{\Gamma(a) \Gamma(b)}
\quad \text{其中} \quad
\Gamma(p) := \int_{0}^{\infty} x^{p-1} e^{-x} dx
$$

我们引入两个Beta分布$f(x)$和$g(x)$，我们将用它们来说明不同的散度度量。

让我们在Python中定义参数和密度函数

```{code-cell} ipython3
# 两个Beta分布中的参数
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# 两个密度函数
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

# 绘制分布图
x_range = np.linspace(0.001, 0.999, 1000)
f_vals = [f(x) for x in x_range]
g_vals = [g(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x) \sim \text{Beta}(1,1)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x) \sim \text{Beta}(3,1.2)$')

# 填充重叠区域
overlap = np.minimum(f_vals, g_vals)
plt.fill_between(x_range, 0, overlap, alpha=0.3, color='purple', label='overlap')

plt.xlabel('x')
plt.ylabel('密度')
plt.legend()
plt.show()
```

(rel_entropy)=
## Kullback–Leibler散度

我们的第一个散度函数是**Kullback–Leibler (KL)散度**。

对于概率密度（或概率质量函数）$f$和$g$，它的定义为

$$
D_{KL}(f\|g) = KL(f, g) = \int f(x) \log \frac{f(x)}{g(x)} \, dx.
$$

我们可以将$D_{KL}(f\|g)$解释为当数据由$f$生成而我们使用$g$时产生的预期超额对数损失（预期超额意外性）。

除非$f$关于$g$绝对连续（$f \ll g$），也就是说除非在$f(x) > 0$的地方都有$g(x) > 0$，否则$D_{KL}(f\|g) = +\infty$。

它有几个重要的性质：

- 非负性（Gibbs不等式）：$D_{KL}(f\|g) \ge 0$，当且仅当$f$几乎处处等于$g$时取等号
- 不对称性：$D_{KL}(f\|g) \neq D_{KL}(g\|f)$（因此它不是度量）
- 信息分解：
  $D_{KL}(f\|g) = H(f,g) - H(f)$，其中$H(f,g)$是交叉熵，$H(f)$是$f$的Shannon熵
- 链式法则：对于联合分布$f(x, y)$和$g(x, y)$，
  $D_{KL}(f(x,y)\|g(x,y)) = D_{KL}(f(x)\|g(x)) + E_{f}\left[D_{KL}(f(y|x)\|g(y|x))\right]$
- 数据处理不等式：如果马尔可夫核$\kappa$将$x$映射为更粗粒度的信号$y$，并且$f\kappa$和$g\kappa$表示$y$的诱导分布，那么$D_{KL}(f\kappa\|g\kappa) \le D_{KL}(f\|g)$。

数据处理不等式说明，对数据进行粗化处理不能使$f$和$g$更容易区分。

{doc}`blackwell_kihlstrom` 中关于 {ref}`数据处理不等式与粗粒度化 <bk_data_processing>` 的部分将其与布莱克韦尔的实验排序联系起来。

下面定义的JS散度和切尔诺夫熵也满足数据处理不等式。

KL散度在统计推断中扮演着核心角色，包括模型选择和假设检验。

{doc}`likelihood_ratio_process` 描述了KL散度与预期对数似然比之间的联系。

{doc}`wald_friedman` 将KL散度与序贯概率比检验的预期停止时间联系起来。

{doc}`mix_model` 使用两个KL散度来确定当数据来自贝叶斯模型所排除的$f$和$g$的混合分布时，贝叶斯后验概率的极限。

让我们计算示例分布$f$和$g$之间的KL散度。

U形贝塔密度在$0$或$1$处发散。

我们对整个$(0,1)$区间进行积分，并在$1/2$处将区间分割，以便`quad`每次只处理一个端点奇异性。

```{code-cell} ipython3
def integrate(F):
    """在(0, 1)上积分F，在1/2处将区间分割。"""
    left, _ = quad(F, 0, 0.5, limit=200)
    right, _ = quad(F, 0.5, 1, limit=200)
    return left + right

def compute_KL(f, g):
    """
    通过数值积分计算KL散度KL(f, g)
    """
    def integrand(w):
        fw = f(w)
        gw = g(w)
        return fw * np.log(fw / gw)
    return integrate(integrand)

# 计算我们示例分布之间的KL散度
kl_fg = compute_KL(f, g)
kl_gf = compute_KL(g, f)

print(f"KL(f, g) = {kl_fg:.4f}")
print(f"KL(g, f) = {kl_gf:.4f}")
```

KL散度的不对称性具有重要的实际意义。

$D_{KL}(f\|g)$ 会严重惩罚那些 $f > 0$ 但 $g$ 接近零的区域，因为$f$在那里将权重赋予了较大的$\log(f/g)$值。

$D_{KL}(f\|g)$ 几乎不惩罚那些 $g > 0$ 但 $f$ 接近零的区域，因为$f$在那里几乎没有权重。

$D_{KL}(g\|f)$ 则将这些角色互换。

在我们的例子中，$g$在$x = 0$处为零，而$f$在该处等于$1$，因此$D_{KL}(f\|g)$超过了$D_{KL}(g\|f)$。

## Jensen-Shannon散度

有时我们需要一个对称的散度度量，用来衡量两个分布之间的差异，而不偏向任何一方。

这种情况经常出现在聚类等应用中，我们想要比较分布，但不假设其中一个是真实模型。

**Jensen-Shannon (JS) 散度**通过将两个分布与它们的混合分布进行比较来使KL散度对称化：

$$
JS(f,g) = \frac{1}{2} D_{KL}(f\|m) + \frac{1}{2} D_{KL}(g\|m), \quad m = \frac{1}{2}(f+g).
$$

其中 $m$ 是对 $f$ 和 $g$ 取平均的混合分布

让我们也可视化混合分布 $m$：

```{code-cell} ipython3
def m(x):
    return 0.5 * (f(x) + g(x))

m_vals = [m(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x)$')
plt.plot(x_range, m_vals, 'g--', linewidth=2, label=r'$m(x) = \frac{1}{2}(f(x) + g(x))$')

plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

JS散度具有以下几个有用的性质：

- 对称性：$JS(f,g)=JS(g,f)$。
- 有界性：$0 \le JS(f,g) \le \log 2$。
- 其平方根$\sqrt{JS}$在概率分布空间上是一个度量（Jensen-Shannon距离）。
- JS散度等于二元随机变量$Z \sim \text{Bernoulli}(1/2)$（用于指示源）与样本$X$之间的互信息，其中当$Z=0$时$X$从$f$抽样，当$Z=1$时从$g$抽样。

Jensen-Shannon散度出现在生成对抗网络的原始构想中：当判别器达到最优时，生成器所最小化的目标是 $2\, JS(p_{\text{data}}, p_{\text{model}}) - \log 4$ {cite}`goodfellow2014`。

有界性是有代价的。

当两个分布的支持几乎不相交时，$JS$ 会接近其上限 $\log 2$，并且在模型参数上几乎是平坦的，这使得基于梯度的训练陷入停滞 {cite}`arjovsky2017`。

让我们计算示例分布$f$和$g$之间的JS散度

```{code-cell} ipython3
def compute_JS(f, g):
    """计算Jensen-Shannon散度。"""
    def integrand(w):
        fw, gw = f(w), g(w)
        mw = 0.5 * (fw + gw)
        return 0.5 * fw * np.log(fw / mw) + 0.5 * gw * np.log(gw / mw)
    return integrate(integrand)

js_div = compute_JS(f, g)
print(f"Jensen-Shannon散度 JS(f,g) = {js_div:.4f}")
```

我们可以使用带权重 $\alpha = (\alpha_i)_{i=1}^{n}$ 的广义 Jensen-Shannon 散度轻松推广到两个以上的分布:

$$
JS_\alpha(f_1, \ldots, f_n) = 
H\left(\sum_{i=1}^n \alpha_i f_i\right) - \sum_{i=1}^n \alpha_i H(f_i)
$$

其中:
- $\alpha_i \geq 0$ 且 $\sum_{i=1}^n \alpha_i = 1$，以及
- $H(f) = -\int f(x) \log f(x) dx$ 是密度 $f$ 的**微分熵**

与概率质量函数的熵不同，微分熵可以为负：$[0, 1/2]$ 上的均匀密度的微分熵为 $-\log 2$。

尽管如此，$JS_\alpha$ 仍是非负的，因为 $H$ 是凹函数。

## Chernoff 熵

Chernoff 熵源自[大偏差理论](https://en.wikipedia.org/wiki/Large_deviations_theory)的早期应用，该理论通过提供罕见事件的指数衰减率来改进中心极限近似。

对于密度函数 $f$ 和 $g$，Chernoff 熵（由 {cite:t}`chernoff1952` 引入，信息论学家称之为 Chernoff 信息）为

$$
C(f,g) = - \log \min_{\phi \in (0,1)} \rho(\phi), \qquad 
\rho(\phi) = \int f^{\phi}(x) g^{1-\phi}(x) \, dx.
$$

注释：

- 积分 $\rho(\phi)$ 是 **Chernoff 系数**。
- 当 $\phi=1/2$ 时，它变成 **Bhattacharyya 系数** $\int \sqrt{f g}$。
- 在具有 $T$ 个独立同分布观测、且对 $f$ 和 $g$ 的先验概率相等的二元假设检验中，可达到的最小错误概率 $P_e(T)$ 以指数速率 $C(f,g)$ 衰减：$-\frac{1}{T} \log P_e(T) \rightarrow C(f,g)$。

{doc}`merging_of_opinions` 将 Bhattacharyya 系数称为 Hellinger 亲和度。

该讲座中的 Kakutani 定理表明，当 $\int \sqrt{fg} < 1$ 时，从 $f$ 和从 $g$ 中抽取的无限独立同分布序列的分布是相互奇异的。

我们将在 {doc}`likelihood_ratio_process` 讲座中看到第三点的一个例子，
我们将在模型选择的背景下研究 Chernoff 熵。

Chernoff 熵介于 Bhattacharyya 距离 $-\log \int \sqrt{fg}$ 与两个 KL 散度的组合之间：

$$
-\log \int \sqrt{f g} \, dx
\;\le\; C(f,g)
\;\le\; \frac{D_{KL}(f\|g)\, D_{KL}(g\|f)}{D_{KL}(f\|g) + D_{KL}(g\|f)}
\;\le\; \min\{D_{KL}(f\|g),\, D_{KL}(g\|f)\} .
$$ (eq:chernoff_sandwich)

当 $f$ 和 $g$ 在同一集合上为正时，$\log \rho(\phi)$ 是 $\phi$ 的凸函数，在 $\phi = 0$ 和 $\phi = 1$ 处均等于 $0$。

它在 $\phi = 0$ 处的斜率为 $-D_{KL}(g\|f)$，在 $\phi = 1$ 处的斜率为 $D_{KL}(f\|g)$。

{eq}`eq:chernoff_sandwich` 中的下界在 $\phi = 1/2$ 处取 $\log \rho$ 的值；上界则使用两端点处的切线，这两条切线都位于 $\log \rho$ 之下。

{ref}`div_ex2` 要求你证明这些论断。

让我们计算示例分布 $f$ 和 $g$ 之间的 Chernoff 熵，并验证 {eq}`eq:chernoff_sandwich` 中的界。

```{code-cell} ipython3
def chernoff_integrand(ϕ, f, g):
    """计算给定 ϕ 的 Chernoff 熵中的积分。"""
    def integrand(w):
        return f(w)**ϕ * g(w)**(1-ϕ)
    return integrate(integrand)

def compute_chernoff_entropy(f, g):
    """计算 Chernoff 熵 C(f,g)。"""
    def objective(ϕ):
        return chernoff_integrand(ϕ, f, g)
    result = minimize_scalar(objective, bounds=(1e-5, 1-1e-5), method='bounded')
    min_value = result.fun
    ϕ_optimal = result.x
    chernoff_entropy = -np.log(min_value)
    return chernoff_entropy, ϕ_optimal

C_fg, ϕ_optimal = compute_chernoff_entropy(f, g)
print(f"Chernoff 熵 C(f,g) = {C_fg:.4f}")
print(f"最优 ϕ = {ϕ_optimal:.4f}")

bhattacharyya = -np.log(chernoff_integrand(0.5, f, g))
kl_bound = kl_fg * kl_gf / (kl_fg + kl_gf)
print(f"{bhattacharyya:.4f} ≤ {C_fg:.4f} ≤ {kl_bound:.4f} "
      f"≤ {min(kl_fg, kl_gf):.4f}")
```

## 比较散度度量

我们现在比较几对Beta分布之间的这些度量。

对于Beta密度，KL散度和切尔诺夫系数都有封闭形式解。

设 $B(a,b)$ 为Beta函数，$\psi = \Gamma'/\Gamma$ 为双伽玛函数。

那么

$$
D_{KL}\bigl(\text{Beta}(a_1,b_1)\,\|\,\text{Beta}(a_2,b_2)\bigr)
= \log\frac{B(a_2,b_2)}{B(a_1,b_1)} + (a_1-a_2)\psi(a_1) + (b_1-b_2)\psi(b_1)
+ (a_2-a_1+b_2-b_1)\psi(a_1+b_1)
$$

以及

$$
\rho(\phi) = \frac{B\bigl(\phi a_1 + (1-\phi) a_2,\ \phi b_1 + (1-\phi) b_2\bigr)}
{B(a_1,b_1)^{\phi}\, B(a_2,b_2)^{1-\phi}} .
$$

{ref}`div_ex1` 要求你推导这些公式。

JS散度没有封闭形式解，因此我们通过在 $(0,1)$ 上进行数值积分来计算它。

积分必须覆盖整个区间：Beta$(0.1, 0.2)$ 将四分之一的概率质量分布在 $[10^{-5}, 1-10^{-5}]$ 之外。

```{code-cell} ipython3
def KL_beta(a1, b1, a2, b2):
    """以封闭形式计算 KL(Beta(a1, b1) ‖ Beta(a2, b2))。"""
    return (betaln(a2, b2) - betaln(a1, b1)
            + (a1 - a2) * digamma(a1) + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1))

def log_chernoff_coef_beta(ϕ, a1, b1, a2, b2):
    """以封闭形式计算 Beta(a1, b1) 和 Beta(a2, b2) 的 log ρ(ϕ)。"""
    return (betaln(ϕ * a1 + (1 - ϕ) * a2, ϕ * b1 + (1 - ϕ) * b2)
            - ϕ * betaln(a1, b1) - (1 - ϕ) * betaln(a2, b2))

def chernoff_beta(a1, b1, a2, b2):
    """计算两个Beta密度的切尔诺夫熵及使其最小化的 ϕ。"""
    res = minimize_scalar(log_chernoff_coef_beta, bounds=(0, 1),
                          args=(a1, b1, a2, b2), method='bounded')
    return -res.fun, res.x

mass_out = beta_dist.cdf(1e-5, 0.1, 0.2) + beta_dist.sf(1 - 1e-5, 0.1, 0.2)
print(f"Mass of Beta(0.1, 0.2) outside [1e-5, 1-1e-5] = {mass_out:.3f}")
```

```{code-cell} ipython3
:tags: [hide-input]

distribution_pairs = [
    # (f_params, g_params)
    ((1, 1), (0.1, 0.2)),
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (0.3, 0.4)),
    ((1, 1), (0.5, 0.5)),
    ((1, 1), (0.7, 0.6)),
    ((1, 1), (0.9, 0.8)),
    ((1, 1), (1.1, 1.05)),
    ((1, 1), (1.2, 1.1)),
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),
    ((1, 1), (2.5, 1.8)),
    ((1, 1), (3, 1.2)),
    ((1, 1), (4, 1)),
    ((1, 1), (5, 1))
]

# 创建比较表
results = []
for (f_a, f_b), (g_a, g_b) in distribution_pairs:
    f_pair = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g_pair = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
    results.append({
        'Pair (f, g)': f"\\text{{Beta}}({f_a},{f_b}), \\text{{Beta}}({g_a},{g_b})",
        'KL(f, g)': KL_beta(f_a, f_b, g_a, g_b),
        'KL(g, f)': KL_beta(g_a, g_b, f_a, f_b),
        'JS': compute_JS(f_pair, g_pair),
        'C': chernoff_beta(f_a, f_b, g_a, g_b)[0]
    })

# 按JS散度排序
df = pd.DataFrame(results).sort_values('JS')

columns = ' & '.join([f'\\text{{{col}}}' for col in df.columns])
rows = ' \\\\\n'.join(
    [' & '.join([row[0]] + [f'{val:.4f}' for val in row[1:]]) 
     for row in df.values])

latex_code = rf"""
\begin{{array}}{{lcccc}}
{columns} \\
\hline
{rows}
\end{{array}}
"""

display(Math(latex_code))
```

JS散度和切尔诺夫熵对这十四对分布的排序相同。

两个KL散度对它们的排序则有所不同。

对于 $g = $ Beta$(0.1, 0.2)$，$KL(g, f)$ 在表中是迄今为止最大的，因为 $g$ 的大部分质量集中在端点附近，而在那里 $g$ 远大于 $f = 1$。

对于 $g = $ Beta$(4, 1)$ 和 Beta$(5, 1)$，$KL(f, g)$ 超过了其在 Beta$(0.1, 0.2)$ 情形下的值，因为这些 $g$ 在 $0$ 处趋于零，而 $f = 1$。

接下来我们可视化KL散度、JS散度和切尔诺夫熵之间的关系。

```{code-cell} ipython3
kl_fg_values = [result['KL(f, g)'] for result in results]
js_values = [result['JS'] for result in results]
chernoff_values = [result['C'] for result in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(kl_fg_values, js_values, alpha=0.7, s=60)
axes[0].set_xlabel('KL散度 KL(f, g)')
axes[0].set_ylabel('JS散度')
axes[0].set_title('JS散度与KL散度的关系')

axes[1].scatter(js_values, chernoff_values, alpha=0.7, s=60)
axes[1].set_xlabel('JS散度')
axes[1].set_ylabel('切尔诺夫熵')
axes[1].set_title('切尔诺夫熵与JS散度的关系')

plt.tight_layout()
plt.show()
```

现在我们生成图表来直观展示随着差异度量的增加，重叠程度如何减少。

```{code-cell} ipython3
param_grid = [
    ((1, 1), (1, 1)),   
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),  
    ((1, 1), (3, 1.2)),  
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (5, 1))
]
```

```{code-cell} ipython3
:tags: [hide-input]

def plot_dist_diff(para_grid):
    """绘制选定Beta分布对的重叠图。"""

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    divergence_data = []
    for i, ((f_a, f_b), (g_a, g_b)) in enumerate(para_grid):
        row, col = divmod(i, 2)
        f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
        g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
        kl_fg = KL_beta(f_a, f_b, g_a, g_b)
        js_div = compute_JS(f, g)
        chernoff_ent, _ = chernoff_beta(f_a, f_b, g_a, g_b)
        divergence_data.append({
            'f_params': (f_a, f_b),
            'g_params': (g_a, g_b),
            'kl_fg': kl_fg,
            'js_div': js_div,
            'chernoff': chernoff_ent
        })
        x_range = np.linspace(0, 1, 200)
        f_vals = [f(x) for x in x_range]
        g_vals = [g(x) for x in x_range]
        axes[row, col].plot(x_range, f_vals, 'b-', 
                        linewidth=2, label=f'f ~ Beta({f_a},{f_b})')
        axes[row, col].plot(x_range, g_vals, 'r-', 
                        linewidth=2, label=f'g ~ Beta({g_a},{g_b})')
        overlap = np.minimum(f_vals, g_vals)
        axes[row, col].fill_between(x_range, 0, 
                        overlap, alpha=0.3, color='purple', label='重叠')
        axes[row, col].set_title(
            f'KL(f,g)={kl_fg:.3f}, JS={js_div:.3f}, C={chernoff_ent:.3f}', 
            fontsize=12)
        axes[row, col].legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    return divergence_data

divergence_data = plot_dist_diff(param_grid)
```

(dm_kl_mle)=
## KL散度和最大似然估计

设 $x_1, x_2, \ldots, x_N$ 是从密度 $p_0$ 中独立同分布抽取的样本。

统计学家通过最大似然法拟合参数密度族 $\{p_\theta\}$。

**经验分布** $p_e$ 对每个观测值赋予概率 $1/N$：

$$
p_e(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i) ,
$$

其中 $\delta(x - x_i)$ 是中心在 $x_i$ 的狄拉克德尔塔函数；它满足对于 $x \neq x_i$ 有 $\delta(x - x_i) = 0$，且 $\int \delta(x - x_i) \, dx = 1$。

参数 $\theta$ 的对数似然函数为

$$
\ell_N(\theta) = \sum_{i=1}^N \log p_\theta(x_i) .
$$

平均对数似然是在 $p_e$ 下的期望：

$$
\frac{1}{N} \ell_N(\theta) = \int p_e(x) \log p_\theta(x) \, dx = - H(p_e, p_\theta) ,
$$

其中 $H(p_e, p_\theta)$ 是上文入门部分定义的交叉熵。

最大似然法选择 $\theta$ 以最小化交叉熵 $H(p_e, p_\theta)$。

KL散度 $D_{KL}(p_e \| p_\theta)$ 在这里没有用处。

当 $p_\theta$ 是一个密度函数时，$p_e$ 将概率一赋予有限集合 $\{x_1, \ldots, x_N\}$，而 $p_\theta$ 对该集合赋予的概率为零。

因此 $p_e$ 相对于 $p_\theta$ 不是绝对连续的，对于任意 $\theta$ 都有 $D_{KL}(p_e \| p_\theta) = +\infty$。

KL散度是通过总体分布引入的。

根据强大数定律，对每个 $\theta$

$$
\frac{1}{N} \ell_N(\theta) \rightarrow E_{p_0} \log p_\theta(X) = - H(p_0) - D_{KL}(p_0 \| p_\theta)
\quad \text{几乎必然成立},
$$

其中等式来自密度情形下的 {eq}`eq:KLcross`，并且我们假设熵 $H(p_0)$ 是有限的。

熵 $H(p_0)$ 不依赖于 $\theta$。

最大似然法是对 $\theta$ 最小化 $D_{KL}(p_0 \| p_\theta)$ 的样本类比。

如果对于某个 $\theta_0$ 有 $p_0 = p_{\theta_0}$，则吉布斯不等式意味着 $\theta_0$ 使 $D_{KL}(p_0 \| p_\theta)$ 最小化。

如果该分布族中没有任何成员等于 $p_0$，那么在正则条件下，最大似然估计量会收敛到**伪真值**

$$
\theta^* = \arg\min_\theta D_{KL}(p_0 \| p_\theta) ,
$$

即该分布族中在KL散度意义下最接近 $p_0$ 的成员 {cite}`White1982`。

在积分号下求导可得

$$
\nabla_\theta D_{KL}(p_0 \| p_\theta) = - E_{p_0} \nabla_\theta \log p_\theta(X) .
$$

期望得分是KL散度的负梯度，并且在 $\theta^*$ 处为零。

{doc}`gallant_tauchen` 基于这一事实构建了一个估计量。

{doc}`mle` 描述了如何计算最大似然估计。

{ref}`div_ex3` 将一个设定有误的贝塔模型拟合到由混合分布生成的数据上。

## 相关讲座

本讲座介绍了我们将在其他地方遇到的工具。

- 其他应用散度度量与统计推断之间联系的 quantecon 讲座包括 {doc}`likelihood_ratio_process`、{doc}`wald_friedman` 和 {doc}`mix_model`。

- 在研究 Lawrence Blume 和 David Easley 的异质信念和金融市场模型的 {doc}`likelihood_ratio_process_2` 中，统计散度函数也占据核心地位。

- {doc}`imp_sample` 使用了 $\chi^2$ 散度 $\chi^2(f\|q) = \int (f - q)^2 / q \, dx = \int f^2/q \, dx - 1$，该值等于重要性权重 $f/q$ 在 $q$ 下的方差。

- {doc}`merging_of_opinions` 使用了巴塔查里亚系数（又称海林格亲和度）来判断两个贝叶斯学派的观点何时会趋于一致。

## 练习

```{exercise}
:label: div_ex1

设 $X \sim \text{Beta}(a, b)$，其在 $(0,1)$ 上的密度为 $x^{a-1}(1-x)^{b-1}/B(a,b)$。

1. 对 $B(a,b) = \int_0^1 x^{a-1}(1-x)^{b-1} dx$ 关于 $a$ 和 $b$ 求导，证明 $E \log X = \psi(a) - \psi(a+b)$ 以及 $E \log(1-X) = \psi(b) - \psi(a+b)$。
2. 利用第 1 部分推导出上文所述 $D_{KL}\bigl(\text{Beta}(a_1,b_1)\,\|\,\text{Beta}(a_2,b_2)\bigr)$ 的闭式解。
3. 证明当 $f$ 和 $g$ 是贝塔密度时，$f^\phi g^{1-\phi}$ 正比于某个贝塔密度，并利用这一事实推导出 $\rho(\phi)$ 的闭式解。
4. 设 $f = \text{Beta}(1,1)$，$g = \text{Beta}(0.1, 0.2)$。通过对截断区间 $[10^{-5}, 1-10^{-5}]$ 应用 `quad`，计算 $KL(g, f)$、$JS(f, g)$ 和 $C(f, g)$。将结果与上表中的值进行比较，并解释差异的原因。
```

```{solution-start} div_ex1
:class: dropdown
```

**第 1 部分。** 在积分号下求导，可得

$$
\frac{\partial B(a,b)}{\partial a} = \int_0^1 \log x \; x^{a-1}(1-x)^{b-1} dx = B(a,b) \, E \log X .
$$

由于 $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$，

$$
E \log X = \frac{\partial \log B(a,b)}{\partial a} = \psi(a) - \psi(a+b) .
$$

用 $b$ 代替 $a$ 做相同的论证，可得 $E \log(1-X) = \psi(b) - \psi(a+b)$。

**第 2 部分。** 设 $f = \text{Beta}(a_1, b_1)$，$g = \text{Beta}(a_2, b_2)$。

则

$$
\log f(x) - \log g(x) = (a_1 - a_2) \log x + (b_1 - b_2) \log(1-x) + \log B(a_2, b_2) - \log B(a_1, b_1) .
$$

对 $f$ 取期望并利用第 1 部分，可得

$$
D_{KL}(f\|g) = \log\frac{B(a_2,b_2)}{B(a_1,b_1)} 
+ (a_1 - a_2)\bigl[\psi(a_1) - \psi(a_1+b_1)\bigr] 
+ (b_1 - b_2)\bigl[\psi(b_1) - \psi(a_1+b_1)\bigr] ,
$$

整理后即可得到正文中的公式。

**第 3 部分。** 设 $a_\phi = \phi a_1 + (1-\phi) a_2$，$b_\phi = \phi b_1 + (1-\phi) b_2$。

则

$$
f(x)^\phi g(x)^{1-\phi} = \frac{x^{a_\phi - 1}(1-x)^{b_\phi - 1}}{B(a_1,b_1)^\phi B(a_2,b_2)^{1-\phi}} ,
$$

这正比于 Beta$(a_\phi, b_\phi)$ 的密度。

在 $(0,1)$ 上积分，可得

$$
\rho(\phi) = \frac{B(a_\phi, b_\phi)}{B(a_1,b_1)^\phi B(a_2,b_2)^{1-\phi}} .
$$

**第 4 部分。** 以下代码在截断区间上重复了上述计算。

```{code-cell} ipython3
def KL_truncated(f, g):
    """KL(f, g) by quad over [1e-5, 1-1e-5]."""
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    return quad(integrand, 1e-5, 1 - 1e-5)[0]

def chernoff_truncated(f, g):
    """Chernoff entropy by quad over [1e-5, 1-1e-5]."""
    ρ = lambda ϕ: quad(lambda w: f(w)**ϕ * g(w)**(1-ϕ),
                       1e-5, 1 - 1e-5)[0]
    res = minimize_scalar(ρ, bounds=(0, 1), method='bounded')
    return -np.log(res.fun)

f_ex = jit(lambda x: p(x, 1, 1))
g_ex = jit(lambda x: p(x, 0.1, 0.2))
m_ex = lambda x: 0.5 * (f_ex(x) + g_ex(x))

truncated = [KL_truncated(g_ex, f_ex),
             0.5 * KL_truncated(f_ex, m_ex) + 0.5 * KL_truncated(g_ex, m_ex),
             chernoff_truncated(f_ex, g_ex)]
accurate = [KL_beta(0.1, 0.2, 1, 1),
            compute_JS(f_ex, g_ex),
            chernoff_beta(1, 1, 0.1, 0.2)[0]]

for name, t, a in zip(['KL(g, f)', 'JS(f, g)', 'C(f, g)'], truncated, accurate):
    print(f"{name}: truncated = {t:.4f}, accurate = {a:.4f}")
print(f"Mass of g outside the truncated interval = {mass_out:.4f}")
```

密度 $g = \text{Beta}(0.1, 0.2)$ 有四分之一的概率质量落在 $[10^{-5}, 1-10^{-5}]$ 之外，而在该区域 $g$ 的值极大，$f = 1$。

$KL(g, f) = \int g \log g$ 中被忽略的部分很大，因为那里的 $\log g$ 很大。

$JS(f, g)$ 中被忽略的部分接近 $\frac{1}{2} \times 0.25 \times \log 2 \approx 0.087$，因为在 $g \gg f$ 的地方，$g \log\bigl(2g/(f+g)\bigr) \approx g \log 2$。

截断同样会漏算 $\rho(\phi)$ 的一部分，这会降低 $\rho$ 的值，并使计算得到的切尔诺夫熵偏高。

```{solution-end}
```

```{exercise}
:label: div_ex2

设 $f$ 和 $g$ 是在同一集合上为正的密度，并令 $\rho(\phi) = \int f^\phi g^{1-\phi} dx$。

1. 利用赫尔德不等式证明 $\log \rho(\phi)$ 在 $[0,1]$ 上是凸函数。
2. 证明 $\log\rho(0) = \log\rho(1) = 0$，$\log \rho$ 在 $\phi = 0$ 处的导数为 $-D_{KL}(g\|f)$，在 $\phi = 1$ 处的导数为 $D_{KL}(f\|g)$。
3. 由此推导出 {eq}`eq:chernoff_sandwich` 中的界。
4. 对上表中的每一对分布进行数值验证，并针对我们例子中的 $f$ 和 $g$，绘制 $-\log\rho(\phi)$ 及其两条切线。
```

```{solution-start} div_ex2
:class: dropdown
```

**第 1 部分。** 取 $\phi_1, \phi_2 \in [0,1]$，$\lambda \in (0,1)$，令 $\phi = \lambda \phi_1 + (1-\lambda) \phi_2$。

写成

$$
f^\phi g^{1-\phi} = \left(f^{\phi_1} g^{1-\phi_1}\right)^{\lambda} \left(f^{\phi_2} g^{1-\phi_2}\right)^{1-\lambda} .
$$

以指数 $1/\lambda$ 和 $1/(1-\lambda)$ 应用赫尔德不等式，可得

$$
\rho(\phi) \le \rho(\phi_1)^{\lambda} \rho(\phi_2)^{1-\lambda} .
$$

取对数即得 $\log \rho(\phi) \le \lambda \log\rho(\phi_1) + (1-\lambda) \log\rho(\phi_2)$。

**第 2 部分。** 因为 $f$ 和 $g$ 在同一集合上为正，所以 $\rho(0) = \int g = 1$，$\rho(1) = \int f = 1$。

在积分号下求导，可得

$$
\frac{d \log \rho(\phi)}{d\phi} = \frac{1}{\rho(\phi)} \int f^\phi g^{1-\phi} \log \frac{f}{g} \, dx .
$$

在 $\phi = 0$ 处，导数为 $\int g \log(f/g) = -D_{KL}(g\|f)$。

在 $\phi = 1$ 处，导数为 $\int f \log(f/g) = D_{KL}(f\|g)$。

**第 3 部分。** 下界成立是因为 $C(f,g) = -\min_\phi \log\rho(\phi) \ge -\log \rho(1/2)$。

凸函数位于其切线之上，所以

$$
\log \rho(\phi) \ge -\phi\, D_{KL}(g\|f)
\quad \text{且} \quad
\log \rho(\phi) \ge -(1-\phi)\, D_{KL}(f\|g) .
$$

因此

$$
-\log\rho(\phi) \le \min\bigl\{\phi\, D_{KL}(g\|f),\ (1-\phi)\, D_{KL}(f\|g)\bigr\} .
$$

右边在两条直线相交处取得最大值，即在 $\phi = D_{KL}(f\|g)/\bigl(D_{KL}(f\|g) + D_{KL}(g\|f)\bigr)$ 处，此时右边等于 $D_{KL}(f\|g)\, D_{KL}(g\|f)/\bigl(D_{KL}(f\|g) + D_{KL}(g\|f)\bigr)$。

对左边关于 $\phi$ 取最大值，即得中间的不等式。

最后一个不等式成立是因为对正数 $x$ 和 $y$，有 $xy/(x+y) \le \min\{x, y\}$。

**第 4 部分。** 我们对上表中的每一对分布检验这些界。

```{code-cell} ipython3
for (f_a, f_b), (g_a, g_b) in distribution_pairs:
    K_fg = KL_beta(f_a, f_b, g_a, g_b)
    K_gf = KL_beta(g_a, g_b, f_a, f_b)
    lower = -log_chernoff_coef_beta(0.5, f_a, f_b, g_a, g_b)
    C, _ = chernoff_beta(f_a, f_b, g_a, g_b)
    upper = K_fg * K_gf / (K_fg + K_gf)
    holds = lower <= C <= upper <= min(K_fg, K_gf)
    print(f"Beta({g_a},{g_b}): {lower:.4f} ≤ {C:.4f} ≤ {upper:.4f} "
          f"≤ {min(K_fg, K_gf):.4f}  {holds}")
```

下图针对 $f = \text{Beta}(1,1)$ 和 $g = \text{Beta}(3,1.2)$，绘制了 $-\log \rho(\phi)$ 及其两条切线。

```{code-cell} ipython3
ϕ_grid = np.linspace(0, 1, 200)
K_fg = KL_beta(F_a, F_b, G_a, G_b)
K_gf = KL_beta(G_a, G_b, F_a, F_b)
C_ex, ϕ_star = chernoff_beta(F_a, F_b, G_a, G_b)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ϕ_grid, -log_chernoff_coef_beta(ϕ_grid, F_a, F_b, G_a, G_b),
        lw=2, label=r'$-\log \rho(\phi)$')
ax.plot(ϕ_grid, ϕ_grid * K_gf, 'k--', lw=1,
        label=r'$\phi \, D_{KL}(g\|f)$')
ax.plot(ϕ_grid, (1 - ϕ_grid) * K_fg, 'k:', lw=1,
        label=r'$(1-\phi) \, D_{KL}(f\|g)$')
ax.axhline(C_ex, color='C1', lw=1, label=r'$C(f,g)$')
ax.axvline(ϕ_star, color='C1', lw=1, ls='--')
ax.set_ylim(0, 0.4)
ax.set_xlabel(r'$\phi$')
ax.legend()
plt.show()
```

```{solution-end}
```

```{exercise}
:label: div_ex3

设数据 $x_1, \ldots, x_N$ 是从以下混合分布中独立同分布抽取的样本

$$
h(x) = \frac{1}{2} f(x) + \frac{1}{2} g(x)
$$

其中 $f = \text{Beta}(1,1)$ 和 $g = \text{Beta}(3, 1.2)$ 是我们例子中的密度。

一位统计学家用最大似然法拟合 Beta$(a, b)$ 分布族。

该分布族不包含 $h$。

1. 证明伪真值 $(a^*, b^*) = \arg\min_{a,b} D_{KL}\bigl(h \,\|\, \text{Beta}(a,b)\bigr)$ 满足方程 $\psi(a) - \psi(a+b) = E_h \log X$ 和 $\psi(b) - \psi(a+b) = E_h \log (1-X)$。
2. 证明最大似然估计量 $(\hat a_N, \hat b_N)$ 满足相同的方程，只是其中的期望被替换为 $\log x_i$ 和 $\log(1-x_i)$ 的样本均值。
3. 计算 $(a^*, b^*)$，并将 $h$ 与 Beta$(a^*, b^*)$ 密度绘制在一起进行比较。
4. 模拟样本量为 $N = 10^2, 10^3, \ldots, 10^6$ 的样本，验证 $(\hat a_N, \hat b_N)$ 是否趋近于 $(a^*, b^*)$。
```

```{solution-start} div_ex3
:class: dropdown
```

**第 1、2 部分。** Beta$(a,b)$ 的对数密度为

$$
\log p_{a,b}(x) = (a-1) \log x + (b-1) \log(1-x) - \log B(a,b) .
$$

因此

$$
D_{KL}(h \| p_{a,b}) = E_h \log h(X) + Q(a, b; s_1, s_2),
\qquad
Q(a, b; s_1, s_2) = \log B(a,b) - (a-1) s_1 - (b-1) s_2 ,
$$

其中 $s_1 = E_h \log X$，$s_2 = E_h \log(1-X)$。

平均负对数似然为 $-\ell_N(a,b)/N = Q(a, b; \hat s_1, \hat s_2)$，其中 $\hat s_1$ 和 $\hat s_2$ 分别是 $\log x_i$ 和 $\log(1-x_i)$ 的样本均值。

函数 $\log B(a,b)$ 是凸函数，因此 $Q$ 关于 $(a, b)$ 也是凸函数。

根据 {ref}`div_ex1` 第 1 部分，$\partial \log B(a,b)/\partial a = \psi(a) - \psi(a+b)$，$\partial \log B(a,b)/\partial b = \psi(b) - \psi(a+b)$。

令 $Q$ 的梯度为零，即可得到题目所述的方程，其中伪真值对应 $(s_1, s_2)$，最大似然估计量对应 $(\hat s_1, \hat s_2)$。

**第 3 部分。** 根据 {ref}`div_ex1` 第 1 部分及期望的线性性，可得 $s_1$ 和 $s_2$ 的闭式解。

我们在 $(\log a, \log b)$ 上对 $Q$ 进行最小化，以保证 $a$ 和 $b$ 为正。

```{code-cell} ipython3
from scipy.optimize import minimize

def Q(z, s1, s2):
    a, b = np.exp(z)
    return betaln(a, b) - (a - 1) * s1 - (b - 1) * s2

def fit_beta(s1, s2):
    res = minimize(Q, x0=np.zeros(2), args=(s1, s2), method='BFGS')
    return np.exp(res.x)

E_log_x = lambda a, b: digamma(a) - digamma(a + b)
s1 = 0.5 * E_log_x(F_a, F_b) + 0.5 * E_log_x(G_a, G_b)
s2 = 0.5 * E_log_x(F_b, F_a) + 0.5 * E_log_x(G_b, G_a)

a_star, b_star = fit_beta(s1, s2)
print(f"pseudo-true value: a* = {a_star:.4f}, b* = {b_star:.4f}")

h = lambda x: 0.5 * p(x, F_a, F_b) + 0.5 * p(x, G_a, G_b)
p_star = lambda x: p(x, a_star, b_star)
print(f"KL(h, Beta(a*, b*)) = {compute_KL(h, p_star):.4f}")

x_grid = np.linspace(0.001, 0.999, 500)
plt.figure(figsize=(8, 5))
plt.plot(x_grid, h(x_grid), lw=2, label='$h$ (data)')
plt.plot(x_grid, p_star(x_grid), lw=2, ls='--',
         label=f'Beta({a_star:.3f}, {b_star:.3f})')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

**第 4 部分。** 我们从 $h$ 中抽取样本，并根据样本矩计算 $(\hat a_N, \hat b_N)$。

```{code-cell} ipython3
rng = np.random.default_rng(1234)

for N in [10**2, 10**3, 10**4, 10**5, 10**6]:
    from_f = rng.random(N) < 0.5
    x = np.where(from_f, rng.beta(F_a, F_b, N), rng.beta(G_a, G_b, N))
    a_hat, b_hat = fit_beta(np.mean(np.log(x)), np.mean(np.log1p(-x)))
    print(f"N = {N:>7}: a_hat = {a_hat:.4f}, b_hat = {b_hat:.4f}")
```

这些估计值逐渐趋近于 $(a^*, b^*)$，即贝塔分布族中在 KL 散度意义下最接近 $h$ 的成员。

{doc}`mix_model` 研究了一位贝叶斯主义者，其模型只包含 $f$ 和 $g$，却观测到来自 $h$ 的数据；从 $h$ 到 $f$ 和到 $g$ 的 KL 散度决定了该贝叶斯主义者信念的最终落点。

{doc}`gallant_tauchen` 将辅助模型的伪真值用作某个估计量的目标。

```{solution-end}
```
