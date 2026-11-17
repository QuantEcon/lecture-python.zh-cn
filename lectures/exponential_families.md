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
  title: 指数族与最大熵
  headings:
    Overview: 概述
    Coordinates matter: 坐标很重要
    Maximum entropy: 最大熵
    Maximum entropy::A die: 一个骰子
    Maximum entropy::The solution is an exponential family: 解是一个指数族
    Maximum entropy::The normal distribution: 正态分布
    Maximum entropy::Other moment conditions: 其他矩条件
    The exponential family: 指数族
    The exponential family::Definition: 定义
    The exponential family::The log-partition function generates moments: 对数配分函数生成矩
    Two coordinate systems: 两套坐标系统
    Two coordinate systems::Expectation parameters: 期望参数
    Two coordinate systems::Legendre duality: 勒让德对偶性
    Two coordinate systems::The conjugate is negative entropy: 共轭函数是负熵
    Two coordinate systems::Logit choice probabilities: 逻辑选择概率
    Bregman divergences and KL divergence: 布雷格曼散度与 KL 散度
    Bregman divergences and KL divergence::Bregman divergence: 布雷格曼散度
    Bregman divergences and KL divergence::KL divergence is a Bregman divergence: KL 散度是一种布雷格曼散度
    Bregman divergences and KL divergence::Fisher information is the curvature of KL divergence: 费舍尔信息是 KL 散度的曲率
    Estimation: 估计
    Estimation::Maximum likelihood matches moments: 最大似然匹配矩
    Estimation::Finite-sample behavior of the two estimators: 两种估计量的有限样本行为
    Concluding remarks: 结束语
    Exercises: 练习
---

(exponential_families)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 指数族与最大熵

```{contents} Contents
:depth: 2
```

## 概述

本讲座及其后续讲座 {doc}`information_projections` 介绍**信息几何**的一些内容，这是一门使用凸分析和微分几何的工具来研究概率分布族的学科。

我们使用信息几何来描述概率分布的**指数族**（exponential family）。

指数族包括伯努利分布、泊松分布、正态分布、伽马分布和贝塔分布。

一个凸函数，即**对数配分函数**（log-partition function），生成了

* 该族成员的两套坐标系统，称为**自然参数**（natural parameters）和**期望参数**（expectation parameters）；
* 该族充分统计量的均值和协方差；
* 费舍尔信息矩阵；
* 当该族的基测度为常数时，通过勒让德变换得到每个成员的熵；以及
* 任意两个成员之间的 Kullback–Leibler（KL）散度，通过**布雷格曼散度**（Bregman divergence）得到。

指数族的成员回答了 {cite:t}`Jaynes1957` 提出的一个问题：在给定某些统计量均值的所有分布中，哪个概率分布按照香农熵衡量表达的信息最少？

有一个特例刻画了正态分布：在实数线上所有具有给定均值和方差的密度中，正态密度具有最大的熵。

经济学家其实已经在以其他名称使用这些工具中的一些了。

对数配分函数及其**勒让德变换**（Legendre transform）之间的关系，与利润函数和生产函数之间的关系相同。

对数配分函数导数的公式是霍特林引理（Hotelling's lemma）的一个版本。

离散选择的多项逻辑模型的**包含值**（inclusive value）就是一个对数配分函数。

它在相差一个常数的意义上，是最佳选择的期望效用。

它的勒让德变换是熵的负数。

后续讲座 {doc}`information_projections` 使用这些工具来研究概率分布在分布族上的投影，并应用于最大似然、鲁棒控制、投入产出表的更新以及最优传输。

本讲座建立在 {doc}`divergence_measures` 的基础上，该讲座描述了熵、交叉熵和 KL 散度；我们沿用其记号。

{doc}`mle` 和 {doc}`multivariate_normal` 提供了额外的背景知识。

{cite:t}`amari_nagaoka2000`、{cite:t}`Nielsen2020` 以及 {cite:t}`CoverThomas2006` 的第 2、8、11 和 12 章讨论了本讲座及其后续讲座的内容。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import logsumexp, digamma, polygamma, gammaln
from scipy.stats import norm, poisson, laplace, logistic, uniform, chi2
from scipy.stats import gamma as gamma_dist
```

## 坐标很重要

我们通常用均值 $\mu$ 和标准差 $\sigma$ 来描述正态分布。

数对 $(\mu, \sigma)$ 是半平面中的一个点，但这两点之间的普通距离几乎无法说明从数据中区分这两个分布有多难。

考虑四个正态分布：

* $A = N(0, 0.2^2)$ 和 $B = N(1, 0.2^2)$，以及
* $C = N(0, 2^2)$ 和 $D = N(1, 2^2)$。

在 $(\mu, \sigma)$ 半平面中，$A$ 与 $B$ 之间的距离等于 $C$ 与 $D$ 之间的距离。

但 $A$ 和 $B$ 几乎不重叠，而 $C$ 和 $D$ 大量重叠。

两个标准差相同为 $\sigma$ 的正态分布之间的 KL 散度为 $(\mu_1 - \mu_2)^2 / (2 \sigma^2)$，对 $A$ 和 $B$ 而言为 $12.5$，对 $C$ 和 $D$ 而言为 $0.125$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Equal parameter distances, unequal KL divergences
    name: fig-ef-four-normals
---
x = np.linspace(-6, 7, 1000)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, σ, names in zip(axes, (0.2, 2.0), (("A", "B"), ("C", "D"))):
    for μ, name in zip((0.0, 1.0), names):
        ax.plot(x, norm.pdf(x, μ, σ), lw=2, label=f"{name}: N({μ:g}, {σ:g}²)")
    ax.set_title(rf"$\sigma = {σ:g}$, KL = {1 / (2 * σ**2):.3g}")
    ax.set_xlabel("$x$")
    ax.legend()
plt.tight_layout()
plt.show()
```

概率分布的几何学应当通过分布之间的可区分程度来度量距离。

对于指数族来说，一个凸函数就能生成这样一种几何结构。

## 最大熵

### 一个骰子

埃德温·杰恩斯（Edwin Jaynes）提出了最大熵原理 {cite}`Jaynes1957`，并在他 1962 年布兰代斯讲座中用下面这个问题进行说明，该问题现在被称为布兰代斯骰子问题 {cite}`Jaynes1963`。

一枚骰子的面为 $x \in \{1, 2, \ldots, 6\}$。

我们只被告知点数的平均值为 $4.5$，而不是公平骰子的 $3.5$。

我们应该为各面分配怎样的概率分布 $p = (p_1, \ldots, p_6)$？

许多分布的均值都是 $4.5$。

杰恩斯建议选择在"关于缺失信息方面最不偏不倚"的那个分布 {cite}`Jaynes1957`，他认为这意味着选择能使香农熵最大化的分布

$$
H(p) = - \sum_{x} p(x) \log p(x)
$$

约束条件为 $\sum_x p(x) = 1$ 和 $\sum_x x \, p(x) = 4.5$。

### 解是一个指数族

让我们求解这个问题的一个更一般的版本。

设 $t(x) = (t_1(x), \ldots, t_d(x))$ 为结果 $x$（取值于某个有限集合）的一组函数构成的向量。

我们寻求在矩条件

$$
\sum_x p(x) \, t(x) = c
$$ (ef_moments)

（对于给定的向量 $c$）以及 $\sum_x p(x) = 1$ 的约束下使 $H(p)$ 最大化的分布 $p$。

对 {eq}`ef_moments` 附加一个拉格朗日乘子向量 $\theta$。

$p(x)$ 的一阶条件为 $-\log p(x) - 1 + \theta \cdot t(x) - \nu = 0$，其中 $\nu$ 是 $\sum_x p(x) = 1$ 的乘子。

解的形式为

$$
p(x; \theta) = \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \sum_x \exp\left( \theta \cdot t(x) \right) ,
$$ (ef_maxent_solution)

其中 $\theta$ 满足 {eq}`ef_moments`。

函数 $F$ 使得概率之和为一。

这样的 $\theta$ 存在当且仅当 $c$ 位于各点 $t(x)$ 凸包的相对内部，即位于包含该凸包的最小仿射子空间意义下的内部；对于下面的骰子而言，当且仅当 $1 < c < 6$。

当 $c$ 位于相对边界上时，例如一枚均值为 $6$ 的骰子，极大化分布仍然存在。

它对某些结果赋予零概率，并且位于该族之外，是其成员在 $\theta$ 发散时的极限。

形如 {eq}`ef_maxent_solution` 的分布族称为**指数族**（exponential family）。

拉格朗日乘子 $\theta$ 是其**自然参数**（natural parameters），$t$ 是其**充分统计量**（sufficient statistic）。

为了确认 {eq}`ef_maxent_solution` 是一个最大值，设 $p$ 是任意满足 {eq}`ef_moments` 的分布，设 $p^*$ 是满足该条件的指数族成员。

因为 $\log p^*(x) = \theta \cdot t(x) - F(\theta)$ 是 $t(x)$ 的仿射函数，而 $p$ 和 $p^*$ 赋予 $t$ 相同的均值 $c$，所以

$$
\sum_x p(x) \log p^*(x) = \theta \cdot c - F(\theta) = \sum_x p^*(x) \log p^*(x) = - H(p^*) .
$$

我们将 $D_{KL}(p \| q) = \sum_x p(x) \log (p(x) / q(x))$ 读作"$p$ 相对于 $q$ 的 KL 散度"。

因此

$$
H(p^*) - H(p) = \sum_x p(x) \log p(x) - \sum_x p(x) \log p^*(x) = D_{KL}(p \| p^*) \geq 0 ,
$$ (ef_entropy_gap)

等号仅在 $p = p^*$ 时成立。

在所有满足矩条件的分布中，指数族成员具有最大的熵，而任何其他分布的熵亏损恰好等于它相对于该成员的 KL 散度。

后续讲座 {doc}`information_projections` 表明 {eq}`ef_entropy_gap` 是一个毕达哥拉斯定理的实例。

对于杰恩斯的骰子，$t(x) = x$，$\theta$ 是标量。

下面的函数 `tilt` 计算**基分布**（base distribution）$q$ 的**指数倾斜**（exponential tilt）：即对于统计量 $t$，正比于 $q(x) \exp(\theta \, t(x))$ 的分布。

对于最大熵问题，$q$ 是均匀分布，其结果即为 {eq}`ef_maxent_solution`。

第二个函数寻找能产生给定均值的 $\theta$。

```{code-cell} ipython3
def tilt(q, t, θ):
    """The distribution proportional to q(x) exp(θ t(x))."""
    log_w = np.log(q) + θ * t
    return np.exp(log_w - logsumexp(log_w))


def match_mean(q, t, c):
    """Find θ such that the tilted distribution has mean c."""
    θ = brentq(lambda θ: tilt(q, t, θ) @ t - c, -50, 50)
    return θ, tilt(q, t, θ)


def entropy(p):
    return -p @ np.log(p)


def kl(p, q):
    return p @ np.log(p / q)


faces = np.arange(1, 7)
uniform_die = np.full(6, 1 / 6)
θ_die, p_die = match_mean(uniform_die, faces, 4.5)
print(f"θ = {θ_die:.4f}")
print("maximum entropy distribution:", np.round(p_die, 4))
print(f"entropy = {entropy(p_die):.4f}, compared with log 6 = {np.log(6):.4f}")
```

概率按因子 $e^{\theta}$ 从一面到下一面几何式地增长。

让我们通过绘制其他均值为 $4.5$ 的分布来检验 {eq}`ef_entropy_gap`。

我们通过沿着同时保持总概率和均值不变的方向偏离 $p^*$ 来构造它们。

```{code-cell} ipython3
rng = np.random.default_rng(0)

# directions orthogonal to (1, ..., 1) and to (1, 2, ..., 6)
constraints = np.vstack([np.ones(6), faces])
null_space = np.linalg.svd(constraints)[2][2:].T

print(" H(p*) - H(p)    KL(p || p*)")
for _ in range(5):
    while True:
        p = p_die + null_space @ rng.normal(scale=0.05, size=4)
        if np.all(p > 0):
            break
    print(f"{entropy(p_die) - entropy(p):12.6f} {kl(p, p_die):14.6f}")
```

下图展示了几个均值下的最大熵分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum entropy die distributions by mean
    name: fig-ef-die
---
fig, ax = plt.subplots()
for c in (2.5, 3.5, 4.5, 5.5):
    θ, p = match_mean(uniform_die, faces, c)
    ax.plot(faces, p, "o-", label=rf"mean {c}, $\theta$ = {θ:.2f}")
ax.set_xlabel("face")
ax.set_ylabel("probability")
ax.legend()
plt.show()
```

### 正态分布

现在设结果 $x$ 为一个实数，并约束其均值与二阶矩：

$$
E[X] = \mu, \qquad E[X^2] = \mu^2 + \sigma^2 .
$$

对于实数线上的密度 $p$，**微分熵**（differential entropy）为 $H(p) = -\int p(x) \log p(x) \, dx$。

导出 {eq}`ef_maxent_solution` 的论证现在表明，具有最大熵的密度形式为

$$
p(x; \theta) = \exp\left( \theta_1 x + \theta_2 x^2 - F(\theta) \right) ,
$$

该式只有在 $\theta_2 < 0$ 时才可积。

配方表明，这就是正态密度，其中 $\theta_1 = \mu / \sigma^2$ 且 $\theta_2 = -1/(2\sigma^2)$。

导出 {eq}`ef_entropy_gap` 的论证在将求和替换为积分后仍然成立，因此每个具有均值 $\mu$ 和方差 $\sigma^2$ 的密度 $p$ 都满足

$$
H(\varphi) - H(p) = D_{KL}(p \| \varphi) \geq 0 ,
$$ (ef_normal_gap)

其中 $\varphi$ 是 $N(\mu, \sigma^2)$ 密度。

*在实数线上所有具有给定均值和方差的密度中，正态密度具有最大的熵。*

它的熵是 $H(\varphi) = \frac{1}{2} \log(2 \pi e \sigma^2)$。

与离散分布的熵不同，微分熵可以为负，并且当我们改变衡量 $x$ 的单位时它也会变化；而差距 {eq}`ef_normal_gap` 是一个 KL 散度，不会出现这两种情况。

我们对几个均值为零、方差为一的密度检验 {eq}`ef_normal_gap`。

我们使用 SciPy 的公式计算熵，并通过数值积分计算 KL 散度。

```{code-cell} ipython3
φ = norm(0, 1)
alternatives = {
    "uniform": uniform(-np.sqrt(3), 2 * np.sqrt(3)),
    "Laplace": laplace(0, 1 / np.sqrt(2)),
    "logistic": logistic(0, np.sqrt(3) / np.pi),
}

print(f"entropy of N(0, 1): {φ.entropy():.4f}\n")
print(f"{'distribution':12} {'variance':>8} {'entropy':>8} "
      f"{'H(φ) - H(p)':>12} {'KL(p || φ)':>11}")
for name, dist in alternatives.items():
    a, b = dist.support()
    a, b = max(a, -30), min(b, 30)
    kl_p_φ = integrate.quad(
        lambda x: dist.pdf(x) * (dist.logpdf(x) - φ.logpdf(x)), a, b,
        limit=200)[0]
    print(f"{name:12} {dist.var():8.4f} {dist.entropy():8.4f} "
          f"{φ.entropy() - dist.entropy():12.4f} {kl_p_φ:11.4f}")
```

我们还可以观察正态分布如何从一个离散的最大熵问题中浮现出来。

将结果放到一个细密的点网格上，并施加两个矩条件。

下面的函数通过牛顿法求出自然参数，使用我们下面证明的两个事实：$F$ 的梯度是 $t$ 的均值向量，$F$ 的 Hessian 矩阵是 $t$ 的协方差矩阵。

```{code-cell} ipython3
def maxent(q, T, c, tol=1e-12, max_iter=100):
    """
    Maximize entropy relative to q subject to E[T] = c.

    T is an m × d array whose rows are t(x).  Returns θ and the
    distribution proportional to q(x) exp(θ · t(x)).  The iterations
    converge only if c is in the interior of the convex hull of the
    rows of T.
    """
    def tilted(θ):
        log_w = np.log(q) + T @ θ
        return np.exp(log_w - logsumexp(log_w))

    θ = np.zeros(T.shape[1])
    for _ in range(max_iter):
        p = tilted(θ)
        mean = p @ T
        cov = (T - mean).T @ ((T - mean) * p[:, None])
        step = np.linalg.solve(cov, c - mean)
        θ = θ + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("no convergence: is c inside the hull of t(x)?")
    return θ, tilted(θ)


μ_grid, σ_grid = 1.0, 1.2
grid = np.linspace(-6, 6, 241)
spacing = grid[1] - grid[0]
T_grid = np.column_stack([grid, grid**2])
θ_grid, p_grid = maxent(np.full(grid.size, 1 / grid.size), T_grid,
                        c=np.array([μ_grid, μ_grid**2 + σ_grid**2]))
print(f"θ = {np.round(θ_grid, 4)}")
print(f"(μ/σ², -1/(2σ²)) = "
      f"{np.round([μ_grid / σ_grid**2, -1 / (2 * σ_grid**2)], 4)}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Grid maximum entropy matches the normal
    name: fig-ef-grid-normal
---
fig, ax = plt.subplots()
ax.plot(grid, p_grid / spacing, lw=4, alpha=0.4,
        label="maximum entropy on a grid")
ax.plot(grid, norm.pdf(grid, μ_grid, σ_grid), "k--", lw=1,
        label=f"N({μ_grid:g}, {σ_grid:g}²)")
ax.set_xlabel("$x$")
ax.legend(loc="upper right")
plt.show()
```

### 其他矩条件

不同的矩条件产生不同的族。

| 结果 | 约束的均值 | 最大熵分布 |
|---|---|---|
| $\{1, \ldots, m\}$ | 无 | 均匀分布 |
| $\{0, 1, 2, \ldots\}$ | $E[X]$ | 几何分布 |
| $[a, b]$ | 无 | 均匀分布 |
| $[0, \infty)$ | $E[X]$ | 指数分布 |
| $(0, \infty)$ | $E[X]$, $E[\log X]$ | 伽马分布 |
| $(0, \infty)$ | $E[\log X]$, $E[(\log X)^2]$ | 对数正态分布 |
| $(0, 1)$ | $E[\log X]$, $E[\log(1 - X)]$ | 贝塔分布 |
| $\mathbb{R}$ | $E[\lvert X \rvert]$ | 拉普拉斯分布 |
| $\mathbb{R}$ | $E[X]$, $E[X^2]$ | 正态分布 |
| $\mathbb{R}^m$ | $E[X]$, $E[X X^\top]$ | 多元正态分布 |

最后一行意味着，在 $\mathbb{R}^m$ 上所有具有给定均值向量和协方差矩阵 $\Sigma$ 的密度中，多元正态密度具有最大的熵 $\frac{1}{2} \log \det(2 \pi e \Sigma)$。

对数正态行是将正态行应用于 $\log X$ 的结果：$(0, \infty)$ 上正比于 $\exp(\theta_1 \log x + \theta_2 (\log x)^2)$ 的密度是对数正态密度。

{ref}`ef_ex_gamma` 要求你推导伽马分布的情形。

## 指数族

### 定义

如果一族密度（或概率质量函数）可以写成

$$
p(x; \theta) = h(x) \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \int h(x) \exp\left( \theta \cdot t(x) \right) dx ,
$$ (ef_family)

（当 $x$ 为离散时，积分替换为求和），则该族被称为**指数族**（exponential family），其中

* $t(x) \in \mathbb{R}^d$ 是**充分统计量**（sufficient statistic），
* $\theta \in \mathbb{R}^d$ 是**自然参数**（natural parameter），
* $h(x) \geq 0$ 是**基测度**（base measure），
* $F(\theta)$ 是**对数配分函数**（log-partition function），也称累积量函数。

自然参数取值范围为使得定义 $F$ 的积分有限的 $\theta$ 的集合 $\Theta$。

在上面的最大熵问题中，$h$ 为常数；在函数 `tilt` 中，基分布 $q$ 即是基测度 $h$。

当我们相对于非均匀的基分布最大化熵时，会出现非常数的 $h$，这个问题将在 {doc}`information_projections` 中讨论。

我们假设该表示是**极小的**（minimal）：在 $h(x) > 0$ 的集合上，任何非平凡的线性组合 $a \cdot t(x)$ 都不是常数。

这样不同的 $\theta$ 对应不同的分布。

我们还假设 $\Theta$ 是一个开集，这在本讲座中的每一个族都成立。

下面是用这种方式写出的一些常见的族。

| 族 | $t(x)$ | $\theta$ | $F(\theta)$ |
|---|---|---|---|
| 伯努利($\pi$) | $x$ | $\log \frac{\pi}{1 - \pi}$ | $\log(1 + e^{\theta})$ |
| 泊松($\lambda$) | $x$ | $\log \lambda$ | $e^{\theta}$ |
| 指数($\lambda$) | $x$ | $-\lambda$ | $-\log(-\theta)$ |
| $N(\mu, \sigma^2)$ | $(x, x^2)$ | $\left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)$ | $-\frac{\theta_1^2}{4 \theta_2} + \frac{1}{2}\log\left(-\frac{\pi}{\theta_2}\right)$ |
| 贝塔($a$, $b$) | $(\log x, \log(1 - x))$ | $(a - 1, b - 1)$ | $\log B(\theta_1 + 1, \theta_2 + 1)$ |
| 分类($\pi$) | $(1_{x = 1}, \ldots, 1_{x = K - 1})$ | $\left(\log \frac{\pi_k}{\pi_K}\right)_{k < K}$ | $\log\left(1 + \sum_{k < K} e^{\theta_k}\right)$ |

每一行中 $h(x) = 1$，泊松行除外，该行中 $h(x) = 1/x!$。

在伯努利和分类行中，$\pi$ 是一个概率或概率向量；在正态行中，它是 $3.14159\ldots$。

在贝塔行中，$B$ 是贝塔函数。

$\{1, \ldots, K\}$ 上的分类族只使用 $K - 1$ 个指示函数，因为全部 $K$ 个指示函数之和为一；若使用全部 $K$ 个，该表示将不是极小的。

多元正态分布 $N(\mu, \Sigma)$ 是一个指数族，其 $t(x) = (x, x x^\top)$，自然参数为 $(\Sigma^{-1}\mu, -\frac{1}{2}\Sigma^{-1})$，其中配对为 $\theta \cdot t(x) = \theta_1 \cdot x + x^\top \Theta_2 x$，$\Theta_2$ 为对称矩阵。

因为 $x_i x_j = x_j x_i$，极小表示只保留 $x x^\top$ 对角线及其上方的元素。

形如 {eq}`ef_family` 的族包含了其任意两个成员的归一化几何混合：正比于 $p(x; \theta_0)^{1 - a} p(x; \theta_1)^a$ 的分布具有自然参数 $(1 - a)\theta_0 + a \theta_1$，对任意 $a \in [0, 1]$ 以及任何使该参数位于 $\Theta$ 中的其他 $a$ 均成立。

后续讲座 {doc}`information_projections` 基于这一事实构建了一种几何结构。

对数配分函数决定了其余的一切。

### 对数配分函数生成矩

对 $F(\theta) = \log \int h(x) \exp(\theta \cdot t(x)) \, dx$ 求导：

$$
\nabla F(\theta) = \frac{\int t(x) h(x) \exp(\theta \cdot t(x)) \, dx}{\int h(x) \exp(\theta \cdot t(x)) \, dx} = E_\theta [t(X)] .
$$ (ef_gradient)

再次求导得到

$$
\nabla^2 F(\theta) = E_\theta\left[ t(X) t(X)^\top \right] - E_\theta[t(X)] \, E_\theta[t(X)]^\top = \operatorname{Cov}_\theta [t(X)] .
$$ (ef_hessian)

函数 $F$ 是凸的，因为协方差矩阵是半正定的；极小性使 $\operatorname{Cov}_\theta[t(X)]$ 为正定矩阵，所以 $F$ 是严格凸的。

更一般地，$\log E_\theta \exp(s \cdot t(X)) = F(\theta + s) - F(\theta)$，所以 $F$ 的导数就是 $t(X)$ 的累积量。

让我们对泊松族和正态族验证 {eq}`ef_gradient` 和 {eq}`ef_hessian`，将 $F$ 的有限差分导数与通过求和及积分计算得到的矩进行比较。

```{code-cell} ipython3
def F_poisson(θ):
    return np.exp(θ)


def F_normal(θ):
    θ1, θ2 = θ
    return -θ1**2 / (4 * θ2) + 0.5 * np.log(-np.pi / θ2)


def normal_θ(μ, σ):
    return np.array([μ / σ**2, -1 / (2 * σ**2)])


def gradient(F, θ, h=1e-5):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([np.squeeze(F(θ + e[i]) - F(θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


def hessian(F, θ, h=1e-4):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([(gradient(F, θ + e[i]) - gradient(F, θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


λ = 3.0
k = np.arange(100)
pk = poisson.pmf(k, λ)
print("Poisson:  F'(θ), F''(θ) =",
      np.round([gradient(F_poisson, np.log(λ))[0],
                hessian(F_poisson, np.log(λ))[0, 0]], 6),
      "  mean, variance =", np.round([pk @ k, pk @ k**2 - (pk @ k)**2], 6))

μ, σ = 0.5, 1.2
moments = [integrate.quad(lambda x: x**j * norm.pdf(x, μ, σ), -15, 15)[0]
           for j in range(1, 5)]
mean_t = np.array(moments[:2])
cov_t = np.array([[moments[1] - moments[0]**2,
                   moments[2] - moments[0] * moments[1]],
                  [moments[2] - moments[0] * moments[1],
                   moments[3] - moments[1]**2]])
print("\nnormal:  ∇F(θ) =", np.round(gradient(F_normal, normal_θ(μ, σ)), 6),
      "  E[t] =", np.round(mean_t, 6))
print("∇²F(θ) =\n", np.round(hessian(F_normal, normal_θ(μ, σ)), 4))
print("Cov[t] =\n", np.round(cov_t, 4))
```

## 两套坐标系统

### 期望参数

由 {eq}`ef_gradient` 可知，映射 $\theta \mapsto \eta = \nabla F(\theta) = E_\theta[t(X)]$ 将每个自然参数映射到充分统计量均值的向量。

由于 $F$ 是严格凸的，这个映射是一一对应的。

对于本讲座中的各族而言，它的值域是 $t(X)$ 可能取值的凸包的内部：伯努利分布的均值严格位于 $0$ 和 $1$ 之间，泊松分布的均值为正，依此类推。

我们可以用自然参数 $\theta$ 或者**期望参数**（expectation parameter）$\eta$ 来标记该族的每个成员。

对正态族，$\eta = (\mu, \mu^2 + \sigma^2)$；对泊松族，$\eta = \lambda$；对伯努利族，$\eta = \pi$。

最大熵问题固定了期望参数 $\eta = c$；它的解给出了自然参数 $\theta$。

### 勒让德对偶性

从 $\eta$ 回到 $\theta$ 的映射是另一个凸函数的梯度。

$F$ 的**凸共轭**（convex conjugate），即勒让德变换，为

$$
F^*(\eta) = \sup_\theta \left\{ \theta \cdot \eta - F(\theta) \right\} .
$$ (ef_conjugate)

对于 $\nabla F$ 值域中的 $\eta$，上确界在 $\eta = \nabla F(\theta)$ 处取得，所以

$$
F(\theta) + F^*(\eta) = \theta \cdot \eta
\quad \text{当 } \eta = \nabla F(\theta) \text{ 时},
$$ (ef_fenchel)

并且，根据包络定理，$\nabla F^*(\eta) = \theta$。

两个梯度映射 $\nabla F$ 和 $\nabla F^*$ 互为反函数，它们的雅可比矩阵互为逆矩阵：

$$
\nabla^2 F^*(\eta) = \left[ \nabla^2 F(\theta) \right]^{-1} .
$$

因为 $F$ 是凸的，两次应用该变换即可还原 $F$：

$$
F(\theta) = \max_\eta \left\{ \theta \cdot \eta - F^*(\eta) \right\} .
$$

经济学家在利润函数和生产函数之间使用相同的对偶性。

在 {doc}`houthakker_aggregation` 中，行业利润函数为 $\Pi(p) = \max_X \{ f(X) - p \cdot X \}$，其中 $X$ 是投入品向量，$p$ 是它们价格的向量，$f$ 是总生产函数（该讲座将其称为 $F$）。

令 $\theta = -p$，得到 $\Pi(-\theta) = \max_X \{ \theta \cdot X - (-f(X)) \}$，其形式与上一个式子相同。

对应关系如下

* 对数配分函数 $F$ $\leftrightarrow$ 利润函数 $\Pi$，
* 自然参数 $\theta$ $\leftrightarrow$ 负的投入价格 $-p$，
* 期望参数 $\eta$ $\leftrightarrow$ 投入数量 $X$，以及
* 共轭函数 $F^*$ $\leftrightarrow$ 负的生产函数 $-f$。

霍特林引理 $\partial \Pi / \partial p_i = -X_i$ 对应着 $\nabla F(\theta) = \eta$，而从利润函数恢复生产函数则对应着从 $F$ 恢复 $F^*$。

### 共轭函数是负熵

在从 $p(\cdot; \theta)$ 抽取的样本 $X$ 处计算 $\log p(x; \theta) = \log h(x) + \theta \cdot t(x) - F(\theta)$ 并取期望。

利用 {eq}`ef_fenchel`，

$$
E_\theta \log p(X; \theta) = E_\theta \log h(X) + \theta \cdot \eta - F(\theta) = E_\theta \log h(X) + F^*(\eta) .
$$

因此，当 $p(x; \theta) > 0$ 处均有 $h(x) = 1$ 时，

$$
F^*(\eta) = - H(p(\cdot; \theta)) ,
$$ (ef_negentropy)

*对数配分函数的共轭函数是熵的负数（以期望参数表示）。*

一般地，$H(p(\cdot; \theta)) = -F^*(\eta) - E_\theta \log h(X)$。

{ref}`ef_ex_poisson` 推导了泊松的情形，其中 $h(x) = 1/x!$。

对于伯努利族，$h = 1$，$F(\theta) = \log(1 + e^\theta)$，而 $F^*(\eta) = \eta \log \eta + (1 - \eta) \log (1 - \eta)$。

在下图的左侧面板中，与 $F$ 相切、斜率为 $\eta$ 的切线在纵轴上交于 $-F^*(\eta)$。

在右侧面板中，$F^*$ 在 $\eta$ 处的切线斜率为 $\theta$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bernoulli log-partition function and its conjugate
    name: fig-ef-legendre
---
def F_bern(θ):
    return np.log1p(np.exp(θ))


def F_star_bern(η):
    return η * np.log(η) + (1 - η) * np.log(1 - η)


θ0 = 2.0
η0 = 1 / (1 + np.exp(-θ0))

θ_vals = np.linspace(-4, 4, 200)
η_vals = np.linspace(0.001, 0.999, 200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(θ_vals, F_bern(θ_vals), lw=2, label=r"$F(\theta)$")
ax.plot(θ_vals, -F_star_bern(η0) + η0 * θ_vals, "--",
        label=rf"tangent line with slope $\eta$ = {η0:.2f}")
ax.plot([0], [-F_star_bern(η0)], "ko")
ax.annotate(r"$-F^*(\eta)$", (0, -F_star_bern(η0)),
            xytext=(8, -14), textcoords="offset points")
ax.plot([θ0], [F_bern(θ0)], "o", color="C1")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_xlabel(r"$\theta$")
ax.legend()

ax = axes[1]
ax.plot(η_vals, F_star_bern(η_vals), lw=2, label=r"$F^*(\eta) = -H$")
ax.plot(η_vals, F_star_bern(η0) + θ0 * (η_vals - η0), "--",
        label=rf"tangent line with slope $\theta$ = {θ0:.2f}")
ax.plot([η0], [F_star_bern(η0)], "o", color="C1")
ax.set_xlabel(r"$\eta$")
ax.legend()
plt.tight_layout()
plt.show()
```

*最大熵问题的拉格朗日乘子是矩约束的影子价格。*

由 {eq}`ef_negentropy`，最大化的熵为 $H(p^*) = -F^*(c)$，所以由包络定理 $\partial H(p^*) / \partial c = -\nabla F^*(c) = -\theta$。

乘子 $\theta_j$ 是约束均值 $c_j$ 上升时熵下降的速率。

对于杰恩斯的骰子，当 $c = 4.5$ 时 $\theta > 0$，因为将均值进一步推离 $3.5$ 需要一个更不均匀的分布。

### 逻辑选择概率

对数配分函数与熵之间的对偶性，对于使用离散选择的多项逻辑模型的经济学家来说是熟悉的。

一个决策者从 $K$ 个备选方案中选择一个，各方案的效用为 $v_1, \ldots, v_K$，加上尺度为 $\tau$ 的独立 I 型极值（耿贝尔）扰动。

各选择的概率为

$$
\pi_k = \frac{\exp(v_k / \tau)}{\sum_j \exp(v_j / \tau)} ,
$$

期望的最大化效用为 $\tau \log \sum_j \exp(v_j / \tau) + \tau \gamma$，其中 $\gamma \approx 0.5772$ 是欧拉常数。

去掉常数 $\tau \gamma$ 剩下**包含值**（inclusive value）$\tau \log \sum_j \exp(v_j / \tau)$。

包含值是 $\tau$ 乘以在 $\theta = v / \tau$ 处计算的 $\log \sum_j e^{\theta_j}$。

函数 $\theta \mapsto \log \sum_j e^{\theta_j}$ 是以全部 $K$ 个指示函数 $1_{x = k}$ 作为充分统计量的分类族的对数配分函数。

该表示不是极小的，因为指示函数之和为一。

给每个 $v_k$ 加上相同的常数不会改变选择概率。

式 {eq}`ef_gradient` 仍然成立，因此包含值关于 $v$ 的梯度是选择概率的向量。

离散选择理论家称这一事实为威廉姆斯-戴利-扎卡里定理（Williams–Daly–Zachary theorem）。

它适用于任何可加随机效用模型；逻辑模型是具有耿贝尔扰动的特例。

因为 $\theta \mapsto \log \sum_j e^{\theta_j}$ 的共轭函数是概率单纯形上的负熵，所以

$$
\tau \log \sum_j \exp(v_j / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k + \tau H(\pi) \right\} ,
$$ (ef_logit)

其中最大值是对概率向量 $\pi$ 取的，并在逻辑概率处取得。

{cite:t}`AndersonDePalmaThisse1988` 将 {eq}`ef_logit` 解释为一个代表性消费者的间接效用，其消费份额即为逻辑概率。

更一般地，对任意满足 $q_k > 0$（对所有 $k$）的基分布 $q$，

$$
\tau \log \sum_k q_k \exp(v_k / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) \right\} ,
$$ (ef_logit_base)

最大值在 $\pi^*_k \propto q_k \exp(v_k / \tau)$ 处取得。

为了验证 {eq}`ef_logit_base`，展开 $D_{KL}(\pi \| \pi^*)$ 可发现，对于任意概率向量 $\pi$，

$$
\sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) = \tau \log \sum_k q_k \exp(v_k / \tau) - \tau D_{KL}(\pi \| \pi^*) ,
$$

最后一项仅在 $\pi = \pi^*$ 时为零。

当 $q$ 为均匀分布时，$D_{KL}(\pi \| q) = \log K - H(\pi)$，{eq}`ef_logit_base` 即为 {eq}`ef_logit` 减去常数 $\tau \log K$。

{cite:t}`MattssonWeibull2002` 将 {eq}`ef_logit_base` 解释为一个决策者在备选方案上选择概率分布 $\pi$ 的问题，该决策者因偏离默认分布 $q$ 而需支付成本 $\tau D_{KL}(\pi \| q)$。

他们将此成本称为努力的负效用。

最大化者 $\pi^*$ 是 $q$ 的一个指数倾斜。

后续讲座 {doc}`information_projections` 在鲁棒控制的最坏情形模型、用于更新投入产出表的 RAS 方法，以及 {cite:t}`ChooSiow2006` 婚姻市场模型背后的熵正则化最优传输问题中，都发现了相同的倾斜。

让我们用数值方法检验 {eq}`ef_logit` 和 {eq}`ef_logit_base`。

```{code-cell} ipython3
v = np.array([1.0, 2.0, 0.5, 1.5])
τ = 0.7
inclusive_value = τ * logsumexp(v / τ)
π_logit = np.exp(v / τ - logsumexp(v / τ))
print(f"{'inclusive value:':36} {inclusive_value:.6f}")
objective = π_logit @ v + τ * entropy(π_logit)
print(f"{'π·v + τ H(π) at logit π:':36} {objective:.6f}")

# other probability vectors do worse
best_other = max(π @ v + τ * entropy(π)
                 for π in rng.dirichlet(np.ones(4), size=10_000))
print(f"{'best of 10,000 random π:':36} {best_other:.6f}")

# a nonuniform base distribution
q = np.array([0.1, 0.2, 0.3, 0.4])
π_q = tilt(q, v / τ, 1.0)
print(f"\n{'τ log Σ q exp(v/τ):':36} {τ * logsumexp(v / τ, b=q):.6f}")
print(f"{'π·v - τ D(π || q) at π ∝ q e^(v/τ):':36} "
      f"{π_q @ v - τ * kl(π_q, q):.6f}")
```

## 布雷格曼散度与 KL 散度

### 布雷格曼散度

凸函数位于其切线之上。

由严格凸函数 $F$ 生成的**布雷格曼散度**（Bregman divergence）度量了超出的量：

$$
B_F(\theta_1, \theta_2) = F(\theta_1) - F(\theta_2) - \nabla F(\theta_2) \cdot (\theta_1 - \theta_2) \geq 0 .
$$ (ef_bregman)

它是在 $\theta_1$ 处 $F$ 与 $\theta_2$ 处切线之间的垂直间隙，且仅在 $\theta_1 = \theta_2$ 时为零。

布雷格曼散度一般不是其两个参数的对称函数。

当 $F(\theta) = \frac{1}{2} \lVert \theta \rVert^2$ 时，它是欧几里得距离平方的一半。

### KL 散度是一种布雷格曼散度

取指数族的两个成员，其自然参数分别为 $\theta_1$ 和 $\theta_2$，期望参数分别为 $\eta_1$ 和 $\eta_2$。

由于 $\log p(x; \theta_1) - \log p(x; \theta_2) = (\theta_1 - \theta_2) \cdot t(x) - F(\theta_1) + F(\theta_2)$，在 $p(\cdot; \theta_1)$ 下取期望得到

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) - F(\theta_1) - \eta_1 \cdot (\theta_2 - \theta_1)
= B_F(\theta_2, \theta_1) .
$$ (ef_kl_bregman)

参数位置互换：*$p_{\theta_1}$ 相对于 $p_{\theta_2}$ 的 KL 散度，是 $\theta_2$ 相对于 $\theta_1$ 的布雷格曼散度。*

利用 {eq}`ef_fenchel` 替换 $F(\theta_1)$，我们也可以写成

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) + F^*(\eta_1) - \theta_2 \cdot \eta_1
= B_{F^*}(\eta_1, \eta_2) ,
$$ (ef_kl_dual)

其中参数的出现顺序与 KL 散度中的顺序相同。

无论在哪个坐标系统中，KL 散度都是一个布雷格曼散度，在自然参数中由 $F$ 生成，在期望参数中由 $F^*$ 生成。

让我们对泊松分布和正态分布检验 {eq}`ef_kl_bregman` 和 {eq}`ef_kl_dual`，通过求和和积分直接计算 KL 散度。

```{code-cell} ipython3
def bregman(F, θ1, θ2):
    """B_F(θ1, θ2), using a numerical gradient of F at θ2."""
    θ1, θ2 = np.atleast_1d(θ1), np.atleast_1d(θ2)
    return np.squeeze(F(θ1) - F(θ2)) - gradient(F, θ2) @ (θ1 - θ2)


def F_star_poisson(η):
    return η * np.log(η) - η


# Poisson
λ1, λ2 = 3.0, 5.0
k = np.arange(150)
kl_direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print("Poisson")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_poisson, np.log(λ2), np.log(λ1)):.6f}")
print(f"  B_F*(η1, η2)          = {bregman(F_star_poisson, λ1, λ2):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_poisson, np.log(λ1), np.log(λ2)):.6f}")

# normal
(μ1, σ1), (μ2, σ2) = (0.0, 1.0), (1.0, 2.0)
kl_direct = integrate.quad(
    lambda x: norm.pdf(x, μ1, σ1)
    * (norm.logpdf(x, μ1, σ1) - norm.logpdf(x, μ2, σ2)), -20, 20)[0]
print("\nnormal")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_normal, normal_θ(μ2, σ2), normal_θ(μ1, σ1)):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_normal, normal_θ(μ1, σ1), normal_θ(μ2, σ2)):.6f}")
```

对于正态族，展开 $B_F(\theta_2, \theta_1)$ 可以恢复出熟悉的公式

$$
D_{KL}\left( N(\mu_1, \sigma_1^2) \| N(\mu_2, \sigma_2^2) \right)
= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2} ,
$$

当 $\sigma_1 = \sigma_2 = \sigma$ 时，该式简化为我们在开头四个正态分布中用到的表达式 $(\mu_1 - \mu_2)^2/(2\sigma^2)$。

该公式的多元版本出现在 {doc}`advanced:entropy` 中。

### 费舍尔信息是 KL 散度的曲率

当 $\theta_2$ 接近 $\theta_1$ 时，对 {eq}`ef_kl_bregman` 做二阶泰勒展开得到

$$
D_{KL}\left( p_{\theta} \| p_{\theta + \delta} \right) \approx \frac{1}{2} \, \delta^\top \nabla^2 F(\theta) \, \delta .
$$

对于指数族，$\nabla^2 F(\theta)$ 就是**费舍尔信息矩阵**（Fisher information matrix）$I(\theta) = \operatorname{Cov}_\theta[\nabla_\theta \log p(X; \theta)]$，因为得分函数 $\nabla_\theta \log p(X; \theta) = t(X) - \nabla F(\theta)$ 具有协方差矩阵 $\operatorname{Cov}_\theta[t(X)] = \nabla^2 F(\theta)$。

*费舍尔信息矩阵度量了邻近分布变得可区分的速度。*

这解释了我们开头提到的四个正态分布。

在 $\sigma$ 固定的情况下，$\mu$ 的费舍尔信息为 $1/\sigma^2$，所以 $\mu$ 的单位变化会使分布移动 $\frac{1}{2} \cdot 1^2 / \sigma^2$，当 $\sigma = 0.2$ 时为 $12.5$，当 $\sigma = 2$ 时为 $0.125$。

对于均值的单独变化，这个二次公式是精确的。

在期望参数中，费舍尔信息矩阵为 $\nabla^2 F^*(\eta) = I(\theta)^{-1}$。

如果 $\theta$ 是另一个参数 $\xi$ 的函数，雅可比矩阵为 $J = \partial \theta / \partial \xi$，那么 $\xi$ 的费舍尔信息矩阵为 $J^\top I(\theta) J$；当 $\xi = \eta$ 时，$J = \nabla^2 F^*(\eta)$，且 $J^\top I(\theta) J = I(\theta)^{-1}$。

两个坐标系统中的二次近似在位移大小的二阶上是一致的，但在位移较大时会出现差异。

下图展示了泊松族的这一情况，比较了 $D_{KL}(p_{\lambda_0} \| p_\lambda)$ 与它在自然参数中的二次近似 $\frac{1}{2} \lambda_0 (\log \lambda - \log \lambda_0)^2$，以及在期望参数中的二次近似 $\frac{1}{2} (\lambda - \lambda_0)^2 / \lambda_0$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Poisson KL divergence and quadratic approximations
    name: fig-ef-fisher
---
λ0 = 4.0
λ_vals = np.linspace(1, 10, 200)
kl_vals = λ0 * np.log(λ0 / λ_vals) + λ_vals - λ0

fig, ax = plt.subplots()
ax.plot(λ_vals, kl_vals, lw=3, label=r"$D_{KL}(p_{\lambda_0} \| p_\lambda)$")
ax.plot(λ_vals, 0.5 * λ0 * np.log(λ_vals / λ0)**2, "--",
        label="quadratic in natural parameter")
ax.plot(λ_vals, 0.5 * (λ_vals - λ0)**2 / λ0, ":",
        label="quadratic in expectation parameter")
ax.set_xlabel(r"$\lambda$")
ax.set_ylim(0, 2.5)
ax.legend(loc="upper center")
plt.show()
```

在 $\lambda_0$ 附近，这三条曲线重合；离得较远时它们会分开。

## 估计

### 最大似然匹配矩

设 $x_1, \ldots, x_n$ 是来自指数族某一成员的独立同分布抽样。

平均对数似然为

$$
\frac{1}{n} \sum_{i=1}^n \log p(x_i; \theta) = \theta \cdot \bar t - F(\theta) + \frac{1}{n} \sum_{i=1}^n \log h(x_i),
\qquad
\bar t = \frac{1}{n} \sum_{i=1}^n t(x_i) .
$$

统计量 $t$ 是充分的：除了不涉及 $\theta$ 的最后一项，似然仅通过 $\bar t$ 依赖于数据。

平均对数似然是 $\theta$ 的凹函数，其一阶条件为 $\nabla F(\hat \theta) = \bar t$。

*期望参数的最大似然估计量是充分统计量的样本平均值*，$\hat \eta = \bar t$。

自然参数的最大似然估计量为 $\hat \theta = \nabla F^*(\bar t)$，前提是 $\bar t$ 位于 $\nabla F$ 的值域中。

若不是这样，例如在没有任何成功的伯努利样本中，$\hat \theta$ 不存在（参见 {ref}`ef_ex_bernoulli`）。

由 {eq}`ef_fenchel`，最大化的平均对数似然为 $F^*(\bar t)$ 加上 $\log h(x_i)$ 的平均值。

在期望参数中，最大似然估计是简单的；在自然参数中，则需要对 $\nabla F$ 求逆。

对于伽马族和贝塔族，对 $\nabla F$ 求逆需要数值方法，例如我们上面使用的牛顿迭代法。

### 两种估计量的有限样本行为

两个坐标系统中的最大似然估计量在有限样本中的表现不同。

估计量 $\hat \eta = \bar t$ 是无偏的，因为它是样本平均值；其协方差矩阵为 $\operatorname{Cov}_\theta[t(X)]/n = \nabla^2 F(\theta)/n$。

$\eta$ 的费舍尔信息矩阵为 $\nabla^2 F^*(\eta)$，所以 $\eta$ 的无偏估计量的克拉默-拉奥下界为 $\left[\nabla^2 F^*(\eta)\right]^{-1}/n = \nabla^2 F(\theta)/n$。

*在期望参数中，最大似然估计量是无偏的，并且在任意样本量下都精确地达到克拉默-拉奥界。*

自然参数的估计量 $\hat \theta = \nabla F^*(\bar t)$ 是 $\bar t$ 的非线性函数，在有限样本中是有偏的；对于正态族，当 $n > 3$ 时，$E \hat \theta = n \theta / (n - 3)$。

克拉默-拉奥界只适用于无偏估计量，所以对于有偏的 $\hat \theta$，我们将 $I(\theta)^{-1}/n$ 作为基准。

在指数族中，无偏估计量只有在它所估计的量是 $\eta$ 的仿射函数时才能在有限样本中达到克拉默-拉奥界。

自然参数 $\theta = \nabla F^*(\eta)$ 仅当 $F$ 是二次函数时才是 $\eta$ 的仿射函数，例如方差已知的正态族。

对于均值和方差均未知的正态族，$\theta_1 = \eta_1 / (\eta_2 - \eta_1^2)$ 和 $\theta_2 = -1/(2(\eta_2 - \eta_1^2))$ 都不是 $\eta$ 的仿射函数，所以两者都没有无偏估计量能达到该界。

对于正态族，当 $n \to \infty$ 时，$\hat \theta$ 的偏差消失，$n$ 乘以其协方差矩阵收敛到 $I(\theta)^{-1}$。

一般来说，渐近理论描述了 $\hat \theta$ 的分布，其有限样本矩未必存在，如 {ref}`ef_ex_bernoulli` 所示。

下面的模拟针对均值为 $\mu = 1$、$\sigma = 1.5$ 的正态族，使用样本量 $n = 30$ 来展示这些事实。

椭圆包含了以真实参数为中心的正态分布概率的 95%，左侧面板中协方差矩阵为 $\nabla^2 F(\theta)/n = I(\theta)/n$，右侧面板中为 $I(\theta)^{-1}/n$。

```{code-cell} ipython3
def ellipse(center, cov, level=chi2.ppf(0.95, 2), n_points=200):
    """Points on the ellipse (z - center)' cov^{-1} (z - center) = level."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    return center[:, None] + np.linalg.cholesky(level * cov) @ circle


μ, σ, n, R = 1.0, 1.5, 30, 20_000
θ_true = normal_θ(μ, σ)
η_true = np.array([μ, μ**2 + σ**2])
I_θ = hessian(F_normal, θ_true)        # Fisher information for θ

samples = rng.normal(μ, σ, size=(R, n))
η_hat = np.column_stack([samples.mean(axis=1), (samples**2).mean(axis=1)])
σ2_hat = η_hat[:, 1] - η_hat[:, 0]**2
θ_hat = np.column_stack([η_hat[:, 0] / σ2_hat, -1 / (2 * σ2_hat)])

print("expectation parameters")
print("  bias:", np.round(η_hat.mean(axis=0) - η_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(η_hat.T), 3))
print("  n × Cramér–Rao bound = ∇²F(θ):\n", np.round(I_θ, 3))
print("natural parameters")
print("  bias:", np.round(θ_hat.mean(axis=0) - θ_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(θ_hat.T), 3))
print("  n × benchmark = I(θ)^{-1}:\n", np.round(np.linalg.inv(I_θ), 3))
print(f"  E θ̂ = nθ/(n-3):", np.round(n * θ_true / (n - 3), 4),
      "  mean of θ̂:", np.round(θ_hat.mean(axis=0), 4))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum likelihood estimates in both parameterizations
    name: fig-ef-cramer-rao
---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
panels = ((axes[0], η_hat, η_true, I_θ / n,
           (r"$\hat\eta_1$", r"$\hat\eta_2$"), "expectation parameters",
           "Cramér–Rao ellipse", "lower right"),
          (axes[1], θ_hat, θ_true, np.linalg.inv(I_θ) / n,
           (r"$\hat\theta_1$", r"$\hat\theta_2$"), "natural parameters",
           r"ellipse from $I(\theta)^{-1}/n$", "lower left"))
for ax, est, truth, cov, labels, title, ell_label, loc in panels:
    ax.scatter(*est[:1000].T, s=3, alpha=0.3)
    ax.plot(*ellipse(truth, cov), "k", lw=1.5, label=ell_label)
    ax.plot(*truth, "r+", ms=14, mew=2, label="true value")
    ax.plot(*est.mean(axis=0), "kx", ms=10, mew=2, label="mean of estimates")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(loc=loc)
plt.tight_layout()
plt.show()
```

在期望参数中，估计量的点云以真实值为中心，其协方差矩阵在模拟误差范围内等于克拉默-拉奥界。

在自然参数中，点云是偏斜的，其均值相对真实值偏移了偏差量 $3\theta/(n - 3)$，其协方差矩阵大于基准 $I(\theta)^{-1}/n$，并且只有随着 $n$ 增大才逐渐接近该基准。

## 结束语

对于指数族，一个凸函数 $F$ 给出了四个事实：

* 它的梯度将自然参数 $\theta$ 映射到期望参数 $\eta$，而其共轭函数 $F^*$ 的梯度则将其映射回去；
* 它的 Hessian 矩阵是充分统计量的协方差矩阵，也是费舍尔信息矩阵；
* 当基测度为常数时，它的共轭函数是熵的负数；以及
* $F$ 和 $F^*$ 的布雷格曼散度都是 KL 散度。

指数族的成员解决了最大熵问题。

在所有具有给定均值和方差的密度中，正态密度使熵最大化。

最大熵问题的拉格朗日乘子就是自然参数和影子价格。

逻辑模型的包含值是一个对数配分函数，它的共轭函数是负熵。

后续讲座 {doc}`information_projections` 利用这些事实来描述 KL 散度的几何结构。

## 练习

```{exercise}
:label: ef_ex_gamma

考虑 $(0, \infty)$ 上的密度。

1. 证明在给定 $E[X]$ 和 $E[\log X]$ 的值的约束下，使熵最大化的密度是伽马密度，并将伽马族写成 {eq}`ef_family` 的形式。
2. 计算对数配分函数 $F$，并利用 {eq}`ef_gradient` 用形状参数和率参数表示 $E[X]$ 和 $E[\log X]$。
3. 从形状参数为 $2.5$、率参数为 $1.5$ 的伽马分布中抽取 500 个观测值，并通过匹配 $X$ 和 $\log X$ 的样本均值，利用 Hessian 矩阵 {eq}`ef_hessian` 的牛顿法计算最大似然估计量。将你的估计与固定位置参数为零时 `scipy.stats.gamma.fit` 给出的估计进行比较。
```

```{solution-start} ef_ex_gamma
:class: dropdown
```

以下是一种解法。

对于第 1 部分，导出 {eq}`ef_maxent_solution` 的论证给出一个正比于 $\exp(\theta_1 x + \theta_2 \log x) = x^{\theta_2} e^{\theta_1 x}$ 的密度，当 $\theta_1 < 0$ 且 $\theta_2 > -1$ 时在 $(0, \infty)$ 上可积。

这就是形状参数 $a = \theta_2 + 1$、率参数 $b = -\theta_1$ 的伽马密度：

$$
p(x; a, b) = \frac{b^a}{\Gamma(a)} x^{a - 1} e^{-b x} .
$$

所以 $t(x) = (x, \log x)$，$\theta = (-b, a - 1)$，$h(x) = 1$。

对于第 2 部分，

$$
F(\theta) = \log \int_0^\infty x^{\theta_2} e^{\theta_1 x} dx = \log \Gamma(\theta_2 + 1) - (\theta_2 + 1) \log(-\theta_1) .
$$

求导得

$$
E[X] = \frac{\partial F}{\partial \theta_1} = \frac{\theta_2 + 1}{-\theta_1} = \frac{a}{b},
\qquad
E[\log X] = \frac{\partial F}{\partial \theta_2} = \psi(a) - \log b ,
$$

其中 $\psi$ 是双伽马函数（digamma function）。

Hessian 矩阵为

$$
\nabla^2 F(\theta) = \begin{bmatrix} a/b^2 & 1/b \\ 1/b & \psi'(a) \end{bmatrix} .
$$

对于第 3 部分，我们对 $\nabla F(\theta) = \bar t$ 应用牛顿法。

我们从矩方法估计值开始，并对任何会使其离开 $\Theta$（其中 $a > 0$ 且 $b > 0$）的牛顿步长进行缩短。

```{code-cell} ipython3
def gamma_mle(x, tol=1e-10, max_iter=100):
    """Match the sample means of x and log x by Newton's method in θ."""
    t_bar = np.array([x.mean(), np.log(x).mean()])
    a = x.mean()**2 / x.var()          # method-of-moments starting values
    b = a / x.mean()
    for _ in range(max_iter):
        grad_F = np.array([a / b, digamma(a) - np.log(b)])
        hess_F = np.array([[a / b**2, 1 / b],
                           [1 / b, polygamma(1, a)]])
        step = np.linalg.solve(hess_F, t_bar - grad_F)  # step in (-b, a - 1)
        s = 1.0
        while b - s * step[0] <= 0 or a + s * step[1] <= 0:
            s /= 2                                      # stay inside Θ
        b, a = b - s * step[0], a + s * step[1]
        if max(abs(s * step[0]) / b, abs(s * step[1]) / a) < tol:
            break
    else:
        raise RuntimeError("no convergence")
    return a, b


rng_gamma = np.random.default_rng(6)
x = rng_gamma.gamma(shape=2.5, scale=1 / 1.5, size=500)
a_hat, b_hat = gamma_mle(x)
a_sp, _, scale_sp = gamma_dist.fit(x, floc=0)
print(f"moment matching: shape = {a_hat:.5f}, rate = {b_hat:.5f}")
print(f"scipy fit:       shape = {a_sp:.5f}, rate = {1 / scale_sp:.5f}")
```

这两组估计值是一致的。

```{solution-end}
```

```{exercise}
:label: ef_ex_poisson

泊松族具有非常数的基测度 $h(x) = 1/x!$。

1. 利用 {eq}`ef_conjugate` 证明 $F^*(\eta) = \eta \log \eta - \eta$，并检验 $\nabla F^*$ 是 $\nabla F$ 的反函数。
2. 证明泊松($\lambda$)分布的熵为 $H = \lambda - \lambda \log \lambda + E[\log X!]$，从而 $H = -F^*(\lambda) - E \log h(X)$ 且 $F^* \neq -H$。通过直接计算 $-\sum_x p(x) \log p(x)$，对 $\lambda \in \{0.5, 3, 20\}$ 数值验证该公式。
3. 利用 {eq}`ef_kl_dual` 推导 $D_{KL}(\text{泊松}(\lambda_1) \| \text{泊松}(\lambda_2))$ 的闭式解，并与直接求和进行比对。
```

```{solution-start} ef_ex_poisson
:class: dropdown
```

以下是一种解法。

对于第 1 部分，$F(\theta) = e^\theta$，所以 $\sup_\theta \{\theta \eta - e^\theta\}$ 的一阶条件为 $\eta = e^\theta$，即 $\theta = \log \eta$。

代入得 $F^*(\eta) = \eta \log \eta - \eta$。

于是 $\nabla F^*(\eta) = \log \eta$，这正是 $\nabla F(\theta) = e^\theta$ 的反函数。

对于第 2 部分，当 $\theta = \log \lambda$ 时 $\log p(x) = \theta x - e^\theta - \log x!$，所以

$$
H = -E \log p(X) = -\lambda \log \lambda + \lambda + E[\log X!] = -F^*(\lambda) - E \log h(X) .
$$

因为只要 $\lambda > 0$ 就有 $E[\log X!] > 0$，$-F^*(\lambda)$ 比 $H$ 小，如下表最后一列所示。

```{code-cell} ipython3
k = np.arange(300)
print(f"{'λ':>5} {'direct H':>10} {'formula':>10} {'-F*(λ)':>10}")
for λ in (0.5, 3.0, 20.0):
    p = poisson.pmf(k, λ)
    H_direct = -p @ poisson.logpmf(k, λ)
    H_formula = λ - λ * np.log(λ) + p @ gammaln(k + 1)
    print(f"{λ:5.1f} {H_direct:10.6f} {H_formula:10.6f} "
          f"{-F_star_poisson(λ):10.6f}")
```

对于第 3 部分，{eq}`ef_kl_dual` 给出

$$
D_{KL}(p_{\lambda_1} \| p_{\lambda_2})
= F^*(\lambda_1) - F^*(\lambda_2) - \log \lambda_2 \, (\lambda_1 - \lambda_2)
= \lambda_1 \log \frac{\lambda_1}{\lambda_2} - \lambda_1 + \lambda_2 ,
$$

这正是我们用来绘制 {numref}`fig-ef-fisher` 的公式。

```{code-cell} ipython3
λ1, λ2 = 3.0, 5.0
direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print(f"direct sum:  {direct:.8f}")
print(f"closed form: {λ1 * np.log(λ1 / λ2) - λ1 + λ2:.8f}")
```

```{solution-end}
```

```{exercise}
:label: ef_ex_bernoulli

对伯努利族重复两个坐标系统的比较。

1. 取 $\pi = 0.2$ 和 $n = 25$，模拟多个样本并计算 $\hat \eta = \bar x$ 和 $\hat \theta = \log(\hat \eta / (1 - \hat \eta))$。
2. 验证 $\hat \eta$ 是无偏的，且方差等于克拉默-拉奥界 $\pi(1 - \pi)/n$。
3. 有多大比例的样本会产生无穷的 $\hat \theta$？在 $\hat \theta$ 有限的样本中，将 $\hat \theta$ 的均值和方差与 $\theta$ 以及基准 $1/(n \pi (1 - \pi))$ 进行比较。当 $n = 400$ 时，这些比较结果有何变化？
```

```{solution-start} ef_ex_bernoulli
:class: dropdown
```

以下是一种解法。

```{code-cell} ipython3
def bernoulli_experiment(π, n, rng, R=100_000):
    η_hat = rng.binomial(n, π, size=R) / n
    θ = np.log(π / (1 - π))
    finite = (η_hat > 0) & (η_hat < 1)
    θ_hat = np.log(η_hat[finite] / (1 - η_hat[finite]))
    print(f"n = {n}")
    print(f"  η̂: mean {η_hat.mean():.4f} (π = {π}), "
          f"variance {η_hat.var():.6f} (bound {π * (1 - π) / n:.6f})")
    print(f"  fraction with infinite θ̂: {1 - finite.mean():.4f}")
    print(f"  finite θ̂: mean {θ_hat.mean():.4f} (θ = {θ:.4f}), "
          f"variance {θ_hat.var():.4f} "
          f"(benchmark {1 / (n * π * (1 - π)):.4f})")


rng_bern = np.random.default_rng(2)
bernoulli_experiment(0.2, 25, rng_bern)
bernoulli_experiment(0.2, 400, rng_bern)
```

当 $n = 25$ 时，约有 $0.4$ 百分比的样本完全没有成功。

在那些样本中，$\bar x = 0$ 位于 $\nabla F$ 值域（即区间 $(0, 1)$）的边界上，所以 $\theta$ 的最大似然估计量不存在；公式给出 $\hat \theta = -\infty$。

在其余样本中，$\hat \theta$ 向下偏斜，其方差比基准 $1/(n \pi (1 - \pi))$ 高出超过 20%。

当 $n = 400$ 时，无穷估计基本上不会出现，偏差以及方差与基准之间的差距都很小，这符合渐近理论的预测。

与此同时，$\hat \eta$ 在两种样本量下都是无偏的，并且达到了克拉默-拉奥界。

```{solution-end}
```

```{exercise}
:label: ef_ex_conjugate

贝叶斯法则在自然参数下具有简单的形式。

1. 设 $x_1, \ldots, x_n$ 为独立同分布的伯努利($\pi$)抽样，设 $\pi$ 的先验为贝塔($a$, $b$)分布。利用上面贝塔分布那一行，将先验写成自然参数为 $(a - 1, b - 1)$ 的贝塔族成员，并证明后验是同一族的成员，其自然参数为先验的自然参数加上 $\left(\sum_i x_i, \, n - \sum_i x_i\right)$。
2. 更一般地，解释为什么如果对数似然是参数的某些函数的线性函数，而先验是以这些函数作为充分统计量的指数族成员，那么后验就在同一族中，且贝叶斯法则是对其自然参数做加法。
3. 从贝塔(1, 1)先验出发，用 $\pi = 0.3$ 的抽样，将 $\pi$ 的后验均值（带有两个后验标准差的带状区域）作为 $n$ 的函数绘图。在 $n = 200$ 次抽样之后，利用 {eq}`ef_gradient` 和 {eq}`ef_hessian` 计算 $\log \pi$ 的后验均值和方差，并通过蒙特卡洛方法进行检验。

贝塔-伯努利模型出现在 {doc}`exchangeable`、{doc}`likelihood_bayes` 和 {doc}`merging_of_opinions` 中。
```

```{solution-start} ef_ex_conjugate
:class: dropdown
```

以下是一种解法。

对于第 1 部分，先验密度正比于 $\pi^{a - 1}(1 - \pi)^{b - 1} = \exp\left( (a - 1) \log \pi + (b - 1) \log(1 - \pi) \right)$。

$n$ 次抽样中有 $s = \sum_i x_i$ 次成功的似然为 $\pi^s (1 - \pi)^{n - s} = \exp\left( s \log \pi + (n - s) \log (1 - \pi) \right)$。

相乘后，后验正比于 $\exp\left( (a - 1 + s) \log \pi + (b - 1 + n - s) \log(1 - \pi) \right)$，即贝塔($a + s$, $b + n - s$)密度。

其自然参数为 $(a - 1 + s, b - 1 + n - s)$。

对于第 2 部分，贝叶斯法则表明，对数后验在相差一个常数的意义下等于对数先验加上对数似然。

如果对数似然是参数某些函数（这里是 $(\log \pi, \log(1 - \pi))$）的线性函数，而对数先验也是这些相同函数的线性函数，那么对数后验也是如此，其系数相加。

以这些参数函数作为充分统计量的先验族称为**共轭**（conjugate）族。

这里"共轭"一词的用法与凸共轭 $F^*$ 无关。

对于第 3 部分，$\pi$ 的后验均值为 $(a + s)/(a + b + n)$，后验方差为 $(a + s)(b + n - s) / \left((a + b + n)^2 (a + b + n + 1)\right)$。

```{code-cell} ipython3
rng_bayes = np.random.default_rng(3)
π_true, a0, b0 = 0.3, 1.0, 1.0
x = rng_bayes.binomial(1, π_true, size=500)
s = np.cumsum(x)
n_vals = np.arange(1, x.size + 1)
a_post, b_post = a0 + s, b0 + n_vals - s

post_mean = a_post / (a_post + b_post)
post_sd = np.sqrt(a_post * b_post
                  / ((a_post + b_post)**2 * (a_post + b_post + 1)))
fig, ax = plt.subplots()
ax.plot(n_vals, post_mean, label="posterior mean")
ax.fill_between(n_vals, post_mean - 2 * post_sd, post_mean + 2 * post_sd,
                alpha=0.3, label="± 2 posterior standard deviations")
ax.axhline(π_true, color="k", ls="--", lw=1, label=r"true $\pi$")
ax.set_xlabel("$n$")
ax.set_ylim(0, 1)
ax.legend()
plt.show()
```

经过 $n = 200$ 次抽样后，后验为贝塔($a$, $b$)分布，其中 $(a, b) = (a_0 + s, b_0 + 200 - s)$。

对 $F(\theta) = \log B(\theta_1 + 1, \theta_2 + 1)$ 应用 {eq}`ef_gradient` 得到 $E[\log \pi] = \psi(a) - \psi(a + b)$，应用 {eq}`ef_hessian` 得到 $\operatorname{Var}[\log \pi] = \psi'(a) - \psi'(a + b)$。

```{code-cell} ipython3
a, b = a_post[199], b_post[199]
draws = rng_bayes.beta(a, b, size=200_000)
print(f"E[log π]:   formula {digamma(a) - digamma(a + b):.5f}, "
      f"Monte Carlo {np.log(draws).mean():.5f}")
print(f"Var[log π]: formula {polygamma(1, a) - polygamma(1, a + b):.5f}, "
      f"Monte Carlo {np.log(draws).var():.5f}")
```

```{solution-end}
```