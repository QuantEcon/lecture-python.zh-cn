---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: 最大似然估计
  headings:
    Overview: 概述
    Overview::Prerequisites: 预备知识
    Set up and assumptions: 设置和假设
    Set up and assumptions::Flow of ideas: 思路流程
    Set up and assumptions::Counting billionaires: 研究亿万富豪
    Conditional distributions: 条件分布
    Maximum likelihood estimation: 最大似然估计
    MLE with numerical methods: 使用数值方法的最大似然估计
    Maximum likelihood estimation with `statsmodels`: 使用 `statsmodels` 进行最大似然估计
    Summary: 总结
    Exercises: 练习
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 最大似然估计

```{include} _admonition/gpu.md
```

```{contents} 目录
:depth: 2
```

## 概述

在{doc}`之前的讲座 <ols>`中，我们使用线性回归估计了因变量和解释变量之间的关系。

但如果线性关系不适合我们的模型假设呢？

一个广泛使用的替代方法是最大似然估计，它涉及指定一类由未知参数索引的分布，然后使用数据来确定这些参数值。

与线性回归相比，其优势在于它允许变量之间有更灵活的概率关系。

在这里，我们通过复现Daniel Treisman（2016）的论文[《俄罗斯的亿万富翁》](https://www.aeaweb.org/articles?id=10.1257/aer.p20161068)来说明最大似然法。该论文将一个国家的亿万富翁数量与其经济特征联系起来。

该论文得出结论：俄罗斯的亿万富翁数量高于经济因素（如市场规模和税率）所预测的水平。

我们需要以下导入：

```{code-cell} ipython3
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
from typing import NamedTuple

from jax.scipy.special import factorial, gammaln
from jax.scipy.stats import norm

from statsmodels.api import Poisson
from statsmodels.iolib.summary2 import summary_col

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

### 预备知识

我们假设读者熟悉基本概率论和多元微积分。

## 设置和假设

让我们考虑最大似然估计所需的步骤以及它们与本研究的关系。

### 思路流程

最大似然估计的第一步是选择一个我们认为能够合理描述数据生成过程的概率分布。

更准确地说，我们需要对产生数据的*参数分布族*做出假设。

* 比如正态分布族或伽马分布族。

每个分布族都由有限个参数索引的分布家族。

* 以正态分布为例，它由均值 $\mu \in (-\infty, \infty)$ 和标准差 $\sigma \in (0, \infty)$ 两个参数来确定。

我们会利用数据来估计这些参数，从而找到最适合数据的具体分布。

这样得到的参数估计值将被称为**最大似然估计**。

### 研究亿万富豪

在Treisman {cite}`Treisman2016` 的研究中，他想要分析各国亿万富豪的数量。

亿万富豪的数量是整数值。

因此我们考虑只取非负整数值的分布。

（这是最小二乘回归不是当前问题最佳工具的原因之一，因为线性回归中的因变量不限于整数值）

一种整数分布是[泊松分布](https://en.wikipedia.org/wiki/Poisson_distribution)，它的概率质量函数(pmf)为

$$
f(y) = \frac{\mu^{y}}{y!} e^{-\mu},
\qquad y = 0, 1, 2, \ldots, \infty
$$

我们可以按如下方式绘制不同 $\mu$ 值下的泊松分布图

```{code-cell} ipython3
@jax.jit
def poisson_pmf(y, μ):
    return μ**y / factorial(y) * jnp.exp(-μ)
```

```{code-cell} ipython3
y_values = range(0, 25)

fig, ax = plt.subplots(figsize=(12, 8))

for μ in [1, 5, 10]:
    distribution = []
    for y_i in y_values:
        distribution.append(poisson_pmf(y_i, μ))
    ax.plot(
        y_values,
        distribution,
        label=rf"$\mu$={μ}",
        alpha=0.5,
        marker="o",
        markersize=8,
    )

ax.grid()
ax.set_xlabel(r"$y$", fontsize=14)
ax.set_ylabel(r"$f(y \mid \mu)$", fontsize=14)
ax.axis(xmin=0, ymin=0)
ax.legend(fontsize=14)

plt.show()
```

注意当 $y$ 的均值增加时，泊松分布开始呈现出类似正态分布的特征。

让我们来看看本讲中我们将要使用的数据分布情况。

Treisman的主要数据来源是《福布斯》年度富豪榜及其估计净资产。

数据集`mle/fp.dta`可以从[这里](https://python.quantecon.org/_static/lecture_specific/mle/fp.dta)
或其[AER页面](https://www.aeaweb.org/articles?id=10.1257/aer.p20161068)下载。

```{code-cell} ipython3
# 加载数据并查看
df = pd.read_stata(
    "https://github.com/QuantEcon/lecture-python.myst/raw/refs/heads/main/lectures/_static/lecture_specific/mle/fp.dta"
)
df.head()
```

通过直方图，我们可以查看2008年各国亿万富翁人数`numbil0`的分布情况（为了方便绘图，我们排除了美国数据）

```{code-cell} ipython3
numbil0_2008 = df[
    (df["year"] == 2008) & (df["country"] != "United States")
].loc[:, "numbil0"]

plt.subplots(figsize=(12, 8))
plt.hist(numbil0_2008, bins=30)
plt.xlim(left=0)
plt.grid()
plt.xlabel("2008年亿万富翁人数")
plt.ylabel("计数")
plt.show()
```

从直方图来看，泊松分布的假设似乎是合理的（尽管 $\mu$ 值很低且有一些异常值）。

## 条件分布

在Treisman的论文中，因变量——国家$i$的亿万富翁数量$y_i$——被建模为人均GDP、人口规模以及加入关贸总协定和世贸组织年限的函数。

这意味着$y_i$的分布取决于这些解释变量(记为向量$\mathbf{x}_i$)。

这种标准表述——即所谓的**泊松回归**模型——如下所示：

```{math}
:label: poissonreg

f(y_i \mid \mathbf{x}_i) = \frac{\mu_i^{y_i}}{y_i!} e^{-\mu_i}; \qquad y_i = 0, 1, 2, \ldots , \infty .
```

$$
\text{其中}\ \mu_i
     = \exp(\mathbf{x}_i' \boldsymbol{\beta})
     = \exp(\beta_0 + \beta_1 x_{i1} + \ldots + \beta_k x_{ik})
$$

为了说明$y_i$的分布依赖于$\mathbf{x}_i$这一概念，让我们进行一个简单的模拟。

我们使用上面的`poisson_pmf`函数和任意值的$\boldsymbol{\beta}$和$\mathbf{x}_i$

```{code-cell} ipython3
y_values = range(0, 20)

# 定义一个带有估计值的参数向量
β = jnp.array([0.26, 0.18, 0.25, -0.1, -0.22])

# 创建一些观测值X
datasets = [
    jnp.array([0, 1, 1, 1, 2]),
    jnp.array([2, 3, 2, 4, 0]),
    jnp.array([3, 4, 5, 3, 2]),
    jnp.array([6, 5, 4, 4, 7]),
]


fig, ax = plt.subplots(figsize=(12, 8))

for X in datasets:
    μ = jnp.exp(X @ β)
    distribution = []
    for y_i in y_values:
        distribution.append(poisson_pmf(y_i, μ))
    ax.plot(
        y_values,
        distribution,
        label=rf"$\mu_i$={μ:.1}",
        marker="o",
        markersize=8,
        alpha=0.5,
    )

ax.grid()
ax.legend()
ax.set_xlabel(r"$y \mid x_i$")
ax.set_ylabel(r"$f(y \mid x_i; \beta )$")
ax.axis(xmin=0, ymin=0)
plt.show()
```

我们可以看到 $y_i$ 的分布是以 $\mathbf{x}_i$ 为条件的（$\mu_i$ 不再是常数）。

## 最大似然估计

在我们的亿万富翁数量模型中，条件分布包含4个（$k = 4$）需要估计的参数。

我们将整个参数向量标记为 $\boldsymbol{\beta}$，其中

$$
\boldsymbol{\beta} = \begin{bmatrix}
                            \beta_0 \\
                            \beta_1 \\
                            \beta_2 \\
                            \beta_3
                      \end{bmatrix}
$$

为了使用最大似然估计来估计模型，我们希望最大化我们的估计值 $\hat{\boldsymbol{\beta}}$ 是真实参数 $\boldsymbol{\beta}$ 的似然。

直观地说，我们想要找到最适合我们数据的 $\hat{\boldsymbol{\beta}}$。

首先，我们需要构建似然函数 $\mathcal{L}(\boldsymbol{\beta})$，它类似于联合概率密度函数。

假设我们有一些数据 $y_i = \{y_1, y_2\}$ 且 $y_i \sim f(y_i)$。

如果 $y_1$ 和 $y_2$ 是独立的，这些数据的联合概率质量函数是
$f(y_1, y_2) = f(y_1) \cdot f(y_2)$。

如果 $y_i$ 服从参数为 $\lambda = 7$ 的泊松分布，我们可以这样可视化联合概率质量函数

```{code-cell} ipython3
def plot_joint_poisson(μ=7, y_n=20):
    yi_values = jnp.arange(0, y_n, 1)

    # 创建 X 和 Y 的坐标点
    X, Y = jnp.meshgrid(yi_values, yi_values)

    # 将分布相乘
    Z = poisson_pmf(X, μ) * poisson_pmf(Y, μ)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z.T, cmap="terrain", alpha=0.6)
    ax.scatter(X, Y, Z.T, color="black", alpha=0.5, linewidths=1)
    ax.set(xlabel=r"$y_1$", ylabel=r"$y_2$")
    ax.set_zlabel(r"$f(y_1, y_2)$", labelpad=10)
    plt.show()


plot_joint_poisson(μ=7, y_n=20)
```

同样，我们的数据（服从条件泊松分布）的联合概率质量函数可以写作：

$$
f(y_1, y_2, \ldots, y_n \mid \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n; \boldsymbol{\beta})
    = \prod_{i=1}^{n} \frac{\mu_i^{y_i}}{y_i!} e^{-\mu_i}
$$

$y_i$ 同时依赖于 $\mathbf{x}_i$ 的值和参数 $\boldsymbol{\beta}$。

似然函数与联合概率质量函数相同，但是将参数 $\boldsymbol{\beta}$ 视为随机变量，并将观测值 $(y_i, \mathbf{x}_i)$ 视为已知：

$$
\begin{split}
\mathcal{L}(\beta \mid y_1, y_2, \ldots, y_n \ ; \ \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n) = &
\prod_{i=1}^{n} \frac{\mu_i^{y_i}}{y_i!} e^{-\mu_i} \\ = &
f(y_1, y_2, \ldots, y_n \mid  \ \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n ; \beta)
\end{split}
$$

现在我们有了似然函数，我们要找到能使似然值最大的 $\hat{\boldsymbol{\beta}}$

$$
\underset{\boldsymbol{\beta}}{\max} \mathcal{L}(\boldsymbol{\beta})
$$

在这种情况下，最大化对数似然通常更容易（比较求导 $f(x) = x \exp(x)$ 与 $f(x) = \log(x) + x$）。

由于对数是单调递增变换，似然函数的最大值点也是对数似然函数的最大值点。

在我们的例子中，对数似然为

$$
\begin{split}
\log{ \mathcal{L}} (\boldsymbol{\beta}) = \ &
    \log \Big(
        f(y_1 ; \boldsymbol{\beta})
        \cdot
        f(y_2 ; \boldsymbol{\beta})
        \cdot \ldots \cdot
        f(y_n ; \boldsymbol{\beta})
        \Big) \\
        = &
        \sum_{i=1}^{n} \log{f(y_i ; \boldsymbol{\beta})} \\
        = &
        \sum_{i=1}^{n}
        \log \Big( {\frac{\mu_i^{y_i}}{y_i!} e^{-\mu_i}} \Big) \\
        = &
        \sum_{i=1}^{n} y_i \log{\mu_i} -
        \sum_{i=1}^{n} \mu_i -
        \sum_{i=1}^{n} \log y_i!
\end{split}
$$

泊松分布的 $\hat{\beta}$ 的最大似然估计可以通过求解以下问题得到：

$$
\underset{\beta}{\max} \Big(
\sum_{i=1}^{n} y_i \log{\mu_i} -
\sum_{i=1}^{n} \mu_i -
\sum_{i=1}^{n} \log y_i! \Big)
$$

然而，上述问题没有解析解——要找到最大似然估计，我们需要使用数值方法。

## 使用数值方法的最大似然估计

许多分布都没有很好的解析解，因此需要数值方法来求解参数估计。

牛顿-拉夫森（Newton-Raphson）算法就是这样一种数值方法。

我们的目标是找到最大似然估计 $\hat{\boldsymbol{\beta}}$。

在 $\hat{\boldsymbol{\beta}}$ 处，对数似然函数的一阶导数将等于0。

让我们通过假设以下函数来说明这一点：

$$
\log \mathcal{L(\beta)} = - (\beta - 10) ^2 - 10
$$

```{code-cell} ipython3
@jax.jit
def logL(β):
    return -((β - 10) ** 2) - 10
```

为了求出上述函数梯度的值，我们可以使用[jax.grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html)，它可以对给定函数自动求导。

我们进一步使用[jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)，它可以对给定函数进行向量化，即原本作用于标量输入的函数现在可以用于向量输入。

```{code-cell} ipython3
dlogL = jax.vmap(jax.grad(logL))
```

```{code-cell} ipython3
β = jnp.linspace(1, 20)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))

ax1.plot(β, logL(β), lw=2)
ax2.plot(β, dlogL(β), lw=2)

ax1.set_ylabel(
    r"$log \mathcal{L(\beta)}$", rotation=0, labelpad=35, fontsize=15
)
ax2.set_ylabel(
    r"$\frac{dlog \mathcal{L(\beta)}}{d \beta}$ ",
    rotation=0,
    labelpad=35,
    fontsize=19,
)
ax2.set_xlabel(r"$\beta$", fontsize=15)
ax1.grid(), ax2.grid()
plt.axhline(c="black")
plt.show()
```

图表显示最大似然值（上图）出现在
$\frac{d \log \mathcal{L(\boldsymbol{\beta})}}{d \boldsymbol{\beta}} = 0$ 时（下图）。

因此，似然函数在 $\beta = 10$ 时达到最大值。

我们还可以通过检查二阶导数（下图的斜率）是否为负来确保这个值是一个*最大值*（而不是最小值）。

牛顿-拉夫森算法用于寻找一阶导数为0的点。

要使用该算法，我们首先对最大值进行初始猜测，
$\beta_0$（OLS参数估计可能是一个合理的猜测），然后

1. 使用更新规则进行迭代

   $$
   \boldsymbol{\beta}_{(k+1)} = \boldsymbol{\beta}_{(k)} - H^{-1}(\boldsymbol{\beta}_{(k)})G(\boldsymbol{\beta}_{(k)})
   $$

   其中：

   $$
   \begin{aligned}
   G(\boldsymbol{\beta}_{(k)}) = \frac{d \log \mathcal{L(\boldsymbol{\beta}_{(k)})}}{d \boldsymbol{\beta}_{(k)}} \\

H(\boldsymbol{\beta}_{(k)}) = \frac{d^2 \log \mathcal{L(\boldsymbol{\beta}_{(k)})}}{d \boldsymbol{\beta}_{(k)}d \boldsymbol{\beta}'_{(k)}}
   \end{aligned}
   $$

2. 检查 $\boldsymbol{\beta}_{(k+1)} - \boldsymbol{\beta}_{(k)} < tol$ 是否成立
    - 如果成立，则停止迭代并设定
      $\hat{\boldsymbol{\beta}} = \boldsymbol{\beta}_{(k+1)}$
    - 如果不成立，则更新 $\boldsymbol{\beta}_{(k+1)}$

从更新方程可以看出，只有当 $G(\boldsymbol{\beta}_{(k)}) = 0$ 时，即一阶导数等于0时，才有 $\boldsymbol{\beta}_{(k+1)} = \boldsymbol{\beta}_{(k)}$。

（在实践中，当差异小于一个很小的容差阈值时，我们就停止迭代）

让我们来实现牛顿-拉夫森算法。

首先，我们创建一个名为 `PoissonRegression` 的类，这样我们就可以在每次迭代时轻松重新计算对数似然、梯度和海森矩阵的值

```{code-cell} ipython3
class PoissonRegression(NamedTuple):
    X: jnp.ndarray
    y: jnp.ndarray
```

现在我们可以用Python定义对数似然函数

```{code-cell} ipython3
@jax.jit
def logL(β, model):
    y = model.y
    μ = jnp.exp(model.X @ β)
    return jnp.sum(model.y * jnp.log(μ) - μ - jnp.log(factorial(y)))
```

为了求出`poisson_logL`的梯度，我们再次使用[jax.grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html)。

根据[相关文档](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev)：

* `jax.jacfwd`使用前向模式自动微分，对于"高"雅可比矩阵更高效，而
* `jax.jacrev`使用反向模式，对于"宽"雅可比矩阵更高效。

（文档还指出，当矩阵接近方阵时，`jax.jacfwd`可能比`jax.jacrev`更有优势。）

因此，为了求Hessian矩阵，我们可以直接使用`jax.jacfwd`。

```{code-cell} ipython3
G_logL = jax.grad(logL)
H_logL = jax.jacfwd(G_logL)
```

我们的函数`newton_raphson`将接收一个`PoissonRegression`对象，该对象包含参数向量$\boldsymbol{\beta}_0$的初始猜测值。

该算法将根据更新规则更新参数向量，并在新的参数估计值处重新计算梯度和Hessian矩阵。

迭代将在以下情况下结束：

* 参数与更新后参数之间的差异低于容差水平。
* 达到最大迭代次数（意味着未达到收敛）。

为了让我们能够了解算法运行时的情况，添加了`display=True`选项来打印每次迭代的值。

```{code-cell} ipython3
def newton_raphson(model, β, tol=1e-3, max_iter=100, display=True):

    i = 0
    error = 100  # 初始误差值

    # 打印输出的标题
    if display:
        header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"θ":<60}'
        print(header)
        print("-" * len(header))

    # 当error中的任何值大于容差且未达到最大迭代次数时，
    # while循环继续运行
    while jnp.any(error > tol) and i < max_iter:
        H, G = jnp.squeeze(H_logL(β, model)), G_logL(β, model)
        β_new = β - (jnp.dot(jnp.linalg.inv(H), G))
        error = jnp.abs(β_new - β)
        β = β_new

        if display:
            β_list = [f"{t:.3}" for t in list(β.flatten())]
            update = f"{i:<13}{logL(β, model):<16.8}{β_list}"
            print(update)

        i += 1

    print(f"迭代次数：{i}")
    print(f"β_hat = {β.flatten()}")

    return β
```

让我们用一个包含5个观测值和3个变量的小数据集来测试我们的算法$\mathbf{X}$。

```{code-cell} ipython3
X = jnp.array([[1, 2, 5], [1, 1, 3], [1, 4, 2], [1, 5, 2], [1, 3, 1]])

y = jnp.array([1, 0, 1, 1, 0])

# 对初始β值进行猜测
init_β = jnp.array([0.1, 0.1, 0.1])

# 创建一个包含泊松模型值的对象
poi = PoissonRegression(X=X, y=y)

# 使用牛顿-拉弗森方法找到最大似然估计
β_hat = newton_raphson(poi, init_β, display=True)
```

由于这是一个观测值较少的简单模型，算法仅用7次迭代就达到了收敛。

你可以看到，每次迭代后对数似然值都在增加。

请记住，我们的目标是最大化对数似然函数，这正是算法所做的。

同时，注意到$\log \mathcal{L}(\boldsymbol{\beta}_{(k)})$的增量在每次迭代后都变得更小。

这是因为当我们接近最大值时，梯度正在接近0，因此我们更新方程中的分子也变得更小。

在$\hat{\boldsymbol{\beta}}$处，梯度向量应该接近0

```{code-cell} ipython3
G_logL(β_hat, poi)
```

迭代过程可以在下图中可视化，其中最大值在 $\beta = 10$ 处

```{code-cell} ipython3
@jax.jit
def logL(x):
    return -((x - 10) ** 2) - 10


@jax.jit
def find_tangent(β, a=0.01):
    y1 = logL(β)
    y2 = logL(β + a)
    x = jnp.array([[β, 1], [β + a, 1]])
    m, c = jnp.linalg.lstsq(x, jnp.array([y1, y2]), rcond=None)[0]
    return m, c
```

```{code-cell} ipython3
:tags: [output_scroll]

β = jnp.linspace(2, 18)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(β, logL(β), lw=2, c="black")

for β in [7, 8.5, 9.5, 10]:
    β_line = jnp.linspace(β - 2, β + 2)
    m, c = find_tangent(β)
    y = m * β_line + c
    ax.plot(β_line, y, "-", c="purple", alpha=0.8)
    ax.text(β + 2.05, y[-1], rf"$G({β}) = {abs(m):.0f}$", fontsize=12)
    ax.vlines(β, -24, logL(β), linestyles="--", alpha=0.5)
    ax.hlines(logL(β), 6, β, linestyles="--", alpha=0.5)

ax.set(ylim=(-24, -4), xlim=(6, 13))
ax.set_xlabel(r"$\beta$", fontsize=15)
ax.set_ylabel(
    r"$log \mathcal{L(\beta)}$", rotation=0, labelpad=25, fontsize=15
)
ax.grid(alpha=0.3)
plt.show()
```

请注意，我们对牛顿-拉夫森算法的实现相当基础 --- 如需更稳健的实现方案，请参考例如 [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)。

## 使用 `statsmodels` 进行最大似然估计

现在我们已经了解了其内部运作原理，我们可以将最大似然估计应用到一个有趣的应用中。

我们将使用 `statsmodels` 中的泊松回归模型来获得更丰富的输出，包括标准误差、检验值等更多信息。

`statsmodels` 使用与上述相同的算法来找到最大似然估计值。

在开始之前，让我们用 `statsmodels` 重新估计我们的简单模型，并确认我们能得到相同的系数和对数似然值。

现在，由于`statsmodels`只接受NumPy数组，我们可以使用`np.array`方法将它们转换为NumPy数组。

```{code-cell} ipython3
X = jnp.array([[1, 2, 5], [1, 1, 3], [1, 4, 2], [1, 5, 2], [1, 3, 1]])

y = jnp.array([1, 0, 1, 1, 0])

y_numpy = np.array(y)
X_numpy = np.array(X)
stats_poisson = Poisson(y_numpy, X_numpy).fit()
print(stats_poisson.summary())
```

现在让我们复现Daniel Treisman的论文[Russia's Billionaires](https://www.aeaweb.org/articles?id=10.1257/aer.p20161068)中的结果，该论文在之前的讲座中提到过。

Treisman首先估计方程{eq}`poissonreg`，其中：

* $y_i$ 是 ${亿万富翁人数}_i$
* $x_{i1}$ 是 $\log{人均GDP}_i$
* $x_{i2}$ 是 $\log{人口}_i$
* $x_{i3}$ 是 ${GATT成员年限}_i$ -- 作为GATT和WTO成员的年限（用于衡量国际市场准入）

论文仅考虑2008年进行估计。

我们将按如下方式设置估计变量（你应该已经从讲座前面部分将数据赋值给了`df`）

```{code-cell} ipython3
# 仅保留2008年数据
df = df[df["year"] == 2008]

# 添加常数项
df["const"] = 1

# 变量集
reg1 = ["const", "lngdppc", "lnpop", "gattwto08"]
reg2 = [
    "const",
    "lngdppc",
    "lnpop",
    "gattwto08",
    "lnmcap08",
    "rintr",
    "topint08",
]
reg3 = [
    "const",
    "lngdppc",
    "lnpop",
    "gattwto08",
    "lnmcap08",
    "rintr",
    "topint08",
    "nrrents",
    "roflaw",
]
```

然后我们可以使用`statsmodels`中的`Poisson`函数来拟合模型。

我们将像作者论文中那样使用稳健标准误

```{code-cell} ipython3
# Specify model
poisson_reg = Poisson(df[["numbil0"]], df[reg1], missing="drop").fit(
    cov_type="HC0"
)
print(poisson_reg.summary())
```

成功！算法在9次迭代后实现了收敛。

我们的输出表明，人均GDP、人口和关税贸易总协定(GATT)的成员年限与一个国家的亿万富翁数量呈正相关，这符合预期。

让我们继续估计作者提出的两个更复杂的模型,并将三个模型的结果并排展示以便比较

```{code-cell} ipython3
regs = [reg1, reg2, reg3]
reg_names = ["Model 1", "Model 2", "Model 3"]
info_dict = {
    "Pseudo R-squared": lambda x: f"{x.prsquared:.2f}",
    "No. observations": lambda x: f"{int(x.nobs):d}",
}
regressor_order = [
    "const",
    "lngdppc",
    "lnpop",
    "gattwto08",
    "lnmcap08",
    "rintr",
    "topint08",
    "nrrents",
    "roflaw",
]
results = []

for reg in regs:
    result = Poisson(df[["numbil0"]], df[reg], missing="drop").fit(
        cov_type="HC0", maxiter=100, disp=0
    )
    results.append(result)

results_table = summary_col(
    results=results,
    float_format="%0.3f",
    stars=True,
    model_names=reg_names,
    info_dict=info_dict,
    regressor_order=regressor_order,
)
results_table.add_title(
    "Table 1 - Explaining the Number of Billionaires \
                        in 2008"
)
print(results_table)
```

结果显示，一个国家的亿万富翁数量会随着人均GDP、人口规模和股票市场规模的增加而增加。相反，较高的最高边际所得税率会降低亿万富翁的数量。

为了更好地理解各国的具体情况，我们来看看模型预测值与实际观测值之间的差异。我们将按差异大小排序，并展示差异最大的前15个国家。

```{code-cell} ipython3
data = [
    "const",
    "lngdppc",
    "lnpop",
    "gattwto08",
    "lnmcap08",
    "rintr",
    "topint08",
    "nrrents",
    "roflaw",
    "numbil0",
    "country",
]
results_df = df[data].dropna()

# 使用最后一个模型（模型3）
results_df["prediction"] = results[-1].predict()

# 计算差异
results_df["difference"] = results_df["numbil0"] - results_df["prediction"]

# 按降序排列
results_df.sort_values("difference", ascending=False, inplace=True)

# 绘制前15个数据点
results_df[:15].plot(
    "country", "difference", kind="bar", figsize=(12, 8), legend=False
)
plt.ylabel("高于预测水平的亿万富翁数量")
plt.xlabel("国家")
plt.show()
```

正如我们所见，俄罗斯的亿万富豪数量远远超出模型预测值（比预期多约50人）。

Treisman利用这一实证结果讨论了俄罗斯亿万富豪过多的可能原因，包括俄罗斯财富的来源、政治环境以及苏联解体后的私有化历史。

## 总结

在本讲中，我们使用最大似然估计法来估计泊松模型的参数。

`statsmodels`包含其他内置的似然模型，如[Probit](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Probit.html)和[Logit](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.html)。

为了提供更大的灵活性，`statsmodels`提供了使用`GenericLikelihoodModel`类手动指定分布的方法 - 示例notebook可以在
[此处](https://www.statsmodels.org/dev/examples/notebooks/generated/generic_mle.html)找到。

## 练习

```{exercise}
:label: mle_ex1

假设我们想要估计事件 $y_i$ 发生的概率，给定一些观测值。

我们可以使用概率回归模型，其中 $y_i$ 的概率质量函数为

$$
\begin{aligned}
f(y_i; \boldsymbol{\beta}) = \mu_i^{y_i} (1-\mu_i)^{1-y_i}, \quad y_i = 0,1 \\
\text{其中} \quad \mu_i = \Phi(\mathbf{x}_i' \boldsymbol{\beta})
\end{aligned}
$$

$\Phi$ 表示**累积正态分布**，它将预测的 $y_i$ 限制在0和1之间（这是概率所必需的）。

$\boldsymbol{\beta}$ 是一个系数向量。

按照讲座中的示例，编写一个类来表示Probit模型。

首先，找出对数似然函数并推导梯度和海森矩阵。

`jax.scipy.stats`模块中的`norm`包含计算正态分布的累积分布函数和概率密度函数所需的函数。
```

```{solution-start} mle_ex1
:class: dropdown
```

对数似然函数可以写作

$$
\log \mathcal{L} = \sum_{i=1}^n
\big[
y_i \log \Phi(\mathbf{x}_i' \boldsymbol{\beta}) +
(1 - y_i) \log (1 - \Phi(\mathbf{x}_i' \boldsymbol{\beta})) \big]
$$

根据**微积分基本定理**，累积概率分布的导数是其边际分布

$$
\frac{ \partial} {\partial s} \Phi(s) = \phi(s)
$$

其中$\phi$是边际正态分布。

Probit模型的梯度向量是

$$
\frac {\partial \log \mathcal{L}} {\partial \boldsymbol{\beta}} =
\sum_{i=1}^n \Big[
y_i \frac{\phi(\mathbf{x}'_i \boldsymbol{\beta})}{\Phi(\mathbf{x}'_i \boldsymbol{\beta)}} -
(1 - y_i) \frac{\phi(\mathbf{x}'_i \boldsymbol{\beta)}}{1 - \Phi(\mathbf{x}'_i \boldsymbol{\beta)}}
\Big] \mathbf{x}_i
$$

Probit模型的Hessian矩阵是

$$
\frac {\partial^2 \log \mathcal{L}} {\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}'} =
-\sum_{i=1}^n \phi (\mathbf{x}_i' \boldsymbol{\beta})
\Big[
y_i \frac{ \phi (\mathbf{x}_i' \boldsymbol{\beta}) + \mathbf{x}_i' \boldsymbol{\beta} \Phi (\mathbf{x}_i' \boldsymbol{\beta}) } { [\Phi (\mathbf{x}_i' \boldsymbol{\beta})]^2 } +
(1 - y_i) \frac{ \phi (\mathbf{x}_i' \boldsymbol{\beta}) - \mathbf{x}_i' \boldsymbol{\beta} (1 - \Phi (\mathbf{x}_i' \boldsymbol{\beta})) } { [1 - \Phi (\mathbf{x}_i' \boldsymbol{\beta})]^2 }
\Big]
\mathbf{x}_i \mathbf{x}_i'
$$

根据这些结果，我们可以按如下方式编写Probit模型的类

```{code-cell} ipython3
class ProbitRegression(NamedTuple):
    X: jnp.ndarray
    y: jnp.ndarray
```

```{code-cell} ipython3
@jax.jit
def logL(β, model):
    y = model.y
    μ = norm.cdf(model.X @ β.T)
    return y @ jnp.log(μ) + (1 - y) @ jnp.log(1 - μ)
```

```{code-cell} ipython3
G_logL = jax.grad(logL)
H_logL = jax.jacfwd(G_logL)
```

```{solution-end}
```

```{exercise-start}
:label: mle_ex2
```

使用以下数据集和$\boldsymbol{\beta}$的初始值，用课程前面介绍的牛顿-拉夫森算法来估计最大似然估计

$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 & 4 \\
1 & 1 & 1 \\
1 & 4 & 3 \\
1 & 5 & 6 \\
1 & 3 & 5
\end{bmatrix}
\quad
y =
\begin{bmatrix}
1 \\
0 \\
1 \\
1 \\
0
\end{bmatrix}
\quad
\boldsymbol{\beta}_{(0)} =
\begin{bmatrix}
0.1 \\
0.1 \\
0.1
\end{bmatrix}
$$

使用`statsmodels`验证你的结果 - 你可以用以下导入语句导入Probit函数

```{code-cell} ipython3
from statsmodels.discrete.discrete_model import Probit
```

请注意，本讲中开发的简单牛顿-拉夫森算法对初始值非常敏感，因此使用不同的起始值可能无法实现收敛。

```{exercise-end}
```

```{solution-start} mle_ex2
:class: dropdown
```

这是一个解决方案

```{code-cell} ipython3
X = jnp.array([[1, 2, 4], [1, 1, 1], [1, 4, 3], [1, 5, 6], [1, 3, 5]])

y = jnp.array([1, 0, 1, 1, 0])

# 对初始β值进行猜测
β = jnp.array([0.1, 0.1, 0.1])

# 创建一个Probit回归模型
prob = ProbitRegression(y=y, X=X)

# 运行牛顿-拉夫森算法
newton_raphson(prob, β)
```

```{code-cell} ipython3
# 使用statsmodels验证结果
y_numpy = np.array(y)
X_numpy = np.array(X)
print(Probit(y_numpy, X_numpy).fit().summary())
```

```{solution-end}
```