---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`最优增长 III：时间迭代 <single: Optimal Growth III: Time Iteration>`

```{contents} 目录
:depth: 2
```

除Anaconda已包含的库外，本讲义还需要安装以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```
## 概述

在本讲中，我们将继续此前对{doc}`随机最优增长模型 <optgrowth>`的研究。

在那一讲中，我们使用价值函数迭代求解了相关的动态规划问题。

这种技术的优点在于其广泛的适用性。

然而，在数值问题中，我们常常可以通过推导出更贴合具体应用的算法，以获得更高的效率。

随机最优增长模型具备丰富的结构可供利用，尤其当我们对原始要素施加某些凹性与光滑性假设时。

我们将利用这一结构，获得基于欧拉方程的方法。

这将是我们在基础讲义{doc}`吃蛋糕问题 <cake_eating_numerical>`中所考虑的时间迭代方法的扩展。

在{doc}`下一讲 <egm_policy_iter>`中，我们将看到，时间迭代可以进一步调整，以获得更高的效率。

接下来，让我们从导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from quantecon.optimize import brentq
from numba import jit
```
## 欧拉方程

我们的第一步是推导欧拉方程，这是此前在{doc}`吃蛋糕问题 <cake_eating_problem>`中得到的欧拉方程的推广。

我们采用{doc}`随机增长模型 <optgrowth>`中的模型设定，并加入以下假设：

1. $u$ 和 $f$ 是连续可微且严格凹函数；
1. $f(0) = 0$；
1. $\lim_{c \to 0} u'(c) = \infty$ 且 $\lim_{c \to \infty} u'(c) = 0$；
1. $\lim_{k \to 0} f'(k) = \infty$ 且 $\lim_{k \to \infty} f'(k) = 0$。

最后两个条件通常被称为**Inada条件**。

回顾贝尔曼方程：

```{math}
:label: cpi_fpb30

v^*(y) = \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
    \right\}
\quad \forall y \in \mathbb R_+
```

令最优消费策略记为 $\sigma^*$。

我们知道 $\sigma^*$ 是一个 $v^*$-逐期最优的，因此 $\sigma^*(y)$ 是{eq}`cpi_fpb30`中的最大化解。

上述条件表明：

* $\sigma^*$ 是随机最优增长模型的唯一最优策略；
* 该最优策略是连续的、严格递增的，并且是**内部解**，即对于所有严格正的 $y$，都有 $0 < \sigma^*(y) < y$；
* 价值函数是严格凹的且连续可微的，并满足：

```{math}
:label: cpi_env

(v^*)'(y) = u' (\sigma^*(y) ) := (u' \circ \sigma^*)(y)
```

最后一个结果被称为**包络条件**，因为它与[包络定理](https://baike.baidu.com/item/%E5%8C%85%E7%BB%9C%E5%AE%9A%E7%90%86/5746200)有关。

要理解为什么{eq}`cpi_env`成立，可以将贝尔曼方程写成等价形式

$$
v^*(y) = \max_{0 \leq k \leq y}
    \left\{
        u(y-k) + \beta \int v^*(f(k) z) \phi(dz)
    \right\},
$$

对 $y$ 求导，并在最优解处求值，即可得到{eq}`cpi_env`。
（[EDTC](https://johnstachurski.net/edtc.html)第12.1节给出了这些结果的完整证明，许多其他教材中也可找到类似讨论。）

价值函数的可微性和最优策略的内部性意味着，最优消费决策满足与{eq}`cpi_fpb30`相关的一阶条件，即

```{math}
:label: cpi_foc

u'(\sigma^*(y)) = \beta \int (v^*)'(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

将{eq}`cpi_env`和该一阶条件{eq}`cpi_foc`结合，得到**欧拉方程**：

```{math}
:label: cpi_euler

(u'\circ \sigma^*)(y)
= \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

我们可以将欧拉方程视为一个函数方程：

```{math}
:label: cpi_euler_func

(u'\circ \sigma)(y)
= \beta \int (u'\circ \sigma)(f(y - \sigma(y)) z) f'(y - \sigma(y)) z \phi(dz)
```
其中 $\sigma$ 为内部消费策略，其解之一即为最优策略 $\sigma^*$。

我们的目标是求解函数方程 {eq}`cpi_euler_func` 从而获得 $\sigma^*$。

### Coleman-Reffett 算子

回顾贝尔曼算子：

```{math}
:label: fcbell20_coleman

Tv(y) := \max_{0 \leq c \leq y}
\left\{
    u(c) + \beta \int v(f(y - c) z) \phi(dz)
\right\}
```

正如我们引入贝尔曼算子来求解贝尔曼方程一样，我们现在将引入一个作用于策略空间的算子，用于帮助我们求解欧拉方程。

该算子 $K$ 将作用于所有连续、严格递增且为内部解的 $\sigma \in \Sigma$。

此后我们将这类策略集合记为 $\mathscr P$。

1. 算子 $K$ 的自变量是一个 $\sigma \in \mathscr P$；
1. 返回一个新函数 $K\sigma$，其中 $(K\sigma)(y)$ 是求解以下方程的 $c \in (0, y)$:

```{math}
:label: cpi_coledef

u'(c)
= \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

我们称这个算子为**Coleman-Reffett算子**，以此致敬{cite}`Coleman1990`和{cite}`Reffett1996`的研究工作。

本质上，$K\sigma$ 表示在给定未来消费策略为 $\sigma$ 时，欧拉方程指导你今天应选择的消费策略。

值得注意的是：依据构造，算子 $K$ 的不动点恰好与函数方程{eq}`cpi_euler_func`的解相一致。

特别地，最优政策 $\sigma^*$ 是一个不动点。

事实上，对于固定的 $y$，$(K\sigma^*)(y)$ 是满足以下方程的 $c$：

$$
u'(c)
= \beta \int (u' \circ \sigma^*) (f(y - c) z ) f'(y - c) z \phi(dz)
$$

根据欧拉方程，该解正是 $\sigma^*(y)$。

### Coleman-Reffett算子是否定义良好？

特别地，是否总存在唯一的 $c \in (0, y)$ 使其满足{eq}`cpi_coledef`？

在我们的假设条件下，答案是肯定的。

对于任何 $\sigma \in \mathscr P$，{eq}`cpi_coledef` 的右侧：

* 在 $(0, y)$ 上关于 $c$ 是连续且严格递增的；
* 当 $c \uparrow y$ 时趋向于 $+\infty$。

{eq}`cpi_coledef` 的左侧：

* 在 $(0, y)$ 上关于 $c$ 是连续且严格递减的；
* 当 $c \downarrow 0$ 时趋向于 $+\infty$。

绘制这些曲线并利用上述信息，二者在 $c \in (0, y)$ 上恰好有且仅有一次交点。

进一步分析可得：若 $\sigma \in \mathscr P$，则 $K \sigma \in \mathscr P$。

### 与价值函数迭代（VFI）的比较（理论部分）

可以证明，算子 $K$ 的迭代与贝尔曼算子的迭代之间存在紧密关系。

从数学上讲，这两个算子是*拓扑共轭的*。

简单来说，这意味着：如果一个算子的迭代收敛，那么另一个算子的迭代也会收敛，反之亦然。

此外，在理论上可以认为二者的收敛速率是相同的。

然而，事实证明，算子 $K$ 在数值计算上更加稳定，因此在我们考虑的应用中更加高效。

下面给出若干示例。

## 实现

与{doc}`上一讲 <optgrowth_fast>`一样，我们继续假设：

* $u(c) = \ln c$；
* $f(k) = k^{\alpha}$；
* $\phi$ 是 $\xi := \exp(\mu + s \zeta)$ 的分布，且 $\zeta$ 服从标准正态分布。

这一设定使我们能够将数值结果与解析解进行比较。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```
如上所述，我们的目标是通过时间迭代来求解模型，即对算子 $K$ 进行迭代。

为此，我们需要函数 $u', f$ 和 $f'$。

我们将使用{doc}`上一讲 <optgrowth_fast>`中构建的`OptimalGrowthModel`类来实现。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```
接下来我们实现一个名为`euler_diff`的方法，该方法返回：

```{math}
:label: euler_diff

u'(c) - \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

```{code-cell} ipython
@jit
def euler_diff(c, σ, y, og):
    """
    设置一个函数，使得关于c的根，
    在给定y和σ的情况下，等于Kσ(y)。

    """

    β, shocks, grid = og.β, og.shocks, og.grid
    f, f_prime, u_prime = og.f, og.f_prime, og.u_prime

    # 首先通过插值将σ转换为函数
    σ_func = lambda x: np.interp(x, grid, σ)

    # 现在设置我们需要找到根的函数
    vals = u_prime(σ_func(f(y - c) * shocks)) * f_prime(y - c) * shocks
    return u_prime(c) - β * np.mean(vals)
```
函数`euler_diff`通过蒙特卡洛方法计算积分，并使用线性插值对函数进行近似。

我们将使用求根算法来求解式{eq}`euler_diff`，给定状态 $y$ 和 $σ$，寻找当前期消费 $c$。

下面是实现该求根算法的算子 $K$。

```{code-cell} ipython3
@jit
def K(σ, og):
    """
    Coleman-Reffett算子

    这里og是OptimalGrowthModel的一个实例。
    """

    β = og.β
    f, f_prime, u_prime = og.f, og.f_prime, og.u_prime
    grid, shocks = og.grid, og.shocks

    σ_new = np.empty_like(σ)
    for i, y in enumerate(grid):
        # 在y处求解最优c
        c_star = brentq(euler_diff, 1e-10, y-1e-10, args=(σ, y, og))[0]
        σ_new[i] = c_star

    return σ_new
```
### 测试

接下来，我们生成一个实例并绘制算子 $K$ 的若干次迭代结果，初始条件取 $σ(y) = y$。

```{code-cell} ipython3
og = OptimalGrowthModel()
grid = og.grid

n = 15
σ = grid.copy()  # 设置初始条件

fig, ax = plt.subplots()
lb = '初始条件 $\sigma(y) = y$'
ax.plot(grid, σ, color=plt.cm.jet(0), alpha=0.6, label=lb)

for i in range(n):
    σ = K(σ, og)
    ax.plot(grid, σ, color=plt.cm.jet(i / n), alpha=0.6)

# 再更新一次并用黑色绘制最后一次迭代
σ = K(σ, og)
ax.plot(grid, σ, color='k', alpha=0.8, label='最后一次迭代')

ax.legend()

plt.show()
```
我们可以看到，迭代过程快速收敛到一个极限，该极限与我们在{doc}`上一讲<optgrowth_fast>`中得到的解非常相似。

这里给出一个名为`solve_model_time_iter`的函数，它接收一个`OptimalGrowthModel`实例作为输入，并通过时间迭代法返回最优策略的近似解。

```{code-cell} ipython3
:load: _static/lecture_specific/coleman_policy_iter/solve_time_iter.py
```
让我们运行它：

```{code-cell} ipython3
σ_init = np.copy(og.grid)
σ = solve_model_time_iter(og, σ_init)
```
这是得到的策略与真实策略的对比图：

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(og.grid, σ, lw=2,
        alpha=0.8, label='近似策略函数')

ax.plot(og.grid, σ_star(og.grid, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='真实策略函数')

ax.legend()
plt.show()
```
再次说明，拟合效果非常好。

两种策略之间的最大绝对偏差是：

```{code-cell} ipython3
np.max(np.abs(σ - σ_star(og.grid, og.α, og.β)))
```
收敛所需时间如下：

```{code-cell} ipython3
%%timeit -n 3 -r 1
σ = solve_model_time_iter(og, σ_init, verbose=False)
```
收敛速度非常快，甚至优于我们{doc}`基于JIT编译的价值函数迭代<optgrowth_fast>`。

总的来说，我们发现，至少对于该模型而言，时间迭代法在效率与准确度上均展现出高度优势。

## 练习

```{exercise}
:label: cpi_ex1

求解具有CRRA效用函数的模型

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

其中`γ = 1.5`。

计算并绘制最优策略。
```

```{solution-start} cpi_ex1
:class: dropdown
```

我们使用{doc}`VFI讲义<optgrowth_fast>`中的`OptimalGrowthModel_CRRA`类。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm_crra.py
```

创建一个实例：

```{code-cell} ipython3
og_crra = OptimalGrowthModel_CRRA()
```

求解并绘制策略：

```{code-cell} ipython3
%%time
σ = solve_model_time_iter(og_crra, σ_init)


fig, ax = plt.subplots()

ax.plot(og.grid, σ, lw=2,
        alpha=0.8, label='近似策略函数')

ax.legend()
plt.show()
```

```{solution-end}
```
