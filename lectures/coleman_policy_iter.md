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

除了Anaconda中已有的库外，本讲座还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```
## 概述

在本讲中，我们将继续我们{doc}`之前 <optgrowth>`对随机最优增长模型的研究。

在那节课中，我们使用值函数迭代求解了相关的动态规划问题。

这种技术的优点在于其广泛的适用性。

然而，对于数值问题，我们通常可以通过推导专门针对具体应用的方法来获得更高的效率。

随机最优增长模型有大量可供利用的结构，特别是当我们对原始函数采用一些凹性和光滑性假设时。

我们将利用这种结构来获得一个基于欧拉方程的方法。

这将是对我们在{doc}`蛋糕食用问题 <cake_eating_numerical>`的基础讲座中考虑的时间迭代法的扩展。

在{doc}`后续讲座 <egm_policy_iter>`中，我们将看到时间迭代可以进一步调整以获得更高的效率。
让我们从一些导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import numpy as np
from quantecon.optimize import brentq
from numba import jit
```
## 欧拉方程

我们的第一步是推导欧拉方程，这是对我们在{doc}`蛋糕食用问题讲座 <cake_eating_problem>`中得到的欧拉方程的推广。

我们采用{doc}`随机增长模型讲座 <optgrowth>`中设定的模型，并添加以下假设：

1. $u$和$f$是连续可微且严格凹函数
1. $f(0) = 0$
1. $\lim_{c \to 0} u'(c) = \infty$且$\lim_{c \to \infty} u'(c) = 0$
1. $\lim_{k \to 0} f'(k) = \infty$且$\lim_{k \to \infty} f'(k) = 0$

最后两个条件通常被称为**稻田条件**。

回顾贝尔曼方程

```{math}
:label: cpi_fpb30

v^*(y) = \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
    \right\}
\quad \text{for all} \quad
y \in \mathbb R_+
```

让最优消费策略用$\sigma^*$表示。

我们知道$\sigma^*$是一个$v^*$-贪婪策略，因此$\sigma^*(y)$是{eq}`cpi_fpb30`中的最大化值。
上述条件表明

* $\sigma^*$ 是随机最优增长模型的唯一最优策略
* 最优策略是连续的、严格递增的，并且是**内部的**，即对于所有严格正的 $y$，都有 $0 < \sigma^*(y) < y$，并且
* 值函数是严格凹的且连续可微的，满足

```{math}
:label: cpi_env

(v^*)'(y) = u' (\sigma^*(y) ) := (u' \circ \sigma^*)(y)
```

最后这个结果被称为**包络条件**，因为它与[包络定理](https://en.wikipedia.org/wiki/Envelope_theorem)有关。

要理解为什么{eq}`cpi_env`成立，可以将贝尔曼方程写成等价形式

$$
v^*(y) = \max_{0 \leq k \leq y}
    \left\{
        u(y-k) + \beta \int v^*(f(k) z) \phi(dz)
    \right\},
$$

对 $y$ 求导，然后在最优点处求值即可得到{eq}`cpi_env`。
（[EDTC](https://johnstachurski.net/edtc.html)第12.1节包含这些结果的完整证明，许多其他教材中也可以找到密切相关的讨论。）

价值函数的可微性和最优策略的内部性意味着最优消费满足与{eq}`cpi_fpb30`相关的一阶条件，即

```{math}
:label: cpi_foc

u'(\sigma^*(y)) = \beta \int (v^*)'(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

将{eq}`cpi_env`和一阶条件{eq}`cpi_foc`结合得到**欧拉方程**

```{math}
:label: cpi_euler

(u'\circ \sigma^*)(y)
= \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

我们可以将欧拉方程视为一个泛函方程

```{math}
:label: cpi_euler_func

(u'\circ \sigma)(y)
= \beta \int (u'\circ \sigma)(f(y - \sigma(y)) z) f'(y - \sigma(y)) z \phi(dz)
```
对于内部消费策略 $\sigma$，其中一个解就是最优策略 $\sigma^*$。

我们的目标是求解函数方程 {eq}`cpi_euler_func` 从而获得 $\sigma^*$。

### Coleman-Reffett 算子

回顾 Bellman 算子

```{math}
:label: fcbell20_coleman

Tv(y) := \max_{0 \leq c \leq y}
\left\{
    u(c) + \beta \int v(f(y - c) z) \phi(dz)
\right\}
```

正如我们引入 Bellman 算子来求解 Bellman 方程一样，我们现在将引入一个作用于策略的算子来帮助我们求解欧拉方程。

这个算子 $K$ 将作用于所有连续、严格递增且内部的 $\sigma \in \Sigma$ 的集合上。

此后我们用 $\mathscr P$ 表示这个策略集合

1. 算子 $K$ 以 $\sigma \in \mathscr P$ 为参数
1. 返回一个新函数 $K\sigma$，其中 $K\sigma(y)$ 是求解以下方程的 $c \in (0, y)$。

```{math}
:label: cpi_coledef

u'(c)
= \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

我们称这个算子为**Coleman-Reffett算子**，以此致敬{cite}`Coleman1990`和{cite}`Reffett1996`的研究工作。

本质上，当你的未来消费政策是$\sigma$时，$K\sigma$是欧拉方程告诉你今天应该选择的消费政策。

关于$K$需要注意的重要一点是，根据其构造，其不动点恰好与函数方程{eq}`cpi_euler_func`的解coincide。

特别地，最优政策$\sigma^*$就是一个不动点。

事实上，对于固定的$y$，$K\sigma^*(y)$是解决以下方程的$c$：

$$
u'(c)
= \beta \int (u' \circ \sigma^*) (f(y - c) z ) f'(y - c) z \phi(dz)
$$

根据欧拉方程，这恰好就是$\sigma^*(y)$。

### Coleman-Reffett算子是否良定义？

特别地，是否总存在唯一的$c \in (0, y)$来解决{eq}`cpi_coledef`？

在我们的假设条件下，答案是肯定的。
对于任何 $\sigma \in \mathscr P$，{eq}`cpi_coledef` 右侧

* 在 $(0, y)$ 上关于 $c$ 是连续且严格递增的
* 当 $c \uparrow y$ 时趋向于 $+\infty$

{eq}`cpi_coledef` 左侧

* 在 $(0, y)$ 上关于 $c$ 是连续且严格递减的
* 当 $c \downarrow 0$ 时趋向于 $+\infty$

绘制这些曲线并利用上述信息，你会确信当 $c$ 在 $(0, y)$ 范围内变化时，这些曲线恰好相交一次。

通过更深入的分析，可以进一步证明当 $\sigma \in \mathscr P$ 时，$K \sigma \in \mathscr P$。

### 与值函数迭代的比较（理论）

可以证明 $K$ 的迭代与贝尔曼算子的迭代之间存在紧密关系。

从数学角度来说，这两个算子是*拓扑共轭的*。

简单来说，这意味着如果一个算子的迭代收敛，那么另一个算子的迭代也会收敛，反之亦然。

而且，至少从理论上讲，它们的收敛速度是相同的。
然而，事实证明算子 $K$ 在数值计算上更加稳定，因此在我们考虑的应用中更加高效。

下面给出一些例子。

## 实现

如同我们在{doc}`之前的研究 <optgrowth_fast>`中一样，我们继续假设

* $u(c) = \ln c$
* $f(k) = k^{\alpha}$
* $\phi$ 是当 $\zeta$ 为标准正态分布时 $\xi := \exp(\mu + s \zeta)$ 的分布

这将使我们能够将结果与解析解进行比较

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```
如上所述，我们计划使用时间迭代来求解模型，这意味着要使用算子$K$进行迭代。

为此，我们需要访问函数$u'$和$f, f'$。

这些函数在我们在{doc}`之前的讲座 <optgrowth_fast>`中构建的`OptimalGrowthModel`类中可用。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```
现在我们实现一个名为`euler_diff`的方法，它返回

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
函数`euler_diff`通过蒙特卡洛方法计算积分，并使用线性插值来近似函数。

我们将使用根查找算法来求解{eq}`euler_diff`，在给定状态$y$和$σ$（当前策略猜测值）的情况下求解$c$。

以下是实现根查找步骤的算子$K$。

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

让我们生成一个实例并绘制$K$的一些迭代结果，从$σ(y) = y$开始。

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
我们可以看到迭代过程快速收敛到一个极限值，这与我们在{doc}`上一讲<optgrowth_fast>`中得到的解相似。

这里有一个名为`solve_model_time_iter`的函数，它接收一个`OptimalGrowthModel`实例作为输入，并通过时间迭代法返回最优策略的近似解。

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

两种策略之间的最大绝对偏差是

```{code-cell} ipython3
np.max(np.abs(σ - σ_star(og.grid, og.α, og.β)))
```
需要多长时间才能收敛？

```{code-cell} ipython3
%%timeit -n 3 -r 1
σ = solve_model_time_iter(og, σ_init, verbose=False)
```
与我们的{doc}`JIT编译的值函数迭代<optgrowth_fast>`相比，收敛速度非常快。

总的来说，我们发现时间迭代方法对于这个模型来说提供了很高的效率和准确性。

## 练习

```{exercise}
:label: cpi_ex1

用CRRA效用函数求解模型

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

设定`γ = 1.5`。

计算并绘制最优策略。
```

```{solution-start} cpi_ex1
:class: dropdown
```

我们使用{doc}`VFI讲座<optgrowth_fast>`中的`OptimalGrowthModel_CRRA`类。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm_crra.py
```

让我们创建一个实例：

```{code-cell} ipython3
og_crra = OptimalGrowthModel_CRRA()
```

现在我们求解并绘制策略：

```{code-cell} ipython3
%%time
σ = solve_model_time_iter(og_crra, σ_init)


fig, ax = plt.subplots()

ax.plot(og.grid, σ, lw=2,
        alpha=0.8, label='approximate policy function')

ax.legend()
plt.show()
```

```{solution-end}
```
