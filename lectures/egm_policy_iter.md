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

# {index}`最优增长 IV：内生网格方法 <single: Optimal Growth IV: The Endogenous Grid Method>`

```{contents} 目录
:depth: 2
```

## 概述

之前，我们使用以下方法求解了随机最优增长模型：

1. {doc}`值函数迭代 <optgrowth_fast>`
1. {doc}`基于欧拉方程的时间迭代 <coleman_policy_iter>`

我们发现时间迭代在精确度和效率方面都明显更好。

在本讲座中，我们将研究时间迭代的一个巧妙变体，称为**内生网格方法**（EGM）。

EGM是由[Chris Carroll](http://www.econ2.jhu.edu/people/ccarroll/)发明的一种实现策略迭代的数值方法。

原始参考文献是{cite}`Carroll2006`。

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
```

## 核心思想

让我们先回顾一下理论，然后看看如何进行数值计算。

### 理论

采用{doc}`时间迭代讲座 <coleman_policy_iter>`中设定的模型，遵循相同的术语和符号。

欧拉方程是

```{math}
:label: egm_euler

(u'\circ \sigma^*)(y)
= \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

如我们所见，Coleman-Reffett算子是一个非线性算子 $K$，其设计使得 $\sigma^*$ 是 $K$ 的不动点。

它以连续严格递增的消费策略 $\sigma \in \Sigma$ 作为参数。

它返回一个新函数 $K \sigma$，其中 $(K \sigma)(y)$ 是满足以下方程的 $c \in (0, \infty)$：

```{math}
:label: egm_coledef

u'(c)
= \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

### 外生网格

如{doc}`时间迭代讲座 <coleman_policy_iter>`中所讨论的，要在计算机上实现该方法，我们需要数值近似。

具体来说，我们通过有限网格上的一组值来表示策略函数。

在必要时，使用插值或其他方法从这种表示中重构函数本身。

{doc}`之前 <coleman_policy_iter>`，为了获得更新后消费策略的有限表示，我们：

* 固定一个收入点网格 $\{y_i\}$
* 使用{eq}`egm_coledef`和一个寻根程序计算对应于每个 $y_i$ 的消费值 $c_i$

每个 $c_i$ 被解释为函数 $K \sigma$ 在 $y_i$ 处的值。

因此，有了点 $\{y_i, c_i\}$，我们可以通过近似重构 $K \sigma$。

然后继续迭代...

### 内生网格

上述方法需要一个寻根程序来找到对应于给定收入值 $y_i$ 的 $c_i$。

寻根计算成本很高，因为它通常涉及大量的函数求值。

正如Carroll {cite}`Carroll2006`指出的，如果 $y_i$ 是内生选择的，我们可以避免这一点。

唯一需要的假设是 $u'$ 在 $(0, \infty)$ 上可逆。

令 $(u')^{-1}$ 为 $u'$ 的逆函数。

思路是这样的：

* 首先，我们为资本（$k = y - c$）固定一个*外生*网格 $\{k_i\}$。
* 然后通过以下方式获得 $c_i$：

```{math}
:label: egm_getc

c_i =
(u')^{-1}
\left\{
    \beta \int (u' \circ \sigma) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
\right\}
```

* 最后，对于每个 $c_i$，我们设定 $y_i = c_i + k_i$。

显然，以这种方式构造的每个 $(y_i, c_i)$ 对都满足{eq}`egm_coledef`。

有了点 $\{y_i, c_i\}$，我们可以像之前一样通过近似重构 $K \sigma$。

EGM这个名称来源于网格 $\{y_i\}$ 是**内生**确定的这一事实。

## 实现

如{doc}`之前 <coleman_policy_iter>`一样，我们将从一个简单的设定开始，其中：

* $u(c) = \ln c$，
* 生产函数是Cobb-Douglas形式，且
* 冲击是对数正态分布的。

这将允许我们与解析解进行比较

```{code-cell} python3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```

我们重用`OptimalGrowthModel`类

```{code-cell} python3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```

### 算子

这里是使用上述EGM实现的 $K$ 算子。

```{code-cell} python3
@jit
def K(σ_array, og):
    """
    使用EGM的Coleman-Reffett算子

    """

    # 简化名称
    f, β = og.f, og.β
    f_prime, u_prime = og.f_prime, og.u_prime
    u_prime_inv = og.u_prime_inv
    grid, shocks = og.grid, og.shocks

    # 确定内生网格
    y = grid + σ_array  # y_i = k_i + c_i

    # 使用内生网格进行策略的线性插值
    σ = lambda x: np.interp(x, y, σ_array)

    # 为新的消费数组分配内存
    c = np.empty_like(grid)

    # 求解更新后的消费值
    for i, k in enumerate(grid):
        vals = u_prime(σ(f(k) * shocks)) * f_prime(k) * shocks
        c[i] = u_prime_inv(β * np.mean(vals))

    return c
```

注意这里没有任何寻根算法。

### 测试

首先我们创建一个实例。

```{code-cell} python3
og = OptimalGrowthModel()
grid = og.grid
```

这是我们的求解程序：

```{code-cell} python3
:load: _static/lecture_specific/coleman_policy_iter/solve_time_iter.py
```

让我们调用它：

```{code-cell} python3
σ_init = np.copy(grid)
σ = solve_model_time_iter(og, σ_init)
```

这是结果策略与真实策略的对比图：

```{code-cell} python3
y = grid + σ  # y_i = k_i + c_i

fig, ax = plt.subplots()

ax.plot(y, σ, lw=2,
        alpha=0.8, label='近似策略函数')

ax.plot(y, σ_star(y, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='真实策略函数')

ax.legend()
plt.show()
```

两个策略之间的最大绝对偏差是：

```{code-cell} python3
np.max(np.abs(σ - σ_star(y, og.α, og.β)))
```

收敛需要多长时间？

```{code-cell} python3
%%timeit -n 3 -r 1
σ = solve_model_time_iter(og, σ_init, verbose=False)
```

相对于已经被证明非常高效的时间迭代，EGM在不影响精确度的情况下进一步缩短了运行时间。

这是因为这个算法没有使用寻根算法。

我们现在掌握了一个可以快速求解最优增长模型的工具。


