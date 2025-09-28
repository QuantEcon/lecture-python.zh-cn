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

# {index}`最优增长 IV：内生网格法 <single: Optimal Growth IV: The Endogenous Grid Method>`

```{contents} 目录
:depth: 2
```

## 概述

在之前，我们使用以下方法求解了随机最优增长模型：

1. {doc}`价值函数迭代 <optgrowth_fast>`
1. {doc}`基于欧拉方程的时间迭代 <coleman_policy_iter>`

我们发现时间迭代在准确性和效率方面都明显更好。

在本讲义中，我们将介绍时间迭代的一种巧妙变体，称为**内生网格法**（EGM）。

EGM是由[Chris Carroll](http://www.econ2.jhu.edu/people/ccarroll/)发明的一种用于实现政策迭代的数值方法。

该方法的原始参考文献是{cite}`Carroll2006`。

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from numba import jit
```

## 核心思想

我们首先回顾理论背景，然后说明数值方法如何融入其中。

### 理论

我们沿用{doc}`时间迭代 <coleman_policy_iter>`中的模型设定，遵循相同的术语和符号。

欧拉方程为：

```{math}
:label: egm_euler

(u'\circ \sigma^*)(y)
= \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)
```

如前所示，Coleman-Reffett算子是一个非线性算子 $K$，其设计使得 $\sigma^*$ 是 $K$ 的不动点。

该算子以一个连续且严格递增的消费策略 $\sigma \in \Sigma$ 作为自变量。

它返回一个新函数 $K \sigma$，其中 $(K \sigma)(y)$ 是满足以下方程的 $c \in (0, \infty)$：

```{math}
:label: egm_coledef

u'(c)
= \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)
```

### 外生网格

如{doc}`时间迭代 <coleman_policy_iter>`中所述，为了在计算机上实现该方法，我们需要数值近似。

具体来说，我们通过在有限网格上取值的方式表示策略函数。

在需要时，使用插值或其他方法从该有限表示中重建原函数。

在{doc}`时间迭代 <coleman_policy_iter>`中，为了获得更新后的消费策略的有限表示，我们：

* 固定一组收入点 $\{y_i\}$；
* 使用{eq}`egm_coledef`与求根算法，计算与每个 $y_i$ 对应的消费值 $c_i$。

每个 $c_i$ 被解释为函数 $K \sigma$ 在 $y_i$ 处的值。

因此，有了点集 $\{y_i, c_i\}$，我们可以通过近似重建 $K \sigma$。

然后继续迭代...

### 内生网格

上述方法需要通过求根算法来确定与给定收入值 $y_i$ 对应的消费水平 $c_i$。

然而，求根运算的代价较高，因为其通常涉及大量函数求值。

正如Carroll {cite}`Carroll2006`所指出的，如果 $y_i$ 是内生选择的，则可避免这一过程。

唯一需要的假设是：$u'$ 在 $(0, \infty)$ 上是可逆的。

令 $(u')^{-1}$ 为 $u'$ 的反函数。

其核心思想如下：

* 首先，固定一个关于资本($k = y - c$)的*外生*网格 $\{k_i\}$；
* 接着，根据以下公式求得 $c_i$:

```{math}
:label: egm_getc

c_i =
(u')^{-1}
\left\{
    \beta \int (u' \circ \sigma) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
\right\}
```

* 最后，对每个 $c_i$，设定 $y_i = c_i + k_i$。

显然，每个通过上述方式构建的 $(y_i, c_i)$ 都满足{eq}`egm_coledef`。

有了点集 $\{y_i, c_i\}$，我们即可通过近似方法重建 $K \sigma$。

新的EGM算法的关键在于：网格 $\{y_i\}$ 是**内生**决定的。

## 实现

与{doc}`时间迭代 <coleman_policy_iter>`相同，我们从一个简单设定开始：

* $u(c) = \ln c$；
* 生产函数是柯布-道格拉斯形式；
* 冲击项服从对数正态分布。

这一设定使我们能够将数值解与解析解进行对比。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```

我们重用 `OptimalGrowthModel` 类

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```

### 算子

以下给出使用EGM方法实现算子 $K$ 的代码：

```{code-cell} ipython3
@jit
def K(σ_array, og):
    """
    使用EGM的Coleman-Reffett算子

    """

    # 简化命名
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

值得注意的是，该算法不需要求根算法。

### 测试

首先，我们创建一个实例。

```{code-cell} ipython3
og = OptimalGrowthModel()
grid = og.grid
```

下面是求解程序：

```{code-cell} ipython3
:load: _static/lecture_specific/coleman_policy_iter/solve_time_iter.py
```

让我们运行它：

```{code-cell} ipython3
σ_init = np.copy(grid)
σ = solve_model_time_iter(og, σ_init)
```

以下是得到的策略与真实策略的比较：

```{code-cell} ipython3
y = grid + σ  # y_i = k_i + c_i

fig, ax = plt.subplots()

ax.plot(y, σ, lw=2,
        alpha=0.8, label='近似策略函数')

ax.plot(y, σ_star(y, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='真实策略函数')

ax.legend()
plt.show()
```

两个策略之间的最大绝对偏差是

```{code-cell} ipython3
np.max(np.abs(σ - σ_star(y, og.α, og.β)))
```

收敛所需的时间为：

```{code-cell} ipython3
%%timeit -n 3 -r 1
σ = solve_model_time_iter(og, σ_init, verbose=False)
```

相较于已被证明高度高效的时间迭代法，内生网格法（EGM）在保持精度不变的前提下，进一步显著减少了运行时间。

其主要原因在于该方法不需要进行数值求根步骤。

因此，我们能够在给定参数下以极高的速度求解最优增长模型。