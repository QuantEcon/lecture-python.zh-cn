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

# 求职搜索 III：拟合值函数迭代

```{contents} 目录
:depth: 2
```

## 概述

在本讲座中，我们再次研究{doc}`带有离职的McCall求职搜索模型 <mccall_model_with_separation>`，但这次使用连续工资分布。

虽然我们在{doc}`第一个求职搜索讲座 <mccall_model>`的练习中已经简要考虑过连续工资分布，但在那种情况下，这种改变相对来说是微不足道的。

这是因为我们能够将问题简化为求解单个标量值（继续值）。

在这里，由于存在离职的可能性，这种改变就不那么微不足道了，因为连续工资分布导致了不可数无限的状态空间。

无限状态空间带来了额外的挑战，特别是在应用值函数迭代(VFI)时。

这些挑战将引导我们通过添加插值步骤来修改VFI。

VFI和这个插值步骤的组合被称为**拟合值函数迭代**（fitted VFI）。

拟合VFI在实践中非常常见，所以我们将花一些时间来详细研究。

我们将使用以下导入：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, float64
from numba.experimental import jitclass
```

## 算法

该模型与我们之前{doc}`研究过的带有离职的McCall模型 <mccall_model_with_separation>`相同，只是工资报价分布是连续的。

我们将从我们在{ref}`简化转换 <ast_mcm>`后得到的两个贝尔曼方程开始。

修改后以适应连续工资抽样，它们采用以下形式：

```{math}
:label: bell1mcmc

d = \int \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w') d w'
```

和

```{math}
:label: bell2mcmc

v(w) = u(w) + \beta
    \left[
        (1-\alpha)v(w) + \alpha d
    \right]
```

这里的未知量是函数 $v$ 和标量 $d$。

这些方程与我们之前处理的贝尔曼方程对的区别在于：

1. 在{eq}`bell1mcmc`中，原来对有限个工资值求和变成了对无限集合的积分。
1. {eq}`bell2mcmc`中的函数 $v$ 定义在所有 $w \in \mathbb R_+$ 上。

{eq}`bell1mcmc`中的函数 $q$ 是工资报价分布的密度函数。

其支撑集被认为等于 $\mathbb R_+$。

### 值函数迭代

理论上，我们现在应该按如下方式进行：

1. 从一个对{eq}`bell1mcmc`--{eq}`bell2mcmc`解的猜测 $v, d$ 开始。
1. 将 $v, d$ 代入{eq}`bell1mcmc`--{eq}`bell2mcmc`的右侧并计算左侧以获得更新值 $v', d'$
1. 除非满足某些停止条件，否则设置 $(v, d) = (v', d')$ 并转到步骤2。

然而，在实现这个程序之前，我们必须面对一个问题：
值函数的迭代既不能精确计算，也不能存储在计算机上。

要理解这个问题，考虑{eq}`bell2mcmc`。

即使 $v$ 是一个已知函数，存储其更新 $v'$ 的唯一方法是记录其对每个 $w \in \mathbb R_+$ 的值 $v'(w)$。

显然，这是不可能的。

### 拟合值函数迭代

我们将使用**拟合值函数迭代**来替代。

程序如下：

给定当前猜测 $v$。

现在我们只在有限个"网格"点 $w_1 < w_2 < \cdots < w_I$ 上记录函数 $v'$ 的值，然后在需要时从这些信息重构 $v'$。

更具体地说，算法将是：

(fvi_alg)=
1. 从表示初始值函数猜测在某些网格点 $\{w_i\}$ 上的值的数组 $\mathbf v$ 开始。
1. 基于 $\mathbf v$ 和 $\{ w_i\}$，通过插值或近似在状态空间 $\mathbb R_+$ 上构建函数 $v$。
1. 获取并记录每个网格点 $w_i$ 上更新函数 $v'(w_i)$ 的样本。
1. 除非满足某些停止条件，否则将此作为新数组并转到步骤1。

我们应该如何进行步骤2？

这是一个函数近似的问题，有很多方法可以处理它。

这里重要的是函数近似方案不仅要对每个 $v$ 产生良好的近似，而且还要与上述更广泛的迭代算法很好地结合。

从这两个方面来看，一个很好的选择是连续分段线性插值。

这种方法：

1. 与值函数迭代很好地结合（参见，例如，{cite}`gordon1995stable`或{cite}`stachurski2008continuous`）
1. 保持有用的形状特性，如单调性和凹凸性。

线性插值将使用[numpy.interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)来实现。

下图说明了在网格点 $0, 0.2, 0.4, 0.6, 0.8, 1$ 上对任意函数进行分段线性插值。

```{code-cell} python3
def f(x):
    y1 = 2 * np.cos(6 * x) + np.sin(14 * x)
    return y1 + 2.5

c_grid = np.linspace(0, 1, 6)
f_grid = np.linspace(0, 1, 150)

def Af(x):
    return np.interp(x, c_grid, f(c_grid))

fig, ax = plt.subplots()

ax.plot(f_grid, f(f_grid), 'b-', label='真实函数')
ax.plot(f_grid, Af(f_grid), 'g-', label='线性近似')
ax.vlines(c_grid, c_grid * 0, f(c_grid), linestyle='dashed', alpha=0.5)

ax.legend(loc="upper center")

ax.set(xlim=(0, 1), ylim=(0, 6))
plt.show()
```

## 实现

第一步是为带有离职和连续工资报价分布的McCall模型构建一个jitted类。

在这个应用中，我们将采用对数函数作为效用函数，即 $u(c) = \ln c$。

我们将采用工资的对数正态分布，当 $z$ 是标准正态分布且 $\mu, \sigma$ 是参数时，$w = \exp(\mu + \sigma z)$。

```{code-cell} python3
@jit
def lognormal_draws(n=1000, μ=2.5, σ=0.5, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws
```

这是我们的类：

```{code-cell} python3
mccall_data_continuous = [
    ('c', float64),          # 失业补偿
    ('α', float64),          # 工作离职率
    ('β', float64),          # 贴现因子
    ('w_grid', float64[:]),  # 拟合VFI的网格点
    ('w_draws', float64[:])  # 蒙特卡洛模拟的工资抽样
]

@jitclass(mccall_data_continuous)
class McCallModelContinuous:

    def __init__(self,
                 c=1,
                 α=0.1,
                 β=0.96,
                 grid_min=1e-10,
                 grid_max=5,
                 grid_size=100,
                 w_draws=lognormal_draws()):

        self.c, self.α, self.β = c, α, β

        self.w_grid = np.linspace(grid_min, grid_max, grid_size)
        self.w_draws = w_draws

    def update(self, v, d):

        # 简化名称
        c, α, β = self.c, self.α, self.β
        w = self.w_grid
        u = lambda x: np.log(x)

        # 插值数组表示的值函数
        vf = lambda x: np.interp(x, w, v)

        # 使用蒙特卡洛方法评估积分来更新d
        d_new = np.mean(np.maximum(vf(self.w_draws), u(c) + β * d))

        # 更新v
        v_new = u(w) + β * ((1 - α) * v + α * d)

        return v_new, d_new
```

然后我们返回当前迭代作为近似解。

```{code-cell} python3
@jit
def solve_model(mcm, tol=1e-5, max_iter=2000):
    """
    对贝尔曼方程迭代直至收敛

    * mcm是McCallModel的一个实例
    """

    v = np.ones_like(mcm.w_grid)    # v的初始猜测
    d = 1                           # d的初始猜测
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, d_new = mcm.update(v, d)
        error_1 = np.max(np.abs(v_new - v))
        error_2 = np.abs(d_new - d)
        error = max(error_1, error_2)
        v = v_new
        d = d_new
        i += 1

    return v, d
```

这里是一个函数`compute_reservation_wage`，它接受一个`McCallModelContinuous`的实例并返回相关的保留工资。

如果对所有 $w$ 都有 $v(w) < h$，则函数返回 np.inf。

```{code-cell} python3
@jit
def compute_reservation_wage(mcm):
    """
    通过找到最小的满足v(w) >= h的w来计算McCall模型实例的保留工资。

    如果不存在这样的w，则w_bar被设为np.inf。
    """
    u = lambda x: np.log(x)

    v, d = solve_model(mcm)
    h = u(mcm.c) + mcm.β * d

    w_bar = np.inf
    for i, wage in enumerate(mcm.w_grid):
        if v[i] > h:
            w_bar = wage
            break

    return w_bar
```

练习要求你探索解决方案以及它如何随参数变化。

## 练习

```{exercise}
:label: mfv_ex1

使用上面的代码探索当工资参数 $\mu$ 变化时保留工资会发生什么变化。

使用默认参数和 `mu_vals = np.linspace(0.0, 2.0, 15)` 中的 $\mu$ 值。

对保留工资的影响是否如你所预期？
```

```{solution-start} mfv_ex1
:class: dropdown
```

这是一个解决方案：

```{code-cell} python3
mcm = McCallModelContinuous()
mu_vals = np.linspace(0.0, 2.0, 15)
w_bar_vals = np.empty_like(mu_vals)

fig, ax = plt.subplots()

for i, m in enumerate(mu_vals):
    mcm.w_draws = lognormal_draws(μ=m)
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set(xlabel='均值', ylabel='保留工资')
ax.plot(mu_vals, w_bar_vals, label=r'$\bar w$ 作为 $\mu$ 的函数')
ax.legend()

plt.show()
```

不出所料，当报价分布向右移动时，求职者更倾向于等待。

```{solution-end}
```

```{exercise}
:label: mfv_ex2

让我们现在考虑求职者如何对波动性的增加做出反应。

为了理解这一点，当工资报价分布在 $(m - s, m + s)$ 上均匀分布且 $s$ 变化时，计算保留工资。

这里的想法是我们保持均值不变，而扩大支撑集。

（这是一种*均值保持扩展*。）

使用 `s_vals = np.linspace(1.0, 2.0, 15)` 和 `m = 2.0`。

说明你预期保留工资如何随 $s$ 变化。

现在计算它。这是否如你所预期？
```

```{solution-start} mfv_ex2
:class: dropdown
```

这是一个解决方案：

```{code-cell} python3
mcm = McCallModelContinuous()
s_vals = np.linspace(1.0, 2.0, 15)
m = 2.0
w_bar_vals = np.empty_like(s_vals)

fig, ax = plt.subplots()

for i, s in enumerate(s_vals):
    a, b = m - s, m + s
    mcm.w_draws = np.random.uniform(low=a, high=b, size=10_000)
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set(xlabel='波动性', ylabel='保留工资')
ax.plot(s_vals, w_bar_vals, label=r'$\bar w$ 作为工资波动性的函数')
ax.legend()

plt.show()
```

保留工资随波动性增加而增加。

人们可能会认为更高的波动性会使求职者更倾向于接受给定的报价，因为这代表确定性，而等待代表风险。

但求职搜索就像持有一个期权：工人只面临上行风险（因为在自由市场中，没有人能强迫他们接受一个糟糕的报价）。

更大的波动性意味着更高的上行潜力，这鼓励求职者等待。

```{solution-end}
```
