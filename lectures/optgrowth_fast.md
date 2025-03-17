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

(optgrowth)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`最优增长 II：使用Numba加速代码 <single: Optimal Growth II: Accelerating the Code with Numba>`

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

{doc}`在之前的内容中 <optgrowth>`，我们研究了一个具有单个代表性个体的随机最优增长模型。

我们使用动态规划方法求解了该模型。

在编写代码时，我们注重清晰性和灵活性。

这些都很重要，但灵活性和速度之间通常存在权衡。

原因是，当代码灵活性较低时，我们可以更容易地利用其结构特点。

（这对算法和数学问题来说普遍适用：更具体的问题具有更多的结构特征，经过思考后，可以利用这些特征获得更好的结果。）

因此，在本讲中，我们将接受较低的灵活性以获得更快的速度，使用即时(JIT)编译来加速我们的代码。

让我们从一些导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, jit
from quantecon.optimize.scalar_maximization import brent_max
```

函数`brent_max`也设计用于嵌入JIT编译代码中。

这些是SciPy中类似函数的替代方案（不幸的是，SciPy的函数不支持JIT）。

## 模型

```{index} single: Optimal Growth; Model
```

这个模型与我们在{doc}`之前的讲座 <optgrowth>`中讨论的最优增长模型相同。

我们将从对数效用函数开始：

$$
u(c) = \ln(c)
$$

我们继续假设：

* $f(k) = k^{\alpha}$
* $\phi$是当$\zeta$为标准正态分布时，$\xi := \exp(\mu + s \zeta)$的分布

我们将再次使用值函数迭代来求解这个模型。

具体来说，算法保持不变，唯一的区别在于实现本身。

和之前一样，我们将能够与真实解进行比较

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth/cd_analytical.py
```

## 计算

```{index} single: Dynamic Programming; Computation
```

我们将再次把最优增长模型的基本要素存储在一个类中。

但这次我们将使用[Numba的](https://python-programming.quantecon.org/numba.html) `@jitclass`装饰器来对我们的类进行JIT编译。

因为我们要使用Numba来编译我们的类，所以需要指定数据类型。

你会在我们的类上方看到一个名为`opt_growth_data`的列表。

与{doc}`上一讲<optgrowth>`不同，我们将生产和效用函数的具体形式直接写入类中。

这是我们为了获得更快的速度而牺牲灵活性的地方。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm.py
```

该类包含一些方法如`u_prime`，我们现在不需要但会在后续课程中使用。

### Bellman算子

我们将使用JIT编译来加速Bellman算子。

首先，这里有一个函数，根据Bellman方程{eq}`fpb30`返回特定消费选择`c`在给定状态`y`下的值。

```{code-cell} ipython3
@jit
def state_action_value(c, y, v_array, og):
    """
    Bellman方程右侧。

     * c是消费
     * y是收入
     * og是OptimalGrowthModel的一个实例
     * v_array表示网格上的值函数猜测

    """

    u, f, β, shocks = og.u, og.f, og.β, og.shocks

    v = lambda x: np.interp(x, og.grid, v_array)

    return u(c) + β * np.mean(v(f(y - c) * shocks))
```

现在我们可以实现贝尔曼算子，它用于最大化贝尔曼方程的右侧：

```{code-cell} ipython3
@jit
def T(v, og):
    """
    贝尔曼算子。

     * og 是 OptimalGrowthModel 的一个实例
     * v 是一个数组，表示价值函数的猜测值

    """

    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)

    for i in range(len(og.grid)):
        y = og.grid[i]

        # 在状态 y 下最大化贝尔曼方程的右侧
        result = brent_max(state_action_value, 1e-10, y, args=(y, v, og))
        v_greedy[i], v_new[i] = result[0], result[1]

    return v_greedy, v_new
```

我们使用`solve_model`函数进行迭代直到收敛。

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth/solve_model.py
```

让我们用默认参数计算近似解。

首先创建一个实例：

```{code-cell} ipython3
og = OptimalGrowthModel()
```

现在我们调用`solve_model`，使用`%%time`魔法指令来检查运行时间。

```{code-cell} ipython3
%%time
v_greedy, v_solution = solve_model(og)
```

你会注意到这比我们的{doc}`原始实现 <optgrowth>`要*快得多*。

下面是生成的策略与真实策略的对比图：

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2,
        alpha=0.8, label='近似策略函数')

ax.plot(og.grid, σ_star(og.grid, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='真实策略函数')

ax.legend()
plt.show()
```

再次，拟合效果非常好 --- 这是意料之中的，因为我们没有改变算法。

两种策略之间的最大绝对偏差是

```{code-cell} ipython3
np.max(np.abs(v_greedy - σ_star(og.grid, og.α, og.β)))
```

## 练习

```{exercise}
:label: ogfast_ex1

计时使用贝尔曼算子迭代20次所需的时间，从初始条件 $v(y) = u(y)$ 开始。

使用默认参数设置。
```

```{solution-start} ogfast_ex1
:class: dropdown
```

让我们设置初始条件。

```{code-cell} ipython3
v = og.u(og.grid)
```

这是时间统计：

```{code-cell} ipython3
%%time

for i in range(20):
    v_greedy, v_new = T(v, og)
    v = v_new
```

与我们对非编译版本的值函数迭代的{ref}`计时 <og_ex2>`相比，JIT编译的代码通常快一个数量级。

```{solution-end}
```

```{exercise}
:label: ogfast_ex2

修改最优增长模型以使用CRRA效用函数规范。

$$
u(c) = \frac{c^{1 - \gamma} } {1 - \gamma}
$$

将`γ = 1.5`设为默认值，并保持其他规范不变。

（注意，`jitclass`目前不支持继承，所以你必须复制类并更改相关参数和方法。）

计算最优策略的估计值，绘制图表，并与第一个最优增长讲座中{ref}`类似练习 <og_ex1>`的相同图表进行视觉比较。

同时比较执行时间。
```

```{solution-start} ogfast_ex2
:class: dropdown
```

这是我们的CRRA版本的`OptimalGrowthModel`：

```{code-cell} ipython3
:load: _static/lecture_specific/optgrowth_fast/ogm_crra.py
```

让我们创建一个实例：

```{code-cell} ipython3
og_crra = OptimalGrowthModel_CRRA()
```

现在我们调用`solve_model`，使用`%%time`魔术命令来检查运行时间。

```{code-cell} ipython3
%%time
v_greedy, v_solution = solve_model(og_crra)
```

以下是得到的策略图：

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2,
        alpha=0.6, label='近似值函数')

ax.legend(loc='lower right')
plt.show()
```

这与我们在非jit代码中得到的解决方案相符，
{ref}`in the exercises <og_ex1>`。

执行时间快了一个数量级。

```{solution-end}
```


```{exercise-start}
:label: ogfast_ex3
```

在这个练习中，我们回到原始的对数效用规范。

一旦给定最优消费策略$\sigma$，收入遵循

$$
y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1}
$$

下图显示了三种不同贴现因子（因此是三种不同策略）下该序列100个元素的模拟。

```{figure} /_static/lecture_specific/optgrowth/solution_og_ex2.png
```

在每个序列中，初始条件是$y_0 = 0.1$。

贴现因子为`discount_factors = (0.8, 0.9, 0.98)`。

我们还通过设置`s = 0.05`稍微降低了冲击的幅度。

除此之外，参数和原始设定与讲座前面讨论的对数线性模型相同。

注意，更有耐心的代理人通常拥有更高的财富。


复现该图形，允许随机性。

```{exercise-end}
```

```{solution-start} ogfast_ex3
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
def simulate_og(σ_func, og, y0=0.1, ts_length=100):
    '''
    根据消费策略σ计算时间序列。
    '''
    y = np.empty(ts_length)
    ξ = np.random.randn(ts_length-1)
    y[0] = y0
    for t in range(ts_length-1):
        y[t+1] = (y[t] - σ_func(y[t]))**og.α * np.exp(og.μ + og.s * ξ[t])
    return y
```

```{code-cell} ipython3
fig, ax = plt.subplots()

for β in (0.8, 0.9, 0.98):

    og = OptimalGrowthModel(β=β, s=0.05)

    v_greedy, v_solution = solve_model(og, verbose=False)

    # 定义最优策略函数
    σ_func = lambda x: np.interp(x, og.grid, v_greedy)
    y = simulate_og(σ_func, og)
    ax.plot(y, lw=2, alpha=0.6, label=rf'$\beta = {β}$')

ax.legend(loc='lower right')
plt.show()
```

```{solution-end}
```

