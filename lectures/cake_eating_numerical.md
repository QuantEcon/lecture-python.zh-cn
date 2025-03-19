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

# 蛋糕食用问题 II：数值方法

```{contents} 目录
:depth: 2
```

## 概述

在本讲中，我们继续研究{doc}`蛋糕食用问题 <cake_eating_problem>`。

本讲的目标是使用数值方法求解该问题。

起初这可能看起来没有必要，因为我们已经通过分析方法得到了最优策略。

然而，蛋糕食用问题过于简单，如果不加修改就不太实用，而一旦我们开始修改问题，数值方法就变得必不可少。

因此，现在引入数值方法并在这个简单问题上测试它们是有意义的。

由于我们知道分析解，这将使我们能够评估不同数值方法的准确性。

我们将使用以下导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from scipy.optimize import minimize_scalar, bisect
```
## 回顾模型

在开始之前，你可能想要{doc}`回顾详细内容 <cake_eating_problem>`。

特别要回顾的是贝尔曼方程：

```{math}
:label: bellman-cen

v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
\quad \text{for all } x \geq 0.
```

其中$u$是CRRA效用函数。

价值函数和最优策略的解析解如下所示。

```{code-cell} ipython3
:load: _static/lecture_specific/cake_eating_numerical/analytical.py
```
我们的首要目标是以数值方式获得这些解析解。

## 值函数迭代

我们将采用的第一种方法是**值函数迭代**。

这是一种**连续逼近**的方法，在我们的{doc}`求职搜索讲座 <mccall_model>`中已经讨论过。

基本思路是：

1. 对$v$取一个任意的初始猜测值。
1. 获得一个更新值$w$，定义为

   $$
   w(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
   $$

1. 如果$w$与$v$近似相等则停止，否则令$v=w$并返回步骤2。

让我们用更数学的方式来表述这个过程。

### 贝尔曼算子

我们引入**贝尔曼算子**$T$，它以函数v为参数，返回一个新函数$Tv$，定义为

$$
Tv(x) = \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

从$v$我们得到$Tv$，将$T$应用于此得到$T^2 v := T (Tv)$，依此类推。

这被称为从初始猜测值$v$开始**迭代贝尔曼算子**。
正如我们在后面的讲座中详细讨论的那样，可以使用Banach收缩映射定理来证明函数序列$T^n v$收敛到Bellman方程的解。

### 拟合值函数迭代

消费$c$和状态变量$x$都是连续的。

这在数值计算方面造成了一些复杂性。

例如，我们需要存储每个函数$T^n v$以计算下一个迭代值$T^{n+1} v$。

但这意味着我们必须在无限多个$x$处存储$T^n v(x)$，这通常是不可能的。

为了解决这个问题，我们将使用拟合值函数迭代，这在之前关于{doc}`求职搜索的讲座 <mccall_fitted_vfi>`中已经讨论过。

这个过程如下：

1. 从一组值$\{ v_0, \ldots, v_I \}$开始，这些值表示初始函数$v$在网格点$\{ x_0, \ldots, x_I \}$上的值。
1. 通过以下方式在状态空间$\mathbb R_+$上构建函数$\hat v$：
基于这些数据点的线性插值。
1. 通过反复求解贝尔曼方程中的最大化问题，获取并记录每个网格点
   $x_i$ 上的值 $T \hat v(x_i)$。
1. 除非满足某些停止条件，否则设置
   $\{ v_0, \ldots, v_I \} = \{ T \hat v(x_0), \ldots, T \hat v(x_I) \}$ 并返回步骤2。

在步骤2中，我们将使用连续分段线性插值。

### 实现

下面的`maximize`函数是一个小型辅助函数，它将SciPy的最小化程序转换为最大化程序。

```{code-cell} ipython3
def maximize(g, a, b, args):
    """
    在区间[a, b]上最大化函数g。

    我们利用了在任何区间上g的最大化器
    也是-g的最小化器这一事实。元组args收集了g的任何额外
    参数。

    返回最大值和最大化器。
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum
```
我们将参数 $\beta$ 和 $\gamma$ 存储在一个名为 `CakeEating` 的类中。

这个类还将提供一个名为 `state_action_value` 的方法，该方法根据特定状态和对 $v$ 的猜测返回消费选择的价值。

```{code-cell} ipython3
class CakeEating:

    def __init__(self,
                 β=0.96,           # 贴现因子
                 γ=1.5,            # 相对风险厌恶程度
                 x_grid_min=1e-3,  # 为了数值稳定性排除零
                 x_grid_max=2.5,   # 蛋糕大小
                 x_grid_size=120):

        self.β, self.γ = β, γ

        # 设置网格
        self.x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)

    # 效用函数
    def u(self, c):

        γ = self.γ

        if γ == 1:
            return np.log(c)
        else:
            return (c ** (1 - γ)) / (1 - γ)

    # 效用函数的一阶导数
    def u_prime(self, c):

        return c ** (-self.γ)

    def state_action_value(self, c, x, v_array):
        """
        给定x和c时贝尔曼方程的右侧。
        """

        u, β = self.u, self.β
        v = lambda x: np.interp(x, self.x_grid, v_array)

        return u(c) + β * v(x - c)
```
现在我们定义贝尔曼算子：

```{code-cell} ipython3
def T(v, ce):
    """
    贝尔曼算子。更新值函数的估计值。

    * ce 是 CakeEating 类的一个实例
    * v 是一个数组，表示值函数的估计值

    """
    v_new = np.empty_like(v)

    for i, x in enumerate(ce.x_grid):
        # 在状态 x 下最大化贝尔曼方程的右侧
        v_new[i] = maximize(ce.state_action_value, 1e-10, x, (x, v))[1]

    return v_new
```
在定义了贝尔曼算子之后，我们就可以开始求解这个模型了。

让我们先用默认参数创建一个`CakeEating`实例。

```{code-cell} ipython3
ce = CakeEating()
```
现在让我们看看值函数的迭代过程。

我们从初始猜测值$v$开始，对每个网格点$x$，令$v(x) = u(x)$。

```{code-cell} ipython3
x_grid = ce.x_grid
v = ce.u(x_grid)       # 初始猜测
n = 12                 # 迭代次数

fig, ax = plt.subplots()

ax.plot(x_grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='初始猜测')

for i in range(n):
    v = T(v, ce)  # 应用贝尔曼算子
    ax.plot(x_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax.legend()
ax.set_ylabel('值', fontsize=12)
ax.set_xlabel('蛋糕大小 $x$', fontsize=12)
ax.set_title('值函数迭代')

plt.show()
```
为了更系统地完成这项工作，我们引入一个名为`compute_value_function`的包装函数，该函数会一直迭代直到满足某些收敛条件。

```{code-cell} ipython3
def compute_value_function(ce,
                           tol=1e-4,
                           max_iter=1000,
                           verbose=True,
                           print_skip=25):

    # 设置循环
    v = np.zeros(len(ce.x_grid)) # 初始猜测
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(v, ce)

        error = np.max(np.abs(v - v_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        v = v_new

    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return v_new
```
现在让我们调用它，注意运行需要一点时间。

```{code-cell} ipython3
v = compute_value_function(ce)
```
现在我们可以绘图查看收敛后的值函数是什么样子。

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v, label='近似值函数')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_title('值函数')
ax.legend()
plt.show()
```
接下来让我们将其与解析解进行比较。

```{code-cell} ipython3
v_analytical = v_star(ce.x_grid, ce.β, ce.γ)
```
```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v_analytical, label='解析解')
ax.plot(x_grid, v, label='数值解')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.legend()
ax.set_title('解析值函数与数值值函数的比较')
plt.show()
```
对于较大的 $x$ 值,近似的质量相当好,但在接近下边界时则不太理想。

这是因为效用函数以及由此产生的价值函数在接近下边界时非常陡峭,因此很难进行近似。

### 策略函数

让我们看看这在计算最优策略时会如何表现。

在{doc}`蛋糕食用问题的第一讲 <cake_eating_problem>`中,最优消费策略被证明为

$$
\sigma^*(x) = \left(1-\beta^{1/\gamma} \right) x
$$

让我们看看我们的数值结果是否能得到类似的结果。

我们的数值策略将是在一系列 $x$ 点上计算

$$
\sigma(x) = \arg \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

然后进行插值。

对于 $v$,我们将使用我们上面获得的价值函数的近似值。

这是相关函数:

```{code-cell} ipython3
def σ(ce, v):
    """
    最优策略函数。给定价值函数,
    它找到每个状态下的最优消费。

    * ce 是 CakeEating 的一个实例
    * v 是一个价值函数数组

    """
    c = np.empty_like(v)

    for i in range(len(ce.x_grid)):
        x = ce.x_grid[i]
        # 在状态 x 下最大化贝尔曼方程的右侧
        c[i] = maximize(ce.state_action_value, 1e-10, x, (x, v))[0]

    return c
```
现在让我们传入近似值函数并计算最优消费：

```{code-cell} ipython3
c = σ(ce, v)
```
(pol_an)=
让我们将其与真实的解析解进行对比绘图

```{code-cell} ipython3
c_analytical = c_star(ce.x_grid, ce.β, ce.γ)

fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label='analytical')
ax.plot(ce.x_grid, c, label='numerical')
ax.set_ylabel(r'$\sigma(x)$')
ax.set_xlabel('$x$')
ax.legend()

plt.show()
```
拟合结果还算合理，但并不完美。

我们可以通过增加网格大小或降低值函数迭代程序中的误差容限来改进它。

但这两种改变都会导致计算时间变长。

另一种可能是使用替代算法，这可以实现更快的计算时间，同时获得更高的精确度。

接下来我们将探讨这一点。

## 时间迭代

现在让我们看看计算最优策略的另一种方法。

回想一下，最优策略满足欧拉方程

```{math}
:label: euler-cen

u' (\sigma(x)) = \beta u' ( \sigma(x - \sigma(x)))
\quad \text{for all } x > 0
```

从计算角度来看，我们可以从任意初始猜测值 $\sigma_0$ 开始，然后选择 $c$ 来求解

$$
u^{\prime}( c ) = \beta u^{\prime} (\sigma_0(x - c))
$$

在所有 $x > 0$ 处选择 $c$ 来满足这个方程会产生一个关于 $x$ 的函数。

将这个新函数称为 $\sigma_1$，把它作为新的猜测值并重复这个过程。
这被称为**时间迭代**。

与值函数迭代一样，我们可以将更新步骤视为一个算子的作用，这次用$K$表示。

* 特别地，$K\sigma$是使用刚才描述的程序从$\sigma$更新得到的策略。
* 我们将在下面的练习中使用这个术语。

相对于值函数迭代，时间迭代的主要优势在于它在策略空间而不是值函数空间中运作。

这很有帮助，因为策略函数的曲率较小，因此更容易近似。

在练习中，你需要实现时间迭代并将其与值函数迭代进行比较。

你应该会发现这个方法更快且更准确。

这是由于

1. 刚才提到的曲率问题，以及
1. 我们使用了更多信息——在这种情况下是一阶条件。

## 练习

```{exercise}
:label: cen_ex1

尝试以下问题的修改。
让蛋糕大小不再按照 $x_{t+1} = x_t - c_t$ 变化，
而是按照

$$
x_{t+1} = (x_t - c_t)^{\alpha}
$$

变化，其中 $\alpha$ 是一个满足 $0 < \alpha < 1$ 的参数。

(我们在学习最优增长模型时会看到这种更新规则。)

对值函数迭代代码进行必要的修改并绘制值函数和策略函数。

尽可能多地重用现有代码。
```

```{solution-start} cen_ex1
:class: dropdown
```

我们需要创建一个类来保存我们的基本参数并返回贝尔曼方程的右侧。

我们将使用[继承](https://en.wikipedia.org/wiki/Inheritance_%28object-oriented_programming%29)来最大化代码重用。

```{code-cell} ipython3
class OptimalGrowth(CakeEating):
    """
    CakeEating的一个子类，添加了参数α并重写了
    state_action_value方法。
    """

    def __init__(self,
                 β=0.96,           # 贴现因子
                 γ=1.5,            # 相对风险厌恶度
                 α=0.4,            # 生产力参数
                 x_grid_min=1e-3,  # 为了数值稳定性排除零
                 x_grid_max=2.5,   # 蛋糕大小
                 x_grid_size=120):

        self.α = α
        CakeEating.__init__(self, β, γ, x_grid_min, x_grid_max, x_grid_size)

    def state_action_value(self, c, x, v_array):
        """
        给定x和c时贝尔曼方程的右侧。
        """

        u, β, α = self.u, self.β, self.α
        v = lambda x: np.interp(x, self.x_grid, v_array)

        return u(c) + β * v((x - c)**α)
```
```{code-cell} ipython3
og = OptimalGrowth()
```
这是计算得到的值函数。

```{code-cell} ipython3
v = compute_value_function(og, verbose=False)

fig, ax = plt.subplots()

ax.plot(x_grid, v, lw=2, alpha=0.6)
ax.set_ylabel('值', fontsize=12)
ax.set_xlabel('状态 $x$', fontsize=12)

plt.show()
```
这是计算得出的策略，与我们之前推导的标准蛋糕食用情况（$\alpha=1$）的解决方案相结合。

```{code-cell} ipython3
c_new = σ(og, v)

fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label=r'$\alpha=1$ 解')
ax.plot(ce.x_grid, c_new, label=fr'$\alpha={og.α}$ 解')

ax.set_ylabel('消费', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)

ax.legend(fontsize=12)

plt.show()
```
当 $\alpha < 1$ 时消费更高,因为至少对于较大的 $x$ 而言,储蓄的回报率更低。

```{solution-end}
```


```{exercise}
:label: cen_ex2

实现时间迭代法,回到原始情况(即,去掉上述练习中的修改)。
```


```{solution-start} cen_ex2
:class: dropdown
```

这是实现时间迭代的一种方法。

```{code-cell} ipython3
def K(σ_array, ce):
    """
    策略函数算子。给定策略函数,
    使用欧拉方程更新最优消费。

    * σ_array 是网格上策略函数值的数组
    * ce 是 CakeEating 的一个实例

    """

    u_prime, β, x_grid = ce.u_prime, ce.β, ce.x_grid
    σ_new = np.empty_like(σ_array)

    σ = lambda x: np.interp(x, x_grid, σ_array)

    def euler_diff(c, x):
        return u_prime(c) - β * u_prime(σ(x - c))

    for i, x in enumerate(x_grid):

        # 单独处理小的 x --- 有助于数值稳定性
        if x < 1e-12:
            σ_new[i] = 0.0

        # 处理其他 x
        else:
            σ_new[i] = bisect(euler_diff, 1e-10, x - 1e-10, x)

    return σ_new
```
```{code-cell} ipython3
def iterate_euler_equation(ce,
                           max_iter=500,
                           tol=1e-5,
                           verbose=True,
                           print_skip=25):

    x_grid = ce.x_grid

    σ = np.copy(x_grid)        # 初始猜测

    i = 0
    error = tol + 1
    while i < max_iter and error > tol:

        σ_new = K(σ, ce)

        error = np.max(np.abs(σ_new - σ))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"第{i}次迭代的误差是{error}。")

        σ = σ_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在{i}次迭代后收敛。")

    return σ
```
```{code-cell} ipython3
ce = CakeEating(x_grid_min=0.0)
c_euler = iterate_euler_equation(ce)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label='解析解')
ax.plot(ce.x_grid, c_euler, label='时间迭代解')

ax.set_ylabel('消费')
ax.set_xlabel('$x$')
ax.legend(fontsize=12)

plt.show()
```

```{solution-end}
```
