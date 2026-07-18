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
  title: 最优储蓄 II：数值方法吃蛋糕问题
  headings:
    Overview: 概述
    Reviewing the Model: 回顾模型
    Value Function Iteration: 价值函数迭代
    Value Function Iteration::The Bellman Operator: 贝尔曼算子
    Value Function Iteration::Fitted Value Function Iteration: 拟合价值函数迭代
    Value Function Iteration::Implementation: 实现
    Value Function Iteration::Policy Function: 策略函数
    Exercises: 练习
---

# 最优储蓄 II：数值方法吃蛋糕问题

```{contents} 目录
:depth: 2
```

## 概述

在本讲中，我们将继续研究{doc}`os`中描述的问题。

本讲的目标是使用数值方法来求解该问题。

乍看之下这似乎没有必要，因为我们已经通过解析方法得到了最优策略。

然而，吃蛋糕问题如果不加修改就过于简单，几乎没有实际用途。一旦我们对问题进行修改，数值方法就变得必不可少。

因此，现在引入数值方法并在这个简单问题上进行测试是有意义的。

由于我们已经知道解析解，这将使我们能够评估不同数值方法的精确度。

```{note}
下面的代码旨在追求清晰性，而非追求最高效率。

在后面的讲座中，我们将探讨提高速度和效率的最佳实践。

现在先把这些算法和代码优化放在一边。
```

我们将使用以下导入：

```{code-cell} python3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from scipy.optimize import minimize_scalar, bisect
from typing import NamedTuple
```

## 回顾模型

在开始之前，你可能想要回顾一下{doc}`os`中的细节。

特别要回顾的是贝尔曼方程：

```{math}
:label: bellman-cen

v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
\quad \text{对所有 } x \geq 0.
```

其中 $u$ 是CRRA效用函数。

价值函数和最优策略的解析解如下所示。

```{code-cell} python3
def c_star(x, β, γ):

    return (1 - β ** (1/γ)) * x


def v_star(x, β, γ):

    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

我们的第一个目标是用数值方法得到这些解析解。

## 价值函数迭代

我们将采用的第一种方法是**价值函数迭代**。

这是一种**逐次逼近**的方法，在我们关于{doc}`工作搜寻的讲座 <mccall_model>`中已经讨论过。

基本思路是：

1. 取一个任意的初始猜测 $v$。
1. 得到一个更新 $\hat v$，其定义为

   $$
       \hat v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
   $$

1. 如果 $\hat v$ 与 $v$ 大致相等，则停止；否则设 $v=\hat v$ 并返回第2步。

让我们把它写得更数学化一些。

### 贝尔曼算子

我们引入**贝尔曼算子** $T$，它以函数 $v$ 为输入，返回一个新函数 $Tv$，定义为

$$
    Tv(x) = \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

从 $v$ 出发，我们得到 $Tv$，再应用 $T$ 得到 $T^2 v := T(Tv)$，依此类推。

这被称为从初始猜测 $v$ 出发，**迭代贝尔曼算子**。

正如我们在后面的讲座中详细讨论的那样，可以使用巴纳赫收缩映射定理来证明函数序列 $T^n v$ 收敛到贝尔曼方程的解。

### 拟合价值函数迭代

消费 $c$ 和状态变量 $x$ 都是连续的。

这在数值计算方面造成了一些复杂性。

例如，我们需要存储每个函数 $T^n v$ 以计算下一个迭代值 $T^{n+1} v$。

但这意味着我们必须在无限多个 $x$ 处存储 $T^n v(x)$，这通常是不可能的。

为了解决这个问题，我们将使用拟合价值函数迭代，这在之前关于{doc}`工作搜寻的讲座 <mccall_fitted_vfi>`中已经讨论过。

这个过程如下：

1. 从一组值 $\{ v_0, \ldots, v_I \}$ 开始，这些值表示初始函数 $v$ 在网格点$\{ x_0, \ldots, x_I \}$ 上的值。
1. 通过线性插值，基于插值点 $\{(x_i, v_i)\}$，在状态空间 $\mathbb{R}_+$ 上建立函数 $\hat{v}$。
1. 将 $\hat v$ 代入贝尔曼方程右侧，获取并记录每个网格点 $x_i$ 上的值 $T \hat v(x_i)$。
1. 除非满足某个停止条件，否则设置 $\{ v_0, \ldots, v_I \} = \{ T \hat v(x_0), \ldots, T \hat v(x_I) \}$ 并返回步骤2。

在步骤2中，我们将使用分段线性插值。

### 实现

下面的 `maximize` 函数是一个小型辅助函数，它将SciPy的最小化程序转换为一个最大化程序。

```{code-cell} python3
def maximize(g, upper_bound):
    """
    在区间 [0, upper_bound] 上最大化函数 g。

    我们利用了这样一个事实：在任意区间上，g 的最大值点，
    同时也是 -g 的最小值点。

    """

    objective = lambda x: -g(x)
    bounds = (0, upper_bound)
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum
```

我们将参数 $\beta$ 和 $\gamma$ 以及网格存储在一个名为 `Model` 的 `NamedTuple` 中。

我们还将创建一个名为 `create_cake_eating_model` 的辅助函数，用于存储默认参数并构建 `Model` 的实例。

```{code-cell} python3
class Model(NamedTuple):
    β: float
    γ: float
    x_grid: np.ndarray

def create_cake_eating_model(
        β: float = 0.96,           # 折现因子
        γ: float = 1.5,            # 相对风险厌恶程度
        x_grid_min: float = 1e-3,  # 为了数值稳定性排除零
        x_grid_max: float = 2.5,   # 蛋糕大小
        x_grid_size: int = 120
    ):
    """
    创建吃蛋糕模型的一个实例。

    """
    x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)
    return Model(β, γ, x_grid)
```

这是CRRA效用函数。

```{code-cell} python3
def u(c, γ):
    return (c ** (1 - γ)) / (1 - γ)
```

为了运用贝尔曼方程，我们把它写成

$$
    v(x) = \max_{0 \leq c \leq x} B(x, c, v)
$$

其中

$$
    B(x, c, v) := u(c) + \beta v(x - c)
$$

现在我们来实现函数 $B$。

```{code-cell} python3
def B(
        x: float,               # 当前状态（剩余蛋糕）
        c: float,               # 当前消费
        v: np.ndarray,          # 当前对价值函数的猜测
        model: Model            # 吃蛋糕模型的实例
    ):
    """
    在给定 x 和 c 的情况下，贝尔曼方程右侧的值。

    """
    # 拆包（简化名称）
    β, γ, x_grid = model

    # 通过线性插值将数组 v 转换为函数
    vf = lambda x: np.interp(x, x_grid, v)

    # 返回 B(x, c, v)
    return u(c, γ) + β * vf(x - c)
```

现在我们定义作用于网格点上的贝尔曼算子：

$$
    Tv(x_i) = \max_{0 \leq c \leq x_i} B(x_i, c, v)
    \qquad \text{对所有 } i
$$

```{code-cell} python3
def T(
        v: np.ndarray,          # 当前对价值函数的猜测
        model: Model            # 吃蛋糕模型的实例
    ):
    " 贝尔曼算子。更新价值函数的猜测。 "

    # 为新数组 v_new = Tv 分配内存
    v_new = np.empty_like(v)

    # 对所有 x 计算 Tv(x)
    for i, x in enumerate(model.x_grid):
        # 在 [0, x] 上关于 c 最大化贝尔曼方程的右侧
        _, v_new[i] = maximize(lambda c: B(x, c, v, model), x)

    return v_new
```

在定义了贝尔曼算子之后，我们就可以开始求解这个模型了。

让我们先用默认参数创建一个模型。

```{code-cell} python3
model = create_cake_eating_model()
β, γ, x_grid = model
```

现在让我们看看价值函数的迭代过程。

我们从初始猜测 $v$ 开始，即对每个网格点 $x$，$v(x) = u(x)$。

```{code-cell} python3
v = u(x_grid, γ)  # 初始猜测
n = 12                  # 迭代次数
fig, ax = plt.subplots()

# 初始图
ax.plot(x_grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='初始猜测')

# 迭代
for i in range(n):
    v = T(v, model)  # 应用贝尔曼算子
    ax.plot(x_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

# 最后一次更新和绘图
v = T(v, model)  
ax.plot(x_grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='最终猜测')

ax.legend()
ax.set_ylabel('价值', fontsize=12)
ax.set_xlabel('蛋糕大小 $x$', fontsize=12)
ax.set_title('价值函数迭代')
plt.show()
```

为了更系统地进行迭代，我们引入一个名为 `compute_value_function` 的包装函数。

它的任务是使用 $T$ 进行迭代，直到满足某些收敛条件。

```{code-cell} python3
def compute_value_function(
        model: Model,
        tol: float = 1e-4,
        max_iter: int = 1_000,
        verbose: bool = True,
        print_skip: int = 25
    ):

    # 设置循环
    v = np.zeros(len(model.x_grid)) # 初始猜测
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(v, model)

        error = np.max(np.abs(v - v_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"第 {i} 次迭代的误差是 {error}.")

        v = v_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在第 {i} 次迭代中收敛。")

    return v_new
```

现在让我们调用它，注意运行需要一点时间。

```{code-cell} python3
v = compute_value_function(model)
```

现在我们可以绘图查看收敛后的价值函数是什么样子。

```{code-cell} python3
fig, ax = plt.subplots()

ax.plot(x_grid, v, label='近似价值函数')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_title('价值函数')
ax.legend()
plt.show()
```

接下来让我们将其与解析解进行比较。

```{code-cell} python3
v_analytical = v_star(x_grid, β, γ)
```

```{code-cell} python3
fig, ax = plt.subplots()

ax.plot(x_grid, v_analytical, label='解析解')
ax.plot(x_grid, v, label='数值解')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.legend()
ax.set_title('解析解与数值解的价值函数对比')
plt.show()
```

对于较大的 $x$ 值，近似的质量相当好，但在下边界附近则要差一些。

这是因为效用函数以及由此产生的价值函数在接近下边界时非常陡峭，因此难以逼近。

```{note}
解决这个问题的一种方法是使用非线性网格，在零点附近设置更多的点。

不过，我们不打算深入探讨这个想法，而是把注意力转向使用策略函数进行处理。

我们将会看到，通过对策略函数的猜测进行迭代，可以避免价值函数迭代。

策略函数的曲率较小，因此比价值函数更容易插值。

这些想法将在接下来的几讲中进行探讨。
```

### 策略函数

下面我们来尝试计算最优策略。

在{doc}`os`中，最优消费策略被证明为

$$
    \sigma^*(x) = \left(1-\beta^{1/\gamma} \right) x
$$

让我们看看我们的数值结果是否能得到类似的形式。

我们的数值策略是，对任意给定的 $v$，计算策略

$$
    \sigma(x) = \arg \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

这个策略被称为 $v$-**贪婪策略**。

在实践中，我们将在 $x$ 点组成的网格上计算 $\sigma$，然后进行插值。

对于 $v$，我们将使用上面得到的价值函数近似值。

这是相关函数：

```{code-cell} python3
def get_greedy(
        v: np.ndarray,          # 当前对价值函数的猜测
        model: Model            # 吃蛋糕模型的实例
    ):
    " 在 x_grid 上计算 v-贪婪策略。"

    σ = np.empty_like(v)

    for i, x in enumerate(model.x_grid):
        # 在状态 x 下最大化贝尔曼方程的右侧
        σ[i], _ = maximize(lambda c: B(x, c, v, model), x)

    return σ
```

现在让我们传入近似价值函数并计算最优消费：

```{code-cell} python3
σ = get_greedy(v, model)
```

(pol_an)=
让我们将其与真实的解析解进行对比绘图

```{code-cell} python3
c_analytical = c_star(model.x_grid, model.β, model.γ)

fig, ax = plt.subplots()

ax.plot(model.x_grid, c_analytical, label='解析解')
ax.plot(model.x_grid, σ, label='数值解')
ax.set_ylabel(r'$\sigma(x)$')
ax.set_xlabel('$x$')
ax.legend()

plt.show()
```

拟合效果还算合理，但并不完美。

我们可以通过增加网格规模，或者在价值函数迭代过程中降低误差容忍度来改进它。

然而，这两种方法都会导致更长的计算时间。

另一种可能性是使用一种替代算法，它既有可能缩短计算时间，同时还能提高精确度。

我们将在{doc}`intermediate:os_time_iter`中探讨这一点。


## 练习

```{exercise}
:label: cen_ex1

尝试如下对问题的修改：

让蛋糕大小不再按照 $x_{t+1} = x_t - c_t$ 变化，而是按照

$$
x_{t+1} = (x_t - c_t)^{\alpha}
$$

变化，其中 $\alpha$ 是一个满足 $0 < \alpha < 1$ 的参数。

（我们在学习最优增长模型时会看到这种更新规则。）

请对价值函数迭代代码进行相应修改，并绘制价值函数和策略函数。

尽量重用已有代码。
```

```{solution-start} cen_ex1
:class: dropdown
```

我们需要创建一个扩展版本的模型和状态-行动价值函数。

我们将为扩展蛋糕模型创建一个新的 `NamedTuple`，以及一个辅助函数。

```{code-cell} python3
# 创建扩展蛋糕模型的数据结构
class ExtendedModel(NamedTuple):
    β: float
    γ: float
    α: float
    x_grid: np.ndarray

def create_extended_model(β=0.96,           # 折现因子
                          γ=1.5,            # 相对风险厌恶程度
                          α=0.4,            # 生产力参数
                          x_grid_min=1e-3,  # 为了数值稳定性排除零
                          x_grid_max=2.5,   # 蛋糕大小
                          x_grid_size=120):
    """
    创建扩展吃蛋糕模型的一个实例。
    """
    x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)
    return ExtendedModel(β, γ, α, x_grid)

def extended_B(c, x, v, model):
    """
    在给定 x 和 c 的情况下，扩展蛋糕模型贝尔曼方程右侧的值。

    """
    β, γ, α, x_grid = model
    vf = lambda x: np.interp(x, x_grid, v)
    return u(c, γ) + β * vf((x - c)**α)
```

我们还需要一个修改后的贝尔曼算子：

```{code-cell} python3
def extended_T(v, model):
    " 扩展蛋糕模型的贝尔曼算子。 "

    v_new = np.empty_like(v)
    for i, x in enumerate(model.x_grid):
        _, v_new[i] = maximize(lambda c: extended_B(c, x, v, model), x)
    return v_new
```

现在创建模型：

```{code-cell} python3
model = create_extended_model()
```

下面是一个计算价值函数的函数。

```{code-cell} python3
def compute_value_function_extended(model,
                                    tol=1e-4,
                                    max_iter=1000,
                                    verbose=True,
                                    print_skip=25):
    """
    计算扩展蛋糕模型的价值函数。
    """
    v = np.zeros(len(model.x_grid))
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = extended_T(v, model)
        error = np.max(np.abs(v - v_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"第 {i} 次迭代的误差是 {error}.")
        v = v_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在第 {i} 次迭代中收敛。")

    return v_new

v = compute_value_function_extended(model, verbose=False)

fig, ax = plt.subplots()

ax.plot(model.x_grid, v, lw=2, alpha=0.6)
ax.set_ylabel('价值', fontsize=12)
ax.set_xlabel('状态 $x$', fontsize=12)

plt.show()
```

下面是计算得到的策略函数，并与我们在标准吃蛋糕问题（$\alpha=1$）情况下推导出的解进行比较。

```{code-cell} python3
def extended_get_greedy(model, v):
    """
    扩展蛋糕模型的最优策略函数。
    """
    σ = np.empty_like(v)

    for i, x in enumerate(model.x_grid):
        # 在 [0, x] 上关于 c 最大化 extended_B
        σ[i], _ = maximize(lambda c: extended_B(c, x, v, model), x)

    return σ

σ = extended_get_greedy(model, v)

# 获取基准模型用于比较
baseline_model = create_cake_eating_model()
c_analytical = c_star(baseline_model.x_grid, baseline_model.β, baseline_model.γ)

fig, ax = plt.subplots()

ax.plot(baseline_model.x_grid, c_analytical, label=r'$\alpha=1$ 的解')
ax.plot(model.x_grid, σ, label=fr'$\alpha={model.α}$ 的解')

ax.set_ylabel('消费', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)

ax.legend(fontsize=12)

plt.show()
```

当 $\alpha < 1$ 时，消费水平更高，因为至少对于较大的 $x$，储蓄的回报较低。

```{solution-end}
```
