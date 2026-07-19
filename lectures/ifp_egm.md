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
  title: 收入波动问题 III：内生网格法
  headings:
    Overview: 概述
    The Household Problem: 家庭问题
    The Household Problem::Set-Up: 建模
    The Household Problem::Value Function and Euler Equation: 价值函数和欧拉方程
    The Household Problem::Optimality Results: 最优性结果
    Computation: 计算
    Computation::Solution Method: 求解方法
    NumPy Implementation: NumPy 实现
    NumPy Implementation::Set Up: 建模
    NumPy Implementation::Solver: 求解器
    JAX Implementation: JAX 实现
    JAX Implementation::Set Up: 建模
    JAX Implementation::Solver: 求解器
    JAX Implementation::Test run: 测试运行
    JAX Implementation::Timing: 计时
    JAX Implementation::Dynamics: 动态性质
    JAX Implementation::A Sanity Check: 合理性检查
    Simulation: 模拟
---

# {index}`收入波动问题 III：内生网格法 <single: The Income Fluctuation Problem III: The Endogenous Grid Method>`

```{include} _admonition/gpu.md
```

```{contents} 目录
:depth: 2
```

## 概述

在本讲中，我们继续研究以下讲座中的一个 IFP 版本：

* {doc}`intermediate:ifp_discrete` 和
* {doc}`intermediate:ifp_opi`。

我们将做两处改动。

1. 将时序改为一个更适合我们设定的时序。
2. 使用内生网格法（EGM）来求解模型。

我们之所以使用 EGM，是因为我们从 {doc}`intermediate:os_egm_jax` 中已经知道它既快速又精确。

下文讨论的技术细节的主要参考文献是 {cite}`ma2020income`。

其他参考文献包括 {cite}`Deaton1991`、{cite}`DenHaan2010`、
{cite}`Kuhn2013`、{cite}`Rabault2002`、{cite}`Reiter2009` 和
{cite}`SchechtmanEscudero1977`。

除了Anaconda中已有的库外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon jax
```

我们还需要以下导入：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import numba
from quantecon import MarkovChain
import jax
import jax.numpy as jnp
from typing import NamedTuple
```

## 家庭问题

```{index} single: Optimal Savings; Problem
```

让我们先写下模型，然后讨论如何求解。

### 建模

考虑一个家庭，它选择一个依赖于状态的消费计划 $\{c_t\}_{t \geq 0}$ 来最大化

$$
\mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

约束条件为

```{math}
:label: eqst

a_{t+1} = R (a_t - c_t) + Y_{t+1}
\quad c_t \geq 0,
\quad a_t \geq 0
\quad t = 0, 1, \ldots
```

这里

* $\beta \in (0,1)$ 是折现因子
* $a_t$ 是 $t$ 时期的资产存量，借贷约束为$a_t \geq 0$
* $c_t$ 是消费
* $Y_t$ 是非资本收入（工资、失业补偿等）
* $R := 1 + r$，其中 $r > 0$ 是储蓄利率

时序安排如下：

1. 在时期 $t$ 开始时，家庭观察到当前的资产存量 $a_t$。
1. 家庭选择当前消费 $c_t$。
1. 储蓄 $s_t := a_t - c_t$ 以利率 $r$ 获得利息。
1. 劳动收入 $Y_{t+1}$ 实现，时间转移到 $t+1$。

非资本收入 $Y_t$ 由 $Y_t = y(Z_t)$ 给出，其中

* $\{Z_t\}$ 是一个外生状态过程，且
* $y$ 是一个取值于 $\mathbb{R}_+$ 的函数。

我们将 $\{Z_t\}$ 视为一个有限状态马尔可夫链，其取值在 $\mathsf Z$ 中，马尔可夫矩阵为 $\Pi$。

```{note}
在前面的讲座中，我们使用了更为标准的家庭预算约束 $a_{t+1} + c_t \leq R a_t + Y_t$。

这种设定在量化经济学中十分常见，是为了适应离散化而开发的。

这意味着控制变量同时也是下一期的状态 $a_{t+1}$，
这使得将资产限制在有限网格上变得很直接。

但是将控制变量固定为下一期状态，就迫使我们在当前状态中包含更多
信息，从而扩大了状态空间的规模。

此外，以离散化为目标并不总是一个好主意，因为
它深受维度诅咒之苦。

这些想法将在{doc}`下一讲 <intermediate:ifp_egm_transient_shocks>`中变得更加清晰。
```

我们进一步假设

1. $\beta R < 1$
1. $u$ 是光滑的、严格递增和严格凹的，且满足 $\lim_{c \to 0} u'(c) = \infty$ 和 $\lim_{c \to \infty} u'(c) = 0$
1. $y(z) = \exp(z)$

资产空间是 $\mathbb R_+$，状态是 $(a,z) \in \mathsf S := \mathbb R_+ \times \mathsf Z$。

从 $(a,z) \in \mathsf S$ 出发的**可行消费路径**是一个消费序列 $\{c_t\}$，使得 $\{c_t\}$ 及其诱导的资产路径 $\{a_t\}$ 满足：

1. $(a_0, z_0) = (a, z)$
1. {eq}`eqst` 中的可行性约束，以及
1. 适应性，即 $c_t$ 是截至时期 $t$ 的随机结果的函数，而不是时期 $t$ 之后的结果的函数。

第三点的含义是：时期 $t$ 的消费不能依赖尚未被观测到的结果。

事实上，在这个问题中，消费的最优选择可以通过仅依赖当前状态来实现。

最优性定义如下。

### 价值函数和欧拉方程

**价值函数** $V \colon \mathsf S \to \mathbb{R}$ 定义为

```{math}
:label: eqvfs_egm

V(a, z) := \max \, \mathbb{E}
\left\{
\sum_{t=0}^{\infty} \beta^t u(c_t)
\right\}
```

其中最大化是在所有从 $(a,z)$ 出发的可行消费路径上进行的。

从 $(a,z)$ 出发的**最优消费路径**是一个从 $(a,z)$ 出发的可行消费路径，且使{eq}`eqvfs_egm`最大化。

为了刻画这样的路径，我们可以使用欧拉方程的一个版本，在当前设定中为

```{math}
:label: ee00

    u' (c_t) \geq \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

以及

```{math}
:label: ee01

    c_t < a_t
    \; \implies \;
    u' (c_t) = \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

当 $c_t$ 达到上界 $a_t$ 时，严格不等式 $u' (c_t) > \beta R \,  \mathbb{E}_t  u'(c_{t+1})$
可能出现，因为 $c_t$ 无法充分增加以达到等式。

$c_t = 0$ 的情形在最优路径上永远不会出现，因为 $u'(0) = \infty$。

### 最优性结果

正如 {cite}`ma2020income` 所示，

1. 对于每个 $(a,z) \in \mathsf S$，从 $(a,z)$ 出发存在唯一的最优消费路径
1. 这条路径是从 $(a,z)$ 出发的、满足欧拉方程 {eq}`ee00`-{eq}`ee01`
   和横截条件

```{math}
:label: eqtv

\lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0
```

的唯一可行路径。

此外，存在一个**最优消费策略**
$\sigma^* \colon \mathsf S \to \mathbb R_+$，使得从 $(a,z)$ 出发的路径由

$$
    (a_0, z_0) = (a, z),
    \quad
    c_t = \sigma^*(a_t, Z_t)
    \quad \text{和} \quad
    a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

生成，该路径同时满足欧拉方程 {eq}`ee00`-{eq}`ee01` 以及 {eq}`eqtv`，因此是从 $(a,z)$
出发的唯一最优路径。

因此，为了解决这个优化问题，我们需要计算策略 $\sigma^*$。

(ifp_computation)=
## 计算

```{index} single: Optimal Savings; Computation
```

我们使用时间迭代法和内生网格法来求解最优消费策略，这两种方法此前已经在以下讲座中讨论过：

* {doc}`intermediate:os_time_iter`
* {doc}`intermediate:os_egm`

### 求解方法

我们将{eq}`ee01`改写为一个关于函数而非随机变量的表达式：

```{math}
:label: eqeul1

    (u' \circ \sigma)  (a, z)
    = \beta R \, \sum_{z'} (u' \circ \sigma)
            [R (a - \sigma(a, z)) + y(z'), \, z'] \, \Pi(z, z')
```

这里

* $(u' \circ \sigma)(s) := u'(\sigma(s))$，
* 带撇号的变量表示下一期的状态（同时也表示导数），且
* $\sigma$ 是未知函数。

等式{eq}`eqeul1`在所有内部选择处成立，即 $\sigma(a, z) < a$。

我们的目标是求出{eq}`eqeul1`的一个不动点 $\sigma$。

为此我们使用 EGM。

下面我们使用关系式 $a_t = c_t + s_t$ 和 $a_{t+1} = R s_t + Y_{t+1}$。

我们从一个外生的储蓄网格 $s_0 < s_1 < \cdots < s_m$（其中 $s_0 = 0$）开始。

我们固定策略函数 $\sigma$ 的当前猜测值。

对于每个 $i \geq 1$ 的外生储蓄水平 $s_i$ 和当前状态 $z_j$，我们设定

```{math}
:label: cfequ

    c_{ij} := (u')^{-1}
        \left[
            \beta R \, \sum_{z'}
            u' [ \sigma(R s_i + y(z'), z') ] \, \Pi(z_j, z')
        \right]
```

这里欧拉方程成立，是因为 $i \geq 1$ 意味着 $s_i > 0$，从而消费是内部的。

对于边界情形 $s_0 = 0$，我们设定

$$
    c_{0j} := 0  \quad \text{对所有 } j
$$

然后我们通过下式获得当前资产的相应内生网格

$$
    a_{ij} := c_{ij} + s_i.
$$

注意，对每个 $j$，都有 $a_{0j} = c_{0j} = 0$。

由于在没有借贷的情况下，当资产为零时消费也为零，这将插值锚定在原点处的正确值上。

我们对策略函数的下一个猜测值，记为 $K\sigma$，是对插值点

$$ \{(a_{0j}, c_{0j}), \ldots, (a_{mj}, c_{mj})\} $$

（对每个 $j$）进行的线性插值。

（一维线性插值的数量等于 $\mathsf Z$ 的大小。）

## NumPy 实现

在本节中，我们将编写一个只追求清晰而非效率的 NumPy 版本代码。

一旦代码运行成功，我们将编写一个效率高得多的 JAX 版本，并检验两者的结果是否一致。

我们使用 CRRA 效用函数：

$$
    u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

### 建模

这里我们构建一个名为 `IFPNumPy` 的类，用来存储模型的基本要素。

外生状态过程 $\{Z_t\}$ 默认为一个两状态马尔可夫链，
转移矩阵为 $\Pi$。

```{code-cell} ipython3
class IFPNumPy(NamedTuple):
    R: float                  # 总利率 R = 1 + r
    β: float                  # 折现因子
    γ: float                  # 偏好参数
    Π: np.ndarray             # 外生冲击的马尔可夫矩阵
    z_grid: np.ndarray        # Z_t 的马尔可夫状态值
    s: np.ndarray             # 外生储蓄网格


def create_ifp(r=0.01,
               β=0.96,
               γ=1.5,
               Π=((0.6, 0.4),
                  (0.05, 0.95)),
               z_grid=(-10.0, np.log(2.0)),
               savings_grid_max=16,
               savings_grid_size=200):

    s = np.linspace(0, savings_grid_max, savings_grid_size)
    Π, z_grid = np.array(Π), np.array(z_grid)
    R = 1 + r
    assert R * β < 1, "Stability condition violated."
    return IFPNumPy(R, β, γ, Π, z_grid, s)
```

### 求解器

下面是算子 $K$，它将当前的猜测值 $\sigma$ 转换为下一期的猜测值 $K\sigma$。

在实践中，它接收

* 一个关于最优消费值 $c_{ij}$ 的猜测，存储为 `c_vec`
* 以及对应的一组内生网格点 $a^e_{ij}$，存储为 `a_vec`

通过对 $(a^e_{ij}, c_{ij})$ 在每个 $j$ 上关于 $i$ 进行线性插值，
这些值被转换为消费策略 $a \mapsto \sigma(a, z_j)$。

由于在本版本的模型中没有需要积分消去的冲击，我们可以通过对有限状态空间 $\mathsf Z$ 求和，直接计算{eq}`cfequ`。

```{code-cell} ipython3
@numba.jit
def K_numpy(
        c_in: np.ndarray,   # σ在内生网格上的初始猜测
        a_in: np.ndarray,   # 初始内生网格
        ifp_numpy: IFPNumPy
    ) -> np.ndarray:
    """
    使用内生网格法求解IFP模型的欧拉方程算子。

    该算子实现EGM算法的一次迭代，用于更新消费策略函数。

    """
    R, β, γ, Π, z_grid, s = ifp_numpy
    n_a, n_z = len(s), len(z_grid)
    c_out = np.zeros_like(c_in)
    u_prime = lambda c: c**(-γ)
    u_prime_inv = lambda c: c**(-1/γ)
    y = lambda z: np.exp(z)

    for i in range(1, n_a):  # 从1开始，对应正的储蓄水平
        for j in range(n_z):

            # 计算 Σ_z' u'(σ(R s_i + y(z'), z')) Π[z_j, z']
            expectation = 0.0
            for k in range(n_z):
                z_prime = z_grid[k]
                # 计算下一期资产
                next_a = R * s[i] + y(z_prime)
                # 插值以得到 σ(R s_i + y(z'), z')
                next_c = np.interp(next_a, a_in[:, k], c_in[:, k])
                # 按转移概率加权并加到期望值中
                expectation += u_prime(next_c) * Π[j, k]

            # 计算更新后的 c_{ij} 值
            c_out[i, j] = u_prime_inv(β * R * expectation)

    a_out = c_out + s[:, None]
    return c_out, a_out
```

为了求解模型，我们使用一个简单的while循环。

```{code-cell} ipython3
def solve_model_numpy(
        ifp_numpy: IFPNumPy,
        c_init: np.ndarray,
        a_init: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 1_000
    ) -> np.ndarray:
    """
    使用带EGM的时间迭代法求解模型。

    """
    c_in, a_in = c_init, a_init
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        c_out, a_out = K_numpy(c_in, a_in, ifp_numpy)
        error = np.max(np.abs(c_out - c_in))
        i = i + 1
        c_in, a_in = c_out, a_out

    return c_out, a_out
```

让我们对EGM代码进行实测。

```{code-cell} ipython3
ifp_numpy = create_ifp()
R, β, γ, Π, z_grid, s = ifp_numpy
# 初始条件——代理人消费所有资产
a_init = s[:, None] * np.ones(len(z_grid))
c_init = a_init
# 从这些初始条件求解
c_vec, a_vec = solve_model_numpy(
    ifp_numpy, c_init, a_init
)
```

下面是每个 $z$ 状态下最优消费策略的图形

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(a_vec[:, 0], c_vec[:, 0], label='bad state')
ax.plot(a_vec[:, 1], c_vec[:, 1], label='good state')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()
```

## JAX 实现

```{index} single: Optimal Savings; Programming Implementation
```

现在我们编写一个效率更高的 JAX 版本，它可以在 GPU 上运行。

### 建模

我们从一个名为 `IFP` 的类开始，它用来存储模型的基本要素。

```{code-cell} ipython3
class IFP(NamedTuple):
    R: float                  # 总利率 R = 1 + r
    β: float                  # 折现因子
    γ: float                  # 偏好参数
    Π: jnp.ndarray            # 外生冲击的马尔可夫矩阵
    z_grid: jnp.ndarray       # Z_t 的马尔可夫状态值
    s: jnp.ndarray            # 外生储蓄网格


def create_ifp(r=0.01,
               β=0.94,
               γ=1.5,
               Π=((0.6, 0.4),
                  (0.05, 0.95)),
               z_grid=(-10.0, jnp.log(2.0)),
               savings_grid_max=16,
               savings_grid_size=200):

    s = jnp.linspace(0, savings_grid_max, savings_grid_size)
    Π, z_grid = jnp.array(Π), jnp.array(z_grid)
    R = 1 + r
    assert R * β < 1, "Stability condition violated."
    return IFP(R, β, γ, Π, z_grid, s)
```

### 求解器

下面是算子 $K$，它将当前的猜测值 $\sigma$ 转换为下一期的猜测值 $K\sigma$。

```{code-cell} ipython3
def K(
        c_in: jnp.ndarray,
        a_in: jnp.ndarray,
        ifp: IFP
    ) -> jnp.ndarray:
    """
    使用内生网格法求解IFP模型的欧拉方程算子。

    该算子实现EGM算法的一次迭代，用于更新消费策略函数。

    """
    R, β, γ, Π, z_grid, s = ifp
    n_z = len(z_grid)
    z_indices = jnp.arange(n_z)
    u_prime = lambda c: c**(-γ)
    u_prime_inv = lambda c: c**(-1/γ)
    y = lambda z: jnp.exp(z)

    def compute_c(i, j):
        " 计算某一对 (i, j) 的消费值，其中 i >= 1。 "

        def compute_mu_k(k):
            " 给定 i，计算边际效用 u'(σ(R s_i + y(z_k), z_k)) "
            next_a = R * s[i] + y(z_grid[k])
            # 插值以得到 σ(R * s_i + y(z_k), z_k)
            next_c = jnp.interp(next_a, a_in[:, k], c_in[:, k])
            # 返回 u'(σ(R * s_i + y(z_k), z_k))
            return u_prime(next_c)

        # 对所有 k 计算边际效用 u'(σ(R * s_i + y(z_k), z_k))
        mu_values = jax.vmap(compute_mu_k)(z_indices)
        # 计算期望值 Σ_k u'(σ(...)) * Π[j, k]
        expectation = jnp.sum(mu_values * Π[j, :])
        # 求逆得到在 (s_i, z_j) 处的消费 c_{ij}
        return u_prime_inv(β * R * expectation)

    # 对每个 i，在 j 上进行 vmap
    compute_c_i = jax.vmap(compute_c, in_axes=(None, 0))
    # 在 i 上进行 vmap
    compute_c = jax.vmap(lambda i: compute_c_i(i, z_indices))
    # 计算 i >= 1 时的消费
    c_out_interior = compute_c(jnp.arange(1, len(s)))  
    # 对 i = 0，将消费设为0
    c_out_boundary = jnp.zeros((1, n_z))
    # 拼接边界与内部部分
    c_out = jnp.concatenate([c_out_boundary, c_out_interior], axis=0)
    # 计算内生资产网格 a_{ij} = c_{ij} + s_i
    a_out = c_out + s[:, None]
    return c_out, a_out
```

下面是一个经过 jit 加速的迭代程序，使用该算子求解模型。

```{code-cell} ipython3
@jax.jit
def solve_model(
        ifp: IFP,
        c_init: jnp.ndarray,  # σ在内生网格上的初始猜测
        a_init: jnp.ndarray,  # 初始内生网格
        tol: float = 1e-5,
        max_iter: int = 1000
    ) -> jnp.ndarray:
    """
    使用带EGM的时间迭代法求解模型。

    """

    def condition(loop_state):
        c_in, a_in, i, error = loop_state
        return (error > tol) & (i < max_iter)

    def body(loop_state):
        c_in, a_in, i, error = loop_state
        c_out, a_out = K(c_in, a_in, ifp)
        error = jnp.max(jnp.abs(c_out - c_in))
        i += 1
        return c_out, a_out, i, error

    i, error = 0, tol + 1
    initial_state = (c_init, a_init, i, error)
    final_loop_state = jax.lax.while_loop(condition, body, initial_state)
    c_out, a_out, i, error = final_loop_state

    return c_out, a_out
```

### 测试运行

让我们对EGM代码进行实测。

```{code-cell} ipython3
ifp = create_ifp()
R, β, γ, Π, z_grid, s = ifp
# 设置初始条件，代理人消费所有资产
a_init = s[:, None] * jnp.ones(len(z_grid))
c_init = a_init
# 从这些初始条件开始求解
c_vec_jax, a_vec_jax = solve_model(ifp, c_init, a_init)
```

为了验证我们的 JAX 实现的正确性，让我们将其与之前开发的 NumPy 版本进行比较。

```{code-cell} ipython3
# 比较结果
max_c_diff = np.max(np.abs(np.array(c_vec) - c_vec_jax))
max_ae_diff = np.max(np.abs(np.array(a_vec) - a_vec_jax))

print(f"Maximum difference in consumption policy: {max_c_diff:.2e}")
print(f"Maximum difference in asset grid:        {max_ae_diff:.2e}")
```

这些数字证实，我们用这两种方法计算出的策略基本上是相同的。

### 计时

现在让我们比较一下 NumPy 和 JAX 实现之间的执行时间。

```{code-cell} ipython3
import time

# 为NumPy版本设置初始条件
s_np = np.array(s)
z_grid_np = np.array(z_grid)
a_init_np = s_np[:, None] * np.ones(len(z_grid_np))
c_init_np = a_init_np.copy()

# 为JAX版本设置初始条件
a_init_jx = s[:, None] * jnp.ones(len(z_grid))
c_init_jx = a_init_jx

# 计时NumPy版本
start = time.time()
c_vec_np, a_vec_np = solve_model_numpy(ifp_numpy, c_init_np, a_init_np)
numpy_time = time.time() - start

# 计时JAX版本（含编译时间）
start = time.time()
c_vec_jx, a_vec_jx = solve_model(ifp, c_init_jx, a_init_jx)
c_vec_jx.block_until_ready()
jax_time_with_compile = time.time() - start

# 计时JAX版本（不含编译时间——第二次运行）
start = time.time()
c_vec_jx, a_vec_jx = solve_model(ifp, c_init_jx, a_init_jx)
c_vec_jx.block_until_ready()
jax_time = time.time() - start

print(f"NumPy time:                 {numpy_time:.4f} seconds")
print(f"JAX time (with compile):    {jax_time_with_compile:.4f} seconds")
print(f"JAX time (without compile): {jax_time:.4f} seconds")
print(f"Speedup (NumPy/JAX):        {numpy_time/jax_time:.2f}x")
```

由于 JIT 编译以及 GPU/TPU 加速（如果可用），JAX 实现的速度更快。

下面是每个 $z$ 状态下最优策略的图形

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(a_vec[:, 0], c_vec[:, 0], label='bad state')
ax.plot(a_vec[:, 1], c_vec[:, 1], label='good state')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()
```

### 动态性质

为了初步了解在默认参数下家庭长期持有的资产水平，让我们来看一个 45 度线图，
它展示了在最优消费策略下资产的运动规律。

```{code-cell} ipython3
fig, ax = plt.subplots()

y = lambda z: jnp.exp(z)

def y_bar(k):
    """
    取 z = z_grid[k]，计算

            E_z Y' = Σ_{z'} y(z') Π[z, z']

    这是在 Z_t = z 条件下 Y_{t+1} 的期望值。
    """
    # 对所有 z' 计算 y(z')
    y_values = jax.vmap(y)(z_grid)
    # 按转移概率加权并求和
    return jnp.sum(y_values * Π[k, :])

for k, label in zip((0, 1), ('low income', 'high income')):
    # 在储蓄网格上对消费策略进行插值
    c_on_grid = jnp.interp(s, a_vec[:, k], c_vec[:, k])
    ax.plot(s, R * (s - c_on_grid) + y_bar(k) , label=label)

ax.plot(s, s, 'k--')
ax.set(xlabel='current assets', ylabel='next period assets')

ax.legend()
plt.show()
```

实线显示了每个 $z$ 值下的资产更新函数，即

$$
    a \mapsto R (a - \sigma^*(a, z)) + \bar{y}(z)
$$

其中

$$
    \bar{y}(z) := \sum_{z'} y(z') \Pi(z, z')
$$

是以当前状态 $z$ 为条件的预期劳动收入。

虚线是45度线。

从图中可以看出，平均而言，这个动态过程是稳定的 --- 即使在最高状态下，资产也不会发散。

事实证明这是对的：资产存在唯一的平稳分布。

* 具体细节参见{cite}`ma2020income`

当家庭面临异质性冲击时，这个平稳分布代表了家庭之间资产的长期分布情况。

### 合理性检查

检查我们结果的一种方法是

* 将每个状态的劳动收入设为零，且
* 将总利率 $R$ 设为1。

在这种情况下，我们的收入波动问题就变成了一个 CRRA 吃蛋糕问题。

那么价值函数和最优消费策略由以下给出：

```{code-cell} ipython3
def c_star(x, β, γ):
    return (1 - β ** (1/γ)) * x


def v_star(x, β, γ):
    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

让我们来看看是否一致：

```{code-cell} ipython3
ifp_cake_eating = create_ifp(r=0.0, z_grid=(-jnp.inf, -jnp.inf))
R, β, γ, Π, z_grid, s = ifp_cake_eating
a_init = s[:, None] * jnp.ones(len(z_grid))
c_init = a_init
c_vec, a_vec = solve_model(ifp_cake_eating, c_init, a_init)

fig, ax = plt.subplots()
ax.plot(a_vec[:, 0], c_vec[:, 0], label='numerical')
ax.plot(a_vec[:, 0],
        c_star(a_vec[:, 0], ifp_cake_eating.β, ifp_cake_eating.γ),
        '--', label='analytical')
ax.set(xlabel='assets', ylabel='consumption')
ax.legend()
plt.show()
```

这看起来相当不错。

## 模拟

让我们回到默认模型，研究资产的平稳分布。

我们的计划是让大量家庭向前模拟 $T$ 期，然后对资产的横截面分布绘制直方图。

设定 `num_households=50_000, T=500`。

首先我们编写一个函数，用来将单个家庭向前模拟并记录最终的资产值。

该函数接收一对解 `c_vec` 和 `a_vec`，将它们理解为与给定模型 `ifp`
相关联的最优策略。

```{code-cell} ipython3
@jax.jit
def simulate_household(
        key, a_0, z_idx_0, c_vec, a_vec, ifp, T
    ):
    """
    对单个家庭模拟T期，以近似资产的平稳分布。

    - key 是随机数生成器的状态
    - ifp 是 IFP 的一个实例
    - c_vec, a_vec 是 ifp 的最优消费策略、内生网格

    """
    R, β, γ, Π, z_grid, s = ifp
    n_z = len(z_grid)

    y = lambda z: jnp.exp(z)
    σ = lambda a, z_idx: jnp.interp(a, a_vec[:, z_idx], c_vec[:, z_idx])

    # 向前模拟T期
    def update(t, state):
        a, z_idx = state
        # 从 Π[z, z'] 中抽取下一期冲击 z'
        current_key = jax.random.fold_in(key, t)
        z_next_idx = jax.random.choice(current_key, n_z, p=Π[z_idx]).astype(jnp.int32)
        z_next = z_grid[z_next_idx]
        # 更新资产：a' = R * (a - c) + Y'
        a_next = R * (a - σ(a, z_idx)) + y(z_next)
        # 返回更新后的状态
        return a_next, z_next_idx

    initial_state = a_0, z_idx_0
    final_state = jax.lax.fori_loop(0, T, update, initial_state)
    a_final, _ = final_state
    return a_final
```

现在我们编写一个函数，用来并行模拟多个家庭。

```{code-cell} ipython3
def compute_asset_stationary(
        c_vec, a_vec, ifp, num_households=50_000, T=500, seed=1234
    ):
    """
    对num_households个家庭模拟T期，以近似资产的平稳分布。

    返回资产持有量的最终横截面。

    - ifp 是 IFP 的一个实例
    - c_vec, a_vec 是最优消费策略和内生网格。

    """
    R, β, γ, Π, z_grid, s = ifp
    n_z = len(z_grid)

    # 为消费策略创建插值函数
    # 在内生网格上进行插值
    σ = lambda a, z_idx: jnp.interp(a, a_vec[:, z_idx], c_vec[:, z_idx])

    # 从资产 = savings_grid_max / 2 开始
    a_0_vector = jnp.full(num_households, s[-1] / 2)
    # 初始化每个家庭的外生状态
    z_idx_0_vector = jnp.zeros(num_households).astype(jnp.int32)

    # 对多个家庭进行向量化
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_households)
    # 对 (key, a_0, z_idx_0) 向量化 simulate_household
    sim_all_households = jax.vmap(
        simulate_household, in_axes=(0, 0, 0, None, None, None, None)
    )
    assets = sim_all_households(keys, a_0_vector, z_idx_0_vector, c_vec, a_vec, ifp, T)

    return np.array(assets)
```

现在我们调用该函数，生成资产分布并绘制直方图：

```{code-cell} ipython3
ifp = create_ifp()
R, β, γ, Π, z_grid, s = ifp
a_init = s[:, None] * jnp.ones(len(z_grid))
c_init = a_init
c_vec, a_vec = solve_model(ifp, c_init, a_init)
assets = compute_asset_stationary(c_vec, a_vec, ifp)

fig, ax = plt.subplots()
ax.hist(assets, bins=20, alpha=0.5, density=True)
ax.set(xlabel='assets', title="Cross-sectional distribution of wealth")
plt.show()
```

这个财富分布看起来与真实数据中典型的财富分布非常不同。

首先，它是左偏而非右偏的。

事实上，即使现实世界中的财富分布有着长长的右尾，这里也几乎没有右尾。

在接下来的几讲中，我们将尽力修正这些问题。
