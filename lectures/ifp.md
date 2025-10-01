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


# 收入波动问题 I：基本模型

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

在本讲中，我们研究一个无限期存活的消费者的最优储蓄问题——即{cite}`Ljungqvist2012`第1.3节中描述的“共同祖先”。

这是许多代表性宏观经济模型的重要子问题。这些宏观模型包括
* {cite}`Aiyagari1994`
* {cite}`Huggett1993`
* 等等

它与{doc}`随机最优增长模型 <optgrowth>`中的决策问题相关，但在一些重要方面有所不同。

例如，代理人的选择问题包含一个附加收入项，这会导致一个偶尔起约束作用的约束条件。

此外，在本讲及后续讲座中，我们将引入更多现实的特征，如相关性冲击。

为了求解该模型，我们将使用基于欧拉方程的时间迭代法。这一方法在我们研究{doc}`随机最优增长模型 <optgrowth>`时被证明是{doc}`快速且准确的 <coleman_policy_iter>`。

在较弱的假设下，时间迭代是全局收敛的，即使效用函数是无界的（无论向上还是向下）。

我们需要以下导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from quantecon.optimize import brentq
from numba import jit, float64
from numba.experimental import jitclass
from quantecon import MarkovChain
```

### 参考文献

我们的演示是{cite}`ma2020income`的简化版本。

其他参考文献包括{cite}`Deaton1991`、{cite}`DenHaan2010`、
{cite}`Kuhn2013`、{cite}`Rabault2002`、{cite}`Reiter2009`和
{cite}`SchechtmanEscudero1977`。

## 最优储蓄问题

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

a_{t+1} \leq  R(a_t - c_t)  + Y_{t+1},
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

1. 在时期 $t$ 开始时，家庭选择消费 $c_t$。
1. 家庭在整个时期提供劳动，并在时期 $t$ 结束时收到劳动收入 $Y_{t+1}$。
1. 金融收入 $R(a_t - c_t)$ 在时期 $t$ 结束时收到。
1. 时间转移到 $t+1$ 并重复此过程。

非资本收入 $Y_t$ 由 $Y_t = y(Z_t)$ 给出，其中 $\{Z_t\}$ 是一个外生状态过程。

按照文献中的常见做法，我们将 $\{Z_t\}$ 视为一个有限状态马尔可夫链，其取值在 $\mathbb{Z}$ 中，对应的马尔可夫转移矩阵为 $P$。

我们进一步假设

1. $\beta R < 1$
1. $u$ 是光滑的、严格递增和严格凹的，且满足 $\lim_{c \to 0} u'(c) = \infty$ 和 $\lim_{c \to \infty} u'(c) = 0$

资产空间是 $\mathbb R_+$，状态是 $(a,z) \in \mathsf S := \mathbb R_+ \times \mathsf Z$。

从 $(a,z) \in \mathsf S$ 出发的*可行消费路径*是一个消费序列 $\{c_t\}$， 使得 $\{c_t\}$ 及其诱导的资产路径 $\{a_t\}$ 满足：

1. $(a_0, z_0) = (a, z)$
1. 可行性约束{eq}`eqst`，以及
1. 可测性，即 $c_t$ 是截至时期 $t$ 的随机结果的函数，而不是时期 $t$ 之后的结果的函数。

第三点的含义是：时期 $t$ 的消费不能依赖尚未被观测到的结果。

事实上，在这个问题中，消费的最优选择可以通过仅依赖当前状态来实现。

最优性定义如下。

### 价值函数和欧拉方程

*价值函数* $V \colon \mathsf S \to \mathbb{R}$ 定义为

```{math}
:label: eqvf

V(a, z) := \max \, \mathbb{E}
\left\{
\sum_{t=0}^{\infty} \beta^t u(c_t)
\right\}
```

其中最大化是在所有从 $(a,z)$ 出发的可行消费路径上进行的。

一个从 $(a,z)$ 出发的*最优消费路径*是一个从 $(a,z)$ 出发的可行消费路径，且这个路径达到了{eq}`eqvf`中的上确界。

为了刻画这样的路径，我们可以使用欧拉方程的一个版本，在当前设定中为

```{math}
:label: ee00

u' (c_t)
\geq \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

和

```{math}
:label: ee01

c_t < a_t
\; \implies \;
u' (c_t) = \beta R \,  \mathbb{E}_t  u'(c_{t+1})
```

当 $c_t = a_t$ 时，显然有 $u'(c_t) = u'(a_t)$，

当 $c_t$ 达到上界 $a_t$ 时，严格不等式 $u' (c_t) > \beta R \,  \mathbb{E}_t  u'(c_{t+1})$ 可能出现，因为 $c_t$ 无法再增加以达到等式

（最优解中永远不会出现下界情况 $c_t = 0$，因为 $u'(0) = \infty$。）

稍加思考可以表明，可以证明 {eq}`ee00` 和 {eq}`ee01` 等价于

```{math}
:label: eqeul0

u' (c_t)
= \max \left\{
    \beta R \,  \mathbb{E}_t  u'(c_{t+1})  \,,\;  u'(a_t)
\right\}
```

### 最优性结果

正如 {cite}`ma2020income` 所示，

1. 对于每个 $(a,z) \in \mathsf S$，从 $(a,z)$ 出发存在唯一的最优消费路径

1. 这条路径是从 $(a,z)$ 出发的、满足欧拉等式{eq}`eqeul0`和横截条件
```{math}
:label: eqtv

\lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0
```
的唯一可行路径。

此外，存在一个*最优消费函数* $\sigma^* \colon \mathsf S \to \mathbb R_+$，使得从 $(a,z)$ 出发的路径

$$
(a_0, z_0) = (a, z),
\quad
c_t = \sigma^*(a_t, Z_t)
\quad \text{和} \quad
a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

同时满足{eq}`eqeul0`和{eq}`eqtv`，因此是从 $(a,z)$ 出发的唯一最优路径。

因此，为了解决这个优化问题，我们需要计算策略 $\sigma^*$。

## 计算

计算 $\sigma^*$ 有两种标准方法

1. 使用欧拉等式的时间迭代法
1. 值函数迭代法

我们对吃蛋糕问题和随机最优增长模型的研究表明，时间迭代会更快且更精确。

这是我们在下文所采用的方法。

### 时间迭代

我们可以将{eq}`eqeul0`改写为一个关于函数而非随机变量的表达式。

具体来说，考虑函数方程

```{math}
:label: eqeul1

(u' \circ \sigma)  (a, z)
= \max \left\{
\beta R \, \mathbb E_z (u' \circ \sigma)
    [R (a - \sigma(a, z)) + \hat Y, \, \hat Z]
\, , \;
     u'(a)
     \right\}
```

其中

* $(u' \circ \sigma)(s) := u'(\sigma(s))$
* $\mathbb E_z$ 表示基于当前状态$z$的条件期望，$\hat X$表示随机变量 $X$ 的下一期值
* $\sigma$ 是未知函数

我们需要为最优消费策略选择一个合适的候选解类别。

正确的做法是考虑解应当具备哪些性质，从而限制搜索空间并确保迭代表现良好。

为此，令 $\mathscr C$ 为连续函数 $\sigma \colon \mathbf S \to \mathbb R$ 的空间，其中 $\sigma$ 对第一个参数是递增的，对所有 $(a,z) \in \mathbf S$ 都有 $0 < \sigma(a,z) \leq a$，且

```{math}
:label: ifpC4

\sup_{(a,z) \in \mathbf S}
\left| (u' \circ \sigma)(a,z) - u'(a) \right| < \infty
```

这将作为我们的候选解类。

此外，令 $K \colon \mathscr{C} \to \mathscr{C}$ 定义如下。

给定 $\sigma \in \mathscr{C}$，值 $K\sigma(a,z)$ 是唯一的 $c \in [0,a]$，使其满足：

```{math}
:label: eqsifc

u'(c)
= \max \left\{
           \beta R \, \mathbb E_z (u' \circ \sigma) \,
           [R (a - c) + \hat Y, \, \hat Z]
           \, , \;
           u'(a)
     \right\}
```

我们称 $K$ 为 Coleman–Reffett 算子。

算子 $K$ 的构造方式保证了 $K$ 的不动点与{eq}`eqeul1`的解一致。

正如{cite}`ma2020income`所示，通过选取任意 $\sigma \in \mathscr{C}$ 并使用{eq}`eqsifc`中定义的算子 $K$ 进行迭代，可以计算出唯一的最优策略。

### 一些技术细节

最后一个结论的证明在技术上比较复杂，但这里给出一个简要总结：

在{cite}`ma2020income`中证明，在以下度量下，$K$ 是 $\mathscr{C}$ 上的压缩映射

$$
\rho(c, d) := \| \, u' \circ \sigma_1 - u' \circ \sigma_2 \, \|
    := \sup_{s \in S} | \, u'(\sigma_1(s))  - u'(\sigma_2(s)) \, |
 \qquad \quad (\sigma_1, \sigma_2 \in \mathscr{C})
$$

该度量衡量的是在边际效用意义下的最大差异。

（这种距离度量的好处在于，虽然 $\mathscr C$ 中的元素通常不是有界的，但在我们的假设下 $\rho$ 始终是有限的。）

还可以证明，度量 $\rho$ 在 $\mathscr{C}$ 上是完备的。

因此，$K$ 在 $\mathscr{C}$ 中有唯一的不动点 $\sigma^*$，且对于任意 $\sigma \in \mathscr{C}$，当 $n \to \infty$ 时，$K^n c \to \sigma^*$。

根据 $K$ 的定义，$K$ 在 $\mathscr{C}$ 中的不动点与方程 {eq}`eqeul1` 在 $\mathscr{C}$ 中的解相一致。

因此，从 $(a_0,z_0) \in S$ 出发，使用策略函数 $\sigma^*$ 生成的路径 $\{c_t\}$ 是从 $(a_0,z_0) \in S$ 出发的唯一最优路径。

## 实现

我们使用 CRRA 效用函数：

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

外生状态过程 $\{Z_t\}$ 默认为一个两状态马尔可夫链，其状态空间为 $\{0, 1\}$，转移矩阵为 $P$。

这里我们构建一个名为 `IFP` 的类，用来存储模型的基本要素。

```{code-cell} ipython3
ifp_data = [
    ('R', float64),              # 利率 1 + r
    ('β', float64),              # 折现因子
    ('γ', float64),              # 偏好参数
    ('P', float64[:, :]),        # Z_t 的二元马尔可夫矩阵
    ('y', float64[:]),           # 收入为 Y_t = y[Z_t]
    ('asset_grid', float64[:])   # 资产网格（数组）
]

@jitclass(ifp_data)
class IFP:

    def __init__(self,
                 r=0.01,
                 β=0.96,
                 γ=1.5,
                 P=((0.6, 0.4),
                    (0.05, 0.95)),
                 y=(0.0, 2.0),
                 grid_max=16,
                 grid_size=50):

        self.R = 1 + r
        self.β, self.γ = β, γ
        self.P, self.y = np.array(P), np.array(y)
        self.asset_grid = np.linspace(0, grid_max, grid_size)

        # 注意我们需要 R β < 1 以确保收敛
        assert self.R * self.β < 1, "稳定性条件被违反。"

    def u_prime(self, c):
        return c**(-self.γ)
```

接下来我们定义一个函数，用来计算下式的差值：

```{math}
:label: euler_diff_eq

u'(c) - \max \left\{
           \beta R \, \mathbb E_z (u' \circ \sigma) \,
           [R (a - c) + \hat Y, \, \hat Z]
           \, , \;
           u'(a)
     \right\}
```

```{code-cell} ipython3
@jit
def euler_diff(c, a, z, σ_vals, ifp):
    """
    基于当前策略σ，计算欧拉方程左右两边的差值。

        * c 是消费选择
        * (a, z) 是状态，其中 z ∈ {0, 1}
        * σ_vals 是以矩阵形式表示的策略
        * ifp 是 IFP 的一个实例

    """

    # 简化名称
    R, P, y, β, γ  = ifp.R, ifp.P, ifp.y, ifp.β, ifp.γ
    asset_grid, u_prime = ifp.asset_grid, ifp.u_prime
    n = len(P)

    # 通过线性插值将策略转换为函数
    def σ(a, z):
        return np.interp(a, asset_grid, σ_vals[:, z])

    # 计算基于当前z的期望值
    expect = 0.0
    for z_hat in range(n):
        expect += u_prime(σ(R * (a - c) + y[z_hat], z_hat)) * P[z, z_hat]

    return u_prime(c) - max(β * R * expect, u_prime(a))
```

请注意，我们在资产网格上使用线性插值来近似策略函数。

下一步是求取欧拉方程差的根。

```{code-cell} ipython3
@jit
def K(σ, ifp):
    """
    算子K。

    """
    σ_new = np.empty_like(σ)
    for i, a in enumerate(ifp.asset_grid):
        for z in (0, 1):
            result = brentq(euler_diff, 1e-8, a, args=(a, z, σ, ifp))
            σ_new[i, z] = result.root

    return σ_new
```

有了算子 $K$ 之后，我们可以选择一个初始条件并开始迭代。

下面的函数执行迭代直至收敛，并返回近似的最优策略。

```{code-cell} ipython3
def solve_model_time_iter(model,    # 模型信息类
                          σ,        # 初始条件
                          tol=1e-4,
                          max_iter=1000,
                          verbose=True,
                          print_skip=25):

    # 设置循环
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        σ_new = K(σ, model)
        error = np.max(np.abs(σ - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"第 {i} 次迭代的误差是 {error}.")
        σ = σ_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在 {i} 次迭代后收敛。")

    return σ_new
```

现在我们用 IFP 类的默认参数来运行：

```{code-cell} ipython3
ifp = IFP()

# 初始化消费策略：在所有状态 z 下消费所有资产
z_size = len(ifp.P)
a_grid = ifp.asset_grid
a_size = len(a_grid)
σ_init = np.repeat(a_grid.reshape(a_size, 1), z_size, axis=1)

σ_star = solve_model_time_iter(ifp, σ_init)
```

下面是每个外生状态 $z$ 下得到的最优策略函数的绘图：

```{code-cell} ipython3
fig, ax = plt.subplots()
for z in range(z_size):
    label = rf'$\sigma^*(\cdot, {z})$'
    ax.plot(a_grid, σ_star[:, z], label=label)
ax.set(xlabel='资产', ylabel='消费')
ax.legend()
plt.show()
```

接下来的练习将带你完成几个应用，在这些应用中会计算策略函数。

### 合理性检查

检查我们结果的一种方法是

* 将每个状态的劳动收入设为零
* 将总利率 $R$ 设为1。

在这种情况下，我们的收入波动问题就变成了一个吃蛋糕问题。

我们知道，在这种情况下，价值函数和最优消费策略由以下给出：

```{code-cell} ipython3
def c_star(x, β, γ):

    return (1 - β ** (1/γ)) * x


def v_star(x, β, γ):

    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

现在我们来看看数值解和解析解是否一致：

```{code-cell} ipython3
ifp_cake_eating = IFP(r=0.0, y=(0.0, 0.0))

σ_star = solve_model_time_iter(ifp_cake_eating, σ_init)

fig, ax = plt.subplots()
ax.plot(a_grid, σ_star[:, 0], label='数值解')
ax.plot(a_grid, c_star(a_grid, ifp.β, ifp.γ), '--', label='解析解')

ax.set(xlabel='资产', ylabel='消费')
ax.legend()

plt.show()
```

成功！

## 练习

```{exercise-start}
:label: ifp_ex1
```

让我们考虑利率如何影响消费。

请复现下图，该图展示了在不同利率下（近似的）最优消费策略：

```{figure} /_static/lecture_specific/ifp/ifp_policies.png
```

* 除了`r`外，所有参数都使用默认值。
* `r`的取值范围为`np.linspace(0, 0.04, 4)`。
* 消费是相对于资产水平绘制的，其中收入冲击固定在最小值。

图中显示，较高的利率会促进储蓄，从而抑制消费。

```{exercise-end}
```

```{solution-start} ifp_ex1
:class: dropdown
```

参考答案：

```{code-cell} ipython3
r_vals = np.linspace(0, 0.04, 4)

fig, ax = plt.subplots()
for r_val in r_vals:
    ifp = IFP(r=r_val)
    σ_star = solve_model_time_iter(ifp, σ_init, verbose=False)
    ax.plot(ifp.asset_grid, σ_star[:, 0], label=f'$r = {r_val:.3f}$')

ax.set(xlabel='资产水平', ylabel='消费（低收入）')
ax.legend()
plt.show()
```

```{solution-end}
```


```{exercise-start}
:label: ifp_ex2
```

现在我们来考虑，在默认参数下，家庭长期持有的资产水平。

下图是一个 45 度线图，显示了在最优消费时，资产的运动规律：

```{code-cell} ipython3
ifp = IFP()

σ_star = solve_model_time_iter(ifp, σ_init, verbose=False)
a = ifp.asset_grid
R, y = ifp.R, ifp.y

fig, ax = plt.subplots()
for z, lb in zip((0, 1), ('低收入', '高收入')):
    ax.plot(a, R * (a - σ_star[:, z]) + y[z] , label=lb)

ax.plot(a, a, 'k--')
ax.set(xlabel='当前资产', ylabel='下一期资产')

ax.legend()
plt.show()
```

实线显示了每个 $z$ 值下的资产更新函数，即

$$
a \mapsto R (a - \sigma^*(a, z)) + y(z)
$$

虚线是45度线。

从图中可以看出，这个动态过程是稳定的 --- 即使在最高收入状态下，资产也不会发散。

事实上，资产存在唯一的平稳分布，我们可以通过模拟来计算：

* 这可以通过{cite}`HopenhaynPrescott1992`的定理2来证明。
* 它表示当家庭面临异质性性冲击时，家庭之间资产的长期分布情况。

由于满足遍历性，可以通过对单个长时间序列取平均来计算平稳分布。

因此，为了近似平稳分布，我们可以模拟一个资产的长时间序列，并绘制其直方图。

你的任务是生成这样一个直方图。

* 使用一个长度为500,000的单一时间序列 $\{a_t\}$。
* 由于时间序列足够长，初始条件 $(a_0, z_0)$ 并不重要；
* 使用`quantecon`中的`MarkovChain`类可能会对你有帮助。

```{exercise-end}
```

```{solution-start} ifp_ex2
:class: dropdown
```

首先我们编写一个函数，用来计算一条长资产序列：

```{code-cell} ipython3
def compute_asset_series(ifp, T=500_000, seed=1234):
    """
    在最优储蓄行为下，模拟长度为 T 的资产时间序列。
    
    ifp是IFP的一个实例
    """
    P, y, R = ifp.P, ifp.y, ifp.R  # 简化名称
    
    # 求解最优策略
    σ_star = solve_model_time_iter(ifp, σ_init, verbose=False)
    σ = lambda a, z: np.interp(a, ifp.asset_grid, σ_star[:, z])
    
    # 模拟外生状态过程
    mc = MarkovChain(P)
    z_seq = mc.simulate(T, random_state=seed)
    
    # 模拟资产路径
    a = np.zeros(T+1)
    for t in range(T):
        z = z_seq[t]
        a[t+1] = R * (a[t] - σ(a[t], z)) + y[z]
    return a
```

接着我们调用该函数，生成序列并绘制直方图：

```{code-cell} ipython3
ifp = IFP()
a = compute_asset_series(ifp)

fig, ax = plt.subplots()
ax.hist(a, bins=20, alpha=0.5, density=True)
ax.set(xlabel='资产')
plt.show()
```

资产分布的形状是不现实的。

这里它是左偏的，而在现实中分布通常有一个较长的右尾。

在{doc}`后续讲座 <ifp_advanced>`中，我们将通过为模型加入更现实的特征来修正这一点。

```{solution-end}
```

```{exercise-start}
:label: ifp_ex3
```

在练习1和2的基础上，我们现在来考察储蓄和总资产持有量如何随利率变化。

```{note}
可以参考{cite}`Ljungqvist2012`第18.6节获取本练习所涉及主题的更多背景知识。
```

在给定的模型参数下，资产平稳分布的均值可以被解释为一个单位质量的 事前相同 的家庭在面对异质性冲击时，经济中的总资本。

你的任务是研究这一总资本度量如何随利率变化。

按照传统，把价格（即利率）放在纵轴上。

横轴上放总资本，即在给定利率下，由资产平稳分布的均值计算得到。

```{exercise-end}
```

```{solution-start} ifp_ex3
:class: dropdown
```

参考答案

```{code-cell} ipython3
M = 25
r_vals = np.linspace(0, 0.02, M)
fig, ax = plt.subplots()

asset_mean = []
for r in r_vals:
    print(f'在r = {r}处求解')
    ifp = IFP(r=r)
    mean = np.mean(compute_asset_series(ifp, T=250_000))
    asset_mean.append(mean)
ax.plot(asset_mean, r_vals)

ax.set(xlabel='资本', ylabel='利率')

plt.show()
```

正如预期的那样，总储蓄随着利率的上升而增加。

```{solution-end}
```

