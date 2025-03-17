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

# {index}`收入波动问题 I：基本模型 <single: The Income Fluctuation Problem I: Basic Model>`

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

在本讲中，我们研究一个无限生命期消费者的最优储蓄问题——即{cite}`Ljungqvist2012`第1.3节中描述的"共同祖先"。

这是许多代表性宏观经济模型的重要子问题

* {cite}`Aiyagari1994`
* {cite}`Huggett1993`
* 等等

它与{doc}`随机最优增长模型 <optgrowth>`中的决策问题相关，但在重要方面有所不同。

例如，代理人的选择问题包含一个加性收入项，这导致了一个偶尔会出现的约束条件。

此外，在本讲及后续讲座中，我们将引入更多现实的特征，如相关性冲击。

为了求解该模型，我们将使用基于欧拉方程的时间迭代法，这在我们研究{doc}`随机最优增长模型 <optgrowth>`时被证明是{doc}`快速且准确的 <coleman_policy_iter>`。

在温和的假设条件下，时间迭代在全局范围内是收敛的，即使效用是无界的（上下都无界）。

我们需要以下导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
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

### 设置

考虑一个家庭，选择状态相关的消费计划$\{c_t\}_{t \geq 0}$以最大化

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

* $\beta \in (0,1)$是贴现因子
* $a_t$是t时期的资产持有量，借贷约束为$a_t \geq 0$
* $c_t$是消费
* $Y_t$是非资本收入（工资、失业补偿等）
* $R := 1 + r$，其中$r > 0$是储蓄利率

时间安排如下：

1. 在第 $t$ 期开始时，家庭选择消费
   $c_t$。
1. 家庭在整个期间提供劳动，并在第 $t$ 期末收到劳动收入
   $Y_{t+1}$。
1. 金融收入 $R(a_t - c_t)$ 在第 $t$ 期末收到。
1. 时间转移到 $t+1$ 并重复此过程。

非资本收入 $Y_t$ 由 $Y_t = y(Z_t)$ 给出，其中
$\{Z_t\}$ 是一个外生状态过程。

按照文献中的常见做法，我们将 $\{Z_t\}$ 视为一个有限状态
马尔可夫链，取值于 $\mathsf Z$，具有马尔可夫矩阵 $P$。

我们进一步假设

1. $\beta R < 1$
1. $u$ 是光滑的、严格递增和严格凹的，且 $\lim_{c \to 0} u'(c) = \infty$ 和 $\lim_{c \to \infty} u'(c) = 0$

资产空间是 $\mathbb R_+$，状态是对 $(a,z)
\in \mathsf S := \mathbb R_+ \times \mathsf Z$。

从 $(a,z) \in \mathsf S$ 开始的*可行消费路径*是一个消费

序列 $\{c_t\}$ 满足 $\{c_t\}$ 及其导出的资产路径 $\{a_t\}$ 需要：

1. $(a_0, z_0) = (a, z)$
1. 满足{eq}`eqst`中的可行性约束，以及
1. 可测性，即 $c_t$ 是直到时间 $t$ 的随机结果的函数，而不是之后的。

第三点的含义仅仅是时间 $t$ 的消费不能是尚未观察到的结果的函数。

事实上，对于这个问题，可以通过仅让消费依赖于当前状态来实现最优选择。

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

其中最大化是针对从$(a,z)$出发的所有可行消费路径。

从$(a,z)$出发的*最优消费路径*是从$(a,z)$出发的可行消费路径中实现{eq}`eqvf`中上确界的路径。

为了确定这些路径，我们可以使用欧拉方程的一个版本，在当前设定中为

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

当 $c_t = a_t$ 时，我们显然有 $u'(c_t) = u'(a_t)$，

当 $c_t$ 达到上界 $a_t$ 时，严格不等式 $u' (c_t) > \beta R \,  \mathbb{E}_t  u'(c_{t+1})$ 可能出现，因为 $c_t$ 无法充分增加以达到等式。

（最优解中永远不会出现下界情况 $c_t = 0$，因为 $u'(0) = \infty$。）

经过一些思考，可以证明 {eq}`ee00` 和 {eq}`ee01` 等价于

```{math}
:label: eqeul0

u' (c_t)
= \max \left\{
    \beta R \,  \mathbb{E}_t  u'(c_{t+1})  \,,\;  u'(a_t)
\right\}
```

### 最优性结果

如 {cite}`ma2020income` 所示，

1. 对于每个 $(a,z) \in \mathsf S$，从 $(a,z)$ 出发存在唯一的最优消费路径

1. 这条路径是从$(a,z)$出发的唯一可行路径，它满足欧拉等式{eq}`eqeul0`和横截条件

```{math}
:label: eqtv

\lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0
```

此外，存在一个*最优消费函数*$\sigma^* \colon \mathsf S \to \mathbb R_+$，使得从$(a,z)$生成的路径

$$
(a_0, z_0) = (a, z),
\quad
c_t = \sigma^*(a_t, Z_t)
\quad \text{和} \quad
a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

同时满足{eq}`eqeul0`和{eq}`eqtv`，因此是从$(a,z)$出发的唯一最优路径。

因此，为了解决这个优化问题，我们需要计算策略$\sigma^*$。

(ifp_computation)=
## 计算

```{index} single: Optimal Savings; Computation
```

计算$\sigma^*$有两种标准方法

1. 使用欧拉等式的时间迭代法
1. 值函数迭代法

我们对蛋糕食用问题和随机最优增长的研究

模型表明时间迭代将更快且更准确。

这就是我们下面要采用的方法。

### 时间迭代

我们可以重写{eq}`eqeul0`，使其成为关于函数而不是随机变量的表述。

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
* $\mathbb E_z$表示基于当前状态$z$的条件期望，$\hat X$表示随机变量$X$的下一期值
* $\sigma$是未知函数

我们需要为最优消费政策选择一个合适的候选解类别。

选择这样一个类别的正确方法是考虑解可能具有什么性质，以便限制搜索空间并确保迭代表现良好。

为此，令 $\mathscr C$ 为连续函数 $\sigma \colon \mathbf S \to \mathbb R$ 的空间，其中 $\sigma$ 对第一个参数是递增的，对所有 $(a,z) \in \mathbf S$ 都有 $0 < \sigma(a,z) \leq a$，且

```{math}
:label: ifpC4

\sup_{(a,z) \in \mathbf S}
\left| (u' \circ \sigma)(a,z) - u'(a) \right| < \infty
```

这将是我们的候选类。

此外，令 $K \colon \mathscr{C} \to \mathscr{C}$ 定义如下。

对给定的 $\sigma \in \mathscr{C}$，$K \sigma (a,z)$ 是解决以下方程的唯一 $c \in [0, a]$：

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

我们将 $K$ 称为 Coleman--Reffett 算子。

构造算子 $K$ 的目的是使得 $K$ 的不动点与泛函方程 {eq}`eqeul1` 的解重合。

在{cite}`ma2020income`中显示，通过选取任意$\sigma \in \mathscr{C}$并使用{eq}`eqsifc`中定义的算子$K$进行迭代，可以计算出唯一的最优策略。

### 一些技术细节

最后这个结论的证明在技术上有些复杂，但这里给出一个简要总结：

在{cite}`ma2020income`中证明，在以下度量下，$K$是$\mathscr{C}$上的压缩映射

$$
\rho(c, d) := \| \, u' \circ \sigma_1 - u' \circ \sigma_2 \, \|
    := \sup_{s \in S} | \, u'(\sigma_1(s))  - u'(\sigma_2(s)) \, |
 \qquad \quad (\sigma_1, \sigma_2 \in \mathscr{C})
$$

该度量计算边际效用的最大差异。

（这种距离度量的好处在于，虽然$\mathscr C$中的元素通常不是有界的，但在我们的假设下$\rho$始终是有限的。）

还证明了度量$\rho$在$\mathscr{C}$上是完备的。

因此，$K$ 在 $\mathscr{C}$ 中有唯一的不动点 $\sigma^*$，且对于任意 $\sigma \in \mathscr{C}$，当 $n \to \infty$ 时，$K^n c \to \sigma^*$。

根据 $K$ 的定义，$K$ 在 $\mathscr{C}$ 中的不动点与方程 {eq}`eqeul1` 在 $\mathscr{C}$ 中的解相一致。

因此，从 $(a_0,z_0) \in S$ 出发，使用策略函数 $\sigma^*$ 生成的路径 $\{c_t\}$ 是从 $(a_0,z_0) \in S$ 出发的唯一最优路径。

## 实现

```{index} single: Optimal Savings; Programming Implementation
```

我们使用 CRRA 效用函数规范

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

外生状态过程 $\{Z_t\}$ 默认为一个两状态马尔可夫链，其状态空间为 $\{0, 1\}$，转移矩阵为 $P$。

这里我们构建一个名为 `IFP` 的类来存储模型的基本要素。

```{code-cell} ipython3
ifp_data = [
    ('R', float64),              # 利率 1 + r
    ('β', float64),              # 贴现因子
    ('γ', float64),              # 偏好参数
    ('P', float64[:, :]),        # Z_t 的二元马尔可夫矩阵
    ('y', float64[:]),           # 收入为 Y_t = y[Z_t]
    ('asset_grid', float64[:])   # 网格（数组）
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

接下来我们提供一个函数来计算差值

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
    欧拉方程左右两边的差值，基于当前策略σ。

        * c 是消费选择
        * (a, z) 是状态，其中 z 在 {0, 1} 中
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

请注意，我们沿着资产网格使用线性插值来近似策略函数。

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

下面的函数进行迭代直至收敛，并返回近似的最优策略。

```{code-cell} ipython3
:load: _static/lecture_specific/coleman_policy_iter/solve_time_iter.py
```

让我们使用`IFP`类的默认参数来执行这个过程：

```{code-cell} ipython3
ifp = IFP()

# 设置初始消费策略，即在所有z状态下消费所有资产
z_size = len(ifp.P)
a_grid = ifp.asset_grid
a_size = len(a_grid)
σ_init = np.repeat(a_grid.reshape(a_size, 1), z_size, axis=1)

σ_star = solve_model_time_iter(ifp, σ_init)
```

这是每个外生状态 $z$ 对应的最优策略图。

```{code-cell} ipython3
fig, ax = plt.subplots()
for z in range(z_size):
    label = rf'$\sigma^*(\cdot, {z})$'
    ax.plot(a_grid, σ_star[:, z], label=label)
ax.set(xlabel='资产', ylabel='消费')
ax.legend()
plt.show()
```

以下练习将带您完成几个需要计算政策函数的应用。

### 合理性检查

检查我们结果的一种方法是

* 将每个状态的劳动收入设为零
* 将总利率 $R$ 设为1。

在这种情况下，我们的收入波动问题就变成了一个蛋糕消费问题。

我们知道，在这种情况下，价值函数和最优消费政策由以下给出：

```{code-cell} ipython3
:load: _static/lecture_specific/cake_eating_numerical/analytical.py
```

让我们看看是否匹配：

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

复现下图，该图显示了不同利率下的（近似）最优消费策略

```{figure} /_static/lecture_specific/ifp/ifp_policies.png
```

* 除了`r`外，所有参数都使用默认值。
* `r`的取值范围为`np.linspace(0, 0.04, 4)`。
* 消费相对于资产绘制，收入冲击固定在最小值。

图中显示，较高的利率会促进储蓄，从而抑制消费。

```{exercise-end}
```

```{solution-start} ifp_ex1
:class: dropdown
```

这是一个解决方案：

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

现在让我们考虑在默认参数下家庭长期持有的资产水平。

下图是一个45度图,显示了在消费最优时资产的变动规律

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

实线显示了每个 $z$ 值下资产的更新函数，即

$$
a \mapsto R (a - \sigma^*(a, z)) + y(z)
$$

虚线是45度线。

从图中可以看出，这个动态过程是稳定的 --- 即使在最高状态下，资产也不会发散。

事实上，存在一个我们可以通过模拟计算的唯一的平稳分布

* 这可以通过{cite}`HopenhaynPrescott1992`的定理2来证明。
* 它表示当家庭面临特质性冲击时，家庭之间资产的长期分布情况。

这里符合遍历性，因此平稳概率可以通过对单个长时间序列取平均值来计算。

因此，为了近似平稳分布，我们可以模拟一个较长的资产时间序列并绘制其直方图。

你的任务是生成这样一个直方图。

* 使用一个长度为500,000的单一时间序列 $\{a_t\}$。
* 考虑到这个时间序列的长度，初始条件 $(a_0, z_0)$ 并不重要。

* 使用`quantecon`中的`MarkovChain`类可能会对你有帮助。

```{exercise-end}
```

```{solution-start} ifp_ex2
:class: dropdown
```

首先我们编写一个函数来计算长期资产序列。

```{code-cell} ipython3
def compute_asset_series(ifp, T=500_000, seed=1234):
    """
    模拟资产的长度为T的时间序列，基于最优储蓄行为。
    
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

现在我们调用函数，生成序列并绘制直方图：

```{code-cell} ipython3
ifp = IFP()
a = compute_asset_series(ifp)

fig, ax = plt.subplots()
ax.hist(a, bins=20, alpha=0.5, density=True)
ax.set(xlabel='资产')
plt.show()
```

资产分布的形状不够真实。

这里呈现为左偏态，而实际上应该具有长右尾。

在{doc}`后续讲座 <ifp_advanced>`中，我们将通过在模型中添加更多真实特征来纠正这一点。

```{solution-end}
```

```{exercise-start}
:label: ifp_ex3
```

在练习1和2的基础上，让我们来看看储蓄和总体资产持有量如何随利率变化

```{note}
可以参考{cite}`Ljungqvist2012`第18.6节获取本练习所涉及主题的更多背景知识。
```

对于模型的给定参数化，资产稳态分布的均值可以被解释为一个经济体中的总资本，该经济体中有单位质量的面临特质性冲击的*事前*相同的家庭。

你的任务是研究这个总资本衡量指标如何随利率变化。

按照传统，将价格（即利率）放在纵轴上。


在横轴上放置总资本，该值通过给定利率下的平稳分布的均值计算得出。

```{exercise-end}
```

```{solution-start} ifp_ex3
:class: dropdown
```

这是一个解决方案

```{code-cell} ipython3
M = 25
r_vals = np.linspace(0, 0.02, M)
fig, ax = plt.subplots()

asset_mean = []
for r in r_vals:
    print(f'Solving model at r = {r}')
    ifp = IFP(r=r)
    mean = np.mean(compute_asset_series(ifp, T=250_000))
    asset_mean.append(mean)
ax.plot(asset_mean, r_vals)

ax.set(xlabel='capital', ylabel='interest rate')

plt.show()
```

正如预期的那样，总储蓄随着利率的上升而增加。

```{solution-end}
```

