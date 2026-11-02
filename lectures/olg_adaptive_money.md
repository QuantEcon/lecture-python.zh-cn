---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 世代交叠货币经济中的适应性主体
  headings:
    Overview: 概述
    The environment: 经济环境
    'Part 1: a stochastic deficit': 第一部分：随机赤字
    'Part 1: a stochastic deficit::Stationary rational expectations equilibrium': 平稳理性预期均衡
    'Part 1: a stochastic deficit::Learning by successive generations': 连续世代的学习
    'Part 1: a stochastic deficit::Do they find it?': 他们能找到均衡吗？
    'Part 1: a stochastic deficit::How much do the agents have to be told?': 需要告诉主体多少信息？
    'Part 2: a constant deficit and two steady states': 第二部分：恒定赤字与两个稳态
    'Part 2: a constant deficit and two steady states::Least squares dynamics': 最小二乘动态
    'Part 2: a constant deficit and two steady states::The stability reversal': 稳定性的反转
    'Part 2: a constant deficit and two steady states::The stability reversal::Why the reversal? E-stability': 为什么会反转？E-稳定性
    'Part 2: a constant deficit and two steady states::Marimon and Sunder''s experiment': 马里蒙与桑德的实验
    'Part 2: a constant deficit and two steady states::A warning': 一个警示
    A government learning the Phillips curve: 一个学习菲利普斯曲线的政府
    A government learning the Phillips curve::The model: 模型
    A government learning the Phillips curve::The self-confirming equilibrium: 自我确认均衡
    A government learning the Phillips curve::Constant-gain learning and escapes: 恒定增益学习与逃逸
    Concluding remarks: 结语
    Exercises: 练习
---

(olg_adaptive_money)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 世代交叠货币经济中的适应性主体

```{index} single: Bounded Rationality; Overlapping Generations
```

```{contents} Contents
:depth: 2
```

除了 Anaconda 中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概述

{doc}`bounded_rationality` 研究了具有多个理性预期均衡的货币模型，其中包括一个凯根货币需求模型，其均衡中包含一族连续的泡沫。

本讲座采用一个与之密切相关的模型：{cite:t}`Samuelson1958` 的法定货币世代交叠模型，采用的是布莱恩特、华莱士等人用来研究通货膨胀融资（即政府印钞以支付其支出）的那个版本。

我们将那些按照价格水平均衡运动规律进行预测的主体，替换为不按此规律预测的"适应性"主体。

我们将出于两个不同的目的进行这一替换。

**第一部分**在模型中引入一个*随机*的政府赤字，并探讨接连不断的世代能否通过反复试错，作为一个社会最终收敛到某个理性预期均衡。

它们确实可以。

每一代人都观察前辈的经历，并使用**罗宾斯-门罗**递归，朝着改善效用的方向调整自己的储蓄决策。

只要时间足够长，经济就会收敛到我们独立计算出的某个平稳均衡，并且在这两个平稳均衡中，它只能找到低通胀的那一个。

这个练习也表明了学习速度对所要学习内容复杂度的敏感程度：一个很少出现的赤字状态被学习得很慢，因为相关观测数据到来得很慢。

**第二部分**去除了随机性，转向一个更尖锐的问题。

在*恒定*赤字的情况下，模型存在**两个**平稳均衡：一个是低通胀均衡，一个是高通胀均衡。

低通胀均衡在帕累托意义上优于另一个均衡。

在理性预期动态下，低通胀均衡是**不稳定的**，而高通胀均衡是稳定的：理论把经济推向了坏的结果。

而在最小二乘学习下，这种稳定性正好**反转**过来。

这种反转并非数值上的偶然：它源于*E-稳定性*，这是 {doc}`bounded_rationality` 在 {ref}`br_one_recursion` 中阐述的学习稳定性条件。

当 {cite:t}`MarimonSunder1993` 用付费人类被试将这个经济体作为实验室实验来运行时，被试的行为表现得像适应性模型，而不像理性预期模型。

这是 {cite:t}`Sargent1993` 中支持适应性动态确实在发挥选择机制作用的最有力证据，不过本杰明·本塔尔（Benjamin Bental）提出的一个反例对此有所调和，提醒我们不要断言适应性总能可靠地选出*好的*均衡。

随后我们以**第三个应用**收尾，其中的适应性主体是一个正在学习菲利普斯曲线的政府。

此时最小二乘学习收敛到该模型唯一的自我确认均衡，该均衡存在通胀偏差；而一个保持恒定增益的政府则会反复**逃逸**到零通胀的拉姆齐结果，但这根本不是一个均衡。

正是这一现象萌发了萨金特后来关于《美国通货膨胀的征服》的研究，而从 {doc}`phillips_two_stories` 开始的菲利普斯曲线系列讲座对此进行了深入研究。

让我们先导入一些库。

```{code-cell} ipython3
import numpy as np
import pandas as pd
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
```

## 经济环境

该经济由生活两期的世代交叠主体构成。

在每个时点 $t \geq 1$，都会出生 $N$ 个相同的主体，他们年轻时禀赋 $w_1$ 单位的单一消费品，年老时禀赋 $w_2$ 单位。

一个年轻主体对生命周期消费 $(c_1, c_2)$ 的偏好由 $u(c_1) + u(c_2)$ 的期望值来排序。

我们全文使用对数效用，$u(c) = \ln c$。

政府通过印制货币来为其支出 $G_t$ 融资，须满足预算约束

```{math}
:label: govt_budget

G_t = \frac{H_t - H_{t-1}}{p_t},
```

其中 $H_t$ 是年轻人在 $t$ 期持有并带入 $t+1$ 期的货币存量，$p_t$ 是价格水平。

货币是唯一的价值储藏手段，因此年轻主体的储蓄 $s_t$ 完全采取实际余额的形式，市场出清要求

```{math}
:label: market_clearing

\frac{H_t}{p_t} = s_t N .
```

记 $R_t = p_t / p_{t+1}$ 为货币在 $t$ 期到 $t+1$ 期之间的总回报率。

将 {eq}`govt_budget` 与 {eq}`market_clearing` 结合，可以得到一个我们将反复用到的关系：货币的回报率完全由储蓄行为和赤字所决定，

```{math}
:label: return_from_saving

R_t = \frac{N s_{t+1} - G_{t+1}}{N s_t} .
```

只有当下一代愿意吸纳未偿还存量加上新发行部分时，货币才能保值。

```{note}
{eq}`return_from_saving` 值得稍作停留细想，因为正是它使得这个系统成为**自我参照**的系统，这一性质在 {doc}`ls_learning` 中有详细研究。

今天年轻人从其储蓄中获得的回报，取决于明天的年轻人愿意储蓄多少，而后者又取决于明天的年轻人预期能获得多少回报。

没有人的信念是关于某个外生过程的。
```

## 第一部分：随机赤字

现在让政府支出遵循一个具有转移概率的马尔可夫链，

$$
P(i, j) = \operatorname{Prob}\{G_{t+1} = \bar G_j \mid G_t = \bar G_i\},
$$

其取值范围是一个有限的水平列表 $G = [\bar G_1, \ldots, \bar G_n]$。

（萨金特写作 $\pi(i, j)$；我们将 $\pi$ 保留用于表示通货膨胀。）

### 平稳理性预期均衡

在平稳均衡中，储蓄取决于当前的赤字状态，因此由一个 $n \times 1$ 的向量 $s = [s_1, \ldots, s_n]$ 来描述，而货币的回报率由一个 $n \times n$ 的矩阵 $R(i,j)$ 来描述。

一个观察到 $G_t = \bar G_i$ 的年轻主体选择 $s_i$ 以最大化

$$
V(s_i) = u(w_1 - s_i) + \sum_j u\bigl(w_2 + s_i R(i,j)\bigr) P(i,j),
$$

其一阶条件为

```{math}
:label: olg_foc

u'(w_1 - s_i) = \sum_{j=1}^n u'\bigl(w_2 + s_i R(i,j)\bigr) R(i,j) P(i,j).
```

代入 {eq}`return_from_saving`——在平稳均衡中它写作 $R(i,j) = (s_j N - \bar G_j)/(s_i N)$——就把 {eq}`olg_foc` 变成了一个仅含储蓄率的 $n$ 个非线性方程组：

```{math}
:label: olg_equilibrium

u'(w_1 - s_i)
= \sum_{j=1}^n u'\left(w_2 + \frac{s_j N - \bar G_j}{N}\right)
  \cdot \frac{s_j N - \bar G_j}{s_i N} \cdot P(i,j) .
```

当且仅当 {eq}`olg_equilibrium` 对每个 $i$ 都存在满足 $s_i \in (0, w_1)$ 且每个回报都为正（即对每个 $j$ 都有 $s_j N > \bar G_j$）的解时，平稳均衡才存在。

下面的代码从一个初始猜测值出发求解 {eq}`olg_equilibrium`，并检验上述两个条件。

```{code-cell} ipython3
w1, w2, N = 20.0, 10.0, 1.0

def u_prime(c):
    return 1 / c                       # log utility

def equilibrium_residual(s, G, P):
    "Left minus right side of the equilibrium condition, one entry per deficit state."
    n = len(s)
    out = np.empty(n)
    for i in range(n):
        rhs = sum(u_prime(w2 + (s[j]*N - G[j])/N)
                  * (s[j]*N - G[j])/(s[i]*N)
                  * P[i, j] for j in range(n))
        out[i] = u_prime(w1 - s[i]) - rhs
    return out

def stationary_equilibrium(G, P, guess=4.0):
    "Solve the equilibrium condition from `guess`; check that the solution is an equilibrium."
    s, _, ier, msg = fsolve(equilibrium_residual, np.full(len(G), guess),
                            args=(G, P), full_output=True)
    R = np.array([[(s[j]*N - G[j])/(s[i]*N) for j in range(len(s))]
                  for i in range(len(s))])
    assert ier == 1, msg
    assert np.all((s > 0) & (s < w1)) and np.all(R > 0), "not an equilibrium"
    return s, R
```

按照 {cite:t}`Sargent1993` 的做法，我们研究两个仅在政府支出过程上有所不同的经济体，其中 $w_1 = 20$，$w_2 = 10$，$N = 1$，因此 $\bar G$ 是以每个年轻人计算的。

在第一个经济体中，赤字始终为零，且该链是一个公平的硬币过程。

```{code-cell} ipython3
G1, P1 = np.array([0.0, 0.0]), np.full((2, 2), 0.5)
s1, R1 = stationary_equilibrium(G1, P1)
print(f"saving rates      {np.round(s1, 4)}")
print(f"returns\n{np.round(R1, 4)}")
```

由于没有赤字需要融资，货币能保持其价值：主体在任一状态下都储蓄 $5$，且 $R \equiv 1$。

在第二个经济体中，状态1的赤字为 $0.8$，状态2的赤字为零，该链具有持久性。

```{code-cell} ipython3
G2 = np.array([0.8, 0.0])
P2 = np.array([[0.75, 0.25],
               [0.50, 0.50]])
s2, R2 = stationary_equilibrium(G2, P2)
print(f"saving rates      {np.round(s2, 4)}      (Sargent reports 4.211, 4.364)")
print(f"returns\n{np.round(R2, 4)}")
print("Sargent reports  [[0.81, 1.0362], [0.7817, 1.00]]")
```

两者都与正文中报告的数值相符。

请注意 $R$ 中的规律：每当政府即将出现赤字（第一列）时，货币的回报都较差，因为新发行的货币稀释了存量货币。

系统 {eq}`olg_equilibrium` 是非线性的，萨金特提醒我们，一般而言它可能有多个解。

经济体2存在第二个平稳均衡，如果我们让求解器从较低的储蓄率出发，就能找到它。

```{code-cell} ipython3
s2_hi, R2_hi = stationary_equilibrium(G2, P2, guess=1.2)    # the high-inflation equilibrium
print(f"saving rates      {np.round(s2_hi, 4)}")
print(f"returns\n{np.round(R2_hi, 4)}")
```

储蓄率要低得多，因此每次赤字都必须依靠大比例新增货币发行来融资，而持有货币进入赤字状态的回报率，从大约 $0.8$ 降到了 $0.24$ 到 $0.33$ 之间。

它的所有回报都是正的，所以这确实是一个真正的平稳均衡，只是通货膨胀率要高得多。

下文中的学习型经济从未找到过这个均衡，{ref}`olg_ex3` 说明了其中的原因。

### 连续世代的学习

现在从主体那里撤去均衡这一信息。

我们仍假设主体知道自己的效用函数，并记得像自己一样的前辈的经历，但他们**没有**被告知回报的分布，而这恰恰是 {eq}`olg_foc` 所需要的对象。

相反，每一代人朝着*已实现*的经验所表明的更优方向，调整前辈的储蓄决策。

一个储蓄了 $s$ 并获得回报 $R$ 的主体，其已实现的生命周期效用为

$$
U(s) = u(w_1 - s) + u(w_2 + s R),
$$

其导数为

$$
U'(s) = -u'(w_1 - s) + u'(w_2 + sR) R,
\qquad
U''(s) = u''(w_1 - s) + u''(w_2 + sR) R^2 .
$$

与均衡的联系在于：在理性预期均衡中，$V'(s_i) = E_t U'(s_i)$，$V''(s_i) = E_t U''(s_i)$，其中 $E_t$ 是以 $G_t = \bar G_i$ 为条件的。

因此，一个能够计算 $E_t U'$ 的主体只需将其设为零即可完成决策。

而我们的主体做不到这一点，必须转而根据经验来估计它。

他们使用**罗宾斯-门罗**算法，逐状态地应用。

设 $\tau_i$ 为到目前为止赤字状态 $\bar G_i$ 被访问的次数，$\gamma_\tau$ 为一个递减的增益序列。

规则为

```{math}
:label: robbins_monro

\begin{aligned}
M(i, \tau_i + 1) &= M(i, \tau_i) + \gamma_{\tau_i}\bigl(U''(s(i, \tau_i)) - M(i, \tau_i)\bigr), \\
s(i, \tau_i + 1) &= s(i, \tau_i) - \gamma_{\tau_i} M(i, \tau_i + 1)^{-1} U'(s(i, \tau_i)) .
\end{aligned}
```

$M$ 累积一个对 $E_i U''$ 的运行估计值，而储蓄规则则针对它采取一个牛顿步。

增益 $\gamma_\tau$ 所扮演的角色，与 {doc}`bounded_rationality` 中松弛参数 $\lambda$ 的角色相同：每个观测都会把规则向该观测所建议的方向移动一个比例 $\gamma_\tau$。

在松弛算法中，增益是常数；而这里它会递减——就像该讲座中布雷的最小二乘学习者一样——因此每一个新观测对规则的移动幅度都比前一个要小。

萨金特设定 $\gamma_\tau = 1/\tau$，并将 $M$ 的初值设为 $-1$。

我们的代码使用 $\gamma_\tau = 1/(\tau + 20)$，这使得最初的几步较为谨慎，就好像初始规则已经是从20次观测中学来的一样，并将 $M$ 的初值设为 $U''(s, 1)$，即若货币保持其价值时初始规则应有的曲率。

该算法的设计使得，*如果*储蓄规则收敛，那么 $M(i, \tau_i) \to E_i U''$，且极限解满足 $E_i U'(s_i) = 0$，而这恰好就是均衡条件 {eq}`olg_foc`。

至于它们是否收敛，以及当 {eq}`olg_equilibrium` 存在多个解时收敛到哪一个，则由一个相关常微分方程的稳定性决定，详见 {ref}`br_one_recursion` 中的说明。

该设定中有两个特征值得说明。

**两类主体。** 要评估一个储蓄决策，我们必须等到该主体消费的*两个*时期都已知晓，因此人口被分成"奇数"和"偶数"两个子序列。

奇数主体在奇数期重置其规则，且只从更早的奇数主体那里学习；偶数主体亦然。

萨金特指出，这种交替方式类似于实验室实验的设计。

{cite:t}`MarimonSunder1993`（其实验出现在第二部分）在每一期都重新随机抽取年轻被试，但 {cite:t}`Arifovic1996` 的双货币实验——在 {doc}`exchange_rate_learning` 中讨论——则是把被试分成两个固定的组，交替充当年轻一代，这与这里的奇数类和偶数类十分相似。

**投影机制。** {eq}`robbins_monro` 中没有任何东西能阻止牛顿步把储蓄推到低于赤字水平。

由 {eq}`govt_budget` 和 {eq}`market_clearing` 所隐含的价格水平 $p_t = H_{t-1}/(N s_t - G_t)$ 将不再为正，经济也就不复存在。

因此我们将每个状态的储蓄率限制在区间 $[\bar G_i/N + 0.4, \; w_1 - 0.4]$ 内，这是一个存在均衡价格水平的区域。

这类机制对于收敛定理也是必需的；{doc}`ls_learning` 详细讨论了投影机制。

该模拟先抽取整条赤字状态路径，然后让两类主体沿着这条路径进行学习。

```{code-cell} ipython3
def U_prime(s, R):
    return -1/(w1 - s) + R/(w2 + s*R)

def U_double(s, R):
    return -1/(w1 - s)**2 - R**2/(w2 + s*R)**2


def simulate(G, P, T=50_000, s0=None, seed=0, tau0=20, band=0.4, gain=None):
    """
    Overlapping generations learning a state-contingent saving rule.

    The gain is 1/(τ + tau0), where τ counts a class's visits to a state,
    unless a constant `gain` is supplied.  Returns the history of saving
    rules, shape (T, 2, n) = (time, class, state), and the deficit states.
    """
    n = len(G)
    states = qe.MarkovChain(P).simulate(ts_length=T, init=0, random_state=seed)
    s = np.array(s0, float)
    M = np.array([[U_double(s[j, k], 1.0) for k in range(n)] for j in range(2)])
    τ = np.zeros((2, n), int)
    floor = G/N + band                 # projection facility: saving must cover the deficit
    ceiling = w1 - band

    hist = np.empty((T, 2, n))
    prev = None

    for t in range(T):
        i = states[t]                  # today's deficit state
        j = t % 2                      # which class is young this period
        s_t = s[j, i]

        if prev is not None:           # last period's young now learn their return
            j_p, i_p, s_p = prev
            R = (N*s_t - G[i]) / (N*s_p)          # the return on currency
            τ[j_p, i_p] += 1
            γ = 1 / (τ[j_p, i_p] + tau0) if gain is None else gain
            M[j_p, i_p] += γ * (U_double(s_p, R) - M[j_p, i_p])
            s_new = s[j_p, i_p] - γ * U_prime(s_p, R) / M[j_p, i_p]
            s[j_p, i_p] = min(max(s_new, floor[i_p]), ceiling)

        prev = (j, i, s_t)
        hist[t] = s

    return hist, states
```

```{note}
这个模拟从不计算价格水平。

由于 {eq}`return_from_saving` 完全用储蓄率和赤字来表达货币的回报率，模型的*实际*部分是自足的。

这不仅仅是一种便利：只要政府运行赤字，这些经济体中的名义货币存量就会无限增长，因此一个直接跟踪 $H_t$ 和 $p_t$ 的模拟，会在学习收敛之前就早早发生数值溢出。
```

### 他们能找到均衡吗？

我们让两类主体都从远离均衡、且不对称的规则出发，让每个经济体运行50,000期。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Saving rates converge in both economies"
    name: fig-olg-saving
---
start = [[8.0, 2.0],
         [2.0, 8.0]]                   # deliberately far from equilibrium, and asymmetric

hist1, _ = simulate(G1, P1, s0=start)
hist2, _ = simulate(G2, P2, s0=start)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, hist, s_star, title in [(axes[0], hist1, s1, "Economy 1:  $\\bar G = (0, 0)$"),
                                (axes[1], hist2, s2, "Economy 2:  $\\bar G = (0.8, 0)$")]:
    for k, colour in enumerate(['C0', 'C1']):
        ax.plot(hist[:, 0, k], color=colour, lw=1, label=f"state {k+1}, even")
        ax.plot(hist[:, 1, k], color=colour, lw=1, ls=':', label=f"state {k+1}, odd")
        ax.axhline(s_star[k], color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel("$t$  (log scale)")
    ax.set_title(title)
axes[0].set_ylabel("saving rate")
axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
```

{numref}`fig-olg-saving` 表明，两个经济体都收敛了，而且两类主体虽然彼此互不学习，却都收敛到了同一条规则。

黑色虚线是先前从 {eq}`olg_equilibrium` 计算出的均衡储蓄率，这是模拟中的任何主体都无法获知的对象。

下表报告了在三个不同时间跨度下学得的规则，取两类主体的平均值。

```{code-cell} ipython3
rows = []
for T in (5_000, 50_000, 200_000):
    h1, _ = simulate(G1, P1, T=T, s0=start)
    h2, _ = simulate(G2, P2, T=T, s0=start)
    rows.append([f"learned, T = {T:,}", *h1[-1].mean(axis=0), *h2[-1].mean(axis=0)])
rows.append(["rational expectations", *s1, *s2])

pd.DataFrame(rows, columns=["", "econ 1 state 1", "econ 1 state 2",
                            "econ 2 state 1", "econ 2 state 2"]).set_index("").round(4)
```

收敛是缓慢的——这正是量级为 $1/\tau$ 的增益所带来的代价——但确实是收敛，而且收敛到了经济体2的低通胀均衡。

### 需要告诉主体多少信息？

主体针对每个赤字水平学习一个*独立*的储蓄率：实际上，他们是以**非参数**方式学习策略函数 $s = f(G)$，每个状态学习一个数值。

由于状态只有两个，这才是可行的，即便如此也要付出代价。

一个很少被访问的状态会被学习得很慢，因为相关观测到来得慢；{ref}`olg_ex1` 把这一点具体化了。

当状态数量很大，或者状态是连续的时，每个状态一个数值的做法就无济于事了，此时主体必须转而学习一个参数化规则。

{doc}`learning_approximation` 探讨了这样做能带来什么、又要付出什么代价。

## 第二部分：恒定赤字与两个稳态

现在去掉随机性。

设政府为每个年轻人融资一笔*恒定*的赤字 $G$，效用仍如前所述为 $\ln c_1 + \ln c_2$。

在对价格水平具有完全预见的情况下，年轻人的储蓄为

```{math}
:label: saving_deterministic

s_t = \frac{w_1 - w_2 \pi_t}{2},
\qquad \pi_t \equiv \frac{p_{t+1}}{p_t},
```

其中 $\pi_t$ 是总通货膨胀率，即货币回报率的倒数。

设 $h_t = H_t/(N p_t)$ 为每个年轻人的人均实际余额。

政府以实际值表示的预算约束为 $h_t = h_{t-1}/\pi_{t-1} + G$，均衡要求 $h_t = s_t$。

在这两式之间消去 $h$，就得到一个关于通货膨胀率的自主差分方程，

```{math}
:label: inflation_map

\pi_{t+1} = g(\pi_t) \equiv A_1 - \frac{A_2}{\pi_t},
\qquad
A_1 = \frac{w_1 + w_2 - 2G}{w_2},
\qquad
A_2 = \frac{w_1}{w_2} .
```

平稳均衡就是 $g$ 的不动点，由于 $g$ 是一条双曲线，一般会有两个不动点。

我们使用 {cite:t}`MarimonSunder1993` 的"经济体7C"参数：$w_1 = 6$，$w_2 = 1$，$G = 1$。

```{code-cell} ipython3
w1_d, w2_d, G_d = 6.0, 1.0, 1.0
A1 = (w1_d + w2_d - 2*G_d) / w2_d
A2 = w1_d / w2_d

def g_map(π):
    return A1 - A2/π

π_low, π_high = np.sort(np.roots([1, -A1, A2]))
print(f"A1 = {A1},  A2 = {A2}")
print(f"stationary gross inflation rates: {π_low} and {π_high}")
print(f"   i.e. net inflation of {100*(π_low-1):.0f}% and {100*(π_high-1):.0f}% per period")
print(f"\nslope of g at the low  rate: {A2/π_low**2:.4f}   -> unstable under RE dynamics")
print(f"slope of g at the high rate: {A2/π_high**2:.4f}   -> stable under RE dynamics")
```

结果正是如此：在理性预期动态下，*低*通胀平稳均衡是**不稳定的**，而高通胀均衡是稳定的。

$G = 0$ 这一情形将该模型与 {doc}`bounded_rationality` 中的泡沫联系起来。

在没有赤字的情况下，$A_1 = 1 + w_1/w_2$，$g$ 的不动点为 $\pi = 1$ 和 $\pi = w_1/w_2$。

这两个不动点分别对应 {ref}`br_bubble` 中货币供给恒定时无泡沫的货币数量论均衡，以及该讲座中每一条泡沫路径最终都会趋近的通胀率。

两处的度量有所不同。

那篇讲座给主体赋予了 $2w_1$ 和 $2w_2$ 的禀赋，因此其实际余额需求为 $w_1 - w_2\pi$，而这里的储蓄为 $(w_1 - w_2\pi)/2$。

它还固定了货币供给的增长率 $\mu$，而这里固定的是赤字 $G$；当 $\mu = 1$ 且 $G = 0$ 时，两个模型是一致的。

因此，下面的稳定性分析回答了那篇讲座提出的问题——我们究竟处于哪一种泡沫之中：最小二乘学习者会使泡沫消亡（{ref}`olg_ex5`）。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Rational expectations inflation dynamics"
    name: fig-olg-re-dynamics
---
π_grid = np.linspace(1.35, 4.2, 400)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(π_grid, g_map(π_grid), lw=1.6, label=r"$g(\pi) = A_1 - A_2/\pi$")
ax.plot(π_grid, π_grid, 'k--', lw=0.8, label="45 degree line")

# cobweb the RE dynamics away from the low steady state
π = π_low + 0.05
for _ in range(30):
    nxt = g_map(π)
    ax.plot([π, π], [π, nxt], color='C3', lw=0.9)
    ax.plot([π, nxt], [nxt, nxt], color='C3', lw=0.9)
    π = nxt

for p, name in [(π_low, "low"), (π_high, "high")]:
    ax.plot(p, p, 'ko', ms=6)
    ax.annotate(f"{name}\n$\\pi = {p:.0f}$", (p, p), textcoords="offset points",
                xytext=(10, -22), fontsize=9)
ax.set_xlabel(r"$\pi_t$")
ax.set_ylabel(r"$\pi_{t+1}$")
ax.legend(frameon=False, loc='upper left')
plt.show()
```

在 {numref}`fig-olg-re-dynamics` 中，红色的阶梯从低通胀稳态稍上方开始，径直远离它，走向高通胀稳态。

下表列出了从四个初始通胀率出发的理性预期路径。

```{code-cell} ipython3
def re_path(π0, T=25):
    path = [π0]
    for _ in range(T):
        nxt = g_map(path[-1])
        path.append(nxt if nxt > 0 else np.nan)
    return np.array(path)

pd.DataFrame({f"$\\pi_0 = {p}$": re_path(p) for p in (1.99, 2.01, 2.5, 4.0)}
             ).rename_axis("t").round(4).iloc[[0, 1, 2, 5, 10, 25]]
```

从略低于低稳态的位置起步，经济会崩溃（由 {eq}`inflation_map` 计算出的通胀率变为负值，因此不存在正的价格水平）；从略高于低稳态的位置起步，经济则收敛到 $\pi = 3$。

这一点之所以重要，是由于两个均衡的排序关系。

低通胀均衡的比较静态分析是"古典的"：提高赤字 $G$ 会降低 $A_1$，从而提高低通胀的平稳通胀率。

高通胀均衡的比较静态分析则相反，因为该经济处于**拉弗曲线**中通货膨胀税的错误一侧；参见 {ref}`olg_ex4`。

而低通胀均衡在帕累托意义上优于该模型的所有其他均衡，无论是否为平稳均衡。

（这一排序依赖于同一世代内所有人都相同这一假设；萨金特指出，在一个世代内存在借款人和贷款人的版本中，各均衡之间并不具有帕累托可比性。）

因此，理性预期动态将经济推离帕累托最优结果，而货币理论中大多数古典教条都依赖于选取那些恰恰被这种动态所排斥的均衡。

### 最小二乘动态

{cite:t}`MarcetSargent1989hyper` 研究了这个经济体的一个版本，其中主体不具备完全预见能力，而是**做回归**。

每一期他们都将价格水平对其自身的滞后值做回归，并用拟合的斜率作为对总通胀率的预测，

$$
\beta_t = \frac{\sum_{s < t} p_s p_{s-1}}{\sum_{s < t} p_{s-1}^2},
\qquad
\pi^e_t = \beta_t ,
$$

然后按照 {eq}`saving_deterministic` 储蓄 $s_t = (w_1 - w_2 \beta_t)/2$。

由于 $\beta_t$ 只使用到 $t-1$ 期为止的数据，所以不存在同期性问题：信念决定储蓄，储蓄决定价格水平，而新的价格水平进入下一期的回归。

由于 $p_s = \pi_{s-1} p_{s-1}$，该估计量可以写成

$$
\beta_t = \frac{\sum_{s<t} \pi_{s-1} \, p_{s-1}^2}{\sum_{s<t} p_{s-1}^2},
$$

即*过去已实现通胀率*的加权平均，权重与 $p_{s-1}^2$ 成比例。

这些权重远非相等。

将其写成递归形式，

$$
\beta_{t+1} = \beta_t + \gamma_t \, (\pi_{t-1} - \beta_t),
\qquad
\gamma_t = \frac{p_{t-1}^2}{\sum_{s \le t} p_{s-1}^2},
$$

可以看出最小二乘法就是增益为 $\gamma_t$ 的适应性预期，其中 $\gamma_t$ 就是最新观测值在权重中所占的份额。

在通货膨胀期间，价格水平呈几何增长，因此该份额并不会缩小：当 $p$ 以总速率 $\pi$ 增长时，$\gamma_t$ 趋于 $1 - 1/\pi^2$，在 $\pi = 2$ 处该值为 $0.75$。

所以按水平值做最小二乘法并不是对一段长历史求平均；在通货膨胀期间，它表现得就像一个增益很大的适应性预期。

同样的形式也使得该递归易于计算。

价格水平呈几何增长，因此这两个求和很快就会溢出，但如果我们在每一步都用最新的权重对两者进行重新标度，那么一切都保持在同一数量级上。

代码将初始信念 $\pi_0$ 当作单个伪观测值来处理。

```{code-cell} ipython3
def ls_dynamics(π_init, T=3000, G=G_d):
    """
    Least squares dynamics.  Carries the two regression sums rescaled by the
    newest p^2, so nothing overflows even though the price level does not stop
    growing.  Returns (beliefs, realized inflation, survived?).
    """
    s_prev = (w1_d - w2_d*π_init) / 2
    num, den, π_last = π_init, 1.0, π_init
    betas, infl = [], []

    for t in range(T):
        β = num / den
        s_t = (w1_d - w2_d*β) / 2
        if s_t <= G or not np.isfinite(s_t):
            return np.array(betas), np.array(infl), False   # no positive price level
        π_t = s_prev / (s_t - G)                            # realized gross inflation
        num = num/π_last**2 + π_t
        den = den/π_last**2 + 1.0
        π_last, s_prev = π_t, s_t
        betas.append(β)
        infl.append(π_t)

    return np.array(betas), np.array(infl), True


rows = []
for π0 in (0.5, 1.2, 1.5, 1.9, 2.1, 2.5, 2.9, 2.99, 3.01):
    b, infl, ok = ls_dynamics(π0)
    if ok:
        settled = np.argmax(np.abs(b - π_low) < 1e-3)
        rows.append([π0, "yes", b[-1], settled, np.nan])
    else:
        rows.append([π0, "no", np.nan, np.nan, len(b)])

pd.DataFrame(rows, columns=["initial belief $\\pi_0$", "survived", "belief $\\beta_T$",
                            "periods to within 0.001 of $\\pi_{low}$",
                            "period of collapse"]).round(5)
```

(olg_reversal)=
### 稳定性的反转

在表中低于 $3$ 的每一个初始信念下，最小二乘学习都会在20期以内收敛到**低**通胀均衡，即便从略低于 $3$ 的信念出发也不例外——而 $\pi = 3$ 正是理性预期动态所收敛*向*的稳态。

而一个略高于 $3$ 的信念则会终结这个货币经济：预期通胀不断上升，直到储蓄不足以覆盖赤字。

{numref}`fig-olg-ls-vs-re` 将最小二乘路径与理性预期路径并列展示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Least squares and rational expectations inflation paths"
    name: fig-olg-ls-vs-re
---
fig, ax = plt.subplots(figsize=(8, 4.5))

for π0 in (1.5, 2.5, 2.95):
    _, infl, _ = ls_dynamics(π0, T=60)
    ax.plot(infl, color='C0', lw=1.2,
            label="least squares dynamics" if π0 == 1.5 else None)

for π0 in (2.05, 2.5, 4.0):
    ax.plot(re_path(π0, 60), color='C3', lw=1.2, ls='--',
            label="rational expectations dynamics" if π0 == 2.05 else None)

ax.axhline(π_low, color='k', lw=0.8, ls=':')
ax.axhline(π_high, color='k', lw=0.8, ls=':')
ax.annotate("low", (52, π_low + 0.04), fontsize=9)
ax.annotate("high", (52, π_high + 0.04), fontsize=9)
ax.set_xlim(0, 60)
ax.set_ylim(1.2, 4.2)
ax.set_xlabel("$t$")
ax.set_ylabel("gross inflation")
ax.legend(frameon=False)
plt.show()
```

这两族路径朝着相反的方向运行。

二分查找定位了最小二乘吸引域的边界。

```{code-cell} ipython3
lo, hi = 2.5, 3.7
for _ in range(40):                      # bisect for the edge of the basin
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if ls_dynamics(mid)[2] else (lo, mid)
print(f"least squares dynamics survive for initial beliefs up to {lo:.6f}")
print(f"the rational-expectations-stable steady state is      {π_high}")
```

该吸引域的边界正是 $\pi_{\text{high}}$ 本身。

恰好在 $\pi_0 = 3$ 处，信念保持不动：$\pi_{\text{high}}$ 是最小二乘动态的一个静止点，但它是不稳定的。

我们从未真正从恰好 $3$ 出发，因为在浮点运算中，量级为 $10^{-16}$ 的舍入误差最终会不断放大，此后经济要么跌落到 $\pi_{\text{low}}$，要么崩溃，具体取决于误差的符号。

因此，理性预期动态选定为*那个*稳定稳态的均衡，恰好正是学习动态朝向另一个均衡的吸引域边界。

在这里，初始信念具有永久性的影响，因为它决定了这个货币经济能否存续。

#### 为什么会反转？E-稳定性

固定一个信念 $\beta$，假设主体永远持有这一信念。

那么储蓄将恒定为 $s(\beta) = (w_1 - w_2\beta)/2$，预算约束 $h_t = h_{t-1}/\pi_{t-1} + G$（其中 $h_t = s(\beta)$）将带来恒定的实际通胀率

```{math}
:label: olg_tmap

T(\beta) = \frac{s(\beta)}{s(\beta) - G} .
```

$T$ 将预期通胀映射为实际通胀；它是本模型中 {doc}`bounded_rationality` 所用的 $T$-映射的对应版本，其不动点即为平稳均衡。

最小二乘法使信念朝其所产生的通胀方向移动，因此当增益很小时，信念的平均运动遵循如下常微分方程

$$
\frac{d\beta}{d\tau} = T(\beta) - \beta .
$$

如果 $T'(\beta^*) < 1$，则该方程下的不动点 $\beta^*$ 是稳定的，即所谓的**E-稳定**（参见 {ref}`br_one_recursion`，以及一般理论参见 {doc}`ls_learning`）。

在经济体7C的参数下，$T(\beta) = (6 - \beta)/(4 - \beta)$，因此 $T'(\beta) = 2/(4-\beta)^2$。

```{code-cell} ipython3
def T_map(β, G=G_d):
    "Actual gross inflation when the belief β is held forever."
    s = (w1_d - w2_d*β) / 2
    return s / (s - G)

def slope(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2*h)

for π_star, name in [(π_low, "low "), (π_high, "high")]:
    print(f"{name} steady state π = {π_star:.0f}:   T'(π) = {slope(T_map, π_star):.3f},"
          f"   g'(π) = {slope(g_map, π_star):.3f}")
```

所以 $T'(2) = 1/2$，$T'(3) = 2$：低稳态是E-稳定的，而高稳态不是。

每一个斜率都位于 $g$ 对应斜率的另一侧。

在完全预见下，一个略高于 $\pi_{\text{low}}$ 的信念会被*验证并放大*，因为 $g$ 在该处的斜率为 $1.5$。

而在学习动态下，同样的信念会降低储蓄，从而提高实际通胀率，但幅度只有一半，因为 $T$ 的斜率为 $1/2$；实现的通胀率会*低于*信念本身，于是信念被向下修正。

在 $\pi_{\text{high}}$ 处情况正好相反。

这并非经济体7C参数所特有的巧合。

在任何储蓄为正的稳态下，都有
$\operatorname{sign}(T' - 1) = -\operatorname{sign}(g' - 1)$，因此只要存在两个稳态，较低的那个在理性预期下不稳定却是E-稳定的，较高的那个则正好相反；{ref}`olg_ex6` 要求你证明这一点。

这种反转是关于该模型的一个定理。

还有第二种方式可以看出这两种动态为何相互矛盾。

在完全预见下，过去的历史并不能确定初始通胀率，因此理性预期下稳定的高稳态会被一个均衡路径的连续统所逼近，而在不稳定的低稳态附近，唯一的均衡路径就是平稳路径本身。

学习动态则选出了局部唯一的那个稳态。

还有一点需要说明。

上述微分方程描述的是当每个观测值只带来微小调整时信念的平均运动，而我们在上面已经看到，这里的最小二乘法给予最新观测值的权重接近 $1 - 1/\pi^2$，在 $\pi_{\text{low}}$ 处为四分之三。

因此我们直接检验这一结论。

在稳态 $\pi^*$ 附近，最小二乘法的表现类似于适应性预期
$\beta_{t+1} = \beta_t + \gamma(\pi_{t-1} - \beta_t)$，其中 $\gamma = 1 - 1/\pi^{*2}$。

实现的通胀率 $\pi_{t-1} = s(\beta_{t-1})/(s(\beta_t) - G)$ 依赖于两个相继的信念，因此这是一个关于 $\beta$ 的二阶差分方程。

代码在每个稳态处对其线性化，并报告其按模最大的根——既针对最小二乘增益，也针对0到1之间的所有增益。

```{code-cell} ipython3
def local_modulus(π_star, γ):
    "Largest root, in modulus, of the adaptive-expectations recursion linearized at π_star."
    s = (w1_d - w2_d*π_star) / 2
    a = -(w2_d/2) / (s - G_d)                   # ∂π_{t-1} / ∂β_{t-1}
    b = (w2_d/2) * s / (s - G_d)**2             # ∂π_{t-1} / ∂β_t
    return np.max(np.abs(np.roots([1, -(1 - γ + γ*b), -γ*a])))

γ_grid = np.linspace(0.01, 1, 100)
for π_star, name in [(π_low, "low "), (π_high, "high")]:
    γ_ls = 1 - 1/π_star**2
    moduli = [local_modulus(π_star, γ) for γ in γ_grid]
    print(f"{name}: least squares gain {γ_ls:.3f}, largest root {local_modulus(π_star, γ_ls):.3f};"
          f"  over gains in (0, 1]: from {min(moduli):.3f} to {max(moduli):.3f}")
```

在 $\pi_{\text{low}}$ 处，无论增益为何值，每个根都位于单位圆内；在 $\pi_{\text{high}}$ 处，无论增益为何值，总有根位于单位圆外。

E-稳定性的结论在大增益下依然成立。

这也解释了为什么 {cite:t}`BrunoFischer1990` 在一个密切相关的模型中，使用弗里德曼式的适应性预期而非最小二乘法，也发现了同样的反转。

不过，萨金特援引 {cite:t}`BullardDuffy1993` 作为不要对这一结果过度推广的警示。

在主体存活 $n$ 期的世代交叠经济体中，他们发现当 $n \geq 4$ 时，最小二乘学习在局部并不收敛到理性预期均衡。

### 马里蒙与桑德的实验

哪一组动态更能描述人的行为？

{cite:t}`MarimonSunder1993` 构建了一个实验室经济体来寻找答案。

固定数量的付费被试被指派为这个模型精确实现中的"年轻人"和"老年人"角色。

每一期，年轻人提交一份供给表，表明他们愿意以何种价格向老年人提供多少产出以换取货币，实验者据此让市场对老年人和政府的需求出清。

暂时闲置的被试参加一个**预测游戏**，该游戏本身也设有现金奖励，因此该实验直接记录了驱动经济运行的价格预期。

会话的长度对被试而言是未知的，但受公开宣布的规则支配，从被试的角度来看，这使得该经济等价于一个永不结束的经济。

这里有两个重要发现。

第一，实验中的通胀路径被收敛到**低**平稳通胀率的最小二乘动态所近似的程度，要远远好于被收敛到高通胀率的理性预期动态所近似的程度。

萨金特报告称，这一模式体现了他们所有结果的共同特征：*平均而言*，实验中的动态被适应性动态所近似的程度，远远好于被理性预期动态所近似的程度。

第二，闲置被试的预测误差，其大小与使用同样数据的最小二乘预测者所会犯的误差相当。

被试们所做的并非什么怪异之事，他们所做的大致就是适应性模型所说的那样。

理论认为不稳定的那个均衡，恰恰是算法和人类都选择了的均衡。

```{note}
实验证据并非甄别这一问题的唯一途径。

{cite:t}`Imrohoroglu1993` 用德国恶性通胀数据估计了该模型的一个理性预期版本，得到了相反的结果：他所估计出的均衡沿着拉弗曲线的*坏的*一侧滑向高平稳通胀率，这与适应性动态所选出的均衡不一致。
```

### 一个警示

人们容易得出这样的结论：适应性动态总能可靠地选出帕累托占优的均衡。

萨金特提供了一个反例，这是本杰明·本塔尔向他展示的，其基础是 {cite:t}`Brock1974` 货币进入效用函数模型的一个特殊情形。

一个拥有禀赋 $y$ 的无限期存活的家庭最大化
$\sum_t \delta^t (\ln c_t + \phi \ln(m_t/p_t))$，须服从一系列预算约束，而政府通过印制货币为固定支出 $G > 0$ 融资。

（萨金特用 $\beta$ 和 $\gamma$ 分别表示贴现因子 $\delta$ 和权重 $\phi$；本文保留 $\beta$ 用于表示信念，$\gamma$ 用于表示增益。）

在参数限制条件 $\phi(y - G) > G$ 下，该模型具有唯一的平稳均衡，其总通胀率为

$$
\pi = \frac{(y-G)\phi - \delta G}{(y-G)\phi - G},
$$

它具有随赤字上升而上升的古典性质。

```{code-cell} ipython3
φ_b, δ_b, y_b = 1.0, 1/1.05, 11.0

def brock_π(G):
    return ((y_b - G)*φ_b - δ_b*G) / ((y_b - G)*φ_b - G)

G_vals = (0.5, 1.0, 1.5, 2.0)
pd.DataFrame({
    "$G$": G_vals,
    "restriction $\\phi(y-G) > G$": [φ_b*(y_b - G) > G for G in G_vals],
    "stationary $\\pi$": [brock_π(G) for G in G_vals],
    "return on currency $1/\\pi$": [1/brock_π(G) for G in G_vals],
}).round(5)
```

但该模型*同时*还存在一个由初始价格水平索引的非平稳均衡连续统，这些初始价格水平低于该平稳均衡的价格水平，在所有这些均衡中，总通胀率都收敛到 $\delta$——也就是说，经济最终陷入**通货紧缩**，货币回报率趋近于 $1/\delta = 1.05$，实际余额则不断膨胀。

这些均衡之所以存在，是因为由 $\phi \ln(m/p)$ 所隐含的货币需求对回报率的弹性极高，以至于即使政府通过通货紧缩向货币支付利息，也仍能筹集到足够的铸币税来为 $G$ 融资。

萨金特报告称，在这里应用最小二乘学习会选出*古典的平稳*均衡，正如在世代交叠模型中那样。

借助 $T$-映射很容易验证这一点。

由于消费恒定为 $y - G$，家庭关于货币的一阶条件表明，当预期通胀为 $\pi^e$ 时，实际余额为 $h(\pi^e) = \phi (y - G) \pi^e / (\pi^e - \delta)$。

预算约束 $h = h/\pi + G$ 由此将一个被永久持有的信念转化为实际通胀率
$T(\pi^e) = h(\pi^e)/(h(\pi^e) - G)$。

```{code-cell} ipython3
def brock_T(πe, G):
    "Actual gross inflation when expected inflation πe is held forever."
    h = φ_b * (y_b - G) * πe / (πe - δ_b)
    return h / (h - G)

G_b = 1.0
print(f"stationary π = {brock_π(G_b):.5f},   T'(π) = {slope(lambda x: brock_T(x, G_b), brock_π(G_b)):.3f}")
```

该斜率远低于1，所以古典平稳均衡是E-稳定的。

与世代交叠模型的区别在于福利排序：在布罗克模型中，所有非平稳均衡都在帕累托意义上优于学习所选出的古典平稳均衡。

因此在这两个例子中，最小二乘动态都选出了*古典*均衡。

但这个均衡是否为好的均衡，则是另一个问题，其答案取决于具体模型。

(olg_phillips)=
## 一个学习菲利普斯曲线的政府

到目前为止的例子都是把适应性*家庭*放入一个货币经济中。

我们以一个应用作为收尾，其中适应性主体是**政府**，它学习一个它据以采取行动的计量经济学关系。

该模型源自 {cite:t}`Sims1988`，并由 {cite:t}`Chung1990` 加以估计，它扩展了 {cite:t}`KydlandPrescott1977` 的菲利普斯曲线例子。

它容纳了关于战后美国通货膨胀的两种叙事：一个被看似可利用的菲利普斯曲线误导的政府，以及一些菲利普斯曲线拟合者，他们的适应性方法及时察觉到了不利的移动。

从 {doc}`phillips_two_stories` 开始的菲利普斯曲线相关讲座，使用相同的校准对这一内容进行了深入阐述；这里给出一个简化版本。

### 模型

通货膨胀被分解为预期部分和非预期部分，

```{math}
:label: phillips_inflation

\pi_t = g_{t-1} + \eta_t,
```

其中 $g_{t-1}$ 是公众预期的通胀（也是政府所控制的），$\eta_t$ 是公众的预测误差，与更早的信息正交。

**真实的**菲利普斯曲线是自然率曲线：只有**意外**部分 $\eta_t$ 才会影响失业率：

```{math}
:label: phillips_true

U_t = U^* - \theta(\pi_t - g_{t-1}) + u_t = U^* - \theta \eta_t + u_t,
\qquad \theta > 0.
```

政府并不知道这一点。

它相信失业取决于通胀的*水平*，因此拟合了以下非预期性回归

```{math}
:label: phillips_perceived

U_t = \alpha_{0} + \alpha_{1}\pi_t + \epsilon_t.
```

每一期它都在其所认知的模型约束下最小化 $\tfrac12 \mathbb{E}(U_t^2 + \pi_t^2)$，这产生了如下短视目标

```{math}
:label: phillips_rule

g_{t-1} = -\frac{\alpha_{0}\,\alpha_{1}}{1 + \alpha_{1}^2}.
```

该系统是**自我参照的**：政府的信念 $\alpha$ 通过 {eq}`phillips_rule` 决定其政策 $g$；该政策塑造了 $(\pi, U)$ 的联合分布；而这一分布正是政府的回归 {eq}`phillips_perceived` 所估计的对象。

```{note}
我们沿用 {cite:t}`Sargent1993` 的记号，其中 $\pi_t$ 是以百分比表示的*净*通货膨胀率。

下表给出了与 {doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 之间的对应关系；{ref}`phillips_notation` 涵盖了菲利普斯曲线相关讲座的其余部分。

| 对象 | 本讲座中 | {doc}`phillips_escaping_nash`、{doc}`phillips_priors` |
|---|---|---|
| 通货膨胀 | $\pi_t$ | $\pi_n$ |
| 政府的目标（系统性）通胀 | $g_{t-1}$ | $x_n$ |
| 自然失业率 | $U^*$ | $u$ |
| 通胀意外 | $\eta_t$ | $\sigma_2 W_{2n}$ |
| 失业冲击 | $u_t$ | $\sigma_1 W_{1n}$ |
| 信念：截距、斜率 | $\alpha_0$、$\alpha_1$ | $\gamma_{-1}$、$\gamma_1$（escaping_nash）；$a$、$b$（priors） |
| 恒定增益 | $\gamma$ | $\varepsilon$ |
| 存在通胀偏差的均衡 | 自我确认（基德兰-普雷斯科特的“一致性”） | 自我确认（纳什） |
```

```{code-cell} ipython3
U_star, θ_pc = 5.0, 1.0             # natural rate 5%, Phillips slope
σ_η, σ_u = 0.3, 0.3                 # inflation-surprise and unemployment shocks

def govt_target(α):
    "Government's myopic optimal target inflation."
    α0, α1 = α
    return -α0 * α1 / (1 + α1**2)
```

### 自我确认均衡

基德兰和普雷斯科特的**一致性**均衡，正是菲利普斯曲线相关讲座所称的**自我确认**均衡：一个信念向量 $\alpha$，其所引出的政策所产生的数据经过回归后，会返回同样的 $\alpha$。

在恒定政策 $g$ 下，意外部分 $\eta_t$ 是唯一使通胀围绕其均值波动的因素，因此对 $U$ 对 $\pi$ 的总体回归，其斜率恰为 $-\theta$，截距为 $U^* + \theta g$。

将该不动点与决策规则 {eq}`phillips_rule` 联立求解，就得到基德兰和普雷斯科特的时间一致性结果，

$$
\alpha_0 = (1 + \theta^2) U^*, \qquad \alpha_1 = -\theta,
\qquad g = \theta U^* .
$$

```{code-cell} ipython3
α_consistent = np.array([(1 + θ_pc**2) * U_star, -θ_pc])
g_consistent = θ_pc * U_star
print(f"self-confirming beliefs α = {α_consistent},  target inflation g = {g_consistent}")
print(f"optimal (Ramsey) outcome:                     target inflation g = 0")
```

这就是**通胀偏差**。

政府以 $\theta U^*$ 的速率制造通胀，却一无所获：由于只有意外冲击才能影响失业率，无论怎样，平均失业率都是 $U^*$。

假如政府理解了真实模型 {eq}`phillips_true`，它本应选择 $g = 0$，即拉姆齐结果：同样的失业率，却没有任何通货膨胀。

自我确认均衡是唯一的，而拉姆齐结果并不是一个均衡：若要选择 $g = 0$，政府必须相信不存在任何权衡关系，但 $g = 0$ 所产生的数据却显示出斜率 $-\theta$。

自我确认均衡是 E-稳定的，因此一个相信自己系数恒定、并以递减增益用最小二乘法估计它们的政府，会被吸引到这一均衡（{doc}`phillips_self_confirming`、{doc}`phillips_learning`）。

这正是自然率的叙事：政府持续制造通胀，是因为它自己的政策不断产生的数据，恰恰确认了它所相信的那种权衡关系。

### 恒定增益学习与逃逸

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 让政府对自己的系数是否恒定*不太*确定：它怀疑系数会以随机游走的方式漂移，并使用卡尔曼滤波器（参见 {doc}`kalman`）来估计它们，该滤波器会对旧数据打折，更看重近期数据。

西姆斯发现存在两类历史路径。

有些路径会停留在自我确认均衡附近；而另一些路径则收敛到一种过程，该过程的大部分时间都停留在零通胀的拉姆齐结果附近，这是因为一个愿意考虑系数可能漂移的政府，无需经历一场大规模通胀，就能学到那种权衡关系其实很糟糕。

究竟会出现哪一类历史路径，部分取决于参数，尤其是政府所考虑的漂移方差。

我们这里的政府以一种更简单的方式进行学习，即**恒定增益递归最小二乘法**，这正是 {cite:t}`Sargent1999` 和 {cite:t}`ChoWilliamsSargent2002` 所研究的版本，它以关于漂移的某种特定先验来近似卡尔曼滤波器 {cite}`SargentWilliams2005`。

设 $x_t = (1, \pi_t)^\top$，并给定恒定增益 $\gamma$，政府利用其估计值 $\alpha_t$，依据 {eq}`phillips_rule` 设定 $g_{t-1}$，观测到 $(\pi_t, U_t)$，然后更新

$$
M_t = M_{t-1} + \gamma \bigl(x_t x_t^\top - M_{t-1}\bigr),
\qquad
\alpha_{t+1} = \alpha_t + \gamma M_t^{-1} x_t \bigl(U_t - x_t^\top \alpha_t\bigr).
$$

与最终会使估计值冻结的 $1/t$ 增益不同，恒定增益使政府始终持续关注刚刚发生的事情。

我们以自我确认信念作为初始值，设定 $M_0 = I$，$\gamma = 0.02$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Constant-gain learning and recurrent escapes toward zero inflation"
    name: fig-olg-escapes
---
def constant_gain(T=40_000, gain=0.02, seed=1):
    """
    Government estimates its Phillips curve by constant-gain recursive least
    squares, starting from the self-confirming beliefs.  Returns the paths of
    target inflation g_{t-1}, inflation π_t and unemployment U_t.
    """
    rng = np.random.default_rng(seed)
    α = α_consistent.copy()
    M = np.eye(2)                                          # moment-matrix estimate
    g_path, π_path, U_path = np.empty(T), np.empty(T), np.empty(T)
    for t in range(T):
        g = govt_target(α)
        η = σ_η * rng.standard_normal()
        π = g + η
        U = U_star - θ_pc * η + σ_u * rng.standard_normal()   # true Phillips curve
        x = np.array([1.0, π])
        M = M + gain * (np.outer(x, x) - M)
        α = α + gain * np.linalg.solve(M, x * (U - x @ α))
        g_path[t], π_path[t], U_path[t] = g, π, U
    return g_path, π_path, U_path

g_path, _, _ = constant_gain()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(g_path, lw=0.5)
ax.axhline(g_consistent, color='C3', ls='--', lw=1, label="self-confirming equilibrium (inflation bias)")
ax.axhline(0, color='C2', ls=':', lw=1, label="Ramsey / zero inflation")
ax.set_ylim(-1, 6.8)
ax.set_xlabel("$t$")
ax.set_ylabel("target inflation $g_t$")
ax.legend(frameon=False, loc='upper center', ncol=2)
plt.show()
```

{numref}`fig-olg-escapes` 显示，采用恒定增益的政府不会稳定在通胀偏差处。

它反复地、突然地**逃逸**至零通胀，随后又攀升回通胀偏差附近，而在这一增益水平下，它通常在到达通胀偏差之前就再次逃逸：呈现出一种循环往复的锯齿状规律。

这些逃逸会自我强化。

一连串使失业率看起来对通胀无反应的冲击，会使所估计的斜率 $\alpha_1$ 趋于平缓，政府随之削减通胀。

这种可预见的通胀下降并不会推高失业率，因为只有意外冲击才会影响失业率，于是政府自身的政策产生的数据恰恰确认了这种更平缓的斜率，并把通胀进一步推向零。

这种反复的逃逸与回归，是 {cite:t}`Sargent1999` 和 {cite:t}`ChoWilliamsSargent2002` 的动态，而非西姆斯的动态——后者那些出现逃逸的历史路径停留在拉姆齐结果附近；{doc}`phillips_priors` 将这一差异追溯到政府设定其卡尔曼滤波器的方式。

{doc}`phillips_learning` 和 {doc}`phillips_escaping_nash` 展示了逃逸发生的频率如何取决于增益，以及 {cite:t}`Sargent1999` 如何将其转化为一种关于美国通货膨胀在20世纪80年代初是如何被征服的理论。

钟则更进一步，直接在战后美国数据上估计了政府的学习规则本身；{ref}`olg_ex7` 要求你求解他所研究问题的一个简化版本。

## 结语

第一部分表明，一个由适应性主体构成的系统能够找到一个它从未被告知过的理性预期均衡。

它还表明，它*找到哪一个*均衡并不取决于主体：经济体 2 存在两个平稳均衡，而学习只能找到低通胀的那一个。

并且它清楚地展现了环境的复杂程度对学习速度的支配作用：一个很少出现的赤字状态被学习得很慢。

第二部分使选择结果更加鲜明。

在模型存在两个稳态的情况下，最小二乘学习挑出了那个被理性预期动态所排斥的稳态，而实验室被试的选择也是一样的。

E-稳定性解释了其中的原因：在每一个具有正储蓄的稳态处，$T$ 映射与理性预期映射的斜率分别位于 1 的两侧。

同样的逻辑也化解了 {doc}`bounded_rationality`（{ref}`olg_ex5`）中的泡沫问题。

这是 {cite:t}`Sargent1993` 中支持把适应性动态视为一种均衡选择机制的最有力例证。

布罗克反例标出了这一论断的界限。

在这两个模型中，学习都选中了古典均衡，而这一均衡在一个模型中是好的结果，在另一个模型中却是坏的结果。

这种选择是关于动态本身的一个事实，而非福利定理，{doc}`prospects_bounded_rationality` 将回到这一结果究竟能承受多大分量的问题。

菲利普斯曲线的应用使*增益*成为了关注的对象。

在递减增益下，政府会稳定在通胀偏误处；而在恒定增益下，它则不断逃向拉姆齐结果——而这根本不是一个均衡。

从 {doc}`phillips_two_stories` 开始的菲利普斯曲线系列讲座深入探讨了这一问题。

{doc}`learning_approximation` 探讨了第一部分遗留的问题：当状态是连续的时候，主体必须放弃什么。

{doc}`exchange_rate_learning` 随后回到这一存在两种货币的世代交叠经济，其中理性预期均衡构成一个连续统，而学习的极限则取决于信念的起点。

再往后，{doc}`marimon_mcgrattan_sargent` 把选择这一问题带入了一个场景，在那里竞争的均衡不是高通胀与低通胀之别，而是不同*货币制度*之别：即哪种商品会成为货币。

## 练习

```{exercise-start}
:label: olg_ex1
```

由于主体针对每个赤字状态学习一个独立的储蓄率，他们只能以该状态出现的速度来学习该状态。

比较两个仅在转移矩阵上有所不同、其余均为 $\bar G = (0.8, 0)$ 的经济体：一个是两个状态等可能，另一个是高赤字状态很少出现。

对每种情形，计算平稳均衡，运行30000期的学习模拟，并报告每个状态所学到的储蓄率与其目标值相差多远，同时报告该状态被访问的频率。

单次运行可能会产生误导，因此请在20个不同的随机种子上重复该比较。

```{exercise-end}
```

```{solution-start} olg_ex1
:class: dropdown
```

对每条转移链，我们在每次运行中记录每个状态所学规则的误差（在两个类别上取平均），以及在每个状态上所花的期数。

```{code-cell} ipython3
G_ex = np.array([0.8, 0.0])
cases = {"equally likely": np.array([[0.5, 0.5], [0.5, 0.5]]),
         "high deficit rare": np.array([[0.5, 0.5], [0.03, 0.97]])}
seeds = range(20)

rows, ratios = [], {}
for name, P in cases.items():
    s_star, _ = stationary_equilibrium(G_ex, P)
    errors, visits = [], []
    for seed in seeds:
        hist, states = simulate(G_ex, P, T=30_000, s0=start, seed=seed)
        errors.append(np.abs(hist[-1].mean(axis=0) - s_star))
        visits.append(np.bincount(states, minlength=2))
    errors, visits = np.array(errors), np.array(visits)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    ratios[name] = errors[:, 0] / errors[:, 1]
    for k in (0, 1):
        rows.append([name, k+1, visits[:, k].mean(), s_star[k], rmse[k]])

display(pd.DataFrame(rows, columns=["chain", "state", "visits (mean over seeds)",
                                    "equilibrium $s_i$", "RMS error over seeds"]
                     ).set_index(["chain", "state"]).round(4))

for name, r in ratios.items():
    print(f"{name:18s}: error ratio state 1 / state 2 across seeds: "
          f"median {np.median(r):.1f}, range {r.min():.1f} to {r.max():.1f}, "
          f"state 1 worse in {np.sum(r > 1)} of {len(r)} seeds")
```

当两个状态等可能出现时，该链是独立同分布的，因此两个状态面对相同的未来回报分布，均衡中有 $s_1 = s_2$。

每个状态大约被访问一半的时间，两条规则被学习得同样好。

当高赤字状态很少出现时，它仅在约百分之六的时间里被访问到，在30000期中大约只有1700期，其规则的均方根误差是常见状态规则的好几倍，尽管两者都已经历了30000个日历期的时间来稳定下来。

在大多数运行中该比值都很大，但在不同的随机种子之间差异颇大，这正是为什么单次运行不足以支撑这种比较。

萨金特的评论值得牢记：从*无条件预期效用*的角度来看，未能学会在一个很少被访问的状态下该怎么做，可能付出的代价非常小。

学习缓慢所付出的代价并不与误差的大小成正比。

还要注意的是，改变转移矩阵也会改变均衡本身——关于赤字持续性的信念会直接反馈进 {eq}`olg_equilibrium`——因此这种比较必须针对每条链分别重新计算目标值。

```{solution-end}
```

```{exercise-start}
:label: olg_ex2
```

在一个随机环境中，衰减增益 $\gamma_\tau = 1/(\tau + 20)$ 正是使 {eq}`robbins_monro` 收敛的关键所在。

一个怀疑环境可能存在漂移的主体，会转而使用**恒定增益**，永远更看重近期经验。

利用 `simulate` 的 `gain` 参数，为主体赋予恒定增益，并描述储蓄规则的极限行为会发生什么变化。

将增益为 $0.02$ 和 $0.005$ 的情形，在经济体1和经济体2中，分别与衰减增益基准进行比较。

```{exercise-end}
```

```{solution-start} olg_ex2
:class: dropdown
```

对每个经济体与每种增益，我们对规则最后5000期的值取平均（在两个类别上取平均），并报告相对均衡的平均误差及其标准差。

```{code-cell} ipython3
rows, paths = [], {}
for G, P, s_star, econ in [(G1, P1, s1, "economy 1"), (G2, P2, s2, "economy 2")]:
    for label, gain in [("0.02", 0.02), ("0.005", 0.005), ("1/(τ+20)", None)]:
        hist, _ = simulate(G, P, T=30_000, s0=start, gain=gain)
        paths[econ, label] = hist[:, 0, 0]
        tail = hist[-5_000:].mean(axis=1)        # average the two classes
        rows.append([econ, label,
                     *[f"{e:.1e}" for e in tail.mean(axis=0) - s_star],
                     *[f"{d:.1e}" for d in tail.std(axis=0)]])

pd.DataFrame(rows, columns=["economy", "gain", "mean error $s_1$", "mean error $s_2$",
                            "s.d. $s_1$", "s.d. $s_2$"]).set_index(["economy", "gain"])
```

下图展示了经济体2中偶数类别关于状态1的规则。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
for label, colour, lw in [("0.02", 'C1', 0.7), ("0.005", 'C2', 0.7), ("1/(τ+20)", 'C0', 1.2)]:
    ax.plot(paths["economy 2", label], color=colour, lw=lw, label=f"gain = {label}")
ax.axhline(s2[0], color='k', lw=0.8, ls='--', label="equilibrium")
ax.set_xlim(0, 30_000)
ax.set_ylim(3.8, 5.0)
ax.set_xlabel("$t$")
ax.set_ylabel("saving rate, state 1")
ax.legend(frameon=False, fontsize=9)
plt.show()
```

在经济体2中，恒定增益规则并不收敛。

它们会稳定在均衡附近的一个*邻域*内，然后在其中永远来回抖动，这是因为每一个新的观测值总是被赋予相同的权重，因此估计值永远不会停止移动。

减小增益能收窄这个带宽，但无法消除它。

相比之下，在经济体1中，恒定增益规则会精确收敛到均衡（达到机器精度），而且比衰减增益规则收敛得更快。

在该均衡处没有任何随机冲击作用于主体：由于没有赤字，每个状态下都有 $R \equiv 1$，因此一旦规则正确，每一次观测都会确认它。

所以正确的说法是有条件的。

当均衡处存在内生冲击时，只有一个衰减到零的增益才能带来向单一点的收敛，这也正是为什么这些系统的收敛定理需要这一点。

而在没有这类冲击时，恒定增益也会收敛，而且很快。

该表也揭示了衰减增益在经济体2中的代价。

经过30000期之后，$1/(\tau + 20)$ 规则在两个状态下仍然比均衡值高出约 $0.01$，而由于其标准差要小好几倍，这个误差属于偏差：该规则尚未到达目标。

相比之下，恒定增益规则在均衡附近游荡，标准差约为 $0.02$ 到 $0.04$，速度之慢，即使对5000期取平均，也可能与均衡相差几个百分点。

恒定增益主体以永久性噪声为代价所换来的，是速度和适应性：如果赤字过程发生变化，衰减增益主体几乎不会察觉，而恒定增益主体则会跟踪这一变化。

这种权衡在这整个文献中反复出现，在上文菲利普斯曲线的应用中也是如此。

```{solution-end}
```

```{exercise-start}
:label: olg_ex3
```

经济体2存在第二个高通胀平稳均衡 `s2_hi`，已在上文计算得出。

在 {numref}`fig-olg-saving` 的模拟中，投影机制将状态1的储蓄保持在 $\bar G_1 + 0.4 = 1.2$ 之上，这一点已经排除了该均衡，因为其 $s_1$ 低于 $1.2$。

1. 将投影带宽降低到 $0.01$，让两个类别都从高通胀均衡出发，并在20个不同的随机种子下运行该学习经济20000期。
   储蓄规则会走向何方？投影机制又在多大程度上发挥作用？
1. 用相关的常微分方程来解释这一结果。
   当两个类别都使用规则 $s$ 时，状态 $i$ 下的平均牛顿步长为
   $-E_i U'(s_i) / E_i U''(s_i)$，其中 $E_i$ 表示对下一期状态取平均，回报为
   $R(i, j) = (N s_j - \bar G_j)/(N s_i)$。
   计算此映射在两个均衡处的雅可比矩阵的特征值。

```{exercise-end}
```

```{solution-start} olg_ex3
:class: dropdown
```

我们让每次运行都恰好从高通胀均衡出发，并记录规则（在两个类别上取平均）最终落于何处，以及状态1的规则是否曾触及下限 $\bar G_1 + 0.01$。

```{code-cell} ipython3
band = 0.01
floor_1 = G2[0]/N + band
final, hit_floor = [], []
for seed in range(20):
    hist, _ = simulate(G2, P2, T=20_000, s0=np.tile(s2_hi, (2, 1)), seed=seed, band=band)
    final.append(hist[-1].mean(axis=0))
    hit_floor.append(np.any(np.isclose(hist[:, :, 0], floor_1)))
final, hit_floor = np.array(final), np.array(hit_floor)
climbed = final[:, 0] > s2_hi[0] + 1

print(f"climbed toward the low-inflation equilibrium:  {climbed.sum()} of 20 runs,"
      f"  mean final rule {np.round(final[climbed].mean(axis=0), 3)}")
print(f"fell to the projection floor:                  {(~climbed).sum()} of 20 runs,"
      f"  mean final rule {np.round(final[~climbed].mean(axis=0), 3)}")
print(f"hit the floor at least once:                   {hit_floor.sum()} of 20 runs")
print(f"low-inflation equilibrium:                     {np.round(s2, 3)}")
```

学习永远不会停留在高通胀均衡处，即使它恰好从那里开始也是如此。

大多数运行都朝着低通胀均衡攀升，在20000期中走完了大部分距离，而且仍在继续攀升。

其余的运行则走向另一个方向：状态1的规则跌至下限，此时的储蓄几乎仅能覆盖赤字，而状态2的规则则远远跌落到其起始水平之下。

若没有投影机制，这些经济体将会终结，而许多最终攀升的运行在途中也曾触及下限。

均值动态很好地解释了这两种结果。

```{code-cell} ipython3
def mean_newton_step(s, G, P):
    "Average Newton step -E_i U'/E_i U'' in each state when both classes use the rule s."
    R = (N*s[None, :] - G[None, :]) / (N*s[:, None])
    EU1 = (P * U_prime(s[:, None], R)).sum(axis=1)
    EU2 = (P * U_double(s[:, None], R)).sum(axis=1)
    return -EU1 / EU2

def jacobian(f, x, h=1e-6):
    J = np.empty((len(x), len(x)))
    for k in range(len(x)):
        e = np.zeros(len(x))
        e[k] = h
        J[:, k] = (f(x + e) - f(x - e)) / (2*h)
    return J

for s_eq, name in [(s2, "low inflation "), (s2_hi, "high inflation")]:
    eig = np.linalg.eigvals(jacobian(lambda s: mean_newton_step(s, G2, P2), s_eq))
    print(f"{name} equilibrium {np.round(s_eq, 4)}:  eigenvalues {np.round(np.sort(eig.real), 3)}")
```

在低通胀均衡处，两个特征值均为负，因此它是相关微分方程的一个稳定静止点。

在高通胀均衡处，有一个特征值为正：它是一个鞍点，实现回报中的噪声会沿着不稳定方向把规则推离该点，这一方向一侧通向低通胀均衡，另一侧通向崩溃。

这正是第二部分在随机经济中出现的稳定性反转：与那里的 $\pi_{\text{high}}$ 一样，高通胀均衡是收敛到低通胀均衡与货币经济终结之间的分界点。

学习本身就排除了高通胀均衡；而投影机制在下行方向上发挥作用，使得正在崩溃的经济体得以维系下去。

```{solution-end}
```

```{exercise-start}
:label: olg_ex4
```

高通胀稳态处于"拉弗曲线错误一侧"这一论断可以被精确化。

在一个总通胀率为 $\pi$ 的平稳均衡中，实际余额为 $h = (w_1 - w_2\pi)/2$，政府征收铸币税 $h(1 - 1/\pi)$。

绘制铸币税作为 $\pi$ 的函数的图像，标出两个平稳均衡，并利用该图解释为何提高 $G$ 会提高低平稳通胀率，却会*降低*高平稳通胀率，以及当 $G$ 过大时会发生什么。

```{exercise-end}
```

```{solution-start} olg_ex4
:class: dropdown
```

下面是经济体7C的通货膨胀税拉弗曲线。

```{code-cell} ipython3
def seigniorage(π):
    return (w1_d - w2_d*π)/2 * (1 - 1/π)

π_grid = np.linspace(1.01, 5.5, 500)
peak = π_grid[np.argmax(seigniorage(π_grid))]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(π_grid, seigniorage(π_grid), lw=1.6)
for G_try, style in [(1.0, '-'), (1.03, '--'), (1.0505, ':')]:
    ax.axhline(G_try, color='C3', lw=0.9, ls=style, label=f"$G = {G_try}$")
ax.plot([π_low, π_high], [seigniorage(π_low), seigniorage(π_high)], 'ko', ms=6)
ax.axvline(peak, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r"gross inflation $\pi$")
ax.set_ylabel("seigniorage")
ax.set_ylim(0, 1.5)
ax.legend(frameon=False)
plt.show()

print(f"peak of the Laffer curve at π = {peak:.4f}, raising {seigniorage(peak):.4f}")
print(f"the two steady states raise {seigniorage(π_low):.4f} and {seigniorage(π_high):.4f}")
```

两个平稳均衡是通货膨胀税拉弗曲线与所需收入 $G$ 相交的两个通胀率，分别位于曲线峰值的两侧。

低的那个位于上升支上，因此在那里为更大的赤字融资需要*更高*的通胀率：这就是古典的比较静态结果。

高的那个位于下降支上，在那里更高的通胀带来的收入*更少*，因此更大的赤字要靠*更低*的平稳通胀率来融资，这种反古典的比较静态结果正是使高均衡如此棘手的原因。

随着 $G$ 上升，水平线向上移动，两个交点也逐渐趋于合并。

超过曲线峰值之后，就根本不存在平稳均衡：赤字超过了通货膨胀税所能筹集到的最大收入。

下表追踪了两个根随 $G$ 上升的变化。

```{code-cell} ipython3
rows = []
for G_try in (0.8, 1.0, 1.04, 1.05, 1.0505, 1.06):
    A1_try = (w1_d + w2_d - 2*G_try)/w2_d
    disc = A1_try**2 - 4*A2
    if disc < 0:
        rows.append([G_try, np.nan, np.nan, "none"])
    else:
        r = np.sort(np.roots([1, -A1_try, A2]))
        rows.append([G_try, r[0], r[1], "two"])

pd.DataFrame(rows, columns=["$G$", "low $\\pi$", "high $\\pi$",
                            "stationary equilibria"]).round(4)
```

随着 $G$ 上升，两个根逐渐靠拢，并在 $G \approx 1.0505$ 处相撞，此后模型就根本不存在平稳均衡。

这个数字并非巧合。

将 {eq}`inflation_map` 不动点方程的判别式设为零，就得到一个位于 $\pi = \sqrt{A_2} = \sqrt{w_1/w_2}$ 处的重根，而使此情形发生的赤字水平，恰好就是上面计算出的拉弗曲线峰值。

```{code-cell} ipython3
G_max = (w1_d + w2_d - 2*np.sqrt(A2)*w2_d) / 2
print(f"roots collide at G       = {G_max:.6f},  where π = √(w1/w2) = {np.sqrt(A2):.6f}")
print(f"peak of the Laffer curve = {seigniorage(np.sqrt(A2)):.6f}  at π = {np.sqrt(A2):.6f}")
```

政府所能融资的最大赤字，就是通货膨胀税的最大值，而在这个赤字水平下，两个平稳均衡已经合而为一。

```{solution-end}
```

```{exercise-start}
:label: olg_ex5
```

本练习将 {ref}`br_bubble` 中的泡沫引入第二部分的模型中。

设 $G = 0$，同时保持 $w_1 = 6$ 和 $w_2 = 1$。

1. 证明平稳通胀率为 $\pi = 1$ 和 $\pi = w_1/w_2 = 6$，并将其对应于 {doc}`bounded_rationality` 中货币供给恒定的货币模型的均衡。
1. 证明 $T$-映射 {eq}`olg_tmap` 为常数，并说明这对最小二乘学习意味着什么。
1. 从几个低于6的初始信念出发，运行 `ls_dynamics`（设 `G=0`），并与从相同起点出发的理性预期路径进行比较。
   一个由最小二乘学习者组成的社会会选择哪一个泡沫？

```{exercise-end}
```

```{solution-start} olg_ex5
:class: dropdown
```

当 $G = 0$ 时，$A_1 = 1 + w_1/w_2$ 且 $A_2 = w_1/w_2$，因此不动点方程 $\pi^2 - A_1 \pi + A_2 = 0$ 可因式分解为 $(\pi - 1)(\pi - w_1/w_2) = 0$。

在没有赤字的情况下，货币供给是恒定的，因此这就是 {doc}`bounded_rationality` 中 $\mu = 1$ 的货币模型。

在那里，$\pi = 1$ 是货币数量论均衡 $c = 0$，此时价格水平与（恒定的）货币供给成正比；而 $\pi = w_1/w_2$ 则是每条 $c > 0$ 的泡沫路径在实际余额收缩至零时所趋近的通胀率。

在 $\pi = w_1/w_2$ 处，储蓄为 $(w_1 - w_2\pi)/2 = 0$，因此它是一个正在去货币化的经济的极限，而非货币稳态。

当 $G = 0$ 时，对任何使储蓄保持为正的信念而言，$T$-映射均为 $T(\beta) = s(\beta)/s(\beta) = 1$：如果主体预期*任何*恒定的通胀率，他们就会储蓄一个恒定的数量，而恒定的货币存量追逐恒定的实际余额，意味着恒定的价格水平。

因此关于信念的微分方程为 $d\beta/d\tau = 1 - \beta$，最小二乘信念应当从任何低于 $6$ 的值收敛到 $1$。

```{code-cell} ipython3
print("T-map at G = 0:", [T_map(β, G=0.0) for β in (0.5, 2.0, 4.0, 5.9)])

def re_path_G0(π0, T=25):
    "Rational expectations path π_{t+1} = A1 - A2/π_t with G = 0."
    A1_0, A2_0 = (w1_d + w2_d) / w2_d, w1_d / w2_d
    path = [π0]
    for _ in range(T):
        nxt = A1_0 - A2_0/path[-1]
        path.append(nxt if path[-1] > 0 and nxt > 0 else np.nan)
    return np.array(path)

dates = (10, 100, 1_000, 20_000)
rows = []
for π0 in (0.5, 1.5, 3.0, 5.0, 5.9):
    b, infl, ok = ls_dynamics(π0, T=dates[-1], G=0.0)
    rows.append([π0, "yes" if ok else "no", *b[[d - 1 for d in dates]],
                 re_path_G0(π0)[-1]])

pd.DataFrame(rows, columns=["initial belief $\\pi_0$", "survived",
                            *[f"belief at t = {d:,}" for d in dates],
                            "RE inflation at t = 25"]).set_index("initial belief $\\pi_0$").round(4)
```

从所有低于 $6$ 的起点出发，最小二乘信念都走向 $\pi = 1$：学习者选择了 $c = 0$，学习扼杀了泡沫。

然而收敛的速度较慢，从较高的初始信念出发收敛得尤为缓慢。

由于没有通胀，价格水平不再增长，因此最小二乘法对各期观测大致赋予相等的权重，其增益大致以 $1/t$ 的速度衰减，这与我们在第二部分中发现的较大有效增益不同。

而且，从接近 $6$ 的信念出发，第一个实现的通胀率恰好为 $1$，这会立即把信念向下拉；第二期则是深度通缩：储蓄从几乎为零猛增，价格水平骤然崩溃，而那些在高价格水平下获得的早期观测值，会在随后数千期内主导回归结果。

从相同起点出发的理性预期路径（起点高于 $1$）则恰恰相反：它们是泡沫路径，通胀率不断上升，趋向 $w_1/w_2 = 6$。

（从低于 $1$ 的起点出发，理性预期路径没有正的价格水平。）

{ref}`olg_ex6` 中的符号结果并不适用于 $\pi = 6$ 处，因为在那里储蓄为零，$T$-映射没有定义。

```{solution-end}
```

```{exercise-start}
:label: olg_ex6
```

本练习证明， {ref}`olg_reversal` 中的稳定性反转是关于该模型的一个定理，而非经济体7C参数所特有的性质。

令 $F(x) = x^2 - A_1 x + A_2$，于是平稳通胀率就是 $F$ 的根，并令 $s(\beta) = (w_1 - w_2\beta)/2$。

1. 证明
   $g(\pi) - \pi = -F(\pi)/\pi$ 以及 $T(\beta) - \beta = \dfrac{w_2}{2}\,\dfrac{F(\beta)}{s(\beta) - G}$。
1. 由此推出，在每个储蓄为正的稳态处，
   $\operatorname{sign}(T' - 1) = -\operatorname{sign}(g' - 1)$。
   由此得出，每当存在两个这样的稳态时，较低的那个在 E-稳定意义下是稳定的，而在理性预期动态下是不稳定的；较高的那个则相反。
1. 对随机抽取且存在两个稳态的 $(w_1, w_2, G)$ 数值进行验证。

```{exercise-end}
```

```{solution-start} olg_ex6
:class: dropdown
```

对于第一个恒等式，

$$
g(\pi) - \pi = A_1 - \frac{A_2}{\pi} - \pi = -\frac{\pi^2 - A_1\pi + A_2}{\pi} = -\frac{F(\pi)}{\pi}.
$$

对于第二个，

$$
T(\beta) - \beta = \frac{s - \beta(s - G)}{s - G} = \frac{s(1 - \beta) + G\beta}{s - G},
$$

以及

$$
2\bigl[s(1 - \beta) + G\beta\bigr] = (w_1 - w_2\beta)(1 - \beta) + 2G\beta
= w_2\beta^2 - (w_1 + w_2 - 2G)\beta + w_1 = w_2 F(\beta).
$$

在稳态 $\pi^*$ 处有 $F(\pi^*) = 0$，因此对这两个恒等式求导可得

$$
g'(\pi^*) - 1 = -\frac{F'(\pi^*)}{\pi^*},
\qquad
T'(\pi^*) - 1 = \frac{w_2}{2}\,\frac{F'(\pi^*)}{s(\pi^*) - G}.
$$

如果在稳态处储蓄为正，那么 $\pi^* = s/(s - G) > 0$ 就要求 $s - G > 0$，因此两个分母都为正，从而

$$
\operatorname{sign}(T' - 1) = \operatorname{sign} F'(\pi^*) = -\operatorname{sign}(g' - 1).
$$

$F$ 是一条开口向上的抛物线，因此在其较小的根处 $F' < 0$，在较大的根处 $F' > 0$。

因此，在 $\pi_{\text{low}}$ 处，$T' < 1$（E-稳定）而 $g' > 1$（在理性预期下不稳定，因为 $g' = A_2/\pi^2 > 0$）；在 $\pi_{\text{high}}$ 处，$T' > 1$ 而 $g' < 1$。

当两个根合并时（在 {ref}`olg_ex4` 中所述的最大可融资赤字处），$F' = 0$，两个斜率都等于一。

下面是一个数值验证，使用 $T'(\beta) = (w_2/2)\,G/(s(\beta) - G)^2$ 和
$g'(\pi) = A_2/\pi^2$。

```{code-cell} ipython3
rng = np.random.default_rng(0)
checked = 0
while checked < 1_000:
    w2_r = rng.uniform(0.5, 5)
    w1_r = w2_r * rng.uniform(1.1, 10)               # w1 > w2
    G_r = rng.uniform(0, (np.sqrt(w1_r) - np.sqrt(w2_r))**2 / 2)   # below the largest deficit
    A1_r, A2_r = (w1_r + w2_r - 2*G_r) / w2_r, w1_r / w2_r
    for π_star in np.roots([1, -A1_r, A2_r]).real:
        s = (w1_r - w2_r*π_star) / 2
        T_slope = (w2_r/2) * G_r / (s - G_r)**2
        g_slope = A2_r / π_star**2
        assert s > G_r and np.sign(T_slope - 1) == -np.sign(g_slope - 1)
    checked += 1
print(f"sign(T' - 1) = -sign(g' - 1) at both steady states in all {checked} economies")
```

```{solution-end}
```

```{exercise-start}
:label: olg_ex7
```

{cite:t}`Chung1990` 估计了政府学习规则本身：他把政府卡尔曼滤波器的参数视为战后美国通胀与失业数据似然函数中的自由参数。

这里给出其问题的一个简化版本。

利用 `constant_gain`（增益取 $0.02$）模拟 $(\pi_t, U_t)$ 共16000期。

假设一位计量经济学家对该模型的一切都了如指掌，唯独不知道增益：$\theta$、$U^*$、$\sigma_\eta$、$\sigma_u$，以及政府的初始信念和矩矩阵都是已知的。

1. 解释为何对于任意候选增益 $\gamma$，计量经济学家都能从数据中重建出政府的信念以及目标通胀 $g_{t-1}(\gamma)$。
1. 利用前 $T$ 个观测值，在 $0.02$ 附近的增益网格上，对 $T$ = 250、1000、4000 和 16000，计算通胀的高斯对数似然
   $\ell_T(\gamma) = -\sum_{t \le T} (\pi_t - g_{t-1}(\gamma))^2 / (2\sigma_\eta^2)$（相差一个常数）。
   随着 $T$ 增大，$\ell_T$ 的形状如何变化？
1. 为什么关于增益的信息会不断积累？

```{exercise-end}
```

```{solution-start} olg_ex7
:class: dropdown
```

政府的信念是数据的确定性函数：给定初始信念和矩矩阵，更新 $(\alpha_t, M_t)$ 的递归式只使用观测到的 $(\pi_t, U_t)$ 以及增益。

因此，对于每一个候选增益，计量经济学家都可以在数据上重新运行政府的递归式，并从 {eq}`phillips_rule` 中读出目标值 $g_{t-1}(\gamma)$。

下面的函数一次性对一整个增益网格完成了这一操作。

```{code-cell} ipython3
def reconstruct_targets(π, U, gains):
    """
    Target inflation g_{t-1}(γ) that a government with gain γ would have chosen,
    reconstructed from data on (π_t, U_t), for each candidate γ in `gains`.
    """
    gains = np.asarray(gains)
    K = len(gains)
    α = np.tile(α_consistent, (K, 1))
    M = np.tile(np.eye(2), (K, 1, 1))
    targets = np.empty((len(π), K))
    for t in range(len(π)):
        targets[t] = -α[:, 0] * α[:, 1] / (1 + α[:, 1]**2)
        x = np.array([1.0, π[t]])
        M = M + gains[:, None, None] * (np.outer(x, x) - M)
        err = U[t] - α @ x
        step = np.linalg.solve(M, (err[:, None] * x)[:, :, None])[:, :, 0]
        α = α + gains[:, None] * step
    return targets

true_gain = 0.02
g_true, π_data, U_data = constant_gain(T=16_000, gain=true_gain, seed=1)
gains = np.round(np.linspace(0.017, 0.023, 61), 6)    # includes 0.02 exactly
targets = reconstruct_targets(π_data, U_data, gains)
print(f"largest error in reconstructed targets at the true gain: "
      f"{np.max(np.abs(targets[:, 30] - g_true)):.1e}")
print(f"first period with target inflation below 1: {np.argmax(g_true < 1)}")
```

现在给出对数似然，对四个样本量分别归一化为在最大值处为零。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4.5))
rows = []
for T in (250, 1_000, 4_000, 16_000):
    ssr = np.sum((π_data[:T, None] - targets[:T])**2, axis=0)
    loglik = -ssr / (2 * σ_η**2)
    loglik -= loglik.max()
    ax.plot(gains, loglik, label=f"$T = {T:,}$")
    rows.append([T, gains[np.argmax(loglik)], loglik[20], loglik[40]])
ax.axvline(true_gain, color='k', lw=0.8, ls=':')
ax.set_ylim(-60, 3)
ax.set_xlabel(r"candidate gain $\gamma$")
ax.set_ylabel(r"$\ell_T(\gamma) - \max_\gamma \ell_T$")
ax.legend(frameon=False)
plt.show()

pd.DataFrame(rows, columns=["T", "maximizing gain",
                            f"drop at γ = {gains[20]:.3f}",
                            f"drop at γ = {gains[40]:.3f}"]).set_index("T").round(4)
```

在真实增益处，重建出的目标值与实际目标值相等，因此残差恰好就是意外冲击 $\eta_t$。

任何其他增益都意味着一段不同的信念和目标值历史，对数似然会对其加以惩罚。

在只有250个观测值时，似然几乎是平的。

样本在第一次逃逸发生之前就已结束，而当信念停留在自我确认均衡附近时，无论增益为何，它们几乎都不会移动，因此数据几乎透露不出关于增益的信息。

一旦逃逸开始，似然就在真实增益处变得尖锐地达到峰值，并且在一个固定的错误增益处的下降幅度会随 $T$ 不断增大，在较晚的样本中大致与 $T$ 成正比。

关于增益的信息之所以不断积累，是因为恒定增益永远不会停止塑造信念：每一期政府都以相同的权重对最新的观测做出反应，因此每一期都带来关于该权重的新证据，而其中大部分证据都出现在逃逸期间，此时信念变动很大。

若使用递减的增益，增益参数对信念的影响就会逐渐消退，关于它们的信息也会随之消退；{doc}`prospects_bounded_rationality` 将探讨这种对比。

钟（Chung）所研究的问题比这个更难：他还利用几十年的季度数据（而不是数千期数据），估计了 $\theta$、$U^*$、冲击方差以及政府的先验。

```{solution-end}
```
