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
    The environment: 环境设定
    'Part 1: a stochastic deficit': 第一部分：随机赤字
    'Part 1: a stochastic deficit::Stationary rational expectations equilibrium': 平稳理性预期均衡
    'Part 1: a stochastic deficit::Learning by successive generations': 连续几代人的学习
    'Part 1: a stochastic deficit::Do they find it?': 他们能找到它吗？
    'Part 1: a stochastic deficit::How much do the agents have to be told?': 主体需要被告知多少信息？
    'Part 2: a constant deficit and two steady states': 第二部分：恒定赤字与两个稳态
    'Part 2: a constant deficit and two steady states::Least squares dynamics': 最小二乘动态
    'Part 2: a constant deficit and two steady states::The stability reversal': 稳定性反转
    'Part 2: a constant deficit and two steady states::Marimon and Sunder''s experiment': 马里蒙与桑德的实验
    'Part 2: a constant deficit and two steady states::A warning': 一个警示
    A government learning the Phillips curve: 一个学习菲利普斯曲线的政府
    A government learning the Phillips curve::Two stories about post-war inflation: 关于战后通货膨胀的两种说法
    A government learning the Phillips curve::The model: 模型
    A government learning the Phillips curve::The consistent equilibrium: 一致均衡
    'A government learning the Phillips curve::Constant-coefficient beliefs: learning the inflation bias': 恒定系数信念：学习通货膨胀偏误
    'A government learning the Phillips curve::Random-coefficient beliefs: escaping toward the optimum': 随机系数信念：逃向最优
    A government learning the Phillips curve::From escape dynamics to the *Conquest of American Inflation*: 从逃逸动态到《美国通货膨胀的征服》
    Concluding remarks: 结束语
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

## 概述

{doc}`bounded_rationality` 展示了一些具有多重理性预期均衡的货币经济模型。

本讲座采用其中之一——萨缪尔森的法定货币世代交叠模型，该模型曾被布莱恩特、华莱士等人用来研究通货膨胀融资——并用不按均衡价格水平运动法则进行预测的"适应性"主体，取代了萨缪尔森模型中的主体。

我们出于两个不同的目的做这件事，做两次。

**第一部分**在模型中引入*随机*的政府赤字，并探讨连续几代人是否能够在通过试错摸索的过程中，作为一个社会最终收敛到理性预期均衡。

答案是肯定的。

每一代人观察前几代人所发生的情况，并使用 **鲁滨斯-门罗（Robbins–Monro）** 递归，朝着改进效用的方向调整自己的储蓄决策。

只要时间足够长，经济就会收敛到我们独立计算出的稳态均衡。

这个练习也说明了学习对所需学习内容的复杂程度有多敏感：一个很少出现的赤字状态由于观测数据到来得慢，因而被缓慢地学习到。

**第二部分**去掉了随机性，转向一个更尖锐的问题。

在*恒定*赤字下，模型存在**两个**稳态均衡，一个低通胀，一个高通胀。

低通胀均衡在帕累托意义上优于另一个均衡。

在理性预期动态下，低通胀均衡是**不稳定**的，而高通胀均衡是稳定的：理论把经济推向坏的结果。

在最小二乘学习下，稳定性**恰好相反**。

而当 {cite:t}`MarimonSunder1993` 用付费的人类被试将该经济作为实验室实验来运行时，被试的行为表现得像适应性模型，而不像理性预期模型。

这是 {cite:t}`Sargent1993` 中最有力的证据，说明适应性动态确实在作为一种选择机制发挥实际作用，但本杰明·本塔尔提出的一个反例对此有所缓和，提醒人们不要断言适应过程一定会挑出*好*的均衡。

随后我们以**第三个应用**作结，其中的适应性主体是一个正在学习菲利普斯曲线的政府。

此时最小二乘学习选择了*坏*的高通胀均衡，但一个对自身模型抱有怀疑的政府却能**逃脱**并转向好的均衡，这一现象正是萨金特后来关于《美国通货膨胀的征服》研究的萌芽。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import pandas as pd
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
```

## 环境设定

该经济由存活两期的世代交叠主体构成。

在每个日期 $t \geq 1$，会出生 $N$ 个相同的主体，年轻时被赋予 $w_1$ 单位的单一消费品，年老时被赋予 $w_2$ 单位。

年轻主体对生命周期消费 $(c_1, c_2)$ 的偏好由 $u(c_1) + u(c_2)$ 的期望值排序。

我们始终使用对数效用，$u(c) = \ln c$。

政府印制货币来为其支出 $G_t$ 融资，受制于以下预算约束：

```{math}
:label: govt_budget

G_t = \frac{H_t - H_{t-1}}{p_t},
```

其中 $H_t$ 是 $t$ 期的年轻人在 $t+1$ 期携带的货币存量，$p_t$ 是价格水平。

货币是唯一的价值储藏手段，因此年轻主体的储蓄 $s_t$ 完全以实际余额的形式存在，市场出清要求

```{math}
:label: market_clearing

\frac{H_t}{p_t} = s_t N .
```

记 $R_t = p_t / p_{t+1}$ 为 $t$ 与 $t+1$ 期之间货币的总回报率。

将 {eq}`govt_budget` 与 {eq}`market_clearing` 结合，得到一个我们会不断用到的关系式：货币的回报完全由储蓄行为和赤字决定，

```{math}
:label: return_from_saving

R_t = \frac{N s_{t+1} - G_{t+1}}{N s_t} .
```

只有当下一代愿意吸纳尚未偿付的存量加上新发行的部分时，货币才能保有其价值。

```{note}
{eq}`return_from_saving` 值得驻足思考，因为正是它使这一系统成为一个*自我指涉*的系统，这一性质在 {doc}`ls_learning` 中得到了深入研究。

今天的年轻人从储蓄中获得的回报，取决于明天的年轻人选择储蓄多少，而这又取决于明天的年轻人预期能获得多少回报。

没有人的信念是关于外生过程的。
```

## 第一部分：随机赤字

现在让政府支出遵循一个马尔可夫链，

$$
\pi(i, j) = \operatorname{Prob}\{G_{t+1} = \bar G_j \mid G_t = \bar G_i\},
$$

覆盖一组有限的水平列表 $G = [\bar G_1, \ldots, \bar G_n]$。

### 平稳理性预期均衡

在平稳均衡中，储蓄取决于当前的赤字状态，因此可用一个 $n \times 1$ 的向量 $s = [s_1, \ldots, s_n]$ 来描述，货币的回报则用一个 $n \times n$ 的矩阵 $R(i,j)$ 描述。

观察到 $G_t = \bar G_i$ 的年轻主体选择 $s_i$ 以最大化

$$
V(s_i) = u(w_1 - s_i) + \sum_j u\bigl(w_2 + s_i R(i,j)\bigr) \pi(i,j),
$$

其一阶条件为

```{math}
:label: olg_foc

u'(w_1 - s_i) = \sum_{j=1}^n u'\bigl(w_2 + s_i R(i,j)\bigr) R(i,j) \pi(i,j).
```

将 {eq}`return_from_saving`——它在平稳均衡中写作 $R(i,j) = (s_j N - \bar G_j)/(s_i N)$——代入，就把 {eq}`olg_foc` 变成了一个只关于储蓄率的 $n$ 个非线性方程组：

```{math}
:label: olg_equilibrium

u'(w_1 - s_i)
= \sum_{j=1}^n u'\left(w_2 + \frac{s_j N - \bar G_j}{N}\right)
  \cdot \frac{s_j N - \bar G_j}{s_i N} \cdot \pi(i,j) .
```

当且仅当 {eq}`olg_equilibrium` 对每个 $i$ 都有满足 $s_i \in (0, w_1)$ 的解时，平稳均衡才存在。

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
    s = fsolve(equilibrium_residual, np.full(len(G), guess), args=(G, P))
    R = np.array([[(s[j]*N - G[j])/(s[i]*N) for j in range(len(s))]
                  for i in range(len(s))])
    return s, R
```

依照 {cite:t}`Sargent1993`，我们研究两个仅在政府支出过程上有所不同的经济体，其中 $w_1 = 20$，$w_2 = 10$，$N = 1$，因此 $\bar G$ 是按人均年轻人计量的。

在第一个经济体中，赤字始终为零，链是一个公平硬币。

```{code-cell} ipython3
G1, P1 = np.array([0.0, 0.0]), np.full((2, 2), 0.5)
s1, R1 = stationary_equilibrium(G1, P1)
print(f"saving rates      {np.round(s1, 4)}")
print(f"returns\n{np.round(R1, 4)}")
```

由于没有赤字需要融资，货币简单地保有其价值：主体在任一状态下都储蓄 $5$，且 $R \equiv 1$。

在第二个经济体中，状态1的赤字是 $0.8$，状态2的赤字为零，并且该链具有持续性。

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

注意 $R$ 中的规律：每当政府即将出现赤字时（第一列），货币的回报都很差，因为新增发行会稀释未偿付的存量。

### 连续几代人的学习

现在把均衡从主体那里拿走。

我们仍假设这些主体了解自己的效用函数，并且记得像他们一样的早期主体的经历，但他们*不*被告知回报的分布，而这恰恰是 {eq}`olg_foc` 所需要的对象。

相反，每一代人朝着*实现的*经验所显示的更好方向，调整其前辈的储蓄决策。

储蓄了 $s$ 并获得回报 $R$ 的主体所实现的一生效用为

$$
U(s) = u(w_1 - s) + u(w_2 + s R),
$$

其导数为

$$
U'(s) = -u'(w_1 - s) + u'(w_2 + sR) R,
\qquad
U''(s) = u''(w_1 - s) + u''(w_2 + sR) R^2 .
$$

与均衡的联系在于：在理性预期均衡中，$V'(s_i) = E_t U'(s_i)$，$V''(s_i) = E_t U''(s_i)$，其中 $E_t$ 是在 $G_t = \bar G_i$ 条件下取期望。

因此，能够计算出 $E_t U'$ 的主体只需将其设为零即可完成任务。

而我们的主体无法这样做，必须从经验中估计它。

他们逐状态地使用一种 **鲁滨斯-门罗（Robbins–Monro）** 算法。

令 $\tau_i$ 计数赤字状态 $\bar G_i$ 迄今为止被访问的次数，令 $\gamma_\tau$ 为一个递减的增益序列。

规则为

```{math}
:label: robbins_monro

\begin{aligned}
M(i, \tau_i + 1) &= M(i, \tau_i) + \gamma_{\tau_i}\bigl(U''(s(i, \tau_i)) - M(i, \tau_i)\bigr), \\
s(i, \tau_i + 1) &= s(i, \tau_i) - \gamma_{\tau_i} M(i, \tau_i + 1)^{-1} U'(s(i, \tau_i)) .
\end{aligned}
```

$M$ 累积了对 $E_t U''$ 的运行估计，而储蓄规则针对它采取牛顿步。

如果 $\tau_i \to \infty$，那么 $M(i, \tau_i) \to E_i U''$，且 $s(i, \tau_i)$ 逼近 $E_i U'(s_i) = 0$ 的解，这恰好就是均衡条件 {eq}`olg_foc`。

该设定的两个特征值得评述。

**两类主体。** 要评估一项储蓄决策，我们必须等到该主体消费的*两*期都已知，因此人口被划分为一个"奇"子序列和一个"偶"子序列。

奇数主体在奇数期重设其规则，只从更早的奇数主体学习；偶数主体同理。

这与第二部分讨论的实验室实验中所用的手法相同。

**一个投影装置。** {eq}`robbins_monro` 中没有任何东西阻止牛顿步把储蓄推到赤字以下，一旦如此，{eq}`market_clearing` 就没有正的价格水平，经济也就不复存在了。

因此我们把 $s$ 限制在存在均衡价格水平的区域内。

这种装置对于收敛定理也是必要的；{doc}`ls_learning` 详细讨论了该投影装置。

```{code-cell} ipython3
def U_prime(s, R):
    return -1/(w1 - s) + R/(w2 + s*R)

def U_double(s, R):
    return -1/(w1 - s)**2 - R**2/(w2 + s*R)**2


def simulate(G, P, T=50_000, s0=None, seed=0, tau0=20, band=0.4):
    """
    Overlapping generations learning a state-contingent saving rule.

    Returns the history of saving rules (shape (T, 2, n): time, class, state)
    and the realized returns.
    """
    rng = np.random.default_rng(seed)
    n = len(G)
    s = np.array(s0, float)
    M = np.array([[U_double(s[j, k], 1.0) for k in range(n)] for j in range(2)])
    τ = np.zeros((2, n), int)
    floor = G/N + band                 # projection facility: saving must cover the deficit

    hist = np.empty((T, 2, n))
    R_hist = np.full(T, np.nan)
    i, prev = 0, None

    for t in range(T):
        j = t % 2                      # which class is young this period
        s_t = s[j, i]

        if prev is not None:           # last period's young now learn their return
            j_p, i_p, s_p = prev
            R = (N*s_t - G[i]) / (N*s_p)          # the return on currency
            R_hist[t-1] = R
            τ[j_p, i_p] += 1
            g = 1 / (τ[j_p, i_p] + tau0)
            M[j_p, i_p] += g * (U_double(s_p, R) - M[j_p, i_p])
            step = g * U_prime(s_p, R) / M[j_p, i_p]
            s[j_p, i_p] = np.clip(s[j_p, i_p] - step, floor[i_p], w1 - 0.4)

        prev = (j, i, s_t)
        hist[t] = s
        i = rng.choice(n, p=P[i])

    return hist, R_hist
```

```{note}
该模拟从不计算价格水平。

由于 {eq}`return_from_saving` 完全用储蓄率和赤字来表达货币的回报，该模型的*实际*一面是自成一体的。

这不仅仅是为了方便：每当政府出现赤字时，这些经济体中的名义货币存量都会无限增长，因此一个直接跟踪 $H_t$ 和 $p_t$ 的模拟，早在学习收敛之前就会发生溢出。
```

### 他们能找到它吗？

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

两个经济体都收敛了，而且两类主体也收敛到同一规则，即使他们彼此互不学习。

黑色虚线是我们先前从 {eq}`olg_equilibrium` 计算出的均衡储蓄率，这是模拟中任何主体都无法获知的量。

```{code-cell} ipython3
rows = []
for T in (5_000, 50_000, 200_000):
    h1, _ = simulate(G1, P1, T=T, s0=start)
    h2, _ = simulate(G2, P2, T=T, s0=start)
    rows.append([T, *h1[-1].mean(axis=0), *h2[-1].mean(axis=0)])

pd.DataFrame(rows, columns=["T", "econ 1 state 1", "econ 1 state 2",
                            "econ 2 state 1", "econ 2 state 2"]).set_index("T").round(4)
```

```{code-cell} ipython3
print(f"rational expectations:  economy 1 = {np.round(s1, 4)},  economy 2 = {np.round(s2, 4)}")
```

收敛很慢——这正是 $1/\tau$ 增益所带来的代价——但它确实收敛了，而且收敛到了正确的地方。

### 主体需要被告知多少信息？

上面的设定让主体为每一个赤字水平学习*单独*的储蓄率。

实际上他们是在**非参数地**逐状态学习策略函数 $s = f(G)$。

由于只有两个状态，这在这里是可行的。

它有两个 Sargent 强调的缺点。

第一，很少被访问到的状态被学习得很慢，因为观测数据到来得慢。

{ref}`olg_ex1` 使这一点更加具体。

第二，当状态数量很大时，每个状态一个参数就变得难以处理。

计量经济学家的应对方式会是施加一个**参数化**形式 $s = f(G, \theta)$，其中 $\theta$ 是低维的，然后用每一个观测数据来估计它，把 {eq}`robbins_monro` 替换为由 $\partial U/\partial \theta$ 驱动的关于 $\theta$ 的递归。

这以*近似*为代价换取了速度：只有当族 $f(\cdot, \theta)$ 中的某个成员本身支持一个理性预期均衡时，参数化学习方案才能收敛到该均衡。

否则，它能做到的最好结果就是收敛到一个近似均衡。

```{note}
这正是学习模型与*计算*均衡的算法之间难以区分之处。

马塞特的参数化预期方法假设一个条件期望的参数化形式，进行模拟，把实现的结果对该参数化形式做回归以更新系数，然后迭代。

以递归形式写出，该算法就是一个与 {eq}`robbins_monro` 形状相同的非线性最小二乘递归。

萨金特的总结是："学习算法和均衡计算算法看起来彼此相似。"

均衡计算是由建模者运行的一种集中式学习算法；而学习型经济则是由主体运行的一种去中心化均衡计算。
```

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

政府以实际值计的预算约束为 $h_t = h_{t-1}/\pi_{t-1} + G$，其中 $h_t = m_t / p_t$，均衡要求 $h_t = s_t$。

在两者之间消去 $h$，得到一个关于通货膨胀率的自主差分方程，

```{math}
:label: inflation_map

\pi_{t+1} = g(\pi_t) \equiv A_1 - \frac{A_2}{\pi_t},
\qquad
A_1 = \frac{w_1 + w_2 - 2G}{w_2},
\qquad
A_2 = \frac{w_1}{w_2} .
```

平稳均衡就是 $g$ 的不动点，由于 $g$ 是一条双曲线，通常会有两个不动点。

我们使用 {cite:t}`MarimonSunder1993` 的"经济7C"参数：$w_1 = 6$，$w_2 = 1$，$G = 1$。

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

结果就在这里：在理性预期动态下，*低*通胀稳态均衡是**不稳定**的，而高通胀均衡是稳定的。

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
for _ in range(7):
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

红色的阶梯图从略高于低通胀稳态的地方开始，一路远离该稳态，径直走向高通胀稳态。

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

若从略低于低稳态的地方开始，经济就会崩溃（{eq}`inflation_map` 隐含的价格水平变为负值）；若从略高于低稳态的地方开始，经济则会收敛到 $\pi = 3$。

这一点之所以重要，是因为两个均衡的排序方式。

低通胀均衡的比较静态是"古典的"：提高赤字 $G$ 会降低 $A_1$，从而提高低稳态通货膨胀率。

高通胀均衡的比较静态则恰恰相反，因为此时经济处在通货膨胀税**拉弗曲线**的错误一侧；参见 {ref}`olg_ex3`。

而且低通胀均衡在帕累托意义上优于该模型中其他任何均衡，无论是稳态还是非稳态。

因此，理性预期动态把经济从帕累托最优的结果中推开，而货币理论中大多数古典学说都依赖于选中那个被这些动态所排斥的均衡。

### 最小二乘动态

{cite:t}`MarcetSargent1989hyper` 研究了该经济的一个版本，其中主体不具有完全预见性，而是**运行一个回归**。

每一期，他们把价格水平对其自身的滞后值做回归，并用拟合的斜率作为对总通货膨胀率的预测，

$$
\beta_t = \frac{\sum_{s < t} p_s p_{s-1}}{\sum_{s < t} p_{s-1}^2},
\qquad
\pi^e_t = \beta_t ,
$$

然后依照 {eq}`saving_deterministic` 储蓄 $s_t = (w_1 - w_2 \beta_t)/2$。

由于 $\beta_t$ 只用到 $t-1$ 及之前的数据，因此不存在联立性问题：信念决定储蓄，储蓄决定价格水平，而新的价格水平又进入下一期的回归。

把 $\beta_t$ 写成

$$
\beta_t = \frac{\sum_{s<t} \pi_s \, p_{s-1}^2}{\sum_{s<t} p_{s-1}^2}
$$

就能看出该估计量在做什么：它是*过去实现的通货膨胀率*的加权平均，权重与 $p_{s-1}^2$ 成正比。

这种形式也使得该递归易于计算。

价格水平呈几何式增长，因此这两个求和会很快溢出，但如果我们在每一步都用最新的权重对两者进行重新标度，一切都会保持在同一数量级上。

```{code-cell} ipython3
def ls_dynamics(π_init, T=3000):
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
        if s_t <= G_d or not np.isfinite(s_t):
            return np.array(betas), np.array(infl), False   # no positive price level
        π_t = s_prev / (s_t - G_d)                          # realized gross inflation
        num = num/π_last**2 + π_t
        den = den/π_last**2 + 1.0
        π_last, s_prev = π_t, s_t
        betas.append(β)
        infl.append(π_t)

    return np.array(betas), np.array(infl), True


rows = []
for π0 in (1.2, 1.5, 1.9, 2.0, 2.1, 2.5, 2.9, 3.0):
    b, infl, ok = ls_dynamics(π0)
    rows.append([π0, b[-1], infl[-1], "yes" if ok else "no"])

pd.DataFrame(rows, columns=["initial belief $\\pi_0$", "belief $\\beta_T$",
                            "realized $\\pi_T$", "survived"]).round(5)
```

### 稳定性反转

在上表中每一个初始信念之下，最小二乘学习都收敛到了**低**通胀均衡，甚至包括 $\pi_0 = 3.0$，这正是理性预期动态所收敛*到*的稳态。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Least squares and rational expectations inflation paths"
    name: fig-olg-ls-vs-re
---
fig, ax = plt.subplots(figsize=(8, 4.5))

for π0 in (1.5, 2.5, 3.0):
    _, infl, _ = ls_dynamics(π0, T=400)
    ax.plot(infl, color='C0', lw=1.2,
            label="least squares dynamics" if π0 == 1.5 else None)

for π0 in (2.05, 2.5, 4.0):
    ax.plot(re_path(π0, 400), color='C3', lw=1.2, ls='--',
            label="rational expectations dynamics" if π0 == 2.05 else None)

ax.axhline(π_low, color='k', lw=0.8, ls=':')
ax.axhline(π_high, color='k', lw=0.8, ls=':')
ax.annotate("low", (330, π_low + 0.04), fontsize=9)
ax.annotate("high", (330, π_high + 0.04), fontsize=9)
ax.set_xlim(0, 400)
ax.set_ylim(1.2, 4.2)
ax.set_xlabel("$t$")
ax.set_ylabel("gross inflation")
ax.legend(frameon=False)
plt.show()
```

这两族路径朝相反方向运行。

这一边界是精确的：最小二乘学习从每一个直到并包括 $\pi_0 = 3$ 的初始信念都收敛到 $\pi = 2$，而超过这个值，经济就根本没有均衡价格水平了。

```{code-cell} ipython3
lo, hi = 3.0, 3.2
for _ in range(40):                      # bisect for the edge of the basin
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if ls_dynamics(mid)[2] else (lo, mid)
print(f"least squares dynamics survive for initial beliefs up to {lo:.6f}")
print(f"the rational-expectations-stable steady state is      {π_high}")
```

理性预期动态挑选为*该*稳态中稳定的那个稳态，恰好正是学习动态朝向另一个稳态的吸引域边缘。

为什么会出现这种反转？

在完全预见下，一个略高于 $\pi_{\text{low}}$ 的信念会被*验证并放大*——映射 $g$ 在该处的斜率为 $1.5$。

在最小二乘下，信念是过去实现的通货膨胀率长期历史的加权平均，而这种平均化会把预测拉低下去。

高稳态对于前瞻性的动态是稳定的，对于后顾性的动态则是排斥性的。

{cite:t}`BrunoFischer1990` 在一个密切相关的模型中，使用弗里德曼式的适应性预期而非最小二乘法，发现了同样的反转，因此这一结果并不依赖于具体估计量的细节。

### 马里蒙与桑德的实验

究竟哪一套动态描述的是人？

{cite:t}`MarimonSunder1993` 建立了一个实验室经济来找出答案。

一定数量的付费被试被分配为"年轻人"和"老年人"的角色，实现了正是这一模型。

每一期，年轻人提交一份表格，说明在每个价格下他们愿意向老年人供应多少产出以换取货币，实验人员则用来自老年人和政府的需求对市场进行出清。

暂时处于场外的被试则参与一个自带现金奖励的**预测博弈**，因此该实验对驱动经济运行的价格预期给出了直接的记录。

会话的长度对被试是未知的，但受公开宣布的规则支配，这些规则使经济从被试的角度看等价于一个永不结束的经济。

这里有两个重要的发现。

第一，实验中的通货膨胀路径远比理性预期动态收敛到的高稳态更接近于收敛到**低**稳态的最小二乘动态。

这一模式在他们所有的经济体中都成立。

第二，场外者的预测误差在大小上与使用相同数据的最小二乘预测者所会犯的误差相当。

被试们并没有做什么异乎寻常的事情，他们做的大致就是适应性模型所说的那样。

理论所称为不稳定的均衡，正是算法和人类都会选中的均衡。

```{note}
实验证据并非唯一的判别方式。

{cite:t}`Imrohoroglu1993` 在德国恶性通货膨胀数据上估计了该模型的理性预期版本，得到了相反的结果：他所估计的均衡沿着拉弗曲线的*坏*的一侧滑向高稳态速率，这与适应性动态所选中的均衡不一致。
```

### 一个警示

人们很容易得出结论说，适应性动态可靠地选出了帕累托占优的均衡。

萨金特给出了一个反例，由本杰明·本塔尔向他展示，该反例来自 {cite:t}`Brock1974` 效用函数中含货币模型的一个特例。

一个无限存活的家庭最大化 $\sum_t \beta^t (\ln c_t + \gamma \ln(m_t/p_t))$，受制于一系列预算约束，而政府通过印制货币来为固定支出 $g > 0$ 融资。

在参数限制 $\gamma(y - g) > g$ 下，该模型存在唯一的稳态均衡，其总通货膨胀率为

$$
\pi = \frac{(y-g)\gamma - \beta g}{(y-g)\gamma - g},
$$

它具有随赤字上升而上升的古典性质。

```{code-cell} ipython3
γ_b, β_b, y_b = 1.0, 1/1.05, 11.0

def brock_π(g):
    return ((y_b - g)*γ_b - β_b*g) / ((y_b - g)*γ_b - g)

pd.DataFrame({
    "$g$": [0.5, 1.0, 1.5, 2.0],
    "restriction $\\gamma(y-g) > g$": [γ_b*(y_b - g) > g for g in (0.5, 1.0, 1.5, 2.0)],
    "stationary $\\pi$": [brock_π(g) for g in (0.5, 1.0, 1.5, 2.0)],
    "return on currency $1/\\pi$": [1/brock_π(g) for g in (0.5, 1.0, 1.5, 2.0)],
}).round(5)
```

但该模型*同时*还存在一个由低于稳态价格水平的初始价格水平所索引的非稳态均衡连续统，在其中所有均衡的总通货膨胀率都收敛到 $\beta$——也就是说，经济最终陷入**通货紧缩**，货币的回报率趋近于 $1/\beta = 1.05$，实际余额爆炸性增长。

这些均衡之所以存在，是因为 $\gamma \ln(m/p)$ 所隐含的货币需求对回报率是如此富有弹性，以至于即使政府通过通货紧缩为货币支付利息，仍然能够筹集到足够的铸币税来为 $g$ 融资。

在此处应用最小二乘学习，选出的正是*古典稳态*均衡，与世代交叠模型中的情形一样。

不同之处在于福利排序：在布洛克模型中，所有的非稳态均衡都在帕累托意义上优于学习所选中的那个古典稳态均衡。

因此，最小二乘动态可靠地选出的是*古典*均衡。

而该均衡是否是好的均衡，则是另一个独立的问题，答案取决于模型本身。

## 一个学习菲利普斯曲线的政府

到目前为止的例子把适应性的*家庭*放入了货币经济之中。

我们以一个应用作为结尾，其中的适应性主体是**政府**，它在学习一种它可以据以行动的计量经济学关系。

这一环境不同于世代交叠模型——它是一个菲利普斯曲线经济——但机制是一样的：一个主体运行一个回归，据此采取行动，从而助长了它正在拟合的数据本身的生成。

这个例子还有另一层重要意义。

它是 {cite:t}`Sargent1993` 中最直接催生后续研究的例子：{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 对其进行了研究，并促使萨金特提出了 {cite:t}`Sargent1999` 的逃逸动态模型，即《美国通货膨胀的征服》，{doc}`Phillips curve lectures <phillips_escaping_nash>` 对该模型进行了深入的研究。

### 关于战后通货膨胀的两种说法

关于二战后美国通货膨胀的兴衰有一种广为人知的说法。

在这种说法中，私人部门始终具有理性预期并理解自然率假说，而政府在一段时间内则错误地相信存在一条*可利用的*菲利普斯曲线，即通货膨胀与失业之间存在持久的权衡关系。

政府用20世纪60年代的数据估计菲利普斯曲线，看到了一种权衡关系，并试图用更高的通货膨胀来换取更低的失业率。

结果是通货膨胀更高，而就业上却没有任何持久的收益：菲利普斯曲线不利地向上移动，正如自然率假说所说的那样必然发生。

但也存在一种反对的说法，由当时拟合菲利普斯曲线的经济学家所讲述。

他们主张自己的方法是*适应性的*，他们足够快地察觉到了这种不利的移动，从而给出了合理的建议，因此不至于误导政府。

下述模型的优点在于，一个单一的设定就能同时容纳这两种说法，由一个参数来决定实际发生的是哪一种。

### 模型

通货膨胀被分解为预期部分和非预期部分，

```{math}
:label: phillips_inflation

\pi_t = g_{t-1} + \eta_t,
```

其中 $g_{t-1}$ 是公众所预期的（且由政府控制的）通货膨胀，$\eta_t$ 是公众的预测误差，与早期信息正交。

*真实的*菲利普斯曲线是自然率曲线：只有**意外**部分 $\eta_t$ 会影响失业：

```{math}
:label: phillips_true

U_t = U^* - \theta(\pi_t - g_{t-1}) + u_t = U^* - \theta \eta_t + u_t,
\qquad \theta > 0.
```

政府并不知道这一点。

它相信失业取决于通货膨胀的*水平*，并拟合了一个非预期形式的回归

```{math}
:label: phillips_perceived

U_t = \alpha_{0} + \alpha_{1}\pi_t + \epsilon_t.
```

每一期它最小化 $\tfrac12 \mathbb{E}(U_t^2 + \pi_t^2)$，受制于它所感知的模型，由此得到短视的目标

```{math}
:label: phillips_rule

g_{t-1} = -\frac{\alpha_{0}\,\alpha_{1}}{1 + \alpha_{1}^2}.
```

这个系统是**自我指涉的**：政府的信念 $\alpha$ 通过 {eq}`phillips_rule` 确定其政策 $g$；该政策塑造了 $(\pi, U)$ 的联合分布；而这个分布正是政府的回归 {eq}`phillips_perceived` 所估计的对象。

```{code-cell} ipython3
U_star, θ_pc = 5.0, 1.0             # natural rate 5%, Phillips slope
σ_η, σ_u = 0.3, 0.3                 # inflation-surprise and unemployment shocks

def govt_target(α):
    "Government's myopic optimal target inflation."
    α0, α1 = α
    return -α0 * α1 / (1 + α1**2)
```

### 一致均衡

一个自我确认的、即**一致的**均衡，是一个信念向量 $\alpha$，它所引致的政策生成的数据经回归后返回同样的 $\alpha$。

在恒定政策 $g$ 下，意外部分 $\eta_t$ 是唯一使通货膨胀围绕其均值波动的因素，因此 $U$ 对 $\pi$ 的总体回归斜率恰好为 $-\theta$，截距为 $U^* + \theta g$。

将该不动点与决策规则 {eq}`phillips_rule` 联立求解，得到基德兰德和普雷斯科特的时间一致结果，

$$
\alpha_0 = (1 + \theta^2) U^*, \qquad \alpha_1 = -\theta,
\qquad g \to \theta U^* .
$$

```{code-cell} ipython3
α_consistent = np.array([(1 + θ_pc**2) * U_star, -θ_pc])
g_consistent = θ_pc * U_star
print(f"consistent equilibrium beliefs α = {α_consistent},  target inflation g = {g_consistent}")
print(f"optimal (Ramsey) outcome:                                  target inflation g = 0")
```

这就是**通货膨胀偏误**。

政府以 $\theta U^*$ 的速率制造通货膨胀，却一无所得：因为只有意外部分才会影响失业，无论如何平均失业率都是 $U^*$。

若政府理解了真实模型 {eq}`phillips_true`，它会选择 $g = 0$：得到同样的失业率，却没有任何通货膨胀。

### 恒定系数信念：学习通货膨胀偏误

假设政府相信其系数是恒定的，并用最小二乘法逐期少量地更新它们的估计。

最小二乘学习平均而言所遵循的确定性路径——它的**均值动态**——将信念推向其当前政策所会引致的回归结果。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Least squares learning converges to the inflation bias"
    name: fig-olg-mean-dynamics
---
def induced_beliefs(α):
    "The OLS fit that the current beliefs' policy would induce in a stationary sample."
    g = govt_target(α)
    return np.array([U_star + θ_pc * g, -θ_pc])          # slope is exactly -θ

def mean_dynamics(α0=(0.0, 0.0), lr=0.2, T=60):
    α = np.array(α0, float)
    path = np.empty((T, 3))
    for t in range(T):
        α = α + lr * (induced_beliefs(α) - α)
        path[t] = [α[0], α[1], govt_target(α)]
    return path

path = mean_dynamics()

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(path[:, 2], 'C0', lw=1.6, label="target inflation $g$")
ax.plot(path[:, 0], 'C1', lw=1.2, label=r"belief $\alpha_0$")
ax.plot(path[:, 1], 'C2', lw=1.2, label=r"belief $\alpha_1$")
ax.axhline(g_consistent, color='C0', ls='--', lw=0.8)
ax.set_xlabel("iteration")
ax.legend(frameon=False)
plt.show()
```

最小二乘学习径直把政府带入了一致均衡：信念稳定在 $\alpha = ((1+\theta^2)U^*, -\theta)$，目标通货膨胀稳定在 $\theta U^* = 5$。

政府一期又一期地持续制造通货膨胀，其速率对它毫无好处，因为它自己的政策不断产生的数据恰好证实了它所相信的那种权衡关系。

### 随机系数信念：逃向最优

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 接着提出了一个问题：如果政府*不那么*确信其系数是恒定的，会发生什么？

他们让政府怀疑系数存在漂移——即 $\alpha$ 服从随机游走——并用一个卡尔曼滤波器来估计它们，该滤波器把旧数据的权重折算得比近期数据更低。

这是一种**常增益**算法：它并不像 $1/t$ 增益那样最终会冻结估计值，而是永远保持一个固定的增益，因此政府从不停止对刚发生的事情保持关注。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Constant-gain learning and recurrent escapes to zero inflation"
    name: fig-olg-escapes
---
def constant_gain(T=40_000, gain=0.02, seed=1):
    "Government estimates a drifting-coefficient Phillips curve; returns target inflation g_t."
    rng = np.random.default_rng(seed)
    α = α_consistent.copy()                              # start at the inflation bias
    R = np.eye(2)
    g_path = np.empty(T)
    for t in range(T):
        g = govt_target(α)
        η = σ_η * rng.standard_normal()
        π = g + η
        U = U_star - θ_pc * η + σ_u * rng.standard_normal()   # true Phillips curve
        x = np.array([1.0, π])
        R = R + gain * (np.outer(x, x) - R)
        α = α + gain * np.linalg.solve(R, x * (U - x @ α))
        g_path[t] = g
    return g_path

g_path = constant_gain()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(g_path, lw=0.5)
ax.axhline(g_consistent, color='C3', ls='--', lw=1, label="consistent equilibrium (inflation bias)")
ax.axhline(0, color='C2', ls=':', lw=1, label="Ramsey / zero inflation")
ax.set_xlabel("$t$")
ax.set_ylabel("target inflation $g_t$")
ax.legend(frameon=False)
plt.show()
```

常增益政府并没有停留在通货膨胀偏误上。

它先向该偏误攀升，随后突然**逃逸**朝向零通货膨胀，然后又逐渐上漂，接着再次逃逸，形成一个反复出现的锯齿状模式。

其机制正是西姆斯和钟所指出的那种。

由于政府对旧数据打折扣，一连串通货膨胀意外较小、失业率保持在 $U^*$ 附近而与通货膨胀无关的观测数据，很快就会使它相信那种权衡关系并不存在，于是它停止利用一种它已不再相信的权衡关系，将通货膨胀降至接近零。

这种愿意考虑系数漂移的态度，使政府*无需长期忍受通货膨胀*就能学到自然率假说的真相。

逃逸发生的频率取决于增益——即政府对自身模型的怀疑程度——这恰好正是 {cite:t}`Sims1988` 发现的、用以在两种说法之间做选择的参数。

较小的增益使经济停留在通货膨胀偏误附近；较大的增益则使它更频繁地走向最优（{ref}`olg_ex4`）。

### 从逃逸动态到《美国通货膨胀的征服》

这两种机制正是我们开篇提到的两种说法的形式化版本。

一致均衡路径就是自然率的说法：一个信任其误设模型的政府不断制造通货膨胀并陷于其中。

逃逸路径就是反对的说法：一个对模型变化保持警觉的政府察觉到了糟糕的权衡关系，并从中撤退。

这是一大批文献的源头。

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 表明，该模型不仅可以容纳这两种说法，而且——就钟的情形而言——还可以在战后美国数据上进行*计量经济学估计*，把一个真实的估计程序赋予了模型内部的政府。

他们的工作促使 {cite:t}`Sargent1999` 提出了《美国通货膨胀的征服》中的**逃逸路径**模型，其中从高通胀纳什均衡到拉姆齐结果的反复逃逸，提供了一种关于美国通货膨胀在20世纪80年代初*究竟是如何*被征服的理论。

QuantEcon 讲座 {doc}`phillips_escaping_nash` 和 {doc}`phillips_learning` 全面展开了该模型及其逃逸动态；上面的锯齿图只是对它们所研究内容的初步一瞥。

## 结束语

第一部分表明，一个由适应性主体组成的系统能够找到一个它从未被告知过的理性预期均衡，并使我们看到*环境复杂性*在多大程度上决定着学习速度：一个很少出现的赤字状态被缓慢地学习到，而一个巨大的状态空间则迫使人们采取参数化的捷径，而这可能使均衡完全无法企及。

第二部分表明了更强的结论。

在模型存在两个均衡的地方，学习动态并不仅仅是随便找到其中一个；它们系统性地挑出了那个被理性预期动态所拒绝的均衡，实验室被试的选择也是同样的方向。

这是 {cite:t}`Sargent1993` 中支持把适应性动态视为一种均衡选择机制的最有力的例子。

而布洛克反例则标出了这一论点的限度。

这种选择是关于算法的一个事实，而非一条福利定理，萨金特对由此产生的不安是坦率的：

> 我知道，对实时动态心存怀疑，却又保留由它们所选出的均衡，这在逻辑上是不一致的。我承认，我对第6章所述货币模型中所执行的选择的偏爱，部分源于我先验的信念，即我认为所选出的均衡在我看来是合理的。

菲利普斯曲线的应用从两个方面加深了这一论点。

它表明，*哪种*学习方案能找到好的结果是依赖于模型的：在这里，最小二乘法走向了坏的均衡，而恰恰是那个从未停止怀疑的常增益政府，逃向了好的均衡。

而且它把*增益*变成了值得关注的对象，预示了 {cite:t}`Sargent1999` 后来在 {doc}`phillips_escaping_nash` 和 {doc}`phillips_learning` 中所展开的逃逸动态研究。

{doc}`marimon_mcgrattan_sargent` 把选择这一想法带向了不同的方向，进入了这样一个设定：相互竞争的均衡不是高通胀与低通胀，而是不同的*货币制度*：哪种商品成为货币。

## 练习

```{exercise-start}
:label: olg_ex1
```

由于主体为每一个赤字状态学习一个单独的储蓄率，他们只能以该状态出现的速度来学习该状态。

比较两个仅在转移矩阵上有所不同的经济体，二者都有 $\bar G = (0.8, 0)$：一个经济体中两个状态的可能性相等，另一个经济体中高赤字状态很少出现。

对每一个经济体，计算其平稳均衡，运行学习模拟，并报告每个状态所学到的储蓄率与其目标值相差多远，以及该状态被访问的频率。

```{exercise-end}
```

```{solution-start} olg_ex1
:class: dropdown
```

```{code-cell} ipython3
G_ex = np.array([0.8, 0.0])
cases = {"equally likely": np.array([[0.5, 0.5], [0.5, 0.5]]),
         "high deficit rare": np.array([[0.5, 0.5], [0.03, 0.97]])}

rows = []
for name, P in cases.items():
    s_star, _ = stationary_equilibrium(G_ex, P)
    ergodic = qe.MarkovChain(P).stationary_distributions[0]
    hist, _ = simulate(G_ex, P, T=30_000, s0=start)
    learned = hist[-1].mean(axis=0)
    for k in (0, 1):
        rows.append([name, k+1, ergodic[k], s_star[k], learned[k],
                     abs(learned[k] - s_star[k])])

pd.DataFrame(rows, columns=["chain", "state", "ergodic prob.",
                            "equilibrium $s_i$", "learned $s_i$", "error"]).round(4)
```

当两个状态出现的可能性相等时，两条储蓄规则被学习得同样好。

而当高赤字状态只在约百分之六的时间出现时，尽管两者都已经有30000个日历时期来稳定下来，它的规则误差却比常见状态规则的误差要大一个数量级。

萨金特的评论值得牢记：就*无条件期望效用*而言，未能学会在一个很少被访问的状态下该怎么做，代价可能非常小。

学习缓慢的代价并不与误差的大小成比例。

还要注意，改变转移矩阵会改变均衡本身——关于赤字持续性的信念直接进入 {eq}`olg_equilibrium`——因此该比较必须针对每条链分别重新计算的目标进行。

```{solution-end}
```

```{exercise-start}
:label: olg_ex2
```

增益序列 $\gamma_\tau = 1/\tau$ 正是使 {eq}`robbins_monro` 收敛的原因。

一个怀疑环境可能在漂移的主体，会转而使用**常增益**，永远更重地加权近期经验。

修改该模拟，使其使用常增益，并描述储蓄规则的极限行为会发生什么变化。

将 $0.02$ 和 $0.005$ 的增益与 $1/\tau$ 基准进行比较。

```{exercise-end}
```

```{solution-start} olg_ex2
:class: dropdown
```

```{code-cell} ipython3
def simulate_constant_gain(G, P, gain, T=30_000, s0=None, seed=0, band=0.4):
    rng = np.random.default_rng(seed)
    n = len(G)
    s = np.array(s0, float)
    M = np.array([[U_double(s[j, k], 1.0) for k in range(n)] for j in range(2)])
    floor = G/N + band
    hist = np.empty((T, 2, n))
    i, prev = 0, None
    for t in range(T):
        j = t % 2
        s_t = s[j, i]
        if prev is not None:
            j_p, i_p, s_p = prev
            R = (N*s_t - G[i]) / (N*s_p)
            M[j_p, i_p] += gain * (U_double(s_p, R) - M[j_p, i_p])
            step = gain * U_prime(s_p, R) / M[j_p, i_p]
            s[j_p, i_p] = np.clip(s[j_p, i_p] - step, floor[i_p], w1 - 0.4)
        prev = (j, i, s_t)
        hist[t] = s
        i = rng.choice(n, p=P[i])
    return hist

rows = []
for label, hist in [("gain = 0.02", simulate_constant_gain(G2, P2, 0.02, s0=start)),
                    ("gain = 0.005", simulate_constant_gain(G2, P2, 0.005, s0=start)),
                    ("gain = 1/τ", simulate(G2, P2, T=30_000, s0=start)[0])]:
    tail = hist[-5_000:].mean(axis=1)        # average the two classes
    rows.append([label, *tail.mean(axis=0), *tail.std(axis=0)])

pd.DataFrame(rows, columns=["", "mean $s_1$", "mean $s_2$",
                            "s.d. $s_1$", "s.d. $s_2$"]).set_index("").round(5)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
for gain, colour in [(0.02, 'C1'), (0.005, 'C2')]:
    h = simulate_constant_gain(G2, P2, gain, s0=start)
    ax.plot(h[:, 0, 0], color=colour, lw=0.7, label=f"gain = {gain}")
h, _ = simulate(G2, P2, T=30_000, s0=start)
ax.plot(h[:, 0, 0], color='C0', lw=1.2, label=r"gain = $1/\tau$")
ax.axhline(s2[0], color='k', lw=0.8, ls='--', label="equilibrium")
ax.set_xlim(0, 30_000)
ax.set_ylim(3.8, 5.0)
ax.set_xlabel("$t$")
ax.set_ylabel("saving rate, state 1")
ax.legend(frameon=False, fontsize=9)
plt.show()
```

常增益规则并不收敛。

它们会稳定在均衡的一个*邻域*内，然后在该邻域内不断摇摆，永不停歇，因为每一个新的观测都始终被赋予相同的权重，因此从不停止移动估计值。

减小增益会收窄该区间，但不会消除它。

只有一个递减到零的增益——例如 $1/\tau$——才能带来向一点的收敛，这正是为什么这些系统的所有收敛定理都要求它。

常增益主体以此换来的是适应性：如果赤字过程发生了变化，$1/\tau$ 主体几乎不会察觉，而常增益主体则会追踪这种变化。

这种权衡关系在这一文献中反复出现。

```{solution-end}
```

```{exercise-start}
:label: olg_ex3
```

关于高通胀稳态处在"拉弗曲线错误一侧"的说法可以被精确化。

在总通货膨胀率为 $\pi$ 的稳态均衡中，实际余额为 $h = (w_1 - w_2\pi)/2$，政府征收铸币税 $h(1 - 1/\pi)$。

绘制铸币税作为 $\pi$ 的函数的图像，标出两个稳态均衡，并利用该图解释为什么提高 $G$ 会提高低稳态通货膨胀率，却*降低*高稳态通货膨胀率，以及当 $G$ 过大时会发生什么。

```{exercise-end}
```

```{solution-start} olg_ex3
:class: dropdown
```

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

两个稳态均衡正是通货膨胀税拉弗曲线与所需收入 $G$ 相交的两个通货膨胀率，各自位于曲线峰值的一侧。

低稳态处于上升段，因此在那里为更大的赤字融资需要*更多*的通货膨胀：这是古典的比较静态。

高稳态处于下降段，在那里更多的通货膨胀带来*更少*的收入，因此更大的赤字要靠*更低*的稳态通货膨胀率来融资，这种反古典的比较静态正是高均衡如此棘手的原因。

随着 $G$ 上升，水平线上移，两个交点趋于收敛。

超过曲线峰值之后，就根本不存在稳态均衡：赤字超过了通货膨胀税所能征收的最大收入。

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

随着 $G$ 上升，两个根相互靠拢，并在 $G \approx 1.0505$ 处相撞，此后模型就根本没有稳态均衡了。

这个数字并非巧合。

将 {eq}`inflation_map` 不动点方程的判别式设为零，得到 $\pi = \sqrt{A_2} = \sqrt{w_1/w_2}$ 处的一个二重根，而该处所对应的赤字恰好就是上面计算出的拉弗曲线的峰值。

```{code-cell} ipython3
G_max = (w1_d + w2_d - 2*np.sqrt(A2)*w2_d) / 2
print(f"roots collide at G       = {G_max:.6f},  where π = √(w1/w2) = {np.sqrt(A2):.6f}")
print(f"peak of the Laffer curve = {seigniorage(np.sqrt(A2)):.6f}  at π = {np.sqrt(A2):.6f}")
```

政府所能融资的最大赤字，就是通货膨胀税的最大值，而在该赤字水平下，两个稳态均衡合并为一个。

```{solution-end}
```

```{exercise-start}
:label: olg_ex4
```

在菲利普斯曲线应用中，常增益政府逃向零通货膨胀，而讲座中提出，*逃逸的频繁程度*取决于增益，即政府对自身模型的怀疑程度。

请把这一论断量化。

对于一系列增益值，模拟常增益经济，并测量平均目标通货膨胀以及处于接近拉姆齐结果（比如说，$g_t < 1$）的时间比例。

更多的怀疑会把经济推向哪个方向？这与关于战后通货膨胀的两种说法有何联系？

```{exercise-end}
```

```{solution-start} olg_ex4
:class: dropdown
```

```{code-cell} ipython3
rows = []
for gain in (0.005, 0.01, 0.02, 0.04, 0.08):
    tails = [constant_gain(T=30_000, gain=gain, seed=s)[-20_000:] for s in range(4)]
    mean_g = np.mean([g.mean() for g in tails])
    near_ramsey = np.mean([np.mean(g < 1) for g in tails])
    rows.append([gain, mean_g, near_ramsey])

pd.DataFrame(rows, columns=["gain", "mean target inflation", "fraction of time $g < 1$"]
             ).set_index("gain").round(3)
```

更多的怀疑——更大的增益——把经济从通货膨胀偏误推开，并推向最优：平均通货膨胀下降，经济花在接近拉姆齐结果附近的时间比例更大。

其直觉是：更高的增益对旧数据打了更重的折扣，因此政府对显示该权衡关系是虚幻的那一连串观测数据反应更快，也就更早、更频繁地逃逸。

这恰好正是 {cite:t}`Sims1988` 发现的、用以在两种说法之间做选择的参数。

一个对其恒定系数模型抱有信心的政府（小增益）是自然率说法中的政府，深陷于制造通货膨胀之中；而一个对模型变化保持警觉的政府（大增益）则是反对说法中的政府，能察觉到糟糕的权衡关系并从中退出。

同样的增益在 {cite:t}`Sargent1999` 中作为核心对象再次出现，其中从高通胀均衡*逃逸的速度*决定了一场通货膨胀被征服的快慢。

```{solution-end}
```