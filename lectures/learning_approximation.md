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
  title: 学习、近似与均衡计算
  headings:
    Overview: 概述
    A continuous-state functional equation: 一个连续状态泛函方程
    A continuous-state functional equation::The deficit process: 赤字过程
    A benchmark solution: 一个基准解
    A benchmark solution::Which equilibrium?: 是哪一个均衡？
    Parameterized expectations: 参数化预期
    Parameterized expectations::Approximate versus exact equilibria: 近似均衡与精确均衡
    Learning is equilibrium computation: 学习就是均衡计算
    Learning is equilibrium computation::Same limit, different speeds: 相同的极限，不同的速度
    Learning without a functional form: 没有函数形式的学习
    Concluding remarks: 结束语
    Exercises: 练习
---

(learning_approximation)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 学习、近似与均衡计算

```{index} single: Bounded Rationality; Parameterized Expectations
```

```{contents} Contents
:depth: 2
```

除了 Anaconda 中已有的库外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概述

在 {doc}`olg_adaptive_money` 中，一个适应性主体系统找到了一个它从未被告知过的理性预期均衡。

那里的主体为政府赤字的每一个水平学习了一个*独立的*储蓄率——一个**非参数**规则，每个状态对应一个数字。

当只有两个状态时，这种方法是有效的。

但在其他情况下，它会遇到两个障碍，二者都在 {cite:t}`Sargent1993` 中被提及：

* 一个很少出现的状态学习得很慢，因为关于它的观测值到来得很慢；
* 当状态数量很大时——尤其是当状态是*连续的*时——每个状态一个参数的做法是无望的。

计量经济学家的应对方法是对决策规则施加**参数**形式，$s = f(G, \theta)$，其中 $\theta$ 是低维的，并利用每一个观测值来估计 $\theta$。

本讲座将详细探讨这样做的收益和代价。

代价与收益是同一个想法的两面。

一个参数族可以被快速学习，因为每一个观测值都能为每一个状态提供信息。

但是，一个局限于某个函数族的学习方案，只有当该族中存在某个成员支持一个理性预期均衡时，才能收敛到该均衡。

否则，它所能达到的最好结果只是一个**近似均衡**。

将这个问题梳理清楚，会呈现出贯穿整个有限理性研究项目的一个主题：模拟学习过程的算法与计算均衡的算法彼此相似。

**马塞特参数化预期方法** {cite}`denHaanMarcet1990,MarcetMarshall1992` 是一种用于*计算*理性预期均衡的标准工具。

当以递归形式书写时，它就变成了一个适应性主体实时学习的模型。

一次均衡计算，是建模者运行的一个集中式学习算法；一个学习型经济，则是主体们运行的一个去中心化的均衡计算。

这两者是不同的递归，但它们共享相同的不动点，并且通过一个二者都遵循的微分方程，共享相同的稳定性条件。

计划如下：

1. 在 {doc}`olg_adaptive_money` 的世代交叠模型中引入一个取许多值的赤字——一个连续过程的精细离散化，将均衡条件转化为储蓄规则 $f(G)$ 中的一个**泛函方程**。
1. 计算一个基准解，以便我们有可以据以衡量的真实值。
1. 用日益丰富的函数族通过**参数化预期**来求解它，观察近似均衡如何逐步让位于精确均衡。
1. 说明同一算法的递归、实时版本**就是**一个学习型经济，并且 $T$ 映射的一个特征值决定了它学习的速度。
1. 用核估计器**非参数地**学习储蓄规则，并权衡它与参数化路径之间的取舍。

让我们从一些导入开始。

```{code-cell} ipython3
import warnings
import numpy as np
import pandas as pd
import quantecon as qe
import matplotlib.pyplot as plt
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
mpl.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
```

## 一个连续状态泛函方程

我们保留 {doc}`olg_adaptive_money` 中的世代交叠货币经济——两期存活的主体、对数效用、年轻时的禀赋 $w_1$ 和年老时的禀赋 $w_2$、作为唯一价值储藏手段的货币，以及一个通过印钞为赤字融资的政府。

我们还保留该讲座第一部分的校准：$w_1 = 20$、$w_2 = 10$ 以及 $N = 1$。

唯一的变化是：赤字 $G_t$ 现在遵循一个**连续**马尔可夫过程，其转移核为 $F(G', G) = \operatorname{Prob}\{G_{t+1} \leq G' \mid G_t = G\}$。

我们寻找一个平稳均衡，其中储蓄是当前赤字的一个*函数*，$s_t = f(G_t)$。

家庭的一阶条件，在 $s_t = f(G_t)$ 处求值，为

$$
u'(w_1 - f(G_t)) = \mathbb{E}_t\bigl[u'(w_2 + f(G_t) R_t)\, R_t\bigr],
\qquad \mathbb{E}_t(\cdot) = \mathbb{E}(\cdot \mid G_t),
$$

而政府预算约束与市场出清一起，给出了以储蓄规则表示的货币回报率，与之前完全一样（当 $N = 1$ 时），

$$
R_t = \frac{f(G_{t+1}) - G_{t+1}}{f(G_t)} .
$$

代入 $R_t$，并利用 $f(G_t) R_t = f(G_{t+1}) - G_{t+1}$ 表示老年期的消费，将一阶条件转化为关于 $f$ 的**泛函方程**：

```{math}
:label: functional_equation

u'\bigl(w_1 - f(G_t)\bigr)
= \mathbb{E}_t\!\left[
u'\!\bigl(w_2 + f(G_{t+1}) - G_{t+1}\bigr)
\cdot \frac{f(G_{t+1}) - G_{t+1}}{f(G_t)}
\right].
```

由于 $f(G_t)$ 在时刻 $t$ 已知，无论效用函数为何，它都可以提到期望符号之外。

写成

```{math}
:label: psi_definition

\psi(G_t) \equiv \mathbb{E}_t\bigl[u'(w_2 + f(G_{t+1}) - G_{t+1})\,(f(G_{t+1}) - G_{t+1})\bigr],
```

方程 {eq}`functional_equation` 变为 $u'(w_1 - f(G_t)) = \psi(G_t)/f(G_t)$。

在对数效用下，$u'(c) = 1/c$，这可以解出储蓄的闭合形式：

$$
\frac{1}{w_1 - f(G_t)} = \frac{\psi(G_t)}{f(G_t)},
\qquad\text{即}\qquad
f(G_t) = \frac{w_1\,\psi(G_t)}{1 + \psi(G_t)} .
$$

所以整个问题就归结为求出**条件期望** $\psi(G)$：一旦我们求得它，储蓄规则就以闭合形式得出。

这个观察是整个讲座的枢纽。

下面的每一种方法——无论是参数的还是非参数的——都是估计同一个对象 $\psi(G)$ 的一种方式。

在对数效用下，{eq}`psi_definition` 中的随机变量是
$y = (s' - G')/(w_2 + s' - G')$，其中 $s'$ 和 $G'$ 分别是下一期的储蓄和赤字。

我们将其称为**回归量**，原因很快就会清楚。

```{code-cell} ipython3
w1, w2 = 20.0, 10.0

def s_from_psi(ψ):
    "从条件期望 ψ 中恢复储蓄规则。"
    return w1 * ψ / (1 + ψ)

def regressand(s, G):
    "在对数效用下，给定储蓄 s 和赤字 G 时的 u'(c_2) (s - G)。"
    return (s - G) / (w2 + s - G)
```

### 赤字过程

我们假设 $\log G_t$ 服从一个 AR(1) 过程，
$\log G_{t+1} = (1-\rho)\log \bar G + \rho \log G_t + \sigma \varepsilon_{t+1}$。

为了对其进行计算，我们用一个精细的马尔可夫链来代替它，使用 {cite:t}`Tauchen1986` 的方法将其离散化为 201 个网格点。

从这里开始，这条链*就是*这个经济体：下面的基准解正是在这条链上计算的，每一个学习经济体也都是从这条链上模拟出来的。

对两者都使用同一条链，能让比较保持公允。

如果学习者生活在 AR(1) 过程中，而基准解却是在该链上计算的，那么即使一个完美求解了自身经济体的学习者，也无法达到基准水平，而这种差距会被错误地归咎于学习者本身。

有了 201 个状态，这条链就是连续过程的一个很好的替代。

这也正是一个足以让类似 {doc}`olg_adaptive_money` 中那种逐状态学习者失效的情形：这样的学习者需要 201 个独立的储蓄率，而最罕见的状态大约每两千期才被访问一次。

```{code-cell} ipython3
ρ, σ, G_bar = 0.7, 0.30, 0.6
n_states = 201

mc = qe.tauchen(n_states, ρ, σ, mu=np.log(G_bar) * (1 - ρ), n_std=2.5)
G, P = np.exp(mc.state_values), mc.P
ergodic = mc.stationary_distributions[0]      # 赤字大部分时间所处的位置

def simulate_deficit(T, seed=0, P=P):
    "一条从其遍历分布出发模拟得到的赤字链路径的状态索引。"
    rng = np.random.default_rng(seed)
    chain = qe.MarkovChain(P)
    init = rng.choice(len(P), p=chain.stationary_distributions[0])
    return chain.simulate_indices(T, init=init, random_state=rng)

G_perm = (np.sqrt(w1) - np.sqrt(w2))**2 / 2   # 能够永远被融资的最大赤字

print(f"赤字网格：{n_states} 个点，从 {G.min():.3f} 到 {G.max():.3f}")
print(f"遍历分布下的平均赤字 E[G]：{ergodic @ G:.3f}")
print(f"访问频率最低状态的概率：{ergodic.min():.1e}")
print(f"存在货币稳态的最大恒定赤字：{G_perm:.3f}")
print(f"网格上限相对于该值的比例：{G.max() / G_perm:.2f}")
print(f"赤字高于该值的时间占比：{ergodic[G > G_perm].sum():.2f}")
```

这个经济体能够为多大的赤字融资？

当赤字固定在 $G$ 时，一个货币稳态要满足一个二次方程（{ref}`lae_ex2`），只有当 $G \leq (\sqrt{w_1} - \sqrt{w_2})^2/2 \approx 0.858$ 时才有实数解。

我们网格的上限是该值的两倍，而赤字大约有五分之一的时间处于该值之上。

然而每个状态下都仍然存在货币均衡，因为高赤字只是暂时的。

当 $\rho = 0.7$ 时，处于高赤字状态的年轻主体会预期赤字——连同其储蓄所承受的通货膨胀税——会回落至其均值附近。

## 一个基准解

为了衡量任何学习方案，我们首先需要真实的 $f(G)$。

在网格上，{eq}`functional_equation` 是一个包含201个方程的方程组，我们通过**迭代感知到实际的映射**来求解它，这正是 {doc}`bounded_rationality` 中的 $T$ 映射。

猜测一个储蓄规则；用它代入右侧计算每个状态实际会引发的储蓄；重复直到两者一致。

在该链上，右侧是一个矩阵-向量乘积，$\psi = P y$，其中 $y_j = (s_j - G_j)/(w_2 + s_j - G_j)$，然后由闭式解给出新的储蓄规则。

这是 {doc}`os_time_iter` 中时间迭代的一种形式，新猜测上的阻尼权重相当于 {doc}`bounded_rationality` 中松弛参数 $\lambda$ 的角色。

```{code-cell} ipython3
def T_map(s):
    "由一个感知规则 s（每个状态一个条目）所引发的实际储蓄规则。"
    return s_from_psi(P @ regressand(s, G))

def solve_benchmark(damp=0.5, tol=1e-12, max_iter=5000):
    "赤字网格上感知到实际的储蓄映射的不动点。"
    s = np.full(n_states, (w1 - w2) / 2)
    for it in range(1, max_iter + 1):
        s_new = T_map(s)
        if np.max(np.abs(s_new - s)) < tol:
            return s_new, it
        s = damp * s_new + (1 - damp) * s
    warnings.warn("the benchmark iteration did not converge")
    return s, it


s_benchmark, iters = solve_benchmark()
print(f"经过 {iters} 次迭代收敛")


def foc_residual(s_rule):
    "在真实核 P 下，一个储蓄规则的最大 |一阶条件误差|。"
    return np.max(np.abs(1 / (w1 - s_rule) - (P @ regressand(s_rule, G)) / s_rule))


print(f"基准一阶条件残差：{foc_residual(s_benchmark):.2e}")
print(f"储蓄规则 f(G) 从 {s_benchmark.max():.3f}（低赤字）"
      f"变化到 {s_benchmark.min():.3f}（高赤字）")
```

下面是这个规则的图示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "基准均衡储蓄规则"
    name: fig-la-benchmark
---
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(G, s_benchmark, 'k-', lw=2)
ax.set_xlabel("赤字 $G$")
ax.set_ylabel("储蓄 $f(G)$")
plt.show()
```

随着赤字上升，储蓄下降：更大的赤字意味着更快的货币创造、更差的货币回报率，以及更少的储蓄。

该规则呈现平缓的曲线，这一点稍后会很重要。

这个 $f(G)$ 是任何适应性主体都无法看到的对象。

下面的一切都在尝试恢复它。

### 是哪一个均衡？

{cite:t}`Sargent1993` 指出，像 {eq}`functional_equation` 这样的系统通常有不止一个解。

我们的迭代从 $s = 5$ 开始，这是无赤字经济体的储蓄率，并找到了一个高储蓄、低通胀的解。

从附近任何一个起点出发都会得到相同的解，因为该解在 **$T$ 映射下是稳定的**：在该点处映射雅可比矩阵的每一个特征值都小于一。

```{code-cell} ipython3
def jacobian(f, x, eps=1e-6):
    "映射 f 在点 x 处的数值雅可比矩阵。"
    return np.column_stack([(f(x + eps * e) - f(x - eps * e)) / (2 * eps)
                            for e in np.eye(len(x))])

eig_benchmark = np.linalg.eigvals(jacobian(T_map, s_benchmark))
print(f"最大 |虚部|：{np.abs(eig_benchmark.imag).max():.1e}")
print("基准点处 DT 的最大特征值：",
      np.sort(eig_benchmark.real)[::-1][:4].round(3))
```

这就是 {ref}`br_one_recursion` 中的 E-稳定性条件：在该条件下，信念被拉向 $T$ 映射输出值的学习者会在某个均衡上稳定下来。

具有恒定赤字的经济体清楚地说明了这一点。

它有两个稳态，而在高通胀的那个稳态处，$T$ 映射的斜率超过一（{ref}`lae_ex2`），因此无论是迭代过程还是学习者都无法稳定在那里。

这正是 {ref}`olg_reversal` 中稳定性反转的计算层面体现，在那里最小二乘学习者会选择低通胀稳态。

上面打印出的最大特征值，约为0.77，还会再次出现：它决定了下面算法的收敛速度。

(lae_pea)=
## 参数化预期

马塞特的**参数化预期方法** {cite}`denHaanMarcet1990,MarcetMarshall1992` 通过对 {eq}`psi_definition` 中的条件期望 $\psi(G)$ 施加一个参数形式来处理 {eq}`functional_equation`，

$$
\psi(G) \approx \psi(G, \theta) = \phi(G)^\top \theta ,
$$

其中 $\phi(G)$ 是一个基函数向量，$\theta$ 是一个短的系数向量。

该算法是关于 $\theta$ 的一个不动点迭代：

1. 给定 $\theta$，储蓄规则为 $f(G) = w_1 \psi(G,\theta) / (1 + \psi(G,\theta))$。
1. 模拟一条较长的赤字路径，并计算每期蕴含的储蓄。
1. 形成 {eq}`psi_definition` 中对象的*已实现*值，
   $y_t = u'(w_2 + s_{t+1} - G_{t+1})(s_{t+1} - G_{t+1})$，并将其对 $\phi(G_t)$ 进行**回归**以得到新的 $\theta$。
1. 迭代直至收敛。

步骤 3 中的回归所做的正是条件期望的工作：最小二乘法将已实现的 $y_t$ 投影到当前状态的函数上，这正是 $\mathbb{E}_t[\cdot \mid G_t]$ 所做的事情。

我们使用重新标度的 $\log G$ 中的单项式作为基函数，并改变其阶数。

本讲中的每个算法都从相同的初始信念 $\psi = 1/3$ 出发，在该信念下主体储蓄 $s = 5$，就好像不存在赤字一样。

这些函数以赤字水平、转移矩阵以及在每个状态下求值的基函数作为参数，以便它们可以在其他经济体中重复使用。

由于回归量在各轮迭代之间不会改变，我们只需在网格上对基函数求值并对回归量矩阵求逆一次。

```{code-cell} ipython3
def basis(G_vals, degree):
    "重新标度到 [-1, 1] 的对数赤字的单项式。"
    x = 2 * (np.log(G_vals) - np.log(G.min())) / (np.log(G.max()) - np.log(G.min())) - 1
    return np.vstack([x**k for k in range(degree + 1)]).T

ψ_init = 1 / 3                     # 主体储蓄 s = 5 所对应的信念

def θ_init(k):
    "初始系数：常数 ψ = ψ_init。"
    θ = np.zeros(k)
    θ[0] = ψ_init
    return θ

def rule(Φ, θ, G=G):
    "由系数 θ 蕴含的每个状态下的储蓄规则，保持可行性。"
    ψ = np.clip(Φ @ θ, 0.05, 50)
    return np.clip(s_from_psi(ψ), G + 0.05, w1 - 0.05)

def errors(s_rule):
    "与基准解之间的 sup 范数距离和遍历加权 RMS 距离。"
    d = s_rule - s_benchmark
    return np.max(np.abs(d)), np.sqrt(ergodic @ d**2)

def parameterized_expectations(Φ, G=G, P=P, T=150_000, seed=0,
                               damp=0.6, tol=1e-11, max_iter=1000):
    """
    批量参数化预期。Φ 保存了在每个赤字状态下求值的基函数，每行对应一个状态。
    返回每个状态下蕴含的储蓄规则、系数以及迭代轮数。
    """
    idx = simulate_deficit(T + 1, seed, P)
    X = Φ[idx[:-1]]                          # 沿历史路径的回归量 φ(G_t)
    X_pinv = np.linalg.pinv(X)               # 它们不会改变，因此只需求逆一次
    θ = θ_init(Φ.shape[1])
    for it in range(1, max_iter + 1):
        y = regressand(rule(Φ, θ, G), G)[idx[1:]]     # 已实现的 y_t
        θ_new = X_pinv @ y                            # 将 y_t 对 φ(G_t) 回归
        if np.max(np.abs(θ_new - θ)) < tol:
            return rule(Φ, θ_new, G), θ_new, it
        θ = (1 - damp) * θ + damp * θ_new
    warnings.warn("parameterized expectations did not converge")
    return rule(Φ, θ, G), θ, it
```

### 近似均衡与精确均衡

现在对日益丰富的函数族运行该算法：常数 $\psi$（储蓄与赤字无关）、线性的，以及二次的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "阶数递增的参数化预期规则"
    name: fig-la-pea
---
pea = {deg: parameterized_expectations(basis(G, deg)) for deg in (0, 1, 2)}
rules = {deg: pea[deg][0] for deg in pea}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(G, s_benchmark, 'k-', lw=2.2, label="基准 $f(G)$")
labels = {0: "0 阶（常数）", 1: "1 阶（线性）", 2: "2 阶（二次）"}
for deg, colour in [(0, 'C0'), (1, 'C1'), (2, 'C2')]:
    ax.plot(G, rules[deg], '--', color=colour, lw=1.4, label=labels[deg])
ax.set_xlabel("赤字 $G$")
ax.set_ylabel("储蓄 $s$")
ax.legend(frameon=False)
plt.show()
```

**常数**族根本无法表示一个依赖于赤字的储蓄规则，因此它所能做到的最好效果就是一条穿过数据中部的水平线。

这条水平线*确实是*一个近似均衡——学习方案的一个不动点——但它只是平均意义上满足一阶条件，而非逐状态满足。

用 {doc}`phillips_misspecified` 中的语言来说，这是一种预测设定误差下的均衡：函数形式是错误的，但其中的系数却是主体自身生成的数据所能允许的最佳系数。

**线性**族的表现要好得多，但在两端出现偏差：其规则在赤字范围的两端都高于基准解，而在中间则低于基准解。

**二次**族几乎与基准解无法区分：$\psi$ 在这里非常接近于 $\log G$ 的二次函数——而 $\log G$ 正是我们构建基函数所依据的变量——因此一个三参数的规则基本上完美地捕捉到了它。

我们可以用两种方式来量化"有多接近"：与基准解的距离，以及在真实核下一阶条件的残差——后者是衡量一个近似均衡与精确均衡有多远的诚实指标。

```{code-cell} ipython3
rows = []
for deg in (0, 1, 2):
    s_hat, _, passes = pea[deg]
    rows.append([labels[deg], *errors(s_hat), foc_residual(s_hat), passes])

pd.DataFrame(rows, columns=["族", "sup $|s - f|$", "遍历 RMS",
                            "一阶条件残差", "迭代轮数"]).set_index("族").round(4)
```

随着函数族变得更加丰富，每一个误差列都在下降。

一阶条件残差——这个量在理性预期均衡处恰好为零，而在近似均衡处为正——从常数族到二次族下降了超过一个数量级。

{cite:t}`MarcetMarshall1992` 给出了使这并非巧合的条件：随着函数族变得更丰富，参数化预期计算出的近似均衡会收敛到理性预期均衡。

最后一列统计了算法收敛所需的迭代轮数：远超过一百轮，原因我们稍后会谈到。

```{note}
在有限数据下，更丰富并*不总是*更好。

随着阶数上升，近似误差持续下降，但回归中的抽样误差却不会随之下降，超过二次阶后抽样误差便占据主导。

此后多余的项就会追逐赤字分布尾部的抽样噪声——那里观测值稀少：遍历加权误差趋于平稳，而 sup 范数误差则可能重新上升，其幅度取决于所模拟的具体历史路径。

这与 {doc}`olg_adaptive_money` 中缓慢学习的稀有状态是同一现象，只是从近似的角度来看：一个方案在数据所在之处拟合得很好，而对一个很少被访问的区域拟合不佳，在预期效用意义上代价很小。

{ref}`lae_ex1` 将这两种误差来源区分开来。
```

## 学习就是均衡计算

到目前为止，参数化预期是*我们*运行来计算均衡的算法。

现在到了整个讲座的关键点。

将同一个算法以**递归**形式书写——每期随着一个新观测值的到来更新一次 $\theta$，而不是对整个模拟历史重新回归——它就变成了一个*适应性主体实时学习*的模型。

递归形式就是普通的递归最小二乘法：

```{math}
:label: recursive_pea

\begin{aligned}
M_{t+1} &= M_t + \gamma_t\bigl(\phi(G_t)\phi(G_t)^\top - M_t\bigr), \\
\theta_{t+1} &= \theta_t + \gamma_t M_{t+1}^{-1} \phi(G_t)\bigl(y_t - \phi(G_t)^\top \theta_t\bigr),
\end{aligned}
```

其中 $y_t$ 与之前相同，是已实现的被回归量，$M_t$ 估计回归量的二阶矩矩阵，增益 $\gamma_t$ 以 $1/t$ 的速度递减。

在下面的代码中，$\gamma_t = 1/(t+1)$，而 $M$ 从单位矩阵的一个小倍数开始，这样它从第一期起就可以求逆。

这是一个具有递减增益的随机逼近递归，与 {doc}`olg_adaptive_money` 中的罗宾斯-门罗学习者属于同一类，其中 $M$ 扮演着它们曲率估计量的角色。

它与那里的方案有两点不同，而非一点。

* **学习的对象。** {doc}`olg_adaptive_money` 中的主体直接学习一个储蓄率，对*已实现的终生效用*采取牛顿步。一个主体的终生效用只有在他年老时才可知，因此那里的人群被分成奇数代和偶数代轮流进行。而这里的主体学习的是一个*条件期望* $\psi$，并在给定该期望的情况下求解自己的欧拉方程——{cite:t}`Sargent1993` 称之为通过模型构建来学习，而非笨拙地学习。每一期已实现的被回归量在一期之后就已知，即下一代储蓄完成之时，因此不需要奇偶分代。
* **一个观测值能触及多远。** $\theta$ 索引的是一个*参数化*规则，因此单个观测值可以同时更新每个状态下的储蓄规则，而不仅仅是被访问到的那个状态。

下面是这个学习型经济。

```{code-cell} ipython3
def online_pea(Φ, G=G, P=P, T=500_000, seed=0, ridge=1e-3, a=1.0):
    """
    递归最小二乘参数化预期：学习型经济。

    时刻 t 的年轻主体根据当前的 θ 进行储蓄。他们的储蓄揭示了
    t-1 期已实现的被回归量，θ 和二阶矩矩阵 M 随即被更新一次，
    增益为 a / (t + a)。返回 θ 的路径。
    """
    idx = simulate_deficit(T, seed, P)
    k = Φ.shape[1]
    θ, M = θ_init(k), ridge * np.eye(k)
    θ_path = np.empty((T, k))
    θ_path[0] = θ
    for t in range(1, T):
        g, x = G[idx[t]], Φ[idx[t - 1]]
        ψ = min(max(Φ[idx[t]] @ θ, 0.05), 50.0)
        s = min(max(s_from_psi(ψ), g + 0.05), w1 - 0.05)    # t 期的储蓄
        y = regressand(s, g)                                 # t-1 期的 y，现已知
        γ = a / (t + a)
        M += γ * (np.outer(x, x) - M)
        θ = θ + γ * np.linalg.solve(M, x * (y - x @ θ))
        θ_path[t] = θ
    return θ_path


Φ2 = basis(G, 2)
θ_path = online_pea(Φ2)
s_online = rule(Φ2, θ_path[-1])
print("经过 {:,} 期后的在线 PEA：sup 误差 = {:.4f}，  遍历 RMS = {:.4f}"
      .format(len(θ_path), *errors(s_online)))
```

我们应该如何评判这一点？

学习者的目标并非基准解，而是二次族的近似均衡，即批量算法所计算出的不动点。

我们可以像拥有无限多数据的批量遍历那样，精确地计算出这个目标。

设 $T(\theta)$ 为当主体信念为 $\theta$ 时，$y_t$ 在 $\phi(G_t)$ 上的总体最小二乘投影系数。

这就是 {doc}`bounded_rationality` 中针对局限于该族内的信念所定义的 $T$ 映射，在链上我们可以用 $P$ 来精确计算条件期望，用遍历分布来计算投影。

```{code-cell} ipython3
def T_pea(θ, Φ, G=G, P=P, π=ergodic):
    "一次批量遍历的总体版本：将 E[y_t | G_t] 投影到基底上。"
    Ey = P @ regressand(rule(Φ, θ, G), G)
    return np.linalg.solve(Φ.T @ (π[:, None] * Φ), Φ.T @ (π * Ey))

def approximate_equilibrium(Φ, G=G, P=P, π=ergodic, damp=0.6, tol=1e-13, max_iter=10_000):
    "具有基底 Φ 的该族的 T 映射的不动点 θ*。"
    θ = θ_init(Φ.shape[1])
    for _ in range(max_iter):
        θ_new = T_pea(θ, Φ, G, P, π)
        if np.max(np.abs(θ_new - θ)) < tol:
            return θ_new
        θ = (1 - damp) * θ + damp * θ_new
    warnings.warn("approximate_equilibrium 未能收敛")
    return θ


θ_star = approximate_equilibrium(Φ2)
s_star = rule(Φ2, θ_star)
eig_pea = np.sort(np.linalg.eigvals(jacobian(lambda θ: T_pea(θ, Φ2), θ_star)).real)[::-1]
λ_max = eig_pea[0]

print("θ*（近似均衡）：", θ_star.round(4))
print("θ（在线，最终值）：", θ_path[-1].round(4))
print("θ（批量，T = 150,000）：", pea[2][1].round(4))
print("近似均衡：sup 误差 = {:.4f}，  遍历 RMS = {:.4f}"
      .format(*errors(s_star)))
print("DT(θ*) 的特征值：", eig_pea.round(3))
```

$DT(\theta^*)$ 的特征值全都小于 1，并且它们都接近于之前计算出的基准 $T$ 映射的三个最大特征值：二次族继承了基准解的缓慢收敛方向。

下面是这个学习型经济的运作情况。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "储蓄规则的在线学习"
    name: fig-la-online
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

t_grid = np.arange(1, len(θ_path) + 1)
for j in range(3):
    axes[0].plot(t_grid, θ_path[:, j], color=f'C{j}', lw=2, label=rf"$\theta_{j}$")
    axes[0].axhline(θ_star[j], color=f'C{j}', lw=0.8, ls=':')
axes[0].set_xscale('log')
axes[0].set_xlabel("$t$（对数刻度）")
axes[0].set_ylabel(r"$\theta_t$")
axes[0].set_title(r"实时学习的系数（$\theta^*$ 为点线）")
axes[0].legend(frameon=False, fontsize=9)

axes[1].plot(G, s_benchmark, 'k-', lw=2.2, label="基准 $f(G)$")
axes[1].plot(G, s_star, 'C2:', lw=2, label="近似均衡")
axes[1].plot(G, s_online, 'C3--', lw=1.5, label="在线 PEA")
axes[1].set_xlabel("赤字 $G$")
axes[1].set_ylabel("储蓄 $s$")
axes[1].set_title("学习后的储蓄规则")
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
```

系数朝着 $\theta^*$ 靠拢，但速度缓慢：经过五十万期之后，在线规则在高赤字水平下仍明显高于近似均衡，并且其偏离基准解的程度比批量规则更大。

### 相同的极限，不同的速度

为什么这两个算法共享同一个极限，而在线算法又为何如此缓慢？

一次阻尼系数为 $\delta = 0.6$ 的批量遍历为
$\theta_k = \theta_{k-1} + \delta\,[\hat T(\theta_{k-1}) - \theta_{k-1}]$，其中 $\hat T$ 是 $T$ 的样本版本。

这正是 {doc}`bounded_rationality` 中的松弛方案，其中 $\delta$ 扮演着其松弛参数 $\lambda$ 的角色。

在线递归式 {eq}`recursive_pea` 是一个随机逼近算法。

关于此类算法的理论 {cite}`Ljung1977,MarcetSargent1989jet,Woodford1990`，后来由 {cite:t}`EvansHonkapohja2001` 系统化并推广，指出：在像 $1/t$ 这样的增益下，信念 $\theta_t$ 平均而言遵循如下常微分方程

```{math}
:label: pea_ode

\frac{d\theta}{d\tau} = T(\theta) - \theta
```

其中 $\tau = \log t$ 为重新标度的时间，而 $M_t$ 则趋于稳定在回归量的二阶矩矩阵上。

{ref}`br_one_recursion` 给出了一般性的论证，{doc}`ls_learning` 则针对线性模型详细推演了这一过程；{cite:t}`Sargent1993` 正是用本讲座中的这个经济体来加以说明的。

由此可以得出三个结论。

* **相同的不动点。** {eq}`pea_ode` 的静止点正是 $T$ 的不动点，而批量算法在抽样误差范围内计算出的就是它们。两种方案都指向同一个近似均衡 $\theta^*$。
* **相同的稳定性条件。** 两者只有在 $DT(\theta^*)$ 的每个特征值的实部都小于 1（即 E-稳定性条件）时才收敛到 $\theta^*$。对于在线算法而言，这正是 {eq}`pea_ode` 的稳定性条件。对于批量算法而言，其迭代矩阵 $I + \delta\,(DT - I)$ 的特征值只有在该条件满足时才位于单位圆内，而且对于足够小的 $\delta$，只要该条件成立，特征值就必定位于单位圆内。
* **不同的速度。** 在 $\theta^*$ 附近，一次批量遍历会将误差按因子 $1 - \delta(1 - \lambda_{\max})$ 缩小，其中 $\lambda_{\max}$ 是 $DT(\theta^*)$ 的最大特征值。而该微分方程使误差按 $e^{-(1-\lambda_{\max})\tau}$ 的方式缩小，换算成日历时间就是 $t^{-(1-\lambda_{\max})}$。

```{code-cell} ipython3
δ = 0.6
print(f"DT(θ*) 的最大特征值：               {λ_max:.3f}")
print(f"批量：每次遍历误差缩小为            {1 - δ * (1 - λ_max):.3f}")
print(f"在线：误差按以下速度衰减            t^(-{1 - λ_max:.2f})")
print(f"数据量增加 500 倍，误差缩小为       {500**(1 - λ_max):.1f}")
```

因此，一次批量遍历仅能将误差削减大约 14%，这就是批量算法需要进行一百多次遍历的原因。

在线误差仅以 $t^{-0.23}$ 的速度下降。

这比抽样噪声平均掉的 $t^{-1/2}$ 速率还要慢，因此是朝向 $\theta^*$ 的漂移，而非噪声，主导了误差。

{cite:t}`Sargent1993` 借鉴 {cite:t}`BenvenisteMetivierPriouret1990` 的研究，给出了通常 $\sqrt{t}$ 收敛速率所需的条件：$DT - I$ 的每个特征值都必须小于 $-1/2$，即 $\lambda_{\max} < 1/2$，而这里并不满足该条件。

在那种速率下，将主体的经验量增加 500 倍，从 1,000 期增加到 500,000 期，也只能将误差削减约四倍。

{ref}`lae_ex5` 检验了这一收敛速率。

这种速度上的差距，衡量的正是一个被限制只能从经验到来时逐步学习的主体，与一个可以随时重复使用整个模拟历史的建模者之间的区别。

但极限是同一个对象。

萨金特的总结：

> 学习算法和均衡计算算法看起来彼此相似。均衡计算算法常常可以被解释为集中式学习算法，其中模型构建者扮演"社会规划者"的角色，摸索出一组能满足所有个体最优条件和市场出清条件的定价函数和主体决策规则。我们也已经看到，具有有限理性主体的学习系统有时可以被解释为去中心化的均衡计算算法。

正如 {cite:t}`Sargent1993` 所指出的，这就是为什么马塞特是通过早期关于*最小二乘学习动态*的研究 {cite}`MarcetSargent1989jet`，才合乎情理地得出参数化预期作为一种*计算*方法的原因。

这二者并非同一个递归从两个方向去阅读。

它们是两个共享不动点、并通过 {eq}`pea_ode` 共享稳定性条件的递归；二者的区别在于速度，以及各自对运行者的要求不同。

(lae_kernel)=
## 没有函数形式的学习

参数化路径速度快，但把一切都押注在函数族上。

如果 $\{f(\cdot, \theta)\}$ 中没有一个成员能支持一个均衡，那么该方案就会收敛到一个近似均衡然后停下来。

**非参数**替代方案不施加任何函数形式。

遵循 {cite:t}`ChenWhite1998` 的递归核估计器，主体直接从过去已实现值的核加权平均中估计条件期望 $\psi(G)$，让数据自己选择形状。

其构造基石是递归核密度估计器，它本身就是一个随机逼近递归，

$$
\hat p_t(x) = \hat p_{t-1}(x)
+ \tfrac{1}{t}\!\left[h_t^{-d}\, K\!\left(\tfrac{x - x_t}{h_t}\right) - \hat p_{t-1}(x)\right],
$$

其中 $x \in \mathbb{R}^d$，$K$ 是一个概率密度（例如标准正态密度），带宽为 $h_t \searrow 0$。

因子 $h_t^{-d}$ 使每一项都成为一个密度；若没有它，正如 {cite:t}`Sargent1993` 所印出的公式那样，该估计值会随着带宽一起收缩到零。

我们的主体需要的是条件期望，而非密度。

将这类递归的两个版本相除，便得到递归**内达拉亚-沃森**估计量

```{math}
:label: recursive_nw

\hat\psi_t(G) = \hat\psi_{t-1}(G) + \gamma_t(G)\bigl[y_t - \hat\psi_{t-1}(G)\bigr],
\qquad
\gamma_t(G) = \frac{K_{h_t}(\log G - \log G_t)}{\sum_{\tau \leq t} K_{h_\tau}(\log G - \log G_\tau)},
```

其中 $K_h(u) = h^{-1} K(u/h)$，距离以 $\log G$ 来度量。

它具有与本讲座中每一个其他递归相同的形状，只是现在每个状态都有一个独立的增益：当赤字落在 $G$ 附近时增益较大，而落在远处时增益基本为零。

把带宽固定在 $h$ 处并在网格上使其收缩，这个增益就会变得熟悉起来。

当 $h \to 0$ 时，若 $G_t = G$，则 $K_h(\log G - \log G_t)/K_h(0)$ 趋于一；否则趋于零。

于是分母便统计了访问状态 $G$ 的次数，并且当赤字处于 $G$ 时 $\gamma_t(G) = 1/\tau_G$，其中 $\tau_G$ 是迄今为止的访问次数，其余情况下为零。

这正是 {doc}`olg_adaptive_money` 中罗宾斯-门罗学习器所用的、按状态分别计时的 $1/\tau$ 时钟——本书所说的"每个事件有各自的时钟"。

一个正的带宽允许对某一状态的一次访问也推动其相邻状态的时钟：$h$ 度量的正是学习器从已见过的状态向未见过的状态进行推广的程度。

我们下面实现的是 {eq}`recursive_nw` 的**批量**对应版本：在固定带宽下对 $\psi(G)$ 的内达拉亚-沃森估计，形式为已实现被回归量在整个模拟历史上的核加权平均，并迭代至不动点。

在带宽固定、数据固定的情况下，该递归会精确地重现批量估计值。

而在一个学习型经济中，数据并非固定的，因为 $y_t$ 取决于正在被学习的规则，因此递归版本与批量版本之间的关系，正如 {eq}`recursive_pea` 与 {ref}`lae_pea` 中批量算法之间的关系：不同的递归，却有相同的不动点。

由于模拟的赤字落在网格上，对历史的核加权求和可以逐状态进行计算，依据的是每个状态被访问的频次，以及这些访问之后所对应的 $y_t$ 之和。

不动点迭代收敛缓慢，其速率大体上由与之前相同的特征值决定，因此我们允许进行多轮迭代，并在未收敛时发出警告。

```{code-cell} ipython3
def kernel_smoother(T=150_000, seed=0, h=0.06, damp=0.5, tol=1e-8, max_iter=2000):
    """
    非参数参数化预期。ψ(G) 通过对已实现被回归量的内达拉亚-沃森
    平滑器来估计——没有函数形式。该规则被迭代至一个自洽的不动点。
    返回每个状态上的储蓄规则以及迭代轮数。
    """
    idx = simulate_deficit(T, seed)
    log_G = np.log(G)
    K = np.exp(-0.5 * ((log_G[:, None] - log_G[None, :]) / h)**2)   # 状态 j 在 i 处的权重
    visits = np.bincount(idx[:-1], minlength=n_states)

    s_grid = np.full(n_states, (w1 - w2) / 2)
    for it in range(1, max_iter + 1):
        y = regressand(s_grid, G)[idx[1:]]                           # 已实现的 y_t
        y_sums = np.bincount(idx[:-1], weights=y, minlength=n_states)
        ψ = (K @ y_sums) / (K @ visits)          # 对历史的核加权平均
        s_new = np.clip(s_from_psi(ψ), G + 0.05, w1 - 0.05)
        if np.max(np.abs(s_new - s_grid)) < tol:
            return s_new, it
        s_grid = damp * s_new + (1 - damp) * s_grid
    warnings.warn(f"kernel_smoother did not converge in {max_iter} passes")
    return s_grid, it


s_kernel, passes = kernel_smoother()
print(f"收敛于第 {passes} 轮")
print("核平滑器：sup 误差 = {:.4f}，  遍历 RMS = {:.4f}".format(*errors(s_kernel)))
```

这个平滑器需要远超过一百轮迭代，原因与批量参数化预期算法相同。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "核平滑器对储蓄规则的近似"
    name: fig-la-kernel
---
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(G, s_benchmark, 'k-', lw=2.2, label="基准 $f(G)$")
ax.plot(G, s_kernel, 'C3o', ms=4, markevery=5, label="核平滑器")
ax.set_xlabel("赤字 $G$")
ax.set_ylabel("储蓄 $s$")
ax.legend(frameon=False)
plt.show()
```

在没有施加任何函数形式的情况下，核学习器很好地恢复了基准解。

在这段历史数据上，它的表现不及对同一数据拟合的二次规则：其两个误差度量都要大两到三倍。

这一差距的大小在不同的模拟历史之间会有所变化，因为核估计比三参数拟合噪声更大，但这里核方法的误差大部分是系统性的。

误差在赤字范围的顶端处会增大，在那里核方法高估了储蓄。

那里的规则很陡峭，而核窗口内的大多数观测值来自更低的赤字——这些赤字被访问得更频繁，所引发的储蓄也更多；在网格的最顶端，所有观测值都是如此。

二次函数族之所以表现得如此之好，是因为真实的 $\psi$*恰好*在 $\log G$ 中几乎正好是二次的，因此我们所选择的参数族几乎完全正确。

{ref}`lae_ex3` 表明，更好地选择带宽能够消除大部分差距——但选择带宽需要用到对基准解的了解，而学习器并不具备这种了解。

当函数族并非近乎正确时，二者就会分道扬镳，其间的权衡与计量经济学中的熟悉情形一样：

* **参数化**方案学习速度快，外推也很顺畅，但其好坏完全取决于所选的函数族；一个选择不当的函数族会得出一个近似均衡，且没有任何警示；
* **非参数**方案不会被一个糟糕的函数形式误导，但要为其灵活性付出学习较慢的代价，并且在数据稀疏之处——即赤字取值范围的边缘——会产生偏差，因为核在那里几乎没有邻居可供平均。

两者都是同一形状的递归。

一个适应性主体应该使用哪一种，再一次成为有限理性研究项目公开摆上台面的建模选择之一，而理性预期悄悄替我们做出了这一选择。

## 结束语

世代交叠模型为我们提供了一个干净的实验室，用以说明一条普遍适用的道理。

一个身处状态空间庞大或连续环境中的适应性主体，无法为每一种可能情形学习一个独立的反应。

它必须进行**泛化**，施加某种结构，使得有限的经验能够为整个状态空间上的行为提供信息。

这种结构究竟是一个低阶多项式、一个核带宽，还是别的什么，是一个具有实际后果的建模选择：它决定了可达到的极限究竟是一个理性预期均衡，还是仅仅是对它的一个近似。

正是这同一种泛化行为，使得学习与均衡计算成为同一个对象的两个侧面。

马塞特的算法通过参数化一个期望并进行回归来计算均衡；一个适应性主体通过参数化一个期望并进行回归来学习。

这两个递归并不完全相同，但它们共享相同的不动点和相同的稳定性条件，而 $T$ 映射的一个特征值决定了它们各自的收敛速度。

下一讲，{doc}`exchange_rate_learning`，将 {doc}`bounded_rationality` 中的双货币经济置于这一世代交叠的设定之中，在那里，学习者必须应对一个理性预期无法确定的汇率。

泛化问题的最尖锐形式留待 {doc}`genetic_classifier` 和 {doc}`marimon_mcgrattan_sargent` 来呈现。

在那里，状态空间和行动空间都大到无法枚举规则的地步，主体必须使用约翰·霍兰德的遗传算法和分类器系统（而非本讲中的最小二乘法和核方法机制）从零开始**发现**一套紧凑的良好规则。

## 练习

```{exercise-start}
:label: lae_ex1
```

上文的说明指出，将多项式阶数推得过高会不再有帮助，因为高阶项会追逐赤字分布稀疏访问尾部的抽样噪声。

请验证这一点，并将两种误差来源区分开来。

1. 对阶数 $0$ 到 $5$，计算*总体*近似均衡，即在拥有无限多数据的情况下该函数族的 $T$ 映射的不动点，正如 `approximate_equilibrium` 所做的那样。它与基准解之间的距离就是纯粹的近似误差。
1. 对相同的阶数，在三条长度为 150,000 的模拟历史序列上运行批量参数化预期算法。

报告相对于基准解的两种误差度量：整个网格上的 sup 范数，以及按赤字的遍历分布加权的 RMS 误差（即按主体实际活动之处加权）。

在有抽样误差和无抽样误差的情况下，每种度量随阶数的提高分别如何变化？

请解释。

```{exercise-end}
```

```{solution-start} lae_ex1
:class: dropdown
```

```{code-cell} ipython3
seeds = (0, 1, 2)
sup_rows, rms_rows = [], []
for deg in range(6):
    Φ = basis(G, deg)
    sup_pop, rms_pop = errors(rule(Φ, approximate_equilibrium(Φ)))
    sups, rmss = [sup_pop], [rms_pop]
    for seed in seeds:
        sup, rms = errors(parameterized_expectations(Φ, seed=seed)[0])
        sups.append(sup)
        rmss.append(rms)
    sup_rows.append(sups)
    rms_rows.append(rmss)

cols = ["总体"] + [f"历史 {seed}" for seed in seeds]
table = pd.concat({"sup 误差（整个网格）": pd.DataFrame(sup_rows, columns=cols),
                   "遍历 RMS（主体所处之地）": pd.DataFrame(rms_rows, columns=cols)},
                  axis=1)
table.index.name = "阶数"
table.round(4)
```

在拥有无限多数据的情况下，两种度量都随阶数提高而稳步下降：在 $\log G$ 的多项式中增加项会缩小近似误差，到 5 阶时该误差已经非常小。

在有 150,000 个观测值的情况下，误差在到达二次项之前都大幅下降，与总体情形一致。

超过这一点，样本误差就不再跟随总体误差变化了：遍历加权误差在每一条历史序列上都高于总体值，并且从 4 阶开始，这一差距与剩余的近似误差相当，甚至更大。

遍历加权误差趋于平稳，但稳定在的水平因历史序列而异。

sup 范数误差的表现则不那么规律：在一条历史序列上它持续下降，而在另外两条上它在 3 阶时上升，并一直高于其 2 阶时的值。

高阶多项式在中心区域拟合得很好，但在网格两端可以自由波动，而遍历分布在那里的质量很小，回归几乎没有数据来约束它们。

这正是 {doc}`olg_adaptive_money` 中稀有状态问题在近似方面的镜像。

一个学习方案会把它的精度分配到观测值所在的地方。

对一个很少被访问的区域拟合不佳，在 sup 范数中会表现得很明显，但在预期效用中几乎不产生任何代价，这正是遍历加权度量才是具有经济意义的度量的原因。

```{solution-end}
```

```{exercise-start}
:label: lae_ex2
```

常数族（0 阶）给出了一个*近似*均衡：一个单一的储蓄率，与赤字无关，仅在平均意义上满足一阶条件。

这里有一个自然的基准可供比较。

计算一个 0 阶参数化预期主体所收敛到的单一储蓄率 $\bar s$，并将其与赤字永远固定在其均值 $\mathbb{E}[G]$ 处的经济中的储蓄率进行比较。

那个固定赤字经济有两个稳态。

求出这两个稳态，计算它的 $T$ 映射在每一个稳态处的斜率，并说明哪一个应该用于比较。

$\bar s$ 和固定赤字经济中的储蓄率相同吗？

它们应该相同吗？

哪一个更大，为什么？

```{exercise-end}
```

```{solution-start} lae_ex2
:class: dropdown
```

当赤字固定为 $G$ 时，$T$ 映射为 $s \mapsto w_1 x/(w_2 + 2x)$，其中 $x = s - G$，因此其斜率为 $w_1 w_2/(w_2 + 2x)^2$，而稳态由 $2x^2 - (w_1 - w_2 - 2G)x + w_2 G = 0$ 求得。

```{code-cell} ipython3
# 0 阶近似均衡
s_const = rules[0][0]                         # 在各状态间为常数
G_mean = ergodic @ G

# 赤字固定在其均值处的确定性经济
x_roots = np.roots([2, -(w1 - w2 - 2 * G_mean), w2 * G_mean])
s_steady = np.sort(x_roots + G_mean)[::-1]    # 高储蓄（低通胀）排在前
slopes = w1 * w2 / (w2 + 2 * (s_steady - G_mean))**2

s_iter = (w1 - w2) / 2                        # 从 s = 5 开始迭代 T 映射
for _ in range(1000):
    s_iter = 0.5 * s_iter + 0.5 * s_from_psi(regressand(s_iter, G_mean))

print(f"平均赤字 E[G]：{G_mean:.4f}")
print(f"赤字固定在 E[G] 处的稳态：{s_steady.round(4)}")
print(f"T 映射在每个稳态处的斜率：{slopes.round(3)}")
print(f"从 s = 5 迭代 T 映射收敛到：{s_iter:.4f}")
print(f"0 阶近似均衡储蓄率：{s_const:.4f}")
```

与 {doc}`olg_adaptive_money` 第二部分中一样，固定赤字经济有两个稳态。

在高储蓄、低通胀的那个稳态处，$T$ 映射的斜率约为 0.71，小于 1；在低储蓄的那个稳态处，斜率约为 1.4，大于 1。

从 $s = 5$ 开始迭代储蓄映射会收敛到第一个稳态，而任何基于信念迭代的过程都不可能停留在第二个稳态。

这正是 {ref}`olg_reversal` 中稳定性反转的计算层面体现，也正是为什么低通胀稳态才是正确的比较对象。

两个储蓄率接近但不相等，而且差距的符号是可预测的。

对常数项做回归所得到的是 $y_t$ 的样本均值，因此 $\bar s$ 是如下映射的不动点：该映射把 $s$ 映射到 $w_1 \psi/(1 + \psi)$，其中 $\psi = \mathbb{E}[y(s, G)]$，期望是在赤字的遍历分布上取的。

而固定赤字经济用 $\psi = y(s, \mathbb{E}[G])$ 取而代之。

对给定的 $s$，$y(s, G) = 1 - w_2/(w_2 + s - G)$ 关于 $G$ 是凹的，因此根据詹森不等式，$\mathbb{E}[y(s, G)] < y(s, \mathbb{E}[G])$：将赤字分散开来会降低储蓄的预期回报。

因此，0 阶映射在每一个 $s$ 处都低于固定赤字映射。

两条映射都在其稳定不动点处从上方穿过 45 度线，所以 $\bar s$ 低于固定赤字经济中的储蓄率，这与上面的数字相符。

只有当赤字一开始就是确定性的，二者才会一致。

其中的教训是，一个近似均衡并非一个简单的"确定性等价"对象。

即使是最粗糙的单参数学习器，也在对它所经历的*整个*结果分布做出反应，而不是对一个点预测做出反应。

```{solution-end}
```

```{exercise-start}
:label: lae_ex3
```

非参数核学习器有一个自由的调节常数：带宽 $h$。

请在一条较短的历史序列（$T = 40{,}000$）上探索它的影响，此时抽样噪声会起作用。

对从 $0.003$ 到 $0.2$ 的一系列带宽，在十条模拟历史序列上运行核学习器。

把它的遍历加权均方误差分解为平方偏差（在各条历史序列上平均后的规则误差）和方差（各条历史序列中规则的离散程度）。

解释这种权衡的形状。

最小的带宽对应着什么？

在最佳带宽处的核学习器，与在同样历史序列上拟合的二次参数规则相比表现如何？

```{exercise-end}
```

```{solution-start} lae_ex3
:class: dropdown
```

```{code-cell} ipython3
T_short, seeds = 40_000, range(10)
bandwidths = (0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)

rows = []
for h in bandwidths:
    S = np.array([kernel_smoother(T=T_short, seed=seed, h=h)[0] for seed in seeds])
    bias2 = ergodic @ (S.mean(axis=0) - s_benchmark)**2
    var = ergodic @ S.var(axis=0)
    rows.append([h, np.sqrt(bias2), np.sqrt(var), np.sqrt(bias2 + var)])

table = pd.DataFrame(rows, columns=["带宽 $h$", "偏差", "标准差",
                                    "RMS 误差"]).set_index("带宽 $h$")

S_quad = np.array([parameterized_expectations(Φ2, T=T_short, seed=seed)[0] for seed in seeds])
rms_quad = np.sqrt(ergodic @ ((S_quad - s_benchmark)**2).mean(axis=0))
step = np.diff(np.log(G))[0]

print(f"二次族在同样历史序列上的表现：RMS 误差 {rms_quad:.4f}")
print(f"h = 0.003 时相邻状态的权重为 {np.exp(-0.5 * (step / 0.003)**2):.4f}")
table.round(4)
```

规则本身展现了这两种失败模式。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.2))
for h, colour in [(0.003, 'C0'), (0.03, 'C1'), (0.20, 'C2')]:
    ax.plot(G, kernel_smoother(T=T_short, h=h)[0], '-', lw=1, color=colour, label=f"$h = {h}$")
ax.plot(G, s_benchmark, 'k-', lw=2, label="基准")
ax.set_xlabel("赤字 $G$")
ax.set_ylabel("储蓄 $s$")
ax.legend(frameon=False)
plt.show()
```

方差随带宽增大而稳步下降。

偏差在带宽约为 $h = 0.03$ 之前都很小，之后则急剧上升。

（在最小的带宽处，偏差一列所测得的基本上是十条含噪规则的平均值中的噪声，大约是标准差除以 $\sqrt{10}$。）

因此 RMS 误差存在一个内部最小点，这里接近 $h = 0.03$。

**较小**的带宽只对少数几个邻居求平均，因此估计带有噪声：它围绕基准解摆动，在赤字范围的高端摆动最为严重。

这是权衡中的**方差**一端。

在 $h = 0.003$ 时，即使对最近的相邻状态，权重也可以忽略不计，因此此时学习器实际上等同于 {doc}`olg_adaptive_money` 中逐状态的学习器，每个状态一个估计值：这是 {eq}`recursive_nw` 在 $h \to 0$ 时的极限。

**较大**的带宽在较宽的窗口上求平均，使估计变得平滑，但也把它压平：规则被拉向取值范围中部的储蓄水平，在低赤字处低于基准解，在高赤字处高于基准解。

这是**偏差**一端。

在其最佳带宽处，核学习器的精度大致与在同样历史序列上拟合的二次族相当，但要找到这个最佳带宽需要用到基准解，而这是任何学习器都无法获得的。

带宽是 {ref}`lae_ex1` 中多项式阶数的非参数对应物：二者都控制着学习器所施加的结构多少，并且都有一个最佳点，在因灵活性不足而导致的拟合不佳与因灵活性过多而导致的噪声之间进行权衡。

有限理性研究项目并不能告诉我们这个最佳点在哪里；这又是一个用一系列合理的备选方案取代理性预期这一单一约束的选择。

```{solution-end}
```

```{exercise-start}
:label: lae_ex4
```

回到 {doc}`olg_adaptive_money` 中的经济体 2。

在那里，赤字在状态 1 为 $0.8$，在状态 2 为 $0$，转移矩阵为

$$
\begin{bmatrix} 0.75 & 0.25 \\ 0.50 & 0.50 \end{bmatrix},
$$

理性预期储蓄率为 $(4.211, 4.364)$，这是该讲中逐状态的罗宾斯-门罗学习器所求得的。

1. 用基 $\phi(G) = (1, G)$ 在这个经济体上运行批量参数化预期算法。（本讲中的对数赤字基并不适用，因为状态 2 中 $G = 0$。）证明它能重现理性预期储蓄率，并解释为什么必然如此。
1. 现在使用常数基 $\phi(G) = 1$。分别运行批量算法和在线学习器 {eq}`recursive_pea`，并将它们的极限相互比较，也与理性预期均衡相比较。

函数 `parameterized_expectations`、`online_pea` 和 `approximate_equilibrium` 接受赤字水平、转移矩阵以及在每个状态下取值的基作为参数。

```{exercise-end}
```

```{solution-start} lae_ex4
:class: dropdown
```

以下是第一部分。

```{code-cell} ipython3
G_2 = np.array([0.8, 0.0])
P_2 = np.array([[0.75, 0.25],
                [0.50, 0.50]])
π_2 = qe.MarkovChain(P_2).stationary_distributions[0]
economy_2 = dict(G=G_2, P=P_2)

Φ_lin = np.column_stack([np.ones(2), G_2])
s_lin_batch = parameterized_expectations(Φ_lin, **economy_2)[0]
s_lin_pop = rule(Φ_lin, approximate_equilibrium(Φ_lin, π=π_2, **economy_2), G_2)

print(f"批量算法，φ = (1, G)，T = 150,000：{s_lin_batch.round(4)}")
print(f"总体不动点：{s_lin_pop.round(4)}")
print( "理性预期：[4.211 4.364]")
```

在一个两点状态空间上，$G$ 的任何函数都是 $G$ 的线性函数：两个系数就能匹配任意两个取值。

因此线性族恰好包含了理性预期均衡，算法的每一个不动点都是其中之一。

这个经济体还有一个高通胀均衡，储蓄约为 $(1.19, 1.63)$，但它不是 E 稳定的，因此算法最终会收敛到上面展示的那个均衡。

总体不动点在所展示的每一位数字上都与之相符，而批量版本与它的差异仅来自抽样误差。

在两个点上，这个"参数化"族实际上变相是非参数的：它达到的均衡与 {doc}`olg_adaptive_money` 中逐状态学习器所求得的均衡相同，只是通过模型构建而非逐个艰难求解达成的。

以下是第二部分，在线学习器在两种基下都运行了 200,000 期。

```{code-cell} ipython3
Φ_const = np.ones((2, 1))
s_const_batch = parameterized_expectations(Φ_const, **economy_2)[0]
s_const_pop = rule(Φ_const, approximate_equilibrium(Φ_const, π=π_2, **economy_2), G_2)

T_2 = 200_000
paths = {"常数": (Φ_const, online_pea(Φ_const, T=T_2, **economy_2)),
         "线性": (Φ_lin,   online_pea(Φ_lin,   T=T_2, **economy_2))}

print(f"常数基，批量算法：{s_const_batch.round(4)}")
print(f"常数基，总体：{s_const_pop.round(4)}")
for name, (Φ, path) in paths.items():
    for t in (1_000, 10_000, 100_000, T_2 - 1):
        print(f"{name:8s}基，在线学习，t = {t:>7,}：{rule(Φ, path[t], G_2).round(4)}")
```

常数族无法让储蓄依赖于状态，因此其不动点是一个介于两个理性预期储蓄率之间的单一储蓄率：这是一个近似均衡。

批量算法和在线学习器所趋向的正是这同一个储蓄率，而不是理性预期均衡。

在线学习器缓慢地从上方逼近它，因为它从 $s = 5$ 开始。

在线性基下，在线学习器则转而趋向理性预期储蓄率。

一个学习器能够达到什么，是由它所采用的函数族决定的，而不是由它是批量学习还是实时学习决定的。

```{solution-end}
```

```{exercise-start}
:label: lae_ex5
```

正文预测，在线学习器与近似均衡 $\theta^*$ 之间的距离会以 $t^{-(1 - \lambda_{\max})}$ 的速率收缩，其中 $\lambda_{\max}$ 是 $DT(\theta^*)$ 的最大特征值。

1. 在对数-对数坐标轴上，绘制本讲中所计算路径以及另外两条模拟历史序列的 $\|\theta_t - \theta^*\|$ 随 $t$ 变化的图像。对 $t \geq 10{,}000$ 的部分拟合斜率，并将其与 $-(1 - \lambda_{\max})$ 以及抽样噪声被平均掉的速率 $-1/2$ 进行比较。
1. 将增益改为 $\gamma_t = 3/(t + 3)$ 来代替 $1/(t + 1)$，重复上述过程，`online_pea` 通过 `a=3` 给出这一增益。发生了什么变化？为什么？

```{exercise-end}
```

```{solution-start} lae_ex5
:class: dropdown
```

```{code-cell} ipython3
T_online = len(θ_path)
ts = np.unique(np.logspace(2, np.log10(T_online - 1), 60).astype(int))
late = ts >= 10_000

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, a in zip(axes, (1, 3)):
    starts = []
    for seed in (0, 1, 2):
        path = θ_path if (a == 1 and seed == 0) else online_pea(Φ2, seed=seed, a=a)
        dist = np.linalg.norm(path[ts] - θ_star, axis=1)
        slope = np.polyfit(np.log(ts[late]), np.log(dist[late]), 1)[0]
        ax.loglog(ts, dist, lw=1.5, label=f"历史 {seed}：斜率 {slope:.2f}")
        starts.append(dist[late][0])
    t_rel = ts[late] / ts[late][0]                # 参考线从 t = 10,000 开始
    anchor = 1.5 * max(starts)
    ax.loglog(ts[late], anchor * t_rel**(-(1 - λ_max)), 'k--', lw=1,
              label=rf"斜率 $-(1-\lambda_{{\max}}) = {-(1 - λ_max):.2f}$")
    ax.loglog(ts[late], anchor * t_rel**(-0.5), 'k:', lw=1, label="斜率 $-1/2$")
    ax.set_xlabel("$t$")
    ax.set_title(f"增益 $\\gamma_t = {a}/(t + {a})$")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel(r"$\|\theta_t - \theta^*\|$")
plt.tight_layout()
plt.show()
```

在增益为 $1/(t + 1)$ 时，拟合的斜率接近 $-(1 - \lambda_{\max}) \approx -0.23$，与 $-1/2$ 相去甚远。

在重新标度的时间 $\tau = \log t$ 下，均值动态 {eq}`pea_ode` 以速率 $1 - \lambda_i$ 收缩，其中最慢的是 $1 - \lambda_{\max}$，因此向 $\theta^*$ 的漂移以 $t^{-(1 - \lambda_{\max})}$ 的速率衰减，而抽样噪声则以 $t^{-1/2}$ 的速率被平均掉。

两者中较慢的那个起主导作用，而在这里正是漂移项。

增益为 $a/t$ 会使微分方程的运行速度加快 $a$ 倍，因此漂移项会以 $t^{-a(1 - \lambda_{\max})}$ 的速率衰减。

当 $a = 3$ 时，这比 $t^{-1/2}$ 更快，因此从长期来看，抽样噪声会占主导地位，误差会以通常的 $\sqrt{t}$ 速率下降。

此时拟合的斜率已接近 $-1/2$ 甚至更陡，而在经过五十万期之后，误差比使用增益 $1/(t+1)$ 时小了好几倍。

这正是本书中 $\sqrt{t}$ 收敛条件既取决于增益、也取决于 $DT$ 这一说法的意义所在。

更大的增益并非没有代价：它们会使远离 $\theta^*$ 的早期阶段更加剧烈，而一个永不衰减的增益会使信念永远在 $\theta^*$ 周围波动，这正是 {ref}`br_one_recursion` 中所描述的常增益行为。

```{solution-end}
```
