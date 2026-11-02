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
  title: 汇率不确定性、学习与实验
  headings:
    Overview: 概览
    The two-currency economy: 双货币经济
    The two-currency economy::The indeterminacy, recalled: 回顾不确定性
    Newton–Raphson learning: 牛顿-拉夫森学习
    Newton–Raphson learning::The indeterminacy shows up as a singular Hessian: 不确定性表现为一个奇异海森矩阵
    Newton–Raphson learning::Convergence to a history-dependent exchange rate: 收敛到一个依赖历史的汇率
    Newton–Raphson learning::The dead hand of history: 历史的死手
    Newton–Raphson learning::The ghost of indeterminacy: 不确定性的幽灵
    Evidence from the laboratory: 来自实验室的证据
    A genetic algorithm economy: 遗传算法经济体
    A genetic algorithm economy::Volatility that does not die out: 不会消退的波动
    A genetic algorithm economy::What the population learns: 群体学到了什么
    A genetic algorithm economy::The shape of the volatility: 波动的形态
    Concluding remarks: 结语
    Exercises: 练习
---

(exchange_rate_learning)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 汇率不确定性、学习与实验

```{index} single: Bounded Rationality; Exchange Rate Indeterminacy
```

```{contents} Contents
:depth: 2
```

## 概览

在 {doc}`bounded_rationality` 中，我们遇到过一个模型，其中理性预期完全无法确定汇率。

两种法定货币，只要回报率相等就是完美替代品，这使得汇率 $e$ *完全不受限制*：如果均衡条件对某个 $e$ 有解，那么对其他任何 $e$ 也都有解，并且所有情形下的实际配置都是相同的。

本讲座遵循 {cite:t}`Sargent1993` 的思路，将理性主体从这个经济中驱逐出去，代之以适应性主体。

它是本系列的枢纽。

其前半部分使用了本系列中最后一种计量经济学学习者——一种类似于 {doc}`olg_adaptive_money` 中的牛顿-拉夫森规则。

其后半部分使用了本系列中第一个遗传算法，这一工具随后将由 {doc}`genetic_classifier` 做系统的阐述。

以下内容围绕两个问题展开。

**第一**，学习能否确定汇率？

一个适应性主体拥有信念、修正信念的规则以及初始条件，而这些初始条件恰恰是均衡条件未能确定的东西。

我们将看到，牛顿-拉夫森学习者确实会收敛到一个恒定的汇率，但学习并不能*选定*某一个汇率。

它们最终落在哪里，取决于其投资组合和储蓄率的出发点，也取决于学习算法的调校方式。

学习使汇率变得*依赖于历史*：萨金特所说的"历史的死手"完成了基本面拒绝完成的锁定任务。

**第二**，不确定性的幽灵是否依然存在？

确实存在。

学习算法的静止点恰好复现了这种不确定性：使算法停止的条件，正是当初使汇率变得自由的那个套利条件。

萨金特判断，一种以这种方式选定汇率路径的适应性机制"可能是确定汇率所依靠的一根脆弱的芦苇"。

接下来我们转向证据。

{cite:t}`Arifovic1996` 将这一经济体作为付费人类受试者参与的实验室实验来运行。

他们的汇率并未稳定下来。

而当她用**遗传算法**——一个由选择、交叉和变异培育出来的二进制字符串主体群体——替换牛顿-拉夫森学习者时，她得到了一个汇率同样无法稳定下来的经济体。

这一遗传群体学会了以接近理性预期的比率进行储蓄，但没有什么能锁定其投资组合份额，它会沿着均衡的连续统持续游移。

不确定性的幽灵以波动率的形式卷土重来。

这种对比——一种收敛的学习规则与一种产生持续波动的学习规则——正是本讲座的落脚点。

让我们先导入一些包。

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
```

## 双货币经济

我们采用 {cite:t}`KarekenWallace1981` 不确定性的世代交叠模型化身。

在每个时点 $t$，有 $N$ 个生存两期的主体出生，年轻时禀赋为 $w_1$，年老时禀赋为 $w_2$，且 $w_1 > w_2$。

（与 {doc}`olg_adaptive_money` 一致，禀赋记为 $w_1$ 和 $w_2$；而 {doc}`bounded_rationality` 将其记为 $2w_1$ 和 $2w_2$。）

存在两种供给量固定的法定货币 $H_1$ 和 $H_2$。

一个年轻主体做出两项决策：储蓄多少 $s_t$，以及将这部分储蓄中多大比例 $\lambda_t$ 持有为货币 1（其余部分持有货币 2）。

选择 $(s_t, \lambda_t)$ 的主体实现的终身效用为

```{math}
:label: xr_utility

U(s_t, \lambda_t)
= u(w_1 - s_t)
+ u\!\left(w_2 + \lambda_t s_t \frac{p_{1t}}{p_{1,t+1}}
              + (1 - \lambda_t) s_t \frac{p_{2t}}{p_{2,t+1}}\right),
```

其中 $p_{it}$ 是以货币 $i$ 计价的价格水平，$p_{it}/p_{i,t+1}$ 是持有货币 $i$ 的总回报率。

我们全程采用 $u(c) = \ln c$。

将每种货币的供给等同于对它的需求，得到价格水平

```{math}
:label: xr_prices

p_{1t} = \frac{H_1}{\sum_i \lambda_{it} s_{it}},
\qquad
p_{2t} = \frac{H_2}{\sum_i (1 - \lambda_{it}) s_{it}},
```

汇率为 $e_t = p_{1t}/p_{2t}$。

### 回顾不确定性

考虑一个价格恒定的平稳均衡。

那么每种货币的回报为 $p_{it}/p_{i,t+1} = 1$，所以两种回报相等，且 {eq}`xr_utility` 中的组合回报无论 $\lambda$ 为何值都是 $1$。

储蓄决策于是求解 $\max_s \ln(w_1 - s) + \ln(w_2 + s)$，得到

$$
s^\star = \frac{w_1 - w_2}{2},
$$

但组合份额 $\lambda$ 是*完全不确定的*：当两种回报相等时，效用与 $\lambda$ 无关。

而 $\lambda$ 恰恰决定了汇率。

由 {eq}`xr_prices`，在公共的 $(s, \lambda)$ 下，

$$
e = \frac{p_{1}}{p_{2}} = \frac{H_1}{H_2}\cdot\frac{1 - \lambda}{\lambda}.
$$

每一个 $\lambda \in (0, 1)$ 都是均衡，因此每一个 $e \in (0, \infty)$ 都是均衡。

这正是 {doc}`bounded_rationality` 中的不确定性，现在通过一个未确定的组合选择表现了出来。

本讲座将运行两个参数不同的经济体。

牛顿-拉夫森方法一节使用萨金特的参数 $(w_1, w_2, H_1, H_2) = (20, 15, 100, 120)$，其货币供给量与 {doc}`bounded_rationality` 中双货币示例的货币供给量相同。

实验室和遗传算法各节则使用阿里福维奇的参数 $(11, 1, 10, 10)$。

因此我们将参数放入一个容器中，并显式传递，而不是让它们作为全局变量随意存在。

```{code-cell} ipython3
class Params(NamedTuple):
    w1: float          # 年轻时的禀赋
    w2: float          # 年老时的禀赋
    H1: float          # 货币1的固定供给
    H2: float          # 货币2的固定供给

    @property
    def s_star(self):
        "理性预期储蓄率，当两种货币回报均为1时。"
        return (self.w1 - self.w2) / 2


kw = Params(w1=20.0, w2=15.0, H1=100.0, H2=120.0)

print(f"rational expectations saving rate s* = {kw.s_star}")
print(f"any portfolio share λ is an equilibrium, and e = (H1/H2)(1-λ)/λ is free")
```

## 牛顿-拉夫森学习

现在驱逐理性主体。

与 {doc}`olg_adaptive_money` 中一样，一个世代的终身效用只有等到该世代年老之后才能被评估，因此总体被分成一个"偶数类"和一个"奇数类"。

每一类都携带自己的规则 $(s, \lambda)$，仅根据同一类之前主体的经验来更新。

一个主体根据实现的效用，通过**牛顿-拉夫森**步骤来修正 $(s, \lambda)$：朝着根据 {eq}`xr_utility` 的二阶展开、依据主体实际经历的回报本应能提高效用的方向移动。

用 $\nabla U_\tau$ 和 $\nabla^2 U_\tau$ 表示实现效用的梯度与海森矩阵，用 $M$ 表示（负定）海森矩阵的滚动估计，递推式为

```{math}
:label: xr_newton

\begin{aligned}
M_{\tau+1} &= M_\tau + \gamma_\tau (\nabla^2 U_\tau - M_\tau), \\
\begin{bmatrix} s \\ \lambda \end{bmatrix}_{\tau+1}
&= \begin{bmatrix} s \\ \lambda \end{bmatrix}_\tau
   - \gamma_\tau M_{\tau+1}^{-1}\, \nabla U_\tau,
\end{aligned}
```

其中 $\gamma_\tau$ 是增益。

这是 {doc}`olg_adaptive_money` 中罗宾斯-门罗规则的双变量版本。

我们使用**常数增益**，$\gamma_\tau = \gamma = 0.3$，这是萨金特在一条脚注中提到的一种变体。

由于该经济体没有冲击，常数增益仍可以收敛到一个点。

下面是 {eq}`xr_utility` 关于回报 $R_1 = p_{1t}/p_{1,t+1}$ 和 $R_2 = p_{2t}/p_{2,t+1}$ 的梯度与海森矩阵。

```{code-cell} ipython3
def grad_hess(p, s, lam, R1, R2):
    "实现效用 U(s, λ) 的梯度与海森矩阵。"
    A = lam*R1 + (1 - lam)*R2               # 组合总回报
    c2 = p.w2 + s*A                         # 老年消费
    dR = R1 - R2
    g = np.array([-1/(p.w1 - s) + A/c2,
                  s*dR/c2])
    U_ss = -1/(p.w1 - s)**2 - A**2/c2**2
    U_ll = -s**2 * dR**2 / c2**2
    U_sl = dR/c2 - s*A*dR/c2**2
    return g, np.array([[U_ss, U_sl], [U_sl, U_ll]])
```

### 不确定性表现为一个奇异海森矩阵

看看海森矩阵中 $\lambda$ 那一块。

当两种回报相等，$R_1 = R_2$ 时，所有 $\lambda$-偏导数都消失了：梯度分量 $s(R_1 - R_2)/c_2$ 为零，曲率 $-s^2(R_1 - R_2)^2/c_2^2$ 也是零。

在回报相等时，效用**沿 $\lambda$ 方向是平坦的**。

这正是局部所见的不确定性：没有任何力量将 $\lambda$ 推向任何方向。

在回报接近相等时，曲率相对于 $R_1 - R_2$ 是二阶的，而梯度是一阶的，因此在 $\lambda$ 方向上一个未加阻尼的牛顿步骤——将梯度除以曲率——会变得巨大。

因此，我们在 $\lambda$ 方向的曲率估计中加入一个常数**脊值** $\kappa > 0$：每一期我们都用矩阵 $\nabla^2 U_\tau - \mathrm{diag}(0, \kappa)$ 来代替 {eq}`xr_newton` 中的 $\nabla^2 U_\tau$。

这是数值优化中列文伯格-马夸特方法的单方向版本，是本讲所新加入的内容，并非原书所有。

它是一个*稳定器*，而非一个使能器。

如果没有它，算法并不会被冻结；而是会爆炸，正如我们下面要检验的那样。

有了它，$\lambda$ 的一步大约是 $\gamma \, |\partial U/\partial \lambda| / \kappa$，因此较大的 $\kappa$ 会使组合调整变得迟缓。

```{code-cell} ipython3
def newton_learning(p, s0, lam0, T=400, gain=0.3, κ=0.5, w2_path=None):
    """
    带有牛顿-拉夫森学习(s, λ)的双货币世代交叠模型，每一类（偶数/奇数）一条规则。
    κ是在λ方向上加到曲率估计M上的脊值。w2_path可选地给出每个日期的老年期禀赋。
    """
    s = np.array(s0, float)
    lam = np.array(lam0, float)
    M = [np.array([[-1.0, 0.0], [0.0, -κ]]) for _ in range(2)]   # 初始曲率
    p1 = np.empty(T)
    p2 = np.empty(T)
    e = np.empty(T)
    s_hist = np.empty((T, 2))
    lam_hist = np.empty((T, 2))

    for t in range(T):
        j = t % 2                                    # 在t时年轻的一类
        p1[t] = p.H1 / (lam[j]*s[j])
        p2[t] = p.H2 / ((1 - lam[j])*s[j])
        e[t] = p1[t] / p2[t]
        if t >= 1:                                   # 在t-1时年轻的一类现在年老了
            jp = 1 - j
            R1, R2 = p1[t-1]/p1[t], p2[t-1]/p2[t]
            q = p if w2_path is None else p._replace(w2=w2_path[t])
            g, h = grad_hess(q, s[jp], lam[jp], R1, R2)
            h[1, 1] -= κ                             # 脊值
            M[jp] = M[jp] + gain*(h - M[jp])
            step = gain * np.linalg.solve(M[jp], g)
            s[jp] = np.clip(s[jp] - step[0], 0.1, p.w1 - 0.1)
            lam[jp] = np.clip(lam[jp] - step[1], 0.02, 0.98)
        s_hist[t] = s
        lam_hist[t] = lam

    return e, s_hist, lam_hist
```

### 收敛到一个依赖历史的汇率

运行两个初始储蓄率和组合份额不同、其余条件完全相同的实验。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Newton-Raphson learning from two initial conditions"
    name: fig-xr-learning
---
e1, s1, l1 = newton_learning(kw, [3.5, 2.0], [0.35, 0.40])
e2, s2, l2 = newton_learning(kw, [2.0, 3.5], [0.62, 0.58])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(np.log(e1), 'C0', lw=1.3, label="experiment 1")
axes[0].plot(np.log(e2), 'C1', lw=1.3, label="experiment 2")
axes[0].set_title("log exchange rate")
axes[0].set_xlabel("$t$")
axes[0].legend(frameon=False)

for s_h, c in [(s1, 'C0'), (s2, 'C1')]:
    axes[1].plot(s_h[:, 0], color=c, lw=1.0)
    axes[1].plot(s_h[:, 1], color=c, lw=1.0, ls=':')
axes[1].axhline(kw.s_star, color='k', lw=0.8, ls='--')
axes[1].set_title("saving  (solid = even, dotted = odd)")
axes[1].set_xlabel("$t$")

for l_h, c in [(l1, 'C0'), (l2, 'C1')]:
    axes[2].plot(l_h[:, 0], color=c, lw=1.0)
    axes[2].plot(l_h[:, 1], color=c, lw=1.0, ls=':')
axes[2].set_title(r"portfolio share $\lambda$")
axes[2].set_xlabel("$t$")
plt.tight_layout()
plt.show()
```

两个经济体都收敛了。

储蓄从任一方向都会攀升或下降至理性预期比率 $s^\star = 2.5$，且两类主体的组合份额都汇合到一个共同的值。

但这两个实验收敛到了*不同的汇率*。

```{code-cell} ipython3
for name, e, s_h, l_h in [("experiment 1", e1, s1, l1), ("experiment 2", e2, s2, l2)]:
    still = np.abs(np.diff(l_h, axis=0)).max(axis=1) > 1e-6
    t_freeze = np.flatnonzero(still)[-1] + 2         # λ moves by < 1e-6 from here on
    print(f"{name}: e → {e[-1]:.4f} (log e → {np.log(e[-1]):.3f}),  s → {s_h[-1][0]:.3f},  "
          f"λ → {l_h[-1][0]:.4f},  λ frozen from t = {t_freeze}")
```

储蓄率是由基本面锁定的；汇率则是由历史锁定的。

实验1的极限与萨金特实验1的结果相近，后者的 $\lambda$ 稳定在接近 $0.37$ 处，对数汇率稳定在接近 $0.3$ 处。

萨金特的路径即便经过数百期仍在漂移，而在我们的常数增益下，$\lambda$ 大约在 $35$ 期之内就停止了变动。

在研究极限如何依赖于历史之前，让我们先看看脊值 $\kappa$ 的作用。

下面再次用 $\kappa = 0$ 重做实验1。

```{code-cell} ipython3
e0, s0_h, l0_h = newton_learning(kw, [3.5, 2.0], [0.35, 0.40], κ=0.0)
print("last four exchange rates:", ", ".join(f"{x:.3f}" for x in e0[-4:]))
print(f"final λ (even, odd): {l0_h[-1]},  final s (even, odd): {s0_h[-1]}")
```

如果没有脊值，算法并不会静止下来。

曲率估计仍保持可逆，但其 $\lambda$ 分量极小，因此 $\lambda$ 的步长变得巨大。

组合份额会撞到代码所施加的边界 $0.02$ 和 $0.98$，储蓄崩溃到其下界，此后汇率会在两个极端值之间来回翻转。

(xr_dead_hand)=
### 历史的死手

为了看清历史对结果的支配，我们扫描初始组合份额，记录每个经济体最终稳定在哪个汇率上。

两类主体必须以*不同*的组合份额起步。

如果它们以相同的 $\lambda_0$ 起步，那么汇率——它只取决于年轻一类的 $\lambda$——从第一期起就是常数。

于是 $R_1 = R_2$ 从一开始就成立，$\lambda$-梯度恒为零，$\lambda$ 永远不会移动，因此按构造极限将是 $(H_1/H_2)(1-\lambda_0)/\lambda_0$。

因此我们让偶数类和奇数类分别以 $\bar\lambda_0 - 0.05$ 和 $\bar\lambda_0 + 0.05$ 起步，并采用实验1的储蓄率，将每个极限对初始份额均值 $\bar\lambda_0$ 作图。

我们将这些极限与曲线 $(H_1/H_2)(1-\lambda)/\lambda$ 在两个点上的取值进行比较：极限份额 $\lambda_\infty$，以及初始份额均值 $\bar\lambda_0$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Limiting exchange rate against the mean initial portfolio share"
    name: fig-xr-limits
---
def e_of_λ(p, lam):
    "当每个主体都持有组合份额λ时的汇率。"
    return (p.H1/p.H2) * (1 - lam) / lam


λ_bar0 = np.linspace(0.15, 0.85, 15)
e_lim = np.empty_like(λ_bar0)
λ_lim = np.empty_like(λ_bar0)
for k, lb in enumerate(λ_bar0):
    e, s_h, l_h = newton_learning(kw, [3.5, 2.0], [lb - 0.05, lb + 0.05])
    e_lim[k], λ_lim[k] = e[-1], l_h[-1].mean()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(λ_bar0, e_lim, 'C0o', ms=6, label="limiting $e$ from learning")
ax.plot(λ_bar0, e_of_λ(kw, λ_lim), 'C0-', lw=1,
        label=r"$(H_1/H_2)(1-\lambda_\infty)/\lambda_\infty$")
ax.plot(λ_bar0, e_of_λ(kw, λ_bar0), 'k--', lw=1,
        label=r"$(H_1/H_2)(1-\bar\lambda_0)/\bar\lambda_0$")
ax.set_yscale('log')
ax.set_xlabel(r"mean initial portfolio share $\bar\lambda_0$")
ax.set_ylabel("limiting exchange rate $e$")
ax.legend(frameon=False)
plt.show()
```

下面是极限值与两条曲线之间的差距。

```{code-cell} ipython3
gap_lim = np.abs(e_lim / e_of_λ(kw, λ_lim) - 1)
gap_init = np.abs(e_lim / e_of_λ(kw, λ_bar0) - 1)
print(f"largest relative gap from the curve at λ∞:  {gap_lim.max():.1e}")
print(f"relative gap from the curve at mean λ0:     "
      f"{gap_init.min():.3f} to {gap_init.max():.3f}")
```

每一个极限值都恰好落在理性预期连续统上，取值于其自身的极限 $\lambda$ 处：每一个都是一个均衡。

这个零差距几乎是自动成立的：在每一期（包括过渡期在内），汇率都等于年轻一类份额对应的 $(H_1/H_2)(1-\lambda)/\lambda$，因此这个差距所确认的仅仅是两类主体的份额已经汇合。

真正有信息量的比较是与虚线曲线的对比。

随着 $\bar\lambda_0$ 从 $0.15$ 扫描到 $0.85$，极限值描绘出连续统中相当宽的一段。

但极限值并不仅仅是初始份额均值所蕴含的那个均衡。

只要两类主体仍存在差异，两种货币的回报就会存在差异，这些过渡期的回报差距会在套利条件关闭该运动之前，把 $\lambda$ 来回推动。

与虚线曲线的差距从百分之零点五到近百分之二十不等。

差距在网格两端最大，那里两类主体的初始份额所蕴含的汇率差别也最大。

这正是萨金特所说的历史的死手：

> 我们的学习系统通过使决策具有足够的迟滞性，即借助'历史的死手'，使汇率路径变得确定。这条汇率路径可以说是'依赖历史的'，因为赋予 $(\lambda, s)$ 的初始条件所具有的重要性并不会随时间流逝而消失。

相关的历史不仅仅是初始组合份额。

极限值还取决于初始储蓄率，以及算法的调参——脊值 $\kappa$ 和增益 $\gamma$；{ref}`xr_ex1` 对这两者都做了度量。

学习把*水平上的不确定性*转化为*对历史的依赖*。

它使汇率依赖于历史；但并不会在其中作出唯一的选择。

(xr_ghost)=
### 不确定性的幽灵

为什么汇率会被冻结，而不是移动到某个特定的值？

因为算法在梯度消失的任何地方都会静止下来，而组合梯度 $s(R_1 - R_2)/c_2$ 恰好在 $R_1 = R_2$ 时消失，而这正是使汇率在理性预期下变得自由的那个**套利条件**。

让我们计算实验1结束时的回报和梯度，以及连续统上其他点处的 $\lambda$ 导数。

```{code-cell} ipython3
s_lim, lam_lim = s1[-1], l1[-1]                  # 结束时的(偶数，奇数)规则
p1_lim = kw.H1 / (lam_lim*s_lim)                 # 各类年轻时的价格水平
p2_lim = kw.H2 / ((1 - lam_lim)*s_lim)
R1, R2 = p1_lim[1]/p1_lim[0], p2_lim[1]/p2_lim[0]  # 奇数类主体获得的回报
g, h = grad_hess(kw, s_lim[1], lam_lim[1], R1, R2)
print(f"end of experiment 1: R1 = {R1:.6f}, R2 = {R2:.6f}, gradient = {g.round(8) + 0.0}")

print("\nλ derivatives at other rest points (s = s*, R1 = R2 = 1):")
for lam in (0.2, 0.5, 0.8):
    g, h = grad_hess(kw, kw.s_star, lam, 1.0, 1.0)
    print(f"  λ = {lam}:  ∂U/∂λ = {g[1] + 0.0:.1f},  ∂²U/∂λ² = {h[1, 1] + 0.0:.1f},  "
          f"e = {e_of_λ(kw, lam):.3f}")
```

学习动态的静止点*就是*理性预期均衡本身，汇率也不例外：它们构成一整个连续统。

{ref}`br_one_recursion` 解释了一个学习递推式的极限是如何由其在静止点附近的线性化所支配的：当线性化后的算法在每个方向上都是压缩的，静止点就具有吸引力。

一整个静止点连续统恰好在一个方向上排除了这一点，表现为一个恰好等于一的特征值。

将两类主体的组合份额同时平移相同的量，并没有任何力量把它们拉回去，因为平移后的点仍是另一个静止点。

为了说明这一点，我们通过有限差分，在实验1的极限处计算该算法两期（每类各更新一次）的雅可比矩阵。

我们将曲率估计 $M$ 固定在其极限值处，这在一阶近似意义下不会丢失任何信息，因为牛顿步骤将 $M^{-1}$ 乘以一个在静止点处消失的梯度。

```{code-cell} ipython3
def two_period_map(p, z, M, gain=0.3):
    "每一类(s, λ)各更新一次，曲率估计固定在M。"
    s, lam = z[:2].copy(), z[2:].copy()
    for jp in (1, 0):                                # 奇数类先更新，然后是偶数类
        j = 1 - jp
        R1 = lam[j]*s[j] / (lam[jp]*s[jp])           # jp年轻时的p1 / 现在的p1
        R2 = (1 - lam[j])*s[j] / ((1 - lam[jp])*s[jp])
        g, _ = grad_hess(p, s[jp], lam[jp], R1, R2)
        step = gain * np.linalg.solve(M, g)
        s[jp], lam[jp] = s[jp] - step[0], lam[jp] - step[1]
    return np.concatenate([s, lam])


def rest_point_jacobian(p, lam, κ=0.5, gain=0.3, ε=1e-6):
    "两期映射在静止点(s*, s*, λ, λ)处的雅可比矩阵。"
    z = np.array([p.s_star, p.s_star, lam, lam])
    _, M = grad_hess(p, p.s_star, lam, 1.0, 1.0)     # 静止点处的海森矩阵，
    M[1, 1] -= κ                                     # 加上脊值：即M的极限
    return np.column_stack([(two_period_map(p, z + ε*d, M, gain)
                             - two_period_map(p, z - ε*d, M, gain)) / (2*ε)
                            for d in np.eye(4)])


J = rest_point_jacobian(kw, lam_lim[0])
eigvals, eigvecs = np.linalg.eig(J)
order = np.argsort(-np.abs(eigvals))
labels = ["s even", "s odd", "λ even", "λ odd"]
print(pd.DataFrame(eigvecs[:, order].real, index=labels,
                   columns=[f"{v:.4f}" for v in eigvals[order].real]).round(3))
```

各列是特征向量，列头标出了对应的特征值。

等于一的那个特征值，属于使两类主体的 $\lambda$ 共同移动的方向：即连续统方向。

接下来的那个特征值，属于使两类主体的组合份额朝相反方向移动的方向，而最小的两个特征值则属于储蓄。

三者都小于一，因此储蓄会回归到 $s^\star$，而两类主体的份额会汇合。

但它们最终汇合在连续统上的哪个位置，则取决于这些稳定方向所遗留下来的东西。

这正是历史的死手，以特征值的形式写出。

不确定性并没有被消除；它被转移到了初始条件和算法的调参之中。

萨金特的结论虽然有所保留，但很明确。

他写道，通过把理性预期均衡所留下的自由的预期固定下来，这种适应性机制选择了一条汇率路径，但是：

> 这似乎是确定汇率的一个薄弱依据。

一条脚注补充道：

> 换句话说，一个使汇率依赖于历史的机制，似乎是一种形式欠佳的机制。

这里的幽灵之所以保持沉默，仅仅是因为牛顿-拉夫森学习器一旦回报相等就不再继续试探。

而一个不断试探下去的学习规则，可能会把它唤醒，正如下文的证据所示。

## 来自实验室的证据

{cite:t}`Arifovic1996` 用 $w_1 = 11$、$w_2 = 1$、$H_1 = H_2 = 10$ 将这个双货币经济作为一项付费人类受试者实验来实施，由此 $s^\star = 5$。

这些并不是上文牛顿-拉夫森部分所用的参数；{ref}`xr_ex4` 在阿里福维奇的参数下运行了牛顿-拉夫森学习器。

每个年轻的受试者选择一个储蓄率，以及在两种货币之间分配的比例；实验者根据这些选择清算两个货币市场，正如 {eq}`xr_prices` 所描述的那样。

阿里福维奇将受试者分成两组人数相等的组别，一组在奇数期年轻，另一组在偶数期年轻。

这些交替出现的组别正是我们所讨论的学习器中奇数类和偶数类在实验室中的对应物。

实验在受试者事先不知道的某个时期结束。

最后一代年轻人持有的货币按前两期价格水平的平均值进行赎回。

结果与牛顿-拉夫森模拟完全不同。

汇率持续波动，大致在 $0.5$ 到 $2$ 之间的区间内，没有任何安定下来的迹象。

如果说有什么变化的话，波动幅度在第二场和第三场实验之间反而增大了。

在萨金特的参数下总是收敛到一个常数值的简单牛顿-拉夫森模型，在这方面表现很差。

因此，阿里福维奇为同一个经济体建立了一个不同的模型，其中主体是由遗传算法培育出来的一个**群体**。

(xr_ga)=
## 遗传算法经济体

{doc}`genetic_classifier` 概述了霍兰德的遗传算法的一般形式；这里我们只需要阿里福维奇所使用的版本。

现在每一类都是 $N = 30$ 个主体组成的群体，每个主体是长度为 $30$ 的**二进制字符串**：前 $20$ 位编码其储蓄率，后 $10$ 位编码其组合份额。

每一期，年轻群体的字符串被解码为 $(s_i, \lambda_i)$ 对，两个货币市场根据 {eq}`xr_prices` 依总量出清，然后读出汇率。

一代之后，当该世代年老时，每个字符串实现的效用 {eq}`xr_utility` 就是其**适应度**，遗传算法据此培育下一代。

```{code-cell} ipython3
N, L_s, L_lam = 30, 20, 10           # 群体规模；储蓄和组合份额的位数

arifovic = Params(w1=11.0, w2=1.0, H1=10.0, H2=10.0)

def decode(p, pop):
    "二进制字符串 → (s, λ)，其中s在(0, w1)中，λ在[0.02, 0.98]中。"
    ints_s = pop[:, :L_s] @ (1 << np.arange(L_s)[::-1])
    ints_l = pop[:, L_s:] @ (1 << np.arange(L_lam)[::-1])
    s = 0.05 + (p.w1 - 0.10) * ints_s / (2**L_s - 1)
    lam = 0.02 + 0.96 * ints_l / (2**L_lam - 1)
    return s, lam

def fitness(p, pop, R1, R2):
    "给定每个字符串所经历的回报，计算其实现的终身效用。"
    s, lam = decode(p, pop)
    c2 = p.w2 + s*(lam*R1 + (1 - lam)*R2)
    return np.log(p.w1 - s) + np.log(np.maximum(c2, 1e-9))
```

这个遗传算法拥有经典的三种算子——父代的轮盘赌**选择**、单点**交叉**和位翻转**变异**——再加上阿里福维奇引入的一种算子，即**选举算子**：只有当一个子代在最近一期的回报下会比其父代更优时，它才被纳入下一代，否则父代得以存续。

我们实现中的一些细节是本讲座自己的选择，不一定是阿里福维奇的做法：

* 每一位的变异概率为 $0.033$；
* 交叉应用于每一对父代；
* 轮盘赌的权重为适应度减去群体中的最小适应度（一种"加窗"轮盘赌，因为对数效用可能为负，所以需要这样处理）；
* $s$ 和 $\lambda$ 的解码范围。

```{code-cell} ipython3
def genetic_step(p, pop, R1, R2, rng, p_mut=0.033, election=True):
    "从当前一代培育出新的一代。"
    fit = fitness(p, pop, R1, R2)
    weight = fit - fit.min() + 1e-6                  # 加窗轮盘赌
    new = np.empty_like(pop)
    for k in range(0, N, 2):
        i, j = np.searchsorted(np.cumsum(weight), rng.random(2) * weight.sum())
        parents = np.array([pop[i], pop[j]])
        cut = rng.integers(1, L_s + L_lam)           # 单点交叉
        kids = parents.copy()
        kids[0, cut:], kids[1, cut:] = parents[1, cut:], parents[0, cut:]
        for c in kids:                               # 变异
            c[rng.random(L_s + L_lam) < p_mut] ^= 1
        if election:                                 # 阿里福维奇的选举算子
            f_kids = fitness(p, kids, R1, R2)
            f_par = fitness(p, parents, R1, R2)
            for m in range(2):
                new[k + m] = kids[m] if f_kids[m] > f_par[m] else parents[m]
        else:
            new[k], new[k + 1] = kids
    return new


def genetic_economy(p, T=3000, seed=0, election=True):
    """
    运行遗传算法经济体。返回汇率e_t，以及对于t时年轻的群体，
    其字符串中s和λ的均值和标准差。
    """
    rng = np.random.default_rng(seed)
    pops = [rng.integers(0, 2, (N, L_s + L_lam)) for _ in range(2)]   # 偶数、奇数
    p1 = np.empty(T)
    p2 = np.empty(T)
    stats = np.empty((T, 4))                         # s均值，s标准差，λ均值，λ标准差
    for t in range(T):
        j = t % 2
        s, lam = decode(p, pops[j])
        p1[t], p2[t] = p.H1 / np.sum(lam*s), p.H2 / np.sum((1 - lam)*s)
        stats[t] = s.mean(), s.std(), lam.mean(), lam.std()
        if t >= 1:
            jp = 1 - j
            R1, R2 = p1[t-1]/p1[t], p2[t-1]/p2[t]
            pops[jp] = genetic_step(p, pops[jp], R1, R2, rng, election=election)
    return p1 / p2, stats
```

### 不会消退的波动

让我们运行这个经济体 $4000$ 期，并舍弃前 $500$ 期作为预烧期。

因为每个字符串的 $\lambda$ 都落在 $[0.02, 0.98]$ 内，且 $H_1 = H_2$，汇率被限制在 $[1/49, 49]$ 之内，因此 $|\log e_t| \leq \ln 49 \approx 3.89$。

图中的虚线标出了这些边界，阴影区域标出了实验范围 $[0.5, 2]$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Exchange rate in the genetic-algorithm economy"
    name: fig-xr-genetic
---
burn = 500
e_ga, stats_ga = genetic_economy(arifovic, T=4000, seed=1)
t_ga = np.arange(burn, len(e_ga))
log_e = np.log(e_ga[burn:])
bound = np.log(0.98 / 0.02)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_ga, log_e, lw=0.5)
ax.axhspan(np.log(0.5), np.log(2), color='C1', alpha=0.2, label="experimental range")
for b in (bound, -bound):
    ax.axhline(b, color='k', ls=':', lw=0.8)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\log e_t$")
ax.legend(frameon=False, loc='upper right')
plt.show()
```

汇率不断波动，而且持续波动下去，其波动范围远远超出了实验范围。

```{code-cell} ipython3
in_band = np.mean((e_ga[burn:] > 0.5) & (e_ga[burn:] < 2))
print(f"e ranges from {e_ga[burn:].min():.3f} to {e_ga[burn:].max():.1f}; "
      f"share of periods in [0.5, 2]: {in_band:.2f}")
print(f"largest |log e| = {np.abs(log_e).max():.2f}, against the bound {bound:.2f}")
print(f"standard deviation of log e, t = 500-1499: {np.log(e_ga[500:1500]).std():.3f}")
print(f"standard deviation of log e, t = 3000-3999: {np.log(e_ga[3000:]).std():.3f}")
```

这种波动是一种永久性特征，而不是一种暂态现象：运行后期的波动幅度大致与早期相当。

而且，它也比实验室中观测到的波动更大。

落在实验区间内的期数不到一半，该序列接近于编码所施加的边界。

萨金特在阿里福维奇的遗传经济体上观察到了相同的现象，其汇率"比阿里福维奇实验中发现的那些汇率更具波动性"。

### 群体学到了什么

与牛顿-拉夫森学习者不同，这个遗传群体从未完全安定下来，因为变异不断注入新的字符串。

但它确实在学习，而它所能学到的和学不到的东西，与 {ref}`xr_ghost` 中的稳定方向和中性方向是一致的。

为了说明这一点，我们来看运行过程中一段时期内，年轻群体 $s$ 和 $\lambda$ 的均值和离散程度。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Saving and portfolio shares inside the genetic population"
    name: fig-xr-ga-population
---
window = np.arange(2000, 2300)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, (m, sd), name in zip(axes, [(0, 1), (2, 3)], ["saving $s$", r"portfolio share $\lambda$"]):
    mean, spread = stats_ga[window, m], stats_ga[window, sd]
    ax.plot(window, mean, lw=1, label="population mean")
    ax.fill_between(window, mean - spread, mean + spread, alpha=0.3,
                    label="± within-population s.d.")
    ax.set_ylabel(name)
axes[0].axhline(arifovic.s_star, color='k', ls='--', lw=0.8, label="$s^\\star$")
axes[0].legend(frameon=False, ncol=3, loc='upper right')
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("$t$")
plt.tight_layout()
plt.show()
```

以下是相同统计量在预烧期之后整个运行期间的平均值。

```{code-cell} ipython3
S = stats_ga[burn:]
print(f"mean saving: {S[:, 0].mean():.3f}   (s* = {arifovic.s_star})")
print(f"within-population s.d. of s: {S[:, 1].mean():.3f}")
print(f"within-population s.d. of λ: {S[:, 3].mean():.3f}")
print(f"s.d. over time of the population's mean λ: {S[:, 2].std():.3f}  "
      f"(a uniform draw on [0.02, 0.98] has s.d. {0.96/np.sqrt(12):.3f})")
print(f"5th and 95th percentiles of the population's mean λ: "
      f"{np.percentile(S[:, 2], 5):.2f}, {np.percentile(S[:, 2], 95):.2f}")
```

这个遗传经济体在一个方向上会收敛，而在另一个方向上则持续漂移。

在确定性的方向上，也就是储蓄，这个群体确实学到了东西：平均储蓄稳定在 $s^\star$ 附近，各字符串彼此高度一致。

在任一时刻，各字符串对 $\lambda$ 的取值也相当一致。

但群体共同的 $\lambda$ 值会在编码所允许的大部分范围内漂移，汇率也随之漂移。

原因在于不确定性的幽灵。

基本面因素没有任何理由偏好某一种组合份额：在回报相等的情况下，任何份额都同样适合。

只有当两种货币的回报出现差异时，某种份额才会比另一种更适合，而这种差异恰恰来自于某一世代的组合与前一世代的组合不同。

因此，除了群体自己产生的回报差距之外，适应度在 $\lambda$ 上是中性的，没有任何东西能固定住群体的组合份额。

变异、交叉以及这些自我产生的回报差距，共同将其沿着均衡连续统推来推去，没有任何力量能使其固定不变。

牛顿-拉夫森学习者将不确定性凝固为一个取决于历史的常数，而在这里，这种不确定性则表现为波动。

选举算子承担了群体大部分的学习工作；{ref}`xr_ex2` 展示了没有选举算子时会发生什么。

### 波动的形态

萨金特将阿里福维奇遗传经济体中的汇率描述为接近随机游走——但带有**均值回复**性质，表现为其一阶差分频谱在零频率处出现一个凹陷。

我们按照萨金特的做法，以弧度为单位、在对数坐标下绘制缩尾处理后的 $\Delta \log e_t$ 频谱随频率变化的图，同时附上 $\log e_t$ 的自相关图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Spectrum and autocorrelation of the exchange rate"
    name: fig-xr-spectrum
---
def averaged_spectrum(x, n_seg=16):
    "巴特利特平均周期图；频率以弧度/期为单位。"
    x = x - x.mean()
    seg = len(x) // n_seg
    acc = np.zeros(seg // 2 + 1)
    for k in range(n_seg):
        acc += np.abs(np.fft.rfft(x[k*seg:(k+1)*seg]))**2 / seg
    return 2 * np.pi * np.fft.rfftfreq(seg), acc / n_seg


d_log_e = np.diff(log_e)
d_log_e = np.clip(d_log_e, *np.percentile(d_log_e, [1, 99]))    # 缩尾处理极端值
freq, spec = averaged_spectrum(d_log_e)

def acf(x, K):
    x = x - x.mean()
    return np.array([np.sum(x[k:]*x[:len(x)-k]) / np.sum(x*x) for k in range(K)])


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(freq[1:], spec[1:], lw=1.2)
axes[0].set_title(r"spectrum of $\Delta \log e$")
axes[0].set_xlabel("frequency (radians)")

axes[1].bar(range(25), acf(log_e, 25))
axes[1].set_title(r"autocorrelation of $\log e$")
axes[1].set_xlabel("lag")
plt.tight_layout()
plt.show()
```

该频谱呈驼峰形。

它在零频率附近较低，在中间频率处升至峰值，然后向 $\pi$ 方向下降，与萨金特图中在 $1.5$ 弧度附近达到峰值的频谱十分相似。

水平值的自相关在约第 $8$ 期滞后处降至一半，到第 $25$ 期滞后处接近于零。

这比近单位根序列所允许的衰减速度快得多。

但样本自相关只是一个非正式的参考，因为随机游走的样本自相关在有限样本中也会出现衰减。

一个更直接的检验方法是，将 $\log e_t$ 对常数项和 $\log e_{t-1}$ 做回归，并比较系数 $\hat\phi$ 与 $1$。

当迪基-福勒统计量 $(\hat\phi - 1)/\mathrm{se}(\hat\phi)$ 低于约 $-2.86$ 时，在 5% 显著性水平上拒绝单位根假设。

```{code-cell} ipython3
def ar1(x):
    "x_t 对 (1, x_{t-1}) 的OLS回归：返回φ̂和迪基-福勒统计量。"
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    b = np.linalg.lstsq(X, x[1:], rcond=None)[0]
    u = x[1:] - X @ b
    se = np.sqrt(u @ u / (len(u) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return b[1], (b[1] - 1) / se


band = slice(1, len(spec)//2)
φ, df_stat = ar1(log_e)
print(f"spectrum of Δlog e at the lowest frequency / mean over (0, π/2): "
      f"{spec[1] / spec[band].mean():.2f}")
print(f"spectral peak at frequency {freq[1:][np.argmax(spec[1:])]:.2f} radians")
print(f"AR(1) coefficient φ = {φ:.3f},  half-life = {np.log(0.5)/np.log(φ):.1f} periods")
print(f"Dickey-Fuller statistic = {df_stat:.1f}   (5% critical value ≈ -2.86)")
```

零频率处的凹陷十分明显。

该频谱在约 $2.2$ 弧度处达到峰值，对应约三期的周期。

水平值表现出强烈的均值回复性，半衰期约为五个半周期，迪基-福勒检验也果断地拒绝了单位根假设；{ref}`xr_ex3` 用其他随机种子验证了这一点。

因此，在我们的实现中，遗传算法得到的汇率是一个平稳序列，而非近似随机游走。

部分均值回复是机械性的：编码对 $\lambda$ 的约束限定了 $|\log e_t|$ 的上限，因此该序列不可能永久漂移下去。

萨金特的评估是：真实的硬通货货币对浮动汇率的对数差分频谱看起来与阿里福维奇的结果非常相似，只是*没有*零频率处的凹陷。

实际汇率更加接近纯粹的随机游走：很难找到一个预测模型能比"不变"这一基准做得更好，而 {cite:t}`KuanLiu1995` 的神经网络预测正是试图超越这一基准。

因此，这个遗传经济体在较高频率的细节特征上与浮动汇率相吻合，但在最低频率上却并不吻合。

它在完全没有基本面冲击的情况下，仅凭一个不断适应的主体群体对两种本质上完全相同的货币重新定价，就能达到这种程度，这才是最引人注目之处。

## 结语

对同一个不确定经济体的两个模型给出了两种截然不同的结论。

牛顿-拉夫森学习者会**收敛**——而且正是在收敛的过程中，暴露了这种不确定性，而不是消除了它。

它的静止点满足使汇率保持自由的套利条件。

这些静止点构成一个连续统，沿着这个连续统，算法具有一个单位特征值。

因此极限落在瞬态过程使经济体停留的任何位置，它取决于初始组合、储蓄率以及算法的调参方式。

学习使汇率依赖于历史路径，但并不会在任何经济学依据上选定某一个汇率。

萨金特有所保留的结论是，这样一种机制"可能是一根脆弱的芦苇"。

遗传算法经济体**不会收敛**，但这并非因为它没有学习到任何东西。

借助阿里福维奇的选举算子，群体学会了一个接近 $s^\star$ 的储蓄率，并变得几乎同质化。

它学不会的是组合份额，因为在收益相等的情况下，没有哪种份额比其他份额更具适应性优势。

因此，群体共同的 $\lambda$ 在均衡的连续统上游走，汇率也随之游走。

不确定性的幽灵以波动的形式重现，而这个经济体根本没有任何基本面扰动。

这种波动正是 {ref}`br_payoffs` 为有限理性研究纲领所列出的收获之一：一种新的动态来源。

它类似于浮动汇率的较高频率纹理，但不类似其低频行为——在低频上，遗传算法经济体表现出的强均值回归在数据中找不到对应物。

本讲座的两部分内容也使用了不同的校准方式。

{ref}`xr_ex4` 在阿里福维奇的校准设定下运行牛顿-拉夫森学习者，发现只有在足够阻尼的情况下它才会收敛。

而在本讲座所用的阻尼程度下，它处于稳定性的边缘，并自行产生反复出现的波动爆发。

经济学家采用哪种学习模型，以及如何对其调参，是有限理性研究纲领被迫摆到台面上来的又一个选择。

{doc}`genetic_classifier` 一般性地阐述了霍兰德的遗传算法，以及其更为丰富的近亲——分类器系统。

随后，{doc}`marimon_mcgrattan_sargent` 让分类器系统构成的群体投入运作：它们不仅要调整组合，还必须*从零开始发现*哪种商品将充当货币。

## 练习

```{exercise-start}
:label: xr_ex1
```

正文部分改变了初始组合份额。

本练习探讨对极限而言，是否还有其他因素起作用。

1. 将（偶数、奇数）两类主体的初始组合份额固定为 $\lambda_0 = (0.30, 0.50)$，并在若干取值上改变初始储蓄率 $s_0$。
   确认两类主体的份额会汇合，且储蓄收敛到 $s^\star$。
   极限汇率是否会随 $s_0$ 变化？
2. 从实验1的初始条件出发，在 $\gamma = 0.3$ 保持不变的情况下，让岭参数 $\kappa$ 在 $(0.1, 0.2, 0.5, 1, 2)$ 之间变化。
   然后在 $\kappa = 0.5$ 保持不变的情况下，让增益 $\gamma$ 在 $(0.05, 0.1, 0.3)$ 之间变化。
   每次都记录极限汇率。

极限汇率是否仅仅是初始组合份额的函数？

```{exercise-end}
```

```{solution-start} xr_ex1
:class: dropdown
```

以下是第1部分。

```{code-cell} ipython3
rows = []
for s0 in ([3.0, 2.2], [2.2, 3.0], [4.0, 4.0], [1.0, 1.0], [6.0, 2.0]):
    e, s_h, l_h = newton_learning(kw, s0, [0.30, 0.50], T=800)
    rows.append([str(s0), l_h[-1][0], l_h[-1][1], s_h[-1].mean(), e[-1]])

pd.DataFrame(rows, columns=["initial s", "λ even (limit)", "λ odd (limit)",
                            "s (limit)", "e (limit)"]).round(4)
```

两类主体的份额会汇合（第2列和第3列一致），且储蓄收敛到 $s^\star = 2.5$。

但即使每一行的初始组合份额都相同，极限汇率也会随初始储蓄率的变化而变化。

$\lambda$ 的梯度 $s(R_1 - R_2)/c_2$ 与储蓄成比例，而暂态期间的回报又取决于两类主体的储蓄。

因此，储蓄走向 $s^\star$ 的路径会决定在套利条件使 $\lambda$ 停止之前，$\lambda$ 究竟移动了多远。

以下是第2部分。

```{code-cell} ipython3
rows = []
for κ in (0.1, 0.2, 0.5, 1.0, 2.0):
    e, s_h, l_h = newton_learning(kw, [3.5, 2.0], [0.35, 0.40], T=2000, κ=κ)
    rows.append([κ, 0.3, l_h[-1][0], e[-1]])
for gain in (0.05, 0.1, 0.3):
    e, s_h, l_h = newton_learning(kw, [3.5, 2.0], [0.35, 0.40], T=4000, gain=gain)
    rows.append([0.5, gain, l_h[-1][0], e[-1]])

pd.DataFrame(rows, columns=["κ", "gain γ", "λ (limit)", "e (limit)"]).round(4)
```

从同一个初始条件出发，极限值会随 $\kappa$ 和 $\gamma$ 两者而变化。

岭参数和增益共同决定了在给定暂态回报差下 $\lambda$ 移动多远，从而决定了运动最终会停在连续区间上的哪个位置。

对于适中的取值，这种影响较为温和，但对于较小的 $\kappa$，由于 $\lambda$ 的步幅较大、暂态也更为剧烈，这种影响就会很大。

因此，极限汇率并不仅仅是初始组合份额的函数。

它还取决于整个初始条件，以及算法的调参方式，这使得这根"脆弱的芦苇"更加脆弱：两位经济学家即使拥有关于初始信念的完全相同的数据，但如果使用不同的学习算法，他们预测出的汇率也会不同。

```{solution-end}
```

```{exercise-start}
:label: xr_ex2
```

阿里福维奇的**选举算子**只在一个子代在最新回报下击败其父代时才将其纳入下一代。

请研究该算子的作用。

在若干个种子值下分别运行带有和不带有该算子的遗传经济体，并针对每种情形报告：

* $\log e_t$ 的标准差；
* 群体的平均储蓄，并与 $s^\star$ 进行比较；
* $s$ 和 $\lambda$ 在群体内部的标准差；
* 群体平均 $\lambda$ 随时间变化的标准差。

该算子将汇率波动性推向哪个方向？为什么？

```{exercise-end}
```

```{solution-start} xr_ex2
:class: dropdown
```

以下是带有和不带有该算子时六个种子值的结果。

```{code-cell} ipython3
rows = []
for election in (True, False):
    for sd in range(6):
        e, st = genetic_economy(arifovic, T=2500, seed=sd, election=election)
        st = st[500:]
        rows.append([election, sd, np.log(e[500:]).std(), st[:, 0].mean(),
                     st[:, 1].mean(), st[:, 3].mean(), st[:, 2].std()])

table = pd.DataFrame(rows, columns=["election", "seed", "s.d. log e", "mean s",
                                    "within s.d. s", "within s.d. λ", "s.d. of mean λ"])
table.drop(columns="seed").groupby("election").agg(["min", "max"]).T.round(3)
```

该表针对每个统计量，报告了其在六个种子值中的最小值和最大值。

选举算子会**提高**汇率波动性，且两种情形下取值范围互不重叠。

其余各行解释了原因。

在没有该算子的情况下，遗传算法学习得少得多。

无论适应度如何，每个子代都会被接纳，因此变异和交叉使群体保持为一个广泛分散的集合。

储蓄在群体内部分布广泛，根据种子值不同，其均值大约在 $4.5$ 到 $5.4$ 之间变化，而 $s^\star = 5$。

组合份额几乎和在 $[0.02, 0.98]$ 上的均匀抽样一样分散，而均匀抽样的标准差为 $0.277$。

{eq}`xr_prices` 中的总量变量是对这一分散群体进行平均的结果，因此总体组合份额、以及随之而来的汇率，相对波动较小。

较低的波动性主要来自对这一分散集合的平均效应。

在开启该算子的情况下，选择机制开始发挥作用。

在每个种子值下，平均储蓄都保持在 $s^\star$ 的约百分之六以内，群体在 $s$ 和 $\lambda$ 两方面都变得近乎同质。

但一个适应度对 $\lambda$ 不敏感的同质化群体，会共同移动其组合份额，而没有任何力量能够锚定它的去向。

因此，群体的平均 $\lambda$ 波动幅度大得多，汇率也随之波动。

使群体得以学习的那个算子，恰恰也是让不确定性显现出来的那个算子：对均衡中可确定部分的学习，暴露出了均衡中不可确定的部分。

```{solution-end}
```

```{exercise-start}
:label: xr_ex3
```

本讲报告称遗传经济体的汇率具有强烈的均值回复性。

请检验这一发现是否依赖于种子值，并检验这一诊断方法本身。

1. 对六个种子值，分别运行遗传经济体 $4000$ 期，剔除前 $500$ 期作为预烧期，并估计 $\log e_t = \mu + \phi \log e_{t-1} + \varepsilon_t$ 中的 $\phi$。
   报告 $\phi$、半衰期 $\ln(1/2)/\ln\phi$，以及迪基-福勒统计量。
   该序列是否平稳？
2. 模拟一个相同长度的纯随机游走，计算其在滞后阶数 $1$、$5$、$10$、$25$ 处的样本自相关系数。
   衰减的样本自相关是否足以将均值回复序列与随机游走区分开来？

```{exercise-end}
```

```{solution-start} xr_ex3
:class: dropdown
```

以下是第1部分。

```{code-cell} ipython3
rows = []
for sd in range(6):
    le = np.log(genetic_economy(arifovic, T=4000, seed=sd)[0][500:])
    φ_sd, df_sd = ar1(le)
    rows.append([sd, φ_sd, np.log(0.5)/np.log(φ_sd), df_sd])

pd.DataFrame(rows, columns=["seed", "φ", "half-life", "Dickey-Fuller"]).round(3)
```

在各个种子值下，$\phi$ 都远小于 $1$，半衰期仅为少数几期，迪基-福勒统计量远低于约 $-2.86$ 的 5% 临界值。

该序列是平稳的，并具有强烈的均值回复性。

以下是第2部分。

```{code-cell} ipython3
rw = np.cumsum(np.random.default_rng(0).standard_normal(3500))
print("sample autocorrelation of a random walk at lags 1, 5, 10, 25:",
      acf(rw, 26)[[1, 5, 10, 25]].round(3))
print("sample autocorrelation of the GA log e at lags 1, 5, 10, 25:   ",
      acf(log_e, 26)[[1, 5, 10, 25]].round(3))
```

随机游走的样本自相关也会衰减，这是因为在有限样本中，自相关是围绕样本均值计算的。

因此，"它会衰减"这一事实本身并不能构成反对单位根的证据。

真正区分这两个序列的是自相关*衰减的速度*，而迪基-福勒统计量正是将这一点转化为了一种正式的检验。

遗传经济体的自相关在 $25$ 个滞后期内几乎降到零，而随机游走的自相关几乎没有变化。

```{solution-end}
```

```{exercise-start}
:label: xr_ex4
```

牛顿-拉夫森部分使用了萨金特的参数 $(w_1, w_2, H_1, H_2) = (20, 15, 100, 120)$，而遗传算法部分使用了阿里福维奇的参数 $(11, 1, 10, 10)$。

为了进行对等的比较，请在阿里福维奇的参数下运行牛顿-拉夫森学习者。

1. 使用本讲的调参设置（$\kappa = 0.5$，$\gamma = 0.3$），从 $s_0 = (6, 4)$、$\lambda_0 = (0.45, 0.55)$ 出发运行 $3000$ 期。
   它是否收敛？
2. 在静止点 $(s^\star, s^\star, \lambda, \lambda)$ 附近进行线性化。
   证明每次更新都会将两类主体组合份额之差乘以 $1 - a$，其中

   $$
   a = \frac{\gamma \, s^\star}{\kappa \, c_2^\star \, \lambda (1 - \lambda)},
   \qquad c_2^\star = w_2 + s^\star .
   $$

   提示：组合回报 $A = \lambda R_1 + (1-\lambda) R_2$ 根本不依赖于组合份额，而在静止点附近，
   $R_1 - R_2 \approx (\lambda_{\text{young}} - \lambda_{\text{old}})/(\lambda(1 - \lambda))$。
   在两种参数校准下分别计算 $a$，并将结果与 `rest_point_jacobian` 进行核对。
3. 在阿里福维奇的参数下，$\kappa$ 需要大到什么程度，才能使某些静止点保持稳定？
   用 $\kappa = 2$ 重新运行第1部分。

```{exercise-end}
```

```{solution-start} xr_ex4
:class: dropdown
```

以下是第1部分。

```{code-cell} ipython3
ea, sa, la = newton_learning(arifovic, [6.0, 4.0], [0.45, 0.55], T=3000)
eb, sb, lb = newton_learning(arifovic, [6.0, 4.0], [0.45, 0.55], T=3000, κ=2.0)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(np.log(ea), lw=0.6, label=r"$\kappa = 0.5$")
axes[0].plot(np.log(eb), lw=1.2, label=r"$\kappa = 2$")
axes[0].set_ylabel(r"$\log e_t$")
axes[0].legend(frameon=False)
axes[1].plot(la[:, 0] - la[:, 1], lw=0.6)
axes[1].plot(lb[:, 0] - lb[:, 1], lw=1.2)
axes[1].set_ylabel(r"$\lambda_{\mathrm{even}} - \lambda_{\mathrm{odd}}$")
axes[1].set_xlabel("$t$")
plt.tight_layout()
plt.show()
```

在本讲的调参设置下，学习者并不收敛。

两类主体的组合份额几乎汇合，储蓄达到 $s^\star = 5$，汇率也暂时趋于稳定。

但随后两类主体份额之间的差距会以交替变化的方式不断扩大，直至发散。

汇率会冲向边界，随后又在别处重新稳定下来，这样的过程不断重复。

对于第2部分，设 $\delta$ 为年轻一代的份额减去年老一代的份额。

组合回报为
$A = \lambda_{\text{old}} R_1 + (1 - \lambda_{\text{old}}) R_2 = s_{\text{young}}/s_{\text{old}}$，
它不涉及份额。

因此储蓄方程根本不涉及 $\lambda$。

在静止点附近，交叉项 $\partial^2 U/\partial s \partial \lambda$ 与 $R_1 - R_2$ 成比例，并趋于零。

曲率估计中 $\lambda$ 对应的项随之变为 $-\kappa$，于是年老一代的更新为

$$
\lambda_{\text{old}} \leftarrow \lambda_{\text{old}} + \frac{\gamma}{\kappa}\,
\frac{s^\star}{c_2^\star} (R_1 - R_2)
\approx \lambda_{\text{old}} + a\,\delta .
$$

下一期两类主体的角色互换，因此新的"年轻减年老"之差为 $-(1 - a)\delta$。

以偶数类减奇数类来度量，每次更新都会使这一差距乘以 $1 - a$，经过两期后乘以 $(1 - a)^2$。

这正是我们在 {ref}`xr_ghost` 中遇到过的特征值。

若 $0 < a < 2$，差距会消失；若 $a > 2$，差距则会发散。

```{code-cell} ipython3
def a_coef(p, lam, κ=0.5, gain=0.3):
    return gain * p.s_star / (κ * (p.w2 + p.s_star) * lam * (1 - lam))


for p, name in [(kw, "Sargent"), (arifovic, "Arifovic")]:
    for lam in (0.3, 0.5, 0.7):
        a = a_coef(p, lam)
        ev = np.sort(np.abs(np.linalg.eigvals(rest_point_jacobian(p, lam))))[::-1]
        print(f"{name:8s} λ = {lam}:  a = {a:.3f},  (1-a)² = {(1 - a)**2:.3f},  "
              f"|eigenvalues| = {ev.round(3)}")
```

该公式与数值求得的雅可比矩阵相吻合。

在萨金特的参数下，除非 $\lambda$ 低于约 $0.05$ 或高于约 $0.95$，否则 $a$ 都远低于 $2$，因此除了连续区间上那个单位特征值之外，各静止点都是稳定的。

在阿里福维奇的参数下，$\lambda$ 的梯度要陡峭得多，因为比值 $s^\star/c_2^\star$ 为 $5/6$，而不是 $1/7$。

于是在 $\kappa = 0.5$、$\gamma = 0.3$ 的情况下，*每一个*静止点上都有 $a = 0.5/(\lambda(1-\lambda)) \geq 2$。

$\lambda = 1/2$ 处的静止点恰好处于边界上，而其余所有静止点都是不稳定的。

这正是学习者不断逃离静止点的原因。

对于第3部分，由于 $\lambda(1 - \lambda) \leq 1/4$，在阿里福维奇的参数下，当且仅当 $\kappa > 2\gamma s^\star / c_2^\star = 0.5$ 时，才会存在稳定的静止点。

本讲所用的 $\kappa$ 恰好处在这一临界值上。

当 $\kappa = 2$ 时，满足 $\lambda(1-\lambda) > 1/16$（即 $\lambda$ 大致在 $0.07$ 到 $0.93$ 之间）的静止点是稳定的，图中也显示学习者收敛到了一个常数。

```{code-cell} ipython3
print(f"κ = 2: e → {eb[-1]:.4f},  s → {sb[-1].round(3)},  λ → {lb[-1].round(4)}")
print(f"largest |Δ log e| over the last 500 periods: "
      f"{np.abs(np.diff(np.log(eb[-500:]))).max():.1e}")
```

因此，本讲前后两部分之间的差异，根源在于自适应学习机制本身，而非参数校准。

一个阻尼足够大的牛顿-拉夫森学习者，在阿里福维奇的参数下也能收敛，收敛到一个依赖于历史的常数。

但多大的阻尼才算"足够"，取决于具体的经济体：在相同的调参下，一种参数校准收敛，而另一种却发散。

```{solution-end}
```

```{exercise-start}
:label: xr_ex5
```

萨金特在一条脚注中区分了"历史"依赖性与"路径"依赖性。

在该模型的随机版本中，从完全相同的初始条件出发的实现，可能收敛到不同的汇率，因为不同的冲击会作用于暂态动态。

请在萨金特的参数下，向老年期禀赋中加入独立同分布的冲击，即 $w_{2t} = w_2 + \sigma \varepsilon_t$，其中 $\varepsilon_t \sim N(0, 1)$，$\sigma = 1$。

（`newton_learning` 的 `w2_path` 参数可实现这一点。）

1. 从相同的初始条件出发，即 $s_0 = (3.0, 2.2)$ 和 $\lambda_0 = (0.30, 0.50)$，让 $200$ 个经济体分别采用不同的冲击序列，每个运行 $600$ 期。
   在每次运行中，汇率是否收敛？储蓄是否收敛？
2. 绘制极限汇率的直方图，并标出无冲击情形下的极限值。
3. 尽管冲击从未停止作用于经济体，为什么它们在一段时间之后就不再影响汇率了？

```{exercise-end}
```

```{solution-start} xr_ex5
:class: dropdown
```

以下是第1和第2部分。

```{code-cell} ipython3
T, σ = 600, 1.0
e_det = newton_learning(kw, [3.0, 2.2], [0.30, 0.50], T=T)[0][-1]

limits, last_move, s_spread = [], [], []
for seed in range(200):
    shocks = σ * np.random.default_rng(seed).standard_normal(T)
    e, s_h, l_h = newton_learning(kw, [3.0, 2.2], [0.30, 0.50], T=T,
                                  w2_path=kw.w2 + shocks)
    limits.append(e[-1])
    last_move.append(np.abs(np.diff(np.log(e[-300:]))).max())
    s_spread.append(s_h[-300:, 0].std())

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(limits, bins=30, alpha=0.7)
ax.axvline(e_det, color='k', ls='--', lw=1, label="no shocks")
ax.set_xlabel("limiting exchange rate")
ax.legend(frameon=False)
plt.show()
```

以下是各次运行结果的汇总。

```{code-cell} ipython3
limits = np.array(limits)
print(f"limit without shocks: {e_det:.4f}")
print(f"limits with shocks: {limits.min():.4f} to {limits.max():.4f}, s.d. {limits.std():.4f}")
print(f"largest |Δ log e| over the last 300 periods, across all runs: {max(last_move):.1e}")
print(f"s.d. of even-class saving over the last 300 periods: "
      f"{min(s_spread):.3f} to {max(s_spread):.3f}")
```

每一条汇率路径都会收敛：在最后 $300$ 期内，除舍入误差外，汇率完全不再变动。

储蓄则不会收敛；由于增益恒定且冲击从未停止，储蓄会持续围绕 $s^\star$ 波动。

然而，这些仅在冲击上有所不同的经济体，却分别稳定在了不同的汇率上。

这正是萨金特意义上的路径依赖性。

{cite:t}`Sargent1993` 引用了埃文斯和洪卡波亚对自适应学习如何解决不确定性问题的描述：

> 具有多重解的模型被转化为具有路径依赖性的模型，在这些模型中，经济体的运行轨迹，以及在极限中所达到的[理性预期均衡]，是通过一条学习规则，由初始预测和外生冲击序列共同决定的。

冲击只在暂态期间才会产生影响，因为汇率仅取决于年轻一代的 $\lambda$。

一旦两类主体的份额汇合，无论储蓄如何变化都有 $R_1 = R_2$，因此 $\lambda$ 的梯度 $s(R_1 - R_2)/c_2$ 恒为零，与禀赋无关。

此后，对 $w_2$ 的冲击只会改变储蓄，却无法改变组合份额，而汇率则停留在暂态结束时所处的位置。

由于冲击只有短暂的暂态期可以发挥作用，极限值在无冲击基准值附近的分布范围较窄。

```{solution-end}
```
