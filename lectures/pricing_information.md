---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 信息的设计与定价
  headings:
    Overview: 概述
    The decision problem: 决策问题
    Experiments and their value: 实验及其价值
    Blackwell's order is only partial: 布莱克韦尔的序只是偏序
    The seller's problem: 卖方的问题
    Two types: 两种类型
    Two types::Noncongruent types: 不一致类型
    A continuum of types: 连续类型
    A continuum of types::Solving it as a linear program: 作为线性规划求解
    A continuum of types::Uniformly distributed types: 均匀分布的类型
    A continuum of types::Bimodal types and the case for versioning: 双峰类型与版本化的理由
    Concluding remarks: 结束语
    Exercises: 练习
---

(pricing_information)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 信息的设计与定价

```{index} single: Information; pricing
```

```{index} single: Blackwell; and screening
```

```{contents} Contents
:depth: 2
```

## 概述

本节前面的讲座探讨了决策者*应该*偏好哪种统计实验的问题。

{doc}`blackwell_kihlstrom` 给出了经典的答案：当*每一个*贝叶斯决策者在实验 $\mu$ 下获得的预期效用弱高于实验 $\nu$ 时，实验 $\mu$ 至少与实验 $\nu$ 一样有信息量。

本讲座提出了一个不同的问题。

假设有人*拥有*数据并想要出售它。

她应该出售什么，定价多少？

我们研究 {cite:t}`BergemannBonattiSmolin2018` 的工作，他们分析了一个垄断数据卖方，面对一个已经拥有自己私人信息的买方。

买方的私人信息恰恰是他想要隐藏的东西，因为它决定了他的支付意愿。

因此卖方通过提供一份统计实验的*菜单*来进行筛选，为了对某些买方收取更高价格，而降低出售给其他买方的信息质量。

核心发现是，降低信息质量不仅仅是添加噪声这么简单。

布莱克韦尔的序是一个*偏*序，因此两个实验可能被不同的决策者以不同方式排序。

卖方正是利用了这些不可比较的对：信息具有一个**纵向**维度（其质量）和一个**横向**维度（其位置）。

这个横向维度在关于质量或数量的普通垄断筛选中没有对应物，正是它使卖方能够攫取本来无法达到的租金。

在这个过程中我们将会：

- 计算任意实验对任意信念类型的价值，
- 数值验证布莱克韦尔的序无法对卖方想要使用的实验进行排序，
- 通过暴力法求解两类型筛选问题，并与论文的封闭形式解进行核对，
- 将连续类型问题作为**线性规划**求解，这重现了论文的熨平（ironing）和混同结果，而无需手动实现熨平过程。

让我们从导入开始。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import linprog

plt.rcParams['figure.figsize'] = (10, 5)
np.set_printoptions(precision=4, suppress=True)
```

## 决策问题

数据买方必须在不知道状态 $\omega$（属于有限集合 $\Omega$）的情况下，从有限集合 $A$ 中选择一个行动 $a$。

我们始终在**匹配**情形下工作，在此情形中买方希望自己的行动与状态匹配，

$$
u(\omega_i, a_j) = \mathbb{1}[i = j] \cdot u_i ,
$$ (eq:pi_matching)

因此匹配状态 $\omega_i$ 支付 $u_i > 0$，而任何不匹配则支付零。

从这里开始，我们采用两个状态和两个行动，$\Omega = \{\omega_1, \omega_2\}$ 和 $A = \{a_1, a_2\}$，这正是 {cite:t}`BergemannBonattiSmolin2018` 完整求解的情形。

买方的**类型**是他的临时信念

$$
\theta = \Pr[\omega = \omega_1] \in [0, 1] ,
$$

这是私人信息。

卖方只知道 $\theta$ 抽取自的分布 $F$。

没有额外信息时，买方在两个固定行动中选择较好的那个，因此他的保留效用为

$$
u(\theta) = \max\{\theta u_1,\ (1 - \theta) u_2\} .
$$ (eq:pi_outside)

*最不*确定该怎么做的类型是使得这两项相等的类型，

$$
\theta^* = \frac{u_2}{u_1 + u_2} .
$$ (eq:pi_thetastar)

高于 $\theta^*$ 的类型自己会选择 $a_1$，低于此值的类型会选择 $a_2$。

```{note}
买方的信念 $\theta$ 可以由一个共同先验和一个私下观察到的信号共同生成，正如 {doc}`likelihood_bayes` 和 {doc}`exchangeable` 中那样。

拥有非常精确私人信号的买方的 $\theta$ 接近 $0$ 或 $1$；一个什么都没学到的买方处于共同先验的位置，而共同先验仅在巧合情况下才等于 $\theta^*$。

本讲座中的"高类型"是指信念接近 $\theta^*$ 的买方，他正是最愿意支付的人。

当共同先验恰好等于 $\theta^*$ 时，高类型就是信息不佳的买方。
```

## 实验及其价值

统计实验是一个将状态映射到信号的随机矩阵。

在两个状态和两个行动的情形下，考虑两个信号就足够了，我们记

$$
E = \begin{pmatrix} \pi_1 & 1 - \pi_1 \\ 1 - \pi_2 & \pi_2 \end{pmatrix},
$$ (eq:pi_experiment)

其中第 $i$ 行给出状态 $\omega_i$ 下的信号分布。

因此 $\pi_1 = \Pr[s_1 \mid \omega_1]$ 且 $\pi_2 = \Pr[s_2 \mid \omega_2]$。

我们采用归一化 $\pi_1 + \pi_2 \geq 1$，这只是说信号 $s_1$ 在状态 $\omega_1$ 中相对于状态 $\omega_2$ 更有可能出现。

**完全信息**实验 $\overline{E}$ 满足 $\pi_1 = \pi_2 = 1$。

在看到信号 $s_k$ 之后，买方选择预期收益最高的行动，因此他的总价值通过逐个信号求和其能做到的最好表现来获得。

减去他的保留效用 {eq}`eq:pi_outside`，得到**信息的净价值**

$$
V(E, \theta)
= \max\{\theta \pi_1 u_1,\ (1-\theta)(1-\pi_2) u_2\}
+ \max\{\theta (1-\pi_1) u_1,\ (1-\theta)\pi_2 u_2\}
- \max\{\theta u_1,\ (1-\theta) u_2\} .
$$ (eq:pi_value)

```{code-cell} ipython3
def value(pi1, pi2, theta, u1=1.0, u2=1.0):
    """Net value of experiment (pi1, pi2) to a buyer with belief theta."""
    theta = np.asarray(theta, dtype=float)
    s1 = np.maximum(theta * pi1 * u1, (1 - theta) * (1 - pi2) * u2)
    s2 = np.maximum(theta * (1 - pi1) * u1, (1 - theta) * pi2 * u2)
    return s1 + s2 - np.maximum(theta * u1, (1 - theta) * u2)
```

如果买方简单地服从每个信号隐含的推荐，在信号 $s_1$ 后采取 $a_1$，在信号 $s_2$ 后采取 $a_2$，则该价值简化为

$$
V(E, \theta) = \max\bigl\{\theta \pi_1 u_1 + (1-\theta)\pi_2 u_2
- \max\{\theta u_1, (1-\theta)u_2\},\ 0\bigr\} ,
$$ (eq:pi_value_obedient)

这正是论文所使用的表达式。

```{code-cell} ipython3
def value_obedient(pi1, pi2, theta, u1=1.0, u2=1.0):
    """Value when the buyer follows the recommendation, or ignores the signal."""
    theta = np.asarray(theta, dtype=float)
    return np.maximum(theta * pi1 * u1 + (1 - theta) * pi2 * u2
                      - np.maximum(theta * u1, (1 - theta) * u2), 0.0)
```

在归一化 $\pi_1 + \pi_2 \geq 1$ 下，这两个表达式完全一致，而在没有该归一化时它们可能差别很大，这正是该归一化的用途。

```{code-cell} ipython3
grid = np.linspace(0, 1, 2001)
worst_ok = worst_bad = 0.0
for p1 in np.linspace(0, 1, 51):
    for p2 in np.linspace(0, 1, 51):
        gap = np.abs(value(p1, p2, grid) - value_obedient(p1, p2, grid)).max()
        if p1 + p2 >= 1:
            worst_ok = max(worst_ok, gap)
        else:
            worst_bad = max(worst_bad, gap)

print(f'largest gap where pi1 + pi2 >= 1:  {worst_ok:.2e}')
print(f'largest gap where pi1 + pi2 <  1:  {worst_bad:.4f}')
```

在该归一化下，反转推荐——在信号 $s_1$ 后采取 $a_2$，在信号 $s_2$ 后采取 $a_1$——的效果劣于直接忽略信号，因此每种类型要么服从，要么忽略。

这就是为什么这两个公式对每一个 $\theta$ 都完全一致，包括那些谎报类型、购买为他人设计的实验的买方。

从此我们使用一般形式 {eq}`eq:pi_value`，因为它不依赖于该归一化。

以下是信息价值作为买方类型的函数。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Value of full and partial information
    name: fig-pi-value
---
theta = np.linspace(0, 1, 1001)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (p1, p2), ttl in zip(
        axes, [(1.0, 1.0), (0.5, 1.0)],
        [r'fully informative $(\pi_1,\pi_2)=(1,1)$',
         r'partially informative $(\pi_1,\pi_2)=(1/2,1)$']):
    ax.plot(theta, value(p1, p2, theta), lw=2)
    ax.axvline(0.5, color='0.6', ls='--', lw=1)
    ax.set(xlabel=r'$\theta$', ylabel=r'$V(E,\theta)$', title=ttl)
fig.suptitle('Value of information, $u_1 = u_2 = 1$')
fig.tight_layout()
plt.show()
```

这些图形有三个特征推动了后续所有内容。

价值在 $\theta$ 上是**分段线性**的，因为类型是概率，而预期效用在概率上是线性的。

价值在 $\theta^*$ 处**最高**，在 $\theta \in \{0, 1\}$ 处降为零：已经知道状态的买方将什么都不支付，而最不知道的买方会支付最多。

右图中部分信息的实验对高于 $2/3$ 的类型来说**完全没有价值**，尽管它对刚低于 $1/2$ 的类型来说价值很大。

最后这个特性是卖方的主要工具。

## 布莱克韦尔的序只是偏序

{doc}`blackwell_kihlstrom` 确立了实验 $E$ 在布莱克韦尔意义上至少与 $E'$ 一样具有信息量，当且仅当 $E'$ 是 $E$ 的**信息降级**（garbling）。

按照 {prf:ref}`def-sufficiency` 中的记号，令 $\mu = E$ 且 $\nu = E'$，则存在一个马尔可夫矩阵 $Q$ 满足

$$
E' = E Q .
$$ (eq:pi_garbling)

当 $E$ 可逆时，这很容易检验：求解 $Q = E^{-1}E'$ 并检查 $Q$ 的每个元素是否都非负。

$Q$ 的各行之和自动为一：$E Q \mathbf{1} = E' \mathbf{1} = \mathbf{1} = E \mathbf{1}$，而 $E$ 是可逆的，所以 $Q \mathbf{1} = \mathbf{1}$。

```{code-cell} ipython3
def experiment(pi1, pi2):
    return np.array([[pi1, 1 - pi1], [1 - pi2, pi2]])


def garbling(E, Ep, tol=1e-9):
    """Return Q with Ep = E @ Q if Ep is a garbling of E, else None."""
    if abs(np.linalg.det(E)) < tol:
        return None
    Q = np.linalg.solve(E, Ep)           # rows of Q sum to one automatically
    if (Q > -tol).all():
        return Q
    return None


pairs = [((1, 1), (0.8, 1)), ((1, 1), (1, 0.8)),
         ((0.9, 0.9), (0.8, 0.8)),
         ((0.8, 1), (1, 0.8)), ((1, 0.8), (0.8, 1))]
for a, b in pairs:
    ok = garbling(experiment(*a), experiment(*b)) is not None
    print(f'  is {b} a garbling of {a}?  {"yes" if ok else "no"}')
```

完全信息实验可以降级为任何东西，而 $(0.9, 0.9)$ 可以降级为噪声更均匀的 $(0.8, 0.8)$。

这些是**纵向**比较，布莱克韦尔定理说所有类型对此都一致同意。

但 $(0.8, 1)$ 和 $(1, 0.8)$ 双方都不能互相降级。

布莱克韦尔的序对它们根本无法排序，这意味着不同的类型可以自由地对它们做出不同的排序。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Two experiments that Blackwell's order does not rank
    name: fig-pi-blackwell
---
va, vb = value(0.8, 1, theta), value(1, 0.8, theta)

fig, ax = plt.subplots()
ax.plot(theta, va, lw=2, label=r'$E_a = (0.8, 1)$')
ax.plot(theta, vb, lw=2, label=r'$E_b = (1, 0.8)$')
ax.fill_between(theta, va, vb, where=va > vb, alpha=0.15, color='C0')
ax.fill_between(theta, va, vb, where=vb > va, alpha=0.15, color='C1')
ax.axvline(0.5, color='0.4', ls='--', lw=1)
ax.set(xlabel=r'$\theta$', ylabel=r'$V(E,\theta)$',
       title='Types below $1/2$ prefer $E_a$, types above prefer $E_b$')
ax.legend()
fig.tight_layout()
plt.show()

for t in [0.2, 0.35, 0.65, 0.8]:
    pref = 'E_a' if value(0.8, 1, t) > value(1, 0.8, t) else 'E_b'
    print(f'  theta = {t}:  V(E_a) = {value(0.8, 1, t):.4f},'
          f'  V(E_b) = {value(1, 0.8, t):.4f}   prefers {pref}')
```

$E_a$ 的信号 $s_1$ 只在状态 $\omega_1$ 下出现，因此它能排除 $\omega_2$；$E_b$ 的信号 $s_2$ 只在状态 $\omega_2$ 下出现，因此它能排除 $\omega_1$。

一个倾向于认为 $\omega_1$ 更可能的买方会自行选择 $a_1$，只有在出现有力证据支持 $\omega_2$ 时才转而选择 $a_2$。

$E_b$ 恰好能为 $\omega_2$ 提供确凿的证据，因此他重视 $E_b$。

出于同样的原因，倾向于 $\omega_2$ 的买方则重视 $E_a$。

这就是信息的**横向**维度。

在关于质量或数量的普通非线性定价中，所有类型都认同产品的排序，卖方只能沿单一阶梯上下移动。

而这里卖方拥有第二个调节旋钮，转动它就能让一种类型得到对另一种类型毫无价值的东西。

## 卖方的问题

卖方承诺提供一份菜单 $\{E(\theta), t(\theta)\}$，为每个申报的类型分配一个实验和一个价格。

支付不能以状态、信号或买方的行动为条件，因此一个实验对买方的价值仅由他的信念决定。

将买方的租金写作 $V(\theta) = V(E(\theta), \theta) - t(\theta)$，卖方求解

$$
\max_{\{E(\theta),\, t(\theta)\}} \int t(\theta) \, dF(\theta)
$$ (eq:pi_sellerproblem)

受制于激励相容和个人理性约束，

$$
V(\theta) \geq V(E(\theta'), \theta) - t(\theta') \ \ \forall \theta, \theta',
\qquad
V(\theta) \geq 0 \ \ \forall \theta .
$$ (eq:pi_icir)

{cite:t}`BergemannBonattiSmolin2018` 建立了一些结构性结果，我们计算的每一份菜单都证实了这些结果。

```{prf:proposition}
:label: pi_prop_structure

1. 完全信息实验 $\overline{E}$ 是最优菜单的一部分。
2. 任何最优菜单中的每个实验都是**非分散的**，意味着对某个 $i \neq j$ 有 $\pi_{ij} = 0$。
3. 在匹配情形下，任何最优菜单中的每个实验都是**集中的**，意味着对某个 $i$ 有 $\pi_{ii} = 1$。
```

第3点是说，在我们的二元设置中，菜单中的每个实验都有 $\pi_1 = 1$ 或 $\pi_2 = 1$。

在两个状态和两个信号的情形下，某个非对角元素为零会迫使同一行的对角元素为一，因此“非分散的”与“集中的”这两个概念是一致的。

最优实验保留一个能完全揭示某一状态的信号；另一个信号则带有噪声。

## 两种类型

取两个类型 $\theta^L$ 和 $\theta^H$，其中 $\theta^H$ 是*高价值*类型，含义是他更重视完全信息实验，

$$
V(\overline{E}, \theta^H) \geq V(\overline{E}, \theta^L) .
$$

当 $u_1 = u_2$ 时，这表明 $|\theta^H - 1/2| \leq |\theta^L - 1/2|$，因此高类型是那个最初信息*较少*的类型。

令 $\gamma = \Pr[\theta = \theta^H]$。

类型是**一致的**，如果 $\theta^* < \theta^H < \theta^L$，即两者都会在没有额外信息的情况下选择同一行动；类型是**不一致的**，如果 $\theta^L < \theta^* < \theta^H$。

最优菜单具有熟悉的形式：高类型购买 $\overline{E}$，低类型的参与约束绑定，高类型的激励约束绑定。

一旦选定低类型的实验，这三个事实就确定了两个价格。

要在*所有*低类型实验上进行搜索，我们还需要为那些高类型比低类型更不看重的实验确定价格。

对于这些实验，使高类型激励约束绑定的价格会超过他对 $\overline{E}$ 的价值，因此这时绑定的是他的参与约束。

下面的函数会向高类型收取这两个价格中较小的一个。

```{code-cell} ipython3
def two_type_revenue(pi1, pi2, tL, tH, gamma, u1=1.0, u2=1.0):
    """Revenue when the high type buys E_bar and the low type buys (pi1, pi2)."""
    VbarH, VbarL = value(1, 1, tH, u1, u2), value(1, 1, tL, u1, u2)
    VL_L, VL_H = value(pi1, pi2, tL, u1, u2), value(pi1, pi2, tH, u1, u2)
    t_low = VL_L                                  # low type's IR binds
    t_high = min(VbarH - VL_H + t_low,            # high type's IC
                 VbarH)                           # high type's IR
    if VbarL - t_high > 1e-12:                    # low type must not deviate
        return -np.inf
    return gamma * t_high + (1 - gamma) * t_low
```

### 不一致类型

设 $\theta^L = 1/5$ 且 $\theta^H = 7/10$，$u_1 = u_2 = 1$，因此 $\theta^* = 1/2$ 位于两者之间。

因为这两种类型自己会选择*不同*的行动，卖方可以构建一个对一种类型有价值而对另一种类型毫无价值的实验。

选取 $\pi_2 = 1$ 和

$$
\pi_1' = \frac{u_1 \theta^H - u_2 (1 - \theta^H)}{u_1 \theta^H}
$$ (eq:pi_zerovalue)

使高类型在信号 $s_2$ 后对他的两个行动恰好无差异，因此该实验对他毫无价值，而低类型则严格地重视它。

这是可行的但不是最优的。

卖方通过让高类型的激励约束改为绑定做得更好，这给出

$$
\pi_1'' = \frac{u_1 \theta^H - u_2 (1 - \theta^H)}{u_1 (\theta^H - \theta^L)} .
$$ (eq:pi_optimal2type)

```{code-cell} ipython3
tL, tH, u1, u2 = 0.2, 0.7, 1.0, 1.0
pi1_zero = (u1 * tH - u2 * (1 - tH)) / (u1 * tH)
pi1_opt = (u1 * tH - u2 * (1 - tH)) / (u1 * (tH - tL))

print(f'zero-value-to-high experiment   pi1 = {pi1_zero:.4f}  (= 4/7)')
print(f'binding-IC experiment           pi1 = {pi1_opt:.4f}  (= 4/5)')
print(f'\n  V(E_zero, theta_H) = {value(pi1_zero, 1, tH):.4f}')
print(f'  V(E_opt,  theta_L) = {value(pi1_opt, 1, tL):.4f}'
      f'    V(E_opt, theta_H) = {value(pi1_opt, 1, tH):.4f}')
```

第二个实验给*两种*类型带来相同的信息净价值，即 $0.16$。

卖方向低类型收取 $0.16$ 作为这个实验的价格，而向高类型收取他对 $\overline{E}$ 的全部价值。

高类型的激励约束绑定，两种类型的参与约束也都绑定。

卖方从两种类型那里榨取了全部剩余；筛选的唯一代价是对低类型实验的扭曲。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Net value of the two menus as a function of the buyer's type
    name: fig-pi-menus
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, p1, ttl in zip(axes, [pi1_zero, pi1_opt],
                       ['suboptimal menu: partial experiment worth zero to $\\theta^H$',
                        'optimal menu: high type indifferent between the two items']):
    t_hi = value(1, 1, tH)                                     # price of E_bar
    t_lo = value(p1, 1, tL)                                    # price of partial item
    ax.plot(theta, value(1, 1, theta) - t_hi, lw=2, label='fully informative')
    ax.plot(theta, value(p1, 1, theta) - t_lo, lw=2, ls='--',
            label=f'partial, $\\pi_1={p1:.3f}$')
    ax.axhline(0, color='0.3', lw=1)
    for t, nm in [(tL, r'$\theta^L$'), (tH, r'$\theta^H$')]:
        ax.axvline(t, color='0.7', ls=':', lw=1)
        ax.annotate(nm, (t, ax.get_ylim()[0]), fontsize=9)
    ax.set(xlabel=r'$\theta$', ylabel=r'$V - t$', title=ttl, ylim=(-0.35, 0.25))
    ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
plt.show()
```

在左图中，高类型对完全信息实验的净价值严格高于他对部分实验的净价值，因此他的激励约束是松弛的，卖方正在把钱留在桌上。

在右图中，两条曲线恰好在 $\theta^H$ 处相交。

现在我们将闭式解 {eq}`eq:pi_optimal2type` 与对*所有*实验的暴力搜索进行比较。

```{code-cell} ipython3
def brute_force(tL, tH, gamma, n=301, u1=1.0, u2=1.0):
    """Search over all (pi1, pi2) for the best low-type experiment."""
    g = np.linspace(0, 1, n)
    best, arg = -np.inf, None
    for p1 in g:
        for p2 in g:
            if p1 + p2 < 1:
                continue
            r = two_type_revenue(p1, p2, tL, tH, gamma, u1, u2)
            if r > best:
                best, arg = r, (p1, p2)
    return best, arg


print(f'{"gamma":>7s}{"brute force":>13s}{"argmax":>18s}'
      f'{"closed form":>14s}{"E_bar to both":>15s}')
for gamma in [0.10, 0.25, 0.30, 0.50, 0.90]:
    best, arg = brute_force(tL, tH, gamma)
    closed = two_type_revenue(pi1_opt, 1.0, tL, tH, gamma)
    both = two_type_revenue(1.0, 1.0, tL, tH, gamma)
    print(f'{gamma:7.2f}{best:13.5f}   ({arg[0]:.3f}, {arg[1]:.3f})'
          f'{closed:14.5f}{both:15.5f}')
print(f'\nthe paper: discriminate iff gamma > theta_L / theta_H = {tL / tH:.4f}')
```

只要价格歧视是有利可图的，暴力搜索的最优点就落在 $(\pi_1, \pi_2) = (0.8, 1)$ 处，与 {eq}`eq:pi_optimal2type` 完全吻合；否则落在 $(1, 1)$。

这个转换恰好发生在 $\gamma = \theta^L / \theta^H$ 处。

当低类型比较常见时，卖方倾向于以较低价格将完全信息实验卖给所有人；当高类型比较常见时，卖方倾向于通过降低低类型获得的信息来保护高价格。

还请注意，最优菜单中的两个实验都有 $\pi_2 = 1$，这证实了 {prf:ref}`pi_prop_structure` 的第3点。

## 连续类型

现在设 $\theta$ 分布在 $[0,1]$ 上，密度为 $f$，分布函数为 $F$。

关键的简化在于：一个*集中*实验的价值仅通过标量

$$
q = \pi_1 u_1 - \pi_2 u_2 \in [-u_2,\ u_1] ,
$$ (eq:pi_q)

来依赖于 $(\pi_1, \pi_2)$，{cite:t}`BergemannBonattiSmolin2018` 称之为该实验的**差异性信息量**。

根据 {prf:ref}`pi_prop_structure` 的第3部分，将卖方限制为只提供集中实验不会造成任何损失。

对于其他实验，$q$ 并不能决定其价值：当 $u_1 = u_2 = 1$ 时，实验 $(0.8, 0.8)$、$(0.9, 0.9)$ 和 $(1, 1)$ 都有 $q = 0$，但在 $\theta = 1/2$ 处，它们的价值分别为 $0.3$、$0.4$ 和 $0.5$。

用 $q$ 表示，该价值变为

$$
V(q, \theta) = \max\bigl\{\theta q + u_2 + \min\{u_1 - u_2 - q,\ 0\}
- \max\{\theta u_1,\ (1-\theta) u_2\},\ 0 \bigr\} .
$$ (eq:pi_valueq)

完全信息实验对应 $q = u_1 - u_2$。

两个端点 $q = -u_2$ 和 $q = u_1$ 是这样的实验：某个信号在两个状态下都以概率一发生，因此不传递任何信息。

```{code-cell} ipython3
def value_q(q, theta, u1=1.0, u2=1.0):
    """Value of the experiment with differential informativeness q."""
    theta = np.asarray(theta, dtype=float)
    gross = theta * q + u2 + np.minimum(u1 - u2 - q, 0.0)
    return np.maximum(gross - np.maximum(theta * u1, (1 - theta) * u2), 0.0)


def q_to_experiment(q, u1=1.0, u2=1.0):
    """Recover (pi1, pi2) from q using pi1 = 1 or pi2 = 1."""
    return (1.0, (u1 - q) / u2) if q >= u1 - u2 else ((q + u2) / u1, 1.0)


for q in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    p1, p2 = q_to_experiment(q)
    print(f'  q = {q:+.2f}  ->  (pi1, pi2) = ({p1:.3f}, {p2:.3f}),'
          f'   max value over types = {value_q(q, theta).max():.4f}')
```

现在菜单是一个函数 $q(\theta)$，而激励相容要求它是非递减的。

认为 $\omega_1$ 更可能的类型想要 $q$ 更高的实验，这些实验对他们认为*不太可能*的状态提供更清晰的证据。

还有一个第二种、不太常见的限制。

因为信息对于类型 $\theta \in \{0, 1\}$ 毫无价值，在 $[0, \theta^*]$ 和 $[\theta^*, 1]$ 上分别应用包络定理，并使临界类型 $\theta^*$ 的租金的两个表达式相匹配，就迫使

$$
\int_0^1 q(\theta) \, d\theta = u_1 - u_2 .
$$ (eq:pi_integral)

注意这个积分是关于 $d\theta$ 的，而不是关于 $dF(\theta)$ 的。

有了这两个约束，卖方的问题简化为

$$
\max_{q(\cdot)} \int_0^1
\Bigl[\bigl(\theta f(\theta) + F(\theta)\bigr) q(\theta)
+ \min\bigl\{\bigl(u_1 - u_2 - q(\theta)\bigr) f(\theta),\ 0 \bigr\}\Bigr] d\theta
$$ (eq:pi_reduced)

受制于 $q$ 非递减以及 {eq}`eq:pi_integral`。

### 作为线性规划求解

{eq}`eq:pi_reduced` 的被积函数在 $q$ 上是**凹的且分段线性的**，因为 $\min\{(d - q) f, 0\} = -f \max\{q - d, 0\}$，其中 $d = u_1 - u_2$ 且 $f \geq 0$。

在线性约束下最大化一个凹的分段线性目标函数是一个线性规划问题。

引入 $z(\theta) \geq \max\{q(\theta) - d,\ 0\}$ 并在网格上离散化 $\theta$，得到

$$
\max_{q, z} \ \sum_n w_n\Bigl[\bigl(\theta_n f_n + F_n\bigr) q_n - f_n z_n\Bigr]
$$

受制于 $z_n \geq q_n - d$，$z_n \geq 0$，$q_{n+1} \geq q_n$，
$-u_2 \leq q_n \leq u_1$，以及 $\sum_n w_n q_n = d$。

```{code-cell} ipython3
def solve_menu(theta, f, u1=1.0, u2=1.0):
    """Solve the seller's problem on a grid of types by linear programming."""
    N = len(theta)
    dth = theta[1] - theta[0]
    f = f / (np.sum(f) * dth)                               # density on the grid
    F = np.cumsum(f) * dth
    w = np.full(N, dth)
    d = u1 - u2

    c = np.concatenate([-(theta * f + F) * w, f * w])       # linprog minimizes
    A_ub = np.hstack([np.eye(N), -np.eye(N)])               # q - z <= d
    b_ub = np.full(N, d)
    D = np.zeros((N - 1, 2 * N))                            # q_n - q_{n+1} <= 0
    rows = np.arange(N - 1)
    D[rows, rows], D[rows, rows + 1] = 1.0, -1.0
    A_ub = np.vstack([A_ub, D])
    b_ub = np.concatenate([b_ub, np.zeros(N - 1)])
    A_eq = np.concatenate([w, np.zeros(N)])[None, :]        # integral constraint
    bounds = [(-u2, u1)] * N + [(0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([d]),
                  bounds=bounds, method='highs')
    return res.x[:N], res
```

这个线性规划自动处理了单调性约束。

这一点很重要，因为另一种选择是手动实现迈尔森的**熨平**（ironing）程序：构造虚拟值

$$
\phi^-(\theta) = \theta f(\theta) + F(\theta),
\qquad
\phi^+(\theta) = (\theta - 1) f(\theta) + F(\theta) ,
$$ (eq:pi_virtual)

用它们积分的凸包的导数来替换它们，然后找到 {eq}`eq:pi_integral` 上乘数的值（参见 {cite:t}`Myerson1981` 和
{cite:t}`Toikka2011`）。

把熨平后的虚拟值记为 $\bar\phi^-$ 和 $\bar\phi^+$。

{cite:t}`BergemannBonattiSmolin2018` 证明，菜单 $q^*$ 是最优的，当且仅当它满足 {eq}`eq:pi_integral`，在相关熨平虚拟值为常数的每个区间上它也为常数，并且存在某个乘数 $\lambda^* > 0$，使得对每个 $\theta$，$q^*(\theta)$ 在 $q \in [-u_2, u_1]$ 范围内使以下表达式最大化

$$
\bar\phi^-(\theta) \min\{q, d\} + \bar\phi^+(\theta) \max\{0, q - d\} - \lambda^* q
$$ (eq:pi_pointwise)

当 $\bar\phi^+(\theta) \leq \bar\phi^-(\theta)$ 时，若 $\bar\phi^-(\theta) < \lambda^*$，最大化者为 $q^* = -u_2$；若 $\bar\phi^+(\theta) < \lambda^* < \bar\phi^-(\theta)$，最大化者为 $q^* = d$；若 $\bar\phi^+(\theta) > \lambda^*$，最大化者为 $q^* = u_1$。

这个线性规划隐式地完成了所有这些工作。

{eq}`eq:pi_integral` 上的乘数 $\lambda^*$ 就是该等式约束的对偶值，即 `-res.eqlin.marginals[0]`；取负号是因为 `linprog` 最小化的是负收入。

我们还需要价格，这来自于要求处于两个项目边界处的买方对两者无差异。

```{code-cell} ipython3
def menu_items(theta, q, u1=1.0, u2=1.0, tol=1e-4, min_width=0.01):
    """Distinct items in the menu, with the interval of types served and the price.

    Values of q taken on a negligible set of types are transition artifacts of the
    grid, not items on the menu, so we drop them.
    """
    qr = np.round(q / tol) * tol
    vals = [v for v in np.unique(qr)
            if theta[qr == v].max() - theta[qr == v].min() >= min_width]
    items = sorted([(v, theta[qr == v].min(), theta[qr == v].max()) for v in vals],
                   key=lambda x: x[1])
    out, prev_v, prev_t = [], None, 0.0
    for v, lo, hi in items:
        if value_q(v, theta, u1, u2).max() < 1e-9:      # uninformative item
            price = 0.0
        elif prev_v is None:
            price = 0.0
        else:
            price = float(value_q(v, lo, u1, u2)
                          - value_q(prev_v, lo, u1, u2) + prev_t)
        out.append((v, lo, hi, price))
        prev_v, prev_t = v, price
    return out
```

### 均匀分布的类型

在 $u_1 = u_2 = 1$ 且 $\theta$ 均匀分布的情况下，虚拟值为 $\phi^-(\theta) = 2\theta$
和 $\phi^+(\theta) = 2\theta - 1$。

两者都严格递增，因此不需要熨平，最优菜单应该只包含一个信息项目。

```{code-cell} ipython3
N = 2001
theta_g = np.linspace(0, 1, N)
q_unif, res = solve_menu(theta_g, np.ones(N))

print('LP status:', res.message)
print('distinct values of q:', np.unique(np.round(q_unif, 4)))
for v, lo, hi, p in menu_items(theta_g, q_unif):
    p1, p2 = q_to_experiment(v)
    print(f'   q = {v:+.4f}  (pi1, pi2) = ({p1:.3f}, {p2:.3f})'
          f'   types [{lo:.3f}, {hi:.3f}]   price {p:.4f}')
```

卖方以单一价格向中间范围的类型提供完全信息实验，而对其他所有人不提供任何东西。

其临界点和价格与 {cite:t}`BergemannBonattiSmolin2018` 的解析解完全一致：对 $\theta \in [1/4, 3/4]$ 的类型以 $1/4$ 的价格提供完全信息。

这就是 {cite:t}`RileyZeckhauser1983` 的"无议价"结果，应用于信息领域。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Optimal menu with uniformly distributed types
    name: fig-pi-uniform
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(theta_g, 2 * theta_g, lw=2, label=r'$\phi^-(\theta) = 2\theta$')
axes[0].plot(theta_g, 2 * theta_g - 1, lw=2, label=r'$\phi^+(\theta) = 2\theta - 1$')
axes[0].axhline(0.5, color='0.4', ls='--', lw=1, label=r'$\lambda^* = 1/2$')
axes[0].set(xlabel=r'$\theta$', title='virtual values, both strictly increasing')
axes[0].legend(fontsize=9)

axes[1].step(theta_g, q_unif, lw=2, where='mid')
axes[1].set(xlabel=r'$\theta$', ylabel=r'$q^*(\theta)$', ylim=(-1.15, 1.15),
            title='optimal menu: one informative item')
axes[1].annotate('no information', (0.06, -0.85), fontsize=9)
axes[1].annotate('full information', (0.38, 0.12), fontsize=9)
axes[1].annotate('no information', (0.78, 0.85), fontsize=9)
fig.tight_layout()
plt.show()
```

### 双峰类型与版本化的理由

{cite:t}`BergemannBonattiSmolin2018` 中的推论1说，只有当虚拟值需要熨平时才会提供第二个实验。

对 {eq}`eq:pi_virtual` 求导，得到

$$
\phi^{-\prime}(\theta) = 2 f(\theta) + \theta f'(\theta),
\qquad
\phi^{+\prime}(\theta) = 2 f(\theta) - (1 - \theta) f'(\theta) .
$$ (eq:pi_virtual_slopes)

因此 $\phi^-$ 递增当且仅当 $\theta^2 f(\theta)$ 递增，$\phi^+$ 递增当且仅当 $(1-\theta)^2 f(\theta)$ 递减。

正则性失效的地方正是密度下降速度快于 $\theta^{-2}$ 或上升速度快于 $(1-\theta)^{-2}$ 的地方，也就是密度陡峭的地方。

{cite:t}`BergemannBonattiSmolin2018` 非正式地指出，当大多数买方事先已信息充分，且大多数类型接近 $0$ 或 $1$ 时，就需要熨平。

这一说法需要加以限定。

对于 $\text{Beta}(a, b)$ 密度，两个虚拟值都递增当且仅当 $a \leq 1$ 且 $b \leq 1$（见 {ref}`pi_ex4`），因此呈U形分布的信息充分买方群体是正则的，只会被提供一个项目。

无论位于 $[0, 1]$ 的两端附近还是内部，信念的陡峭聚集都是不正则的，而 {ref}`pi_ex4` 给出了一个这样的信息充分群体的例子，它被提供了两个项目。

不正则性是第二个项目存在的必要条件，但不是充分条件。

$\text{Beta}(30, 30)$ 是不正则的，但它关于 $1/2$ 对称，而在 $u_1 = u_2$ 的情况下，{cite:t}`BergemannBonattiSmolin2018` 的推论1随即给出单一项目的结果。

我们遵循论文的做法，取 $\text{Beta}(8, 30)$ 和 $\text{Beta}(60, 30)$ 的等权混合。

每个成分都将信念集中在一个狭窄的范围内，因此该混合分布有两个陡峭的聚集点，分别在 $\theta \approx 0.2$ 和 $\theta \approx 0.67$ 附近。

```{code-cell} ipython3
f_bimodal = (0.5 * stats.beta(8, 30).pdf(theta_g)
             + 0.5 * stats.beta(60, 30).pdf(theta_g))

def falling_intervals(theta, phi, min_width=0.01):
    """Intervals of types on which phi decreases."""
    down = np.concatenate([[0], (np.diff(phi) < 0).astype(int), [0]])
    edges = np.flatnonzero(np.diff(down))
    return [(theta[a], theta[b]) for a, b in zip(edges[::2], edges[1::2])
            if theta[b] - theta[a] >= min_width]


dth = theta_g[1] - theta_g[0]
F_bi = np.cumsum(f_bimodal) * dth
for name, phi in [('phi-', theta_g * f_bimodal + F_bi),
                  ('phi+', (theta_g - 1) * f_bimodal + F_bi)]:
    print(f'{name} decreases on',
          ', '.join(f'[{a:.3f}, {b:.3f}]'
                    for a, b in falling_intervals(theta_g, phi)))

q_bi, res_bi = solve_menu(theta_g, f_bimodal)

print('\nLP status:', res_bi.message)
print('distinct values of q:', np.unique(np.round(q_bi, 3)))
print(f'multiplier lambda* = {-res_bi.eqlin.marginals[0]:.4f}')
print()
for v, lo, hi, p in menu_items(theta_g, q_bi):
    p1, p2 = q_to_experiment(v)
    label = 'no information' if abs(p) < 1e-9 else (
        'full information' if abs(v) < 1e-6 else 'partial information')
    print(f'   q = {v:+.4f}  (pi1, pi2) = ({p1:.3f}, {p2:.3f})'
          f'   types [{lo:.3f}, {hi:.3f}]   price {p:.4f}   {label}')
```

$\phi^-$ 在每个聚集点的上侧翼递减，那里密度陡峭下降；$\phi^+$ 在每个聚集点的下侧翼递减，那里密度陡峭上升。

两个虚拟值都需要熨平。

现在菜单包含**两个**信息项目，这与 {prf:ref}`pi_prop_structure` 以及 {cite:t}`BergemannBonattiSmolin2018` 中某个最优菜单最多包含两个项目的结果相符。

部分项目具有 $\pi_2 = 1$，因此信号 $s_1$ 只在状态 $\omega_1$ 中出现，能够完美揭示该状态，而信号 $s_2$ 则使买方保持不确定。

类型 $\theta \in [0.175, 0.550]$ 购买它。

该区间包含 $\theta^* = 1/2$，这是最不确定该怎么做、最看重信息的类型。

在两个信息项目之间无差异的类型大约是 $\theta \approx 0.55$。

如果向 $\theta^*$ 出售 $\overline{E}$，就需要对部分项目进一步扭曲并降低其价格，这将损失来自 $\theta \approx 0.2$ 附近众多类型的收入。

部分项目使卖方能够对 $\theta \approx 0.7$ 附近的大量买方以高价出售 $\overline{E}$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bimodal type density and the resulting two-item menu
    name: fig-pi-bimodal
---
items = menu_items(theta_g, q_bi)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(theta_g, f_bimodal, lw=2, color='C2')
axes[0].fill_between(theta_g, f_bimodal, alpha=0.2, color='C2')
axes[0].set(xlabel=r'$\theta$', ylabel='density',
            title='beliefs packed into two steep clusters')

axes[1].step(theta_g, q_bi, lw=2, where='mid')
axes[1].set(xlabel=r'$\theta$', ylabel=r'$q^*(\theta)$', ylim=(-1.15, 1.15),
            title='optimal menu: two informative items')
for v, lo, hi, p in items:
    if p > 1e-9:
        axes[1].annotate(f'price {p:.3f}', ((lo + hi) / 2, v + 0.12),
                         ha='center', fontsize=9)
fig.tight_layout()
plt.show()
```

我们可以直接看出为什么卖方要费这个功夫。

每个类型对 $\overline{E}$ 的评价都不低于任何其他实验，因此最佳的单一项目菜单是以一个挂牌价格 $p$ 出售 $\overline{E}$。

当 $u_1 = u_2 = 1$ 时，$V(\overline{E}, \theta) = \min\{\theta, 1 - \theta\}$，因此类型 $\theta \in [p, 1 - p]$ 会购买，收入为 $p\,[F(1 - p) - F(p)]$。

```{code-cell} ipython3
def revenue(theta, q, f, u1=1.0, u2=1.0):
    """Expected revenue from the menu q under density f."""
    dth = theta[1] - theta[0]
    price = np.zeros_like(theta)
    for v, lo, hi, p in menu_items(theta, q, u1, u2):
        price[(theta >= lo) & (theta <= hi)] = p
    return np.sum(price * f) * dth / (np.sum(f) * dth)


def F_mix(x):
    """Distribution function of the bimodal mixture."""
    return 0.5 * stats.beta(8, 30).cdf(x) + 0.5 * stats.beta(60, 30).cdf(x)


p_post = np.linspace(0, 0.5, 5001)
rev_post = p_post * (F_mix(1 - p_post) - F_mix(p_post))
k = rev_post.argmax()
rev_two = revenue(theta_g, q_bi, f_bimodal)

print(f'revenue, optimal two-item menu      {rev_two:.4f}')
print(f'revenue, best posted price for E_bar {rev_post[k]:.4f}'
      f'   at p = {p_post[k]:.3f}')
print(f'gain from versioning                {100 * (rev_two / rev_post[k] - 1):.1f}%')
```

增加部分项目使收入比全信息的最佳挂牌价格提高了约8%。

部分实验不是同一产品的加噪版本，它是一个*位置不同*的产品，价格足够低以吸引信息充分的类型，同时又足够无用以至于不会对信息不充分的类型冲击高价格。

## 结束语

布莱克韦尔定理告诉我们，何时*所有*决策者都一致认为一个实验优于另一个。

将其作为一种设计原则来解读，其真正的内涵在于它保持沉默的那个集合的大小。

{cite:t}`BergemannBonattiSmolin2018` 表明，出售数据的垄断者恰恰生活在那个集合中，因为基于信念的筛选需要不同类型对之做出不同排序的产品。

有两个教训超越了这个模型。

第一，信息的最优降级是结构化的而非随机的：菜单上的每个实验都有一个能完全揭示某一状态的信号。

{cite:t}`BergemannBonattiSmolin2018` 得出结论，最优菜单上的产品都不应向卖方的数据添加无偏噪声，除非买方在下游相互竞争，或数据必须匿名化。

第二，版本化需要熨平的虚拟价值，而陡峭的信念密度恰恰会产生这种虚拟价值。

买方信息充分与否并不能决定一切：一个呈 U 形贝塔分布的信息充分买方群体会得到单一产品的供应，而信念在 $[0, 1]$ 两端附近或其内部呈陡峭聚集的群体，则可以得到两种产品的供应。

向信息不完全的买方出售信息有着悠久的历史。

{cite:t}`AdmatiPfleiderer1986` 研究了一个卖方面对一个*事前相同*的连续交易者群体，这些交易者随后交易一个共同价值资产，他们发现卖方想要提供带噪声且*个性化*的信息，从而使每个交易者对自己所知的信息保持局部垄断。

他们模型中的交易者在下游相互竞争，因此第一个教训并不适用于他们。

那里的异质性是由卖方创造的；而在这里，异质性是买方自己的先验信息，正是这一点使问题变成了筛选问题。

{cite:t}`BergemannBonatti2015` 研究了一个数据提供商，该提供商为每个“cookie”（关于单个消费者的记录）设定价格，并面对一个没有私人信息的广告商，由其选择购买哪些 cookie。

一个有用的对比是 {cite:t}`KamenicaGentzkow2011`，其中发送方也承诺一个信息结构，但没有货币转移，而是直接关心接收方的行动；这里的卖方只关心收入，且不能以状态、信号或买方的行动为条件设定支付。

想要了解统计学背景的读者可以回到 {doc}`blackwell_kihlstrom`，了解经济性、充分性和不确定性降低标准之间的等价关系；可以参阅 {doc}`likelihood_bayes`，了解私人信号如何产生此处买方类型所依据的临时信念；还可以参阅
{doc}`information_market_equilibrium`，了解当信息通过价格而非直接出售来传递时会发生什么。

## 练习

```{exercise-start}
:label: pi_ex1
```

本练习研究**一致**情形，即两种类型在没有额外信息的情况下会采取相同的行动。

设 $u_1 = u_2 = 1$，$\theta^L = 0.9$ 且 $\theta^H = 0.7$，因此
$\theta^* = 1/2 < \theta^H < \theta^L$。

因为两种类型自己都会选择 $a_1$，卖方没有理由降低低类型关于 $\omega_1$ 所学到的信息，因此设 $\pi_1 = 1$，并将 $\pi_2$ 作为唯一的选择变量。

1. 对若干个 $\gamma$ 值，绘制卖方收入关于 $\pi_2 \in [0, 1]$ 的图形，并确认其为*线性*的。

2. 得出结论：最优点总是在端点处，因此低类型要么得到完全信息，要么什么都得不到。

3. {cite:t}`BergemannBonattiSmolin2018` 表明，低类型获得完全信息实验当且仅当
   $\gamma \leq (1 - \theta^L)/(1 - \theta^H)$。

   用二分法数值定位切换点并进行比较。

为什么这里的答案是极端的，而讲座中不一致情形的例子却产生了内部解 $\pi_1 = 4/5$？

```{exercise-end}
```

```{solution-start} pi_ex1
:class: dropdown
```

这里给出一种解法：

```{code-cell} ipython3
tL_c, tH_c = 0.9, 0.7
p2_grid = np.linspace(0, 1, 401)

fig, ax = plt.subplots()
for gamma in [0.1, 0.25, 1/3, 0.5, 0.7]:
    r = np.array([two_type_revenue(1.0, p2, tL_c, tH_c, gamma) for p2 in p2_grid])
    dev = np.abs(r - np.interp(p2_grid, [0, 1], [r[0], r[-1]])).max()
    ax.plot(p2_grid, r, lw=2, label=rf'$\gamma = {gamma:.3f}$')
    print(f'gamma = {gamma:.3f}:  revenue at pi2=0 is {r[0]:.5f}, '
          f'at pi2=1 is {r[-1]:.5f},  deviation from linear {dev:.1e}')
ax.set(xlabel=r'$\pi_2$', ylabel='revenue',
       title='revenue is linear in $\pi_2$, so the optimum is at an endpoint')
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
lo, hi = 0.0, 1.0
for _ in range(60):
    mid = (lo + hi) / 2
    if two_type_revenue(1, 1, tL_c, tH_c, mid) >= two_type_revenue(1, 0, tL_c, tH_c, mid):
        lo = mid
    else:
        hi = mid

print(f'numerical switch point            gamma = {lo:.6f}')
print(f'(1 - theta_L) / (1 - theta_H)           = {(1 - tL_c) / (1 - tH_c):.6f}')
```

收入在 $\pi_2$ 上的线性程度达到了机器精度，因此最优点总是位于
$\pi_2 \in \{0, 1\}$，且切换点确切地发生在 $\gamma = 1/3$，正如预测的那样。

极端解的原因在于，在一致信念下，两种类型无论如何都会选择 $a_1$，因此唯一的问题在于卖方对 $\omega_2$ 揭示了多少信息。

这时两种类型都通过同一项 $(1 - \theta)\pi_2 u_2$ 来评估该实验，这就是为什么目标函数和约束在单一变量 $\pi_2$ 上是线性的，也是为什么 {cite:t}`RileyZeckhauser1983` 的无议价逻辑在此适用的原因。

在不一致信念下，两种类型自己会采取不同的行动，价值函数中的拐点位于两者之间，卖方可以将某个实验定位，使其对一种类型有很大价值，而对另一种类型价值很小。

正是这种可能性使得内部扭曲成为最优。

```{solution-end}
```

```{exercise-start}
:label: pi_ex2
```

{cite:t}`BergemannBonattiSmolin2018` 中的推论1指出，只要两个虚拟值 {eq}`eq:pi_virtual` 都是严格递增的，最优菜单就只包含一个项目，而对于均匀分布的类型，这一点**与支付函数** $(u_1, u_2)$ **无关**。

1. 通过对若干非对称支付对，例如 $(u_1, u_2) \in \{(1, 1), (2, 1), (1, 3),
   (5, 1)\}$，求解均匀类型下卖方的问题，来验证这一点。

2. 对每种情况，报告 $\theta^*$、被服务类型的区间以及价格。

3. 确认所提供的项目始终是完全信息实验，正如 {prf:ref}`pi_prop_structure` 所要求的那样。

4. 利用 {eq}`eq:pi_pointwise` 和 {eq}`eq:pi_integral` 从解析上证明乘子为
   $\lambda^* = \theta^*$，被服务的类型为
   $\theta \in [\theta^*/2,\ (1 + \theta^*)/2]$，且价格为
   $u_1 u_2 / \bigl(2 (u_1 + u_2)\bigr)$。

   将这些值与线性规划得到的结果并列打印出来。

```{exercise-end}
```

```{solution-start} pi_ex2
:class: dropdown
```

这里给出一种解法：

```{code-cell} ipython3
print(f'{"u1":>3s}{"u2":>3s}{"theta*":>8s}{"q":>6s}{"":3s}'
      f'{"served (LP)":<16s}{"served (theory)":<17s}'
      f'{"price":>7s}{"theory":>8s}{"lambda*":>9s}{"theory":>8s}')
for u1_, u2_ in [(1, 1), (2, 1), (1, 3), (5, 1)]:
    q_a, res_a = solve_menu(theta_g, np.ones(N), u1_, u2_)
    star = u2_ / (u1_ + u2_)
    served = [it for it in menu_items(theta_g, q_a, u1_, u2_) if it[3] > 1e-9]
    assert len(served) == 1                     # a single informative item
    v, lo, hi, p = served[0]
    assert abs(v - (u1_ - u2_)) < 1e-3          # the item is fully informative
    lam = -res_a.eqlin.marginals[0]
    print(f'{u1_:3d}{u2_:3d}{star:8.4f}{v:6.1f}{"":3s}'
          f'{f"[{lo:.3f}, {hi:.3f}]":<16s}'
          f'{f"[{star / 2:.3f}, {(1 + star) / 2:.3f}]":<17s}'
          f'{p:7.4f}{u1_ * u2_ / (2 * (u1_ + u2_)):8.4f}'
          f'{lam:9.4f}{star:8.4f}')
print('\nevery menu contains exactly one informative item, '
      'and it is the fully informative one')
```

均匀密度的虚拟值为 $\phi^-(\theta) = 2\theta$ 和 $\phi^+(\theta) = 2\theta - 1$，无论支付函数如何，因为 $u_1$ 和 $u_2$ 仅通过 $d = u_1 - u_2$ 以及 $q$ 的边界进入卖方的问题，而不通过 $f$ 或 $F$。

两者都严格递增，因此不需要熨平，单一项目就是最优的。

支付函数确实会移动 $\theta^*$，从而影响哪些类型被服务以及以什么价格服务，但它们从不会使均匀密度下的版本化变得值得。

对于解析基准，由于不需要熨平，{eq}`eq:pi_pointwise` 给出：当 $\theta < \lambda^*/2$ 时 $q^* = -u_2$，当
$\lambda^*/2 < \theta < (1 + \lambda^*)/2$ 时 $q^* = u_1 - u_2$，当
$\theta > (1 + \lambda^*)/2$ 时 $q^* = u_1$。

代入 {eq}`eq:pi_integral` 得到

$$
-u_2 \frac{\lambda^*}{2} + (u_1 - u_2) \frac{1}{2} + u_1 \frac{1 - \lambda^*}{2}
= u_1 - u_2 ,
$$

因此 $\lambda^* (u_1 + u_2) = u_2$，即 $\lambda^* = \theta^*$。

被服务的最低类型 $\theta^*/2$ 在 $\overline{E}$ 和不购买之间是无差异的。

对于 $\theta \leq \theta^*$，我们有 $V(\overline{E}, \theta) = \theta u_1$，因此价格为
$u_1 \theta^* / 2 = u_1 u_2 / \bigl(2(u_1 + u_2)\bigr)$。

被服务的最高类型 $(1 + \theta^*)/2$ 对 $\overline{E}$ 的估值为
$(1 - \theta) u_2 = u_1 u_2 / \bigl(2(u_1 + u_2)\bigr)$，与价格相同。

线性规划的结果在类型网格的离散化精度范围内与这三个量完全一致。

```{solution-end}
```

```{exercise-start}
:label: pi_ex3
```

本练习将本讲座与 {doc}`blackwell_kihlstrom` 联系起来。

布莱克韦尔定理说，如果 $E'$ 是 $E$ 的信息降级，那么*每一个*决策者都弱偏好 $E$。

1. 抽取许多满足 $\pi_1 + \pi_2 \geq 1$ 的随机二元实验对。

2. 对于每一对，使用 `garbling` 来判断一个是否是另一个的信息降级，并单独计算在每个类型中，一个是否在价值上支配另一个。

   因为 $V(E, \theta)$ 在 $\theta$ 上是分段线性的，所以只需要在 $\theta \in \{0, 1\}$ 以及任一价值函数存在拐点的类型处比较这两个值即可。

3. 确认信息降级蕴含一致的偏好，并报告布莱克韦尔的序无法排序的随机对所占的比例。

4. 在无法排序的对中，验证确实有些类型偏好一个实验，而有些类型偏好另一个。

5. 解释为什么对于二状态实验，布莱克韦尔的序无法排序的那些对，恰好应该是类型存在分歧的那些对。

   如果仅在有限的类型网格上检验支配关系，会出现什么问题？

```{exercise-end}
```

```{solution-start} pi_ex3
:class: dropdown
```

这里给出一种解法：

```{code-cell} ipython3
def kinks(pi1, pi2, u1=1.0, u2=1.0):
    """Endpoints and types at which V(E, theta) can change slope."""
    k = [0.0, 1.0, u2 / (u1 + u2)]
    for a, b in [(pi1 * u1, (1 - pi2) * u2), ((1 - pi1) * u1, pi2 * u2)]:
        if a + b > 0:
            k.append(b / (a + b))       # indifference after one of the signals
    return np.array(k)


rng = np.random.default_rng(0)
grid_t = np.linspace(0.001, 0.999, 999)

n_pairs, n_garble, n_unranked, violations = 4000, 0, 0, 0
n_disagree, n_disagree_grid, n_unanimous_unranked = 0, 0, 0
for _ in range(n_pairs):
    (a1, a2), (b1, b2) = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
    if a1 + a2 < 1 or b1 + b2 < 1:
        continue
    Ea, Eb = experiment(a1, a2), experiment(b1, b2)
    t = np.concatenate([kinks(a1, a2), kinks(b1, b2)])
    va, vb = value(a1, a2, t), value(b1, b2, t)
    va_g, vb_g = value(a1, a2, grid_t), value(b1, b2, grid_t)

    a_garbles_b = garbling(Ea, Eb) is not None      # Eb is a garbling of Ea
    b_garbles_a = garbling(Eb, Ea) is not None
    a_dominates = np.all(va >= vb - 1e-12)
    b_dominates = np.all(vb >= va - 1e-12)

    if a_garbles_b:
        n_garble += 1
        if not a_dominates:
            violations += 1
    if b_garbles_a:
        n_garble += 1
        if not b_dominates:
            violations += 1
    if not (a_garbles_b or b_garbles_a):
        n_unranked += 1
        if not (a_dominates or b_dominates):
            n_disagree += 1
        if not (np.all(va_g >= vb_g - 1e-9) or np.all(vb_g >= va_g - 1e-9)):
            n_disagree_grid += 1
    if (a_dominates and not a_garbles_b) or (b_dominates and not b_garbles_a):
        n_unanimous_unranked += 1

print(f'garbling relations found              {n_garble}')
print(f'violations of Blackwell               {violations}')
print(f'unanimous rankings without garbling   {n_unanimous_unranked}')
print(f'pairs unranked by Blackwell           {n_unranked}')
print(f'  of which types disagree (kinks)     {n_disagree}')
print(f'  of which types disagree (grid)      {n_disagree_grid}')
```

布莱克韦尔定理从未被违反：只要一个实验降级为另一个，每种类型都偏好作为降级源的那个实验。

其逆命题在每次抽取中也都成立：只要所有类型对两个实验的排序一致，其中一个就是另一个的信息降级。

大部分随机对都未被排序，而对于每一个这样的对，确实都存在某些类型偏好一个实验，而某些类型偏好另一个实验。

这个未被排序的区域正是数据卖方所需要的空间。

如果布莱克韦尔的序是完全的，那么每个买方都会对所有信息产品的排序达成一致，卖方的问题将退化为对单一质量指标的标准非线性定价，而本讲座所描述的横向筛选也就不可能存在。

对于二状态实验，信息降级与一致偏好之间的对应关系是精确的。

固定一个参照先验 $c \in (0, 1)$，令 $x$ 表示在 $c$ 下某个信号所诱导出的 $\omega_1$ 的后验概率。

在 $u_1 = u_2 = 1$ 的情况下，$E$ 对类型 $\theta$ 的毛价值等于在 $c$ 下，凸函数
$k_\theta(x) = \max\{\theta x / c,\ (1 - \theta)(1 - x)/(1 - c)\}$ 的期望值，该函数只有一个拐点。

当 $\theta$ 在 $(0, 1)$ 上变化时，拐点也在 $(0, 1)$ 上变化。

$[0, 1]$ 上的每个凸函数都可以表示为一个仿射函数加上这类单拐点函数的非负混合，而仿射部分在所有实验下的期望值相同。

根据 {prf:ref}`thm-kihlstrom`，当且仅当对每个 $\theta$ 都有
$V(E, \theta) \geq V(E', \theta)$ 时，$E'$ 才是 $E$ 的信息降级。

在有限的类型网格上进行检验，可能会遗漏局限在两个网格点之间的一段短区间内的分歧。

在上述抽取中，由999个类型组成的网格恰好遗漏了一次这样的分歧。

在拐点处进行评估则是精确的，因为两个分段线性函数之差在拐点或端点处取得最大值和最小值。

```{solution-end}
```

```{exercise-start}
:label: pi_ex4
```

本练习刻画了使虚拟值 {eq}`eq:pi_virtual` 不需要熨平的类型密度。

1. 利用 {eq}`eq:pi_virtual_slopes` 证明，在 $(0, 1)$ 上，$\phi^-$ 严格递增当且仅当
   $\theta^2 f(\theta)$ 严格递增，而 $\phi^+$ 严格递增当且仅当
   $(1 - \theta)^2 f(\theta)$ 严格递减。

2. 由此推出，对于 $\text{Beta}(a, b)$ 密度，两个虚拟值都严格递增当且仅当
   $a \leq 1$ 且 $b \leq 1$。

3. 在 $(a, b)$ 对的网格上数值验证该条件。

4. 用 `solve_menu` 证明，呈U形的密度 $\text{Beta}(0.5, 0.5)$、
   $\text{Beta}(0.3, 0.3)$ 和 $\text{Beta}(0.5, 0.8)$ 都只被提供单一项目，不规则的
   $\text{Beta}(30, 30)$ 同样只被提供单一项目，而不规则的 $\text{Beta}(60, 30)$ 则被提供两个信息性项目。

5. 证明 $\text{Beta}(2, 60)$ 和 $\text{Beta}(40, 4)$ 的等权混合——即大多数买方都信息充分的总体——是不规则的，并被提供两个信息性项目。

```{exercise-end}
```

```{solution-start} pi_ex4
:class: dropdown
```

这里给出一种解法。

对于 $\theta \in (0, 1)$，

$$
\frac{d}{d\theta}\bigl[\theta^2 f(\theta)\bigr]
= \theta \bigl[2 f(\theta) + \theta f'(\theta)\bigr] = \theta\, \phi^{-\prime}(\theta),
\qquad
\frac{d}{d\theta}\bigl[(1 - \theta)^2 f(\theta)\bigr]
= -(1 - \theta)\, \phi^{+\prime}(\theta) .
$$

所以 $\phi^{-\prime}$ 的符号与 $\theta^2 f$ 的导数符号相同，而 $\phi^{+\prime}$
的符号与 $(1-\theta)^2 f$ 的导数符号相反。

对于 $f(\theta) \propto \theta^{a-1}(1 - \theta)^{b-1}$，

$$
\frac{d}{d\theta} \log\bigl[\theta^2 f(\theta)\bigr]
= \frac{a + 1}{\theta} - \frac{b - 1}{1 - \theta},
\qquad
\frac{d}{d\theta} \log\bigl[(1 - \theta)^2 f(\theta)\bigr]
= \frac{a - 1}{\theta} - \frac{b + 1}{1 - \theta} .
$$

第一个表达式在整个 $(0, 1)$ 上都为正当且仅当 $b \leq 1$，因为当 $b > 1$ 时，
当 $\theta \to 1$ 时它会发散到 $-\infty$。

第二个表达式在整个 $(0, 1)$ 上都为负当且仅当 $a \leq 1$，因为当
$a > 1$ 时，当 $\theta \to 0$ 时它会发散到 $+\infty$。

```{code-cell} ipython3
th_in = theta_g[1:-1]                   # Beta densities can be infinite at 0 and 1


def mixture(components):
    """Density and distribution function of an equal mixture of Beta laws."""
    f_c = np.mean([stats.beta(a, b).pdf(th_in) for a, b in components], axis=0)
    F_c = np.mean([stats.beta(a, b).cdf(th_in) for a, b in components], axis=0)
    return f_c, F_c


def regular(f_c, F_c):
    """True if both virtual values increase on the grid."""
    return (np.all(np.diff(th_in * f_c + F_c) > 0)
            and np.all(np.diff((th_in - 1) * f_c + F_c) > 0))


ab_grid = [0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0]
mismatch = [(a, b) for a in ab_grid for b in ab_grid
            if regular(*mixture([(a, b)])) != (a <= 1 and b <= 1)]
print(f'(a, b) pairs checked: {len(ab_grid)**2},  mismatches: {mismatch}')

for comps in [[(0.5, 0.5)], [(0.3, 0.3)], [(0.5, 0.8)], [(30, 30)], [(60, 30)],
              [(2, 60), (40, 4)]]:
    f_c, F_c = mixture(comps)
    near_ends = np.mean([stats.beta(a, b).cdf(0.15) + stats.beta(a, b).sf(0.85)
                         for a, b in comps])
    q_c, _ = solve_menu(th_in, f_c)
    informative = [it for it in menu_items(th_in, q_c) if it[3] > 1e-9]
    print('\n' + ' + '.join(f'Beta{ab}' for ab in comps)
          + f':  regular = {regular(f_c, F_c)},'
          f'  mass within 0.15 of 0 or 1 = {near_ends:.2f},'
          f'  informative items = {len(informative)}')
    for v, lo, hi, p in informative:
        print(f'   q = {v:+.3f}   types [{lo:.3f}, {hi:.3f}]   price {p:.4f}')
```

数值检验在每一个 $(a, b)$ 组合处都与 $a \leq 1$、$b \leq 1$ 的条件一致。

这三个U形密度都是规则的，都只被提供 $\overline{E}$。

$\text{Beta}(30, 30)$ 是不规则的，但它关于 $\theta^* = 1/2$ 对称，因此它也只被提供 $\overline{E}$。

陡峭且非对称的 $\text{Beta}(60, 30)$ 除了 $\overline{E}$ 之外，还被提供了一个 $q < 0$ 的部分项目。

$\text{Beta}(2, 60)$ 和 $\text{Beta}(40, 4)$ 的混合，其质量落在距 $0$ 或 $1$ 的 $0.15$ 范围内的比例比任何一个U形密度都高。

它的两个簇都很陡峭，是不规则的，被提供了两个信息性项目。

这两类总体中买方大多都信息充分；密度的陡峭程度将它们区分开来。

```{solution-end}
```

```{exercise-start}
:label: pi_ex5
```

本练习手工熨平双峰例子中的虚拟值，并检验线性规划找到的菜单。

设 $u_1 = u_2 = 1$，使得 $d = 0$，并使用密度 `f_bimodal`。

1. 在网格上计算 $\Phi^\pm(\theta) = \int_0^\theta \phi^\pm(x)\, dx$、它们的最大凸下界，以及作为凸下界斜率的熨平虚拟值 $\bar\phi^\pm$。

2. 由 {eq}`eq:pi_pointwise`，$\bar\phi^- < \lambda^*$ 的类型得到 $q = -1$，
   $\bar\phi^+ < \lambda^* < \bar\phi^-$ 的类型得到 $q = 0$，
   $\bar\phi^+ > \lambda^*$ 的类型得到 $q = 1$，而在 $\bar\phi^- = \lambda^*$
   的区间上，类型共享一个共同的 $q_m \in [-1, 0]$。

   由积分约束 {eq}`eq:pi_integral` 求出 $\lambda^*$ 和 $q_m$。

3. 将 $\lambda^*$、$q_m$、购买部分项目的类型区间，以及不购买任何项目的类型下界，与线性规划的解进行比较。

4. 绘制 $\phi^\pm$、$\bar\phi^\pm$ 以及 $\lambda^*$ 的图形。

```{exercise-end}
```

```{solution-start} pi_ex5
:class: dropdown
```

这里给出一种解法。

```{code-cell} ipython3
def ironed(theta, phi):
    """Slopes of the greatest convex minorant of the integral of phi."""
    dth = theta[1] - theta[0]
    Phi = np.concatenate([[0.0], np.cumsum((phi[1:] + phi[:-1]) / 2) * dth])
    hull = [0]                                       # lower convex hull
    for i in range(1, len(theta)):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            if ((Phi[k] - Phi[j]) * (theta[i] - theta[j])
                    >= (Phi[i] - Phi[j]) * (theta[k] - theta[j])):
                hull.pop()
            else:
                break
        hull.append(i)
    phi_bar = np.empty_like(phi)
    for j, k in zip(hull[:-1], hull[1:]):
        phi_bar[j:k + 1] = (Phi[k] - Phi[j]) / (theta[k] - theta[j])
    return phi_bar


dth = theta_g[1] - theta_g[0]
f_n = f_bimodal / (np.sum(f_bimodal) * dth)
F_n = np.cumsum(f_n) * dth
phi_m, phi_p = theta_g * f_n + F_n, (theta_g - 1) * f_n + F_n
bar_m, bar_p = ironed(theta_g, phi_m), ironed(theta_g, phi_p)
print('phi_bar+ <= phi_bar- everywhere:', np.all(bar_p <= bar_m + 1e-12))


def integral_q(lam):
    """Integral of q when types with bar_m = lam are assigned q = 0."""
    return (-(bar_m < lam).sum() + (bar_p > lam).sum()) * dth


lo, hi = bar_m.min(), bar_m.max()                   # integral_q decreases in lam
for _ in range(200):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if integral_q(mid) > 0 else (lo, mid)
lam_star = hi

pooled = np.abs(bar_m - lam_star) < 1e-9
th_a, th_b = theta_g[pooled].min(), theta_g[pooled].max()
th_c = theta_g[bar_p > lam_star].min()
q_m = (th_a - (1 - th_c)) / (th_b - th_a)

lp_items = [it for it in menu_items(theta_g, q_bi) if it[3] > 1e-9]
(q_lp, a_lp, b_lp, _), (_, _, c_lp, _) = lp_items
print(f'\n{"":24s}{"by hand":>10s}{"LP":>10s}')
print(f'{"multiplier lambda*":24s}{lam_star:10.4f}{-res_bi.eqlin.marginals[0]:10.4f}')
print(f'{"partial item q_m":24s}{q_m:10.4f}{q_lp:10.4f}')
print(f'{"lowest buyer":24s}{th_a:10.4f}{a_lp:10.4f}')
print(f'{"partial / full cutoff":24s}{th_b:10.4f}{b_lp:10.4f}')
print(f'{"highest buyer":24s}{th_c:10.4f}{c_lp:10.4f}')
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(theta_g, phi_m, lw=1, color='C0', alpha=0.5, label=r'$\phi^-$')
ax.plot(theta_g, bar_m, lw=2, color='C0', label=r'$\bar\phi^-$')
ax.plot(theta_g, phi_p, lw=1, color='C1', alpha=0.5, label=r'$\phi^+$')
ax.plot(theta_g, bar_p, lw=2, color='C1', label=r'$\bar\phi^+$')
ax.axhline(lam_star, color='0.3', ls='--', lw=1, label=r'$\lambda^*$')
ax.axvspan(th_a, th_b, color='C2', alpha=0.15, label='partial item')
ax.set(xlabel=r'$\theta$', ylim=(-4, 4),
       title='ironed virtual values for the bimodal density')
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
plt.show()
```

乘子就是使 $q$ 的积分跨越零点而发生跳跃的那个水平。

刚好在 $\lambda^*$ 之下，被合并的类型得到 $q = 0$，积分为正；刚好在 $\lambda^*$
之上，它们得到 $q = -1$，积分为负。

在合并区间 $[\theta_a, \theta_b]$ 上，积分约束写作
$-\theta_a + q_m (\theta_b - \theta_a) + (1 - \theta_c) = 0$，由此得到
$q_m = \bigl(\theta_a - (1 - \theta_c)\bigr) / (\theta_b - \theta_a)$。

合并区间就是熨平操作将 $\phi^-$ 拉平至水平 $\lambda^*$ 的那个区间。

其上端，约 $\theta \approx 0.55$，是在部分项目和 $\overline{E}$ 之间感到无差异的类型。

手工计算与线性规划的结果在网格间距范围内一致。

```{solution-end}
```
