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
  title: 可信的政府政策
  headings:
    Overview: 概览
    Overview::What we build: 我们要构建什么
    The repeated economy: 重复经济
    Recursive strategies and promised values: 递归策略与承诺值
    Recursive strategies and promised values::Historical antecedents: 历史渊源
    The Abreu–Pearce–Stacchetti method: 阿布鲁-皮尔斯-斯塔凯蒂方法
    The Abreu–Pearce–Stacchetti method::Computing the operator: 计算算子
    The best and the worst equilibrium: 最优与最差均衡
    The best and the worst equilibrium::The worst: 最差
    The best and the worst equilibrium::The best: 最优
    Examples of recursive equilibria: 递归均衡示例
    Examples of recursive equilibria::Infinite repetition of Nash: 纳什结果的无限重复
    Examples of recursive equilibria::Infinite repetition of something better: 更优结果的无限重复
    'Examples of recursive equilibria::Something worse: Abreu''s stick and carrot': 更差的均衡：阿布鲁的大棒加胡萝卜
    Multiplicity: 多重性
    Numerical example: 数值示例
    Interpretations: 解读
    Interpretations::Whose expectations are they?: 到底是谁的预期？
    Interpretations::Remedies: 补救措施
    Interpretations::Where this leaves us: 我们由此得出的结论
    Exercises: 练习
---

(phillips_credible_policies)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 可信的政府政策

```{index} single: Phillips Curve; Credible Policies
```

```{contents} Contents
:depth: 2
```

## 概览

{doc}`phillips_credibility` 让政府陷入了一个陷阱。

如果没有承诺自身行为的技术手段，一个每期都重新优化的政府最终会落入纳什结果，以 $\theta U^*$ 的速率通货膨胀，却一无所获。

那是一个单期的故事，它招致了一个显而易见的反驳：一个预期明天将再次面对同一公众的政府，是有"声誉"需要维护的。

本讲座遵循 {cite:t}`Sargent1999` 第4章的思路，认真对待这一反驳意见。

我们让基德兰德-普雷斯科特经济永远重复，让政府当期的行动依赖于整个结果历史，并追问哪些结果路径可以作为**子博弈完美均衡**得到支持。

答案并不是反驳意见所预期的那个。

声誉并没有单独确立拉姆齐结果，也没有确认纳什结果。

它带来了均衡值的一个*连续统*，其中一些优于纳什结果，一些则差得多，而模型内部没有任何原则可以在其中做出选择。

这正是本讲座旨在确立的结论，萨金特用一句话表达出来：

> 我的结论是，可信计划的多重性用不可知论取代了悲观主义。

这对整个系列讲座的论证都很重要。

{doc}`phillips_two_stories` 反对*自然率理论胜利*这一说法的部分理由，正是建立在这一弱点之上：一个具有如此多均衡的理论几乎不能做出任何预测，因此依赖于政策制定者已经学到了正确均衡的故事，实际上是在为理论本身无法完成的工作做辩护。

### 我们要构建什么

所使用的机制是阿布鲁、皮尔斯和斯塔凯蒂 {cite}`APS1990` 的递归方法，它计算的不是一个最优*值*，而是整个均衡值的*集合*。

普通的动态规划迭代一个将延续*值*映射为值的算子。

APS 迭代的算子将延续值的*集合*映射为值的*集合*，而子博弈完美均衡值的集合正是它的最大不动点。

```{note}
将一个承诺值作为状态变量，然后在该状态下进行动态规划，这种技巧有时被称为**双重动态规划**。

这是 QuantEcon 关于 {doc}`斯塔克尔伯格计划 <advanced:dyn_stack>` 讲座以及 {cite:t}`Ljungqvist2012` 中拉姆齐问题递归处理方法的核心思想。

本讲座是一个精简的示例：状态是一个承诺值，我们所计算的对象是一个集合。

这里之所以用一个*标量*承诺值就足够，是因为私人部门的问题是静态的：公众的预测 $x_t$ 只涉及本期的通货膨胀，因此它完全由政府当期的指令决定。

而在 {cite:t}`chang1998credible` 的货币模型中，家庭的货币需求是前瞻性的，因此政府还必须向他们承诺货币的延续边际价值，这时状态变成了一对变量；{doc}`advanced:chang_ramsey` 和 {doc}`advanced:chang_credible` 用同样的集合迭代思想计算由此产生的二维集合。

{doc}`advanced:calvo_abreu` 在卡尔沃模型的线性二次版本中，构建了阿布鲁的"大棒加胡萝卜"计划，该计划在下文会出现。
```

接下来我们以三种方式使用这一机制。

我们通过猜测-验证法构造特定的均衡——纳什结果的无限重复、某种更优结果的无限重复，以及阿布鲁的"大棒加胡萝卜"策略，这一策略可能比纳什结果*更差*。

我们通过两个小型规划问题计算最优和最差均衡值，并将其与 APS 算子的直接迭代进行核对。

我们还展示了三种截然不同却都达到相同最差值的均衡，这是均衡概念约束力有多弱的最鲜明例证。

```{note}
萨金特本人给读者的建议值得转达：如果读者之前没有接触过这一理论，本章会显得困难；愿意接受其结论——即可信政策理论带来的是不可知论而非预测——的读者，可以直接跳转到 {doc}`phillips_adaptive`，而不会失去论证的线索。
```

让我们从导入模块开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import NamedTuple
```

## 重复经济

单期经济就是 {doc}`phillips_credibility` 中的那个经济。

令 $(U, y, x)$ 分别为失业率、通货膨胀率和公众对通货膨胀的预期。

失业率遵循预期增广菲利普斯曲线 $U = U^* - \theta(y - x)$，将政府的单期收益写成 $(x, y)$ 的函数为

```{math}
:label: cp_r

r(x, y) = -\frac{1}{2}\left[\bigl(U^* - \theta(y - x)\bigr)^2 + y^2\right].
```

政府对预期 $x$ 的单期最优反应为

```{math}
:label: cp_B

B(x) = \frac{\theta\,(U^* + \theta x)}{\theta^2 + 1},
```

纳什结果为 $y^N = \theta U^*$，拉姆齐结果为 $y^R = 0$。

现在有两点新内容。

第一，经济在 $t = 1, 2, \ldots$ 不断重复，政府按照下式对结果路径 $(x, y) = \{x_t, y_t\}_{t=1}^\infty$ 排序：

```{math}
:label: cp_value

V^g(x, y) = (1 - \delta)\sum_{t=1}^{\infty}\delta^{t-1} r(x_t, y_t),
\qquad \delta \in (0, 1).
```

因子 $(1-\delta)$ 使 $V^g$ 与单期收益处于相同的单位，从而使值与单期回报可以直接比较。

第二，通货膨胀被限制在一个有界区间 $Y = [0, y^\#]$ 内。

下界是拉姆齐结果；上界 $y^\#$ 使政府的问题变得非平凡，我们在寻找*最差*均衡时会看到这一点。

我们使用 {cite:t}`Sargent1999` 第 4 章数值例子中的参数值：$\theta = 1.25$、$U^* = 5.5$、$y^\# = 10$ 以及 $\delta = 0.95$。

这些值不同于 {doc}`phillips_credibility` 中的 $\theta = 1$、$U^* = 5$，这样我们就可以用书中的结果来核对我们的数值。

```{code-cell} ipython3
class Model(NamedTuple):
    θ: float = 1.25          # slope of the Phillips curve
    U_star: float = 5.5      # natural rate of unemployment
    y_max: float = 10.0      # y^#, the highest admissible inflation rate
    δ: float = 0.95          # discount factor


def r(m, x, y):
    "One-period government payoff when the public expects x and inflation is y."
    U = m.U_star - m.θ * (y - x)
    return -0.5 * (U**2 + y**2)


def B(m, x):
    "The government's one-period best response to expected inflation x."
    return m.θ * (m.U_star + m.θ * x) / (m.θ**2 + 1)


def y_nash(m):
    return m.θ * m.U_star
```

下面所有的推导都依赖于两个收益方案，两者都是公众已形成预期的单一通货膨胀率 $y$ 的函数。

第一个是**理性预期收益** $r(y, y)$：如果政府实现了公众所预期的通货膨胀，它得到的收益。

第二个是**背离收益** $r(y, B(y))$：如果公众预期为 $y$，而政府屈服于诱惑做出最优反应，它得到的收益。

两者都有值得记录的闭合形式，因为它们解释了后面的一切：

```{math}
:label: cp_schedules

r(y, y) = -\frac{1}{2}\left(U^{*2} + y^2\right),
\qquad
r\bigl(y, B(y)\bigr) = -\frac{\left(U^* + \theta y\right)^2}{2\left(1 + \theta^2\right)} .
```

```{code-cell} ipython3
def r_keep(m, y):
    "Payoff from delivering the expected inflation rate: r(y, y)."
    return -0.5 * (m.U_star**2 + y**2)


def r_cheat(m, y):
    "Payoff from best-responding to an expectation of y: r(y, B(y))."
    return -(m.U_star + m.θ * y)**2 / (2 * (1 + m.θ**2))


m = Model()
ys = np.linspace(0, m.y_max, 7)
print("closed forms agree with the definitions:",
      np.allclose(r_keep(m, ys), r(m, ys, ys)),
      np.allclose(r_cheat(m, ys), r(m, ys, B(m, ys))))
```

两者之间的差距是一个完全平方式：

```{math}
:label: cp_gap

r\bigl(y, B(y)\bigr) - r(y, y) = \frac{\left(y - y^N\right)^2}{2\left(1 + \theta^2\right)} \;\geq\; 0 .
```

因此，除了在纳什速率处，背离方案的曲线在任何地方都高于另一条曲线；而在纳什速率处，两条曲线*相切*：在该处背离的诱惑消失了，因为 $B(y^N) = y^N$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "两个收益方案：实现预期通货膨胀与背离预期通货膨胀"
    name: fig-cpol-schedules
---
grid = np.linspace(0, m.y_max, 400)
yN = y_nash(m)
v_R, v_N = r_keep(m, 0.0), r_keep(m, yN)
v_cand = r_cheat(m, m.y_max)                 # candidate worst value

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(grid, r_keep(m, grid), 'C0', lw=1.6, label='$r(y, y)$: deliver what is expected')
ax.plot(grid, r_cheat(m, grid), 'C1', lw=1.6, label='$r(y, B(y))$: deviate')
ax.plot(0.0, v_R, 'ko', ms=6)
ax.annotate('$v^R$', (0.0, v_R), textcoords='offset points', xytext=(8, 4))
ax.plot(yN, v_N, 'ko', ms=6)
ax.annotate('$v^N$', (yN, v_N), textcoords='offset points', xytext=(8, 4))
ax.plot(m.y_max, v_cand, 'ko', ms=6)
ax.annotate(r'$r(y^\#, B(y^\#))$', (m.y_max, v_cand), textcoords='offset points',
            xytext=(-70, 6))
ax.axvline(yN, color='k', lw=0.6, ls=':')
ax.set_xlabel('inflation rate $y$ expected by the public')
ax.set_ylabel('one-period payoff')
ax.legend(loc='lower left')
plt.show()
```

两条曲线之间的垂直差距，是政府在单期内通过背离所能获得的收益。

在 $y^N$ 以下（包括拉姆齐结果在内）的区间，$B(y) > y$，因此诱惑是使通货膨胀*高于*预期。

在 $y^N$ 以上的区间，$B(y) < y$，诱惑则是使通货膨胀*低于*预期。

根据 {eq}`cp_gap`，这一差距关于 $y^N$ 是对称的，因此仅凭差距本身并不能说明最差情形出现在 $Y$ 的顶端。

真正使*最差*均衡落在 $Y$ 顶端的，是背离方案本身的水平。

$r(y, B(y))$ 随 $y$ 单调下降，因此当公众预期为 $y^\#$ 时，政府通过背离所能为自己保证的最低收益也是最小的。

图中标出的点 $r(y^\#, B(y^\#))$ 是我们对最差均衡值的第一个候选。

## 递归策略与承诺值

政府的策略必须能够使今天的行动依赖于整个过去。

携带完整的历史是不可管理的，因此我们效仿 {cite:t}`Sargent1999`，将注意力限制在具有递归表示的策略上——这一限制不会带来任何损失，因为它不会排除任何均衡*值*。

```{prf:definition} 递归政府策略
:label: cp_strategy

一个递归政府策略是一对函数 $\sigma = (\sigma_1, \sigma_2)$，连同一个初始条件 $v_1$，具有以下结构：

$$
v_1 \in \mathbb{R} \text{ given}, \qquad
y_t = \sigma_1(v_t), \qquad
v_{t+1} = \sigma_2(v_t, x_t, y_t),
$$

其中 $v_t$ 是一个状态变量，用于概括 $t$ 之前结果的历史。
```

私人部门的每个成员都知道 $v_t$ 及策略 $\sigma$，因此预测

```{math}
:label: cp_expect

x_t = \sigma_1(v_t) .
```

方程 {eq}`cp_expect` 将理性预期内置到私人部门中：公众在均衡路径上永远不会受到意外。

一个策略 $(\sigma, v_1)$ 生成一整条结果路径，从而通过 {eq}`cp_value` 生成一个值 $V^g(\sigma, v_1)$。

可信政策理论此刻做了一件乍看之下像是把戏的事情。

它通过要求状态变量*就是*它所生成的值，将过去与未来联系起来，

```{math}
:label: cp_fixedpt

v = V^g(\sigma, v).
```

于是 $v$ 同时承担两份职责。

在 $v_{t+1} = \sigma_2(v_t, x_t, y_t)$ 中，它是一个记录已发生事情的记账工具。

在 {eq}`cp_fixedpt` 中，它是一个**承诺值**——政府在进入这一期时所应得的贴现未来。

以第二种方式来理解，正是使这一机制运作起来的关键：策略给予政府当前和未来的结果，使其*希望*去做别人对它的预期。

```{note}
$\sigma$ 有两种解读，在一个均衡内部，二者是无法区分的。

它既可以是政府所选择的决策规则，也可以是政府所遵循的公众预期体系的描述。

我们会在解释一节回到这一含糊之处，因为正是从这里产生了该理论的不可知论。
```

### 历史渊源

将可信性形式化是二十世纪八十年代的成就，但对这一思想的精深理解要古老得多。

1784年，财政大臣雅克·内克尔向路易十六解释，为什么一位可以随意拖欠债务的绝对君主发现借款很难 {cite:p}`SargentVelde1995`：

> 因此，只有通过对君主意图给予保证，并证明没有任何动机能诱使他违背其义务，才能重新点燃或维持公众信任。

这句话的每一个从句都出现在现代定义中。

一位君主要证明他永远不会违背自己的义务，办法是永远不*想*违背它们——通过遵从一套自带激励约束的公众预期体系，使得在每一个日期、每一种情形下，他若确认预期而非令预期落空，能获得更高的当期收益加延续值。

```{prf:definition} 子博弈完美均衡
:label: cp_spe

一个具有承诺值 $v$ 的递归策略是**子博弈完美均衡**（SPE），当且仅当

(a) 对于每一个 $\eta \in Y$，$\sigma_2(v, \sigma_1(v), \eta)$ 本身就是某个子博弈完美均衡可达到的值；且

(b) 记 $y = \sigma_1(v)$，则

$$
v = (1-\delta)\, r(y, y) + \delta\, \sigma_2(v, y, y)
  \;\geq\; (1-\delta)\, r(y, \eta) + \delta\, \sigma_2(v, y, \eta),
  \qquad \forall\, \eta \in Y .
$$
```

该定义为一个均衡附加了四个对象：一个承诺值 $v$；一个第一期结果 $(y, y)$；若遵从规定结果被观察到，一个延续值 $v'$；以及若未被观察到，另一个延续值 $\tilde v$。

按照这些记号，条件 (b) 可写为

$$
v = (1-\delta) r(y, y) + \delta v' \;\geq\; (1-\delta) r(y, \eta) + \delta \tilde v,
\qquad \forall \eta \in Y,
$$

它简单地说明政府遵从要比背离做得更好。

条件 (a) 说明延续值本身必须是均衡值。

该定义是循环的——均衡值同时出现在等式两侧——而这种循环性正是递归性所换来的，也是 APS 学会加以利用的东西。

(cpol_aps)=
## 阿布鲁-皮尔斯-斯塔凯蒂方法

动态规划通过迭代贝尔曼方程来计算最优值函数，这是一个将明天的值函数转化为今天的值函数的映射。

APS 将这一思想应用于均衡*集合*。

从一个候选延续值集合 $W \subset \mathbb{R}$ 开始。

选取第一期理性预期结果 $(y, y)$ 以及来自 $W$ 的两个延续值：一个用来奖励遵从的 $w_1$，一个用来惩罚背离的 $w_2$。

如果 $w_1$ 足够高，$w_2$ 足够低，这对值就支持 $y$，并带来以下值：

$$
w = (1-\delta) r(y, y) + \delta w_1 \;\geq\; (1-\delta) r(y, \eta) + \delta w_2,
\qquad \forall \eta \in Y .
$$

```{prf:definition} 可容许性
:label: cp_admissible

若存在 $w_1, w_2 \in W$，使得对所有 $\eta \in Y$ 都有
$w = (1-\delta) r(y,y) + \delta w_1 \geq (1-\delta) r(y,\eta) + \delta w_2$，
则称 $(y, w)$ 关于延续值集合 $W$ 是**可容许的**。
```

令 $\mathcal{T}(W)$ 收集所有可容许对的 $w$ 分量。

APS 和 {cite:t}`Sargent1999` 将这一算子称为 $B$；我们用 $\mathcal{T}$ 来表示它，以便将 $B$ 保留给政府的最优反应。

这一构造内置了 {prf:ref}`cp_spe` 的条件 (b)，但忽略了条件 (a)，因为延续值是从一个任意集合中抽取的。

```{prf:definition} 自我生成性
:label: cp_selfgen

若 $W \subseteq \mathcal{T}(W)$，则称潜在延续值的集合 $W$ 是**自我生成的**。
```

自我生成集合中的每一个值都由取自同一集合的延续值来支持——这正是条件 (a)。

APS 证明了四件事：

1. SPE 值的集合 $V$ 是最大的自我生成集合；
1. $\mathcal{T}$ 将紧集映射为紧集；
1. $\mathcal{T}$ 是单调的：$W_2 \subseteq W_1$ 蕴含 $\mathcal{T}(W_2) \subseteq \mathcal{T}(W_1)$；
1. 从任意满足 $V \subseteq \mathcal{T}(W_0) \subseteq W_0$ 的紧集 $W_0$ 出发，迭代
   $W_j = \mathcal{T}(W_{j-1})$ 单调收敛于 $V = \mathcal{T}(V)$
   （{cite:t}`APS1990`，定理 5）。

最后一项中要求 $V \subseteq \mathcal{T}(W_0)$ 不可省略，尽管该教材对此定理的表述省略了这一条件。

单点集合 $W_0 = \{v^N\}$ 满足 $\mathcal{T}(W_0) \subseteq W_0$，但它却是 $\mathcal{T}$ 的一个不动点，远小于 $V$（练习 {ref}`cpol_ex6`）。

一个安全的选择是取所有可行值的集合，$W_0 = [r(y^\#, y^\#),\, 0]$。

它包含 $V$，而且 $\mathcal{T}$ 将其映射到自身，因此由单调性可得
$V = \mathcal{T}(V) \subseteq \mathcal{T}(W_0) \subseteq W_0$。

### 计算算子

有两个简化使得在此处计算 $\mathcal{T}$ 变得容易。

因为政府的诱惑在其做出最优反应时最强，{prf:ref}`cp_admissible` 中的约束在 $\eta = B(y)$ 处约束最紧，所以我们可以用这一单一背离来替代"对所有 $\eta$"。

而且，由于更低的惩罚会放松约束，我们总可以将 $w_2$ 设为 $W$ 中的最小元素。

记 $W = [\underline w, \overline w]$，则只要 $W$ 中存在某个 $w_1$ 满足

```{math}
:label: cp_bound

w_1 \;\geq\; \frac{(1-\delta)\left[r(y, B(y)) - r(y, y)\right]}{\delta} + \underline w
\;\equiv\; \ell(y) ,
```

结果 $y$ 就是可容许的，它所生成的值遍历区间

$$
\left[(1-\delta) r(y,y) + \delta \max(\underline w, \ell(y)),\;
(1-\delta) r(y,y) + \delta \overline w\right] .
$$

```{code-cell} ipython3
def T_operator(m, W, n_grid=4001):
    """
    One application of the APS operator 𝒯 to an interval W = [w_lo, w_hi].

    Returns the new interval, or None if no first-period outcome is admissible.
    """
    w_lo, w_hi = W
    y = np.linspace(0.0, m.y_max, n_grid)
    ℓ = (1 - m.δ) * (r_cheat(m, y) - r_keep(m, y)) / m.δ + w_lo   # the bound ℓ(y) of cp_bound
    ok = ℓ <= w_hi                                                 # admissible outcomes
    if not ok.any():
        return None
    keep = (1 - m.δ) * r_keep(m, y)
    lows = keep[ok] + m.δ * np.maximum(w_lo, ℓ[ok])
    highs = keep[ok] + m.δ * w_hi
    return lows.min(), highs.max()


def solve_aps(m, W0=None, tol=1e-12, max_iter=10_000):
    """
    Iterate the APS operator from W0 until it settles.

    The default W0 is the set of all feasible values, [r(y^#, y^#), 0].
    """
    W = (r_keep(m, m.y_max), 0.0) if W0 is None else W0
    for it in range(max_iter):
        W_new = T_operator(m, W)
        if W_new is None:
            return None, it
        if max(abs(W_new[0] - W[0]), abs(W_new[1] - W[1])) < tol:
            return W_new, it
        W = W_new
    return W, max_iter
```

我们从 $W_0 = [r(y^\#, y^\#),\, 0]$ 出发，然后进行迭代。

```{code-cell} ipython3
W, iters = solve_aps(m)
print(f"converged in {iters} iterations")
print(f"set of SPE values V = [{W[0]:.4f}, {W[1]:.4f}]")
```

## 最优与最差均衡

APS 的迭代过程具有一般性，但对于*为什么*该集合最终落在这个位置，它没有告诉我们太多。

对于这一经济，两个端点都可以手工求得，其论证具有启发性。

### 最差

最差均衡值满足以下方程：

$$
\underline v = \min_{y \in Y,\; v_1 \in V}\ \left[(1-\delta) r(y,y) + \delta v_1\right]
\quad\text{subject to}\quad
(1-\delta) r(y,y) + \delta v_1 \geq (1-\delta) r(y, B(y)) + \delta \underline v ,
$$

在这里，*最差*值被用作背离时的延续值——这是最严厉的惩罚。

对给定的 $y$，当 $v_1$ 取约束所允许的最低值时，目标函数最小，因此在最小值处约束恰好约束（constraint binds），此时方程两侧都简化为对使之最小化的 $y$ 有 $\underline v = r(y, B(y))$。

如果每个 $y \in Y$ 都可以使用，那么找到最差均衡问题就归结为一个一维问题：

```{math}
:label: cp_worst

\underline v = \min_{y \in Y}\ r\bigl(y, B(y)\bigr)
= -\frac{\left(U^* + \theta y^\#\right)^2}{2\left(1 + \theta^2\right)} ,
```

其中闭合形式源自 {eq}`cp_schedules`，最小化行动是 $y^\#$，因为背离收益随 $y$ 单调下降。

但 {eq}`cp_worst` 仅仅是一个*候选*值。

要以第一期处于 $y^\#$ 的方式实现它，该策略必须以满足 $\underline v = (1-\delta) r(y^\#, y^\#) + \delta v_1$ 的延续值 $v_1$ 来奖励遵从行为，

```{math}
:label: cp_reward

v_1 = \frac{r\bigl(y^\#, B(y^\#)\bigr) - (1-\delta)\, r(y^\#, y^\#)}{\delta} ,
```

而且这个奖励本身必须是一个均衡值：$v_1 \leq \overline v$。

这一检验是本书构造方法的最后一步：计算候选值，计算该候选值所隐含的最优值 $\overline v$（见下文），并验证 $v_1 \leq \overline v$。

这项检验可能会失败。

奖励 {eq}`cp_reward` 按 $1/\delta$ 的速度增长：对于一个缺乏耐心的政府，无法承诺足够大的未来奖励以使其接受处于 $y^\#$ 的一期。

此时最差均衡会从一个更低的通货膨胀率开始，两个规划问题必须联合求解，而我们让 APS 迭代来完成这项工作。

现在上界 $y^\#$ 的作用清晰起来。

{eq}`cp_reward` 中 $y^{\#2}$ 的系数为 $\left[(1-\delta) - \theta^2/(1+\theta^2)\right] / (2\delta)$。

因此，当 $\delta > 1/(1+\theta^2)$ 时，随着 $y^\#$ 增大，所需的奖励不断下降且没有下限，检验持续通过，此时 $y^\#$ 正是使最差惩罚保持有限的原因。

对于一个不那么有耐心的政府，该检验最终会失败，超过某一点后再提高 $y^\#$ 就不会再有任何变化。

```{prf:proposition} 最差 SPE 是自我实施的
:label: cp_prop_worst

在最差均衡中，背离之后的延续值等于初始承诺值，因此一次背离只是简单地重新开始均衡。
```

### 最优

给定 $\underline v$ 作为威胁，最优值满足

```{math}
:label: cp_best

\overline v = \max_{y \in Y}\ r(y, y)
\quad\text{subject to}\quad
r(y, y) \geq (1-\delta) r\bigl(y, B(y)\bigr) + \delta \underline v ,
```

这里我们利用了这样一个事实：最优值必须以自身来奖励遵从，因此
$\overline v = (1-\delta) r(y,y) + \delta \overline v = r(y,y)$。

```{prf:proposition} 最优 SPE 是自我奖励的
:label: cp_prop_best

在最优均衡中，遵从之后的延续值等于承诺值。
```

下面的代码实现了这两个规划问题，并包含上述检验。

```{code-cell} ipython3
def next_value(m, v):
    "Solve v = (1-δ) r(y^#, y^#) + δ v' for the continuation value v'."
    return (v - (1 - m.δ) * r_keep(m, m.y_max)) / m.δ


def best_value(m, v_lo, n_grid=200_001):
    "The best SPE value, given the worst value as the punishment threat."
    y = np.linspace(0.0, m.y_max, n_grid)
    feasible = r_keep(m, y) >= (1 - m.δ) * r_cheat(m, y) + m.δ * v_lo
    if not feasible.any():
        return None, None
    vals = np.where(feasible, r_keep(m, y), -np.inf)
    k = vals.argmax()
    return vals[k], y[k]


def worst_value(m):
    """
    The worst SPE value and the action that attains it.

    The candidate r(y^#, B(y^#)) of cp_worst is accepted if the reward v_1 of
    cp_reward is no larger than the best value the candidate implies.
    Otherwise the worst value comes from the APS iteration, and it is attained
    at the highest action whose required reward ℓ(y) of cp_bound equals the
    best value (solved for y using cp_gap).
    """
    v_cand = r_cheat(m, m.y_max)
    v_best, _ = best_value(m, v_cand)
    if next_value(m, v_cand) <= v_best:
        return v_cand, m.y_max
    (w_lo, w_hi), _ = solve_aps(m)
    y_hat = y_nash(m) + np.sqrt(2 * (1 + m.θ**2) * m.δ * (w_hi - w_lo) / (1 - m.δ))
    return w_lo, y_hat


v_lo, y_sharp = worst_value(m)
v_hi, y_best = best_value(m, v_lo)

print(f"worst value  v_lo = {v_lo:.4f}   attained with y = {y_sharp:.2f}")
print(f"reward required by the candidate  v_1 = {next_value(m, v_lo):.4f}"
      f"  <= v_hi? {next_value(m, v_lo) <= v_hi}")
print(f"best  value  v_hi = {v_hi:.4f}   attained with y = {y_best:.2f}")
print(f"APS iteration gave [{W[0]:.4f}, {W[1]:.4f}]")
print(f"the two agree: {np.allclose(W, (v_lo, v_hi), atol=1e-6)}")
```

规划问题和集合迭代结果一致，在此贴现因子下，最优均衡正是拉姆齐结果本身。

在 $\delta = 0.95$ 时检验很容易通过，但并非对每一个贴现因子都是如此。

```{code-cell} ipython3
print("    δ    candidate   reward v_1   best given candidate   check   worst value  at y")
for δ in (0.02, 0.04, 0.05, 0.2, 0.95):
    md = m._replace(δ=δ)
    v_cand = r_cheat(md, md.y_max)
    v_1, v_best = next_value(md, v_cand), best_value(md, v_cand)[0]
    v_worst, y_worst = worst_value(md)
    print(f"{δ:6.2f} {v_cand:11.4f} {v_1:12.4f} {v_best:16.4f} {str(v_1 <= v_best):>11}"
          f" {v_worst:12.4f} {y_worst:6.2f}")
```

对于最小的两个贴现因子，候选值所需要的奖励超过了最优均衡值。

此时最差均衡会从低于 $y^\#$ 的水平开始，其取值高于
$r(y^\#, B(y^\#))$；练习 {ref}`cpol_ex2` 将求出这一现象开始出现时的贴现因子。

下面这段代码针对一个有耐心的政府和一个缺乏耐心的政府，考察上界 $y^\#$ 变化的影响。

```{code-cell} ipython3
print(f"1/(1 + θ²) = {1 / (1 + m.θ**2):.3f}")
for δ in (0.2, 0.95):
    for y_max in (10.0, 30.0, 100.0):
        v_worst, y_worst = worst_value(m._replace(δ=δ, y_max=y_max))
        print(f"δ = {δ:4.2f}, y^# = {y_max:5.1f}:  worst value {v_worst:10.2f}"
              f"  attained with y = {y_worst:6.2f}")
```

对于有耐心的政府，随着 $y^\#$ 增大，最差均衡值持续下降，并且总是在 $y^\#$ 处取得。

而对于缺乏耐心的政府，这一下降会停止：一旦 $y^\#$ 足够大，无论上界如何，最差均衡都会从低于 $y^\#$ 的同一个通货膨胀率开始（输出中的细微差异来自 `T_operator` 中的网格）。

如果没有通货膨胀的上界，几乎任何结果都可以被支持，但这仅对有耐心的政府成立。

## 递归均衡示例

现在我们按照文献发现它们的顺序，通过猜测-验证法构造具体的均衡。

### 纳什结果的无限重复

最简单的均衡就是永远重复单期纳什结果。

取 $v_1 = v^N = r(y^N, y^N)$，对每个 $v$ 都有 $\sigma_1(v) = y^N$，且对每个 $(v, x, y)$ 都有
$\sigma_2(v, x, y) = v^N$。

条件 (a) 由构造自动满足，而条件 (b) 简化为
$r(y^N, y^N) \geq r(y^N, B(y^N))$，此式*恰好以等式成立*，因为 $y^N$ 是最优反应映射的不动点。

这里没有任何东西能约束政府，也不需要：它已经在做自己最想做的事了。

这就是重复博弈的马尔可夫完美均衡。

这个经济体没有与收益相关的状态变量，因此马尔可夫策略无法以任何变量为条件，政府只是每期都做出其单期最优反应
（{doc}`markov_perf` 研究了确实具有状态变量的动态博弈的马尔可夫完美均衡）。

### 更优结果的无限重复

设 $v^b = r(y^b, y^b) > v^N$ 为一个优于纳什的值，假设

```{math}
:label: cp_bg

r\bigl(y^b, B(y^b)\bigr) - r(y^b, y^b)
\;\leq\; \frac{\delta}{1-\delta}\left(v^b - v^N\right).
```

左边是背离的单期收益；右边是永久回归纳什带来的贴现损失。

当 {eq}`cp_bg` 成立时，规定 $y^b$（当承诺值为 $v^b$ 时）并在任何背离后回归纳什的策略是一个 SPE。

{cite:t}`BarroGordon1983` 和 {cite:t}`stokey1989reputation` 研究了 $y^b = y^R$ 的情形：预期中的回归纳什永远支持拉姆齐结果。

至于它是否成立取决于耐心程度，对于这个经济体，其阈值具有一个非常简洁的形式。

在 {eq}`cp_bg` 中设 $y^b = 0$，并使用闭合形式 {eq}`cp_schedules`，每一个 $U^*$ 和每一个尺度因子都会消去，只剩下

```{math}
:label: cp_cutoff

\delta \;\geq\; \delta^\star = \frac{1}{2 + \theta^2} .
```

```{code-cell} ipython3
def supports_ramsey_by_nash(m):
    "Does reversion to Nash sustain Ramsey forever?"
    gain = r_cheat(m, 0.0) - r_keep(m, 0.0)
    loss = m.δ / (1 - m.δ) * (r_keep(m, 0.0) - r_keep(m, y_nash(m)))
    return gain <= loss


δ_star = 1 / (2 + m.θ**2)
print(f"closed-form cutoff  δ* = 1/(2 + θ²) = {δ_star:.6f}\n")
for δ in (0.20, 0.27, δ_star - 1e-6, δ_star + 1e-6, 0.50, 0.95):
    md = m._replace(δ=δ)
    print(f"  δ = {δ:.8f}: Nash reversion supports Ramsey? "
          f"{supports_ramsey_by_nash(md)}")
```

在 $\delta^\star \approx 0.28$ 以上，回归纳什的威胁足以将政府维持在零通货膨胀水平；在此之下，威胁则过于薄弱。

(cpol_abreu)=
### 更差的均衡：阿布鲁的大棒加胡萝卜

当回归纳什的威胁不够强时，某个*更差*的均衡可能可以用作惩罚。

{cite:t}`Abreu1986` 在研究寡头垄断时、{cite:t}`Stokey1991` 在研究可信公共政策时，分别展示了如何构造这样的均衡；
{cite:t}`Abreu` 发展了此类最优惩罚的一般理论。

该策略是一种大棒加胡萝卜策略：要求政府以最大速率 $y^\#$ 通货膨胀一期——这是大棒——之后奖励它以永远的拉姆齐结果。

此值为

```{math}
:label: cp_abreu

\tilde v = (1-\delta)\, r(y^\#, y^\#) + \delta\, v^R ,
```

并且在任一阶段的任何时点发生背离都会*重新开始*整个策略，因此背离后的延续值始终就是 $\tilde v$ 本身。

该策略是 SPE，当且仅当两个阶段都不会诱使政府背离。

*大棒阶段。* 在第一期，激励约束为

$$
\tilde v = (1-\delta)\, r(y^\#, y^\#) + \delta\, v^R
\;\geq\; (1-\delta)\, r\bigl(y^\#, B(y^\#)\bigr) + \delta\, \tilde v .
$$

将 $\delta \tilde v$ 移到左边并除以 $1-\delta$，得到
$\tilde v \geq r\bigl(y^\#, B(y^\#)\bigr)$：大棒的值必须至少等于候选最差值 {eq}`cp_worst`。

*胡萝卜阶段。* 一旦处于拉姆齐状态，政府必须更愿意留在那里，而不是攫取来自 $B(0)$ 的单期收益然后重新开始大棒：

```{math}
:label: cp_abreu_carrot

v^R \;\geq\; (1-\delta)\, r\bigl(0, B(0)\bigr) + \delta\, \tilde v .
```

代入 {eq}`cp_abreu`，并利用 {eq}`cp_gap` 以及 $v^R - r(y^\#, y^\#) = y^{\#2}/2$，这两个约束变为

$$
\delta \;\geq\; \frac{\left(y^\# - y^N\right)^2}{(1+\theta^2)\, y^{\#2}}
\quad \text{（大棒约束）},
\qquad
\delta \;\geq\; \delta_A = \frac{\theta^2 U^{*2}}{(1+\theta^2)\, y^{\#2}}
\quad \text{（胡萝卜约束）} .
$$

这里 $y^N > y^\# - y^N$，因此胡萝卜约束是起作用的约束。

```{code-cell} ipython3
def abreu_value(m):
    "Value of the stick-and-carrot strategy: one period at y^#, then Ramsey."
    return (1 - m.δ) * r_keep(m, m.y_max) + m.δ * r_keep(m, 0.0)


def supports_ramsey_with_threat(m, threat):
    "Does a permanent threat value sustain Ramsey forever?"
    return r_keep(m, 0.0) >= (1 - m.δ) * r_cheat(m, 0.0) + m.δ * threat


def is_abreu_spe(m):
    """
    Check both incentive constraints of the stick-and-carrot strategy, in which
    any deviation restarts the strategy.

    Returns (stick constraint holds, carrot constraint holds, both hold).
    """
    v_tilde = abreu_value(m)
    stick = v_tilde >= (1 - m.δ) * r_cheat(m, m.y_max) + m.δ * v_tilde
    carrot = supports_ramsey_with_threat(m, v_tilde)        # cp_abreu_carrot
    return stick, carrot, stick and carrot


δ_stick = (m.y_max - y_nash(m))**2 / ((1 + m.θ**2) * m.y_max**2)
δ_A = m.θ**2 * m.U_star**2 / ((1 + m.θ**2) * m.y_max**2)
print(f"stick holds for δ >= {δ_stick:.4f};  carrot holds for δ >= δ_A = {δ_A:.4f}\n")
for δ in (0.10, 0.20, 0.95):
    md = m._replace(δ=δ)
    va, vn = abreu_value(md), r_keep(md, y_nash(md))
    stick, carrot, spe = is_abreu_spe(md)
    print(f"δ = {δ:4.2f}:  v_abreu = {va:8.4f}   v^N = {vn:8.4f}   "
          f"worse than Nash? {str(va < vn):5}   stick? {str(stick):5}   "
          f"carrot? {str(carrot):5}   SPE? {spe}")
```

在 $\delta = 0.95$ 时，大棒加胡萝卜策略的值反而*优于*纳什，因为单个糟糕的时期被永久的拉姆齐结果远远抵消了。

在 $\delta = 0.2$ 时，它则远*差于*纳什——这正是要点所在。

在 $\delta = 0.1$ 时，政府仍愿意接受大棒，但不会留在拉姆齐状态：通货膨胀带来的单期收益超过了重新开始大棒所带来的贴现成本。

因此在 $\delta = 0.1$ 时，该策略不是一个均衡，即便其第一期约束成立。

胡萝卜约束 {eq}`cp_abreu_carrot` 问的是威胁 $\tilde v$ 是否能够永远支持拉姆齐结果。

比纳什更差的惩罚是比回归纳什更强的威慑，因此它可以支持纳什回归所无法支持的结果。

```{code-cell} ipython3
m2 = m._replace(δ=0.20)

print(f"at δ = 0.2:")
print(f"  threat = Nash    ({r_keep(m2, y_nash(m2)):8.4f}): "
      f"supports Ramsey? {supports_ramsey_with_threat(m2, r_keep(m2, y_nash(m2)))}")
print(f"  threat = Abreu   ({abreu_value(m2):8.4f}): "
      f"supports Ramsey? {supports_ramsey_with_threat(m2, abreu_value(m2))}")
```

在 $\delta = 0.2$ 时，一个缺乏耐心的政府无法被纳什的前景约束住，但*可以*被阿布鲁大棒的前景约束住。

最差均衡值 $\underline v$ 是更严厉的惩罚，它能够在更低的贴现因子下仍然将拉姆齐结果维持住
（练习 {ref}`cpol_ex4`）。

有两个定义可以命名我们刚刚遇到的结构。

```{prf:definition} 自我实施与自我奖励
:label: cp_selfenf

一个递归 SPE 若背离后的延续值等于初始承诺值，则称为**自我实施的**；若遵从后的延续值等于承诺值，则称为**自我奖励的**。
```

阿布鲁的大棒加胡萝卜策略是自我实施的；对优于纳什的结果进行无限重复则是自我奖励的。

{prf:ref}`cp_prop_worst` 和 {prf:ref}`cp_prop_best` 表明，这两种结构并非奇特案例：它们恰恰就是极端均衡的样子。

## 多重性

这一理论内蕴含着两层多重性。

存在一个均衡*值*的连续统，而且——更为尖锐的一点是——许多不同的结果*路径*都能达到同一个值。

为了说明第二点，我们构造了三个均衡，它们都得到最差值 $\underline v$。

三者都以相同的方式开始：第一期承诺值为 $\underline v$，规定的行动是 $y^\#$，延续值 $v_2$ 满足
$\underline v = (1-\delta) r(y^\#, y^\#) + \delta v_2$。

它们的区别在于此后的做法。

以下代码复用了最差值一节中的 `next_value`。

```{code-cell} ipython3
def action_delivering(m, v):
    "The inflation rate ỹ with r(ỹ, ỹ) = v."
    return np.sqrt(max(-2.0 * v - m.U_star**2, 0.0))


def method_1(m, v_lo, v_hi, max_steps=500):
    "Climb at y^# until one more step would overshoot v_hi, then settle at v_hi."
    v, y = [v_lo], []
    for _ in range(max_steps):
        nxt = next_value(m, v[-1])
        if nxt > v_hi:
            break
        y.append(m.y_max)
        v.append(nxt)
    target = (v[-1] - m.δ * v_hi) / (1 - m.δ)     # r(ỹ, ỹ) for the switching period
    y.append(action_delivering(m, target))
    v.append(v_hi)
    y.append(action_delivering(m, v_hi))          # stay at v_hi forever after
    return np.array(v), np.array(y)


def method_2(m, v_lo, max_steps=500):
    "Climb at y^# only until the promised value first exceeds v^N, then freeze."
    v_N = r_keep(m, y_nash(m))
    v, y = [v_lo], []
    for _ in range(max_steps):
        nxt = next_value(m, v[-1])
        y.append(m.y_max)
        v.append(nxt)
        if nxt > v_N:
            break
    y.append(action_delivering(m, v[-1]))          # freeze at v** forever
    return np.array(v), np.array(y)


def method_3(m, v_lo):
    "One period at y^#, then freeze immediately at v_2."
    v_2 = next_value(m, v_lo)
    return (np.array([v_lo, v_2, v_2]),
            np.array([m.y_max, action_delivering(m, v_2),
                      action_delivering(m, v_2)]))
```

下图绘制了这三条路径上的承诺值与通货膨胀率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 达到相同最差值的三种子博弈完美均衡（原书图4.2-4.4）
    name: fig-cpol-three-methods
---
paths = {'method 1': method_1(m, v_lo, v_hi),
         'method 2': method_2(m, v_lo),
         'method 3': method_3(m, v_lo)}

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex='col')
for k, (name, (v, y)) in enumerate(paths.items()):
    axes[0, k].plot(range(1, len(v) + 1), v, 'C0o-', ms=3.5, lw=1)
    axes[0, k].axhline(v_lo, color='k', ls='--', lw=0.8)
    axes[0, k].axhline(v_hi, color='C2', ls=':', lw=0.8)
    axes[0, k].set_title(f'{name}: continuation values', fontsize=10)
    axes[1, k].plot(range(1, len(y) + 1), y, 'C1o-', ms=3.5, lw=1)
    axes[1, k].set_ylim(-0.4, m.y_max + 0.4)
    axes[1, k].set_xlabel('$t$')
    axes[1, k].set_title(f'{name}: inflation', fontsize=10)
axes[0, 0].set_ylabel('promised value $v_t$')
axes[1, 0].set_ylabel('inflation $y_t$')
plt.tight_layout()
plt.show()
```

下面给出了它们的持续期长度与终端值。

```{code-cell} ipython3
print(f"Nash value {r_keep(m, y_nash(m)):.4f} at inflation {y_nash(m):.4f};  "
      f"worst value {v_lo:.4f}\n")
for name, (v, y) in paths.items():
    print(f"{name:9s}: {len(y):3d} periods before settling,  "
          f"first-period value {v[0]:.4f},  "
          f"terminal value {v[-1]:8.4f} at inflation {y[-1]:.4f}")
```

这三条路径几乎完全不同。

第一条路径以最大通货膨胀速率攀升约六十期，然后才降至拉姆齐结果。

第二条路径攀升的时间较短，随后永远冻结在一个刚刚*优于*纳什的值上，由一个略低于纳什速率的通货膨胀率所支撑。

第三条路径以最大速率通货膨胀一期，然后立即稳定下来，其值只比最差值高一点点。

三者都是子博弈完美的，且三者给政府带来的值完全相同。

均衡概念对于这三者中哪一个描述了真实世界，无话可说。

## 数值示例

以下是 {cite:t}`Sargent1999` 第4章报告的数值示例，我们从头进行了计算。

```{code-cell} ipython3
rows = [
    ("$\\theta$", m.θ, ""),
    ("$U^*$", m.U_star, ""),
    ("$y^\\#$", m.y_max, ""),
    ("$\\delta$", m.δ, ""),
    ("$y^N$ (Nash inflation)", y_nash(m), "6.8750"),
    ("$y^R$ (Ramsey inflation)", 0.0, "0"),
    ("$v^R$", r_keep(m, 0.0), "-15.1250"),
    ("$v^N$", r_keep(m, y_nash(m)), "-38.7578"),
    ("$\\underline v$", v_lo, "-63.2195"),
    ("$v_{\\rm abreu}$", abreu_value(m), "-17.6250"),
]
pd.DataFrame(rows, columns=["object", "computed", "reported in the book"]).set_index("object")
```

书中还报告了纳什反转临界值以及 $\delta = 0.2$ 时胡萝卜加大棒策略的值。

```{code-cell} ipython3
print(f"cutoff discount factor δ*        : {1 / (2 + m.θ**2):.4f}  (book reports 0.2807)")
print(f"Abreu stick-and-carrot at δ = 0.2: {abreu_value(m._replace(δ=0.2)):.4f}"
      f"  (book reports -55.125)")
```

每一项都与书中一致。

## 解读

关于可信计划的文献，对于 {doc}`phillips_two_stories` 中*自然率理论胜利*的说法喜忧参半。

它确实将政府从 {doc}`phillips_credibility` 的悲观境地中拯救了出来：优于纳什的结果是可以实现的，而且在合理的贴现因子下，拉姆齐结果本身就是一个均衡。

但它拯救得太多了。

比纳什更差的值也是均衡，而两个极端之间还存在一个连续统。

众多结果在经验上使模型变得不明确，也消解了早期理性预期研究者从这一假设中所期望得到的东西——消除描述预期的自由参数。

### 到底是谁的预期？

在1979年对保罗·麦克拉肯编辑的一份经合组织（OECD）报告的评论中，卢卡斯抗议该报告的建议——"各国政府应努力促进良好的预期"，仿佛预期是一套额外的政策工具。

在1979年，这一抗议是有的放矢的：当时理性预期模型将政府政策视为外生的，并将预期作为政府政策的一个*函数*，通过跨方程约束加以关联。

可信政策理论以某种方式改变了这一局面，但对双方都不能算作清楚的支持。

它将预期体系变成了会影响结果的自由参数——但这些参数是模型内部没有任何人可以选择的。

政府遵从关于其自身行为的均衡预期。

在一个均衡内部，政府的策略同时是一条决策规则，也是对公众预期的描述，而两者是无法分开的。

正如萨金特所言：麦克拉肯报告的作者们相信多重性和操纵性，而卢卡斯对两者都持怀疑态度；关于可信计划的文献支持多重性，但不支持操纵性。

(cpol_remedies)=
### 补救措施

单凭声誉是反通货膨胀政策的一个薄弱基础，这种薄弱之处激发了改变博弈规则、而非寄望于博弈内部出现一个好均衡的提议。

{cite:t}`Rogoff1985` 提议将货币政策委托给一个比社会整体更不关心失业的人。

将其委托给一个甚至不了解*暂时性*权衡取舍的人，同样有效；艾伦·布林德后来提出了一个相关的方案：委托给一个了解自然率、且从不希望失业率偏离该水平的当局。

练习 {ref}`pc_ex3` 在 {doc}`phillips_credibility` 的单期经济模型中详细推导了这两种补救措施，包括当存在需要加以稳定的冲击时，罗格夫方案所付出的代价。

维持一个具有不同通货膨胀-失业偏好的潜在中央银行家人才库——其中一些像罗格夫方案中那样保守，另一些则像基德兰德和普雷斯科特模型中的那种当局——也能改善结果 {cite:p}`BarroGordon1983,Ball1995`。

当公众不知道在位的是哪一类银行家时，即便是通胀型政府，也能通过表现得像保守型政府并借此树立声誉而获益；{cite:t}`BackusDriffill1985` 和 {cite:t}`Barro1986` 将此建模为一个不完全信息博弈。

### 我们由此得出的结论

上述每一项补救措施改变的都是*制度*，而非理论本身。

在理论内部，多重性是不可消解的，萨金特精准地指出了它的来源：它源于赋予系统中*每一个人*的理性。

之所以会产生这一连续统，正是因为完美理性——一套足以支持某一均衡的复杂预期体系，也足以支持许多均衡。

因此本书从完美理性中撤退。

后续的讲座将完全理性的参与者替换为对经济理解有限的参与者。

首先是公众采用适应性预测，这正是接下来的 {doc}`phillips_adaptive`。

然后是在 {doc}`phillips_misspecified` 和 {doc}`phillips_self_confirming` 中，双方都用设定有误的模型去拟合由自身信念所生成的数据；在后者的自我实现均衡中，设定有误的一方正是政府。

最后，从 {doc}`phillips_learning` 开始，政府实时重新估计其模型。

这些模型更接近卢卡斯批评麦克拉肯报告时心目中所想的东西，而且与本讲座的理论不同，它们能够做出预测。

## 练习

```{exercise-start}
:label: cpol_ex1
```

阈值 {eq}`cp_cutoff` 声称：当且仅当 $\delta \geq 1/(2 + \theta^2)$ 时，回归纳什能永远维持拉姆齐结果——这一阈值只依赖于 $\theta$ 本身，而不依赖于自然率 $U^*$ 或上界 $y^\#$。

请通过数值方法验证这两个说法。

对于一系列 $\theta$ 值，在细网格上找出能使回归纳什支持拉姆齐结果的最小 $\delta$，并与 $1/(2+\theta^2)$ 进行比较。

然后检验改变 $U^*$ 是否会影响这一结果。

```{exercise-end}
```

```{solution-start} cpol_ex1
:class: dropdown
```

```{code-cell} ipython3
δ_grid = np.linspace(0.01, 0.99, 9801)

rows = []
for θ in (0.5, 1.0, 1.25, 2.0, 3.0):
    for U_star in (5.5, 11.0):
        md = Model(θ=θ, U_star=U_star)
        ok = [δ for δ in δ_grid
              if supports_ramsey_by_nash(md._replace(δ=δ))]
        rows.append((θ, U_star, min(ok), 1 / (2 + θ**2)))

pd.DataFrame(rows, columns=["$\\theta$", "$U^*$", "smallest $\\delta$ found",
                            "$1/(2+\\theta^2)$"]).round(4)
```

网格搜索的结果与闭合形式的匹配误差不超过网格间距，且 $U^*$ 加倍不改变任何结果。

$U^*$ 和 $y^\#$ 都被消去的原因，在 {eq}`cp_schedules` 中是可见的：$y = 0$ 处的单期诱惑以及纳什-拉姆齐值差距都与 $U^{*2}$ 成正比，因此尺度从不等式中消去，而 $y^\#$ 从不出现，因为不等式两侧都不涉及上界。

意外通货膨胀导致的失业反应 $\theta$ 越大，纳什结果相对于拉姆齐结果就越差，这会强化威胁，使得一个*更缺乏耐心*的政府也能被维持在零通货膨胀水平。

（在 {doc}`phillips_credibility` 的 $(U, y)$ 平面中，菲利普斯曲线的斜率为 $-1/\theta$，因此 $\theta$ 越大意味着曲线越*平坦*。）

```{solution-end}
```

```{exercise-start}
:label: cpol_ex2
```

随着政府变得缺乏耐心，均衡值的集合会收缩。

使用 `solve_aps` 计算贴现因子在 $0.001$ 到 $0.98$ 之间的均衡值集合 $V = [\underline v, \overline v]$，并将两个端点相对 $\delta$ 在对数坐标轴上作图，同时标出纳什值和拉姆齐值。

1. 在怎样的贴现因子 $\delta_R$ 以下，最优均衡值不再是拉姆齐结果？
   通过询问惩罚 $r(y^\#, B(y^\#))$ 何时使政府更偏好拉姆齐结果而非背离所得的单期收益，精确地求出 $\delta_R$，并用 `solve_aps` 加以确认。
1. 在较小的 $\delta$ 处，$V$ 的下边界会偏离 $r(y^\#, B(y^\#))$。
   请找出这一偏离发生的位置，并借助最差值一节中的检验加以解释。
1. 当 $\delta \to 0$ 时，$V$ 会发生什么变化？为什么这正是你应该预料到的答案？

```{exercise-end}
```

```{solution-start} cpol_ex2
:class: dropdown
```

令 $G = r(0, B(0)) - v^R$ 为在拉姆齐结果处背离所获得的单期收益。

以 $\underline v$ 作为永久惩罚，拉姆齐结果能被维持的条件是
$v^R \geq (1-\delta)(v^R + G) + \delta \underline v$，即当
$\delta \geq \delta_R = G / (G + v^R - \underline v)$ 时成立。

对于这一偏离，我们通过二分法求出贴现因子 $\delta_w$，在该处候选值 $r(y^\#, B(y^\#))$ 所需要的报酬 {eq}`cp_reward` 恰好等于它所隐含的最优值。

```{code-cell} ipython3
v_R, v_N = r_keep(m, 0.0), r_keep(m, y_nash(m))
v_cand = r_cheat(m, m.y_max)
G = r_cheat(m, 0.0) - v_R                  # one-period gain from deviating at Ramsey
δ_R = G / (G + v_R - v_cand)


def check_gap(δ):
    "Reward required by the candidate worst value minus the best value it implies."
    md = m._replace(δ=δ)
    return next_value(md, v_cand) - best_value(md, v_cand)[0]


lo, hi = 0.01, 0.1                     # the check fails at lo and passes at hi
for _ in range(50):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if check_gap(mid) > 0 else (lo, mid)
δ_w = hi

print(f"δ_R = {δ_R:.4f};  δ_w = {δ_w:.4f};  δ* = {1 / (2 + m.θ**2):.4f}")
for δ in (δ_R - 0.001, δ_R + 0.001):
    Wδ, _ = solve_aps(m._replace(δ=δ))
    print(f"  δ = {δ:.4f}: best value {Wδ[1]:.4f}, equals Ramsey? "
          f"{np.isclose(Wδ[1], v_R, atol=1e-6)}")
for δ in (0.001, 0.01, 0.02):
    Wδ, _ = solve_aps(m._replace(δ=δ))
    print(f"  δ = {δ:.3f}: V = [{Wδ[0]:.3f}, {Wδ[1]:.3f}], "
          f"width / δ = {(Wδ[1] - Wδ[0]) / δ:.0f}")
```

下图标出了这三个贴现因子。

```{code-cell} ipython3
δs = np.geomspace(0.001, 0.98, 120)
lows, highs = [], []
for δ in δs:
    Wδ, _ = solve_aps(m._replace(δ=δ))
    lows.append(Wδ[0])
    highs.append(Wδ[1])

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.fill_between(δs, lows, highs, alpha=0.2, color='C0', label='set of SPE values $V$')
ax.plot(δs, highs, 'C0', lw=1.4)
ax.plot(δs, lows, 'C0', lw=1.4)
ax.axhline(v_R, color='C2', ls=':', lw=1.2, label='Ramsey $v^R$')
ax.axhline(v_N, color='k', ls='--', lw=1, label='Nash $v^N$')
for δ_mark, name in [(δ_w, r'$\delta_w$'), (δ_R, r'$\delta_R$'),
                     (1 / (2 + m.θ**2), r'$\delta^\star$')]:
    ax.axvline(δ_mark, color='gray', lw=0.8)
    ax.annotate(name, (δ_mark, -60), textcoords='offset points', xytext=(3, 0))
ax.set_xscale('log')
ax.set_xlabel(r'discount factor $\delta$')
ax.set_ylabel('value')
ax.legend(loc='center right')
plt.show()
```

最优均衡值在 $\delta \geq \delta_R \approx 0.161$ 时等于拉姆齐结果，远*低于* {eq}`cp_cutoff` 中的 $\delta^\star \approx 0.28$。

阈值 $\delta^\star$ 使用回归*纳什*作为威胁，而 APS 集合使用最差均衡，这是一种严厉得多的威胁，因此拉姆齐结果在比巴罗和戈登的论证单独所暗示的更低的贴现因子下依然存活。

当 $\delta < \delta_w \approx 0.044$ 时，最差值一节中的检验会失效。

使单期在 $y^\#$ 处的惩罚可被接受所需要的报酬 {eq}`cp_reward` 大致按 $1/\delta$ 增长，而在 $\delta_w$ 以下它超过了最优均衡值。

此时最差均衡必须从更低的通货膨胀率开始，$V$ 的下边界随之上升，偏离 $r(y^\#, B(y^\#))$，朝向 $v^N$ 靠拢。

当 $\delta \to 0$ 时，该集合会坍缩至单点 $v^N$，其宽度大致与 $\delta$ 成正比。

这正是它必然要发生的：一个完全没有耐心的政府只关心当期，对未来的承诺没有任何分量，唯一能被维持的结果就是政府会短视地选择的那个结果——纳什结果。

这些集合是嵌套的，并随 $\delta$ 的增大而扩大；{cite:t}`APS1990` 证明，在他们的设定中，平均均衡值的集合随贴现因子单调变化。

```{solution-end}
```

```{exercise-start}
:label: cpol_ex3
```

多重性一节中的方法3是三者中最为大胆的：它以 $y^\#$ 通货膨胀一期，然后永远冻结在一个固定的通货膨胀率上。

因为它立即稳定下来，其激励约束是三者中最紧的，因此值得核实一下，而不是想当然地认为它成立。

请直接验证方法3是一个子博弈完美均衡：确认其第一期的承诺得到兑现，其冻结的延续值落在 $V$ 内，并确认政府在*两个*阶段都更偏好遵从而非背离，使用 $\underline v$ 作为惩罚。

```{exercise-end}
```

```{solution-start} cpol_ex3
:class: dropdown
```

```{code-cell} ipython3
v_2 = next_value(m, v_lo)
y_tilde = action_delivering(m, v_2)

# phase 1: promised v_lo, prescribed action y^#, continuation v_2 on adherence
lhs_1 = (1 - m.δ) * r_keep(m, m.y_max) + m.δ * v_2
rhs_1 = (1 - m.δ) * r_cheat(m, m.y_max) + m.δ * v_lo

# phase 2: promised v_2, prescribed action ỹ, continuation v_2 on adherence
lhs_2 = (1 - m.δ) * r_keep(m, y_tilde) + m.δ * v_2
rhs_2 = (1 - m.δ) * r_cheat(m, y_tilde) + m.δ * v_lo

print(f"v_2 = {v_2:.4f},  ỹ = {y_tilde:.4f}")
print(f"v_2 lies inside V = [{v_lo:.4f}, {v_hi:.4f}]: "
      f"{v_lo <= v_2 <= v_hi}")
print(f"phase 1 delivers the promise: {np.isclose(lhs_1, v_lo)}")
print(f"phase 1 incentive: {lhs_1:.4f} vs {rhs_1:.4f}  ->  "
      f"holds with equality? {np.isclose(lhs_1, rhs_1)}")
print(f"phase 2 delivers the promise: {np.isclose(lhs_2, v_2)}")
print(f"phase 2 incentive: {lhs_2:.4f} >= {rhs_2:.4f}  ->  {lhs_2 >= rhs_2}")
print(f"slack in phase 2: {lhs_2 - rhs_2:.6f}")
```

所有条件都成立，且第一期的承诺被精确地兑现。

第一阶段的约束恰好取等号，而这正是 $\underline v$ 的定义：在 $y^\#$ 处，政府总是可以背离至 $B(y^\#)$ 并接受惩罚 $\underline v$，其价值为
$(1-\delta) r(y^\#, B(y^\#)) + \delta \underline v$，而
$\underline v = r(y^\#, B(y^\#))$ 恰好承诺了这么多。

第二阶段的余量非常小，这同样并非偶然。

冻结值 $v_2$ 只是勉强高于 $\underline v$，因此背离的惩罚只比均衡本身稍差一点——这恰恰意味着它接近均衡值集合的底部。

```{solution-end}
```

```{exercise-start}
:label: cpol_ex4
```

本练习比较将政府永远维持在拉姆齐结果的三种方式。

1. 假设任何偏离拉姆齐结果的行为都会招致不依赖于 $\delta$ 的延续值 $v_p$。
   证明拉姆齐结果能被维持当且仅当 $\delta \geq G/(G + v^R - v_p)$，其中
   $G = r(0, B(0)) - v^R$。
1. 对纳什回归（$v_p = v^N$）以及最差均衡（$v_p = \underline v = r(y^\#, B(y^\#))$，在练习 {ref}`cpol_ex2` 中 $\delta_w$ 以上的每一个 $\delta$ 下都是最差值）这两种情形求出这一阈值。
1. 阿布鲁的胡萝卜加大棒策略以 $\tilde v$ 作为惩罚，它依赖于 $\delta$；其阈值即为 {ref}`cpol_abreu` 中的 $\delta_A$。
   比较 $\delta_A$ 处的 $\tilde v$ 与 $\underline v$，并解释两者的差异。
1. 对这三个阈值进行排序，并用 `is_abreu_spe` 核实 $\delta_A$。

```{exercise-end}
```

```{solution-start} cpol_ex4
:class: dropdown
```

**1.** 背离拉姆齐结果在当期获得 $r(0, B(0)) = v^R + G$，此后获得 $v_p$，因此约束为
$v^R \geq (1-\delta)(v^R + G) + \delta v_p$，整理可得
$\delta (G + v^R - v_p) \geq G$。

**2.** 根据 {eq}`cp_gap`，$G = \theta^2 U^{*2} / \bigl(2(1+\theta^2)\bigr)$，且
$v^R - v^N = \theta^2 U^{*2}/2$，因此纳什回归给出 $1/(2+\theta^2)$，即阈值 {eq}`cp_cutoff`。

最差均衡给出练习 {ref}`cpol_ex2` 中的 $\delta_R$。

```{code-cell} ipython3
def ramsey_threshold(m, v_p):
    "Smallest δ at which the permanent punishment v_p sustains Ramsey."
    G = r_cheat(m, 0.0) - r_keep(m, 0.0)
    return G / (G + r_keep(m, 0.0) - v_p)


δ_nash = ramsey_threshold(m, r_keep(m, y_nash(m)))
δ_opt = ramsey_threshold(m, r_cheat(m, m.y_max))
δ_A = m.θ**2 * m.U_star**2 / ((1 + m.θ**2) * m.y_max**2)
print(f"Nash reversion      δ* = {δ_nash:.4f}")
print(f"Abreu stick/carrot  δ_A = {δ_A:.4f}")
print(f"worst equilibrium   δ_R = {δ_opt:.4f}\n")

m_A = m._replace(δ=δ_A)
print(f"at δ_A: Abreu's punishment {abreu_value(m_A):.4f},  worst value "
      f"{r_cheat(m, m.y_max):.4f},  reward after the stick in the worst equilibrium "
      f"{next_value(m_A, r_cheat(m, m.y_max)):.4f}\n")
for δ in (δ_A - 1e-4, δ_A + 1e-4):
    print(f"δ = {δ:.5f}: stick and carrot is an SPE? {is_abreu_spe(m._replace(δ=δ))[2]}")
```

**3.** 两种惩罚都以一期 $y^\#$ 开始。

阿布鲁策略随后支付全部的胡萝卜，即永远的拉姆齐结果。

最差均衡在大棒之后支付的报酬较小，即 {eq}`cp_reward` 中的 $v_1$，它本身比纳什结果还要差，因此其惩罚更为严厉。

**4.** 排序为 $\delta^\star > \delta_A > \delta_R$。

惩罚越严厉，威慑单期收益 $G$ 所需要的贴现因子就越低。

纳什回归是三者中最温和的惩罚，最差均衡是最严厉的，而阿布鲁简单的胡萝卜加大棒策略介于两者之间。

```{solution-end}
```

```{exercise-start}
:label: cpol_ex5
```

对于 $\delta < \delta^\star$，回归纳什无法维持拉姆齐结果。

它所能维持的最优值是 {cite:t}`Sargent1999` 第4章方程(25)：

$$
v^{nr} = r(y^{nr}, y^{nr}) = (1-\delta)\, r\bigl(y^{nr}, B(y^{nr})\bigr) + \delta\, v^N ,
$$

其中 $y^{nr}$ 是使遵从至少与背离并永远回归纳什一样好的最低通货膨胀率。

1. 利用 {eq}`cp_gap`，证明对 $\delta \leq \delta^\star$ 有

   $$
   y^{nr} = \theta U^* \, \frac{1 - \delta(2 + \theta^2)}{1 + \delta \theta^2} .
   $$

1. 将 $v^{nr}$ 相对 $\delta \in (0, \delta^\star]$ 作图，并一同画出来自 `solve_aps` 的最优 SPE 值 $\overline v$，比较两者在 $\delta = 0.2$ 处的取值。

```{exercise-end}
```

```{solution-start} cpol_ex5
:class: dropdown
```

**1.** 约束 $r(y, y) \geq (1-\delta) r(y, B(y)) + \delta v^N$ 等价于
$\delta \left[r(y, y) - v^N\right] \geq (1-\delta)\left[r(y, B(y)) - r(y, y)\right]$。

令 $z = y^N - y \geq 0$ 为低于纳什结果的距离。

则 $r(y, y) - v^N = z (2 y^N - z)/2$，而根据 {eq}`cp_gap`，右侧为
$(1-\delta) z^2 / \bigl(2(1+\theta^2)\bigr)$。

除以 $z > 0$，约束化为 $z \leq 2 y^N \delta (1+\theta^2)/(1 + \delta\theta^2)$。

最低可接受的通货膨胀率是 $y^N$ 减去该上界，由此得到该公式。

当 $\delta = 1/(2+\theta^2) = \delta^\star$ 时，该值恰好降为零。

在 $v^N$ 和 $v^{nr}$ 之间的每一个值也都能被回归纳什维持，但高于 $v^{nr}$ 的值则不能。

```{code-cell} ipython3
def nash_reversion_best(m):
    "The best value sustainable by reversion to Nash (book eq. (25)) and its action."
    y_nr = max(y_nash(m) * (1 - m.δ * (2 + m.θ**2)) / (1 + m.δ * m.θ**2), 0.0)
    return r_keep(m, y_nr), y_nr


δ_star = 1 / (2 + m.θ**2)
δs_nr = np.linspace(0.005, δ_star, 60)
v_nr = [nash_reversion_best(m._replace(δ=δ))[0] for δ in δs_nr]
v_best = [solve_aps(m._replace(δ=δ))[0][1] for δ in δs_nr]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(δs_nr, v_best, 'C0', lw=1.6, label=r'best SPE value $\overline{v}$')
ax.plot(δs_nr, v_nr, 'C1', lw=1.6, label=r'best value with Nash reversion $v^{nr}$')
ax.axhline(r_keep(m, 0.0), color='C2', ls=':', lw=1.2, label='Ramsey $v^R$')
ax.axhline(r_keep(m, y_nash(m)), color='k', ls='--', lw=1, label='Nash $v^N$')
ax.set_xlabel(r'discount factor $\delta$')
ax.set_ylabel('value')
ax.legend()
plt.show()

m_02 = m._replace(δ=0.2)
v_nr_02, y_nr_02 = nash_reversion_best(m_02)
print(f"δ = 0.2: Nash reversion sustains at best {v_nr_02:.4f}, with y = {y_nr_02:.4f};"
      f"  best SPE value {solve_aps(m_02)[0][1]:.4f}")
```

在 $\delta = 0.2$ 时，回归纳什至多能将通货膨胀压低至约1.5个百分点，而最优惩罚则能维持拉姆齐结果本身。

两条曲线在 $\delta^\star$ 处相交，此处回归纳什刚好开始能够维持拉姆齐结果。

当 $\delta \to 0$ 时，两条曲线都回落至 $v^N$，因为最差均衡值本身也逼近 $v^N$（练习 {ref}`cpol_ex2`）。

两者的差距只在水平上收窄：从 $v^N$ 算起，最优惩罚所维持的改善幅度仍约为回归纳什的两倍。

```{code-cell} ipython3
v_N = r_keep(m, y_nash(m))
for δ in [0.02, 0.005]:
    m_δ = m._replace(δ=δ)
    ratio = (solve_aps(m_δ)[0][1] - v_N) / (nash_reversion_best(m_δ)[0] - v_N)
    print(f"δ = {δ}: (v̄ - v^N) / (v^nr - v^N) = {ratio:.2f}")
```

```{solution-end}
```

```{exercise-start}
:label: cpol_ex6
```

{ref}`cpol_aps` 中的算法定理要求初始集合满足
$V \subseteq \mathcal{T}(W_0) \subseteq W_0$。

1. 证明 $W_0 = \{v^N\}$ 满足 $\mathcal{T}(W_0) = W_0$。
   用 `T_operator` 和 `solve_aps` 加以确认，并指出定理的哪一条假设不成立。
1. 改为从 $W_0 = \{v^R\}$ 出发。
   这一迭代会收敛到哪里？为什么？
1. 这两个实验说明了延续值集合要维持纳什结果以外的任何结果，需要满足什么条件？

```{exercise-end}
```

```{solution-start} cpol_ex6
:class: dropdown
```

**1.** 当延续值只有单一取值 $w$ 时，奖励和惩罚重合，$w_1 = w_2 = w$，因此可行性要求
$(1-\delta) r(y, y) \geq (1-\delta) r(y, B(y))$。

根据 {eq}`cp_gap`，只有 $y = y^N$ 满足条件，并生成唯一的值
$(1-\delta) v^N + \delta w$。

因此 $\mathcal{T}(\{w\}) = \{(1-\delta) v^N + \delta w\}$，且 $\mathcal{T}(\{v^N\}) = \{v^N\}$。

不成立的假设是 $V \subseteq \mathcal{T}(W_0)$。

纳什速率 $y^N = 6.875$ 恰好是 `T_operator` 网格上的一点，因此数值算子能够找到它。

```{code-cell} ipython3
v_N, v_R = r_keep(m, y_nash(m)), r_keep(m, 0.0)
print(f"v^N = {v_N:.4f};  T({{v^N}}) = {T_operator(m, (v_N, v_N))}")
W_N, it_N = solve_aps(m, W0=(v_N, v_N))
print(f"iterating from {{v^N}}: stops after {it_N} iterations at {W_N}")
W_R, it_R = solve_aps(m, W0=(v_R, v_R))
print(f"iterating from {{v^R}}: stops after {it_R} iterations at "
      f"[{W_R[0]:.4f}, {W_R[1]:.4f}]")
```

**2.** 从 $\{v^R\}$ 出发，迭代值是单点序列
$(1-\delta^j)\, v^N + \delta^j v^R$，它收敛到 $\{v^N\}$，而不是 $V$。

这一起点不满足另一条假设，因为 $\mathcal{T}(\{v^R\}) \not\subseteq \{v^R\}$。

**3.** 激励来自遵从的奖励与背离的惩罚之间的*差距*。

单一的延续值不提供任何差距，因此它所能维持的唯一结果就是政府在没有激励时会选择的结果，即静态最优反应。

初始集合必须足够宽，以包含均衡所使用的每一种奖励和每一种惩罚，这正是 $V \subseteq \mathcal{T}(W_0)$ 所保证的，也正是全部可行值集合 $[r(y^\#, y^\#), 0]$ 所提供的。

```{solution-end}
```
