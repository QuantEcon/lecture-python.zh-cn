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
  title: 均衡搜寻与失业
  headings:
    Overview: 概述
    The economy: 该经济
    The economy::Timing: 时序
    The economy::The value of search: 搜寻的价值
    A worker's problem: 工人的问题
    Employment, wages, and two thresholds: 就业、工资与两个阈值
    Employment, wages, and two thresholds::How many work: 有多少人工作
    Employment, wages, and two thresholds::How many arrive: 有多少人到达
    Employment, wages, and two thresholds::Three regions, one transition law: 三个区域，一条转移规律
    A computable version: 一个可计算版本
    Economy-wide equilibrium: 全经济均衡
    Learning the value of search: 学习搜寻的价值
    Inside the equilibrium: 均衡内部
    Inside the equilibrium::What a market looks like: 一个市场的样子
    Inside the equilibrium::The workforce distribution and a sample path: 工作队伍的分布与一条样本路径
    Inside the equilibrium::Where markets sit: 市场所处的位置
    Inside the equilibrium::The wage distribution: 工资分布
    Efficiency: 效率
    Efficiency::Surplus: 剩余
    Efficiency::A planner for one market: 单个市场的规划者
    Efficiency::What the theorem says, and what it does not: 该定理说了什么，又没有说什么
    Relation to other lectures: 与其他讲座的关系
    Relation to other lectures::What the island model became: 岛屿模型的后续发展
    Relation to other lectures::Returns to labor mobility: 劳动力流动的回报
    Exercises: 练习
---

(lucas_prescott_search)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 均衡搜寻与失业

```{contents} Contents
:depth: 2
```

除了 Anaconda 中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

本讲座研究 {cite:t}`LucasPrescott1974`，这是与我们在 {doc}`lucas_prescott_investment` 中研究的投资论文相对应的劳动力市场论文。

这两篇论文由同一批作者在同一地点、相隔仅几年写成，并且使用了相同的工具：通过逐次近似求解的贝尔曼方程、布莱克韦尔定理，以及一个不变分布描述长期状态的马尔可夫过程。

变化的是问题本身。

卢卡斯和普雷斯科特关注的目标是 {cite:t}`friedman1968role` 和 {cite:t}`Phelps1967` 所提出的自然失业率。

用他们的话说，讨论自然失业率时所用的语言"似乎暗示它可能具有竞争均衡的性质"，然而

> *并不存在任何理论模型，在其中一个非零的均衡失业率被确定下来并对其性质加以研究。*

没有这样的模型，就没有一个框架可以用来提出失业所引发的福利问题，也无法解释为什么各国的平均失业率差异如此之大。

20世纪60年代的搜寻文献无法提供这样的模型。

{doc}`mccall_model` 描述了那一文献的核心问题：一个工人从已知的分布中抽取工资报价，并决定何时停止搜寻。

这是一个个体模型，它将工资分布视为给定的。

卢卡斯和普雷斯科特认为，他们所关心的问题无法以这种方式回答：

> *最优劳动供给行为与其所依据的工资分布，必须在一个关于市场（而非个体）行为的模型中同时被确定。*

他们构建了一个由许多空间上分离的市场组成的经济，每个市场都有其自身随机变动的劳动需求。

这一图景来自埃德蒙·费尔普斯（Edmund Phelps）的"岛屿经济"，他们称之为"本模型的直接前身" {cite}`PhelpsEtAl1970`。

它不同于 {cite:t}`LucasNeutrality1972` 的岛屿模型，在那个模型中就业随总需求波动，但失业作为一种活动并未与闲暇区分开来。

在这里，二者是有区别的。

一个市场中的工人可以在出清该市场的工资下工作，或者离开。

离开意味着本期一无所获，下期到达其他地方，因此*失业就是在市场之间转移的行为*。

一切都是竞争性的，每个人都在优化，并且预期在 {cite:t}`muth1961` 以及投资论文 {cite:t}`Lucas_Prescott_1971` 的意义上是理性的：工人用来评估自身前景的分布，正是其自身决策所产生的分布。

该论文的核心对象是一个单一的数值 $\lambda$，即搜寻的预期现值。

单个市场将 $\lambda$ 视为给定的，正如一个竞争性企业将价格视为给定的一样。

在整个经济范围内，$\lambda$ 不断调整，直到各市场希望保留的劳动力总量等于经济实际拥有的劳动力。

在论文的均衡中，总有一些市场的工人数量超过其想要雇用的数量，因此失业在每一期都是正的；{ref}`lps_regions` 解释了我们所使用的可计算版本何时保持这一性质。

卢卡斯和普雷斯科特还论证说，这种失业未必意味着一种低效率。

```{note}
在该模型中不存在拥堵外部性，也没有匹配函数、没有讨价还价、也没有工资刚性：失业之所以存在，是因为转移需要一个时期，而在均衡中总有人正在转移之中。

正在搜寻的工人对其同伴造成的拥挤，与任何商品的卖方对其他卖方造成的拥挤方式相同，而 $\lambda$，正如任何其他均衡价格一样，传递出论文所称的增加一名搜寻者所带来的"正确的社会回报"。

该论文没有证明任何福利定理，也未解决实际劳动力市场是否存在重要外部效应这一问题。

{ref}`lps_efficiency` 表明，在该模型内部，这一主张可以被精确地表述：当转移需要一个时期时，平稳均衡最大化了从其自身工人分布中所能获得的剩余的预期现值，而 $\lambda$ 正是增加一名工人到搜寻者池中所带来的现值。
```

本讲座首先阐述理论：工人的问题、描述一个市场的两个阈值，以及市场劳动力的运动规律。接着计算一个例子，通过求出均衡的 $\lambda$ 来封闭该经济，并探讨工人如何能够了解这一数值。

在考察了均衡内部的市场、劳动力和工资之后，本讲座证明了效率的主张，并将该模型与其后续发展联系起来。练习部分涉及论文中的比较静态分析、一种转移成本、论文自身的两状态例子、非定向搜寻、菲利普斯曲线，以及一项效率的数值检验。

让我们先引入一些库：

```{code-cell} ipython3
import warnings
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from collections import namedtuple
from scipy.optimize import brentq
```

## 该经济

存在许多市场，每个市场都有其自身的需求状态。

需求冲击*在各市场间是独立的*，因此总需求保持不变，而*在一个市场内部*则是持续的，因此今天的需求能告诉工人一些关于明天的信息。

在每个市场中，一个竞争性产业以劳动作为唯一投入生产单一商品。

如果 $p = D(s, Q)$ 是需求状态 $s$ 下的逆需求曲线，$q(n)$ 是 $m$ 个相同企业各自的生产函数，那么利润最大化会使工资等于劳动的边际价值产量，函数

```{math}
:label: lps_R
R(s, n) = D\big(s, m\,q(n)\big)\, q'(n)
```

概括了关于这个劳动力市场需求方的一切信息。

卢卡斯和普雷斯科特舍弃了 $D$ 和 $q$，而直接对 $R$ 做出假设：它为正、有界、可微，且满足

$$
R_s(s,n) > 0, \qquad R_n(s,n) < 0 .
$$

需求状态 $s$ 服从一个具有转移密度 $f(s', s)$ 的马尔可夫过程，该过程拥有唯一的平稳分布。

其条件分布函数 $F(s', s)$ 随 $s$ 递减，因此今天较高的需求会使明天出现较高需求的可能性增大。

### 时序

一个市场在一期开始时拥有一个**工作队伍**（workforce）$y$，这是它所能雇用的最大人数。

然后，在该期内：

1. 需求 $s$ 被实现，劳动力市场出清，以工资 $w = R(s, n)$ 雇用 $n \le y$ 名工人
1. 未被雇用的 $y - n$ 名工人一无所获，加入搜寻者的队伍
1. 被雇用的工人留在该市场进入下一期；搜寻者被分配到各个市场，下一期到达

因此，每当一个工人移动时，他恰好失业*一期*，失业者就是处于转移之中的工人。

### 搜寻的价值

由于需求冲击在各市场间是独立的，且市场数量众多，因此一旦经济稳定下来，整个经济中工人在各市场间的分布就保持不变，尽管每个市场的工作队伍都在波动。

因此，一个搜寻者的前景不取决于他正在离开的那个市场，搜寻的预期现值是一个单一的数值 $\lambda$。

搜寻者被分配到各个市场的方式*使到达的预期价值相等*，因此没有任何市场会给予到达的工人超过 $\lambda$ 的价值。

目前我们将 $\lambda$ 视为给定，并在 {ref}`lps_economy_wide` 中确定它。

## 工人的问题

设 $v(s, y, \lambda)$ 为一名工人在一期开始时处于需求为 $s$、工作队伍为 $y$ 的市场中时，其收入的预期现值，并设 $\beta \in (0,1)$ 为贴现因子。

卢卡斯和普雷斯科特通过考察三种情形得出了关于 $v$ 的泛函方程。

**情形 A：部分工人离开。** 离开的工人获得 $\lambda$。留下的工人所得既不能更多——因为离开者本可以留下，也不能更少——因为他们本可以离开。所以 $v = \lambda$。

**情形 B1：所有工人留下，没有新工人到达。** 当前就业人数为 $y$，工资为 $R(s,y)$，下一期的工作队伍仍为 $y$：

$$
v(s,y,\lambda) = R(s,y) + \beta \int v(s', y, \lambda) f(s', s)\,ds' .
$$

**情形 B2：所有工人留下，有新工人到达。** 到达的工人是搜寻者，因此下期处于该市场的贴现预期价值 $\beta \int v f\, ds'$ 必须恰好等于 $\lambda$，即促使他们搜寻而来的那个搜寻价值：

$$
v(s,y,\lambda) = R(s,y) + \lambda .
$$

当工作队伍保持不变时的贴现延续值低于 $\lambda$ 时，没有人到达，情形 B1 适用；当它本会超过 $\lambda$ 时，新到达者会将其拉低至 $\lambda$，情形 B2 适用。

无论哪种情况，延续值都是 $\lambda$ 与 $\beta \int v(s', y, \lambda) f(s', s)\, ds'$ 中的较小者，即下面的 $\min$。

当即使如此也会使一名工人所得低于 $\lambda$ 时，情形 A 适用，他会选择 $\lambda$，即 $\max$。

这三种情形合并为一个方程：

```{math}
:label: lps_bellman
v(s, y, \lambda) = \max\left\{\lambda,\;
R(s,y) + \min\left[\lambda,\;
\beta \int v(s', y, \lambda) f(s', s)\, ds' \right]\right\} .
```

这两个算子承担不同的职能。*max* 是工人自身的选择权：他总是可以离开并获得 $\lambda$。

而 *min* 根本不是工人的选择，它是竞争在搜寻者之间施加的一种限制：如果该市场的延续值超过了 $\lambda$，搜寻者就会不断涌入，直到它不再超过为止。

```{prf:proposition}
:label: lps_prop_v

对每一个 $\lambda$，方程 {eq}`lps_bellman` 都有唯一的有界连续解 $v$，并且从任意起点出发的逐次近似都收敛于它。

该解在 $s$ 和 $\lambda$ 上非递减，在 $y$ 上非递增。
```

{prf:ref}`lps_prop_v` 的证明与 {doc}`lucas_prescott_investment` 中所使用的证明相同：由 {eq}`lps_bellman` 右边所定义的算子是单调的，并满足一个贴现条件，因此布莱克韦尔定理 {cite}`Blackwell1965` 适用。

在 $s$ 上的单调性正是使用了 $F(s', s)$ 随 $s$ 递减这一假设之处。一个工人所在的市场越拥挤，他的处境就越差，因为更大的工作队伍意味着更低的边际产量和更低的工资，这就是为什么 $v$ 在 $y$ 上非递增。

## 就业、工资与两个阈值

有了 $v$ 之后，市场行为的其余部分就取决于两条阈值规则。

### 有多少人工作

设 $\bar n(s, \lambda)$ 为这样一种就业水平：在此水平下，边际工人恰好在留下和搜寻之间感到无差异：

```{math}
:label: lps_nbar
R(s, \bar n) + \beta \int v(s', \bar n, \lambda) f(s', s)\, ds' = \lambda .
```

左边是当就业和下一期的工作队伍都为 $\bar n$ 时，留下的工人所获得的：本期工资加上贴现的延续值。它在 $\bar n$ 上非递增。

如果边际收入在某个工作队伍水平上降为零——下面的可计算版本正是这样——那么该市场拥挤到这种程度时的价值已降至 $\lambda$，左边保持在 $\beta \lambda < \lambda$ 这一水平不变，{eq}`lps_nbar` 在那里不可能成立。

在整数的工作队伍网格上，{eq}`lps_nbar` 很少恰好相等，就业人数就是使左边至少等于 $\lambda$ 的最大 $\bar n$。

于是就业为

```{math}
:label: lps_employment
n(s, y, \lambda) = \min\{\bar n(s,\lambda),\; y\},
\qquad w = R(s, n) ,
```

该市场的失业为 $y - n$，恰好在工作队伍超过 $\bar n$ 时为正。

### 有多少人到达

一个市场只有在雇用了其全部工作队伍、且延续值仍超过 $\lambda$ 时，才会吸引搜寻者。

设 $\bar y(s,\lambda)$ 为使这一情形不再成立的工作队伍水平：

```{math}
:label: lps_ybar
\beta \int v(s', \bar y, \lambda) f(s', s)\, ds' = \lambda .
```

到达者将市场填充到 $\bar y(s, \lambda)$，不再更多。注意 $\bar y \le \bar n$，因为 $R \ge 0$ 意味着任何仅凭延续值就足以覆盖 $\lambda$ 的工作队伍，在加上工资之后也必定覆盖它。

在边际产量为正的任何地方，这个不等式都是严格的：一个刚好值得搜寻者进入的市场，严格处于每个人都工作的那个范围之内。

在下文校准中最差的需求状态下，两个阈值在均衡的 $\lambda$ 值处都为零，该市场既不雇用任何人也不吸引任何人。

(lps_regions)=
### 三个区域，一条转移规律

{eq}`lps_employment` 和 {eq}`lps_ybar` 共同将 $(s,y)$ 平面划分为论文图3中的三个区域：

* 区域 I，其中 $y > \bar n(s, \lambda)$，市场将工人释放到 $\bar n(s,\lambda)$；
* 区域 II，处于两个阈值之间，没有人移动；
* 区域 III，其中 $y < \bar y(s, \lambda)$，市场被填充到 $\bar y(s,\lambda)$。

它们赋予工作队伍的运动规律一个非常简单的形式：

```{math}
:label: lps_transition
y' = \min\big\{\max\{y,\ \bar y(s,\lambda)\},\ \bar n(s,\lambda)\big\} .
```

贝尔曼方程是在*不变*的工作队伍 $y$ 处计算延续值的，而转移规律则将工作队伍转移到别处。

这并不矛盾，因为 $\max$ 和 $\min$ 已经处理了工作队伍发生转移的两种情形。

在到达者涌入之处，$\min$ 将延续值替换为 $\lambda$，而一个填充到 $\bar y$ 的市场，对其中任何人而言至少值 $\lambda$：在连续的工作队伍下恰好为 $\lambda$，在下文所用的整数网格上可能略高一些。在工人离开之处，$\max$ 将整个表达式替换为 $\lambda$。

因此，只有在工作队伍确实保持不变的那个区域，才会用到在不变的 $y$ 处计算的延续值。

```{note}
论文让到达人数为随机的：一个缺乏工人的市场会获得 $a(s,y,\lambda)\,x$ 名工人，其中 $x$ 是均值为1的正随机变量，因此搜寻必须"平均上消除租金"，而不是精确地消除。

与论文附录中的数值例子一样，我们设 $x \equiv 1$，这就给出了 {eq}`lps_transition`。

在论文中，随机到达使平稳分布具有严格为正的密度（其命题3），因此总有某个市场在释放工人，失业在每一期都为正。

当 $x \equiv 1$、需求状态数有限且需求过程能在一期内从任何一个状态移动到任何另一个状态时，有三种情形。

* 如果 $\max_s \bar y(s) > \min_s \bar n(s)$，平稳分布是唯一的，失业为正：一个在 $\bar y$ 最大的状态下被填满的市场，会在需求降至 $\bar n$ 最小的状态时释放工人，而这两个时期会把一个市场从任何初始工作队伍带到 $\min_s \bar n(s)$。
* 如果 $\max_s \bar y(s) = \min_s \bar n(s)$，平稳分布同样是唯一的，每个市场都保持该工作队伍水平，失业为零。
* 如果 $\max_s \bar y(s) < \min_s \bar n(s)$，任何介于两者之间的工作队伍在每个需求状态下都保持不变，因此存在一族连续的平稳分布，失业为零。

{ref}`lps_ex_persistence` 涉及到第三类经济。
```

## 一个可计算版本

我们遵循 {cite:t}`LucasPrescott1974` 的附录：有限数量的需求状态、整数工作队伍，以及对 {eq}`lps_bellman` 的价值函数迭代。

对于需求过程，我们使用论文脚注7中的例子，其中 $\log s$ 遵循一个持续性介于0和1之间的 AR(1) 过程。

我们用 Rouwenhorst 方法将其离散化为七个状态。

与 {cite:t}`Tauchen1986` 的方法不同，即使持续性很高，它也能精确匹配 $\log s$ 的持续性和平稳方差；{ref}`lps_ex_persistence` 的解答展示了若不如此会出现什么问题。

对于 $R$，我们采用一个随就业水平线性下降、在容量 $N$ 处降为零的边际收入表：

$$
R(s, n) = s\,\big(1 - n/N\big)^{+} .
$$

在两个需求状态且 $N = 100$ 的情况下，这正是论文例1中的表，绘制在其图5左上面板中，并按两倍放大。

容量设定偏离了理论——理论假设 $R$ 处处为正——但这是论文自身例子所采用的做法。它将一个市场可能想要的最大工作队伍限制在 $N$ 以下，因此论文的命题5——它承诺对每个市场的每个工作队伍水平都存在均衡——在此处仅对低于该上限的工作队伍水平成立。

它还决定了当工人充裕时失业如何变化，我们将在后面看到这一点。

$R$ 下降得多陡，决定了需求变动时就业变动的幅度，从而决定了把一名工人从一个市场送到另一个市场能获得多少收益；{ref}`lps_ex_two_states` 将讨论这一点。

函数 `make_model` 可以由任意需求链和边际收入表构建一个市场，`create_model` 则用它来构建本讲座的经济。

```{code-cell} ipython3
Model = namedtuple('Model', ['P', 's', 'y', 'R', 'β'])

def linear_R(s, y, N=200):
    "Marginal revenue s (1 - n/N)^+, which reaches zero at the capacity N."
    return s[:, None]*np.maximum(1 - y[None, :]/N, 0.0)

def make_model(P, s, β=0.9, y_max=260, R=linear_R, **R_args):
    "A market with demand chain (P, s) and marginal revenue R on y = 0, ..., y_max."
    y = np.arange(y_max + 1)
    R_sy = R(s, y, **R_args)
    R_sy[:, 0] = 0.0                                # an empty market pays nothing
    return Model(P=P, s=s, y=y, R=R_sy, β=β)

def create_model(n_s=7, ρ=0.9, sd=0.26, β=0.9, N=200, y_max=260,
                 method='rouwenhorst'):
    """The island economy, with log demand an AR(1) discretized into n_s states.

    ρ and sd are the persistence and the stationary standard deviation of
    log demand, so that changing ρ alone does not change how variable
    demand is.  N is a market's capacity: marginal revenue reaches zero there.
    """
    σ = sd*np.sqrt(1 - ρ**2)
    if method == 'rouwenhorst':
        with warnings.catch_warnings():         # quantecon notes a change in its API
            warnings.filterwarnings('ignore', message='The API of rouwenhorst',
                                    category=UserWarning)
            mc = qe.markov.rouwenhorst(n_s, ρ, σ)
    else:
        mc = qe.markov.tauchen(n_s, ρ, σ)
    return make_model(mc.P, np.exp(mc.state_values), β, y_max, N=N)

model = create_model()
print("demand states:", model.s.round(3))
```

贝尔曼算子就是把 {eq}`lps_bellman` 中的积分替换为求和。

因为延续值是在不变的工作队伍处计算的，所以该方程对每个 $y$ 都是一个独立的问题，仅通过需求状态相互关联，这就是为什么求解它成本很低，也是为什么我们对空市场所采用的约定不会影响其他任何事情的原因。

我们需要某个搜寻价值才能求解它，这里我们使用 $\lambda = 7.1$，这是一个接近 {ref}`lps_economy_wide` 所计算出的均衡值。

参数 `c`——离开的成本，在 {ref}`lps_ex_cost` 之前均为零——将工人的外部选择从 $\lambda$ 变为 $\lambda - c$。

```{code-cell} ipython3
def worker_value(model, λ, c=0.0, tol=1e-11, max_iter=10_000):
    "Solve the worker's functional equation; a worker who leaves collects λ - c."
    P, R, β = model.P, model.R, model.β
    v = np.full(R.shape, λ - c)
    for i in range(max_iter):
        v_new = np.maximum(λ - c, R + np.minimum(λ, β*(P @ v)))
        v_new[:, 0] = λ - c                         # an empty market holds no workers
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, i
        v = v_new
    raise RuntimeError(f"no convergence in {max_iter} iterations at λ = {λ}")

v, iters = worker_value(model, 7.1)
print(f"converged in {iters} iterations")
```

这两个阈值分别是使相关价值仍覆盖 $\lambda$ 的最大就业和最大工作队伍。

```{code-cell} ipython3
def last_at_least(A, bar):
    "For each row of A, the largest index past 0 at which A ≥ bar, or 0 if none."
    return np.array([np.max(np.where(row >= bar)[0]) if (row[1:] >= bar).any() else 0
                     for row in A])

def thresholds(model, v, λ, c=0.0):
    "The employment threshold n̄(s) and the arrival threshold ȳ(s)."
    EV = model.β*(model.P @ v)                  # the continuation at an unchanged workforce
    n_bar = last_at_least(model.R + EV, λ - c)  # staying on as the marginal worker
    y_bar = last_at_least(EV, λ)                # ≤ n_bar, because R ≥ 0 and c ≥ 0
    return n_bar, y_bar

n_bar, y_bar = thresholds(model, v, 7.1)
print("n̄(s) =", n_bar)
print("ȳ(s) =", y_bar)
```

就业受 $\bar n$ 限制，到达者填充到 $\bar y$，而 {eq}`lps_transition` 则移动工作队伍。

这个在 $(s, y)$ 上的马尔可夫链的不变分布可以被精确地计算出来。

{eq}`lps_transition` 的一期要么使工作队伍保持原状，要么把它带到某个阈值上，因此从某个阈值出发，该链永远不会离开阈值的集合，于是我们可以求解一个由 $(s, t)$ 这种小的对所构成的链，其中 $t$ 取遍至多 $2 \times 7$ 个阈值。

一个不在阈值上的工作队伍，只有在所有需求状态下都保持不变时，才能在长期内携带概率。

这样一个工作队伍区间的两端本身就是保持不变的阈值，因此阈值上的链就会有不止一个不变分布：这正是 {ref}`lps_regions` 中的注记所述的第三种情形。

因此我们让 `qe.MarkovChain` 求出这个小链的每一个不变分布，并在出现多于一个时报错停止，而不是任意挑选一个。

该函数在某个阈值触及工作队伍网格顶端时也会停止，否则它会被悄无声息地截断。

```{code-cell} ipython3
class NotBracketed(ValueError):
    "A bisection was given a bracket that does not contain a switch."

class NotUnique(ValueError):
    "The invariant distribution of a market is not unique."

class GridTooSmall(ValueError):
    "A threshold has reached the top of the workforce grid."

def stationary_distribution(model, n_bar, y_bar):
    "The invariant distribution of the chain on (s, y), computed exactly."
    P, n_s = model.P, len(model.s)
    if n_bar.max() == model.y[-1]:
        raise GridTooSmall(f"n̄ reaches the top of the grid, y = {model.y[-1]}")
    T = np.unique(np.concatenate([n_bar, y_bar]))     # the threshold values
    k = len(T)
    Q = np.zeros((n_s*k, n_s*k))                       # the chain on (s, t), t in T
    for i in range(n_s):
        t_next = np.searchsorted(T, np.clip(T, y_bar[i], n_bar[i]))
        for j in range(k):
            Q[i*k + j, t_next[j]::k] = P[i]            # to (s', t_next[j]) for every s'
    ψ_T = qe.MarkovChain(Q).stationary_distributions
    if len(ψ_T) > 1:
        raise NotUnique(f"the chain on the thresholds has {len(ψ_T)} recurrent classes")
    ψ = np.zeros((n_s, len(model.y)))
    ψ[:, T] = ψ_T[0].reshape(n_s, k)
    return ψ

def market_outcomes(model, λ, c=0.0):
    "Everything about a market at a given value of search, when leaving costs c."
    v, _ = worker_value(model, λ, c)
    n_bar, y_bar = thresholds(model, v, λ, c)
    ψ = stationary_distribution(model, n_bar, y_bar)
    y = model.y[None, :]
    n = np.minimum(y, n_bar[:, None])                      # employment in each (s, y)
    w = np.take_along_axis(model.R, np.maximum(n, 1), axis=1)    # and the wage
    Φ = np.cumsum(model.R, axis=1)                         # surplus: see lps_efficiency
    employment, wage_bill = (ψ*n).sum(), (ψ*n*w).sum()
    return dict(v=v, n_bar=n_bar, y_bar=y_bar, ψ=ψ, n=n, w=w,
                workforce=(ψ*y).sum(), employment=employment,
                unemployment=(ψ*(y - n)).sum(),
                # the mean wage per employed worker; empty markets get no weight
                wage=wage_bill/employment if employment > 0 else np.nan,
                wage_bill=wage_bill,
                surplus=(ψ*np.take_along_axis(Φ, n, axis=1)).sum())

out = market_outcomes(model, 7.1)
print(f"workforce {out['workforce']:.2f}, employment {out['employment']:.2f}, "
      f"unemployment {out['unemployment']:.2f}")
```

这个小链有 $n_s |T|$ 个状态，其中 $|T| \le 2 n_s$ 是不同阈值的数量，求出其不变分布所需的运算量约为 $(n_s |T|)^3$ 次。

这在七个状态时可忽略不计，在25个状态时也可处理——那里整个均衡的计算需要一两秒钟，但超过大约35个状态后就会变慢。

(lps_economy_wide)=
## 全经济均衡

到目前为止，$\lambda$ 一直是一个参数。

现在来封闭这个模型。

该经济拥有固定的工作队伍，每个市场为 $\mu$。

对任意的 $\lambda$，上面的计算给出了市场需求和工作队伍的平稳分布 $\Psi(s, y; \lambda)$，其在我们网格上的概率即数组 `ψ`。

用它对各市场的工作队伍求平均，得到该经济的**对工作队伍的需求**：

```{math}
:label: lps_demand
E(y; \lambda) = \int y \, \Psi(ds, dy; \lambda) .
```

更高的 $\lambda$ 使每个市场更快地释放工人，并且填充得更少，因此 $E(y;\lambda)$ 是递减的。

均衡就是使各市场所希望保留的工人数恰好等于该经济所拥有数量的那个 $\lambda^*$：

```{math}
:label: lps_equilibrium
E(y; \lambda^*) = \mu .
```

由于该表是递减的，二分法可以求出 $\lambda^*$。

在整数网格上，$E(y; \lambda)$ 是 $\lambda$ 的一个阶梯函数，每当某个阈值移动一个工人时就会跳跃，因此 {eq}`lps_equilibrium` 只能在一个阶梯之内成立，下文及练习中打印出的工作队伍与 $\mu$ 略有出入。

`equilibrium_λ` 首先检查 $\mu$ 是否位于区间两端所需求的工作队伍之间，若不是，则抛出 `NotBracketed`。

```{code-cell} ipython3
def workforce_demand(model, λ, c=0.0):
    "The average workforce per market at the value of search λ."
    return market_outcomes(model, λ, c)['workforce']

def bisect(test, lo, hi, tol=1e-4, checked=False):
    "Locate, to within tol, where a test that fails at lo and passes at hi switches."
    if not checked and (test(lo) or not test(hi)):
        raise NotBracketed(f"the test does not switch between {lo} and {hi}")
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        lo, hi = (lo, mid) if test(mid) else (mid, hi)
    return hi

def equilibrium_λ(model, μ, c=0.0, lo=0.01, hi=40.0):
    "Bisect on the decreasing demand schedule E(y; λ) = μ."
    E_lo, E_hi = workforce_demand(model, lo, c), workforce_demand(model, hi, c)
    if not E_lo > μ > E_hi:
        raise NotBracketed(f"μ = {μ} is not between E(y; {lo}) = {E_lo:.2f} "
                           f"and E(y; {hi}) = {E_hi:.2f}")
    return bisect(lambda λ: workforce_demand(model, λ, c) <= μ, lo, hi, checked=True)

μ = 50                        # workers per market, against a capacity of 200
λ_star = equilibrium_λ(model, μ)
eq = market_outcomes(model, λ_star)

print(f"equilibrium value of search  λ* = {λ_star:.3f}")
print(f"workforce per market            = {eq['workforce']:.2f}   (target {μ})")
print(f"employment per market           = {eq['employment']:.2f}")
print(f"unemployment per market         = {eq['unemployment']:.2f}")
print(f"unemployment rate               = {100*eq['unemployment']/eq['workforce']:.2f}%")
print(f"separation rate                 = "
      f"{100*eq['unemployment']/eq['employment']:.2f}% of the employed per period")
print(f"mean wage                       = {eq['wage']:.3f}")
```

容量限制在括号检验中体现出来。

即使在 $\lambda = 0.01$ 时，一个市场也最多想要 $N - 1 = 199$ 名工人，因此一个每个市场有200名工人的经济不存在均衡，`equilibrium_λ` 会报告这一点，而不是返回一个数值。

```{code-cell} ipython3
try:
    equilibrium_λ(model, 200)
except NotBracketed as err:
    print(err)
```

卢卡斯和普雷斯科特将均衡绘制为一条需求曲线与一条垂直的供给曲线相交，这正是均衡本身的样子。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The market for workers, and unemployment along the demand curve
    name: fig-lps-equilibrium
---
grid = np.linspace(0.6, 15.0, 45)
demand = [market_outcomes(model, λ) for λ in grid]
E_y = np.array([d['workforce'] for d in demand])
E_u = np.array([d['unemployment'] for d in demand])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(E_y, grid, lw=2, label=r'demand $E(y;\lambda)$')
axes[0].axvline(μ, color='C1', lw=2, ls='--', label=r'supply $\mu$')
axes[0].plot(eq['workforce'], λ_star, 'ko', ms=8)
axes[0].set_xlabel('workforce per market')
axes[0].set_ylabel(r'value of search $\lambda$')
axes[0].legend()

axes[1].plot(E_y, E_u, lw=2)
axes[1].plot(eq['workforce'], eq['unemployment'], 'ko', ms=8)
axes[1].set_xlabel('workforce per market')
axes[1].set_ylabel('unemployment per market')
plt.tight_layout()
plt.show()
```

左图即论文的图4。右图对应论文图5的底部面板，那里绘制了两个数值例子中平均失业随 $\lambda$ 变化的情况；由于 $E(y;\lambda)$ 是递减的，改为绘制它随工作队伍变化的图像会呈现出相同的隆起形状，只是被翻转了。

每个市场的失业并*非单调*地随工作队伍规模变化，这是卢卡斯和普雷斯科特从他们自己的例子中报告的一个特征，而均衡恰好位于这个隆起的顶部附近。

在工人充裕之处，$\lambda$ 较低，每个市场都被挤压到接近其容量，而每个需求状态的两个阈值都被挤压到一起，因此即使需求发生很大变化，被移动的工人也很少。

隆起的这一侧是容量所造成的。

在工人稀缺之处，$\lambda$ 较高，工作队伍较小，许多市场完全关闭。那里每个市场的失业人数很少，仅仅是因为每个市场本身的工人就很少；而作为一个*比率*，失业率却很高，因为留下的工人都处于最好的市场中，一旦需求下降就会被释放。

失业率和空置市场的占比都随 $\lambda$ 上升而上升：

```{code-cell} ipython3
for λ in (0.6, λ_star, 10.0, 13.0):
    r = market_outcomes(model, λ)
    print(f"λ = {λ:6.3f}:  unemployment rate {100*r['unemployment']/r['workforce']:5.2f}%,"
          f"  markets with no workers {100*r['ψ'][:, 0].sum():5.1f}%")
```

(lps_learning)=
## 学习搜寻的价值

卢卡斯和普雷斯科特用论文的一个章节讨论了一个均衡本身无法回答的问题：我们已经假设工人知道 $\lambda$，但没有说明他们是如何知道这一点的。

他们写道，归根结底这是"一个心理学理论而非经济学理论的问题，因此我们在此以一种特定的'适应性'（adaptive）方式来表述它，并不为此道歉"。

他们还说明了为什么这个问题是可以处理的。工人必须学习的分布 $\Psi$ 取决于他们自身的行为，因此随着他们认知的改变，他们正在学习的对象也随之改变，这"在一般情况下，可能带来难以克服的分析困难"。

而在此处却并非如此，"因为 $\Psi$ 中与工人决策相关的唯一特征就是参数 $\lambda$"。这正是 {doc}`ls_learning` 中那种自我指涉结构，被压缩为一个单一的数值。

他们给出的答案是一个适应性的故事。假设工人相信搜寻的价值高于实际值；那么离开其市场的人数就会多于按该回报率所能安置的数量。

在论文中，过量的搜寻者是通过使到达变量 $x$ 的均值上升来安置的，这样接收的市场获得的工人数就超过了其预期，而搜寻者和留任者所得到的工资都低于他们的预期。于是他们将这一数值向下修正。

记 $g$ 为一个在零点处取零值的递减函数：

$$
\frac{d\lambda}{dt} = g(\lambda - \lambda^{*}) .
$$

稳定性是通过假设 $g$ 具有特定符号而内置于模型中的，而这个符号的依据在于需求表是向下倾斜的：当 $\lambda$ 过高时，各市场想要的工人数少于该经济所拥有的数量。`equilibrium_λ` 也依赖于同一事实：它通过括住根来求解，而非遵循论文的调整过程，但二者都依赖于 $E(y;\lambda)$ 是递减的这一事实。

这个故事可以用离散时间来表述：工人持有一个信念 $\lambda_k$，各市场据此行动，这一信念会朝着对工人的超额需求的方向被修正，

```{math}
:label: lps_tatonnement
\lambda_{k+1} = \lambda_k + \gamma\, \big(E(y; \lambda_k) - \mu\big) .
```

这正是瓦尔拉斯试错过程（tâtonnement），$\lambda$ 充当价格，$E(y;\lambda) - \mu$ 充当超额需求。它具有 {doc}`bounded_rationality` 中松弛算法 {eq}`relaxation` 的形式，应用于 $T(\lambda) = \lambda + E(y;\lambda) - \mu$，并且具有 E-稳定性的符号结构：$\lambda^*$ 之所以稳定，是因为 $E(y;\lambda)$ 向下倾斜。

这种类比只是形式上的，因为 $T$ 是在一个现值上加上若干工人，而不是从一个感知的 $\lambda$ 值映射到一个实际的 $\lambda$ 值。

在 $\lambda^*$ 附近，误差 $\lambda_k - \lambda^*$ 每一步都被乘以 $1 + \gamma E'(\lambda^*)$，其中 $E'$ 是该表的斜率。因此当 $0 < \gamma < 2/|E'(\lambda^*)|$ 时迭代收敛，当 $\gamma < 1/|E'(\lambda^*)|$ 时收敛且不会出现超调。

由于 $E(y;\lambda)$ 在整数网格上是一个阶梯函数，我们用一个跨越足够多阶梯的中心差分来估计其斜率。

```{code-cell} ipython3
h = 0.5
slope = (workforce_demand(model, λ_star + h) - workforce_demand(model, λ_star - h))/(2*h)
print(f"slope of E(y; λ) near λ*: {slope:.1f}")
print(f"monotone convergence for γ < {1/abs(slope):.3f}, "
      f"convergence for γ < {2/abs(slope):.3f}")

def learn_λ(model, μ, γ, λ_0=10.0, n=40):
    "Revise the belief about λ in the direction of the excess demand for workers."
    path = [λ_0]
    for _ in range(n):
        path.append(path[-1] + γ*(workforce_demand(model, path[-1]) - μ))
    return np.array(path)

paths = {γ: learn_λ(model, μ, γ) for γ in (0.01, 0.05, 0.12)}
```

这里给出三个增益系数（gain），两个低于该界限，一个高于该界限。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Learning the value of search with gains below and above the bound
    name: fig-lps-learning
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, gains in zip(axes, ((0.01, 0.05), (0.12,))):
    for γ in gains:
        ax.plot(paths[γ], lw=2, label=fr'$\gamma = {γ}$')
    ax.axhline(λ_star, color='k', lw=0.8, ls='--', label=r'$\lambda^*$')
    ax.set_xlabel('iteration $k$')
    ax.legend(loc='upper right', framealpha=1)
axes[0].set_ylabel(r'belief $\lambda_k$')
plt.tight_layout()
plt.show()
```

当 $\gamma = 0.01$ 时，信念缓慢地下降到 $\lambda^*$；而当 $\gamma = 0.05$，接近 $1/|E'|$ 时，它只需几步即可到达。

当 $\gamma = 0.12$，超过该界限时，它并不收敛：它在一个高值和一个低值之间来回跳跃，在那两处表的斜率不同，超调不再增长。

这些界限的单位是每个工人的价值：如果每个市场的规模都翻一番，$E'$ 也会翻一番，稳定的增益系数就会减半。

卢卡斯和普雷斯科特特意补充说，这种稳定性"与对经常性重复发生的冲击的动态反应毫无关联"，这正是学习一个均衡与在一个均衡内做出反应之间的区别。{doc}`ls_learning` 研究了当主体通过对数据做最小二乘估计，而非精确地计算 $E(y;\lambda)$ 时，会发生什么变化。

## 均衡内部

现在让我们看看均衡的内部：单个市场、工作队伍的分布，以及相同的工人所获得的工资。

### 一个市场的样子

这两个阈值描述了一个市场行为的全部内容。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The two thresholds at the equilibrium, and the locus of employment and wages as demand shifts
    name: fig-lps-thresholds
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(model.s, eq['n_bar'], 'o-', lw=2, label=r'$\bar n(s)$: employment')
axes[0].plot(model.s, eq['y_bar'], 's--', lw=2, label=r'$\bar y(s)$: arrivals fill to')
axes[0].set_xlabel('demand state $s$')
axes[0].set_ylabel('workers')
axes[0].legend()

employs = eq['n_bar'] > 0                 # a state with n̄(s) = 0 employs nobody
wage_at_nbar = model.R[np.arange(len(model.s)), eq['n_bar']]
axes[1].plot(eq['n_bar'][employs], wage_at_nbar[employs], 'o-', lw=2)
axes[1].set_xlabel(r'employment $\bar n(s)$')
axes[1].set_ylabel('wage')
plt.tight_layout()
plt.show()
```

右图正是卢卡斯和普雷斯科特所称的 $SS$ 曲线，即如果一个市场的工作队伍从不对其形成约束，它在需求变动时所描出的就业与工资的组合。

它略去了 $\bar n(s) = 0$ 的那些需求状态，在那里一个市场不雇用任何人，也不支付任何工资。

这条曲线*不是水平的*。一条水平的 $SS$ 曲线意味着工人拥有单一的保留工资：高于它就接受工作，低于它就搜寻，这正是 {doc}`mccall_model` 中偏局部均衡的搜寻模型所*推导出*的结论。

在 $\bar n(s)$ 处，工资是能使留下恰好值 $\lambda$ 的那个数值，在整数网格误差范围内为 $w = \lambda - \beta \int v(s', \bar n, \lambda) f(s', s)\, ds'$，因此它与延续值的运动方向相反。

需求的下降以两种相反方向的方式改变延续值：

1. 今天较低的需求使得明天出现较低需求更为可能，这使得留下的价值降低；
1. 今天较低的就业意味着明天工作队伍较小，这使得留下的价值升高。

第一种效应存在于具有持续性报价的局部均衡模型中，例如 {doc}`mccall_persist_trans`，在那里保留工资取决于状态。

第二种效应来自其他工人的决策，只有均衡模型才具备它。二者皆可能占主导，因此 $SS$ 曲线可以向任一方向倾斜。

在此次校准中，它向上倾斜——除了在几乎从不被访问的最高需求状态附近，在更精细的网格上会出现一些细微的波动——因此第二种效应占主导：需求下降之后，较小市场中的延续值更高，而使其边际工人留下所需的工资更低。

均衡也直接显示出保留工资的缺失。

```{code-cell} ipython3
ψ, n, w = eq['ψ'], eq['n'], eq['w']
sheds = model.y[None, :] > eq['n_bar'][:, None]          # region I: workers leave
occurs = (ψ > 1e-12) & (n > 0)                           # markets that occur and employ

print(f"lowest wage in a market that nobody leaves:  {w[occurs & ~sheds].min():.3f}")
print(f"highest wage in a market that workers leave: {w[occurs & sheds].max():.3f}")
```

工人们留在支付低至第一个数字那样低工资的市场中，没有人离开；而在其他地方，工人们却离开支付第二个数字——更高工资的市场，因此没有任何一个分界点能把工人所接受的市场与他们所离开的市场区分开来。

但离开者并非是在拒绝一个报价以追求更好的东西，也没有人把他们拒之门外。在 $\bar n(s)$ 处，该市场中的每一个人，无论留下者还是离开者，获得的都是 $\lambda$；离开者本可以留下，但那样就业就会是整个工作队伍 $y$，工资就会降到 $R(s, y)$，而留下所带来的价值就会低于 $\lambda$。

因此，没有任何报价会因对照一个分界工资而被拒绝，也没有任何人在当前工资下被拒之门外：这里的边际是市场的就业水平。

### 工作队伍的分布与一条样本路径

由于到达人数是确定性的，且需求只取有限多个值，工作队伍在少数几个值之间移动，每个值都是某个需求状态的阈值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The invariant distribution of the workforce, and one market over sixty periods
    name: fig-lps-distribution
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ψ_y = eq['ψ'].sum(axis=0)                      # the distribution of the workforce
on_support = ψ_y > 1e-10
axes[0].stem(model.y[on_support], ψ_y[on_support], basefmt=' ')
axes[0].set_xlabel('workforce $y$')
axes[0].set_ylabel('probability')

T_path = 60
idx = qe.MarkovChain(model.P).simulate_indices(T_path, init=3, random_state=0)
path_y = np.empty(T_path, dtype=int)
path_y[0] = eq['n_bar'][idx[0]]
for t in range(T_path - 1):
    path_y[t+1] = np.clip(path_y[t], eq['y_bar'][idx[t]], eq['n_bar'][idx[t]])
path_n = np.minimum(path_y, eq['n_bar'][idx])

periods = np.arange(T_path)
axes[1].plot(periods, path_y, lw=2, drawstyle='steps-post', label='workforce')
axes[1].plot(periods, path_n, lw=2, ls='--', drawstyle='steps-post', label='employment')
axes[1].fill_between(periods, path_n, path_y, step='post', alpha=0.3, label='unemployed')
axes[1].set_xlabel('period')
axes[1].set_ylabel('workers')
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"on this path demand falls {np.sum(idx[1:] < idx[:-1])} times, "
      f"and the market sheds workers in {np.sum(path_n < path_y)} periods")
```

右图用一幅图展示了该模型的机制：只要需求不发生变动，市场就处于静止状态。

当需求下降到足以使 $\bar n(s)$ 跌破工作队伍水平时，两条线之间的缺口就打开了，而且只在那一期打开：该期的离开者就是失业者，到下一期他们已经身处别处。

一次较小的下降未必会打开这一缺口：一个只填充到 $\bar y(s)$ 的市场，能够承受需求下降到某个 $\bar n$ 仍高于其工作队伍水平的状态，而上面打印出的次数显示了这种情况在这条路径上发生的频率。

### 市场所处的位置

{eq}`lps_transition` 的三个区域构成了状态空间的一个划分，因此不变分布能够告诉我们经济中有多少部分处于每个区域。

```{code-cell} ipython3
def where_markets_sit(model, r):
    "Shares of markets in each region, and how often a market's demand stays put."
    y = model.y[None, :]
    sheds, fills = y > r['n_bar'][:, None], y < r['y_bar'][:, None]
    ψ, still = r['ψ'], ~sheds & ~fills
    return {'shedding workers': ψ[sheds].sum(), 'receiving workers': ψ[fills].sum(),
            'neither': ψ[still].sum(), 'with no workers': ψ[:, 0].sum(),
            'still, among the empty': ψ[still[:, 0], 0].sum()/ψ[:, 0].sum(),
            'demand stays put': ψ.sum(axis=1) @ np.diag(model.P)}

for name, share in where_markets_sit(model, eq).items():
    print(f"{name:>24}: {100*share:5.1f}%")
```

在任何给定时期，大多数市场都是完全静止的。有百分之几在释放工人，该经济中全部的失业都来自这些市场；还有大约同样多的市场正在吸纳工人。

必须相互平衡的是工人的流动，而这一点确实成立：离开释放工人市场的工人，恰好等于到达吸纳工人市场的工人，正如 {ref}`lps_ex_undirected` 所验证的那样。

第四行统计的是完全没有工人的市场，第五行显示其中大多数是静止的。

这样的一个市场之所以变空，是因为需求下降到了 $\bar n(s) = 0$ 的状态，并且在需求低到使 $\bar y(s)$ 无法为正之前，它会一直保持空置；其余的则正在被重新填充，因为需求已经恢复。

这些占比取决于离散化方式。在七个需求状态下，一个市场的需求四分之三的时间保持原状，正如最后一行所示；下面给出了同样的数据，但使用25个状态。

```{code-cell} ipython3
model_25 = create_model(n_s=25)
eq_25 = market_outcomes(model_25, equilibrium_λ(model_25, μ))
print(f"{'':>24}{'7 states':>10}{'25 states':>11}")
for (name, a), b in zip(where_markets_sit(model, eq).items(),
                        where_markets_sit(model_25, eq_25).values()):
    print(f"{name:>24}{100*a:>9.1f}%{100*b:>10.1f}%")
print(f"{'unemployment rate':>24}{100*eq['unemployment']/eq['workforce']:>9.2f}%"
      f"{100*eq_25['unemployment']/eq_25['workforce']:>10.2f}%")
```

在25个状态下，需求变动的次数更多，但每次的幅度更小，释放和吸纳工人的市场占比略有上升，但大多数市场仍然保持静止，失业率几乎没有变化。

### 工资分布

现在我们可以回到工资分布，概述部分曾说过它必须与工人的行为一起被确定。搜寻文献把工资报价的分布交给其工人，并询问他应当如何利用它。

在这里，这个分布是一个输出：每个市场都按照其劳动的边际产量支付工资，而一个市场支付多少取决于它所得到的需求，以及有多少工人选择留在那里。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Wages across identical workers, at the equilibrium
    name: fig-lps-wages
---
weight = eq['ψ']*eq['n']                     # weight a market by the workers it employs
weight = weight/weight.sum()

w, p = eq['w'].ravel(), weight.ravel()
order = np.argsort(w)
w, p = w[order], p[order]
cdf = np.cumsum(p)
quantile = lambda q: w[np.searchsorted(cdf, q)]
mean = (w*p).sum()
flow = λ_star*(1 - model.β)                # the flow value of search

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(w, bins=30, weights=p, color='C0', alpha=0.85)
ax.axvline(mean, color='k', lw=1.5, label=f'mean {mean:.3f}')
ax.axvline(flow, color='C1', lw=1.5, ls='--', label=f'flow value of search {flow:.3f}')
ax.set_xlabel('wage')
ax.set_ylabel('share of workers')
ax.set_xlim(0.5, 1.45)            # a thin tail runs past this; see the print below
ax.legend()
plt.tight_layout()
plt.show()

print(f"mean {mean:.3f}, "
      f"p10 {quantile(0.1):.3f}, median {quantile(0.5):.3f}, p90 {quantile(0.9):.3f}")
print(f"p90/p10 = {quantile(0.9)/quantile(0.1):.2f},  "
      f"highest wage paid {w[p > 1e-12].max():.3f}")
print(f"share paid less than the flow value of search: {100*p[w < flow].sum():.0f}%")
near = np.abs(w - flow) < 0.001*flow
print(f"share paid within 0.1% of it: {100*p[near].sum():.1f}%")
```

这个经济中的工人彼此完全相同：他们拥有相同的技能、相同的偏好、相同的信息，并且可以在任何时候自由离开。然而他们获得的报酬却并不相同，第九十百分位数的工资比第十百分位数的工资高出约一半。

这种分散性并非一种市场失灵或错误。它正是在任何会吸纳工人的市场中使*到达*的价值相等、并在任何会释放工人的市场中使留下或离开的价值相等的那种机制的体现。

在其他任何地方，在职者都获得了租金，因为新工人需要花费一期才能到达，所以各个职位的价值也是分散的。

```{code-cell} ipython3
held = eq['ψ']*model.y[None, :]              # weight a market by its workforce
print(f"λ* = {λ_star:.3f},  average value of a worker's position = "
      f"{(held*eq['v']).sum()/held.sum():.3f}")
print(f"share of workers whose position is worth more than λ*: "
      f"{100*held[eq['v'] > λ_star + 1e-9].sum()/held.sum():.0f}%")
```

大约十分之九的工人所持有的职位，对他们而言价值超过 $\lambda^*$。

图中的虚线是搜寻的流量价值，$(1-\beta)\lambda^*$，这是一个搜寻者的职位每期所值的数额。

大约三分之一的工人所获得的报酬低于这一水平，而他们依然留下。确切的比例是不稳定的。

在此网格上，大约十一分之一的一批工人所获工资与该流量价值的差距在千分之一以内，正如前面那次打印的最后一行所示，因此网格的微小变化就能把整个这批工人移到那条线的任一侧；在我们尝试过的各个网格中，这一比例在约30%到40%之间变动。

这其中并无任何疑难之处：一个工人所持有的并非他当前的工资，而是一个特定市场中的职位，它带有对该市场未来的一种求偿权，以及之后离开的选择权。

(lps_efficiency)=
## 效率

卢卡斯和普雷斯科特说，$\lambda$ 传递的是"增加一单位供给所带来的正确社会回报"，但他们并未证明任何福利定理。

该模型确实有一个福利定理，其证明使用的正是 {doc}`lucas_prescott_investment` 中所研究的1971年论文所使用的手法：找到一个规划问题，使均衡正是它的解。

### 剩余

写出

$$
\Phi(s, n) = \sum_{m=1}^{n} R(s, m)
$$

表示边际收入表在就业水平达到 $n$ 之前所围成的面积。

由 {eq}`lps_R` 可知，这是逆需求曲线 $D(s, \cdot)$ 在该产业产出（按企业计）之前所围成的面积：即总消费者剩余，这正是1971年论文的目标函数。

它仅在产品价格给定时（即 $D(s, Q) = s$ 时）才等于产出的价值，因此我们称之为**剩余**。

### 单个市场的规划者

考虑一个运营单个市场的规划者，它可以将工人移入或移出该市场，每释放一名工人到搜寻者队伍中就记入 $\lambda$ 的信用，每引入一名工人就被收取 $\lambda$ 的费用。

如果下一期的工作队伍为 $y'$，规划者本期雇用 $\min\{y, y'\}$ 名工人，并释放或引入其差额，因此其价值满足

```{math}
:label: lps_planner
W(s, y) = \max_{y'} \Big\{ \Phi\big(s, \min\{y, y'\}\big) + \lambda\, (y - y')
+ \beta \int W(s', y')\, f(s', s)\, ds' \Big\} .
```

这里是**边际价值论证**。$\Phi(s, \cdot)$ 是凹的，因为 $R$ 随 $n$ 下降，所以 $\Phi(s, \min\{y, y'\})$ 在 $(y, y')$ 上联合凹，而对 $y'$ 求最大化保持了凹性。

因此 $W$ 在 $y$ 上是凹的，规划者的选择由边际比较所主导，使用边际值 $\Delta W(s, y) = W(s, y) - W(s, y-1)$ 来衡量。

将多一名工人留到下一期，其价值为他的工资加上 $\beta \int \Delta W(s', y') f(s', s)\, ds'$，而释放他的价值是 $\lambda$；引入一人则花费 $\lambda$，其价值仅为贴现的延续值本身。

因此，规划者将工人释放到由 {eq}`lps_nbar` 定义的阈值（以 $\Delta W$ 替代 $v$），填充到由 {eq}`lps_ybar` 以同样方式定义的阈值，其他情况下则不改变工作队伍。

逐情形来看，边际工人在规划者释放工人处值 $\lambda$，在规划者填充工人处值 $R + \lambda$（因为他省下了一次到达的费用），而在中间情形下值 $R + \beta \int \Delta W f\, ds'$。

这正是情形 A、B2 和 B1，因此 $\Delta W$ 满足 {eq}`lps_bellman`。由 {prf:ref}`lps_prop_v` 中的唯一性，$\Delta W = v$，因此规划者的阈值就是均衡阈值，其策略就是 {eq}`lps_transition`。

这里是**加总论证**。给每个市场都配备这样一个规划者，使用相同的 $\lambda$，并将它们的目标函数相加。

项 $\lambda (y_t - y_{t+1})$ 相加之后，等于 $\lambda$ 乘以该经济总工作队伍的下降量，因为在一期中被释放的每一名工人，都会在下一期到达某处。

对任何可行的计划，总工作队伍是保持不变的，因此这些项相加为零，于是各规划者目标函数之和就是剩余的预期现值。

均衡策略使每个规划者的目标函数最大化，因此它也使其总和最大化。在 $\lambda = \lambda^*$ 处，该策略也是可行的，因为 $E(y; \lambda^*) = \mu$ 使总工作队伍保持不变。

因此，没有任何转移工人的可行方式能够产生比平稳均衡更高的剩余预期现值。

### 该定理说了什么，又没有说什么

精确地说，平稳均衡在以下条件下最大化**剩余的预期现值**：

* 从其自身的平稳分布 $\Psi(\lambda^*)$ 出发，
* 在所有可行计划之中，包括非平稳的计划以及依赖于其他市场历史的计划，并要求转移需要花费一期，
* 市场数量连续统，因此均衡计划使该经济每期的工作队伍保持不变，
* 在整数网格的误差范围之内，在该网格上 $E(y; \lambda^*)$ 与 $\mu$ 相差一个阶梯，而非完全相等。

$\lambda^*$ 是增加一名工人到搜寻者队伍中的现值；就每期而言，经济工作队伍固定这一约束的乘子是 $(1-\beta)\lambda^*$，即前面工资图中的搜寻流量价值。

该定理关乎总剩余，而不论谁获得它，它并未说明剩余如何被分配：{ref}`lps_ex_cost` 找到了一个能降低剩余、却仍能使工人境况更好的政策。

它也并未对各个稳态进行排序。它的比较是从 $\Psi(\lambda^*)$ 出发，并计算通向任何其他配置途中发生的一切。

一个导向不同稳态的政策，可能会在该稳态中提高剩余，同时降低现值，因为到达那里首先需要耗费剩余。

{ref}`lps_ex_planner` 计算了两个例子，一个是对转移的小额补贴，另一个是比工人更有耐心的规划者所采用的阈值，二者都把稳态下每个工人的剩余提高到均衡水平之上。

这正是增长理论中**修正黄金法则**的逻辑：一个对未来进行贴现的规划者所选择的资本量，会少于使消费最大化的稳态资本量，因为建设额外的资本首先要花费消费。

此处重新分配一名工人，当前要花费他的一期产出，而回报则在以后获得，因此一个进行贴现的规划者所进行的重新分配会更少，其失业也会比使稳态剩余最大化的那种规划者更少。

这种效率是相对于转移这项需要花费一期的技术而言的，一旦未来按 $\beta$ 贴现，均衡所拥有的失业量，恰好等于这一技术所值得承受的失业量。

## 与其他讲座的关系

在 {cite:t}`BaleyLjungqvistSargent2023` 的第一条脚注中提到，摩擦性失业并不存在"标准模型"，而是存在三类可行的模型：戴蒙德-莫滕森-皮萨里德斯传统中的匹配模型、{cite:t}`McCall1970` 的均衡版本，以及 {cite:t}`LucasPrescott1974` 传统中的搜寻-岛屿模型。

这三类模型的校准版本都能与劳动力市场流动数据相吻合，并对诸如失业保险和裁员税之类的政策给出合理的失业反应。

本讲座属于第三类，本系列中关于麦考尔（McCall）的讲座研究的则是第二类模型核心中局部均衡下工人的问题。此模型为它们所增添的内容，正是报价之外的一切。

一名工人所获得的工资报价，是其他工人决定前往何处所决定的工作队伍规模的那个市场中劳动的边际产量，而寻找他处的价值 $\lambda$，正是使整个经济中工人市场出清的那个数值。

一个结果是不存在保留工资，正如 $SS$ 曲线所显示的那样。另一个涉及就业与失业之间的流动率。

{doc}`lake_model` 将这两个比率都视为参数：找到工作的比率为 $0.283$（该讲座称之为 $\lambda$，不要与此处搜寻的价值相混淆），分离率为 $0.013$，这些数值作为输入，失业率由此得出。

{doc}`endogenous_lake` 使找到工作的比率成为一个麦考尔问题的结果，但仍然对照一个给定的工资分布。

在本讲座中，分离率是一个均衡对象，而找到工作的比率实质上由构造决定，近似为1。

上面打印出的分离率，约为每期就业人数的 $2.2$ 百分比，它结合了需求变动把市场带到其就业阈值以下的频繁程度，以及这样一个市场每次释放多少工人。

```{warning}
本模型中找到工作的比率*实质上为1*。

一名离开的工人在下一期开始时会被安置到某个市场中，因此一次失业的持续时间恰好为一期，而该模型对失业*持续时间*无话可说。

严格来说，一个接收搜寻者的市场，其需求可能下降得如此之多，以至于它释放的工人数超过了新工人到达前它所拥有的数量，使得部分新来者不得不再次移动；在此次校准中，即使新来者是首先被释放的对象，这种情况发生的概率也不到百分之一的搜寻者。
```

### 岛屿模型的后续发展

{cite:t}`AlvarezVeracierto1999` 采用了这一模型并将其变成一个定量政策分析的工具，用他们的话说是"对卢卡斯和普雷斯科特（1974）的岛屿模型进行修改，以纳入非定向搜寻和退出劳动力的参与"。

有四点发生了变化。搜寻变得**非定向**：搜寻者"均匀地分布到所有岛屿"，因此一个岛屿下一期的工作队伍是其自身的就业加上其他每一个岛屿都获得的同样的流入量，无论它是否想要这些工人。

于是一名搜寻者可能会落入一个自身正在释放工人的市场，不得不再次移动，因此失业的持续时间就成了一个均衡对象；在他们的自由放任基准情形下，平均为 $2.4$ 个时期。

非就业者获得了第二个选择，即在家生产，因此劳动力参与成为一个边际选择，模型从而有三种状态而非两种。

有了在家生产，搜寻者的流入由搜寻与留在家中之间的无差异来确定，因此劳动力规模成为一个结果，这一点很重要，因为他们所研究的政策恰恰作用于人们是否出去找工作这一问题。

岛屿以一种受持续性生产率冲击影响的递减报酬技术进行生产，而不是面对一条变动的需求曲线。并且各主体交易一整套完备的或有状态证券。

他们的自由放任均衡"是有效率的（尽管存在搜寻摩擦）"，这是他们经济中与 {ref}`lps_efficiency` 相对应的结论，他们以此作为基准，评估了最低工资、两种工会模型、裁员税和失业保险，模型的一个时期为六周。

举例来说，一项相当于平均工资十二个月的裁员税，会使他们的失业率从 $5.3\%$ *下降*到 $3.7\%$，并使平均失业持续时间从 $2.4$ 个时期*上升*到 $5.1$ 个时期，同时将福利降低消费当量的 $2.1\%$ 到 $2.3\%$。

```{note}
非定向搜寻改变了 $\lambda$ 被确定的方式。

在定向搜寻下，$\lambda$ 是一种*价格*：提高它，每个市场就会更快地释放工人，并且填充得更少，因此 {eq}`lps_equilibrium` 把它确定下来，正如市场出清条件确定任何其他价格一样，而 {eq}`lps_bellman` 中的 $\min$ 使每个接收工人的市场对他们而言恰好值 $\lambda$。

在非定向搜寻下，一名搜寻者会随机落入一个市场，因此 $\lambda$ 需要其自身的一致性条件，共同的流入量 $U$ 成为第二个未知数。

在人口固定的情况下，该模型仍然是确定的：存在两个条件——人口条件和 $\lambda$ 的一致性条件——对应两个未知数。

在此次校准中，一旦流入量允许为分数，均衡就存在且唯一，但一般而言唯一性不再能够从单一的递减表中得出。

{ref}`lps_ex_undirected` 对此进行了计算：非定向搜寻降低了搜寻的价值，也降低了失业。
```

### 劳动力流动的回报

卢卡斯和普雷斯科特在讨论转移成本时以一个猜想收尾："这类实际的或感知到的更换工作成本的差异，可能有助于解释所观察到的不同职业之间以及不同国家之间平均失业率的差异"。

为什么平均失业率在各国之间存在差异，成了这类模型所关注的核心问题；例如 {cite:t}`LjungqvistSargent1998` 将欧洲失业率的上升追溯到慷慨的失业救济金与更为动荡的经济环境相互作用的结果。

{doc}`turbulence_inference` 计算了他们模型的自由放任版本，并探讨了动荡能在多大程度上解释美国收入不稳定性的上升。

{cite:t}`BaleyLjungqvistSargent2025` 将**劳动力流动的回报**——一名工人在他将要去的地方比在他现在所在的地方多产出多少——称为"所有带有摩擦性劳动力市场的现代宏观经济学模型中重要的中介力量"。

{cite:t}`BaleyLjungqvistSargent2023` 表明，同样的数量决定了裁员税对失业的影响，以及辞职时的技能损失能否逆转动荡对失业的影响。这是冲击本身以及传导这一冲击的技术二者共同作用下的一种性质。

{cite:t}`BaleyLjungqvistSargent2025` 发现，对企业规模动态进行校准、并通过新古典生产函数传导冲击的方法，能够给出对参数扰动具有稳健性的劳动力流动回报，而把人均生产率过程校准到劳动力市场统计数据的方法则可能较为脆弱。

他们所研究的模型之一是 {cite:t}`AlvarezVeracierto2001` 的模型，他们将一个生产率过程校准到戴维斯和霍尔蒂旺格的企业层面职位创造与破坏数据上。

那篇论文与 {cite:t}`AlvarezVeracierto1999` 是不同的模型：企业在一个共同的劳动力市场中招聘，该市场的位置随机变动，工资合同是刚性的，风险厌恶的主体在没有保险市场的情况下自我保险，并选择搜寻的努力程度。

一个企业用资本 $k$ 和劳动 $n$ 生产 $z\, k^{\theta_k} n^{\theta_n}$，生产率为 $z$，且 $\theta_k + \theta_n < 1$。

{cite:t}`BaleyLjungqvistSargent2025` 对该模型进行了简化，去掉了创建企业的成本和一项小额裁员税，使得每个企业都在即期市场上租赁资本和雇用劳动，并且全部企业都使边际产量相等，这给出了

```{math}
:label: lps_av_ratio
\frac{n_{\text{high}}}{n_{\text{low}}}
= \left(\frac{z_{\text{high}}}{z_{\text{low}}}\right)^{1/(1-\theta_k-\theta_n)} ,
```

因此一个冲击被一个指数放大，而当规模报酬趋近于1时，该指数会爆炸。

在阿尔瓦雷斯和维拉西耶托的参数设定下，他们发现一家低生产率企业的雇用人数仅为高生产率企业的 $3.81\%$，比例约为 $26$，并且高生产率企业占据了超过 $96\%$ 的全部就业：一个约为 $4.35$ 的指数把一个 $2.12$ 的生产率比率转化为一个约为 $26$ 的就业比率。

这是一种无摩擦的配置，因为每个企业都在同一个即期市场中招聘。在本模型中，转移需要一期，{ref}`lps_ex_two_states` 探讨了像 $26$ 这样的比率在这一摩擦之下能保留多少，而技术和冲击都与阿尔瓦雷斯和维拉西耶托的设定保持一致。

## 练习

```{exercise}
:label: lps_ex_persistence

{cite:t}`LucasPrescott1974` 从他们自己的数值例子中报告了两项比较静态分析结果：

> *总体而言，我们发现当需求的持续性较低或较高时，失业率都较低。在前一种情况下，重新分配工人几乎没有收益；而在后一种情况下，重新分配很少发生。正如预期的那样，在持续性固定不变的情况下，需求的可变性越大，失业水平就越高。*

在保持 $\mu = 50$ 不变的情况下，检验这一校准中的两点结论。

1. 计算持续性 $\rho \in \{0.6, 0.8, 0.9, 0.95, 0.99\}$ 时的均衡失业率，在此过程中保持 $\log s$ 的*平稳*标准差固定为 $0.26$；并计算标准差为 $\{0.10, 0.18, 0.26, 0.34, 0.45\}$ 时的均衡失业率，此时保持 $\rho = 0.9$。解释你所发现的形状。
1. 在需求独立（$\rho = 0$）时，`equilibrium_λ` 会报错停止。证明此时失业恰好为零，且均衡的 $\lambda$ 并不唯一。这为什么不与论文的命题5相矛盾，也不与其失业在每一期都为正的主张相矛盾？
1. 在 $\rho = 0.3$ 时重复第2部分，先用七个需求状态，再用25个需求状态。这一比较对第1部分的低端说明了什么？
```

```{solution-start} lps_ex_persistence
:class: dropdown
```

在 $\rho$ 变化时保持平稳标准差不变，正是 `create_model` 的参数 `sd` 已经实现的功能，因此每个实验只需一次调用。

```{code-cell} ipython3
def u_rate(model, μ):
    "The equilibrium unemployment rate."
    r = market_outcomes(model, equilibrium_λ(model, μ))
    return r['unemployment']/r['workforce']

for ρ in (0.6, 0.8, 0.9, 0.95, 0.99):
    print(f"ρ = {ρ:<5}  unemployment rate {100*u_rate(create_model(ρ=ρ), μ):.2f}%")
print()
for sd in (0.10, 0.18, 0.26, 0.34, 0.45):
    print(f"sd = {sd:<5} unemployment rate {100*u_rate(create_model(sd=sd), μ):.2f}%")
```

可变性所带来的效果符合预期：移动幅度更大的需求能移动更多的工人，失业率随之稳步上升。

持续性则不然：失业在 $\rho$ 约为 $0.8$ 到 $0.9$ 之间达到峰值，而在 $\rho = 0.99$ 处又回落到峰值的约四分之一。这正是卢卡斯和普雷斯科特所描述的形状，而两端较低分别出于不同的原因。

当需求在各期之间几乎独立时，一个今天运气不佳的市场预计明天会恢复平均水平，因此留下的价值几乎不依赖于今天的需求，到达阈值 $\bar y(s)$ 随 $s$ 的变化极小，几乎没有工人被释放；第2部分将这一点推向极限。

当需求几乎是永久性的时，它从一期到下一期几乎不发生变化：在七状态链上，它很少变化，且一变化就是一整档；在更精细的链上变化更频繁，但每次幅度更小。无论哪种情况，在任何一期都很少有工人有理由移动，这正是论文所说的"重新分配很少发生"。

失业需要既值得对之做出反应、又频繁发生的需求变动，只有中等程度的持续性才能带来这样的组合。

对于第2部分，`equilibrium_λ` 之所以报错，是因为不变分布不唯一，这是 {ref}`lps_regions` 注记中的第三种情形。因此我们转而寻找使一个拥有 $\mu$ 名工人的市场永不移动的那些 $\lambda$ 值，即 $\max_s \bar y(s) \le \mu \le \min_s \bar n(s)$。

两个阈值都随 $\lambda$ 上升而下降，因此该集合的两端都可以用二分法求出。

```{code-cell} ipython3
def rest_interval(model, μ, lo=0.01, hi=40.0):
    "The interval of λ at which a market holding μ workers never moves, or None."
    def ends(λ):
        n_bar, y_bar = thresholds(model, worker_value(model, λ)[0], λ)
        return y_bar.max(), n_bar.min()
    λ_a = bisect(lambda λ: ends(λ)[0] <= μ, lo, hi)     # no market fills past μ
    λ_b = bisect(lambda λ: ends(λ)[1] < μ, lo, hi)      # some market sheds below μ
    return (λ_a, λ_b) if λ_a < λ_b else None

m_iid = create_model(ρ=0.0)
n_bar_iid, y_bar_iid = thresholds(m_iid, worker_value(m_iid, 7.1)[0], 7.1)
print("rows of P identical:", np.allclose(m_iid.P, m_iid.P[0]))
print("at λ = 7.1:  n̄(s) =", n_bar_iid, "  ȳ(s) =", y_bar_iid)
print("λ at which a market holding μ never moves: from {:.3f} to {:.3f}".format(
      *rest_interval(m_iid, μ)))
```

在需求独立的情况下，$P$ 的每一行都相同，因此延续值 $\beta \sum_{s'} v(s', y) P(s, s')$ 不依赖于 $s$，$\bar y$ 也是如此。

由于在每个状态下都有 $\bar y(s) \le \bar n(s)$，这个共同的 $\bar y$ 低于每个就业阈值。因此，任何介于 $\bar y$ 和 $\min_s \bar n(s)$ 之间的工作队伍，在每个需求状态下都保持原状，各市场会逐渐漂移进入这一区间并停留下来，长期来看没有任何人移动，失业恰好为零。

该区间内的每一个工作队伍都是吸收态，因此不变分布并不唯一，打印出的区间中每一个 $\lambda$ 值都支持一个每个市场持有 $\mu$ 名工人的均衡。

该区间的宽度是离散化链的一种性质：随着状态数的增加，该区间会收窄，因为最差需求状态会移向更远处，其就业阈值会向共同的 $\bar y$ 靠近。然而失业始终保持为零，因为每个 $\bar n(s)$ 都至少等于 $\bar y$。

这一切都不与论文相矛盾，起决定作用的差异在于 $x \equiv 1$。命题5中的唯一 $\lambda$ 以及失业在每一期都为正的主张，都依赖于命题3中唯一的平稳密度，而这是随机到达通过使某些市场过度填充所带来的，在确定性到达下，这一论证并不适用。

需求独立也违反了论文所坚持的 $F(s', s)$ 在 $s$ 上严格单调这一性质，其脚注7中的例子要求 AR(1) 系数严格为正。

对于第3部分，在 $\rho = 0.3$ 时同样的检验在两个网格上给出了不同的答案。

```{code-cell} ipython3
for n_s in (7, 25):
    m = create_model(n_s=n_s, ρ=0.3)
    rest = rest_interval(m, μ)
    if rest is not None:
        print(f"{n_s:>2} states: at rest for λ from {rest[0]:.3f} to {rest[1]:.3f}")
    else:
        print(f"{n_s:>2} states: unique λ*, unemployment rate {100*u_rate(m, μ):.3f}%")
```

在七个状态下，$0.3$ 的持续性不足以使到达阈值跨越最低的就业阈值，因此经济处于静止状态，就像 $\rho = 0$ 时一样。在25个状态下，极端的需求状态更加分散，均衡是唯一的，失业为正但非常小。

因此在 $\rho = 0.3$ 时，无论哪种网格，失业都很低，正如论文所说，但它是否恰好为零，则是离散化链的一种性质。

离散化方法在另一端的影响更大。

```{code-cell} ipython3
def chain_moments(m):
    "Standard deviation and autocorrelation of log s, and the chance that s stays put."
    π = qe.MarkovChain(m.P).stationary_distributions[0]
    x = np.log(m.s) - π @ np.log(m.s)
    sd = np.sqrt(π @ x**2)
    return sd, (π*x) @ (m.P @ x)/sd**2, π @ np.diag(m.P)

print(f"{'method':>12}{'ρ':>6}{'sd':>8}{'autocorr':>10}{'P(stay)':>9}{'u rate':>9}")
for ρ in (0.9, 0.99):
    for method in ('rouwenhorst', 'tauchen'):
        m = create_model(ρ=ρ, method=method)
        sd, ac, stay = chain_moments(m)
        print(f"{method:>12}{ρ:>6}{sd:>8.3f}{ac:>10.4f}{stay:>9.3f}"
              f"{100*u_rate(m, μ):>8.2f}%")
```

Rouwenhorst 链精确地拥有目标标准差 $0.26$ 以及目标自相关系数。Tauchen 的七状态链在 $\rho = 0.9$ 时波动性过大，这使失业率上升了约四分之一。

在 $\rho = 0.99$ 时，它不仅波动性过大，持续性也过高：其需求状态几乎从不变化，因此市场几乎从不重新分配，失业几乎消失。

在七个状态下，这会使第1部分的右端看起来像左端；而在25个状态下，两种方法几乎一致。

```{solution-end}
```

```{exercise}
:label: lps_ex_cost

在其结论部分，卢卡斯和普雷斯科特考虑了离开一个市场去搜寻所需的一次性成本 $c$，使得 {eq}`lps_nbar` 的右边从 $\lambda$ 变为 $\lambda - c$：一名离开的工人获得的是扣除到达成本之后的搜寻价值净值。

到达条件 {eq}`lps_ybar` 保持不变，因为它为那些已经在搜寻的工人所进入的市场定价。

他们说这项成本"会抬高图3中的曲线 $y = \bar n(s, \lambda)$，并使图4中的'需求曲线'向下移动。结果是失业下降，均衡工资现值 $\lambda$ 下降"，并评论说这

> *说明较低的平均失业率，一般而言并不意味着工人的福利更高。*

`market_outcomes` 和 `equilibrium_λ` 已经将该成本作为其参数 `c`。

对 $c \in \{0, 0.2, 0.5, 1, 2\}$，报告均衡的 $\lambda^*$ 和 $\lambda^* - c$、失业率、每市场剩余、每市场工资总额、租金（剩余减去工资总额），以及每工人每期的工人收入——即扣除工人为移动所支付费用后的工资总额，除以工作队伍。

1. $\lambda^*$ 是否如论文所说下降？如果不是，那么下降的是什么？
1. 较低的失业是否意味着境况更好？以每个工人的现值来比较工人所获得的收益与经济所损失的部分。
1. 谁承担了成本？一种情形是将 $c$ 视为一种资源，在移动过程中被耗尽；另一种情形是将其视为一种税，其收入以总额方式返还给工人，分别回答这个问题。
1. {cite:t}`BaleyLjungqvistSargent2023` 使用能够关闭所有内生分离的最小裁员税，作为一个模型中劳动力流动回报的诊断指标。用平均工资作为单位，计算此处在7、15和25个需求状态下能够做到这一点的最小 $c$。它为什么会随状态数变化？
```

```{solution-start} lps_ex_cost
:class: dropdown
```

在进行扫描之前，先观察一下该成本在自由放任（laissez-faire）搜寻价值下对工作队伍需求的影响。

```{code-cell} ipython3
λ_0 = equilibrium_λ(model, μ)
for c in (0.0, 2.0):
    n_bar_c, y_bar_c = thresholds(model, worker_value(model, λ_0, c)[0], λ_0, c)
    print(f"c = {c}: workforce demand {workforce_demand(model, λ_0, c):5.1f},  "
          f"n̄(s) = {n_bar_c},  ȳ(s) = {y_bar_c}")
```

该成本提高了工作队伍需求，因此新的 $\lambda^*$ 高于旧的，我们从那里开始每次二分法。这也使它远离低 $\lambda$ 值区域——在那里，存在成本时不变分布并不唯一，二分法会因此停止。

在一个平稳分布中，每一位失业工人本期都恰好离开了一次他的市场，因此工人所支付的移动成本总计为 $c$ 乘以每市场的失业人数。

```{code-cell} ipython3
results = {}
for c in (0.0, 0.2, 0.5, 1.0, 2.0):
    λ_c = equilibrium_λ(model, μ, c, lo=λ_0) if c > 0 else λ_0
    results[c] = (λ_c, market_outcomes(model, λ_c, c))

income = lambda r, c: (r['wage_bill'] - c*r['unemployment'])/r['workforce']
rents = lambda r: r['surplus'] - r['wage_bill']

print(f"{'c':>5}{'λ*':>8}{'λ*-c':>8}{'u rate':>9}{'surplus':>9}{'wages':>8}"
      f"{'rents':>8}{'income':>9}")
for c, (λ_c, r) in results.items():
    print(f"{c:>5}{λ_c:>8.3f}{λ_c - c:>8.3f}"
          f"{100*r['unemployment']/r['workforce']:>8.2f}%{r['surplus']:>9.3f}"
          f"{r['wage_bill']:>8.3f}{rents(r):>8.3f}{income(r, c):>9.4f}")
```

剩余、工资和租金都是各市场的流量，而收入是各工人的流量。为了比较工人所获得的收益与经济所损失的部分，我们把二者都换算成每个工人的现值，即把每个工人的流量除以 $1 - \beta$。

```{code-cell} ipython3
β = model.β
r_0 = results[0.0][1]
pv = lambda flow: flow/(1 - β)
surplus_pw = lambda r: r['surplus']/r['workforce']
print(f"{'c':>5}{'workers gain':>14}{'surplus lost':>14}{'ratio':>7}")
for c, (_, r) in results.items():
    gain = pv(income(r, c) - income(r_0, 0.0))
    loss = pv(surplus_pw(r_0) - surplus_pw(r))
    ratio = f"{loss/gain:7.1f}" if gain > 0 else f"{'':>7}"
    print(f"{c:>5}{gain:>14.4f}{loss:>14.4f}{ratio}")

r_2 = results[2.0][1]
print(f"\nc = 2: unemployment falls {100*(1 - r_2['unemployment']/r_0['unemployment']):.0f}%, "
      f"the wage bill rises {100*(r_2['wage_bill']/r_0['wage_bill'] - 1):.1f}%, "
      f"rents fall {100*(1 - rents(r_2)/rents(r_0)):.0f}%")
print(f"moving costs paid, a present value per worker: "
      f"{pv(2.0*r_2['unemployment']/r_2['workforce']):.4f}")
```

对于第1部分，卢卡斯和普雷斯科特说该成本会使图4中的需求曲线下移，并降低 $\lambda$，但此处恰好相反。

在自由放任的 $\lambda^*$ 处，该成本使工作队伍需求提高了约一半。它在每一个有人工作的需求状态下都大幅提高了就业阈值 $\bar n(s)$，因为一个本会离开的工人现在还要放弃 $c$；同时它略微降低了到达阈值。

这正是论文自己的第一步，"抬高曲线 $y = \bar n(s, \lambda)$"，而由于图4的纵轴是 $\lambda$，这实际上是把需求曲线向上移动，而非向下移动。

由于想要的工人数超过了该经济所拥有的数量，$\lambda^*$ 必须上升，事实也确实如此。下降的是 $\lambda^* - c$，即扣除成本之后离开这一选择权的价值，这正是一名工人在决定是否留下时所比较的数值，论文的表述对这个数值成立，而非对 $\lambda^*$ 本身成立。

对于第2部分，成本为2时，按度量得到的失业下降了接近十分之九。而按扣除移动成本后的收入来衡量，从 $c = 0.5$ 起，工人明显受益。

对于更小的成本，他们为移动所支付的费用几乎抵消了额外的工资，因此收益接近于零，在此网格上，$c = 0.2$ 时甚至略为负值。

经济所损失的超过了工人所获得的：在 $c = 2$ 时，剩余的下降幅度约为工人收入上升幅度的两到两倍半。这些都是稳态之间的比较，并未计入从一个稳态过渡到另一个稳态的过程。

因此较低的失业并非一个福利指标，但这一教训与论文所给出的形式有所不同：卢卡斯和普雷斯科特的工人境况变差是因为 $\lambda$ 下降，而我们的工人境况变好，整个经济却变差了。

该成本所消除的失业是*生产性的*。那些工人在任何有意义的层面上都不是闲置的；他们正处于从边际产量已经下降的市场转移到边际产量更高的市场的途中，而该成本使他们留在原地。

对于第3部分，工资总额略有上升，而租金——剩余减去工资总额——下降了约五分之一。

剩余是 $R$ 之下的面积，因此租金是企业的利润，加上——当产品需求向下倾斜时——该市场商品购买者的消费者剩余，承担这部分的正是他们。

账目如何平衡取决于 $c$ 究竟是什么。如果它是一种真实的资源，上面所打印出的移动成本也会被额外耗费，在剩余损失之上，由工人承担。

如果它是一种以总额形式返还给工人的税，除了剩余损失之外没有任何东西被耗费，返还的部分会在每一期把这些移动成本原封不动地还给工人，而不改变任何人的决策，因此工人获益更多。

剩余损失的含义在这两种情形下也有所不同。{ref}`lps_ex_planner` 表明，在资源成本下，均衡正是对应于更昂贵的移动技术下规划者的配置，因此较低的剩余是更差技术所付出的代价，而非一种低效率；而在返还式的税收下，技术并未改变，均衡不再是该技术下规划者的配置，因此较低的剩余是一种效率损失。

对于第4部分，回想一下 {ref}`lps_regions` 中的注记。分离恰好在存在这样一个 $\lambda$ 时停止：在该 $\lambda$ 下，一个拥有 $\mu$ 名工人的市场永不移动——没有任何状态的到达阈值高于 $\mu$，也没有任何状态的就业阈值低于 $\mu$。

该成本会提高就业阈值，因此我们寻找使这样一个 $\lambda$ 存在的最小成本。

一旦分离停止，每个市场就会永远持有 $\mu$ 名工人，因此需求状态为 $s$ 的工人所持有的价值为 $v_s$，其中 $v = (I - \beta P)^{-1} R(\cdot, \mu)$，而当 $\beta (Pv)_s \le \lambda \le v_s + c$ 对每个 $s$ 都成立时，没有人离开或到达。

这给出了一个可以与计算结果相对照的公式，$c_{\min} \approx \beta \max_s (Pv)_s - \min_s v_s$，它忽略了整数网格。

```{code-cell} ipython3
def at_rest_values(model, μ):
    "The value of a position when every market holds μ workers for ever."
    return np.linalg.solve(np.eye(len(model.s)) - model.β*model.P, model.R[:, μ])

def min_cost(model, μ, lo=1.0, hi=20.0, c_hi=20.0):
    "The smallest c at which some λ leaves a market holding μ workers at rest."
    def at_rest(c):
        ends = lambda λ: thresholds(model, worker_value(model, λ, c)[0], λ, c)
        λ_a = bisect(lambda λ: ends(λ)[1].max() <= μ, lo, hi, tol=1e-3)  # nobody fills past μ
        n_bar, y_bar = ends(λ_a)
        return y_bar.max() <= μ <= n_bar.min()
    return bisect(at_rest, 0.0, c_hi, tol=1e-2)

print(f"{'states':>7}{'c_min/wage':>12}{'formula':>9}{'log s ranges over':>20}")
for n_s in (7, 15, 25):
    m = create_model(n_s=n_s)
    v_rest = at_rest_values(m, μ)
    wage = qe.MarkovChain(m.P).stationary_distributions[0] @ m.R[:, μ]
    formula = m.β*(m.P @ v_rest).max() - v_rest.min()
    print(f"{n_s:>7}{min_cost(m, μ)/wage:>12.2f}{formula/wage:>9.2f}"
          f"{np.log(m.s).min():>11.2f} to {np.log(m.s).max():.2f}")
```

这里作为单位所使用的工资，是在每个市场都持有 $\mu$ 名工人时的平均工资，这正是分离停止之后该经济所呈现的状态；它与自由放任下的平均工资相差不到百分之零点五。

使所有分离停止的成本随状态数的增加而增大，公式解释了原因。它由一个市场可能处于的最好与最差需求状态之间的差距所决定，而 Rouwenhorst 链把其极端状态置于 $\log s$ 的 $\pm 0.26 \sqrt{n_s - 1}$ 处，因此随着状态数增加，它们彼此分离得更远。

$\log s$ 本身是一个支撑无界的 AR(1) 过程，因此对于具有连续需求的模型而言，没有任何有限的成本能够停止所有分离：此处所计算出的数值衡量的是离散化需求过程的支撑范围。

对于 {ref}`lps_ex_two_states` 中那种需求以概率 $\rho$ 保持不变、$R(s,n) = s\, g(n)$ 的对称两状态链，该公式有一个闭式解：

$$
\frac{c_{\min}}{\bar w}
= -1 + \frac{s_{\text{high}} - s_{\text{low}}}{s_{\text{high}} + s_{\text{low}}}
\cdot \frac{1 + \beta(2\rho - 1)}{1 - \beta(2\rho - 1)} ,
$$

在那里约为 $1.05$ 个时期的工资，它取决于需求过程和 $\beta$，但不取决于 $g$ 的形状：在这个模型中，该诊断指标衡量的是需求的分散程度和持续性，而非 $R$ 的曲率。

该校准并未说明一个时期有多长；在 $\beta = 0.9$ 时，很自然地把它解读为一年，这将使每一段失业持续一年之久。作为比较，在 {cite:t}`BaleyLjungqvistSargent2023` 所用的作为基准的 Ljungqvist–Sargent 匹配模型中，相当于每位工人一年产出 $163\%$ 到 $184\%$ 的裁员税能够关闭所有内生分离，而该模型的一个时期为半个季度的一半。

贝利、永奎斯特和萨金特的观点是，这个数字必须是*很大*的才有可信度，因为在裁员成本千差万别的各个经济中，都观察到了大量的重新分配。

```{solution-end}
```

```{exercise}
:label: lps_ex_two_states

{cite:t}`LucasPrescott1974` 附录中的数值例子只使用了两个需求状态，其中

$$
P = \begin{bmatrix} \rho & 1 - \rho \\ 1 - \rho & \rho \end{bmatrix} ,
\qquad \rho = 0.9,
\qquad s \in \{1, 2\},
\qquad \beta = 0.9 .
$$

此处 $\rho$ 是需求保持原状的概率，因此 $s$ 的自相关系数为 $2\rho - 1 = 0.8$；而在七状态链中，$\rho$ 本身就是自相关系数。

构建这个经济，保持 $R(s,n) = s\,(1 - n/N)^{+}$ 且 $N = 200$，并在需要均衡的任何地方取 $\mu = 50$。

1. 在一个 $\lambda$ 的网格上，绘制工作队伍需求和平均失业随 $\lambda$ 变化的图像，如论文图5的中间和底部面板所示。
1. 在 $\lambda = 6$ 处，证明不变分布将其全部的概率质量集中在工作队伍的*两个*值上，并说明是哪两个值。
1. 利用这一事实，导出一个用 $\bar n$、$\bar y$ 和 $\rho$ 表示的每市场平均失业的公式，将其与网格上每个 $\lambda$ 的计算值相对照，并用它来解释第1部分图像的形状。
1. 证明失业*率*仅取决于 $\rho$ 以及 $r = \bar y(s_{\text{high}})/\bar n(s_{\text{low}})$，并且它永远不会超过 $1 - \rho$。
1. 现在用递减报酬技术的边际产量代替 $R$：产出为 $s\, n^{\alpha}$，其中 $0 < \alpha < 1$，因此 $R(s, n) = s\,\alpha\, n^{\alpha-1}$。对 $\alpha \in \{0.4, 0.6, 0.75\}$求解，并针对一个类似阿尔瓦雷斯和维拉西耶托设定的经济求解，其中 $\alpha = \theta_n/(1-\theta_k)$ 且 $s_{\text{high}} = 2.12^{1/(1-\theta_k)}$（取他们的 $\theta_k = 0.19$ 和 $\theta_n = 0.58$），这正是他们的技术在资本（每期租赁）已被最优选定之后的形式。报告两个需求状态下平均就业的比率、其对应的无摩擦量 {eq}`lps_av_ratio`，以及失业率。证明平均就业的比率为 $\rho r + 1 - \rho$，用它来表示失业率，并说明当劳动力流动回报上升时失业会发生什么变化。
```

```{solution-start} lps_ex_two_states
:class: dropdown
```

两状态链只需要它自己的 $P$ 和 $s$。

```{code-cell} ipython3
def two_state_model(ρ=0.9, s_high=2.0, β=0.9, y_max=260, R=linear_R, **R_args):
    "The paper's two-state demand process: demand stays put with probability ρ."
    P = np.array([[ρ, 1 - ρ], [1 - ρ, ρ]])
    return make_model(P, np.array([1.0, s_high]), β, y_max, R, **R_args)

two = two_state_model()
grid2 = np.arange(0.4, 14.0, 0.1)
res2 = [market_outcomes(two, λ) for λ in grid2]
Ey2 = np.array([r['workforce'] for r in res2])
Eu2 = np.array([r['unemployment'] for r in res2])
```

论文把 $\lambda$ 放在纵轴上。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Workforce demand and unemployment in the paper's two-state economy
    name: fig-lps-two-state
---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].plot(Ey2, grid2, lw=2)
axes[0].set_xlabel('workforce demand per market')
axes[1].plot(Eu2, grid2, lw=2)
axes[1].set_xlabel('average unemployment per market')
for ax in axes:
    ax.set_ylabel(r'$\lambda$')
plt.tight_layout()
plt.show()
```

右图具有论文图5底部面板的形状：平均失业在中等的 $\lambda$ 值处最大。

对于第2部分，看看 $\lambda = 6$ 时不变分布的支撑。

```{code-cell} ipython3
r_six = market_outcomes(two, 6.0)
support = np.where(r_six['ψ'].sum(axis=0) > 1e-12)[0]
print("workforce values with positive probability:", support)
print("n̄(s) =", r_six['n_bar'], "   ȳ(s) =", r_six['y_bar'])
```

这两个值分别是 $\bar n(s_{\text{low}})$ 和 $\bar y(s_{\text{high}})$。要了解原因，请回顾 {eq}`lps_transition`。

在低需求状态下，工作队伍被截取到 $[\bar y(s_{\text{low}}), \bar n(s_{\text{low}})]$ 之内；在高需求状态下，被截取到 $[\bar y(s_{\text{high}}), \bar n(s_{\text{high}})]$ 之内，而在这个 $\lambda$ 值下，第二个区间完全位于第一个区间之上。

因此处于低需求状态一期会将任何工作队伍带到 $\bar n(s_{\text{low}})$，处于高需求状态一期会将其带到 $\bar y(s_{\text{high}})$，于是该市场就在这两个值之间往返穿梭。在每一个使 $\bar n(s_{\text{low}}) < \bar y(s_{\text{high}})$ 成立的 $\lambda$ 处都是如此，而整个网格都满足这一点。

对于第3部分，失业只在衰退的第一期才为正：市场以 $\bar y(s_{\text{high}})$ 名工人开始，需求刚刚下降，就业为 $\bar n(s_{\text{low}})$。

这个事件就是"上一期处于高需求状态，这一期处于低需求状态"，其平稳概率为 $\tfrac12 (1-\rho)$，因此

$$
E[u] = \tfrac12 (1-\rho)\,
\big[\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})\big] .
$$

我们在网格的每一点都对该公式进行检验，$\rho$ 从转移矩阵中读取。

```{code-cell} ipython3
ρ_2 = two.P[0, 0]
gap2 = np.array([r['y_bar'][1] - r['n_bar'][0] for r in res2])
print(f"largest difference between formula and computation: "
      f"{np.abs(0.5*(1 - ρ_2)*gap2 - Eu2).max():.1e}")

n_lo = np.array([r['n_bar'][0] for r in res2])
print(f"unemployment per market peaks at λ = {grid2[Eu2.argmax()]:.1f}; "
      f"the low-demand market is closed for λ ≥ {grid2[n_lo == 0].min():.1f}")
```

该公式也解释了那个隆起：每市场的平均失业是两个阈值之间的差距，乘以跨越这一差距的概率。

在高 $\lambda$ 处，低需求市场处于关闭状态，$\bar n(s_{\text{low}}) = 0$，差距仅为 $\bar y(s_{\text{high}})$，它随 $\lambda$ 下降而上升。

一旦 $\lambda$ 低到足以使低需求市场保留一些工人，随着 $\lambda$ 下降，$\bar n(s_{\text{low}})$ 的上升速度就超过了 $\bar y(s_{\text{high}})$，差距因而缩小。因此，每市场的失业在低需求市场停止关闭的那一点达到峰值，正如打印结果的第二行所示。

对于第4部分，工作队伍在一半的时期为 $\bar n(s_{\text{low}})$，在另一半的时期为 $\bar y(s_{\text{high}})$，因此 $E[y] = \tfrac12 [\bar n(s_{\text{low}}) + \bar y(s_{\text{high}})]$，失业率为

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,
\frac{\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})}{\bar y(s_{\text{high}}) + \bar n(s_{\text{low}})}
= (1-\rho)\, \frac{r - 1}{r + 1} ,
\qquad r = \frac{\bar y(s_{\text{high}})}{\bar n(s_{\text{low}})} .
$$

它随 $r$ 上升，趋向于 $1 - \rho$，当低需求市场关闭时达到这一值。

对于第5部分，低需求状态下的平均就业为 $\bar n(s_{\text{low}})$；在高需求状态下，以概率 $\rho$ 为 $\bar y(s_{\text{high}})$，以概率 $1 - \rho$（上升期的第一期）为 $\bar n(s_{\text{low}})$，因此平均就业的比率为

$$
\mathcal{E} = \frac{E[n \mid s_{\text{high}}]}{E[n \mid s_{\text{low}}]} = \rho\, r + 1 - \rho .
$$

将 $r = (\mathcal{E} - 1 + \rho)/\rho$ 代入第4部分的失业率公式，得到

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,\frac{\mathcal{E} - 1}{\mathcal{E} - 1 + 2\rho} ,
$$

它随 $\mathcal{E}$ 上升而上升。

只有 `R` 的构造发生了变化：在整数网格上，第一名工人的边际产量 $s\,\alpha$ 是有限的，一个空置市场不支付任何工资。工人在边际上远处变得廉价，因此我们使用一个更大的网格，并在一个拥有 $\mu$ 名工人的市场恰好停止吸引到达者时的搜寻价值附近——即 {ref}`lps_ex_cost` 记号中的 $\beta \max_s (Pv)_s$——来括住 $\lambda^*$。

```{code-cell} ipython3
def power_R(s, y, α):
    "The marginal product s α n^(α-1) of the technology s n^α; R(s, 0) is reset to 0."
    return s[:, None]*α*np.maximum(y[None, :], 1)**(α - 1)

def solve_two_state(m, μ):
    "The equilibrium, bracketed around the value of search of an economy at rest."
    λ_rest = m.β*(m.P @ np.linalg.solve(np.eye(2) - m.β*m.P, m.R[:, μ])).max()
    r = market_outcomes(m, equilibrium_λ(m, μ, lo=0.5*λ_rest, hi=3*λ_rest))
    E_n = (r['ψ']*r['n']).sum(axis=1)/r['ψ'].sum(axis=1)   # mean employment by state
    return r, E_n

θ_k, θ_n = 0.19, 0.58
α_AV, s_AV = θ_n/(1 - θ_k), 2.12**(1/(1 - θ_k))
print(f"like AV: α = {α_AV:.3f}, s_high = {s_AV:.3f}")
economies = [('linear', two, None)]
economies += [(f'α = {α}', two_state_model(y_max=3000, R=power_R, α=α), α)
              for α in (0.4, 0.6, 0.75)]
economies += [('like AV', two_state_model(s_high=s_AV, y_max=3000, R=power_R, α=α_AV),
               α_AV)]

print(f"{'technology':>12}{'ratio of n':>12}{'ρr + 1 - ρ':>12}{'frictionless':>14}"
      f"{'u rate':>9}{'identity':>10}")
for name, m, α in economies:
    r, E_n = solve_two_state(m, μ)
    ratio, r_ratio = E_n[1]/E_n[0], r['y_bar'][1]/r['n_bar'][0]
    frictionless = (f"{(m.s[1]/m.s[0])**(1/(1 - α)):>14.2f}" if α is not None
                    else f"{'corner':>14}")
    print(f"{name:>12}{ratio:>12.2f}{ρ_2*r_ratio + 1 - ρ_2:>12.2f}{frictionless}"
          f"{100*r['unemployment']/r['workforce']:>8.2f}%"
          f"{100*(1 - ρ_2)*(ratio - 1)/(ratio - 1 + 2*ρ_2):>9.2f}%")
```

恒等式 $\mathcal{E} = \rho r + 1 - \rho$ 以及失业率的公式在每一个经济中都成立，正如各列所示。

无摩擦那一列是在 $\theta_k = 0$ 且 $\theta_n = \alpha$ 时的 {eq}`lps_av_ratio`：使两个状态下的边际产量 $s\,\alpha\, n^{\alpha-1}$ 相等，得到 $n_{\text{high}}/n_{\text{low}} = (s_{\text{high}}/s_{\text{low}})^{1/(1-\alpha)}$。

类似阿尔瓦雷斯和维拉西耶托设定的经济，其无摩擦比率约为 $26$，与他们带有资本的技术所给出的比率相同，这是因为将资本优化掉之后，$z\, k^{\theta_k} n^{\theta_n}$ 就变成了 $z^{1/(1-\theta_k)} n^{\theta_n/(1-\theta_k)}$ 的一个倍数。那种仅含劳动的形式正是此处恰当的比较对象，因为在这个模型中摩擦只作用于劳动。

对于线性表而言，无摩擦基准是一个角点解，而且只是刚好达到。由 $2(1 - n_{\text{high}}/N) = 1 - n_{\text{low}}/N$ 以及 $n_{\text{high}} + n_{\text{low}} = 2\mu$ 可得 $n_{\text{low}} = (4\mu - N)/3$，因此对任何 $\mu \le N/4$，低需求市场都会关闭，而这恰好就是本讲座的 $\mu = 50$。

```{code-cell} ipython3
N = 200                                    # the capacity of the two-state economy
for μ_f in (50, 55, 60):
    n_low = max((4*μ_f - N)/3, 0.0)
    ratio = (2*μ_f - n_low)/n_low if n_low > 0 else np.inf
    print(f"μ = {μ_f}: frictionless n_low = {n_low:.1f}, ratio {ratio:.1f}")
```

因此在 $\mu = 50$ 处，无摩擦比率为无穷大，在略高于此处依然很大。

在每一个经济中，该模型的反应都远小于其无摩擦基准，正如一种单期转移摩擦理应如此。一个无摩擦的经济会不断转移工人，直到各市场的边际产量相等为止；而在此处，工人只有在到达值 $\lambda$ 时才会移动，到达在 $\bar y(s_{\text{high}})$ 处停止，因为需求可能回落，而一次上升期根本需要一期才能得到完全的回应。

同类比较出现在最后一行：给阿尔瓦雷斯和维拉西耶托的企业带来约 $26$ 的就业比率的那种技术和冲击，在本模型的单期转移下给出了约 $5.6$ 的比率，尽管该冲击的持续性仍然与他们的设定有所不同。

在 $s_{\text{high}} = 2$ 的各行中，就业反应随 $\alpha$ 上升（在摩擦固定不变的情况下），失业率也随之上升。这并非巧合，而且是本讲座的发现，而非贝利、永奎斯特和萨金特的发现。

此模型中的失业就是工人花在移动上的时间，而移动所能带来的收益越大，移动的人就越多：在摩擦固定不变的情况下，一个具有更高劳动力流动回报的经济会展现出更多的重新分配，并记录下与之相伴的失业。

```{solution-end}
```

```{exercise}
:label: lps_ex_undirected

本练习探讨模型中定向搜寻所起的作用。

{cite:t}`AlvarezVeracierto1999` 用**非定向**搜寻取代了它：搜寻者不能选择去向何处，每个市场获得相同的流入量 $U$，即该经济每市场的失业人数。

1. 证明在非定向搜寻下，{eq}`lps_bellman` 中的 $\min$ 消失，并说明它原本是在实施什么经济条件。
1. 写出一个市场工作队伍的转移规律，并证明在任何平稳分布中，流入量 $U$ 等于平均流出量。在定向搜寻下，这一恒等式是否同样成立？
1. 在定向搜寻下，{eq}`lps_equilibrium` 确定了 $\lambda$。在非定向搜寻下，$\lambda$ 必须满足什么条件？对一个每市场工作队伍固定为 $\mu$ 的均衡，数一数未知数和条件的个数。
1. 计算本讲座经济在 $\mu/N = 1/4$ 下的非定向均衡。一个流入整数值到容量为 $N = 200$ 的市场的设定过于粗糙，因此以十分之一为单位计数工人：使用 $N = 2000$ 以及一个延伸到 $2600$ 的网格，这就是本讲座的经济在一个精细十倍的工作队伍网格上的版本。对流入量 $U = 1, 3, 5, 7, 9$ 和 $10$，求出满足第3部分一致性条件的 $\lambda$ 值以及每个值所服务的平均工作队伍，并绘制平均工作队伍随 $U$ 变化的图像。对一个给定的 $\mu$，一个经济有多少个均衡？在同一网格上，比较 $\mu/N = 1/4$ 时定向搜寻下的 $\lambda$ 和失业率。
1. 解释为什么非定向搜寻在此处降低了搜寻的价值，以及为什么它同时也降低了失业。
```

```{solution-start} lps_ex_undirected
:class: dropdown
```

对于第1部分，{eq}`lps_bellman` 中的 $\min$ 是竞争在搜寻者之间施加的一种限制，即一个市场的延续值不能超过 $\lambda$。一旦剥夺了他们选择去向何处的能力，就没有任何东西能够实施这一限制。

一个留下的工人只是简单地获得工资以及他自己所在的市场恰好带来的延续值，

$$
v(s, y) = \max\left\{\lambda,\; R(s, y)
 + \beta \int v\big(s', y + U\big) f(s', s)\, ds'\right\} ,
$$

而延续值可能高于或低于 $\lambda$，因为没有人能够对这一差异采取行动。

对于第2部分，就业仍然是 $n(s,y) = \min\{\bar n(s), y\}$，未被雇用者离开，流入量在各处都相同，因此

$$
y' = n(s, y) + U .
$$

在一个平稳分布下 $E[y'] = E[y]$，因此在各市场间求平均给出

$$
U = E[y] - E[n(s,y)] = E[\,y - n(s,y)\,] :
$$

流入量等于平均流出量。

这是一个会计恒等式，它在定向搜寻下也同样成立：在 {eq}`lps_transition` 的平稳分布下，到达正在填充的市场的工人等于离开正在释放工人的市场的工人，在任何 $\lambda$ 值处都成立，而不只是在 $\lambda^*$ 处。

```{code-cell} ipython3
print(f"{'λ':>8}{'E[y]':>9}{'outflow':>10}{'inflow':>9}")
y = model.y[None, :]
for λ in (4.0, 7.1, 10.0):
    r = market_outcomes(model, λ)
    y_next = np.clip(y, r['y_bar'][:, None], r['n_bar'][:, None])
    print(f"{λ:>8.3f}{r['workforce']:>9.2f}"
          f"{(r['ψ']*np.maximum(y - y_next, 0)).sum():>10.4f}"
          f"{(r['ψ']*np.maximum(y_next - y, 0)).sum():>9.4f}")
```

因此这一恒等式不可能是区分这两个模型的关键；真正不同的是 $\lambda$ 是如何被确定的。

对于第3部分，在非定向搜寻下，一名搜寻者会落入从不变分布中随机抽取的一个市场，因此搜寻的价值必须满足

$$
\lambda = \beta \sum_{s,y} \psi(s, y)\, v(s, y) ,
$$

即落入一个随机抽取的市场的贴现价值，一个以市场为权重的平均值。

因此，一个人口固定的均衡有两个未知数：$\lambda$ 和 $U$，以及两个条件：$\mu = \sum_{s,y} \psi(s,y)\, y$，以及上面的条件。

在定向搜寻下，$\min$ 使每一个接收工人的市场对他们而言恰好值 $\lambda$，因此 $\lambda$ 不需要单独的一致性条件，仅凭 $\mu$ 就能通过一条向下倾斜的表唯一地确定它。而在此处，$\lambda$ 需要其自身的条件，并且没有什么能使该条件在 $\lambda$ 上单调。

对于第4部分，一个市场的就业阈值，是使留下仍然值 $\lambda$ 的最大工作队伍，前提是下一期的工作队伍将是 $n(s, y) + U$。

不变分布是唯一的，因为一个经历过最差需求状态的市场，无论从哪里出发，最终都会持有 $\bar n(s_1) + U$ 名工人，因此我们通过对其进行迭代来求出它。

```{code-cell} ipython3
def undirected_market(model, λ, U, tol=1e-10, max_iter=100_000):
    """A market that receives U searchers every period, whether it wants them or not.
    Returns the implied value of search, the mean workforce, and n̄(s)."""
    P, R, β, y = model.P, model.R, model.β, model.y
    n_s, n_y = R.shape
    up = np.minimum(y + U, y[-1])                  # next workforce if nobody leaves
    v = np.full(R.shape, λ)
    for _ in range(max_iter):
        v_new = np.maximum(λ, R + β*(P @ v)[:, up])
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new
    else:
        raise RuntimeError(f"the value function did not converge at λ = {λ}")
    n_bar = last_at_least(R + β*(P @ v_new)[:, up], λ)
    if n_bar.max() + U > y[-1]:
        raise GridTooSmall(f"n̄ + U passes the top of the grid at λ = {λ}")
    # iterate on the distribution of (s, y) under y' = min(y, n̄(s)) + U
    to = (np.arange(n_s)[:, None]*n_y + np.minimum(y[None, :], n_bar[:, None]) + U).ravel()
    ψ = np.full(n_s*n_y, 1/(n_s*n_y))
    for _ in range(max_iter):
        ψ_new = (P.T @ np.bincount(to, weights=ψ, minlength=n_s*n_y).reshape(n_s, n_y)).ravel()
        if np.abs(ψ_new - ψ).sum() < tol:
            break
        ψ = ψ_new
    else:
        raise RuntimeError(f"the distribution did not converge at λ = {λ}")
    ψ = ψ_new.reshape(n_s, n_y)
    return β*(ψ*v_new).sum(), (ψ*y).sum(), n_bar
```

隐含的搜寻价值与 $\lambda$ 之间的差距，在两个根之间为正、在之外为负，因此对每个流入量，我们在该差距为正的一个值的两侧分别寻找一个根。

在此网格上，定向搜寻的 $\lambda^*$ 在每一个存在根的流入量处都是这样一个值；SciPy 中的 `brentq` 在括号失败时会报错停止。

```{code-cell} ipython3
def consistent_λs(model, U, λ_mid, lo=0.1, hi=12.0):
    "The roots of λ = β Σ ψ v below and above λ_mid, with the mean workforce of each."
    gap = lambda λ: undirected_market(model, λ, U)[0] - λ
    roots = (brentq(gap, lo, λ_mid, xtol=2e-3), brentq(gap, λ_mid, hi, xtol=2e-3))
    return [(λ, undirected_market(model, λ, U)[1]) for λ in roots]

N_fine = 2000                            # the lecture's economy, in tenths of a worker
m_fine = create_model(N=N_fine, y_max=13*N_fine//10)
λ_dir = equilibrium_λ(m_fine, N_fine//4)
r_dir = market_outcomes(m_fine, λ_dir)

locus = {U: consistent_λs(m_fine, U, λ_dir) for U in (1, 3, 5, 7, 9)}
gaps_10 = [undirected_market(m_fine, λ, 10)[0] - λ for λ in np.arange(6.0, 8.6, 0.5)]
print(f"at U = 10 the largest gap on a grid of λ is {max(gaps_10):.4f}")
```

这些流入量中的每一个都有两个解，每个解服务于各自的人口；在 $U = 10$ 时则没有解。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Undirected search, the populations served by the solutions at each inflow
    name: fig-lps-undirected
---
Us = np.array(list(locus))
scale = 200/N_fine                       # back to the lecture's units
fig, ax = plt.subplots(figsize=(7, 4))
for k, label in ((0, r'crowded: lower $\lambda$'), (1, r'sparse: higher $\lambda$')):
    ax.plot(scale*Us, [scale*locus[U][k][1] for U in Us], 'o-', lw=2, label=label)
ax.axhline(50, color='k', lw=0.8, ls='--', label=r'$\mu = 50$')
ax.set_xlabel('inflow $U$ per market of capacity 200')
ax.set_ylabel(r'workforce per market $\mu$')
ax.legend()
plt.tight_layout()
plt.show()
```

在给定的流入量下，这两个解属于*不同*的人口：一个是拥挤的经济，搜寻价值较低；另一个是稀疏的经济，搜寻价值较高。沿着每一条分支，人口都随流入量单调变化，这两条分支在 $U = 9$ 和 $U = 10$ 之间的一个折点相遇，它们各自的 $\mu$ 取值范围并不重叠。

因此，在此次校准中，每个人口恰好拥有一个非定向均衡 $(\lambda, U)$，所需的流入量随 $\mu$ 的增加而上升，直到折点处，此后再随之下降。

这个折点也解释了为什么 $N = 200$ 不够用：最大的均衡流入量略低于每个容量为200的市场一个工人，因此即使该网格上的 $U = 1$ 也已超过了这一水平。本讲座的经济是最接近 $\mu = 50$ 的那个点。

```{code-cell} ipython3
U_eq, k_eq = min(((U, k) for U in locus for k in (0, 1)),
                 key=lambda Uk: abs(locus[Uk[0]][Uk[1]][1] - N_fine//4))
λ_und, y_und = locus[U_eq][k_eq]
print(f"undirected: U = {U_eq}, λ = {λ_und:.3f}, workforce {scale*y_und:.2f}, "
      f"unemployment rate {100*U_eq/y_und:.2f}%")
print(f"directed:   λ* = {λ_dir:.3f}, workforce {scale*r_dir['workforce']:.2f}, "
      f"unemployment rate {100*r_dir['unemployment']/r_dir['workforce']:.2f}%")
```

它位于拥挤的那条分支上，接近折点，非定向搜寻给出了比定向搜寻更低的搜寻价值和更低的失业率。

对于第5部分，比较各个就业阈值。

```{code-cell} ipython3
n_bar_und = undirected_market(m_fine, λ_und, U_eq)[2]
n_bar_low, _ = thresholds(m_fine, worker_value(m_fine, λ_und)[0], λ_und)
print("n̄(s), undirected:                       ", n_bar_und)
print("n̄(s), directed, at its own λ*:          ", r_dir['n_bar'])
print("n̄(s), directed, at the undirected λ:    ", n_bar_low)
```

非定向搜寻降低了搜寻的价值，因为搜寻者不再能够避开那些不需要他们的市场：在定向搜寻下，每个搜寻者都会前往到达价值恰为 $\lambda$ 的市场，而在此处，许多人却落入正在释放工人或即将释放工人的市场。

搜寻价值较低时，那些所在市场需求已经下降的工人离开得更慢：就业阈值在每个需求状态下都更高，而除了在最低的需求状态之外，大部分差距正是定向搜寻本身在同样较低的 $\lambda$ 下所会给出的，正如最后一行所示。因此各市场保留了更多的工人，该经济以更小的搜寻者流量来维持其人口，而这就是它的失业。

注意，对于给定的流入量，从一个感知的 $\lambda$ 到隐含值 $\beta \sum \psi v$ 的映射，是一个从感知的搜寻价值到实际搜寻价值的真正映射，这与 {ref}`lps_learning` 中的试错过程不同。

```{solution-end}
```

```{exercise}
:label: lps_ex_phillips

该模型没有总量冲击，但卢卡斯和普雷斯科特在其结束语的开头论证说，其机制"与现在为人熟知的、用预期来解释观察到的菲利普斯曲线的说法是一致的"：

> *总需求的一次未被预期到的变化（即 $E(s'|s)$ 的一次变化），会使失业与工资变化朝相反方向运动。*

从本讲座均衡的不变分布（它描述了上一期的经济）出发检验这一点。

本期，随机抽取的一部分市场所抽到的需求状态，比每个人都知道的运动规律本会给出的状态低一档；其余市场照常抽取；这一部分的大小被选定为使 $\log s$ 的均值下降 $\Delta$。

没有人预见到这一点，因此阈值 $\bar n(s)$ 和 $\bar y(s)$ 保持原样，$\lambda$ 也保持在 $\lambda^*$。

1. 对 $\Delta = 0.01, 0.02, 0.04$，以及对 $\log s$ 均值上升同样幅度的情形，计算本期的失业率和平均工资。
1. 绘制失业率随平均工资百分比变化的图像。解释其形状。
1. 卢卡斯和普雷斯科特补充说，如果这类变化"是一种反复出现的事件，正如现实中那样，那么这一事实就会被纳入工人所面对的最大化问题之中"。计算过程中会有什么变化？
```

```{solution-start} lps_ex_phillips
:class: dropdown
```

这一意外冲击只改变了本期所适用的转移矩阵，因此我们按照上一期市场已有的阈值将其带到本期，并从改变后的矩阵中抽取本期的需求。

```{code-cell} ipython3
def demand_surprise(model, r, Δ):
    """This period's unemployment rate and mean wage when mean log demand moves by Δ:
    a share |Δ|/notch of markets draws a state one notch away from the usual draw."""
    n_s, y = len(model.s), model.y
    notch = np.log(model.s[1]/model.s[0])
    step = int(np.sign(Δ))
    move = np.zeros((n_s, n_s))                       # shift a state, within the grid
    move[np.arange(n_s), np.clip(np.arange(n_s) + step, 0, n_s - 1)] = 1
    P_now = (1 - abs(Δ)/notch)*model.P + abs(Δ)/notch*(model.P @ move)
    y_carried = np.clip(y[None, :], r['y_bar'][:, None], r['n_bar'][:, None])
    ψ_now = sum(np.outer(P_now[i], np.bincount(y_carried[i], weights=r['ψ'][i],
                                               minlength=len(y)))
                for i in range(n_s))
    n, w = r['n'], r['w']
    rate = (ψ_now*(y[None, :] - n)).sum()/(ψ_now*y[None, :]).sum()
    return rate, (ψ_now*n*w).sum()/(ψ_now*n).sum()

r_star = market_outcomes(model, equilibrium_λ(model, μ))
base_rate, base_wage = demand_surprise(model, r_star, 0.0)
print(f"no surprise: unemployment rate {100*base_rate:.2f}%, mean wage {base_wage:.4f}")
print(f"{'Δ':>7}{'u rate':>9}{'wage change':>13}{'per unit of Δ':>15}")
points = [(100*base_rate, 0.0)]
for Δ in (-0.04, -0.02, -0.01, 0.01, 0.02, 0.04):
    rate, mean_wage = demand_surprise(model, r_star, Δ)
    change = 100*(mean_wage/base_wage - 1)
    points.append((100*rate, change))
    print(f"{Δ:>7}{100*rate:>8.2f}%{change:>12.2f}%{change/(100*Δ):>15.2f}")
```

无意外情形这一行再现了均衡，这是应当的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: An unanticipated shift in demand moves unemployment and wages in opposite directions
    name: fig-lps-phillips
---
u_pts, w_pts = np.array(sorted(points)).T
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(u_pts, w_pts, 'o-', lw=2)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('unemployment rate (%)')
ax.set_ylabel('change in the mean wage (%)')
plt.tight_layout()
plt.show()
```

失业和工资朝相反的方向运动，正如卢卡斯和普雷斯科特所说，这些点描出了一条向下倾斜的曲线，在均衡处出现一个拐点：一条菲利普斯曲线。

需求的一次意外下降，把一些市场推到其就业阈值之下，它们就释放工人，因此失业大幅上升。工资也会下降，但下降幅度小于需求，因为释放工人的市场削减了就业，这提高了留下者的边际产量。

需求的一次意外上升只能通过阻止原本会发生的分离来降低失业，而这样的分离本就不多；与此同时，每个市场的工资都会随其需求几乎等比例地上升，而在该期内也不会有新工人到来对其形成抑制。

因此这条曲线在失业较低处陡峭，在失业较高处平缓。

符号和拐点并不依赖于网格，但失业反应的大小确实如此：在七状态链上，一档约为 $\log s$ 的五分之一，因此给定的平均 $\log$ 需求下降会集中在少数几个市场上，每个都受到重创；而在更精细的链上，同样的下降所引起的失业上升幅度更小。

对于第3部分，这一计算之所以保持阈值和 $\lambda$ 不变，是因为没有人预料到这一变化。如果这类变化反复出现，工人就会把它们纳入自己所使用的转移规律之中，贝尔曼方程就会改变，$\bar n$、$\bar y$ 和 $\lambda^*$ 也会随之改变；卢卡斯和普雷斯科特将此留给"未来研究中的非平凡发展"。

{doc}`phillips_two_stories` 沿着菲利普斯曲线的预期论述及其背后的自然率假说，深入到了数据之中。

```{solution-end}
```

```{exercise}
:label: lps_ex_planner

{ref}`lps_efficiency` 证明了均衡求解了规划者的问题 {eq}`lps_planner`。对此进行数值检验，看看它意味着什么，又不意味着什么。

1. 在 $\lambda^*$ 处，通过价值函数迭代（在整个网格上选取 $y'$）求解 {eq}`lps_planner`，验证 $W(s, y) - W(s, y-1) = v(s, y)$，并验证规划者对 $y'$ 的选择就是 {eq}`lps_transition`。
1. 现在让移动耗费资源：每一名离开的工人耗费 $c = 2$。证明 {ref}`lps_ex_cost` 中的均衡，正是面对这一成本、每释放一名工人获记 $\lambda - c$ 的信用、每引入一名工人被收取 $\lambda$ 费用的规划者所做出的配置。这对 {ref}`lps_ex_cost` 中较低的剩余意味着什么？
1. 在保持 $\mu = 50$ 不变的情况下，比较均衡下稳态每工人剩余与两种替代方案：一种是对移动的小额补贴（$c = -0.1$ 和 $c = -0.2$），另一种是一个更有耐心的规划者会选择的阈值（$\beta' = 0.95$ 和 $0.99$，$\lambda$ 被选定以保持每市场 $\mu$ 名工人）。均衡是否使稳态剩余最大化？将这一答案与 {ref}`lps_efficiency` 相协调。
```

```{solution-start} lps_ex_planner
:class: dropdown
```

对于第1部分，`planner_value` 求解 {eq}`lps_planner`，对第2部分而言，释放信用降低为 $\lambda - c$。流量项不依赖于 $W$，因此我们只计算一次。

```{code-cell} ipython3
def planner_value(model, λ, c=0.0, tol=1e-10, max_iter=10_000):
    """W(s, y) = max over y' of Φ(s, min(y, y')) + credit for releases - charge for
    arrivals + β E W(s', y'), with releases credited at λ - c and arrivals charged λ."""
    P, R, β, y = model.P, model.R, model.β, model.y
    Φ = np.cumsum(R, axis=1)
    moved = y[:, None] - y[None, :]                  # workers released, for each (y, y')
    flow = (Φ[:, np.minimum(y[:, None], y[None, :])]
            + np.where(moved > 0, (λ - c)*moved, λ*moved))
    W = np.zeros(R.shape)
    for _ in range(max_iter):
        value = flow + β*(P @ W)[:, None, :]
        W_new = value.max(axis=2)
        if np.max(np.abs(W_new - W)) < tol:
            return W_new, y[value.argmax(axis=2)]
        W = W_new
    raise RuntimeError(f"no convergence in {max_iter} iterations at λ = {λ}")

def check_planner(model, λ, c=0.0):
    "Compare the planner's marginal values and policy with the equilibrium."
    r = market_outcomes(model, λ, c)
    W, y_choice = planner_value(model, λ, c)
    law = np.clip(model.y[None, :], r['y_bar'][:, None], r['n_bar'][:, None])
    print(f"c = {c}, λ = {λ:.3f}: largest |ΔW - v| = "
          f"{np.abs(np.diff(W, axis=1) - r['v'][:, 1:]).max():.1e}, "
          f"policy equals the transition law: {np.array_equal(y_choice, law)}")

λ_0 = equilibrium_λ(model, μ)
check_planner(model, λ_0)
```

对于第2部分，使用资源成本进行同样的检验。

```{code-cell} ipython3
λ_2 = equilibrium_λ(model, μ, 2.0, lo=λ_0)
check_planner(model, λ_2, 2.0)
```

在有资源成本时，均衡正是对应于更昂贵的移动技术下规划者的配置：{ref}`lps_efficiency` 中的论证，在释放信用为 $\lambda - c$ 的情况下同样成立，而资源成本则计入了剩余之中。

因此，{ref}`lps_ex_cost` 在存在资源成本时所发现的较低剩余，正是更差技术所付出的代价，而均衡正是对这一代价所能做出的最优应对。一个返还式的税以完全相同的方式改变了决策，却没有耗费任何东西，因此技术仍是自由放任下的技术，其规划者会选择自由放任下的配置，而被征税的均衡则是被扭曲的：它较低的剩余是一种效率损失。

对于第3部分，一项补贴就是一个负的 $c$，而一个更有耐心的规划者的阈值来自以 $\beta'$ 取代 $\beta$ 的工人问题。

```{code-cell} ipython3
def steady_state(model, policy, λ):
    """Steady-state surplus, unemployment and workforce per market of model when its
    markets follow the thresholds that the worker's problem in policy gives at λ."""
    n_bar, y_bar = thresholds(policy, worker_value(policy, λ)[0], λ)
    ψ = stationary_distribution(model, n_bar, y_bar)
    y = model.y[None, :]
    n = np.minimum(y, n_bar[:, None])
    return ((ψ*np.take_along_axis(np.cumsum(model.R, axis=1), n, axis=1)).sum(),
            (ψ*(y - n)).sum(), (ψ*y).sum())

r_0 = market_outcomes(model, λ_0)
print(f"{'':>24}{'surplus per worker':>20}{'u rate':>9}{'workforce':>11}")
print(f"{'equilibrium':>24}{r_0['surplus']/r_0['workforce']:>20.5f}"
      f"{100*r_0['unemployment']/r_0['workforce']:>8.2f}%{r_0['workforce']:>11.2f}")
for c in (-0.1, -0.2):
    r = market_outcomes(model, equilibrium_λ(model, μ, c, lo=2.0, hi=λ_0), c)
    print(f"{f'subsidy {-c}':>24}{r['surplus']/r['workforce']:>20.5f}"
          f"{100*r['unemployment']/r['workforce']:>8.2f}%{r['workforce']:>11.2f}")
for β_p in (0.95, 0.99):
    patient = model._replace(β=β_p)
    λ_p = bisect(lambda λ: steady_state(model, patient, λ)[2] <= μ, 0.01, 2000.0)
    s_p, u_p, y_p = steady_state(model, patient, λ_p)
    print(f"{f'patient planner, β = {β_p}':>24}{s_p/y_p:>20.5f}"
          f"{100*u_p/y_p:>8.2f}%{y_p:>11.2f}")
```

这两种替代方案都把每工人的稳态剩余提升到了均衡水平之上，且都是通过更多的重新分配和更高的失业率来实现的。因此均衡并没有使稳态剩余最大化，而 {ref}`lps_efficiency` 也从未声称它做到了这一点。

该定理所比较的是从均衡自身的分布出发所得到的现值，而转移到这些稳态中的任何一个，途中都要耗费剩余：更多在途工人意味着现在就损失他们一期的产出，而收益要在之后才能获得，且经过了贴现。

一个以 $\beta' > \beta$ 贴现的规划者会更看重那些之后的收益，这就是为什么它的阈值会移动更多工人；在没有贴现的极限情况下，它会使稳态剩余最大化，这正是黄金法则的对应物，而均衡则是修正黄金法则的对应物。

{ref}`lps_ex_cost` 中大额成本下的稳态比较所指向的方向，与该定理所指的方向一致，但这并非该定理所蕴含的：一个小额补贴，同样会使经济偏离规划者的配置，却能提高稳态剩余。

```{solution-end}
```