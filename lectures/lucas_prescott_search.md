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
    The economy: 经济体
    The economy::Timing: 时序
    The economy::The value of search: 搜寻的价值
    A worker's problem: 工人的问题
    Employment, wages, and two thresholds: 就业、工资与两个阈值
    Employment, wages, and two thresholds::How many work: 多少人工作
    Employment, wages, and two thresholds::How many arrive: 多少人到来
    Employment, wages, and two thresholds::Three regions, one transition law: 三个区域，一条运动规律
    A computable version: 一个可计算版本
    Economy-wide equilibrium: 经济体层面的均衡
    Learning the value of search: 学习搜寻的价值
    Inside the equilibrium: 深入均衡内部
    Inside the equilibrium::What a market looks like: 一个市场看起来是什么样子
    Inside the equilibrium::The workforce distribution and a sample path: 劳动力分布与一条样本路径
    Inside the equilibrium::Where markets sit: 市场所处的位置
    Inside the equilibrium::The wage distribution: 工资分布
    Efficiency: 效率
    Efficiency::Surplus: 剩余
    Efficiency::A planner for one market: 单个市场的计划者
    Efficiency::What the theorem says, and what it does not: 这个定理说了什么，又没说什么
    Relation to other lectures: 与其他讲座的关系
    Relation to other lectures::What the island model became: 孤岛模型演变成了什么
    Relation to other lectures::Returns to labor mobility: 劳动力流动性的回报
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

除了Anaconda中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

本讲座研究 {cite:t}`LucasPrescott1974`，这是我们在 {doc}`lucas_prescott_investment` 中研究过的投资论文在劳动力市场方面的姊妹篇。

这两篇论文由同一批作者在同一地点、相隔几年内写就，并使用了相同的工具：通过逐次近似求解的贝尔曼方程、布莱克韦尔定理，以及一个其不变分布描述长期行为的马尔可夫过程。

变化的是问题本身。

卢卡斯和普雷斯科特的研究目标是 {cite:t}`friedman1968role` 和 {cite:t}`Phelps1967` 提出的自然失业率。

用他们自己的话说，讨论自然失业率时所用的语言"暗示它可能具有竞争均衡的性质"，然而

> *并不存在任何理论模型，能够确定出一个非零的均衡失业率并研究其性质。*

没有这样一个模型，就没有一个框架来提出失业所引发的福利问题，也无法解释为什么各国的平均失业率差异如此之大。

20世纪60年代的搜寻文献无法提供这样的模型。

{doc}`mccall_model` 描述了那一文献的核心问题：工人从一个已知的分布中抽取工资要约，并决定何时停止搜寻。

这是一个关于个体的模型，它将工资分布视为给定的。

卢卡斯和普雷斯科特认为，他们所关心的问题无法以这种方式回答：

> *最优劳动供给行为及其所依据的工资分布，必须在一个市场行为（而非个体行为）模型中同时确定。*

他们构建了一个由许多空间上分离的市场组成的经济体，每个市场都有其自身随机波动的劳动需求。

这一图景来自埃德蒙·费尔普斯的"孤岛经济"，他们称之为"本模型的直接前身" {cite}`PhelpsEtAl1970`。

它不同于 {cite:t}`LucasNeutrality1972` 中的孤岛模型，在那里就业随总需求波动，但失业作为一种活动并未与闲暇区分开来。

而在这里，两者是有区分的。

一个市场中的工人可以在出清该市场的工资水平下工作，也可以离开。

离开意味着本期什么也得不到，下期会出现在别的地方，所以*失业就是工人在市场间移动的行为*。

一切都是竞争性的，每个人都在优化，并且预期是 {cite:t}`muth1961` 以及投资论文 {cite:t}`Lucas_Prescott_1971` 意义上的理性预期：工人用来评估自身前景的分布，正是由他们自己的决策所生成的分布。

这篇论文的核心对象是一个单一的数字 $\lambda$，即搜寻的预期现值。

单个市场将 $\lambda$ 视为给定，就像一个竞争性企业将价格视为给定一样。

就整个经济而言，$\lambda$ 不断调整，直到各市场想要保留的劳动力总量等于经济体实际拥有的劳动力总量。

在论文的均衡中，总有一些市场的工人数超过它们想要雇用的数量，所以每一期失业率都为正；{ref}`lps_regions` 解释了我们所使用的可计算版本在何种情况下保持这一性质。

卢卡斯和普雷斯科特还论证说，这种失业未必意味着一种低效率。

```{note}
该模型中不存在拥挤外部性，也没有匹配函数、没有议价、也没有工资粘性：失业的存在是因为移动需要一期时间，而在均衡中总有人在移动。

那个使同伴拥挤的搜寻工人，其所做的事情与任何商品卖方使其他卖方拥挤的情形是一样的，而 $\lambda$ 就像任何其他均衡价格一样，传递的是论文所称的"额外搜寻者的正确社会回报"。

这篇论文没有证明任何福利定理，它也没有判定实际的劳动力市场是否具有重要的外部效应。

{ref}`lps_efficiency` 表明，在该模型内部，这一论断可以被精确地证明：当移动需要一期时间时，平稳均衡使从其自身的工人分布中所能获得的剩余的预期现值最大化，而 $\lambda$ 正是加入搜寻者队伍中的一名工人的现值。
```

本讲座首先阐述理论：工人的问题、描述一个市场的两个阈值，以及市场劳动力的运动规律。然后计算一个示例，通过求解均衡 $\lambda$ 使经济封闭，并探讨工人如何才能学习到这个值。

在深入考察均衡内部——市场、劳动力和工资——之后，本讲座证明了效率的论断，并将该模型与其后续发展联系起来。练习部分涉及论文自身的比较静态分析、一项移动成本、论文自身的两状态例子、非定向搜寻、菲利普斯曲线，以及效率的数值检验。

让我们先导入一些库：

```{code-cell} ipython3
import warnings
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from collections import namedtuple
from scipy.optimize import brentq
```

## 经济体

存在许多市场，每个市场都有其自身的需求状态。

需求冲击在*各市场之间是相互独立的*，因此总需求是不变的；但在一个市场*内部具有持续性*，因此今天的需求能够告诉工人一些关于明天需求的信息。

在每个市场中，一个竞争性行业以劳动力作为唯一投入生产单一商品。

如果 $p = D(s, Q)$ 是需求状态 $s$ 下的逆需求曲线，而 $q(n)$ 是 $m$ 家相同企业中每一家的生产函数，那么利润最大化就使工资等于劳动的边际价值产品，而函数

```{math}
:label: lps_R
R(s, n) = D\big(s, m\,q(n)\big)\, q'(n)
```

概括了该劳动力市场需求方的全部信息。

卢卡斯和普雷斯科特舍弃了 $D$ 和 $q$，直接对 $R$ 施加假设：$R$ 为正、有界且可微，且满足

$$
R_s(s,n) > 0, \qquad R_n(s,n) < 0 .
$$

需求状态 $s$ 服从一个具有转移密度 $f(s', s)$ 的马尔可夫过程，该过程拥有唯一的平稳分布。

其条件分布函数 $F(s', s)$ 关于 $s$ 是递减的，因此今天较高的需求使明天出现较高需求的可能性更大。

### 时序

一个市场在期初拥有一个**劳动力**（workforce）$y$，这是该市场能够雇用的最大数量。

然后，在该期内：

1. 需求 $s$ 实现，劳动力市场出清，在工资 $w = R(s, n)$ 下雇用 $n \le y$ 名工人；
1. 未被雇用的 $y - n$ 名工人本期一无所获，加入搜寻者队伍；
1. 被雇用的工人留在该市场进入下一期；搜寻者被分配到各个市场，并在下一期到达。

所以每次移动时，一名工人*恰好失业一期*，失业者就是正在转移中的工人。

### 搜寻的价值

由于各市场间的需求冲击相互独立，且市场数量众多，一旦经济体内工人在各市场间的分布稳定下来，它就是恒定的，尽管每个市场自身的劳动力数量会有波动。

因此，一名搜寻者的前景并不取决于他正在离开的那个市场，搜寻的预期现值是一个单一的数字 $\lambda$。

搜寻者被分配到各个市场的方式使到达的预期价值*相等*，因此没有任何市场会给予到达的工人高于 $\lambda$ 的价值。

目前我们先将 $\lambda$ 视为给定，并在 {ref}`lps_economy_wide` 中确定它。

## 工人的问题

设 $v(s, y, \lambda)$ 为一名工人在期初身处需求为 $s$、劳动力为 $y$ 的市场中时，其收入的预期现值，并设 $\beta \in (0,1)$ 为贴现因子。

卢卡斯和普雷斯科特通过考虑三种情形，得到了 $v$ 的一个函数方程。

**情形A：部分工人离开。** 离开的工人得到 $\lambda$。留下的工人得到的不会更多，因为离开者本可以留下；也不会更少，因为留下者本可以离开。所以 $v = \lambda$。

**情形B1：所有工人留下，且没有新工人到达。** 当前就业为 $y$，工资为 $R(s,y)$，下一期的劳动力仍为 $y$：

$$
v(s,y,\lambda) = R(s,y) + \beta \int v(s', y, \lambda) f(s', s)\,ds' .
$$

**情形B2：所有工人留下，且有新工人到达。** 到达的工人是搜寻者，所以下一期身处该市场的贴现预期价值 $\beta \int v f\, ds'$ 必须恰好等于 $\lambda$，即促使他们前来搜寻的价值：

$$
v(s,y,\lambda) = R(s,y) + \lambda .
$$

当劳动力不变时的贴现延续值低于 $\lambda$ 时，没有人到达，适用情形B1；当其超过 $\lambda$ 时，到达者将其拉低至 $\lambda$，适用情形B2。

无论哪种情况，延续值都是 $\lambda$ 与 $\beta \int v(s', y, \lambda) f(s', s)\, ds'$ 二者中较小的那个，这就是下面的 $\min$ 运算。

情形A适用于即使如此工人所得仍少于 $\lambda$ 的情况，此时他就选择 $\lambda$，这就是 $\max$ 运算。

这三种情形合并为一个方程：

```{math}
:label: lps_bellman
v(s, y, \lambda) = \max\left\{\lambda,\;
R(s,y) + \min\left[\lambda,\;
\beta \int v(s', y, \lambda) f(s', s)\, ds' \right]\right\} .
```

这两个算子承担不同的职能。*max* 是工人自己的选择权：他总是可以离开并获得 $\lambda$。

*min* 则完全不是工人的选择。它是竞争在搜寻者之间施加的一种限制：如果该市场的延续值超过 $\lambda$，搜寻者就会不断涌入，直到它不再超过为止。

```{prf:proposition}
:label: lps_prop_v

对任意 $\lambda$，方程 {eq}`lps_bellman` 有唯一的有界连续解 $v$，并且从任意起点出发的逐次近似都收敛于该解。

该解关于 $s$ 和 $\lambda$ 是非递减的，关于 $y$ 是非递增的。
```

{prf:ref}`lps_prop_v` 的证明与 {doc}`lucas_prescott_investment` 全文所用的证明相同：由 {eq}`lps_bellman` 右边所定义的算子是单调的，并满足一个贴现条件，因此布莱克韦尔定理 {cite}`Blackwell1965` 适用。

关于 $s$ 的单调性正是用到了 $F(s', s)$ 关于 $s$ 递减这一假设之处。一名工人所处的市场越拥挤，他的状况就越差，因为更大的劳动力意味着更低的边际产量和更低的工资，这正是 $v$ 关于 $y$ 非递增的原因。

## 就业、工资与两个阈值

有了 $v$，市场行为的其余部分就由两条阈值规则得出。

### 多少人工作

设 $\bar n(s, \lambda)$ 为这样的就业水平：在该水平下，边际工人恰好在留下与搜寻之间无差异：

```{math}
:label: lps_nbar
R(s, \bar n) + \beta \int v(s', \bar n, \lambda) f(s', s)\, ds' = \lambda .
```

左边是当就业和下一期劳动力均为 $\bar n$ 时，一名工人留下所获得的值：本期工资加上贴现的延续值。它关于 $\bar n$ 是非递增的。

如果边际收益在某个劳动力水平处降为零——正如下面可计算版本所呈现的那样——那么身处那个拥挤市场的价值已降至 $\lambda$，左边就平稳地停在 $\beta \lambda < \lambda$ 处，而 {eq}`lps_nbar` 在此处无法成立。

在整数劳动力网格上，{eq}`lps_nbar` 很少精确成立，就业就是使左边至少等于 $\lambda$ 的最大 $\bar n$。

于是就业为

```{math}
:label: lps_employment
n(s, y, \lambda) = \min\{\bar n(s,\lambda),\; y\},
\qquad w = R(s, n) ,
```

该市场中的失业为 $y - n$，当劳动力超过 $\bar n$ 时该值为正。

### 多少人到来

只有当一个市场雇用了它的全部劳动力，且延续值仍超过 $\lambda$ 时，该市场才会吸引搜寻者。

设 $\bar y(s,\lambda)$ 为使这一情况不再成立的劳动力水平：

```{math}
:label: lps_ybar
\beta \int v(s', \bar y, \lambda) f(s', s)\, ds' = \lambda .
```

到来者会把市场填充到 $\bar y(s, \lambda)$，不会更多。注意 $\bar y \le \bar n$，因为 $R \ge 0$ 意味着任何仅凭延续值就能达到 $\lambda$ 的劳动力水平，一旦加上工资，也同样能达到 $\lambda$。

在边际产量为正的地方，这个不等式是严格的：一个对搜寻者而言刚好值得进入的市场，严格地处于所有人都工作的那个区间之内。

在下文标定中的最差需求状态下，这两个阈值在均衡的 $\lambda$ 值处都为零，该市场不雇用任何人，也不吸引任何人。

(lps_regions)=
### 三个区域，一条运动规律

{eq}`lps_employment` 和 {eq}`lps_ybar` 共同将 $(s,y)$ 平面划分为论文图3中的三个区域：

* 区域I，其中 $y > \bar n(s, \lambda)$，市场把工人裁减到 $\bar n(s,\lambda)$；
* 区域II，介于两个阈值之间，没有人移动；
* 区域III，其中 $y < \bar y(s, \lambda)$，市场被填充到 $\bar y(s,\lambda)$。

它们使劳动力的运动规律呈现出非常简单的形式：

```{math}
:label: lps_transition
y' = \min\big\{\max\{y,\ \bar y(s,\lambda)\},\ \bar n(s,\lambda)\big\} .
```

贝尔曼方程在*不变的*劳动力 $y$ 处评估延续值，而运动规律则把劳动力移到别处。

这并不矛盾，因为 $\max$ 和 $\min$ 已经处理了劳动力发生移动的那两种情形。

当有到来者时，$\min$ 把延续值替换为 $\lambda$，而一个被填充到 $\bar y$ 的市场对其中任何人来说至少值 $\lambda$：在连续劳动力情形下恰好为 $\lambda$，在下文所用的整数网格上可能略高一些。当工人离开时，$\max$ 把整个表达式替换为 $\lambda$。

因此，不变 $y$ 处的延续值仅在劳动力确实不变的那个区域内被用到。

```{note}
论文让到来者的数量是随机的：一个劳动力不足的市场会收到 $a(s,y,\lambda)\,x$ 名工人，其中 $x$ 是一个均值为一的正随机变量，因此搜寻必须"平均而言消除租金"，而非精确消除。

和论文附录中的数值例子一样，我们令 $x \equiv 1$，这就得到了 {eq}`lps_transition`。

在论文中，随机到来使平稳分布具有严格为正的密度（其命题3），因此总有某个市场在裁减工人，失业在每一期都为正。

当 $x \equiv 1$、需求状态数量有限，且需求过程可以在一期内从任意一个状态转移到任意另一个状态时，会有三种情形。

* 如果 $\max_s \bar y(s) > \min_s \bar n(s)$，平稳分布是唯一的，失业为正：在 $\bar y$ 最高的状态下被填满的市场，在需求降至 $\bar n$ 最低的状态时会裁减工人，这两期会使一个市场从任何起始劳动力到达 $\min_s \bar n(s)$。
* 如果 $\max_s \bar y(s) = \min_s \bar n(s)$，平稳分布同样是唯一的，每个市场都保有这个劳动力水平，失业为零。
* 如果 $\max_s \bar y(s) < \min_s \bar n(s)$，任何介于两者之间的劳动力在每个需求状态下都原地不动，因此存在一个平稳分布的连续统，失业为零。

{ref}`lps_ex_persistence` 遇到了第三类经济体。
```

## 一个可计算版本

我们遵循 {cite:t}`LucasPrescott1974` 的附录：有限多个需求状态、整数劳动力，并对 {eq}`lps_bellman` 进行价值函数迭代。

对于需求过程，我们使用论文脚注7中的例子，其中 $\log s$ 服从一个持续性介于零和一之间的AR(1)过程。

我们用鲁文霍斯特（Rouwenhorst）方法将其离散化为七个状态。

与 {cite:t}`Tauchen1986` 的方法不同，它即使在持续性很高时也能精确匹配 $\log s$ 的持续性和平稳方差；{ref}`lps_ex_persistence` 的解答展示了若不这样做会出什么问题。

对于 $R$，我们取一个随就业线性下降、并在容量 $N$ 处降为零的边际收益函数，

$$
R(s, n) = s\,\big(1 - n/N\big)^{+} .
$$

在两个需求状态、$N = 100$ 的情形下，这就是论文例1中的函数，其图形绘制在论文图5的左上方，并按比例放大了两倍。

容量设定是对理论假设（即 $R$ 处处为正）的一种偏离，但这正是论文自身的例子所采用的做法。它把一个市场永远想要的劳动力上限限制在 $N$ 以下，所以论文承诺对每个每市场劳动力水平都存在均衡的命题5，在此处仅对低于该上限的劳动力水平成立。

它也塑造了当工人充裕时失业如何变化，我们将在下文看到这一点。

$R$ 下降得多陡，决定了需求变动时就业变动的幅度，从而决定了把一名工人从一个市场送到另一个市场能获得多大收益；{ref}`lps_ex_two_states` 就讨论了这一点。

函数 `make_model` 可以从任意需求链和边际收益函数构建一个市场，`create_model` 则用它来构建本讲座的经济体。

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

贝尔曼算子就是 {eq}`lps_bellman`，只是把积分替换成了求和。

由于延续值是在不变劳动力处求值，该方程对每个 $y$ 而言都是一个独立的问题，仅在需求状态之间相互关联，这正是为什么求解成本很低，也正是为什么我们对空市场所采用的约定不会影响其他任何部分。

我们需要某个搜寻价值才能求解它，所以此处使用 $\lambda = 7.1$，这个值接近 {ref}`lps_economy_wide` 所计算出的均衡值。

参数 `c` 是离开成本，在 {ref}`lps_ex_cost` 之前一直为零，它把工人的外部选项从 $\lambda$ 变为 $\lambda - c$。

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

这两个阈值分别是使相应的值仍不低于 $\lambda$ 的最大就业水平，以及最大劳动力水平。

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

就业受 $\bar n$ 的限制，到来者将劳动力填充到 $\bar y$，而 {eq}`lps_transition` 推动劳动力的变化。

$(s, y)$ 上这个马尔可夫链的不变分布可以被精确地计算出来。

{eq}`lps_transition` 的一期运动要么使劳动力原地不动，要么将其带到某个阈值上，因此从一个阈值出发，该链永远不会离开阈值构成的集合，我们可以求解关于 $(s, t)$ 对的小型链，其中 $t$ 遍历至多 $2 \times 7$ 个阈值。

一个处于阈值之外的劳动力水平，只有在它在每个需求状态下都保持不动时，才能在长期内携带正的概率。

这样一个劳动力区间的两端本身就是保持不动的阈值，因此阈值上的链在这种情况下会有不止一个不变分布：这就是 {ref}`lps_regions` 中说明的第三种情形。

因此我们让 `qe.MarkovChain` 找出这个小型链的每一个不变分布，并在存在不止一个时报错停止，而不是随意选取其中一个。

当某个阈值到达工作力网格顶端时，该函数也会停止，因为此时阈值会在没有警告的情况下被截断。

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

这个小型链有 $n_s |T|$ 个状态，其中 $|T| \le 2 n_s$ 是不同阈值的数量，寻找其不变分布的计算量约为 $(n_s |T|)^3$ 次运算。

在七个状态的情形下这一计算量可以忽略不计，在25个状态时也仍可承受——此时整个均衡的求解需要一两秒钟——但超过约35个状态之后，计算就会变慢。

(lps_economy_wide)=
## 经济体层面的均衡

到目前为止，$\lambda$ 一直是一个参数。

现在我们让模型封闭起来。

该经济体每个市场拥有固定的劳动力 $\mu$。

对任意 $\lambda$，上面的计算都给出一个市场需求和劳动力的平稳分布 $\Psi(s, y; \lambda)$，其在我们网格上的概率就是数组 `ψ`。

用它对各市场的劳动力取平均，就得到了经济体的**劳动力需求**，

```{math}
:label: lps_demand
E(y; \lambda) = \int y \, \Psi(ds, dy; \lambda) .
```

更高的 $\lambda$ 使每个市场更早裁减工人，且较少被填满，因此 $E(y;\lambda)$ 是递减的。

均衡就是使各市场恰好想要保有经济体所拥有的全部工人数量的那个 $\lambda^*$：

```{math}
:label: lps_equilibrium
E(y; \lambda^*) = \mu .
```

由于该函数是递减的，用二分法即可找到 $\lambda^*$。

在整数网格上，$E(y; \lambda)$ 是 $\lambda$ 的阶梯函数，每当某个阈值移动一名工人的量时就会跳变，因此 {eq}`lps_equilibrium` 只能在一步之差的范围内成立，下文及练习中打印出的劳动力数值与 $\mu$ 略有出入。

`equilibrium_λ` 首先检查 $\mu$ 是否落在区间两端所要求的劳动力之间，若不是，则抛出 `NotBracketed`。

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

容量限制体现在区间检查当中。

即使在 $\lambda = 0.01$ 时，一个市场也最多只想要 $N - 1 = 199$ 名工人，因此一个每市场有200名工人的经济体不存在均衡，`equilibrium_λ` 会说明这一点，而不是返回一个数字。

```{code-cell} ipython3
try:
    equilibrium_λ(model, 200)
except NotBracketed as err:
    print(err)
```

卢卡斯和普雷斯科特把均衡画成一条需求曲线与一条垂直的供给曲线相交，而它正是如此。

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

左图正是论文中的图4。右图对应论文图5下方两幅图，它们将各市场的平均失业率绘制为 $\lambda$ 的函数；由于 $E(y;\lambda)$ 是递减的，将其改为对劳动力作图所展示的是同一个驼峰，只是被镜像了一下。

每个市场的失业人数*并非*随劳动力规模单调变化，这是卢卡斯和普雷斯科特从他们自己的例子中报告的一个特征，而均衡恰好位于该驼峰的顶端附近。

当工人充裕时，$\lambda$ 较低，每个市场都被压向其容量上限，每个需求状态下的两个阈值在那里被压缩到一起，因此即使需求发生较大的变动，移动的工人也很少。

驼峰的这一侧是容量上限在起作用。

当工人稀缺时，$\lambda$ 较高，劳动力规模较小，许多市场完全关闭。那里每个市场的失业人数之所以很少，仅仅是因为每个市场的工人本就很少；作为*比率*来看，它却很高，因为留下来的那些工人身处最好的市场，一旦需求下降，就会被裁减。

失业率和空市场的比例都随 $\lambda$ 上升而增加：

```{code-cell} ipython3
for λ in (0.6, λ_star, 10.0, 13.0):
    r = market_outcomes(model, λ)
    print(f"λ = {λ:6.3f}:  unemployment rate {100*r['unemployment']/r['workforce']:5.2f}%,"
          f"  markets with no workers {100*r['ψ'][:, 0].sum():5.1f}%")
```

(lps_learning)=
## 学习搜寻的价值

卢卡斯和普雷斯科特用论文的一节讨论了一个均衡本身并未回答的问题：我们一直假设工人知道 $\lambda$，但没有说明他们是如何得知这一点的。

他们写道，归根结底这"是一个心理学理论而非经济学理论所应回答的问题，因此我们在此处以一种临时的'适应性'方式来提出这个问题，并不为此道歉"。

他们还解释了这个问题之所以是可以处理的原因。工人必须学习的分布 $\Psi$ 取决于他们自身的行为，因此随着他们看法的变化，他们所学习的对象本身也在变化，这"一般而言可能带来难以克服的分析困难"。

而在此处却不是这样，"因为 $\Psi$ 中与工人决策相关的唯一特征就是参数 $\lambda$"。这正是 {doc}`ls_learning` 中那种自我指涉的结构，被压缩成了单单一个数字。

他们给出的答案是一个适应性的故事。假设工人相信搜寻的价值比实际更高；那么离开各自市场的人数就会超过以该回报率能够被安置的人数。

在论文中，超额的搜寻者是通过让到达变量 $x$ 的均值上升来安置的，因此接收市场得到的工人数超出了它们的预期，搜寻者和留下者对他们所预期的工资都感到失望。于是他们把该数字向下修正。

将 $g$ 记为一个在零点取零值的递减函数，则

$$
\frac{d\lambda}{dt} = g(\lambda - \lambda^{*}) .
$$

稳定性内置于 $g$ 所假设的符号之中，而使这一符号成立的依据正是需求函数向下倾斜这一事实：当 $\lambda$ 过高时，市场想要的工人数少于经济体所拥有的数量。`equilibrium_λ` 依赖于同一事实：它将根括在一个区间内，而不是遵循论文的调整方式，但两者都是因为 $E(y;\lambda)$ 是递减的才能生效。

这个故事可以用离散时间来表述：工人持有一个信念 $\lambda_k$，各市场据此行动，然后该信念朝着对工人的超额需求的方向被修正，

```{math}
:label: lps_tatonnement
\lambda_{k+1} = \lambda_k + \gamma\, \big(E(y; \lambda_k) - \mu\big) .
```

这就是瓦尔拉斯式的试探过程（tâtonnement），其中 $\lambda$ 扮演价格的角色，而 $E(y;\lambda) - \mu$ 扮演超额需求的角色。它具有 {doc}`bounded_rationality` 中松弛算法 {eq}`relaxation` 的形式，应用于 $T(\lambda) = \lambda + E(y;\lambda) - \mu$，并具有E-稳定性的符号结构：$\lambda^*$ 是稳定的，因为 $E(y;\lambda)$ 向下倾斜。

这个类比仅仅是形式上的，因为 $T$ 把若干工人数加到一个现值上，并不是一个从感知值映射到 $\lambda$ 实际值的映射。

在 $\lambda^*$ 附近，误差 $\lambda_k - \lambda^*$ 每一步都乘以 $1 + \gamma E'(\lambda^*)$，其中 $E'$ 是该函数的斜率。因此当 $0 < \gamma < 2/|E'(\lambda^*)|$ 时该迭代收敛，而当 $\gamma < 1/|E'(\lambda^*)|$ 时收敛且不发生超调。

由于 $E(y;\lambda)$ 在整数网格上是阶梯函数，我们用一个跨越足够多步的中心差分来估计其斜率。

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

下面给出三种增益系数，其中两个低于该界限，一个高于该界限。

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

当 $\gamma = 0.01$ 时，信念慢慢地向 $\lambda^*$ 靠拢；当 $\gamma = 0.05$（接近 $1/|E'|$）时，它在几步之内就到达了。

当 $\gamma = 0.12$，超过了该界限时，它不再收敛：它在一个高值和一个低值之间来回跳动，那里函数的斜率不同，超调不再增大。

这些界限的单位是每名工人的价值：如果每个市场的规模都翻倍，$E'$ 也会翻倍，稳定增益的范围就会减半。

卢卡斯和普雷斯科特小心地补充说，这种稳定性"与对经常反复出现的冲击的动态反应毫无关系"，这正是学习一个均衡与在一个均衡内部作出反应之间的区别。{doc}`ls_learning` 研究了当主体根据数据用最小二乘法估计，而不是精确计算 $E(y;\lambda)$ 时会发生什么变化。

## 深入均衡内部

现在我们深入均衡内部：看单个市场、劳动力的分布，以及相同的工人所获得的工资。

### 一个市场看起来是什么样子

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

右图就是卢卡斯和普雷斯科特所称的 $SS$ 曲线，它是在一个市场的劳动力从未对其构成约束的情况下，随需求变动所描绘出的就业和工资的点对序列。

它略去了 $\bar n(s) = 0$ 的那些需求状态，在那些状态下市场不雇用任何人，也不支付任何工资。

这条曲线*并非平坦的*。一条平坦的 $SS$ 曲线意味着工人拥有单一的保留工资，高于它就接受工作、低于它就继续搜寻——这正是 {doc}`mccall_model` 中局部均衡搜寻模型所*推导*出的结果。

在 $\bar n(s)$ 处，工资恰好使留下来的价值等于 $\lambda$，即在整数网格精度下 $w = \lambda - \beta \int v(s', \bar n, \lambda) f(s', s)\, ds'$，因此它与延续值的变动方向相反。

需求的下降以两种相互对立的方式改变延续值：

1. 今天较低的需求使明天出现较低需求的可能性更大，这使留下的价值降低；
1. 今天较低的就业意味着明天劳动力规模较小，这使留下的价值提高。

第一种效应存在于具有持续性要约的局部均衡模型中，比如 {doc}`mccall_persist_trans`，其中保留工资取决于状态。

第二种效应来自其他工人的决策，只有均衡模型才具有这种效应。二者中任何一个都可能占主导，因此 $SS$ 曲线可以朝任何一个方向倾斜。

在这个标定中，它向上倾斜——除去在市场几乎从不会访问到的高需求状态中、在更精细的网格上出现的一些小波动——所以是第二种效应占主导：需求下降之后，较小市场中的延续值更高，而维持其边际工人所需的工资更低。

均衡也直接显示出保留工资的缺失。

```{code-cell} ipython3
ψ, n, w = eq['ψ'], eq['n'], eq['w']
sheds = model.y[None, :] > eq['n_bar'][:, None]          # region I: workers leave
occurs = (ψ > 1e-12) & (n > 0)                           # markets that occur and employ

print(f"lowest wage in a market that nobody leaves:  {w[occurs & ~sheds].min():.3f}")
print(f"highest wage in a market that workers leave: {w[occurs & sheds].max():.3f}")
```

工人留下来、没有人离开的市场工资可以低至第一个数字，而工人离开的市场工资却更高——是第二个数字——因此没有哪个分界线能够把工人接受的市场与他们离开的市场区分开来。

但离开者并非是拒绝了一个要约、以换取更好的东西，也没有人把他们拒之门外。在 $\bar n(s)$ 处，该市场中的每个工人，无论是留下者还是离开者，都得到 $\lambda$；离开者本可以留下，但那样一来就业就会是全部劳动力 $y$，工资就会降到 $R(s, y)$，留下来的价值就会低于 $\lambda$。

所以没有要约是相对于一个分界工资被拒绝的，也没有人在现行工资下被拒之门外：这里的边际是市场的就业水平。

### 劳动力分布与一条样本路径

由于到来者数量是确定性的，且需求只取有限多个值，劳动力在少数几个值之间移动，每一个都是某个需求状态下的阈值。

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

右图在一幅图中展示了该模型的机制：只要需求不发生变动，市场就处于静止状态。

当需求下降幅度足够大、以致 $\bar n(s)$ 降到劳动力水平以下时，两条线之间就会出现缺口，并且仅在那一期出现：那一期的离开者就是失业者，到了下一期他们已身处别处。

较小幅度的需求下降未必会打开这个缺口：一个此前只被填充到 $\bar y(s)$ 的市场，可以吸收一次需求下降，只要下降后状态的 $\bar n$ 仍高于其劳动力水平，而打印出的计数显示了这种情况在这条路径上发生的频率。

### 市场所处的位置

{eq}`lps_transition` 中的三个区域构成了状态空间的一个划分，因此不变分布说明了经济体各部分分别处于哪个区域。

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

在任何给定一期中，大多数市场完全处于静止状态。有几个百分点在裁减工人，经济体全部的失业都来自这些市场；差不多同样多的市场正在吸纳工人。

必须保持平衡的是工人的流量，而它确实保持平衡：离开裁员市场的工人恰好等于到达吸纳市场的工人，正如 {ref}`lps_ex_undirected` 所验证的那样。

第四行统计了完全没有工人的市场，第五行显示其中大多数都处于静止状态。

这样一个市场是在需求降到 $\bar n(s) = 0$ 的某个状态时空出来的，并在需求过低、使 $\bar y(s)$ 无法为正时保持空置；其余的则因为需求已经恢复而正在被重新填充。

这些比例取决于离散化方式。用七个需求状态时，一个市场的需求有四分之三的时间保持不变，如最后一行所示；下面给出了用25个状态时的相同数字。

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

使用25个状态时，需求变动的次数更多，每次变动的幅度更小，裁员和吸纳工人的市场略有增加，但大多数市场仍然静止不动，失业率几乎没有变化。

### 工资分布

现在我们可以回到工资分布这个问题上，概述中提到它必须与工人的行为一并确定。搜寻文献把一个工资要约的分布交给工人，然后问他应该如何应对。

而在这里，这个分布是一个输出结果：每个市场都在其中支付劳动的边际产量，而一个市场所支付的工资取决于它所抽到的需求，以及有多少工人选择了待在那里。

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

这个经济体中的工人是同质的：他们具有相同的技能、相同的偏好和相同的信息，并且可以随时自由离开。然而他们获得的工资却各不相同，第九十百分位数的工资大约比第十百分位数的工资高出一半左右。

这种离散并非市场失灵或错误。它正是使得在任何吸纳工人的市场中*到达*的价值相等、以及在任何裁减工人的市场中留下或离开的价值相等所必需的。

在其他所有地方，在任那个岗位上的工人都在赚取租金，因为新工人需要一期时间才能到达，所以职位的价值也呈现出离散分布。

```{code-cell} ipython3
held = eq['ψ']*model.y[None, :]              # weight a market by its workforce
print(f"λ* = {λ_star:.3f},  average value of a worker's position = "
      f"{(held*eq['v']).sum()/held.sum():.3f}")
print(f"share of workers whose position is worth more than λ*: "
      f"{100*held[eq['v'] > λ_star + 1e-9].sum()/held.sum():.0f}%")
```

大约十分之九的工人所持有的职位，对他们而言的价值超过 $\lambda^*$。

图中虚线是搜寻的流量价值，$(1-\beta)\lambda^*$，它正是一名搜寻者所持有的职位每期的价值。

大约三分之一的工人获得的工资低于这个值，而他们仍然留了下来。确切的比例是不稳健的。

在这个网格上，大约十一分之一的工人所获工资与流量价值相差不到千分之一，正如前面打印结果的最后一行所示，因此网格的微小变化就可能使整块工人越过那条分界线；在我们尝试的各个网格中，该比例大致介于百分之三十到百分之四十之间。

这当中没有什么令人费解之处：工人所持有的并非他目前的工资。他持有的是特定市场中的一个职位，它附带着对该市场未来的一种求偿权，以及日后离开的选择权。

(lps_efficiency)=
## 效率

卢卡斯和普雷斯科特说 $\lambda$ 传递了"额外单位供给的正确社会回报"，但他们并未证明任何福利定理。

该模型却有这样一个定理，其证明使用的正是 {doc}`lucas_prescott_investment` 中研究的1971年论文所用的方法：找到一个与该均衡相一致的计划问题。

### 剩余

令

$$
\Phi(s, n) = \sum_{m=1}^{n} R(s, m)
$$

表示边际收益函数到就业水平 $n$ 为止下方的面积。

根据 {eq}`lps_R`，这就是逆需求曲线 $D(s, \cdot)$ 到该行业产出为止下方的面积（按每家企业计算）：总消费者剩余，即1971年那篇论文的目标函数。

它只有在产品价格为给定值（$D(s, Q) = s$）时才等于产出的价值，所以我们称之为**剩余**。

### 单个市场的计划者

考虑一个管理单个市场的计划者，他可以让工人进出该市场，每释放一名工人到搜寻者队伍中，他会被记入 $\lambda$，每带入一名工人，他会被收取 $\lambda$。

如果下一期的劳动力为 $y'$，则该计划者本期雇用 $\min\{y, y'\}$ 名工人，并释放或带入差额部分，于是其价值满足

```{math}
:label: lps_planner
W(s, y) = \max_{y'} \Big\{ \Phi\big(s, \min\{y, y'\}\big) + \lambda\, (y - y')
+ \beta \int W(s', y')\, f(s', s)\, ds' \Big\} .
```

以下是**边际价值论证**。$\Phi(s, \cdot)$ 是凹函数，因为 $R$ 随 $n$ 下降，因此 $\Phi(s, \min\{y, y'\})$ 关于 $(y, y')$ 联合凹，而对 $y'$ 取最大化保持了凹性。

所以 $W$ 关于 $y$ 是凹的，计划者的选择由边际比较所主导，用边际价值 $\Delta W(s, y) = W(s, y) - W(s, y-1)$ 来表示。

把一名工人多留一期到下一期，其价值是他的工资加上 $\beta \int \Delta W(s', y') f(s', s)\, ds'$，而释放他的价值是 $\lambda$；带入一名工人需花费 $\lambda$，其价值仅仅是贴现延续值本身。

所以计划者会把工人释放到由 {eq}`lps_nbar` 以 $\Delta W$ 替代 $v$ 所定义的某个阈值为止，填充到由 {eq}`lps_ybar` 以同样方式定义的某个阈值为止，其余情况下则不改变劳动力水平。

逐案而言，边际工人的价值在计划者释放工人时为 $\lambda$，在计划者填充工人时为 $R + \lambda$（因为他省下了一次到来的费用），在二者之间则为 $R + \beta \int \Delta W f\, ds'$。

这分别对应情形A、情形B2和情形B1，所以 $\Delta W$ 满足 {eq}`lps_bellman`。根据 {prf:ref}`lps_prop_v` 中的唯一性，$\Delta W = v$，所以计划者的阈值就是均衡的阈值，其政策就是 {eq}`lps_transition`。

以下是**加总论证**。为每个市场配备这样一个计划者，使用相同的 $\lambda$，然后把它们的目标函数相加。

各项 $\lambda (y_t - y_{t+1})$ 加总起来等于 $\lambda$ 乘以经济体总劳动力的下降量，因为任何一期被释放的工人都会在下一期到达某个市场。

对任何可行的方案而言，总劳动力是恒定的，所以这些项加总为零，于是计划者们目标函数之和就等于剩余的预期现值。

均衡政策使每个计划者自身的目标函数最大化，所以它也使总和最大化。在 $\lambda = \lambda^*$ 处，该政策同时也是可行的，因为 $E(y; \lambda^*) = \mu$ 使总劳动力保持恒定。

所以，没有任何移动工人的可行方式能够产生比平稳均衡更高的剩余预期现值。

### 这个定理说了什么，又没说什么

精确地陈述，平稳均衡使**剩余的预期现值**最大化，

* 从其自身的平稳分布 $\Psi(\lambda^*)$ 出发，
* 在所有可行方案中——包括非平稳的方案以及以其他市场历史为条件的方案——当移动需要一期时间时，
* 存在一个市场的连续统，因此均衡方案使经济体的劳动力逐期保持恒定，
* 在整数网格的误差范围之内，即 $E(y; \lambda^*)$ 与 $\mu$ 只相差一步，而非精确相等。

$\lambda^*$ 是加入搜寻者队伍中的一名工人的现值；就每一期而言，经济体劳动力固定这一约束的乘子是 $(1-\beta)\lambda^*$，即上述工资图中搜寻的流量价值。

这个定理关乎总剩余，无论它归谁所有，它没有说明剩余是如何被分配的：{ref}`lps_ex_cost` 找到了一项降低剩余、却使工人境况变得更好的政策。

它也没有对各个稳态进行排序。它的比较从 $\Psi(\lambda^*)$ 出发，并计入通往任何其他配置途中发生的一切。

一项通向不同稳态的政策，可能会在那个稳态中提高剩余，同时却降低现值，因为到达那里首先要付出剩余的代价。

{ref}`lps_ex_planner` 计算了两个例子，一项对移动的小额补贴，以及一个比工人更有耐心的计划者的阈值，这两者都把每工人稳态剩余提升到高于均衡水平。

这正是增长理论中**修正黄金法则**的逻辑：一个对未来进行贴现的计划者所选择的资本量，会少于使消费最大化的稳态所对应的资本量，因为建造额外的资本首先要牺牲消费。

在这里，重新配置一名工人以牺牲他当前一期的产出为代价，而在以后才有回报，因此一个进行贴现的计划者会减少重新配置，其失业也会比使稳态剩余最大化的计划者更少。

效率是相对于移动这项需要一期时间的技术而言的，而均衡所拥有的失业量，恰好就是在未来以 $\beta$ 贴现之后，这项技术所值得承受的那个量。

## 与其他讲座的关系

在 {cite:t}`BaleyLjungqvistSargent2023` 的第一条脚注中，摩擦性失业并不存在一个"标准模型"，但存在三类可操作的模型：戴蒙德-莫滕森-皮萨里德斯传统中的匹配模型、{cite:t}`McCall1970` 的均衡版本，以及 {cite:t}`LucasPrescott1974` 传统中的搜寻-孤岛模型。

这三类模型经过标定后都能拟合劳动力市场流量方面的数据，并对诸如失业保险和裁员税之类的政策给出合理的失业反应。

本讲座属于第三类，而本系列中关于麦考尔的讲座则在局部均衡下研究第二类模型核心的工人问题。本模型为它们增添的是要约另一方的全部内容。

一名工人所获得的工资要约，是其所在市场劳动的边际产量，而该市场的劳动力水平是由其他工人决定去哪里所确定的，而到别处寻找的价值 $\lambda$，正是使整个经济体工人市场出清的那个数字。

其中一个结果是不存在保留工资，正如 $SS$ 曲线所示。另一个涉及就业与失业之间的流动率。

{doc}`lake_model` 把这两个比率都当作参数：一个求职率$0.283$（该讲座称之为 $\lambda$，不要与此处的搜寻价值混淆）和一个分离率$0.013$作为输入，失业率作为输出。

{doc}`endogenous_lake` 使求职率成为一个麦考尔问题的结果，尽管仍然是针对一个给定的工资分布。

在本讲座中，分离率是一个均衡对象，而求职率本质上通过构造就为一。

上文打印出的分离率，大约为每期已就业人数的$2.2$个百分点，它综合了需求变动使市场跌破就业阈值的频率，以及这样一个市场裁减多少工人这两个因素。

```{warning}
本模型中的求职率*本质上为一*。

一名离开者会在下一期开始时被安置到某个市场，所以一次失业的持续期为一期，该模型对失业的*持续时间*没有任何说法。

严格说来，一个接收搜寻者的市场，其需求可能下降得如此之多，以致它裁减的工人数超过到来者抵达之前它所拥有的工人数，于是部分新来者不得不再次移动；在这个标定中，即便新来者是最先被裁减的，这种情况发生的概率也不到百分之一。
```

### 孤岛模型演变成了什么

{cite:t}`AlvarezVeracierto1999` 把这个模型拿来，使之成为一个用于定量政策分析的工具，用他们自己的话说，是通过"修改卢卡斯和普雷斯科特（1974年）的孤岛模型，以纳入非定向搜寻和退出劳动力队伍的参与"。

其中四处发生了变化。搜寻变成了**非定向**的：搜寻者"均匀地分布到所有孤岛上"，所以一个孤岛下一期的劳动力就是它自身的就业加上其他每个孤岛所得到的相同流入量，无论它是否想要这些工人。

一名搜寻者可能因此落入一个本身正在裁减工人的市场，不得不再度移动，于是失业的持续时间成为一个均衡对象；在他们的自由放任基准情形下，它平均为$2.4$期。

非就业者获得了第二个选项——家庭生产，于是参与劳动力市场成为一个决策边际，该模型由此拥有三个状态而非两个。

有了家庭生产之后，搜寻者的流入量由搜寻与待在家里之间的无差异条件所确定，于是劳动力规模成为一个内生结果，这一点很重要，因为他们所研究的政策正是作用于人们是否参与求职这一决策之上的。

各孤岛以一种受到持续性生产率冲击影响的规模报酬递减的技术进行生产，而不是面对一条波动的需求曲线。并且主体之间交易一整套完备的或有债权。

他们的自由放任均衡"是有效率的（尽管存在搜寻摩擦）"，这是他们经济体中与 {ref}`lps_efficiency` 相对应的结果，他们将其作为基准，用以评估最低工资、两种工会模型、裁员税和失业保险，模型的一期取为六周。

举例来说，一项相当于平均一年工资十二个月的裁员税，会使他们的失业率从$5.3$个百分点*降至*$3.7$个百分点，并使平均失业持续期从$2.4$期*升至*$5.1$期，同时使福利下降相当于消费的$2.1$至$2.3$个百分点。

```{note}
非定向搜寻改变了 $\lambda$ 的确定方式。

在定向搜寻下，$\lambda$ 是一个*价格*：提高它，每个市场都会更早裁减工人，并较少被填满，所以 {eq}`lps_equilibrium` 像市场出清条件确定任何其他价格一样确定了它，而 {eq}`lps_bellman` 中的 $\min$ 使每个接收工人的市场对他们而言恰好值 $\lambda$。

在非定向搜寻下，一名搜寻者会落入一个随机抽取的市场，所以 $\lambda$ 需要它自身的一个一致性条件，而共同的流入量 $U$ 成为第二个未知量。

在人口固定的情况下，该模型仍然是可确定的：对于这两个未知量，存在两个条件——人口条件和 $\lambda$ 的一致性条件。

在这个标定中，一旦允许流入量为分数，均衡就存在且唯一，但一般而言，唯一性不再能由单一一条递减函数自然得出。

{ref}`lps_ex_undirected` 计算了这一情形：非定向搜寻降低了搜寻的价值，也降低了失业。
```

### 劳动力流动性的回报

卢卡斯和普雷斯科特在关于移动成本的讨论结尾处提出了一个猜想，即"这类实际或感知的换工作成本的差异，有助于解释所观察到的各职业之间以及各国之间平均失业率的差异"。

为什么平均失业率在各国之间存在差异，成为这一类模型的核心问题；举例来说，{cite:t}`LjungqvistSargent1998` 把欧洲失业的上升归因于慷慨的失业救济与更加动荡的经济环境之间的相互作用。

{cite:t}`BaleyLjungqvistSargent2025` 把一名工人在其所去之处相比于所在之处能多生产多少——即**劳动力流动性的回报**，称作"在所有带有摩擦性劳动力市场的现代宏观经济模型中都起着重要中介作用的力量"。

{cite:t}`BaleyLjungqvistSargent2023` 表明，正是同一个量，决定了一项裁员税对失业会产生什么影响，也决定了离职时的技能损失是否能够逆转动荡对失业的影响。这是冲击本身的性质与传导该冲击的技术二者共同作用的结果。

{cite:t}`BaleyLjungqvistSargent2025` 发现，通过新古典生产函数传导冲击、标定到企业规模动态的模型，所得出的劳动力流动性回报对参数的扰动是稳健的，而标定到劳动力市场统计数据的人均生产率过程则可能是脆弱的。

他们所研究的模型之一是 {cite:t}`AlvarezVeracierto2001` 的模型，该模型把一个生产率过程标定到戴维斯和哈尔蒂万格关于岗位创造与岗位破坏的企业数据上。

那篇论文是一个与 {cite:t}`AlvarezVeracierto1999` 不同的模型：各生产单位在一个位置随机变动的共同劳动力市场上招聘，工资合同是刚性的，缺乏保险市场的风险厌恶型主体进行自我保险，并选择搜寻的努力程度。

一个生产单位用资本 $k$ 和劳动 $n$ 生产 $z\, k^{\theta_k} n^{\theta_n}$，其中生产率为 $z$，且 $\theta_k + \theta_n < 1$。

{cite:t}`BaleyLjungqvistSargent2025` 对该模型进行了简化，去掉了创建一个生产单位的成本以及一项小额裁员税，使得每个生产单位都在现货市场上租用资本并雇用劳动，并且全都使边际产品相等，这就得到

```{math}
:label: lps_av_ratio
\frac{n_{\text{high}}}{n_{\text{low}}}
= \left(\frac{z_{\text{high}}}{z_{\text{low}}}\right)^{1/(1-\theta_k-\theta_n)} ,
```

因此，一个冲击被一个随着规模报酬趋近于一而发散的指数放大了。

在阿尔瓦雷斯和维拉谢尔托的参数之下，他们发现一个低生产率的生产单位所雇用的工人仅为高生产率生产单位的$3.81$个百分点，比率大约为$26$，而高生产率生产单位占据了全部就业的$96$个百分点以上：约$4.35$的指数把一个$2.12$的生产率比率转变为约$26$的就业比率。

那是一种无摩擦的配置，因为每个生产单位都在同一个现货市场上招聘。在本模型中，移动需要一期时间，{ref}`lps_ex_two_states` 探讨了像$26$这样的一个比率在这种摩擦之下能够保留多少，同时保持与阿尔瓦雷斯和维拉谢尔托相同的技术与冲击。

## 练习

```{exercise}
:label: lps_ex_persistence

{cite:t}`LucasPrescott1974` 从他们自己的数值例子中报告了两个比较静态结果：

> *总体而言，我们发现无论需求持续性是低还是高，失业率都较低。在持续性较低的情形中，重新配置工人几乎没有什么收益；而在持续性较高的情形中，重新配置发生得很少。正如预期，在持续性固定不变的情况下，需求的变动性越大，失业水平就越高。*

在保持 $\mu = 50$ 不变的情况下，在这个标定中检验这两个结论。

1. 在保持 $\log s$ 的*平稳*标准差固定为$0.26$的情况下，对持续性 $\rho \in \{0.6, 0.8, 0.9, 0.95, 0.99\}$ 计算均衡失业率；并在保持 $\rho = 0.9$ 的情况下，对平稳标准差 $\{0.10, 0.18, 0.26, 0.34, 0.45\}$ 计算均衡失业率。解释你所发现的形状。
1. 在需求相互独立的情形下，即 $\rho = 0$，`equilibrium_λ` 会报错停止。证明此时失业恰好为零，且均衡的 $\lambda$ 并不唯一。这为何不与论文的命题5、也不与其关于失业在每一期都为正的论断相矛盾？
1. 在 $\rho = 0.3$ 处重复第2部分，先用七个需求状态，再用25个需求状态。这一比较对第1部分的低持续性端说明了什么？
```

```{solution-start} lps_ex_persistence
:class: dropdown
```

在 $\rho$ 变化时保持平稳标准差不变，正是 `create_model` 的参数 `sd` 已经做到的事，所以每次实验都只是一次函数调用。

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

变动性的作用正如预期：移动幅度更大的需求会移动更多的工人，失业率随之稳步上升。

持续性则不然：失业率在 $\rho$ 约为$0.8$至$0.9$处达到峰值，然后在 $\rho = 0.99$ 时又回落到峰值的大约四分之一。这正是卢卡斯和普雷斯科特所描述的形状，而两端之所以较低，原因并不相同。

当需求在各期之间几乎独立时，今天运气不好的市场预期明天会回归平均水平，所以留下来的价值几乎不依赖于今天的需求，到达阈值 $\bar y(s)$ 几乎不随 $s$ 变化，因此几乎从没有工人被裁减；第2部分把这推向了极限。

当需求几乎是永久性的时，它从一期到下一期几乎没有变化：在七状态链上它很少变化，一变就是一整档；在更精细的链上它变化更频繁，但每次幅度更小。无论哪种情形，在任何一期都几乎没有工人有理由移动，这正是论文所说的"重新配置发生得很少"。

失业需要既*值得*应对、又*频繁*出现的需求变动，而只有中等程度的持续性才能同时提供这两者。

对于第2部分，`equilibrium_λ` 之所以报错停止，是因为不变分布不唯一，这正是 {ref}`lps_regions` 中说明的第三种情形。所以我们转而寻找这样的 $\lambda$ 值：在这些值处，一个持有 $\mu$ 名工人的市场永不移动，即 $\max_s \bar y(s) \le \mu \le \min_s \bar n(s)$。

两个阈值都随 $\lambda$ 上升而下降，因此该集合的每一端都可以用二分法找到。

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

需求相互独立时，$P$ 的每一行都相同，因此延续值 $\beta \sum_{s'} v(s', y) P(s, s')$ 不依赖于 $s$，$\bar y$ 也是如此。

由于在每个状态下都有 $\bar y(s) \le \bar n(s)$，这个共同的 $\bar y$ 低于每一个就业阈值。所以任何介于 $\bar y$ 和 $\min_s \bar n(s)$ 之间的劳动力水平在每个需求状态下都原地不动，各市场最终漂移到这个区间内并停止；长期而言没有人移动，失业恰好为零。

该区间内的每一个劳动力水平都是吸收态，所以不变分布不唯一，打印出的区间内的每一个 $\lambda$ 都支撑着一个每个市场都持有 $\mu$ 名工人的均衡。

该区间的宽度是离散化之链的一个性质：随着状态数的增加，它会收窄，因为最差的需求状态被推得更远，其就业阈值向共同的 $\bar y$ 靠近。然而失业始终保持为零，因为每一个 $\bar n(s)$ 都至少等于 $\bar y$。

这一切都不与论文矛盾，起作用的关键区别在于 $x \equiv 1$。命题5中 $\lambda$ 的唯一性，以及失业在每一期都为正的论断，二者都依赖于命题3中唯一的平稳密度，而该密度是通过随机到来使某些市场过度填满而得出的；在确定性到来的情形下，这一论证就不再适用了。

需求相互独立也违反了论文所维持的 $F(s', s)$ 关于 $s$ 的严格单调性，而论文脚注7中的例子要求AR(1)系数严格为正。

对于第3部分，在 $\rho = 0.3$ 处进行相同的检验，在两种网格下给出了不同的答案。

```{code-cell} ipython3
for n_s in (7, 25):
    m = create_model(n_s=n_s, ρ=0.3)
    rest = rest_interval(m, μ)
    if rest is not None:
        print(f"{n_s:>2} states: at rest for λ from {rest[0]:.3f} to {rest[1]:.3f}")
    else:
        print(f"{n_s:>2} states: unique λ*, unemployment rate {100*u_rate(m, μ):.3f}%")
```

在七个状态下，持续性$0.3$不足以把到达阈值推过最低的就业阈值，经济体处于静止状态，就像 $\rho = 0$ 时一样。在25个状态下，极端需求状态被推得更远，均衡是唯一的，失业为正但非常微小。

所以在 $\rho = 0.3$ 处，无论用哪种网格，失业都很低，这正如论文所说，但它是否恰好为零，则是离散化之链的一个性质。

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

鲁文霍斯特之链恰好具有目标标准差$0.26$和目标自相关性。塔琛（Tauchen）七状态链在 $\rho = 0.9$ 时波动性过大，这使失业率上升了大约四分之一。

在 $\rho = 0.99$ 时，它既波动性过大，又过于持续：其需求状态几乎从不变化，所以市场几乎从不重新配置，失业几乎消失。

用七个状态时，这会使第1部分的右端看起来像左端；而用25个状态时，两种方法几乎一致。

```{solution-end}
```

```{exercise}
:label: lps_ex_cost

在他们的结论部分，卢卡斯和普雷斯科特考虑了离开一个市场进行搜寻需支付一笔总额成本 $c$ 的情形，于是 {eq}`lps_nbar` 的右边从 $\lambda$ 变成了 $\lambda - c$：一名离开者所获得的是净去其移动成本之后的搜寻价值。

到来条件 {eq}`lps_ybar` 保持不变，因为它为已经处于搜寻中的工人所面对的市场定价。

他们说这项成本"会抬高图3中 $y = \bar n(s, \lambda)$ 这条曲线，并使图4中的'需求曲线'向下移动。其结果是失业下降，而均衡的工资现值 $\lambda$ 也下降"，并指出这

> *表明，更低的平均失业率一般而言并不意味着工人获得了更高的福利。*

`market_outcomes` 和 `equilibrium_λ` 已经将该成本作为参数 `c`。

对 $c \in \{0, 0.2, 0.5, 1, 2\}$，报告均衡 $\lambda^*$ 和 $\lambda^* - c$、失业率、每市场剩余、每市场工资总额、租金（剩余减去工资总额），以及每个工人每期的工人收入——即工资总额净去工人为移动所支付的费用，再除以劳动力规模。

1. $\lambda^*$ 是否如论文所说那样下降？如果不是，那么下降的是什么？
1. 更低的失业是否意味着境况更好？将工人所获得的与经济体所损失的，以每工人现值的形式进行比较。
1. 谁为此买单？分别就两种情形各回答一次：一次将 $c$ 视为一项资源，在移动过程中被消耗掉；另一次将其视为一项税收，其收入以总额方式返还给工人。
1. {cite:t}`BaleyLjungqvistSargent2023` 使用能够关闭全部内生性分离的最小裁员税，作为一个模型中劳动力流动性回报的诊断指标。以平均工资为单位，用7个、15个和25个需求状态分别计算能够做到这一点的最小 $c$。这个值为何随状态数而变化？
```

```{solution-start} lps_ex_cost
:class: dropdown
```

在进行扫描之前，先看看这项成本在自由放任的搜寻价值处对劳动力需求做了什么。

```{code-cell} ipython3
λ_0 = equilibrium_λ(model, μ)
for c in (0.0, 2.0):
    n_bar_c, y_bar_c = thresholds(model, worker_value(model, λ_0, c)[0], λ_0, c)
    print(f"c = {c}: workforce demand {workforce_demand(model, λ_0, c):5.1f},  "
          f"n̄(s) = {n_bar_c},  ȳ(s) = {y_bar_c}")
```

该成本提高了劳动力需求，所以新的 $\lambda^*$ 高于旧值，我们从那里开始每一次二分法。这也使它远离低 $\lambda$ 值的区域，因为在有成本的情况下，不变分布在那里不唯一，二分法会在那里停止。

在一个平稳分布中，每个失业的工人本期都恰好离开了一次他所在的市场，所以工人支付的移动成本总额为 $c$ 乘以每市场的失业人数。

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

剩余、工资和租金都是每市场的流量，而收入是每工人的流量。为了比较工人所获得的与经济体所损失的，把两者都换算为每工人的现值，即用流量除以 $1 - \beta$。

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

对于第1部分，卢卡斯和普雷斯科特说这项成本使他们图4中的需求曲线向下移动，并降低 $\lambda$，但在这里它的作用恰好相反。

在自由放任的 $\lambda^*$ 处，这项成本使劳动力需求提高了约一半。它在每一个有人工作的需求状态下都大幅提高了就业阈值 $\bar n(s)$，因为一个本会离开的工人现在还要放弃 $c$；同时它使到达阈值略有下降。

这正是论文自身的第一步，"抬高曲线 $y = \bar n(s, \lambda)$"，而在图4中把 $\lambda$ 放在纵轴上时，这会使需求曲线向上移动，而非向下。

由于所需求的工人数超过经济体所拥有的数量，$\lambda^*$ 必须上升，而它确实上升了。下降的是 $\lambda^* - c$，即离开这一选择权扣除其成本后的价值，这正是一名决定是否留下的工人所比较的数值，论文的论断对这个数字成立，而不是对 $\lambda^*$ 本身成立。

对于第2部分，成本为二时把测得的失业削减了近十分之九。如果以扣除移动成本之后的收入来衡量，从 $c = 0.5$ 起，工人明显获益。

对于较小的成本，工人为移动所支付的费用几乎抵消了额外的工资，所以净收益接近零，而在这个网格上，$c = 0.2$ 时净收益略为负值。

经济体的损失超过工人的所得：在 $c = 2$ 时，剩余的下降幅度大约是工人收入上升幅度的两到两倍半。这些都是稳态之间的比较，没有计入从一个稳态过渡到另一个稳态的过程。

所以，更低的失业并非一个福利指标，但这个教训所呈现的形式与论文的并不相同：卢卡斯和普雷斯科特笔下的工人境况变差是因为 $\lambda$ 下降，而我们这里的工人境况变好，整个经济体却变差。

这项成本所消除的失业是*生产性*的。那些工人在任何有意义的层面上都不是闲置的；他们正处于从边际产量已下降的市场转移到边际产量较高的市场的途中，而这项成本使他们留在原地。

对于第3部分，工资总额略有上升，而租金——剩余减去工资总额——下降了约五分之一。

剩余就是 $R$ 之下的面积，所以租金就是企业的利润，加上——当产品需求向下倾斜时——该市场商品购买者的消费者剩余，而承担这项成本的正是他们。

账目如何平衡，取决于 $c$ 究竟是什么。如果它是一项真实资源，上面打印出的移动成本也会被消耗掉，叠加在剩余损失之上，由工人承担。

如果它是一项以总额方式返还给工人的税收，那么除了剩余损失之外没有任何东西被消耗，这笔返还每期把那些相同的移动成本还给工人，而不改变任何人的决策，因此工人获益更多。

这两种情形在剩余损失意味着什么这一点上也有所不同。{ref}`lps_ex_planner` 表明，在资源成本的情形下，均衡就是针对那项代价更高的移动技术而言计划者的配置，所以较低的剩余是更差技术所付出的代价，而非一种低效率；而在返还式税收的情形下，技术本身没有改变，均衡不再是针对该技术的计划者配置，所以较低的剩余是一种效率损失。

对于第4部分，回想一下 {ref}`lps_regions` 中的说明。分离恰好在存在这样一个 $\lambda$ 时停止：在该 $\lambda$ 下，一个持有 $\mu$ 名工人的市场永不移动，没有任何状态的到达阈值高于 $\mu$，也没有任何状态的就业阈值低于 $\mu$。

一项成本会提高就业阈值，所以我们寻找使这样一个 $\lambda$ 存在的最小成本。

一旦分离停止，每个市场就永远持有 $\mu$ 名工人，于是处于需求状态 $s$ 的工人持有 $v_s$，其中 $v = (I - \beta P)^{-1} R(\cdot, \mu)$，且当对每个 $s$ 都有 $\beta (Pv)_s \le \lambda \le v_s + c$ 时，没有人离开或到来。

这给出了一个可以与数值计算相对照的公式，$c_{\min} \approx \beta \max_s (Pv)_s - \min_s v_s$，它忽略了整数网格。

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

这里作为单位所用的工资，是当每个市场都持有 $\mu$ 名工人时的平均工资，这正是分离停止之后经济体的样貌；它与自由放任下的平均工资相差不到半个百分点。

使所有分离都停止的那项成本随着状态数的增加而增大，公式说明了原因。它由一个市场可能处于的最好与最差需求状态之间的差距所决定，而鲁文霍斯特之链把它的极端状态放在 $\log s$ 中的 $\pm 0.26 \sqrt{n_s - 1}$ 处，所以随着状态增多它们会彼此分得更开。

$\log s$ 本身是一个支撑无界的AR(1)过程，所以对于具有连续需求的模型，没有任何有限的成本能够使所有分离都停止：此处计算出的数值衡量的是离散化需求过程的支撑范围。

对于 {ref}`lps_ex_two_states` 中那个对称的两状态链——需求以概率 $\rho$ 保持不变，且 $R(s,n) = s\, g(n)$——该公式有一个闭合形式，

$$
\frac{c_{\min}}{\bar w}
= -1 + \frac{s_{\text{high}} - s_{\text{low}}}{s_{\text{high}} + s_{\text{low}}}
\cdot \frac{1 + \beta(2\rho - 1)}{1 - \beta(2\rho - 1)} ,
$$

在该情形下约为$1.05$期的工资，它取决于需求过程和 $\beta$，但不依赖于 $g$ 的形状：在本模型中，该诊断指标衡量的是需求的分散程度和持续性，而不是 $R$ 的曲率。

该标定并未说明一期有多长；在 $\beta = 0.9$ 的情形下，自然可以把它理解为一年，那样一来每次失业都会持续一年。作为对照，在 {cite:t}`BaleyLjungqvistSargent2023` 用作其基准的永奎斯特-萨金特匹配模型中，相当于每工人一年产出$163$至$184$个百分点的裁员税足以关闭全部内生性分离，而该模型一期的长度为半个季度。

贝利、永奎斯特和萨金特想要表达的观点是，这个数字必须*很大*才具有可信性，因为在裁员成本存在巨大差异的各个经济体中，都能观察到大量的重新配置活动。

```{solution-end}
```

```{exercise}
:label: lps_ex_two_states

{cite:t}`LucasPrescott1974` 附录中的数值例子只使用了两个需求状态，具体为

$$
P = \begin{bmatrix} \rho & 1 - \rho \\ 1 - \rho & \rho \end{bmatrix} ,
\qquad \rho = 0.9,
\qquad s \in \{1, 2\},
\qquad \beta = 0.9 .
$$

这里 $\rho$ 是需求保持不变的概率，所以 $s$ 的自相关性为 $2\rho - 1 = 0.8$；而在七状态链中，$\rho$ 本身就是自相关性。

构建这个经济体，保持 $R(s,n) = s\,(1 - n/N)^{+}$，其中 $N = 200$，并在需要均衡的地方取 $\mu = 50$。

1. 对一个 $\lambda$ 网格，绘制劳动力需求和平均失业对 $\lambda$ 的图形，如论文图5的中图和下图所示。
1. 在 $\lambda = 6$ 时，证明不变分布把全部概率质量集中在*两个*劳动力数值上，并指出这两个值分别是什么。
1. 利用这一事实，推导一个用 $\bar n$、$\bar y$ 和 $\rho$ 表示每市场平均失业的公式，用网格上每一个 $\lambda$ 处的计算值进行验证，并用它解释第1部分图形的形状。
1. 证明失业*率*仅取决于 $\rho$ 以及 $r = \bar y(s_{\text{high}})/\bar n(s_{\text{low}})$，并且它永远不会超过 $1 - \rho$。
1. 现在把 $R$ 替换为某种规模报酬递减技术的边际产品：产出为 $s\, n^{\alpha}$，其中 $0 < \alpha < 1$，于是 $R(s, n) = s\,\alpha\, n^{\alpha-1}$。对 $\alpha \in \{0.4, 0.6, 0.75\}$ 求解；并对一个类似阿尔瓦雷斯和维拉谢尔托模型的经济体求解，取 $\alpha = \theta_n/(1-\theta_k)$ 和 $s_{\text{high}} = 2.12^{1/(1-\theta_k)}$，在他们的参数 $\theta_k = 0.19$ 和 $\theta_n = 0.58$ 之下，这正是在资本（每期租用）已被最优选定之后的那项技术。报告两个需求状态之间平均就业的比率、其无摩擦对应值 {eq}`lps_av_ratio`，以及失业率。证明平均就业的比率为 $\rho r + 1 - \rho$，用它来表示失业率，并说明随着劳动力流动性回报的提高，失业会发生什么变化。
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

论文将 $\lambda$ 置于纵轴上。

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

右图具有论文图5下方两幅图的形状：在中等的 $\lambda$ 值处，平均失业最大。

对于第2部分，考察 $\lambda = 6$ 时不变分布的支撑。

```{code-cell} ipython3
r_six = market_outcomes(two, 6.0)
support = np.where(r_six['ψ'].sum(axis=0) > 1e-12)[0]
print("workforce values with positive probability:", support)
print("n̄(s) =", r_six['n_bar'], "   ȳ(s) =", r_six['y_bar'])
```

这两个值分别是 $\bar n(s_{\text{low}})$ 和 $\bar y(s_{\text{high}})$。要理解原因，可以遵循 {eq}`lps_transition`。

在低状态下，劳动力被限制到 $[\bar y(s_{\text{low}}), \bar n(s_{\text{low}})]$ 之中，在高状态下则被限制到 $[\bar y(s_{\text{high}}), \bar n(s_{\text{high}})]$ 之中，而在这个 $\lambda$ 值处，第二个区间完全位于第一个区间之上。

所以处于低状态的一期会把任何劳动力水平带到 $\bar n(s_{\text{low}})$，处于高状态的一期会把它带到 $\bar y(s_{\text{high}})$，于是该市场就在这两个值之间来回往复。在每一个使 $\bar n(s_{\text{low}}) < \bar y(s_{\text{high}})$ 成立的 $\lambda$ 处都是如此，而这覆盖了整个网格。

对于第3部分，失业只在经济下行的第一期为正：该市场以 $\bar y(s_{\text{high}})$ 名工人进入该期，此时需求刚好下降，就业为 $\bar n(s_{\text{low}})$。

这一事件就是"上一期为高状态，而现在为低状态"，其平稳概率为 $\tfrac12 (1-\rho)$，所以

$$
E[u] = \tfrac12 (1-\rho)\,
\big[\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})\big] .
$$

我们对网格上的每一点都检验该公式，$\rho$ 从转移矩阵中读取。

```{code-cell} ipython3
ρ_2 = two.P[0, 0]
gap2 = np.array([r['y_bar'][1] - r['n_bar'][0] for r in res2])
print(f"largest difference between formula and computation: "
      f"{np.abs(0.5*(1 - ρ_2)*gap2 - Eu2).max():.1e}")

n_lo = np.array([r['n_bar'][0] for r in res2])
print(f"unemployment per market peaks at λ = {grid2[Eu2.argmax()]:.1f}; "
      f"the low-demand market is closed for λ ≥ {grid2[n_lo == 0].min():.1f}")
```

这个公式也解释了那个驼峰：每市场的平均失业是两个阈值之间的差距，再乘以跨越该差距的概率。

在高 $\lambda$ 处，低需求市场已关闭，$\bar n(s_{\text{low}}) = 0$，此时差距就是 $\bar y(s_{\text{high}})$，它随 $\lambda$ 下降而上升。

一旦 $\lambda$ 低到足以使低需求市场保留一些工人，$\bar n(s_{\text{low}})$ 随 $\lambda$ 下降而上升的速度就会快于 $\bar y(s_{\text{high}})$，差距因而收窄。所以每市场的失业人数在低需求市场停止关闭的那一点达到峰值，正如打印结果的第二行所示。

对于第4部分，劳动力在一半的期数内为 $\bar n(s_{\text{low}})$，在另一半的期数内为 $\bar y(s_{\text{high}})$，所以 $E[y] = \tfrac12 [\bar n(s_{\text{low}}) + \bar y(s_{\text{high}})]$，失业率为

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,
\frac{\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})}{\bar y(s_{\text{high}}) + \bar n(s_{\text{low}})}
= (1-\rho)\, \frac{r - 1}{r + 1} ,
\qquad r = \frac{\bar y(s_{\text{high}})}{\bar n(s_{\text{low}})} .
$$

它随 $r$ 上升而朝着 $1 - \rho$ 增大，当低需求市场关闭时达到该值。

对于第5部分，低状态下的平均就业为 $\bar n(s_{\text{low}})$，而高状态下的平均就业以概率 $\rho$ 为 $\bar y(s_{\text{high}})$，以概率 $1 - \rho$ 为 $\bar n(s_{\text{low}})$（即经济上行的第一期），所以平均就业的比率为

$$
\mathcal{E} = \frac{E[n \mid s_{\text{high}}]}{E[n \mid s_{\text{low}}]} = \rho\, r + 1 - \rho .
$$

把 $r = (\mathcal{E} - 1 + \rho)/\rho$ 代入第4部分的失业率中，得到

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,\frac{\mathcal{E} - 1}{\mathcal{E} - 1 + 2\rho} ,
$$

它随 $\mathcal{E}$ 上升而增大。

只有 `R` 的构造发生了变化：在整数网格上，第一名工人的边际产品 $s\,\alpha$ 是有限的，一个空市场不支付任何工资。工人在边际上现在远处变得廉价，所以我们使用一个更大的网格，并围绕使一个持有 $\mu$ 名工人的市场恰好停止吸引到来者的那个搜寻价值来确定 $\lambda^*$ 的区间，按照 {ref}`lps_ex_cost` 的记号，该值为 $\beta \max_s (Pv)_s$。

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

恒等式 $\mathcal{E} = \rho r + 1 - \rho$ 以及失业率公式在每个经济体中都成立，正如各列所示。

无摩擦那一列就是取 $\theta_k = 0$ 和 $\theta_n = \alpha$ 时的 {eq}`lps_av_ratio`：使两个状态下的边际产品 $s\,\alpha\, n^{\alpha-1}$ 相等，得到 $n_{\text{high}}/n_{\text{low}} = (s_{\text{high}}/s_{\text{low}})^{1/(1-\alpha)}$。

那个类似阿尔瓦雷斯和维拉谢尔托模型的经济体，其无摩擦比率与他们带有资本的技术相同，约为$26$，因为把资本优化掉之后，$z\, k^{\theta_k} n^{\theta_n}$ 就变成了 $z^{1/(1-\theta_k)} n^{\theta_n/(1-\theta_k)}$ 的一个倍数。那种仅含劳动的形式才是此处恰当的比较对象，因为在本模型中摩擦仅作用于劳动。

对于线性函数而言，无摩擦基准是一个角点解，而且差点就不是了。令 $2(1 - n_{\text{high}}/N) = 1 - n_{\text{low}}/N$ 且 $n_{\text{high}} + n_{\text{low}} = 2\mu$ 相等，得到 $n_{\text{low}} = (4\mu - N)/3$，所以每当 $\mu \le N/4$ 时低需求市场就会关闭，而这恰好就是本讲座的 $\mu = 50$。

```{code-cell} ipython3
N = 200                                    # the capacity of the two-state economy
for μ_f in (50, 55, 60):
    n_low = max((4*μ_f - N)/3, 0.0)
    ratio = (2*μ_f - n_low)/n_low if n_low > 0 else np.inf
    print(f"μ = {μ_f}: frictionless n_low = {n_low:.1f}, ratio {ratio:.1f}")
```

所以在 $\mu = 50$ 处，无摩擦比率为无穷大，在略高于该值处仍然很大。

在每个经济体中，该模型的反应都远小于其无摩擦基准，这正是一期移动摩擦所应有的表现。一个无摩擦经济体会不断移动工人，直到各市场之间的边际产品相等为止；而在这里，工人只移动到使到达的价值等于 $\lambda$ 为止，到来在 $\bar y(s_{\text{high}})$ 处停止，因为需求可能会回落，而一次经济上行需要一期时间才能被完全满足。

把同类情形相比较，就是最后一行：阿尔瓦雷斯和维拉谢尔托的生产单位在其技术与冲击之下得到约$26$的就业比率，而在本模型的一期移动摩擦之下，这一比率约为$5.6$，尽管冲击的持续性仍与他们的不同。

在 $s_{\text{high}} = 2$ 的各行之间，在摩擦保持不变的情况下，就业反应随 $\alpha$ 上升而增大，失业率也随之上升。这并非巧合，而且这是本讲座自身的发现，而非贝利、永奎斯特和萨金特的发现。

本模型中的失业就是工人花在移动上的时间，而移动所能获得的收益越大，移动的人就越多：在摩擦保持不变的情况下，劳动力流动性回报较高的经济体会表现出更多的重新配置，并记录下与之相伴的失业。

```{solution-end}
```

```{exercise}
:label: lps_ex_undirected

本练习探讨模型中的定向搜寻究竟在做什么。

{cite:t}`AlvarezVeracierto1999` 把它替换为**非定向**搜寻：一名搜寻者无法选择去哪里，每个市场都获得相同的流入量 $U$，它就是经济体每市场的失业人数。

1. 证明在非定向搜寻下，{eq}`lps_bellman` 中的 $\min$ 消失了，并说明它原本在实施哪一个经济学条件。
1. 写出一个市场劳动力的转移规律，并证明在任何平稳分布下，流入量 $U$ 都等于平均流出量。在定向搜寻下，是否也成立同样的恒等式？
1. 在定向搜寻下，{eq}`lps_equilibrium` 确定了 $\lambda$。在非定向搜寻下，$\lambda$ 必须满足什么条件？对一个每市场劳动力固定为 $\mu$ 的均衡，数一数未知量和条件的数目。
1. 计算本讲座经济体在非定向情形下的均衡，$\mu/N = 1/4$。流入容量为 $N = 200$ 的各市场的流入量取整数过于粗略，所以我们以十分之一工人为单位计数：使用 $N = 2000$ 以及一个上限为 $2600$ 的网格，这就是本讲座经济体在十倍精细的劳动力网格上的版本。对流入量 $U = 1, 3, 5, 7, 9$ 和$10$，找出满足第3部分一致性条件的 $\lambda$ 值，以及每个值所服务的平均劳动力，并将平均劳动力对 $U$ 作图。一个具有给定 $\mu$ 的经济体有多少个均衡？在同一网格上，将 $\lambda$ 和失业率与定向搜寻在 $\mu/N = 1/4$ 处的结果进行比较。
1. 解释为什么非定向搜寻在这里降低了搜寻的价值，以及为什么它也降低了失业。
```

```{solution-start} lps_ex_undirected
:class: dropdown
```

对于第1部分，{eq}`lps_bellman` 中的 $\min$ 是由搜寻者之间的竞争所施加的限制，即一个市场的延续值不能超过 $\lambda$。一旦取消他们选择去向的能力，就没有什么能够实施这一限制了。

一名留下来的工人只是简单地获得工资以及他自己所在市场恰好带来的延续值，

$$
v(s, y) = \max\left\{\lambda,\; R(s, y)
 + \beta \int v\big(s', y + U\big) f(s', s)\, ds'\right\} ,
$$

该延续值可能高于或低于 $\lambda$，因为没有人能够据此采取行动。

对于第2部分，就业仍然是 $n(s,y) = \min\{\bar n(s), y\}$，未被雇用者离开，流入量处处相同，所以

$$
y' = n(s, y) + U .
$$

在一个平稳分布下 $E[y'] = E[y]$，所以对各市场取平均，得到

$$
U = E[y] - E[n(s,y)] = E[\,y - n(s,y)\,] :
$$

流入量等于平均流出量。

这是一个会计恒等式，在定向搜寻下也成立：在 {eq}`lps_transition` 的平稳分布处，到达被填充市场的工人等于离开被裁减市场的工人，在每一个 $\lambda$ 值处都成立，而不仅仅是在 $\lambda^*$ 处。

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

所以这个恒等式不可能是区分这两个模型的关键；真正的区别在于 $\lambda$ 是如何确定的。

对于第3部分，在非定向搜寻下，一名搜寻者会落入从不变分布中随机抽取的一个市场，所以搜寻的价值必须满足

$$
\lambda = \beta \sum_{s,y} \psi(s, y)\, v(s, y) ,
$$

即落入一个随机抽取的市场的贴现价值，是一个以市场为权重的平均值。

因此，一个人口固定的均衡有两个未知量，$\lambda$ 和 $U$，以及两个条件，即 $\mu = \sum_{s,y} \psi(s,y)\, y$ 和上面这个条件。

在定向搜寻下，$\min$ 使每个接收工人的市场对他们而言恰好值 $\lambda$，所以 $\lambda$ 不需要单独的一致性条件，$\mu$ 单独就通过一条向下倾斜的函数确定了它。而在这里，$\lambda$ 需要它自身的一个条件，没有任何机制使这一条件关于 $\lambda$ 单调。

对于第4部分，一个市场的就业阈值是使留下仍值 $\lambda$ 的最大劳动力水平，前提是下一期的劳动力将为 $n(s, y) + U$。

不变分布是唯一的，因为一个经过最差需求状态的市场，无论从哪里出发，最终都会持有 $\bar n(s_1) + U$ 名工人，所以我们通过迭代来求解它。

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

事实证明，隐含的搜寻价值与 $\lambda$ 之间的差距在两个根之间为正，在两根之外为负，所以对每个流入量，我们在一个使其为正的值的两侧各寻找一个根。

在这个网格上，定向搜寻的 $\lambda^*$ 在每一个存在根的流入量处都是这样一个值；若某次区间选取失败，来自SciPy的 `brentq` 会报错停止。

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

这些流入量中的每一个都有两个解，每个解各自服务其自身的人口；而在 $U = 10$ 时则没有解。

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

在给定的流入量下，这两个解属于*不同的*人口规模：一个拥挤的经济体，其搜寻价值较低；一个稀疏的经济体，其搜寻价值较高。沿着每一条分支，人口规模随流入量单调变化，两条分支在 $U = 9$ 和 $U = 10$ 之间相遇并折返，它们各自所覆盖的 $\mu$ 范围互不重叠。

所以在这个标定中，每个人口规模恰好对应一个非定向均衡 $(\lambda, U)$，而它所需的流入量随 $\mu$ 上升而增大，直到折返点，之后又下降。

这个折返点也解释了为何 $N = 200$ 的网格不够用：最大的均衡流入量略低于容量为200的每个市场一名工人，所以即便是那个网格上的 $U = 1$ 也已经超出了它。本讲座的经济体是最接近 $\mu = 50$ 的那一点。

```{code-cell} ipython3
U_eq, k_eq = min(((U, k) for U in locus for k in (0, 1)),
                 key=lambda Uk: abs(locus[Uk[0]][Uk[1]][1] - N_fine//4))
λ_und, y_und = locus[U_eq][k_eq]
print(f"undirected: U = {U_eq}, λ = {λ_und:.3f}, workforce {scale*y_und:.2f}, "
      f"unemployment rate {100*U_eq/y_und:.2f}%")
print(f"directed:   λ* = {λ_dir:.3f}, workforce {scale*r_dir['workforce']:.2f}, "
      f"unemployment rate {100*r_dir['unemployment']/r_dir['workforce']:.2f}%")
```

它位于拥挤的那条分支上，靠近折返点，非定向搜寻给出的搜寻价值和失业率都比定向搜寻更低。

对于第5部分，比较各个就业阈值。

```{code-cell} ipython3
n_bar_und = undirected_market(m_fine, λ_und, U_eq)[2]
n_bar_low, _ = thresholds(m_fine, worker_value(m_fine, λ_und)[0], λ_und)
print("n̄(s), undirected:                       ", n_bar_und)
print("n̄(s), directed, at its own λ*:          ", r_dir['n_bar'])
print("n̄(s), directed, at the undirected λ:    ", n_bar_low)
```

非定向搜寻降低了搜寻的价值，因为搜寻者不再能够避开不想要他们的市场：在定向搜寻下，每位搜寻者都前往到达价值恰好等于 $\lambda$ 的市场，而在这里，许多人会落入正在裁减工人、或即将裁减工人的市场。

由于搜寻价值较低，那些所在市场需求已经下降的工人离开得更慢：在每个需求状态下，就业阈值都更高，而除了最低的需求状态之外，大部分差距正是定向搜寻在相同的较低 $\lambda$ 下本会给出的结果，正如最后一行所示。所以各市场保留了更多工人，经济体以更小的搜寻者流量——即其失业——维持了它的人口规模。

注意，对一个给定的流入量而言，从感知的 $\lambda$ 到隐含值 $\beta \sum \psi v$ 的映射，是一个真正从感知值映射到搜寻实际价值的映射，这与 {ref}`lps_learning` 中的试探过程不同。

```{solution-end}
```

```{exercise}
:label: lps_ex_phillips

该模型没有总体冲击，但卢卡斯和普雷斯科特在其结论部分的开头却论证说，该模型的机制"与关于所观察到的菲利普斯曲线那种以预期为基础的常见解释是一致的"：

> *总需求的一次未被预期到的变化（即 $E(s'|s)$ 的变化），会使失业与工资变动朝相反的方向移动。*

在本讲座的均衡处检验这一点，从 $\lambda^*$ 处的不变分布出发，该分布描述的是上一期的经济状况。

在本期，一批随机选取的市场抽到的需求状态比每个人都知道的运动规律本应给出的状态低一档，其余市场照常抽取；选取的比例使 $\log s$ 的均值下降 $\Delta$。

没有人预见到这一点，所以阈值 $\bar n(s)$ 和 $\bar y(s)$ 保持不变，$\lambda$ 也保持在 $\lambda^*$。

1. 对 $\Delta = 0.01, 0.02, 0.04$，以及相同幅度的 $\log s$ 均值上升，计算本期的失业率和平均工资。
1. 将失业率对平均工资的百分比变化作图。解释这一形状。
1. 卢卡斯和普雷斯科特补充说，如果这类变动"是反复出现的事件，正如现实中那样，那么这一事实就会被纳入工人所面对的最大化问题之中"。这会使计算中的哪些部分发生变化？
```

```{solution-start} lps_ex_phillips
:class: dropdown
```

这次意外冲击只改变了本期适用的转移矩阵，所以我们用各市场已有的阈值将上一期的情形延续下来，并从被改动过的矩阵中抽取本期的需求。

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

无意外冲击这条基准线重现了均衡，这正是应有的结果。

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

失业和工资朝相反方向移动，正如卢卡斯和普雷斯科特所说，这些点描绘出一条向下倾斜、在均衡处带有一个拐点的曲线：一条菲利普斯曲线。

需求意外下降会把部分市场推到其就业阈值以下，于是它们裁减工人，失业大幅上升。工资也下降，但降幅小于需求降幅，因为裁减工人的市场削减了就业，这提高了留下者的边际产品。

需求的意外上升只能通过阻止本来会发生的分离来降低失业，而这样的分离本就为数不多；与此同时，每个市场的工资都随其需求上升，大致是一比一的关系，而且在这一期内不可能有新工人到来把工资压低。

所以这条曲线在失业率较低处陡峭，在失业率较高处平坦。

符号和拐点并不依赖于网格，但失业反应的大小却依赖于网格：在七状态链上，一档大约相当于 $\log s$ 的五分之一，所以给定的平均对数需求下降会集中作用于少数几个市场，每个都受到重创；而在更精细的链上，相同的下降幅度对失业的提升作用更小。

对于第3部分，这个计算之所以保持阈值和 $\lambda$ 不变，是因为没有人预期到这一变动。如果这类变动反复出现，工人就会把它们纳入自己所使用的转移规律之中，贝尔曼方程就会随之改变，$\bar n$、$\bar y$ 和 $\lambda^*$ 也会随之改变；卢卡斯和普雷斯科特把这一点留作"未来研究中一项非平凡的进一步工作"。

{doc}`phillips_two_stories` 把菲利普斯曲线的预期论述及其背后的自然率假说，一直追溯到数据当中。

```{solution-end}
```

```{exercise}
:label: lps_ex_planner

{ref}`lps_efficiency` 证明了均衡求解了计划者的问题 {eq}`lps_planner`。用数值方法进行检验，看看它能说明什么、不能说明什么。

1. 在 $\lambda^*$ 处，通过价值函数迭代求解 {eq}`lps_planner`，在整个网格上选择 $y'$，验证 $W(s, y) - W(s, y-1) = v(s, y)$，并验证计划者对 $y'$ 的选择就是 {eq}`lps_transition`。
1. 现在让移动消耗资源：每位离开的工人花费 $c = 2$。证明 {ref}`lps_ex_cost` 中的均衡正是一个面对该项成本的计划者的配置，该计划者为每位被释放的工人记入 $\lambda - c$，为每位被带入的工人收取 $\lambda$。这对 {ref}`lps_ex_cost` 中较低的剩余意味着什么？
1. 将均衡处每工人的稳态剩余，与保持 $\mu = 50$ 的两种替代方案进行比较：一项对移动的小额补贴，$c = -0.1$ 和 $c = -0.2$；以及一个更有耐心的计划者会选择的阈值，取 $\beta' = 0.95$ 和 $0.99$，并选取 $\lambda$ 使每市场工人数保持为 $\mu$。均衡是否使稳态剩余最大化？将答案与 {ref}`lps_efficiency` 相调和。
```

```{solution-start} lps_ex_planner
:class: dropdown
```

对于第1部分，`planner_value` 求解 {eq}`lps_planner`，对于第2部分，释放的记入值降为 $\lambda - c$。流量项不依赖于 $W$，所以我们只计算一次。

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

对于第2部分，用资源成本进行同样的检验。

```{code-cell} ipython3
λ_2 = equilibrium_λ(model, μ, 2.0, lo=λ_0)
check_planner(model, λ_2, 2.0)
```

在有资源成本的情形下，均衡正是针对那项代价更高的移动技术而言计划者的配置：{ref}`lps_efficiency` 的论证在以 $\lambda - c$ 作为释放记入值的情形下仍然成立，而资源成本会计入剩余之中。

所以 {ref}`lps_ex_cost` 中在有资源成本时发现的较低剩余，是较差技术所付出的代价，而均衡在这种情况下已经做到了最好。返还式税收以完全相同的方式改变决策，却不消耗任何东西，所以技术仍然是自由放任下的那种，而计划者针对它所选择的是自由放任的配置，于是被征税的均衡就是被扭曲的：它较低的剩余是一种效率损失。

对于第3部分，补贴就是负的 $c$，一个更有耐心的计划者的阈值来自用 $\beta'$ 替代 $\beta$ 的工人问题。

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

这两种替代方案都把每工人稳态剩余提升到高于均衡水平，而且都伴随着更多的重新配置和更多的失业。所以均衡并不使稳态剩余最大化，而 {ref}`lps_efficiency` 也从未说过它会这样做。

该定理所比较的是从均衡自身的分布出发的现值，而迁移到这些稳态之一的过程本身会消耗剩余：更多工人处于转移途中，意味着现在就要损失他们一期的产出，以换取之后才会到来、并被贴现的收益。

一个以 $\beta' > \beta$ 贴现的计划者，会更看重那些较晚到来的收益，这正是其阈值会移动更多工人的原因；在不贴现的极限情形下，它会使稳态剩余最大化，这正是黄金法则的对应情形，而均衡正是修正黄金法则的对应情形。

{ref}`lps_ex_cost` 中的稳态比较与该定理在成本较大时所指向的方向相同，但这并非该定理所蕴含的：一项小额补贴同样会使经济体偏离计划者的配置，却提高了稳态剩余。

```{solution-end}
```