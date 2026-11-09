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
    Overview: 概览
    The economy: 经济体
    The economy::Timing: 时序
    The economy::The value of search: 搜寻的价值
    A worker's problem: 工人的问题
    Employment, wages, and two thresholds: 就业、工资与两个阈值
    Employment, wages, and two thresholds::How many work: 有多少人工作
    Employment, wages, and two thresholds::How many arrive: 有多少人到达
    Employment, wages, and two thresholds::Three regions, one transition law: 三个区域，一条转移规律
    A computable version: 一个可计算的版本
    Economy-wide equilibrium: 全经济均衡
    Learning the value of search: 学习搜寻的价值
    Inside the equilibrium: 均衡内部
    Inside the equilibrium::What a market looks like: 一个市场是什么样子
    Inside the equilibrium::The workforce distribution and a sample path: 劳动力分布与一条样本路径
    Inside the equilibrium::Where markets sit: 市场所处的位置
    Inside the equilibrium::The wage distribution: 工资分布
    Efficiency: 效率
    Efficiency::Surplus: 剩余
    Efficiency::A planner for one market: 一个管理单一市场的计划者
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

除 Anaconda 中已有的库外，本讲座还需要以下库：

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

## 概览

本讲座研究 {cite:t}`LucasPrescott1974`，它是我们在 {doc}`lucas_prescott_investment` 中研究的投资论文在劳动力市场方面的姊妹篇。

这两篇论文由同一批作者在同一地点、相隔仅几年的时间内写成，使用的工具也相同：通过逐次近似求解的贝尔曼方程、布莱克韦尔定理，以及一个不变分布描述长期行为的马尔可夫过程。

变化的是问题本身。

卢卡斯和普雷斯科特所针对的目标，是 {cite:t}`friedman1968role` 和 {cite:t}`Phelps1967` 提出的自然失业率。

用他们的话说，讨论自然失业率时所用的语言"暗示它可能具有竞争均衡的性质"，然而

> *并不存在这样的理论模型，其中非零的均衡失业率被确定下来并对其性质进行研究。*

没有这样的模型，就没有框架来提出失业所引发的福利问题，也无法解释为什么平均失业率在不同国家之间差异如此之大。

20 世纪 60 年代的搜寻文献无法提供这样的模型。

{doc}`mccall_model` 描述了那段文献的核心问题：工人从一个已知分布中抽取工资要约，并决定何时停止搜寻。

这是一个关于个体的模型，它将工资分布视为给定。

卢卡斯和普雷斯科特认为，他们所关心的问题不能用那种方式回答：

> *最优的劳动供给行为及其所依据的工资分布，必须在一个市场（而非个体）行为模型中同时确定。*

他们构建了一个由许多空间上分离的市场组成的经济体，每个市场都有其自身随机波动的劳动需求。

这一图景来自埃德蒙·费尔普斯的"岛屿经济"，他们称之为"本模型的直接前身" {cite}`PhelpsEtAl1970`。

它与 {cite:t}`LucasNeutrality1972` 中的岛屿不同，在那里就业随总需求波动，但失业作为一种活动并未与闲暇区分开来。

而在这里，二者是有区别的。

一个市场中的工人既可以按出清工资就业，也可以离开。

离开意味着本期一无所获，下期到达别处，因此*失业就是在市场之间转移的行为*。

一切都是竞争性的，每个人都在优化，而预期在 {cite:t}`muth1961` 和投资论文 {cite:t}`Lucas_Prescott_1971` 的意义上是理性的：工人用来评估其前景的分布，正是他们自己的决策所生成的分布。

论文的核心对象是一个单一数值 $\lambda$，即搜寻的预期现值。

单个市场将 $\lambda$ 视为给定，正如竞争性企业将价格视为给定一样。

就整个经济而言，$\lambda$ 不断调整，直到各市场想要保留的劳动力等于经济体所拥有的劳动力。

在论文的均衡中，总有一些市场拥有的工人多于它们想雇用的数量，因此每一期的失业率都为正；{ref}`lps_regions` 解释了我们所使用的可计算版本在何种情况下保留了这一性质。

卢卡斯和普雷斯科特还论证说，这种失业未必意味着低效率。

```{note}
该模型中不存在拥挤外部性，也没有匹配函数、没有议价、没有工资刚性：失业之所以存在，是因为转移需要一期时间，而在均衡中总有人正在转移。

正在搜寻的工人对其同伴造成的拥挤，就如同任何商品的卖家都会对其他卖家造成拥挤一样，而 $\lambda$，正如任何其他均衡价格一样，传递出论文所称的"额外搜寻者的正确社会回报"。

论文并未证明任何福利定理，也未解决实际劳动力市场是否存在重要外部效应这一问题。

{ref}`lps_efficiency` 表明，在模型内部，这一论断可以被精确化：当转移需要一期时间时，平稳均衡使得从其自身的工人分布中可获得的剩余的预期现值最大化，而 $\lambda$ 正是向搜寻者队伍中增加一名工人的现值。
```

本讲座首先阐述理论：工人的问题、描述一个市场的两个阈值，以及一个市场劳动力的运动规律。然后计算一个例子，通过求出均衡 $\lambda$ 来封闭经济，并探讨工人如何能够了解这一数值。

在考察均衡内部——市场、劳动力和工资——之后，它证明了效率主张并将该模型与其后续模型相联系。练习涉及论文的比较静态分析、转移成本、论文自身的两状态示例、无方向搜寻、菲利普斯曲线以及效率的数值检验。

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

存在许多市场，每个市场都有自己的需求状态。

需求冲击*在各市场间相互独立*，因此总需求保持不变，但在*单个市场内部具有持续性*，因此今天的需求能够透露关于明天需求的信息。

在每个市场中，一个竞争性产业以劳动为唯一投入生产单一商品。

如果 $p = D(s, Q)$ 是需求状态 $s$ 下的反需求曲线，$q(n)$ 是 $m$ 家相同厂商中每一家的生产函数，那么利润最大化会使工资等于劳动的边际价值产品，函数

```{math}
:label: lps_R
R(s, n) = D\big(s, m\,q(n)\big)\, q'(n)
```

总结了该劳动力市场需求端的一切信息。

卢卡斯和普雷斯科特舍弃了 $D$ 和 $q$，而是直接对 $R$ 作出假设：它是正的、有界的、可微的，且满足

$$
R_s(s,n) > 0, \qquad R_n(s,n) < 0 .
$$

需求状态 $s$ 遵循一个具有转移密度 $f(s', s)$ 的马尔可夫过程，该过程存在唯一的平稳分布。

其条件分布函数 $F(s', s)$ 关于 $s$ 递减，因此今天较高的需求使得明天较高的需求更有可能发生。

### 时序

一个市场在一期开始时拥有一个**劳动力**量 $y$，这是它所能雇用的最大数量。

然后，在该期内：

1. 需求 $s$ 实现，劳动力市场出清，以工资 $w = R(s, n)$ 雇用 $n \le y$ 名工人
1. 未被雇用的 $y - n$ 名工人本期一无所获，加入搜寻者队伍
1. 被雇用的工人留在该市场进入下一期；搜寻者被分配到各市场，于下一期到达

因此每当工人转移时，他恰好*失业一期*，失业者就是正处于转移途中的工人。

### 搜寻的价值

由于需求冲击在各市场间相互独立，且市场数量众多，一旦经济体的工人在各市场间的分布稳定下来，它就保持不变，尽管每个市场的劳动力量各自波动。

因此，搜寻者的前景并不取决于他正在离开的那个市场，搜寻的预期现值是一个单一数值 $\lambda$。

搜寻者被分配到各市场的方式使得到达的预期价值*相等*，因此没有任何市场给到达的工人提供超过 $\lambda$ 的价值。

我们暂且将 $\lambda$ 视为给定，并在 {ref}`lps_economy_wide` 中确定它。

## 工人的问题

设 $v(s, y, \lambda)$ 为某工人在一期开始时发现自己处于需求为 $s$、劳动力量为 $y$ 的市场中时，其收入的预期现值，设 $\beta \in (0,1)$ 为贴现因子。

卢卡斯和普雷斯科特通过考察三种情形得出了 $v$ 的函数方程。

**情形 A：有工人离开。** 离开的工人获得 $\lambda$。留下的工人不能获得更多，因为离开者本可以留下；也不能获得更少，因为他们本可以离开。所以 $v = \lambda$。

**情形 B1：所有工人留下且没有新工人到达。** 当前就业为 $y$，工资为 $R(s,y)$，下一期的劳动力量仍为 $y$：

$$
v(s,y,\lambda) = R(s,y) + \beta \int v(s', y, \lambda) f(s', s)\,ds' .
$$

**情形 B2：所有工人留下且有新工人到达。** 到达的工人是搜寻者，因此下一期处于该市场的贴现预期价值 $\beta \int v f\, ds'$ 必须恰好等于 $\lambda$，即促使其搜寻的那个价值：

$$
v(s,y,\lambda) = R(s,y) + \lambda .
$$

当劳动力量不变情况下的贴现延续值低于 $\lambda$ 时，没有人到达，适用情形 B1；当该值本会超过 $\lambda$ 时，到达者会将其拉低到 $\lambda$，适用情形 B2。

无论哪种情形，延续值都是 $\lambda$ 与 $\beta \int v(s', y, \lambda) f(s', s)\, ds'$ 二者中较小的一个，这就是下式中的 $\min$。

当即便如此仍使某工人的所得低于 $\lambda$ 时，情形 A 适用，此时他选择 $\lambda$，这就是 $\max$。

三种情形合并为一个方程：

```{math}
:label: lps_bellman
v(s, y, \lambda) = \max\left\{\lambda,\;
R(s,y) + \min\left[\lambda,\;
\beta \int v(s', y, \lambda) f(s', s)\, ds' \right]\right\} .
```

这两个算子的作用不同。*max* 是工人自身的选择权：他总可以离开并获得 $\lambda$。

*min* 则完全不是工人的选择，而是竞争在搜寻者之间施加的一种约束：如果该市场中的延续值超过 $\lambda$，搜寻者就会不断涌入，直至不再超过为止。

```{prf:proposition}
:label: lps_prop_v

对每一个 $\lambda$，方程 {eq}`lps_bellman` 都存在唯一的有界连续解 $v$，并且从任意起点出发的逐次近似都收敛于它。

该解关于 $s$ 和 $\lambda$ 非减，关于 $y$ 非增。
```

{prf:ref}`lps_prop_v` 的证明与 {doc}`lucas_prescott_investment` 中通篇所用的方法相同：方程 {eq}`lps_bellman` 右端所定义的算子是单调的，且满足一个贴现条件，因此布莱克韦尔定理 {cite}`Blackwell1965` 适用。

关于 $s$ 的单调性正是用到了 $F(s', s)$ 关于 $s$ 递减这一假设之处。一个工人所在的市场越拥挤，他的处境就越差，因为更大的劳动力量意味着更低的边际产品和更低的工资，这也是为什么 $v$ 关于 $y$ 非增。

## 就业、工资与两个阈值

有了 $v$，市场行为的其余部分便可由两条阈值规则得出。

### 有多少人工作

设 $\bar n(s, \lambda)$ 为这样的就业水平：在该水平下，边际工人恰好对留下和搜寻无差异：

```{math}
:label: lps_nbar
R(s, \bar n) + \beta \int v(s', \bar n, \lambda) f(s', s)\, ds' = \lambda .
```

左边是当就业和下一期劳动力量都是 $\bar n$ 时，一个工人留下所得到的：本期工资加上贴现后的延续值。它关于 $\bar n$ 非增。

如果边际收入在某一劳动力水平处降为零——正如下文可计算版本中那样——那么处于如此拥挤的市场中的价值已经降到了 $\lambda$，左边在 $\beta \lambda < \lambda$ 处持平，{eq}`lps_nbar` 在那一点无法成立。

在整数劳动力网格上，{eq}`lps_nbar` 很少精确成立，就业水平就是使左边至少等于 $\lambda$ 的最大 $\bar n$。

于是就业为

```{math}
:label: lps_employment
n(s, y, \lambda) = \min\{\bar n(s,\lambda),\; y\},
\qquad w = R(s, n) ,
```

该市场的失业为 $y - n$，当劳动力超过 $\bar n$ 时该值恰为正。

### 有多少人到达

一个市场只有在其雇用了全部劳动力且延续值仍超过 $\lambda$ 时，才会吸引搜寻者。

设 $\bar y(s,\lambda)$ 为该性质不再成立时的劳动力量：

```{math}
:label: lps_ybar
\beta \int v(s', \bar y, \lambda) f(s', s)\, ds' = \lambda .
```

到达者会将市场填充到 $\bar y(s, \lambda)$，不会更多。注意 $\bar y \le \bar n$，因为 $R \ge 0$ 意味着：任何仅凭延续值就能满足 $\lambda$ 的劳动力量，一旦加上工资，也必然能满足 $\lambda$。

在边际产品为正的地方，这一不等式是严格的：一个刚好值得搜寻者进入的市场，严格地位于人人都工作的那个范围之内。

在下文校准中最差的需求状态下，两个阈值在均衡的 $\lambda$ 值处都为零，该市场既不雇用任何人，也不吸引任何人。

(lps_regions)=
### 三个区域，一条转移规律

{eq}`lps_employment` 和 {eq}`lps_ybar` 共同将 $(s,y)$ 平面划分为论文图 3 中的三个区域：

* 区域 I，其中 $y > \bar n(s, \lambda)$，该市场释放工人直至降到 $\bar n(s,\lambda)$；
* 区域 II，位于两个阈值之间，此处没有人移动；
* 区域 III，其中 $y < \bar y(s, \lambda)$，该市场被填充至 $\bar y(s,\lambda)$。

这使得劳动力的运动规律具有非常简单的形式：

```{math}
:label: lps_transition
y' = \min\big\{\max\{y,\ \bar y(s,\lambda)\},\ \bar n(s,\lambda)\big\} .
```

贝尔曼方程是在*不变*的劳动力量 $y$ 处评估延续值，而转移规律则把劳动力移到别处。

这并不矛盾，因为 $\max$ 和 $\min$ 已经处理了劳动力移动的两种情形。

在有工人到达的地方，$\min$ 将延续值替换为 $\lambda$，而被填满到 $\bar y$ 的市场对其中任何人而言至少值 $\lambda$：在连续劳动力量下恰好等于 $\lambda$，而在下文所用的整数网格上可能略高一些。在工人离开的地方，$\max$ 将整个表达式替换为 $\lambda$。

因此，在不变的 $y$ 处的延续值，只在劳动力量真正不变的区域才被用到。

```{note}
论文允许到达量是随机的：一个劳动力不足的市场会收到 $a(s,y,\lambda)\,x$ 名工人，其中 $x$ 是均值为一的正随机变量，因此搜寻须"平均地消除租金"而非精确地消除。

与论文附录中的数值例子一样，我们设 $x \equiv 1$，这就给出了 {eq}`lps_transition`。

在论文中，随机到达使平稳分布具有严格为正的密度（其命题 3），因此总有某个市场正在释放工人，每一期失业都为正。

在 $x \equiv 1$、有限多个需求状态，且需求过程能在一期内从任意一个状态转移到任意另一个状态的情况下，有三种情形。

* 若 $\max_s \bar y(s) > \min_s \bar n(s)$，平稳分布是唯一的，失业为正：在 $\bar y$ 最高的状态下被填满的市场，当需求降至 $\bar n$ 最低的状态时会释放工人，这两期就能把一个市场从任意初始劳动力量带到 $\min_s \bar n(s)$。
* 若 $\max_s \bar y(s) = \min_s \bar n(s)$，平稳分布依然唯一，每个市场都保有该劳动力量，失业为零。
* 若 $\max_s \bar y(s) < \min_s \bar n(s)$，任何介于二者之间的劳动力量在每种需求状态下都保持原地不动，因此存在一个平稳分布的连续统，失业为零。

{ref}`lps_ex_persistence` 遇到了第三种类型的经济。
```

## 一个可计算的版本

我们遵循 {cite:t}`LucasPrescott1974` 的附录：有限数量的需求状态、整数劳动力量，以及对 {eq}`lps_bellman` 的价值函数迭代。

对需求过程，我们采用论文脚注 7 中的例子，其中 $\log s$ 遵循一个持续性参数介于零和一之间的 AR(1) 过程。

我们用 Rouwenhorst 方法将其离散化为七个状态。

与 {cite:t}`Tauchen1986` 的方法不同，即使在持续性较高时，它也能精确匹配 $\log s$ 的持续性和平稳方差；{ref}`lps_ex_persistence` 的解答展示了若不这样做会出什么问题。

对于 $R$，我们采用一个随就业线性下降、在某容量 $N$ 处降为零的边际收入表达式，

$$
R(s, n) = s\,\big(1 - n/N\big)^{+} .
$$

在两个需求状态、$N = 100$ 的情形下，这正是论文例 1 中的表达式，绘制于其图 5 的左上图，这里放大了两倍。

容量设定偏离了理论假设（该假设认为 $R$ 处处为正），但这也是论文自身例子所采用的做法。它将一个市场想要的劳动力量上限设定在 $N$ 以下，因此论文的命题 5（保证每个单位劳动力市场都存在均衡）在此仅对低于该上限的劳动力量成立。

它还塑造了当工人充裕时失业的行为方式，我们稍后会看到这一点。

$R$ 下降的陡峭程度决定了需求变动时就业变动的幅度，因此也决定了将工人从一个市场送到另一个市场能获得多少收益；{ref}`lps_ex_two_states` 将对此作进一步探讨。

函数 `make_model` 能根据任意需求链和边际收入表达式构建一个市场，`create_model` 则用它构建本讲座的经济体。

```{code-cell} ipython3
Model = namedtuple('Model', ['P', 's', 'y', 'R', 'β'])

def linear_R(s, y, N=200):
    "边际收入 s (1 - n/N)^+，在容量 N 处降为零。"
    return s[:, None]*np.maximum(1 - y[None, :]/N, 0.0)

def make_model(P, s, β=0.9, y_max=260, R=linear_R, **R_args):
    "一个需求链为 (P, s)、边际收入为 R、定义在 y = 0, ..., y_max 上的市场。"
    y = np.arange(y_max + 1)
    R_sy = R(s, y, **R_args)
    R_sy[:, 0] = 0.0                                # 空市场不支付任何工资
    return Model(P=P, s=s, y=y, R=R_sy, β=β)

def create_model(n_s=7, ρ=0.9, sd=0.26, β=0.9, N=200, y_max=260,
                 method='rouwenhorst'):
    """岛屿经济，其对数需求为 AR(1)，离散化为 n_s 个状态。

    ρ 和 sd 分别是对数需求的持续性和平稳标准差，因此仅改变 ρ
    并不会改变需求的波动程度。N 是一个市场的容量：边际收入在此处降为零。
    """
    σ = sd*np.sqrt(1 - ρ**2)
    if method == 'rouwenhorst':
        with warnings.catch_warnings():         # quantecon 提示其 API 发生了变化
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

由于延续值是在不变劳动力量处评估的，该方程对每个 $y$ 而言都是一个独立的问题，只在不同需求状态之间相连，这也是它求解代价低廉的原因，也是我们对空市场所采用的约定不会影响其他任何东西的原因。

我们需要某个搜寻价值来求解它，所以这里用 $\lambda = 7.1$，这个值接近 {ref}`lps_economy_wide` 所计算出的均衡值。

参数 `c`，即离开的成本，在 {ref}`lps_ex_cost` 之前始终为零，它将工人的外部选择由 $\lambda$ 变为 $\lambda - c$。

```{code-cell} ipython3
def worker_value(model, λ, c=0.0, tol=1e-11, max_iter=10_000):
    "求解工人的函数方程；离开的工人获得 λ - c。"
    P, R, β = model.P, model.R, model.β
    v = np.full(R.shape, λ - c)
    for i in range(max_iter):
        v_new = np.maximum(λ - c, R + np.minimum(λ, β*(P @ v)))
        v_new[:, 0] = λ - c                         # 空市场不保有工人
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, i
        v = v_new
    raise RuntimeError(f"no convergence in {max_iter} iterations at λ = {λ}")

v, iters = worker_value(model, 7.1)
print(f"converged in {iters} iterations")
```

两个阈值分别是使相应价值仍能满足 $\lambda$ 的最大就业水平和最大劳动力量。

```{code-cell} ipython3
def last_at_least(A, bar):
    "对 A 的每一行，返回超过 0 的位置中使该行 ≥ bar 的最大索引，若无则为 0。"
    return np.array([np.max(np.where(row >= bar)[0]) if (row[1:] >= bar).any() else 0
                     for row in A])

def thresholds(model, v, λ, c=0.0):
    "就业阈值 n̄(s) 和到达阈值 ȳ(s)。"
    EV = model.β*(model.P @ v)                  # 不变劳动力量下的延续值
    n_bar = last_at_least(model.R + EV, λ - c)  # 作为边际工人留下
    y_bar = last_at_least(EV, λ)                # ≤ n_bar，因为 R ≥ 0 且 c ≥ 0
    return n_bar, y_bar

n_bar, y_bar = thresholds(model, v, 7.1)
print("n̄(s) =", n_bar)
print("ȳ(s) =", y_bar)
```

就业受 $\bar n$ 限制，到达者填充至 $\bar y$，{eq}`lps_transition` 使劳动力量移动。

$(s, y)$ 上这一马尔可夫链的不变分布是可以精确计算出来的。

{eq}`lps_transition` 的一期要么使劳动力量保持不变，要么将其带到某个阈值上，因此从某个阈值出发，该链永远不会离开阈值集合，于是我们可以求解关于 $(s, t)$ 对的小链，其中 $t$ 遍历至多 $2 \times 7$ 个阈值。

一个不在阈值上的劳动力量，只有在每一种需求状态下都保持不变，才能在长期携带概率。

这样一个区间的两端本身就是保持不变的阈值，于是阈值链就有多于一个不变分布：这正是 {ref}`lps_regions` 中注释所述的第三种情形。

因此我们令 `qe.MarkovChain` 找出小链的每一个不变分布，当不止一个时报错而不是随意挑选一个。

当某个阈值到达网格顶端——在那里它会被悄无声息地截断——该函数也会报错。

```{code-cell} ipython3
class NotBracketed(ValueError):
    "二分查找被给定了一个不包含切换点的区间。"

class NotUnique(ValueError):
    "市场的不变分布不唯一。"

class GridTooSmall(ValueError):
    "某个阈值已到达劳动力网格的顶端。"

def stationary_distribution(model, n_bar, y_bar):
    "通过精确计算得到 (s, y) 上该链的不变分布。"
    P, n_s = model.P, len(model.s)
    if n_bar.max() == model.y[-1]:
        raise GridTooSmall(f"n̄ reaches the top of the grid, y = {model.y[-1]}")
    T = np.unique(np.concatenate([n_bar, y_bar]))     # 阈值集合
    k = len(T)
    Q = np.zeros((n_s*k, n_s*k))                       # (s, t) 上的链，t 属于 T
    for i in range(n_s):
        t_next = np.searchsorted(T, np.clip(T, y_bar[i], n_bar[i]))
        for j in range(k):
            Q[i*k + j, t_next[j]::k] = P[i]            # 到任意 s' 的 (s', t_next[j])
    ψ_T = qe.MarkovChain(Q).stationary_distributions
    if len(ψ_T) > 1:
        raise NotUnique(f"the chain on the thresholds has {len(ψ_T)} recurrent classes")
    ψ = np.zeros((n_s, len(model.y)))
    ψ[:, T] = ψ_T[0].reshape(n_s, k)
    return ψ

def market_outcomes(model, λ, c=0.0):
    "在给定搜寻价值下、离开成本为 c 时，关于市场的一切信息。"
    v, _ = worker_value(model, λ, c)
    n_bar, y_bar = thresholds(model, v, λ, c)
    ψ = stationary_distribution(model, n_bar, y_bar)
    y = model.y[None, :]
    n = np.minimum(y, n_bar[:, None])                      # 每个 (s, y) 下的就业
    w = np.take_along_axis(model.R, np.maximum(n, 1), axis=1)    # 以及相应的工资
    Φ = np.cumsum(model.R, axis=1)                         # 剩余：见 lps_efficiency
    employment, wage_bill = (ψ*n).sum(), (ψ*n*w).sum()
    return dict(v=v, n_bar=n_bar, y_bar=y_bar, ψ=ψ, n=n, w=w,
                workforce=(ψ*y).sum(), employment=employment,
                unemployment=(ψ*(y - n)).sum(),
                # 每位在职工人的平均工资；空市场不计权重
                wage=wage_bill/employment if employment > 0 else np.nan,
                wage_bill=wage_bill,
                surplus=(ψ*np.take_along_axis(Φ, n, axis=1)).sum())

out = market_outcomes(model, 7.1)
print(f"workforce {out['workforce']:.2f}, employment {out['employment']:.2f}, "
      f"unemployment {out['unemployment']:.2f}")
```

小链有 $n_s |T|$ 个状态，其中 $|T| \le 2 n_s$ 是不同阈值的数量，求其不变分布的代价约为 $(n_s |T|)^3$ 次运算。

当有七个状态时这一代价可忽略不计，在 25 个状态时也尚可承受——此时求解整个均衡大约需要一两秒——但状态数超过约 35 时就会变慢。

(lps_economy_wide)=
## 全经济均衡

到目前为止 $\lambda$ 一直是一个参数。

现在来封闭模型。

该经济体的劳动力总量固定，每个市场为 $\mu$。

对任意 $\lambda$，上述计算给出一个市场的需求和劳动力量的平稳分布 $\Psi(s, y; \lambda)$，其在我们网格上的概率即为数组 `ψ`。

用它对各市场的劳动力量求平均，就得到了经济体的**劳动力需求**，

```{math}
:label: lps_demand
E(y; \lambda) = \int y \, \Psi(ds, dy; \lambda) .
```

$\lambda$ 越高，每个市场就越早释放工人、填充程度越低，因此 $E(y;\lambda)$ 是递减的。

均衡是使各市场恰好想保有经济体现有工人数量的那个 $\lambda^*$：

```{math}
:label: lps_equilibrium
E(y; \lambda^*) = \mu .
```

由于该曲线是递减的，二分法可以求出 $\lambda^*$。

在整数网格上，$E(y; \lambda)$ 是 $\lambda$ 的阶梯函数，每当某个阈值移动一个工人单位时就会跳变，因此 {eq}`lps_equilibrium` 只在一个台阶的精度内成立，下文和练习中打印出的劳动力量与 $\mu$ 略有差异。

`equilibrium_λ` 首先检验 $\mu$ 是否位于区间两端所对应的劳动力需求之间，若不是则报出 `NotBracketed`。

```{code-cell} ipython3
def workforce_demand(model, λ, c=0.0):
    "搜寻价值为 λ 时，每个市场的平均劳动力量。"
    return market_outcomes(model, λ, c)['workforce']

def bisect(test, lo, hi, tol=1e-4, checked=False):
    "在 lo 处为假、hi 处为真的测试中，以 tol 的精度定位切换点。"
    if not checked and (test(lo) or not test(hi)):
        raise NotBracketed(f"the test does not switch between {lo} and {hi}")
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        lo, hi = (lo, mid) if test(mid) else (mid, hi)
    return hi

def equilibrium_λ(model, μ, c=0.0, lo=0.01, hi=40.0):
    "在递减的需求曲线 E(y; λ) = μ 上二分查找。"
    E_lo, E_hi = workforce_demand(model, lo, c), workforce_demand(model, hi, c)
    if not E_lo > μ > E_hi:
        raise NotBracketed(f"μ = {μ} is not between E(y; {lo}) = {E_lo:.2f} "
                           f"and E(y; {hi}) = {E_hi:.2f}")
    return bisect(lambda λ: workforce_demand(model, λ, c) <= μ, lo, hi, checked=True)

μ = 50                        # 每个市场的工人数量，相对于容量 200
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

容量限制体现在区间检验中。

即使 $\lambda = 0.01$ 时，一个市场也最多只想要 $N - 1 = 199$ 名工人，因此每个市场拥有 200 名工人的经济体不存在均衡，`equilibrium_λ` 会如实报告，而不会返回一个数值。

```{code-cell} ipython3
try:
    equilibrium_λ(model, 200)
except NotBracketed as err:
    print(err)
```

卢卡斯和普雷斯科特将均衡绘制为一条需求曲线与一条垂直供给曲线相交，这正是它本身的样子。

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

左图是论文的图 4。右图对应于论文图 5 的下方两个面板，后者绘制了平均失业与 $\lambda$ 的关系，适用于两个数值例子；由于 $E(y;\lambda)$ 是递减的，改为将其对劳动力量作图就会显示出同样的隆起形状，只是镜像翻转了。

每个市场的失业*并非*随劳动力总量单调变化——卢卡斯和普雷斯科特在他们自己的例子中也报告了这一特征——而均衡恰好处于该隆起的顶端附近。

当工人充裕时，$\lambda$ 较低，每个市场都紧贴其容量运行，每种需求状态下的两个阈值在此被挤压到一起，因此即便需求发生较大变动，被移动的工人也很少。

隆起的那一侧是容量限制所致。

当工人稀缺时，$\lambda$ 较高，劳动力量较小，许多市场彻底关闭。此处每个市场的失业较小，仅仅是因为每个市场的工人太少；而作为一个*比率*，它却很高，因为留下的工人都处在最好的市场中，并且一旦需求下降就会被释放出来。

失业率与空置市场的占比都随 $\lambda$ 上升：

```{code-cell} ipython3
for λ in (0.6, λ_star, 10.0, 13.0):
    r = market_outcomes(model, λ)
    print(f"λ = {λ:6.3f}:  unemployment rate {100*r['unemployment']/r['workforce']:5.2f}%,"
          f"  markets with no workers {100*r['ψ'][:, 0].sum():5.1f}%")
```

(lps_learning)=
## 学习搜寻的价值

卢卡斯和普雷斯科特用论文的一节来讨论均衡本身没有回答的一个问题：我们一直假设工人知道 $\lambda$，却没有说明他们是如何知道的。

归根结底，他们写道，这"是一个心理学理论而非经济学理论的问题，因此我们不为在此以临时的'适应性'方式提出这一问题而道歉"。

他们还说明了为什么这个问题是可以处理的。工人必须学习的分布 $\Psi$ 取决于他们自身的行为，因此随着他们认知的改变，他们正在学习的对象也随之改变，这"原则上可能引发无法克服的分析困难"。

但在这里并非如此，"因为 $\Psi$ 中与工人决策相关的唯一特征就是参数 $\lambda$"。这正是 {doc}`ls_learning` 中那种自我指涉结构，坍缩为一个单一数值。

他们给出的答案是一个适应性的故事。假设工人相信搜寻的价值高于其实际值；那么离开所在市场的人数就会超过能以该回报被安置的人数。

在论文中，超额的搜寻者是通过让到达变量 $x$ 的均值上升来被安置的，于是接收方市场获得的工人超出了预期，搜寻者和留任者都在所获得的工资上感到失望。他们随后向下修正这一数值。

记 $g$ 为一个在零点处取值为零的递减函数，则

$$
\frac{d\lambda}{dt} = g(\lambda - \lambda^{*}) .
$$

稳定性由 $g$ 所假设的符号内置于其中，而证成该符号的理由在于需求曲线是向下倾斜的：当 $\lambda$ 过高时，各市场想要的工人数量少于经济体所拥有的数量。`equilibrium_λ` 正是依赖于同一事实：它对根进行区间搜索而非沿用论文的调整过程，但二者之所以都能奏效，都是因为 $E(y;\lambda)$ 是递减的。

这个故事可以用离散时间来表述：工人持有一个信念 $\lambda_k$，各市场据此行动，该信念朝着对工人的超额需求方向被修正，

```{math}
:label: lps_tatonnement
\lambda_{k+1} = \lambda_k + \gamma\, \big(E(y; \lambda_k) - \mu\big) .
```

这就是瓦尔拉斯试探过程，以 $\lambda$ 为价格，以 $E(y;\lambda) - \mu$ 为超额需求。它具有 {doc}`bounded_rationality` 中松弛算法 {eq}`relaxation` 的形式，应用于 $T(\lambda) = \lambda + E(y;\lambda) - \mu$，并具有 E-稳定性的符号结构：$\lambda^*$ 是稳定的，因为 $E(y;\lambda)$ 向下倾斜。

这一类比只是形式上的，因为 $T$ 是在一个现值上加上若干工人数量，而不是从一个被感知的值映射到 $\lambda$ 的实际值。

在 $\lambda^*$ 附近，误差 $\lambda_k - \lambda^*$ 每一步都乘以 $1 + \gamma E'(\lambda^*)$，其中 $E'$ 是该曲线的斜率。因此当 $0 < \gamma < 2/|E'(\lambda^*)|$ 时迭代收敛，当 $\gamma < 1/|E'(\lambda^*)|$ 时迭代无振荡地收敛。

由于 $E(y;\lambda)$ 在整数网格上是阶梯函数，我们用足够宽、能跨越多个台阶的中心差分来估计其斜率。

```{code-cell} ipython3
h = 0.5
slope = (workforce_demand(model, λ_star + h) - workforce_demand(model, λ_star - h))/(2*h)
print(f"slope of E(y; λ) near λ*: {slope:.1f}")
print(f"monotone convergence for γ < {1/abs(slope):.3f}, "
      f"convergence for γ < {2/abs(slope):.3f}")

def learn_λ(model, μ, γ, λ_0=10.0, n=40):
    "朝着对工人的超额需求方向修正关于 λ 的信念。"
    path = [λ_0]
    for _ in range(n):
        path.append(path[-1] + γ*(workforce_demand(model, path[-1]) - μ))
    return np.array(path)

paths = {γ: learn_λ(model, μ, γ) for γ in (0.01, 0.05, 0.12)}
```

这里是三个增益值，两个低于界限、一个高于界限。

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

当 $\gamma = 0.01$ 时，信念缓慢下降到 $\lambda^*$；当 $\gamma = 0.05$——接近 $1/|E'|$——时，只需几步就能到达。

当 $\gamma = 0.12$，超过了界限，它不会收敛：它在一个高值和一个低值之间来回跳动，在那里曲线的斜率不同，超调不再增长。

这些界限的单位是每个工人的价值：如果每个市场的规模都翻倍，$E'$ 也会翻倍，稳定的增益值则会减半。

卢卡斯和普雷斯科特谨慎地补充道，这种稳定性"与对经常反复发生的冲击的动态响应毫不相关"，这正是学习一个均衡与在某个均衡内作出响应之间的区别。{doc}`ls_learning` 研究了当主体用最小二乘法从数据中估计，而不是精确地计算 $E(y;\lambda)$ 时，会发生什么变化。

## 均衡内部

现在来看看均衡内部：单个市场、劳动力量的分布，以及相同工人所获得的工资。

### 一个市场是什么样子

两个阈值描述了一个市场行为的全部。

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

employs = eq['n_bar'] > 0                 # n̄(s) = 0 的状态不雇用任何人
wage_at_nbar = model.R[np.arange(len(model.s)), eq['n_bar']]
axes[1].plot(eq['n_bar'][employs], wage_at_nbar[employs], 'o-', lw=2)
axes[1].set_xlabel(r'employment $\bar n(s)$')
axes[1].set_ylabel('wage')
plt.tight_layout()
plt.show()
```

右图是卢卡斯和普雷斯科特所称的 $SS$ 曲线：如果一个市场的劳动力量从未对其构成约束，那么随着需求变动，它会描绘出的一系列就业和工资的组合。

它略去了 $\bar n(s) = 0$ 的需求状态，此处市场不雇用任何人、不支付工资。

这条曲线*并不是水平的*。一条水平的 $SS$ 曲线意味着工人拥有单一的保留工资，高于该水平就接受工作、低于就搜寻，而这正是 {doc}`mccall_model` 中局部均衡搜寻模型所*推导出*的东西。

在 $\bar n(s)$ 处，工资恰好使得留下的价值等于 $\lambda$，即 $w = \lambda - \beta \int v(s', \bar n, \lambda) f(s', s)\, ds'$（在整数网格精度下），因此它与延续值反向变动。

需求下降会以两种方向相反的方式改变延续值：

1. 今天较低的需求使得明天较低的需求更有可能发生，这使留下的价值降低；
1. 今天较低的就业意味着下一期更小的劳动力量，这使留下的价值提高。

第一种效应存在于具有持续性要约的局部均衡模型中，例如 {doc}`mccall_persist_trans`，其中保留工资取决于状态。

第二种效应来自其他工人的决策，只有均衡模型才有这种效应。二者中任何一个都可能占主导，因此 $SS$ 曲线可以朝任意方向倾斜。

在本校准中它向上倾斜，除了在市场几乎从不会出现的高需求状态附近的细网格上出现的一些小波动，所以第二种效应占主导：需求下降之后，较小市场中的延续值更高，维持其边际工人所需的工资更低。

均衡也直接展示了保留工资的缺失。

```{code-cell} ipython3
ψ, n, w = eq['ψ'], eq['n'], eq['w']
sheds = model.y[None, :] > eq['n_bar'][:, None]          # 区域 I：工人离开
occurs = (ψ > 1e-12) & (n > 0)                           # 实际发生且有人雇用的市场

print(f"lowest wage in a market that nobody leaves:  {w[occurs & ~sheds].min():.3f}")
print(f"highest wage in a market that workers leave: {w[occurs & sheds].max():.3f}")
```

在工资低至第一个数值的市场中，工人留任，无人离开；而在支付更高——即第二个数值——工资的市场中，工人却会离开，因此在这二者之间不存在一个分界工资，能把工人接受的市场与他离开的市场区分开来。

但离开者并不是为了某种更好的东西而拒绝一个要约，也没有人拒绝给予他们工作。在 $\bar n(s)$ 处，该市场中的每一位工人，无论留下还是离开，获得的都是 $\lambda$；离开者本可以留下，但那样一来就业就会是整个劳动力量 $y$，工资就会降至 $R(s, y)$，留下的价值就会低于 $\lambda$。

所以没有要约是相对于某个分界工资被拒绝的，也没有人在现行工资下被拒绝工作：真正的边际在于市场的就业水平。

### 劳动力分布与一条样本路径

由于到达是确定性的，且需求只取有限多个值，劳动力量在少数几个值之间移动，每个值都是某种需求状态的阈值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The invariant distribution of the workforce, and one market over sixty periods
    name: fig-lps-distribution
---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ψ_y = eq['ψ'].sum(axis=0)                      # 劳动力量的分布
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

右图用一幅图展示了该模型的机制：只要需求不变动，市场就处于静止状态。

当需求下降幅度足够大，使得 $\bar n(s)$ 降到劳动力量之下时，两条线之间的缺口就会打开，而且仅限于那一期：那一期的离开者就是失业者，到下一期他们就处于别处了。

较小幅度的下降不一定会打开缺口：一个只被填充到 $\bar y(s)$ 的市场，可以承受需求降至某个 $\bar n$ 仍高于其劳动力量的状态，而印出的次数统计展示了这种情况在该路径上发生的频率。

### 市场所处的位置

{eq}`lps_transition` 的三个区域构成了状态空间的一个划分，因此不变分布能够说明经济体有多大比例处于每个区域。

```{code-cell} ipython3
def where_markets_sit(model, r):
    "市场在各区域中的占比，以及一个市场的需求保持不变的频率。"
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

在任意给定的一期中，大多数市场完全静止不动。少数几个百分点的市场正在释放工人，经济体的全部失业都来自这些市场；大约相同比例的市场正在接纳工人。

需要平衡的是工人的流动，而它确实是平衡的：离开释放中市场的工人，恰好就是到达接纳中市场的工人，正如 {ref}`lps_ex_undirected` 所验证的那样。

第四行统计了完全没有工人的市场，第五行表明其中大多数都是静止的。

这样的市场是在需求降至 $\bar n(s) = 0$ 的某个状态时空置的，并且只要需求低到使 $\bar y(s)$ 无法为正，它就会一直空置下去；其余的市场正在被重新填充，因为需求已经回升。

这些占比取决于离散化方式。用七个需求状态，一个市场的需求有四分之三的时期保持不变，正如最后一行所示；这里给出了具有 25 个状态时的相同数字。

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

有 25 个状态时，需求变动的次数更多，但每次变动的幅度更小，释放和接纳工人的市场占比略有增加，但大多数市场仍然静止，失业率几乎没有变化。

### 工资分布

现在我们可以回到工资分布，概览部分曾提到它必须与工人的行为一并确定。搜寻文献给工人一个工资要约分布，并问他该如何应对它。

而这里，该分布是一个输出结果：每个市场都在其中支付劳动的边际产品，一个市场所支付的工资取决于它所抽到的需求，以及有多少工人选择留在那里。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Wages across identical workers, at the equilibrium
    name: fig-lps-wages
---
weight = eq['ψ']*eq['n']                     # 按某市场雇用的工人数对其赋权
weight = weight/weight.sum()

w, p = eq['w'].ravel(), weight.ravel()
order = np.argsort(w)
w, p = w[order], p[order]
cdf = np.cumsum(p)
quantile = lambda q: w[np.searchsorted(cdf, q)]
mean = (w*p).sum()
flow = λ_star*(1 - model.β)                # 搜寻的流量价值

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(w, bins=30, weights=p, color='C0', alpha=0.85)
ax.axvline(mean, color='k', lw=1.5, label=f'mean {mean:.3f}')
ax.axvline(flow, color='C1', lw=1.5, ls='--', label=f'flow value of search {flow:.3f}')
ax.set_xlabel('wage')
ax.set_ylabel('share of workers')
ax.set_xlim(0.5, 1.45)            # 一条细长的尾部延伸过此处；见下方打印内容
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

该经济体中的工人是完全相同的：他们具有相同的技能、相同的偏好和相同的信息，并且可以随时自由离开。然而他们获得的报酬却各不相同，第九十百分位的工资大约比第十百分位高出一半。

这种差异并非市场失灵或错误。它正是使得在接纳工人的任何市场中*到达*的价值相等、以及在释放工人的任何市场中留下或离开的价值相等所必需的东西。

而在其他所有地方，在职者都赚取租金，因为新工人需要一期时间才能到达，所以各个岗位的价值也存在分散。

```{code-cell} ipython3
held = eq['ψ']*model.y[None, :]              # 按某市场的劳动力量对其赋权
print(f"λ* = {λ_star:.3f},  average value of a worker's position = "
      f"{(held*eq['v']).sum()/held.sum():.3f}")
print(f"share of workers whose position is worth more than λ*: "
      f"{100*held[eq['v'] > λ_star + 1e-9].sum()/held.sum():.0f}%")
```

大约十分之九的工人所处的岗位对他们而言价值超过 $\lambda^*$。

图中的虚线是搜寻的流量价值 $(1-\beta)\lambda^*$，即搜寻者的处境每一期值多少。

大约三分之一的工人获得的报酬低于该值，并且仍然留任。确切的比例是脆弱的。

在该网格上，大约十一分之一的工人所获报酬与流量价值之差不足千分之一，正如前面打印内容的最后一行所示，所以网格的微小变化就能使整个这一区块跨过这条分界线；在我们尝试过的各个网格上，这一比例大约在 30% 到 40% 之间波动。

这其中并无费解之处：工人所持有的并非他目前的工资，而是一个特定市场中的一个岗位，该岗位附带着对该市场未来的一种求偿权，以及日后离开的选择权。

(lps_efficiency)=
## 效率

卢卡斯和普雷斯科特说 $\lambda$ 传递出"额外供给一个单位的正确社会回报"，但他们并未证明任何福利定理。

该模型确有这样一条定理，其证明用到了 {doc}`lucas_prescott_investment` 中研究过的 1971 年论文所采用的手段：找到一个由均衡求解的计划问题。

### 剩余

写出

$$
\Phi(s, n) = \sum_{m=1}^{n} R(s, m)
$$

作为边际收入表达式下方、直到就业量 $n$ 为止的面积。

根据 {eq}`lps_R`，它就是反需求曲线 $D(s, \cdot)$ 下方、直到该产业产出（按厂商计）为止的面积：即消费者总剩余，这正是 1971 年论文的目标函数。

它只有在产品价格给定时，即 $D(s, Q) = s$，才等于产出的价值，所以我们称之为**剩余**。

### 一个管理单一市场的计划者

考虑一个管理单一市场、可以让工人进出该市场的计划者，该计划者每释放一个工人进入搜寻者队伍就获得 $\lambda$ 的记入，每引入一个工人就被扣除 $\lambda$。

如果下一期的劳动力量是 $y'$，该计划者本期雇用 $\min\{y, y'\}$ 名工人，并释放或引入其差额，因此其价值满足

```{math}
:label: lps_planner
W(s, y) = \max_{y'} \Big\{ \Phi\big(s, \min\{y, y'\}\big) + \lambda\, (y - y')
+ \beta \int W(s', y')\, f(s', s)\, ds' \Big\} .
```

这里是**边际值论证**。$\Phi(s, \cdot)$ 是凹的，因为 $R$ 随 $n$ 下降，因此 $\Phi(s, \min\{y, y'\})$ 关于 $(y, y')$ 联合凹，对 $y'$ 求最大值保持了凹性。

因此 $W$ 关于 $y$ 是凹的，计划者的选择由边际比较所主导，这种比较用边际值 $\Delta W(s, y) = W(s, y) - W(s, y-1)$ 来进行。

把一个工人多留一期的价值是他的工资加上 $\beta \int \Delta W(s', y') f(s', s)\, ds'$，而释放他的价值是 $\lambda$；引入一个工人的成本是 $\lambda$，其价值仅是贴现延续值。

因此计划者释放工人直到由 {eq}`lps_nbar` 定义的阈值（以 $\Delta W$ 代替 $v$），填充工人直到以同样方式由 {eq}`lps_ybar` 定义的阈值，在其余情况下不去动劳动力量。

逐一来看，在计划者释放工人之处，边际工人的价值是 $\lambda$；在计划者填充之处，价值是 $R + \lambda$，因为它省下了一次到达的费用；在两者之间，价值是 $R + \beta \int \Delta W f\, ds'$。

这些正是情形 A、B2 和 B1，因此 $\Delta W$ 满足 {eq}`lps_bellman`。由 {prf:ref}`lps_prop_v` 中的唯一性，$\Delta W = v$，因此计划者的阈值即为均衡阈值，其策略为 {eq}`lps_transition`。

这里是**加总论证**。给每个市场都配备这样一个计划者，使用相同的 $\lambda$，然后将它们的目标函数相加。

项 $\lambda (y_t - y_{t+1})$ 相加起来等于 $\lambda$ 乘以经济体总劳动力量的下降量，因为一期中释放的每一个工人都会在下一期到达某处。

对任何可行计划而言，总劳动力量都是不变的，因此这些项加起来等于零，于是计划者们目标函数之和就是剩余的预期现值。

均衡策略使每个计划者的目标函数最大化，因此也使总和最大化。在 $\lambda = \lambda^*$ 处，它也是可行的，因为 $E(y; \lambda^*) = \mu$ 使总劳动力量保持不变。

因此，任何在工人之间进行再配置的可行方式，都不能产生比平稳均衡更高的剩余预期现值。

### 该定理说了什么，又没有说什么

精确地说，平稳均衡在以下条件下使**剩余的预期现值**最大化：

* 从其自身的平稳分布 $\Psi(\lambda^*)$ 出发，
* 在所有可行计划之中，包括非平稳的计划以及依赖于其他市场历史的计划，并且转移需要一期时间，
* 在一个连续统的市场中，使得均衡计划逐期保持经济体的劳动力量不变，
* 在整数网格精度范围内，$E(y; \lambda^*)$ 只在一个台阶之内接近 $\mu$，而非完全相等。

$\lambda^*$ 是向搜寻者队伍中增加一个工人的现值；按每期计算，经济体劳动力量固定这一约束的乘子是 $(1-\beta)\lambda^*$，即上述工资图中的搜寻流量价值。

该定理关于的是总剩余，无论由谁获得，它并未说明剩余是如何分配的：{ref}`lps_ex_cost` 找到了一种降低剩余却仍使工人境况改善的政策。

它也没有对各个稳态进行排序。其比较起点是 $\Psi(\lambda^*)$，并计入了通往任何其他配置途中发生的一切。

一项导致不同稳态的政策，可能会在该稳态中提高剩余，却降低其现值，因为到达那里的过程首先要消耗剩余。

{ref}`lps_ex_planner` 计算了两个例子，一个是对转移的小额补贴，另一个是比工人更有耐心的计划者所采用的阈值，二者都使稳态下人均剩余高于均衡水平。

这正是增长理论中**修正黄金法则**的逻辑：一个对未来进行贴现的计划者所选择的资本量，会少于使消费最大化的稳态资本量，因为建造额外的资本首先要消耗消费。

在这里，重新配置一个工人以牺牲他当期的产出为代价，而在以后才得到回报，因此一个进行贴现的计划者所进行的再配置更少，其失业也比一个使稳态剩余最大化的计划者更少。

这里的效率是相对于转移这一技术而言的，该技术需要一期时间，而均衡所具有的失业量恰好等于在以 $\beta$ 贴现未来的情况下，该技术所使之值得付出的代价。

## 与其他讲座的关系

在 {cite:t}`BaleyLjungqvistSargent2023` 的第一个脚注中指出，不存在摩擦性失业的"标准模型"，但有三类可行的模型：戴蒙德-莫滕森-皮萨里德传统中的匹配模型、{cite:t}`McCall1970` 的均衡版本，以及 {cite:t}`LucasPrescott1974` 传统中的搜寻-岛屿模型。

这三类模型的校准版本都能拟合劳动力市场流动数据，并对诸如失业保险和裁员税等政策下失业的反应给出合理的预测。

本讲座属于第三类，本系列中的麦考尔讲座研究的是第二类的核心——局部均衡下的工人问题。本模型为它们增添的是要约的另一方所涉及的一切。

工人所获得的要约工资，是某个市场中劳动的边际产品，而该市场的劳动力量是由其他工人选择去往何处所决定的，寻找别处的价值 $\lambda$，则是使整个经济的劳动力市场出清的那个数值。

一个结果是不存在保留工资，正如 $SS$ 曲线所展示的。另一个涉及就业与失业之间的流量比率。

{doc}`lake_model` 将这两个比率都视为参数：找工作率为 $0.283$（该讲座称之为 $\lambda$，不要与此处的搜寻价值混淆），离职率为 $0.013$，输入这两个值，失业率就作为输出结果得出。

{doc}`endogenous_lake` 把找工作率作为一个麦考尔问题的结果，但所针对的仍是一个给定的工资分布。

在本讲座中，离职率是一个均衡对象，而找工作率从构造上说本质上就是一个确定值。

上文打印出的离职率，约为每期在职工人中的 2.2%，综合了需求变动使市场降至其就业阈值以下的频率，以及每个这样的市场释放了多少工人。

```{warning}
本模型中的找工作率*本质上为一*。

离开的工人会在下一期开始时被安置到某个市场中，因此一次失业的持续时间为一期，该模型对失业*持续时间*无话可说。

严格来说，一个接纳搜寻者的市场，其需求可能大幅下降，以至于它释放的工人比到达前所拥有的工人还多，于是部分新到达者不得不再次转移；在本校准中，即使新到达者是最先被释放的，这种情况发生的概率也不到百分之一。
```

### 岛屿模型的后续发展

{cite:t}`AlvarezVeracierto1999` 采纳了这一模型并将其变成了量化政策分析的工具，用他们自己的话说，是通过"修改卢卡斯和普雷斯科特（1974）的岛屿模型，纳入无方向搜寻和劳动力退出参与"。

四件事发生了变化。搜寻变为**无方向的**：搜寻者"均匀地分布到所有岛屿"，因此一个岛屿下一期的劳动力量就是其自身的就业量加上其他每个岛屿所获得的同样数量的流入，无论它是否需要这些工人。

于是一个搜寻者可能落入一个自身正在释放工人的市场，不得不再次转移，因此失业的持续时间就成为一个均衡对象；在他们的自由放任基准情形中，该值平均为 2.4 期。

未就业者获得了第二种选择——家庭生产，因此劳动力参与成为一个边际选择，模型有三种状态而非两种。

有了家庭生产，搜寻者的流入量由搜寻与留在家中二者之间的无差异条件所确定，因此劳动力规模就成为一个结果，这一点很重要，因为他们所研究的政策会作用于人们是否完全去寻找工作这一选择上。

岛屿以报酬递减的技术进行生产，受制于持续性的生产率冲击，而非面对一条移动的需求曲线。并且主体交易一整套完备的或有债权证券。

他们的自由放任均衡"是有效的（尽管存在搜寻摩擦）"——这是对应于他们经济体的 {ref}`lps_efficiency` 的类比——他们将其作为评估最低工资、两种工会模型、裁员税和失业保险的基准，所采用的模型期长为六周。

例如，一项相当于一年平均工资十二个月份额的裁员税，会使他们的失业率从 5.3% *下降*到 3.7%，并使平均失业持续时间从 2.4 期*上升*到 5.1 期，同时使福利降低相当于消费的 2.1% 到 2.3%。

```{note}
无方向搜寻改变了 $\lambda$ 的确定方式。

在定向搜寻下，$\lambda$ 是一个*价格*：提高它，每个市场就会更早地释放工人、填充程度更低，因此 {eq}`lps_equilibrium` 像市场出清条件确定任何其他价格一样将其确定下来，而 {eq}`lps_bellman` 中的 $\min$ 使每个接纳工人的市场对他们而言恰好值 $\lambda$。

在无方向搜寻下，搜寻者落入一个随机抽取的市场，因此 $\lambda$ 需要其自身的一个一致性条件，共同流入量 $U$ 成为第二个未知数。

在人口固定的情况下，该模型依然是确定的：存在两个条件——人口条件和 $\lambda$ 的一致性条件——对应两个未知数。

在本校准中，一旦流入量可以取分数值，均衡就存在且唯一，但一般而言，唯一性不再能够仅由单一一条递减曲线得出。

{ref}`lps_ex_undirected` 对其进行了计算：无方向搜寻降低了搜寻的价值，也降低了失业。
```

### 劳动力流动的回报

卢卡斯和普雷斯科特在对转移成本的讨论末尾提出了一个猜想："这种实际或感知到的换工作成本上的差异，有助于解释不同职业之间以及不同国家之间所观察到的平均失业差异"。

为何各国的平均失业率存在差异，成为这类模型的核心问题；例如，{cite:t}`LjungqvistSargent1998` 将欧洲失业率的上升追溯到慷慨的失业救济与更动荡的经济环境相互作用的结果。

{cite:t}`BaleyLjungqvistSargent2025` 称之为**劳动力流动的回报**——即一个工人在其要去往的地方比在其目前所在之处多产出多少——是"所有具有摩擦性劳动力市场的现代宏观经济模型中重要的中介力量"。

{cite:t}`BaleyLjungqvistSargent2023` 表明，同一个量决定了裁员税对失业的影响，也决定了辞职时的技能损失是否能够逆转动荡对失业的影响。这是冲击本身和传导该冲击的技术二者共同作用的一个属性。

{cite:t}`BaleyLjungqvistSargent2025` 发现，针对企业规模动态的校准——冲击通过新古典生产函数传导——所得出的劳动力流动回报对参数的扰动是稳健的，而针对劳动力市场统计数据所校准的人均生产率过程则可能是脆弱的。

他们所研究的模型之一是 {cite:t}`AlvarezVeracierto2001` 的模型，该模型根据戴维斯和哈尔蒂万格的企业层面就业创造与破坏数据校准了一个生产率过程。

那篇论文是与 {cite:t}`AlvarezVeracierto1999` 不同的一个模型：企业在一个位置随机变化的共同劳动力市场上雇用工人，工资合同是刚性的，风险厌恶的主体在没有保险市场的情况下自我保险并选择搜寻的努力程度。

一家企业用资本 $k$ 和劳动 $n$ 生产 $z\, k^{\theta_k} n^{\theta_n}$，生产率为 $z$，且 $\theta_k + \theta_n < 1$。

{cite:t}`BaleyLjungqvistSargent2025` 对该模型进行了简化，去掉了建立一家企业的成本和一项小额裁员税，使得每家企业都在现货市场上租用资本和雇用劳动，它们都使边际产品相等，由此给出

```{math}
:label: lps_av_ratio
\frac{n_{\text{high}}}{n_{\text{low}}}
= \left(\frac{z_{\text{high}}}{z_{\text{low}}}\right)^{1/(1-\theta_k-\theta_n)} ,
```

因此一个冲击会被一个随规模报酬趋近于一而爆炸式增长的指数所放大。

在阿尔瓦雷斯和维拉西耶托的参数设定下，他们发现一家低生产率企业所雇用的工人数量仅为高生产率企业的 3.81%，比率约为 26，而高生产率企业拥有全部就业的 96% 以上：约 4.35 的指数把 2.12 的生产率比率转化为约 26 的就业比率。

这是一种无摩擦的配置，因为每家企业都在同一个现货市场上雇用劳动。而在本模型中，转移需要一期时间，{ref}`lps_ex_two_states` 探讨了在保持阿尔瓦雷斯和维拉西耶托的技术与冲击不变的情况下，像 26 这样的比率有多少能在这种摩擦下存续。

## 练习

```{exercise}
:label: lps_ex_persistence

{cite:t}`LucasPrescott1974` 从他们自己的数值例子中报告了两条比较静态结论：

> *总体而言，我们发现在需求持续性低和高两种情形下，失业率都较低。在前一种情形中，再配置工人几乎没有收益；而在后一种情形中，再配置很少发生。正如预期的那样，在持续性程度固定的情况下，需求的波动性越大，失业水平越高。*

在本校准中，保持 $\mu = 50$ 不变，检验这两点。

1. 在保持 $\log s$ 的*平稳*标准差固定为 $0.26$ 的情况下，计算持续性 $\rho \in \{0.6, 0.8, 0.9, 0.95, 0.99\}$ 时的均衡失业率；并在保持 $\rho = 0.9$ 的情况下，计算平稳标准差为 $\{0.10, 0.18, 0.26, 0.34, 0.45\}$ 时的均衡失业率。解释你所发现的形状。
1. 当需求独立，即 $\rho = 0$ 时，`equilibrium_λ` 会报错停止。证明此时失业恰好为零，且均衡的 $\lambda$ 不唯一。为什么这与论文的命题 5 以及其关于每期失业都为正的论断并不矛盾？
1. 在 $\rho = 0.3$ 时重复第 2 部分，先用七个需求状态，再用 25 个。这一比较对第 1 部分的低端结果说明了什么？
```

```{solution-start} lps_ex_persistence
:class: dropdown
```

在 $\rho$ 变化时保持平稳标准差不变，正是 `create_model` 的参数 `sd` 已经完成的工作，因此每个实验都只是一次调用。

```{code-cell} ipython3
def u_rate(model, μ):
    "均衡失业率。"
    r = market_outcomes(model, equilibrium_λ(model, μ))
    return r['unemployment']/r['workforce']

for ρ in (0.6, 0.8, 0.9, 0.95, 0.99):
    print(f"ρ = {ρ:<5}  unemployment rate {100*u_rate(create_model(ρ=ρ), μ):.2f}%")
print()
for sd in (0.10, 0.18, 0.26, 0.34, 0.45):
    print(f"sd = {sd:<5} unemployment rate {100*u_rate(create_model(sd=sd), μ):.2f}%")
```

波动性所带来的结果正如预期：移动幅度更大的需求会移动更多的工人，失业率随之稳步上升。

而持续性则不然：失业在 $\rho$ 约为 0.8 到 0.9 时升至峰值，而在 $\rho = 0.99$ 时又降至峰值的约四分之一。这正是卢卡斯和普雷斯科特所描述的形状，两端的低值出于不同的原因。

当需求几乎逐期独立时，今天运气不好的市场预计明天会恢复平均水平，因此留任的价值几乎不依赖于今天的需求，到达阈值 $\bar y(s)$ 几乎不随 $s$ 变动，很少有工人会被释放；第 2 部分将把这一点推向极限。

当需求几乎是永久不变的，它从一期到下一期的变化很小：在七状态链上它很少变化，每次变化一整档；在更细的链上变化更频繁，但每次幅度更小。无论哪种情形，任一期内都很少有工人有理由转移，这正是论文所说的"再配置很少发生"。

失业需要的是*既*值得作出反应*又*频繁发生的需求变动，这种组合只有中等程度的持续性才能带来。

对于第 2 部分，`equilibrium_λ` 之所以报错，是因为不变分布不唯一，这正是 {ref}`lps_regions` 注释中的第三种情形。因此我们转而寻找使得持有 $\mu$ 名工人的市场永不变动的那些 $\lambda$ 值，即 $\max_s \bar y(s) \le \mu \le \min_s \bar n(s)$。

两个阈值都随 $\lambda$ 上升而下降，因此该集合的每一端都可以通过二分法求出。

```{code-cell} ipython3
def rest_interval(model, μ, lo=0.01, hi=40.0):
    "使得持有 μ 名工人的市场永不变动的 λ 区间，若不存在则返回 None。"
    def ends(λ):
        n_bar, y_bar = thresholds(model, worker_value(model, λ)[0], λ)
        return y_bar.max(), n_bar.min()
    λ_a = bisect(lambda λ: ends(λ)[0] <= μ, lo, hi)     # 没有市场填充超过 μ
    λ_b = bisect(lambda λ: ends(λ)[1] < μ, lo, hi)      # 有市场在低于 μ 时释放工人
    return (λ_a, λ_b) if λ_a < λ_b else None

m_iid = create_model(ρ=0.0)
n_bar_iid, y_bar_iid = thresholds(m_iid, worker_value(m_iid, 7.1)[0], 7.1)
print("rows of P identical:", np.allclose(m_iid.P, m_iid.P[0]))
print("at λ = 7.1:  n̄(s) =", n_bar_iid, "  ȳ(s) =", y_bar_iid)
print("λ at which a market holding μ never moves: from {:.3f} to {:.3f}".format(
      *rest_interval(m_iid, μ)))
```

在需求独立的情况下，$P$ 的每一行都相同，因此延续值 $\beta \sum_{s'} v(s', y) P(s, s')$ 不依赖于 $s$，$\bar y$ 也是如此。

由于在每一种状态下都有 $\bar y(s) \le \bar n(s)$，这个共同的 $\bar y$ 低于每一个就业阈值。因此 $\bar y$ 与 $\min_s \bar n(s)$ 之间的每一个劳动力量在每种需求状态下都保持原地不动，各个市场逐渐漂移进入该区间然后停止，长期来看无人移动，失业恰好为零。

该区间中的每一个劳动力量都是吸收态，因此不变分布不唯一，所打印区间中的每一个 $\lambda$ 都支撑一个使每个市场持有 $\mu$ 名工人的均衡。

该区间的宽度是离散化链的一个性质：随着状态的增加它会收窄，因为最差的需求状态更向外延伸，其就业阈值向共同的 $\bar y$ 靠拢下降。但失业仍保持为零，因为每个 $\bar n(s)$ 都至少等于 $\bar y$。

这一切都不与论文相矛盾，起作用的关键差异在于 $x \equiv 1$。命题 5 所说的唯一 $\lambda$，以及每期失业都为正的论断，都依赖于命题 3 所给出的唯一平稳密度，这是由随机到达通过使部分市场超额填充所实现的，而在确定性到达的情形下，该论证就不成立了。

需求独立也违反了论文所坚持的 $F(s', s)$ 关于 $s$ 的严格单调性，并且其脚注 7 中的例子要求 AR(1) 系数严格为正。

对于第 3 部分，在 $\rho = 0.3$ 处用两种网格进行同样的检验给出了不同的答案。

```{code-cell} ipython3
for n_s in (7, 25):
    m = create_model(n_s=n_s, ρ=0.3)
    rest = rest_interval(m, μ)
    if rest is not None:
        print(f"{n_s:>2} states: at rest for λ from {rest[0]:.3f} to {rest[1]:.3f}")
    else:
        print(f"{n_s:>2} states: unique λ*, unemployment rate {100*u_rate(m, μ):.3f}%")
```

在七个状态下，0.3 的持续性不足以使到达阈值的分散超过最低的就业阈值，因此经济体处于静止状态，就如同 $\rho = 0$ 时一样。而用 25 个状态时，极端需求状态更向外延伸，均衡是唯一的，失业为正但极小。

所以在 $\rho = 0.3$ 时，两种网格下失业率都很低，正如论文所言，但它是否恰好为零，是离散化链的一个性质。

离散化方法在另一端的影响更大。

```{code-cell} ipython3
def chain_moments(m):
    "log s 的标准差和自相关系数，以及 s 保持不变的概率。"
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

Rouwenhorst 的链具有恰好等于目标值的标准差 $0.26$ 以及目标自相关系数。而 Tauchen 的七状态链在 $\rho = 0.9$ 时波动性过大，这使失业率抬高了约四分之一。

在 $\rho = 0.99$ 时，它不仅波动性过大，而且持续性也严重过高：其需求状态几乎从不变化，因此市场几乎从不进行再配置，失业率几乎消失。

用七个状态时，这会使第 1 部分的右端看起来像左端；而用 25 个状态时，两种方法几乎一致。

```{solution-end}
```

```{exercise}
:label: lps_ex_cost

在结语部分，卢卡斯和普雷斯科特考虑了离开一个市场去搜寻所需的一次性成本 $c$，使得 {eq}`lps_nbar` 的右边由 $\lambda$ 变为 $\lambda - c$：一个离开的工人获得的是扣除了到达那里所需成本之后的净搜寻价值。

到达条件 {eq}`lps_ybar` 不变，因为它为已经处于搜寻中的工人所面对的市场定价。

他们说这一成本"会在图 3 中抬高曲线 $y = \bar n(s, \lambda)$，并使图 4 中的'需求曲线'向下移动。结果是失业下降，且均衡的工资现值 $\lambda$ 也下降"，并评论道，这

> *表明较低的平均失业，一般而言，并不与工人更高的福利相关联。*

`market_outcomes` 和 `equilibrium_λ` 已经把该成本作为其参数 `c`。

对 $c \in \{0, 0.2, 0.5, 1, 2\}$，报告均衡的 $\lambda^*$ 和 $\lambda^* - c$、失业率、每个市场的剩余、每个市场的工资总额、租金（剩余减去工资总额），以及每名工人每期的收入——工资总额扣除转移所支付的成本后，除以劳动力总量。

1. $\lambda^*$ 是否如论文所说下降？如果不是，那么下降的是什么？
1. 较低的失业是否意味着境况更好？将工人所获得的与经济体所损失的，作为每名工人的现值进行比较。
1. 谁为此买单？分别在将 $c$ 视为资源（在转移过程中被耗费）和将 $c$ 视为一项税收（其收入以一次性方式返还给工人）两种情形下作答。
1. {cite:t}`BaleyLjungqvistSargent2023` 用能够完全消除内生离职的最小裁员税作为检验模型劳动力流动回报的一种诊断工具。计算此处使之成立的最小 $c$，以平均工资为单位，分别用 7、15 和 25 个需求状态。为什么它会随状态数而变化？
```

```{solution-start} lps_ex_cost
:class: dropdown
```

在进行扫描之前，先看看该成本在自由放任搜寻价值下对劳动力需求的影响。

```{code-cell} ipython3
λ_0 = equilibrium_λ(model, μ)
for c in (0.0, 2.0):
    n_bar_c, y_bar_c = thresholds(model, worker_value(model, λ_0, c)[0], λ_0, c)
    print(f"c = {c}: workforce demand {workforce_demand(model, λ_0, c):5.1f},  "
          f"n̄(s) = {n_bar_c},  ȳ(s) = {y_bar_c}")
```

该成本提高了劳动力需求，所以新的 $\lambda^*$ 高于旧的，我们从那里开始每次二分查找。这也使之远离较低的 $\lambda$，在那里有成本时不变分布并不唯一，二分查找会报错停止。

在平稳分布中，每个失业的工人本期恰好都离开过一次他所在的市场，因此工人为每个市场的失业人数支付 $c$ 倍的转移成本。

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

剩余、工资和租金都是每个市场的流量，而收入是每名工人的流量。为了比较工人所获得的和经济体所损失的，我们将两者都换算为每名工人的现值，即把每名工人的流量除以 $1 - \beta$。

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

对于第 1 部分，卢卡斯和普雷斯科特说该成本会使他们图 4 中的需求曲线下移并降低 $\lambda$，但此处的结果恰好相反。

在自由放任的 $\lambda^*$ 处，该成本使劳动力需求提高了约一半。它在每一个有人工作的需求状态下都大幅提高了就业阈值 $\bar n(s)$，因为一个本会离开的工人现在还要放弃 $c$，同时它也略微降低了到达阈值。

这正是论文自身的第一步，"抬高曲线 $y = \bar n(s, \lambda)$"，而在图 4 中纵轴为 $\lambda$ 的情况下，这会使需求曲线上移而非下移。

当市场想要的工人数量超过经济体现有数量时，$\lambda^*$ 必然上升，而它确实上升了。下降的是 $\lambda^* - c$，即扣除成本后离开这一选择权的价值，这正是一个正在决定是否留任的工人所比较的数值，而论文的论断对这个数值成立，而非对 $\lambda^*$ 本身成立。

对于第 2 部分，成本为二时，所度量的失业削减了近十分之九。若按扣除转移成本后的收入来衡量，工人从 $c = 0.5$ 起就明显获益。

对于更小的成本，他们为转移所支付的费用几乎抵消了额外的工资，因此所获净收益接近于零，在本网格上，$c = 0.2$ 时甚至略为负值。

经济体所损失的超过了工人所获得的：在 $c = 2$ 时，剩余下降的幅度约为工人收入上升幅度的两到两倍半。这些都是稳态的比较，不包含从一个稳态到另一个稳态的过渡过程。

因此较低的失业并非一个福利指标，但这一教训以不同于论文的形式呈现出来：卢卡斯和普雷斯科特笔下的工人境况变差是因为 $\lambda$ 下降，而我们这里的工人境况更好，整个经济体却更差。

该成本所消除的失业是*生产性*的。那些工人在任何有意义的层面上都不是闲置的；他们正处于从边际产品已经下降的市场转移到边际产品很高的市场的途中，而该成本使他们留在原地。

对于第 3 部分，工资总额略有上升，而租金——剩余减去工资总额——下降了约五分之一。

剩余是 $R$ 下方的面积，所以租金就是企业的利润，加上在产品需求向下倾斜的情形下，该市场商品购买者的消费者剩余，而正是这些人为此买单。

账目如何平衡取决于 $c$ 是什么。如果它是一种真实资源，上面所打印的转移成本也同样被消耗掉了，在损失的剩余之上额外产生，由工人承担。

如果它是一项以一次性方式返还给工人的税收，那么除了损失的剩余之外没有任何东西被消耗掉，而返还会在不改变任何人决策的情况下，将同样的转移成本每期都返还给工人，因此工人获益更多。

这两种情形在损失的剩余所意味的东西上也有所不同。{ref}`lps_ex_planner` 表明，在资源成本的情形下，均衡就是针对那种代价更高的转移技术的计划者配置，所以较低的剩余是更差技术所付出的代价，而非低效率；而在返还税收的情形下，技术并未改变，均衡不再是该技术下的计划者配置，所以较低的剩余是一种效率损失。

对于第 4 部分，回顾 {ref}`lps_regions` 中的注释。离职恰好在存在某个 $\lambda$ 使得持有 $\mu$ 名工人的市场永不变动时停止，即没有任何状态的到达阈值高于 $\mu$，也没有任何状态的就业阈值低于 $\mu$。

某一成本会提高就业阈值，因此我们寻找使这样的 $\lambda$ 存在的最小成本。

一旦离职停止，每个市场就永远持有 $\mu$ 名工人，因此需求状态为 $s$ 中的一名工人持有 $v_s$，其中 $v = (I - \beta P)^{-1} R(\cdot, \mu)$，当每个 $s$ 都满足 $\beta (Pv)_s \le \lambda \le v_s + c$ 时，没有人离开或到达。

这给出了一个可与计算结果对照的公式，$c_{\min} \approx \beta \max_s (Pv)_s - \min_s v_s$，它忽略了整数网格。

```{code-cell} ipython3
def at_rest_values(model, μ):
    "当每个市场永远持有 μ 名工人时，一个岗位的价值。"
    return np.linalg.solve(np.eye(len(model.s)) - model.β*model.P, model.R[:, μ])

def min_cost(model, μ, lo=1.0, hi=20.0, c_hi=20.0):
    "使某个 λ 使持有 μ 名工人的市场保持静止的最小 c。"
    def at_rest(c):
        ends = lambda λ: thresholds(model, worker_value(model, λ, c)[0], λ, c)
        λ_a = bisect(lambda λ: ends(λ)[1].max() <= μ, lo, hi, tol=1e-3)  # 没有人填充超过 μ
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

这里用作单位的工资，是当每个市场都持有 $\mu$ 名工人时的平均工资，这正是离职停止之后经济体的样貌；它与自由放任的平均工资相差不到半个百分点。

能够停止所有离职的成本会随着状态数增加而增长，公式解释了原因：它由一个市场所能处于的最好和最差需求状态之间的差距所决定，而 Rouwenhorst 链把其极端状态置于 $\log s$ 中的 $\pm 0.26 \sqrt{n_s - 1}$ 处，因此随着状态增加，它们会相互分离得更远。

$\log s$ 本身是一个具有无界支撑的 AR(1) 过程，所以对连续需求的模型而言，没有任何有限的成本能够停止所有离职：这里计算出的数值衡量的是离散化需求过程的支撑范围。

对于 {ref}`lps_ex_two_states` 中的对称两状态链——其中需求保持不变的概率为 $\rho$，$R(s,n) = s\, g(n)$——该公式有一个闭式解，

$$
\frac{c_{\min}}{\bar w}
= -1 + \frac{s_{\text{high}} - s_{\text{low}}}{s_{\text{high}} + s_{\text{low}}}
\cdot \frac{1 + \beta(2\rho - 1)}{1 - \beta(2\rho - 1)} ,
$$

在那里约为 1.05 期的工资，它取决于需求过程和 $\beta$，但不取决于 $g$ 的形状：在本模型中，该诊断指标度量的是需求的分散程度和持续性，而非 $R$ 的曲率。

本校准并未说明一期是多长；在 $\beta = 0.9$ 下，自然地可以理解为一年，这将使得每一次失业都持续一年。作为对照，在 {cite:t}`BaleyLjungqvistSargent2023` 作为其基准所用的永奎斯特-萨金特匹配模型中，相当于一年工人平均产出 163% 到 184% 的裁员税能够停止所有内生离职，其模型期长为半个季度的一半。

贝利、永奎斯特和萨金特的观点是，这个数字必须*很大*才能令人信服，因为在裁员成本差异巨大的各个经济体中，都能观察到广泛的再配置。

```{solution-end}
```

```{exercise}
:label: lps_ex_two_states

{cite:t}`LucasPrescott1974` 附录中的数值例子仅使用两个需求状态，取

$$
P = \begin{bmatrix} \rho & 1 - \rho \\ 1 - \rho & \rho \end{bmatrix} ,
\qquad \rho = 0.9,
\qquad s \in \{1, 2\},
\qquad \beta = 0.9 .
$$

这里 $\rho$ 是需求保持原地不动的概率，因此 $s$ 的自相关系数为 $2\rho - 1 = 0.8$；在七状态链中，$\rho$ 本身就是自相关系数。

构建该经济体，保持 $R(s,n) = s\,(1 - n/N)^{+}$ 且 $N = 200$，在需要均衡的任何地方取 $\mu = 50$。

1. 对 $\lambda$ 取一个网格，绘制劳动力需求和平均失业关于 $\lambda$ 的图，就像论文图 5 的中间和下方面板那样。
1. 在 $\lambda = 6$ 处，证明不变分布将全部概率集中在劳动力量的*两个*值上，并说明是哪两个值。
1. 利用这一事实，推导出一个以 $\bar n$、$\bar y$ 和 $\rho$ 表示每个市场平均失业的公式，在网格上的每一个 $\lambda$ 处用计算值进行检验，并用它来解释第 1 部分所得图形的形状。
1. 证明失业*率*只取决于 $\rho$ 和 $r = \bar y(s_{\text{high}})/\bar n(s_{\text{low}})$，并且它永远不能超过 $1 - \rho$。
1. 现在将 $R$ 替换为报酬递减技术的边际产品：产出为 $s\, n^{\alpha}$，$0 < \alpha < 1$，因此 $R(s, n) = s\,\alpha\, n^{\alpha-1}$。对 $\alpha \in \{0.4, 0.6, 0.75\}$ 求解，并针对一个类似阿尔瓦雷斯和维拉西耶托模型的经济体求解，其中 $\alpha = \theta_n/(1-\theta_k)$，$s_{\text{high}} = 2.12^{1/(1-\theta_k)}$，取他们的参数 $\theta_k = 0.19$、$\theta_n = 0.58$，这正是在资本（按期出租）已被最优选定之后的那种技术。报告两个需求状态下平均就业的比率，其对应的无摩擦版本 {eq}`lps_av_ratio`，以及失业率。证明平均就业的比率为 $\rho r + 1 - \rho$，用它表示失业率，并说明随着劳动力流动回报的上升，失业会发生什么变化。
```

```{solution-start} lps_ex_two_states
:class: dropdown
```

该两状态链只需要它自己的 $P$ 和 $s$。

```{code-cell} ipython3
def two_state_model(ρ=0.9, s_high=2.0, β=0.9, y_max=260, R=linear_R, **R_args):
    "论文中的两状态需求过程：需求保持不变的概率为 ρ。"
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

右图具有论文图 5 下方面板的形状：平均失业在 $\lambda$ 取中间值时最大。

对于第 2 部分，查看 $\lambda = 6$ 时不变分布的支撑。

```{code-cell} ipython3
r_six = market_outcomes(two, 6.0)
support = np.where(r_six['ψ'].sum(axis=0) > 1e-12)[0]
print("workforce values with positive probability:", support)
print("n̄(s) =", r_six['n_bar'], "   ȳ(s) =", r_six['y_bar'])
```

这两个值分别是 $\bar n(s_{\text{low}})$ 和 $\bar y(s_{\text{high}})$。要理解原因，可以沿着 {eq}`lps_transition` 来跟踪。

在低需求状态下，劳动力量被截断到 $[\bar y(s_{\text{low}}), \bar n(s_{\text{low}})]$ 之间，在高需求状态下被截断到 $[\bar y(s_{\text{high}}), \bar n(s_{\text{high}})]$ 之间，而在这个 $\lambda$ 下，第二个区间完全位于第一个区间之上。

因此在低状态经历一期，就会把任何劳动力量带到 $\bar n(s_{\text{low}})$；在高状态经历一期，就会把它带到 $\bar y(s_{\text{high}})$，该市场就在这两者之间穿梭。在满足 $\bar n(s_{\text{low}}) < \bar y(s_{\text{high}})$ 的每一个 $\lambda$ 下（这包含了整个网格），同样的情况也成立。

对于第 3 部分，失业只在一次衰退的第一期为正：市场以 $\bar y(s_{\text{high}})$ 名工人进入该期，需求刚刚下降，就业变为 $\bar n(s_{\text{low}})$。

这一事件是"上一期状态为高、本期状态为低"，其平稳概率为 $\tfrac12 (1-\rho)$，因此

$$
E[u] = \tfrac12 (1-\rho)\,
\big[\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})\big] .
$$

我们在网格上的每一点检验该公式，从转移矩阵中读出 $\rho$。

```{code-cell} ipython3
ρ_2 = two.P[0, 0]
gap2 = np.array([r['y_bar'][1] - r['n_bar'][0] for r in res2])
print(f"largest difference between formula and computation: "
      f"{np.abs(0.5*(1 - ρ_2)*gap2 - Eu2).max():.1e}")

n_lo = np.array([r['n_bar'][0] for r in res2])
print(f"unemployment per market peaks at λ = {grid2[Eu2.argmax()]:.1f}; "
      f"the low-demand market is closed for λ ≥ {grid2[n_lo == 0].min():.1f}")
```

该公式也解释了这个隆起形状：平均失业是两个阈值之间的差距，乘以跨越该差距的概率。

在高 $\lambda$ 时，低需求市场是关闭的，$\bar n(s_{\text{low}}) = 0$，差距就只是 $\bar y(s_{\text{high}})$，而该值随 $\lambda$ 降低而上升。

一旦 $\lambda$ 低到使得低需求市场能够留住一些工人，$\bar n(s_{\text{low}})$ 随 $\lambda$ 下降而上升得比 $\bar y(s_{\text{high}})$ 更快，差距随之缩小。因此每个市场的失业在低需求市场停止关闭之处达到峰值，这正如打印内容的第二行所示。

对于第 4 部分，劳动力量在一半的时期为 $\bar n(s_{\text{low}})$，另一半为 $\bar y(s_{\text{high}})$，因此 $E[y] = \tfrac12 [\bar n(s_{\text{low}}) + \bar y(s_{\text{high}})]$，失业率为

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,
\frac{\bar y(s_{\text{high}}) - \bar n(s_{\text{low}})}{\bar y(s_{\text{high}}) + \bar n(s_{\text{low}})}
= (1-\rho)\, \frac{r - 1}{r + 1} ,
\qquad r = \frac{\bar y(s_{\text{high}})}{\bar n(s_{\text{low}})} .
$$

它随 $r$ 上升，趋向 $1 - \rho$，当低需求市场关闭时达到该值。

对于第 5 部分，低状态下的平均就业为 $\bar n(s_{\text{low}})$，高状态下的平均就业以概率 $\rho$ 为 $\bar y(s_{\text{high}})$，以概率 $1 - \rho$ 为 $\bar n(s_{\text{low}})$（即一次景气回升的第一期），因此平均就业的比率为

$$
\mathcal{E} = \frac{E[n \mid s_{\text{high}}]}{E[n \mid s_{\text{low}}]} = \rho\, r + 1 - \rho .
$$

将 $r = (\mathcal{E} - 1 + \rho)/\rho$ 代入第 4 部分的失业率公式，得到

$$
\frac{E[u]}{E[y]} = (1 - \rho)\,\frac{\mathcal{E} - 1}{\mathcal{E} - 1 + 2\rho} ,
$$

它随 $\mathcal{E}$ 上升。

只有 `R` 的构造方式发生变化：在整数网格上，第一名工人的边际产品 $s\,\alpha$ 是有限的，空市场不支付任何工资。工人在远离原点处的边际上很"便宜"，所以我们用更大的网格，并围绕使持有 $\mu$ 名工人的市场恰好停止吸引到达者的搜寻价值——以 {ref}`lps_ex_cost` 的记号表示即 $\beta \max_s (Pv)_s$——来确定 $\lambda^*$ 的区间。

```{code-cell} ipython3
def power_R(s, y, α):
    "技术 s n^α 的边际产品 s α n^(α-1)；R(s, 0) 被重置为 0。"
    return s[:, None]*α*np.maximum(y[None, :], 1)**(α - 1)

def solve_two_state(m, μ):
    "围绕处于静止状态经济体的搜寻价值进行区间搜索得到的均衡。"
    λ_rest = m.β*(m.P @ np.linalg.solve(np.eye(2) - m.β*m.P, m.R[:, μ])).max()
    r = market_outcomes(m, equilibrium_λ(m, μ, lo=0.5*λ_rest, hi=3*λ_rest))
    E_n = (r['ψ']*r['n']).sum(axis=1)/r['ψ'].sum(axis=1)   # 按状态计的平均就业
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

恒等式 $\mathcal{E} = \rho r + 1 - \rho$ 以及失业率公式在每一个经济体中都成立，正如各列所示。

"无摩擦"一列是 {eq}`lps_av_ratio` 在 $\theta_k = 0$、$\theta_n = \alpha$ 情形下的版本：使两个状态下的边际产品 $s\,\alpha\, n^{\alpha-1}$ 相等，得到 $n_{\text{high}}/n_{\text{low}} = (s_{\text{high}}/s_{\text{low}})^{1/(1-\alpha)}$。

类似阿尔瓦雷斯和维拉西耶托的经济体具有相同的无摩擦比率，约为 26，这与他们带资本的技术相同，因为将资本优化掉之后，$z\, k^{\theta_k} n^{\theta_n}$ 就变为 $z^{1/(1-\theta_k)} n^{\theta_n/(1-\theta_k)}$ 的某个倍数。那个只含劳动的形式正是此处恰当的对比对象，因为在本模型中摩擦只落在劳动身上。

对于线性表达式，无摩擦基准是一个角点解，而且仅仅是刚好而已。令 $2(1 - n_{\text{high}}/N) = 1 - n_{\text{low}}/N$ 并取 $n_{\text{high}} + n_{\text{low}} = 2\mu$，得到 $n_{\text{low}} = (4\mu - N)/3$，因此对每个 $\mu \le N/4$，低需求市场都会关闭，这恰好就是本讲座所用的 $\mu = 50$。

```{code-cell} ipython3
N = 200                                    # 两状态经济体的容量
for μ_f in (50, 55, 60):
    n_low = max((4*μ_f - N)/3, 0.0)
    ratio = (2*μ_f - n_low)/n_low if n_low > 0 else np.inf
    print(f"μ = {μ_f}: frictionless n_low = {n_low:.1f}, ratio {ratio:.1f}")
```

所以在 $\mu = 50$ 时无摩擦比率为无穷大，在略高于该值时仍然很大。

在每一个经济体中，该模型所产生的反应都远小于其无摩擦基准，正如一期的转移摩擦所应有的那样。一个无摩擦的经济体会不断移动工人直至各市场间边际产品相等；而在这里，工人只移动到使到达的价值等于 $\lambda$ 为止，到达在 $\bar y(s_{\text{high}})$ 处停止，因为需求可能会回落，而一次景气回升需要一期时间才能完全得到响应。

与之相当的比较是最后一行：赋予阿尔瓦雷斯和维拉西耶托的企业约 26 的就业比率的技术和冲击，在本模型一期转移的约束下给出约 5.6 的比率，尽管该冲击的持续性仍与他们的设定有所不同。

在 $s_{\text{high}} = 2$ 的各行中，就业反应随 $\alpha$ 上升而增大（在摩擦固定的情况下），失业率也随之上升。这并非巧合，而是本讲座的发现，而非贝利、永奎斯特和萨金特的发现。

本模型中的失业是工人用于转移所花费的时间，而转移带来的收益越大，转移的人就越多：在摩擦固定的情况下，一个劳动力流动回报更高的经济体会展现出更多的再配置，也会记录下与之相伴的失业。

```{solution-end}
```

```{exercise}
:label: lps_ex_undirected

本练习探讨该模型的定向搜寻到底在做什么。

{cite:t}`AlvarezVeracierto1999` 将其替换为**无方向**搜寻：搜寻者无法选择去往何处，每个市场获得相同的流入量 $U$，即经济体每个市场的失业量。

1. 证明在无方向搜寻下，{eq}`lps_bellman` 中的 $\min$ 消失了，并说明它原本在强制执行哪一个经济学条件。
1. 写出一个市场劳动力量的转移规律，并证明在任何平稳分布下，流入量 $U$ 都等于平均流出量。在定向搜寻下这一恒等式是否也成立？
1. 在定向搜寻下，{eq}`lps_equilibrium` 确定了 $\lambda$。在无方向搜寻下，$\lambda$ 必须满足什么条件？在固定每个市场 $\mu$ 名工人的情况下，数出均衡的未知数个数和条件个数。
1. 计算本讲座经济体（$\mu/N = 1/4$）的无方向均衡。流入到容量为 $N = 200$ 的各市场中的整数流入量过于粗糙，因此以十分之一为单位计算工人数：用 $N = 2000$ 和上限为 $2600$ 的网格，这是本讲座经济体在十倍精细劳动力网格上的版本。对流入量 $U = 1, 3, 5, 7, 9$ 和 $10$，求出满足第 3 部分一致性条件的 $\lambda$ 值及其所服务的平均劳动力量，并绘制平均劳动力量关于 $U$ 的图。给定 $\mu$ 的经济体有多少个均衡？在同一网格上，比较 $\mu/N = 1/4$ 处定向搜寻下的 $\lambda$ 和失业率。
1. 解释为什么无方向搜寻在此降低了搜寻的价值，以及为什么它也降低了失业。
```

```{solution-start} lps_ex_undirected
:class: dropdown
```

对于第 1 部分，{eq}`lps_bellman` 中的 $\min$ 是竞争在搜寻者之间所施加的约束，即一个市场内的延续值不能超过 $\lambda$。一旦剥夺了他们选择去往何处的能力，就没有任何东西能强制执行这一约束。

一个留下的工人只是简单地获得工资以及其所在市场恰好带来的延续值，

$$
v(s, y) = \max\left\{\lambda,\; R(s, y)
 + \beta \int v\big(s', y + U\big) f(s', s)\, ds'\right\} ,
$$

该延续值可能高于或低于 $\lambda$，因为没有人能够基于这一差异采取行动。

对于第 2 部分，就业仍为 $n(s,y) = \min\{\bar n(s), y\}$，未就业者离开，流入量在各处相同，因此

$$
y' = n(s, y) + U .
$$

在平稳分布下，$E[y'] = E[y]$，因此对各市场求平均得到

$$
U = E[y] - E[n(s,y)] = E[\,y - n(s,y)\,] :
$$

流入量等于平均流出量。

这是一个会计恒等式，在定向搜寻下也同样成立：在 {eq}`lps_transition` 的平稳分布下，到达填充中市场的工人等于离开释放中市场的工人，这在每一个 $\lambda$ 值下都成立，而不仅仅在 $\lambda^*$ 处成立。

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

所以这一恒等式不能作为区分二者的依据；真正的区别在于 $\lambda$ 是如何确定的。

对于第 3 部分，在无方向搜寻下，一个搜寻者会落入从不变分布中随机抽取的一个市场，因此搜寻的价值必须满足

$$
\lambda = \beta \sum_{s,y} \psi(s, y)\, v(s, y) ,
$$

即落入一个随机抽取市场的贴现价值，一个按市场加权的平均值。

因此，一个人口固定的均衡有两个未知数，$\lambda$ 和 $U$，以及两个条件：$\mu = \sum_{s,y} \psi(s,y)\, y$ 和上面的条件。

在定向搜寻下，$\min$ 使每个接纳工人的市场对他们而言恰好值 $\lambda$，因此 $\lambda$ 不需要单独的一致性条件，而 $\mu$ 独自通过一条向下倾斜的曲线就确定了它。而在这里，$\lambda$ 需要其自身的一个条件，而没有什么能使这个条件关于 $\lambda$ 单调。

对于第 4 部分，一个市场的就业阈值是使留下仍然值 $\lambda$ 的最大劳动力量，前提是下一期的劳动力量将是 $n(s, y) + U$。

不变分布是唯一的，因为一个经历过最差需求状态的市场，最终都会从任意起点持有 $\bar n(s_1) + U$ 名工人，因此我们通过对其进行迭代来求得它。

```{code-cell} ipython3
def undirected_market(model, λ, U, tol=1e-10, max_iter=100_000):
    """一个每期都收到 U 名搜寻者的市场，无论它是否想要他们。
    返回隐含的搜寻价值、平均劳动力量，以及 n̄(s)。"""
    P, R, β, y = model.P, model.R, model.β, model.y
    n_s, n_y = R.shape
    up = np.minimum(y + U, y[-1])                  # 若无人离开，下一期的劳动力量
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
    # 对 y' = min(y, n̄(s)) + U 下 (s, y) 的分布进行迭代
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

隐含搜寻价值与 $\lambda$ 之差，原来在两个根之间为正、在之外为负，因此对每个流入量，我们在该差为正的某个值的两侧各寻找一个根。

在本网格上，在每一个存在根的流入量下，定向搜寻的 $\lambda^*$ 正是这样一个取正值的点，而 SciPy 中的 `brentq` 若区间不满足条件则报错停止。

```{code-cell} ipython3
def consistent_λs(model, U, λ_mid, lo=0.1, hi=12.0):
    "λ_mid 上下两侧 λ = β Σ ψ v 的根，连同各自的平均劳动力量。"
    gap = lambda λ: undirected_market(model, λ, U)[0] - λ
    roots = (brentq(gap, lo, λ_mid, xtol=2e-3), brentq(gap, λ_mid, hi, xtol=2e-3))
    return [(λ, undirected_market(model, λ, U)[1]) for λ in roots]

N_fine = 2000                            # 以十分之一为单位计算的本讲座经济体
m_fine = create_model(N=N_fine, y_max=13*N_fine//10)
λ_dir = equilibrium_λ(m_fine, N_fine//4)
r_dir = market_outcomes(m_fine, λ_dir)

locus = {U: consistent_λs(m_fine, U, λ_dir) for U in (1, 3, 5, 7, 9)}
gaps_10 = [undirected_market(m_fine, λ, 10)[0] - λ for λ in np.arange(6.0, 8.6, 0.5)]
print(f"at U = 10 the largest gap on a grid of λ is {max(gaps_10):.4f}")
```

这些流入量中的每一个都有两个解，每个解各自服务于自己的人口；在 $U = 10$ 时则没有解。

```{code-cell} ipython3
Us = np.array(list(locus))
scale = 200/N_fine                       # 换算回本讲座的单位
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

在给定的流入量下，这两个解属于*不同*的人口：一个是搜寻价值较低的拥挤经济体，另一个是搜寻价值较高的稀疏经济体。沿着每一条分支，人口都随流入量单调变化，两条分支在 $U = 9$ 到 $U = 10$ 之间的一个折点相遇，它们各自所对应的 $\mu$ 范围并不重叠。

因此在本校准中，每个人口恰好对应一个无方向均衡 $(\lambda, U)$，所需的流入量随 $\mu$ 上升，直到折点为止，之后则随之下降。

该折点也解释了为什么 $N = 200$ 不够用：最大的均衡流入量略低于每个容量为 200 的市场一名工人，因此即使该网格上的 $U = 1$ 也已经超过了它。本讲座的经济体是最接近 $\mu = 50$ 的那个点。

```{code-cell} ipython3
U_eq, k_eq = min(((U, k) for U in locus for k in (0, 1)),
                 key=lambda Uk: abs(locus[Uk[0]][Uk[1]][1] - N_fine//4))
λ_und, y_und = locus[U_eq][k_eq]
print(f"undirected: U = {U_eq}, λ = {λ_und:.3f}, workforce {scale*y_und:.2f}, "
      f"unemployment rate {100*U_eq/y_und:.2f}%")
print(f"directed:   λ* = {λ_dir:.3f}, workforce {scale*r_dir['workforce']:.2f}, "
      f"unemployment rate {100*r_dir['unemployment']/r_dir['workforce']:.2f}%")
```

它位于拥挤分支上，靠近折点处，无方向搜寻给出的搜寻价值和失业率都低于定向搜寻。

对于第 5 部分，比较就业阈值。

```{code-cell} ipython3
n_bar_und = undirected_market(m_fine, λ_und, U_eq)[2]
n_bar_low, _ = thresholds(m_fine, worker_value(m_fine, λ_und)[0], λ_und)
print("n̄(s), undirected:                       ", n_bar_und)
print("n̄(s), directed, at its own λ*:          ", r_dir['n_bar'])
print("n̄(s), directed, at the undirected λ:    ", n_bar_low)
```

无方向搜寻降低了搜寻的价值，因为搜寻者不再能够避开不需要他们的市场：在定向搜寻下，每一位搜寻者都去往到达价值恰好等于 $\lambda$ 的地方，而在这里，许多人落入正在释放工人或即将这样做的市场。

由于搜寻价值较低，需求已经下降的市场中的工人离开得更慢：在每一种需求状态下，就业阈值都更高，而在最低需求状态之外，大部分差异正是定向搜寻本身在同一较低 $\lambda$ 下所给出的结果，正如最后一行所示。所以各市场保留了更多的工人，经济体以更小的搜寻者流动——即它的失业——来维持其人口。

请注意，对于给定的流入量，从一个被感知的 $\lambda$ 到隐含值 $\beta \sum \psi v$ 的映射，是一个真正的从被感知值到搜寻实际价值的映射，这不同于 {ref}`lps_learning` 中的试探过程。

```{solution-end}
```

```{exercise}
:label: lps_ex_phillips

该模型没有总量冲击，但卢卡斯和普雷斯科特在其结语的开头论证说，该模型的机制"与现在所熟知的对所观察到的菲利普斯曲线的预期论述是一致的"：

> *总需求的一次未预期到的变化（$E(s'|s)$ 的变化）会使失业和工资变化朝相反方向运动。*

从本讲座的均衡出发，以 $\lambda^*$ 处的不变分布描述上一期的经济状态，来检验这一点。

本期，随机选出的一部分市场抽到的需求状态，比大家所知道的运动规律本应给出的状态低了一档；其余市场照常抽取；该比例的选取使得 $\log s$ 的均值下降了 $\Delta$。

没有人预见到这一点，因此阈值 $\bar n(s)$ 和 $\bar y(s)$ 保持原样，$\lambda$ 保持在 $\lambda^*$。

1. 对 $\Delta = 0.01, 0.02, 0.04$ 以及等幅度的 $\log s$ 均值上升，计算本期的失业率和平均工资。
1. 绘制失业率关于平均工资百分比变化的图。解释其形状。
1. 卢卡斯和普雷斯科特补充说，如果这类变化"是一种反复发生的事件，正如现实中那样，那么这一事实就会被纳入工人所面对的最大化问题之中"。这会使计算发生什么变化？
```

```{solution-start} lps_ex_phillips
:class: dropdown
```

这一意外只改变了本期所适用的转移矩阵，因此我们把上一期的各市场连同它们已有的阈值继续向前推进，并从被改变后的矩阵中抽取本期的需求。

```{code-cell} ipython3
def demand_surprise(model, r, Δ):
    """当平均对数需求变动 Δ 时，本期的失业率和平均工资：
    有 |Δ|/notch 比例的市场抽到比通常抽取结果偏离一档的状态。"""
    n_s, y = len(model.s), model.y
    notch = np.log(model.s[1]/model.s[0])
    step = int(np.sign(Δ))
    move = np.zeros((n_s, n_s))                       # 在网格范围内移动一个状态
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

无意外情形的结果再现了均衡本身，这是应该的。

```{code-cell} ipython3
u_pts, w_pts = np.array(sorted(points)).T
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(u_pts, w_pts, 'o-', lw=2)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('unemployment rate (%)')
ax.set_ylabel('change in the mean wage (%)')
plt.tight_layout()
plt.show()
```

失业和工资朝相反方向运动，正如卢卡斯和普雷斯科特所言，各点描绘出一条向下倾斜、在均衡处有一个折点的曲线：一条菲利普斯曲线。

需求出乎意料地下降，会把一些市场推到其就业阈值以下，使它们释放工人，因此失业大幅上升。工资也下降，但降幅小于需求的降幅，因为那些正在释放工人的市场削减了就业，这提高了留下者的边际产品。

需求出乎意料地上升，只能通过阻止本来会发生的离职来降低失业，而这样的离职本就不多，与此同时，每个市场的工资都随其需求上升，近乎等比例上升，而在该期之内也不会有新工人到来对其形成压制。

所以该曲线在失业率低时陡峭，在失业率高时平缓。

符号和折点并不依赖于网格，但失业反应的大小确实依赖于网格：在七状态链上，一档大约相当于 $\log s$ 中的五分之一，因此平均对数需求的某个给定下降会集中体现在少数几个市场上，每个市场都受到重创，而在更细的链上，同样的下降所引起的失业上升幅度更小。

对于第 3 部分，该计算之所以保持阈值和 $\lambda$ 不变，是因为没有人预见到这种变化。如果这种变化反复发生，工人就会把它们纳入自己所使用的转移规律之中，贝尔曼方程也会随之改变，$\bar n$、$\bar y$ 和 $\lambda^*$ 也会跟着改变；卢卡斯和普雷斯科特将此留作"未来研究中一项非平凡的扩展"。

{doc}`phillips_two_stories` 将菲利普斯曲线的预期论述及其背后的自然率假说一路带入到数据中进行检验。

```{solution-end}
```

```{exercise}
:label: lps_ex_planner

{ref}`lps_efficiency` 证明了均衡求解计划问题 {eq}`lps_planner`。对此进行数值检验，并看看它蕴含了什么、又没有蕴含什么。

1. 通过价值函数迭代，在整个网格上选取 $y'$，在 $\lambda^*$ 处求解 {eq}`lps_planner`，并验证 $W(s, y) - W(s, y-1) = v(s, y)$，以及计划者对 $y'$ 的选择恰为 {eq}`lps_transition`。
1. 现在让转移消耗资源：每个离开的工人耗费 $c = 2$。证明 {ref}`lps_ex_cost` 中的均衡，正是一个面对该成本、每释放一名工人获得 $\lambda - c$ 的记入、每引入一名工人被扣除 $\lambda$ 的计划者所选择的配置。这对 {ref}`lps_ex_cost` 中较低的剩余意味着什么？
1. 在保持 $\mu = 50$ 的情况下，比较均衡处的稳态每工人剩余，与两种替代方案：对转移的小额补贴，$c = -0.1$ 和 $c = -0.2$；以及一个更有耐心的计划者将会选择的阈值，取 $\beta' = 0.95$ 和 $0.99$，$\lambda$ 则选取使每个市场保持 $\mu$ 名工人。均衡是否使稳态剩余最大化？将答案与 {ref}`lps_efficiency` 相调和。
```

```{solution-start} lps_ex_planner
:class: dropdown
```

对于第 1 部分，`planner_value` 求解 {eq}`lps_planner`，对于第 2 部分，释放的记入降为 $\lambda - c$。流量项不依赖于 $W$，所以我们只计算一次。

```{code-cell} ipython3
def planner_value(model, λ, c=0.0, tol=1e-10, max_iter=10_000):
    """W(s, y) = 对 y' 取最大值：Φ(s, min(y, y')) + 释放的记入 - 引入的扣除
    + β E W(s', y')，其中释放以 λ - c 记入，引入以 λ 扣除。"""
    P, R, β, y = model.P, model.R, model.β, model.y
    Φ = np.cumsum(R, axis=1)
    moved = y[:, None] - y[None, :]                  # 对每个 (y, y') 而言，被释放的工人数
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
    "将计划者的边际值和策略与均衡进行比较。"
    r = market_outcomes(model, λ, c)
    W, y_choice = planner_value(model, λ, c)
    law = np.clip(model.y[None, :], r['y_bar'][:, None], r['n_bar'][:, None])
    print(f"c = {c}, λ = {λ:.3f}: largest |ΔW - v| = "
          f"{np.abs(np.diff(W, axis=1) - r['v'][:, 1:]).max():.1e}, "
          f"policy equals the transition law: {np.array_equal(y_choice, law)}")

λ_0 = equilibrium_λ(model, μ)
check_planner(model, λ_0)
```

对于第 2 部分，用资源成本进行同样的检验。

```{code-cell} ipython3
λ_2 = equilibrium_λ(model, μ, 2.0, lo=λ_0)
check_planner(model, λ_2, 2.0)
```

在有资源成本的情况下，均衡就是针对那种更昂贵的转移技术的计划者配置：{ref}`lps_efficiency` 中的论证经过释放记入 $\lambda - c$ 之后依然成立，且该资源成本计入了剩余之中。

因此 {ref}`lps_ex_cost` 在资源成本下所发现的较低剩余，正是更差技术所付出的代价，均衡正是在这种条件下所能做到的最好结果。而一项返还的税收以完全相同的方式改变决策，却不消耗任何东西，因此技术仍是自由放任下的技术，其计划者所选择的是自由放任的配置，而被征税的均衡则是被扭曲的：它较低的剩余是一种效率损失。

对于第 3 部分，补贴是负的 $c$，一个更有耐心的计划者的阈值来自以 $\beta'$ 代替 $\beta$ 的工人问题。

```{code-cell} ipython3
def steady_state(model, policy, λ):
    """当 model 的各市场遵循 policy 中工人问题在 λ 处给出的阈值时，
    model 中每个市场的稳态剩余、失业和劳动力量。"""
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

这两种替代方案都使稳态下每工人剩余高于均衡水平，并且都是通过更多的再配置和更多的失业来实现这一点的。因此均衡并不能使稳态剩余最大化，{ref}`lps_efficiency` 也从未如此宣称。

该定理所比较的是从均衡自身的分布出发所得的现值，而转向这两个稳态中的任何一个都要在途中付出剩余作为代价：更多正在转移中的工人意味着现在损失一期他们的产出，而收益要到以后才会出现并被贴现。

一个以 $\beta' > \beta$ 贴现的计划者会更重视那些较晚出现的收益，这就是为什么其阈值会移动更多工人的原因；在不贴现的极限情形下，它会使稳态剩余最大化，这正是黄金法则的类比，而均衡则是修正黄金法则的类比。

{ref}`lps_ex_cost` 中的稳态比较，对于大额成本而言，其方向与该定理一致，但这并非该定理所蕴含的东西：一项小额补贴——同样会使经济体偏离计划者配置——反而提高了稳态剩余。

```{solution-end}
```