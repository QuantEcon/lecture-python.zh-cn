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
  title: 可交换性和贝叶斯更新
  headings:
    Overview: 概述
    Independently and identically distributed: 独立同分布
    Independently and identically distributed::IID means past observations don't tell us anything about future observations: IID意味着过去的观测不能告诉我们任何关于未来观测的信息
    A setting in which past observations are informative: 过去观测具有信息性的情况
    Relationship between IID and exchangeable: IID和可交换之间的关系
    Exchangeability: 可交换性
    Bayes' Law: 贝叶斯定律
    More details about Bayesian updating: 关于贝叶斯更新的更多细节
    More details about Bayesian updating::The likelihood ratio: 似然比
    More details about Bayesian updating::The densities and the probabilities of moving in each direction: 密度函数与各方向移动的概率
    More details about Bayesian updating::Dynamics of the belief: 信念的动态变化
    More details about Bayesian updating::Another instance: 另一个实例
    Appendix: 附录
    Appendix::Sample paths of $\pi_t$: $\pi_t$ 的样本路径
    Appendix::Rates of convergence: 收敛速率
    Appendix::Graph of ensemble dynamics of $\pi_t$: $\pi_t$ 的集合动态图
    Sequels: 后续内容
---

(odu_v3)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 可交换性和贝叶斯更新

```{contents} 目录
:depth: 2
```

## 概述

本讲座研究通过贝叶斯定律进行的学习。

我们涉及由Bruno DeFinetti {cite}`definetti`发明的贝叶斯统计推断的基础。

DeFinetti的工作对经济学家的相关性在David Kreps的{cite}`Kreps88`第11章中得到了有力的阐述。

我们下面研究的一个例子是 {doc}`odu` 的一个关键组成部分。

该讲座扩充了McCall的经典工作搜索模型{cite}`McCall1970`（在 {doc}`mccall_model` 中研究过），通过为失业劳动者提供一个统计推断问题来展示。

我们创建图表来说明似然比在贝叶斯定律中所起的作用。

我们将使用这些图表来深入理解 {doc}`odu` 中驱动结果的运作机制。

除此之外，本讲座还讨论了随机变量序列的统计概念之间的联系，这些序列是：

- 独立同分布的
- 可交换的（也称为*条件*独立同分布）

理解这些概念对于领会贝叶斯更新的工作原理至关重要。

你可以在[这里](https://en.wikipedia.org/wiki/Exchangeable_random_variables)阅读关于可交换性的内容。

因为**可交换性**的另一个术语是**条件独立性**，我们想要回答*基于什么条件*这个问题。

我们还要解释为什么独立性假设阻碍了学习，而条件独立性假设使学习成为可能。

在下文中，我们经常使用

- $W$ 表示一个随机变量
- $w$ 表示随机变量 $W$ 的一个特定实现值

让我们从一些导入开始：

```{code-cell} ipython3
:tags: [hide-output]

import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
from math import gamma

import numpy as np
import scipy.optimize as op
from scipy.integrate import quad
```

## 独立同分布

我们首先来看看**独立同分布序列**这个概念。

独立同分布序列通常简写为IID。

这个概念包含两个方面：

- **独立性**

- **同分布**

如果一个序列$W_0, W_1, \ldots$的联合概率密度等于序列各个组成部分的密度的**乘积**，则称该序列是**独立分布**的。

如果序列$W_0, W_1, \ldots$是**独立同分布**的（IID），那么除了独立性之外，对于所有$t =0, 1, \ldots$，$W_t$的边际密度都相同。

例如，设$p(W_0, W_1, \ldots)$为序列的**联合密度**，$p(W_t)$为特定$W_t$的**边际密度**（对所有$t =0, 1, \ldots$成立）。

那么，如果序列$W_0, W_1, \ldots$是IID的，则其联合密度满足：

$$
p(W_0, W_1, \ldots) =  p(W_0) p(W_1) \cdots
$$

因此联合密度是一系列相同边际密度的乘积。

### IID意味着过去的观测不能告诉我们任何关于未来观测的信息

如果一个随机变量序列是IID的，过去的信息对未来的实现没有任何指示作用。

因此，从过去**无法学到**任何关于未来的信息。

为了理解这些陈述，让我们考虑一个不一定是IID的随机变量序列$\{W_t\}_{t=0}^T$的联合分布

$$
p(W_T, W_{T-1}, \ldots, W_1, W_0)
$$

根据概率定律，我们总可以将这样的联合密度分解为条件密度的乘积：

$$
\begin{aligned}
  p(W_T, W_{T-1}, \ldots, W_1, W_0)    = & p(W_T | W_{T-1}, \ldots, W_0) p(W_{T-1} | W_{T-2}, \ldots, W_0) \cdots  \cr
  & \quad \quad \cdots p(W_1 | W_0) p(W_0)
\end{aligned}
$$

一般来说，

$$
p(W_t | W_{t-1}, \ldots, W_0)   \neq   p(W_t)
$$

这表明左边的**条件密度**不等于右边的**边际密度**。

但在特殊的独立同分布(IID)情况下，

$$
p(W_t | W_{t-1}, \ldots, W_0)   =  p(W_t) ,
$$

且部分历史$W_{t-1}, \ldots, W_0$不包含关于$W_t$概率的任何信息。

因此在IID情况下，从过去的随机变量中**无法学习到**关于未来随机变量密度的任何信息。

但当序列不是IID时，我们可以从过去随机变量的观测中学习到关于未来的一些信息。

接下来我们来看一个序列不是IID的一般情况的例子。

请注意从过去可以学到什么以及何时可以学到。

## 过去观测具有信息性的情况

设$\{W_t\}_{t=0}^\infty$是一个非负标量随机变量序列，其联合概率分布按如下方式构建。

有两个不同的累积分布函数$F$和$G$，它们分别具有密度函数$f$和$g$，用于描述一个非负标量随机变量$W$。

在时间开始之前，比如在时间$t=-1$时，"自然"一次性地选择了**要么**$f$**要么**$g$。

此后在每个时间$t \geq 0$，自然从所选的分布中抽取一个随机变量$W_t$。

因此，数据被永久地生成为从**要么**$F$**要么**$G$中独立同分布(IID)的抽样。

我们可以说*客观上*，即在自然选择了$F$或$G$*之后*，数据是从$F$中生成的概率要么是$0$要么是$1$。

现在我们在这个设定中引入一个部分知情的决策者，他

- 知道$F$和$G$两者，但是

- 不知道自然在$t=-1$时一次性选择的是$F$还是$G$

因此，尽管我们的决策者知道$F$也知道$G$，他却不知道自然选择从这两个已知分布中的哪一个进行抽样。

决策者用**主观概率**$\tilde \pi$来描述他的不确定性，并推理时就好像自然以概率$\tilde \pi \in (0,1)$选择了$F$，以概率$1 - \tilde \pi$选择了$G$。

因此，我们假设决策者：

- **知道**$F$和$G$这两个分布
- **不知道**自然选择了这两个分布中的哪一个
- 通过**表现得好像**或**认为**自然以概率$\tilde \pi \in (0,1)$选择了分布$F$，以概率$1 - \tilde \pi$选择了分布$G$来表达他的不确定性
- 在时间$t \geq 0$时知道部分历史$w_t, w_{t-1}, \ldots, w_0$

为了继续，我们需要了解决策者对部分历史的联合分布的信念。

接下来我们将讨论这一点，并在此过程中描述**可交换性**的概念。

## IID和可交换之间的关系

在自然选择$F$的条件下，序列$W_0, W_1, \ldots$的联合密度是

$$
f(W_0) f(W_1) \cdots
$$

在自然选择$G$的条件下，序列$W_0, W_1, \ldots$的联合密度为

$$
g(W_0) g(W_1) \cdots
$$

因此，**在自然选择**$F$**的条件下**，序列$W_0, W_1, \ldots$是独立同分布的。

此外，**在自然选择**$G$**的条件下**，序列$W_0, W_1, \ldots$也是独立同分布的。

但是部分历史的**无条件分布**又如何呢？

$W_0, W_1, \ldots$的无条件分布显然是

```{math}
:label: eq_definetti

h(W_0, W_1, \ldots ) \equiv \tilde \pi [f(W_0) f(W_1) \cdots \ ] + ( 1- \tilde \pi) [g(W_0) g(W_1) \cdots \ ]
```

在无条件分布$h(W_0, W_1, \ldots )$下，序列$W_0, W_1, \ldots$**不是**独立同分布的。

要验证这个说法，只需注意到，例如

$$
h(W_0, W_1) = \tilde \pi f(W_0)f (W_1) + (1 - \tilde \pi) g(W_0)g(W_1) \neq
              (\tilde \pi f(W_0) + (1-\tilde \pi) g(W_0))(
               \tilde \pi f(W_1) + (1-\tilde \pi) g(W_1))
$$

因此，条件分布

$$
h(W_1 | W_0) \equiv \frac{h(W_0, W_1)}{(\tilde \pi f(W_0) + (1-\tilde \pi) g(W_0))}
 \neq ( \tilde \pi f(W_1) + (1-\tilde \pi) g(W_1))
$$

这意味着随机变量 $W_0$ 包含了关于随机变量 $W_1$ 的信息。

所以过去确实包含了可以用来了解未来的信息。

## 可交换性

虽然序列 $W_0, W_1, \ldots$ 不是独立同分布的，但可以验证它是**可交换的**，这意味着"重新排序"的联合分布 $h(W_0, W_1)$ 和 $h(W_1, W_0)$ 满足

$$
h(W_0, W_1) = h(W_1, W_0)
$$

等等。

更一般地说，如果一个随机变量序列的联合概率分布在有限个随机变量的位置发生改变时保持不变，则称该序列是**可交换的**。

方程 {eq}`eq_definetti` 表示了我们这个例子中的可交换联合密度，它是由两个针对随机变量序列的独立同分布(IID)联合密度构成的**混合**。

贝叶斯统计学家将混合参数 $\tilde \pi \in (0,1)$ 解释为决策者的主观信念——决策者的**先验概率**——即自然选择了概率分布 $F$ 的概率。

```{note}
DeFinetti {cite}`definetti` 建立了一个相关的可交换过程表示，该过程是通过混合参数为 $\theta \in (0,1)$ 的独立同分布伯努利随机变量序列，以及混合概率密度 $\pi(\theta)$ 得到的，贝叶斯统计学家会将这个混合概率密度解释为未知伯努利参数 $\theta$ 的先验分布。
```

## 贝叶斯定律

我们在上面注意到，在我们的示例模型中，从可交换但非独立同分布过程的历史数据中可以学到关于未来的一些信息。

但是我们如何学习？

以及学习什么？

*关于什么*问题的答案是 $\tilde \pi$。

*如何*问题的答案是使用贝叶斯定律。

另一种表述*使用贝叶斯定律*的方式是说*从一个（主观的）联合分布中，计算适当的条件分布*。

让我们在这个背景下深入了解贝叶斯定律。

令 $q$ 表示自然实际从中抽取 $w$ 的分布，并令

$$
\pi = \mathbb{P}\{q = f \}
$$

这里我们将 $\pi$ 视为决策者的**主观概率**（也称为**个人概率**）。

假设在 $t \geq 0$ 时，决策者已观察到历史序列
$w^t \equiv [w_t, w_{t-1}, \ldots, w_0]$。

我们令

$$
\pi_t  = \mathbb{P}\{q = f  | w^t \}
$$

其中我们采用如下约定

$$
\pi_{-1}  = \tilde \pi
$$

在给定 $w^t$ 条件下，$w_{t+1}$ 的分布为

$$
\pi_t f + (1 - \pi_t) g .
$$

更新 $\pi_{t+1}$ 的贝叶斯规则为

$$
\pi_{t+1} = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}
$$ (eq_Bayes102)


等式 {eq}`eq_Bayes102` 源自贝叶斯法则，该法则告诉我们

$$
\mathbb{P}\{q = f \,|\, W = w\}
= \frac{\mathbb{P}\{W = w \,|\, q = f\}\mathbb{P}\{q = f\}}
{\mathbb{P}\{W = w\}}
$$

其中

$$
\mathbb{P}\{W = w\} = \sum_{a \in \{f, g\}} \mathbb{P}\{W = w \,|\, q = a \} \mathbb{P}\{q = a \}
$$

## 关于贝叶斯更新的更多细节

让我们仔细观察并重新整理等式{eq}`eq_Bayes102`中表示的贝叶斯法则，目的是理解**后验**概率$\pi_{t+1}$如何受到**先验**概率$\pi_t$和**似然比**的影响

$$
l(w) = \frac{f(w)}{g(w)}
$$

我们可以方便地将更新规则{eq}`eq_Bayes102`重写为

$$
\pi_{t+1}   =\frac{\pi_{t}f\left(w_{t+1}\right)}{\pi_{t}f\left(w_{t+1}\right)+\left(1-\pi_{t}\right)g\left(w_{t+1}\right)}
    =\frac{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}}{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}+\left(1-\pi_{t}\right)}
    =\frac{\pi_{t}l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}
$$

这意味着

```{math}
:label: eq_Bayes103

\frac{\pi_{t+1}}{\pi_{t}}=\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\begin{cases} >1 &
\text{if }l\left(w_{t+1}\right)>1\\
\leq1 & \text{if }l\left(w_{t+1}\right)\leq1
\end{cases}
```

注意似然比和先验是如何相互作用，以决定观测值$w_{t+1}$是导致决策者增加还是减少他/她对分布$F$的主观概率。

当似然比$l(w_{t+1})$大于1时，观测值$w_{t+1}$会将分布$F$的概率$\pi$向上推动，当似然比$l(w_{t+1})$小于1时，观测值$w_{t+1}$会将$\pi$向下推动。

表达式{eq}`eq_Bayes103`是我们将用来显示由贝叶斯定律引起的$\{\pi_t\}_{t=0}^\infty$动态的一些图表的基础。

我们将绘制 $l\left(w\right)$ 来帮助我们理解学习过程是如何进行的——即，如何通过贝叶斯更新来更新自然选择分布 $f$ 的概率 $\pi$。

我们分三步构建这幅图景，每一步产生一张图表。

这三张图都是由相同的素材构建的：密度函数 $f$ 和 $g$，以及似然比 $l(w) = f(w)/g(w)$ 等于1时对应的 $w$ 值。

$f$ 和 $g$ 都是贝塔密度函数，所以我们先从一般的贝塔密度函数开始。

```{code-cell} ipython3
def p(w, a, b):
    "参数为a和b的贝塔密度函数。"
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * w**(a - 1) * (1 - w)**(b - 1)
```

下一个函数为给定的一对贝塔分布组装素材。

```{code-cell} ipython3
def create_model(F_a=1, F_b=1, G_a=3, G_b=1.2):
    """
    构建密度函数f和g，以及似然比l(w) = f(w) / g(w)
    等于1时对应的两个w值。
    """
    f = lambda w: p(w, F_a, F_b)
    g = lambda w: p(w, G_a, G_b)

    # g的众数将[0, 1]分为两个区间，每个区间各含一个根
    G_mode = (G_a - 1) / (G_a + G_b - 2)
    obj = lambda w: f(w) / g(w) - 1
    roots = np.array([op.root_scalar(obj, bracket=[1e-10, G_mode]).root,
                      op.root_scalar(obj, bracket=[G_mode, 1 - 1e-10]).root])
    return f, g, roots
```

### 似然比

我们的第一张图将似然比 $l(w)$ 绘制在横坐标轴上，将 $w$ 绘制在纵坐标轴上。

我们采用这种方式绘制，以便 $w$ 能与接下来的两张图共享同一个坐标轴。

```{code-cell} ipython3
def plot_likelihood_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    w_grid = np.linspace(1e-12, 1 - 1e-12, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(f(w_grid) / g(w_grid), w_grid, label='$l$', lw=2)
    ax.vlines(1, 0, 1, linestyle='--')
    ax.hlines(roots, 0, 2, linestyle='--')
    ax.set_xlim(0, 2)
    ax.legend(loc=4)
    ax.set(xlabel='$l(w) = f(w) / g(w)$', ylabel='$w$')
    plt.show()
```

我们从 $f$ 是 $[0,1]$ 上的均匀分布（即参数为 $F_a=1, F_b=1$ 的贝塔分布）开始，而 $g$ 是参数为 $G_a=3, G_b=1.2$ 的贝塔分布。

```{code-cell} ipython3
plot_likelihood_ratio()
```

两条水平虚线标记了 $l(w) = 1$ 时对应的 $w$ 值。

在这两条线之间，似然比小于1，因此根据{eq}`eq_Bayes103`，落在该区域的抽样会使 $\pi$ 向下推动。

在这两条线之外，似然比大于1，因此落在该区域的抽样会使 $\pi$ 向上推动。

### 密度函数与各方向移动的概率

我们的第二张图将 $f(w)$ 和 $g(w)$ 对 $w$ 进行绘制，并对由同样两个 $w$ 值划分出的区域进行着色。

对这些区域着色使我们能够为每个移动方向赋予一个概率，我们通过在相关区域上对相应密度函数积分来计算这个概率。

```{code-cell} ipython3
def plot_densities(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    w_grid = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(f(w_grid), w_grid, label='$f$', lw=2)
    ax.plot(g(w_grid), w_grid, label='$g$', lw=2)
    ax.vlines(1, 0, 1, linestyle='--')
    ax.hlines(roots, 0, 2, linestyle='--')
    ax.legend(loc=4)
    ax.set(xlabel='$f(w), g(w)$', ylabel='$w$')

    # 无论哪个密度函数被着色，落入各区域的概率
    area_lower = quad(f, 0, roots[0])[0]
    area_middle = quad(g, roots[0], roots[1])[0]
    area_upper = quad(f, roots[1], 1)[0]

    ax.fill_between([0, 1], 0, roots[0], color='blue', alpha=0.15)
    ax.text((f(0) + f(roots[0])) / 4, roots[0] / 2, f"{area_lower: .3g}")
    w_middle = np.linspace(roots[0], roots[1], 20)
    ax.fill_betweenx(w_middle, 0, g(w_middle), color='orange', alpha=0.15)
    ax.text(np.mean(g(roots)) / 2, np.mean(roots), f"{area_middle: .3g}")
    ax.fill_between([0, 1], roots[1], 1, color='blue', alpha=0.15)
    ax.text((f(roots[1]) + f(1)) / 4, (roots[1] + 1) / 2, f"{area_upper: .3g}")
    plt.show()
```

让我们来看看与之前相同的一对分布。

```{code-cell} ipython3
plot_densities()
```

彩色区域中的分数是 $w$ 的实现值落入其旁边区域的概率。

蓝色区域是 $f$ 的积分，橙色区域是 $g$ 的积分。

例如，在真实分布$F$下，如果$w$落入区间$[0.524, 0.999]$，$\pi$将向$0$更新，这在$F$下发生的概率是$1 - .524 = .476$。

但如果$G$是真实分布，这种情况发生的概率将是$0.816$。

### 信念的动态变化

我们的第三张图在 $(\pi, w)$ 平面上的每个点附加一个箭头，显示当当前信念为 $\pi$ 且新的抽样为 $w$ 时，贝叶斯定律所引起的 $\pi$ 的变化。

```{code-cell} ipython3
def plot_belief_dynamics(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    π_grid = np.linspace(1e-3, 1 - 1e-3, 100)

    # 在(π, w)的粗网格上每个点的π变化量
    W = np.arange(0.01, 0.99, 0.08)
    Π = np.arange(0.01, 0.99, 0.08)
    lw = (f(W) / g(W))[:, None]     # 每个w处的似然比，作为一列
    ΔΠ = Π * (lw / (Π * lw + 1 - Π) - 1)
    ΔW = np.zeros_like(ΔΠ)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(Π, W, ΔΠ, ΔW, scale=2, color='r', alpha=0.8)
    ax.fill_between(π_grid, 0, roots[0], color='blue', alpha=0.15)
    ax.fill_between(π_grid, roots[0], roots[1], color='green', alpha=0.15)
    ax.fill_between(π_grid, roots[1], 1, color='blue', alpha=0.15)
    ax.hlines(roots, 0, 1, linestyle='--')
    ax.set(xlabel=r'$\pi$', ylabel='$w$')
    ax.grid()
    plt.show()
```

我们再次使用同样的一对分布。

```{code-cell} ipython3
plot_belief_dynamics()
```

向右指的箭头表示贝叶斯定律使 $\pi$ 增加的情况，向左指的箭头表示贝叶斯定律使 $\pi$ 减少的情况。

箭头的长度表示贝叶斯定律驱使 $\pi$ 改变的力的大小。

这些长度取决于两个因素：横坐标轴上的先验概率 $\pi$，以及以当前 $w$ 值形式出现的证据（在纵坐标轴上）。

在蓝色区域（$l(w) > 1$ 的地方），箭头指向右方；在虚线之间的绿色区域（$l(w) < 1$ 的地方），箭头指向左方。

对于这些参数，上方的蓝色区域是紧贴在 $w = 1$ 下方的一个薄片区域，因为只有在非常接近上限的抽样中，$l(w)$ 才会再次回升到大于1。

### 另一个实例

接下来我们使用代码为我们模型的另一个实例创建图形。

我们保持$F$与前一个实例相同，即均匀分布，但现在假设$G$是一个参数为$G_a=2, G_b=1.6$的Beta分布。

```{code-cell} ipython3
plot_likelihood_ratio(G_a=2, G_b=1.6)
plot_densities(G_a=2, G_b=1.6)
plot_belief_dynamics(G_a=2, G_b=1.6)
```

注意观察似然比、密度函数以及箭头与我们之前例子的对比。

## 附录


### $\pi_t$ 的样本路径

现在我们将通过绘制 $\pi_t$ 的多个样本路径来进行一些有趣的探索，这些路径基于两种可能的自然分布选择假设：

- 自然永久从 $F$ 分布中抽取
- 自然永久从 $G$ 分布中抽取

结果取决于似然比过程的一个特殊性质，这在 {doc}`advanced:additive_functionals` 中有详细讨论。

在进行模拟之前，值得将贝叶斯定律用**赔率**而非概率重新表述。

用赔率 $\pi / (1 - \pi)$ 表示，更新规则 {eq}`eq_Bayes102` 变为

```{math}
:label: eq_odds

\frac{\pi_{t+1}}{1 - \pi_{t+1}} = l(w_{t+1}) \frac{\pi_{t}}{1 - \pi_{t}}
```

因此贝叶斯定律在赔率上是*乘法*的，将 {eq}`eq_odds` 向前迭代回先验 $\pi_{-1}$，可得

```{math}
:label: eq_odds_product

\frac{\pi_{t}}{1 - \pi_{t}} = \frac{\pi_{-1}}{1 - \pi_{-1}} \prod_{s=0}^{t} l(w_{s})
```

因此信念路径是似然比的累积乘积，这就是为什么似然比过程的性质决定了它的行为。

这也意味着，我们可以使用累积乘积来模拟整个路径集合，而不需要按日期逐步迭代。

```{code-cell} ipython3
def simulate(rng, a, b, T=50, N=1000, π_init=0.5,
             F_a=1, F_b=1, G_a=3, G_b=1.2):
    """
    在自然从 Beta(a, b) 中独立同分布抽取的情况下，模拟 N 条信念 π 在 T 个时期内的路径。
    返回一个形状为 (N, T+1) 的数组，其第一列为共同的先验 π_init。
    """
    w = rng.beta(a, b, size=(N, T))
    l = p(w, F_a, F_b) / p(w, G_a, G_b)
    odds = (π_init / (1 - π_init)) * np.cumprod(l, axis=1)
    return np.column_stack([np.full(N, π_init), odds / (1 + odds)])
```

以下函数用于绘制路径。

```{code-cell} ipython3
def plot_paths(π_paths):
    fig, ax = plt.subplots()
    ax.plot(π_paths.T, color='b', lw=0.8, alpha=0.5)
    ax.set(xlabel='$t$', ylabel=r'$\pi_t$')
    plt.show()
```

我们首先生成 $N$ 条模拟的 $\{\pi_t\}$ 路径，每条路径包含 $T$ 个时期，其中序列是真实的从分布 $F$ 中独立同分布抽取的。我们设定初始先验 $\pi_{-1} = .5$。

```{code-cell} ipython3
rng = np.random.default_rng(42)
T = 50
```

```{code-cell} ipython3
# 当自然选择F时
π_paths_F = simulate(rng, a=1, b=1, T=T, N=1000)
plot_paths(π_paths_F)
```

在上述例子中，对于大多数路径 $\pi_t \rightarrow 1$。

因此，贝叶斯定律显然最终能够在我们的大多数路径中发现真相。

接下来，当序列确实是来自 $G$ 的独立同分布抽样时，我们生成 $T$ 期的路径。同样，我们设定初始先验 $\pi_{-1} = .5$。

```{code-cell} ipython3
# 当自然选择G时
π_paths_G = simulate(rng, a=3, b=1.2, T=T, N=1000)
plot_paths(π_paths_G)
```

在上图中我们观察到现在大多数路径 $\pi_t \rightarrow 0$。

### 收敛速率

我们研究当自然生成的数据是来自 $F$ 的独立同分布抽样时 $\pi_t$ 向 $1$ 的收敛速率，以及当自然生成的数据是来自 $G$ 的独立同分布抽样时 $\pi_t$ 向 $0$ 的收敛速率。

我们通过对 $\{\pi_t\}_{t=0}^T$ 的模拟路径进行平均来实现这一点。

使用 $N$ 条模拟的 $\pi_t$ 路径，当数据是从 $F$ 中抽样生成时，我们在每个 $t$ 时刻计算 $1 - \sum_{i=1}^{N}\pi_{i,t}$，当数据是从 $G$ 中抽样生成时，我们计算 $\sum_{i=1}^{N}\pi_{i,t}$。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(range(T + 1), 1 - np.mean(π_paths_F, axis=0), label='F生成')
ax.plot(range(T + 1), np.mean(π_paths_G, axis=0), label='G生成')
ax.set(xlabel='$t$', title='收敛')
ax.legend()
plt.show()
```

从上图可以看出，收敛速率似乎不依赖于是 $F$ 还是 $G$ 生成数据。

### $\pi_t$ 的集合动态图

通过对相关概率分布进行积分计算 $\frac{\pi_{t+1}}{\pi_{t}}$ 的条件期望作为 $\pi_t$ 的函数，可以获得关于 $\{\pi_t\}$ 动态的更多见解：

$$
\begin{aligned}
E\left[\frac{\pi_{t+1}}{\pi_{t}}\biggm|q=a, \pi_{t}\right] &=E\left[\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\biggm|q= a, \pi_{t}\right], \\
    &=\int_{0}^{1}\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)} a\left(w_{t+1}\right)dw_{t+1}
\end{aligned}
$$

其中 $a =f,g$。

以下代码近似计算上述积分：

```{code-cell} ipython3
def plot_expected_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    # 直接构造 f 和 g：这里它们可能相同，此时 l(w) = 1 没有根
    f = lambda w: p(w, F_a, F_b)
    g = lambda w: p(w, G_a, G_b)
    l = lambda w: f(w) / g(w)
    π_grid = np.linspace(0.02, 0.98, 100)

    fig, ax = plt.subplots()
    for label, a in [('f', f), ('g', g)]:
        integrand = lambda w, π: a(w) * l(w) / (π * l(w) + 1 - π)
        ratios = [quad(integrand, 0, 1, args=(π,))[0] for π in π_grid]
        ax.plot(π_grid, ratios, label=f'{label} generates')

    ax.hlines(1, 0, 1, linestyle='--')
    ax.set(xlabel=r'$\pi_t$', ylabel=r'$E[\pi_{t+1} / \pi_t]$')
    ax.legend()
    plt.show()
```

首先，考虑 $F_a=F_b=1$ 且 $G_a=3, G_b=1.2$ 的情况。

```{code-cell} ipython3
plot_expected_ratio()
```

上图显示，当数据由 $F$ 生成时，$\pi_t$ 平均总是向北移动，而当数据由 $G$ 生成时，$\pi_t$ 向南移动。

接下来，我们将看一个退化情况，其中 $f$ 和 $g$ 是相同的贝塔分布，且 $F_a=G_a=3, F_b=G_b=1.2$。

从某种意义上说，这里没有什么可学习的。

```{code-cell} ipython3
plot_expected_ratio(F_a=3, F_b=1.2)
```

上图表明 $\pi_t$ 是惰性的，保持在其初始值。

最后，让我们看一个 $f$ 和 $g$ 既不是非常不同也不完全相同的情况，特别是当 $F_a=2, F_b=1$ 且 $G_a=3, G_b=1.2$ 时。

```{code-cell} ipython3
plot_expected_ratio(F_a=2, F_b=1, G_a=3, G_b=1.2)
```

## 后续内容

我们将在以下讲座中应用并深入探讨本讲座中提出的一些想法：

* {doc}`likelihood_ratio_process` 描述了**似然比过程**及其在频率派和贝叶斯统计理论中的作用
* {doc}`navy_captain` 研究了二战时期一位美国海军上尉的直觉，即海军要求他使用的（频率派）决策规则不如亚伯拉罕·瓦尔德尚未设计的序贯规则。
