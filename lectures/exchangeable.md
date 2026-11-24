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
    Exchangeability::De Finetti's theorem: 德菲内蒂定理
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
    Exercises: 练习
---

(exchangeable)=
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

我们涉及由 {cite:t}`definetti` 发明的贝叶斯统计推断的基础。

德·芬内蒂的工作对经济学家的相关性在David Kreps的 {cite}`Kreps88` 第11章中得到了有力的阐述。

我们下面研究的一个例子是 {doc}`odu` 的一个关键组成部分。

该讲座扩充了McCall的经典工作搜索模型 {cite}`McCall1970` （在 {doc}`mccall_model` 中研究过），通过为失业劳动者提供一个统计推断问题来展示。

我们创建图表来说明似然比在贝叶斯定律中所起的作用。

我们将使用这些图表来深入理解 {doc}`odu` 中驱动结果的运作机制。

除此之外，本讲座还讨论了随机变量序列的统计概念之间的联系，这些序列是：

- 独立同分布的
- 可交换的

理解这些概念对于领会贝叶斯更新的工作原理至关重要。

你可以在 [这里](https://en.wikipedia.org/wiki/Exchangeable_random_variables) 阅读关于可交换性的内容。

德·芬内蒂定理指出，一个无限可交换序列是**条件独立同分布**的：它在以观察者未知其取值的某个随机变量为条件时是独立同分布的。[^condind]

我们想要回答*基于什么条件*这个问题。

我们还要解释为什么独立性假设阻碍了学习，而序列条件独立同分布的假设使学习成为可能。

[^condind]: 仅有条件独立并不意味着可交换性。设 $W_t = t \theta + \varepsilon_t$，其中 $\theta$ 是一个随机变量，$\{\varepsilon_t\}$ 是独立于 $\theta$ 的独立同分布 $N(0,1)$ 序列。在以 $\theta$ 为条件时，$W_t$ 是独立的，但它们并非同分布，因此序列 $\{W_t\}$ 不是可交换的。

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
from scipy.special import expit
```

## 独立同分布

我们首先来看看**独立同分布序列**这个概念。

独立同分布序列通常简写为IID。

这个概念包含两个方面：

- **独立性**

- **同分布**

如果对于每个$n \geq 0$，$W_0, W_1, \ldots, W_n$的联合概率密度等于$W_0, W_1, \ldots, W_n$的边际密度的**乘积**，则称序列$W_0, W_1, \ldots$是**独立分布**的。

如果序列$W_0, W_1, \ldots$是**独立同分布**的（IID），那么除了独立性之外，对于所有$t =0, 1, \ldots$，$W_t$的边际密度都相同。

例如，设$p(W_0, W_1, \ldots, W_n)$为$W_0, W_1, \ldots, W_n$的**联合密度**，$p(W_t)$为每个$W_t$共同的**边际密度**。

那么，序列$W_0, W_1, \ldots$是IID的，当且仅当

$$
p(W_0, W_1, \ldots, W_n) =  p(W_0) p(W_1) \cdots p(W_n) \quad \text{对所有 } n \geq 0 \text{ 成立}
$$

因此每个联合密度都是一系列相同边际密度的乘积。

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

在自然选择$F$的条件下，$W_0, W_1, \ldots, W_n$的联合密度是

$$
f(W_0) f(W_1) \cdots f(W_n)
$$

对任意 $n \geq 0$ 都成立。

在自然选择$G$的条件下，$W_0, W_1, \ldots, W_n$的联合密度为

$$
g(W_0) g(W_1) \cdots g(W_n)
$$

因此，**在自然选择**$F$**的条件下**，序列$W_0, W_1, \ldots$是独立同分布的。

此外，**在自然选择**$G$**的条件下**，序列$W_0, W_1, \ldots$也是独立同分布的。

但是部分历史的**无条件分布**又如何呢？

$W_0, W_1, \ldots, W_n$的无条件联合密度显然是

```{math}
:label: eq_definetti

h(W_0, W_1, \ldots, W_n ) \equiv \tilde \pi [f(W_0) f(W_1) \cdots f(W_n)] + ( 1- \tilde \pi) [g(W_0) g(W_1) \cdots g(W_n)], \quad n \geq 0
```

在无条件分布 {eq}`eq_definetti` 下，序列$W_0, W_1, \ldots$**不是**独立同分布的。

要验证这个说法，设 $h(W_t) = \tilde \pi f(W_t) + (1-\tilde \pi) g(W_t)$ 为 $W_t$ 的边缘密度。

那么

```{math}
:label: eq_hcov

h(W_0, W_1) - h(W_0) h(W_1) = \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) - g(W_0)\bigr) \bigl(f(W_1) - g(W_1)\bigr)
```

当 $f(W_0) \neq g(W_0)$ 且 $f(W_1) \neq g(W_1)$ 时，上式右边不为零，因此联合密度 $h(W_0, W_1)$ 不等于边缘密度 $h(W_0)$ 与 $h(W_1)$ 的乘积。

因此，条件分布

$$
h(W_1 | W_0) \equiv \frac{h(W_0, W_1)}{(\tilde \pi f(W_0) + (1-\tilde \pi) g(W_0))}
 \neq ( \tilde \pi f(W_1) + (1-\tilde \pi) g(W_1))
$$

这意味着随机变量 $W_0$ 包含了关于随机变量 $W_1$ 的信息。

所以过去确实包含了可以用来了解未来的信息。

练习 {ref}`exchangeable_ex1` 利用 {eq}`eq_hcov` 来说明 $W_0$ 和 $W_1$ 是正相关的。

## 可交换性

虽然序列 $W_0, W_1, \ldots$ 不是独立同分布的，但可以验证它是**可交换的**，这意味着"重新排序"的联合分布 $h(W_0, W_1)$ 和 $h(W_1, W_0)$ 满足

$$
h(W_0, W_1) = h(W_1, W_0)
$$

等等。

更一般地说，如果一个随机变量序列的联合概率分布在有限个随机变量的位置发生改变时保持不变，则称该序列是**可交换的**。

方程 {eq}`eq_definetti` 表示了我们这个例子中的可交换联合密度，它是由两个针对随机变量序列的独立同分布(IID)联合密度构成的**混合**。

贝叶斯统计学家将混合参数 $\tilde \pi \in (0,1)$ 解释为决策者的主观信念——决策者的**先验概率**——即自然选择了概率分布 $F$ 的概率。

### 德菲内蒂定理

正如 {cite:t}`definetti` 所证明的，每一个无限可交换序列都是独立同分布序列的混合。

对于二元随机变量 $W_t \in \{0, 1\}$，德菲内蒂定理指出，无限序列 $W_0, W_1, \ldots$ 是可交换的，当且仅当存在 $[0,1]$ 上的一个概率测度 $\mu$，使得

```{math}
:label: eq_definetti_bernoulli

\textrm{Prob}(W_0 = w_0, \ldots, W_{n-1} = w_{n-1}) = \int_0^1 \theta^{k} (1-\theta)^{n-k} \mu(d\theta), \quad k = \sum_{t=0}^{n-1} w_t
```

对任意 $n \geq 1$ 和任意 $(w_0, \ldots, w_{n-1}) \in \{0,1\}^n$ 都成立。

混合测度 $\mu$ 是唯一的。

它是极限相对频率的概率分布

$$
\theta = \lim_{n \rightarrow \infty} \frac{1}{n} \sum_{t=0}^{n-1} W_t ,
$$

该极限以概率一存在。

给定 $\theta$ 的条件下，$W_t$ 是成功概率为 $\theta$ 的独立同分布伯努利随机变量。

这回答了"以什么为条件？"这个问题：以极限相对频率 $\theta$ 为条件。

贝叶斯统计学家将 $\mu$ 解释为对未知伯努利参数 $\theta \in [0,1]$ 的先验概率分布。

测度 $\mu$ 不必有密度；它可以在有限多个点上赋予正概率。

{cite:t}`HewittSavage1955` 将德菲内蒂定理推广到了取值于更一般空间的序列。

在他们的版本中，混合测度是关于 $W$ 的概率分布的概率分布，而条件变量是 $W_0, W_1, \ldots$ 的极限经验分布。

方程 {eq}`eq_definetti` 是其中一个实例，它采用一个两点混合测度，赋予 $F$ 概率 $\tilde \pi$，赋予 $G$ 概率 $1 - \tilde \pi$。

在这个实例中，$W_0, W_1, \ldots$ 的极限经验分布在自然选择了 $F$ 时为 $F$，在自然选择了 $G$ 时为 $G$。

德菲内蒂定理要求序列是无限的。

练习 {ref}`exchangeable_ex1` 描述了一个有限可交换序列，它不是独立同分布序列的混合。

{doc}`merging_of_opinions` 中的贝塔-伯努利模型是 {eq}`eq_definetti_bernoulli` 的一个实例，其中 $\mu$ 是贝塔分布。

练习 {ref}`exchangeable_ex3` 描述了一个能生成它的瓮模型方案。

## 贝叶斯定律

我们在上面注意到，在我们的示例模型中，从可交换但非独立同分布过程的历史数据中可以学到关于未来的一些信息。

但是我们如何学习？

以及学习什么？

*关于什么*问题的答案是：关于自然在时间 $-1$ 选择了哪个分布，$F$ 还是 $G$。

决策者知道他的先验概率 $\tilde \pi$。

他所学到的内容体现在一个后验概率中，随着数据的到来，该后验概率会偏离 $\tilde \pi$。

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

在给定 $q$ 的条件下，抽取值 $w_{t+1}$ 与历史序列 $w^t$ 相互独立。

在给定 $w^t$ 条件下，$w_{t+1}$ 的密度为

$$
\pi_t f + (1 - \pi_t) g .
$$

更新 $\pi_{t+1}$ 的贝叶斯规则为

$$
\pi_{t+1} = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}
$$ (eq_Bayes102)


等式 {eq}`eq_Bayes102` 源自贝叶斯法则，该法则告诉我们

$$
\mathbb{P}\{q = f \,|\, w^{t+1}\}
= \frac{p(w_{t+1} \,|\, q = f, w^t)\, \mathbb{P}\{q = f \,|\, w^t\}}
{p(w_{t+1} \,|\, w^t)}
$$

其中 $p$ 表示密度，且

$$
p(w_{t+1} \,|\, w^t) = \sum_{a \in \{f, g\}} p(w_{t+1} \,|\, q = a, w^t)\, \mathbb{P}\{q = a \,|\, w^t\} .
$$

由于在给定 $q$ 的条件下 $w_{t+1}$ 与 $w^t$ 相互独立，我们有 $p(w_{t+1} \,|\, q = f, w^t) = f(w_{t+1})$ 以及 $p(w_{t+1} \,|\, q = g, w^t) = g(w_{t+1})$。

## 关于贝叶斯更新的更多细节

让我们仔细观察并重新整理等式{eq}`eq_Bayes102`中表示的贝叶斯法则，目的是理解**后验**概率$\pi_{t+1}$如何受到**先验**概率$\pi_t$和**似然比**的影响

$$
\ell(w) = \frac{f(w)}{g(w)}
$$

我们可以方便地将更新规则{eq}`eq_Bayes102`重写为

$$
\pi_{t+1}   =\frac{\pi_{t}f\left(w_{t+1}\right)}{\pi_{t}f\left(w_{t+1}\right)+\left(1-\pi_{t}\right)g\left(w_{t+1}\right)}
    =\frac{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}}{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}+\left(1-\pi_{t}\right)}
    =\frac{\pi_{t}\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}
$$

这意味着

```{math}
:label: eq_Bayes103

\frac{\pi_{t+1}}{\pi_{t}}=\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\begin{cases} >1 &
\text{if }\ell\left(w_{t+1}\right)>1\\
\leq1 & \text{if }\ell\left(w_{t+1}\right)\leq1
\end{cases}
```

似然比本身就决定了观测值$w_{t+1}$是导致决策者增加还是减少他/她对分布$F$的主观概率。

当似然比$\ell(w_{t+1})$大于1时，观测值$w_{t+1}$会将分布$F$的概率$\pi$向上推动，当似然比$\ell(w_{t+1})$小于1时，观测值$w_{t+1}$会将$\pi$向下推动。

先验$\pi_t$会影响变动的大小：

$$
\pi_{t+1} - \pi_t = \frac{\pi_t (1 - \pi_t) \bigl(\ell(w_{t+1}) - 1\bigr)}{\pi_t \ell(w_{t+1}) + 1 - \pi_t}
$$

当$\pi_t$趋近于$0$或$1$时，这个变动量会收缩至零。

表达式{eq}`eq_Bayes103`是我们将用来显示由贝叶斯定律引起的$\{\pi_t\}_{t=0}^\infty$动态的一些图表的基础。

我们将绘制 $\ell\left(w\right)$ 来帮助我们理解学习过程是如何进行的——即，如何通过贝叶斯更新来更新自然选择分布 $f$ 的概率 $\pi$。

我们分三步构建这幅图景，每一步产生一张图表。

这三张图都是由相同的素材构建的：密度函数 $f$ 和 $g$，以及似然比 $\ell(w) = f(w)/g(w)$ 等于1时对应的 $w$ 值。

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
    构建密度函数f和g，以及似然比ℓ(w) = f(w) / g(w)
    等于1时对应的两个w值。

    这个根查找过程假设f是均匀分布（F_a = F_b = 1），且
    G_a > 1且G_b > 1。这样g在0和1处为零，并且在内部
    只有一个众数，在该众数处g大于1，因此
    ℓ(w) = 1 / g(w)在g的众数两侧各恰好等于1一次。
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

我们的第一张图将似然比 $\ell(w)$ 绘制在横坐标轴上，将 $w$ 绘制在纵坐标轴上。

我们采用这种方式绘制，以便 $w$ 能与接下来的两张图共享同一个坐标轴。

```{code-cell} ipython3
def plot_likelihood_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    f, g, roots = create_model(F_a, F_b, G_a, G_b)
    w_grid = np.linspace(1e-12, 1 - 1e-12, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(f(w_grid) / g(w_grid), w_grid, label=r'$\ell$', lw=2)
    ax.vlines(1, 0, 1, linestyle='--')
    ax.hlines(roots, 0, 2, linestyle='--')
    ax.set_xlim(0, 2)
    ax.legend(loc=4)
    ax.set(xlabel=r'$\ell(w) = f(w) / g(w)$', ylabel='$w$')
    plt.show()
```

我们从 $f$ 是 $[0,1]$ 上的均匀分布（即参数为 $F_a=1, F_b=1$ 的贝塔分布）开始，而 $g$ 是参数为 $G_a=3, G_b=1.2$ 的贝塔分布。

```{code-cell} ipython3
plot_likelihood_ratio()
```

两条水平虚线标记了 $\ell(w) = 1$ 时对应的 $w$ 值。

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

例如，在真实分布$F$下，如果$w$落入区间$[0.524, 0.999]$，$\pi$将向$0$更新，这在$F$下发生的概率约为$0.999 - 0.524 \approx 0.475$。

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

在蓝色区域（$\ell(w) > 1$ 的地方），箭头指向右方；在虚线之间的绿色区域（$\ell(w) < 1$ 的地方），箭头指向左方。

对于这些参数，上方的蓝色区域是紧贴在 $w = 1$ 下方的一个薄片区域，因为只有在非常接近上限的抽样中，$\ell(w)$ 才会再次回升到大于1。

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

结果取决于似然比过程的一个特殊性质，这在 {doc}`likelihood_ratio_process` 的 {ref}`这一节 <nature_likeli>` 中有详细讨论。

在进行模拟之前，值得将贝叶斯定律用**赔率**而非概率重新表述。

用赔率 $\pi / (1 - \pi)$ 表示，更新规则 {eq}`eq_Bayes102` 变为

```{math}
:label: eq_odds

\frac{\pi_{t+1}}{1 - \pi_{t+1}} = \ell(w_{t+1}) \frac{\pi_{t}}{1 - \pi_{t}}
```

因此贝叶斯定律在赔率上是*乘法*的，将 {eq}`eq_odds` 向前迭代回先验 $\pi_{-1}$，可得

```{math}
:label: eq_odds_product

\frac{\pi_{t}}{1 - \pi_{t}} = \frac{\pi_{-1}}{1 - \pi_{-1}} \prod_{s=0}^{t} \ell(w_{s})
```

因此信念路径是似然比的累积乘积，这就是为什么似然比过程的性质决定了它的行为。

取对数可以把乘积变成求和：

$$
\log \frac{\pi_{t}}{1 - \pi_{t}} = \log \frac{\pi_{-1}}{1 - \pi_{-1}} + \sum_{s=0}^{t} \log \ell(w_{s})
$$

我们利用这个求和来进行模拟。

这样做可以避免乘积运算中的溢出问题，以及当 $\pi_t$ 接近 $0$ 或 $1$ 时的精度损失。

这也意味着，我们可以使用累积求和来模拟整个路径集合，而不需要按日期逐步迭代。

```{code-cell} ipython3
def simulate(rng, a, b, T=50, N=1000, π_init=0.5,
             F_a=1, F_b=1, G_a=3, G_b=1.2):
    """
    在自然从 Beta(a, b) 中独立同分布抽取 w_0, ..., w_{T-1} 的情况下，
    模拟 N 条信念 π 的路径。返回一个形状为 (N, T+1) 的数组，
    其第 0 列为先验 π_{-1} = π_init，第 t+1 列为 π_t。
    """
    w = rng.beta(a, b, size=(N, T))
    log_l = np.log(p(w, F_a, F_b)) - np.log(p(w, G_a, G_b))
    log_odds = np.log(π_init / (1 - π_init)) + np.cumsum(log_l, axis=1)
    return np.column_stack([np.full(N, π_init), expit(log_odds)])
```

以下函数用于绘制路径。

```{code-cell} ipython3
def plot_paths(π_paths):
    "针对日期 t = -1, 0, ..., T-1 绘制信念路径。"
    t = np.arange(π_paths.shape[1]) - 1
    fig, ax = plt.subplots()
    ax.plot(t, π_paths.T, color='b', lw=0.8, alpha=0.5)
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

我们通过对 $\{\pi_t\}_{t=-1}^{T-1}$ 的模拟路径进行平均来实现这一点。

使用 $N$ 条模拟的 $\pi_t$ 路径，当数据是从 $F$ 中抽样生成时，我们在每个 $t$ 时刻计算 $1 - \frac{1}{N}\sum_{i=1}^{N}\pi_{i,t}$，当数据是从 $G$ 中抽样生成时，我们计算 $\frac{1}{N}\sum_{i=1}^{N}\pi_{i,t}$。

```{code-cell} ipython3
t_grid = np.arange(-1, T)
fig, ax = plt.subplots()
ax.plot(t_grid, 1 - np.mean(π_paths_F, axis=0), label='F生成')
ax.plot(t_grid, np.mean(π_paths_G, axis=0), label='G生成')
ax.set(xlabel='$t$', title='收敛')
ax.legend()
plt.show()
```

这两条曲线几乎重合。

一个精确的恒等式解释了原因：

```{math}
:label: eq_conv_identity

\pi_{-1} E_f[1 - \pi_t] = (1 - \pi_{-1}) E_g[\pi_t] = E[\pi_t (1 - \pi_t)]
```

这里 $E_f$ 和 $E_g$ 分别表示在数据从 $F$ 和从 $G$ 独立同分布抽取条件下的期望，而 $E$ 表示在决策者主观联合分布 {eq}`eq_definetti` 下的期望。

为证明 {eq}`eq_conv_identity`，设 $\phi$ 为历史 $w^t$ 的有界函数。

由于 $\pi_t = \mathbb{P}\{q = f \,|\, w^t\}$，迭代期望法则意味着

$$
E[\pi_t \phi(w^t)] = E[\mathbf{1}\{q = f\} \phi(w^t)] = \pi_{-1} E_f[\phi(w^t)] .
$$

令 $\phi = 1 - \pi_t$，即得 {eq}`eq_conv_identity` 中第一项与第三项相等。

将同样的论证应用于 $1 - \pi_t = \mathbb{P}\{q = g \,|\, w^t\}$，并令 $\phi = \pi_t$，即得第二项与第三项相等。

当 $\pi_{-1} = 0.5$ 时，该恒等式表明对每个 $t$ 都有 $E_f[1 - \pi_t] = E_g[\pi_t]$，因此两条模拟曲线之间的差异仅来自抽样误差。

若采用其他先验，这两条曲线之间相差一个常数倍数 $(1 - \pi_{-1})/\pi_{-1}$。

该恒等式并不意味着在 $F$ 和 $G$ 下，典型路径以相同的速率收敛。

根据 {eq}`eq_odds_product` 和大数定律，当 $F$ 生成数据时，$\frac{1}{t+1} \log \frac{\pi_t}{1 - \pi_t}$ 收敛到 $E_f[\log \ell(w)] = KL(f, g) = 0.759$；当 $G$ 生成数据时，它收敛到 $E_g[\log \ell(w)] = -KL(g, f) = -0.344$，其中 $KL(f, g) = \int f(w) \log \frac{f(w)}{g(w)} dw$ 是 $g$ 相对于 $f$ 的 Kullback–Leibler 散度。

因此，典型路径在 $F$ 下趋近 $1$ 的速度要快于其在 $G$ 下趋近 $0$ 的速度。

上图所绘制的平均值是由那些 $\pi_t$ 长期朝错误方向偏移的罕见路径主导的。

这两条平均曲线均以指数速率

$$
C(f, g) = - \log \min_{s \in [0,1]} \int f(w)^{s} g(w)^{1-s} dw = 0.121 ,
$$

衰减，这正是 {doc}`divergence_measures` 中所描述的 $f$ 和 $g$ 的切尔诺夫熵（Chernoff entropy）。

练习 {ref}`exchangeable_ex2` 对这些结论进行了验证。

### $\pi_t$ 的集合动态图

通过对相关概率分布进行积分计算 $\frac{\pi_{t+1}}{\pi_{t}}$ 的条件期望作为 $\pi_t$ 的函数，可以获得关于 $\{\pi_t\}$ 动态的更多见解：

$$
\begin{aligned}
E\left[\frac{\pi_{t+1}}{\pi_{t}}\biggm|q=a, \pi_{t}\right] &=E\left[\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\biggm|q= a, \pi_{t}\right], \\
    &=\int_{0}^{1}\frac{\ell\left(w_{t+1}\right)}{\pi_{t}\ell\left(w_{t+1}\right)+\left(1-\pi_{t}\right)} a\left(w_{t+1}\right)dw_{t+1}
\end{aligned}
$$

其中 $a =f,g$。

以下代码近似计算上述积分：

```{code-cell} ipython3
def plot_expected_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):
    # 直接构造 f 和 g：这里它们可能相同，此时 ℓ(w) = 1 没有根
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

对于任意一对 $f \neq g$ 以及任意 $\pi_t = \pi \in (0,1)$，都成立两个不等式。

设 $r(\ell) = \ell / (\pi \ell + 1 - \pi)$ 为比值 $\pi_{t+1}/\pi_t$ 作为 $\ell = \ell(w_{t+1})$ 的函数。

函数 $r$ 是严格凹的，且 $E_g[\ell(w)] = \int f(w) dw = 1$，因此由詹森不等式可得

$$
E_g[r(\ell(w))] < r(1) = 1 .
$$

两个期望按 $\pi$ 加权平均后等于一：

$$
\pi E_f[r(\ell(w))] + (1 - \pi) E_g[r(\ell(w))] = \int \frac{f(w)}{\pi f(w) + (1-\pi) g(w)} \bigl(\pi f(w) + (1-\pi) g(w)\bigr) dw = 1 .
$$

这个等式正是 {doc}`likelihood_bayes` 中所研究的、$\pi_t$ 在决策者主观分布下的鞅性质。

结合第一个不等式，可得

$$
E_f[r(\ell(w))] > 1 .
$$

因此，当数据由 $F$ 生成时 $\pi_t$ 是一个下鞅，而当数据由 $G$ 生成时 $\pi_t$ 是一个上鞅。

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

这两个不等式依然成立，但此时两条曲线都接近于一。

在所绘网格范围内，$E_f[\pi_{t+1}/\pi_t]$ 不超过 $1.16$，而 $E_g[\pi_{t+1}/\pi_t]$ 不低于 $0.978$。

而在第一种情形中，相应的数值分别为 $5.38$ 和 $0.811$。

贝塔分布 Beta$(2,1)$ 和 Beta$(3,1.2)$ 彼此十分接近：$KL(f, g) = 0.052$，$KL(g, f) = 0.039$，相比之下第一种情形中为 $0.759$ 和 $0.344$。

因此一次抽样平均只会让 $\pi_t$ 发生很小的变化，学习速度较慢。

## 后续内容

我们将在以下讲座中应用并深入探讨本讲座中提出的一些想法：

* {doc}`likelihood_ratio_process` 描述了**似然比过程**及其在频率派和贝叶斯统计理论中的作用
* {doc}`likelihood_bayes` 研究了似然比过程如何驱动后验概率 $\pi_t$，并证明 $\pi_t$ 在决策者的主观分布 {eq}`eq_definetti` 下是一个鞅
* {doc}`mix_model` 研究了一位使用本讲座统计模型的决策者，而自然实际上每期都重新混合 $F$ 和 $G$ 的情形
* {doc}`merging_of_opinions` 研究了贝塔-伯努利模型，它是德·费内蒂表示 {eq}`eq_definetti_bernoulli` 的一个实例，并探讨了两位具有不同先验的贝叶斯主义者的意见何时会趋同
* {doc}`divergence_measures` 描述了支配学习速率的库尔贝克-莱布勒散度和切尔诺夫熵
* {doc}`wald_friedman_2` 研究了一位持有本讲座信念的决策者如何依次决定何时停止抽样
* {doc}`navy_captain` 研究了二战时期一位美国海军上尉的直觉，即海军要求他使用的（频率派）决策规则不如亚伯拉罕·瓦尔德尚未设计的序贯规则

## 练习

```{exercise}
:label: exchangeable_ex1

本练习说明独立同分布序列的混合具有非负的序列协方差，并且有限可交换序列不一定是独立同分布序列的混合。

1. 证明等式 {eq}`eq_hcov`。

2. 设 $\phi$ 是一个使 $E_f[\phi(W)]$ 和 $E_g[\phi(W)]$ 均有限的函数。利用 {eq}`eq_hcov` 证明在联合密度 {eq}`eq_definetti` 下

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = \tilde \pi (1 - \tilde \pi) \bigl(E_f[\phi(W)] - E_g[\phi(W)]\bigr)^2 \geq 0 .
$$

3. 对于本讲座中的密度 $f = $ Beta$(1, 1)$ 和 $g = $ Beta$(3, 1.2)$，取 $\tilde \pi = 0.5$，使用第 2 部分中的公式计算 $\textrm{Cov}(W_0, W_1)$，并与模拟结果进行比较。

4. 一个瓮中有一个红球和一个黑球。我们不放回地抽取两个球。令 $W_t = 1$ 表示第 $t \in \{0, 1\}$ 次抽取为红色，$W_t = 0$ 表示为黑色。证明 $(W_0, W_1)$ 是可交换的，并且 $\textrm{Cov}(W_0, W_1) < 0$。解释为什么 $(W_0, W_1)$ 不是独立同分布对的混合，以及为什么德菲内蒂定理要求无限序列。
```

```{solution-start} exchangeable_ex1
:class: dropdown
```

第 1 部分。

令 $h(W_t) = \tilde \pi f(W_t) + (1 - \tilde \pi) g(W_t)$，则

$$
\begin{aligned}
h(W_0) h(W_1) & = \tilde \pi^2 f(W_0) f(W_1) + \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) g(W_1) + g(W_0) f(W_1)\bigr) + (1 - \tilde \pi)^2 g(W_0) g(W_1) .
\end{aligned}
$$

用 $h(W_0, W_1) = \tilde \pi f(W_0) f(W_1) + (1 - \tilde \pi) g(W_0) g(W_1)$ 减去上式，并利用 $\tilde \pi - \tilde \pi^2 = (1 - \tilde \pi) - (1 - \tilde \pi)^2 = \tilde \pi (1 - \tilde \pi)$，得到

$$
h(W_0, W_1) - h(W_0) h(W_1) = \tilde \pi (1 - \tilde \pi) \bigl(f(W_0) f(W_1) + g(W_0) g(W_1) - f(W_0) g(W_1) - g(W_0) f(W_1)\bigr) ,
$$

这可因式分解为 {eq}`eq_hcov` 的右边。

第 2 部分。

将 {eq}`eq_hcov` 乘以 $\phi(W_0) \phi(W_1)$ 并对 $W_0$ 和 $W_1$ 积分：

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = \tilde \pi (1 - \tilde \pi) \int \phi(w) \bigl(f(w) - g(w)\bigr) dw \int \phi(w) \bigl(f(w) - g(w)\bigr) dw = \tilde \pi (1 - \tilde \pi) \bigl(E_f[\phi(W)] - E_g[\phi(W)]\bigr)^2 .
$$

对于具有任意混合测度的独立同分布序列混合，同样的结论成立。

令 $Q$ 为从混合测度中抽取的分布，$W_t$ 是从该分布中独立同分布抽取的。

全协方差公式给出

$$
\textrm{Cov}\bigl(\phi(W_0), \phi(W_1)\bigr) = E\bigl[\textrm{Cov}(\phi(W_0), \phi(W_1) \,|\, Q)\bigr] + \textrm{Cov}\bigl(E[\phi(W_0) | Q], E[\phi(W_1) | Q]\bigr) = 0 + \textrm{Var}\bigl(E[\phi(W) | Q]\bigr) \geq 0 .
$$

第 3 部分。

Beta$(1,1)$ 和 Beta$(3, 1.2)$ 的均值分别为 $1/2$ 和 $3/4.2$。

```{code-cell} ipython3
π_tilde = 0.5
μ_f, μ_g = 1 / (1 + 1), 3 / (3 + 1.2)
cov_theory = π_tilde * (1 - π_tilde) * (μ_f - μ_g)**2

rng = np.random.default_rng(1234)
N = 1_000_000
nature_draws_f = rng.random(N) < π_tilde
W = np.where(nature_draws_f[:, None],
             rng.beta(1, 1, size=(N, 2)),
             rng.beta(3, 1.2, size=(N, 2)))
cov_sim = np.cov(W[:, 0], W[:, 1])[0, 1]

print(f"Cov(W_0, W_1): formula {cov_theory:.5f}, simulation {cov_sim:.5f}")
```

第 4 部分。

第一次抽取为红色的概率为 $1/2$，此后第二次抽取必定为黑色。

因此 $\textrm{Prob}(W_0 = 1, W_1 = 0) = \textrm{Prob}(W_0 = 0, W_1 = 1) = 1/2$，且 $\textrm{Prob}(W_0 = W_1) = 0$。

联合分布在交换 $W_0$ 和 $W_1$ 时保持不变，因此 $(W_0, W_1)$ 是可交换的。

$E[W_0] = E[W_1] = 1/2$，$E[W_0 W_1] = 0$，因此 $\textrm{Cov}(W_0, W_1) = -1/4$。

由第 2 部分可知，每一种独立同分布对的混合都满足 $\textrm{Cov}(W_0, W_1) \geq 0$。

所以 $(W_0, W_1)$ 不是独立同分布对的混合。

如果 $(W_0, W_1)$ 是某个无限可交换序列的前两个成员，那么根据德菲内蒂定理，该对将是独立同分布对的混合。

因此瓮序列无法扩展为无限可交换序列。

更一般地，设 $W_0, \ldots, W_{n-1}$ 可交换，具有共同方差 $\sigma^2 > 0$ 和共同成对相关系数 $\rho$。

则 $0 \leq \textrm{Var}(\sum_t W_t) = n \sigma^2 (1 + (n-1) \rho)$，因此 $\rho \geq -1/(n-1)$。

瓮在 $n = 2$ 且 $\rho = -1$ 时达到此界。

当 $n \rightarrow \infty$ 时，该界趋于 $0$。

```{solution-end}
```

```{exercise}
:label: exchangeable_ex2

本练习研究等式 {eq}`eq_conv_identity` 以及 $\pi_t$ 的收敛速度。

1. 通过第二种方法证明 {eq}`eq_conv_identity`。利用 $\pi_t = \pi_{-1} L_t / (\pi_{-1} L_t + 1 - \pi_{-1})$，其中 $L_t = \prod_{s=0}^t \ell(w_s)$，以及对于 $w^t$ 的每一个有界函数 $\phi$，都有 $E_f[\phi(w^t)] = E_g[L_t \phi(w^t)]$。

2. 设 $\pi_{-1} = 0.3$。绘制 $1 - E_f[\pi_t]$ 和 $E_g[\pi_t]$ 随 $t$ 的变化图，然后绘制 $\pi_{-1}(1 - E_f[\pi_t])$ 和 $(1 - \pi_{-1}) E_g[\pi_t]$ 随 $t$ 的变化图。

3. 计算在 $F$ 下和在 $G$ 下，路径间 $\log(\pi_t / (1 - \pi_t))$ 中位数随 $t$ 变化的斜率。将它们与 $KL(f, g)$ 和 $-KL(g, f)$ 进行比较。

4. 计算 $\log(1 - E_f[\pi_t])$ 随 $t$ 变化的斜率，并将其与 $-C(f, g)$ 进行比较，其中 $C(f, g)$ 是切诺夫熵。

第 4 部分提示：罕见路径主导了 $1 - E_f[\pi_t]$ 的值，因此对于较大的 $t$，普通的蒙特卡洛估计并不精确。

令 $A(s) = \int f(w)^s g(w)^{1-s} dw$，令 $s^*$ 为在 $[0,1]$ 上使 $A(s)$ 最小化的值，令 $m(w) = f(w)^{s^*} g(w)^{1-s^*} / A(s^*)$。

那么 $f(w) / m(w) = A(s^*) \ell(w)^{1 - s^*}$，因此

$$
E_f[\phi(w^t)] = A(s^*)^{t+1} E_m\bigl[\phi(w^t) L_t^{1-s^*}\bigr] .
$$

从 $m$ 中抽取样本以估计右边的期望。

当 $f$ 为均匀分布时，$m$ 为 Beta$(3 - 2 s^*, 1.2 - 0.2 s^*)$ 密度。
```

```{solution-start} exchangeable_ex2
:class: dropdown
```

第 1 部分。

在 $F$ 下，$w^t$ 的密度为 $f(w_0) \cdots f(w_t) = L_t \, g(w_0) \cdots g(w_t)$，由此得到 $E_f[\phi(w^t)] = E_g[L_t \phi(w^t)]$。

由于 $1 - \pi_t = (1 - \pi_{-1}) / (\pi_{-1} L_t + 1 - \pi_{-1})$，

$$
\pi_{-1} E_f[1 - \pi_t] = E_g\left[\frac{\pi_{-1} L_t (1 - \pi_{-1})}{\pi_{-1} L_t + 1 - \pi_{-1}}\right] = (1 - \pi_{-1}) E_g[\pi_t] .
$$

主观分布 {eq}`eq_definetti` 的密度为 $\pi_{-1} f(w^t) + (1 - \pi_{-1}) g(w^t) = (\pi_{-1} L_t + 1 - \pi_{-1}) g(w^t)$，因此

$$
E[\pi_t (1 - \pi_t)] = E_g\bigl[(\pi_{-1} L_t + 1 - \pi_{-1}) \pi_t (1 - \pi_t)\bigr] = E_g\bigl[\pi_{-1} L_t (1 - \pi_t)\bigr] = \pi_{-1} E_f[1 - \pi_t] .
$$

第 2 部分。

我们编写一个返回对数赔率的 `simulate` 版本，因为一旦 $\pi_t$ 的对数赔率超过约 $37$，它就会四舍五入为恰好 $1$。

```{code-cell} ipython3
def simulate_log_odds(rng, a, b, T, N, π_init):
    "Log odds of π_t for t = -1, 0, ..., T-1 along N paths of IID Beta(a, b) draws."
    w = rng.beta(a, b, size=(N, T))
    log_l = np.log(p(w, 1, 1)) - np.log(p(w, 3, 1.2))
    log_odds_init = np.log(π_init / (1 - π_init))
    log_odds = log_odds_init + np.cumsum(log_l, axis=1)
    return np.column_stack([np.full(N, log_odds_init), log_odds])

π_init, T_ex, N_ex = 0.3, 40, 100_000
rng = np.random.default_rng(2024)
lo_F = simulate_log_odds(rng, 1, 1, T_ex, N_ex, π_init)
lo_G = simulate_log_odds(rng, 3, 1.2, T_ex, N_ex, π_init)
t_ex = np.arange(-1, T_ex)

err_F = np.mean(expit(-lo_F), axis=0)      # 1 - E_f[π_t]
err_G = np.mean(expit(lo_G), axis=0)       # E_g[π_t]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(t_ex, err_F, label=r'$1 - E_f[\pi_t]$')
axes[0].plot(t_ex, err_G, label=r'$E_g[\pi_t]$')
axes[1].plot(t_ex, π_init * err_F, label=r'$\pi_{-1}(1 - E_f[\pi_t])$')
axes[1].plot(t_ex, (1 - π_init) * err_G, '--',
             label=r'$(1 - \pi_{-1}) E_g[\pi_t]$')
for ax in axes:
    ax.set_xlabel('$t$')
    ax.legend()
plt.tight_layout()
plt.show()

for t in [0, 5, 10, 20]:
    print(f"t = {t:2d}: (1 - E_f[π_t]) / E_g[π_t] = {err_F[t+1] / err_G[t+1]:.3f}")
print(f"(1 - π_init) / π_init = {(1 - π_init) / π_init:.3f}")
```

左图中的两条曲线相差一个 $7/3$ 的因子；右图中的两条曲线在抽样误差范围内一致。

第 3 部分。

```{code-cell} ipython3
f = lambda w: p(w, 1, 1)
g = lambda w: p(w, 3, 1.2)
KL_fg = quad(lambda w: f(w) * np.log(f(w) / g(w)), 0, 1)[0]
KL_gf = quad(lambda w: g(w) * np.log(g(w) / f(w)), 0, 1)[0]

slope_F = np.polyfit(t_ex, np.median(lo_F, axis=0), 1)[0]
slope_G = np.polyfit(t_ex, np.median(lo_G, axis=0), 1)[0]
print(f"slope of median log odds under F: {slope_F:.3f},  KL(f, g) = {KL_fg:.3f}")
print(f"slope of median log odds under G: {slope_G:.3f}, -KL(g, f) = {-KL_gf:.3f}")
```

典型路径在 $F$ 下比在 $G$ 下收敛得更快。

第 4 部分。

我们计算切诺夫熵及其最小值点 $s^*$。

```{code-cell} ipython3
A = lambda s: quad(lambda w: f(w)**s * g(w)**(1 - s), 0, 1)[0]
res = op.minimize_scalar(A, bounds=(0, 1), method='bounded')
s_star, C = res.x, -np.log(res.fun)
print(f"s* = {s_star:.3f}, C(f, g) = {C:.4f}")
```

接下来我们通过从 $m$ 进行重要性抽样，估计直到 $t=199$ 为止的 $\log(1 - E_f[\pi_t])$。

```{code-cell} ipython3
T_is, N_is = 200, 100_000
rng = np.random.default_rng(2025)
w = rng.beta(3 - 2 * s_star, 1.2 - 0.2 * s_star, size=(N_is, T_is))
log_L = np.cumsum(np.log(f(w)) - np.log(g(w)), axis=1)    # log L_t, t = 0, ..., 199
t_is = np.arange(T_is)

log_odds = np.log(π_init / (1 - π_init)) + log_L
weights = np.exp((1 - s_star) * log_L)                    # L_t^(1 - s*)
log_err_F = (t_is + 1) * np.log(A(s_star)) \
            + np.log(np.mean(expit(-log_odds) * weights, axis=0))

for t in [10, 20, 30]:
    print(f"t = {t}: log(1 - E_f[π_t]) importance sampling {log_err_F[t]:.3f}, "
          f"plain Monte Carlo {np.log(err_F[t+1]):.3f}")

late = t_is >= 100
slope = np.polyfit(t_is[late], log_err_F[late], 1)[0]
slope_adj = np.polyfit(t_is[late], log_err_F[late] + 0.5 * np.log(t_is[late] + 1), 1)[0]
print(f"slope of log(1 - E_f[π_t]) for t in [100, 199]: {slope:.4f}")
print(f"slope after adding log(t+1)/2:                 {slope_adj:.4f}")
print(f"-C(f, g):                                      {-C:.4f}")
```

在较小的 $t$ 时，两种估计结果是一致的。

在 $m$ 下，$E_m[\log \ell(w)] = A'(s^*)/A(s^*) = 0$，因此 $\log L_t$ 是一个没有漂移的随机游走。

因此，期望 $E_m[(1 - \pi_t) L_t^{1-s^*}]$ 的衰减仅为 $t^{-1/2}$ 量级，于是

$$
1 - E_f[\pi_t] \approx K \, t^{-1/2} e^{-C(f,g) t}
$$

其中 $K$ 为常数。

拟合得到的斜率接近 $-C(f,g)$，加上 $\frac{1}{2}\log(t+1)$ 以消除 $t^{-1/2}$ 因子后，结果更接近该值。

根据 {eq}`eq_conv_identity`，$E_g[\pi_t]$ 以相同的速率衰减。

```{solution-end}
```

```{exercise}
:label: exchangeable_ex3

一个波利亚瓮最初有 $a$ 个红球和 $b$ 个黑球。

在每个时刻 $t = 0, 1, \ldots$，我们随机抽取一个球，若为红色则设 $W_t = 1$，若为黑色则设 $W_t = 0$，然后将其放回瓮中，并再加入一个同色的球。

1. 证明 $\textrm{Prob}(W_0 = 1, W_1 = 0) = \textrm{Prob}(W_0 = 0, W_1 = 1)$。更一般地，证明序列 $(w_0, \ldots, w_{n-1})$ 出现的概率仅取决于红色抽取的次数 $k$，因此 $W_0, W_1, \ldots$ 是可交换的。求出德菲内蒂表示 {eq}`eq_definetti_bernoulli` 中的混合测度 $\mu$。

2. 模拟多个瓮，取 $a = 2$，$b = 3$，抽取 $n = 2000$ 次。比较各瓮间红色抽取比例的分布与 Beta$(a, b)$ 密度。

3. 证明在 $n$ 次抽取（其中 $k$ 次为红色）之后，瓮中红球的比例，等于一个具有 Beta$(a, b)$ 先验（针对独立同分布伯努利抽取的成功概率 $\theta$）的贝叶斯主义者赋予下一次抽取为红色的概率 $\textrm{Prob}(W_n = 1 \,|\, w_0, \ldots, w_{n-1})$。
```

```{solution-start} exchangeable_ex3
:class: dropdown
```

第 1 部分。

$\textrm{Prob}(W_0 = 1, W_1 = 0) = \frac{a}{a+b} \frac{b}{a+b+1}$，$\textrm{Prob}(W_0 = 0, W_1 = 1) = \frac{b}{a+b} \frac{a}{a+b+1}$。

两者相等。

对于 $n$ 次抽取中有 $k$ 次为红色的序列，无论顺序如何，连续条件概率的分母均为 $a+b, a+b+1, \ldots, a+b+n-1$。

红色抽取对应的分子为 $a, a+1, \ldots, a+k-1$，黑色抽取对应的分子为 $b, b+1, \ldots, b+n-k-1$，顺序任意。

因此

$$
\textrm{Prob}(W_0 = w_0, \ldots, W_{n-1} = w_{n-1}) = \frac{\prod_{j=0}^{k-1} (a+j) \prod_{j=0}^{n-k-1} (b+j)}{\prod_{j=0}^{n-1} (a+b+j)} = \frac{B(a+k, b+n-k)}{B(a,b)} ,
$$

其中 $B$ 为贝塔函数。

上式右边等于 $\int_0^1 \theta^k (1-\theta)^{n-k} \mu(d\theta)$，其中 $\mu$ 为 Beta$(a,b)$ 分布。

这就是德菲内蒂表示 {eq}`eq_definetti_bernoulli`。

第 2 部分。

```{code-cell} ipython3
from scipy.stats import beta as beta_dist, kstest

def polya_urn(rng, a, b, n, N):
    "Simulate N Pólya urns for n draws; return the N x n array of draws."
    red = np.full(N, float(a))
    total = float(a + b)
    W = np.empty((N, n), dtype=np.int8)
    for t in range(n):
        W[:, t] = rng.random(N) < red / total
        red += W[:, t]
        total += 1
    return W

a, b, n, N = 2, 3, 2000, 20_000
rng = np.random.default_rng(7)
W = polya_urn(rng, a, b, n, N)
freq = W.mean(axis=1)

grid = np.linspace(0, 1, 200)
fig, ax = plt.subplots()
ax.hist(freq, bins=50, density=True, alpha=0.5,
        label=f'fraction of red draws, $n = {n}$')
ax.plot(grid, beta_dist.pdf(grid, a, b), lw=2, label=f'Beta$({a}, {b})$ density')
ax.set_xlabel(r'$\theta$')
ax.legend()
plt.show()

print(f"Prob(W_0=1, W_1=0) = {np.mean((W[:, 0] == 1) & (W[:, 1] == 0)):.4f}, "
      f"Prob(W_0=0, W_1=1) = {np.mean((W[:, 0] == 0) & (W[:, 1] == 1)):.4f}, "
      f"exact = {a * b / ((a + b) * (a + b + 1)):.4f}")
print(f"Kolmogorov-Smirnov test against Beta({a}, {b}): "
      f"p-value = {kstest(freq, beta_dist(a, b).cdf).pvalue:.3f}")
```

各瓮间红色抽取比例的分布近似为 Beta$(2, 3)$，科尔莫戈罗夫-斯米尔诺夫检验未能拒绝该分布。

第 3 部分。

在 $n$ 次抽取（其中 $k$ 次为红色）之后，瓮中共有 $a + k$ 个红球，总球数为 $a + b + n$，因此下一次抽取为红色的概率为 $(a+k)/(a+b+n)$。

一个对 $\theta$ 具有 Beta$(a, b)$ 先验的贝叶斯主义者，在观察到 $n$ 次独立同分布伯努利试验中有 $k$ 次成功后，得到 Beta$(a + k, b + n - k)$ 后验分布。

他认为下一次抽取为成功的概率，等于 $\theta$ 的后验均值，即 $(a + k)/(a + b + n)$。

瓮与该贝叶斯主义者的结论一致。

下面的代码单元根据前 $10$ 次抽取中红球的数量，对模拟的瓮进行分组，以此验证这一点。

基于较少瓮计算的频率精度较低。

```{code-cell} ipython3
n0 = 10
k = W[:, :n0].sum(axis=1)
for k_val in range(0, n0 + 1, 2):
    sel = k == k_val
    print(f"k = {k_val:2d} ({sel.sum():4d} urns): "
          f"frequency of red at draw {n0} = {W[sel, n0].mean():.3f}, "
          f"(a + k) / (a + b + n) = {(a + k_val) / (a + b + n0):.3f}")
```

```{solution-end}
```
