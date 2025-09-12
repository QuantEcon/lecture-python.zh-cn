---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(likelihood_ratio_process_2)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 异质信念与金融市场

```{contents} 目录
:depth: 2
```

## 概述

似然比过程是Lawrence Blume和David Easley回答他们提出的问题"如果你那么聪明，为什么不富有？"的基础 {cite}`blume2006if`。

Blume和Easley构建了正式模型，研究关于风险收入过程概率的不同观点如何影响结果，以及如何反映在个人用来分享和对冲风险的股票、债券和保险政策的价格中。

```{note}
{cite}`alchian1950uncertainty`和{cite}`friedman1953essays`推测，通过奖励具有更现实概率模型的交易者，金融证券的竞争市场将财富置于信息更充分的交易者手中，并有助于使风险资产的价格反映现实的概率评估。
```

在这里，我们将提供一个示例来说明Blume和Easley分析的基本组成部分。

我们将只关注他们对完全市场环境的分析，在这种环境中可以进行所有可想象的风险证券交易。

我们将研究两种替代安排：

* 完全的社会主义制度，在这种制度下，个人每期都将消费品的禀赋交给中央计划者，然后由计划者独裁式地分配这些商品
* 分散的竞争市场体系，在这种体系中，自私的价格接受者在竞争市场中自愿相互交易

福利经济学的基本定理将适用，并向我们保证这两种安排最终会产生完全相同的消费品分配结果，**前提是**社会计划者分配了一组适当的**帕累托权重**。

```{note}
你可以在{doc}`这篇关于规划问题的讲座 <cass_koopmans_1>`和{doc}`这篇关于相关竞争均衡的讲座 <cass_koopmans_2>`中了解现代宏观经济模型如何应用这两个福利定理。{doc}`这篇quantecon讲座 <ge_arrow>`介绍了具有同质信念的完全市场模型的递归表述。
```

让我们首先导入一些Python工具。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit, prange
from math import gamma
from scipy.integrate import quad
```

## 回顾：似然比过程

让我们首先回顾似然比过程的定义和性质。

一个非负随机变量 $W$ 具有两个概率密度函数之一，要么是 $f$，要么是 $g$。

在时间开始之前，自然界一劳永逸地决定是从 $f$ 还是 $g$ 中进行一系列独立同分布的抽样。

我们有时用 $q$ 表示自然界永久选择的密度，所以 $q$ 要么是 $f$，要么是 $g$，且是永久性的。

自然界知道它永久从哪个密度中抽样，但我们这些观察者并不知道。

我们知道 $f$ 和 $g$ 两个密度，但不知道自然界选择了哪个。

但我们想要知道。

为此，我们使用观测值。

我们观察到一个序列 $\{w_t\}_{t=1}^T$，包含 $T$ 个独立同分布的抽样，我们知道这些抽样要么来自 $f$，要么来自 $g$。

我们想要利用这些观测值来推断自然界选择了 $f$ 还是 $g$。

**似然比过程**是完成这项任务的有用工具。

首先，我们定义似然比过程的一个关键组成部分，即时间 $t$ 的似然比，它是一个随机变量：

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

我们假设 $f$ 和 $g$ 在随机变量 $W$ 的相同可能实现区间上都赋予正概率。

这意味着在 $g$ 密度下，$\ell (w_t)= \frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$ 是一个均值为1的非负随机变量。

序列的**似然比过程**为

$\left\{ w_{t}\right\} _{t=1}^{\infty}$ 定义为

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

其中 $w^t=\{ w_1,\dots,w_t\}$ 是直到时间 $t$ (包括 $t$) 的观测历史。

有时为简便起见，我们会写 $L_t = L(w^t)$。

注意，似然过程满足以下*递归*关系

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

似然比及其对数是使用 Neyman 和 Pearson 经典频率派方法进行推断的关键工具 {cite}`Neyman_Pearson`。

为了帮助我们理解其工作原理，以下 Python 代码将 $f$ 和 $g$ 评估为两个不同的 Beta 分布，然后通过从两个概率分布中的一个生成序列 $w^t$（例如，从 $g$ 生成的 IID 抽样序列）来计算和模拟相关的似然比过程。

```{code-cell} ipython3
# 两个 Beta 分布的参数
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# 两个密度函数
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
```

```{code-cell} ipython3
@jit
def simulate(a, b, T=50, N=500):
    '''
    生成N组T个似然比观测值，
    以N x T矩阵形式返回。
    '''

    l_arr = np.empty((N, T))

    for i in range(N):
        for j in range(T):
            w = np.random.beta(a, b)
            l_arr[i, j] = f(w) / g(w)

    return l_arr
```

## Blume和Easley的设定

令随机变量 $s_t \in (0,1)$ 在时间 $t =0, 1, 2, \ldots$ 按照具有参数 $\theta = \{\theta_1, \theta_2\}$ 的相同Beta分布分布。

我们将这个概率密度表示为

$$
\pi(s_t|\theta)
$$

为了节省空间，下面我们通常会直接写 $\pi(s_t)$ 而不是 $\pi(s_t|\theta)$。

令 $s_t \equiv y_t^1$ 为我们称为"个体1"在时间 $t$ 获得的不可储存消费品的禀赋。

令历史 $s^t = [s_t, s_{t-1}, \ldots, s_0]$ 为具有联合分布的独立同分布随机变量序列

$$
\pi_t(s^t) = \pi(s_t) \pi(s_{t-1}) \cdots \pi(s_0)
$$

因此在我们的例子中，历史 $s^t$ 是从时间 $0$ 到时间 $t$ 个体 $1$ 的消费品禀赋的完整记录。

如果个体 $1$ 独自生活在一个岛上，个体 $1$ 在时间 $t$ 的消费 $c^1(s_t)$ 是

$$c^1(s_t) = y_t^1 = s_t. $$

但在我们的模型中，个体1并不是孤独的。

## 自然和个体的信念

自然从 $\pi_t(s^t)$ 中抽取独立同分布序列 $\{s_t\}_{t=0}^\infty$。

* 所以没有上标的 $\pi$ 是自然的模型
* 但除了自然之外，我们的模型中还有其他实体——我们称之为"个体"的人工个体
* 每个个体对 $t=0, \ldots$ 时的 $s^t$ 都有一个概率分布序列
* 个体 $i$ 认为自然从 $\{\pi_t^i(s^t)\}_{t=0}^\infty$ 中抽取独立同分布序列 $\{s_t\}_{t=0}^\infty$

* 除非 $\pi_t^i(s^t) = \pi_t(s^t)$，否则个体 $i$ 是错误的

```{note}
**理性预期**模型会对所有个体 $i$ 设定 $\pi_t^i(s^t) = \pi_t(s^t)$。
```

有两个个体，分别标记为 $i=1$ 和 $i=2$。

在时间 $t$，个体 $1$ 获得一个不可储存的消费品的禀赋

$$
y_t^1 = s_t 
$$

而个体 $2$ 获得禀赋

$$
y_t^2 = 1 - s_t 
$$

消费品的总禀赋为

$$
y_t^1 + y_t^2 = 1
$$

这在每个时间点 $t \geq 0$ 都成立。

在时间 $t$，个体 $i$ 消费 $c_t^i(s^t)$ 单位的商品。

每期总禀赋为1的(无浪费的)可行分配满足

$$
c_t^1 + c_t^2 = 1 .
$$

## 社会主义风险分担安排

为了分担风险，一个仁慈的社会规划者制定了一个依赖于历史的消费分配，它采用一系列函数的形式

$$
c_t^i = c_t^i(s^t)
$$

这些函数满足

$$
c_t^1(s^t) + c_t^2(s^t) = 1  
$$ (eq:feasibility)

对所有 $s^t$ 和所有 $t \geq 0$ 成立。

为了设计一个社会最优分配，社会规划者需要知道个体1对禀赋序列的信念以及他们对承担风险的态度。

关于禀赋序列，个体 $i$ 认为自然从联合密度中独立同分布地抽取序列

$$
\pi_t^i(s^t) = \pi^i(s_t) \pi^i(s_{t-1}) \cdots \pi^i(s_0)
$$ 

关于承担风险的态度，个体 $i$ 有一个单期效用函数

$$
u(c_t^i) = \ln (c_t^i)
$$

在第$t$期的消费边际效用为

$$
u'(c_t^i) = \frac{1}{c_t^i}
$$

将其对随机禀赋序列的信念和对承担风险的态度结合起来，个体$i$的跨期效用函数为

$$
V^i = \sum_{t=0}^{\infty} \sum_{s^t} \delta^t u(c_t^i(s^t)) \pi_t^i(s^t) ,
$$ (eq:objectiveagenti)

其中$\delta \in (0,1)$是跨期贴现因子，$u(\cdot)$是严格递增、凹的单期效用函数。

## 社会规划者的分配问题

仁慈的独裁者拥有所需的全部信息来选择一个消费分配，以最大化社会福利标准

$$
W = \lambda V^1 + (1-\lambda) V^2
$$ (eq:welfareW)

其中$\lambda \in [0,1]$是帕累托权重，表示规划者对个体$1$的偏好程度，而$1 - \lambda$是帕累托权重，表示社会规划者对个体$2$的偏好程度。

设定$\lambda = .5$表示"平等主义"的社会偏好。

注意社会福利标准{eq}`eq:welfareW`如何通过公式{eq}`eq:objectiveagenti`考虑了两个个体的偏好。

这意味着社会规划者知道并尊重：

* 每个个体的单期效用函数$u(\cdot) = \ln(\cdot)$
* 每个个体$i$的概率模型$\{\pi_t^i(s^t)\}_{t=0}^\infty$

因此，我们预期这些对象将出现在社会规划者分配每期总禀赋的规则中。

对福利标准{eq}`eq:welfareW`在可行性约束{eq}`eq:feasibility`下最大化的一阶必要条件是

$$\frac{\pi_t^2(s^t)}{\pi_t^1(s^t)} \frac{(1/c_t^2(s^t))}{(1/c_t^1(s^t))} = \frac{\lambda}{1-\lambda}$$

可以重新整理为

$$
\frac{c_t^1(s^t)}{c_t^2(s^t)} = \frac{\lambda}{1-\lambda} l_t(s^t)
$$ (eq:allocationrule0)

其中

$$ l_t(s^t) = \frac{\pi_t^1(s^t)}{\pi_t^2(s^t)} $$

是个体1的联合密度与个体2的联合密度的似然比。

使用

$$c_t^1(s^t) + c_t^2(s^t) = 1$$

我们可以将分配规则{eq}`eq:allocationrule0`重写为

$$\frac{c_t^1(s^t)}{1 - c_t^1(s^t)} = \frac{\lambda}{1-\lambda} l_t(s^t)$$

或

$$c_t^1(s^t) = \frac{\lambda}{1-\lambda} l_t(s^t)(1 - c_t^1(s^t))$$

这意味着社会规划者的分配规则是

$$
c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda + \lambda l_t(s^t)}
$$ (eq:allocationrule1)

如果我们定义一个临时或**延续帕累托权重**过程为

$$
\lambda_t(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda + \lambda l_t(s^t)},
$$

那么我们可以将社会规划者的分配规则表示为

$$
c_t^1(s^t) = \lambda_t(s^t) .
$$

## 如果你这么聪明，$\ldots$

让我们计算一下对于一些有趣的似然比过程$l_t(s^t)$的极限值，其极限分配{eq}`eq:allocationrule1`的值：

$$l_\infty (s^\infty)= 1; \quad c_\infty^1 = \lambda$$

* 在上述情况下，两个个体同样聪明（或同样不聪明），消费分配在两个个体之间保持在 $\lambda, 1 - \lambda$ 的分配比例。

$$l_\infty (s^\infty) = 0; \quad c_\infty^1 = 0$$

* 在上述情况下，个体2比个体1"更聪明"，个体1在总禀赋中的份额趋近于零。

$$l_\infty (s^\infty)= \infty; \quad c_\infty^1 = 1$$

* 在上述情况下，个体1比个体2更聪明，个体1在总禀赋中的份额趋近于1。

```{note}
这三种情况某种程度上告诉我们随着时间推移个体的相对**财富**是如何演变的。
* 当两个个体同样聪明且 $\lambda \in (0,1)$ 时，个体1的财富份额永远保持在 $\lambda$。
* 当个体1更聪明且 $\lambda \in (0,1)$ 时，个体1最终"拥有"全部的延续禀赋，而个体2最终"一无所有"。
* 当个体2更聪明且 $\lambda \in (0,1)$ 时，个体2最终"拥有"全部的延续禀赋，而个体1最终"一无所有"。
延续财富可以在我们引入竞争均衡**价格**体系后被精确定义。
```

很快我们将进行一些模拟，这将进一步阐明可能的结果。

但在此之前，让我们先转向研究社会规划问题的一些"影子价格"，这些价格可以很容易地转换为竞争均衡的"均衡价格"。

这样做将使我们能够将分析与{cite}`alchian1950uncertainty`和{cite}`friedman1953essays`的论点联系起来,即竞争市场过程可以使风险资产的价格更好地反映现实的概率评估。

## 竞争均衡价格

一般均衡模型的两个基本福利定理使我们预期,在我们一直研究的社会规划问题的解决方案与具有完整历史或有商品市场的**竞争均衡**配置之间存在联系。

```{note}
关于两个福利定理及其历史,请参见 <https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics>。
另外,关于经典宏观经济增长模型的应用,请参见{doc}`这篇关于规划问题的讲座 <cass_koopmans_1>`和{doc}`这篇关于相关竞争均衡的讲座 <cass_koopmans_2>`
```

这种联系在我们的模型中也存在。

我们现在来简要说明。

在竞争均衡中,不存在独裁式地收集每个人的禀赋然后重新分配的社会规划者。

相反,存在一个在某个时间点举行的全面的集中市场。

有**价格**,价格接受者可以按这些价格买卖他们想要的任何商品。

贸易是多边的,因为存在一个生活在模型之外的"瓦尔拉斯拍卖师",其工作是验证

每个个体的预算约束都得到满足。

这个预算约束涉及个体的禀赋流总值和消费流总值。

这些价值是根据个体视为既定的价格向量计算的——他们是"价格接受者"，假定他们可以按这些价格买入或卖出任何数量。

假设在时间$-1$（即时间$0$开始之前），个体$i$可以以价格$p_t(s^t)$购买一单位在历史$s^t$后时间$t$的消费$c_t(s^t)$。

注意这是一个（很长的）价格**向量**。

* 对每个历史$s^t$和每个日期$t = 0, 1, \ldots, $都有一个价格$p_t(s^t)$。
* 所以价格的数量与历史和日期的数量一样多。

这些价格在经济开始前的时间$-1$确定。

市场在时间$-1$只开放一次。

在时间$t =0, 1, 2, \ldots$执行在时间$-1$达成的交易。

* 在背景中，有一个"执行"程序强制个体履行他们在时间$-1$同意的交换或"交付"。

我们想研究个体的信念如何影响均衡价格。

个体$i$面临**单一**的跨期预算约束

$$
\sum_{t=0}^\infty\sum_{s^t} p_t(s^t) c_t^i (s^t) \leq \sum_{t=0}^\infty\sum_{s^t} p_t(s^t) y_t^i (s^t)
$$ (eq:budgetI)

根据预算约束{eq}`eq:budgetI`，交易在以下意义上是**多边的**

* 我们可以想象个体 $i$ 首先出售他的随机禀赋流 $\{y_t^i (s^t)\}$，然后用所得收益（即他的"财富"）购买随机消费流 $\{c_t^i (s^t)\}$。

个体 $i$ 在 {eq}`eq:budgetI` 上设置拉格朗日乘数 $\mu_i$，并一次性选择消费计划 $\{c^i_t(s^t)\}_{t=0}^\infty$ 以最大化目标函数 {eq}`eq:objectiveagenti`，同时受预算约束 {eq}`eq:budgetI` 的限制。

这意味着个体 $i$ 需要选择多个对象，即对于 $t = 0, 1, 2, \ldots$ 的所有 $s^t$ 的 $c_t^i(s^t)$。

为方便起见，让我们回顾一下在 {eq}`eq:objectiveagenti` 中定义的目标函数 $V^i$：

$$
V^i = \sum_{t=0}^{\infty} \sum_{s^t} \delta^t u(c_t^i(s^t)) \pi_t^i(s^t)
$$

最大化目标函数 $V^i$（在 {eq}`eq:objectiveagenti` 中定义）关于 $c_t^i(s^t)$ 的一阶必要条件是：

$$
\delta^t u'(c^i_t(s^t)) \pi_t^i(s^t) = \mu_i p_t(s^t) ,
$$

我们可以重新整理得到：

$$
p_t(s^t) = \frac{ \delta^t \pi_t^i(s^t)}{\mu_i c^i_t(s^t)}   
$$ (eq:priceequation1)

对于 $i=1,2$。

如果我们将个体1的方程 {eq}`eq:priceequation1` 除以个体2的相应方程，使用 $c^2_t(s^t) = 1 - c^1_t(s^t)$，并进行一些代数运算，我们将得到：

$$
c_t^1(s^t) = \frac{\mu_1 l_t(s^t)}{\mu_2 + \mu_1 l_t(s^t)} .
$$ (eq:allocationce)

我们现在进行一个扩展的"猜测和验证"练习，涉及将我们的竞争均衡中的对象与社会规划问题中的对象进行匹配。

* 我们将规划问题中的消费分配与竞争均衡中的均衡消费分配相匹配
* 我们将规划问题中的"影子"价格与竞争均衡价格相匹配

注意，如果我们设定$\mu_1 = 1-\lambda$且$\mu_2 = \lambda$，那么公式{eq}`eq:allocationce`就与公式{eq}`eq:allocationrule1`一致。

  * 这相当于为价格系统$\{p_t(s^t)\}_{t=0}^\infty$选择一个**计价单位**或标准化

```{note}
关于在像我们这样只决定相对价格的模型中，如何选择计价单位来确定绝对价格水平的信息，请参见<https://en.wikipedia.org/wiki/Num%C3%A9raire>。
```

如果我们将公式{eq}`eq:allocationce`代入公式{eq}`eq:priceequation1`中的$c_t^1(s^t)$并重新整理，我们得到

$$
p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr]
$$

或

$$
p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)}  \bigl[(1 - \lambda) \pi_t^2(s^t) + \lambda \pi_t^1(s^t)\bigr]
$$ (eq:pformulafinal)

根据公式{eq}`eq:pformulafinal`，我们有以下可能的极限情况：

* 当 $l_\infty = 0$ 时，$c_\infty^1 = 0$，竞争均衡价格的尾部反映了个体 $2$ 的概率模型 $\pi_t^2(s^t)$，即 $p_t(s^t) \propto \delta^t \pi_t^2(s^t)$
* 当 $l_\infty = \infty$ 时，$c_\infty^1 = 1$，竞争均衡价格的尾部反映了个体 $1$ 的概率模型 $\pi_t^1(s^t)$，即 $p_t(s^t) \propto \delta^t \pi_t^1(s^t)$
* 对于较小的 $t$，竞争均衡价格反映了两个个体的概率模型。

我们将影子价格的验证留给读者，因为它遵循相同的推理过程。

## 模拟

现在让我们实现一些模拟，其中个体 $1$ 相信边际密度

$$\pi^1(s_t) = f(s_t)$$

而个体 $2$ 相信边际密度

$$\pi^2(s_t) = g(s_t)$$

这里 $f$ 和 $g$ 是 Beta 分布，类似于我们在本讲座前面章节中使用的分布。

同时，我们假设自然界相信边际密度

$$
\pi(s_t) = h(s_t)
$$

其中 $h(s_t)$ 可能是 $f$ 和 $g$ 的混合。

首先，我们编写一个函数来计算似然比过程

```{code-cell} ipython3
def compute_likelihood_ratios(sequences, f, g):
    """计算似然比和累积乘积。"""
    l_ratios = f(sequences) / g(sequences)
    L_cumulative = np.cumprod(l_ratios, axis=1)
    return l_ratios, L_cumulative
```

让我们通过求积分计算Kullback-Leibler差异。

```{code-cell} ipython3
def compute_KL(f, g):
    """
    计算KL散度 KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val
```

我们还创建一个辅助函数来计算相对于参考分布$h$的KL散度

```{code-cell} ipython3
def compute_KL_h(h, f, g):
    """
    计算相对于参考分布h的KL散度
    """

    Kf = compute_KL(h, f)
    Kg = compute_KL(h, g)

    return Kf, Kg
```

让我们编写一个Python函数来计算个体1的消费份额

```{code-cell} ipython3
def simulate_blume_easley(sequences, f_belief=f, g_belief=g, λ=0.5):
    """模拟Blume-Easley模型的消费份额。"""
    l_ratios, l_cumulative = compute_likelihood_ratios(sequences, f_belief, g_belief)
    c1_share = λ * l_cumulative / (1 - λ + λ * l_cumulative)
    return l_cumulative, c1_share
```

现在让我们使用这个函数来生成以下序列：

* 每期自然从 $f$ 中抽取，或者
* 每期自然从 $g$ 中抽取，或者
* 每期自然抛一枚公平硬币来决定是从 $f$ 还是从 $g$ 中抽取

```{code-cell} ipython3
λ = 0.5
T = 100
N = 10000

# 自然遵循 f、g 或混合
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

h = jit(lambda x: 0.5 * f(x) + 0.5 * g(x))
model_choices = np.random.rand(N, T) < 0.5
s_seq_h = np.empty((N, T))
s_seq_h[model_choices] = np.random.beta(F_a, F_b, size=model_choices.sum())
s_seq_h[~model_choices] = np.random.beta(G_a, G_b, size=(~model_choices).sum())

l_cum_f, c1_f = simulate_blume_easley(s_seq_f)
l_cum_g, c1_g = simulate_blume_easley(s_seq_g)
l_cum_h, c1_h = simulate_blume_easley(s_seq_h)
```

在查看下图之前，让我们先来猜一猜，随着时间推移，在我们的三种情况下，个体1或个体2的消费份额会变得越来越大。

为了做出更好的猜测，让我们来可视化这三种情况下的似然比过程的实例。

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

titles = ["Nature = f", "Nature = g", "Nature = mixture"]
data_pairs = [(l_cum_f, c1_f), (l_cum_g, c1_g), (l_cum_h, c1_h)]

for i, ((l_cum, c1), title) in enumerate(zip(data_pairs, titles)):
    # 似然比
    ax = axes[0, i]
    for j in range(min(50, l_cum.shape[0])):
        ax.plot(l_cum[j, :], alpha=0.3, color='blue')
    ax.set_yscale('log')
    ax.set_xlabel('时间')
    ax.set_ylabel('似然比 $l_t$')
    ax.set_title(title)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)

    # 消费份额
    ax = axes[1, i]
    for j in range(min(50, c1.shape[0])):
        ax.plot(c1[j, :], alpha=0.3, color='green')
    ax.set_xlabel('时间')
    ax.set_ylabel("个体1的消费份额")
    ax.set_ylim([0, 1])
    ax.axhline(y=λ, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

在左侧面板中，自然选择$f$。个体1的消费很快达到$1$。

在中间面板中，自然选择$g$。个体1的消费比率趋向于$0$，但速度不如第一种情况快。

在右侧面板中，自然每期抛硬币。我们看到与左侧面板中的过程非常相似的模式。

顶部面板的图形让我们想起[本节](KL_link)中的讨论。

我们邀请读者重新访问[该节](llr_h)并尝试推断$D_{KL}(f\|g)$、$D_{KL}(g\|f)$、$D_{KL}(h\|f)$和$D_{KL}(h\|g)$之间的关系。

让我们计算KL散度的值

```{code-cell} ipython3
shares = [np.mean(c1_f[:, -1]), np.mean(c1_g[:, -1]), np.mean(c1_h[:, -1])]
Kf_g, Kg_f = compute_KL(f, g), compute_KL(g, f)
Kf_h, Kg_h = compute_KL_h(h, f, g)

print(f"Final shares: f={shares[0]:.3f}, g={shares[1]:.3f}, mix={shares[2]:.3f}")
print(f"KL divergences: \nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")
```

我们发现 $KL(f,g) > KL(g,f)$ 且 $KL(h,g) > KL(h,f)$。

第一个不等式告诉我们，当自然选择 $f$ 而信念为 $g$ 时的平均"惊讶度"大于当自然选择 $g$ 而信念为 $f$ 时的"惊讶度"。

这解释了我们在上面注意到的前两个面板之间的差异。

第二个不等式告诉我们，个体1的信念分布 $f$ 比个体2的信念 $g$ 更接近自然的选择。

+++

为了使这个想法更具体，让我们比较两种情况：

- 个体1的信念分布 $f$ 接近个体2的信念分布 $g$；
- 个体1的信念分布 $f$ 远离个体2的信念分布 $g$。

我们使用下面可视化的两个分布

```{code-cell} ipython3
def plot_distribution_overlap(ax, x_range, f_vals, g_vals, 
                            f_label='f', g_label='g', 
                            f_color='blue', g_color='red'):
    """Plot two distributions with their overlap region."""
    ax.plot(x_range, f_vals, color=f_color, linewidth=2, label=f_label)
    ax.plot(x_range, g_vals, color=g_color, linewidth=2, label=g_label)
    
    overlap = np.minimum(f_vals, g_vals)
    ax.fill_between(x_range, 0, overlap, alpha=0.3, color='purple', label='Overlap')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()
    
# Define close and far belief distributions
f_close = jit(lambda x: p(x, 1, 1))
g_close = jit(lambda x: p(x, 1.1, 1.05))

f_far = jit(lambda x: p(x, 1, 1))
g_far = jit(lambda x: p(x, 3, 1.2))

# Visualize the belief distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x_range = np.linspace(0.001, 0.999, 200)

# Close beliefs
f_close_vals = [f_close(x) for x in x_range]
g_close_vals = [g_close(x) for x in x_range]
plot_distribution_overlap(ax1, x_range, f_close_vals, g_close_vals,
                         f_label='f (Beta(1, 1))', g_label='g (Beta(1.1, 1.05))')
ax1.set_title(f'Close Beliefs')

# Far beliefs
f_far_vals = [f_far(x) for x in x_range]
g_far_vals = [g_far(x) for x in x_range]
plot_distribution_overlap(ax2, x_range, f_far_vals, g_far_vals,
                         f_label='f (Beta(1, 1))', g_label='g (Beta(3, 1.2))')
ax2.set_title(f'Far Beliefs')

plt.tight_layout()
plt.show()
```

让我们绘制与上面相同的代理1的消费比例图。

我们用中位数和百分位数替代模拟路径，使图形更清晰。

观察下面的图形，我们能推断出$KL(f,g)$和$KL(g,f)$之间的关系吗？

从右侧面板，我们能推断出$KL(h,g)$和$KL(h,f)$之间的关系吗？

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
nature_params = {'close': [(1, 1), (1.1, 1.05), (2, 1.5)],
                 'far':   [(1, 1), (3, 1.2),   (2, 1.5)]}
nature_labels = ["Nature = f", "Nature = g", "Nature = h"]
colors = {'close': 'blue', 'far': 'red'}

threshold = 1e-5  # "接近零"的截断值

for row, (f_belief, g_belief, label) in enumerate([
                        (f_close, g_close, 'close'),
                        (f_far, g_far, 'far')]):
    
    for col, nature_label in enumerate(nature_labels):
        params = nature_params[label][col]
        s_seq = np.random.beta(params[0], params[1], (1000, 200))
        _, c1 = simulate_blume_easley(s_seq, f_belief, g_belief, λ)
        
        median_c1 = np.median(c1, axis=0)
        p10, p90 = np.percentile(c1, [10, 90], axis=0)
        
        ax = axes[row, col]
        color = colors[label]
        ax.plot(median_c1, color=color, linewidth=2, label='中位数')
        ax.fill_between(range(len(median_c1)), p10, p90, alpha=0.3, color=color, label='10-90%')
        ax.set_xlabel('时间')
        ax.set_ylabel("代理1的份额")
        ax.set_ylim([0, 1])
        ax.set_title(nature_label)
        ax.axhline(y=λ, color='gray', linestyle='--', alpha=0.5)
        below = np.where(median_c1 < threshold)[0]
        above = np.where(median_c1 > 1-threshold)[0]
        if below.size > 0: first_zero = (below[0], True)
        elif above.size > 0: first_zero = (above[0], False)
        else: first_zero = None
        if first_zero is not None:
            ax.axvline(x=first_zero[0], color='black', linestyle='--',
                       alpha=0.7, 
                       label=fr'中位数 $\leq$ {threshold}' if first_zero[1]
                       else fr'中位数 $\geq$ 1-{threshold}')
        ax.legend()

plt.tight_layout()
plt.show()
```

让我们按照我们的猜测来计算这四个值

```{code-cell} ipython3
# 近距离情况
Kf_g, Kg_f = compute_KL(f_close, g_close), compute_KL(g_close, f_close)
Kf_h, Kg_h = compute_KL_h(h, f_close, g_close)

print(f"KL散度（近距离）：\nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")

# 远距离情况
Kf_g, Kg_f = compute_KL(f_far, g_far), compute_KL(g_far, f_far)
Kf_h, Kg_h = compute_KL_h(h, f_far, g_far)

print(f"KL散度（远距离）：\nKL(f,g)={Kf_g:.3f}, KL(g,f)={Kg_f:.3f}")
print(f"KL(h,f)={Kf_h:.3f}, KL(h,g)={Kg_h:.3f}")
```

我们发现在第一种情况下，$KL(f,g) \approx KL(g,f)$ 且两者都相对较小，所以尽管个体1或个体2最终会消耗所有资源，但在顶部前两个面板中显示的收敛是相当缓慢的。

在底部的前两个面板中，我们看到收敛发生得更快（如黑色虚线所示），这是因为差异度 $KL(f, g)$ 和 $KL(g, f)$ 更大。

由于 $KL(f,g) > KL(g,f)$，我们看到当自然选择 $f$ 时，底部第一个面板的收敛比自然选择 $g$ 时的第二个面板更快。

这与{eq}`eq:kl_likelihood_link`很好地联系在一起。



## 相关讲座

具有同质信念的完全市场模型，这种模型在宏观经济学和金融学中经常使用，在这个quantecon讲座{doc}`ge_arrow`中有研究。

{cite}`blume2018case`讨论了反对完全市场的家长式论点。他们的分析假设社会规划者应该忽视个人偏好，即应该忽视其偏好中的主观信念成分。

似然过程在贝叶斯学习中扮演重要角色，这在{doc}`likelihood_bayes`中有描述，并在{doc}`odu`中有应用。

似然比过程在{doc}`advanced:additive_functionals`中再次出现。



## 练习

```{exercise}
:label: lr_ex3

从{eq}`eq:priceequation1`开始，证明竞争均衡价格可以表示为

$$

p_t(s^t) = \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr]
$$

```

```{solution-start} lr_ex3
:class: dropdown
```

从以下式子开始

$$
p_t(s^t) = \frac{\delta^t \pi_t^i(s^t)}{\mu_i c_t^i(s^t)}, \qquad i=1,2.
$$

由于两个表达式等于相同的价格,我们可以令它们相等

$$
\frac{\pi_t^1(s^t)}{\mu_1 c_t^1(s^t)} = \frac{\pi_t^2(s^t)}{\mu_2 c_t^2(s^t)}
$$

重新整理得到

$$
\frac{c_t^1(s^t)}{c_t^2(s^t)} = \frac{\mu_2}{\mu_1} l_t(s^t)
$$

其中 $l_t(s^t) \equiv \pi_t^1(s^t)/\pi_t^2(s^t)$ 是似然比过程。

使用 $c_t^2(s^t) = 1 - c_t^1(s^t)$:

$$
\frac{c_t^1(s^t)}{1 - c_t^1(s^t)} = \frac{\mu_2}{\mu_1} l_t(s^t)
$$

求解 $c_t^1(s^t)$

$$
c_t^1(s^t) = \frac{\mu_2 l_t(s^t)}{\mu_1 + \mu_2 l_t(s^t)}
$$

规划者的解给出

$$
c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1 - \lambda + \lambda l_t(s^t)}
$$

为了使个体1在竞争均衡中的选择与规划者为个体1做出的选择相匹配,必须满足以下等式

$$
\frac{\mu_2}{\mu_1} = \frac{\lambda}{1 - \lambda}
$$

因此我们有

$$
\mu_1 = 1 - \lambda, \qquad \mu_2 = \lambda
$$

当 $\mu_1 = 1-\lambda$ 且 $c_t^1(s^t) = \frac{\lambda l_t(s^t)}{1-\lambda+\lambda l_t(s^t)}$ 时,
我们有

$$
\begin{aligned}
p_t(s^t) &= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda) c_t^1(s^t)} \\
&= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda)} \cdot \frac{1 - \lambda + \lambda l_t(s^t)}{\lambda l_t(s^t)} \\

&= \frac{\delta^t \pi_t^1(s^t)}{(1-\lambda)\lambda l_t(s^t)} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr].
\end{aligned}
$$

由于 $\pi_t^1(s^t) = l_t(s^t) \pi_t^2(s^t)$，我们有

$$
\begin{aligned}
p_t(s^t) &= \frac{\delta^t l_t(s^t) \pi_t^2(s^t)}{(1-\lambda)\lambda l_t(s^t)} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr] \\
&= \frac{\delta^t \pi_t^2(s^t)}{(1-\lambda)\lambda} \bigl[1 - \lambda + \lambda l_t(s^t)\bigr] \\
&= \frac{\delta^t}{\lambda(1-\lambda)} \pi_t^2(s^t) \bigl[1 - \lambda + \lambda l_t(s^t)\bigr].
\end{aligned}
$$

```{solution-end}
```

```{exercise}
:label: lr_ex4

在这个练习中，我们将研究两个主体，每个主体在数据到达时都会更新其后验概率。

* 每个主体都按照{doc}`likelihood_bayes`中研究的方式应用贝叶斯法则。

以下是两个待考虑的模型

$$
f(s^t) = f(s_1) f(s_2) \cdots f(s_t) 
$$

和

$$
g(s^t) = g(s_1) g(s_2) \cdots g(s_t)  
$$

以及相关的似然比过程

$$
L(s^t) = \frac{f(s^t)}{g(s^t)} .
$$

令 $\pi_0 \in (0,1)$ 为先验概率，且

$$
\pi_t = \frac{ \pi_0 L(s^t)}{ \pi_0 L(s^t) + (1-\pi_0) } .
$$

我们的两个主体各自使用混合模型的自己的版本

$$
m(s^t) = \pi_t f(s^t) + (1- \pi_t) g(s^t)
$$ (eq:be_mix_model)

我们将为每种类型的消费者配备模型{eq}`eq:be_mix_model`。

* 两个主体共享相同的 $f$ 和 $g$，但是
* 他们有不同的初始先验概率，即 $\pi_0^1$ 和 $\pi_0^2$

因此，消费者 $i$ 的概率模型是

$$

m^i(s^t) = \pi^i_t f(s^t) + (1- \pi^i_t) g(s^t)
$$ (eq:prob_model)

现在我们将概率模型{eq}`eq:prob_model`（其中i=1,2）交给社会规划者。

我们想要推导分配$c^i(s^t), i = 1,2$，并观察当以下情况发生时会发生什么：

  * 自然的模型是$f$
  * 自然的模型是$g$

我们预期消费者最终会学习到"真相"，但其中一个人会学习得更快。

为了探索这些问题，请设定$f \sim \text{Beta}(1.5, 1)$和$g \sim \text{Beta}(1, 1.5)$。

请编写Python代码回答以下问题：

 * 消费份额如何演变？
 * 当自然遵循$f$时，哪个代理学习得更快？
 * 当自然遵循$g$时，哪个代理学习得更快？
 * 初始先验$\pi_0^1$和$\pi_0^2$的差异如何影响收敛速度？

```{solution-start} lr_ex4
:class: dropdown
```

首先，让我们编写辅助函数来计算模型组件，包括每个代理的主观信念函数。

```{code-cell} ipython3
def bayesian_update(π_0, L_t):
    """
    给定似然比的贝叶斯信念概率更新。
    """
    return (π_0 * L_t) / (π_0 * L_t + (1 - π_0))

def mixture_density_belief(s_seq, f_func, g_func, π_seq):
    """
    计算代理i的混合密度信念m^i(s^t)。
    """
    f_vals = f_func(s_seq)
    g_vals = g_func(s_seq)
    return π_seq * f_vals + (1 - π_seq) * g_vals
```

现在让我们编写代码来模拟包含两个个体的Blume-Easley模型。

```{code-cell} ipython3
def simulate_learning_blume_easley(sequences, f_belief, g_belief, 
                                        π_0_1, π_0_2, λ=0.5):
    """
    模拟包含学习个体的Blume-Easley模型。
    """
    N, T = sequences.shape
    
    # 初始化存储结果的数组
    π_1_seq = np.full((N, T), np.nan)
    π_2_seq = np.full((N, T), np.nan)
    c1_share = np.full((N, T), np.nan)
    l_agents_seq = np.full((N, T), np.nan)

    π_1_seq[:, 0] = π_0_1
    π_2_seq[:, 0] = π_0_2
    
    for n in range(N):
        # 初始化信念的累积似然比
        L_cumul = 1.0
        
        # 初始化个体密度之间的似然比
        l_agents_cumul = 1.0
        
        for t in range(1, T):
            s_t = sequences[n, t]
            
            # 计算此观测的似然比
            l_t = f_belief(s_t) / g_belief(s_t)
            
            # 更新累积似然比
            L_cumul *= l_t
            
            # 贝叶斯更新信念
            π_1_t = bayesian_update(π_0_1, L_cumul)
            π_2_t = bayesian_update(π_0_2, L_cumul)
            
            # 存储信念
            π_1_seq[n, t] = π_1_t
            π_2_seq[n, t] = π_2_t
            
            # 计算每个个体的混合密度
            m1_t = π_1_t * f_belief(s_t) + (1 - π_1_t) * g_belief(s_t)
            m2_t = π_2_t * f_belief(s_t) + (1 - π_2_t) * g_belief(s_t)
            
            # 更新个体之间的累积似然比
            l_agents_cumul *= (m1_t / m2_t)
            l_agents_seq[n, t] = l_agents_cumul
            
            # c_t^1(s^t) = λ * l_t(s^t) / (1 - λ + λ * l_t(s^t))
            # 其中l_t(s^t)是个体之间的累积似然比
            c1_share[n, t] = λ * l_agents_cumul / (1 - λ + λ * l_agents_cumul)
    
    return {
        'π_1': π_1_seq,
        'π_2': π_2_seq, 
        'c1_share': c1_share,
        'l_agents': l_agents_seq
    }
```

让我们运行不同场景的模拟。

我们使用 $\lambda = 0.5$，$T=40$，以及 $N=1000$。

```{code-cell} ipython3
λ = 0.5
T = 40
N = 1000

F_a, F_b = 1.5, 1
G_a, G_b = 1, 1.5

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
```

我们将从不同的初始先验概率 $\pi^i_0 \in (0, 1)$ 开始，并扩大它们之间的差距。

```{code-cell} ipython3
# 不同的初始先验概率
π_0_scenarios = [
    (0.3, 0.7),
    (0.7, 0.3),
    (0.1, 0.9),
]
```

现在我们可以为不同场景运行模拟

```{code-cell} ipython3
# 自然遵循 f
s_seq_f = np.random.beta(F_a, F_b, (N, T))

# 自然遵循 g
s_seq_g = np.random.beta(G_a, G_b, (N, T)) 

results_f = {}
results_g = {}

for i, (π_0_1, π_0_2) in enumerate(π_0_scenarios):
    # 当自然遵循 f 时
    results_f[i] = simulate_learning_blume_easley(
            s_seq_f, f, g, π_0_1, π_0_2, λ)
    # 当自然遵循 g 时  
    results_g[i] = simulate_learning_blume_easley(
            s_seq_g, f, g, π_0_1, π_0_2, λ)
```

让我们可视化结果

```{code-cell} ipython3
def plot_learning_results(results, π_0_scenarios, nature_type, truth_value):
    """
    绘制学习智能体的信念和消费份额。
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    
    scenario_labels = [
        rf'$\pi_0^1 = {π_0_1}, \pi_0^2 = {π_0_2}$'
        for π_0_1, π_0_2 in π_0_scenarios
    ]
    
    for row, (scenario_idx, scenario_label) in enumerate(
                zip(range(3), scenario_labels)):

        res = results[scenario_idx]
        
        # 绘制信念
        ax = axes[row, 0]
        π_1_med = np.median(res['π_1'], axis=0)
        π_2_med = np.median(res['π_2'], axis=0) 
        ax.plot(π_1_med, 'C0', label=r'智能体1', linewidth=2)
        ax.plot(π_2_med, 'C1', label=r'智能体2', linewidth=2)
        ax.axhline(y=truth_value, color='gray', linestyle='--', 
                   alpha=0.5, label=f'真值({nature_type})')
        ax.set_title(f'当自然状态 = {nature_type}时的信念\n{scenario_label}')
        ax.set_ylabel(r'中位数 $\pi_i^t$')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        
        # 绘制消费份额
        ax = axes[row, 1]
        c1_med = np.median(res['c1_share'], axis=0)
        ax.plot(c1_med, 'g-', linewidth=2, label='中位数')
        ax.axhline(y=0.5, color='gray', linestyle='--', 
                   alpha=0.5)
        ax.set_title(f'智能体1的消费份额(自然状态 = {nature_type})')
        ax.set_ylabel('消费份额')
        ax.set_ylim([0, 1])
        ax.legend()
        
        # 添加x轴标签
        for col in range(2):
            axes[row, col].set_xlabel('$t$')
    
    plt.tight_layout()
    return fig, axes
```

现在我们将绘制当自然遵循 f 时的结果：

```{code-cell} ipython3
fig_f, axes_f = plot_learning_results(
                results_f, π_0_scenarios, 'f', 1.0)
plt.show()
```

我们可以看到，具有更准确信念的个体获得更高的消费份额。

此外，初始信念差异越大，消费比率收敛所需的时间就越长。

"不太准确"的个体学习时间越长，其最终消费份额就越低。

现在让我们绘制当自然遵循g时的结果：

```{code-cell} ipython3
fig_g, axes_g = plot_learning_results(results_g, π_0_scenarios, 'g', 0.0)
plt.show()
```

我们观察到对称的结果。

```{solution-end}
```

```{exercise}
:label: lr_ex5

在前面的练习中,我们故意将两个 beta 分布设置得相对接近。

这使得区分这些分布变得具有挑战性。

现在让我们研究当这些分布相距更远时的结果。

让我们设置 $f \sim \text{Beta}(2, 5)$ 和 $g \sim \text{Beta}(5, 2)$。

请使用你已经编写的 Python 代码来研究结果。
```

```{solution-start} lr_ex5
:class: dropdown
```

这是一个解决方案

```{code-cell} ipython3
λ = 0.5
T = 40
N = 1000

F_a, F_b = 2, 5
G_a, G_b = 5, 2

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

π_0_scenarios = [
    (0.3, 0.7),
    (0.7, 0.3),
    (0.1, 0.9),
]

s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T)) 

results_f = {}
results_g = {}

for i, (π_0_1, π_0_2) in enumerate(π_0_scenarios):
    # 当自然遵循 f 时
    results_f[i] = simulate_learning_blume_easley(
            s_seq_f, f, g, π_0_1, π_0_2, λ)
    # 当自然遵循 g 时  
    results_g[i] = simulate_learning_blume_easley(
            s_seq_g, f, g, π_0_1, π_0_2, λ)
```

现在让我们将结果可视化

```{code-cell} ipython3
fig_f, axes_f = plot_learning_results(results_f, π_0_scenarios, 'f', 1.0)
plt.show()
```

```{code-cell} ipython3
fig_g, axes_g = plot_learning_results(results_g, π_0_scenarios, 'g', 0.0)
plt.show()
```

显然，由于两个分布之间的距离更远，更容易区分它们。

因此学习发生得更快。

消费份额也是如此。

```{solution-end}
```

```{exercise}
:label: lr_ex6

两个代理对三个可能的模型有不同的信念。

假设对于 $x \in X$，$f(x) \geq 0$，$g(x) \geq 0$，且 $h(x) \geq 0$，并且：
- $\int_X f(x) dx = 1$
- $\int_X g(x) dx = 1$
- $\int_X h(x) dx = 1$

我们将考虑两个代理：
* 代理1：$\pi^g_0 = 1 - \pi^f_0$，$\pi^f_0 \in (0,1)$，$\pi^h_0 = 0$
（仅对模型 $f$ 和 $g$ 赋予正概率）
* 代理2：$\pi^g_0 = \pi^f_0 = 1/3$，$\pi^h_0 = 1/3$
（对所有三个模型赋予相等权重）

令 $f$ 和 $g$ 为两个贝塔分布，其中 $f \sim \text{Beta}(3, 2)$ 且
$g \sim \text{Beta}(2, 3)$，并且
设 $h = \pi^f_0 f + (1-\pi^f_0) g$，其中 $\pi^f_0 = 0.5$。

贝叶斯法则告诉我们，模型 $f$ 和 $g$ 的后验概率按如下方式演变：

$$
\pi^f(s^t) := \frac{\pi^f_0 f(s^t)}{\pi^f_0 f(s^t) 
+ \pi^g_0 g(s^t) + (1 - \pi^f_0 - \pi^g_0) h(s^t)}
$$

和

$$
\pi^g(s^t) := \frac{\pi^g_0 g(s^t)}{\pi^f_0 f(s^t) 
+ \pi^g_0 g(s^t) + (1 - \pi^f_0 - \pi^g_0) h(s^t)}
$$

请模拟并可视化以下情况下的后验概率和消费分配的演变：

* 自然永久从 $f$ 中抽取
* 自然永久从 $g$ 中抽取
```

```{solution-start} lr_ex6
:class: dropdown
```

让我们实现这个具有两个代理的三模型案例。

让我们定义相距较远的函数$f$和$g$，并让$h$作为$f$和$g$的混合。

```{code-cell} ipython3
F_a, F_b = 3, 2
G_a, G_b = 2, 3
λ = 0.5 
π_f_0 = 0.5

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
h = jit(lambda x: π_f_0 * f(x) + (1 - π_f_0) * g(x))
```

现在我们可以为模型定义信念更新

```{code-cell} ipython3
@jit(parallel=True)
def compute_posterior_three_models(
    s_seq, f_func, g_func, h_func, π_f_0, π_g_0):
    """
    计算三个模型的后验概率。
    """
    N, T = s_seq.shape
    π_h_0 = 1 - π_f_0 - π_g_0
    
    π_f = np.zeros((N, T))
    π_g = np.zeros((N, T))
    π_h = np.zeros((N, T))
    
    for n in prange(N):
        # 用先验概率初始化
        π_f[n, 0] = π_f_0
        π_g[n, 0] = π_g_0
        π_h[n, 0] = π_h_0
        
        # 计算累积似然
        f_cumul = 1.0
        g_cumul = 1.0
        h_cumul = 1.0
        
        for t in range(1, T):
            s_t = s_seq[n, t]
            
            # 更新累积似然
            f_cumul *= f_func(s_t)
            g_cumul *= g_func(s_t)
            h_cumul *= h_func(s_t)
            
            # 使用贝叶斯法则计算后验概率
            denominator = π_f_0 * f_cumul + π_g_0 * g_cumul + π_h_0 * h_cumul
            
            π_f[n, t] = π_f_0 * f_cumul / denominator
            π_g[n, t] = π_g_0 * g_cumul / denominator
            π_h[n, t] = π_h_0 * h_cumul / denominator
    
    return π_f, π_g, π_h
```

让我们也写一些类似之前练习的模拟代码

```{code-cell} ipython3
@jit
def bayesian_update_three_models(π_f_0, π_g_0, L_f, L_g, L_h):
    """三个模型的贝叶斯更新。"""
    π_h_0 = 1 - π_f_0 - π_g_0
    denom = π_f_0 * L_f + π_g_0 * L_g + π_h_0 * L_h
    return π_f_0 * L_f / denom, π_g_0 * L_g / denom, π_h_0 * L_h / denom

@jit
def compute_mixture_density(π_f, π_g, π_h, f_val, g_val, h_val):
    """计算代理的混合密度。"""
    return π_f * f_val + π_g * g_val + π_h * h_val

@jit(parallel=True)
def simulate_three_model_allocation(sequences, f_func, g_func, h_func,
                                     π_f_0_1, π_g_0_1, π_f_0_2, π_g_0_2, λ=0.5):
    """
    模拟具有学习代理和三个模型的Blume-Easley模型。
    """
    N, T = sequences.shape
    
    # 初始化数组以存储结果
    beliefs_1 = {k: np.full((N, T), np.nan) for k in ['π_f', 'π_g', 'π_h']}
    beliefs_2 = {k: np.full((N, T), np.nan) for k in ['π_f', 'π_g', 'π_h']}
    c1_share = np.full((N, T), np.nan)
    l_agents_seq = np.full((N, T), np.nan)
    
    # 设置初始信念
    beliefs_1['π_f'][:, 0] = π_f_0_1
    beliefs_1['π_g'][:, 0] = π_g_0_1
    beliefs_1['π_h'][:, 0] = 1 - π_f_0_1 - π_g_0_1
    beliefs_2['π_f'][:, 0] = π_f_0_2
    beliefs_2['π_g'][:, 0] = π_g_0_2
    beliefs_2['π_h'][:, 0] = 1 - π_f_0_2 - π_g_0_2
    
    for n in range(N):
        # 初始化累积似然
        L_cumul = {'f': 1.0, 'g': 1.0, 'h': 1.0}
        l_agents_cumul = 1.0
        
        # 计算t=0时的初始消费份额
        l_agents_seq[n, 0] = 1.0
        c1_share[n, 0] = λ * 1.0 / (1 - λ + λ * 1.0)  # 等于λ
        
        for t in range(1, T):
            s_t = sequences[n, t]
            
            # 计算当前观察的密度
            densities = {
                'f': f_func(s_t),
                'g': g_func(s_t),
                'h': h_func(s_t)
            }
            
            # 更新累积似然
            for model in L_cumul:
                L_cumul[model] *= densities[model]
            
            # 两个代理的贝叶斯更新
            π_f_1, π_g_1, π_h_1 = bayesian_update_three_models(
                π_f_0_1, π_g_0_1, L_cumul['f'], L_cumul['g'], L_cumul['h'])
            π_f_2, π_g_2, π_h_2 = bayesian_update_three_models(
                π_f_0_2, π_g_0_2, L_cumul['f'], L_cumul['g'], L_cumul['h'])
            
            # 存储信念
            beliefs_1['π_f'][n, t] = π_f_1
            beliefs_1['π_g'][n, t] = π_g_1
            beliefs_1['π_h'][n, t] = π_h_1
            beliefs_2['π_f'][n, t] = π_f_2
            beliefs_2['π_g'][n, t] = π_g_2
            beliefs_2['π_h'][n, t] = π_h_2
            
            # 计算混合密度
            m1_t = compute_mixture_density(
                π_f_1, π_g_1, π_h_1, densities['f'], 
                densities['g'], densities['h'])
            m2_t = compute_mixture_density(
                π_f_2, π_g_2, π_h_2, densities['f'], 
                densities['g'], densities['h'])
            
            # 更新代理之间的累积似然比
            l_agents_cumul *= (m1_t / m2_t)
            l_agents_seq[n, t] = l_agents_cumul
            
            # 代理1的消费份额
            c1_share[n, t] = λ * l_agents_cumul / (1 - λ + λ * l_agents_cumul)
    
    return {
        'π_f_1': beliefs_1['π_f'],
        'π_g_1': beliefs_1['π_g'],
        'π_h_1': beliefs_1['π_h'],
        'π_f_2': beliefs_2['π_f'],
        'π_g_2': beliefs_2['π_g'],
        'π_h_2': beliefs_2['π_h'],
        'c1_share': c1_share,
        'l_agents': l_agents_seq
    }
```

以下代码单元定义了一个绘图函数，用于显示信念和消费比例的演变

```{code-cell} ipython3
:tags: [hide-input]

def plot_belief_evolution(results, nature='f', figsize=(15, 5)):
    """
    创建显示三个模型(f, g, h)信念演变的图表。
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    model_names = ['f', 'g', 'h']
    belief_keys = [('π_f_1', 'π_f_2'), 
                   ('π_g_1', 'π_g_2'), 
                   ('π_h_1', 'π_h_2')]
    
    for j, (model_name, (key1, key2)) in enumerate(
                        zip(model_names, belief_keys)):
        ax = axes[j]
        
        # 绘制个体信念
        ax.plot(np.median(results[key1], axis=0), 'C0-', 
                            linewidth=2, label='个体1')
        ax.plot(np.median(results[key2], axis=0), 'C1-', 
                            linewidth=2, label='个体2')
        
        # 真实值指示器
        if model_name == nature:
            ax.axhline(y=1.0, color='grey', linestyle='-.', 
                      alpha=0.7, label='真实值')
        else:
            ax.axhline(y=0.0, color='grey', linestyle='-.', 
                      alpha=0.7, label='真实值')
        
        ax.set_title(f'π({model_name}) 当自然状态 = {nature}')
        ax.set_xlabel('$t$')
        ax.set_ylabel(f'中位数 π({model_name})')
        ax.set_ylim([-0.01, 1.01])
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, axes


def plot_consumption_dynamics(results_f, results_g, λ=0.5, figsize=(14, 5)):
    """
    创建显示个体1消费份额动态的图表。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    results_list = [results_f, results_g]
    nature_labels = ['f', 'g']
    colors = ['blue', 'green']
    
    for i, (results, nature_label, color) in enumerate(
                    zip(results_list, nature_labels, colors)):
        ax = axes[i]
        c1 = results['c1_share']
        c1_med = np.median(c1, axis=0)
        
        # 绘制中位数和百分位数
        ax.plot(c1_med, color=color, linewidth=2, label="中位数")
        
        # 添加百分位区间带
        c1_25 = np.percentile(c1, 25, axis=0)
        c1_75 = np.percentile(c1, 75, axis=0)
        ax.fill_between(range(len(c1_med)), c1_25, c1_75, 
                        color=color, alpha=0.2, label="25-75百分位")
        
        ax.axhline(y=0.5, color='grey', linestyle='--', 
                        alpha=0.5, label='等份')
        
        ax.set_title(f'个体1消费份额(自然状态 = {nature_label})')
        ax.set_xlabel('$t$')
        ax.set_ylabel("消费份额")
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, axes
```

现在让我们运行模拟。

在下面的模拟中，个体1只对$f$和$g$分配正概率，而个体2对所有三个模型赋予相等的权重。

```{code-cell} ipython3
T = 100
N = 1000

# 为自然状态f和g生成序列
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

# 运行模拟
results_f = simulate_three_model_allocation(s_seq_f, 
                        f, g, h, π_f_0, 1-π_f_0, 
                        1/3, 1/3, λ)
results_g = simulate_three_model_allocation(s_seq_g, 
                        f, g, h, π_f_0, 1-π_f_0, 
                        1/3, 1/3, λ)
```

下面的图表分别展示了每个模型（f、g、h）的信念演变。

首先我们展示当自然选择$f$时的图表

```{code-cell} ipython3
plot_belief_evolution(results_f, nature='f', figsize=(15, 5))
plt.show()
```

智能体1的后验信念用蓝色表示，智能体2的后验信念用橙色表示。

显然，当自然选择$f$时，智能体1比智能体2学习得更快，这是因为智能体2（与智能体1不同）对模型$h$赋予了正的先验概率：

- 在最左边的面板中，两个智能体对$\pi(f)$的信念都逐渐收敛到1（真实值）
- 智能体2对模型$h$的信念（最右边的面板）在初期上升后逐渐收敛到0

现在让我们绘制当自然选择$g$时的信念演化：

```{code-cell} ipython3
plot_belief_evolution(results_g, nature='g', figsize=(15, 5))
plt.show()
```

再次可以看到，智能体1比智能体2学习得更快。

在查看下一张图之前，请猜测消费份额是如何变化的。

请记住，智能体1比智能体2更快地达到正确的模型。

```{code-cell} ipython3
plot_consumption_dynamics(results_f, results_g, λ=0.5, figsize=(14, 5))
plt.show()
```

正如我们所预期的，个体1比个体2有更高的消费份额。

在这个练习中，"真实情况"是两个个体模型中可能的结果之一。

个体2的模型"更一般化"，因为它允许一种个体1的模型中不包含的可能性——即自然从$h$中抽取。

个体1学习得更快是因为他使用了一个更简单的模型。

```{solution-end}
```

```{exercise}
:label: lr_ex7

现在考虑两个对三个模型有极端先验的个体。

考虑与前一个练习相同的设置，但现在：
* 个体1：$\pi^g_0 = \pi^f_0 = \frac{\epsilon}{2} > 0$，其中$\epsilon$接近$0$（例如，$\epsilon = 0.01$）
* 个体2：$\pi^g_0 = \pi^f_0 = 0$（对模型$h$的刚性信念）

选择$h$使其在KL散度度量下接近但不等于$f$或$g$。
例如，设置$h \sim \text{Beta}(1.2, 1.1)$且$f \sim \text{Beta}(1, 1)$。

请模拟并可视化以下情况下的后验概率和消费分配的演变：

* 自然永久从$f$中抽取
* 自然永久从$g$中抽取
```

```{solution-start} lr_ex7
:class: dropdown
```

为了探索这个练习，我们将$T$增加到1000。

让我们指定$f$、$g$和$h$，并验证$h$和$f$比$h$和$g$更接近

```{code-cell} ipython3
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2
H_a, H_b = 1.2, 1.1

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
h = jit(lambda x: p(x, H_a, H_b))

Kh_f = compute_KL(h, f)
Kh_g = compute_KL(h, g)
Kf_h = compute_KL(f, h)
Kg_h = compute_KL(g, h)

print(f"KL散度：")
print(f"KL(h,f) = {Kh_f:.4f}, KL(h,g) = {Kh_g:.4f}")
print(f"KL(f,h) = {Kf_h:.4f}, KL(g,h) = {Kg_h:.4f}")
```

现在我们可以为两个智能体设置信念模型

```{code-cell} ipython3
ε = 0.01
λ = 0.5

# 智能体1: π_f = ε/2, π_g = ε/2, π_h = 1-ε 
# (对h几乎完全坚信)
π_f_1 = ε/2
π_g_1 = ε/2

# 智能体2: π_f = 0, π_g = 0, π_h = 1 
# (对h完全坚信)
π_f_2 = 1e-10
π_g_2 = 1e-10
```

现在我们可以运行模拟

```{code-cell} ipython3
T = 1000
N = 1000

# 为不同的自然情景生成序列
s_seq_f = np.random.beta(F_a, F_b, (N, T))
s_seq_g = np.random.beta(G_a, G_b, (N, T))

# 为两种情景运行模拟
results_f = simulate_three_model_allocation(
                                s_seq_f, 
                                f, g, h, 
                                π_f_1, π_g_1, π_f_2, π_g_2, λ)
results_g = simulate_three_model_allocation(
                                s_seq_g, 
                                f, g, h, 
                                π_f_1, π_g_1, π_f_2, π_g_2, λ)
```

让我们绘制当自然选择$f$时的信念演变

```{code-cell} ipython3
plot_belief_evolution(results_f, nature='f', figsize=(15, 5))
plt.show()
```

观察最左侧面板中$\pi(f)$显示的个体1如何缓慢地学习真相。

还要注意个体2没有更新。

这是因为我们已经指定$f$很难与$h$区分，这是通过$KL(f, h)$来衡量的。

对$h$的刚性阻止了个体2在观察到非常相似的模型$f$时更新其信念。

现在让我们绘制当自然选择$g$时的信念演变

```{code-cell} ipython3
plot_belief_evolution(results_g, nature='g', figsize=(15, 5))
plt.show()
```

当自然从$g$中抽取时，它与$h$的距离更远，这是通过KL散度来衡量的。

这有助于两个个体更快地学习真相。

```{code-cell} ipython3
plot_consumption_dynamics(results_f, results_g, 
                               λ=0.5, figsize=(14, 5))
plt.show()
```

在消费动态图中，注意到无论自然是永久从$f$中抽取还是永久从$g$中抽取，个体1的消费份额都收敛到1。

```{solution-end}
```

