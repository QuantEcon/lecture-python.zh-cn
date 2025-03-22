---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(mc)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`有限马尔可夫链 <single: Finite Markov Chains>`

```{contents} 目录
:depth: 2
```

除了Anaconda中已有的库外，本讲座还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

马尔可夫链是最有用的随机过程类别之一，因为它们：

* 简单、灵活，并且有许多优雅的理论支持
* 对于建立随机动态模型的直观认识很有价值
* 本身就是定量建模的核心

你会在经济学和金融学的许多基础模型中发现它们。

在本讲座中，我们将回顾马尔可夫链的一些理论。

我们还将介绍[QuantEcon.py](https://quantecon.org/quantecon-py/)中提供的一些用于处理马尔可夫链的高质量程序。

所需的预备知识是基础概率论和线性代数。

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import quantecon as qe
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
```

## 定义

以下概念是基础性的。

(finite_dp_stoch_mat)=
### {index}`随机矩阵 <single: Stochastic Matrices>`

```{index} single: Finite Markov Chains; Stochastic Matrices
```

**随机矩阵**（或**马尔可夫矩阵**）是一个 $n \times n$ 的方阵 $P$，满足：

1. $P$ 的每个元素都是非负的，且
1. $P$ 的每一行之和等于一

$P$ 的每一行都可以被视为在 $n$ 个可能结果上的概率质量函数。

不难证明[^pm]，如果 $P$ 是随机矩阵，那么对于所有 $k \in \mathbb N$，其 $k$ 次幂 $P^k$ 也是随机矩阵。

### {index}`马尔可夫链 <single: Markov Chains>`

```{index} single: Finite Markov Chains
```

随机矩阵和马尔可夫链之间有着密切的联系。

首先，令 $S$ 为具有 $n$ 个元素 $\{x_1, \ldots, x_n\}$ 的有限集。

集合 $S$ 被称为**状态空间**，而 $x_1, \ldots, x_n$ 被称为**状态值**。

**马尔可夫链** $\{X_t\}$ 是定义在状态空间 $S$ 上的一系列具有**马尔可夫性质**的随机变量。

这意味着，对于任意时刻 $t$ 和任意状态 $y \in S$，

```{math}
:label: fin_markov_mp

\mathbb P \{ X_{t+1} = y  \,|\, X_t \}
= \mathbb P \{ X_{t+1}  = y \,|\, X_t, X_{t-1}, \ldots \}
```

换句话说，知道当前状态就足以确定未来状态的概率。

特别地，马尔可夫链的动态完全由以下值集确定

```{math}
:label: mpp

P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}
\qquad (x, y \in S)
```

根据构造，

* $P(x, y)$ 是在一个时间单位（一步）内从 $x$ 到 $y$ 的概率
* $P(x, \cdot)$ 是在给定 $X_t = x$ 条件下 $X_{t+1}$ 的条件分布

我们可以将 $P$ 视为一个随机矩阵，其中

$$
P_{ij} = P(x_i, x_j)
\qquad 1 \leq i, j \leq n
$$

反过来，如果我们有一个随机矩阵 $P$，我们可以生成一个马尔可夫链 $\{X_t\}$，方法如下：

* 从边际分布 $\psi$ 中抽取 $X_0$
* 对每个 $t = 0, 1, \ldots$, 从 $P(X_t,\cdot)$ 中抽取 $X_{t+1}$

根据构造，所得到的过程满足 {eq}`mpp`。

(mc_eg1)=
### 示例1

考虑一个工人，在任何给定时间 $t$，要么失业（状态0）要么就业（状态1）。

假设在一个月的时间段内，

1. 失业工人找到工作的概率为 $\alpha \in (0, 1)$。
1. 就业工人失去工作并变成失业的概率为 $\beta \in (0, 1)$。

就马尔可夫模型而言，我们有

* $S = \{ 0, 1\}$
* $P(0, 1) = \alpha$ 且 $P(1, 0) = \beta$

我们可以用矩阵形式写出转移概率：

```{math}
:label: p_unempemp

P
= \left(
\begin{array}{cc}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{array}
  \right)
```

一旦我们有了 $\alpha$ 和 $\beta$ 的值，我们就可以解决一系列问题，比如

* 失业的平均持续时间是多少？

* 从长期来看，一个工人处于失业状态的时间比例是多少？
* 在就业的条件下，未来12个月内至少失业一次的概率是多少？

我们将在下面讨论这些应用。

(mc_eg2)=
### 示例2

根据美国失业数据，Hamilton {cite}`Hamilton2005` 估算出如下随机矩阵

$$
P =
\left(
  \begin{array}{ccc}
     0.971 & 0.029 & 0 \\
     0.145 & 0.778 & 0.077 \\
     0 & 0.508 & 0.492
  \end{array}
\right)
$$

其中

* 频率为每月
* 第一个状态代表"正常增长"
* 第二个状态代表"轻度衰退"
* 第三个状态代表"严重衰退"

例如，该矩阵告诉我们，当状态为正常增长时，下个月仍然是正常增长的概率为0.97。

一般来说，主对角线上的较大值表明过程 $\{ X_t \}$ 具有持续性。

这个马尔可夫过程也可以用一个有向图表示，边上标注着转移概率

```{figure} /_static/lecture_specific/finite_markov/hamilton_graph.png

```

这里"ng"表示正常增长，"mr"表示温和衰退，等等。

## 模拟

```{index} single: Markov Chains; Simulation
```

回答关于马尔可夫链问题的一个自然方法是对其进行模拟。

(要近似事件 $E$ 发生的概率，我们可以进行多次模拟并统计 $E$ 发生的次数比例)。

[QuantEcon.py](http://quantecon.org/quantecon-py)提供了模拟马尔可夫链的良好功能。

* 高效，并捆绑了许多其他用于处理马尔可夫链的实用程序。

然而，编写我们自己的程序也是一个很好的练习 —— 让我们先来做这个，然后再回到[QuantEcon.py](http://quantecon.org/quantecon-py)中的方法。

在这些练习中，我们将状态空间设定为 $S = 0,\ldots, n-1$。

### 自行编写程序

要模拟马尔可夫链，我们需要其随机矩阵 $P$ 和一个边际概率分布 $\psi$，用于抽取 $X_0$ 的一个实现值。

马尔可夫链的构建过程如上所述。重复如下：

1. 在时间 $t=0$ 时，从 $\psi$ 中抽取 $X_0$ 的一个实现值。
1. 在之后的每个时间 $t$，从 $P(X_t, \cdot)$ 中抽取新状态 $X_{t+1}$ 的一个实现值。

为了实现这个模拟过程，我们需要一个从离散分布中生成抽样的方法。

对于这个任务，我们将使用来自 [QuantEcon](http://quantecon.org/quantecon-py) 的 `random.draw`，其使用方法如下：

```{code-cell} ipython3
ψ = (0.3, 0.7)           # {0, 1}上的概率
cdf = np.cumsum(ψ)       # 转换为累积分布
qe.random.draw(cdf, 5)   # 从ψ中生成5个独立抽样
```

我们将把代码写成一个接受以下三个参数的函数

* 一个随机矩阵 `P`
* 一个初始状态 `init`
* 一个正整数 `sample_size` 表示函数应返回的时间序列长度

```{code-cell} ipython3
def mc_sample_path(P, ψ_0=None, sample_size=1_000):

    # 设置
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)

    # 将P的每一行转换为累积分布函数
    n = len(P)
    P_dist = [np.cumsum(P[i, :]) for i in range(n)]

    # 抽取初始状态，默认为0
    if ψ_0 is not None:
        X_0 = qe.random.draw(np.cumsum(ψ_0))
    else:
        X_0 = 0

    # 模拟
    X[0] = X_0
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])

    return X
```

让我们用一个小矩阵来看看它是如何工作的

```{code-cell} ipython3
P = [[0.4, 0.6],
     [0.2, 0.8]]
```

我们稍后会看到，对于从`P`中抽取的长序列，样本中取值为0的比例大约是0.25。

而且，无论$X_0$的初始分布如何，这个结论都成立。

以下代码演示了这一点

```{code-cell} ipython3
X = mc_sample_path(P, ψ_0=[0.1, 0.9], sample_size=100_000)
np.mean(X == 0)
```

你可以尝试改变初始分布，以确认输出总是接近0.25，至少对于上面的`P`矩阵是这样。

### 使用QuantEcon的程序

如上所述，[QuantEcon.py](http://quantecon.org/quantecon-py)有处理马尔可夫链的程序，包括模拟。

这里用与前面例子相同的P进行说明

```{code-cell} ipython3
from quantecon import MarkovChain

mc = qe.MarkovChain(P)
X = mc.simulate(ts_length=1_000_000)
np.mean(X == 0)
```

[QuantEcon.py](http://quantecon.org/quantecon-py)程序使用了[JIT编译](https://python-programming.quantecon.org/numba.html#numba-link)，运行速度更快。

```{code-cell} ipython
%time mc_sample_path(P, sample_size=1_000_000) # 我们自制代码版本
```

```{code-cell} ipython
%time mc.simulate(ts_length=1_000_000) # qe代码版本
```

#### 添加状态值和初始条件

如果需要，我们可以为`MarkovChain`提供状态值的规范。

这些状态值可以是整数、浮点数，甚至是字符串。

以下代码演示了这一点

```{code-cell} ipython3
mc = qe.MarkovChain(P, state_values=('unemployed', 'employed'))
mc.simulate(ts_length=4, init='employed')
```

```{code-cell} ipython3
mc.simulate(ts_length=4, init='unemployed')
```

```{code-cell} ipython3
mc.simulate(ts_length=4)  # 从随机选择的初始状态开始
```

如果我们想要看到索引而不是状态值作为输出，我们可以使用

```{code-cell} ipython3
mc.simulate_indices(ts_length=4)
```

(mc_md)=
## {index}`边际分布 <single: 边际分布>`

```{index} single: 马尔可夫链; 边际分布
```

假设

1. $\{X_t\}$ 是一个具有随机矩阵 $P$ 的马尔可夫链
1. $X_t$ 的边际分布已知为 $\psi_t$

那么 $X_{t+1}$ 或更一般地 $X_{t+m}$ 的边际分布是什么？

为了回答这个问题，我们令 $\psi_t$ 为 $t = 0, 1, 2, \ldots$ 时 $X_t$ 的边际分布。

我们的首要目标是根据 $\psi_t$ 和 $P$ 找到 $\psi_{t + 1}$。

首先，选取任意 $y \in S$。

使用[全概率公式](https://en.wikipedia.org/wiki/Law_of_total_probability)，我们可以将 $X_{t+1} = y$ 的概率分解如下：

$$
\mathbb P \{X_{t+1} = y \}
   = \sum_{x \in S} \mathbb P \{ X_{t+1} = y \, | \, X_t = x \}
               \cdot \mathbb P \{ X_t = x \}
$$

用文字来说，要得到明天处于状态 $y$ 的概率，我们需要考虑所有可能发生的情况并将它们的概率相加。

用边际概率和条件概率重写这个表达式得到

$$
\psi_{t+1}(y) = \sum_{x \in S} P(x,y) \psi_t(x)
$$

对于每个 $y \in S$，都有一个这样的方程，共 $n$ 个方程。

如果我们将 $\psi_{t+1}$ 和 $\psi_t$ 视为*行向量*，这 $n$ 个方程可以用矩阵表达式概括为

```{math}
:label: fin_mc_fr

\psi_{t+1} = \psi_t P
```

因此，要将边际分布向前推进一个时间单位，我们需要右乘 $P$。

通过右乘 $m$ 次，我们可以将边际分布向未来推进 $m$ 步。

因此，通过迭代 {eq}`fin_mc_fr`，表达式 $\psi_{t+m} = \psi_t P^m$ 也是有效的 --- 这里 $P^m$ 是 $P$ 的 $m$ 次幂。

作为一个特例，我们可以看到，如果 $\psi_0$ 是 $X_0$ 的初始分布，那么 $\psi_0 P^m$ 就是 $X_m$ 的分布。

这一点非常重要，让我们重复一遍

```{math}
:label: mdfmc

X_0 \sim \psi_0 \quad \implies \quad X_m \sim \psi_0 P^m
```

更一般地，

```{math}
:label: mdfmc2

X_t \sim \psi_t \quad \implies \quad X_{t+m} \sim \psi_t P^m
```

(finite_mc_mstp)=
### 多步转移概率

我们知道从状态 $x$ 到 $y$ 的一步转移概率是 $P(x,y)$。

事实证明，从 $x$ 到 $y$ 的 $m$ 步转移概率是 $P^m(x,y)$，即 $P$ 的 $m$ 次幂的第 $(x,y)$ 个元素。

要理解这一点，再次考虑 {eq}`mdfmc2`，但现在假设 $\psi_t$ 在状态 $x$ 上的概率为1，其他位置为零，使得转移概率为：

* 在第 $x$ 个位置为1，其他位置为零

将此代入 {eq}`mdfmc2`，我们可以看到，在条件 $X_t = x$ 下，$X_{t+m}$ 的分布是 $P^m$ 的第 $x$ 行。

具体来说

$$
\mathbb P \{X_{t+m} = y \,|\, X_t = x \} = P^m(x, y) = P^m \text{的第} (x, y) \text{个元素}
$$

### 示例：衰退的概率

```{index} single: Markov Chains; Future Probabilities
```

回顾上面{ref}`考虑过的<mc_eg2>`衰退和增长的随机矩阵$P$。

假设当前状态未知 --- 可能统计数据仅在当前月份*结束时*才能获得。

我们猜测经济处于状态$x$的概率是$\psi(x)$。

6个月后经济处于衰退状态（轻度或严重）的概率由以下内积给出：

$$
\psi P^6
\cdot
\left(
  \begin{array}{c}
     0 \\
     1 \\
     1
  \end{array}
\right)
$$

(mc_eg1-1)=
### 示例2：横截面分布

```{index} single: Markov Chains; Cross-Sectional Distributions
```

我们一直在研究的边际分布可以被视为概率，也可以被视为大数定律使我们预期在大样本中出现的横截面频率。

为了说明这一点，让我们回顾{ref}`上面讨论过的<mc_eg1>`关于某个工人就业/失业动态的模型。

考虑一个大规模的工人群体，每个工人的终身经历都由特定的动态过程描述，每个工人的结果都是统计上独立于其他所有工人过程的实现。

让$\psi$表示当前在$\{ 0, 1 \}$上的*横截面*分布。

横截面分布记录了在某一时刻就业和失业工人的比例。

* 例如，$\psi(0)$是失业率。

10个周期后的横截面分布会是什么？

答案是$\psi P^{10}$，其中$P$是{eq}`p_unempemp`中的随机矩阵。

这是因为每个工人的状态都按照$P$演变，所以$\psi P^{10}$是随机选择的单个工人的边际分布。

但当样本量很大时，结果和概率大致相等（根据大数定律）。

因此，对于一个非常大的（趋向于无限的）人口，

$\psi P^{10}$ 也表示每个状态中工人的比例。

这正是横截面分布。

## {index}`不可约性和非周期性 <single: Irreducibility and Aperiodicity>`

```{index} single: Markov Chains; Irreducibility, Aperiodicity
```

不可约性和非周期性是现代马尔可夫链理论的核心概念。

让我们来看看它们是什么。

### 不可约性

设 $P$ 是一个固定的随机矩阵。

如果存在正整数 $j$ 和 $k$ 使得

$$
P^j(x, y) > 0
\quad \text{和} \quad
P^k(y, x) > 0
$$

则称两个状态 $x$ 和 $y$ 是**互通的**。

根据我们{ref}`上面的讨论 <finite_mc_mstp>`，这正好意味着

* 状态 $x$ 最终可以从状态 $y$ 到达，且
* 状态 $y$ 最终可以从状态 $x$ 到达

如果所有状态都是互通的，即对于 $S \times S$ 中的所有 $(x, y)$，$x$ 和 $y$ 都是互通的，则称随机矩阵 $P$ 是**不可约的**。

例如，考虑以下一组虚构家庭财富的转移概率

```{figure} /_static/lecture_specific/finite_markov/mc_irreducibility1.png

```

我们可以将其转换为随机矩阵，在节点之间没有边的地方填零

$$
P :=
\left(
  \begin{array}{ccc}
     0.9 & 0.1 & 0 \\
     0.4 & 0.4 & 0.2 \\
     0.1 & 0.1 & 0.8
  \end{array}
\right)
$$

从图中可以清楚地看出，这个随机矩阵是不可约的：我们最终可以从任何一个状态到达任何其他状态。

我们也可以使用[QuantEcon.py](http://quantecon.org/quantecon-py)的MarkovChain类来测试这一点

```{code-cell} ipython3
P = [[0.9, 0.1, 0.0],
     [0.4, 0.4, 0.2],
     [0.1, 0.1, 0.8]]

mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))
mc.is_irreducible
```

这是一个更悲观的情景，其中穷人永远保持贫穷

```{figure} /_static/lecture_specific/finite_markov/mc_irreducibility2.png

```

这个随机矩阵不是不可约的，因为，例如，从穷人状态无法到达富人状态。

让我们来验证这一点

```{code-cell} ipython3
P = [[1.0, 0.0, 0.0],
     [0.1, 0.8, 0.1],
     [0.0, 0.2, 0.8]]

mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))
mc.is_irreducible
```

我们也可以确定"通信类"

```{code-cell} ipython3
mc.communication_classes
```

你可能已经清楚地意识到，不可约性对于长期结果来说将会很重要。

例如，在第二个图中贫困是一个终身状态，但在第一个图中则不是。

我们稍后会再回到这个话题。

### 非周期性

简单来说，如果马尔可夫链以可预测的方式循环，我们称之为**周期性**的，否则称为**非周期性**的。

这里有一个包含三个状态的简单例子

```{figure} /_static/lecture_specific/finite_markov/mc_aperiodicity1.png

```

该链以周期3循环：

```{code-cell} ipython3
P = [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]]

mc = qe.MarkovChain(P)
mc.period
```

更正式地说，一个状态 $x$ 的**周期**是一个整数集合的最大公约数

$$
D(x) := \{j \geq 1 : P^j(x, x) > 0\}
$$

在上一个例子中，对于每个状态 $x$，$D(x) = \{3, 6, 9, \ldots\}$，所以周期是3。

如果每个状态的周期都是1，随机矩阵被称为**非周期的**，否则称为**周期的**。

例如，下面这个转移概率对应的随机矩阵是周期的，因为状态 $a$ 的周期是2

```{figure} /_static/lecture_specific/finite_markov/mc_aperiodicity2.png

```

我们可以用以下代码确认这个随机矩阵是周期的

```{code-cell} ipython3
P = [[0.0, 1.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0],
     [0.0, 0.5, 0.0, 0.5],
     [0.0, 0.0, 1.0, 0.0]]

mc = qe.MarkovChain(P)
mc.period
```

```{code-cell} ipython3
mc.is_aperiodic
```

## {index}`平稳分布 <single: 平稳分布>`

```{index} single: 马尔可夫链; 平稳分布
```

如{eq}`fin_mc_fr`所示，我们可以通过后乘$P$将边际分布向前移动一个时间单位。

某些分布在这种更新过程下保持不变 --- 例如，

```{code-cell} ipython3
P = np.array([[0.4, 0.6],
              [0.2, 0.8]])
ψ = (0.25, 0.75)
ψ @ P
```

这种分布被称为**平稳分布**或**不变分布**。

(mc_stat_dd)=
形式上，如果对于转移矩阵$P$，边际分布$\psi^*$满足$\psi^* = \psi^* P$，则称其为$P$的**平稳分布**。

（这与我们在{doc}`AR(1)过程讲座 <intro:ar1_processes>`中学到的平稳性概念是相同的，只是应用在不同的场景中。）

从这个等式中，我们可以立即得到对于所有$t$都有$\psi^* = \psi^* P^t$。

这告诉我们一个重要的事实：如果$X_0$的分布是平稳分布，那么对于所有$t$，$X_t$将具有相同的分布。

因此，平稳分布可以自然地解释为**随机稳态**——我们很快会详细讨论这一点。

从数学角度来看，当将$P$视为从（行）向量到（行）向量的映射$\psi \mapsto \psi P$时，平稳分布是$P$的不动点。

**定理：**每个随机矩阵$P$至少有一个平稳分布。

(我们在此假设状态空间 $S$ 是有限的；如果不是则需要更多假设)

对于这个结果的证明，你可以应用[布劳威尔不动点定理](https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem)，或参见[EDTC](https://johnstachurski.net/edtc.html)中的定理4.3.5。

对于给定的随机矩阵 $P$，可能存在多个平稳分布。

* 例如，如果 $P$ 是单位矩阵，那么所有边际分布都是平稳的。

要获得唯一的不变分布，转移矩阵 $P$ 必须具有这样的性质：状态空间的任何非平凡子集都不能是**无限持续的**。

如果从状态空间的某个子集无法到达其他部分，则该子集是无限持续的。

因此，非平凡子集的无限持续性与不可约性是相对的。

这为以下基本定理提供了一些直观理解。

(mc_conv_thm)=

**定理。** 如果$P$既是非周期的又是不可约的，那么

1. $P$恰好有一个平稳分布$\psi^*$。
1. 对于任何初始边际分布$\psi_0$，当$t \to \infty$时，有$\| \psi_0 P^t - \psi^* \| \to 0$。

证明可参见{cite}`haggstrom2002finite`的定理5.2。

(注意定理的第1部分只需要不可约性，而第2部分需要同时具备不可约性和非周期性)

满足该定理条件的随机矩阵有时被称为**一致遍历的**。

$P$的每个元素都严格大于零是非周期性和不可约性的充分条件。

* 试着说服自己这一点是正确的。

### 示例

回想我们之前{ref}`讨论过的 <mc_eg1>`关于某个工人就业/失业动态的模型。

假设$\alpha \in (0,1)$且$\beta \in (0,1)$，一致遍历条件是满足的。

令 $\psi^* = (p, 1-p)$ 为平稳分布，其中 $p$ 对应失业状态（状态0）。

使用 $\psi^* = \psi^* P$ 并经过一些代数运算得到

$$
p = \frac{\beta}{\alpha + \beta}
$$

这在某种意义上是失业的稳态概率 --- 关于这个解释的更多内容见下文。

不出所料，当 $\beta \to 0$ 时，它趋近于零；当 $\alpha \to 0$ 时，它趋近于一。

### 计算平稳分布

```{index} single: Markov Chains; Calculating Stationary Distributions
```

如上所述，一个特定的马尔可夫矩阵 $P$ 可能有多个平稳分布。

也就是说，可能存在多个行向量 $\psi$ 满足 $\psi = \psi P$。

事实上，如果 $P$ 有两个不同的平稳分布 $\psi_1, \psi_2$，那么它就有无限多个平稳分布，因为在这种情况下，正如你可以验证的那样，对于任意 $\lambda \in [0, 1]$

$$
\psi_3 := \lambda \psi_1 + (1 - \lambda) \psi_2
$$

都是 $P$ 的一个平稳分布。

如果我们将注意力限制在只存在一个平稳分布的情况下，找到它的一种方法是求解系统

$$
\psi (I_n - P) = 0
$$ (eq:eqpsifixed)

其中$I_n$是$n \times n$单位矩阵，求解$\psi$。

但是零向量可以满足系统{eq}`eq:eqpsifixed`，所以我们必须谨慎处理。

我们要施加$\psi$是概率分布的限制条件。

有多种方法可以做到这一点。

一种选择是将求解系统{eq}`eq:eqpsifixed`视为特征向量问题：满足$\psi = \psi P$的向量$\psi$是与单位特征值$\lambda = 1$相关的左特征向量。

[QuantEcon.py](http://quantecon.org/quantecon-py)实现了一个专门用于随机矩阵的稳定且复杂的算法。

这是我们推荐的方法：

```{code-cell} ipython3
P = [[0.4, 0.6],
     [0.2, 0.8]]

mc = qe.MarkovChain(P)
mc.stationary_distributions  # 显示所有平稳分布
```

### 收敛到平稳分布

```{index} single: 马尔可夫链; 收敛到平稳分布
```

马尔可夫链收敛定理的第2部分{ref}`如上所述<mc_conv_thm>`告诉我们，无论从何处开始，$X_t$的边际分布都会收敛到平稳分布。

这大大加强了我们将$\psi^*$解释为随机稳态的观点。

下图展示了定理中的收敛过程

```{code-cell} ipython
P = ((0.971, 0.029, 0.000),
     (0.145, 0.778, 0.077),
     (0.000, 0.508, 0.492))
P = np.array(P)

ψ = (0.0, 0.2, 0.8)        # 初始条件

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
       xticks=(0.25, 0.5, 0.75),
       yticks=(0.25, 0.5, 0.75),
       zticks=(0.25, 0.5, 0.75))

x_vals, y_vals, z_vals = [], [], []
for t in range(20):
    x_vals.append(ψ[0])
    y_vals.append(ψ[1])
    z_vals.append(ψ[2])
    ψ = ψ @ P

ax.scatter(x_vals, y_vals, z_vals, c='r', s=60)
ax.view_init(30, 210)

mc = qe.MarkovChain(P)
ψ_star = mc.stationary_distributions[0]
ax.scatter(ψ_star[0], ψ_star[1], ψ_star[2], c='k', s=60)

plt.show()
```

这里

* $P$ 是{ref}`上文讨论的 <mc_eg2>`衰退和增长的随机矩阵。
* 最高的红点是任意选择的初始边际概率分布 $\psi$，表示为 $\mathbb R^3$ 中的一个向量。
* 其他红点是边际分布 $\psi P^t$，其中 $t = 1, 2, \ldots$。
* 黑点是 $\psi^*$。

你可以尝试用不同的初始条件进行实验。

(ergodicity)=
## {index}`遍历性 <single: Ergodicity>`

```{index} single: 马尔可夫链; 遍历性
```

在不可约性条件下，还有另一个重要的结果：对于所有 $x \in S$，

```{math}
:label: llnfmc0

\frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = x\}  \to \psi^*(x)
    \quad \text{当 } m \to \infty
```

这里

* $\mathbf{1}\{X_t = x\} = 1$ 当且仅当 $X_t = x$，否则为零
* 收敛是以概率1发生的
* 该结果不依赖于 $X_0$ 的边际分布

结果表明，随着时间趋向无穷，马尔可夫链在状态 $x$ 停留的时间比例收敛到 $\psi^*(x)$。

(new_interp_sd)=
这为我们提供了另一种解释平稳分布的方式 --- 前提是{eq}`llnfmc0`中的收敛结果成立。

{eq}`llnfmc0`中所述的收敛是马尔可夫链大数定律结果的一个特例 --- 更多相关信息请参见[EDTC](http://johnstachurski.net/edtc.html)第4.3.4节。

(mc_eg1-2)=
### 示例

回想我们之前{ref}`讨论过的<mc_eg1-1>`就业/失业模型的横截面解释。

假设 $\alpha \in (0,1)$ 且 $\beta \in (0,1)$，因此不可约性和非周期性都成立。

我们看到平稳分布是 $(p, 1-p)$，其中

$$
p = \frac{\beta}{\alpha + \beta}
$$

在横截面解释中，这是失业人口的比例。

根据我们最新的（遍历性）结果，这也是单个工人预期处于失业状态的时间比例。

因此，从长远来看，群体的横截面平均值和个人的时间序列平均值是一致的。

这是遍历性概念的一个方面。

(finite_mc_expec)=
## 计算期望值

```{index} single: 马尔可夫链; 预测未来值
```

我们有时需要计算 $X_t$ 函数的数学期望值，形式如下：

```{math}
:label: mc_une

\mathbb E [ h(X_t) ]
```

以及条件期望值，如：

```{math}
:label: mc_cce

\mathbb E [ h(X_{t + k})  \mid X_t = x]
```

其中：

* $\{X_t\}$ 是由 $n \times n$ 随机矩阵 $P$ 生成的马尔可夫链
* $h$ 是给定函数，从矩阵代数的角度来看，我们将其视为列向量

$$
h
= \left(
\begin{array}{c}
    h(x_1) \\
    \vdots \\
    h(x_n)
\end{array}
  \right)
$$

计算无条件期望值 {eq}`mc_une` 很简单。

我们只需对 $X_t$ 的边际分布求和即可得到

$$
\mathbb E [ h(X_t) ]
= \sum_{x \in S} (\psi P^t)(x) h(x)
$$

这里 $\psi$ 是 $X_0$ 的分布。

由于 $\psi$ 以及 $\psi P^t$ 都是行向量，我们也可以写成

$$
\mathbb E [ h(X_t) ]
=  \psi P^t h
$$

对于条件期望 {eq}`mc_cce`，我们需要对给定 $X_t = x$ 时 $X_{t + k}$ 的条件分布求和。

我们已经知道这是 $P^k(x, \cdot)$，所以

```{math}
:label: mc_cce2

\mathbb E [ h(X_{t + k})  \mid X_t = x]
= (P^k h)(x)
```

向量 $P^k h$ 存储了所有 $x$ 的条件期望 $\mathbb E [ h(X_{t + k})  \mid X_t = x]$。

### 迭代期望

**迭代期望法则**指出

$$
\mathbb E \left[ \mathbb E [ h(X_{t + k})  \mid X_t = x] \right] = \mathbb E [  h(X_{t + k}) ]
$$

其中左边的外部期望 $ \mathbb E$ 是关于 $X_t$ 的边际分布 $\psi_t$ 的无条件分布（参见方程 {eq}`mdfmc2`）。

为了验证迭代期望法则，使用方程 {eq}`mc_cce2` 将 $ (P^k h)(x)$ 代入 $E [ h(X_{t + k})  \mid X_t = x]$，写作

$$
\mathbb E \left[ \mathbb E [ h(X_{t + k})  \mid X_t = x] \right] = \psi_t P^k h,
$$

并注意 $\psi_t P^k h = \psi_{t+k} h = \mathbb E [  h(X_{t + k}) ] $。

### 几何和的期望

有时我们想要计算几何和的数学期望，例如 $\sum_t \beta^t h(X_t)$。

根据前面的讨论，这等于

$$
\mathbb{E} [
        \sum_{j=0}^\infty \beta^j h(X_{t+j}) \mid X_t = x
    \Bigr]
= [(I - \beta P)^{-1} h](x)
$$

其中

$$
(I - \beta P)^{-1}  = I + \beta P + \beta^2 P^2 + \cdots
$$

乘以 $(I - \beta P)^{-1}$ 相当于"应用**预解算子**"。

## 练习

```{exercise}
:label: fm_ex1

根据{ref}`上述讨论 <mc_eg1-2>`，如果一个工人的就业动态遵循随机矩阵

$$
P
= \left(
\begin{array}{cc}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{array}
  \right)
$$

其中 $\alpha \in (0,1)$ 且 $\beta \in (0,1)$，那么从长期来看，失业时间的比例将是

$$
p := \frac{\beta}{\alpha + \beta}
$$

换句话说，如果 $\{X_t\}$ 表示就业的马尔可夫链，那么当 $m \to \infty$ 时，$\bar X_m \to p$，其中

$$
\bar X_m := \frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = 0\}
$$

本练习要求你通过计算大规模 $m$ 的 $\bar X_m$ 并验证其接近 $p$ 来说明这种收敛性。

你会发现，只要 $\alpha, \beta$ 都在 $(0, 1)$ 区间内，无论选择什么初始条件，这个结论都是成立的。
```


```{solution-start} fm_ex1
:class: dropdown
```

我们将通过图形方式解决这个练习。

图表显示了两种初始条件下 $\bar X_m - p$ 的时间序列。

随着 $m$ 变大，两个序列都收敛于零。

```{code-cell} ipython3
α = β = 0.1
N = 10000
p = β / (α + β)

P = ((1 - α,       α),               # 注意：P和p是不同的
     (    β,   1 - β))
mc = MarkovChain(P)

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_ylim(-0.25, 0.25)
ax.grid()
ax.hlines(0, 0, N, lw=2, alpha=0.6)   # 在零处画水平线

for x0, col in ((0, 'blue'), (1, 'green')):
    # 生成从x0开始的工人的时间序列
    X = mc.simulate(N, init=x0)
    # 计算每个n的失业时间比例
    X_bar = (X == 0).cumsum() / (1 + np.arange(N, dtype=float))
    # 绘图
    ax.fill_between(range(N), np.zeros(N), X_bar - p, color=col, alpha=0.1)
    ax.plot(X_bar - p, color=col, label=f'$X_0 = \, {x0} $')
    # 用黑色覆盖--使线条更清晰
    ax.plot(X_bar - p, 'k-', alpha=0.6)

ax.legend(loc='upper right')
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: fm_ex2
```

*排名*是经济学和许多其他学科关注的一个话题。

现在让我们来考虑一个最实用且重要的排名问题——搜索引擎对网页的排名。

（尽管这个问题源自经济学之外，但搜索排名系统与某些竞争均衡中的价格实际上存在深层联系——参见{cite}`DLP2013`。）

为了理解这个问题，考虑一下网络搜索引擎查询返回的结果集。

对用户来说，理想的是：

1. 获得大量准确的匹配结果
1. 按顺序返回匹配结果，这个顺序对应某种"重要性"的衡量标准

根据重要性衡量标准进行排名就是我们现在要考虑的问题。

Google创始人拉里·佩奇和谢尔盖·布林开发的解决这个问题的方法被称为[PageRank](https://en.wikipedia.org/wiki/PageRank)。

为了说明这个概念，请看下面的图表

```{figure} /_static/lecture_specific/finite_markov/web_graph.png
```

想象这是万维网的一个微型版本，其中

* 每个节点代表一个网页
* 每个箭头代表从一个页面到另一个页面的链接存在

现在让我们思考哪些页面可能是重要的，即对搜索引擎用户来说具有价值的页面。

衡量页面重要性的一个可能标准是入站链接的数量——这表明了受欢迎程度。

按照这个标准，`m`和`j`是最重要的页面，各有5个入站链接。

但是，如果链接到`m`的页面本身并不重要呢？

这样想的话，似乎应该根据相对重要性来对入站节点进行加权。

PageRank算法正是这样做的。

下面是一个稍微简化的介绍，但捕捉到了基本思想。

令 $j$ 为（整数索引的）典型页面，$r_j$ 为其排名，我们设定

$$
r_j = \sum_{i \in L_j} \frac{r_i}{\ell_i}
$$

其中

* $\ell_i$ 是从页面 $i$ 出发的外链总数
* $L_j$ 是所有链接到页面 $j$ 的页面 $i$ 的集合

这是一个衡量入站链接数量的指标，根据链接来源页面的排名进行加权（并通过 $1 / \ell_i$ 进行归一化）。

然而，还有另一种解释，这让我们回到马尔可夫链。

令 $P$ 为矩阵，其中 $P(i, j) = \mathbf 1\{i \to j\} / \ell_i$，这里 $\mathbf 1\{i \to j\} = 1$ 表示页面 $i$ 有链接指向 $j$，否则为零。

如果每个页面至少有一个链接，则矩阵 $P$ 是一个随机矩阵。

基于这个 $P$ 的定义，我们有

$$
r_j
= \sum_{i \in L_j} \frac{r_i}{\ell_i}
= \sum_{\text{所有 } i} \mathbf 1\{i \to j\} \frac{r_i}{\ell_i}
= \sum_{\text{所有 } i} P(i, j) r_i
$$

将排名写成行向量 $r$，这就变成了 $r = r P$。

因此 $r$ 是随机矩阵 $P$ 的平稳分布。

让我们将 $P(i, j)$ 理解为从页面 $i$ "移动"到页面 $j$ 的概率。

$P(i, j)$ 的值可以解释为：

* 如果页面 $i$ 有 $k$ 个出站链接，且 $j$ 是其中之一，则 $P(i, j) = 1/k$
* 如果页面 $i$ 没有直接链接到 $j$，则 $P(i, j) = 0$

因此，从一个页面到另一个页面的移动就像一个网络浏览者通过随机点击页面上的某个链接来移动。

这里的"随机"意味着每个链接被选中的概率相等。

由于 $r$ 是 $P$ 的平稳分布，假设一致遍历性条件成立，我们{ref}`可以解释 <new_interp_sd>` $r_j$ 为一个（非常持久的）随机浏览者在页面 $j$ 停留的时间比例。

你的练习是将这个排名算法应用到上图所示的图中，并返回按排名顺序排列的页面列表。

总共有14个节点（即网页），第一个命名为 `a`，最后一个命名为 `n`。

文件中的典型行具有以下形式

```{code-block} none
d -> h;
```

这应该被理解为存在一个从`d`到`h`的链接。

下面显示了这个图的数据，当单元格执行时，这些数据被读入名为`web_graph_data.txt`的文件中。

```{code-cell} ipython
%%file web_graph_data.txt
a -> d;
a -> f;
b -> j;
b -> k;
b -> m;
c -> c;
c -> g;
c -> j;
c -> m;
d -> f;
d -> h;
d -> k;
e -> d;
e -> h;
e -> l;
f -> a;
f -> b;
f -> j;
f -> l;
g -> b;
g -> j;
h -> d;
h -> g;
h -> l;
h -> m;
i -> g;
i -> h;
i -> n;
j -> e;
j -> i;
j -> k;
k -> n;
l -> m;
m -> g;
n -> c;
n -> j;
n -> m;
```

要解析这个文件并提取相关信息，你可以使用[正则表达式](https://docs.python.org/3/library/re.html)。

下面的代码片段提供了一个关于如何实现的提示

```{code-cell} ipython3
import re
re.findall(r'\w', 'x +++ y ****** z')  # \w 匹配字母数字字符
```

```{code-cell} ipython3
re.findall(r'\w', 'a ^^ b &&& $$ c')
```

当你解出排名时，你会发现实际上`g`是排名最高的节点，而`a`是排名最低的。

```{exercise-end}
```


```{solution-start} fm_ex2
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
"""
返回按排名排序的页面列表
"""
import re
from operator import itemgetter

infile = 'web_graph_data.txt'
alphabet = 'abcdefghijklmnopqrstuvwxyz'

n = 14 # 网页（节点）总数

# 创建一个表示链接存在的矩阵Q
#  * Q[i, j] = 1 表示从i到j存在链接
#  * Q[i, j] = 0 表示不存在链接
Q = np.zeros((n, n), dtype=int)
with open(infile) as f:
    edges = f.readlines()
for edge in edges:
    from_node, to_node = re.findall('\w', edge)
    i, j = alphabet.index(from_node), alphabet.index(to_node)
    Q[i, j] = 1
# 创建相应的马尔可夫矩阵P
P = np.empty((n, n))
for i in range(n):
    P[i, :] = Q[i, :] / Q[i, :].sum()
mc = MarkovChain(P)
# 计算稳态分布r
r = mc.stationary_distributions[0]
ranked_pages = {alphabet[i] : r[i] for i in range(n)}
# 打印解决方案，从最高排名到最低排名排序
print('排名\n ***')
for name, rank in sorted(ranked_pages.items(), key=itemgetter(1), reverse=1):
    print(f'{name}: {rank:.4}')
```

```{solution-end}
```


```{exercise}
:label: fm_ex3

在数值计算中，有时用离散模型替代连续模型会比较方便。

特别是，马尔可夫链经常被用作如下AR(1)过程的离散近似：

$$
y_{t+1} = \rho y_t + u_{t+1}
$$

这里假设${u_t}$是独立同分布的，且服从$N(0, \sigma_u^2)$。

$\{ y_t \}$的平稳概率分布的方差为

$$
\sigma_y^2 := \frac{\sigma_u^2}{1-\rho^2}
$$

Tauchen方法{cite}`Tauchen1986`是将这个连续状态过程近似为有限状态马尔可夫链最常用的方法。

[QuantEcon.py](http://quantecon.org/quantecon-py)中已经有了这个程序，但让我们作为练习自己写一个版本。

作为第一步，我们选择：

* $n$，离散近似的状态数
* $m$，一个用于参数化状态空间宽度的整数

接下来，我们创建一个状态空间$\{x_0, \ldots, x_{n-1}\} \subset \mathbb R$

以及一个随机 $n \times n$ 矩阵 $P$，满足以下条件：

* $x_0 = - m \, \sigma_y$
* $x_{n-1} = m \, \sigma_y$
* $x_{i+1} = x_i + s$ 其中 $s = (x_{n-1} - x_0) / (n - 1)$

令 $F$ 为正态分布 $N(0, \sigma_u^2)$ 的累积分布函数。

$P(x_i, x_j)$ 的值是为了近似 AR(1) 过程而计算的 --- 省略推导过程，规则如下：

1. 如果 $j = 0$，则设

   $$
   P(x_i, x_j) = P(x_i, x_0) = F(x_0-\rho x_i + s/2)
   $$

1. 如果 $j = n-1$，则设

   $$
   P(x_i, x_j) = P(x_i, x_{n-1}) = 1 - F(x_{n-1} - \rho x_i - s/2)
   $$

1. 否则，设

   $$
   P(x_i, x_j) = F(x_j - \rho x_i + s/2) - F(x_j - \rho x_i - s/2)
   $$

练习要求编写一个函数 `approx_markov(rho, sigma_u, m=3, n=7)`，返回
$\{x_0, \ldots, x_{n-1}\} \subset \mathbb R$ 和 $n \times n$ 矩阵
$P$，如上所述。

* 更好的方法是编写一个函数，返回[QuantEcon.py](http://quantecon.org/quantecon-py)的MarkovChain类的实例。
```

```{solution} fm_ex3
:class: dropdown

可以在[QuantEcon.py](http://quantecon.org/quantecon-py)库中找到解决方案，
具体见[这里](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/approximation.py)。

```

[^pm]: 提示：首先证明如果P和Q是随机矩阵，那么它们的乘积也是随机矩阵——要检查行和，试着用一列1向量进行后乘。最后，用归纳法论证P^n是随机矩阵。

