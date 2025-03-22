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

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Kesten过程与企业动态

```{index} single: Linear State Space Models
```

```{contents} Contents
:depth: 2
```

除了Anaconda中包含的内容外，本讲座还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
!pip install --upgrade yfinance
```

## 概述

{doc}`之前 <intro:ar1_processes>` 我们学习了线性标量值随机过程（AR(1)模型）。

现在，我们通过允许乘数系数是随机的来稍微推广这些线性模型。

这些过程被称为Kesten过程，以德裔美国数学家Harry Kesten（1931-2019）的名字命名。

尽管写起来很简单，但Kesten过程至少有两个有趣的原因：

1. 许多重要的经济过程可以或已经被描述为Kesten过程。
1. Kesten过程产生有趣的动态，在某些情况下，包括重尾横截面分布。

我们将在讨论过程中详细说明这些问题。

让我们从一些导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
import quantecon as qe
```

以下两行仅用于避免pandas和matplotlib之间的兼容性问题导致的`FutureWarning`。

```{code-cell} ipython
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
```

与本讲座相关的额外技术背景可以在{cite}`buraczewski2016stochastic`的专著中找到。

## Kesten过程

```{index} single: Kesten processes; heavy tails
```

**Kesten过程**是以下形式的随机过程：

```{math}
:label: kesproc

X_{t+1} = a_{t+1} X_t + \eta_{t+1}
```

其中$\{a_t\}_{t \geq 1}$和$\{\eta_t\}_{t \geq 1}$是独立同分布序列。

我们感兴趣的是当$X_0$给定时$\{X_t\}_{t \geq 0}$的动态。

我们将关注非负标量情况，其中$X_t$取值于$\mathbb R_+$。

特别地，我们假设：

* 初始条件$X_0$是非负的，
* $\{a_t\}_{t \geq 1}$是非负的独立同分布随机过程，且
* $\{\eta_t\}_{t \geq 1}$是另一个非负的独立同分布随机过程，与第一个过程独立。

### 示例：GARCH波动率

GARCH模型在金融应用中很常见，其中时间序列（如资产收益）表现出时变波动率。

例如，考虑以下纳斯达克综合指数从2006年1月1日到2019年11月1日的日收益率图。

(ndcode)=
```{code-cell} python3
import yfinance as yf

s = yf.download('^IXIC', '2006-1-1', '2019-11-1', auto_adjust=False)['Adj Close']

r = s.pct_change()

fig, ax = plt.subplots()

ax.plot(r, alpha=0.7)

ax.set_ylabel('收益率', fontsize=12)
ax.set_xlabel('日期', fontsize=12)

plt.show()
```

注意该序列如何表现出波动率爆发（高方差）然后又趋于平稳。

GARCH模型可以复制这一特征。

GARCH(1, 1)波动率过程的形式为：

```{math}
:label: garch11v

\sigma_{t+1}^2 = \alpha_0 + \sigma_t^2 (\alpha_1 \xi_{t+1}^2 + \beta)
```

其中$\{\xi_t\}$是独立同分布的，$\mathbb E \xi_t^2 = 1$且所有参数都是正的。

给定资产的收益率则建模为：

```{math}
:label: garch11r

r_t = \sigma_t \zeta_t
```

其中$\{\zeta_t\}$也是独立同分布的，且与$\{\xi_t\}$独立。

波动率序列$\{\sigma_t^2 \}$，驱动收益率的动态，是一个Kesten过程。

### 示例：财富动态

假设每个家庭在每个时期都将其当前财富的固定比例$s$用于储蓄。

家庭在时间$t$开始时获得劳动收入$y_t$。

财富则按照以下方式演化：

```{math}
:label: wealth_dynam

w_{t+1} = R_{t+1} s w_t  + y_{t+1}
```

其中$\{R_t\}$是资产的总收益率。

如果$\{R_t\}$和$\{y_t\}$都是独立同分布的，那么{eq}`wealth_dynam`就是一个Kesten过程。

### 平稳性

在之前的讲座中，如{doc}`AR(1)过程 <intro:ar1_processes>`，我们引入了平稳分布的概念。

在当前背景下，我们可以如下定义平稳分布：

$\mathbb R$上的分布$F^*$被称为Kesten过程{eq}`kesproc`的**平稳分布**，如果：

```{math}
:label: kp_stationary0

X_t \sim F^*
\quad \implies \quad
a_{t+1} X_t + \eta_{t+1} \sim F^*
```

换句话说，如果当前状态$X_t$具有分布$F^*$，那么下一期状态$X_{t+1}$也具有相同的分布。

我们可以将其写为：

```{math}
:label: kp_stationary

F^*(y) = \int \mathbb P\{ a_{t+1} x + \eta_{t+1} \leq y\} F^*(dx)
\quad \text{对所有 } y \geq 0
```

左边是当前状态从$F^*$中抽取时下一期状态的分布。

{eq}`kp_stationary`中的等式表明这个分布保持不变。

### 横截面解释

平稳分布有一个重要的横截面解释，之前已经讨论过但值得在这里重复。

假设，例如，我们对财富分布感兴趣 --- 即某个国家当前家庭财富的分布。

进一步假设：

* 每个家庭的财富根据{eq}`wealth_dynam`独立演化，
* $F^*$是这个随机过程的平稳分布，且
* 有大量家庭。

那么$F^*$就是这个国家横截面财富分布的稳态。

换句话说，如果$F^*$是当前的财富分布，那么在后续时期它将保持不变，*其他条件不变*。

要理解这一点，假设$F^*$是当前的财富分布。

下一期财富小于$y$的家庭比例是多少？

为了得到这个，我们对财富明天小于$y$的概率进行求和，给定当前财富为$w$，权重为具有财富$w$的家庭比例。

注意到具有财富在区间$dw$中的家庭比例是$F^*(dw)$，我们得到：

$$
\int \mathbb P\{ R_{t+1} s w  + y_{t+1} \leq y\} F^*(dw)
$$

根据平稳性的定义和$F^*$是财富过程的平稳分布的假设，这正好是$F^*(y)$。

因此，财富在$[0, y]$中的家庭比例在下一期与当前期相同。

由于$y$是任意选择的，分布保持不变。

### 平稳性条件

Kesten过程$X_{t+1} = a_{t+1} X_t + \eta_{t+1}$并不总是具有平稳分布。

例如，如果对所有$t$都有$a_t \equiv \eta_t \equiv 1$，那么$X_t = X_0 + t$，它会发散到无穷大。

为了防止这种发散，我们要求$\{a_t\}$在大多数时候严格小于1。

特别地，如果：

```{math}
:label: kp_stat_cond

\mathbb E \ln a_t < 0
\quad \text{且} \quad
\mathbb E \eta_t < \infty
```

那么在$\mathbb R_+$上存在唯一的平稳分布。

* 参见，例如，{cite}`buraczewski2016stochastic`的定理2.1.3，它提供了稍弱的条件。

作为这个结果的一个应用，我们看到，只要劳动收入具有有限均值且$\mathbb E \ln R_t  + \ln s < 0$，财富过程{eq}`wealth_dynam`就会有唯一的平稳分布。

## 重尾

在某些条件下，Kesten过程的平稳分布具有帕累托尾。

（参见我们{doc}`之前关于重尾分布的讲座 <intro:heavy_tails>`。）

这个事实对经济学很重要，因为帕累托尾分布很普遍。

### Kesten--Goldie定理

为了说明Kesten过程的平稳分布具有帕累托尾的条件，我们首先回顾，如果随机变量的分布不集中在$\{\dots, -2t, -t, 0, t, 2t, \ldots \}$上（对任何$t \geq 0$），则称该随机变量为**非算术的**。

例如，任何具有密度的随机变量都是非算术的。

著名的Kesten--Goldie定理（参见，例如，{cite}`buraczewski2016stochastic`，定理2.4.4）指出，如果：

1. {eq}`kp_stat_cond`中的平稳性条件成立，
1. 随机变量$a_t$以概率1为正且是非算术的，
1. 对所有$x \in \mathbb R_+$有$\mathbb P\{a_t x + \eta_t = x\} < 1$，且
1. 存在正常数$\alpha$使得：

$$
\mathbb E a_t^\alpha = 1,
    \quad
\mathbb E \eta_t^\alpha < \infty,
    \quad \text{且} \quad
\mathbb E [a_t^{\alpha+1} ] < \infty
$$

那么Kesten过程的平稳分布具有帕累托尾，尾指数为$\alpha$。

更精确地说，如果$F^*$是唯一的平稳分布且$X^* \sim F^*$，则：

$$
\lim_{x \to \infty} x^\alpha \mathbb P\{X^* > x\} = c
$$

其中$c$是某个正常数。

### 直觉

稍后我们将使用秩-规模图来说明Kesten--Goldie定理。

在此之前，我们可以对条件给出以下直觉。

两个重要条件是$\mathbb E \ln a_t < 0$（所以模型是平稳的）和$\mathbb E a_t^\alpha = 1$（对某个$\alpha > 0$）。

第一个条件意味着$a_t$的分布在1以下有大量概率质量。

第二个条件意味着$a_t$的分布在1或以上至少有一些概率质量。

第一个条件给我们平稳性的存在性。

第二个条件意味着当前状态可以被$a_t$扩大。

如果这种情况在几个连续时期发生，效果会相互复合，因为$a_t$是乘性的。

这导致时间序列中的尖峰，填充分布的极端右尾。

时间序列中的尖峰在以下模拟中可见，其中当$a_t$和$b_t$是对数正态分布时生成了10条路径：

```{code-cell} ipython3
μ = -0.5
σ = 1.0

def kesten_ts(ts_length=100):
    x = np.zeros(ts_length)
    for t in range(ts_length-1):
        a = np.exp(μ + σ * np.random.randn())
        b = np.exp(np.random.randn())
        x[t+1] = a * x[t] + b
    return x

fig, ax = plt.subplots()

num_paths = 10
np.random.seed(12)

for i in range(num_paths):
    ax.plot(kesten_ts())

ax.set(xlabel='时间', ylabel='$X_t$')
plt.show()
```

## 应用：企业动态

正如我们在{doc}`关于重尾的讲座 <intro:heavy_tails>`中提到的，对于收入或就业等常见的企业规模衡量指标，美国企业规模分布表现出帕累托尾（参见，例如，{cite}`axtell2001zipf`，{cite}`gabaix2016power`）。

让我们尝试使用Kesten--Goldie定理来解释这个相当惊人的事实。

### Gibrat定律

多年前，Robert Gibrat {cite}`gibrat1931inegalites`提出，企业规模按照一个简单的规则演化，即下一期规模与当前规模成比例。

这现在被称为[Gibrat比例增长定律](https://en.wikipedia.org/wiki/Gibrat%27s_law)。

我们可以通过说明企业规模的适当衡量指标$s_t$满足以下关系来表达这个想法：

```{math}
:label: firm_dynam_gb

\frac{s_{t+1}}{s_t} = a_{t+1}
```

其中$\{a_t\}$是某个正的独立同分布序列。

Gibrat定律的一个含义是，个别企业的增长率不依赖于它们的规模。

然而，在过去几十年里，与Gibrat定律相矛盾的研究在文献中积累。

例如，通常发现，平均而言：

1. 小企业比大企业增长更快（参见，例如，{cite}`evans1987relationship`和{cite}`hall1987relationship`），且
1. 小企业的增长率比大企业更不稳定{cite}`dunne1989growth`。

另一方面，Gibrat定律通常被发现对大企业是一个合理的近似{cite}`evans1987relationship`。

我们可以通过修改{eq}`firm_dynam_gb`为以下形式来适应这些实证发现：

```{math}
:label: firm_dynam

s_{t+1} = a_{t+1} s_t + b_{t+1}
```

其中$\{a_t\}$和$\{b_t\}$都是独立同分布的，且相互独立。

在练习中，你被要求证明{eq}`firm_dynam`比{eq}`firm_dynam_gb`中的Gibrat定律更符合上述实证发现。

### 重尾

那么这与帕累托尾有什么关系？

答案是{eq}`firm_dynam`是一个Kesten过程。

如果Kesten--Goldie定理的条件满足，那么企业规模分布预计会有重尾 --- 这正是我们在数据中看到的。

在下面的练习中，我们进一步探索这个想法，推广企业规模动态并检查相应的秩-规模图。

我们还试图说明为什么帕累托尾的发现对定量分析很重要。

## 练习

```{exercise}
:label: kp_ex1

使用{eq}`garch11v`--{eq}`garch11r`中的GARCH(1, 1)过程模拟并绘制15年的日收益率（考虑每年有250个工作日）。

取$\xi_t$和$\zeta_t$为独立的标准正态分布。

设置$\alpha_0 = 0.00001, \alpha_1 = 0.1, \beta = 0.9$和$\sigma_0 = 0$。

与{ref}`上面显示的 <ndcode>`纳斯达克综合指数收益率进行视觉比较。

虽然时间路径不同，但你应该能看到高波动率的爆发。
```


```{solution-start} kp_ex1
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
α_0 = 1e-5
α_1 = 0.1
β = 0.9

years = 15
days = years * 250

def garch_ts(ts_length=days):
    σ2 = 0
    r = np.zeros(ts_length)
    for t in range(ts_length-1):
        ξ = np.random.randn()
        σ2 = α_0 + σ2 * (α_1 * ξ**2 + β)
        r[t] = np.sqrt(σ2) * np.random.randn()
    return r

fig, ax = plt.subplots()

np.random.seed(12)

ax.plot(garch_ts(), alpha=0.7)

ax.set(xlabel='时间', ylabel='$\\sigma_t^2$')
plt.show()
```

```{solution-end}
```

```{exercise}
:label: kp_ex2

在我们对企业动态的讨论中，声称{eq}`firm_dynam`比{eq}`firm_dynam_gb`中的Gibrat定律更符合实证文献。

（实证文献在{eq}`firm_dynam`之前立即进行了回顾。）

在什么意义上这是正确的（或错误的）？
```

```{solution-start} kp_ex2
:class: dropdown
```

实证发现是：

1. 小企业比大企业增长更快，且
1. 小企业的增长率比大企业更不稳定。

此外，Gibrat定律通常被发现对大企业比小企业更合理。

这个说法是{eq}`firm_dynam`中的动态比Gibrat定律更符合点1-2。

要理解原因，我们将{eq}`firm_dynam`重写为增长动态：

```{math}
:label: firm_dynam_2

\frac{s_{t+1}}{s_t} = a_{t+1} + \frac{b_{t+1}}{s_t}
```

给定$s_t = s$，企业增长率的均值和方差为：

$$
\mathbb E a
+ \frac{\mathbb E b}{s}
\quad \text{和} \quad
\mathbb V a
+ \frac{\mathbb V b}{s^2}
$$

这两者都随着企业规模$s$的增大而下降，与数据一致。

此外，当$s_t$变大时，运动规律{eq}`firm_dynam_2`显然接近Gibrat定律{eq}`firm_dynam_gb`。

```{solution-end}
```

```{exercise}
:label: kp_ex3

考虑{eq}`kesproc`中给出的任意Kesten过程。

假设$\{a_t\}$是对数正态分布，参数为$(\mu, \sigma)$。

换句话说，当$Z$是标准正态分布时，每个$a_t$与$\exp(\mu + \sigma Z)$具有相同的分布。

进一步假设$\mathbb E \eta_t^r < \infty$对所有$r > 0$成立，如果$\eta_t$也是对数正态分布，情况就会如此。

证明Kesten--Goldie定理的条件满足当且仅当$\mu < 0$。

求出使Kesten--Goldie条件成立的$\alpha$值。
```

```{solution-start} kp_ex3
:class: dropdown
```

由于$a_t$有密度，它是非算术的。

由于$a_t$与$a = \exp(\mu + \sigma Z)$具有相同的密度（当$Z$是标准正态分布时），我们有：

$$
\mathbb E \ln a_t = \mathbb E (\mu + \sigma Z) = \mu,
$$

且由于$\eta_t$具有所有阶的有限矩，平稳性条件成立当且仅当$\mu < 0$。

给定对数正态分布的性质（它具有所有阶的有限矩），唯一值得怀疑的条件是存在正常数$\alpha$使得$\mathbb E a_t^\alpha = 1$。

这等价于：

$$
\exp \left( \alpha \mu + \frac{\alpha^2 \sigma^2}{2} \right) = 1.
$$

解出$\alpha$得到$\alpha = -2\mu / \sigma^2$。

```{solution-end}
```


```{exercise-start}
:label: kp_ex4
```

{eq}`firm_dynam`中指定的企业动态的一个不现实方面是它忽略了进入和退出。

在任何给定时期和任何给定市场中，我们观察到大量企业进入和退出市场。

这个问题的实证讨论可以在Hugo Hopenhayn {cite}`hopenhayn1992entry`的一篇著名论文中找到。

在同一篇论文中，Hopenhayn建立了一个包含企业利润最大化和市场出清数量、工资和价格的进入退出模型。

在他的模型中，当进入企业数量等于退出企业数量时，出现平稳均衡。

在这种背景下，企业动态可以表示为：

```{math}
:label: firm_dynam_ee

s_{t+1} = e_{t+1} \mathbb{1}\{s_t < \bar s\} +
(a_{t+1} s_t + b_{t+1}) \mathbb{1}\{s_t \geq \bar s\}
```

其中：

* 状态变量$s_t$代表生产率（这是产出和企业规模的代理变量），
* 独立同分布序列$\{ e_t \}$被视为新进入企业的生产率抽取，且
* 变量$\bar s$是一个阈值，我们将其视为给定的，尽管它在Hopenhayn的模型中是内生的。

{eq}`firm_dynam_ee`背后的思想是，只要企业的生产率$s_t$保持在$\bar s$或以上，它们就留在市场中。

* 在这种情况下，它们的生产率根据{eq}`firm_dynam`更新。

当企业的生产率$s_t$低于$\bar s$时，它们选择退出。

* 在这种情况下，它们被生产率$e_{t+1}$的新企业替代。

我们能对动态说些什么？

虽然{eq}`firm_dynam_ee`不是Kesten过程，但当$s_t$很大时，它的更新方式与Kesten过程相同。

那么也许它的平稳分布仍然有帕累托尾？

你的任务是通过模拟和秩-规模图来研究这个问题。

方法将是：

1. 当$M$和$T$很大时生成$M$个$s_T$的抽取，且
1. 在秩-规模图中绘制结果中最大的1,000个抽取。

（当$T$很大时，$s_T$的分布将接近平稳分布。）

在模拟中，假设：

* $a_t, b_t$和$e_t$都是对数正态分布，
* 参数为：

```{code-cell} ipython3
μ_a = -0.5        # a的位置参数
σ_a = 0.1         # a的尺度参数
μ_b = 0.0         # b的位置参数
σ_b = 0.5         # b的尺度参数
μ_e = 0.0         # e的位置参数
σ_e = 0.5         # e的尺度参数
s_bar = 1.0       # 阈值
T = 500           # 采样日期
M = 1_000_000     # 企业数量
s_init = 1.0      # 每个企业的初始条件
```

```{exercise-end}
```

```{solution-start} kp_ex4
:class: dropdown
```

这是一个解决方案。
首先我们生成观测值：

```{code-cell} ipython3
from numba import jit, prange
from numpy.random import randn


@jit(parallel=True)
def generate_draws(μ_a=-0.5,
                   σ_a=0.1,
                   μ_b=0.0,
                   σ_b=0.5,
                   μ_e=0.0,
                   σ_e=0.5,
                   s_bar=1.0,
                   T=500,
                   M=1_000_000,
                   s_init=1.0):

    draws = np.empty(M)
    for m in prange(M):
        s = s_init
        for t in range(T):
            if s < s_bar:
                new_s = np.exp(μ_e + σ_e *  randn())
            else:
                a = np.exp(μ_a + σ_a * randn())
                b = np.exp(μ_b + σ_b * randn())
                new_s = a * s + b
            s = new_s
        draws[m] = s

    return draws

data = generate_draws()
```

现在我们来生成秩-规模图：

```{code-cell} ipython3
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(data, c=0.01)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("对数秩")
ax.set_ylabel("对数规模")

plt.show()
```

该图产生一条直线，与帕累托尾一致。

```{solution-end}
```