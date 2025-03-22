---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(wald_friedman)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`让弥尔顿·弗里德曼困惑的问题 <single: A Problem that Stumped Milton Friedman>`

(而亚伯拉罕·瓦尔德通过发明序贯分析解决了这个问题)

```{index} single: Models; Sequential analysis
```

```{contents} 目录
:depth: 2
```

## 概述

本讲座描述了二战期间提交给弥尔顿·弗里德曼和W·艾伦·沃利斯的一个统计决策问题，当时他们是分析师在

美国哥伦比亚大学政府统计研究组。

这个问题促使Abraham Wald {cite}`Wald47` 提出了**序贯分析**，
这是一种与动态规划密切相关的统计决策问题方法。

在本讲中，我们将动态规划算法应用于Friedman、Wallis和Wald的问题。

主要涉及的概念包括：

- 贝叶斯定理
- 动态规划
- 第一类和第二类统计错误
    - 第一类错误是指在原假设为真时拒绝原假设
    - 第二类错误是指在原假设为假时接受原假设
- Abraham Wald的**序贯概率比检验**
- 统计检验的**功效**
- 统计检验的**临界区域**
- **一致最优检验**

我们先导入一些包：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from numba import jit, prange, float64, int64
from numba.experimental import jitclass
from math import gamma
```

本讲座使用了{doc}`本讲座 <likelihood_ratio_process>`、{doc}`本讲座 <likelihood_bayes>`和{doc}`本讲座 <exchangeable>`中研究的概念。

## 问题的起源

在米尔顿·弗里德曼与罗斯·弗里德曼1998年合著的《两个幸运儿》一书第137-139页{cite}`Friedman98`中，米尔顿·弗里德曼描述了二战期间他和艾伦·沃利斯在哥伦比亚大学美国政府统计研究组工作时遇到的一个问题。

```{note}
参见艾伦·沃利斯1980年发表的文章{cite}`wallis1980statistical`第25和26页，其中讲述了二战期间在哥伦比亚大学统计研究组的这段经历，以及哈罗德·霍特林对问题形成所做的重要贡献。另见珍妮弗·伯恩斯关于米尔顿·弗里德曼的著作{cite}`Burns_2023`第5章。
```

让我们听听米尔顿·弗里德曼是如何讲述这件事的

> 为了理解这个故事，有必要先了解一个

> 这是一个简单的统计问题，以及处理它的标准程序。让我们以顺序分析发展的实际问题为例。海军有两种备选的炮弹设计（比如说A和B）。他们想要确定哪一种更好。为此，他们进行了一系列成对的发射试验。在每一轮中，如果A的性能优于B，则给A赋值1，B赋值0；反之，如果A的性能劣于B，则给A赋值0，B赋值1。海军请统计学家告诉他们如何进行测试并分析结果。

> 标准的统计学答案是指定发射次数（比如1,000次）和一对百分比（例如，53%和47%），并告诉客户：如果A在超过53%的发射中获得1，则可以认为A更优；如果A在少于47%的发射中获得1，则可以认为B更优；如果百分比在47%和53%之间，则无法得出任何一方更优的结论。

> 当艾伦·沃利斯与（海军）加勒特·L·施莱尔上尉讨论这样一个问题时，
> 上尉提出反对意见，引用艾伦的说法，这样的测试可能会造成浪费。如果
> 像施莱尔这样明智且经验丰富的军需官在现场，他在看到前几千发甚至
> 前几百发[弹药]后就会发现实验不需要完成，要么是因为新方法明显
> 较差，要么是因为它明显优于预期$\ldots$。

弗里德曼和沃利斯为这个问题苦苦挣扎，但在意识到他们无法解决后，
向亚伯拉罕·瓦尔德描述了这个问题。

这让瓦尔德开始了通往*序贯分析*{cite}`Wald47`的道路。

我们将使用动态规划来阐述这个问题。

## 动态规划方法

以下对问题的介绍紧密遵循德米特里·伯斯克卡斯在**动态规划与随机控制**{cite}`Bertekas75`中的处理方式。

决策者可以观察一个随机变量$z$的一系列抽样。

他（或她）想要知道是概率分布$f_0$还是$f_1$支配着$z$。

在已知连续观测值是从分布$f_0$中抽取的条件下，这个随机变量序列是独立同分布的（IID）。

在已知连续观测值是从分布$f_1$中抽取的条件下，这个随机变量序列也是独立同分布的（IID）。

但观察者并不知道是哪一个分布生成了这个序列。

由[可交换性和贝叶斯更新](https://python.quantecon.org/exchangeable.html)中解释的原因，这意味着该序列不是独立同分布的。

观察者有需要学习的东西，即观测值是从$f_0$还是从$f_1$中抽取的。

决策者想要决定是哪一个分布在生成这些结果。

我们采用贝叶斯公式。

决策者从先验概率开始

$$
\pi_{-1} =
\mathbb P \{ f = f_0 \mid \textrm{ no observations} \} \in (0, 1)
$$

在观察到$k+1$个观测值$z_k, z_{k-1}, \ldots, z_0$后，他更新其个人对观测值由分布$f_0$描述的概率为

$$
\pi_k = \mathbb P \{ f = f_0 \mid z_k, z_{k-1}, \ldots, z_0 \}
$$

这是通过递归应用贝叶斯法则计算得出：

$$
\pi_{k+1} = \frac{ \pi_k f_0(z_{k+1})}{ \pi_k f_0(z_{k+1}) + (1-\pi_k) f_1 (z_{k+1}) },
\quad k = -1, 0, 1, \ldots
$$

在观察到$z_k, z_{k-1}, \ldots, z_0$后，决策者认为$z_{k+1}$具有概率分布

$$
f_{{\pi}_k} (v) = \pi_k f_0(v) + (1-\pi_k) f_1 (v) ,
$$

这是分布$f_0$和$f_1$的混合，其中$f_0$的权重是$f = f_0$的后验概率[^f1]。

为了说明这样的分布，让我们检查一些beta分布的混合。

具有参数 $a$ 和 $b$ 的贝塔概率分布的密度函数为

$$
f(z; a, b) = \frac{\Gamma(a+b) z^{a-1} (1-z)^{b-1}}{\Gamma(a) \Gamma(b)}
\quad \text{其中} \quad
\Gamma(t) := \int_{0}^{\infty} x^{t-1} e^{-x} dx
$$

上图的上半部分显示了两个贝塔分布。

下半部分展示了这些分布的混合，其中使用了不同的混合概率 $\pi_k$

```{code-cell} ipython3
@jit
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x**(a-1) * (1 - x)**(b-1)

f0 = lambda x: p(x, 1, 1)
f1 = lambda x: p(x, 9, 9)
grid = np.linspace(0, 1, 50)

fig, axes = plt.subplots(2, figsize=(10, 8))

axes[0].set_title("原始分布")
axes[0].plot(grid, f0(grid), lw=2, label="$f_0$")
axes[0].plot(grid, f1(grid), lw=2, label="$f_1$")

axes[1].set_title("混合分布")
for π in 0.25, 0.5, 0.75:
    y = π * f0(grid) + (1 - π) * f1(grid)
    axes[1].plot(y, lw=2, label=rf"$\pi_k$ = {π}")

for ax in axes:
    ax.legend()
    ax.set(xlabel="$z$ 值", ylabel="$z_k$ 的概率")

plt.tight_layout()
plt.show()
```

### 损失和成本

在观察到 $z_k, z_{k-1}, \ldots, z_0$ 后，决策者可以在三种不同的行动中选择：

- 他确定 $f = f_0$ 并停止获取更多 $z$ 值
- 他确定 $f = f_1$ 并停止获取更多 $z$ 值
- 他推迟现在做决定，转而选择获取一个新的 $z_{k+1}$

与这三种行动相关联，决策者可能遭受三种损失：

- 当实际上 $f=f_1$ 时，他决定 $f = f_0$ 会遭受损失 $L_0$
- 当实际上 $f=f_0$ 时，他决定 $f = f_1$ 会遭受损失 $L_1$
- 如果他推迟决定并选择获取另一个 $z$ 值，会产生成本 $c$

### 关于第一类和第二类错误的补充说明

如果我们将 $f=f_0$ 视为原假设，将 $f=f_1$ 视为备择假设，那么 $L_1$ 和 $L_0$ 是与两类统计错误相关的损失

- 第一类错误是错误地拒绝了真实的原假设（"假阳性"）
- 第二类错误是未能拒绝错误的原假设（"假阴性"）

当我们将 $f=f_0$ 作为零假设时

- 我们可以将 $L_1$ 视为第一类错误相关的损失。
- 我们可以将 $L_0$ 视为第二类错误相关的损失。

### 直观理解

在继续之前，让我们试着猜测一个最优决策规则可能是什么样的。

假设在某个特定时间点，$\pi$ 接近1。

那么我们的先验信念和目前的证据都强烈指向 $f = f_0$。

另一方面，如果 $\pi$ 接近0，那么 $f = f_1$ 就更有可能。

最后，如果 $\pi$ 在区间 $[0, 1]$ 的中间位置，那么我们就面临更多的不确定性。

这种推理表明决策规则可能如图所示

```{figure} /_static/lecture_specific/wald_friedman/wald_dec_rule.png

```

正如我们将看到的，这确实是决策规则的正确形式。

我们的问题是确定阈值 $\alpha, \beta$，这些阈值以某种方式依赖于上述参数。

在这一点上，你可能想要暂停并尝试预测参数如$c$或$L_0$对$\alpha$或$\beta$的影响。

### 贝尔曼方程

设$J(\pi)$为当前信念为$\pi$且做出最优选择的决策者的总损失。

经过思考，你会同意$J$应该满足贝尔曼方程

```{math}
:label: new1

J(\pi) =
    \min
    \left\{
        (1-\pi) L_0, \; \pi L_1, \;
        c + \mathbb E [ J (\pi') ]
    \right\}
```

其中$\pi'$是由贝叶斯法则定义的随机变量

$$
\pi' = \kappa(z', \pi) = \frac{ \pi f_0(z')}{ \pi f_0(z') + (1-\pi) f_1 (z') }
$$

当$\pi$固定且$z'$从当前最佳猜测分布中抽取时，该分布$f$定义为

$$
f_{\pi}(v) = \pi f_0(v) + (1-\pi) f_1 (v)
$$

在贝尔曼方程中，最小化是针对三个行动：

1. 接受假设$f = f_0$
1. 接受假设$f = f_1$
1. 推迟决定并再次抽样

我们可以将贝尔曼方程表示为

```{math}
:label: optdec

J(\pi) =
\min \left\{ (1-\pi) L_0, \; \pi L_1, \; h(\pi) \right\}
```

其中 $\pi \in [0,1]$ 且

- $(1-\pi) L_0$ 是接受 $f_0$ 相关的预期损失（即犯II类错误的成本）。
- $\pi L_1$ 是接受 $f_1$ 相关的预期损失（即犯I类错误的成本）。
- $h(\pi) :=  c + \mathbb E [J(\pi')]$；这是继续值；即与再抽取一个 $z$ 相关的预期成本。

最优决策规则由两个数 $\alpha, \beta \in (0,1) \times (0,1)$ 来表征，这两个数满足

$$
(1- \pi) L_0 < \min \{ \pi L_1, c + \mathbb E [J(\pi')] \}  \textrm { if } \pi \geq \alpha
$$

且

$$
\pi L_1 < \min \{ (1-\pi) L_0,  c + \mathbb E [J(\pi')] \} \textrm { if } \pi \leq \beta
$$

最优决策规则则为

$$
\begin{aligned}
\textrm { 接受 } f=f_0 \textrm{ 如果 } \pi \geq \alpha \\

\textrm { 接受 } f=f_1 \textrm{ 如果 } \pi \leq \beta \\
\textrm { 再抽取一个 }  z \textrm{ 如果 }  \beta \leq \pi \leq \alpha
\end{aligned}
$$

我们的目标是计算成本函数 $J$，并由此得出相关的临界值 $\alpha$ 和 $\beta$。

为了使计算更易于管理，使用{eq}`optdec`，我们可以将延续成本 $h(\pi)$ 写作

```{math}
:label: optdec2

\begin{aligned}
h(\pi) &= c + \mathbb E [J(\pi')] \\
&= c + \mathbb E_{\pi'} \min \{ (1 - \pi') L_0, \pi' L_1, h(\pi') \} \\
&= c + \int \min \{ (1 - \kappa(z', \pi) ) L_0, \kappa(z', \pi)  L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz'
\end{aligned}
```

等式

```{math}
:label: funceq

h(\pi) =
c + \int \min \{ (1 - \kappa(z', \pi) ) L_0, \kappa(z', \pi)  L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz'
```

是一个未知函数 $h$ 的**泛函方程**。

使用延续成本的泛函方程{eq}`funceq`，我们可以通过{eq}`optdec`的右侧推导出最优选择。

这个函数方程可以通过取一个初始猜测值并迭代来找到不动点来求解。

因此，我们用算子$Q$进行迭代，其中

$$
Q h(\pi) =
c + \int \min \{ (1 - \kappa(z', \pi) ) L_0, \kappa(z', \pi)  L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz'
$$

## 实现

首先，我们将构造一个`jitclass`来存储模型的参数

```{code-cell} ipython3
wf_data = [('a0', float64),          # beta分布的参数
           ('b0', float64),
           ('a1', float64),
           ('b1', float64),
           ('c', float64),           # 再次抽样的成本
           ('π_grid_size', int64),
           ('L0', float64),          # 当f1为真时选择f0的成本
           ('L1', float64),          # 当f0为真时选择f1的成本
           ('π_grid', float64[:]),
           ('mc_size', int64),
           ('z0', float64[:]),
           ('z1', float64[:])]
```

```{code-cell} ipython3
@jitclass(wf_data)
class WaldFriedman:

    def __init__(self,
                 c=1.25,
                 a0=1,
                 b0=1,
                 a1=3,
                 b1=1.2,
                 L0=25,
                 L1=25,
                 π_grid_size=200,
                 mc_size=1000):

        self.a0, self.b0 = a0, b0
        self.a1, self.b1 = a1, b1
        self.c, self.π_grid_size = c, π_grid_size
        self.L0, self.L1 = L0, L1
        self.π_grid = np.linspace(0, 1, π_grid_size)
        self.mc_size = mc_size

        self.z0 = np.random.beta(a0, b0, mc_size)
        self.z1 = np.random.beta(a1, b1, mc_size)

    def f0(self, x):

        return p(x, self.a0, self.b0)

    def f1(self, x):

        return p(x, self.a1, self.b1)

    def f0_rvs(self):
        return np.random.beta(self.a0, self.b0)

    def f1_rvs(self):
        return np.random.beta(self.a1, self.b1)

    def κ(self, z, π):
        """
        使用贝叶斯法则和当前观测值z更新π
        """

        f0, f1 = self.f0, self.f1

        π_f0, π_f1 = π * f0(z), (1 - π) * f1(z)
        π_new = π_f0 / (π_f0 + π_f1)

        return π_new
```

如同{doc}`最优增长讲座 <optgrowth>`中所述，为了近似连续的值函数

* 我们在有限的 $\pi$ 值网格上进行迭代。
* 当我们在网格点之间评估 $\mathbb E[J(\pi')]$ 时，我们使用线性插值。

我们在下面定义算子函数 `Q`。

```{code-cell} ipython3
@jit(nopython=True, parallel=True)
def Q(h, wf):

    c, π_grid = wf.c, wf.π_grid
    L0, L1 = wf.L0, wf.L1
    z0, z1 = wf.z0, wf.z1
    mc_size = wf.mc_size

    κ = wf.κ

    h_new = np.empty_like(π_grid)
    h_func = lambda p: np.interp(p, π_grid, h)

    for i in prange(len(π_grid)):
        π = π_grid[i]

        # Find the expected value of J by integrating over z
        integral_f0, integral_f1 = 0, 0
        for m in range(mc_size):
            π_0 = κ(z0[m], π)  # Draw z from f0 and update π
            integral_f0 += min((1 - π_0) * L0, π_0 * L1, h_func(π_0))

            π_1 = κ(z1[m], π)  # Draw z from f1 and update π
            integral_f1 += min((1 - π_1) * L0, π_1 * L1, h_func(π_1))

        integral = (π * integral_f0 + (1 - π) * integral_f1) / mc_size

        h_new[i] = c + integral

    return h_new
```

为了求解关键的函数方程，我们将使用`Q`进行迭代以找到不动点

```{code-cell} ipython3
@jit
def solve_model(wf, tol=1e-4, max_iter=1000):
    """
    计算延续成本函数

    * wf 是 WaldFriedman 的一个实例
    """

    # 设置循环
    h = np.zeros(len(wf.π_grid))
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        h_new = Q(h, wf)
        error = np.max(np.abs(h - h_new))
        i += 1
        h = h_new

    if error > tol:
        print("未能收敛！")

    return h_new
```

## 分析

让我们检查结果。

我们将使用默认参数化的分布，如下所示

```{code-cell} ipython3
wf = WaldFriedman()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wf.f0(wf.π_grid), label="$f_0$")
ax.plot(wf.f1(wf.π_grid), label="$f_1$")
ax.set(ylabel="$z_k$ 的概率", xlabel="$z_k$", title="分布")
ax.legend()

plt.show()
```

### 值函数

为了求解模型，我们将调用`solve_model`函数

```{code-cell} ipython3
h_star = solve_model(wf)    # 求解模型
```

我们还将设置一个函数来计算截断值 $\alpha$ 和 $\beta$，并在成本函数图上绘制这些值

```{code-cell} ipython3
@jit
def find_cutoff_rule(wf, h):

    """
    该函数接收一个延续成本函数，并返回在继续采样和选择特定模型之间
    转换的对应截断点
    """

    π_grid = wf.π_grid
    L0, L1 = wf.L0, wf.L1

    # 在网格上计算选择模型的所有点的成本
    payoff_f0 = (1 - π_grid) * L0
    payoff_f1 = π_grid * L1

    # 通过将这些成本与贝尔曼方程的差值可以找到截断点
    # (J 总是小于或等于 p_c_i)
    β = π_grid[np.searchsorted(
                              payoff_f1 - np.minimum(h, payoff_f0),
                              1e-10)
               - 1]
    α = π_grid[np.searchsorted(
                              np.minimum(h, payoff_f1) - payoff_f0,
                              1e-10)
               - 1]

    return (β, α)

β, α = find_cutoff_rule(wf, h_star)
cost_L0 = (1 - wf.π_grid) * wf.L0
cost_L1 = wf.π_grid * wf.L1

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(wf.π_grid, h_star, label='再次采样')
ax.plot(wf.π_grid, cost_L1, label='选择 f1')
ax.plot(wf.π_grid, cost_L0, label='选择 f0')
ax.plot(wf.π_grid,
        np.amin(np.column_stack([h_star, cost_L0, cost_L1]),axis=1),
        lw=15, alpha=0.1, color='b', label=r'$J(\pi)$')

ax.annotate(r"$\beta$", xy=(β + 0.01, 0.5), fontsize=14)
ax.annotate(r"$\alpha$", xy=(α + 0.01, 0.5), fontsize=14)

plt.vlines(β, 0, β * wf.L0, linestyle="--")
plt.vlines(α, 0, (1 - α) * wf.L1, linestyle="--")

ax.set(xlim=(0, 1), ylim=(0, 0.5 * max(wf.L0, wf.L1)), ylabel="成本",
       xlabel=r"$\pi$", title="成本函数 $J(\pi)$")

plt.legend(borderpad=1.1)
plt.show()
```

成本函数$J$在$\pi \leq \beta$时等于$\pi L_1$，在$\pi \geq \alpha$时等于$(1-\pi )L_0$。

成本函数$J(\pi)$两个线性部分的斜率由$L_1$和$-L_0$决定。

在内部区域，当后验概率分配给$f_0$处于不确定区域$\pi \in (\beta, \alpha)$时，成本函数$J$是平滑的。

决策者继续采样，直到他对模型$f_0$的概率低于$\beta$或高于$\alpha$。

### 模拟

下图显示了决策过程的500次模拟结果。

左图是**停止时间**的直方图，即做出决策所需的$z_k$抽样次数。

平均抽样次数约为6.6。

右图是在停止时间点做出正确决策的比例。

在这种情况下，决策者80%的时间做出了正确的决策。

```{code-cell} ipython3
def simulate(wf, true_dist, h_star, π_0=0.5):

    """
    This function takes an initial condition and simulates until it
    stops (when a decision is made)
    """

    f0, f1 = wf.f0, wf.f1
    f0_rvs, f1_rvs = wf.f0_rvs, wf.f1_rvs
    π_grid = wf.π_grid
    κ = wf.κ

    if true_dist == "f0":
        f, f_rvs = wf.f0, wf.f0_rvs
    elif true_dist == "f1":
        f, f_rvs = wf.f1, wf.f1_rvs

    # Find cutoffs
    β, α = find_cutoff_rule(wf, h_star)

    # Initialize a couple of useful variables
    decision_made = False
    π = π_0
    t = 0

    while decision_made is False:
        # Maybe should specify which distribution is correct one so that
        # the draws come from the "right" distribution
        z = f_rvs()
        t = t + 1
        π = κ(z, π)
        if π < β:
            decision_made = True
            decision = 1
        elif π > α:
            decision_made = True
            decision = 0

    if true_dist == "f0":
        if decision == 0:
            correct = True
        else:
            correct = False

    elif true_dist == "f1":
        if decision == 1:
            correct = True
        else:
            correct = False

    return correct, π, t

def stopping_dist(wf, h_star, ndraws=250, true_dist="f0"):

    """
    Simulates repeatedly to get distributions of time needed to make a
    decision and how often they are correct
    """

    tdist = np.empty(ndraws, int)
    cdist = np.empty(ndraws, bool)

    for i in range(ndraws):
        correct, π, t = simulate(wf, true_dist, h_star)
        tdist[i] = t
        cdist[i] = correct

    return cdist, tdist

def simulation_plot(wf):
    h_star = solve_model(wf)
    ndraws = 500
    cdist, tdist = stopping_dist(wf, h_star, ndraws)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].hist(tdist, bins=np.max(tdist))
    ax[0].set_title(f"Stopping times over {ndraws} replications")
    ax[0].set(xlabel="time", ylabel="number of stops")
    ax[0].annotate(f"mean = {np.mean(tdist)}", xy=(max(tdist) / 2,
                   max(np.histogram(tdist, bins=max(tdist))[0]) / 2))

    ax[1].hist(cdist.astype(int), bins=2)
    ax[1].set_title(f"Correct decisions over {ndraws} replications")
    ax[1].annotate(f"% correct = {np.mean(cdist)}",
                   xy=(0.05, ndraws / 2))

    plt.show()

simulation_plot(wf)
```

### 比较静态分析

现在让我们来看下面的练习。

我们将获取额外观测值的成本提高一倍。

在查看结果之前，请思考会发生什么：

- 决策者的判断正确率会提高还是降低？
- 他会更早还是更晚做出决定？

```{code-cell} ipython3
wf = WaldFriedman(c=2.5)
simulation_plot(wf)
```

每次抽样成本的增加导致决策者在做出决定前减少抽样次数。

由于他用更少的抽样做决定，正确判断的比例下降。

这导致他在对两个模型赋予相同权重时产生更高的预期损失。

### 笔记本实现

为了便于比较静态分析，我们提供了一个[Jupyter笔记本](https://nbviewer.org/github/QuantEcon/lecture-python.notebooks/blob/main/wald_friedman.ipynb)，它可以生成相同的图表，但带有滑块控件。

使用这些滑块，你可以调整参数并立即观察

* 当我们增加分段线性近似中的网格点数时，对犹豫不决的中间范围内价值函数平滑度的影响。
* 不同成本参数$L_0, L_1, c$、两个beta分布$f_0$和$f_1$的参数，以及在价值函数分段连续近似中使用的点数和线性函数$m$的设置对结果的影响。

* 从 $f_0$ 进行的各种模拟以及相关的决策等待时间分布。
* 正确和错误决策的相关直方图。

## 与奈曼-皮尔逊公式的比较

出于几个原因，有必要描述海军上尉G. S. Schuyler被要求使用的测试背后的理论，这也是他向Milton Friedman和Allan Wallis提出他的推测，认为存在更好的实用程序的原因。

显然，海军告诉Schuyler上尉使用他们所知的最先进的奈曼-皮尔逊检验。

我们将依据Abraham Wald的{cite}`Wald47`对奈曼-皮尔逊理论的优雅总结。

就我们的目的而言，需要注意设置中的以下特征：

- 假设*固定*样本量 $n$
- 应用大数定律，在备择概率模型的条件下，解释奈曼-皮尔逊理论中定义的概率 $\alpha$ 和 $\beta$

回顾上述顺序分析公式中，

- 样本量 $n$ 不是固定的，而是一个待选择的对象；从技术上讲，$n$ 是一个随机变量。
- 参数 $\beta$ 和 $\alpha$ 表征用于确定随机变量 $n$ 的截止规则。
- 大数定律在顺序构造中并未出现。

在**顺序分析**{cite}`Wald47`第一章中，Abraham Wald总结了Neyman-Pearson的假设检验方法。

Wald将问题描述为对部分已知的概率分布做出决策。

（为了提出一个明确的问题，你必须假设*某些东西*是已知的 -- 通常，*某些东西*意味着*很多东西*）

通过限制未知的内容，Wald使用以下简单结构来说明主要思想：

- 决策者想要决定两个分布 $f_0$、$f_1$ 中的哪一个支配着独立同分布随机变量 $z$。

- 零假设 $H_0$ 是指 $f_0$ 支配数据的陈述。
- 备择假设 $H_1$ 是指 $f_1$ 支配数据的陈述。
- 问题在于基于固定数量 $n$ 个独立观测值 $z_1, z_2, \ldots, z_n$（来自随机变量 $z$）的样本，设计并分析一个针对零假设 $H_0$ 和备择假设 $H_1$ 的假设检验。

引用 Abraham Wald 的话：

> 导致接受或拒绝[零]假设的检验程序，简单来说就是一个规则，它为每个可能的大小为 $n$ 的样本指定是否应该基于该样本接受或拒绝[零]假设。这也可以表述如下：检验程序就是将所有可能的大小为 $n$ 的样本总体划分为两个互斥的部分，比如说第1部分和第2部分，同时应用规则，如果观察到的样本属于

> 包含在第2部分中。第1部分也被称为临界区域。由于
> 第2部分是所有大小为$n$的样本中不包含在第1部分的
> 总体，第2部分由第1部分唯一确定。因此，
> 选择检验程序等同于确定临界区域。

让我们继续听瓦尔德的说法：

> 关于选择临界区域的依据，奈曼和皮尔逊提出了以下
> 考虑：在接受或拒绝$H_0$时，我们可能会犯两种
> 错误。当$H_0$为真而我们拒绝它时，我们犯第一类
> 错误；当$H_1$为真而我们接受$H_0$时，我们犯
> 第二类错误。在选定特定的临界区域$W$后，
> 犯第一类错误的概率和犯第二类错误的概率就
> 被唯一确定了。犯第一类错误的概率等于由

> 假设 $H_0$ 为真，观察到的样本将落在临界区域 $W$ 内。
> 犯第二类错误的概率等于在假设 $H_1$ 为真的情况下，
> 概率落在临界区域 $W$ 之外的概率。对于任何给定的
> 临界区域 $W$，我们用 $\alpha$ 表示第一类错误的概率，
> 用 $\beta$ 表示第二类错误的概率。

让我们仔细听听Wald如何运用大数定律来解释 $\alpha$ 和 $\beta$：

> 概率 $\alpha$ 和 $\beta$ 有以下重要的实际解释：
> 假设我们抽取大量规模为 $n$ 的样本。设 $M$ 为
> 抽取的样本总数。假设对于这 $M$ 个样本中的每一个，
> 如果样本落在 $W$ 内则拒绝 $H_0$，如果样本落在
> $W$ 外则接受 $H_0$。通过这种方式，我们做出了 $M$ 个
> 拒绝或

> 接受。这些陈述中的一些通常是错误的。如果
> $H_0$为真且$M$很大，那么错误陈述的比例（即错误陈述的数量
> 除以$M$）约为$\alpha$的概率接近于$1$（即几乎是确定的）。如果
> $H_1$为真，错误陈述的比例约为$\beta$的概率接近于$1$。
> 因此，我们可以说从长远来看[这里Wald通过令$M \rightarrow \infty$应用了大数定律（这是我们的注释，
> 不是Wald的）]，如果$H_0$为真，错误陈述的比例将是
> $\alpha$，如果$H_1$为真，则为$\beta$。

量$\alpha$被称为临界区域的*大小*，
而量$1-\beta$被称为临界区域的*检验力*。

Wald指出

> 如果一个临界区域$W$具有更小的$\alpha$和$\beta$值，
> 那么它比另一个更可取。虽然

> 通过适当选择临界区域 $W$，$\alpha$ 或 $\beta$ 中的任一个都可以被任意缩小，
> 但在固定样本量 $n$ 的情况下，不可能同时使 $\alpha$ 和 $\beta$ 都任意小。

Wald 总结了 Neyman 和 Pearson 的设置如下：

> Neyman 和 Pearson 证明，由满足以下不等式的所有样本
> $(z_1, z_2, \ldots, z_n)$ 构成的区域
>
> $$
  \frac{ f_1(z_1) \cdots f_1(z_n)}{f_0(z_1) \cdots f_0(z_n)} \geq k
  $$
>
> 是检验假设 $H_0$ 对立假设 $H_1$ 的最优势临界区域。
> 右侧的常数项 $k$ 的选择使得该区域具有所需的显著性水平 $\alpha$。

Wald 接着讨论了 Neyman 和 Pearson 的*一致最优势*检验的概念。

以下是 Wald 引入序贯检验概念的方式：

> 在任何阶段都给出一个规则来做出以下三个决定中的一个：

> 实验（在第m次试验中，m为整数值）：(1)接受假设H，(2)拒绝假设H，(3)
> 通过进行额外观察来继续实验。因此，这样的检验程序是按顺序进行的。基于第一次
> 观察，做出上述决策之一。如果做出第一个或第二个决策，过程就终止。如果做出第
> 三个决策，则进行第二次试验。同样，基于前两次观察，做出三个决策中的一个。如
> 果做出第三个决策，则进行第三次试验，依此类推。这个过程持续进行，直到做出第
> 一个或第二个决策为止。这种检验程序所需的观察次数n是一个随机变量，因为n的值
> 取决于观察的结果。

[^f1]: 决策者的行为就像他相信随机变量序列

$[z_{0}, z_{1}, \ldots]$ 是*可交换的*。参见[可交换性和贝叶斯更新](https://python.quantecon.org/exchangeable.html)和
{cite}`Kreps88`第11章对可交换性的讨论。

## 后续内容

我们将在以下讲座中深入探讨这里使用的一些概念：

* {doc}`本讲座 <exchangeable>` 讨论了合理化统计学习的关键概念**可交换性**
* {doc}`本讲座 <likelihood_ratio_process>` 描述了**似然比过程**及其在频率派和贝叶斯统计理论中的作用
* {doc}`本讲座 <likelihood_bayes>` 讨论了似然比过程在**贝叶斯学习**中的作用
* {doc}`本讲座 <navy_captain>` 回到本讲座的主题，研究海军命令舰长使用的（频率派）决策规则是否可以预期会比Abraham Wald设计的序贯规则更好或更差

