---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: 贝叶斯与频率主义决策规则的比较
  headings:
    Overview: 概述
    Setup: 设置
    Frequentist Decision Rule: 频率主义决策规则
    Bayesian Decision Rule: 贝叶斯决策规则
    Was the Navy Captain’s Hunch Correct?: 海军上尉的直觉是否正确？
    More Details: 更多细节
    Distribution of Bayesian Decision Rule’s Time to Decide: 贝叶斯决策规则的决策时间分布
    Probability of Making Correct Decision: 做出正确决策的概率
    Distribution of Likelihood Ratios at Neyman-Pearson's $t$: Neyman-Pearson的$t$处的似然比分布
    Exercises: 练习
---

(bayesian_vs_frequentist__v1)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 贝叶斯与频率主义决策规则的比较

```{contents} 目录
:depth: 2
```


```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from numba import jit, prange, float64, int64
from numba.experimental import jitclass
from math import gamma
from scipy.integrate import quad
```

## 概述

本讲座延续了以下讲座中提出的观点：

* {doc}`令米尔顿·弗里德曼困惑的问题 <wald_friedman>`
* {doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>`
* {doc}`可交换性和贝叶斯更新 <exchangeable>`
* {doc}`似然比过程 <likelihood_ratio_process>`
* {doc}`似然比过程与贝叶斯学习 <likelihood_bayes>`

{doc}`令米尔顿·弗里德曼困惑的问题 <wald_friedman>` 描述了二战期间一位海军上尉向米尔顿·弗里德曼提出的问题。

海军告诉这位上尉，要使用一个决策规则，通过试射测试弹药来比较两种军械装置的设计。

具体来说，海军命令上尉使用一个**频率主义决策规则**的实例。

上尉怀疑这个规则是否是一个好规则。

米尔顿·弗里德曼认识到上尉的推测提出了一个具有挑战性的统计问题，他和哥伦比亚大学美国政府统计研究小组的其他成员随后试图解决这个问题。

该小组的成员之一，伟大的数学家和经济学家亚伯拉罕·瓦尔德很快就解决了这个问题。

阐述这个问题的一个好方法是使用一些贝叶斯统计中的思想，这些思想我们在 {doc}`exchangeable` 以及 {doc}`likelihood_bayes` 中关于 {ref}`似然比过程与贝叶斯法则 <lb_bayes_law>` 的章节中有所描述，后者阐述了贝叶斯更新和似然比过程之间的联系。

本讲座使用 Python 生成模拟，评估在海军上尉所质疑的 Neyman-Pearson**频率主义**程序，以及 {doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>` 中描述的**贝叶斯**决策规则下的预期损失。

我们选择频率主义规则的样本量和截断值，以在贝叶斯主义者所使用的先验概率和损失下，使预期损失最小化。

这样一来，频率主义规则就成为一个样本量预先固定的贝叶斯规则。

而贝叶斯规则则是按顺序选择其样本量的。

固定样本量的规则是一种特殊的序贯规则，因此最优序贯规则的预期损失不会大于最优固定样本规则的预期损失。

{cite:t}`WaldWolfowitz1948` 证明了，在给定第一类和第二类错误概率的所有检验中，瓦尔德的序贯概率比检验能够使两种假设下所需抽取样本数量的期望值最小化。

{cite:t}`ArrowBlackwellGirshick1949` 展示了如何通过逆向归纳法计算序贯决策问题的贝叶斯解。

我们的模拟说明了这些定理，并衡量了海军上尉的直觉究竟有多大价值。

## 设置

为了形式化困扰海军上尉的问题，我们考虑一个包含以下部分的设定。

- 每个时期决策者都会抽取一个非负随机变量
  $Z$。他知道可能存在两种概率分布，$f_{0}$和$f_{1}$，并且无论是哪种分布都会随时间保持不变。决策者认为在时间开始之前，自然界一次性地选择了$f_{0}$或$f_1$，且选择$f_0$的概率为$\pi^{*}$。
- 决策者从自然界选择的分布中观察到样本$\left\{ z_{i}\right\} _{i=1}^{t}$。

决策者想要确定究竟是哪个分布在支配$Z$。

他担心两种类型的错误以及它们将给他带来的损失。

- **第一类错误**造成的损失$\bar L_{1}$，即当实际分布为$f=f_{0}$时却判定$f=f_{1}$
- **第二类错误**造成的损失$\bar L_{0}$，即当实际分布为$f=f_{1}$时却判定$f=f_{0}$

决策者需要支付成本$c$来获取另一个$z$。

```{note}
本讲座用$\pi$表示$f_0$的概率，并将似然比定义为$L = f_0/f_1$。

按照这一约定，较高的$\pi$和$L$值支持$f_0$，贝叶斯决策者在$\pi \geq A$时接受$f_0$。

{doc}`wald_friedman_2` 用$\pi$表示$f_1$的概率；其决策者在$\pi \geq A$时接受$f_1$。

{doc}`wald_friedman` 将似然比定义为$f_1/f_0$。

我们按照本讲座的约定重写了从 {doc}`wald_friedman_2` 借鉴来的代码：在贝叶斯更新$\kappa$、停止决策的预期损失以及贝尔曼算子`Q`中，$\pi$与$1-\pi$互换了位置。
```

我们主要借鉴了quantecon讲座{doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>`中的参数，不过我们将$\bar L_{0}$和$\bar L_{1}$都从$25$增加到了$100$，以鼓励贝叶斯决策规则在做决定前进行更多次抽样。

我们将每次额外抽样的成本$c$设为$1.25$。

我们将概率分布$f_{0}$和$f_{1}$设定为贝塔分布，其中$a_{0}=b_{0}=1$，$a_{1}=3$，且$b_{1}=1.2$。

下面是设置这些对象的Python代码。

```{code-cell} ipython3
@jit
def p(x, a, b):
    "贝塔分布。"

    r = gamma(a + b) / (gamma(a) * gamma(b))

    return r * x**(a-1) * (1 - x)**(b-1)
```

我们首先定义一个`jitclass`，用于存储参数和函数，这些参数和函数将用于解决贝叶斯派和频率派海军上尉的问题。

```{code-cell} ipython3
wf_data = [
    ('c', float64),           # 抽取另一个z的成本
    ('a0', float64),          # beta分布的参数
    ('b0', float64),
    ('a1', float64),
    ('b1', float64),
    ('L0', float64),          # 当f1为真时选择f0的成本
    ('L1', float64),          # 当f0为真时选择f1的成本
    ('π_grid', float64[:]),   # 信念π的网格
    ('π_grid_size', int64),
    ('mc_size', int64),       # 蒙特卡洛模拟的规模
    ('z0', float64[:]),       # 随机值序列
    ('z1', float64[:])        # 随机值序列
]
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
                 L0=100,
                 L1=100,
                 π_grid_size=200,
                 mc_size=1000):

        self.c, self.π_grid_size = c, π_grid_size
        self.a0, self.b0, self.a1, self.b1 = a0, b0, a1, b1
        self.L0, self.L1 = L0, L1
        self.π_grid = np.linspace(0, 1, π_grid_size)
        self.mc_size = mc_size

        self.z0 = np.random.beta(a0, b0, mc_size)
        self.z1 = np.random.beta(a1, b1, mc_size)

    def f0(self, x):

        return p(x, self.a0, self.b0)

    def f1(self, x):

        return p(x, self.a1, self.b1)

    def κ(self, z, π):
        """
        使用贝叶斯法则和当前观测值z更新π
        """

        a0, b0, a1, b1 = self.a0, self.b0, self.a1, self.b1

        π_f0, π_f1 = π * p(z, a0, b0), (1 - π) * p(z, a1, b1)
        π_new = π_f0 / (π_f0 + π_f1)

        return π_new
```

`WaldFriedman`的构造函数使用numba的随机数生成器从每个密度中抽取`mc_size`个$z$的蒙特卡洛值。

从Python调用`np.random.seed`并不会为numba的生成器设定种子。

我们在一个经过jit编译的函数内部为其设定种子。

```{code-cell} ipython3
@jit
def seed_numba(seed):
    "为numba编译代码所使用的随机数生成器设定种子。"
    np.random.seed(seed)

seed_numba(1234)
wf = WaldFriedman()

grid = np.linspace(0, 1, 50)

plt.figure()

plt.title("两个分布")
plt.plot(grid, wf.f0(grid), lw=2, label="$f_0$")
plt.plot(grid, wf.f1(grid), lw=2, label="$f_1$")

plt.legend()
plt.xlabel("$z$ 值")
plt.ylabel("$z_k$ 的密度")

plt.tight_layout()
plt.show()
```

上面，我们绘制了两个可能的概率密度 $f_0$ 和 $f_1$

## 频率主义决策规则

海军告诉上尉要使用Neyman-Pearson似然比决策规则。

该决策规则的特征是由：

 * 样本量 $t$，以及
 * 似然比的临界值 $d$

令
$L\left(z^{t}\right)=\prod_{i=1}^{t}\frac{f_{0}\left(z_{i}\right)}{f_{1}\left(z_{i}\right)}$
为观察序列 $\left\{ z_{i}\right\} _{i=1}^{t}$ 的似然比。

与样本量 $t$ 相关的决策规则是：

- 如果似然比大于 $d$，则判定 $f_0$ 是分布
- 如果似然比小于 $d$，则判定 $f_1$ 是分布

为此，我们想要计算使用这一规则所产生的预期损失，其中我们借用了 {doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>` 中的损失参数 $\bar L_0$ 和 $\bar L_1$。

令零假设和备择假设为

- 零假设：$H_{0}$：$f=f_{0}$，
- 备择假设 $H_{1}$：$f=f_{1}$。

在给定样本量 $t$ 和临界值 $d$ 的情况下，在上述模型下，
总损失的数学期望为

```{math}
:label: val1

\begin{aligned}
\bar{V}_{fre}\left(t,d\right)=ct+\pi^{*}PFA\times \bar L_{1}+\left(1-\pi^{*}\right)\left(1-PD\right)\times \bar L_{0}
\end{aligned}
```

$$
\begin{aligned}
\textrm{其中} \quad PFA & =\Pr\left\{ L\left(z^{t}\right)<d\mid q=f_{0}\right\} \\
PD & =\Pr\left\{ L\left(z^{t}\right)<d\mid q=f_{1}\right\}
\end{aligned}
$$

这里

- $PFA$ 表示**虚警**的概率，即
  当 $H_0$ 为真时却拒绝它
- $PD$ 表示**检测**的概率，即
  当 $H_1$ 为真时拒绝 $H_0$（即检验的功效）；$1 - PD$ 为第二类错误的概率

对于给定的样本量 $t$，$\left(PFA,PD\right)$ 对
位于**接收者操作特征曲线**上。
  * 通过选择 $d$，我们为给定的 $t$ 选择曲线上的一个特定 $\left(PFA,PD\right)$ 对

要查看一些接收者操作特征曲线，请参阅本课程
{doc}`似然比过程 <likelihood_ratio_process>`。

要数值求解 $\bar{V}_{fre}\left(t,d\right)$，我们首先
模拟当$f_0$或$f_1$生成数据时的$z$序列。

让我们绘制与$f_0$和$f_1$相关的经验分布（即直方图）。

```{code-cell} ipython3
N = 10000
T = 100
```

```{code-cell} ipython3
rng = np.random.default_rng(1234)

z0_arr = rng.beta(wf.a0, wf.b0, (N, T))
z1_arr = rng.beta(wf.a1, wf.b1, (N, T))
```

```{code-cell} ipython3
plt.hist(z0_arr.flatten(), bins=50, alpha=0.4, label='f0')
plt.hist(z1_arr.flatten(), bins=50, alpha=0.4, label='f1')
plt.legend()
plt.show()
```

我们可以使用模拟样本计算似然比序列。

```{code-cell} ipython3
l = lambda z: wf.f0(z) / wf.f1(z)
```

```{code-cell} ipython3
l0_arr = l(z0_arr)
l1_arr = l(z1_arr)

L0_arr = np.cumprod(l0_arr, 1)
L1_arr = np.cumprod(l1_arr, 1)
```

`L0_arr` 和 `L1_arr` 的第 `t-1` 列保存的是经过 $t$ 次抽取后的似然比 $L(z^t)$。

有了似然比的经验分布后，我们可以通过列举每个样本量 $t$ 下的 $\left(PFA,PD\right)$ 对来绘制**接收者操作特征曲线**。

```{code-cell} ipython3
PFA = np.arange(0, 100, 1)

for t in range(1, 15, 4):
    percentile = np.percentile(L0_arr[:, t-1], PFA)
    PD = [np.sum(L1_arr[:, t-1] < p) / N for p in percentile]

    plt.plot(PFA / 100, PD, label=f"t={t}")

plt.scatter(0, 1, label="perfect detection")
plt.plot([0, 1], [0, 1], color='k', ls='--', label="random detection")

plt.arrow(0.5, 0.5, -0.15, 0.15, head_width=0.03)
plt.text(0.35, 0.7, "better")
plt.xlabel("虚警概率")
plt.ylabel("检测概率")
plt.legend()
plt.title("接收者操作特征曲线")
plt.show()
```

我们可以通过选择 $\left(t,d\right)$ 来最小化方程 {eq}`val1` 中给出的预期总损失。

这样做会得到预期损失

$$
\bar{V}_{fre}=\min_{t,d}\bar{V}_{fre}\left(t,d\right).
$$

我们让 $t$ 取遍 $1, 2, \ldots, T$。[^t0]

我们首先考虑
$\pi^{*}=\Pr\left\{ \text{自然选择}f_{0}\right\} =0.5$
的情况。

我们可以通过两个步骤来解决这个最小化问题。

首先，我们固定$t$并找到最优截断值$d$，
从而得到最小值$\bar{V}_{fre}\left(t\right)$。

对于每个 $t$，临界值

```{math}
:label: eq_dstar

d^{*}=\frac{\left(1-\pi^{*}\right)\bar L_{0}}{\pi^{*}\bar L_{1}}
```

可以最小化 $\bar{V}_{fre}\left(t,d\right)$。

"若 $L(z^t) < d^*$ 则判定为 $f_1$"这条规则，在经过 $t$ 次抽取后，选择的是后验期望损失较小的那个决策。

因此，当 $d = d^*$ 时，频率派的规则实际上就是一条样本量事先固定的贝叶斯规则。

{ref}`nc_ex1` 要求你推导 {eq}`eq_dstar` 并通过数值方法进行验证。

其次，我们对 $t$ 最小化 $\bar{V}_{fre}\left(t\right) = \bar{V}_{fre}\left(t, d^*\right)$。

以下是执行该操作并绘制有用图表的Python代码。

```{code-cell} ipython3
@jit
def V_fre_d_t(d, t, L0_arr, L1_arr, π_star, wf):

    N = L0_arr.shape[0]

    PFA = np.sum(L0_arr[:, t-1] < d) / N
    PD = np.sum(L1_arr[:, t-1] < d) / N

    V = π_star * PFA * wf.L1 + (1 - π_star) * (1 - PD) * wf.L0

    return V
```

```{code-cell} ipython3
def d_star(π_star, wf):
    "对于任意t，使期望损失最小化的L = f0/f1的临界值。"

    return (1 - π_star) * wf.L0 / (π_star * wf.L1)


def V_fre_t(t, L0_arr, L1_arr, π_star, wf):

    d = d_star(π_star, wf)
    V = V_fre_d_t(d, t, L0_arr, L1_arr, π_star, wf)

    PFA = np.mean(L0_arr[:, t-1] < d)
    PD = np.mean(L1_arr[:, t-1] < d)

    return V, PFA, PD
```

```{code-cell} ipython3
def compute_V_fre(L0_arr, L1_arr, π_star, wf):

    T = L0_arr.shape[1]

    V_fre_arr = np.empty(T)
    PFA_arr = np.empty(T)
    PD_arr = np.empty(T)

    for t in range(1, T+1):
        V, PFA, PD = V_fre_t(t, L0_arr, L1_arr, π_star, wf)
        V_fre_arr[t-1] = wf.c * t + V
        PFA_arr[t-1] = PFA
        PD_arr[t-1] = PD

    return V_fre_arr, PFA_arr, PD_arr
```

```{code-cell} ipython3
π_star = 0.5
V_fre_arr, PFA_arr, PD_arr = compute_V_fre(L0_arr, L1_arr, π_star, wf)

plt.plot(range(1, T+1), V_fre_arr, label=r'$\min_{d} \overline{V}_{fre}(t,d)$')
plt.xlabel('t')
plt.title(r'$\pi^*=0.5$')
plt.legend()
plt.show()
```

```{code-cell} ipython3
t_optimal = np.argmin(V_fre_arr) + 1
```

上图说明了对$t$进行最小化如何告诉频率派要抽取$t_{\rm optimal}$个观测值然后做出决定。

现在让我们改变 $\pi^{*}$ 的值，观察决策规则如何变化。

```{code-cell} ipython3
n_π = 20
π_star_arr = np.linspace(0.1, 0.9, n_π)

V_fre_bar_arr = np.empty(n_π)
t_optimal_arr = np.empty(n_π)
PFA_optimal_arr = np.empty(n_π)
PD_optimal_arr = np.empty(n_π)

for i, π_star in enumerate(π_star_arr):
    V_fre_arr, PFA_arr, PD_arr = compute_V_fre(L0_arr, L1_arr, π_star, wf)
    t_idx = np.argmin(V_fre_arr)

    V_fre_bar_arr[i] = V_fre_arr[t_idx]
    t_optimal_arr[i] = t_idx + 1
    PFA_optimal_arr[i] = PFA_arr[t_idx]
    PD_optimal_arr[i] = PD_arr[t_idx]
```

```{code-cell} ipython3
plt.plot(π_star_arr, V_fre_bar_arr)
plt.xlabel(r'$\pi^*$')
plt.title(r'$\overline{V}_{fre}$')

plt.show()

V_no_data = np.minimum(π_star_arr * wf.L1, (1 - π_star_arr) * wf.L0)
print("在以下π*值下，不经抽样直接做决定优于任何t ≥ 1的情形：",
      np.round(π_star_arr[V_no_data < V_fre_bar_arr], 2))
```

下图展示了当 $\pi^{*}$ 变化时，最优样本量 $t$ 和目标 $\left(PFA,PD\right)$ 如何变化。

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(π_star_arr, t_optimal_arr)
axs[0].set_xlabel(r'$\pi^*$')
axs[0].set_title(r'给定$\pi^*$的最优样本量')

axs[1].plot(π_star_arr, PFA_optimal_arr, label=r'$PFA^*(\pi^*)$')
axs[1].plot(π_star_arr, PD_optimal_arr, label=r'$PD^*(\pi^*)$')
axs[1].set_xlabel(r'$\pi^*$')
axs[1].legend()
axs[1].set_title(r'给定$\pi^*$的最优PFA和PD')

plt.show()
```

## 贝叶斯决策规则

在 {doc}`令米尔顿·弗里德曼困惑的问题 <wald_friedman>` 中，我们了解到亚伯拉罕·瓦尔德如何证实了海军上尉的直觉，即存在一个更好的决策规则。

在 {doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>` 中，我们提出了一个贝叶斯程序，该程序通过比较贝叶斯后验概率$\pi$与称为$A$和$B$的临界概率来做出决策。

为了继续，我们从 quantecon 讲座 {doc}`弗里德曼与瓦尔德问题的贝叶斯表述 <wald_friedman_2>` 中借用一些Python代码，用于计算$A$和$B$的最优值。

我们按照本讲座的约定重写该代码，即$\pi$是$f_0$的概率。

```{code-cell} ipython3
@jit(parallel=True)
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

```{code-cell} ipython3
@jit
def solve_model(wf, tol=1e-4, max_iter=1000):
    """
    计算延续值函数

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

```{code-cell} ipython3
h_star = solve_model(wf)
```

```{code-cell} ipython3
def find_cutoff_rule(wf, h):
    """
    此函数接收一个延续值函数并返回临界点 (B, A)：
    当 π ≤ B 时决策者接受 f1，当 π ≥ A 时接受 f0，
    当 B < π < A 时抽取另一个 z
    """

    π_grid = wf.π_grid

    # 接受每个模型的期望损失
    payoff_f0 = (1 - π_grid) * wf.L0
    payoff_f1 = π_grid * wf.L1

    # B 是接受 f1 为最优选择的最大 π；
    # A 是接受 f0 为最优选择的最小 π
    B = π_grid[payoff_f1 <= np.minimum(payoff_f0, h)].max()
    A = π_grid[payoff_f0 <= np.minimum(payoff_f1, h)].min()

    return (B, A)

B, A = find_cutoff_rule(wf, h_star)
print(f"B = {B:.3f}, A = {A:.3f}")

cost_L0 = (1 - wf.π_grid) * wf.L0
cost_L1 = wf.π_grid * wf.L1

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(wf.π_grid, h_star, label='延续值')
ax.plot(wf.π_grid, cost_L1, label='选择f1')
ax.plot(wf.π_grid, cost_L0, label='选择f0')
ax.plot(wf.π_grid,
        np.amin(np.column_stack([h_star, cost_L0, cost_L1]),axis=1),
        lw=15, alpha=0.1, color='b', label='最小成本')

ax.annotate(r"$B$", xy=(B + 0.01, 0.5), fontsize=14)
ax.annotate(r"$A$", xy=(A + 0.01, 0.5), fontsize=14)

plt.vlines(B, 0, B * wf.L1, linestyle="--")
plt.vlines(A, 0, (1 - A) * wf.L0, linestyle="--")

ax.set(xlim=(0, 1), ylim=(0, 0.5 * max(wf.L0, wf.L1)), ylabel="成本",
       xlabel=r"$\pi$", title="值函数")

plt.legend(borderpad=1.1)
plt.show()
```

上图描绘了价值函数对决策者贝叶斯后验概率的关系。

图中还显示了临界概率 $A$ 和 $B$。

贝叶斯决策规则是：

- 当 $\pi \geq A$ 时接受 $H_0$
- 当 $\pi \leq B$ 时接受 $H_1$
- 当 $B < \pi < A$ 时延迟决定并抽取另一个 $z$

在这种情况下，我们可以计算两个“客观”损失函数，分别基于确知自然选择了 $f_{0}$ 或 $f_{1}$ 的条件。

1. 在 $f_{0}$ 下，

   $$
   V^{0}\left(\pi\right)=\begin{cases}
   0 & \text{if }A \leq\pi,\\
   c+E_{f_0}V^{0}\left(\pi^{\prime}\right) & \text{if }B<\pi< A,\\
   \bar L_{1} & \text{if }\pi\leq B.
   \end{cases}
   $$

1. 在 $f_{1}$ 下

   $$
   V^{1}\left(\pi\right)=\begin{cases}
   \bar L_{0} & \text{if }A \leq\pi,\\
   c+E_{f_1}V^{1}\left(\pi^{\prime}\right) & \text{if }B <\pi<A,\\
   0 & \text{if }\pi\leq B.
   \end{cases}
   $$

其中
$\pi^{\prime}=\frac{\pi f_{0}\left(z^{\prime}\right)}{\pi f_{0}\left(z^{\prime}\right)+\left(1-\pi\right)f_{1}\left(z^{\prime}\right)}$，
且 $E_{f_i}$ 是对从 $f_i$ 中抽取的 $z^{\prime}$ 求积分。

给定先验概率 $\pi_{0}$，贝叶斯方法的期望损失为

$$
\bar{V}_{Bayes}\left(\pi_{0}\right)=\pi^{*}V^{0}\left(\pi_{0}\right)+\left(1-\pi^{*}\right)V^{1}\left(\pi_{0}\right).
$$

下面我们编写一些 Python 代码来数值计算 $V^{0}\left(\pi\right)$ 和 $V^{1}\left(\pi\right)$。

该代码在延续区域 $(B, A)$ 内的网格点上对两个函数方程进行迭代。

它用存储在 `wf` 中的 `mc_size` 次蒙特卡洛抽取的平均值来替代每个期望值。

当 $\pi^{\prime}$ 离开 $(B, A)$ 时，代码直接使用终端损失，并仅在 $(B, A)$ 内部的网格点之间进行插值。

```{code-cell} ipython3
@jit(parallel=True)
def V_q(wf, A, B, flag, tol=1e-5, max_iter=10_000):
    """
    当 f0 (flag=0) 或 f1 (flag=1) 生成数据时，
    贝叶斯临界规则 (B, A) 的期望损失
    """

    π_grid = wf.π_grid

    if flag == 0:
        z_arr = wf.z0
        loss_f0, loss_f1 = 0.0, wf.L1   # 接受 f0、f1 的损失
    else:
        z_arr = wf.z1
        loss_f0, loss_f1 = wf.L0, 0.0

    m = len(z_arr)

    # 延续区域之外的终端损失
    V = np.where(π_grid >= A, loss_f0, np.where(π_grid <= B, loss_f1, 0.0))

    cont = (π_grid > B) & (π_grid < A)
    π_cont = π_grid[cont]
    n_cont = len(π_cont)
    if n_cont == 0:
        return V

    V_cont = np.zeros(n_cont)
    for it in range(max_iter):
        V_old = V_cont.copy()

        for i in prange(n_cont):
            total = 0.0
            for j in range(m):
                π_next = wf.κ(z_arr[j], π_cont[i])
                if π_next >= A:
                    total += loss_f0
                elif π_next <= B:
                    total += loss_f1
                else:
                    total += np.interp(π_next, π_cont, V_old)
            V_cont[i] = wf.c + total / m

        if np.max(np.abs(V_cont - V_old)) < tol:
            break

    V[cont] = V_cont

    return V
```

```{code-cell} ipython3
V0 = V_q(wf, A, B, 0)
V1 = V_q(wf, A, B, 1)

plt.plot(wf.π_grid, V0, label='$V^0$')
plt.plot(wf.π_grid, V1, label='$V^1$')
plt.vlines(B, 0, wf.L0, linestyle='--')
plt.text(B+0.01, wf.L0/2, 'B')
plt.vlines(A, 0, wf.L0, linestyle='--')
plt.text(A+0.01, wf.L0/2, 'A')
plt.xlabel(r'$\pi$')
plt.title(r'目标值函数 $V(\pi)$')
plt.legend()
plt.show()
```

给定一个假设值
$\pi^{*}=\Pr\left\{ \text{自然选择 }f_{0}\right\}$，我们就可以
计算 $\bar{V}_{Bayes}\left(\pi_{0}\right)$。

然后我们可以确定一个初始贝叶斯先验概率 $\pi_{0}^{*}$，使其
最小化这个期望损失的客观概念。

一个定理告诉我们 $\pi_{0}^{*}=\pi^{*}$。

从先验 $\pi^{*}$ 出发的贝叶斯规则，在所有序贯决策规则中最小化以 $\pi^{*}$ 加权的期望损失。

从另一个先验 $\pi_0$ 出发的贝叶斯决策者会使用其中某一条规则。

因此对于任意 $\pi_0$，都有 $\bar{V}_{Bayes}\left(\pi_{0}\right)\geq\bar{V}_{Bayes}\left(\pi^{*}\right)$。

下面的图展示了四种情况，分别对应
$\pi^{*}=0.25,0.3,0.5,0.7$。

数值求出的最小值点接近 $\pi^{*}$，但并不完全相等。

函数 $\bar{V}_{Bayes}\left(\pi_{0}\right)$ 在其最小值附近较为平坦，因此网格和蒙特卡洛误差会使计算得到的 $\pi_{0}^{*}$ 有所偏移。

```{code-cell} ipython3
def compute_V_baye_bar(π_star, V0, V1, wf):

    V_baye = π_star * V0 + (1 - π_star) * V1
    π_idx = np.argmin(V_baye)
    π_optimal = wf.π_grid[π_idx]
    V_baye_bar = V_baye[π_idx]

    return V_baye, π_optimal, V_baye_bar
```

```{code-cell} ipython3
π_star_arr = [0.25, 0.3, 0.5, 0.7]

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, π_star in enumerate(π_star_arr):
    row_i = i // 2
    col_i = i % 2

    V_baye, π_optimal, V_baye_bar = compute_V_baye_bar(π_star, V0, V1, wf)

    axs[row_i, col_i].plot(wf.π_grid, V_baye)
    axs[row_i, col_i].hlines(V_baye_bar, 0, 1, linestyle='--')
    axs[row_i, col_i].vlines(π_optimal, V_baye_bar, V_baye.max(), linestyle='--')
    axs[row_i, col_i].text(π_optimal+0.05, (V_baye_bar + V_baye.max()) / 2,
                        r'${\pi_0^*}=$'+f'{π_optimal:0.2f}')
    axs[row_i, col_i].set_xlabel(r'$\pi$')
    axs[row_i, col_i].set_ylabel(r'$\overline{V}_{baye}(\pi)$')
    axs[row_i, col_i].set_title(r'$\pi^*=$' + f'{π_star}')

fig.suptitle(r'$\overline{V}_{baye}(\pi)=\pi^*V^0(\pi) + (1-\pi^*)V^1(\pi)$', fontsize=16)
plt.show()
```

以下Python代码对一系列 $\pi^{*}$ 的网格值计算 $\pi_{0}^{*}$ 和 $\bar{V}_{Bayes} = \bar{V}_{Bayes}\left(\pi_{0}^{*}\right)$。

```{code-cell} ipython3
π_star_arr = np.linspace(0.1, 0.9, n_π)
V_baye_bar_arr = np.empty_like(π_star_arr)
π_optimal_arr = np.empty_like(π_star_arr)

for i, π_star in enumerate(π_star_arr):

    V_baye, π_optimal, V_baye_bar = compute_V_baye_bar(π_star, V0, V1, wf)

    V_baye_bar_arr[i] = V_baye_bar
    π_optimal_arr[i] = π_optimal

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(π_star_arr, V_baye_bar_arr)
axs[0].set_xlabel(r'$\pi^*$')
axs[0].set_title(r'$\overline{V}_{baye}$')

axs[1].plot(π_star_arr, π_optimal_arr, label='optimal prior')
axs[1].plot([π_star_arr.min(), π_star_arr.max()],
            [π_star_arr.min(), π_star_arr.max()],
            c='k', linestyle='--', label='45 degree line')
axs[1].set_xlabel(r'$\pi^*$')
axs[1].set_title(r'给定$\pi^*$的最优先验')
axs[1].legend()

plt.show()

print(f"largest |π_0^* - π^*| = {np.max(np.abs(π_optimal_arr - π_star_arr)):.3f}")
```

计算得到的 $\pi_{0}^{*}$ 与45度线之间的偏差保持在 $0.04$ 以内。

## 海军上尉的直觉是否正确？

现在我们比较通过频率派 Neyman-Pearson 规则和贝叶斯决策规则得到的平均损失。

这里比较的是固定样本量规则和序贯规则。

两种规则都使用先验 $\pi^{*}$ 以及损失 $\bar L_0$、$\bar L_1$ 和 $c$。

固定样本规则 $(t, d)$ 是一种序贯规则，它忽略数据直到日期 $t$，然后停止。

贝叶斯规则在所有序贯规则中最小化期望损失。

因此，对于每一个 $\pi^{*}$，都有 $\bar{V}_{Bayes}\leq\bar{V}_{fre}$。

下面的模拟说明了这一不等式，并测量了两者之间的差距。

让我们先从比较 $\pi^{*}=0.5$ 时的平均损失函数开始。

```{code-cell} ipython3
π_star = 0.5
```

```{code-cell} ipython3
# 频率派
V_fre_arr, PFA_arr, PD_arr = compute_V_fre(L0_arr, L1_arr, π_star, wf)

# 贝叶斯派
V_baye = π_star * V0 + (1 - π_star) * V1
V_baye_bar = V_baye.min()

print(f"V_fre = {V_fre_arr.min():.2f} at t = {np.argmin(V_fre_arr) + 1}")
print(f"V_Bayes = {V_baye_bar:.2f}")
```

```{code-cell} ipython3
plt.plot(range(1, T+1), V_fre_arr, label=r'$\min_{d} \overline{V}_{fre}(t,d)$')
plt.plot([1, T], [V_baye_bar, V_baye_bar], label=r'$\overline{V}_{baye}$')
plt.xlabel('t')
plt.title(r'$\pi^*=0.5$')
plt.legend()
plt.show()
```

显然，在任何样本量 $t$ 下，Neyman-Pearson 决策规则都无法获得比贝叶斯规则更低的损失函数。

在 $\pi^{*}=0.5$ 时，最优固定样本规则抽取 $8$ 个观测值，期望损失为 $18.33$；贝叶斯规则的期望损失为 $11.36$。

贝叶斯值是使用存储在 `wf.z0` 和 `wf.z1` 中的 $1{,}000$ 次蒙特卡洛抽样计算得到的，因此其末位数字并不可靠；后面一节中的模拟显示贝叶斯损失接近 $12.3$，仍远低于 $18.33$。

下图比较了所有 $\pi^{*}$ 值下的两种规则。

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(π_star_arr, V_fre_bar_arr, label=r'$\overline{V}_{fre}$')
axs[0].plot(π_star_arr, V_baye_bar_arr, label=r'$\overline{V}_{baye}$')
axs[0].legend()
axs[0].set_xlabel(r'$\pi^*$')

axs[1].plot(π_star_arr, V_fre_bar_arr - V_baye_bar_arr, label='$diff$')
axs[1].legend()
axs[1].set_xlabel(r'$\pi^*$')

plt.show()

diff = V_fre_bar_arr - V_baye_bar_arr
print(f"min diff = {diff.min():.2f} at π* = {π_star_arr[np.argmin(diff)]:.2f}")
print(f"max diff = {diff.max():.2f} at π* = {π_star_arr[np.argmax(diff)]:.2f}")
```

上图右侧面板绘制了差值 $\bar{V}_{fre}-\bar{V}_{Bayes}$。

正如该定理所要求的那样，这个差值始终为正值。

差距在 $\pi^{*}=0.1$ 处最小，为 $2.63$；在 $\pi^{*}\approx 0.82$ 处最大，为 $8.62$。

## 更多细节

我们可以通过聚焦于$\pi^{*}=0.5=\pi_{0}$的情况来提供更多见解。

```{code-cell} ipython3
π_star = 0.5
```

回顾当$\pi^*=0.5$时，频率派 Neyman-Pearson 决策规则会**事先**设定一个样本量`t_optimal`。

对于我们的参数设置，我们可以计算它的值：

```{code-cell} ipython3
t_optimal
```

为了方便，让我们将 `t_idx` 定义为对应于 `t_optimal` 样本量的 Python 数组索引。

```{code-cell} ipython3
t_idx = t_optimal - 1
```

## 贝叶斯决策规则的决策时间分布

通过模拟，我们计算贝叶斯决策规则的决策时间频率分布，并将该时间与频率派规则的固定$t$进行比较。

以下Python代码创建了一个图表，显示了贝叶斯决策者的贝叶斯决策时间的频率分布，条件是数据由分布$q=f_{0}$或$q=f_{1}$生成。

蓝色和橙色虚线显示了贝叶斯决策规则在$f_0$和$f_1$下的平均值，而黑色虚线显示了频率派的最优样本量$t$。

当$q=f_0$时，贝叶斯规则平均比频率派规则**更早**做出决定，而当$q=f_1$时则**更晚**做出决定。

函数 `check_results` 会为那些在$T$之前未能做出决定的路径记录时间$T+1$。

```{code-cell} ipython3
@jit(parallel=True)
def check_results(L_arr, A, B, flag, π0):

    N, T = L_arr.shape

    time_arr = np.full(N, T + 1)                  # T + 1 表示直到T仍未做出决定
    correctness = np.zeros(N, dtype=np.bool_)

    π_arr = π0 * L_arr / (π0 * L_arr + 1 - π0)

    for i in prange(N):
        for t in range(T):
            if (π_arr[i, t] <= B) or (π_arr[i, t] >= A):
                time_arr[i] = t + 1
                correctness[i] = (flag == 0 and π_arr[i, t] >= A) or (flag == 1 and π_arr[i, t] <= B)
                break

    return time_arr, correctness
```

```{code-cell} ipython3
time_arr0, correctness0 = check_results(L0_arr, A, B, 0, π_star)
time_arr1, correctness1 = check_results(L1_arr, A, B, 1, π_star)

# 无条件分布
time_arr_u = np.concatenate((time_arr0, time_arr1))
correctness_u = np.concatenate((correctness0, correctness1))

print("直到T仍未做出决定的路径数：", np.sum(time_arr_u > T))
print(f"平均决策时间：f0下为{time_arr0.mean():.2f}，"
      f"f1下为{time_arr1.mean():.2f}，合并后为{time_arr_u.mean():.2f}")
```

```{code-cell} ipython3
n1 = plt.hist(time_arr0, bins=range(1, 30), alpha=0.4, label='f0生成')[0]
n2 = plt.hist(time_arr1, bins=range(1, 30), alpha=0.4, label='f1生成')[0]
plt.vlines(t_optimal, 0, max(n1.max(), n2.max()), linestyle='--',
           color='k', label='频率派')
plt.vlines(np.mean(time_arr0), 0, max(n1.max(), n2.max()),
           linestyle='--', color='C0', label='f0下的E(t)')
plt.vlines(np.mean(time_arr1), 0, max(n1.max(), n2.max()),
           linestyle='--', color='C1', label='f1下的E(t)')
plt.legend();

plt.xlabel('t')
plt.ylabel('n')
plt.title('时间的条件频率分布')

plt.show()
```

稍后我们将弄清这些分布最终如何影响Neyman-Pearson和贝叶斯决策规则下的客观期望值。

首先，让我们看看贝叶斯信念随时间的模拟。

利用 {doc}`likelihood_bayes` 中关于 {ref}`似然比过程与贝叶斯法则 <lb_bayes_law>` 一节所描述的从$L_{t}$到$\pi_{t}$的一一映射（给定$\pi_0$），我们可以计算任意时间$t$的更新信念。

```{code-cell} ipython3
π0_arr = π_star * L0_arr / (π_star * L0_arr + 1 - π_star)
π1_arr = π_star * L1_arr / (π_star * L1_arr + 1 - π_star)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(np.arange(1, π0_arr.shape[1]+1), np.mean(π0_arr, 0), label='f0生成')
axs[0].plot(np.arange(1, π1_arr.shape[1]+1), 1 - np.mean(π1_arr, 0), label='f1生成')
axs[0].set_xlabel('t')
axs[0].set_ylabel(r'$E(\pi_t)$ 或 ($1 - E(\pi_t)$)')
axs[0].set_title('抽取t个观测值后信念的期望')
axs[0].legend()

axs[1].plot(np.arange(1, π0_arr.shape[1]+1), np.var(π0_arr, 0), label='f0生成')
axs[1].plot(np.arange(1, π1_arr.shape[1]+1), np.var(π1_arr, 0), label='f1生成')
axs[1].set_xlabel('t')
axs[1].set_ylabel(r'var($\pi_t$)')
axs[1].set_title('抽取t个观测值后信念的方差')
axs[1].legend()

plt.show()
```

上图比较了经过$t$次抽样后贝叶斯后验分布的均值和方差。

左图比较了$E_{f_0}\left(\pi_{t}\right)$和$1-E_{f_1}\left(\pi_{t}\right)$：它们完全重合。

$\pi_t$的鞅性质说明了这一结果。

在贝叶斯主观分布下（该分布以权重$\pi_0$和$1-\pi_0$混合了$f_0$和$f_1$），$\pi_t$是一个均值为$\pi_0$的鞅。

因此对任意$t$都有$\pi_0 E_{f_0}\left(\pi_{t}\right)+\left(1-\pi_0\right)E_{f_1}\left(\pi_{t}\right)=\pi_0$。

当$\pi_0=1/2$时，这一恒等式变为$E_{f_0}\left(\pi_{t}\right)=1-E_{f_1}\left(\pi_{t}\right)$。

两条模拟曲线之间的差异仅来自蒙特卡洛误差。

右图显示，当$t$较小时，$\pi_t$的方差在$f_{1}$下更低。

库尔贝克-莱布勒散度的不对称性解释了贝叶斯决策者在$f_{1}$生成数据时等待更长时间才做出决定的原因。

{doc}`divergence_measures` 描述了库尔贝克-莱布勒散度。

在$f_0$下，每次抽样$\log L_t$平均上升$KL(f_0\|f_1) = E_{f_0}\left[\log\left(f_0(z)/f_1(z)\right)\right]$。

在$f_1$下，每次抽样$\log L_t$平均下降$KL(f_1\|f_0) = E_{f_1}\left[\log\left(f_1(z)/f_0(z)\right)\right]$。

当$\pi_0 = 1/2$时，贝叶斯决策者在$\log L_t$离开区间$(\log L_B, \log L_A)$时停止，其中$L_A = A/(1-A)$，$L_B = B/(1-B)$。

沃尔德对期望抽样次数的近似忽略了$\log L_t$超出阈值的幅度：

$$
E_{f_1}[N] \approx \frac{\Pr_{f_1}\{\text{接受} f_1\}\log L_B + \Pr_{f_1}\{\text{接受} f_0\}\log L_A}{-KL(f_1\|f_0)}
$$

在$f_0$下情况类似。

```{code-cell} ipython3
def KL(f, g):
    "KL(f || g)，即f log(f/g)在[0, 1]上的积分。"
    integrand = lambda z: f(z) * np.log(f(z) / g(z))
    return quad(integrand, 0, 1)[0]

KL_01 = KL(wf.f0, wf.f1)
KL_10 = KL(wf.f1, wf.f0)

LA = (1 - π_star) * A / (π_star * (1 - A))
LB = (1 - π_star) * B / (π_star * (1 - B))

p0 = correctness0.mean()   # f0下接受f0的概率
p1 = correctness1.mean()   # f1下接受f1的概率

EN0_wald = (p0 * np.log(LA) + (1 - p0) * np.log(LB)) / KL_01
EN1_wald = (p1 * np.log(LB) + (1 - p1) * np.log(LA)) / (-KL_10)

print(f"KL(f0||f1) = {KL_01:.3f}，KL(f1||f0) = {KL_10:.3f}")
print(f"log L_A = {np.log(LA):.2f}，log L_B = {np.log(LB):.2f}")
print(f"E_f0[N]：沃尔德近似值为{EN0_wald:.1f}，模拟值为{time_arr0.mean():.1f}")
print(f"E_f1[N]：沃尔德近似值为{EN1_wald:.1f}，模拟值为{time_arr1.mean():.1f}")

# f0下log L_A的平均超出量，以及f1下log L_B的平均超出量
logL0_stop = np.log(L0_arr[np.arange(N), time_arr0 - 1])
logL1_stop = np.log(L1_arr[np.arange(N), time_arr1 - 1])
print(f"平均超出量：f0下为{np.mean(logL0_stop[correctness0] - np.log(LA)):.2f}，"
      f"f1下为{np.mean(np.log(LB) - logL1_stop[correctness1]):.2f}")
```

散度$KL(f_0\|f_1)\approx 0.759$是$KL(f_1\|f_0)\approx 0.344$的两倍多。

阈值$\log L_A \approx 2.84$和$\log L_B \approx -2.94$到零的距离大致相等。

在$f_1$下，$\log L_t$向$\log L_B$漂移的速度不到它在$f_0$下向$\log L_A$漂移速度的一半。

沃尔德近似给出$E_{f_1}[N]\approx 8.3$，接近模拟值$9.1$。

它给出的$E_{f_0}[N]\approx 3.4$则远低于模拟值$6.0$。

在$f_0$下，接近零的$z$取值会使$\log\left(f_0(z)/f_1(z)\right)$变得非常大，因此$\log L_t$平均会超出$\log L_A$约$2$；而在$f_1$下，$\log L_t$超出$\log L_B$的幅度平均约为$0.3$。

下面的代码通过简单地将两个可能分布$f_0$和$f_1$的模拟数据合并,绘制了无条件分布的结果。

这个合并分布从某种意义上描述了贝叶斯决策者平均会更早做出决定,这似乎在一定程度上证实了海军上尉的直觉判断。

合并后的平均决策时间为$7.54$次抽样，而$t_{\rm optimal}=8$。

```{code-cell} ipython3
n = plt.hist(time_arr_u, bins=range(1, 30), alpha=0.4, label='bayesian')[0]
plt.vlines(np.mean(time_arr_u), 0, n.max(), linestyle='--',
           color='b', label='bayesian E(t)')
plt.vlines(t_optimal, 0, n.max(), linestyle='--', color='k', label='frequentist')
plt.legend()

plt.xlabel('t')
plt.ylabel('n')
plt.title('时间的无条件分布')

plt.show()
```

## 做出正确决策的概率

现在我们使用模拟来计算贝叶斯和频率派Neyman-Pearson决策规则做出正确决定的样本比例。

对于频率派Neyman-Pearson规则，在$f_{1}$下做出正确决定的概率是我们之前定义的给定$t$时的最优检测概率，同样地，它等于1减去$f_{0}$下的最优虚警概率。

下面我们绘制频率派规则的这两个概率，以及贝叶斯规则在$t$之前做出决定*且*决定正确的条件概率。

```{code-cell} ipython3
# 频率派最优样本量下的最优虚警概率和检测概率
V, PFA, PD = V_fre_t(t_optimal, L0_arr, L1_arr, π_star, wf)
```

```{code-cell} ipython3
plt.plot([1, 20], [PD, PD], linestyle='--', label='PD: 正确选择f1的频率')
plt.plot([1, 20], [1-PFA, 1-PFA], linestyle='--', label='1-PFA: 正确选择f0的频率')
plt.vlines(t_optimal, 0, 1, linestyle='--', label='频率论最优样本量')

T_arr = np.arange(1, 21)
plt.plot(T_arr, [np.mean(correctness0 & (time_arr0 <= t)) for t in T_arr],
        label='q=f0且贝叶斯选择f0')
plt.plot(T_arr, [np.mean(correctness1 & (time_arr1 <= t)) for t in T_arr],
        label='q=f1且贝叶斯选择f1')
plt.legend(loc=4)

plt.xlabel('t')
plt.ylabel('概率')
plt.title('t之前做出正确决定的条件概率')

plt.show()
```

通过使用 $\pi^{*}$ 进行平均，我们还绘制了无条件分布。

```{code-cell} ipython3
plt.plot([1, 20], [(PD + 1 - PFA) / 2, (PD + 1 - PFA) / 2],
        linestyle='--', label='频率派正确决策')
plt.vlines(t_optimal, 0, 1, linestyle='--', label='频率派最优样本量')

plt.plot(T_arr, [np.mean(correctness_u & (time_arr_u <= t)) for t in T_arr],
        label="贝叶斯派正确决策")
plt.legend()

plt.xlabel('t')
plt.ylabel('概率')
plt.title('t时刻前做出正确决策的无条件概率')

plt.show()
```

模拟得到的决策时间和正确决策的概率给出了贝叶斯期望损失的第二个估计值。

这个估计值使用了与频率派规则相同的模拟路径。

```{code-cell} ipython3
V0_sim = wf.c * time_arr0.mean() + wf.L1 * (1 - correctness0.mean())
V1_sim = wf.c * time_arr1.mean() + wf.L0 * (1 - correctness1.mean())
V_baye_sim = π_star * V0_sim + (1 - π_star) * V1_sim

V_fre_sim = wf.c * t_optimal + V

print(f"贝叶斯规则：     {V_baye_sim:.2f}")
print(f"固定样本规则：   {V_fre_sim:.2f}")
```

贝叶斯规则的模拟期望损失为 $12.27$，超过了 `V_q` 给出的 $11.36$。

这一差距来自于 `V_q` 用来近似期望值的 $1{,}000$ 次抽取 `wf.z0` 和 `wf.z1` 中的蒙特卡洛误差。

这两个值都远低于固定样本规则的损失 $18.33$。

## Neyman-Pearson的$t$处的似然比分布

接下来我们使用模拟来构建在 $t$ 次抽样后的似然比分布。

作为有用的参考点，我们还展示了对应于贝叶斯截断值 $A$ 和 $B$ 的似然比 $L_A$ 和 $L_B$。

为了更清晰地展示分布，我们报告似然比的对数值。

下面的图表报告了两个分布，一个是在 $f_0$ 生成数据的条件下的分布，另一个是在 $f_1$ 生成数据的条件下的分布。

```{code-cell} ipython3
L_min = min(L0_arr[:, t_idx].min(), L1_arr[:, t_idx].min())
L_max = max(L0_arr[:, t_idx].max(), L1_arr[:, t_idx].max())
bin_range = np.linspace(np.log(L_min), np.log(L_max), 50)
n0 = plt.hist(np.log(L0_arr[:, t_idx]), bins=bin_range, alpha=0.4, label='f0生成')[0]
n1 = plt.hist(np.log(L1_arr[:, t_idx]), bins=bin_range, alpha=0.4, label='f1生成')[0]

plt.vlines(np.log(LB), 0, max(n0.max(), n1.max()), linestyle='--', color='r', label='log($L_B$)')
plt.vlines(np.log(LA), 0, max(n0.max(), n1.max()), linestyle='--', color='b', label='log($L_A$)')
plt.legend()

plt.xlabel('log(L)')
plt.ylabel('n')
plt.title('频率学派t时对数似然比的条件分布')

plt.show()
```

下一个图表绘制了在频率学派 $t$ 处 $\log L$ 的无条件分布，这是通过将两个条件分布合并而构建的。

当 $\pi^* = 0.5$ 时，合并来自 $f_0$ 和 $f_1$ 的相同数量的路径，相当于用 $\pi^*$ 和 $1-\pi^*$ 对它们进行加权。

```{code-cell} ipython3
n = plt.hist(np.log(np.concatenate([L0_arr[:, t_idx], L1_arr[:, t_idx]])),
             bins=50, alpha=0.4, label='log(L)的无条件分布')[0]
plt.vlines(np.log(LB), 0, n.max(), linestyle='--', color='r', label='log($L_B$)')
plt.vlines(np.log(LA), 0, n.max(), linestyle='--', color='b', label='log($L_A$)')
plt.legend()

plt.xlabel('log(L)')
plt.ylabel('n')
plt.title('频率论者t时刻的对数似然比的无条件分布')

plt.show()
```

## 练习

```{exercise}
:label: nc_ex1

固定样本量 $t$，并将期望损失 {eq}`val1` 视为临界值 $d$ 的函数。

1. 证明对于每个 $t$，$d^{*}=\left(1-\pi^{*}\right)\bar L_{0}/\left(\pi^{*}\bar L_{1}\right)$ 最小化 {eq}`val1`。提示：将 $PFA$ 和 $1-PD$ 写成对样本 $z^t$ 的积分，并在每个 $z^t$ 处最小化被积函数。
2. 设 $\pi_t$ 为从先验 $\pi^{*}$ 出发经过 $t$ 次抽样后 $f_0$ 的后验概率。证明规则"若 $L(z^t)<d^{*}$ 则判定为 $f_1$"选择的是后验期望损失较小的决策。解释为什么这使得本讲座的频率派规则成为一种预先固定样本量的贝叶斯规则。
3. 对于 `π_star_arr` 中的 20 个 $\pi^{*}$ 值以及 $t = 1, \ldots, 20$，比较在 $d^{*}$ 处的模拟损失 `V_fre_d_t` 与其在 $d$ 的精细网格上的最小值。对从 $d=1$ 开始的 Nelder-Mead 搜索所得到的损失也进行同样的比较。
```

```{solution-start} nc_ex1
:class: dropdown
```

记 $f_i(z^t) = f_i(z_1)\cdots f_i(z_t)$。

该规则在集合 $R_d = \{z^t : L(z^t) < d\}$ 上拒绝 $H_0$。

那么

$$
PFA = \int_{R_d} f_0(z^t)\, dz^t, \qquad 1 - PD = \int_{R_d^c} f_1(z^t)\, dz^t
$$

而 {eq}`val1` 中依赖于 $d$ 的损失项等于

$$
\int \left[ \mathbf{1}\{z^t \in R_d\}\, \pi^{*}\bar L_1 f_0(z^t) + \mathbf{1}\{z^t \notin R_d\}\, (1-\pi^{*})\bar L_0 f_1(z^t) \right] dz^t 。
$$

当 $z^t$ 满足 $\pi^{*}\bar L_1 f_0(z^t) < (1-\pi^{*})\bar L_0 f_1(z^t)$ 时将其归入 $R_d$，否则归入其补集，此时每个 $z^t$ 处的被积函数最小。

将其除以 $\pi^{*}\bar L_1 f_1(z^t)$，该条件变为 $L(z^t) < d^{*}$。

集合 $R_{d^*}$ 在每个 $z^t$ 处都最小化被积函数，因此对任何 $t$，没有其他拒绝域能表现得更好。

贝叶斯法则给出后验

$$
\pi_t = \frac{\pi^{*} L(z^t)}{\pi^{*} L(z^t) + 1 - \pi^{*}} 。
$$

判定为 $f_1$ 的后验期望损失为 $\pi_t \bar L_1$；判定为 $f_0$ 的后验期望损失为 $(1-\pi_t)\bar L_0$。

不等式 $\pi_t \bar L_1 < (1-\pi_t)\bar L_0$ 等价于 $\pi^{*} L(z^t)\bar L_1 < (1-\pi^{*})\bar L_0$，即 $L(z^t)<d^{*}$。

因此，在经过 $t$ 次抽样后，使用 $d=d^{*}$ 的频率派规则所采取的行动，与先验为 $\pi^{*}$ 的贝叶斯主义者将采取的行动相同。

然后对 $t$ 最小化 $\bar V_{fre}(t, d^*)$，就是在任何数据到来之前选择最优样本量。

贝叶斯序贯规则的区别仅在于让数据来决定何时停止。

下面是数值检验。

```{code-cell} ipython3
from scipy.optimize import minimize

d_grid = np.exp(np.linspace(-10, 10, 4001))

def V_grid_min(t, π_s):
    "Minimum over d_grid of the simulated loss V_fre_d_t."
    n = L0_arr.shape[0]
    PFA = np.searchsorted(np.sort(L0_arr[:, t-1]), d_grid) / n   # Pr{L < d | f0}
    PD = np.searchsorted(np.sort(L1_arr[:, t-1]), d_grid) / n    # Pr{L < d | f1}
    return np.min(π_s * PFA * wf.L1 + (1 - π_s) * (1 - PD) * wf.L0)

gap_closed, gap_nm = [], []
for π_s in π_star_arr:
    for t in range(1, 21):
        V_min = V_grid_min(t, π_s)
        V_closed = V_fre_d_t(d_star(π_s, wf), t, L0_arr, L1_arr, π_s, wf)
        res = minimize(V_fre_d_t, 1, args=(t, L0_arr, L1_arr, π_s, wf),
                       method='Nelder-Mead')
        gap_closed.append(V_closed - V_min)
        gap_nm.append(res.fun - V_min)

gap_closed, gap_nm = np.array(gap_closed), np.array(gap_nm)
print(f"closed form: largest excess over grid minimum {gap_closed.max():.3f}")
print(f"Nelder-Mead: largest excess over grid minimum {gap_nm.max():.3f}, "
      f"cases with excess > 0.5: {np.sum(gap_nm > 0.5)} of {gap_nm.size}")
```

在全部 $400$ 种情形下，闭式解与网格最小值之差均在 $0.25$ 以内。

网格最小值低于 $d^{*}$ 处的损失，因为前者是针对模拟样本拟合出的 $d$；而 $d^{*}$ 最小化的是总体损失。

Nelder-Mead 将阶梯函数 `V_fre_d_t` 当作光滑函数处理，可能在平坦的台阶上停止。

在此例中，它在 $400$ 种情形中的 $2$ 种情形下停止时高出网格最小值超过 $0.5$，在最差情形下高出 $0.69$。

```{solution-end}
```

```{exercise}
:label: nc_ex2

频率派统计学家可以在不使用先验的情况下检验海军上校的直觉。

设 $\pi^{*}=0.5$，并取最优固定样本规则在 $t_{\rm optimal}$ 处的误差概率：$\alpha = PFA$ 和 $\beta = 1 - PD$。

具有目标 $(\alpha, \beta)$ 的瓦尔德序贯概率比检验使用 {doc}`wald_friedman` 中的临界值 $a(\alpha,\beta) = (1-\beta)/\alpha$ 和 $b(\alpha,\beta) = \beta/(1-\alpha)$。

该讲座将它们应用于似然比 $f_1/f_0 = 1/L$。

用我们的 $L_t = L(z^t)$ 表示，该检验为

- 若 $L_t \geq A_L \equiv 1/b(\alpha,\beta) = (1-\alpha)/\beta$，则接受 $f_0$
- 若 $L_t \leq B_L \equiv 1/a(\alpha,\beta) = \alpha/(1-\beta)$，则接受 $f_1$
- 若 $B_L < L_t < A_L$，则继续抽取另一个 $z$

在路径 `L0_arr` 和 `L1_arr` 上模拟该检验。

报告其实际的误差概率、$E_{f_0}[N]$ 和 $E_{f_1}[N]$，并与 $t_{\rm optimal}$ 进行比较。
```

```{solution-start} nc_ex2
:class: dropdown
```

当 $\pi_0 = 1/2$ 时，$\pi_t = L_t/(1+L_t)$。

因此瓦尔德检验就是 `check_results` 中的规则，取 $\pi_0 = 1/2$，概率临界值为 $A_L/(1+A_L)$ 和 $B_L/(1+B_L)$。

```{code-cell} ipython3
V_t, PFA_t, PD_t = V_fre_t(t_optimal, L0_arr, L1_arr, 0.5, wf)
α, β = PFA_t, 1 - PD_t

A_L = (1 - α) / β
B_L = α / (1 - β)

n0, accept0 = check_results(L0_arr, A_L / (1 + A_L), B_L / (1 + B_L), 0, 0.5)
n1, accept1 = check_results(L1_arr, A_L / (1 + A_L), B_L / (1 + B_L), 1, 0.5)

print(f"targets:   α = {α:.3f}, β = {β:.3f}")
print(f"realized:  α = {1 - accept0.mean():.3f}, β = {1 - accept1.mean():.3f}")
print(f"E_f0[N] = {n0.mean():.2f},  E_f1[N] = {n1.mean():.2f},  "
      f"t_optimal = {t_optimal}")
print("paths with no decision by T:", np.sum(n0 > T) + np.sum(n1 > T))
```

瓦尔德检验满足两个目标。

其实际误差概率 $0.087$ 和 $0.019$ 均低于 $\alpha = 0.108$ 和 $\beta = 0.059$。

在 $f_0$ 下平均抽取 $5.42$ 次观测值，在 $f_1$ 下平均抽取 $6.84$ 次，而固定样本规则的 $t_{\rm optimal} = 8$。

序贯检验在两个假设下都具有更小的误差概率和更少的期望抽样次数，因此它在不依赖任何先验的情况下优于海军的规则。

{cite:t}`WaldWolfowitz1948` 证明了，任何误差概率不大于序贯概率比检验的检验，在任一假设下的期望抽样次数都不会更小。

```{solution-end}
```

```{exercise}
:label: nc_ex3

本练习衡量序贯抽样的价值。

在模拟路径 `L0_arr` 和 `L1_arr` 上评估两种规则，使两者使用相同的随机数。

1. 设 $\pi^{*}=0.5$。对于 $0.25$ 到 $14$ 之间的一组抽样成本 $c$，计算 $\bar V_{fre}$ 和 $\bar V_{Bayes}$，并绘制 $\bar V_{fre}-\bar V_{Bayes}$ 相对于 $c$ 的曲线。对每个 $c$，重新求解贝叶斯问题以得到新的临界值 $A$ 和 $B$。让固定样本规则选择 $t=0$，即在期望损失为 $\min\{\pi^{*}\bar L_1, (1-\pi^{*})\bar L_0\}$ 的情况下不使用数据做决策。
2. 设 $c=1.25$ 和 $\pi^{*}=0.5$，并让贝叶斯主义者从一个 $\pi_0 \neq \pi^{*}$ 的先验出发。绘制 $\bar V_{fre}-\bar V_{Bayes}(\pi_0)$ 相对于 $\pi_0$ 的曲线。当固定样本规则用 $\pi_0$ 代替 $\pi^{*}$ 来选择 $(t, d)$ 时也做同样的处理。
```

```{solution-start} nc_ex3
:class: dropdown
```

下面的函数在自然界以概率 $\pi^{*}$ 选择 $f_0$ 的情况下，计算具有临界值 $(B, A)$ 和先验 $\pi_0$ 的贝叶斯规则的期望损失。

先验位于 $(B, A)$ 之外的贝叶斯主义者将不使用数据做决策。

```{code-cell} ipython3
def V_baye_sim(A, B, π0, π_star, wf):

    if π0 >= A:
        return (1 - π_star) * wf.L0
    if π0 <= B:
        return π_star * wf.L1

    t0, c0 = check_results(L0_arr, A, B, 0, π0)
    t1, c1 = check_results(L1_arr, A, B, 1, π0)
    assert max(t0.max(), t1.max()) <= T     # every path decides by T

    V0 = wf.c * t0.mean() + wf.L1 * (1 - c0.mean())
    V1 = wf.c * t1.mean() + wf.L0 * (1 - c1.mean())

    return π_star * V0 + (1 - π_star) * V1
```

固定样本规则的误判损失不依赖于 $c$，因此我们只需计算一次。

```{code-cell} ipython3
π_star = 0.5
t_grid = np.arange(1, T+1)
loss_t = np.array([V_fre_t(t, L0_arr, L1_arr, π_star, wf)[0] for t in t_grid])
loss_0 = min(π_star * wf.L1, (1 - π_star) * wf.L0)   # decide with t = 0

c_grid = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10, 14])
V_fre_c = np.empty_like(c_grid)
V_baye_c = np.empty_like(c_grid)

for i, c in enumerate(c_grid):
    seed_numba(1234)
    wf_c = WaldFriedman(c=c)
    B_c, A_c = find_cutoff_rule(wf_c, solve_model(wf_c))

    V_fre_c[i] = min(loss_0, (c * t_grid + loss_t).min())
    V_baye_c[i] = V_baye_sim(A_c, B_c, π_star, π_star, wf_c)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(c_grid, V_fre_c, label=r'$\overline{V}_{fre}$')
axs[0].plot(c_grid, V_baye_c, label=r'$\overline{V}_{baye}$')
axs[0].set_xlabel('$c$')
axs[0].legend()

axs[1].plot(c_grid, V_fre_c - V_baye_c)
axs[1].set_xlabel('$c$')
axs[1].set_title(r'$\overline{V}_{fre} - \overline{V}_{baye}$, $\pi^*=0.5$')

plt.show()

print(np.round(V_fre_c - V_baye_c, 2))
```

该差距从 $c=0.25$ 时的 $2.82$ 上升到 $c=2.5$ 时的 $6.77$，再下降到 $c=14$ 时的 $0.12$。

当抽样成本低廉时，两种规则都会抽取大量样本，且很少出错。

当抽样成本高昂时，两种规则都只抽取约一次样本。

在成本适中时，序贯规则在抽取到有信息量的样本后会提前停止，而固定样本规则则要为其并不需要的抽样付出代价。

对于第二部分，用 $\pi_0$ 代替 $\pi^{*}$ 的 `compute_V_fre` 给出了先验为 $\pi_0$ 的决策者会选择的固定样本规则。

我们用权重 $\pi^{*}$ 和 $1-\pi^{*}$ 来评估该规则的误差概率。

```{code-cell} ipython3
V_fre_arr, _, _ = compute_V_fre(L0_arr, L1_arr, π_star, wf)
V_fre_bar = V_fre_arr.min()

π0_grid = np.linspace(0.1, 0.9, 17)
V_baye_π0 = np.empty_like(π0_grid)
V_fre_π0 = np.empty_like(π0_grid)

for i, π0 in enumerate(π0_grid):
    V_baye_π0[i] = V_baye_sim(A, B, π0, π_star, wf)

    V_arr, PFA_arr, PD_arr = compute_V_fre(L0_arr, L1_arr, π0, wf)
    k = np.argmin(V_arr)
    V_fre_π0[i] = (wf.c * (k + 1) + π_star * PFA_arr[k] * wf.L1
                   + (1 - π_star) * (1 - PD_arr[k]) * wf.L0)

plt.plot(π0_grid, V_fre_bar - V_baye_π0,
         label=r'fixed-sample rule designed with $\pi^*$')
plt.plot(π0_grid, V_fre_π0 - V_baye_π0,
         label=r'fixed-sample rule designed with $\pi_0$')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(π_star, color='k', ls=':', lw=1)
plt.xlabel(r'$\pi_0$')
plt.title(r'$\overline{V}_{fre} - \overline{V}_{baye}(\pi_0)$, $\pi^*=0.5$')
plt.legend()
plt.show()

print(np.round(V_fre_bar - V_baye_π0, 2))
print(np.round(V_fre_π0 - V_baye_π0, 2))
```

即使先验有误，贝叶斯主义者在 $0.15 \leq \pi_0 \leq 0.85$ 的范围内仍优于正确设计的固定样本规则。

在 $\pi_0 = 0.1$ 和 $\pi_0 = 0.9$ 时，贝叶斯主义者的起点接近临界值，常常会接受错误的分布。

当固定样本规则使用同样错误的先验时，序贯规则在网格上的每个 $\pi_0$ 处都占优。

```{solution-end}
```

[^t0]: 决策者也可以在不抽取任何 $z$ 的情况下做决策，期望损失为 $\min\{\pi^{*}\bar L_1, (1-\pi^{*})\bar L_0\}$。在我们所取的一组 $\pi^{*}$ 值中，在 $\pi^{*} = 0.1$、$0.86$ 和 $0.9$ 处，该选项优于任何 $t \geq 1$ 的情形。{ref}`nc_ex3` 让固定样本规则选择 $t = 0$。
