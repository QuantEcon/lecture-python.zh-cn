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
---

(likelihood_ratio_process)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 似然比过程和贝叶斯学习

## 概述

本讲解描述了**似然比过程**在**贝叶斯学习**中所扮演的角色。

如同{doc}`本讲 <likelihood_ratio_process>`中所述，我们将使用{doc}`本讲 <exchangeable>`中的一个简单统计设置。

我们将重点关注似然比过程和**先验**概率如何决定**后验**概率。

我们将推导出一个便利的递归公式，表示今天的后验概率是昨天后验概率和今天似然过程乘法增量的函数。

我们还将介绍该公式的一个有用推广，该推广将今天的后验概率表示为初始先验和今天似然比过程实现值的函数。

我们将研究在我们的设置中，贝叶斯学习者如何最终学习到生成数据的概率分布，这个结果

这建立在{doc}`本讲座 <likelihood_ratio_process>`中研究的似然比过程的渐近行为之上。

我们还将深入探讨贝叶斯学习者的心理，研究其主观信念下的动态变化。

本讲座提供了技术性结果，这些结果是{doc}`本讲座 <odu>`、{doc}`本讲座 <wald_friedman>`和{doc}`本讲座 <navy_captain>`中将要研究的结果的基础。

我们先加载一些Python模块。

```{code-cell} ipython3
:hide-output: false

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from numba import vectorize, jit, prange
from math import gamma
import pandas as pd
from scipy.integrate import quad


import seaborn as sns
colors = sns.color_palette()

@jit
def set_seed():
    np.random.seed(142857)
set_seed()
```

## 背景设置

我们首先回顾{doc}`本讲座 <likelihood_ratio_process>`中的设置，这也是我们在这里采用的设置。

一个非负随机变量 $W$ 具有两个概率密度函数之一，要么是 $f$，要么是 $g$。

在时间开始之前，自然界一劳永逸地决定是从 $f$ 还是从 $g$ 中进行一系列独立同分布的抽样。

我们有时会用 $q$ 表示自然界一劳永逸选择的密度，所以 $q$ 要么是 $f$ 要么是 $g$，且是永久性的。

自然界知道它永久从哪个密度中抽样，但我们这些观察者并不知道。

我们知道 $f$ 和 $g$ 这两个密度，但不知道自然界选择了哪一个。

但我们想要知道。

为此，我们使用观测数据。

我们观察到一个序列 $\{w_t\}_{t=1}^T$，它包含 $T$ 个从 $f$ 或 $g$ 中独立同分布抽取的样本。

我们想要利用这些观测来推断自然界选择了 $f$ 还是 $g$。

**似然比过程**是完成这项任务的有用工具。

首先，我们定义似然比过程的关键组成部分，即时间 $t$ 的似然比，它是一个随机变量：

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

我们假设 $f$ 和 $g$ 在随机变量 $W$ 可能实现值的相同区间上都赋予正概率。

这意味着在 $g$ 密度下，$\ell (w_t)= \frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$ 显然是一个均值为1的非负随机变量。

序列的**似然比过程**

$\left\{ w_{t}\right\} _{t=1}^{\infty}$ 定义如下：

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

其中 $w^t=\{ w_1,\dots,w_t\}$ 是直到时间 $t$ （包括 $t$）的观测历史。

有时为简便起见，我们会写作

$$
L_t =  L(w^t) = \frac{f(w^t)}{g(w^t)}
$$ 

这里我们使用如下约定：
$f(w^t) = f(w_1) f(w_2) \ldots f(w_t)$ 且 $g(w^t) = g(w_1) g(w_2) \ldots g(w_t)$。

注意，似然过程满足以下*递归*或*乘法分解*：

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

似然比及其对数是使用 Neyman 和 Pearson 经典频率派方法进行推断的关键工具 {cite}`Neyman_Pearson`。

我们将再次使用来自{doc}`本讲座 <likelihood_ratio_process>`的以下 Python 代码，该代码将 $f$ 和 $g$ 评估为两个不同的贝塔分布，然后通过从*某个*概率分布（例如，从 $g$ 生成的 IID 序列）生成序列 $w^t$ 来计算和模拟相关的似然比过程。

```{code-cell} ipython3
# Parameters in the two beta distributions.
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# The two density functions.
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

我们还将使用以下Python代码来准备一些信息丰富的模拟

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b, N=50000)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

```{code-cell} ipython3
l_arr_f = simulate(F_a, F_b, N=50000)
l_seq_f = np.cumprod(l_arr_f, axis=1)
```

## 似然比过程和贝叶斯定律

设 $\pi_0 \in [0,1]$ 是贝叶斯统计学家对自然生成的 $w^t$ 序列是来自分布 $f$ 的独立同分布抽样的先验概率。

* 这里的"概率"应被理解为总结或表达主观观点的一种方式
* 它**不**意味着样本量无限增长时的预期相对频率

设 $\pi_{t+1}$ 是定义如下的贝叶斯后验概率：

$$
\pi_{t+1} = {\rm Prob}(q=f|w^{t+1})
$$ (eq:defbayesposterior)

似然比过程是后验概率 $\pi_t$ 演化公式中的主要角色，这是**贝叶斯定律**的一个实例。

让我们推导 $\pi_{t+1}$ 的两个公式，一个用似然比 $l(w_t)$ 表示，另一个用 $L(w^t)$ 表示。

首先，我们使用以下符号约定：

* $f(w^{t+1}) \equiv f(w_1) f(w_2) \cdots f(w_{t+1})$
* $g(w^{t+1}) \equiv g(w_1) g(w_2) \cdots g(w_{t+1})$
* $\pi_0 ={\rm Prob}(q=f |\emptyset)$
* $\pi_t = {\rm Prob}(q=f |w^t)$

这里符号 $\emptyset$ 表示"空集"或"无数据"。

在没有数据的情况下，我们的贝叶斯统计学家认为序列 $w^{t+1}$ 的概率密度是：

$$
{\rm Prob}(w^{t+1} |\emptyset) = \pi_0 f(w^{t+1})+ (1 -  \pi_0) g(w^{t+1}) 
$$

概率定律表明，事件 $A$ 和 $B$ 的联合分布 ${\rm Prob}(AB)$ 与条件分布 ${\rm Prob}(A |B)$ 和 ${\rm Prob}(B |A)$ 之间的关系是：

$$

{\rm Prob}(AB) = {\rm Prob}(A |B) {\rm Prob}(B) = {\rm Prob}(B |A) {\rm Prob}(A) . 
$$ (eq:problawAB)

我们关注的事件是

$$
A = \{q=f\},  \quad B = \{w^{t+1}\}, \quad
$$

其中大括号$\{\cdot\}$是我们用来表示"事件"的简写。

因此在我们的设定中，概率法则{eq}`eq:problawAB`意味着

$$
{\rm Prob}(q=f |w^{t+1})  {\rm Prob}(w^{t+1}  |\emptyset) = {\rm Prob}(w^{t+1} |q=f) {\rm Prob}(q=f  | \emptyset)
$$

或者

$$
\pi_{t+1} \left[\pi_0 f(w^{t+1}) + (1- \pi_0) g(w^{t+1})\right] = f(w^{t+1}) \pi_0 
$$

或者

$$
\pi_{t+1}  = \frac{ f(w^{t+1}) \pi_0 }{\pi_0 f(w^{t+1}) + (1- \pi_0) g(w^{t+1})}
$$

将上述等式右边的分子和分母都除以$g(w^{t+1})$得到

```{math}
:label: eq_Bayeslaw1033

\pi_{t+1}=\frac{\pi_{0}L\left(w^{t+1}\right)}{\pi_{0}L\left(w^{t+1}\right)+1-\pi_{0}} .
```

公式{eq}`eq_Bayeslaw1033`可以被视为在看到数据批次$\left\{ w_{i}\right\} _{i=1}^{t+1}$后对先验概率$\pi_0$的一步修正。

公式{eq}`eq_Bayeslaw1033`显示了似然比过程$L\left(w^{t+1}\right)$在确定后验概率$\pi_{t+1}$中起到的关键作用。

公式{eq}`eq_Bayeslaw1033`是理解以下观点的基础：由于似然比过程在$t \rightarrow + \infty$时的行为特征，似然比过程在决定$\pi_t$的极限行为时会主导初始先验$\pi_0$的影响。

### 递归公式

我们可以使用类似的推理方法得到公式{eq}`eq_Bayeslaw1033`的递归版本。

概率法则表明

$$
{\rm Prob}(q=f|w^{t+1}) = \frac { {\rm Prob}(q=f|w^{t} ) f(w_{t+1})}{ {\rm Prob}(q=f|w^{t} ) f(w_{t+1}) + (1 - {\rm Prob}(q=f|w^{t} )) g(w_{t+1})}
$$

或

$$
\pi_{t+1} = \frac { \pi_t f(w_{t+1})}{ \pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}
$$ (eq:bayes150)

显然，上述方程表明

$$
{\rm Prob}(q=f|w^{t+1}) = \frac{{\rm Prob}(q=f|w^{t}) f(w_{t+1} )} {{\rm Prob}(w_{t+1})}
$$

将方程{eq}`eq:bayes150`右侧的分子和分母都除以$g(w_{t+1})$得到递归式

```{math}
:label: eq_recur1

\pi_{t+1}=\frac{\pi_{t} l_t(w_{t+1})}{\pi_{t} l_t(w_{t+1})+1-\pi_{t}}
```

其中$\pi_{0}$是$q = f$的贝叶斯先验概率，即在我们尚未看到任何数据时基于个人或主观判断的关于$q$的信念。

通过迭代方程{eq}`eq_recur1`可以推导出公式{eq}`eq_Bayeslaw1033`。

下面我们定义一个Python函数，该函数根据递归式{eq}`eq_recur1`使用似然比$\ell$更新信念$\pi$

```{code-cell} ipython3
@jit
def update(π, l):
    "Update π using likelihood l"

    # Update belief
    π = π * l / (π * l + 1 - π)

    return π
```

如上所述，公式 {eq}`eq_Bayeslaw1033` 显示了似然比过程 $L\left(w^{t+1}\right)$ 在确定后验概率 $\pi_{t+1}$ 中发挥的关键作用。

当 $t \rightarrow + \infty$ 时，似然比过程在决定 $\pi_t$ 的极限行为中占主导地位，超过了初始先验 $\pi_0$ 的影响。

为了说明这一见解，我们将绘制图表，展示似然比过程 $L_t$ 的**一条**模拟路径，以及与*相同*似然比过程实现但*不同*初始先验概率 $\pi_{0}$ 相关联的两条 $\pi_t$ 路径。

首先，我们在Python中设定两个 $\pi_0$ 的值。

```{code-cell} ipython3
π1, π2 = 0.2, 0.8
```

接下来我们为从密度函数$f$中独立同分布抽取的历史数据生成似然比过程$L_t$和后验概率$\pi_t$的路径。

```{code-cell} ipython3
T = l_arr_f.shape[1]
π_seq_f = np.empty((2, T+1))
π_seq_f[:, 0] = π1, π2

for t in range(T):
    for i in range(2):
        π_seq_f[i, t+1] = update(π_seq_f[i, t], l_arr_f[0, t])
```

```{code-cell} ipython3
fig, ax1 = plt.subplots()

for i in range(2):
    ax1.plot(range(T+1), π_seq_f[i, :], label=fr"$\pi_0$={π_seq_f[i, 0]}")

ax1.set_ylabel(r"$\pi_t$")
ax1.set_xlabel("t")
ax1.legend()
ax1.set_title("当f支配数据时")

ax2 = ax1.twinx()
ax2.plot(range(1, T+1), np.log(l_seq_f[0, :]), '--', color='b')
ax2.set_ylabel("$log(L(w^{t}))$")

plt.show()
```

图中的虚线记录了似然比过程的对数 $\log L(w^t)$。

请注意 $y$ 轴上有两个不同的刻度。

现在让我们研究当历史由密度 $g$ 产生的独立同分布抽样构成时会发生什么

```{code-cell} ipython3
T = l_arr_g.shape[1]
π_seq_g = np.empty((2, T+1))
π_seq_g[:, 0] = π1, π2

for t in range(T):
    for i in range(2):
        π_seq_g[i, t+1] = update(π_seq_g[i, t], l_arr_g[0, t])
```

```{code-cell} ipython3
fig, ax1 = plt.subplots()

for i in range(2):
    ax1.plot(range(T+1), π_seq_g[i, :], label=fr"$\pi_0$={π_seq_g[i, 0]}")

ax1.set_ylabel(r"$\pi_t$")
ax1.set_xlabel("t")
ax1.legend()
ax1.set_title("当g支配数据时")

ax2 = ax1.twinx()
ax2.plot(range(1, T+1), np.log(l_seq_g[0, :]), '--', color='b')
ax2.set_ylabel("$log(L(w^{t}))$")

plt.show()
```

以下我们提供Python代码，验证自然界永久选择从密度$f$中抽取。

```{code-cell} ipython3
π_seq = np.empty((2, T+1))
π_seq[:, 0] = π1, π2

for i in range(2):
    πL = π_seq[i, 0] * l_seq_f[0, :]
    π_seq[i, 1:] = πL / (πL + 1 - π_seq[i, 0])
```

```{code-cell} ipython3
np.abs(π_seq - π_seq_f).max() < 1e-10
```

因此，我们得出结论，似然比过程是公式{eq}`eq_Bayeslaw1033`中贝叶斯后验概率的关键组成部分，该后验概率表示自然界从密度$f$中重复抽样得到历史$w^t$的概率。

## 另一种时序协议

让我们研究当自然界在不同的时序协议下生成历史$w^t = \{w_1, w_2, \dots, w_t\}$时，后验概率$\pi_t = {\rm Prob}(q=f|w^{t})$的表现。

到目前为止，我们假设在时间1之前，自然界以某种方式选择从**要么**$f$**要么**$g$中进行iid序列抽样来得到$w^t$。

自然界关于是从$f$还是从$g$中抽样的决定因此是**永久性的**。

现在我们假设一个不同的时序协议，在**每个**时期$t =1, 2, \ldots$之前，自然界：

* 抛一个权重为$x$的硬币，然后
* 如果抛出"正面"就从$f$中抽样
* 如果抛出"反面"就从$g$中抽样

在这个时序协议下，自然界**既不**永久地从$f$抽样**也不**永久地从$g$抽样，所以认为自然界在**永久地**从其中之一进行i.i.d.抽样的统计学家是错误的。

* 事实上，自然界实际上是**永久地**从$f$和$g$的$x$-混合分布中抽样——当$x \in (0,1)$时，这个分布既不是$f$也不是$g$

因此，贝叶斯先验 $\pi_0$ 和方程 {eq}`eq_Bayeslaw1033` 描述的后验概率序列**不应该**被解释为统计学家对于另一种时序协议（即自然从 $f$ 和 $g$ 的 $x$ 混合分布中抽样）下混合参数 $x$ 的观点。

当我们回顾方程 {eq}`eq:defbayesposterior` 中 $\pi_t$ 的定义时，这一点就很清楚了。为方便起见，我们在这里重复该方程：

$$
\pi_{t+1} = {\rm Prob}(q=f|w^{t+1})
$$

让我们编写一些 Python 代码来研究当自然实际上既不是从 $f$ 也不是从 $g$ 生成数据，而是从两个 beta 分布的 $x$ 混合分布中进行 i.i.d. 抽样时，$\pi_t$ 的行为。

```{note}
这是一个统计学家的模型被错误指定的情况，因此我们应该预期相对于 $x$ 混合分布的 Kullback-Liebler 散度将影响结果。
```

我们可以研究对于自然混合概率 $x$ 的不同值，$\pi_t$ 会如何表现。

首先，让我们创建一个函数来模拟混合时序协议下的数据：

```{code-cell} ipython3
@jit
def simulate_mixture_path(x_true, T):
    """
    模拟混合时序协议下的 T 个观测值。
    """
    w = np.empty(T)
    for t in range(T):
        if np.random.rand() < x_true:
            w[t] = np.random.beta(F_a, F_b)
        else:
            w[t] = np.random.beta(G_a, G_b)
    return w
```

让我们从这个混合模型生成一系列观测值，其真实混合概率为$x=0.5$。

我们首先用这个序列来研究$\pi_t$的行为。

```{note}
之后，我们可以用它来研究一个统计学家如何在知道数据是由$f$和$g$的$x$混合生成的情况下，构建$x$的最大似然估计或贝叶斯估计，以及$f$和$g$的自由参数。
```

```{code-cell} ipython3
x_true = 0.5
T_mix = 200

# 三个不同的先验，均值分别为0.25, 0.5, 0.75
prior_params = [(1, 3), (1, 1), (3, 1)]
prior_means = [a/(a+b) for a, b in prior_params]

# 从混合模型生成一条观测路径
set_seed()
w_mix = simulate_mixture_path(x_true, T_mix)
```

### 在错误模型下 $\pi_t$ 的行为

让我们研究当数据实际上是由 $f$ 和 $g$ 的 $x$-混合生成时，从 $f$ 永久抽取的后验概率 $\pi_t$ 的表现。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
T_plot = 200

for i, mean0 in enumerate(prior_means):
    π_wrong = np.empty(T_plot + 1)
    π_wrong[0] = mean0
    
    # 计算混合数据的似然比
    for t in range(T_plot):
        l_t = f(w_mix[t]) / g(w_mix[t])
        π_wrong[t + 1] = update(π_wrong[t], l_t)
    
    ax.plot(range(T_plot + 1), π_wrong, 
            label=fr'$\pi_0 = ${mean0:.2f}', 
            color=colors[i], linewidth=2)

ax.axhline(y=x_true, color='black', linestyle='--', 
           label=f'True x = {x_true}', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel(r'$\pi_t$')
ax.legend()
plt.show()
```

显然，$\pi_t$ 收敛到1。

这表明模型得出结论认为数据是由 $f$ 生成的。

为什么会这样呢？

给定 $x = 0.5$，数据生成过程是 $f$ 和 $g$ 的混合：$m(w) = \frac{1}{2}f(w) + \frac{1}{2}g(w)$。

让我们检查一下混合分布 $m$ 与 $f$ 和 $g$ 之间的 [KL散度](rel_entropy)。

```{code-cell} ipython3
def compute_KL(f, g):
    """
    计算KL散度 KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val


def compute_div_m(f, g):
    """
    计算Jensen-Shannon散度
    """
    def m(w):
        return 0.5 * (f(w) + g(w))
    
    return compute_KL(m, f), compute_KL(m, g)


KL_f, KL_g = compute_div_m(f, g)

print(f'KL(m, f) = {KL_f:.3f}\nKL(m, g) = {KL_g:.3f}')
```

由于 $KL(m, f) < KL(m, g)$，$f$ 相对于混合分布 $m$ 来说"更接近"。

因此根据我们在{doc}`likelihood_ratio_process`中关于 KL 散度和似然比过程的讨论，当 $t \to \infty$ 时，$\log(L_t) \to \infty$。

现在回看关键方程{eq}`eq_Bayeslaw1033`。

考虑函数

$$
h(z) = \frac{\pi_0 z}{\pi_0 z + 1 - \pi_0}.
$$

极限 $\lim_{z \to \infty} h(z)$ 等于1。

因此对任意 $\pi_0 \in (0,1)$，当 $t \to \infty$ 时，$\pi_t \to 1$。

这解释了我们在上图中观察到的现象。

但我们如何学习真实的混合参数 $x$？

这个主题将在{doc}`mix_model`中讨论。

我们将在{doc}`mix_model`的练习中探讨如何学习真实的混合参数 $x$。

## 后验概率 $\{\pi_t\}$ 在主观概率分布下的行为

我们将通过简要研究贝叶斯学习者在贝叶斯法则产生的主观信念 $\pi_t$ 下期望学到什么来结束本讲。

这将为我们应用贝叶斯法则作为学习理论提供一些视角。

我们将看到，在每个时刻 $t$，贝叶斯学习者知道他将会感到惊讶。

但他预期新信息不会导致他改变信念。

而且在他的主观信念下，平均来说确实不会改变。

我们将继续在这样的设定下讨论：McCall工人知道他的工资连续抽样要么来自 $F$ 要么来自 $G$，但不知道是这两个分布中的哪一个。

自然在时间 $0$ 之前已经一次性地做出了选择。

让我们回顾、重申并重新整理我们在上文和相关讲座中遇到的一些公式。

工人的初始信念导致了一个关于潜在无限序列抽样 $w_0, w_1, \ldots $ 的联合概率分布。

贝叶斯定律仅仅是概率法则的应用，用于计算第 $t$ 次抽样 $w_t$ 在已知 $[w_0, \ldots, w_{t-1}]$ 条件下的条件分布。

在我们的工人对自然选择分布 $F$ 赋予主观概率 $\pi_{-1}$ 后，我们实际上从一开始就假设决策者**知道**过程 $\{w_t\}_{t=0}$ 的联合分布。

我们假设工人也知道概率论的法则。

一个值得尊重的观点是，贝叶斯定律与其说是一个学习理论，不如说是一个关于信息流入对决策者（他认为从一开始就知道真相，即联合概率分布）的影响的陈述。

### 再谈机械细节

在时间 $0$ **之前**抽取工资报价时，工人将概率 $\pi_{-1} \in (0,1)$ 赋予分布 $F$。

在时间 $0$ 抽取工资之前，工人因此认为 $w_0$ 的密度是

$$
h(w_0;\pi_{-1}) = \pi_{-1} f(w_0) + (1-\pi_{-1}) g(w_0).
$$

令 $a \in \{ f, g\} $ 为一个指标，表示自然是永久地从分布 $f$ 还是从分布 $g$ 中抽样。

在绘制$w_0$后，工人使用贝叶斯定理推导出后验概率$\pi_0 = {\rm Prob}({a = f | w_0})$（即密度为$f(w)$的概率）为：

$$
\pi_0 = { \pi_{-1} f(w_0) \over \pi_{-1} f(w_0) + (1-\pi_{-1}) g(w_0)} .
$$

更一般地，在进行第$t$次抽取并观察到$w_t, w_{t-1}, \ldots, w_0$后，工人认为$w_{t+1}$是从分布$F$中抽取的概率为：

$$
\pi_t = \pi_t(w_t | \pi_{t-1}) \equiv { \pi_{t-1} f(w_t)/g(w_t) \over \pi_{t-1} f(w_t)/g(w_t) + (1-\pi_{t-1})}
$$ (eq:like44)

或

$$
\pi_t=\frac{\pi_{t-1} l_t(w_t)}{\pi_{t-1} l_t(w_t)+1-\pi_{t-1}}
$$

而在给定$w_t, w_{t-1}, \ldots, w_0$条件下$w_{t+1}$的密度为：

$$
h(w_{t+1};\pi_{t}) = \pi_{t} f(w_{t+1}) + (1-\pi_{t}) g(w_{t+1}) .
$$

注意到：

$$
\begin{aligned}
E(\pi_t | \pi_{t-1}) & = \int \Bigl[  { \pi_{t-1} f(w) \over \pi_{t-1} f(w) + (1-\pi_{t-1})g(w)  } \Bigr]
 \Bigl[ \pi_{t-1} f(w) + (1-\pi_{t-1})g(w) \Bigr]  d w \cr
& = \pi_{t-1} \int  f(w) dw  \cr
              & = \pi_{t-1}, \cr
\end{aligned}
$$

因此过程$\pi_t$是一个**鞅**。

事实上，它是一个**有界鞅**，因为每个$\pi_t$作为概率都在0和1之间。

在上述等式串的第一行中，第一个方括号中的项就是作为$w_{t}$函数的$\pi_t$，而第二个方括号中的项是在给定条件下$w_{t}$的密度。

在 $w_{t-1}, \ldots , w_0$ 上，或等价地在 $w_{t-1}, \ldots , w_0$ 的*充分统计量* $\pi_{t-1}$ 上的条件。

注意这里我们是在括号中第二项所描述的**主观**密度下计算 $E(\pi_t | \pi_{t-1})$。

因为 $\{\pi_t\}$ 是一个有界鞅序列，根据**鞅收敛定理**，$\pi_t$ 几乎必然收敛到 $[0,1]$ 中的一个随机变量。

实际上，这意味着概率为1的样本路径 $\{\pi_t\}_{t=0}^\infty$ 是收敛的。

根据定理，不同的样本路径可以收敛到不同的极限值。

因此，让 $\{\pi_t(\omega)\}_{t=0}^\infty$ 表示由特定 $\omega \in \Omega$ 索引的特定样本路径。

我们可以认为自然从概率分布 ${\textrm{Prob}} \Omega$ 中抽取一个 $\omega \in \Omega$，然后生成该过程的单个实现（或_模拟_）$\{\pi_t(\omega)\}_{t=0}^\infty$。

当 $t \rightarrow +\infty$ 时，$\{\pi_t(\omega)\}_{t=0}^\infty$ 的极限点是一个随机变量的实现，这个随机变量是当我们从 $\Omega$ 中采样 $\omega$ 并构造 $\{\pi_t(\omega)\}_{t=0}^\infty$ 的重复抽样时产生的。

通过观察运动方程 {eq}`eq_recur1` 或 {eq}`eq:like44`，我们可以推断出一些关于极限点概率分布的信息

$$
\pi_\infty(\omega) = \lim_{t \rightarrow + \infty} \pi_t(\omega).
$$

显然，由于我们假设 $f \neq g$ 时似然比 $\ell(w_t)$ 与 1 不同，{eq}`eq:like44` 的唯一可能的固定点是

$$
\pi_\infty(\omega) =1
$$

和

$$
\pi_\infty(\omega) =0
$$

因此，对某些实现来说，$\lim_{\rightarrow + \infty} \pi_t(\omega) =1$，而对其他实现来说，$\lim_{\rightarrow + \infty} \pi_t(\omega) =0$。

现在让我们记住 $\{\pi_t\}_{t=0}^\infty$ 是一个鞅，并应用迭代期望法则。

迭代期望法则意味着

$$
E_t \pi_{t+j}  = \pi_t
$$

特别是

$$
E_{-1} \pi_{t+j} = \pi_{-1}
$$

将上述公式应用于 $\pi_\infty$，我们得到

$$
E_{-1} \pi_\infty(\omega) = \pi_{-1}
$$

这里的数学期望 $E_{-1}$ 是相对于概率测度 ${\textrm{Prob}(\Omega)}$ 计算的。

由于 $\pi_\infty(\omega)$ 只能取 1 和 0 两个值，我们知道对某个 $\lambda \in [0,1]$

$$
{\textrm{Prob}}\Bigl(\pi_\infty(\omega) = 1\Bigr) = \lambda, \quad {\textrm{Prob}}\Bigl(\pi_\infty(\omega) = 0\Bigr) = 1- \lambda
$$

因此

$$
E_{-1} \pi_\infty(\omega) = \lambda \cdot 1 + (1-\lambda) \cdot 0 = \lambda
$$

将此方程与方程(20)结合，我们推断出 ${\textrm{Prob}(\Omega)}$ 赋予 $\pi_\infty(\omega)$ 为 1 的概率必须是 $\pi_{-1}$。

因此，在工人的主观分布下，$\pi_{-1}$ 的样本路径

$\{\pi_t\}$将有$\pi_{-1}$的样本路径逐点收敛到$1$，有$1 - \pi_{-1}$的样本路径逐点收敛到$0$。

### 一些模拟

让我们通过一些模拟来观察鞅收敛定理在工人主观分布下的学习模型中的表现。

让我们模拟$\left\{ \pi_{t}\right\} _{t=0}^{T}$和$\left\{ w_{t}\right\} _{t=0}^{T}$的路径，其中对于每个$t\geq0$，$w_t$从主观分布中抽取：

$$
\pi_{t-1}f\left(w_{t}\right)+\left(1-\pi_{t-1}\right)g\left(w_{t}\right)
$$

我们将绘制大量样本路径。

```{code-cell} ipython3
@jit
def martingale_simulate(π0, N=5000, T=200):

    π_path = np.empty((N,T+1))
    w_path = np.empty((N,T))
    π_path[:,0] = π0

    for n in range(N):
        π = π0
        for t in range(T):
            # draw w
            if np.random.rand() <= π:
                w = np.random.beta(F_a, F_b)
            else:
                w = np.random.beta(G_a, G_b)
            π = π*f(w)/g(w)/(π*f(w)/g(w) + 1 - π)
            π_path[n,t+1] = π
            w_path[n,t] = w

    return π_path, w_path

def fraction_0_1(π0, N, T, decimals):

    π_path, w_path = martingale_simulate(π0, N=N, T=T)
    values, counts = np.unique(np.round(π_path[:,-1], decimals=decimals), return_counts=True)
    return values, counts

def create_table(π0s, N=10000, T=500, decimals=2):

    outcomes = []
    for π0 in π0s:
        values, counts = fraction_0_1(π0, N=N, T=T, decimals=decimals)
        freq = counts/N
        outcomes.append(dict(zip(values, freq)))
    table = pd.DataFrame(outcomes).sort_index(axis=1).fillna(0)
    table.index = π0s
    return table

# simulate
T = 200
π0 = .5

π_path, w_path = martingale_simulate(π0=π0, T=T, N=10000)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for i in range(100):
    ax.plot(range(T+1), π_path[i, :])

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\pi_t$')
plt.show()
```

上图表明

* 每条路径都收敛

* 一些路径收敛到 $1$

* 一些路径收敛到 $0$

* 没有路径收敛到不等于 $0$ 或 $1$ 的极限点

如下图所示，通过观察不同小 $t$ 值时 $\pi_t$ 的跨集合分布，可以看出收敛实际上发生得相当快。

```{code-cell} ipython3
fig, ax = plt.subplots()
for t in [1, 10, T-1]:
    ax.hist(π_path[:,t], bins=20, alpha=0.4, label=f'T={t}')

ax.set_ylabel('count')
ax.set_xlabel(r'$\pi_T$')
ax.legend(loc='lower right')
plt.show()
```

显然，到 $t = 199$ 时，$\pi_t$ 已收敛到 $0$ 或 $1$。

收敛到 $1$ 的路径比例是 $.5$

收敛到 $0$ 的路径比例也是 $.5$。

这个比例 $.5$ 是否让你想起什么？

是的：它等于我们用来生成整个集合中每个序列的初始值 $\pi_0 = .5$。

那么让我们把 $\pi_0$ 改为 $.3$，看看对于不同的 $t$ 值，$\pi_t$ 集合的分布会发生什么变化。

```{code-cell} ipython3
# 模拟
T = 200
π0 = .3

π_path3, w_path3 = martingale_simulate(π0=π0, T=T, N=10000)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for t in [1, 10, T-1]:
    ax.hist(π_path3[:,t], bins=20, alpha=0.4, label=f'T={t}')

ax.set_ylabel('计数')
ax.set_xlabel(r'$\pi_T$')
ax.legend(loc='upper right')
plt.show()
```

对于前面假设 $\pi_0 = .5$ 的集成模型，下图展示了两条 $w_t$ 路径和产生它们的 $\pi_t$ 序列。

注意其中一条路径涉及系统性更高的 $w_t$ 值，这些结果将 $\pi_t$ 向上推动。

在模拟早期的随机抽样会将主观分布推向更频繁地从 $F$ 中抽样的方向，这会将 $\pi_t$ 推向 $0$。

```{code-cell} ipython3
fig, ax = plt.subplots()
for i, j in enumerate([10, 100]):
    ax.plot(range(T+1), π_path[j,:], color=colors[i], label=fr'$\pi$_path, {j}-th simulation')
    ax.plot(range(1,T+1), w_path[j,:], color=colors[i], label=fr'$w$_path, {j}-th simulation', alpha=0.3)

ax.legend(loc='upper right')
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\pi_t$')
ax2 = ax.twinx()
ax2.set_ylabel("$w_t$")
plt.show()
```

## 通过从主观条件密度中抽取的路径验证初始先验

现在让我们使用Python代码生成一个表格，来验证我们之前关于点态极限$\pi_{\infty}(\omega)$的概率分布的说法。

我们将使用模拟来生成这个分布的直方图。

在下表中，粗体显示的左列报告了$\pi_{-1}$的假设值。

第二列报告了在$N = 10000$次模拟中，对于每次模拟在终止日期$T=500$时$\pi_{t}$收敛到$0$的比例。

第三列报告了在$N = 10000$次模拟中，对于每次模拟在终止日期$T=500$时$\pi_{t}$收敛到$1$的比例。

```{code-cell} ipython3
# create table
table = create_table(list(np.linspace(0,1,11)), N=10000, T=500)
table
```

对于$\pi_{t}$收敛到1的模拟比例确实总是接近$\pi_{-1}$，这与预期一致。

## 深入分析

为了理解$\pi_t$的局部动态行为，研究$\pi_t$在给定$\pi_{t-1}$条件下的方差是很有启发性的。

在主观分布下，这个条件方差定义为

$$
\sigma^2(\pi_t | \pi_{t-1})  = \int \Bigl[  { \pi_{t-1} f(w) \over \pi_{t-1} f(w) + (1-\pi_{t-1})g(w)  } - \pi_{t-1} \Bigr]^2
 \Bigl[ \pi_{t-1} f(w) + (1-\pi_{t-1})g(w) \Bigr]  d w
$$

我们可以使用蒙特卡洛模拟来近似这个条件方差。

我们对$\pi_{t-1} \in [0,1]$的网格点进行近似计算。

然后绘制图表。

```{code-cell} ipython3
@jit
def compute_cond_var(pi, mc_size=int(1e6)):
    # create monte carlo draws
    mc_draws = np.zeros(mc_size)

    for i in prange(mc_size):
        if np.random.rand() <= pi:
            mc_draws[i] = np.random.beta(F_a, F_b)
        else:
            mc_draws[i] = np.random.beta(G_a, G_b)

    dev = pi*f(mc_draws)/(pi*f(mc_draws) + (1-pi)*g(mc_draws)) - pi
    return np.mean(dev**2)

pi_array = np.linspace(0, 1, 40)
cond_var_array = []

for pi in pi_array:
    cond_var_array.append(compute_cond_var(pi))

fig, ax = plt.subplots()
ax.plot(pi_array, cond_var_array)
ax.set_xlabel(r'$\pi_{t-1}$')
ax.set_ylabel(r'$\sigma^{2}(\pi_{t}\vert \pi_{t-1})$')
plt.show()
```

条件方差作为 $\pi_{t-1}$ 的函数的形状，能够告诉我们 $\{\pi_t\}$ 样本路径的行为特征。

注意当 $\pi_{t-1}$ 接近 0 或 1 时，条件方差趋近于 0。

只有当代理几乎确定 $w_t$ 是从 $F$ 分布中抽取的，或者几乎确定是从 $G$ 分布中抽取的时候，条件方差才接近于零。

## 相关讲座

本讲座致力于建立一些有用的基础设施，这将有助于我们理解在{doc}`这个讲座 <odu>`、{doc}`这个讲座 <wald_friedman>` 和{doc}`这个讲座 <navy_captain>` 中描述的结果的推理基础。

