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
  title: 似然比过程和贝叶斯学习
  headings:
    Overview: 概述
    The setting: 背景设置
    Likelihood ratio processes and Bayes’ law: 似然比过程和贝叶斯定律
    Likelihood ratio processes and Bayes’ law::A recursive formula: 递归公式
    Another timing protocol: 另一种时序协议
    Behavior of posterior probability $\{\pi_t\}$ under subjective probability distribution: 后验概率 $\{\pi_t\}$ 在主观概率分布下的行为
    Behavior of posterior probability $\{\pi_t\}$ under subjective probability distribution::Mechanical details again: 再谈机械细节
    Behavior of posterior probability $\{\pi_t\}$ under subjective probability distribution::Some simulations: 一些模拟
    Initial prior is verified by paths drawn from subjective conditional densities: 通过从主观条件密度中抽取的路径验证初始先验
    Drilling down a little bit: 深入分析
    Related lectures: 相关讲座
    Exercises: 练习
---

(likelihood_bayes)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 似然比过程和贝叶斯学习

## 概述

本讲解描述了*似然比过程*在*贝叶斯学习*中所扮演的角色。

如同 {doc}`likelihood_ratio_process` 中所述，我们将使用 {doc}`exchangeable` 中的一个简单统计设置。

我们将重点关注似然比过程和*先验*概率如何决定*后验*概率。

我们将推导出一个便利的递归公式，表示今天的后验概率是昨天后验概率和今天似然过程乘法增量的函数。

我们还将介绍该公式的一个有用推广，该推广将今天的后验概率表示为初始先验和今天似然比过程实现值的函数。

我们将研究在我们的设置中，贝叶斯学习者如何最终学习到生成数据的概率分布，这个结果建立在 {doc}`likelihood_ratio_process` 中研究的似然比过程的渐近行为之上。

我们还将深入探讨贝叶斯学习者的心理，研究其主观信念下的动态变化。

本讲座提供了技术性结果，这些结果是 {doc}`odu`、{doc}`wald_friedman` 和 {doc}`navy_captain` 中将要研究的结果的基础。

我们先加载一些Python模块。

```{code-cell} ipython3
:hide-output: false

FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from numba import vectorize, jit
from math import gamma
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit, logit
from scipy.stats import ks_2samp


import seaborn as sns
colors = sns.color_palette()

rng = np.random.default_rng(142857)
```

## 背景设置

我们首先回顾 {doc}`likelihood_ratio_process` 中的设置，这也是我们在这里采用的设置。

一个非负随机变量 $W$ 具有两个概率密度函数之一，要么是 $f$，要么是 $g$。

在时间开始之前，自然界一劳永逸地决定是从 $f$ 还是从 $g$ 中进行一系列独立同分布的抽样。

我们用 $q$ 表示自然界一劳永逸选择的密度，所以 $q$ 要么是 $f$ 要么是 $g$，且是永久性的。

自然界知道它永久从哪个密度中抽样，但我们这些观察者并不知道。

我们知道 $f$ 和 $g$ 这两个密度，但不知道自然界选择了哪一个。

但我们想要知道。

为此，我们使用观测数据。

我们观察到一个序列 $w_0, w_1, \ldots$，它是从 $f$ 或 $g$ 中独立同分布抽取的样本。

我们将第一次抽样的时间记为 $t = 0$，将先验的时间记为 $t = -1$，这与 {doc}`exchangeable` 中一致。

{doc}`likelihood_ratio_process` 将第一次抽样的时间记为 $t = 1$。

我们想要利用这些观测来推断自然界选择了 $f$ 还是 $g$。

似然比过程是完成这项任务的有用工具。

首先，我们定义似然比过程的关键组成部分，即时间 $t$ 的似然比，它是一个随机变量：

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq0.
$$

我们假设 $f$ 和 $g$ 在随机变量 $W$ 可能实现值的相同区间上都赋予正概率。

这意味着在 $g$ 密度下，$\ell (w_t)= \frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$ 显然是一个均值为1的非负随机变量。

序列的**似然比过程**

$\left\{ w_{t}\right\} _{t=0}^{\infty}$ 定义如下：

$$
L\left(w^{t}\right)=\prod_{s=0}^{t} \ell (w_s),
$$

其中 $w^t=\{ w_0,\dots,w_t\}$ 是直到时间 $t$ （包括 $t$）的观测历史。

有时为简便起见，我们会写作

$$
L_t =  L(w^t) = \frac{f(w^t)}{g(w^t)}
$$ 

这里我们使用如下约定：
$f(w^t) = f(w_0) f(w_1) \ldots f(w_t)$ 且 $g(w^t) = g(w_0) g(w_1) \ldots g(w_t)$。

注意，似然过程满足以下*递归*或*乘法分解*：

$$
L(w^t) = \ell (w_t) L (w^{t-1}) ,
$$

其中 $L(w^{-1}) \equiv 1$。

似然比及其对数是使用 Neyman 和 Pearson 经典频率派方法进行推断的关键工具 {cite}`Neyman_Pearson`。

我们将再次使用来自 {doc}`likelihood_ratio_process` 的以下 Python 代码，该代码将 $f$ 和 $g$ 评估为两个不同的贝塔分布，然后通过从*某个*概率分布（例如，从 $g$ 生成的 IID 序列）生成序列 $w^t$ 来计算和模拟相关的似然比过程。

```{code-cell} ipython3
# Parameters in the two beta distributions.
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x**(a-1) * (1 - x)**(b-1)

# The two density functions.
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))
```

```{code-cell} ipython3
@jit
def simulate(a, b, rng, T=50, N=500):
    '''
    生成N组T个似然比观测值，
    以N x T矩阵形式返回。第s列保存 ℓ(w_s)，s = 0, ..., T-1。
    '''

    l_arr = np.empty((N, T))

    for i in range(N):

        for j in range(T):
            w = rng.beta(a, b)
            l_arr[i, j] = f(w) / g(w)

    return l_arr
```

我们还将使用以下Python代码来准备一些信息丰富的模拟。

`l_seq_g` 和 `l_seq_f` 的第 $t$ 列保存 $L(w^t)$。

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b, rng, N=50000)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

```{code-cell} ipython3
l_arr_f = simulate(F_a, F_b, rng, N=50000)
l_seq_f = np.cumprod(l_arr_f, axis=1)
```

(lb_bayes_law)=
## 似然比过程和贝叶斯定律

设 $\pi_{-1} \in [0,1]$ 是贝叶斯统计学家对自然生成的 $w^t$ 序列是来自分布 $f$ 的独立同分布抽样的先验概率。

* 这里的"概率"应被理解为总结或表达主观观点的一种方式
* 它*不*意味着样本量无限增长时的预期相对频率

设 $\pi_{t}$ 是定义如下的贝叶斯后验概率：

$$
\pi_{t} = {\rm Prob}(q=f|w^{t})
$$ (eq:defbayesposterior)

似然比过程是后验概率 $\pi_t$ 演化公式中的主要角色，这是*贝叶斯定律*的一个实例。

让我们推导 $\pi_{t}$ 的两个公式，一个用似然比 $\ell(w_{t})$ 表示，另一个用 $L(w^{t})$ 表示。

首先，我们使用以下符号约定：

* $f(w^{t}) \equiv f(w_0) f(w_1) \cdots f(w_{t})$
* $g(w^{t}) \equiv g(w_0) g(w_1) \cdots g(w_{t})$
* $\pi_{-1} ={\rm Prob}(q=f |\emptyset)$
* $\pi_t = {\rm Prob}(q=f |w^t)$

这里符号 $\emptyset$ 表示"空集"或"无数据"。

在没有数据的情况下，我们的贝叶斯统计学家认为序列 $w^{t}$ 的概率密度是：

$$
{\rm Prob}(w^{t} |\emptyset) = \pi_{-1} f(w^{t})+ (1 -  \pi_{-1}) g(w^{t})
$$

概率定律表明，事件 $A$ 和 $B$ 的联合分布 ${\rm Prob}(AB)$ 与条件分布 ${\rm Prob}(A |B)$ 和 ${\rm Prob}(B |A)$ 之间的关系是：

$$

{\rm Prob}(AB) = {\rm Prob}(A |B) {\rm Prob}(B) = {\rm Prob}(B |A) {\rm Prob}(A) . 
$$ (eq:problawAB)

我们关注的事件是

$$
A = \{q=f\},  \quad B = \{w^{t}\}, \quad
$$

其中大括号$\{\cdot\}$是我们用来表示"事件"的简写。

因此在我们的设定中，概率法则 {eq}`eq:problawAB` 意味着

$$
{\rm Prob}(q=f |w^{t})  {\rm Prob}(w^{t}  |\emptyset) = {\rm Prob}(w^{t} |q=f) {\rm Prob}(q=f  | \emptyset)
$$

或者

$$
\pi_{t} \left[\pi_{-1} f(w^{t}) + (1- \pi_{-1}) g(w^{t})\right] = f(w^{t}) \pi_{-1}
$$

或者

$$
\pi_{t}  = \frac{ f(w^{t}) \pi_{-1} }{\pi_{-1} f(w^{t}) + (1- \pi_{-1}) g(w^{t})}
$$

将上述等式右边的分子和分母都除以$g(w^{t})$得到

```{math}
:label: eq_Bayeslaw1033

\pi_{t}=\frac{\pi_{-1}L\left(w^{t}\right)}{\pi_{-1}L\left(w^{t}\right)+1-\pi_{-1}} .
```

公式 {eq}`eq_Bayeslaw1033` 可以被视为在看到数据批次 $\left\{ w_{s}\right\} _{s=0}^{t}$ 后对先验概率 $\pi_{-1}$ 的一步修正。

公式 {eq}`eq_Bayeslaw1033` 显示了似然比过程 $L\left(w^{t}\right)$ 在确定后验概率 $\pi_{t}$ 中起到的关键作用。

公式 {eq}`eq_Bayeslaw1033` 是理解以下观点的基础：由于似然比过程在 $t \rightarrow + \infty$ 时的行为特征，似然比过程在决定 $\pi_t$ 的极限行为时会主导初始先验 $\pi_{-1}$ 的影响。

### 递归公式

我们可以使用类似的推理方法得到公式 {eq}`eq_Bayeslaw1033` 的递归版本。

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
{\rm Prob}(q=f|w^{t+1}) = \frac{{\rm Prob}(q=f|w^{t}) f(w_{t+1} )} {{\rm Prob}(w_{t+1} | w^t)}
$$

其中

$$
{\rm Prob}(w_{t+1} | w^t) = \pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})
$$

是在给定历史 $w^t$ 条件下 $w_{t+1}$ 的密度。

在给定 $q$ 的条件下，抽样 $w_{t+1}$ 与 $w^t$ 相互独立，因此 $f(w_{t+1})$ 是在 $q = f$ 和 $w^t$ 条件下 $w_{t+1}$ 的密度。

将方程 {eq}`eq:bayes150` 右侧的分子和分母都除以$g(w_{t+1})$得到递归式

```{math}
:label: eq_recur1

\pi_{t+1}=\frac{\pi_{t} \ell(w_{t+1})}{\pi_{t} \ell(w_{t+1})+1-\pi_{t}}
```

其中$\pi_{-1}$是$q = f$的贝叶斯先验概率，即在我们尚未看到任何数据时基于个人或主观判断的关于$q$的信念。

通过迭代方程 {eq}`eq_recur1` 可以推导出公式 {eq}`eq_Bayeslaw1033`。

用对数比率（log odds）表示，递归式 {eq}`eq_recur1` 为

```{math}
:label: eq_logodds

\log \frac{\pi_{t+1}}{1 - \pi_{t+1}} = \log \frac{\pi_{t}}{1 - \pi_{t}} + \log \ell(w_{t+1})
```

而公式 {eq}`eq_Bayeslaw1033` 为

$$
\log \frac{\pi_{t}}{1 - \pi_{t}} = \log \frac{\pi_{-1}}{1 - \pi_{-1}} + \log L(w^t) .
$$

我们的代码对对数比率进行迭代。

直接对 {eq}`eq_recur1` 本身进行迭代会积累舍入误差，一旦 $\pi_t$ 接近 $1$ 或 $0$，就会使其略微超过 $1$ 或低于 $0$。

下面我们定义一个Python函数，该函数根据递归式 {eq}`eq_logodds` 使用似然比$\ell$更新 $q = f$ 的对数比率

```{code-cell} ipython3
@jit
def update(log_odds, l):
    "Update the log odds of q = f using likelihood ratio l"
    return log_odds + np.log(l)
```

为了说明似然比过程如何主导初始先验的影响，我们将绘制图表，展示似然比过程 $L_t$ 的*一条*模拟路径，以及与*相同*似然比过程实现但*不同*初始先验概率 $\pi_{-1}$ 相关联的两条 $\pi_t$ 路径。

首先，我们在Python中设定两个 $\pi_{-1}$ 的值。

```{code-cell} ipython3
π1, π2 = 0.2, 0.8
```

接下来我们为从密度函数$f$中独立同分布抽取的历史数据生成似然比过程$L_t$和后验概率$\pi_t$的路径。

`π_seq_f` 的第 $0$ 列保存先验 $\pi_{-1}$，第 $t+1$ 列保存 $\pi_t$。

```{code-cell} ipython3
T = l_arr_f.shape[1]
log_odds_f = np.empty((2, T+1))
log_odds_f[:, 0] = logit([π1, π2])

for t in range(T):
    for i in range(2):
        log_odds_f[i, t+1] = update(log_odds_f[i, t], l_arr_f[0, t])

π_seq_f = expit(log_odds_f)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 后验路径与对数似然比
    name: fig-posterior-lratio-f
---
fig, ax1 = plt.subplots()

for i in range(2):
    ax1.plot(range(-1, T), π_seq_f[i, :],
    label=fr"$\pi_{{-1}}$={π_seq_f[i, 0]:.1f}", lw=2)

ax1.set_ylabel(r"$\pi_t$")
ax1.set_xlabel(r"$t$")
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(range(T), np.log(l_seq_f[0, :]), '--', color='b', lw=2)
ax2.set_ylabel(r"$\log(L(w^{t}))$")

plt.show()
```

图中的虚线记录了似然比过程的对数 $\log L(w^t)$。

请注意 $y$ 轴上有两个不同的刻度。

现在让我们研究当历史由密度 $g$ 产生的独立同分布抽样构成时会发生什么

```{code-cell} ipython3
T = l_arr_g.shape[1]
log_odds_g = np.empty((2, T+1))
log_odds_g[:, 0] = logit([π1, π2])

for t in range(T):
    for i in range(2):
        log_odds_g[i, t+1] = update(log_odds_g[i, t], l_arr_g[0, t])

π_seq_g = expit(log_odds_g)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 后验路径与对数似然比
    name: fig-posterior-lratio-g
---
fig, ax1 = plt.subplots()

for i in range(2):
    ax1.plot(range(-1, T), π_seq_g[i, :],
        label=fr"$\pi_{{-1}}$={π_seq_g[i, 0]:.1f}", lw=2)

ax1.set_ylabel(r"$\pi_t$")
ax1.set_xlabel(r"$t$")
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(range(T), np.log(l_seq_g[0, :]), '--', color='b', lw=2)
ax2.set_ylabel(r"$\log(L(w^{t}))$")

plt.show()
```

下一个代码单元检验批量公式 {eq}`eq_Bayeslaw1033` 是否能够重现由递归式 {eq}`eq_logodds` 从密度 $f$ 抽样生成的 $\pi_t$ 路径。

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

因此，我们得出结论，似然比过程是公式 {eq}`eq_Bayeslaw1033` 中贝叶斯后验概率的关键组成部分，该后验概率表示自然界从密度$f$中重复抽样得到历史$w^t$的概率。

## 另一种时序协议

让我们研究当自然界在不同的时序协议下生成历史$w^t = \{w_0, w_1, \dots, w_t\}$时，后验概率$\pi_t = {\rm Prob}(q=f|w^{t})$的表现。

到目前为止，我们假设在时间$0$之前，自然界以某种方式选择从$f$或$g$中进行IID序列抽样来得到$w^t$。

自然界关于是从$f$还是从$g$中抽样的决定因此是*永久性的*。

现在我们假设一个不同的时序协议，在*每个*时期$t =0, 1, \ldots$之前，自然界：

* 抛一个权重为$\alpha$的硬币，然后
* 如果抛出"正面"就从$f$中抽样
* 如果抛出"反面"就从$g$中抽样

在这个时序协议下，自然界*既不*永久地从$f$抽样*也不*永久地从$g$抽样，所以认为自然界*永久地*从其中之一进行IID抽样的统计学家是错误的。

* 事实上，自然界是从混合分布 $h(w|\alpha) = \alpha f(w) + (1-\alpha) g(w)$ 中进行IID抽样——当$\alpha \in (0,1)$时，这个密度既不是$f$也不是$g$

统计学家的后验概率 $\pi_t$ 仍然是他认为 $q = f$ 的概率，正如 {eq}`eq:defbayesposterior` 中所定义的那样。

它*不是*对混合参数 $\alpha$ 的估计。

统计学家的模型是错误指定的，$f$ 和 $g$ 相对于 $h$ 的 Kullback–Leibler 散度决定了 $\pi_t$ 的极限。

根据大数定律，$\frac{1}{t+1} \log L(w^t)$ 以概率1收敛于

$$
E_h[\log \ell(w)] = E_h\left[\log \frac{h(w|\alpha)}{g(w)}\right] - E_h\left[\log \frac{h(w|\alpha)}{f(w)}\right] = KL(h, g) - KL(h, f) ,
$$

其中 $E_h$ 表示在 $h(\cdot|\alpha)$ 下的期望，$KL(h, f) = \int h(w|\alpha) \log \frac{h(w|\alpha)}{f(w)} dw$；参见 {doc}`likelihood_ratio_process` 中的 {ref}`KL_link`。

所以当 $KL(h, f) < KL(h, g)$ 时，$\log L(w^t) \rightarrow + \infty$，$\pi_t \rightarrow 1$。

而当 $KL(h, f) > KL(h, g)$ 时，$\log L(w^t) \rightarrow - \infty$，$\pi_t \rightarrow 0$。

统计学家会逐渐相信 $\{f, g\}$ 中在 Kullback–Leibler 散度意义上更接近 $h$ 的那一个。

让我们编写一些 Python 代码，在 $\alpha = 0.5$ 时验证这一点。

```{code-cell} ipython3
@jit
def simulate_mixture_path(α, T, rng):
    """
    在混合时序协议下模拟 w_0, ..., w_{T-1} 的抽样。
    """
    w = np.empty(T)
    for t in range(T):
        if rng.random() < α:
            w[t] = rng.beta(F_a, F_b)
        else:
            w[t] = rng.beta(G_a, G_b)
    return w


α = 0.5
T_mix = 200
rng = np.random.default_rng(142857)
w_mix = simulate_mixture_path(α, T_mix, rng)
```

我们针对三个先验 $\pi_{-1}$ 计算 $\pi_t$ 的路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 混合模型下的后验
    name: fig-posterior-mixture
---
log_L_mix = np.cumsum(np.log(f(w_mix) / g(w_mix)))

fig, ax = plt.subplots(figsize=(10, 6))
for i, π_init in enumerate([0.25, 0.5, 0.75]):
    π_path = expit(logit(π_init) + log_L_mix)
    ax.plot(range(-1, T_mix), np.concatenate(([π_init], π_path)),
            label=fr'$\pi_{{-1}} = ${π_init:.2f}', color=colors[i], linewidth=2)

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\pi_t$')
ax.legend()
plt.show()
```

显然，$\pi_t$ 收敛到1。

让我们检查 $f$ 和 $g$ 相对于 $h$ 的 {ref}`KL散度 <rel_entropy>`。

```{code-cell} ipython3
def compute_KL(f, g):
    """
    计算KL散度 KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val


h = lambda w: α * f(w) + (1 - α) * g(w)
KL_hf, KL_hg = compute_KL(h, f), compute_KL(h, g)

print(f'KL(h, f) = {KL_hf:.3f}\nKL(h, g) = {KL_hg:.3f}')
```

由于 $KL(h, f) < KL(h, g)$，$f$ 更接近 $h$，$\log L(w^t) \rightarrow + \infty$，因此对每个 $\pi_{-1} \in (0,1)$，$\pi_t \rightarrow 1$。

{doc}`mix_model` 进一步研究了这个被错误指定的统计学家的问题，包括使 $KL(h, f) = KL(h, g)$（此时 $\pi_t$ 不收敛）的 $\alpha$ 值。

它还研究了一个知道自然界每期都进行混合抽样，并使用贝叶斯法则来学习 $\alpha$ 的统计学家的情形。

## 后验概率 $\{\pi_t\}$ 在主观概率分布下的行为

我们将通过简要研究贝叶斯学习者在贝叶斯法则产生的主观信念 $\pi_t$ 下期望学到什么来结束本讲。

这将为我们应用贝叶斯法则作为学习理论提供一些视角。

我们将看到，在每个时刻 $t$，贝叶斯学习者知道他将会感到惊讶。

但他预期新信息不会导致他改变信念。

而且在他的主观信念下，平均来说确实不会改变。

我们将继续在这样的设定下讨论：McCall工人知道他的工资连续抽样要么来自 $F$ 要么来自 $G$，但不知道自然界在时间 $0$ 之前已经一次性地选择了这两个分布中的哪一个。

让我们回顾、重申并重新整理我们在上文和相关讲座中遇到的一些公式。

工人的初始信念导致了一个关于潜在无限序列抽样 $w_0, w_1, \ldots $ 的联合概率分布。

贝叶斯定律仅仅是概率法则的应用，用于计算第 $t$ 次抽样 $w_t$ 在已知 $[w_0, \ldots, w_{t-1}]$ 条件下的条件分布。

在我们的工人对自然选择分布 $F$ 赋予主观概率 $\pi_{-1}$ 后，我们实际上从一开始就假设决策者*知道*过程 $\{w_t\}_{t=0}^\infty$ 的联合分布。

我们假设工人也知道概率论的法则。

一个值得尊重的观点是，贝叶斯定律与其说是一个学习理论，不如说是一个关于信息流入对决策者（他认为从一开始就知道真相，即联合概率分布）的影响的陈述。

### 再谈机械细节

在时间 $0$ *之前*抽取工资报价时，工人将概率 $\pi_{-1} \in (0,1)$ 赋予分布 $F$。

在时间 $0$ 抽取工资之前，工人因此认为 $w_0$ 的密度是

$$
h(w_0|\pi_{-1}) = \pi_{-1} f(w_0) + (1-\pi_{-1}) g(w_0).
$$

这里以及下文中，$h(w|\pi) \equiv \pi f(w) + (1 - \pi) g(w)$ 表示以权重 $\pi$ 对 $f$ 加权的 $f$ 与 $g$ 的混合分布。

回忆一下，$q \in \{ f, g\}$ 表示自然是永久地从分布 $f$ 还是从分布 $g$ 中抽样。

在绘制$w_0$后，工人使用贝叶斯定理推导出后验概率$\pi_0 = {\rm Prob}({q = f | w_0})$（即密度为$f(w)$的概率）为：

$$
\pi_0 = { \pi_{-1} f(w_0) \over \pi_{-1} f(w_0) + (1-\pi_{-1}) g(w_0)} .
$$

更一般地，在进行第$t$次抽取并观察到$w_t, w_{t-1}, \ldots, w_0$后，工人认为$w_{t+1}$是从分布$F$中抽取的概率为：

$$
\pi_t = \pi_t(w_t | \pi_{t-1}) \equiv { \pi_{t-1} f(w_t)/g(w_t) \over \pi_{t-1} f(w_t)/g(w_t) + (1-\pi_{t-1})}
$$ (eq:like44)

或

$$
\pi_t=\frac{\pi_{t-1} \ell(w_t)}{\pi_{t-1} \ell(w_t)+1-\pi_{t-1}}
$$

而在给定$w_t, w_{t-1}, \ldots, w_0$条件下$w_{t+1}$的密度为：

$$
h(w_{t+1}|\pi_{t}) = \pi_{t} f(w_{t+1}) + (1-\pi_{t}) g(w_{t+1}) .
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

因此过程$\pi_t$是一个*鞅*。

事实上，它是一个*有界鞅*，因为每个$\pi_t$作为概率都在0和1之间。

在上述等式串的第一行中，第一个方括号中的项就是作为$w_{t}$函数的$\pi_t$，而第二个方括号中的项是在给定$w_{t-1}, \ldots , w_0$，或等价地在$w_{t-1}, \ldots , w_0$的*充分统计量*$\pi_{t-1}$的条件下$w_{t}$的密度。

注意这里我们是在括号中第二项所描述的*主观*密度下计算 $E(\pi_t | \pi_{t-1})$。

因为 $\{\pi_t\}$ 是一个有界鞅序列，根据*鞅收敛定理*，$\pi_t$ 几乎必然收敛到 $[0,1]$ 中的一个随机变量。

实际上，这意味着概率为1的样本路径 $\{\pi_t\}_{t=0}^\infty$ 是收敛的。

根据定理，不同的样本路径可以收敛到不同的极限值。

因此，让 $\{\pi_t(\omega)\}_{t=0}^\infty$ 表示由特定 $\omega \in \Omega$ 索引的特定样本路径。

我们可以认为自然从概率分布 ${\textrm{Prob}}(\Omega)$ 中抽取一个 $\omega \in \Omega$，然后生成该过程的单个实现（或_模拟_）$\{\pi_t(\omega)\}_{t=0}^\infty$。

当 $t \rightarrow +\infty$ 时，$\{\pi_t(\omega)\}_{t=0}^\infty$ 的极限点是一个随机变量的实现，这个随机变量是当我们从 $\Omega$ 中采样 $\omega$ 并构造 $\{\pi_t(\omega)\}_{t=0}^\infty$ 的重复抽样时产生的。

令

$$
\pi_\infty(\omega) = \lim_{t \rightarrow + \infty} \pi_t(\omega).
$$

鞅收敛定理并没有告诉我们 $\pi_t(\omega)$ 会收敛到哪里。

有两个事实能解决这个问题。

第一，$\pi_t$ 是事件 $q = f$ 的指示函数的条件期望 $E[\mathbf{1}\{q = f\} \,|\, w^t]$。

第二，观察到无限历史 $w_0, w_1, \ldots$ 的观察者能够确定 $q$。

根据强大数定律，当 $q = f$ 时，$\frac{1}{t+1} \log L(w^t)$ 收敛到 $E_f[\log \ell(w)] = KL(f, g) > 0$；当 $q = g$ 时，它收敛到 $E_g[\log \ell(w)] = -KL(g, f) < 0$。

杜布鞅收敛定理 {cite}`doob1953` 随后表明 $\pi_t$ 以概率一收敛到 $E[\mathbf{1}\{q = f\} \,|\, w_0, w_1, \ldots] = \mathbf{1}\{q = f\}$。

因此，对于自然抽取 $f$ 的实现，$\lim_{t \rightarrow + \infty} \pi_t(\omega) =1$，而对于自然抽取 $g$ 的实现，$\lim_{t \rightarrow + \infty} \pi_t(\omega) =0$。

{ref}`lb_drill_down` 一节和练习 {ref}`likelihood_bayes_ex2` 给出了 $\pi_\infty(\omega) \in \{0, 1\}$ 的第二种证明。

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
$$ (eq:expect_pi_infty)

这里的数学期望 $E_{-1}$ 是相对于概率测度 ${\textrm{Prob}(\Omega)}$ 计算的。

由于 $\pi_\infty(\omega)$ 只能取 1 和 0 两个值，我们知道对某个 $\lambda \in [0,1]$

$$
{\textrm{Prob}}\Bigl(\pi_\infty(\omega) = 1\Bigr) = \lambda, \quad {\textrm{Prob}}\Bigl(\pi_\infty(\omega) = 0\Bigr) = 1- \lambda
$$

因此

$$
E_{-1} \pi_\infty(\omega) = \lambda \cdot 1 + (1-\lambda) \cdot 0 = \lambda
$$

将此方程与方程 {eq}`eq:expect_pi_infty` 结合，我们推断出 ${\textrm{Prob}(\Omega)}$ 赋予 $\pi_\infty(\omega)$ 为 1 的概率必须是 $\pi_{-1}$。

从 $\pi_\infty = \mathbf{1}\{q = f\}$ 和 ${\rm Prob}(q = f) = \pi_{-1}$ 出发，也可以得到相同的结果。

因此，在工人的主观分布下，$\{\pi_t\}$将有$\pi_{-1}$的样本路径逐点收敛到$1$，有$1 - \pi_{-1}$的样本路径逐点收敛到$0$。

### 一些模拟

让我们通过一些模拟来观察鞅收敛定理在工人主观分布下的学习模型中的表现。

让我们模拟$\left\{ \pi_{t}\right\} _{t=-1}^{T-1}$和$\left\{ w_{t}\right\} _{t=0}^{T-1}$的路径，其中对于每个$t\geq0$，$w_t$从主观分布中抽取：

$$
h(w_t|\pi_{t-1}) = \pi_{t-1}f\left(w_{t}\right)+\left(1-\pi_{t-1}\right)g\left(w_{t}\right)
$$

这种抽样方式使得历史 $w^t$ 具有与下述方式相同的联合分布：先一次性地以 ${\rm Prob}(q = f) = \pi_{-1}$ 的概率抽取 $q$，然后从 $q$ 中独立同分布地抽取 $w_0, w_1, \ldots$：

$$
\prod_{s=0}^{t} h(w_s|\pi_{s-1}) = \pi_{-1} f(w^t) + (1 - \pi_{-1}) g(w^t) .
$$

练习 {ref}`likelihood_bayes_ex1` 要求你证明并验证这一点。

以*固定*权重 $\alpha$ 进行逐期混合会得到不同的联合密度 $\prod_{s=0}^{t} h(w_s|\alpha)$；这正是 {doc}`mix_model` 中的时序方案。

我们将绘制大量样本路径。

我们的代码跟踪 $q = f$ 的对数几率，并将其转换为概率。

```{code-cell} ipython3
@jit
def martingale_simulate(π_prior, rng, N=5000, T=200):
    """
    Simulate N paths of w_0, ..., w_{T-1} from the subjective distribution.
    Column 0 of π_path holds the prior π_{-1}; column t+1 holds π_t.
    """
    π_path = np.empty((N,T+1))
    w_path = np.empty((N,T))
    π_path[:,0] = π_prior

    for n in range(N):
        log_odds = np.log(π_prior) - np.log(1 - π_prior)
        π = π_prior
        for t in range(T):
            # draw w from h(w | π)
            if rng.random() <= π:
                w = rng.beta(F_a, F_b)
            else:
                w = rng.beta(G_a, G_b)
            log_odds = update(log_odds, f(w) / g(w))
            π = 1 / (1 + np.exp(-log_odds))
            π_path[n,t+1] = π
            w_path[n,t] = w

    return π_path, w_path

def fraction_0_1(π_prior, rng, N, T, decimals):

    π_path, w_path = martingale_simulate(π_prior, rng, N=N, T=T)
    values, counts = np.unique(np.round(π_path[:,-1], decimals=decimals), return_counts=True)
    return values, counts

def create_table(π_priors, rng, N=10000, T=500, decimals=2):

    outcomes = []
    for π_prior in π_priors:
        values, counts = fraction_0_1(π_prior, rng, N=N, T=T, decimals=decimals)
        freq = counts/N
        outcomes.append(dict(zip(values, freq)))
    table = pd.DataFrame(outcomes).sort_index(axis=1).fillna(0)
    table.index = π_priors
    return table

# simulate
T = 200
π_prior = .5

π_path, w_path = martingale_simulate(π_prior=π_prior, rng=rng, T=T, N=10000)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for i in range(100):
    ax.plot(range(-1, T), π_path[i, :], lw=2)

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\pi_t$')
plt.show()
```

上图表明

* 每条路径都收敛

* 一些路径收敛到 $1$

* 一些路径收敛到 $0$

* 没有路径收敛到不等于 $0$ 或 $1$ 的极限点

如下图所示，通过观察 $t = 1, 10$ 以及 $199$ 时 $\pi_t$ 的跨集合分布，可以看出收敛实际上发生得相当快。

```{code-cell} ipython3
fig, ax = plt.subplots()
for t in [1, 10, T-1]:
    ax.hist(π_path[:, t+1], bins=20, alpha=0.4, label=f'$t={t}$')

ax.set_ylabel('count')
ax.set_xlabel(r'$\pi_t$')
ax.legend(loc='lower right')
plt.show()
```

下一个单元格报告了 $\pi_{199}$ 高于 $0.99$、低于 $0.01$ 以及介于两者之间的路径所占的比例。

```{code-cell} ipython3
def report_fractions(π_path):
    π_T = π_path[:, -1]
    print(f"fraction with π_T > 0.99:        {np.mean(π_T > 0.99):.4f}")
    print(f"fraction with π_T < 0.01:        {np.mean(π_T < 0.01):.4f}")
    print(f"fraction with 0.01 <= π_T <= 0.99: {np.mean((π_T >= 0.01) & (π_T <= 0.99)):.4f}")

report_fractions(π_path)
```

显然，到 $t = 199$ 时，几乎每条路径的 $\pi_t$ 都已收敛到 $0$ 或 $1$。

收敛到 $1$ 的路径比例接近 $.5$。

收敛到 $0$ 的路径比例也接近 $.5$。

这个比例 $.5$ 是否让你想起什么？

是的：它等于我们用来生成整个集合中每个序列的初始值 $\pi_{-1} = .5$。

那么让我们把 $\pi_{-1}$ 改为 $.3$，看看对于不同的 $t$ 值，$\pi_t$ 集合的分布会发生什么变化。

```{code-cell} ipython3
# simulate
T = 200
π_prior = .3

π_path3, w_path3 = martingale_simulate(π_prior=π_prior, rng=rng, T=T, N=10000)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for t in [1, 10, T-1]:
    ax.hist(π_path3[:, t+1], bins=20, alpha=0.4, label=f'$t={t}$')

ax.set_ylabel('count')
ax.set_xlabel(r'$\pi_t$')
ax.legend(loc='upper right')
plt.show()

report_fractions(π_path3)
```

对于前面假设 $\pi_{-1} = .5$ 的集成模型，下图展示了两条 $w_t$ 路径和产生它们的 $\pi_t$ 序列。

我们选取 $\pi_{199}$ 最大的路径和 $\pi_{199}$ 最小的路径。

抽样 $w \in [0.524, 0.999]$ 满足 $\ell(w) < 1$，会将 $\pi_t$ 向下推（参见 {doc}`exchangeable`）。

$\pi_t$ 趋向 $1$ 的路径，其 $w_t$ 值系统性地低于 $\pi_t$ 趋向 $0$ 的路径。

在模拟早期的随机抽样会将主观分布推向更频繁地从 $F$ 中抽样的方向，这会将 $\pi_t$ 推向 $1$。

由于预测抽样与一次性抽取 $q$ 具有相同的联合分布，$\pi_t \rightarrow 1$ 的路径看起来就像是从 $f$ 中独立同分布抽取的一个序列，而 $\pi_t \rightarrow 0$ 的路径看起来就像是从 $g$ 中独立同分布抽取的一个序列。

```{code-cell} ipython3
j_up, j_down = np.argmax(π_path[:, -1]), np.argmin(π_path[:, -1])

fig, ax = plt.subplots()
ax2 = ax.twinx()
for i, j in enumerate([j_up, j_down]):
    ax.plot(range(-1, T), π_path[j, :], color=colors[i],
            label=fr'$\pi_t$, path {j}', lw=2)
    ax2.plot(range(T), w_path[j, :], color=colors[i],
             label=fr'$w_t$, path {j}', alpha=0.3, lw=2)

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\pi_t$')
ax2.set_ylabel(r'$w_t$')
handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax.legend(handles=handles, loc='center right')
plt.show()

for name, j in [('π_t → 1', j_up), ('π_t → 0', j_down)]:
    print(f"path with {name}: mean of w_t = {w_path[j, :].mean():.3f}")

up = π_path[:, -1] > 0.5
print(f"all paths with π_199 > 0.5: mean of w_t = {w_path[up, :].mean():.3f}")
print(f"all paths with π_199 < 0.5: mean of w_t = {w_path[~up, :].mean():.3f}")
print(f"means of f and g: {F_a / (F_a + F_b):.3f}, {G_a / (G_a + G_b):.3f}")
```

在所有路径中，$\pi_{199} > 0.5$ 的路径上 $w_t$ 的均值接近 $f$ 的均值，而 $\pi_{199} < 0.5$ 的路径上 $w_t$ 的均值接近 $g$ 的均值。

## 通过从主观条件密度中抽取的路径验证初始先验

现在让我们使用 Python 代码生成一个表格，来验证我们之前关于点态极限 $\pi_{\infty}(\omega)$ 的概率分布的说法。

我们将使用模拟来生成这个分布的直方图。

在下表中，粗体显示的左列报告了 $\pi_{-1}$ 的假设值。

第二列报告了在 $N = 10000$ 次模拟中，对于每次模拟在终止日期 $T=500$ 时 $\pi_{t}$ 收敛到 $0$ 的比例。

第三列报告了在 $N = 10000$ 次模拟中，对于每次模拟在终止日期 $T=500$ 时 $\pi_{t}$ 收敛到 $1$ 的比例。

```{code-cell} ipython3
# create table
table = create_table(list(np.linspace(0,1,11)), rng, N=10000, T=500)
table
```

对于 $\pi_{t}$ 收敛到 1 的模拟比例确实总是接近 $\pi_{-1}$，这与预期一致。


(lb_drill_down)=
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
def compute_cond_var(π, rng, mc_size=int(1e6)):
    # 生成蒙特卡洛抽样
    mc_draws = np.zeros(mc_size)

    for i in range(mc_size):
        if rng.random() <= π:
            mc_draws[i] = rng.beta(F_a, F_b)
        else:
            mc_draws[i] = rng.beta(G_a, G_b)

    dev = π*f(mc_draws)/(π*f(mc_draws) + (1-π)*g(mc_draws)) - π
    return np.mean(dev**2)

π_array = np.linspace(0, 1, 40)
cond_var_array = []

for π in π_array:
    cond_var_array.append(compute_cond_var(π, rng))

fig, ax = plt.subplots()
ax.plot(π_array, cond_var_array, lw=2)
ax.set_xlabel(r'$\pi_{t-1}$')
ax.set_ylabel(r'$\sigma^{2}(\pi_{t}\vert \pi_{t-1})$')
plt.show()
```

条件方差作为 $\pi_{t-1}$ 的函数的形状，能够告诉我们 $\{\pi_t\}$ 样本路径的行为特征。

注意当 $\pi_{t-1}$ 接近 0 或 1 时，条件方差趋近于 0。

只有当代理几乎确定 $w_t$ 是从 $F$ 分布中抽取的，或者几乎确定是从 $G$ 分布中抽取的时候，条件方差才接近于零。

在每一个 $\pi_{t-1} \in (0,1)$ 处，条件方差都是正的。

由于 $\pi_t$ 是一个鞅，沿着一条路径的条件方差之和的期望是有界的：

$$
E_{-1} \sum_{t=0}^{\infty} \sigma^2(\pi_t | \pi_{t-1}) = E_{-1} \sum_{t=0}^{\infty} (\pi_t - \pi_{t-1})^2 = E_{-1} \pi_\infty^2 - \pi_{-1}^2 \leq \pi_{-1}(1 - \pi_{-1}) .
$$

因此，几乎沿着每一条路径都有 $\sigma^2(\pi_t | \pi_{t-1}) \rightarrow 0$，并且 $\pi_t$ 必须趋近于条件方差为零的点，即 $0$ 或 $1$。

练习 {ref}`likelihood_bayes_ex2` 补充了相关细节。

## 相关讲座

本讲座致力于建立一些有用的基础设施，这将有助于我们理解在 {doc}`odu`、{doc}`wald_friedman` 和 {doc}`navy_captain` 中描述的结果的推理基础。

以本讲座为基础的其他讲座包括

* {doc}`exchangeable`，描述了构成本讲座主观分布基础的 $w_0, w_1, \ldots$ 的可交换联合分布
* {doc}`mix_model`，研究当自然每期重新混合 $f$ 和 $g$ 时的贝叶斯学习
* {doc}`merging_of_opinions`，研究具有不同先验的两个贝叶斯主义者的后验何时会趋同
* {doc}`likelihood_ratio_process_2`，在进一步的背景下研究似然比过程和贝叶斯学习
* {doc}`wald_friedman_2`，使用后验 $\pi_t$ 作为序贯决策问题的状态变量

## 练习

```{exercise}
:label: likelihood_bayes_ex1

考虑两种模拟历史 $w_0, w_1, \ldots, w_{T-1}$ 的方法：

* 方法（i）：以 ${\rm Prob}(q = f) = \pi_{-1}$ 一次性抽取 $q$，然后从 $q$ 中独立同分布地抽取 $w_0, w_1, \ldots$
* 方法（ii）：从预测密度 $h(w_t|\pi_{t-1})$ 中抽取每个 $w_t$，如 `martingale_simulate` 中所做的那样

1. 证明 $\prod_{s=0}^{t} h(w_s|\pi_{s-1}) = \pi_{-1} f(w^t) + (1 - \pi_{-1}) g(w^t)$，由此说明对于每个 $t$，这两种方法给出 $w^t$ 相同的联合分布。

2. 取 $\pi_{-1} = 0.3$，用每种方法模拟 $10{,}000$ 条路径。用双样本科尔莫戈罗夫-斯米尔诺夫检验比较 $t = 1, 10, 100$ 时 $\pi_t$ 的分布。

3. 在方法（i）下，验证在 $q = f$ 的路径上 $\pi_t$ 趋近于 $1$，在 $q = g$ 的路径上 $\pi_t$ 趋近于 $0$。
```

```{solution-start} likelihood_bayes_ex1
:class: dropdown
```

第 1 部分。

对于 $s \geq 0$，设 $D_s = \pi_{-1} f(w^s) + (1 - \pi_{-1}) g(w^s)$，并设 $D_{-1} = 1$。

由 {eq}`eq_Bayeslaw1033` 可知，$\pi_{s-1} = \pi_{-1} f(w^{s-1}) / D_{s-1}$，且 $1 - \pi_{s-1} = (1-\pi_{-1}) g(w^{s-1}) / D_{s-1}$。

因此

$$
h(w_s|\pi_{s-1}) = \frac{\pi_{-1} f(w^{s-1}) f(w_s) + (1 - \pi_{-1}) g(w^{s-1}) g(w_s)}{D_{s-1}} = \frac{D_s}{D_{s-1}} .
$$

对 $s = 0, \ldots, t$ 求积，经过套叠消去得到 $D_t / D_{-1} = D_t$。

第 2 部分。

两种方法都产生对数几率的路径，我们比较对数几率。

科尔莫戈罗夫-斯米尔诺夫统计量对将对数几率映射为概率的递增变换是不变的。

```{code-cell} ipython3
@jit
def draw_q_once(π_prior, rng, N, T):
    """
    方法（i）。返回 π_t 的对数几率（第 t+1 列保存 π_t）
    以及每条路径 q = f 的指示变量。
    """
    log_odds = np.empty((N, T+1))
    q_is_f = np.empty(N, dtype=np.bool_)
    for n in range(N):
        q_is_f[n] = rng.random() < π_prior
        log_odds[n, 0] = np.log(π_prior) - np.log(1 - π_prior)
        for t in range(T):
            if q_is_f[n]:
                w = rng.beta(F_a, F_b)
            else:
                w = rng.beta(G_a, G_b)
            log_odds[n, t+1] = update(log_odds[n, t], f(w) / g(w))
    return log_odds, q_is_f


@jit
def predictive_sampling(π_prior, rng, N, T):
    "方法（ii）。返回 π_t 的对数几率（第 t+1 列保存 π_t）。"
    log_odds = np.empty((N, T+1))
    for n in range(N):
        log_odds[n, 0] = np.log(π_prior) - np.log(1 - π_prior)
        for t in range(T):
            π = 1 / (1 + np.exp(-log_odds[n, t]))
            if rng.random() < π:
                w = rng.beta(F_a, F_b)
            else:
                w = rng.beta(G_a, G_b)
            log_odds[n, t+1] = update(log_odds[n, t], f(w) / g(w))
    return log_odds


π_prior, N_ex, T_ex = 0.3, 10_000, 200
rng_ex = np.random.default_rng(1)
lo_i, q_is_f = draw_q_once(π_prior, rng_ex, N_ex, T_ex)
lo_ii = predictive_sampling(π_prior, rng_ex, N_ex, T_ex)

for t in [1, 10, 100]:
    res = ks_2samp(lo_i[:, t+1], lo_ii[:, t+1])
    print(f"t = {t:3d}: KS statistic = {res.statistic:.4f}, p-value = {res.pvalue:.3f}")
```

在 5% 的水平上，检验未能拒绝两种方法产生相同 $\pi_t$ 分布的假设。

第 3 部分。

```{code-cell} ipython3
π_T = expit(lo_i[:, -1])
print(f"fraction of paths with q = f: {q_is_f.mean():.4f}")
print(f"among paths with q = f, fraction with π_{T_ex-1} > 0.99: {np.mean(π_T[q_is_f] > 0.99):.4f}")
print(f"among paths with q = g, fraction with π_{T_ex-1} < 0.01: {np.mean(π_T[~q_is_f] < 0.01):.4f}")
```

在方法（i）下，$\pi_t$ 在每条模拟路径上都会发现 $q$，正如杜布定理所断言的那样。

```{solution-end}
```

```{exercise}
:label: likelihood_bayes_ex2

设 $\Delta_t = \pi_t - \pi_{t-1}$，其中 $t \geq 0$。

1. 证明 $E[\Delta_t^2 \,|\, w^{t-1}] = \sigma^2(\pi_t|\pi_{t-1})$，且

$$
\sigma^2(\pi_t|\pi_{t-1}) = \pi_{t-1}^2 (1 - \pi_{t-1})^2 \int \frac{(f(w) - g(w))^2}{h(w|\pi_{t-1})} dw .
$$

2. 利用鞅性质证明 $E_{-1} \sum_{s=0}^{t} \Delta_s^2 = E_{-1} \pi_t^2 - \pi_{-1}^2$。由此推出 $E_{-1} \sum_{t=0}^{\infty} \Delta_t^2 = E_{-1} \pi_\infty^2 - \pi_{-1}^2 \leq \pi_{-1}(1 - \pi_{-1})$，当 $\pi_\infty \in \{0, 1\}$ 以概率 1 成立时取等号。

3. 对 $\pi_{-1} = 0.3$ 和 $\pi_{-1} = 0.5$ 通过模拟验证该等式。

4. 利用第 1、2 部分给出 $\pi_\infty \in \{0, 1\}$ 以概率 1 成立的第二种证明。
```

```{solution-start} likelihood_bayes_ex2
:class: dropdown
```

第 1 部分。

在给定 $w^{t-1}$ 的条件下，$w_t$ 的密度为 $h(w_t|\pi_{t-1})$，且 $E[\pi_t | w^{t-1}] = \pi_{t-1}$，因此 $E[\Delta_t^2 | w^{t-1}]$ 就是条件方差 $\sigma^2(\pi_t|\pi_{t-1})$。

记 $\pi = \pi_{t-1}$。

则 $\pi_t - \pi = \pi f / h(w|\pi) - \pi = \pi (1 - \pi)(f - g) / h(w|\pi)$，且

$$
\sigma^2(\pi_t|\pi_{t-1}) = \int \frac{\pi^2 (1-\pi)^2 (f(w) - g(w))^2}{h(w|\pi)^2} h(w|\pi) dw = \pi^2 (1-\pi)^2 \int \frac{(f(w) - g(w))^2}{h(w|\pi)} dw .
$$

当 $f \neq g$ 时该积分为正，所以 $\sigma^2$ 在 $(0,1)$ 上为正。

一个取值在 $[0,1]$ 中、均值为 $\pi$ 的随机变量，其方差至多为 $\pi(1-\pi)$，所以当 $\pi \rightarrow 0$ 或 $\pi \rightarrow 1$ 时 $\sigma^2 \rightarrow 0$。

第 2 部分。

由于 $E[\Delta_t | w^{t-1}] = 0$，

$$
E_{-1} \pi_t^2 = E_{-1} (\pi_{t-1} + \Delta_t)^2 = E_{-1} \pi_{t-1}^2 + E_{-1} \Delta_t^2 + 2 E_{-1}\bigl[\pi_{t-1} E[\Delta_t | w^{t-1}]\bigr] = E_{-1} \pi_{t-1}^2 + E_{-1} \Delta_t^2 .
$$

从 $0$ 到 $t$ 求和得到 $E_{-1} \sum_{s=0}^{t} \Delta_s^2 = E_{-1} \pi_t^2 - \pi_{-1}^2$。

令 $t \rightarrow \infty$。

左边依单调收敛收敛，右边依有界收敛收敛到 $E_{-1} \pi_\infty^2 - \pi_{-1}^2$。

因为 $0 \leq \pi_\infty \leq 1$，所以 $\pi_\infty^2 \leq \pi_\infty$，因此 $E_{-1} \pi_\infty^2 \leq E_{-1} \pi_\infty = \pi_{-1}$。

当且仅当 $\pi_\infty^2 = \pi_\infty$ 时等号成立，即 $\pi_\infty \in \{0, 1\}$ 以概率 1 成立。

第 3 部分。

```{code-cell} ipython3
for π_prior in [0.3, 0.5]:
    π_path_ex, _ = martingale_simulate(π_prior, rng_ex, N=20_000, T=500)
    sum_sq = np.sum(np.diff(π_path_ex, axis=1)**2, axis=1)
    print(f"π_-1 = {π_prior}: mean of sum of squared increments = {sum_sq.mean():.4f}, "
          f"π_-1 (1 - π_-1) = {π_prior * (1 - π_prior):.4f}")
```

第 4 部分。

由第 1、2 部分可知，

$$
E_{-1} \sum_{t=0}^{\infty} \sigma^2(\pi_t|\pi_{t-1}) = E_{-1} \sum_{t=0}^{\infty} \Delta_t^2 \leq \pi_{-1}(1 - \pi_{-1}) < \infty ,
$$

所以 $\sum_{t} \sigma^2(\pi_t|\pi_{t-1}) < \infty$，且 $\sigma^2(\pi_t|\pi_{t-1}) \rightarrow 0$ 以概率 1 成立。

函数 $\pi \mapsto \sigma^2$ 在 $[0,1]$ 上连续，在 $(0,1)$ 上为正，在 $0$ 和 $1$ 处为零。

由于 $\pi_t \rightarrow \pi_\infty$，根据连续性，在 $\pi_\infty$ 处取值的 $\sigma^2$ 为零，所以 $\pi_\infty \in \{0, 1\}$。

```{solution-end}
```

```{exercise}
:label: likelihood_bayes_ex3

两位贝叶斯主义者观察到相同的历史 $w^t$，但从不同的先验出发，$\pi^1_{-1} = 0.2$ 和 $\pi^2_{-1} = 0.8$。

1. 证明他们对数几率之间的差在每个日期都相同。

2. 证明当大自然从 $f$ 中抽取时，$\frac{1}{t+1} \log |\pi^1_t - \pi^2_t|$ 以概率 1 收敛到 $-KL(f, g)$；而当大自然从 $g$ 中抽取时，以概率 1 收敛到 $-KL(g, f)$。

3. 通过模拟验证第 2 部分。

{doc}`merging_of_opinions` 一般性地研究了不同先验的贝叶斯主义者的意见何时会趋同。
```

```{solution-start} likelihood_bayes_ex3
:class: dropdown
```

第 1 部分。

由 {eq}`eq_logodds` 可知，每位贝叶斯主义者在每个日期都对其对数几率加上相同的 $\log \ell(w_{t})$，所以他们对数几率之差 $c$ 等于其初始值 $\log \frac{0.2}{0.8} - \log \frac{0.8}{0.2} = -2 \log 4$。

第 2 部分。

设 $x_i$ 为贝叶斯主义者 $i$ 的对数几率，使得 $\pi^i_t = 1/(1 + e^{-x_i})$。

直接计算得到

$$
\pi^1_t - \pi^2_t = \frac{\sinh(c/2)}{2 \cosh(x_1/2) \cosh(x_2/2)} .
$$

由于当 $|x| \rightarrow \infty$ 时 $\log \cosh(x/2) = |x|/2 - \log 2 + o(1)$，且 $x_i = x_i(-1) + \log L(w^t)$，

$$
\log |\pi^1_t - \pi^2_t| = - |\log L(w^t)| + O(1) .
$$

由强大数定律，$\frac{1}{t+1} \log L(w^t)$ 在 $f$ 下收敛到 $KL(f, g)$，在 $g$ 下收敛到 $-KL(g, f)$。

第 3 部分。

我们根据第 2 部分中的公式计算 $\log|\pi^1_t - \pi^2_t|$。

直接计算 $\pi^1_t - \pi^2_t$ 在两个概率都四舍五入为 $1$ 或 $0$ 后会返回零。

```{code-cell} ipython3
def log_gap(x1, x2):
    "log |expit(x1) - expit(x2)|, computed without cancellation."
    log_cosh_half = lambda x: np.logaddexp(x / 2, -x / 2) - np.log(2)
    c = x1 - x2
    return (np.log(np.abs(np.sinh(c / 2))) - np.log(2)
            - log_cosh_half(x1) - log_cosh_half(x2))

KL_fg = compute_KL(f, g)
KL_gf = compute_KL(g, f)

T_m, N_m = 200, 1000
t_m = np.arange(T_m)
fig, ax = plt.subplots()
for name, (a, b), KL in [('f', (F_a, F_b), KL_fg), ('g', (G_a, G_b), KL_gf)]:
    log_L = np.cumsum(np.log(simulate(a, b, rng_ex, T=T_m, N=N_m)), axis=1)
    gap = log_gap(logit(0.2) + log_L, logit(0.8) + log_L)
    med = np.median(gap, axis=0)
    slope = np.polyfit(t_m[50:], med[50:], 1)[0]
    ax.plot(t_m, med, lw=2, label=f'nature draws from ${name}$')
    print(f"nature draws from {name}: slope of median log gap = {slope:.3f}, "
          f"-KL = {-KL:.3f}")

ax.set_xlabel('$t$')
ax.set_ylabel(r'median of $\log|\pi^1_t - \pi^2_t|$')
ax.legend()
plt.show()
```

这些斜率与 $-KL(f, g)$ 和 $-KL(g, f)$ 相匹配。

两位贝叶斯主义者的意见以指数速度趋同，当大自然从 $f$ 中抽取时比从 $g$ 中抽取时趋同得更快。

```{solution-end}
```
