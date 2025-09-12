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

# 似然比过程

```{contents} 目录
:depth: 2
```

## 概述

本讲座介绍似然比过程及其一些用途。

我们将研究与{doc}`可交换性讲座 <exchangeable>`中相同的设定。

我们将学习的内容包括：

* 似然比过程如何成为频率派假设检验的关键要素
* 接收者操作特征曲线如何总结频率派假设检验中关于虚警概率和检验效能的信息
* 统计学家如何将第一类和第二类错误的频率派概率结合起来，形成模型选择或个体分类问题中的错误后验概率
* 如何使用Kullback-Leibler散度来量化具有相同支撑集的两个概率分布之间的差异

* 二战期间美国海军如何制定了一个用于弹药批次质量控制的决策规则，这为{doc}`本讲座 <wald_friedman>`奠定了基础
* 似然比过程的一个特殊性质



让我们先导入一些Python工具。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import beta as beta_dist
import pandas as pd
from IPython.display import display, Math
import quantecon as qe
```

## 似然比过程

一个非负随机变量 $W$ 具有两个概率密度函数之一，要么是 $f$，要么是 $g$。

在时间开始之前，自然界一次性地决定是从 $f$ 还是 $g$ 中进行一系列独立同分布的抽样。

我们有时用 $q$ 表示自然界一次性选择的密度，所以 $q$ 要么是 $f$ 要么是 $g$，且是永久性的。

自然界知道它永久性地从哪个密度中抽样，但我们这些观察者并不知道。

我们知道 $f$ 和 $g$ 两个密度，但不知道自然界选择了哪一个。

但我们想要知道。

为此，我们使用观测值。

我们观察到一个序列 $\{w_t\}_{t=1}^T$，包含 $T$ 个独立同分布的抽样，我们知道这些抽样要么来自 $f$ 要么来自 $g$。

我们想要利用这些观测值来推断自然界选择了 $f$ 还是 $g$。

**似然比过程**是完成这项任务的有用工具。

首先，我们定义似然比过程的一个关键组成部分，即时间 $t$ 的似然比，它是一个随机变量：

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

我们假设 $f$ 和 $g$ 在随机变量 $W$ 的相同可能取值区间上都赋予正概率。

这意味着在 $g$ 密度下，$\ell (w_t)=\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$ 是一个均值为1的非负随机变量。

序列 $\left\{ w_{t}\right\} _{t=1}^{\infty}$ 的**似然比过程**定义为：

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

其中 $w^t=\{ w_1,\dots,w_t\}$ 是直到时间 $t$ (包括 $t$) 的观测历史。

为简便起见，我们有时会写作 $L_t = L(w^t)$。

注意似然过程满足以下*递归*关系

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

似然比及其对数是 Neyman 和 Pearson {cite}`Neyman_Pearson` 经典频率派推断方法中的关键工具。

为了帮助我们理解其工作原理，以下 Python 代码将 $f$ 和 $g$ 定义为两个不同的 Beta 分布，然后通过从两个概率分布之一(例如，从 $g$ 生成 IID 序列)生成序列 $w^t$ 来计算和模拟相关的似然比过程。

```{code-cell} ipython3
# Parameters for the two Beta distributions
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    """Beta distribution density function."""
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

def create_beta_density(a, b):
    """Create a beta density function with specified parameters."""
    return jit(lambda x: p(x, a, b))

def likelihood_ratio(w, f_func, g_func):
    """Compute likelihood ratio for observation(s) w."""
    return f_func(w) / g_func(w)

@jit
def simulate_likelihood_ratios(a, b, f_func, g_func, T=50, N=500):
    """
    Generate N sets of T observations of the likelihood ratio.
    """
    l_arr = np.empty((N, T))
    for i in range(N):
        for j in range(T):
            w = np.random.beta(a, b)
            l_arr[i, j] = f_func(w) / g_func(w)
    return l_arr

def simulate_sequences(distribution, f_func, g_func, 
        F_params=(1, 1), G_params=(3, 1.2), T=50, N=500):
    """
    Generate N sequences of T observations from specified distribution.
    """
    if distribution == 'f':
        a, b = F_params
    elif distribution == 'g':
        a, b = G_params
    else:
        raise ValueError("distribution must be 'f' or 'g'")
    
    l_arr = simulate_likelihood_ratios(a, b, f_func, g_func, T, N)
    l_seq = np.cumprod(l_arr, axis=1)
    return l_arr, l_seq

def plot_likelihood_paths(l_seq, title="Likelihood ratio paths", 
                        ylim=None, n_paths=None):
    """Plot likelihood ratio paths."""
    N, T = l_seq.shape
    n_show = n_paths or min(N, 100)
    
    plt.figure(figsize=(10, 6))
    for i in range(n_show):
        plt.plot(range(T), l_seq[i, :], color='b', lw=0.8, alpha=0.5)
    
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('$L(w^t)$')
    plt.show()
```

(nature_likeli)=
## 当自然永久从密度g中抽取时

我们首先模拟当自然永久从$g$中抽取时的似然比过程。

```{code-cell} ipython3
# 模拟当自然从g中抽取时
l_arr_g, l_seq_g = simulate_sequences('g', f, g, (F_a, F_b), (G_a, G_b))
plot_likelihood_paths(l_seq_g, 
                     title="当自然从g中抽取时的$L(w^{t})$路径",
                     ylim=[0, 3])
```

显然，随着样本长度 $T$ 的增长，大部分概率质量
向零靠近

为了更清楚地看到这一点，我们绘制了随时间变化的
路径 $L\left(w^{t}\right)$ 落在区间
$\left[0, 0.01\right]$ 内的比例。

```{code-cell} ipython3
N, T = l_arr_g.shape
plt.plot(range(T), np.sum(l_seq_g <= 0.01, axis=0) / N)
plt.show()
```

尽管大部分概率质量明显收敛到接近$0$的一个很小区间内，但在概率密度$g$下，$L\left(w^t\right)$的无条件均值对所有$t$恒等于$1$。

为了验证这个论断，首先注意到如前所述，对所有$t$，无条件均值$E\left[\ell \left(w_{t}\right)\bigm|q=g\right]$等于$1$：

$$
\begin{aligned}
E\left[\ell \left(w_{t}\right)\bigm|q=g\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
    &=\int f\left(w_{t}\right)dw_{t} \\
    &=1,
\end{aligned}
$$

这直接推出

$$
\begin{aligned}
E\left[L\left(w^{1}\right)\bigm|q=g\right]  &=E\left[\ell \left(w_{1}\right)\bigm|q=g\right]\\
    &=1.\\
\end{aligned}
$$

因为$L(w^t) = \ell(w_t) L(w^{t-1})$且$\{w_t\}_{t=1}^t$是独立同分布序列，我们有

$$
\begin{aligned}
E\left[L\left(w^{t}\right)\bigm|q=g\right]  &=E\left[L\left(w^{t-1}\right)\ell \left(w_{t}\right)\bigm|q=g\right] \\
         &=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g,w^{t-1}\right]\bigm|q=g\right] \\
     &=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g\right]\bigm|q=g\right] \\
    &=E\left[L\left(w^{t-1}\right)\bigm|q=g\right] \\
\end{aligned}
$$

对任意$t \geq 1$成立。

数学归纳法表明对所有$t \geq 1$，$E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$。

## 特殊性质

当似然比过程的大部分概率质量在 $t \rightarrow + \infty$ 时堆积在 $0$ 附近时，$E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ 怎么可能成立？

答案是，当 $t \rightarrow + \infty$ 时，$L_t$ 的分布变得越来越厚尾：足够多的质量向 $L_t$ 的更大值移动，使得尽管大部分概率质量堆积在 $0$ 附近，$L_t$ 的均值仍然保持为1。

为了说明这个特殊性质，我们模拟多条路径，并通过在每个时刻 $t$ 对这些路径取平均来计算 $L\left(w^t\right)$ 的无条件均值。

```{code-cell} ipython3
l_arr_g, l_seq_g = simulate_sequences('g', 
                f, g, (F_a, F_b), (G_a, G_b), N=50000)
```

使用模拟来验证无条件期望值$E\left[L\left(w^{t}\right)\right]$等于1(通过对样本路径取平均)会很有用。

但是在这里使用标准蒙特卡洛模拟方法会消耗太多计算时间,因此我们不会这样做。

原因是对于较大的$t$值,$L\left(w^{t}\right)$的分布极度偏斜。

因为右尾部的概率密度接近于0,从右尾部采样足够多的点需要太多计算时间。

我们在{doc}`这篇讲座 <imp_sample>`中更详细地解释了这个问题。

在那里我们描述了一种通过从不同的概率分布中采样来计算不同随机变量的均值,从而计算似然比均值的替代方法。

## 自然永久从密度f中抽样

现在假设在时间0之前,自然界永久决定反复从密度f中抽样。

虽然似然比$\ell \left(w_{t}\right)$在密度$g$下的均值为1,但在密度$f$下的均值超过1。

为了说明这一点,我们计算:

$$
\begin{aligned}
E\left[\ell \left(w_{t}\right)\bigm|q=f\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}f\left(w_{t}\right)dw_{t} \\
     &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
     &=\int \ell \left(w_{t}\right)^{2}g\left(w_{t}\right)dw_{t} \\

&=E\left[\ell \left(w_{t}\right)^{2}\mid q=g\right] \\
     &=E\left[\ell \left(w_{t}\right)\mid q=g\right]^{2}+Var\left(\ell \left(w_{t}\right)\mid q=g\right) \\
     &>E\left[\ell \left(w_{t}\right)\mid q=g\right]^{2} = 1 \\
       \end{aligned}
$$

这反过来意味着似然比过程$L(w^t)$的无条件均值将趋向于$+ \infty$。

下面的模拟验证了这个结论。

请注意$y$轴的刻度。

```{code-cell} ipython3
# 模拟当自然从f中抽取时的情况
l_arr_f, l_seq_f = simulate_sequences('f', f, g, 
                        (F_a, F_b), (G_a, G_b), N=50000)
```

```{code-cell} ipython3
N, T = l_arr_f.shape
plt.plot(range(T), np.mean(l_seq_f, axis=0))
plt.show()
```

我们还绘制了 $L\left(w^t\right)$ 落入区间 $[10000, \infty)$ 的概率随时间的变化图，观察概率质量向 $+\infty$ 发散的速度。

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_f > 10000, axis=0) / N)
plt.show()
```

## 似然比检验

我们现在描述如何使用
Neyman和Pearson {cite}`Neyman_Pearson` 的方法来检验历史数据 $w^t$ 是否由密度函数 $f$ 的重复独立同分布抽样生成。

令 $q$ 为数据生成过程，因此
$q=f \text{ 或 } g$。

在观察到样本 $\{W_i\}_{i=1}^t$ 后，我们想通过执行(频率派)
假设检验来判断自然是从 $g$ 还是从 $f$ 中抽样。

我们指定

- 零假设 $H_0$: $q=f$,
- 备择假设 $H_1$: $q=g$。

Neyman和Pearson证明了检验这个假设的最佳方法是使用**似然比检验**，
形式为:

- 当 $L(W^t) > c$ 时接受 $H_0$,
- 当 $L(W^t) < c$ 时拒绝 $H_0$,

其中 $c$ 是给定的判别阈值。

设置 $c =1$ 是一个常见的选择。

我们将在下面讨论其他 $c$ 值选择的后果。

这个检验是*最佳的*，因为它是**一致最优势的**。

为了理解这意味着什么，我们需要定义两个重要事件的概率，这些概率
可以帮助我们描述与给定阈值 $c$ 相关的检验。

这两个概率是:

- 第一类错误概率，即在 $H_0$ 为真时拒绝它:
  
  $$
  \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}
  $$

- 第二类错误概率，即在 $H_0$ 为假时接受它:

  $$
  \beta \equiv \Pr\left\{ L\left(w^{t}\right)>c\mid q=g\right\}
  $$

这两个概率构成了以下两个概念的基础:

- 虚警概率（=显著性水平=第一类错误概率）：

  $$
  \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}
  $$

- 检测概率（=检验力=1减去第二类错误概率）：

  $$
  1-\beta \equiv \Pr\left\{ L\left(w^{t}\right)<c\mid q=g\right\}
  $$

[奈曼-皮尔逊引理](https://en.wikipedia.org/wiki/Neyman–Pearson_lemma)指出，在所有可能的检验中，似然比检验在给定虚警概率的情况下能最大化检测概率。

换句话说，在所有可能的检验中，似然比检验在给定**显著性水平**的情况下能最大化**检验力**。

我们希望虚警概率小，检测概率大。

当样本量$t$固定时，我们可以通过调整$c$来改变这两个概率。

一个令人困扰的"现实"事实是，当我们改变临界值$c$时，这两个概率会朝同一方向变化。

如果不指定第一类和第二类错误的具体损失，我们很难说应该如何权衡这两种错误的概率。

我们知道增加样本量$t$可以改善统计推断。

下面我们将绘制一些说明性图表来展示这一点。

我们还将介绍一个用于选择样本量$t$的经典频率派方法。

让我们从将阈值$c$固定为$1$的情况开始。

```{code-cell} ipython3
c = 1
```

下面我们绘制上面模拟的累积似然比的对数的经验分布，这些分布是由$f$或$g$生成的。

取对数不会影响概率的计算，因为对数是单调变换。

随着$t$的增加，第一类错误和第二类错误的概率都在减小，这是好事。

这是因为当$g$是数据生成过程时，log$(L(w^t))$的大部分概率质量向$-\infty$移动，而当数据由$f$生成时，log$(L(w^t))$趋向于$\infty$。

log$(L(w^t))$在$f$和$g$下的这种不同行为使得最终能够区分$q=f$和$q=g$成为可能。

```{code-cell} ipython3
def plot_log_histograms(l_seq_f, l_seq_g, c=1, time_points=[1, 7, 14, 21]):
    """绘制对数似然比直方图。"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, t in enumerate(time_points):
        nr, nc = i // 2, i % 2
        
        axs[nr, nc].axvline(np.log(c), color="k", ls="--")
        
        hist_f, x_f = np.histogram(np.log(l_seq_f[:, t]), 200, density=True)
        hist_g, x_g = np.histogram(np.log(l_seq_g[:, t]), 200, density=True)
        
        axs[nr, nc].plot(x_f[1:], hist_f, label="f下的分布")
        axs[nr, nc].plot(x_g[1:], hist_g, label="g下的分布")
        
        # 填充错误区域
        for j, (x, hist, label) in enumerate(
            zip([x_f, x_g], [hist_f, hist_g], 
            ["第一类错误", "第二类错误"])):
            ind = x[1:] <= np.log(c) if j == 0 else x[1:] > np.log(c)
            axs[nr, nc].fill_between(x[1:][ind], hist[ind], 
                                    alpha=0.5, label=label)
        
        axs[nr, nc].legend()
        axs[nr, nc].set_title(f"t={t}")
    
    plt.show()

plot_log_histograms(l_seq_f, l_seq_g, c=c)
```

在上述图表中，
  * 蓝色区域与第一类错误的概率 $\alpha$ 相关但不相等，因为
它们是在拒绝域 $L_t < 1$ 上对 $\log L_t$ 的积分，而不是对 $L_t$ 的积分
* 橙色区域与第二类错误的概率 $\beta$ 相关但不相等，因为
它们是在接受域 $L_t > 1$ 上对 $\log L_t$ 的积分，而不是对 $L_t$ 的积分

当我们将 $c$ 固定在 $c=1$ 时，下图显示：
  * 检测概率随着 $t$ 的增加单调增加
  * 虚警概率随着 $t$ 的增加单调减少

```{code-cell} ipython3

def compute_error_probabilities(l_seq_f, l_seq_g, c=1):
    """
    计算第一类和第二类错误概率。
    """
    N, T = l_seq_f.shape
    
    # 第一类错误（虚警）- 在H0为真时拒绝H0
    PFA = np.array([np.sum(l_seq_f[:, t] < c) / N for t in range(T)])
    
    # 第二类错误 - 在H0为假时接受H0
    beta = np.array([np.sum(l_seq_g[:, t] >= c) / N for t in range(T)])
    
    # 检测概率（功效）
    PD = np.array([np.sum(l_seq_g[:, t] < c) / N for t in range(T)])
    
    return {
        'alpha': PFA,
        'beta': beta, 
        'PD': PD,
        'PFA': PFA
    }

def plot_error_probabilities(error_dict, T, c=1, title_suffix=""):
    """绘制随时间变化的错误概率。"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(T), error_dict['PD'], label="检测概率")
    plt.plot(range(T), error_dict['PFA'], label="虚警概率")
    plt.xlabel("t")
    plt.ylabel("概率")
    plt.title(f"错误概率 (c={c}){title_suffix}")
    plt.legend()
    plt.show()

error_probs = compute_error_probabilities(l_seq_f, l_seq_g, c=c)
N, T = l_seq_f.shape
plot_error_probabilities(error_probs, T, c)
```

对于给定的样本量 $t$，阈值 $c$ 唯一确定了两种类型错误的概率。

如果在固定 $t$ 的情况下，我们释放并移动 $c$，我们将得到检测概率作为虚警概率的函数。

这就产生了[接收者操作特征曲线（ROC曲线）](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)。

下面，我们为不同的样本量 $t$ 绘制接收者操作特征曲线。

```{code-cell} ipython3
def plot_roc_curves(l_seq_f, l_seq_g, t_values=[1, 5, 9, 13], N=None):
    """绘制不同样本量的ROC曲线。"""
    if N is None:
        N = l_seq_f.shape[0]
    
    PFA = np.arange(0, 100, 1)
    
    plt.figure(figsize=(10, 6))
    for t in t_values:
        percentile = np.percentile(l_seq_f[:, t], PFA)
        PD = [np.sum(l_seq_g[:, t] < p) / N for p in percentile]
        plt.plot(PFA / 100, PD, label=f"t={t}")
    
    plt.scatter(0, 1, label="完美检测")
    plt.plot([0, 1], [0, 1], color='k', ls='--', label="随机检测")
    
    plt.arrow(0.5, 0.5, -0.15, 0.15, head_width=0.03)
    plt.text(0.35, 0.7, "更好")
    plt.xlabel("虚警概率")
    plt.ylabel("检测概率")
    plt.legend()
    plt.title("ROC曲线")
    plt.show()


plot_roc_curves(l_seq_f, l_seq_g, t_values=range(1, 15, 4), N=N)
```

注意到随着$t$的增加，对于给定的判别阈值$c$，我们可以获得更高的检测概率和更低的虚警概率。

对于给定的样本量$t$，当我们改变$c$时，$\alpha$和$\beta$都会发生变化。

当我们增加$c$时

* $\alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}$ 增加
* $\beta \equiv \Pr\left\{ L\left(w^{t}\right)>c\mid q=g\right\}$ 减少

当$t \rightarrow + \infty$时，我们接近完美检测曲线，该曲线在蓝点处呈直角。

对于给定的样本量$t$，判别阈值$c$决定了接收者操作特征曲线上的一个点。

测试设计者需要权衡这两种类型错误的概率。

但我们知道如何选择最小样本量来达到给定的概率目标。

通常，频率学派的目标是在虚警概率有上限的情况下实现高检测概率。

下面我们展示一个例子，其中我们将虚警概率固定在$0.05$。

做出决策所需的样本量则由目标检测概率决定，例如$0.9$，如下图所示。

```{code-cell} ipython3
PFA = 0.05
PD = np.empty(T)

for t in range(T):

    c = np.percentile(l_seq_f[:, t], PFA * 100)
    PD[t] = np.sum(l_seq_g[:, t] < c) / N

plt.plot(range(T), PD)
plt.axhline(0.9, color="k", ls="--")

plt.xlabel("t")
plt.ylabel("检测概率")
plt.title(f"虚警概率={PFA}")
plt.show()
```

美国海军显然在第二次世界大战期间使用类似这样的程序来选择质量控制测试的样本大小 $t$。

一位被命令执行此类测试的海军上校对此产生了疑虑，他向米尔顿·弗里德曼提出了这些疑虑，我们在{doc}`这篇讲座 <wald_friedman>`中对此进行了描述。

(llr_h)=
### 第三个分布 $h$

现在让我们考虑一种既不是 $g$ 也不是 $f$ 生成数据的情况。

而是由第三个分布 $h$ 生成。

让我们研究当 $h$ 支配数据时，累积似然比 $L$ 的表现。

这里的一个关键工具被称为**库尔贝克-莱布勒散度**，我们在{doc}`divergence_measures`中已经研究过。

在我们的应用中，我们想要度量 $f$ 或 $g$ 与 $h$ 的偏离程度。

与我们相关的两个库尔贝克-莱布勒散度是 $K_f$ 和 $K_g$，定义如下：

$$
\begin{aligned}
K_{f} = D_{KL}\bigl(h\|f\bigr) = KL(h, f)
          &= E_{h}\left[\log\frac{h(w)}{f(w)}\right] \\
          &= \int \log\left(\frac{h(w)}{f(w)}\right)h(w)dw .
\end{aligned}
$$

$$
\begin{aligned}
K_{g} = D_{KL}\bigl(h\|g\bigr) = KL(h, g)
          &= E_{h}\left[\log\frac{h(w)}{g(w)}\right] \\
          &= \int \log\left(\frac{h(w)}{g(w)}\right)h(w)dw .
\end{aligned}
$$

让我们使用{doc}`divergence_measures`中的相同代码来计算库尔贝克-莱布勒差异。

```{code-cell} ipython3
def compute_KL(f, g):
    """
    计算KL散度 KL(f, g)
    """
    integrand = lambda w: f(w) * np.log(f(w) / g(w))
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val

def compute_KL_h(h, f, g):
    """
    计算相对于参考分布h的KL散度
    """
    Kf = compute_KL(h, f)
    Kg = compute_KL(h, g)
    return Kf, Kg
```

(KL_link)=
### 一个有用的公式

似然比和KL散度之间存在数学关系。

当数据由分布$h$生成时，期望对数似然比为：

$$
\frac{1}{t} E_{h}\!\bigl[\log L_t\bigr] = K_g - K_f
$$ (eq:kl_likelihood_link)

其中$L_t=\prod_{j=1}^{t}\frac{f(w_j)}{g(w_j)}$是似然比过程。

方程{eq}`eq:kl_likelihood_link`告诉我们：
- 当$K_g < K_f$（即$g$比$f$更接近$h$）时，期望对数似然比为负，所以$L\left(w^t\right) \rightarrow 0$。
- 当$K_g > K_f$（即$f$比$g$更接近$h$）时，期望对数似然比为正，所以$L\left(w^t\right) \rightarrow + \infty$。

让我们通过模拟来验证这一点。

在模拟中，我们使用Beta分布$f$、$g$和$h$生成多条路径，并计算$\log(L(w^t))$的路径。

首先，我们编写一个函数来计算似然比过程

```{code-cell} ipython3
def compute_likelihood_ratios(sequences, f, g):
    """计算似然比和累积乘积。"""
    l_ratios = f(sequences) / g(sequences)
    L_cumulative = np.cumprod(l_ratios, axis=1)
    return l_ratios, L_cumulative
```

我们考虑三种情况：(1) $h$ 更接近 $f$，(2) $f$ 和 $g$ 与 $h$ 的距离大致相等，以及 (3) $h$ 更接近 $g$。

```{code-cell} ipython3
:tags: [hide-input]

# Define test scenarios
scenarios = [
    {
        "name": "KL(h,g) > KL(h,f)",
        "h_params": (1.2, 1.1),
        "expected": r"$L_t \to \infty$"
    },
    {
        "name": "KL(h,g) ≈ KL(h,f)",
        "h_params": (2, 1.35),
        "expected": "$L_t$ fluctuates"
    },
    {
        "name": "KL(h,g) < KL(h,f)", 
        "h_params": (3.5, 1.5),
        "expected": r"$L_t \to 0$"
    }
]

fig, axes = plt.subplots(2, 3, figsize=(15, 12))

for i, scenario in enumerate(scenarios):
    # Define h
    h = lambda x: p(x, scenario["h_params"][0], 
                    scenario["h_params"][1])
    
    # Compute KL divergences
    Kf, Kg = compute_KL_h(h, f, g)
    kl_diff = Kg - Kf
    
    # Simulate paths
    N_paths = 100
    T = 150

    # Generate data from h
    h_data = np.random.beta(scenario["h_params"][0], 
                scenario["h_params"][1], (N_paths, T))
    l_ratios, l_cumulative = compute_likelihood_ratios(h_data, f, g)
    log_l_cumulative = np.log(l_cumulative)
    
    # Plot distributions
    ax = axes[0, i]
    x_range = np.linspace(0.001, 0.999, 200)
    ax.plot(x_range, [f(x) for x in x_range], 
        'b-', label='f', linewidth=2)
    ax.plot(x_range, [g(x) for x in x_range], 
        'r-', label='g', linewidth=2)
    ax.plot(x_range, [h(x) for x in x_range], 
        'g--', label='h (data)', linewidth=2)
    ax.set_xlabel('w')
    ax.set_ylabel('density')
    ax.set_title(scenario["name"], fontsize=16)
    ax.legend()
    
    # Plot log likelihood ratio paths
    ax = axes[1, i]
    for j in range(min(20, N_paths)):
        ax.plot(log_l_cumulative[j, :], alpha=0.3, color='purple')
    
    # Plot theoretical expectation
    theory_line = kl_diff * np.arange(1, T+1)
    ax.plot(theory_line, 'k--', linewidth=2, label=r'$t \times (K_g - K_f)$')
    
    ax.set_xlabel('t')
    ax.set_ylabel('$log L_t$')
    ax.set_title(f'KL(h,f)={Kf:.3f}, KL(h,g)={Kg:.3f}\n{scenario["expected"]}', 
                 fontsize=16)
    ax.legend(fontsize=16)

plt.tight_layout()
plt.show()
```

请注意

- 在第一张图中，由于 $K_g > K_f$，$\log L(w^t)$ 发散到 $\infty$。
- 在第二张图中，虽然仍然有 $K_g > K_f$，但差值较小，所以 $L(w^t)$ 发散到无穷的速度较慢。
- 在最后一张图中，由于 $K_g < K_f$，$\log L(w^t)$ 发散到 $-\infty$。
- 黑色虚线 $t \left(D_{KL}(h\|g) - D_{KL}(h\|f)\right)$ 与验证 {eq}`eq:kl_likelihood_link` 的路径紧密吻合。

这些观察结果与理论相符。

在 {doc}`likelihood_ratio_process_2` 中，我们将看到这些思想的一个应用。

## 假设检验和分类

本节讨论似然比过程的另一个应用。

我们描述统计学家如何结合第一类和第二类错误的频率主义概率来

* 计算基于样本长度 $T$ 选择错误模型的预期频率
* 计算分类问题中的预期错误率

我们考虑这样一种情况：自然界用已知的混合参数 $\pi_{-1} \in (0,1)$ 混合已知密度 $f$ 和 $g$ 来生成数据，使得随机变量 $w$ 从以下密度中抽取

$$
h (w) = \pi_{-1} f(w) + (1-\pi_{-1}) g(w) 
$$

我们假设统计学家知道密度 $f$ 和 $g$ 以及混合参数 $\pi_{-1}$。

下面，我们将设定 $\pi_{-1} = .5$，尽管使用其他 $\pi_{-1} \in (0,1)$ 的值进行分析也是可行的。

我们假设 $f$ 和 $g$ 在随机变量 $W$ 的相同可能实现区间上都赋予正概率。

在下面的模拟中，我们指定 $f$ 是 $\text{Beta}(1, 1)$ 分布，$g$ 是 $\text{Beta}(3, 1.2)$ 分布。

我们考虑两种替代的时序协议。

* 时序协议1用于模型选择问题
* 时序协议2用于个体分类问题

**时序协议1：** 自然只在时间 $t=-1$ **一次性**抛硬币，以概率 $\pi_{-1}$ 从 $f$ 生成一个 IID 序列 $\{w_t\}_{t=1}^T$，以概率 $1-\pi_{-1}$ 从 $g$ 生成一个 IID 序列 $\{w_t\}_{t=1}^T$。

**时序协议2：** 自然**频繁**抛硬币。在每个时间 $t \geq 0$，自然抛一次硬币，以概率 $\pi_{-1}$ 从 $f$ 中抽取 $w_t$，以概率 $1-\pi_{-1}$ 从 $g$ 中抽取 $w_t$。

以下是我们用来实现时序协议1和2的Python代码

```{code-cell} ipython3
def protocol_1(π_minus_1, T, N=1000, F_params=(1, 1), G_params=(3, 1.2)):
    """
    Simulate Protocol 1: Nature decides once at t=-1 which model to use.
    """
    F_a, F_b = F_params
    G_a, G_b = G_params
    
    # Single coin flip for the true model
    true_models_F = np.random.rand(N) < π_minus_1
    sequences = np.empty((N, T))
    
    n_f = np.sum(true_models_F)
    n_g = N - n_f
    
    if n_f > 0:
        sequences[true_models_F, :] = np.random.beta(F_a, F_b, (n_f, T))
    if n_g > 0:
        sequences[~true_models_F, :] = np.random.beta(G_a, G_b, (n_g, T))
    
    return sequences, true_models_F

def protocol_2(π_minus_1, T, N=1000, F_params=(1, 1), G_params=(3, 1.2)):
    """
    Simulate Protocol 2: Nature decides at each time step which model to use.
    """
    F_a, F_b = F_params
    G_a, G_b = G_params
    
    # Coin flips for each time step
    true_models_F = np.random.rand(N, T) < π_minus_1
    sequences = np.empty((N, T))
    
    n_f = np.sum(true_models_F)
    n_g = N * T - n_f
    
    if n_f > 0:
        sequences[true_models_F] = np.random.beta(F_a, F_b, n_f)
    if n_g > 0:
        sequences[~true_models_F] = np.random.beta(G_a, G_b, n_g)
    
    return sequences, true_models_F
```

**注释：** 在时序协议2下，$\{w_t\}_{t=1}^T$ 是从 $h(w)$ 中独立同分布(IID)抽取的序列。在时序协议1下，$\{w_t\}_{t=1}^T$ 不是独立同分布的。它是**条件独立同分布**的 -- 意味着以概率 $\pi_{-1}$ 它是从 $f(w)$ 中抽取的IID序列，以概率 $1-\pi_{-1}$ 它是从 $g(w)$ 中抽取的IID序列。更多相关内容，请参见{doc}`这篇关于可交换性的讲座 <exchangeable>`。

我们再次部署一个**似然比过程**，其时间 $t$ 分量是似然比

$$
\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

序列 $\left\{ w_{t}\right\} _{t=1}^{\infty}$ 的**似然比过程**是

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

为简便起见，我们将写作 $L_t = L(w^t)$。

### 模型选择错误概率

我们首先研究假设时序协议1的问题。

考虑一个决策者想要知道是模型 $f$ 还是模型 $g$ 支配着长度为 $T$ 观测值的数据集。

决策者已经观察到序列 $\{w_t\}_{t=1}^T$。

基于观察到的序列，似然比检验在 $L_T \geq 1$ 时选择模型 $f$，在 $L_T < 1$ 时选择模型 $g$。

当模型 $f$ 生成数据时，似然比检验选择错误模型的概率是

$$ 
p_f = {\rm Prob}\left(L_T < 1\Big| f\right) = \alpha_T .
$$

当模型 $g$ 生成数据时，似然比检验选择错误模型的概率为

$$ 
p_g = {\rm Prob}\left(L_T \geq 1 \Big|g \right) = \beta_T.
$$

我们可以通过赋予自然选择模型 $f$ 的贝叶斯先验概率 $\pi_{-1} = .5$，然后对 $p_f$ 和 $p_g$ 取平均值来构造似然比选择错误模型的概率，从而得到检测错误的贝叶斯后验概率等于

$$ 
p(\textrm{wrong decision}) = {1 \over 2} (\alpha_T + \beta_T) .
$$ (eq:detectionerrorprob)

现在让我们模拟时序协议1和2并计算错误概率

```{code-cell} ipython3

def compute_protocol_1_errors(π_minus_1, T_max, N_simulations, f_func, g_func, 
                              F_params=(1, 1), G_params=(3, 1.2)):
    """
    计算协议1的错误概率。
    """
    sequences, true_models = protocol_1(
        π_minus_1, T_max, N_simulations, F_params, G_params)
    l_ratios, L_cumulative = compute_likelihood_ratios(sequences, 
                                    f_func, g_func)
    
    T_range = np.arange(1, T_max + 1)
    
    mask_f = true_models
    mask_g = ~true_models
    
    L_f = L_cumulative[mask_f, :]
    L_g = L_cumulative[mask_g, :]
    
    α_T = np.mean(L_f < 1, axis=0)
    β_T = np.mean(L_g >= 1, axis=0)
    error_prob = 0.5 * (α_T + β_T)
    
    return {
        'T_range': T_range,
        'alpha': α_T,
        'beta': β_T, 
        'error_prob': error_prob,
        'L_cumulative': L_cumulative,
        'true_models': true_models
    }

def compute_protocol_2_errors(π_minus_1, T_max, N_simulations, f_func, g_func,
                              F_params=(1, 1), G_params=(3, 1.2)):
    """
    计算协议2的错误概率。
    """
    sequences, true_models = protocol_2(π_minus_1, 
                        T_max, N_simulations, F_params, G_params)
    l_ratios, _ = compute_likelihood_ratios(sequences, f_func, g_func)
    
    T_range = np.arange(1, T_max + 1)
    
    accuracy = np.empty(T_max)
    for t in range(T_max):
        predictions = (l_ratios[:, t] >= 1)
        actual = true_models[:, t]
        accuracy[t] = np.mean(predictions == actual)
    
    return {
        'T_range': T_range,
        'accuracy': accuracy,
        'l_ratios': l_ratios,
        'true_models': true_models
    }
```

以下代码可视化了时序协议1和2的错误概率

```{code-cell} ipython3
:tags: [hide-input]

def analyze_protocol_1(π_minus_1, T_max, N_simulations, f_func, g_func, 
                      F_params=(1, 1), G_params=(3, 1.2)):
    """分析协议1"""
    result = compute_protocol_1_errors(π_minus_1, T_max, N_simulations, 
                                      f_func, g_func, F_params, G_params)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(result['T_range'], result['alpha'], 'b-', 
             label=r'$\alpha_T$', linewidth=2)
    ax1.plot(result['T_range'], result['beta'], 'r-', 
             label=r'$\beta_T$', linewidth=2)
    ax1.set_xlabel('$T$')
    ax1.set_ylabel('错误概率')
    ax1.legend()
    
    ax2.plot(result['T_range'], result['error_prob'], 'g-', 
             label=r'$\frac{1}{2}(\alpha_T+\beta_T)$', linewidth=2)
    ax2.set_xlabel('$T$')
    ax2.set_ylabel('错误概率')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印总结
    print(f"在 T={T_max} 时:")
    print(f"α_{T_max} = {result['alpha'][-1]:.4f}")
    print(f"β_{T_max} = {result['beta'][-1]:.4f}")
    print(f"模型选择错误概率 = {result['error_prob'][-1]:.4f}")
    
    return result

def analyze_protocol_2(π_minus_1, T_max, N_simulations, f_func, g_func, 
                      theory_error=None, F_params=(1, 1), G_params=(3, 1.2)):
    """分析协议2"""
    result = compute_protocol_2_errors(π_minus_1, T_max, N_simulations, 
                                      f_func, g_func, F_params, G_params)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(result['T_range'], result['accuracy'], 
            'b-', linewidth=2, label='经验准确率')
    
    if theory_error is not None:
        plt.axhline(1 - theory_error, color='r', linestyle='--', 
                   label=f'理论准确率 = {1 - theory_error:.4f}')
    
    plt.xlabel('$t$')
    plt.ylabel('准确率')
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.show()
    
    return result

def compare_protocols(result1, result2):
    """比较两个协议的结果"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(result1['T_range'], result1['error_prob'], linewidth=2, 
            label='协议1(模型选择)')
    plt.plot(result2['T_range'], 1 - result2['accuracy'], 
            linestyle='--', linewidth=2, 
            label='协议2(分类)')
    
    plt.xlabel('$T$')
    plt.ylabel('错误概率')
    plt.legend()
    plt.show()

# 分析协议1
π_minus_1 = 0.5
T_max = 30
N_simulations = 10_000

result_p1 = analyze_protocol_1(π_minus_1, T_max, N_simulations, 
                                f, g, (F_a, F_b), (G_a, G_b))
```

注意随着$T$的增长，模型选择的错误概率趋近于零。

### 分类

我们现在考虑一个假设采用时序协议2的问题。

决策者想要将观察序列$\{w_t\}_{t=1}^T$的组成部分分类为来自$f$或$g$。

决策者使用以下分类规则：

$$
\begin{aligned}
w_t  & \ {\rm 来自 \ }  f  \ {\rm 如果 \ } l_t > 1 \\
w_t  & \ {\rm 来自 \ } g  \ {\rm 如果 \ } l_t \leq 1 . 
\end{aligned}
$$

在这个规则下，预期的错误分类率为

$$
p(\textrm{misclassification}) = {1 \over 2} (\tilde \alpha_t + \tilde \beta_t) 
$$ (eq:classerrorprob)

其中$\tilde \alpha_t = {\rm Prob}(l_t < 1 \mid f)$且$\tilde \beta_t = {\rm Prob}(l_t \geq 1 \mid g)$。

由于每个$t$的决策边界都相同，决策边界可以通过以下方式计算：

```{code-cell} ipython3
root = brentq(lambda w: f(w) / g(w) - 1, 0.001, 0.999)
```

我们可以绘制$f$和$g$的分布以及决策边界

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(7, 6))

w_range = np.linspace(1e-5, 1-1e-5, 1000)
f_values = [f(w) for w in w_range]
g_values = [g(w) for w in w_range]
ratio_values = [f(w)/g(w) for w in w_range]

ax.plot(w_range, f_values, 'b-', 
        label=r'$f(w) \sim Beta(1,1)$', linewidth=2)
ax.plot(w_range, g_values, 'r-', 
        label=r'$g(w) \sim Beta(3,1.2)$', linewidth=2)

type1_prob = 1 - beta_dist.cdf(root, F_a, F_b)
type2_prob = beta_dist.cdf(root, G_a, G_b)

w_type1 = w_range[w_range >= root]
f_type1 = [f(w) for w in w_type1]
ax.fill_between(w_type1, 0, f_type1, alpha=0.3, color='blue', 
                label=fr'$\tilde \alpha_t = {type1_prob:.2f}$')

w_type2 = w_range[w_range <= root]
g_type2 = [g(w) for w in w_type2]
ax.fill_between(w_type2, 0, g_type2, alpha=0.3, color='red', 
                label=fr'$\tilde \beta_t = {type2_prob:.2f}$')

ax.axvline(root, color='green', linestyle='--', alpha=0.7, 
            label=f'decision boundary: $w=${root:.3f}')

ax.set_xlabel('w')
ax.set_ylabel('概率密度')
ax.legend()

plt.tight_layout()
plt.show()
```

在绿色垂直线的左侧，$g < f$，所以 $l_t < 1$；因此，落在绿线左侧的 $w_t$ 被归类为 $g$ 类个体。

* 橙色阴影区域等于 $\beta$ -- 将实际为 $f$ 类的个体错误分类为 $g$ 类的概率。

在绿色垂直线的右侧，$g > f$，所以 $l_t > 1$；因此，落在绿线右侧的 $w_t$ 被归类为 $f$ 类个体。

* 蓝色阴影区域等于 $\alpha$ -- 将实际为 $g$ 类的个体错误分类为 $f$ 类的概率。

这给了我们计算理论分类错误概率的线索

```{code-cell} ipython3
# 计算理论 tilde α_t 和 tilde β_t
def α_integrand(w):
    """用于计算 tilde α_t = P(l_t < 1 | f) 的积分"""
    return f(w) if f(w) / g(w) < 1 else 0

def β_integrand(w):
    """用于计算 tilde β_t = P(l_t >= 1 | g) 的积分"""
    return g(w) if f(w) / g(w) >= 1 else 0

# 计算积分
α_theory, _ = quad(α_integrand, 0, 1, limit=100)
β_theory, _ = quad(β_integrand, 0, 1, limit=100)

theory_error = 0.5 * (α_theory + β_theory)

print(f"理论 tilde α_t = {α_theory:.4f}")
print(f"理论 tilde β_t = {β_theory:.4f}")
print(f"理论分类错误概率 = {theory_error:.4f}")
```

现在我们模拟时序协议2并计算分类错误概率。

在下一个单元格中，我们还将理论分类准确率与实验分类准确率进行比较

```{code-cell} ipython3
# 分析协议2
result_p2 = analyze_protocol_2(π_minus_1, T_max, N_simulations, f, g, 
                              theory_error, (F_a, F_b), (G_a, G_b))
```

让我们观察随着观测数据的不断累积，这两种时序协议所做出的决策。

```{code-cell} ipython3
# 比较两种协议
compare_protocols(result_p1, result_p2)
```

从上图可以看出：

- 对于两种时序协议，误差概率都从相同的水平开始，只是受到一些随机性的影响。

- 对于时序协议1，随着样本量的增加，误差概率会降低，因为我们只做**一个**决定 -- 即选择是$f$还是$g$支配**所有**个体。更多的数据提供了更好的证据。

- 对于时序协议2，误差概率保持不变，因为我们在做**多个**决定 -- 对每个观测都要做一个分类决定。

**注意：**思考一下大数定律是如何应用于计算模型选择问题和分类问题的误差概率的。

### 误差概率和散度度量

一个合理的猜测是，似然比区分分布$f$和$g$的能力取决于它们有多"不同"。

我们在{doc}`divergence_measures`中已经学习了一些衡量分布之间"差异"的度量。

现在让我们研究两个在模型选择和分类背景下有用的分布之间"差异"的度量。

回顾一下，概率密度$f$和$g$之间的Chernoff熵定义为：

$$
C(f,g) = - \log \min_{\phi \in (0,1)} \int f^\phi(x) g^{1-\phi}(x) dx
$$

模型选择误差概率的上界是

$$
e^{-C(f,g)T} .
$$

让我们用Python代码来数值计算Chernoff熵

```{code-cell} ipython3
def chernoff_integrand(ϕ, f, g):
    """
    计算Chernoff熵的被积函数
    """
    def integrand(w):
        return f(w)**ϕ * g(w)**(1-ϕ)
    
    result, _ = quad(integrand, 1e-5, 1-1e-5)
    return result

def compute_chernoff_entropy(f, g):
    """
    计算Chernoff熵C(f,g)
    """
    def objective(ϕ):
        return chernoff_integrand(ϕ, f, g)
    
    # 在(0,1)区间内找到最小值
    result = minimize_scalar(objective, 
                             bounds=(1e-5, 1-1e-5), 
                             method='bounded')
    min_value = result.fun
    ϕ_optimal = result.x
    
    chernoff_entropy = -np.log(min_value)
    return chernoff_entropy, ϕ_optimal
C_fg, ϕ_optimal = compute_chernoff_entropy(f, g)
print(f"Chernoff熵C(f,g) = {C_fg:.4f}")
print(f"最优ϕ = {ϕ_optimal:.4f}")
```

现在让我们来研究 $e^{-C(f,g)T}$ 作为 $T$ 的函数时的表现，并将其与模型选择错误概率进行比较

```{code-cell} ipython3
T_range = np.arange(1, T_max+1)
chernoff_bound = np.exp(-C_fg * T_range)

# 绘制比较图
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(T_range, chernoff_bound, 'r-', linewidth=2, 
           label=f'$e^{{-C(f,g)T}}$')
ax.semilogy(T_range, result_p1['error_prob'], 'b-', linewidth=2, 
           label='模型选择错误概率')

ax.set_xlabel('T')
ax.set_ylabel('错误概率（对数刻度）')
ax.legend()
plt.tight_layout()
plt.show()
```

显然，$e^{-C(f,g)T}$是误差率的上界。

在`{doc}`divergence_measures`中，我们还研究了**Jensen-Shannon散度**作为分布之间的对称距离度量。

我们可以使用Jensen-Shannon散度来测量分布$f$和$g$之间的距离，并计算它与模型选择错误概率的协方差。

我们还可以通过一些Python代码来数值计算Jensen-Shannon散度

```{code-cell} ipython3
def compute_JS(f, g):
    """
    计算Jensen-Shannon散度
    """
    def m(w):
        return 0.5 * (f(w) + g(w))
    
    js_div = 0.5 * compute_KL(f, m) + 0.5 * compute_KL(g, m)
    return js_div
```

现在让我们回到我们之前的猜想，即在大样本量情况下的错误概率与两个分布之间的Chernoff熵有关。

我们通过计算时序协议1下$T=50$时错误概率的对数与散度度量之间的相关性来验证这一点。

在下面的模拟中，自然界从$g$中抽取$N/2$个序列，从$f$中抽取$N/2$个序列。

```{note}
自然界采用这种方式，而不是在每次长度为$T$的模拟之前通过抛掷一枚公平硬币来决定是从$g$还是$f$中抽取。
```

我们使用以下Beta分布对作为$f$和$g$的测试用例

```{code-cell} ipython3
distribution_pairs = [
    # (f_params, g_params)
    ((1, 1), (0.1, 0.2)),
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (0.3, 0.4)),
    ((1, 1), (0.5, 0.5)),
    ((1, 1), (0.7, 0.6)),
    ((1, 1), (0.9, 0.8)),
    ((1, 1), (1.1, 1.05)),
    ((1, 1), (1.2, 1.1)),
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),
    ((1, 1), (2.5, 1.8)),
    ((1, 1), (3, 1.2)),
    ((1, 1), (4, 1)),
    ((1, 1), (5, 1))
]
```

现在让我们运行模拟

```{code-cell} ipython3
# 模拟参数
T_large = 50
N_sims = 5000
N_half = N_sims // 2

# 初始化数组
n_pairs = len(distribution_pairs)
kl_fg_vals = np.zeros(n_pairs)
kl_gf_vals = np.zeros(n_pairs) 
js_vals = np.zeros(n_pairs)
chernoff_vals = np.zeros(n_pairs)
error_probs = np.zeros(n_pairs)
pair_names = []

for i, ((f_a, f_b), (g_a, g_b)) in enumerate(distribution_pairs):
    # 创建密度函数
    f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))

    # 计算散度度量
    kl_fg_vals[i] = compute_KL(f, g)
    kl_gf_vals[i] = compute_KL(g, f)
    js_vals[i] = compute_JS(f, g)
    chernoff_vals[i], _ = compute_chernoff_entropy(f, g)

    # 生成样本
    sequences_f = np.random.beta(f_a, f_b, (N_half, T_large))
    sequences_g = np.random.beta(g_a, g_b, (N_half, T_large))

    # 计算似然比和累积乘积
    _, L_cumulative_f = compute_likelihood_ratios(sequences_f, f, g)
    _, L_cumulative_g = compute_likelihood_ratios(sequences_g, f, g)
    
    # 获取最终值
    L_cumulative_f = L_cumulative_f[:, -1]
    L_cumulative_g = L_cumulative_g[:, -1]

    # 计算错误概率
    error_probs[i] = 0.5 * (np.mean(L_cumulative_f < 1) + 
                            np.mean(L_cumulative_g >= 1))
    pair_names.append(f"Beta({f_a},{f_b}) and Beta({g_a},{g_b})")

cor_data =  {
    'kl_fg': kl_fg_vals,
    'kl_gf': kl_gf_vals,
    'js': js_vals, 
    'chernoff': chernoff_vals,
    'error_prob': error_probs,
    'names': pair_names,
    'T': T_large}
```

现在让我们来可视化这些相关性

```{code-cell} ipython3
:tags: [hide-input]

def plot_error_divergence(data):
    """
    绘制误差概率和散度测量之间的相关性。
    """
    # 过滤掉接近零的误差概率以适应对数刻度
    nonzero_mask = data['error_prob'] > 1e-6
    log_error = np.log(data['error_prob'][nonzero_mask])
    js_vals = data['js'][nonzero_mask]
    chernoff_vals = data['chernoff'][nonzero_mask]

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制相关性的函数
    def plot_correlation(ax, x_vals, x_label, color):
        ax.scatter(x_vals, log_error, alpha=0.7, s=60, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f'T={data["T"]}时的对数误差概率')
        
        # 计算相关性和趋势线
        corr = np.corrcoef(x_vals, log_error)[0, 1]
        z = np.polyfit(x_vals, log_error, 2)
        x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_trend, np.poly1d(z)(x_trend), 
                "r--", alpha=0.8, linewidth=2)
        ax.set_title(f'对数误差概率与{x_label}的关系\n'
                     f'相关性 = {corr:.3f}')
    
    # 绘制两个相关性图
    plot_correlation(ax1, js_vals, 'JS散度', 'C0')
    plot_correlation(ax2, chernoff_vals, 'Chernoff熵', 'C1')

    plt.tight_layout()
    plt.show()

plot_error_divergence(cor_data)
```

显然，Chernoff熵和Jensen-Shannon熵都与模型选择错误概率密切相关。

我们很快将在{doc}`wald_friedman`中遇到相关概念。

(lrp_markov)=
## 马尔可夫链

现在让我们来看看一个非独立同分布随机变量序列的似然比过程。

这里我们假设该序列是由有限状态空间上的马尔可夫链生成的。

我们考虑在相同状态空间{1, 2, ..., n}上的两个n状态不可约非周期马尔可夫链模型，它们具有正转移矩阵$P^{(f)}$、$P^{(g)}$和初始分布$\pi_0^{(f)}$、$\pi_0^{(g)}$。

我们假设自然从链f中采样。

对于样本路径$(x_0, x_1, \ldots, x_T)$，让$N_{ij}$计算从状态i到j的转移次数。

模型$m \in \{f, g\}$下的似然过程为

$$
L_T^{(m)} = \pi_{0,x_0}^{(m)} \prod_{i=1}^n \prod_{j=1}^n \left(P_{ij}^{(m)}\right)^{N_{ij}}
$$

因此，

$$
\log L_T^{(m)} =\log\pi_{0,x_0}^{(m)} +\sum_{i,j}N_{ij}\log P_{ij}^{(m)}
$$

对数似然比为

$$
\log \frac{L_T^{(f)}}{L_T^{(g)}} = \log \frac{\pi_{0,x_0}^{(f)}}{\pi_{0,x_0}^{(g)}} + \sum_{i,j}N_{ij}\log \frac{P_{ij}^{(f)}}{P_{ij}^{(g)}}
$$ (eq:llr_markov)

### KL散度率

根据不可约非周期马尔可夫链的遍历定理，我们有

$$
\frac{N_{ij}}{T} \xrightarrow{a.s.} \pi_i^{(f)}P_{ij}^{(f)} \quad \text{当 } T \to \infty
$$

其中 $\boldsymbol{\pi}^{(f)}$ 是满足 $\boldsymbol{\pi}^{(f)} = \boldsymbol{\pi}^{(f)} P^{(f)}$ 的平稳分布。

因此，

$$
\frac{1}{T}\log \frac{L_T^{(f)}}{L_T^{(g)}} = \frac{1}{T}\log \frac{\pi_{0,x_0}^{(f)}}{\pi_{0,x_0}^{(g)}} + \frac{1}{T}\sum_{i,j}N_{ij}\log \frac{P_{ij}^{(f)}}{P_{ij}^{(g)}}
$$

当 $T \to \infty$ 时，我们有：
- 第一项：$\frac{1}{T}\log \frac{\pi_{0,x_0}^{(f)}}{\pi_{0,x_0}^{(g)}} \to 0$
- 第二项：$\frac{1}{T}\sum_{i,j}N_{ij}\log \frac{P_{ij}^{(f)}}{P_{ij}^{(g)}} \xrightarrow{a.s.} \sum_{i,j}\pi_i^{(f)}P_{ij}^{(f)}\log \frac{P_{ij}^{(f)}}{P_{ij}^{(g)}}$

定义**KL散度率**为

$$
h_{KL}(f, g) = \sum_{i=1}^n \pi_i^{(f)} \underbrace{\sum_{j=1}^n P_{ij}^{(f)} \log \frac{P_{ij}^{(f)}}{P_{ij}^{(g)}}}_{=: KL(P_{i\cdot}^{(f)}, P_{i\cdot}^{(g)})}
$$

其中 $KL(P_{i\cdot}^{(f)}, P_{i\cdot}^{(g)})$ 是逐行的KL散度。

根据遍历定理，我们有

$$
\frac{1}{T}\log \frac{L_T^{(f)}}{L_T^{(g)}} \xrightarrow{a.s.} h_{KL}(f, g) \quad \text{当 } T \to \infty
$$

取期望并使用控制收敛定理，我们得到

$$
\frac{1}{T}E_f\left[\log \frac{L_T^{(f)}}{L_T^{(g)}}\right] \to h_{KL}(f, g) \quad \text{当 } T \to \infty
$$

在这里我们邀请读者停下来比较这个结果与{eq}`eq:kl_likelihood_link`。

让我们在下面的模拟中验证这一点。

### 模拟

让我们通过三状态马尔可夫链的模拟来说明这些概念。

首先编写函数来计算马尔可夫链模型的平稳分布和KL散度率。

```{code-cell} ipython3
:tags: [hide-input]

def compute_stationary_dist(P):
    """
    计算转移矩阵P的平稳分布
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmax(np.abs(eigenvalues))
    stationary = np.real(eigenvectors[:, idx])
    return stationary / stationary.sum()

def markov_kl_divergence(P_f, P_g, pi_f):
    """
    计算两个马尔可夫链之间的KL散度率
    """
    if np.any((P_f > 0) & (P_g == 0)):
        return np.inf
    
    valid_mask = (P_f > 0) & (P_g > 0)
    log_ratios = np.zeros_like(P_f)
    log_ratios[valid_mask] = np.log(P_f[valid_mask] / P_g[valid_mask])
    
    # 用平稳概率加权并求和
    kl_rate = np.sum(pi_f[:, np.newaxis] * P_f * log_ratios)
    return kl_rate

def simulate_markov_chain(P, pi_0, T, N_paths=1000):
    """
    模拟马尔可夫链的N_paths条样本路径
    """
    mc = qe.MarkovChain(P, state_values=None)
    initial_states = np.random.choice(len(P), size=N_paths, p=pi_0)
    paths = np.zeros((N_paths, T+1), dtype=int)
    
    for i in range(N_paths):
        path = mc.simulate(T+1, init=initial_states[i])
        paths[i, :] = path
    
    return paths

def compute_likelihood_ratio_markov(paths, P_f, P_g, π_0_f, π_0_g):
    """
    计算马尔可夫链路径的似然比过程
    """
    N_paths, T_plus_1 = paths.shape
    T = T_plus_1 - 1
    L_ratios = np.ones((N_paths, T+1))
    
    # 初始似然比
    L_ratios[:, 0] = π_0_f[paths[:, 0]] / π_0_g[paths[:, 0]]
    
    # 计算序列似然比
    for t in range(1, T+1):
        prev_states = paths[:, t-1]
        curr_states = paths[:, t]
        
        transition_ratios = (P_f[prev_states, curr_states] / 
                           P_g[prev_states, curr_states])
        L_ratios[:, t] = L_ratios[:, t-1] * transition_ratios
    
    return L_ratios

def analyze_markov_chains(P_f, P_g, 
                T=500, N_paths=1000, plot_paths=True, n_show=50):
    """
    两个马尔可夫链的完整分析
    """
    # 计算平稳分布
    π_f = compute_stationary_dist(P_f)
    π_g = compute_stationary_dist(P_g)
    
    print(f"平稳分布 (f): {π_f}")
    print(f"平稳分布 (g): {π_g}")
    
    # 计算KL散度率
    kl_rate_fg = markov_kl_divergence(P_f, P_g, π_f)
    kl_rate_gf = markov_kl_divergence(P_g, P_f, π_g)
    
    print(f"\nKL散度率 h(f, g): {kl_rate_fg:.4f}")
    print(f"KL散度率 h(g, f): {kl_rate_gf:.4f}")
    
    if plot_paths:
        # 模拟并绘制路径
        paths_from_f = simulate_markov_chain(P_f, π_f, T, N_paths)
        L_ratios_f = compute_likelihood_ratio_markov(
            paths_from_f, P_f, P_g, π_f, π_g)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制个别路径
        for i in range(min(n_show, N_paths)):
            plt.plot(np.log(L_ratios_f[i, :]), alpha=0.3, color='blue', lw=0.8)
        
        # 绘制理论期望
        theory_line = kl_rate_fg * np.arange(T+1)
        plt.plot(theory_line, 'k--', linewidth=2.5, 
                label=r'$T \times h_{KL}(f,g)$')
        
        # 绘制经验均值
        avg_log_L = np.mean(np.log(L_ratios_f), axis=0)
        plt.plot(avg_log_L, 'r-', linewidth=2.5, 
                label='经验平均值', alpha=0.7)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel(r'$T$')
        plt.ylabel(r'$\log L_T$')
        plt.title('马尔可夫链似然比(本质 = f)')
        plt.legend()
        plt.show()
    
    return {
        'stationary_f': π_f,
        'stationary_g': π_g,
        'kl_rate_fg': kl_rate_fg,
        'kl_rate_gf': kl_rate_gf
    }

def compute_markov_selection_error(T_values, P_f, P_g, π_0_f, π_0_g, N_sim=1000):
    """
    计算马尔可夫链的模型选择错误概率
    """
    errors = []
    
    for T in T_values:
        # 从两个模型中模拟
        paths_f = simulate_markov_chain(P_f, π_0_f, T, N_sim//2)
        paths_g = simulate_markov_chain(P_g, π_0_g, T, N_sim//2)
        
        # 计算似然比
        L_f = compute_likelihood_ratio_markov(paths_f, P_f, P_g, π_0_f, π_0_g)
        L_g = compute_likelihood_ratio_markov(paths_g, P_f, P_g, π_0_f, π_0_g)
        
        # 决策规则：如果L_T >= 1则选择f
        error_f = np.mean(L_f[:, -1] < 1)   # 第一类错误
        error_g = np.mean(L_g[:, -1] >= 1)  # 第二类错误
        
        total_error = 0.5 * (error_f + error_g)
        errors.append(total_error)
    
    return np.array(errors)
```

现在让我们创建一个包含两个不同的3状态马尔可夫链的示例。

我们现在准备模拟路径并可视化似然比是如何演变的。

我们通过绘制经验平均值和理论预测线来验证从平稳分布开始的 $\frac{1}{T}E_f\left[\log \frac{L_T^{(f)}}{L_T^{(g)}}\right] = h_{KL}(f, g)$

```{code-cell} ipython3
# 定义示例马尔可夫链转移矩阵
P_f = np.array([[0.7, 0.2, 0.1],
                [0.3, 0.5, 0.2],
                [0.1, 0.3, 0.6]])

P_g = np.array([[0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]])

markov_results = analyze_markov_chains(P_f, P_g)
```

## 相关讲座

似然过程在贝叶斯学习中扮演重要角色，正如在{doc}`likelihood_bayes`中所描述的，并在{doc}`odu`中得到应用。

似然比过程是Lawrence Blume和David Easley回答他们提出的问题"如果你那么聪明，为什么不富有？" {cite}`blume2006if`的核心，这是讲座{doc}`likelihood_ratio_process_2`的主题。

似然比过程也出现在{doc}`advanced:additive_functionals`中，其中包含了另一个关于上述似然比过程**特殊性质**的说明。

## 练习

```{exercise}
:label: lr_ex1

考虑自然从第三个密度函数$h$生成数据的情况。

设$\{w_t\}_{t=1}^T$是从$h$中得到的独立同分布样本，且$L_t = L(w^t)$是如讲座中定义的似然比过程。

证明：

$$
\frac{1}{t} E_h[\log L_t] = K_g - K_f
$$

其中$K_g, K_f$有限，$E_h |\log f(W)| < \infty$且$E_h |\log g(W)| < \infty$。

*提示：* 首先将$\log L_t$表示为$\log \ell(w_i)$项的和，并与$K_f$和$K_g$的定义进行比较。
```

```{solution-start} lr_ex1
:class: dropdown
```

由于$w_1, \ldots, w_t$是从$h$中得到的独立同分布样本，我们可以写成

$$
\log L_t = \log \prod_{i=1}^t \ell(w_i) = \sum_{i=1}^t \log \ell(w_i) = \sum_{i=1}^t \log \frac{f(w_i)}{g(w_i)}
$$

在$h$下取期望

$$
E_h[\log L_t] 
= E_h\left[\sum_{i=1}^t \log \frac{f(w_i)}{g(w_i)}\right]

= \sum_{i=1}^t E_h\left[\log \frac{f(w_i)}{g(w_i)}\right]
$$

由于 $w_i$ 是同分布的

$$
E_h[\log L_t] = t \cdot E_h\left[\log \frac{f(w)}{g(w)}\right]
$$

其中 $w \sim h$。

因此

$$
\frac{1}{t} E_h[\log L_t] = E_h\left[\log \frac{f(w)}{g(w)}\right] = E_h[\log f(w)] - E_h[\log g(w)]
$$

根据 Kullback-Leibler 散度的定义

$$
K_f = \int h(w) \log \frac{h(w)}{f(w)} dw = E_h[\log h(w)] - E_h[\log f(w)]
$$

这给出

$$
E_h[\log f(w)] = E_h[\log h(w)] - K_f
$$

类似地

$$
E_h[\log g(w)] = E_h[\log h(w)] - K_g
$$

代回得到

$$
\begin{aligned}
\frac{1}{t} E_h[\log L_t] &= E_h[\log f(w)] - E_h[\log g(w)] \\
&= [E_h[\log h(w)] - K_f] - [E_h[\log h(w)] - K_g] \\
&= K_g - K_f
\end{aligned}
$$

```{solution-end}
```

```{exercise}
:label: lr_ex2

基于{ref}`lr_ex1`的结果，解释当 $t \to \infty$ 时在以下情况下 $L_t$ 会发生什么:

1. 当 $K_g > K_f$ 时(即 $f$ 比 $g$ 更"接近" $h$)
2. 当 $K_g < K_f$ 时(即 $g$ 比 $f$ 更"接近" $h$)

将你的答案与{ref}`本节<llr_h>`中的模拟结果联系起来。
```

```{solution-start} lr_ex2
:class: dropdown
```

从{ref}`lr_ex1`中,我们知道:

$$
\frac{1}{t} E_h[\log L_t] = K_g - K_f
$$

**情况1:** 当 $K_g > K_f$ 时

这里, $f$ 比 $g$ 更"接近" $h$。由于 $K_g - K_f > 0$

$$
E_h[\log L_t] = t \cdot (K_g - K_f) \to +\infty \text{ 当 } t \to \infty
$$

根据大数定律，$\frac{1}{t} \log L_t \to K_g - K_f > 0$ 几乎必然成立。

因此 $L_t \to +\infty$ 几乎必然成立。

**情况2:** 当 $K_g < K_f$ 时

这里，$g$ 比 $f$ "更接近" $h$。由于 $K_g - K_f < 0$

$$
E_h[\log L_t] = t \cdot (K_g - K_f) \to -\infty \text{ 当 } t \to \infty
$$

因此通过类似的推理 $L_t \to 0$ 几乎必然成立。

```{solution-end}
```

