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

我们将使用{doc}`本讲座 <exchangeable>`中描述的设置。

我们将学习的内容包括：

* 似然比过程的一个特殊性质
* 似然比过程如何成为频率派假设检验中的关键要素

* **接收者操作特征曲线**如何总结频率论假设检验中的虚警概率和检验效能的信息
* 在第二次世界大战期间，美国海军制定了一个决策规则，Garret L. Schyler上尉对此提出质疑，并要求Milton Friedman向他解释其合理性，这个话题将在{doc}`本讲座中 <wald_friedman>`进行研究

让我们先导入一些Python工具。

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
```

## 似然比过程

一个非负随机变量 $W$ 具有两个概率密度函数之一，要么是 $f$，要么是 $g$。

在时间开始之前，自然界一劳永逸地决定是从 $f$ 还是 $g$ 中抽取一系列独立同分布的样本。

我们有时会用 $q$ 表示自然界一劳永逸选择的密度函数，所以 $q$ 要么是 $f$ 要么是 $g$，且是永久性的。

自然界知道它永久性地从哪个密度函数中抽样，但我们这些观察者并不知道。

我们知道 $f$ 和 $g$ 这两个密度函数，但不知道自然界选择了哪一个。

但我们想要知道。

为此，我们使用观测值。

我们观察到一个序列 $\{w_t\}_{t=1}^T$，它包含 $T$ 个从 $f$ 或 $g$ 中抽取的独立同分布样本。

我们想要利用这些观测值来推断自然界选择了 $f$ 还是 $g$。

**似然比过程**是完成这项任务的有用工具。

首先，我们定义似然比过程的一个关键组成部分，即时间 $t$ 的似然比，它是如下随机变量

$$

\ell (w_t)=\frac{f\left(w_t\right)}{g\left(w_t\right)},\quad t\geq1.
$$

我们假设 $f$ 和 $g$ 在随机变量 $W$ 的相同可能实现区间上都具有正概率。

这意味着在 $g$ 密度下，$\ell (w_t)=
\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}$
显然是一个均值为1的非负随机变量。

对序列 $\left\{ w_{t}\right\} _{t=1}^{\infty}$ 的**似然比过程**定义为

$$
L\left(w^{t}\right)=\prod_{i=1}^{t} \ell (w_i),
$$

其中 $w^t=\{ w_1,\dots,w_t\}$ 是直到时间 $t$ （包括 $t$）的观测历史。

有时为简便起见，我们会写作 $L_t =  L(w^t)$。

注意，似然比过程满足以下*递归*或*乘法分解*

$$
L(w^t) = \ell (w_t) L (w^{t-1}) .
$$

似然比及其对数是使用 Neyman 和 Pearson {cite}`Neyman_Pearson` 的经典频率派方法进行推断的关键工具。

为了帮助我们理解其工作原理，以下Python代码将$f$和$g$评估为两个不同的beta分布，然后通过从两个概率分布之一生成序列$w^t$（例如，从$g$生成的IID序列）来计算和模拟相关的似然比过程。

```{code-cell} ipython3
# 两个beta分布中的参数
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

## 自然永久地从密度 g 中抽取

我们首先模拟当自然永久地从 $g$ 中抽取时的似然比过程。

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

```{code-cell} ipython3
N, T = l_arr_g.shape

for i in range(N):

    plt.plot(range(T), l_seq_g[i, :], color='b', lw=0.8, alpha=0.5)

plt.ylim([0, 3])
plt.title("$L(w^{t})$ 路径");
```

显然，随着样本长度 $T$ 的增长，大部分概率质量向零偏移

为了更清楚地看到这一点，我们绘制了随时间变化的路径分数 $L\left(w^{t}\right)$ 落在区间 $\left[0, 0.01\right]$ 内的比例。

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_g <= 0.01, axis=0) / N)
```

尽管大部分概率质量明显收敛到接近$0$的一个很小区间内，但在概率密度$g$下，$L\left(w^t\right)$的无条件均值对所有$t$恒等于$1$。

为了验证这个断言，首先注意到如前所述，对所有$t$，无条件均值$E\left[\ell \left(w_{t}\right)\bigm|q=g\right]$等于$1$：

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

因为$L(w^t) = \ell(w_t) L(w^{t-1})$且$\{w_t\}_{t=1}^t$是IID序列，我们有

$$
\begin{aligned}
E\left[L\left(w^{t}\right)\bigm|q=g\right]  &=E\left[L\left(w^{t-1}\right)\ell \left(w_{t}\right)\bigm|q=g\right] \\

&=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g,w^{t-1}\right]\bigm|q=g\right] \\
    &=E\left[L\left(w^{t-1}\right)E\left[\ell \left(w_{t}\right)\bigm|q=g\right]\bigm|q=g\right] \\
    &=E\left[L\left(w^{t-1}\right)\bigm|q=g\right] \\
\end{aligned}
$$

对任意 $t \geq 1$。

数学归纳法表明
$E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ 对所有
$t \geq 1$ 成立。

## 特殊性质

当似然比过程的大部分概率质量在
$t \rightarrow + \infty$ 时堆积在0附近时，
$E\left[L\left(w^{t}\right)\bigm|q=g\right]=1$ 怎么可能成立？

答案必须是当 $t \rightarrow + \infty$ 时，
$L_t$ 的分布变得越来越重尾：
足够多的质量转移到越来越大的 $L_t$ 值上，使得
尽管大部分概率质量堆积在0附近，$L_t$ 的均值仍然保持为1。

为了说明这个特殊性质，我们模拟了多条路径并且

通过在每个时刻$t$对这些路径取平均值来计算$L\left(w^t\right)$的无条件均值。

```{code-cell} ipython3
l_arr_g = simulate(G_a, G_b, N=50000)
l_seq_g = np.cumprod(l_arr_g, axis=1)
```

使用模拟来验证无条件期望值$E\left[L\left(w^{t}\right)\right]$等于1(通过对样本路径取平均)会很有用。

但是在这里仅仅使用标准蒙特卡洛模拟方法会消耗太多计算时间,因此我们不会这样做。

原因是对于较大的$t$值，$L\left(w^{t}\right)$的分布极度偏斜。

因为右尾部的概率密度接近于0，从右尾部采样足够多的点需要太多计算时间。

我们在{doc}`这篇讲座 <imp_sample>`中更详细地解释了这个问题。

在那里我们描述了一种替代方法来计算似然比的均值，即通过从一个_不同的_概率分布中采样来计算一个_不同的_随机变量的均值。

## 自然永久地从密度f中抽样

现在假设在时间0之前，自然界永久地决定反复从密度f中抽样。

虽然似然比 $\ell \left(w_{t}\right)$ 在密度 $g$ 下的均值为 $1$，但在密度 $f$ 下的均值大于 1。

为了证明这一点，我们计算：

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

这反过来意味着似然比过程 $L(w^t)$ 的无条件均值趋向于 $+ \infty$。

下面的模拟验证了这个结论。

请注意 $y$ 轴的刻度。

```{code-cell} ipython3
l_arr_f = simulate(F_a, F_b, N=50000)
l_seq_f = np.cumprod(l_arr_f, axis=1)
```

```{code-cell} ipython3
N, T = l_arr_f.shape
plt.plot(range(T), np.mean(l_seq_f, axis=0))
```

我们还绘制了 $L\left(w^t\right)$ 落入区间 $[10000, \infty)$ 的概率随时间的变化图，观察概率质量向 $+\infty$ 发散的速度。

```{code-cell} ipython3
plt.plot(range(T), np.sum(l_seq_f > 10000, axis=0) / N)
```

## 似然比检验

我们现在描述如何运用Neyman和Pearson {cite}`Neyman_Pearson`的方法来检验历史数据$w^t$是否由密度函数$g$的重复独立同分布抽样生成。

设$q$为数据生成过程，即$q=f \text{ 或 } g$。

在观察到样本$\{W_i\}_{i=1}^t$后，我们想通过执行（频率学派的）假设检验来判断自然是从$g$还是从$f$中抽样。

我们指定：

- 原假设$H_0$：$q=f$
- 备择假设$H_1$：$q=g$

Neyman和Pearson证明，检验这个假设的最佳方法是使用**似然比检验**，其形式为：

- 当$L(W^t) < c$时拒绝$H_0$
- 否则接受$H_0$

其中$c$是一个给定的判别阈值，我们稍后将描述如何选择它。

这个检验是*最佳的*，因为它是一个**一致最优**检验。

为了理解这意味着什么，我们需要定义两个重要事件的概率：

让我们描述与给定阈值 $c$ 相关的检验特征。

这两个概率是：

- 检测概率（= 检验效力 = 1减去第II类错误概率）：

  $$
  1-\beta \equiv \Pr\left\{ L\left(w^{t}\right)<c\mid q=g\right\}
  $$

- 虚警概率（= 显著性水平 = 第I类错误概率）：

  $$
  \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}
  $$

[奈曼-皮尔逊引理](https://en.wikipedia.org/wiki/Neyman–Pearson_lemma)指出，在所有可能的检验中，似然比检验在给定虚警概率的情况下能最大化检测概率。

换句话说，在所有可能的检验中，似然比检验在给定**显著性水平**的情况下能最大化**检验效力**。

为了得到良好的推断结果，我们希望虚警概率较小而检测概率较大。

当样本量 $t$ 固定时，我们可以通过调整 $c$ 来改变这两个概率。

一个令人困扰的"生活就是如此"的事实是，当我们改变临界值$c$时，这两个概率会朝着相同的方向变化。

在没有具体量化第一类和第二类错误所带来的损失的情况下，我们很难说应该*如何*权衡这两种错误的概率。

我们知道增加样本量$t$可以改善统计推断。

下面我们将绘制一些说明性的图表来展示这一点。

我们还将介绍一个用于选择样本量$t$的经典频率派方法。

让我们从一个将阈值$c$固定为$1$的情况开始。

```{code-cell} ipython3
c = 1
```

下面我们绘制上面模拟的累积似然比的对数的经验分布，这些分布是由$f$或$g$生成的。

取对数不会影响概率的计算，因为对数是单调变换。

随着$t$的增加，第一类错误和第二类错误的概率都在减小，这是好事。

这是因为当$g$是数据生成过程时，log$(L(w^t))$的大部分概率质量向$-\infty$移动；而当数据由$f$生成时，log$(L(w^t))$趋向于$\infty$。

log$(L(w^t))$在$f$和$q$下的这种不同行为使得区分$q=f$和$q=g$成为可能。

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('distribution of $log(L(w^t))$ under f or g', fontsize=15)

for i, t in enumerate([1, 7, 14, 21]):
    nr = i // 2
    nc = i % 2

    axs[nr, nc].axvline(np.log(c), color="k", ls="--")

    hist_f, x_f = np.histogram(np.log(l_seq_f[:, t]), 200, density=True)
    hist_g, x_g = np.histogram(np.log(l_seq_g[:, t]), 200, density=True)

    axs[nr, nc].plot(x_f[1:], hist_f, label="dist under f")
    axs[nr, nc].plot(x_g[1:], hist_g, label="dist under g")

    for i, (x, hist, label) in enumerate(zip([x_f, x_g], [hist_f, hist_g], ["Type I error", "Type II error"])):
        ind = x[1:] <= np.log(c) if i == 0 else x[1:] > np.log(c)
        axs[nr, nc].fill_between(x[1:][ind], hist[ind], alpha=0.5, label=label)

    axs[nr, nc].legend()
    axs[nr, nc].set_title(f"t={t}")

plt.show()
```

下图更清楚地显示，当我们固定阈值$c$时，检测概率随着$t$的增加而单调增加，而虚警概率则单调减少。

```{code-cell} ipython3
PD = np.empty(T)
PFA = np.empty(T)

for t in range(T):
    PD[t] = np.sum(l_seq_g[:, t] < c) / N
    PFA[t] = np.sum(l_seq_f[:, t] < c) / N

plt.plot(range(T), PD, label="Probability of detection")
plt.plot(range(T), PFA, label="Probability of false alarm")
plt.xlabel("t")
plt.title("$c=1$")
plt.legend()
plt.show()
```

对于给定的样本量 $t$，阈值 $c$ 唯一确定了两种类型错误的概率。

如果对于固定的 $t$，我们现在释放并移动 $c$，我们将得到检测概率作为虚警概率的函数。

这就产生了所谓的[接收者操作特征曲线](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)。

下面，我们绘制不同样本量 $t$ 的接收者操作特征曲线。

```{code-cell} ipython3
PFA = np.arange(0, 100, 1)

for t in range(1, 15, 4):
    percentile = np.percentile(l_seq_f[:, t], PFA)
    PD = [np.sum(l_seq_g[:, t] < p) / N for p in percentile]

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

注意随着 $t$ 的增加，对于给定的判别阈值 $c$，我们可以确保获得更高的检测概率和更低的虚警概率。

当 $t \rightarrow + \infty$ 时，我们接近完美检测曲线，该曲线在蓝点处呈直角。

对于给定的样本量 $t$，判别阈值 $c$ 决定了接收者操作特征曲线上的一个点。

权衡两种类型错误的概率是由测试设计者决定的。

但我们知道如何选择最小样本量来达到给定的概率目标。

通常，频率学派的目标是在虚警概率有上限的情况下获得高检测概率。

下面我们展示一个例子，其中我们将虚警概率固定在 $0.05$。

做出决策所需的样本量由一个

目标检测概率，例如 $0.9$，如下图所示。

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

一位被命令执行此类测试的海军上尉对此产生了疑虑，他向米尔顿·弗里德曼提出了这些疑虑，我们在{doc}`这篇讲座 <wald_friedman>`中对此进行了描述。

## Kullback–Leibler 散度

现在让我们考虑一种既不是 $g$ 也不是 $f$ 生成数据的情况。

而是由第三个分布 $h$ 生成。

让我们观察当 $h$ 支配数据时，累积似然比 $f/g$ 的表现。

这里的一个关键工具被称为 **Kullback–Leibler 散度**。

它也被称为**相对熵**。

它用来衡量一个概率分布与另一个概率分布的差异程度。

在我们的应用中，我们想要衡量 $f$ 或 $g$ 与 $h$ 的差异。

与我们相关的两个 Kullback–Leibler 散度是 $K_f$ 和 $K_g$，定义如下：

$$
\begin{aligned}

$$
\begin{aligned}
K_{f}   &=E_{h}\left[\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)\frac{f\left(w\right)}{h\left(w\right)}\right] \\
    &=\int\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)\frac{f\left(w\right)}{h\left(w\right)}h\left(w\right)dw \\
    &=\int\log\left(\frac{f\left(w\right)}{h\left(w\right)}\right)f\left(w\right)dw
\end{aligned}
$$

$$
\begin{aligned}
K_{g}   &=E_{h}\left[\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)\frac{g\left(w\right)}{h\left(w\right)}\right] \\
    &=\int\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)\frac{g\left(w\right)}{h\left(w\right)}h\left(w\right)dw \\
    &=\int\log\left(\frac{g\left(w\right)}{h\left(w\right)}\right)g\left(w\right)dw
\end{aligned}
$$

当 $K_g < K_f$ 时，$g$ 比 $f$ 更接近 $h$。

- 在这种情况下，我们会发现 $L\left(w^t\right) \rightarrow 0$。

当 $K_g > K_f$ 时，$f$ 比 $g$ 更接近 $h$。

- 在这种情况下，我们会发现 $L\left(w^t\right) \rightarrow + \infty$

我们现在将尝试一个$h$也是贝塔分布的情况

我们首先设置参数$G_a$和$G_b$，使得
$h$更接近$g$

```{code-cell} ipython3
H_a, H_b = 3.5, 1.8

h = jit(lambda x: p(x, H_a, H_b))
```

```{code-cell} ipython3
x_range = np.linspace(0, 1, 100)
plt.plot(x_range, f(x_range), label='f')
plt.plot(x_range, g(x_range), label='g')
plt.plot(x_range, h(x_range), label='h')

plt.legend()
plt.show()
```

让我们通过求积分计算Kullback-Leibler差异。

```{code-cell} ipython3
def KL_integrand(w, q, h):

    m = q(w) / h(w)

    return np.log(m) * q(w)
```

```{code-cell} ipython3
def compute_KL(h, f, g):

    Kf, _ = quad(KL_integrand, 0, 1, args=(f, h))
    Kg, _ = quad(KL_integrand, 0, 1, args=(g, h))

    return Kf, Kg
```

```{code-cell} ipython3
Kf, Kg = compute_KL(h, f, g)
Kf, Kg
```

我们有 $K_g < K_f$。

接下来，我们可以通过模拟来验证我们关于 $L\left(w^t\right)$ 的猜想。

```{code-cell} ipython3
l_arr_h = simulate(H_a, H_b)
l_seq_h = np.cumprod(l_arr_h, axis=1)
```

下图绘制了随时间变化的路径分数$L\left(w^t\right)$在区间$[0,0.01]$内的比例。

注意当$g$比$f$更接近$h$时，该比例如预期般收敛到1。

```{code-cell} ipython3
N, T = l_arr_h.shape
plt.plot(range(T), np.sum(l_seq_h <= 0.01, axis=0) / N)
```

我们也可以尝试一个比$g$更接近$f$的$h$，这样$K_g$就会大于$K_f$。

```{code-cell} ipython3
H_a, H_b = 1.2, 1.2
h = jit(lambda x: p(x, H_a, H_b))
```

```{code-cell} ipython3
Kf, Kg = compute_KL(h, f, g)
Kf, Kg
```

```{code-cell} ipython3
l_arr_h = simulate(H_a, H_b)
l_seq_h = np.cumprod(l_arr_h, axis=1)
```

现在$L\left(w^t\right)$的概率质量在10000以上的部分趋向于$+\infty$。

```{code-cell} ipython3
N, T = l_arr_h.shape
plt.plot(range(T), np.sum(l_seq_h > 10000, axis=0) / N)
```

## 后续内容

似然过程在贝叶斯学习中扮演着重要角色，正如在{doc}`这篇讲座<likelihood_bayes>`中所描述的，并在{doc}`这篇讲座<odu>`中得到应用。

似然比过程在[这篇讲座](https://python-advanced.quantecon.org/additive_functionals.html)中再次出现，其中包含了另一个关于上述似然比过程**特殊性质**的说明。

