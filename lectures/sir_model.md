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
  title: 新冠病毒建模
  headings:
    Overview: 概述
    The SEIR model: SEIR 模型
    The SEIR model::Time path: 时间路径
    The SEIR model::Parameters: 参数
    Implementation: 实现
    Experiments: 实验
    'Experiments::Experiment 1: constant R0 case': 实验1：固定R0的情况
    'Experiments::Experiment 2: changing mitigation': 实验2：改变缓解措施
    Ending lockdown: 解除封锁
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`新冠病毒建模 <single: Modeling COVID-19>`

```{contents} Contents
:depth: 2
```

## 概述

这是由 [Andrew Atkeson](https://sites.google.com/site/andyatkeson/) 提供的用于分析新冠疫情的Python代码。

特别参见

* [NBER工作论文第26867号](https://www.nber.org/papers/w26867)
* [COVID-19工作论文和代码](https://sites.google.com/site/andyatkeson/home/covid-work)

他的这些笔记主要是介绍了定量建模传染病动态研究。

疾病传播使用标准SEIR（易感者-暴露者-感染者-移出者）模型进行建模。

模型动态用常微分方程组表示。

其主要目的是研究通过社交距离实施的抑制措施对感染传播的影响。

本课程主要模拟的是美国的结果，当然，也可以调整参数来研究其他国家。

我们将使用以下标准导入：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
from numpy import exp
```

我们还将使用SciPy的数值例程odeint来求解微分方程。

```{code-cell} ipython3
from scipy.integrate import odeint
```

这个程序调用了FORTRAN库odepack中的编译代码。

## SEIR 模型

在我们将要分析的这个版本的 SEIR 模型中，有四种状态。

假设人群中的所有个体都处于这四种状态之一。

这些状态是：易感（$S$）、暴露（$E$）、感染（$I$）和移除（$R$）。

```{prf:assumption}
* 处于 R 状态的人已被感染，并已经康复或死亡。
* 假设已康复者已获得免疫力。
* 处于暴露组的人尚未具有传染性。
```

### 时间路径

跨状态的流动遵循路径 $S \to E \to I \to R$。

当有效再生数超过 1 时，感染人数最初会增加，但随着易感人群的减少最终会下降。

我们主要关注的是：

* 特定时间的感染人数（这决定了医疗系统是否会不堪重负），以及
* 病例负荷能够被推迟多久（希望能推迟到疫苗到来之时）

用小写字母表示每种状态所占人口比例，其动态方程为

```{math}
:label: sir_system

\begin{aligned}
     \dot s(t)  & = - \beta(t) \, s(t) \,  i(t)
     \\
     \dot e(t)  & = \beta(t) \,  s(t) \,  i(t)  - \sigma e(t)
     \\
     \dot i(t)  & = \sigma e(t)  - \gamma i(t)
\end{aligned}
```

在这些方程中，

* $\beta(t)$ 称为**传播率**（个体之间相互接触并使彼此暴露于病毒的速率）。
* $\sigma$ 称为**感染率**（暴露者转变为感染者的速率）。
* $\gamma$ 称为**移除率**（感染者康复或死亡的速率）。
* 点符号 $\dot y$ 表示时间导数 $dy/dt$。

由于这些状态构成一个划分，我们无需单独对处于 R 状态的人口比例 $r$ 进行建模。

具体而言，人口中"已移除"的比例为 $r = 1 - s - e - i$。

我们还将跟踪 $c = i + r$，即累计病例负荷（即所有目前感染或曾经感染过的人）。

系统 {eq}`sir_system` 可以写成向量形式：

```{math}
:label: dfcv

\dot x = F(x, t),  \qquad x := (s, e, i)
```

其中 $F$ 有适当的定义（见下面的代码）。

### 参数

$\sigma$ 和 $\gamma$ 都被视为固定的、由生物学决定的参数。

与阿特金森的笔记一致，我们设定：

* $\sigma = 1/5.2$，以反映平均 5.2 天的潜伏期。
* $\gamma = 1/18$，以匹配平均 18 天的患病持续时间。

传播率的建模方式为：

* $\beta(t) := R(t) \gamma$，其中 $R(t)$ 是时间 $t$ 时的**有效再生数**。

（这个符号略微令人困惑，因为 $R(t)$ 与代表移除状态的符号 $R$ 是不同的。）

## 实现

首先我们将人口规模设置为与美国相匹配。

```{code-cell} ipython3
pop_size = 3.3e8
```

接下来我们按照上述方法固定参数。

```{code-cell} ipython3
γ = 1 / 18
σ = 1 / 5.2
```

现在我们构建一个函数来表示 {eq}`dfcv` 中的 $F$

```{code-cell} ipython3
def F(x, t, R0=1.6):
    """
    状态向量的时间导数。

        * x是状态向量（类数组）
        * t是时间（标量）
        * R0是有效传播率，默认为常数

    """
    s, e, i = x

    # 计算新增感染人数
    β = R0(t) * γ if callable(R0) else R0 * γ
    ne = β * s * i

    # 导数
    ds = - ne
    de = ne - σ * e
    di = σ * e - γ * i

    return ds, de, di
```

注意 `R0` 可以是常数或给定的时间函数。

初始条件是根据3.3亿人口进行校准的。

$i_0 = 10^{-7}$ 表示最初有33人被感染，而 $e_0 = 4i_0$ 表示有132人处于暴露状态。

设定 $s_0 = 1 - i_0 - e_0$，将剩余人口分配为易感状态，并将初始移除比例设为零。

```{code-cell} ipython3
# 初始条件
i_0 = 1e-7
e_0 = 4 * i_0
s_0 = 1 - i_0 - e_0
```

用向量形式表示的初始条件是

```{code-cell} ipython3
x_0 = s_0, e_0, i_0
```

我们使用 `odeint` 在一系列时间点 `t_vec` 上通过数值积分求解时间路径。

```{code-cell} ipython3
def solve_path(R0, t_vec, x_init=x_0):
    """
    给定R0的时间路径，通过数值积分求解i(t)和c(t)。

    """
    G = lambda x, t: F(x, t, R0)
    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path       # 累计病例
    return i_path, c_path
```

## 实验

让我们用这段代码进行一些实验。

我们要研究的时间段为550天，大约18个月：

```{code-cell} ipython3
t_length = 550
grid_size = 1000
t_vec = np.linspace(0, t_length, grid_size)
```

### 实验1：固定R0的情况

让我们从 `R0`为常数的情况开始。

我们在不同 `R0`值的假设下计算感染人数的时间路径：

```{code-cell} ipython3
R0_vals = np.linspace(1.6, 3.0, 6)
labels = [f'$R_0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths = [], []

for r in R0_vals:
    i_path, c_path = solve_path(r, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
```

这是一些用于绘制时间路径的代码。

```{code-cell} ipython3
def plot_paths(paths, labels, ylabel, times=t_vec):

    fig, ax = plt.subplots()

    for path, label in zip(paths, labels):
        ax.plot(times, path, lw=2, label=label)

    ax.set_xlabel('days')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')

    plt.show()
```

让我们绘制当前病例数占人口的比例。

```{code-cell} ipython3
plot_paths(i_paths, labels, ylabel='fraction of the population')
```

正如预期的那样，较低的有效传播率会推迟感染高峰。

同时也会导致当前病例的峰值降低。

以下是累计病例数（占总人口的比例）：

```{code-cell} ipython3
plot_paths(c_paths, labels, ylabel='fraction of the population')
```

### 实验2：改变缓解措施

让我们来看一个逐步实施缓解措施（例如社交距离）的场景。

以下是一个关于 `R0`随时间变化的函数规范。

```{code-cell} ipython3
def R0_mitigating(t, r0=3, η=1, r_bar=1.6):
    R0 = r0 * exp(- η * t) + (1 - exp(- η * t)) * r_bar
    return R0
```

`R0` 从 3 开始下降到 1.6。

这是由于逐步采取更严格的缓解措施所致。

参数 `η` 控制限制措施实施的速率或速度。

由于 $t$ 以天为单位度量，$\eta$ 以每天为单位度量，其倒数 $1/\eta$ 即为调整周期。

以下数值分别对应5天、10天、20天、50天和100天的调整周期：

```{code-cell} ipython3
η_vals = 1/5, 1/10, 1/20, 1/50, 1/100
labels = [fr'$\eta = {η:.2f}$' for η in η_vals]
```

以下是在这些不同速率下 `R0` 的时间路径：

```{code-cell} ipython3
fig, ax = plt.subplots()

for η, label in zip(η_vals, labels):
    ax.plot(t_vec, R0_mitigating(t_vec, η=η), lw=2, label=label)

ax.set_xlabel('days')
ax.set_ylabel('$R_0$')
ax.legend()
plt.show()
```

让我们计算感染者人数的时间路径：

```{code-cell} ipython3
i_paths, c_paths = [], []

for η in η_vals:
    R0 = lambda t: R0_mitigating(t, η=η)
    i_path, c_path = solve_path(R0, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
```

以下是不同场景下的当前案例：

```{code-cell} ipython3
plot_paths(i_paths, labels, ylabel='fraction of the population')
```

以下是累计病例数（占总人口的比例）：

```{code-cell} ipython3
plot_paths(c_paths, labels, ylabel='fraction of the population')
```

更快地实施缓解措施主要会延迟感染高峰的到来，而对峰值高度的影响较小。

## 解除封锁

以下内容复现了 Andrew Atkeson 关于解除封锁时机的[附加研究结果](https://drive.google.com/file/d/1uS7n-7zq5gfSgrL3S0HByExmpq4Bn3oh/view)。

我们对比两种解封方案：

1. $R_t = 0.5$持续30天，之后17个月$R_t = 2$。这相当于30天后解除封锁。
2. $R_t = 0.5$持续120天，之后14个月$R_t = 2$。这相当于4个月后解除封锁。

这里所考虑的参数设定模型初始时有25,000名活跃感染者，
以及75,000名已经暴露于病毒、即将具有传染性的人群。

```{code-cell} ipython3
# 初始条件
i_0 = 25_000 / pop_size
e_0 = 75_000 / pop_size
s_0 = 1 - i_0 - e_0
x_0 = s_0, e_0, i_0
```

让我们计算路径：

```{code-cell} ipython3
R0_paths = (lambda t: 0.5 if t < 30 else 2,
            lambda t: 0.5 if t < 120 else 2)

labels = [f'场景 {i}' for i in (1, 2)]

i_paths, c_paths = [], []

for R0 in R0_paths:
    i_path, c_path = solve_path(R0, t_vec, x_init=x_0)
    i_paths.append(i_path)
    c_paths.append(c_path)
```

以下是处于活跃感染状态的人口比例：

```{code-cell} ipython3
plot_paths(i_paths, labels, ylabel='fraction of the population')
```

两种情形产生的感染高峰大致相同，但更长的封锁会推迟高峰的到来。

在这些场景下，死亡率会是怎样的呢？

假设1%的病例会导致死亡。

```{code-cell} ipython3
ν = 0.01
```

这是累计死亡人数：

```{code-cell} ipython3
paths = [(c_path - i_path) * ν * pop_size
         for c_path, i_path in zip(c_paths, i_paths)]
plot_paths(paths, labels, ylabel='cumulative deaths')
```

这是每日死亡人数：

```{code-cell} ipython3
paths = [path * ν * γ * pop_size for path in i_paths]
plot_paths(paths, labels, ylabel='deaths per day')
```

如果我们能够将感染高峰进一步推迟到疫苗研发出来之前，就有可能降低累计死亡人数。
