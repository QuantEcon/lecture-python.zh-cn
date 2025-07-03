---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

(mccall_with_sep)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 工作搜寻 II：搜寻与离职

```{index} single: 工作搜寻导论
```

```{contents} 目录
:depth: 2
```

除了Anaconda中包含的内容外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概述

在{doc}`之前的讲座 <mccall_model>`中，我们研究了McCall工作搜寻模型 {cite}`McCall1970`作为理解失业和劳动者决策的一种方式。

在之前的模型中，我们假设工作是永久性的，这不太符合现实。

本讲座将通过引入离职的可能性来使McCall模型更加贴近现实。

一旦引入离职，个体会有不同的考虑：

* 失业不仅仅是一个暂时状态，而是随时可能发生的资本损失
* 失业期间的工作搜寻成为寻找下一份工作的*投资*

我们还将引入效用函数来更好地刻画劳动者的偏好。

让我们首先导入所需的包：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
from numba import jit, float64
from numba.experimental import jitclass
from quantecon.distributions import BetaBinomial
```

## 模型

该模型与{doc}`基础McCall工作搜寻模型 <mccall_model>`类似。

它关注一个无限期生存的劳动者的生活，以及：

* 他或她（为节省一个字符，我们称"他"）在不同工资水平工作的机会
* 摧毁他当前工作的外生事件
* 他在失业期间的决策过程

劳动者可以处于两种状态之一：就业或失业。

他希望最大化：

```{math}
:label: objective

{\mathbb E} \sum_{t=0}^\infty \beta^t u(y_t)
```

在这个阶段，与{doc}`基础模型 <mccall_model>`的唯一区别是我们通过引入效用函数$u$增加了偏好的灵活性。

它满足$u'> 0$和$u'' < 0$。

### 工资过程

为了简化模型，我们不再像{doc}`基础模型 <mccall_model>`那样将状态过程和工资过程分开。

我们直接假设工资报价$\{w_t\}$是从一个共同分布$q$中独立抽取的。

我们用$\mathbb W$表示所有可能的工资值集合。

（在后面的章节中，我们会重新引入独立的状态过程$\{s_t\}$来驱动随机结果，因为这种方式在处理更复杂的模型时会更加方便。）

### 时间安排和决策

在每个时期开始时，个体可以是：

* 失业，或
* 在某个现有工资水平$w_e$就业。

在给定时期开始时，观察到当前工资报价$w_t$。

如果当前*就业*，劳动者：

1. 获得效用$u(w_e)$，并且
1. 以某个（小的）概率$\alpha$被解雇。

如果当前*失业*，劳动者可以接受或拒绝当前报价$w_t$。

如果他接受，则立即以工资$w_t$开始工作。

如果他拒绝，则获得失业补偿$c$。

然后过程重复。

```{note}
我们不允许在就业期间进行工作搜寻---这个主题将在{doc}`后续讲座 <jv>`中讨论。
```

## 求解模型

我们在下文中省略时间下标，用撇号表示下一期的值。

令：

* $v(w_e)$为进入当前时期*就业*且现有工资为$w_e$的劳动者的总终身价值
* $h(w)$为进入当前时期*失业*且收到工资报价$w$的劳动者的总终身价值。

这里的*价值*是指当劳动者在所有未来时间点都做出最优决策时目标函数{eq}`objective`的值。

我们的首要目标是获得这些函数。

### 贝尔曼方程

假设现在劳动者可以计算函数$v$和$h$并在决策中使用它们。

那么$v$和$h$应该满足：

```{math}
:label: bell1_mccall

v(w_e) = u(w_e) + \beta
    \left[
        (1-\alpha)v(w_e) + \alpha \sum_{w' \in \mathbb W} h(w') q(w')
    \right]
```

和

```{math}
:label: bell2_mccall

h(w) = \max \left\{ v(w), \,  u(c) + \beta \sum_{w' \in \mathbb W} h(w') q(w') \right\}
```

方程{eq}`bell1_mccall`表达了以工资$w_e$就业的价值，包括：

* 当前报酬$u(w_e)$加上
* 考虑到$\alpha$被解雇概率的贴现预期价值

方程{eq}`bell2_mccall`表达了失业且手中有报价$w$的价值，作为两个选项的最大值：接受或拒绝当前报价。

接受使劳动者转为就业，因此获得价值$v(w)$。

拒绝导致失业补偿和明天的失业。

方程{eq}`bell1_mccall`和{eq}`bell2_mccall`是该模型的贝尔曼方程。

它们提供了足够的信息来求解$v$和$h$。

(ast_mcm)=
### 简化变换

与其直接求解这些方程，让我们看看是否能简化它们。

（这个过程将类似于我们对普通McCall模型的{ref}`第二次尝试 <mm_op2>`，在那里我们简化了贝尔曼方程。）

首先，令：

```{math}
:label: defd_mm

d := \sum_{w' \in \mathbb W} h(w') q(w')
```

为明天失业的预期价值。

我们现在可以将{eq}`bell2_mccall`写为：

$$
h(w) = \max \left\{ v(w), \,  u(c) + \beta d \right\}
$$

或者，将时间向前移动一个时期：

$$
\sum_{w' \in \mathbb W} h(w') q(w')
 = \sum_{w' \in \mathbb W} \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w')
$$

再次使用{eq}`defd_mm`现在给出：

```{math}
:label: bell02_mccall

d = \sum_{w' \in \mathbb W} \max \left\{ v(w'), \,  u(c) + \beta d \right\} q(w')
```

最后，{eq}`bell1_mccall`现在可以重写为：

```{math}
:label: bell01_mccall

v(w) = u(w) + \beta
    \left[
        (1-\alpha)v(w) + \alpha d
    \right]
```

在最后一个表达式中，我们将$w_e$写为$w$以简化符号。

### 保留工资

假设我们可以使用{eq}`bell02_mccall`和{eq}`bell01_mccall`来求解$d$和$v$。

（我们很快就会这样做。）

然后我们可以确定劳动者的最优行为。

从{eq}`bell2_mccall`中，我们看到失业个体接受当前报价$w$如果$v(w) \geq  u(c) + \beta d$。
这表明接受工作的价值超过了继续搜索的预期价值。

由于更高的工资报价不会让个体更差，$v$是关于$w$的（弱）递增函数。

这意味着个体的最优策略可以用一个保留工资来表示 - 当且仅当工资报价$w$超过某个临界值时接受工作：

$$
w \geq \bar w
\quad \text{其中} \quad
\bar w \text{ 满足 } v(\bar w) =  u(c) + \beta d
$$

### 求解贝尔曼方程

我们将采用与{doc}`第一个工作搜寻讲座 <mccall_model>`相同的迭代方法来求解贝尔曼方程。

具体步骤如下：

1. 首先对$d$和$v$的值进行初始猜测
1. 将这些猜测值代入{eq}`bell02_mccall`和{eq}`bell01_mccall`右侧的表达式
1. 计算得到新的左侧值,并用这些新值更新猜测
1. 重复以上步骤直到收敛

换句话说，我们使用以下规则进行迭代：

```{math}
:label: bell1001

d_{n+1} = \sum_{w' \in \mathbb W}
    \max \left\{ v_n(w'), \,  u(c) + \beta d_n \right\} q(w')
```

```{math}
:label: bell2001

v_{n+1}(w) = u(w) + \beta
    \left[
        (1-\alpha)v_n(w) + \alpha d_n
    \right]
```

从一些初始条件$d_0, v_0$开始。

如前所述，系统总是收敛到真实解---在这种情况下，是满足{eq}`bell02_mccall`和{eq}`bell01_mccall`的$v$和$d$。

（可以通过巴拿赫压缩映射定理获得证明。）

## 实现

让我们实现这个迭代过程。

在代码中，你会看到我们使用一个类来存储与给定模型相关的各种参数和其他对象。

这有助于整理代码并提供一个易于传递给函数的对象。

默认效用函数是CRRA效用函数：

```{code-cell} ipython3
@jit
def u(c, σ=2.0):
    return (c**(1 - σ) - 1) / (1 - σ)
```

另外，这是一个基于Beta-二项分布的默认工资分布：

```{code-cell} ipython3
n = 60                                  # w的n个可能结果
w_default = np.linspace(10, 20, n)      # 10到20之间的工资
a, b = 600, 400                         # 形状参数
dist = BetaBinomial(n-1, a, b)
q_default = dist.pdf()
```

这是我们的McCall模型与离职的即时编译类：

```{code-cell} ipython3
mccall_data = [
    ('α', float64),      # 工作离职率
    ('β', float64),      # 贴现因子
    ('c', float64),      # 失业补偿
    ('w', float64[:]),   # 工资值列表
    ('q', float64[:])    # 随机变量w的概率质量函数
]

@jitclass(mccall_data)
class McCallModel:
    """
    存储与给定模型相关的参数和函数。
    """

    def __init__(self, α=0.2, β=0.98, c=6.0, w=w_default, q=q_default):

        self.α, self.β, self.c, self.w, self.q = α, β, c, w, q


    def update(self, v, d):

        α, β, c, w, q = self.α, self.β, self.c, self.w, self.q

        v_new = np.empty_like(v)

        for i in range(len(w)):
            v_new[i] = u(w[i]) + β * ((1 - α) * v[i] + α * d)

        d_new = np.sum(np.maximum(v, u(c) + β * d) * q)

        return v_new, d_new
```

现在我们迭代直到连续实现之间的差异小于某个小的容差水平。

然后我们将当前迭代作为近似解返回。

```{code-cell} ipython3
@jit
def solve_model(mcm, tol=1e-5, max_iter=2000):
    """
    迭代求解贝尔曼方程直到收敛

    * mcm是McCallModel的实例
    """

    v = np.ones_like(mcm.w)    # v的初始猜测
    d = 1                      # d的初始猜测
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, d_new = mcm.update(v, d)
        error_1 = np.max(np.abs(v_new - v))
        error_2 = np.abs(d_new - d)
        error = max(error_1, error_2)
        v = v_new
        d = d_new
        i += 1

    return v, d
```

### 保留工资：第一次尝试

个体的最优选择由保留工资总结。

如上所述，保留工资是满足$v(\bar w) = h$的$\bar w$，其中$h := u(c) + \beta d$是继续值。

让我们比较$v$和$h$看看它们的样子。

我们将使用代码中的默认参数化。

```{code-cell} ipython3
mcm = McCallModel()
v, d = solve_model(mcm)
h = u(mcm.c) + mcm.β * d

fig, ax = plt.subplots()

ax.plot(mcm.w, v, 'b-', lw=2, alpha=0.7, label='$v$')
ax.plot(mcm.w, [h] * len(mcm.w),
        'g-', lw=2, alpha=0.7, label='$h$')
ax.set_xlim(min(mcm.w), max(mcm.w))
ax.legend()

plt.show()
```

价值$v$是递增的，因为更高的$w$在保持就业的条件下产生更高的工资流。

### 保留工资：计算

这是一个函数`compute_reservation_wage`，它接受`McCallModel`的实例并返回相关的保留工资。

```{code-cell} ipython3
@jit
def compute_reservation_wage(mcm):
    """
    通过找到最小的w使得v(w) >= h来计算McCall模型的保留工资。

    如果不存在这样的w，则w_bar设置为np.inf。
    """

    v, d = solve_model(mcm)
    h = u(mcm.c) + mcm.β * d

    i = np.searchsorted(v, h, side='right')
    w_bar = mcm.w[i]

    return w_bar
```

接下来我们将研究保留工资如何随参数变化。

## 参数的影响

现在我们将研究保留工资如何随不同参数变化。

对于每个参数，我们会展示一副图，并在练习中让你动手来重现这些图形。

### 保留工资和失业补偿

首先，让我们看看$\bar w$如何随失业补偿变化。

在下面的图中，我们使用`McCallModel`类中的默认参数，除了c（它在水平轴上取给定值）

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_c.png

```

正如预期的那样，更高的失业补偿导致劳动者等待更高的工资。

实际上，继续工作搜寻的成本降低了。

### 保留工资和贴现

接下来，让我们研究$\bar w$如何随贴现因子变化。

下一个图绘制了与不同$\beta$值相关的保留工资

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_beta.png

```

同样，结果是直观的：更有耐心的劳动者会等待更高的工资。

### 保留工资和工作破坏

最后，让我们看看$\bar w$如何随工作离职率$\alpha$变化。

更高的$\alpha$意味着劳动者在就业后每个时期面临终止的可能性更大。

```{figure} /_static/lecture_specific/mccall_model_with_separation/mccall_resw_alpha.png

```

再次，结果符合我们的直觉。

如果离职率高，那么等待更高工资的收益就会下降。

因此保留工资较低。

## 练习

```{exercise-start}
:label: mmws_ex1
```

重现上面显示的所有保留工资图。

关于水平轴上的值，使用：

```{code-cell} ipython3
grid_size = 25
c_vals = np.linspace(2, 12, grid_size)         # 失业补偿
beta_vals = np.linspace(0.8, 0.99, grid_size)  # 贴现因子
alpha_vals = np.linspace(0.05, 0.5, grid_size) # 离职率
```

```{exercise-end}
```

```{solution-start} mmws_ex1
:class: dropdown
```

这是第一幅图。

```{code-cell} ipython3
mcm = McCallModel()

w_bar_vals = np.empty_like(c_vals)

fig, ax = plt.subplots()

for i, c in enumerate(c_vals):
    mcm.c = c
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set(xlabel='失业补偿',
       ylabel='保留工资')
ax.plot(c_vals, w_bar_vals, label=r'$\bar w$作为$c$的函数')
ax.legend()

plt.show()
```

这是第二幅图。

```{code-cell} ipython3
fig, ax = plt.subplots()

for i, β in enumerate(beta_vals):
    mcm.β = β
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set(xlabel='贴现因子', ylabel='保留工资')
ax.plot(beta_vals, w_bar_vals, label=r'$\bar w$作为$\beta$的函数')
ax.legend()

plt.show()
```

这是第三幅图。

```{code-cell} ipython3
fig, ax = plt.subplots()

for i, α in enumerate(alpha_vals):
    mcm.α = α
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set(xlabel='离职率', ylabel='保留工资')
ax.plot(alpha_vals, w_bar_vals, label=r'$\bar w$作为$\alpha$的函数')
ax.legend()

plt.show()
```

```{solution-end}
```
