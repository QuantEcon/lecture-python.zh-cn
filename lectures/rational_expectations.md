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

(ree)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`理性预期均衡 <single: Rational Expectations Equilibrium>`

```{contents} 目录
:depth: 2
```

```{epigraph}
"如果你那么聪明，为什么不富有？"
```

除了Anaconda中已有的库外，本课程还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

本讲座介绍*理性预期均衡*的概念。

为了说明这一点，我们描述了一个由Lucas和Prescott提出的线性二次模型的版本{cite}`LucasPrescott1971`。

这篇1971年的论文是引发*理性预期革命*的少数几篇研究文章之一。

我们遵循Lucas和Prescott的做法，采用一个容易"贝尔曼化"的设置（即可以被表述为动态规划问题）。

由于我们对需求和成本使用线性二次设置，我们可以运用{doc}`这篇讲座 <lqcontrol>`中描述的LQ规划技术。

我们将了解代表性个体的问题如何与规划者的问题不同，以及如何使用规划问题来计算理性预期均衡中的数量和价格。

我们还将学习如何将理性预期均衡表示为从*感知运动规律*到*实际运动规律*的映射的[不动点](https://en.wikipedia.org/wiki/Fixed_point_%28mathematics%29)。

感知运动规律和实际运动规律在内生市场整体变量上的相等性，简明扼要地体现了理性预期均衡概念的本质。

最后，我们将学习重要的"大$K$，小$k$"技巧，这是宏观经济学中广泛使用的建模方法。

不过对我们来说

* "大$K$"将变成"大$Y$"
* "小$k$"将变成"小$y$"

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
```

我们还将使用`QuantEcon.py`中的LQ类。

```{code-cell} ipython
from quantecon import LQ
```

### 大Y，小y技巧

这种广泛使用的方法适用于**代表性企业**或经济主体作为"价格接受者"在竞争均衡中运营的情况。

以下设定证明了代表性企业可以代表大量其他企业的概念。

存在一个统一的单位测度的相同企业，记为$\omega \in \Omega = [0,1]$。

企业$\omega$的产出为$y(\omega)$。

所有企业的总产出为$Y = \int_{0}^1 y(\omega) d \, \omega $。

所有企业最终都选择生产相同的产出，因此最终$ y(\omega) = y $且$Y =y = \int_{0}^1 y(\omega) d \, \omega $。

这种设定使我们可以讨论选择生产$y$的代表性企业。

我们要求：

* 当代表性企业或个别企业选择个体产出$y(\omega)$时，将总量$Y$视为给定，但是$\ldots$。

* 最终，$Y = y(\omega) = y$，因此代表性企业确实具有代表性。

大写$Y$、小写$y$的技巧通过以下两点实现这些目标：

* 在设定选择$y$的决策问题时，将$Y$视为不可控变量；但是$\ldots$
* 在解决个体优化问题*之后*，强制执行$Y = y$。

请注意这个策略在讲座展开过程中是如何应用的。

我们首先在一个非常简单的静态环境中应用大写$Y$、小写$y$的技巧。

#### 大写Y、小写y技巧的简单静态示例

考虑一个静态模型，其中单位数量的企业生产一种在竞争市场中销售的同质商品。

这些企业最终都生产和销售产出$y(\omega) = y$。

商品价格$p$位于反需求曲线上

```{math}
:label: ree_comp3d_static

p = a_0 - a_1 Y
```

其中

* $a_i > 0$，对于$i = 0, 1$
* $Y = \int_0^1 y(\omega) d \omega$是市场总产出水平

为了方便，当我们描述单个企业 $\omega \in \Omega$ 的选择问题时，我们通常直接写 $y$ 而不是 $y(\omega)$。

每个企业都有一个总成本函数

$$
c(y) = c_1 y + 0.5 c_2 y^2,
\qquad c_i > 0 \text{ for } i = 1,2
$$

一个代表性企业的利润是 $p y - c(y)$。

使用{eq}`ree_comp3d_static`，我们可以将代表性企业的问题表示为

```{math}
:label: max_problem_static

\max_{y} \Bigl[ (a_0 - a_1 Y) y - c_1 y - 0.5 c_2 y^2 \Bigr]
```

在提出问题{eq}`max_problem_static`时，我们希望企业是一个*价格接受者*。

我们通过将 $p$ 以及因此的 $Y$ 视为对企业而言是外生的来实现这一点。

大 $Y$，小 $y$ 技巧的本质是在对问题{eq}`max_problem_static`中的 $y$ 求一阶条件*之前*，*不*设定 $Y = n y$。

这确保了企业是一个价格接受者。

问题{eq}`max_problem_static`的一阶条件是

```{math}
:label: BigYsimpleFONC

a_0 - a_1 Y - c_1 - c_2 y = 0
```

在这一点上，*而不是在此之前*，我们将 $Y = y$ 代入 {eq}`BigYsimpleFONC` 得到以下线性方程

```{math}
:label: staticY

a_0 - c_1 - (a_1 +  c_2) Y = 0
```

用于求解竞争均衡市场总产出 $Y$。

在求解出 $Y$ 后，我们可以从反需求曲线 {eq}`ree_comp3d_static` 计算出竞争均衡价格 $p$。

### 相关规划问题

定义**消费者剩余**为反需求曲线下的面积：

$$
S_c (Y)= \int_0^Y (a_0 - a_1 s) ds = a_o Y - \frac{a_1}{2} Y^2 .
$$

定义生产的社会成本为

$$ S_p (Y) = c_1 Y + \frac{c_2}{2} Y^2  $$

考虑规划问题

$$
\max_{Y} [ S_c(Y) - S_p(Y) ]
$$

规划问题的一阶必要条件是方程 {eq}`staticY`。

因此，满足 {eq}`staticY` 的 $Y$ 既是竞争均衡产出，也是解决规划问题的产出。

这种结果为偏好竞争均衡提供了理论依据。

### 延伸阅读

本讲的参考文献包括

* {cite}`LucasPrescott1971`
* {cite}`Sargent1987`, 第XIV章
* {cite}`Ljungqvist2012`, 第7章

## 理性预期均衡

```{index} single: Rational Expectations Equilibrium; Definition
```

我们对理性预期均衡的第一个说明涉及一个由单位数量的相同企业组成的市场，每个企业都试图在面临调整成本的情况下最大化利润的现值。

调整成本使企业不得不进行渐进式调整，这反过来又需要考虑未来的价格。

各个企业都明白，通过反向需求曲线，价格是由其他企业的供应量决定的。

因此每个企业都需要预测未来的总产业产出。

在我们的语境中，预测是由对总体状态运动规律的信念产生的。

当这种信念与由该信念引导的生产选择所产生的实际运动规律相一致时，就形成了理性预期均衡。

我们将理性预期均衡表述为一个算子的不动点，该算子将信念映射为最优信念。

(ree_ce)=
### 具有调整成本的竞争均衡

```{index} single: Rational Expectations Equilibrium; Competitive Equilbrium (w. Adjustment Costs)
```

为了说明这一点，考虑n个生产同质产品的企业的集合，这些产品在竞争市场中销售。

每个企业销售产出 $y_t(\omega) = y_t$。

商品价格 $p_t$ 位于反需求曲线上

```{math}
:label: ree_comp3d

p_t = a_0 - a_1 Y_t
```

其中

* $a_i > 0$ 对于 $i = 0, 1$
* $Y_t = \int_0^1 y_t(\omega) d \omega = y_t$ 是市场整体产出水平

(ree_fp)=
#### 企业的问题

每个企业都是价格接受者。

虽然它不面临不确定性，但确实面临调整成本。

具体来说，它选择一个生产计划来最大化

```{math}
:label: ree_obj

\sum_{t=0}^\infty \beta^t r_t
```

其中

```{math}
:label: ree_comp2

r_t := p_t y_t - \frac{ \gamma (y_{t+1} - y_t )^2 }{2},
\qquad  y_0 \text{ 给定}
```

关于参数，

* $\beta \in (0,1)$ 是折现因子
* $\gamma > 0$ 衡量产出率调整的成本

关于时间安排，企业在t时刻选择 $y_{t+1}$ 时能观察到 $p_t$ 和 $y_t$。

要完整地描述企业的优化问题，我们需要明确所有状态变量的动态变化。

这包括企业关心但无法控制的变量，如 $p_t$。

我们现在来处理这个问题。

#### 价格和总产出

根据 {eq}`ree_comp3d`，企业预测市场价格的动机转化为预测总产出 $Y_t$ 的动机。

总产出取决于其他企业的选择。

单个企业 $\omega$ 的产出 $y_t(\omega)$ 对总产出 $\int_0^1 y_t(\omega) d \omega$ 的影响可以忽略不计。

这使得企业有理由认为它们对总产出的预测不会受到自身产出决策的影响。

#### 代表性企业的信念

我们假设企业认为市场总产出 $Y_t$ 遵循如下运动规律

```{math}
:label: ree_hlom

Y_{t+1} =  H(Y_t)
```

其中 $Y_0$ 是已知的初始条件。

*信念函数* $H$ 是一个均衡对象，因此还有待确定。

#### 给定信念下的最优行为

现在，让我们先固定{eq}`ree_hlom`中的特定信念 $H$，并研究企业对它的反应。

设 $v$ 是给定 $H$ 时企业问题的最优价值函数。

价值函数满足贝尔曼方程

```{math}
:label: comp4

v(y,Y) = \max_{y'} \left\{ a_0 y - a_1 y Y - \frac{ \gamma (y' - y)^2}{2}   + \beta v(y', H(Y))\right\}
```

让我们用$h$表示企业的最优策略函数，因此

```{math}
:label: comp9

y_{t+1} = h(y_t, Y_t)
```

其中

```{math}
:label: ree_opbe

h(y, Y) := \textrm{argmax}_{y'}
\left\{ a_0 y - a_1 y Y - \frac{ \gamma (y' - y)^2}{2}   + \beta v(y', H(Y))\right\}
```

显然$v$和$h$都依赖于$H$。

#### 使用一阶必要条件的特征化

在接下来的内容中，基于一阶条件对$h$进行第二次特征化将会很有帮助。

选择$y'$的一阶必要条件是

```{math}
:label: comp5

-\gamma (y' - y) + \beta v_y(y', H(Y) ) = 0
```

Benveniste-Scheinkman {cite}`BenvenisteScheinkman1979`的一个重要且有用的包络定理表明，要对$v$关于$y$求导，我们可以直接对{eq}`comp4`右侧进行求导，得到

$$
v_y(y,Y) = a_0 - a_1 Y + \gamma (y' - y)
$$

将此方程代入 {eq}`comp5` 得到*欧拉方程*

```{math}
:label: ree_comp7

-\gamma (y_{t+1} - y_t) + \beta [a_0 - a_1 Y_{t+1} + \gamma (y_{t+2} - y_{t+1} )] =0
```

企业在以下条件下最优地设定满足 {eq}`ree_comp7` 的产出路径，同时将 {eq}`ree_hlom` 视为给定：

* $(y_0, Y_0)$ 的初始条件。
* 终端条件 $\lim_{t \rightarrow \infty } \beta^t y_t v_y(y_{t}, Y_t) = 0$。

最后这个条件被称为*横截条件*，它作为"无穷远处"的一阶必要条件。

代表性企业的决策规则求解差分方程 {eq}`ree_comp7`，需满足给定的初始条件 $y_0$ 和横截条件。

注意，求解贝尔曼方程 {eq}`comp4` 得到 $v$，然后在 {eq}`ree_opbe` 中求解 $h$，可以得到一个自动满足欧拉方程 {eq}`ree_comp7` 和横截条件的决策规则。

#### 产出的实际运动规律

正如我们所见，一个特定的信念会转化为一个特定的决策规则 $h$。

回想一下，在均衡状态下 $Y_t = y_t$，市场整体产出的*实际运动规律*为

```{math}
:label: ree_comp9a

Y_{t+1} =  h(Y_t, Y_t)
```

因此，当企业相信市场整体产出的运动规律是{eq}`ree_hlom`时，它们的优化行为使得实际运动规律变为{eq}`ree_comp9a`。

(ree_def)=
### 理性预期均衡的定义

具有调整成本模型的*理性预期均衡*或*递归竞争均衡*是一个决策规则 $h$ 和一个总体运动规律 $H$，使得：

1. 在给定信念 $H$ 的情况下，映射 $h$ 是企业的最优政策函数。
1. 运动规律 $H$ 满足对所有 $Y$ 都有 $H(Y)= h(Y,Y)$。

因此，理性预期均衡使感知的运动规律{eq}`ree_hlom`和实际的运动规律{eq}`ree_comp9a`相等。

#### 不动点表征

正如我们所见，企业的最优化问题产生了一个映射$\Phi$，它将市场总产出的感知运动规律$H$映射到实际运动规律$\Phi(H)$。

映射$\Phi$是两个映射的组合，第一个映射通过{eq}`comp4`--{eq}`ree_opbe`将感知运动规律映射到决策规则，第二个映射通过{eq}`ree_comp9a`将决策规则映射到实际运动规律。

理性预期均衡的$H$分量是$\Phi$的一个不动点。

## 计算均衡

```{index} single: Rational Expectations Equilibrium; Computation
```

现在让我们来计算一个理性预期均衡。

### 收缩性的失效

熟悉动态规划论证的读者可能会尝试通过选择某个总体运动规律的初始猜测值$H_0$，然后对$\Phi$进行迭代来解决这个问题。

不幸的是，映射$\Phi$并不是一个压缩映射。

事实上，对$\Phi$的直接迭代并不能保证收敛[^fn_im]。

有些情况下这些迭代会发散。

幸运的是，这里还有另一种可行的方法。

该方法利用了福利经济学基本定理中所表达的均衡与帕累托最优之间的联系（参见{cite}`MCWG1995`）。

Lucas和Prescott {cite}`LucasPrescott1971` 使用这种方法构建了理性预期均衡。

以下是一些细节。

(ree_pp)=
### 规划问题方法

```{index} single: Rational Expectations Equilibrium; Planning Problem Approach
```

我们的解决思路是将市场问题的欧拉方程与单个个体选择问题的欧拉方程相匹配。

正如我们将看到的，这个规划问题可以通过LQ控制（{doc}`linear regulator <lqcontrol>`）来解决。

规划问题的最优数量就是理性预期均衡的数量。

理性预期均衡价格可以作为规划问题中的影子价格获得。

我们首先计算时间$t$时消费者和生产者剩余的总和

```{math}
:label: comp10

s(Y_t, Y_{t+1})
:= \int_0^{Y_t} (a_0 - a_1 x) \, dx - \frac{ \gamma (Y_{t+1} - Y_t)^2}{2}
```

第一项是需求曲线下的面积，而第二项衡量的是产出变化的社会成本。

*规划问题*是选择一个生产计划$\{Y_t\}$来最大化

$$
\sum_{t=0}^\infty \beta^t s(Y_t, Y_{t+1})
$$

同时满足$Y_0$的初始条件。

### 规划问题的解决方案

计算{eq}`comp10`中的积分得到二次形式$a_0 Y_t - a_1 Y_t^2 / 2$。

因此，规划问题的贝尔曼方程为

```{math}
:label: comp12

V(Y) = \max_{Y'}
\left\{a_0  Y - {a_1 \over 2} Y^2 - \frac{ \gamma (Y' - Y)^2}{2} + \beta V(Y') \right\}
```

相关的一阶条件是

```{math}
:label: comp14

-\gamma (Y' - Y) + \beta V'(Y') = 0
```

应用相同的Benveniste-Scheinkman公式得到

$$

V'(Y) = a_0 - a_1 Y + \gamma (Y' - Y)
$$

将其代入方程 {eq}`comp14` 并重新整理，得到欧拉方程

```{math}
:label: comp16

\beta a_0 + \gamma Y_t - [\beta a_1 + \gamma (1+ \beta)]Y_{t+1} + \gamma \beta Y_{t+2} =0
```

### 关键洞察

回到方程 {eq}`ree_comp7` 并对所有 t 设定 $y_t = Y_t$。

通过一些简单的代数运算，你会发现当 $y_t=Y_t$ 时，方程 {eq}`comp16` 和 {eq}`ree_comp7` 是完全相同的。

因此，规划问题的欧拉方程与我们通过以下方式得到的二阶差分方程相匹配：

1. 找到代表性企业的欧拉方程，并且
1. 将使"代表性企业具有代表性"的表达式 $Y_t = y_t$ 代入其中。

如果对这两个差分方程应用相同的终端条件是合适的（确实如此），那么我们就验证了规划问题的解也是理性预期均衡的数量序列。

因此，对于这个例子，我们可以通过构建对应于贝尔曼方程{eq}`comp12`的最优线性调节器问题来计算均衡量。

规划问题的最优策略函数就是代表性企业在理性预期均衡中面临的总体运动规律$H$。

#### 运动规律的结构

正如练习中要求你证明的那样，规划者问题是LQ控制问题这一事实意味着最优策略——因此也是总体运动规律——具有以下形式

```{math}
:label: ree_hlom2

Y_{t+1}
= \kappa_0 + \kappa_1 Y_t
```

其中$\kappa_0, \kappa_1$为某个参数对。

现在我们知道总体运动规律是线性的，从企业的贝尔曼方程{eq}`comp4`可以看出，企业的问题也可以被构建为一个LQ问题。

正如练习中要求你证明的那样，企业问题的LQ表述意味着一个如下形式的运动规律

```{math}
:label: ree_ex5

y_{t+1} = h_0 + h_1 y_t + h_2 Y_t
```

因此，理性预期均衡将由{eq}`ree_hlom2`--{eq}`ree_ex5`中的参数$(\kappa_0, \kappa_1, h_0, h_1, h_2)$来定义。

## 练习

```{exercise}
:label: ree_ex1

考虑{ref}`上述描述的<ree_fp>`企业问题。

假设企业的信念函数$H$如{eq}`ree_hlom2`所示。

将企业的问题表述为一个贴现最优线性调节器问题，注意要详细描述所需的所有对象。

使用[QuantEcon.py](http://quantecon.org/quantecon-py)包中的`LQ`类来解决以下参数值的企业问题：

$$
a_0= 100, a_1= 0.05, \beta = 0.95, \gamma=10, \kappa_0 = 95.5, \kappa_1 = 0.95
$$

将企业问题的解以{eq}`ree_ex5`的形式表示，并给出每个$h_j$的值。

如果有一个单位数量的完全相同的竞争性企业都按照{eq}`ree_ex5`行事，那么对于市场供给的*实际*运动规律{eq}`ree_hlom`意味着什么。
```

```{solution-start} ree_ex1
:class: dropdown
```

要将问题映射到[折现最优线性控制问题](https://python.quantecon.org/lqcontrol.html)中，我们需要定义

- 状态向量$x_t$和控制向量$u_t$
- 定义偏好和状态运动规律的矩阵$A, B, Q, R$

对于状态和控制向量，我们选择

$$
x_t = \begin{bmatrix} y_t \\ Y_t \\ 1 \end{bmatrix},
\qquad
u_t = y_{t+1} - y_{t}
$$

对于$B, Q, R$，我们设定

$$
A =
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & \kappa_1 & \kappa_0 \\
    0 & 0 & 1
\end{bmatrix},
\quad
B = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} ,
\quad
R =
\begin{bmatrix}
    0 & a_1/2 & -a_0/2 \\
    a_1/2 & 0 & 0 \\
    -a_0/2 & 0 & 0
\end{bmatrix},
\quad
Q = \gamma / 2
$$

通过展开可以确认

- $x_t' R x_t + u_t' Q u_t = - r_t$
- $x_{t+1} = A x_t + B u_t$

我们将使用模块 `lqcontrol.py` 来解决在给定参数值下的公司问题。

这将返回一个 LQ 策略 $F$，其解释为 $u_t = - F x_t$，或

$$
y_{t+1} - y_t = - F_0 y_t - F_1 Y_t - F_2
$$

将参数与 $y_{t+1} = h_0 + h_1 y_t + h_2 Y_t$ 对应，得到

$$
h_0 = -F_2, \quad h_1 = 1 - F_0, \quad h_2 = -F_1
$$

这是我们的解决方案

```{code-cell} ipython3
# 模型参数

a0 = 100
a1 = 0.05
β = 0.95
γ = 10.0

# 信念

κ0 = 95.5
κ1 = 0.95

# 构建 LQ 问题

A = np.array([[1, 0, 0], [0, κ1, κ0], [0, 0, 1]])
B = np.array([1, 0, 0])
B.shape = 3, 1
R = np.array([[0, a1/2, -a0/2], [a1/2, 0, 0], [-a0/2, 0, 0]])
Q = 0.5 * γ

# 求解最优策略

lq = LQ(Q, R, A, B, beta=β)
P, F, d = lq.stationary_values()
F = F.flatten()
out1 = f"F = [{F[0]:.3f}, {F[1]:.3f}, {F[2]:.3f}]"
h0, h1, h2 = -F[2], 1 - F[0], -F[1]
out2 = f"(h0, h1, h2) = ({h0:.3f}, {h1:.3f}, {h2:.3f})"

print(out1)
print(out2)
```

这意味着

$$
y_{t+1} = 96.949 + y_t - 0.046 \, Y_t
$$

对于 $n > 1$ 的情况，回想一下 $Y_t = n y_t$，将其与前面的方程结合，得到

$$
Y_{t+1}
= n \left( 96.949 + y_t - 0.046 \, Y_t \right)
= n 96.949 + (1 - n 0.046) Y_t
$$

```{solution-end}
```


```{exercise}
:label: ree_ex2

考虑以下 $\kappa_0, \kappa_1$ 对作为理性预期均衡中总体运动定律组成部分的候选（参见{eq}`ree_hlom2`）。

扩展你为 {ref}`ree_ex1` 编写的程序，确定哪些（如果有的话）满足理性预期均衡的{ref}`定义 <ree_def>`

* (94.0886298678, 0.923409232937)
* (93.2119845412, 0.984323478873)
* (95.0818452486, 0.952459076301)

描述一个迭代算法，该算法使用你为 {ref}`ree_ex1` 编写的程序来计算理性预期均衡。

（你不需要实际使用你所建议的算法）
```

```{solution-start} ree_ex2
:class: dropdown
```

要确定一对 $\kappa_0, \kappa_1$ 是否构成理性预期均衡的总量运动规律组成部分，我们可以按以下步骤进行：

- 确定相应的企业运动规律
  $y_{t+1} = h_0 + h_1 y_t + h_2 Y_t$。
- 检验相关的总量运动规律
  :$Y_{t+1} = n h(Y_t/n, Y_t)$ 是否等价于
  $Y_{t+1} = \kappa_0 + \kappa_1 Y_t$。

在第二步中，我们可以使用 $Y_t = n y_t = y_t$，因此
$Y_{t+1} = n h(Y_t/n, Y_t)$ 变为

$$
Y_{t+1} = h(Y_t, Y_t) = h_0 + (h_1 + h_2) Y_t
$$

因此要检验第二步，我们可以检验 $\kappa_0 = h_0$ 和
$\kappa_1 = h_1 + h_2$。

以下代码实现了这个检验

```{code-cell} ipython3
candidates = ((94.0886298678, 0.923409232937),
              (93.2119845412, 0.984323478873),
              (95.0818452486, 0.952459076301))

for κ0, κ1 in candidates:

    # 构建相关的运动规律
    A = np.array([[1, 0, 0], [0, κ1, κ0], [0, 0, 1]])

    # 求解企业的LQ问题
    lq = LQ(Q, R, A, B, beta=β)
    P, F, d = lq.stationary_values()
    F = F.flatten()
    h0, h1, h2 = -F[2], 1 - F[0], -F[1]

    # 检验均衡条件
    if np.allclose((κ0, κ1), (h0, h1 + h2)):
        print(f'均衡对 = {κ0}, {κ1}')
        print('f(h0, h1, h2) = {h0}, {h1}, {h2}')
        break
```

输出告诉我们答案是第(iii)组，这意味着
$(h_0, h_1, h_2) = (95.0819, 1.0000, -.0475)$。

（注意我们使用`np.allclose`来测试浮点数的相等性，因为精确相等要求太严格）。

关于迭代算法，可以从给定的$(\kappa_0, \kappa_1)$对循环到相关的企业法则，然后再到新的$(\kappa_0, \kappa_1)$对。

这相当于实现了讲座中描述的算子$\Phi$。

（一般来说，无法保证这个迭代过程一定会收敛到理性预期均衡）

```{solution-end}
```



```{exercise}
:label: ree_ex3

回顾{ref}`上述<ree_pp>`规划者问题

1. 将规划者问题表述为LQ问题。
1. 使用练习1中相同的参数值求解
    * $a_0= 100, a_1= 0.05, \beta = 0.95, \gamma=10$
1. 将解表示为$Y_{t+1} = \kappa_0 + \kappa_1 Y_t$的形式。
1. 将你的答案与练习2的结果进行比较。
```

```{solution-start} ree_ex3
:class: dropdown
```

我们需要将规划者问题写成LQ问题。

对于状态和控制向量，我们选择

$$
x_t = \begin{bmatrix} Y_t \\ 1 \end{bmatrix},
\quad
u_t = Y_{t+1} - Y_{t}
$$

对于LQ矩阵，我们设置

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},
\quad
B = \begin{bmatrix} 1 \\ 0 \end{bmatrix},
\quad
R = \begin{bmatrix} a_1/2 & -a_0/2 \\ -a_0/2 & 0 \end{bmatrix},
\quad
Q = \gamma / 2
$$

通过展开计算可以确认

- $x_t' R x_t + u_t' Q u_t = - s(Y_t, Y_{t+1})$
- $x_{t+1} = A x_t + B u_t$

通过获得最优策略并使用 $u_t = - F x_t$ 或

$$
Y_{t+1} - Y_t = -F_0 Y_t - F_1
$$

我们可以通过 $\kappa_0 = -F_1$ 和 $\kappa_1 = 1-F_0$ 得到隐含的总量运动规律。

解决此问题的Python代码如下：

```{code-cell} ipython3
# 构建规划者的LQ问题

A = np.array([[1, 0], [0, 1]])
B = np.array([[1], [0]])
R = np.array([[a1 / 2, -a0 / 2], [-a0 / 2, 0]])
Q = γ / 2

# 求解最优策略

lq = LQ(Q, R, A, B, beta=β)
P, F, d = lq.stationary_values()

# 打印结果

F = F.flatten()
κ0, κ1 = -F[1], 1 - F[0]
print(κ0, κ1)
```

输出得到的$(\kappa_0, \kappa_1)$对与上一个练习中得到的均衡结果相同。

```{solution-end}
```

```{exercise}
:label: ree_ex4

一个垄断者面对产业需求曲线{eq}`ree_comp3d`，并选择$\{Y_t\}$来最大化$\sum_{t=0}^{\infty} \beta^t r_t$，其中

$$
r_t = p_t Y_t - \frac{\gamma (Y_{t+1} - Y_t)^2 }{2}
$$

将此问题表述为LQ问题。

使用与{ref}`ree_ex2`相同的参数计算最优策略。

特别地，求解以下方程中的参数：

$$
Y_{t+1} = m_0 + m_1 Y_t
$$

将你的结果与{ref}`ree_ex2`进行比较并评论。
```

```{solution-start} ree_ex4
:class: dropdown
```

垄断者的LQ问题与前一个练习中的规划者问题几乎相同，除了

$$
R = \begin{bmatrix}
    a_1 & -a_0/2 \\
    -a_0/2 & 0
\end{bmatrix}
$$

问题可以按如下方式求解：

```{code-cell} ipython3
A = np.array([[1, 0], [0, 1]])
B = np.array([[1], [0]])
R = np.array([[a1, -a0 / 2], [-a0 / 2, 0]])
Q = γ / 2

lq = LQ(Q, R, A, B, beta=β)
P, F, d = lq.stationary_values()

F = F.flatten()
m0, m1 = -F[1], 1 - F[0]
print(m0, m1)
```

我们看到垄断者的运动定律大约为
$Y_{t+1} = 73.4729 + 0.9265 Y_t$。

在理性预期的情况下，运动定律大约为
$Y_{t+1} = 95.0818 + 0.9525 Y_t$。

比较这两个运动定律的一种方法是通过它们的不动点，这些不动点给出了每种情况下的长期均衡产出。

对于形如 $Y_{t+1} = c_0 + c_1 Y_t$ 的定律，不动点为
$c_0 / (1 - c_1)$。

如果你计算这些数字，你会发现垄断者采用的长期产量比竞争市场获得的要低，这意味着更高的市场价格。

这与基础静态情况的结果类似

```{solution-end}
```

[^fn_im]: 研究由学习型代理构成的模型是否能收敛到理性预期均衡的文献，其特点是对映射 $\Phi$ 的修改进行迭代，可以近似为 $\gamma \Phi + (1-\gamma)I$。这里 $I$ 是


恒等算子和 $\gamma \in (0,1)$ 是一个*松弛参数*。
参见 {cite}`MarcetSargent1989` 和 {cite}`EvansHonkapohja2001` 中关于这种方法的阐述和应用，该方法用于确定在什么条件下使用最小二乘学习的自适应代理群体会收敛到理性预期均衡。

