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

(cass_koopmans_2)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Cass-Koopmans 竞争均衡

```{contents} 目录
:depth: 2
```

## 概述

本讲座继续我们在 {doc}`Cass-Koopmans 规划模型 <cass_koopmans_1>` 讲座中对 Tjalling Koopmans {cite}`Koopmans` 和 David Cass {cite}`Cass` 用于研究最优资本积累模型的分析。

本讲座说明了一个实际上更普遍的联系，即**计划经济**和一个经济体之间的联系
组织为竞争均衡或**市场经济**。

之前的讲座 {doc}`Cass-Koopmans规划模型 <cass_koopmans_1>` 研究了一个规划问题并使用了以下概念：

- 规划问题的拉格朗日公式，导致一个差分方程系统。
- 用于求解受初始和终端条件约束的差分方程的**射击算法**。
- 描述长期但有限期限经济最优路径的**转折点**特性。

本讲座使用了额外的概念，包括：

- 希克斯-阿罗价格，以约翰·R·希克斯和肯尼思·阿罗命名。
- 规划问题中某些拉格朗日乘数与希克斯-阿罗价格之间的联系。
- 宏观经济动态中广泛使用的**大** $K$ **，小** $k$ 技巧。
    * 我们将在[本讲座](https://python.quantecon.org/rational_expectations.html)中遇到这个技巧
在[本讲座](https://python-advanced.quantecon.org/dyn_stack.html)中也有介绍。
- 利率期限结构理论的**非随机版本**。
- 组织经济的两种方式之间存在密切联系，即：
    * **社会主义**，由中央计划者指挥资源分配
    * **竞争市场**，其中竞争均衡**价格**引导个体消费者和生产者做出选择，使其自私的决定无意中导致了社会最优配置

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
from numba import jit, float64
from numba.experimental import jitclass
import numpy as np
```
## Cass-Koopmans模型回顾

物理环境与{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中的相同。

时间是离散的，取值为$t = 0, 1 , \ldots, T$。

单一商品的产出可以用于消费或投资于实物资本。

资本品是耐用的，但每期以固定比率部分折旧。

我们用$C_t$表示t时期的非耐用消费品。

用$K_t$表示t时期的实物资本存量。

令$\vec{C}$ = $\{C_0,\dots, C_T\}$且
$\vec{K}$ = $\{K_0,\dots,K_{T+1}\}$。

代表性家庭在每个时期t都拥有一单位的劳动力，并且喜欢每个时期的消费品。

代表性家庭在每个时期t非弹性地供应一单位劳动力$N_t$，因此
$N_t =1 \text{ 对所有 } t \in \{0, 1, \ldots, T\}$。

代表性家庭对消费组合的偏好由以下效用函数排序：

$$
U(\vec{C}) = \sum_{t=0}^{T} \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}
$$

其中 $\beta \in (0,1)$ 是贴现因子，$\gamma >0$ 决定单期效用函数的曲率。

我们假设 $K_0 > 0$。

存在一个经济范围内的生产函数

$$
F(K_t,N_t) = A K_t^{\alpha}N_t^{1-\alpha}
$$

其中 $0 < \alpha<1$，$A > 0$。

可行配置 $\vec{C}, \vec{K}$ 满足

$$
C_t + K_{t+1} \leq F(K_t,N_t) + (1-\delta) K_t \quad \text{for all } t \in \{0, 1, \ldots, T\}
$$

其中 $\delta \in (0,1)$ 是资本的折旧率。

### 规划问题

在本讲 {doc}`Cass-Koopmans 规划模型 <cass_koopmans_1>` 中，我们研究了一个规划者选择配置 $\{\vec{C},\vec{K}\}$ 以最大化 {eq}`utility-functional` 并受限于 {eq}`allocation` 的问题。

正如我们将在下面看到的，解决规划问题的配置在竞争均衡中会再次出现。

## 竞争均衡

我们现在研究这个经济的分散化版本。
它与本讲座中研究的计划经济 {doc}`Cass-Koopmans Planning Model <cass_koopmans_1>` 具有相同的技术和偏好结构。

但现在没有计划者了。

有（单位质量的）价格接受者消费者和企业。

市场价格的设定是为了协调由代表性消费者和代表性企业分别做出的不同决策。

有一个代表性消费者，其消费计划的偏好与计划经济中的消费者相同。

消费者（也称为*家庭*）不是由计划者告知消费和储蓄什么，而是在预算约束下自行选择。

- 在每个时间 $t$，消费者从企业获得工资和资本租金——这构成了其在时间 $t$ 的**收入**。
- 消费者决定将多少收入分配给消费或储蓄。
- 家庭可以通过获得额外的实物资本来储蓄
资本（它与时间$t$的消费一比一交换的资本）或通过获取非$t$时期的消费索取权。
- 家庭拥有实物资本和劳动力，并将其租赁给企业。
- 家庭进行消费、提供劳动力，并投资实物资本。
- 利润最大化的代表性企业经营生产技术。
- 企业每期从代表性家庭租用劳动力和资本，并将其产出销售给家庭。
- 代表性家庭和代表性企业都是**价格接受者**，他们认为自己的选择不会影响价格。

```{note}
再次说明，我们可以认为存在单位数量的相同代表性消费者和相同代表性企业。
```

## 市场结构

代表性家庭和代表性企业都是价格接受者。

家庭拥有劳动力和实物资本这两种生产要素。
每个时期，企业都从家庭租用这两种生产要素。

在一个**单一**的大型竞争市场中，家庭将0期的商品与其他所有时期（$t=1, 2, \ldots, T$）的商品进行交易。

### 价格

存在价格序列
$\{w_t,\eta_t\}_{t=0}^T= \{\vec{w}, \vec{\eta} \}$
其中

- $w_t$ 是工资，即t时期的劳动力租赁率

- $\eta_t$ 是t时期的资本租赁率

此外还有一个跨期价格向量 $\{q_t^0\}$，其中

- $q^0_t$ 是0时期一单位t期商品的价格

我们将 $\{q^0_t\}_{t=0}^T$ 称为**希克斯-阿罗价格**，这个名称来自1972年诺贝尔经济学奖得主。

因为这是一个**相对价格**，$q^0_t$ 的计价单位是任意的；我们可以通过将所有价格乘以一个正标量$\lambda > 0$来重新标准化它们。

$q_t^0$ 的单位可以设定为

$$
\frac{\text{0时刻商品数量}}{\text{t时刻商品数量}}
$$

在这种情况下,我们将时刻$0$的消费品作为**计价单位**。

## 企业问题

在时刻$t$,代表性企业雇佣劳动力$\tilde n_t$和资本$\tilde k_t$。

企业在时刻$t$的利润为

$$
F(\tilde k_t, \tilde n_t)-w_t \tilde n_t -\eta_t \tilde k_t
$$

其中$w_t$是时刻$t$的工资率,$\eta_t$是时刻$t$的资本租赁率。

如同计划经济模型中

$$
F(\tilde k_t, \tilde n_t) = A \tilde k_t^\alpha \tilde n_t^{1-\alpha}
$$

### 零利润条件

资本和劳动力的零利润条件为

$$
F_k(\tilde k_t, \tilde n_t) =\eta_t
$$

和

```{math}
:label: Zero-profits

F_n(\tilde k_t, \tilde n_t) =w_t
```

这些条件源于无套利要求。

为了描述这种无套利利润推理,我们首先应用欧拉关于线性齐次函数的定理。
该定理适用于柯布-道格拉斯生产函数，因为它具有规模报酬不变的特性：

$$
\alpha F(\tilde k_t, \tilde n_t) =  F(\alpha  \tilde k_t, \alpha \tilde n_t)
$$

对于 $\alpha \in (0,1)$。

对上述等式两边取偏导数 $\frac{\partial  }{\partial \alpha}$ 得到：

$$
F(\tilde k_t,\tilde n_t) =  \frac{\partial F}{\partial \tilde k_t}
\tilde k_t + \frac{\partial F}{\partial \tilde  n_t} \tilde n_t
$$

将企业利润重写为：

$$
\frac{\partial F}{\partial \tilde k_t} \tilde k_t +
\frac{\partial F}{\partial \tilde  n_t} \tilde n_t-w_t \tilde n_t -\eta_t k_t
$$

或

$$
\left(\frac{\partial F}{\partial \tilde k_t}-\eta_t\right) \tilde k_t +
\left(\frac{\partial F}{\partial \tilde  n_t}-w_t\right) \tilde n_t
$$

因为 $F$ 是一次齐次函数，所以 $\frac{\partial F}{\partial \tilde k_t}$ 和 $\frac{\partial F}{\partial \tilde n_t}$ 是零次齐次函数，因此相对于
$\tilde k_t$ 和 $\tilde n_t$。

如果 $\frac{\partial F}{\partial \tilde k_t}> \eta_t$，那么
企业在每增加一单位 $\tilde k_t$ 时都会获得正利润，
因此它会想要将 $\tilde k_t$ 设置为任意大。

但设置 $\tilde k_t = + \infty$ 在物理上是不可行的，
所以**均衡**价格必须取能阻止企业进行这种套利机会的值。

如果 $\frac{\partial F}{\partial \tilde n_t}> w_t$，
类似的论证也适用。

如果 $\frac{\partial \tilde k_t}{\partial \tilde k_t}< \eta_t$，
企业会想要将 $\tilde k_t$ 设为零，这是不可行的。

为方便起见，我们定义
$\vec{w} =\{w_0, \dots,w_T\}$ 和 $\vec{\eta}= \{\eta_0, \dots, \eta_T\}$。

## 家庭问题

一个代表性家庭生活在 $t=0,1,\dots, T$ 期间。

在 $t$ 时期，家庭向企业出租 1 单位劳动力
和 $k_t$ 单位资本，并获得收入

$$
w_t 1+ \eta_t k_t
$$

在 $t$ 时期，家庭将其收入分配到以下用途
在以下两个类别之间的购买：

* 消费 $c_t$

* 净投资 $k_{t+1} -(1-\delta)k_t$

这里 $\left(k_{t+1} -(1-\delta)k_t\right)$ 是家庭在实物资本上的净投资，而 $\delta \in (0,1)$ 仍然是资本的折旧率。

在 $t$ 期，消费者可以自由购买超过其从向企业提供资本和劳动所得收入的消费品和实物资本投资，只要在其他某些时期其收入超过其购买即可。

消费者在 $t$ 时期消费品的净超额需求是以下差额：

$$
e_t \equiv \left(c_t + (k_{t+1} -(1-\delta)k_t)\right)-(w_t 1 + \eta_t k_t)
$$

令 $\vec{c} = \{c_0,\dots,c_T\}$ 且令 $\vec{k} = \{k_1,\dots,k_{T+1}\}$。

$k_0$ 是给定给家庭的。

家庭面临**单一**预算约束，要求家庭净超额需求的现值必须为零：

$$
\sum_{t=0}^T q^0_t e_t  \leq 0
$$

或

$$
\sum_{t=0}^T q^0_t  \left(c_t + (k_{t+1} -(1-\delta)k_t)\right) \leq \sum_{t=0}^T q^0_t(w_t 1 + \eta_t k_t)  \
$$

作为价格接受者，家庭面对价格体系 $\{q^0_t, w_t, \eta_t\}$ 并选择一个配置来解决如下约束优化问题：

$$
\begin{aligned}& \max_{\vec{c}, \vec{k} }  \sum_{t=0}^T \beta^t u(c_t) \\ \text{subject to} \ \   & \sum_{t=0}^T q_t^0\left(c_t +\left(k_{t+1}-(1-\delta) k_t \right) - (w_t -\eta_t k_t) \right)\leq 0  \notag \end{aligned}
$$

**价格体系**的组成部分具有以下单位：

* $w_t$ 以时间 $t$ 雇佣的劳动单位所获得的时间 $t$ 商品单位来衡量

* $\eta_t$ 以时间 $t$ 雇佣的资本单位所获得的时间 $t$ 商品单位来衡量

* $q_t^0$ 以计价单位对时间 $t$ 商品单位来衡量

### 定义

- **价格体系**是一个序列
  $\{q_t^0,\eta_t,w_t\}_{t=0}^T= \{\vec{q}, \vec{\eta}, \vec{w}\}$。
- **配置**是一个序列
$\{c_t,k_{t+1},n_t=1\}_{t=0}^T = \{\vec{c}, \vec{k}, \vec{n}\}$。
- **竞争均衡**是一个价格体系和资源配置，具有以下特性：
    - 在给定价格体系的情况下，该配置解决了家庭的问题。
    - 在给定价格体系的情况下，该配置解决了企业的问题。

这里的设想是均衡价格体系和资源配置一次性确定。

实际上，我们假设所有交易都在时间$0$之前发生。

## 计算竞争均衡

我们通过使用**猜测和验证**的方法来计算竞争均衡。

- 我们**猜测**均衡价格序列
  $\{\vec{q}, \vec{\eta}, \vec{w}\}$。
- 然后我们**验证**在这些价格下，家庭和
  企业选择相同的配置。

### 价格体系的猜测

在{doc}`Cass-Koopmans规划模型<cass_koopmans_1>`这一讲中，我们计算了解决规划问题的配置$\{\vec{C}, \vec{K}, \vec{N}\}$。
我们使用该分配来构建均衡价格体系的猜测。

```{note}
在这个例子中，这个分配将构成**大**$K$，我们将按照[这个讲座](https://python.quantecon.org/rational_expectations.html)和[这个讲座](https://python-advanced.quantecon.org/dyn_stack.html)的精神，在竞争均衡中应用**大**$K$**，小**$k$技巧。
```

具体来说，我们将使用以下步骤：

* 获取代表性企业和代表性消费者的一阶条件。
* 从这些方程中，通过将企业的选择变量$\tilde k, \tilde n$和消费者的选择变量替换为规划问题解$\vec C, \vec K$，得到一组新的方程。
* 求解由此产生的方程，得到$\{\vec{q}, \vec{\eta}, \vec{w}\}$作为$\vec C, \vec K$的函数。
* 验证在这些价格下，$c_t = C_t, k_t = \tilde k_t = K_t, \tilde n_t = 1$ 对于 $t = 0, 1, \ldots, T$ 成立。

因此，我们猜测对于 $t=0,\dots,T$：

```{math}
:label: eq-price

q_t^0 = \beta^t u'(C_t) 
```

```{math}
:label: eq-price2

w_t = f(K_t) -K_t f'(K_t)
```

```{math}
:label: eq-price3

\eta_t = f'(K_t)
```

在这些价格下，家庭选择的资本为

```{math}
:label: eq-pr4

k^*_t(\vec {q}, \vec{w}, \vec{\eta)} , \quad t \geq 0
```

而企业选择的配置为

$$
\tilde k^*_t(\vec{q}, \vec{w}, \vec{\eta}), \quad t \geq 0
$$

等等。

如果我们对均衡价格体系的猜测是正确的，那么必须有

```{math}
:label: ge1

k_t^*  = \tilde k_t^*
```

```{math}
:label: ge2

1   = \tilde n_t^*
```

$$
c_t^* + k_{t+1}^* - (1-\delta) k_t^*  = F(\tilde k_t^*, \tilde n_t^*)
$$

我们将验证对于 $t=0,\dots,T$，家庭和企业选择的配置都等于规划问题的解：
```{math}
:label: eq-pl

k^*_t = \tilde k^*_t=K_t, \tilde n_t=1, c^*_t=C_t
```

### 验证程序

我们的方法首先是仔细研究家庭和企业优化问题的一阶必要条件。

在我们猜测的价格体系下，我们将验证这两组一阶条件在规划问题的解决方案中是否都得到满足。

### 家庭的拉格朗日函数

为了解决家庭的问题，我们构建拉格朗日函数

$$
\mathcal{L}(\vec{c},\vec{k},\lambda) = \sum_{t=0}^T \beta^t u(c_t)+ \lambda \left(\sum_{t=0}^T q_t^0\left(\left((1-\delta) k_t -w_t\right)
+\eta_t k_t -c_t  - k_{t+1}\right)\right)
$$

并处理极小极大问题：

$$
\min_{\lambda} \max_{\vec{c},\vec{k}}  \mathcal{L}(\vec{c},\vec{k},\lambda)
$$

一阶条件是

```{math}
:label: cond1

c_t: \quad \beta^t u'(c_t)-\lambda q_t^0=0 \quad  t=0,1,\dots,T
```

```{math}
:label: cond2
k_t: \quad -\lambda q_t^0 \left[(1-\delta)+\eta_t \right]+\lambda q^0_{t-1}=0 \quad  t=1,2,\dots,T+1
```

```{math}
:label: cond3

\lambda:  \quad \left(\sum_{t=0}^T q_t^0\left(c_t + \left(k_{t+1}-(1-\delta) k_t\right) -w_t -\eta_t k_t\right)\right) \leq 0
```

```{math}
:label: cond4

k_{T+1}: \quad -\lambda q_0^{T+1} \leq 0, \ \leq 0 \text{ 若 } k_{T+1}=0; \ =0 \text{ 若 } k_{T+1}>0
```

现在我们将我们猜测的价格代入并进行一些代数运算，希望能从本讲座{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中恢复所有一阶必要条件{eq}`constraint1`-{eq}`constraint4`。

将{eq}`cond1`和{eq}`eq-price`结合，我们得到：

$$
u'(C_t) = \mu_t
$$

这就是{eq}`constraint1`。

将{eq}`cond2`、{eq}`eq-price`和{eq}`eq-price3`结合，我们得到：

```{math}
:label: co-re

-\lambda \beta^t \mu_t\left[(1-\delta) +f'(K_t)\right] +\lambda \beta^{t-1}\mu_{t-1}=0
```

通过在{eq}`co-re`两边除以$\lambda$
两边（由于u'>0，所以非零）我们得到：

$$
\beta^t \mu_t [(1-\delta+f'(K_t)] = \beta^{t-1} \mu_{t-1}
$$

或

$$
\beta \mu_t [(1-\delta+f'(K_t)] = \mu_{t-1}
$$

这就是{eq}`constraint2`。

将{eq}`cond3`、{eq}`eq-price`、{eq}`eq-price2`和{eq}`eq-price3`结合起来，在将{eq}`cond3`两边乘以$\lambda$后，我们得到

$$
\sum_{t=0}^T \beta^t \mu_{t} \left(C_t+ (K_{t+1} -(1-\delta)K_t)-f(K_t)+K_t f'(K_t)-f'(K_t)K_t\right) \leq 0
$$

简化为

$$
\sum_{t=0}^T  \beta^t \mu_{t} \left(C_t +K_{t+1} -(1-\delta)K_t - F(K_t,1)\right) \leq 0
$$

由于对于$t =0, \ldots, T$，$\beta^t \mu_t >0$，因此

$$
C_t+K_{t+1}-(1-\delta)K_t -F(K_t,1)=0 \quad  \text{ for all }t \text{ in } \{0, 1, \ldots, T\}
$$

这就是{eq}`constraint3`。

将{eq}`cond4`和{eq}`eq-price`结合，我们得到：

$$
-\beta^{T+1} \mu_{T+1} \leq 0
$$

两边除以$\beta^{T+1}$得到

$$
-\mu_{T+1} \leq 0
$$

这就是规划问题的{eq}`constraint4`。
因此，在我们猜测的均衡价格体系下，解决规划问题的配置也同样解决了在竞争均衡中代表性家庭面临的问题。

### 代表性企业的问题

现在我们来看竞争均衡中企业面临的问题：

如果我们将{eq}`eq-pl`代入所有t时期的{eq}`Zero-profits`中，我们得到

$$
\frac{\partial F(K_t, 1)}{\partial K_t} = f'(K_t) = \eta_t
$$

这就是{eq}`eq-price3`。

如果我们现在将{eq}`eq-pl`代入所有t时期的{eq}`Zero-profits`中，我们得到：

$$
\frac{\partial F(\tilde K_t, 1)}{\partial \tilde L_t} = f(K_t)-f'(K_t)K_t=w_t
$$

这正好是{eq}`eq-pr4`。

因此，在我们猜测的均衡价格体系下，解决规划问题的配置也同样解决了竞争均衡中企业面临的问题。

根据{eq}`ge1`和{eq}`ge2`，这个配置与解决消费者问题的配置是相同的。

```{note}
由于预算集只受相对价格的影响，$\{q^0_t\}$ 的确定仅限于乘以一个正常数。

**标准化：** 我们可以自由选择一个使 $\lambda=1$ 的 $\{q_t^0\}$，这样我们就是用时间 0 商品的边际效用单位来度量 $q_t^0$。

我们将在下面绘制 $q, w, \eta$ 以显示这些均衡价格引起的总体变动与我们之前在规划问题中看到的相同。

为了继续，我们引入 {doc}`Cass-Koopmans 规划模型 <cass_koopmans_1>` 中用于求解规划问题的 Python 代码。

首先让我们定义一个用于存储表征经济的参数和函数的 `jitclass`。

```{code-cell} python3
planning_data = [
    ('γ', float64),    # 相对风险厌恶系数
    ('β', float64),    # 贴现因子
    ('δ', float64),    # 资本折旧率
    ('α', float64),    # 人均资本回报
    ('A', float64)     # 技术
]
```
```{code-cell} python3
@jitclass(planning_data)
class PlanningProblem():

    def __init__(self, γ=2, β=0.95, δ=0.02, α=0.33, A=1):

        self.γ, self.β = γ, β
        self.δ, self.α, self.A = δ, α, A

    def u(self, c):
        '''
        效用函数
        注意：如果你有一个难以手动求解的效用函数
        你可以使用自动或符号微分
        参见 https://github.com/HIPS/autograd
        '''
        γ = self.γ

        return c ** (1 - γ) / (1 - γ) if γ!= 1 else np.log(c)

    def u_prime(self, c):
        '效用函数的导数'
        γ = self.γ

        return c ** (-γ)

    def u_prime_inv(self, c):
        '效用函数导数的逆函数'
        γ = self.γ

        return c ** (-1 / γ)

    def f(self, k):
        '生产函数'
        α, A = self.α, self.A

        return A * k ** α

    def f_prime(self, k):
        '生产函数的导数'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)

    def f_prime_inv(self, k):
        '生产函数导数的逆函数'
        α, A = self.α, self.A

        return (k / (A * α)) ** (1 / (α - 1))

    def next_k_c(self, k, c):
        ''''
        给定当前资本Kt和任意可行的
        消费选择Ct，通过状态转移定律计算Kt+1
        并通过欧拉方程计算最优Ct+1。
        '''
        β, δ = self.β, self.δ
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f, f_prime = self.f, self.f_prime

        k_next = f(k) + (1 - δ) * k - c
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next) + (1 - δ))))

        return k_next, c_next
```
```{code-cell} python3
@jit
def shooting(pp, c0, k0, T=10):
    '''
    给定资本的初始条件k0和消费的初始猜测值c0，
    使用状态转移定律和欧拉方程计算T期内c和k的完整路径。
    '''
    if c0 > pp.f(k0):
        print("初始消费不可行")

        return None

    # 初始化c和k的向量
    c_vec = np.empty(T+1)
    k_vec = np.empty(T+2)

    c_vec[0] = c0
    k_vec[0] = k0

    for t in range(T):
        k_vec[t+1], c_vec[t+1] = pp.next_k_c(k_vec[t], c_vec[t])

    k_vec[T+1] = pp.f(k_vec[T]) + (1 - pp.δ) * k_vec[T] - c_vec[T]

    return c_vec, k_vec
```
```{code-cell} python3
@jit
def bisection(pp, c0, k0, T=10, tol=1e-4, max_iter=500, k_ter=0, verbose=True):

    # 初始猜测c0的边界
    c0_upper = pp.f(k0)
    c0_lower = 0

    i = 0
    while True:
        c_vec, k_vec = shooting(pp, c0, k0, T)
        error = k_vec[-1] - k_ter

        # 检查是否满足终端条件
        if np.abs(error) < tol:
            if verbose:
                print('在第', i+1, '次迭代成功收敛')
            return c_vec, k_vec

        i += 1
        if i == max_iter:
            if verbose:
                print('收敛失败。')
            return c_vec, k_vec

        # 如果迭代继续，更新边界和c0的猜测值
        if error > 0:
            c0_lower = c0
        else:
            c0_upper = c0

        c0 = (c0_lower + c0_upper) / 2
```
```{code-cell} python3
pp = PlanningProblem()

# 稳态
ρ = 1 / pp.β - 1
k_ss = pp.f_prime_inv(ρ+pp.δ)
c_ss = pp.f(k_ss) - pp.δ * k_ss
```
上述来自讲座 {doc}`Cass-Koopmans规划模型 <cass_koopmans_1>` 的代码让我们能够计算规划问题的最优配置。

* 从前面的分析中，我们知道这也将是一个与竞争均衡相关的配置。

现在我们准备引入Python代码，这些代码用于计算竞争均衡中出现的其他对象。

```{code-cell} python3
@jit
def q(pp, c_path):
    # 这里我们选择u'(c_0)作为计价单位 -- 这是q^(t_0)_t
    T = len(c_path) - 1
    q_path = np.ones(T+1)
    q_path[0] = 1
    for t in range(1, T+1):
        q_path[t] = pp.β ** t * pp.u_prime(c_path[t])
    return q_path

@jit
def w(pp, k_path):
    w_path = pp.f(k_path) - k_path * pp.f_prime(k_path)
    return w_path

@jit
def η(pp, k_path):
    η_path = pp.f_prime(k_path)
    return η_path
```
现在我们对每个$T$进行计算和绘图

```{code-cell} python3
T_arr = [250, 150, 75, 50]

fix, axs = plt.subplots(2, 3, figsize=(13, 6))
titles = ['Arrow-Hicks价格', '劳动租赁率', '资本租赁率',
          '消费', '资本', '拉格朗日乘数']
ylabels = ['$q_t^0$', '$w_t$', '$\eta_t$', '$c_t$', '$k_t$', '$\mu_t$']

for T in T_arr:
    c_path, k_path = bisection(pp, 0.3, k_ss/3, T, verbose=False)
    μ_path = pp.u_prime(c_path)

    q_path = q(pp, c_path)
    w_path = w(pp, k_path)[:-1]
    η_path = η(pp, k_path)[:-1]
    paths = [q_path, w_path, η_path, c_path, k_path, μ_path]

    for i, ax in enumerate(axs.flatten()):
        ax.plot(paths[i])
        ax.set(title=titles[i], ylabel=ylabels[i], xlabel='t')
        if titles[i] == '资本':
            ax.axhline(k_ss, lw=1, ls='--', c='k')
        if titles[i] == '消费':
            ax.axhline(c_ss, lw=1, ls='--', c='k')

plt.tight_layout()
plt.show()
```
#### 变化的曲率

现在我们来看看当保持$T$不变，但允许曲率参数$\gamma$变化时，结果会如何变化，从低于稳态的$K_0$开始。

我们绘制$T=150$的结果

```{code-cell} python3
T = 150
γ_arr = [1.1, 4, 6, 8]

fix, axs = plt.subplots(2, 3, figsize=(13, 6))

for γ in γ_arr:
    pp_γ = PlanningProblem(γ=γ)
    c_path, k_path = bisection(pp_γ, 0.3, k_ss/3, T, verbose=False)
    μ_path = pp_γ.u_prime(c_path)

    q_path = q(pp_γ, c_path)
    w_path = w(pp_γ, k_path)[:-1]
    η_path = η(pp_γ, k_path)[:-1]
    paths = [q_path, w_path, η_path, c_path, k_path, μ_path]

    for i, ax in enumerate(axs.flatten()):
        ax.plot(paths[i], label=f'$\gamma = {γ}$')
        ax.set(title=titles[i], ylabel=ylabels[i], xlabel='t')
        if titles[i] == 'Capital':
            ax.axhline(k_ss, lw=1, ls='--', c='k')
        if titles[i] == 'Consumption':
            ax.axhline(c_ss, lw=1, ls='--', c='k')

axs[0, 0].legend()
plt.tight_layout()
plt.show()
```
调整 $\gamma$ 意味着调整个体对消费平滑的偏好程度。

较高的 $\gamma$ 意味着个体更倾向于平滑消费，
导致达到稳态配置的收敛速度更慢。

较低的 $\gamma$ 意味着个体较少倾向于平滑消费，
导致达到稳态配置的收敛速度更快。

## 收益率曲线和希克斯-阿罗价格

我们回到希克斯-阿罗价格，并计算它们与不同期限贷款的**收益率**之间的关系。

这将让我们绘制一条**收益率曲线**，该曲线展示了期限为 $j=1, 2, \ldots$ 的债券收益率与 $j=1,2, \ldots$ 的关系。

我们使用以下公式。

在时间 $t_0$ 发放并在时间 $t > t_0$ 到期的贷款的**到期收益率**

$$
r_{t_0,t}= -\frac{\log q^{t_0}_t}{t - t_0}
$$

基准年份为 $t_0\leq t$ 的希克斯-阿罗价格体系满足

$$
q^{t_0}_t = \beta^{t-t_0} \frac{u'(c_t)}{u'(c_{t_0})}= \beta^{t-t_0}
\frac{c_t^{-\gamma}}{c_{t_0}^{-\gamma}}
$$
我们重新定义$q$函数以允许任意基准年份，并定义一个新的$r$函数，然后绘制这两个函数。

我们继续假设$t_0=0$，并为不同的到期时间$t=T$绘图，其中$K_0$低于稳态值

```{code-cell} python3
@jit
def q_generic(pp, t0, c_path):
    # 简化符号
    β = pp.β
    u_prime = pp.u_prime

    T = len(c_path) - 1
    q_path = np.zeros(T+1-t0)
    q_path[0] = 1
    for t in range(t0+1, T+1):
        q_path[t-t0] = β ** (t-t0) * u_prime(c_path[t]) / u_prime(c_path[t0])
    return q_path

@jit
def r(pp, t0, q_path):
    '''到期收益率'''
    r_path = - np.log(q_path[1:]) / np.arange(1, len(q_path))
    return r_path

def plot_yield_curves(pp, t0, c0, k0, T_arr):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for T in T_arr:
        c_path, k_path = bisection(pp, c0, k0, T, verbose=False)
        q_path = q_generic(pp, t0, c_path)
        r_path = r(pp, t0, q_path)

        axs[0].plot(range(t0, T+1), q_path)
        axs[0].set(xlabel='t', ylabel='$q_t^0$', title='希克斯-阿罗价格')

        axs[1].plot(range(t0+1, T+1), r_path)
        axs[1].set(xlabel='t', ylabel='$r_t^0$', title='收益率')
```
```{code-cell} python3
T_arr = [150, 75, 50]
plot_yield_curves(pp, 0, 0.3, k_ss/3, T_arr)
```

现在我们绘制 $t_0=20$ 时的图像

```{code-cell} python3
plot_yield_curves(pp, 20, 0.3, k_ss/3, T_arr)
```
