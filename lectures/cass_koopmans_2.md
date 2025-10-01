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

# Cass-Koopmans竞争均衡

```{contents} 目录
:depth: 2
```

## 概述

本讲座继续我们在{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中关于Tjalling Koopmans {cite}`Koopmans`和David Cass {cite}`Cass`用来研究最优资本积累的模型的分析。

本讲座说明了**计划经济**和以**竞争均衡**或**市场经济**形式组织的经济之间实际上存在的更普遍联系。

在{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中，我们研究了规划问题并使用了以下思想：

- 规划问题的拉格朗日公式，它导出了一个差分方程组。
- 用于求解受初始和终端条件约束的差分方程的**射击算法**。
- 描述长期但有限期经济最优路径的**收费公路**性质。

本讲座使用了额外的思想，包括：

- 以John R. Hicks和Kenneth Arrow命名的Hicks-Arrow价格。
- 规划问题中的一些拉格朗日乘数与Hicks-Arrow价格之间的联系。
- 在宏观经济学动态中广泛使用的**大** $K$ **，小** $k$ **技巧**。
    * 我们将在[第70讲](https://python.quantecon.org/rational_expectations.html)和[第41讲](https://python-advanced.quantecon.org/dyn_stack.html)中遇到这个技巧。
- 利率期限结构的非随机版本。
- 两种组织经济方式之间的密切联系，即：
    * **社会主义**：中央计划者指挥资源分配，和
    * **竞争市场**：竞争均衡**价格**诱导个人消费者和生产者选择社会最优分配，作为他们自私决策的无意后果

让我们从一些标准导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图片尺寸
from numba import jit, float64
from numba.experimental import jitclass
import numpy as np
```

## Cass-Koopmans模型回顾

物理环境与{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`完全相同。

时间是离散的，取值为 $t = 0, 1 , \ldots, T$。

单一商品可以被消费或投资于实物资本。

资本品是耐用的，但每期以恒定比率折旧。

我们用 $C_t$ 表示第 $t$ 期的非耐用消费品。

用 $K_t$ 表示第 $t$ 期的实物资本存量。

令 $\vec{C}$ = $\{C_0,\dots, C_T\}$ 且
$\vec{K}$ = $\{K_0,\dots,K_{T+1}\}$。

代表性家庭在每个时期 $t$ 都拥有一单位的劳动力，并且喜欢在每个时期消费商品。

代表性家庭在每个时期 $t$ 非弹性地供应一单位劳动力 $N_t$，因此，对所有 $ t \in \{0, 1, \ldots, T\}$，$N_t =1$。

代表性家庭对消费组合的偏好由以下效用函数给出：

$$
U(\vec{C}) = \sum_{t=0}^{T} \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}
$$

其中$\beta \in (0,1)$是贴现因子，$\gamma >0$
决定单期效用函数的曲率。

我们假设$K_0 > 0$。

存在一个全经济范围的生产函数

$$
F(K_t,N_t) = A K_t^{\alpha}N_t^{1-\alpha}
$$

其中 $0 < \alpha<1$，$A > 0$。

一个可行的配置 $\vec{C}, \vec{K}$ 满足

$$
C_t + K_{t+1} \leq F(K_t,N_t) + (1-\delta) K_t \quad \text{对所有 } t \in \{0, 1, \ldots, T\}
$$

其中 $\delta \in (0,1)$ 是资本的折旧率。

### 规划问题

在{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中，我们研究了规划者选择配置 $\{\vec{C},\vec{K}\}$ 以
最大化 {eq}`utility-functional`，同时受约束于 {eq}`allocation`。

规划问题求解的配置将在竞争均衡中重现，我们将在下面看到。

## 竞争均衡

我们现在研究经济的去中心化版本。

它与{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中研究的计划经济具有相同的技术和偏好结构。

但现在没有计划者。

有（单位质量的）价格接受型消费者和企业。

市场价格被设定为协调代表性消费者和代表性企业各自独立做出的不同决策。

有一个代表性消费者，其对消费计划的偏好与计划经济中的消费者相同。

消费者（也称为*家庭*）不再由计划者告诉消费和储蓄什么，而是自己选择，但受预算约束。

- 在每个时期 $t$，消费者从企业获得工资和资本租金
  -- 这些构成其在第 $t$ 期的**收入**。
- 消费者决定将多少收入用于消费或储蓄。
- 家庭可以通过获得额外的实物资本（它与第 $t$ 期的消费一比一交换）
  或通过在非 $t$ 时期获得消费权来储蓄。
- 家庭拥有实物资本和劳动力，并将它们出租给企业。
- 家庭消费、供应劳动力并投资于实物资本。
- 一个利润最大化的代表性企业运营生产技术。
- 企业每个时期从代表性家庭租用劳动力和资本，每个时期将其产出出售给家庭。
- 代表性家庭和代表性企业都是
  **价格接受者**，他们认为价格不会受到他们选择的影响

```{note}
同样，我们可以认为有单位质量的相同代表性消费者和相同代表性企业。
```

## 市场结构

代表性家庭和代表性企业都是价格接受者。

家庭拥有两种生产要素，即劳动力和实物资本。

每个时期，企业从家庭租用这两种要素。

有一个**单一**的大型竞争市场，其中家庭用第 $0$ 期的商品交换所有其他期 $t=1, 2, \ldots, T$ 的商品。

### 价格

有一个价格序列
$\{w_t,\eta_t\}_{t=0}^T= \{\vec{w}, \vec{\eta} \}$，
其中

- $w_t$ 是第 $t$ 期的工资，即劳动力的租赁率

- $\eta_t$ 是第 $t$ 期的资本租赁率

此外，还有一个跨期价格向量 $\{q_t^0\}$，其中

- $q^0_t$ 是一单位第 $t$ 期商品在第 $0$ 期的价格。

我们称 $\{q^0_t\}_{t=0}^T$ 为**Hicks-Arrow价格**向量。
这是以1972年经济学诺贝尔奖获得者命名的。

因为 $q_t^0$ 是一个**相对价格**，$q_t^0$ 的计价单位是；我们可以通过将它们全部乘以一个正标量，比如 $\lambda > 0$，来重新标准化它们。

$q_t^0$ 的单位可以设置为

$$
\frac{\text{第 $0$ 期商品数量}}{\text{第 $t$ 期商品数量}}
$$

在这种情况下，我们将第 $t$ 期的消费品作为**计价单位**。

## 企业问题

在第 $t$ 期，代表性企业雇佣劳动力
$\tilde n_t$ 和资本 $\tilde k_t$。

企业在第 $t$ 期的利润为

$$
F(\tilde k_t, \tilde n_t)-w_t \tilde n_t -\eta_t \tilde k_t
$$

其中 $w_t$ 是第 $t$ 期的工资率，而 $\eta_t$ 是第 $t$ 期的资本租赁率。

与计划经济模型一样

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

这些条件来自无套利要求。

为了描述这个无套利利润推理，我们首先应用关于线性齐次函数的欧拉定理。

该定理适用于Cobb-Douglas生产函数，因为它表现出规模报酬不变：

$$
\alpha F(\tilde k_t, \tilde n_t) =  F(\alpha  \tilde k_t, \alpha \tilde n_t)
$$

对于 $\alpha \in (0,1)$。

对上述方程两边取
$\frac{\partial  }{\partial \alpha}$ 的偏导数得到

$$
F(\tilde k_t,\tilde n_t) =  \frac{\partial F}{\partial \tilde k_t}
\tilde k_t + \frac{\partial F}{\partial \tilde  n_t} \tilde n_t
$$

将企业的利润重写为

$$
\frac{\partial F}{\partial \tilde k_t} \tilde k_t +
\frac{\partial F}{\partial \tilde  n_t} \tilde n_t-w_t \tilde n_t -\eta_t k_t
$$

或

$$
\left(\frac{\partial F}{\partial \tilde k_t}-\eta_t\right) \tilde k_t +
\left(\frac{\partial F}{\partial \tilde  n_t}-w_t\right) \tilde n_t
$$

因为 $F$ 是1次齐次的，所以
$\frac{\partial F}{\partial \tilde k_t}$和
$\frac{\partial F}{\partial \tilde n_t}$是 0 次齐次的，因此相对于
$\tilde k_t$ 和 $\tilde n_t$ 是固定的。

如果$\frac{\partial F}{\partial \tilde k_t}> \eta_t$，那么企业在每个额外单位的 $\tilde k_t$ 上获得正利润，所以它想要使 $\tilde k_t$ 任意大。

但设置 $\tilde k_t = + \infty$ 在物理上是不可行的，所以**均衡**价格必须取使企业没有这种套利机会的值。

类似的论证适用于
$\frac{\partial F}{\partial \tilde n_t}> w_t$ 的情况。

如果 $\frac{\partial \tilde k_t}{\partial \tilde k_t}< \eta_t$，
企业会想要将 $\tilde k_t$ 设为零，这是不可行的。

方便起见，定义
$\vec{w} =\{w_0, \dots,w_T\}$ 和 $\vec{\eta}= \{\eta_0, \dots, \eta_T\}$。

## 家庭问题

代表性家庭生活在 $t=0,1,\dots, T$。

在第 $t$ 期时，家庭出租1单位劳动力和 $k_t$ 单位资本给企业并获得收入

$$
w_t 1+ \eta_t k_t
$$

在第 $t$ 期时，家庭将其收入分配于以下两个类别的购买：

* 消费 $c_t$

* 净投资 $k_{t+1} -(1-\delta)k_t$


这里 $\left(k_{t+1} -(1-\delta)k_t\right)$ 是家庭的实物资本净投资，$\delta \in (0,1)$ 是资本的折旧率。

在第 $t$ 期时，消费者可以自由地购买比其通过向企业提供资本和劳动所获得的收入更多的商品，用于消费和实物资本投资，只要在其他时期其收入超过其购买即可。

消费者在第 $t$ 期的消费品的净超额需求是

$$
e_t \equiv \left(c_t + (k_{t+1} -(1-\delta)k_t)\right)-(w_t 1 + \eta_t k_t)
$$

令$\vec{c} = \{c_0,\dots,c_T\}$ 且 $\vec{k} = \{k_1,\dots,k_{T+1}\}$。

对家庭来说，$k_0$ 是给定的。

家庭面临一个**单一**预算约束，要求家庭净超额需求的现值必须为零：

$$
\sum_{t=0}^T q^0_t e_t  \leq 0
$$

或

$$
\sum_{t=0}^T q^0_t  \left(c_t + (k_{t+1} -(1-\delta)k_t)\right) \leq \sum_{t=0}^T q^0_t(w_t 1 + \eta_t k_t)  \
$$

家庭，作为价格接受者，面临价格体系 $\{q^0_t, w_t, \eta_t\}$，并选择一个配置来解决受约束的优化问题：

$$
\begin{aligned}& \max_{\vec{c}, \vec{k} }  \sum_{t=0}^T \beta^t u(c_t) \\ \text{使得} \ \   & \sum_{t=0}^T q_t^0\left(c_t +\left(k_{t+1}-(1-\delta) k_t \right) - (w_t -\eta_t k_t) \right)\leq 0  \notag \end{aligned}
$$

**价格体系**有以下组成部分：

* $w_t$ 表示在第 $t$ 期，每雇佣一单位劳动所支付的工资，以第 $t$ 期商品为计价单位。

* $\eta_t$ 表示在第 $t$ 期，每雇佣一单位资本所支付的回报，以第 $t$ 期商品为计价单位。

* $q_t^0$ 表示在第 $t$ 期，每一单位商品的价格，以计价物为单位进行度量。


### 定义

- **价格体系**是一个序列
  $\{q_t^0,\eta_t,w_t\}_{t=0}^T= \{\vec{q}, \vec{\eta}, \vec{w}\}$。
- **配置**是一个序列
  $\{c_t,k_{t+1},n_t=1\}_{t=0}^T = \{\vec{c}, \vec{k}, \vec{n}\}$。
- **竞争均衡**是一个价格体系和一个配置，并具有以下性质：
    - 给定价格体系，该配置解决家庭的问题。
    - 给定价格体系，该配置解决企业的问题。


这里的设想是，均衡价格体系和配置一旦确定，就固定下来。

实际上，我们可以想象所有交易都在时刻 $0$ 之前一次性完成。

## 计算竞争均衡

我们使用**猜测和验证**方法来计算竞争均衡。

- 我们**猜测**均衡价格序列
  $\{\vec{q}, \vec{\eta}, \vec{w}\}$。
- 然后我们**验证**在这些价格下，家庭和企业选择相同的配置。

### 价格体系的猜测

在{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中，我们计算了一个配置 $\{\vec{C}, \vec{K}, \vec{N}\}$
，它解决了规划问题。


我们使用该配置来构造均衡价格体系的猜测。


```{note}
这个配置将构成**大** $K$，即我们将在[第70讲](https://python.quantecon.org/rational_expectations.html)和[第41讲](https://python-advanced.quantecon.org/dyn_stack.html)中应用于竞争均衡的**大** $K$ **，小** $k$ **技巧**中的大 $K$.
```

特别是，我们将使用以下程序：

* 获得代表性企业和代表性消费者的一阶条件。
* 从这些方程中，通过用解决规划问题的数量 $\vec C, \vec K$ 替换企业的选择变量 $\tilde k, \tilde n$ 和消费者的选择变量，获得一组新方程。
* 求解所得方程，得到 $\{\vec{q}, \vec{\eta}, \vec{w}\}$ 作为 $\vec C, \vec K$ 的函数。
* 验证在这些价格下，$c_t = C_t, k_t = \tilde k_t = K_t, \tilde n_t = 1$， 对于 $t = 0, 1, \ldots, T$。

因此，我们猜测，对于 $t=0,\dots,T$：

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

在这些价格下，让家庭选择的资本为

```{math}
:label: eq-pr4

k^*_t(\vec {q}, \vec{w}, \vec{\eta)} , \quad t \geq 0
```

让企业选择的配置为

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

我们将验证对于 $t=0,\dots,T$，家庭和企业选择的配置都等于解决规划问题的配置：

```{math}
:label: eq-pl

k^*_t = \tilde k^*_t=K_t, \tilde n_t=1, c^*_t=C_t
```

### 验证程序

我们的方法是首先研究家庭和企业优化问题的一阶必要条件。

在我们猜测的价格体系下，我们将验证这两组一阶条件在解决规划问题的配置下都得到满足。

### 家庭的拉格朗日函数

为了求解家庭的问题，我们构建拉格朗日函数

$$
\mathcal{L}(\vec{c},\vec{k},\lambda) = \sum_{t=0}^T \beta^t u(c_t)+ \lambda \left(\sum_{t=0}^T q_t^0\left(\left((1-\delta) k_t -w_t\right)
+\eta_t k_t -c_t  - k_{t+1}\right)\right)
$$

并提出极小极大问题：

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

k_{T+1}: \quad -\lambda q_0^{T+1} \leq 0, \ \leq 0 \text{ if } k_{T+1}=0; \ =0 \text{ if } k_{T+1}>0
```

现在我们将价格猜测代入并进行一些代数运算，希望从{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中的规划问题的一阶必要条件{eq}`constraint1`-{eq}`constraint4`中恢复所有条件。

将{eq}`cond1`和{eq}`eq-price`结合，我们得到：

$$
u'(C_t) = \mu_t
$$

这是{eq}`constraint1`。

将{eq}`cond2`、{eq}`eq-price`和{eq}`eq-price3`结合，我们得到：

```{math}
:label: co-re

-\lambda \beta^t \mu_t\left[(1-\delta) +f'(K_t)\right] +\lambda \beta^{t-1}\mu_{t-1}=0
```

通过将{eq}`co-re`两边除以 $\lambda$（因为 $u'>0$， 所以不为零），我们得到：

$$
\beta^t \mu_t [(1-\delta+f'(K_t)] = \beta^{t-1} \mu_{t-1}
$$

或

$$
\beta \mu_t [(1-\delta+f'(K_t)] = \mu_{t-1}
$$

这是{eq}`constraint2`。

将{eq}`cond3`、{eq}`eq-price`、{eq}`eq-price2`
和{eq}`eq-price3`结合，在将{eq}`cond3`两边乘以 $\lambda$ 后，我们得到

$$
\sum_{t=0}^T \beta^t \mu_{t} \left(C_t+ (K_{t+1} -(1-\delta)K_t)-f(K_t)+K_t f'(K_t)-f'(K_t)K_t\right) \leq 0
$$

简化为

$$
\sum_{t=0}^T  \beta^t \mu_{t} \left(C_t +K_{t+1} -(1-\delta)K_t - F(K_t,1)\right) \leq 0
$$

因为 $\beta^t \mu_t >0$ 对于 $t =0, \ldots, T$，所以

$$
C_t+K_{t+1}-(1-\delta)K_t -F(K_t,1)=0 \quad  \text{对所有 }t \in \{0, 1, \ldots, T\}
$$

这是{eq}`constraint3`。

将{eq}`cond4`和{eq}`eq-price`结合，我们得到：

$$
-\beta^{T+1} \mu_{T+1} \leq 0
$$

两边除以 $\beta^{T+1}$ 得到

$$
-\mu_{T+1} \leq 0
$$

这是规划问题的{eq}`constraint4`。

因此，在我们猜测的均衡价格体系下，解决规划问题的配置也解决了代表性家庭在竞争均衡中面临的问题。

### 代表性企业的问题

我们现在转向竞争均衡中企业面临的问题：

如果对于所有t，我们将{eq}`eq-pl`代入{eq}`Zero-profits`，我们得到

$$
\frac{\partial F(K_t, 1)}{\partial K_t} = f'(K_t) = \eta_t
$$

这是{eq}`eq-price3`。

如果对于所有t，我们现在将{eq}`eq-pl`代入{eq}`Zero-profits`，我们得到：

$$
\frac{\partial F(\tilde K_t, 1)}{\partial \tilde L_t} = f(K_t)-f'(K_t)K_t=w_t
$$

这正好是{eq}`eq-pr4`。

因此，在我们猜测的均衡价格体系下，解决规划问题的配置也解决了竞争均衡中企业面临的问题。

由{eq}`ge1`和{eq}`ge2`，这个配置与解决消费者问题的配置相同。

```{note}
因为预算集只受相对价格影响，$\{q^0_t\}$ 仅在乘以一个正的常数的意义下被确定。
```

**标准化：** 我们可以选择 $\{q_t^0\}$ 使 $\lambda=1$，这样 $q_t^0$ 是以第 $0$ 期商品的边际效用为计量单位来表示的。

我们将在下面绘制 $q, w, \eta$ 以显示这些均衡价格诱导出与我们在规划问题中看到的相同的总体运动。

接下来，我们引入{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`中用来解决规划问题的Python代码

首先，让我们定义一个`jitclass`来存储定义我们经济的参数和函数。

```{code-cell} python3
planning_data = [
    ('γ', float64),    # 相对风险厌恶系数
    ('β', float64),    # 贴现因子
    ('δ', float64),    # 资本折旧率
    ('α', float64),    # 人均资本回报率
    ('A', float64)     # 技术水平
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
    使用状态转移方程和欧拉方程计算T期内c和k的完整路径。
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

    # 设置c0的初始边界
    c0_upper = pp.f(k0)
    c0_lower = 0

    i = 0
    while True:
        c_vec, k_vec = shooting(pp, c0, k0, T)
        error = k_vec[-1] - k_ter

        # check if the terminal condition is satisfied
        if np.abs(error) < tol:
            if verbose:
                print('在第', i+1, '迭代步收敛成功')
            return c_vec, k_vec

        i += 1
        if i == max_iter:
            if verbose:
                print('收敛失败')
            return c_vec, k_vec

        # 如果迭代继续, 更新c0的猜测值和边界
        if error > 0:
            c0_lower = c0
        else:
            c0_upper = c0

        c0 = (c0_lower + c0_upper) / 2
```

```{code-cell} python3
pp = PlanningProblem()

# Steady states
ρ = 1 / pp.β - 1
k_ss = pp.f_prime_inv(ρ+pp.δ)
c_ss = pp.f(k_ss) - pp.δ * k_ss
```

来自{doc}`Cass-Koopmans规划模型 <cass_koopmans_1>`的上述代码让我们能够计算规划问题的最优配置。

* 从前面的分析中，我们知道它也将是与竞争均衡相关的配置。

现在我们准备引入计算竞争均衡中出现的额外对象所需的Python代码。

```{code-cell} python3
@jit
def q(pp, c_path):
    # 这里我们选择计价单位为u'(c_0) -- 这是q^(t_0)_t
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

现在我们对每个 $T$ 计算并绘制

```{code-cell} python3
T_arr = [250, 150, 75, 50]

fix, axs = plt.subplots(2, 3, figsize=(13, 6))
titles = ['Arrow-Hicks价格', '劳动力租赁率', '资本租赁率',
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

#### 改变曲率

现在让我们来看看，如果我们保持 $T$ 不变，但允许曲率参数 $\gamma$ 变化，从 $K_0$ 低于稳态开始，结果会如何变化。

我们绘制 $T=150$ 的结果

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
        if titles[i] == '资本':
            ax.axhline(k_ss, lw=1, ls='--', c='k')
        if titles[i] == '消费':
            ax.axhline(c_ss, lw=1, ls='--', c='k')

axs[0, 0].legend()
plt.tight_layout()
plt.show()
```

调整 $\gamma$ 意味着调整个人偏好平滑消费的程度。

较高的 $\gamma$ 意味着个体更偏好平滑消费，从而导致向稳态配置的收敛速度较慢。

较低的 $\gamma$ 意味着个体较少偏好平滑消费，从而导致向稳态配置的收敛速度较快。

## 收益率曲线和Hicks-Arrow价格

我们回到Hicks-Arrow价格，并计算它们如何与不同期限贷款的**收益率**相关。

这将让我们绘制一个**收益率曲线**，将期限 $j=1, 2, \ldots$ 的债券收益率与 $j=1,2, \ldots$ 对应。

我们使用以下公式。

在第 $t_0$ 期发放并在第 $t > t_0$ 期到期的贷款的**到期收益率**

$$
r_{t_0,t}= -\frac{\log q^{t_0}_t}{t - t_0}
$$

基准年 $t_0\leq t$ 的Hicks-Arrow价格体系满足

$$
q^{t_0}_t = \beta^{t-t_0} \frac{u'(c_t)}{u'(c_{t_0})}= \beta^{t-t_0}
\frac{c_t^{-\gamma}}{c_{t_0}^{-\gamma}}
$$

我们重新定义了 $q$ 函数，使其允许任意基准年，并定义了一个新的 $r$ 函数，然后将两者绘制出来。

我们继续假设 $t_0=0$，并在 $K_0$ 低于稳态的情况下，对不同到期期限 $t=T$ 进行绘图。

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
        axs[0].set(xlabel='t', ylabel='$q_t^0$', title='Hicks-Arrow价格')

        axs[1].plot(range(t0+1, T+1), r_path)
        axs[1].set(xlabel='t', ylabel='$r_t^0$', title='收益率')
```

```{code-cell} python3
T_arr = [150, 75, 50]
plot_yield_curves(pp, 0, 0.3, k_ss/3, T_arr)
```

现在我们绘制 $t_0=20$ 时的图

```{code-cell} python3
plot_yield_curves(pp, 20, 0.3, k_ss/3, T_arr)
```