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

(harrison_kreps)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 异质信念与泡沫

```{index} single: Models; Harrison Kreps
```

```{contents} 目录
:depth: 2
```

除了Anaconda中包含的库外，本讲座还使用以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

本讲解介绍了Harrison和Kreps模型{cite}`HarrKreps1978`的一个版本。

该模型确定了一种由两类自利投资者交易的产生股息的资产的价格。

该模型具有以下特点：

* 异质信念
* 不完全市场
* 卖空限制，以及可能的$\ldots$
* （杠杆）对投资者为购买风险资产而进行借贷的能力的限制

让我们从一些标准导入开始：

```{code-cell} ipython
import numpy as np
import quantecon as qe
import scipy.linalg as la
```

### 参考文献

在阅读以下内容之前，建议您回顾我们关于以下内容的讲座：

* {doc}`马尔可夫链 <finite_markov>`
* {doc}`有限状态空间的资产定价 <markov_asset>`

### 泡沫

经济学家对*泡沫*的定义有所不同。

Harrison-Kreps模型阐述了许多经济学家认同的泡沫概念：

> *当所有投资者都认为资产当前价格超过他们认为资产基本股息流所能支撑的水平时，资产价格的某个组成部分可以被解释为泡沫*。

## 模型结构

该模型通过忽略具有固定不同信念的投资者之间的财富分配变化来简化问题，这些投资者对决定资产收益的基本面持有不同看法。

有固定数量 $A$ 股的资产。

每股赋予其所有者一个股息流 $\{d_t\}$，该股息流由定义在状态空间 $S \in \{0, 1\}$ 上的马尔可夫链控制。

股息遵循

$$
d_t =

\begin{cases}
    0 & \text{ if } s_t = 0 \\
    1 & \text{ if } s_t = 1
\end{cases}
$$

在时间$t$结束和时间$t+1$开始时持有股份的所有者有权获得在时间$t+1$支付的股息。

因此，该股票是**除息**交易。

在时间$t+1$开始时持有股份的所有者也有权在时间$t+1$期间将股份卖给其他投资者。

两种类型$h=a, b$的投资者仅在其对马尔可夫转移矩阵$P$的信念上有所不同，其典型元素为

$$
P(i,j) = \mathbb P\{s_{t+1} = j \mid s_t = i\}
$$

$a$类型投资者认为转移矩阵为

$$
P_a =
    \begin{bmatrix}
        \frac{1}{2} & \frac{1}{2} \\
        \frac{2}{3} & \frac{1}{3}
    \end{bmatrix}
$$

$b$类型投资者认为转移矩阵为

$$
P_b =
    \begin{bmatrix}
        \frac{2}{3} & \frac{1}{3} \\
        \frac{1}{4} & \frac{3}{4}
    \end{bmatrix}
$$

因此，在状态$0$时，$a$类投资者对下一期的股息比$b$类投资者更乐观。

但在状态$1$时，$a$类投资者对下一期的股息比$b$类投资者更悲观。

这两个矩阵的平稳（即不变）分布可以按如下方式计算：

```{code-cell} ipython3
qa = np.array([[1/2, 1/2], [2/3, 1/3]])
qb = np.array([[2/3, 1/3], [1/4, 3/4]])
mca = qe.MarkovChain(qa)
mcb = qe.MarkovChain(qb)
mca.stationary_distributions
```

```{code-cell} ipython3
mcb.stationary_distributions
```

$P_a$ 的平稳分布约为 $\pi_a = \begin{bmatrix} .57 & .43 \end{bmatrix}$。

$P_b$ 的平稳分布约为 $\pi_b = \begin{bmatrix} .43 & .57 \end{bmatrix}$。

因此，a 类投资者平均来说更悲观。

### 所有权权利

在 t 时刻结束时资产的所有者有权获得 t+1 时刻的股息，并且有权在 t+1 时刻出售该资产。

两类投资者都是风险中性的，并且都有相同的固定贴现因子 $\beta \in (0,1)$。

在我们的数值示例中，我们将设定 $\beta = .75$，这与 Harrison 和 Kreps {cite}`HarrKreps1978` 的设定相同。

我们最终将研究关于股票数量 A 相对于两类投资者可投资资源的两个替代性假设的后果。

1. 两类投资者都有足够的资源（无论是财富还是借贷能力）来购买全部可用的资产股票[^f1]。

1. 没有任何一类投资者拥有足够的资源来购买全部股票。

案例1是Harrison和Kreps研究的案例。

在案例2中，两类投资者始终至少持有一些资产。

### 禁止卖空

不允许卖空交易。

这很重要，因为它限制了悲观者表达他们观点的方式。

* 他们**可以**通过出售自己的股份来表达观点。
* 他们**不能**通过人为"制造股份"来更强烈地表达观点——也就是说，他们不能从更乐观的投资者那里借入股份然后立即卖出。

### 乐观与悲观

上述感知转移矩阵$P_a$和$P_b$的规范，直接来自Harrison和Kreps的研究，内置了随机交替的暂时性乐观和悲观情绪。

请记住，状态$1$是高股息状态。

* 在状态$0$中，类型$a$的投资者比类型$b$的投资者对下一期的股息更乐观。

* 在状态$1$中，类型$b$的个体对下一期的股息比类型$a$的个体更乐观。

然而，平稳分布$\pi_a = \begin{bmatrix} .57 & .43 \end{bmatrix}$和$\pi_b = \begin{bmatrix} .43 & .57 \end{bmatrix}$告诉我们，从长期来看，类型$b$的人对股息过程比类型$a$的人更乐观。

### 信息

投资者知道一个价格函数，该函数将$t$时刻的状态$s_t$映射到在该状态下的均衡价格$p(s_t)$。

这个价格函数是内生的，将在下面确定。

当投资者在$t$时刻选择是购买还是出售资产时，他们也知道$s_t$。

## 求解模型

现在让我们开始求解模型。

我们将在特定的信念设定和交易限制条件下确定均衡价格，这些设定是从上述规范中选择的。

我们将比较在以下不同情况下的均衡价格函数

关于信念的假设：

1. 只有一种类型的个体，要么是 $a$ 要么是 $b$。
1. 有两种类型的个体，仅在其信念上有所不同。每种类型的个体都有足够的资源购买所有资产（Harrison和Kreps的设定）。
1. 有两种具有不同信念的个体，但由于财富和/或杠杆的限制，两种类型的投资者在每个时期都持有资产。

### 总结表

下表总结了本讲座其余部分获得的结果（在练习中，你将被要求重新创建该表并重新解释其中的部分内容）。

该表报告了Harrison和Kreps对$P_a, P_b, \beta$的规范所产生的影响。


|    $ s_t $    |   0   |   1   |
|---------------|-------|-------|
|    $ p_a $    | 1.33  | 1.22  |
|    $ p_b $    | 1.45  | 1.91  |
|    $ p_o $    | 1.85  | 2.08  |
|    $ p_p $    |   1   |   1   |
| $ \hat{p}_a $ | 1.85  | 1.69  |
| $ \hat{p}_b $ | 1.69  | 2.08  |

这里

* $p_a$ 是在同质信念 $P_a$ 下的均衡价格函数
* $p_b$ 是在同质信念 $P_b$ 下的均衡价格函数
* $p_o$ 是在异质信念下且边际投资者持乐观态度时的均衡价格函数
* $p_p$ 是在异质信念下且边际投资者持悲观态度时的均衡价格函数
* $\hat{p}_a$ 是 $a$ 类型投资者愿意为资产支付的金额
* $\hat{p}_b$ 是 $b$ 类型投资者愿意为资产支付的金额

我们将逐行解释这些值及其计算方法。

对应于 $p_o$ 的行适用于当两类投资者都有足够的资源购买全部资产，且存在严格的卖空限制，因此暂时乐观的投资者始终决定资产价格的情况。

如果两类投资者都没有足够的资源购买全部资产，且两类投资者都必须持有资产，则对应于 $p_p$ 的行将适用。

如果两类投资者都有足够的资源购买全部资产，但同时也允许卖空，使得暂时悲观的投资者为资产定价，则对应于 $p_p$ 的行也将适用。

### 单一信念价格

我们先来看看在同质信念下的资产定价。

(这种情况在{doc}`关于有限马尔可夫状态下资产定价的讲座 <markov_asset>`中已经讨论过)

假设只有一种类型的投资者，要么是类型 $a$ 要么是类型 $b$，并且这类投资者始终"为资产定价"。

设 $p_h = \begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}$ 为当所有投资者都是类型 $h$ 时的均衡价格向量。

今天的价格等于明天的股息和明天的资产价格的预期贴现值：

$$

p_h(s) = \beta \left( P_h(s,0) (0 + p_h(0)) + P_h(s,1) ( 1 + p_h(1)) \right), \quad s = 0, 1
$$ (eq:assetpricehomog)

这些方程意味着均衡价格向量为

```{math}
:label: HarrKrep1

\begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}
= \beta [I - \beta P_h]^{-1} P_h \begin{bmatrix} 0 \cr 1 \end{bmatrix}
```

表格的前两行报告了$p_a(s)$和$p_b(s)$的值。

这里有一个可以用来计算这些值的函数

```{code-cell} ipython3
def price_single_beliefs(transition, dividend_payoff, β=.75):
    """
    求解单一信念的函数
    """
    # 首先计算逆矩阵部分
    imbq_inv = la.inv(np.eye(transition.shape[0]) - β * transition)

    # 接下来计算价格
    prices = β * imbq_inv @ transition @ dividend_payoff

    return prices
```

#### 单一信念价格作为基准

这些在同质信念下的均衡价格是后续分析的重要基准。

* $p_h(s)$ 表示类型 $h$ 投资者认为的资产"基本价值"。
* 这里的"基本价值"指的是未来股息的预期贴现现值。

我们将把这些资产的基本价值与交易者持有不同信念时的均衡价值进行比较。

### 异质信念下的定价

需要考虑几种情况。

第一种是当两种类型的投资者都有足够的财富来独自购买所有资产时。

在这种情况下，为资产定价的边际投资者是更乐观的类型，因此均衡价格 $\bar p$ 满足Harrison和Kreps的关键方程：

```{math}
:label: hakr2

\bar p(s) =
\beta
\max
\left\{
        P_a(s,0) \bar p(0) + P_a(s,1) ( 1 +  \bar p(1))
        ,\;
        P_b(s,0) \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))
\right\}
```

对于$s=0,1$。

在上述等式中，右侧的$max$是针对下一期持有资产可能获得的两个支付值。

如果在状态$s$中定价资产的边际投资者是类型$a$，则满足：

$$
P_a(s,0)  \bar p(0) + P_a(s,1) ( 1 +  \bar p(1)) >
P_b(s,0)  \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))
$$

如果边际投资者是类型$b$，则满足：

$$
P_a(s,1)  \bar p(0) + P_a(s,1) ( 1 +  \bar  p(1)) <
P_b(s,1)  \bar p(0) + P_b(s,1) ( 1 +  \bar  p(1))
$$

**因此边际投资者是（暂时的）乐观型**。

方程{eq}`hakr2`是一个函数方程，类似于贝尔曼方程，可以通过以下方式求解：

* 从价格向量$\bar p$的一个猜测开始
* 对运算符进行迭代直至收敛，该运算符将猜测值$\bar p^j$映射到由{eq}`hakr2`右侧定义的更新猜测值$\bar p^{j+1}$，即

```{math}
:label: HarrKrep3

\bar  p^{j+1}(s)
 = \beta \max
 \left\{
        P_a(s,0) \bar p^j(0) + P_a(s,1) ( 1 + \bar p^j(1))

,\;
        P_b(s,0) \bar p^j(0) + P_b(s,1) ( 1 + \bar p^j(1))
\right\}
```

对于$s=0,1$。

表格中标记为$p_o$的第三行报告了当$\beta = .75$时解函数方程的均衡价格。

在这里，对$s_{t+1}$持乐观态度的类型在状态$s_t$中为资产定价。

将这些价格与在信念$P_a$和$P_b$下求解的同质信念经济的均衡价格进行比较是很有启发性的，这些价格分别在标记为$p_a$和$p_b$的行中报告。

在异质信念经济中的均衡价格$p_o$显然超过了任何潜在投资者在每个可能状态下认为的资产基本价值。

尽管如此，经济会反复进入一种状态，使每个投资者都愿意以超过他们认为的未来股息价值的价格购买资产。

投资者愿意支付超过他认为基本面股息流所应有的价格，因为他预期之后能够选择将资产卖给另一个会给出更高估值的投资者。

* $a$类型的投资者愿意为资产支付以下价格

$$
\hat p_a(s) =
\begin{cases}
\bar p(0)  & \text{ if } s_t = 0 \\
\beta(P_a(1,0) \bar p(0) + P_a(1,1) ( 1 +  \bar p(1))) & \text{ if } s_t = 1
\end{cases}
$$

* $b$类型的投资者愿意为资产支付以下价格

$$
\hat p_b(s) =
\begin{cases}
    \beta(P_b(0,0) \bar p(0) + P_b (0,1) ( 1 +  \bar p(1)))  & \text{ if } s_t = 0 \\
    \bar p(1)  & \text{ if } s_t =1
\end{cases}
$$

显然，$\hat p_a(1) < \bar p(1)$ 且 $\hat p_b(0) < \bar p(0)$。

$a$类型的投资者想在状态$1$时卖出资产，而$b$类型的投资者想在状态$0$时卖出资产。

* 当状态从$0$变为$1$或从$1$变为$0$时，资产就会易手。
* 估值$\hat p_a(s)$和$\hat p_b(s)$显示在表格的第四行和第五行。
* 即使是不买入资产的悲观投资者也认为资产价值高于他们认为的未来股息价值。

以下是使用上述迭代方法求解$\bar p$、$\hat p_a$和$\hat p_b$的代码

```{code-cell} ipython3
def price_optimistic_beliefs(transitions, dividend_payoff, β=.75,
                            max_iter=50000, tol=1e-16):
    """
    Function to Solve Optimistic Beliefs
    """
    # We will guess an initial price vector of [0, 0]
    p_new = np.array([[0], [0]])
    p_old = np.array([[10.], [10.]])

    # We know this is a contraction mapping, so we can iterate to conv
    for i in range(max_iter):
        p_old = p_new
        p_new = β * np.max([q @ p_old
                            + q @ dividend_payoff for q in transitions],
                            1)

        # If we succeed in converging, break out of for loop
        if np.max(np.sqrt((p_new - p_old)**2)) < tol:
            break

    ptwiddle = β * np.min([q @ p_old
                          + q @ dividend_payoff for q in transitions],
                          1)

    phat_a = np.array([p_new[0], ptwiddle[1]])
    phat_b = np.array([ptwiddle[0], p_new[1]])

    return p_new, phat_a, phat_b
```

### 资金不足

当乐观型投资者的财富不足——或无法借到足够资金——以持有全部资产时，结果会有所不同。

在这种情况下，资产价格必须调整以吸引悲观型投资者。

不同于方程{eq}`hakr2`，均衡价格满足

```{math}
:label: HarrKrep4

\check p(s)
= \beta \min
\left\{
    P_a(s,1)  \check  p(0) + P_a(s,1) ( 1 +   \check  p(1)) ,\;
    P_b(s,1)  \check p(0) + P_b(s,1) ( 1 + \check p(1))
\right\}
```

定价资产的边际投资者总是对资产估值较低的那一类。

现在边际投资者始终是（暂时）悲观的那一类。

从第六行可以看出，悲观价格$p_o$在两种状态下都低于同质信念价格$p_a$和$p_b$。

当悲观投资者按照{eq}`HarrKrep4`定价资产时，乐观投资者认为资产被低估了。

如果可能的话，乐观的投资者会愿意以一期无风险总利率$\beta^{-1}$借款来购买更多的资产。

杠杆方面的隐性约束禁止他们这样做。

当乐观的投资者按照方程{eq}`hakr2`给资产定价时，悲观的投资者认为资产被高估了，想要做空这个资产。

卖空限制阻止了这种行为。

以下是使用迭代法求解$\check p$的代码

```{code-cell} ipython3
def price_pessimistic_beliefs(transitions, dividend_payoff, β=.75,
                            max_iter=50000, tol=1e-16):
    """
    求解悲观信念的函数
    """
    # 我们将猜测一个初始价格向量[0, 0]
    p_new = np.array([[0], [0]])
    p_old = np.array([[10.], [10.]])

    # 我们知道这是一个压缩映射，所以我们可以迭代至收敛
    for i in range(max_iter):
        p_old = p_new
        p_new = β * np.min([q @ p_old
                            + q @ dividend_payoff for q in transitions],
                           1)

        # 如果成功收敛，跳出for循环
        if np.max(np.sqrt((p_new - p_old)**2)) < tol:
            break

    return p_new
```

### 进一步解释

Jose Scheinkman {cite}`Scheinkman2014` 将Harrison-Kreps模型解释为泡沫模型——即资产价格超过每个投资者基于其对资产基本股息流的信念所认为的合理价值的情况。

Scheinkman强调Harrison-Kreps模型的以下特点：

* 当Harrison-Kreps定价公式{eq}`hakr2`成立时，会出现高交易量。

* 每当状态从$s_t =0$转换到$s_t =1$时，A类投资者就会将全部资产卖给B类投资者。

* 每当状态从$s_t = 1$转换到$s_t =0$时，B类投资者就会将资产卖给A类投资者。

Scheinkman认为这是模型的优点，因为他观察到在*著名的泡沫*期间都存在高交易量。

* 如果资产的*供给*充分增加，无论是实物形式（建造更多"房屋"）还是人为形式（发明做空"房屋"的方法），当资产供给增长到超过乐观投资者购买资产的资源时，泡沫就会结束。
* 如果乐观投资者通过借贷来融资购买，收紧杠杆约束可以消除泡沫。

Scheinkman提取了关于金融监管对泡沫影响的见解。

他强调了限制做空和限制杠杆具有相反的效果。

## 练习

```{exercise-start}
:label: hk_ex1
```

本练习邀请你使用我们上面构建的函数重新创建汇总表。

|    $s_t$    |   0   |   1   |
|-------------|-------|-------|
|    $p_a$    | 1.33  | 1.22  |
|    $p_b$    | 1.45  | 1.91  |
|    $p_o$    | 1.85  | 2.08  |
|    $p_p$    |   1   |   1   |
| $\hat{p}_a$ | 1.85  | 1.69  |
| $\hat{p}_b$ | 1.69  | 2.08  |

首先你需要定义转移矩阵和股息支付向量。

此外，在下面我们将通过引入两种额外类型的投资者来解释对应于$p_o$的行，一种是**永久乐观型**，另一种是**永久悲观型**。

我们为永久乐观型和永久悲观型投资者构建主观转移概率矩阵如下。

永久乐观型投资者(即在每个状态下持最乐观信念的投资者)认为转移矩阵为

$$
P_o =
    \begin{bmatrix}
        \frac{1}{2} & \frac{1}{2} \\
        \frac{1}{4} & \frac{3}{4}
    \end{bmatrix}
$$

永久悲观型投资者认为转移矩阵为

$$
P_p =
    \begin{bmatrix}
        \frac{2}{3} & \frac{1}{3} \\
        \frac{2}{3} & \frac{1}{3}
    \end{bmatrix}
$$

我们将在下面展示练习1的解答时使用这些转移矩阵。

```{exercise-end}
```

```{solution-start} hk_ex1
:class: dropdown
```

首先，我们将获得具有同质信念的均衡价格向量，包括当所有投资者都持乐观或悲观态度时的情况。

```{code-cell} ipython3
qa = np.array([[1/2, 1/2], [2/3, 1/3]])    # a类型转移矩阵
qb = np.array([[2/3, 1/3], [1/4, 3/4]])    # b类型转移矩阵
# 乐观投资者转移矩阵
qopt = np.array([[1/2, 1/2], [1/4, 3/4]])
# 悲观投资者转移矩阵
qpess = np.array([[2/3, 1/3], [2/3, 1/3]])

dividendreturn = np.array([[0], [1]])

transitions = [qa, qb, qopt, qpess]
labels = ['p_a', 'p_b', 'p_optimistic', 'p_pessimistic']

for transition, label in zip(transitions, labels):
    print(label)
    print("=" * 20)
    s0, s1 = np.round(price_single_beliefs(transition, dividendreturn), 2)
    print(f"状态0: {s0}")
    print(f"状态1: {s1}")
    print("-" * 20)
```

我们将使用price_optimistic_beliefs函数来找出在异质信念下的价格。

```{code-cell} ipython3
opt_beliefs = price_optimistic_beliefs([qa, qb], dividendreturn)
labels = ['p_optimistic', 'p_hat_a', 'p_hat_b']

for p, label in zip(opt_beliefs, labels):
    print(label)
    print("=" * 20)
    s0, s1 = np.round(p, 2)
    print(f"State 0: {s0}")
    print(f"State 1: {s1}")
    print("-" * 20)
```

注意，在异质信念下的均衡价格等于在**永久乐观**投资者单一信念下的价格 - 这是因为在异质信念均衡中的边际投资者总是暂时乐观的那类投资者。

```{solution-end}
```

[^f1]: 通过假设两类个体总是有"足够深的口袋"来购买所有资产，该模型将财富动态排除在外。Harrison-Kreps模型在状态从0变为1或从1变为0时会产生大量交易量。

