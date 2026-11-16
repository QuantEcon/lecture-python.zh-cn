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
  title: 人工智能主体中作为交换媒介的货币
  headings:
    Overview: 概览
    The Kiyotaki-Wright environment: Kiyotaki-Wright 环境
    The Kiyotaki-Wright environment::Two equilibria: 两个均衡
    The Kiyotaki-Wright environment::The economies: 经济体
    Classifier systems for the Kiyotaki-Wright environment: '##岐泷-赖特环境下的分类器系统'
    Implementation: 实现
    'Economy A1.1: does a medium of exchange emerge?': 经济体 A1.1：交换媒介会出现吗？
    'Economy A2.1: when theory predicts speculation': 经济体 A2.1：当理论预测投机时
    The bucket brigade's discount factor: 桶队算法的贴现因子
    'Economy B.1: selecting between two equilibria': 经济体 B.1：在两个均衡之间进行选择
    The genetic algorithm: 遗传算法
    'Economy A1.2: learning from random rules': 经济体 A1.2：从随机规则中学习
    'Economy C: fiat money': 经济体 C：法定货币
    'Economy D: five goods, five types': 经济体 D：五种商品，五种类型
    Concluding remarks: 结束语
    Exercises: 练习
---

(marimon_mcgrattan_sargent)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 人工智能主体中作为交换媒介的货币

```{index} single: Bounded Rationality; Money as a Medium of Exchange
```

```{contents} Contents
:depth: 2
```

## 概览

Kiyotaki 和 Wright {cite}`KiyotakiWright1989` 研究了一个不存在需求双重巧合的经济。

只有当某种商品被接受不是因为它被需要，而是因为它可以在以后被转手时，交易才能发生。

扮演这种角色的商品就是**交换媒介**。

Kiyotaki 和 Wright 在假设主体完全理性——主体知道其交易伙伴之间商品的分布并对此做出最优反应——的前提下，刻画了这种经济的*平稳纳什均衡*。

Marimon、McGrattan 和 Sargent {cite}`MarimonMcGrattanSargent1990` 提出了一个不同的问题。

假设我们彻底抛弃理性假设，取而代之的是一群**人工智能主体**，他们最初遵循任意的、甚至是随机的经验法则，并且仅仅通过记录过去哪些法则奏效来调整这些法则。

这样的主体会*学会*使用交换媒介吗？

而当经济存在两个均衡，或者其唯一的均衡要求主体必须有耐心时，他们又会达成什么样的结果？

这些问题正是 {doc}`bounded_rationality` 在理性预期背景下提出的那个问题（{ref}`br_quest`）的实例：适应性主体能否找到一个均衡，又会是哪一个？

Marimon、McGrattan 和 Sargent 使用的学习装置是 John Holland 的**分类器系统** {cite}`Holland1975,HollandHolyoakNisbettThagard1986`。

{doc}`genetic_classifier` 以抽象的方式介绍了分类器系统和桶队算法；在这里，它们被组装进了各类主体之中。

每种类型的主体都拥有一个由“如果-那么”规则组成的种群、一个决定哪条规则起作用的拍卖机制，以及一个对导致良好结果的规则记入贷方、对导致不良结果的规则记入借方的记账系统。

当规则是随机生成时，**遗传算法**会繁育新规则并淘汰旧规则。

本讲重建了他们的计算实验，并针对每个实验使用多个随机种子进行运行，从而我们能够说明每种结果出现的频率。

主要发现是：

1. 在规则列表完备的情况下，经济 A1 中的主体在每次运行中都学会了其唯一的平稳均衡：商品 1——储藏成本最低的商品——充当货币流通。从随机规则出发，它们能达到与论文中大致相同的结果。
1. 在经济 A2 中，有耐心的主体本应通过接受一种昂贵商品来进行*投机*，但人工主体从不这样做。
   大多数运行最终稳定在基本模式的贸易格局上，而这并非该经济的均衡；有些运行要经过数千期才摆脱交易过少的陷入模式，还有少数运行则永远被困在其中。
   我们揭示了其中原因：桶队算法使主体的行为表现得如同以约 $0.016$ 的贴现因子来对未来进行贴现。
1. 在经济 B 中，该经济同时存在基本均衡与投机均衡，主体在大多数运行中选择基本均衡，而没有一次选择投机均衡。
1. 在几乎所有的运行中，一种本质上毫无价值、储藏无成本的物品——**法定货币**——都会被必须自己发现其用处的主体在交易中接受。
1. 在一个有五种商品和五种类型的经济中，作者对此经济的均衡并没有解析刻画，该算法提出了一种贸易格局，我们会对其附带的限定条件加以说明。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import NamedTuple
from numba import njit
```

## Kiyotaki-Wright 环境

存在三种类型的主体，用 $i = 1, 2, 3$ 索引，以及三种商品，用 $k = 1, 2, 3$ 索引。

类型 $i$ 的主体只从消费商品 $i$ 中获得效用。

他拥有生产商品 $i^*$ 的技术，其中 $i^* \neq i$。

在 Kiyotaki 和 Wright 的*模型 A* 中，生产模式是“维克塞尔三角”

| 类型 $i$ | 生产 $i^*$ | 消费 |
|---|---|---|
| 1 | 2 | 1 |
| 2 | 3 | 2 |
| 3 | 1 | 3 |

因此不存在需求的双重巧合：拥有你想要的东西的主体从不想要你拥有的东西。

所有商品都是不可分的，每个主体从一期到下一期恰好能储藏一单位恰好一种商品。

储藏商品 $k$ 一期的成本为 $s_k$，且

$$
s_3 > s_2 > s_1 > 0 .
$$

每种类型有相等数量 $A_i$ 的主体，因此总数为 $A = 3 A_i$。

每期每个主体都会被随机配对到恰好一个其他主体，不考虑类型。

记 $x_{at}$ 为主体 $a$ 在 $t$ 期携带的商品，$\rho_t(a)$ 为与 $a$ 配对的主体。

主体 $a$ 的**交易前状态**是一对

$$
z_{at} = \bigl(x_{at},\; x_{\rho_t(a)t}\bigr) .
$$

每期每个主体依次做出两个决策。

**第一**，在看到 $z_{at}$ 之后，他决定是否提出交易，

$$
\lambda_{at} = \begin{cases}
1 & \text{提出用 } x_{at} \text{ 换取 } x_{\rho_t(a)t} \\
0 & \text{拒绝}
\end{cases}
$$

当且仅当 $\lambda_{at} \lambda_{\rho_t(a)t} = 1$ 时交易才会发生，因此交易后的持有量为

```{math}
:label: mms_posttrade

x^+_{at} = (1 - \lambda_{at}\lambda_{\rho_t(a)t}) x_{at}
          + \lambda_{at}\lambda_{\rho_t(a)t} x_{\rho_t(a)t} .
```

**第二**，他决定是否消费他手中留下的东西，

$$
\gamma_{at} = \begin{cases}
1 & \text{消费 } x^+_{at} \\
0 & \text{将 } x^+_{at} \text{ 带入 } t+1
\end{cases}
$$

如果他消费，他会立即生产商品 $f(a) = i^*$，并将其带入 $t+1$。

因此

```{math}
:label: mms_lom

x_{a,t+1} = \gamma_{at} f(a) + (1 - \gamma_{at}) x^+_{at} .
```

单期净收益为

```{math}
:label: mms_payoff

U_a(\gamma_{at}) =
\gamma_{at}\bigl[u_i(x^+_{at}) - s(f(a))\bigr]
- (1 - \gamma_{at})\, s(x^+_{at}) ,
```

其中当 $k = i$ 时 $u_i(k) = u_i > 0$，否则 $u_i(k) = 0$。

请注意，主体*可以*消费他不想要的商品；他只是从中获得零效用，同时仍然要生产并支付储藏 $f(a)$ 的费用。

这种“可消费任意商品”的特性有一个值得一提的后果。

一个主体用自己生产的商品交换另一种商品并随即消费掉它，这一期结束时他所处的境地与拒绝交易完全相同：持有自己的生产商品并支付其储藏成本。

这样的交易在收益上是中性的，因此下文描述的记账系统中没有任何机制会将其淘汰，它们会在我们的若干经济体中出现。

```{note}
Kiyotaki 和 Wright 按照预期贴现效用对收益流进行排序。

Marimon、McGrattan 和 Sargent 则假设每个主体关心他的**长期平均**效用，并正是因此将分类器系统内部的记账系统构建为滚动平均值。

我们将在 {ref}`mms_impatience` 中看到，这种记账系统所产生的主体，其行为表现得仿佛对未来有着非常严重的贴现。
```

### 两个均衡

由于主体的收益取决于其他主体的行为，该模型可以存在多个平稳均衡。

Kiyotaki 和 Wright 用一组概率来刻画均衡，其中对我们最有用的是

$$
\pi^h_{it}(k) = \text{类型 } i \text{ 的主体在 } t \text{ 时持有商品 } k \text{ 的概率} .
$$

在**基本均衡**中，商品 1——储藏成本最低的商品——成为普遍的交换媒介：

* 类型 1 主体总是持有商品 2（他们自己生产的），并用其交换商品 1；
* 类型 3 主体总是持有商品 1，并用其交换商品 3；
* 类型 2 主体一半时间持有商品 1，一半时间持有商品 3。

所以均衡持有概率为

|  | $k=1$ | $k=2$ | $k=3$ |
|---|---|---|---|
| $i=1$ | 0 | 1 | 0 |
| $i=2$ | 0.5 | 0 | 0.5 |
| $i=3$ | 1 | 0 | 0 |

类型 2 主体接受商品 1，即使他们从不消费它：他们把它当作货币使用。

在**投机均衡**中，类型 1 主体还额外接受商品 3——储藏成本*最高*的商品——因为他们预期能很快将其换成商品 1。

原因在于：持有商品 2 的类型 1 主体只能从类型 2 主体那里获得商品 1，而持有商品 3 的类型 1 主体则可以从类型 3 主体那里获得商品 1，因为类型 3 主体想要商品 3 且持有商品 1。

每种配对类型被遇到的概率都是 $1/3$，因此持有商品 3 而非商品 2，会使每期获得商品 1 的概率提高 $\bigl(\pi^h_3(1) - \pi^h_2(1)\bigr)/3$，而代价是额外的储藏成本 $s_3 - s_2$。

Kiyotaki 和 Wright 证明，在贴现率趋于零的极限下，如果

```{math}
:label: mms_fundcond

s_3 - s_2 > \bigl(\pi^h_3(1) - \pi^h_2(1)\bigr)\tfrac{1}{3} u_1 ,
```

在基本均衡持有量下成立，则基本均衡是唯一的平稳均衡；而在投机均衡持有量下不等式反向成立时，投机均衡是唯一的均衡。

论文的脚注 7 中写的是 $\pi^h_1(3) - \pi^h_1(2)$，这是把下标弄反了：若使用类型 1 自身的持有量，右边将变为 $-u_1/3$，不等式就永远不可能不成立。

在基本均衡中 $\pi^h_3(1) = 1$ 且 $\pi^h_2(1) = 1/2$，而在投机均衡（下表列出）中 $\pi^h_2(1) = 0.586$。

下面的函数求解 {eq}`mms_fundcond` 以得到 $u_1$ 的临界值；其中的参数 $\beta$ 将在 {ref}`mms_impatience` 中用到。

```{code-cell} ipython3
s_A = np.array([0.1, 1.0, 20.0])           # 经济体 A 中的储藏成本


def speculation_threshold(s, π31, π21, β=1.0):
    "高于此值的 u_1，使得贴现因子为 β 的类型 1 主体更偏好商品 3。"
    return 3 * (s[2] - s[1]) / (β * (π31 - π21))


u_fund = speculation_threshold(s_A, π31=1.0, π21=0.5)     # 在基本均衡持有量下
u_spec = speculation_threshold(s_A, π31=1.0, π21=0.586)   # 在投机均衡持有量下
print(f"fundamental equilibrium requires u_1 < {u_fund:.1f}")
print(f"speculative equilibrium requires u_1 > {u_spec:.1f}")
```

经济体 A1 的 $u_1 = 100$，满足基本均衡条件，不过临界值仅比它高出14%。

经济体 A2 的 $u_1 = 500$，违反了该条件，因此对于有耐心的主体而言，A2 的唯一平稳均衡是投机均衡。

因此，对于有耐心的主体而言，这些经济体中的每一个都只有一个平稳均衡，它们提出的问题是该均衡是否可学习。

下面的经济体 B 有两个均衡，提出的则是选择问题。

### 经济体

下表仿照论文中的表4，列出了我们模拟的经济体。

所有经济体都有每种类型各五十个主体，且每种类型的效用相等，均为 $u_i = u$。

| 经济体 | 类型 $i$ 生产 | 储藏成本 | $u$ | 出价 $b_{11}, b_{12}, b_{21}, b_{22}$ | 初始规则 | 对有耐心主体的均衡 |
|---|---|---|---|---|---|---|
| A1.1 | 2, 3, 1 | 0.1, 1, 20 | 100 | 0.025, 0.025, 0.25, 0.25 | 全部 72 + 12 | 基本 |
| A2.1 | 2, 3, 1 | 0.1, 1, 20 | 500 | 0.025, 0.025, 0.25, 0.25 | 全部 72 + 12 | 投机 |
| B.1 | 3, 1, 2 | 1, 4, 9 | 100 | 0.25, 0.25, 0.25, 0.25 | 全部 72 + 12 | 基本和投机 |
| A1.2 | 2, 3, 1 | 0.1, 1, 20 | 100 | 0.025, 0.025, 0.25, 0.25 | 72 + 12 随机 | 基本 |
| A2.2 | 2, 3, 1 | 0.1, 1, 20 | 500 | 0.025, 0.025, 0.25, 0.25 | 72 + 12 随机 | 投机 |
| B.2 | 3, 1, 2 | 1, 4, 9 | 100 | 0.025, 0.025, 0.25, 0.25 | 72 + 12 随机 | 基本和投机 |
| C | 2, 3, 1 | 9, 14, 29；法定货币 0 | 100 | 0.025, 0.025, 0.025, 0.25 | 150 + 20 随机 | 基本 |
| D | 3, 4, 5, 1, 2 | 1, 4, 9, 20, 30 | 200 | 0.025, 0.025, 0.25, 0.25 | 180 + 20 随机 | 未刻画 |

出价将在下文中解释；“全部 72 + 12”表示对交易规则和消费规则的完整枚举。

A2.2 和 B.2 出现在练习中。

论文的表4将经济体 C 的储藏成本列为0.1、20和70，但报告模拟结果的表15中列出的是9、14和29，我们采用的是后者。

##岐泷-赖特环境下的分类器系统

{doc}`genetic_classifier` 在 {ref}`gc_classifier` 中给出了通用机制：字母表 $\{0, 1, \#\}$、匹配、特异性，以及带有强度的规则之间的拍卖。

其 {ref}`gc_bucket_brigade` 展示了设置后续奖励的规则是如何为此获得报酬的。

这里我们只记录马里蒙、麦格拉坦和萨金特所特有的内容：商品如何编码、每个主体每期所做的两个决策，以及将它们联系起来的两个强度方程。

### 编码

每种类型都有两个分类器系统：一个用于交换决策，一个用于消费决策。

论文的表 1 用三个位置对商品进行编码：一个表示"商品 $k$"的独热码，以及一个在位置 $k$ 处为 $0$ 表示"非商品 $k$"的编码。

```{code-cell} ipython3
WILD = -1                                   # the "don't care" symbol #


def table1(n_goods):
    "The paper's Table 1: one-hot codes for 'good k', then codes for 'not good k'."
    good = np.eye(n_goods, dtype=np.int8)
    not_good = np.full((n_goods, n_goods), WILD, dtype=np.int8)
    np.fill_diagonal(not_good, 0)
    return np.vstack([good, not_good])


def code_string(c):
    return ''.join('#' if b == WILD else str(int(b)) for b in c)


def rule_string(cond, action, n_goods):
    "Print a classifier the way the paper does, for example '010 100 -> 1'."
    s = code_string(cond)
    if len(cond) == 2 * n_goods:
        s = s[:n_goods] + ' ' + s[n_goods:]
    return f"{s} -> {int(action)}"


pd.DataFrame({'code': [code_string(c) for c in table1(3)],
              'meaning': [f'good {k}' for k in (1, 2, 3)]
                         + [f'not good {k}' for k in (1, 2, 3)]})
```

一个**交换分类器**是一个长度为 7 的字符串：三个位置表示主体自己持有的商品，三个表示交易对手持有的商品，还有一个二元动作（$1$ = 提议交易，$0$ = 拒绝）。

论文给出的两个例子是

```
100 001 -> 1     "if I hold good 1 and my partner holds good 3, propose to trade"
100 ##0 -> 0     "if I hold good 1 and my partner does not hold good 3, refuse"
```

一个**消费分类器**是一个长度为 4 的字符串：三个位置表示交易后持有的 $x^+_{at}$，一个动作位（$1$ = 消费）。

*完全枚举*是将表 1 中的每一个条件与两种动作分别配对。

这给出了 $6 \times 6 \times 2 = 72$ 个交换分类器和 $6 \times 2 = 12$ 个消费分类器，足以表达所有马尔可夫策略。

表 1 中没有表示"任意商品"的条件，因此枚举中不包含诸如"吃你所持有的任何东西"这样的规则。

当规则被随机抽取时，每个位置以相等概率取 $0$、$1$ 或 $\#$。

这样就有 $3^6 \times 2 = 1458$ 种可能的交换字符串和 $3^3 \times 2 = 54$ 种消费字符串，其中许多根本不匹配任何状态：例如 `110` 就不是任何商品的编码。

对于四种物品（经济体 C）或五种商品（经济体 D），编码是长度为四或五的独热字符串。

对于经济体 C，这给出了论文中 $3^8 \times 2 = 13{,}122$ 种可能交换字符串的数目。

论文没有说明它是如何对经济体 D 的商品进行编码的，因此五位编码是我们自己的选择。

规则的**特异性**是 $\sigma = 1/(1 + \#\text{的数量})$。

一个关于两种特定商品的交换规则有 $\sigma = 1$，像 `100 ##0` 这样的规则有 $\sigma = 1/3$，而一个关于两个"非"条件的规则有 $\sigma = 1/5$。

### 拍卖

在日期 $t$，分类器 $e$ 附有一个**强度** $S^a_e(t)$。

给定状态 $z_{at}$，令 $M_e(z_{at})$ 为条件被 $z_{at}$ 满足的交换分类器集合。

参与行动的分类器是匹配到的分类器中强度最高的那个，

```{math}
:label: mms_auction

e_t(z_{at}) = \arg\max\,\{S^a_e(t) : e \in M_e(z_{at})\} ,
```

消费分类器也以同样的方式从匹配 $x^+_{at}$ 的分类器中选出。

这是论文的公式（6）；我们以随机方式打破平局，这是论文未明确说明的一个细节。

每个被匹配的分类器也有一个**出价**，是其强度的一个比例，

$$
b_1(e) = b_{11} + b_{12}\sigma_e
\qquad\text{and}\qquad
b_2(c) = b_{21} + b_{22}\sigma_c
$$

分别对应交换分类器和消费分类器。

在 {eq}`mms_auction` 下，出价在决定获胜者方面不起作用：它只决定下文所述的支付。

论文的文字描述将获胜者称为"出价最高者"，并说随特异性上升的出价"有利于特定规则而非更一般的规则"。

如果是在强度为正的规则之间按 $b \cdot S$ 进行拍卖，这一点是成立的，但在 {eq}`mms_auction` 下，较大的出价比例会将规则在不动点处的强度拉向*零*，如下面的说明所示。

因此，在下游支付为正的规则中，较高的出价会降低平稳强度，并有利于一般规则；只有在下游支付为负的规则中，它们才会有利于特定规则，正如论文所说。

argmax 拍卖是 {doc}`genetic_classifier` 的双臂老虎机练习（{ref}`gc_ex3`）中 softmax 拍卖的贪婪极限，在该练习中，被匹配的规则 $r$ 以正比于 $e^{\eta S_r}$ 的概率获胜，而当 $\eta \to \infty$ 时就得到 argmax；该讲座将强度参数记为 $\beta$，这里我们将该字母保留给贴现因子使用。

论文的脚注 5 提到了这样一种随机拍卖作为替代方案。

贪婪拍卖避免了概率匹配，但它将所有的试探都留给初始强度和产生新规则的算子，这一选择在经济体 A2 中将会很重要。

### 分支结算法

强度通过霍兰德所称的*分支结算法（bucket brigade）*这一内部支付系统进行更新。

只有赢得拍卖的分类器会改变其强度，因此每个分类器都携带一个计数器 $\tau^a_e(t)$，记录它赢得拍卖的次数，初始化为 1。

支付流程如下。

* 外部收益 $U_a(\gamma_t)$ 流向 $t$ 时获胜的**消费**分类器。
* $t$ 时获胜的消费分类器将其出价支付给 $t$ 时获胜的**交换**分类器，正是后者创造了使其有机会行动的状态。
* $t$ 时获胜的交换分类器将其出价支付给 $t-1$ 时获胜的**消费**分类器，正是后者设定了状态 $z_{at}$。

由此得到的运动规律即论文的公式（12）和（13），

```{math}
:label: mms_strengthc

S^a_{c,\tau_c(t)} = S^a_{c,\tau_c(t)-1}
 - \frac{1}{\tau_c(t)-1}\Bigl[(1 + b_2(c))S^a_{c,\tau_c(t)-1}
   - \sum_e I^a_e(t) b_1(e) S^a_{e,\tau_e(t)} - U_a(\gamma_{ct})\Bigr]
```

```{math}
:label: mms_strengthe

S^a_{e,\tau_e(t)+1} = S^a_{e,\tau_e(t)}
 - \frac{1}{\tau_e(t)}\Bigl[(1 + b_1(e))S^a_{e,\tau_e(t)}
   - \sum_c I^a_c(t) b_2(c) S^a_{c,\tau_c(t)}\Bigr]
```

其中 $I^a_e(t)$ 和 $I^a_c(t)$ 是 $t$ 时赢得拍卖的指示变量。

因此在日期 $t$ 被更新的消费分类器正是在 $t-1$ 时获胜的分类器，每笔出价都沿着链条向后传递一步

$$
\cdots \;\to\; c_{t-1} \;\to\; e_t \;\to\; c_t \;\to\; e_{t+1} \;\to\; \cdots
$$

一个提议了交易但*未*得到回应的交换分类器不会被收费，其计数器不会前进，也不会向后支付任何款项：主体不会从一个被拒绝的提议中学到任何东西。

正如论文所述，同一类型的所有主体共享一个分类器系统。

论文是以"合成时间"（而非周期）来运行 {eq}`mms_strengthc`-{eq}`mms_strengthe` 的，即按匹配次数计数，因此按字面理解，消费分类器是由下一个行动的、同一*类型*的交换分类器来支付的，而这通常属于不同的主体。

我们则改为将每个主体自身连续的决策关联起来，这正是方程针对单个主体所描述的内容，也是一种决策因其自身后果而获得回报的理解方式。

```{note}
方程 {eq}`mms_strengthc`-{eq}`mms_strengthe` 使强度成为过去净收入的滚动**平均值**，而非累计总和，这与霍兰德最初的设定不同；论文脚注 8 说明选择平均化是为了使强度能够收敛。

由于增益为 $1/\tau$，这些是随机逼近递归，与驱动 {doc}`olg_adaptive_money` 和 {doc}`learning_approximation`（{ref}`br_one_recursion`）中学习者的递减增益递归相同。

因此任何极限点都必须满足论文的公式（17），

$$
\mathbb{E}\Bigl[(1 + b_2(c))S_c - \sum_e I_e b_1(e) S_e - U(\gamma_c)\Bigr] = 0,
\qquad
\mathbb{E}\Bigl[(1 + b_1(e))S_e - \sum_c I_c b_2(c) S_c\Bigr] = 0 .
$$

马里蒙、麦格拉坦和萨金特将满足这些方程的强度称为*平稳的*，并说当在平稳强度下赢得拍卖的规则恰好是实现均衡行为的规则时，一个平稳纳什均衡便得到了*支持*。

第二个方程给出 $S_e = \mathbb{E}[b_2(c) S_c]/(1 + b_1(e))$，因此较大的出价比例 $b_1(e)$ 会使规则的平稳强度趋近于零：当 $\mathbb{E}[b_2(c) S_c] > 0$ 时它降低 $S_e$，当 $\mathbb{E}[b_2(c) S_c] < 0$ 时它提高 $S_e$。
```

该系统还有一个特征，是后文大部分内容的驱动因素。

所有强度都从零开始，而主体所经历的几乎每一次净收益都是负的，因为储存是有成本的。

因此一个未被尝试过的规则会胜过任何迄今为止净成本经历为负的规则，并且在 argmax 拍卖下，主体在早期会扫过其所有未尝试的规则。

这种*乐观初始化*是主体在从完全枚举开始时进行试探的主要方式。

它有一种失败模式。

一个从未行动过的规则，或者其所有提议都被拒绝从而没有学到任何东西的规则，其强度会一直保持为零，而只要对手的平均值维持在略高于零的水平，它就会在每一次拍卖中都输给这个对手。

## 实现

状态空间很小，因此对于每条规则，我们预先计算它所匹配的状态集合，并将其存储为一个整数的各个比特位：交换规则的状态 $(x, y)$ 对应第 $x G + y$ 位，其中 $G$ 是商品数量。

模拟循环在 Numba 的 `njit` 下运行。

分类器系统存储为按类型和规则索引的并行数组。

```{code-cell} ipython3
class Rules(NamedTuple):
    "Classifier systems for all types, stored as arrays indexed [type, rule]."
    cond: np.ndarray        # condition part, entries in {0, 1, WILD}
    action: np.ndarray      # action part: 1 = trade (or consume), 0 = refuse
    strength: np.ndarray    # S, a running average of net receipts
    used: np.ndarray        # the counter tau, initialized at 1
    match: np.ndarray       # bit s is set if the rule matches state s
    sigma: np.ndarray       # specificity, 1 / (1 + number of #'s)
    born: np.ndarray        # a stamp that changes whenever the rule is replaced


@njit
def goods_matched(field):
    "Bitmask of the goods whose one-hot code satisfies a condition field."
    n_goods = field.shape[0]
    mask = 0
    for g in range(n_goods):
        ok = True
        for j in range(n_goods):
            if field[j] != WILD and field[j] != (1 if j == g else 0):
                ok = False
        if ok:
            mask |= 1 << g
    return mask


@njit
def refresh(R, i, r, n_goods):
    "Record which states rule r of type i matches, and its specificity."
    c = R.cond[i, r]
    if c.shape[0] == n_goods:                    # consumption rule: state = holding
        m = goods_matched(c)
    else:                                        # exchange rule: state = (own, partner)
        own, partner = goods_matched(c[:n_goods]), goods_matched(c[n_goods:])
        m = 0
        for x in range(n_goods):
            for y in range(n_goods):
                if (own >> x) & 1 and (partner >> y) & 1:
                    m |= 1 << (x * n_goods + y)
    R.match[i, r] = m
    R.sigma[i, r] = 1.0 / (1.0 + np.sum(c == WILD))


@njit
def set_rule(R, i, r, cond, action, strength, n_goods, stamp):
    "Overwrite rule r of type i with a new classifier, whose counter starts at 1."
    R.cond[i, r] = cond
    R.action[i, r] = action
    R.strength[i, r] = strength
    R.used[i, r] = 1
    stamp[0] += 1
    R.born[i, r] = stamp[0]
    refresh(R, i, r, n_goods)
```

完全枚举将表 1 中的每个条件与两个动作配对，而随机种群则独立地抽取每个位置。

遵循论文的做法，所有初始强度均为零。

```{code-cell} ipython3
def make_rules(cond, action, n_goods):
    n_types, n = action.shape
    R = Rules(cond.astype(np.int8), action.astype(np.int8),
              np.zeros((n_types, n)), np.ones((n_types, n), np.int64),
              np.zeros((n_types, n), np.int64), np.zeros((n_types, n)),
              np.zeros((n_types, n), np.int64))
    for i in range(n_types):
        for r in range(n):
            refresh(R, i, r, n_goods)
    return R


def complete_rules(n_types, n_goods, exchange, specific_only=False):
    "Every Table 1 condition (or pair of conditions) with both actions."
    conds = table1(n_goods)[:n_goods] if specific_only else table1(n_goods)
    if exchange:
        conds = np.array([np.concatenate([a, b]) for a in conds for b in conds])
    cond = np.tile(np.repeat(conds, 2, axis=0), (n_types, 1, 1))
    action = np.tile([0, 1], (n_types, len(conds)))
    return make_rules(cond, action, n_goods)


def random_rules(n_types, n_rules, length, n_goods, rng):
    "Rules whose positions are drawn independently and uniformly from {0, 1, #}."
    return make_rules(rng.integers(-1, 2, size=(n_types, n_rules, length)),
                      rng.integers(0, 2, size=(n_types, n_rules)), n_goods)
```

接下来是拍卖 {eq}`mms_auction`。

在练习中，它还提供了强度为 $\eta$ 的软最大化拍卖。

下面有几个算子都需要删除一条"弱"规则，论文将弱规则定义为"强度与过去在竞价中获胜次数的某种组合"来衡量。

我们的组合方式是将两者分别在候选规则范围内重新缩放到 $[0, 1]$ 后相加。

```{code-cell} ipython3
@njit
def auction(R, i, s, eta, rng):
    """
    The rule of type i that decides in state s, or -1 if no rule matches.

    eta = inf is the paper's auction: the strongest matched rule wins, with
    ties broken at random.  A finite eta gives a softmax auction instead.
    """
    S, match = R.strength[i], R.match[i]
    best, top, n_ties = -1, -np.inf, 0
    for r in range(S.shape[0]):
        if (match[r] >> s) & 1:
            if S[r] > top:
                best, top, n_ties = r, S[r], 1
            elif S[r] == top:
                n_ties += 1
                if rng.random() * n_ties < 1:
                    best = r
    if best < 0 or eta == np.inf:
        return best
    total = 0.0
    for r in range(S.shape[0]):
        if (match[r] >> s) & 1:
            total += np.exp(eta * (S[r] - top))
    x = rng.random() * total
    for r in range(S.shape[0]):
        if (match[r] >> s) & 1:
            x -= np.exp(eta * (S[r] - top))
            if x <= 0:
                return r
    return best


@njit
def weakest(R, i, s, exclude):
    """
    The weakest rule of type i other than `exclude`, scoring each candidate by
    its strength plus its number of wins, each rescaled to [0, 1].  Candidates
    are the rules matching state s, or all rules if none does or if s < 0.
    """
    S, used, match = R.strength[i], R.used[i], R.match[i]
    n = S.shape[0]
    cand = np.zeros(n, np.bool_)
    for r in range(n):
        cand[r] = r != exclude and (s < 0 or ((match[r] >> s) & 1) == 1)
    if not cand.any():
        for r in range(n):
            cand[r] = r != exclude
    lo, hi = S[cand].min(), S[cand].max()
    u_lo, u_hi = used[cand].min(), used[cand].max()
    best, best_score = -1, np.inf
    for r in range(n):
        if cand[r]:
            score = ((S[r] - lo) / (hi - lo + 1e-12)
                     + (used[r] - u_lo) / (u_hi - u_lo + 1e-12))
            if score < best_score:
                best, best_score = r, score
    return best
```

论文为随机规则（其第 6 节）新增的四个算子中有三个在拍卖内部起作用，因此我们在这里实现它们。

**创造**在没有规则匹配状态时触发：最弱的规则被一条条件恰好等于该状态、动作随机的规则取代。

**多样化**在所有匹配规则都要求相同动作时触发：匹配该状态的*另一条*最弱规则会被一条条件恰好对应此状态、动作相反、强度与获胜者相同的规则取代，使其与获胜者打平。

**专门化**在第 $t$ 期以概率 $f_s(t) = 1/(2\sqrt{t})$ 启动。

启动后，获胜者条件中的每个 $\#$ 都会以概率 $p_s = 0.01$ 被替换为与当前状态相匹配的比特，而这条经过锐化的副本会取代一条较弱的匹配规则。

在完全枚举下，每个状态都有匹配规则，且每个条件都以两种动作出现，因此创造和多样化永远不会触发，正如论文所指出的那样；专门化仅在使用随机规则时才会用到。

```{code-cell} ipython3
@njit
def decide(R, i, s, code, eta, specialize, ps, n_goods, rng, stamp):
    """
    Run the auction of type i in state s, whose encoding is `code`, applying
    the paper's creation, diversification and specialization operators.
    """
    w = auction(R, i, s, eta, rng)
    if w < 0:                                          # creation
        j = weakest(R, i, -1, -1)
        set_rule(R, i, j, code, rng.integers(0, 2), 0.0, n_goods, stamp)
        return j

    one_action = True                                  # diversification
    for r in range(R.action.shape[1]):
        if (R.match[i, r] >> s) & 1 and R.action[i, r] != R.action[i, w]:
            one_action = False
            break
    if one_action:
        j = weakest(R, i, s, w)
        set_rule(R, i, j, code, 1 - R.action[i, w], R.strength[i, w],
                 n_goods, stamp)
        w = auction(R, i, s, eta, rng)

    if specialize:                                     # specialization
        new = R.cond[i, w].copy()
        changed = False
        for k in range(new.shape[0]):
            if new[k] == WILD and rng.random() < ps:
                new[k] = code[k]
                changed = True
        if changed:
            j = weakest(R, i, s, w)
            set_rule(R, i, j, new, R.action[i, w], R.strength[i, w],
                     n_goods, stamp)
    return w
```

`Economy` 将物理环境的参数与学习算法的设置一起收集起来。

商品在代码中从 0 开始编号，在文中从 1 开始编号；如果存在法定货币，它是最后一种商品。

最后三个字段仅在练习中使用。

```{code-cell} ipython3
@dataclass
class Economy:
    name: str
    produces: np.ndarray               # produces[i] = good produced by type i
    storage_costs: np.ndarray          # one per good; fiat money, if any, is last
    u: float = 100.0                   # utility from consuming one's own good
    n_agents_per_type: int = 50
    random_rules: bool = False         # False: complete enumeration, no GA
    n_trade_rules: int = 72            # population sizes when rules are random
    n_consume_rules: int = 12
    b_trade: tuple = (0.025, 0.025)    # (b11, b12)
    b_consume: tuple = (0.25, 0.25)    # (b21, b22)
    n_fiat: int = 0                    # units of fiat money handed out at t = 0
    ps: float = 0.01                   # specialization: chance a # is filled in
    eta: float = np.inf                # inf: argmax auction; finite: softmax
    no_eat: bool = False               # True: only one's own good can be eaten
    specific_only: bool = False        # True: enumerate fully specific rules only

    @property
    def n_types(self):
        return len(self.produces)

    @property
    def n_goods(self):
        return len(self.storage_costs)

    @property
    def fiat(self):
        "Index of fiat money, or -1 if there is none."
        return self.n_goods - 1 if self.n_goods > self.n_types else -1
```

桶队算法每期为每个主体更新两个强度：如 {eq}`mms_strengthe` 所示，今天获胜的交换规则；如 {eq}`mms_strengthc` 所示，昨天为该主体获胜的消费规则——其第二笔收款直到现在才可知。

`born` 标记用于防止在此期间向已被覆盖的规则支付报酬。

```{code-cell} ipython3
@njit
def bucket_brigade(E, C, i, a, e, active, c, payoff, b, pending):
    """
    Update strengths for agent a of type i: first the exchange rule e that
    won today, then the consumption rule that won for this agent yesterday.
    """
    b11, b12, b21, b22 = b
    last_c, last_payoff, last_born = pending
    S_e = E.strength[i, e]                        # before its own update
    b1 = b11 + b12 * E.sigma[i, e]
    if active:                                    # equation (13)
        tau = E.used[i, e]
        E.used[i, e] = tau + 1
        b2 = b21 + b22 * C.sigma[i, c]
        E.strength[i, e] = S_e - ((1 + b1) * S_e - b2 * C.strength[i, c]) / tau
    p = last_c[a]
    if p >= 0 and C.born[i, p] == last_born[a]:   # equation (12)
        inflow = b1 * S_e if active else 0.0
        b2p = b21 + b22 * C.sigma[i, p]
        tau = C.used[i, p]
        C.used[i, p] = tau + 1
        C.strength[i, p] -= ((1 + b2p) * C.strength[i, p]
                             - inflow - last_payoff[a]) / tau
    last_c[a], last_payoff[a], last_born[a] = c, payoff, C.born[i, c]
```

每期，所有 $A$ 个主体被打乱并配成 $A/2$ 对，每对主体做出交换决策，若双方都愿意则成交，然后做出消费决策并结算账户。

```{code-cell} ipython3
@njit
def run_period(E, C, hold, types, produces, s, u, fiat, b, pending, eta,
               specialize, ps, no_eat, rng, stamp, exch, cons):
    "One period: random matching, two auctions per agent, and the accounting."
    n_goods = s.shape[0]
    order = rng.permutation(hold.shape[0])
    code = np.zeros(2 * n_goods, np.int8)
    e = np.zeros(2, np.int64)
    e_born = np.zeros(2, np.int64)
    offer = np.zeros(2, np.int64)
    for k in range(hold.shape[0] // 2):
        pair = (order[2 * k], order[2 * k + 1])
        x = (hold[pair[0]], hold[pair[1]])

        for m in range(2):                        # exchange decisions
            i = types[pair[m]]
            code[:] = 0
            code[x[m]] = 1
            code[n_goods + x[1 - m]] = 1
            e[m] = decide(E, i, x[m] * n_goods + x[1 - m], code, eta,
                          specialize, ps, n_goods, rng, stamp)
            e_born[m], offer[m] = E.born[i, e[m]], E.action[i, e[m]]
        trade = offer[0] == 1 and offer[1] == 1
        if trade:
            hold[pair[0]], hold[pair[1]] = x[1], x[0]
            exch[types[pair[0]], x[0], x[1]] += 1
            exch[types[pair[1]], x[1], x[0]] += 1

        for m in range(2):                        # consumption decisions
            a = pair[m]
            i, g = types[a], hold[a]
            code[:] = 0
            code[g] = 1
            c = decide(C, i, g, code[:n_goods], eta, specialize, ps,
                       n_goods, rng, stamp)
            cons[i, g, 0] += 1
            if C.action[i, c] == 1 and g != fiat and (not no_eat or g == i):
                cons[i, g, 1] += 1
                hold[a] = produces[i]
                payoff = (u if g == i else 0.0) - s[produces[i]]
            else:
                payoff = -s[g]
            active = (trade or offer[m] == 0) and E.born[i, e[m]] == e_born[m]
            bucket_brigade(E, C, i, a, e[m], active, c, payoff, b, pending)
```

`Simulation` 保存一个经济体的状态，并逐期运行，记录持有量、交易和消费决策。

初始持有量是随机的，与论文一致，法定货币被分发给随机选中的主体。

在使用随机规则时，遗传算法 `generalize` 在每期之后运行；它定义在 {ref}`mms_ga` 中，位于第一个需要用到它的经济体之前。

```{code-cell} ipython3
class Simulation:

    def __init__(self, econ, seed=0):
        self.econ = econ
        self.rng = rng = np.random.default_rng(seed)
        K, G = econ.n_types, econ.n_goods
        if econ.random_rules:
            self.E = random_rules(K, econ.n_trade_rules, 2 * G, G, rng)
            self.C = random_rules(K, econ.n_consume_rules, G, G, rng)
        else:
            self.E = complete_rules(K, G, True, econ.specific_only)
            self.C = complete_rules(K, G, False, econ.specific_only)
        self.types = np.repeat(np.arange(K), econ.n_agents_per_type)
        A = len(self.types)
        self.hold = rng.integers(0, K, size=A)       # random commodity holdings
        if econ.n_fiat:
            self.hold[rng.choice(A, econ.n_fiat, replace=False)] = econ.fiat
        self.pending = (np.full(A, -1), np.zeros(A), np.zeros(A, np.int64))
        self.stamp = np.zeros(1, np.int64)
        self.t = 0
        self.hold_hist, self.exch_hist, self.cons_hist = [], [], []

    def run(self, T):
        econ, rng = self.econ, self.rng
        K, G = econ.n_types, econ.n_goods
        b = tuple(float(v) for v in econ.b_trade + econ.b_consume)
        for _ in range(T):
            self.t += 1
            f = 1 / (2 * np.sqrt(self.t))   # rate of specialization and the GA
            specialize = econ.random_rules and rng.random() < f
            exch, cons = np.zeros((K, G, G)), np.zeros((K, G, 2))
            run_period(self.E, self.C, self.hold, self.types, econ.produces,
                       econ.storage_costs, econ.u, econ.fiat, b, self.pending,
                       econ.eta, specialize, econ.ps, econ.no_eat, rng,
                       self.stamp, exch, cons)
            if econ.random_rules:
                for i in range(K):
                    if rng.random() < f:
                        generalize(self.E, i, G, rng, self.stamp, n_crowd=8)
                    if rng.random() < f:
                        generalize(self.C, i, G, rng, self.stamp, n_crowd=4)
            self.hold_hist.append([np.bincount(self.hold[self.types == i],
                                               minlength=G) for i in range(K)])
            self.exch_hist.append(exch)
            self.cons_hist.append(cons)
```

下面的代码单元格定义了一些辅助函数，用于将输出整理成论文表格（报告十期移动平均值）和图形的格式。

`winners` 列出每个状态下将在拍卖中获胜的规则，`contenders` 列出在某状态下参与竞争的所有规则。

`across_seeds` 对一个经济体使用二十个不同的种子重新运行。

```{code-cell} ipython3
:tags: [hide-input]

def good_names(econ):
    names = [f"good {k+1}" for k in range(econ.n_types)]
    return names + ["fiat"] if econ.fiat >= 0 else names


def type_names(econ):
    return [f"type {i+1}" for i in range(econ.n_types)]


def holdings(sim, t=None, window=10):
    "Table of pi^h_i(k), averaged over the `window` periods ending at t."
    h = np.array(sim.hold_hist) / sim.econ.n_agents_per_type
    t = len(h) if t is None else t
    return pd.DataFrame(h[max(0, t - window):t].mean(axis=0),
                        index=type_names(sim.econ),
                        columns=good_names(sim.econ)).round(3)


def strongest(R, i, s):
    "Strongest rule of type i matching state s (ties go to the first), or -1."
    idx = np.flatnonzero((R.match[i] >> s) & 1)
    return -1 if idx.size == 0 else idx[np.argmax(R.strength[i, idx])]


def winners(sim, i, exchange=True):
    "The rule of type i that wins in each state, with its strength and uses."
    econ, G = sim.econ, sim.econ.n_goods
    R = sim.E if exchange else sim.C
    names = good_names(econ)
    states = ([(j, k) for j in range(G) for k in range(G)] if exchange
              else [(j,) for j in range(G)])
    rows = []
    for st in states:
        w = strongest(R, i, st[0] * G + st[1] if exchange else st[0])
        rows.append({'state': ', '.join(names[g] for g in st),
                     'winning rule': rule_string(R.cond[i, w], R.action[i, w], G),
                     'strength': round(R.strength[i, w], 2),
                     'times used': R.used[i, w] - 1})
    return pd.DataFrame(rows).set_index('state')


def eating_rates(sim, window=200):
    "Fraction of the time each type eats each good it holds after trading."
    c = np.array(sim.cons_hist[-window:]).sum(axis=0)
    rate = c[:, :, 1] / np.maximum(c[:, :, 0], 1)
    return pd.DataFrame(rate, index=type_names(sim.econ),
                        columns=good_names(sim.econ)).round(2)


def contenders(sim, i, j, k):
    "All exchange rules of type i matching the state (good j+1, good k+1), strongest first."
    G, R = sim.econ.n_goods, sim.E
    idx = np.flatnonzero((R.match[i] >> (j * G + k)) & 1)
    idx = idx[np.argsort(-R.strength[i, idx], kind='stable')]
    return pd.DataFrame({'rule': [rule_string(R.cond[i, r], R.action[i, r], G)
                                  for r in idx],
                         'strength': R.strength[i, idx].round(2),
                         'times used': R.used[i, idx] - 1})


def trade_rates(sim, window=100):
    "Trades per agent per period over the last `window` periods, indexed [type, gives, gets]."
    return np.array(sim.exch_hist[-window:]).mean(axis=0) / sim.econ.n_agents_per_type


def across_seeds(econ, T, seeds=range(20)):
    "Run the economy for T periods once for each seed."
    sims = []
    for seed in seeds:
        sim = Simulation(econ, seed)
        sim.run(T)
        sims.append(sim)
    return sims


def final_holdings(sims, t=None):
    "Holdings at t (ten-period averages) for each simulation, as an array."
    return np.array([holdings(sim, t=t).to_numpy() for sim in sims])


def distance(H, target):
    "Largest absolute difference between each run's holdings and a target table."
    return np.abs(H - np.asarray(target)).max(axis=(-2, -1))


def plot_holdings(sim):
    econ = sim.econ
    h = np.array(sim.hold_hist) / econ.n_agents_per_type
    names = good_names(econ)
    fig, axes = plt.subplots(1, econ.n_types, figsize=(3.2 * econ.n_types, 3.2),
                             sharey=True)
    for i, ax in enumerate(axes):
        for k in range(econ.n_goods):
            ax.plot(h[:, i, k], lw=1.0, label=names[k])
        ax.set_title(f"type {i+1}")
        ax.set_xlabel("$t$")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(r"$\pi^h_{it}(k)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc='center right',
               frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0, 0.92, 1))
    plt.show()


def plot_flows(sim, window=100, cutoff=0.02):
    """
    One panel per type.  An arrow from good j to good k means that agents of
    that type give up j and receive k; its width is the frequency of that
    exchange per agent per period over the last `window` periods.
    """
    econ = sim.econ
    f = trade_rates(sim, window)
    names, n = good_names(econ), econ.n_goods
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    xy = np.column_stack([np.cos(ang), np.sin(ang)])
    fig, axes = plt.subplots(1, econ.n_types, figsize=(2.9 * econ.n_types, 3.1))
    for i, ax in enumerate(axes):
        for k in range(n):
            ax.plot(*xy[k], 'o', ms=22, mfc='white', mec='black', zorder=2)
            ax.annotate(names[k].replace(' ', '\n'), xy[k], ha='center',
                        va='center', fontsize=6.5, zorder=3)
        for j in range(n):
            for k in range(n):
                if j == k or f[i, j, k] < cutoff:
                    continue
                d = xy[k] - xy[j]
                ax.annotate("", xy=xy[k] - 0.24 * d, xytext=xy[j] + 0.24 * d,
                            zorder=1,
                            arrowprops=dict(arrowstyle="-|>", color="C0",
                                            lw=1 + 8 * f[i, j, k], alpha=0.7,
                                            connectionstyle="arc3,rad=0.15"))
        ax.set_title(f"type {i+1}", fontsize=10)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

## 经济体 A1.1：交换媒介会出现吗？

我们的第一个经济体是维克塞尔三角，其参数为

$$
s_1 = 0.1, \quad s_2 = 1, \quad s_3 = 20, \quad u_i = 100 ,
$$

每种类型五十个主体，并对 72 个交换分类器和 12 个消费分类器进行完全枚举，全部强度为零。

我们展示的每个经济体都使用种子 0 运行，然后在 20 个种子（经济体 D 为 10 个种子）上检验其代表性。

```{code-cell} ipython3
economy_a11 = Economy(
    name='A1.1',
    produces=np.array([1, 2, 0]),               # type 1 -> good 2, etc. (0-indexed)
    storage_costs=s_A,
    u=100.0,
)

sim_a11 = Simulation(economy_a11, seed=0)
sim_a11.run(1000)

fundamental = np.array([[0.0, 1.0, 0.0],
                        [0.5, 0.0, 0.5],
                        [1.0, 0.0, 0.0]])
paper_a11 = np.array([[0.0, 1.0, 0.0],          # the paper's Table 6b, t = 1000
                      [0.506, 0.0, 0.494],
                      [1.0, 0.0, 0.0]])

print("holdings at t = 500\n", holdings(sim_a11, t=500), "\n")
print("holdings at t = 1000\n", holdings(sim_a11), "\n")
print(f"max |simulated - fundamental equilibrium| = "
      f"{distance(holdings(sim_a11).to_numpy(), fundamental):.3f}")
print(f"max |simulated - paper's table|           = "
      f"{distance(holdings(sim_a11).to_numpy(), paper_a11):.3f}")
```

到 $t = 500$ 时，类型 1 持有商品 2，类型 3 持有商品 1，类型 2 在商品 1 和商品 3 之间分配其时间，正如上面列出的基本均衡所示。

时间路径显示了类型 1 和类型 3 收敛得有多快，以及类型 2 学习过程中的一段停顿。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Holdings in Economy A1.1"
    name: fig-mms-a11-holdings
---
plot_holdings(sim_a11)
```

在大约最初的两百个时期内，类型 2 主体只持有商品 1，别无其他：他们已经获得了货币，但尚未将其花出去。

用商品 1 购买商品 2 的规则尚未获得机会，原因我们将在 {ref}`mms_a2` 中解释；一旦该规则在首次拍卖中获胜，类型 2 便进入“获取货币、花费货币”的循环。

从那以后，类型 2 的持有量开始波动，因为类型 2 主体持有哪种商品取决于它在该循环中所处的位置。

下图显示了交易本身，是对最后一百个时期的平均值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Exchange pattern discovered in Economy A1.1"
    name: fig-mms-a11-flows
---
plot_flows(sim_a11)
```

基本均衡三角形就在其中：类型 1 放弃商品 2 换取商品 1，类型 3 放弃商品 1 换取商品 3，类型 2 完成两段流程，先放弃商品 3 换取商品 1，随后再放弃商品 1 换取商品 2。

商品 1 作为货币流通。

图中还有基本均衡所没有的箭头，而且这些箭头并不细：类型 3 放弃商品 1 换取商品 2 的频率大约是放弃商品 1 换取商品 3 的两倍，而类型 1 放弃商品 2 换取商品 3 的频率大约与类型 2 完成其均衡流程中任一段的频率相当。

每一个都是上文所述的那种支付中性交易：主体吃掉所得到的商品，然后重新生产自己的商品；其交易伙伴因此获益，因为类型 1 现在既能从类型 2 也能从类型 3 那里获得商品 1，而类型 2 则可以直接用商品 3 换得商品 2。

论文中的那次运行没有出现这些交易，但核算规则本身并不排除它们。

我们还可以深入类型 1 的分类器系统内部，读出每个状态下的获胜规则，先是交换规则，然后是消费规则。

```{code-cell} ipython3
pd.concat([winners(sim_a11, 0), winners(sim_a11, 0, exchange=False)])
```

起作用的规则 `##0 100 -> 1` 表示“如果我持有的不是商品 3，而我的伙伴持有商品 1，则交易”：这是一条通用规则。

在状态（商品 2，商品 3）下，获胜规则会*接受*商品 3，但获得商品 3 的类型 1 主体会立即将其吃掉，并重新生产商品 2，这使他恰好回到拒绝交易时的状态。

这正是上文所述的支付中性交易，也是我们之所以从持有量（从未显示类型 1 持有商品 3）而非从获胜动作中读取结果的原因。

论文中的那次运行（其表 6）显示类型 1 拒绝商品 3，但得到了相同的持有量结果。

种子 0 具有代表性吗？

```{code-cell} ipython3
sims_a11 = across_seeds(economy_a11, 1000)
H_a11 = final_holdings(sims_a11)
F_a11 = np.array([trade_rates(sim) for sim in sims_a11])
d, n = distance(H_a11, fundamental), len(sims_a11)
print(f"runs within 0.1 of the fundamental equilibrium: {np.sum(d < 0.1)} of {n}")
print(f"largest distance in any run: {d.max():.3f}")
print(f"runs in which type 3 gives good 1 for good 2:   {np.sum(F_a11[:, 2, 0, 1] > 0.02)} of {n}")
print(f"runs in which type 1 gives good 2 for good 3:   {np.sum(F_a11[:, 0, 1, 2] > 0.02)} of {n}")
```

在每一次运行中，持有量都会收敛到唯一的平稳均衡，而在大多数运行中，支付中性交易也会随之出现。

交换媒介是可以学得的。

(mms_a2)=
## 经济体 A2.1：当理论预测投机时

经济体 A2 与 A1 唯一的区别在于 $u_i = 500$ 而非 $100$。

这足以违反 Kiyotaki-Wright 不等式 {eq}`mms_fundcond`，因此对于耐心的主体来说，此时唯一的平稳均衡变成了**投机**均衡，在其中类型 1 主体预期能将商品 3 换成商品 1，因而接受商品 3。

我们的适应性主体能找到它吗？

```{code-cell} ipython3
economy_a21 = replace(economy_a11, name='A2.1', u=500.0)

sim_a21 = Simulation(economy_a21, seed=0)
sim_a21.run(1000)

speculative = pd.DataFrame([[0, 0.707, 0.293],        # the paper's Table 8b
                            [0.586, 0, 0.414],
                            [1, 0, 0]],
                           index=type_names(economy_a21),
                           columns=good_names(economy_a21))
print("simulated holdings at t = 1000\n", holdings(sim_a21), "\n")
print("speculative equilibrium\n", speculative)
```

答案是否定的。

模拟的持有量是*基本*模式的持有量：类型 1 主体持有商品 2，从不持有商品 3，而投机均衡则要求他们大约三成的时间持有商品 3。

以下是不同随机种子下的结果。

```{code-cell} ipython3
sims_a21 = across_seeds(economy_a21, 1000)
H_a21 = final_holdings(sims_a21)
n = len(sims_a21)
print(f"type 1 holds good 3 more than 5% of the time:  {np.sum(H_a21[:, 0, 2] > 0.05)} of {n}")
print(f"within 0.1 of the fundamental pattern:         "
      f"{np.sum(distance(H_a21, fundamental) < 0.1)} of {n}")
print(f"type 2 holds good 1 less than 10% of the time: {np.sum(H_a21[:, 1, 0] < 0.1)} of {n}")
print(f"type 2 holds good 1 more than 90% of the time: {np.sum(H_a21[:, 1, 0] > 0.9)} of {n}")
```

类型 1 从不投机。

一半的运行在 $t = 1000$ 时已接近基本模式。

在几乎所有其余的运行中，类型 2 几乎从不持有商品 1：类型 2 和类型 3 已经停止相互交易，因此类型 2 永远无法获得用来购买商品 2 所需的货币。

在一次运行中，类型 2 始终持有商品 1：它有货币但从不花掉它，正如我们 A1.1 运行中头两百个周期的情形一样。

以下是这种更常见情形中的第一次运行，通过持有商品 1 并遇到持有商品 3 的类型 2 主体的类型 3 主体的交换规则来观察。

```{code-cell} ipython3
locked = int(np.flatnonzero(H_a21[:, 1, 0] < 0.1)[0])
sim_locked = sims_a21[locked]
print(f"seed {locked}\n", holdings(sim_locked), "\n")
c = contenders(sim_locked, 2, 0, 2)          # type 3 holds good 1, meets good 3
pd.concat([c.head(2), c[c['rule'] == '100 001 -> 1']])
```

类型 3 主体想要的商品 3 对它而言价值 500，但获胜的规则 `##0 001 -> 0` 却拒绝交易。

它的强度仅为 $0.1$，但为正值；而本应进行交易的特定规则 `100 001 -> 1` 从未被使用过，仍保持其初始强度零。

这条拒绝规则在最初几个周期获得了一笔可观的回报，当时有一条同样覆盖商品 3 的通用消费规则紧随其后。

由于采用 $1/\tau$ 平均法，这个平均值会在数千次使用后仍然记住那笔意外之财。

这正是乐观初始化的失效模式，而贪婪拍卖机制直到拒绝规则的平均值降至零以下之前，都不会给交易规则机会。

但这种情况最终确实会发生，因此以这种方式被锁定的运行只是缓慢而非永久卡住。

我们将那些在 $t = 1000$ 时尚未接近基本模式的运行再延续 5000 个周期。

```{code-cell} ipython3
slow = [sim for sim, h in zip(sims_a21, H_a21) if distance(h, fundamental) >= 0.1]
for sim in slow:
    sim.run(5000)
H_slow = final_holdings(slow)
print(f"runs not near the fundamental pattern at t = 1000: {len(slow)}")
print(f"of these, within 0.1 of it at t = 6000:            "
      f"{np.sum(distance(H_slow, fundamental) < 0.1)}")
print(f"of these, type 1 holds good 3 more than 5%:        {np.sum(H_slow[:, 0, 2] > 0.05)}")
```

所有这些运行到 $t = 6000$ 时都达到了基本模式，且类型 1 仍然从不投机。

这里的锁定现象比经济体 A1.1 中更为频繁，这很可能是因为 $u_1$ 大了五倍，因此平均值需要遗忘的早期意外之财也更大。

不过，并非每一次锁定都会消退。

其他随机种子揭示了第二种失效模式，种子 24 便展示了这一点。

```{code-cell} ipython3
sim_24 = Simulation(economy_a21, seed=24)
sim_24.run(6000)
print(holdings(sim_24), "\n")
winners(sim_24, 1, exchange=False)
```

这一次类型 3 确实接受了商品 3；问题出在类型 2 身上。

它像基本模式那样用商品 3 换得商品 1，但随后却吃掉商品 1，而不是留着它向类型 1 购买商品 2，结果每个周期结束时又重新持有商品 3。

造成这种情况的规则是通用消费规则 `##0 -> 1`，无论类型 2 持有商品 1 还是商品 2，它都说“吃掉”。

吃掉商品 2 是正确的，它能为类型 2 带来消费效用，这使得该规则的强度保持高位；吃掉商品 1 是错误的，但同一条规则也决定了这种状态下的行为。

本应保留商品 1 的特定规则 `100 -> 0` 只被尝试过几次，其强度远低于通用规则——后者的平均值包含了每一次吃掉商品 2 的收益——因此在贪婪最大化（argmax）拍卖机制下，它再也无法获胜。

因此这次运行永久卡住了：种子 24 以及以同样方式卡住的种子 26，在 $t = 24{,}000$ 时仍处于这种模式。

乐观初始化使得一条曾经走运的规则能够排斥未经尝试的对手规则；这第二种锁定现象，是一条过度泛化的消费规则所致——它在一种状态下表现良好，却排斥了本应在另一种状态下正确行动的特定规则。

类型 1 仍然从不投机：在种子 20 到 59 的所有运行中，它持有商品 3 的时间都未超过 5%。

因此，主体们并未找到 A2 的均衡。

对于耐心的主体而言，投机均衡是该经济体*唯一*的平稳均衡，因此这并非学习在多个均衡之间进行选择。

而是收敛到了一个并非该经济体均衡的结果。

论文从单次运行中得出了相同的结论，其诊断值得完整引用：

> *耐心需要经验。* 分类器系统内部的转移系统被设计为收敛到一组长期平均强度。在极限情况下，人工智能主体应该表现为长期平均收益最大化者，因为稳态强度是按相对频率对收益加权的……然而，最优规则要达到期望的强度需要时间。我们的人工智能主体在早期的行为可能非常短视。在诸如经济体 A2 这样均衡的性质随贴现率变化的经济体中，这种早期的短视可能产生反常效果，将经济体引向一个低贴现因子的平稳均衡，例如基本均衡。所采用的算法必须提供足够的实验，以避免早期的“锁定”。目前的算法似乎存在缺陷，即使在我们运行的长时间模拟中，它的实验也太少，不足以支持投机均衡。

论文的脚注 14 补充说，只有当贴现因子足够低时，A2 中才存在基本均衡。

其脚注 15 报告了一个耐人寻味的对比：在 Marimon 和 Miller 的实验中，他们使用一种培育整套交易策略的遗传算法（论文脚注 1 和脚注 3 所描述的那种方法），经济体 A2 的 30 次运行中有 25 次收敛到了投机均衡。

他们的主体没有赛段分配制（bucket brigade）的功劳归属机制：每种策略是作为一个整体来评判的。

论文还指出了第二个罪魁祸首。

在其运行中，类型 1 获胜的交换规则*接受*了商品 3（见其表 9d），但类型 1 获胜的消费规则“过于泛化（带有太多 # 号），无法区分储藏的商品”，因此类型 1 把商品 3 吃掉了，而不是把它带给类型 3 主体。

接受商品 3 然后又吃掉它，与拒绝它在收益上是等价的，这就是为什么持有量看起来像基本模式的原因。

我们的种子 0 也展示了同样的情况。

```{code-cell} ipython3
pd.concat([winners(sim_a21, 0).loc[['good 2, good 1', 'good 2, good 3']],
           winners(sim_a21, 0, exchange=False).loc[['good 1', 'good 3']]])
```

类型 1 在状态（商品 2，商品 3）下获胜的交换规则 `010 0## -> 1` 接受了商品 3，而在类型 1 持有商品 3 时获胜的消费规则是通用规则 `#0# -> 1`，意为“如果我不持有商品 2，就吃掉”。

到目前为止，这正是论文的表 9d。

但使类型 1 在这里吃掉商品 3 的原因并非泛化本身。

以下是类型 1 所有匹配商品 3 的消费规则。

```{code-cell} ipython3
c = sim_a21.C
rows = [r for r in range(c.cond.shape[1]) if (c.match[0, r] >> 2) & 1]
pd.DataFrame({'rule': [rule_string(c.cond[0, r], c.action[0, r], 3) for r in rows],
              'strength': c.strength[0, rows].round(2),
              'times used': c.used[0, rows] - 1})
```

吃掉商品 3 的特定规则强度接近零，而带着商品 3 的特定规则，只被尝试过一次，强度接近 $-12$。

在当前周期携带商品 3 的成本为 20，而吃掉它的成本仅为 1，而携带它所换来的奖励——即下一期有机会将其换成商品 1——只能通过赛段分配制传递到携带规则，而且被大幅贴现。

因此类型 1 从不将商品 3 携带到下一期，持有量呈现为基本模式。

因此，A2 中存在三种机制：

1. **不耐心。** 即便处于静止状态，赛段分配制也会以一个极小的贴现因子评估未来价值，下一节将推导出这一点。
1. **贪婪锁定。** argmax 拍卖机制一旦用光了未经尝试的规则，就会停止实验，如上文被困的运行所示。
1. **通用消费规则导致的过度消费**，这使得接受商品 3 变得毫无意义，正如论文的表 9 所示，并且在种子 24 中使得类型 2 无法持有货币。

{ref}`mms_ex3`、{ref}`mms_ex4` 和 {ref}`mms_ex5` 将逐一拆解这些机制。

(mms_impatience)=
## 桶队算法的贴现因子

该论文曾希望，由长期平均值构建起来的强度一旦稳定下来，会使智能体的行为表现得如同长期平均收益最大化者。

然而平稳性条件却说明了相反的情况。

考虑这样一个智能体：其在 $t+1$ 时刻的交换规则 $e'$ 跟随 $t$ 时刻的消费规则 $c$，而其在 $t+1$ 时刻的消费规则 $c'$ 跟随 $e'$。

由第二个平稳性条件，$(1 + b_1(e'))S_{e'} = \mathbb{E}[b_2(c') S_{c'}]$。

将其代入第一个条件，得到

```{math}
:label: mms_effbeta

(1 + b_2(c)) S_c = \mathbb{E}\bigl[U(\gamma_c)\bigr]
 + \mathbb{E}\Bigl[\beta_{\rm eff}\,(1 + b_2(c')) S_{c'}\Bigr],
\qquad
\beta_{\rm eff} = \frac{b_1(e')\, b_2(c')}{(1 + b_1(e'))(1 + b_2(c'))} .
```

因此，经缩放后的强度 $V_c = (1 + b_2(c)) S_c$ 满足一个关于智能体当前策略的贝尔曼方程，$V_c = \mathbb{E}[U] + \mathbb{E}[\beta_{\rm eff} V_{c'}]$，其贴现因子为 $\beta_{\rm eff}$。

交换强度也继承了这种排序关系：在具有相同特定程度的交换规则中，$S_e$ 与跟随 $e$ 之后的消费规则的期望值成正比。

当某个报价被拒绝时，$e'$ 不会向后传递任何值，这会进一步降低有效贴现因子。

```{code-cell} ipython3
def beta_eff(b1, b2):
    "Effective discount factor of the bucket brigade for bid fractions b1, b2."
    return b1 * b2 / ((1 + b1) * (1 + b2))


b11, b12, b21, b22 = 0.025, 0.025, 0.25, 0.25            # Economy A's bids
β_A = beta_eff(b11 + b12, b21 + b22)                     # fully specific rules
print(f"β_eff with A's bids, specific rules:   {β_A:.4f}")
print(f"β_eff with A's bids, general rules:    {beta_eff(b11 + b12 / 5, b21 + b22 / 3):.4f}")
print(f"β_eff with B.1's bids, specific rules: {beta_eff(0.5, 0.5):.4f}")
print(f"β_eff as both bid fractions approach 1: {beta_eff(1, 1):.4f}")
```

按照论文给出的出价，智能体在静止状态下的行为就如同以约 $0.016$ 的因子逐期贴现未来；而对于出价最低的最一般规则，这个因子约为 $0.007$。

由于对任何小于一的出价比例，都有 $b/(1+b) < 1/2$，所以 $\beta_{\rm eff}$ 永远达不到 $1/4$。

其根源就在于求平均本身：某条规则自身的出价也会进入它所保留的平均值之中，因此链条上的每一环只会把所获得价值中 $b/(1+b)$ 这一部分传回去。

贴现因子为 $\beta$ 的智能体在经济 A 中会如何行动呢？

重新进行支撑 {eq}`mms_fundcond` 的那套论证，针对一个将每期收益按 $\beta$ 贴现的智能体，在给定其他类型持有量的情况下，第一类智能体当且仅当满足以下条件时才偏好持有商品 3：

$$
s_3 - s_2 < \beta \bigl(\pi^h_3(1) - \pi^h_2(1)\bigr) \tfrac{1}{3} u_1 ,
$$

因此，降低 $\beta$ 会把 $u_1$ 的阈值按 $1/\beta$ 的比例放大。

```{code-cell} ipython3
for label, β in [("patient agents, β = 1", 1.0), ("bucket brigade, β = β_eff", β_A)]:
    print(f"{label}: fundamental for u_1 < "
          f"{speculation_threshold(s_A, 1.0, 0.5, β):,.0f}, speculative for u_1 > "
          f"{speculation_threshold(s_A, 1.0, 0.586, β):,.0f}")
print(f"A2's u_1 = 500 supports speculation only if β > "
      f"{speculation_threshold(s_A, 1.0, 0.586) / 500:.3f}")
```

对于 $\beta = \beta_{\rm eff}$ 的智能体而言，A2 的 $u_1 = 500$ 远远落在基本模式为唯一均衡的区域之内。

下图绘制出了两个阈值随 $\beta$ 变化的关系。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Speculation thresholds for $u_1$ against the discount factor"
    name: fig-mms-thresholds
---
βs = np.logspace(-2.5, 0, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(βs, speculation_threshold(s_A, 1.0, 0.5, βs),
          label="fundamental equilibrium exists below")
ax.loglog(βs, speculation_threshold(s_A, 1.0, 0.586, βs), '--',
          label="speculative equilibrium exists above")
for u, name in [(100, "A1"), (500, "A2")]:
    ax.axhline(u, color='k', lw=0.6)
    ax.annotate(f"{name}: $u_1 = {u}$", (βs[0], u * 1.1), fontsize=9)
ax.axvline(β_A, color='C3', lw=1)
ax.axvline(0.25, color='C3', lw=1, ls=':')
ax.annotate(r"$\beta_{\rm eff}$", (β_A * 1.08, 150), color='C3')
ax.annotate("1/4", (0.25 * 1.08, 150), color='C3')
ax.set_xlabel(r"discount factor $\beta$")
ax.set_ylabel(r"$u_1$")
ax.legend(frameon=False, loc='upper right')
plt.show()
```

经济 A2 只有在 $\beta$ 高于 $0.275$ 时才支持投机行为，这超出了任何出价比例所能达到的范围，因此在静止状态下，无论出价如何，桶队算法在完全特定规则下都无法支撑 A2 的均衡。

这使论文的诊断更加精确。

早期的短视确实存在，但并不仅限于早期：平稳状态下的强度本身就是缺乏耐心的。

基本模式——只在吃掉某种商品或储存该商品更便宜时才接受它——正是一个缺乏耐心的智能体所会采取的行为，这正是论文所称的“低贴现因子平稳均衡”。

本讲座中分类器系统在其他每个经济体中所发现的模式也是如此：经济 B 的基本均衡、经济 C 中对法定货币的接受（法定货币的储存成本为零），以及论文针对经济 D 所报告的“更低储存成本”模式。

方程 {eq}`mms_effbeta` 是一个我们熟悉的对象。

经缩放的强度的评估方式，与强化学习中的时序差分价值估计方法相似 {cite}`Sutton_2018`，{doc}`mccall_q` 和 {doc}`inventory_q` 分别将其应用于求职者和企业的问题。

区别在于，此处的贴现因子并非根据智能体的偏好来选定：它是出价比例的一个副产物。

{ref}`gc_bucket_brigade` 在一个两环链条中发现了同样的因子 $b/(1+b)$，并表明霍兰德提出的累积余额（它不进行平均）会无损地向后传递收益，但未必收敛。

{ref}`br_one_recursion` 解释了为什么，在采用 $1/\tau$ 平均方式的情况下，{eq}`mms_strengthc`—{eq}`mms_strengthe` 的静止点正是我们需要研究的对象。

{ref}`mms_ex5` 通过模拟检验了该阈值。

## 经济体 B.1：在两个均衡之间进行选择

经济体 B 改变了生产模式——类型 1 生产商品 3，类型 2 生产商品 1，类型 3 生产商品
2——并将存储成本压缩为 $s = (1, 4, 9)$，其中 $u_i = 100$。

现在耐心主体存在两个平稳均衡：在基本均衡中，商品 1 和 2 充当交换媒介；而在投机均衡中，商品 2 和 3 充当交换媒介。

B.1 还使用了是 A 中十倍的交换出价，$b_{11} = b_{12} = 0.25$，与论文表 12 中一致，这使得 $\beta_{\rm eff}$ 升至约 $0.11$。

```{code-cell} ipython3
economy_b1 = Economy(
    name='B.1',
    produces=np.array([2, 0, 1]),               # type 1 -> good 3, type 2 -> good 1, ...
    storage_costs=np.array([1.0, 4.0, 9.0]),
    u=100.0,
    b_trade=(0.25, 0.25),
)

sim_b1 = Simulation(economy_b1, seed=0)
sim_b1.run(1000)

fundamental_b = np.array([[0, 0.293, 0.707],    # the paper's Table 10b
                          [1, 0, 0],
                          [0.586, 0.414, 0]])
speculative_b = np.array([[0, 0.586, 0.414],    # the paper's Table 11b
                          [0.707, 0, 0.293],
                          [0, 1, 0]])
paper_b1 = np.array([[0.0, 0.28, 0.72],         # the paper's Table 12b, t = 1000
                     [0.994, 0.0, 0.006],
                     [0.526, 0.474, 0.0]])

print("holdings at t = 500\n", holdings(sim_b1, t=500), "\n")
print("holdings at t = 1000\n", holdings(sim_b1), "\n")
h = holdings(sim_b1).to_numpy()
for name, target in [("fundamental equilibrium", fundamental_b),
                     ("speculative equilibrium", speculative_b),
                     ("paper's table", paper_b1)]:
    print(f"max |simulated - {name}| = {distance(h, target):.3f}")
print(f"max |paper's table - fundamental equilibrium| = "
      f"{distance(paper_b1, fundamental_b):.3f}")
```

该次运行最终收敛到经济体 B 的基本均衡附近，且比论文自身的运行结果更接近该均衡。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Holdings in Economy B.1"
    name: fig-mms-b1-holdings
---
plot_holdings(sim_b1)
```

跨种子（seed）结果如下：

```{code-cell} ipython3
H_b1 = final_holdings(across_seeds(economy_b1, 1000))
d_f, d_s = distance(H_b1, fundamental_b), distance(H_b1, speculative_b)
print(f"within 0.1 of the fundamental equilibrium: {np.sum(d_f < 0.1)} of {len(H_b1)}")
print(f"within 0.1 of the speculative equilibrium: {np.sum(d_s < 0.1)} of {len(H_b1)}")
print(f"smallest distance from the speculative equilibrium: {d_s.min():.2f}")
print("holdings in the runs farther than 0.1 from the fundamental equilibrium:")
print(H_b1[d_f >= 0.1].round(2))
```

在 20 次运行中有 18 次的最终持有量落在距基本均衡 0.1 以内，没有任何一次运行接近投机均衡到 0.29 以内。

在另外两次运行中，一次恰好略微超出 0.1 的范围，另一次中类型 3 的主体始终没有用其商品 2 交换商品 1。

在这里，学习确实在两个均衡之间进行了选择，并且选择了基本均衡。

论文还报告了一个我们的运行未能重现的特征：在 $t = 500$ 时，其 B.1 看起来呈投机性，直到 $t = 1000$ 才"实际上收敛到了基本均衡"。

论文将这一变化解读为该选择并非"短视行为的结果"的证据。

在我们的运行中，除了最初由随机初始持有量主导的前几期之外，并不存在投机阶段，因此我们无法证实这一变化。

而 {ref}`mms_impatience` 提示了另一种解读：经济体 B 的基本均衡也正是一个不耐心的主体所会选择的均衡，而"流水线式传递"（bucket brigade）在稳定状态下本身就是不耐心的，而不仅仅是在初期如此。

(mms_ga)=
## 遗传算法

完全枚举之所以可行，只是因为 Kiyotaki-Wright 的状态空间很小。

在更大的问题中，主体必须使用一个持续修订的有限规则种群。

论文让这类主体从随机生成的规则起步，并添加了四种算子：我们已经在拍卖中遇到过的三种，以及**泛化**——一种遗传算法。

泛化在第 $t$ 期以概率 $f_g(t) = 1/(2\sqrt{t})$ 被调用。

当它被调用时，其工作方式如下（见论文第 6 节和表 7）：

* **潜在淘汰对象**是既弱又很少被使用的规则：强度低于 $\bar S = 0$ *且* 计数器低于最大计数器的 $p_3 = 0.2$ 倍。
* **潜在父本**是种群中按“强度和频率”随机选取的 $p_2 = 0.7$ 比例的规则；我们以与其计数器成比例的概率抽取它们，然后在其中通过基于强度的轮盘赌选择配对。
* 配对不断繁殖，直至种群中 $p_1 = 0.2$ 的比例被替换。
* **泛化交叉**（论文图 4）：这对父本抽取两个切点，并聚焦于切点之间的位置，或者以二分之一的概率聚焦于切点之外的位置；凡是父本在聚焦区域内携带不同的具体比特之处，两个子代都会得到一个 $\#$。因此子代能够匹配其父本所匹配的一切，甚至更多。
* 每个子代获得其父本强度的平均值，并通过**拥挤**取代它最相似的潜在淘汰对象：抽取 $N = 8$ 次（消费规则为 4 次），每次抽取潜在淘汰对象的一半作为样本，取该样本中最弱的成员，按它与子代共享的位置数、加上若其动作不同则再加一分来评分，得分最高者将被淘汰。

论文的脚注 12 指出，更标准的 Holland 算法"最初曾被尝试过，但效果不佳，这促使我们设计出第 6 节所描述的改进算法"。

因此，我们在每个采用随机规则的经济体中都使用泛化交叉，并且由于论文表 7 未列出任何变异率，我们不使用变异。

在论文未明确说明之处，我们做出了以下选择：

* 拍卖中的平局随机打破；
* 对每种类型的交换系统和消费系统，泛化分别独立抽取；
* 新规则（包括子代）的计数器都从 1 开始，因此在它们首次经历之后，它们出生时所带的强度只能通过用它支付的出价得以延续；
* 创造所产生的新规则强度为零；
* 在遗传算法之外，“弱”指的是 `weakest` 中的得分；
* 若在某一期启用特殊化，则它适用于该期的每一次拍卖；
* 如上文所述，每个主体自身的决策构成一条链式交易链（bucket-brigade chain）；
* 经济体 D 中的商品采用五位独热编码。

```{code-cell} ipython3
@njit
def roulette(w, rng):
    "Draw an index with probability proportional to the nonnegative weights w."
    x = rng.random() * w.sum()
    for k in range(w.shape[0]):
        x -= w[k]
        if x <= 0:
            return k
    return w.shape[0] - 1


@njit
def generalize(R, i, n_goods, rng, stamp, n_crowd,
               p1=0.2, p2=0.7, p3=0.2, p4=0.5, S_bar=0.0):
    "One call of the paper's genetic algorithm on the classifier system of type i."
    cond, action, S, used = R.cond[i], R.action[i], R.strength[i], R.used[i]
    n, L = cond.shape

    # potential exterminants: weak *and* seldom used
    doomed = [r for r in range(n) if S[r] < S_bar and used[r] < p3 * used.max()]
    if len(doomed) == 0:
        return

    # potential parents: p2 * n rules drawn without replacement, weighted by use
    w = used.astype(np.float64)
    parents = np.zeros(int(p2 * n), np.int64)
    for k in range(parents.shape[0]):
        parents[k] = roulette(w, rng)
        w[parents[k]] = 0.0
    fitness = S[parents] - S[parents].min() + 1e-6

    for _ in range(max(1, int(p1 * n) // 2)):
        mum = parents[roulette(fitness, rng)]
        dad = parents[roulette(fitness, rng)]
        kids = (cond[mum].copy(), cond[dad].copy())

        # generalizing crossover
        cuts = np.sort(rng.integers(0, L + 1, size=2))
        inside = rng.random() < 0.5
        for j in range(L):
            if ((cuts[0] <= j < cuts[1]) == inside and kids[0][j] != WILD
                    and kids[1][j] != WILD and kids[0][j] != kids[1][j]):
                kids[0][j] = WILD
                kids[1][j] = WILD
        strength = 0.5 * (S[mum] + S[dad])

        # crowding: each child replaces the potential exterminant it most resembles
        for kid, act in ((kids[0], action[mum]), (kids[1], action[dad])):
            if len(doomed) == 0:
                return
            size = max(1, int(p4 * len(doomed)))
            victim, best = doomed[0], -1
            for _ in range(n_crowd):
                sample = rng.permutation(len(doomed))[:size]
                cand = doomed[sample[0]]
                for q in sample:
                    if S[doomed[q]] < S[cand]:
                        cand = doomed[q]
                likeness = np.sum(kid == cond[cand]) + (act != action[cand])
                if likeness > best:
                    victim, best = cand, likeness
            set_rule(R, i, victim, kid, act, strength, n_goods, stamp)
            doomed.remove(victim)
```

## 经济体 A1.2：从随机规则中学习

经济体 A1.2 的参数与 A1.1 相同，但主体从 1458 种和 54 种可能的字符串中随机抽取 72 条交换规则和 12 条消费规则开始。

这些规则中的大多数都是无意义的，也不能保证种群中包含支持某个均衡所需的规则。

算子必须制造出这些规则。

```{code-cell} ipython3
economy_a12 = replace(economy_a11, name='A1.2', random_rules=True)

sim_a12 = Simulation(economy_a12, seed=0)
sim_a12.run(2000)

paper_a12 = np.array([[0.0, 0.98, 0.02],        # the paper's Table 7b, t = 2000
                      [0.318, 0.0, 0.682],
                      [1.0, 0.0, 0.0]])
print("holdings at t = 1000\n", holdings(sim_a12, t=1000), "\n")
print("holdings at t = 2000\n", holdings(sim_a12), "\n")
print("paper's table at t = 2000\n", paper_a12, "\n")
print(f"max |paper's table - fundamental equilibrium| = "
      f"{distance(paper_a12, fundamental):.3f}")
```

下面是其时间路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Holdings in Economy A1.2"
    name: fig-mms-a12-holdings
---
plot_holdings(sim_a12)
```

基本模式再次出现，尽管不如完全枚举时那样清晰。

类型 1 主要持有商品 2，类型 2 只持有商品 1 和商品 3，类型 3 主要持有商品 1。

但类型 1 和类型 3 也持有基本均衡所排除的商品：在这次运行中，类型 1 大约有 5% 的时间持有商品 3，类型 3 大约有五分之一的时间持有商品 2。

论文的运行结果对类型 1 和类型 3 来说更为干净，但对类型 2 却并非如此：在 $t = 2000$ 时，其类型 2 主体只有 32% 的时间持有商品 1，而其表 7e 显示，类型 2 在 $t = 1000$ 时以 0.45 的概率吃掉商品 1（即其货币），在 $t = 2000$ 时以 0.23 的概率吃掉商品 1。

论文称这种过度消费是“我们若干经济体中的一个小问题”，与分类器系统访问太少以至于无法学习的状态有关。

以下是我们运行的最后 200 期中的进食频率。

```{code-cell} ipython3
eating_rates(sim_a12)
```

在这次运行中，类型 2 几乎从不吃掉自己的货币，这就是为什么其持有份额接近均衡的均等分配。

类型 1 和类型 3 大约有一半的时间吃掉“错误”的商品，但这对它们没有任何损失，因为这种商品恰好是它们自己的生产商品（对类型 1 来说是商品 2，对类型 3 来说是商品 1）。

但类型 1 大约有一半的时间获得商品 3 后会将其携带到下一期，代价是 20 而不是 1，类型 3 大约有一半的时间携带商品 2：这些正是论文所归咎的那些鲜少被访问的状态。

跨种子的结果：

```{code-cell} ipython3
sims_a12 = across_seeds(economy_a12, 2000)
H_a12 = final_holdings(sims_a12)
eat_money = np.array([eating_rates(sim).iloc[1, 0] for sim in sims_a12])
n = len(H_a12)
print(f"type 1 holds good 2 at least 90% of the time: {np.sum(H_a12[:, 0, 1] >= 0.9)} of {n}")
print(f"type 3 holds good 1 at least 90% of the time: {np.sum(H_a12[:, 2, 0] >= 0.9)} of {n}")
print(f"type 2 holds only goods 1 and 3:              {np.sum(H_a12[:, 1, 1] < 0.05)} of {n}")
print(f"type 2's share of good 1: median {np.median(H_a12[:, 1, 0]):.2f}, "
      f"range {H_a12[:, 1, 0].min():.2f} to {H_a12[:, 1, 0].max():.2f}")
print(f"within 0.1 of the fundamental equilibrium:    "
      f"{np.sum(distance(H_a12, fundamental) < 0.1)} of {n}")
print(f"type 2 eats good 1 at least 40% of the time:  {np.sum(eat_money >= 0.4)} of {n}")
```

种子 0 是较好的运行结果之一。

在每次运行中，类型 2 都只持有商品 1 和商品 3，而在大多数运行中，类型 1 和类型 3 至少有 90% 的时间持有其均衡商品。

但类型 2 持有商品 1 份额的中位数只有 0.19，大约一半的运行中，类型 2 至少有 40% 的时间吃掉商品 1，这正是论文运行中出现的过度消费现象。

只有四分之一的运行最终结果与基本均衡的差距在 0.1 以内；论文自己的运行结果则相差 0.18。

通用规则在遗传算法中存活了下来。

```{code-cell} ipython3
share = lambda R: np.mean(R.cond == WILD)
print(f"share of # in exchange rules:    {share(sim_a12.E):.2f}")
print(f"share of # in consumption rules: {share(sim_a12.C):.2f}")
```

一个随机字符串在三分之一的位置上含有 $\#$，而泛化交叉操作不断增加这个比例，专化操作则缓慢地将其减少。

## 经济体 C：法定货币

现在添加第四种物品——法定货币，论文称之为商品 0。

它

* 储藏无成本，$s_0 = 0$；
* 对任何人都不产生效用；且
* 不能被消费。

它本质上是毫无价值的。

它是通过在 $t = 0$ 时将 48 个单位交给 48 个随机选出的主体来引入的，商品的储藏成本被提高到 $s = (9, 14, 29)$，使得没有一种商品的储藏成本能接近货币那么低。

论文的脚注 17 报告说，若采用经济体 A 的储藏成本，该版本并未收敛，“可能是因为商品 1 的低储藏成本（0.1）使其成为法定货币的过强替代品”。

每个主体各自从 150 条随机交换规则和 20 条随机消费规则开始，而这些规则的可能总数分别为 13,122 条和 162 条。

```{code-cell} ipython3
economy_c = Economy(
    name='C',
    produces=np.array([1, 2, 0]),
    storage_costs=np.array([9.0, 14.0, 29.0, 0.0]),    # fiat money is the last good
    u=100.0,
    random_rules=True,
    n_trade_rules=150,
    n_consume_rules=20,
    b_consume=(0.025, 0.25),
    n_fiat=48,
)

sim_c = Simulation(economy_c, seed=0)
sim_c.run(2000)

fundamental_c = np.array([[0, 0.74, 0, 0.26],       # the paper's Table 14b
                          [0.26, 0, 0.42, 0.32],
                          [0.62, 0, 0, 0.38]])
paper_c = np.array([[0, 0.54, 0, 0.46],             # the paper's Table 15b, t = 1250
                    [0.18, 0, 0.53, 0.28],
                    [0.77, 0.01, 0, 0.21]])
print("holdings at t = 750\n", holdings(sim_c, t=750), "\n")
print("holdings at t = 1250\n", holdings(sim_c, t=1250), "\n")
h = holdings(sim_c, t=1250).to_numpy()
print(f"max |simulated - fundamental equilibrium| = {distance(h, fundamental_c):.3f}")
print(f"max |simulated - paper's table|           = {distance(h, paper_c):.3f}")
```

每种类型大约有 30% 的时间持有法定货币，这接近理论上的基本均衡（论文的表 14b）；而论文自身的运行结果（其表 15b）中，类型 1 持有法定货币的时间更多，达到 46%。

与理论值偏差最大的是类型 2 对商品 1 的持有：在我们的运行中约为 0.10，而理论值为 0.26，论文运行结果为 0.18——这说明我们的类型 2 主体相较于理论更多地依赖法定货币而非商品 1 作为交换媒介。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Holdings in Economy C"
    name: fig-mms-c-holdings
---
plot_holdings(sim_c)
```

交换模式显示出货币在流通易手。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Exchange pattern discovered in Economy C"
    name: fig-mms-c-flows
---
plot_flows(sim_c)
```

进出法定货币节点的箭头显示了每种类型都在双向进行货币的转手：主体放弃自己的生产品以获取货币，也放弃货币以获取他们所消费的商品。

商品 1 仍然充当类型 2 的第二种货币——类型 2 用商品 3 换取商品 1，再用商品 1 换取商品 2。

经济体 A1.1 中那种与收益无关的交易再次出现：类型 1 用商品 2 换取商品 3，类型 3 用商品 1 换取商品 2。

在不同的随机种子下：

```{code-cell} ipython3
H_c = final_holdings(across_seeds(economy_c, 1250))
n = len(H_c)
print(f"every type holds fiat at least 10% of the time: "
      f"{np.sum(H_c[:, :, 3].min(axis=1) >= 0.1)} of {n}")
print("holdings averaged over runs:")
print(pd.DataFrame(H_c.mean(axis=0), index=type_names(economy_c),
                   columns=good_names(economy_c)).round(2))
print(f"within 0.1 of the fundamental equilibrium:      "
      f"{np.sum(distance(H_c, fundamental_c) < 0.1)} of {n}")
print(f"within 0.2 of the fundamental equilibrium:      "
      f"{np.sum(distance(H_c, fundamental_c) < 0.2)} of {n}")
print("holdings in runs where some type holds fiat less than 10% of the time:")
print(H_c[H_c[:, :, 3].min(axis=1) < 0.1].round(2))
```

在 20 次运行中的 18 次里，每种类型持有法定货币的时间都至少占 10%。

在另外两次运行中的一次里，类型 1 主体囤积法定货币，而类型 2 和类型 3 几乎不持有；在另一次中，类型 2 持有法定货币的比例刚好略低于 10%。

多次运行的平均持有量与理论均衡相差约 0.15：类型 1 持有的法定货币比理论值更多，持有的商品 2 更少；类型 2 持有的商品 1 更少，商品 3 更多，这与种子 0 的情形一致。

正是这些差距导致只有 2 次运行落在与均衡相差 0.1 以内，而有 14 次落在相差 0.2 以内。

环境中没有任何东西告诉主体要这样做。

每个主体只是发现，最终持有那种无成本物品的规则比不这样做的规则赚得更多；而由于所有人都在同时发现这一点，这种信念变得自我实现。

这一发现对主体要求甚微：法定货币是储藏成本最低的物品，因此即便是没有耐心的主体，只要别人愿意轮流接受它，也会接受它。

## 经济体 D：五种商品，五种类型

最后一个经济体有五种类型和五种商品，生产模式为

| 类型 $i$ | 生产 | 消费 |
|---|---|---|
| 1 | 商品 3 | 商品 1 |
| 2 | 商品 4 | 商品 2 |
| 3 | 商品 5 | 商品 3 |
| 4 | 商品 1 | 商品 4 |
| 5 | 商品 2 | 商品 5 |

储藏成本 $s = (1, 4, 9, 20, 30)$（取自论文表 4），$u_i = 200$。

每个主体携带 180 条交换规则和 20 条消费规则。

作者强调，他们对这个经济体*没有解析的均衡刻画*，而是利用模拟来揭示均衡可能是什么样子。

```{code-cell} ipython3
economy_d = Economy(
    name='D',
    produces=np.array([2, 3, 4, 0, 1]),         # type 1 -> good 3, type 2 -> good 4, ...
    storage_costs=np.array([1.0, 4.0, 9.0, 20.0, 30.0]),
    u=200.0,
    random_rules=True,
    n_trade_rules=180,
    n_consume_rules=20,
)

sim_d = Simulation(economy_d, seed=0)
sim_d.run(2000)
holdings(sim_d)
```

时间路径显示了持有量是如何演变到这一步的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Holdings in Economy D"
    name: fig-mms-d-holdings
---
plot_holdings(sim_d)
```

有两个特征引人注目。

第一，对角线为零：没有一种类型会在一期结束时持有它所消费的商品，因为它把它吃掉了。

第二，每种类型大部分时间持有的都是自己生产的商品；在这次运行结束时，只有类型 5 在相当多的时间里持有其他商品，而类型 1 大约有十分之一的时间持有更便宜的商品 2。

时间路径尚未完全稳定下来；特别是类型 2 的持有量在整个运行过程中大部分时间都在商品 4 和商品 1 之间摆动。

论文自己的运行在细节上有所不同：在 $t = 1750$ 时，其类型 1 主体有 83% 的时间持有更便宜的商品 2，而其类型 3 主体则将持有量分配在商品 2 和商品 5 之间。

让我们按主体获取的商品类型，对实际发生的交易进行分类。

```{code-cell} ipython3
def trade_composition(sim, window=200):
    """
    Shares of realized trades in which the agent acquires its own consumption
    good, a good cheaper to store than the one given up, or neither; and the
    share of 'neither' trades that give up the agent's own production good
    and are then eaten (payoff-neutral).
    """
    econ = sim.econ
    f = np.array(sim.exch_hist)[-window:].sum(axis=0)
    c = np.array(sim.cons_hist[-window:]).sum(axis=0)
    eat = c[:, :, 1] / np.maximum(c[:, :, 0], 1)
    s = econ.storage_costs
    out = {'own consumption good': 0.0, 'a cheaper good': 0.0, 'neither': 0.0}
    neutral = 0.0
    for i in range(econ.n_types):
        for j in range(econ.n_goods):
            for k in range(econ.n_goods):
                if j == k:
                    continue
                key = ('own consumption good' if k == i else
                       'a cheaper good' if s[k] < s[j] else 'neither')
                out[key] += f[i, j, k]
                if key == 'neither' and j == econ.produces[i]:
                    neutral += f[i, j, k] * eat[i, k]
    total = sum(out.values())
    out = {k: v / total for k, v in out.items()}
    out['  of which payoff-neutral'] = neutral / total
    return pd.Series(out, name='share of realized trades').round(3)


trade_composition(sim_d)
```

大约三分之二的实际交易获取的要么是主体自己的消费商品，要么是储藏成本更便宜的商品，这与论文所描述的模式一致。

剩下三分之一中约有一半是收益中性的：主体放弃自己的生产商品换取更贵的商品，把它吃掉，然后再次生产自己的商品。

其余部分，约占全部交易的六分之一，与这两种描述都不相符。

下面是十个种子对应的相同统计数据，以及每种类型的主体每期吃掉自己商品的次数。

```{code-cell} ipython3
sims_d = across_seeds(economy_d, 2000, seeds=range(10))
comp = pd.DataFrame([trade_composition(sim) for sim in sims_d])
print(comp.describe().loc[['mean', 'min', 'max']].round(2), "\n")
own = np.array([[np.array(sim.cons_hist[-200:]).sum(axis=0)[i, i, 1] / (200 * 50)
                 for i in range(5)] for sim in sims_d])
print("own-good consumption per agent per period, smallest over runs:",
      own.min(axis=0).round(3))
```

这些份额在不同运行之间变化很小，并且在每次运行中，每种类型都会吃掉自己的商品。

论文对其自身运行的总结是：

> 从模拟结果可以看到，交易模式几乎描绘出一个基本均衡的样子，主体只愿意为储藏成本低于当前所储藏商品的商品进行交易，除非它们总是接受本类型的商品。可以检测到一些投机性的举动。

我们的运行结果与这一描述的前半部分相符，但有两点需要说明：三分之一的交易与该模式都不相符，其中一半是因为“消费任何商品”规则使它们成为收益中性的交易，并且在我们展示的运行中，持有量尚未稳定下来。

论文中关于投机性举动的例子——类型 2 主体从类型 1 处接受商品 3 以换取商品 1，以便再将其与类型 3 交换商品 2——来自论文自身的运行。

模拟所给出的是一种候选的交易模式，正如作者所指出的，它仍然需要用均衡条件来加以检验。

## 结束语

Marimon、McGrattan 和 Sargent 的主体知道得很少。

他们不知道自己的效用函数，不知道储藏成本，不知道商品在人群中的分布，当然也不求解动态规划问题。

他们只是在体验效用时认出它，在承担成本时认出它，并保留滚动平均值。

从中产生了以下结果。

* **交换媒介是可学习的。** 在规则列表完备的情况下，经济体 A1 在每次运行中都能达到其唯一的平稳均衡；而从随机规则出发，主体们也能大致接近这一结果，像论文中的主体一样过度持有货币。
* **选择及其局限。** 在存在两个均衡的经济体 B 中，分类器系统在大多数运行中选择了基本均衡，而从未选择投机均衡。
  在经济体 A2 中，对于耐心的主体来说唯一的均衡是投机性的，但它们从未投机：大多数运行会收敛——有时非常缓慢——到一个并非该经济体均衡的结果，还有少数运行则锁定在交易更少的模式中。
* **原因：主体是没有耐心的。** 在稳定强度下，桶队算法对未来的贴现因子为 $b_1 b_2/((1+b_1)(1+b_2))$，按论文中的出价约为 $0.016$，而无论出价如何该值都低于 $1/4$，主体所找到的模式正是没有耐心的主体所会表现出的模式。
  贪婪拍卖机制和过度泛化的消费规则加剧了这一效应。
* **制度是可以被发现的。** 在几乎每次运行中，经济体 C 的主体仅凭他们自己对储藏成本的体验，就建立起了一套法定货币体系。
* **该方法提出了候选方案。** 经济体 D 在其作者尚未求解的模型中产生了一种交易模式，但尚未被验证为一个均衡。

论文对缺失之处很坦诚。

没有收敛定理，只有对随机逼近论证如何能够提供收敛定理的一个概述，作者判断自己的算法提供的“实验太少”。

实验固然是故事的一部分，但稳定强度表明，即便增加实验也未必足够：以这种方式构建的主体，无论其出价如何，对提前一期收益的估值都不到其真实价值的四分之一。

{doc}`prospects_bounded_rationality` 对该系列成果做了总结，并将经济体 A2 列入了账本的负债一方。

## 练习

```{exercise-start}
:label: mms_ex1
```

经济体 A2.2 就是经济体 A2——$u_i = 500$——但从随机生成的规则开始，并用遗传算法进行演化，正如 A1.2 之于 A1.1 的关系一样。

论文的总结表将其均衡类型列为投机型，作者报告说在 1000 次迭代之后经济体尚未收敛，在第 1000 期时交易模式更接近基本均衡而非投机均衡——“（在第 500 期情况正相反。）”

将其模拟 2000 期，并报告 $t = 500$、$t = 1000$ 和 $t = 2000$ 时的持有量。

然后对 20 个种子重复此实验。

遗传算法提供的额外实验在这些时点中是否产生了投机？

```{exercise-end}
```

```{solution-start} mms_ex1
:class: dropdown
```

```{code-cell} ipython3
economy_a22 = replace(economy_a21, name='A2.2', random_rules=True)

sim_a22 = Simulation(economy_a22, seed=0)
sim_a22.run(2000)
for t in (500, 1000, 2000):
    print(f"t = {t}\n", holdings(sim_a22, t=t), "\n")
```

现在对 20 个种子进行相同的实验，记录每个时点的持有量。

```{code-cell} ipython3
sims_a22 = across_seeds(economy_a22, 2000)
for t in (500, 1000, 2000):
    H = final_holdings(sims_a22, t)
    closer = distance(H, speculative.to_numpy()) < distance(H, fundamental)
    print(f"t = {t}: type 1 holds good 3 > 5% in {np.sum(H[:, 0, 2] > 0.05)} runs "
          f"(at most {H[:, 0, 2].max():.2f} of the time); closer to speculative "
          f"than to fundamental in {np.sum(closer)} runs")
```

没有产生投机。

在这次运行中，类型 1 在任何一个时点都从未持有商品 3。

在 20 次运行中，在每个时点，类型 1 持有商品 3 超过 5% 的时间的情形最多出现在三次运行中，且从未达到投机均衡的 29%。

只有一次运行（在 $t = 2000$ 时）与投机均衡的持有量相比略微更接近，而且我们的运行结果没有重现论文中所报告的其运行在 $t = 500$ 时看起来更接近投机型的情况。

将初始规则随机化并加入遗传算法可以制造出接受商品 3 的规则，但这并不会改变连锁交易系统赋予持有该商品的价值所依据的贴现因子（见 {ref}`mms_impatience`）。

```{solution-end}
```

```{exercise-start}
:label: mms_ex2
```

经济体 B.2 是从随机规则出发、使用遗传算法并采用 A 的出价规则的经济体 B。

论文报告说它“在 2000 期后仍未收敛”，但正“朝着基本均衡移动”，在 $t = 2000$ 时的持有量为：类型 1 为 $(0, 0.354, 0.646)$，类型 2 为 $(0.996, 0, 0.004)$，类型 3 为 $(0.268, 0.732, 0)$。

对其进行模拟，将其持有量与经济体 B 的两种均衡以及论文的运行结果进行比较，并在 20 个种子上比较它与基本均衡的距离和 B.1 的相应距离。

```{exercise-end}
```

```{solution-start} mms_ex2
:class: dropdown
```

```{code-cell} ipython3
economy_b2 = replace(economy_b1, name='B.2', random_rules=True,
                     b_trade=(0.025, 0.025))
sim_b2 = Simulation(economy_b2, seed=0)
sim_b2.run(2000)

paper_b2 = np.array([[0, 0.354, 0.646], [0.996, 0, 0.004], [0.268, 0.732, 0]])
rows = {'B.2 simulated, t = 2000': holdings(sim_b2).to_numpy(),
        'fundamental equilibrium': fundamental_b,
        'speculative equilibrium': speculative_b,
        "paper's B.2, t = 2000": paper_b2}
pd.concat({k: pd.DataFrame(v, index=type_names(economy_b2),
                           columns=good_names(economy_b2)).round(3)
           for k, v in rows.items()})
```

现在将 B.2 与 B.1 在多个种子上进行比较。

```{code-cell} ipython3
H_b2 = final_holdings(across_seeds(economy_b2, 2000))
for name, H in [("B.1", H_b1), ("B.2", H_b2)]:
    print(f"{name}: median distance to fundamental "
          f"{np.median(distance(H, fundamental_b)):.2f}, to speculative "
          f"{np.median(distance(H, speculative_b)):.2f}; "
          f"median share of good 1 for type 3 {np.median(H[:, 2, 0]):.2f}")
```

B.2 的这次运行既不接近基本均衡也不接近投机均衡：类型 3 持有商品 1 的时间仅占 11%（基本均衡中为 59%，论文的运行中为 27%），类型 1 持有商品 2 的时间仅占 10%（基本均衡中为 29%，投机均衡中为 59%）。

在多个种子上，B.2 距基本均衡的中位距离为 0.25，而 B.1 为 0.06，两者与投机均衡的距离都保持在约 0.5 左右。

B.2 中典型的类型 3 主体大约有三分之一的时间持有商品 1，介于论文的 27% 和均衡值 59% 之间。

与论文的运行结果类似，B.2 最终远比投机均衡更接近基本均衡，但在 2000 期内未能达到该均衡：从随机材料中构建所需的规则要花费时间，而枚举经济体从不需要花费这些时间。

```{solution-end}
```

```{exercise-start}
:label: mms_ex3
```

清泷（Kiyotaki）和赖特（Wright）的主体不能消费他们并不看重的商品。

在经济体 A2.1 上通过设置 `no_eat=True` 施加这一限制，这会移除 {ref}`mms_a2` 中的过度消费机制。

在 20 个种子上，类型 1 持有商品 3 的频率有多高？

在它确实持有商品 3 的那些运行中，类型 1 在状态（商品 2，商品 3）下获胜的是哪条规则，这条规则还涵盖了哪些其他情形？

```{exercise-end}
```

```{solution-start} mms_ex3
:class: dropdown
```

```{code-cell} ipython3
economy_noeat = replace(economy_a21, name='A2.1, no eating', no_eat=True)

rows = []
for seed in range(20):
    sim = Simulation(economy_noeat, seed)
    sim.run(1000)
    h = holdings(sim, window=100).to_numpy()
    w = strongest(sim.E, 0, 1 * 3 + 2)
    f = trade_rates(sim)[0]
    rows.append({'seed': seed, 'type 1 holds good 3': h[0, 2],
                 'type 1 trades 2 for 3': f[1, 2], 'type 1 trades 3 for 1': f[2, 0],
                 'winning rule in (good 2, good 3)': rule_string(sim.E.cond[0, w],
                                                                 sim.E.action[0, w], 3)})
noeat = pd.DataFrame(rows).set_index('seed').round(3)
noeat[noeat['type 1 holds good 3'] > 0.05]
```

在 20 次运行中有 9 次，类型 1 持有商品 3 的时间超过 5%，但原因有两种不同的情形。

在四次运行中（种子 3、8、10 和 11），类型 1 用商品 2 换取商品 3，再用商品 3 换取商品 1，其持有量看起来类似于投机均衡。

但接受商品 3 的规则是一条泛化规则，即“如果我的交易对手不持有商品 2 则交易”这一规则的变体，它也涵盖了对手持有商品 1 的情形。

它的强度是在后一种情形下获得的，在该情形下交易会导致消费商品 1，而这一强度被转嫁到商品 3 上：这是由于泛化性而产生的投机，而不是因连锁交易系统认可其所要求的耐心而产生的投机。

在另外五次运行中，类型 1 虽然持有商品 3，但从未进行交易。

这些主体在早期就获得了商品 3，无法消费它，并且遇到了已经锁定拒绝接受商品 3 的类型 3 主体。

因此，移除过度消费机制并不能使主体出于正确的原因进行投机，泛化性是一把双刃剑：它既可以把某种状态下获得的回报转移到另一种状态（在这里就模拟出了投机的表象），也会产生 {ref}`mms_a2` 中的过度消费现象。

这正是 {cite:t}`Sargent1993` 在其关于分类器系统的讨论中所强调的泛化性与判别性之间的权衡。

```{solution-end}
```

```{exercise-start}
:label: mms_ex4
```

将经济体 A2.1 中的 argmax 拍卖替换为强度 $\eta = 0.5$ 的软最大化（softmax）拍卖（设置 `eta=0.5`），这与 {ref}`gc_ex3` 以及论文脚注 5 的精神一致，其余设置保持不变。

在 20 个种子上，类型 1 在状态（商品 2，商品 3）下获胜的交换规则接受商品 3 的频率有多高，类型 1 实际持有商品 3 的频率又有多高？

将你的结果与论文表 9d 进行比较。

```{exercise-end}
```

```{solution-start} mms_ex4
:class: dropdown
```

```{code-cell} ipython3
economy_soft = replace(economy_a21, name='A2.1, softmax', eta=0.5)

rows = []
for seed in range(20):
    sim = Simulation(economy_soft, seed)
    sim.run(1000)
    w, c = strongest(sim.E, 0, 1 * 3 + 2), strongest(sim.C, 0, 2)
    rows.append({'accepts good 3': sim.E.action[0, w] == 1,
                 'type 1 holds good 3': holdings(sim, window=100).iloc[0, 2],
                 'consumption rule for good 3': rule_string(sim.C.cond[0, c],
                                                            sim.C.action[0, c], 3)})
soft = pd.DataFrame(rows)
print(f"winning rule accepts good 3: {soft['accepts good 3'].sum()} of 20")
print(f"largest share of time type 1 holds good 3: {soft['type 1 holds good 3'].max():.3f}")
soft['consumption rule for good 3'][soft['accepts good 3']].value_counts()
```

在有探索的情况下，接受商品 3 的规则在 20 次运行中赢得了 14 次，但类型 1 几乎从未持有商品 3。

在每一次接受商品 3 获胜的运行中，针对商品 3 获胜的消费规则都是一条泛化的消费规则：要么是“如果我不持有商品 2 则消费”，这也涵盖了商品 1 的情形；要么是“如果我不持有商品 1 则消费”，这也涵盖了商品 2 的情形——在这些情形下消费并不会造成损害。

这正是论文表 9d 所体现的模式：投机型交换规则获胜，而泛化的消费规则立即消费掉商品 3。

软最大化拍卖持续保持探索，因此不会有任何规则因早期的意外收获而被永久排除，但它保留了另外两种机制。

```{solution-end}
```

```{exercise-start}
:label: mms_ex5
```

{ref}`mms_impatience` 预测，如果一个类型 1 主体的所有规则都是完全特定的，那么除非 $u_1$ 超过约 $6(s_3 - s_2)/\beta_{\rm eff}$，否则它在经济体 A2 中不会进行投机。

通过模拟来检验这一预测。

只给主体完全特定的规则（`specific_only=True`），让它们通过软最大化拍卖（$\eta = 0.5$）进行探索，禁止消费不想要的商品，并改变 $u_1$ 的值。

对每个 $u_1$ 值使用六个种子，记录类型 1 在状态（商品 2，商品 3）下获胜的规则是否接受商品 3。

在什么样的 $u_1$ 值下投机开始获胜？

```{exercise-end}
```

```{solution-start} mms_ex5
:class: dropdown
```

```{code-cell} ipython3
print(f"predicted thresholds: {speculation_threshold(s_A, 1, 0.5, β_A):,.0f} "
      f"(fundamental holdings), {speculation_threshold(s_A, 1, 0.586, β_A):,.0f} "
      f"(speculative holdings)\n")

for u in (500, 4_000, 8_000, 12_000, 20_000, 40_000):
    econ = replace(economy_a21, u=float(u), specific_only=True, eta=0.5,
                   no_eat=True)
    accept, share = 0, []
    for seed in range(6):
        sim = Simulation(econ, seed)
        sim.run(2000)
        accept += sim.E.action[0, strongest(sim.E, 0, 1 * 3 + 2)] == 1
        share.append(holdings(sim, window=200).iloc[0, 2])
    print(f"u_1 = {u:>6,}: accepts good 3 in {accept} of 6 runs; "
          f"type 1 holds good 3 {np.mean(share):.2f} of the time")
```

在这六次运行中，在 A2 的 $u_1 = 500$ 以及一直到 12,000 的各个水平上，接受商品 3 的规则从未获胜。

它在 12,000 到 20,000 之间开始获胜，在 40,000 时类型 1 持有商品 3 的频率已大致接近投机均衡中的水平。

这一预测在数量级上是正确的：投机所需的效用要比 A2 的效用大几十倍，而耐心主体的阈值为 138。

这六次运行所需的效用略高于预测值，这很可能是因为在有限长度的运行中，拍卖所比较的强度带有噪声；在其他种子上，接受规则在 12,000 时就已偶尔获胜。

```{solution-end}
```
