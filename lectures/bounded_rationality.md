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
  title: 一种奇特的有限理性定义
  headings:
    Overview: 概览
    Overview::The knowledge that rational expectations imputes: 理性预期赋予人们的知识
    Overview::Sargent's formulation of bounded rationality: 萨金特对有限理性的表述
    Overview::What the program is good for: 该研究纲领的用处
    Overview::The rest of the series: 本系列的其余部分
    Overview::Notation across the series: 本系列中的符号说明
    Rational expectations as a fixed point: 理性预期作为不动点
    Rational expectations as a fixed point::A static market: 一个静态市场
    Rational expectations as a fixed point::Computing the fixed point: 计算不动点
    Rational expectations as a fixed point::Adaptive expectations: 适应性预期
    Rational expectations as a fixed point::Muth's inverse problem: 穆思的逆问题
    Rational expectations as a fixed point::The dynamic analogue: 动态类比
    One recursion, many learners: 一个递归，多种学习者
    'One recursion, many learners::From relaxation to learning: Bray''s market': 从松弛到学习：布雷市场
    One recursion, many learners::Stochastic approximation and E-stability: 随机逼近与 E-稳定性
    One recursion, many learners::How the series uses it: 该系列如何运用这一理论
    Money and prices: 货币与价格
    Money and prices::Equilibrium: 均衡
    Money and prices::The bubble: 泡沫
    Two currencies: 两种货币
    Where this leaves us: 我们的处境
    Exercises: 练习
---

(bounded_rationality)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 一种奇特的有限理性定义

```{index} single: Bounded Rationality
```

```{contents} Contents
:depth: 2
```

## 概览

本讲开启了一个系列，该系列基于 {cite:t}`Sargent1993` 关于宏观经济学中有限理性的著作，反映了20世纪80年代末和90年代初的一种观点。

那些年，苏联解体，原华沙条约国家正在重新安排它们的政治经济体制。

对这些国家来说，那确实是现代动态宏观经济学技术意义上的*体制转变*。

正如 {ref}`br_payoffs` 所解释的那样，理性预期模型处理这类变化时，其舒适程度往往不如使用者通常所设想的那样。

在博弈论、宏观经济学和一般均衡理论方面，两代关于经济动态的研究已经产生了拥抱理性预期假设的理论，这些模型旨在理解人们面对他们已经经历过很多次的反复出现的情形的环境。

20世纪90年代初东欧正在进行的体制转变并非如此。

那里的人们面对的是前所未有的机遇、新的且定义不清的规则，以及每天为弄清楚最终将支配贸易和生产的机制而进行的努力。

拥有良好市场经济模型的经济学家们拥有大量*均衡理论*，描述了一个体系一旦完全适应了一套新的、连贯的规则和预期后将如何运作。

但他们对从苏联体制向市场经济*转变*的动态知之甚少。

他们可能持有关于如何管理这种转变的偏见和轶事，但没有经过实证检验的正式理论。

在这种背景下，一些经济学家冒险进入了 {cite:t}`Sims1980` 所称的非理性预期和有限理性的"荒野"。

其目的一部分是构建转变动态的理论，一部分是理解均衡动态本身的性质，还有一部分是研究那些永远不会稳定下来的体系。

本系列跟随 {cite:t}`Sargent1993` 一起，深入到那片荒野之中。

(br_quest)=
### 理性预期赋予人们的知识

要看清有限理性*退却*的是什么，先从理性预期说起，它有 **两个** 要求：

1. **个体理性** — 每个人造主体的行为都在其感知约束条件下最大化一个目标函数。
1. **相互一致性** — 体系中每个人所感知的约束彼此一致，并且与他们的决策共同产生的结果一致。

第二个要求就是穆思的*理性预期*假设 {cite}`muth1961`。

在一个经济体中，一个人的决策是另一个人约束的一部分，因此一致性要求每个人对其他所有人的决策、决策过程和信念都持有正确的信念。

一致性也正是理性预期力量的来源：如果对感知没有任何限制，一个行为依赖于关于主观信念的任意假设的模型几乎可以产生任何结果。

但请看一看，一旦将模型应用于数据，这一要求赋予人们的东西是什么。

理性预期模型中的主体使用*均衡*概率分布来评估他们的欧拉方程。

而这些正是研究他们的计量经济学家仍在努力估计的分布。

换句话说，这些主体已经以某种方式解决了经济学家才做了一半的推断问题。

### 萨金特对有限理性的表述

萨金特的 **有限理性** 研究纲领保留了个体理性，并以一种特定的方式退却了相互一致性，这一方式的选择旨在消除上文所述模型中主体与研究他们的计量经济学家之间的不对称性。

萨金特版本的有限理性研究纲领是（{cite:t}`Sargent1993`，第1章）：

> 我将建立具有"有限理性"主体模型的提议解读为一种呼吁，即通过将理性主体从我们的模型环境中驱逐，代之以行为类似计量经济学家的"人工智能"主体，从而退却理性预期的第二部分（感知的相互一致性）。这些"计量经济学家"进行理论构建、估计和调适，以试图了解在理性预期下他们本已知道的概率分布。

这些主体被塑造得更像那些建立模型的人：他们收集数据，形成理论，进行估计，并进行调适。

```{note}
为什么说这"奇特"？

对创造这一术语的赫伯特·西蒙来说，有限理性意味着人们收集和处理信息的能力存在限制，因此他们会*满足化*而非最优化。

萨金特的主体以一种不同的方式受到限制：他们像计量经济学家一样进行计算，但他们并不知道均衡概率分布。

本书将消除主体与计量经济学家之间不对称性这一目标归功于西蒙，但西蒙并不认同这种实现方式。

在看到本书的手稿后，西蒙给萨金特写了一封信，表示反对这种表述，并建议萨金特不要将其称为"有限理性"。

西蒙特别不喜欢让模型内的主体表现得像计量经济学家，他认为这种假装是荒谬的。

理性预期与有限理性拥有共同的发源地：20世纪50年代，穆思和西蒙曾是卡内基理工学院（现卡内基梅隆大学）的同事与合著者 {cite}`Holt_Modigliani_Muth_Simon`。
```

萨金特的提议使模型建立者的工作变得更难，而不是更容易。

撤回对共同理解环境的假设意味着我们必须用某种东西来取代它，而可供选择的合理选项有很多（{cite:t}`Sargent1993`，第1章）：

> 这个领域之所以是荒野，是因为研究者一旦决定放弃均衡理论化所提供的约束，就要面对如此多的选择。对均衡理论化的承诺通过要求将人建模为在共同理解的环境中的最优决策者，为他做了许多选择。当我们撤回共同理解环境的假设时，我们必须用某种东西来代替它，而合理的可能性有如此之多。

(br_payoffs)=
### 该研究纲领的用处

{cite:t}`Sargent1993` 第2章列出了四种可能证明这一努力合理的收益。

1. **均衡选择。** 当一个模型存在多个均衡时，适应性主体可能只收敛到其中一部分，这就把学习变成了在理性预期均衡中进行选择的工具；而在均衡唯一的情形下，通过学习走向该均衡的主体则赋予其一种作为单纯假设时所缺乏的合理性。
1. **计算。** 学习算法同时也可以作为计算那些手工难以求解的均衡的算法，而为主体提供"大脑"的文献，也为经济学家提供了新的优化和估计方法。
1. **转变动态与体制变化。** 适应性动态带来了一种关于均衡外调整过程的理论的希望——东欧改革者不得不盲目应对的那种调整过程。
1. **新的动态来源。** 当理性预期做出了被数据所拒绝的明确预测时，适应性主体可以弥补这一缺失；本书的主要例子是金融学中的无交易定理，根据该定理，均衡中不会发生任何交易。

第三种收益回应了本书对标准做法的一种批评。

理性预期模型的一种常见用法是，在"改革前"政策规则下计算一个均衡，在"改革后"规则下计算另一个均衡，并将两者之差解读为改革的效果。

这种比较本身与理性预期是不一致的：每个均衡都假定政府将永远信守其规则，因此这种比较所预测的是，倘若政府做了主体们确信它绝不会做的事情，将会发生什么。

本书以德国和奥地利恶性通货膨胀的骤然结束为例说明，在这种情形下，这种简化方法所付出的代价似乎很小；同时也以关于大萧条起因的两个故事为例，这两个故事都只有在某个环节放弃理性预期假设时才能成立。

一种关于人们如何通过学习走入新体制的理论将填补这一空白。

正如本系列将会承认的那样，这一承诺是实现得最少的；选择和计算方面的收益则更为可靠。

本讲以最尖锐的形式提出了选择问题：具有**过多均衡**的模型。

当一个理性预期模型有一系列连续的均衡时，经济的物理描述加上均衡概念本身无法确定会发生什么。

必须有其他东西来做出选择，而关于人们如何摸索走向均衡的合理描述正是这个"其他东西"的自然候选者。

我们构建了后续讲座会反复回到的两个货币模型示例，两者都具有一系列连续的理性预期均衡：

* 一个关于货币和价格的数量理论模型，其中价格水平只能被确定到一个任意的**泡沫**项；以及
* 一个双货币版本，其中*汇率完全不受约束*。

在此过程中，我们建立了一些工具——均衡的不动点观点、松弛算法、适应性预期、穆思的逆最优预测问题，以及其相关微分方程决定适应性主体最终归宿的随机逼近递归式——后续讲座将利用这些工具驱逐理性主体，并代之以适应性主体。

### 本系列的其余部分

* {doc}`olg_adaptive_money` — 萨缪尔森世代交叠货币模型中的适应性家庭。
  - 最小二乘学习选择了理性预期动态所排斥的低通胀均衡，而实验室中的受试者也走向了同样的方向。
  - 一个学习菲利普斯曲线的政府作为本讲的结尾。
* {doc}`learning_approximation`— 当状态是连续的、对每一种情形都做出单独反应已不可能时，主体必须放弃什么，这引出了近似均衡以及学习算法与均衡计算算法之间的相似性。
* {doc}`exchange_rate_learning` — 本讲的双货币模型，配以适应性主体，其中学习仅通过使汇率依赖于历史来固定汇率，而一个遗传算法经济体反而产生了永不消失的波动。
* {doc}`genetic_classifier` — 来自霍兰德和联结主义者的候选"大脑"目录：感知器、联想记忆、遗传算法、分类器系统。
* {doc}`marimon_mcgrattan_sargent` — 分类器系统种群从零开始发现哪种商品将充当货币。
* {doc}`prospects_bounded_rationality` — 1993年对该研究纲领已取得成就和尚未取得成就的总结，以及关于此后三十年的后记。

这些讲座是按照建模者向主体提供多少信息的程度来排序的，这是一种*提示程度*上的梯度：

* 布雷的最小二乘学习者（见{ref}`下文 <br_bray>`）对其所处环境的一切都了然于胸，唯独不知道一个系数。
* {doc}`olg_adaptive_money` 中的罗宾斯-蒙罗主体知道自己的效用函数，但不知道他们所面对回报的分布。
* {doc}`learning_approximation` 中的主体还必须为所学内容选择一种函数形式。
* {doc}`exchange_rate_learning` 中的牛顿学习者知道其自身收益的梯度；而其遗传算法字符串，正如 {doc}`genetic_classifier` 和 {doc}`marimon_mcgrattan_sargent` 中的分类器系统一样，只能看到一个收益信号。

{doc}`prospects_bounded_rationality` 随后将这种"提示"计入了该研究纲领账本的借方一侧。

(br_notation)=
### 本系列中的符号说明

每一讲都遵循其各自来源的符号约定，因此某些符号在不同讲座中承担着不同的含义。

| 符号 | 本讲中 | 系列后续中 |
|---|---|---|
| $\lambda$ | 松弛参数（增益）；泡沫的增长率 | 投资组合份额（{doc}`exchange_rate_learning`）；交易决策（{doc}`marimon_mcgrattan_sargent`） |
| $\gamma$ | 成本的曲率；价格水平中货币的系数 | 学习增益（{doc}`olg_adaptive_money` 至 {doc}`genetic_classifier`）；消费决策（{doc}`marimon_mcgrattan_sargent`） |
| $\theta$ | 穆思的移动平均系数；一般性信念参数 | 菲利普斯曲线斜率（{doc}`olg_adaptive_money`）；近似系数（{doc}`learning_approximation`） |
| $w_1, w_2$ | 禀赋为 $2w_1, 2w_2$ | 禀赋为 $w_1, w_2$（{doc}`olg_adaptive_money` 至 {doc}`exchange_rate_learning`）；权重（{doc}`genetic_classifier`） |
| $\pi$ | — | 总通货膨胀率（{doc}`olg_adaptive_money`）；选择与持有概率（{doc}`genetic_classifier`、{doc}`marimon_mcgrattan_sargent`） |
| $s$ | — | 储蓄（{doc}`olg_adaptive_money` 至 {doc}`exchange_rate_learning`）；网络状态（{doc}`genetic_classifier`）；存储成本（{doc}`marimon_mcgrattan_sargent`） |
| $N$ | （企业数量为 $n$） | 世代规模（{doc}`olg_adaptive_money`、{doc}`exchange_rate_learning`）；种群或模式规模（{doc}`exchange_rate_learning`、{doc}`genetic_classifier`） |

从 {doc}`olg_adaptive_money` 开始，$R$ 专用于总回报，$M$ 专用于估计的矩阵或曲率矩阵。

*适应性预期* 指的是下文中凯根和弗里德曼的几何分布滞后，而*适应性主体*则指根据经验修正任何规则的主体——无论是预测、回归系数，还是经验法则的强度。

此处学习增益记作 $\lambda_t$，因为 $\gamma$ 已被占用，而从 {doc}`olg_adaptive_money` 起则改记为 $\gamma_t$。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
```

## 理性预期作为不动点


### 一个静态市场

考虑一个由大量$n$家相同企业组成的竞争性行业。

每家企业选择产量$x$以最大化

$$
\Pi(x, p) = p x - c(x),
$$

其中$c$是一个递增的凸成本函数，价格由一条向下倾斜的逆需求曲线决定

$$
p = p(nX)
$$

其中$X$是*平均*企业的产量。

每家企业都是价格接受者和$X$接受者：它将$p$视为给定，并使边际成本等于价格，$p = c'(x)$。

将解写作$x = g(p)$。

代入需求曲线得到**最优反应映射**

```{math}
:label: br_map

x = g(p(nX)) \equiv h(X),
```

它将假设的行业平均值$X$映射为一个优化企业相对于该假设值所选择的个体产量。

这是理性预期的第一个组成部分，也是个体理性所能提供的全部内容。

第二个组成部分——一致性——要求每个企业的选择与其假设的平均企业的选择相一致：

```{math}
:label: static_ree

X = h(X).
```

一个静态理性预期均衡就是**最优反应映射的一个不动点**。

举一个具体的例子，取一个二次成本函数$c(x) = \tfrac{\gamma}{2} x^2$和一条线性逆需求曲线$p = a - b n X$。

那么$p = c'(x) = \gamma x$给出$x = p / \gamma$，因此

$$
h(X) = \frac{a - b n X}{\gamma},
\qquad\text{不动点为}\qquad
X^* = \frac{a}{\gamma + b n}.
$$

```{code-cell} ipython3
a, b, n, γ = 10.0, 0.3, 5, 1.0

def h(X):
    "Best response of an individual firm to an industry average X."
    return (a - b * n * X) / γ

X_star = a / (γ + b * n)
print(f"X*     = {X_star}")
print(f"h(X*)  = {h(X_star)}")
print(f"h'(X)  = {-b * n / γ}")
```

### 计算不动点

现在假设我们想在不求出闭式解的情况下*找到*$X^*$。

一个自然的出发点是**松弛算法**：保留对均衡的估计$X^*_k$，计算对它的最优反应，然后朝该最优反应移动一部分距离，

```{math}
:label: relaxation

X^*_k = X^*_{k-1} + \lambda\bigl(h(X^*_{k-1}) - X^*_{k-1}\bigr),
```

其中$\lambda \in (0, 1]$是一个**松弛参数**。

当$\lambda = 1$时，这就是对$h$的简单迭代，即经典的蛛网模型。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Relaxation algorithm paths: the cobweb and three smaller relaxation parameters"
    name: fig-br-relaxation
---
def relax(λ, X0=1.0, n_iter=12):
    "Iterate the relaxation algorithm, returning the whole path."
    path = np.empty(n_iter + 1)
    path[0] = X0
    for k in range(n_iter):
        path[k + 1] = path[k] + λ * (h(path[k]) - path[k])
    return path


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(relax(1.0), 'o-', ms=4, lw=1, color='C0',
             label=r"$\lambda = 1.0$ (cobweb)")
axes[0].set_title(r"$\lambda = 1$: the cobweb")
axes[0].set_ylabel("$X^*_k$")

for i, λ in enumerate((0.8, 0.5, 0.3)):
    axes[1].plot(relax(λ), 'o-', ms=4, lw=1, color=f'C{i+1}',
                 label=fr"$\lambda = {λ}$")
axes[1].set_title(r"$\lambda < 1$: relaxation")
axes[1].set_ylim(-1, 9)

for ax in axes:
    ax.axhline(X_star, color='k', lw=0.8, ls='--', label="$X^*$")
    ax.set_xlabel("$k$")
    ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

朴素的蛛网模型$\lambda = 1$会**发散**，越来越远地振荡偏离它试图寻找的均衡；请注意左侧面板的坐标尺度。

由于$h$是仿射的，迭代 {eq}`relaxation` 的乘子为$1 + \lambda(h' - 1)$，因此它当且仅当以下条件成立时收敛

$$
\bigl|1 + \lambda(h' - 1)\bigr| < 1
\qquad\Longleftrightarrow\qquad
\lambda < \frac{2}{1 + bn/\gamma} .
$$

```{code-cell} ipython3
λ_max = 2 / (1 + b * n / γ)
print(f"converges iff λ < {λ_max}")
print(f"λ = 1.0 : multiplier = {1 + 1.0 * (-b*n/γ - 1):+.2f}   (diverges)")
print(f"λ = 0.8 : multiplier = {1 + 0.8 * (-b*n/γ - 1):+.2f}   (period-2 cycle)")
print(f"λ = 0.5 : multiplier = {1 + 0.5 * (-b*n/γ - 1):+.2f}   (converges)")
```

在$\lambda = 0.8$时，乘子恰好为$-1$：该方案将$X$映射为$8 - X$，因此永远在$1$和$7$之间弹跳，既不收敛也不发散，这就是右侧面板中永不消退的锯齿形。

充分衰减调整过程能使算法找到均衡；衰减不足则会使算法追逐自己的尾巴。

请记住这一观察结果。

本系列一个反复出现的主题是：*如何*进行调适，决定了你最终会到达哪里，甚至决定了你是否能够到达终点。

### 适应性预期

方程 {eq}`relaxation` 最初是作为一个在迭代次数$k$下运行的算法引入的。

将$k$重新解释为**日历时间**$t$，将$X^*_t$解读为人们*预期*的值，将$X_t = h(X^*_{t-1})$解读为实际发生的情况，它就变成了一种预期形成理论：

```{math}
:label: adaptive_exp

X^*_t = (1 - \lambda) X^*_{t-1} + \lambda X_t
      = \lambda \sum_{j=0}^{\infty} (1 - \lambda)^j X_{t-j}.
```

这就是凯根 {cite:p}`Cagan` 用来研究恶性通货膨胀、以及 {cite:t}`Friedman1957` 用来研究消费的**适应性预期**方案。

预期是过去观测值的几何递减分布滞后，用单一自由参数$\lambda$来概括信念。

凯根和弗里德曼把$\lambda$当作一个自由参数，而没有解答为什么会有人以这种方式形成预期这一问题。

### 穆思的逆问题

{cite:t}`Muth1960` 着手通过反过来提出问题来消除$\lambda$这个自由参数。

他没有问某个给定环境暗示什么预测，而是问：**在什么样的环境下，指数平滑会是最优预测？**

他的答案是：当且仅当$X_t$遵循以下过程时，{eq}`adaptive_exp` 才是在每个视界$k$上对$X_{t+k}$的最小二乘预测：

```{math}
:label: muth_process

X_t = X_{t-1} + \epsilon_t - \theta \epsilon_{t-1},
```

其中$\{\epsilon_t\}$是一个鞅差序列，且$\theta \in [0, 1)$，并且平滑权重与移动平均系数通过下式相关联：

$$
\lambda = 1 - \theta \in (0, 1] .
$$

```{note}
萨金特将 {eq}`adaptive_exp` 中的平滑权重和 {eq}`muth_process` 中的移动平均系数都写作$\lambda$。

我们用$\theta$表示后者，以使$\lambda = 1 - \theta$这一关系保持清晰可见； {ref}`br_notation` 列出了其他身兼两职的符号。
```

预测规则从被预测的随机过程中继承了其唯一的参数。

让我们在视界$k = 1$处通过数值方法来验证这一点（ {ref}`br_ex4` 检验了更长的视界）：模拟 {eq}`muth_process`，在一系列权重上运行指数平滑，看看哪个权重预测效果最好。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Forecast MSE of exponential smoothing"
    name: fig-br-smoothing-mse
---
def smoothing_mse(x, λ, k=1):
    "MSE of exponential smoothing with weight λ as a forecast k steps ahead."
    # f_i = λ x_i + (1 - λ) f_{i-1}, started at f_0 = x_0
    f, _ = lfilter([λ], [1, λ - 1], x, zi=[(1 - λ) * x[0]])
    return np.mean((x[k:] - f[:-k]) ** 2)


rng = np.random.default_rng(0)
T = 200_000
ε = rng.standard_normal(T + 1)           # one shock path, shared across all λ
grid = np.linspace(0.02, 0.98, 97)

fig, ax = plt.subplots(figsize=(7, 4))
for θ in (0.2, 0.5, 0.8):
    X = np.cumsum(ε[1:] - θ * ε[:-1])    # Muth's process
    mse = np.array([smoothing_mse(X, λ) for λ in grid])
    line, = ax.plot(grid, mse, lw=1.2, label=fr"$\theta = {θ}$")
    ax.axvline(1 - θ, color=line.get_color(), lw=0.8, ls='--')
    print(f"θ = {θ}:  best λ = {grid[mse.argmin()]:.2f},  1 - θ = {1 - θ:.2f},"
          f"  min MSE = {mse.min():.4f}")
ax.set_xlabel(r"smoothing weight $\lambda$")
ax.set_ylabel("forecast MSE")
ax.set_ylim(0.9, 3)
ax.legend(frameon=False)
plt.show()
```

每条曲线恰好在其虚线$\lambda = 1 - \theta$处触底，且最小化后的均方误差在模拟噪声范围内等于$\mathbb{V}[\epsilon_t] = 1$——在该权重下的指数平滑*就是*条件期望，没有什么能比它做得更好。

穆思的这一练习是理性预期理念以其后来成为标准形式的首次应用：找出将一种预测方案与它所使用的环境联系起来的限制条件。

它也促使后来的研究者 {cite}`LucasPrescott1971` 将**预测方案本身**当作定义均衡的对象来处理。

### 动态类比

当我们从静态模型转向动态模型时，正是这种情形发生了。

让每个主体选择一个序列而非单一行动，即一个使下式最大化的$x_t$的权变计划，

$$
E \sum_{t=0}^{\infty} \beta^t \, \Pi(x_t, x_{t-1}, p_t),
$$

将一个价格方程$p_t = p(X_t, u_t)$以及有关总量状态的**感知运动规律**视为给定，

$$
X_t = H(X_{t-1}, u_t),
$$

其中$\{u_t\}$是独立同分布的，$\beta \in (0, 1)$是贴现因子。

该问题的欧拉方程为

$$
\Pi_1(x_t, x_{t-1}, p_t)
+ \beta \int \Pi_2(x_{t+1}, x_t, p_{t+1}) \, dF(p_{t+1} \mid X_t) = 0,
$$

其中$\Pi_i$是对第$i$个参数求导，$F(\cdot \mid X_t)$是感知运动规律$H$所引致的下一期价格的分布。

在理性预期均衡中，$F$正是*均衡*分布，也正是我们在开头所看到的、计量经济学家仍在努力估计的那个对象。

求解主体的动态规划问题得到个体决策规则
$x_t = h(x_{t-1}, X_{t-1}, u_t)$。

施加代表性主体确实具有代表性这一条件，即$x_t = X_t$，得出**实际运动规律**

$$
X_t = h(X_{t-1}, X_{t-1}, u_t) \equiv H^*(X_{t-1}, u_t),
$$

从而得到从感知运动规律到实际运动规律的一个映射，

```{math}
:label: t_map

H^* = T(H).
```

一个动态理性预期均衡是一个不动点$H = T(H)$，与 {eq}`static_ree` 是同一个思想，只是现在这个不动点存在于*函数*空间中，而不是数值空间中。

松弛算法可原样照搬，

```{math}
:label: t_relaxation

H^*_k = H^*_{k-1} + \lambda\bigl(T(H^*_{k-1}) - H^*_{k-1}\bigr),
```

只不过现在它是根据预测与实际发生情况之间的差距，对整个预期生成函数进行修订。

本系列中的计量经济学习器正是 {eq}`t_relaxation` 的各种版本，其中修订是由数据驱动的，而不是对$T$的精确求值，并且增益$\lambda$有时随时间递减，有时保持不变。

下一节将使这一族算法变得精确。

```{seealso}
{doc}`rational_expectations` 针对一个卢卡斯-普雷斯科特行业模型详细展开了$T$映射，并计算了其不动点，而 {doc}`lucas_prescott_investment` 则研究了卢卡斯-普雷斯科特模型本身。

{doc}`ls_learning` 研究了当主体通过最小二乘法估计感知运动规律，同时其估计又反过来影响它们所处的体系时会发生什么。
```

(br_one_recursion)=
## 一个递归，多种学习者

松弛算法是贯穿整个系列的一族递归中最简单的成员；要看到这个家族，只需在静态市场中加入一些噪声，让厂商去学习。

(br_bray)=
### 从松弛到学习：布雷市场

假设厂商必须在看到平均厂商产量之前就承诺产量。

每个厂商在进入第 $t$ 期时，持有一个关于平均产量的信念 $X^*_{t-1}$，该信念由截至 $t-1$ 的数据形成，并据此产出其最优反应 $h(X^*_{t-1})$；一个共同的供给冲击 $u_t$，独立同分布，均值为零，方差为 $\sigma_u^2$，随后决定实际发生的情况，厂商便朝该结果修正其信念：

```{math}
:label: bray_learning

X_t = h(X^*_{t-1}) + u_t,
\qquad
X^*_t = X^*_{t-1} + \lambda_t \bigl(X_t - X^*_{t-1}\bigr).
```

在没有冲击、增益为常数 $\lambda_t = \lambda$ 的情况下，这就是以日历时间运行的松弛算法 {eq}`relaxation`，也就是适应性预期。

当增益为 $\lambda_t = 1/t$ 时，$X^*_t$ 即为 $X_1, \ldots, X_t$ 的样本均值，是对一个常数的最小二乘估计，该模型在记号上即为 {cite:t}`Bray1982` 的模型，其反馈系数即为本讲座中的 $h'$（布雷将其记为 $b$，而在此处 $b$ 用来表示需求的斜率）。

信念误差 $e_t = X^*_t - X^*$ 满足

$$
e_t = \bigl[1 + \lambda_t (h' - 1)\bigr] e_{t-1} + \lambda_t u_t ,
$$

因此每一期都以该期的增益应用松弛乘数。

若常数增益高于界限 $2/(1 + bn/\gamma)$，信念会像蛛网模型那样发散；若低于该界限，它们不会发散，但每次冲击都会使其移动 $\lambda u_t$，从而使信念在 $X^*$ 附近的一个区间内永远游走。

一个递减的增益最终会降到任何界限之下，并且当 $\lambda \to 0$ 时，稳定性条件 $|1 + \lambda(h' - 1)| < 1$ 退化为 $h' < 1$，这正是布雷给出的最小二乘学习者收敛的条件，而蛛网市场中 $h' = -bn/\gamma < 0$，无论蛛网多陡峭，该条件都能满足。

下面这段代码将厂商置于本讲座的市场中，此处 $h' = -1.5$，高于 $0.8$ 的固定增益会发散，并在同一条冲击路径上比较三种增益。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Belief errors in Bray's market under fixed and declining gains"
    name: fig-br-bray
---
def learn(gain, u, X0=1.0):
    "Beliefs X*_t of firms that update with gain(t) in the market with shocks u."
    X = np.empty(len(u) + 1)
    X[0] = X0
    for t in range(1, len(u) + 1):
        X_t = h(X[t-1]) + u[t-1]                    # what actually happens
        X[t] = X[t-1] + gain(t) * (X_t - X[t-1])    # belief revision
    return X


σ_u, T_b = 0.5, 2_000
u = σ_u * np.random.default_rng(1).standard_normal(T_b)
t_b = np.arange(1, T_b + 1)

fig, ax = plt.subplots(figsize=(8, 4))
for label, gain, lw in [(r"$\lambda_t = 0.9$", lambda t: 0.9, 1.0),
                        (r"$\lambda_t = 0.3$", lambda t: 0.3, 0.5),
                        (r"$\lambda_t = 1/t$", lambda t: 1 / t, 1.0)]:
    ax.plot(t_b, np.abs(learn(gain, u)[1:] - X_star), lw=lw, label=label)
ax.plot(t_b, σ_u / np.sqrt(t_b), 'k--', lw=0.8, label=r"$\sigma_u / \sqrt{t}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e3)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$|X^*_t - X^*|$")
ax.legend(frameon=False, fontsize=9, ncol=2)
plt.show()
```

高于界限的固定增益会发散，而低于界限的固定增益会在一个区间内徘徊，该区间的宽度由增益决定（{ref}`br_ex5` 计算了这一宽度）。

只有递减的增益会收敛，其误差以 $t^{-1/2}$ 的速率缩小，这正是样本均值稳定下来的速率。

{doc}`ls_learning` 详细分析了布雷的模型，包括最小二乘发散的情形 $h' > 1$，而 {doc}`phillips_misspecified` 则以常数增益重新考察了该模型。

### 随机逼近与 E-稳定性

本系列中的大多数学习者通过如下形式的递归来更新信念向量 $\theta_t$：

```{math}
:label: stoch_approx

\theta_t = \theta_{t-1} + \lambda_t \, Q(\theta_{t-1}, x_t),
```

其中 $x_t$ 是最新的数据，$\lambda_t$ 是增益，$Q$ 是改进方向，例如预测误差或梯度，有时还会预先乘以某个估计的矩或曲率矩阵的逆，就像递归最小二乘法和牛顿法那样。

方程 {eq}`bray_learning` 就是 $\theta = X^*$，$Q = X_t - X^*_{t-1}$ 的情形。

{cite:t}`RobbinsMonro1951` 引入了此类递归，用以在只能带噪声观测函数的情况下寻找该函数的零点，而 {cite:t}`Ljung1977` 证明了，对于小增益，这些递归在平均意义上遵循微分方程 $d\theta/d\tau = E\, Q(\theta, x)$（在重新标度的时间 $\tau_t = \lambda_1 + \cdots + \lambda_t$ 下），其中期望是在固定 $\theta$ 的条件下取的。

当主体通过最小二乘法估计一个感知运动规律时，平均更新会从感知规律指向实际规律，微分方程就变为

```{math}
:label: e_ode

\frac{d\theta}{d\tau} = T(\theta) - \theta ,
```

其中 $T$ 是从感知运动规律到实际运动规律的映射 {eq}`t_map`。

对于布雷市场，$T = h$，{eq}`e_ode` 即为 $dX/d\tau = h(X) - X$，这正是 {cite:t}`Sargent1993`（第2章）早已赋予松弛算法的方程。

不动点 $\theta^* = T(\theta^*)$ 是一个理性预期均衡；如果它是 {eq}`e_ode` 的一个稳定静止点，则称其为 **E-稳定的**（期望稳定的），这要求雅可比矩阵 $DT(\theta^*)$ 的每个特征值的实部都小于一。

学习者在 E-稳定均衡附近的行为取决于增益。

* 对于**递减增益**（$\sum_t \lambda_t = \infty$ 且 $\sum_t \lambda_t^2 < \infty$，例如 $\lambda_t = 1/t$），在可能包括将信念限制在界限内的投影机制等正则性条件下，递归最小二乘法会收敛到一个 E-稳定均衡，而以概率零收敛到 E-不稳定均衡。
* 在随机环境中采用**常数增益**时，信念永远不会稳定下来：它们会在 E-稳定均衡附近以一个遍历分布波动，当增益较小时，其方差与增益成正比，并偶尔出现大幅逃逸 {cite}`Sargent1999,ChoWilliamsSargent2002,Williams2019`。
* 在确定性环境中，没有什么可以被平均掉，因此足够小的常数增益也会收敛，正如松弛算法那样。

E-稳定性是均衡附近最小二乘学习的正确条件，但它并非适用于所有算法的定理：随机梯度学习者可能无法收敛到 E-稳定均衡 {cite}`Giannitsarou2005`。

诸如后续讲座中的遗传算法和分类器系统之类的群体算法，不在该理论范围之内；借鉴阿里福维奇（Arifovic）1991年的学位论文，该书（第5章）报告称，遗传算法群体有时能够学会收敛到布雷模型的均衡，即使在 $h' > 1$ 的情形下也是如此。

该理论的核心早于该书，其第5章（脚注73及"一个相关的常微分方程"一节）使用 {eq}`e_ode` 来研究最小二乘学习。

这一思想可追溯到 {cite:t}`DeCanio1979` 和 {cite:t}`Evans1985,Evans1989`，他们利用 $T$ 映射定义了期望稳定性，以及 {cite:t}`MarcetSargent1989jet` 和 {cite:t}`Woodford1990`，他们利用 {cite:t}`Ljung1977` 的方法证明了该映射支配着最小二乘学习的收敛性；{cite:t}`EvansHonkapohja2001` 其后将该理论系统化并加以推广。

{doc}`ls_learning` 对此进行了详细阐述，而 {doc}`phillips_learning` 则对递归算法、均值动态和常数增益逃逸给出了简明的入门介绍。

### 该系列如何运用这一理论

| 讲座 | 学习者与增益 | 微分方程所解释的内容 |
|---|---|---|
| {doc}`olg_adaptive_money` | 罗宾斯-蒙罗储蓄规则，递减增益 | 收敛到平稳均衡 |
| | 对通货膨胀的最小二乘估计，权重随价格水平增长（实际上相当于常数增益） | {ref}`稳定性逆转 <olg_reversal>`：在低稳态 $\pi = 2$ 处 $T' = \tfrac12 < 1$，在高稳态 $\pi = 3$ 处 $T' = 2 > 1$ |
| | 学习菲利普斯曲线的政府，常数增益 | 反复出现的逃逸 |
| {doc}`learning_approximation` | 在线 {ref}`参数化预期 <lae_pea>`，递减增益 | 缓慢收敛，速率由 $DT$ 的最大特征值决定 |
| {doc}`exchange_rate_learning` | 牛顿-拉夫森投资组合规则，常数增益 | 一连续统的静止点，沿其没有漂移，因此极限取决于信念的初始位置 |
| | 遗传算法 | 不在该理论范围之内 |
| {doc}`genetic_classifier`、{doc}`marimon_mcgrattan_sargent` | 分类器系统，规则强度的 $1/\tau$ 平均化 | 如同主体极其缺乏耐心一般对未来收益进行估值的静止点 |

## 货币与价格

我们现在构建后续所有内容的两个模型中的第一个。

一个代表性的货币持有者选择从$t$到$t+1$期携带的名义余额$m_t$，以最大化

```{math}
:label: money_objective

\ln\left(2 w_1 - \frac{m_t}{p_t}\right) + \ln\left(2 w_2 + \frac{m_t}{p^*_{t+1}}\right),
\qquad w_1 > w_2 > 0,
```

其中$p_t$是当前价格水平，$p^*_{t+1}$是下一期预期的价格水平。

第一项是今天的消费，由于为获得货币而放弃的实际资源$m_t / p_t$而减少；第二项是明天的消费，由于该货币预期能够购买的商品$m_t / p^*_{t+1}$而增加。

对 {eq}`money_objective` 关于$m_t$求导并整理，得到货币需求

```{math}
:label: money_demand

\frac{m_t}{p_t} = w_1 - w_2 \frac{p^*_{t+1}}{p_t} .
```

当货币预期贬值得更快时，实际余额下降。

这是凯根 {cite:p}`Cagan` 用来研究恶性通货膨胀的需求函数的一个版本，它也出现在萨缪尔森 {cite:p}`Samuelson1958` 的世代交叠模型中。

按照原书的写法，禀赋被写为$2 w_1$和$2 w_2$，因此实际余额的需求为$w_1 - w_2 \pi$，其中$\pi = p^*_{t+1}/p_t$。

从 {doc}`olg_adaptive_money` 开始，禀赋记为$w_1$和$w_2$，于是同一个家庭的储蓄为$(w_1 - w_2 \pi)/2$。

要闭合该模型，我们需要一个关于$p^*_{t+1}$的理论。

假设货币供给以恒定的速率增长，

```{math}
:label: money_supply

M_{t+1} = \mu M_t ,
```

并且家庭相信价格水平与货币供给存在如下关系

```{math}
:label: price_belief

p_t = \gamma M_t + \lambda^t c ,
```

其中$(\gamma, \lambda, c)$是概括其信念的常数，满足$\gamma > 0$、$\lambda > 0$和$c \geq 0$，以使价格水平保持为正。

已知$\mu$，家庭预测
$p^*_{t+1} = \gamma \mu M_t + \lambda^{t+1} c$。

将该预测和 {eq}`price_belief` 代入 {eq}`money_demand`，得到货币需求作为当前货币供给的函数，

```{math}
:label: money_demand_solved

m_t = \gamma(w_1 - w_2 \mu) M_t + \lambda^t (w_1 - w_2 \lambda) c .
```

请注意在预期起作用的模型中的一个共同特征：货币的*需求*取决于其*供给*，因为今天的需求取决于明天的预期价格水平，而这被认为取决于明天的货币供给。

### 均衡

令需求等于供给，$m_t = M_t$，将 {eq}`money_demand_solved` 变为一个函数方程，

$$
M_t = \gamma (w_1 - w_2 \mu) M_t + \lambda^t (w_1 - w_2 \lambda) c ,
$$

该方程必须在每个日期都成立。

匹配这两项得到

```{math}
:label: money_ree

\gamma = (w_1 - \mu w_2)^{-1},
\qquad
\lambda = \frac{w_1}{w_2},
\qquad
c \geq 0 \ \text{ 为任意值} ,
```

因此均衡价格水平为

```{math}
:label: money_price_level

p_t = (w_1 - \mu w_2)^{-1} M_t + \left(\frac{w_1}{w_2}\right)^t c .
```

参数$\gamma$和$\lambda$被确定下来了。

常数$c$却没有被确定。

*每一个$c \geq 0$都是一个理性预期均衡*，并且在每一个均衡中，根据 {eq}`price_belief` 形成的预期总是恰好正确。

让我们数值验证一下，对于我们想尝试的任何$c$，残差确实为零。

```{code-cell} ipython3
w1, w2, μ, M0 = 2.0, 1.0, 1.5, 1.0
γ_m, λ_m = 1 / (w1 - μ * w2), w1 / w2

t = np.arange(12)
M = M0 * μ ** t

def price_path(c):
    "Equilibrium price level for bubble constant c."
    return γ_m * M + λ_m ** t * c

for c in (0.0, 0.5, 3.0, 25.0):
    p = price_path(c)
    p_star = γ_m * μ * M + λ_m ** (t + 1) * c     # forecast of next period's price
    m = w1 * p - w2 * p_star                      # money demand
    print(f"c = {c:5}:  max |demand - supply| = {np.max(np.abs(m - M)):.2e}")
```

在每一个日期，对于每一个$c$，货币需求都恰好等于货币供给。

(br_bubble)=
### 泡沫

由于$w_1 > w_2$，{eq}`money_price_level` 中的第二项以总增长率$\lambda = w_1 / w_2 > 1$增长。

在$c = 0$的均衡中，价格水平与货币供给成比例：这就是教科书形式的数量理论。

在其他每一个均衡中，价格水平都携带一个与货币供给无关、且呈指数增长的成分：一个纯粹投机性的**泡沫**。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Price levels and real balances under bubbles"
    name: fig-br-bubbles
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for c in (0.0, 0.5, 3.0):
    lab = f"$c = {c}$" + (" (quantity theory)" if c == 0 else "")
    axes[0].plot(t, price_path(c), 'o-', ms=3, lw=1, label=lab)
    axes[1].plot(t, M / price_path(c), 'o-', ms=3, lw=1, label=lab)

axes[0].set_yscale('log')
axes[0].set_ylabel("$p_t$  (log scale)")
axes[1].set_ylabel("real balances  $M_t / p_t$")
axes[1].axhline(w1 - μ * w2, color='k', lw=0.8, ls='--')
axes[1].set_ylim(0, 0.6)
for ax in axes:
    ax.set_xlabel("$t$")
    ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

左侧面板显示，随着$c$的增大，价格水平与数量理论路径之间的偏离越来越大。

右侧面板显示了这对货币存量的实际价值造成的影响：当$c = 0$时，实际余额恒定在$w_1 - \mu w_2$；而当$c > 0$时，泡沫使实际余额稳步趋向于零。

沿着一条泡沫路径，总通胀率不断攀升，趋向于$w_1 / w_2$，根据 {eq}`money_demand`，这恰恰是实际余额需求消失的速率。

经济体自我去货币化了，纯粹是因为每个人都预期它会这样。

```{code-cell} ipython3
infl = pd.DataFrame(
    {f"c = {c}": price_path(c)[1:] / price_path(c)[:-1] for c in (0.0, 0.5, 3.0)},
    index=pd.Index(t[1:], name="t"),
).round(3)
infl
```

模型中没有任何东西——无论是偏好、技术还是政策——能说明我们究竟处于哪一个$c$之中。

正如萨金特所说，"仅凭理性预期并不是一个足以确定结果的、充分限制性的原则"。

{doc}`olg_adaptive_money` 在货币供给恒定（$\mu = 1$，对应该讲座中的零赤字情形）的情况下，重新讨论了这个家庭。

其两个平稳通胀率分别是$\pi = 1$（即本节中的无泡沫均衡）和$\pi = w_1/w_2$（每条泡沫路径的极限）。

完全预见动态会偏离$\pi = 1$，但该讲座表明，最小二乘学习者会收敛到该值：学习消灭了泡沫。

## 两种货币

当我们允许存在一种以上货币时，不确定性会变得更严重。

以下结果的不同版本可归功于拉塞尔·博耶（Russell Boyer，1971年）和 {cite:t}`KarekenWallace1981`。

保持 {eq}`money_demand` 作为货币*总量*的需求，并假设存在两种法定货币，供给量分别为$M_{1t}$和$M_{2t}$，只要它们的回报率相等，它们就是完全替代品：

```{math}
:label: equal_returns

\frac{p^*_{1,t+1}}{p_{1t}} = \frac{p^*_{2,t+1}}{p_{2t}} .
```

这种对持有*哪种*货币的无差异性，正是使汇率不确定的原因。

假设人们相信价格水平由下式给出

```{math}
:label: two_currency_belief

p_{1t} = \gamma_1 M_{1t} + \gamma_2 e M_{2t} + c \lambda^t ,
\qquad
p_{2t} = e^{-1} p_{1t} ,
```

其中$e$是一个恒定的汇率。

要求以货币1单位计价的货币需求等于总供给$M_{1t} + e M_{2t}$，得到

```{math}
:label: two_currency_ree

\gamma_1 = (w_1 - \mu_1 w_2)^{-1},
\quad
\gamma_2 = (w_1 - \mu_2 w_2)^{-1},
\quad
\lambda = \frac{w_1}{w_2},
\quad
c \geq 0,
\quad
e \in [0, \infty) .
```

这些方程之所以引人注目，正是因为它们所省略的内容。

汇率$e$*完全不受约束*——如果这些方程对某个$e$有解，那么它们对其他任何$e$都有解——而$\gamma_1$和$\gamma_2$的公式根本不涉及$e$。

举最简单的情形：两种货币供给固定，$\mu_1 = \mu_2 = 1$，设$c = 0$。

那么$\gamma_1 = \gamma_2 = (w_1 - w_2)^{-1}$，价格水平为常数。

```{code-cell} ipython3
w1, w2 = 2.0, 1.0
H1, H2 = 100.0, 120.0                 # fixed supplies of the two currencies
γ_e = 1 / (w1 - w2)

def two_currency(e):
    "Price levels and the real allocation at exchange rate e."
    p1 = γ_e * (H1 + e * H2)
    p2 = p1 / e
    supply = H1 + e * H2              # total currency, in units of currency 1
    demand = (w1 - w2) * p1           # money demand, with p* = p since prices are constant
    return p1, p2, supply / p1, demand - supply

pd.DataFrame(
    [two_currency(e) for e in (0.25, 0.5, 1.0, 2.0, 4.0)],
    index=pd.Index([0.25, 0.5, 1.0, 2.0, 4.0], name="e"),
    columns=["$p_1$", "$p_2$", "real balances", "excess demand"],
).round(4)
```

每一行都是一个均衡。

随着$e$变化，名义价格水平的变动幅度很大。

由于$e$是货币2的一个单位以货币1的单位计价的价值，$e$越大意味着货币2越值钱，这会推高$p_1$并压低$p_2$。

但最后两列讲述了真正的故事：无论$e$为何值，总实际余额都是$w_1 - w_2 = 1$，市场恰好出清。

*在这些均衡中的每一个中，真实配置都是相同的。*

模型确定了人们消费什么，以及货币存量能够购买到多少购买力；但它对两种货币之间的兑换比率完全没有任何说法。

这是一个困扰国际货币理论已久的问题的一个尖锐版本。

这种不确定性并非局限于少数几个孤立的均衡：每一个$e \geq 0$都是一个均衡。

$e$不影响真实配置这一结论确实依赖于两种货币增长率相等的假设，正如 {ref}`br_ex3` 所示。

## 我们的处境

我们现在有了两个模型，在这两个模型中，理性预期对我们非常想预测的某件事情保持沉默。

对此有三种应对方式。

第一种是不断给环境增加限制条件，直到均衡变得唯一为止。

第二种是宣称这种不确定性是这个世界的一个真实特征，并且循着一个可追溯至李嘉图和维克塞尔的传统，去批判导致这种不确定性的制度安排；该书将这种使汇率依赖于历史的体制称为“一种设计不良的机制”。

第三种——也是本系列所追求的——是问，当我们**用适应性主体取代理性主体**并观察体系走向何方时，会发生什么。

这是一种实质性的改变，而不是技术性的改变。

一个适应性主体并没有被赋予均衡；它拥有信念、修正信念的规则以及初始条件。

那些额外的对象恰恰是理性预期均衡条件未能确定的东西，因此一个由适应性主体构成的体系可以选出一个理性预期无法确定的结果。

本系列的其余部分认真对待这一想法，其结果以一种颇具启发性的方式呈现出好坏参半的局面。

在 {doc}`olg_adaptive_money` 中的世代交叠货币经济中，最小二乘学习所选择的均衡，恰恰与理性预期动态所收敛到的均衡相反（在没有赤字的情形下，是无泡沫均衡），而人类实验受试者站在适应性模型这一边。

在 {doc}`exchange_rate_learning` 中的双货币模型中，适应性主体确实能够固定汇率，但只是通过使其依赖于信念的初始取值以及算法的调校方式来实现的。

学习算法的静止点恰好重现了这种不确定性；决定最终结果的，正是该书所称的 {ref}`历史那只死去的手 <xr_dead_hand>`，而该书也承认，这“作为确定汇率的依据，似乎是一根相当脆弱的稻草”。

与其说学习选定了某一汇率，不如说它只是记住了某一汇率。

在一个北原-赖特搜寻经济中，{doc}`marimon_mcgrattan_sargent` 中的分类器系统主体学会使用一种交换媒介。

但在 {ref}`其中一个经济体 <mms_a2>` 中——在该经济体里，投机性均衡是耐心主体唯一的平稳均衡——他们最终却选定了基本面的交易模式。

这并不是学习在多个均衡之间做出选择；而是收敛到了一个并非主体所处经济体均衡的结果。

马里蒙、麦格拉坦和萨金特曾怀疑原因在于早期的短视，而 {ref}`那一讲阐明了其中的缘由 <mms_impatience>`：分类器系统分配信用的方式，使得这些主体实际上变得极为缺乏耐心。

在前两种情形中，算法提供了均衡概念未能提供的东西；而在第三种情形中，算法则凌驾于均衡概念之上。

这类结果究竟是关于经济体的发现，还是算法的产物，是本系列不断回到的问题，而 {doc}`prospects_bounded_rationality` 将给出一个定论。

## 练习

```{exercise-start}
:label: br_ex1
```

松弛算法 {eq}`relaxation` 对静态市场收敛，当且仅当$\lambda < 2 / (1 + bn/\gamma)$。

比率$bn/\gamma$衡量的是需求斜率相对于成本曲率的大小，因此一个需求陡峭、成本近似线性的市场，正是朴素蛛网模型（$\lambda = 1$）表现糟糕的市场。

1. 请通过数值方法验证这一稳定性边界：对于一系列$bn/\gamma$的取值，找出使算法收敛的最大$\lambda$（在一个精细的网格上），并与解析预测进行比较。
1. 现在让松弛参数随迭代次数的增加而递减，即$\lambda_k = 1/k$。
   请证明对于每一种斜率，该算法都会收敛到$X^*$，包括那些使蛛网模型和大多数固定取值的$\lambda$都无法收敛的斜率。
   请借助 {ref}`br_one_recursion` 中的微分方程$dX/d\tau = h(X) - X$解释其原因。

```{exercise-end}
```

```{solution-start} br_ex1
:class: dropdown
```

针对第1部分，我们通过迭代400次并检验最后一步的变化是否小于$10^{-8}$来检验收敛性。

```{code-cell} ipython3
def converges(slope, λ, n_iter=400, tol=1e-8):
    """Does the relaxation algorithm converge when h'(X) = -slope?"""
    a_, X = 10.0, 1.0
    for _ in range(n_iter):
        X_new = X + λ * ((a_ - slope * X) - X)
        if not np.isfinite(X_new) or abs(X_new) > 1e12:
            return False
        X, X_prev = X_new, X
    return abs(X - X_prev) < tol


λ_grid = np.linspace(0.01, 1.5, 300)
rows = []
for slope in (0.5, 1.0, 1.5, 2.0, 4.0):
    ok = [λ for λ in λ_grid if converges(slope, λ)]
    rows.append((slope, max(ok) if ok else np.nan, 2 / (1 + slope)))

pd.DataFrame(rows, columns=["$bn/\\gamma$", "largest $\\lambda$ found",
                            "$2/(1 + bn/\\gamma)$"]).round(3)
```

数值边界始终位于解析边界的略*下方*，而这个差距并非网格间距造成的；它是该检验方法的一个真实特征。

当$\lambda$接近边界时，乘子$m = 1 - \lambda(1 + bn/\gamma)$趋近于$-1$，因此收敛变得任意缓慢，一个具有固定迭代次数和容差的检验会在真正边界之前就宣告失败。

这个差距的大小是可以预测的。

从$X_0 = 1$出发，该检验最后一步的大小为
$|X_{400} - X_{399}| = \lambda \, |9 - bn/\gamma| \, |m|^{399}$，因此只有当该值低于$10^{-8}$时检验才会通过。

```{code-cell} ipython3
fine = np.linspace(0.01, 1.5, 1_000_001)
predicted = []
for slope in (0.5, 1.0, 1.5, 2.0, 4.0):
    m = np.abs(1 - fine * (1 + slope))
    with np.errstate(over='ignore'):
        passes = (m < 1) & (fine * abs(9 - slope) * m ** 399 < 1e-8)
    predicted.append(fine[passes].max())

pd.DataFrame({"$bn/\\gamma$": [0.5, 1.0, 1.5, 2.0, 4.0],
              "found": [r[1] for r in rows],
              "predicted by the test": predicted,
              "true boundary": [r[2] for r in rows]}).round(3)
```

预测出的临界值恰好略高于通过网格找到的取值，相差不超过一个网格步长。

请注意第一行：当$bn/\gamma < 1$时，边界超过1，因此朴素蛛网模型本身就能收敛，无需任何衰减。

衰减正是在陡峭市场中换取收敛的手段，在固定$\lambda$的情况下，市场越陡峭，所需的衰减就越多。

针对第2部分，我们从相同的起点运行采用递减增益的算法版本，并报告400次和4000次迭代后与$X^*$的距离。

```{code-cell} ipython3
def relax_declining(slope, n_iter):
    "Distance from X* after n_iter steps of relaxation with λ_k = 1/k."
    a_, X = 10.0, 1.0
    for k in range(1, n_iter + 1):
        X = X + (1 / k) * ((a_ - slope * X) - X)
    return abs(X - a_ / (1 + slope))


slopes = (0.25, 0.5, 1.5, 4.5, 19.5)
pd.DataFrame({"$bn/\\gamma$": slopes,
              "fixed-λ bound": [2 / (1 + s_) for s_ in slopes],
              "error, k = 400": [relax_declining(s_, 400) for s_ in slopes],
              "error, k = 4000": [relax_declining(s_, 4000) for s_ in slopes]})
```

每一种斜率都能收敛，包括$bn/\gamma = 19.5$的情形——在该情形下，任何高于$0.1$的固定$\lambda$都会失败。

在$\lambda_k = 1/k$的情形下，误差在第$k$步会被乘以$1 - (1 + bn/\gamma)/k$；一旦$k > (1 + bn/\gamma)/2$，每一步都会产生收缩，乘子的乘积表现得类似于$k^{-(1 + bn/\gamma)}$，无论斜率如何，这个值都会趋于零。

微分方程说明了同样的道理：在重新标度的时间
$\tau_k = 1 + \tfrac12 + \cdots + \tfrac1k \approx \log k$下，该迭代跟踪的是
$dX/d\tau = h(X) - X$，其解为
$X(\tau) - X^* = e^{(h' - 1)\tau}\bigl(X(0) - X^*\bigr) \approx k^{h' - 1}\bigl(X(0) - X^*\bigr)$，
只要$h' < 1$，该解就是稳定的，而每一个向下倾斜的$h$都满足这一条件。

因此，在递减增益的情形下，市场越陡峭，收敛得*越快*，这与固定增益的情形恰好相反；代价在于，在平坦的市场中误差仅以$k$的幂次收缩（比较第一行中的两个误差列），而一个选择得当的固定$\lambda$则能使误差呈几何级数收缩。

```{solution-end}
```

```{exercise-start}
:label: br_ex2
```

均衡 {eq}`money_ree` 的推导过程中并未检验它是否具有经济意义。

请通过找出当货币增长速度快于此时实际余额需求会出现什么问题，来说明货币均衡要求$\mu < w_1 / w_2$。

```{exercise-end}
```

```{solution-start} br_ex2
:class: dropdown
```

沿着$c = 0$的均衡，实际余额恒定为
$M_t / p_t = \gamma^{-1} = w_1 - \mu w_2$。

只有当$\mu < w_1 / w_2$时，这个值才为正。

如果货币增长得更快，{eq}`money_ree` 仍然会"求解"该函数方程，但它要求家庭持有一个负数量的货币。

```{code-cell} ipython3
w1, w2 = 2.0, 1.0
μ_grid = np.array([0.5, 1.0, 1.5, 1.9, 2.0, 2.5])

with np.errstate(divide='ignore'):        # γ is infinite exactly at μ = w1/w2
    table = pd.DataFrame({
        "$\\mu$": μ_grid,
        "$\\gamma = (w_1 - \\mu w_2)^{-1}$": 1 / (w1 - μ_grid * w2),
        "real balances $w_1 - \\mu w_2$": w1 - μ_grid * w2,
        "monetary equilibrium?": np.where(μ_grid < w1 / w2, "yes", "no"),
    }).round(3)
table
```

在$\mu = w_1 / w_2 = 2$时，对实际余额的需求降为零，$\gamma$则趋于无穷大；超过这一点后，两者都变为负值。

其直觉可以通过 {eq}`money_demand` 来理解：只有当购买力的预期损失$p^*_{t+1} / p_t$小于$w_1 / w_2$时，家庭才会持有货币。

以速率$\mu$增长的货币会产生速率为$\mu$的通货膨胀，因此$\mu \geq w_1 / w_2$会将货币需求推向零，而这恰恰是*泡沫*均衡从下方渐近趋近的同一个边界。

```{solution-end}
```

```{exercise-start}
:label: br_ex3
```

在双货币示例中，我们设定$\mu_1 = \mu_2 = 1$，并发现真实配置在每个汇率下都是相同的。

这一点是特殊的。

请用$\mu_1 \neq \mu_2$——比如$\mu_1 = 1.0$、$\mu_2 = 1.3$，配以$w_1 = 2$、$w_2 = 1$、$M_{1,0} = M_{2,0} = 100$以及$c = 0$——重复这一计算，并在最初几期内计算若干汇率下的总实际余额。

汇率$e$的选择是否仍然不影响真实配置？

```{exercise-end}
```

```{solution-start} br_ex3
:class: dropdown
```

在$c = 0$的情形下，以货币1计价的总实际余额为
$(M_{1t} + e M_{2t}) / (\gamma_1 M_{1t} + \gamma_2 e M_{2t})$。

```{code-cell} ipython3
w1, w2 = 2.0, 1.0
μ1, μ2 = 1.0, 1.3
γ1, γ2 = 1 / (w1 - μ1 * w2), 1 / (w1 - μ2 * w2)
t = np.arange(10)
M1, M2 = 100.0 * μ1 ** t, 100.0 * μ2 ** t

def real_balances(e):
    p1 = γ1 * M1 + γ2 * e * M2
    return (M1 + e * M2) / p1

pd.DataFrame({f"e = {e}": real_balances(e) for e in (0.25, 1.0, 4.0)},
             index=pd.Index(t, name="t")).round(4)
```

下面是在图中展示的同一组路径：

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for e in (0.25, 1.0, 4.0):
    ax.plot(t, real_balances(e), 'o-', ms=3, lw=1, label=f"$e = {e}$")
ax.axhline(1 / γ1, color='k', lw=0.8, ls='--', label=r"$1/\gamma_1$")
ax.axhline(1 / γ2, color='gray', lw=0.8, ls=':', label=r"$1/\gamma_2$")
ax.set_xlabel("$t$")
ax.set_ylabel("total real balances")
ax.legend(frameon=False)
plt.show()
```

不会。

在货币增长率不相等的情况下，汇率会影响真实配置，而且这种配置甚至不再随时间保持恒定。

原因在于$\gamma_1 \neq \gamma_2$：这两种货币被赋予了不同的价值，因为人们预期它们会以不同的速率被稀释，因此货币存量的*构成*就变得重要起来，而$e$正是决定这种构成的因素。

由于货币2增长得更快，无论我们选择哪个$e$，它最终都会主导货币存量，实际余额都会从$e$的起始点开始收敛到$1/\gamma_2$。

因此，$e$对真实配置不产生影响这一结论是一种刀锋边缘现象：它要求$\mu_1 = \mu_2$（在任意共同增长率$\mu$下，实际余额在每个$e$下都是$w_1 - \mu w_2$）。

而$e$本身的不确定性并非刀锋边缘现象：表中的每一个$e$仍然都是一个均衡，只是此时在这些均衡之间做出选择会带来真实的后果。

```{solution-end}
```

```{exercise-start}
:label: br_ex4
```

缪斯的结果表明，在*每一个*预测时域$k$下，使用$\lambda = 1 - \theta$的指数平滑法都是对$X_{t+k}$的最优预测，但 {numref}`fig-br-smoothing-mse` 仅检验了$k = 1$的情形。

1. 请证明对于 {eq}`muth_process`，条件期望$E_t X_{t+k}$对于每一个$k \geq 1$都是相同的，它等于使用$\lambda = 1 - \theta$得到的指数平滑值，并且当$\mathbb{V}[\epsilon_t] = 1$时，$k$步预测误差的方差为$1 + (k - 1)(1 - \theta)^2$。
1. 请在$k = 4$时通过数值方法验证这一结果，重复使用 `smoothing_mse`。

```{exercise-end}
```

```{solution-start} br_ex4
:class: dropdown
```

将$X_{t+k}$写成$X_{t+k} = X_t + \sum_{j=1}^{k} (\epsilon_{t+j} - \theta \epsilon_{t+j-1})$。

由于每一个发生在$t$之后的冲击的条件均值都为零，因此对于每一个$k \geq 1$都有$E_t X_{t+k} = X_t - \theta
\epsilon_t \equiv f_t$。

由于$\epsilon_t = X_t - f_{t-1}$，我们有$f_t = X_t - \theta(X_t - f_{t-1}) = (1 -
\theta) X_t + \theta f_{t-1}$，这正是取$\lambda = 1 - \theta$的 {eq}`adaptive_exp`。

预测误差为

$$
X_{t+k} - f_t = \epsilon_{t+k} + (1 - \theta)(\epsilon_{t+1} + \cdots + \epsilon_{t+k-1}),
$$

其方差为$1 + (k - 1)(1 - \theta)^2$。

下面是基于与图中相同冲击序列的数值检验。

```{code-cell} ipython3
k = 4
rows = []
for θ in (0.2, 0.5, 0.8):
    X = np.cumsum(ε[1:] - θ * ε[:-1])
    mse = np.array([smoothing_mse(X, λ, k) for λ in grid])
    rows.append((θ, grid[mse.argmin()], 1 - θ, mse.min(), 1 + (k - 1) * (1 - θ) ** 2))

pd.DataFrame(rows, columns=["$\\theta$", "best $\\lambda$", "$1 - \\theta$",
                            "min MSE", "$1 + (k-1)(1-\\theta)^2$"]).round(3)
```

在时域4下，最优权重再次等于$1 - \theta$（在网格精度和抽样噪声的范围内），且最小化后的均方误差与公式吻合。

误差方差会随着预测时域的增加而增大，但预测本身并不会改变：在每一个预测时域下，最优的都是同一个平滑值。

```{solution-end}
```

```{exercise-start}
:label: br_ex5
```

在布雷市场 {eq}`bray_learning` 中，当固定增益$\lambda$低于稳定性边界时，信念永远不会稳定下来。

1. 请证明信念误差$e_t = X^*_t - X^*$是一个平稳的一阶自回归过程，其方差为

   $$
   V(\lambda) = \frac{\lambda \sigma_u^2}{(1 - h')\bigl(2 - \lambda(1 - h')\bigr)} .
   $$

1. 请使用 {ref}`br_bray` 中的 `learn`，针对
   $\lambda \in \{0.025, 0.05, 0.1, 0.2, 0.4\}$ 模拟100,000期，并将$e_t$的样本方差与
   $V(\lambda)$进行比较。
   信念的离散程度是如何随着增益变化的？

```{exercise-end}
```

```{solution-start} br_ex5
:class: dropdown
```

由 {eq}`bray_learning` 可知，$e_t = \rho e_{t-1} + \lambda u_t$，其中$\rho = 1 - \lambda(1 - h')$。

在边界以下$|\rho| < 1$，因此$e_t$是一个平稳自回归过程，其方差为
$\lambda^2 \sigma_u^2 / (1 - \rho^2)$，而
$1 - \rho^2 = \lambda (1 - h')\bigl(2 - \lambda(1 -
h')\bigr)$正好给出了这个公式。

该模拟以信念等于$X^*$为起点，并舍弃最初的1,000期。

```{code-cell} ipython3
u_long = σ_u * np.random.default_rng(2).standard_normal(100_000)
h_prime = -b * n / γ

rows = []
for λ in (0.025, 0.05, 0.1, 0.2, 0.4):
    e = learn(lambda t: λ, u_long, X0=X_star)[1_000:] - X_star
    V = λ * σ_u**2 / ((1 - h_prime) * (2 - λ * (1 - h_prime)))
    rows.append((λ, e.var(), V, e.var() / λ))

pd.DataFrame(rows, columns=["gain $\\lambda$", "sample variance", "$V(\\lambda)$",
                            "variance / $\\lambda$"]).round(5)
```

样本方差与公式相吻合。

当增益较小时，$V(\lambda) \approx \lambda \sigma_u^2 / \bigl(2(1 - h')\bigr)$，因此信念的方差与增益成正比，其标准差则与增益的平方根成正比；最后一列中的比值在较大增益下会逐渐上升，这是分母中第二个因子导致的。

因此，具有固定增益的学习者仅能依分布收敛，收敛到围绕$X^*$的一团云，只有当$\lambda \to 0$时，这团云才会收缩为一个点。

{doc}`phillips_learning` 展示了在一个具有更丰富反馈机制的模型中，这样一团云会带来怎样的后果：逃逸。

```{solution-end}
```
