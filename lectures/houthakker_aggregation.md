---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 帕累托分布与总生产函数
  headings:
    Overview: 概述
    Production cells: 生产单元
    Fixed coefficients: 固定系数
    A Pareto distribution: 帕累托分布
    A Pareto distribution::Deriving the Cobb–Douglas form: 推导科布-道格拉斯形式
    A Pareto distribution::Why a Pareto distribution gives a Cobb–Douglas function: 为什么帕累托分布会给出科布-道格拉斯函数
    A Pareto distribution::Checking the formulas: 验证公式
    Activity analysis with finitely many cells: 有限多个单元的活动分析
    Activity analysis with finitely many cells::The aggregate production function as a linear program: 作为线性规划的总生产函数
    Activity analysis with finitely many cells::Drawing cells from Houthakker's distribution: 从 Houthakker 分布中抽取单元
    Activity analysis with finitely many cells::Substitution through entry and exit: 通过进入和退出实现替代
    Activity analysis with finitely many cells::Checking efficiency: 验证效率
    Activity analysis with finitely many cells::Comparing with Cobb–Douglas: 与科布-道格拉斯函数比较
    Levhari's inverse problem: Levhari 的逆问题
    Levhari's inverse problem::One variable input: 单一可变投入
    Levhari's inverse problem::Inverting the aggregate production function: 反演总生产函数
    Levhari's inverse problem::The CES case: CES 情形
    Levhari's inverse problem::The Cobb–Douglas limit: 科布-道格拉斯极限情形
    Later work: 后续研究
    Concluding remarks: 结语
    Exercises: 练习
---

(houthakker_aggregation)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 帕累托分布与总生产函数

```{contents} Contents
:depth: 2
```

## 概述

宏观经济模型通常用一个各投入要素可以相互替代的光滑生产函数来描述一个行业，或整个经济体。

然而，许多单独的机器和工厂却是按几乎固定的比例运作的。

一台纺纱机每生产一磅纱线需要一定数量的动力和劳动力，给一台闲置的纺纱机增加工人并不会产生更多的纱线。

{cite:t}`Houthakker1955` 阐明了这两种描述是如何契合在一起的。

他研究了一个由许多小型**生产单元**组成的行业，每个单元具有固定的投入比例，其产能由固定投入所决定，而各单元之间的比例互不相同。

当投入价格发生变化时，一些单元开始生产，另一些则停产，因此即使没有任何单个单元改变其投入组合，整个行业的投入组合也会发生变化。

行业层面的替代效应来源于哪些单元在运作这一变化。

Houthakker 的主要结论是：如果各单元的投入系数服从帕累托分布，那么该行业的总生产函数就是科布-道格拉斯函数。

这个科布-道格拉斯函数的指数并不是任何单元技术的属性：它们是单元分布的参数。

因此，即使没有任何单元进行替代，总生产函数也可能看起来光滑而稳定，并且每当进入、退出或投资改变了这一分布时，它就会随之移动。

{cite:t}`Levhari1968` 把问题反过来提出：给定一个总生产函数，什么样的单元分布能产生它？

他针对 CES 生产函数回答了这个问题。

在本讲座中，我们将：

* 证明 Houthakker 的效率定理：如果每个单元都在相同的价格下实现利润最大化，那么该行业的总投入与总产出是有效率的；
* 推导 Houthakker 的科布-道格拉斯生产函数，并识别出帕累托分布中产生该函数的性质；
* 以 {cite:t}`Koopmans1951` 的活动分析精神，将有限多个单元的总生产函数表示为一个线性规划问题来计算；
* 利用该规划问题的对偶，将任意单元分布的总生产函数写成关于价格的最小化问题，即便 Houthakker 认为没有一般方法消去价格，这一表达式也依然可以计算；
* 重现 Levhari 的反演方法，给出其一般公式，并纠正他对产生 CES 生产函数的分布形态的描述。

本讲座用到线性规划与对偶性（如 {doc}`intro:lp_intro`、{doc}`opt_transport` 和 {doc}`von_neumann_model` 中所用）、帕累托尾（如 {doc}`intro:heavy_tails` 中所用），以及洛伦兹曲线（如 {doc}`wealth_dynamics` 中所用）。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import linprog, brentq, minimize
from scipy.special import beta as beta_function, gamma as gamma_function
```

## 生产单元

Houthakker 考虑一个由许多**生产单元**组成的行业。

一个单元可以是一家企业、一个工厂、一台单独的机器，或是一名工人。

每个单元使用两种**可变投入** $x_1$ 和 $x_2$，以及若干**固定投入**，生产一种产出 $x_0$。

固定投入在所研究的期间内无法改变。

它们界定了该单元，并限定了它能生产的产量。

（Houthakker 允许多种产出和多种可变投入，但他很快就专门研究一种产出和两种可变投入的情形，我们在此也采用这种做法。）

对各单元求和得到该行业的总产出 $X_0$ 以及总可变投入 $X_1$ 和 $X_2$。

Houthakker 并不对固定投入求和，因为这些投入甚至可能无法度量。

如果各单元凭借其所拥有的固定投入，能够共同从 $X_1$ 和 $X_2$ 中生产出 $X_0$，则总量 $(X_0, X_1, X_2)$ 称为**可达的**。

按照 Koopmans 的思路，如果不存在另一个可达的总量，在产出上至少同样多、在每种投入上至多同样多，并且在这三者之中至少有一个方面严格更优，那么这个可达总量就是**有效率的**。

```{prf:theorem} 效率定理（Houthakker 定理）
:label: ha_thm_efficiency

设产出价格为 $p_0 > 0$，可变投入价格为 $p_1 > 0$ 和 $p_2 > 0$。

如果每个单元都在其生产可能性中选择使利润 $p_0 x_0 - p_1 x_1 - p_2 x_2$ 最大化的那个，那么该行业的总量 $(X_0, X_1, X_2)$ 就是有效率的。

而且，在所有可达总量中，它也使行业的总利润最大化。
```

```{prf:proof}
总利润 $p_0 X_0 - p_1 X_1 - p_2 X_2$ 是各单元利润之和。

每个单元的利润都尽可能大，因此没有任何可达总量能赚取更多利润。

假设另一个可达总量 $(X_0', X_1', X_2')$ 满足 $X_0' \geq X_0$、$X_1' \leq X_1$、$X_2' \leq X_2$，且至少有一个不等式是严格的。

由于所有价格均为正，该总量所获利润将严格高于 $(X_0, X_1, X_2)$，这是不可能的。
```

这个证明除了假设每个单元都实现利润最大化之外，没有用到关于这些单元的任何其他信息。

它是生产者第一福利定理的一个版本。

在只有一种产出的情形下，每个有效率的总量都落在**总生产函数**

$$
X_0 = F(X_1, X_2),
$$

的图像上，这是在各单元固定投入给定的情况下，行业从总投入 $X_1$ 和 $X_2$ 中所能达到的最大产出。

在 $F$ 对两种投入都严格递增的地方（如下文 Houthakker 的情形），有效率的总量正好就是这个函数的图像。

{prf:ref}`ha_thm_efficiency` 表明，价格接受型的各单元会把整个行业带到这个函数上。

## 固定系数

从现在起，假设每个单元具有固定系数。

一个单元生产一单位产出需要 $a_1$ 单位的 $x_1$ 和 $a_2$ 单位的 $x_2$，它可以生产从零到由其固定投入所确定的**产能**之间的任意数量。

系数 $a = (a_1, a_2)$ 在各单元之间互不相同。

由于所有价格按比例变化不会影响任何单元的选择，我们设定 $p_0 = 1$，并记 $p = (p_1, p_2)$。

系数为 $a$ 的单元每生产一单位就获得 $1 - a_1 p_1 - a_2 p_2$ 的利润。

如果这是正的，该单元就满负荷生产；如果是负的，该单元就不生产。

Houthakker 假设各单元数量众多且规模很小，因此恰好处于边际（利润为零）的单元可以忽略不计。

在 $(a_1, a_2)$ 平面上，进行生产的单元就是位于**价格三角形**内的那些单元，该三角形位于直线 $a_1 p_1 + a_2 p_2 = 1$ 之下，此直线与坐标轴分别交于 $1/p_1$ 和 $1/p_2$。

Houthakker 用**投入产出分布** $\varphi$ 来概括这些单元：系数落在一个小矩形 $[a_1, a_1 + da_1] \times [a_2, a_2 + da_2]$ 内的单元，其总产能为 $\varphi(a_1, a_2) \, da_1 \, da_2$。

在价格三角形上积分，即可得到价格 $p$ 下行业的产出与投入：

$$
X_0(p) = \iint_{a \cdot p < 1} \varphi(a) \, da,
\qquad
X_i(p) = \iint_{a \cdot p < 1} a_i \, \varphi(a) \, da,
\quad i = 1, 2 .
$$ (ha_totals)

这是关于两个价格的三个方程。

消去价格后，剩下 $X_0$、$X_1$ 与 $X_2$ 之间的一个关系式。

根据 {prf:ref}`ha_thm_efficiency`，利润最大化的各单元所生产出的总量是有效率的，因此这个关系式就是总生产函数。

**行业利润函数**

$$
\Pi(p) = \iint \max\{0, 1 - a \cdot p\} \, \varphi(a) \, da = X_0(p) - p_1 X_1(p) - p_2 X_2(p)
$$ (ha_profit)

把各单元的利润加总起来。

一个单元的固定投入限制了其产出，因此其利润就是对这些固定投入的回报，马歇尔称之为**准租金**。

在积分号下求导可得到霍特林引理，$\partial \Pi / \partial p_i = -X_i(p)$：行业利润对某种投入价格的导数等于该行业对该投入使用量的相反数。

$\Pi$ 在 $p$ 上是凸的，因为它对 $p$ 的凸函数进行积分。

我们在下文会再次用到 $\Pi$。

## 帕累托分布

Houthakker 假设投入产出分布为

$$
\varphi(a_1, a_2) = A \, a_1^{\alpha_1 - 1} a_2^{\alpha_2 - 1},
$$ (ha_pareto)

其中常数 $A > 0$、$\alpha_1 > 0$、$\alpha_2 > 0$。

（Houthakker 要求 $\alpha_i \geq 1$，但下面的推导只需要 $\alpha_i > 0$。）

这个名称来自生产率的分布。

对于只有一种可变投入、系数为 $a$ 的情形，一个单元每单位投入可获得 $y = 1/a$ 单位产出。

在 {eq}`ha_pareto` 的单投入版本 $\varphi(a) = A a^{\alpha - 1}$ 下，生产率超过 $y$ 的单元其总产能为

$$
\int_0^{1/y} A a^{\alpha - 1} \, da = \frac{A}{\alpha} \, y^{-\alpha},
$$

这正是帕累托关于超过 $y$ 的收入数量的定律。

总产能 $\iint \varphi(a) \, da$ 是无穷的：存在数量无限、生产率越来越低的单元供给。

在任意正价格下，只有价格三角形内的单元才有意义，而它们的总产能是有限的。

### 推导科布-道格拉斯形式

为计算 $X_0$，先对 $a_2$ 积分，再代入 $\tau = a_1 p_1$：

$$
\begin{aligned}
X_0
&= A \int_0^{1/p_1} a_1^{\alpha_1 - 1} \int_0^{(1 - a_1 p_1)/p_2} a_2^{\alpha_2 - 1} \, da_2 \, da_1
= \frac{A}{\alpha_2} \int_0^{1/p_1} a_1^{\alpha_1 - 1} \left(\frac{1 - a_1 p_1}{p_2}\right)^{\alpha_2} da_1 \\
&= \frac{A}{\alpha_2 \, p_1^{\alpha_1} p_2^{\alpha_2}} \int_0^1 \tau^{\alpha_1 - 1} (1 - \tau)^{\alpha_2} \, d\tau
= \frac{A \, B(\alpha_1, \alpha_2 + 1)}{\alpha_2 \, p_1^{\alpha_1} p_2^{\alpha_2}} ,
\end{aligned}
$$

其中 $B$ 是贝塔函数。

同样的步骤可以得到 $X_1$ 和 $X_2$。

令 $D = A \, B(\alpha_1 + 1, \alpha_2 + 1)$。

利用恒等式 $B(x, y) = \frac{x + y}{x} B(x + 1, y)$，可得 $B(\alpha_1, \alpha_2 + 1) = \frac{1 + \alpha_1 + \alpha_2}{\alpha_1} B(\alpha_1 + 1, \alpha_2 + 1)$，于是 Houthakker 将结果写成

$$
X_0 = \frac{(1 + \alpha_1 + \alpha_2) D}{\alpha_1 \alpha_2 \, p_1^{\alpha_1} p_2^{\alpha_2}},
\qquad
X_1 = \frac{D}{\alpha_2 \, p_1^{\alpha_1 + 1} p_2^{\alpha_2}},
\qquad
X_2 = \frac{D}{\alpha_1 \, p_1^{\alpha_1} p_2^{\alpha_2 + 1}} .
$$ (ha_pareto_totals)

（在原论文中，$X_2$ 的公式被误印成了 $X_1$ 的标签。）

两式相除可见，每种投入在产出价值中所占份额是常数：

$$
\frac{p_i X_i}{X_0} = \frac{\alpha_i}{1 + \alpha_1 + \alpha_2} \equiv \gamma_i,
\quad i = 1, 2 .
$$ (ha_shares)

将 $p_i = \gamma_i X_0 / X_i$ 代入 $X_0$ 的公式，得到一个两边都含有 $X_0$ 的方程；解出它可得

$$
X_0 = C X_1^{\gamma_1} X_2^{\gamma_2},
\qquad
C = (1 + \alpha_1 + \alpha_2) \left( \frac{D}{\alpha_1^{\alpha_1 + 1} \alpha_2^{\alpha_2 + 1}} \right)^{1/(1 + \alpha_1 + \alpha_2)} .
$$ (ha_cobb_douglas)

这就是 Houthakker 的科布-道格拉斯生产函数。

它有三个特征值得评述。

第一，投入 $i$ 的边际产量是 $\gamma_i X_0 / X_i = p_i$，正如 {prf:ref}`ha_thm_efficiency` 所要求的那样。

第二，各指数之和为 $(\alpha_1 + \alpha_2)/(1 + \alpha_1 + \alpha_2) < 1$。

由于固定投入保持不变，可变投入的回报是递减的，而固定投入则以准租金 $\Pi$ 的形式获得余下的产出份额 $1/(1 + \alpha_1 + \alpha_2)$。

第三，若加入一个虚构的投入 $X_3$，其指数为 $1/(1 + \alpha_1 + \alpha_2)$，就会使该函数变成一次齐次函数。

Houthakker 警告说，这样做会产生误导：真正重要的是固定投入在各单元间是如何分布的，而不是它们的总量。

### 为什么帕累托分布会给出科布-道格拉斯函数

原因在于一种标度性质。

将 $p_1$ 降低为 $p_1 / \lambda$，会使价格三角形沿 $a_1$ 方向伸展 $\lambda$ 倍。

在 {eq}`ha_pareto` 下，$\varphi(\lambda a_1, a_2) = \lambda^{\alpha_1 - 1} \varphi(a_1, a_2)$，因此伸展后三角形中的各单元看起来就像旧三角形中各单元经过重新标度后的样子。

在 {eq}`ha_totals` 中做变量代换 $a_1 = \lambda \tilde a_1$ 可知，$X_0$ 和 $X_2$ 都乘以 $\lambda^{\alpha_1}$，而 $X_1$ 乘以 $\lambda^{\alpha_1 + 1}$。

因此当 $p_1$ 变化时，$p_1 X_1 / X_0$ 保持不变，对 $p_2$ 也有同样的论证成立。

因此每种投入的份额都是常数。

由于每个份额都等于弹性 $\partial \log F / \partial \log X_i$，份额恒定的生产函数必定是科布-道格拉斯函数。

真正起作用的性质是 $\varphi$ 在 $a_1$ 和 $a_2$ 上*分别*按幂律进行标度。

{ref}`ha_ex_homogeneous` 表明，仅在 $(a_1, a_2)$ 联合标度是不够的。

### 验证公式

下面的函数计算 {eq}`ha_pareto_totals`，接下来的函数计算 {eq}`ha_cobb_douglas`。

```{code-cell} ipython3
def pareto_totals(p, A, α):
    """Industry output and inputs (X0, X1, X2) at input prices p."""
    (p1, p2), (α1, α2) = p, α
    D = A * beta_function(α1 + 1, α2 + 1)
    X0 = (1 + α1 + α2) * D / (α1 * α2 * p1**α1 * p2**α2)
    X1 = D / (α2 * p1**(α1 + 1) * p2**α2)
    X2 = D / (α1 * p1**α1 * p2**(α2 + 1))
    return X0, X1, X2


def cobb_douglas(X1, X2, A, α):
    """Houthakker's aggregate production function."""
    α1, α2 = α
    n = 1 + α1 + α2
    D = A * beta_function(α1 + 1, α2 + 1)
    C = n * (D / (α1**(α1 + 1) * α2**(α2 + 1)))**(1 / n)
    return C * X1**(α1 / n) * X2**(α2 / n)
```

作为独立的检验，我们对积分 {eq}`ha_totals` 进行数值计算。

```{code-cell} ipython3
def totals_by_quadrature(p, φ):
    """Integrate φ, a1 φ and a2 φ over the price triangle."""
    p1, p2 = p
    limits = (0, 1 / p1, 0, lambda a1: (1 - a1 * p1) / p2)
    X0 = integrate.dblquad(lambda a2, a1: φ(a1, a2), *limits)[0]
    X1 = integrate.dblquad(lambda a2, a1: a1 * φ(a1, a2), *limits)[0]
    X2 = integrate.dblquad(lambda a2, a1: a2 * φ(a1, a2), *limits)[0]
    return X0, X1, X2
```

我们在全文中使用 $A = 1$、$\alpha_1 = 2$ 和 $\alpha_2 = 1.5$，因此 $\gamma_1 = 4/9$、$\gamma_2 = 1/3$，固定投入的份额为 $2/9$。

```{code-cell} ipython3
A, α = 1.0, (2.0, 1.5)
φ = lambda a1, a2: A * a1**(α[0] - 1) * a2**(α[1] - 1)

print(f"{'p':12} {'X0 (quad)':>10} {'X0 (formula)':>12} {'F(X1, X2)':>11} "
      f"{'p1 X1/X0':>9} {'p2 X2/X0':>9}")
for p in [(1.0, 1.0), (0.7, 1.6), (2.0, 0.8)]:
    X0_q, _, _ = totals_by_quadrature(p, φ)
    X0, X1, X2 = pareto_totals(p, A, α)
    print(f"{str(p):12} {X0_q:10.6f} {X0:12.6f} "
          f"{cobb_douglas(X1, X2, A, α):11.6f} "
          f"{p[0] * X1 / X0:9.4f} {p[1] * X2 / X0:9.4f}")
```

公式与数值积分相符，科布-道格拉斯函数 {eq}`ha_cobb_douglas` 根据投入重现了产出，并且在每个价格向量下各份额都分别为 $4/9$ 和 $1/3$。

## 有限多个单元的活动分析

Houthakker 的标题将他的论文置于**活动分析**这一领域之中，这是 {cite:t}`Koopmans1951` 及其同事所发展出的、把生产看作线性活动之间选择的研究方法。

在只有有限多个单元时，他的加总方法就变成了一个线性规划问题。

### 作为线性规划的总生产函数

设单元 $j = 1, \ldots, N$ 的产能为 $c_j$，系数为 $a_j = (a_{1j}, a_{2j})$，并以强度 $z_j \in [0, 1]$（即其产能被使用的比例）运行。

从投入 $X = (X_1, X_2)$ 中所能达到的最大产出为

$$
\begin{aligned}
F_N(X) = \max_{z} \ & \sum_{j=1}^N c_j z_j \\
\text{subject to} \ & \sum_{j=1}^N a_{ij} c_j z_j \leq X_i, \quad i = 1, 2, \\
& 0 \leq z_j \leq 1, \quad j = 1, \ldots, N .
\end{aligned}
$$ (ha_lp)

对两个投入约束赋予乘子 $p_i \geq 0$，对产能约束 $z_j \leq 1$ 赋予乘子 $\mu_j \geq 0$。

对偶规划为

$$
\min_{p \geq 0, \ \mu \geq 0} \ p_1 X_1 + p_2 X_2 + \sum_{j=1}^N \mu_j
\quad \text{subject to} \quad
\mu_j \geq c_j (1 - a_j \cdot p), \quad j = 1, \ldots, N .
$$

对于给定的 $p$，最小的可行 $\mu_j$ 为 $c_j \max\{0, 1 - a_j \cdot p\}$，即单元 $j$ 在价格 $p$ 下的准租金。

将其代入，并利用线性规划的对偶定理（它说明这两个规划具有相同的最优值，参见 {doc}`opt_transport`），可得

$$
F_N(X) = \min_{p \geq 0} \left\{ p \cdot X + \Pi_N(p) \right\},
\qquad
\Pi_N(p) = \sum_{j=1}^N c_j \max\{0, 1 - a_j \cdot p\} .
$$ (ha_duality)

这个公式告诉我们四件事。

第一，弱对偶性给出了 {prf:ref}`ha_thm_efficiency`。

在任意价格 $p \geq 0$ 下，每个单元的产出至多等于其投入成本加上准租金，因此任何可行的强度组合所产生的产出至多为 $p \cdot X + \Pi_N(p)$。

在 $p$ 下实现利润最大化的各单元，以其自身的投入 $X$ 达到了这个上界，因此没有任何可达总量能做得更好。

与 {doc}`opt_transport` 中一样，互补松弛性给出了逆命题：在最优对偶价格下，每个满足 $a_j \cdot p < 1$ 的单元都满负荷运行，每个满足 $a_j \cdot p > 1$ 的单元都闲置，因此每个有效率的总量都是利润最大化的各单元在某个价格下所能生产出来的。

第二，这一结论对连续多个单元的情形也同样成立。

用积分代替求和，同样的论证给出 $F(X) = \min_{p \geq 0} \{ p \cdot X + \Pi(p) \}$，其中 $\Pi$ 来自 {eq}`ha_profit`，对任何由某些价格所生成的投入向量 $X$ 均成立。

在 Houthakker 的分布下，这对每个 $X > 0$ 都成立。

反过来，$\Pi(p) = \max_X \{ F(X) - p \cdot X \}$，因此总生产函数与行业利润函数互为对偶，两者中的任何一个都可以从另一个恢复出来。

第三，这个公式给出的是一种数值方法，而不是一个封闭形式的解。

Houthakker 写道，"似乎没有一种普遍适用的方法"可以从他那些版本的 {eq}`ha_totals` 中"消去 $p_1$ 和 $p_2$"，"从而直接由 $\varphi$ 推导出 $F$"。

这个最小化过程并不能以封闭形式消去价格：根据霍特林引理，其一阶条件正是方程 $X_i(p) = X_i$。

它所增添的，是一个只涉及两个变量的凸问题，对于任何 $\varphi$，这个问题都容易用数值方法求解，而且在有限多个单元的情形下——此时方程 $X_i(p) = X_i$ 通常没有精确解——它仍然是有意义的。

第四，$F_N$ 是凹的且分段线性的，因此它的等产量线也是分段线性的。

随着从 $\varphi$ 中抽取的单元数 $N$ 增大，这些等产量线会逼近 $F$ 的光滑等产量线。

为说明第三点，我们通过对 $p \cdot X + \Pi(p)$ 在价格上求最小值来计算 Houthakker 分布下的 $F$，其中 $\Pi$ 是通过数值积分而非封闭形式求出的。

我们在 $\log p$ 上进行搜索以保证价格为正。

```{code-cell} ipython3
def profit_by_quadrature(p, φ):
    """Industry profit Π(p) from (ha_profit), by numerical integration."""
    p1, p2 = p
    return integrate.dblquad(
        lambda a2, a1: (1 - a1 * p1 - a2 * p2) * φ(a1, a2),
        0, 1 / p1, 0, lambda a1: (1 - a1 * p1) / p2)[0]


def F_by_duality(X, φ):
    """Minimize p·X + Π(p) over p > 0; return the value and the prices."""
    def objective(log_p):
        p = np.exp(log_p)
        return p @ X + profit_by_quadrature(p, φ)
    res = minimize(objective, np.zeros(2), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-12})
    return res.fun, np.exp(res.x)


print(f"{'X':14} {'min p·X + Π(p)':>15} {'Cobb-Douglas':>13}   prices")
for X in ([0.03, 0.025], [0.05, 0.02]):
    value, p = F_by_duality(np.array(X), φ)
    print(f"{str(X):14} {value:15.8f} {cobb_douglas(*X, A, α):13.8f}   "
          f"({p[0]:.4f}, {p[1]:.4f})")
```

最小化过程重现了科布-道格拉斯函数，且使之最小化的价格正是边际产量 $\gamma_i X_0 / X_i$。

### 从 Houthakker 分布中抽取单元

在 {eq}`ha_pareto` 下，总产能是无穷的，因此我们无法从全部产能中进行抽取。

取而代之，我们将各单元限定在一个大三角形 $a_1/\bar a_1 + a_2/\bar a_2 \leq 1$ 内。

只要 $p_1 \geq 1/\bar a_1$ 且 $p_2 \geq 1/\bar a_2$，这个大三角形就包含价格三角形，而在这些价格下，截断不会改变任何东西。

在这个大三角形内部，经标度的系数 $(a_1/\bar a_1, a_2/\bar a_2)$ 在集合 $b_1 + b_2 \leq 1$ 上具有正比于 $b_1^{\alpha_1 - 1} b_2^{\alpha_2 - 1}$ 的密度。

这正是参数为 $(\alpha_1, \alpha_2, 1)$ 的狄利克雷随机向量前两个坐标的分布。

这个大三角形就是在 $p = (1/\bar a_1, 1/\bar a_2)$ 处的价格三角形，因此它的总产能为 $X_0(1/\bar a_1, 1/\bar a_2) = A \, \bar a_1^{\alpha_1} \bar a_2^{\alpha_2} \, \Gamma(\alpha_1) \Gamma(\alpha_2) / \Gamma(1 + \alpha_1 + \alpha_2)$，其中 $\Gamma$ 是伽马函数；我们将其平均分配给 $N$ 个单元。

```{code-cell} ipython3
ā = np.array([2.0, 2.0])


def draw_cells(N, A, α, ā, rng):
    """N cells from φ restricted to the triangle a1/ā1 + a2/ā2 <= 1."""
    b = rng.dirichlet([α[0], α[1], 1.0], size=N)[:, :2]
    total = (A * ā[0]**α[0] * ā[1]**α[1] * gamma_function(α[0])
             * gamma_function(α[1]) / gamma_function(1 + α[0] + α[1]))
    return ā * b, np.full(N, total / N)


def F_lp(X, a, c):
    """Solve the linear program (ha_lp); return output and dual prices."""
    res = linprog(-c, A_ub=(a * c[:, None]).T, b_ub=X,
                  bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return -res.fun, -res.ineqlin.marginals
```

### 通过进入和退出实现替代

Houthakker 的图 1 只画出了直线 $a_1 p_1 + a_2 p_2 = 1$。

下图增加了 600 个单元，标出了在价格 $p = (0.75, 0.9)$ 下生产的单元，以及当 $p_1$ 降至 $0.55$ 时新进入的单元。

单元云在 $a_1 + a_2 = 2$ 处截止，这仅仅是因为我们在那里截断了分布；而它在靠近原点处变得稀疏，是因为当 $\alpha_i > 1$ 时，投入需求很低的单元是稀缺的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Cells that enter when $p_1$ falls
    name: fig-ha-cells
---
rng = np.random.default_rng(1)
a, c = draw_cells(600, A, α, ā, rng)
p, p_new = np.array([0.75, 0.9]), np.array([0.55, 0.9])
active, active_new = a @ p < 1, a @ p_new < 1
entering = active_new & ~active

fig, ax = plt.subplots()
ax.scatter(*a[~active_new].T, s=8, color="0.75", label="idle")
ax.scatter(*a[active].T, s=8, color="C0", label="producing at $p$")
ax.scatter(*a[entering].T, s=8, color="C2", label="entering at $p'$")
ax.plot([0, 1 / p[0]], [1 / p[1], 0], color="C1", lw=2,
        label=r"$a \cdot p = 1$")
ax.plot([0, 1 / p_new[0]], [1 / p_new[1], 0], color="C1", lw=2, ls="--",
        label=r"$a \cdot p' = 1$")
ax.set_xlabel(r"$a_1$")
ax.set_ylabel(r"$a_2$")
ax.set_aspect("equal")
ax.legend()
plt.show()

for label, cells in (("producing at p", active),
                     ("entering at p'", entering),
                     ("producing at p'", active_new)):
    X = (a[cells] * c[cells, None]).sum(axis=0)
    print(f"{label:16}: {cells.sum():3} cells, X1/X2 = {X[0] / X[1]:.2f}")
```

价格线以其与 $a_2$ 轴的截距为轴进行转动，因此新进入的单元位于一个向 $a_1$ 轴方向张开的楔形区域内。

相对于已经在生产的单元，它们平均使用更多的投入 1（相对于投入 2 而言）。

因此，当投入 1 变得更便宜时，行业的比率 $X_1/X_2$ 会上升，尽管没有任何单元改变自身的投入组合。

### 验证效率

接下来，我们对一个单元样本检验 {prf:ref}`ha_thm_efficiency`。

在价格 $p = (1, 1)$ 下，我们让每个单元实现利润最大化，把各单元的产出与投入加总起来，然后在这些总投入下求解线性规划。

```{code-cell} ipython3
rng = np.random.default_rng(1234)
a, c = draw_cells(3000, A, α, ā, rng)

p = np.array([1.0, 1.0])
active = a @ p < 1
X0_cells = c[active].sum()
X_cells = (a[active] * c[active, None]).sum(axis=0)
F_value, duals = F_lp(X_cells, a, c)
print(f"output when each cell maximizes profit: {X0_cells:.6f}")
print(f"largest attainable output F_N(X):      {F_value:.6f}")
print(f"dual prices:                           "
      f"({duals[0]:.4f}, {duals[1]:.4f})")
```

这两个产出值相符，正如 {prf:ref}`ha_thm_efficiency` 所要求的那样。

对偶价格与各单元所面对的价格 $p = (1, 1)$ 相近，但并不完全相同。

在有限多个单元的情形下，一小组价格可以支持所有单元的相同选择，求解器所报告的只是其中之一。

$p = (1, 1)$ 也在这一价格集合之中：在这些价格下，{eq}`ha_duality` 中的对偶目标函数 $p \cdot X + \Pi_N(p)$ 等于正在生产的各单元的产出。

### 与科布-道格拉斯函数比较

现在我们将 $F_N$ 与 {eq}`ha_pareto_totals` 给出的总量 $X(p)$ 处的 Houthakker 科布-道格拉斯函数进行比较。

```{code-cell} ipython3
print(f"{'p':12} {'F_N(X(p))':>10} {'Cobb-Douglas':>13}   dual prices")
for p in [(1.0, 1.0), (0.7, 1.6), (1.3, 0.8)]:
    X0, X1, X2 = pareto_totals(p, A, α)
    F_value, duals = F_lp([X1, X2], a, c)
    print(f"{str(p):12} {F_value:10.4f} {X0:13.4f}   "
          f"({duals[0]:.3f}, {duals[1]:.3f})")
```

有了 3,000 个单元（其中几百个进行生产），$F_N$ 在这三个价格向量处都与科布-道格拉斯函数相差不到百分之一，对偶价格也与生成 $X(p)$ 的价格相差约 6% 以内。

其中部分是运气使然。

产出取决于正在生产的那几百个单元，在随机抽取的 3,000 个单元样本中，$F_N$ 与科布-道格拉斯函数之间的差距的标准差约为 2%。

下图比较了等产量线。

我们将投入限定在一个使科布-道格拉斯价格满足 $p_i \geq 1/\bar a_i$ 的区域，这样截断后的分布就与完整分布具有相同的总生产函数 {eq}`ha_cobb_douglas`。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Isoquants of $F_N$ and Cobb–Douglas
    name: fig-ha-isoquants
---
X0_ref, X1_ref, X2_ref = pareto_totals((1.0, 1.0), A, α)
levels = X0_ref * np.array([0.8, 1.0, 1.2])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, N, n_grid in zip(axes, (100, 3000), (80, 30)):
    rng = np.random.default_rng(0)
    a, c = draw_cells(N, A, α, ā, rng)
    x1 = np.linspace(0.5, 1.6, n_grid) * X1_ref
    x2 = np.linspace(0.5, 1.6, n_grid) * X2_ref
    F_grid = np.array([[F_lp([u, v], a, c)[0] for u in x1] for v in x2])
    G1, G2 = np.meshgrid(x1, x2)
    ax.contour(G1, G2, cobb_douglas(G1, G2, A, α), levels=levels,
               colors="0.4", linestyles="--", linewidths=1)
    ax.contour(G1, G2, F_grid, levels=levels, colors="C0", linewidths=2)
    ax.plot([], [], color="C0", lw=2, label=r"$F_N$")
    ax.plot([], [], color="0.4", lw=1, ls="--", label="Cobb–Douglas")
    ax.set_title(f"{N:,} cells")
    ax.set_xlabel(r"$X_1$")
    ax.legend(loc="upper right")
axes[0].set_ylabel(r"$X_2$")
plt.show()
```

当只有 100 个单元时，在网格上的任意一点通常只有大约十个单元在运行，因此等产量线明显呈多边形状。

在这个样本中，它们也明显位于科布-道格拉斯等产量线内侧，意味着用相同的投入产生了更多的产出。

这是运气使然：在 100 个单元的样本中，$F_N$ 与科布-道格拉斯函数之间差距的标准差约为 9%。

当有 3,000 个单元时，它们变得光滑，且接近 {eq}`ha_cobb_douglas` 的等产量线。

剩余的差距——约占产出的百分之一或二——是抽样误差，其符号在不同样本间会发生变化。

## Levhari 的逆问题

Houthakker 是从单元的分布出发得到总生产函数。

{cite:t}`Levhari1968` 则反过来：给定一个总生产函数，什么样的单元分布能产生它？

如果对每一个合理的生产函数都存在答案，那么 Houthakker 的构造方法就不是一种针对科布-道格拉斯函数的特殊技巧，而是一种将任意总生产函数读解为固定比例单元之和的方式。

和 Levhari 一样，我们简化到只有一种可变投入——我们称之为劳动——这样逆问题就变成了一维问题。

### 单一可变投入

每个单元生产一单位产出需要 $t$ 单位劳动，$g(t) \, dt$ 是需求介于 $t$ 和 $t + dt$ 之间的单元的总产能。

这里 $t$ 扮演系数 $a$ 的角色，$g$ 扮演 $\varphi$ 的角色：Houthakker 的单投入分布为 $g(t) = A t^{\alpha - 1}$。

在工资 $w$（以产出为单位计量）下，若 $t < 1/w$，某单元就会生产。

设 $x = 1/w$ 为**边际单元**的需求量。

该行业的产出与就业为

$$
Q = G(x) = \int_0^x g(t) \, dt,
\qquad
L = H(x) = \int_0^x t \, g(t) \, dt .
$$ (ha_QL)

这些就是单一投入（其价格为 $w$）情形下 {eq}`ha_totals` 中的 $X_0$ 和 $X_1$。

当 $x$ 变化时，$(L, Q)$ 描绘出总生产函数 $Q = F(L)$。

Levhari 指出，这一关系式是一条洛伦兹曲线。

如果我们把各单元按需求量从低到高排成一列，那么 $Q$ 就是队列中前面那些单元的产能，而 $L$ 则是它们所使用的劳动量。

因此，$L$ 作为 $Q$ 的函数，就是一条按产能加权的劳动需求量的非归一化洛伦兹曲线，正如 {doc}`wealth_dynamics` 中那样。

它是凸的，因为沿队列需求量不断上升，这使得 $F$ 是凹的。

对 {eq}`ha_QL` 求导得到

$$
F'(L) = \frac{G'(x)}{H'(x)} = \frac{g(x)}{x \, g(x)} = \frac{1}{x} = w .
$$

劳动的边际产量就是边际单元的生产率，并且等于工资，这正是 {prf:ref}`ha_thm_efficiency` 所要求的。

### 反演总生产函数

这个关系式解决了逆问题。

给定一个凹的、递增的 $F$，在就业量为 $L$ 时的边际单元需求量为 $x = 1/F'(L)$，而需求量低于 $x$ 的各单元共生产 $F(L)$：

$$
G\!\left(\frac{1}{F'(L)}\right) = F(L) .
$$ (ha_inversion)

等价地，在队列中位于位置 $Q$ 的那个单元需要 $1 / F'(F^{-1}(Q))$ 单位劳动。

对 {eq}`ha_inversion` 关于 $L$ 求导，可得产能的密度：

$$
g(x) = -\frac{F'(L)^3}{F''(L)},
\qquad \text{where } F'(L) = 1/x .
$$ (ha_density)

在生产函数接近线性之处，产能是密集的。

如果许多单元的需求量几乎相同，那么随着劳动的增加，边际产量几乎不下降，因此 $F''$ 很小。

作为检验，考虑 Houthakker 的科布-道格拉斯情形 $F(L) = L^\gamma$。

于是 $F'(L) = \gamma L^{\gamma - 1} = 1/x$ 给出 $L = (\gamma x)^{1/(1 - \gamma)}$，{eq}`ha_inversion` 给出

$$
G(x) = (\gamma x)^{\gamma/(1 - \gamma)} .
$$

这就是 {eq}`ha_pareto` 的单投入版本，其中 $\alpha = \gamma / (1 - \gamma)$，这与 {eq}`ha_shares` 中的 $\gamma = \alpha / (1 + \alpha)$ 是等价的。

### CES 情形

Levhari 将反演方法应用于 CES 生产函数

$$
F(L) = \left( \gamma L^\rho + \beta \right)^{1/\rho},
\qquad \gamma > 0, \ \beta > 0, \ \rho < 0 .
$$ (ha_ces)

Levhari 用 $\alpha$ 表示我们的 $\gamma$，用 $A$ 表示我们的 $L$，用 $p$ 表示我们的工资 $w$，用 $\phi$ 和 $\psi$ 表示我们的 $G$ 和 $H$。

我们用 $\gamma$ 表示劳动的权重，因为当 $\gamma + \beta = 1$ 时，随着 $\rho \to 0$，它会变成科布-道格拉斯指数。

常数 $\beta$ 代表固定投入，设为 1，因此 $F(L) = \tilde F(L, 1)$，其中 $\tilde F(L, K) = (\gamma L^\rho + \beta K^\rho)^{1/\rho}$。

劳动与固定投入之间的替代弹性为 $\sigma = 1/(1 - \rho)$，它小于 1。

如果 $\rho$ 为正，那么 $F(0) = \beta^{1/\rho}$ 就会是正的：该行业无需劳动即可生产，而这是 Houthakker 的假设所排除的。

{eq}`ha_ces` 的两个性质塑造了这一分布。

1. 平均产量 $F(L)/L = (\gamma + \beta L^{-\rho})^{1/\rho}$ 随 $L$ 从零开始上升而从 $\gamma^{1/\rho}$ 下降，因此没有任何单元每单位产出所需的劳动低于 $x_{\min} = \gamma^{-1/\rho}$。
2. 当 $L \to \infty$ 时，$F(L) \to \beta^{1/\rho}$，因此总产能是有限的。

为求解 {eq}`ha_inversion`，记 $v = (x_{\min}/x)^{1 - \sigma}$，并利用 $\rho = 1 - 1/\sigma$。

在边际单元处，$F'(L) = \gamma (F(L)/L)^{1 - \rho} = 1/x$，因此平均产量为 $F(L)/L = (\gamma x)^{-\sigma}$。

于是 $\gamma (L/F(L))^\rho = \gamma (\gamma x)^{\sigma - 1} = v$，因为 $x_{\min}^{1 - \sigma} = \gamma^\sigma$。

将 $F(L)^\rho = \gamma L^\rho + \beta$ 两边除以 $F(L)^\rho$，得到 $1 = v + \beta F(L)^{-\rho}$，再由 {eq}`ha_inversion` 得到

$$
G(x) = \left( \frac{\beta}{1 - v} \right)^{1/\rho}
= \beta^{1/\rho} \left[ 1 - \left(\frac{x_{\min}}{x}\right)^{1 - \sigma} \right]^{\sigma/(1 - \sigma)},
\qquad x \geq x_{\min} .
$$ (ha_ces_dist)

这就是改写后的 Levhari 方程 (7)。

将其除以总产能 $\beta^{1/\rho}$，即可得到需求量的概率分布。

求导可得产能密度：

$$
g(x) = \beta^{1/\rho} \, \frac{\sigma}{x} \left(\frac{x_{\min}}{x}\right)^{1 - \sigma}
\left[ 1 - \left(\frac{x_{\min}}{x}\right)^{1 - \sigma} \right]^{(2 \sigma - 1)/(1 - \sigma)}
= \frac{\sigma}{x} \, \frac{v}{1 - v} \, G(x) .
$$ (ha_ces_density)

指数 $(2\sigma - 1)/(1 - \sigma)$ 决定了在最低需求量 $x_{\min}$（此处括号内的项为零）附近会发生什么。

* 若 $1/2 < \sigma < 1$，指数为正，当 $x \downarrow x_{\min}$ 时 $g(x) \to 0$。
* 若 $\sigma = 1/2$，指数为零，$g(x_{\min}) = \beta^{1/\rho} \sigma / x_{\min} = 1/(2 \gamma \beta)$；下图所绘出的归一化密度在 $\sigma / x_{\min} = 1/(2\gamma)$ 处起始。
* 若 $\sigma < 1/2$，指数为负，当 $x \downarrow x_{\min}$ 时 $g(x) \to \infty$。

这些情形在经济上是合理的。

当 $\sigma \to 0$ 时，CES 函数趋近于固定比例函数，各单元趋近于需求量为 $x_{\min}$ 的单一类型，因此产能在该处堆积。

当 $\sigma$ 趋近于 1 时，需求量更加分散，极少有单元位于最低值附近。

```{note}
Levhari 的正文及其图 1 把第一种和第三种情形弄反了：他说当 $-1 < \rho < 0$ 时 $g(x) \to \infty$（在 $x_{\min}$ 处），而当 $\rho < -1$ 时 $g(x) \to 0$。

他自己给出的公式 (8) 结果是正确的，因为其括号项在 $x_{\min}$ 处为零，并带有指数 $-(1 + \rho)/\rho = (2\sigma - 1)/(1 - \sigma)$，当 $-1 < \rho < 0$ 时该值为正。

他在旁注中写道"（即 $(1 + \rho)/\rho < 0$）"，这表明在读取各情形时，指数前面的负号被遗漏了。

公式 (8) 还把 $\beta^{1/\rho}$ 放在分母上：对他的公式 (7) 求导后，$\beta^{1/\rho}$ 出现在分子中，正如 {eq}`ha_ces_density` 所示，而归一化密度中不含 $\beta$，所以公式 (8) 只有在 $\beta = 1$ 时才是正确的。

他的图 1 给出边界情形下的密度为（用我们的记号表示）$\gamma^{1/2} x^{-3/2} / (2 \beta)$，这与 {eq}`ha_ces_density` 相符。
```

下面的代码实现了 CES 函数、封闭形式 {eq}`ha_ces_dist` 和 {eq}`ha_ces_density`，以及一般的反演公式 {eq}`ha_inversion`——即数值求解 $F'(L) = 1/x$。

封闭形式通过 $v$ 来编码，$v$ 计算为 $(\gamma x^\rho)^{1/(1 - \rho)}$，它等于 $(x_{\min}/x)^{1 - \sigma}$。

这样在 $\rho$ 接近零时，可以避免出现非常大的数 $\beta^{1/\rho}$ 和非常小的数 $x_{\min}$。

```{code-cell} ipython3
def ces(L, γ, β, ρ):
    return (γ * L**ρ + β)**(1 / ρ)


def ces_marginal_product(L, γ, β, ρ):
    return γ * (ces(L, γ, β, ρ) / L)**(1 - ρ)


def ces_capacity(x, γ, β, ρ):
    """G(x) from (ha_ces_dist), written as (β / (1 - v))**(1/ρ)."""
    v = (γ * x**ρ)**(1 / (1 - ρ))       # equals (x_min / x)**(1 - σ)
    return (β / (1 - v))**(1 / ρ)


def ces_density(x, γ, β, ρ):
    """g(x) from (ha_ces_density), written as (σ/x) (v/(1 - v)) G(x)."""
    σ = 1 / (1 - ρ)
    v = (γ * x**ρ)**(1 / (1 - ρ))
    return σ / x * v / (1 - v) * ces_capacity(x, γ, β, ρ)


def capacity_by_inversion(x, F, F_prime, L_bounds=(1e-12, 1e12)):
    """G(x) from (ha_inversion): find L with F'(L) = 1/x, return F(L)."""
    L = brentq(lambda L: F_prime(L) - 1 / x, *L_bounds,
               xtol=1e-14, rtol=1e-12)
    return F(L)


def levhari_7(x, γ, β, ρ):
    """Levhari's equation (7), in our notation."""
    z = (γ * x)**(ρ / (ρ - 1))
    return (β * z / (z - γ))**(1 / ρ)
```

我们检验这三种 $G$ 的表达式是否相符，以及 {eq}`ha_ces_density` 是否确实是 {eq}`ha_ces_dist` 的导数。

我们在全文中使用 $\gamma = 0.6$ 和 $\beta = 0.4$；下文中 $\gamma + \beta = 1$ 将会很重要。

```{code-cell} ipython3
γ, β = 0.6, 0.4

print(f"{'ρ':<6}{'closed - inversion':>20}{'closed - Levhari (7)':>22}"
      f"{'g/(dG/dx) - 1':>15}")
for ρ in (-0.5, -1.0, -2.0):
    x_min = γ**(-1 / ρ)
    xs = x_min * np.array([1.01, 1.5, 3.0, 10.0])
    G_closed = ces_capacity(xs, γ, β, ρ)
    G_inv = [capacity_by_inversion(
                 x, lambda L: ces(L, γ, β, ρ),
                 lambda L: ces_marginal_product(L, γ, β, ρ)) for x in xs]
    h = 1e-6
    g_numeric = (ces_capacity(xs * (1 + h), γ, β, ρ)
                 - ces_capacity(xs * (1 - h), γ, β, ρ)) / (2 * h * xs)
    e1 = np.max(np.abs(G_closed - G_inv))
    e2 = np.max(np.abs(G_closed - levhari_7(xs, γ, β, ρ)))
    e3 = np.max(np.abs(ces_density(xs, γ, β, ρ) / g_numeric - 1))
    print(f"{ρ:<6}{e1:>20.1e}{e2:>22.1e}{e3:>15.1e}")
```

下面是三种替代弹性下归一化分布 $G/\beta^{1/\rho}$ 的密度。

将它们与 Levhari 的图 1 比较，他的图把第一种和第三种形态弄反了。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Densities that generate CES functions
    name: fig-ha-ces-densities
---
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, ρ in zip(axes, (-0.5, -1.0, -2.0)):
    σ = 1 / (1 - ρ)
    x_min = γ**(-1 / ρ)
    xs = x_min * (1 + np.geomspace(1e-4, 4, 400))
    density = ces_density(xs, γ, β, ρ) / β**(1 / ρ)
    ax.plot(xs, density, lw=2)
    ax.axvline(x_min, color="0.5", lw=1, ls="--")
    ax.set_title(rf"$\rho = {ρ}$, $\sigma = {σ:.2f}$")
    ax.set_xlabel("labor requirement $x$")
    ax.set_xlim(0, 5 * x_min)
    ax.set_ylim(0, 1.6 * density[xs > 1.05 * x_min].max())
    if σ < 0.5:
        ax.annotate(r"$\to \infty$", xy=(x_min, ax.get_ylim()[1]),
                    xytext=(6, -14), textcoords="offset points")
axes[0].set_ylabel("density")
plt.tight_layout()
plt.show()
```

虚线标出了 $x_{\min}$ 的位置，每个子图都有各自的纵轴比例。

当 $\sigma = 2/3$ 时，密度从零开始；当 $\sigma = 1/2$ 时，密度从一个有限值开始；当 $\sigma = 1/3$ 时，密度在 $x_{\min}$ 处是无界的，图中将其截断处理。

为做独立检验，我们从 {eq}`ha_ces_dist` 中抽取各单元，并按照 {eq}`ha_QL` 对它们进行加总。

对 {eq}`ha_ces_dist` 求逆，可得出在归一化分布中分位数为 $u \in (0, 1)$ 的单元的需求量：

$$
x = x_{\min} \left( 1 - u^{(1 - \sigma)/\sigma} \right)^{-1/(1 - \sigma)} .
$$

```{code-cell} ipython3
def draw_requirements(N, γ, ρ, rng):
    """Draw N labor requirements from the normalized distribution."""
    σ = 1 / (1 - ρ)
    u = rng.random(N)
    return γ**(-1 / ρ) * (1 - u**((1 - σ) / σ))**(-1 / (1 - σ))
```

我们将各单元按需求量从低到高排序，并给每个单元分配相等份额的总产能 $\beta^{1/\rho}$。

累积求和描绘出总生产函数。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Aggregating cells drawn from Levhari's distributions
    name: fig-ha-ces-check
---
rng = np.random.default_rng(2)
N = 20_000
L_grid = np.linspace(1e-3, 6, 200)

fig, ax = plt.subplots()
for ρ, color in zip((-0.5, -2.0), ("C0", "C1")):
    t = np.sort(draw_requirements(N, γ, ρ, rng))
    capacity = β**(1 / ρ) / N
    Q = capacity * np.arange(1, N + 1)
    L = capacity * np.cumsum(t)
    keep = L <= L_grid[-1]
    ax.plot(L[keep], Q[keep], color=color, lw=4, alpha=0.4,
            label=rf"{N:,} cells, $\rho = {ρ}$")
    ax.plot(L_grid, ces(L_grid, γ, β, ρ), color=color, lw=1, ls="--",
            label=rf"CES, $\rho = {ρ}$")
ax.set_xlabel("labor $L$")
ax.set_ylabel("output $Q$")
ax.legend()
plt.show()
```

加总后的各单元重现了 CES 函数。

### 科布-道格拉斯极限情形

当 $\rho \to 0$ 时，带有 $\gamma + \beta = 1$ 的 CES 函数 {eq}`ha_ces` 趋近于 $L^\gamma$。

Levhari 利用洛必达法则证明，此时 $G$ 趋近于幂律 $(\gamma x)^{\gamma/(1 - \gamma)}$，因此 CES 分布趋近于 Houthakker 的分布。

（他没有陈述条件 $\gamma + \beta = 1$，但这个极限需要该条件：否则 $(\gamma L^\rho + \beta)^{1/\rho}$ 在 $\rho \to 0$ 时会趋于零或趋于无穷。）

随着这一过程进行，$x_{\min} = \gamma^{-1/\rho}$ 降至零。

```{code-cell} ipython3
xs = np.array([0.5, 1.0, 2.0, 4.0])
print(f"{'power law':12}", np.round((γ * xs)**(γ / (1 - γ)), 5))
for ρ in (-0.1, -0.01, -0.001, -1e-5):
    print(f"ρ = {ρ:<8}", np.round(ces_capacity(xs, γ, β, ρ), 5))
```

这两篇论文的分布都具有非常肥大的上尾。

在 Houthakker 的分布中，总产能是无穷的；而在 Levhari 的分布中，总产能是有限的，但平均需求量却是无穷的，正如 {ref}`ha_ex_tail` 要求你证明的那样。

## 后续研究

{cite:t}`Levhari1968` 在开篇指出，Houthakker 的论文曾被忽视，正如 {cite:t}`Solow1967` 在其生产理论综述中所指出的那样。

其核心思想此后在若干方向上得到了发展。

{cite:t}`Johansen1972` 将 Houthakker 的构造方法作为短期行业生产函数理论的基础，该理论由固定比例生产单位的投入系数的产能分布推导而来。

他将其与一种“黏土-黏土”式世代模型相结合，其中对新单位的投资会随时间推移改变这一分布。

{cite:t}`Hildenbrand1981` 研究了多投入多产出情形下的这一构造方法，并证明行业的短期生产集是一个**带域体（zonoid）**。

在有限多个单元的情形下，这个集合是若干线段之和，每条线段都从原点连到某个单元满负荷时的投入产出向量；在连续多个单元的情形下，它则是这类和式的极限。

{cite:t}`Jones2005` 把各种构思当作技术来处理，在其分析模型中采用列昂惕夫式处理方式，其中劳动和资本系数分别取自独立的帕累托分布。

他所得到的全球生产函数是已发现各种技术的上包络线，而不是对受产能约束的各单元求和，因此它具有规模报酬不变的性质。

他证明了其期望值是科布-道格拉斯型的，且长期技术变化是劳动增进型的，并将其构造方法与 Houthakker 的方法作了对比。

{cite:t}`Lagos2006` 将 Houthakker 的加总方法与一个存在摩擦的劳动力市场相结合，在该市场中工作岗位不断被创造和摧毁，因此劳动力市场政策会通过改变哪些单元在运作而影响所测得的全要素生产率。

## 结语

Houthakker 的构造方法把总生产函数混在一起的两件事分开了：每个单元的技术，以及各单元间技术的分布。

没有任何单元能够用一种投入替代另一种投入，然而整个行业却可以，因为相对价格的变化会改变哪些单元在运作。

总生产函数的形式是分布的一种性质：帕累托分布给出科布-道格拉斯函数，而在只有一种可变投入的情形下，Levhari 的反演方法能为任何满足 $F(0) = 0$ 的光滑、递增、严格凹的生产函数找到背后的单元分布。

由于总生产函数依赖于固定投入在各单元间的分布方式，它描述的是短期情形，并且会随着进入、退出或投资改变这一分布而发生变化。

这正是 Houthakker 告诫人们不要把固定投入作为该函数的另一个自变量加进去的原因，也是他在论文结尾处要求提供关于各企业和各机器（例如棉纺锭）之间投入产出比分布数据的原因。

## 练习

```{exercise}
:label: ha_ex_homogeneous

我们已经看到，Houthakker 的分布 {eq}`ha_pareto` 在 $a_1$ 和 $a_2$ 上分别按幂律进行标度。

本练习表明，仅在 $(a_1, a_2)$ 上联合标度，不足以得到科布-道格拉斯型的总生产函数。

考虑

$$
\varphi(a_1, a_2) = (a_1 + a_2)^{\kappa - 2}, \qquad \kappa > 0,
$$

它是 $\kappa - 2$ 次齐次的。

设 $s_i = p_i X_i / X_0$ 为投入 $i$ 的份额（在 Houthakker 的情形中，$s_i = \gamma_i$）。

1. 证明 $X_0(\lambda p) = \lambda^{-\kappa} X_0(p)$ 以及 $X_i(\lambda p) = \lambda^{-\kappa - 1} X_i(p)$，并由此得出 $F$ 是 $\kappa/(1 + \kappa)$ 次齐次的，且 $s_1 + s_2 = \kappa/(1 + \kappa)$。
2. 作变量代换 $r = a_1 + a_2$、$u = a_1 / r$，把 {eq}`ha_totals` 化为关于 $u \in [0, 1]$ 的一维积分。
3. 当 $p_1/p_2$ 在 $0.1$ 到 $10$ 之间变化时，对 $\kappa = 1$、$2$ 和 $4$ 分别计算份额 $s_1$。对于哪个 $\kappa$，$F$ 是科布-道格拉斯型的？对于齐次生产函数，两种投入之间的替代弹性为 $\sigma_{12} = 1 - d \log(s_1/s_2) / d \log(p_1/p_2)$。计算它，并解释为何在 $\kappa = 1$ 时它小于 1，而在 $\kappa = 4$ 时它大于 1。提示：沿每一条直线 $a_1 + a_2 = r$，比较 $\varphi$ 与同样是该次数齐次的 Houthakker 分布 $a_1^{\kappa/2 - 1} a_2^{\kappa/2 - 1}$，二者如何在投入组合 $u = a_1/r$ 上分配产能。
```

```{solution-start} ha_ex_homogeneous
:class: dropdown
```

以下给出一种解法。

对于第 1 部分，在价格 $\lambda p$ 下的 {eq}`ha_totals` 中代入 $a = \lambda^{-1} \tilde a$。

区域 $a \cdot \lambda p < 1$ 变为 $\tilde a \cdot p < 1$，$da = \lambda^{-2} d\tilde a$，且 $\varphi(\lambda^{-1} \tilde a) = \lambda^{2 - \kappa} \varphi(\tilde a)$，因此 $X_0(\lambda p) = \lambda^{-\kappa} X_0(p)$。

额外的因子 $a_i = \lambda^{-1} \tilde a_i$ 给出 $X_i(\lambda p) = \lambda^{-\kappa - 1} X_i(p)$。

因此将价格缩放 $\lambda$ 倍，会使投入缩放 $\lambda^{-\kappa - 1}$ 倍，产出缩放 $\lambda^{-\kappa} = (\lambda^{-\kappa - 1})^{\kappa/(1 + \kappa)}$ 倍，因此 $F$ 是 $\kappa/(1 + \kappa)$ 次齐次的。

由于边际产量等于价格，欧拉定理给出 $p_1 X_1 + p_2 X_2 = \frac{\kappa}{1 + \kappa} X_0$。

对于第 2 部分，$a_1 = r u$ 和 $a_2 = r(1 - u)$ 的雅可比行列式为 $r$，价格三角形变为 $r < R(u) \equiv 1/(u p_1 + (1 - u) p_2)$。

对 $r$ 积分得到

$$
X_0 = \frac{1}{\kappa} \int_0^1 R(u)^\kappa \, du,
\qquad
X_1 = \frac{1}{\kappa + 1} \int_0^1 u \, R(u)^{\kappa + 1} \, du,
\qquad
X_2 = \frac{1}{\kappa + 1} \int_0^1 (1 - u) \, R(u)^{\kappa + 1} \, du .
$$

```{code-cell} ipython3
def homogeneous_totals(p, κ):
    p1, p2 = p
    R = lambda u: 1 / (u * p1 + (1 - u) * p2)
    X0 = integrate.quad(lambda u: R(u)**κ, 0, 1)[0] / κ
    X1 = integrate.quad(lambda u: u * R(u)**(κ + 1), 0, 1)[0] / (κ + 1)
    X2 = integrate.quad(lambda u: (1 - u) * R(u)**(κ + 1), 0, 1)[0] / (κ + 1)
    return X0, X1, X2


κ = 4.0
p = (0.5, 1.0)
print("one-dimensional:", np.round(homogeneous_totals(p, κ), 10))
print("two-dimensional:", np.round(totals_by_quadrature(
    p, lambda a1, a2: (a1 + a2)**(κ - 2)), 10))
```

对于第 3 部分，我们在价格比率的网格上计算份额和替代弹性。

我们还用一个小的中心差分，在等价格处计算 $\sigma_{12}$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Shares and elasticities under homogeneous $\varphi$
    name: fig-ha-homogeneous
---
def log_share_ratio(r, κ):
    """log(s1/s2) at prices (r, 1)."""
    X0, X1, X2 = homogeneous_totals((r, 1.0), κ)
    return np.log(r * X1 / X2)


ratios = np.geomspace(0.1, 10, 41)
h = 1e-4

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for κ in (1.0, 2.0, 4.0):
    shares = []
    for r in ratios:
        X0, X1, X2 = homogeneous_totals((r, 1.0), κ)
        shares.append((r * X1 / X0, X2 / X0))
    s1, s2 = np.array(shares).T
    σ12 = 1 - np.gradient(np.log(s1 / s2), np.log(ratios), edge_order=2)
    σ12_equal = 1 - (log_share_ratio(np.exp(h), κ)
                     - log_share_ratio(np.exp(-h), κ)) / (2 * h)
    print(f"κ = {κ}: s1 + s2 in [{(s1 + s2).min():.4f}, "
          f"{(s1 + s2).max():.4f}], κ/(1+κ) = {κ / (1 + κ):.4f}; "
          f"σ12 at p1 = p2: {σ12_equal:.4f}")
    axes[0].plot(ratios, s1, lw=2, label=rf"$\kappa = {κ:g}$")
    axes[1].plot(ratios, σ12, lw=2, label=rf"$\kappa = {κ:g}$")
axes[0].set_ylabel("share of input 1")
axes[1].set_ylabel("elasticity of substitution")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_1 / p_2$")
    ax.legend()
plt.tight_layout()
plt.show()
```

正如第 1 部分所预测的那样，对每个价格比率，各份额之和都等于 $\kappa/(1 + \kappa)$。

只有当 $\kappa = 2$ 时，投入 1 的份额才是常数（为 $1/3$），因此只有在这种情况下 $F$ 才是科布-道格拉斯型的。

这并非巧合：当 $\kappa = 2$ 时，$\varphi$ 是常数，这正是 $\alpha_1 = \alpha_2 = 1$ 时的 Houthakker 分布 {eq}`ha_pareto`。

当 $\kappa = 1$ 时，投入 1 的份额随其相对价格的上升而上升，因此 $\sigma_{12} < 1$。

当 $\kappa = 4$ 时，该份额随其相对价格的上升而下降，因此 $\sigma_{12} > 1$。

在等价格处，这两个弹性分别为 $2/3$ 和 $5/3$。

要理解原因，回顾一下行业层面的替代效应是由相对价格的变化使各单元开关所导致的。

那些投入组合极端（即靠近坐标轴）的单元被开关，对行业投入组合的改变最大。

沿每一条直线 $a_1 + a_2 = r$，$\varphi$ 在组合 $u = a_1/r$ 上均匀地分配产能，而 Houthakker 分布 $a_1^{\kappa/2 - 1} a_2^{\kappa/2 - 1}$ 则按与 $u^{\kappa/2 - 1} (1 - u)^{\kappa/2 - 1}$ 成比例的方式分配产能。

当 $\kappa < 2$ 时，这个权重在 $u = 0$ 和 $u = 1$ 附近最大，因此 $\varphi$ 中投入组合极端的单元相对较少，替代效应弱于科布-道格拉斯情形。

当 $\kappa > 2$ 时，这个权重在 $u = 0$ 和 $u = 1$ 附近趋于零，因此 $\varphi$ 中投入组合极端的单元相对较多，替代效应更强。

这一点可以精确地表达出来。

假设 $\varphi(a) = r^{\kappa - 2} w(u)$，其中权重函数 $w$ 关于 $u = 1/2$ 对称，并设 $p_1 = e^\varepsilon$、$p_2 = 1$。

则 $R(u)^{\kappa + 1} \approx 1 - (1 + \kappa) u \varepsilon$，记 $\mathbb{E}$ 为按正比于 $w$ 的权重对 $u$ 求平均，于是 $\mathbb{E} u = 1/2$，

$$
\log \frac{X_1}{X_2} \approx -2 (1 + \kappa) \varepsilon \left( \mathbb{E} u^2 - \mathbb{E} [u(1 - u)] \right)
= -4 (1 + \kappa) \varepsilon \operatorname{Var}(u) .
$$

由于 $s_1/s_2 = e^\varepsilon X_1/X_2$，在等价格处

$$
\sigma_{12} = 4 (1 + \kappa) \operatorname{Var}(u) .
$$

对于我们的 $\varphi$，$w$ 是均匀的，$\operatorname{Var}(u) = 1/12$，这给出 $\sigma_{12} = (1 + \kappa)/3$，与计算出的 $2/3$ 和 $5/3$ 相符。

对于 Houthakker 的分布，$w$ 是方差为 $1/(4(1 + \kappa))$ 的 Beta$(\kappa/2, \kappa/2)$ 密度，这给出 $\sigma_{12} = 1$。

在等价格处，替代弹性与各单元投入组合的离散程度成正比。

```{solution-end}
```

```{exercise}
:label: ha_ex_elasticity

回到只有一种可变投入的情形。

将总生产函数写成 $F(L) = \tilde F(L, 1)$，其中 $\tilde F(L, K) = K F(L/K)$ 是一次齐次的，并令

$$
s(L) = \frac{L F'(L)}{F(L)},
\qquad
\sigma(L) = \frac{F'(L) \left[ F(L) - L F'(L) \right]}{-L F(L) F''(L)}
$$

分别为劳动在产出中的份额，以及劳动与固定投入之间的替代弹性。

1. 利用 {eq}`ha_inversion` 证明产能分布关于边际单元需求量的弹性 $d \log G / d \log x = x g(x)/G(x)$ 为

   $$
   \frac{x \, g(x)}{G(x)} = \sigma(L) \, \frac{s(L)}{1 - s(L)}, \qquad \text{其中 } F'(L) = 1/x .
   $$

2. 利用这一恒等式证明，科布-道格拉斯函数 $F(L) = C L^\gamma$ 要求 $G(x) \propto x^{\gamma/(1 - \gamma)}$。然后直接由 {eq}`ha_QL` 证明，任意幂律 $G(x) = k x^\alpha$ 都会产生一个科布-道格拉斯函数。因此，在只有一种可变投入的情形下，Houthakker 分布是唯一能产生科布-道格拉斯函数的分布。
3. 对 CES 函数 {eq}`ha_ces` 对此恒等式进行数值检验。
```

```{solution-start} ha_ex_elasticity
:class: dropdown
```

以下给出一种解法。

对于第 1 部分，记 $x(L) = 1/F'(L)$，使得 {eq}`ha_inversion` 可写成 $G(x(L)) = F(L)$。

取对数并对 $L$ 求导：

$$
\frac{x g(x)}{G(x)} \cdot \frac{d \log x}{dL} = \frac{F'}{F},
\qquad
\frac{d \log x}{dL} = -\frac{F''}{F'} .
$$

因此

$$
\frac{x g(x)}{G(x)} = \frac{(F')^2}{-F F''} .
$$

由 $\sigma$ 的定义可得 $-F F'' = F'(F - L F') / (\sigma L)$，代入后得到

$$
\frac{x g(x)}{G(x)} = \frac{\sigma L F'}{F - L F'} = \sigma \frac{s}{1 - s} .
$$

对于第 2 部分，科布-道格拉斯函数在每个 $L$ 处都有 $\sigma = 1$ 和 $s = \gamma$，且其边际产量 $F'(L) = C \gamma L^{\gamma - 1}$ 在 $L$ 取遍 $(0, \infty)$ 时取遍所有正值。

该恒等式于是说明，对每个 $x > 0$ 都有 $x g(x)/G(x) = \gamma/(1 - \gamma)$，因此 $\log G$ 是 $\log x$ 的线性函数，斜率为 $\gamma/(1 - \gamma)$。

反过来，若 $G(x) = k x^\alpha$，则 $g(x) = k \alpha x^{\alpha - 1}$，{eq}`ha_QL` 给出 $Q = k x^\alpha$ 和 $L = k \alpha x^{\alpha + 1}/(\alpha + 1)$。

消去 $x$，

$$
Q = k^{1/(1 + \alpha)} \left( \frac{1 + \alpha}{\alpha} L \right)^{\alpha/(1 + \alpha)},
$$

这是科布-道格拉斯型的，其中 $\gamma = \alpha/(1 + \alpha)$。

对于第 3 部分，CES 函数在每个 $L$ 处都有 $\sigma = 1/(1 - \rho)$，而劳动份额则随之变化。

```{code-cell} ipython3
for ρ in (-0.5, -1.0, -2.0):
    σ = 1 / (1 - ρ)
    for L in (0.1, 1.0, 10.0):
        F_L = ces(L, γ, β, ρ)
        mp = ces_marginal_product(L, γ, β, ρ)
        x, s = 1 / mp, L * mp / F_L
        lhs = x * ces_density(x, γ, β, ρ) / ces_capacity(x, γ, β, ρ)
        print(f"ρ = {ρ:5}, L = {L:5}: x g/G = {lhs:9.6f}, "
              f"σ s/(1 - s) = {σ * s / (1 - s):9.6f}")
```

两边的结果相符。

用 {eq}`ha_ces_density` 除以 {eq}`ha_ces_dist` 得到 $x g(x)/G(x) = \sigma v/(1 - v)$，其中 $v = (x_{\min}/x)^{1 - \sigma}$，因此该恒等式表明，当边际单元需要 $x$ 单位劳动时，劳动份额为 $s = (x_{\min}/x)^{1 - \sigma}$。

当 $x \downarrow x_{\min}$ 时，劳动份额趋近于 1，而 $G$ 的弹性无限增大。

```{solution-end}
```

```{exercise}
:label: ha_ex_tail

Levhari 指出，分布 {eq}`ha_ces_dist` 没有有限的均值。

1. 证明归一化分布 $P(x) = G(x)/\beta^{1/\rho}$ 满足

   $$
   1 - P(x) \approx \frac{\sigma}{1 - \sigma} \left( \frac{x_{\min}}{x} \right)^{1 - \sigma}
   \qquad \text{当 } x \to \infty \text{时},
   $$

   因此，按照 {doc}`intro:heavy_tails` 中的意义，它的上尾是一个尾指数为 $1 - \sigma < 1$ 的帕累托尾，其均值是无穷的。
2. 对于任意一个总生产函数，如果它在每个 $L$ 处的边际产量都为正，且总产能有限，请给出一个经济学理由说明为什么平均需求量必定是无穷的。
3. 用 `draw_requirements` 对 $\rho = -0.5$、$-1$ 和 $-2$ 各抽取 $100{,}000$ 个需求量。在双对数坐标轴上绘制经验生存函数 $1 - P$，并与第 1 部分中的近似表达式一同展示。
```

```{solution-start} ha_ex_tail
:class: dropdown
```

以下给出一种解法。

对于第 1 部分，设 $v = (x_{\min}/x)^{1 - \sigma}$，当 $x \to \infty$ 时它趋于零。

于是 $P(x) = (1 - v)^{\sigma/(1 - \sigma)} = 1 - \frac{\sigma}{1 - \sigma} v + O(v^2)$，这就给出了近似式。

一个生存函数以 $x^{-\zeta}$ 速率下降的分布，只有当 $\zeta > 1$ 时才具有有限均值。

此处 $\zeta = 1 - \sigma < 1$，因此均值是无穷的。

对于第 2 部分，该行业的就业量为 $L = H(x)$，即所有需求量至多为 $x$ 的单元所使用的劳动量。

如果边际产量 $F'(L) = 1/x$ 在每个 $L$ 处都为正，那么无论就业水平多大，都会在某个有限的 $x$ 处被达到。

因此当 $x \to \infty$ 时 $H(x) \to \infty$：如果行业中所有单元都在运作，它们将雇用无穷多的劳动。

在总产能有限的情况下，这意味着平均需求量是无穷的。

Levhari 是这样表述的：随着工资降至零，就业量会无限上升，因为劳动的边际产量始终为正。

对于第 3 部分，我们将经验生存函数与第 1 部分的近似值（以虚线表示）进行比较。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Upper tails of Levhari's distributions
    name: fig-ha-tails
---
rng = np.random.default_rng(3)
N = 100_000

fig, ax = plt.subplots()
for ρ, color in zip((-0.5, -1.0, -2.0), ("C0", "C1", "C2")):
    σ = 1 / (1 - ρ)
    x_min = γ**(-1 / ρ)
    t = np.sort(draw_requirements(N, γ, ρ, rng))
    survival = 1 - np.arange(1, N + 1) / N
    ax.loglog(t[:-1], survival[:-1], color=color, lw=2, alpha=0.6,
              label=rf"draws, $\sigma = {σ:.2f}$")
    x_grid = np.geomspace(10 * x_min, t[-2], 50)
    ax.loglog(x_grid, σ / (1 - σ) * (x_min / x_grid)**(1 - σ),
              color=color, lw=1, ls="--")
ax.set_xlabel("labor requirement $x$")
ax.set_ylabel(r"$1 - P(x)$")
ax.legend()
plt.show()
```

在双对数坐标轴下，生存函数变成斜率为 $-(1 - \sigma)$ 的直线，分别为 $-1/3$、$-1/2$ 和 $-2/3$。

替代弹性越小，尾部越薄，但即便 $\sigma \to 0$，尾指数仍然保持在 1 以下。

```{solution-end}
```