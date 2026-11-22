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
  title: 指数族与最大熵
  headings:
    Overview: 概述
    Models and families: 模型与族
    Models and families::Models, families, and parameters: 模型、族与参数
    Models and families::Score and Fisher information: 得分与费希尔信息
    Models and families::Sufficient statistics: 充分统计量
    Invariants: 不变量
    Invariants::Distances between parameter vectors: 参数向量之间的距离
    Invariants::What changes and what does not: 什么会变，什么不会变
    Invariants::Moving mass and comparing heights: 移动质量与比较高度
    Invariants::Statistics and sufficiency: 统计量与充分性
    The exponential family: 指数族
    The exponential family::Definition: 定义
    The exponential family::Sufficient statistics of fixed dimension: 维数固定的充分统计量
    The exponential family::The log-partition function generates moments: 对数配分函数生成矩
    Two coordinate systems: 两套坐标系
    Two coordinate systems::Expectation parameters: 期望参数
    Two coordinate systems::Legendre duality: 勒让德对偶
    Two coordinate systems::The conjugate is negative entropy: 共轭函数是负熵
    Maximum entropy: 最大熵
    Maximum entropy::A die: 一个骰子
    Maximum entropy::The solution is a model in an exponential family: 解是指数族中的一个模型
    Maximum entropy::Shadow prices: 影子价格
    Maximum entropy::The normal distribution: 正态分布
    Maximum entropy::Other moment conditions: 其他矩约束
    Maximum entropy::Logit choice probabilities: 逻辑选择概率
    Bregman divergences and KL divergence: 布雷格曼散度与 KL 散度
    Bregman divergences and KL divergence::Bregman divergence: 布雷格曼散度
    Bregman divergences and KL divergence::KL divergence is a Bregman divergence: KL 散度是一种布雷格曼散度
    Bregman divergences and KL divergence::Fisher information is the curvature of KL divergence: 费希尔信息是 KL 散度的曲率
    Bregman divergences and KL divergence::The Fisher–Rao distance: 费希尔-拉奥距离
    Estimation: 估计
    Estimation::Maximum likelihood matches moments: 最大似然匹配矩
    Estimation::Finite-sample behavior of the two estimators: 两种估计量的有限样本行为
    Concluding remarks: 结语
    Exercises: 练习
---

(exponential_families)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 指数族与最大熵

```{contents} Contents
:depth: 2
```

## 概述

本讲座及其续篇 {doc}`information_projections` 描述了**信息几何**的内容，这是一个运用凸分析和微分几何工具来研究概率分布的参数化族的领域。

信息几何中的一个关键概念是概率分布的**指数族**。

要理解什么是指数族，我们必须先理解概率分布族的参数的**充分统计量**的定义。

充分统计量包含了样本中关于参数的全部信息。

```{note}
{cite:t}`Darmois1935`、{cite:t}`Koopman1936` 和 {cite:t}`Pitman1936` 描述了光滑性条件，在这些条件下，若一个密度族的可能结果集合不依赖于其参数，且各种样本量下都存在维度相同的有限维充分统计量，则该密度族是一个指数族。
```

指数族包括伯努利族、指数族、泊松族、正态族、伽马族以及贝塔族等。

几何学研究**不变量**，也就是在我们改变坐标系时不变的性质。

为了刻画这类**不变量**，我们要问：当我们重新标记一个族的参数或对数据进行变换时，哪些对象发生变化，哪些保持不变。

当对参数或数据施加适当的变换时，KL 散度不会改变。

然而，当我们重新标记参数时，参数向量之间的欧几里得距离*确实*会改变。

当我们变换数据时，微分熵*确实*会改变。

用于定义指数族的一个关键工具是一个称为**对数配分函数**的凸函数。

对数配分函数生成了：

* 一个族中模型的两套坐标系，称为**自然参数**和**期望参数**；
* 该族充分统计量的均值和协方差；
* 费希尔信息矩阵；
* 当该族的基测度为常数时，通过勒让德变换得到的每个模型的熵；以及
* 通过**布雷格曼散度**得到的任意两个模型之间的库尔贝克-莱布勒（KL）散度。

```{note}
这里的"坐标系"是"族的参数化"的同义词。
```

指数族中的每个模型都回答了 {cite:t}`Jaynes1957` 提出的一个问题：在所有具有若干给定统计量均值的分布中，哪个概率分布表达的信息最少（以香农熵衡量）？

一个特殊情形刻画了正态分布：在实数轴上所有具有给定均值和方差的密度中，正态密度具有最大的熵。

**费希尔信息矩阵**为模型路径赋予长度，并定义了模型之间的距离——**费希尔-拉奥距离**。

费希尔-拉奥距离在我们重新标记参数或变换数据时不会改变。

对于正态族，它是双曲几何中的一种距离。

最优传输的**沃瑟斯坦距离**则是衡量将一个模型转化为另一个模型所需移动的概率质量有多少；它在重新标记参数时不会改变，但在变换数据时会改变。

```{note}
经济学家在其他名称下已经使用了这些工具中的若干种。

对数配分函数及其**勒让德变换**之间的关系，正如利润函数和生产函数之间的关系一样。

对数配分函数导数的公式是霍特林引理的一个版本。

离散选择的多项逻辑模型的**包容值**（inclusive value）就是一个对数配分函数。

在相差一个常数的意义下，它是最优选择的期望效用。

它的勒让德变换是负熵。
```

续篇 {doc}`information_projections` 利用这些工具研究概率分布在分布族上的投影，并应用于最大似然、鲁棒控制、投入产出表的更新以及最优传输。

本讲座建立在 {doc}`divergence_measures` 的基础上，后者描述了熵、交叉熵和 KL 散度；我们借用了其中的部分记号。

{doc}`mle` 和 {doc}`multivariate_normal` 提供了额外的背景知识。

{cite:t}`amari_nagaoka2000`、{cite:t}`Nielsen2020` 以及 {cite:t}`CoverThomas2006` 的第 2、8、11、12 章讨论了本讲座及其续篇的内容。

我们的主题顺序遵循 {cite:t}`NielsenMLSS2015` 讲座幻灯片的安排。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import logsumexp, digamma, polygamma, gammaln
from scipy.stats import norm, poisson, laplace, logistic, uniform, chi2
from scipy.stats import lognorm, cauchy
from scipy.stats import gamma as gamma_dist
```

## 模型与族

### 模型、族与参数

**模型**是取值于集合 $\mathcal X$ 中的随机变量 $X$ 的概率分布。

我们用密度函数或概率质量函数来描述一个模型。

**参数族**是一组模型的集合

$$
\mathcal P = \{ p(\cdot\,; \theta) : \theta \in \Theta \} ,
$$

其中**参数空间** $\Theta$ 是 $\mathbb R^D$ 的一个子集。

映射 $\theta \mapsto p(\cdot\,; \theta)$ 是 $\mathcal P$ 的一个**参数化**；它为 $\mathcal P$ 中的模型赋予坐标 $\theta$。

统计学家常把整个族称为"统计模型"。

我们不这样称呼。

在本讲座中，模型是指在单一参数向量 $\theta$ 处的单一分布 $p(\cdot\,; \theta)$，而族是指由遍历 $\Theta$ 的参数向量索引的一组模型。

以下是六个族。

* **伯努利族**：$\mathcal X = \{0, 1\}$，$p(x; \pi) = \pi^x (1 - \pi)^{1 - x}$，$\Theta = (0, 1)$。
* **泊松族**：$\mathcal X = \{0, 1, 2, \ldots\}$，$p(x; \lambda) = \lambda^x e^{-\lambda} / x!$，$\Theta = (0, \infty)$。
* **正态族**：$\mathcal X = \mathbb R$，$p(x; \mu, \sigma) = (2 \pi \sigma^2)^{-1/2} \exp\left(-(x - \mu)^2 / (2 \sigma^2)\right)$，$\Theta$ 是半平面 $\mathbb R \times (0, \infty)$。
* **均匀族**：$\mathcal X = (0, \infty)$，当 $0 < x < \theta$ 时 $p(x; \theta) = 1/\theta$，当 $x \geq \theta$ 时 $p(x; \theta) = 0$，$\Theta = (0, \infty)$。
* **柯西位置族**：$\mathcal X = \mathbb R$，$p(x; \mu) = 1 / \left(\pi (1 + (x - \mu)^2)\right)$，$\Theta = \mathbb R$。
* **正态混合族**：$\mathcal X = \mathbb R$，$p(x; w, \mu_1, \mu_2) = w \, \varphi(x - \mu_1) + (1 - w) \, \varphi(x - \mu_2)$，其中 $\varphi$ 是 $N(0, 1)$ 的密度，$\Theta = [0, 1] \times \mathbb R^2$。

在伯努利族中 $\pi$ 是一个概率；在正态族和柯西族中它是 $3.14159\ldots$。

如果不同的参数向量给出不同的模型，则称一个参数化是**可识别的**。

前五个族的参数化是可识别的。

混合族的参数化则不是：$(w, \mu_1, \mu_2)$ 与 $(1 - w, \mu_2, \mu_1)$ 给出相同的模型，当 $\mu_1 = \mu_2$ 时每个 $w$ 给出相同的模型，当 $w = 0$ 时每个 $\mu_1$ 给出相同的模型。

我们假设参数化是可识别且光滑的，并且 $D$ 个函数 $\partial p(\cdot\,; \theta) / \partial \theta_i$ 是线性无关的。

这样，$\mathcal P$ 就是 $\mathcal X$ 上所有概率分布集合中的一个 $D$ 维曲面，即**流形**，而 $\theta$ 是 $\mathcal P$ 上的一个坐标系。

哈罗德·霍特林（Harold Hotelling）在 1930 年和 {cite:t}`Rao1945` 都将一个参数族作为这样的空间来研究。

一个族可以有很多坐标系。

我们可以用 $(\mu, \sigma)$、$(\mu, \sigma^2)$，或者用均值对 $(E X, E X^2) = (\mu, \mu^2 + \sigma^2)$ 来描述正态族。

### 得分与费希尔信息

模型的**得分**（score）是其对数密度关于参数向量的梯度 $\nabla_\theta \log p(X; \theta)$。

当我们可以在积分号下求导时，得分的均值为零：

$$
E_\theta \left[ \nabla_\theta \log p(X; \theta) \right] = \int \nabla_\theta \, p(x; \theta) \, dx = \nabla_\theta \int p(x; \theta) \, dx = 0 .
$$

得分的协方差矩阵是**费希尔信息矩阵**

$$
I(\theta) = E_\theta \left[ \nabla_\theta \log p(X; \theta) \, \nabla_\theta \log p(X; \theta)^\top \right] .
$$ (ef_fisher)

规模为 $n$ 的独立同分布样本具有费希尔信息 $n I(\theta)$。

克拉美-拉奥不等式表明，从这样一个样本得到的 $\theta$ 的无偏估计量的协方差矩阵，超出 $I(\theta)^{-1} / n$ 的部分是一个半正定矩阵。

对于坐标 $(\mu, \sigma)$ 下的正态族，

$$
I(\mu, \sigma) = \frac{1}{\sigma^2} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} .
$$ (ef_fisher_normal)

### 充分统计量

设 $\mathbf X = (X_1, \ldots, X_n)$ 是来自族 $\mathcal P$ 中模型 $p(\cdot\,; \theta)$ 的独立同分布样本。

**统计量**是样本的一个不涉及 $\theta$ 的函数 $s(\mathbf X)$。

如果给定 $s(\mathbf X)$ 时 $\mathbf X$ 的条件分布对每个 $\theta \in \Theta$ 都相同，则称统计量 $s$ 对族 $\mathcal P$ 中的 $\theta$ 是**充分的**。

一旦我们知道 $s(\mathbf X)$，样本的其余部分就不再告诉我们关于 $\theta$ 的任何额外信息。

{cite:t}`Fisher1922` 引入了充分性这一概念。

费希尔-奈曼**分解定理**表明，当且仅当联合密度可以分解为如下形式时，$s$ 才是充分的：

$$
\prod_{i=1}^n p(x_i; \theta) = g\left(s(\mathbf x); \theta\right) a(\mathbf x)
$$ (ef_factorization)

其中 $g$ 和 $a$ 是某些函数，且 $a$ 不依赖于 $\theta$。

由 {eq}`ef_factorization`，两个具有相同 $s$ 值的样本具有成比例的似然函数。

它们给出相同的最大似然估计，并且在任何先验下给出相同的后验分布。

充分性是一个统计量相对于某个族而言的性质。

我们将在定义指数族之后给出充分统计量的例子。

(ef_invariants)=
## 不变量

费利克斯·克莱因（Felix Klein）1872 年的埃尔兰根纲领用一组变换来定义一种几何，并研究在这些变换下保持不变的图形性质。

信息几何研究模型和族在三类变换下保持不变的性质。

1. **重新参数化**将 $\theta$ 替换为 $\xi = \phi(\theta)$，其中 $\phi$ 光滑、一一对应且具有光滑的逆映射。每个模型保持不变；只有其坐标发生变化。
2. **对数据的可逆变换**将 $X$ 替换为 $Y = f(X)$，其中 $f$ 光滑且一一对应。每个关于 $X$ 的模型变为关于 $Y$ 的模型，而族保持其参数空间不变。
3. **统计量**将数据替换为数据的一个函数。更一般地，{doc}`blackwell_kihlstrom` 中研究的那种**信息减损**（garbling）将数据替换为一个随机变量，其给定数据的条件分布不依赖于 $\theta$。统计量和信息减损都可能丢弃信息。

### 参数向量之间的距离

我们通常用均值 $\mu$ 和标准差 $\sigma$ 来描述一个正态模型。

点对 $(\mu, \sigma)$ 是半平面中的一个点，但两个这样的点之间的普通距离几乎无法说明从数据中区分这两个模型有多难。

考虑四个正态模型：

* $A = N(0, 0.2^2)$ 和 $B = N(1, 0.2^2)$；以及
* $C = N(0, 2^2)$ 和 $D = N(1, 2^2)$。

在 $(\mu, \sigma)$ 半平面中，$A$ 到 $B$ 的距离与 $C$ 到 $D$ 的距离相同。

但 $A$ 和 $B$ 几乎不重叠，而 $C$ 和 $D$ 大量重叠。

我们把 $D_{KL}(p \| q) = \int p(x) \log (p(x) / q(x)) \, dx$ 读作"$p$ 相对于 $q$ 的 KL 散度"。

两个具有相同标准差 $\sigma$ 的正态模型之间的 KL 散度是 $(\mu_1 - \mu_2)^2 / (2 \sigma^2)$，对于 $A$ 和 $B$ 它等于 $12.5$，对于 $C$ 和 $D$ 它等于 $0.125$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Equal parameter distances, unequal KL divergences
    name: fig-ef-four-normals
---
x = np.linspace(-6, 7, 1000)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, σ, names in zip(axes, (0.2, 2.0), (("A", "B"), ("C", "D"))):
    for μ, name in zip((0.0, 1.0), names):
        ax.plot(x, norm.pdf(x, μ, σ), lw=2, label=f"{name}: N({μ:g}, {σ:g}²)")
    ax.set_title(rf"$\sigma = {σ:g}$, KL = {1 / (2 * σ**2):.3g}")
    ax.set_xlabel("$x$")
    ax.legend()
plt.tight_layout()
plt.show()
```

概率分布的几何学应当通过分布之间的可区分程度来衡量距离。

对于指数族，一个凸函数就能生成这样的几何结构。

### 什么会变，什么不会变

在下表中，$J = \partial \theta / \partial \xi$ 是重新参数化 $\phi$ 的逆映射的雅可比矩阵，$f'$ 是标量数据变换 $f$ 的导数。

| 对象 | 重新参数化 $\theta \to \xi = \phi(\theta)$ | 数据变换 $x \to y = f(x)$ |
|---|---|---|
| 两个模型之间的 KL 散度 | 不变 | 不变 |
| 费希尔信息矩阵 | 变为 $J^\top I(\theta) J$ | 不变 |
| 参数向量之间的欧几里得距离 | 改变 | 不变 |
| 两个模型之间的沃瑟斯坦距离 | 不变 | 改变 |
| 某点处模型的密度 | 不变 | 除以 $\lvert f'(x) \rvert$ |
| 模型的微分熵 | 不变 | 改变 $E \log \lvert f'(X) \rvert$ |
| 最大似然估计 | 变为 $\phi(\hat \theta)$ | 不变 |
| 估计量的无偏性 | 可能丧失 | 不变 |
| 平坦先验密度 | 在 $\xi$ 下通常不再平坦 | 不变 |
| 杰弗里斯先验 | 不变 | 不变 |

在 $y = f(x)$ 变换下，每个密度都会获得因子 $1 / \lvert f'(x) \rvert$，而这个因子不依赖于 $\theta$。

似然比和得分不会改变，因此 KL 散度、费希尔信息矩阵以及最大似然估计也都不会改变。

微分熵会改变，因为它是对数密度本身的期望：以美分而非美元来衡量收入，会使它增加 $\log 100$。

在重新参数化下，链式法则将得分乘以 $J^\top$，因此费希尔信息矩阵变为 $J^\top I(\theta) J$。

二次型 $d\theta^\top I(\theta) \, d\theta$ 不会改变，因为 $d\theta = J \, d\xi$。

{ref}`ef_fisher_rao` 利用这个二次型定义了不依赖于坐标的模型之间的距离。

最大似然估计将其坐标从 $\hat \theta$ 变为 $\phi(\hat \theta)$，但指代的是同一个模型。

当 $\phi$ 非线性时，由于 $E \phi(\hat \theta) \neq \phi(E \hat \theta)$，无偏性可能丧失；{ref}`ef_estimation` 给出了一个例子。

$\sigma$ 的平坦先验在 $\sigma^2$ 下并不平坦。

{cite:t}`jeffreys1946invariant` 提出的先验具有正比于 $\sqrt{\det I(\theta)}$ 的密度。

在坐标 $\xi$ 下，其密度正比于 $\sqrt{\det (J^\top I J)} = \lvert \det J \rvert \sqrt{\det I}$，这正是变量替换公式所要求的，因此杰弗里斯先验在每一种坐标系下都对每一组模型赋予相同的概率。

{ref}`ef_ex_jeffreys` 给出了伯努利情形的详细推导。

我们为两个正态模型 $N(1, 1.5^2)$ 和 $N(-0.5, 1)$ 计算了这些对象中的若干个。

我们使用四种坐标系：$(\mu, \sigma)$、$(\mu, \sigma^2)$、均值对 $(\mu, \mu^2 + \sigma^2)$，以及 $(\mu / \sigma^2, -1 / (2 \sigma^2))$，我们随后会在自然参数中再次遇到后者。

变换 $y = e^x$ 将每个正态模型变成一个对数正态模型。

函数 `fisher_matrix` 通过数值积分计算 {eq}`ef_fisher`，其中得分用有限差分计算。

```{code-cell} ipython3
def fisher_matrix(logpdf, v, a, b, h=1e-5):
    """Fisher information matrix (ef_fisher) at parameter vector v, by quadrature."""
    v = np.asarray(v, dtype=float)
    e = np.eye(v.size) * h

    def score(z):
        return np.array([(logpdf(z, v + e[i]) - logpdf(z, v - e[i])) / (2 * h)
                         for i in range(v.size)])

    return np.array([[integrate.quad(
        lambda z: score(z)[i] * score(z)[j] * np.exp(logpdf(z, v)),
        a, b, limit=200)[0] for j in range(v.size)] for i in range(v.size)])


def kl_quad(logpdf, v1, v2, a, b):
    """KL divergence of model v1 from model v2, by quadrature."""
    f = lambda z: np.exp(logpdf(z, v1)) * (logpdf(z, v1) - logpdf(z, v2))
    return integrate.quad(f, a, b, limit=200)[0]


def x_normal(x, v):        # log density of X, coordinates (μ, σ)
    return norm.logpdf(x, v[0], v[1])


def y_lognormal(y, v):     # log density of Y = exp(X), coordinates (μ, σ)
    return lognorm.logpdf(y, s=v[1], scale=np.exp(v[0]))


def x_natural(x, v):       # log density of X, coordinates (μ/σ², -1/(2σ²))
    return norm.logpdf(x, -v[0] / (2 * v[1]), np.sqrt(-1 / (2 * v[1])))


(μ1, σ1), (μ2, σ2) = (1.0, 1.5), (-0.5, 1.0)
coordinates = {
    "(μ, σ)": lambda μ, σ: np.array([μ, σ]),
    "(μ, σ²)": lambda μ, σ: np.array([μ, σ**2]),
    "(μ, μ² + σ²)": lambda μ, σ: np.array([μ, μ**2 + σ**2]),
    "(μ/σ², -1/(2σ²))": lambda μ, σ: np.array([μ / σ**2, -1 / (2 * σ**2)]),
}
print("Euclidean distance between the two parameter vectors")
for name, c in coordinates.items():
    print(f"  {name:18} {np.linalg.norm(c(μ1, σ1) - c(μ2, σ2)):.4f}")

print("\nKL divergence")
print(f"  from X:          "
      f"{kl_quad(x_normal, (μ1, σ1), (μ2, σ2), -np.inf, np.inf):.6f}")
print(f"  from Y = exp(X): "
      f"{kl_quad(y_lognormal, (μ1, σ1), (μ2, σ2), 0, np.inf):.6f}")

H_x = norm(μ1, σ1).entropy()
H_y = lognorm(s=σ1, scale=np.exp(μ1)).entropy()
print("\ndifferential entropy of the first model")
print(f"  of X: {H_x:.4f}, of Y: {H_y:.4f}, difference: {H_y - H_x:.4f}")

θ1 = coordinates["(μ/σ², -1/(2σ²))"](μ1, σ1)
cases = (("from X, coordinates (μ, σ)", x_normal, (μ1, σ1), -np.inf),
         ("from Y, coordinates (μ, σ)", y_lognormal, (μ1, σ1), 0),
         ("from X, coordinates (μ/σ², -1/(2σ²))", x_natural, θ1, -np.inf))
print("\nFisher information matrix of the first model")
for label, logpdf, v, a in cases:
    I_v = np.round(fisher_matrix(logpdf, v, a, np.inf), 4) + 0.0
    print(f"  {label}:\n{I_v}")
```

这四个欧几里得距离各不相同。

由 $X$ 计算出的 KL 散度等于由 $Y$ 计算出的 KL 散度。

$Y$ 的微分熵超出 $X$ 的微分熵的量是 $E \log \lvert f'(X) \rvert = E X = 1$。

从 $X$ 和从 $Y$ 在坐标 $(\mu, \sigma)$ 下计算得到的费希尔信息矩阵与 {eq}`ef_fisher_normal` 相一致。

在坐标 $(\mu / \sigma^2, -1 / (2 \sigma^2))$ 下，费希尔信息矩阵是另一个矩阵 $J^\top I J$。

(ef_wasserstein_section)=
### 移动质量与比较高度

最优传输以另一种方式衡量两个模型之间的距离。

实数轴上分布 $p$ 和 $q$ 之间的 **2-沃瑟斯坦距离**为

$$
W_2(p, q) = \left( \min E\left[ (X - Y)^2 \right] \right)^{1/2} ,
$$ (ef_wasserstein)

其中最小值是对所有边缘分布为 $p$ 和 $q$ 的 $(X, Y)$ 联合分布取的。

{doc}`opt_transport` 求解了该问题的离散版本。

在实数轴上，最小值通过配对分位数来达到：$X = F_p^{-1}(U)$ 和 $Y = F_q^{-1}(U)$，其中 $F_p$ 和 $F_q$ 是两个累积分布函数，$U$ 是 $(0, 1)$ 上的均匀分布。

对于两个正态模型，这种配对是 $Y = \mu_2 + (\sigma_2 / \sigma_1)(X - \mu_1)$，且

$$
W_2\left(N(\mu_1, \sigma_1^2), N(\mu_2, \sigma_2^2)\right) = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2} ,
$$

即参数向量 $(\mu_1, \sigma_1)$ 与 $(\mu_2, \sigma_2)$ 之间的欧几里得距离。

因此 $W_2(A, B) = W_2(C, D) = 1$：最优传输将 $A$ 的质量移动到 $B$，以及将 $C$ 的质量移动到 $D$，都移动了一个单位，尽管区分 $A$ 和 $B$ 要比区分 $C$ 和 $D$ 容易得多。

这两种距离回答的是不同的问题。

KL 散度和费希尔信息逐个结果比较两个密度的高度，不使用结果之间的距离；而沃瑟斯坦距离衡量概率质量沿实数轴必须移动多远。

{cite:t}`KhanZhang2022` 把第一种比较称为垂直比较，把第二种称为水平比较。

沃瑟斯坦距离是模型之间的距离，因此重新标记参数不会改变它；在本讲座的各种参数化中，只有在 $(\mu, \sigma)$ 下它才等于参数向量之间的欧几里得距离。

由于它以数据的单位来衡量位移，因此变换数据会改变它。

对于那些把概率放在不同集合上的模型，它也是有限的：$A$ 与原点处的点质量之间的沃瑟斯坦距离是 $0.2$，而它们之间的 KL 散度在任一方向上都是无穷大。

下面的函数计算 {eq}`ef_wasserstein`，对象是一个标准正态随机变量 $Z$ 的两个递增函数所给出的分布，这样对 $Z$ 的值进行配对就等于配对分位数。

我们为四个正态模型以及变换 $y = e^x$ 所产生的对数正态模型计算这一距离。

```{code-cell} ipython3
def w2_increasing(g1, g2):
    """W2 between the distributions of g1(Z) and g2(Z), where Z is standard
    normal and g1 and g2 are increasing; |Z| > 12 has negligible probability."""
    integrand = lambda z: (g1(z) - g2(z))**2 * norm.pdf(z)
    return np.sqrt(integrate.quad(integrand, -12, 12, limit=200)[0])


four_normals = {"A, B": ((0.0, 0.2), (1.0, 0.2)), "C, D": ((0.0, 2.0), (1.0, 2.0))}
for name, ((μa, σa), (μb, σb)) in four_normals.items():
    w_x = w2_increasing(lambda z: μa + σa * z, lambda z: μb + σb * z)
    w_y = w2_increasing(lambda z: np.exp(μa + σa * z),
                        lambda z: np.exp(μb + σb * z))
    kl_x = kl_quad(x_normal, (μa, σa), (μb, σb), -np.inf, np.inf)
    kl_y = kl_quad(y_lognormal, (μa, σa), (μb, σb), 0, np.inf)
    print(f"{name}: W2 from X {w_x:.4f}, from Y = exp(X) {w_y:.4f};   "
          f"KL from X {kl_x:.4f}, from Y {kl_y:.4f}")
```

经过变换后，沃瑟斯坦距离在 $A$ 与 $B$ 之间为 $1.79$，在 $C$ 与 $D$ 之间为 $93.8$，这是因为指数变换拉伸了上尾部，而 $C$ 和 $D$ 在该处放置的概率比 $A$ 和 $B$ 多得多。

而 KL 散度不会改变。

{ref}`ef_fisher_rao` 比较了两种距离所选取的最短路径。

### 统计量与充分性

统计量或信息减损只能导致信息的损失。

设 $X$ 为数据——可以是单一观测值也可以是整个样本——并设 $p^s(\cdot\,; \theta)$ 是当 $X$ 具有分布 $p(\cdot\,; \theta)$ 时统计量 $s(X)$ 的分布。

对于任意两个参数向量 $\theta_1$ 和 $\theta_2$，

$$
D_{KL}\left(p^s(\cdot\,; \theta_1) \| p^s(\cdot\,; \theta_2)\right) \leq D_{KL}\left(p(\cdot\,; \theta_1) \| p(\cdot\,; \theta_2)\right) ,
$$ (ef_dpi)

且 $s(X)$ 的费希尔信息矩阵小于 $X$ 的费希尔信息矩阵，相差一个半正定矩阵。

不等式 {eq}`ef_dpi` 就是 {doc}`blackwell_kihlstrom` 中的数据处理不等式。

{cite:t}`kullback1951information` 证明，当且仅当 $s$ 是充分统计量时，{eq}`ef_dpi` 对每一对参数向量都取等号。

当参数空间连通时，当且仅当 $s$ 是充分统计量时，对每个 $\theta$，$s(X)$ 的费希尔信息矩阵都等于 $X$ 的费希尔信息矩阵。

{cite:t}`chentsov1981` 通过这种不变性刻画了费希尔信息。

在有限集合上的概率分布集合中，在相差一个常数因子的意义下，费希尔信息度量是唯一在充分统计量所诱导的变换下保持不变的黎曼度量。

{doc}`blackwell_kihlstrom` 陈述了琴特索夫定理。

## 指数族

### 定义

如果一组模型的密度（或概率质量函数）可以写成如下形式，则称该族模型是一个**指数族**：

$$
p(x; \theta) = h(x) \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \int h(x) \exp\left( \theta \cdot t(x) \right) dx ,
$$ (ef_family)

当 $x$ 是离散的时积分被替换为求和，其中

* $t(x) \in \mathbb{R}^d$ 是**充分统计量**，
* $\theta \in \mathbb{R}^d$ 是**自然参数**，
* $h(x) \geq 0$ 是**基测度**，
* $F(\theta)$ 是**对数配分函数**，也称为累积量函数。

这个名称来自统计力学，在那里 $e^{F(\theta)}$ 是玻尔兹曼-吉布斯分布的配分函数。

自然参数遍历使定义 $F$ 的积分有限的所有 $\theta$ 组成的集合 $\Theta$。

在 {ref}`ef_maxent` 的最大熵问题中，$h$ 为常数；那里定义的函数 `tilt` 把基分布 $q$ 用作 $h$。

当我们相对于一个非均匀的基分布来最大化熵时，就会出现非常数的 $h$，这是续篇 {doc}`information_projections` 要讨论的问题。

我们假设该表示是**极小的**：在 $h(x) > 0$ 的集合上，不存在非平凡的线性组合 $a \cdot t(x)$ 恒为常数。

这样，不同的 $\theta$ 给出不同的模型。

我们还假设 $\Theta$ 是一个开集，本讲座中的每个族都是如此。

极小表示在仿射坐标变换下是唯一的。

如果 $A$ 是一个可逆的 $d \times d$ 矩阵，那么充分统计量 $A t(x)$ 和自然参数 $A^{-\top} \theta$ 描述的是相同的模型，因为 $(A^{-\top} \theta) \cdot (A t(x)) = \theta \cdot t(x)$。

如果 $b \in \mathbb{R}^d$，那么自然参数 $\theta + b$ 和基测度 $h(x) e^{-b \cdot t(x)}$ 也描述相同的模型。

以下是用这种方式写出的一些熟悉的族。

| 族 | $t(x)$ | $\theta$ | $F(\theta)$ |
|---|---|---|---|
| 伯努利($\pi$) | $x$ | $\log \frac{\pi}{1 - \pi}$ | $\log(1 + e^{\theta})$ |
| 泊松($\lambda$) | $x$ | $\log \lambda$ | $e^{\theta}$ |
| 指数($\lambda$) | $x$ | $-\lambda$ | $-\log(-\theta)$ |
| $N(\mu, \sigma^2)$ | $(x, x^2)$ | $\left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)$ | $-\frac{\theta_1^2}{4 \theta_2} + \frac{1}{2}\log\left(-\frac{\pi}{\theta_2}\right)$ |
| 贝塔($a$, $b$) | $(\log x, \log(1 - x))$ | $(a - 1, b - 1)$ | $\log B(\theta_1 + 1, \theta_2 + 1)$ |
| 分类分布($\pi$) | $(1_{x = 1}, \ldots, 1_{x = K - 1})$ | $\left(\log \frac{\pi_k}{\pi_K}\right)_{k < K}$ | $\log\left(1 + \sum_{k < K} e^{\theta_k}\right)$ |

在每一行中 $h(x) = 1$，泊松那一行除外，其中 $h(x) = 1/x!$。

在伯努利和分类分布那两行中，$\pi$ 是一个概率或者一个概率向量；在正态那一行中它是 $3.14159\ldots$。

在贝塔那一行中，$B$ 是贝塔函数。

$\{1, \ldots, K\}$ 上的分类分布族只使用 $K - 1$ 个指示函数，因为所有 $K$ 个指示函数之和为一；如果使用全部 $K$ 个，该表示就不是极小的了。

多元正态族 $N(\mu, \Sigma)$ 是一个指数族，其 $t(x) = (x, x x^\top)$，自然参数为 $(\Sigma^{-1}\mu, -\frac{1}{2}\Sigma^{-1})$，其中配对为 $\theta \cdot t(x) = \theta_1 \cdot x + x^\top \Theta_2 x$，$\Theta_2$ 是对称矩阵。

因为 $x_i x_j = x_j x_i$，极小表示只保留 $x x^\top$ 中对角线及其以上的元素。

形如 {eq}`ef_family` 的族包含了它自身任意两个模型的每一个归一化几何混合：正比于 $p(x; \theta_0)^{1 - a} p(x; \theta_1)^a$ 的分布具有自然参数 $(1 - a)\theta_0 + a \theta_1$，对每个 $a \in [0, 1]$，以及对使该参数落在 $\Theta$ 中的任何其他 $a$ 都成立。

续篇 {doc}`information_projections` 基于这一事实建立了一种几何结构。

### 维数固定的充分统计量

对于指数族中的一个模型，一个大小为 $n$ 的独立同分布样本的联合密度为

$$
\prod_{i=1}^n p(x_i; \theta) = \exp\left( n \left[ \theta \cdot \bar t - F(\theta) \right] \right) \prod_{i=1}^n h(x_i),
\qquad
\bar t = \frac{1}{n} \sum_{i=1}^n t(x_i) .
$$

由分解定理 {eq}`ef_factorization`，平均值 $\bar t$ 对 $\theta$ 是充分的。

无论样本量 $n$ 为多少，它的维数都是 $d$。

对于正态族，$\bar t$ 是样本矩对 $\left(\frac{1}{n} \sum_i x_i, \frac{1}{n} \sum_i x_i^2\right)$。

对于伯努利族，$\bar t$ 是成功的比例。

对于泊松族，$\bar t$ 是样本均值。

{cite:t}`Darmois1935`、{cite:t}`Koopman1936` 和 {cite:t}`Pitman1936` 证明了一个逆命题。

假设一个族中的模型在实数轴上有密度，可能取值 $x$（使 $p(x; \theta) > 0$ 的集合）不依赖于 $\theta$，并且对每个样本量 $n$，独立同分布样本都有一个维数不随 $n$ 增长的充分统计量。

那么，在光滑性条件下，该族是一个指数族。[^pkd]

[^pkd]: 这些光滑性条件排除了那些把很多数字打包成一个数字的统计量，例如将 $x_1, \ldots, x_n$ 的十进制数字交错排列而得到的统计量。对于单次观测，$x$ 本身就是一个有限维的充分统计量；该定理关注的是充分统计量的维数在 $n$ 增长时的表现。

两个族说明了这些条件为何重要。

$(0, \theta)$ 上的均匀族对每个 $n$ 都有一维充分统计量 $\max_i x_i$，但它不是指数族：它的可能结果集合 $(0, \theta)$ 依赖于 $\theta$。

柯西位置族满足关于结果的条件，但对它而言，没有固定维数的统计量是充分的：它最小的充分统计量是次序统计量的集合，其维数随 $n$ 增长。

我们从 $N(1, 1.5^2)$ 中抽取一个大小为 $n = 8$ 的样本 $\mathbf x$，并用一个将向量 $\mathbf 1 = (1, \ldots, 1)$ 映射到自身的正交矩阵 $Q$ 乘以它。

旋转后的样本 $Q \mathbf x$ 具有相同的和 $\mathbf 1^\top Q \mathbf x = \mathbf 1^\top \mathbf x$，以及相同的平方和 $\mathbf x^\top Q^\top Q \mathbf x = \mathbf x^\top \mathbf x$，因此对正态族来说，它具有相同的充分统计量。

```{code-cell} ipython3
rng_rotate = np.random.default_rng(1)
n = 8
x_sample = rng_rotate.normal(1.0, 1.5, n)

# Q is orthogonal and maps (1, ..., 1) into itself
ones = np.ones(n)
basis = np.linalg.svd(ones[None, :])[2][1:].T     # orthonormal basis of 1's complement
rotation = np.linalg.qr(rng_rotate.normal(size=(n - 1, n - 1)))[0]
Q = np.outer(ones, ones) / n + basis @ rotation @ basis.T
x_rotated = Q @ x_sample

print("sample:        ", np.round(x_sample, 3))
print("rotated sample:", np.round(x_rotated, 3))
print(f"sums:            {x_sample.sum():.6f}  {x_rotated.sum():.6f}")
print(f"sums of squares: {x_sample @ x_sample:.6f}  {x_rotated @ x_rotated:.6f}")

μ_vals, σ_vals = np.meshgrid(np.linspace(-2, 4, 61), np.linspace(0.3, 4, 38))


def log_likelihood(dist, sample, μ, σ):
    return sum(dist(μ, σ).logpdf(xi) for xi in sample)


for name, dist in (("normal", norm), ("Cauchy", cauchy)):
    gap = np.max(np.abs(log_likelihood(dist, x_sample, μ_vals, σ_vals)
                        - log_likelihood(dist, x_rotated, μ_vals, σ_vals)))
    print(f"{name:6} family: largest gap between the two log likelihood "
          f"functions = {gap:.2e}")
```

下图展示了位置参数 $\mu$ 的两个对数似然函数，其尺度参数固定在 $1.5$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log likelihoods before and after rotation
    name: fig-ef-rotated-sample
---
μ_line = np.linspace(-2, 4, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, dist) in zip(axes, (("normal", norm), ("Cauchy", cauchy))):
    for sample, label, ls in ((x_sample, "sample", "-"),
                              (x_rotated, "rotated sample", "--")):
        ax.plot(μ_line, log_likelihood(dist, sample, μ_line, 1.5), ls, lw=2,
                label=label)
    ax.set_title(f"{name} family")
    ax.set_xlabel(r"$\mu$")
    ax.legend()
axes[0].set_ylabel("log likelihood")
plt.tight_layout()
plt.show()
```

正态族无法区分这两个样本；柯西族则可以。

### 对数配分函数生成矩

对数配分函数决定了其他一切。

对 $F(\theta) = \log \int h(x) \exp(\theta \cdot t(x)) \, dx$ 求导：

$$
\nabla F(\theta) = \frac{\int t(x) h(x) \exp(\theta \cdot t(x)) \, dx}{\int h(x) \exp(\theta \cdot t(x)) \, dx} = E_\theta [t(X)] .
$$ (ef_gradient)

再次求导得到

$$
\nabla^2 F(\theta) = E_\theta\left[ t(X) t(X)^\top \right] - E_\theta[t(X)] \, E_\theta[t(X)]^\top = \operatorname{Cov}_\theta [t(X)] .
$$ (ef_hessian)

函数 $F$ 是凸函数，因为协方差矩阵是半正定的；极小性使得 $\operatorname{Cov}_\theta[t(X)]$ 为正定矩阵，因此 $F$ 是严格凸的。

更一般地，$\log E_\theta \exp(s \cdot t(X)) = F(\theta + s) - F(\theta)$，因此 $F$ 的各阶导数就是 $t(X)$ 的累积量。

让我们验证 {eq}`ef_gradient` 和 {eq}`ef_hessian` 对泊松族和正态族成立，将有限差分计算得到的 $F$ 的导数与通过求和及积分计算得到的矩进行比较。

```{code-cell} ipython3
def F_poisson(θ):
    return np.exp(θ)


def F_normal(θ):
    θ1, θ2 = θ
    return -θ1**2 / (4 * θ2) + 0.5 * np.log(-np.pi / θ2)


def normal_θ(μ, σ):
    return np.array([μ / σ**2, -1 / (2 * σ**2)])


def gradient(F, θ, h=1e-5):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([np.squeeze(F(θ + e[i]) - F(θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


def hessian(F, θ, h=1e-4):
    θ = np.atleast_1d(np.asarray(θ, dtype=float))
    e = np.eye(θ.size) * h
    return np.array([(gradient(F, θ + e[i]) - gradient(F, θ - e[i])) / (2 * h)
                     for i in range(θ.size)])


λ = 3.0
k = np.arange(100)
pk = poisson.pmf(k, λ)
print("Poisson:  F'(θ), F''(θ) =",
      np.round([gradient(F_poisson, np.log(λ))[0],
                hessian(F_poisson, np.log(λ))[0, 0]], 6),
      "  mean, variance =", np.round([pk @ k, pk @ k**2 - (pk @ k)**2], 6))

μ, σ = 0.5, 1.2
moments = [integrate.quad(lambda x: x**j * norm.pdf(x, μ, σ), -15, 15)[0]
           for j in range(1, 5)]
mean_t = np.array(moments[:2])
cov_t = np.array([[moments[1] - moments[0]**2,
                   moments[2] - moments[0] * moments[1]],
                  [moments[2] - moments[0] * moments[1],
                   moments[3] - moments[1]**2]])
print("\nnormal:  ∇F(θ) =", np.round(gradient(F_normal, normal_θ(μ, σ)), 6),
      "  E[t] =", np.round(mean_t, 6))
print("∇²F(θ) =\n", np.round(hessian(F_normal, normal_θ(μ, σ)), 4))
print("Cov[t] =\n", np.round(cov_t, 4))
```

## 两套坐标系

### 期望参数

由 {eq}`ef_gradient`，映射 $\theta \mapsto \eta = \nabla F(\theta) = E_\theta[t(X)]$ 把每个自然参数映射为充分统计量均值构成的向量。

由于 $F$ 是严格凸的，这个映射是一一对应的。

对于本讲座中的各个族，它的值域是 $t(X)$ 可能取值的凸包的内部：伯努利均值严格介于 $0$ 和 $1$ 之间，泊松均值为正，等等。

我们可以用自然参数 $\theta$ 或用**期望参数** $\eta$ 来标记该族中的每个模型。

对于正态族，$\eta = (\mu, \mu^2 + \sigma^2)$；对于泊松族，$\eta = \lambda$；对于伯努利族，$\eta = \pi$。

一个指数族带有三套坐标系：如 $(\mu, \sigma)$ 这样的常规参数（用于正态族）、自然参数 $\theta$，以及期望参数 $\eta$。

{ref}`ef_maxent` 中的最大熵问题固定了期望参数 $\eta = c$；它们的解给出了自然参数 $\theta$。

### 勒让德对偶

从 $\eta$ 返回到 $\theta$ 的映射是另一个凸函数的梯度。

$F$ 的**凸共轭**，即勒让德变换，是

$$
F^*(\eta) = \sup_\theta \left\{ \theta \cdot \eta - F(\theta) \right\} .
$$ (ef_conjugate)

对于位于 $\nabla F$ 值域中的 $\eta$，上确界在 $\eta = \nabla F(\theta)$ 处取得，因此

$$
F(\theta) + F^*(\eta) = \theta \cdot \eta
\quad \text{当 } \eta = \nabla F(\theta) \text{ 时} ,
$$ (ef_fenchel)

并且，根据包络定理，$\nabla F^*(\eta) = \theta$。

两个梯度映射 $\nabla F$ 和 $\nabla F^*$ 是互逆的函数，它们的雅可比矩阵也互为逆矩阵：

$$
\nabla^2 F^*(\eta) = \left[ \nabla^2 F(\theta) \right]^{-1} .
$$

因为 $F$ 是凸函数，对它施行两次变换就回到 $F$ 本身：

$$
F(\theta) = \max_\eta \left\{ \theta \cdot \eta - F^*(\eta) \right\} .
$$

```{note}
经济学家在利润函数和生产函数之间使用相同的对偶关系。

在 {doc}`houthakker_aggregation` 中，行业利润函数为 $\Pi(p) = \max_X \{ f(X) - p \cdot X \}$，其中 $X$ 是一个投入向量，$p$ 是它们的价格向量，$f$ 是总生产函数（该讲座称之为 $F$）。

写 $\theta = -p$，则 $\Pi(-\theta) = \max_X \{ \theta \cdot X - (-f(X)) \}$，这与前面的式子形式相同。

对应关系如下：

* 对数配分函数 $F$ $\leftrightarrow$ 利润函数 $\Pi$，
* 自然参数 $\theta$ $\leftrightarrow$ 负投入价格 $-p$，
* 期望参数 $\eta$ $\leftrightarrow$ 投入数量 $X$，
* 共轭函数 $F^*$ $\leftrightarrow$ 负生产函数 $-f$。

霍特林引理 $\partial \Pi / \partial p_i = -X_i$ 对应于 $\nabla F(\theta) = \eta$，而从利润函数恢复生产函数对应于从 $F$ 恢复 $F^*$。
```

### 共轭函数是负熵

在从 $p(\cdot; \theta)$ 抽取的一个样本 $X$ 处计算 $\log p(x; \theta) = \log h(x) + \theta \cdot t(x) - F(\theta)$ 并取期望。

利用 {eq}`ef_fenchel`，

$$
E_\theta \log p(X; \theta) = E_\theta \log h(X) + \theta \cdot \eta - F(\theta) = E_\theta \log h(X) + F^*(\eta) .
$$

因此当 $p(x; \theta) > 0$ 时 $h(x) = 1$，有

$$
F^*(\eta) = - H(p(\cdot; \theta)) ,
$$ (ef_negentropy)

于是*对数配分函数的共轭，作为期望参数的函数，就是熵的负值。*

一般而言，$H(p(\cdot; \theta)) = -F^*(\eta) - E_\theta \log h(X)$。

{ref}`ef_ex_poisson` 推导了泊松情形，其中 $h(x) = 1/x!$。

对于伯努利族，$h = 1$，$F(\theta) = \log(1 + e^\theta)$，$F^*(\eta) = \eta \log \eta + (1 - \eta) \log (1 - \eta)$。

在下图左侧面板中，斜率为 $\eta$ 的 $F$ 的切线与纵轴相交于 $-F^*(\eta)$。

在右侧面板中，$F^*$ 在 $\eta$ 处的切线斜率为 $\theta$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bernoulli log-partition function and its conjugate
    name: fig-ef-legendre
---
def F_bern(θ):
    return np.log1p(np.exp(θ))


def F_star_bern(η):
    return η * np.log(η) + (1 - η) * np.log(1 - η)


θ0 = 2.0
η0 = 1 / (1 + np.exp(-θ0))

θ_vals = np.linspace(-4, 4, 200)
η_vals = np.linspace(0.001, 0.999, 200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(θ_vals, F_bern(θ_vals), lw=2, label=r"$F(\theta)$")
ax.plot(θ_vals, -F_star_bern(η0) + η0 * θ_vals, "--",
        label=rf"tangent line with slope $\eta$ = {η0:.2f}")
ax.plot([0], [-F_star_bern(η0)], "ko")
ax.annotate(r"$-F^*(\eta)$", (0, -F_star_bern(η0)),
            xytext=(8, -14), textcoords="offset points")
ax.plot([θ0], [F_bern(θ0)], "o", color="C1")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_xlabel(r"$\theta$")
ax.legend()

ax = axes[1]
ax.plot(η_vals, F_star_bern(η_vals), lw=2, label=r"$F^*(\eta) = -H$")
ax.plot(η_vals, F_star_bern(η0) + θ0 * (η_vals - η0), "--",
        label=rf"tangent line with slope $\theta$ = {θ0:.2f}")
ax.plot([η0], [F_star_bern(η0)], "o", color="C1")
ax.set_xlabel(r"$\eta$")
ax.legend()
plt.tight_layout()
plt.show()
```

(ef_maxent)=
## 最大熵

### 一个骰子

埃德温·杰恩斯（Edwin Jaynes）提出了最大熵原理 {cite}`Jaynes1957`，并在他 1962 年布兰代斯讲座中用下面这个问题来说明，该问题现在称为布兰代斯骰子问题 {cite}`Jaynes1963`。

一个骰子的面为 $x \in \{1, 2, \ldots, 6\}$。

我们只被告知点数的平均值是 $4.5$，而不是公平骰子的 $3.5$。

我们应该给各个面赋予什么样的概率分布 $p = (p_1, \ldots, p_6)$？

有很多分布的均值都是 $4.5$。

杰恩斯建议选择在缺失信息方面"最不自以为是的"（maximally noncommittal with regard to missing information）那一个 {cite}`Jaynes1957`，他把这理解为使香农熵

$$
H(p) = - \sum_{x} p(x) \log p(x)
$$

在约束条件 $\sum_x p(x) = 1$ 和 $\sum_x x \, p(x) = 4.5$ 下最大化的那个分布。

### 解是指数族中的一个模型

让我们求解这个问题更一般的版本。

设 $t(x) = (t_1(x), \ldots, t_d(x))$ 是结果 $x$ 的一组函数构成的向量，$x$ 取值于有限集合。

我们寻求使 $H(p)$ 在矩约束

$$
\sum_x p(x) \, t(x) = c
$$ (ef_moments)

（对给定向量 $c$）以及 $\sum_x p(x) = 1$ 下最大化的分布 $p$。

为 {eq}`ef_moments` 附加一个拉格朗日乘子向量 $\theta$。

关于 $p(x)$ 的一阶条件是 $-\log p(x) - 1 + \theta \cdot t(x) - \nu = 0$，其中 $\nu$ 是 $\sum_x p(x) = 1$ 对应的乘子。

解的形式为

$$
p(x; \theta) = \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \sum_x \exp\left( \theta \cdot t(x) \right) ,
$$ (ef_maxent_solution)

其中 $\theta$ 满足 {eq}`ef_moments`。

函数 $F$ 使得概率之和等于一。

这样的 $\theta$ 存在，当且仅当 $c$ 落在点 $t(x)$ 凸包的相对内部中，也就是说，落在包含该凸包的最小仿射子空间中它的内部；对于下面的骰子问题，即当且仅当 $1 < c < 6$。

当 $c$ 落在相对边界上时，比如一个均值为 $6$ 的骰子，极大化分布仍然存在。

它在某些结果上赋予零概率，并且位于该族之外，是 $\theta$ 发散时模型序列的极限。

模型族 {eq}`ef_maxent_solution` 构成一个指数族，充分统计量为 $t$，基测度为常数。

拉格朗日乘子 $\theta$ 是自然参数，约束均值 $c$ 是期望参数 $\eta$。

为了确认 {eq}`ef_maxent_solution` 是一个最大值，设 $p$ 是满足 {eq}`ef_moments` 的任意分布，设 $p^*$ 是满足它的指数族中的模型。

因为 $\log p^*(x) = \theta \cdot t(x) - F(\theta)$ 是 $t(x)$ 的仿射函数，而 $p$ 和 $p^*$ 对 $t$ 赋予相同的均值 $c$，所以

$$
\sum_x p(x) \log p^*(x) = \theta \cdot c - F(\theta) = \sum_x p^*(x) \log p^*(x) = - H(p^*) .
$$

因此

$$
H(p^*) - H(p) = \sum_x p(x) \log p(x) - \sum_x p(x) \log p^*(x) = D_{KL}(p \| p^*) \geq 0 ,
$$ (ef_entropy_gap)

当且仅当 $p = p^*$ 时取等号。

在所有满足矩约束的分布中，指数族中的那个模型具有最大的熵，而任何其他分布的熵亏缺都等于它相对于该模型的 KL 散度。

续篇 {doc}`information_projections` 表明 {eq}`ef_entropy_gap` 是一个毕达哥拉斯定理的实例。

对于杰恩斯的骰子问题，$t(x) = x$，$\theta$ 是一个标量。

下面的函数 `tilt` 计算**基分布** $q$ 的**指数倾斜**（exponential tilt）：即正比于 $q(x) \exp(\theta \, t(x))$ 的分布，对应某个统计量 $t$。

对于最大熵问题，$q$ 是均匀分布，结果即为 {eq}`ef_maxent_solution`。

第二个函数寻找能够给出指定均值的 $\theta$。

```{code-cell} ipython3
def tilt(q, t, θ):
    """The distribution proportional to q(x) exp(θ t(x))."""
    log_w = np.log(q) + θ * t
    return np.exp(log_w - logsumexp(log_w))


def match_mean(q, t, c):
    """Find θ such that the tilted distribution has mean c."""
    θ = brentq(lambda θ: tilt(q, t, θ) @ t - c, -50, 50)
    return θ, tilt(q, t, θ)


def entropy(p):
    return -p @ np.log(p)


def kl(p, q):
    return p @ np.log(p / q)


faces = np.arange(1, 7)
uniform_die = np.full(6, 1 / 6)
θ_die, p_die = match_mean(uniform_die, faces, 4.5)
print(f"θ = {θ_die:.4f}")
print("maximum entropy distribution:", np.round(p_die, 4))
print(f"entropy = {entropy(p_die):.4f}, compared with log 6 = {np.log(6):.4f}")
```

概率按几何方式递增，从一个面到下一个面的比例因子为 $e^{\theta}$。

让我们通过构造其他均值为 $4.5$ 的分布来检验 {eq}`ef_entropy_gap`。

我们通过在保持总概率和均值不变的方向上偏离 $p^*$ 来构造这些分布。

```{code-cell} ipython3
rng = np.random.default_rng(0)

# directions orthogonal to (1, ..., 1) and to (1, 2, ..., 6)
constraints = np.vstack([np.ones(6), faces])
null_space = np.linalg.svd(constraints)[2][2:].T

print(" H(p*) - H(p)    KL(p || p*)")
for _ in range(5):
    while True:
        p = p_die + null_space @ rng.normal(scale=0.05, size=4)
        if np.all(p > 0):
            break
    print(f"{entropy(p_die) - entropy(p):12.6f} {kl(p, p_die):14.6f}")
```

下图展示了几种均值下的最大熵分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum entropy die distributions by mean
    name: fig-ef-die
---
fig, ax = plt.subplots()
for c in (2.5, 3.5, 4.5, 5.5):
    θ, p = match_mean(uniform_die, faces, c)
    ax.plot(faces, p, "o-", label=rf"mean {c}, $\theta$ = {θ:.2f}")
ax.set_xlabel("face")
ax.set_ylabel("probability")
ax.legend()
plt.show()
```

### 影子价格

*最大熵问题的拉格朗日乘子就是矩约束的影子价格。*

由 {eq}`ef_negentropy`，极大化的熵为 $H(p^*) = -F^*(c)$，因此根据包络定理，$\partial H(p^*) / \partial c = -\nabla F^*(c) = -\theta$。

乘子 $\theta_j$ 是当约束均值 $c_j$ 升高时熵下降的速率。

对于杰恩斯的骰子问题，在 $c = 4.5$ 处 $\theta > 0$，因为把均值进一步推离 $3.5$ 需要一个更不均匀的分布。

### 正态分布

现在设结果 $x$ 是一个实数，并约束其均值和二阶矩：

$$
E[X] = \mu, \qquad E[X^2] = \mu^2 + \sigma^2 .
$$

对于实数轴上的密度 $p$，**微分熵**是 $H(p) = -\int p(x) \log p(x) \, dx$。

导出 {eq}`ef_maxent_solution` 的论证现在说明，具有最大熵的密度具有如下形式：

$$
p(x; \theta) = \exp\left( \theta_1 x + \theta_2 x^2 - F(\theta) \right) ,
$$

它只有在 $\theta_2 < 0$ 时才是可积的。

通过配方可知，这正是具有 $\theta_1 = \mu / \sigma^2$ 和 $\theta_2 = -1/(2\sigma^2)$ 的正态密度。

导出 {eq}`ef_entropy_gap` 的论证在把求和替换为积分后依然成立，因此每个均值为 $\mu$、方差为 $\sigma^2$ 的密度 $p$ 都满足

$$
H(\varphi) - H(p) = D_{KL}(p \| \varphi) \geq 0 ,
$$ (ef_normal_gap)

其中 $\varphi$ 是 $N(\mu, \sigma^2)$ 密度。

*在实数轴上所有具有给定均值和方差的密度中，正态密度具有最大的熵。*

它的熵为 $H(\varphi) = \frac{1}{2} \log(2 \pi e \sigma^2)$。

与离散分布的熵不同，微分熵可以为负，并且当我们改变衡量 $x$ 的单位时它会改变，这在 {ref}`ef_invariants` 中的表格里有所记录；而差距 {eq}`ef_normal_gap` 是一个 KL 散度，两者都不会改变。

我们对几个均值为零、方差为一的密度检验 {eq}`ef_normal_gap`。

我们用 SciPy 的公式计算熵，并用数值积分计算 KL 散度。

```{code-cell} ipython3
φ = norm(0, 1)
alternatives = {
    "uniform": uniform(-np.sqrt(3), 2 * np.sqrt(3)),
    "Laplace": laplace(0, 1 / np.sqrt(2)),
    "logistic": logistic(0, np.sqrt(3) / np.pi),
}

print(f"entropy of N(0, 1): {φ.entropy():.4f}\n")
print(f"{'distribution':12} {'variance':>8} {'entropy':>8} "
      f"{'H(φ) - H(p)':>12} {'KL(p || φ)':>11}")
for name, dist in alternatives.items():
    a, b = dist.support()
    a, b = max(a, -30), min(b, 30)
    kl_p_φ = integrate.quad(
        lambda x: dist.pdf(x) * (dist.logpdf(x) - φ.logpdf(x)), a, b,
        limit=200)[0]
    print(f"{name:12} {dist.var():8.4f} {dist.entropy():8.4f} "
          f"{φ.entropy() - dist.entropy():12.4f} {kl_p_φ:11.4f}")
```

我们也可以看到正态分布从一个离散的最大熵问题中出现。

将结果放在一个精细网格上，并施加两个矩约束。

下面的函数用牛顿法求出自然参数，使用 {eq}`ef_gradient` 和 {eq}`ef_hessian`：$F$ 的梯度是 $t$ 的均值向量，$F$ 的黑塞矩阵是 $t$ 的协方差矩阵。

```{code-cell} ipython3
def maxent(q, T, c, tol=1e-12, max_iter=100):
    """
    Maximize entropy relative to q subject to E[T] = c.

    T is an m × d array whose rows are t(x).  Returns θ and the
    distribution proportional to q(x) exp(θ · t(x)).  The iterations
    converge only if c is in the interior of the convex hull of the
    rows of T.
    """
    def tilted(θ):
        log_w = np.log(q) + T @ θ
        return np.exp(log_w - logsumexp(log_w))

    θ = np.zeros(T.shape[1])
    for _ in range(max_iter):
        p = tilted(θ)
        mean = p @ T
        cov = (T - mean).T @ ((T - mean) * p[:, None])
        step = np.linalg.solve(cov, c - mean)
        θ = θ + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("no convergence: is c inside the hull of t(x)?")
    return θ, tilted(θ)


μ_grid, σ_grid = 1.0, 1.2
grid = np.linspace(-6, 6, 241)
spacing = grid[1] - grid[0]
T_grid = np.column_stack([grid, grid**2])
θ_grid, p_grid = maxent(np.full(grid.size, 1 / grid.size), T_grid,
                        c=np.array([μ_grid, μ_grid**2 + σ_grid**2]))
print(f"θ = {np.round(θ_grid, 4)}")
print(f"(μ/σ², -1/(2σ²)) = "
      f"{np.round([μ_grid / σ_grid**2, -1 / (2 * σ_grid**2)], 4)}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Grid maximum entropy matches the normal
    name: fig-ef-grid-normal
---
fig, ax = plt.subplots()
ax.plot(grid, p_grid / spacing, lw=4, alpha=0.4,
        label="maximum entropy on a grid")
ax.plot(grid, norm.pdf(grid, μ_grid, σ_grid), "k--", lw=1,
        label=f"N({μ_grid:g}, {σ_grid:g}²)")
ax.set_xlabel("$x$")
ax.legend(loc="upper right")
plt.show()
```

### 其他矩约束

不同的矩约束产生不同的族。

| 结果 | 约束均值 | 最大熵分布 |
|---|---|---|
| $\{1, \ldots, m\}$ | 无 | 均匀分布 |
| $\{0, 1, 2, \ldots\}$ | $E[X]$ | 几何分布 |
| $[a, b]$ | 无 | 均匀分布 |
| $[0, \infty)$ | $E[X]$ | 指数分布 |
| $(0, \infty)$ | $E[X]$，$E[\log X]$ | 伽马分布 |
| $(0, \infty)$ | $E[\log X]$，$E[(\log X)^2]$ | 对数正态分布 |
| $(0, 1)$ | $E[\log X]$，$E[\log(1 - X)]$ | 贝塔分布 |
| $\mathbb{R}$ | $E[\lvert X \rvert]$ | 拉普拉斯分布 |
| $\mathbb{R}$ | $E[X]$，$E[X^2]$ | 正态分布 |
| $\mathbb{R}^m$ | $E[X]$，$E[X X^\top]$ | 多元正态分布 |

最后一行意味着，在 $\mathbb{R}^m$ 上所有具有给定均值向量和协方差矩阵 $\Sigma$ 的密度中，多元正态密度具有最大的熵，即 $\frac{1}{2} \log \det(2 \pi e \Sigma)$。

对数正态那一行就是把正态那一行应用到 $\log X$ 上：在 $(0, \infty)$ 上正比于 $\exp(\theta_1 \log x + \theta_2 (\log x)^2)$ 的密度就是一个对数正态密度。

{ref}`ef_ex_gamma` 要求你推导伽马分布的情形。

### 逻辑选择概率

对数配分函数与熵之间的对偶关系对使用离散选择多项逻辑模型的经济学家来说是熟悉的。

一个决策者在 $K$ 个备选方案（效用为 $v_1, \ldots, v_K$）中进行选择，并加上尺度为 $\tau$ 的独立第一型极值（Gumbel）冲击。

各选择的概率为

$$
\pi_k = \frac{\exp(v_k / \tau)}{\sum_j \exp(v_j / \tau)} ,
$$

而期望的最大化效用为 $\tau \log \sum_j \exp(v_j / \tau) + \tau \gamma$，其中 $\gamma \approx 0.5772$ 是欧拉常数。

去掉常数 $\tau \gamma$ 剩下的即为**包容值**（inclusive value）$\tau \log \sum_j \exp(v_j / \tau)$。

包容值是 $\tau$ 乘以在 $\theta = v / \tau$ 处求值的 $\log \sum_j e^{\theta_j}$。

函数 $\theta \mapsto \log \sum_j e^{\theta_j}$ 是分类分布族的对数配分函数，该族以所有 $K$ 个指示函数 $1_{x = k}$ 作为充分统计量。

这种表示不是极小的，因为这些指示函数之和为一。

给每个 $v_k$ 都加上相同的常数，选择概率不变。

方程 {eq}`ef_gradient` 仍然成立，所以包容值关于 $v$ 的梯度就是选择概率构成的向量。

离散选择理论学者把这一事实称为威廉姆斯-达利-扎卡里定理。

它对任何加性随机效用模型都成立；逻辑模型是服从 Gumbel 冲击的特殊情形。

因为 $\theta \mapsto \log \sum_j e^{\theta_j}$ 的共轭是概率单纯形上的负熵，

$$
\tau \log \sum_j \exp(v_j / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k + \tau H(\pi) \right\} ,
$$ (ef_logit)

其中最大值是对概率向量 $\pi$ 取的，并且在逻辑概率处取得。

{cite:t}`AndersonDePalmaThisse1988` 把 {eq}`ef_logit` 解释为一个代表性消费者的间接效用，该消费者的消费份额就是逻辑概率。

更一般地，对任何满足 $q_k > 0$（对所有 $k$）的基分布 $q$，

$$
\tau \log \sum_k q_k \exp(v_k / \tau) = \max_{\pi} \left\{ \sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) \right\} ,
$$ (ef_logit_base)

其中最大值在 $\pi^*_k \propto q_k \exp(v_k / \tau)$ 处取得。

为了验证 {eq}`ef_logit_base`，展开 $D_{KL}(\pi \| \pi^*)$，可以发现对每个概率向量 $\pi$，

$$
\sum_k \pi_k v_k - \tau D_{KL}(\pi \| q) = \tau \log \sum_k q_k \exp(v_k / \tau) - \tau D_{KL}(\pi \| \pi^*) ,
$$

而最后一项只有当 $\pi = \pi^*$ 时才为零。

当 $q$ 是均匀分布时，$D_{KL}(\pi \| q) = \log K - H(\pi)$，{eq}`ef_logit_base` 就是 {eq}`ef_logit` 减去常数 $\tau \log K$。

{cite:t}`MattssonWeibull2002` 把 {eq}`ef_logit_base` 解释为一个决策者选择备选方案上的概率分布 $\pi$ 并为偏离默认分布 $q$ 付出代价 $\tau D_{KL}(\pi \| q)$ 的问题。

他们把这种代价称为努力的负效用。

极大化子 $\pi^*$ 是 $q$ 的一个指数倾斜。

续篇 {doc}`information_projections` 在鲁棒控制的最坏情形模型中、在更新投入产出表的 RAS 方法中，以及在 {cite:t}`ChooSiow2006` 婚姻市场模型背后的熵正则化最优传输问题中，都发现了相同的倾斜形式。

让我们用数值方法检验 {eq}`ef_logit` 和 {eq}`ef_logit_base`。

```{code-cell} ipython3
v = np.array([1.0, 2.0, 0.5, 1.5])
τ = 0.7
inclusive_value = τ * logsumexp(v / τ)
π_logit = np.exp(v / τ - logsumexp(v / τ))
print(f"{'inclusive value:':36} {inclusive_value:.6f}")
objective = π_logit @ v + τ * entropy(π_logit)
print(f"{'π·v + τ H(π) at logit π:':36} {objective:.6f}")

# other probability vectors do worse
best_other = max(π @ v + τ * entropy(π)
                 for π in rng.dirichlet(np.ones(4), size=10_000))
print(f"{'best of 10,000 random π:':36} {best_other:.6f}")

# a nonuniform base distribution
q = np.array([0.1, 0.2, 0.3, 0.4])
π_q = tilt(q, v / τ, 1.0)
print(f"\n{'τ log Σ q exp(v/τ):':36} {τ * logsumexp(v / τ, b=q):.6f}")
print(f"{'π·v - τ D(π || q) at π ∝ q e^(v/τ):':36} "
      f"{π_q @ v - τ * kl(π_q, q):.6f}")
```

## 布雷格曼散度与 KL 散度

### 布雷格曼散度

一个凸函数位于它的切线之上。

由严格凸函数 $F$ 生成的**布雷格曼散度**衡量这种超出的程度：

$$
B_F(\theta_1, \theta_2) = F(\theta_1) - F(\theta_2) - \nabla F(\theta_2) \cdot (\theta_1 - \theta_2) \geq 0 .
$$ (ef_bregman)

它是在 $\theta_1$ 处 $F$ 与 $\theta_2$ 处切线之间的竖直差距，并且只有当 $\theta_1 = \theta_2$ 时才为零。

布雷格曼散度通常关于其两个自变量不对称。

当 $F(\theta) = \frac{1}{2} \lVert \theta \rVert^2$ 时，它就是欧几里得距离平方的一半。

### KL 散度是一种布雷格曼散度

取指数族中的两个模型，其自然参数为 $\theta_1$ 和 $\theta_2$，期望参数为 $\eta_1$ 和 $\eta_2$。

由于 $\log p(x; \theta_1) - \log p(x; \theta_2) = (\theta_1 - \theta_2) \cdot t(x) - F(\theta_1) + F(\theta_2)$，在 $p(\cdot; \theta_1)$ 下取期望得到

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) - F(\theta_1) - \eta_1 \cdot (\theta_2 - \theta_1)
= B_F(\theta_2, \theta_1) .
$$ (ef_kl_bregman)

两个参数互换了位置：*$p_{\theta_1}$ 相对于 $p_{\theta_2}$ 的 KL 散度，就是 $\theta_2$ 相对于 $\theta_1$ 的布雷格曼散度。*

利用 {eq}`ef_fenchel` 替换 $F(\theta_1)$，我们也可以写出

$$
D_{KL}\left( p_{\theta_1} \| p_{\theta_2} \right)
= F(\theta_2) + F^*(\eta_1) - \theta_2 \cdot \eta_1
= B_{F^*}(\eta_1, \eta_2) ,
$$ (ef_kl_dual)

其中两个参数的顺序与 KL 散度中的顺序相同。

KL 散度在任一坐标系下都是一种布雷格曼散度：在自然参数下由 $F$ 生成，在期望参数下由 $F^*$ 生成。

这两个公式在两套坐标系中计算的是同一个不变量。

让我们为泊松分布和正态分布检验 {eq}`ef_kl_bregman` 和 {eq}`ef_kl_dual`，通过求和与积分直接计算 KL 散度。

```{code-cell} ipython3
def bregman(F, θ1, θ2):
    """B_F(θ1, θ2), using a numerical gradient of F at θ2."""
    θ1, θ2 = np.atleast_1d(θ1), np.atleast_1d(θ2)
    return np.squeeze(F(θ1) - F(θ2)) - gradient(F, θ2) @ (θ1 - θ2)


def F_star_poisson(η):
    return η * np.log(η) - η


# Poisson
λ1, λ2 = 3.0, 5.0
k = np.arange(150)
kl_direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print("Poisson")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_poisson, np.log(λ2), np.log(λ1)):.6f}")
print(f"  B_F*(η1, η2)          = {bregman(F_star_poisson, λ1, λ2):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_poisson, np.log(λ1), np.log(λ2)):.6f}")

# normal
(μ1, σ1), (μ2, σ2) = (0.0, 1.0), (1.0, 2.0)
kl_direct = integrate.quad(
    lambda x: norm.pdf(x, μ1, σ1)
    * (norm.logpdf(x, μ1, σ1) - norm.logpdf(x, μ2, σ2)), -20, 20)[0]
print("\nnormal")
print(f"  direct KL(p1 || p2)   = {kl_direct:.6f}")
print(f"  B_F(θ2, θ1)           = "
      f"{bregman(F_normal, normal_θ(μ2, σ2), normal_θ(μ1, σ1)):.6f}")
print(f"  B_F(θ1, θ2), reversed = "
      f"{bregman(F_normal, normal_θ(μ1, σ1), normal_θ(μ2, σ2)):.6f}")
```

对于正态族，把 $B_F(\theta_2, \theta_1)$ 展开可以恢复出我们熟悉的公式

$$
D_{KL}\left( N(\mu_1, \sigma_1^2) \| N(\mu_2, \sigma_2^2) \right)
= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2} ,
$$ (ef_kl_normal)

当 $\sigma_1 = \sigma_2 = \sigma$ 时，该式化简为我们在最开始四个正态分布的例子中使用的表达式 $(\mu_1 - \mu_2)^2/(2\sigma^2)$。

这个公式的多元版本出现在 {doc}`advanced:entropy` 中。

{doc}`approximation_errors` 逐频率应用 {eq}`ef_kl_normal`，计算两个平稳高斯过程之间的 KL 散度率。

### 费希尔信息是 KL 散度的曲率

当 $\theta_2$ 接近 $\theta_1$ 时，对 {eq}`ef_kl_bregman` 做二阶泰勒展开得到

$$
D_{KL}\left( p_{\theta} \| p_{\theta + \delta} \right) \approx \frac{1}{2} \, \delta^\top \nabla^2 F(\theta) \, \delta .
$$

对于指数族，$\nabla^2 F(\theta)$ 就是费希尔信息矩阵 {eq}`ef_fisher`，因为得分 $\nabla_\theta \log p(X; \theta) = t(X) - \nabla F(\theta)$ 的协方差矩阵为 $\operatorname{Cov}_\theta[t(X)] = \nabla^2 F(\theta)$。

*费希尔信息矩阵衡量邻近分布变得可区分的速度。*

这就解释了我们最初给出的四个正态模型的例子。

当 $\sigma$ 固定时，$\mu$ 的费希尔信息是 $1/\sigma^2$，因此 $\mu$ 的单位变化使分布移动 $\frac{1}{2} \cdot 1^2 / \sigma^2$，当 $\sigma = 0.2$ 时为 $12.5$，当 $\sigma = 2$ 时为 $0.125$。

对于仅在均值上发生的变化，二次公式是精确的。

在期望参数下，费希尔信息矩阵是 $\nabla^2 F^*(\eta) = I(\theta)^{-1}$。

如果 $\theta$ 是另一个参数 $\xi$ 的函数，雅可比矩阵为 $J = \partial \theta / \partial \xi$，那么 $\xi$ 的费希尔信息矩阵就是 $J^\top I(\theta) J$；当 $\xi = \eta$ 时，$J = \nabla^2 F^*(\eta)$，且 $J^\top I(\theta) J = I(\theta)^{-1}$。

两套坐标系下的二次近似在位移大小的二阶意义下是一致的，但在位移较大时会有所不同。

{doc}`specification_tests` 表明，无关备选方案独立性检验的似然比统计量、拉格朗日乘子统计量和豪斯曼统计量都是 $2m$ 倍的 KL 散度，并且是这三种坐标下这种二次近似。

下图展示了泊松族的这一情形，将 $D_{KL}(p_{\lambda_0} \| p_\lambda)$ 与它在自然参数下的二次近似 $\frac{1}{2} \lambda_0 (\log \lambda - \log \lambda_0)^2$，以及在期望参数下的二次近似 $\frac{1}{2} (\lambda - \lambda_0)^2 / \lambda_0$ 进行比较。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Poisson KL divergence and quadratic approximations
    name: fig-ef-fisher
---
λ0 = 4.0
λ_vals = np.linspace(1, 10, 200)
kl_vals = λ0 * np.log(λ0 / λ_vals) + λ_vals - λ0

fig, ax = plt.subplots()
ax.plot(λ_vals, kl_vals, lw=3, label=r"$D_{KL}(p_{\lambda_0} \| p_\lambda)$")
ax.plot(λ_vals, 0.5 * λ0 * np.log(λ_vals / λ0)**2, "--",
        label="quadratic in natural parameter")
ax.plot(λ_vals, 0.5 * (λ_vals - λ0)**2 / λ0, ":",
        label="quadratic in expectation parameter")
ax.set_xlabel(r"$\lambda$")
ax.set_ylim(0, 2.5)
ax.legend(loc="upper center")
plt.show()
```

在 $\lambda_0$ 附近，三条曲线重合；在较远处则会分开。

(ef_fisher_rao)=
### 费希尔-拉奥距离

KL 散度不对称，并且违反三角不等式。

费希尔信息矩阵定义了一种距离。

设 $\theta(\tau)$，$0 \leq \tau \leq 1$，是一族中模型的一条光滑路径。

它的**长度**是

$$
L = \int_0^1 \sqrt{\dot \theta(\tau)^\top I(\theta(\tau)) \, \dot \theta(\tau)} \, d\tau .
$$ (ef_path_length)

**费希尔-拉奥距离**是指该族中连接两个模型的最短路径的长度 {cite}`Rao1945`。

长度 {eq}`ef_path_length` 不依赖于坐标系：在重新参数化下，$\dot \theta = J \dot \xi$，费希尔信息矩阵变为 $J^\top I J$，所以被积函数不变。

费希尔信息在数据的可逆变换或充分统计量变换下不变，因此费希尔-拉奥距离也不变。

对于邻近的模型，费希尔-拉奥距离近似为 $\sqrt{\delta^\top I(\theta) \delta} \approx \sqrt{2 D_{KL}(p_\theta \| p_{\theta + \delta})}$。

对于坐标 $(\mu, \sigma)$ 下的正态族，{eq}`ef_fisher_normal` 给出了长度元的平方：

$$
ds^2 = \frac{d\mu^2 + 2 \, d\sigma^2}{\sigma^2} .
$$

令 $u = \mu / \sqrt{2}$，则 $ds^2 = 2 (du^2 + d\sigma^2) / \sigma^2$，这是庞加莱半平面——双曲几何的一个模型——长度元平方的两倍。

在坐标 $(u, \sigma)$ 下，最短路径是垂直线以及以轴 $\sigma = 0$ 为中心的半圆。

两个正态模型之间的费希尔-拉奥距离为 {cite}`CostaSantosStrapasson2015`

$$
d_{FR}\left(N(\mu_1, \sigma_1^2), N(\mu_2, \sigma_2^2)\right)
= \sqrt{2} \, \operatorname{arccosh}\left( 1 + \frac{(\mu_1 - \mu_2)^2 / 2 + (\sigma_1 - \sigma_2)^2}{2 \sigma_1 \sigma_2} \right) .
$$ (ef_fisher_rao_normal)

我们为最初那四个正态模型计算 {eq}`ef_fisher_rao_normal`。

```{code-cell} ipython3
def fisher_rao_normal(μ1, σ1, μ2, σ2):
    """Fisher–Rao distance (ef_fisher_rao_normal) between two normal models."""
    return np.sqrt(2) * np.arccosh(
        1 + ((μ1 - μ2)**2 / 2 + (σ1 - σ2)**2) / (2 * σ1 * σ2))


def normal_geodesic(μ1, σ1, μ2, σ2, n_points=401):
    """
    Points (μ, σ) on the shortest path between two normal models with
    μ1 ≠ μ2: a semicircle in the (μ/√2, σ) half-plane.
    """
    u1, u2 = μ1 / np.sqrt(2), μ2 / np.sqrt(2)
    center = (u1**2 + σ1**2 - u2**2 - σ2**2) / (2 * (u1 - u2))
    radius = np.hypot(u1 - center, σ1)
    angles = np.linspace(np.arctan2(σ1, u1 - center),
                         np.arctan2(σ2, u2 - center), n_points)
    return np.sqrt(2) * (center + radius * np.cos(angles)), radius * np.sin(angles)


pairs = {"A, B": (0.0, 0.2, 1.0, 0.2), "C, D": (0.0, 2.0, 1.0, 2.0)}
for name, (μa, σa, μb, σb) in pairs.items():
    kl_ab = np.log(σb / σa) + (σa**2 + (μa - μb)**2) / (2 * σb**2) - 0.5
    μ_path, σ_path = normal_geodesic(μa, σa, μb, σb)
    print(f"{name}: Fisher–Rao distance {fisher_rao_normal(μa, σa, μb, σb):.4f}, "
          f"sqrt(2 KL) = {np.sqrt(2 * kl_ab):.4f}, "
          f"midpoint of the shortest path N({μ_path[200]:.2f}, {σ_path[200]:.3f}²)")
```

从 $A$ 到 $B$ 保持 $\sigma = 0.2$ 不变的路径长度为 $1 / 0.2 = 5$，这正等于 $\sqrt{2 D_{KL}(A \| B)}$。

最短路径的长度是 $3.775$；它在途中拓宽了分布，在其中点处到达 $N(0.5, 0.406^2)$。

对于 $C$ 和 $D$，最短路径几乎不弯曲，而费希尔-拉奥距离 $0.497$ 接近于 $\sqrt{2 D_{KL}(C \| D)} = 0.5$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Shortest paths between normal models
    name: fig-ef-fisher-rao
---
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, (μa, σa, μb, σb)) in zip(axes, pairs.items()):
    μ_path, σ_path = normal_geodesic(μa, σa, μb, σb)
    ax.plot(μ_path, σ_path, lw=2, label="shortest path")
    ax.plot([μa, μb], [σa, σb], "k--", lw=1, label=r"path with $\sigma$ fixed")
    ax.plot([μa, μb], [σa, σb], "ko")
    for μ, σ, label in ((μa, σa, name[0]), (μb, σb, name[-1])):
        ax.annotate(label, (μ, σ), xytext=(4, -14), textcoords="offset points")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_ylim(0, 1.3 * σ_path.max())
    ax.legend(loc="lower center")
plt.tight_layout()
plt.show()
```

我们在两套坐标系中测量从 $A$ 到 $B$ 最短路径的长度：一是在 $(\mu, \sigma)$ 下用 {eq}`ef_fisher_normal`，二是在自然参数下用费希尔信息矩阵 $\nabla^2 F(\theta)$。

```{code-cell} ipython3
μ_path, σ_path = normal_geodesic(0.0, 0.2, 1.0, 0.2, n_points=2001)

dμ, dσ = np.diff(μ_path), np.diff(σ_path)
σ_mid = (σ_path[1:] + σ_path[:-1]) / 2
length_μσ = np.sum(np.sqrt(dμ**2 + 2 * dσ**2) / σ_mid)

θ_path = np.array([normal_θ(μ, σ) for μ, σ in zip(μ_path, σ_path)])
dθ = np.diff(θ_path, axis=0)
θ_mid = (θ_path[1:] + θ_path[:-1]) / 2
length_θ = sum(np.sqrt(d @ hessian(F_normal, m) @ d) for d, m in zip(dθ, θ_mid))

print(f"length in (μ, σ):                {length_μσ:.4f}")
print(f"length in natural parameters:    {length_θ:.4f}")
print(f"formula (ef_fisher_rao_normal):  {fisher_rao_normal(0.0, 0.2, 1.0, 0.2):.4f}")
```

这两个长度彼此一致，也与 {eq}`ef_fisher_rao_normal` 一致。

费希尔-拉奥距离依赖于路径可以经过的那个族。

两个正态模型之间的路径可以离开正态族，经过其他密度。

映射 $p \mapsto 2 \sqrt{p}$ 把每个密度送到半径为 $2$ 的球面上的一点，因为 $\int (2 \sqrt{p})^2 \, dx = 4$。

它还把费希尔二次型 $\int (dp)^2 / p \, dx$ 变成欧几里得长度的平方 $\int (d \, 2 \sqrt{p})^2 \, dx$。

因此，当允许所有密度参与时，两个密度之间的最短路径是一段大圆弧，其长度为

$$
d_{FR}(p, q) = 2 \arccos \int \sqrt{p(x) \, q(x)} \, dx ,
$$ (ef_fisher_rao_all)

其中 $\int \sqrt{p q} \, dx$ 是 {doc}`divergence_measures` 中的巴氏系数（Bhattacharyya coefficient）。

{cite:t}`KhanZhang2022` 对有限集合上的分布描述了相同的构造。

```{code-cell} ipython3
for name, (μa, σa, μb, σb) in pairs.items():
    bc = integrate.quad(lambda x: np.sqrt(norm.pdf(x, μa, σa) * norm.pdf(x, μb, σb)),
                        -np.inf, np.inf)[0]
    print(f"{name}: through normal models {fisher_rao_normal(μa, σa, μb, σb):.5f}, "
          f"through all densities {2 * np.arccos(bc):.5f}")
```

对于 $A$ 和 $B$，一条经过非正态密度的路径更短，$3.054$ 对比 $3.775$。

对于 $C$ 和 $D$，两种距离精确到小数点后四位都一致。

沃瑟斯坦距离 {eq}`ef_wasserstein` 选择的是另一类最短路径。

在坐标 $(\mu, \sigma)$ 下它就是欧几里得距离，因此它在两个正态模型之间的最短路径就是它们参数向量之间的直线段。

对于 $A$ 和 $B$，那正是 $\sigma$ 固定的那条路径，即最短路径图中的虚线。

该路径上的每个模型都是 $(1 - \tau) X + \tau Y$（$0 \leq \tau \leq 1$）的分布，其中 $(X, Y)$ 具有能够达到 {eq}`ef_wasserstein` 的联合分布。

{cite:t}`McCann1997` 把这条路径称为**位移插值**（displacement interpolation），{cite:t}`Takatsu2011` 对正态模型研究了它。

下图绘制了从 $A$ 到 $B$ 三条插值路径的中点处的模型：沃瑟斯坦路径、正态族内的费希尔-拉奥路径，以及混合路径 $(1 - \tau) A + \tau B$——{doc}`information_projections` 研究的那种。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Midpoints of three interpolating paths
    name: fig-ef-midpoints
---
x_mid = np.linspace(-1, 2, 600)
μ_fr, σ_fr = normal_geodesic(0.0, 0.2, 1.0, 0.2, n_points=401)
midpoints = {
    "Wasserstein: N(0.5, 0.2²)": norm.pdf(x_mid, 0.5, 0.2),
    f"Fisher–Rao: N({μ_fr[200]:.1f}, {σ_fr[200]:.3f}²)":
        norm.pdf(x_mid, μ_fr[200], σ_fr[200]),
    "mixture: (A + B) / 2": 0.5 * (norm.pdf(x_mid, 0, 0.2) + norm.pdf(x_mid, 1, 0.2)),
}
fig, ax = plt.subplots()
for μ in (0.0, 1.0):
    ax.plot(x_mid, norm.pdf(x_mid, μ, 0.2), color="0.75", lw=1,
            label="A and B" if μ == 0 else None)
for label, density in midpoints.items():
    ax.plot(x_mid, density, lw=2, label=label)
ax.set_xlabel("$x$")
ax.set_ylim(0, 2.9)
ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
plt.show()
```

沃瑟斯坦路径把 $A$ 的质量滑向 $B$ 并保持其形状不变。

费希尔-拉奥路径先把质量展开再重新聚拢。

混合路径让质量停留在原处，而改变两个峰的高度。

(ef_estimation)=
## 估计

### 最大似然匹配矩

设 $x_1, \ldots, x_n$ 是来自指数族中一个模型的独立同分布样本。

平均对数似然为

$$
\frac{1}{n} \sum_{i=1}^n \log p(x_i; \theta) = \theta \cdot \bar t - F(\theta) + \frac{1}{n} \sum_{i=1}^n \log h(x_i),
\qquad
\bar t = \frac{1}{n} \sum_{i=1}^n t(x_i) .
$$

统计量 $t$ 是充分的：除了那个不涉及 $\theta$ 的最后一项外，似然函数只通过 $\bar t$ 依赖于数据。

平均对数似然是 $\theta$ 的凹函数，其一阶条件是 $\nabla F(\hat \theta) = \bar t$。

*期望参数的最大似然估计量是充分统计量的样本均值*，即 $\hat \eta = \bar t$。

当 $\bar t$ 落在 $\nabla F$ 的值域中时，自然参数的最大似然估计量为 $\hat \theta = \nabla F^*(\bar t)$。

当它不落在该值域中时，例如一个没有成功事件的伯努利样本，$\hat \theta$ 就不存在（参见 {ref}`ef_ex_bernoulli`）。

当两者都存在时，估计值 $\hat \eta$ 和 $\hat \theta$ 是同一个估计模型的两组坐标。

由 {eq}`ef_fenchel`，极大化的平均对数似然是 $F^*(\bar t)$ 加上 $\log h(x_i)$ 的均值。

在期望参数下，最大似然估计是平凡的；在自然参数下，它需要对 $\nabla F$ 求逆。

对于伽马族和贝塔族，对 $\nabla F$ 求逆需要数值方法，比如我们上面用过的牛顿迭代法。

### 两种估计量的有限样本行为

两套坐标系中的最大似然估计量在有限样本中的表现不同。

两个估计量指代的是同一个估计模型；偏差和协方差依赖于坐标系，这在 {ref}`ef_invariants` 中的表格里有所记录。

估计量 $\hat \eta = \bar t$ 是无偏的，因为它是一个样本均值；其协方差矩阵是 $\operatorname{Cov}_\theta[t(X)]/n = \nabla^2 F(\theta)/n$。

$\eta$ 的费希尔信息矩阵是 $\nabla^2 F^*(\eta)$，因此 $\eta$ 的无偏估计量的克拉美-拉奥下界是 $\left[\nabla^2 F^*(\eta)\right]^{-1}/n = \nabla^2 F(\theta)/n$。

*在期望参数下，最大似然估计量在任何样本量下都是无偏的，并且恰好达到克拉美-拉奥界。*

自然参数的估计量 $\hat \theta = \nabla F^*(\bar t)$ 是 $\bar t$ 的非线性函数，在有限样本中是有偏的；对于正态族，当 $n > 3$ 时，$E \hat \theta = n \theta / (n - 3)$。

克拉美-拉奥界只适用于无偏估计量，因此对于有偏的 $\hat \theta$，我们把 $I(\theta)^{-1}/n$ 当作一个基准。

在指数族中，一个无偏估计量只有在它所估计的量是 $\eta$ 的仿射函数时，才能在有限样本中达到克拉美-拉奥界。

只有当 $F$ 是二次函数时——例如方差已知的正态族——自然参数 $\theta = \nabla F^*(\eta)$ 才是 $\eta$ 的仿射函数。

对于均值和方差都未知的正态族，$\theta_1 = \eta_1 / (\eta_2 - \eta_1^2)$ 和 $\theta_2 = -1/(2(\eta_2 - \eta_1^2))$ 都不是 $\eta$ 的仿射函数，因此两者都没有无偏估计量能达到该界。

对于正态族，当 $n \to \infty$ 时，$\hat \theta$ 的偏差消失，并且 $n$ 乘以它的协方差矩阵收敛到 $I(\theta)^{-1}$。

一般而言，渐近理论描述的是 $\hat \theta$ 的分布，而它的有限样本矩未必存在，正如 {ref}`ef_ex_bernoulli` 所示。

下面的模拟针对均值 $\mu = 1$、$\sigma = 1.5$ 的正态族展示了这些事实，使用大小为 $n = 30$ 的样本。

椭圆圈出了以真实参数为中心的正态分布 95% 的概率，其协方差矩阵在左侧面板中为 $\nabla^2 F(\theta)/n = I(\theta)/n$，在右侧面板中为 $I(\theta)^{-1}/n$。

```{code-cell} ipython3
def ellipse(center, cov, level=chi2.ppf(0.95, 2), n_points=200):
    """Points on the ellipse (z - center)' cov^{-1} (z - center) = level."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    return center[:, None] + np.linalg.cholesky(level * cov) @ circle


μ, σ, n, R = 1.0, 1.5, 30, 20_000
θ_true = normal_θ(μ, σ)
η_true = np.array([μ, μ**2 + σ**2])
I_θ = hessian(F_normal, θ_true)        # Fisher information for θ

samples = rng.normal(μ, σ, size=(R, n))
η_hat = np.column_stack([samples.mean(axis=1), (samples**2).mean(axis=1)])
σ2_hat = η_hat[:, 1] - η_hat[:, 0]**2
θ_hat = np.column_stack([η_hat[:, 0] / σ2_hat, -1 / (2 * σ2_hat)])

print("expectation parameters")
print("  bias:", np.round(η_hat.mean(axis=0) - η_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(η_hat.T), 3))
print("  n × Cramér–Rao bound = ∇²F(θ):\n", np.round(I_θ, 3))
print("natural parameters")
print("  bias:", np.round(θ_hat.mean(axis=0) - θ_true, 4))
print("  n × sample covariance:\n", np.round(n * np.cov(θ_hat.T), 3))
print("  n × benchmark = I(θ)^{-1}:\n", np.round(np.linalg.inv(I_θ), 3))
print(f"  E θ̂ = nθ/(n-3):", np.round(n * θ_true / (n - 3), 4),
      "  mean of θ̂:", np.round(θ_hat.mean(axis=0), 4))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum likelihood estimates in both parameterizations
    name: fig-ef-cramer-rao
---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
panels = ((axes[0], η_hat, η_true, I_θ / n,
           (r"$\hat\eta_1$", r"$\hat\eta_2$"), "expectation parameters",
           "Cramér–Rao ellipse", "lower right"),
          (axes[1], θ_hat, θ_true, np.linalg.inv(I_θ) / n,
           (r"$\hat\theta_1$", r"$\hat\theta_2$"), "natural parameters",
           r"ellipse from $I(\theta)^{-1}/n$", "lower left"))
for ax, est, truth, cov, labels, title, ell_label, loc in panels:
    ax.scatter(*est[:1000].T, s=3, alpha=0.3)
    ax.plot(*ellipse(truth, cov), "k", lw=1.5, label=ell_label)
    ax.plot(*truth, "r+", ms=14, mew=2, label="true value")
    ax.plot(*est.mean(axis=0), "kx", ms=10, mew=2, label="mean of estimates")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(loc=loc)
plt.tight_layout()
plt.show()
```

在期望参数下，估计值的点云以真实值为中心，其协方差矩阵在模拟误差范围内等于克拉美-拉奥界。

在自然参数下，点云是偏斜的，其均值因偏差 $3\theta/(n - 3)$ 而偏离真实值，其协方差矩阵大于基准值 $I(\theta)^{-1}/n$，只有当 $n$ 增大时才会逐渐逼近该基准。

## 结语

族是一组模型的集合，而参数化为这些模型赋予坐标。

KL 散度和费希尔-拉奥距离既不依赖于坐标，也不依赖于数据的可逆变换。

充分统计量能够保持它们不变。

如果一个光滑密度族可能的结果集合不依赖于其参数，并且它的样本具有维数固定有限的充分统计量，那么该族就是一个指数族。

对于指数族，一个凸函数 $F$ 给出了四个事实：

* 它的梯度把自然参数 $\theta$ 映射到期望参数 $\eta$，而它的共轭 $F^*$ 的梯度把它们映射回去；
* 它的黑塞矩阵是充分统计量的协方差矩阵，也是费希尔信息矩阵；
* 当基测度为常数时，它的共轭是熵的负值；并且
* $F$ 和 $F^*$ 的布雷格曼散度就是 KL 散度。

指数族中的模型是最大熵问题的解。

在所有具有给定均值和方差的密度中，正态密度使熵最大化。

最大熵问题的拉格朗日乘子就是自然参数和影子价格。

逻辑模型的包容值是一个对数配分函数，它的共轭是负熵。

对于正态族，费希尔-拉奥距离是双曲几何中的一种距离。

最优传输的沃瑟斯坦距离衡量概率质量移动的距离；它忽略参数化方式，但依赖于数据的单位。

续篇 {doc}`information_projections` 利用这些事实来描述 KL 散度的几何结构。

## 练习

```{exercise}
:label: ef_ex_gamma

考虑 $(0, \infty)$ 上的密度。

1. 证明在给定 $E[X]$ 和 $E[\log X]$ 的值的约束下使熵最大化的密度是伽马密度，并把伽马族写成 {eq}`ef_family` 的形式。
2. 计算对数配分函数 $F$，并用 {eq}`ef_gradient` 把 $E[X]$ 和 $E[\log X]$ 用形状参数和率参数表示出来。
3. 从形状为 $2.5$、率为 $1.5$ 的伽马分布中抽取 500 个观测值，通过匹配 $X$ 和 $\log X$ 的样本均值，用牛顿法结合黑塞矩阵 {eq}`ef_hessian` 计算最大似然估计量。把你的估计值与用位置参数固定为零的 `scipy.stats.gamma.fit` 得到的结果进行比较。
```

```{solution-start} ef_ex_gamma
:class: dropdown
```

以下是一种解法。

对于第 1 部分，导出 {eq}`ef_maxent_solution` 的论证给出一个正比于 $\exp(\theta_1 x + \theta_2 \log x) = x^{\theta_2} e^{\theta_1 x}$ 的密度，当 $\theta_1 < 0$ 且 $\theta_2 > -1$ 时它在 $(0, \infty)$ 上是可积的。

这就是形状为 $a = \theta_2 + 1$、率为 $b = -\theta_1$ 的伽马密度：

$$
p(x; a, b) = \frac{b^a}{\Gamma(a)} x^{a - 1} e^{-b x} .
$$

因此 $t(x) = (x, \log x)$，$\theta = (-b, a - 1)$，$h(x) = 1$。

对于第 2 部分，

$$
F(\theta) = \log \int_0^\infty x^{\theta_2} e^{\theta_1 x} dx = \log \Gamma(\theta_2 + 1) - (\theta_2 + 1) \log(-\theta_1) .
$$

求导得

$$
E[X] = \frac{\partial F}{\partial \theta_1} = \frac{\theta_2 + 1}{-\theta_1} = \frac{a}{b},
\qquad
E[\log X] = \frac{\partial F}{\partial \theta_2} = \psi(a) - \log b ,
$$

其中 $\psi$ 是双伽马函数。

黑塞矩阵为

$$
\nabla^2 F(\theta) = \begin{bmatrix} a/b^2 & 1/b \\ 1/b & \psi'(a) \end{bmatrix} .
$$

对于第 3 部分，我们对 $\nabla F(\theta) = \bar t$ 应用牛顿法。

我们从矩估计值开始，并且在任何会使 $\Theta$（其中 $a > 0$ 且 $b > 0$）越界的牛顿步长上进行缩短。

```{code-cell} ipython3
def gamma_mle(x, tol=1e-10, max_iter=100):
    """Match the sample means of x and log x by Newton's method in θ."""
    t_bar = np.array([x.mean(), np.log(x).mean()])
    a = x.mean()**2 / x.var()          # method-of-moments starting values
    b = a / x.mean()
    for _ in range(max_iter):
        grad_F = np.array([a / b, digamma(a) - np.log(b)])
        hess_F = np.array([[a / b**2, 1 / b],
                           [1 / b, polygamma(1, a)]])
        step = np.linalg.solve(hess_F, t_bar - grad_F)  # step in (-b, a - 1)
        s = 1.0
        while b - s * step[0] <= 0 or a + s * step[1] <= 0:
            s /= 2                                      # stay inside Θ
        b, a = b - s * step[0], a + s * step[1]
        if max(abs(s * step[0]) / b, abs(s * step[1]) / a) < tol:
            break
    else:
        raise RuntimeError("no convergence")
    return a, b


rng_gamma = np.random.default_rng(6)
x = rng_gamma.gamma(shape=2.5, scale=1 / 1.5, size=500)
a_hat, b_hat = gamma_mle(x)
a_sp, _, scale_sp = gamma_dist.fit(x, floc=0)
print(f"moment matching: shape = {a_hat:.5f}, rate = {b_hat:.5f}")
print(f"scipy fit:       shape = {a_sp:.5f}, rate = {1 / scale_sp:.5f}")
```

两组估计值是一致的。

```{solution-end}
```

```{exercise}
:label: ef_ex_moments4

在充分统计量中加入 $x$ 的更高次幂，可以得到更丰富的指数族，但代价是对数配分函数没有闭式解。

考虑实数轴上满足以下条件的密度：

$$
E[X] = 0, \qquad E[X^2] = 1, \qquad E[X^3] = 0.5, \qquad E[X^4] = 3.5 .
$$

1. 证明具有最大熵的密度具有形式 $\exp\left(\sum_{k=1}^4 \theta_k x^k - F(\theta)\right)$，并解释为什么 $\theta_4 < 0$。
2. 在网格点上用函数 `maxent` 近似求解。检验当你加宽和细化网格时自然参数是否保持不变。
3. 把解的熵与 $N(0, 1)$ 的熵进行比较，并验证 {eq}`ef_normal_gap`。
```

```{solution-start} ef_ex_moments4
:class: dropdown
```

以下是一种解法。

对于第 1 部分，导出 {eq}`ef_maxent_solution` 的论证给出一个正比于 $\exp\left(\sum_{k=1}^4 \theta_k x^k\right)$ 的密度。

只有当系数非零的 $x$ 的最高次幂是偶数且其系数为负时，该密度才是可积的。

如果 $\theta_4 = 0$，可积性将要求 $\theta_3 = 0$，这样密度就是正态的，其 $E[X^3] = 0 \neq 0.5$。

因此 $\theta_4 < 0$。

定义 $F$ 的积分没有闭式解，所以我们用数值方法计算 $\theta$。

对于第 2 部分，我们在三个网格上求解该问题。

```{code-cell} ipython3
c4 = np.array([0.0, 1.0, 0.5, 3.5])
for bound, n_points in ((8, 641), (10, 1601), (12, 2401)):
    grid4 = np.linspace(-bound, bound, n_points)
    T4 = np.column_stack([grid4**k for k in range(1, 5)])
    θ4, p4 = maxent(np.full(grid4.size, 1 / grid4.size), T4, c4)
    print(f"grid [-{bound}, {bound}] with {n_points} points: θ = {np.round(θ4, 5)}")
```

自然参数在小数点后五位上是一致的。

对于第 3 部分，我们在最后一个网格上计算解的熵、其相对于 $N(0, 1)$ 的 KL 散度，以及熵亏缺。

```{code-cell} ipython3
spacing4 = grid4[1] - grid4[0]
density4 = p4 / spacing4
H4 = -p4 @ np.log(density4)
kl4 = p4 @ (np.log(density4) - norm.logpdf(grid4))
print(f"entropy of the solution:  {H4:.5f}")
print(f"entropy of N(0, 1):       {norm.entropy():.5f}")
print(f"H(φ) - H(p):              {norm.entropy() - H4:.5f}")
print(f"KL(p || φ):               {kl4:.5f}")

fig, ax = plt.subplots()
ax.plot(grid4, density4, lw=2, label="maximum entropy density")
ax.plot(grid4, norm.pdf(grid4), "k--", lw=1, label="N(0, 1)")
ax.set_xlim(-5, 5)
ax.set_xlabel("$x$")
ax.legend(loc="upper left")
plt.show()
```

该解具有与 $N(0, 1)$ 相同的均值和方差，因此 {eq}`ef_normal_gap` 适用，熵亏缺等于 KL 散度。

```{solution-end}
```

```{exercise}
:label: ef_ex_poisson

泊松族具有非常数的基测度 $h(x) = 1/x!$。

1. 用 {eq}`ef_conjugate` 证明 $F^*(\eta) = \eta \log \eta - \eta$，并检验 $\nabla F^*$ 是 $\nabla F$ 的反函数。
2. 证明泊松($\lambda$)分布的熵为 $H = \lambda - \lambda \log \lambda + E[\log X!]$，从而 $H = -F^*(\lambda) - E \log h(X)$ 且 $F^* \neq -H$。对 $\lambda \in \{0.5, 3, 20\}$ 通过直接计算 $-\sum_x p(x) \log p(x)$ 用数值方法检验此公式。
3. 用 {eq}`ef_kl_dual` 推导 $D_{KL}(\text{泊松}(\lambda_1) \| \text{泊松}(\lambda_2))$ 的闭式解，并与直接求和进行比较。
```

```{solution-start} ef_ex_poisson
:class: dropdown
```

以下是一种解法。

对于第 1 部分，$F(\theta) = e^\theta$，所以 $\sup_\theta \{\theta \eta - e^\theta\}$ 的一阶条件为 $\eta = e^\theta$，即 $\theta = \log \eta$。

代入可得 $F^*(\eta) = \eta \log \eta - \eta$。

于是 $\nabla F^*(\eta) = \log \eta$，它正是 $\nabla F(\theta) = e^\theta$ 的反函数。

对于第 2 部分，$\log p(x) = \theta x - e^\theta - \log x!$，其中 $\theta = \log \lambda$，所以

$$
H = -E \log p(X) = -\lambda \log \lambda + \lambda + E[\log X!] = -F^*(\lambda) - E \log h(X) .
$$

因为只要 $\lambda > 0$ 就有 $E[\log X!] > 0$，所以 $-F^*(\lambda)$ 比 $H$ 小，如下表的最后一列所示。

```{code-cell} ipython3
k = np.arange(300)
print(f"{'λ':>5} {'direct H':>10} {'formula':>10} {'-F*(λ)':>10}")
for λ in (0.5, 3.0, 20.0):
    p = poisson.pmf(k, λ)
    H_direct = -p @ poisson.logpmf(k, λ)
    H_formula = λ - λ * np.log(λ) + p @ gammaln(k + 1)
    print(f"{λ:5.1f} {H_direct:10.6f} {H_formula:10.6f} "
          f"{-F_star_poisson(λ):10.6f}")
```

对于第 3 部分，{eq}`ef_kl_dual` 给出

$$
D_{KL}(p_{\lambda_1} \| p_{\lambda_2})
= F^*(\lambda_1) - F^*(\lambda_2) - \log \lambda_2 \, (\lambda_1 - \lambda_2)
= \lambda_1 \log \frac{\lambda_1}{\lambda_2} - \lambda_1 + \lambda_2 ,
$$

这正是我们用来绘制 {numref}`fig-ef-fisher` 的公式。

```{code-cell} ipython3
λ1, λ2 = 3.0, 5.0
direct = poisson.pmf(k, λ1) @ (poisson.logpmf(k, λ1) - poisson.logpmf(k, λ2))
print(f"direct sum:  {direct:.8f}")
print(f"closed form: {λ1 * np.log(λ1 / λ2) - λ1 + λ2:.8f}")
```

```{solution-end}
```

```{exercise}
:label: ef_ex_bernoulli

对伯努利族重复两套坐标系的比较。

1. 取 $\pi = 0.2$ 和 $n = 25$，模拟大量样本，计算 $\hat \eta = \bar x$ 和 $\hat \theta = \log(\hat \eta / (1 - \hat \eta))$。
2. 验证 $\hat \eta$ 是无偏的，其方差等于克拉美-拉奥界 $\pi(1 - \pi)/n$。
3. 有多少比例的样本会产生无穷的 $\hat \theta$？在 $\hat \theta$ 有限的样本中，把 $\hat \theta$ 的均值和方差与 $\theta$ 以及基准值 $1/(n \pi (1 - \pi))$ 进行比较。当 $n = 400$ 时这些比较会如何变化？
```

```{solution-start} ef_ex_bernoulli
:class: dropdown
```

以下是一种解法。

```{code-cell} ipython3
def bernoulli_experiment(π, n, rng, R=100_000):
    η_hat = rng.binomial(n, π, size=R) / n
    θ = np.log(π / (1 - π))
    finite = (η_hat > 0) & (η_hat < 1)
    θ_hat = np.log(η_hat[finite] / (1 - η_hat[finite]))
    print(f"n = {n}")
    print(f"  η̂: mean {η_hat.mean():.4f} (π = {π}), "
          f"variance {η_hat.var():.6f} (bound {π * (1 - π) / n:.6f})")
    print(f"  fraction with infinite θ̂: {1 - finite.mean():.4f}")
    print(f"  finite θ̂: mean {θ_hat.mean():.4f} (θ = {θ:.4f}), "
          f"variance {θ_hat.var():.4f} "
          f"(benchmark {1 / (n * π * (1 - π)):.4f})")


rng_bern = np.random.default_rng(2)
bernoulli_experiment(0.2, 25, rng_bern)
bernoulli_experiment(0.2, 400, rng_bern)
```

当 $n = 25$ 时，大约 $0.4$ 百分比的样本不含任何成功事件。

在那些样本中，$\bar x = 0$ 落在 $\nabla F$ 值域（即区间 $(0, 1)$）的边界上，所以 $\theta$ 的最大似然估计量不存在；公式给出 $\hat \theta = -\infty$。

在其余样本中，$\hat \theta$ 向下偏，其方差比基准值 $1/(n \pi (1 - \pi))$ 高出超过 20%。

当 $n = 400$ 时，无穷估计值基本上不再出现，偏差以及方差与基准之间的差距都很小，与渐近理论的预测一致。

与此同时，$\hat \eta$ 在两种样本量下都是无偏的，并且达到了克拉美-拉奥界。

```{solution-end}
```

```{exercise}
:label: ef_ex_conjugate

贝叶斯法则在自然参数下具有简单的形式。

1. 设 $x_1, \ldots, x_n$ 是独立同分布的伯努利($\pi$)抽样，设 $\pi$ 的先验是贝塔($a$, $b$)分布。利用上面表格中的贝塔那一行，把先验写成贝塔族中自然参数为 $(a - 1, b - 1)$ 的分布，并证明后验是同一族中的一个分布，其自然参数为先验的自然参数加上 $\left(\sum_i x_i, \, n - \sum_i x_i\right)$。
2. 更一般地，解释为什么当对数似然在参数的某些函数下是线性的，并且先验是以这些函数作为充分统计量的指数族中的一个分布时，后验也在同一族中，而贝叶斯法则对其自然参数做加法。
3. 从贝塔(1, 1)先验出发，用 $\pi = 0.3$ 的抽样，作为 $n$ 的函数绘制 $\pi$ 的后验均值，并附上两个后验标准差的带状区域。在 $n = 200$ 次抽样之后，用 {eq}`ef_gradient` 和 {eq}`ef_hessian` 计算 $\log \pi$ 的后验均值和方差，并用蒙特卡洛方法进行检验。

贝塔-伯努利模型出现在 {doc}`exchangeable`、{doc}`likelihood_bayes` 和 {doc}`merging_of_opinions` 中。
```

```{solution-start} ef_ex_conjugate
:class: dropdown
```

以下是一种解法。

对于第 1 部分，先验密度正比于 $\pi^{a - 1}(1 - \pi)^{b - 1} = \exp\left( (a - 1) \log \pi + (b - 1) \log(1 - \pi) \right)$。

$n$ 次抽样中有 $s = \sum_i x_i$ 次成功的似然为 $\pi^s (1 - \pi)^{n - s} = \exp\left( s \log \pi + (n - s) \log (1 - \pi) \right)$。

两者相乘，后验正比于 $\exp\left( (a - 1 + s) \log \pi + (b - 1 + n - s) \log(1 - \pi) \right)$，这是一个贝塔($a + s$, $b + n - s$)密度。

它的自然参数为 $(a - 1 + s, b - 1 + n - s)$。

对于第 2 部分，贝叶斯法则表明，在相差一个常数的意义下，对数后验等于对数先验加对数似然。

如果对数似然在参数的某些函数（这里是 $(\log \pi, \log(1 - \pi))$）下是线性的，并且对数先验在同样的函数下也是线性的，那么对数后验也是线性的，且系数相加。

那些以参数的这些函数作为充分统计量的先验族，被称为**共轭**族。

这里"共轭"一词的用法与凸共轭 $F^*$ 无关。

对于第 3 部分，$\pi$ 的后验均值是 $(a + s)/(a + b + n)$，后验方差是 $(a + s)(b + n - s) / \left((a + b + n)^2 (a + b + n + 1)\right)$。

```{code-cell} ipython3
rng_bayes = np.random.default_rng(3)
π_true, a0, b0 = 0.3, 1.0, 1.0
x = rng_bayes.binomial(1, π_true, size=500)
s = np.cumsum(x)
n_vals = np.arange(1, x.size + 1)
a_post, b_post = a0 + s, b0 + n_vals - s

post_mean = a_post / (a_post + b_post)
post_sd = np.sqrt(a_post * b_post
                  / ((a_post + b_post)**2 * (a_post + b_post + 1)))
fig, ax = plt.subplots()
ax.plot(n_vals, post_mean, label="posterior mean")
ax.fill_between(n_vals, post_mean - 2 * post_sd, post_mean + 2 * post_sd,
                alpha=0.3, label="± 2 posterior standard deviations")
ax.axhline(π_true, color="k", ls="--", lw=1, label=r"true $\pi$")
ax.set_xlabel("$n$")
ax.set_ylim(0, 1)
ax.legend()
plt.show()
```

在 $n = 200$ 次抽样之后，后验是贝塔($a$, $b$)分布，其中 $(a, b) = (a_0 + s, b_0 + 200 - s)$。

将 {eq}`ef_gradient` 应用到 $F(\theta) = \log B(\theta_1 + 1, \theta_2 + 1)$ 上，得到 $E[\log \pi] = \psi(a) - \psi(a + b)$，而 {eq}`ef_hessian` 给出 $\operatorname{Var}[\log \pi] = \psi'(a) - \psi'(a + b)$。

```{code-cell} ipython3
a, b = a_post[199], b_post[199]
draws = rng_bayes.beta(a, b, size=200_000)
print(f"E[log π]:   formula {digamma(a) - digamma(a + b):.5f}, "
      f"Monte Carlo {np.log(draws).mean():.5f}")
print(f"Var[log π]: formula {polygamma(1, a) - polygamma(1, a + b):.5f}, "
      f"Monte Carlo {np.log(draws).var():.5f}")
```

```{solution-end}
```

```{exercise}
:label: ef_ex_jeffreys

一个族的杰弗里斯先验具有正比于 $\sqrt{\det I(\theta)}$ 的密度。

1. 对于坐标为 $\pi$ 的伯努利族，证明 $I(\pi) = 1 / (\pi (1 - \pi))$，并证明杰弗里斯先验是贝塔($1/2$, $1/2$)分布。
2. 在自然参数 $\theta = \log(\pi / (1 - \pi))$ 下，证明 $I(\theta) = \pi (1 - \pi)$。证明 $\theta$ 下密度正比于 $\sqrt{\pi (1 - \pi)}$ 的杰弗里斯先验，正是当 $\pi$ 服从贝塔($1/2$, $1/2$)分布时 $\theta$ 的分布。
3. 从 $(0, 1)$ 上的均匀分布以及贝塔($1/2$, $1/2$)分布中抽取 $\pi$，把每次抽样都变换为 $\theta$，并把 $\theta$ 的直方图与变量替换公式所给出的密度进行比较。$\pi$ 上的均匀先验在 $\theta$ 下是平坦的吗？
```

```{solution-start} ef_ex_jeffreys
:class: dropdown
```

以下是一种解法。

对于第 1 部分，得分为 $x / \pi - (1 - x) / (1 - \pi)$，其方差为 $\pi / \pi^2 + (1 - \pi) / (1 - \pi)^2 = 1 / (\pi (1 - \pi))$。

杰弗里斯先验具有正比于 $\pi^{-1/2} (1 - \pi)^{-1/2}$ 的密度，即贝塔($1/2$, $1/2$)密度。

对于第 2 部分，$F(\theta) = \log(1 + e^\theta)$，所以由 {eq}`ef_hessian`，$I(\theta) = F''(\theta) = \pi (1 - \pi)$。

因为 $d\pi / d\theta = \pi (1 - \pi)$，变量替换公式把贝塔($1/2$, $1/2$)密度变为 $\theta$ 的密度

$$
\frac{\pi^{-1/2} (1 - \pi)^{-1/2}}{B(1/2, 1/2)} \, \pi (1 - \pi) = \frac{\sqrt{\pi (1 - \pi)}}{B(1/2, 1/2)}
$$

它正比于 $\sqrt{I(\theta)}$。

同样的变量替换把 $\pi$ 上的均匀密度变为 $\theta$ 的逻辑斯谛密度 $\pi (1 - \pi) = e^\theta / (1 + e^\theta)^2$。

```{code-cell} ipython3
rng_jeffreys = np.random.default_rng(4)
R_draws = 200_000
θ_vals = np.linspace(-8, 8, 400)
π_vals = 1 / (1 + np.exp(-θ_vals))
cases = {
    "uniform prior on π": (rng_jeffreys.uniform(size=R_draws),
                           π_vals * (1 - π_vals)),
    "Jeffreys prior, Beta(1/2, 1/2)": (rng_jeffreys.beta(0.5, 0.5, size=R_draws),
                                       np.sqrt(π_vals * (1 - π_vals)) / np.pi),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (name, (π_draws, density)) in zip(axes, cases.items()):
    θ_draws = np.log(π_draws / (1 - π_draws))
    counts, edges = np.histogram(θ_draws, bins=80, range=(-8, 8))
    ax.stairs(counts / (R_draws * np.diff(edges)), edges, fill=True, alpha=0.4,
              label=r"histogram of $\theta$")
    ax.plot(θ_vals, density, "k", lw=1.5, label="change of variables")
    ax.set_title(name)
    ax.set_xlabel(r"$\theta$")
    ax.legend()
plt.tight_layout()
plt.show()
```

在代码中，`np.pi` 是 $B(1/2, 1/2) = \Gamma(1/2)^2 = 3.14159\ldots$。

$\pi$ 上的均匀先验在 $\theta$ 下并不平坦：它把大部分概率放在 $\lvert \theta \rvert < 3$ 这一区域。

$\theta$ 上的平坦先验是非正常的（improper）；在 $\pi$ 下它的密度正比于 $1 / (\pi (1 - \pi))$。

杰弗里斯先验在两套坐标系下指代的是模型上的同一个分布。

```{solution-end}
```