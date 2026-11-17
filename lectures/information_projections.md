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
  title: 信息投影
  headings:
    Overview: 概览
    A three-point identity: 一个三点恒等式
    Two kinds of straight lines: 两种直线
    Two projections and two Pythagorean theorems: 两种投影和两个勾股定理
    Two projections and two Pythagorean theorems::The e-projection onto a linear family: 线性族上的 e-投影
    Two projections and two Pythagorean theorems::The m-projection onto an exponential family: 指数族上的 m-投影
    Maximum likelihood is an m-projection: 最大似然是一个 m-投影
    Maximum likelihood is an m-projection::Correct specification: 模型设定正确
    Maximum likelihood is an m-projection::Misspecification: 模型设定错误
    Independence and mutual information: 独立性与互信息
    Maximum entropy and large deviations: 最大熵与大偏差
    Worst-case models in robust control: 稳健控制中的最坏情况模型
    'Matrix balancing: RAS and Sinkhorn': 矩阵平衡：RAS 和辛克霍恩
    'Matrix balancing: RAS and Sinkhorn::Updating an input–output table': 更新投入产出表
    'Matrix balancing: RAS and Sinkhorn::Entropy-regularized optimal transport': 熵正则化最优传输
    Chernoff information: 切尔诺夫信息
    Concluding remarks: 结束语
    Exercises: 练习
---

(information_projections)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 信息投影

```{contents} Contents
:depth: 2
```

## 概览

在欧几里得几何中，一个点在一个集合上的投影是该集合中离该点最近的那个点。

当这个集合是一条直线或一个平面时，投影是唯一的，连接该点与其投影的线段与该集合成直角相交，并且勾股定理成立。

本讲座描述了概率分布的类似结果，用 Kullback–Leibler（KL）散度代替平方距离。

一个分布在分布集合 $\mathcal{S}$ 上有两种投影，因为 KL 散度不是对称的：

* $q$ 在 $\mathcal{S}$ 上的 **e-投影** 是使 $D_{KL}(p \| q)$ 最小化的 $p \in \mathcal{S}$，因此被投影的分布是第二个参数；且
* $p$ 在 $\mathcal{S}$ 上的 **m-投影** 是使 $D_{KL}(p \| q)$ 最小化的 $q \in \mathcal{S}$，因此被投影的分布是第一个参数。

有两种"平坦"的分布集合可以充当直线和平面：

* **线性族**，例如使某个统计量具有规定均值的分布集合；线性族对其任意两个成员的每个混合 $(1 - a) p_0 + a p_1$ 都包含在内，对于使该混合是概率分布的每个实数 $a$ 都成立；以及
* **指数族**，在 {doc}`exponential_families` 中描述；指数族包含其任意两个成员的每个归一化几何混合，对于使该几何混合可以被归一化的每个实数 $a$ 都成立。

线性族上的 e-投影和指数族上的 m-投影都是唯一的，且各自满足一个勾股定理。

这两种投影都有简单的形式：

* 由矩条件定义的线性族上 $q$ 的 e-投影是 $q$ 的一个**指数倾斜**，即一个正比于 $q(x) e^{\theta \cdot t(x)}$ 的分布，其中向量 $\theta$ 满足矩条件；且
* 指数族上 $p$ 的 m-投影是**匹配矩**：它是该族中使充分统计量与 $p$ 具有相同均值的成员。

这两个事实组织起经济学家使用的若干思想：

* 最大似然估计，在模型设定正确和模型设定错误的情况下（m-投影）；
* 互信息，用于衡量随机变量之间的依赖性（一个 m-投影）；
* 最大熵、萨诺夫定理和大偏差（e-投影）；
* 稳健控制中的最坏情况模型（e-投影）；
* 用于更新投入产出表的 RAS 方法，以及用于熵正则化最优传输的辛克霍恩算法（重复的 e-投影）；以及
* 支配基于似然比检验误差率的切尔诺夫信息（同时对两个分布的一个 e-投影）。

稳健控制的最坏情况模型以及 RAS 和辛克霍恩问题的解都是达到 {doc}`exponential_families` 的 logit 公式 {eq}`ef_logit_base` 的那种指数倾斜。

本讲座建立在 {doc}`exponential_families` 和 {doc}`divergence_measures` 的基础之上。

{cite:t}`Csiszar1975` 发展了 e-投影的理论，他称之为 I-投影。

{cite:t}`CsiszarShields2004` 综述了这一理论及其统计应用。

{cite:t}`Nielsen2018` 对信息投影做了简短介绍，而 {cite:t}`amari_nagaoka2000` 将其作为信息几何的一部分来处理。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq, linprog, minimize, minimize_scalar
from scipy.special import logsumexp, betaln, digamma
from scipy.stats import norm, poisson, chi2
import warnings
```

我们用 $D(p \| q)$ 表示 KL 散度 $D_{KL}(p \| q)$，并且和 {doc}`exponential_families` 中一样，将其读作"$p$ 相对于 $q$ 的散度"。

我们用 $q_\theta$ 表示指数族的成员，因为它们通常出现在 $D$ 的第二个参数中。

我们的大多数例子使用有限集上的分布。

下面的函数计算 KL 散度，构造按比例 $e^{\theta \cdot t(x)}$ 提高每个结果 $x$ 的概率的指数倾斜，并找到使标量统计量具有规定均值的倾斜。

这些函数是 {doc}`exponential_families` 中同名函数的推广：`tilt` 也接受向量值统计量，`match_mean` 接受其求根器的括号区间。

```{code-cell} ipython3
def kl(p, q):
    """KL divergence of distributions (or arrays) p and q of equal shape."""
    p, q = np.ravel(p), np.ravel(q)
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("probabilities must be nonnegative")
    mask = p > 0                       # use the convention 0 log 0 = 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def tilt(q, t, θ):
    """The distribution proportional to q(x) exp(θ · t(x)).

    t is either a vector of values t(x) or an array whose rows are t(x).
    """
    T = np.reshape(t, (len(q), -1))
    log_w = np.log(q) + T @ np.atleast_1d(θ)
    return np.exp(log_w - logsumexp(log_w))


def match_mean(q, t, c, bracket=(-50, 50)):
    """Find θ such that the tilt of q has E[t(X)] = c, for scalar t."""
    θ = brentq(lambda θ: tilt(q, t, θ) @ t - c, *bracket)
    return θ, tilt(q, t, θ)


rng = np.random.default_rng(1234)
```

## 一个三点恒等式

设 $p$、$q$ 和 $r$ 是有限集上的三个概率分布，$q$ 和 $r$ 对每个结果都赋予正概率。

展开 KL 散度的定义可得恒等式

$$
D(p \| r) = D(p \| q) + D(q \| r) + \sum_x \left( p(x) - q(x) \right) \left( \log q(x) - \log r(x) \right) .
$$ (ip_three_point)

欧几里得的类似物是

$$
\lVert p - r \rVert^2 = \lVert p - q \rVert^2 + \lVert q - r \rVert^2 + 2 (p - q) \cdot (q - r) .
$$

欧几里得交叉项是两个位移的内积，当从 $q$ 到 $p$ 以及从 $q$ 到 $r$ 的线段互相垂直时它消失。

在 {eq}`ip_three_point` 中，交叉项是用不同坐标测量的两个位移的内积：$p - q$ 是概率中的位移，而 $\log q - \log r$ 是对数概率中的位移。

用 {doc}`exponential_families` 的语言来说，有限集 $\{x_0, x_1, \ldots\}$ 上具有完全支撑的分布形成一个指数族，其期望参数是概率 $p(x)$，其自然参数是对数比 $\log(p(x)/p(x_0))$，对于 $x \neq x_0$。

概率是期望参数；对数概率在归一化意义下是自然参数。

当交叉项消失时，KL 散度满足一个**勾股定理**，

$$
D(p \| r) = D(p \| q) + D(q \| r) .
$$

我们用随机抽取的分布来验证 {eq}`ip_three_point`。

```{code-cell} ipython3
p, q, r = rng.dirichlet(np.ones(5), size=3)
lhs = kl(p, r)
rhs = kl(p, q) + kl(q, r) + (p - q) @ (np.log(q) - np.log(r))
print(f"D(p || r) = {lhs:.10f}")
print(f"right side = {rhs:.10f}")
```

本讲座的其余部分讨论交叉项消失的分布集合。

我们针对有限集上的分布陈述并证明下面的定理。

恒等式 {eq}`ip_three_point` 以及我们由它推出的一切，只要散度有限，也对密度成立，只需用积分代替求和即可。

## 两种直线

从一个分布 $p_0$ 到一个分布 $p_1$ 有两条自然的路径。

**混合路径**沿着概率空间中的一条直线移动：

$$
p_a = (1 - a) \, p_0 + a \, p_1 .
$$

**指数路径**沿着对数概率空间中的一条直线移动然后归一化：

$$
\log p_a = (1 - a) \log p_0 + a \log p_1 - F(a),
$$

其中 $F(a) = \log \sum_x p_0(x)^{1 - a} p_1(x)^a$。

$[0, 1]$ 中的 $a$ 值描绘出从 $p_0$ 到 $p_1$ 的路径；其他 $a$ 值将路径延伸到其端点之外，指数路径对每个实数 $a$ 都可延伸，而混合路径只要 $p_a$ 保持非负就可延伸。

指数路径是一个具有充分统计量 $\log(p_1 / p_0)$ 和自然参数 $a$ 的单参数指数族。

**测地线**是几何学中直线的类似概念。

信息几何学家称这些路径为 **m-测地线**（代表混合）和 **e-测地线**（代表指数）。

在 {eq}`ip_three_point` 之后所述的坐标系中，m-测地线是期望参数中的一条直线，e-测地线是自然参数中的一条直线。

一个分布集合称为 **m-平坦**，如果每当它包含 $p_0$ 和 $p_1$ 时，它就包含混合路径上使 $p_a$ 成为概率分布的每个实数 $a$ 所对应的 $p_a$，而不仅仅是 $a \in [0, 1]$ 的情形；称为 **e-平坦**，如果同样的结论对指数路径成立。

一个包含 $a \in [0, 1]$ 的混合线段、但不一定包含过其成员的整条直线的集合，只是普通意义上的凸集。

在闭凸集上的 e-投影仍然是唯一的，但下面的勾股恒等式 {eq}`ip_pythagoras_e` 弱化为不等式 $D(p \| q) \geq D(p \| p^*) + D(p^* \| q)$ {cite}`Csiszar1975`。

有两类集合是主要的例子。

* **线性族** $\mathcal{L} = \{ p : E_p[t(X)] = c \}$，即使统计量 $t$ 具有规定均值的分布集合，是 m-平坦的，因为如果 $p_0$ 和 $p_1$ 满足 $E[t(X)] = c$，那么对于每个实数 $a$，$(1 - a) p_0 + a p_1$ 也满足该条件。
* **指数族** $\mathcal{E} = \{ q_\theta : q_\theta(x) \propto h(x) \exp(\theta \cdot t(x)) \}$ 是 e-平坦的，因为对于使归一化几何混合存在的每个实数 $a$，$q_{\theta_0}$ 和 $q_{\theta_1}$ 的归一化几何混合的对数是 $\log h + ((1 - a) \theta_0 + a \theta_1) \cdot t$ 加上一个常数。

线性族是期望参数（即概率）空间中的一条直线或平面。

当 $h > 0$ 处处成立时，指数族是自然参数空间中的一条直线或平面，因为 $\log(q_\theta(x) / q_\theta(x_0)) = \log(h(x)/h(x_0)) + \theta \cdot (t(x) - t(x_0))$ 是 $\theta$ 的仿射函数。

三个结果上的分布存在于一个三角形中，即概率单纯形，我们可以在其中绘制两种集合。

下面的函数将三个结果上的分布映射到一个等边三角形中的一个点，该三角形的角代表三个结果。

每个角是将概率 1 赋予某个结果的分布，中心是均匀分布，点 $p$ 赋予结果 $x$ 的概率正比于从 $p$ 到与角 $x$ 相对的边的距离。

```{code-cell} ipython3
corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def to_triangle(p):
    """Map distributions on three outcomes (rows of p) to the plane."""
    return np.atleast_2d(p) @ corners


def draw_simplex(ax, labels=("$x = 0$", "$x = 1$", "$x = 2$")):
    ax.plot(*np.vstack([corners, corners[0]]).T, color="0.6", lw=1)
    offsets = [(-0.07, -0.05), (0.02, -0.05), (-0.03, 0.03)]
    for corner, label, (dx, dy) in zip(corners, labels, offsets):
        ax.text(corner[0] + dx, corner[1] + dy, label)
    ax.set_aspect("equal")
    ax.axis("off")


def mixture_path(p0, p1, a):
    return (1 - a)[:, None] * p0 + a[:, None] * p1


def exponential_path(p0, p1, a):
    log_w = (1 - a)[:, None] * np.log(p0) + a[:, None] * np.log(p1)
    return np.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))
```

下图绘制了两个分布之间的两条路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Mixture and exponential paths
    name: fig-ip-paths
---
p0 = np.array([0.80, 0.15, 0.05])
p1 = np.array([0.05, 0.15, 0.80])
a = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(6, 5.5))
draw_simplex(ax)
ax.plot(*to_triangle(mixture_path(p0, p1, a)).T, lw=2, label="mixture path")
ax.plot(*to_triangle(exponential_path(p0, p1, a)).T, lw=2,
        label="exponential path")
ax.plot(*to_triangle(np.vstack([p0, p1])).T, "ko")
ax.text(*to_triangle(p0)[0] + [0.02, 0.0], "$p_0$")
ax.text(*to_triangle(p1)[0] + [0.02, 0.0], "$p_1$")
ax.legend(loc="upper right")
plt.show()
```

我们以概率坐标绘制单纯形，因此混合路径是直的。

指数路径向中间结果弯曲。

两个概率的几何平均数在其中任何一个很小时都会很小，这惩罚了结果 0 和 2，每个结果在两个分布中的一个分布下都是不太可能的。

## 两种投影和两个勾股定理

### 线性族上的 e-投影

设 $q$ 是具有完全支撑的分布，设 $\mathcal{L} = \{ p : E_p[t(X)] = c \}$ 是包含一个具有完全支撑的分布的线性族。

这恰好在 $c$ 位于点 $t(x)$ 的凸包的**相对内部**时成立，即在包含该凸包的最小仿射子空间的子集意义下的内部。

同样的条件保证了下面定理中的倾斜存在。

```{prf:theorem} e-投影
:label: ip_thm_e_projection

使 $D(p \| q)$ 在 $p \in \mathcal{L}$ 上最小化的分布 $p^* \in \mathcal{L}$ 是指数倾斜

$$
p^*(x) = q(x) \exp\left( \theta \cdot t(x) - F(\theta) \right),
\qquad
F(\theta) = \log \sum_x q(x) \exp(\theta \cdot t(x)) ,
$$

其中 $\theta$ 的选取使得 $E_{p^*}[t(X)] = c$。

此外，对于每个 $p \in \mathcal{L}$，

$$
D(p \| q) = D(p \| p^*) + D(p^* \| q) .
$$ (ip_pythagoras_e)
```

```{prf:proof}
用 $(p, p^*, q)$ 代替 $(p, q, r)$ 应用 {eq}`ip_three_point`。

交叉项为

$$
\sum_x \left( p(x) - p^*(x) \right) \left( \theta \cdot t(x) - F(\theta) \right) = \theta \cdot (c - c) - F(\theta) (1 - 1) = 0 ,
$$

这证明了 {eq}`ip_pythagoras_e`。

由于除非 $p = p^*$ 否则 $D(p \| p^*) > 0$，倾斜 $p^*$ 是唯一的最小化点。
```

当 $\mathcal{L}$ 的每个成员都对某些结果赋予零概率时（例如均值为 6 的骰子），e-投影仍然存在且唯一。

它位于倾斜族之外，是 $\theta$ 发散的倾斜序列的一个极限。

$q$ 的 e-投影位于由 $q$ 和统计量 $t$ 生成的指数族上，位于该族与 $\mathcal{L}$ 相交的点处。

在证明中，交叉项将沿 $\mathcal{L}$ 分布的概率位移 $p - p^*$，与沿从 $q$ 到 $p^*$ 的 e-测地线分布的对数概率位移 $\log p^* - \log q$ 配对。

这种配对恰好是由 $p^*$ 处的费舍尔信息所定义的内积。

对于在 $p^*$ 处以概率表示的切向量 $u$ 和 $v$，该内积是 $\sum_x u(x) v(x) / p^*(x)$。

取 $u = p - p^*$，即 $\mathcal{L}$ 指向 $p$ 的方向，取 $v$ 为 e-测地线 $p_s \propto q^{1 - s} (p^*)^s$ 在 $p^*$ 处的切向量。

那么 $v(x)/p^*(x) = \log p^*(x) - \log q(x) - D(p^* \| q)$，常数 $D(p^* \| q)$ 在内积中会消失，因为 $\sum_x u(x) = 0$。

交叉项消失正是从 $q$ 出发的 e-测地线与 $\mathcal{L}$ 成直角相遇的精确意义。

我们称 $p^*$ 为 e-投影，因为 $q$ 是沿 e-测地线到达 $\mathcal{L}$ 的。

当 $q$ 是 $m$ 个结果上的均匀分布时，$D(p \| q) = \log m - H(p)$。

均匀分布的 e-投影即为 {doc}`exponential_families` 中的最大熵分布，{eq}`ip_pythagoras_e` 变为那里证明的熵差公式。

对于一般的 $q$，e-投影最大化相对于 $q$ 的熵：它最小化 $D(p \| q)$，即 $p$ 相对于 $q$ 的**相对熵**。

### 指数族上的 m-投影

现在设 $\mathcal{E} = \{ q_\theta \}$ 是具有充分统计量 $t$ 和基准测度 $h$ 的指数族，设 $p$ 是在 $h$ 为零的地方也赋予零概率的任意分布。

若无此条件，则对每个 $\theta$，$D(p \| q_\theta)$ 都将是无穷大。

```{prf:theorem} m-投影
:label: ip_thm_m_projection

如果 $q_{\theta^*} \in \mathcal{E}$ 匹配充分统计量的均值，

$$
E_{q_{\theta^*}}[t(X)] = E_p[t(X)] ,
$$

那么 $q_{\theta^*}$ 是使 $D(p \| q_\theta)$ 在 $\theta$ 上最小化的唯一最小值点，且对于每个 $\theta$，

$$
D(p \| q_\theta) = D(p \| q_{\theta^*}) + D(q_{\theta^*} \| q_\theta) .
$$ (ip_pythagoras_m)
```

```{prf:proof}
用 $(p, q_{\theta^*}, q_\theta)$ 代替 $(p, q, r)$ 应用 {eq}`ip_three_point`。

因为 $\log q_{\theta^*}(x) - \log q_\theta(x) = (\theta^* - \theta) \cdot t(x) - F(\theta^*) + F(\theta)$，交叉项为

$$
(\theta^* - \theta) \cdot \left( E_p[t(X)] - E_{q_{\theta^*}}[t(X)] \right) = 0 ,
$$

这证明了 {eq}`ip_pythagoras_m`。

由于除非 $\theta = \theta^*$ 否则 $D(q_{\theta^*} \| q_\theta) > 0$，$q_{\theta^*}$ 是唯一的最小化点。
```

反过来，每个最小化点都匹配矩。

展开定义给出 $D(p \| q_\theta) = F(\theta) - \theta \cdot E_p[t(X)]$ 加上不依赖于 $\theta$ 的项，因此最小值的一阶条件是 $\nabla F(\theta) = E_p[t(X)]$。

当且仅当 $E_p[t(X)]$ 位于 $\nabla F$ 的值域中时，m-投影才存在。

例如，把所有概率都赋予 $1$ 的分布在伯努利族上的 m-投影不存在。

出于同样的原因，一个没有失败的伯努利样本没有自然参数的最大似然估计。

$p$ 的 m-投影位于与 $p$ 共享 $t$ 均值的分布的线性族上，位于该线性族与 $\mathcal{E}$ 相交的点处。

这里角色互换了：交叉项将沿从 $p$ 到 $q_{\theta^*}$ 的 m-测地线（它留在该线性族内）的概率位移，与沿 $\mathcal{E}$ 的自然参数位移配对。

从 $p$ 出发的 m-测地线与 $\mathcal{E}$ 成直角相遇。

我们称 $q_{\theta^*}$ 为 m-投影，因为 $p$ 是沿 m-测地线到达 $\mathcal{E}$ 的。

在两个定理中，投影都是直角的顶点，即出现在右侧两项中的那个点。

```{note}
文献中的约定各不相同。

检查某个来源在 KL 散度的哪个参数上最小化，以及它将哪个正交条件附加到哪个勾股恒等式上。

设 $p$、$q$ 和 $r$ 是具有对数配分函数 $F$ 和布雷格曼散度 $B_F$ 的指数族的成员，如 {doc}`exponential_families` 中所述，其自然参数为 $\theta_p$、$\theta_q$、$\theta_r$，期望参数为 $\eta_p$、$\eta_q$、$\eta_r$。

那么

$$
B_F(\theta_p, \theta_r) = B_F(\theta_p, \theta_q) + B_F(\theta_q, \theta_r)
\iff (\theta_p - \theta_q) \cdot (\eta_q - \eta_r) = 0
$$

以及

$$
D(p \| r) = D(p \| q) + D(q \| r)
\iff (\eta_p - \eta_q) \cdot (\theta_q - \theta_r) = 0 .
$$

这两个恒等式是同一个命题，只是 $p$ 和 $r$ 互换了：根据 {doc}`exponential_families` 的 {eq}`ef_kl_bregman`，第一个说的是 $D(r \| p) = D(r \| q) + D(q \| p)$。

在两种记号中以相同顺序书写字母会附加不同的正交条件。

{cite:t}`Nielsen2018`（在第 322–323 页的正文中以及图 2 中印出的公式里）和 {cite:t}`Nielsen2020`（在图 6 的面板标签中）将每个恒等式与另一个恒等式的条件配对；他们对 e-投影和 m-投影的定义，以及 {cite:t}`Nielsen2020` 的定理 6，都是正确的。
```

让我们为三个结果上具有 $t(x) = x$ 的分布计算这两种投影。

对于 e-投影，我们取 $q = (0.6, 0.3, 0.1)$ 以及均值为 $1.2$ 的分布的线性族。

对于 m-投影，我们取 $p = (0.5, 0.1, 0.4)$ 以及由 $h = (0.2, 0.5, 0.3)$ 和 $t$ 生成的指数族。

```{code-cell} ipython3
outcomes = np.arange(3.0)

# e-projection of q onto {p : E_p[X] = 1.2}
q_e = np.array([0.6, 0.3, 0.1])
θ_e, p_star = match_mean(q_e, outcomes, 1.2)

# members of the linear family: p_star plus multiples of (1, -2, 1),
# small enough to keep every probability positive
s_lo, s_hi = -min(p_star[0], p_star[2]), p_star[1] / 2
print("e-projection: D(p || q) and D(p || p*) + D(p* || q)")
for s in np.linspace(0.9 * s_lo, 0.9 * s_hi, 4):
    p = p_star + s * np.array([1.0, -2.0, 1.0])
    print(f"  {kl(p, q_e):.8f}   {kl(p, p_star) + kl(p_star, q_e):.8f}")

# m-projection of p onto the exponential family generated by h
h = np.array([0.2, 0.5, 0.3])
p_m = np.array([0.5, 0.1, 0.4])
θ_m, q_star = match_mean(h, outcomes, p_m @ outcomes)
print("\nm-projection: D(p || q_θ) and D(p || q*) + D(q* || q_θ)")
for θ in (-1.0, 0.0, 0.5, 2.0):
    q_θ = tilt(h, outcomes, θ)
    print(f"  {kl(p_m, q_θ):.8f}   {kl(p_m, q_star) + kl(q_star, q_θ):.8f}")
```

下图展示了这两种投影。

在左图中，直线是线性族，经过 $q$ 的曲线是由 $q$ 生成的指数族；e-投影 $p^*$ 是它们相交之处。

细曲线是 $D(p \| q)$ 作为 $p$ 的函数的等高线；线性族与经过 $p^*$ 的等高线相切。

在右图中，曲线是由 $h$ 生成的指数族，经过 $p$ 的直线是与 $p$ 共享均值的分布的线性族；m-投影 $q^*$ 是它们相交之处。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The e-projection and the m-projection
    name: fig-ip-projections
---
θ_grid = np.linspace(-6, 6, 400)


def family_curve(base, t, θ_grid):
    """Members of the exponential family generated by base and t."""
    return np.vstack([tilt(base, t, θ) for θ in θ_grid])


def linear_family(c, n=400):
    """Distributions on {0, 1, 2} with mean c."""
    lo, hi = max(0, c - 1), min(1, c / 2)   # range of p(2)
    p2 = np.linspace(lo, hi, n)
    p1 = c - 2 * p2
    return np.column_stack([1 - p1 - p2, p1, p2])


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
draw_simplex(ax)
# a barycentric grid of distributions with full support
k = 300
i, j = np.meshgrid(np.arange(1, k), np.arange(1, k))
inside = i + j < k
grid = np.column_stack([i[inside], j[inside], k - i[inside] - j[inside]]) / k
xy = to_triangle(grid)
ax.tricontour(xy[:, 0], xy[:, 1],
              np.sum(grid * np.log(grid / q_e), axis=1),
              levels=kl(p_star, q_e) * np.array([0.25, 0.5, 1.0, 1.5, 2.0]),
              colors="0.75", linewidths=0.8)
ax.plot(*to_triangle(linear_family(1.2)).T, lw=2,
        label=r"linear family $E[X] = 1.2$")
ax.plot(*to_triangle(family_curve(q_e, outcomes, θ_grid)).T, lw=2,
        label="exponential family through $q$")
for point, name in ((q_e, "$q$"), (p_star, "$p^*$")):
    ax.plot(*to_triangle(point)[0], "ko")
    ax.text(*to_triangle(point)[0] + [0.02, 0.01], name)
ax.set_title("e-projection", pad=18)
ax.legend(loc="upper left", fontsize=9, frameon=False)

ax = axes[1]
draw_simplex(ax)
ax.plot(*to_triangle(family_curve(h, outcomes, θ_grid)).T, lw=2, color="C1",
        label="exponential family through $h$")
ax.plot(*to_triangle(linear_family(p_m @ outcomes)).T, lw=2, color="C0",
        label=rf"linear family $E[X] = {p_m @ outcomes:.1f}$")
for point, name in ((p_m, "$p$"), (q_star, "$q^*$"), (h, "$h$")):
    ax.plot(*to_triangle(point)[0], "ko")
    ax.text(*to_triangle(point)[0] + [0.02, 0.01], name)
ax.set_title("m-projection", pad=18)
ax.legend(loc="upper left", fontsize=9, frameon=False)
plt.tight_layout()
plt.show()
```

## 最大似然是一个 m-投影

### 模型设定正确

设 $x_1, \ldots, x_n$ 是独立同分布的抽样，假设我们通过最大似然拟合一个指数族 $\{ q_\theta \}$。

{doc}`exponential_families` 表明最大似然估计量 $\hat \theta$ 匹配充分统计量的样本均值。

当结果是离散的时，设 $\hat p$ 为样本的经验分布。

那么平均对数似然为 $\sum_x \hat p(x) \log q_\theta(x) = -H(\hat p) - D(\hat p \| q_\theta)$。

因此最大化似然就是最小化 $D(\hat p \| q_\theta)$，因为 $H(\hat p)$ 不依赖于 $\theta$。

根据 {prf:ref}`ip_thm_m_projection`，*最大似然估计是经验分布在模型上的 m-投影*。

用 $\hat p$ 代入 $p$ 的角色应用勾股定理 {eq}`ip_pythagoras_m` 给出 $D(\hat p \| q_\theta) - D(\hat p \| q_{\hat \theta}) = D(q_{\hat\theta} \| q_\theta)$，因此对于每个 $\theta$，

$$
\frac{1}{n} \left[ \ell_n(\hat \theta) - \ell_n(\theta) \right] = D(q_{\hat\theta} \| q_\theta) ,
$$ (ip_lr_kl)

其中 $\ell_n$ 是对数似然。

恒等式 {eq}`ip_lr_kl` 对连续结果也成立：根据 {doc}`exponential_families` 的公式，左边等于 $(\hat\theta - \theta) \cdot \bar t - F(\hat\theta) + F(\theta)$，由于 $\bar t = \nabla F(\hat \theta)$，这就是布雷格曼散度 $B_F(\theta, \hat\theta) = D(q_{\hat\theta} \| q_\theta)$。

对于指数族，只要最大似然估计存在，*似然比统计量 $2[\ell_n(\hat\theta) - \ell_n(\theta_0)]$ 就恰好等于 $2 n D(q_{\hat\theta} \| q_{\theta_0})$*。

根据威尔克斯定理，当 $\theta_0$ 是真实参数时，该统计量近似服从自由度等于参数数目的 $\chi^2$ 分布。

我们用泊松数据来验证这一点，对于泊松数据，$\hat \lambda = \bar x$ 且 $D(q_{\hat\lambda} \| q_{\lambda_0}) = \hat\lambda \log(\hat\lambda/\lambda_0) - \hat\lambda + \lambda_0$。

```{code-cell} ipython3
λ0, n, R = 3.0, 50, 20_000
x = rng.poisson(λ0, size=(R, n))
λ_hat = x.mean(axis=1)

lr = 2 * (poisson.logpmf(x, λ_hat[:, None]).sum(axis=1)
          - poisson.logpmf(x, λ0).sum(axis=1))
two_n_kl = 2 * n * (λ_hat * np.log(λ_hat / λ0) - λ_hat + λ0)
print(f"max |LR - 2n KL| = {np.max(np.abs(lr - two_n_kl)):.2e}")
print(f"P(LR > 3.84) = {np.mean(lr > chi2.ppf(0.95, 1)):.4f}  (χ²_1: 0.05)")
```

讲座 {doc}`likelihood_ratio_process` 比较了两个固定模型；根据 {eq}`ip_lr_kl`，在一个指数族内，最佳拟合模型与任何其他模型之间的 KL 散度支配着它们之间的似然比。

### 模型设定错误

现在假设数据来自模型之外的一个分布 $p_0$。

正如 {doc}`divergence_measures` 所解释的，在正则条件下，最大似然估计量此时收敛于最小化 $D(p_0 \| q_\theta)$ 的**伪真值**参数 $\theta^*$ {cite}`White1982`。

{prf:ref}`ip_thm_m_projection` 确定了它：对于指数族，伪真值参数匹配充分统计量的总体均值。

例如，当我们用正态分布拟合从正态混合中抽取的数据时，伪真值正态分布具有该混合的均值和方差。

让我们通过数值积分来验证这一点以及勾股定理 {eq}`ip_pythagoras_m`。

```{code-cell} ipython3
w, (μa, σa), (μb, σb) = 0.3, (-2.0, 0.7), (1.0, 1.0)


def log_mixture_pdf(x):
    return np.logaddexp(np.log(w) + norm.logpdf(x, μa, σa),
                        np.log(1 - w) + norm.logpdf(x, μb, σb))


def mixture_pdf(x):
    return np.exp(log_mixture_pdf(x))


def kl_mixture_normal(μ, σ):
    def integrand(x):
        log_p = log_mixture_pdf(x)
        return np.exp(log_p) * (log_p - norm.logpdf(x, μ, σ))
    return integrate.quad(integrand, -np.inf, np.inf, limit=200)[0]


def kl_normal(μ1, σ1, μ2, σ2):
    return np.log(σ2 / σ1) + (σ1**2 + (μ1 - μ2)**2) / (2 * σ2**2) - 0.5


μ_star = w * μa + (1 - w) * μb
σ_star = np.sqrt(w * (σa**2 + μa**2) + (1 - w) * (σb**2 + μb**2) - μ_star**2)
print(f"moment-matched normal: μ* = {μ_star:.4f}, σ* = {σ_star:.4f}")

print("\n   μ     σ    D(p0 || N)   D(p0 || N*) + D(N* || N)")
for μ, σ in ((0.0, 1.5), (μ_star, 1.0), (1.0, 2.0), (μ_star, σ_star)):
    lhs = kl_mixture_normal(μ, σ)
    rhs = kl_mixture_normal(μ_star, σ_star) + kl_normal(μ_star, σ_star, μ, σ)
    print(f"{μ:6.2f} {σ:5.2f} {lhs:12.6f} {rhs:16.6f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Moment-matched normal for a bimodal mixture
    name: fig-ip-mixture
---
x_vals = np.linspace(-5, 5, 400)
fig, ax = plt.subplots()
ax.plot(x_vals, mixture_pdf(x_vals), lw=2, label="mixture $p_0$")
ax.plot(x_vals, norm.pdf(x_vals, μ_star, σ_star), lw=2, ls="--",
        label=rf"m-projection $N(\mu^*, \sigma^{{*2}})$")
ax.set_xlabel("$x$")
ax.legend()
plt.show()
```

就 $D(p_0 \| q)$ 的意义而言，矩匹配的正态分布是最佳的正态近似，即使它在两个峰值之间赋予了相当大的概率，而混合分布在那里赋予的概率很小。

讲座 {doc}`mix_model` 研究了一个使用错误模型的贝叶斯学习者如何表现。

{doc}`加朗和陶臣的有效矩方法 <gallant_tauchen>` 基于辅助模型的期望得分在伪真值处消失这一事实构建了一个估计量。

## 独立性与互信息

设 $p(x, y)$ 是两个取有限多个值的随机变量的联合分布，其边缘分布为 $p_X$ 和 $p_Y$。

$X$ 和 $Y$ 相互独立的联合分布集合，

$$
\mathcal{E}_{\text{ind}} = \{ q_1(x) q_2(y) \} ,
$$

是一个指数族：其充分统计量分别是 $x$ 和 $y$ 取值的指示变量，这些统计量的均值是两个边缘分布。

根据 {prf:ref}`ip_thm_m_projection`，$p$ 在 $\mathcal{E}_{\text{ind}}$ 上的 m-投影是边缘分布的乘积，$p_X p_Y$。

最小化的散度是**互信息**

$$
I(X; Y) = D(p \| p_X p_Y) = \min_{q_1, q_2} D(p \| q_1 q_2) ,
$$

即 $p$ 到最近的使 $X$ 和 $Y$ 独立的联合分布的 KL 散度。

对于任意边缘分布 $q_1$ 和 $q_2$，勾股定理 {eq}`ip_pythagoras_m` 给出

$$
D(p \| q_1 q_2) = D(p \| p_X p_Y) + D(p_X p_Y \| q_1 q_2) ,
$$

由于乘积的对数是和，最后一项分解为 $D(p_X \| q_1) + D(p_Y \| q_2)$。

所以

$$
D(p \| q_1 q_2) = I(X; Y) + D(p_X \| q_1) + D(p_Y \| q_2) ,
$$

这将 $p$ 到独立模型的散度分解为由依赖性引起的部分和由边缘分布引起的部分。

```{code-cell} ipython3
p_xy = np.array([[0.20, 0.05, 0.05],
                 [0.05, 0.25, 0.05],
                 [0.02, 0.08, 0.25]])
p_x, p_y = p_xy.sum(axis=1), p_xy.sum(axis=0)
mutual_info = kl(p_xy, np.outer(p_x, p_y))
print(f"I(X; Y) = {mutual_info:.6f}")

for _ in range(3):
    q1, q2 = rng.dirichlet(np.ones(3), size=2)
    lhs = kl(p_xy, np.outer(q1, q2))
    rhs = mutual_info + kl(p_x, q1) + kl(p_y, q2)
    print(f"D(p || q1 q2) = {lhs:.6f},  decomposition = {rhs:.6f}")
```

互信息是 {cite:t}`Sims2003` 创立的理性迟钝理论中信息流的度量。

{ref}`ip_ex_interactions` 将这一分解扩展到三个随机变量。

## 最大熵与大偏差

均值为 $4.5$ 的骰子的最大熵分布 $p^*$ 是均匀分布 $q$ 在线性族 $\{ p : E_p[X] = 4.5 \}$ 上的 e-投影。

我们重新计算它，因为下面需要用到它。

```{code-cell} ipython3
faces = np.arange(1.0, 7.0)
q_die = np.full(6, 1 / 6)
θ_die, p_die = match_mean(q_die, faces, 4.5)
print("e-projection of the uniform distribution:", np.round(p_die, 4))
print(f"D(p* || q) = {kl(p_die, q_die):.4f}")
```

大偏差理论解释了为什么 e-投影能回答杰恩斯的问题。

**萨诺夫定理**涉及来自 $q$ 的 $n$ 个独立同分布抽样的经验分布 $\hat p_n$。

它说，对于满足一个温和正则条件的分布集合 $\mathcal{S}$（例如，$\mathcal{S}$ 是其内部的闭包），

$$
\lim_{n \to \infty} -\frac{1}{n} \log P\left( \hat p_n \in \mathcal{S} \right) = \min_{p \in \mathcal{S}} D(p \| q) = D(p_{\mathcal{S}} \| q) ,
$$

其中 $p_{\mathcal{S}}$ 是 $q$ 在 $\mathcal{S}$ 上的 e-投影。

经验分布落在 $\mathcal{S}$ 中的概率以速率 $D(p_{\mathcal{S}} \| q)$ 呈指数衰减。

当我们掷一个公平骰子 $n$ 次时，平均值至少为 $4.5$ 的事件就是 $\hat p_n$ 位于半空间 $\mathcal{S} = \{ p : E_p[X] \geq 4.5 \}$ 中的事件。

因为 $q$ 的均值是 $3.5$，$q$ 在这个半空间上的 e-投影位于其边界上，即线性族 $\{ p : E_p[X] = 4.5 \}$。

沿着从均值更大的任何 $p$ 指向 $q$ 的直线，$D(\cdot \| q)$ 下降，因为它是凸的且在 $q$ 处为零；这条直线经过一个均值为 $4.5$ 的点。

因此 e-投影是 $p^*$，且

$$
\lim_{n \to \infty} -\frac{1}{n} \log P\left( \frac{1}{n} \sum_{i=1}^n X_i \geq 4.5 \right) = D(p^* \| q) .
$$

同样的计算说明了这种罕见事件最可能发生的方式：在 $n$ 次公平掷骰子可能产生平均值 $4.5$ 的所有方式中，压倒性最有可能的是各点数的频率近似于 $p^*$。

**条件极限定理**使这一点精确化。

假设我们掷一个公平骰子 $n$ 次，只知道点数的平均值恰好为 $4.5$，这要求 $n$ 为偶数。

给定这个信息，第一次投掷的点数的分布随着 $n \to \infty$ 收敛到 e-投影 $p^*$ {cite}`VanCampenhoutCover1981,Csiszar1984`。

{cite:t}`CoverThomas2006` 的第 11 章证明了这两个定理。

我们可以精确地验证它们。

$n$ 次投掷的和 $S_n$ 的分布是一个 $n$ 重卷积，且

$$
P(X_1 = k \mid S_n = s) = \frac{P(X_1 = k) \, P(S_{n - 1} = s - k)}{P(S_n = s)} ,
$$

这是 {cite:t}`VanCampenhoutCover1981` 证明所依据的公式。

```{code-cell} ipython3
def sum_distribution(pmf, n):
    """Distribution of the sum of n IID draws from pmf on {1, ..., m}.

    Entry j of the result is the probability that the sum equals j + n.
    """
    dist = np.array([1.0])
    for _ in range(n):
        dist = np.convolve(dist, pmf)
    return dist


print("    n   P(X_1 = k | average = 4.5), k = 1, ..., 6     "
      "-log P(average ≥ 4.5)/n")
n_vals = [10, 20, 40, 80, 160, 320, 640]
rates = []
for n in n_vals:
    s = int(4.5 * n)
    dist_n1 = sum_distribution(q_die, n - 1)
    dist_n = np.convolve(dist_n1, q_die)
    cond = np.array([q_die[k - 1] * dist_n1[s - k - (n - 1)]
                     for k in range(1, 7)]) / dist_n[s - n]
    rates.append(-np.log(dist_n[s - n:].sum()) / n)
    print(f"{n:5d}   {np.round(cond, 4)}   {rates[-1]:.4f}")
print(f"  p* = {np.round(p_die, 4)}   D(p* || q) = {kl(p_die, q_die):.4f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Decay rate converges to $D(p^* \| q)$
    name: fig-ip-sanov
---
fig, ax = plt.subplots()
ax.plot(n_vals, rates, "o-", label=r"$-\frac{1}{n}\log P(\bar X_n \geq 4.5)$")
ax.axhline(kl(p_die, q_die), color="k", ls="--", lw=1,
           label=r"$D(p^* \| q)$")
ax.set_xscale("log")
ax.set_xlabel("number of rolls $n$")
ax.legend()
plt.show()
```

条件分布很快收敛到 $p^*$；衰减速率收敛得更慢，因为概率中还含有关于 $n$ 的多项式因子。

{doc}`likelihood_ratio_process` 和 {doc}`关于意见融合的讲座 <merging_of_opinions>` 中的大偏差结果基于同样的数学原理。

## 稳健控制中的最坏情况模型

设 $\pi$ 是状态 $x$ 上的基准概率分布，设 $u(x)$ 是决策者的效用。

一个担心 $\pi$ 可能设定错误的决策者可以通过对备选分布 $q$ 的最坏期望效用来评估前景，并以 KL 散度惩罚偏离 $\pi$ 的备选方案：

$$
\min_q \left\{ E_q[u(X)] + \lambda \, D(q \| \pi) \right\} ,
$$ (ip_multiplier)

其中 $\lambda > 0$ 衡量决策者对 $\pi$ 的信心。

这些是 {cite:t}`HansenSargent2001` 的乘子偏好。

为求解 {eq}`ip_multiplier`，定义指数倾斜

$$
q^*_\lambda(x) = \frac{\pi(x) \exp(-u(x)/\lambda)}{E_\pi \exp(-u(X)/\lambda)} .
$$

因为 $\log q^*_\lambda(x) = \log \pi(x) - u(x)/\lambda - \log E_\pi \exp(-u(X)/\lambda)$，我们有

$$
\lambda \, D(q \| q^*_\lambda) = \lambda \, D(q \| \pi) + E_q[u(X)] + \lambda \log E_\pi \exp\left( -u(X)/\lambda \right) ,
$$

即对每个 $q$，

$$
E_q[u(X)] + \lambda \, D(q \| \pi) = \lambda \, D(q \| q^*_\lambda) - \lambda \log E_\pi \exp\left( -u(X)/\lambda \right) ,
$$

因此最坏情况分布是 $q^*_\lambda$，且 {eq}`ip_multiplier` 的值是 $-\lambda \log E_\pi \exp(-u(X)/\lambda)$。

每个最坏情况模型 $q^*_\lambda$ 也是 $\pi$ 在线性族 $\{ q : E_q[u(X)] = E_{q^*_\lambda}[u(X)] \}$ 上的 e-投影：在所有能带来那个期望效用的分布中，它离 $\pi$ 最近。

值 $-\lambda \log E_\pi \exp(-u(X)/\lambda)$ 是 {cite:t}`Whittle_1990` 和 {cite:t}`HansenSargent2008` 的风险敏感价值函数。

它也是 {doc}`exponential_families` 的 logit 公式 {eq}`ef_logit_base` 的一个实例，其中 $v = -u$，$\tau = \lambda$。

字母互换了位置：这里的基准 $\pi$ 是那里的基准分布 $q$，这里选择的分布 $q$ 是那里的向量 $\pi$。

最小化 $E_q[u] + \lambda D(q \| \pi)$ 就是最大化 $E_q[-u] - \lambda D(q \| \pi)$，因此 {eq}`ip_multiplier` 的值是 {eq}`ef_logit_base` 左边的负值。

最坏情况模型形成一个经过 $\pi$ 的单参数指数族，其充分统计量为 $u$，自然参数为 $-1/\lambda$。

随着 $\lambda$ 下降，决策者对 $\pi$ 的信任降低，最坏情况沿此族移动，将更多概率赋予低效用状态。

下面的例子具有对消费的对数效用 $u(x) = \log x$，其中在 $\pi$ 下消费的对数服从 $N(0, 0.2^2)$ 分布。

记 $z = \log x$；用 $\exp(-z/\lambda)$ 对 $z$ 的 $N(0, 0.2^2)$ 密度进行倾斜，可得 $N(-0.2^2/\lambda, 0.2^2)$ 密度，因此每个最坏情况仍是对数正态分布。

那么 $E_{q^*_\lambda}[u] = -0.2^2/\lambda$ 且 $D(q^*_\lambda \| \pi) = 0.2^2/(2 \lambda^2)$。

我们在一个足够宽以包含我们所考虑最坏情况的网格上离散化对数消费，并将结果与这些公式进行比较。

```{code-cell} ipython3
σ_c = 0.2
log_c = np.linspace(-2.8, 0.8, 91)
π_base = norm.pdf(log_c, 0, σ_c)
π_base /= π_base.sum()
u_c = log_c                                # u(c) = log c


def worst_case(λ):
    return tilt(π_base, u_c, -1 / λ)


def multiplier_value(λ):
    return -λ * logsumexp(-u_c / λ, b=π_base)


λ_shown = (0.2, 0.05, 0.02)
print("    λ   E_q[u]  exact   D(q || π)   exact   E_q[u] + λ D   "
      "-λ log E exp(-u/λ)")
for λ in λ_shown:
    q = worst_case(λ)
    print(f"{λ:5.2f} {q @ u_c:8.4f} {-σ_c**2 / λ:6.2f} {kl(q, π_base):10.4f} "
          f"{σ_c**2 / (2 * λ**2):7.2f} {q @ u_c + λ * kl(q, π_base):12.4f} "
          f"{multiplier_value(λ):16.4f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Worst-case distributions under multiplier preferences
    name: fig-ip-worst-case
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(log_c, π_base, "k", lw=2, label=r"baseline $\pi$")
for λ in λ_shown:
    ax.plot(log_c, worst_case(λ), label=rf"worst case, $\lambda$ = {λ}")
ax.set_xlabel("log consumption")
ax.set_ylabel("probability")
ax.legend()

ax = axes[1]
λ_vals = np.geomspace(0.02, 2, 100)
rel_entropies = [kl(worst_case(λ), π_base) for λ in λ_vals]
utilities = [worst_case(λ) @ u_c for λ in λ_vals]
ax.plot(rel_entropies, utilities, lw=3, alpha=0.5, label="on the grid")
D_vals = np.linspace(0, max(rel_entropies), 200)
ax.plot(D_vals, -σ_c * np.sqrt(2 * D_vals), "k--", lw=1,
        label=r"exact, $-0.2\sqrt{2D}$")
ax.set_xlabel(r"$D(q^*_\lambda \| \pi)$")
ax.set_ylabel(r"$E_{q^*_\lambda}[u(X)]$")
ax.set_title("worst-case expected utility and relative entropy")
ax.legend()
plt.tight_layout()
plt.show()
```

右图描绘了最坏情况期望效用与最坏情况相对熵之间的关系；从上述两个公式中消去 $\lambda$，得到 $E_{q^*_\lambda}[u] = -0.2 \sqrt{2 D(q^*_\lambda \| \pi)}$。

这条曲线上的每个点也求解了最小化期望效用、同时满足约束 $D(q \| \pi) \leq \kappa$ 的问题，其中 $\kappa$ 等于该点的横坐标；{ref}`ip_ex_constraint` 要求你验证这一点。

这些是讲座 {doc}`advanced:five_preferences` 中研究的约束偏好，该讲座用 $\theta$ 表示我们的 $\lambda$，用 $\eta$ 表示我们的 $\kappa$。

{doc}`另一个高级讲座 <advanced:risk_aversion_or_mistaken_beliefs>` 使用这种最坏情况模型来探讨资产价格是反映风险厌恶还是错误信念。

指数倾斜也出现在 {doc}`imp_sample` 中，在那里它提供了高效的重要性采样器。

## 矩阵平衡：RAS 和辛克霍恩

### 更新投入产出表

投入产出分析师经常需要更新一张表。

{doc}`intro:input_output` 描述了这样的表。

基准年的产业间商品流动表 $Q$ 是可得的。

对于后续年份，分析师只知道行总计 $r$（每个产业对其他产业的销售额）和列总计 $c$（每个产业从其他产业的采购额）。

在本节中，$r$ 和 $c$ 表示总计向量。

**矩阵平衡**找到一张具有给定总计且与基准表保持接近的表。

新表的一个自然估计是具有这些总计、且使 $D(P \| Q)$ 最小化的矩阵 $P^*$。

因为 $P$ 和 $Q$ 不必有相同的总和，我们对非负数组使用**广义 KL 散度**，

$$
D(P \| Q) = \sum_{i, j} \left[ P_{ij} \log \frac{P_{ij}}{Q_{ij}} - P_{ij} + Q_{ij} \right] ,
$$

它是非负的，只有在 $P = Q$ 时才为零，并且三点恒等式 {eq}`ip_three_point` 仍然成立。

具有行总计 $r$ 的矩阵集合是一个线性族，具有列总计 $c$ 的矩阵集合也是如此。

矩阵 $P^*$ 是 $Q$ 在这两个族的交集上的 e-投影。

证明 {prf:ref}`ip_thm_e_projection` 的论证，用行和列的指示变量作为统计量 $t$，在这里同样适用。

它表明，如果**双比例**形式

$$
P_{ij} = a_i \, Q_{ij} \, b_j ,
$$

的矩阵（其中 $a$ 和 $b$ 是正向量）具有所需的总计，那么它就是 $P^*$，因为 {eq}`ip_three_point` 中的交叉项变为 $\sum_i \log a_i \, (r_i - r_i) + \sum_j \log b_j \, (c_j - c_j) = 0$。

当 $Q$ 的每个元素都为正、$r$ 和 $c$ 都为正，且 $\sum_i r_i = \sum_j c_j$ 时，这样的矩阵存在 {cite}`Bacharach1970`。

**RAS 方法**通过交替重新缩放当前表的行以匹配 $r$ 以及其列以匹配 $c$ 来求出 $a$ 和 $b$；每次重新缩放都是当前表在两个线性族之一上的 e-投影。

{cite:t}`DemingStephan1940` 提出了这个算法，统计学家称之为**迭代比例拟合**，用于将普查表调整为已知的边际总计。

{cite:t}`StoneBrown1962` 用它来更新投入产出表。

{cite:t}`IrelandKullback1968` 确定了它的极限是使与基准表的 KL 散度最小化的表，但他们对收敛性的证明并不完整。

{cite:t}`Bacharach1970` 研究了它在投入产出分析中的性质。

{cite:t}`Csiszar1975` 给出了它收敛性的完整证明。

勾股定理解释了为什么这个算法会收敛。

设 $P$ 是具有这两种总计且在 $Q$ 为零的地方也为零的任意矩阵，使得 $D(P \| Q) < \infty$。

因为每次重新缩放都是在包含 $P$ 的线性族上的 e-投影，{eq}`ip_pythagoras_e` 给出

$$
D(P \| P^{(k)}) = D(P \| P^{(k + 1)}) + D(P^{(k + 1)} \| P^{(k)}) ,
$$

其中 $P^{(k)}$ 是经过 $k$ 次重新缩放后的表，一次行重新缩放和一次列重新缩放算作两步。

特别地，$P^*$ 与迭代值的散度在每一步都在下降。

求和得到 $D(P \| Q) = D(P \| P^{(k)}) + \sum_{j < k} D(P^{(j + 1)} \| P^{(j)})$，其中的和不依赖于 $P$。

取 $P = P^*$ 表明 $\sum_k D(P^{(k + 1)} \| P^{(k)}) \leq D(P^* \| Q)$，因此相继的迭代值变得任意接近，且迭代值的每个极限点 $\bar P$ 都具有这两种总计。

沿收敛到 $\bar P$ 的一个子序列令 $k \to \infty$，取 $P = \bar P$，得到 $D(\bar P \| Q) = \sum_k D(P^{(k + 1)} \| P^{(k)})$，根据同样的恒等式，对每个可容许的 $P$ 这都至多等于 $D(P \| Q)$。

因此每个极限点都是最小化点 $P^*$，迭代值收敛于它 {cite}`Csiszar1975`。

这个论证不需要双比例形式；当 $Q$ 有零元素时，双比例解可能不存在。

例如，设 $Q$ 的行为 $(1, 1)$ 和 $(0, 1)$，并且所有总计都等于 $1$。

唯一可容许的表是单位矩阵。

它不是 $a_i Q_{ij} b_j$（$a$、$b$ 为正）的形式，但 RAS 仍然收敛于它。

```{code-cell} ipython3
def gen_kl(P, Q):
    """Generalized KL divergence of nonnegative arrays."""
    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / Q[mask])) - P.sum() + Q.sum()


def ras(Q, r, c, tol=1e-12, max_iter=10_000):
    """Rescale rows and columns of Q to match totals r and c."""
    if not np.isclose(r.sum(), c.sum()):
        raise ValueError("row and column totals must have the same sum")
    P = Q.copy()
    history = [P.copy()]
    for _ in range(max_iter):
        P = P * (r / P.sum(axis=1))[:, None]
        history.append(P.copy())
        P = P * (c / P.sum(axis=0))[None, :]
        history.append(P.copy())
        if np.max(np.abs(P.sum(axis=1) - r)) < tol * r.sum():
            break
    else:
        warnings.warn("RAS did not converge")
    return P, history


Q_io = np.array([[20.0, 10.0, 5.0],
                 [8.0, 30.0, 12.0],
                 [6.0, 9.0, 25.0]])
r_new = np.array([40.0, 55.0, 45.0])
c_new = np.array([38.0, 52.0, 50.0])

P_io, history = ras(Q_io, r_new, c_new)
print("base table:\n", Q_io)
print("updated table:\n", np.round(P_io, 3))
print("row totals:", np.round(P_io.sum(axis=1), 6),
      " column totals:", np.round(P_io.sum(axis=0), 6))
```

比率 $P^*_{ij} / Q_{ij} = a_i b_j$ 分解为一个行效应乘以一个列效应。

RAS 这个名字描述了这一点：更新后的表是 $R A S$，其中 $A$ 是基准表，$R$ 和 $S$ 是行乘数和列乘数的对角矩阵，即我们的 $a$ 和 $b$。

如果比率 $R_{ij} = P^*_{ij}/Q_{ij}$ 分解为 $a_i b_j$，那么 $R_{ij} = R_{i1} R_{1j} / R_{11}$，因此用 $R_{i1} R_{1j} / R_{11}$ 去除 $R_{ij}$ 会得到一个全为 1 的矩阵。

```{code-cell} ipython3
ratio = P_io / Q_io
print("P*/Q divided by its first column and first row:")
print(np.round(ratio / np.outer(ratio[:, 0], ratio[0, :] / ratio[0, 0]), 10))
```

$P^*$ 与迭代值的散度单调下降，直到达到舍入误差的水平；该图省略了低于这个水平的值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Convergence of the RAS algorithm
    name: fig-ip-ras
---
divergences = np.array([gen_kl(P_io, P) for P in history])
steps = np.arange(len(divergences))
keep = divergences > 1e-12
fig, ax = plt.subplots()
ax.semilogy(steps[keep], divergences[keep], "o-")
ax.set_xlabel("rescaling step $k$")
ax.set_ylabel(r"$D(P^* \| P^{(k)})$")
plt.show()
```

我们对具有相同总计的其他表验证勾股定理 {eq}`ip_pythagoras_e`。

```{code-cell} ipython3
# add multiples of a pattern with zero row and column sums
pattern = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
for s in (-3.0, 2.0, 5.0):
    P_other = P_io + s * pattern
    print(f"D(P || Q) = {gen_kl(P_other, Q_io):.6f},  "
          f"D(P || P*) + D(P* || Q) = "
          f"{gen_kl(P_other, P_io) + gen_kl(P_io, Q_io):.6f}")
```

### 熵正则化最优传输

{doc}`opt_transport` 研究了选择一个具有行总计 $p$ 和列总计 $q$（该讲座的记号）的传输计划 $P$ 以最小化总成本 $\sum_{i, j} C_{ij} P_{ij}$ 的问题。

加上 $\varepsilon$ 倍的计划的负熵得到**熵正则化**问题

$$
\min_{P} \sum_{i, j} C_{ij} P_{ij} + \varepsilon \sum_{i, j} P_{ij} \left( \log P_{ij} - 1 \right) .
$$

令 $K_{ij} = \exp(-C_{ij}/\varepsilon)$，目标函数等于 $\varepsilon \, D(P \| K) - \varepsilon \sum_{ij} K_{ij}$。

因此正则化问题就是求矩阵 $K$ 在具有所需总计的计划集合上的 e-投影。

该计划具有双比例形式 $P_{ij} = a_i K_{ij} b_j = a_i \exp(-C_{ij}/\varepsilon) b_j$，这与求解 {doc}`exponential_families` 中 {eq}`ef_logit_base` 的指数倾斜相同。

RAS 的重新缩放算法可以求解它；在这个场景下，该算法被称为**辛克霍恩算法** {cite}`Sinkhorn1964`。

{cite:t}`Cuturi2013` 表明辛克霍恩算法使大规模最优传输问题变得可处理。

{cite:t}`Galichon_2016` 和 {cite:t}`GalichonSalanie2022` 解释了 {cite:t}`ChooSiow2006` 的匹配模型（其中代理人对伴侣具有 logit 偏好）如何导致这种形式的问题。

当每个代理人都被匹配时，Choo–Siow 模型恰好是一个具有固定总计的熵正则化传输问题。

当 $\varepsilon$ 很小时，$K$ 的元素非常小，因此我们在对数空间中进行重新缩放。

我们使用 {doc}`opt_transport` 的例子，用 `linprog` 求解线性规划，并将其解与正则化问题在 $\varepsilon$ 递减时的解进行比较。

```{code-cell} ipython3
def sinkhorn(C, p, q, ε, tol=1e-10, max_iter=100_000):
    """Entropy-regularized optimal transport by rescaling in logs."""
    log_K = -C / ε
    log_a, log_b = np.zeros(len(p)), np.zeros(len(q))
    for _ in range(max_iter):
        log_a = np.log(p) - logsumexp(log_K + log_b[None, :], axis=1)
        log_b = np.log(q) - logsumexp(log_K + log_a[:, None], axis=0)
        P = np.exp(log_a[:, None] + log_K + log_b[None, :])
        if np.max(np.abs(P.sum(axis=1) - p)) < tol * p.sum():
            break
    else:
        warnings.warn("Sinkhorn's algorithm did not converge")
    return P


p_ot = np.array([50.0, 100.0, 150.0])
q_ot = np.array([25.0, 115.0, 60.0, 30.0, 70.0])
C_ot = np.array([[10.0, 15.0, 20.0, 20.0, 40.0],
                 [20.0, 40.0, 15.0, 30.0, 30.0],
                 [30.0, 35.0, 40.0, 55.0, 25.0]])
m, n = C_ot.shape

# the linear program of the optimal transport lecture
A_eq = np.vstack([np.kron(np.ones((1, n)), np.identity(m)),
                  np.kron(np.identity(n), np.ones((1, m)))])
res_lp = linprog(C_ot.reshape(-1, order="F"), A_eq=A_eq[:-1],
                 b_eq=np.hstack([p_ot, q_ot])[:-1])
P_lp = res_lp.x.reshape((m, n), order="F")

print(f"linear program: cost = {res_lp.fun:.3f}")
for ε in (10.0, 3.0, 1.0, 0.1):
    P_ε = sinkhorn(C_ot, p_ot, q_ot, ε)
    print(f"ε = {ε:5.1f}: cost = {np.sum(C_ot * P_ε):9.3f}")
```

随着 $\varepsilon \to 0$，正则化解的成本收敛于最小成本。

正则化计划不一定收敛于 `linprog` 给出的计划。

```{code-cell} ipython3
P_small = sinkhorn(C_ot, p_ot, q_ot, 0.05)
print("linprog plan:\n", np.round(P_lp, 2))
print("limit of regularized plans:\n", np.round(P_small, 2))


def plan_entropy(P):
    """Entropy of the plan, normalized to be a probability distribution."""
    P = P[P > 1e-12] / P.sum()
    return -np.sum(P * np.log(P))


print(f"costs: {np.sum(C_ot * P_lp):.3f} and {np.sum(C_ot * P_small):.3f}")
print(f"entropies: {plan_entropy(P_lp):.4f} and {plan_entropy(P_small):.4f}")
```

这个问题有许多最优计划。

`linprog` 通常返回最优计划集合的一个顶点。

正则化计划收敛于熵最大的最优计划 {cite}`CominettiSanMartin1994`。

那个极限本身就是一个 e-投影。

根据互补松弛条件，最优计划恰好是具有所需总计、且只使用某个最优计划所使用的格点的计划。

熵最大的最优计划是对一个在这些格点上取 1、其余取 0 的矩阵应用 RAS 方法的结果。

```{code-cell} ipython3
support = (P_small > 1e-6 * p_ot.sum()).astype(float)
P_maxent, _ = ras(support, p_ot, q_ot)
print("RAS applied to the optimal support:\n", np.round(P_maxent, 2))
print(f"cost = {np.sum(C_ot * P_maxent):.3f}, "
      f"entropy = {plan_entropy(P_maxent):.4f}")
```

## 切尔诺夫信息

{doc}`divergence_measures` 将密度 $f$ 和 $g$ 的切尔诺夫熵（信息论学家称之为切尔诺夫信息）定义为

$$
C(f, g) = - \log \min_{\phi \in (0,1)} \rho(\phi), \qquad
\rho(\phi) = \int f^{\phi}(x) g^{1 - \phi}(x) \, dx ,
$$

{doc}`likelihood_ratio_process` 表明它支配着随着观测值的累积，错误模型选择的概率下降的速率。

我们假设 $f$ 和 $g$ 不同且具有相同的支撑，使得 $\log \rho$（在 $\phi = 0$ 和 $\phi = 1$ 处为零且严格凸）在内部点 $\phi^*$ 处取得最小值。

信息投影赋予切尔诺夫信息一个几何解释 {cite}`Nielsen2013`。

归一化的几何混合

$$
p_\phi(x) = \frac{f^{\phi}(x) g^{1 - \phi}(x)}{\rho(\phi)}
$$

形成了 $g$（在 $\phi = 0$ 处）与 $f$（在 $\phi = 1$ 处）之间的指数路径。

这条路径是一个单参数指数族，其充分统计量为 $\log(f/g)$，自然参数为 $\phi$，对数配分函数为 $\log \rho(\phi)$。

在 $\log \rho$ 的最小化点 $\phi^*$ 处，对数配分函数的导数消失，因此 $E_{p_{\phi^*}}[\log(f(X)/g(X))] = 0$。

由于 $D(p_\phi \| f) = (\phi - 1) E_{p_\phi}[\log(f/g)] - \log \rho(\phi)$ 且 $D(p_\phi \| g) = \phi E_{p_\phi}[\log(f/g)] - \log \rho(\phi)$，由此可得

$$
D(p_{\phi^*} \| f) = D(p_{\phi^*} \| g) = C(f, g) .
$$

*切尔诺夫信息是 $D(p_{\phi^*} \| f)$ 和 $D(p_{\phi^*} \| g)$ 的公共值，其中 $p_{\phi^*}$ 是指数路径上到 $f$ 和 $g$ 距离相等的唯一点。*

因为 $D(p \| f) - D(p \| g) = -E_p[\log(f(X)/g(X))]$，当且仅当对数似然比在某分布下的均值为零时，该分布到 $f$ 和 $g$ 的距离相等。

线性族 $\mathcal{L}_0 = \{ p : E_p[\log(f/g)] = 0 \}$ 是垂直平分线的类似物。

点 $p_{\phi^*}$ 是 $f$ 和 $g$ 都在 $\mathcal{L}_0$ 上的 e-投影：$f$ 的 e-投影是 $f$ 被 $\exp(\theta \log(f/g))$ 倾斜的结果，它位于指数路径上，而 $p_{\phi^*}$ 是该路径在 $\mathcal{L}_0$ 中的唯一点。

这解释了为什么切尔诺夫信息支配着错误率。

一个选择似然较高模型的检验，当 $\log(f/g)$ 的样本均值符号错误时就会出错。

如果数据来自 $f$，那就是经验分布位于半空间 $\{ p : E_p[\log(f/g)] \leq 0 \}$ 中的事件。

由于 $E_f[\log(f/g)] = D(f \| g) > 0$，$f$ 在这个半空间上的 e-投影位于其边界 $\mathcal{L}_0$ 上，正如骰子例子中那样，该事件的概率以速率 $D(p_{\phi^*} \| f) = C(f, g)$ 衰减。

对于有限集上的分布，这就是萨诺夫定理；对于密度，将克拉默定理应用于 $\log(f(X)/g(X))$ 的样本均值会给出相同的速率，因为它在零点处的速率函数是 $-\min_\phi \log \rho(\phi)$。

如果数据来自 $g$，错误事件是 $\{ p : E_p[\log(f/g)] \geq 0 \}$，速率是 $D(p_{\phi^*} \| g) = C(f, g)$。

我们使用 {doc}`divergence_measures` 中的两个贝塔分布，$f = $ Beta(1, 1) 以及 $g = $ Beta(3, 1.2)。

因为贝塔族是一个指数族，两个贝塔分布之间的指数路径仍留在贝塔族内：$p_\phi$ 是参数为 $f$ 的参数乘以 $\phi$ 加上 $g$ 的参数乘以 $1 - \phi$ 的贝塔分布。

我们需要的所有量都有闭式解。

```{code-cell} ipython3
fa, fb, ga, gb = 1.0, 1.0, 3.0, 1.2


def path_params(ϕ):
    return ϕ * fa + (1 - ϕ) * ga, ϕ * fb + (1 - ϕ) * gb


def log_ρ(ϕ):
    return (betaln(*path_params(ϕ)) - ϕ * betaln(fa, fb)
            - (1 - ϕ) * betaln(ga, gb))


def kl_beta(a1, b1, a2, b2):
    """KL divergence of Beta(a1, b1) from Beta(a2, b2)."""
    return (betaln(a2, b2) - betaln(a1, b1) + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1) + (a2 - a1 + b2 - b1) * digamma(a1 + b1))


res_ϕ = minimize_scalar(log_ρ, bounds=(1e-8, 1 - 1e-8), method="bounded",
                        options={"xatol": 1e-12})
ϕ_star, C_fg = res_ϕ.x, -res_ϕ.fun
print(f"ϕ* = {ϕ_star:.6f},  Chernoff information C(f, g) = {C_fg:.6f}")
print(f"D(p_ϕ* || f) = {kl_beta(*path_params(ϕ_star), fa, fb):.6f}")
print(f"D(p_ϕ* || g) = {kl_beta(*path_params(ϕ_star), ga, gb):.6f}")
print("p_ϕ* = Beta({:.4f}, {:.4f})".format(*path_params(ϕ_star)))
```

切尔诺夫信息与 {doc}`divergence_measures` 中通过数值积分计算出的值一致。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Point equally far from both distributions
    name: fig-ip-chernoff
---
ϕ_vals = np.linspace(0.001, 0.999, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(ϕ_vals, [kl_beta(*path_params(ϕ), fa, fb) for ϕ in ϕ_vals],
        label=r"$D(p_\phi \| f)$")
ax.plot(ϕ_vals, [kl_beta(*path_params(ϕ), ga, gb) for ϕ in ϕ_vals],
        label=r"$D(p_\phi \| g)$")
ax.plot(ϕ_vals, -log_ρ(ϕ_vals), "k--", lw=1, label=r"$-\log\rho(\phi)$")
ax.plot([ϕ_star], [C_fg], "ko")
ax.axvline(ϕ_star, color="0.7", lw=0.8)
ax.set_xlabel(r"$\phi$")
ax.set_ylim(0, 0.6)
ax.legend()

ax = axes[1]
z = np.linspace(0.001, 0.999, 300)
for (a_, b_), label in (((fa, fb), "$f$"), ((ga, gb), "$g$"),
                        (path_params(ϕ_star), r"$p_{\phi^*}$")):
    ax.plot(z, np.exp((a_ - 1) * np.log(z) + (b_ - 1) * np.log(1 - z)
                      - betaln(a_, b_)), lw=2, label=label)
ax.set_xlabel("$x$")
ax.legend()
plt.tight_layout()
plt.show()
```

在左图中，两个散度在 $\phi^*$ 处交叉，它们的公共值等于 $-\log \rho(\phi)$ 的最大值，即切尔诺夫信息。

在右图中，$p_{\phi^*}$ 位于 $f$ 和 $g$ 之间。

## 结束语

KL 散度不是一种距离。

它拥有一种带有两种直线的几何结构——混合与归一化几何混合——以及相应的两种投影。

由矩条件定义的集合上的 e-投影是一种指数倾斜，指数族上的 m-投影匹配矩；每一个都是唯一的，并满足一个勾股定理。

本讲座将 m-投影应用于最大似然估计和互信息，将 e-投影应用于最大熵和大偏差、稳健控制、投入产出表的更新、最优传输以及切尔诺夫信息。

稳健控制的最坏情况模型、RAS 的双比例表以及正则化传输计划，都是达到 {doc}`exponential_families` 的 logit 公式 {eq}`ef_logit_base` 的那种指数倾斜。

其他应用包括 EM 算法（{cite:t}`CsiszarTusnady1984` 将其解释为交替投影）以及 {cite:t}`Amari2001` 对多个随机变量之间相互作用的层级分解，{ref}`ip_ex_interactions` 将讨论这个问题。

## 练习

```{exercise}
:label: ip_ex_reweight

调查统计学家和计量经济学家经常对样本重新加权，使其与从其他来源已知的一个总体矩相匹配。

设 $x_1, \ldots, x_n$ 是一个样本，其均值与已知的总体均值 $\mu$ 不同。

两种选择求和为一且满足 $\sum_i w_i x_i = \mu$ 的权重 $w_1, \ldots, w_n$ 的方法是

* **指数倾斜**：选择 $w$ 以最小化 $D(w \| q)$，其中 $q$ 对每个观测赋予权重 $1/n$；以及
* **经验似然**：选择 $w$ 以最小化 $D(q \| w)$，等价于最大化 $\sum_i \log w_i$。

1. 证明指数倾斜是一个 e-投影，且其权重具有 $w_i \propto \exp(\theta x_i)$ 的形式。
2. 证明经验似然权重具有 $w_i = 1 / \left( n (1 + \gamma (x_i - \mu)) \right)$ 的形式，其中 $\gamma$ 是一个标量。
3. 抽取 $n = 200$ 对 $(x_i, y_i)$，其中 $x_i$ 是对数正态分布，$y_i = 1 + 0.5 x_i + e_i$，其中 $e_i$ 是正态噪声。计算这两组权重，使得 $x$ 的加权均值等于其样本均值的 1.2 倍，比较这些权重，并比较这两种重新加权的 $y$ 均值估计。

{cite:t}`KitamuraStutzer1997` 和 {cite:t}`Owen1988` 开发了基于这两种权重的估计量。
```

```{solution-start} ip_ex_reweight
:class: dropdown
```

这是一个解答。

对于第 1 部分，满足 $\sum_i w_i x_i = \mu$ 的权重向量集合是一个线性族，因此在其上最小化 $D(w \| q)$ 是一个 e-投影；根据 {prf:ref}`ip_thm_e_projection`，$w_i \propto q_i \exp(\theta x_i) \propto \exp(\theta x_i)$。

而最小化 $D(q \| w)$ 则是 $q$ 在该线性族上的一个 m-投影；我们的定理没有涵盖这种情况。

对于第 2 部分，在约束 $\sum_i w_i = 1$ 和 $\sum_i w_i (x_i - \mu) = 0$ 下最大化 $\sum_i \log w_i$ 的拉格朗日函数的一阶条件为 $1/w_i = \alpha + \beta (x_i - \mu)$。

乘以 $w_i$ 并求和得到 $n = \alpha$，因此 $w_i = 1/(n(1 + \gamma (x_i - \mu)))$，其中 $\gamma = \beta / n$。

对于第 3 部分，我们通过求根来求解 $\theta$ 和 $\gamma$。

```{code-cell} ipython3
rng_w = np.random.default_rng(11)
n = 200
x = rng_w.lognormal(0.0, 0.5, size=n)
y = 1 + 0.5 * x + rng_w.normal(0, 0.3, size=n)
μ = 1.2 * x.mean()
q_n = np.full(n, 1 / n)

θ_et, w_et = match_mean(q_n, x, μ, bracket=(-20, 20))

# γ must keep every weight positive
lo, hi = -1 / (x.max() - μ), -1 / (x.min() - μ)
γ_el = brentq(lambda γ: np.sum((x - μ) / (1 + γ * (x - μ))),
              lo + 1e-9, hi - 1e-9)
w_el = 1 / (n * (1 + γ_el * (x - μ)))

for name, wts in (("exponential tilting", w_et),
                  ("empirical likelihood", w_el)):
    print(f"{name:21}: sum {wts.sum():.6f}, weighted mean of x {wts @ x:.4f} "
          f"(target {μ:.4f}), weighted mean of y {wts @ y:.4f}")
print(f"unweighted mean of y: {y.mean():.4f}")

order = np.argsort(x)
fig, ax = plt.subplots()
ax.plot(x[order], n * w_et[order], "o-", ms=3, label="exponential tilting")
ax.plot(x[order], n * w_el[order], "s--", ms=3, label="empirical likelihood")
ax.set_xlabel("$x_i$")
ax.set_ylabel(r"$n \times$ weight")
ax.legend()
plt.show()
```

两组权重都使 $y$ 的加权均值大约提高了 $x$ 均值变化量的一半。

给定 $x_i$，线性关系 $y = 1 + 0.5 x + e$ 使预期变化恰好为一半；偏差来自噪声 $e_i$。

这两个估计值不同，因为权重不同。

指数倾斜权重随 $x$ 呈指数增长，而经验似然权重一开始增长得更为温和，然后对最大的观测值增长得更为陡峭。

```{solution-end}
```

```{exercise}
:label: ip_ex_constraint

考虑讲座中稳健决策问题的约束版本：

$$
\min_q E_q[u(X)] \quad \text{subject to} \quad D(q \| \pi) \leq \kappa .
$$

1. 利用恒等式 $E_q[u] + \lambda D(q \| \pi) = \lambda D(q \| q^*_\lambda) - \lambda \log E_\pi \exp(-u/\lambda)$，证明如果 $\lambda > 0$ 满足 $D(q^*_\lambda \| \pi) = \kappa$，那么 $q^*_\lambda$ 求解这个约束问题。
2. 对于讲座中的消费例子以及 $\kappa \in \{0.01, 0.05, 0.2\}$，求出 $\lambda$ 以及最小化的期望效用。
3. 对于 $\kappa = 0.05$，用一个通用约束优化器在满足 $D(q \| \pi) \leq \kappa$ 的条件下，对消费网格上所有分布最小化期望效用，以此检验你的答案。
```

```{solution-start} ip_ex_constraint
:class: dropdown
```

这是一个解答。

对于第 1 部分，设 $q$ 满足 $D(q \| \pi) \leq \kappa = D(q^*_\lambda \| \pi)$。

该恒等式给出

$$
E_q[u] = \lambda D(q \| q^*_\lambda) - \lambda D(q \| \pi) - \lambda \log E_\pi e^{-u/\lambda}
\geq - \lambda \kappa - \lambda \log E_\pi e^{-u/\lambda} = E_{q^*_\lambda}[u] ,
$$

其中最后一个等式是在 $q = q^*_\lambda$ 处应用该恒等式。

所以没有可行的 $q$ 能比 $q^*_\lambda$ 更好。

对于第 2 部分，$D(q^*_\lambda \| \pi)$ 随 $\lambda$ 上升而下降，因此我们可以通过对 $\log \lambda$ 求根来找到 $\lambda$。

```{code-cell} ipython3
def λ_for_entropy(κ):
    log_λ = brentq(lambda s: kl(worst_case(np.exp(s)), π_base) - κ, -6, 3)
    return np.exp(log_λ)


for κ in (0.01, 0.05, 0.2):
    λ = λ_for_entropy(κ)
    print(f"κ = {κ:4.2f}: λ = {λ:.4f} (exact {σ_c / np.sqrt(2 * κ):.4f}), "
          f"min E_q[u] = {worst_case(λ) @ u_c:.5f} "
          f"(exact {-σ_c * np.sqrt(2 * κ):.5f})")
```

精确值来自对数正态示例的公式：$D(q^*_\lambda \| \pi) = 0.2^2/(2\lambda^2) = \kappa$ 给出 $\lambda = 0.2/\sqrt{2\kappa}$，然后 $E_{q^*_\lambda}[u] = -0.2\sqrt{2\kappa}$。

对于第 3 部分，我们用对数概率 $z$ 来参数化分布，并使用能处理非线性不等式约束的 SciPy SLSQP 方法。

令 $q = \operatorname{softmax}(z)$，$E_q[u]$ 和 $D(q \| \pi)$ 相对于 $z_k$ 的梯度分别是 $q_k (u_k - E_q[u])$ 和 $q_k (\log(q_k / \pi_k) - D(q \| \pi))$，我们将其传递给优化器。

```{code-cell} ipython3
κ = 0.05


def softmax(z):
    return np.exp(z - logsumexp(z))


def expected_u(z):
    q = softmax(z)
    return q @ u_c, q * (u_c - q @ u_c)


def entropy_slack(z):
    q = softmax(z)
    return κ - kl(q, π_base)


def entropy_slack_grad(z):
    q = softmax(z)
    log_ratio = np.log(q / π_base)
    return -q * (log_ratio - q @ log_ratio)


res_q = minimize(expected_u, np.log(π_base), jac=True, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": entropy_slack,
                               "jac": entropy_slack_grad}],
                 options={"ftol": 1e-12, "maxiter": 1000})
q_opt = softmax(res_q.x)
q_tilt = worst_case(λ_for_entropy(κ))
print(f"constrained optimizer: E_q[u] = {res_q.fun:.6f}, "
      f"D(q || π) = {kl(q_opt, π_base):.6f}")
print(f"exponential tilt:      E_q[u] = {q_tilt @ u_c:.6f}")
print(f"max |q_opt - q_tilt| = {np.max(np.abs(q_opt - q_tilt)):.2e}")
```

优化器找到了指数倾斜。

```{solution-end}
```

```{exercise}
:label: ip_ex_interactions

设 $(X, Y, Z)$ 是三个二元随机变量，具有对每个八种结果中的每一个都赋予正概率的联合分布 $p$。

考虑三个嵌套的联合分布指数族：

* $\mathcal{E}_1$：$X$、$Y$ 和 $Z$ 相互独立；
* $\mathcal{E}_2$：$\log q(x, y, z)$ 是关于配对 $(x, y)$、$(x, z)$ 和 $(y, z)$ 的函数之和，因此不存在三方相互作用；以及
* $\mathcal{E}_3$：所有分布。

1. 使用 {prf:ref}`ip_thm_m_projection` 证明 $p$ 在 $\mathcal{E}_1$ 上的 m-投影是其单变量边缘分布的乘积，且在 $\mathcal{E}_2$ 上的 m-投影是 $\mathcal{E}_2$ 中与 $p$ 具有相同双变量边缘分布的成员。
2. 通过迭代比例拟合计算 $\mathcal{E}_2$ 上的 m-投影：从均匀分布开始，重复重新缩放当前分布以匹配 $p$ 的三个双变量边缘分布中的每一个。
3. 验证分解 $D(p \| p_1) = D(p \| p_2) + D(p_2 \| p_1)$，其中 $p_k$ 是在 $\mathcal{E}_k$ 上的 m-投影。左边被称为多元信息，衡量三个变量之间的总依赖性；右边第一项衡量三方相互作用，第二项衡量通过配对产生的依赖性。
4. 将此分解应用于以下分布：$X$ 和 $Y$ 是独立的公平抛硬币结果，$Z = (X + Y) \bmod 2$ 的概率为 $0.9$，$Z = (X + Y + 1) \bmod 2$ 的概率为 $0.1$。解释这个结果。

{cite:t}`Amari2001` 为任意数量的变量发展了这种分解。
```

```{solution-start} ip_ex_interactions
:class: dropdown
```

这是一个解答。

对于第 1 部分，$\mathcal{E}_1$ 是一个指数族，其充分统计量是每个变量取值的指示变量；它们的均值是单变量边缘分布。

集合 $\mathcal{E}_2$ 是一个指数族，其充分统计量是每对变量取值的指示变量；它们的均值是双变量边缘分布。

{prf:ref}`ip_thm_m_projection` 表明每个 m-投影都匹配这些均值。

单变量边缘分布的乘积属于 $\mathcal{E}_1$ 且与之匹配，因此它是 $\mathcal{E}_1$ 上的 m-投影。

类似地，$\mathcal{E}_2$ 中与 $p$ 具有相同双变量边缘分布的成员是 $\mathcal{E}_2$ 上的 m-投影。

对于第 2 部分，均匀起点位于 $\mathcal{E}_2$ 中，且每次重新缩放都将当前分布乘以一对变量的函数，因此每个迭代值都位于 $\mathcal{E}_2$ 中。

每次重新缩放都是在具有 $p$ 的双变量边缘分布之一的分布的线性族上的 e-投影。

根据证明 RAS 收敛性的论证，迭代值收敛于 $\mathcal{E}_2$ 中具有 $p$ 的双变量边缘分布的分布，根据第 1 部分，这就是 $p_2$。

```{code-cell} ipython3
def ipf_pairwise(p, tol=1e-13, max_iter=10_000):
    """The distribution with p's pairwise marginals and no 3-way interaction."""
    q = np.full(p.shape, 1 / p.size)
    for _ in range(max_iter):
        q_old = q.copy()
        for axis in (2, 1, 0):          # match the marginal of the other two
            q = q * (p.sum(axis=axis, keepdims=True)
                     / q.sum(axis=axis, keepdims=True))
        if np.max(np.abs(q - q_old)) < tol:
            break
    else:
        warnings.warn("iterative proportional fitting did not converge")
    return q


def independence_projection(p):
    px, py, pz = p.sum(axis=(1, 2)), p.sum(axis=(0, 2)), p.sum(axis=(0, 1))
    return px[:, None, None] * py[None, :, None] * pz[None, None, :]


def decompose(p):
    p1, p2 = independence_projection(p), ipf_pairwise(p)
    total, three_way, pairs = kl(p, p1), kl(p, p2), kl(p2, p1)
    print(f"multi-information D(p || p1) = {total:.6f}")
    print(f"three-way interaction D(p || p2) = {three_way:.6f}")
    print(f"pairwise dependence D(p2 || p1) = {pairs:.6f}")
    print(f"sum of the two parts = {three_way + pairs:.6f}")


rng_ipf = np.random.default_rng(12)
p_random = rng_ipf.dirichlet(np.ones(8)).reshape(2, 2, 2)
decompose(p_random)
```

对于第 3 部分，$p_2$ 是 $p$ 在 $\mathcal{E}_2$ 上的 m-投影，而 $p_1$ 位于 $\mathcal{E}_1 \subset \mathcal{E}_2$ 中，因此针对 $\mathcal{E}_2$、以 $p_1$ 作为 $q_\theta$ 角色的 {eq}`ip_pythagoras_m` 给出 $D(p \| p_1) = D(p \| p_2) + D(p_2 \| p_1)$，输出结果证实了这一点。

因为 $p$ 和 $p_2$ 具有相同的单变量边缘分布，$p_1$ 也是 $p_2$ 在 $\mathcal{E}_1$ 上的 m-投影，因此 $D(p_2 \| p_1)$ 是 $p_2$ 的多元信息。

对于第 4 部分，每对变量都是独立的，但三个变量却高度相关。

```{code-cell} ipython3
p_xor = np.zeros((2, 2, 2))
for x_ in (0, 1):
    for y_ in (0, 1):
        for z_ in (0, 1):
            agree = (x_ + y_) % 2 == z_
            p_xor[x_, y_, z_] = 0.25 * (0.9 if agree else 0.1)
decompose(p_xor)
```

所有的依赖性都是三方相互作用：每对变量都是独立的，因此在 $\mathcal{E}_2$ 和 $\mathcal{E}_1$ 上的 m-投影是一致的（都是均匀分布），然而知道其中任意两个变量几乎就能确定第三个变量。

只关注变量对的依赖性度量（例如相关系数）会完全错过这一点。

```{solution-end}
```