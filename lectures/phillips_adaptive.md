---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 适应性预期与费尔普斯问题
  headings:
    Overview: 概览
    Adaptive expectations: 适应性预期
    The Phelps problem: 费尔普斯问题
    The Phelps problem::Casting the problem in LQ form: 将问题转化为 LQ 形式
    The Phelps problem::A proposition: 一个命题
    Disinflation paths: 反通货膨胀路径
    The general Phelps problem: 一般化的费尔普斯问题
    The general Phelps problem::The adaptive case in general form: 适应性情形的一般形式
    Testing the natural-rate hypothesis: 检验自然率假说
    Sacrifice ratios and the subversion of the Phelps model: 牺牲率与费尔普斯模型的颠覆
    Sacrifice ratios and the subversion of the Phelps model::The sacrifice ratio: 牺牲率
    Sacrifice ratios and the subversion of the Phelps model::The subversion of the Phelps model: 费尔普斯模型的颠覆
    Exercises: 练习
---

(phillips_adaptive)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 适应性预期与费尔普斯问题

```{index} single: Phillips Curve; Adaptive Expectations
```

```{contents} Contents
:depth: 2
```

除了 Anaconda 中已有的库之外，本讲座还将使用以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概览

> 他们无法望向远方，
> 他们无法看透深处，
> 但这何时曾是
> 他们守望的障碍？
>
> ——罗伯特·弗罗斯特

本讲座延续了 {doc}`phillips_credibility` 中开始研究的菲利普斯曲线权衡关系。

本讲座遵循 {cite:t}`Sargent1999` 第 5 章的内容。

{doc}`phillips_credible_policies` 追求优于纳什结果的结果，同时保持*每个人*都是理性的，结果发现了太多这样的结果：一个可持续值的连续统，而理论内部没有任何依据可以从中做出选择。

在这里，我们以尽可能小的方式退让这种完美性，保持政府是理性的，但给予公众一个机械式的预测规则。

我们将描述：

* 凯根-弗里德曼适应性预期假设，
* {cite:t}`Phelps1967` 如何用它来构建一个政府控制问题，
* 该假设所隐含的牺牲率，以及
* 适应性预期在早期自然率假说计量经济学检验中所起的作用，以及针对这些检验的理性预期批判。

其核心对象是**费尔普斯问题**：一个理性的政府在解决一个最优控制问题，而公众则用一个固定的、机械式的适应性法则来预测通货膨胀。

与 {doc}`phillips_credibility` 中的单期模型不同，现在政府考虑到经济会持续多个时期，并且今天的通货膨胀会影响明天的预期。

这种跨期联系可以改善结果，在极限情形下，甚至能维持拉姆齐结果。

费尔普斯问题是一个线性二次型（LQ）控制问题，因此我们用 QuantEcon 的 {doc}`LQ 控制 <lqcontrol>` 工具来求解它。

让我们从一些导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
from scipy.signal import lfilter
```

## 适应性预期

{cite:t}`Phelps1967` 为一个自然率模型构建了一个控制问题。

他放弃了公众的理性预期假设，但保留了政府的理性预期，并给公众分配了一个特定的机械式预测法则，而这个法则是政府已知的。

公众使用米尔顿·弗里德曼和 {cite:t}`Cagan` 的适应性预期方案（关于凯根本人的用法，参见 {doc}`intro:cagan_adaptive`）：

```{math}
:label: pa_adaptive

x_t - x_{t-1} = (1 - \lambda)(y_{t-1} - x_{t-1}), \qquad \lambda \in (0, 1),
```

其中 $x_t$ 是公众对通货膨胀的预期，$y_t$ 是通货膨胀率。

重新整理后得到 $x_t = \lambda x_{t-1} + (1 - \lambda) y_{t-1}$，因此预期通货膨胀是过去通货膨胀的几何分布滞后，

```{math}
:label: pa_geom

x_t = (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i} .
```

注意，{eq}`pa_adaptive` 是 {doc}`phillips_credibility` 中最小二乘学习算法的一个*常数增益*版本，其中常数 $(1 - \lambda)$ 起到了那里递减增益 $t^{-1}$ 所起的作用。

它也是 {ref}`br_bray` 中松弛算法在日历时间下、增益为 $1 - \lambda$ 的版本。

```{note}
这里 $\lambda$ 是旧预测的权重，因此*增益*为 $1 - \lambda$。

有限理性系列讲座将增益本身记为 $\lambda$（{ref}`br_notation`），而
{doc}`phillips_misspecified` 和 {doc}`phillips_self_confirming` 则将其记为 $C = 1 - \lambda$。
```

方程 {eq}`pa_geom` 具有 {cite:t}`ChoMatsui1995` 所称的**归纳性质**：如果政府持续重复一个恒定的 $y_t = \tilde y$ 政策，最终公众会逐渐将 $x_t \approx \tilde y$ 作为其预期。

权重 $(1 - \lambda)\lambda^{i-1}$ 之和为一，因此一个永久维持的通货膨胀率最终会被完全预期到。

让我们用数值方法验证这个归纳性质。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "适应性预期收敛到恒定通货膨胀政策：归纳性质"
    name: fig-adapt-induction
---
def adaptive_forecast(y, λ, x0=0.0):
    "Simulate x_t = λ x_{t-1} + (1-λ) y_{t-1}."
    T = len(y)
    x = np.empty(T)
    x[0] = x0
    for t in range(1, T):
        x[t] = λ * x[t - 1] + (1 - λ) * y[t - 1]
    return x

T = 60
y_const = np.full(T, 10.0)                 # a constant inflation policy

fig, ax = plt.subplots(figsize=(9, 4.5))
for λ in [0.7, 0.9]:
    x = adaptive_forecast(y_const, λ)
    ax.plot(x, lw=1.5, label=rf'$\lambda = {λ}$')
ax.axhline(10.0, color='k', ls='--', lw=1, label='policy $\\tilde y$')
ax.set_xlabel('$t$')
ax.set_ylabel('expected inflation $x_t$')
ax.legend()
plt.show()
```

在恒定政策下，公众的预期会收敛到该政策，$\lambda$ 越大，收敛速度越慢。

索洛和托宾在检验自然率假说时正是利用了这个归纳性质，我们将在{ref}`下文 <pa_induction>`中加以讨论。

## 费尔普斯问题

经济永远重复运行，政府用如下贴现准则来评估结果序列：

```{math}
:label: pa_criterion

V^g = (1 - \delta) \sum_{t=1}^{\infty} \delta^{t-1} p(U_t, y_t),
\qquad p(U_t, y_t) = - \frac{1}{2}(U_t^2 + y_t^2),
\qquad \delta \in (0, 1] .
```

当 $\delta = 1$ 时，我们在均值极限（切萨罗）意义上理解 {eq}`pa_criterion`。

在该准则下，唯一重要的是结果序列的极限，因此任何将 $y_t$ 驱动至 $0$ 的路径都是最优的，过渡路径并未被唯一确定。

因此下文中 $\delta = 1$ 的结果都是 $\delta \uparrow 1$ 时的极限：代码中设定 $\beta = 1 - 10^{-7}$，而该规则在这个数字进一步缩小时几乎不再变化。

政府通过选择通货膨胀 $y_t$ 的规则来最大化 {eq}`pa_criterion`，同时受制于适应性预期方案 {eq}`pa_adaptive` 和预期增广的菲利普斯曲线：

```{math}
:label: pa_phillips

U_t = U^* - \theta(y_t - x_t) .
```

由于预期通货膨胀 $x_t$ 在第 $t$ 期开始时就已预先确定，因此它是唯一的内生状态变量。

因此，政府的问题是一个带有以下要素的贴现 LQ 控制问题：

* 状态 $s_t = \begin{bmatrix} 1 & x_t \end{bmatrix}^\top$，
* 控制 $y_t$，以及
* 转移方程 $x_{t+1} = \lambda x_t + (1 - \lambda) y_t$。

### 将问题转化为 LQ 形式

令 $U_t = a^\top s_t - \theta y_t$，其中 $a = \begin{bmatrix} U^* & \theta \end{bmatrix}^\top$。

那么每期损失 $\tfrac{1}{2}(U_t^2 + y_t^2)$ 等于

$$
\frac{1}{2}\left[ s_t^\top (a a^\top) s_t + (\theta^2 + 1) y_t^2 - 2 \theta \, y_t \, (a^\top s_t) \right] .
$$

将其与 QuantEcon 的 LQ 损失 $s_t^\top R s_t + y_t^\top Q y_t + 2 y_t^\top N s_t$ 进行匹配（关于如何处理此类交叉项，请参阅 {doc}`cross_product_trick`），并将转移方程与 $s_{t+1} = A s_t + B y_t$ 进行匹配，得到

$$
R = \tfrac{1}{2} a a^\top, \quad
Q = \tfrac{1}{2}(\theta^2 + 1), \quad
N = -\tfrac{1}{2}\theta\, a^\top, \quad
A = \begin{bmatrix} 1 & 0 \\ 0 & \lambda \end{bmatrix}, \quad
B = \begin{bmatrix} 0 \\ 1 - \lambda \end{bmatrix} .
$$

贴现因子为 $\beta = \delta$；{eq}`pa_criterion` 中的缩放因子 $(1 - \delta)$ 不会影响最优策略。

```{code-cell} ipython3
class PhelpsProblem:
    """
    The Phelps optimal-control problem: a rational government facing a
    public that forecasts inflation adaptively with parameter λ.
    """

    def __init__(self, θ=1.0, U_star=5.0, λ=0.7, δ=0.96):
        self.θ, self.U_star, self.λ, self.δ = θ, U_star, λ, δ

        a = np.array([[U_star], [θ]])
        R = 0.5 * (a @ a.T)
        Q = np.array([[0.5 * (θ**2 + 1)]])
        N = -0.5 * θ * a.T
        A = np.array([[1.0, 0.0], [0.0, λ]])
        B = np.array([[0.0], [1 - λ]])

        # δ = 1 (limit of means) is handled as the limit δ → 1
        β = min(δ, 1 - 1e-7)
        self.lq = qe.LQ(Q, R, A, B, N=N, beta=β)
        P, F, d = self.lq.stationary_values()

        # optimal rule y_t = f1 + f2 x_t
        self.f1, self.f2 = -F[0, 0], -F[0, 1]

    def simulate(self, x0=12.0, T=60):
        "Disinflation path (U_t, y_t) starting from expectation x0."
        θ, U_star, λ = self.θ, self.U_star, self.λ
        U, y, x = np.empty(T), np.empty(T), x0
        for t in range(T):
            y[t] = self.f1 + self.f2 * x
            U[t] = U_star - θ * (y[t] - x)
            x = λ * x + (1 - λ) * y[t]
        return U, y
```

最优规则的形式为 $y_t = f_1 + f_2 x_t$，其中 $f_1 \neq 0$ 且 $f_2 \neq 1$。

这些不等式反映了公众并未使用最优的预测规则；如果反而是 $f_1 = 0$ 且 $f_2 = 1$，那么我们将在所有历史情形下得到 $y_t = x_t$。

```{code-cell} ipython3
pp = PhelpsProblem(θ=1.0, U_star=5.0, λ=0.7, δ=0.96)
print(f"optimal rule:  y_t = {pp.f1:.3f} + {pp.f2:.3f} x_t")
```

### 一个命题

费尔普斯问题之所以有趣，原因在于以下结果。

```{prf:proposition} δ = 1 最终维持拉姆齐结果
:label: pa_prop

在没有贴现的情况下（$\delta = 1$），政府会将 $y_t$ 驱动至 $0$，即拉姆齐结果。
```

当 $\delta = 1$ 时，$\lambda$ 决定了向拉姆齐结果收敛的速度。

当 $\delta < 1$ 时，$y_t$ 的极限点取决于 $\lambda$ 与 $\delta$ 之间的比较关系，而我们可以将这一关系精确地表达出来。

令 $q_t$ 表示预期通货膨胀 $x_t$ 对政府的边际价值。

关于 $y_t$ 的一阶条件和关于 $x_t$ 的包络条件为

$$
\theta U_t - y_t + \delta (1 - \lambda) q_{t+1} = 0,
\qquad
q_t = -\theta U_t + \delta \lambda q_{t+1} .
$$

在稳态下 $x = y$，因此 $U = U^*$ 且 $q = -\theta U^* / (1 - \delta\lambda)$，由此一阶条件给出

```{math}
:label: pa_yinf

y_\infty = \theta U^* \, \frac{1 - \delta}{1 - \delta\lambda} .
```

该公式具有边际意义上的解释。

从稳态永久性地降低一个百分点的通货膨胀，每期都会使通货膨胀损失减少 $y_\infty$，这相当于贴现收益 $y_\infty/(1-\delta)$。

同时，在预期逐渐赶上的过程中，这也会使第 $t$ 期的失业率上升 $\theta\lambda^t$，而每一点失业率在边际上的成本约为 $U^*$，相当于贴现成本 $\theta U^*/(1 - \delta\lambda)$。

将两者等同，即得到 {eq}`pa_yinf`。

由于 $1 - \delta\lambda = (1 - \delta) + \delta(1 - \lambda)$，稳态通货膨胀是纳什速率 $\theta U^*$ 的一个比例 $(1 - \delta) / [(1 - \delta) + \delta(1 - \lambda)]$。

只有当政府的不耐心程度 $1 - \delta$ 相对于公众的收益 $1 - \lambda$ 较小时，该比例才会接近拉姆齐值 $0$。

这正是 {cite:t}`Sargent1999` 第五章中 $\lambda \ll \delta$ 且 $\delta$ 接近 1 所表达的含义。

仅仅满足 $\lambda < \delta$ 是不够的：当 $\lambda = 0.98$ 且 $\delta = 0.99$ 时，公式 {eq}`pa_yinf` 给出 $y_\infty \approx 0.34\, \theta U^*$。

在过渡路径上，公众的预期是错误的，但在稳态下由于归纳性质的作用，公众的预期是正确的。

规则 $y_t = f_1 + f_2 x_t$ 的稳态为 $f_1 / (1 - f_2)$，因此我们可以将 {eq}`pa_yinf` 与 LQ 解进行对照检验。

```{code-cell} ipython3
for λ, δ in [(0.7, 0.96), (0.9, 0.96), (0.7, 0.99), (0.98, 0.99)]:
    pp = PhelpsProblem(λ=λ, δ=δ)
    y_lq = pp.f1 / (1 - pp.f2)
    y_cf = pp.θ * pp.U_star * (1 - δ) / (1 - δ * λ)
    print(f"λ = {λ}, δ = {δ}:  LQ steady state = {y_lq:.4f},  formula = {y_cf:.4f}")
```

## 反通货膨胀路径

现在我们来重现 {cite:t}`Sargent1999` 第 5 章中的反通货膨胀实验。

设 $\theta = 1$ 且 $U^* = 5$，并从 20 世纪 70 年代末的初始条件 $x_{-1} = y_{-1} = 12$ 开始政府的问题，这意味着 $U = U^* = 5$。

下表记录了在选定滞后期下失业率 $U$ 和通货膨胀率 $y$ 的路径，涉及两个贴现因子 $\delta \in \{0.96, 1\}$ 和两个适应性参数 $\lambda \in \{0.7, 0.9\}$。

```{code-cell} ipython3
def disinflation_table(δ, lags=(1, 5, 20, 50)):
    rows = []
    for λ in [0.7, 0.9]:
        U, y = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=max(lags) + 1)
        for lag in lags:
            rows.append((λ, lag, U[lag - 1], y[lag - 1]))
    return rows

for δ in [0.96, 1.0]:
    print(f"\n δ = {δ}")
    print(f" {'λ':>4} {'lag':>4} {'U':>7} {'y':>7}")
    for λ, lag, U, y in disinflation_table(δ):
        print(f" {λ:>4} {lag:>4} {U:>7.2f} {y:>7.2f}")
```

在每种参数设置下，政府都会制造一场重大衰退，并立即将通货膨胀率降低到超过其最终极限值一半以上的程度。

在贴现情形（$\delta = 0.96$）中，通货膨胀率最终稳定在 {eq}`pa_yinf` 中给出的正值水平 $y_\infty$，且政府在 $\lambda = 0.9$ 时接受比 $\lambda = 0.7$ 时更长但更温和的衰退。

在无贴现情形（$\delta = 1$）中，通货膨胀率会一直被驱动到拉姆齐值零，$\lambda$ 越大速度越慢。

让我们绘制完整路径，其中在 $t = 0$ 处显示继承的通货膨胀率 12。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 菲尔普斯问题在有贴现和无贴现情形下的最优反通货膨胀路径
    name: fig-adapt-disinflation
---
T = 40
t = np.arange(T + 1)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

for j, δ in enumerate([0.96, 1.0]):
    for λ in [0.7, 0.9]:
        U, y = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=T)
        axes[0, j].plot(t, np.concatenate([[12.0], y]), lw=1.5,
                        label=rf'$\lambda = {λ}$')
        axes[1, j].plot(t, np.concatenate([[5.0], U]), lw=1.5,
                        label=rf'$\lambda = {λ}$')
    axes[0, j].set_title(rf'$\delta = {δ}$')
    axes[0, j].set_ylabel('inflation $y_t$')
    axes[1, j].axhline(5.0, color='k', ls=':', lw=1)
    axes[1, j].set_ylabel('unemployment $U_t$')
    axes[1, j].set_xlabel('period ($t = 0$: inherited $y = 12$)')
    axes[0, j].legend()

plt.tight_layout()
plt.show()
```

无贴现问题将通货膨胀驱动至拉姆齐结果；而有贴现问题则未能达到这一水平。

在这两种情形中，政府都是以一场衰退为代价换取立即的反通货膨胀：失业率在第一期远远跃升至 $U^*$ 之上，随后随着预期逐渐跟上而回落至 $U^*$。

## 一般化的费尔普斯问题

对于政府的控制问题而言，重要的是菲利普斯曲线的*简化形式*，而不是确定 $x_t$ 的底层结构。

有必要陈述费尔普斯问题的一个更一般化版本，其中政府的模型是一个简化形式的分布滞后菲利普斯曲线。

遵循 {cite:t}`Sargent1999`，定义向量

$$
X_{U,t-1} = \begin{bmatrix} U_{t-1} & \cdots & U_{t-m_U} \end{bmatrix}^\top,
\qquad
X_{y,t-1} = \begin{bmatrix} y_{t-1} & \cdots & y_{t-m_y} \end{bmatrix}^\top,
$$

以及 $X_{t-1} = \begin{bmatrix} X_{U,t-1}^\top & X_{y,t-1}^\top & 1 \end{bmatrix}^\top$，其中收集了 $t-1$ 期及更早时期的信息。

$X_{t-1}$ 是一般化的费尔普斯问题的状态向量。

我们可以写出两条简化形式的菲利普斯曲线，它们仅在*拟合方向*上有所不同：

$$
\text{古典型：} \quad U_t = \gamma^\top X_{C,t} + \varepsilon_{C,t},
\qquad X_{C,t} = \begin{bmatrix} y_t & X_{t-1}^\top \end{bmatrix}^\top,
$$

$$
\text{凯恩斯型：} \quad y_t = \beta^\top X_{K,t} + \varepsilon_{K,t},
\qquad X_{K,t} = \begin{bmatrix} U_t & X_{t-1}^\top \end{bmatrix}^\top.
$$

下标 $C$ 和 $K$ 分别代表*古典型*（将 $U$ 对 $y$ 回归）和*凯恩斯型*（将 $y$ 对 $U$ 回归）。

一般化的**费尔普斯问题**是要选择一个控制规律 $\hat y_t = h X_{t-1}$，以在受政府所信奉的菲利普斯曲线以及 $y_t = \hat y_t + v_{2t}$（其中 $v_{2t}$ 为控制误差）约束的条件下，最大化 {eq}`pa_criterion` 的期望值。

这诱导出一个从政府信念 $\gamma$ 到其决策规则 $h$ 的映射：

```{math}
:label: pa_map

h = h(\gamma) .
```

这些对象——$\gamma$、$\beta$、$h(\gamma)$，以及两种拟合方向——正是我们在 {doc}`phillips_self_confirming` 中定义**自我确认均衡**所需要的关键要素。

### 适应性情形的一般形式

上文所求解的具体问题是一个特殊情形，其中通过将适应性预期假设 {eq}`pa_adaptive` 代入菲利普斯曲线 {eq}`pa_phillips`，对 $\gamma$ 施加了限制。

要找出这一限制，利用 {eq}`pa_phillips` 在 $t-1$ 期的形式写出 $x_{t-1} = y_{t-1} + (U_{t-1} - U^*)/\theta$，将其代入 {eq}`pa_adaptive`，再将结果代入 {eq}`pa_phillips`：

```{math}
:label: pa_general

U_t = (1 - \lambda) U^* - \theta y_t + \lambda U_{t-1} + \theta y_{t-1} .
```

这是一条古典型菲利普斯曲线，每个变量都只有一期滞后（$m_U = m_y = 1$），其系数为 $\gamma = \begin{bmatrix} -\theta & \lambda & \theta & (1 - \lambda) U^* \end{bmatrix}^\top$，作用于 $X_{C,t} = \begin{bmatrix} y_t & U_{t-1} & y_{t-1} & 1 \end{bmatrix}^\top$。

一旦完成这种代换，状态变量 $x_t$ 就从视野中消失了。

归纳假设在 {eq}`pa_general` 中显而易见：当期与滞后通货膨胀上的权重之和为 $-\theta + \theta = 0$。

对 $y_t$ 求解后，{eq}`pa_general` 变为凯恩斯型曲线 $y_t = y_{t-1} + \theta^{-1}\left[(1 - \lambda) U^* + \lambda U_{t-1} - U_t\right]$，其滞后通货膨胀上的权重恰好为一。

由于 $x_t = y_{t-1} + \lambda (U_{t-1} - U^*)/\theta$，原问题中的规则 $y_t = f_1 + f_2 x_t$，用 $X_{t-1} = \begin{bmatrix} U_{t-1} & y_{t-1} & 1 \end{bmatrix}^\top$ 表示，即为

```{math}
:label: pa_h

h = \begin{bmatrix} f_2 \lambda / \theta & f_2 & f_1 - f_2 \lambda U^* / \theta \end{bmatrix} .
```

让我们通过直接求解一般化的费尔普斯问题来验证这一点。

以 $X_{t-1}$ 为状态、$y_t$ 为控制、以 $U_t = \gamma_y y_t + c^\top X_{t-1}$ 为所信奉的曲线（其中 $\gamma_y$ 是 $y_t$ 上的系数，$c$ 收集其余部分），该问题再次是一个 LQ 问题。

{doc}`phillips_learning` 求解了同一问题，其中每个变量都有两期滞后，适用于每期都重新估计 $\gamma$ 的政府。

```{code-cell} ipython3
def phelps_general(γ, δ):
    """
    Solve the general Phelps problem for the believed classical curve
    U_t = γ[0] y_t + γ[1] U_{t-1} + γ[2] y_{t-1} + γ[3].
    Returns h, with ŷ_t = h @ [U_{t-1}, y_{t-1}, 1].
    """
    γ_y = γ[0]
    c = np.asarray(γ[1:], dtype=float).reshape(-1, 1)
    R = 0.5 * (c @ c.T)
    Q = np.array([[0.5 * (γ_y**2 + 1)]])
    N = 0.5 * γ_y * c.T
    A = np.vstack([c.T, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # next state (U_t, y_t, 1)
    B = np.array([[γ_y], [1.0], [0.0]])
    lq = qe.LQ(Q, R, A, B, N=N, beta=min(δ, 1 - 1e-7))
    P, F, d = lq.stationary_values()
    return -F[0]

θ, U_star, λ, δ = 1.0, 5.0, 0.7, 0.96
γ_adaptive = np.array([-θ, λ, θ, (1 - λ) * U_star])
pp = PhelpsProblem(θ=θ, U_star=U_star, λ=λ, δ=δ)

h_general = phelps_general(γ_adaptive, δ)
h_from_x = np.array([pp.f2 * λ / θ, pp.f2, pp.f1 - pp.f2 * λ * U_star / θ])
print("h from the general problem:  ", h_general.round(4))
print("h from the x-state rule:     ", h_from_x.round(4))
```

两者一致：原始的费尔普斯问题正是通过 {eq}`pa_general` 对 $\gamma$ 施加限制后得到的一般化问题的特殊情形。

```{note}
**归纳假设**是这样一种限制：在凯恩斯型菲利普斯曲线 $y_t = \beta^\top X_{K,t} + \varepsilon_{K,t}$ 中，
滞后 $y$ 值上的权重之和为一（等价地，在古典形式中，当期与滞后 $y$ 值上的权重之和为零）。

在适应性预期下，这一点成立，因为 {eq}`pa_geom` 中的权重之和为一，正如 {eq}`pa_general` 所示。
```

(pa_induction)=
## 检验自然率假说

罗伯特·索洛和詹姆斯·托宾 {cite}`Solow1968,Tobin1968` 利用归纳假设检验了自然率假说。

将几何分布滞后 {eq}`pa_geom` 代入一条反转的菲利普斯曲线，得到

```{math}
:label: pa_invphill

y_t = (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i}
      + \theta^{-1}(U^* - U_t) .
```

他们提出通过运行如下回归来检验自然率假说：

```{math}
:label: pa_tobin

y_t = b_0 + b_1 (1 - \lambda) \sum_{i=1}^{\infty} \lambda^{i-1} y_{t-i}
      + b_2 U_t + \varepsilon_t ,
```

并将 $b_1 < 1$ 的发现解读为存在通货膨胀与失业之间长期权衡关系的证据。

在通货膨胀保持恒定的稳态中，{eq}`pa_tobin` 蕴含了长期斜率 $dU/dy = (1 - b_1)/b_2$，当 $b_2$ 取 {eq}`pa_invphill` 中的值 $-1/\theta$ 时，该斜率等于 $\theta(b_1 - 1)$。

早期的实证研究发现 $b_1 < 1$，因此拒绝了自然率假说，转而支持一种长期权衡关系。

后来对相同检验的实施则未能拒绝该假说。

{cite:t}`Lucas1972` 和 {cite:t}`Sargent1971` 指出，这些拒绝结果与自然率假说无关。

索洛和托宾是通过利用归纳假设来识别预期通货膨胀，从而推导出限制条件 $b_1 = 1$ 的。

但归纳假设描述的只是一种特定的实验情形，即永久且固定的通货膨胀率，而在样本中实际支配通货膨胀的随机过程下，这种情形可能是极不可能发生的。

在理性预期下，将预期通货膨胀与过去通货膨胀联系起来的权重受该过程的约束，其总和未必等于一。

例如，假设通货膨胀是一个根为 $\rho$ 的平稳 AR(1) 过程，$x_t = \mathbb{E}_{t-1} y_t$，并且自然率菲利普斯曲线成立。

那么总体回归 {eq}`pa_tobin` 给出

```{math}
:label: pa_bias

b_1 = \frac{\rho (1 + \lambda)}{1 + \lambda \rho} < 1 ,
```

这是对一个实际成立的假说的"拒绝"（{ref}`pa_ex2` 推导了该公式并通过模拟进行了验证）。

当 $\rho \to 1$ 时，该估计值趋近于一。

根据 {cite:t}`Sargent1971` 和 {cite:t}`KingWatson1994` 的论证，这正是为何拒绝与不拒绝的模式，与通货膨胀率在 20 世纪 60 年代之后（而非之前）呈现单位根这一趋势相一致的原因：当 $y_t$ 存在单位根时，单位和限制条件 $b_1 = 1$ 与理性预期是*相容*的。

{doc}`phillips_self_confirming` 中的自我确认均衡提供了另一个例证：在那里，理性预期和自然率假说都成立，而归纳假设不成立。

从费尔普斯控制问题的视角来看，自然率假说是否成立是次要的：无论 $b_1 = 1$ 是否成立，费尔普斯问题都会为通货膨胀-失业选择赋予有趣的动态特征。

公众的预期本身从该问题中消失了；真正重要的是 {eq}`pa_tobin` 中 $y$ 的分布滞后在零频率处的行为。

## 牺牲率与费尔普斯模型的颠覆

**牺牲率**——即将通货膨胀降低一个百分点所需放弃的估计产出数量——的估计值在 20 世纪 70 年代末被广泛流传。

适应性模型给出了一个简单精确形式的牺牲率。

(pa_sacrifice)=
### 牺牲率

由 {eq}`pa_phillips` 和 {eq}`pa_adaptive` 可得：

$$
U_t - U^* = -\theta (y_t - x_t) = - \frac{\theta}{1 - \lambda} (x_{t+1} - x_t) .
$$

对 $t = 1, \ldots, T$ 求和，右侧呈望远镜式抵消：

```{math}
:label: pa_sacrifice_eq

\sum_{t=1}^{T} (U_t - U^*) = \frac{\theta}{1 - \lambda} \, (x_1 - x_{T+1}) .
```

因此，预期通货膨胀每降低一个百分点所累积的超额失业率为 $\theta / (1 - \lambda)$，这对于*任意*政策路径以及政府的任意贴现因子都成立。

由于在稳态中预期通货膨胀与实际通货膨胀相一致，这也是每一个百分点永久性反通货膨胀所需付出的代价。

政府无法改变这一比率。

它只能选择反通货膨胀的幅度以及如何将失业分摊到不同时期；在二次损失下，它会将其分摊开来。

当 $\theta = 1$ 时，$\lambda = 0.7$ 时该比率为 $3.33$，$\lambda = 0.9$ 时为 $10$。

让我们沿着最优路径来验证这一点。

```{code-cell} ipython3
for λ in [0.7, 0.9]:
    for δ in [0.96, 1.0]:
        U, y = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=2000)
        ratio = np.sum(U - 5.0) / (12.0 - y[-1])
        print(f"λ = {λ}, δ = {δ}:  sacrifice ratio = {ratio:.3f},  "
              f"θ/(1-λ) = {1 / (1 - λ):.3f}")
```

较长的预期滞后（即 $\lambda$ 接近 $1$）意味着较大的牺牲率。

### 费尔普斯模型的颠覆

尽管费尔普斯的控制问题在归纳假设下对维持拉姆齐结果具有令人鼓舞的意义，但它也带有一段不光彩的历史。

对于固定的 $\delta < 1$，总能找到一个足够接近 $1$ 的 $\lambda$，使得继承而来的高通胀预期会让政府想要避免拉姆齐结果。

公式 {eq}`pa_yinf` 说明了其中的原因：当 $\lambda \to 1$ 时，$y_\infty \to \theta U^*$，即纳什利率。

下一个代码单元固定 $\delta = 0.96$，并让 $\lambda$ 趋近于 $1$。

```{code-cell} ipython3
δ = 0.96
print(f"{'λ':>6} {'first-period y':>15} {'y_∞':>7} {'θ/(1-λ)':>9}")
for λ in [0.7, 0.9, 0.95, 0.99, 0.999]:
    pp = PhelpsProblem(λ=λ, δ=δ)
    y_first = pp.f1 + pp.f2 * 12.0
    y_inf = pp.θ * pp.U_star * (1 - δ) / (1 - δ * λ)
    print(f"{λ:>6} {y_first:>15.2f} {y_inf:>7.2f} {pp.θ / (1 - λ):>9.1f}")
```

政府在第一期仍然会大幅削减通货膨胀，但远未降到零，而每削减一个百分点，就要付出 $\theta / (1 - \lambda)$ 个百分点的失业率代价。

在 20 世纪 70 年代末，具有较长预期调整滞后的模型正是这样被用来建议*不要*降低通货膨胀的，这进一步坚定了那些主张应学会容忍高通胀的人士的立场。

{cite:t}`Lucas1972` 和 {cite:t}`Sargent1971` 通过强调理性预期的跨方程约束指出，历史上对 $\lambda$ 的估计并不能说明反通货膨胀的最优速度：$\lambda$ 描述的是预期对过去政策的反应方式，而非对一项新政策将如何反应。

需要吸取并延续下去的教训是：激活归纳假设最终可以带来更好的结果，但以费尔普斯、托宾和索洛所使用的形式，归纳假设背离了理性预期。

在 {doc}`phillips_misspecified` 和 {doc}`phillips_self_confirming` 中，适应性参数变成了一个均衡结果，$C = 1 - \lambda$，由公众选择以拟合其自身预测所助推产生的数据。

从 {doc}`phillips_learning` 起，政府将类似 {eq}`pa_adaptive` 的更新方案应用于函数——即其菲利普斯曲线的系数——而非数字。

这里所求解的 LQ 费尔普斯问题会在 {doc}`phillips_learning`、{doc}`phillips_escaping_nash`、{doc}`phillips_priors` 以及 {doc}`phillips_lost_conquest` 中再次出现，在那些讲座中，一个进行学习的政府每期都重新求解该问题。

在 {doc}`phillips_learning` 中，正是政府所估计的菲利普斯曲线向归纳假设漂移这一过程，触发了沃尔克式的稳定化措施。

## 练习

```{exercise-start}
:label: pa_ex1
```

公式 {eq}`pa_yinf` 使得 $\lambda$ 与 $\delta$ 的比较变得精确。

1. 对于 $\lambda = 0.8$，请针对贴现因子网格 $\delta \in \{0.90, 0.91, \ldots, 0.99\}$，计算极限通货膨胀率 $y_\infty$（即经过长时间模拟后 $y_t$ 所稳定到的值），将其对 $\delta$ 作图，并叠加绘制 {eq}`pa_yinf`。
2. 对于 $\delta \in \{0.9, 0.96, 0.99\}$，将 $y_\infty / \theta U^*$ 对 $\lambda \in [0, 0.999]$ 作图。
3. 对于每个这样的 $\delta$，$\lambda$ 需要多小才能使 $y_\infty$ 落在拉姆齐值的 10% 以内，即 $y_\infty \le 0.1\, \theta U^*$？

```{exercise-end}
```

```{solution-start} pa_ex1
:class: dropdown
```

```{code-cell} ipython3
δ_grid = np.arange(0.90, 0.995, 0.01)
y_sim = [PhelpsProblem(λ=0.8, δ=δ).simulate(x0=12.0, T=400)[1][-1]
         for δ in δ_grid]
y_formula = 5.0 * (1 - δ_grid) / (1 - δ_grid * 0.8)

λ_grid = np.linspace(0, 0.999, 500)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(δ_grid, y_sim, 'o', label='simulation')
axes[0].plot(δ_grid, y_formula, lw=2, label='formula')
axes[0].set_xlabel(r'discount factor $\delta$')
axes[0].set_ylabel(r'limiting inflation $y_\infty$')
axes[0].set_title(r'$\lambda = 0.8$')
axes[0].legend()

for δ in [0.9, 0.96, 0.99]:
    axes[1].plot(λ_grid, (1 - δ) / (1 - δ * λ_grid), lw=2,
                 label=rf'$\delta = {δ}$')
    λ_max = (1 - 10 * (1 - δ)) / δ
    print(f"δ = {δ}: y_∞ ≤ 0.1 θU* requires λ ≤ {λ_max:.3f}")
axes[1].axhline(0.1, color='k', ls=':', lw=1)
axes[1].set_xlabel(r'adaptation parameter $\lambda$')
axes[1].set_ylabel(r'$y_\infty / \theta U^*$')
axes[1].legend()
plt.tight_layout()
plt.show()
```

模拟得到的极限值落在公式曲线上。

随着 $\delta$ 向一上升，政府愿意接受为获得低预期通货膨胀的长期收益所需的过渡性衰退，因此极限通货膨胀率会向拉姆齐值零下降。

根据 {eq}`pa_yinf`，$y_\infty \le 0.1\, \theta U^*$ 要求 $1 - \delta\lambda \ge 10(1 - \delta)$，即 $\lambda \le [1 - 10(1 - \delta)]/\delta$。

当 $\delta = 0.9$ 时，没有任何 $\lambda \in (0, 1)$ 能够满足此条件；即使 $\delta = 0.99$ 时，公众也必须调整得足够快，使得 $\lambda$ 低于约 $0.91$。

当 $\lambda \to 1$ 时，每条曲线都会上升至一：无论贴现因子如何，极慢的适应调整都会使费尔普斯模型失效。

```{solution-end}
```

```{exercise-start}
:label: pa_ex2
```

本练习以 {eq}`pa_bias` 的形式重现了卢卡斯-萨金特对索洛-托宾检验的批判。

考虑这样一种经济：自然率假说与理性预期同时成立：

$$
y_t = (1 - \rho)\mu + \rho y_{t-1} + e_t,
\qquad x_t = \mathbb{E}_{t-1} y_t,
\qquad U_t = U^* - \theta (y_t - x_t) + v_{1t},
$$

其中 $e_t$ 和 $v_{1t}$ 是方差分别为 $\sigma_e^2$ 和 $\sigma_1^2$ 的独立白噪声，且 $|\rho| < 1$。

1. 证明总体回归 {eq}`pa_tobin` 满足 $b_1 = \rho(1+\lambda)/(1+\lambda\rho)$ 以及 $b_2 = -\theta\sigma_e^2/(\theta^2\sigma_e^2 + \sigma_1^2)$。
2. 取 $\lambda = 0.7$、$\theta = 1$、$U^* = 5$、$\mu = 4$、$\sigma_e = 1$、$\sigma_1 = 0.5$，以及 $\rho \in \{0, 0.5, 0.9, 0.99\}$，通过模拟验证这两个结果。
3. 解释这些结果。

```{exercise-end}
```

```{solution-start} pa_ex2
:class: dropdown
```

记 {eq}`pa_tobin` 中的回归元为 $z_t = (1 - \lambda)\sum_{i \geq 1} \lambda^{i-1} y_{t-i}$。

在理性预期下，$y_t = x_t + e_t$，其中 $x_t = (1 - \rho)\mu + \rho y_{t-1}$，且 $U_t - U^* = -\theta e_t + v_{1t}$。

由于 $U_t$ 只依赖于当期冲击，它与 $z_t$ 以及 $x_t$ 都不相关，因此两个斜率可以分别计算。

$U_t$ 的系数为 $b_2 = \operatorname{cov}(e_t, U_t) / \mathbb{V}[U_t] = -\theta\sigma_e^2/(\theta^2\sigma_e^2 + \sigma_1^2)$。

$z_t$ 的系数为 $b_1 = \operatorname{cov}(x_t, z_t)/\mathbb{V}[z_t] = \rho \operatorname{cov}(y_{t-1}, z_t)/\mathbb{V}[z_t]$。

设 $\gamma_0 = \mathbb{V}[y_t]$，自协方差为 $\gamma_0 \rho^j$，则

$$
\operatorname{cov}(y_{t-1}, z_t) = \frac{(1 - \lambda)\gamma_0}{1 - \lambda\rho},
\qquad
\mathbb{V}[z_t] = \frac{(1 - \lambda)^2 \gamma_0 (1 + \lambda\rho)}{(1 - \lambda^2)(1 - \lambda\rho)},
$$

两者之比即给出 $b_1 = \rho(1 + \lambda)/(1 + \lambda\rho)$。

```{code-cell} ipython3
rng = np.random.default_rng(0)
λ, θ, U_star, μ, σ_e, σ_1, T = 0.7, 1.0, 5.0, 4.0, 1.0, 0.5, 200_000
burn = 2_000

for ρ in [0.0, 0.5, 0.9, 0.99]:
    e = σ_e * rng.standard_normal(T)
    v1 = σ_1 * rng.standard_normal(T)
    y = μ + lfilter([1.0], [1.0, -ρ], e)                  # AR(1) around μ
    U = U_star - θ * e + v1                                # y_t - x_t = e_t
    z = lfilter([0.0, 1 - λ], [1.0, -λ], y)               # (1-λ) Σ λ^{i-1} y_{t-i}
    X = np.column_stack([np.ones(T - burn), z[burn:], U[burn:]])
    b0, b1, b2 = np.linalg.lstsq(X, y[burn:], rcond=None)[0]
    print(f"ρ = {ρ:<4}:  b1 = {b1:.3f} (formula {ρ * (1 + λ) / (1 + λ * ρ):.3f}),"
          f"  b2 = {b2:.3f} (formula {-θ * σ_e**2 / (θ**2 * σ_e**2 + σ_1**2):.3f})")
```

估计结果与公式相符。

在每种情形下，$b_1 < 1$，因此即使自然率假说在此成立，索洛-托宾检验仍会拒绝它。

归纳假设内置了求和为一的权重，但平稳过程的理性预期并不如此。

当 $\rho \to 1$ 时，偏差消失，这正是金-沃森（King-Watson）对后续检验为何不再拒绝该假说的解读：这些检验所用的数据中，通货膨胀具有单位根。

系数 $b_2$ 正是*凯恩斯主义*回归（将通货膨胀对失业率回归）所恢复出的斜率。

它正是 {doc}`phillips_self_confirming` 中凯恩斯主义自我确认均衡的斜率 $\beta_1$，其中 $\sigma_e$ 扮演了 $\sigma_2$ 的角色。

```{solution-end}
```

```{exercise-start}
:label: pa_ex3
```

牺牲率 {eq}`pa_sacrifice_eq` 并不依赖于政策本身。

固定 $\delta = 0.96$，取 $\lambda \in \{0.7, 0.9, 0.95, 0.99\}$，起始点为 $x = 12$。

1. 沿最优路径计算每一个百分点反通货膨胀所对应的累积超额失业。
2. 对一种永远设定 $y_t = 0$ 的"速战速决"政策，以及一种从 11 开始每期将通货膨胀降低一个百分点直至为零的渐进政策，分别做同样的计算。
3. 将这三者与 $\theta / (1 - \lambda)$ 作比较，并解释为什么最优政策仍然与速战速决政策不同。

```{exercise-end}
```

```{solution-start} pa_ex3
:class: dropdown
```

```{code-cell} ipython3
def sacrifice_ratio(y_path, λ, θ=1.0, x0=12.0):
    "Cumulative excess unemployment per point of fall in expected inflation."
    x, excess = x0, 0.0
    for y in y_path:
        excess += -θ * (y - x)           # U_t - U*
        x = λ * x + (1 - λ) * y
    return excess / (x0 - x)

T = 5_000
δ = 0.96
for λ in [0.7, 0.9, 0.95, 0.99]:
    U, y_opt = PhelpsProblem(λ=λ, δ=δ).simulate(x0=12.0, T=T)
    y_cold = np.zeros(T)
    y_grad = np.maximum(11.0 - np.arange(T), 0.0)
    print(f"λ = {λ}:  optimal {sacrifice_ratio(y_opt, λ):6.2f},  "
          f"cold turkey {sacrifice_ratio(y_cold, λ):6.2f},  "
          f"gradual {sacrifice_ratio(y_grad, λ):6.2f},  "
          f"θ/(1-λ) = {1 / (1 - λ):6.2f}")
```

正如 {eq}`pa_sacrifice_eq` 所要求的那样，这三种政策每降低一个百分点的通货膨胀，都需要付出 $\theta/(1 - \lambda)$ 个百分点的失业代价。

这些政策的差异在于它们将通货膨胀降低到何种程度，以及如何将失业分摊到不同时期。

速战速决政策会将通货膨胀一路降到零，并将衰退前置。

最优贴现政策则在 $y_\infty > 0$ 处停止，原因正如 {eq}`pa_yinf` 之后所述：在稳态附近，最后一个百分点反通货膨胀的贴现收益 $y_\infty/(1-\delta)$，恰好等于其贴现的失业成本 $\theta U^*/(1-\delta\lambda)$。

当 $\lambda \to 1$ 时，该成本会上升至 $\theta U^*/(1-\delta)$，因此 $y_\infty$ 也会上升至 $\theta U^*$。

在二次损失函数下，政府也更倾向于将失业分摊开来，而不是一次性承受。

```{solution-end}
```
