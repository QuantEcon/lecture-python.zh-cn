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
  title: 先验、逃逸与学习循环
  headings:
    Overview: 概述
    The static model: 静态模型
    The self-confirming equilibrium: 自证均衡
    Drifting beliefs and the Kalman filter: 漂移信念与卡尔曼滤波
    Mean dynamics and E-stability: 均值动态与 E-稳定性
    Learning cycles: 学习循环
    Escape dynamics and the direction of escape: 逃逸动态与逃逸方向
    Sims's nonconvergence: 西姆斯的不收敛现象
    Sims's nonconvergence::The variance of the regression error: 回归误差的方差
    Sims's nonconvergence::Replicating Sims's example: 重现西姆斯的例子
    Sims's nonconvergence::Why the mean dynamics do not reassert themselves: 为何均值动态不再重新发挥作用
    Sims's nonconvergence::Constant gains and the sawtooth: 常增益与锯齿形走势
    Conclusion: 结论
    Conclusion::Are the estimated beliefs realistic?: 估计出的信念是否现实？
    Exercises: 练习
---

(phillips_priors)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 先验、逃逸与学习循环

```{index} single: Phillips Curve; Priors and Learning Cycles
```

```{contents} Contents
:depth: 2
```

## 概述

> 波旁家族什么都记得，却什么也没学到。
>
> —— 夏尔·莫里斯·德·塔列朗

本讲座是 {doc}`phillips_learning` 和 {doc}`phillips_escaping_nash` 的续篇。

它遵循 {cite:t}`SargentWilliams2005` 的方法，该文献推广了 {cite:t}`Sargent1999` 第 8 章以及 {cite:t}`ChoWilliamsSargent2002`（CWS）的适应性模型，我们在 {doc}`phillips_escaping_nash` 中推导过后者的逃逸动态。

在 {doc}`phillips_learning` 中，政府通过常增益递归最小二乘法估计其菲利普斯曲线。

那是一种非常特定的学习方式。

在大样本下，常增益最小二乘学习者的行为就好像它相信其模型的系数遵循一个具有特定新息协方差矩阵的随机游走。

在这里，我们让政府对其系数如何漂移持有*任意*先验信念，用协方差矩阵 $V$ 来编码，并考察这种先验如何影响驱动模型的两种力量：

* **均值动态**，它将政府的信念拉向自证（纳什）均衡；以及
* **逃逸动态**，它反复将信念推离自证均衡，转向低通胀的拉姆齐结果。

我们将发现三件事。

1. 某些先验会使自证均衡变得*不稳定*，从而使均值动态本身产生反复出现的反通胀：这是一种诞生于霍普夫分岔的**学习循环**，而非随机逃逸。
2. 先验决定了逃逸的*方向*和*速度*，但在我们研究的每种情况下，逃逸的目的地都是拉姆齐结果。
3. 先验解决了文献中的一个谜题：为什么 {cite:t}`Sims1988` 和 {cite:t}`Chung1990` 模拟的政府会逃离纳什通胀一次之后便停留在拉姆齐附近，看似无限期地持续下去，而 {cite:t}`Sargent1999` 和 CWS 的模拟却只是一次次逃逸后又被拉回。

在整个讲座中，我们使用 {doc}`phillips_escaping_nash` 和 {ref}`olg_phillips` 中可解析处理的**静态**模型，在该模型中政府对失业率关于通胀和常数进行回归。

让我们从导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
```

## 静态模型

真实经济是 {cite:t}`KydlandPrescott1977` 自然率模型的一个版本：

```{math}
:label: pp_truth

\begin{aligned}
U_n &= u - (\pi_n - \hat x_n) + \sigma_1 W_{1n}, \qquad u > 0, \\
\pi_n &= x_n + \sigma_2 W_{2n}, \\
\hat x_n &= x_n,
\end{aligned}
```

其中 $U_n$ 是失业率，$\pi_n$ 是通货膨胀率，$x_n$ 是政府设定的通胀的系统性部分，$\hat x_n$ 是公众的（理性）预测，$W_n = (W_{1n}, W_{2n})^\top$ 是独立同分布的标准高斯噪声。

由于 $\pi_n - \hat x_n = \sigma_2 W_{2n}$，真实的失业率是 $U_n = u - \sigma_2 W_{2n} + \sigma_1 W_{1n}$——无论系统性政策如何，它都围绕自然率 $u$ 波动。

政府并不知道这一点。

在**静态模型**中，政府通过对失业率关于当前通胀和常数进行回归来拟合一个非预期性的菲利普斯曲线，

```{math}
:label: pp_belief

U_n = a + b\, \pi_n + \eta_n ,
```

信念向量为 $\gamma = (a, b)$（截距和斜率），并将 $\eta_n$ 视为外生冲击。

```{warning}
{doc}`phillips_escaping_nash` 研究的是同一个静态模型，但信念向量的排列顺序相反，写作
$\gamma = (\gamma_1, \gamma_{-1})$，*斜率*在前，回归量为 $\Phi = (\pi, 1)$，遵循
{cite:t}`ChoWilliamsSargent2002` 的做法。

在这里，我们把*截距*放在前面，$\Phi = (1, \pi)$，遵循 {cite:t}`SargentWilliams2005` 的做法，{ref}`olg_phillips` 也采用了同样的做法。

这两者其实是同一个模型在不同坐标下的表示：$a = \gamma_{-1}$ 且 $b = \gamma_1$，因此本讲中的自我确认信念
$(2u, -1)$ 对应于上一讲中的 $(-1, u(1+\theta^2))$，其中 $\theta = 1$。

诸如 $M$、$V$ 和 $P$ 这样的矩阵，其行和列也相应地进行了转置。

{ref}`olg_phillips` 使用了我们的校准设定，但采用了不同的符号，正如有限理性系列讲座所做的那样
（{ref}`br_notation`）：在那里，信念记为 $\alpha$，增益记为 $\gamma$。

而在这里，遵循 {cite:t}`SargentWilliams2005` 的做法，信念记为 $\gamma$，增益记为 $\varepsilon$，
$\alpha$ 则表示政府认为生成其数据的漂移系数。
```

在相信 {eq}`pp_belief` 的前提下，政府求解菲尔普斯问题——最小化 $\hat{\mathbb{E}} \sum_n \delta^n (U_n^2 + \pi_n^2)$——其静态最优反应将通胀设定为常数

```{math}
:label: pp_bestresp

x(\gamma) = -\frac{a\, b}{1 + b^2} .
```

```{code-cell} ipython3
class StaticPhillips:
    "The static Sargent-Williams model: government regresses U on (1, π)."

    def __init__(self, u=5.0, σ1=0.3, σ2=0.3):
        self.u, self.σ1, self.σ2 = u, σ1, σ2
        self.σ = σ1                       # govt regression error std = σ1

    def x(self, γ):
        "Government best response (systematic inflation) given beliefs γ."
        a, b = γ
        return -a * b / (1 + b**2)

    def M(self, γ):
        "Second moment matrix E[ΦΦ'] of regressors Φ = (1, π), given γ."
        x = self.x(γ)
        return np.array([[1.0, x], [x, x**2 + self.σ2**2]])

    def g_bar(self, γ):
        "Least squares moment E[Φ(U - Φ'γ)] under the data γ generates."
        u, σ2 = self.u, self.σ2
        x = self.x(γ)
        E_ΦU = np.array([u, x * u - σ2**2])
        return E_ΦU - self.M(γ) @ γ
```

有三个信念向量值得命名。

* **信念 1（纳什）：** $b = -1$，截距使政府设定 $x = u$，这是 {cite:t}`KydlandPrescott1977` 的时间一致性结果。
* **信念 2（拉姆齐）：** $b = 0$，因此政府认为*不存在*权衡，设定 $x = 0$。
* **信念 3（归纳）：** 在动态版本中，当前和滞后通胀的系数之和为零，这对于一个有耐心的政府来说也会使通胀趋向 $0$。

## 自证均衡

自证均衡是一个能自我再生的信念 $\bar\gamma$：当政府依据 $\bar\gamma$ 行事时所产生的数据，使得总体回归系数恰好等于 $\bar\gamma$，即 $\bar g(\bar\gamma) = 0$。

对于静态模型，这可以手工轻松求解。

斜率为 $b = \operatorname{cov}(U, \pi)/\mathbb{V}[\pi] = -\sigma_2^2/\sigma_2^2 = -1$，均值匹配给出截距 $a = u + x(\bar\gamma)$。

将最优反应 {eq}`pp_bestresp`（$b = -1$）代入，得到 $x = a/2$，因此 $a = u + a/2$，即 $a = 2u$。

```{code-cell} ipython3
model = StaticPhillips(u=5.0, σ1=0.3, σ2=0.3)
γ_sce = np.array([2 * model.u, -1.0])

print(f"self-confirming beliefs  γ = {γ_sce}  (intercept 2u, slope -1)")
print(f"self-confirming inflation x = {model.x(γ_sce):.2f}  (= Nash = u)")
print(f"check g_bar(γ_sce) = {model.g_bar(γ_sce)}")
```

自证均衡的通胀率等于纳什（时间一致性）结果 $u = 5$，即使政府的模型是设定错误的（{ref}`sc_why_nash` 解释了其中的原因）。

政府的非预期性菲利普斯曲线在均衡路径*上*与真实模型在观测上等价，但在均衡路径*之外*是错误的——而这正是它的适应性行为反复试探的地方。

## 漂移信念与卡尔曼滤波

现在我们让政府变得具有适应性。

依照 {cite:t}`SargentWilliams2005`，政府相信其数据来自一条系数以随机游走形式*漂移*的菲利普斯曲线，

```{math}
:label: pp_drift

U_n = \Phi_n^\top \alpha_n + \eta_n,
\qquad
\alpha_{n+1} = \alpha_n + \Lambda_{n+1},
\qquad
\mathbb{E}\, \eta_n^2 = \sigma^2,
\qquad
\operatorname{cov}(\Lambda_n) = V ,
```

其中回归量为 $\Phi_n = (1, \pi_n)^\top$，$\eta_n$ 和 $\Lambda_n$ 为独立同分布的冲击。

协方差矩阵 $V$ 是政府对**参数漂移的先验信念**，这是我们要放开的对象，而 $\sigma^2$ 是政府赋予其回归误差 $\eta_n$ 的方差。

政府的估计值 $\gamma_n$ 是其基于到 $n-1$ 期为止的数据对 $\alpha_n$ 做出的最佳预测，它通过卡尔曼滤波来计算这一估计值。

这正是 {doc}`kalman` 中的线性状态空间模型，其隐藏状态为 $\alpha_n$，转移矩阵为 $I$，状态噪声协方差为 $V$，观测矩阵随时间变化为 $\Phi_n^\top$，测量噪声方差为 $\sigma^2$。

依照 {cite:t}`SargentWilliams2005`，我们以 $\sigma^2$ 为单位度量估计误差的协方差，记 $\operatorname{cov}(\alpha_n - \gamma_n) = \sigma^2 P_n$。

于是卡尔曼滤波为

```{math}
:label: pp_kalman_exact

\begin{aligned}
\gamma_{n+1} &= \gamma_n + \frac{P_n \Phi_n}{1 + \Phi_n^\top P_n \Phi_n}\left(U_n - \Phi_n^\top \gamma_n\right), \\
P_{n+1} &= P_n - \frac{P_n \Phi_n \Phi_n^\top P_n}{1 + \Phi_n^\top P_n \Phi_n} + \sigma^{-2} V .
\end{aligned}
```

正是这种归一化，使得分母中出现 $1$，并使 $P_n$ 的更新式中出现经过缩放的漂移项 $\sigma^{-2} V$。

对于较大的 $n$，{cite:t}`BenvenisteMetivierPriouret1990` 证明该滤波器可以很好地近似为

```{math}
:label: pp_kalman

\begin{aligned}
\gamma_{n+1} &= \gamma_n + P_n \Phi_n\left(U_n - \Phi_n^\top \gamma_n\right), \\
P_{n+1} &= P_n - P_n M(\gamma_n) P_n + \sigma^{-2} V ,
\end{aligned}
```

其中二阶矩矩阵 $M(\gamma_n)$ 在 $P_n$ 的更新式中取代了 $\Phi_n \Phi_n^\top$。

{cite:t}`SargentWilliams2005` 的分析以 {eq}`pp_kalman` 为基础，并报告称，在他们所研究的小漂移情形下，该式给出的模拟结果与 {eq}`pp_kalman_exact` 的结果几乎没有区别。

在本文中，我们始终使用精确滤波器 {eq}`pp_kalman_exact` 进行模拟，因为当漂移较大时（如下文西姆斯的例子中那样），该近似可能不稳定。

对于固定的 $\gamma$，矩阵 $P_n$ 收敛到代数里卡蒂方程的解

```{math}
:label: pp_riccati

- P M(\gamma) P + \sigma^{-2} V = 0 ,
```

这是本模型对应于 {ref}`kalman_convergence` 中代数里卡蒂方程的对应形式。

依照 {cite:t}`SargentWilliams2005`，我们用一个标量 $\varepsilon$ 来刻画漂移的大小，记 $V = \varepsilon^2 \hat V$，其中 $\hat V$ 为固定矩阵。

于是方程 {eq}`pp_riccati` 使得 $P$ 与 $\varepsilon / \sigma$ 成正比，因此 $\varepsilon$ 起到了 {doc}`phillips_learning` 中固定增益的作用。

我们的大多数计算使用 $\varepsilon = 0.05$，这是 {cite:t}`SargentWilliams2005` 表 1 和表 2 中最大的增益；在与固定增益最小二乘法比较时，我们使用 {ref}`olg_phillips` 中的增益 $0.02$，而西姆斯的例子则带有其自身更大得多的漂移。

固定增益递归最小二乘法是一种特殊情形。

如果政府的先验为 $V = \varepsilon^2 V^*$，其中 $V^* \equiv \sigma^2 M(\bar\gamma)^{-1}$，且 $\sigma = \sigma_1$，那么 {eq}`pp_riccati` 给出 $P = \varepsilon M(\bar\gamma)^{-1}$，而 {eq}`pp_kalman` 就变成了 {doc}`phillips_learning` 中增益为 $\varepsilon$ 的固定增益递归最小二乘算法，其中估计的矩矩阵被替换为其极限值。

这种联系是渐近的，而非精确的。

在大样本下两种算法会趋于一致，但它们的瞬态反应有所不同：{cite:t}`SargentWilliams2005`（第 3.4 节）表明，在一次逃逸之后，卡尔曼滤波回到自我确认均衡的速度要比递归最小二乘法快得多（见 {ref}`ppr_ex3`）。

```{code-cell} ipython3
def solve_riccati(V, M, σ):
    "Symmetric positive-definite P solving P M P = σ^{-2} V."
    W = V / σ**2
    Mh = sqrtm(M).real
    Mh_inv = np.linalg.inv(Mh)
    return Mh_inv @ sqrtm(Mh @ W @ Mh).real @ Mh_inv

M_sce = model.M(γ_sce)
V_star = model.σ**2 * np.linalg.inv(M_sce)      # the RLS prior (ε = 1)
P_star = solve_riccati(V_star, M_sce, model.σ)

print("RLS prior gives P = M^{-1}?", np.allclose(P_star, np.linalg.inv(M_sce)))
```

下面的函数模拟了政府使用精确滤波器 {eq}`pp_kalman_exact` 时的经济运行过程。

```{code-cell} ipython3
def kalman_path(model, V, σ, T, seed, γ0=None):
    """
    Simulate T periods in which the government uses the exact Kalman filter
    with prior (V, σ), while the truth uses model.σ1.

    Beliefs start at γ0 (default: the self-confirming equilibrium), with P at
    its Riccati value.  Returns inflation π, unemployment U, systematic
    inflation x, and beliefs γ.
    """
    rng = np.random.default_rng(seed)
    u, σ1, σ2 = model.u, model.σ1, model.σ2
    γ = np.array([2 * u, -1.0]) if γ0 is None else np.array(γ0, dtype=float)
    P = solve_riccati(V, model.M(γ), σ)
    W = V / σ**2
    π, U, x = np.empty(T), np.empty(T), np.empty(T)
    beliefs = np.empty((T, 2))
    for n in range(T):
        beliefs[n] = γ
        x[n] = model.x(γ)
        w1, w2 = rng.standard_normal(2)
        π[n] = x[n] + σ2 * w2
        U[n] = u - σ2 * w2 + σ1 * w1                # truth uses σ1
        Φ = np.array([1.0, π[n]])
        PΦ = P @ Φ
        denom = 1 + Φ @ PΦ
        γ = γ + PΦ / denom * (U[n] - Φ @ γ)
        P = P - np.outer(PΦ, PΦ) / denom + W
    return π, U, x, beliefs
```

## 均值动态与 E-稳定性

与 {doc}`phillips_learning` 一样，信念由均值动态所组织，现在是关于 $(\gamma, P)$ 的*联合*常微分方程，时间以周期计量：

```{math}
:label: pp_ode

\dot\gamma = P\, \bar g(\gamma),
\qquad
\dot P = \sigma^{-2} V - P M(\gamma) P .
```

{eq}`pp_ode` 的一个静止点满足 $\bar g(\gamma) = 0$ 且 $P = $ 里卡蒂解——即一个自证均衡。

局部稳定性取决于自证均衡处 $\bar g$ 的雅可比矩阵。

对于静态模型，这可以求得闭式解，

```{math}
:label: pp_jacobian

\frac{\partial \bar g}{\partial \gamma}(\bar\gamma)
= - \begin{bmatrix} \tfrac12 & u \\[1mm] \tfrac12 u & u^2 + \sigma_2^2 \end{bmatrix} .
```

```{code-cell} ipython3
def jacobian(model, γ, h=1e-6):
    "Numerical Jacobian of g_bar at γ."
    J = np.zeros((2, 2))
    for j in range(2):
        gp, gm = γ.copy(), γ.copy()
        gp[j] += h; gm[j] -= h
        J[:, j] = (model.g_bar(gp) - model.g_bar(gm)) / (2 * h)
    return J

J = jacobian(model, γ_sce)
print("∂g/∂γ at SCE =\n", J.round(3))
```

在常增益递归最小二乘法下，相应微分方程中的矩方阵分块会解耦，稳定性由 $M^{-1}\,\partial\bar g/\partial\gamma$ 的特征值决定。

将 $\bar g(\gamma) = M(\gamma)\,[T(\gamma) - \gamma]$ 写出，其中 $T(\gamma) = M(\gamma)^{-1}\, \mathbb{E}[\Phi U]$ 将政府感知到的菲利普斯曲线映射为其政策所生成的总体回归，可以证明在自证均衡处 $M^{-1}\,\partial\bar g/\partial\gamma = DT - I$。

这些特征值的实部为负，即为 **E-稳定性**条件。

这一思想可追溯到 {cite:t}`DeCanio1979` 和 {cite:t}`Evans1985,Evans1989`；{cite:t}`MarcetSargent1989jet` 和 {cite:t}`Woodford1990` 证明了它决定了最小二乘学习的收敛性，而 {cite:t}`EvansHonkapohja2001` 则将这一理论系统化（{ref}`br_one_recursion`）。

```{code-cell} ipython3
eig_rls = np.linalg.eigvals(np.linalg.inv(M_sce) @ J)
print(f"E-stability eigenvalues (RLS): {eig_rls.round(3)}")
print("both negative ⇒ the SCE is E-stable")
```

两个特征值均为负：它们分别是 $-1$ 和 $-\tfrac12$。

它们正是这一系列讲座中先前遇到的数值在静态模型中的对应版本。

当 $\theta = 1$（如本例）时，$-\tfrac12 = -1/(1+\theta^2)$ 正是 {ref}`pc_mean_dynamics` 中预期通货膨胀均值动态的斜率 $\mathcal M$，而 {ref}`pl_eigen` 在 {doc}`phillips_learning` 的动态模型中再次得到了特征值 $-1$ 和 $-1/(1+\theta^2)$。

在 $\theta = 1$ 时，第二个特征值恰好位于边界 $\mathcal M = -\tfrac12$ 上，而 {doc}`phillips_credibility` 指出这正是递减增益最小二乘法 $\sqrt{t}$ 渐近正态性所需的边界。

该边界关乎收敛速度；而信念多久逃逸一次，则是由下文讨论的大偏差率所决定的。

因此，在递减增益下，最小二乘信念渐近地收敛于纳什自证均衡 {cite}`MarcetSargent1989jet`，不过在有限样本中，增益的下降也会减缓回归速度，使得一次早期的逃逸可能长期滞留（{ref}`pl_decreasing`）。

在常增益下，信念会在自证均衡附近徘徊，并反复从中逃逸。

但化简为 $M^{-1}\,\partial\bar g/\partial\gamma$ 依赖于特殊的递归最小二乘先验。

在*一般*先验 $V$ 下，{eq}`pp_ode` 中的 $P$ 分块**不会**解耦，局部稳定性反而由 $\bar P\, \partial\bar g/\partial\gamma$ 的特征值决定——其中 $\bar P$ 是该先验下的里卡蒂解。

关于参数漂移的先验信念现在会影响自证均衡是否稳定。

## 学习循环

这里是论文最引人注目的结果：某些先验会使自证均衡变得*不稳定*，此时均值动态会出现循环。

正如 {cite:t}`SargentWilliams2005` 所做的那样，我们通过*收紧政府对斜率系数的先验*来说明这一点，从递归最小二乘先验 $V^*$ 出发，用因子 $\lambda \in [0, 1]$ 收缩与斜率相关的项：

```{math}
:label: pp_Vlambda

V(\lambda) = \begin{bmatrix} V^*_{11} & \sqrt\lambda\, V^*_{12} \\ \sqrt\lambda\, V^*_{12} & \lambda\, V^*_{22} \end{bmatrix} .
```

对每个 $\lambda$，我们以 $\varepsilon = 0.05$ 求解里卡蒂方程，并观察 $\bar P(\lambda)\, \partial\bar g/\partial\gamma$ 特征值中最大实部：当该值为正时，自证均衡是不稳定的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 随着斜率先验收紧，自证均衡的稳定性变化
    name: fig-pri-stability
---
def V_tighten_slope(λ, V_star):
    V = V_star.copy()
    V[0, 1] = V[1, 0] = np.sqrt(λ) * V_star[0, 1]
    V[1, 1] = λ * V_star[1, 1]
    return V

ε = 0.05                                   # gain (sets the timescale)

def eig_PJ(V, σ=model.σ):
    "Eigenvalues of P̄ ∂g/∂γ at the SCE for the prior (V, σ)."
    return np.linalg.eigvals(solve_riccati(V, M_sce, σ) @ J)

λ_grid = np.linspace(0.01, 0.999, 200)
max_re = np.array([eig_PJ(ε**2 * V_tighten_slope(λ, V_star)).real.max()
                   for λ in λ_grid])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(λ_grid, max_re)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel(r'prior-tightening parameter $\lambda$')
ax.set_ylabel('max real part of eigenvalue')
plt.show()
```

在一个中间范围的 $\lambda$ 内，最大实部变为*正值*：自证均衡失去稳定性。

该图复现了 {cite:t}`SargentWilliams2005` 的图 4。

让我们找出稳定性发生变化的两个点，并观察那里以及不稳定范围深处的特征值。

```{code-cell} ipython3
def max_real(λ):
    return eig_PJ(ε**2 * V_tighten_slope(λ, V_star)).real.max()

sign_changes = np.where(np.diff(np.sign(max_re)) != 0)[0]
for i in sign_changes:
    λ_c = brentq(max_real, λ_grid[i], λ_grid[i + 1])
    eigs = eig_PJ(ε**2 * V_tighten_slope(λ_c, V_star))
    print(f"stability changes at λ = {λ_c:.3f}; eigenvalues there: {eigs.round(4)}")
print(f"eigenvalues at λ = 0.7: {eig_PJ(ε**2 * V_tighten_slope(0.7, V_star)).round(4)}")
```

在不稳定范围的每一端，都有一对复特征值穿过虚轴。

这正是霍普夫分岔定理所描述的情形（见 {cite:t}`Perko1996`）：当实部越过零时，会从自证均衡分岔出一个稳定的极限循环（{cite:t}`SargentWilliams2005` 的定理 4.1）。

{cite:t}`Bullard1994` 将这种循环称为*学习均衡*：底层模型本身没有循环，是学习产生了循环。

在不稳定范围深处，当 $\lambda = 0.7$ 时，两个特征值均为实数且为正，因此自证均衡是一个不稳定节点。

霍普夫定理描述的是分岔点附近的循环，因此在 $\lambda = 0.7$ 处，我们像 {cite:t}`SargentWilliams2005` 在其图 5 中所做的那样，通过从自证均衡附近的一点出发对 $(\gamma, P)$ 的联合均值动态 {eq}`pp_ode` 进行积分，以数值方式求出该循环。

```{code-cell} ipython3
def mean_dynamics(t, z, V, σ):
    "The joint ODE pp_ode for (γ, P), stacked into one vector."
    γ, P = z[:2], z[2:].reshape(2, 2)
    γ_dot = P @ model.g_bar(γ)
    P_dot = V / σ**2 - P @ model.M(γ) @ P
    return np.concatenate([γ_dot, P_dot.ravel()])

λ = 0.7
V = ε**2 * V_tighten_slope(λ, V_star)
P_bar = solve_riccati(V, M_sce, model.σ)

z0 = np.concatenate([γ_sce + np.array([0.3, 0.05]), P_bar.ravel()])
sol = solve_ivp(mean_dynamics, [0, 3000], z0, args=(V, model.σ),
                max_step=1.0, rtol=1e-9, atol=1e-11)
t_path, a_path, b_path = sol.t, sol.y[0], sol.y[1]
x_path = model.x((a_path, b_path))

# period: time between upward crossings of x through u/2, once the cycle has settled
up = t_path[1:][(x_path[:-1] < model.u / 2) & (x_path[1:] >= model.u / 2)]
period = np.diff(up[up > 1000]).mean()
mask = t_path > t_path[-1] - 2 * period        # the last two cycles

print(f"period of the cycle ≈ {period:.0f} periods")
print(f"along the cycle: intercept in [{a_path[mask].min():.2f}, {a_path[mask].max():.2f}], "
      f"slope in [{b_path[mask].min():.2f}, {b_path[mask].max():.2f}], "
      f"inflation in [{x_path[mask].min():.2f}, {x_path[mask].max():.2f}]")
```

下面展示最近两个循环，分别从时间维度和信念空间来看。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: λ = 0.7 时的均值动态，左图为两个循环中的系数，右图为信念空间中的极限循环
    name: fig-pri-cycle
---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(t_path[mask], a_path[mask], label='intercept', lw=2)
axes[0].plot(t_path[mask], b_path[mask], label='slope', lw=2)
axes[0].set_xlabel('time (periods)')
axes[0].set_ylabel('coefficient')
axes[0].legend()

axes[1].plot(a_path[mask], b_path[mask], lw=2)
axes[1].plot(*γ_sce, 'kx', ms=10, label='SCE')
axes[1].set_xlabel('intercept')
axes[1].set_ylabel('slope')
axes[1].legend()

plt.tight_layout()
plt.show()
```

信念稳定在围绕自证均衡的一个闭合轨道上，这与 {cite:t}`SargentWilliams2005` 的图 5 相吻合。

截距从接近 $2u$ 下降到接近 $u$，而斜率从约 $-1$ 上升到接近 $0$，然后两者又突然弹回。

由于通胀通过最优反应 {eq}`pp_bestresp` 是系数的函数，信念中的循环表现为通胀在纳什和拉姆齐结果之间的循环。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 沿着学习循环，通胀在纳什和拉姆齐结果之间振荡
    name: fig-pri-inflation-cycle
---
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_path[mask], x_path[mask])
ax.axhline(model.u, color='k', ls='--', lw=1, label='Nash')
ax.axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
ax.set_xlabel('time (periods)')
ax.set_ylabel('inflation $x$')
ax.legend()
plt.show()
```

这与 {doc}`phillips_learning` 和 {doc}`phillips_escaping_nash` 中的逃逸是不同的机制。

在那里，自证均衡是稳定的，反通胀需要一连串不寻常的冲击才能将信念推离该均衡；而在这里，均值动态本身就会把信念从纳什状态带向拉姆齐状态，然后再带回来。

当增益缩小时，这种差异会变得最为明显。

随着 $\varepsilon \to 0$，该循环依然存在，其周期与 $1/\varepsilon$ 成正比；而从稳定的自证均衡中逃逸则会变得指数级罕见。

但“罕见”是针对小增益而言的一种说法。

对于基准先验，{cite:t}`SargentWilliams2005` 计算出的大偏差率约为 $\bar S \approx 3 \times 10^{-4}$（见其表 1），因此在本讲座及本系列其他讲座中所使用的 0.02 到 0.05 这一增益范围内，使逃逸变得罕见的因子 $\exp(\bar S/\varepsilon)$ 接近于 1，逃逸其实相当频繁；{ref}`en_simulation` 展示了逃逸时间如何随增益缩小而增长。

在这些增益水平下，从稳定自证均衡中逃逸的频率与本循环的周转时间尺度相当，而 {ref}`ppr_ex4` 表明，一条模拟时间序列几乎无法将这两种情形区分开来。

## 逃逸动态与逃逸方向

对于使自证均衡保持稳定的先验，反通胀就是逃逸事件。

大偏差理论（{ref}`pl_primer`），以诺亚·威廉姆斯博士论文 {cite}`Williams2001` 中发展的形式（后发表为 {cite:t}`Williams2019`），将最可能的逃逸刻画为一个控制问题的解：施加代价最小的信念扰动，将信念推离 $\bar\gamma$ 固定距离。

最小化代价 $\bar S$ 是逃逸随增益收缩而变得罕见的速率：若 $\tau$ 为首次逃逸的时间，则当 $\varepsilon \to 0$ 时，$\varepsilon \log \mathbb{E}\tau \to \bar S$。

*瞬时*逃逸方向有一个非常简洁的刻画：它是信念-新息协方差 $Q(\bar\gamma, \bar P) = \hat V$——即先验本身——的最大特征值对应的特征向量。

```{code-cell} ipython3
w, vecs = np.linalg.eigh(V_star)            # baseline prior V*
v_escape = vecs[:, np.argmax(w)]
v_escape = v_escape / np.linalg.norm(v_escape)

# scale the escape so the slope moves from -1 up to 0
t_star = 1.0 / v_escape[1]
terminal = γ_sce + t_star * v_escape

print(f"escape direction  ≈ {v_escape.round(3)}   (∝ [-u, 1])")
print(f"terminal beliefs  ≈ {terminal.round(2)}   (Belief 2 = [u, 0] = Ramsey)")
```

对于基准先验，逃逸方向近似正比于 $(-u, 1)$，在 $\sigma \to 0$ 的极限下恰好成立，它使信念从纳什 $(2u, -1)$ 移向 $(u, 0)$——即支持拉姆齐结果的信念 2。

因此，逃逸一旦发生，就是朝着零通胀方向的运动。

不同的先验会改变逃逸的*路径*，并改变逃逸的*速率*。

收紧斜率先验可能彻底使均衡失稳（如上面的循环所示）。

收紧截距先验则会使均衡保持稳定，但会使逃逸*慢得多*：在 {cite:t}`SargentWilliams2005` 的表 2 中，它使 $\bar S$ 从 $3.1 \times 10^{-4}$ 升至 $0.28$，并使 $\varepsilon = 0.05$ 时的平均逃逸时间从 23 期增加到 167 期（{ref}`ppr_ex1`）。

然而，在 {cite:t}`SargentWilliams2005` 研究的每一种情形中，逃逸最终都导向零通胀，只是路径不同、所依托的信念也不同（{ref}`ppr_ex2`）。

(ppr_sims)=
## 西姆斯的不收敛现象

我们现在可以解决 {doc}`phillips_two_stories` 中描述的谜题。

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 让其政府对漂移系数使用卡尔曼滤波；西姆斯的政府拟合了一条与我们相同的静态菲利普斯曲线，而钟（Chung）的政府拟合了一个分布滞后模型。

在他们的模拟中，政府从纳什通胀逃逸一次之后，就*停留*在拉姆齐结果附近，似乎无限期地持续下去。

而 {cite:t}`Sargent1999` 和 CWS 则改用常增益递归最小二乘法，他们的政府逃逸后又被一再拉回，产生了 {ref}`olg_phillips` 中的锯齿形走势。

{cite:t}`SargentWilliams2005` 将这一差异归因于政府的先验：其滤波器所假设的漂移协方差 $V$ 和噪声方差 $\sigma^2$。

### 回归误差的方差

到目前为止，按照 CWS 的设定，政府的 $\sigma$ 一直等于真实的 $\sigma_1$。

这是自然的：在自证均衡中，回归 {eq}`pp_belief` 与真实模型 {eq}`pp_truth` 一致，且当 $\sigma = \sigma_1$ 时，政府能正确地将其在失业率中观察到的变异在回归量与冲击之间进行拆分。

西姆斯则将 $\sigma$ 设定为远低于 $\sigma_1$ 的水平，并且没有让系数的漂移收缩至零。

一个对其回归误差赋予过少方差的政府，必须将它所观察到的变异归因于其他因素，而唯一可用的其他因素就是其自身系数的漂移。

里卡蒂方程 {eq}`pp_riccati` 使这一点更加精确：$\bar P$ 满足 $\bar P M \bar P = \sigma^{-2} V$，因此 $\bar P$ 与 $1/\sigma$ 成正比。

因此，将 $\sigma$ 低估某个倍数，会使政府的增益提高相同的倍数，其效果与提高 $\varepsilon$ 完全一致。

```{code-cell} ipython3
Φ_sce = np.array([1.0, model.x(γ_sce)])
P_by_σ = {}
for σ_govt in (model.σ1, model.σ1 / 3):
    P_by_σ[σ_govt] = solve_riccati(ε**2 * V_star, M_sce, σ_govt)
    P = P_by_σ[σ_govt]
    gain = (P @ Φ_sce)[0] / (1 + Φ_sce @ P @ Φ_sce)
    print(f"σ = {σ_govt:.2f}: steady-state weight on a forecast error "
          f"(intercept) = {gain:.4f}")

ratio = P_by_σ[model.σ1 / 3] / P_by_σ[model.σ1]
print(f"P(σ1/3) / P(σ1), element by element: {ratio.round(6).ravel()}")
```

将 $\sigma$ 削减到 $\sigma_1$ 的三分之一，恰好使 $\bar P$ 增大三倍。

滤波器对每个预测误差赋予的权重也几乎同等幅度地增大：在此增益水平下，{eq}`pp_kalman_exact` 分母中的 $1$ 使得这一比值略低于三，而随着 $\varepsilon$ 收缩，该比值会趋于三。

在小增益情形下，增益变大会改变学习的速度，但不会改变均值动态与噪声之间的局部平衡：{cite:t}`SargentWilliams2005`（第 5 节）表明，向自证均衡的拉力的局部强度（以单位标准差的漂移量衡量）并不依赖于先验。

而大增益可能改变的是均值动态是否还能描述这些信念，因为均值动态本身只是一种小增益近似。

### 重现西姆斯的例子

{cite:t}`SargentWilliams2005` 模拟了一个与西姆斯的某个基准情形类似的设定，即他的图 5（Chart 5），用我们的记号表示为 $u = 6$、$\sigma_1 = 0.5$、$\sigma_2 = 0.4$、$V = 0.01\, I$、$\sigma = 0.1$，因此 $\sigma_1$ 是 $\sigma$ 的五倍；他们将其视为 $\varepsilon = 1$ 的基准情形。

我们将政府的初始状态设定在自证均衡 $(2u, -1) = (12, -1)$ 处，$P$ 取其里卡蒂方程的解，并像西姆斯以及萨金特—威廉姆斯那样运行 10,000 个周期。

我们将其与 $\sigma = \sigma_1$ 的变体进行比较，同时保持 $V$ 不变。

这一变体使得 $\sigma^{-2} V = 0.04\, I$，与在 $\sigma = 0.1$ 的情况下设定 $\varepsilon = 0.2$ 来缩小漂移等价，因此两者给出的路径*完全相同*，正如 {cite:t}`SargentWilliams2005` 所指出的那样。

我们还需要一种方法来统计沿一条通胀路径发生的逃逸和回归次数。

```{code-cell} ipython3
def escapes_and_returns(π, lo, hi, k=50):
    """
    Count escapes and returns along a path of inflation that starts near Nash.

    An escape is a fall of the k-period moving average of inflation below lo;
    a return is a later rise back above hi.  Also returns the period by which
    the first escape is complete (None if there is none).
    """
    ma = np.convolve(π, np.ones(k) / k, mode='valid')
    near_nash, n_esc, n_ret, first = True, 0, 0, None
    for n, m in enumerate(ma):
        if near_nash and m < lo:
            near_nash, n_esc = False, n_esc + 1
            if first is None:
                first = n + k - 1
        elif not near_nash and m > hi:
            near_nash, n_ret = True, n_ret + 1
    return n_esc, n_ret, first
```

现在我们分别模拟西姆斯设定以及该变体下的十条路径，当通胀的 50 期平均值降至 $u/3 = 2$ 以下时记为一次逃逸，回升至 $5u/6 = 5$ 以上时记为一次回归。

```{code-cell} ipython3
sims = StaticPhillips(u=6.0, σ1=0.5, σ2=0.4)
V_sims, σ_sims = 0.01 * np.eye(2), 0.1
T_sims, seeds = 10_000, range(10)
lo, hi = sims.u / 3, 5 * sims.u / 6

runs = {s: kalman_path(sims, V_sims, σ_sims, T_sims, s) for s in seeds}
runs_σ1 = {s: kalman_path(sims, V_sims, sims.σ1, T_sims, s) for s in seeds}

same = all(np.allclose(kalman_path(sims, 0.04 * V_sims, σ_sims, T_sims, s)[0],
                       runs_σ1[s][0]) for s in (0, 1))
print(f"σ = σ1 and ε = 0.2 give identical paths: {same}\n")

print(f"{'':6}{'Sims: σ = 0.1':^42}| {'σ = σ1':^24}")
print(f"{'seed':>4} {'escapes':>8} {'by period':>10} {'returns':>8} "
      f"{'mean π after':>13} | {'escapes':>8} {'mean π':>8}")
first_escape = {}
for s in seeds:
    π = runs[s][0]
    n_esc, n_ret, first = escapes_and_returns(π, lo, hi)
    first_escape[s] = first
    after = π[first:].mean() if first is not None else np.nan
    n_esc1 = escapes_and_returns(runs_σ1[s][0], lo, hi)[0]
    print(f"{s:>4} {n_esc:>8} {str(first):>10} {n_ret:>8} {after:>13.2f} | "
          f"{n_esc1:>8} {runs_σ1[s][0].mean():>8.2f}")
```

西姆斯设定下的每一条路径都会逃逸，尽管发生的时点各不相同，但没有一条路径回归：每条路径逃逸之后，都会进入一个带噪声的状态，其平均通胀接近 {cite:t}`SargentWilliams2005` 所报告的 $0.61$。

而在 $\sigma = \sigma_1$（等价地，$\varepsilon = 0.2$）的情况下，没有任何路径逃逸，通胀始终停留在自证均衡值 $u = 6$ 附近。

下图展示了逃逸最早发生的那条路径前 1,000 个周期的情况，这类似于 {cite:t}`SargentWilliams2005` 的图 14。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 西姆斯设定与 σ = σ1（等价地 ε = 0.2）下的通胀，前 1,000 个周期
    name: fig-pri-sims
---
s_show = min(seeds, key=lambda s: first_escape[s])

fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.plot(runs[s_show][0][:1000], lw=0.8, label=r"Sims: $\sigma = 0.1$")
ax.plot(runs_σ1[s_show][0][:1000], lw=0.8, ls='--',
        label=r"$\sigma = \sigma_1$ or $\varepsilon = 0.2$")
ax.axhline(sims.u, color='k', ls='--', lw=1)
ax.axhline(0, color='C2', ls=':', lw=1)
ax.set_xlabel('$n$')
ax.set_ylabel('inflation')
ax.legend(frameon=False, loc='lower right')
plt.show()
```

### 为何均值动态不再重新发挥作用

均值动态 {eq}`pp_ode` 在保持信念（从而保持政府政策 $x(\gamma)$）不变的情况下，对数据做多期平均。

当信念变化缓慢时，这是一个很好的近似，但在西姆斯的先验下却并非如此，因为该先验允许信念在相邻周期之间发生很大变化。

此时，政府自身的政策会给通胀注入变异，而这部分变异并不会使失业率发生变动，因为只有意外冲击才会使失业率变动。

将 $U$ 对 $\pi$ 做回归时，就会把这种由政策引致的变异解读为证据，表明菲利普斯曲线几乎是垂直的：在政策方差为 $\mathbb{V}[x]$ 的情况下，总体斜率为 $-\sigma_2^2 / (\sigma_2^2 + \mathbb{V}[x])$，而不是 $-1$。

一条被感知为平坦的曲线支持低通胀，于是低通胀状态便自我维持下去。

让我们在逃逸发生之后，对模拟路径验证这一点。

```{code-cell} ipython3
var_x, slope, theory = [], [], []
for s in seeds:
    π, U, x, _ = runs[s]
    n0 = first_escape[s] + 200                  # well after the escape
    var_x.append(x[n0:].var())
    slope.append(np.cov(U[n0:], π[n0:])[0, 1] / π[n0:].var())
    theory.append(-sims.σ2**2 / (sims.σ2**2 + var_x[-1]))

print(f"variance of systematic inflation x after the escape: "
      f"{min(var_x):.2f} to {max(var_x):.2f}")
print(f"sample slope of U on π after the escape:  {min(slope):.3f} to {max(slope):.3f}")
print(f"-σ2²/(σ2² + var x):                       {min(theory):.3f} to {max(theory):.3f}")
```

政府的信念以及由此而来的政策，在逃逸之后仍持续变化，而其所产生的数据印证了一条斜率接近于零的菲利普斯曲线，即拉姆齐信念。

这正是 {cite:t}`SargentWilliams2005` 所说的关于均值动态与逃逸动态的结果只是"小参数漂移下的近似"的含义：在像西姆斯设定那样大的漂移下，均值动态无法重新发挥作用。

### 常增益与锯齿形走势

现在，让同一经济中的政府改用常增益递归最小二乘法，增益取 {ref}`olg_phillips` 中的 $0.02$（该节经济模型的校准与本讲座其余部分相同）。

```{code-cell} ipython3
def rls_path(model, gain, T, seed):
    "Constant-gain recursive least squares, as in olg_phillips, from the SCE."
    rng = np.random.default_rng(seed)
    u, σ1, σ2 = model.u, model.σ1, model.σ2
    γ = np.array([2 * u, -1.0])
    R = model.M(γ)
    π = np.empty(T)
    for n in range(T):
        w1, w2 = rng.standard_normal(2)
        π[n] = model.x(γ) + σ2 * w2
        U = u - σ2 * w2 + σ1 * w1
        Φ = np.array([1.0, π[n]])
        R = R + gain * (np.outer(Φ, Φ) - R)
        γ = γ + gain * np.linalg.solve(R, Φ * (U - Φ @ γ))
    return π

runs_cg = {s: rls_path(sims, 0.02, T_sims, s) for s in seeds}
counts_cg = [escapes_and_returns(runs_cg[s], lo, hi)[:2] for s in seeds]
print("constant gain 0.02, (escapes, returns) by seed:", counts_cg)
```

在常增益最小二乘法下，每条路径都会逃逸又回归，而且大多数路径发生了多次。

回归次数的统计是保守的：许多回升到纳什水平附近的走势，在 50 期平均值达到 5 之前就又转为新一轮逃逸。

下图比较了两种学习规则下通胀 50 期移动平均值的走势。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 西姆斯卡尔曼滤波下只有一次逃逸，而常增益最小二乘法下逃逸反复出现（50 期移动平均值）
    name: fig-pri-sawtooth
---
def moving_average(π, k=50):
    return np.convolve(π, np.ones(k) / k, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, sharey=True)
for s in seeds:
    axes[0].plot(moving_average(runs[s][0]), lw=0.6)
for s in (0, 1, 2):
    axes[1].plot(moving_average(runs_cg[s]), lw=0.8, label=f'seed {s}')
titles = [r"Sims's Kalman filter ($\sigma = 0.1$), ten seeds",
          'constant-gain least squares (gain 0.02), three seeds']
for ax, title in zip(axes, titles):
    ax.axhline(sims.u, color='k', ls='--', lw=1)
    ax.axhline(0, color='C2', ls=':', lw=1)
    ax.set_ylabel('inflation, 50-period average')
    ax.set_title(title)
axes[1].legend(frameon=False, fontsize=8, ncol=3, loc='lower right')
axes[1].set_xlabel('$n$')
plt.tight_layout()
plt.show()
```

同一个经济可以产生文献中出现的两种历史走势。

在西姆斯的先验下，政府逃逸一次之后便停留在拉姆齐水平附近；而在常增益最小二乘法下，均值动态在每次逃逸之后都会把政府拉回纳什水平，形成 {cite:t}`Sargent1999` 和 CWS 所描述的锯齿形走势。

在西姆斯的先验且 $\sigma = \sigma_1$ 的情况下，政府在 10,000 个周期内从未逃逸。

递减增益则给出了第三条使逃逸得以维持的路径，其原因有所不同：此时最小二乘法会渐近地收敛到自证均衡，但在有限样本中，不断下降的增益会减缓回归的速度（{ref}`pl_decreasing`）。

正如 {cite:t}`SargentWilliams2005` 所言，西姆斯的结果与他们的结果之间的差异可以用两种等价的方式来理解：要么是西姆斯允许了过多的参数漂移以至于无法收敛，要么是他没有让政府对其回归误差归因足够多的变异。

在这里，这两种理解方式是完全等价的，因为滤波器 {eq}`pp_kalman_exact` 仅通过 $\varepsilon / \sigma$ 依赖于 $\varepsilon$ 和 $\sigma$；{ref}`ppr_ex6` 给出了这一点在计量经济学上的含义。

## 结论

自由参数是危险的，正如阿瑟·戈德伯格和罗伯特·卢卡斯所警告的那样。

本讲座中的政府在其信念漂移的协方差矩阵 $V$ 中携带着自由参数。

但这些参数换来了一些东西。

后续的实证研究 {cite:t}`SargentWilliamsZha2006` 赋予了政府本讲座中的卡尔曼滤波器，根据战后美国数据估计 $V$，从而反推出一系列政策制定者关于菲利普斯曲线不断演变的主观模型，使美国通胀实际的兴衰历程得到合理解释——这正是 {doc}`phillips_two_stories` 中平反故事的量化版本。

不过，其作者也给出了一个审慎的结论：他们的估计"教导我们要弱化均值动态和逃逸动态的实证重要性"，而应将关注点转向冲击对政府信念的短期影响；而且他们估计出的纳什通胀与拉姆齐通胀之间的差距很小，这支持了艾伦·布林德对该差距在政策上是否真正重要所持的怀疑态度。

关于简化形式系数与冲击波动率都随时间漂移的相关证据见 {cite:t}`CogleySargent2005`（{doc}`phillips_drifts_volatilities`），该模型将系数漂移的协方差保持为常数。

之所以能够估计出 $V$，是因为具有固定先验的卡尔曼滤波器的增益永不消失，因而漂移参数在整个样本中始终是可识别的；{ref}`pbr_compliment` 讲述了计量经济学家是如何利用这一点的，而 {ref}`olg_ex7` 和 {ref}`ppr_ex6` 则给出了 {cite:t}`Chung1990` 最早解决的估计问题的简化版本。

### 估计出的信念是否现实？

那一实证上的成功也伴随着一个挑战，而这个挑战其实是关于*先验*本身的挑战。

为了拟合数据，{cite:t}`SargentWilliamsZha2006` 估计出一个很大的漂移协方差 $V$——这意味着一个对新数据如此开放的政府，其对货币传导机制的信念每月都会大幅摇摆。

{cite:t}`Primiceri2006`、{cite:t}`Sims2007`，以及坦率地说，{cite:t}`Sargent2008` 本人在其学会主席致辞中，都提出了异议：这种剧烈波动的信念是*不现实的*——由此推算出的政府对失业率的预测很差，持有的观点也是任何真实的中央银行都不会持有的。

{cite:t}`CarboniEllison2009` 通过用美联储实际产生的数据来约束先验，回应了这一异议。

他们在要求模型的信念必须能再现美联储《绿皮书》中公布的失业率预测这一约束下重新估计了模型——这为学习模型施加了一种理性预期计量经济学中常见的跨方程约束。

施加这一约束后，估计出的常数项漂移方差缩小了大约五十倍，从 30.57 降到 0.63，消除了剧烈的信念波动，但低频的"征服"故事依然完好无损：一种*稳定*演化的美联储信念，仍然能够解释通胀的兴衰。

他们得出结论，那些剧烈波动的信念"似乎仅在对高频通胀波动进行'过拟合'时才起作用"，而这些高频波动正是 {cite:t}`Sargent1999` 从未打算解释的内容。

受约束模型所感知的权衡关系峰值仅约为无约束模型的十分之一，因此《绿皮书》提供的证据更支持 {cite:t}`Sargent1999` 的论述，而非 {cite:t}`SargentWilliamsZha2006` 的估计结果。

本讲座的寓意很直接：先验 $V$ 并非一个可以自由最大化的讨厌参数——它是一个经济对象，用关于政策制定者实际信念的独立证据将其确定下来，才使平反故事变得可信。

更广泛的信息是：一个适应性政府*如何*学习——它带入自身信念漂移中的先验——并非一个技术细节。

它决定了经济是在纳什状态附近徘徊并伴有反复逃逸、在纳什和拉姆齐之间循环，还是逃逸到拉姆齐状态并停留在那里。

下一讲 {doc}`phillips_policy_evaluation` 将把这些适应性模型用于实际数据，正如 {cite:t}`Sargent1999` 第九章在 {cite:t}`Chung1990` 之后所做的那样，通过在美国数据上进行最大似然估计来实现。

{doc}`phillips_drifts_volatilities` 和 {doc}`phillips_drifts_volatilities_2` 随后将一个带漂移系数、随机波动率的向量自回归模型拟合到战后数据，探究大通胀在多大程度上源于信念漂移，又在多大程度上仅仅是运气不佳；而 {doc}`phillips_lost_conquest` 则通过将这些工具带入当下，用以解释美联储对 2020 年代通胀的应对，为本系列画上句点。

## 练习

```{exercise-start}
:label: ppr_ex1
```

上文的学习循环收紧了对*斜率*系数的先验。

改为收紧对*截距*系数的先验——使用

$$
V(\lambda) = \begin{bmatrix} \lambda\, V^*_{11} & \sqrt\lambda\, V^*_{12} \\ \sqrt\lambda\, V^*_{12} & V^*_{22} \end{bmatrix}
$$

——这*不会*使自证均衡失稳。

1. 请通过绘制该截距收紧族中 $\bar P(\lambda)\,\partial\bar g/\partial\gamma$ 特征值最大实部关于 $\lambda$ 的图像来验证这一点，并确认其始终为负。
1. {cite:t}`SargentWilliams2005` 发现，这种先验仍会改变逃逸的*速度*。设 $\varepsilon = 0.05$，从自证均衡出发，在基准先验和他们截距收紧的先验（$\lambda = 0.63$）下各模拟 100 条路径。对每条路径，记录信念首次偏离 $\bar\gamma$ 超过 $\sqrt{26}$（即从自证均衡到信念 2 的距离）的时期。在更紧的先验下，逃逸是更快还是更慢？

```{exercise-end}
```

```{solution-start} ppr_ex1
:class: dropdown
```

下面是稳定性计算。

```{code-cell} ipython3
def V_tighten_intercept(λ, V_star):
    V = V_star.copy()
    V[0, 0] = λ * V_star[0, 0]
    V[0, 1] = V[1, 0] = np.sqrt(λ) * V_star[0, 1]
    return V

max_re_int = [eig_PJ(ε**2 * V_tighten_intercept(λ, V_star)).real.max()
              for λ in λ_grid]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(λ_grid, max_re_int, label='tighten intercept')
ax.plot(λ_grid, max_re, ls='--', label='tighten slope (for comparison)')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('max real part of eigenvalue')
ax.set_title('Stability under a tighter intercept prior')
ax.legend()
plt.show()

print(f"largest value over the grid: {max(max_re_int):.2e}")
```

收紧截距先验使最大实部对于所有 $\lambda$ 都保持为负：自证均衡保持稳定，因此不存在学习循环。

现在来看逃逸时间。

下面的函数会在信念一离开该球体时立即停止每条路径。

```{code-cell} ipython3
def first_exit(model, V, σ, radius, seed, T_max=50_000):
    "First period in which |γ - γ_sce| > radius under the exact Kalman filter."
    rng = np.random.default_rng(seed)
    u, σ1, σ2 = model.u, model.σ1, model.σ2
    γ_bar = np.array([2 * u, -1.0])
    γ, P, W = γ_bar.copy(), solve_riccati(V, model.M(γ_bar), σ), V / σ**2
    for n in range(T_max):
        w1, w2 = rng.standard_normal(2)
        π = model.x(γ) + σ2 * w2
        U = u - σ2 * w2 + σ1 * w1
        Φ = np.array([1.0, π])
        PΦ = P @ Φ
        denom = 1 + Φ @ PΦ
        γ = γ + PΦ / denom * (U - Φ @ γ)
        P = P - np.outer(PΦ, PΦ) / denom + W
        if np.linalg.norm(γ - γ_bar) > radius:
            return n + 1
    return np.nan

radius = np.sqrt(26)
for name, V_hat in [("baseline V*", V_star),
                    ("tighter intercept, λ = 0.63", V_tighten_intercept(0.63, V_star))]:
    times = np.array([first_exit(model, ε**2 * V_hat, model.σ, radius, s)
                      for s in range(100)])
    print(f"{name:28s}: mean escape time {np.nanmean(times):6.1f} periods "
          f"(median {np.nanmedian(times):.0f}, censored {np.isnan(times).sum()})")
```

在更紧的截距先验下，逃逸要*慢得多*。

基准情形的时间接近 {cite:t}`SargentWilliams2005` 在表 2 中在相同增益下报告的 22.9 个时期；而截距收紧情形的时间与他们的 167.3 个时期数量级相当，是基准情形的许多倍。

更确信自己的截距没有漂移的政府，会对那些本应推动其信念沿逃逸路径前进的预测误差赋予更小的权重，而 {cite:t}`SargentWilliams2005`（第 6.3 节）表明，使 $\bar P^{-1}$ 更大的先验会提高决定逃逸罕见程度的速率 $\bar S$。

因此这种先验大幅改变了逃逸动态，但并没有推翻均值动态。

```{solution-end}
```

```{exercise-start}
:label: ppr_ex2
```

瞬时逃逸方向是先验协方差 $\hat V$ 的主特征向量。

1. 对于基准先验 $V^*$ 以及 {ref}`ppr_ex1` 中 $\lambda = 0.63$ 的截距收紧先验（即 {cite:t}`SargentWilliams2005` 图 9 所对应的情形），计算逃逸方向，以及沿该方向直线前进直到斜率达到 $0$ 所隐含的终端信念。为什么斜率收紧先验 $V(0.5)$ 不适合用于此比较？
1. 设 $\varepsilon = 0.05$，分别在两种先验下各模拟五次逃逸，当斜率首次达到 $0$ 时停止每条路径，并在（截距，斜率）平面上绘制信念路径及两个瞬时方向。
1. 逃逸最终落在何处，瞬时方向对其路径的预测效果如何？

```{exercise-end}
```

```{solution-start} ppr_ex2
:class: dropdown
```

斜率收紧先验 $V(0.5)$ 位于 {numref}`fig-pri-stability` 的不稳定范围内，因此不存在可供逃逸的稳定自证均衡。

截距收紧先验则保持了均衡的稳定性（见 {ref}`ppr_ex1`）。

```{code-cell} ipython3
def escape_direction(V):
    "Dominant eigenvector of V, oriented toward a higher slope, and its terminal point."
    w, vecs = np.linalg.eigh(V)
    v = vecs[:, np.argmax(w)]
    v = v / np.linalg.norm(v)
    if v[1] < 0:
        v = -v
    return v, γ_sce + v / v[1]

priors = {"baseline": V_star,
          "tighter intercept (λ = 0.63)": V_tighten_intercept(0.63, V_star)}
for name, V_hat in priors.items():
    v, term = escape_direction(V_hat)
    stable = eig_PJ(ε**2 * V_hat).real.max() < 0
    print(f"{name:30s}: direction {v.round(3)}, straight-line terminal "
          f"{term.round(2)}, SCE stable: {stable}")
```

两个方向仅相差几度，但由于逃逸覆盖的距离很长，直线终端截距的差异约为一个单位。

现在来看模拟的逃逸。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
for (name, V_hat), color in zip(priors.items(), ('C0', 'C1')):
    v, term = escape_direction(V_hat)
    ax.plot([γ_sce[0], term[0]], [γ_sce[1], term[1]], color=color, lw=2,
            label=f'{name}: instantaneous direction')
    ends = []
    for s in range(5):
        _, _, _, beliefs = kalman_path(model, ε**2 * V_hat, model.σ, 5_000, s)
        n_end = np.argmax(beliefs[:, 1] >= 0)
        ax.plot(beliefs[:n_end + 1, 0], beliefs[:n_end + 1, 1], color=color,
                lw=0.6, alpha=0.7)
        ends.append((n_end, beliefs[n_end, 0]))
    print(f"{name:30s}: (period, intercept) when the slope reaches 0: "
          f"{[(int(n), round(a, 2)) for n, a in ends]}")
ax.plot(*γ_sce, 'kx', ms=10)
ax.set_xlabel('intercept')
ax.set_ylabel('slope')
ax.legend(frameon=False, fontsize=9)
plt.show()
```

在基准先验下，模拟的逃逸路径紧贴直线，正如 {cite:t}`SargentWilliams2005` 图 7 所示，并终止于信念 2，即 $(u, 0)$ 附近。

在更紧的截距先验下，逃逸开始得要晚得多：信念先在自证均衡附近游走，一旦开始逃逸，便沿着与基准情形相近的直线前进，而非沿着自身的瞬时方向，最终在斜率为 $0$ 处终止，其截距接近基准情形而非直线终端点。

因此瞬时方向只有在基准情形下才能较好地预测路径，因为随着 $\sigma \to 0$，均值动态本身会在逃逸开始后持续沿该方向推动信念（见 {cite:t}`SargentWilliams2005` 第 6.2 节）；而对于其他先验，整个动态控制问题都起作用。

在这两种情形下，逃逸最终都停在斜率为 $0$ 处，此时政府认为不存在可利用的权衡关系，{eq}`pp_bestresp` 无论截距如何，都会将通胀设为零。

目的地是稳健的；而路径及其耗时取决于先验的形状。

```{solution-end}
```

```{exercise-start}
:label: ppr_ex3
```

{cite:t}`SargentWilliams2005`（第 3.4 节）强调，常增益递归最小二乘法只有在*渐近意义*下才是 $V = \varepsilon^2 V^*$ 的卡尔曼滤波的特例。

设 $\varepsilon = 0.05$，$V = \varepsilon^2 V^*$，将两种算法的均值动态都从拉姆齐信念 $\gamma_0 = (u, 0)$ 出发，卡尔曼滤波的 $P$ 取其对应 $M(\gamma_0)$ 的里卡蒂值，递归最小二乘的矩阵取 $R = M(\gamma_0)$。

递归最小二乘的均值动态为

$$
\dot\gamma = \varepsilon R^{-1} \bar g(\gamma),
\qquad
\dot R = \varepsilon \left[ M(\gamma) - R \right] .
$$

对两个系统积分 800 个时期，并绘制截距、斜率以及通胀 $x(\gamma)$ 的图像。

哪种算法更快地回到自证均衡，二者是否终止于同一处？

```{exercise-end}
```

```{solution-start} ppr_ex3
:class: dropdown
```

卡尔曼滤波均值动态即讲座中的函数 `mean_dynamics`。

```{code-cell} ipython3
def rls_mean_dynamics(t, z):
    "Mean dynamics of constant-gain RLS for (γ, R), stacked into one vector."
    γ, R = z[:2], z[2:].reshape(2, 2)
    γ_dot = ε * np.linalg.solve(R, model.g_bar(γ))
    R_dot = ε * (model.M(γ) - R)
    return np.concatenate([γ_dot, R_dot.ravel()])

γ0 = np.array([model.u, 0.0])                   # Belief 2 (Ramsey)
M0 = model.M(γ0)
V = ε**2 * V_star
t_eval = np.linspace(0, 800, 801)

kal = solve_ivp(mean_dynamics, [0, 800],
                np.concatenate([γ0, solve_riccati(V, M0, model.σ).ravel()]),
                args=(V, model.σ), t_eval=t_eval, rtol=1e-8, atol=1e-10)
rls = solve_ivp(rls_mean_dynamics, [0, 800], np.concatenate([γ0, M0.ravel()]),
                t_eval=t_eval, rtol=1e-8, atol=1e-10)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for out, label, ls in [(kal, 'Kalman filter', '-'), (rls, 'constant-gain RLS', '--')]:
    a, b = out.y[0], out.y[1]
    for ax, series in zip(axes, (a, b, model.x((a, b)))):
        ax.plot(out.t, series, ls=ls, label=label)
for ax, title in zip(axes, ('intercept', 'slope', 'inflation $x$')):
    ax.set_title(title)
    ax.set_xlabel('time (periods)')
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

for out, label in [(kal, 'Kalman filter'), (rls, 'constant-gain RLS')]:
    x = model.x((out.y[0], out.y[1]))
    print(f"{label:18s}: x first exceeds 0.8u at period {out.t[np.argmax(x > 0.8 * model.u)]:.0f}; "
          f"beliefs at period 800 = {out.y[:2, -1].round(3)}")
```

两种算法都终止于自证均衡 $(2u, -1)$，但卡尔曼滤波要快得多，其斜率表现为突然反转，而非逐渐变化，正如 {cite:t}`SargentWilliams2005` 图 1 和图 2 所示。

这一差异来自两种增益矩阵对数据矩的变化作出反应的方式不同：在卡尔曼滤波中，$P$ 通过 $P M(\gamma) P$ 这一项作出反应，而递归最小二乘的矩阵 $R$ 只能滞后跟随 $M(\gamma)$。

由于两种算法共享同一个驻点及其局部稳定性，它们在渐近意义上是一致的；而它们的瞬态响应——这在每次逃逸之后都很重要——则有所不同。

```{solution-end}
```

```{exercise-start}
:label: ppr_ex4
```

本练习将学习循环与反复发生的逃逸进行比较，正如 {cite:t}`SargentWilliams2005` 在其图 6 中所做的那样。

1. 设 $\varepsilon = 0.05$，从自证均衡出发，在斜率收紧先验 $V(0.7)$ 下模拟精确卡尔曼滤波 2,000 个时期，并将通胀与讲座中计算的均值动态路径 $x(t)$ 一并绘制。在基准先验 $V^*$ 下重复该模拟，此时自证均衡是稳定的。
1. 对 $\varepsilon \in \{0.05, 0.02, 0.005\}$ 以及两种先验，统计每 100 个 $\varepsilon n$ 单位内发生的通缩次数（20 期平均通胀在曾高于 4 之后又跌破 1），并对三个随机种子取平均。
1. 在这些增益下，学习循环与反复发生的逃逸有何区别？当 $\varepsilon \to 0$ 时，又该如何区分二者？

```{exercise-end}
```

```{solution-start} ppr_ex4
:class: dropdown
```

下面是两种模拟。

```{code-cell} ipython3
priors = {"slope-tightened V(0.7): unstable SCE": V_tighten_slope(0.7, V_star),
          "baseline V*: stable SCE": V_star}

fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, sharey=True)
for ax, (name, V_hat) in zip(axes, priors.items()):
    π = kalman_path(model, ε**2 * V_hat, model.σ, 2_000, 0)[0]
    ax.plot(π, lw=0.5, label='simulated inflation')
    ax.set_title(name)
    ax.set_ylabel('inflation')
keep = t_path <= 2_000
axes[0].plot(t_path[keep], x_path[keep], lw=2, label='mean dynamics')
axes[0].legend(frameon=False, loc='lower right', fontsize=8)
axes[1].set_xlabel('$n$')
plt.tight_layout()
plt.show()
```

两个模拟序列都表现出反复发生的通缩。

在斜率收紧先验下，模拟紧密跟随均值动态的确定性循环；在基准先验下，这些通缩是逃逸，但其发生频率同样高。

现在来看计数结果。

```{code-cell} ipython3
def disinflations_per_100(V_hat, ε_, seeds=range(3)):
    "Disinflations per 100 units of εn under the exact Kalman filter."
    T = int(100 / ε_)
    counts = [escapes_and_returns(kalman_path(model, ε_**2 * V_hat, model.σ, T, s)[0],
                                  lo=1.0, hi=4.0, k=20)[0] for s in seeds]
    return np.mean(counts)

print(f"{'ε':>6} | {'V(0.7): cycle':>14} | {'V*: escapes':>12}")
for ε_ in (0.05, 0.02, 0.005):
    print(f"{ε_:>6} | {disinflations_per_100(priors['slope-tightened V(0.7): unstable SCE'], ε_):>14.1f}"
          f" | {disinflations_per_100(V_star, ε_):>12.1f}")
```

在表中每一个增益下，两种先验产生的每单位 $\varepsilon n$ 的通缩速率都相近，因此在这些增益下，稳定先验的逃逸并不罕见。

对于循环而言，这是意料之中的：其周期与 $1/\varepsilon$ 成正比，约为 20 个 $\varepsilon n$ 单位。

对于稳定先验而言，这反映了 {cite:t}`SargentWilliams2005` 计算的较小的大偏差速率 $\bar S \approx 3 \times 10^{-4}$：平均逃逸时间只有在 $\varepsilon$ 相对于 $\bar S$ 较小时才会以 $\exp(\bar S/\varepsilon)$ 的方式增长，在此之前，其规模大致按 $1/\varepsilon$ 增长，与循环的周期一样。

稳定先验每单位 $\varepsilon n$ 的速率确实随 $\varepsilon$ 缩小而下降，但下降得很慢。

因此，在这些增益下，仅凭模拟时间序列很难区分学习循环与反复发生的逃逸。

二者的区别是渐近意义上的：当 $\varepsilon \to 0$ 时，循环每单位 $\varepsilon n$ 的速率保持不变，而稳定先验的逃逸最终会变得指数级地罕见。

```{solution-end}
```

```{exercise-start}
:label: ppr_ex5
```

本练习对 {ref}`ppr_sims` 中西姆斯例子的复现做出一些变化。

1. {cite:t}`SargentWilliams2005` 图 14 中的时间序列似乎起始于远离自证均衡的低通胀处。请改为从拉姆齐信念 $(u, 0)$ 出发启动西姆斯的设定以及 $\sigma = \sigma_1$ 的变体，令 $P$ 取其里卡蒂值，并报告 10,000 个时期中前 1,000 期与后 2,000 期在五个随机种子下的平均通胀。
1. 从自证均衡出发，将西姆斯的漂移协方差按 $V = \varepsilon^2 \times 0.01\, I$ 缩放，其中 $\varepsilon \in \{0.2, 0.3, 0.4, 0.6\}$，保持 $\sigma = 0.1$。对每个 $\varepsilon$，如讲座中那样，在十个随机种子、每条 10,000 个时期的路径下统计逃逸与回归的次数。随着漂移增大，历史路径如何变化？

```{exercise-end}
```

```{solution-start} ppr_ex5
:class: dropdown
```

首先是从拉姆齐信念出发的路径。

```{code-cell} ipython3
γ_ramsey = np.array([sims.u, 0.0])
for name, σ_govt in [("Sims, σ = 0.1", σ_sims), ("σ = σ1", sims.σ1)]:
    early, late = [], []
    for s in range(5):
        π = kalman_path(sims, V_sims, σ_govt, T_sims, s, γ0=γ_ramsey)[0]
        early.append(π[:1000].mean())
        late.append(π[-2000:].mean())
    print(f"{name:14s}: mean π, first 1,000 periods {np.round(early, 2)}")
    print(f"{'':14s}  mean π, last 2,000 periods  {np.round(late, 2)}")
```

从低通胀出发时，西姆斯的政府从未收敛到自证均衡：通胀始终保持在低位。

在 $\sigma = \sigma_1$ 的情形下，通胀在前一千个时期内向自证均衡值 $u = 6$ 攀升，随后保持在其附近，这正是 {cite:t}`SargentWilliams2005` 图 14 所展示的内容。

现在来看对漂移幅度的扫描。

```{code-cell} ipython3
for ε_ in (0.2, 0.3, 0.4, 0.6):
    counts = [escapes_and_returns(
                  kalman_path(sims, ε_**2 * V_sims, σ_sims, T_sims, s)[0], lo, hi)[:2]
              for s in seeds]
    n_escape = sum(c[0] > 0 for c in counts)
    n_return = sum(c[1] > 0 for c in counts)
    print(f"ε = {ε_}: paths that escape {n_escape}/10, paths that return {n_return}/10, "
          f"(escapes, returns) by seed {counts}")
```

历史路径随漂移幅度的变化而变化。

在漂移最小时，没有任何路径在 10,000 个时期内发生逃逸：均值动态使信念保持在自证均衡附近。

当漂移稍大一些时，部分路径会逃逸，其中一些会返回，呈现出锯齿形的雏形。

当漂移更大时，几乎每条路径都只逃逸一次而从不返回，正如讲座中西姆斯自己所取的情形 $\varepsilon = 1$。

文献中出现的这两类历史，其实是由政府相对于其回归所归因的噪声所设想的漂移大小所索引的一个连续统一体的两端。

```{solution-end}
```

```{exercise-start}
:label: ppr_ex6
```

{ref}`olg_ex7` 曾要求一位计量经济学家根据常增益政府的政策所生成的数据来估计该政府的增益。

此处政府运行本讲座中精确的卡尔曼滤波，先验为 $V = \varepsilon^2 V^*$，$\varepsilon = 0.02$，$\sigma = \sigma_1$。

从自证均衡出发模拟 5,000 个时期。

假设一位计量经济学家对该经济及政府的一切都了如指掌，唯独不知道 $\varepsilon$ 与政府的 $\sigma$。

1. 证明候选参数对 $(\varepsilon, \sigma)$ 与 $(c\varepsilon, c\sigma)$ 对任意 $c > 0$ 都给出完全相同的目标历史 $x_n$。
1. 在一张 $(\varepsilon, \sigma)$ 网格上计算通胀的高斯对数似然 $\ell(\varepsilon, \sigma) = -\sum_n (\pi_n - x_n(\varepsilon, \sigma))^2 / (2 \sigma_2^2)$（可相差一个常数），并绘制其等高线图。
1. 数据能够识别出什么？请将你的答案与 {ref}`ppr_sims` 中西姆斯例子的两种等价解释联系起来。

```{exercise-end}
```

```{solution-start} ppr_ex6
:class: dropdown
```

政府的信念是数据与滤波器参数的确定性函数，因此计量经济学家可以对每一个候选参数对重新运行滤波，正如 {ref}`olg_ex7` 中所做的那样。

下面的函数可以一次性对多个候选参数完成这一计算。

```{code-cell} ipython3
def reconstruct_targets(π, U, ε_vals, σ_vals):
    """
    Targets x_n implied by the exact Kalman filter with prior (ε² V*, σ),
    reconstructed from data (π, U), for each candidate pair (ε, σ).
    """
    K = len(ε_vals)
    γ = np.tile(γ_sce, (K, 1))
    P = np.array([solve_riccati(e**2 * V_star, M_sce, s)
                  for e, s in zip(ε_vals, σ_vals)])
    W = np.array([e**2 * V_star / s**2 for e, s in zip(ε_vals, σ_vals)])
    targets = np.empty((len(π), K))
    for n in range(len(π)):
        targets[n] = -γ[:, 0] * γ[:, 1] / (1 + γ[:, 1]**2)
        Φ = np.array([1.0, π[n]])
        PΦ = P @ Φ
        denom = 1 + PΦ @ Φ
        γ = γ + PΦ / denom[:, None] * (U[n] - γ @ Φ)[:, None]
        P = P - PΦ[:, :, None] * PΦ[:, None, :] / denom[:, None, None] + W
    return targets

ε_true, σ_true = 0.02, model.σ1
π_data, U_data, x_data, _ = kalman_path(model, ε_true**2 * V_star, σ_true, 5_000, 1)

check = reconstruct_targets(π_data, U_data, [ε_true, 2 * ε_true, ε_true],
                            [σ_true, 2 * σ_true, σ_true / 2])
print(f"largest error at the true (ε, σ):          {np.max(np.abs(check[:, 0] - x_data)):.1e}")
print(f"largest difference, (2ε, 2σ) vs (ε, σ):    {np.max(np.abs(check[:, 1] - check[:, 0])):.1e}")
print(f"largest difference, (ε, σ/2) vs (ε, σ):    {np.max(np.abs(check[:, 2] - check[:, 0])):.1e}")
```

将 $\varepsilon$ 与 $\sigma$ 同时加倍不会改变目标，而仅改变 $\sigma$ 则会。

原因在于，滤波公式 {eq}`pp_kalman_exact` 仅通过 $\sigma^{-2} V = (\varepsilon/\sigma)^2 V^*$ 来使用 $V$ 与 $\sigma$，且 $P_0$ 的里卡蒂值也以同样的方式依赖于二者。

现在来看网格上的似然。

```{code-cell} ipython3
ε_grid = np.linspace(0.005, 0.04, 22)
σ_grid = np.linspace(0.1, 0.6, 22)
EE, SS = np.meshgrid(ε_grid, σ_grid)
targets = reconstruct_targets(π_data, U_data, EE.ravel(), SS.ravel())
loglik = -np.sum((π_data[:, None] - targets)**2, axis=0) / (2 * model.σ2**2)
loglik = (loglik - loglik.max()).reshape(EE.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cs = axes[0].contour(SS, EE, loglik, levels=[-3000, -1000, -300, -100])
axes[0].clabel(cs, fontsize=8)
s_line = np.linspace(0.1, 0.6, 2)
axes[0].plot(s_line, (ε_true / σ_true) * s_line, 'k--', lw=1,
             label=r'$\varepsilon / \sigma$ = true value')
axes[0].plot(σ_true, ε_true, 'k*', ms=10, label='truth')
axes[0].set_xlabel(r'government $\sigma$')
axes[0].set_ylabel(r'drift scale $\varepsilon$')
axes[0].set_title('contours of the log likelihood')
axes[0].legend(frameon=False, loc='upper left')

axes[1].scatter((EE / SS).ravel(), loglik.ravel(), s=8)
axes[1].axvline(ε_true / σ_true, color='k', ls='--', lw=1)
axes[1].set_xscale('log')
axes[1].set_xlim(0.03, 0.15)
axes[1].set_ylim(-3000, 100)
axes[1].set_xlabel(r'$\varepsilon / \sigma$')
axes[1].set_title(f'all {loglik.size} candidates against their ratio')
plt.tight_layout()
plt.show()

ratio_hat = (EE.ravel() / SS.ravel())[np.argmax(loglik)]
print(f"true ε/σ = {ε_true / σ_true:.4f}; ratio at the grid maximum = {ratio_hat:.4f}")
```

似然的等高线是经过原点的射线，若以 $\varepsilon/\sigma$ 为横轴绘图，所有候选参数的对数似然都落在同一条曲线上：数据确定了 $\varepsilon/\sigma$，即政府的有效增益，但对 $\varepsilon$ 与 $\sigma$ 各自的取值却毫无信息。

这正是西姆斯例子那两种等价解释在计量经济学上的对应：“漂移过大”与“误差方差过小”是对同一学习规则的两种可观测上等价的描述。

要区分二者，需要有关政府自称*有多不确定*的证据，例如其预测方差 $\sigma^2 (1 + \Phi_n^\top P_n \Phi_n)$，而不仅仅是它做了什么或预测了什么。

数据确实能识别的东西，会随着样本的增大而持续被识别，因为滤波器的增益永远不会消失，因而会持续影响信念（见 {ref}`pbr_compliment`）。

```{solution-end}
```
