---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: 线性二次均值场博弈：扩展
  headings:
    Overview: 概述
    What we need from the previous lecture: 我们需要用到上一讲的内容
    State-action cross terms: 状态-行动交叉项
    State-action cross terms::The individual problem: 个体问题
    State-action cross terms::The equilibrium: 均衡
    State-action cross terms::Investment rates and Hayashi adjustment costs: 投资率与 Hayashi 调整成本
    State-action cross terms::When the restriction fails: 当限制条件不成立时
    Forcing processes and pass-through: 强制过程与传导
    Coupling through higher moments: 通过高阶矩的耦合
    Exercises: 练习
    Further reading: 延伸阅读
---

(lq_mean_field_games_extensions)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 线性二次均值场博弈：扩展

```{contents} Contents
:depth: 2
```

## 概述

本讲座延续 {doc}`lq_mean_field_games`，并假定读者已熟悉该讲座的理论——包括将系统拆分为个体动态与总量动态的做法，以及其中的资本积累示例。

在那一讲中，我们研究了一个由连续多个主体组成的经济体，这些主体的二次支付取决于状态与行动的横截面平均值，并发现均衡的总量动态求解的是单主体问题的黎卡提方程，策略互动只改变其中的曲率。

这里我们考察 {cite:t}`AlvarezArgente2026` 提出的三个扩展：

* **状态与行动之间的交叉项。** 当状态与行动之间的混合互动满足某种对称性限制时，等价性结果依然成立，此时混合互动需从自身的交叉项中减去；而当该限制不满足时，唯一性可能失效。
* **强制过程。** 诸如成本这类主体无法控制的状态变量可以被纳入模型，而不改变受控部分的黎卡提方程；一个定价示例表明，企业以不同速度、不同程度地将特异性和共同的成本冲击传导出去。
* **高阶矩。** 当支付取决于横截面分布的高阶矩时，特异性风险会影响总量的持续性。

让我们先导入一些模块：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, expm
```

## 我们需要用到上一讲的内容

全文中，小写字母表示主体自身的状态 $x$ 与行动 $\alpha$，大写字母或花体字母表示它们的横截面平均值 $X$ 和 $\mathcal A$。

在 {doc}`lq_mean_field_games` 中，一个主体的期间收益与状态方程为

$$
-\tfrac12 x^\top Qx - X^\top\Theta_X x - \tfrac12\alpha^\top\Gamma\alpha - \mathcal A^\top\Theta_{\mathcal A}\alpha ,
\qquad
dx = (B\alpha - Ax)dt + \Sigma^{1/2}dW ,
$$

贴现率为 $\rho$。

给定总量路径，该主体的价值函数为 $u(x,t) = \beta_0(t) + \beta_1(t)^\top x + \tfrac12 x^\top\bar\beta_2 x$，其中 $\bar\beta_2$ 求解单主体黎卡提方程，且不涉及任何互动矩阵。

在均衡中，总量状态遵循 $\dot X = \left(B(\Gamma+\Theta_{\mathcal A})^{-1}B^\top P - A\right)X$，其中对称矩阵 $P$ 满足

$$
P B(\Gamma+\Theta_{\mathcal A})^{-1}B^\top P = Q + \Theta_X + \rho P + PA + A^\top P ,
$$

并且闭环矩阵的特征值的实部必须小于 $\rho/2$。

该方程来自一个状态-协态系统，其特征值成对出现，为 $\lambda$ 和 $\rho - \lambda$，因此至多有一半位于 $\rho/2$ 以下；由于 $B(\Gamma + \Theta_{\mathcal A})^{-1}B^\top$ 是正定的，这正是均衡唯一的原因。

一个主体与横截面均值的偏差 $z = x - X$ 遵循 $dz = (B\Gamma^{-1}B^\top\bar\beta_2 - A)z\,dt + \Sigma^{1/2}dW$，其中不出现任何互动矩阵。

因此，关于偏离均值的数据不包含任何有关策略互动的信息——这正是缺失的截距所在。

对所有主体共同的冲击，用一个补偿跳跃过程 $\mathcal J$ 建模，它进入 $X$ 的运动规律，但不改变 $P$。

我们重复使用两个工具：标量情形下总量特征值的闭式解，

$$
\lambda = \frac\rho2 - \sqrt{\left(\frac\rho2 + a\right)^2 + \frac{b^2(q + \theta_X)}{\gamma + \theta_{\mathcal A}}} ,
$$

以及资本积累示例的校准。

在该例中，若行业中其余部分持有资本 $K$，则持有资本 $k$ 的企业获得收益 $\Pi(k, K) = k^{\nu(1-1/\eta)}K^{\nu/\eta}$，支付调整成本 $\psi(i) = \frac\phi2\bar i\left(\frac{i - \bar i}{\bar i}\right)^2$，并以价格 $\mathcal P(I) = (I/\bar i)^{1/\varepsilon_s}$ 购买投资品。

以相对稳态的百分比偏差表示，该模型映射为标量模型，其中 $a = b = \delta$、$q + \theta_X = \frac{\eta-1}{\eta}\nu(1-\nu)$、$\gamma = \phi\,\delta\bar k^{1-\nu}$、$\theta_{\mathcal A} = \delta\bar k^{1-\nu}/\varepsilon_s$，而校准结果给出企业相对于行业均值的偏差半衰期为三年，行业本身的半衰期为五年。

```{code-cell} ipython3
def scalar_λ(ρ, a, b, q, γ, θ_X, θ_A):
    "Closed-form aggregate eigenvalue in the scalar case."
    return ρ/2 - np.sqrt((ρ/2 + a)**2 + b**2*(q + θ_X)/(γ + θ_A))

def half_life(λ):
    "Time for a state with eigenvalue λ < 0 to close half of a gap."
    return np.log(2)/(-λ)

def cap_q(η, ν):
    "Own-state curvature in the capital accumulation example."
    return ν*(η - 1)/η**2*(η*(1 - ν) + ν)

def cap_θ_X(η, ν):
    "State interaction in the capital accumulation example."
    return -(η - 1)/η*ν**2/η

ρ_c, δ_c, η_c, ν_c = 0.05, 0.10, 4.0, 0.7
φ_c, ε_s_c = 1/7, 35/9

def cap_scale(η, ν, ρ=ρ_c, δ=δ_c):
    "δ k̄^{1-ν}, which converts φ and 1/ε_s into γ and θ_A."
    return δ*ν*(η - 1)/(η*(ρ + δ))

γ_c = φ_c*cap_scale(η_c, ν_c)
θ_A_c = cap_scale(η_c, ν_c)/ε_s_c
q_c, θ_X_c = cap_q(η_c, ν_c), cap_θ_X(η_c, ν_c)

λ_c = scalar_λ(ρ_c, δ_c, δ_c, q_c, γ_c, θ_X_c, θ_A_c)          # industry
λ_micro = scalar_λ(ρ_c, δ_c, δ_c, q_c, γ_c, 0.0, 0.0)          # single firm
print(f"γ = {γ_c:.3f}, θ_A = {θ_A_c:.3f}")
print(f"half-lives: firm {half_life(λ_micro):.2f} years, industry {half_life(λ_c):.2f} years")
```

(mfgx_cross_terms)=
## 状态-行动交叉项

{doc}`lq_mean_field_games` 中的收益并非最一般的二次收益形式。

对一般期间收益作二阶近似可得

```{math}
:label: mfgx_return_general
\begin{aligned}
\tilde F(x, X, \alpha, \mathcal A)
= &-\tfrac12 x^\top Q x - X^\top\Theta_X x
   -\tfrac12 \alpha^\top \Gamma \alpha - \mathcal A^\top \Theta_{\mathcal A}\alpha \\
  &+ x^\top C\alpha - X^\top\Theta_{X\alpha}\alpha - x^\top\Theta_{x\mathcal A}\mathcal A \\
  &-\tfrac12 X^\top\Lambda_X X - \tfrac12\mathcal A^\top\Lambda_{\mathcal A}\mathcal A
   - X^\top\Lambda_{X\mathcal A}\mathcal A .
\end{aligned}
```

上一讲自始至终只用到了第一行，以及在研究计划者问题时第三行中仅涉及总量本身的项。

这里我们恢复第二行。

自身交叉项 $C$ 把主体的状态与行动联系起来。

当调整成本不仅取决于行动本身，还取决于状态时，这一项就会出现，例如调整成本取决于投资*率*的情形 {cite}`Hayashi1982,CooperHaltiwanger2006`。

混合互动则出现在：平均状态影响主体行动的边际收益（$\Theta_{X\alpha}$），或平均行动影响主体状态的边际收益（$\Theta_{x\mathcal A}$）的情形下。

### 个体问题

给定总量路径，主体的一阶条件现在变为

$$
\alpha^{*}(x,t) = \Gamma^{-1}\left[B^\top u_x(x,t) + C^\top x - \Theta_{\mathcal A}\mathcal A(t) - \Theta_{X\alpha}^\top X(t)\right] ,
$$

其价值函数的曲率满足

```{math}
:label: mfgx_beta2_cross
(\bar\beta_2 B + C)\Gamma^{-1}(B^\top\bar\beta_2 + C^\top) = Q + \rho\bar\beta_2 + \bar\beta_2 A + A^\top\bar\beta_2 .
```

自身交叉项进入了个体的曲率方程，但与之前一样，任何互动矩阵都不会出现在其中。

对行动完成配方，

$$
-\tfrac12 x^\top Qx - \tfrac12\alpha^\top\Gamma\alpha + x^\top C\alpha
= -\tfrac12 x^\top(Q - C\Gamma^{-1}C^\top)x
  - \tfrac12(\alpha - \Gamma^{-1}C^\top x)^\top\Gamma(\alpha - \Gamma^{-1}C^\top x),
$$

表明 $Q \succ 0$ 的对应条件是 $Q - C\Gamma^{-1}C^\top \succ 0$，我们假设该条件成立。

### 均衡

将一阶条件加总可得

$$
(\Gamma + \Theta_{\mathcal A})\mathcal A = B^\top(\beta_1 + \bar\beta_2 X) + (C - \Theta_{X\alpha})^\top X .
$$

跟踪 $w \equiv \beta_1 + \bar\beta_2 X$ 比较方便，它是均值处主体状态的边际价值。

求导、加总，并利用 {eq}`mfgx_beta2_cross` 可得状态-协态系统

```{math}
:label: mfgx_cross_system
\begin{aligned}
\dot w &= (\rho I + A^\top) w + (Q + \Theta_X) X - (C - \Theta_{x\mathcal A})\mathcal A , \\
\dot X &= B\mathcal A - AX .
\end{aligned}
```

协态方程中出现了 $\Theta_{x\mathcal A}$，它反映了平均行动如何改变主体状态的边际价值。

总量行动方程中出现了 $\Theta_{X\alpha}$，它反映了平均状态如何改变主体行动的边际收益。

这两者未必相等，但当它们相等时，该系统保留了基准情形的哈密顿结构。

```{prf:proposition}
:label: mfgx_prop_cross

假设 $\Theta_{X\alpha} = \Theta_{x\mathcal A}$，记 $\tilde C \equiv C - \Theta_{X\alpha}$。

均衡由满足如下方程的对称矩阵 $P$ 刻画：

$$
(PB + \tilde C)(\Gamma + \Theta_{\mathcal A})^{-1}(B^\top P + \tilde C^\top) = Q + \Theta_X + \rho P + PA + A^\top P ,
$$

总量动态为

$$
\dot X = \left[B(\Gamma+\Theta_{\mathcal A})^{-1}(B^\top P + \tilde C^\top) - A\right] X .
$$

这正是单主体的黎卡提方程，该主体的收益为

$$
-\tfrac12 X^\top(Q + \Theta_X)X - \tfrac12\mathcal A^\top(\Gamma + \Theta_{\mathcal A})\mathcal A + X^\top \tilde C \mathcal A .
$$

若 $\Gamma + \Theta_{\mathcal A} \succ 0$ 且 $Q + \Theta_X - \tilde C(\Gamma + \Theta_{\mathcal A})^{-1}\tilde C^\top \succ 0$，则均衡存在且唯一。
```

因此，{doc}`lq_mean_field_games` 中的等价性结果依然成立，只是需要再做一个调整：混合互动将有效交叉项从 $C$ 变为 $C - \Theta_{X\alpha}$。

在有效问题中完成配方表明，交叉项有两个作用。

它将有效状态曲率降低为 $Q + \Theta_X - \tilde C(\Gamma+\Theta_{\mathcal A})^{-1}\tilde C^\top$，并将有效漂移矩阵从 $A$ 改为 $A - B(\Gamma+\Theta_{\mathcal A})^{-1}\tilde C^\top$。

上一讲的比较静态分析适用于这些经过变换后的对象，而不是直接适用于互动矩阵本身。

主体偏离均值的动态变为

$$
dz = \left[B\Gamma^{-1}(B^\top\bar\beta_2 + C^\top) - A\right] z \, dt + \Sigma^{1/2} dW ,
$$

其中涉及自身交叉项 $C$，但仍不涉及互动矩阵，因此关于偏离均值的数据仍不包含有关策略互动的信息。

计划者问题的处理方式相同。

其有效交叉项为 $C - \Theta_{X\alpha} - \Theta_{x\mathcal A} - \Lambda_{X\mathcal A}$，因此不需要对混合互动施加任何限制，因为它们只通过其和进入该问题。

为了计算带有交叉项的均衡，我们扩展了 {doc}`lq_mean_field_games` 中的求解器。

`solve_continuous_are` 的 `s` 参数将标准黎卡提方程二次项中的 $\tilde P B$ 替换为 $\tilde P B + s$，因此设置 $s = -\tilde C$ 即可处理有效交叉项 $\tilde C$。

与上一讲一样，该函数会检查黎卡提方程的残差以及特征值条件，若两者之一不满足则报错。

```{code-cell} ipython3
def closed_loop(A, B, Γ_eff, P, C=None):
    "The matrix governing aggregate dynamics, Ẋ = (B Γ_eff^{-1} (B' P + C') - A) X."
    K = B.T @ P if C is None else B.T @ P + C.T
    return B @ np.linalg.solve(Γ_eff, K) - A

def mfg_riccati(A, B, Q, Γ, ρ, C=None, tol=1e-8):
    """
    Solve  (P B + C) Γ^{-1} (B' P + C') = Q + ρ P + P A + A' P, with C = 0 by
    default, for the solution P whose closed-loop matrix has eigenvalues with
    real parts below ρ/2, and raise ValueError if there is none.
    """
    n, k = B.shape
    C_use = np.zeros((n, k)) if C is None else C
    try:
        P = -solve_continuous_are(-(A + ρ/2 * np.eye(n)), B, Q, Γ, s=-C_use)
    except (np.linalg.LinAlgError, ValueError):
        raise ValueError("no admissible solution of the Riccati equation")
    quad = (P @ B + C_use) @ np.linalg.solve(Γ, B.T @ P + C_use.T)
    residual = quad - (Q + ρ*P + P @ A + A.T @ P)
    scale = 1 + np.abs(Q).max() + np.abs(quad).max()
    if (np.abs(residual).max() > tol*scale
            or np.linalg.eigvals(closed_loop(A, B, Γ, P, C)).real.max() >= ρ/2):
        raise ValueError("no admissible solution of the Riccati equation")
    return P
```

### 投资率与 Hayashi 调整成本

在资本积累示例中，调整成本仅取决于投资本身。

一种常见的替代方式是让它取决于投资率，

$$
\psi(i, k) = \frac{\phi}{2}\,\delta k\left(\frac{i}{\delta k} - 1\right)^2 ,
$$

这是一次齐次的，如 {cite:t}`Hayashi1982` 所述，当 $k = \bar k$ 时它退化为前文的成本 $\psi(i)$。

由于以百分比偏差表示时 $i - \delta k = \delta\bar k(\alpha - x)$，归一化后的成本在二阶近似下为 $\frac{\gamma}{2}(\alpha - x)^2$，其中 $\gamma = \phi\,\delta\bar k^{1-\nu}$ 与之前相同。

展开平方项表明，该成本给自身状态曲率增加了 $\gamma$，并产生了一个自身交叉项，

$$
q \to q + \gamma , \qquad C = \gamma ,
$$

而 $Q - C\Gamma^{-1}C^\top = q$ 保持不变。

这里没有混合互动，因此 $\tilde C = \gamma$。

在 {prf:ref}`mfgx_prop_cross` 的有效问题中完成配方，把该标量问题转化为一个不带交叉项的问题，其中漂移 $a = \delta$ 被替换为

$$
\tilde a = \delta - \frac{\delta\gamma}{\gamma+\theta_{\mathcal A}} = \frac{\delta\,\theta_{\mathcal A}}{\gamma + \theta_{\mathcal A}}
$$

有效状态曲率为 $q + \theta_X + \gamma - \gamma^2/(\gamma+\theta_{\mathcal A}) = q + \theta_X + \gamma\theta_{\mathcal A}/(\gamma + \theta_{\mathcal A})$。

因此总量特征值为

$$
\lambda = \frac\rho2 - \sqrt{\left(\frac\rho2 + \tilde a\right)^2
+ \frac{\delta^2}{\gamma+\theta_{\mathcal A}}\left(q + \theta_X + \frac{\gamma\,\theta_{\mathcal A}}{\gamma+\theta_{\mathcal A}}\right)} ,
$$

令 $\theta_X = \theta_{\mathcal A} = 0$（此时 $\tilde a = 0$），即得个体的特征值。

在 Hayashi 成本下，无论资本存量为多少，按重置率投资都是免费的，因此折旧不再将企业拉回其稳态。

在没有交叉项的公式中位于根号内部的漂移项 $\delta$ 从个体问题中消失了，它只通过投资品价格重新出现在行业问题中。

让我们在 $\gamma$ 保持其校准值的情况下，比较有无 Hayashi 成本时的半衰期。

```{code-cell} ipython3
def λ_cross(ρ, a, b, q_eff, γ_eff, c_eff):
    "Scalar aggregate eigenvalue with an effective cross term, by completing the square."
    a_t = a - b*c_eff/γ_eff
    q_t = q_eff - c_eff**2/γ_eff
    return ρ/2 - np.sqrt((ρ/2 + a_t)**2 + b**2*q_t/γ_eff)

λ_ind_H = λ_cross(ρ_c, δ_c, δ_c, q_c + γ_c, γ_c, γ_c)
λ_agg_H = λ_cross(ρ_c, δ_c, δ_c, q_c + γ_c + θ_X_c, γ_c + θ_A_c, γ_c)

# the general solver agrees
one = np.array([[1.0]])
P_H = mfg_riccati(δ_c*one, δ_c*one, (q_c + γ_c + θ_X_c)*one,
                  (γ_c + θ_A_c)*one, ρ_c, C=γ_c*one)
λ_agg_solver = closed_loop(δ_c*one, δ_c*one, (γ_c + θ_A_c)*one, P_H, C=γ_c*one)[0, 0]
print(f"aggregate λ: closed form {λ_agg_H:.6f}, solver {λ_agg_solver:.6f}\n")

print(f"{'':>20}{'individual':>12}{'industry':>10}{'ratio':>8}")
for label, (λ_i, λ_a) in {'investment costs': (λ_micro, λ_c),
                          'Hayashi costs':    (λ_ind_H, λ_agg_H)}.items():
    print(f"{label:>20}{half_life(λ_i):>12.2f}{half_life(λ_a):>10.2f}"
          f"{half_life(λ_a)/half_life(λ_i):>8.2f}")
```

Hayashi 成本会同时拖慢单个企业与整个行业的调整速度，并且两者半衰期之比大致不变。

{ref}`mfgx_ex_hayashi` 重新校准 $\phi$ 以恢复三年的个体半衰期，并重新审视规模报酬不变时的极限情形。

(mfgx_restriction)=
### 当限制条件不成立时

{cite:t}`AlvarezArgente2026` 指出，限制条件 $\Theta_{X\alpha} = \Theta_{x\mathcal A}$ 并没有一般性的经济学依据。

考虑一个国家拥有资本 $k$ 和投资 $i$，并将世界资本 $K$ 和世界投资 $I$ 视为给定。

假设其消费为 $c = f(k, K) - i$，期间效用为 $u(c)$，其中 $u'' < 0$ 且 $f_K > 0$。

那么 $\partial^2 u/\partial i\,\partial K = -u''(c) f_K > 0$，因为更高的世界资本会提高消费，从而降低投资的边际效用成本。

但 $\partial^2 u/\partial k\,\partial I = 0$，因为世界投资不会影响资本的回报。

因此这两个混合互动是不同的，用 {eq}`mfgx_return_general` 的记号表示即为 $\Theta_{X\alpha} < 0 = \Theta_{x\mathcal A}$。

当两者不同时，{eq}`mfgx_cross_system` 仍是一个线性状态-协态系统，但黎卡提方程的两个外部因子不再互为转置，该系统的矩阵也不再是哈密顿型的。

此时其特征值未必成对出现为 $\lambda$ 与 $\rho - \lambda$，而正是这一点使 {doc}`lq_mean_field_games` 中得出“至多一个均衡”结论的论证不再成立。

即使只有单一状态，两个特征值也可能都位于 $\rho/2$ 以下，从而产生 {cite:t}`BenhabibFarmer1994` 所研究的那种连续统均衡；或者两者都位于其上，此时不存在均衡。

为了展示这种配对关系的失效，我们编写一个函数来构造 {eq}`mfgx_cross_system` 的矩阵，允许两个混合互动不同。

```{code-cell} ipython3
def state_costate(A, B, Q_eff, Γ_eff, ρ, C1, C2):
    """
    Matrix of the state-costate system for (w, X).  C1 = C - Θ_xA enters the
    costate equation and C2 = C - Θ_Xα enters the aggregate action.
    """
    n = A.shape[0]
    G_B = np.linalg.solve(Γ_eff, B.T)      # (Γ + Θ_A)^{-1} B'
    G_C = np.linalg.solve(Γ_eff, C2.T)     # (Γ + Θ_A)^{-1} C2'
    return np.block([[ρ*np.eye(n) + A.T - C1 @ G_B, Q_eff - C1 @ G_C],
                     [B @ G_B,                      B @ G_C - A]])
```

我们将其应用于 Hayashi 经济体，先保持原样，再加入一个具有开放经济符号特征的小混合互动，$\Theta_{X\alpha} = -0.05$，$\Theta_{x\mathcal A} = 0$。

```{code-cell} ipython3
for label, θ_Xα, θ_xA in (('restriction holds', 0.0, 0.0),
                          ('restriction fails', -0.05, 0.0)):
    M_sc = state_costate(δ_c*one, δ_c*one, (q_c + γ_c + θ_X_c)*one, (γ_c + θ_A_c)*one,
                         ρ_c, (γ_c - θ_xA)*one, (γ_c - θ_Xα)*one)
    ev = np.sort(np.linalg.eigvals(M_sc).real)
    print(f"{label}: eigenvalues {ev.round(4)}, sum {ev.sum():.4f}, ρ = {ρ_c}")
```

当限制条件成立时，特征值之和为 $\rho$，因此它们关于 $\rho/2$ 对称。

当该条件不成立时，它们之和为 $\rho + b(\theta_{x\mathcal A} - \theta_{X\alpha})/(\gamma + \theta_{\mathcal A})$，而开放经济的符号会把它们向上推，趋向于不存在均衡的情形。

{ref}`mfgx_ex_restriction` 找到了一组参数，使两个特征值都位于 $\rho/2$ 的同一侧。

计划者不受这一切的影响，因为它将两个混合互动都内部化，因而只看到它们的和。

(mfgx_forcing)=
## 强制过程与传导

同样的方法也可以处理主体无法控制的状态变量，例如边际成本。

将状态划分为 $x = (x^{(1)}, x^{(2)})$，其中 $x^{(1)}$ 是受控的，$x^{(2)}$ 是具有自身运动规律的强制过程，

$$
A = \begin{bmatrix} A_{11} & A_{12} \\ 0 & A_{22}\end{bmatrix},
\qquad
B = \begin{bmatrix} B_1 \\ 0 \end{bmatrix} .
$$

此时 $B$ 不再具有满行秩，$B\Gamma^{-1}B^\top$ 是奇异的，因此黎卡提方程不再属于标准的满秩类型，但它们是分块递归的。

遵循 {cite:t}`AlvarezArgente2026`，我们只保留通过状态进行的互动，即 $\Theta_{\mathcal A} = 0$。

```{prf:proposition}
:label: mfgx_prop_forcing

假设 $\Theta_{\mathcal A} = 0$，$B_1\Gamma^{-1}B_1^\top$、$Q_{11}$ 和 $Q_{11} + \Theta_{X,11}$ 均为正定矩阵，且 $-A_{22}$ 的特征值实部均小于 $\rho/2$。

那么 $P$ 的左上块求解仅由受控部分构成的均衡黎卡提方程，

$$
P_{11}B_1\Gamma^{-1}B_1^\top P_{11} = Q_{11} + \Theta_{X,11} + \rho P_{11} + P_{11}A_{11} + A_{11}^\top P_{11} ,
$$

其余各块求解线性方程，而总量闭环矩阵的特征值由 $B_1\Gamma^{-1}B_1^\top P_{11} - A_{11}$ 的特征值与 $-A_{22}$ 的特征值共同组成。
```

$Q$ 和 $\Theta_X$ 中只涉及强制过程的那些分块会影响问题的值，但不会影响策略，因此可以自由选取它们，以保持两个矩阵都正定。

下面是一个具有 {cite:t}`BallRomer1990` 的实际摩擦精神的定价示例。

某商店的对数价格 $x^{(1)}$ 有一个目标值，该目标将权重 $1-\chi$ 赋予其自身的对数边际成本 $x^{(2)}$，将权重 $\chi \in [0, 1)$ 赋予平均价格 $X^{(1)}$，而商店改变价格需支付一项 Rotemberg 成本：

$$
-\tfrac q2\left(x^{(1)} - (1-\chi)x^{(2)} - \chi X^{(1)}\right)^2 - \tfrac\gamma2\alpha^2,
\qquad dx^{(1)} = \alpha \, dt .
$$

权重 $\chi$ 是竞争对手价格的静态传导率，因此它衡量了策略互补性，且 $\Theta_{X,11} = -q\chi$。

将强制过程与平均价格配对的项既不影响策略也不影响动态，因此我们将其舍去。

成本遵循 $dx^{(2)} = -a_{22}x^{(2)}dt + \sigma dW + \upsilon \, d\mathcal J$，其中 $W$ 是特异性的，$\mathcal J$ 是共同的跳跃过程。

特异性成本冲击会使商店偏离均值，因此由单主体矩阵 $\bar\beta_2$ 支配其响应；而共同成本冲击会使整个行业移动，由 $P$ 支配。

让我们跟踪这两种响应，考察成本一次性永久上升百分之一的情形，$a_{22} = 0$。

```{code-cell} ipython3
def pricing_with_costs(χ, a_22, ρ=0.04, q=1.0, γ=1.0):
    "Aggregate and deviation drift matrices in the pricing example with cost shocks."
    A_f = np.array([[0.0, 0.0],
                    [0.0, a_22]])
    B_f = np.array([[1.0],
                    [0.0]])
    Γ_f = np.array([[γ]])
    Q_f = q*np.array([[1.0, -(1 - χ)],
                      [-(1 - χ), 2.0]])      # (2, 2) element chosen freely
    Θ_f = np.array([[-q*χ, 0.0],
                    [0.0,  0.0]])
    β2_f = mfg_riccati(A_f, B_f, Q_f, Γ_f, ρ)
    P_f = mfg_riccati(A_f, B_f, Q_f + Θ_f, Γ_f, ρ)
    return closed_loop(A_f, B_f, Γ_f, P_f), closed_loop(A_f, B_f, Γ_f, β2_f)
```

第一个矩阵描述共同成本冲击后行业平均价格的变化，第二个矩阵描述特异性冲击后商店价格相对于均值的变化。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Common and idiosyncratic cost pass-through
    name: fig-mfgx-pass-through
---
t_f = np.linspace(0, 6, 200)
cost_shock = np.array([0.0, 1.0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for χ in (0.0, 0.5, 0.8):
    U_agg, U_dev = pricing_with_costs(χ, 0.0)
    agg = [(expm(U_agg*t) @ cost_shock)[0] for t in t_f]
    dev = [(expm(U_dev*t) @ cost_shock)[0] for t in t_f]
    axes[0].plot(t_f, agg, lw=2, label=f"$\\chi = {χ}$")
    axes[1].plot(t_f, dev, lw=2, label=f"$\\chi = {χ}$")

axes[0].set_title('average price, common cost shock')
axes[1].set_title("store's relative price, idiosyncratic cost shock")
for ax in axes:
    ax.set_xlabel('years')
    ax.set_ylabel('percent')
    ax.legend()
plt.tight_layout()
plt.show()
```

商店会迅速传导特异性成本上升，但只传导一部分，因为其竞争对手的价格不会变化：长期来看，它会传导其中的 $1 - \chi$。

而整个行业会完全传导总量成本上升，但速度较慢，因为每家商店都在等待其他商店行动。

让我们比较一下半衰期。

```{code-cell} ipython3
for χ in (0.0, 0.5, 0.8):
    U_agg, U_dev = pricing_with_costs(χ, 0.0)
    λ_agg_f, λ_dev_f = U_agg[0, 0], U_dev[0, 0]
    print(f"χ = {χ}: store half-life {half_life(λ_dev_f):.2f} years, "
          f"industry half-life {half_life(λ_agg_f):.2f} years")
```

当 $\chi = 0.8$ 时，商店大约在 0.7 年内就能消除与新目标差距的一半，而整个行业大约需要 1.6 年。

商店的半衰期对每个 $\chi$ 都相同，因为商店与均值的偏差只由其自身的曲率 $q$ 决定。

因此商店对自身成本冲击的响应速度完全无法揭示 $\chi$ 的信息——这正是 {doc}`lq_mean_field_games` 中那个缺失截距问题的另一种表现形式。

但响应的*幅度*则是另一回事：长期来看，商店会传导其自身成本冲击的 $1 - \chi$，因为 $\chi$ 也通过成本权重进入了商店自身的收益函数。

当一个模型以这种方式把自身曲率 $Q$ 与互动系数 $\Theta_X$ 联系起来时，微观数据终究能提供信息；而上一讲的结果表明，在给定 $Q$ 的情况下，它们并不能。

{ref}`mfgx_ex_forcing` 在这个例子中验证了 {prf:ref}`mfgx_prop_forcing`，并研究了暂时性成本冲击。

(mfgx_higher_moments)=
## 通过高阶矩的耦合

到目前为止，主体只通过均值来关心状态分布。

{cite:t}`AlvarezArgente2026` 也允许支付取决于更高阶矩。

取一个标量状态与标量行动，设 $\theta_{\mathcal A} = 0$，记 $M_r(t) = \int x^r m(x,t) dx$ 为横截面分布的第 $r$ 阶矩。

期间收益为

$$
-\tfrac12 q x^2 - \left(\theta_0 + \sum_{r=1}^{\bar r}\theta_r M_r\right) x - \tfrac12\gamma\alpha^2 ,
$$

状态遵循 $dx = (b\alpha - ax)dt + \sigma dW$，而 {doc}`lq_mean_field_games` 的基准情形对应于 $\bar r = 1$ 且 $\theta_0 = 0$ 的情形。

一种自然的策略互补性概念是：随机意义上更高的分布会提高 $x$ 的边际收益，这里意味着 $-\sum_r \theta_r x^r$ 关于 $x$ 非减。

我们下面的例子中，$\theta_1$ 和 $\theta_3$ 为负，其余系数为零，满足这一点。

价值函数仍是二次的，其曲率 $\bar\beta_2$ 仍与单主体问题相同，个体特征值

$$
\lambda_{\text{ind}} \equiv \frac{b^2}{\gamma}\bar\beta_2 - a = \frac\rho2 - \sqrt{\left(a + \frac\rho2\right)^2 + \frac{b^2 q}{\gamma}}
$$

保持不变。

发生变化的是加总过程。

柯尔莫戈罗夫前向方程意味着

$$
\dot M_r = r\lambda_{\text{ind}} M_r + r\frac{b^2}{\gamma}\beta_1 M_{r-1} + \frac{r(r-1)\sigma^2}{2}M_{r-2} ,
\qquad
\dot\beta_1 = (\rho - \lambda_{\text{ind}})\beta_1 + \theta_0 + \sum_{r=1}^{\bar r}\theta_r M_r ,
$$

因此这些矩服从一个有限系统，但由于存在乘积项 $\beta_1 M_{r-1}$，该系统是非线性的。

当 $\sigma > 0$、$\lambda_{\text{ind}} < 0$ 且 $\theta_0 = -\sum_{r\ \mathrm{even}}\theta_r \bar M_r$ 时，存在一个 $\beta_1 = 0$ 的稳态，其中不变分布为均值为零、方差为 $v = \sigma^2/(-2\lambda_{\text{ind}})$ 的正态分布。

它的奇数阶矩为零，偶数阶矩为 $\bar M_r = (r-1)!!\, v^{r/2}$；当 $\sigma \to 0$ 时，它退化为零点处的点质量分布，我们在下文中将其作为一种极限情形纳入考察。

在该稳态附近线性化该系统，可得 $\dot y = \tilde{\mathcal H} y$，其中 $y = (\hat\beta_1, \hat M_1, \ldots, \hat M_{\bar r})$，字母上方的尖号表示相对于稳态的偏差。

```{prf:proposition}
:label: mfgx_prop_moments

设 $\bar r$ 为偶数，并定义

$$
\bar\theta \equiv \sum_{r\ \mathrm{odd}} r\,\bar M_{r-1}\,\theta_r .
$$

$\tilde{\mathcal H}$ 的特征值为 $\tilde\lambda_r = r\lambda_{\text{ind}}$（$r = 2, \ldots, \bar r$），以及

$$
\tilde\lambda_1 = \frac\rho2 - \sqrt{\left(a + \frac\rho2\right)^2 + \frac{b^2(q + \bar\theta)}{\gamma}},
\qquad
\tilde\lambda_{\bar r+1} = \frac\rho2 + \sqrt{\left(a + \frac\rho2\right)^2 + \frac{b^2(q + \bar\theta)}{\gamma}} .
$$

若 $q + \bar\theta + \gamma a(a+\rho)/b^2 > 0$，该稳态是局部确定的，有一个不稳定特征值和 $\bar r$ 个稳定特征值。
```

有三点值得注意。

第一，互动系数只通过 $\bar\theta$ 进入，并且只有作用于奇数阶矩的系数才会进入其中，因此 $\bar\theta$ 恰好扮演着 {doc}`lq_mean_field_games` 标量公式中 $\theta_X$ 的角色。

第二，$\bar\theta$ 把 $\theta_3, \theta_5, \ldots$ 按不变分布的偶数阶矩加权，而这些偶数阶矩会随 $\sigma^2$ 增大而增长。

在基准情形中，特异性冲击的方差从不进入黎卡提方程，因此对总量动态没有影响；而在存在高阶矩的情形下，特异性风险会影响总量的持续性。

第三，作用于偶数阶矩的系数不影响局部特征值，尽管它们会影响过渡动态，如 {ref}`mfgx_ex_moments` 所示。

让我们构造 $\tilde{\mathcal H}$ 并验证 {prf:ref}`mfgx_prop_moments`，取 $\bar r = 4$，且互动通过第一阶矩和第三阶矩进行。

```{code-cell} ipython3
def moment_system(ρ, a, b, q, γ, σ, θ):
    """
    Linearized matrix for (β̂₁, M̂₁, ..., M̂_r̄) around the zero-mean steady
    state, with θ = (θ₁, ..., θ_r̄).  Also returns the steady-state moments
    M̄₀, ..., M̄_r̄ and the individual eigenvalue.
    """
    r_bar = len(θ)
    λ_ind = ρ/2 - np.sqrt((a + ρ/2)**2 + b**2*q/γ)
    v = σ**2/(-2*λ_ind)                       # invariant variance
    M_bar = np.zeros(r_bar + 1)
    M_bar[0] = 1.0
    for r in range(2, r_bar + 1, 2):
        M_bar[r] = (r - 1)*v*M_bar[r - 2]
    H_tilde = np.zeros((r_bar + 1, r_bar + 1))
    H_tilde[0, 0] = ρ - λ_ind
    H_tilde[0, 1:] = θ
    for r in range(1, r_bar + 1):
        H_tilde[r, 0] = r*M_bar[r - 1]*b**2/γ
        H_tilde[r, r] = r*λ_ind
        if r >= 3:                            # M̂₀ is identically zero
            H_tilde[r, r - 2] = r*(r - 1)*σ**2/2
    return H_tilde, M_bar, λ_ind

ρ_m, a_m, b_m, q_m, γ_m = 0.05, 0.3, 1.0, 1.0, 1.0
θ_m = np.array([-0.3, 0.0, -0.2, 0.0])       # complementarity through M₁ and M₃

for σ in (0.0, 0.5, 1.0):
    H_m, M_bar, λ_ind = moment_system(ρ_m, a_m, b_m, q_m, γ_m, σ, θ_m)
    θ_bar = θ_m[0] + 3*M_bar[2]*θ_m[2]
    root = np.sqrt((a_m + ρ_m/2)**2 + b_m**2*(q_m + θ_bar)/γ_m)
    predicted = np.r_[ρ_m/2 - root, ρ_m/2 + root, [r*λ_ind for r in range(2, 5)]]
    print(f"σ = {σ}: eigenvalues {np.sort(np.linalg.eigvals(H_m).real).round(4)}")
    print(f"        predicted   {np.sort(predicted).round(4)}")
```

为了追踪均衡路径，我们保留 $\bar r$ 个位于 $\rho/2$ 以下的特征值，将 $\hat\beta_1$ 表示为稳定子空间上矩偏差的线性函数，然后代入系统。

这样就得到了一个仅涉及矩偏差的运动规律，因此初始条件只需给定这些偏差，而 $\hat\beta_1(0)$ 则由鞍路径确定。

```{code-cell} ipython3
def moment_dynamics(H_tilde, ρ):
    "Law of motion of the moment deviations on the stable subspace of H_tilde."
    ev, V = np.linalg.eig(H_tilde)
    V_s = V[:, ev.real < ρ/2]
    s = np.real(V_s[0] @ np.linalg.inv(V_s[1:]))     # β̂₁ = s · M̂
    return H_tilde[1:, 1:] + np.outer(H_tilde[1:, 0], s)
```

作为初始条件，我们将不变分布向右平移 $\Delta$，这在一阶近似下会使 $M_1$ 提高 $\Delta$，使 $M_3$ 提高 $3v\Delta$，而偶数阶矩保持不变。

下图展示了在若干特异性波动率水平下均值的路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Idiosyncratic risk and mean persistence
    name: fig-mfgx-moments
---
t_m = np.linspace(0, 6, 200)
Δ = 0.1

fig, ax = plt.subplots(figsize=(8, 4.5))
for σ in (0.0, 0.6, 1.0, 1.3):
    H_m, M_bar, λ_ind = moment_system(ρ_m, a_m, b_m, q_m, γ_m, σ, θ_m)
    U_m = moment_dynamics(H_m, ρ_m)
    M_hat0 = np.array([Δ, 0.0, 3*M_bar[2]*Δ, 0.0])
    path = [(expm(U_m*t) @ M_hat0)[0] for t in t_m]
    θ_bar = θ_m[0] + 3*M_bar[2]*θ_m[2]
    λ_1 = ρ_m/2 - np.sqrt((a_m + ρ_m/2)**2 + b_m**2*(q_m + θ_bar)/γ_m)
    ax.plot(t_m, path, lw=2,
            label=f"$\\sigma = {σ}$, slowest half-life {half_life(λ_1):.2f}")
ax.set_xlabel('$t$')
ax.set_ylabel('cross-sectional mean $M_1(t)$')
ax.legend()
plt.tight_layout()
plt.show()
```

在这里，更大的特异性波动率使均值变得更具持续性。

更大的方差会对第三阶矩施加更大的权重，而互补性正是通过第三阶矩发挥作用，因此有效互动 $\bar\theta$ 变得更强。

## 练习

```{exercise}
:label: mfgx_ex_hayashi

本练习延续 Hayashi 示例。

1. 选取 $\phi$，使得一个忽略整个行业的企业在 Hayashi 成本下仍需三年才能弥合一半的资本缺口，并重新计算行业的半衰期以及两个半衰期之比。
1. 保持第1部分的 $\phi$ 和固定的 $\varepsilon_s$ 不变，计算规模报酬不变的极限情形（$\nu \to 1$，此时 $q + \theta_X \to 0$）下的总量特征值：一次不考虑价格效应，一次使用校准后的供给弹性。解释为什么这些结果与无 Hayashi 成本时成立的极限 $\lambda \to -\delta$ 不同。
```

```{solution-start} mfgx_ex_hayashi
:class: dropdown
```

对于第1部分，个体特征值为 $\rho/2 - \sqrt{\rho^2/4 + \delta^2 q/\gamma}$，因此目标值 $\lambda$ 要求 $\gamma = \delta^2 q/[(\rho/2 - \lambda)^2 - \rho^2/4]$。

```{code-cell} ipython3
λ_target = -np.log(2)/3
γ_H = δ_c**2*q_c/((ρ_c/2 - λ_target)**2 - ρ_c**2/4)

λ_ind_H = λ_cross(ρ_c, δ_c, δ_c, q_c + γ_H, γ_H, γ_H)
λ_agg_H = λ_cross(ρ_c, δ_c, δ_c, q_c + γ_H + θ_X_c, γ_H + θ_A_c, γ_H)

print(f"γ = {γ_H:.4f}, so φ = {γ_H/cap_scale(η_c, ν_c):.4f}")
print(f"individual half-life {half_life(λ_ind_H):.2f}, "
      f"industry half-life {half_life(λ_agg_H):.2f}, "
      f"ratio {half_life(λ_agg_H)/half_life(λ_ind_H):.2f}")
```

匹配三年的个体半衰期需要一种比之前更便宜的安装技术，$\phi \approx 0.11$，而非 $1/7$。

此时行业的半衰期为 5.4 年，因此微观与宏观调整之间的差距会扩大：比值从 1.66 升至 1.81。

对于第2部分，当 $q + \theta_X = 0$ 时，{prf:ref}`mfgx_prop_cross` 中的有效状态曲率为 $\gamma$，行业的有效收益为

$$
-\tfrac12\gamma X^2 - \tfrac12(\gamma + \theta_{\mathcal A})\mathcal A^2 + \gamma X\mathcal A
= -\tfrac\gamma2(\mathcal A - X)^2 - \tfrac{\theta_{\mathcal A}}2\mathcal A^2 .
$$

当 $\nu \to 1$ 时，将 $\phi$ 和 $1/\varepsilon_s$ 转化为 $\gamma$ 和 $\theta_{\mathcal A}$ 的尺度 $\delta\bar k^{1-\nu}$ 趋于 $\delta(\eta-1)/[\eta(\rho+\delta)]$。

让我们在有无价格效应两种情形下计算该极限。

```{code-cell} ipython3
φ_H = γ_H/cap_scale(η_c, ν_c)
scale_lim = cap_scale(η_c, 1.0)
γ_lim = φ_H*scale_lim
for inv_ε in (0.0, 1/ε_s_c):
    θ_A_lim = inv_ε*scale_lim
    λ_lim = λ_cross(ρ_c, δ_c, δ_c, γ_lim, γ_lim + θ_A_lim, γ_lim)
    print(f"1/ε_s = {inv_ε:.3f}: θ_A = {θ_A_lim:.4f}, λ = {λ_lim:.5f}")
```

当 $\theta_{\mathcal A} = 0$ 时，唯一的成本就是净投资的成本，因此任何资本存量都能被免费维持，没有任何力量将行业拉回：总量状态具有单位根。

而在没有 Hayashi 成本的情况下，要维持高于稳态的资本存量需要高于 $\bar i$ 的总投资，而这会受到 $\psi(i)$ 的惩罚，因此行业任由折旧发挥作用，$\lambda \to -\delta$。

当 $\theta_{\mathcal A} > 0$ 时，投资品价格会惩罚偏离 $\bar i$ 的总投资，从而恢复了一定程度的均值回归。

```{solution-end}
```

```{exercise}
:label: mfgx_ex_restriction

本练习说明，即使只有单一状态，限制条件 $\Theta_{X\alpha} = \Theta_{x\mathcal A}$ 为何重要。

取 $\rho = 0.05$、$a = 0.3$、$b = q = \gamma = 1$、$\theta_X = \theta_{\mathcal A} = 0$，不设自身交叉项，使用 `state_costate` 并设 $C_1 = -\theta_{x\mathcal A}$、$C_2 = -\theta_{X\alpha}$。

1. 证明状态-协态矩阵的迹为 $\rho + b(\theta_{x\mathcal A} - \theta_{X\alpha})/(\gamma + \theta_{\mathcal A})$，因此其特征值只有在 $\theta_{X\alpha} = \theta_{x\mathcal A}$ 时才可能关于 $\rho/2$ 对称。
1. 计算 $(\theta_{X\alpha}, \theta_{x\mathcal A})$ 分别为 $(0, 0)$、$(-2, -2)$、$(-1.5, -2.5)$ 和 $(-2.5, -1.5)$ 时的特征值，并统计其中有多少特征值的实部低于 $\rho/2$。每种计数情形对均衡数量意味着什么？
```

```{solution-start} mfgx_ex_restriction
:class: dropdown
```

对于第1部分，令 $g = \gamma + \theta_{\mathcal A}$，矩阵的对角元素为 $\rho + a - b C_1/g$ 和 $b C_2/g - a$。

它们之和为 $\rho + b(C_2 - C_1)/g = \rho + b(\theta_{x\mathcal A} - \theta_{X\alpha})/g$。

关于 $\rho/2$ 对称的特征值之和必须等于 $\rho$，这要求 $\theta_{X\alpha} = \theta_{x\mathcal A}$。

对于第2部分，我们在各情形下计算特征值。

```{code-cell} ipython3
ρ_e, a_e = 0.05, 0.3

print(f"{'(θ_Xα, θ_xA)':>14}{'eigenvalues':>34}{'trace':>8}{'below ρ/2':>11}")
for θ_Xα_e, θ_xA_e in ((0.0, 0.0), (-2.0, -2.0), (-1.5, -2.5), (-2.5, -1.5)):
    M_e = state_costate(a_e*one, one, one, one, ρ_e,
                        -θ_xA_e*one, -θ_Xα_e*one)
    ev = np.linalg.eigvals(M_e)
    print(f"{str((θ_Xα_e, θ_xA_e)):>14}{str(np.round(ev, 4)):>34}"
          f"{ev.real.sum():>8.3f}{np.sum(ev.real < ρ_e/2):>11}")
```

当没有混合互动时，$\rho/2$ 两侧各有一个特征值，均衡唯一。

在 $(-2, -2)$ 处限制条件成立，但 {prf:ref}`mfgx_prop_cross` 中的充分条件 $q - \tilde C^2/g > 0$ 不满足，因为 $1 - 4 < 0$。

此时特征值为复数，实部恰好等于 $\rho/2$，因此当 $X(0) \neq 0$ 时不存在可行路径，也就不存在均衡。

在 $(-1.5, -2.5)$ 处，两个特征值都低于 $\rho/2$，因此任何初始协态 $w(0)$ 都能产生一条可行路径。

这些路径中的每一条都是一个均衡。

给定增长速度慢于 $e^{\rho t/2}$ 的总量路径，个体的协态方程有唯一的可行解，因为其自身的根 $\rho - \lambda_{\text{ind}}$ 超过 $\rho/2$，系统任一路径上的 $w$ 就是那个解。

因此对每个初始分布都存在一个均衡的连续统。

在 $(-2.5, -1.5)$ 处，两个特征值都高于 $\rho/2$，同样当 $X(0) \neq 0$ 时不存在均衡。

```{solution-end}
```

```{exercise}
:label: mfgx_ex_forcing

本练习运用 {ref}`mfgx_forcing` 中的定价示例。

1. 取 $\chi = 0.8$、$a_{22} = 0.5$。检验总量闭环矩阵的特征值是否由 $B_1\Gamma^{-1}B_1^\top P_{11} - A_{11}$ 的特征值与 $-a_{22}$ 共同组成，正如 {prf:ref}`mfgx_prop_forcing` 所述。同时检验改变 $Q$ 和 $\Theta_X$ 的 $(2,2)$ 元素是否会使 $P$ 的第一行（从而定价策略）保持不变。
1. 在暂时性成本冲击下，取 $a_{22} = 0.5$，绘制 $\chi \in \{0, 0.5, 0.8\}$ 时行业对共同成本上升百分之一的响应。峰值响应的大小和时间如何依赖于 $\chi$？
```

```{solution-start} mfgx_ex_forcing
:class: dropdown
```

对于第1部分，我们以 $(2,2)$ 元素作为参数重建该例。

```{code-cell} ipython3
χ_e, a_22_e, ρ_f, q_f, γ_f = 0.8, 0.5, 0.04, 1.0, 1.0
A_f = np.array([[0.0, 0.0],
                [0.0, a_22_e]])
B_f = np.array([[1.0],
                [0.0]])
Γ_f = np.array([[γ_f]])

def P_forcing(q_22, θ_22):
    "Equilibrium P when the (2, 2) elements of Q and Θ_X are q_22 and θ_22."
    Q_f = q_f*np.array([[1.0, -(1 - χ_e)],
                        [-(1 - χ_e), 0.0]])
    Q_f[1, 1] = q_22
    Θ_f = np.array([[-q_f*χ_e, 0.0],
                    [0.0, θ_22]])
    return mfg_riccati(A_f, B_f, Q_f + Θ_f, Γ_f, ρ_f)

P_a, P_b = P_forcing(2.0, 0.0), P_forcing(7.0, 3.0)
λ_11 = P_a[0, 0]/γ_f          # B_1 Γ^{-1} B_1' P_11 - A_11
ev_cl = np.sort(np.linalg.eigvals(closed_loop(A_f, B_f, Γ_f, P_a)).real)
print("closed-loop eigenvalues:   ", ev_cl.round(5))
print("controlled block and -a_22:", np.sort([λ_11, -a_22_e]).round(5))
print("first row of P unchanged:", np.allclose(P_a[0], P_b[0]))
```

对于第2部分，我们绘制平均价格的响应。

```{code-cell} ipython3
t_e = np.linspace(0, 8, 400)
fig, ax = plt.subplots(figsize=(8, 4.5))
for χ in (0.0, 0.5, 0.8):
    U_agg, _ = pricing_with_costs(χ, 0.5)
    agg = np.array([(expm(U_agg*t) @ np.array([0.0, 1.0]))[0] for t in t_e])
    k = agg.argmax()
    ax.plot(t_e, agg, lw=2, label=f"$\\chi = {χ}$")
    print(f"χ = {χ}: peak {agg[k]:.3f} percent after {t_e[k]:.2f} years")
ax.set_xlabel('years')
ax.set_ylabel('average price, percent')
ax.legend()
plt.tight_layout()
plt.show()
```

更强的互补性使峰值响应变得更小、出现得更晚。

每家商店提价都更慢，因为其他商店也更慢，因此等到行业价格上涨较多时，成本冲击已经开始消退。

```{solution-end}
```

```{exercise}
:label: mfgx_ex_moments

本练习表明，一个作用于偶数阶矩的互动项会如何在不改变局部特征值的情况下影响过渡动态。

取 $\bar r = 2$、$\theta_1 = -0.3$、$\sigma = 1$，以及 {ref}`mfgx_higher_moments` 中矩示例的其余参数。

1. 验证 $\theta_2$ 不会改变 $\tilde{\mathcal H}$ 的任何特征值。
1. 假设横截面方差上升 0.1，而均值保持为零。对 $\theta_2 \in \{-0.5, 0, 0.5\}$ 描绘均值的路径，并解释你所看到的现象。
1. {cite:t}`AlvarezArgente2026` 证明，沿鞍路径有 $\dot{\hat M}_1 = \tilde\lambda_1\hat M_1 + \zeta\hat M_2$，其中 $\zeta = b^2\theta_2/[\gamma(\tilde\lambda_1 + 2\lambda_{\text{ind}} - \rho)]$。验证该公式。
```

```{solution-start} mfgx_ex_moments
:class: dropdown
```

对于第1部分，我们计算每个 $\theta_2$ 对应的特征值。

```{code-cell} ipython3
θ_1 = -0.3
for θ_2 in (-0.5, 0.0, 0.5):
    H_2, _, _ = moment_system(ρ_m, a_m, b_m, q_m, γ_m, 1.0, np.array([θ_1, θ_2]))
    print(f"θ_2 = {θ_2:>4}: eigenvalues {np.sort(np.linalg.eigvals(H_2).real).round(5)}")
```

它们不会移动。

在零均值稳态处，第二阶矩独立演化，$\dot{\hat M}_2 = 2\lambda_{\text{ind}}\hat M_2$，因为其运动规律中的项 $2(b^2/\gamma)\beta_1 M_1$ 在一阶近似下消失了。

对于第2部分，我们从 $\hat M_1 = 0$、$\hat M_2 = 0.1$ 出发。

```{code-cell} ipython3
t_e = np.linspace(0, 4, 200)
fig, ax = plt.subplots(figsize=(8, 4.5))
for θ_2 in (-0.5, 0.0, 0.5):
    H_2, _, _ = moment_system(ρ_m, a_m, b_m, q_m, γ_m, 1.0, np.array([θ_1, θ_2]))
    U_2 = moment_dynamics(H_2, ρ_m)
    path = [(expm(U_2*t) @ np.array([0.0, 0.1]))[0] for t in t_e]
    ax.plot(t_e, path, lw=2, label=f"$\\theta_2 = {θ_2}$")
ax.set_xlabel('$t$')
ax.set_ylabel('cross-sectional mean $M_1(t)$')
ax.legend()
plt.tight_layout()
plt.show()
```

当 $\theta_2 > 0$ 时，较高的第二阶矩会通过项 $-\theta_2 M_2 x$ 降低所有人对 $x$ 的边际收益，因此在方差较高时主体会降低 $x$，于是均值先下降，之后随方差回归其不变水平而回升。

当 $\theta_2 < 0$ 时，均值反而上升；而当 $\theta_2 = 0$ 时它根本不会移动。

因此，一个关于离散程度的冲击会使均值发生变化，即使决定局部持续性的特征值并未改变。

对于第3部分，我们将鞍路径运动规律的系数与公式进行比较。

```{code-cell} ipython3
for θ_2 in (-0.5, 0.5):
    H_2, _, λ_ind_2 = moment_system(ρ_m, a_m, b_m, q_m, γ_m, 1.0, np.array([θ_1, θ_2]))
    U_2 = moment_dynamics(H_2, ρ_m)
    λ_1 = ρ_m/2 - np.sqrt((a_m + ρ_m/2)**2 + b_m**2*(q_m + θ_1)/γ_m)
    ζ = b_m**2*θ_2/(γ_m*(λ_1 + 2*λ_ind_2 - ρ_m))
    print(f"θ_2 = {θ_2}: coefficients ({U_2[0, 0]:.5f}, {U_2[0, 1]:.5f}), "
          f"formula ({λ_1:.5f}, {ζ:.5f})")
```

```{solution-end}
```

## 延伸阅读

{cite:t}`AlvarezArgente2026` 在其论文的附录中阐述了交叉项与强制过程，并在正文中阐述了通过高阶矩进行的耦合。

Hayashi 示例、传导示例，以及 {ref}`mfgx_ex_restriction` 中关于单一状态下混合互动的分析，均为我们对其结果所作的自有示例说明。