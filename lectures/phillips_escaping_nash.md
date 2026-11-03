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
  title: 逃离纳什通胀
  headings:
    Overview: 概览
    The model: 模型
    The self-confirming equilibrium: 自我确认均衡
    Adaptation and the mean dynamics: 适应性与均值动态
    Escape dynamics as a control problem: 作为控制问题的逃逸动态
    The changed-measure escape path: 变测度逃逸路径
    The race of four ODEs: 四条常微分方程的竞速
    Mean dynamics reinforce the escape: 均值动态强化逃逸
    The experimentation trap: 试验陷阱
    Simulating escapes: 模拟逃逸
    Escape frequency and model richness: 逃逸频率与模型丰富度
    Escaping volatile inflation: 逃离高波动通货膨胀
    Exercises: 练习
---

(phillips_escaping_nash)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 逃离纳什通胀

```{index} single: Phillips Curve; Escaping Nash Inflation
```

```{contents} Contents
:depth: 2
```

## 概览

> 如果一个不太可能发生的事件发生了，它极有可能是以最可能的方式发生的。
>
> —— 迈克尔·哈里森

本讲座是 {doc}`phillips_learning` 的分析性姊妹篇。

它遵循 {cite:t}`ChoWilliamsSargent2002`（CWS），该文为 {cite:t}`Sargent1999` 第 8 章通过模拟发现的逃逸动态给出了一种*确定性刻画*。

在 {doc}`phillips_learning` 中，我们观察到一个恒定增益的政府反复地逃离纳什自我确认均衡，并在拉姆齐结果附近长时间停留，但我们只是通过模拟对这些逃逸做了非正式的描述。

CWS 表明，这些逃逸受它们自身的常微分方程支配，该方程由**大偏差理论**得出。

由此出现的图景包含两个确定性部分：

* **均值动态**：一个把政府信念拉*向*自我确认均衡的常微分方程；以及
* **逃逸动态**：第二个常微分方程，由一个"最可能的不太可能"的冲击序列驱动，将信念推*离*自我确认均衡，朝支持拉姆齐结果的信念方向移动。

一个引人注目的发现是，逃逸存在一条*主导路径*：在发生逃逸的条件下，政府的信念会沿着一条几乎确定的路线演变，沿途它暂时习得了自然率假说的某个版本，并削减了通货膨胀。

这些逃逸之所以反复发生，是因为政府使用了恒定增益最小二乘法，正如 {cite:t}`Sargent1999` 与 CWS 中那样。

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 中的政府则是对漂移系数运行卡尔曼滤波；在他们的模拟中，一旦政府逃逸成功便只会发生一次，此后似乎会无限期地停留在拉姆齐结果附近，而 {doc}`phillips_priors` 将这一差异追溯到政府的先验。

大偏差理论的结果是关于小增益情形的论断。

{ref}`en_simulation` 对模型进行了模拟，发现在本系列讲座所采用的增益水平下，逃逸实际上频繁发生，而非罕见事件。

我们使用便于分析处理的**静态**模型，其中政府将失业率对当期通货膨胀和一个常数项进行回归。

{doc}`phillips_learning` 模拟了 CWS 中更丰富的*动态*模型，其中政府的回归还包含滞后项；我们将在本讲座接近结尾处对两者进行比较。

{ref}`olg_phillips` 以浓缩的形式呈现了相同的静态模型及相同的校准设定。

从那里过来的读者应当注意，那里将信念记为 $\alpha$（截距项在前），增益记为 $\gamma$；而在本讲座中，$\gamma$ 表示信念，$\varepsilon$ 表示增益。

下一讲 {doc}`phillips_priors` 探讨政府关于参数漂移的*先验*如何重塑均值动态与逃逸动态。

让我们从导入相关模块开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
```

## 模型

真实经济是 {cite:t}`KydlandPrescott1977` 的自然率模型，

```{math}
:label: en_truth

U_n = u - \theta(\pi_n - \hat x_n) + \sigma_1 W_{1n},
\qquad
\pi_n = x_n + \sigma_2 W_{2n},
\qquad
\hat x_n = x_n,
```

其中 $\theta, u > 0$，且 $W_n = (W_{1n}, W_{2n})^\top$ 为独立同分布的标准高斯变量。

政府并不知道 {eq}`en_truth`。

在静态模型中，它通过将失业率对通货膨胀和一个常数项做回归，拟合一条非预期性菲利普斯曲线，

```{math}
:label: en_belief

U_n = \gamma_1 \pi_n + \gamma_{-1} + \eta_n ,
```

信念为 $\gamma = (\gamma_1, \gamma_{-1})$（斜率与截距），并将 $\eta_n$ 视为外生的。

```{note}
我们沿用 {cite:t}`ChoWilliamsSargent2002` 的顺序：斜率在前，回归量为 $\Phi = (\pi, 1)$。

这也与 {doc}`phillips_self_confirming` 中 $\gamma_1, \gamma_{-1}$ 的约定保持一致。

{doc}`phillips_priors` 和 {ref}`olg_phillips` 则遵循 {cite:t}`SargentWilliams2005` 的做法，将截距放在前面。

每篇讲座都遵循各自所依据论文的约定；{doc}`phillips_priors` 中的提示框给出了两种约定之间的转换方法。
```

在相信 {eq}`en_belief` 的前提下，政府求解 {doc}`菲尔普斯问题 <phillips_adaptive>`，其静态最优反应将通货膨胀设定为常数

```{math}
:label: en_bestresp

x(\gamma) = -\frac{\gamma_{-1}\, \gamma_1}{1 + \gamma_1^2} .
```

按照 CWS 的说法，三种信念值得单独命名：

* **信念 1（纳什）：** $\gamma_1 = -\theta$，截距使 $x = \theta u$，这正是 {cite:t}`KydlandPrescott1977` 的时间一致结果。
* **信念 2（拉姆齐）：** $\gamma_1 = 0$，此时政府认为不存在权衡取舍，设定 $x = 0$。
* **信念 3（归纳）：** 在 CWS 的动态模型中，当期与滞后通货膨胀系数之和为零，对于一个有耐心的政府，这同样会使通货膨胀趋于 $0$。

在静态模型中，通货膨胀只有一个系数，因此信念 3 与信念 2 重合：垂直的感知菲利普斯曲线“自然满足归纳性质”（CWS，第 6.2 节）。

```{code-cell} ipython3
class EscapeModel:
    "CWS 2002 static model. γ = (γ₁ slope, γ₋₁ intercept), regressors Φ = (π, 1)."

    def __init__(self, θ=1.0, u=5.0, σ1=0.3, σ2=0.3):
        self.θ, self.u, self.σ1, self.σ2 = θ, u, σ1, σ2

    def γ_sce(self):
        "Self-confirming beliefs (slope -θ, intercept u(1+θ²))."
        return np.array([-self.θ, self.u * (1 + self.θ**2)])

    def x(self, γ):
        γ1, γm1 = γ
        return -γm1 * γ1 / (1 + γ1**2)

    def M(self, γ):
        "E[ΦΦ'] with Φ = (π, 1)."
        x = self.x(γ)
        return np.array([[x**2 + self.σ2**2, x], [x, 1.0]])

    def g_bar(self, γ):
        "Mean-dynamics forcing E[Φ(U − Φ'γ)] = M(T(γ) − γ)."
        x = self.x(γ)
        E_ΦU = np.array([x * self.u - self.θ * self.σ2**2, self.u])
        return E_ΦU - self.M(γ) @ γ
```

## 自我确认均衡

自我确认均衡是一种能够自我复制的信念：政府依据 $\gamma$ 采取行动所生成的数据上，{eq}`en_belief` 的总体回归系数等于 $\gamma$ 本身。

将这些总体系数记为 $T(\gamma)$，并将回归变量的总体矩矩阵记为 $\bar M(\gamma) = \mathbb{E}\,\Phi\Phi^\top$，CWS 证明

$$
\bar g(\gamma) \equiv \mathbb{E}\left[\Phi(U - \Phi^\top \gamma)\right] = \bar M(\gamma) \left(T(\gamma) - \gamma\right),
$$

因此自我确认均衡解出 $\bar g(\gamma) = 0$。

在恒定政策 $x$ 下，只有意外冲击 $\sigma_2 W_{2n}$ 会影响通货膨胀，因此 $T(\gamma) = (-\theta,\ u + \theta x(\gamma))$。

因此，对于静态模型，均衡是直线 $\gamma_1 = -\theta$ 与抛物线 $\gamma_{-1} = u(1 + \gamma_1^2)$ 的交点，一个唯一点，它支持 {doc}`phillips_credibility` 中的纳什结果。

它与 {doc}`phillips_self_confirming` 中构造的自我确认均衡相同，此处以静态（常数加斜率）特例的形式给出，{doc}`phillips_priors` 中同样使用这一形式，这正是 {cite:t}`KydlandPrescott1977` 所称的*一致*均衡（{ref}`olg_phillips`）。

```{code-cell} ipython3
model = EscapeModel(θ=1.0, u=5.0, σ1=0.3, σ2=0.3)
γ_sce = model.γ_sce()

print(f"self-confirming beliefs γ = {γ_sce}  (slope -θ, intercept u(1+θ²))")
print(f"self-confirming inflation x = {model.x(γ_sce):.2f}  (= Nash = θu)")
print(f"check g_bar = {model.g_bar(γ_sce)}")
```

下图绘制了其交点即为均衡的两条轨迹。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The unique self-confirming equilibrium in belief space
    name: fig-esc-sce
---
γ1_grid = np.linspace(-2, 1, 200)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(γ1_grid, model.u * (1 + γ1_grid**2), label=r'$\gamma_{-1} = u(1+\gamma_1^2)$', lw=2)
ax.axvline(-model.θ, color='C1', ls='--', label=r'$\gamma_1 = -\theta$')
ax.plot(γ_sce[0], γ_sce[1], 'ko', ms=8)
ax.annotate('SCE (Nash)', γ_sce, (γ_sce[0] + 0.1, γ_sce[1] + 1))
ax.set_xlabel(r'slope $\gamma_1$')
ax.set_ylabel(r'intercept $\gamma_{-1}$')
ax.set_ylim(0, 25)
ax.legend()
plt.show()
```

## 适应性与均值动态

我们令政府具有适应性。

每一期，它设定 $x_n = x(\gamma_n)$，观察 $(\pi_n, U_n)$，并通过恒定增益递归最小二乘法更新其估计值：

```{math}
:label: en_rls

R_n = R_{n-1} + \varepsilon \left(\Phi_n \Phi_n^\top - R_{n-1}\right),
\qquad
\gamma_{n+1} = \gamma_n + \varepsilon R_n^{-1} \Phi_n \left(U_n - \Phi_n^\top \gamma_n\right),
```

其中 $\Phi_n = (\pi_n, 1)^\top$，$\varepsilon > 0$ 为增益。

这正是 {ref}`olg_phillips` 中 `constant_gain` 的递归式，只是回归量采用了 CWS 的顺序。

CWS 的方程（4.24）–（4.25）将 $R$ 的时间标记提前了一期，但当 $\varepsilon \to 0$ 时这没有区别。

由于政府依据其当前估计值采取行动，仿佛它是永久不变的，这是 {cite:t}`Kreps1998` 意义上的**预期效用**模型。

关于自我指涉模型中最小二乘学习的文献 {cite}`MarcetSargent1989jet,Woodford1990,EvansHonkapohja2001` 表明，当 $\varepsilon \to 0$ 且以 $t = n\varepsilon$ 度量时间时，信念可以由**均值动态**常微分方程近似：

```{math}
:label: en_mean

\dot\gamma = R^{-1} \bar g(\gamma),
\qquad
\dot R = \bar M(\gamma) - R .
```

{ref}`br_one_recursion` 针对标量递归式推导出此类常微分方程，并将其稳定性与 E-稳定性联系起来。

```{note}
**各讲座间的记号差异。** {ref}`pl_primer` 将信念堆叠为
$\phi = (\gamma, \operatorname{col}(R))$，并将均值动态写作
$\dot\phi = b(\phi)$。

在静态模型中，
$b(\phi) = \left(R^{-1}\bar g(\gamma),\ \operatorname{col}(\bar M(\gamma) - R)\right)$，因此
{eq}`en_mean` 正是该常微分方程的展开写法。

下文的逃逸动态仅对 $\gamma$ 分量添加了一个强制项 $v$。

{doc}`phillips_priors` 用卡尔曼滤波的协方差矩阵 $P$ 替代了缩放后的逆矩矩阵 $\varepsilon R^{-1}$，并将 $\gamma$ 的截距项排在前面。
```

{eq}`en_mean` 的一个静止点即为一个自我确认均衡，CWS 证明该常微分方程围绕它是*全局稳定*的。

在局部，当 $R = \bar M(\bar\gamma)$ 时，$\gamma$ 分量的特征值为 $-1$ 和 $-1/(1+\theta^2)$：这正是 {ref}`sc_estability` 与 {ref}`pl_eigen` 中的 E-稳定性特征值。

```{code-cell} ipython3
R_bar = model.M(γ_sce)                     # moment matrix at the SCE
h = 1e-6
J = np.column_stack([(model.g_bar(γ_sce + h * e) - model.g_bar(γ_sce - h * e)) / (2 * h)
                     for e in np.eye(2)])
print("eigenvalues of R̄⁻¹ Dḡ at the SCE:", np.linalg.eigvals(np.linalg.solve(R_bar, J)).round(4))
print("-1/(1+θ²) =", -1 / (1 + model.θ**2))
```

因此，仅在均值动态作用下，适应性政府会被吸引到纳什通货膨胀水平。

但这条路径可能是迂回曲折的。

下图将信念的起点设在通向拉姆齐水平的部分路程上，即 $\gamma(0) = \bar\gamma + (0.4, -3)$，并以两种方式展开。

在第一种方式中，矩矩阵 $R$ 从其纳什时期的值 $\bar M(\bar\gamma)$ 开始；在第二种方式中，它从新信念所对应的矩 $\bar M(\gamma(0))$ 开始。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "从受扰动起点出发的均值动态：当 R 从其纳什时期的值开始时，通胀先行下降"
    name: fig-esc-mean-dynamics
---
def mean_ode(t, z, model):
    γ, R = z[:2], z[2:].reshape(2, 2)
    return np.concatenate([np.linalg.solve(R, model.g_bar(γ)),
                           (model.M(γ) - R).ravel()])

γ0 = γ_sce + np.array([0.4, -3.0])
fig, ax = plt.subplots(figsize=(8, 4.5))
starts = [(R_bar, r'$R(0) = \bar M(\bar\gamma)$ (Nash-era moments)', 'R(0) = M(SCE)'),
          (model.M(γ0), r'$R(0) = \bar M(\gamma(0))$', 'R(0) = M(γ(0))')]
for R0, label, short in starts:
    sol = solve_ivp(mean_ode, [0, 60], np.concatenate([γ0, R0.ravel()]), args=(model,),
                    max_step=0.1, rtol=1e-9, atol=1e-11)
    x_path = -sol.y[1] * sol.y[0] / (1 + sol.y[0]**2)
    ax.plot(sol.t, x_path, label=label)
    k = np.argmin(x_path)
    print(f"{short:15s}: x(0) = {x_path[0]:.2f}, min x = {x_path[k]:.2f} "
          f"at t = {sol.t[k]:.2f}, x(60) = {x_path[-1]:.2f}")
ax.axhline(model.θ * model.u, color='k', ls='--', lw=1, label='Nash')
ax.set_xlabel('time $t = n\\varepsilon$')
ax.set_ylabel('inflation $x$')
ax.legend()
plt.show()
```

两条路径最终都回归纳什水平，但到达的方式不同。

当 $R$ 从纳什时期的矩开始时，通胀会先*进一步下降*，从约 3.1 降到约 1.7，之后才回升。

当 $R$ 从 $\bar M(\gamma(0))$ 开始时，通胀则单调上升。

这种下降正是 CWS（第 6.4 节）所描述的迂回路径的直观呈现，它源于 $R$ 滞后于 $\bar M(\gamma)$。

我们将在 {ref}`en_vector_field` 中重新讨论这一滞后现象，并发现正是它驱动了逃逸过程。

仅凭均值动态本身，无法解释 {doc}`phillips_learning` 中模拟所显示的反复出现的低通胀访问现象。

要解释这一点，我们需要第二种力量。

## 作为控制问题的逃逸动态

尽管当 $\varepsilon \to 0$ 时噪声的影响会消失，但对于固定的正增益，不寻常的冲击序列可以将信念推离自我确认均衡很远。

{ref}`pl_primer` 阐述了一般理论。

它构建了创新项的对数矩生成函数 $H$，其勒让德变换 $L$，以及一个**行动泛函** $S(T, \gamma) = \int_0^T L\, ds$，用以衡量一条信念路径有多不可能。

**主导逃逸路径**在离开 $\bar\gamma$ 的一个邻域 $G$ 的所有路径中最小化该行动值。

这里我们只需要其操作化版本。

CWS 借鉴了诺亚·威廉姆斯的博士论文 {cite}`Williams2001`（以修正形式发表为 {cite:t}`Williams2019`），将该问题归纳为一个简洁的控制问题。

H-泛函变为具有归一化矩阵 $Q$（由李雅普诺夫方程得到的四阶矩矩阵）的二次形式，逃逸路径求解

```{math}
:label: en_control

\bar S = \inf_{v(\cdot),\, T} \; \frac12 \int_0^T v(s)^\top Q(\gamma(s), R(s))^{-1} v(s)\, ds
```

约束条件为*受扰动的*均值动态

```{math}
:label: en_perturbed

\dot\gamma = R^{-1}\bar g(\gamma) + v,
\qquad
\dot R = \bar M(\gamma) - R,
\qquad
\gamma(0) = \bar\gamma, \; \gamma(T) \notin G .
```

可以将 {eq}`en_control` 理解为一个最小二乘问题：$v$ 是均值动态要实现逃逸所需的额外"强制项"，$Q$ 起协方差矩阵的作用，代价最低的强制项即是最可能出现的异常冲击序列。

{cite:t}`Williams2019`（第 892 页）指出，2001 年的版本在计算 $H$ 时错误地套用了 Worms（1999）的一个定理。

这一差异仅对动态模型有影响；对于静态模型，CWS 直接计算了 $H$。

因此，下文关于静态模型的结果依然成立，而 CWS 对动态模型给出的速率则依赖于早先的那个捷径算法。

设 $\tau^\varepsilon$ 为信念离开 $G$ 的首个时刻，以 $t = n\varepsilon$ 来度量。

其两个推论（CWS，定理 5.3）将该控制问题与随机模型联系起来：

* 当 $\varepsilon \to 0$ 时，$\varepsilon \log \mathbb{P}(\text{在 } T \text{ 之前逃逸}) \to -\bar S$，且 $\varepsilon \log \mathbb{E}\,\tau^\varepsilon \to \bar S$；且
* 在发生逃逸的条件下，信念以趋近于一的概率沿主导逃逸路径演变。

第一个结论是关于极限 $\varepsilon \to 0$ 的陈述。

{cite:t}`Williams2019`（脚注 16）将其写为 $\log \mathbb{E}\,\tau^\varepsilon \approx S_0 + \bar S/\varepsilon$，其中常数 $S_0$ 的取值并非该理论所能确定的。

因此，$\bar S$ 决定的是逃逸时间随增益缩小而增长的速度，而非其具体水平。

对于静态模型，CWS（表 1）得到 $\bar S \approx 5 \times 10^{-4}$。

在 {doc}`phillips_learning` 和 {ref}`olg_phillips` 所用的增益值（0.025 和 0.02）下，$e^{\bar S/\varepsilon} \approx 1.02$，因此仅凭大偏差因子本身并不能使逃逸变得罕见。

{ref}`en_simulation` 测量了逃逸实际发生的频率。

逃逸动态与均值动态一样，都是*确定性*的。

## 变测度逃逸路径

求解 {eq}`en_control` 需要矩阵 $Q$ 以及一个两点边值问题。

对于带有二项冲击的静态模型，CWS（第 7 节）用一种不同且更简单的论证方法来确定逃逸路径。

他们将冲击的概率测度变换为将信念推离 $\bar\gamma$ 的最可能的异常分布，并在该测度下对最小二乘更新取平均。

由此产生的逃逸强制项异常简单：

```{math}
:label: en_force

v = R^{-1} \begin{bmatrix} \sigma_1 \sigma_2 \\ 0 \end{bmatrix} ,
```

因此变测度逃逸路径求解

```{math}
:label: en_escape

\dot\gamma = R^{-1}\left( \bar g(\gamma) + \begin{bmatrix} \sigma_1 \sigma_2 \\ 0 \end{bmatrix} \right),
\qquad
\dot R = \bar M(\gamma) - R .
```

该路径与主导逃逸路径共享同一个退出点，但到达该点的速度更快。

CWS（脚注 14）警告说，变换后的测度会低估逃逸所需的时间。

在他们的图 7 中，变测度路径甚至比 1000 次模拟逃逸中最短的那条还要短。

让我们对 {eq}`en_escape` 从自证均衡开始积分，直到信念离开半径为 5 的圆圈——这是 CWS 在静态模型中使用的逃逸集合。

```{code-cell} ipython3
def escape_ode(t, z, model):
    γ, R = z[:2], z[2:].reshape(2, 2)
    force = np.array([model.σ1 * model.σ2, 0.0])
    return np.concatenate([np.linalg.solve(R, model.g_bar(γ) + force),
                           (model.M(γ) - R).ravel()])

def left_circle(t, z, *args):
    "Terminal event: beliefs leave the radius-5 circle around the SCE."
    return 5.0 - np.linalg.norm(z[:2] - γ_sce)
left_circle.terminal = True
left_circle.direction = -1

z0 = np.concatenate([γ_sce, R_bar.ravel()])
esc = solve_ivp(lambda t, z: escape_ode(t, z, model), [0, 200], z0,
                events=left_circle, max_step=0.02, rtol=1e-9, atol=1e-11)

slope, intercept = esc.y[0], esc.y[1]
infl = -intercept * slope / (1 + slope**2)
```

该图绘制了这两个系数以及它们所暗示的通货膨胀率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The changed-measure escape path and the inflation rate along it
    name: fig-esc-dominant-path
---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(esc.t, intercept, label='intercept $\\gamma_{-1}$')
axes[0].plot(esc.t, slope, label='slope $\\gamma_1$')
axes[0].axhline(0, color='C3', ls=':', lw=1, label='vertical curve ($\\gamma_1 = 0$)')
axes[0].set_xlabel('time $t = n\\varepsilon$')
axes[0].set_ylabel('coefficient')
axes[0].set_title('Changed-measure escape path (cf. CWS Figures 4 and 7)')
axes[0].legend()

axes[1].plot(esc.t, infl)
axes[1].axhline(model.θ * model.u, color='k', ls='--', lw=1, label='Nash')
axes[1].axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
axes[1].set_xlabel('time $t = n\\varepsilon$')
axes[1].set_ylabel('inflation $x$')
axes[1].set_title('Inflation along the escape')
axes[1].legend()

plt.tight_layout()
plt.show()
```

以下是该路径的起点和终点。

```{code-cell} ipython3
print(f"exit time : t = {esc.t[-1]:.3f}")
print(f"slope     : {slope[0]:.2f} → {slope[-1]:.3f}   (CWS Table 1, radius 5: -0.025)")
print(f"intercept : {intercept[0]:.1f} → {intercept[-1]:.2f}")
print(f"inflation : {infl[0]:.2f} → {infl[-1]:.2f}   (Nash {model.θ*model.u:.0f} → Ramsey 0)")
```

沿着逃逸路径，斜率从其自证均衡值 $-1$ 升高到接近零，而通货膨胀从纳什值降低到接近拉姆齐值。

该路径在斜率约为 $-0.023$ 时离开圆圈，接近 CWS（表 1）在相同半径下为主导路径报告的 $-0.025$。

它在 $t \approx 0.34$ 时离开，而 CWS 图 4 中的主导路径需要大约 $t = 3$ 才能离开：两条路径共享同一条路线，但时间表不同。

在静态模型中，接近零的斜率意味着感知的菲利普斯曲线呈垂直状态，这显然满足归纳假设。

{doc}`phillips_learning` 模拟的动态模型则以不同方式逃逸。

在该模型中，当期通货膨胀与滞后通货膨胀系数的*总和*升高到接近零，而当期通货膨胀的系数仍保持为负，因此短期菲利普斯曲线得以保留，而长期曲线则变为垂直（CWS，第 6.2 节和图 3）。

在这两个模型中，政府都偶然产生了足够多的通货膨胀*实验*，从而发现了一个足够好的自然率假说版本，并据此采取行动。

## 四条常微分方程的竞速

逃逸强制项 {eq}`en_force` 从何而来？

对于二项冲击 $W_{in} \in \{-1, +1\}$，每一期都会出现四种冲击实现 $w \in \{-1, 1\}^2$ 之一，每种的概率均为 $1/4$。

逃逸需要一连串持续将信念推向同一方向的更新。

CWS 证明，最高效的连续更新会反复抽取其更新处于同一半空间内的两种实现，每种的概率为 $1/2$；这样的组合共有四对。

对某一组合的更新取平均，便得到一个*条件*均值常微分方程，

```{math}
:label: en_cond

\dot\gamma = R^{-1}\left(\bar g(\gamma) + f_j(\gamma)\right),
\qquad
\dot R = \bar M(\gamma) - R,
```

其强制项为

$$
f_1 = \begin{bmatrix} \sigma_1\sigma_2 \\ 0 \end{bmatrix},
\quad
f_2 = \begin{bmatrix} -\sigma_1\sigma_2 \\ 0 \end{bmatrix},
\quad
f_3 = \sigma_1 \begin{bmatrix} x(\gamma) \\ 1 \end{bmatrix},
\quad
f_4 = -\sigma_1 \begin{bmatrix} x(\gamma) \\ 1 \end{bmatrix}
$$

分别对应组合 $\{(1,1), (-1,-1)\}$、$\{(1,-1), (-1,1)\}$、$\{(1,1), (1,-1)\}$ 以及 $\{(-1,1), (-1,-1)\}$。

$R$ 遵循无条件均值动态，因为 $\bar M$ 只依赖于 $W_2$，而在每个组合内 $W_2$ 都以 $1/2$ 的概率取各个值。

每条条件常微分方程都是一条候选逃逸路径。

每种冲击实现的概率都是 $1/4$，因此一连串更新出现的概率随其长度呈几何衰减，最先到达边界的候选路径给出最可能的逃逸点（CWS，第 7.2 节）。

我们先比较各候选路径在自我确认均衡处的速度，然后再进行竞速。

```{code-cell} ipython3
def cond_ode(t, z, model, f):
    "Conditional mean dynamics for one shock pair with forcing f(γ)."
    γ, R = z[:2], z[2:].reshape(2, 2)
    return np.concatenate([np.linalg.solve(R, model.g_bar(γ) + f(γ, model)),
                           (model.M(γ) - R).ravel()])

pairs = {
    "1: {(1,1),(-1,-1)}":  lambda γ, m: np.array([m.σ1 * m.σ2, 0.0]),
    "2: {(1,-1),(-1,1)}":  lambda γ, m: np.array([-m.σ1 * m.σ2, 0.0]),
    "3: {(1,1),(1,-1)}":   lambda γ, m: m.σ1 * np.array([m.x(γ), 1.0]),
    "4: {(-1,1),(-1,-1)}": lambda γ, m: -m.σ1 * np.array([m.x(γ), 1.0]),
}

race = {}
print(f"{'pair':22s} {'speed at SCE':>12s} {'exit time':>10s} {'exit γ':>17s} {'exit x':>7s}")
for name, f in pairs.items():
    speed = np.linalg.norm(np.linalg.solve(R_bar, f(γ_sce, model)))
    s = solve_ivp(cond_ode, [0, 50], z0, args=(model, f), events=left_circle,
                  max_step=0.02, rtol=1e-9, atol=1e-11)
    race[name] = s
    γ_end = s.y[:2, -1]
    if s.status == 1:
        print(f"{name:22s} {speed:12.2f} {s.t[-1]:10.3f}   ({γ_end[0]:+.3f}, {γ_end[1]:5.2f}) {model.x(γ_end):7.2f}")
    else:
        print(f"{name:22s} {speed:12.2f} {'never':>10s}   (rests at ({γ_end[0]:+.2f}, {γ_end[1]:.2f}))")
```

在自我确认均衡处，组合 1 和组合 2 打成平手：它们的初始速度大小相等、方向相反，且都远快于组合 3 和组合 4。

因此起点处的速度无法决出胜负，于是我们对这四条常微分方程进行积分求解。

组合 1 最先到达边界，所到之处接近拉姆齐信念。

组合 4 也在拉姆齐信念附近退出，但所需时间更长。

组合 2 是组合 1 的镜像，朝向*更高*的通货膨胀方向发展，需要长得多的时间才能到达边界。

组合 3 永远不会退出：它会在圆内的一个静止点处停下。

因此，组合 1 赢得了这场竞速，它所诱导出的逃逸强制项正是 {eq}`en_force` 中的 $R^{-1}(\sigma_1\sigma_2, 0)^\top$。

CWS 将获胜者的速度描述为远大于其余三者；而上表显示，这一点只在逃逸路径沿途成立，而非在起点处成立。

(en_vector_field)=
## 均值动态强化逃逸

为什么朝向拉姆齐的逃逸能够如此迅速地成功，而其镜像却步履蹒跚？

在自我确认均衡附近，均值动态指*向*该均衡，对抗任何逃逸。

但 CWS（第 6.4 节，图 8-9）表明，一旦信念沿逃逸路径移动了一小段距离，均值动态本身就开始指*向*拉姆齐，从而强化了逃逸。

原因在于 $R$ 滞后于 $\bar M(\gamma)$。

在逃逸过程中，信念 $\gamma$ 移动很快，而 $R$ 仅以速率 1 朝 $\bar M(\gamma)$ 调整，因此 $R$ 大致仍保留着纳什时期的矩，即 $\bar R = \bar M(\bar\gamma)$。

在自我确认均衡处，通货膨胀几乎不变，因此 $\det \bar R = \sigma_2^2$ 很小，$\bar R^{-1}$ 很大。

用 $\bar R^{-1}$ 而非 $\bar M(\gamma)^{-1}$ 左乘漂移项 $\bar g(\gamma) = \bar M(\gamma)(T(\gamma) - \gamma)$，既会旋转又会放大它。

CWS（第 6.5 节）用文字描述了同样的自我强化过程，并补充道："对近期观测值赋予过高权重有助于推动这一过程。"

为了说明这一点，我们绘制信念空间中归一化的均值动态漂移项 $R^{-1}\bar g(\gamma)$，一次将 $R$ 固定在其纳什时期的值 $\bar R$，另一次让 $R$ 追上 $\bar M(\gamma)$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: R 滞后（左）和 R 已追上（右）时的均值动态漂移，以及逃逸路径
    name: fig-esc-vector-field
---
gs = np.linspace(-1.2, 0.1, 16)      # slope
gi = np.linspace(4.5, 10.5, 16)      # intercept
GS, GI = np.meshgrid(gs, gi)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
titles = [r'$R = \bar M(\bar\gamma)$: $R$ lags', r'$R = \bar M(\gamma)$: $R$ caught up']
for ax, lagging, title in zip(axes, [True, False], titles):
    DS, DI = np.zeros_like(GS), np.zeros_like(GI)
    for i in range(GS.shape[0]):
        for j in range(GS.shape[1]):
            γ = np.array([GS[i, j], GI[i, j]])
            R = R_bar if lagging else model.M(γ)
            d = np.linalg.solve(R, model.g_bar(γ))
            DS[i, j], DI[i, j] = d / np.linalg.norm(d)
    ax.quiver(GS, GI, DS, DI, angles='xy', alpha=0.7)
    ax.plot(*γ_sce, 'ko', ms=8, label='SCE (Nash)')
    ax.plot(0, model.u, 'C2s', ms=8, label='Ramsey belief')
    ax.plot(slope, intercept, 'C3', lw=2, label='escape path')
    ax.set_xlabel(r'slope $\gamma_1$')
    ax.set_title(title)
axes[0].set_ylabel(r'intercept $\gamma_{-1}$')
axes[0].legend(loc='lower left')
plt.tight_layout()
plt.show()
```

在左图中，当 $R$ 固定在其纳什时期的值时，沿逃逸路径的箭头朝拉姆齐信念方向扫动。

在右图中，当 $R$ 已经追上时，箭头在各处都指回斜率 $-\theta$ 的方向。

此时漂移项为 $T(\gamma) - \gamma$，其斜率分量 $-\theta - \gamma_1$ 在 $\gamma_1 > -\theta$ 时总是为负。

因此，这种强化效应是 $\gamma$ 与 $R$ *联合*动态的一个属性；正是同样的滞后造就了 {numref}`fig-esc-mean-dynamics` 中的凹陷。

在逃逸路径本身上，$R$ 仅朝 $\bar M(\gamma)$ 移动了一部分，因为逃逸只占据单位时间的一小部分；{ref}`en_ex2` 衡量了其与路径自身 $R$ 的一致程度。

逃逸动态只需要*启动*这一背离过程；均值动态负责完成剩余的工作。

正是在这个意义上，均值动态描绘出一条*迂回*的路线：被逃逸路径推离均衡后，系统会在拉姆齐附近停留一段时间，直到 $R$ 追上，残留的短期菲利普斯曲线被重新发现，均值动态最终将信念带回纳什水平。

(en_trap)=
## 试验陷阱

这种逃逸具有一种令人信服的行为解释。

在其近似模型内部，政府只有当通货膨胀存在足够的*离散度*时，才能检测到自然率假说。

但在自我确认均衡内，政府设定了一个恒定的系统性通货膨胀率，因此不会产生这种离散度：它被困在一个**试验陷阱**中。

只有一段异常的冲击序列，才会使政府充分改变通货膨胀，从而使其估计的菲利普斯曲线变得更陡；一条更陡峭的感知曲线（通过最优反应）会引导它削减通货膨胀，这又会产生进一步的有影响力的观测值，使曲线进一步变陡。

正如 CWS 所指出的，对近期观测值赋予更高权重会助推这一过程。

这种自我强化的过程会在感知的菲利普斯曲线变得垂直时停止，通货膨胀接近拉姆齐水平。

系统无法永远停留在那里：事实上*确实*存在一条短期菲利普斯曲线，政府最终会重新发现它，从而重新点燃将其带回纳什水平的均值动态。

(en_simulation)=
## 模拟逃逸

大偏差结果描述的是极限 $\varepsilon \to 0$ 的情形。

为了看清这些结果在该程序包实际使用的增益下意味着什么，我们在常数增益递归式 {eq}`en_rls` 下模拟静态模型。

每条路径都从自我确认均衡开始，即 $R_0 = \bar M(\bar\gamma)$。

```{code-cell} ipython3
def simulate_path(model, ε, n_periods, seed=0):
    "One path of the static model under constant-gain RLS (en_rls), started at the SCE."
    rng = np.random.default_rng(seed)
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ = model.γ_sce()
    R = model.M(γ)
    x_path, γ_path = np.empty(n_periods), np.empty((n_periods, 2))
    for n in range(n_periods):
        x = model.x(γ)
        w1, w2 = rng.standard_normal(2)
        π = x + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.array([π, 1.0])
        R = R + ε * (np.outer(Φ, Φ) - R)
        γ = γ + ε * np.linalg.solve(R, Φ * (U - Φ @ γ))
        x_path[n], γ_path[n] = x, γ
    return x_path, γ_path
```

我们使用 $\varepsilon = 0.02$，即 {ref}`olg_phillips` 中的增益；{doc}`phillips_learning` 使用的是 $0.025$，即该书所用的 $\lambda = .975$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: A simulated path of the static model with gain 0.02
    name: fig-esc-sim-path
---
ε = 0.02
x_path, γ_path = simulate_path(model, ε, 4_000, seed=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(x_path, lw=0.8)
axes[0].axhline(model.θ * model.u, color='k', ls='--', lw=1, label='Nash')
axes[0].axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
axes[0].set_ylabel('target inflation $x_n$')
axes[0].legend(loc='upper right')
axes[1].plot(γ_path[:, 0], lw=0.8, color='C1')
axes[1].axhline(-model.θ, color='k', ls='--', lw=1)
axes[1].set_ylabel(r'slope $\gamma_{1n}$')
axes[1].set_xlabel('period $n$')
plt.tight_layout()
plt.show()
```

和 {ref}`olg_phillips` 中的长期模拟一样，该路径呈现出一种循环往复的锯齿状：向零通货膨胀方向的突然逃逸，紧随其后的是向纳什均衡缓慢爬升的过程，这一过程常常又被下一次逃逸打断。

更长时间的模拟运行表明，政府在纳什均衡附近停留的时间非常短。

```{code-cell} ipython3
x_long, _ = simulate_path(model, ε, 20_000, seed=2)
print(f"ε = {ε}, 20,000 periods:")
print(f"  share of periods with x > 4: {np.mean(x_long > 4):.2f}")
print(f"  share of periods with x < 2: {np.mean(x_long < 2):.2f}")
print(f"  mean target inflation:       {x_long.mean():.2f}   (Nash {model.θ * model.u:.0f})")
```

目标通货膨胀仅在很少一部分时期内高于4。

为了与定理相联系，我们测量了从自我确认均衡出发、多条独立路径首次脱离半径为5的圆的首次退出时间。

```{code-cell} ipython3
def first_exits(model, ε, n_paths=400, radius=5.0, seed=0):
    """
    Simulate n_paths independent copies of (en_rls) from the SCE.  Return each
    path's first exit period from the circle of the given radius and its beliefs
    at exit (NaN if it has not exited after 100/ε periods).
    """
    rng = np.random.default_rng(seed)
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ_bar = model.γ_sce()
    γ = np.tile(γ_bar, (n_paths, 1))
    R = np.tile(model.M(γ_bar), (n_paths, 1, 1))
    τ = np.full(n_paths, np.nan)
    γ_exit = np.full((n_paths, 2), np.nan)
    alive = np.ones(n_paths, dtype=bool)
    n = 0
    while alive.any() and n < 100 / ε:
        n += 1
        x = -γ[:, 1] * γ[:, 0] / (1 + γ[:, 0]**2)
        w1, w2 = rng.standard_normal((2, n_paths))
        π = x + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.stack([π, np.ones(n_paths)], axis=1)
        R += ε * (Φ[:, :, None] * Φ[:, None, :] - R)
        e = U - (Φ * γ).sum(axis=1)
        γ += ε * np.linalg.solve(R, (Φ * e[:, None])[:, :, None])[:, :, 0]
        out = alive & (np.linalg.norm(γ - γ_bar, axis=1) > radius)
        τ[out], γ_exit[out] = n, γ[out]
        alive &= ~out
    return τ, γ_exit


S_bar = 4.987e-4        # CWS Table 1: static model, normal shocks, radius 5

print(f"{'ε':>6s} {'mean':>7s} {'median':>7s} {'ε·mean':>7s} {'exp(S̄/ε)':>9s} "
      f"{'x at exit < 1':>13s} {'not exited':>10s}")
for ε in [0.1, 0.05, 0.03, 0.02, 0.01, 0.005]:
    τ, γ_exit = first_exits(model, ε)
    done = ~np.isnan(τ)
    x_exit = -γ_exit[done, 1] * γ_exit[done, 0] / (1 + γ_exit[done, 0]**2)
    print(f"{ε:6.3f} {np.mean(τ[done]):7.1f} {np.median(τ[done]):7.0f} "
          f"{ε * np.mean(τ[done]):7.2f} {np.exp(S_bar / ε):9.3f} "
          f"{np.mean(x_exit < 1):13.2f} {np.sum(~done):10d}")
print(f"\nexp(S̄/ε) at ε = 0.0002: {np.exp(S_bar / 0.0002):.1f}")
```

平均首次退出时间从 $\varepsilon = 0.1$ 时约16期，到 $\varepsilon = 0.005$ 时约400期不等。

在 $\varepsilon = 0.01$ 和 $0.005$ 时，这些数值接近 {cite:t}`SargentWilliams2005`（表1）中针对常数增益最小二乘法报告的165.7期和409.6期。

每条路径都在拉姆齐一侧退出，且退出点接近逃逸路径的退出点。

乘积 $\varepsilon \times$ 均值保持在大约1.4到2之间，因此平均退出时间大致按 $1/\varepsilon$ 的规律变化。

相比之下，因子 $e^{\bar S/\varepsilon}$ 在表中从未超过1.11。

在这些增益水平下，大偏差因子几乎不起作用；退出时间由常数 $S_0$ 决定，也就是说，由信念偏离均衡并沿路径游走所需的时间决定。

**在该程序包所使用的增益水平下，逃逸是频繁发生的，而非罕见事件。**

从自我确认均衡出发，增益为 $\varepsilon$ 的政府平均在大约 $1.5/\varepsilon$ 到 $2/\varepsilon$ 期后发生逃逸，且在两次逃逸之间，它很少能完全回到纳什均衡。

罕见性是一种小增益条件下的陈述。

只有当增益的量级与 $\bar S$ 本身相当时，指数因子才会产生显著影响：在 $\varepsilon = 0.0002$（比 {cite:t}`SargentWilliams2005` 表1中最小增益还要小五倍）时，$e^{\bar S/\varepsilon} \approx 12$。

模拟得到的退出时间也印证了CWS脚注14的说法。

若以 $t = n\varepsilon$ 为单位度量，这些退出时间是变换测度路径所需的 $0.34$ 的数倍；{ref}`en_ex5` 对两者进行了详细比较。

## 逃逸频率与模型丰富度

被最小化的行动值 $\bar S$ 决定了随增益缩小，逃逸时间增长的指数速率。

CWS 的一个引人注目的发现是，当政府的模型更加丰富时，逃逸会*更加频繁*。

CWS 的表 1 给出，{doc}`phillips_learning` 中带有滞后失业率与滞后通货膨胀的动态模型，其速率 $\bar S \approx 9.7 \times 10^{-6}$，约为静态模型的 $5.0 \times 10^{-4}$ 的五十分之一，尽管两者具有相同的自我确认均衡。

这个动态模型的数值需要一个说明：它是用 {cite:t}`Williams2019` 所纠正的 2001 年捷径方法计算得到的，而这一点对动态模型而言十分重要。

不过，这一比较并非仅仅依赖于这个数值。

在 CWS 的模拟中（其图 5），动态模型在每一种增益下的平均逃逸时间都更短，{ref}`en_ex6` 对首次逃逸时间的中位数也得出了相同的结论。

一个更丰富的模型使政府能够检测到自然率假说更为微妙的分布滞后（"归纳假说"）版本，因此它更容易朝拉姆齐方向逃逸。

```{note}
逃逸动态继承了使均值动态如此有用的同一种"近似确定性"：
对于较小的增益，模拟的逃逸会聚集在确定性逃逸路径周围，这无论是在动态模型中
（CWS，图 6）还是在静态模型中（{ref}`en_ex5`；CWS，图 7）都成立。

下一讲 {doc}`phillips_priors` 表明，政府关于其系数如何漂移的*先验*
会重塑这两种动态，甚至可能使逃逸变成一个确定性的
*循环*。
```

## 逃离高波动通货膨胀

逃逸带来的是通货膨胀*水平*的下降。

{cite:t}`EllisonYates2007` 扩展了该模型，以解释战后的第二个事实：通货膨胀*波动性*与通货膨胀水平同升同降。

他们的方法是给政府一个稳定化的动机。

沿用 {cite:t}`PhelpsTaylor1977` 的思路，他们加入了一个失业冲击 $W_3$，政府可以对其做出反应，而制定价格的私人部门则不能。

现在，一个相信存在可利用的菲利普斯曲线的政府，倾向于通过改变通货膨胀来*对抗* $W_3$，因此政策的感知有效性 $|\gamma_1|$ 不仅驱动通货膨胀的水平，也驱动其波动性。

在他们的模型中，私人主体所面临的预期通货膨胀波动性为

```{math}
:label: en_vol

\mathbb{E}(\sigma_\pi \mid \gamma) = \left[ \sigma_2^2 + \left(\frac{\gamma_1}{1 + \gamma_1^2}\right)^2 \sigma_3^2 \right]^{1/2} .
```

在自我确认均衡 $\gamma_1 = -\theta$ 处，政府相信政策是有效的，并积极对抗 $W_3$，因此通货膨胀是*波动的*。

沿着逃逸路径，$\gamma_1 \to 0$：政府不再相信自己能够利用菲利普斯曲线，放弃了稳定化操作，波动性项收缩为控制误差 $\sigma_2$。

作为一种近似，我们将 {eq}`en_vol` 应用于上文中针对 $\sigma_3 = 0$ 所计算出的信念路径。

在埃利森和耶茨的模型中，正的 $\sigma_3$ 也会改变逃逸本身，我们将在下文讨论这一点，但该近似已能展示出通货膨胀的水平与波动性*同步*逃逸。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 通货膨胀的水平与波动性同步逃逸
    name: fig-esc-volatility
---
σ3 = 0.9                                   # size of the stabilizable shock
infl_vol = np.sqrt(model.σ2**2 + (slope / (1 + slope**2))**2 * σ3**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(esc.t, infl)
axes[0].axhline(model.θ * model.u, color='k', ls='--', lw=1, label='Nash')
axes[0].axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
axes[0].set_xlabel('time $t = n\\varepsilon$'); axes[0].set_ylabel('inflation level')
axes[0].set_title('level escapes'); axes[0].legend()

axes[1].plot(esc.t, infl_vol, 'C1')
axes[1].axhline(model.σ2, color='k', ls='--', lw=1, label=r'control error $\sigma_2$')
axes[1].set_xlabel('time $t = n\\varepsilon$'); axes[1].set_ylabel('inflation volatility')
axes[1].set_title('volatility escapes too'); axes[1].legend()

plt.tight_layout()
plt.show()
```

随着信念的逃逸，二者都会下降，因为它们都源自同一个根源：政府对存在可利用权衡取舍的信念。

{cite:t}`EllisonYates2007` 还就逃逸的*时机*得出了一个更为微妙的进一步结论。

一个更大的可稳定冲击 $\sigma_3$ 会使逃逸*更难触发*：为了制造出通货膨胀在变动而失业率保持不变的假象，一段异常的冲击序列现在不仅要抵消控制误差，还必须抵消政府自身针对 $W_3$ 的稳定化反应。

政府能够抵消的冲击越多，触发逃逸所需的序列就越复杂，等待时间也就越长。

埃利森和耶茨由此得出结论：在政策可以抵消的冲击相对较少的时期，更有可能出现逃向低通胀的情况。

人们很容易——这是我们自己的推断，而非他们的原文——将二十世纪八十年代中期的经济平静解读为这样一个时期，并将其与反通胀联系起来。

但时间上的吻合并不那么容易成立。

在 {doc}`phillips_drifts_volatilities` 的估计结果中，核心通货膨胀在1980年至1982年间下降，而冲击方差的大部分下降却发生在二十世纪八十年代中期。

反通胀出现得更早，恰恰发生在冲击异常巨大的时期。

那一讲构建了一个统计模型，可以同时容纳*冲击的缩小*与*动态的转变*，并让数据来判定二者各自的贡献。

在转向数据之前，{doc}`phillips_priors` 将回到静态模型，探讨政府关于系数漂移的先验信念如何改变以上的一切结论。

## 练习

```{exercise-start}
:label: en_ex1
```

在自我确认均衡处，矩矩阵为 $\bar R = \bar M(\bar\gamma)$，且 $\det \bar R = \sigma_2^2$。

1. 证明在自我确认均衡处的逃逸速度为 $\bar R^{-1}(\sigma_1\sigma_2, 0)^\top = (\sigma_1/\sigma_2)(1, -\bar x)^\top$，其中 $\bar x = \theta u$。因此当 $\sigma_1 = \sigma_2$ 时，该速度与冲击的大小无关。
2. 对 $\sigma_1 = \sigma_2 = \sigma \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ 积分逃逸常微分方程 {eq}`en_escape`，并报告从半径为 5 的圆中退出的时间。
3. 尽管如此，退出时间仍随 $\sigma$ 的增大而上升。请解释原因。提示：将 $\bar M(\gamma)$ 写成 $\bar R + \Delta(\gamma)$，并观察 $\Delta$ 与 $\bar R^{-1}$ 如何依赖于 $\sigma_2$。
4. 现在逐个改变冲击：取 $\sigma_2 \in \{0.2, 0.3, 0.45\}$，固定 $\sigma_1 = 0.3$；再取 $\sigma_1 \in \{0.2, 0.3, 0.45\}$，固定 $\sigma_2 = 0.3$。

```{exercise-end}
```

```{solution-start} en_ex1
:class: dropdown
```

对于第 1 部分，$\bar R = \begin{bmatrix} \bar x^2 + \sigma_2^2 & \bar x \\ \bar x & 1 \end{bmatrix}$ 的行列式为 $\sigma_2^2$，其逆矩阵为 $\sigma_2^{-2}\begin{bmatrix} 1 & -\bar x \\ -\bar x & \bar x^2 + \sigma_2^2 \end{bmatrix}$，因此 $\bar R^{-1}(\sigma_1\sigma_2, 0)^\top = (\sigma_1/\sigma_2)(1, -\bar x)^\top$。

对于第 2 和第 4 部分，我们对每一对冲击大小分别积分逃逸常微分方程。

```{code-cell} ipython3
def exit_time(σ1, σ2):
    m = EscapeModel(θ=1.0, u=5.0, σ1=σ1, σ2=σ2)
    γ0 = m.γ_sce()

    def leave(t, z):
        return 5.0 - np.linalg.norm(z[:2] - γ0)
    leave.terminal = True
    leave.direction = -1

    v0 = np.linalg.solve(m.M(γ0), np.array([σ1 * σ2, 0.0]))
    s = solve_ivp(lambda t, z: escape_ode(t, z, m), [0, 500],
                  np.concatenate([γ0, m.M(γ0).ravel()]),
                  events=leave, max_step=0.02, rtol=1e-9, atol=1e-11)
    return v0, s.t[-1]

for σ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    v0, T = exit_time(σ, σ)
    print(f"σ1 = σ2 = {σ}: velocity at SCE = {v0.round(3)}, exit time = {T:.3f}")
print()
for σ2 in [0.2, 0.3, 0.45]:
    print(f"σ1 = 0.3, σ2 = {σ2}: exit time = {exit_time(0.3, σ2)[1]:.3f}")
for σ1 in [0.2, 0.3, 0.45]:
    print(f"σ1 = {σ1}, σ2 = 0.3: exit time = {exit_time(σ1, 0.3)[1]:.3f}")
```

当 $\sigma_1 = \sigma_2$ 时，每个 $\sigma$ 对应的起始速度都相同，然而退出时间却随 $\sigma$ 稳步上升。

这种速度来自 {ref}`en_vector_field` 中所说的 $R$ 滞后放大效应。

在逃逸过程中，$R$ 一直接近 $\bar R$，因此均值漂移为 $\bar R^{-1}\bar M(\gamma)(T(\gamma) - \gamma) = (T(\gamma) - \gamma) + \bar R^{-1}\Delta(\gamma)(T(\gamma) - \gamma)$。

这里 $\Delta(\gamma) = \begin{bmatrix} x^2 - \bar x^2 & x - \bar x \\ x - \bar x & 0 \end{bmatrix}$ 只取决于通货膨胀偏离了多远，而与 $\sigma_2$ 无关，而 $\bar R^{-1}$ 的量级为 $1/\sigma_2^2$。

因此，控制误差越小，滞后的矩矩阵就越强烈地放大任何偏离纳什均衡的移动，信念沿该路线移动得也就越快。

这就是为什么在第 4 部分中 $\sigma_2$ 的影响如此之大，而更大的 $\sigma_1$ 只是通过初始推动力 $\sigma_1/\sigma_2$ 加快逃逸速度。

这一切都与逃逸的*频率*无关，后者由 {ref}`en_simulation` 衡量；一个噪声更大的经济体，一旦逃逸开始，走完确定性路线反而会更慢。

```{solution-end}
```

```{exercise-start}
:label: en_ex2
```

将向量场图中的强化效应量化。

在逃逸路径*沿途*的若干点上，使用路径自身的 $R$，计算均值动态漂移 $R^{-1}\bar g(\gamma)$，并测量它与当前信念指向拉姆齐信念 $(0, u)$ 方向之间的余弦对齐度。

余弦值接近 $+1$ 意味着均值动态正将信念推向拉姆齐，强化了逃逸。

```{exercise-end}
```

```{solution-start} en_ex2
:class: dropdown
```

我们在逃逸路径上的五个点处计算漂移。

```{code-cell} ipython3
γ_ramsey = np.array([0.0, model.u])            # Belief 2

def cosine_toward_ramsey(γ, R):
    drift = np.linalg.solve(R, model.g_bar(γ))
    to_ramsey = γ_ramsey - γ
    denom = np.linalg.norm(drift) * np.linalg.norm(to_ramsey)
    return np.nan if denom < 1e-9 else drift @ to_ramsey / denom

for frac in [0.0, 0.25, 0.5, 0.75, 0.95]:
    k = min(int(frac * len(esc.t)), len(esc.t) - 1)
    γ, R = esc.y[:2, k], esc.y[2:, k].reshape(2, 2)
    print(f"frac {frac:.2f}:  γ = ({γ[0]:+.2f}, {γ[1]:.1f}),  "
          f"cos(drift, →Ramsey) = {cosine_toward_ramsey(γ, R):+.2f},  "
          f"with R = M(γ): {cosine_toward_ramsey(γ, model.M(γ)):+.2f}")
```

正处于自我确认均衡时，漂移为零（余弦值未定义），因此均值动态既不助力也不阻碍逃逸。

但一旦信念沿逃逸路线移动了哪怕一点点，均值动态的漂移方向就几乎精确地指向拉姆齐信念（余弦值 $\approx +1$）：均值动态一路强化逃逸，直至到达拉姆齐水平。

最后一列将路径自身的 $R$ 替换为 $\bar M(\gamma)$，此时余弦值变为负值：可见这种强化效应正是来自 $R$ 的滞后。

CWS 所强调的对抗力量，仅局限于均衡附近的一个*极小*邻域；逃逸动态只需要将信念推出这个邻域，之后均值动态便会完成剩余的工作。

```{solution-end}
```

```{exercise-start}
:label: en_ex3
```

更仔细地观察四个常微分方程之间的竞赛。

1. 在 $(\gamma_1, \gamma_{-1})$ 平面上绘制四条条件路径 `race`，并画出半径为 5 的圆，同时绘制沿每条路径的通货膨胀变化。
2. CWS 给出了条件常微分方程第 3 对和第 4 对的静止点为 $(-\theta,\ (u \pm \sigma_1)(1+\theta^2))$，两者都位于圆内。第 3 对会稳定在该点。为什么第 4 对却会逃逸？
3. 将 $R$ 固定为 $\bar M(\gamma)$ 重新运行这场竞赛，就好像矩矩阵能瞬间调整一样。此时哪一对获胜，退出点又在哪里？

```{exercise-end}
```

```{solution-start} en_ex3
:class: dropdown
```

我们首先绘制正文中计算得到的四条路径。

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
φ = np.linspace(0, 2 * np.pi, 200)
axes[0].plot(γ_sce[0] + 5 * np.cos(φ), γ_sce[1] + 5 * np.sin(φ), 'k:', lw=1)
for name, s in race.items():
    axes[0].plot(s.y[0], s.y[1], label=name)
    axes[1].plot(s.t, -s.y[1] * s.y[0] / (1 + s.y[0]**2), label=name)
axes[0].plot(*γ_sce, 'ko')
axes[0].set_xlabel(r'slope $\gamma_1$')
axes[0].set_ylabel(r'intercept $\gamma_{-1}$')
axes[1].set_xscale('log')
axes[1].set_xlabel('time $t$ (log scale)')
axes[1].set_ylabel('inflation $x$')
axes[0].legend()
plt.tight_layout()
plt.show()
```

接下来，我们在强制 $R = \bar M(\gamma)$ 的条件下重新运行这场竞赛，这会将每个条件常微分方程化简为 $\dot\gamma = T(\gamma) - \gamma + \bar M(\gamma)^{-1} f_j(\gamma)$。

```{code-cell} ipython3
def cond_ode_no_lag(t, γ, model, f):
    return np.linalg.solve(model.M(γ), model.g_bar(γ) + f(γ, model))

def left_circle_γ(t, γ, *args):
    return 5.0 - np.linalg.norm(γ - γ_sce)
left_circle_γ.terminal = True
left_circle_γ.direction = -1

for name, f in pairs.items():
    s = solve_ivp(cond_ode_no_lag, [0, 50], γ_sce, args=(model, f), events=left_circle_γ,
                  max_step=0.02, rtol=1e-9, atol=1e-11)
    γ_end = s.y[:, -1]
    status = f"exits at t = {s.t[-1]:.3f}" if s.status == 1 else "never exits"
    print(f"{name:22s} {status:22s} γ_end = ({γ_end[0]:+.3f}, {γ_end[1]:.2f}), "
          f"x = {model.x(γ_end):.2f}")
```

当 $R = \bar M(\gamma)$ 时，每个条件常微分方程的静止点都是经过平移的 $T$ 映射的不动点。

对于第 4 对，该点为 $(-\theta, (u - \sigma_1)(1+\theta^2))$；漂移在该点处的雅可比矩阵同样具有特征值 $-1$ 和 $-1/(1+\theta^2)$，因此路径会收敛到该点。

而在真实的、带有滞后的 $R$ 下，同样的强制项会先使截距和通货膨胀略微降低；随后滞后的矩矩阵会将这一移动放大为彻底逃向拉姆齐均衡。

去掉滞后之后，竞赛的结果完全改变了。

第 1 对仍朝拉姆齐方向移动，但只是缓慢地漂移过去，而第 2 对现在率先退出，朝着*更高*的通货膨胀方向。

因此，$R$ 的滞后同时决定了逃逸的方向和速度：正是它使得通往拉姆齐的路线成为最可能发生的逃逸路径。

```{solution-end}
```

```{exercise-start}
:label: en_ex4
```

该定理表明 $\varepsilon \log \mathbb{E}\,\tau^\varepsilon \to \bar S$，其中 $\tau^\varepsilon = n\varepsilon$ 是常微分方程时间尺度下的退出时间；等价地，$\log \mathbb{E}\,\tau^\varepsilon \approx S_0 + \bar S/\varepsilon$。

1. 使用 `first_exits` 计算 $\varepsilon \in \{0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001\}$ 下的平均退出周期。
2. 将 $\log(\varepsilon \times \text{平均退出周期})$ 对 $1/\varepsilon$ 作图，并将斜率与 CWS 的 $\bar S$ 相比较。
3. 将平均退出周期与 {cite:t}`SargentWilliams2005`（表 1）相比较，后者报告在 $\varepsilon = 0.05, 0.01, 0.005, 0.002, 0.001$ 时分别为 20.6、165.7、409.6、2414.8 和 4486.7 个周期。

```{exercise-end}
```

```{solution-start} en_ex4
:class: dropdown
```

我们在更精细的小增益网格上计算平均退出周期。

```{code-cell} ipython3
gains = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001])
mean_exit = np.array([np.nanmean(first_exits(model, ε, seed=1)[0]) for ε in gains])
log_scaled = np.log(gains * mean_exit)

for ε, m, l in zip(gains, mean_exit, log_scaled):
    print(f"ε = {ε:6.4f}: mean exit period = {m:7.0f},  ε·mean = {ε * m:5.2f}")
local_slope = np.diff(log_scaled) / np.diff(1 / gains)
print("\nlocal slopes of log(ε·mean) in 1/ε:", local_slope.round(5))
print(f"CWS S̄ = {S_bar:.5f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(1 / gains, log_scaled, 'o-', label='simulation')
ax.plot(1 / gains, log_scaled[-1] + S_bar * (1 / gains - 1 / gains[-1]), 'k--',
        label=r'slope $\bar S$ (CWS), through last point')
ax.set_xlabel(r'$1/\varepsilon$')
ax.set_ylabel(r'$\log(\varepsilon \times$ mean exit period$)$')
ax.legend()
plt.show()
```

在增益跨越百倍范围的情况下，缩放后的均值 $\varepsilon \times$ 平均值始终保持在大约 1.4 到 4.6 之间。

在这一范围内，$\bar S/\varepsilon$ 仅从 0.005 升至 0.5，因此 $\log \mathbb{E}\,\tau^\varepsilon$ 的水平主要由常数 $S_0$ 所主导。

在较大的增益下，曲线甚至向下倾斜，只有在最小的增益下，其斜率才接近 $\bar S$ 的量级：该定理所描述的渐近机制，出现在远比本套件所用增益更小的增益水平上。

模拟得到的均值在 $\varepsilon = 0.01$、$0.005$ 和 $0.001$ 时与 SW 的结果高度吻合。

```{solution-end}
```

```{exercise-start}
:label: en_ex5
```

CWS 的图 7 将模拟逃逸与主导路径及变测度路径进行了比较。

1. 从自我确认均衡出发，在 $\varepsilon = 0.005$ 下模拟 50 条路径，记录信念直至每条路径首次离开半径为 5 的圆。
2. 将每条路径的斜率和截距对 $\log n$ 作图，并与 $n = t/\varepsilon$ 下的变测度路径一同绘制。
3. 比较退出点、总退出时间，以及最后一轮运行的持续时间（从斜率最后一次低于 $-0.9$ 的周期开始计算）。

```{exercise-end}
```

```{solution-start} en_ex5
:class: dropdown
```

我们记录每条路径直至退出为止的完整信念历史。

```{code-cell} ipython3
def escape_histories(model, ε, n_paths=50, n_max=10_000, radius=5.0, seed=0):
    rng = np.random.default_rng(seed)
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ_bar = model.γ_sce()
    γ = np.tile(γ_bar, (n_paths, 1))
    R = np.tile(model.M(γ_bar), (n_paths, 1, 1))
    hist = np.full((n_max + 1, n_paths, 2), np.nan)
    hist[0] = γ
    τ = np.zeros(n_paths, dtype=int)
    alive = np.ones(n_paths, dtype=bool)
    for n in range(1, n_max + 1):
        x = -γ[:, 1] * γ[:, 0] / (1 + γ[:, 0]**2)
        w1, w2 = rng.standard_normal((2, n_paths))
        π = x + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.stack([π, np.ones(n_paths)], axis=1)
        R += ε * (Φ[:, :, None] * Φ[:, None, :] - R)
        e = U - (Φ * γ).sum(axis=1)
        γ += ε * np.linalg.solve(R, (Φ * e[:, None])[:, :, None])[:, :, 0]
        hist[n, alive] = γ[alive]
        out = alive & (np.linalg.norm(γ - γ_bar, axis=1) > radius)
        τ[out] = n
        alive &= ~out
        if not alive.any():
            break
    return hist[:n + 1], τ

ε = 0.005
hist, τ = escape_histories(model, ε, seed=3)
print(f"all exited: {np.all(τ > 0)}")
print(f"simulated exit periods: min {τ.min()}, median {np.median(τ):.0f}, max {τ.max()}")
print(f"changed-measure path:   exits at n = t/ε = {esc.t[-1] / ε:.0f}")

γ_end = hist[τ, np.arange(len(τ))]
print(f"exit slope: simulated mean {γ_end[:, 0].mean():+.3f}, "
      f"changed-measure {slope[-1]:+.3f}")

final_run = np.array([(τ[i] - np.where(hist[:τ[i] + 1, i, 0] < -0.9)[0].max()) * ε
                      for i in range(len(τ))])
k_cm = np.where(slope < -0.9)[0].max()
print(f"final run (in t = nε): simulated median {np.median(final_run):.2f}, "
      f"changed-measure {esc.t[-1] - esc.t[k_cm]:.2f}")
```

接下来，我们在 CWS 的对数时间尺度上绘制这些路径。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(len(τ)):
    n_i = np.arange(1, τ[i] + 1)
    ax.plot(np.log(n_i), hist[1:τ[i] + 1, i, 1], color='C0', lw=0.4, alpha=0.5)
    ax.plot(np.log(n_i), hist[1:τ[i] + 1, i, 0], color='C1', lw=0.4, alpha=0.5)
n_cm = esc.t[1:] / ε
ax.plot(np.log(n_cm), intercept[1:], 'k--', lw=2, label='changed measure: intercept')
ax.plot(np.log(n_cm), slope[1:], 'k-.', lw=2, label='changed measure: slope')
ax.set_xlabel(r'$\log n$')
ax.set_ylabel('coefficient')
ax.set_title(f'50 simulated escapes (ε = {ε}) and the changed-measure path')
ax.legend()
plt.show()
```

正如定理 5.3 对主导路径所预言的那样，模拟出的逃逸几乎在与变测度路径完全相同的点上离开圆。

但正如 CWS 所发现的那样，变测度路径的退出要早于即使是最快的模拟逃逸路径。

这其中的大部分差异来自等待时间：一条模拟路径在最后一轮运行之前，会在自我确认均衡附近停留相当长的一段时间。

而一旦这最后一轮运行开始，所花费的时间与变测度路径的量级相同。

```{solution-end}
```

```{exercise-start}
:label: en_ex6
```

CWS 发现，在其动态模型中逃逸比在静态模型中更为频繁。

通过模拟来验证这一点。

在 {doc}`phillips_learning` 的动态模型中（其中用 $y$ 表示通货膨胀），政府将 $U_n$ 对 $(\pi_n, U_{n-1}, U_{n-2}, \pi_{n-1}, \pi_{n-2}, 1)$ 进行回归，并通过求解折现率 $\delta = 0.98$ 的折现菲尔普斯问题来设定通货膨胀。

1. 使用递归式 {eq}`en_rls`（此时有六个回归元）模拟该动态模型，从其自我确认均衡出发，设 $R_0$ 等于该自我确认均衡下的矩矩阵。
2. 将第一次逃逸定义为通货膨胀的 20 期移动平均首次跌破 1 的那一期。
3. 对于 $\varepsilon \in \{0.002, 0.005, 0.01, 0.02\}$，报告两个模型在多条路径下第一次逃逸时间的中位数。

```{exercise-end}
```

```{solution-start} en_ex6
:class: dropdown
```

政府的决策规则使用与 {doc}`phillips_learning` 相同的菲尔普斯求解器。

状态变量为 $(U_{n-1}, U_{n-2}, \pi_{n-1}, \pi_{n-2}, 1)$，折现里卡蒂方程中的交叉项以未缩放的形式进入方程。

```{code-cell} ipython3
def phelps_rule(γ, δ=0.98):
    "Rule π̂_n = h·(U_{n-1}, U_{n-2}, π_{n-1}, π_{n-2}, 1) for beliefs γ."
    γ0, c = γ[0], γ[1:]
    A, B = np.zeros((5, 5)), np.zeros((5, 1))
    A[0, :], B[0, 0] = c, γ0            # U_n
    A[1, 0] = 1.0                        # U_{n-1}
    B[2, 0] = 1.0                        # π_n
    A[3, 2] = 1.0                        # π_{n-1}
    A[4, 4] = 1.0                        # constant
    Q_x, Q_u, N = np.outer(c, c), np.array([[γ0**2 + 1.0]]), (γ0 * c).reshape(1, -1)
    Ad, Bd = np.sqrt(δ) * A, np.sqrt(δ) * B
    P = solve_discrete_are(Ad, Bd, Q_x, Q_u, s=N.T)
    F = np.linalg.solve(Q_u + Bd.T @ P @ Bd, Bd.T @ P @ Ad + N)
    return -F.ravel()


def dynamic_sce(model):
    "SCE beliefs and moments of (π_n, U_{n-1}, U_{n-2}, π_{n-1}, π_{n-2}, 1)."
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ = np.array([-θ, 0, 0, 0, 0, (1 + θ**2) * u])
    mean = np.array([θ * u, u, u, θ * u, θ * u, 1.0])
    lag = [0, 1, 2, 1, 2, None]                       # date of each regressor
    var = {('π', 'π'): σ2**2, ('U', 'U'): θ**2 * σ2**2 + σ1**2,
           ('π', 'U'): -θ * σ2**2, ('U', 'π'): -θ * σ2**2}
    kind = ['π', 'U', 'U', 'π', 'π', 'c']
    M = np.outer(mean, mean)
    for i in range(6):
        for j in range(6):
            if lag[i] is not None and lag[i] == lag[j]:
                M[i, j] += var[(kind[i], kind[j])]
    return γ, M


def first_escape_dynamic(model, ε, seed, n_max=10_000, window=20):
    rng = np.random.default_rng(seed)
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ, R = dynamic_sce(model)
    U1 = U2 = u
    π1 = π2 = θ * u
    recent = np.full(window, θ * u)
    for n in range(n_max):
        π_hat = phelps_rule(γ) @ np.array([U1, U2, π1, π2, 1.0])
        w1, w2 = rng.standard_normal(2)
        π = π_hat + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.array([π, U1, U2, π1, π2, 1.0])
        R = R + ε * (np.outer(Φ, Φ) - R)
        γ = γ + ε * np.linalg.solve(R, Φ * (U - Φ @ γ))
        U2, U1, π2, π1 = U1, U, π1, π
        recent[n % window] = π
        if n >= window and recent.mean() < 1:
            return n + 1
    return np.nan


def first_escape_static(model, ε, n_paths=200, n_max=20_000, window=20, seed=0):
    rng = np.random.default_rng(seed)
    θ, u, σ1, σ2 = model.θ, model.u, model.σ1, model.σ2
    γ = np.tile(model.γ_sce(), (n_paths, 1))
    R = np.tile(model.M(model.γ_sce()), (n_paths, 1, 1))
    recent = np.full((n_paths, window), θ * u)
    τ = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, dtype=bool)
    for n in range(n_max):
        x = -γ[:, 1] * γ[:, 0] / (1 + γ[:, 0]**2)
        w1, w2 = rng.standard_normal((2, n_paths))
        π = x + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.stack([π, np.ones(n_paths)], axis=1)
        R += ε * (Φ[:, :, None] * Φ[:, None, :] - R)
        e = U - (Φ * γ).sum(axis=1)
        γ += ε * np.linalg.solve(R, (Φ * e[:, None])[:, :, None])[:, :, 0]
        recent[:, n % window] = π
        hit = alive & (n >= window) & (recent.mean(axis=1) < 1)
        τ[hit] = n + 1
        alive &= ~hit
        if not alive.any():
            break
    return τ


print("check: dynamic rule at the SCE =", phelps_rule(dynamic_sce(model)[0]).round(4))
```

现在我们比较第一次逃逸时间的中位数。

```{code-cell} ipython3
print(f"{'ε':>6s} {'static median':>14s} {'dynamic median':>15s}")
for ε in [0.002, 0.005, 0.01, 0.02]:
    τ_s = first_escape_static(model, ε)
    τ_d = np.array([first_escape_dynamic(model, ε, seed) for seed in range(24)])
    print(f"{ε:6.3f} {np.nanmedian(τ_s):14.0f} {np.nanmedian(τ_d):15.0f}"
          f"   (not escaped: {np.isnan(τ_s).sum()}/{len(τ_s)}, {np.isnan(τ_d).sum()}/{len(τ_d)})")
```

在每一种增益下，动态模型都更早发生逃逸，并且随着增益的降低，这一差距还在扩大，这与 CWS 对动态模型给出更小的 $\bar S$ 这一点相吻合。

带有滞后结构的政府可以在保持短期菲利普斯曲线的同时发现归纳假设，因此只需要一次不那么异常的冲击序列就能开始逃逸。

即使两个模型共享同一个自我确认均衡，这一比较结果依然成立。

```{solution-end}
```
