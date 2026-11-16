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
  title: 信誉问题
  headings:
    Overview: 概述
    A one-period economy: 单期经济
    Nash and Ramsey outcomes: 纳什和拉姆齐结果
    Nash and Ramsey outcomes::A picture of the two outcomes: 两种结果的图示
    Best response dynamics: 最优反应动态
    Least squares learning converges to Nash: 最小二乘学习收敛于纳什结果
    Least squares learning converges to Nash::How fast?: 收敛速度如何？
    More foresight: 更多的前瞻性
    'Appendix: stochastic approximation': 附录：随机逼近
    Exercises: 练习
---

(phillips_credibility)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 信誉问题

```{index} single: Phillips Curve; Credibility Problem
```

```{contents} Contents
:depth: 2
```

## 概述

本讲座描述了一个由 {cite:t}`KydlandPrescott1977` 和 {cite:t}`BarroGordon1983` 所研究的那种基本预期菲利普斯曲线模型。

这是基于 {cite:t}`Sargent1999` 各章内容的系列讲座中第一讲*建模*讲座，紧接 {doc}`phillips_two_stories` 之后，后者阐述了两种叙事并回顾了卢卡斯批判。

本讲座对应该书第 3 章的内容，而 {doc}`phillips_credible_policies` 对应第 4 章。

这两讲共同形式化了：

* 发现菲利普斯曲线所激发的通胀诱惑；
* 承诺技术在抵御这种诱惑方面的价值；以及
* 声誉机制作为承诺替代品的脆弱性。

在整个讲座中，我们只使用理性预期这一均衡概念。

政府和私人部门决策*时序*上的改变会诱发不同的经济体，产生不同的结果。

每当政府希望比其必须做出决策的时间更早地做出决策时，它就面临**信誉问题**。

我们将比较两种时序协议下的结果：

* 在其中一种协议下，政府在私人部门形成其预期*之前*选择通货膨胀率，因此政府会考虑其选择将如何影响这些预期。
* 在另一种协议下，政府在私人部门已经形成其预期*之后*选择通货膨胀率。

第二种协议下结果的恶化程度衡量了无法承诺所造成的损失。

{doc}`advanced:calvo` 在卡尔沃的动态货币模型中研究了同样的时间不一致性问题。

我们还研究了两种收敛于无承诺（纳什）结果的*非均衡*动态：

* 最优反应动态，以及
* 最小二乘学习。

让我们从一些标准的导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC', 'DejaVu Sans']
import numpy as np
```

## 单期经济

尽管信誉问题本质上是动态的，但在不同的期内时序模式下，可以在一个单期模型中描述它们。

这为后续讲座中的多期分析做好了准备。

我们用南希·斯托基（Nancy Stokey）在 {cite}`stokey1989reputation` 中所使用的术语来描述 {cite:t}`KydlandPrescott1977` 的单期模型的一个版本。

设 $(U, y, x)$ 分别为失业率、通货膨胀率，以及公众对通货膨胀率的预期。

政府的单期收益为：

```{math}
:label: pc_payoff

- \frac{1}{2} \left( U^2 + y^2 \right) .
```

失业率由一条预期增广的菲利普斯曲线决定：

```{math}
:label: pc_phillips

U = U^* - \theta (y - x), \qquad \theta > 0 .
```

该方程表明，只有当出现意外的通胀或通缩时，失业率才会偏离自然率 $U^*$。

将 {eq}`pc_phillips` 代入 {eq}`pc_payoff`，可将政府的收益表示为函数 $r(x, y)$：

```{math}
:label: pc_r

r(x, y) = - \frac{1}{2} \left[ \left(U^* - \theta (y - x)\right)^2 + y^2 \right] .
```

我们使用以下概念。

**理性预期均衡：** 满足 {eq}`pc_phillips` 且 $y = x$ 的三元组 $(U, x, y)$。

**政府（单期）最优反应：** 给定公众的预期 $x$，设定 $y$ 的决策规则 $B(x) = \operatorname{argmax}_y r(x, y)$。

**纳什均衡：** 满足 (i) $x = y$，以及 (ii) $y = B(x)$ 的二元组 $(x, y)$。

**拉姆齐问题：** $\max_y r(y, y)$。

*拉姆齐结果*是达到最大值的 $y$。

**最优反应动态：** 动态系统 $y_t = B(y_{t-1})$，给定 $y_0$。

理性预期均衡是位于菲利普斯曲线上、且给定 $x$ 时私人主体不会被欺骗的三元组 $(U, x, y)$。

将 $x = y$ 代入菲利普斯曲线 {eq}`pc_phillips` 表明，在任何理性预期均衡中 $U = U^*$。

这确定了 $U^*$ 为自然失业率。

## 纳什和拉姆齐结果

纳什均衡建立在给定预期状态 $x$ 时政府的最优反应之上，同时结合市场的反应 $x = y$，即对给定 $y$ 的理性预期。

在固定 $x$ 的情况下，对 {eq}`pc_r` 关于 $y$ 求最大化，得到政府的最优反应函数：

```{math}
:label: pc_B

y = B(x) = \frac{\theta}{\theta^2 + 1} U^* + \frac{\theta^2}{\theta^2 + 1} x .
```

纳什均衡设定 $x = y = B(x)$，由此得到：

$$
y^N = x^N = \theta U^*, \qquad U = U^* .
$$

拉姆齐问题则在最大化*之前*就施加 $x = y$，因此它最大化 $r(y, y) = -\tfrac{1}{2}(U^{*2} + y^2)$，得到拉姆齐结果：

$$
y^R = x^R = 0, \qquad U = U^* .
$$

因此 $r(x^R, y^R) = -\tfrac{1}{2} U^{*2}$，而 $r(x^N, y^N) = -\tfrac{1}{2}(1 + \theta^2) U^{*2}$。

两种结果都实现了自然率 $U^*$，但纳什均衡以正的通货膨胀率实现它，因此收益严格更低。

让我们把这些公式整理到一个类中。

```{code-cell} ipython3
class CredibilityModel:
    """
    A one-period expectational Phillips curve economy.
    """

    def __init__(self, θ=1.0, U_star=5.0):
        self.θ, self.U_star = θ, U_star

    def phillips(self, y, x):
        "Unemployment implied by inflation y and expected inflation x."
        return self.U_star - self.θ * (y - x)

    def r(self, x, y):
        "Government one-period payoff."
        U = self.phillips(y, x)
        return -0.5 * (U**2 + y**2)

    def B(self, x):
        "Government best response to expected inflation x."
        θ = self.θ
        return θ / (θ**2 + 1) * self.U_star + θ**2 / (θ**2 + 1) * x

    def nash(self):
        "Nash equilibrium inflation (= expected inflation)."
        return self.θ * self.U_star

    def ramsey(self):
        "Ramsey inflation (= expected inflation)."
        return 0.0
```

默认校准为 $\theta = 1$ 和 $U^* = 5$。

```{code-cell} ipython3
cm = CredibilityModel()

y_N, y_R = cm.nash(), cm.ramsey()
print(f"Nash inflation      y^N = {y_N:.2f}")
print(f"Ramsey inflation    y^R = {y_R:.2f}")
print(f"Nash payoff         r   = {cm.r(y_N, y_N):.3f}")
print(f"Ramsey payoff       r   = {cm.r(y_R, y_R):.3f}")
```

纳什结果的收益比拉姆齐结果的收益差。

政府会更愿意实现拉姆齐结果，但如果没有一种能在公众形成预期*之前*承诺 $y = 0$ 的技术，它就无法实现这一结果。

纳什均衡由这样一种时序协议支持：政府在私人部门形成预期*之后*才做出决策。

拉姆齐结果则与这样一种时序协议相关联：政府*首先*做出选择，并且知道在理性预期均衡中，公众的预期会随其选择而变化，因为 $y = x$。

### 两种结果的图示

政府的无差异曲线是 $(U, y)$ 空间中以原点为中心的圆，因为收益 {eq}`pc_payoff` 仅依赖于 $U^2 + y^2$。

对于给定的预期 $x$，菲利普斯曲线 {eq}`pc_phillips` 是 $(U, y)$ 空间中一条向下倾斜的直线。

给定 $x$ 时，政府对 $y$ 的最优反应出现在无差异曲线与由 $x$ 索引的菲利普斯曲线相切的位置。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 以预期通货膨胀率为索引的菲利普斯曲线、政府无差异曲线以及纳什和拉姆齐结果
    name: fig-cred-nash-ramsey
---
fig, ax = plt.subplots(figsize=(7, 6))

U_grid = np.linspace(0, 12, 200)

# a family of Phillips curves indexed by expected inflation x
for x in [0.0, y_N / 2, y_N]:
    # U = U_star - θ (y - x)  =>  y = x + (U_star - U) / θ
    y_line = x + (cm.U_star - U_grid) / cm.θ
    ax.plot(U_grid, y_line, 'C0', lw=1)

# government indifference curves (circles U^2 + y^2 = const)
ξ = np.linspace(0, 2 * np.pi, 200)
for R in [np.hypot(cm.U_star, y_R), np.hypot(cm.U_star, y_N)]:
    if R > 0:
        ax.plot(R * np.cos(ξ), R * np.sin(ξ), 'C1--', lw=1)

ax.plot(cm.U_star, y_N, 'ko')
ax.annotate('Nash', (cm.U_star, y_N), (cm.U_star + 0.4, y_N + 0.4))
ax.plot(cm.U_star, y_R, 'ko')
ax.annotate('Ramsey', (cm.U_star, y_R), (cm.U_star + 0.4, y_R + 0.4))

ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_xlabel('unemployment $U$')
ax.set_ylabel('inflation $y$')
plt.show()
```

实线是预期通货膨胀率 $x \in \{0, y^N/2, y^N\}$ 对应的菲利普斯曲线；虚线圆是政府的无差异曲线。

纳什结果 $(U^*, y^N)$ 位于比拉姆齐结果 $(U^*, 0)$ 更大的圆上（收益更低）。

## 最优反应动态

最优反应动态通过假设一种自适应机制，将单期模型转化为动态模型，在该机制中，预期通货膨胀率等于上一期的通货膨胀率，即 $x_t = y_{t-1}$。

这导致了以下动态：

$$
y_t = B(y_{t-1}), \qquad y_0 \text{ given} .
$$

由于 $B$ 是一个斜率为 $\theta^2 / (\theta^2 + 1) \in (0, 1)$ 的仿射映射，对其迭代会从任意起始点收敛到不动点 $y^N = \theta U^*$。

让我们将最优反应函数与 45 度线一起绘图，并模拟该动态过程。

```{code-cell} ipython3
def best_response_path(cm, y0, T=20):
    "Iterate y_{t+1} = B(y_t)."
    y = np.empty(T + 1)
    y[0] = y0
    for t in range(T):
        y[t + 1] = cm.B(y[t])
    return y

y_path = best_response_path(cm, y0=0.0, T=20)
```

下图将该路径描绘为最优反应函数与 45 度线之间的蛛网图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 最优反应动态蛛网图收敛至纳什通货膨胀率
    name: fig-cred-best-response
---
fig, ax = plt.subplots(figsize=(6, 6))

x_grid = np.linspace(0, y_N + 1, 100)
ax.plot(x_grid, cm.B(x_grid), 'C0', label='$B(x)$')
ax.plot(x_grid, x_grid, 'k--', lw=1, label='45 degrees')

# cobweb of the best response dynamics
for t in range(len(y_path) - 1):
    ax.plot([y_path[t], y_path[t]], [y_path[t], y_path[t + 1]], 'C1', lw=0.8)
    ax.plot([y_path[t], y_path[t + 1]], [y_path[t + 1], y_path[t + 1]],
            'C1', lw=0.8)

ax.plot(y_N, y_N, 'ko')
ax.annotate('Nash', (y_N, y_N), (y_N - 1.5, y_N + 0.3))
ax.set_xlabel('$x$')
ax.set_ylabel('$B(x)$')
ax.legend()
plt.show()
```

从 $x = 0$（拉姆齐通货膨胀率）开始，政府设定 $y = B(0) > 0$。

这促使公众提高其预期，从而导致政府进一步提高通货膨胀率。

这一过程的极限是纳什结果 $y = x = y^N$，一种自我确认的状态。

因此，最优反应动态收敛于纳什均衡，进一步强化了 $(U^*, y^N)$ 作为无承诺技术模型的预测。

(pc_mean_dynamics)=
## 最小二乘学习收敛于纳什结果

最优反应动态的一个变体也可以从最小二乘学习中产生。

最小二乘学习在整个这一系列讲座中都起着关键作用，这个简单的例子引入了一些分析要素，这些要素将在后面更复杂的场景中再次出现。

{cite:t}`Bray1982` 研究了一个竞争性市场，在该市场中交易者通过对过去价格取平均来形成预期价格。

本着同样的精神，假设对于 $t \geq 2$，预期通货膨胀率 $x_t$ 是过去通货膨胀率的平均值：

$$
x_t = \frac{1}{t - 1} \sum_{s=1}^{t-1} y_s ,
$$

它可以递归地表示为：

```{math}
:label: pc_expect1

x_t = x_{t-1} + \frac{1}{t-1} (y_{t-1} - x_{t-1}), \qquad x_1 = 0 .
```

实际通货膨胀率是在 $x_t$ 处评估的最优反应映射的一个扰动版本：

```{math}
:label: pc_expect2

y_t = B(x_t) + \eta_t ,
```

其中 $\eta_t$ 是一个独立同分布的均值为零的项，代表政府对通货膨胀的不完全控制。

将日期为 $t-1$ 的 {eq}`pc_expect2` 代入 {eq}`pc_expect1`，得到随机递归式：

```{math}
:label: pc_expect3

x_t = x_{t-1} + \frac{1}{t-1} \left[ B(x_{t-1}) - x_{t-1} + \eta_{t-1} \right] .
```

预测值 $x_t$ 只取决于日期为 $t-1$ 及更早的冲击，这是必然的，因为公众是在看到 $y_t$ 之前形成该预测的。

（{cite:t}`Sargent1999` 中对应的公式（8）将该冲击的日期标记为 $t$，这是一处笔误。）

这正是本讲座所用记号下的 {ref}`br_bray` 中布雷的市场。

* 政府的最优反应 $B$ 扮演着企业最优反应 $h$ 的角色。
* 预测值 $x_t$ 扮演着企业信念 $X^*_{t-1}$ 的角色，该信念由截至 $t-1$ 的数据形成。
* 通货膨胀率 $y_t$ 扮演着实际平均产出 $X_t$ 的角色。

从信念到结果的反馈为：

$$
h' = B'(x) = \frac{\theta^2}{1 + \theta^2} \in (0, 1),
$$

因此布雷提出的最小二乘学习者收敛条件 $h' < 1$ 对任意 $\theta$ 都成立。

{doc}`ls_learning` 在 {cite:t}`MarcetSargent1989jet` 的一般框架下分析了布雷的模型（其例 2）。

根据随机逼近理论，$x_t$ 的极限行为由相关的常微分方程（ODE）描述：

```{math}
:label: pc_ode

\frac{d x}{d t} = B(x) - x .
```

这个常微分方程的静止点满足 $x = B(x)$，即纳什均衡通货膨胀率 $x = \theta U^*$。

由于映射 $B$ 是仿射的，该常微分方程是线性的，其斜率为：

$$
\mathcal{M} = \frac{d}{d x}\left( B(x) - x \right) = B'(x) - 1 = - \frac{1}{\theta^2 + 1} .
$$

由于 $\mathcal{M} < 0$，该常微分方程在其静止点附近是稳定的。

用 {ref}`br_one_recursion` 的语言来说，纳什均衡是 **E 稳定的**（预期稳定）：$B$ 将感知的通货膨胀映射为实际的通货膨胀，而该映射的导数 $h' = B'$ 小于 1。

$\mathcal{M} = h' - 1$ 就是这个一维系统的 E 稳定性特征值。

正是 E 稳定性使得 {cite:t}`MarcetSargent1989jet` 的定理得以应用，这些定理给出了 $x_t$ 全局收敛到 $y^N$ 的条件。

同样的数值会贯穿这一系列讲座反复出现。

当政府本身通过最小二乘法估计一条菲利普斯曲线时，其信念的均值动态具有特征值 $-1$ 和 $-1/(1+\theta^2)$（参见 {ref}`sc_estability`、{ref}`pl_eigen`）。

让我们模拟递归式 {eq}`pc_expect3` 并确认其收敛到纳什结果。

```{code-cell} ipython3
def ls_learning(cm, T=2000, σ_η=1.0, seed=0, ε=None):
    """
    Simulate learning of expected inflation.

    x[t] is the forecast formed after observing y_1, ..., y_t, which is the
    x_{t+1} of pc_expect1; x[0] = x_1 = 0, and y[0] is unused.

    With ε=None the gain is 1/t (least squares: the 1/(t-1) gain of
    pc_expect1, reindexed); otherwise it is the constant ε.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(T + 1)
    y = np.full(T + 1, np.nan)
    x[0] = 0.0
    for t in range(1, T + 1):
        y[t] = cm.B(x[t - 1]) + σ_η * rng.standard_normal()
        gain = 1.0 / t if ε is None else ε
        x[t] = x[t - 1] + gain * (y[t] - x[t - 1])
    return x, y
```

我们在默认校准下模拟 2000 期。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 预期通货膨胀率的最小二乘学习收敛于纳什结果
    name: fig-cred-ls-learning
---
x, y = ls_learning(cm, T=2000, σ_η=1.0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, 'C0', lw=1, label='expected inflation $x_t$')
ax.axhline(y_N, color='k', ls='--', lw=1, label='Nash $y^N$')
ax.axhline(y_R, color='C2', ls=':', lw=1, label='Ramsey $y^R$')
ax.set_xlabel('$t$')
ax.set_ylabel('inflation')
ax.legend()
plt.show()
```

最小二乘动态证实了最优反应动态所展现的悲观结论。

正如 {cite:t}`Sargent1999` 所言，这种悲观结论削弱了 {doc}`phillips_two_stories` 中*自然率理论的胜利*这一说法，尤其是其中"胜利"的那一面。

给定一个以低通胀、金本位式或布雷顿森林体系下的 $x$ 值形式出现的初始条件，最优反应或最小二乘动态可以解释通货膨胀的*加速*。

但它们无法解释像沃尔克式那样将通货膨胀率重新拉低的*稳定化*过程。

后续讲座将以旨在缓和这种悲观情绪的方式重新构建最小二乘学习的各种版本，这样做也使我们从"胜利"的说法转向"平反"的说法。

### 收敛速度如何？

$\mathcal{M}$ 也决定着*收敛速度*（参见 {cite:t}`Sargent1999` 第 3 章脚注 3；{cite:t}`EvansHonkapohja2001`）。

对 {eq}`pc_expect3` 取期望可以看出，均值误差 $E x_t - y^N$ 服从该递归式的无噪声版本，因此它以 $t^{\mathcal{M}}$ 的速度收缩。

以通常速度收敛的一个必要条件——即 $\sqrt{t}\,(x_t - y^N)$ 最终趋于正态分布——是 $\mathcal{M} < -1/2$，这里要求 $\theta < 1$。

当 $-1/2 \leq \mathcal{M} < 0$ 时，$x_t$ 仍然收敛于 $y^N$，但速度更慢，收敛速率由 $|\mathcal{M}|$ 决定。

在 $\theta = 1$（本讲座及后续大多数讲座所采用的值）处，$\mathcal{M} = -1/2$ 恰好位于边界上：此时 $x_t$ 的标准差以 $\sqrt{\log t / t}$ 的速度收缩，而非 $1/\sqrt{t}$。

与此相对，在 {ref}`br_bray` 的市场中，$h' = -1.5$ 给出 $\mathcal{M} = -2.5$，误差以通常的速率 $t^{-1/2}$ 收缩。

练习 {ref}`pc_ex1` 对这些速率进行了测算。

这里 $-1/2$ 的边界所涉及的是增益以 $1/t$ *递减*的学习者的收敛速率。

它并不决定后续讲座中增益为*常数*的学习者多久会脱离纳什结果一次；决定后者的是一个大偏差速率（参见 {ref}`pl_primer`）。

## 更多的前瞻性

最优反应和最小二乘学习是附加在单期经济上的非均衡动态。

它们迫使所有的变动都通过预期形成来实现：在选择通货膨胀率时，政府忘记了经济会持续多于一期。

如果政府为未来进行规划，可以出现更好的结果。

后续讲座描述了三种建模前瞻性的方式，它们赋予不同程度的理性，并预测不同质量的结果：

1. 一种声誉方法，将理性预期同时赋予政府和公众，这是 {doc}`phillips_credible_policies` 的主题，紧接下文介绍。
   - 许多结果都是可持续的，从拉姆齐结果的重复到比纳什结果重复更差的路径都有可能，而这种多重性正是该理论的主要启示。
2. 一种方法，保持政府的理性，但赋予公众原始凯根-弗里德曼意义上的*适应性*预期，这是 {doc}`phillips_adaptive` 的主题。
   - 根据贴现因子与适应参数的比较，这种设定可以改善结果，甚至可能维持拉姆齐结果的重复。
3. 同时使政府也具有有限理性的几种方法。
   - 在 {doc}`phillips_self_confirming` 中（由 {doc}`phillips_misspecified` 为其做铺垫），政府持有一个可能设定有误的菲利普斯曲线模型，而其产生的数据恰好证实了该模型。
   - 从 {doc}`phillips_learning` 开始，政府会实时重新估计该模型。
   - 结果取决于赋予政府信念的具体细节，并可能由此得出一种对计量经济学政策评估的证实。

## 附录：随机逼近

在这里我们简要说明为什么常微分方程 {eq}`pc_ode` 支配着随机差分方程 {eq}`pc_expect3` 的尾部行为。

这一论证源自 {cite:t}`KushnerClark1978`，包含两个关键组成部分：时间尺度的转换和对平均化方法的充分运用。

{ref}`br_one_recursion` 中“随机逼近与 E-稳定性”一节从有限理性系列讲座的角度介绍了相同的工具，而 {doc}`ls_learning` 则陈述了针对信念向量的 Marcet–Sargent 收敛定理。

设 $\{a_n\}_{n \geq 0}$ 为满足以下条件的正增益序列：

$$
\lim_{n \to \infty} a_n = 0, \qquad \sum_n a_n = +\infty, \qquad \sum_n a_n^2 < +\infty .
$$

选择 $a_n = 1 / (n + 1)$ 满足这些假设。

对 {eq}`pc_expect3` 重新标记索引，使其 $x_{n+1}$ 和 $\eta_{n+1}$ 分别变为 $x_n$ 和 $\eta_n$。

于是增益为 $a_n = 1/(n+1)$，递归式可写为：

```{math}
:label: pc_sa1

x_{n+1} = x_n + a_n \left[ B(x_n) - x_n + \eta_n \right] ,
```

其中 $\eta_n$ 是独立同分布的，均值为零，方差有限。

引入变换后的时间尺度 $t_0 = 0$，$t_n = \sum_{i=0}^{n-1} a_i$，并将离散序列 $\{x_n\}$ 插值为连续时间过程 $x^0(t)$。

映射 $m(t) = \max\{n : t_n \leq t\}$ 将变换后的时间转换回原来的索引。

库什纳和克拉克表明，在这个变换后的时间尺度上，插值过程可以很好地由以下积分方程逼近：

$$
x^0(t) = x^0(0) + \int_0^t \left[ B(x^0(s)) - x^0(s) \right] d s + R(t) ,
$$

其中逼近误差 $R(t)$ 有两个组成部分——一个来自用积分逼近 $B(x) - x$ 中的分布滞后项，另一个来自 $\eta_s$ 中的分布滞后项。

通过研究该过程的一系列左移版本，他们证明当 $n \to \infty$ 时，两个误差组成部分都可以被驱动趋于零。

对于噪声成分，关键步骤是注意到在从 $t_n$ 到 $t_n + t$ 这段变换后的时间内，噪声是通过鞅 $W_n(t) = \sum_{i=n}^{m(t_n + t)} a_i \eta_i$ 进入的。

它的方差为 $\mathbb{V}[\eta] \sum_{i=n}^{m(t_n + t)} a_i^2$，这是收敛级数 $\sum_i a_i^2$ 的一个*尾部*和，因此当 $n \to \infty$ 时趋于零。

剩余的误差被送到零，是因为 $a_i \to 0$ 缩小了用于逼近积分的黎曼和的网格尺度。

在极限情形下，随机差分方程 {eq}`pc_sa1` 与非随机常微分方程：

$$
\frac{d}{d t} \tilde x(t) = B(\tilde x(t)) - \tilde x(t) ,
$$

共享相同的行为，后者被称为描述原系统的*均值动态*。

后续讲座研究了类似 {eq}`pc_sa1` 的系统，其中 $a_i$ *不*随 $i$ 增大而趋于零——即所谓的常增益算法。

练习 {ref}`pc_ex2` 初步展示了其中一个例子。

均值动态 {eq}`pc_ode` 和这些常增益算法成为 {doc}`phillips_learning`、{doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 中的核心工具，在那里，本讲中所研究的标量预期 $x$ 会扩展为一整套漂移的菲利普斯曲线系数向量。

## 练习

```{exercise-start}
:label: pc_ex1
```

本练习考察最小二乘学习的速度如何取决于 $\mathcal{M} = -1/(1 + \theta^2)$。

令 $e_t = x_t - \theta U^*$ 为相对于纳什值的预测误差。

1. 证明均值误差 $E e_t$ 满足 {eq}`pc_expect3` 的无噪声版本，且其以 $t^{\mathcal{M}}$ 的速度缩小。
1. 对于 $\theta \in \{0.5, 1, 2\}$ 和 $U^* = 5$，模拟 200 条长度为 $T = 3000$ 的路径（种子为 $0, \ldots, 199$）。
   在对数-对数坐标轴上绘制 $|\bar e_t| / (\theta U^*)$，其中 $\bar e_t$ 是各路径的平均误差，并与精确均值误差进行比较。
   除以初始误差 $\theta U^*$ 可使三种情形具有可比性。
1. 绘制 $\sqrt{t}$ 乘以各路径间 $x_t$ 标准差的图形，并解释为何它仅在 $\theta < 1$ 时才会趋于平稳。
1. 在 $t = 3000$ 时，$\theta = 2$ 的路径中有多大比例仍低于纳什值？
   $\theta = 2$ 的学习者是否会在其极限附近波动？

```{exercise-end}
```

```{solution-start} pc_ex1
:class: dropdown
```

写出 $B(x) - x = \mathcal{M}(x - y^N)$。

将 {eq}`pc_expect3` 向前移动一期，可得 $e_{t+1} = (1 + \mathcal{M}/t)\, e_t + \eta_t / t$，其中 $e_1 = -\theta U^*$，取期望后冲击项消失。

因此均值误差恰好就是 `ls_learning` 在 $\sigma_\eta = 0$ 时所产生的路径：

$$
E e_{t+1} = e_1 \prod_{s=1}^{t} \left(1 + \frac{\mathcal{M}}{s}\right)
\approx \frac{e_1\, t^{\mathcal{M}}}{\Gamma(1 + \mathcal{M})} ,
$$

这是因为 $\log \prod_s (1 + \mathcal{M}/s) \approx \mathcal{M} \sum_{s \leq t} 1/s \approx \mathcal{M} \log t$。

$e_t$ 的方差 $v_t$ 满足 $v_{t+1} = (1 + \mathcal{M}/t)^2 v_t + \sigma_\eta^2 / t^2$。

当 $\mathcal{M} < -1/2$ 时，其解的行为类似于 $\sigma_\eta^2 / [(-2\mathcal{M} - 1)\, t]$；当 $\mathcal{M} = -1/2$ 时，类似于 $\sigma_\eta^2 \log t / t$；当 $\mathcal{M} > -1/2$ 时，则类似于 $t^{2\mathcal{M}}$ 的某个倍数。

下面的模拟以两种方式计算均值误差，并在 $100 \leq t \leq 3000$ 范围内拟合斜率。

```{code-cell} ipython3
T, n_paths = 3000, 200
t = np.arange(1, T + 1)
fit = t >= 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for k, θ in enumerate([0.5, 1.0, 2.0]):
    cm_θ = CredibilityModel(θ=θ, U_star=5.0)
    y_N_θ, M = cm_θ.nash(), -1 / (1 + θ**2)

    # 观测 t = 1, ..., T 次之后的误差，每条路径一行
    e = np.array([ls_learning(cm_θ, T=T, seed=s)[0][1:]
                  for s in range(n_paths)]) - y_N_θ
    exact = ls_learning(cm_θ, T=T, σ_η=0.0)[0][1:] - y_N_θ    # E e_t
    mean, sd = e.mean(axis=0), e.std(axis=0)

    slope_exact = np.polyfit(np.log(t[fit]), np.log(np.abs(exact[fit])), 1)[0]
    slope_mc = np.polyfit(np.log(t[fit]), np.log(np.abs(mean[fit])), 1)[0]
    noisy = np.abs(exact) < 2 * sd / np.sqrt(n_paths)
    print(f"θ = {θ}:  M = {M:+.2f},  slope of exact mean {slope_exact:+.3f},  "
          f"slope of 200-path mean {slope_mc:+.3f}")
    if noisy.any():
        print(f"         exact mean below 2 Monte Carlo standard errors "
              f"from t = {t[noisy][0]}")
    print(f"         at t = {T}: mean error {mean[-1]:+.3f}, sd {sd[-1]:.3f}, "
          f"share of paths below Nash {np.mean(e[:, -1] < 0):.2f}")

    axes[0].loglog(t, np.abs(mean) / y_N_θ, color=f'C{k}', lw=0.7, alpha=0.6)
    axes[0].loglog(t, np.abs(exact) / y_N_θ, color=f'C{k}', ls='--', lw=1.5,
                   label=rf'$\theta = {θ}$, $\mathcal{{M}} = {M:.1f}$')
    axes[1].semilogx(t, np.sqrt(t) * sd, color=f'C{k}', lw=1.2,
                     label=rf'$\theta = {θ}$')

M_half = -1 / (1 + 0.5**2)                     # θ = 0.5 时的 M
axes[1].axhline(1 / np.sqrt(-2 * M_half - 1), color='C0', ls=':', lw=1,
                label=r'$\sigma_\eta / \sqrt{-2\mathcal{M} - 1}$, $\theta = 0.5$')
axes[1].semilogx(t, np.sqrt(np.log(t)), color='C1', ls=':', lw=1,
                 label=r'$\sigma_\eta \sqrt{\log t}$')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel(r'$|$均值误差$| \,/\, \theta U^*$')
axes[0].set_title('均值误差：200 条路径（实线），精确值（虚线）', fontsize=10)
axes[0].legend(fontsize=9)
axes[1].set_xlabel('$t$')
axes[1].set_ylabel(r'$\sqrt{t}\ \times$ $x_t$ 的标准差')
axes[1].set_ylim(0, 8)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()
```

精确均值误差的下降斜率等于 $\mathcal{M}$：分别为 $-0.8$、$-0.5$ 和 $-0.2$。

对于 $\theta = 1$ 和 $\theta = 2$，200 条路径的平均值与精确均值重合。

对于 $\theta = 0.5$，仅在最初几十期内如此。

此后，均值误差比蒙特卡洛标准误差还小，其平均值本身便是噪声。

这就是第一个结论：当 $\mathcal{M} < -1/2$ 时，初始条件被遗忘的速度（约为 $t^{-0.8}$）要快于抽样噪声消失的速度（约为 $t^{-1/2}$）。

因此收敛受噪声限制，$\sqrt{t}$ 乘以标准差最终趋于 $\sigma_\eta / \sqrt{-2\mathcal{M} - 1} \approx 1.29$。

当 $\mathcal{M} > -1/2$ 时，情形相反。

对于 $\theta = 2$，均值误差和标准差都仅以 $t^{-0.2}$ 的速度缩小，因此 $\sqrt{t}$ 乘以标准差持续增大，而均值误差占主导地位。

经过 3000 期后，平均而言 $\theta = 2$ 的学习者仍比纳什值低约 1.7 个单位，相差好几个标准差，且全部 200 条路径都低于纳什值。

$\theta = 2$ 的学习者并非在其极限附近波动：它是从下方缓慢且几乎单调地逼近极限的。

默认的 $\theta = 1$ 是这两种情形之间的分界线。

在此情形下，均值误差和标准差都以 $t^{-1/2}$ 的速度缩小，只不过多了一个 $\sqrt{\log t}$ 的因子，使得 $\sqrt{t}$ 乘以标准差会缓慢爬升。

```{solution-end}
```

```{exercise-start}
:label: pc_ex2
```

将 {eq}`pc_expect1` 中的递减增益 $1/(t-1)$ 替换为常数增益 $\varepsilon \in (0, 1)$，使得 $x_{t+1} = x_t + \varepsilon (y_t - x_t)$。

这就是 {doc}`phillips_adaptive` 中 $\varepsilon = 1 - \lambda$ 时的适应性预期规则，但此处政府仍是逐期进行最优反应。

1. 证明 $e_t = x_t - y^N$ 是系数为 $1 + \varepsilon \mathcal{M}$ 的一阶自回归过程。
   证明当该过程平稳时，其方差为

   $$
   V(\varepsilon) = \frac{\varepsilon \sigma_\eta^2}{(1 - h')\bigl(2 - \varepsilon (1 - h')\bigr)},
   \qquad h' = \frac{\theta^2}{1 + \theta^2} .
   $$

   这就是 {ref}`br_ex5` 中针对布雷市场的公式，其中 $h' = B'$。
   对于哪些增益该递归是稳定的？
1. 设 $\theta = 1$，$\sigma_\eta = 1$，使用 `ls_learning` 的选项 `ε`，针对 $\varepsilon \in \{0.01, 0.025, 0.05, 0.1, 0.2\}$ 模拟 100,000 期。
   舍弃前 5,000 期，比较 $e_t$ 的样本方差与 $V(\varepsilon)$。
1. 使用相同的冲击，绘制 $\varepsilon = 0.025$ 下与最小二乘增益下 3000 期 $x_t$ 的图形。
   常数增益带来了什么好处，又付出了什么代价？

```{exercise-end}
```

```{solution-start} pc_ex2
:class: dropdown
```

由于 $B(x) - x = \mathcal{M}(x - y^N)$，更新式给出 $e_{t+1} = (1 + \varepsilon \mathcal{M})\, e_t + \varepsilon \eta_t$。

令 $\rho = 1 + \varepsilon\mathcal{M} = 1 - \varepsilon(1 - h')$，当 $|\rho| < 1$ 时该自回归过程是稳定的，即 $\varepsilon < 2/(1 - h') = 2(1 + \theta^2)$，小于 1 的任何增益都满足此条件。

其方差为 $\varepsilon^2 \sigma_\eta^2 / (1 - \rho^2)$，而 $1 - \rho^2 = \varepsilon (1 - h')\bigl(2 - \varepsilon(1 - h')\bigr)$ 给出了该公式。

对于较小的增益，$V(\varepsilon) \approx \varepsilon \sigma_\eta^2 / (2|\mathcal{M}|)$，当 $\theta = \sigma_\eta = 1$ 时它等于 $\varepsilon$。

```{code-cell} ipython3
cm_1 = CredibilityModel(θ=1.0, U_star=5.0)
h_prime = cm_1.θ**2 / (1 + cm_1.θ**2)

print("   ε     样本方差          V(ε)    方差 / ε")
for ε in (0.01, 0.025, 0.05, 0.1, 0.2):
    x_ε, _ = ls_learning(cm_1, T=100_000, seed=3, ε=ε)
    e_ε = x_ε[5_000:] - cm_1.nash()
    V = ε / ((1 - h_prime) * (2 - ε * (1 - h_prime)))
    print(f"{ε:6.3f}   {e_ε.var():12.5f}   {V:10.5f}   {e_ε.var() / ε:9.3f}")
```

样本方差与公式相符，且方差几乎与增益成正比。

下图在相同冲击下比较了常数增益与最小二乘增益。

```{code-cell} ipython3
x_c, _ = ls_learning(cm_1, T=3000, seed=4, ε=0.025)
x_ls, _ = ls_learning(cm_1, T=3000, seed=4)
band = 2 * np.sqrt(0.025 / ((1 - h_prime) * (2 - 0.025 * (1 - h_prime))))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(x_c, 'C1', lw=0.8, label=r'常数增益 $\varepsilon = 0.025$')
ax.plot(x_ls, 'C0', lw=1.2, label='最小二乘增益 $1/t$')
ax.axhline(cm_1.nash(), color='k', ls='--', lw=1, label='纳什值 $y^N$')
ax.fill_between(np.arange(3001), cm_1.nash() - band, cm_1.nash() + band,
                color='C1', alpha=0.15, label=r'$y^N \pm 2\sqrt{V(\varepsilon)}$')
ax.set_xlabel('$t$')
ax.set_ylabel('预期通货膨胀 $x_t$')
ax.legend(fontsize=9)
plt.show()
```

常数增益带来的好处是更快的遗忘速度。

常数增益的学习者以每期 $1 - \varepsilon(1 - h')$ 的速率几何式地遗忘初始条件。

最小二乘学习者最初移动得更快，因为其增益 $1/t$ 在前 40 期内都超过 $0.025$。

但正如 {ref}`pc_ex1` 所示，其均值误差此后仅以 $t^{-1/2}$ 的速度缩小，在几百期后就会比常数增益学习者的误差更大。

代价在于常数增益学习者永远不会稳定下来。

它仅依分布收敛，收敛到 $y^N$ 周围的一片"云雾"，其方差与增益成正比。

在这个标量模型中，这片"云雾"就是全部的情形，因为均值动态会把 $x$ 直接拉回 $y^N$，没有别的去处。

在 {doc}`phillips_learning` 中，政府的信念是一整个菲利普斯曲线系数向量，以相同的增益 $0.025$ 进行学习。

在那里，噪声可能将信念推向通往拉姆齐结果的路径，然后均值动态再将其拉回纳什值。

这就是 {cite:t}`Sargent1999` 和 {cite:t}`ChoWilliamsSargent2002` 所研究的反复逃逸现象。

```{solution-end}
```

```{exercise-start}
:label: pc_ex3
```

本练习研究*通货膨胀偏差* $y^N - y^R$ 以及 {ref}`cpol_remedies` 中讨论的两种补救措施。

将政府的支付函数 {eq}`pc_payoff` 替换为 $-\frac{1}{2}\left[(U - k U^*)^2 + \alpha y^2\right]$，其中 $k \in [0, 1]$，$\alpha > 0$。

讲座中的情形对应 $k = 0$，$\alpha = 1$。

1. 证明最优反应为 $B(x) = [\theta(1 - k) U^* + \theta^2 x] / (\theta^2 + \alpha)$，纳什通货膨胀率为 $y^N = \theta (1 - k) U^* / \alpha$。
   通过迭代最优反应来检验该公式。
1. 阿兰·布林德的补救措施是设立一位了解自然失业率、且从不希望失业率偏离自然失业率的机构（$k = 1$）。
   {cite:t}`Rogoff1985` 的补救措施则是设立一位比社会更重视通货膨胀的保守型中央银行家（$\alpha > 1$）。
   这两种措施各自对通货膨胀偏差有何影响？
1. 现在令菲利普斯曲线为 $U = U^* - \theta(y - x) + \zeta$。
   这里 $\zeta$ 是均值为零、方差为 $\sigma_\zeta^2$ 的独立同分布冲击，决策者在设定 $y$ 之前能观测到它，但公众在形成 $x$ 时无法观测到。
   证明在纳什均衡中 $x = \theta(1-k)U^*/\alpha$，$y = x + \theta\zeta/(\theta^2 + \alpha)$，$U = U^* + \alpha \zeta / (\theta^2 + \alpha)$。
1. 社会仍保留支付函数 {eq}`pc_payoff`。
   设 $\theta = 1$，$U^* = 5$。
   对于罗戈夫式银行家（$k = 0$）以及 $\sigma_\zeta \in \{1, 2.5, 5\}$，绘制社会预期损失 $\frac{1}{2} E(U^2 + y^2)$ 关于 $\alpha$ 的图形，并找出最优的 $\alpha$。
   将其与布林德式机构（$k = 1$，$\alpha = 1$）进行比较。

```{exercise-end}
```

```{solution-start} pc_ex3
:class: dropdown
```

**1.** 一阶条件为 $\theta\left[(1-k)U^* - \theta(y - x)\right] = \alpha y$，由此得到 $B(x)$。

令 $x = B(x)$，得到 $\alpha y^N = \theta(1-k)U^*$。

由于 $B$ 的斜率 $\theta^2/(\theta^2 + \alpha)$ 小于 1，最优反应动态会收敛到 $y^N$，这使我们能够检验该公式。

```{code-cell} ipython3
def B_general(x, θ, U_star, k, α):
    "支付函数为 -[(U - k U*)^2 + α y^2]/2 的决策者的最优反应。"
    return (θ * (1 - k) * U_star + θ**2 * x) / (θ**2 + α)


θ_b, U_b = 1.0, 5.0
for k, α in [(0, 1), (0, 2), (0, 5), (0.5, 1), (1, 1)]:
    y = 0.0
    for _ in range(2000):
        y = B_general(y, θ_b, U_b, k, α)
    print(f"k = {k:3}, α = {α}:  迭代最优反应 {y:.4f}，  "
          f"θ(1-k)U*/α = {θ_b * (1 - k) * U_b / α:.4f}")
```

**2.** 两种补救措施都会使失业率维持在自然失业率水平，这是任何决策者平均而言所能达到的最佳结果，因此它们真正改变的是通货膨胀率。

布林德式机构（$k = 1$）彻底消除了偏差：它没有理由将失业率压低到 $U^*$ 以下，因此 $y^N = 0$，即为拉姆齐结果。

罗戈夫式银行家则按 $1/\alpha$ 的比例缩小偏差，只有当 $\alpha \to \infty$ 时才能彻底消除。

**3.** 存在冲击时，一阶条件变为 $\theta\left[(1-k)U^* + \zeta - \theta(y - x)\right] = \alpha y$。

取期望，并利用 $x = E y$，得到 $x = \theta(1-k)U^*/\alpha$；两式相减得到 $y - x = \theta \zeta / (\theta^2 + \alpha)$。

于是 $U = U^* + \zeta - \theta(y - x) = U^* + \alpha\zeta/(\theta^2 + \alpha)$。

更保守的银行家对冲击的对冲程度更低，因此更大的 $\alpha$ 会使失业率*更*不稳定，而使通货膨胀率的波动性降低。

**4.** 社会的预期损失为

$$
\frac{1}{2}\left[U^{*2} + \left(\frac{\theta(1-k)U^*}{\alpha}\right)^2
+ \sigma_\zeta^2\,\frac{\alpha^2 + \theta^2}{(\theta^2 + \alpha)^2}\right] .
$$

中间一项是通货膨胀偏差，最后一项是冲击带来的成本。

偏差项随 $\alpha$ 的增大而减小。

冲击项在 $\alpha = 1$ 时最小，此时等于 $\sigma_\zeta^2/(1 + \theta^2)$，并随着 $\alpha \to \infty$ 逐渐趋于 $\sigma_\zeta^2$。

这就是罗戈夫权衡。

```{code-cell} ipython3
def society_loss(α, k, σ_ζ, θ=θ_b, U_star=U_b):
    "当机构 (k, α) 设定 y 时社会的预期损失 (1/2) E(U^2 + y^2)。"
    bias = θ * (1 - k) * U_star / α
    return 0.5 * (U_star**2 + bias**2
                  + σ_ζ**2 * (α**2 + θ**2) / (θ**2 + α)**2)


α_grid = np.geomspace(1, 100, 4001)
fig, ax = plt.subplots(figsize=(8, 4.5))
for j, σ_ζ in enumerate([1.0, 2.5, 5.0]):
    loss = society_loss(α_grid, 0, σ_ζ) - 0.5 * U_b**2
    i = loss.argmin()
    blinder = society_loss(1.0, 1, σ_ζ) - 0.5 * U_b**2
    ax.plot(α_grid, loss, color=f'C{j}', label=rf'罗戈夫，$\sigma_\zeta = {σ_ζ}$')
    ax.plot(α_grid[i], loss[i], 'o', color=f'C{j}')
    ax.axhline(blinder, color=f'C{j}', ls=':', lw=1)
    print(f"σ_ζ = {σ_ζ}:  最优 α = {α_grid[i]:.2f}，  该处的超额损失 {loss[i]:.3f}；"
          f"  α = 1 处为 {loss[0]:.3f}；  布林德情形为 {blinder:.3f}")
ax.set_xscale('log')
ax.set_xlabel(r'银行家对通货膨胀的权重 $\alpha$')
ax.set_ylabel(r'预期损失减去 $U^{*2}/2$')
ax.legend(fontsize=9)
plt.show()
```

虚线表示布林德式机构。

对于每一个 $\sigma_\zeta$，最优的罗戈夫式银行家都比社会更保守（$\alpha > 1$），但并非无限保守，而且经济的噪声越大，最优银行家的保守程度就越低。

布林德式机构胜过任何罗戈夫式银行家。

它在不扭曲稳定化功能的前提下消除了偏差：其损失 $\frac{1}{2}\left[U^{*2} + \sigma_\zeta^2/(1 + \theta^2)\right]$ 正是社会承诺执行最优状态依存规则 $y = \theta\zeta/(1+\theta^2)$ 所能获得的结果。

原因在于 $-\frac{1}{2}(U - U^*)^2$ 与 $-\frac{1}{2}U^2$ 之间相差一个关于 $U$ 的线性项，而正是这个线性项构成了制造通货膨胀的全部诱因。

```{solution-end}
```
