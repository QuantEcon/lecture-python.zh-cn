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
  title: 自我确认均衡
  headings:
    Overview: 概述
    Objects in the Phelps problem: 费尔普斯问题中的要素
    The actual Phillips curve: 实际菲利普斯曲线
    Self-confirming equilibria: 自我确认均衡
    The special case solved by hand: 手工可解的特殊情形
    The special case solved by hand::Why the classical equilibrium is Nash: 为什么古典均衡是纳什均衡
    The special case solved by hand::Why not Ramsey?: 为什么不是拉姆齐？
    Uniqueness and E-stability: 唯一性与 E-稳定性
    Equilibrium with misspecified beliefs: 带有误设信念的均衡
    Equilibrium with misspecified beliefs::Approaching Ramsey: 趋近拉姆齐
    Equilibrium with misspecified beliefs::The role of $\rho$: $\rho$ 的作用
    Equilibrium with misspecified beliefs::Spectra and impulse responses: 谱与脉冲响应
    Equilibrium with misspecified beliefs::Grounds for optimism: 乐观的理由
    Exercises: 练习
---

(phillips_self_confirming)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 自我确认均衡

```{index} single: Phillips Curve; Self-Confirming Equilibria
```

```{contents} Contents
:depth: 2
```

除了 Anaconda 中已有的库之外，本讲座还将使用以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概述

本讲座完成了 {doc}`phillips_credibility` 中开始研究的*均衡*部分。

它遵循 {cite:t}`Sargent1999` 的第 7 章，在此之后，本系列将从固定信念转向实时学习的信念。

我们寻求这样一些模型：它们与 {doc}`phillips_credibility` 中 {cite:t}`KydlandPrescott1977` 的基本模型偏离最小，但同时也让政府的*信念*由其自身政策所产生的数据来塑造。

关键的均衡概念是**自我确认均衡**：政府对菲利普斯曲线可能持有*错误*的模型，但它通过最小二乘法将该模型拟合到数据上，而它所观测到的数据恰好证实了它的信念。

{cite:t}`FudenbergLevine1993` 在博弈论中使用了自我确认均衡这一术语来描述一个相关的概念。

我们结合了两个文献流派的思想：

* {cite:t}`KingWatson1994` 记录了关于菲利普斯曲线的推断如何取决于*拟合方向*——是将失业率对通货膨胀回归（*古典*识别），还是将通货膨胀对失业率回归（*凯恩斯*识别）。
* 一个可追溯到 Muth、Lucas 和 Prescott 的文献流派 {cite}`muth1961,LucasPrescott1971`，并被 {cite:t}`MarcetSargent1989jet` 和 {cite:t}`EvansHonkapohja2001` 所运用，将理性预期均衡表述为从信念到统计模型总体矩的映射的不动点；{doc}`rational_expectations` 描述了这种不动点的观点。

我们构建了两个除拟合方向外完全相同的自我确认均衡，并发现最小化的方向会影响结果。

这两个均衡都是唯一的，并且都是 E-稳定的。

然后我们研究一个**带有误设信念的均衡**，它将 {doc}`phillips_adaptive` 中的费尔普斯问题与 {doc}`phillips_misspecified` 中的最优误设机制结合起来。

只要公众所感知模型使 {doc}`phillips_misspecified` 中的根 $\rho$ 严格小于一，该均衡就能带来优于纳什均衡的结果，并随着政府变得更有耐心而趋近拉姆齐结果。

在自我确认均衡中，只有政府的模型是错误的；而在带有误设信念的均衡中，只有公众的模型是错误的。

让我们导入所需的库：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
from scipy.optimize import minimize_scalar
```

## 费尔普斯问题中的要素

回顾 {doc}`phillips_adaptive` 中一般费尔普斯问题的构成要素。

政府相信一个简约形式的菲利普斯曲线，它可以按两个方向拟合：

$$
\text{古典：} \quad U_t = \gamma^\top X_{C,t} + \varepsilon_{C,t},
\qquad X_{C,t} = \begin{bmatrix} y_t & X_{t-1}^\top \end{bmatrix}^\top,
$$

$$
\text{凯恩斯：} \quad y_t = \beta^\top X_{K,t} + \varepsilon_{K,t},
\qquad X_{K,t} = \begin{bmatrix} U_t & X_{t-1}^\top \end{bmatrix}^\top,
$$

其中 $X_{t-1}$ 收集了 $U$ 的滞后值、$y$ 的滞后值以及一个常数项。

我们将 $\gamma$ 分块为 $\gamma = \begin{bmatrix} \gamma_1 & \gamma_{-1}^\top \end{bmatrix}^\top$，其中 $\gamma_1$ 是 $y_t$ 的系数，$\gamma_{-1}$ 收集了 $X_{t-1}$ 的系数，并以同样的方式对 $\beta$ 进行分块。

求解费尔普斯问题以政府的信念 $\gamma$ 为给定，得出通货膨胀的决策规则 $h(\gamma)$，即 $\hat y_t = h(\gamma) X_{t-1}$。

这两种参数化通过如下反演公式相关联

```{math}
:label: sc_invert

\gamma_1 = \beta_1^{-1}, \qquad \gamma_{-1} = - \beta_{-1} / \beta_1 .
```

## 实际菲利普斯曲线

*实际的*菲利普斯曲线在早期讲座使用的基础上进行了扩展，以允许序列相关的冲击：

```{math}
:label: sc_actual

U_t = U^* - \frac{\theta}{1 - \rho_2 L}(y_t - x_t) + \frac{v_{1t}}{1 - \rho_1 L},
```

其中 $|\rho_1| < 1$ 且 $|\rho_2| < 1$。

这里 $v_t = (v_{1t}, v_{2t})^\top$ 是一个具有*对角*协方差矩阵的向量白噪声：菲利普斯曲线冲击 $v_{1t}$（方差为 $\sigma_1^2$）与通货膨胀的意外部分 $v_{2t} \equiv y_t - x_t$（方差为 $\sigma_2^2$）不相关。

在本讲座的大部分内容中，我们设 $\rho_1 = \rho_2 = 0$ 以阐明理论要点，这将 {eq}`sc_actual` 化简为

$$
U_t = U^* - \theta(y_t - x_t) + v_{1t} .
$$

## 自我确认均衡

自我确认均衡使政府的信念与这些信念所产生的环境相协调。

```{prf:definition} 自我确认均衡
:label: sc_def

自我确认均衡是一个固定的信念向量 $\gamma$、一条政府决策规则 $h = h(\gamma)$，以及一个 $(y_t, U_t, x_t)$ 的平稳过程，满足

（a）通货膨胀求解费尔普斯问题，$y_t = h X_{t-1} + v_{2t}$；

（b）公众对通货膨胀作出最优预测，$x_t = h X_{t-1}$；

（c）失业率由实际菲利普斯曲线 {eq}`sc_actual` 生成；以及

（d）政府的信念满足最小二乘正交性条件
$\mathbb{E}\left[\left(U_t - \gamma^\top X_{C,t}\right) X_{C,t}^\top\right] = 0$（**古典**拟合方向）。
```

条件（d）使政府的信念依赖于矩矩阵，而这些矩矩阵通过（a）-（c）本身又依赖于政府的信念。

政府的信念隐含着一种行为，该行为产生的数据的矩恰好*证实*了这些信念。

将（d）替换为**凯恩斯**拟合方向，会得到一个不同的自我确认均衡：

> （d′）政府拟合凯恩斯菲利普斯曲线，$\mathbb{E}\left[\left(y_t - \beta^\top X_{K,t}\right) X_{K,t}^\top\right] = 0$，然后通过反演公式 {eq}`sc_invert` 恢复 $\gamma$。

由于政府的信念会影响数据的整个概率分布，最小化的方向会影响结果。

```{note}
一般而言，计算自我确认均衡意味着寻找映射 $\gamma = T(h(\gamma))$
（古典）或 $\beta = S(h(\gamma(\beta)))$（凯恩斯）的不动点。

正交性条件中的矩是通过求解一个离散李雅普诺夫方程，从该系统的状态空间表示中得到的。

实践中，人们迭代一个松弛算法 $\beta_{j+1} = \kappa\beta_j + (1-\kappa) S(\beta_j)$，
即 {doc}`bounded_rationality` 中的阻尼迭代，这与 {doc}`phillips_credibility` 中的最小二乘学习递归相似；
{ref}`sc_estability` 使用了该方法。
```

## 手工可解的特殊情形

当 $\rho_1 = \rho_2 = 0$ 时，政府的问题退化为一系列静态问题，每个自我确认均衡都可以手工计算。

我们取 $X_{t-1} = 1$，因此政府相信 $U_t = \gamma_{-1} + \gamma_1 y_t + \varepsilon_{C,t}$；{ref}`sc_estability` 表明，将 $U$ 和 $y$ 的滞后项加入 $X_{t-1}$ 不会改变任何结果。

在此信念下最小化 $\mathbb{E}\left[U_t^2 + y_t^2\right]$ 得到静态费尔普斯规则

```{math}
:label: sc_h

\hat y = h(\gamma) = - \frac{\gamma_1 \gamma_{-1}}{1 + \gamma_1^2} .
```

实际菲利普斯曲线以及 $v_t$ 的对角协方差矩阵蕴含以下二阶矩

```{math}
:label: sc_moments

\mathbb{V}[U_t] = \theta^2 \sigma_2^2 + \sigma_1^2,
\qquad
\mathbb{V}[y_t] = \sigma_2^2,
\qquad
\operatorname{cov}(U_t, y_t) = -\theta \sigma_2^2 .
```

**古典拟合方向**（$U$ 对 $y$）：斜率为

$$
\gamma_1 = \frac{\operatorname{cov}(U_t, y_t)}{\mathbb{V}[y_t]} = -\theta .
$$

平均失业率为 $U^*$，平均通货膨胀为 $\bar x = h(\gamma)$，而均值必须落在回归线上这一要求 $U^* = \gamma_{-1} + \gamma_1 \bar x$ 给出截距 $\gamma_{-1} = (\gamma_1^2 + 1) U^*$。

**凯恩斯拟合方向**（$y$ 对 $U$）：斜率为

$$
\beta_1 = \frac{\operatorname{cov}(U_t, y_t)}{\mathbb{V}[U_t]}
        = \frac{-\theta \sigma_2^2}{\sigma_1^2 + \theta^2 \sigma_2^2},
$$

截距为 $\beta_{-1} = -\frac{\beta_1^2 + 1}{\beta_1} U^*$，由此通过反演得到隐含的古典系数，$\gamma_1 = \beta_1^{-1}$ 和 $\gamma_{-1} = \frac{\beta_1^2 + 1}{\beta_1^2} U^*$。

```{code-cell} ipython3
class SelfConfirmingStatic:
    """
    The two static self-confirming equilibria (ρ1 = ρ2 = 0), one for
    each direction of fit.  γ1 is the slope; γ_int is the intercept γ_{-1}.
    """

    def __init__(self, θ=1.0, U_star=5.0, σ1=0.3, σ2=0.3):
        self.θ, self.U_star, self.σ1, self.σ2 = θ, U_star, σ1, σ2

    def classical(self):
        "Perceived Phillips curve U = γ_{-1} + γ_1 y under classical fit."
        θ, U_star = self.θ, self.U_star
        γ1 = -θ
        γ_int = (γ1**2 + 1) * U_star
        y_bar = -γ_int * γ1 / (γ1**2 + 1)      # mean (= Nash) inflation
        return γ1, γ_int, y_bar

    def keynesian(self):
        "Perceived Phillips curve under Keynesian fit, inverted to γ."
        θ, U_star, σ1, σ2 = self.θ, self.U_star, self.σ1, self.σ2
        β1 = -θ * σ2**2 / (σ1**2 + θ**2 * σ2**2)
        β_int = -(β1**2 + 1) / β1 * U_star
        γ1 = 1 / β1
        γ_int = (β1**2 + 1) / β1**2 * U_star
        y_bar = β_int / (β1**2 + 1)             # mean inflation
        return γ1, γ_int, y_bar
```

我们针对 {cite:t}`Sargent1999` 第 7 章的参数计算这两个均衡。

```{code-cell} ipython3
sce = SelfConfirmingStatic(θ=1.0, U_star=5.0, σ1=0.3, σ2=0.3)

γ1_C, γ0_C, y_C = sce.classical()
γ1_K, γ0_K, y_K = sce.keynesian()

print("Classical direction of fit")
print(f"  γ_1 = {γ1_C:.1f},  γ_(-1) = {γ0_C:.1f},  mean inflation = {y_C:.1f}")
print("Keynesian direction of fit")
print(f"  γ_1 = {γ1_K:.1f},  γ_(-1) = {γ0_K:.1f},  mean inflation = {y_K:.1f}")
```

这些结果再现了 {cite:t}`Sargent1999` 第 7 章中的数值例子。

在古典拟合方向下，平均通货膨胀为纳什值 $\theta U^* = 5$。

在凯恩斯拟合方向下，政府估计出一条*更平坦*的菲利普斯曲线，并相信权衡关系比实际更有利。

其平均通货膨胀为

```{math}
:label: sc_ykeynes

\bar y_K = - \frac{U^*}{\beta_1} = \theta U^* \left(1 + \frac{\sigma_1^2}{\theta^2 \sigma_2^2}\right),
```

只要 $\sigma_1 > 0$，该值就超过纳什水平。

正如此处所取的 $\sigma_1 = \theta \sigma_2$，它恰好是纳什水平的两倍，即 $10$。

让我们绘制这两条自我确认的菲利普斯曲线，重现图 7.1。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The two self-confirming Phillips curves, one for each direction of fit
    name: fig-sce-two-curves
---
fig, ax = plt.subplots(figsize=(7, 6))

U_grid = np.linspace(0, 12, 100)

# perceived Phillips curves U = γ_{-1} + γ_1 y  =>  y = (U - γ_{-1}) / γ_1
ax.plot(U_grid, (U_grid - γ0_C) / γ1_C, 'C0', label='P: classical fit', lw=2)
ax.plot(U_grid, (U_grid - γ0_K) / γ1_K, 'C1', label='Q: Keynesian fit', lw=2)

ax.plot(sce.U_star, y_C, 'C0o')
ax.annotate('Nash', (sce.U_star, y_C), (sce.U_star + 0.4, y_C - 0.6))
ax.plot(sce.U_star, y_K, 'C1o')
ax.annotate('Keynesian mean', (sce.U_star, y_K),
            (sce.U_star + 0.4, y_K + 0.2))

ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_xlabel('unemployment $U$')
ax.set_ylabel('inflation $y$')
ax.legend()
plt.show()
```

曲线 P 是用古典拟合方向估计出的自我确认菲利普斯曲线；曲线 Q 是用凯恩斯拟合方向估计出的。

由于凯恩斯拟合使政府相信权衡关系更可利用，平均通货膨胀更高。

(sc_why_nash)=
### 为什么古典均衡是纳什均衡

古典自我确认均衡以一个简单的原因再现了 {doc}`phillips_credibility` 中的纳什结果。

其感知截距为 $\gamma_{-1} = (1 + \theta^2) U^* = U^* + \theta \bar x$，其中 $\bar x = \theta U^*$。

因此感知曲线 $U_t = U^* + \theta \bar x - \theta y_t$ 正是在 $x_t = \bar x$ 处取值的真实菲利普斯曲线。

政府的模型对其在均衡中所观察到的一切都是正确的。

它唯一的错误是把 $\bar x$ 视为不受自身规则影响的不变量，而这恰恰是定义纳什政府的假设——即把公众的预期视为给定的。

这就是为什么 {cite:t}`KydlandPrescott1977` 的时间一致（或称“一致”）均衡是一个自我确认均衡（{ref}`olg_phillips`）。

{cite:t}`KalaiLehrer1993Nash` 的读者本可预见到这一点。

当决策者的概率模型涵盖纳什均衡时，学习会收敛到真相，而古典识别方案满足这一涵盖条件。

凯恩斯方案则不然：它使政府对通货膨胀与失业的扰动项之间的同期协方差产生错误解读，从而使结果相对于纳什均衡变得更差。

这两个结论都依赖于实际菲利普斯曲线中的正交性条件：$v_{1t}$ 与 $v_{2t}$ 不相关，且 $v_{2t}$ 与 $\hat y_t$ 正交。

不同的正交性条件将会改变两个均衡中结果的优劣，而这正是尽管我们对政府的建模与金和沃森不同，我们仍与他们的研究相呼应之处。

### 为什么不是拉姆齐？

两个自我确认均衡给出的平均结果都*劣于*拉姆齐结果。

这表明 {doc}`phillips_adaptive` 中命题的假设条件出现了失效——该命题原本认为求解费尔普斯问题最终能维持接近拉姆齐的结果。

失效的是**归纳假设**。

由于 $\rho_1 = \rho_2 = 0$，自我确认均衡是序列不相关的，因此滞后通货膨胀会从经验菲利普斯曲线中消失。

停用归纳假设实际上使政府求解一个单期问题，并关闭了促进更好结果的跨期渠道。

这两个均衡印证了 {cite:t}`Lucas1972` 和 {cite:t}`Sargent1971` 的论断，即理性预期和自然率假设并不蕴含归纳假设，相关讨论见 {ref}`pa_induction`：此处二者均成立，而归纳假设却失效。

(sc_estability)=
## 唯一性与 E-稳定性

对于后续讲座而言，这些均衡有两个重要性质：每个均衡都是唯一的，并且都是 E-稳定的，因此最小二乘学习可以收敛到它。

在静态情形下，这两点都很容易确立。

**唯一性。** 在古典拟合中，斜率 $T_1 = \operatorname{cov}(U_t, y_t)/\mathbb{V}[y_t] = -\theta$ 来自不依赖于政策的矩 {eq}`sc_moments`：政府的信念会改变通货膨胀的均值，但不会改变其方差或它与失业率的协方差。

给定斜率后，截距必须满足

$$
\gamma_{-1} = T_{-1}(\gamma) = U^* + \theta h(\gamma) = U^* + \frac{\theta^2}{1 + \theta^2} \gamma_{-1},
$$

这是一个斜率为 $\theta^2/(1 + \theta^2) < 1$ 的线性方程，因此有唯一解。

凯恩斯情形也是一样，截距方程中的斜率为 $1/(1 + \beta_1^2)$。

**E-稳定性。** 将古典均衡写成从信念到其所生成的总体回归系数这一映射的不动点 $\gamma = T(\gamma)$。

相应的微分方程 $d\gamma/d\tau = T(\gamma) - \gamma$ 就是 {ref}`br_one_recursion` 中的 E-稳定性条件，{doc}`ls_learning` 表明其稳定性决定了最小二乘学习是否能够收敛。

由于 $T_1$ 不依赖于信念，均衡处 $T - I$ 的雅可比矩阵呈三角形，其特征值为 $-1$（斜率）和 $\theta^2/(1 + \theta^2) - 1 = -1/(1 + \theta^2)$（截距）。

对于凯恩斯拟合 $\beta = S(\beta)$，特征值为 $-1$ 和 $1/(1 + \beta_1^2) - 1 = -\beta_1^2/(1 + \beta_1^2)$。

让我们通过对这两个映射进行数值微分来验证这些结果。

```{code-cell} ipython3
def T_classical(γ, θ=1.0, U_star=5.0):
    "Static classical T-map: beliefs (γ_1, γ_{-1}) ↦ regression of U on (y, 1)."
    γ1, γ_int = γ
    y_hat = -γ1 * γ_int / (1 + γ1**2)
    return np.array([-θ, U_star + θ * y_hat])

def S_keynesian(β, θ=1.0, U_star=5.0, σ1=0.3, σ2=0.3):
    "Static Keynesian S-map: beliefs (β_1, β_{-1}) ↦ regression of y on (U, 1)."
    β1, β_int = β
    y_hat = β_int / (1 + β1**2)
    slope = -θ * σ2**2 / (σ1**2 + θ**2 * σ2**2)
    return np.array([slope, y_hat - slope * U_star])

def jacobian(f, x, ε=1e-6):
    "Central-difference Jacobian of f at x."
    J = np.empty((len(x), len(x)))
    for j in range(len(x)):
        dx = np.zeros(len(x))
        dx[j] = ε
        J[:, j] = (f(x + dx) - f(x - dx)) / (2 * ε)
    return J

for θ in [0.5, 1.0, 2.0]:
    γ_sce = np.array([-θ, (1 + θ**2) * 5.0])
    eig = np.linalg.eigvals(jacobian(lambda g: T_classical(g, θ), γ_sce) - np.eye(2))
    print(f"classical, θ = {θ}:  eigenvalues {np.sort(eig).round(4)},"
          f"  -1/(1+θ²) = {-1 / (1 + θ**2):.4f}")

β1 = -0.3**2 / (0.3**2 + 0.3**2)
β_sce = np.array([β1, -(β1**2 + 1) / β1 * 5.0])
eig = np.linalg.eigvals(jacobian(S_keynesian, β_sce) - np.eye(2))
print(f"Keynesian, θ = 1:  eigenvalues {np.sort(eig).round(4)},"
      f"  -β1²/(1+β1²) = {-β1**2 / (1 + β1**2):.4f}")
```

两个均衡都是 E-稳定的。

古典情形下的特征值 $-1/(1 + \theta^2)$ 正是 {ref}`pc_mean_dynamics` 中均值动态的斜率 $\mathcal{M}$，在那里公众通过最小二乘法学习预期通货膨胀。

在系列示例的 $\theta = 1$ 时，它等于 $-1/2$，恰好是递减增益最小二乘法以 $\sqrt{t}$ 速率收敛的临界边界。

而凯恩斯情形下的特征值此处为 $-0.2$，位于该边界之上，因此最小二乘法收敛得更慢，类似于 $t^{-0.2}$。

这些速率描述的是均衡附近的收敛情况。

它们并不支配一个正在学习的政府多久会逃离均衡一次，后者取决于 {ref}`pl_primer` 中的大偏差分析；{ref}`sc_ex2` 展示了二者的区别。

**滞后项。** 这些结果是否依赖于 $X_{t-1} = 1$？

设政府的回归包含每个变量的一期滞后，因此 $X_{t-1} = \begin{bmatrix} U_{t-1} & y_{t-1} & 1 \end{bmatrix}^\top$，而 $\gamma = \begin{bmatrix} \gamma_1 & \gamma_{-1}^\top \end{bmatrix}^\top$ 包含四个元素。

现在政府的规则 $h(\gamma)$ 来自一个真正动态的菲尔普斯问题。

当 $\rho_1 = \rho_2 = 0$ 时，总体回归仍然是精确的。

由于 $x_t = h X_{t-1}$，且 $v_{1t}$ 与每个回归变量正交，

$$
U_t = -\theta y_t + \left[\theta h(\gamma) + \begin{bmatrix} 0 & 0 & U^* \end{bmatrix}\right] X_{t-1} + v_{1t},
$$

因此 $T_1(\gamma) = -\theta$，且 $T_{-1}(\gamma) = \theta h(\gamma)^\top + \begin{bmatrix} 0 & 0 & U^* \end{bmatrix}^\top$。

我们用 {doc}`phillips_adaptive` 中通用的菲尔普斯求解器计算 $h(\gamma)$，此处再次给出以便本讲座可独立运行。

我们验证纳什信念 $\gamma_N = \begin{bmatrix} -\theta & 0 & 0 & (1 + \theta^2) U^* \end{bmatrix}^\top$ 是一个不动点，计算 $T - I$ 雅可比矩阵的特征值，并从随机初始信念出发运行松弛算法。

```{code-cell} ipython3
def phelps_general(γ, δ):
    """
    The general Phelps problem of phillips_adaptive for the believed curve
    U_t = γ[0] y_t + γ[1] U_{t-1} + γ[2] y_{t-1} + γ[3].
    Returns h, with ŷ_t = h @ [U_{t-1}, y_{t-1}, 1].
    """
    γ_y = γ[0]
    c = np.asarray(γ[1:], dtype=float).reshape(-1, 1)
    R = 0.5 * (c @ c.T)
    Q = np.array([[0.5 * (γ_y**2 + 1)]])
    N = 0.5 * γ_y * c.T
    A = np.vstack([c.T, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.array([[γ_y], [1.0], [0.0]])
    lq = qe.LQ(Q, R, A, B, N=N, beta=min(δ, 1 - 1e-7))
    P, F, d = lq.stationary_values()
    return -F[0]

def T_one_lag(γ, δ, θ=1.0, U_star=5.0):
    "Regression of U_t on (y_t, U_{t-1}, y_{t-1}, 1) when ŷ_t = h(γ) X_{t-1}."
    h = phelps_general(γ, δ)
    return np.concatenate([[-θ], θ * h + np.array([0.0, 0.0, U_star])])

for θ in [0.5, 1.0, 2.0]:
    γ_N = np.array([-θ, 0.0, 0.0, (1 + θ**2) * 5.0])
    for δ in [0.9, 0.98]:
        J = jacobian(lambda g: T_one_lag(g, δ, θ), γ_N)
        eig = np.sort(np.linalg.eigvals(J - np.eye(4)).real)
        gap = np.max(np.abs(T_one_lag(γ_N, δ, θ) - γ_N))
        print(f"θ = {θ}, δ = {δ}:  |T(γ_N) - γ_N| = {gap:.1e},  "
              f"eigenvalues {eig.round(4)}")

rng = np.random.default_rng(0)
κ, δ = 0.5, 0.96
γ_N = np.array([-1.0, 0.0, 0.0, 10.0])
limits = []
for trial in range(10):
    γ = γ_N + np.array([0.0, rng.uniform(-0.9, 0.9),
                        rng.uniform(-1.5, 1.5), rng.uniform(-5, 5)])
    for _ in range(2_000):
        γ_new = κ * γ + (1 - κ) * T_one_lag(γ, δ)
        if np.max(np.abs(γ_new - γ)) < 1e-10:
            break
        γ = γ_new
    limits.append(γ_new)
dist = max(np.max(np.abs(g - γ_N)) for g in limits)
print(f"\nrelaxation from 10 random starts: largest distance of the limit from γ_N = {dist:.1e}")
```

无论 $\theta$ 和 $\delta$ 取何值，纳什信念都是不动点，特征值为 $-1$ 和 $-1/(1 + \theta^2)$，其中后者重复出现三次，且与贴现因子无关。

从每个起点出发的松弛算法都收敛到纳什信念，滞后项的权重均为零。

加入滞后项给了政府发现归纳假设的空间，但在自我确证均衡中，滞后权重为零，该假设始终处于未被激活的状态。

{doc}`phillips_learning` 在每个变量取两期滞后时得到了相同的一对特征值（{ref}`pl_eigen`）。

## 带有误设信念的均衡

为了探讨施加一个*不同的*错误模型如何可能*改善*纳什结果，我们现在让*公众*而非政府犯一个微妙的设定误差。

这将适应性预期、归纳假设和费尔普斯问题联系了起来。

政府知道正确的模型，

```{math}
:label: sc_bray10

U_t = U^* - \theta(y_t - x_t) + v_{1t},
\qquad
x_t = C y_{t-1} + (1 - C) x_{t-1}, \quad C \in (0, 1),
```

其中公众持有常数增益的适应性预期，参数 $C$ 由其*根据数据进行调整*。

将 $x_t$ 视为状态变量，政府求解费尔普斯问题：通过选择一个反馈规则 $y_t = f_1 + f_2 x_t + v_{2t}$，最大化 $-\mathbb{E}_0 \sum_{t=0}^\infty \delta^t\left[(U^* - \theta(y_t - x_t))^2 + y_t^2\right]$。

这恰好是 {doc}`phillips_adaptive` 中带有适应参数 $\lambda = 1 - C$ 的 LQ 费尔普斯问题；下面的代码重复了该过程，以便本讲座可以独立运行。

```{code-cell} ipython3
def phelps_policy(C, δ, θ=1.0, U_star=5.0):
    "Government feedback rule y_t = f1 + f2 x_t, with public gain C."
    λ = 1 - C
    a = np.array([[U_star], [θ]])
    R = 0.5 * (a @ a.T)
    Q = np.array([[0.5 * (θ**2 + 1)]])
    N = -0.5 * θ * a.T
    A = np.array([[1.0, 0.0], [0.0, λ]])
    B = np.array([[0.0], [1 - λ]])
    lq = qe.LQ(Q, R, A, B, N=N, beta=min(δ, 1 - 1e-7))
    P, F, d = lq.stationary_values()
    return -F[0, 0], -F[0, 1]
```

政府的行为使得*实际*通货膨胀率为

```{math}
:label: sc_bray11

y_t = \frac{f_1}{1 - f_2}
    + \frac{1 - (1 - C)L}{1 - (1 - C(1 - f_2))L} v_{2t}
    = \nu + f(L) v_{2t},
```

均值为 $\nu = f_1 / (1 - f_2)$，谱为 $F(\omega; C) = |f(e^{i\omega})|^2 \sigma_2^2$。

均值 $\nu$ 是费尔普斯规则的稳态，因此 {doc}`phillips_adaptive` 中推导出的闭式解 $y_\infty = \theta U^*(1 - \delta)/(1 - \delta\lambda)$（令 $\lambda = 1 - C$）给出了它：

```{math}
:label: sc_nu

\nu = \theta U^* \, \frac{1 - \delta}{1 - \delta + \delta C} .
```

只要 $C > 0$，平均通货膨胀就低于纳什利率 $\theta U^*$；在 $C$ 固定时，随着 $\delta \to 1$ 它趋于零，而随着 $C \to 0$ 它则上升至 $\theta U^*$。

因此，一切都取决于 $C$ 的均衡值。

给定 $C$，公众寻求与 {doc}`phillips_misspecified` 中相同形式的最优拟合误设模型，即平稳的 ARMA(1,1) 模型，

$$
y_t = \frac{1 - (1 - c)L}{1 - \rho L}\epsilon_t = g(L)\epsilon_t ,
$$

其中根 $\rho$ 略小于 $1$（不要与 {eq}`sc_actual` 中的根 $\rho_1, \rho_2$ 混淆），这使得该模型能够以接近单位根的持续性来拟合均值 $\nu$。

由此得出的最优估计映射 $c = B(C)$ 将**预测误设均衡**定义为不动点 $C = B(C)$。

这使得真实模型和近似模型都成为均衡结果，并将适应参数 $C$ 从自由参数转变为一个结果。

最优估计代码与 {doc}`phillips_misspecified` 中相同。

```{code-cell} ipython3
def forecast_mse(F, ν, c, ρ, z):
    """
    Mean square one-step forecast error of the perceived model
    g(L) = (1 - (1-c)L)/(1 - ρL) when the true process has mean ν and
    spectral density F on the grid z = exp(iω).  ρ = 1 is allowed.
    """
    W = np.abs((1 - ρ * z) / (1 - (1 - c) * z))**2
    return np.mean(F * W) + ν**2 * W[0]

class MisspecifiedPhillips:
    """
    Equilibrium with misspecified public beliefs: a fixed point of the
    best-estimate map, with the government solving the Phelps problem.
    """

    def __init__(self, θ=1.0, U_star=5.0, σ2=1.0, δ=0.97, ρ=0.995, N=1024):
        self.θ, self.U_star, self.σ2 = θ, U_star, σ2
        self.δ, self.ρ, self.N = δ, ρ, N
        ω = 2 * np.pi * np.arange(N) / N
        self.z = np.exp(1j * ω)
        self.ω = ω

    def true_process(self, C):
        "Return the mean ν, spectrum F, and policy (f1, f2) given belief C."
        f1, f2 = phelps_policy(C, self.δ, self.θ, self.U_star)
        ν = f1 / (1 - f2)
        ψ = 1 - C * (1 - f2)
        f = (1 - (1 - C) * self.z) / (1 - ψ * self.z)
        F = np.abs(f)**2 * self.σ2**2
        return ν, F, (f1, f2)

    def best_estimate(self, C):
        "The best-estimate map c = B(C)."
        ν, F, _ = self.true_process(C)
        return minimize_scalar(lambda c: forecast_mse(F, ν, c, self.ρ, self.z),
                               bounds=(1e-4, 0.99), method='bounded',
                               options={'xatol': 1e-10}).x

    def solve(self, C0=0.1, tol=1e-10, maxit=500):
        "Iterate the best-estimate map to a fixed point."
        C = C0
        for _ in range(maxit):
            C_new = self.best_estimate(C)
            if abs(C_new - C) < tol:
                break
            C = C_new
        ν, _, (f1, f2) = self.true_process(C_new)
        return C_new, ν, (f1, f2)
```

**通货膨胀意外的大小。**

{cite:t}`Sargent1999` 第 7 章在此示例中说明 $\sigma_1 = \sigma_2 = 0.3$。

其所报告的数字是用 $\sigma_2 = 1$ 重现的，这也与其所述真实模型的一步预测误差标准差为 1 的说法一致。

我们使用 $\sigma_2 = 1$。

这一选择很重要，因为拟合准则不是尺度不变的。

{doc}`phillips_misspecified` 中的准则等于 $\sigma_2^2$ 乘以一个关于 $c$ 的函数，该函数仅通过被感知模型必须拟合的均值与意外规模之比 $\nu/\sigma_2$ 来依赖于 $\sigma_2$。

较小的 $\sigma_2$ 会使均值相对更大，从而促使公众选择更大的增益，由 {eq}`sc_nu`，这会降低平均通货膨胀。

```{code-cell} ipython3
mp = MisspecifiedPhillips(δ=0.97)
C_star, ν_star, (f1, f2) = mp.solve()

print(f"equilibrium gain     C = {C_star:.4f}")
print(f"policy rule          y = {f1:.4f} + {f2:.4f} x")
print(f"mean inflation       ν = {ν_star:.3f}   (formula: "
      f"{mp.θ * mp.U_star * (1 - mp.δ) / (1 - mp.δ + mp.δ * C_star):.3f})")
print(f"Nash inflation   θ U*  = {mp.θ * mp.U_star:.3f}")

C_03, ν_03, _ = MisspecifiedPhillips(δ=0.97, σ2=0.3).solve()
print(f"\nwith σ2 = 0.3:  C = {C_03:.4f},  mean inflation = {ν_03:.3f}")
```

当 $\sigma_2 = 1$ 时，这些正是 {cite:t}`Sargent1999` 第 7 章所报告的值。

最重要的结果是：隐含的平均通货膨胀率大幅*低于*纳什值 $5$。

嵌入在适应性预期方案中的归纳假设，连同较高的贴现因子，共同带来了这一改进。

由于 $C$ 现在是一个均衡结果，这一机制比 {doc}`phillips_adaptive` 中更为鲜明，在那里 $\lambda = 1 - C$ 可以独立于 $\delta$ 被操纵。

与 {doc}`phillips_misspecified` 一样，该均衡具有 E-稳定性。

```{code-cell} ipython3
h = 1e-4
B_prime = (mp.best_estimate(C_star + h) - mp.best_estimate(C_star - h)) / (2 * h)
print(f"B'(C*) = {B_prime:.3f}")
```

此处 $B'(C^*)$ 为负但大于 $-1$，因此 $B$ 的迭代会收敛，并围绕 $C^*$ 振荡。

斜率为负的一个原因是：更高的增益会促使政府进一步压低通货膨胀，这降低了公众模型必须拟合的均值，从而降低了公众所希望的增益。

### 趋近拉姆齐

随着政府变得更有耐心，均衡平均通货膨胀率会降至拉姆齐值零附近。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 随着政府变得更有耐心，平均通货膨胀率与均衡增益的变化
    name: fig-sce-patience
---
δ_grid = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
C_vals, ν_vals = [], []
for δ in δ_grid:
    C, ν, _ = MisspecifiedPhillips(δ=δ).solve()
    C_vals.append(C)
    ν_vals.append(ν)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(δ_grid, ν_vals, 'o-')
axes[0].axhline(mp.θ * mp.U_star, color='k', ls='--', lw=1, label='Nash')
axes[0].set_xlabel(r'discount factor $\delta$')
axes[0].set_ylabel('mean inflation')
axes[0].legend()

axes[1].plot(δ_grid, C_vals, 'o-', color='C1', lw=2)
axes[1].set_xlabel(r'discount factor $\delta$')
axes[1].set_ylabel('equilibrium gain $C$')

plt.tight_layout()
plt.show()

print(f"{'δ':>6} {'C':>8} {'ν':>7}")
for δ, C, ν in zip(δ_grid, C_vals, ν_vals):
    print(f"{δ:>6} {C:>8.4f} {ν:>7.2f}")
ν_book = mp.θ * mp.U_star * 0.01 / (0.01 + 0.99 * 0.019)
print(f"\nformula (sc_nu) at the book's δ = 0.99 entry, C = 0.019:  ν = {ν_book:.2f}")
```

在每个贴现因子下，平均通货膨胀都远低于纳什值，并随着 $\delta \to 1$ 趋近于拉姆齐值零。

$C$ 和 $\nu$ 的值与 {cite:t}`Sargent1999` 第 7 章脚注 14 中给出的 $\delta = 0.95$、$0.96$ 和 $0.98$ 的结果相符。

而该书对 $\delta = 0.99$ 所给出的条目（$C = 0.019$，平均通货膨胀 $0.25$）与 {eq}`sc_nu` 不一致——如上所印，在 $C = 0.019$ 时该公式给出的平均通货膨胀要高得多。

### $\rho$ 的作用

以上所有数值都是在公众被感知的根 $\rho = 0.995$ 的条件下计算的。

{ref}`pm_rho` 表明，在布雷模型中，均衡增益随着 $\rho \to 1$ 而消失。

下面的代码单元使用更精细的频率网格，针对两个 $\sigma_2$ 值考察这里会发生什么。

```{code-cell} ipython3
print(f"{'ρ':>7} {'C (σ2=1)':>10} {'ν (σ2=1)':>10} {'C (σ2=.3)':>11} {'ν (σ2=.3)':>11}")
for ρ in [0.99, 0.995, 0.999, 0.9995]:
    row = []
    for σ2 in [1.0, 0.3]:
        C, ν, _ = MisspecifiedPhillips(δ=0.97, σ2=σ2, ρ=ρ, N=2**14).solve()
        row += [C, ν]
    print(f"{ρ:>7} {row[0]:>10.4f} {row[1]:>10.3f} {row[2]:>11.4f} {row[3]:>11.3f}")

m1 = MisspecifiedPhillips(δ=0.97, ρ=1.0, N=2**14)
ratio = max(m1.best_estimate(C) / C for C in np.linspace(0.005, 0.3, 30))
print(f"\nρ = 1: largest value of B(C)/C on the grid = {ratio:.3f}")
```

对于两个 $\sigma_2$ 值，平均通货膨胀都随着 $\rho \to 1$ 而上升至纳什值。

当恰好为单位根时，对每个 $C$ 都有 $B(C) < C$，因此均衡增益为零，公众的预期不再变动，由 {eq}`sc_nu`，平均通货膨胀即为纳什利率 $\theta U^*$。

因此，优于纳什的结果对每个固定的 $\rho < 1$ 都成立，而随着 $\rho \to 1$ 则会消失。

对每个固定的 $\rho < 1$，当 $\delta \to 1$ 时结果趋近拉姆齐；但对每个固定的 $\delta < 1$，当 $\rho \to 1$ 时结果会消失（{ref}`sc_ex3` 研究了这两种极限）。

其原因与 {ref}`pm_rho` 中给出的相同：朝向大增益的拉力来自于若 $\rho < 1$，被感知模型会留下的均值偏差，而这一偏差会随着公众预测规则实际内含的单位根而消失。

### 谱与脉冲响应

让我们比较均衡处真实与近似的通货膨胀过程，如 {cite:t}`Sargent1999` 的图 7.2 和图 7.3 所示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 均衡处真实与近似通货膨胀过程的谱密度
    name: fig-sce-spectra
---
ν_star, F, _ = mp.true_process(C_star)
c_star = mp.best_estimate(C_star)
σ_ε2 = forecast_mse(F, ν_star, c_star, mp.ρ, mp.z)
H = np.abs((1 - (1 - c_star) * mp.z) / (1 - mp.ρ * mp.z))**2
G = H * σ_ε2

half = mp.N // 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mp.ω[:half], np.log(F[:half]), 'C0', label='true model', lw=2)
ax.plot(mp.ω[:half], np.log(G[:half]), 'C1--', label='approximating model', lw=2)
ax.set_xlabel(r'angular frequency $\omega$')
ax.set_ylabel('log spectral density')
ax.legend()
plt.show()
```

真实谱密度与近似谱密度除了在最低频率处之外都拟合得很好。

真实的通货膨胀率仅具有中等程度的序列相关性，而——正如 {doc}`phillips_misspecified` 中的布雷模型一样——近似模型使用近单位根来模拟均值，即用二阶矩来捕捉一阶矩。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 真实与近似通货膨胀过程的脉冲响应
    name: fig-sce-irf
---
def ima_impulse(num, den, T=25):
    "IRF of (1 - num L)/(1 - den L)."
    h = np.empty(T)
    h[0] = 1.0
    for k in range(1, T):
        h[k] = den * h[k - 1]
    h[1:] -= num * h[:-1]
    return h

ψ = 1 - C_star * (1 - f2)
irf_true = ima_impulse(1 - C_star, ψ)
irf_approx = ima_impulse(1 - c_star, mp.ρ)

lags = np.arange(len(irf_true))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, start in zip(axes, [0, 1]):
    ax.plot(lags[start:], irf_true[start:], 'C0o-', ms=4, label='true model', lw=2)
    ax.plot(lags[start:], irf_approx[start:], 'C1s--', ms=4,
            label='approximating model', lw=2)
    ax.set_xlabel('lag')
    ax.set_ylabel('response')
axes[0].set_title('all lags')
axes[1].set_title('lags 1 and beyond')
axes[0].legend()
plt.tight_layout()
plt.show()
```

此处两个模型都对单位大小的一步预测误差做出响应，因为 $f(0) = g(0) = 1$。

近似模型中的近单位根表现为一个仅以速率 $\rho$ 衰减的响应，在所绘的时间范围内看起来像是一个非零渐近值。

### 乐观的理由

在自我确认均衡带来失望之后，带有预测误设的均衡令人振奋：它支持了优于纳什的结果。

这个均衡概念不是自我确认的，但它具有那种精神——它体现了一种用*错误*模型进行自我确认的类型。

近似误差很小。

```{code-cell} ipython3
print(f"one-step RMSE of the approximating model / true innovation s.d. = "
      f"{np.sqrt(σ_ε2) / mp.σ2:.4f}")
```

近似模型的一步预测误差仅略大于真实的新息，因此存在一个近乎自我确认、且结果远优于纳什的模型。

但这种乐观是有条件的。

良好的结果依赖于公众所感知的模型具有一个严格小于 1 的根 $\rho$；若其预测规则实际内含的是恰好的单位根，均衡增益为零，结果即为纳什结果。

这一示例所表明的是：公众模型中一个微小、难以察觉的误设，可能对结果产生巨大影响。

本系列后续的讲座——{doc}`phillips_learning`、{doc}`phillips_escaping_nash` 和 {doc}`phillips_priors`——构建了这些模型的自适应实时版本，其中政府从最新数据中递归地估计其菲利普斯曲线。

在那里，经典（纳什）自我确认均衡成为学习动态的吸引子。

在常数增益下，政府会反复从该吸引子逃逸向拉姆齐，又被拉回，于是优于纳什的结果在逃逸期间重新出现。

而这些逃逸是通过*政府*的信念来实现的，而非通过公众的误设。

## 练习

```{exercise-start}
:label: sc_ex1
```

两个静态自我确认均衡之间的差距是由方差 $\sigma_1^2$（菲利普斯曲线冲击的方差）和 $\sigma_2^2$（通货膨胀意外的方差）驱动的。

古典均衡总是给出等于纳什值 $\theta U^*$ 的平均通货膨胀，但凯恩斯平均通货膨胀取决于比率 $\sigma_1 / \sigma_2$。

固定 $\sigma_2 = 0.3$、$\theta = 1$、$U^* = 5$，绘制凯恩斯平均通货膨胀作为 $\sigma_1 \in [0.05, 1.0]$ 的函数，并与公式 {eq}`sc_ykeynes` 进行比较。

当 $\sigma_1 \to 0$ 时会发生什么，为什么？

```{exercise-end}
```

```{solution-start} sc_ex1
:class: dropdown
```

```{code-cell} ipython3
σ1_grid = np.linspace(0.05, 1.0, 50)
y_keynes = [SelfConfirmingStatic(σ1=σ1, σ2=0.3).keynesian()[2]
            for σ1 in σ1_grid]
y_formula = 5.0 * (1 + σ1_grid**2 / 0.3**2)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(σ1_grid, y_keynes, lw=2, label='Keynesian mean inflation')
ax.plot(σ1_grid, y_formula, 'k:', lw=2, label='formula')
ax.axhline(5.0, color='k', ls='--', lw=1, label='Nash')
ax.set_xlabel(r'$\sigma_1$')
ax.set_ylabel('mean inflation')
ax.legend()
plt.show()
```

计算出的值恰好落在公式 $\theta U^*(1 + \sigma_1^2/(\theta^2 \sigma_2^2))$ 所给出的曲线上。

当 $\sigma_1 \to 0$ 时，菲利普斯曲线趋于近乎确定，凯恩斯斜率 $\beta_1 \to -1/\theta$，两个拟合方向趋于一致，因此凯恩斯平均通货膨胀趋近纳什值。

随着 $\sigma_1$ 增大，失业率变得更嘈杂，$y$ 对 $U$ 的凯恩斯回归变得更平坦，政府感知到一个更可利用的权衡关系，使平均通货膨胀随 $\sigma_1$ 二次方式上升。

```{solution-end}
```

```{exercise-start}
:label: sc_ex2
```

自我确认均衡是最小二乘学习的静止点。

下面展示了递减增益学习在静态模型中的表现，在该模型中政府每期都重新估计其菲利普斯曲线。

每一期，政府根据其当前信念通过 {eq}`sc_h` 设定 $\hat y_t$，公众预期 $x_t = \hat y_t$，$(U_t, y_t)$ 由实际的菲利普斯曲线生成（其中 $\rho_1 = \rho_2 = 0$），政府以增益 $1/(t + t_0)$ 通过递归最小二乘法更新其回归。

值 $t_0 = 100$ 表示一个包含 100 次观测的先验样本，回归的矩矩阵从自我确认均衡中的总体值开始。

1. 对于每个拟合方向，从各自的自我确认均衡开始，模拟 500 条长度为 20,000 期的路径。
2. 对于凯恩斯拟合，证明 $\hat y_t$ 与其均衡值 $10$ 的均方根偏差大致以 $t^{-0.2}$ 的速度缩小，正如其 E-稳定性特征值所预测的那样。
3. 对于古典拟合，计算目标值最终偏离纳什值 $5$ 超过一个单位的路径所占的比例。
4. 利用目标值 $\hat y_t$ 随时间变化时 $(U_t, y_t)$ 的矩，解释两者之间的差异。

```{exercise-end}
```

```{solution-start} sc_ex2
:class: dropdown
```

```{code-cell} ipython3
def rls_static(direction, n_paths=500, T=20_000, t0=100,
               θ=1.0, U_star=5.0, σ1=0.3, σ2=0.3, seed=0):
    """
    Decreasing-gain recursive least squares in the static model.
    Beliefs are (intercept, slope).  Returns the targets ŷ_t, shape
    (n_paths, T), and the target in the self-confirming equilibrium.
    """
    rng = np.random.default_rng(seed)
    if direction == 'classical':                    # regress U on (1, y)
        y_sce = θ * U_star
        b0 = np.array([(1 + θ**2) * U_star, -θ])
        R0 = np.array([[1.0, y_sce], [y_sce, y_sce**2 + σ2**2]])
    else:                                           # regress y on (1, U)
        β1 = -θ * σ2**2 / (σ1**2 + θ**2 * σ2**2)
        y_sce = -U_star / β1
        b0 = np.array([(1 + β1**2) * y_sce, β1])
        R0 = np.array([[1.0, U_star],
                       [U_star, U_star**2 + θ**2 * σ2**2 + σ1**2]])
    b = np.tile(b0, (n_paths, 1))
    R = np.tile(R0, (n_paths, 1, 1))
    targets = np.empty((n_paths, T))
    ones = np.ones(n_paths)

    for t in range(T):
        intercept, slope = b[:, 0], b[:, 1]
        if direction == 'classical':
            y_hat = -slope * intercept / (1 + slope**2)
        else:
            y_hat = intercept / (1 + slope**2)
        targets[:, t] = y_hat

        v1 = σ1 * rng.standard_normal(n_paths)
        v2 = σ2 * rng.standard_normal(n_paths)
        y = y_hat + v2
        U = U_star - θ * v2 + v1
        if direction == 'classical':
            X, lhs = np.column_stack([ones, y]), U
        else:
            X, lhs = np.column_stack([ones, U]), y

        g = 1 / (t + 1 + t0)
        R = R + g * (X[:, :, None] * X[:, None, :] - R)
        err = lhs - np.sum(X * b, axis=1)
        b = b + g * np.linalg.solve(R, (X * err[:, None])[:, :, None])[:, :, 0]

    return targets, y_sce

results = {d: rls_static(d) for d in ['classical', 'keynesian']}
t_axis = np.arange(1, 20_001)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for d, color in [('classical', 'C0'), ('keynesian', 'C1')]:
    targets, y_sce = results[d]
    axes[0].plot(t_axis, targets[:15].T, color=color, lw=0.6, alpha=0.7)
    axes[0].axhline(y_sce, color=color, ls='--', lw=1.5, label=f'{d.capitalize()} SCE')
    far = np.mean(np.abs(targets - y_sce) > 1.0, axis=0)
    axes[1].plot(t_axis, far, color=color, lw=2, label=d.capitalize())
axes[0].set_xlabel('$t$')
axes[0].set_ylabel(r'target inflation $\hat y_t$')
axes[0].legend()
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('fraction of paths more than 1 from the SCE')
axes[1].legend()
plt.tight_layout()
plt.show()

targets, y_sce = results['keynesian']
rms = np.sqrt(np.mean((targets - y_sce)**2, axis=0))
keep = slice(200, None)
slope = np.polyfit(np.log(t_axis[keep] + 100), np.log(rms[keep]), 1)[0]
print(f"Keynesian: slope of log RMS deviation on log t = {slope:.2f}  (eigenvalue -0.2)")
for d in ['classical', 'keynesian']:
    targets, y_sce = results[d]
    print(f"{d}: fraction of paths more than 1 from the SCE at the end = "
          f"{np.mean(np.abs(targets[:, -1] - y_sce) > 1.0):.2f}")
```

凯恩斯学习者收敛得很慢，速度大致符合其特征值所预测的速率。

古典学习者的特征值 $-1/2$ 本应预示更快的收敛，但其表现却差得多：很大一部分路径逃逸向拉姆齐方向，在 20,000 期之后仍远离纳什值。

原因在于目标值变化时数据所呈现的特征。

由于只有意外冲击才会影响失业率，$\hat y_t$ 的变化会改变 $y_t$，但不会改变 $U_t$。

在 $U$ 对 $y$ 的回归中，这种变化只会增加回归元的方差，而不增加其与 $U$ 的协方差，因此会使估计的斜率趋于平坦，趋近于零。

更平坦的感知曲线使费尔普斯法则 {eq}`sc_h` 建议更低的通货膨胀，这进一步移动了目标值，并再次使斜率变平：形成一种自我强化的逃逸，趋向拉姆齐。

在 $y$ 对 $U$ 的回归中，同样的变化只会增加因变量的方差，而斜率不受影响。

E-稳定性是关于均衡附近小偏差的论断；逃逸属于大偏差，这在 {doc}`phillips_learning` 和 {doc}`phillips_escaping_nash` 中有详细研究。

在递减增益下，最小二乘法仍会渐近地收敛到均衡，但较早开始的逃逸可能会持续很长时间，因为不断下降的增益也会减缓回归的速度（见 {ref}`pl_decreasing`）。

```{solution-end}
```

```{exercise-start}
:label: sc_ex3
```

公式 {eq}`sc_nu` 表明，误设信念均衡中的平均通货膨胀取决于 $\delta$ 以及均衡增益 $C$，而 $C$ 又取决于 $\delta$ 和公众感知的根 $\rho$。

1. 对 $\delta \in \{0.97, 0.99, 0.999\}$ 和 $\rho \in \{0.995, 0.9995, 1\}$，取 $\sigma_2 = 1$，计算均衡的 $C$ 和平均通货膨胀 $\nu$。
2. 当 $\rho$ 固定、$\delta \to 1$ 时会发生什么？当 $\delta$ 固定、$\rho \to 1$ 时又会发生什么？
3. 这对“随着政府变得更有耐心，均衡趋近于拉姆齐”这一论断有何含义？

```{exercise-end}
```

```{solution-start} sc_ex3
:class: dropdown
```

```{code-cell} ipython3
def ν_closed(C, δ, θ=1.0, U_star=5.0):
    "Mean inflation (sc_nu) given the public's gain C."
    return θ * U_star * (1 - δ) / (1 - δ + δ * C)

print(f"{'δ':>6} {'ρ':>7} {'C':>8} {'ν':>7}")
for δ in [0.97, 0.99, 0.999]:
    for ρ in [0.995, 0.9995]:
        C, ν, _ = MisspecifiedPhillips(δ=δ, ρ=ρ, N=2**14).solve()
        print(f"{δ:>6} {ρ:>7} {C:>8.4f} {ν:>7.3f}")
    m1 = MisspecifiedPhillips(δ=δ, ρ=1.0, N=2**14)
    ratio = max(m1.best_estimate(C) / C for C in np.linspace(0.005, 0.3, 20))
    print(f"{δ:>6} {1:>7} {0:>8.4f} {ν_closed(0.0, δ):>7.3f}"
          f"   (largest B(C)/C = {ratio:.2f})")
```

对于固定的 $\rho < 1$，随着 $\delta$ 增大，平均通货膨胀趋向于零，因为均衡增益下降的速度比 {eq}`sc_nu` 中的 $1 - \delta$ 要慢得多。

对于固定的 $\delta$，随着 $\rho \to 1$，平均通货膨胀趋向纳什值，因为均衡增益趋于零，而当 $\rho = 1$ 时，对于每个 $\delta < 1$，它恰好等于 $\theta U^*$。

这两个极限并不可交换。

“随着政府变得更有耐心，均衡趋近于拉姆齐”这一论断对每个固定的 $\rho < 1$ 都成立，但在 $\rho$ 上并不一致：公众感知的根越接近其法则所内含的单位根，政府就需要越有耐心才能接近拉姆齐。

```{solution-end}
```
