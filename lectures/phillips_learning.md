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
  title: 适应性学习与逃逸动态
  headings:
    Overview: 概述
    A primer on recursive algorithms: 递归算法入门
    A primer on recursive algorithms::Beliefs, moment conditions, and self-confirming equilibria: 信念、矩条件与自我确认均衡
    A primer on recursive algorithms::Iteration: 迭代法
    A primer on recursive algorithms::Stochastic approximation: 随机逼近
    A primer on recursive algorithms::Mean dynamics: 均值动态
    A primer on recursive algorithms::Constant gain and convergence in distribution: 常数增益与依分布收敛
    A primer on recursive algorithms::Escape routes and the theory of large deviations: 逃逸路径与大偏差理论
    A primer on recursive algorithms::A tractable action functional: 一个可处理的行动泛函
    A primer on recursive algorithms::From computation to adaptation: 从计算到适应
    The adaptive model: 适应性模型
    The adaptive model::Government beliefs and behavior: 政府的信念与行为
    The adaptive model::The Phelps problem with lags: 带滞后项的菲尔普斯问题
    Least squares learning converges: 最小二乘法学习收敛
    Constant gain and escape dynamics: 常数增益与逃逸动态
    The escape route and the induction hypothesis: 逃逸路径与归纳假设
    Relation to equilibria under forecast misspecification: 与预测误设均衡的关系
    Role of the discount factor: 折扣因子的作用
    Anticipated utility: 预期效用
    Conclusions: 结论
    Exercises: 练习
---

(phillips_learning)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 适应性学习与逃逸动态

```{contents} Contents
:depth: 2
```

## 概述

本讲座是*菲利普斯曲线权衡*系列讲座的集大成之作。

它遵循 {cite}`Sargent1999` 的第 8 章，这是该书最具野心的一章。

在 {doc}`phillips_self_confirming` 中，政府对菲利普斯曲线持有*固定*的信念——这些信念被由它们本身所生成的数据所证实。

在这里，我们把政府变成一个实时经济计量学家。

每一期它都：

* 根据到目前为止观测到的数据，通过递归最小二乘法重新估计其菲利普斯曲线；并且
* 根据其*当前*的估计值，设定通货膨胀率为 {doc}`phelps 问题 <phillips_adaptive>` 第一期建议的水平。

我们要问的是：这样一个适应性的政府是否会收敛到自我确认均衡。

答案取决于一个单一的参数——支配旧数据被折扣速度的**增益**（gain）：

* 若采用实现最小二乘法的*递减*增益，均值动态会将经济拉向自我确认均衡，我们不会得到什么新结果：系统被困在纳什结果附近。
* 若采用*常数*增益，主体会对过去数据打折扣，收敛被阻止，**新的结果便会涌现**。系统会反复地从自我确认均衡*逃逸*向拉姆齐（零通胀）结果——这种自发的稳定化现象，颇似沃尔克时代的到来。

这些逃逸正是 {doc}`phillips_two_stories` 中"econometric 政策评估的证明"故事的核心：一个学习索洛-托宾分布滞后版本自然率假说的适应性政府，会因偶然的观察结果而稳定住通货膨胀。

本讲座复现、分析并重新诠释了类似克里斯托弗·西姆斯（Christopher Sims）和郑喜泰（Heetaik Chung）的模拟结果 {cite}`Sims1988,Chung1990`。

在这里，我们通过*模拟*来研究这些逃逸；{doc}`phillips_escaping_nash` 随后将其解析地刻画为第二个确定性 ODE，而 {doc}`phillips_priors` 则探讨政府关于系数漂移的先验如何重塑这两种力量。

让我们导入所需的库：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
```

## 递归算法入门

本节是一个独立完整的入门介绍。

它构建了贯穿后文全部内容的两个分析对象——**均值动态**（mean dynamics）与**逃逸路径**（escape routes）——并解释了同一个递归公式既可以被视为计算自我确认均衡的*算法*，也可以被视为政府实时适应的*模型*的意义所在。

这部分内容较为技术性，只想了解结论的读者可以直接跳到下面的模拟部分，需要时再回来查阅。

### 信念、矩条件与自我确认均衡

在经典识别方案下，自我确认均衡由政府关于某些总体矩及其所隐含的回归系数的信念所确定。

在经典识别方案下，这些信念由三元组 $(\gamma, \, E X_{C} X_{C}', \, E U X_{C})$ 度量，其中 $\gamma$ 是菲利普斯曲线系数向量。

在本讲座的适应性模型中，这些对象的*时间 $t$ 的取值*是经济的状态变量之一；它们只有在自我确认均衡中才不再作为状态变量存在，因为在那里它们是常数。

经典识别方案下的自我确认均衡满足以下矩条件：

```{math}
:label: pl_scemoments

\begin{aligned}
E\, R_{XC}^{-1}(\gamma)\left[ U_t X_{Ct}' - \left(X_{Ct} X_{Ct}'\right)\gamma \right] &= 0, \\
E\, X_{Ct} X_{Ct}' - R_{XC}(\gamma) &= 0,
\end{aligned}
```

其中数学期望是关于 $(U_t, X_{Ct})$ 的分布计算的，而该分布通过菲尔普斯问题的解 $h(\gamma)$ 依赖于 $\gamma$。

自我指涉正是通过分布对 $\gamma$ 的这种依赖性而浮现出来的：政府的信念塑造其政策，而政策塑造了随后用来检验信念的数据。

{eq}`pl_scemoments` 的第一行是 $\gamma$ 的最小二乘正规方程，两边预先乘以二阶矩矩阵的逆；第二行将 $R_{XC}$ 定义为回归量的二阶矩矩阵。

将所有未知量组合成一个向量会比较方便：

```{math}
:label: pl_phivec

\phi = \begin{bmatrix} \gamma \\ \operatorname{col}(R_{XC}) \end{bmatrix},
```

其中 $\operatorname{col}(R_{XC})$ 将 $R_{XC}$ 的各列堆叠起来。

矩条件 {eq}`pl_scemoments` 随即可以写成紧凑形式：

```{math}
:label: pl_bdef

E\left[F(\phi, \zeta)\right] = 0,
\qquad
b(\phi) \equiv E\left[F(\phi, \zeta)\right],
```

其中 $\zeta$ 是一个随机向量，期望是关于其分布计算的（同样，该分布依赖于 $\phi$）。

自我确认均衡是 $b$ 的一个零点，即一组信念 $\phi_f$，满足：

```{math}
:label: pl_scezero

b(\phi_f) = 0 .
```

本节余下的部分描述了寻找这样一个零点的递归算法，以及一种将每个计算算法转化为实时适应模型的视角转换。

### 迭代法

最简单的算法通过以下公式计算估计值序列 $\{\phi_k\}$：

```{math}
:label: pl_iterate

\phi_{k+1} = \phi_k + a\, b(\phi_k),
```

其中，用来计算 {eq}`pl_bdef` 中定义 $b(\phi_k)$ 的期望所用的分布，本身是在当前估计值 $\phi_k$ 处计算的，而 $a > 0$ 是步长。

这正是 {doc}`phillips_self_confirming` 中用来计算自我确认均衡的松弛算法。

每一步都需要计算数学期望 $b(\phi) = E[F(\phi, \zeta)]$——这正是我们在那里需要矩（李雅普诺夫）公式的原因。

### 随机逼近

将 {eq}`pl_iterate` 中的均值 $b(\phi_n)$ 替换为单次随机抽样 $F(\phi_n, \zeta_n)$，并让*步长*来完成平均，就得到了 {eq}`pl_iterate` 的一个随机版本：

```{math}
:label: pl_sa

\phi_{n+1} = \phi_n + a_n F(\phi_n, \zeta_n),
\qquad
a_n > 0, \quad \sum_{n=0}^\infty a_n = +\infty .
```

要研究 {eq}`pl_sa` 的极限行为，我们定义**人为时间**（artificial time）：

```{math}
:label: pl_artificial

t_n = \sum_{k=0}^n a_k ,
```

构造抽样过程 $\phi(t_n) = \phi_n$，并对其进行插值（通常是分段线性插值），得到一个连续时间过程 $\phi^o(t)$。

随后，随着 $n \to \infty$，用一个连续时间过程来逼近 $\phi^o(t)$，并利用它来刻画原始序列的尾部行为。

增益序列 $\{a_n\}$ 不同的递减速率会产生不同的逼近过程，因为它们改变了从实际时间 $n$ 到人为时间 $t_n$ 的映射 {eq}`pl_artificial`。

```{note}
递归随机逼近起源于 {cite}`RobbinsMonro1951`，他设计了 {eq}`pl_sa` 用于在噪声观测下寻找回归函数的根；以及 {cite}`KieferWolfowitz1952`，他将其改造用于寻找回归函数的最大值（下文提到的"K-W"算法）。用于分析此类递归的"ODE 方法"——即用微分方程的解来逼近插值过程——归功于 {cite}`Ljung1977`；书籍级的详尽处理见 {cite}`BenvenisteMetivierPriouret1990` 和 {cite}`KushnerYin2003`。将其用于研究自我指涉宏观经济模型中的学习问题，由 {cite}`MarcetSargent1989` 开创，并被 {cite}`EvansHonkapohja2001` 全面发展。
```

### 均值动态

{cite}`KushnerClark1978` 和 {cite}`Ljung1977` 的经典随机逼近算法将增益设定为按 $a_n \sim 1/n$ 的速度递减（至少对某个 $N > 0$ 满足 $t \geq N$）。

这使得我们能够对 {eq}`pl_sa` 几乎必然收敛到 $b(\phi)$ 的零点这一结论做出强有力的论断。

当 $a_n \sim 1/n$ 时，随着 $n \to \infty$，插值过程 $\phi^o(t)$ 逼近以下常微分方程的解：

```{math}
:label: pl_ode

\frac{d \phi^o(t)}{dt} = b\left(\phi^o(t)\right),
```

我们称之为**均值动态**——这是 {doc}`phillips_credibility` 附录中推导的标量最小二乘学习 ODE 的向量化推广。

大数定律使连续时间逼近中的随机项以足够快的速度消失，从而使均值动态 {eq}`pl_ode` 刻画了随机过程 {eq}`pl_sa` 的*尾部*行为。

因此：

* 如果算法收敛（几乎必然地），它就收敛到均值动态的零点 $b(\phi) = 0$——即一个自我确认均衡；并且
* ODE {eq}`pl_ode` 携带着关于算法局部与全局稳定性的信息。

局部稳定性由 $b$ 在静止点处雅可比矩阵的特征值支配：如果所有特征值的实部都为负，该静止点就是局部稳定的；而在关于增益的一定条件下，实部最大的特征值支配着收敛的*速率*（通常的 $\sqrt{T}$ 速率要求该特征值低于 $-\tfrac12$）。

我们将在下面看到，在我们所用参数下，经典模型的这个特征值恰好位于 $-\tfrac12$ 的边界上，因此收敛是边际性的——事实证明，这一点对系统能否轻易逃逸至关重要。

```{note}
{cite}`BrockHommes1997` 构建的模型，其全局行为由远离理性预期均衡的稳定均值动态，以及在均衡附近适应的局部不稳定性共同驱动——这是一种从学习中产生内生波动的互补机制。
```

### 常数增益与依分布收敛

我们同样关注 {eq}`pl_sa` 的另一版本，其中对所有 $n$ 都采用*常数*增益 $a_n = \epsilon > 0$。

常数增益算法的极限定理使用了一种弱于 $a_n \sim 1/n$ 时可用的几乎必然收敛的收敛概念——即依分布收敛。

它们关注的是当 $\epsilon \to 0$ 同时 $n\epsilon \to +\infty$ 时的小噪声极限。

同样使用人为时间 {eq}`pl_artificial`，构造过程族：

```{math}
:label: pl_cgain

\phi_{n+1}^\epsilon = \phi_n^\epsilon + \epsilon\, F(\phi_n^\epsilon, \zeta_n),
```

对其插值以得到 $\phi^\epsilon(t)$，并研究其小 $\epsilon$ 极限。

杜皮伊（Dupuis）与库什纳（Kushner），见 {cite}`KushnerYin2003`，以及其他学者验证了在何种条件下，当 $\epsilon \to 0$ 且 $\epsilon n \to \infty$ 时，过程 $\phi_n^\epsilon$ *依分布*收敛到同一均值动态 {eq}`pl_ode` 的零点；均值动态所需满足的限制条件与经典 $a_n \sim 1/n$ 理论中的要求相同。

与递减增益算法不同，常数增益算法不会稳定下来：$(\gamma, R_{XC})$ 会收敛到一个*平稳随机过程*，围绕自我确认均衡永久地波动——并偶尔远离它。

### 逃逸路径与大偏差理论

对本讲座而言，常数增益机制中最重要的特征不是向 $\phi_f$ 的收敛，而是*远离它的偏离*。

我们对远离自我确认均衡的运动与趋向它的运动同样感兴趣，因为模拟中反复出现的稳定化现象恰恰就是这样的偏离。

**大偏差理论**通过三个对象来刻画这些偏离。

首先是（创新过程 $F(\phi_n, \zeta_n)$ 某种平均化版本的）对数矩生成函数：对于与 $F$ 维数一致的向量 $\theta$，

```{math}
:label: pl_mgf

H(\theta, \phi) = \log E \exp\left(\theta' F(\phi, \zeta)\right),
```

其中期望是关于 $\zeta$ 的分布计算的。

```{note}
方程 {eq}`pl_mgf` 是一个启发性的简写。实际进入理论的对象是一个*时间平均*极限；{cite}`DupuisKushner1987` 和 {cite}`KushnerYin2003` 假设对每个 $\delta > 0$，以下极限在任意紧集上关于 $\phi_i, \alpha_i$ 一致存在：
$$
\sum_{i=0}^{T/\delta - 1} \delta\, H(\alpha_i, \phi_i)
= \lim_{N \to \infty} \frac{\delta}{N}
  \log E \exp \sum_{i=0}^{T/\delta - 1} \alpha_i'
  \sum_{j=iN}^{iN+N-1} F(\phi_i, \zeta_j) .
$$
内部的求和对长度为 $N$ 的一个区块内的创新项做平均；双重极限使我们能够处理序列相关的创新项。
```

其次是 $H$ 的**勒让德变换**（Legendre transform），它扮演速率函数的角色：

```{math}
:label: pl_legendre

L(\beta, \phi) = \sup_\theta \left[ \theta'\beta - H(\theta, \phi) \right] .
```

第三是**行动泛函**（action functional），它衡量候选逃逸路径 $\phi(\cdot)$ 的"代价"：

```{math}
:label: pl_action

S(T, \phi) =
\begin{cases}
\displaystyle \int_0^T L\!\left(\tfrac{d}{ds}\phi(s),\, \phi(s)\right) ds
  & \text{如果 } \phi(s) \text{ 绝对连续且 } \phi(0) = \phi_f, \\[2mm]
\infty & \text{否则。}
\end{cases}
```

杜皮伊和库什纳将寻找最可能逃逸路径的问题转化为一个*确定性控制问题*。

设 $D$ 是包含 $\phi_f$ 的一个紧集，其边界为 $\partial D$，设 $C[0,T]$ 是 $[0,T]$ 上的连续函数集合。

逃逸路径是求解以下问题的路径 $\tilde\phi(\cdot)$：

```{math}
:label: pl_escapeproblem

\inf_{T > 0} \; \inf_{\phi \in A} S(T, \phi),
\qquad
A = \left\{ \phi(\cdot) \in C[0,T] : \phi(T) \in \partial D \right\} .
```

假设最小化路径 $\tilde\phi(\cdot)$ 是唯一的，并设 $t_D^\epsilon$ 为常数增益过程 $\phi^\epsilon(t)$ 离开 $D$ 的首个时刻，{cite}`DupuisKushner1987` 证明了对每个 $\delta > 0$ 都有：

```{math}
:label: pl_escapelim

\lim_{\epsilon \to 0} \operatorname{Prob}\left(
\left| \phi^\epsilon(t_D^\epsilon) - \tilde\phi(T) \right| > \delta
\right) = 0 .
```

换句话说：*在系统逃离集合 $D$ 的条件下，它离开该集合的位置会接近最小行动路径的终点*——因此逃逸具有确定的方向和形态，尽管它是由偶然性触发的。

正是在这个意义上，下文中的逃逸虽然不需要任何巨大的冲击触发，却"看起来有目的性"：它们遵循 {eq}`pl_escapeproblem` 所规定的最小行动路径。

一个关键的对比：均值动态 {eq}`pl_ode` *不*依赖于其周围的噪声，而逃逸路径*却*依赖于噪声——噪声不仅在 {eq}`pl_ode` 周围添加随机波动，它还开辟出了这第二类路径。

```{note}
其数学基础是 {cite}`FreidlinWentzell1998`（特别是其第 4 章）关于随机扰动动态系统的大偏差理论，被 {cite}`DupuisKushner1987` 和 {cite}`DupuisKushner1989` 专门用于随机逼近；一种弱收敛处理见 {cite}`DupuisEllis1997`。
```

### 一个可处理的行动泛函

逃逸路径的计算承诺提供关于算法中心趋势的低成本信息，但行动泛函 {eq}`pl_action` 通常难以计算。

一个重要的特例极大地简化了它。

假设创新项是加性的且服从高斯分布：

$$
F(\phi, \zeta) = b(\phi) + \sigma(\phi)\, \zeta,
$$

其中 $\zeta_n$ 是平稳的且服从高斯分布，但不一定是序列不相关的，并定义 $R = \sum_j E\, \zeta_t \zeta_{t-j}'$。

那么行动泛函取如下二次形式：

```{math}
:label: pl_action2

S(T, \phi) = \frac{1}{2} \int_0^T
\left(\tfrac{d}{ds}\phi - b(\phi)\right)'
\left[\sigma(\phi)\, R\, \sigma(\phi)'\right]^{+}
\left(\tfrac{d}{ds}\phi - b(\phi)\right)
h(s)\, ds ,
```

其中 $(\cdot)^{+}$ 是摩尔-彭若斯广义逆（用于处理 $\sigma R \sigma'$ 可能出现的随机奇异性）。

权重 $h(s)$ 依赖于增益：当 $a_n = a_0 / n^\gamma$ 中的 $\gamma = 1$ 时，$h(s) = \exp(s)$；当 $\gamma < 1$ 时，$h(s) = 1$。

将 {eq}`pl_action2` 理解为一种代价，它惩罚*实际漂移* $\tfrac{d}{ds}\phi$ 与*均值漂移* $b(\phi)$ 之间的偏离，并按局部噪声协方差 $\sigma R \sigma'$ 的逆对每个方向加权。

因此最小行动逃逸路径会穿过均值动态较弱而噪声信息量较大的区域——而在我们的模型中，这恰恰是*归纳假设*所指的方向。

```{note}
这个二次行动泛函正是 {cite}`ChoWilliamsSargent2002` 中所最小化的对象，该文是本讲座模型的正式发表版本。他们求解了纳什自我确认均衡的控制问题 {eq}`pl_escapeproblem`，并从解析上证明了最小行动逃逸会将通货膨胀权重之和推向激活归纳假设的值——也就是趋向拉姆齐结果。{cite}`SargentWilliams2005` 研究了政府先验（等价地，即增益算法的协方差结构，我们的 $P_0$ 与遗忘因子）如何重塑逃逸，而 {cite}`Kasa2004` 将同样的大偏差机制应用于反复出现的货币危机。
```

### 从计算到适应

前述递归公式是作为逼近自我确认均衡的*算法*被引入的。

同样的数学也告诉我们，当我们将自我确认均衡模型*改造*为纳入实时适应时会发生什么——只需将 $\phi_n$ 理解为政府在时间 $n$ 的信念，而非某个求解器的第 $n$ 次迭代结果。

以下两个事实统领了后文的一切：

1. 实现最小二乘法的增益序列（按 $1/t$ 递减）使均值动态将经济拉*向*自我确认均衡；而
2. 递减速度更慢的增益序列——极限情形下即为对过去打折扣的常数增益——则*阻止*了这种拉力，并增加了逃逸动态影响结果的频率。

```{note}
一段简短的思想史。{cite}`Lucas_Prescott_1971` 曾摒弃对矩条件 {eq}`pl_scezero` 进行迭代这一计算策略，但 {cite}`Townsend1983` 却使用了它。{cite}`Woodford1990` 和 {cite}`MarcetSargent1989` 用均值动态 {eq}`pl_ode` 建立了含有自我指涉的模型中最小二乘学习收敛于理性预期的条件，两者都要求 $b(\phi)$ 的连续性。曹寅坤（In-Koo Cho）研究了带有*不连续* $b(\phi)$ 的问题，这种不连续性源自可信度和搜索问题中不连续的决策规则（触发策略）；为了使最小二乘学习逼近理性预期，他使用了满足 $\tfrac{1}{\log n} < a_n < \tfrac{1}{\sqrt n}$ 的增益，这为 {eq}`pl_sa` 产生了一个*扩散*逼近，促进了足够的试探以发现均衡。{cite}`KandoriMailathRob1993` 运用相关数学方法，通过突变在博弈中选择长期均衡，而罗杰·迈尔森（Roger Myerson）将逃逸路径的计算应用于一个投票问题。这些学习方法的现代综合体现在 {cite}`EvansHonkapohja2001` 中。
```

## 适应性模型

现在我们来构建经典的适应性模型。

### 政府的信念与行为

政府相信存在一个分布滞后的菲利普斯曲线：

```{math}
:label: pl_belief

U_t = \gamma' X_{C,t} + \varepsilon_{C,t},
\qquad
X_{C,t} = \begin{bmatrix} y_t & U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}' .
```

在时间 $t$ 到达时，凭借估计值 $\gamma_{t-1}$，政府通过求解菲尔普斯问题来设定通货膨胀的系统性部分，*就好像* $\gamma_{t-1}$ 将永远支配菲利普斯曲线一样：

```{math}
:label: pl_rule

y_t = h(\gamma_{t-1}) X_{t-1} + v_{2t},
\qquad
X_{t-1} = \begin{bmatrix} U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}' .
```

随后它通过**递归最小二乘法**（RLS）更新其信念：

```{math}
:label: pl_rls

\begin{aligned}
\gamma_t &= \gamma_{t-1} + g_t R_{XC,t}^{-1} X_{C,t}\left(U_t - \gamma_{t-1}' X_{C,t}\right), \\
R_{XC,t} &= R_{XC,t-1} + g_t\left(X_{C,t} X_{C,t}' - R_{XC,t-1}\right),
\end{aligned}
```

其中 $\{g_t\}$ 是增益序列。

最小二乘法设定 $g_t = 1/t$；常数增益算法设定 $g_t = g_0 > 0$，并对过去的观测打折扣，如果政府怀疑菲利普斯曲线会随时间漂移，这样做是合理的。

假设公众知道政府的规则，因此其通货膨胀预测为 $x_t = h(\gamma_{t-1}) X_{t-1}$，即 {eq}`pl_rule` 中的系统性部分。

失业率是由 {doc}`phillips_self_confirming` 中实际的菲利普斯曲线在 $\rho_1 = \rho_2 = 0$ 时生成的：

$$
U_t = U^* - \theta(y_t - x_t) + v_{1t} = U^* - \theta v_{2t} + v_{1t} .
$$

### 带滞后项的菲尔普斯问题

给定一个信念 $\gamma$，决策规则 $h(\gamma)$ 求解一个 LQ 控制问题。

将所信奉的菲利普斯曲线写为 $U_t = \gamma_0 y_t + c' s_t$，其中 $\gamma_0$ 是当期通货膨胀的系数，$c$ 收集了状态 $s_t = X_{t-1}$ 上的系数。

政府最小化 $E\sum_t \delta^t (U_t^2 + y_t^2)$，因此每期损失为 $s_t' (cc') s_t + (\gamma_0^2 + 1) y_t^2 + 2\gamma_0\, y_t\, c' s_t$，状态按 $s_{t+1} = A s_t + B y_t$ 演化，其中：

$$
s_{t+1} = \begin{bmatrix} U_t \\ U_{t-1} \\ y_t \\ y_{t-1} \\ 1 \end{bmatrix},
\qquad
U_t = c' s_t + \gamma_0 y_t .
$$

我们使用 `scipy` 的离散代数李卡提方程求解器来求解这个折扣 LQ 问题。

```{code-cell} ipython3
class AdaptivePhillips:
    """
    Classical adaptive Phillips curve model: the government re-estimates a
    distributed-lag Phillips curve by recursive least squares and each period
    acts on the first-period recommendation of the Phelps problem.
    """

    def __init__(self, θ=1.0, U_star=5.0, σ1=0.3, σ2=0.3, δ=0.98):
        self.θ, self.U_star, self.σ1, self.σ2, self.δ = θ, U_star, σ1, σ2, δ

        # classical self-confirming belief: U = -θ y + (θ²+1)U*
        self.γ_sce = np.array([-θ, 0.0, 0.0, 0.0, 0.0, (θ**2 + 1) * U_star])

        # self-confirming moment matrix M = E[X_C X_C'] and residual variance
        self.M = self._sce_moments()
        self.σC2 = σ1**2                       # var(U | X_C) at the SCE

    def _sce_moments(self):
        "E[X_C X_C'] at the serially-uncorrelated classical SCE."
        θ, σ1, σ2 = self.θ, self.σ1, self.σ2
        μU, μy = self.U_star, self.θ * self.U_star
        Σ = {('U', 'U'): θ**2 * σ2**2 + σ1**2, ('y', 'y'): σ2**2,
             ('U', 'y'): -θ * σ2**2, ('y', 'U'): -θ * σ2**2}
        # regressors: (time, type) for [y_t, U_{t-1}, U_{t-2}, y_{t-1}, y_{t-2}, 1]
        regs = [(0, 'y'), (1, 'U'), (2, 'U'), (1, 'y'), (2, 'y'), (None, 'c')]
        mean = {'U': μU, 'y': μy, 'c': 1.0}
        M = np.zeros((6, 6))
        for i, (ti, tyi) in enumerate(regs):
            for j, (tj, tyj) in enumerate(regs):
                if tyi == 'c' or tyj == 'c' or ti != tj:
                    M[i, j] = mean[tyi] * mean[tyj]
                else:
                    M[i, j] = Σ[(tyi, tyj)] + mean[tyi] * mean[tyj]
        return M

    def phelps_h(self, γ):
        "Government decision rule ŷ_t = h(γ)·X_{t-1} for belief γ."
        δ, γ0, c = self.δ, γ[0], γ[1:]
        R = np.outer(c, c)
        Q = np.array([[γ0**2 + 1.0]])
        N = (γ0 * c).reshape(1, -1)
        A = np.zeros((5, 5)); B = np.zeros((5, 1))
        A[0, :] = c; B[0, 0] = γ0          # U_t
        A[1, 0] = 1.0                       # U_{t-1}
        B[2, 0] = 1.0                       # y_t
        A[3, 2] = 1.0                       # y_{t-1}
        A[4, 4] = 1.0                       # constant
        sb = np.sqrt(δ)
        Ad, Bd = sb * A, sb * B
        P = solve_discrete_are(Ad, Bd, R, Q, s=sb * N.T)
        F = np.linalg.solve(Q + Bd.T @ P @ Bd, Bd.T @ P @ Ad + N)
        return -F.ravel()
```

我们使用 {cite}`Sargent1999` 附录 A 中递归最小二乘法的卡尔曼滤波实现。

遗忘因子 $\lambda \in (0, 1]$ 映射到增益：$\lambda = 1$ 给出最小二乘法（$g_t \to 1/t$），而 $\lambda < 1$ 给出常数增益 $g_0 = 1 - \lambda$。

先验被初始化，就好像政府已经观测到 $T$ 期的自我确认均衡数据一样，通过 $P_0 = (\sigma_C^2 / T)\, M^{-1}$；$T$ 越大意味着先验越紧。

```{code-cell} ipython3
def simulate(model, λ, T_prior, n=1000, seed=0):
    "Simulate the adaptive system. λ=1 is least squares; λ<1 is constant gain."
    rng = np.random.default_rng(seed)
    θ, U_star, σ1, σ2 = model.θ, model.U_star, model.σ1, model.σ2

    γ = model.γ_sce.copy()
    P = (model.σC2 / T_prior) * np.linalg.inv(model.M)
    R2 = model.σC2
    g0 = 1 - λ

    U1 = U2 = U_star
    y1 = y2 = θ * U_star
    y_path, U_path, sumweights, constant = (np.empty(n) for _ in range(4))

    for t in range(n):
        h = model.phelps_h(γ)
        X_lag = np.array([U1, U2, y1, y2, 1.0])
        yhat = h @ X_lag

        v2, v1 = σ2 * rng.standard_normal(), σ1 * rng.standard_normal()
        y = yhat + v2
        U = U_star - θ * (y - yhat) + v1

        φ = np.array([y, U1, U2, y1, y2, 1.0])           # X_C,t
        denom = R2 + φ @ P @ φ
        gain = P @ φ / denom
        γ = γ + gain * (U - γ @ φ)
        R1 = (g0 / (1 - g0)) * P if λ < 1 else 0.0        # constant vs decreasing
        P = P - np.outer(P @ φ, φ @ P) / denom + R1

        y_path[t], U_path[t] = y, U
        sumweights[t] = γ[0] + γ[3] + γ[4]               # weights on current+lagged y
        constant[t] = γ[5]
        U2, U1 = U1, U
        y2, y1 = y1, y

    return dict(y=y_path, U=U_path, sumweights=sumweights, constant=constant)
```

```{code-cell} ipython3
model = AdaptivePhillips()
h_sce = model.phelps_h(model.γ_sce)
print("decision rule at the self-confirming belief:")
print(f"  h(γ_sce) = {np.round(h_sce, 3)}  (a constant rule of "
      f"{h_sce[-1]:.1f} = Nash inflation)")
```

在自我确认信念处，菲尔普斯规则是一个等于纳什通货膨胀率的常数，信念自我复制——适应性系统的静止点正是 {doc}`phillips_self_confirming` 中的自我确认均衡。

## 最小二乘法学习收敛

我们遵循 {cite}`Sargent1999` 的做法，将真实的数据生成参数设定为 {doc}`phillips_self_confirming` 结尾处的经典示例：$U^* = 5$、$\theta = 1$、$\sigma_1 = \sigma_2 = 0.3$、$\rho_1 = \rho_2 = 0$、$\delta = 0.98$。

经典自我确认均衡具有序列不相关的 $(U, y)$，围绕均值 $(5, 5)$ 波动。

首先，采用最小二乘法（递减增益）。

```{code-cell} ipython3
ls = simulate(model, λ=1.0, T_prior=5000, n=1000, seed=1)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(ls['y'], lw=0.8)
ax.axhline(5, color='k', ls='--', lw=1, label='self-confirming (Nash)')
ax.axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.set_title('Figure 8.1: classical adaptive model, least squares')
ax.legend()
plt.show()
```

在最小二乘法下，均值动态占据主导：通货膨胀紧贴着自我确认值 5，模拟结果看起来就像是从自我确认均衡本身抽取出来的一样。

我们没有得到什么新结果——政府被困在纳什结果附近。

## 常数增益与逃逸动态

现在赋予政府一个*常数*增益 $\lambda = 0.975$，使其对过去数据打折扣。

```{code-cell} ipython3
cg = simulate(model, λ=0.975, T_prior=300, n=1000, seed=1)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(cg['y'], lw=0.8)
ax.axhline(5, color='k', ls='--', lw=1, label='self-confirming (Nash)')
ax.axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.set_title('Figure 8.2: classical adaptive model, constant gain '
             r'($\lambda = 0.975$)')
ax.legend()
plt.show()
```

图景完全不同了。

通货膨胀最初接近自我确认值 5，随后几乎跌落至零并停留在那里很长一段时间，然后缓慢地朝 5 迈进，却又再一次被推向零。

将系统拉向自我确认均衡的均值动态，遭到了一种反复出现的力量的对抗，这种力量将通货膨胀推向接近拉姆齐结果的水平。

关键在于，没有大的冲击触发这些稳定化：它们是常数增益学习动态的一个*内生*特征。

```{code-cell} ipython3
print(f"constant-gain inflation: mean {cg['y'].mean():.2f}, "
      f"fraction of periods near Ramsey (y<2): {(cg['y'] < 2).mean():.0%}")
```

## 逃逸路径与归纳假设

为什么系统会朝拉姆齐方向逃逸，而不是朝其他方向？

答案是 {doc}`phillips_adaptive` 中的**归纳假设**：当估计出的菲利普斯曲线中当期与滞后通货膨胀的权重之和趋近于零时，菲尔普斯问题会建议政府*降低*通货膨胀。

让我们将通货膨胀与这一权重之和一起绘图。

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(cg['y'], lw=0.8)
axes[0].axhline(0, color='C2', ls=':', lw=1)
axes[0].set_ylabel('inflation $y_t$')

axes[1].plot(cg['sumweights'], lw=0.8, color='C1')
axes[1].axhline(-1, color='k', ls='--', lw=1, label='self-confirming value')
axes[1].axhline(0, color='C3', ls=':', lw=1, label='induction hypothesis')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('sum of weights on $y$')
axes[1].legend()

fig.suptitle('Escape route: stabilizations coincide with the sum of '
             'weights rising toward zero')
plt.tight_layout()
plt.show()
```

每一次稳定化都恰好对应着权重之和从其自我确认值 $-1$ 跳向零。

当它达到零时，归纳假设（暂时）得到满足，菲尔普斯问题要求近乎拉姆齐水平的通货膨胀，而由此产生的数据短暂地*强化*了归纳假设——从技术意义上说这不是自我确认的，但却是自我强化的。

我们可以通过绘制估计出的菲利普斯曲线中常数项与权重之和的联合路径，直接看出这条逃逸路径。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(cg['constant'], cg['sumweights'], c=np.arange(len(cg['y'])),
                cmap='viridis', s=6)
ax.axhline(0, color='C3', ls=':', lw=1.5, label='induction hypothesis')
ax.axhline(-1, color='k', ls='--', lw=1, label='self-confirming value')
ax.set_xlabel('constant in estimated Phillips curve')
ax.set_ylabel('sum of weights on $y$')
ax.legend()
plt.colorbar(sc, label='time $t$')
plt.show()
```

信念大部分时间都停留在自我确认值附近（权重之和 $\approx -1$），但反复地朝归纳线（权重之和 $= 0$）猛冲——这正是逃逸路径，沿着这条路径，政府学会了一个索洛-托宾版本的自然率假说并稳定住了通货膨胀。

## 与预测误设均衡的关系

近乎拉姆齐的这些插曲，让人想起 {doc}`phillips_misspecified` 和 {doc}`phillips_self_confirming` 中带有最优误设预测的均衡。

在那里，一个*不含常数项但含单位根*的预测模型能够很好地逼近一个*含有*常数项的真实模型。

在这里，一种类似风味的逼近在近乎拉姆齐的插曲中发挥作用：政府估计出的菲利普斯曲线，通过将权重之和推向零，利用归纳假设来逼近一个常数项——只不过被逼近的模型并非固定不变，而是随着政府自身的信念通过菲尔普斯问题反馈回来而不断变化。

## 折扣因子的作用

朝拉姆齐方向反复出现的稳定化，取决于折扣因子 $\delta$ 是否接近于 1。

降低 $\delta$ 会提高低通货膨胀插曲期间观测到的通货膨胀率，这与归纳假设下菲尔普斯问题的运作机制是一致的。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 4.5))
for δ in [0.90, 0.95, 0.98]:
    m = AdaptivePhillips(δ=δ)
    sim = simulate(m, λ=0.975, T_prior=300, n=1000, seed=1)
    ax.plot(sim['y'], lw=0.7, label=rf'$\delta = {δ}$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.legend()
ax.set_title('Escapes toward Ramsey deepen as the government becomes patient')
plt.show()
```

## 预期效用

这个适应性模型是大卫·克雷普斯（David Kreps）所称的**预期效用**（anticipated utility）模型的一个例子 {cite}`Kreps1998`。

政府对一个临时误设的模型——一个系数固定的菲利普斯曲线——加以调整，纳入最新的观测数据，并沿途重新优化。

在 $t$ 时刻做决策时，它的行为就好像其当前的估计值 $\gamma_{t-1}$ 将永远支配菲利普斯曲线一样，使用的是若 $\gamma$ 真的是时间不变的话本应是最优的同一政策函数 $h(\cdot)$。

这是对理性预期的一个微小偏离：日历时间只通过漂移的信念 $\gamma_t$ 进入。

与贝叶斯或稳健型决策者不同，一个预期效用型政府忽视了自身逐期模型误设的问题——它并不考虑自己的系数会漂移，即便它们确实在漂移。

然而，正如模拟所示，这种对理性的适度偏离已足以对美国通货膨胀的兴衰给出一个丰富的解释，并为"econometric 政策评估的证明"提供支撑。

## 结论

在很长的时间段内，一个适应性政府学会了产生*优于纳什*的结果。

这些结果来自适应性的反复动态：在最小二乘法下将系统拉向自我确认均衡的均值动态继续发挥作用，但在常数增益下，噪声使系统能够反复地逃逸向拉姆齐结果。

从一个自我确认均衡出发，适应性算法逐渐使政府将足够的权重放在归纳假设上，以至于偶然的观测结果最终促成了一次稳定化。

适应性使得政府的信念成为一种隐藏状态，它给通货膨胀和失业率注入了序列相关性——因此，一个外部预测者若使用随机系数模型，或者进行卢卡斯在其批判 {cite}`lucas1976econometric` 中所指出的那种不断调整，会做得更好。

在这个意义上，适应性模型包含了证明 econometric 政策评估的基础——这正是 {doc}`phillips_two_stories` 中两个故事的第二个。

## 练习

```{exercise-start}
:label: pl_ex1
```

构建**凯恩斯主义**适应性模型，其中政府沿相反方向拟合菲利普斯曲线，将通货膨胀对失业率做回归。

回归量是 $X_{K,t} = \begin{bmatrix} U_t & U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}'$，政府在 $y_t = \beta' X_{K,t} + \varepsilon_{K,t}$ 中估计 $\beta$，然后在求解菲尔普斯问题之前将其反转为 $\gamma$。

不必重新推导所有内容，而是探索*经典*模型对常数增益的敏感性：用 $\lambda \in \{0.99, 0.975, 0.95\}$ 进行模拟，比较通货膨胀朝拉姆齐逃逸的频率。

更大的增益（更小的 $\lambda$，对过去更快地打折扣）如何影响逃逸的频率？

```{exercise-end}
```

```{solution-start} pl_ex1
:class: dropdown
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 4.5))
for λ in [0.99, 0.975, 0.95]:
    sim = simulate(model, λ=λ, T_prior=300, n=1000, seed=1)
    frac = (sim['y'] < 2).mean()
    ax.plot(sim['y'], lw=0.6,
            label=rf'$\lambda = {λ}$ (near-Ramsey {frac:.0%})')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.legend()
plt.show()
```

更大的常数增益（更小的 $\lambda$）对过去的数据打折扣更重，从而更有力地阻止了向自我确认均衡的收敛，产生了更频繁——尽管也更嘈杂——的朝拉姆齐结果的逃逸。

```{solution-end}
```

```{exercise-start}
:label: pl_ex2
```

图 8.1 与图 8.2 的对比取决于增益，但最小二乘法的结果同样取决于先验的紧密程度。

对先验紧密度 $T \in \{500, 2000, 5000\}$，模拟最小二乘系统（$\lambda = 1$），并报告多个随机种子下的平均通货膨胀率。

解释为什么更松的先验（更小的 $T$）会使即便是最小二乘系统也容易发生逃逸。

```{exercise-end}
```

```{solution-start} pl_ex2
:class: dropdown
```

```{code-cell} ipython3
for T in [500, 2000, 5000]:
    means = [simulate(model, λ=1.0, T_prior=T, n=1000, seed=s)['y'].mean()
             for s in range(6)]
    print(f"T = {T:>4}:  mean inflation across seeds = "
          f"{np.round(means, 1)}")
```

在更松的先验下，有效增益 $1/(T + t)$ 一开始就更大，因此早期的更新足够大，能够将信念从自我确认均衡上踢出去。

由于该均衡只是边际稳定的——均值动态的一个特征值恰好位于快速收敛区域的边界上——系统随后可能朝归纳假设漂移，并被困在接近拉姆齐的水平附近，模拟出常数增益下的逃逸现象。

更紧的先验则会使增益始终保持较小，因此最小二乘法能可靠地紧贴自我确认均衡。

```{solution-end}
```