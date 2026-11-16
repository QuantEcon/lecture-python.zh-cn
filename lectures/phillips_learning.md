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
    Mean dynamics near the self-confirming equilibrium: 自我确认均衡附近的均值动态
    'Least squares learning: a decreasing gain': 最小二乘学习：递减增益
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

```{index} single: Phillips Curve; Adaptive Learning and Escape Dynamics
```

```{contents} Contents
:depth: 2
```

## 概述

本讲座是 {cite:t}`Sargent1999` 所构建理论的集大成之作。

它遵循该书第 8 章——这是全书最具野心的一章。

本系列此后的所有内容都建立在此基础之上：

* {doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 进一步深化其分析；
* {doc}`phillips_policy_evaluation` 遵循该书第 9 章的思路，将这些适应性模型用于数据分析；
* {doc}`phillips_drifts_volatilities` 和 {doc}`phillips_drifts_volatilities_2` 对其核心论断进行实证检验；并且
* {doc}`phillips_lost_conquest` 将其工具应用于更晚近的一段历史事件。

在 {doc}`phillips_self_confirming` 中，政府对菲利普斯曲线持有*固定*的信念——这些信念被由它们本身所生成的数据所证实。

在这里，我们把政府变成一个实时经济计量学家。

每一期它都：

* 根据到目前为止观测到的数据，通过递归最小二乘法重新估计其菲利普斯曲线；并且
* 根据其*当前*的估计值，设定通货膨胀率为 {doc}`phelps 问题 <phillips_adaptive>` 第一期建议的水平。

我们要问的是：这样一个适应性的政府是否会收敛到自我确认均衡。

答案取决于一个单一的参数——支配旧数据被折扣速度的**增益**（gain）：

* 若采用实现最小二乘法的*递减*增益，均值动态会*渐近地*将经济拉向自我确认均衡，而在极限情形下我们不会得到什么新结果：系统被困在纳什结果附近。
* 若采用*常数*增益，主体会对过去数据打折扣，收敛被阻止，**新的结果便会涌现**：系统会反复地从自我确认均衡*逃逸*向拉姆齐（零通胀）结果，呈现出颇似沃尔克时代到来的自发稳定化现象。

"渐近地"这个词很重要。

在有限样本中，递减的增益也会减缓经济*返回*自我确认均衡的旅程，因此，一个先验不够紧的最小二乘政府可能很早就发生逃逸，然后在拉姆齐结果附近停留很长一段时间（{ref}`pl_decreasing`）。

这些逃逸正是 {doc}`phillips_two_stories` 中"econometric 政策评估的证明"故事的核心：一个学习索洛-托宾分布滞后版本自然率假说的适应性政府，会因偶然的观察结果而稳定住通货膨胀。

本讲座复现、分析并重新诠释了类似克里斯托弗·西姆斯（Christopher Sims）和郑喜泰（Heetaik Chung）的模拟结果 {cite}`Sims1988,Chung1990`。

他们的政府对他们认为按随机游走漂移的菲利普斯曲线系数运行**卡尔曼滤波**（参见 {doc}`kalman`）。

西姆斯的政府拟合的是一条*静态*菲利普斯曲线，而郑喜泰拟合的是一个分布滞后模型。

在他们的模拟中，政府从纳什结果逃逸一次后便停留在拉姆齐结果附近，看起来是无限期地停留着。

郑喜泰的滤波器采用了一个不会消失的增益，这使他得以根据战后美国的数据估计出政府的学习规则（{ref}`pbr_compliment`；{ref}`olg_ex7` 提出了他问题的一个简化版本）。

遵循 {cite:t}`Sargent1999` 的思路，我们转而赋予政府**常数增益递归最小二乘法**，这会产生*反复出现*的逃逸：均值动态不断将信念拉回自我确认均衡，而经济又不断再次逃逸。

{cite:t}`SargentWilliams2005` 将这一差异追溯到政府关于系数漂移的先验，{doc}`phillips_priors` 重现了他们的论述。

由于西姆斯的政府拟合的是一条静态曲线，下文所研究的归纳假设机制并不适用于他的模拟结果（{cite:t}`Sargent1999` 第 8 章，脚注 15）。

在这里，我们通过*模拟*来研究这些逃逸；{doc}`phillips_escaping_nash` 随后将其解析地刻画为第二个确定性 ODE，而 {doc}`phillips_priors` 则探讨政府关于系数漂移的先验如何重塑这两种力量。

让我们导入所需的库：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
```

(pl_primer)=
## 递归算法入门

本节是一个独立完整的入门介绍。

它构建了贯穿后文全部内容的两个分析对象——**均值动态**（mean dynamics）与**逃逸路径**（escape routes）——并解释了同一个递归公式既可以被视为计算自我确认均衡的*算法*，也可以被视为政府实时适应的*模型*的意义所在。

这部分内容较为技术性，只想了解结论的读者可以直接跳到下面的模拟部分，需要时再回来查阅。

从有限理性（Bounded Rationality）系列讲座过来的读者，已经在 {ref}`br_one_recursion` 中见过均值动态的标量版本，以及 $T$ 映射与 E-稳定性，而 {doc}`ls_learning` 证明了线性自我指涉模型的收敛定理。

本入门文档包含了适用于整个模型体系的一般大偏差理论；{doc}`phillips_escaping_nash` 将其应用于该模型的一个静态版本。

### 信念、矩条件与自我确认均衡

在经典识别方案下，自我确认均衡由政府关于某些总体矩及其所隐含的回归系数的信念所确定。

在经典识别方案下，这些信念由三元组 $(\gamma, \, E X_{C} X_{C}^\top, \, E U X_{C})$ 度量，其中 $\gamma$ 是菲利普斯曲线系数向量。

在本讲座的适应性模型中，这些对象的*时间 $t$ 的取值*是经济的状态变量之一；它们只有在自我确认均衡中才不再作为状态变量存在，因为在那里它们是常数。

经典识别方案下的自我确认均衡满足以下矩条件：

```{math}
:label: pl_scemoments

\begin{aligned}
\mathbb{E}\, R_{XC}^{-1}(\gamma)\left[ U_t X_{Ct}^\top - \left(X_{Ct} X_{Ct}^\top \right)\gamma \right] &= 0, \\
\mathbb{E}\, X_{Ct} X_{Ct}^\top - R_{XC}(\gamma) &= 0,
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

\mathbb{E}\left[F(\phi, \zeta)\right] = 0,
\qquad
b(\phi) \equiv \mathbb{E}\left[F(\phi, \zeta)\right],
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

每一步都需要计算数学期望 $b(\phi) = \mathbb{E}[F(\phi, \zeta)]$——这正是我们在那里需要矩（李雅普诺夫）公式的原因。

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
递归随机逼近起源于 {cite:t}`RobbinsMonro1951`，他设计了 {eq}`pl_sa` 用于在噪声观测下寻找回归函数的根；以及 {cite:t}`KieferWolfowitz1952`，他将其改造用于寻找回归函数的最大值（下文提到的“K-W”算法）。

用于分析此类递归的“ODE 方法”——即用微分方程的解来逼近插值过程——归功于 {cite:t}`Ljung1977`；书籍级的详尽处理见 {cite:t}`BenvenisteMetivierPriouret1990` 和 {cite:t}`KushnerYin2003`。

将其用于研究自我指涉宏观经济模型中的学习问题，由 {cite:t}`MarcetSargent1989jet` 开创，并被 {cite:t}`EvansHonkapohja2001` 全面发展。
```

### 均值动态

{cite:t}`KushnerClark1978` 和 {cite:t}`Ljung1977` 的经典随机逼近算法将增益设定为按 $a_n \sim 1/n$ 的速度递减（至少对某个 $N > 0$ 满足 $t \geq N$）。

这使得我们能够对 {eq}`pl_sa` 几乎必然收敛到 $b(\phi)$ 的零点这一结论做出强有力的论断。

当 $a_n \sim 1/n$ 时，随着 $n \to \infty$，插值过程 $\phi^o(t)$ 逼近以下常微分方程的解：

```{math}
:label: pl_ode

\frac{d \phi^o(t)}{dt} = b\left(\phi^o(t)\right),
```

我们称之为**均值动态**——这是 {ref}`pc_mean_dynamics` 中标量最小二乘学习 ODE 的向量化推广。

大数定律使连续时间逼近中的随机项以足够快的速度消失，从而使均值动态 {eq}`pl_ode` 刻画了随机过程 {eq}`pl_sa` 的*尾部*行为。

因此：

* 如果算法收敛（几乎必然地），它就收敛到均值动态的零点 $b(\phi) = 0$——即一个自我确认均衡；并且
* ODE {eq}`pl_ode` 携带着关于算法局部与全局稳定性的信息。

局部稳定性由 $b$ 在静止点处雅可比矩阵的特征值支配：如果所有特征值的实部都为负，该静止点就是局部稳定的。

这一条件被称为 **E-稳定性**（{ref}`br_one_recursion`）。

对最小二乘法而言，雅可比矩阵具有一个简单的结构，这由 {doc}`ls_learning` 中雅可比约简命题（{prf:ref}`prop-ms-jacobian-eigenvalues`）所确立：

* 属于二阶矩矩阵的那一部分贡献了等于 $-1$ 的重复特征值；而
* 其余特征值则是 $DT - I$ 的特征值，其中映射 $T$ 将一个信念送到该信念所诱导出的总体回归系数。

在递减增益下，实部最大的特征值也支配着收敛的*速率*。

通常的 $\sqrt{t}$ 速率（伴随渐近正态的估计量）所需的一个必要条件是，每个特征值的实部都低于 $-\tfrac12$。

{ref}`pl_eigen` 计算了我们模型中的这些特征值；其中一个特征值重复了五次，恰好位于 $-\tfrac12$ 的边界上。

这一边界关系到递减增益估计量的渐近分布。

它并不支配常数增益政府逃逸的频率；下文引入的大偏差对象才支配这一点。

```{note}
{cite:t}`BrockHommes1997` 构建的模型，其全局行为由远离理性预期均衡的稳定均值动态，以及在均衡附近适应的局部不稳定性共同驱动——这是一种从学习中产生内生波动的互补机制。
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

杜皮伊（Dupuis）与库什纳（Kushner），见 {cite:t}`KushnerYin2003`，以及其他学者验证了在何种条件下，当 $\epsilon \to 0$ 且 $\epsilon n \to \infty$ 时，过程 $\phi_n^\epsilon$ *依分布*收敛到同一均值动态 {eq}`pl_ode` 的零点；均值动态所需满足的限制条件与经典 $a_n \sim 1/n$ 理论中的要求相同。

与递减增益算法不同，常数增益算法不会稳定下来：$(\gamma, R_{XC})$ 会收敛到一个*平稳随机过程*，围绕自我确认均衡永久地波动——并偶尔远离它。

### 逃逸路径与大偏差理论

对本讲座而言，常数增益机制中最重要的特征不是向 $\phi_f$ 的收敛，而是*远离它的偏离*。

我们对远离自我确认均衡的运动与趋向它的运动同样感兴趣，因为模拟中反复出现的稳定化现象恰恰就是这样的偏离。

**大偏差理论**通过三个对象来刻画这些偏离。

首先是（创新过程 $F(\phi_n, \zeta_n)$ 某种平均化版本的）对数矩生成函数：对于与 $F$ 维数一致的向量 $\alpha$，

```{math}
:label: pl_mgf

H(\alpha, \phi) = \log \mathbb{E} \exp\left(\alpha^\top F(\phi, \zeta)\right),
```

其中期望是关于 $\zeta$ 的分布计算的。

```{note}
方程 {eq}`pl_mgf` 是一个启发性的简写。

实际进入理论的对象是一个*时间平均*极限；{cite:t}`DupuisKushner1987` 和 {cite:t}`KushnerYin2003` 假设对每个区块宽度 $\Delta > 0$，以下极限在任意紧集上关于 $\phi_i, \alpha_i$ 一致存在：
$$
\sum_{i=0}^{T/\Delta - 1} \Delta\, H(\alpha_i, \phi_i)
= \lim_{N \to \infty} \frac{\Delta}{N}
  \log \mathbb{E} \exp \sum_{i=0}^{T/\Delta - 1} \alpha_i^\top
  \sum_{j=iN}^{iN+N-1} F(\phi_i, \zeta_j) .
$$
内部的求和对长度为 $N$ 的一个区块内的创新项做平均；双重极限使我们能够处理序列相关的创新项。
```

其次是 $H$ 的**勒让德变换**（Legendre transform），它扮演速率函数的角色：

```{math}
:label: pl_legendre

L(\xi, \phi) = \sup_\alpha \left[ \alpha^\top \xi - H(\alpha, \phi) \right] .
```

第三是**行动泛函**（action functional），它衡量候选逃逸路径 $\phi(\cdot)$ 的“代价”：

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

假设最小化路径 $\tilde\phi(\cdot)$ 是唯一的，并设 $t_D^\epsilon$ 为常数增益过程 $\phi^\epsilon(t)$ 离开 $D$ 的首个时刻，{cite:t}`DupuisKushner1987` 证明了对每个 $\eta > 0$ 都有：

```{math}
:label: pl_escapelim

\lim_{\epsilon \to 0} \mathbb{P}\left(
\left| \phi^\epsilon(t_D^\epsilon) - \tilde\phi(T) \right| > \eta
\right) = 0 .
```

换句话说：*在系统逃离集合 $D$ 的条件下，它离开该集合的位置会接近最小行动路径的终点*——因此逃逸具有确定的方向和形态，尽管它是由偶然性触发的。

正是在这个意义上，下文中的逃逸虽然不需要任何巨大的冲击触发，却“看起来有目的性”：它们遵循 {eq}`pl_escapeproblem` 所规定的最小行动路径。

同样的理论也给出了逃逸的*平均时间*。

设 $\bar S$ 为 {eq}`pl_escapeproblem` 的最小化值；那么当 $\epsilon \to 0$ 时，$\epsilon \log \mathbb{E}\, t_D^\epsilon \to \bar S$。

因此，随着增益的缩小，逃逸会变得指数级地罕见。

这是一个小增益意义下的结论。

当 $\bar S$ 相对于 $\epsilon$ 较小时——正如本模型体系所使用的增益情形那样——逃逸会*频繁*发生：平均逃逸时间大致按 $1/\epsilon$ 增长，而不是按 $\exp(\bar S / \epsilon)$ 增长（{ref}`en_simulation` 对此进行了度量），而在下面的模拟中，政府大部分时间都停留在拉姆齐结果附近。

一个关键的对比：均值动态 {eq}`pl_ode` *不*依赖于其周围的噪声，而逃逸路径*却*依赖于噪声——噪声不仅在 {eq}`pl_ode` 周围添加随机波动，它还开辟出了这第二类路径。

```{note}
其数学基础是 {cite:t}`FreidlinWentzell1998`（特别是其第 4 章）关于随机扰动动态系统的大偏差理论，被 {cite:t}`DupuisKushner1987` 和 {cite:t}`DupuisKushner1989` 专门用于随机逼近；一种弱收敛处理见 {cite:t}`DupuisEllis1997`。
```

### 一个可处理的行动泛函

逃逸路径的计算承诺提供关于算法中心趋势的低成本信息，但行动泛函 {eq}`pl_action` 通常难以计算。

一个重要的特例极大地简化了它。

假设创新项是加性的且服从高斯分布：

$$
F(\phi, \zeta) = b(\phi) + \sigma(\phi)\, \zeta,
$$

其中 $\zeta_n$ 是平稳的且服从高斯分布，但不一定是序列不相关的，并定义 $\Omega = \sum_j \mathbb{E}\, \zeta_t \zeta_{t-j}^\top$。

那么行动泛函取如下二次形式：

```{math}
:label: pl_action2

S(T, \phi) = \frac{1}{2} \int_0^T
\left(\tfrac{d}{ds}\phi - b(\phi)\right)^\top
\left[\sigma(\phi)\, \Omega\, \sigma(\phi)^\top \right]^{+}
\left(\tfrac{d}{ds}\phi - b(\phi)\right)
w(s)\, ds ,
```

其中 $(\cdot)^{+}$ 是摩尔-彭若斯广义逆（用于处理 $\sigma \Omega \sigma^\top$ 可能出现的随机奇异性）。

权重 $w(s)$ 依赖于增益：当 $a_n = a_0 / n^p$ 中的 $p = 1$ 时，$w(s) = \exp(s)$；当 $p < 1$ 时，$w(s) = 1$。

将 {eq}`pl_action2` 理解为一种代价，它惩罚*实际漂移* $\tfrac{d}{ds}\phi$ 与*均值漂移* $b(\phi)$ 之间的偏离，并按局部噪声协方差 $\sigma \Omega \sigma^\top$ 的逆对每个方向加权。

因此最小行动逃逸路径会穿过均值动态较弱而噪声较大的区域——而在我们的模型中，这恰恰是*归纳假设*所指的方向。

我们的模型并不精确满足 {eq}`pl_action2` 背后的假设。

它的创新项 $F$ 含有诸如 $X_{C,t} X_{C,t}^\top$ 和 $X_{C,t} U_t$ 这样的乘积项，因此它在冲击上是二次的，而非加性且高斯的。

```{note}
{cite:t}`ChoWilliamsSargent2002`——本讲座模型的正式发表版本——将逃逸问题约简为最小化一个形如 {eq}`pl_action2` 的二次代价，其加权矩阵是由李雅普诺夫方程计算得到的四阶矩矩阵。

这一约简借鉴了 Williams 早期研究的一个版本，{cite}`Williams2001`。

{cite:t}`Williams2019`（正式发表版本）指出，该早期版本误用了一个大偏差定理，这一差异只对像本讲座这样的动态模型才有实质影响，因此 CWS 针对动态模型的计算所依赖的正是这一捷径。

他们对动态模型的纳什自我确认均衡*数值*求解了控制问题 {eq}`pl_escapeproblem`，并发现（其图 3）沿着主导逃逸路径，通货膨胀权重之和从 $-1$ 上升到接近零——这正是激活归纳假设的值——也就是说，逃逸路径朝向拉姆齐结果。

{cite:t}`SargentWilliams2005` 用一个漂移协方差 $V$ 固定的卡尔曼滤波器取代递归最小二乘法，并研究了 $V$ 如何重塑逃逸路径（{doc}`phillips_priors`）。

本讲座的常数增益算法是一个漂移协方差与当前 $P_{t-1}$ 成比例的卡尔曼滤波器（见下文）；固定的 $V$ 只有在渐近意义下才能复现它。

{cite:t}`Kasa2004` 将同样的大偏差机制应用于反复出现的货币危机。
```

### 从计算到适应

前述递归公式是作为逼近自我确认均衡的*算法*被引入的。

同样的数学也告诉我们，当我们将自我确认均衡模型*改造*为纳入实时适应时会发生什么——只需将 $\phi_n$ 理解为政府在时间 $n$ 的信念，而非某个求解器的第 $n$ 次迭代结果。

以下两个事实统领了后文的一切：

1. 实现最小二乘法的增益序列（按 $1/t$ 递减）至少在渐近意义下使均值动态将经济拉*向*自我确认均衡；而
2. 递减速度更慢的增益序列——极限情形下即为对过去打折扣的常数增益——则*阻止*了这种拉力，并增加了逃逸动态影响结果的频率。

```{note}
一段简短的思想史。

{cite:t}`Lucas_Prescott_1971` 曾摒弃对矩条件 {eq}`pl_scezero` 进行迭代这一计算策略，但 {cite:t}`Townsend1983` 却使用了它。

{cite:t}`DeCanio1979` 以及 {cite:t}`Evans1985,Evans1989` 利用从感知规律到实际运动规律的映射，定义了预期稳定性。

{cite:t}`Woodford1990` 和 {cite:t}`MarcetSargent1989jet,MarcetSargent1989` 用均值动态 {eq}`pl_ode` 建立了含有自我指涉的模型中最小二乘学习收敛于理性预期的条件，两者都要求 $b(\phi)$ 的连续性。

曹寅坤（In-Koo Cho）研究了带有*不连续* $b(\phi)$ 的问题，这种不连续性源自可信度和搜索问题中不连续的决策规则（触发策略）；为了使最小二乘学习逼近理性预期，他使用了满足 $\tfrac{1}{\sqrt n} < a_n < \tfrac{1}{\log n}$ 的增益，这为 {eq}`pl_sa` 产生了一个*扩散*逼近，促进了足够的试探以发现均衡（{cite:t}`Sargent1999` 第 8 章脚注 8 将这两个界限的顺序印反了，那样写出的是一个空集，因为 $\log n < \sqrt n$）。

{cite:t}`KandoriMailathRob1993` 运用相关数学方法，通过突变在博弈中选择长期均衡，而罗杰·迈尔森（Roger Myerson）将逃逸路径的计算应用于一个投票问题。

这些学习方法的现代综合体现在 {cite:t}`EvansHonkapohja2001` 中。
```

## 适应性模型

现在我们来构建经典的适应性模型。

### 政府的信念与行为

政府相信存在一个分布滞后的菲利普斯曲线：

```{math}
:label: pl_belief

U_t = \gamma^\top X_{C,t} + \varepsilon_{C,t},
\qquad
X_{C,t} = \begin{bmatrix} y_t & U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}^\top.
```

在时间 $t$ 到达时，凭借估计值 $\gamma_{t-1}$，政府通过求解菲尔普斯问题来设定通货膨胀的系统性部分，*就好像* $\gamma_{t-1}$ 将永远支配菲利普斯曲线一样：

```{math}
:label: pl_rule

y_t = h(\gamma_{t-1}) X_{t-1} + v_{2t},
\qquad
X_{t-1} = \begin{bmatrix} U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}^\top.
```

随后它通过**递归最小二乘法**（RLS）更新其信念：

```{math}
:label: pl_rls

\begin{aligned}
\gamma_t &= \gamma_{t-1} + g_t R_{XC,t}^{-1} X_{C,t}\left(U_t - \gamma_{t-1}^\top X_{C,t}\right), \\
R_{XC,t} &= R_{XC,t-1} + g_t\left(X_{C,t} X_{C,t}^\top - R_{XC,t-1}\right),
\end{aligned}
```

其中 $\{g_t\}$ 是增益序列。

最小二乘法设定 $g_t = 1/t$；常数增益算法设定 $g_t = g_0 > 0$，并对过去的观测打折扣，如果政府怀疑菲利普斯曲线会随时间漂移，这样做是合理的。

从 {ref}`olg_phillips` 过来的读者应当注意记号上的变化：在那里以及整个有限理性讲座系列中（{ref}`br_notation`），$\gamma$ 表示增益，$\alpha$ 表示信念；而在这里，遵循原书的记法，$\gamma$ 表示信念，增益则用 $g_t$（或下文的 $1 - \lambda$）表示（另见 {ref}`phillips_notation`）。

假设公众知道政府的规则，因此其通货膨胀预测为 $x_t = h(\gamma_{t-1}) X_{t-1}$，即 {eq}`pl_rule` 中的系统性部分。

失业率是由 {doc}`phillips_self_confirming` 中实际的菲利普斯曲线在 $\rho_1 = \rho_2 = 0$ 时生成的：

$$
U_t = U^* - \theta(y_t - x_t) + v_{1t} = U^* - \theta v_{2t} + v_{1t} .
$$

### 带滞后项的菲尔普斯问题

给定一个信念 $\gamma$，决策规则 $h(\gamma)$ 求解一个 LQ 控制问题。

将所信奉的菲利普斯曲线写为 $U_t = \gamma_0 y_t + c^\top s_t$，其中 $\gamma_0$ 是当期通货膨胀的系数，$c$ 收集了状态 $s_t = X_{t-1}$ 上的系数。

政府最小化 $\mathbb{E}\sum_t \delta^t (U_t^2 + y_t^2)$，因此每期损失为 $s_t^\top (cc^\top) s_t + (\gamma_0^2 + 1) y_t^2 + 2\gamma_0\, y_t\, c^\top s_t$，状态按 $s_{t+1} = A s_t + B y_t$ 演化，其中：

$$
s_{t+1} = \begin{bmatrix} U_t \\ U_{t-1} \\ y_t \\ y_{t-1} \\ 1 \end{bmatrix},
\qquad
U_t = c^\top s_t + \gamma_0 y_t .
$$

这正是 {doc}`phillips_adaptive` 中的菲尔普斯问题，此处换成了分布滞后的菲利普斯曲线。

我们使用 `scipy` 的离散代数李卡提方程求解器来求解这个折扣 LQ 问题。

为了将折扣因子纳入一个无折扣问题，我们用 $\sqrt{\delta}$ 对 $A$ 和 $B$ 进行缩放。

损失中的交叉项 $2\gamma_0\, y_t\, c^\top s_t$（代码中的 `N`）和另外两项一样，是每期成本，因此不对其进行缩放。

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
        Ad, Bd = sb * A, sb * B             # discounting folded into A, B
        P = solve_discrete_are(Ad, Bd, R, Q, s=N.T)   # cross term unscaled
        F = np.linalg.solve(Q + Bd.T @ P @ Bd, Bd.T @ P @ Ad + N)
        return -F.ravel()
```

我们使用 {cite:t}`Sargent1999` 附录 A 中递归最小二乘法的卡尔曼滤波实现。

政府将 $\gamma$ 视为一个遵循随机游走的隐藏状态，$\tilde\gamma_t = \tilde\gamma_{t-1} + w_t$，并通过回归 $U_t = \tilde\gamma_t^\top X_{C,t} + \varepsilon_{C,t}$ 来观测它；随后 {doc}`kalman` 中的卡尔曼滤波器会更新其估计值 $\gamma_t$ 以及该估计值的协方差矩阵 $P_t$。

遗忘因子 $\lambda \in (0, 1]$ 映射到增益：

* $\lambda = 1$ 将漂移协方差 $\mathbb{E} w_t w_t^\top$ 设为零，给出最小二乘法（$g_t \to 1/t$）；
* $\lambda < 1$ 将 $\mathbb{E} w_t w_t^\top$ 设为 $(1/\lambda - 1) P_{t-1}$，即与政府当前不确定性成比例的漂移协方差，给出常数增益 $g_0 = 1 - \lambda$。

先验被初始化，就好像政府已经观测到 $T_0$ 期的自我确认均衡数据一样，通过 $P_0 = (\sigma_C^2 / T_0)\, M^{-1}$；$T_0$（代码中的 `T_prior`）越大意味着先验越紧。

该函数还会返回最终的信念和协方差矩阵，下文将会用到它们。

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

    return dict(y=y_path, U=U_path, sumweights=sumweights, constant=constant,
                γ=γ, P=P)
```

让我们来检验一下在自我确认信念处的决策规则。

```{code-cell} ipython3
model = AdaptivePhillips()
h_sce = model.phelps_h(model.γ_sce)
print("decision rule at the self-confirming belief:")
print(f"  h(γ_sce) = {np.round(h_sce, 3) + 0.0}  (a constant rule of "
      f"{h_sce[-1]:.1f} = Nash inflation)")
```

在自我确认信念处，菲尔普斯规则是一个等于纳什通货膨胀率的常数，信念自我复制——适应性系统的静止点正是 {doc}`phillips_self_confirming` 中的自我确认均衡。

我们遵循 {cite:t}`Sargent1999`，将真实的数据生成参数设定为 {doc}`phillips_self_confirming` 末尾所给出的经典例子：$U^* = 5$，$\theta = 1$，$\sigma_1 = \sigma_2 = 0.3$，$\rho_1 = \rho_2 = 0$，$\delta = 0.98$。

经典的自我确认均衡具有序列不相关的 $(U, y)$，围绕均值 $(5, 5)$ 波动。

(pl_eigen)=
## 自我确认均衡附近的均值动态

在进行模拟之前，我们先计算该模型的均值动态 {eq}`pl_ode`，以及它们在自我确认均衡附近的行为。

将政府的规则 {eq}`pl_rule` 和公众的预测代入真实的菲利普斯曲线，得到

$$
U_t = -\theta y_t + \theta\, h(\gamma) X_{t-1} + U^* + v_{1t} .
$$

冲击 $v_{1t}$ 与 $X_{C,t}$ 中的每个回归变量都是独立的。

因此，当政府使用规则 $h(\gamma)$ 时，$U_t$ 对 $X_{C,t}$ 的总体回归恰好为

```{math}
:label: pl_tmap

T(\gamma) = \begin{bmatrix} -\theta & \theta h_1(\gamma) & \theta h_2(\gamma) & \theta h_3(\gamma) & \theta h_4(\gamma) & U^* + \theta h_5(\gamma) \end{bmatrix}^\top ,
```

其中 $h_j(\gamma)$ 是 $h(\gamma)$ 的第 $j$ 个分量。

这是我们模型中 {ref}`br_one_recursion` 的 $T$ 映射：它将政府所感知的菲利普斯曲线映射为其政策实际产生的真实曲线。

将其应用于递归最小二乘算法 {eq}`pl_rls`，随机逼近法得到了均值动态（该书的公式 (89)）

```{math}
:label: pl_ode_model

\frac{d\gamma}{d\tau} = R_{XC}^{-1} M(\gamma)\left[T(\gamma) - \gamma\right],
\qquad
\frac{d R_{XC}}{d\tau} = M(\gamma) - R_{XC},
```

其中 $\tau$ 是人工时间，$M(\gamma) = \mathbb{E} X_{C,t} X_{C,t}^\top$ 是在政府的信念被冻结在 $\gamma$ 时，根据系统的平稳分布计算出来的。

一个静止点满足 $T(\gamma) = \gamma$ 以及 $R_{XC} = M(\gamma)$：这就是一个自我确认均衡。

在静止点处，{eq}`pl_ode_model` 的雅可比矩阵是块三角的。

因此其特征值是 $DT - I$ 的特征值，加上来自 $R_{XC}$ 块的重复特征值 $-1$（参见 {doc}`ls_learning` 中的 {prf:ref}`prop-ms-jacobian-eigenvalues`）。

我们通过有限差分法计算自我确认信念处的 $DT$。

```{code-cell} ipython3
def T_map(model, γ):
    "Population regression of U_t on X_C,t when the government uses h(γ)."
    θ, h = model.θ, model.phelps_h(γ)
    return np.concatenate([[-θ], θ * h[:4], [model.U_star + θ * h[4]]])


def jacobian_T(model, ε=1e-5):
    "Finite-difference Jacobian of T at the self-confirming belief."
    J = np.empty((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = ε
        J[:, j] = (T_map(model, model.γ_sce + d)
                   - T_map(model, model.γ_sce - d)) / (2 * ε)
    return J


DT_I = jacobian_T(model) - np.eye(6)
print("DT - I at the self-confirming belief:")
print(np.round(DT_I, 3) + 0.0)
print("eigenvalues:", np.round(np.sort(np.linalg.eigvals(DT_I).real), 4))
print(f"-1/(1 + θ²) = {-1 / (1 + model.θ**2)}")
```

矩阵 $DT - I$ 是下三角矩阵，所以其特征值就是其对角线元素。

* 第一个特征值 $-1$ 属于当期通货膨胀的系数：无论政府相信什么，$T$ 都会赋予该系数 $-\theta$。
* 其余五个特征值均等于 $-1/(1+\theta^2) = -\tfrac12$。

第二组特征值有一个简单的来源。

在自我确认均衡处，滞后项不携带任何权重，并且在一阶近似下，菲尔普斯规则对所信念的滞后变量上的小系数 $c_j$ 的响应，正如静态一阶条件 $y_t = \theta\, c^\top X_{t-1} / (1+\theta^2)$ 所规定的那样。

这使得 $DT$ 的对角线元素等于 $\theta^2/(1+\theta^2)$。

非对角线元素全部位于最后一行：对滞后项所信念的系数，通过菲尔普斯问题的动态部分，也会改变规则中的常数项。

重复特征值 $-1/(1+\theta^2)$ 与 {ref}`pc_mean_dynamics` 中标量均值动态的斜率 $\mathcal{M}$ 相同（在那里公众只需学习一个数），也与 {ref}`sc_estability` 中经典的 E-稳定性特征值相同。

{cite:t}`Sargent1999` 报告称存在一个"重数为 6"的重复特征值 $-.5$，以及 $R_{XC}$ 的重复特征值 $-1$。

该计算表明，重数实际上为五；而信念的第六个特征值，即当期通货膨胀系数所对应的特征值，是 $-1$。

所有特征值都是负数，因此自我确认均衡是 E-稳定的。

当 $\theta = 1$ 时，重复特征值恰好位于 $-\tfrac12$ 边界上，因此最小二乘法的收敛速度比通常的 $\sqrt{t}$ 速率更慢，正如 {ref}`pc_mean_dynamics` 中标量情形一样。

正如入门读物中所指出的，该边界涉及最小二乘估计的渐近分布；而恒定增益政府逃逸的频率，则由最小化作用量 $\bar S$ 决定。

{ref}`pl_ex3` 要求探讨特征值如何随 $\theta$ 变化。

(pl_decreasing)=
## 最小二乘学习：递减增益

首先是最小二乘法（递减增益），先验分布相当于 $T_0 = 5000$ 期数据，置信度较高。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "具有紧先验的经典适应性模型在最小二乘法下：通货膨胀紧贴自我确认值"
    name: fig-learn-ls
---
ls = simulate(model, λ=1.0, T_prior=5000, n=1000, seed=1)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(ls['y'], lw=0.8)
ax.axhline(5, color='k', ls='--', lw=1, label='self-confirming (Nash)')
ax.axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.legend()
plt.show()
```

在 {numref}`fig-learn-ls` 中，通货膨胀紧贴自我确认值 5，模拟结果看起来就像是直接取自自我确认均衡本身。

这正是本书描述的结果：最小二乘法没有带来任何新东西，政府仍困在纳什均衡附近。

但这个结果取决于先验分布。

本书的图 8.1 使用了更宽松的先验，$T_0 = 800$，并展示了 400 期内通货膨胀保持在 5 附近的情况。

让我们比较每种先验下 1000 期的八条路径。

为了看清噪声背后的规律，我们绘制通货膨胀的 20 期移动平均值，并将该平均值首次跌破 1 的日期称为政府的*首次逃逸*。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "紧先验和宽松先验下的最小二乘法：宽松先验下的逃逸会持续存在"
    name: fig-learn-ls-prior
---
def moving_average(y, k=20):
    return np.convolve(y, np.ones(k) / k, mode='valid')


def first_escape(y, k=20, level=1.0):
    "First date at which the k-period moving average of y falls below level."
    below = moving_average(y, k) < level
    return int(np.argmax(below)) if below.any() else None


def returns_to_nash(y, k=20, level=4.0):
    "Whether the moving average climbs back above level after the first escape."
    t = first_escape(y, k)
    return t is not None and bool((moving_average(y, k)[t:] > level).any())


ls_runs = {T0: [simulate(model, λ=1.0, T_prior=T0, n=1000, seed=s)
                for s in range(8)]
           for T0 in (5000, 800)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (T0, runs) in zip(axes, ls_runs.items()):
    for sim in runs:
        ax.plot(np.arange(19, 1000), moving_average(sim['y']), lw=0.8)
    ax.axhline(5, color='k', ls='--', lw=1)
    ax.axhline(0, color='C2', ls=':', lw=1)
    ax.set_title(rf'$T_0 = {T0}$')
    ax.set_xlabel('$t$')
axes[0].set_ylabel('20-period moving average of $y_t$')
plt.tight_layout()
plt.show()

for T0, runs in ls_runs.items():
    escapes = [first_escape(sim['y']) for sim in runs]
    print(f"T_0 = {T0}: {sum(e is not None for e in escapes)} of {len(runs)} "
          f"paths escape; {sum(returns_to_nash(sim['y']) for sim in runs)} "
          f"of them return above 4")
```

在更宽松的先验下，{numref}`fig-learn-ls-prior` 显示最小二乘法逃逸向拉姆齐均衡，并且每一条逃逸的路径都停留在那里。

如果最小二乘法收敛于自我确认均衡，为什么会这样呢？

因为收敛定理讨论的是极限情形。

对于相当于 $T_0$ 期数据的先验，最小二乘增益为 $1/(T_0 + t)$，因此经过 $t$ 期后，人工时间 {eq}`pl_artificial` 大约推进了 $\tau = \log\left((T_0 + t)/T_0\right)$。

时钟越走越慢：$\tau$ 每增加一个单位，所耗费的期数大约是前一个单位的 $e \approx 2.7$ 倍。

相比之下，恒定增益 $g_0$ 每期都会使 $\tau$ 推进 $g_0$。

这解释了两个面板的现象。

紧的先验从一开始就使时钟变慢：当 $T_0 = 5000$ 时，1000 期仅提供了 $\log(6000/5000) \approx 0.18$ 单位的人工时间，而当 $T_0 = 800$ 时则为 $\log(1800/800) \approx 0.81$ 单位。

而且一旦政府逃逸之后，减速的时钟也会延迟其返回。

为了了解返回需要多少人工时间，我们从某条逃逸路径在 $t = 1000$ 时所持有的信念 $(\gamma_t, R_{XC,t})$ 出发，对均值动态 {eq}`pl_ode_model` 进行积分。

我们利用离散李雅普诺夫方程，从冻结规则 $h(\gamma)$ 下 $(y_t, U_t)$ 的平稳分布中得到 $M(\gamma)$。

在最小二乘法下，$R_{XC,t} = \sigma_C^2 P_t^{-1}/(T_0 + t)$。

```{code-cell} ipython3
def stationary_moments(model, γ):
    """
    Mean inflation and M(γ) = E[X_C X_C'] in the stationary distribution
    of the true system when the government's beliefs are frozen at γ.
    """
    θ, U_star, σ1, σ2 = model.θ, model.U_star, model.σ1, model.σ2
    h = model.phelps_h(γ)
    # z_t = (y_t, y_{t-1}, y_{t-2}, U_t, U_{t-1}, U_{t-2}) obeys
    # z_t = c + A z_{t-1} + C (v_1t/σ1, v_2t/σ2)
    A = np.zeros((6, 6))
    A[0, [0, 1, 3, 4]] = h[2], h[3], h[0], h[1]   # the rule h(γ)
    A[1, 0] = A[2, 1] = A[4, 3] = A[5, 4] = 1.0
    c = np.array([h[4], 0, 0, U_star, 0, 0])
    C = np.zeros((6, 2))
    C[0] = 0, σ2                                  # y_t = h X_{t-1} + v_2t
    C[3] = σ1, -θ * σ2                            # U_t = U* - θ v_2t + v_1t
    μ = np.linalg.solve(np.eye(6) - A, c)
    Σ = solve_discrete_lyapunov(A, C @ C.T)
    # X_C = (y_t, U_{t-1}, U_{t-2}, y_{t-1}, y_{t-2}, 1)
    idx = [0, 4, 5, 1, 2]
    M = np.ones((6, 6))
    M[:5, :5] = Σ[np.ix_(idx, idx)] + np.outer(μ[idx], μ[idx])
    M[:5, 5] = M[5, :5] = μ[idx]
    return μ[0], M


def mean_dynamics(τ, x, model):
    "Right side of the mean dynamics for the stacked vector (γ, col R_XC)."
    γ, R = x[:6], x[6:].reshape(6, 6)
    _, M = stationary_moments(model, γ)
    dγ = np.linalg.solve(R, M @ (T_map(model, γ) - γ))
    return np.concatenate([dγ, (M - R).ravel()])


def halfway(τ, x, model):
    "Event: mean inflation climbs back to half the Nash rate."
    return stationary_moments(model, x[:6])[0] - 0.5 * model.θ * model.U_star


halfway.direction = 1

T0, n = 800, 1000
sim = next(s for s in ls_runs[T0] if first_escape(s['y']) is not None)
R_n = model.σC2 * np.linalg.inv(sim['P']) / (T0 + n)
x0 = np.concatenate([sim['γ'], R_n.ravel()])

τ_grid = [0, 1, 2, 5, 10, 20, 30]
sol = solve_ivp(mean_dynamics, (0, τ_grid[-1]), x0, t_eval=τ_grid,
                events=halfway, args=(model,), rtol=1e-6, atol=1e-8)

print(f"{'τ':>4} {'sum of weights':>15} {'mean inflation':>15} "
      f"{'periods, least squares':>23} {'periods, gain 0.025':>20}")
for τ, x in zip(sol.t, sol.y.T):
    γ = x[:6]
    print(f"{τ:4.0f} {γ[0] + γ[3] + γ[4]:15.2f} "
          f"{stationary_moments(model, γ)[0]:15.2f} "
          f"{(T0 + n) * (np.exp(τ) - 1):23.1e} {τ / 0.025:20,.0f}")

τ_half = sol.t_events[0][0]
print(f"\nmean inflation is back to {0.5 * model.θ * model.U_star} at τ = "
      f"{τ_half:.1f}: after {(T0 + n) * (np.exp(τ_half) - 1):.1e} more periods "
      f"of least squares, or {τ_half / 0.025:.0f} periods at a gain of 0.025")
```

最后两列统计的是 $t = 1000$ 之后的期数。

均值动态确实将信念带回自我确认均衡附近：通货膨胀权重之和回落向 $-1$，均值通货膨胀也回升向 5。

因此最小二乘法确实收敛，但只是渐近地收敛。

在归纳假设附近，均值动态较弱，返回需要许多单位的人工时间。

恒定增益 0.025 能在数百期内就提供这些单位。

而最小二乘法要提供这些单位，需要的时间跨度远远超过任何历史样本，因为递减的增益既会减缓逃逸，也会减缓返回。

因此在有限样本中，递减增益可能会把一次早期逃逸变成一次持续存在的逃逸。

这类路径——一次逃逸之后长期停留在拉姆齐均衡附近——与西姆斯和钟用卡尔曼滤波器发现的结果相似，尽管这里的机制与他们的不同；{doc}`phillips_priors`（{ref}`ppr_sims`）通过他们滤波器中的先验分布解释了他们的结果。

{ref}`pl_ex2` 衡量了最小二乘法在三种先验下逃逸的频率。

## 常数增益与逃逸动态

现在赋予政府一个*常数*增益 $g_0 = 1 - \lambda = 0.025$，使其对过去数据打折扣。

我们使用该书图 8.2 的设定，$\lambda = 0.975$ 且 $T_0 = 300$；本系列中的其他讲座则各自采用不同的增益，均取自各自的来源。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Classical adaptive model under a constant gain: recurrent escapes toward Ramsey"
    name: fig-learn-cgain
---
cg = simulate(model, λ=0.975, T_prior=300, n=1000, seed=1)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(cg['y'], lw=0.8)
ax.axhline(5, color='k', ls='--', lw=1, label='self-confirming (Nash)')
ax.axhline(0, color='C2', ls=':', lw=1, label='Ramsey')
ax.set_xlabel('$t$')
ax.set_ylabel('inflation $y_t$')
ax.legend()
plt.show()
```

在 {numref}`fig-learn-cgain` 中，通货膨胀最初接近自我确认值 5，在几十个时期内便几乎跌落至零。

在接下来的数百个时期中，它又缓慢回升，但只走了朝向 5 的部分路程，随后第二次稳定化又将其推回零附近。

将系统拉向自我确认均衡的均值动态，遭到了一种反复出现的力量的对抗，这种力量将通货膨胀推向接近拉姆齐结果的水平。

这种回升正是在 {ref}`pl_decreasing` 中计算得到的那种缓慢回归，只不过现在运行在一个不再放慢的时钟上：均值动态将信念拉回，而噪声在它们到达之前就引发了新一轮逃逸。

{cite:t}`Sargent1999` 验证了没有大的冲击触发这些稳定化：它们是常数增益学习动态的一个*内生*特征。

单条路径可能具有误导性，因此我们针对 24 个随机种子重复该模拟。

对每条路径，我们记录首次逃逸的日期、通货膨胀低于 2 的时期比例，以及在首次逃逸之后通货膨胀的移动平均值是否曾重新攀升至 4 以上。

我们还记录了首次逃逸之后，当前及滞后通货膨胀权重之和所达到的最低值；下文我们将回到这个加权和。

```{code-cell} ipython3
cg_runs = [simulate(model, λ=0.975, T_prior=300, n=1000, seed=s)
           for s in range(24)]

firsts = [first_escape(sim['y']) for sim in cg_runs]
frac_low = [(sim['y'] < 2).mean() for sim in cg_runs]
low_sw = [sim['sumweights'][t + 20:].min()
          for sim, t in zip(cg_runs, firsts) if t is not None]

print(f"paths that escape: {sum(t is not None for t in firsts)} of {len(cg_runs)}")
escaped = [t for t in firsts if t is not None]
print(f"first escape: median t = {np.median(escaped):.0f}, latest t = {max(escaped)}")
print(f"fraction of periods with y < 2: {min(frac_low):.0%} to "
      f"{max(frac_low):.0%} across seeds (mean {np.mean(frac_low):.0%})")
print(f"paths that return above 4 after escaping: "
      f"{sum(returns_to_nash(sim['y']) for sim in cg_runs)}")
print(f"lowest sum of weights after the first escape: {min(low_sw):.2f}")
```

每条路径都很早就发生了逃逸，通货膨胀在大多数时期都低于 2，并且没有任何一条路径的移动平均值在 1000 个时期内重新攀升至 4 以上。

在这个增益水平下，逃逸是*频繁*发生的，而非罕见事件。

逃逸呈指数级稀少这一大偏差命题，是小增益极限下的结论；{ref}`en_simulation` 衡量了随着增益缩小，逃逸时间是如何增长的。

## 逃逸路径与归纳假设

为什么系统会朝拉姆齐方向逃逸，而不是朝其他方向？

答案是 {doc}`phillips_adaptive` 中的**归纳假设**：当估计出的菲利普斯曲线中当期与滞后通货膨胀的权重之和趋近于零时，菲尔普斯问题会建议政府*降低*通货膨胀。

让我们将通货膨胀与这一权重之和一起绘图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 稳定化恰好对应着通货膨胀权重之和上升趋近于零
    name: fig-learn-escape-route
---
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

plt.tight_layout()
plt.show()
```

在 {numref}`fig-learn-escape-route` 中，第一次稳定化恰好对应着权重之和从其自我确认值 $-1$ 跳到约为零的水平。

权重之和此后再也没有回到 $-1$，在这次模拟中如此，在上文的 24 次模拟中也都是如此。

随着通货膨胀逐渐回升，权重之和只部分地回落向 $-1$，而下一次稳定化又恰好对应着它重新升向零。

当它达到零时，归纳假设（暂时）得到满足，菲尔普斯问题要求近乎拉姆齐水平的通货膨胀，而由此产生的数据短暂地*强化*了归纳假设——从技术意义上说这不是自我确认的，但却是自我强化的。

我们可以通过绘制估计出的菲利普斯曲线中常数项与权重之和的联合路径，直接看出这条逃逸路径。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 信念空间中的逃逸路径，按时间着色
    name: fig-learn-belief-path
---
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

在 {numref}`fig-learn-belief-path` 中，信念很早就离开了自我确认点（常数项为 10，权重之和为 $-1$），此后再也没有回到这一点。

此后，信念在归纳线下方来回移动：随着通货膨胀逐渐上升，权重之和下降而常数项上升，而每一次稳定化都会将信念重新推向归纳线（权重之和 $= 0$）。

那条线正是逃逸路径的终点，沿着这条路径，政府学会了一个索洛-托宾版本的自然率假说并稳定住了通货膨胀。

## 与预测误设均衡的关系

近乎拉姆齐的这些插曲，让人想起 {doc}`phillips_misspecified` 和 {doc}`phillips_self_confirming` 中带有最优误设预测的均衡。

在那里，一个*不含常数项但含单位根*的预测模型能够很好地逼近一个*含有*常数项的真实模型。

在这里，一种类似风味的逼近在近乎拉姆齐的插曲中发挥作用：政府估计出的菲利普斯曲线，通过将权重之和推向零，利用归纳假设来逼近一个常数项——只不过被逼近的模型并非固定不变，而是随着政府自身的信念通过菲尔普斯问题反馈回来而不断变化。

## 折扣因子的作用

{cite:t}`Sargent1999` 报告了关于政府折扣因子 $\delta$ 的两个发现：

* 朝拉姆齐方向反复出现的稳定化取决于 $\delta$ 是否接近于 1；且
* 降低 $\delta$ 会提高低通货膨胀插曲期间观测到的通货膨胀率。

该书并未报告支撑这些发现的模拟结果，因此让我们来运行一些模拟。

我们为三个折扣因子中的每一个模拟 16 个种子，对于 $\delta = 0.98$，重复使用上文前 16 次常数增益运行的结果。

```{code-cell} ipython3
δ_runs = {0.98: cg_runs[:16]}
for δ in (0.90, 0.95):
    m = AdaptivePhillips(δ=δ)
    δ_runs[δ] = [simulate(m, λ=0.975, T_prior=300, n=1000, seed=s)
                 for s in range(16)]

for δ in sorted(δ_runs):
    runs = δ_runs[δ]
    firsts = [first_escape(sim['y']) for sim in runs]
    eig = np.linalg.eigvals(jacobian_T(AdaptivePhillips(δ=δ)) - np.eye(6)).real
    print(f"δ = {δ}: {sum(t is not None for t in firsts)} of {len(runs)} escape, "
          f"median first escape t = {np.median([t for t in firsts if t is not None]):.0f}, "
          f"mean inflation for t ≥ 200 = {np.mean([sim['y'][200:].mean() for sim in runs]):.2f}, "
          f"eig(DT - I) = {np.unique(np.round(eig, 4))}")
```

在这个折扣因子取值范围内，第二个发现成立，但第一个发现并不成立。

首次逃逸的时间几乎不依赖于 $\delta$；发生变化的是政府逃逸之后通货膨胀下降的程度。

这是有道理的。

在自我确认均衡处，对每一个 $\delta$，菲尔普斯规则都是一个常数——纳什通货膨胀率，并且在 {ref}`pl_eigen` 中计算出的特征值也不依赖于 $\delta$，因此支配逃逸过程的力量大体相同。

然而，一旦归纳假设生效，政府便相信自己面对的是一条垂直的长期菲利普斯曲线以及一条短期权衡关系，而 {doc}`phillips_adaptive` 表明，此类带折扣的菲尔普斯问题不会达到零通货膨胀，政府越缺乏耐心，这种偏离就越大。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 政府的折扣因子改变的是逃逸之后的通货膨胀水平，而非逃逸发生的时间
    name: fig-learn-discount
---
fig, ax = plt.subplots(figsize=(9, 4.5))
for δ in sorted(δ_runs):
    ax.plot(np.arange(19, 1000), moving_average(δ_runs[δ][1]['y']),
            lw=0.9, label=rf'$\delta = {δ}$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$')
ax.set_ylabel('20-period moving average of $y_t$')
ax.legend()
plt.show()
```

{numref}`fig-learn-discount` 展示了一个种子的结果：三个政府大约在同一时点发生逃逸，而在逃逸发生后，耐心较低的政府会稳定在更高的通货膨胀水平上。

## 预期效用

这个适应性模型是大卫·克雷普斯（David Kreps）所称的**预期效用**（anticipated utility）模型的一个例子 {cite}`Kreps1998`。

政府对一个临时误设的模型——一个系数固定的菲利普斯曲线——加以调整，纳入最新的观测数据，并沿途重新优化。

在 $t$ 时刻做决策时，它的行为就好像其当前的估计值 $\gamma_{t-1}$ 将永远支配菲利普斯曲线一样，使用的是若 $\gamma$ 真的是时间不变的话本应是最优的同一政策函数 $h(\cdot)$。

这是对理性预期的一个微小偏离：日历时间只通过漂移的信念 $\gamma_t$ 进入。

政府以两种不同的方式对待系数漂移。

当它进行*估计*时，它允许漂移存在：常数增益之所以会贴现旧数据，正是因为政府怀疑菲利普斯曲线在游走，而且在 {eq}`pl_rls` 的卡尔曼滤波形式中，它赋予了漂移一个正的协方差。

当它进行*决策*时，它却忽略了漂移：它的行为就好像 $\gamma_{t-1}$ 永远不会改变一样。

预期效用因此拉伸了线性二次问题的一个性质。

确定性等价（{doc}`theil_1`）使得面对*可加*不确定性的决策者能够用期望值替代随机变量。

预期效用型政府对其不确定的系数 $\gamma$ 也采用了同样的做法，然而这些系数是以乘法形式进入的，而在这种情形下确定性等价不再成立。

贝叶斯型决策者在选择 $y_t$ 时，则会考虑到其信念将会发生变化（{doc}`rational_learning_re`），而稳健型决策者则会防范其模型的误设。

然而，正如模拟所示，这种对理性的适度偏离已足以对美国通货膨胀的兴衰给出一个丰富的解释，并为"econometric 政策评估的证明"提供支撑。

## 结论

在很长的时间段内，一个适应性政府学会了产生*优于纳什*的结果。

这些结果来自适应性的反复动态：在最小二乘法下将系统拉向自我确认均衡的均值动态继续发挥作用，但在常数增益下，噪声使系统能够反复地逃逸向拉姆齐结果。

从一个自我确认均衡出发，适应性算法逐渐使政府将足够的权重放在归纳假设上，以至于偶然的观测结果最终促成了一次稳定化。

在递减增益下，同样的力量在一个不断变慢的时钟上发挥作用：最小二乘法只是渐近地收敛到自我确认均衡，而在有限样本中，一次早期的逃逸可能会持续存在。

适应性使得政府的信念成为一种隐藏状态，它给通货膨胀注入了序列相关性——因此，一个外部预测者若使用随机系数模型，或者进行卢卡斯在其批判 {cite}`lucas1976econometric` 中所指出的那种不断调整，会做得更好。

（在这个模型中，失业率保持序列不相关，因为公众知道政府的规则，只有通货膨胀的意外变动才会影响失业率。）

在这个意义上，适应性模型包含了证明经济计量政策评估的基础——这正是 {doc}`phillips_two_stories` 中两个故事的第二个。

最后这一观察是可以检验的，值得在任何人查看数据之前记录下模型的预测。

如果政府的信念确实是一个隐藏的漂移状态，那么对战后通货膨胀与失业率的简化形式描述应当显示出*漂移的系数*，而不仅仅是漂移的冲击方差。

该模型还预测了通货膨胀的持续性应当如何随其水平变化。

在自我确认均衡处，通货膨胀是序列不相关的，而在一次常数增益逃逸之后，它则表现出持续性（{ref}`pl_ex4`），因此低通货膨胀应当伴随着高持续性。

{doc}`phillips_drifts_volatilities` 将一个漂移系数模型拟合到数据上，并给出了结论：持续性随通货膨胀水平的变化而升降，并且在沃尔克反通胀期间下降，这与上述预测相反。

在此之前，本系列将继续介绍相关理论。

{doc}`phillips_escaping_nash` 对逃逸路径进行了解析刻画，{doc}`phillips_priors` 展示了政府关于系数漂移的先验信念如何塑造均值动态与逃逸行为，而 {doc}`phillips_policy_evaluation` 则沿用本书第9章的方法，在美国数据上对适应性模型进行了估计。

## 练习

```{exercise-start}
:label: pl_ex1
```

以上所有内容都取决于常数增益，因此值得看看其影响究竟有多大。

用 $\lambda \in \{0.99, 0.975, 0.95\}$ 和 $T_0 = 300$ 模拟经典适应性模型，每种情况取 24 个随机种子，衡量通货膨胀朝拉姆齐逃逸的频率。

使用两个衡量*频率*的统计量：

* 首次逃逸的中位日期，定义方式见 {ref}`pl_decreasing`；以及
* 每 1000 期中的逃逸事件数，其中一次事件在当前与滞后通货膨胀权重之和升至 $-0.1$ 以上时开始，并且该权重和必须先回落至 $-0.3$ 以下，才能开始下一次事件计数。

同时计算 $y_t < 2$ 的时期所占比例。

更大的增益（更小的 $\lambda$，对过去更快地打折扣）如何影响每个统计量？

为什么低通货膨胀时期所占比例是衡量政府逃逸频率的一个较差指标？

```{note}
本练习的一个更具挑战性的版本是构建**凯恩斯主义**适应性模型，其中政府沿相反方向拟合菲利普斯曲线，将通货膨胀对失业率做回归。

回归量是 $X_{K,t} = \begin{bmatrix} U_t & U_{t-1} & U_{t-2} & y_{t-1} & y_{t-2} & 1 \end{bmatrix}^\top$；
政府在 $y_t = \beta^\top X_{K,t} + \varepsilon_{K,t}$ 中估计 $\beta$，然后使用 {doc}`phillips_self_confirming` 中的反演公式
$\gamma_1 = \beta_1^{-1}$、$\gamma_{-1} = -\beta_{-1}/\beta_1$ 将其反转为 $\gamma$，之后再求解菲尔普斯问题。

凯恩斯主义自我确认通货膨胀率为 10。

在本书的模拟中（$\lambda = 0.99$，$T_0 = 300$，图 8.3a–8.3d），凯恩斯主义系统“长时间处于自我确认均衡之外，并且与经典系统一样，反复逃逸至拉姆齐结果”{cite}`Sargent1999`。

其雅可比矩阵存在一个重复特征值 $-.2$，这使得均值动态变得缓慢，用书中的话说，“维持了接近拉姆齐结果的长期阶段”。
```

```{exercise-end}
```

```{solution-start} pl_ex1
:class: dropdown
```

我们用一个小函数来统计事件数，并重复利用上文中 $\lambda = 0.975$ 的 24 次模拟运行结果。

```{code-cell} ipython3
def escape_episodes(sw, up=-0.1, down=-0.3):
    "Upcrossings of `up` by the sum of weights, re-armed only below `down`."
    count, armed = 0, True
    for v in sw:
        if armed and v > up:
            count, armed = count + 1, False
        elif not armed and v < down:
            armed = True
    return count


gain_runs = {0.975: cg_runs}
for λ in (0.99, 0.95):
    gain_runs[λ] = [simulate(model, λ=λ, T_prior=300, n=1000, seed=s)
                    for s in range(24)]

print(f"{'λ':>6} {'median first escape':>20} {'episodes per 1000':>18} "
      f"{'fraction y < 2':>15}")
for λ in (0.99, 0.975, 0.95):
    runs = gain_runs[λ]
    firsts = [first_escape(sim['y']) for sim in runs]
    episodes = [escape_episodes(sim['sumweights']) * 1000 / len(sim['y'])
                for sim in runs]
    print(f"{λ:>6} {np.median([t for t in firsts if t is not None]):20.0f} "
          f"{np.mean(episodes):18.2f} "
          f"{np.mean([(sim['y'] < 2).mean() for sim in runs]):15.2f}")
```

更大的增益使逃逸来得更早也更频繁：随着 $\lambda$ 的降低，首次逃逸的中位日期提前，每 1000 期中的事件数也上升。

而 $y_t < 2$ 的时期所占比例几乎没有变化，并且这种变化不一定是单调的。

原因在于，更大的增益也会加快返回的进程，因为 {ref}`pl_decreasing` 中的那个时钟走得更快。

因此，每一段接近拉姆齐的时期都更短，而且其间的通货膨胀也更嘈杂。

接近拉姆齐的时间占比，混合了政府逃逸的频率与停留时长这两个因素，而当增益上升时，这两者的变化方向正好相反。

```{solution-end}
```

```{exercise-start}
:label: pl_ex2
```

{numref}`fig-learn-ls` 与 {numref}`fig-learn-ls-prior` 表明，最小二乘法是否发生逃逸取决于先验的紧密程度。

对 $T_0 \in \{300, 800, 5000\}$，模拟最小二乘系统（$\lambda = 1$），每种情况取 24 个随机种子，每条路径模拟 1000 期。

1. 对每个 $T_0$，报告逃逸的路径数、首次逃逸的中位日期，以及已逃逸路径中有多少重新回到 4 以上。
1. 对发生逃逸的路径，计算首次逃逸时的人工时间 $\tau = \log\left((T_0 + t)/T_0\right)$。利用这一结果解释为何紧先验在 1000 期内很少发生逃逸。
1. 解释为何没有一条已逃逸的路径会返回，尽管最小二乘法会收敛到自我确认均衡。{ref}`pl_eigen` 中位于 $\sqrt{t}$ 收敛边界上的特征值 $-\tfrac12$，是否就是这一现象的原因？

```{exercise-end}
```

```{solution-start} pl_ex2
:class: dropdown
```

```{code-cell} ipython3
ls_prior_runs = {T0: [simulate(model, λ=1.0, T_prior=T0, n=1000, seed=s)
                      for s in range(24)]
                 for T0 in (300, 800, 5000)}

for T0, runs in ls_prior_runs.items():
    firsts = np.array([t for t in (first_escape(sim['y']) for sim in runs)
                       if t is not None])
    line = f"T_0 = {T0:>4}: {len(firsts):>2} of {len(runs)} escape"
    if len(firsts):
        τ = np.log((T0 + firsts) / T0)
        line += (f", median first escape t = {np.median(firsts):.0f} "
                 f"(τ = {np.median(τ):.2f}), "
                 f"{sum(returns_to_nash(sim['y']) for sim in runs)} return")
    else:
        line += f"; τ at t = 1000 is only {np.log((T0 + 1000) / T0):.2f}"
    print(line)
```

更松的先验会导致更频繁、更早发生的逃逸，并且没有一条已逃逸的路径会返回。

以人工时间衡量，在两种较松的先验下，首次逃逸发生在数十分之一个单位之后。

当 $T_0 = 5000$ 时，1000 期仅提供约 $\log(6000/5000) \approx 0.18$ 个单位的人工时间，少于较松先验下大多数发生逃逸的路径所需要的时间。

没有一条已逃逸的路径会返回，这是因为正如 {ref}`pl_decreasing` 所示，返回需要许多单位的人工时间，而最小二乘法的时钟是以指数方式放慢的。

特征值 $-\tfrac12$ 并不是原因。

它支配的是递减增益估计量一旦接近自我确认均衡后收敛的速度，以及其渐近分布。

它并不能说明脱离自我确认均衡的概率有何特殊之处——这由本入门读物中的大偏差对象所支配——也无法说明从远处返回的速度，后者取决于全局均值动态；而这些动态在归纳假设附近是缓慢的。

{doc}`phillips_priors` 表明，在增益不会消失的卡尔曼滤波中，政府的先验会如何产生持续存在的逃逸，而 {ref}`br_one_recursion` 中关于递减增益与常数增益的要点总结了一般理论。

```{solution-end}
```

```{exercise-start}
:label: pl_ex3
```

{ref}`pl_eigen` 计算了在 $\theta = 1$ 时 $DT - I$ 的特征值。

1. 验证 {eq}`pl_tmap` 在偏离自我确认均衡时依然成立。将政府的信念冻结在某次常数增益模拟在 $t = 1000$ 时所持有的水平上，在该固定规则下模拟经济 50,000 期，将 $U_t$ 对 $X_{C,t}$ 做回归，并将回归系数与 `T_map` 比较。
1. 在固定其他参数的情况下，对 $\theta \in \{0.5, 1, 2\}$，计算自我确认均衡处 $DT - I$ 的特征值，并与 $-1/(1+\theta^2)$ 比较。
1. 对这些 $\theta$ 中哪些值，最小二乘估计以通常的 $\sqrt{t}$ 速率收敛？将结果与 {ref}`pc_mean_dynamics` 和 {ref}`sc_estability` 比较。

```{exercise-end}
```

```{solution-start} pl_ex3
:class: dropdown
```

第一部分我们冻结规则并进行模拟。

```{code-cell} ipython3
γ_frozen = cg_runs[1]['γ']
h = model.phelps_h(γ_frozen)
rng = np.random.default_rng(0)
N = 50_000
v1 = model.σ1 * rng.standard_normal(N)
v2 = model.σ2 * rng.standard_normal(N)
U = model.U_star - model.θ * v2 + v1
y = np.zeros(N)
y[:2] = h[4]
for t in range(2, N):
    y[t] = h @ [U[t-1], U[t-2], y[t-1], y[t-2], 1.0] + v2[t]

X_C = np.column_stack([y[2:], U[1:-1], U[:-2], y[1:-1], y[:-2], np.ones(N - 2)])
coef = np.linalg.lstsq(X_C, U[2:], rcond=None)[0]
print("regression of U on X_C:", np.round(coef, 3))
print("T(γ) from the formula: ", np.round(T_map(model, γ_frozen), 3))
```

两者在抽样误差范围内是一致的，因为 $v_{1t}$ 与每一个回归量都相互独立。

接下来计算特征值。

```{code-cell} ipython3
for θ in (0.5, 1.0, 2.0):
    m = AdaptivePhillips(θ=θ)
    eig = np.sort(np.linalg.eigvals(jacobian_T(m) - np.eye(6)).real)
    print(f"θ = {θ}: eig(DT - I) = {np.round(eig, 4)}, "
          f"-1/(1 + θ²) = {-1 / (1 + θ**2):.4f}")
```

对每个 $\theta$，特征值为 $-1$（一次）和 $-1/(1+\theta^2)$（五次）。

$\sqrt{t}$ 速率要求 $-1/(1+\theta^2) < -\tfrac12$，也就是 $\theta < 1$。

因此，当 $\theta = 0.5$ 时，最小二乘法以通常的速率收敛；当 $\theta = 1$ 时恰好处于边界；而当 $\theta = 2$ 时，由于均值动态存在重复特征值 $-0.2$，收敛会更慢。

这些数字与 {ref}`pc_mean_dynamics` 中标量均值动态的斜率 $\mathcal{M} = -1/(1+\theta^2)$，以及 {ref}`sc_estability` 中经典的 E-稳定性特征值完全一致。

更陡峭的菲利普斯曲线使静态菲尔普斯规则对一个被相信的滞后系数反应更为强烈，这会将 $DT$ 的对角元素 $\theta^2/(1+\theta^2)$ 推向 1，并削弱均值动态的拉力。

```{solution-end}
```

```{exercise-start}
:label: pl_ex4
```

结论部分指出，在自我确认均衡处通货膨胀是序列不相关的，而在一次常数增益逃逸之后则表现出持续性。

通过在 $y_t$ 对常数项和 $y_{t-1}$ 做最小二乘回归中的斜率系数来衡量持续性，以验证这一说法。

1. 对于 $T_0 = 5000$ 的最小二乘经济（它一直停留在自我确认均衡附近），对 {ref}`pl_ex2` 中每条路径的前 200 期计算该系数。
1. 对于 {ref}`pl_ex1` 中的常数增益经济，在首次逃逸之后 20 期开始的 200 期窗口内计算该系数。
1. 对 $T_0 = 800$ 的最小二乘路径中发生逃逸的那些路径做同样的计算。
1. 该模型预测通货膨胀水平与其持续性之间存在何种关系？将结果与 {doc}`phillips_drifts_volatilities` 比较。

```{exercise-end}
```

```{solution-start} pl_ex4
:class: dropdown
```

```{code-cell} ipython3
def ar1(z):
    "Slope of a least squares regression of z_t on a constant and z_{t-1}."
    X = np.column_stack([np.ones(len(z) - 1), z[:-1]])
    return np.linalg.lstsq(X, z[1:], rcond=None)[0][1]


def after_escape(runs, start=20, length=200):
    "AR(1) coefficients over a window that starts `start` periods after the first escape."
    out = []
    for sim in runs:
        t = first_escape(sim['y'])
        if t is not None and t + start + length <= len(sim['y']):
            out.append(ar1(sim['y'][t + start:t + start + length]))
    return np.array(out)


def summarize(label, values):
    print(f"{label:28s} median {np.median(values):+.2f}, "
          f"range {values.min():+.2f} to {values.max():+.2f} ({len(values)} paths)")


summarize("least squares, T_0 = 5000:",
          np.array([ar1(sim['y'][:200]) for sim in ls_prior_runs[5000]]))
for λ in (0.99, 0.975, 0.95):
    summarize(f"constant gain, λ = {λ}:", after_escape(gain_runs[λ]))
summarize("least squares, T_0 = 800:", after_escape(ls_prior_runs[800]))
```

在自我确认均衡附近，AR(1) 系数集中在零附近。

在一次常数增益逃逸之后，通货膨胀表现出持续性，且增益越大持续性越强；在 $\lambda = 0.99$ 时这一效应较小。

在一次能够持续下去的最小二乘逃逸之后，持续性再次变小。

因此，持续性来自政府信念的运动，而常数增益使这种运动得以维持，并非来自低通货膨胀本身。

在常数增益下，该模型因而预测：当通货膨胀较低时持续性应较高，当通货膨胀较高时持续性应较低，而从高通货膨胀向低通货膨胀的转变会提高持续性。

{doc}`phillips_drifts_volatilities`（{ref}`csdv_persistence`）在美国数据中发现了相反的情形：持续性随通货膨胀水平的升降而升降，并在沃尔克反通货膨胀期间下降。

```{solution-end}
```
