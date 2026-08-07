---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: 卡尔曼滤波器与向量自回归
  headings:
    Overview: 概述
    The state space system: 状态空间系统
    The Kalman filter: 卡尔曼滤波器
    The Kalman filter::Starting distribution: 起始分布
    The Kalman filter::Derivation: 推导
    The Kalman filter::The Kalman filter recursions: 卡尔曼滤波器的递归
    The Kalman filter::The matrix Riccati equation: 矩阵黎卡提方程
    The Gram-Schmidt process: Gram-Schmidt 过程
    Hidden Markov model: 隐马尔可夫模型
    Estimation: 估计
    Estimation::The innovations representation: 新息表示
    Estimation::The likelihood function: 似然函数
    Estimation::Bayesian inference: 贝叶斯推断
    Vector autoregressions and the Kalman filter: 向量自回归与卡尔曼滤波器
    Vector autoregressions and the Kalman filter::Convergence to a steady state: 收敛到稳态
    Vector autoregressions and the Kalman filter::A time-invariant VAR: 一个时不变 VAR
    Vector autoregressions and the Kalman filter::Interpreting VARs: 解释 VAR
    Spectral factorization identity: 谱分解恒等式
    Spectral factorization identity::Two representations of the spectral density: 谱密度的两种表示
    Spectral factorization identity::The spectral factorization identity: 谱分解恒等式
    Spectral factorization identity::Wold and autoregressive representations: Wold 表示和自回归表示
    Python implementation: Python 实现
    Python implementation::A scalar hidden AR(1) model: 一个标量隐藏 AR(1) 模型
    Python implementation::Convergence of the Riccati equation: 黎卡提方程的收敛
    Python implementation::The VAR representation: VAR 表示
    Python implementation::Likelihood evaluation: 似然评估
    An example: 一个例子
    An example::A linear state-space system and its filter: 一个线性状态空间系统及其滤波器
    An example::Impulse responses of $y_t$ to the innovations $a_t$: $y_t$ 对新息 $a_t$ 的脉冲响应
    An example::Bivariate VAR(2) in state-space form: 状态空间形式的二元 VAR(2)
    'An example::Numerical example: impulse responses to innovations': 数值例子：对新息的脉冲响应
    Summary: 总结
    Exercises: 练习
---

# 卡尔曼滤波器与向量自回归

```{index} single: Kalman Filter
```

```{index} single: Vector Autoregression; and Kalman filter
```

除了 Anaconda 中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## 概述

本讲座为线性高斯状态空间系统推导**卡尔曼滤波器**，然后用它来构建**向量自回归（VAR）**。

它建立在 {doc}`kalman` 之上，在那里滤波器是通过滤波分布和预测分布引入的。

它补充了 {doc}`kalman_2`，在那里同样的递归被用于一个工人-企业学习问题中。

本讲座将注意力从对隐藏状态的滤波，转移到表示和估计由该隐藏状态所生成的可观测过程。

我们的方法依赖于反复应用**总体线性最小二乘**投影公式，其核心洞见是：计算一个联合正态随机向量的条件期望，等同于运行一个总体普通最小二乘法回归。

本讲座涵盖：

- 从第一性原理推导卡尔曼滤波器的递归
- 支配条件协方差矩阵的**矩阵黎卡提差分方程**
- **新息表示**以及 **Gram-Schmidt** 白化性质
- **隐马尔可夫模型**的结构
- 状态空间系统的**似然函数**及其在最大似然估计和贝叶斯估计中的作用
- 时不变卡尔曼滤波器如何生成一个**向量自回归**
- 为什么卡尔曼滤波器是*解释*由经济数据估计得到的 *VAR* 的一个基本工具

## 状态空间系统

卡尔曼滤波器适用于以下针对 $t \geq 0$ 的**状态空间系统**：

$$
\begin{aligned}
x_{t+1} &= A x_t + C w_{t+1} \\
y_t      &= G x_t + v_t
\end{aligned}
$$ (eq:statespace)

其中

- $x_t$ 是一个 $n \times 1$ 的**状态向量**（隐藏的、不可观测的）
- $y_t$ 是一个关于隐藏状态的 $m \times 1$ **信号**向量（可观测的）
- $w_{t+1}$ 是一个 $p \times 1$ 的独立同分布的正态随机变量序列，均值为
  $0$，协方差矩阵为单位矩阵
- $v_t$ 是一个独立同分布的正态随机变量序列，均值为零，协方差矩阵为 $R$
- 对所有 $t+1$ 和 $s \geq 0$，$w_{t+1}$ 与 $v_s$ 正交

系数矩阵具有以下维度：
$A$ 是 $n \times n$，$C$ 是 $n \times p$，$G$ 是 $m \times n$，$R$ 是 $m \times m$。

初始状态满足

$$
x_0 \sim N(\hat{x}_0, \Sigma_0)
$$ (eq:kalf3)

在时刻 $t$，我们观测到 $y_t, \ldots, y_0$ 而*不能*观测到 $x_t, \ldots, x_0$，
并且我们知道由 {eq}`eq:statespace` 和 {eq}`eq:kalf3` 所隐含的所有一阶矩和二阶矩。

## 卡尔曼滤波器

### 起始分布

从 $t = 0$ 开始沿时间向前推进，在观测到 $y_0$ 之前，
设定 {eq}`eq:statespace`-{eq}`eq:kalf3` 意味着 $y_0$ 的边缘分布为

$$
y_0 \sim N(G \hat{x}_0,\; G \Sigma_0 G^\top + R)
$$ (eq:kalf4)

对于 $t \geq 0$，令 $y^t = [y_t, y_{t-1}, \ldots, y_0]$。

我们想要一个便于表示的、关于 $y_t$ 在给定历史 $y^{t-1}$ 条件下的条件分布的递归表示。

卡尔曼滤波器通过为 $\hat{x}_t$ 和 $\Sigma_t$ 构造递归公式来实现这一点，使得在 $y^{t-1}$ 条件下 $y_t$ 的分布将 {eq}`eq:kalf4` 推广为

$$
y_t \sim N(G \hat{x}_t,\; G \Sigma_t G^\top + R)
$$ (eq:kalf400)

其中 $t \geq 1$，而在 $y^{t-1}$ 条件下 $x_t$ 的分布为 $N(\hat{x}_t, \Sigma_t)$。

对象 $\hat{x}_t$ 和 $\Sigma_t$ 刻画了**总体回归**

$$
\hat{x}_t = \mathbb{E}[x_t \mid y_{t-1}, \ldots, y_0]
$$

以及**条件协方差矩阵**

$$
\Sigma_t = \mathbb{E}\!\left[(x_t - \hat{x}_t)(x_t - \hat{x}_t)^\top \mid y_{t-1}, \ldots, y_0\right]
$$

### 推导

在每个时刻，我们的方法是*将我们所不知道的对我们所知道的进行回归*。

```{note}
因为我们的假设意味着 $\{x_t, y_t\}_{t=0}^\infty$ 是一个联合正态随机过程，所以线性最小二乘回归等于条件数学期望。

下面每一步都是对贝叶斯法则的应用。

在没有联合正态性、只假设所有均值和协方差存在的较弱假设下，同样的计算得到"广义条件期望"，只有当这些条件期望是线性的时候它们才与真正的条件期望一致。
```

我们在 $t = 0$ 时刻已知 $\hat{x}_0$ 和 $\Sigma_0$。

$y_0$ 中相对于 $(\hat{x}_0, \Sigma_0)$ 关于 $x_0$ 的新信息即为**新息**

$$
a_0 \equiv y_0 - G \hat{x}_0
$$

令 $L_0$ 为隐藏状态误差 $x_0-\hat{x}_0$ 对信号意外 $y_0-G\hat{x}_0$ 的总体回归系数。

条件均值
$\mathbb{E}[x_0 \mid y_0] = \hat{x}_0 + L_0(y_0 - G\hat{x}_0)$ 满足总体回归公式

$$
x_0 - \hat{x}_0 = L_0(y_0 - G\hat{x}_0) + \eta
$$ (eq:kalf5)

其中 $\eta$ 是最小二乘残差。

$\eta$ 与 $(y_0 - G\hat{x}_0)$ 的正交性通过正规方程确定了 $L_0$

$$
\mathbb{E}(x_0 - \hat{x}_0)(y_0 - G\hat{x}_0)^\top
= L_0\, \mathbb{E}(y_0 - G\hat{x}_0)(y_0 - G\hat{x}_0)^\top
$$

计算矩矩阵并求解 $L_0$ 得到

$$
L_0 = \Sigma_0 G^\top(G \Sigma_0 G^\top + R)^{-1}
$$ (eq:kalf6)

因此 $L_0$ 更新对 $x_0$ 的估计，而 $K_0=A L_0$ 更新对 $x_1$ 的预测。

为了预测 $x_1$，注意

$$
x_1 = A\hat{x}_0 + A(x_0 - \hat{x}_0) + C w_1
$$ (eq:kalf6a)

应用 {eq}`eq:kalf5` 得到 $\mathbb{E}[x_1 \mid y_0] = A\hat{x}_0 + AL_0(y_0 - G\hat{x}_0)$，
我们将其写为

$$
\hat{x}_1 = A\hat{x}_0 + K_0(y_0 - G\hat{x}_0)
$$ (eq:kalf7)

其中时刻 0 的**卡尔曼增益**为

$$
K_0 = A \Sigma_0 G^\top(G \Sigma_0 G^\top + R)^{-1}
$$ (eq:kalf7a)

将 {eq}`eq:kalf7` 从 {eq}`eq:kalf6a` 中减去得到

$$
x_1 - \hat{x}_1 = A(x_0 - \hat{x}_0) + C w_1 - K_0(y_0 - G\hat{x}_0)
$$ (eq:kalf8)

利用 {eq}`eq:kalf8` 和 $y_0 = G x_0 + v_0$ 来计算
$\Sigma_1 \equiv \mathbb{E}[(x_1 - \hat{x}_1)(x_1 - \hat{x}_1)^\top \mid y_0]$ 得到

$$
\Sigma_1 = (A - K_0 G)\Sigma_0(A - K_0 G)^\top + CC^\top + K_0 R K_0^\top
$$ (eq:kalf9)

因此 $f(x_1 \mid y_0) \sim N(\hat{x}_1, \Sigma_1)$。

收集时刻 $0$ 的方程：

$$
\begin{aligned}
a_0       &= y_0 - G\hat{x}_0 \\
K_0       &= A\Sigma_0 G^\top(G\Sigma_0 G^\top + R)^{-1} \\
\hat{x}_1 &= A\hat{x}_0 + K_0 a_0 \\
\Sigma_1  &= CC^\top + K_0 R K_0^\top + (A - K_0 G)\Sigma_0(A - K_0 G)^\top
\end{aligned}
$$ (eq:kalf1000)

系统 {eq}`eq:kalf1000` 将一个均值-协方差对 $(\hat{x}_0, \Sigma_0)$ 映射为一个新的对 $(\hat{x}_1, \Sigma_1)$，并带有辅助输出 $(a_0, K_0)$。

认识到"我们在第 1 期开始时所处的情形与第 0 期开始时相同"激活了一个递归，即**卡尔曼滤波器**。

### 卡尔曼滤波器的递归

迭代系统 {eq}`eq:kalf1000` 得到针对 $t \geq 0$ 的卡尔曼滤波器：

$$
\begin{aligned}
a_t           &= y_t - G\hat{x}_t \\
K_t           &= A\Sigma_t G^\top(G\Sigma_t G^\top + R)^{-1} \\
\hat{x}_{t+1} &= A\hat{x}_t + K_t a_t \\
\Sigma_{t+1}  &= CC^\top + K_t R K_t^\top + (A - K_t G)\Sigma_t(A - K_t G)^\top
\end{aligned}
$$ (eq:kalf10)

这里 $K_t$ 是时刻 $t$ 的**卡尔曼增益**。

### 矩阵黎卡提方程

将 {eq}`eq:kalf10` 第二行中 $K_t$ 的表达式代入第四行，得到一个等价的更新公式：

$$
\Sigma_{t+1} = A\Sigma_t A^\top + CC^\top
  - A\Sigma_t G^\top(G\Sigma_t G^\top + R)^{-1} G\Sigma_t A^\top
$$ (eq:riccati)

方程 {eq}`eq:riccati` 是**矩阵黎卡提差分方程**。

它支配着条件协方差矩阵序列 $\{\Sigma_t\}_{t=0}^\infty$，而不涉及观测值 $\{y_t\}$。

```{index} single: Riccati equation; matrix difference
```

## Gram-Schmidt 过程

随机向量

$$
a_t = y_t - \mathbb{E}[y_t \mid y_{t-1}, \ldots, y_0]
$$

是 $y_t$ 相对于 $y^{t-1}$ 的**新息**，即 $y_t$ 中无法由过去观测预测的部分。

注意 $\mathbb{E} a_t a_t^\top = G\Sigma_t G^\top + R$，即在卡尔曼增益公式 {eq}`eq:kalf10` 中其逆矩阵出现的那个矩阵。

利用 $a_t = G(x_t - \hat{x}_t) + v_t$ 直接计算可以表明
$\mathbb{E} a_t a_{t-1}^\top = 0$，更一般地，$\mathbb{E}[a_t \mid a_{t-1}, \ldots, a_0] = 0$。

```{note}
一个从第一性原理出发的替代论证：令 $H(y^t)$ 表示 $y^t$ 的闭线性张成空间。

由于 $a_{t+1} = y_{t+1} - \mathbb{E}[y_{t+1} \mid y^t]$ 是一个最小二乘误差，$a_{t+1} \perp H(y^t)$，特别地 $a_{t+1} \perp a_t$。

因此 $\{a_t\}$ 是关于 $\{y_t\}$ 的一个白噪声新息过程。
```

有时 {eq}`eq:kalf10` 被称为**白化滤波器**：它以信号过程 $\{y_t\}$ 为输入，并产生白噪声新息过程 $\{a_t\}$ 作为输出。

类似地定义 $H(a^t)$，线性空间 $H(a^t)$ 是线性空间 $H(y^t)$ 的一个正交基。

卡尔曼滤波器不是通过一个大的回归来计算 $\mathbb{E}[x_t \mid y_{t-1}, \ldots, y_0]$，而是对基 $[a_{t-1}, \ldots, a_0]$ 的一系列相继正交分量执行一系列小回归，这是 **Gram-Schmidt 过程**的一个实例。

```{index} single: Gram-Schmidt process
```

## 隐马尔可夫模型

系统 {eq}`eq:statespace`-{eq}`eq:kalf3` 是一个**隐马尔可夫模型**的例子。

```{index} single: hidden Markov model
```

可观测过程 $\{y_t\}_{t=0}^\infty$ *不是*马尔可夫的，但隐藏过程 $\{x_t\}_{t=0}^\infty$ *是*马尔可夫的。

均值和协方差过程 $\{(\hat{x}_t, \Sigma_t)\}$ 也是马尔可夫的，它们是在 $[y_{t-1}, \ldots, y_0]$ 条件下 $x_t$ 分布的充分统计量。

## 估计

### 新息表示

从卡尔曼滤波器中产生的**新息表示**为

$$
\begin{aligned}
\hat{x}_{t+1} &= A\hat{x}_t + K_t a_t \\
y_t           &= G\hat{x}_t + a_t
\end{aligned}
$$ (eq:innovrep)

其中对于 $t \geq 1$ 有 $\hat{x}_t = \mathbb{E}[x_t \mid y^{t-1}]$，且
$\mathbb{E}[a_t a_t^\top \mid y^{t-1}] = G\Sigma_t G^\top + R \equiv \Omega_t$。

对于 $t \geq 1$，$\mathbb{E}[y_t \mid y^{t-1}] = G\hat{x}_t$，且在给定 $y^{t-1}$ 条件下 $y_t$ 的条件分布为 $N(G\hat{x}_t, \Omega_t)$。

因此，从卡尔曼滤波器递归中产生的对象 $(G\hat{x}_t, \Omega_t)$ 完全刻画了这个条件分布。

### 似然函数

我们可以将样本 $(y_T, y_{T-1}, \ldots, y_0)$ 的似然分解为

$$
f(y_T, \ldots, y_0)
  = f(y_T \mid y^{T-1})\, f(y_{T-1} \mid y^{T-2}) \cdots f(y_1 \mid y_0)\, f(y_0)
$$ (eq:diff100)

$m \times 1$ 向量 $y_t$ 的对数条件密度为

$$
\log f(y_t \mid y^{t-1})
  = -\frac{m}{2}\log(2\pi)
    - \frac{1}{2}\log\det(\Omega_t)
    - \frac{1}{2}\, a_t^\top \Omega_t^{-1} a_t
$$ (eq:gauss100)

同时使用 {eq}`eq:gauss100` 和 {eq}`eq:kalf10`，我们可以对任何构成矩阵 $A, G, C, R$ 基础的参数向量 $\theta$ 递归地计算似然 {eq}`eq:diff100`。

此类计算是高效计算自由参数的**最大似然估计**策略的核心。

### 贝叶斯推断

似然函数在**贝叶斯推断**中也是核心。

其中 $\theta$ 是参数向量，$y_0^T$ 是数据，$\tilde{p}(\theta)$ 是在看到 $y_0^T$ 之前关于 $\theta$ 的先验密度，贝叶斯法则给出**后验**

$$
\tilde{p}(\theta \mid y_0^T)
  = \frac{f(y_0^T \mid \theta)\,\tilde{p}(\theta)}
         {\int f(y_0^T \mid \theta)\,\tilde{p}(\theta)\, d\theta}
$$

分母是边缘联合密度 $f(y_0^T)$。

## 向量自回归与卡尔曼滤波器

### 收敛到稳态

在 {cite:t}`AHMS1996` 所讨论的条件下，对黎卡提方程 {eq}`eq:riccati` 的迭代从任何正半定初始值 $\Sigma_0$ 出发都收敛到一个**时不变**矩阵 $\Sigma$。

{eq}`eq:riccati` 的一个时不变不动点 $\Sigma_t = \Sigma$ 是 $x_t$ 围绕

$$
\mathbb{E}\!\left[x_t \mid \{y_s\}_{s \leq t-1}\right]
$$

的协方差矩阵，其中条件作用扩展到**半无限**的过去 $s \leq t-1$。

### 一个时不变 VAR

如果不动点 $\Sigma$ 存在，且我们在 $\Sigma_0 = \Sigma$ 处初始化滤波器，则新息表示 {eq}`eq:innovrep` 变为时不变：

$$
\begin{aligned}
\hat{x}_{t+1} &= A\hat{x}_t + K a_t \\
y_t           &= G\hat{x}_t + a_t
\end{aligned}
$$ (eq:innovti)

其中 $\mathbb{E} a_t a_t^\top = G\Sigma G^\top + R$，且**稳态卡尔曼增益**为
$K = A\Sigma G^\top(G\Sigma G^\top + R)^{-1}$。

从 {eq}`eq:innovti` 我们得到 $\hat{x}_{t+1} = (A - KG)\hat{x}_t + K y_t$。

如果 $A - KG$ 的特征值的模严格有界地低于 1，我们可以向前求解此方程得到

$$
\hat{x}_{t+1} = \sum_{j=0}^\infty (A - KG)^j K\, y_{t-j}
$$ (eq:xhatform)

将 {eq}`eq:xhatform` 代入 {eq}`eq:innovti` 的观测方程，得到**向量自回归**

$$
y_t = G \sum_{j=0}^\infty (A - KG)^j K\, y_{t-j-1} + a_t
$$ (eq:var1)

由构造可知

$$
\mathbb{E}\!\left[a_t\, y_{t-j-1}^\top\right] = 0 \quad \forall\, j \geq 0
$$ (eq:varorth)

正交条件 {eq}`eq:varorth` 将 {eq}`eq:var1` 识别为一个向量自回归。

通过 $L x_{t+1} \equiv x_t$ 定义滞后算子 $L$，由 {eq}`eq:innovti` 推导出的**移动平均表示**为

$$
y_t = \left[I + G(I - AL)^{-1} KL\right] a_t
    = \left[I + G\sum_{j=0}^\infty A^j K L^{j+1}\right] a_t
$$

```{index} single: vector autoregression
```

### 解释 VAR

经济模型的均衡（或它们的线性或对数线性近似）通常采用状态空间系统 {eq}`eq:statespace` 的形式。

这个隐马尔可夫模型通过 $p \times 1$ 冲击向量 $w_{t+1}$ 扰动状态 $x_t$，并通过 $m \times 1$ 测量误差 $v_t$ 扰动可观测量的 $m \times 1$ 向量 $y_t$。

一个经济理论通常使得 $w_{t+1}$ 和 $v_t$ 可以直接被解释为对偏好、技术、禀赋或信息集的冲击。

状态空间系统 {eq}`eq:statespace` 用这些**可解释的冲击**来表示 $\{y_t\}$。

然而，在通常情形下，即使 $A, G, C, R$ 已知，这些冲击也*不能*直接从 $y_t$ 中恢复出来。

新息表示 {eq}`eq:innovti` 用 $m \times 1$ 新息向量 $a_t$ 来表示*同一个*随机过程 $\{y_t\}$，这些新息可以通过运行无限阶总体向量自回归来恢复。

它在将原始表示 {eq}`eq:statespace` 映射到 VAR {eq}`eq:var1` 中的作用，使得卡尔曼滤波器成为*解释向量自回归*不可或缺的工具。

```{index} single: Kalman Filter; and vector autoregressions
```

## 谱分解恒等式

```{index} single: spectral factorization identity
```

因为原始状态空间系统 {eq}`eq:statespace` 和新息表示 {eq}`eq:innovti` 描述的是*同一个*随机过程 $\{y_t\}$，它们对 $\{y_t\}$ 的**谱密度矩阵**给出了两个不同的公式。

令这两个公式相等就得到*谱分解恒等式*。

### 谱密度的两种表示

首先考虑原始状态空间系统。

将 {eq}`eq:statespace` 的第一行写为
$x_t = (zI - A)^{-1} C w_{t+1}$（使用 $z$ 变换约定
$z^{-1} x_t = x_{t-1}$），$\{x_t\}$ 的协方差生成函数为

$$
S_x(z) = (zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1}.
$$

由于 $v_t$ 与 $x_t$ 正交，$\{y_t\}$ 的谱密度为

$$
S_y(z) = G(zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1} G^\top + R.
$$ (eq:sf_original)

现在考虑新息表示。

时不变新息表示 {eq}`eq:innovti` 给出
$y_t = [G(zI - A)^{-1}K + I]\, a_t$。

由于 $a_t$ 是协方差矩阵为 $G\Sigma G^\top + R$ 的白噪声，谱密度也为

$$
S_y(z) = \bigl[G(zI-A)^{-1}K + I\bigr]
          \bigl(G\Sigma G^\top + R\bigr)
          \bigl[K^\top(z^{-1}I - A^\top)^{-1}G^\top + I\bigr].
$$ (eq:sf_innov)

### 谱分解恒等式

令 {eq}`eq:sf_original` 和 {eq}`eq:sf_innov` 相等，得到**谱分解恒等式**：

$$
G(zI - A)^{-1} CC^\top (z^{-1}I - A^\top)^{-1} G^\top + R =
\bigl[G(zI-A)^{-1}K + I\bigr]
\bigl(G\Sigma G^\top + R\bigr)
\bigl[K^\top(z^{-1}I - A^\top)^{-1}G^\top + I\bigr].
$$ (eq:sf_identity)

左侧用**结构性冲击** $(w_{t+1}, v_t)$ 和矩阵 $(A, C, G, R)$ 表示 $S_y(z)$。

右侧将同一对象表示为由新息 $a_t$ 和稳态卡尔曼增益 $K$ 构建的谱因子。

### Wold 表示和自回归表示

从新息表示 {eq}`eq:innovti` 出发，我们既可以得到 Wold 移动平均表示，也可以得到自回归表示。

对于 Wold 表示，迭代 {eq}`eq:innovti` 中的状态方程，用当前和过去的新息表示当前观测。

用 $L$ 表示滞后算子，{eq}`eq:innovti` 意味着

$$
y_t = \left[I + G(I - AL)^{-1} K L\right] a_t
$$ (eq:sf_wold)

这是用 $\{y_t\}$ 的一步预测误差表示的 Wold 移动平均表示。

对于自回归表示，将 {eq}`eq:sf_wold` 中的移动平均算子求逆，并用当前和过去的观测来求解 $a_t$。

利用恒等式

$$
\left[I + G(I - AL)^{-1} K L\right]^{-1}
    = I - G\left[I - (A - KG)L\right]^{-1} K L
$$

得到

$$
y_t = G\bigl[I-(A-KG)L\bigr]^{-1}K\, y_{t-1} + a_t
    = \sum_{j=1}^\infty G(A-KG)^{j-1}K\, y_{t-j} + a_t,
$$ (eq:sf_var)

这就是已在 {eq}`eq:var1` 中陈述的向量自回归。

关键的分析事实是：在温和的稳定性条件下，$\det[G(zI-A)^{-1}K + I]$ 的零点全部位于单位圆*内部*。

这确保了 {eq}`eq:sf_wold` 中的移动平均算子有一个因果的单边逆。

因此 $a_t$ 位于当前和过去观测 $y^t$ 的闭线性张成空间中，所以 $a_t$ 是 VAR 中的总体预测误差。

## Python 实现

我们现在使用 `quantecon` 库来说明该理论，它提供了 `LinearStateSpace` 和 `Kalman` 类，实现了上面推导的所有内容。

我们使用以下导入：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
mpl.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
```

### 一个标量隐藏 AR(1) 模型

考虑一个带有测量噪声观测的标量隐藏 AR(1) 状态：

$$
\begin{aligned}
x_{t+1} &= \rho\, x_t + \sigma_w\, w_{t+1} \\
y_t      &= x_t + \sigma_v\, v_t
\end{aligned}
$$

其中 $w_t, v_t \sim N(0, 1)$ 独立同分布。

这里 $\rho$ 是持续性参数，而 $\sigma_w$ 和 $\sigma_v$ 是冲击标准差。

`LinearStateSpace` 类通过矩阵 $H$ 来参数化测量噪声，使得 $R = HH^\top$。

```{code-cell} ipython3
# 模型参数
ρ = 0.9
σ_w = 0.5
σ_v = 1.0

# 状态空间矩阵
A = np.array([[ρ]])
C = np.array([[σ_w]])
G = np.array([[1.0]])
R = np.array([[σ_v**2]])

# 构建一个 LinearStateSpace 和一个 Kalman 滤波器对象
H = np.array([[σ_v]])   # 测量噪声因子: R = H @ H.T
lss = qe.LinearStateSpace(
  A, C, G, H, mu_0=np.zeros(1), Sigma_0=np.eye(1) * 10.0)
kf = qe.Kalman(lss)
kf.set_state(np.zeros(1), np.eye(1) * 10.0)  # 弥散先验
```

我们首先模拟一条真实隐藏状态和带噪声观测的样本路径。

```{code-cell} ipython3
T = 200
x_path, y_path = lss.simulate(ts_length=T, random_state=42)

# 形状: x_path 是 (n, T+1), y_path 是 (m, T)
x_true = x_path[0, :T]
y_obs = y_path[0, :]
```

然后我们手动逐步运行卡尔曼滤波器，以收集滤波后的估计值。

```{code-cell} ipython3
x_hats = np.zeros(T)
Sigmas = np.zeros(T)

for t in range(T):
    kf.update(y_obs[t:t+1])          # 一个完整的滤波周期
    x_hats[t] = kf.x_hat.item()
    Sigmas[t] = kf.Sigma.item()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 标量卡尔曼滤波
    name: fig-kfvar-scalar
---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

t_range = np.arange(T)

axes[0].plot(t_range, x_true, lw=2,
        label='真实状态 $x_t$')
axes[0].plot(t_range, x_hats, lw=2,
        linestyle='--', label=r'$\hat{x}_t$ (卡尔曼)')
axes[0].plot(t_range, y_obs, alpha=0.35, lw=2, label='观测 $y_t$')
axes[0].set_title('状态与观测')
axes[0].legend(fontsize=9)

axes[1].plot(t_range, Sigmas, color='C1', lw=2,
             label=r'条件方差 $\Sigma_t$')
axes[1].axhline(kf.Sigma_infinity[0, 0], ls='--', color='k',
                label=r'稳态 $\Sigma_\infty$')
axes[1].set_title('条件方差')
axes[1].legend(fontsize=9)

axes[2].plot(t_range, y_obs - x_hats, color='C2', lw=2, alpha=0.7,
             label=r'新息 $a_t = y_t - G\hat{x}_t$')
axes[2].set_title('新息')
axes[2].set_xlabel('时间 $t$')
axes[2].legend(fontsize=9)
fig.tight_layout()
plt.show()
```

经过短暂的调整后，卡尔曼估计比原始的带噪声观测更紧密地跟随隐藏状态。

条件方差从弥散先验迅速下降，然后稳定到其稳态值。

新息序列围绕零波动，正如一步预测误差所应有的那样。

### 黎卡提方程的收敛

`Kalman` 类通过直接求解离散代数黎卡提方程来计算稳态协方差 $\Sigma_\infty$。

```{code-cell} ipython3
Sigma_inf, K_inf = kf.stationary_values()

print(f"Steady-state covariance  Σ_inf = {Sigma_inf[0, 0]:.6f}")
print(f"Kalman filter converged to Σ_t = {Sigmas[-1]:.6f}")
print(f"Steady-state Kalman gain K  = {K_inf[0, 0]:.6f}")

A_minus_KG = A - K_inf @ G
eigval = np.linalg.eigvals(A_minus_KG)[0]
print(f"\nEigenvalue of (A - KG)      = {eigval:.6f}")
print(f"Stable VAR: {np.abs(eigval) < 1}")
```

### VAR 表示

利用 {eq}`eq:var1`，无限阶 VAR 表示中的系数为
$G(A - KG)^j K$，其中 $j = 0, 1, 2, \ldots$

我们通过 `stationary_coefficients` 来获取它们：

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 来自新息表示的 VAR 系数
    name: fig-kfvar-varcoef
---
J = 30
var_coeffs = kf.stationary_coefficients(J, coeff_type='var')

# 滞后 j+1 的系数矩阵
lags = np.arange(1, J + 1)
coeff_values = np.array([var_coeffs[j][0, 0] for j in range(J)])

fig, ax = plt.subplots()
ax.stem(lags, coeff_values, basefmt=' ')
ax.set_xlabel('滞后 $j$')
ax.set_ylabel(r'VAR 系数 $G(A{-}KG)^{j-1}K$')
fig.tight_layout()
plt.show()
```

大部分自回归权重集中在前几个滞后上。

后面的系数几乎为零，因此在这个例子中，一个短的有限滞后 VAR 就捕获了无限阶表示的大部分。

### 似然评估

我们使用 {eq}`eq:gauss100` 来计算模拟样本的对数似然。

```{code-cell} ipython3
def log_likelihood(A, C, G, R, y_data, x_hat_0, Sigma_0):
    """使用卡尔曼滤波器递归计算对数似然。"""
    H_ = np.linalg.cholesky(R)   # R = H_ @ H_.T
    lss_ = qe.LinearStateSpace(A, C, G, H_, mu_0=x_hat_0, Sigma_0=Sigma_0)
    kf_ = qe.Kalman(lss_)
    kf_.set_state(x_hat_0, Sigma_0)

    T_, m_ = y_data.shape
    loglik = 0.0

    for t in range(T_):
        x_h = kf_.x_hat
        Sig = kf_.Sigma
        Omega = G @ Sig @ G.T + R        # 新息协方差
        a_t = y_data[t] - (G @ x_h).flatten()

        sign, logdet = np.linalg.slogdet(Omega)
        loglik += -0.5 * (m_ * np.log(2 * np.pi) + logdet
                          + float(a_t @ np.linalg.solve(Omega, a_t)))
        kf_.update(y_data[t])

    return loglik


y_data_col = y_obs.reshape(-1, 1)
ll = log_likelihood(A, C, G, R,
                    y_data_col,
                    np.zeros(1), np.eye(1) * 10.0)
print(f"Log-likelihood of sample: {ll:.4f}")
```

## 一个例子

我们现在通过一个结构化的例子来说明一个二元 VAR(2) 如何自然地嵌入状态空间框架，以及卡尔曼滤波器如何给出一个 Wold（新息）表示。

### 一个线性状态空间系统及其滤波器

状态方程和观测方程为

$$
x_{t+1} = A x_t + C w_{t+1}
$$ (eq:ex_state)

$$
y_t = G x_t + v_t
$$ (eq:ex_obs)

具有初始条件和冲击分布

$$
x_0 \sim N(\hat{x}_0, \Sigma_0), \quad
w_{t+1} \sim N(0, I), \quad
v_t \sim N(0, R).
$$

稳态误差协方差矩阵 $\Sigma$ 满足黎卡提方程

$$
\Sigma = A \Sigma A^\top + CC^\top
         - A \Sigma G^\top \bigl(G \Sigma G^\top + R\bigr)^{-1} G \Sigma A^\top
$$ (eq:ex_riccati)

且相关的稳态卡尔曼增益为

$$
K = A \Sigma G^\top \bigl(G \Sigma G^\top + R\bigr)^{-1}
$$ (eq:ex_gain)

从初始估计 $\hat{x}_0$ 出发，卡尔曼滤波器通过下式更新状态估计

$$
\hat{x}_{t+1} = A \hat{x}_t + K a_t
$$ (eq:ex_kf_update)

其中新息为

$$
a_t = y_t - G \hat{x}_t
$$ (eq:ex_innovation)

将 {eq}`eq:ex_innovation` 代入 {eq}`eq:ex_kf_update` 并展开：

$$
\hat{x}_{t+1} = A \hat{x}_t + K(y_t - G\hat{x}_t)
              = (A - KG)\hat{x}_t + K y_t
              = (A - KG)\hat{x}_t + K G x_t + K v_t
$$ (eq:ex_kf_expanded)

### $y_t$ 对新息 $a_t$ 的脉冲响应

计算可观测向量 $y_t$ 对其自身新息 $a_t$ 的**普通脉冲响应函数**是有用的，这个移动平均（Wold）表示是 VAR {eq}`eq:var1` 的镜像。

从时不变新息表示 {eq}`eq:innovti`

$$
\hat{x}_{t+1} = A\hat{x}_t + K a_t, \qquad y_t = G\hat{x}_t + a_t,
$$

移动平均表示 {eq}`eq:sf_wold` 为

$$
y_t = \bigl[I + G(I - AL)^{-1} K L\bigr]\, a_t
    = a_t + \sum_{h=1}^{\infty} G A^{h-1} K\, a_{t-h}.
$$

因此 $y_t$ 对单位新息 $a_t$ 的脉冲响应为

$$
\Psi_0 = I, \qquad \Psi_h = G A^{h-1} K \quad (h \ge 1).
$$ (eq:ex_y_to_a)

这些系数以由 $A$ 的特征值支配的速率衰减。

我们可以直接从一个 `quantecon` `LinearStateSpace` 对象中读取系数 {eq}`eq:ex_y_to_a`。

我们构建一个状态空间系统，其状态是滤波估计 $\hat{x}_t$，其单一"冲击"是通过 $C = K$ 加载的新息 $a_t$，其观测矩阵为 $G$。

该对象的 `impulse_response` 方法返回序列 $G A^{j} K$，其中 $j = 0, 1, 2, \ldots$，这些正好是 $h \ge 1$ 时的 $\Psi_h$；我们在前面加上 $\Psi_0 = I$ 以捕获当期直达效应 $y_t = G\hat{x}_t + a_t$。

下面返回的数组，其元素 `[h, i, j]` 等于可观测量 `i` 在视界 `h` 处对新息分量 `j` 的响应。

```{code-cell} ipython3
def y_to_a_irf(A, K, G, T=40):
    """
    返回 y_t 对其自身新息 a_t 的 Wold 脉冲响应函数。
    """
    n, m = A.shape[0], G.shape[0]
    lss = qe.LinearStateSpace(A, K, G, np.zeros((m, m)), mu_0=np.zeros(n))
    _, ycoef = lss.impulse_response(j=T - 2)      # [GK, GAK, GA^2K, ...]
    Psi = np.empty((T, m, m))
    Psi[0] = np.eye(m)                          # 当期响应
    for h in range(1, T):
        Psi[h] = ycoef[h - 1]
    return Psi
```

### 状态空间形式的二元 VAR(2)

考虑两个可观测序列 $r_t$ 和 $z_t$。

将它们堆叠成状态向量 $x_t = (r_t,\; r_{t-1},\; z_t,\; z_{t-1})^\top$。

我们假设 VAR(2) 状态转移方程：

$$
\begin{pmatrix} r_{t+1} \\ r_t \\ z_{t+1} \\ z_t \end{pmatrix}
=
\begin{pmatrix}
  d_1      & d_2      & d_3      & d_4      \\
  1        & 0        & 0        & 0        \\
  \delta_1 & \delta_2 & \delta_3 & \delta_4 \\
  0        & 0        & 1        & 0
\end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+
\begin{pmatrix}
  c_{11} & c_{12} \\
  0      & 0      \\
  c_{21} & c_{22} \\
  0      & 0
\end{pmatrix}
\begin{pmatrix} w_{1,t+1} \\ w_{2,t+1} \end{pmatrix}
$$ (eq:ex_var2_state)

我们考虑两种可能的观测方程。

第一种是对 $r_t$ 和 $z_t$ 的二元观测：

$$
\begin{pmatrix} r_t \\ z_t \end{pmatrix}
=
\begin{pmatrix}
  1 & 0 & 0 & 0 \\
  0 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+
\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
\begin{pmatrix} v_{1t} \\ v_{2t} \end{pmatrix}
$$ (eq:ex_var2_obs)

第二种是对 $r_t$ 的单变量观测：

$$
y_t = \begin{pmatrix} 1 & 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} r_t \\ r_{t-1} \\ z_t \\ z_{t-1} \end{pmatrix}
+ v_{1t}
$$ (eq:ex_scalar_obs)

我们现在比较这两个观测系统生成的 Wold 脉冲响应。

系统 1 同时观测 $r_t$ 和 $z_t$，因此其新息 $a_t$ 是 $2 \times 1$ 的。

系统 2 只观测 $r_t$，因此其新息 $u_t$ 是标量。

两个系统的转移矩阵相同，但观测矩阵不同。

因此，稳态卡尔曼增益也不同，可观测量对其自身新息的 Wold 响应也不同。

### 数值例子：对新息的脉冲响应

参数值为：

$$
\begin{aligned}
d_1 &= 0.80,\quad d_2 = 0.05,\quad d_3 = 0.75,\quad d_4 = -0.72 \\
\delta_1 &= 0.00,\quad \delta_2 = 0.00,\quad \delta_3 = 0.75,\quad \delta_4 = 0.20 \\
c_{11} &= 1.0,\quad c_{12} = 0.0,\quad c_{21} = 0.0,\quad c_{22} = 1.0 \\
R &= 0.0001 \times I_2 \quad \text{(二元情形)}, \qquad
R = 0.0001 \quad \text{(单变量情形)}.
\end{aligned}
$$

这些给出 $4 \times 4$ 转移矩阵和 $4 \times 2$ 冲击加载矩阵

$$
A = \begin{pmatrix}
0.80 & 0.05 & 0.75 & -0.72 \\
1    & 0    & 0    & 0    \\
0    & 0    & 0.75 & 0.20 \\
0    & 0    & 1    & 0
\end{pmatrix}, \qquad
C = \begin{pmatrix}
1   & 0   \\
0   & 0   \\
0   & 1   \\
0   & 0
\end{pmatrix}.
$$

**系统 1** 使用二元观测方程 {eq}`eq:ex_var2_obs`，因此
$G$ 从状态中选取 $(r_t, z_t)^\top$，新息 $a_t$ 是 $2 \times 1$ 的。

**系统 2** 使用单变量观测方程 {eq}`eq:ex_scalar_obs`，因此该方程中的行向量只选取 $r_t$，新息 $u_t$ 是标量。

```{code-cell} ipython3
# 参数
d1, d2, d3, d4 = 0.80, 0.05, 0.75, -.72
δ1, δ2, δ3, δ4 = 0.00, 0.00, 0.75, 0.20
c11, c12, c21, c22 = 1.0,  0.0,  0.0,  1.0
σ_v = 0.01  # sqrt(0.0001)

# 共享矩阵
A_var = np.array([[d1,     d2,     d3,     d4    ],
                  [1.0,    0.0,    0.0,    0.0   ],
                  [δ1, δ2, δ3, δ4],
                  [0.0,    0.0,    1.0,    0.0   ]])

C_var = np.array([[c11, c12],
                  [0.0, 0.0],
                  [c21, c22],
                  [0.0, 0.0]])

# 系统 1: 二元观测
G_biv = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
H_biv = σ_v * np.eye(2)          # H @ H.T = 0.0001 * I_2

lss_biv = qe.LinearStateSpace(A_var, C_var, G_biv, H_biv,
                               mu_0=np.zeros(4), Sigma_0=np.eye(4))
kf_biv = qe.Kalman(lss_biv)
_, K_biv = kf_biv.stationary_values()

print("System 1 - steady-state Kalman gain K (4x2):")
print(np.round(K_biv, 5))

# 系统 2: 单变量观测
G_uni = np.array([[1.0, 0.0, 0.0, 0.0]])
H_uni = np.array([[σ_v]])         # H @ H.T = 0.0001

lss_uni = qe.LinearStateSpace(A_var, C_var, G_uni, H_uni,
                               mu_0=np.zeros(4), Sigma_0=np.eye(4))
kf_uni = qe.Kalman(lss_uni)
_, K_uni = kf_uni.stationary_values()

print("\nSystem 2 - steady-state Kalman gain K (4x1):")
print(np.round(K_uni, 5))
```

我们现在应用上面定义的辅助函数 `y_to_a_irf` 来计算可观测量 $y_t$ 对其自身新息 $a_t$ 的普通脉冲响应 {eq}`eq:ex_y_to_a`，分别针对系统 1（二元，因此 $a_t$ 是 $2 \times 1$ 的）和系统 2（单变量，因此 $u_t$ 是标量）。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 系统 1 对自身新息的响应
    name: fig-kfvar-sys1-ya
---
T_irf = 40
horizons = np.arange(T_irf)

Psi_biv = y_to_a_irf(A_var, K_biv, G_biv, T_irf)   # 系统 1: (T, 2, 2)
Psi_uni = y_to_a_irf(A_var, K_uni, G_uni, T_irf)   # 系统 2: (T, 1, 1)

obs_labels = [r'$r_t$', r'$z_t$']
innov_labels = [r'$a_{1,t}$', r'$a_{2,t}$']

# 系统 1 的响应
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for i, obs in enumerate(obs_labels):
    for j, inn in enumerate(innov_labels):
        ax = axes[i, j]
        ax.plot(horizons, Psi_biv[:, i, j], lw=2)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title(fr'{obs} 对 {inn}', fontsize=9)
        if i == 1:
            ax.set_xlabel('视界 $h$')
        if j == 0:
            ax.set_ylabel('响应')
fig.tight_layout()
plt.show()
```

自身新息在冲击时刻对其自身可观测量产生一对一的影响，然后逐渐消退。

对角面板从 1 开始，非对角面板从 0 开始，因为 $\Psi_0 = I$。

$r_t$ 对 $a_{2,t}$ 的交叉响应在短视界上是相当大的，而 $z_t$ 对 $a_{1,t}$ 的响应在显示的尺度上非常微小。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 系统 2 对自身新息的响应
    name: fig-kfvar-sys2-ya
---
fig, ax = plt.subplots()
ax.plot(horizons, Psi_uni[:, 0, 0], lw=2)
ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel('视界 $h$')
ax.set_ylabel('响应')
fig.tight_layout()
plt.show()
```

只观测 $r_t$ 时，单一新息在冲击时刻对 $r_t$ 产生一对一的影响，然后单调地衰减到零。

对于 $h \ge 1$，响应通过状态矩阵 $A$ 传播并以几何速率衰减，勾勒出二元（系统 1）和单变量（系统 2）过程的 Wold 移动平均表示。

这些是来自 Wold 表示的预测误差响应，不是结构性冲击响应。

有了这个例子，我们现在可以把本讲座的主要要点汇总起来。

## 总结

卡尔曼滤波器通过将先验预测与当前观测中包含的新信息相结合，递归地更新关于隐藏状态的信念。

黎卡提方程跟踪滤波器的条件协方差如何演化，其稳态使滤波器成为时不变的。

在该稳态处，新息表示用由构造而成的白噪声一步预测误差来表示可观测过程。

向后求解新息表示得到一个无限阶 VAR，而向前求解则得到 Wold 移动平均表示。

数值例子表明，改变观测变量会改变卡尔曼增益，从而改变 Wold 响应，即使底层状态动态相同也是如此。

## 练习

```{exercise-start}
:label: kf_ex1
```

考虑上面使用的标量 AR(1) 状态空间系统，其中 $\rho = 0.9$，
$\sigma_w = 0.5$，$\sigma_v = 1.0$。

通过在其不动点 $\Sigma_{t+1} = \Sigma_t = \Sigma$ 处求解标量黎卡提方程 {eq}`eq:riccati`，为**稳态**条件方差 $\Sigma_\infty$ 推导一个代数表达式。

证明 $\Sigma$ 满足一个二次方程，求出其正根，并数值验证你的公式与 `kf.Sigma_infinity` 相符。

```{exercise-end}
```

```{solution-start} kf_ex1
:class: dropdown
```

这是一个解答：

在 {eq}`eq:riccati` 的标量版本中设 $\Sigma_{t+1} = \Sigma_t = \Sigma$，其中 $A = \rho$，$CC^\top = \sigma_w^2$，$GG^\top = 1$，
$R = \sigma_v^2$：

$$
\Sigma = \rho^2 \Sigma + \sigma_w^2 - \frac{\rho^2 \Sigma^2}{\Sigma + \sigma_v^2}
$$

两边同乘 $\Sigma + \sigma_v^2$ 并重新整理：

$$
\Sigma^2 + \left[\sigma_v^2(1-\rho^2) - \sigma_w^2\right]\Sigma
  - \sigma_v^2 \sigma_w^2 = 0
$$

取此二次方程的正根：

$$
\Sigma_\infty
  = \frac{\sigma_w^2 - \sigma_v^2(1-\rho^2)
          + \sqrt{\left[\sigma_v^2(1-\rho^2) - \sigma_w^2\right]^2
          + 4 \sigma_v^2 \sigma_w^2}}{2}
$$

```{code-cell} ipython3
ρ_, σ_w_, σ_v_ = 0.9, 0.5, 1.0

b = σ_v_**2 * (1 - ρ_**2) - σ_w_**2
discriminant = b**2 + 4 * σ_v_**2 * σ_w_**2
Sigma_formula = (-b + np.sqrt(discriminant)) / 2

A_ = np.array([[ρ_]])
C_ = np.array([[σ_w_]])
G_ = np.array([[1.0]])
R_ = np.array([[σ_v_**2]])
H_ = np.array([[σ_v_]])   # R_ = H_ @ H_.T
lss_ = qe.LinearStateSpace(A_, C_, G_, H_, mu_0=np.zeros(1), Sigma_0=np.eye(1))
kf_ = qe.Kalman(lss_)

print(f"Analytical Σ_inf   = {Sigma_formula:.8f}")
print(f"Numerical  Σ_inf   = {kf_.Sigma_infinity[0, 0]:.8f}")
```

```{solution-end}
```

```{exercise-start}
:label: kf_ex2
```

本练习考虑一个二维状态和一维观测：

$$
A = \begin{pmatrix} 0.9 & 0.1 \\ 0 & 0.8 \end{pmatrix}, \quad
C = \begin{pmatrix} 0.4 \\ 0.1 \end{pmatrix}, \quad
G = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
R = [0.5]
$$

1. 从弥散先验出发，从这个系统模拟 $T = 500$ 个观测。

2. 运行卡尔曼滤波器，并绘制 $\hat{x}_t$ 的两个分量与真实隐藏状态路径的对比。

3. 计算并报告稳态协方差 $\Sigma_\infty$ 和卡尔曼增益 $K_\infty$。

4. 检查 $A - K_\infty G$ 的特征值是否严格位于单位圆内，从而确认 VAR 表示 {eq}`eq:var1` 是稳定的。

```{exercise-end}
```

```{solution-start} kf_ex2
:class: dropdown
```

这是一个解答：

```{code-cell} ipython3
A2 = np.array([[0.9, 0.1],
               [0.0, 0.8]])
C2 = np.array([[0.4],
               [0.1]])
G2 = np.array([[1.0, 0.0]])
R2 = np.array([[0.5]])
H2 = np.array([[np.sqrt(0.5)]])   # R2 = H2 @ H2.T

lss2 = qe.LinearStateSpace(A2, C2, G2, H2,
                             mu_0=np.zeros(2),
                             Sigma_0=np.eye(2) * 5.0)
kf2 = qe.Kalman(lss2)
kf2.set_state(np.zeros(2), np.eye(2) * 5.0)

T2 = 500
x2_path, y2_path = lss2.simulate(ts_length=T2, random_state=0)

x_hats2 = np.zeros((T2, 2))
for t in range(T2):
    kf2.update(y2_path[:, t])
    x_hats2[t] = kf2.x_hat.ravel()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(x2_path[i, :T2], lw=2, label=f'真实 $x_{{{i+1},t}}$')
    ax.plot(x_hats2[:, i], lw=2, ls='--', label=rf'$\hat{{x}}_{{{i+1},t}}$')
    ax.set_title(f'分量 {i+1}')
    ax.legend(fontsize=9)
    ax.set_ylabel(f'分量 {i+1}')
axes[1].set_xlabel('时间 $t$')
fig.suptitle('卡尔曼滤波器: 二元隐藏状态')
fig.tight_layout()
plt.show()

# 稳态值
Sigma2_inf, K2_inf = kf2.stationary_values()
print("Steady-state covariance Σ_inf:")
print(np.round(Sigma2_inf, 5))
print("\nSteady-state Kalman gain K_inf:")
print(np.round(K2_inf, 5))

# A - K_inf G 的特征值
AKG2 = A2 - K2_inf @ G2
eigvals2 = np.linalg.eigvals(AKG2)
print(f"\nEigenvalues of A - K_inf G: {np.round(eigvals2, 5)}")
print(f"Stable VAR: {np.all(np.abs(eigvals2) < 1)}")
```

在弥散先验的暂态之后，两个卡尔曼估计都紧密地跟随其对应的隐藏状态路径。

第二个分量没有被直接观测，因此它的跟踪来自于状态动态及其与观测信号的联系。

打印出的特征值随后检查了 VAR 表示的独立稳定性条件。

```{solution-end}
```

```{exercise-start}
:label: kf_ex3
```

本练习使用正文中的标量模型研究似然和参数估计，真实参数为
$(\rho, \sigma_w, \sigma_v) = (0.9, 0.5, 1.0)$：

1. 模拟 $T = 300$ 个观测。

2. 编写一个函数，将**对数似然**作为 $\rho \in (0, 1)$ 的函数进行计算，保持 $\sigma_w = 0.5$ 和 $\sigma_v = 1.0$ 固定，并针对一组网格值绘制对数似然关于 $\rho$ 的图。

3. 数值定位最大值，并检查它是否接近真实值 $\rho = 0.9$。

```{exercise-end}
```

```{solution-start} kf_ex3
:class: dropdown
```

这是一个解答：

```{code-cell} ipython3
# 真实参数
ρ_true, sw_true, sv_true = 0.9, 0.5, 1.0

A_t = np.array([[ρ_true]])
C_t = np.array([[sw_true]])
G_t = np.array([[1.0]])
R_t = np.array([[sv_true**2]])
H_t = np.array([[sv_true]])   # R_t = H_t @ H_t.T

lss_t = qe.LinearStateSpace(A_t, C_t, G_t, H_t,
                             mu_0=np.zeros(1), Sigma_0=np.eye(1))
_, y_sim = lss_t.simulate(ts_length=300, random_state=7)
y_sim = y_sim.T          # 形状 (300, 1)

def ll_rho(ρ_val):
    A_ = np.array([[ρ_val]])
    C_ = np.array([[sw_true]])
    G_ = np.array([[1.0]])
    R_ = np.array([[sv_true**2]])
    return log_likelihood(A_, C_, G_, R_, y_sim,
                          np.zeros(1), np.eye(1) * 10.0)

ρ_grid = np.linspace(0.5, 0.99, 60)
ll_vals = np.array([ll_rho(r) for r in ρ_grid])

ρ_mle = ρ_grid[np.argmax(ll_vals)]

fig, ax = plt.subplots()
ax.plot(ρ_grid, ll_vals, lw=2)
ax.axvline(ρ_true, color='k',   ls='--', label=f'真实 ρ = {ρ_true}')
ax.axvline(ρ_mle, color='C1', ls=':', label=f'MLE  $\\hat{{\\rho}}$ = {ρ_mle:.3f}')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('对数似然')
ax.set_title('作为 $\\rho$ 函数的剖面对数似然')
ax.legend()
fig.tight_layout()
plt.show()

print(f"True ρ = {ρ_true},  MLE ρ_hat = {ρ_mle:.4f}")
```

似然曲线是单峰的，其最大值接近真实值 $\rho = 0.9$。

网格最大化器与真实值之间的微小差距来自于有限样本的随机性和离散网格的使用。

```{solution-end}
```