---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(kalman)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 初探卡尔曼滤波器

```{index} single: 卡尔曼滤波器
```

```{contents} 目录
:depth: 2
```

除了 Anaconda 中的内容，本讲座还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```
## 概述

本讲座为卡尔曼滤波器提供了一个简单直观的介绍，适合那些

* 听说过卡尔曼滤波器但不知道其工作原理的人，或
* 知道卡尔曼滤波器方程，但不知道其来源的人

有关卡尔曼滤波器的更多（更高级）阅读，请参见

* {cite}`Ljungqvist2012`, 第2.7节
* {cite}`AndersonMoore2005`

第二个参考文献对卡尔曼滤波器进行了全面的处理。

所需知识：熟悉矩阵操作、多元正态分布、协方差矩阵等。

我们需要以下导入：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
from scipy import linalg
import numpy as np
import matplotlib.cm as cm
from quantecon import Kalman, LinearStateSpace
from scipy.stats import norm
from scipy.integrate import quad
from scipy.linalg import eigvals
```
## 基本概念

卡尔曼滤波器在经济学中有许多应用，但现在让我们假装自己是火箭科学家。

一枚导弹从国家 Y 发射，我们的任务是追踪它。

令 $x \in \mathbb{R}^2$ 表示导弹的当前位置——地图上的一对纬度-经度坐标。

在当前时刻，精确位置 $x$ 是未知的，但我们对 $x$ 有一些信念。

总结我们知识的一种方法是点预测 $\hat x$。

* 但如果总统想知道导弹目前在日本海上空的概率呢？
* 那么用双变量概率密度 $p$ 来总结我们的初始信念会更好
  * $\int_E p(x)dx$ 表示我们认为导弹在区域 $E$ 的概率。

密度 $p$ 被称为随机变量 $x$ 的*先验*。

为了使我们的例子易于处理，我们假设我们的先验是高斯分布。

特别是，我们取
```{math}
:label: prior

p = N(\hat x, \Sigma)
```

其中 $\hat x$ 是分布的均值，$\Sigma$ 是一个 $2 \times 2$ 的协方差矩阵。在我们的模拟中，我们假设

```{math}
:label: kalman_dhxs

\hat x
= \left(
\begin{array}{c}
    0.2 \\
    -0.2
\end{array}
  \right),
\qquad
\Sigma
= \left(
\begin{array}{cc}
    0.4 & 0.3 \\
    0.3 & 0.45
\end{array}
  \right)
```

这个密度 $p(x)$ 如下图所示为等高线图，红色椭圆的中心等于 $\hat x$。

```{code-cell} python3
---
tags: [output_scroll]
---
# 设置高斯先验密度 p
Σ = [[0.4, 0.3], [0.3, 0.45]]
Σ = np.matrix(Σ)
x_hat = np.matrix([0.2, -0.2]).T
# 从方程 y = G x + N(0, R) 定义矩阵 G 和 R
G = [[1, 0], [0, 1]]
G = np.matrix(G)
R = 0.5 * Σ
# 矩阵 A 和 Q
A = [[1.2, 0], [0, -0.2]]
A = np.matrix(A)
Q = 0.3 * Σ
# 观测值 y
y = np.matrix([2.3, -1.9]).T

# 设置绘图网格
x_grid = np.linspace(-1.5, 2.9, 100)
y_grid = np.linspace(-3.1, 1.7, 100)
X, Y = np.meshgrid(x_grid, y_grid)

def bivariate_normal(x, y, σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0):
    """
    计算并返回双变量正态分布的概率密度函数

    参数
    ----------
    x : array_like(float)
        随机变量

    y : array_like(float)
        随机变量

    σ_x : array_like(float)
          随机变量 x 的标准差

    σ_y : array_like(float)
          随机变量 y 的标准差

    μ_x : scalar(float)
          随机变量 x 的均值

    μ_y : scalar(float)
          随机变量 y 的均值

    σ_xy : array_like(float)
           随机变量 x 和 y 的协方差

    """

    x_μ = x - μ_x
    y_μ = y - μ_y

    ρ = σ_xy / (σ_x * σ_y)
    z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
    denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)
    return np.exp(-z / (2 * (1 - ρ**2))) / denom

def gen_gaussian_plot_vals(μ, C):
    "用于绘制双变量高斯 N(μ, C) 的 Z 值"
    m_x, m_y = float(μ[0]), float(μ[1])
    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    s_xy = C[0, 1]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)

# 绘制图形

fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)

plt.show()
```
### 过滤步骤

我们现在面临一些好消息和一些坏消息。

好消息是导弹已经被我们的传感器定位，报告当前位置为 $y = (2.3, -1.9)$。

下图显示了原始先验 $p(x)$ 和新报告的位置 $y$

```{code-cell} python3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

plt.show()
```
坏消息是我们的传感器不精确。

特别是，我们应该将传感器的输出解释为不是 $y=x$，而是

```{math}
:label: kl_measurement_model

y = G x + v, \quad \text{where} \quad v \sim N(0, R)
```

这里 $G$ 和 $R$ 是 $2 \times 2$ 矩阵，$R$ 是正定的。两者都被假设为已知，并且噪声项 $v$ 被假设为与 $x$ 独立。

那么我们应该如何结合我们的先验 $p(x) = N(\hat x, \Sigma)$ 和这个新信息 $y$ 来改善我们对导弹位置的理解？

正如您可能猜到的，答案是使用贝叶斯定理，它告诉我们通过以下方式更新我们的先验 $p(x)$ 到 $p(x \,|\, y)$：

$$
p(x \,|\, y) = \frac{p(y \,|\, x) \, p(x)} {p(y)}
$$

其中 $p(y) = \int p(y \,|\, x) \, p(x) dx$。

在求解 $p(x \,|\, y)$ 时，我们观察到

* $p(x) = N(\hat x, \Sigma)$。
* 根据 {eq}`kl_measurement_model`，条件密度 $p(y \,|\, x)$ 是 $N(Gx, R)$。
* $p(y)$ 不依赖于 $x$，在计算中仅作为一个归一化常数。

因为我们处于线性和高斯框架中，可以通过计算总体线性回归来计算更新后的密度。

特别地，已知解 [^f1] 为

$$
p(x \,|\, y) = N(\hat x^F, \Sigma^F)
$$

其中

```{math}
:label: kl_filter_exp

\hat x^F := \hat x + \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat x)
\quad \text{和} \quad
\Sigma^F := \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma
```

这里 $\Sigma G' (G \Sigma G' + R)^{-1}$ 是隐藏对象 $x - \hat x$ 对惊讶 $y - G \hat x$ 的总体回归系数矩阵。

这个新的密度 $p(x \,|\, y) = N(\hat x^F, \Sigma^F)$ 在下图中通过等高线和色彩图显示。

原始密度以等高线形式保留以供比较

```{code-cell} python3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)
M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
x_hat_F = x_hat + M * (y - G * x_hat)
Σ_F = Σ - M * G * Σ
new_Z = gen_gaussian_plot_vals(x_hat_F, Σ_F)
cs2 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

plt.show()
```
我们的新密度在由新信息 $y - G \hat x$ 确定的方向上扭曲了先验 $p(x)$。

在生成图形时，我们将 $G$ 设置为单位矩阵，并将 $R = 0.5 \Sigma$，其中 $\Sigma$ 在 {eq}`kalman_dhxs` 中定义。

(kl_forecase_step)=
### 预测步骤

到目前为止我们取得了什么成就？

我们已经获得了关于状态（导弹）当前位置的概率，给定先验和当前信息。

这被称为“滤波”而不是预测，因为我们是在滤除噪声而不是展望未来。

* $p(x \,|\, y) = N(\hat x^F, \Sigma^F)$ 被称为*滤波分布*

但现在让我们假设我们被赋予了另一个任务：预测导弹在经过一个时间单位（无论那是什么）后的位置。

为此，我们需要一个关于状态如何演变的模型。

假设我们有一个，并且它是线性和高斯的。特别是，

```{math}
:label: kl_xdynam
x_{t+1} = A x_t + w_{t+1}, \quad \text{其中} \quad w_t \sim N(0, Q)
```

我们的目标是结合这个运动定律和我们当前的分布 $p(x \,|\, y) = N(\hat x^F, \Sigma^F)$，以得出一个新的*预测*分布，用于预测一个时间单位后的位置。

根据 {eq}`kl_xdynam`，我们所要做的就是引入一个随机向量 $x^F \sim N(\hat x^F, \Sigma^F)$，并求出 $A x^F + w$ 的分布，其中 $w$ 独立于 $x^F$ 且服从分布 $N(0, Q)$。

由于高斯分布的线性组合仍为高斯分布，$A x^F + w$ 是高斯分布。

基本计算和 {eq}`kl_filter_exp` 中的表达式告诉我们

$$
\mathbb{E} [A x^F + w]
= A \mathbb{E} x^F + \mathbb{E} w
= A \hat x^F
= A \hat x + A \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat x)
$$

和

$$
\operatorname{Var} [A x^F + w]
= A \operatorname{Var}[x^F] A' + Q
= A \Sigma^F A' + Q
= A \Sigma A' - A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q
$$
矩阵 $A \Sigma G' (G \Sigma G' + R)^{-1}$ 通常写作 $K_{\Sigma}$，称为*卡尔曼增益*。

* 下标 $\Sigma$ 被添加以提醒我们 $K_{\Sigma}$ 依赖于 $\Sigma$，但不依赖于 $y$ 或 $\hat x$。

使用这种符号，我们可以总结我们的结果如下。

我们更新后的预测是密度 $N(\hat x_{new}, \Sigma_{new})$，其中

```{math}
:label: kl_mlom0

\begin{aligned}
    \hat x_{new} &:= A \hat x + K_{\Sigma} (y - G \hat x) \\
    \Sigma_{new} &:= A \Sigma A' - K_{\Sigma} G \Sigma A' + Q \nonumber
\end{aligned}
```

* 密度 $p_{new}(x) = N(\hat x_{new}, \Sigma_{new})$ 被称为*预测分布*

预测分布是下图中显示的新密度，其中更新使用了参数。

$$
A
= \left(
\begin{array}{cc}
    1.2 & 0.0 \\
    0.0 & -0.2
\end{array}
  \right),
  \qquad
Q = 0.3 * \Sigma
$$

```{code-cell} python3
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

# 密度 1
Z = gen_gaussian_plot_vals(x_hat, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)

# 密度 2
M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
x_hat_F = x_hat + M * (y - G * x_hat)
Σ_F = Σ - M * G * Σ
Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)
cs2 = ax.contour(X, Y, Z_F, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)

# 密度 3
new_x_hat = A * x_hat_F
new_Σ = A * Σ_F * A.T + Q
new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
cs3 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs3, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

plt.show()
```
### 递归过程

```{index} single: 卡尔曼滤波器; 递归过程
```

让我们回顾一下我们所做的事情。

我们以导弹位置 $x$ 的先验 $p(x)$ 开始当前周期。

然后我们使用当前测量值 $y$ 更新到 $p(x \,|\, y)$。

最后，我们使用运动定律 {eq}`kl_xdynam` 对 $\{x_t\}$ 更新到 $p_{new}(x)$。

如果我们现在进入下一个周期，我们准备再次循环，将 $p_{new}(x)$ 作为当前的先验。

将符号 $p_t(x)$ 替换为 $p(x)$，$p_{t+1}(x)$ 替换为 $p_{new}(x)$，完整的递归过程是：

1. 以先验 $p_t(x) = N(\hat x_t, \Sigma_t)$ 开始当前周期。
2. 观察当前测量值 $y_t$。
3. 从 $p_t(x)$ 和 $y_t$ 计算滤波分布 $p_t(x \,|\, y) = N(\hat x_t^F, \Sigma_t^F)$，应用贝叶斯规则和条件分布 {eq}`kl_measurement_model`。
1. 从滤波分布和 {eq}`kl_xdynam` 计算预测分布 $p_{t+1}(x) = N(\hat x_{t+1}, \Sigma_{t+1})$。
1. 将 $t$ 增加一并返回步骤 1。

重复 {eq}`kl_mlom0`，$\hat x_t$ 和 $\Sigma_t$ 的动态如下

```{math}
:label: kalman_lom

\begin{aligned}
    \hat x_{t+1} &= A \hat x_t + K_{\Sigma_t} (y_t - G \hat x_t) \\
    \Sigma_{t+1} &= A \Sigma_t A' - K_{\Sigma_t} G \Sigma_t A' + Q \nonumber
\end{aligned}
```

这些是卡尔曼滤波器的标准动态方程（参见例如 {cite}`Ljungqvist2012`，第 58 页）。

(kalman_convergence)=
## 收敛

矩阵 $\Sigma_t$ 是我们对 $x_t$ 的预测 $\hat x_t$ 不确定性的度量。

除了特殊情况外，这种不确定性永远不会完全解决，无论经过多长时间。

原因之一是我们的预测 $\hat x_t$ 是基于在 $t-1$ 时可用的信息，而不是 $t$。
即使我们知道 $x_{t-1}$ 的精确值（实际上我们并不知道），转移方程 {eq}`kl_xdynam` 表明 $x_t = A x_{t-1} + w_t$。

由于冲击 $w_t$ 在 $t-1$ 时不可观测，任何关于 $x_t$ 的 $t-1$ 时预测都会产生一些误差（除非 $w_t$ 是退化的）。

然而，$\Sigma_t$ 确实有可能在 $t \to \infty$ 时收敛到一个常数矩阵。

为了研究这个主题，让我们展开 {eq}`kalman_lom` 中的第二个方程：

```{math}
:label: kalman_sdy

\Sigma_{t+1} = A \Sigma_t A' -  A \Sigma_t G' (G \Sigma_t G' + R)^{-1} G \Sigma_t A' + Q
```

这是一个关于 $\Sigma_t$ 的非线性差分方程。

{eq}`kalman_sdy` 的一个不动点是一个常数矩阵 $\Sigma$，使得

```{math}
:label: kalman_dare

\Sigma = A \Sigma A' -  A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q
```

方程 {eq}`kalman_sdy` 被称为离散时间 Riccati 差分方程。
方程 {eq}`kalman_dare` 被称为[离散时间代数 Riccati 方程](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation)。

在何种条件下存在一个固定点并且序列 $\{\Sigma_t\}$ 收敛到该点的讨论见于 {cite}`AHMS1996` 和 {cite}`AndersonMoore2005`，第4章。

一个充分（但非必要）的条件是 $A$ 的所有特征值 $\lambda_i$ 满足 $|\lambda_i| < 1$（参见例如 {cite}`AndersonMoore2005`，第77页）。

（这一强条件确保了当 $t \rightarrow + \infty$ 时，$x_t$ 的无条件分布收敛。）

在这种情况下，对于任何既非负又对称的初始选择 $\Sigma_0$，{eq}`kalman_sdy` 中的序列 $\{\Sigma_t\}$ 收敛到一个非负对称矩阵 $\Sigma$，该矩阵解 {eq}`kalman_dare`。

## 实现

```{index} single: 卡尔曼滤波器; 编程实现
```
`Kalman` 类来自 [QuantEcon.py](http://quantecon.org/quantecon-py) 包，实现了卡尔曼滤波器

* 实例数据包括：
  * 当前先验的矩 $(\hat x_t, \Sigma_t)$。
  * [QuantEcon.py](http://quantecon.org/quantecon-py) 中 [LinearStateSpace](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py) 类的一个实例。

后者表示形式为

$$
\begin{aligned}
    x_{t+1} & = A x_t + C w_{t+1}
    \\
    y_t & = G x_t + H v_t
\end{aligned}
$$

的线性状态空间模型，其中冲击 $w_t$ 和 $v_t$ 是独立同分布的标准正态分布。

为了将其与本讲的符号联系起来，我们设置

$$
Q := CC' \quad \text{and} \quad R := HH'
$$

* [QuantEcon.py](http://quantecon.org/quantecon-py) 包中的 `Kalman` 类有许多方法，其中一些我们将在后续讲座中研究更高级的应用时使用。
* 本讲相关的方法有：
* `prior_to_filtered`，将 $(\hat x_t, \Sigma_t)$ 更新为 $(\hat x_t^F, \Sigma_t^F)$
    * `filtered_to_forecast`，将过滤分布更新为预测分布——这成为新的先验 $(\hat x_{t+1}, \Sigma_{t+1})$
    * `update`，结合了最后两种方法
    * 一个 `stationary_values`，计算 {eq}`kalman_dare` 的解和相应的（平稳）卡尔曼增益

您可以在 [GitHub](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/kalman.py) 上查看程序。

## 练习

```{exercise-start}
:label: kalman_ex1
```

考虑卡尔曼滤波器的以下简单应用，松散地基于 {cite}`Ljungqvist2012`，第2.9.2节。

假设

* 所有变量都是标量
* 隐藏状态 $\{x_t\}$ 实际上是常数，等于某个模型未知的 $\theta \in \mathbb{R}$

因此，状态动态由 {eq}`kl_xdynam` 给出，其中 $A=1$，$Q=0$ 且 $x_0 = \theta$。
测量方程是 $y_t = \theta + v_t$，其中 $v_t$ 是 $N(0,1)$ 且独立同分布。

本练习的任务是模拟模型，并使用 `kalman.py` 中的代码，绘制前五个预测密度 $p_t(x) = N(\hat x_t, \Sigma_t)$。

如 {cite}`Ljungqvist2012` 第 2.9.1--2.9.2 节所示，这些分布渐近地将所有质量集中在未知值 $\theta$ 上。

在模拟中，取 $\theta = 10$，$\hat x_0 = 8$ 和 $\Sigma_0 = 1$。

你的图形应该——除去随机性——看起来像这样

```{figure} /_static/lecture_specific/kalman/kl_ex1_fig.png
```

```{exercise-end}
```


```{solution-start} kalman_ex1
:class: dropdown
```

```{code-cell} python3
# 参数
θ = 10  # 状态 x_t 的常数值
A, C, G, H = 1, 0, 1, 1
ss = LinearStateSpace(A, C, G, H, mu_0=θ)

# 设置先验，初始化卡尔曼滤波器
x_hat_0, Σ_0 = 8, 1
kalman = Kalman(ss, x_hat_0, Σ_0)

# 从状态空间模型中绘制 y 的观测值
N = 5
x, y = ss.simulate(N)
y = y.flatten()

# 设置绘图
fig, ax = plt.subplots(figsize=(10,8))
xgrid = np.linspace(θ - 5, θ + 2, 200)

for i in range(N):
    # 记录当前预测的均值和方差
    m, v = [float(z) for z in (kalman.x_hat, kalman.Sigma)]
    # 绘图，更新滤波器
    ax.plot(xgrid, norm.pdf(xgrid, loc=m, scale=np.sqrt(v)), label=f'$t={i}$')
    kalman.update(y[i])

ax.set_title(f'当 $\\theta = {θ:.1f}$ 时的前 {N} 个密度')
ax.legend(loc='upper left')
plt.show()
```
```{solution-end}
```

```{exercise-start}
:label: kalman_ex2
```

上图支持了概率质量收敛到 $\theta$ 的观点。

为了更好地理解这一点，选择一个小的 $\epsilon > 0$ 并计算

$$
z_t := 1 - \int_{\theta - \epsilon}^{\theta + \epsilon} p_t(x) dx
$$

对于 $t = 0, 1, 2, \ldots, T$。

将 $z_t$ 与 $T$ 作图，设定 $\epsilon = 0.1$ 和 $T = 600$。

你的图形应显示误差不规则地下降，如下所示

```{figure} /_static/lecture_specific/kalman/kl_ex2_fig.png
```

```{exercise-end}
```


```{solution-start} kalman_ex2
:class: dropdown
```

```{code-cell} python3
ϵ = 0.1
θ = 10  # 状态 x_t 的常量值
A, C, G, H = 1, 0, 1, 1
ss = LinearStateSpace(A, C, G, H, mu_0=θ)

x_hat_0, Σ_0 = 8, 1
kalman = Kalman(ss, x_hat_0, Σ_0)

T = 600
z = np.empty(T)
x, y = ss.simulate(T)
y = y.flatten()

for t in range(T):
    # 记录当前预测的均值和方差并绘制其密度
    m, v = [float(temp) for temp in (kalman.x_hat, kalman.Sigma)]

    f = lambda x: norm.pdf(x, loc=m, scale=np.sqrt(v))
    integral, error = quad(f, θ - ϵ, θ + ϵ)
    z[t] = 1 - integral

    kalman.update(y[t])

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_ylim(0, 1)
ax.set_xlim(0, T)
ax.plot(range(T), z)
ax.fill_between(range(T), np.zeros(T), z, color="blue", alpha=0.2)
plt.show()
```
```{solution-end}
```

```{exercise-start}
:label: kalman_ex3
```

如上所述 {ref}`above <kalman_convergence>`，如果冲击序列 $\{w_t\}$ 不是退化的，那么通常情况下在时间 $t-1$ 无法无误地预测 $x_t$（即使我们可以观察到 $x_{t-1}$，情况也是如此）。

现在让我们比较卡尔曼滤波器对 $\hat x_t$ 的预测与一个可以观察到 $x_{t-1}$ 的竞争者的预测。

这个竞争者将使用条件期望 $\mathbb E[ x_t \,|\, x_{t-1}]$，在这种情况下是 $A x_{t-1}$。

已知条件期望是在最小化均方误差方面的最佳预测方法。

（更准确地说，关于 $g$ 的 $\mathbb E \, \| x_t - g(x_{t-1}) \|^2$ 的最小化器是 $g^*(x_{t-1}) := \mathbb E[ x_t \,|\, x_{t-1}]$）

因此，我们正在将卡尔曼滤波器与一个拥有更多信息的竞争者进行比较（在能够观察到潜在状态的意义上）并
在最小化平方误差方面表现最佳。

我们的赛马将根据平方误差进行评估。

特别是，你的任务是生成一个图表，将 $\| x_t - A x_{t-1} \|^2$ 和 $\| x_t - \hat x_t \|^2$ 对 $t$ 的观察值绘制在一起，其中 $t = 1, \ldots, 50$。

对于参数，设置 $G = I, R = 0.5 I$ 和 $Q = 0.3 I$，其中 $I$ 是 $2 \times 2$ 的单位矩阵。

设置

$$
A
= \left(
\begin{array}{cc}
    0.5 & 0.4 \\
    0.6 & 0.3
\end{array}
  \right)
$$

为了初始化先验密度，设置

$$
\Sigma_0
= \left(
\begin{array}{cc}
    0.9 & 0.3 \\
    0.3 & 0.9
\end{array}
  \right)
$$

和 $\hat x_0 = (8, 8)$。

最后，设置 $x_0 = (0, 0)$。

你应该得到一个类似于以下的图（随机性除外）

```{figure} /_static/lecture_specific/kalman/kalman_ex3.png
```

观察到，在初始学习期后，卡尔曼滤波器表现得相当好，即使相对于那些在知道潜在状态的情况下进行最佳预测的竞争者。
```{exercise-end}
```

```{solution-start} kalman_ex3
:class: dropdown
```

```{code-cell} python3
# 定义 A, C, G, H
G = np.identity(2)
H = np.sqrt(0.5) * np.identity(2)

A = [[0.5, 0.4],
     [0.6, 0.3]]
C = np.sqrt(0.3) * np.identity(2)

# 设置状态空间模型，初始值 x_0 设为零
ss = LinearStateSpace(A, C, G, H, mu_0 = np.zeros(2))

# 定义先验密度
Σ = [[0.9, 0.3],
     [0.3, 0.9]]
Σ = np.array(Σ)
x_hat = np.array([8, 8])

# 初始化卡尔曼滤波器
kn = Kalman(ss, x_hat, Σ)

# 打印 A 的特征值
print("A 的特征值:")
print(eigvals(A))

# 打印平稳 Σ
S, K = kn.stationary_values()
print("平稳预测误差方差:")
print(S)

# 生成图表
T = 50
x, y = ss.simulate(T)

e1 = np.empty(T-1)
e2 = np.empty(T-1)

for t in range(1, T):
    kn.update(y[:,t])
    e1[t-1] = np.sum((x[:, t] - kn.x_hat.flatten())**2)
    e2[t-1] = np.sum((x[:, t] - A @ x[:, t-1])**2)

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(range(1, T), e1, 'k-', lw=2, alpha=0.6,
        label='卡尔曼滤波误差')
ax.plot(range(1, T), e2, 'g-', lw=2, alpha=0.6,
        label='条件期望误差')
ax.legend()
plt.show()
```
```{solution-end}
```

```{exercise}
:label: kalman_ex4

尝试将 $Q = 0.3 I$ 中的系数 $0.3$ 上下调整。

观察在平稳解 $\Sigma$ 中的对角值（见 {eq}`kalman_dare`）如何随着这个系数的变化而增加和减少。

这意味着在 $x_t$ 的运动定律中更多的随机性会导致预测中的更多（永久性）不确定性。
```

[^f1]: 参见，例如 {cite}`Bishop2006` 的第93页。要从他的表达式转换到上面使用的表达式，你还需要应用 [Woodbury 矩阵恒等式](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)。
