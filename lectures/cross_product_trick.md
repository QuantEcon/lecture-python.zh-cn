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
---

# 消除交叉项

## 概述

本讲座描述了消除以下内容的公式：

  * 线性二次动态规划问题中状态和控制之间的交叉项
  
  * 卡尔曼滤波问题中状态噪声和测量噪声之间的协方差

对于线性二次动态规划问题，主要思路包括以下步骤：

 * 对状态和控制进行变换，得到一个等价问题，其中转换后的状态和控制之间没有交叉项
 * 使用本讲座 {doc}`线性控制：基础 <lqcontrol>` 中介绍的标准公式求解转换后的问题
 * 将转换后问题的最优决策规则转换回原始问题（即有状态和控制交叉项的问题）的最优决策规则

+++

## 无折现动态规划问题

这里是一个目标函数中包含状态和控制交叉项的非随机无折现线性二次动态规划问题。

该问题由矩阵5元组 $(A, B, R, Q, H)$ 定义，其中 $R$ 和 $Q$ 是正定对称矩阵，且
$A \sim m \times m, B \sim m \times k, Q \sim k \times k, R \sim m \times m$ 以及 $H \sim k \times m$。

问题是选择 $\{x_{t+1}, u_t\}_{t=0}^\infty$ 以最大化

$$
 - \sum_{t=0}^\infty (x_t' R x_t + u_t' Q u_t + 2 u_t H x_t) 
$$

受限于线性约束

$$ x_{t+1} = A x_t + B u_t,  \quad t \geq 0 $$

其中 $x_0$ 是给定的初始条件。

这个无限期无折现问题的解是一个时不变的反馈规则

$$ u_t  = -F x_t $$

其中

$$ F = -(Q + B'PB)^{-1} B'PA $$

且 $P \sim m \times m $ 是代数矩阵Riccati方程的正定解

$$
P = R + A'PA - (A'PB + H')(Q + B'PB)^{-1}(B'PA + H).
$$

+++

可以验证，一个**等价的**没有状态和控制交叉项的问题可以由矩阵4元组定义：$(A^*, B, R^*, Q)$。

省略的矩阵 $H=0$ 表示在等价问题中没有状态和控制之间的交叉项。

定义等价问题的矩阵 $(A^*, B, R^*, Q)$ 及其值函数、策略函数矩阵 $P, F^*$ 与定义原始问题的矩阵 $(A, B, R, Q, H)$ 及其值函数、策略函数矩阵 $P, F$ 之间的关系如下：

\begin{align*}
A^* & = A - B Q^{-1} H, \\
R^* & = R - H'Q^{-1} H, \\
P & = R^* + {A^*}' P A - ({A^*}' P B) (Q + B' P B)^{-1} B' P A^*, \\
F^* & = (Q + B' P B)^{-1} B' P A^*, \\
F & = F^* + Q^{-1} H.
\end{align*}

+++

## 卡尔曼滤波

线性二次最优控制和卡尔曼滤波问题之间存在的**对偶性**意味着存在一个类似的变换，允许我们将状态噪声和测量噪声之间具有非零协方差矩阵的卡尔曼滤波问题转换为一个等价的、状态噪声和测量噪声之间协方差为零的卡尔曼滤波问题。

让我们看看适当的变换。

首先，让我们回顾一下具有状态噪声和测量噪声之间协方差的卡尔曼滤波。

隐马尔可夫模型为：

\begin{align*}
x_{t+1} & = A x_t + B w_{t+1},  \\
z_{t+1} & = D x_t + F w_{t+1},  
\end{align*}

其中 $A \sim m \times m, B \sim m \times p $ 且 $D \sim k \times m, F \sim k \times p $，
且 $w_{t+1}$ 是一个独立同分布的 $p \times 1$ 正态分布随机向量序列的时间 $t+1$ 分量，其均值向量为零，协方差矩阵等于 $p \times p$ 单位矩阵。

因此，$x_t$ 是 $m \times 1$ 且 $z_t$ 是 $k \times 1$。

卡尔曼滤波公式为：

\begin{align*}
K(\Sigma_t) & = (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1}, \\
\Sigma_{t+1}&  = A \Sigma_t A' + BB' - (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1} (D \Sigma_t A' + FB').
\end{align*} (eq:Kalman102)

定义转换后的矩阵：

\begin{align*}
A^* & = A - BF' (FF')^{-1} D, \\
B^* {B^*}' & = BB' - BF' (FF')^{-1} FB'.
\end{align*}

### 算法

公式 {eq}`eq:Kalman102` 的一个结果是，我们可以使用以下算法来求解涉及状态噪声和信号噪声之间非零协方差的卡尔曼滤波问题。

首先，使用普通卡尔曼滤波公式计算 $\Sigma, K^*$，其中 $BF' = 0$，即状态随机噪声和测量随机噪声之间的协方差矩阵为零。

也就是说，计算满足以下条件的 $K^*$ 和 $\Sigma$：

\begin{align*}
K^* & = (A^* \Sigma D')(D \Sigma D' + FF')^{-1} \\
\Sigma & = A^* \Sigma {A^*}' + B^* {B^*}' - (A^* \Sigma D')(D \Sigma D' + FF')^{-1} (D \Sigma {A^*}').
\end{align*}

原始问题（具有**非零协方差**的状态和测量噪声）的卡尔曼增益为：

$$
K = K^* + BF' (FF')^{-1},
$$

原始问题的状态重构协方差矩阵 $\Sigma$ 等于转换后问题的状态重构协方差矩阵。

+++

## 对偶表

这是一个便于记忆卡尔曼滤波和动态规划关系的表格。

| 动态规划 | 卡尔曼滤波 |
| :-------------: | :-----------: |
|       $A$       |     $A'$      |
|       $B$       |     $D'$      |
|       $H$       |     $FB'$     |
|       $Q$       |     $FF'$     |
|       $R$       |     $BB'$     |
|       $F$       |     $K'$      |
|       $P$       |   $\Sigma$    |

+++


```{code-cell} ipython3

```
