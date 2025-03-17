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

# LQ控制的拉格朗日方法

```{code-cell} ipython3
:tags: [hide-output]
!pip install quantecon
```

```{code-cell} ipython3
import numpy as np
from quantecon import LQ
from scipy.linalg import schur
```

+++

## 概述

本讲是 {doc}`线性二次动态规划 <lqcontrol>` 的续篇

也可以视为对我们在 {doc}`线性理性预期模型中的稳定性<re_with_feedback>` 一讲中遇到的**不变子空间**技术的扩展

我们将介绍一个无限期线性二次无贴现动态规划问题的拉格朗日形式。

这类问题有时也被称为最优线性调节器问题。

拉格朗日形式

* 能够揭示稳定性和最优性之间的联系

* 是求解黎卡提方程的快速算法的基础

* 为构建不直接来自跨期优化问题的动态系统的解决方案开辟了道路

本讲中的一个关键工具是 $n \times n$ **辛矩阵**的概念。

辛矩阵的特征值以**倒数对**的形式出现，这意味着如果$\lambda_i \in (-1,1)$是一个特征值，那么$\lambda_i^{-1}$也是特征值。

矩阵特征值具有这种倒数对的性质，这是一个明显的标志，表明该矩阵描述了一个系统方程的联合动态，这些方程描述了构成求解无折扣线性二次无限期优化问题的一阶必要条件的**状态**和**协态**。

我们感兴趣的辛矩阵描述了最优控制系统中**状态**和**协态**向量的一阶动态。

在研究这个矩阵的特征值和特征向量时，我们着重分析**不变子空间**。

这些线性二次动态规划问题的不变子空间表述提供了一个桥梁，连接了递归

（即动态规划）公式和线性控制与线性滤波问题的经典公式，这些公式使用相关的矩阵分解（参见[这节课](https://python-advanced.quantecon.org/lu_tricks.html)和[这节课](https://python-advanced.quantecon.org/classical_filtering.html)）。

虽然本课程大部分内容关注非贴现问题，但后面的章节会描述将贴现问题转换为非贴现问题的便捷方法。

本课程中的这些技术在我们学习[这节课](https://python-advanced.quantecon.org/dyn_stack.html)中的斯塔克伯格和拉姆齐问题时将会很有用。

## 非贴现LQ动态规划问题

该问题是要选择一个控制序列 $\{u_t\}_{t=0}^\infty$ 来最大化以下准则

$$ 
- \sum_{t=0}^\infty \{x'_t Rx_t+u'_tQu_t\} 
$$

约束条件为 $x_{t+1}=Ax_t+Bu_t$，其中 $x_0$ 是给定的初始状态向量。

这里 $x_t$ 是一个 $(n\times 1)$ 的状态变量向量，$u_t$ 是一个 $(k\times 1)$ 的控制向量，$R$ 是一个半正定对称矩阵，$Q$ 是一个正定对称矩阵，$A$ 是一个 $(n\times n)$ 的矩阵，而 $B$ 是一个 $(n\times k)$ 的矩阵。

最优值函数被证明是二次型的，$V(x)= - x'Px$，其中 $P$ 是一个半正定对称矩阵。

使用转移规律消除下一期的状态，贝尔曼方程变为

$$ 
- x'Px=\max_u \{- x' Rx-u'Qu-(Ax+Bu)' P(Ax+Bu)\}
$$ (bellman0)

方程 {eq}`bellman0` 右侧最大化问题的一阶必要条件是

```{note} 
我们使用以下规则来对二次型和双线性矩阵形式求导：
${\partial x' A x \over \partial x} = (A + A') x; {\partial y' B z \over \partial y} = B z, {\partial
y' B z \over \partial z} = B' y$。
```

$$
(Q+B'PB)u=-B'PAx,
$$

这意味着 $u$ 的最优决策规则是

$$

$$
u=-(Q+B'PB)^{-1} B'PAx
$$ 

或

$$
u=-Fx,
$$

其中

$$ 
F=(Q+B'PB)^{-1}B'PA.
$$

将 $u = - (Q+B'PB)^{-1}B'PAx$ 代入方程 {eq}`bellman0` 的右侧并重新整理得到

$$
P=R+A'PA-A'PB(Q+B'PB)^{-1} B'PA.
$$ (riccati)

方程 {eq}`riccati` 被称为**代数矩阵黎卡提**方程。

方程 {eq}`riccati` 有多个解。

但只有一个解是正定的。

这个正定解与我们问题的最大值相关。

它将矩阵 $P$ 表示为矩阵 $R,Q,A,B$ 的隐函数。

注意**值函数的梯度**为

$$
\frac{\partial V(x)}{\partial x} = - 2 P x 
$$ (eqn:valgrad)

我们稍后将使用公式 {eq}`eqn:valgrad`。

+++

## 拉格朗日量

对于无折扣最优线性调节器问题，构造拉格朗日量

$$
{\cal L} = - \sum^\infty_{t=0} \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t +

2 \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}
$$ (lag-lqdp-eq1)

其中 $2 \mu_{t+1}$ 是时间 $t$ 转移方程 $x_{t+1} = A x_t + B u_t$ 的拉格朗日乘子向量。

（我们在 $\mu_{t+1}$ 前面加上 $2$ 是为了使其与方程 {eq}`eqn:valgrad` 很好地对应。）

关于 $\{u_t,x_{t+1}\}_{t=0}^\infty$ 的最大化一阶条件是

$$
\begin{aligned}
2 Q u_t &+ 2B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr
\end{aligned}
$$ (lag-lqdp-eq2)

定义 $\mu_0$ 为 $x_0$ 的影子价格向量，并对 {eq}`lag-lqdp-eq1` 应用包络条件可推导出

$$
\mu_0 = R x_0 + A' \mu_1,
$$

这是系统 {eq}`lag-lqdp-eq2` 中第二个方程在时间 $t=0$ 时的对应形式。

一个重要的事实是

$$ 
\mu_{t+1} = P x_{t+1}
$$ (eqn:muPx)

其中 $P$ 是一个正定矩阵，它是代数黎卡提方程 {eq}`riccati` 的解。

因此，根据方程 {eq}`eqn:valgrad` 和 {eq}`eqn:muPx`，$- 2 \mu_{t}$ 是值函数对 $x_t$ 的梯度。

Lagrange乘子向量 $\mu_{t}$ 通常被称为与**状态**向量 $x_t$ 对应的**协状态**向量。

按照以下步骤进行是很有用的：

* 从 {eq}`lag-lqdp-eq2` 的第一个方程解出关于 $\mu_{t+1}$ 的 $u_t$。

* 将结果代入运动方程 $x_{t+1} = A x_t + B u_t$。

* 将得到的方程和 {eq}`lag-lqdp-eq2` 的第二个方程整理成如下形式

$$
L\ \begin{pmatrix}x_{t+1}\cr \mu_{t+1}\cr\end{pmatrix}\ = \ N\ \begin{pmatrix}x_t\cr \mu_t\cr\end{pmatrix}\
,\ t \geq 0,
$$ (eq:systosolve)

其中

$$
L = \ \begin{pmatrix}I & BQ^{-1} B^\prime \cr 0 & A^\prime\cr\end{pmatrix}, \quad N = \
\begin{pmatrix}A & 0\cr -R & I\cr\end{pmatrix}.
$$

当 $L$ 满秩时（即当 $A$ 满秩时），我们可以将系统 {eq}`eq:systosolve` 写作

$$

\begin{pmatrix}x_{t+1}\cr \mu_{t+1}\cr\end{pmatrix}\ = M\ \begin{pmatrix}x_t\cr\mu_t\cr\end{pmatrix}
$$ (eq4orig)

其中

$$
M\equiv L^{-1} N = \begin{pmatrix}A+B Q^{-1} B^\prime A^{\prime-1}R &
-B Q^{-1} B^\prime A^{\prime-1}\cr -A^{\prime -1} R & A^{\prime -1}\cr\end{pmatrix}.
$$ (Mdefn)

+++

## 状态-协状态动态

我们寻求解差分方程系统{eq}`eq4orig`，以得到满足以下条件的序列$\{x_t\}_{t=0}^\infty$：

* $x_0$的初始条件
* 终端条件$\lim_{t \rightarrow +\infty} x_t =0$

这个终端条件反映了我们对**稳定**解的需求，即当$t \rightarrow \infty$时不会发散的解。

我们对$\{x_t\}$序列稳定性的要求源于最大化以下表达式的愿望：

$$ 
-\sum_{t=0}^\infty \bigl[ x_t ' R x_t + u_t' Q u_t \bigr],
$$

这要求当$t \rightarrow + \infty$时，$x_t' R x_t$收敛于零。

+++

## 互反对特性

为了继续，我们研究在{eq}`Mdefn`中定义的$(2n \times 2n)$矩阵$M$的性质。

引入一个$(2n \times 2n)$矩阵会有帮助：

$$
J = \begin{pmatrix}0 & -I_n\cr I_n & 0\cr\end{pmatrix}.
$$

矩阵$J$的秩为$2n$。

**定义：** 如果矩阵$M$满足

$$
MJM^\prime = J.
$$ (lag-lqdp-eq3)

则称其为**辛矩阵**。

辛矩阵的显著性质包括（容易验证）：

  * 如果$M$是辛矩阵，那么$M^2$也是辛矩阵
  * 如果$M$是辛矩阵，那么$\textrm{det}(M) = 1$

可以直接验证方程{eq}`Mdefn`中的$M$是辛矩阵。

从方程{eq}`lag-lqdp-eq3`和$J^{-1} = J^\prime = -J$可以推导出，对任意辛矩阵$M$，有

$$
M^\prime = J^{-1} M^{-1} J.
$$ (lag-lqdp-eq4)

方程{eq}`lag-lqdp-eq4`表明$M^\prime$与$M$的逆矩阵通过**相似变换**相关。

对于方阵，请记住：

* 相似矩阵具有相同的特征值

* 矩阵的逆的特征值是该矩阵特征值的倒数

* 矩阵和它的转置矩阵具有相同的特征值

从方程{eq}`lag-lqdp-eq4`可以得出，$M$的特征值成倒数对：如果$\lambda$是$M$的特征值，那么$\lambda^{-1}$也是。

将方程{eq}`eq4orig`写作

$$
y_{t+1} = M y_t
$$ (eq658)

其中$y_t = \begin{pmatrix}x_t\cr \mu_t\cr\end{pmatrix}$。

考虑$M$的**三角化**

$$
V^{-1} M V= \begin{pmatrix}W_{11} & W_{12} \cr 0 & W_{22}\cr\end{pmatrix}
$$ (eqn:triangledecomp)

其中

* 右侧的每个块都是$(n\times n)$维的
* $V$是非奇异的
* $W_{22}$的所有特征值的模都大于1
* $W_{11}$的所有特征值的模都小于1

## 舒尔分解

**舒尔分解**和**特征值分解**是两种形如{eq}`eqn:triangledecomp`的分解。

将方程{eq}`eq658`写作

$$
y_{t+1} = V W V^{-1} y_t.
$$ (eq659)

方程{eq}`eq659`对任意初始条件$y_0$的解显然为

$$
y_{t} = V \left[\begin{matrix}W^t_{11} & W_{12,t}\cr 0 & W^t_{22}\cr\end{matrix}\right]
\ V^{-1} y_0
$$ (eq6510)

其中当$t=1$时$W_{12,t} = W_{12}$，对于$t \geq 2$遵循递归关系

$$
W_{12, t} = W^{t-1}_{11} W_{12,t-1} + W_{12,t-1} W^{t-1}_{22}
$$

这里$W^t_{ii}$是$W_{ii}$的$t$次幂。

将方程{eq}`eq6510`写作

$$
\begin{pmatrix}y^\ast_{1t}\cr y^\ast_{2t}\cr\end{pmatrix}\ =\ \left[\begin{matrix} W^t_{11} &
W_{12, t}\cr 0 & W^t_{22}\cr\end{matrix}\right]\quad \begin{pmatrix}y^\ast_{10}\cr
y^\ast_{20}\cr\end{pmatrix}
$$

其中$y^\ast_t = V^{-1} y_t$，特别地

$$
y^\ast_{2t} = V^{21} x_t + V^{22} \mu_t,
$$ (eq6511)

这里$V^{ij}$表示分块矩阵$V^{-1}$的$(i,j)$部分。

由于$W_{22}$是一个不稳定矩阵，除非$y^\ast_{20} = 0$，否则$y^\ast_t$将发散。

令 $V^{ij}$ 表示分块矩阵 $V^{-1}$ 的第 $(i,j)$ 块。

为了获得稳定性，我们必须设定 $y^\ast_{20} =0$，根据方程 {eq}`eq6511` 可得

$$
V^{21} x_0 + V^{22} \mu_0 = 0
$$

或

$$
\mu_0 = - (V^{22})^{-1} V^{21} x_0.
$$

这个方程在时间上会重复出现，即意味着

$$
\mu_t = - (V^{22})^{-1} V^{21} x_t.
$$

但注意，由于 $(V^{21}\ V^{22})$ 是 $V$ 的逆矩阵的第二行块，因此

$$
(V^{21} \ V^{22})\quad \begin{pmatrix}V_{11}\cr V_{21}\cr\end{pmatrix} = 0
$$

这意味着

$$
V^{21} V_{11} + V^{22} V_{21} = 0.
$$

因此，

$$
-(V^{22})^{-1} V^{21} = V_{21} V^{-1}_{11}.
$$

所以我们可以写成

$$
\mu_0 = V_{21} V_{11}^{-1} x_0
$$

和

$$
\mu_t = V_{21} V^{-1}_{11} x_t.
$$

然而，我们知道 $\mu_t = P x_t$，其中 $P$ 出现在求解黎卡提方程的矩阵中。

因此，前面的论证确立了

$$
P = V_{21} V_{11}^{-1}.
$$ (eqn:Pvaughn)

值得注意的是，公式{eq}`eqn:Pvaughn`为我们提供了一种计算效率高的方法来计算正定矩阵$P$，该矩阵可以解决动态规划中出现的代数黎卡提方程{eq}`riccati`。

即使$M$的特征值不以互反对的形式出现，这种方法也可以用来计算任何形如{eq}`eq4orig`的系统的解（如果解存在的话）。

只要$M$的特征值在单位圆内外各半分布，这种方法通常就能奏效。

当系统是一个存在扭曲的模型的均衡时，特征值（经适当贴现调整后）可能不会以互反对的形式出现，这种扭曲使得均衡无法求解任何最优化问题。参见{cite}`Ljungqvist2012`第12章。

## 应用

这里我们通过一个示例来演示计算过程，这个示例是从[quantecon讲座](https://python.quantecon.org/lqcontrol.html)中借鉴的确定性版本。

```{code-cell} ipython3
# 模型参数
r = 0.05
c_bar = 2
μ = 1

# 构建为LQ问题
Q = np.array([[1]])
R = np.zeros((2, 2))
A = [[1 + r, -c_bar + μ],
     [0,              1]]
B = [[-1],
     [0]]

# 构造一个LQ实例
lq = LQ(Q, R, A, B)
```

给定矩阵 $A$、$B$、$Q$、$R$，我们可以计算 $L$、$N$ 和 $M=L^{-1}N$。

```{code-cell} ipython3
def construct_LNM(A, B, Q, R):

    n, k = lq.n, lq.k

    # 构造 L 和 N
    L = np.zeros((2*n, 2*n))
    L[:n, :n] = np.eye(n)
    L[:n, n:] = B @ np.linalg.inv(Q) @ B.T
    L[n:, n:] = A.T

    N = np.zeros((2*n, 2*n))
    N[:n, :n] = A
    N[n:, :n] = -R
    N[n:, n:] = np.eye(n)

    # 计算 M
    M = np.linalg.inv(L) @ N

    return L, N, M
```

```{code-cell} ipython3
L, N, M = construct_LNM(lq.A, lq.B, lq.Q, lq.R)
```

```{code-cell} ipython3
M
```

让我们验证 $M$ 是辛矩阵。

```{code-cell} ipython3
n = lq.n
J = np.zeros((2*n, 2*n))
J[n:, :n] = np.eye(n)
J[:n, n:] = -np.eye(n)

M @ J @ M.T - J
```

我们可以使用`np.linalg.eigvals`计算矩阵$M$的特征值，并按升序排列。

```{code-cell} ipython3
eigvals = sorted(np.linalg.eigvals(M))
eigvals
```

当我们应用舒尔分解使得$M=V W V^{-1}$时，我们希望

* $W$的左上块$W_{11}$的所有特征值的模小于1，并且
* 右下块$W_{22}$的特征值的模大于1。

为了得到我们想要的结果，让我们定义一个排序函数，告诉`scipy.schur`将模小于1的对应特征值排序到左上角。

```{code-cell} ipython3
stable_eigvals = eigvals[:n]

def sort_fun(x):
    "将模小于1的特征值排序到左上角。"

    if x in stable_eigvals:
        stable_eigvals.pop(stable_eigvals.index(x))
        return True
    else:
        return False

W, V, _ = schur(M, sort=sort_fun)
```

```{code-cell} ipython3
W
```

```{code-cell} ipython3
V
```

我们可以检查 $W_{11}$ 和 $W_{22}$ 的特征值的模。

由于它们都是三角矩阵，特征值就是对角线元素。

```{code-cell} ipython3
# W11
np.diag(W[:n, :n])
```

```{code-cell} ipython3
# W22
np.diag(W[n:, n:])
```

以下函数封装了 $M$ 矩阵构建、舒尔分解以及稳定性约束下的 $P$ 计算。

```{code-cell} ipython3
def stable_solution(M, verbose=True):
    """
    给定一个线性差分方程系统

        y' = |a b| y
        x' = |c d| x

    该系统可能不稳定，通过施加稳定性约束来求解。

    参数
    ---------
    M : np.ndarray(float)
        表示线性差分方程系统的矩阵。
    """
    n = M.shape[0] // 2
    stable_eigvals = list(sorted(np.linalg.eigvals(M))[:n])

    def sort_fun(x):
        "将模小于1的特征值排序到左上角。"

        if x in stable_eigvals:
            stable_eigvals.pop(stable_eigvals.index(x))
            return True
        else:
            return False

    W, V, _ = schur(M, sort=sort_fun)
    if verbose:
        print('特征值：\n')
        print('    W11: {}'.format(np.diag(W[:n, :n])))
        print('    W22: {}'.format(np.diag(W[n:, n:])))

    # 计算 V21 V11^{-1}
    P = V[n:, :n] @ np.linalg.inv(V[:n, :n])

    return W, V, P

def stationary_P(lq, verbose=True):
    """
    计算表示值函数的矩阵 :math:`P`

         V(x) = x' P x

    在无限时域情况下。通过在解路径上施加稳定性约束
    并使用舒尔分解来进行计算。

    参数
    ----------
    lq : qe.LQ
        用于分析无限时域形式的线性二次最优控制问题的
        QuantEcon类。

    返回值
    -------
    P : array_like(float)
        值函数表示中的P矩阵。
    """

    Q = lq.Q
    R = lq.R
    A = lq.A * lq.beta ** (1/2)
    B = lq.B * lq.beta ** (1/2)

    n, k = lq.n, lq.k

    L, N, M = construct_LNM(A, B, Q, R)
    W, V, P = stable_solution(M, verbose=verbose)

    return P
```

```{code-cell} ipython3
# 计算P
stationary_P(lq)
```

请注意，以这种方式计算得到的矩阵 $P$ 与 quantecon 中通过迭代 Riccati 差分方程直至收敛来求解代数 Riccati 方程的程序得到的结果非常接近。

这种微小的差异来自计算误差，可以通过增加最大迭代次数或降低收敛容差来减小。

```{code-cell} ipython3
lq.stationary_values()
```

使用舒尔分解效率要高得多。

```{code-cell} ipython3
%%timeit
stationary_P(lq, verbose=False)
```

```{code-cell} ipython3
%%timeit
lq.stationary_values()
```

## 其他应用

上述用于稳定潜在不稳定线性差分方程系统的方法并不仅限于线性二次动态优化问题。

例如，在我们的[线性理性预期模型中的稳定性](https://python.quantecon.org/re_with_feedback.html#another-perspective)讲座中也使用了相同的方法。

让我们尝试使用本讲座上文定义的`stable_solution`函数来求解该讲座中描述的模型。

```{code-cell} ipython3
def construct_H(ρ, λ, δ):
    "根据参数构建矩阵H。"

    H = np.empty((2, 2))
    H[0, :] = ρ,δ
    H[1, :] = - (1 - λ) / λ, 1 / λ

    return H

H = construct_H(ρ=.9, λ=.5, δ=0)
```

```{code-cell} ipython3
W, V, P = stable_solution(H)
P
```

## 贴现问题

+++

### 转换状态和控制以消除贴现

一对有用的转换允许我们将贴现问题转换为非贴现问题。

假设我们有一个贴现问题，其目标函数为

$$
 - \sum^\infty_{t=0} \beta^t \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t \biggr\}
$$ 

且状态转移方程仍为 $x_{t +1 }=Ax_t+Bu_t$。

定义转换后的状态和控制变量

* $\hat x_t = \beta^{\frac{t}{2}} x_t $
* $\hat u_t = \beta^{\frac{t}{2}} u_t$

以及转换后的转移方程矩阵

* $\hat A = \beta^{\frac{1}{2}} A$
* $\hat B =  \beta^{\frac{1}{2}} B  $

使得调整后的状态和控制变量遵循转移规律

$$
\hat x_{t+1} = \hat A \hat x_t + \hat B \hat u_t. 
$$ 

那么由 $A, B, R, Q, \beta$ 定义的贴现最优控制问题，其最优策略由 $P, F$ 表征，与一个等价的

由 $\hat A, \hat B, Q, R$ 定义的非贴现问题具有最优策略，其特征由满足以下方程的 $\hat F, \hat P$ 表示：

$$
\hat F=(Q+B'\hat PB)^{-1}\hat B'P \hat A
$$

和

$$
\hat P=R+\hat A'P \hat A-\hat A'P \hat B(Q+B'\hat P \hat B)^{-1} \hat B'P \hat A
$$

从 $\hat A, \hat B$ 的定义可以直接得出 $\hat F = F$ 和 $\hat P = P$。

通过利用这些转换，我们可以通过求解相关的非贴现问题来解决贴现问题。

特别地，我们可以先将贴现线性二次问题转换为非贴现问题，然后使用上述拉格朗日和不变子空间方法求解该贴现最优调节器问题。

+++

例如，当 $\beta=\frac{1}{1+r}$ 时，我们可以用 $\hat{A}=\beta^{1/2} A$ 和 $\hat{B}=\beta^{1/2} B$ 求解 $P$。

这些设置在上面定义的 `stationary_P` 函数中是默认采用的。

```{code-cell} ipython3
β = 1 / (1 + r)
lq.beta = β
```

```{code-cell} ipython3
stationary_P(lq)
```

我们可以验证该解与使用 quantecon 包中的 `LQ.stationary_values` 例程得到的解是一致的。

```{code-cell} ipython3
lq.stationary_values()
```

### 贴现问题的拉格朗日量

出于多种目的，有必要简要地明确描述贴现问题的拉格朗日量。

因此，对于贴现最优线性调节器问题，构建拉格朗日量：

$$
{\cal{L}} = - \sum^\infty_{t=0} \beta^t \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t
+ 2 \beta \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}
$$ (eq661)

其中$2 \mu_{t+1}$是状态向量$x_{t+1}$的拉格朗日乘子向量。

对于$\{u_t,x_{t+1}\}_{t=0}^\infty$的最大化一阶条件是：

$$
\begin{aligned}
2 Q u_t &+ 2  \beta B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + \beta A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr
\end{aligned}
$$ (eq662)

定义$2 \mu_0$为$x_0$的影子价格向量，并对{eq}`eq661`应用包络条件可得：

$$
\mu_0 = R x_0 + \beta A' \mu_1 ,
$$

这是系统{eq}`eq662`中第二个方程在$t=0$时刻的对应形式。

按照上述未贴现系统{eq}`lag-lqdp-eq2`的处理方法,我们可以将一阶条件重新整理为如下系统:

$$
\left[\begin{matrix} I & \beta B Q^{-1} B' \cr
             0 & \beta A' \end{matrix}\right]
\left[\begin{matrix} x_{t+1} \cr \mu_{t+1} \end{matrix}\right] =
\left[\begin{matrix} A & 0 \cr
             - R & I \end{matrix}\right] 
\left[\begin{matrix} x_t \cr \mu_t \end{matrix}\right]
$$ (eq663)

在特殊情况$\beta = 1$时,该式与方程{eq}`lag-lqdp-eq2`一致,这是符合预期的。

+++

通过观察系统{eq}`eq663`,我们可以推断出一些揭示最优线性调节器问题结构的恒等式。其中一些在[这篇讲座](https://python-advanced.quantecon.org/dyn_stack.html)中会很有用,当我们应用和扩展本讲座的方法来研究斯塔克尔伯格和拉姆齐问题时。

首先,注意方程系统{eq}`eq663`的第一个区块表明,当$\mu_{t+1} = P x_{t+1}$时,

$$

(I + \beta Q^{-1} B' P B P ) x_{t+1} = A x_t, 
$$
 
可以重新整理为

$$
x_{t+1} = (I + \beta B Q^{-1} B' P)^{-1}  A x_t .
$$

状态最优闭环动态的这个表达式必须与我们之前通过动态规划得到的另一个表达式一致,即:

$$
x_{t+1} = (A - BF) x_t .
$$

但使用

$$
F=\beta (Q+\beta B'PB)^{-1} B'PA 
$$ (eqn:optimalFformula)

可以推导出

$$ 
A- B F = (I - \beta B (Q+ \beta B' P B)^{-1} B' P) A .
$$ 

因此,我们的两个闭环动态表达式当且仅当满足以下条件时才相等:

$$ 
(I + \beta B Q^{-1} B' P )^{-1} =    (I - \beta B (Q+\beta  B' P B)^{-1} B' P) .
$$ (eqn:twofeedbackloops)

矩阵方程{eq}`eqn:twofeedbackloops`可以通过应用分块矩阵求逆公式来验证。

```{note}
只需对适当选择的矩阵$a, b, c, d$使用公式$(a - b d^{-1} c)^{-1} = a^{-1} + a^{-1} b (d - c a^{-1} b)^{-1} c a^{-1}$即可。
```

接下来，注意对于*任何*固定的$F$，只要$A-BF$的特征值的模小于$\frac{1}{\beta}$，使用这个规则的值函数永远是$-x_0 \tilde P x_0$，其中$\tilde P$满足以下矩阵方程：

$$
\tilde P = (R + F' Q F) + \beta (A - B F)' P (A - BF) .
$$ (eq666)

显然，只有当$F$满足公式{eq}`eqn:optimalFformula`时，$\tilde P = P$才成立。

接下来，注意系统{eq}`eq663`的第二个方程暗示了拉格朗日乘数的"前瞻"方程

$$ 
\mu_t = R x_t + \beta A' \mu_{t+1}
$$

其解为

$$
\mu_t = P x_t ,
$$

其中

$$
P = R + \beta A' P (A - BF)  
$$ (eq667)

这里我们必须要求$F$满足方程{eq}`eqn:optimalFformula`。

方程{eq}`eq666`和{eq}`eq667`为最优值函数提供了不同的视角。

