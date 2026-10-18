---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
translation:
  title: QR分解
  headings:
    Overview: 概述
    Matrix factorization: 矩阵分解
    Gram-Schmidt process: 格拉姆-施密特正交化
    Gram-Schmidt process::Gram-Schmidt process for square $A$: 方阵$A$的格拉姆-施密特正交化
    Gram-Schmidt process::$A$ not square: $A$ 非方阵
    Some code: 一些代码
    Example: 示例
    Using QR decomposition to compute eigenvalues: 使用QR分解计算特征值
    QR and PCA: $QR$ 分解与主成分分析（PCA）
---

# QR分解

## 概述

本讲解释QR分解及其与以下内容的关系：

* 正交投影和最小二乘法

* 格拉姆-施密特（Gram-Schmidt）正交化

* 特征值和特征向量

我们将编写一些Python代码来帮助巩固理解。

## 矩阵分解

在本讲座中，我们考虑一个实矩阵 $A \in \mathbb{R}^{n \times m}$，其中 $m \geq n$。

$A$ 的QR分解（也称为QR因式分解）形式为：

$$
A = QR, \qquad
Q \in \mathbb{R}^{n \times n}, \qquad
R \in \mathbb{R}^{n \times m}, \qquad
Q^\top Q = I_n.
$$

矩阵 $R$ 在其主对角线下方均为零。

因此，当 $m = n$ 时 $R$ 是上三角矩阵，当 $m > n$ 时 $R$ 是上梯形矩阵。

我们将使用*格拉姆-施密特正交化*来计算QR分解。

由于这个过程很有意思，我们将编写自己的Python代码来完成这项工作。

## 格拉姆-施密特正交化

我们从一个*方阵*$A$开始。

如果方阵$A$是非奇异的，那么QR分解在符号意义下是唯一的。

我们稍后会处理矩形矩阵$A$。

实际上，我们的算法也适用于非方阵的矩形矩阵$A$。

### 方阵$A$的格拉姆-施密特正交化

这里我们对矩阵$A$的*列*运用格拉姆-施密特正交化。

具体来说，令

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_n \end{bmatrix}
$$

令$|| \cdot ||$表示L2范数。

格拉姆-施密特算法按特定顺序重复以下两个步骤

* **标准化**向量使其具有单位范数

* **正交化**下一个向量

首先，我们设$u_1 = a_1$然后进行*标准化*：

$$
u_1=a_1, \ \ \ e_1=\frac{u_1}{||u_1||}
$$

我们先*正交化*计算$u_2$，然后*标准化*得到$e_2$：

$$
u_2=a_2-(a_2\cdot e_1)e_1, \ \ \  e_2=\frac{u_2}{||u_2||}
$$

我们邀请读者通过验证 $e_1 \cdot e_2 = 0$ 来确认 $e_1$ 与 $e_2$ 正交。

格拉姆-施密特算法过程迭代这个过程。

因此，对于 $k= 2, \ldots, n-1$，我们构造

$$
u_{k+1}=a_{k+1}-(a_{k+1}\cdot e_1)e_1-\cdots-(a_{k+1}\cdot e_k)e_k, \ \ \ e_{k+1}=\frac{u_{k+1}}{||u_{k+1}||}
$$

这里的 $(a_j \cdot e_i)$ 可以被解释为 $a_j$ 在 $e_i$ 上的线性最小二乘*回归系数*

* 它是 $a_j$ 和 $e_i$ 的内积除以 $e_i$ 的内积，其中由于*标准化*，我们知道 $e_i \cdot e_i = 1$
    
* 这个回归系数可以解释为*协方差*除以*方差*

可以验证

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_n \end{bmatrix}=
\begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
\begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1\\ 0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n\cdot e_n \end{bmatrix}
$$

因此，我们构造了矩阵分解

$$ 
A = Q R
$$

其中

$$ 
Q = \begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
$$

且

$$
R = \begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1\\ 0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n\cdot e_n \end{bmatrix}
$$

### $A$ 非方阵

现在假设 $A$ 是一个 $n \times m$ 矩阵，其中 $m > n$。

那么QR分解为

$$
A= \begin{bmatrix} a_1 & a_2 & \cdots & a_m \end{bmatrix}=\begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix}
\begin{bmatrix} a_1\cdot e_1 & a_2\cdot e_1 & \cdots & a_n\cdot e_1 & a_{n+1}\cdot e_1 & \cdots & a_{m}\cdot e_1 \\
0 & a_2\cdot e_2 & \cdots & a_n\cdot e_2 & a_{n+1}\cdot e_2 & \cdots & a_{m}\cdot e_2 \\ \vdots & \vdots & \ddots & \quad  \vdots & \vdots & \ddots & \vdots
\\ 0 & 0 & \cdots & a_n\cdot e_n & a_{n+1}\cdot e_n & \cdots & a_{m}\cdot e_n \end{bmatrix}
$$

这里，$R$ 是一个 $n \times m$ 的上梯形矩阵。

这意味着

\begin{align*}
a_1 & = (a_1\cdot e_1) e_1 \cr
a_2 & = (a_2\cdot e_1) e_1 + (a_2\cdot e_2) e_2 \cr
\vdots & \quad \vdots \cr
a_n & = (a_n\cdot e_1) e_1 + (a_n\cdot e_2) e_2 + \cdots + (a_n \cdot e_n) e_n  \cr
a_{n+1} & = (a_{n+1}\cdot e_1) e_1 + (a_{n+1}\cdot e_2) e_2 + \cdots + (a_{n+1}\cdot e_n) e_n  \cr
\vdots & \quad \vdots \cr
a_m & = (a_m\cdot e_1) e_1 + (a_m\cdot e_2) e_2 + \cdots + (a_m \cdot e_n) e_n  \cr
\end{align*}

## 一些代码

现在让我们编写一些自制的Python代码，通过部署上述的格拉姆-施密特正交化来实现QR分解。

```{code-cell} ipython3
import numpy as np
from scipy.linalg import qr

rng = np.random.default_rng(1234)
```

下面的`qr_decomposition`函数展示了方阵的情况，以及将其推广到宽矩阵（即$m \geq n$的$n \times m$矩阵）的情况，前提是假设前$n$列线性无关。

```{code-cell} ipython3
def qr_decomposition(A):
    n, m = A.shape # 获取A的形状

    Q = np.empty((n, n)) # 初始化矩阵Q
    u = np.empty((n, n)) # 初始化矩阵u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # 获取每个u向量

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # 计算每个e向量

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R
```

前面的代码没问题，但可以进一步进行整理。

这样做的目的是为了后续能够将我们自制的QR分解代码与Python SciPy包中的QR分解函数进行对比。

不同的数值算法产生的$Q$和$R$矩阵之间可能存在符号差异。

由于在计算$QR$时这些符号差异会相互抵消，所以这些都是有效的QR分解。

为了使我们自制函数和SciPy中QR模块的结果具有可比性，我们要求$R$的对角线元素为非负数。

我们通过适当调整$Q$中列的符号和$R$中对应行的符号来实现这一点。

我们将定义一对函数来完成上述要求。

```{code-cell} ipython3
def diag_sign(A):
    """计算对角符号矩阵，将零符号映射为1。"""

    signs = np.where(np.diag(A) < 0, -1.0, 1.0)
    D = np.diag(signs)

    return D

def adjust_sign(Q, R):
    """
    调整Q中列的符号和R中行的符号，
    以确保R具有非负对角线。
    """

    D = diag_sign(R)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R
```

## 示例

现在让我们举一个例子。

```{code-cell} ipython3
A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0]])

A
```

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(A))
```

```{code-cell} ipython3
Q
```

```{code-cell} ipython3
R
```

让我们将结果与 SciPy 包产生的结果进行比较

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
```

```{code-cell} ipython3
print('我们的 Q: \n', Q)
print('\n')
print('SciPy 的 Q: \n', Q_scipy)
```

```{code-cell} ipython3
print('我们的 R: \n', R)
print('\n')
print('SciPy 的 R: \n', R_scipy)
```

上述结果给我们带来好消息，我们自制的函数与 SciPy 产生的结果一致。

现在让我们对一个矩形矩阵 $A$ 进行 QR 分解，这个矩阵是 $n \times m$ 的，其中 $m > n$。

```{code-cell} ipython3
A = np.array([[1, 3, 4], [2, 0, 9]])
```

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(A))
Q, R
```

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
Q_scipy, R_scipy
```

## 使用QR分解计算特征值

现在介绍一个关于QR算法的有用事实。

以下基于QR分解的迭代可用于计算 *方阵* $A$ 的 *特征值*。

算法如下：

1. 设 $A_0 = A$ 并构建 $A_0 = Q_0 R_0$

2. 构建 $A_1 = R_0 Q_0$。注意 $A_1$ 与 $A_0$ 相似（易于验证），因此具有相同的特征值。

3. 构建 $A_1 = Q_1 R_1$（即构建 $A_1$ 的QR分解）。

4. 构建 $A_2 = R_1 Q_1$ 然后构建 $A_2 = Q_2 R_2$。

5. 迭代直至收敛。

6. 计算 $A$ 的特征值，并将其与从该过程得到的极限 $A_n$ 的对角线值进行比较。

```{prf:remark}
这个算法实际上非常接近计算特征值最高效的方法！
```

让我们编写一些Python代码来尝试这个算法

```{code-cell} ipython3
def qr_eigvals(A, tol=1e-12, maxiter=1000):
    "使用QR分解找到A的特征值。"

    A_old = np.array(A, dtype=float, copy=True)
    A_new = np.array(A, dtype=float, copy=True)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = qr_decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals
```

现在让我们试试这段代码，并将结果与 `np.linalg.eigvals` 进行比较。

开始吧

```{code-cell} ipython3
# 用一个随机矩阵A做实验
A = rng.random((3, 3))
```

```{code-cell} ipython3
sorted(qr_eigvals(A))
```

与 NumPy 的实现进行比较。

```{code-cell} ipython3
sorted(np.linalg.eigvals(A))
```

## $QR$ 分解与主成分分析（PCA）

QR 分解与主成分分析（PCA）之间存在一些有趣的联系。

以下是一些联系：

1. 设 $X^\top$ 是一个 $k \times n$ 的随机矩阵，其中第 $j$ 列是从 $N(\mu, \Sigma)$ 分布中随机抽取的样本，这里 $\mu$ 是 $k \times 1$ 的均值向量，$\Sigma$ 是 $k \times k$ 的协方差矩阵。我们需要 $n \gg k$ —— 这是一个"计量经济学"的例子。

2. 将 $X^\top$ 分解为 $X^\top = Q R$，其中 $Q$ 是 $k \times k$ 矩阵，$R$ 是 $k \times n$ 矩阵。

3. 计算 $R R^\top$ 的特征值，即我们将计算 $R R^\top = \tilde P \Lambda \tilde P^\top$。

4. 构造 $X^\top X = Q \tilde P \Lambda \tilde P^\top Q^\top$ 并与特征分解 $X^\top X = P \hat \Lambda P^\top$ 进行比较。

5. 我们应该会发现 $\Lambda = \hat \Lambda$ 且 $P = Q \tilde P$。

让我们用Python代码来验证推测5。

首先模拟一个随机的 $\left(n, k\right)$ 矩阵 $X$。

```{code-cell} ipython3
k = 5
n = 1000

# 生成一些随机矩
𝜇 = rng.random(size=k)
C = rng.random((k, k))
Σ = C.T @ C
```

```{code-cell} ipython3
# X 的每一行都是从多元正态分布中抽取的样本
X = rng.multivariate_normal(𝜇, Σ, size=n)
```

```{code-cell} ipython3
X.shape
```

让我们对 $X^\top$ 进行QR分解。

```{code-cell} ipython3
Q, R = adjust_sign(*qr_decomposition(X.T))
```

检查 $Q$ 和 $R$ 的形状。

```{code-cell} ipython3
Q.shape, R.shape
```


现在我们可以构造 $R R^\top=\tilde{P} \Lambda \tilde{P}^\top$ 并构建特征分解。

```{code-cell} ipython3
RR = R @ R.T

𝜆, P_tilde = np.linalg.eigh(RR)
Λ = np.diag(𝜆)
```

我们也可以对 $X^\top X=P \hat{\Lambda} P^\top$ 进行分解。

```{code-cell} ipython3
XX = X.T @ X

𝜆_hat, P = np.linalg.eigh(XX)
Λ_hat = np.diag(𝜆_hat)
```

比较 $\Lambda$ 和 $\hat{\Lambda}$ 对角线上的特征值。

```{code-cell} ipython3
𝜆, 𝜆_hat
```

让我们比较 $P$ 和 $Q \tilde{P}$。

同样，我们需要注意 $P$ 和 $Q\tilde{P}$ 的列之间的符号差异。

```{code-cell} ipython3
QP_tilde = Q @ P_tilde

np.abs(P @ diag_sign(P) - QP_tilde @ diag_sign(QP_tilde)).max()
```

让我们验证 $X^\top X$ 可以被分解为 $Q \tilde{P} \Lambda \tilde{P}^\top Q^\top$。

```{code-cell} ipython3
QPΛPQ = Q @ P_tilde @ Λ @ P_tilde.T @ Q.T
```

```{code-cell} ipython3
np.abs(QPΛPQ - XX).max()
```
