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
---

# QR分解

## 概述

本讲座介绍QR分解及其与以下内容的关系：

 * 正交投影和最小二乘法

 * Gram-Schmidt正交化过程

 * 特征值和特征向量

我们将编写一些Python代码来帮助巩固我们的理解。

## 矩阵分解

QR分解（也称为QR因式分解）是将矩阵分解为一个正交矩阵和一个三角矩阵的乘积。

实矩阵 $A$ 的QR分解形式为：

$$
A=QR
$$

其中：

* $Q$ 是正交矩阵（因此 $Q^TQ = I$）

* $R$ 是上三角矩阵

我们将使用**Gram-Schmidt正交化过程**来计算QR分解

因为这样做很有教育意义，我们将编写自己的Python代码来完成这项工作

## Gram-Schmidt正交化过程

我们将从一个**方阵** $A$ 开始。

如果方阵 $A$ 是非奇异的，那么QR分解是唯一的。

我们稍后会处理非方阵 $A$ 的情况。

实际上，我们的算法也适用于非方形的矩形矩阵 $A$。

### 方阵 $A$ 的Gram-Schmidt过程

这里我们对矩阵 $A$ 的**列**应用Gram-Schmidt过程。

具体来说，设

$$
A= \left[ \begin{array}{c|c|c|c} a_1 & a_2 & \cdots & a_n \end{array} \right]
$$

令 $|| · ||$ 表示L2范数。

Gram-Schmidt算法按特定顺序重复以下两个步骤：

* **归一化**向量使其具有单位范数

* **正交化**下一个向量

首先，我们设 $u_1 = a_1$ 然后**归一化**：

$$
u_1=a_1, \ \ \ e_1=\frac{u_1}{||u_1||}
$$

我们先**正交化**计算 $u_2$ 然后**归一化**得到 $e_2$：

$$
u_2=a_2-(a_2· e_1)e_1, \ \ \  e_2=\frac{u_2}{||u_2||}
$$

我们邀请读者通过检验 $e_1 \cdot e_2 = 0$ 来验证 $e_1$ 与 $e_2$ 正交。

Gram-Schmidt过程继续迭代。

因此，对于 $k= 2, \ldots, n-1$，我们构造：

$$
u_{k+1}=a_{k+1}-(a_{k+1}· e_1)e_1-\cdots-(a_{k+1}· e_k)e_k, \ \ \ e_{k+1}=\frac{u_{k+1}}{||u_{k+1}||}
$$

这里 $(a_j \cdot e_i)$ 可以解释为 $a_j$ 在 $e_i$ 上的线性最小二乘**回归系数**

* 它是 $a_j$ 和 $e_i$ 的内积除以 $e_i$ 的内积，其中 $e_i \cdot e_i = 1$，这是由*归一化*保证的。
    
* 这个回归系数可以解释为**协方差**除以**方差**

可以验证：

$$
A= \left[ \begin{array}{c|c|c|c} a_1 & a_2 & \cdots & a_n \end{array} \right]=
\left[ \begin{array}{c|c|c|c} e_1 & e_2 & \cdots & e_n \end{array} \right]
\left[ \begin{matrix} a_1·e_1 & a_2·e_1 & \cdots & a_n·e_1\\ 0 & a_2·e_2 & \cdots & a_n·e_2 
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n·e_n \end{matrix} \right]
$$

因此，我们已经构造了分解：

$$ 
A = Q R
$$

其中：

$$ 
Q = \left[ \begin{array}{c|c|c|c} a_1 & a_2 & \cdots & a_n \end{array} \right]=
\left[ \begin{array}{c|c|c|c} e_1 & e_2 & \cdots & e_n \end{array} \right]
$$

且：

$$
R = \left[ \begin{matrix} a_1·e_1 & a_2·e_1 & \cdots & a_n·e_1\\ 0 & a_2·e_2 & \cdots & a_n·e_2 
\\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_n·e_n \end{matrix} \right]
$$

### 非方阵 $A$ 

现在假设 $A$ 是一个 $n \times m$ 矩阵，其中 $m > n$。

那么QR分解为：

$$
A= \left[ \begin{array}{c|c|c|c} a_1 & a_2 & \cdots & a_m \end{array} \right]=\left[ \begin{array}{c|c|c|c} e_1 & e_2 & \cdots & e_n \end{array} \right]
\left[ \begin{matrix} a_1·e_1 & a_2·e_1 & \cdots & a_n·e_1 & a_{n+1}\cdot e_1 & \cdots & a_{m}\cdot e_1 \\
0 & a_2·e_2 & \cdots & a_n·e_2 & a_{n+1}\cdot e_2 & \cdots & a_{m}\cdot e_2 \\ \vdots & \vdots & \ddots & \quad  \vdots & \vdots & \ddots & \vdots
\\ 0 & 0 & \cdots & a_n·e_n & a_{n+1}\cdot e_n & \cdots & a_{m}\cdot e_n \end{matrix} \right]
$$

这意味着：

\begin{align*}
a_1 & = (a_1\cdot e_1) e_1 \cr
a_2 & = (a_2\cdot e_1) e_1 + (a_2\cdot e_2) e_2 \cr
\vdots & \quad \vdots \cr
a_n & = (a_n\cdot e_1) e_1 + (a_n\cdot e_2) e_2 + \cdots + (a_n \cdot e_n) e_n  \cr
a_{n+1} & = (a_{n+1}\cdot e_1) e_1 + (a_{n+1}\cdot e_2) e_2 + \cdots + (a_{n+1}\cdot e_n) e_n  \cr
\vdots & \quad \vdots \cr
a_m & = (a_m\cdot e_1) e_1 + (a_m\cdot e_2) e_2 + \cdots + (a_m \cdot e_n) e_n  \cr
\end{align*}

## 代码实现

现在让我们编写一些自制的Python代码，通过上述的Gram-Schmidt过程来实现QR分解。

```{code-cell} ipython3
import numpy as np
from scipy.linalg import qr
```

```{code-cell} ipython3
def QR_Decomposition(A):
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

上述代码很好，但可以进行一些进一步的整理。

我们这样做是因为在本笔记本的后面，我们想要比较使用我们上面的自制代码和Python `scipy` 包提供的QR代码的结果。

不同的数值算法产生的 $Q$ 和 $R$ 矩阵之间可能存在符号差异。

由于符号差异在计算 $QR$ 时会相互抵消，所以这些都是有效的QR分解。

然而，为了使我们自制函数和 `scipy` 中的QR模块的结果具有可比性，让我们要求 $Q$ 具有正对角线元素。

我们通过适当调整 $Q$ 中列和 $R$ 中行的符号来实现这一点。

为此，我们将定义一对函数。

```{code-cell} ipython3
def diag_sign(A):
    "计算矩阵A对角线元素的符号"

    D = np.diag(np.sign(np.diag(A)))

    return D

def adjust_sign(Q, R):
    """
    调整Q中列和R中行的符号，
    以使Q的对角线为正
    """

    D = diag_sign(Q)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R
```

## 示例

现在让我们看一个例子。

```{code-cell} ipython3
A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0]])

A
```

```{code-cell} ipython3
Q, R = adjust_sign(*QR_Decomposition(A))
```

```{code-cell} ipython3
Q
```

```{code-cell} ipython3
R
```

让我们将结果与`scipy`包的结果进行比较：

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
```

```{code-cell} ipython3
print('我们的Q: \n', Q)
print('\n')
print('Scipy的Q: \n', Q_scipy)
```

```{code-cell} ipython3
print('我们的R: \n', R)
print('\n')
print('Scipy的R: \n', R_scipy)
```

上述结果表明我们自制的函数与scipy的结果一致，这是个好消息。

现在让我们对一个 $n \times m$ 且 $m > n$ 的矩形矩阵 $A$ 进行QR分解。

```{code-cell} ipython3
A = np.array([[1, 3, 4], [2, 0, 9]])
```

```{code-cell} ipython3
Q, R = adjust_sign(*QR_Decomposition(A))
Q, R
```

```{code-cell} ipython3
Q_scipy, R_scipy = adjust_sign(*qr(A))
Q_scipy, R_scipy
```

## 使用QR分解计算特征值

现在介绍一个关于QR算法的有用事实。

以下QR分解的迭代可用于计算**方阵** $A$ 的**特征值**。

算法如下：

1. 设 $A_0 = A$ 并形成 $A_0 = Q_0 R_0$

2. 形成 $A_1 = R_0 Q_0$。注意 $A_1$ 与 $A_0$ 相似（容易验证），因此具有相同的特征值。

3. 形成 $A_1 = Q_1 R_1$（即，形成 $A_1$ 的 $QR$ 分解）。

4. 形成 $A_2 = R_1 Q_1$ 然后 $A_2 = Q_2 R_2$。

5. 迭代直至收敛。

6. 计算 $A$ 的特征值并将其与从此过程得到的极限 $A_n$ 的对角值进行比较。

```{todo}
@mmcky 将此迁移到使用[sphinx-proof](https://sphinx-proof.readthedocs.io/en/latest/syntax.html#algorithms)
```

**注意：**这个算法接近于计算特征值最有效的方法之一！

让我们编写一些Python代码来尝试这个算法：

```{code-cell} ipython3
def QR_eigvals(A, tol=1e-12, maxiter=1000):
    "使用QR分解找到A的特征值。"

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = QR_Decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals
```

现在让我们尝试这段代码并将结果与`scipy.linalg.eigvals`的结果进行比较：

```{code-cell} ipython3
# 用一个随机A矩阵进行实验
A = np.random.random((3, 3))
```

```{code-cell} ipython3
sorted(QR_eigvals(A))
```

与`scipy`包进行比较：

```{code-cell} ipython3
sorted(np.linalg.eigvals(A))
```

## QR与PCA

QR分解与主成分分析(PCA)之间有一些有趣的联系。

以下是一些联系：

1. 设 $X'$ 是一个 $k \times n$ 随机矩阵，其中第 $j$ 列是从 ${\mathcal N}(\mu, \Sigma)$ 中随机抽取的，这里 $\mu$ 是 $k \times 1$ 均值向量，$\Sigma$ 是 $k \times k$ 协方差矩阵。我们希望 $n > > k$ -- 这是一个"计量经济学示例"。

2. 形成 $X' = Q R$，其中 $Q$ 是 $k \times k$ 且 $R$ 是 $k \times n$。

3. 形成 $R R'$ 的特征值，即我们将计算 $R R' = \tilde P \Lambda \tilde P'$。

4. 形成 $X' X = Q \tilde P \Lambda \tilde P' Q'$ 并将其与特征分解 $X'X = P \hat \Lambda P'$ 进行比较。

5. 将发现 $\Lambda = \hat \Lambda$ 且 $P = Q \tilde P$。

让我们用一些Python代码来验证猜想5。

首先模拟一个随机的 $\left(n, k\right)$ 矩阵 $X$。

```{code-cell} ipython3
k = 5
n = 1000

# 生成一些随机矩
𝜇 = np.random.random(size=k)
C = np.random.random((k, k))
Σ = C.T @ C
```

```{code-cell} ipython3
# X是一个随机矩阵，其中每列遵循多元正态分布
X = np.random.multivariate_normal(𝜇, Σ, size=n)
```

```{code-cell} ipython3
X.shape
```

让我们对 $X^{\prime}$ 应用QR分解。

```{code-cell} ipython3
Q, R = adjust_sign(*QR_Decomposition(X.T))
```

检查 $Q$ 和 $R$ 的形状。

```{code-cell} ipython3
Q.shape, R.shape
```

现在我们可以构造 $R R^{\prime}=\tilde{P} \Lambda \tilde{P}^{\prime}$ 并进行特征分解。

```{code-cell} ipython3
RR = R @ R.T

𝜆, P_tilde = np.linalg.eigh(RR)
Λ = np.diag(𝜆)
```

我们也可以对 $X^{\prime} X=P \hat{\Lambda} P^{\prime}$ 应用分解。

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

同样，我们需要注意 $P$ 和 $Q\tilde{P}$ 列之间的符号差异。

```{code-cell} ipython3
QP_tilde = Q @ P_tilde

np.abs(P @ diag_sign(P) - QP_tilde @ diag_sign(QP_tilde)).max()
```

让我们验证 $X^{\prime}X$ 可以分解为 $Q \tilde{P} \Lambda \tilde{P}^{\prime} Q^{\prime}$。

```{code-cell} ipython3
QPΛPQ = Q @ P_tilde @ Λ @ P_tilde.T @ Q.T
```

```{code-cell} ipython3
np.abs(QPΛPQ - XX).max()
```