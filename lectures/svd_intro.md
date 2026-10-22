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
  title: 奇异值分解
  headings:
    Overview: 概述
    The setting: 基本设定
    Singular value decomposition: 奇异值分解
    Four fundamental subspaces: 四个基本子空间
    Eckart-Young theorem: Eckart-Young定理
    Full and reduced SVD's: 完全SVD和简化SVD
    Polar decomposition: 极分解
    'Application: principal components analysis (PCA)': 应用：主成分分析(PCA)
    Relationship of PCA to SVD: PCA与SVD的关系
    PCA with eigenvalues and eigenvectors: 基于特征值和特征向量的PCA
    Connections: 联系
    Exercises: 练习
---

# 奇异值分解

## 概述

**奇异值分解**（SVD）是一个强大的数学工具，在数据分析和机器学习中有着广泛的应用。它不仅可以用于最小二乘投影，还是许多统计和机器学习方法的基石。

在这一讲中，我们将首先介绍SVD的基本概念，然后探讨它与以下几个重要领域的关系：

* 线性代数中的**四个基本子空间**
* 矩阵的**低秩近似**
* **最小二乘回归**（包括欠定和超定情况，参见练习部分）
* **主成分分析**（PCA）

在这一讲的后续内容 {doc}`VAR 与 DMD <var_dmd>` 中，我们将介绍如何利用 SVD，借助**动态模态分解**（DMD）来快速计算一阶向量自回归（VAR）模型的低秩近似。

与主成分分析类似，动态模态分解（DMD）也可以被看作是一种数据降维方法，它通过将数据投影到一组有限的因子上来表示数据中的显著模式。

## 基本设定

假设我们有一个$m \times n$的矩阵$X$，其秩为$p$。

显然，$p$不会超过$m$和$n$的较小值，即$p \leq \min(m,n)$。

在实际应用中，$X$通常代表一个**数据矩阵**：

* 每一列代表一个**观测单位**（可以是某个时间点的数据，或是某个个体的信息）
* 每一行代表一个**变量**（用来描述观测单位的某个特征或属性）


我们将关注两种情况：

* **矮胖**情况，即$m \ll n$，表示列数（个体）远多于行数（属性）。
* **高瘦**情况，即$m \gg n$，表示行数（属性）远多于列数（个体）。


我们将在这两种情况下对$X$进行**奇异值分解**。

在 $m \ll n$ 的情况下，即个体数量 $n$ 远大于属性数量 $m$ 时，我们可以通过对观测值函数取平均来计算联合分布的样本矩。

在这种 $m \ll n$ 的情况下，我们将使用**奇异值分解**来进行**主成分分析**（PCA）以寻找**模式**。

在 $m \gg n$ 的情况下，即属性数量 $m$ 远大于个体数量 $n$，且在时间序列环境中 $n$ 等于数据集 $X$ 中所覆盖的时间段数量时，我们将采用不同的方法。

在后续的 {doc}`VARs 和 DMDs <var_dmd>` 中，我们仍然使用奇异值分解，但目的是构建动态模态分解（DMD）。

## 奇异值分解

一个秩为 $p \leq \min(m,n)$ 的 $m \times n$ 矩阵 $X$ 的**奇异值分解**为：

$$
X  = U \Sigma V^\top
$$ (eq:SVD101)

其中：

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

且

* $U$ 是 $X$ 的 $m \times m$ 正交矩阵，由**左奇异向量**组成
* $U$ 的列是 $X X^\top $ 的特征向量
* $V$ 是 $X$ 的 $n \times n$ 正交矩阵，由**右奇异向量**组成
* $V$ 的列是 $X^\top  X$ 的特征向量
* $\Sigma$ 是一个 $m \times n$ 矩阵，其主对角线上的前 $p$ 个位置是正数 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p$，称为**奇异值**；$\Sigma$ 的其余元素都为零

* 这 $p$ 个奇异值是 $m \times m$ 矩阵 $X X^\top $ 以及 $n \times n$ 矩阵 $X^\top  X$ 的非零特征值的正平方根

* 我们约定，当 $U$ 是复值矩阵时，$U^\top $ 表示 $U$ 的**共轭转置**或**厄米特转置**，即 $U_{ij}^\top $ 是 $U_{ji}$ 的复共轭。

* 类似地，当 $V$ 是复值矩阵时，$V^\top$ 表示 $V$ 的**共轭转置**或**厄米特转置**

矩阵 $U,\Sigma,V$ 通过以下方式对向量进行线性变换：

* 用酉矩阵 $U$ 和 $V$ 乘以向量会使其**旋转**，但保持**向量之间的角度**和**向量的长度**不变。
* 用对角矩阵 $\Sigma$ 乘以向量会按每个奇异值**重新缩放**各个坐标（当 $m \neq n$ 时还会增加或舍弃坐标）；一般来说，这既会改变向量的长度，也会改变向量之间的角度。

因此，表示式 {eq}`eq:SVD101` 表明，用 $m \times n$ 矩阵 $X$ 乘以 $n \times 1$ 向量 $y$ 相当于按顺序执行以下三个乘法运算：

* 通过计算 $V^\top y$ 来**旋转** $y$
* 通过乘以 $\Sigma$ 来**重新缩放** $V^\top y$
* 通过乘以 $U$ 来**旋转** $\Sigma V^\top y$

$m \times n$ 矩阵 $X$ 的这种结构为构建系统开启了大门

数据**编码器**和**解码器**。

因此，

* $V^\top$ 是一个编码器
* $\Sigma$ 是一个应用于编码数据的运算符
* $U$ 是一个解码器，用于处理将运算符 $\Sigma$ 应用于编码数据后的输出

我们将在后续的 {doc}`VARs and DMDs <var_dmd>` 中研究动态模态分解时应用这些概念。

**未来路线**

我们上面描述的是所谓的**完全** SVD。

在**完全** SVD 中，$U$、$\Sigma$ 和 $V$ 的形状分别为 $\left(m, m\right)$、$\left(m, n\right)$、$\left(n, n\right)$。

稍后我们还将描述**经济型**或**简化** SVD。

在研究**简化** SVD之前，我们将进一步讨论**完全** SVD的性质。

## 四个基本子空间

让 ${\mathcal C}$ 表示列空间，${\mathcal N}$ 表示零空间，${\mathcal R}$ 表示行空间。

让我们首先回顾一下秩为 $p$ 的 $m \times n$ 矩阵 $X$ 的四个基本子空间。

* **列空间**$X$，记作${\mathcal C}(X)$，是$X$的列向量的张成空间，即所有可以写成$X$的列向量的线性组合的向量$y$。其维数为$p$。
* **零空间**$X$，记作${\mathcal N}(X)$，包含所有满足$Xy=0$的向量$y$。其维数为$n-p$。
* **行空间**$X$，记作${\mathcal R}(X)$，是$X^\top$的列空间。它包含所有可以写成$X$的行向量的线性组合的向量$z$。其维数为$p$。
* **左零空间**$X$，记作${\mathcal N}(X^\top)$，包含所有满足$X^\top z=0$的向量$z$。其维数为$m-p$。

对于矩阵$X$的完全奇异值分解，左奇异向量矩阵$U$和右奇异向量矩阵$V$包含了所有四个子空间的正交基。

它们形成两对正交子空间，我们现在来描述。

令$u_i, i = 1, \ldots, m$为$U$的$m$个列向量，令

设 $v_i, i = 1, \ldots, n$ 为 $V$ 的 $n$ 个列向量。

让我们将 $X$ 的完整奇异值分解写作

$$
X = \begin{bmatrix} U_L & U_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
     \begin{bmatrix} V_L & V_R \end{bmatrix}^\top
$$ (eq:fullSVDpartition)

其中 $\Sigma_p$ 是一个 $p \times p$ 对角矩阵，对角线上是 $p$ 个奇异值，且

$$
\begin{aligned}
U_L & = \begin{bmatrix}u_1 & \cdots  & u_p \end{bmatrix},  \quad U_R  = \begin{bmatrix}u_{p+1} & \cdots & u_m \end{bmatrix}  \cr
V_L & = \begin{bmatrix}v_1 & \cdots  & v_p \end{bmatrix} , \quad V_R  = \begin{bmatrix}v_{p+1} & \cdots & v_n \end{bmatrix}
\end{aligned}
$$

表示式 {eq}`eq:fullSVDpartition` 意味着

$$
X \begin{bmatrix} V_L & V_R \end{bmatrix} = \begin{bmatrix} U_L & U_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
$$

或

$$
\begin{aligned}
X V_L & = U_L \Sigma_p \cr
X V_R & = 0
\end{aligned}
$$ (eq:Xfour1a)

或

$$
\begin{aligned}
X v_i & = \sigma_i u_i , \quad i = 1, \ldots, p \cr
X v_i & = 0 ,  \quad i = p+1, \ldots, n
\end{aligned}
$$ (eq:orthoortho1)

方程 {eq}`eq:orthoortho1` 说明了变换 $X$ 如何将一对正交单位向量 $v_i, v_j$（其中 $i$ 和 $j$ 都小于或等于 $X$ 的秩 $p$）映射到一对长度分别为 $\sigma_i$ 和 $\sigma_j$ 的正交向量 $\sigma_i u_i, \sigma_j u_j$。

方程 {eq}`eq:Xfour1a` 表明

$$
\begin{aligned}
{\mathcal C}(X) & = {\mathcal C}(U_L) \cr
{\mathcal N}(X) & = {\mathcal C} (V_R)
\end{aligned}
$$

对表示式 {eq}`eq:fullSVDpartition` 两边取转置得到

$$
X^\top  \begin{bmatrix} U_L & U_R \end{bmatrix} = \begin{bmatrix} V_L & V_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
$$

或

$$
\begin{aligned}
X^\top  U_L & = V_L \Sigma_p \cr
X^\top  U_R & = 0
\end{aligned}
$$  (eq:Xfour1b)

或

$$
\begin{aligned}
X^\top  u_i & = \sigma_i v_i, \quad i=1, \ldots, p \cr
X^\top  u_i & = 0 \quad i= p+1, \ldots, m
\end{aligned}
$$ (eq:orthoortho2)

注意方程 {eq}`eq:orthoortho2` 表明变换 $X^\top$ 将一对不同的正交单位向量 $u_i, u_j$（其中 $i$ 和 $j$ 都小于或等于 $X$ 的秩 $p$）映射到一对正交向量 $\sigma_i v_i, \sigma_j v_j$。

方程 {eq}`eq:Xfour1b` 表明：

$$
\begin{aligned}
{\mathcal R}(X) & \equiv  {\mathcal C}(X^\top ) = {\mathcal C} (V_L) \cr
{\mathcal N}(X^\top ) & = {\mathcal C}(U_R)
\end{aligned}
$$

因此，方程组 {eq}`eq:Xfour1a` 和 {eq}`eq:Xfour1b` 共同描述了 $X$ 的四个基本子空间，如下所示：

$$
\begin{aligned}
{\mathcal C}(X) & = {\mathcal C}(U_L) \cr
{\mathcal N}(X^\top ) & = {\mathcal C}(U_R) \cr
{\mathcal R}(X) & \equiv  {\mathcal C}(X^\top ) = {\mathcal C} (V_L) \cr
{\mathcal N}(X) & = {\mathcal C} (V_R)
\end{aligned}
$$ (eq:fourspaceSVD)

由于 $U$ 和 $V$ 都是正交矩阵，集合 {eq}`eq:fourspaceSVD` 表明

* $U_L$ 是 $X$ 列空间的标准正交基
* $U_R$ 是 $X^\top$ 零空间的标准正交基
* $V_L$ 是 $X$ 行空间的标准正交基
* $V_R$ 是 $X$ 零空间的标准正交基

我们通过执行 {eq}`eq:fullSVDpartition` 右侧要求的乘法并读取结果，已经验证了 {eq}`eq:fourspaceSVD` 中的四个声明。

{eq}`eq:fourspaceSVD` 中的声明以及 $U$ 和 $V$ 都是酉矩阵（即正交矩阵）这一事实意味着：

* $X$ 的列空间与 $X^\top$ 的零空间正交
* $X$ 的零空间与 $X$ 的行空间正交

这些性质有时用以下两对正交补空间来描述：

* ${\mathcal C}(X)$ 是 ${\mathcal N}(X^\top )$ 的正交补
* ${\mathcal R}(X)$ 是 ${\mathcal N}(X)$ 的正交补

让我们看一个例子。

```{code-cell} ipython3
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

rng = np.random.default_rng(1234)
```

导入这些模块后，让我们来看示例。

我们使用一个秩为 $2$ 的 $4 \times 5$ 矩阵，因此四个子空间的维数分别为 $2$、$2$、$2$ 和 $3$。

注意 `np.linalg.svd` 返回的第三个输出是 $V^\top$，而不是 $V$。

因此行空间和零空间是由该输出的行所张成的。

```{code-cell} ipython3
np.set_printoptions(precision=2, suppress=True)

# 定义矩阵
A = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8]])

# 计算矩阵的奇异值分解；第三个输出是 V^T
U, S, VT = np.linalg.svd(A, full_matrices=True)

# 计算矩阵的秩
rank = np.linalg.matrix_rank(A)

print("矩阵的秩:\n", rank)
print("S: \n", S)

# 四个基本子空间的标准正交基
col_space = U[:, :rank]            # C(A)，R^4 的子空间
left_null_space = U[:, rank:]      # N(A^T)，R^4 的子空间
row_space = VT[:rank, :].T         # R(A)，R^5 的子空间
null_space = VT[rank:, :].T        # N(A)，R^5 的子空间

print("列空间:\n", col_space)
print("左零空间:\n", left_null_space)
print("行空间:\n", row_space)
print("零空间:\n", null_space)
```

让我们验证这些基是否具有 {eq}`eq:fourspaceSVD` 所断言的性质。

```{code-cell} ipython3
print("A @ null_space = 0:        ", np.allclose(A @ null_space, 0))
print("A.T @ left_null_space = 0: ", np.allclose(A.T @ left_null_space, 0))
print("col_space ⟂ left_null_space:", np.allclose(col_space.T @ left_null_space, 0))
print("row_space ⟂ null_space:    ", np.allclose(row_space.T @ null_space, 0))
```

## Eckart-Young定理

假设我们要构造一个$m \times n$矩阵$X$的最佳秩$r$近似。

这里的最佳，指的是在所有维度为$m \times n$的秩为$r < p$的矩阵中，找到一个矩阵$X_r$使得以下范数最小：

$$
|| X - X_r ||
$$

其中$|| \cdot ||$表示矩阵的范数。

一个$m \times n$矩阵$X$的三种常用**矩阵范数**可以用$X$的奇异值表示：

* **谱范数**或$l^2$范数 $|| X ||_2 = \max_{||y|| \neq 0} \frac{||X y ||}{||y||} = \sigma_1$
* **Frobenius范数** $||X ||_F = \sqrt{\sigma_1^2 + \cdots + \sigma_p^2}$
* **核范数** $ || X ||_N = \sigma_1 + \cdots + \sigma_p $

Eckart-Young定理指出，对于这三种范数，最佳的秩$r$矩阵是相同的，等于：

$$
\hat X_r = \sigma_1 u_1 v_1^\top  + \sigma_2 u_2 v_2^\top  + \cdots + \sigma_r u_r v_r^\top = U_r \Sigma_r V_r^\top
$$ (eq:Ekart)

其中$U_r$和$V_r$分别由$U$和$V$的前$r$列组成，$\Sigma_r$是由$r$个最大奇异值构成的$r \times r$对角矩阵。

由此产生的近似误差为

$$
\begin{aligned}
|| X - \hat X_r ||_2 & = \sigma_{r+1} \cr
|| X - \hat X_r ||_F & = \sqrt{\sigma_{r+1}^2 + \cdots + \sigma_p^2} \cr
|| X - \hat X_r ||_N & = \sigma_{r+1} + \cdots + \sigma_p
\end{aligned}
$$

这是一个非常强大的定理。

它表明，我们可以用由$X$的$r$个最大奇异值及其奇异向量构造的秩$r < p$的$m \times n$矩阵，来近似一个秩为$p$的$m \times n$矩阵$X$，而我们所舍弃的奇异值则衡量了近似误差的大小。

如果少数几个奇异值远大于其他奇异值，那么低秩近似就能捕捉到$X$的大部分信息，同时我们只需存储$r(m + n + 1)$个数字，而不是$mn$个。

你可以在 [这里](https://en.wikipedia.org/wiki/Low-rank_approximation) 阅读关于Eckart-Young定理及其应用的内容。

在下面讨论主成分分析（PCA）时，我们将会用到这个定理。

它同样构成了后续内容 {doc}`VAR与DMD <var_dmd>` 的基础，其中近似矩阵$\hat X_r$的伪逆

$$
\hat X_r^+ = V_r \Sigma_r^{-1} U_r^\top
$$

是动态模态分解的一个关键组成部分。

{ref}`svd_ex2` 和 {ref}`svd_ex3` 探讨了这些思想。

## 完全SVD和简化SVD

到目前为止，我们描述的是**完全**SVD的性质，其中 $U$、$\Sigma$ 和 $V$ 的形状分别为 $\left(m, m\right)$、$\left(m, n\right)$、$\left(n, n\right)$。

有一种替代性的矩阵分解记法，称为**经济型**或**简化型** SVD，其中 $U, \Sigma$ 和 $V$ 的形状与完全SVD中的不同。

因为我们假设 $X$ 的秩为 $p$，所以只有 $p$ 个非零奇异值，其中 $p=\textrm{rank}(X)\leq\min\left(m, n\right)$。

**简化型** SVD利用这一事实，将 $U$、$\Sigma$ 和 $V$ 表示为形状分别为 $\left(m, p\right)$、$\left(p, p\right)$、$\left(n, p\right)$ 的矩阵。

你可以在这里了解简化型和完全型SVD
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>

（使用选项 `full_matrices=False` 时，`numpy` 会返回 $U$ 和 $V$ 的 $\min(m, n)$ 列；当 $p < \min(m,n)$ 时，我们会去掉对应于零奇异值的列，从而得到这里所指的简化型SVD。）

对于完全型SVD，

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

但这些性质并非都适用于**简化型** SVD。

对于简化型SVD，$U^\top U = I_{p \times p}$ 和 $V^\top V = I_{p \times p}$ 总是成立，但只有当 $p = m$ 时 $U U^\top = I_{m \times m}$ 才成立，只有当 $p = n$ 时 $V V^\top = I_{n \times n}$ 才成立。

因此，哪些性质成立取决于我们处理的是**高瘦型**矩阵还是**矮胖型**矩阵。

* 在**高瘦型**情况下，即 $m \gg n$ 且 $X$ 具有满列秩 $p = n$，对于**简化型** SVD

$$
\begin{aligned}
UU^\top  &  \neq I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

* 在**矮胖型**情况下，即 $m \ll n$ 且 $X$ 具有满行秩 $p = m$，对于**简化型** SVD

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & \neq I & \quad V^\top  V = I
\end{aligned}
$$

当我们在后续 {doc}`VARs and DMDs <var_dmd>` 中研究动态模态分解时，我们需要记住这些性质，因为在高瘦型情况下我们会使用简化SVD。

让我们做一个例子来比较**完全**和**简化**SVD。

回顾一下，

* 在**完全**SVD中

  - $U$ 是 $m \times m$ 维
  - $\Sigma$ 是 $m \times n$ 维
  - $V$ 是 $n \times n$ 维

* 在**简化**SVD中

  - $U$ 是 $m \times p$ 维
  - $\Sigma$ 是 $p \times p$ 维
  - $V$ 是 $n \times p$ 维

首先，让我们研究一个 $m = 5 > n = 2$ 的情况。

（这是我们在后续研究动态模态分解时会遇到的**高瘦**情况的一个小例子。）

```{code-cell} ipython3
X = rng.random((5, 2))
U, S, V = np.linalg.svd(X, full_matrices=True)  # 完整SVD
Uhat, Shat, Vhat = np.linalg.svd(X, full_matrices=False) # 简化SVD
print('U, S, V =')
U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = ')
Uhat, Shat, Vhat
```

```{code-cell} ipython3
rr = np.linalg.matrix_rank(X)
print(f'X的秩 = {rr}')
```

**性质：**

* 当$U$通过完全SVD构造时，$U^\top U = I_{m\times m}$ 且 $U U^\top = I_{m \times m}$
* 当$\hat U$通过简化SVD构造时，虽然$\hat U^\top \hat U = I_{p\times p}$，但$\hat U \hat U^\top \neq I_{m \times m}$

我们通过以下代码单元来说明这些性质。

```{code-cell} ipython3
UTU = U.T@U
UUT = U@U.T
print('UUT, UTU = ')
UUT, UTU
```

```{code-cell} ipython3
UhatUhatT = Uhat@Uhat.T
UhatTUhat = Uhat.T@Uhat
print('UhatUhatT, UhatTUhat= ')
UhatUhatT, UhatTUhat
```

**注释：**

上述代码展示了 `full_matrices=True` 和 `full_matrices=False` 选项的应用。

使用 `full_matrices=False` 会返回简化的奇异值分解。

**完整**和**简化**的奇异值分解都能准确地分解一个 $m \times n$ 矩阵 $X$。

当我们在后续研究动态模态分解时，记住在这种高瘦矩阵情况下完整和简化奇异值分解的上述性质将很重要。

现在让我们来看一个矮胖矩阵的情况。

为了说明这种情况，我们将设置 $m = 2 < 5 = n$，并计算完整和简化的奇异值分解。

```{code-cell} ipython3
X = rng.random((2, 5))
U, S, V = np.linalg.svd(X, full_matrices=True)  # 完整SVD
Uhat, Shat, Vhat = np.linalg.svd(X, full_matrices=False) # 简化SVD
print('U, S, V = ')
U, S, V
```

```{code-cell} ipython3
print('Uhat, Shat, Vhat = ')
Uhat, Shat, Vhat
```

让我们验证我们的简化SVD是否准确表示$X$

```{code-cell} ipython3
SShat=np.diag(Shat)
np.allclose(X, Uhat@SShat@Vhat)
```

## 极分解

矩阵 $X$ 的**简化**奇异值分解与其**极分解**相关

$$
X = SQ
$$

其中

$$
\begin{aligned}
S & = U\Sigma U^\top \cr
Q & = UV^\top
\end{aligned}
$$

在我们的简化SVD中

* $U$ 是一个 $m \times p$ 正交矩阵
* $\Sigma$ 是一个 $p \times p$ 对角矩阵
* $V$ 是一个 $n \times p$ 正交矩阵

由于 $U^\top U = I$，我们有 $SQ = U \Sigma U^\top U V^\top = U \Sigma V^\top = X$。

这里

* $S$ 是一个 $m \times m$ **对称半正定**矩阵
* $Q$ 是一个 $m \times n$ 矩阵，满足 $Q Q^\top = U U^\top$；因此当 $p = m$ 时，例如当 $X$ 是具有满行秩的矮胖矩阵时，$Q Q^\top = I$，此时 $Q$ 具有正交的行

## 应用：主成分分析(PCA)

让我们从 $n \gg m$ 的情况开始，即个体数量 $n$ 远大于属性数量 $m$ 的情况。

在 $n \gg m$ 的情况下，矩阵 $X$ 是**矮胖型**的，这与后文要讨论的 $m \gg n$ 情况下的**高瘦型**相对。

我们将 $X$ 视为一个 $m \times n$ 的**数据**矩阵：

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

其中对于 $j = 1, \ldots, n$，列向量 $X_j = \begin{bmatrix}x_{1j}\\x_{2j}\\\vdots\\x_{mj}\end{bmatrix}$ 是变量 $1, 2, \ldots, m$ 的观测值向量。

在**时间序列**分析中，列索引 $j$ 代表不同的时间点，而行索引代表不同的随机变量。

在**横截面**分析中，列索引 $j$ 代表不同的个体，而行索引代表它们的不同属性。

正如我们之前所见，SVD 是一种将矩阵分解为有用组件的方法，就像极分解、特征分解等其他方法一样。

而 PCA 则是一种基于 SVD 进行数据分析的方法。

其目标是找到变量的若干线性组合，以捕捉数据中最重要的变化模式。

**第1步：数据中心化：**

由于我们关注的是数据围绕其均值的变化，因此我们首先减去样本均值。

（由于数据矩阵可能包含不同单位和尺度的变量，我们也可以通过将每一行除以其样本标准差来对数据进行**标准化**。）

我们首先计算 $X$ 的每一行的平均值

$$
\bar{X_i}= \frac{1}{n} \sum_{j = 1}^{n} x_{ij}
$$

然后用这些平均值创建一个矩阵：

$$
\bar{X} =  \begin{bmatrix} \bar{X_1} \\ \bar{X_2} \\ \vdots \\ \bar{X_m}\end{bmatrix}\begin{bmatrix}1 \mid 1 \mid \cdots \mid 1 \end{bmatrix}
$$

然后从原始矩阵中减去它，得到一个均值中心化矩阵：

$$
B = X - \bar{X}
$$

**第2步：计算协方差矩阵：**

由于我们希望提取变量之间的关系，而不仅仅是它们的大小——换句话说，我们想知道它们能在多大程度上相互解释——因此我们计算 $B$ 的样本协方差矩阵。

$$
C = \frac{1}{n} BB^{\top}
$$

**第3步：分解协方差矩阵并排列奇异值：**

由于矩阵 $C$ 是对称且半正定的，我们可以对其进行特征分解，找出其特征值，并将特征值和特征向量矩阵按降序排列。

$C$ 的特征分解可以通过分解 $B$ 来求得。由于 $B$ 不是方阵，我们对 $B$ 进行 SVD 分解：

$$
\begin{aligned}
B B^\top &= U \Sigma V^\top (U \Sigma V^{\top})^{\top}\\
&= U \Sigma V^\top V \Sigma^\top U^\top\\
&= U \Sigma \Sigma^\top U^\top
\end{aligned}
$$

因此

$$
C = \frac{1}{n} U \Sigma \Sigma^\top U^\top
$$

按照惯例，奇异值按降序排列，这也是 `numpy` 的做法。

**第4步：选择奇异值，（可选）截断其余部分：**

我们现在可以根据想要保留的方差量来决定保留多少个奇异值（例如，保留95%的总方差）。

我们可以通过计算前 $r$ 个主要因子所包含的方差除以总方差来获得这个百分比：

$$
\frac{\sum_{i = 1}^{r} \sigma^2_{i}}{\sum_{i = 1}^{p} \sigma^2_{i}}
$$

根据埃克特-杨定理（Eckart-Young theorem），$1$ 减去这个比率，也等于 $B$ 的最佳秩 $r$ 近似 $\hat B_r$ 的平方相对弗罗贝尼乌斯范数误差 $||B - \hat B_r||_F^2 / ||B||_F^2$。

**第5步：创建得分矩阵：**

**主成分**（或**得分**）矩阵为

$$
\begin{aligned}
T & = U^\top B \cr
& = U^\top U \Sigma V^\top \cr
& = \Sigma V^\top
\end{aligned}
$$

它的第 $k$ 行是第 $k$ 个主成分在各观测值 $j = 1, \ldots, n$ 上构成的序列。

只保留前 $r$ 行，就得到了前 $r$ 个主成分构成的 $r \times n$ 矩阵 $T_r = U_r^\top B$，而 $U_r T_r = U_r \Sigma_r V_r^\top = \hat B_r$ 就是 $B$ 的埃克特-杨近似。

在后文 {doc}`VAR 与 DMD <var_dmd>` 中，时刻 $t$ 横截面的前 $r$ 个主成分所构成的向量 $U_r^\top X_t$，将作为**主成分回归**中的回归变量，用于预测下一期的横截面数据。

## PCA与SVD的关系

为了将SVD与数据集$X$的PCA联系起来，首先构造数据矩阵$X$的SVD。

让我们假设所有变量的样本均值都为零，这样我们就不需要对矩阵进行中心化了。

$$
X = U \Sigma V^\top  = \sigma_1 U_1 V_1^\top  + \sigma_2 U_2 V_2^\top  + \cdots + \sigma_p U_p V_p^\top
$$ (eq:PCA1)

其中

$$
U=\begin{bmatrix}U_1|U_2|\ldots|U_m\end{bmatrix}
$$

$$
V^\top  = \begin{bmatrix}V_1^\top \\V_2^\top \\\vdots\\V_n^\top \end{bmatrix}
$$

在方程{eq}`eq:PCA1`中，每个$m \times n$矩阵$U_{j}V_{j}^\top $显然是秩1的。

因此，我们有

$$
X = \sigma_1 \begin{bmatrix}U_{11}V_{1}^\top \\U_{21}V_{1}^\top \\\cdots\\U_{m1}V_{1}^\top \\\end{bmatrix} + \sigma_2\begin{bmatrix}U_{12}V_{2}^\top \\U_{22}V_{2}^\top \\\cdots\\U_{m2}V_{2}^\top \\\end{bmatrix}+\ldots + \sigma_p\begin{bmatrix}U_{1p}V_{p}^\top \\U_{2p}V_{p}^\top \\\cdots\\U_{mp}V_{p}^\top \\\end{bmatrix}
$$ (eq:PCA2)

在时间序列分析的背景下，我们可以这样解释方程{eq}`eq:PCA2`中的对象：

* 对于每个$k=1, \ldots, p$，对象 $\lbrace V_{jk} \rbrace_{j=1}^n$（即$V$的第$k$列）是第$k$个**主成分**的时间序列，经过归一化使其具有单位长度；未归一化的主成分为$\sigma_k V_k^\top = U_k^\top X$

* $U_k = \begin{bmatrix}U_{1k}\\U_{2k}\\\vdots\\U_{mk}\end{bmatrix}, \  k=1, \ldots, p$，
是变量$X_i$在第$k$个主成分上的**载荷**向量，其中$i=1, \ldots, m$

* 对于每个$k=1, \ldots, p$，$\sigma_k$是第$k$个**主成分**的强度，这里的强度指的是对$X$的整体协方差的贡献：第$k$个主成分对这$m$个变量样本方差之和的贡献为$\sigma_k^2 / n$

## 基于特征值和特征向量的PCA

现在我们使用样本协方差矩阵的特征分解来进行PCA。

设$X_{m \times n}$为我们的$m \times n$数据矩阵。

假设所有变量的样本均值都为零。

我们可以通过减去样本均值的**预处理**来确保这一点。

定义一个与样本协方差矩阵成比例（等于其$n$倍）的矩阵$\Omega$为

$$
\Omega = XX^\top
$$

然后使用特征分解将$\Omega$表示为：

$$
\Omega =P\Lambda P^\top
$$

这里

* $P$是$\Omega$的$m×m$正交特征向量矩阵

* $\Lambda$是$\Omega$的特征值对角矩阵

我们可以将$X$表示为:

$$
X=P\epsilon
$$

其中

$$
\epsilon = P^{-1} X = P^\top X
$$

且

$$
\epsilon\epsilon^\top =\Lambda .
$$

我们可以验证

$$
XX^\top =P\Lambda P^\top  .
$$ (eq:XXo)

因此数据矩阵$X$可以写成:

$$
X =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\vdots\\\epsilon_m\end{bmatrix}
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
$$

其中$P_j$是$P$的第$j$列，$\epsilon_j$是$\epsilon$的第$j$行，为$1 \times n$向量。

为了将前面的表示与我们之前通过SVD获得的PCA相协调，我们首先注意到$\epsilon_j \epsilon_j^\top = \lambda_j \equiv \sigma^2_j$。

现在对每个满足$\lambda_j > 0$的$j$定义$\tilde{\epsilon}_j = \frac{\epsilon_j}{\sqrt{\lambda_j}}$，
这意味着$\tilde{\epsilon}_j\tilde{\epsilon}_j^\top =1$。

当$\lambda_j = 0$时相应的项会消失，因为此时$\epsilon_j = 0$。

因此

$$
\begin{aligned}
X&=\sqrt{\lambda_1}P_1\tilde{\epsilon}_1+\sqrt{\lambda_2}P_2\tilde{\epsilon}_2+\ldots+\sqrt{\lambda_p}P_p\tilde{\epsilon}_p\\
&=\sigma_1P_1\tilde{\epsilon}_1+\sigma_2P_2\tilde{\epsilon}_2+\ldots+\sigma_pP_p\tilde{\epsilon}_p ,
\end{aligned}
$$

这与SVD分解

$$
X=\sigma_1U_1{V_1}^{\top}+\sigma_2 U_2{V_2}^{\top}+\ldots+\sigma_{p} U_{p}{V_{p}}^{\top}
$$

是等价的，只要我们令:

* $U_j=P_j$ (变量在第j个主成分上的载荷向量)

* ${V_k}^{\top}=\tilde{\epsilon}_k$ (第k个主成分)

由于对于给定的数据矩阵$X$，计算$P$和$U$有不同的算法，根据所使用的算法，我们可能会得到符号差异或特征向量顺序的不同。

我们可以通过以下方式解决关于$U$和$P$的这些歧义：

1. 将特征值和奇异值按降序排列
2. 在$P$和$U$中强制使对角线为正，并相应地调整$V^\top$中的符号

## 联系

为了把这些内容整合起来，有必要把上面给出的一些公式汇总并加以比较。

首先，考虑一个$m \times n$矩阵的SVD：

$$
X = U\Sigma V^\top
$$

计算：

$$
\begin{aligned}
XX^\top &=U\Sigma V^\top V\Sigma^\top  U^\top \cr
&= U\Sigma\Sigma^\top U^\top \cr
&\equiv U\Lambda U^\top
\end{aligned}
$$  (eq:XXcompare)

将表示式{eq}`eq:XXcompare`与上面的方程{eq}`eq:XXo`进行比较。

显然，SVD中的$U$就是$XX^\top$的特征向量矩阵$P$，而$\Sigma \Sigma^\top$就是特征值矩阵$\Lambda$。

其次，让我们计算

$$
\begin{aligned}
X^\top X &=V\Sigma^\top  U^\top U\Sigma V^\top \\
&=V\Sigma^\top {\Sigma}V^\top
\end{aligned}
$$

因此，SVD中的矩阵$V$是$X^\top X$的特征向量矩阵。

总结并整合这些内容，我们得到样本协方差矩阵的特征分解

$$
X X^\top  = P \Lambda P^\top
$$

其中$P$是一个正交矩阵。

此外，从$X$的SVD分解中，我们知道

$$
X X^\top  = U \Sigma \Sigma^\top  U^\top
$$

其中$U$是一个正交矩阵。

因此，$P = U$（在列的排序和符号上可能有所不同），我们得到$X$的表示

$$
X = P \epsilon = U \Sigma V^\top
$$

由此可得

$$
U^\top  X = \Sigma V^\top  = \epsilon
$$

因此$\epsilon$就是我们在上面第5步中构造的主成分（得分）矩阵。

注意，上述结果意味着

$$
\epsilon \epsilon^\top  = \Sigma V^\top  V \Sigma^\top  = \Sigma \Sigma^\top  = \Lambda ,
$$

因此所有内容都能够互相吻合。

下面我们定义一个`DecomAnalysis`类，用于对给定的数据矩阵`X`进行PCA和SVD分析。

```{code-cell} ipython3
class DecomAnalysis:
    """
    用于进行PCA和SVD分析的类。
    X: 数据矩阵
    r_component: 最佳近似所选择的秩
    """

    def __init__(self, X, r_component=None):

        self.X = X

        self.Ω = (X @ X.T)

        self.m, self.n = X.shape
        self.r = LA.matrix_rank(X)

        if r_component:
            self.r_component = r_component
        else:
            self.r_component = min(self.m, self.n)

    def pca(self):

        𝜆, P = LA.eigh(self.Ω)    # P的列是特征向量

        ind = sorted(range(𝜆.size), key=lambda x: 𝜆[x], reverse=True)

        # 按特征值排序
        self.𝜆 = 𝜆[ind]
        P = P[:, ind]
        self.P = P @ diag_sign(P)

        self.Λ = np.diag(self.𝜆)

        self.explained_ratio_pca = np.cumsum(self.𝜆) / self.𝜆.sum()

        # 计算m乘n的主成分矩阵
        self.𝜖 = self.P.T @ self.X

        P = self.P[:, :self.r_component]
        𝜖 = self.𝜖[:self.r_component, :]

        # 转换数据
        self.X_pca = P @ 𝜖

    def svd(self):

        U, 𝜎, VT = LA.svd(self.X)

        ind = sorted(range(𝜎.size), key=lambda x: 𝜎[x], reverse=True)

        # 按特征值排序
        d = min(self.m, self.n)

        self.𝜎 = 𝜎[ind]
        U = U[:, ind]
        D = diag_sign(U)
        self.U = U @ D
        VT[:d, :] = D @ VT[ind, :]
        self.VT = VT

        self.Σ = np.zeros((self.m, self.n))
        self.Σ[:d, :d] = np.diag(self.𝜎)

        𝜎_sq = self.𝜎 ** 2
        self.explained_ratio_svd = np.cumsum(𝜎_sq) / 𝜎_sq.sum()

        # 按使用的成分数量切分矩阵
        U = self.U[:, :self.r_component]
        Σ = self.Σ[:self.r_component, :self.r_component]
        VT = self.VT[:self.r_component, :]

        # 转换数据
        self.X_svd = U @ Σ @ VT

    def fit(self, r_component):

        # pca
        P = self.P[:, :r_component]
        𝜖 = self.𝜖[:r_component, :]

        # 转换数据
        self.X_pca = P @ 𝜖

        # svd
        U = self.U[:, :r_component]
        Σ = self.Σ[:r_component, :r_component]
        VT = self.VT[:r_component, :]

        # 转换数据
        self.X_svd = U @ Σ @ VT

def diag_sign(A):
    "计算矩阵A对角线元素的符号"

    D = np.diag(np.sign(np.diag(A)))

    return D
```

我们还定义一个函数来打印信息，以便比较不同算法得到的分解结果。

```{code-cell} ipython3
def compare_pca_svd(da):
    """
    比较PCA和SVD的结果。
    """

    da.pca()
    da.svd()

    print('特征值和奇异值\n')
    print(f'λ = {da.λ}\n')
    print(f'σ^2 = {da.σ**2}\n')
    print('\n')

    k = da.r

    # 载荷矩阵
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('载荷')
    axs[0].plot(da.P[:, :k])
    axs[0].set_title('P')
    axs[0].set_xlabel('变量 $i$')
    axs[1].plot(da.U[:, :k])
    axs[1].set_title('U')
    axs[1].set_xlabel('变量 $i$')
    plt.show()

    # 主成分
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('主成分')
    axs[0].plot(da.ε[:k, :].T)
    axs[0].set_title('ε')
    axs[0].set_xlabel('n')
    axs[1].plot(da.VT[:k, :].T * np.sqrt(da.λ[:k]))
    axs[1].set_title(r'$V^\top *\sqrt{\lambda}$')
    axs[1].set_xlabel('n')
    plt.show()
```

让我们把这些工具应用到一个由两个因子加噪声生成的、具有$m = 5$个变量和$n = 200$个观测值的“矮胖”数据矩阵上。

```{code-cell} ipython3
m, n = 5, 200
G = rng.normal(size=(m, 2))              # 两个因子上的载荷
Z = rng.normal(size=(2, n))              # 因子
X = G @ Z + 0.2 * rng.normal(size=(m, n))
X = X - X.mean(axis=1, keepdims=True)    # 减去样本均值

da = DecomAnalysis(X)
compare_pca_svd(da)
```

这两种算法得到了相同的载荷和相同的主成分。

累积方差解释比例显示，两个主成分几乎解释了$X$中的全部变动，这与我们生成数据的方式是一致的。

```{code-cell} ipython3
print("explained ratio (PCA):", da.explained_ratio_pca)
print("explained ratio (SVD):", da.explained_ratio_svd)
```

关于将PCA应用于智力测验结构分析的例子，请参阅 {doc}`多元正态分布 <multivariate_normal>` 讲座。

请看该讲座中描述并演示经典因子分析模型的部分。

如前所述，在本讲座的续篇 {doc}`VAR与DMD <var_dmd>` 中，我们将描述SVD如何提供快速计算一阶向量自回归（VARs）降阶近似的方法。

## 练习

```{exercise}
:label: svd_ex1

在普通最小二乘法（OLS）中，我们学会计算 $ \hat{\beta} = (X^\top X)^{-1} X^\top y $，其中回归量矩阵 $X$ 每一行对应一个观测值，每一列对应一个回归量。

但当回归量完全共线，或者当回归量数量多于观测值数量，使得 $X$ 是**矮胖**矩阵且方程组 $X \beta = y$ 是欠定的时候，矩阵 $X^\top X$ 不可逆（其行列式为零）。

而当回归量近似共线时，$X^\top X$ 是**病态的**，这意味着其最大特征值与最小特征值之比非常大。

在这些情况下，我们可以改用所谓的 [伪逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) $X^+$，用它代替 $(X^\top X)^{-1} X^\top$。

1. 使用 $X$ 的SVD构建伪逆矩阵 $X^{+}$，并用它来计算 $\hat{\beta} = X^+ y$。
1. 证明当 $X$ 的列线性无关时（即**超定**系统），$X^+ = (X^\top X)^{-1} X^\top$，因此 $\hat \beta$ 就是通常的OLS估计量。
1. 证明当 $X$ 的行线性无关时（即**欠定**系统），$\hat \beta$ 恰好求解 $X \beta = y$，并且在所有解中范数最小。
1. 对一个 $5 \times 8$ 的随机数矩阵 $X$，用数值方法验证上述结论。
```

```{solution-start} svd_ex1
:class: dropdown
```

我们可以使用SVD来计算伪逆。

从简约SVD出发

$$
X  = U \Sigma V^\top
$$

其中 $\Sigma$ 是由正奇异值组成的 $p \times p$ 对角矩阵，伪逆为

$$
X^{+}  = V \Sigma^{-1} U^\top
$$

等价地，用完整SVD表示，$X^+ = V \Sigma^+ U^\top$，其中 $n \times m$ 矩阵 $\Sigma^+$ 是通过转置 $\Sigma$ 并将每个正奇异值 $\sigma_j$ 替换为 $1/\sigma_j$ 得到的：

$$
\Sigma^{+} = \begin{bmatrix}
\frac{1}{\sigma_1} & 0 & \cdots & 0 & 0 \\
0 & \frac{1}{\sigma_2} & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & \frac{1}{\sigma_p} & 0 \\
0 & 0 & \cdots & 0 & 0 \\
\end{bmatrix}
$$

最后：

$$
\hat{\beta} = X^{+}y = V \Sigma^{-1} U^\top y
$$

当 $X$ 的列线性无关时，$V$ 是一个方阵正交矩阵，所以

$$
(X^\top X)^{-1} X^\top = (V \Sigma^2 V^\top)^{-1} V \Sigma U^\top = V \Sigma^{-2} V^\top V \Sigma U^\top = V \Sigma^{-1} U^\top = X^+ .
$$

当 $X$ 的行线性无关时，$U$ 是一个方阵正交矩阵，所以

$$
X \hat \beta = U \Sigma V^\top V \Sigma^{-1} U^\top y = U U^\top y = y .
$$

$X \beta = y$ 的其他任意解都具有形式 $\beta = \hat \beta + \eta$，其中 $X \eta = 0$。

向量 $\hat \beta = V (\Sigma^{-1} U^\top y)$ 位于 $X$ 的行空间中，该行空间由 $V$ 的列张成，而 $\eta$ 位于 $X$ 的零空间中。

这两个子空间是正交的，所以 $\|\beta\|^2 = \|\hat \beta\|^2 + \|\eta\|^2 \geq \|\hat \beta\|^2$。

下面给出数值验证。

```{code-cell} ipython3
n_obs, k = 5, 8
X = rng.normal(size=(n_obs, k))
y = rng.normal(size=n_obs)

U, σ, VT = np.linalg.svd(X, full_matrices=False)
X_plus = VT.T @ np.diag(1 / σ) @ U.T
β_hat = X_plus @ y

print("X_plus equals np.linalg.pinv(X):", np.allclose(X_plus, np.linalg.pinv(X)))
print("X β_hat = y:                    ", np.allclose(X @ β_hat, y))

# another exact solution: add an element of the null space of X
β_other = β_hat + (np.eye(k) - X_plus @ X) @ rng.normal(size=k)
print("X β_other = y:                  ", np.allclose(X @ β_other, y))
print(f"norms: |β_hat| = {LA.norm(β_hat):.3f}, |β_other| = {LA.norm(β_other):.3f}")

# over-determined case: pseudo-inverse reproduces OLS
X_tall = rng.normal(size=(8, 5))
print("pinv equals (X^T X)^{-1} X^T:   ",
      np.allclose(np.linalg.pinv(X_tall), LA.inv(X_tall.T @ X_tall) @ X_tall.T))
```

```{solution-end}
```

```{exercise}
:label: svd_ex2

本练习通过数值方法验证Eckart-Young定理。

构造一个 $60 \times 40$ 的矩阵 $X$，使其等于一个秩为 $4$ 的矩阵加上噪声，例如

`X = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 40)) + 0.3 * rng.normal(size=(60, 40))`

对于 $r = 1, 2, 4, 8$

1. 计算Eckart-Young近似 $\hat X_r$，并验证 $|| X - \hat X_r ||$ 在谱范数、弗罗贝尼乌斯范数和核范数下的公式
1. 对于 $500$ 个具有 $r$ 个标准正交列、张成随机抽取子空间的矩阵 $Q$，将 $|| X - \hat X_r ||_F$ 与 $|| X - Q Q^\top X ||_F$ 进行比较（对每个 $Q$，$Q Q^\top X$ 是列空间位于 $Q$ 列空间中的 $X$ 的最佳近似）

最后，绘制比值 $\sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$ 随 $r$ 变化的图形，并将其与上文PCA流程中的第4步联系起来。
```

```{solution-start} svd_ex2
:class: dropdown
```

以下是一种解法。

```{code-cell} ipython3
m, n = 60, 40
X = rng.normal(size=(m, 4)) @ rng.normal(size=(4, n)) + 0.3 * rng.normal(size=(m, n))
U, σ, VT = np.linalg.svd(X, full_matrices=False)

print(" r   spectral  Frobenius  nuclear   |X - X_r|_F   best random |X - QQ'X|_F")
for r in (1, 2, 4, 8):
    X_r = U[:, :r] @ np.diag(σ[:r]) @ VT[:r, :]
    E = X - X_r
    checks = (np.isclose(LA.norm(E, 2), σ[r]),
              np.isclose(LA.norm(E, 'fro'), np.sqrt(np.sum(σ[r:]**2))),
              np.isclose(LA.norm(E, 'nuc'), np.sum(σ[r:])))
    best_random = np.inf
    for _ in range(500):
        Q = LA.qr(rng.normal(size=(m, r)))[0]
        best_random = min(best_random, LA.norm(X - Q @ Q.T @ X, 'fro'))
    print(f"{r:2d}   {str(checks[0]):8s}  {str(checks[1]):9s}  {str(checks[2]):8s}"
          f"  {LA.norm(E, 'fro'):10.2f}   {best_random:10.2f}")
```

所有这三个误差公式都成立，而且没有任何一个随机抽取的秩-$r$近似能接近Eckart-Young近似的效果。

```{code-cell} ipython3
share = np.cumsum(σ**2) / np.sum(σ**2)

fig, ax = plt.subplots()
ax.plot(np.arange(1, len(σ) + 1), share, 'o-')
ax.set_xlabel('$r$')
ax.set_ylabel('fraction of $||X||_F^2$ captured by $\\hat X_r$')
plt.show()
```

这个比值在 $r = 4$（即 $X$ 中信号部分的秩）之前急剧上升，之后趋于平缓。

这个比值正是PCA流程第4步用来决定保留多少主成分的“保留方差比例”，而 $1$ 减去它就是 $\hat X_r$ 的相对弗罗贝尼乌斯误差的平方。

```{solution-end}
```

```{exercise}
:label: svd_ex3

本练习为后续讲座 {doc}`VAR与DMD <var_dmd>` 搭建桥梁。

模拟一个**瘦高型**时间序列数据集 $\tilde X$，包含 $m = 100$ 个变量和 $61$ 个时期，其中

$$
\tilde X_t = G z_t + 0.5 v_t, \qquad z_{t+1} = \begin{bmatrix} 0.9 & 0 \cr 0 & 0.6 \end{bmatrix} z_t + w_{t+1}
$$

其中 $G$ 是一个 $100 \times 2$ 的标准正态随机数矩阵，$v_t$ 和 $w_{t+1}$ 是标准正态随机向量。

减去每个变量的样本均值，然后使用前 $41$ 个时期构造 $X = \begin{bmatrix} \tilde X_1 \mid \cdots \mid \tilde X_{40} \end{bmatrix}$ 和 $X' = \begin{bmatrix} \tilde X_2 \mid \cdots \mid \tilde X_{41} \end{bmatrix}$。

1. 计算一阶VAR $\tilde X_{t+1} = A \tilde X_t + \epsilon_{t+1}$ 中矩阵 $A$ 的最小二乘估计量 $\hat A = X' X^+$，并验证它能完美拟合：$\hat A X = X'$。
1. 设 $\hat X_2$ 为 $X$ 的Eckart-Young秩-$2$近似。用数值方法验证 $\hat X_2^+ = V_2 \Sigma_2^{-1} U_2^\top$，并计算秩-$2$估计量 $\hat A_2 = X' \hat X_2^+$。
1. 设 $F = U_2^\top X$ 为 $X$ 的前两个主成分组成的 $2 \times 40$ 矩阵。验证 $\hat A_2 = X' F^\top (F F^\top)^{-1} U_2^\top$，即 $\hat A_2 \tilde X_t$ 等于将 $\tilde X_{t+1}$ 对 $\tilde X_t$ 的前两个主成分回归所得的拟合值。
1. 使用剩余的 $20$ 个时期，比较 $\check A = \hat A$ 和 $\check A = \hat A_2$ 的一步样本外预测误差 $\|X'_{\rm out} - \check A X_{\rm out}\|_F / \|X'_{\rm out}\|_F$。对几个不同的随机种子重复此过程。
```

```{solution-start} svd_ex3
:class: dropdown
```

以下是一种解法。

```{code-cell} ipython3
def simulate_tall(m=100, T=61, σ_v=0.5, seed=1):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(m, 2))
    a = np.array([0.9, 0.6])
    z = np.zeros((2, T))
    for t in range(T - 1):
        z[:, t+1] = a * z[:, t] + rng.normal(size=2)
    X_tilde = G @ z + σ_v * rng.normal(size=(m, T))
    return X_tilde - X_tilde.mean(axis=1, keepdims=True)

def rel_error(A, X, X_prime):
    return LA.norm(X_prime - A @ X) / LA.norm(X_prime)

X_tilde = simulate_tall()
X, X_prime = X_tilde[:, :40], X_tilde[:, 1:41]           # estimation sample
X_out, X_prime_out = X_tilde[:, 40:-1], X_tilde[:, 41:]  # hold-out sample

# 1. minimum-norm least squares estimator
A_hat = X_prime @ np.linalg.pinv(X)
print("A_hat X = X':", np.allclose(A_hat @ X, X_prime))

# 2. pseudo-inverse of the Eckart-Young approximation
U, σ, VT = np.linalg.svd(X, full_matrices=False)
r = 2
U_r, σ_r, V_r = U[:, :r], σ[:r], VT[:r, :].T
X_r = U_r @ np.diag(σ_r) @ V_r.T
X_r_plus = V_r @ np.diag(1 / σ_r) @ U_r.T
print("pinv(X_r) = V_r Σ_r^{-1} U_r^T:", np.allclose(np.linalg.pinv(X_r, rcond=1e-10), X_r_plus))
A_r = X_prime @ X_r_plus

# 3. principal components regression
F = U_r.T @ X
A_pcr = X_prime @ F.T @ LA.inv(F @ F.T) @ U_r.T
print("A_r equals principal components regression:", np.allclose(A_r, A_pcr))

# 4. in-sample and out-of-sample fit
print(f"\nin-sample:     A_hat {rel_error(A_hat, X, X_prime):.3f},  A_r {rel_error(A_r, X, X_prime):.3f}")
print(f"out-of-sample: A_hat {rel_error(A_hat, X_out, X_prime_out):.3f},  A_r {rel_error(A_r, X_out, X_prime_out):.3f}")

print("\nout-of-sample errors for other seeds")
for seed in range(2, 7):
    X_tilde = simulate_tall(seed=seed)
    X, X_prime = X_tilde[:, :40], X_tilde[:, 1:41]
    X_out, X_prime_out = X_tilde[:, 40:-1], X_tilde[:, 41:]
    U, σ, VT = np.linalg.svd(X, full_matrices=False)
    A_r = X_prime @ VT[:r, :].T @ np.diag(1 / σ[:r]) @ U[:, :r].T
    A_hat = X_prime @ np.linalg.pinv(X)
    print(f"seed {seed}:  A_hat {rel_error(A_hat, X_out, X_prime_out):.3f},"
          f"  A_r {rel_error(A_r, X_out, X_prime_out):.3f}")
```

最小范数估计量 $\hat A$ 能完美拟合估计样本，但样本外预测效果较差，因为伪逆 $X^+ = V \Sigma^{-1} U^\top$ 对那些奇异值较小、大多只包含噪声的方向赋予了较大的权重 $1/\sigma_j$。

秩-$2$估计量 $\hat A_2$ 舍弃了这些方向，因此样本外预测效果更好。

估计量 $\hat A_2 = X' \hat X_2^+$ 正是**动态模态分解**所计算的估计量；后续讲座 {doc}`VAR与DMD <var_dmd>` 将对此进行详细研究。

```{solution-end}
```
