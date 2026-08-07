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
    The Setting: 基本设定
    Singular Value Decomposition: 奇异值分解
    Four Fundamental Subspaces: 四个基本子空间
    Eckart-Young Theorem: Eckart-Young定理
    Full and Reduced SVD's: 完全SVD和简化SVD
    Polar Decomposition: 极分解
    'Application: Principal Components Analysis (PCA)': 应用：主成分分析(PCA)
    Relationship of PCA to SVD: PCA与SVD的关系
    PCA with Eigenvalues and Eigenvectors: 基于特征值和特征向量的PCA
    Connections: 联系
    Exercises: 练习
---

# 奇异值分解

## 概述

**奇异值分解**（SVD）是一个强大的数学工具，在数据分析和机器学习中有着广泛的应用。它不仅可以用于最小二乘投影，还是许多统计和机器学习方法的基石。

在这一讲中，我们将首先介绍SVD的基本概念，然后探讨它与以下几个重要领域的关系：

* 线性代数中的**四个基本子空间**
* **最小二乘回归**（包括欠定和超定情况）
* **主成分分析**（PCA）

与主成分分析（PCA）类似，动态模态分解（DMD）也可以被看作是一种数据降维方法，它通过将数据投影到一组有限的因子上来表示数据中的显著模式。

在后续的{doc}`动态模式分解<var_dmd>`讲座中，我们会看到如何利用SVD来高效地计算向量自回归（VAR）模型的简化形式。

## 基本设定

假设我们有一个$m \times n$的矩阵$X$，其秩为$p$。

显然，$p$不会超过$m$和$n$的较小值，即$p \leq \min(m,n)$。

在实际应用中，$X$通常代表一个**数据矩阵**：

* 每一列代表一个**观测单位**（可以是某个时间点的数据，或是某个个体的信息）
* 每一行代表一个**变量**（用来描述观测单位的某个特征或属性）


我们将关注两种情况：

* **矮胖**情况，即$m << n$，表示列数（个体）远多于行数（属性）。
* **高瘦**情况，即$m >> n$，表示行数（属性）远多于列数（个体）。


我们将在这两种情况下对$X$进行**奇异值分解**。

在 $m << n$ 的情况下，即个体数量 $n$ 远大于属性数量 $m$ 时，我们可以通过对观测值函数取平均来计算联合分布的样本矩。

在这种 $m << n$ 的情况下，我们将使用**奇异值分解**来进行**主成分分析**(PCA)以寻找**模式**。

在 $m >> n$ 的情况下，即属性数量 $m$ 远大于个体数量 $n$，且在时间序列环境中 $n$ 等于数据集 $X$ 中所覆盖的时间段数量时，我们将采用不同的方法。

在这种情况下，我们仍然使用奇异值分解，但目的是构建动态模态分解(DMD)。

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
* $\Sigma$ 是一个 $m \times n$ 矩阵，其主对角线上的前 $p$ 个位置是正数 $\sigma_1, \sigma_2, \ldots, \sigma_p$，称为**奇异值**；$\Sigma$ 的其余元素都为零

* 这 $p$ 个奇异值是 $m \times m$ 矩阵 $X X^\top $ 以及 $n \times n$ 矩阵 $X^\top  X$ 的特征值的正平方根

* 我们约定，当 $U$ 是复值矩阵时，$U^\top $ 表示 $U$ 的**共轭转置**或**厄米特转置**，即 $U_{ij}^\top $ 是 $U_{ji}$ 的复共轭。

* 类似地，当 $V$ 是复值矩阵时，$V^\top$ 表示 $V$ 的**共轭转置**或**厄米特转置**

矩阵 $U,\Sigma,V$ 通过以下方式对向量进行线性变换：

* 用酉矩阵 $U$ 和 $V$ 乘以向量会使其**旋转**，但保持**向量之间的角度**和**向量的长度**不变。
* 用对角矩阵 $\Sigma$ 乘以向量会保持**向量之间的角度**不变，但会**重新缩放**向量。

因此，表示式 {eq}`eq:SVD101` 表明，用 $m \times n$ 矩阵 $X$ 乘以 $n \times 1$ 向量 $y$ 相当于按顺序执行以下三个乘法运算：

* 通过计算 $V^\top y$ 来**旋转** $y$
* 通过乘以 $\Sigma$ 来**重新缩放** $V^\top y$
* 通过乘以 $U$ 来**旋转** $\Sigma V^\top y$

$m \times n$ 矩阵 $X$ 的这种结构为构建系统开启了大门

数据**编码器**和**解码器**。

因此，

* $V^\top y$ 是一个编码器
* $\Sigma$ 是一个应用于编码数据的运算符
* $U$ 是一个解码器，用于处理将运算符 $\Sigma$ 应用于编码数据后的输出

我们将在本讲稍后研究动态模态分解时应用这些概念。

**未来路线**

我们上面描述的是所谓的**完全** SVD。

在**完全** SVD中，$U$、$\Sigma$ 和 $V$ 的形状分别为 $\left(m, m\right)$、$\left(m, n\right)$、$\left(n, n\right)$。

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

让我们将 X 的完整奇异值分解写作

$$
X = \begin{bmatrix} U_L & U_R \end{bmatrix} \begin{bmatrix} \Sigma_p & 0 \cr 0 & 0 \end{bmatrix}
     \begin{bmatrix} V_L & V_R \end{bmatrix}^\top
$$ (eq:fullSVDpartition)

其中 $\Sigma_p$ 是一个 $p \times p$ 对角矩阵，对角线上是 $p$ 个奇异值，且

$$
\begin{aligned}
U_L & = \begin{bmatrix}u_1 & \cdots  & u_p \end{bmatrix},  \quad U_R  = \begin{bmatrix}u_{p+1} & \cdots u_m \end{bmatrix}  \cr
V_L & = \begin{bmatrix}v_1 & \cdots  & v_p \end{bmatrix} , \quad U_R  = \begin{bmatrix}v_{p+1} & \cdots u_n \end{bmatrix}
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

方程 {eq}`eq:orthoortho1` 说明了变换 $X$ 如何将一对正交单位向量 $v_i, v_j$（其中 $i$ 和 $j$ 都小于或等于 $X$ 的秩 $p$）映射到一对正交单位向量 $u_i, u_j$。

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

注意方程 {eq}`eq:orthoortho2` 表明变换 $X^\top$ 将一对不同的正交单位向量 $u_i, u_j$（其中 $i$ 和 $j$ 都小于或等于 $X$ 的秩 $p$）映射到一对不同的正交单位向量 $v_i, v_j$。

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
{\mathcal N}(X) & = {\mathcal C} (V_R) \cr
\end{aligned}
$$ (eq:fourspaceSVD)

由于 $U$ 和 $V$ 都是正交矩阵，集合 {eq}`eq:fourspaceSVD` 表明

* $U_L$ 是 $X$ 列空间的标准正交基
* $U_R$ 是 $X^\top$ 零空间的标准正交基
* $V_L$ 是 $X$ 行空间的标准正交基
* $V_R$ 是 $X$ 零空间的标准正交基

我们通过执行{eq}`eq:fullSVDpartition`右侧要求的乘法并读取结果，已经验证了{eq}`eq:fourspaceSVD`中的四个声明。

{eq}`eq:fourspaceSVD`中的声明以及$U$和$V$都是酉矩阵（即正交矩阵）这一事实意味着：

* $X$的列空间与$X^\top$的零空间正交
* $X$的零空间与$X$的行空间正交

这些性质有时用以下两对正交补空间来描述：

* ${\mathcal C}(X)$是${\mathcal N}(X^\top)$的正交补
* ${\mathcal R}(X)$是${\mathcal N}(X)$的正交补

让我们看一个例子。

```{code-cell} ipython3
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

```

导入这些模块后，让我们来看示例。

```{code-cell} ipython3
np.set_printoptions(precision=2)

# 定义矩阵
A = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8],
              [5, 6, 7, 8, 9]])

# 计算矩阵的奇异值分解
U, S, V = np.linalg.svd(A,full_matrices=True)

# 计算矩阵的秩
rank = np.linalg.matrix_rank(A)

# 打印矩阵的秩
print("矩阵的秩:\n", rank)
print("S: \n", S)

# 计算四个基本子空间
row_space = U[:, :rank]
col_space = V[:, :rank]
null_space = V[:, rank:]
left_null_space = U[:, rank:]


print("U:\n", U)
print("列空间:\n", col_space)
print("左零空间:\n", left_null_space)
print("V.T:\n", V.T)
print("行空间:\n", row_space.T)
print("右零空间:\n", null_space.T)
```

## Eckart-Young定理

假设我们要构造一个$m \times n$矩阵$X$的最佳秩$r$近似。

这里的最佳，指的是在所有秩为$r < p$的矩阵中，找到一个矩阵$X_r$使得以下范数最小：

$$ 
|| X - X_r || 
$$

其中$|| \cdot ||$表示矩阵$X$的范数，且$X_r$属于所有维度为$m \times n$的秩$r$矩阵空间。

一个$m \times n$矩阵$X$的三种常用**矩阵范数**可以用$X$的奇异值表示：

* **谱范数**或$l^2$范数 $|| X ||_2 = \max_{||y|| \neq 0} \frac{||X y ||}{||y||} = \sigma_1$
* **Frobenius范数** $||X ||_F = \sqrt{\sigma_1^2 + \cdots + \sigma_p^2}$
* **核范数** $ || X ||_N = \sigma_1 + \cdots + \sigma_p $

Eckart-Young定理指出，对于这三种范数，最佳的秩$r$矩阵是相同的，等于：

$$
\hat X_r = \sigma_1 U_1 V_1^\top  + \sigma_2 U_2 V_2^\top  + \cdots + \sigma_r U_r V_r^\top
$$ (eq:Ekart)

这是一个非常强大的定理，它表明我们可以将一个非满秩的 $m \times n$ 矩阵 $X$ 通过SVD分解，用一个满秩的 $p \times p$ 矩阵来最佳近似。

此外，如果这些 $p$ 个奇异值中有些携带的信息比其他的更多，而我们想用最少的数据获得最多的信息，我们可以取按大小排序的 $r$ 个主要奇异值。

在介绍主成分分析时，我们会对此进行更详细的讨论。

你可以在[这里](https://en.wikipedia.org/wiki/Low-rank_approximation)阅读关于Eckart-Young定理及其应用的内容。

在讨论主成分分析(PCA)和动态模态分解(DMD)时，我们将会用到这个定理。

## 完全SVD和简化SVD

到目前为止，我们描述的是**完全**SVD的性质，其中 $U$、$\Sigma$ 和 $V$ 的形状分别为 $\left(m, m\right)$、$\left(m, n\right)$、$\left(n, n\right)$。

有一种替代性的矩阵分解记法，称为**经济型**或**简化型** SVD，其中 $U, \Sigma$ 和 $V$ 的形状与完全SVD中的不同。

注意，因为我们假设 $X$ 的秩为 $p$，所以只有 $p$ 个非零奇异值，其中 $p=\textrm{rank}(X)\leq\min\left(m, n\right)$。

**简化型** SVD利用这一事实，将 $U$、$\Sigma$ 和 $V$ 表示为形状分别为 $\left(m, p\right)$、$\left(p, p\right)$、$\left(n, p\right)$ 的矩阵。

你可以在这里了解简化型和完全型SVD
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>

对于完全型SVD，

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

但这些性质并非都适用于**简化型** SVD。

哪些性质成立取决于我们处理的是**高瘦型**矩阵还是**矮胖型**矩阵。

* 在**高瘦型**情况下，即 $m > > n$，对于**简化型** SVD

$$
\begin{aligned}
UU^\top  &  \neq I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V = I
\end{aligned}
$$

* 在**矮胖**情况下(即 $m < < n$),对于**简化**SVD

$$
\begin{aligned}
UU^\top  &  = I  &  \quad U^\top  U = I \cr
VV^\top  & = I & \quad V^\top  V \neq I
\end{aligned}
$$

当我们研究动态模态分解时,我们需要记住这些性质,因为我们会使用简化SVD来计算一些DMD表示。

让我们做一个练习来比较**完全**和**简化**SVD。

回顾一下,

* 在**完全**SVD中

  - $U$ 是 $m \times m$ 维
  - $\Sigma$ 是 $m \times n$ 维
  - $V$ 是 $n \times n$ 维

* 在**简化**SVD中

  - $U$ 是 $m \times p$ 维
  - $\Sigma$ 是 $p \times p$ 维
  - $V$ 是 $n \times p$ 维

首先,让我们研究一个 $m = 5 > n = 2$ 的情况。

(这是我们在研究**动态模态分解**时会遇到的**高瘦**情况的一个小例子。)

```{code-cell} ipython3
import numpy as np
X = np.random.rand(5,2)
U, S, V = np.linalg.svd(X,full_matrices=True)  # 完整SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # 简化SVD
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

上述代码展示了 `full_matrices=True` 和 `full_matrices=False` 选项的应用。使用 `full_matrices=False` 会返回简化的奇异值分解。

**完整**和**简化**的奇异值分解都能准确地分解一个 $m \times n$ 矩阵 $X$

当我们在后面学习动态模态分解时，记住在这种高瘦矩阵情况下完整和简化奇异值分解的上述性质将很重要。

现在让我们来看一个矮胖矩阵的情况。

为了说明这种情况，我们将设置 $m = 2 < 5 = n$，并计算完整和简化的奇异值分解。

```{code-cell} ipython3
import numpy as np
X = np.random.rand(2,5)
U, S, V = np.linalg.svd(X,full_matrices=True)  # 完整SVD
Uhat, Shat, Vhat = np.linalg.svd(X,full_matrices=False) # 简化SVD
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

这里

* $S$ 是一个 $m \times m$ **对称**矩阵
* $Q$ 是一个 $m \times n$ **正交**矩阵

在我们的简化SVD中

* $U$ 是一个 $m \times p$ 正交矩阵
* $\Sigma$ 是一个 $p \times p$ 对角矩阵
* $V$ 是一个 $n \times p$ 正交矩阵

## 应用：主成分分析(PCA)

让我们从 $n >> m$ 的情况开始，即个体数量 $n$ 远大于属性数量 $m$ 的情况。

在 $n >> m$ 的情况下，矩阵 $X$ 是**矮胖型**的，这与后面要讨论的 $m >> n$ 情况下的**高瘦型**相对。

我们将 $X$ 视为一个 $m \times n$ 的**数据**矩阵：

$$
X = \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n\end{bmatrix}
$$

其中对于 $j = 1, \ldots, n$，列向量 $X_j = \begin{bmatrix}x_{1j}\\x_{2j}\\\vdots\\x_{mj}\end{bmatrix}$ 是变量 $\begin{bmatrix}X_1\\X_2\\\vdots\\X_m\end{bmatrix}$ 的观测值向量。

在**时间序列**分析中，列索引 $j$ 代表不同的时间点，而行索引代表不同的随机变量。

在**横截面**分析中，列索引 $j$ 代表不同的个体，而行索引代表它们的不同属性。

正如我们之前所见，SVD是一种将矩阵分解为有用组件的方法，就像极分解、特征分解等其他方法一样。

而PCA则是一种基于SVD进行数据分析的方法。其目标是应用一定的步骤，借助统计工具来捕捉数据中最重要的模式，从而更好地可视化数据中的规律。

**第1步：数据标准化**

由于数据矩阵可能包含不同单位和尺度的变量，我们首先需要对数据进行标准化。

首先计算 $X$ 的每一行的平均值。

$$
\bar{X_i}= \frac{1}{n} \sum_{j = 1}^{n} x_{ij}
$$

然后用这些平均值创建一个平均值矩阵：

$$
\bar{X} =  \begin{bmatrix} \bar{X_1} \\ \bar{X_2} \\ \ldots \\ \bar{X_m}\end{bmatrix}\begin{bmatrix}1 \mid 1 \mid \cdots \mid 1 \end{bmatrix}
$$

从原始矩阵中减去平均值矩阵，得到一个均值中心化矩阵：

$$
B = X - \bar{X}
$$

**第2步：计算协方差矩阵**

由于我们希望提取变量之间的关系，而不仅仅是它们的大小——换句话说，我们想知道它们能在多大程度上相互解释——因此我们计算$B$的协方差矩阵。

$$
C = \frac{1}{n} BB^{\top}
$$

**第3步：分解协方差矩阵并排列奇异值：**

由于矩阵$C$是正定的，我们可以对其进行特征分解，找出其特征值，并将特征值和特征向量矩阵按降序重新排列。

$C$的特征分解可以通过分解$B$来求得。由于$B$不是方阵，我们对$B$进行SVD分解：

$$
\begin{aligned}
B B^\top &= U \Sigma V^\top (U \Sigma V^{\top})^{\top}\\
&= U \Sigma V^\top V \Sigma^\top U^\top\\
&= U \Sigma \Sigma^\top U^\top
\end{aligned}
$$

$$
C = \frac{1}{n} U \Sigma \Sigma^\top U^\top
$$

然后我们可以重新排列矩阵$U$和$\Sigma$中的列，使奇异值按降序排列。

**第4步：选择奇异值，（可选）截断其余部分：**

我们现在可以根据想要保留的方差量来决定选择多少个奇异值（例如，保留95%的总方差）。

我们可以通过计算前$r$个主要因子所包含的方差除以总方差来获得这个百分比：

$$
\frac{\sum_{i = 1}^{r} \sigma^2_{i}}{\sum_{i = 1}^{p} \sigma^2_{i}}
$$

**第5步：创建得分矩阵：**

$$
\begin{aligned}
T&= BV \cr
&= U\Sigma V^\top V \cr
&= U\Sigma
\end{aligned}
$$

## PCA与SVD的关系

为了将SVD与数据集$X$的PCA联系起来，首先构造数据矩阵$X$的SVD：

让我们假设所有变量的样本均值都为零，这样我们就不需要对矩阵进行标准化了。

$$
X = U \Sigma V^\top  = \sigma_1 U_1 V_1^\top  + \sigma_2 U_2 V_2^\top  + \cdots + \sigma_p U_p V_p^\top
$$ (eq:PCA1)

其中

$$
U=\begin{bmatrix}U_1|U_2|\ldots|U_m\end{bmatrix}
$$

$$
V^\top  = \begin{bmatrix}V_1^\top \\V_2^\top \\\ldots\\V_n^\top \end{bmatrix}
$$

在方程{eq}`eq:PCA1`中，每个$m \times n$矩阵$U_{j}V_{j}^\top $显然是秩1的。

因此，我们有

$$
X = \sigma_1 \begin{bmatrix}U_{11}V_{1}^\top \\U_{21}V_{1}^\top \\\cdots\\U_{m1}V_{1}^\top \\\end{bmatrix} + \sigma_2\begin{bmatrix}U_{12}V_{2}^\top \\U_{22}V_{2}^\top \\\cdots\\U_{m2}V_{2}^\top \\\end{bmatrix}+\ldots + \sigma_p\begin{bmatrix}U_{1p}V_{p}^\top \\U_{2p}V_{p}^\top \\\cdots\\U_{mp}V_{p}^\top \\\end{bmatrix}
$$ (eq:PCA2)

在时间序列分析的背景下，我们可以这样解释方程{eq}`eq:PCA2`中的对象：

* $ \textrm{对于每个} \ k=1, \ldots, n $，对象 $\lbrace V_{kj} \rbrace_{j=1}^n$ 是第$k$个**主成分**的时间序列

* $U_j = \begin{bmatrix}U_{1k}\\U_{2k}\\\ldots\\U_{mk}\end{bmatrix} \ k=1, \ldots, m$
是变量$X_i$在第$k$个主成分上的**载荷**向量，其中$i=1, \ldots, m$

* 对于每个$k=1, \ldots, p$，$\sigma_k$是第$k$个**主成分**的强度，这里的强度指的是对$X$的整体协方差的贡献。

## 基于特征值和特征向量的PCA

现在我们使用样本协方差矩阵的特征分解来进行PCA。

设$X_{m \times n}$为我们的$m \times n$数据矩阵。

假设所有变量的样本均值都为零。

我们可以通过减去样本均值的**预处理**来确保这一点。

定义样本协方差矩阵$\Omega$为

$$
\Omega = XX^\top
$$

然后使用特征分解将$\Omega$表示为：

$$
\Omega =P\Lambda P^\top
$$

这里

* $P$是$\Omega$的$m×m$特征向量矩阵

* $\Lambda$是$\Omega$的特征值对角矩阵

我们可以将$X$表示为:

$$
X=P\epsilon
$$

其中

$$
\epsilon = P^{-1} X
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

\begin{equation*}
X=\begin{bmatrix}X_1|X_2|\ldots|X_m\end{bmatrix} =\begin{bmatrix}P_1|P_2|\ldots|P_m\end{bmatrix}
\begin{bmatrix}\epsilon_1\\\epsilon_2\\\ldots\\\epsilon_m\end{bmatrix}
= P_1\epsilon_1+P_2\epsilon_2+\ldots+P_m\epsilon_m
\end{equation*}

为了将前面的表示与我们之前通过SVD获得的PCA相协调，我们首先注意到$\epsilon_j^2=\lambda_j\equiv\sigma^2_j$。

现定义$\tilde{\epsilon_j} = \frac{\epsilon_j}{\sqrt{\lambda_j}}$，
这意味着$\tilde{\epsilon}_j\tilde{\epsilon}_j^\top =1$。

因此

$$
\begin{aligned}
X&=\sqrt{\lambda_1}P_1\tilde{\epsilon_1}+\sqrt{\lambda_2}P_2\tilde{\epsilon_2}+\ldots+\sqrt{\lambda_m}P_m\tilde{\epsilon_m}\\
&=\sigma_1P_1\tilde{\epsilon_2}+\sigma_2P_2\tilde{\epsilon_2}+\ldots+\sigma_mP_m\tilde{\epsilon_m} ,
\end{aligned}
$$

这与SVD分解

$$
X=\sigma_1U_1{V_1}^{T}+\sigma_2 U_2{V_2}^{T}+\ldots+\sigma_{r} U_{r}{V_{r}}^{T}
$$

是等价的,只要我们令:

* $U_j=P_j$ (变量在第j个主成分上的载荷向量)

* ${V_k}^{T}=\tilde{\epsilon_k}$ (第k个主成分)

由于计算数据矩阵$X$的$P$和$U$有不同的算法，根据所使用的算法，我们可能会得到符号差异或特征向量顺序的不同。

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
&\equiv U\Sigma\Sigma^\top U^\top \cr
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

因此，SVD中的矩阵$V$是$X^\top X$的特征向量矩阵

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

因此，$P = U$，我们得到$X$的表示

$$
X = P \epsilon = U \Sigma V^\top
$$

由此可得

$$
U^\top  X = \Sigma V^\top  = \epsilon
$$

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
            self.r_component = self.m

    def pca(self):

        𝜆, P = LA.eigh(self.Ω)    # P的列是特征向量

        ind = sorted(range(𝜆.size), key=lambda x: 𝜆[x], reverse=True)

        # 按特征值排序
        self.𝜆 = 𝜆[ind]
        P = P[:, ind]
        self.P = P @ diag_sign(P)

        self.Λ = np.diag(self.𝜆)

        self.explained_ratio_pca = np.cumsum(self.𝜆) / self.𝜆.sum()

        # 计算N乘T的主成分矩阵
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

    # 载荷矩阵
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('载荷')
    axs[0].plot(da.P.T)
    axs[0].set_title('P')
    axs[0].set_xlabel('m')
    axs[1].plot(da.U.T)
    axs[1].set_title('U')
    axs[1].set_xlabel('m')
    plt.show()

    # 主成分
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('主成分')
    axs[0].plot(da.ε.T)
    axs[0].set_title('ε')
    axs[0].set_xlabel('n')
    axs[1].plot(da.VT[:da.r, :].T * np.sqrt(da.λ))
    axs[1].set_title(r'$V^\top *\sqrt{\lambda}$')
    axs[1].set_xlabel('n')
    plt.show()
```

## 练习

```{exercise}
:label: svd_ex1

在普通最小二乘法(OLS)中，我们学会计算 $ \hat{\beta} = (X^\top X)^{-1} X^\top y $，但在某些情况下，比如当我们遇到共线性或欠定系统时：即**矮胖**矩阵。

在这些情况下，$ (X^\top X) $矩阵不可逆（其行列式为零）或病态（其行列式非常接近零）。

我们可以改用所谓的[伪逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)，即创建一个满秩的逆矩阵近似，以此来计算 $ \hat{\beta} $。

根据Eckart-Young定理，构建伪逆矩阵 $ X^{+} $ 并用它来计算 $ \hat{\beta} $。

```

```{solution-start} svd_ex1
:class: dropdown
```

我们可以使用SVD来计算伪逆：

$$
X  = U \Sigma V^\top
$$

对X求逆，我们得到：

$$
X^{+}  = V \Sigma^{+} U^\top
$$

其中：

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
\hat{\beta} = X^{+}y = V \Sigma^{+} U^\top y 
$$

```{solution-end}
```

关于PCA应用于分析智力测试结构的示例，请参见本讲座 {doc}`多元正态分布 <multivariate_normal>`。

查看该讲座中描述和说明经典因子分析模型的部分。

如前所述，在后续关于 {doc}`动态模态分解 <var_dmd>` 的讲座中，我们将描述SVD如何提供快速计算一阶向量自回归(VARs)的降阶近似的方法。
