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
---

# 向量自回归和动态模态分解

本节讲座中，我们将应用在 {doc}`奇异值分解 <svd_intro>` 中学到的计算方法来研究：

* 一阶向量自回归 (VARs)
* 动态模态分解 (DMDs)
* 一阶向量自回归和动态模态分解之间的联系

## 一阶向量自回归

我们想要拟合一个**一阶向量自回归**

$$
X_{t+1} = A X_t + C \epsilon_{t+1}, \quad \epsilon_{t+1} \perp X_t 
$$ (eq:VARfirstorder)

其中，$\epsilon_{t+1}$ 是一个独立同分布的随机向量序列 $m \times 1$ 在时间$t+1$ 的分量，且该序列有零均值向量和单位协方差矩阵；而 $ m \times 1 $ 的向量 $ X_t $ 是：

$$
X_t = \begin{bmatrix}  X_{1,t} & X_{2,t} & \cdots & X_{m,t}     \end{bmatrix}^\top 
$$ (eq:Xvector)

其中 $\cdot ^\top $ 表示共轭转置，$ X_{i,t} $ 是时间 $ t $ 时的变量 $ i $。 

我们想要拟合方程 {eq}`eq:VARfirstorder`。

我们的数据则存储在一个 $ m \times (n+1) $ 的矩阵 $ \tilde X $ 中

$$
\tilde X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_n \mid X_{n+1} \end{bmatrix}
$$

其中对于 $ t = 1, \ldots, n +1 $时，$ m \times 1 $ 的向量 $ X_t $ 由 {eq}`eq:Xvector` 给出。

因此，我们想要估计一个系统 {eq}`eq:VARfirstorder`，它由 $ m $ 个最小二乘回归组成，将**所有变量**对**所有变量**的一阶滞后值进行回归。

{eq}`eq:VARfirstorder` 的第 $i$ 个方程是将 $X_{i,t+1}$ 对向量 $X_t$ 进行回归。

我们按如下步骤进行。

从 $ \tilde X $ 中，我们构造以下两个 $m \times n$ 矩阵

$$
X =  \begin{bmatrix} X_1 \mid X_2 \mid \cdots \mid X_{n}\end{bmatrix}
$$

和

$$
X' =  \begin{bmatrix} X_2 \mid X_3 \mid \cdots \mid X_{n+1}\end{bmatrix}
$$

这里的 $ ' $ 是矩阵 $ X' $ 的名称的一部分，并不表示矩阵转置。

我们使用 $\cdot^\top $ 来表示矩阵转置或其在复矩阵中的扩展。

在构造 $ X $ 和 $ X' $ 的过程中，我们都从 $ \tilde X $ 中删除了某一列，$ X $ 是删除最后一列，$ X' $ 则是删除第一列。

显然，$ X $ 和 $ X' $ 都是 $ m \times n $ 的矩阵。

我们用 $ p \leq \min(m, n) $ 表示 $ X $ 的秩。

我们感兴趣的两种情况是：

* $ n > > m $，即时间序列观测值的数量 $n$ 远大于变量的数量 $m$
* $ m > > n $，即变量的数量 $m$ 远大于时间序列观测值的数量 $n$

在考虑了这两种特殊情况的一般情况中，有一个通用的公式描述了 $A$ 的最小二乘估计量 $\hat A$。

但重要的细节有所不同。

这个通用的公式是：

$$ 
\hat A = X' X^+ 
$$ (eq:commonA)

其中 $X^+$ 是 $X$ 的广义逆矩阵，或伪逆。

关于**穆尔-彭罗斯广义逆矩阵**的详细信息，请参见[穆尔-彭罗斯广义逆矩阵](https://baike.baidu.com/item/穆尔-彭罗斯广义逆矩阵/22770999?fr=aladdin)<!-- 中文语境下使用广义逆矩阵的语料远多于伪逆，所以后文统一使用广义逆矩阵 -->

在我们的两种情况下，广义逆矩阵适用的公式有所不同。

**短胖型情况：**

当$n >> m$时，即时间序列的观测值$n$远多于变量的数量$m$，且当$X$具有线性独立的**行**时，$X X^\top$的逆矩阵存在，且广义逆矩阵$X^+$为

$$
X^+ = X^\top  (X X^\top )^{-1} 
$$

这里$X^+$是一个满足$X X^+ = I_{m \times m}$的**右逆**，。

在这种情况下，我们用于估计总体回归系数矩阵$A$的最小二乘估计量的公式{eq}`eq:commonA`就变为

$$ 
\hat A = X' X^\top  (X X^\top )^{-1}
$$ (eq:Ahatform101)

这个计算最小二乘回归系数的公式在计量经济学中被广泛地使用。

它也被用于估计向量自回归。

公式{eq}`eq:Ahatform101`的右边，正比于$X_{t+1}$和$X_t$的经验交叉二阶矩矩，乘以$X_t$二阶矩阵的逆矩阵。

**高瘦型情况：**

当$m > > n$时，即属性数量$m$远大于时间序列的观测值$n$，且当$X$的**列**线性独立时，$X^\top X$的逆矩阵存在，且广义逆矩阵$X^+$为

$$
X^+ = (X^\top X)^{-1} X^\top 
$$

这里$X^+$是一个满足$X^+ X = I_{n \times n}$的**左逆**，。

在这种情况下，我们用于估计$A$的最小二乘估计公式{eq}`eq:commonA`变为

$$
\hat A = X' (X^\top X)^{-1} X^\top 
$$ (eq:hatAversion0)

请比较{eq}`eq:Ahatform101`和{eq}`eq:hatAversion0`中$\hat A$的表达式。

这里我们特别关注公式{eq}`eq:hatAversion0`。

$\hat A$的第$i$行是一个$m \times 1$的向量，其中包含了$X_{i,t+1}$对$X_{j,t}, j = 1, \ldots, m$回归的系数。

如果我们使用公式{eq}`eq:hatAversion0`来计算$\hat A X$，我们发现

$$
\hat A X = X'
$$

因此回归方程**完全拟合**。

这是**欠定最小二乘**模型中典型的结果。

再次重申，**高瘦**情况(见{doc}`奇异值分解<svd_intro>`)指观测的数量$n$相对于向量$X_t$属性的数量$m$较小时，我们想要拟合方程{eq}`eq:VARfirstorder`。

我们面临着最小二乘估计量是欠定的，且回归方程完全拟合的事实。

接下来，我们想要更加高效地计算广义逆矩阵$X^+$。

广义逆矩阵$X^+$将是我们$A$估计量的一个组成部分。

作为对$A$的估计量$\hat A$，我们想要形成一个$m \times m$的矩阵，来解决最小二乘最佳拟合问题

$$ 
\hat A = \textrm{argmin}_{\check A} || X' - \check  A X ||_F

$$ (eq:ALSeqn)

其中 $|| \cdot ||_F$ 表示矩阵的Frobenius（或欧几里得）范数。

Frobenius范数定义为

$$
 ||A||_F = \sqrt{ \sum_{i=1}^m \sum_{j=1}^m |A_{ij}|^2 }
$$

方程{eq}`eq:ALSeqn`右侧的最小值解为

$$
\hat A =  X'  X^{+}  
$$ (eq:hatAform)

其中（可能是巨大的）$ n \times m $ 的矩阵 $ X^{+} = (X^\top  X)^{-1} X^\top $ 同样是 $ X $ 的广义逆矩阵。

对于我们感兴趣的一些情况，$X^\top  X $ 可能接近奇异，这种情况会使某些数值算法变得不准确。

为了应对这种可能性，我们将使用高效的算法来构建公式{eq}`eq:hatAversion0`中 $\hat A$ 的**降秩近似**。

这种近似方式，让我们的向量自回归估计不再完全拟合。

$ \hat A $ 的第 $ i $ 行是一个 $ m \times 1 $ 的回归系数向量，表示 $ X_{i,t+1} $ 对 $ X_{j,t}, j = 1, \ldots, m $ 的回归。

一种高效计算广义逆矩阵$X^+$的方式是从奇异值分解开始

$$
X =  U \Sigma  V^\top  
$$ (eq:SVDDMD)

这里我们提醒自己，这个**简化**SVD中，$X$是一个$m \times n$的数据矩阵，$U$是一个$m \times p$的矩阵，$\Sigma$是一个$p \times p$的矩阵，而$V$是一个$n \times p$的矩阵。

通过以下一系列等式，我们可以有效地构造相关的广义逆矩阵$X^+$。

$$
\begin{aligned}
X^{+} & = (X^\top  X)^{-1} X^\top  \\
  & = (V \Sigma U^\top  U \Sigma V^\top )^{-1} V \Sigma U^\top  \\
  & = (V \Sigma \Sigma V^\top )^{-1} V \Sigma U^\top  \\
  & = V \Sigma^{-1} \Sigma^{-1} V^\top  V \Sigma U^\top  \\
  & = V \Sigma^{-1} U^\top  
\end{aligned}
$$ (eq:efficientpseudoinverse)

（由于$m > > n$，在简化SVD中$V^\top  V = I_{p \times p}$，因此我们可以将前面的一系列等式同时用于简化SVD和完整SVD。）

因此，我们将使用方程{eq}`eq:SVDDMD`中$X$的奇异值分解来构造$X$的广义逆矩阵$X^+$，计算方法为：

$$
X^{+} =  V \Sigma^{-1}  U^\top  
$$ (eq:Xplusformula)

其中矩阵$\Sigma^{-1}$是通过将$\Sigma$中的每个非零元素替换为$\sigma_j^{-1}$构造而成。

我们可以将公式{eq}`eq:Xplusformula`与公式{eq}`eq:hatAform`结合使用来计算回归系数矩阵$\hat A$。

因此，我们对$m \times m$的系数矩阵$A$的估计量$\hat A = X' X^+$为：

$$
\hat A = X' V \Sigma^{-1}  U^\top  
$$ (eq:AhatSVDformula)

## 动态模态分解(DMD)

接下来我们将研究一个特殊情况 -- 当变量数量$m >> n$时的情形。

**动态模态分解**可以用于处理这种"高瘦型"矩阵。

假设有一个$m \times (n+1)$的数据矩阵$\tilde X$，它包含了比时间周期$n+1$多得多的属性（或变量）$m$。

动态模态分解由{cite}`schmid2010`首次提出，

你可以在 {cite}`DMD_book` 和 {cite}`Brunton_Kutz_2019`（第7.2节）中阅读有关动态模态分解的内容。

**动态模态分解**（DMD）的目标是找到最小二乘回归系数矩阵$\hat A$的一个低秩近似，其中近似矩阵的秩$r$小于原始矩阵的秩$p$。

这个近似可以通过公式{eq}`eq:AhatSVDformula`来构造。

我们将逐步构建一种适合应用的表示。

我们将通过三种不同的表示方式来描述一阶线性动态系统（即我们的向量自回归），从而实现这一点。

**三种表示的指南：** 在实践中，我们主要关注表示3。

我们使用前两种表示来呈现一些有用的中间推导，这些步骤有助于我们理解表示3的内在原理。

在应用中，我们将只使用**DMD模态**的一小部分子集来近似动态。

我们使用这样一个小的DMD模态子集来构建对$A$的降秩近似。

为此，我们需要使用与表示法3相关的**简化**SVD，而不是与表示法1和2相关的**完整**SVD。

**快速指南：** 如果您想直接应用这些方法，可以直接跳到表示法3。

第一次阅读时，您可以跳过铺垫性的表示法1和2。

+++

## 表示法1

在这个表示法中，我们将使用$X$的**完整**SVD。

我们使用$U$的$m$个**列**，即$U^\top$的$m$个**行**，来定义一个$m \times 1$的向量$\tilde b_t$：

$$
\tilde b_t = U^\top  X_t .
$$ (eq:tildeXdef2)

原始数据$X_t$可以表示为：

$$ 
X_t = U \tilde b_t
$$ (eq:Xdecoder)

（这里我们使用$b$来提醒自己我们正在创建一个**基**向量。）

由于我们现在使用的是**完全**SVD，$U U^\top  = I_{m \times m}$。

因此从方程{eq}`eq:tildeXdef2`可以得出，我们可以用$\tilde b_t$重新构造$X_t$。

特别地，

* 方程 {eq}`eq:tildeXdef2` 作为一个**编码器**，将 $m \times 1$ 向量 $X_t$ **旋转**成一个 $m \times 1$ 的向量 $\tilde b_t$
  
* 方程 {eq}`eq:Xdecoder` 作为一个**解码器**，通过旋转 $m \times 1$ 向量 $\tilde b_t$ 来**重新构造** $m \times 1$ 的向量 $X_t$

为 $m \times 1$ 的基向量 $\tilde b_t$ 定义一个转移矩阵：

$$ 
\tilde A = U^\top  \hat A U 
$$ (eq:Atilde0)

我们可以通过以下方式表示 $\hat A$：

$$
\hat A = U \tilde A U^\top  
$$

$m \times 1$ 的基向量 $\tilde b_t$ 的动态由以下方程支配：

$$
\tilde b_{t+1} = \tilde A \tilde b_t 
$$

为了构建基于 $X_1$ 的 $X_t$ 未来值的预测 $\overline X_t$，我们可以对这个方程的两边应用解码器（即旋转器），从而推导出：

$$
\overline X_{t+1} = U \tilde A^t U^\top  X_1
$$

这里我们用 $\overline X_{t+1}, t \geq 1$ 表示预测值。

+++

## 表示法 2

这种表示方法与{cite}`schmid2010`最初提出的方法有关。

它可以被视为推导表示3的一个中间步骤。

与表示1一样，我们继续：

* 使用**完整**SVD而**不是**简化SVD

在{doc}`奇异值分解<svd_intro>`课程中我们学到:

  * (a) 对于完整SVD，$U U^\top = I_{m \times m}$和$U^\top U = I_{p \times p}$都是单位矩阵
  
  * (b) 对于$X$的简化SVD，$U^\top U$不是单位矩阵。

这个区别很重要，因为我们后面会使用简化SVD而不是完整SVD。这意味着我们需要处理$U^\top U$不是单位矩阵的情况。

但现在，让我们假设我们使用的是完整SVD，这样条件(a)和(b)都得到满足。

对方程{eq}`eq:Atilde0`中定义的$m \times m$矩阵$\tilde A = U^\top  \hat A U$进行特征分解：

$$
\tilde A = W \Lambda W^{-1} 
$$ (eq:tildeAeigen)

其中$\Lambda$是特征值的对角矩阵，$W$是一个$m \times m$的矩阵，其每一列都对应于$\Lambda$中行(特征值)的特征向量。

当$U U^\top  = I_{m \times m}$时（这在$X$的完全SVD中是成立的），可得：

$$ 
\hat A = U \tilde A U^\top  = U W \Lambda W^{-1} U^\top  
$$ (eq:eqeigAhat)

根据方程{eq}`eq:eqeigAhat`，对角矩阵$\Lambda$包含$\hat A$的特征值，而$\hat A$对应的特征向量是矩阵$UW$的列。

因此，我们的一阶向量自回归所捕获的$X_t$动态的系统部分（即非随机部分）可以描述为：

$$
X_{t+1} = U W \Lambda W^{-1} U^\top   X_t 
$$

将上述方程两边同时乘以$W^{-1} U^\top $，得到：

$
W^{-1} U^\top  X_{t+1} = \Lambda W^{-1} U^\top  X_t 
$$

或

$$
\hat b_{t+1} = \Lambda \hat b_t
$$

其中，我们的**编码器**是

$$ 
\hat b_t = W^{-1} U^\top  X_t
$$

我们的**解码器**是

$$
X_t = U W \hat b_t
$$

我们可以使用这种表示来构建一个基于$X_1$的$X_{t+1}$的预测器$\overline X_{t+1}$：

$$
\overline X_{t+1} = U W \Lambda^t W^{-1} U^\top  X_1 
$$ (eq:DSSEbookrepr)

实际上，
{cite}`schmid2010`定义了一个$m \times m$的矩阵$\Phi_s$为

$$ 
\Phi_s = UW 
$$ (eq:Phisfull)

和一个广义逆矩阵

$$
\Phi_s^+ = W^{-1}U^\top  
$$ (eq:Phisfullinv)

{cite}`schmid2010`随后将方程{eq}`eq:DSSEbookrepr`表示为

$$
\overline X_{t+1} = \Phi_s \Lambda^t \Phi_s^+ X_1 
$$ (eq:schmidrep)

基向量的分量$ \hat b_t = W^{-1} U^\top  X_t \equiv \Phi_s^+ X_t$是
DMD**投影模态**。

要理解为什么它们被称为**投影模态**，注意到

$$ 
\Phi_s^+ = ( \Phi_s^\top  \Phi_s)^{-1} \Phi_s^\top 
$$

所以 $m \times p$ 的矩阵

$$
\hat b =  \Phi_s^+ X
$$ 

是 $m \times n$ 矩阵 $X$ 在 $m \times p$ 矩阵 $\Phi_s$ 上的回归系数矩阵。

我们将在讨论由 Tu 等人 {cite}`tu_Rowley` 提出的表示法3时进一步讨论。

当我们想要使用简化SVD时（这在实践中经常出现），使用表示法3更为合适。

## 表示法3

与构建表示法1和表示法2的程序不同（它们都使用了**完整**SVD），我们现在使用**简化**SVD。

同样，令 $p \leq \textrm{min}(m,n)$ 为 $X$ 的秩。

构造一个**简化**SVD

$$
X = \tilde U \tilde \Sigma \tilde V^\top , 
$$

其中现在 $\tilde U$ 是 $m \times p$ 的矩阵，$\tilde \Sigma$ 是 $p \times p$ 的矩阵，而 $\tilde V^\top$ 是 $p \times n$ 的矩阵。

我们的 $A$ 的最小范数最小二乘近似器现在的表示为

$$

\hat A = X' \tilde V \tilde \Sigma^{-1} \tilde U^\top 
$$ (eq:Ahatwithtildes)


**计算$\hat A$的主要特征向量**

我们首先参照构建表示法1时使用的步骤，通过以下方式为旋转的$p \times 1$状态$\tilde b_t$定义一个转移矩阵：

$$ 
\tilde A =\tilde  U^\top  \hat A \tilde U 
$$ (eq:Atildered)


**作为投影系数的解释**


{cite}`DDSE_book`指出$\tilde A$可以被理解为$\hat A$在$\tilde U$中$p$个模态上的投影。

要验证这一点，首先注意到，由于$ \tilde U^\top  \tilde U = I$，因此：

$$
\tilde A = \tilde U^\top  \hat A \tilde U = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top  \tilde U 
= \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top 
$$ (eq:tildeAverify)


接下来，我们将使用标准最小二乘公式计算$\hat A$在$\tilde U$上的投影的回归系数

$$
(\tilde U^\top  \tilde U)^{-1} \tilde U^\top  \hat A = (\tilde U^\top  \tilde U)^{-1} \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top  = 
\tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde U^\top   = \tilde A .
$$

至此，我们验证了$\tilde A$是$\hat A$在$\tilde U$上的最小二乘投影。

**一个逆运算的挑战**

因为我们使用的是简化SVD，所以$\tilde U \tilde U^\top  \neq I$。

因此，

$$
\hat A \neq \tilde U \tilde A \tilde U^\top ,
$$

所以我们不能简单地用$\tilde A$和$\tilde U$计算$\hat A$。

**死胡同**

我们可以先抱着最好的希望，继续构造$p \times p$矩阵$\tilde A$的特征分解：

$$
 \tilde A =  \tilde  W  \Lambda \tilde  W^{-1} 
$$ (eq:tildeAeigenred)

其中$\Lambda$是包含$p$个特征值的对角矩阵，$\tilde W$的列是对应的特征向量。

仿照表示法2中的步骤，我们可以轻松计算出一个$m \times p$的矩阵

$$
\tilde \Phi_s = \tilde U \tilde W
$$ (eq:Phisred)

该矩阵对应到完整SVD中的{eq}`eq:Phisfull`。

此时，当$\hat A$由公式{eq}`eq:Ahatwithtildes`给出时，计算$\hat A \tilde \Phi_s$很有意思：

$$
\begin{aligned}
\hat A \tilde \Phi_s & = (X' \tilde V \tilde \Sigma^{-1} \tilde U^\top ) (\tilde U \tilde W) \\
  & = X' \tilde V \tilde \Sigma^{-1} \tilde  W \\
  & \neq (\tilde U \tilde  W) \Lambda \\
  & = \tilde \Phi_s \Lambda
  \end{aligned}
$$

$\hat A \tilde \Phi_s \neq \tilde \Phi_s \Lambda$意味着，与表示法2中的相应情况不同，$\tilde \Phi_s = \tilde U \tilde W$的列**不是**$\hat A$对应于矩阵$\Lambda$对角线上特征值的特征向量。

**一种可行的方法**

我们继续寻找**能够**通过简化SVD计算的$\hat A$的特征向量，这里不妨定义一个$m \times p$的矩阵$\Phi$：

$$
\Phi \equiv \hat A \tilde \Phi_s = X' \tilde V \tilde \Sigma^{-1}  \tilde  W
$$ (eq:Phiformula)

不难发现，$\Phi$的列**确实是**$\hat A$的特征向量。

这是Tu等人{cite}`tu_Rowley`证明的一个结果，我们下面来介绍。


**命题** $\Phi$的$p$列是$\hat A$的特征向量。

**证明：** 根据公式{eq}`eq:Phiformula`，我们有

$$  
\begin{aligned}
  \hat A \Phi & =  (X' \tilde  V \tilde  \Sigma^{-1} \tilde  U^\top ) (X' \tilde  V \Sigma^{-1} \tilde  W) \cr
  & = X' \tilde V \tilde  \Sigma^{-1} \tilde A \tilde  W \cr
  & = X' \tilde  V \tilde  \Sigma^{-1}\tilde  W \Lambda \cr
  & = \Phi \Lambda 
  \end{aligned}
$$ 

因此

$$  
\hat A \Phi = \Phi \Lambda 
$$ (eq:APhiLambda)

令 $\phi_i$ 为 $\Phi$ 的第 $i$ 列，$\lambda_i$ 为分解式 {eq}`eq:tildeAeigenred` 中 $\tilde A$ 对应的第 $i$ 个特征值。

将等式 {eq}`eq:APhiLambda` 两边的 $m \times 1$ 向量对应项相等得到：

$$
\hat A \phi_i = \lambda_i \phi_i .
$$

这个等式证实了 $\phi_i$ 是 $\hat A$ 的特征向量，对应于 $\tilde A$ 和 $\hat A$ 的特征值 $\lambda_i$。

证明至此完成。

另见 {cite}`DDSE_book` (第238页)。


### $\check b$ 的解码器作为线性投影

根据特征分解 {eq}`eq:APhiLambda` ，我们可以将 $\hat A$ 表示为：

$$ 
\hat A = \Phi \Lambda \Phi^+ .
$$ (eq:Aform12)

从公式 {eq}`eq:Aform12` 我们可以推导出 $p \times 1$ 向量 $\check b_t$ 的动态：

$$ 
\check b_{t+1} = \Lambda \check b_t 
$$

其中

$$
\check b_t  = \Phi^+ X_t  
$$ (eq:decoder102)

由于 $m \times p$ 矩阵 $\Phi$ 有 $p$ 个线性独立的列，$\Phi$ 的广义逆矩阵为

$$
\Phi^{+} = (\Phi^\top  \Phi)^{-1} \Phi^\top 
$$

因此

$$ 
\check b = (\Phi^\top  \Phi)^{-1} \Phi^\top  X
$$ (eq:checkbform)

$p \times n$ 矩阵 $\check b$ 可以被视为是 $m \times n$ 的矩阵 $X$ 在 $m \times p$ 的矩阵 $\Phi$ 上的最小二乘回归系数矩阵，因此

$$
\check X = \Phi \check b
$$ (eq:Xcheck_)

是 $X$ 在 $\Phi$ 上的最小二乘投影的 $m \times n$ 矩阵。

**$X$ 的方差分解**

根据这个 QuantEcon 讲座 <https://python-advanced.quantecon.org/orth_proj.html> 中讨论的最小二乘的投影理论，我们可以将 $X$ 表示为 $X$ 在 $\Phi$ 上的投影 $\check X$ 和误差矩阵的和。

要验证这一点，注意到最小二乘投影 $\check X$ 与 $X$ 的关系是

$$ 
X = \check X + \epsilon 
$$

或

$$
X = \Phi \check b + \epsilon

$$ (eq:Xbcheck)

其中 $\epsilon$ 是一个 $m \times n$ 的最小二乘误差矩阵，满足最小二乘正交条件 $\epsilon^\top \Phi =0$ 或

$$ 
(X - \Phi \check b)^\top \Phi = 0_{m \times p}
$$ (eq:orthls)

重新整理正交条件 {eq}`eq:orthls` 得到 $X^\top \Phi = \check b \Phi^\top \Phi$，这就推导出公式 {eq}`eq:checkbform`。


### 一种近似方法

我们现在描述一种不使用公式 {eq}`eq:decoder102`的近似计算 $p \times 1$ 的向量 $\check b_t$ 的方法。

具体来说，以下论述改编自 {cite}`DDSE_book`（第240页）提供的一种高效计算方法，从而近似 $\check b_t$。

为方便起见，我们将在时间 $t=1$ 应用该方法。

对于 $t=1$，根据方程 {eq}`eq:Xbcheck`，我们有

$$ 
   \check X_1 = \Phi \check b_1
$$ (eq:X1proj)

其中 $\check b_1$ 是一个 $p \times 1$ 的向量。

回顾上面表示1中的 $X_1 = U \tilde b_1$,其中 $\tilde b_1$ 是表示1的时间1基向量,而 $U$ 来自完整SVD分解 $X = U \Sigma V^\top$。

从方程 {eq}`eq:Xbcheck` 可以得出:

$$ 
  U \tilde b_1 = X' \tilde V \tilde \Sigma^{-1} \tilde  W \check b_1 + \epsilon_1
$$

其中 $\epsilon_1$ 是方程 {eq}`eq:Xbcheck` 中的最小二乘误差向量。

因此可得:

$$
\tilde b_1 = U^\top  X' V \tilde \Sigma^{-1} \tilde W \check b_1 + U^\top  \epsilon_1
$$

将误差项 $U^\top  \epsilon_1$ 替换为零,并将完整SVD中的 $U$ 替换为简化SVD中的 $\tilde U$,我们得到 $\tilde b_1$ 的近似值 $\hat b_1$:

$$ 
  \hat b_1 = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1} \tilde  W \check b_1
$$

回顾方程 {eq}`eq:tildeAverify` 中的 $ \tilde A = \tilde U^\top  X' \tilde V \tilde \Sigma^{-1}$。

因此可得:

$$

\hat  b_1 = \tilde   A \tilde W \check b_1
$$

因此，根据 $\tilde A$ 的特征分解 {eq}`eq:tildeAeigenred`，我们有

$$ 
  \hat b_1 = \tilde W \Lambda \check b_1
$$ 

因此，
  
$$ 
  \hat b_1 = ( \tilde W \Lambda)^{-1} \tilde b_1
$$ 

或者

$$ 
   \hat b_1 = ( \tilde W \Lambda)^{-1} \tilde U^\top  X_1 ,
$$ (eq:beqnsmall)

这是对以下方程 {eq}`eq:decoder102` 中初始向量 $\check b_1$ 的计算效率较高的近似：

$$
  \check b_1= \Phi^{+} X_1
$$ (eq:bphieqn)

（为了强调 {eq}`eq:beqnsmall` 是一个近似值，DMD的使用者有时将基向量 $\check b_t  = \Phi^+ X_t $ 的分量称为**精确**DMD模态，将 $\hat b_t = ( \tilde W \Lambda)^{-1} \tilde U^\top  X_t$ 的分量称为**近似**模态。）

在给定 $X_t$ 的条件下，我们可以通过精确模态计算解码后的 $\check X_{t+j},   j = 1, 2, \ldots $ ：

$$
\check X_{t+j} = \Phi \Lambda^j \Phi^{+} X_t

$$ (eq:checkXevoln)


或者通过近似模态计算解码的 $\hat X_{t+j}$:

$$ 
  \hat X_{t+j} = \Phi \Lambda^j (\tilde W \Lambda)^{-1}  \tilde U^\top  X_t .
$$ (eq:checkXevoln2)

然后我们可以使用解码后的 $\check X_{t+j}$ 或 $\hat X_{t+j}$ 来预测 $X_{t+j}$。


### 使用更少的模态

在实际应用中，我们通常只使用少数几个模态，通常不多于三个。

前面的一些公式假设中，我们保留了与 $X$ 的奇异值相关的所有 $p$ 个模态。

我们可以调整公式，描述只保留 $r < p$ 个最大奇异值的情况。

在这种情况下，我们只需将 $\tilde \Sigma$ 替换为相应的 $r\times r$ 奇异值矩阵，将 $\tilde U$ 替换为对应于 $r$ 个最大奇异值的 $m \times r$ 的矩阵，将 $\tilde V$ 替换为对应于 $r$ 个最大奇异值的 $n \times r$ 的矩阵。

上述所有重要公式都有其对应的形式。


## Python代码来源

你可以在[这里](https://mathlab.sissa.it/pydmd)找到DMD的Python实现。
