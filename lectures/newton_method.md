---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(newton_method)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```
```{index} single: python
```

# 使用牛顿法求解经济模型

```{contents} 目录
:depth: 2
```

```{seealso}
**GPU:** 这个讲座的一个使用[jax](https://jax.readthedocs.io)在`GPU`上运行代码的版本[可在此处获得](https://jax.quantecon.org/newtons_method.html)
```

## 概述

许多经济问题涉及寻找[不动点](https://baike.baidu.com/item/%E4%B8%8D%E5%8A%A8%E7%82%B9?fromModule=lemma_search-box)或[零点](https://baike.baidu.com/item/%E9%9B%B6%E7%82%B9/19736260?fromModule=lemma_search-box)（有时也称为"根"）。

例如，在简单的供需模型中，均衡价格是使超额需求为零的价格。

换句话说，均衡是超额需求函数的零点。

有各种计算技术可用于求解不动点和零点。

在本讲中，我们将学习一种重要的基于梯度的技术，称为[牛顿法](https://baike.baidu.com/item/%E7%89%9B%E9%A1%BF%E8%BF%AD%E4%BB%A3%E6%B3%95?fromModule=lemma_search-box)。

牛顿法并非总是有效，但在适用的情况下，其收敛速度通常比其他方法更快。

本讲将在一维和多维环境中应用牛顿法来解决不动点和零点查找问题。

* 在寻找函数$f$的不动点时，牛顿法通过求解一个函数 $f$ 的线性近似。

* 在寻找函数 $f$ 的零点时，牛顿法通过求解函数 $f$ 的线性近似的零点来更新
  现有的猜测值。

为了建立直观认识，我们首先考虑一个简单的一维不动点问题，其中我们已知解，并使用连续
近似和牛顿法来求解。

然后我们将牛顿法应用到多维环境中，求解多种商品的市场均衡。

在讲座最后，我们利用 [`autograd`](https://github.com/HIPS/autograd) 中自动微分的强大功能来求解一个非常高维的均衡问题。

```{code-cell} ipython3
:tags: [hide-output]

!pip install autograd
```

我们在本讲中使用以下导入语句

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from collections import namedtuple
from scipy.optimize import root
from autograd import jacobian
# 经过简单封装的numpy，以支持自动微分
import autograd.numpy as np

plt.rcParams["figure.figsize"] = (10, 5.7)
```

## 用牛顿法计算不动点

在本节中，我们将在[索洛增长模型](https://baike.baidu.com/item/%E6%96%B0%E5%8F%A4%E5%85%B8%E5%A2%9E%E9%95%BF%E6%A8%A1%E5%9E%8B?fromtitle=%E7%B4%A2%E6%B4%9B%E5%A2%9E%E9%95%BF%E6%A8%A1%E5%9E%8B&fromid=7557049&fromModule=lemma_search-box)的框架下求解资本运动规律的不动点。

我们将通过可视化方式检查不动点，用连续逼近法求解，然后应用牛顿法来实现更快的收敛。

(solow)=
### 索洛模型

在索洛增长模型中，假设采用柯布-道格拉斯生产技术且人口零增长，资本的运动规律为

```{math}
:label: motion_law
    k_{t+1} = g(k_t) \quad \text{where} \quad
    g(k) := sAk^\alpha + (1-\delta) k
```

其中

- $k_t$ 是人均资本存量
- $A, \alpha>0$ 是生产参数，$\alpha<1$
- $s>0$ 是储蓄率
- $\delta \in(0,1)$ 是折旧率

在这个例子中，我们希望计算资本运动规律$g$的唯一严格正不动点。

换句话说，我们要寻找一个 $k^* > 0$ 使得 $g(k^*)=k^*$。

* 这样的 $k^*$ 被称为[稳态](https://zh.wikipedia.org/wiki/%E7%A9%A9%E6%85%8B_(%E7%B3%BB%E7%B5%B1))，
  因为当 $k_t = k^*$ 时意味着 $k_{t+1} = k^*$。

用纸笔解方程 $g(k)=k$，你可以验证

$$ k^* = \left(\frac{s A}{δ}\right)^{1/(1 - α)}  $$

### 实现

让我们使用 [`namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple) 来存储我们的参数，这有助于保持代码的整洁和简洁。

```{code-cell} ipython3
SolowParameters = namedtuple("SolowParameters", ('A', 's', 'α', 'δ'))
```

此函数创建一个带有默认参数值的适当的`namedtuple`。

```{code-cell} ipython3
def create_solow_params(A=2.0, s=0.3, α=0.3, δ=0.4):
    "创建一个带有默认值的索洛模型参数化。"
    return SolowParameters(A=A, s=s, α=α, δ=δ)
```

接下来的两个函数实现运动定律[](motion_law)并存储真实的不动点$k^*$。

```{code-cell} ipython3
def g(k, params):
    A, s, α, δ = params
    return A * s * k**α + (1 - δ) * k
    
def exact_fixed_point(params):
    A, s, α, δ = params
    return ((s * A) / δ)**(1/(1 - α))
```

这是一个用于绘制45度动态图的函数。

```{code-cell} ipython3
def plot_45(params, ax, fontsize=14):
    
    k_min, k_max = 0.0, 3.0
    k_grid = np.linspace(k_min, k_max, 1200)

    # 绘制函数
    lb = r"$g(k) = sAk^{\alpha} + (1 - \delta)k$"
    ax.plot(k_grid, g(k_grid, params),  lw=2, alpha=0.6, label=lb)
    ax.plot(k_grid, k_grid, "k--", lw=1, alpha=0.7, label="45")

    # 显示并标注固定点
    kstar = exact_fixed_point(params)
    fps = (kstar,)
    ax.plot(fps, fps, "go", ms=10, alpha=0.6)
    ax.annotate(r"$k^* = (sA / \delta)^{\frac{1}{1-\alpha}}$", 
             xy=(kstar, kstar),
             xycoords="data",
             xytext=(20, -20),
             textcoords="offset points",
             fontsize=fontsize)

    ax.legend(loc="upper left", frameon=False, fontsize=fontsize)

    ax.set_yticks((0, 1, 2, 3))
    ax.set_yticklabels((0.0, 1.0, 2.0, 3.0), fontsize=fontsize)
    ax.set_ylim(0, 3)
    ax.set_xlabel("$k_t$", fontsize=fontsize)
    ax.set_ylabel("$k_{t+1}$", fontsize=fontsize)
```

让我们看看两个参数化的45度图。

```{code-cell} ipython3
params = create_solow_params()
fig, ax = plt.subplots(figsize=(8, 8))
plot_45(params, ax)
plt.show()
```

```{code-cell} ipython3
params = create_solow_params(α=0.05, δ=0.5)
fig, ax = plt.subplots(figsize=(8, 8))
plot_45(params, ax)
plt.show()
```

我们看到 $k^*$ 确实是唯一的正固定点。


#### 连续近似法

首先让我们用连续近似法来计算固定点。

在这种情况下，连续近似法意味着从某个初始状态 $k_0$ 开始，使用运动规律反复更新资本。

这里是从特定选择的 $k_0$ 得到的时间序列。

```{code-cell} ipython3
def compute_iterates(k_0, f, params, n=25):
    "计算由任意函数f生成的长度为n的时间序列。"
    k = k_0
    k_iterates = []
    for t in range(n):
        k_iterates.append(k)
        k = f(k, params)
    return k_iterates
```

```{code-cell} ipython3
params = create_solow_params()
k_0 = 0.25
k_series = compute_iterates(k_0, g, params)
k_star = exact_fixed_point(params)

fig, ax = plt.subplots()
ax.plot(k_series, 'o')
ax.plot([k_star] * len(k_series), 'k--')
ax.set_ylim(0, 3)
plt.show()
```

让我们看看长时间序列的输出。

```{code-cell} ipython3
k_series = compute_iterates(k_0, g, params, n=10_000)
k_star_approx = k_series[-1]
k_star_approx
```

这接近真实值。


```{code-cell} ipython3
k_star
```

#### 牛顿法

一般来说，当对某个函数$g$应用牛顿不动点法时，我们从一个不动点的猜测值$x_0$开始，然后通过求解$x_0$处切线的不动点来更新。

首先，我们回顾一下$g$在$x_0$处的一阶近似（即$g$在$x_0$处的一阶泰勒近似）是以下函数：

```{math}
:label: motivation

\hat g(x) \approx g(x_0)+g'(x_0)(x-x_0)
```

我们通过计算满足以下等式的$x_1$来求解$\hat g$的不动点：

$$
x_1=\frac{g(x_0)-g'(x_0) x_0}{1-g'(x_0)}
$$

推广上述过程，牛顿不动点法的迭代公式为：

```{math}
:label: newtons_method

x_{t+1} = \frac{g(x_t) - g'(x_t) x_t}{ 1 - g'(x_t) },
\quad x_0 \text{ 给定}
```

要实现牛顿法，我们观察到资本运动定律[](motion_law)的导数为：

```{math}
:label: newton_method2

g'(k) = \alpha s A k^{\alpha-1} + (1-\delta)

```

让我们定义这个：

```{code-cell} ipython3
def Dg(k, params):
    A, s, α, δ = params
    return α * A * s * k**(α-1) + (1 - δ)
```

这里有一个函数 $q$ 表示 [](newtons_method)。

```{code-cell} ipython3
def q(k, params):
    return (g(k, params) - Dg(k, params) * k) / (1 - Dg(k, params))
```

现在让我们绘制一些轨迹。

```{code-cell} ipython3
def plot_trajectories(params, 
                      k0_a=0.8,  # 第一个初始条件
                      k0_b=3.1,  # 第二个初始条件
                      n=20,      # 时间序列长度
                      fs=14):    # 字体大小

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax1, ax2 = axes

    ks1 = compute_iterates(k0_a, g, params, n)
    ax1.plot(ks1, "-o", label="连续近似")

    ks2 = compute_iterates(k0_b, g, params, n)
    ax2.plot(ks2, "-o", label="连续近似")

    ks3 = compute_iterates(k0_a, q, params, n)
    ax1.plot(ks3, "-o", label="牛顿步骤")

    ks4 = compute_iterates(k0_b, q, params, n)
    ax2.plot(ks4, "-o", label="牛顿步骤")

    for ax in axes:
        ax.plot(k_star * np.ones(n), "k--")
        ax.legend(fontsize=fs, frameon=False)
        ax.set_ylim(0.6, 3.2)
        ax.set_yticks((k_star,))
        ax.set_yticklabels(("$k^*$",), fontsize=fs)
        ax.set_xticks(np.linspace(0, 19, 20))
        
    plt.show()
```

```{code-cell} ipython3
params = create_solow_params()
plot_trajectories(params)
```

我们可以看到牛顿法比连续逼近法收敛得更快。


##  一维求根

在上一节中我们计算了不动点。

事实上，牛顿法更常与寻找函数零点的问题相关联。

让我们讨论这个"求根"问题，然后说明它与寻找不动点的问题是如何联系的。



### 牛顿法求零点

假设我们想要找到一个 $x$ 使得对某个光滑函数 $f$ (从实数映射到实数)有 $f(x)=0$。

假设我们有一个猜测值 $x_0$ 并且想要将其更新为新的点 $x_1$。

作为第一步，我们取 $f$ 在 $x_0$ 处的一阶近似：

$$
\hat f(x) \approx f\left(x_0\right)+f^{\prime}\left(x_0\right)\left(x-x_0\right)
$$

现在我们求解 $\hat f$ 的零点。

具体来说，我们令 $\hat{f}(x_1) = 0$ 并求解 $x_1$，得到

$$
x_1 = x_0 - \frac{ f(x_0) }{ f'(x_0) },
\quad x_0 \text{ 已知}
$$

对于一维零点查找问题，牛顿法的迭代公式可以概括为

```{math}
:label: oneD-newton
x_{t+1} = x_t - \frac{ f(x_t) }{ f'(x_t) },
\quad x_0 \text{ 给定}
```

以下代码实现了迭代公式 [](oneD-newton)

```{code-cell} ipython3
def newton(f, Df, x_0, tol=1e-7, max_iter=100_000):
    x = x_0

    # 实现零点查找公式
    def q(x):
        return x - f(x) / Df(x)

    error = tol + 1
    n = 0
    while error > tol:
        n += 1
        if(n > max_iter):
            raise Exception('达到最大迭代次数但未收敛')
        y = q(x)
        error = np.abs(x - y)
        x = y
        print(f'迭代 {n}, 误差 = {error:.5f}')
    return x
```

许多库都实现了一维牛顿法，包括SciPy，所以这里的代码仅作说明用途。

（话虽如此，当我们想要使用自动微分或GPU加速等技术来应用牛顿法时，了解如何自己实现牛顿法会很有帮助。）

### 在寻找不动点中的应用

现在再次考虑索洛不动点计算，我们要求解满足$g(k) = k$的$k$值。

我们可以通过设定$f(x) := g(x)-x$将其转换为零点寻找问题。

显然，$f$的任何零点都是$g$的不动点。

让我们将这个想法应用到索洛问题中

```{code-cell} ipython3
params = create_solow_params()
k_star_approx_newton = newton(f=lambda x: g(x, params) - x,
                              Df=lambda x: Dg(x, params) - 1,
                              x_0=0.8)
```

```{code-cell} ipython3
k_star_approx_newton
```

结果证实了我们在上面图表中看到的收敛情况：仅需5次迭代就达到了非常精确的结果。



## 多元牛顿法

在本节中，我们将介绍一个双商品问题，展示问题的可视化，并使用`SciPy`中的零点查找器和牛顿法来求解这个双商品市场的均衡。

然后，我们将这个概念扩展到一个包含5,000种商品的更大市场，并再次比较这两种方法的性能。

我们将看到使用牛顿法时能获得显著的性能提升。


### 双商品市场均衡

让我们从计算双商品问题的市场均衡开始。

我们考虑一个包含两种相关产品的市场，商品0和商品1，价格向量为$p = (p_0, p_1)$

在价格$p$下，商品$i$的供给为，

$$ 
q^s_i (p) = b_i \sqrt{p_i} 
$$

在价格$p$下，商品$i$的需求为，

$$ 
q^d_i (p) = \exp(-(a_{i0} p_0 + a_{i1} p_1)) + c_i
$$

这里的$c_i$、$b_i$和$a_{ij}$都是参数。

例如，这两种商品可能是通常一起使用的计算机组件，在这种情况下它们是互补品。因此需求取决于两种组件的价格。

超额需求函数为，

$$
e_i(p) = q^d_i(p) - q^s_i(p), \quad i = 0, 1
$$

均衡价格向量$p^*$满足$e_i(p^*) = 0$。

我们设定

$$
A = \begin{pmatrix}
            a_{00} & a_{01} \\
            a_{10} & a_{11}
        \end{pmatrix},
            \qquad 
    b = \begin{pmatrix}
            b_0 \\
            b_1
        \end{pmatrix}
    \qquad \text{和} \qquad
    c = \begin{pmatrix}
            c_0 \\
            c_1
        \end{pmatrix}
$$

用于这个特定问题。

#### 图形化探索

由于我们的问题只是二维的，我们可以使用图形分析来可视化并帮助理解这个问题。

我们的第一步是定义超额需求函数

$$
e(p) = 
    \begin{pmatrix}
    e_0(p) \\
    e_1(p)
    \end{pmatrix}
$$

下面的函数计算给定参数的超额需求

```{code-cell} ipython3
def e(p, A, b, c):
    return np.exp(- A @ p) + c - b * np.sqrt(p)
```

我们的默认参数值将是


$$
A = \begin{pmatrix}
            0.5 & 0.4 \\
            0.8 & 0.2
        \end{pmatrix},
            \qquad 
    b = \begin{pmatrix}
            1 \\
            1
        \end{pmatrix}
    \qquad \text{和} \qquad
    c = \begin{pmatrix}
            1 \\
            1
        \end{pmatrix}
$$

```{code-cell} ipython3
A = np.array([
    [0.5, 0.4],
    [0.8, 0.2]
])
b = np.ones(2)
c = np.ones(2)
```

在价格水平 $p = (1, 0.5)$ 时，超额需求为

```{code-cell} ipython3
ex_demand = e((1.0, 0.5), A, b, c)

print(f'商品0的超额需求为 {ex_demand[0]:.3f} \n'
      f'商品1的超额需求为 {ex_demand[1]:.3f}')
```

接下来我们在$(p_0, p_1)$值的网格上绘制两个函数$e_0$和$e_1$的等高线图和曲面。

我们将使用以下函数来构建等高线图

```{code-cell} ipython3
def plot_excess_demand(ax, good=0, grid_size=100, grid_max=4, surface=True):

    # Create a 100x100 grid
    p_grid = np.linspace(0, grid_max, grid_size)
    z = np.empty((100, 100))

    for i, p_1 in enumerate(p_grid):
        for j, p_2 in enumerate(p_grid):
            z[i, j] = e((p_1, p_2), A, b, c)[good]

    if surface:
        cs1 = ax.contourf(p_grid, p_grid, z.T, alpha=0.5)
        plt.colorbar(cs1, ax=ax, format="%.6f")

    ctr1 = ax.contour(p_grid, p_grid, z.T, levels=[0.0])
    ax.set_xlabel("$p_0$")
    ax.set_ylabel("$p_1$")
    ax.set_title(f'Excess Demand for Good {good}')
    plt.clabel(ctr1, inline=1, fontsize=13)
```

这是我们对 $e_0$ 的绘图：

```{code-cell} ipython3
fig, ax = plt.subplots()
plot_excess_demand(ax, good=0)
plt.show()
```

这是我们对 $e_1$ 的绘图：

```{code-cell} ipython3
fig, ax = plt.subplots()
plot_excess_demand(ax, good=1)
plt.show()
```

我们看到黑色的零等高线，它告诉我们何时$e_i(p)=0$。

对于使得$e_i(p)=0$的价格向量$p$，我们知道商品$i$处于均衡状态（需求等于供给）。

如果这两条等高线在某个价格向量$p^*$处相交，那么$p^*$就是一个均衡价格向量。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5.7))
for good in (0, 1):
    plot_excess_demand(ax, good=good, surface=False)
plt.show()
```

看起来在 $p = (1.6, 1.5)$ 附近存在一个均衡点。

#### 使用多维根查找器

为了更精确地求解 $p^*$，我们使用 `scipy.optimize` 中的零点查找算法。

我们以 $p = (1, 1)$ 作为初始猜测值。

```{code-cell} ipython3
init_p = np.ones(2)
```

这使用[改进的Powell方法](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html#optimize-root-hybr)来寻找零点

```{code-cell} ipython3
%%time
solution = root(lambda p: e(p, A, b, c), init_p, method='hybr')
```

这是得到的值：

```{code-cell} ipython3
p = solution.x
p
```

这个结果看起来和我们从图中观察到的猜测很接近。我们可以把它代回到 $e$ 中验证 $e(p) \approx 0$：

```{code-cell} ipython3
np.max(np.abs(e(p, A, b, c)))
```

这确实是一个很小的误差。


#### 添加梯度信息

在许多情况下，对于应用于光滑函数的零点查找算法，提供函数的[雅可比矩阵](https://baike.baidu.com/item/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5?fromModule=lemma_search-box)可以带来更好的收敛性质。

这里我们手动计算雅可比矩阵的元素

$$
J(p) = 
    \begin{pmatrix}
        \frac{\partial e_0}{\partial p_0}(p) & \frac{\partial e_0}{\partial p_1}(p) \\
        \frac{\partial e_1}{\partial p_0}(p) & \frac{\partial e_1}{\partial p_1}(p)
    \end{pmatrix}
$$

```{code-cell} ipython3
def jacobian_e(p, A, b, c):
    p_0, p_1 = p
    a_00, a_01 = A[0, :]
    a_10, a_11 = A[1, :]
    j_00 = -a_00 * np.exp(-a_00 * p_0) - (b[0]/2) * p_0**(-1/2)
    j_01 = -a_01 * np.exp(-a_01 * p_1)
    j_10 = -a_10 * np.exp(-a_10 * p_0)
    j_11 = -a_11 * np.exp(-a_11 * p_1) - (b[1]/2) * p_1**(-1/2)
    J = [[j_00, j_01],
         [j_10, j_11]]
    return np.array(J)
```

```{code-cell} ipython3
%%time
solution = root(lambda p: e(p, A, b, c),
                init_p, 
                jac=lambda p: jacobian_e(p, A, b, c), 
                method='hybr')
```

现在的解更加精确了（尽管在这个低维问题中，差异非常小）：

```{code-cell} ipython3
p = solution.x
np.max(np.abs(e(p, A, b, c)))
```

#### 使用牛顿法

现在让我们使用牛顿法来计算均衡价格，采用多变量版本的牛顿法

```{math}
:label: multi-newton

p_{n+1} = p_n - J_e(p_n)^{-1} e(p_n)
```

这是[](oneD-newton)的多变量版本

（这里的$J_e(p_n)$是在$p_n$处计算的$e$的雅可比矩阵。）

迭代从价格向量$p_0$的某个初始猜测开始。

在这里，我们不手动编写雅可比矩阵，而是使用`autograd`库中的`jacobian()`函数来自动求导并计算雅可比矩阵。

只需稍作修改，我们就可以将[我们之前的尝试](first_newton_attempt)推广到多维问题

```{code-cell} ipython3
def newton(f, x_0, tol=1e-5, max_iter=10):
    x = x_0
    q = lambda x: x - np.linalg.solve(jacobian(f)(x), f(x))
    error = tol + 1
    n = 0
    while error > tol:
        n+=1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        if(any(np.isnan(y))):
            raise Exception('Solution not found with NaN generated')
        error = np.linalg.norm(x - y)
        x = y
        print(f'iteration {n}, error = {error:.5f}')
    print('\n' + f'Result = {x} \n')
    return x
```

```{code-cell} ipython3
def e(p, A, b, c):
    return np.exp(- np.dot(A, p)) + c - b * np.sqrt(p)
```

我们发现算法在4步内终止

```{code-cell} ipython3
%%time
p = newton(lambda p: e(p, A, b, c), init_p)
```

```{code-cell} ipython3
np.max(np.abs(e(p, A, b, c)))
```

结果非常准确。

由于较大的开销，速度并不比优化后的 `scipy` 函数更好。

### 高维问题

我们的下一步是研究一个有3,000种商品的大型市场。

使用GPU加速线性代数和自动微分的JAX版本可在[此处](https://jax.quantecon.org/newtons_method.html#application)获取

超额需求函数基本相同，但现在矩阵 $A$ 是 $3000 \times 3000$ 的，参数向量 $b$ 和 $c$ 是 $3000 \times 1$ 的。

```{code-cell} ipython3
dim = 3000
np.random.seed(123)

# 创建随机矩阵A并将行归一化使其和为1
A = np.random.rand(dim, dim)
A = np.asarray(A)
s = np.sum(A, axis=0)
A = A / s

# 设置b和c
b = np.ones(dim)
c = np.ones(dim)
```

这是我们的初始条件

```{code-cell} ipython3
init_p = np.ones(dim)
```

```{code-cell} ipython3
%%time
p = newton(lambda p: e(p, A, b, c), init_p)
```

```{code-cell} ipython3
np.max(np.abs(e(p, A, b, c)))
```

在相同的容差条件下，我们比较牛顿法与SciPy的`root`函数的运行时间和精确度

```{code-cell} ipython3
%%time
solution = root(lambda p: e(p, A, b, c),
                init_p, 
                jac=lambda p: jacobian(e)(p, A, b, c), 
                method='hybr',
                tol=1e-5)
```

```{code-cell} ipython3
p = solution.x
np.max(np.abs(e(p, A, b, c)))
```

## 练习

```{exercise-start}
:label: newton_ex1
```

考虑索洛固定点问题的三维扩展，其中

$$
A = \begin{pmatrix}
            2 & 3 & 3 \\
            2 & 4 & 2 \\
            1 & 5 & 1 \\
        \end{pmatrix},
            \quad
s = 0.2, \quad α = 0.5, \quad δ = 0.8
$$

和之前一样，运动方程为

```{math}
    k_{t+1} = g(k_t) \quad \text{where} \quad
    g(k) := sAk^\alpha + (1-\delta) k
```

但现在 $k_t$ 是一个 $3 \times 1$ 向量。

使用牛顿法求解固定点，初始值如下：

$$
\begin{aligned}
    k1_{0} &= (1, 1, 1) \\
    k2_{0} &= (3, 5, 5) \\
    k3_{0} &= (50, 50, 50)
\end{aligned}
$$

````{hint}
:class: dropdown

- 固定点的计算等价于计算满足 $f(k^*) - k^* = 0$ 的 $k^*$。

- 如果你对你的解决方案不确定，可以从已解决的示例开始：

```{math}
A = \begin{pmatrix}
            2 & 0 & 0 \\
            0 & 2 & 0 \\

0 & 0 & 2 \\
        \end{pmatrix}
```

其中 $s = 0.3$、$α = 0.3$ 和 $δ = 0.4$，初始值为：

```{math}
k_0 = (1, 1, 1)
```

结果应该收敛到[解析解](solved_k)。
````

```{exercise-end}
```


```{solution-start} newton_ex1
:class: dropdown
```

让我们首先定义这个问题的参数

```{code-cell} ipython3
A = np.array([[2.0, 3.0, 3.0],
              [2.0, 4.0, 2.0],
              [1.0, 5.0, 1.0]])

s = 0.2
α = 0.5
δ = 0.8

initLs = [np.ones(3),
          np.array([3.0, 5.0, 5.0]),
          np.repeat(50.0, 3)]
```

然后定义[资本运动定律](motion_law)的多元版本

```{code-cell} ipython3
def multivariate_solow(k, A=A, s=s, α=α, δ=δ):
    return (s * np.dot(A, k**α) + (1 - δ) * k)
```

让我们遍历每个初始值并查看输出结果

```{code-cell} ipython3
attempt = 1
for init in initLs:
    print(f'尝试 {attempt}: 初始值为 {init} \n')
    %time k = newton(lambda k: multivariate_solow(k) - k, \
                    init)
    print('-'*64)
    attempt += 1
```

我们发现，由于这个问题具有明确定义的性质，结果与初始值无关。

但是收敛所需的迭代次数取决于初始值。

让我们把输出结果代回公式中验证我们的最终结果

```{code-cell} ipython3
multivariate_solow(k) - k
```

注意误差非常小。

我们也可以在已知解上测试我们的结果

```{code-cell} ipython3
A = np.array([[2.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 2.0]])

s = 0.3
α = 0.3
δ = 0.4

init = np.repeat(1.0, 3)


%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k, \
                 init)
```

结果与真实值非常接近，但仍有细微差异。

```{code-cell} ipython3
%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k, \
                 init,\
                 tol=1e-7)
```

我们可以看到它正在朝着更精确的解决方案迈进。

```{solution-end}
```


```{exercise-start}
:label: newton_ex2
```

在这个练习中，让我们尝试不同的初始值，看看牛顿法对不同起始点的反应如何。

让我们定义一个具有以下默认值的三商品问题：

$$
A = \begin{pmatrix}
            0.2 & 0.1 & 0.7 \\
            0.3 & 0.2 & 0.5 \\
            0.1 & 0.8 & 0.1 \\
        \end{pmatrix},
            \qquad 
b = \begin{pmatrix}
            1 \\
            1 \\
            1
        \end{pmatrix}
    \qquad \text{和} \qquad
c = \begin{pmatrix}
            1 \\
            1 \\
            1
        \end{pmatrix}
$$

对于这个练习，使用以下极端价格向量作为初始值：

$$

\begin{aligned}
    p1_{0} &= (5, 5, 5) \\
    p2_{0} &= (1, 1, 1) \\
    p3_{0} &= (4.5, 0.1, 4)
\end{aligned}
$$

将容差设置为$0.0$以获得更精确的输出。

```{exercise-end}
```

```{solution-start} newton_ex2

:class: dropdown
```

定义参数和初始值

```{code-cell} ipython3
A = np.array([
    [0.2, 0.1, 0.7],
    [0.3, 0.2, 0.5],
    [0.1, 0.8, 0.1]
])

b = np.array([1.0, 1.0, 1.0])
c = np.array([1.0, 1.0, 1.0])

initLs = [np.repeat(5.0, 3),
          np.ones(3),
          np.array([4.5, 0.1, 4.0])] 
```

让我们检查每个初始猜测值并查看输出结果

```{code-cell} ipython3
---
tags: [raises-exception]
---

attempt = 1
for init in initLs:
    print(f'尝试 {attempt}: 初始值为 {init} \n')
    %time p = newton(lambda p: e(p, A, b, c), \
                init, \
                tol=1e-15, \
                max_iter=15)
    print('-'*64)
    attempt += 1
```

我们可以发现牛顿法对某些初始值可能会失败。

有时可能需要尝试几个初始猜测值才能实现收敛。

将结果代回公式中检验我们的结果

```{code-cell} ipython3
e(p, A, b, c)
```

我们可以看到结果非常精确。

```{solution-end}
```

