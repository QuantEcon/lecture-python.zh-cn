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

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 库存动态

```{index} single: 马尔可夫过程, 库存
```

```{contents} 目录
:depth: 2
```

## 概述

在本讲座中，我们将研究遵循所谓s-S库存动态的企业的库存时间路径。

这些企业

1. 等待直到库存降至某个水平$s$以下，然后
1. 订购足够数量以将库存补充到容量$S$。

这类政策在实践中很常见，并且在某些情况下也是最优的。

早期文献综述和一些宏观经济含义可以在{cite}`caplin1985variability`中找到。

我们的主要目标是学习更多关于模拟、时间序列和马尔可夫动态的知识。

虽然我们的马尔可夫环境和许多我们考虑的概念与{doc}`有限马尔可夫链讲座 <finite_markov>`中的概念相关，但在当前应用中状态空间是连续的。

让我们从一些导入开始

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
from numba import jit, float64, prange
from numba.experimental import jitclass
```

## 样本路径

考虑一个拥有库存 $X_t$ 的公司。

当库存 $X_t \leq s$ 时，公司会补货至 $S$ 单位。

公司面临随机需求 $\{ D_t \}$，我们假设这是独立同分布的。

使用符号 $a^+ := \max\{a, 0\}$，库存动态可以写作

$$
X_{t+1} =
    \begin{cases}
      ( S - D_{t+1})^+ & \quad \text{if } X_t \leq s \\
      ( X_t - D_{t+1} )^+ &  \quad \text{if } X_t > s
    \end{cases}
$$

在下文中，我们将假设每个 $D_t$ 都是对数正态分布的，即

$$
D_t = \exp(\mu + \sigma Z_t)
$$

其中 $\mu$ 和 $\sigma$ 是参数，$\{Z_t\}$ 是独立同分布的标准正态分布。

这里有一个类，用于存储参数并生成库存的时间路径。

```{code-cell} ipython3
firm_data = [
   ('s', float64),          # 补货触发水平
   ('S', float64),          # 容量
   ('mu', float64),         # 冲击位置参数
   ('sigma', float64)       # 冲击规模参数
]


@jitclass(firm_data)
class Firm:

    def __init__(self, s=10, S=100, mu=1.0, sigma=0.5):

        self.s, self.S, self.mu, self.sigma = s, S, mu, sigma

    def update(self, x):
        "根据当前状态x更新t到t+1的状态。"

        Z = np.random.randn()
        D = np.exp(self.mu + self.sigma * Z)
        if x <= self.s:
            return max(self.S - D, 0)
        else:
            return max(x - D, 0)

    def sim_inventory_path(self, x_init, sim_length):

        X = np.empty(sim_length)
        X[0] = x_init

        for t in range(sim_length-1):
            X[t+1] = self.update(X[t])
        return X
```

让我们运行第一个模拟，模拟单个路径：

```{code-cell} ipython3
firm = Firm()

s, S = firm.s, firm.S
sim_length = 100
x_init = 50

X = firm.sim_inventory_path(x_init, sim_length)

fig, ax = plt.subplots()
bbox = (0., 1.02, 1., .102)
legend_args = {'ncol': 3,
               'bbox_to_anchor': bbox,
               'loc': 3,
               'mode': 'expand'}

ax.plot(X, label="库存")
ax.plot(np.full(sim_length, s), 'k--', label="$s$")
ax.plot(np.full(sim_length, S), 'k-', label="$S$")
ax.set_ylim(0, S+10)
ax.set_xlabel("时间")
ax.legend(**legend_args)

plt.show()
```

现在让我们模拟多条路径，以便更全面地了解不同结果的概率：

```{code-cell} ipython3
sim_length=200
fig, ax = plt.subplots()

ax.plot(np.full(sim_length, s), 'k--', label="$s$")
ax.plot(np.full(sim_length, S), 'k-', label="$S$")
ax.set_ylim(0, S+10)
ax.legend(**legend_args)

for i in range(400):
    X = firm.sim_inventory_path(x_init, sim_length)
    ax.plot(X, 'b', alpha=0.2, lw=0.5)

plt.show()
```

## 边际分布

现在让我们来看看某个固定时间点 $T$ 时 $X_T$ 的边际分布 $\psi_T$。

我们将通过在给定初始条件 $X_0$ 的情况下生成多个 $X_T$ 的样本来实现这一点。

通过这些 $X_T$ 的样本，我们可以构建其分布 $\psi_T$ 的图像。

这里是一个可视化示例，其中 $T=50$。

```{code-cell} ipython3
T = 50
M = 200  # 样本数量

ymin, ymax = 0, S + 10

fig, axes = plt.subplots(1, 2, figsize=(11, 6))

for ax in axes:
    ax.grid(alpha=0.4)

ax = axes[0]

ax.set_ylim(ymin, ymax)
ax.set_ylabel('$X_t$', fontsize=16)
ax.vlines((T,), -1.5, 1.5)

ax.set_xticks((T,))
ax.set_xticklabels((r'$T$',))

sample = np.empty(M)
for m in range(M):
    X = firm.sim_inventory_path(x_init, 2 * T)
    ax.plot(X, 'b-', lw=1, alpha=0.5)
    ax.plot((T,), (X[T+1],), 'ko', alpha=0.5)
    sample[m] = X[T+1]

axes[1].set_ylim(ymin, ymax)

axes[1].hist(sample,
             bins=16,
             density=True,
             orientation='horizontal',
             histtype='bar',
             alpha=0.5)

plt.show()
```

通过绘制更多样本，我们可以得到一个更清晰的图像

```{code-cell} ipython3
T = 50
M = 50_000

fig, ax = plt.subplots()

sample = np.empty(M)
for m in range(M):
    X = firm.sim_inventory_path(x_init, T+1)
    sample[m] = X[T]

ax.hist(sample,
         bins=36,
         density=True,
         histtype='bar',
         alpha=0.75)

plt.show()
```

请注意分布呈双峰

* 大多数公司已经补货两次，但少数公司只补货一次（见上图路径）。
* 第二类公司的库存较低。

我们也可以使用[核密度估计](https://en.wikipedia.org/wiki/Kernel_density_estimation)来近似这个分布。

核密度估计可以被理解为平滑的直方图。

当被估计的分布可能是平滑的时候，核密度估计比直方图更可取。

我们将使用[scikit-learn](https://scikit-learn.org/stable/)中的核密度估计器

```{code-cell} ipython3
from sklearn.neighbors import KernelDensity

def plot_kde(sample, ax, label=''):

    xmin, xmax = 0.9 * min(sample), 1.1 * max(sample)
    xgrid = np.linspace(xmin, xmax, 200)
    kde = KernelDensity(kernel='gaussian').fit(sample[:, None])
    log_dens = kde.score_samples(xgrid[:, None])

    ax.plot(xgrid, np.exp(log_dens), label=label)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
plot_kde(sample, ax)
plt.show()
```

概率质量的分配与上面直方图所显示的类似。

## 练习

```{exercise}
:label: id_ex1

这个模型是渐近平稳的，具有唯一的平稳分布。

（有关平稳性的背景讨论，请参见{doc}`我们关于AR(1)过程的讲座 <intro:ar1_processes>`——基本概念是相同的。）

特别是，边际分布序列$\{\psi_t\}$正在收敛到一个唯一的极限分布，该分布不依赖于初始条件。

虽然我们在这里不会证明这一点，但我们可以通过模拟来研究它。

你的任务是根据上述讨论，在时间点$t = 10, 50, 250, 500, 750$生成并绘制序列$\{\psi_t\}$。

（核密度估计器可能是呈现每个分布的最佳方式。）

你应该能看到收敛性，体现在连续分布之间的差异越来越小。

尝试不同的初始条件来验证，从长远来看，分布在不同初始条件下是不变的。
```

```{solution-start} id_ex1
:class: dropdown
```

以下是一个可能的解决方案：

这些计算涉及大量的CPU周期，所以我们试图高效地编写代码。

这意味着编写一个专门的函数，而不是使用上面的类。

```{code-cell} ipython3
s, S, mu, sigma = firm.s, firm.S, firm.mu, firm.sigma

@jit(parallel=True)
def shift_firms_forward(current_inventory_levels, num_periods):

    num_firms = len(current_inventory_levels)
    new_inventory_levels = np.empty(num_firms)

    for f in prange(num_firms):
        x = current_inventory_levels[f]
        for t in range(num_periods):
            Z = np.random.randn()
            D = np.exp(mu + sigma * Z)
            if x <= s:
                x = max(S - D, 0)
            else:
                x = max(x - D, 0)
        new_inventory_levels[f] = x

    return new_inventory_levels
```

```{code-cell} ipython3
x_init = 50
num_firms = 50_000

sample_dates = 0, 10, 50, 250, 500, 750

first_diffs = np.diff(sample_dates)

fig, ax = plt.subplots()

X = np.full(num_firms, x_init)

current_date = 0
for d in first_diffs:
    X = shift_firms_forward(X, d)
    current_date += d
    plot_kde(X, ax, label=f't = {current_date}')

ax.set_xlabel('库存')
ax.set_ylabel('概率')
ax.legend()
plt.show()
```

注意到在 $t=500$ 或 $t=750$ 时密度几乎不再变化。

我们已经得到了平稳密度的合理近似。

你可以通过测试几个不同的初始条件来确信初始条件并不重要。

例如，尝试用所有公司从 $X_0 = 20$ 或 $X_0 = 80$ 开始重新运行上面的代码。

```{solution-end}
```

```{exercise}
:label: id_ex2

使用模拟计算从 $X_0 = 70$ 开始的公司在前50个周期内需要订货两次或更多次的概率。

你需要一个较大的样本量来获得准确的结果。
```

```{solution-start} id_ex2
:class: dropdown
```

这是一个解决方案。

同样，由于计算量相对较大，我们编写了一个专门的函数而不是使用上面的类。

我们还将使用跨公司的并行化处理。

```{code-cell} ipython3
@jit(parallel=True)
def compute_freq(sim_length=50, x_init=70, num_firms=1_000_000):

    firm_counter = 0  # 记录补货2次或以上的公司数量
    for m in prange(num_firms):
        x = x_init
        restock_counter = 0  # 将记录公司m的补货次数

        for t in range(sim_length):
            Z = np.random.randn()
            D = np.exp(mu + sigma * Z)
            if x <= s:
                x = max(S - D, 0)
                restock_counter += 1
            else:
                x = max(x - D, 0)

        if restock_counter > 1:
            firm_counter += 1

    return firm_counter / num_firms
```


记录程序运行所需的时间和输出结果。

```{code-cell} ipython3
%%time

freq = compute_freq()
print(f"至少发生两次缺货的频率 = {freq}")
```

尝试将上面jitted函数中的`parallel`标志改为`False`。

根据你的系统配置，运行速度的差异可能会很大。

（在我们的台式机上，速度提升了5倍。）

```{solution-end}
```

