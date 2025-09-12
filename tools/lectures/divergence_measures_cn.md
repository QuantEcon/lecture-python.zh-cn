---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(divergence_measures)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 统计散度度量

```{contents} 目录
:depth: 2
```

## 概述

统计散度用于量化两个不同概率分布之间的差异，这些差异可能难以区分，原因如下：

  * 在一个分布下具有正概率的每个事件在另一个分布下也具有正概率

  * 这意味着没有"确凿证据"事件的发生能让统计学家确定数据一定服从其中某一个概率分布

统计散度是一个将两个概率分布映射到非负实数的**函数**。

统计散度函数在统计学、信息论和现在许多人称之为"机器学习"的领域中发挥着重要作用。

本讲座描述了三种散度度量：

* **库尔贝克-莱布勒(KL)散度**

* **Jensen–Shannon (JS) 散度**
* **切尔诺夫熵**

这些概念将在多个 quantecon 课程中出现。

让我们首先导入必要的 Python 工具。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, jit
from math import gamma
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import pandas as pd
from IPython.display import display, Math
```

## 熵、交叉熵、KL散度入门

在深入之前,我们先介绍一些有用的概念。

我们暂时假设 $f$ 和 $g$ 是离散随机变量在状态空间 $I = \{1, 2, \ldots, n\}$ 上的两个概率质量函数,满足 $f_i \geq 0, \sum_{i} f_i =1, g_i \geq 0, \sum_{i} g_i =1$。

我们遵循一些统计学家和信息论学家的做法,将从分布 $f$ 中观察到单次抽样 $x = i$ 所关联的**惊奇度**或**惊奇量**定义为

$$
\log\left(\frac{1}{f_i}\right)
$$

他们然后将你预期从单次实现中获得的**信息量**定义为期望惊奇度

$$
H(f) = \sum_i f_i \log\left(\frac{1}{f_i}\right).  
$$

Claude Shannon {cite}`shannon1948mathematical` 将 $H(f)$ 称为分布 $f$ 的**熵**。

```{note}
通过对 $\{f_1, f_2, \ldots, f_n\}$ 在约束 $\sum_i f_i = 1$ 下最大化 $H(f)$,我们可以验证使熵最大化的分布是均匀分布
$
f_i = \frac{1}{n} .
$
均匀分布的熵 $H(f)$ 显然等于 $- \log(n)$。
```

Kullback 和 Leibler {cite}`kullback1951information` 将单次抽样 $x$ 提供的用于区分 $f$ 和 $g$ 的信息量定义为对数似然比

$$
\log \frac{f(x)}{g(x)}
$$

以下两个概念被广泛用于比较两个分布 $f$ 和 $g$。

**交叉熵:**

\begin{equation}
H(f,g) = -\sum_{i} f_i \log g_i
\end{equation}

**KL散度(Kullback-Leibler散度):**
\begin{equation}
D_{KL}(f \parallel g) = \sum_{i} f_i \log\left[\frac{f_i}{g_i}\right]
\end{equation}

这些概念通过以下等式相关联。

$$
D_{KL}(f \parallel g) = H(f,g) - H(f)
$$ (eq:KLcross)

要证明{eq}`eq:KLcross`，注意到

\begin{align}
D_{KL}(f \parallel g) &= \sum_{i} f_i \log\left[\frac{f_i}{g_i}\right] \\
&= \sum_{i} f_i \left[\log f_i - \log g_i\right] \\
&= \sum_{i} f_i \log f_i - \sum_{i} f_i \log g_i \\
&= -H(f) + H(f,g) \\
&= H(f,g) - H(f)
\end{align}

记住$H(f)$是从$f$中抽取$x$时的预期惊异度。

那么上述等式告诉我们，KL散度是当预期$x$从$f$中抽取而实际从$g$中抽取时产生的"额外惊异度"的期望值。

## 两个Beta分布：运行示例

我们将广泛使用Beta分布来说明概念。

Beta分布特别方便，因为它定义在$[0,1]$区间上，通过适当选择其两个参数可以呈现多样的形状。

具有参数$a$和$b$的Beta分布的密度函数由下式给出：

$$
f(z; a, b) = \frac{\Gamma(a+b) z^{a-1} (1-z)^{b-1}}{\Gamma(a) \Gamma(b)}
\quad \text{其中} \quad
\Gamma(p) := \int_{0}^{\infty} x^{p-1} e^{-x} dx
$$

让我们在Python中定义参数和密度函数

```{code-cell} ipython3
# 两个Beta分布的参数
F_a, F_b = 1, 1
G_a, G_b = 3, 1.2

@vectorize
def p(x, a, b):
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x** (a-1) * (1 - x) ** (b-1)

# 两个密度函数
f = jit(lambda x: p(x, F_a, F_b))
g = jit(lambda x: p(x, G_a, G_b))

# 绘制分布图
x_range = np.linspace(0.001, 0.999, 1000)
f_vals = [f(x) for x in x_range]
g_vals = [g(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x) \sim \text{Beta}(1,1)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x) \sim \text{Beta}(3,1.2)$')

# 填充重叠区域
overlap = np.minimum(f_vals, g_vals)
plt.fill_between(x_range, 0, overlap, alpha=0.3, color='purple', label='overlap')

plt.xlabel('x')
plt.ylabel('密度')
plt.legend()
plt.show()
```

(rel_entropy)=
## Kullback–Leibler散度

我们的第一个散度函数是**Kullback–Leibler (KL)散度**。

对于概率密度（或概率质量函数）$f$和$g$，它的定义为

$$
D_{KL}(f\|g) = KL(f, g) = \int f(x) \log \frac{f(x)}{g(x)} \, dx.
$$

我们可以将$D_{KL}(f\|g)$解释为当数据由$f$生成而我们使用$g$时产生的预期超额对数损失（预期超额意外性）。

它有几个重要的性质：

- 非负性（吉布斯不等式）：$D_{KL}(f\|g) \ge 0$，当且仅当$f$几乎处处等于$g$时取等号
- 不对称性：$D_{KL}(f\|g) \neq D_{KL}(g\|f)$（因此它不是度量）
- 信息分解：
  $D_{KL}(f\|g) = H(f,g) - H(f)$，其中$H(f,g)$是交叉熵，$H(f)$是$f$的香农熵
- 链式法则：对于联合分布$f(x, y)$和$g(x, y)$，
  $D_{KL}(f(x,y)\|g(x,y)) = D_{KL}(f(x)\|g(x)) + E_{f}\left[D_{KL}(f(y|x)\|g(y|x))\right]$

KL散度在统计推断中扮演着核心角色，包括模型选择和假设检验。

{doc}`likelihood_ratio_process`描述了KL散度与预期对数似然比之间的联系，
而讲座{doc}`wald_friedman`将其与序贯概率比检验的检验性能联系起来。

让我们计算示例分布$f$和$g$之间的KL散度。

```{code-cell} ipython3
def compute_KL(f, g):
    """
    通过数值积分计算KL散度KL(f, g)
    """
    def integrand(w):
        fw = f(w)
        gw = g(w)
        return fw * np.log(fw / gw)
    val, _ = quad(integrand, 1e-5, 1-1e-5)
    return val

# 计算我们示例分布之间的KL散度
kl_fg = compute_KL(f, g)
kl_gf = compute_KL(g, f)

print(f"KL(f, g) = {kl_fg:.4f}")
print(f"KL(g, f) = {kl_gf:.4f}")
```

KL散度的不对称性具有重要的实际意义。

$D_{KL}(f\|g)$ 惩罚那些 $f > 0$ 但 $g$ 接近零的区域，反映了使用 $g$ 来建模 $f$ 的代价，反之亦然。

## Jensen-Shannon散度

有时我们需要一个对称的散度度量，用来衡量两个分布之间的差异，而不偏向任何一方。

这种情况经常出现在聚类等应用中，我们想要比较分布，但不假设其中一个是真实模型。

**Jensen-Shannon (JS) 散度**通过将两个分布与它们的混合分布进行比较来使KL散度对称化：

$$
JS(f,g) = \frac{1}{2} D_{KL}(f\|m) + \frac{1}{2} D_{KL}(g\|m), \quad m = \frac{1}{2}(f+g).
$$

其中 $m$ 是对 $f$ 和 $g$ 取平均的混合分布

让我们也可视化混合分布 $m$：

```{code-cell} ipython3
def m(x):
    return 0.5 * (f(x) + g(x))

m_vals = [m(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, f_vals, 'b-', linewidth=2, label=r'$f(x)$')
plt.plot(x_range, g_vals, 'r-', linewidth=2, label=r'$g(x)$')
plt.plot(x_range, m_vals, 'g--', linewidth=2, label=r'$m(x) = \frac{1}{2}(f(x) + g(x))$')

plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

JS散度具有以下几个有用的性质：

- 对称性：$JS(f,g)=JS(g,f)$。
- 有界性：$0 \le JS(f,g) \le \log 2$。
- 其平方根$\sqrt{JS}$在概率分布空间上是一个度量（Jensen-Shannon距离）。
- JS散度等于二元随机变量$Z \sim \text{Bernoulli}(1/2)$（用于指示源）与样本$X$之间的互信息，其中当$Z=0$时$X$从$f$抽样，当$Z=1$时从$g$抽样。

Jensen-Shannon散度在某些生成模型的优化中起着关键作用，因为它是有界的、对称的，且比KL散度更平滑，通常能为训练提供更稳定的梯度。

让我们计算示例分布$f$和$g$之间的JS散度

```{code-cell} ipython3
def compute_JS(f, g):
    """计算Jensen-Shannon散度。"""
    def m(w):
        return 0.5 * (f(w) + g(w))
    js_div = 0.5 * compute_KL(f, m) + 0.5 * compute_KL(g, m)
    return js_div

js_div = compute_JS(f, g)
print(f"Jensen-Shannon散度 JS(f,g) = {js_div:.4f}")
```

我们可以使用带权重 $\alpha = (\alpha_i)_{i=1}^{n}$ 的广义 Jensen-Shannon 散度轻松推广到两个以上的分布:

$$
JS_\alpha(f_1, \ldots, f_n) = 
H\left(\sum_{i=1}^n \alpha_i f_i\right) - \sum_{i=1}^n \alpha_i H(f_i)
$$

其中:
- $\alpha_i \geq 0$ 且 $\sum_{i=1}^n \alpha_i = 1$，以及
- $H(f) = -\int f(x) \log f(x) dx$ 是分布 $f$ 的**香农熵**

## Chernoff熵

Chernoff熵源自[大偏差理论](https://en.wikipedia.org/wiki/Large_deviations_theory)的早期应用，该理论通过提供罕见事件的指数衰减率来改进中心极限近似。

对于密度函数 $f$ 和 $g$，Chernoff熵为

$$
C(f,g) = - \log \min_{\phi \in (0,1)} \int f^{\phi}(x) g^{1-\phi}(x) \, dx.
$$

注释：

- 内部积分是**Chernoff系数**。
- 当 $\phi=1/2$ 时，它变成**Bhattacharyya系数** $\int \sqrt{f g}$。
- 在具有 $T$ 个独立同分布观测的二元假设检验中，最优错误概率以 $e^{-C(f,g) T}$ 的速率衰减。

在{doc}`likelihood_ratio_process`讲座中，我们将在模型选择的背景下研究Chernoff熵时看到第三点的一个例子。

让我们计算示例分布 $f$ 和 $g$ 之间的Chernoff熵。

```{code-cell} ipython3
def chernoff_integrand(ϕ, f, g):
    """计算给定ϕ的Chernoff熵中的积分。"""
    def integrand(w):
        return f(w)**ϕ * g(w)**(1-ϕ)
    result, _ = quad(integrand, 1e-5, 1-1e-5)
    return result

def compute_chernoff_entropy(f, g):
    """计算Chernoff熵 C(f,g)。"""
    def objective(ϕ):
        return chernoff_integrand(ϕ, f, g)
    result = minimize_scalar(objective, bounds=(1e-5, 1-1e-5), method='bounded')
    min_value = result.fun
    ϕ_optimal = result.x
    chernoff_entropy = -np.log(min_value)
    return chernoff_entropy, ϕ_optimal

C_fg, ϕ_optimal = compute_chernoff_entropy(f, g)
print(f"Chernoff熵 C(f,g) = {C_fg:.4f}")
print(f"最优 ϕ = {ϕ_optimal:.4f}")
```

## 比较散度度量

我们现在比较几对Beta分布之间的这些度量

```{code-cell} ipython3
:tags: [hide-input]

distribution_pairs = [
    # (f_params, g_params)
    ((1, 1), (0.1, 0.2)),
    ((1, 1), (0.3, 0.3)),
    ((1, 1), (0.3, 0.4)),
    ((1, 1), (0.5, 0.5)),
    ((1, 1), (0.7, 0.6)),
    ((1, 1), (0.9, 0.8)),
    ((1, 1), (1.1, 1.05)),
    ((1, 1), (1.2, 1.1)),
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),
    ((1, 1), (2.5, 1.8)),
    ((1, 1), (3, 1.2)),
    ((1, 1), (4, 1)),
    ((1, 1), (5, 1))
]

# 创建比较表
results = []
for i, ((f_a, f_b), (g_a, g_b)) in enumerate(distribution_pairs):
    f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
    g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
    kl_fg = compute_KL(f, g)
    kl_gf = compute_KL(g, f)
    js_div = compute_JS(f, g)
    chernoff_ent, _ = compute_chernoff_entropy(f, g)
    results.append({
        'Pair (f, g)': f"\\text{{Beta}}({f_a},{f_b}), \\text{{Beta}}({g_a},{g_b})",
        'KL(f, g)': f"{kl_fg:.4f}",
        'KL(g, f)': f"{kl_gf:.4f}",
        'JS': f"{js_div:.4f}",
        'C': f"{chernoff_ent:.4f}"
    })

df = pd.DataFrame(results)
# 按JS散度排序
df['JS_numeric'] = df['JS'].astype(float)
df = df.sort_values('JS_numeric').drop('JS_numeric', axis=1)

columns = ' & '.join([f'\\text{{{col}}}' for col in df.columns])
rows = ' \\\\\n'.join(
    [' & '.join([f'{val}' for val in row]) 
     for row in df.values])

latex_code = rf"""
\begin{{array}}{{lcccc}}
{columns} \\
\hline
{rows}
\end{{array}}
"""

display(Math(latex_code))
```

当我们改变Beta分布的参数时，我们可以清楚地看到各种散度测度之间的协同变化。

接下来我们可视化KL散度、JS散度和切尔诺夫熵之间的关系。

```{code-cell} ipython3
kl_fg_values = [float(result['KL(f, g)']) for result in results]
js_values = [float(result['JS']) for result in results]
chernoff_values = [float(result['C']) for result in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(kl_fg_values, js_values, alpha=0.7, s=60)
axes[0].set_xlabel('KL散度 KL(f, g)')
axes[0].set_ylabel('JS散度')
axes[0].set_title('JS散度与KL散度的关系')

axes[1].scatter(js_values, chernoff_values, alpha=0.7, s=60)
axes[1].set_xlabel('JS散度')
axes[1].set_ylabel('切尔诺夫熵')
axes[1].set_title('切尔诺夫熵与JS散度的关系')

plt.tight_layout()
plt.show()
```

现在我们生成图表来直观展示重叠如何随着差异度量的增加而减少。


```{code-cell} ipython3
param_grid = [
    ((1, 1), (1, 1)),   
    ((1, 1), (1.5, 1.2)),
    ((1, 1), (2, 1.5)),  
    ((1, 1), (3, 1.2)),  
    ((1, 1), (5, 1)),
    ((1, 1), (0.3, 0.3))
]
```

```{code-cell} ipython3
:tags: [hide-input]

def plot_dist_diff(para_grid):
    """绘制选定Beta分布对的重叠图。"""

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    divergence_data = []
    for i, ((f_a, f_b), (g_a, g_b)) in enumerate(param_grid):
        row, col = divmod(i, 2)
        f = jit(lambda x, a=f_a, b=f_b: p(x, a, b))
        g = jit(lambda x, a=g_a, b=g_b: p(x, a, b))
        kl_fg = compute_KL(f, g)
        js_div = compute_JS(f, g)
        chernoff_ent, _ = compute_chernoff_entropy(f, g)
        divergence_data.append({
            'f_params': (f_a, f_b),
            'g_params': (g_a, g_b),
            'kl_fg': kl_fg,
            'js_div': js_div,
            'chernoff': chernoff_ent
        })
        x_range = np.linspace(0, 1, 200)
        f_vals = [f(x) for x in x_range]
        g_vals = [g(x) for x in x_range]
        axes[row, col].plot(x_range, f_vals, 'b-', 
                        linewidth=2, label=f'f ~ Beta({f_a},{f_b})')
        axes[row, col].plot(x_range, g_vals, 'r-', 
                        linewidth=2, label=f'g ~ Beta({g_a},{g_b})')
        overlap = np.minimum(f_vals, g_vals)
        axes[row, col].fill_between(x_range, 0, 
                        overlap, alpha=0.3, color='purple', label='重叠')
        axes[row, col].set_title(
            f'KL(f,g)={kl_fg:.3f}, JS={js_div:.3f}, C={chernoff_ent:.3f}', 
            fontsize=12)
        axes[row, col].legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    return divergence_data

divergence_data = plot_dist_diff(param_grid)
```

## KL散度和最大似然估计

给定n个观测样本 $X = \{x_1, x_2, \ldots, x_n\}$，**经验分布**为

$$p_e(x) = \frac{1}{n} \sum_{i=1}^n \delta(x - x_i)$$

其中 $\delta(x - x_i)$ 是中心在 $x_i$ 的狄拉克德尔塔函数：

$$
\delta(x - x_i) = \begin{cases}
+\infty & \text{如果 } x = x_i \\
0 & \text{如果 } x \neq x_i
\end{cases}
$$

- **离散概率测度**：对每个观测数据点赋予概率 $\frac{1}{n}$
- **经验期望**：$\langle X \rangle_{p_e} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{\mu}$
- **支撑集**：仅在观测数据点 $\{x_1, x_2, \ldots, x_n\}$ 上

从经验分布 $p_e$ 到参数模型 $p_\theta(x)$ 的KL散度为：

$$D_{KL}(p_e \parallel p_\theta) = \int p_e(x) \log \frac{p_e(x)}{p_\theta(x)} dx$$

利用狄拉克德尔塔函数的数学性质，可得

$$D_{KL}(p_e \parallel p_\theta) = \sum_{i=1}^n \frac{1}{n} \log \frac{\left(\frac{1}{n}\right)}{p_\theta(x_i)}$$

$$= \frac{1}{n} \sum_{i=1}^n \log \frac{1}{n} - \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i)$$

$$= -\log n - \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i)$$

由于参数 $\theta$ 的对数似然函数为：

$$
\ell(\theta; X) = \sum_{i=1}^n \log p_\theta(x_i) ,
$$

因此最大似然估计选择参数以最小化

$$ D_{KL}(p_e \parallel p_\theta) $$

因此，MLE等价于最小化从经验分布到统计模型$p_\theta$的KL散度。

## 相关讲座

本讲座介绍了我们将在其他地方遇到的工具。

- 其他应用散度度量与统计推断之间联系的quantecon讲座包括{doc}`likelihood_ratio_process`、{doc}`wald_friedman`和{doc}`mix_model`。

- 在研究Lawrence Blume和David Easley的异质信念和金融市场模型的{doc}`likelihood_ratio_process_2`中，统计散度函数也占据核心地位。

