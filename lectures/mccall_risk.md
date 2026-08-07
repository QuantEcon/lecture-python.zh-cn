---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 工作搜寻 V：风险敏感型偏好
  headings:
    Overview: 概述
    Outline: 大纲
    Introduction to risk-sensitivity: 风险敏感性简介
    Introduction to risk-sensitivity::A Gaussian example: 一个高斯分布的例子
    Introduction to risk-sensitivity::A more general case: 更一般的情形
    Introduction to risk-sensitivity::A mean preserving spread: 保持均值不变的展宽
    Back to job search: 回到工作搜寻
    Back to job search::Setup: 设置
    Back to job search::Bellman equations: 贝尔曼方程
    Back to job search::How does the reservation wage vary with $\theta$?: 保留工资如何随 $\theta$ 变化？
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 工作搜寻 V：风险敏感型偏好

```{include} _admonition/gpu.md
```

```{contents} Contents
:depth: 2
```


## 概述

风险敏感型偏好是各类动态规划问题中常见的一种扩展。

本讲座通过工作搜寻问题，介绍风险敏感型递归偏好。

下面给出一些动机说明。

## 大纲

在现实世界中与工作相关的决策中，个人和家庭都会关心风险。

例如，一些人可能更愿意接受一个手头已有的适中报价，而不是冒险等待下一期可能出现的更高报价，即使不对未来收益进行贴现也是如此。

（这就是所谓"双鸟在林不如一鸟在手"。）

在本系列前面的工作搜寻讲座中，我们通过加入一个凹的流量效用函数 $u$ 引入了某种程度的风险厌恶。

不幸的是，这种策略并不能单独分离出我们上面所描述的那种风险偏好。

这是因为添加凹效用函数会以其他方式改变主体的偏好，比如改变他们对消费平滑的偏好程度。

因此，如果我们想研究风险的纯粹效应，就需要一种不同的解决方案。

一种可能的方法是引入风险敏感型偏好。

在这里我们将展示如何做到这一点，并研究它对主体选择的影响。

我们将使用 JAX 和 QuantEcon 库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

我们使用以下导入。

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import lax
import quantecon as qe
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
```

## 风险敏感性简介

让我们先从一个静态环境开始讨论。

如果 $Y$ 是一个随机收益，且主体对该收益的评估为 $e := \mathbb{E} Y$，那么我们称该主体是**风险中性**的。

有时我们希望将主体建模为风险厌恶型。

一种方法是将他们对收益 $Y$ 的评估改为

$$
e_{\theta} = \frac{1}{\theta} \ln\left( \mathbb{E} [ \exp(\theta Y) ] \right)
$$

其中 $\theta$ 是一个满足 $\theta < 0$ 的数值。

值 $e_{\theta}$ 有时被称为 $Y$ 的**熵风险调整期望值**。

### 一个高斯分布的例子

理解这种影响的一种方法是假设 $Y$ 服从正态分布 $N(\mu, \sigma^2)$，即其均值为 $\mu$，方差为 $\sigma^2$。

对于这个 $Y$，我们的目标是计算风险调整期望值。

如果我们意识到 $\mathbb{E}[\exp(\theta Y)]$ 正是正态分布的[矩生成函数](https://en.wikipedia.org/wiki/Moment-generating_function)（MGF），那么这就变得非常直接了。

利用正态分布 MGF 的著名表达式，我们得到

$$
\mathbb{E}[\exp(\theta Y)] = \exp\left(\theta\mu + \frac{\theta^2\sigma^2}{2}\right)
$$

因此，

$$
e_\theta 
= \frac{1}{\theta} \ln\left( \exp\left(\theta\mu + \frac{\theta^2\sigma^2}{2}\right) \right) 
= \frac{1}{\theta} \left(\theta\mu + \frac{\theta^2\sigma^2}{2}\right)
$$

化简后得到

$$
e_\theta = \mu + \frac{\theta\sigma^2}{2}
$$

我们立刻可以看到，主体偏好更高的平均收益 $\mu$。

同时，鉴于 $\theta < 0$，风险调整期望值随着 $\sigma$ 的增大而减小。

具体来说，$e_\theta$ 随着风险的增大而减小。

下面是使用等高线图对 $e_\theta$ 作为 $\mu$ 和 $\sigma$ 的函数的可视化展示，其中 $\theta=-1$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Entropic risk-adjusted expectation, Gaussian case
    name: fig-mcr-gaussian
---
θ = -1

μ_vals = np.linspace(-2, 5, 200)
σ_vals = np.linspace(0.1, 3, 200)
μ_grid, σ_grid = np.meshgrid(μ_vals, σ_vals)

e_θ = μ_grid + (θ * σ_grid**2) / 2

# Create contour plot
fig, ax = plt.subplots()
contour = ax.contour(
    μ_grid, σ_grid, e_θ, levels=20, colors='black', linewidths=0.5
)
contourf = ax.contourf(
    μ_grid, σ_grid, e_θ, levels=20, cmap='viridis'
)
ax.clabel(contour, inline=True)
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label(r'$e_\theta$', rotation=0)

ax.set_xlabel(r'$\mu$ (mean)')
ax.set_ylabel(r'$\sigma$ (standard deviation)')
plt.tight_layout()
plt.show()
```

同样，我们看到主体偏好更高的平均收益，但不喜欢风险。

### 更一般的情形

前面的分析依赖于高斯（正态）假设来获得解析解。

我们可以使用模拟方法来研究所有其他情形。

例如，假设 $Y$ 服从 Beta$(a, b)$ 分布。

这里我们设 $a=b=2.0$，并使用蒙特卡洛方法计算 $e_{\theta}$。

方法如下：

1. 从 Beta$(2,2)$ 中抽取样本 $Y_1, \ldots, Y_n$
2. 用 $\exp(\theta Y_i)$ 的平均值代替 $\mathbb{E}$

我们对 $-2$ 到 $-0.1$ 之间的 100 个网格点的 $\theta$ 值都进行这样的计算。

下面是 $e_{\theta}$ 关于 $\theta$ 的绘图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Risk-adjusted evaluation against risk aversion
    name: fig-mcr-beta
---
# Set parameters
a, b = 2.0, 2.0
mc_size = 1_000_000  # Large number of Monte Carlo samples
θ_grid = jnp.linspace(-2, -0.1, 100)

# Draw samples from Beta(2, 2) distribution using JAX
key = jax.random.key(1234)
Y_samples = jax.random.beta(key, a, b, shape=(mc_size,))

# Define function to compute e_θ for a single θ value
def compute_e_θ(θ):
    """Compute e_θ = (1/θ) * ln(E[exp(θ * Y)])"""
    expectation = jnp.mean(jnp.exp(θ * Y_samples))
    return (1 / θ) * jnp.log(expectation)

# Vectorize over θ_grid using vmap
compute_e_θ_vec = jax.vmap(compute_e_θ)
e_θ_values = compute_e_θ_vec(θ_grid)

# Plot results
fig, ax = plt.subplots()
ax.plot(θ_grid, e_θ_values, lw=2)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$e_\theta$')
ax.axhline(y=0.5, color='black', linestyle='--',
           linewidth=1, label=r'$\mathbb{E}[Y] = 0.5$')
ax.legend()
plt.tight_layout()
plt.show()
```

该图展示了风险调整评估值 $e_\theta$ 如何随风险厌恶参数 $\theta$ 的变化而变化。

当 $\theta \to 0$ 时，$e_\theta$ 趋近于 $Y$ 的期望值，对于 Beta(2,2) 而言即 $\mathbb{E}[Y] = \frac{a}{a+b} = \frac{2}{4} = 0.5$。

这是合理的，因为当 $\theta \to 0$ 时，主体变为风险中性。

随着 $\theta$ 变得更负，$e_\theta$ 会减小。

这反映出更加风险厌恶的主体对不确定收益 $Y$ 的估值低于其期望值。

### 保持均值不变的展宽

下一个练习要求你研究保持均值不变的展宽（mean-preserving spread）对风险调整期望值的影响。

```{exercise}
:label: mcr_ex0

保持 $Y \sim \text{Beta}(2, 2)$ 不变，固定 $\theta = -2$。

再次使用蒙特卡洛方法，计算

$$
e_{\theta} = \frac{1}{\theta} \ln\left( \mathbb{E} [ \exp(\theta X) ] \right)
$$

其中 $X = Y + \sigma Z$，$Z$ 服从标准正态分布。

$e_\theta$ 如何随 $\sigma$ 变化？

你能否给出一些直观解释来说明所发生的现象（考虑到主体是风险厌恶的）？

请用图像来说明你的结果。
```

```{solution-start} mcr_ex0
:class: dropdown
```

以下是我们的解答。

```{code-cell} ipython3
a, b = 2.0, 2.0
θ = -2.0
mc_size = 1_000_000  # Large number of Monte Carlo samples
σ_grid = jnp.linspace(0.0, 1.0, 50)

# Set random seed for reproducibility
key = jax.random.key(1234)
key_y, key_z = jax.random.split(key)

# Draw samples from Beta(2, 2) distribution
Y_samples = jax.random.beta(key_y, a, b, shape=(mc_size,))

# Draw standard normal samples (reused for all σ values)
Z_samples = jax.random.normal(key_z, shape=(mc_size,))

# Define function to compute e_θ for a single σ value
def compute_e_θ(σ):
    """Compute e_θ for X = Y + σ * Z"""
    # Compute X = Y + σ * Z
    X_samples = Y_samples + σ * Z_samples

    # Calculate E[exp(θ * X)] using Monte Carlo
    expectation = jnp.mean(jnp.exp(θ * X_samples))

    # Calculate e_θ
    return (1 / θ) * jnp.log(expectation)

# Vectorize over σ_grid using vmap
compute_e_θ_vec = jax.vmap(compute_e_θ)
e_θ_values = compute_e_θ_vec(σ_grid)

# Plot results
fig, ax = plt.subplots()
ax.plot(σ_grid, e_θ_values, lw=2, label=r'$e_\theta(\sigma)$')
ax.set_xlabel(r'$\sigma$ (noise level)')
ax.set_ylabel(r'$e_\theta$')
ax.set_title(r'Risk-adjusted evaluation as noise increases')
ax.axhline(y=e_θ_values[0], color='black', linestyle='--', linewidth=1,
           label=f'No noise: $e_\\theta$ = {e_θ_values[0]:.3f}')
ax.legend()
plt.tight_layout()
plt.show()
```

该图清楚地显示，随着 $\sigma$ 的增大，$e_\theta$ 单调递减。

由于主体是风险厌恶的（$\theta = -2 < 0$），她不喜欢不确定性。

当我们增大 $\sigma$ 时，波动性变大，因为

$$
\text{Var}(X) = \text{Var}(Y) + \sigma^2 \text{Var}(Z) = \text{Var}(Y) + \sigma^2
$$

与此同时，期望值保持不变，因为

$$
\mathbb{E}[X] = \mathbb{E}[Y + \sigma Z] = \mathbb{E}[Y] + \sigma \mathbb{E}[Z] = \mathbb{E}[Y] = 0.5
$$

因此平均收益不随 $\sigma$ 变化。

换句话说，风险厌恶的主体并没有因承担额外风险而获得补偿。

这就是为什么该随机收益的估值会下降。

```{solution-end}
```

## 回到工作搜寻

在 {doc}`mccall_fitted_vfi` 讲座中，我们研究了带有工作分离、马尔可夫工资抽取以及拟合值函数迭代的工作搜寻模型。

工资报价过程是连续的，并满足

$$
    W_t = \exp(X_t)
    \quad \text{where} \quad
    X_{t+1} = \rho X_t + \nu Z_{t+1}
$$

其中 $\{Z_t\}$ 是独立同分布的标准正态随机变量。

现在让我们研究相同的模型，但用风险厌恶型期望取代风险中性期望的假设。

具体来说，该讲座中的条件期望

$$
    (P v_u)(w) = \mathbb{E} v_u( w^\rho  \exp(\nu Z) )
$$

被替换为

$$
(P_\theta v_u)(w)
= \frac{1}{\theta} \ln
    \left[
        \mathbb{E} \exp(\theta v_u( w^\rho  \exp(\nu Z) ))
    \right]
$$

除此之外，模型的其余部分保持不变。

我们现在求解该动态规划问题，并研究 $\theta$ 对保留工资的影响。

### 设置

以下是一个用于存储参数和默认参数值的类。

```{code-cell} ipython3

class Model(NamedTuple):
    c: float              # unemployment compensation
    α: float              # job separation rate
    β: float              # discount factor
    ρ: float              # wage persistence
    ν: float              # wage volatility
    θ: float              # risk aversion parameter
    w_grid: jnp.ndarray   # grid of points for fitted VFI
    z_draws: jnp.ndarray  # draws from the standard normal distribution

def create_mccall_model(
        c: float = 1.0,
        α: float = 0.1,
        β: float = 0.96,
        ρ: float = 0.9,
        ν: float = 0.2,
        θ: float = -1.5,
        grid_size: int = 100,
        mc_size: int = 1000,
        seed: int = 1234
    ):
    """Factory function to create a McCall model instance."""

    key = jax.random.key(seed)
    z_draws = jax.random.normal(key, (mc_size,))

    # Discretize just to get a suitable wage grid for interpolation
    mc = qe.markov.tauchen(grid_size, ρ, ν)
    w_grid = jnp.exp(jnp.array(mc.state_values))

    return Model(c, α, β, ρ, ν, θ, w_grid, z_draws)
```

### 贝尔曼方程

我们的构造是对 {doc}`mccall_fitted_vfi` 中贝尔曼方程的直接扩展。

首先，我们利用已就业工人的贝尔曼方程，用 $(P_\theta v_u)(w)$ 表示 $v_e(w)$：

$$
v_e(w) = 
\frac{1}{1-\beta(1-\alpha)} \cdot (u(w) + \alpha\beta(P_\theta v_u)(w))
$$

我们将其代入失业主体的贝尔曼方程，得到：

$$
v_u(w) = 
\max
\left\{
    \frac{1}{1-\beta(1-\alpha)} \cdot (u(w) + \alpha\beta(P_\theta v_u)(w)),
    u(c) + \beta(P_\theta v_u)(w)
\right\}
$$

我们使用值函数迭代来求解 $v_u$。

然后我们计算最优策略：如果 $v_e(w) ≥ u(c) + β(P_\theta v_u)(w)$ 则接受。

以下是更新 $v_u$ 的贝尔曼算子。

```{code-cell} ipython3
def T(model, v):
    # Unpack model parameters
    c, α, β, ρ, ν, θ, w_grid, z_draws = model

    # Interpolate array represented value function
    vf = lambda x: jnp.interp(x, w_grid, v)

    def compute_expectation(w):
        # Use Monte Carlo to evaluate integral (P_θ v)(w)
        inner = jnp.mean(jnp.exp(θ * vf(w**ρ * jnp.exp(ν * z_draws))))
        return (1 / θ) * jnp.log(inner)

    compute_exp_all = jax.vmap(compute_expectation)
    P_θ_v = compute_exp_all(w_grid)

    d = 1 / (1 - β * (1 - α))
    accept = d * (w_grid + α * β * P_θ_v)
    reject = c + β * P_θ_v
    return jnp.maximum(accept, reject)
```

以下是求解器：

```{code-cell} ipython3
@jax.jit
def vfi(
        model: Model,
        tolerance: float = 1e-6,   # Error tolerance
        max_iter: int = 100_000,   # Max iteration bound
    ):

    v_init = jnp.zeros(model.w_grid.shape)

    def cond(loop_state):
        v, error, i = loop_state
        return (error > tolerance) & (i <= max_iter)

    def update(loop_state):
        v, error, i = loop_state
        v_new = T(model, v)
        error = jnp.max(jnp.abs(v_new - v))
        new_loop_state = v_new, error, i + 1
        return new_loop_state

    initial_state = (v_init, tolerance + 1, 1)
    final_loop_state = lax.while_loop(cond, update, initial_state)
    v_final, error, i = final_loop_state

    return v_final
```

下面的函数在假设 $v$ 是值函数的情况下计算最优策略：

```{code-cell} ipython3
def get_greedy(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    """Get a v-greedy policy."""
    c, α, β, ρ, ν, θ, w_grid, z_draws = model

    # Interpolate array represented value function
    vf = lambda x: jnp.interp(x, w_grid, v)

    def compute_expectation(w):
        # Use Monte Carlo to evaluate integral (P_θ v)(w)
        inner = jnp.mean(jnp.exp(θ * vf(w**ρ * jnp.exp(ν * z_draws))))
        return (1 / θ) * jnp.log(inner)

    compute_exp_all = jax.vmap(compute_expectation)
    P_θ_v = compute_exp_all(w_grid)

    d = 1 / (1 - β * (1 - α))
    accept = d * (w_grid + α * β * P_θ_v)
    reject = c + β * P_θ_v
    σ = accept >= reject
    return σ
```

以下函数接收一个 `Model` 实例，并返回相关的保留工资。

```{code-cell} ipython3
@jax.jit
def get_reservation_wage(σ: jnp.ndarray, model: Model) -> float:
    """
    Calculate the reservation wage from a given policy.

    Parameters:
    - σ: Policy array where σ[i] = True means accept wage w_grid[i]
    - model: Model instance containing wage values

    Returns:
    - Reservation wage (lowest wage for which policy indicates acceptance)
    """
    c, α, β, ρ, ν, θ, w_grid, z_draws = model

    # Find the first index where policy indicates acceptance
    # σ is a boolean array, argmax returns the first True value
    first_accept_idx = jnp.argmax(σ)

    # If no acceptance (all False), return infinity
    # Otherwise return the wage at the first acceptance index
    return jnp.where(jnp.any(σ), w_grid[first_accept_idx], jnp.inf)
```

让我们在默认参数下求解该模型：

```{code-cell} ipython3
# First, let's solve for the default θ = -1.5
model = create_mccall_model()
c, α, β, ρ, ν, θ, w_grid, z_draws = model

print(f"Solving model with θ = {θ}")
v_star = vfi(model)
σ_star = get_greedy(v_star, model)
w_bar = get_reservation_wage(σ_star, model)
print(f"Reservation wage at default parameters: {w_bar:.4f}")
```

### 保留工资如何随 $\theta$ 变化？

现在让我们考察保留工资如何随风险厌恶参数的变化而变化。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Reservation wage against risk aversion
    name: fig-mcr-reservation
---
# Create a grid of θ values (all negative for risk aversion)
θ_grid = jnp.linspace(-3.0, -0.1, 25)

# Define function to compute reservation wage for a single θ value
def compute_res_wage_for_theta(θ):
    """Compute reservation wage for a given θ value"""
    model = create_mccall_model(θ=θ)
    v = vfi(model)
    σ = get_greedy(v, model)
    w_bar = get_reservation_wage(σ, model)
    return w_bar

# Vectorize over θ_grid using vmap
compute_res_wages_vec = jax.vmap(compute_res_wage_for_theta)
reservation_wages = compute_res_wages_vec(θ_grid)

# Plot the results
fig, ax = plt.subplots()
ax.plot(θ_grid, reservation_wages,
        lw=2, marker='o', markersize=4)
ax.set_xlabel(r'$\theta$ (risk aversion parameter)')
ax.set_ylabel('Reservation wage')
ax.axvline(x=-1.5, color='black', ls='--',
           linewidth=1, label=r'Default $\theta = -1.5$')
ax.legend()
plt.tight_layout()
plt.show()
```

保留工资随着 $\theta$ 变得不那么负（趋向于零）而增大。

等价地说，当主体变得更加风险厌恶（$\theta$ 更负）时，保留工资会下降。

原因在于，一个更加风险厌恶的主体更加重视就业所带来的确定收入，相对于继续搜寻所带来的不确定未来前景。

因此，他们愿意接受更低的工资以摆脱失业状态。

```{exercise}
:label: mcr_ex1

利用模拟研究长期失业率如何随 $\theta$ 变化。

使用上一节中的参数，即我们研究保留工资如何随 $\theta$ 变化时所用的参数。

你可以使用来自 {doc}`mccall_fitted_vfi` 的模拟代码，并进行适当修改。
```

```{solution-start} mcr_ex1
:class: dropdown
```

为了计算长期失业率，我们首先编写一个用于更新单个主体状态的函数。

```{code-cell} ipython3
@jax.jit
def simulate_single_agent(key, model, w_star, num_periods=200):
    """
    Simulate a single agent for num_periods periods.
    Returns final employment status (1 if employed, 0 if unemployed).
    """
    c, α, β, ρ, ν, θ, w_grid, z_draws = model

    # Start from arbitrary initial conditions
    w = 1.0
    status = 1

    def update(t, loop_state):
        w, status, key = loop_state
        key, k1, k2 = jax.random.split(key, 3)

        # Update wage
        z = jax.random.normal(k2)
        w_new = w**ρ * jnp.exp(ν * z)

        # Employment transitions
        sep_draw = jax.random.uniform(k1)
        becomes_unemployed = sep_draw < α

        # Check if unemployed worker accepts wage
        accepts_job = w >= w_star

        # Update employment status
        new_status = jnp.where(
            status,
            1 - becomes_unemployed,   # employed path
            accepts_job               # unemployed path
        )

        new_wage = jnp.where(
            status,
            jnp.where(becomes_unemployed, w_new, w),  # employed path
            jnp.where(accepts_job, w, w_new)          # unemployed path
        )

        return (new_wage, new_status, key)

    init_state = (w, status, key)
    final_state = lax.fori_loop(0, num_periods, update, init_state)
    _, final_status, _ = final_state
    return final_status


def compute_unemployment_rate(model, w_star, num_agents=1000, num_periods=200, seed=12345):
    """
    Compute unemployment rate via cross-sectional simulation.

    Instead of simulating one agent for a long time series, we simulate
    many agents in parallel for a shorter time period. This is much more
    efficient with JAX parallelization.

    The steady state satisfies:
    - Employed workers lose jobs at rate α
    - Unemployed workers find acceptable jobs at rate (1 - F(w*))

    We simulate num_agents agents for num_periods each, then compute
    the fraction unemployed at the end.
    """
    # Create keys for each agent
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_agents)

    # Vectorize simulation across agents (parallelization!)
    simulate_agents = jax.vmap(
        lambda k: simulate_single_agent(k, model, w_star, num_periods)
    )

    # Run all agents in parallel
    status_cross_section = simulate_agents(keys)

    # Unemployment rate is 1 - mean(status) since status=1 means employed
    unemployment_rate = 1 - jnp.mean(status_cross_section)

    return unemployment_rate


# Define function to compute unemployment rate for a single θ value
def compute_u_rate_for_theta(θ):
    """Compute unemployment rate for a given θ value"""
    model = create_mccall_model(θ=θ)
    v = vfi(model)
    σ = get_greedy(v, model)
    w_star = get_reservation_wage(σ, model)
    u_rate = compute_unemployment_rate(
        model, w_star, num_agents=5000, num_periods=200
    )
    return u_rate

# Vectorize over θ_grid using vmap
compute_u_rates_vec = jax.vmap(compute_u_rate_for_theta)
unemployment_rates = compute_u_rates_vec(θ_grid)

# Plot the results
fig, ax = plt.subplots()
ax.plot(θ_grid, unemployment_rates * 100,
        lw=2, marker='s', markersize=4)
ax.set_xlabel(r'$\theta$ (risk aversion parameter)')
ax.set_ylabel('Long-run unemployment rate (%)')
ax.set_title('Unemployment rate as a function of risk aversion')
ax.axvline(x=-1.5, color='black', ls='--', linewidth=1, label=r'Default $\theta = -1.5$')
ax.legend()
plt.tight_layout()
plt.show()
```

我们看到，随着主体变得更加风险厌恶（$\theta$ 更负），失业率会下降。

这是因为更加风险厌恶的工人拥有更低的保留工资，因此他们会接受更广泛范围内的工作报价。

结果，他们花在寻找更好机会上的失业时间更少。

```{solution-end}
```