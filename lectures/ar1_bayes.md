---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# AR(1)参数的后验分布

我们先从导入一些Python包开始。

```{code-cell} ipython3
:tags: [hide-output]

!pip install arviz pymc numpyro jax
```

```{code-cell} ipython3
import arviz as az
import pymc as pmc
import numpyro
from numpyro import distributions as dist

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']


import logging
logging.basicConfig()
logger = logging.getLogger('pymc')
logger.setLevel(logging.CRITICAL)
```

本讲座使用[pymc](https://www.pymc.io/projects/docs/en/stable/)和[numpyro](https://num.pyro.ai/en/stable/)提供的贝叶斯方法对一阶自回归模型的两个参数进行统计推断。

该模型为我们提供了一个很好的机会来研究不同的初始值$y_0$的分布对结果的影响。

我们研究两种不同的初始值$y_0$的分布：

- $y_0$ 作为一个固定数值

- $y_0$ 作为从$\{y_t\}$随机过程的平稳分布中抽取的随机变量

这个模型的第一个组成部分是

$$
y_{t+1} = \rho y_t + \sigma_x \epsilon_{t+1}, \quad t \geq 0
$$ (eq:themodel)

其中标量$\rho$和$\sigma_x$满足$|\rho| < 1$和$\sigma_x > 0$；
$\{\epsilon_{t+1}\}$是一个均值为$0$、方差为$1$的独立同分布正态随机变量序列。

统计模型的第二个组成部分是

$$
y_0 \sim {\cal N}(\mu_0, \sigma_0^2)
$$ (eq:themodel_2)

对于由该统计模型生成的样本序列$\{y_t\}_{t=0}^T$，
我们可以将其似然函数**分解**成以下形式：

$$
f(y_T, y_{T-1}, \ldots, y_0) = f(y_T| y_{T-1}) f(y_{T-1}| y_{T-2}) \cdots f(y_1 | y_0 ) f(y_0)
$$

这里我们用 $f$ 表示一般的概率密度。

统计模型 {eq}`eq:themodel`-{eq}`eq:themodel_2` 表明

$$
\begin{aligned}
f(y_t | y_{t-1})  & \sim {\mathcal N}(\rho y_{t-1}, \sigma_x^2) \\
        f(y_0)  & \sim {\mathcal N}(\mu_0, \sigma_0^2)
\end{aligned}
$$

我们想探究未知参数 $(\rho, \sigma_x)$ 的推断如何依赖于对 $y_0$ 分布的参数 $\mu_0, \sigma_0$。

下面，我们研究两种广泛使用的替代假设：

- 第一种情况，我们将 $y_0$ 视为已知的固定值，即 $(\mu_0,\sigma_0) = (y_0, 0)$。这相当于对观察到的初始值进行条件化。

- 第二种情况，我们假设 $y_0$ 来自模型的平稳分布，此时 $\mu_0$ 和 $\sigma_0$ 由参数 $\rho$ 和 $\sigma_x$ 决定。

**注意：** 我们**不**考虑将 $\mu_0$ 和 $\sigma_0$ 作为待估计参数的情况。

未知参数是 $\rho, \sigma_x$。

我们有 $\rho, \sigma_x$ 的独立**先验概率分布**，并希望在观察到样本 $\{y_{t}\}_{t=0}^T$ 后计算后验概率分布。

本讲使用 `pymc4` 和 `numpyro` 来计算 $\rho, \sigma_x$ 的后验分布。

我们将使用 NUTS 采样器在链中生成后验分布的样本。

NUTS是一种蒙特卡洛马尔可夫链（MCMC）算法，它避免了随机游走行为，能更快地收敛到目标分布。

这不仅具有速度上的优势，还允许在不掌握拟合方法的理论知识的情况下，拟合复杂模型。

让我们来探讨对$y_0$分布做出不同假设的影响：

- 第一种方法是直接使用观察到的$y_0$值作为条件。这相当于将$y_0$视为一个确定的值，而不是一个随机变量。

- 第二种方法假设$y_0$是从{eq}`eq:themodel`所描述过程的平稳分布中抽取的，
因此$y_0 \sim {\cal N} \left(0, {\sigma_x^2\over (1-\rho)^2} \right)$。

当初始值$y_0$位于平稳分布尾部较远处时，对初始值进行条件化会得到一个**更准确的**后验分布，我们将对此进行解释。

基本上，当$y_0$恰好位于平稳分布的尾部，而我们**不对$y_0$进行条件化**时，$\{y_t\}_{t=0}^T$的似然函数会调整后验分布的参数$\rho, \sigma_x$，使得观测到的$y_0$值在平稳分布下比实际情况更可能出现，从而在短样本中对后验分布产生扭曲。

下面的例子展示了不对$y_0$进行条件化是如何导致$\rho$的后验概率分布向更大的值偏移的。

为了展示这一点，我们先通过模拟生成一个AR(1)过程的样本数据。

选择初始值$y_0$的方式很重要。

* 如果我们认为 $y_0$ 是从平稳分布 ${\mathcal N}(0, \frac{\sigma_x^{2}}{1-\rho^2})$ 中抽取的，那么使用这个分布作为 $f(y_0)$ 是个好主意。为什么？因为 $y_0$ 包含了关于 $\rho, \sigma_x$ 的信息。

* 如果我们怀疑 $y_0$ 位于平稳分布的尾部很远的位置——以至于样本中早期观测值的变化具有显著的**瞬态成分**——最好通过设置 $f(y_0) = 1$ 来对 $y_0$ 进行条件化。

为了说明这个问题，我们将从选择一个位于平稳分布尾部很远的初始值 $y_0$ 开始。

```{code-cell} ipython3

def ar1_simulate(rho, sigma, y0, T):

    # Allocate space and draw epsilons
    y = np.empty(T)
    eps = np.random.normal(0.,sigma,T)

    # Initial condition and step forward
    y[0] = y0
    for t in range(1, T):
        y[t] = rho*y[t-1] + eps[t]

    return y

sigma =  1.
rho = 0.5
T = 50

np.random.seed(145353452)
y = ar1_simulate(rho, sigma, 10, T)
```
```{code-cell} ipython3
plt.plot(y)
plt.tight_layout()
```

现在我们将使用贝叶斯定理来构建后验分布，以初始值$y_0$为条件。

(稍后我们会假设$y_0$是从平稳分布中抽取的，但现在不作此假设。)

首先我们将使用**pymc4**。

## PyMC实现

对于`pymc`中的正态分布，
$var = 1/\tau = \sigma^{2}$。

```{code-cell} ipython3

AR1_model = pmc.Model()

with AR1_model:

    # 首先设定先验分布
    rho = pmc.Uniform('rho', lower=-1., upper=1.) # 假设rho是稳定的
    sigma = pmc.HalfNormal('sigma', sigma = np.sqrt(10))

    # 下一期y的期望值(rho * y)
    yhat = rho * y[:-1]

    # 实际值的似然函数
    y_like = pmc.Normal('y_obs', mu=yhat, sigma=sigma, observed=y[1:])
```

[pmc.sample](https://www.pymc.io/projects/docs/en/v5.10.0/api/generated/pymc.sample.html#pymc-sample) 默认使用NUTS采样器来生成样本，如下面的代码单元所示：

```{code-cell} ipython3
:tag: [hide-output]

with AR1_model:
    trace = pmc.sample(50000, tune=10000, return_inferencedata=True)
```

```{code-cell} ipython3
with AR1_model:
    az.plot_trace(trace, figsize=(17,6))
```

显然，后验分布并没有以我们用来生成数据的真实值 $\rho = .5, \sigma_x = 1$ 为中心。

这是一阶自回归过程中经典的**赫维奇偏差**（Hurwicz bias）的表现（参见 Leonid Hurwicz {cite}`hurwicz1950least`）。

赫维奇偏差在样本量越小时表现得越明显（参见 {cite}`Orcutt_Winokur_69`）。

不管怎样，这里是关于后验分布的更多信息。

```{code-cell} ipython3
with AR1_model:
    summary = az.summary(trace, round_to=4)

summary
```

现在让我们计算另一种情况下的后验分布：假设初始观测值 $y_0$ 是从平稳分布中抽取的，而不是将其视为固定值。

这意味着

$$
y_0 \sim N \left(0, \frac{\sigma_x^{2}}{1 - \rho^{2}} \right)
$$

我们按如下方式修改代码：

```{code-cell} ipython3
AR1_model_y0 = pmc.Model()

with AR1_model_y0:

    # 首先设定先验分布
    rho = pmc.Uniform('rho', lower=-1., upper=1.) # 假设 rho 是稳定的
    sigma = pmc.HalfNormal('sigma', sigma=np.sqrt(10))

    # 平稳 y 的标准差
    y_sd = sigma / np.sqrt(1 - rho**2)

    # yhat
    yhat = rho * y[:-1]
    y_data = pmc.Normal('y_obs', mu=yhat, sigma=sigma, observed=y[1:])
    y0_data = pmc.Normal('y0_obs', mu=0., sigma=y_sd, observed=y[0])
```

```{code-cell} ipython3
:tag: [hide-output]

with AR1_model_y0:
    trace_y0 = pmc.sample(50000, tune=10000, return_inferencedata=True)

# 灰色垂直线表示发散的情况
```
```{code-cell} ipython3
with AR1_model_y0:
    az.plot_trace(trace_y0, figsize=(17,6))
```
```{code-cell} ipython3
with AR1_model:
    summary_y0 = az.summary(trace_y0, round_to=4)

summary_y0
```

请注意当我们基于$y_0$进行条件化而不是假设$y_0$来自平稳分布时，$\rho$的后验分布相对向右偏移。

思考一下为什么会发生这种情况。

```{hint}
这与贝叶斯定律如何解决**逆问题**有关 - 它通过给那些能更好地解释观测数据的参数值分配更高的概率来实现这一点。
```

在我们使用`numpyro`来计算这两种关于$y_0$分布的假设下的后验分布之前,我们会回到这个问题。

我们现在用`numpyro`重复这些计算。

## Numpyro实现

```{code-cell} ipython3


def plot_posterior(sample):
    """
    绘制轨迹和直方图
    """
    # 转换为np数组
    rhos = sample['rho']
    sigmas = sample['sigma']
    rhos, sigmas, = np.array(rhos), np.array(sigmas)

    fig, axs = plt.subplots(2, 2, figsize=(17, 6))
    # 绘制轨迹
    axs[0, 0].plot(rhos)   # rho
    axs[1, 0].plot(sigmas) # sigma

    # 绘制后验分布
    axs[0, 1].hist(rhos, bins=50, density=True, alpha=0.7)
    axs[0, 1].set_xlim([0, 1])
    axs[1, 1].hist(sigmas, bins=50, density=True, alpha=0.7)

    axs[0, 0].set_title("rho")
    axs[0, 1].set_title("rho")
    axs[1, 0].set_title("sigma")
    axs[1, 1].set_title("sigma")
    plt.show()
```

```{code-cell} ipython3
def AR1_model(data):
    # 设置先验分布
    rho = numpyro.sample('rho', dist.Uniform(low=-1., high=1.))
    sigma = numpyro.sample('sigma', dist.HalfNormal(scale=np.sqrt(10)))

    # 下一期y的期望值 (rho * y)
    yhat = rho * data[:-1]

    # 实际值的似然函数
    y_data = numpyro.sample('y_obs', dist.Normal(loc=yhat, scale=sigma), obs=data[1:])
```

```{code-cell} ipython3
:tag: [hide-output]

# 创建 jnp 数组
y = jnp.array(y)

# 设置 NUTS 核心
NUTS_kernel = numpyro.infer.NUTS(AR1_model)

# 运行 MCMC
mcmc = numpyro.infer.MCMC(NUTS_kernel, num_samples=50000, num_warmup=10000, progress_bar=False)
mcmc.run(rng_key=random.PRNGKey(1), data=y)
```

```{code-cell} ipython3
plot_posterior(mcmc.get_samples())
```

```{code-cell} ipython3
mcmc.print_summary()
```

接下来，我们再次计算后验分布，这次假设 $y_0$ 是从平稳分布中抽取的，因此

$$
y_0 \sim N \left(0, \frac{\sigma_x^{2}}{1 - \rho^{2}} \right)
$$

以下是实现这一目的的新代码。

```{code-cell} ipython3
def AR1_model_y0(data):
    # 设置先验分布
    rho = numpyro.sample('rho', dist.Uniform(low=-1., high=1.))
    sigma = numpyro.sample('sigma', dist.HalfNormal(scale=np.sqrt(10)))

    # 平稳y的标准差
    y_sd = sigma / jnp.sqrt(1 - rho**2)

    # 下一期y的期望值(rho * y)
    yhat = rho * data[:-1]

    # 实际实现值的似然
    y_data = numpyro.sample('y_obs', dist.Normal(loc=yhat, scale=sigma), obs=data[1:])
    y0_data = numpyro.sample('y0_obs', dist.Normal(loc=0., scale=y_sd), obs=data[0])
```
```{code-cell} ipython3
:tag: [hide-output]

# 创建jnp数组
y = jnp.array(y)

# 设置NUTS核心
NUTS_kernel = numpyro.infer.NUTS(AR1_model_y0)

# 运行MCMC
mcmc2 = numpyro.infer.MCMC(NUTS_kernel, num_samples=50000, num_warmup=10000, progress_bar=False)
mcmc2.run(rng_key=random.PRNGKey(1), data=y)
```
```{code-cell} ipython3
plot_posterior(mcmc2.get_samples())
```

```{code-cell} ipython3
mcmc2.print_summary()
```

看看后验分布发生了什么！

贝叶斯推断试图通过调整参数来解释这个"异常"的初始观测值。这导致后验分布偏离了生成数据时使用的真实参数值。

贝叶斯定律通过驱使$\rho \rightarrow 1$和$\sigma \uparrow$来提高平稳分布的方差，从而能够为第一个观测值生成一个合理的似然。

这个例子很好地说明了在贝叶斯推断中，我们对初始条件分布的假设会对最终的推断结果产生重要影响。
