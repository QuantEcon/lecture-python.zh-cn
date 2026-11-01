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
translation:
  title: 似然比过程的均值
  headings:
    Overview: 概述
    Mathematical expectation of likelihood ratio: 似然比的数学期望
    Importance sampling: 重要性采样
    Approximating a cumulative likelihood ratio: 近似累积似然比
    Distribution of sample mean: 样本均值的分布
    Choosing a sampling distribution: 选择抽样分布
    Exercises: 练习
---

# 似然比过程的均值

```{include} _admonition/gpu.md
```

```{contents} 目录
:depth: 2
```

## 概述

在 {doc}`likelihood_ratio_process` 中，我们描述了似然比过程的一个特殊性质，即尽管它几乎必然收敛于零，但对于所有 $t \geq 0$，其均值都等于1。

虽然在理论上（即在总体中）很容易验证这个特殊性质，但要通过计算机模拟来验证它却很具有挑战性，因为这需要应用大数定律来研究重复模拟的样本平均值。

为了应对这个挑战，本讲座使用__重要性抽样__来加速样本平均值向总体均值的收敛。

我们使用重要性抽样来估计累积似然比 $L\left(\omega^t\right) = \prod_{i=1}^t \ell \left(\omega_i\right)$ 的均值。

除了Anaconda中包含的库外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install jax
```

首先导入一些Python包。

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
import matplotlib as mpl
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from jax.scipy.stats import beta
from typing import NamedTuple
from functools import partial

# 设置JAX使用64位浮点数
jax.config.update("jax_enable_x64", True)
```

## 似然比的数学期望

在 {doc}`likelihood_ratio_process` 中，我们研究了似然比 $\ell \left(\omega_t\right)$

$$
\ell \left( \omega_t \right) = \frac{f\left(\omega_t\right)}{g\left(\omega_t\right)}
$$

其中 $f$ 是 Beta$(F_a, F_b) = $ Beta$(1, 1)$ 密度，即 $[0,1]$ 上的均匀密度，而 $g$ 是 Beta$(G_a, G_b) = $ Beta$(3, 1.2)$ 密度。

假设独立同分布的随机变量 $\omega_t \in \Omega$ 由 $g$ 生成。

**累积似然比** $L \left(\omega^t\right)$ 为

$$
L\left(\omega^t\right) = \prod_{i=1}^t \ell \left(\omega_i\right)
$$

我们的目标是很好地近似数学期望 $E \left[ L\left(\omega^t\right) \right]$。

在 {doc}`likelihood_ratio_process` 的 {ref}`这一节 <nature_likeli>` 中，我们证明了对所有 $t$，$E \left[ L\left(\omega^t\right) \right]$ 等于1。

我们想要检验当用模拟的样本均值替代 $E$ 时，这个结论的准确程度如何。

事实证明这比说起来要难，因为对于上述假设的Beta分布，当 $t \rightarrow \infty$ 时，$L\left(\omega^t\right)$ 具有非常偏斜的分布，且有很长的尾部。

这个特性使得用标准蒙特卡洛模拟方法来有效且准确地估计均值变得困难。

在本课中，我们将探讨标准蒙特卡洛方法为何会失效。

我们还将展示**重要性抽样**如何提供一种计算上更有效的方法来近似累积似然比的均值。

我们首先来看一下密度函数 `f` 和 `g`。

```{code-cell} ipython3
# 模型参数
class ImpSampleParams(NamedTuple):
    F_a: float = 1.0    # f 的贝塔分布参数
    F_b: float = 1.0
    G_a: float = 3.0    # g 的贝塔分布参数
    G_b: float = 1.2

params = ImpSampleParams()

def f(w):
    return beta.pdf(w, params.F_a, params.F_b)

def g(w):
    return beta.pdf(w, params.G_a, params.G_b)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '贝塔密度函数 $f$ 和 $g$'
    name: fig-imp-densities
---
w_range = np.linspace(1e-2, 1-1e-5, 1000)

plt.plot(w_range, g(w_range), lw=2, label='g')
plt.plot(w_range, f(w_range), lw=2, label='f')
plt.xlabel(r'$\omega$')
plt.legend()
plt.show()
```

似然比是 `l(w)=f(w)/g(w)`。

```{code-cell} ipython3
def l(w):
    return f(w) / g(w)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '似然比 $\ell(\omega)$，对数坐标'
    name: fig-imp-likelihood-ratio
---
plt.plot(w_range, l(w_range), lw=2)
plt.yscale('log')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\ell(\omega)$')
plt.show()
```

请注意纵轴使用了对数坐标：如果使用线性坐标，整条曲线会被左边缘的尖峰压平到接近零的位置。

{numref}`fig-imp-densities` 显示，当 $\omega \rightarrow 0$ 时，$f \left(\omega\right)$ 保持不变，而 $g \left(\omega\right) \rightarrow 0$。

因此 {numref}`fig-imp-likelihood-ratio` 中的似然比以 $\omega^{-2}$ 的速率发散至无穷大。

当 $\omega \rightarrow 1$ 时也会出现同样的情况，因为 $G_b > 1$ 也会迫使 $g\left(\omega\right) \rightarrow 0$，这就解释了右边缘的上扬现象。

但那里的发散速度要慢得多——速率为 $\left(1-\omega\right)^{-1/5}$——正是 $\omega = 0$ 附近的行为才导致了下文中的问题。

对 $E \left[L\left(\omega^t\right)\right]$ 的蒙特卡洛近似会重复从 $g$ 中抽取 $t$ 个独立观测值组成的序列 $\omega^t = \left(\omega_1, \ldots, \omega_t\right)$，对每个这样的序列计算乘积 $L\left(\omega^t\right) = \prod_{i=1}^t \ell \left(\omega_i\right)$，然后对独立抽取的多个序列所得的乘积取平均值。

由于当 $\omega \rightarrow 0$ 时 $g(\omega) \rightarrow 0$，这种模拟程序对样本空间 [0,1] 中的某些部分采样不足，而这些部分对于很好地近似 $\ell \left(\omega\right)$ 的数学期望来说是需要经常访问的重要区域。

$L\left(\omega^t\right)$ 中的每一个因子都以同样的方式发生了扭曲，因此这个问题会随着 $t$ 的增大而不断累积加剧。

我们在下面通过数值方法说明这一点。

## 重要性采样

我们通过使用一种称为**重要性采样**的_分布变换_来解决这个问题。

在模拟过程中，我们不从 $g$ 中抽取数据，而是使用另一个分布 $h$ 来生成 $\omega$ 的抽样。

在 {doc}`likelihood_ratio_process` 中，$h$ 表示生成数据的第三个密度；而在这里，$h$ 表示我们所选择的重要性分布。

这个想法是设计 $h$ 使其对 $\Omega$ 中那些 $\ell \left(\omega_t\right)$ 取值较大但在 $g$ 下密度较低的区域进行过采样。

用这种方式构造样本后，当我们计算似然比的经验均值时，必须用 $g$ 和 $h$ 的似然比对每个实现值进行加权。

通过这样做，我们恰当地考虑到了我们使用 $h$ 而不是 $g$ 来模拟数据的事实。

为了说明，假设我们对 ${E}\left[\ell\left(\omega\right)\right]$ 感兴趣。

我们可以简单地计算：

$$
\hat{E}^g \left[\ell\left(\omega\right)\right] = \frac{1}{N} \sum_{i=1}^{N} \ell\left(\omega_i^g\right)
$$

其中 $\omega_i^g$ 表示 $\omega_i$ 是从 $g$ 中抽取的。

但是利用重要性抽样的见解，我们可以改为计算这个对象：

$$
\hat{E}^h \left[\ell\left(\omega\right) \frac{g\left(\omega\right)}{h\left(\omega\right)} \right] = \frac{1}{N} \sum_{i=1}^{N} \ell\left(\omega_i^h\right) \frac{g\left(\omega_i^h\right)}{h\left(\omega_i^h\right)}
$$

其中 $\omega_i$ 现在是从重要性分布 $h$ 中抽取的。

注意上述两个在总体上是完全相同的对象：

$$
E^g\left[\ell\left(\omega\right)\right] = \int_\Omega \ell(\omega) g(\omega) d\omega = \int_\Omega \ell(\omega) \frac{g(\omega)}{h(\omega)} h(\omega) d\omega = E^h\left[\ell\left(\omega\right) \frac{g(\omega)}{h(\omega)}\right]
$$

第二个等式要求只要 $\ell(\omega) g(\omega) = f(\omega) > 0$，就有 $h(\omega) > 0$。

由于我们必须使用一个在 $g$ 分布质量较低的部分具有更大质量的 $h$，我们选用 $h=Beta(0.5, 0.5)$ 作为我们的重要性分布。

下面的图比较了 $g$ 和 $h$。

```{code-cell} ipython3
g_a, g_b = params.G_a, params.G_b
h_a, h_b = 0.5, 0.5
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '数据分布与重要性采样分布'
    name: fig-imp-real
---
w_range = np.linspace(1e-5, 1-1e-5, 1000)

plt.plot(w_range, g(w_range),
         lw=2, label=f'g=Beta({g_a}, {g_b})')
plt.plot(w_range, beta.pdf(w_range, 0.5, 0.5),
         lw=2, label=f'h=Beta({h_a}, {h_b})')
plt.legend()
plt.ylim([0., 3.])
plt.show()
```

## 近似累积似然比

我们现在研究如何使用重要性抽样来近似
${E} \left[L\left(\omega^t\right)\right] = E \left[\prod_{i=1}^t \ell \left(\omega_i\right)\right]$。

为了编写适用于一般贝塔密度的代码，我们用 $p$ 表示我们想要求期望所依据的密度，用 $q$ 表示我们实际抽样所依据的密度。

在我们的应用中，$p = g$，而 $q$ 是一个重要性分布，例如 $h$。

在 {doc}`likelihood_ratio_process` 中，$q$ 表示自然选择的密度；而在这里，$q$ 表示重要性分布。

如上所述，我们的计划是从 $q$ 中抽取序列 $\omega^t$，然后对似然比进行适当的重新加权。

分布的变换在总体上成立，这与单次抽取的情况完全一样：

$$
E^p \left[L\left(\omega^t\right)\right] = E^p \left[\prod_{i=1}^t \ell \left(\omega_i\right)\right] = E^q \left[\prod_{i=1}^t \ell \left(\omega_i\right) \frac{p\left(\omega_i\right)}{q\left(\omega_i\right)}\right]
$$

这启发了以下估计量

$$
\hat{E}^q \left[L\left(\omega^t\right)\right] =
\frac{1}{N} \sum_{n=1}^{N}\left( \prod_{i=1}^{t} \ell\left(\omega_{n,i}^q\right)\frac{p\left(\omega_{n,i}^q\right)}{q\left(\omega_{n,i}^q\right)}\right)
$$

其中 $\omega_{n,i}^q$ 是从重要性分布 $q$ 中抽取的第 $n$ 条序列的第 $i$ 个观测值。

这里 $\frac{p\left(\omega_{n,i}^q\right)}{q\left(\omega_{n,i}^q\right)}$ 是我们分配给每个数据点 $\omega_{n,i}^q$ 的权重。

下面我们准备一个 Python 函数，用于计算给定任何 beta 分布 $p$、$q$ 的重要性抽样估计。

```{code-cell} ipython3
def estimate_single_path(key, p_a, p_b, q_a, q_b, T):
    """
    Estimation for a single sample path.
    """

    def loop_body(i, carry):
        L, weight, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        w = jax.random.beta(subkey, q_a, q_b)

        # Keep draws off the boundary, where the log densities
        # below produce the undefined form 0 * inf
        w = jnp.clip(w, 1e-12, 1 - 1e-12)

        # Compute likelihood ratio using f/g functions
        likelihood_ratio = f(w) / g(w)
        L = L * likelihood_ratio

        # Importance sampling weight
        p_w = beta.pdf(w, p_a, p_b)
        q_w = beta.pdf(w, q_a, q_b)
        weight = weight * (p_w / q_w)

        return (L, weight, key_state)

    # Use fori_loop for dynamic T values
    final_L, final_weight, _ = jax.lax.fori_loop(
        0, T, loop_body, (1.0, 1.0, key)
    )
    return final_L * final_weight

@partial(jax.jit, static_argnames=['N'])
def estimate(key, p_a, p_b, q_a, q_b, T=1, N=10000):
    """Estimation of a batch of sample paths."""
    keys = jax.random.split(key, N)

    # Vectorize over keys, holding the parameters fixed
    estimates = jax.vmap(
        estimate_single_path,
        in_axes=(0, None, None, None, None, None)
    )(keys, p_a, p_b, q_a, q_b, T)

    return jnp.mean(estimates)
```

考虑 $T=1$ 的情况，这相当于近似 $E\left[\ell\left(\omega\right)\right]$。

对于标准蒙特卡洛估计，我们可以设置 $p=g$ 和 $q=g$。

```{code-cell} ipython3
estimate(jax.random.key(0), g_a, g_b, g_a, g_b,
         T=1, N=10000)
```

对于我们的重要性抽样估计，我们设定 $q = h$。

```{code-cell} ipython3
estimate(jax.random.key(1), g_a, g_b, h_a, h_b,
         T=1, N=10000)
```

显然，即使在 $T=1$ 时，我们的重要性采样估计比蒙特卡洛估计更接近 $1$。

在计算更长序列的期望值 $E\left[L\left(\omega^t\right)\right]$ 时，差异会更大。

当设置 $T=10$ 时，我们发现蒙特卡洛方法严重低估了均值，而重要性采样仍然产生接近其理论值 1 的估计。

```{code-cell} ipython3
estimate(jax.random.key(2), g_a, g_b, g_a, g_b,
         T=10, N=10000)
```

```{code-cell} ipython3
estimate(jax.random.key(3), g_a, g_b, h_a, h_b,
         T=10, N=10000)
```

蒙特卡洛方法会低估是因为在分布 $g$ 下，似然比 $L(\omega^T) = \prod_{t=1}^T \frac{f(\omega_t)}{g(\omega_t)}$ 具有高度偏斜的分布。

从 $g$ 中抽取的大多数样本产生较小的似然比，而真实均值需要偶尔出现的非常大的值，这些值很少被采样到。

在我们的情况下，由于当 $\omega \to 0$ 时 $g(\omega) \to 0$ 而 $f(\omega)$ 保持不变，蒙特卡洛过程恰恰在似然比 $\frac{f(\omega)}{g(\omega)}$ 最大的地方采样不足。

事实上情况比偏斜还要严重——蒙特卡洛估计量的方差是**无穷大**的。

要说明这一点，注意单个似然比的二阶矩为

$$
E^g \left[\ell\left(\omega\right)^2\right]
= \int_0^1 \left(\frac{f\left(\omega\right)}{g\left(\omega\right)}\right)^2 g\left(\omega\right) d\omega
= \int_0^1 \frac{f\left(\omega\right)^2}{g\left(\omega\right)} d\omega
$$

这里 $f$ 是均匀密度，而 $g\left(\omega\right)$ 在 $\omega \to 0$ 时以 $\omega^{G_a - 1} = \omega^2$ 的速度趋于零。

因此被积函数在原点附近的行为类似于 $\omega^{-2}$，该积分是发散的。

所以 $\ell\left(\omega\right)$ 没有有限的方差，并且由于各次抽取是独立的，对于任意 $t$，$L\left(\omega^t\right)$ 同样没有有限的方差。

{doc}`likelihood_ratio_process` 中遇到的发散积分与 $\ell(\omega)$ 在 $f$ 下的均值相同，因为 $E^g\left[\ell(\omega)^2\right] = E^f\left[\ell(\omega)\right]$。

均值 $E^g\left[\ell(\omega)\right] = 1$ 是有限的，因此强大数定律仍然适用：随着抽取次数 $N$ 的增长，蒙特卡洛平均值几乎必然收敛到 $1$。

但普通的中心极限定理并不适用，其 $N^{-1/2}$ 的收敛速率也会失效。

在原点附近，$g(\omega) \approx k \omega^2$，其中 $k = 1/B(3, 1.2)$，$B$ 为贝塔函数。

对于较大的 $x$，事件 $\ell(\omega) > x$ 主要发生在 $\omega < (k x)^{-1/2}$ 时，在 $g$ 下该事件的概率约为 $\frac{1}{3} k^{-1/2} x^{-3/2}$。

以 $x^{-3/2}$ 速度衰减的尾部使 $\ell(\omega)$ 落入指数为 $3/2$ 的稳定律的吸引域。

根据广义中心极限定理，由 $N$ 次抽取构成的样本均值的误差以 $N^{-1/3}$ 的速度缩小，其极限分布是偏斜的，具有较长的右尾。

{ref}`imp_ex2` 通过数值方式验证了该尾指数以及 $N^{-1/3}$ 的收敛速率。

随着 $T$ 的增加，这个问题会不断恶化，使标准蒙特卡洛方法变得越来越不可靠。

使用 $q = h$ 的重要性采样通过在 $\ell$ 取值较大的 $\omega = 0$ 附近更频繁地采样，解决了方差无穷大的问题。

## 样本均值的分布

接下来我们研究蒙特卡洛和重要性采样方法的偏差和效率。

下面的代码将估计重复`N_simu`次，这样我们就可以查看每种方法产生的估计值的分布。

```{code-cell} ipython3
@partial(jax.jit, static_argnames=['N_simu', 'N_samples'])
def simulate(key, p_a, p_b, q_a, q_b, N_simu, T=1,
             N_samples=10000):
    """Repeat the estimate N_simu times, drawing from q."""
    keys = jax.random.split(key, N_simu)
    return jax.vmap(
        lambda k: estimate(k, p_a, p_b, q_a, q_b, T,
                           N_samples)
    )(keys)
```

设置 $q = p$ 可以恢复标准蒙特卡洛方法，因为这样每个重要性权重都恒等于1。

再次，我们首先通过设置T=1来估计${E} \left[\ell\left(\omega\right)\right]$。

我们对每种方法进行1000次模拟。

```{code-cell} ipython3
N_simu = 1000
μ_L_g = simulate(jax.random.key(4), g_a, g_b,
                 g_a, g_b, N_simu)
μ_L_h = simulate(jax.random.key(5), g_a, g_b,
                 h_a, h_b, N_simu)
```

```{code-cell} ipython3
# 标准蒙特卡洛（均值和方差）
jnp.mean(μ_L_g), jnp.var(μ_L_g)
```

```{code-cell} ipython3
# 重要性采样（均值和方差）
jnp.mean(μ_L_h), jnp.var(μ_L_h)
```

虽然两种方法都倾向于给出接近1的${E} \left[\ell\left(\omega\right)\right]$均值估计，但重要性抽样估计的方差更小。

接下来，我们展示$\hat{E} \left[L\left(\omega^t\right)\right]$在$T=1, 5, 10, 20$这些情况下的估计值分布。

```{code-cell} ipython3
def simulate_multiple_T(key, p_a, p_b, q_a, q_b, T_values,
                        N_simu, N_samples=10000):
    """Run simulate once per T, returning a dict keyed by T."""
    keys = jax.random.split(key, len(T_values))
    return {T: simulate(keys[i], p_a, p_b, q_a, q_b,
                        N_simu, T, N_samples)
            for i, T in enumerate(T_values)}
```

下面这个函数绘制我们用来比较两种方法的直方图。

```{code-cell} ipython3
def plot_estimates(T_values, mc, imp, imp_label, n_rows=1):
    """Compare Monte Carlo and importance sampling estimates."""
    n_cols = len(T_values) // n_rows
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(14, 5 * n_rows))
    μ_range = np.linspace(0, 2, 100)

    for ax, T in zip(np.ravel(axs), T_values):
        μ_L_g, μ_L_h = np.asarray(mc[T]), np.asarray(imp[T])

        ax.set_xlabel('$μ_L$')
        ax.set_ylabel('frequency')
        ax.set_title(f'$T$={T}')
        ax.hist(μ_L_g, bins=μ_range,
                color='r', alpha=0.5, label='$g$ generating')
        ax.hist(μ_L_h, bins=μ_range,
                color='b', alpha=0.5, label=imp_label)
        ax.legend(loc=4)

        # Summarize each distribution in an upper corner
        for μ_L, color, x, ha in ((μ_L_g, 'r', 0.02, 'left'),
                                  (μ_L_h, 'b', 0.98, 'right')):
            ax.text(x, 0.98, transform=ax.transAxes, ha=ha,
                    va='top', color=color, fontsize=9,
                    s=r'$\hat{μ}$=' + f'{np.mean(μ_L):.3g}' +
                      '\n' + 'med=' + f'{np.median(μ_L):.3g}' +
                      '\n' + r'$\hat{σ}^2$=' + f'{np.var(μ_L):.3g}')

    plt.show()
```

蒙特卡洛估计不依赖于重要性分布，因此我们只计算一次，并在下面的每个图中重复使用它们。

```{code-cell} ipython3
T_values = [1, 5, 10, 20]

mc = simulate_multiple_T(jax.random.key(6), g_a, g_b,
                         g_a, g_b, T_values, N_simu)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 'Monte Carlo and importance sampling estimates'
    name: fig-imp-estimates
---
imp_h1 = simulate_multiple_T(jax.random.key(7), g_a, g_b,
                             h_a, h_b, T_values, N_simu)

plot_estimates(T_values, mc, imp_h1, '$h$ generating',
               n_rows=2)
```

上述模拟练习表明，对于每一个$T$，重要性抽样估计都保持在$1$附近，而标准蒙特卡洛估计的分布则随着$T$的增加而持续向左偏移。

有必要仔细分析一下这里究竟出了什么问题。

蒙特卡洛估计量在总体意义上是无偏的——对于每个$T$，其期望恰好为$1$。

随着$T$的增大而恶化的是其抽样分布的*形状*：几乎所有的概率质量都向零坍缩，而期望值则由罕见的极端离群值来维持。

这就是为什么每个面板都同时报告了中位数。

中位数随$T$稳步下降，而报告的均值$\hat{μ}$则是一个不可靠的统计量——回想一下，其潜在方差是无穷的，因此每个面板中打印的$\hat{σ}^2$所估计的量根本不存在，并且换一个不同的随机种子就可能使$\hat{μ}$发生很大的变化。

使用$h$进行重要性抽样消除了无穷方差的问题，但并不能阻止方差随$T$的增大而增长。

从$h$中抽取一条路径并进行重加权，会使该路径获得值$\prod_{i=1}^T \ell(\omega_i) g(\omega_i)/h(\omega_i) = \prod_{i=1}^T f(\omega_i)/h(\omega_i)$。

其二阶矩为$\left(\int_0^1 f^2/h \, d\omega\right)^T = \left(1 + \chi^2(f\|h)\right)^T$，其中$\chi^2(f\|h) = \int_0^1 (f-h)^2/h \, d\omega$是$f$相对于$h$的$\chi^2$散度（参见 {doc}`divergence_measures`）。

对于$h = $ 贝塔分布$(0.5, 0.5)$，每个权重$f(\omega)/h(\omega) = \pi \sqrt{\omega(1-\omega)}$至多为$\pi/2$，并且$\int_0^1 f^2/h \, d\omega = \pi^2/8 \approx 1.234$。

单条路径贡献的方差$(\pi^2/8)^T - 1$是有限的，但会呈几何级数增长：在$T = 20$时等于$66$，在$T = 50$时约为$3.6 \times 10^4$。

## 选择抽样分布

上文中，我们任意选择了 $h = Beta(0.5,0.5)$ 作为重要性分布。

是否存在一个最优的重要性分布？

当重要性密度与被积函数（此处为 $\ell g$）成比例时，重要性抽样估计量的方差为零。

由于 $\ell g = f$，对于单次抽取，零方差重要性密度就是 $f$ 本身；对于长度为 $T$ 的路径，它是乘积密度 $f(\omega_1) f(\omega_2) \cdots f(\omega_T)$。

要使用它，我们只需要能够从 $f$ 中抽样即可；我们不需要事先知道 $E \left[ L\left(\omega^t\right) \right] = 1$。

因此，假设我们直接使用 $h = f$。

在估计似然比的均值时（$T=1$），我们得到：

$$
\hat{E}^f \left[\ell(\omega) \frac{g(\omega)}{f(\omega)} \right] = \hat{E}^f \left[\frac{f(\omega)}{g(\omega)} \frac{g(\omega)}{f(\omega)} \right] = \frac{1}{N} \sum_{i=1}^{N} \ell\left(\omega_i^f\right) \frac{g\left(\omega_i^f\right)}{f\left(\omega_i^f\right)} = 1
$$

```{code-cell} ipython3
μ_L_f = simulate(jax.random.key(8), g_a, g_b,
                 params.F_a, params.F_b, N_simu)
```

```{code-cell} ipython3
# importance sampling (mean and variance)
jnp.mean(μ_L_f), jnp.var(μ_L_f)
```

我们也可以使用其他分布作为我们的重要性分布。

下面我们选取几个分布，并比较它们的抽样特性。

```{code-cell} ipython3
a_list = [0.5, 1., 2.]
b_list = [0.5, 1.2, 5.]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '重要性抽样分布的比较'
    name: fig-imp-sampling-distributions
---
w_range = np.linspace(1e-5, 1-1e-5, 1000)

plt.plot(w_range, g(w_range),
         lw=2, label=f'g=Beta({g_a}, {g_b})')
plt.plot(w_range, beta.pdf(w_range, a_list[0], b_list[0]),
         lw=2, label=f'$h_1$=Beta({a_list[0]},{b_list[0]})')
plt.plot(w_range, beta.pdf(w_range, a_list[1], b_list[1]),
         lw=2, label=f'$h_2$=Beta({a_list[1]},{b_list[1]})')
plt.plot(w_range, beta.pdf(w_range, a_list[2], b_list[2]),
         lw=2, label=f'$h_3$=Beta({a_list[2]},{b_list[2]})')
plt.legend()
plt.ylim([0., 3.])
plt.show()
```

我们再考虑另外两个分布。

提醒一下，$h_1$ 就是我们上面使用的原始 $Beta(0.5,0.5)$ 分布。

$h_2$ 是 $Beta(1,1.2)$ 分布。

注意 $h_2$ 在 $\omega$ 较大的取值处与 $g$ 的形状相似，但在较小的取值处有更多的质量。

我们的直觉是 $h_2$ 应该是一个很好的重要性抽样分布。

$h_3$ 是 $Beta(2,5)$ 分布。

注意 $h_3$ 在接近 0 和接近 1 的取值处几乎没有质量。

我们的直觉是 $h_3$ 会是一个较差的重要性抽样分布。

上面的方差计算可以让我们精确地验证这些直觉。

从 $h$ 中抽样并重新加权，会使每个观测值获得权重 $\ell\left(\omega\right) g\left(\omega\right) / h\left(\omega\right) = f\left(\omega\right) / h\left(\omega\right)$，因此估计量具有有限方差，当且仅当

$$
\int_0^1 \frac{f\left(\omega\right)^2}{h\left(\omega\right)} d\omega < \infty
$$

对于一个 Beta$(a, b)$ 重要性密度 $h$ 和均匀分布 $f$，

$$
\int_0^1 \frac{f\left(\omega\right)^2}{h\left(\omega\right)} d\omega 
= B(a, b) \int_0^1 \omega^{1-a} (1-\omega)^{1-b} d\omega 
= B(a, b) \, B(2-a, 2-b)
$$

当 $a < 2$ 且 $b < 2$ 时成立，否则该积分为无穷大。

该积分对 $h_1$ 等于 $\pi^2/8 \approx 1.234$，对 $h_2$ 等于 $1/0.96 \approx 1.042$，对 $h = f$ 等于 $1$。

候选分布 $h_2$ 之所以优良，是因为它与 $f$ 很接近。

对于 $h_3$，该积分在**两个**端点处都是发散的，因为 $h_3$ 在原点处以 $\omega$ 的速度趋于零，在 1 处以 $\left(1-\omega\right)^4$ 的速度趋于零。

因此使用 $h_3$ 进行重要性抽样具有无穷方差，这和标准蒙特卡洛方法一样，正是这一点使其成为一个糟糕的选择。

我们首先模拟并绘制使用 $h_2$ 作为重要性抽样分布时，$\hat{E} \left[L\left(\omega^t\right)\right]$ 估计值的分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '使用重要性分布 $h_2$ 的估计结果'
    name: fig-imp-estimates-h2
---
T_values_h2 = [1, 20]
imp_h2 = simulate_multiple_T(jax.random.key(9), g_a, g_b,
                             a_list[1], b_list[1],
                             T_values_h2, N_simu)

plot_estimates(T_values_h2, mc, imp_h2, '$h_2$ generating')
```

我们的模拟结果表明，$h_2$ 确实是我们这个问题的一个相当好的重要性抽样分布。

即使在 $T=20$ 时，均值也非常接近 $1$，而且方差很小。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: '使用重要性分布 $h_3$ 的估计结果'
    name: fig-imp-estimates-h3
---
T_values_h3 = [1, 20]
imp_h3 = simulate_multiple_T(jax.random.key(10), g_a, g_b,
                             a_list[2], b_list[2],
                             T_values_h3, N_simu)

plot_estimates(T_values_h3, mc, imp_h3, '$h_3$ generating')
```

然而，对于我们的这个问题，$h_3$ 显然是一个较差的重要性抽样分布，在 $T = 20$ 时其均值估计远离 $1$。

由于 $h_3 > 0$ 在 $(0,1)$ 上成立，使用 $h_3$ 的重要性抽样估计量对任意 $T$ 都是无偏的，正如在 $g$ 之下的蒙特卡洛估计量一样。

问题出在其抽样分布的形状上。

即使在 $T = 1$ 时，使用 $h_3$ 得到的估计值也比在 $g$ 之下的蒙特卡洛估计值更加分散，并且中位数估计值更明显地低于 $1$。

在 $\omega = 1$ 附近，$h_3$ 以 $(1-\omega)^4$ 的速度趋于零，而 $f = 1$，因此在 $h_3$ 之下，权重 $f/h_3$ 的尾指数为 $5/4$，小于在 $g$ 之下 $f/g$ 的尾指数 $3/2$。

因此，由 $N$ 次抽取构成的样本均值的误差，在 $h_3$ 之下按 $N^{-1/5}$ 的速度缩小，而在 $g$ 之下则按 $N^{-1/3}$ 的速度缩小（参见 {ref}`imp_ex2`）。

因此，我们的模拟结果表明，对于我们的这个问题，直接在 $g$ 之下使用蒙特卡洛近似，要比使用 $h_3$ 作为重要性抽样分布更好。

## 练习

```{exercise}
:label: imp_ex1

考虑通过从 Beta$(a, b)$ 重要性密度 $q$ 中抽取长度为 $T$ 的路径来估计 $E^g\left[L\left(\omega^T\right)\right] = 1$。

1. 证明每条路径对估计量的贡献为 $\prod_{i=1}^T f(\omega_i)/q(\omega_i)$，并且该贡献的二阶矩为 $M(a,b)^T$，其中当 $a, b < 2$ 时 $M(a,b) = \int_0^1 f^2/q \, d\omega = B(a,b) B(2-a, 2-b)$，否则 $M(a,b) = +\infty$。
2. 使用柯西-施瓦茨不等式证明 $M(a,b) \geq 1$，且仅在 $(a,b) = (1,1)$ 处取等号。在 $(0,2) \times (0,2)$ 上绘制 $\log M(a,b)$ 的等高线图。
3. $N$ 条加权路径的有效样本量约为 $N / M(a,b)^T$。绘制 $h_1$ 和 $h_2$ 的有效样本量与 $N$ 的比值随 $T$ 变化的图像。在 $T = 20$ 和 $T = 50$ 时，$h_1$ 需要多少条路径才能使有效样本量达到 $1000$？
```

```{solution-start} imp_ex1
:class: dropdown
```

**第 1 部分。** 每条路径贡献 $\prod_{i=1}^T \ell(\omega_i) g(\omega_i)/q(\omega_i) = \prod_{i=1}^T f(\omega_i)/q(\omega_i)$。

因为 $\omega_i$ 是来自 $q$ 的独立同分布抽样，二阶矩为 $\left(E^q\left[(f/q)^2\right]\right)^T = M(a,b)^T$，其中 $M(a,b) = \int_0^1 f^2/q \, d\omega$。

由于 $f = 1$，$M(a,b) = B(a,b) \int_0^1 \omega^{1-a} (1-\omega)^{1-b} d\omega$，当 $a < 2$ 且 $b < 2$ 时该值等于 $B(a,b) B(2-a, 2-b)$，否则为无穷大。

**第 2 部分。** 由柯西-施瓦茨不等式，

$$
1 = \left(\int_0^1 f \, d\omega\right)^2 
= \left(\int_0^1 \frac{f}{\sqrt{q}} \sqrt{q} \, d\omega\right)^2 
\leq \int_0^1 \frac{f^2}{q} d\omega \int_0^1 q \, d\omega = M(a,b) ,
$$

当且仅当 $f/\sqrt{q}$ 与 $\sqrt{q}$ 成比例时取等号，即 $q = f$。

单个权重的方差为 $M(a,b) - 1 = \chi^2(f\|q)$。

```{code-cell} ipython3
from scipy.special import betaln

def M(a, b):
    """q = Beta(a, b) 时重要性权重 f/q 的二阶矩。"""
    if a < 2 and b < 2:
        return np.exp(betaln(a, b) + betaln(2 - a, 2 - b))
    return np.inf

grid = np.linspace(0.05, 1.95, 191)
log_M = np.array([[np.log(M(a, b)) for a in grid] for b in grid])

fig, ax = plt.subplots(figsize=(7, 6))
cs = ax.contourf(grid, grid, np.minimum(log_M, 2), 
                 levels=np.linspace(0, 2, 21), cmap='viridis')
fig.colorbar(cs, label=r'$\log M(a, b)$ (capped at 2)')
lines = ax.contour(grid, grid, log_M, levels=[0.01, 0.05, 0.2, 0.5, 1],
                   colors='w', linewidths=0.8)
ax.clabel(lines, fmt='%.2f', fontsize=8)
for (a, b), name in zip([(0.5, 0.5), (1.0, 1.2), (1.0, 1.0)],
                        ['$h_1$', '$h_2$', '$f$']):
    ax.plot(a, b, 'wo')
    ax.annotate(name, (a, b), xytext=(5, 5), textcoords='offset points',
                color='w')
ax.set_xlabel('$a$')
ax.set_ylabel('$b$')
plt.show()

i_b, i_a = np.unravel_index(np.argmin(log_M), log_M.shape)
print(f"Minimum of M on the grid at (a, b) = ({grid[i_a]:.2f}, {grid[i_b]:.2f})")
print(f"M(h_1) = {M(0.5, 0.5):.4f}, M(h_2) = {M(1, 1.2):.4f}")
```

**第 3 部分。**

```{code-cell} ipython3
T_grid = np.arange(1, 51)
plt.figure(figsize=(8, 5))
for (a, b), name in zip([(0.5, 0.5), (1.0, 1.2)], ['$h_1$', '$h_2$']):
    plt.semilogy(T_grid, M(a, b)**(-T_grid), lw=2, label=name)
plt.xlabel('$T$')
plt.ylabel('effective sample size / $N$')
plt.legend()
plt.show()

for T in [20, 50]:
    print(f"T = {T}: h_1 needs N = {1000 * M(0.5, 0.5)**T:,.0f} paths, "
          f"h_2 needs N = {1000 * M(1, 1.2)**T:,.0f} paths")
```

在 $h_1$ 下，有效样本量每增加一期都会按因子 $8/\pi^2$ 缩小。

```{solution-end}
```

```{exercise}
:label: imp_ex2

1. 设 $\omega \sim g$。证明 $\Pr\{\ell(\omega) > x\}$ 在 $x$ 较大时大致正比于 $x^{-3/2}$，因此 $\ell(\omega)$ 在 $g$ 下的尾指数为 $3/2$。
2. 设 $\omega \sim h_3 = $ Beta$(2, 5)$。证明权重 $f(\omega)/h_3(\omega)$ 在 $h_3$ 下的尾指数为 $5/4$。$(0,1)$ 的哪个端点决定了尾指数？
3. 用希尔估计量 {cite}`hill1975` 确认这两个尾指数，该估计量使用 $n$ 个抽样中最大的 $k$ 个值 $X_{(1)} \geq X_{(2)} \geq \cdots \geq X_{(k+1)}$ 来估计尾指数：

   $$
   \hat \alpha_k = \left(\frac{1}{k} \sum_{i=1}^k \log \frac{X_{(i)}}{X_{(k+1)}}\right)^{-1} .
   $$

4. 设 $T = 1$。对于 $N = 10^2, 10^3, 10^4, 10^5$，当 $q$ 分别为 $g$（标准蒙特卡洛）、$h_2$ 和 $h_3$ 时，计算从 $q$ 中抽取 $N$ 个样本得到的 $f/q$ 的 $500$ 次重复样本均值。在对数-对数坐标上绘制重复样本均值的四分位距与 $N$ 的关系，并将斜率与 $-1/3$、$-1/2$ 和 $-1/5$ 进行比较。
```

```{solution-start} imp_ex2
:class: dropdown
```

**第 1 部分。** 在原点附近，$g(\omega) \approx k \omega^2$，其中 $k = 1/B(3, 1.2)$。

当 $x$ 较大时，$\ell(\omega) = 1/g(\omega) > x$ 主要在 $\omega < (kx)^{-1/2}$ 时发生。

由于当 $u$ 较小时 $\Pr_g\{\omega < u\} \approx k u^3/3$，

$$
\Pr_g\{\ell(\omega) > x\} \approx \frac{k}{3} (kx)^{-3/2} = \frac{1}{3} k^{-1/2} x^{-3/2} .
$$

在 $\omega = 1$ 附近，$g$ 以 $(1-\omega)^{0.2}$ 的速度趋于零，这贡献了一个正比于 $x^{-6}$ 的尾部，太薄而不足为虑。

**第 2 部分。** 密度 $h_3(\omega) = 30\, \omega (1-\omega)^4$ 在两个端点都趋于零。

在 $\omega = 0$ 附近，$1/h_3 > x$ 在 $\omega < 1/(30x)$ 时发生，其概率约为 $15/(30x)^2$，正比于 $x^{-2}$。

在 $\omega = 1$ 附近，$1/h_3 > x$ 在 $1 - \omega < (30x)^{-1/4}$ 时发生，其概率约为 $6 (30x)^{-5/4}$，正比于 $x^{-5/4}$。

尾指数为 $5/4$ 的较重尾部来自右端点。

根据广义中心极限定理，尾指数 $\alpha \in (1,2)$ 使样本均值的误差按 $N^{1/\alpha - 1}$ 的速度缩小：当 $\alpha = 3/2$ 时为 $N^{-1/3}$，当 $\alpha = 5/4$ 时为 $N^{-1/5}$。

**第 3 部分。**

```{code-cell} ipython3
def log_beta_pdf(w, a, b):
    """Beta(a, b) 的对数密度。"""
    return (a - 1) * np.log(w) + (b - 1) * np.log1p(-w) - betaln(a, b)

def hill(x, k):
    """由 x 中最大的 k 个值得到的尾指数希尔估计量。"""
    x_sorted = np.sort(x)[::-1]
    return 1 / np.mean(np.log(x_sorted[:k] / x_sorted[k]))

rng = np.random.default_rng(0)
n = 1_000_000
w_g = rng.beta(g_a, g_b, n)
w_h3 = rng.beta(2, 5, n)
weights = {'f/g under g': np.exp(-log_beta_pdf(w_g, g_a, g_b)),
           'f/h_3 under h_3': np.exp(-log_beta_pdf(w_h3, 2, 5))}

for name, x in weights.items():
    estimates = [f"{hill(x, k):.3f}" for k in (500, 1000, 2000, 5000)]
    print(f"{name}: Hill estimates for k = 500, 1000, 2000, 5000: {estimates}")
```

希尔估计值接近 $3/2$ 和 $5/4$。

**第 4 部分。**

```{code-cell} ipython3
N_values = np.array([10**2, 10**3, 10**4, 10**5])
R = 500
proposals = {'$g$ (standard Monte Carlo)': (g_a, g_b, 1/3),
             '$h_2$': (1.0, 1.2, 1/2),
             '$h_3$': (2.0, 5.0, 1/5)}

plt.figure(figsize=(8, 5))
for name, (a, b, rate) in proposals.items():
    iqr = []
    for N in N_values:
        means = np.empty(R)
        for r in range(R):
            w = np.clip(rng.beta(a, b, N), 1e-16, 1 - 1e-16)
            means[r] = np.mean(np.exp(-log_beta_pdf(w, a, b)))
        q75, q25 = np.quantile(means, [0.75, 0.25])
        iqr.append(q75 - q25)
    slope = np.polyfit(np.log(N_values), np.log(iqr), 1)[0]
    line, = plt.loglog(N_values, iqr, 'o-', 
                       label=f'{name}: slope {slope:.2f}')
    plt.loglog(N_values, iqr[0] * (N_values / N_values[0])**(-rate), 
               '--', color=line.get_color(), lw=1)
plt.xlabel('$N$')
plt.ylabel('interquartile range of replicate means')
plt.legend()
plt.show()
```

虚线的斜率分别为 $-1/3$、$-1/2$ 和 $-1/5$。

拟合的斜率与之接近。

```{solution-end}
```

```{exercise}
:label: imp_ex3

在 {doc}`likelihood_ratio_process` 的 {ref}`这一节 <lrp_chernoff>` 中，第一类错误的概率 $\alpha_T = \Pr_f\{L(\omega^T) < 1\}$ 随 $T$ 呈指数衰减，因此标准蒙特卡洛方法无法在 $T$ 较大时对其进行估计。

设 $\phi^*$ 使切尔诺夫系数 $\rho(\phi) = \int_0^1 f^\phi g^{1-\phi} d\omega$ 最小化，并设 $q = f^{\phi^*} g^{1-\phi^*}/\rho(\phi^*)$ 为倾斜密度。

1. 证明 $q$ 是 Beta$(\phi^* F_a + (1-\phi^*) G_a, \ \phi^* F_b + (1-\phi^*) G_b)$ 密度，并计算 $\phi^*$ 以及 $q$ 的参数。
2. 证明 

   $$
   \alpha_T = \rho(\phi^*)^T \, E^q\left[\mathbf{1}\{L(\omega^T) < 1\} \, L(\omega^T)^{1-\phi^*}\right] .
   $$

3. 对于 $T = 30, 60, 100$，用从 $f$ 中抽取的 $N = 20{,}000$ 条路径进行标准蒙特卡洛估计 $\alpha_T$，并用从 $q$ 中抽取的 $N = 20{,}000$ 条路径进行重要性抽样估计。报告重要性抽样估计量的相对标准误差。
```

```{solution-start} imp_ex3
:class: dropdown
```

**第 1 部分。** 对于 Beta 密度，

$$
f(\omega)^\phi g(\omega)^{1-\phi} \propto \omega^{\phi F_a + (1-\phi) G_a - 1} (1-\omega)^{\phi F_b + (1-\phi) G_b - 1} ,
$$

这与所述的 Beta 密度成比例。

**第 2 部分。** 由于 $f/q = \rho(\phi^*) (f/g)^{1-\phi^*}$，

$$
\alpha_T = E^q\left[\mathbf{1}\{L(\omega^T) < 1\} \prod_{i=1}^T \frac{f(\omega_i)}{q(\omega_i)}\right]
= \rho(\phi^*)^T \, E^q\left[\mathbf{1}\{L(\omega^T) < 1\} \, L(\omega^T)^{1-\phi^*}\right] .
$$

在事件 $L(\omega^T) < 1$ 上，权重 $\rho(\phi^*)^T L(\omega^T)^{1-\phi^*}$ 至多为 $\rho(\phi^*)^T = e^{-C(f,g) T}$，因此该估计量的方差较小。

在 $\phi^*$ 处，$\log \rho(\phi)$ 的导数（等于 $E^q \log \ell(\omega)$）为零。

因此 $\log L(\omega^T)$ 在 $q$ 下均值为零，而在 $f$ 下的罕见事件 $L(\omega^T) < 1$ 在 $q$ 下大约有一半的概率发生。

以这种方式选取 $q$ 被称为指数倾斜 {cite}`Bucklew_2004`。

**第 3 部分。**

```{code-cell} ipython3
from scipy.optimize import minimize_scalar

F_a, F_b = params.F_a, params.F_b

def log_ρ(ϕ):
    """f = Beta(F_a, F_b) 且 g = Beta(g_a, g_b) 时的对数切尔诺夫系数。"""
    return (betaln(ϕ * F_a + (1 - ϕ) * g_a, ϕ * F_b + (1 - ϕ) * g_b)
            - ϕ * betaln(F_a, F_b) - (1 - ϕ) * betaln(g_a, g_b))

res = minimize_scalar(log_ρ, bounds=(0, 1), method='bounded')
ϕ_star, log_ρ_star = res.x, res.fun
q_a = ϕ_star * F_a + (1 - ϕ_star) * g_a
q_b = ϕ_star * F_b + (1 - ϕ_star) * g_b
print(f"ϕ* = {ϕ_star:.4f}, C(f, g) = {-log_ρ_star:.4f}, "
      f"q = Beta({q_a:.4f}, {q_b:.4f})")

rng = np.random.default_rng(1)
N = 20_000

for T in [30, 60, 100]:
    # Standard Monte Carlo under f
    w = rng.uniform(size=(N, T))
    log_L = np.sum(log_beta_pdf(w, F_a, F_b) - log_beta_pdf(w, g_a, g_b), axis=1)
    α_mc = np.mean(log_L < 0)

    # Importance sampling under the tilted density q
    w = np.clip(rng.beta(q_a, q_b, (N, T)), 1e-300, 1 - 1e-16)
    log_L = np.sum(log_beta_pdf(w, F_a, F_b) - log_beta_pdf(w, g_a, g_b), axis=1)
    X = np.where(log_L < 0, np.exp(T * log_ρ_star + (1 - ϕ_star) * log_L), 0.0)
    α_is = np.mean(X)
    rel_se = np.std(X) / np.sqrt(N) / α_is

    print(f"T = {T:>3}: standard MC = {α_mc:.2e}, "
          f"importance sampling = {α_is:.3e} (relative s.e. {rel_se:.3f})")
```

在 $T = 30$ 时，两种估计值一致。

在 $T = 60$ 时，标准蒙特卡洛仅依赖于少数几条满足 $L(\omega^T) < 1$ 的路径，而重要性抽样的相对标准误差接近 $1$ 个百分点。

在 $T = 100$ 时，标准蒙特卡洛没有记录到任何满足 $L(\omega^T) < 1$ 的路径，而重要性抽样仍能给出相对标准误差接近 $1.5$ 个百分点的估计值。

```{solution-end}
```
