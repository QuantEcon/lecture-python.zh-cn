---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 时间序列模型中的近似误差
  headings:
    Overview: 概述
    Kullback–Leibler divergence between Gaussian processes: 高斯过程之间的库尔贝克-莱布勒散度
    Kullback–Leibler divergence between Gaussian processes::Spectral densities: 谱密度
    Kullback–Leibler divergence between Gaussian processes::Divergence rate: 散度率
    Kullback–Leibler divergence between Gaussian processes::Hansen and Sargent's approximation criterion: 汉森与萨金特的近似准则
    Kullback–Leibler divergence between Gaussian processes::Periodogram and likelihood: 周期图与似然函数
    Misspecified autoregressions: 设定有误的自回归模型
    Misspecified autoregressions::Projection onto autoregressions: 自回归模型上的投影
    Misspecified autoregressions::Seasonal example: 季节性示例
    Misspecified autoregressions::Deleting seasonal bands: 删除季节性频段
    Misspecified autoregressions::Filtering: 滤波
    Distributed lags: 分布滞后
    Distributed lags::Least squares as weighted projection: 最小二乘作为加权投影
    Distributed lags::Sampling continuous-time lags: 连续时间滞后的抽样
    Distributed lags::Seasonal noise in regressors: 回归量中的季节性噪声
    Seasonal adjustment in dynamic models: 动态模型中的季节调整
    Seasonal adjustment in dynamic models::Filtering in dynamic systems: 动态系统中的滤波
    Seasonal adjustment in dynamic models::Hansen and Sargent's experiment: 汉森与萨金特的实验
    Seasonal adjustment in dynamic models::Sims's permanent income examples: 西姆斯的永久收入示例
    Garbling and the data processing inequality: 弱化与数据处理不等式
    Garbling and the data processing inequality::Filters, frequency bands, and the data processing inequality: 滤波器、频段与数据处理不等式
    Garbling and the data processing inequality::A Blackwell chain: 布莱克韦尔链
    Garbling and the data processing inequality::Two pairs of distributions: 两对分布
    Garbling and the data processing inequality::Divergences, not estimates: 散度，而非估计值
    Garbling and the data processing inequality::Low-frequency filters: 低频滤波器
    Garbling and the data processing inequality::Sampling: 抽样
    Concluding remarks: 结语
    Exercises: 练习
---

(approximation_errors)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 时间序列模型中的近似误差

```{contents} Contents
:depth: 2
```

## 概述

当计量经济学家估计一族不包含生成数据的真实过程的模型时，在正则条件下，最大似然估计量会收敛到该族中在库尔贝克-莱布勒（KL）散度意义下最接近真实过程的模型。

在 {doc}`information_projections` 中，我们把那个模型称为真实分布在该族上的 **m-投影**。

本讲座围绕这一事实的一个推论来组织：通过选择模型、数据以及任何对数据进行变换的滤波器，计量经济学家实际上选择了估计量所计算的是哪一种投影。

对于平稳高斯过程，每个观测值的 KL 散度会收敛到一个在各频率上的积分，因此 m-投影会在不同频率上的近似误差之间进行权衡。

{cite:t}`Sims1971`、{cite:t}`Sims1974`、{cite:t}`Sims1993` 以及 {cite:t}`HansenSargent1993` 研究了计量经济学家在拟合一个设定有误的时间序列模型时所犯的误差，并说明了抽样、季节性噪声以及季节调整如何改变这些误差。

本讲座共分为五个部分。

* {ref}`ae_kl_section` 将两个平稳高斯过程之间的 KL 散度描述为在各频率上的一个积分。
* {ref}`ae_ar_section` 将设定有误的自回归模型描述为谱密度的投影，并说明对数据和模型同时滤波、仅对数据滤波、以及删除频段这几种做法如何改变该投影。
* {ref}`ae_lag_section` 将分布滞后的最小二乘估计描述为以回归量谱密度加权的 m-投影，并介绍西姆斯关于抽样和季节性噪声的结果。
* {ref}`ae_seasonal_section` 描述了动态理性预期模型中的季节调整。
* {ref}`ae_garbling_section` 将滤波、删除频段、去除季节均值以及抽样都视为 {doc}`blackwell_kihlstrom` 意义下的弱化（garbling），并利用数据处理不等式来整理前面各节的结果。

{doc}`specification_tests` 是本讲座的延续：它将设定检验描述为两个投影之间的比较。

每一节都沿用其所描述论文的记号，因此有些符号在不同节之间含义会有所变化。

具体来说，$A$ 在不同处分别表示汉森与萨金特的准则和西姆斯的移动平均算子；$c$ 分别表示滤波器、西姆斯关于季节性噪声的滞后分布以及消费；$w$ 分别表示有限傅里叶变换和西姆斯的季节性噪声；$Y$ 分别表示抽样序列和收入；$\gamma$ 分别表示自协方差和生产率参数；$d$ 分别表示滞后分布和禀赋冲击；$v$ 分别表示季节性新息和西姆斯在 $z$ 中的噪声。

{doc}`协方差平稳过程讲座 <advanced:arma>` 描述了谱密度。

{doc}`phillips_misspecified` 使用汉森与萨金特的准则 {eq}`ae_hs_criterion` 来计算那些使用设定有误的模型进行预测的主体的信念。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.linalg import toeplitz, cho_factor, cho_solve, sqrtm
from scipy.optimize import minimize
```

(ae_kl_section)=
## 高斯过程之间的库尔贝克-莱布勒散度

### 谱密度

设 $\{y_t\}$ 是一个均值为零、自协方差为 $\gamma_k = E[y_t y_{t-k}]$ 的协方差平稳过程。

它的**谱密度**为

$$
f(\omega) = \sum_{k=-\infty}^{\infty} \gamma_k e^{-i \omega k} ,
\qquad
\gamma_k = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) e^{i \omega k} \, d\omega .
$$ (ae_spectral)

我们遵循 {cite:t}`HansenSargent1993` 的约定。

{doc}`sargent_surico` 和 {doc}`phillips_drifts_volatilities` 在求和前面加了一个因子 $1/(2\pi)$；这个因子只是对 $f$ 做了重新标度，使得诸如 $f/g$ 这样的比率保持不变。

在第二个公式中令 $k = 0$ 可以看出，$f$ 把 $y_t$ 的方差 $\gamma_0$ 分布在各个频率 $\omega \in [-\pi, \pi]$ 上。

带有 $|\rho| < 1$ 以及单位方差白噪声 $\varepsilon_t$ 的一阶自回归 $y_t = \rho y_{t-1} + \sigma \varepsilon_t$ 的谱密度为 $\sigma^2 / |1 - \rho e^{-i\omega}|^2$。

更一般地，带有 $a(L) = 1 - a_1 L - \cdots - a_p L^p$ 的自回归 $a(L) y_t = \sigma \varepsilon_t$ 的谱密度为 $\sigma^2 / |a(e^{-i\omega})|^2$。

我们在 $[0, 2\pi)$ 上一个细密的频率网格上计算谱密度，在该网格上，一个函数的平均值近似等于它在一个周期上的积分乘以 $\frac{1}{2\pi}$。

```{code-cell} ipython3
n_grid = 2**14
ω = 2 * np.pi * (np.arange(n_grid) + 0.5) / n_grid
z = np.exp(-1j * ω)


def ar_poly(a):
    """a(e^{-iω}) = 1 - a_1 e^{-iω} - ... - a_p e^{-ipω} on the grid."""
    return 1 - sum(a_j * z**(j + 1) for j, a_j in enumerate(np.atleast_1d(a)))


def ar_spectrum(a, σ2=1.0):
    return σ2 / np.abs(ar_poly(a))**2


def autocov(f, n_lags):
    """γ_0, ..., γ_{n_lags} from the spectral density f on the grid."""
    return np.array([np.mean(f * np.cos(ω * k)) for k in range(n_lags + 1)])


def kl_rate(f, g, keep=None):
    """(1/4π) ∫ [f/g - 1 - log(f/g)] dω over the frequencies in keep."""
    ratio = f / g
    integrand = ratio - 1 - np.log(ratio)
    if keep is not None:
        integrand = np.where(keep, integrand, 0.0)
    return 0.5 * np.mean(integrand)
```

### 散度率

设 $f$ 和 $g$ 分别是两个均值为零的平稳高斯过程的谱密度，并假设二者都是有界的，且都远离零。

在第一个过程下，前 $T$ 个观测组成的向量 $y^T = (y_1, \ldots, y_T)$ 服从 $N(0, \Gamma_T(f))$，其中 $\Gamma_T(f)$ 是元素为 $\gamma_{j-k}$ 的 $T \times T$ 矩阵；在第二个过程下则服从 $N(0, \Gamma_T(g))$。

这两个多元正态分布之间的 KL 散度为 $\frac{1}{2}[\operatorname{tr}(\Gamma_T(g)^{-1}\Gamma_T(f)) - T - \log\det(\Gamma_T(g)^{-1}\Gamma_T(f))]$，并且它大致按比例随 $T$ 增长。

除以 $T$ 后，它收敛到

$$
D(f \| g) = \frac{1}{4\pi} \int_{-\pi}^{\pi} \left[ \frac{f(\omega)}{g(\omega)} - 1 - \log \frac{f(\omega)}{g(\omega)} \right] d\omega .
$$ (ae_kl_rate)

我们称 $D(f \| g)$ 为谱密度为 $f$ 的过程相对于谱密度为 $g$ 的过程的 **KL 散度率**。

{doc}`likelihood_ratio_process` 和 {doc}`likelihood_var` 把这个散度率记作 $h_{KL}(f, g)$，并在时域中计算它，适用于马尔可夫链以及高斯向量自回归；对于均值为零的两个一阶向量自回归，下面的向量公式 {eq}`ae_kl_rate_vector` 等于 {eq}`eq:var_kl_rate`。

*KL 散度率是各频率上两个正态分布之间 KL 散度的平均值，这两个正态分布的方差就是两个谱密度在该频率上的值。*

在频率 $\omega_j = 2\pi j / T$（$0 < \omega_j < \pi$）处的变换 $w_j = T^{-1/2} \sum_{t=1}^T y_t e^{-i \omega_j t}$，当 $T$ 较大时，近似地是方差为 $f(\omega_j)$ 的独立复正态随机变量。

因此，$T$ 个观测的两个分布之间的 KL 散度，近似等于在大约 $T/2$ 个频率上 $w_j$ 的两个分布之间散度的总和。

在每个频率上，该散度是方差分别为 $f(\omega_j)$ 和 $g(\omega_j)$、均值为零的两个正态分布之间的散度，根据 {doc}`exponential_families` 的公式 {eq}`ef_kl_normal`，对于实正态随机变量该值为 $\frac{1}{2}[f/g - 1 - \log(f/g)]$，对于复正态随机变量则为该值的两倍。

对 $T/2$ 个频率求和再除以 $T$，就得到了 {eq}`ae_kl_rate`。

我们通过计算两个一阶自回归（$\rho = 0.9$ 和 $\rho = 0.5$）的 $T$ 个观测分布之间的精确散度，来验证 {eq}`ae_kl_rate`。

```{code-cell} ipython3
def ar1_autocov(ρ, T):
    return ρ**np.arange(T) / (1 - ρ**2)


def kl_normal_vectors(Σ1, Σ2):
    """KL divergence of N(0, Σ1) from N(0, Σ2)."""
    M = cho_solve(cho_factor(Σ2), Σ1)
    return 0.5 * (np.trace(M) - len(M) - np.linalg.slogdet(M)[1])


ρ1, ρ2 = 0.9, 0.5
for T in (25, 100, 400, 1600):
    Σ1 = toeplitz(ar1_autocov(ρ1, T))
    Σ2 = toeplitz(ar1_autocov(ρ2, T))
    print(f"T = {T:5d}:  divergence / T = {kl_normal_vectors(Σ1, Σ2) / T:.5f}")
print(f"divergence rate:       {kl_rate(ar_spectrum(ρ1), ar_spectrum(ρ2)):.5f}")
```

对于具有谱密度矩阵 $F$ 和 $G$、均值为 $\nu$ 和 $\mu$ 的向量过程，相同的论证给出其对应形式：

$$
\frac{1}{4\pi} \int_{-\pi}^{\pi} \left[ \operatorname{tr}\left(G(\omega)^{-1} F(\omega)\right) - n - \log \det \left(G(\omega)^{-1} F(\omega)\right) \right] d\omega
+ \frac{1}{2} (\nu - \mu)^\top G(0)^{-1} (\nu - \mu) ,
$$ (ae_kl_rate_vector)

其中 $n$ 是该过程的维数。

均值通过频率零处的变换 $w_0 = T^{1/2} \bar y$ 进入该式，在第二个过程下其协方差矩阵近似为 $G(0)$。

{doc}`phillips_misspecified` 描述了一个均值为零的模型，该模型通过提高 $G(0)$ 来拟合非零均值，这样做会使均值项收缩。

### 汉森与萨金特的近似准则

{cite:t}`HansenSargent1993` 写道：

> 我们想要研究计量经济学家在估计一个设定有误的模型时，在经济解释上所犯的误差。众所周知，似然估计与库尔贝克-莱布勒（1951）信息准则之间存在重要联系。

他们设 $F(\omega)$ 和 $\nu$ 分别为可观测变量向量的谱密度矩阵和均值，设 $G(\omega, \delta)$ 和 $\mu(\delta)$ 分别为参数向量为 $\delta$ 的近似模型所隐含的谱密度矩阵和均值。

他们证明，当最小化元唯一时，$\delta$ 的高斯最大似然估计量几乎必然收敛到以下**近似准则**的最小化元：

$$
A(\delta) = \frac{1}{2\pi} \int_{-\pi}^{\pi} \log \det G(\omega, \delta) \, d\omega
+ \frac{1}{2\pi} \int_{-\pi}^{\pi} \operatorname{trace}\left[G(\omega, \delta)^{-1} F(\omega)\right] d\omega
+ [\nu - \mu(\delta)]^\top G(0, \delta)^{-1} [\nu - \mu(\delta)] .
$$ (ae_hs_criterion)

从 $A(\delta)$ 中减去不依赖于 $\delta$ 的量 $\frac{1}{2\pi}\int \log \det F(\omega) \, d\omega + n$，剩下的正是散度率 {eq}`ae_kl_rate_vector` 的两倍。

因此，*设定有误的最大似然估计量的概率极限，是真实过程在该模型上的 m-投影，其散度用 KL 散度率来度量。*

{cite:t}`HansenSargent1993` 在每个频率 $\omega$ 处应用了他们所称的"正态分布随机变量的常见信息不等式"：

$$
\log \det G(\omega, \delta) + \operatorname{trace}\left[G(\omega, \delta)^{-1} F(\omega)\right]
\geq \log \det F(\omega) + n .
$$

在各个频率上积分可以看出，$A(\delta)$ 的前两项至少等于 $\frac{1}{2\pi}\int \log \det F(\omega) \, d\omega + n$，当 $G(\cdot, \delta) = F$ 时取等号；均值项非负，当 $\mu(\delta) = \nu$ 时为零。

### 周期图与似然函数

{cite:t}`HanSar1980` 在 {cite:t}`Hannan1970` 的基础上，使用频域近似方法来计算 $T$ 个观测样本的高斯对数似然函数，这常被称为惠特尔似然（以 {cite:t}`Whittle1953` 命名）：

$$
- \frac{1}{2} \sum_{j=0}^{T-1} \log \det G(\omega_j, \delta) - \frac{1}{2} \sum_{j=0}^{T-1} \operatorname{tr}\left[ G(\omega_j, \delta)^{-1} I(\omega_j) \right]
$$ (ae_whittle)

再加上一个常数，其中 $\omega_j = 2\pi j / T$，$I(\omega_j) = T^{-1} \left(\sum_t y_t e^{-i\omega_j t}\right)\left(\sum_t y_t e^{i\omega_j t}\right)^\top$ 是**周期图**。

汉森和萨金特用 $S$ 表示模型的谱密度矩阵；我们沿用他们 1993 年论文中的记号 $G(\omega, \delta)$。

{doc}`kalman_filter_var` 从状态空间模型的新息出发计算精确的高斯对数似然；而 {eq}`ae_whittle` 则逐频率地近似同一个似然函数。

最大化 {eq}`ae_whittle` 等价于最小化 {eq}`ae_hs_criterion` 前两项的一个样本版本，其中用周期图代替真实谱密度 $F$。

在 {doc}`information_projections` 中，最大似然估计量是经验分布在模型上的 m-投影；在 {eq}`ae_whittle` 中，周期图取代了经验分布的位置。

周期图在任何频率处都不收敛到 $f(\omega)$：对于标量过程，其方差始终接近 $f(\omega)^2$。

$\operatorname{tr}[G(\omega_j, \delta)^{-1} I(\omega_j)]$ 在各频率上的平均值，确实收敛到 $\operatorname{tr}[G(\omega, \delta)^{-1}F(\omega)]$ 对应的平均值，而估计量收敛到 {eq}`ae_hs_criterion` 最小化元的证明只需要这些平均值。

(ae_ar_section)=
## 设定有误的自回归模型

### 自回归模型上的投影

假设计量经济学家对一个谱密度为 $f$ 的标量过程拟合自回归 $a(L) y_t = \sigma \varepsilon_t$，因此模型的谱密度为 $g(\omega) = \sigma^2 / |a(e^{-i\omega})|^2$。

对于固定的多项式 $a$，使散度率 {eq}`ae_kl_rate` 最小化的方差 $\sigma^2$ 为

$$
\sigma^2(a) = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) |a(e^{-i\omega})|^2 \, d\omega = E\left[ \left( a(L) y_t \right)^2 \right] ,
$$

即预测法则 $\hat y_t = a_1 y_{t-1} + \cdots + a_p y_{t-p}$ 的均方一步向前预测误差。

当 $a$ 的根位于单位圆之外时，柯尔莫哥洛夫公式给出 $\frac{1}{2\pi}\int \log |a(e^{-i\omega})|^2 \, d\omega = 0$。

将 $\sigma^2(a)$ 代入 {eq}`ae_kl_rate` 便得到

$$
D(f \| g) = \frac{1}{2} \left[ \log \sigma^2(a) - \frac{1}{2\pi} \int_{-\pi}^{\pi} \log f(\omega) \, d\omega \right] .
$$

根据柯尔莫哥洛夫公式，$\frac{1}{2\pi}\int_{-\pi}^{\pi} \log f(\omega) \, d\omega = \log \sigma^2_\infty$，其中 $\sigma^2_\infty$ 是给定 $y_t$ 的无穷过去条件下其一步向前预测误差的方差，因此 $D(f \| g) = \frac{1}{2} \log [\sigma^2(a)/\sigma^2_\infty]$。

对于向量过程，{doc}`kalman_filter_var` 中的谱分解恒等式 {eq}`eq:sf_identity` 给出对应的结果：状态空间模型可观测变量谱密度矩阵的对数行列式在各频率上的平均值，等于其新息协方差矩阵的对数行列式，在该讲座记号中即为 $G\Sigma G^\top + R$。

第二项不依赖于 $a$，因此 *一个平稳过程在 $p$ 阶自回归上的 m-投影，选取的是使均方一步向前预测误差最小的系数。*

这些系数满足尤尔-沃克方程 $\sum_{k=1}^p a_k \gamma_{j-k} = \gamma_j$（$j = 1, \ldots, p$）；对于 $p = 1$，$a_1 = \gamma_1 / \gamma_0$。

在频域似然 {eq}`ae_whittle` 中，$p$ 阶自回归构成一个指数族，其充分统计量是滞后为 $0, \ldots, p$ 的自协方差。

与 {doc}`information_projections` 中一样，m-投影匹配各阶矩：其滞后为 $0, \ldots, p$ 的自协方差与 $y_t$ 的相同。

同样的论证适用于任何移动平均算子为首项系数为一、根位于单位圆外的多项式之比的模型；当均值非零时，集中方差还要加上预测误差均值的平方。

{doc}`phillips_misspecified` 利用这一点来计算模型设定有误的主体的预测法则。

下面的函数通过直接最小化散度率来计算 m-投影。

其参数 `keep` 将准则限制在一部分频率上。

将 $a$ 的某个根替换为其倒数，会使谱密度的形状保持不变，因此该准则在平稳区域之外也存在最小化元。

我们通过用 $(-1, 1)$ 内的偏自相关系数来参数化，将搜索限制在平稳自回归范围内。

```{code-cell} ipython3
def yule_walker(γ, p):
    return np.linalg.solve(toeplitz(γ[:p]), γ[1:p + 1])


def pacf_to_ar(pacf):
    """AR coefficients from partial autocorrelations (Durbin–Levinson)."""
    a = np.zeros(0)
    for φ_kk in pacf:
        a = np.append(a - φ_kk * a[::-1], φ_kk)
    return a


def ar_to_pacf(a):
    a, pacf = np.array(a, dtype=float), []
    while len(a):
        φ_kk = a[-1]
        pacf.append(φ_kk)
        a = (a[:-1] + φ_kk * a[:-1][::-1]) / (1 - φ_kk**2)
    return np.array(pacf[::-1])


def ar_projection(f, p, keep=None):
    """Stationary AR(p) coefficients and innovation variance that minimize
    the divergence rate of f from the AR model, over the frequencies in keep."""
    keep = np.ones(n_grid, bool) if keep is None else keep

    def criterion(θ):
        a2 = np.abs(ar_poly(pacf_to_ar(np.tanh(θ)))[keep])**2
        return np.log(np.mean(f[keep] * a2)) - np.mean(np.log(a2))

    θ0 = np.arctanh(ar_to_pacf(yule_walker(autocov(f, p), p)))
    θ = minimize(criterion, θ0, method="BFGS", options={"gtol": 1e-8}).x
    a = pacf_to_ar(np.tanh(θ))
    σ2 = np.mean((f * np.abs(ar_poly(a))**2)[keep])
    return a, σ2
```

### 季节性示例

设观测到的季度序列是一个非季节性成分与一个相互独立的季节性成分之和 $y_t = x_t + s_t$：

$$
x_t = 0.9 \, x_{t-1} + e_t ,
\qquad
(1 + \varphi L + \varphi^2 L^2 + \varphi^3 L^3) \, s_t = v_t ,
\qquad \varphi = 0.98 .
$$

季节性多项式的根恰好位于季节频率 $\pi/2$ 和 $\pi$ 附近的单位圆外侧，因此 $s_t$ 的谱密度在这些频率处有窄峰。

我们对 $v_t$ 进行标度，使得 $s_t$ 的方差为 $x_t$ 方差的四分之一。

一位计量经济学家对 $y_t$ 拟合一阶自回归模型。

```{code-cell} ipython3
ρ_x, φ = 0.9, 0.98
f_x = ar_spectrum(ρ_x)
f_s = ar_spectrum([-φ, -φ**2, -φ**3])
f_s = f_s * 0.25 * np.mean(f_x) / np.mean(f_s)
f_y = f_x + f_s

a_full, σ2_full = ar_projection(f_y, 1)
g_full = ar_spectrum(a_full, σ2_full)
γ_y = autocov(f_y, 1)
print(f"m-projection: ρ* = {a_full[0]:.4f},  γ_1/γ_0 = {γ_y[1] / γ_y[0]:.4f}")
print(f"divergence rate at the projection: {kl_rate(f_y, g_full):.4f}")
print(f"g/f at frequency zero: {g_full[0] / f_y[0]:.2f}")
```

m-投影得到 $\rho^* = 0.62$，明显低于非季节性成分的系数 $0.9$。

图的左侧面板绘制了 $y_t$、$x_t$ 以及投影的谱密度的对数。

右侧面板绘制了每个频率处的 $\frac{1}{2}[f/g - 1 - \log(f/g)]$；其在各频率上的平均值即为散度率。

在右侧面板中，竖直线标出了季节频率 $\pi/2$ 和 $\pi$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: AR(1) projection of a seasonal series
    name: fig-ae-seasonal-ar1
---
half = ω <= np.pi
π_ticks = (np.arange(5) * np.pi / 4,
           ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(ω[half], np.log(f_y[half]), lw=2, label="$y_t$")
ax.plot(ω[half], np.log(f_x[half]), "--", label="$x_t$")
ax.plot(ω[half], np.log(g_full[half]), lw=2,
        label=rf"AR(1) projection, $\rho^*$ = {a_full[0]:.2f}")
ax.set_xticks(*π_ticks)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("log spectral density")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9,
          frameon=False)

ax = axes[1]
ratio = f_y / g_full
ax.semilogy(ω[half], 0.5 * (ratio - 1 - np.log(ratio))[half], lw=2)
ax.set_ylim(1e-3, 30)
ax.set_xticks(*π_ticks)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"KL divergence at frequency $\omega$")
for c in (np.pi / 2, np.pi):
    ax.axvline(c, color="0.7", lw=0.8)
plt.tight_layout()
plt.show()
```

一阶自回归无法在 $\pi/2$ 处产生峰值，而且只有在系数接近 $-1$ 时才能在 $\pi$ 处产生一个窄峰，但这样会放弃频率零附近的功率。

因此，散度在这两个频率处最大。

对于给定的 $\sigma^2$，降低 $\rho$ 会提高模型在高频处的谱密度，同时降低其在频率零附近的谱密度。

第一个效应减少了 $\pi/2$ 和 $\pi$ 处的散度；第二个效应提高了频率零附近的散度，而那里恰恰是 $x_t$ 功率最集中的地方。

令 $r = f/g$，被积函数 $\frac{1}{2}(r - 1 - \log r)$ 关于 $r$ 线性增长，而关于 $1/r$ 仅对数增长，因此在季节性峰值处低估谱密度的代价，要高于在其他地方高估谱密度的代价；{doc}`likelihood_var` 对冲击方差也作出了同样的观察。

在 $\rho^* = 0.62$ 处，投影在频率零处的谱密度大约是 $y_t$ 谱密度的 28%。

### 删除季节性频段

{cite:t}`HansenSargent1993` 描述了他们所称的理想季节调整。

设 $K$ 为季节频率周围各频段之外的频率集合，令 $1_K(\omega)$ 在 $\omega \in K$ 时等于一，否则为零；汉森和萨金特用 $C(\omega)$ 表示这个指示函数。

他们将 {eq}`ae_hs_criterion` 替换为

$$
A(\delta) = \frac{1}{2\pi} \int_{-\pi}^{\pi} 1_K(\omega) \left\{ \log \det G(\omega, \delta) + \operatorname{trace}\left[ G(\omega, \delta)^{-1} F(\omega) \right] \right\} d\omega
+ [\nu - \mu(\delta)]^\top G(0, \delta)^{-1} [\nu - \mu(\delta)] .
$$ (ae_hs_ideal)

在删除季节性频段之后，计量经济学家将该过程在频段之外部分投影到模型在频段之外的部分。

限制在 $K$ 上的散度率至多等于完整的散度率，因为信息不等式使得每个被删除项都是非负的；{ref}`ae_garbling_section` 将这一不等式理解为数据处理不等式的一个实例。

如果对于某个 $\delta_0$，该模型在季节性频段之外与 $F$ 相匹配，且均值也匹配，即 $\mu(\delta_0) = \nu$，那么无论模型在频段内表现如何，$\delta_0$ 都是 {eq}`ae_hs_ideal` 的最小化元之一。

汉森和萨金特补充说，省略频率可能会通过引入额外的最小化元而妨碍识别；而且，当模型仍可识别时，省略频率通常会增大估计量的渐近方差，除非季节性频率受到了测量误差的污染。

当模型在所有频率上都设定正确时，使用所有频率的估计量是有效的，根据豪斯曼引理，渐近方差的增加量等于两个估计量之差的方差。

{doc}`specification_tests` 把这个差异转化为一个检验。

在我们的示例中，没有任何一阶自回归能在频段之外匹配 $y_t$ 的谱密度，因为季节性成分在每个频率上都有一定的功率。

但季节性成分约 94% 的方差落在季节频率周围宽度为 $\pi/8$ 的频段内。

```{code-cell} ipython3
def seasonal_keep(Δ):
    """True outside bands of width Δ centered on π/2, π, and 3π/2."""
    def dist(c):
        return np.minimum(np.abs(ω - c), 2 * np.pi - np.abs(ω - c))
    inside = ((dist(np.pi / 2) < Δ / 2) | (dist(np.pi) < Δ / 2)
              | (dist(3 * np.pi / 2) < Δ / 2))
    return ~inside


keep_8 = seasonal_keep(np.pi / 8)
print(f"share of the seasonal variance in bands of width π/8: "
      f"{1 - np.mean(f_s * keep_8) / np.mean(f_s):.2f}")
print(f"divergence rate of the AR(1) projection: all frequencies "
      f"{kl_rate(f_y, g_full):.4f}, outside the bands "
      f"{kl_rate(f_y, g_full, keep_8):.4f}")
for Δ in (np.pi / 16, np.pi / 8, np.pi / 4):
    a_K, _ = ar_projection(f_y, 1, seasonal_keep(Δ))
    print(f"band width {Δ / np.pi:.4f}π:  ρ* = {a_K[0]:.4f}")
```

删除宽度为 $\pi/8$ 的频段，将 $\rho^*$ 从 $0.62$ 提高到 $0.87$。

$0.87$ 与 $0.9$ 之间的剩余差距，来自落在被删除频段之外的季节性功率，并且随着频段加宽而逐渐缩小。

### 滤波

设 $c(L)$ 为一个频率响应 $c(e^{-i\omega})$ 永不为零的滤波器。

用 $c(L)$ 对数据滤波，会使 $f$ 乘以 $|c(e^{-i\omega})|^2$；用 $c(L)$ 对模型滤波，则使 $g$ 乘以相同的因子。

每个频率处 $f/g$ 的比值保持不变，因此用同一个可逆滤波器对数据和模型同时滤波，散度率 {eq}`ae_kl_rate` 和投影都不会改变。

另外还有两种操作会改变投影。

* **删除频率**，如 {eq}`ae_hs_ideal` 所示，将频段从准则中移除，而保留其他频率不变。在回归中，用一个在频段内为零、在其他地方为一的滤波器对两个变量同时滤波，等价于删除这些频段。该滤波器不可逆，因此不变性论证不适用。
* **仅对数据滤波而不对模型滤波**，会将 $f$ 替换为 $|c(e^{-i\omega})|^2 f$，而 $g$ 保持不变，因此计量经济学家实际上是在把一个不同的过程投影到同一个模型上。

{ref}`ae_garbling_section` 表明，不变性与删除频率的效应分别对应数据处理不等式的等号情形与不等号情形，而数据处理不等式对仅滤波数据的情形并无说明。

下面的代码验证了这种不变性，并计算了仅用 $c(L) = 1 - 0.5L$ 对数据滤波所产生的效应。

```{code-cell} ipython3
c2 = np.abs(1 - 0.5 * z)**2
print(f"D(f || g) = {kl_rate(f_y, g_full):.6f},  "
      f"D(|c|²f || |c|²g) = {kl_rate(c2 * f_y, c2 * g_full):.6f}")
a_c, _ = ar_projection(c2 * f_y, 1)
print(f"AR(1) projection of the data filtered by 1 - 0.5L: ρ* = {a_c[0]:.4f}")
print(f"share of the filtered variance from the seasonal component: "
      f"{np.mean(c2 * f_s) / np.mean(c2 * f_y):.2f}")
```

该滤波器将 $f$ 乘以 $|1 - 0.5 e^{-i\omega}|^2 = 1.25 - \cos\omega$。

它将频率零处（即 $x_t$ 功率最集中处）的谱密度削减为原来的四分之一，同时将季节性峰值 $\pi$ 处的谱密度乘以 $2.25$。

季节性成分的一阶自相关系数为负，此后它占滤波后序列方差的一半以上，滤波后序列的 AR(1) 投影得到 $\rho^* = -0.08$。

(ae_lag_section)=
## 分布滞后

### 最小二乘作为加权投影

设

$$
y_t = \sum_{s=-\infty}^{\infty} b_s x_{t-s} + u_t ,
$$ (ae_lag)

其中 $\{x_t\}$ 和 $\{u_t\}$ 都是平稳的，且对每个 $s$，$u_t$ 都与 $x_{t-s}$ 不相关。

记 $\tilde b(\omega) = \sum_s b_s e^{-i\omega s}$ 为滞后分布的傅里叶变换，$S_x$ 为 $x$ 的谱密度。

计量经济学家通过最小二乘法来估计滞后分布，并受限于诸如有限长度或参数形式等约束。

在大样本下，受约束的估计量 $\hat b$ 收敛到使以下式子

$$
E\left[ \left( y_t - \sum_s d_s x_{t-s} \right)^2 \right]
= \frac{1}{2\pi} \int_{-\pi}^{\pi} S_x(\omega) \left| \tilde b(\omega) - \tilde d(\omega) \right|^2 d\omega + E[u_t^2]
$$ (ae_sims_criterion)

在满足约束的滞后分布中取最小值的滞后分布 $d$。

最小二乘法是以下高斯模型的最大似然估计量：在给定 $x$ 的路径下，$y_t$ 的均值为 $\sum_s d_s x_{t-s}$，偏差序列无关且方差为 $\sigma^2$。

在 $\sigma^2$ 上最小化真实条件分布相对于该模型的 KL 散度率，剩下的就是均方误差 {eq}`ae_sims_criterion` 的对数的一半，再加上一个不依赖于滞后分布 $d$ 的项。

因此，受约束的最小二乘估计量收敛到真实条件分布在受约束模型上的 m-投影。

谱密度 $S_x$ 对投影进行加权：$\tilde d$ 中的误差在 $x$ 功率最集中的频率处代价最大。

### 连续时间滞后的抽样

{cite:t}`Sims1971` 研究了一个在整数日期被观测的连续时间分布滞后 $y(t) = \int b(s) x(t - s) \, ds + u(t)$。

他使用傅里叶变换 $\tilde b(\omega) = \int e^{-i\omega t} b(t) \, dt$ 和连续时间过程 $x$ 的谱密度 $S_x$，二者均定义在整条实数轴上。

在整数日期抽样会使频率 $\omega, \omega \pm 2\pi, \omega \pm 4\pi, \ldots$ 无法区分，这一现象被称为混叠。

对抽样后的 $Y_t = y(t)$ 关于抽样后的 $X_t = x(t)$ 的当期、过去和未来值做最小二乘回归，得到的离散时间滞后分布 $B$ 的傅里叶变换为

$$
\tilde B(\omega) = \frac{\sum_{k=-\infty}^{\infty} S_x(\omega + 2\pi k) \, \tilde b(\omega + 2\pi k)}{\sum_{k=-\infty}^{\infty} S_x(\omega + 2\pi k)} .
$$ (ae_folding)

用西姆斯的话说，"$\tilde B(\omega)$ 是 $\tilde b(\omega + 2k\pi)$ 的加权平均"。

根据前一小节的论证，$B$ 是抽样后 $Y$ 的真实条件分布在离散时间分布滞后模型上的 m-投影，而 {eq}`ae_folding` 中的权重正是抽样折叠到频率 $\omega$ 上的 $S_x$ 的各个值。

一位估计出 $B$ 并把它解读为连续时间滞后分布 $b$ 的计量经济学家，就会产生近似误差 $B - b$。

西姆斯的表 I 给出了 $b(t) = e^{-0.5t}$（$t \geq 0$）对应的 $B$。

其第 (1) 列是 $b(t)$，第 (2) 至 (5) 列对应关于 $x$ 的四种假设，我们重现其中三种：

* 第 (2) 列：$x$ 的自协方差函数在整数日期之间是线性的，这使得 $B(t) = \int b(s) \max(1 - |t - s|, 0) \, ds$；
* 第 (3) 列：$x$ 的自协方差函数为 $e^{-0.5|t|}$，谱密度为 $S_x(\omega) = 2 r / (r^2 + \omega^2)$，其中 $r = 0.5$；
* 第 (5) 列：$S_x$ 在 $(-\pi, \pi)$ 上为正，在其外为零，这使得 $\tilde B(\omega) = \tilde b(\omega)$ 在 $(-\pi, \pi)$ 上成立。

在第 (4) 列中，西姆斯在抽样之前先将第 (3) 列的过程在每个单位区间内取平均。

```{code-cell} ipython3
def sampled_lag(b_ft, S_x, t_vals, K=400, n=512):
    """Discrete lag B(t) of formula (ae_folding), by summing over aliases."""
    w = -np.pi + 2 * np.pi * (np.arange(n) + 0.5) / n
    W = w[None, :] + 2 * np.pi * np.arange(-K, K + 1)[:, None]
    B_ft = (S_x(W) * b_ft(W)).sum(axis=0) / S_x(W).sum(axis=0)
    return np.array([np.mean(B_ft * np.exp(1j * w * t)).real for t in t_vals])


def band_limited_lag(b_ft, t_vals, n=512):
    w = -np.pi + 2 * np.pi * (np.arange(n) + 0.5) / n
    return np.array([np.mean(b_ft(w) * np.exp(1j * w * t)).real
                     for t in t_vals])


def triangle_lag(b, t_vals):
    return np.array([integrate.quad(lambda s: b(s) * (1 - abs(t - s)),
                                    max(t - 1, 0), max(t + 1, 0))[0]
                     for t in t_vals])


def print_lag_table(t_vals, *columns):
    print("   t   b(t)   (2)    (3)    (5)")
    for t, *row in zip(t_vals, *columns):
        print(f"{t:4d} " + " ".join(f"{np.round(v, 2) + 0.0:6.2f}" for v in row))


t_vals = np.arange(-4, 9)
b_exp = lambda t: np.where(t >= 0, np.exp(-0.5 * np.maximum(t, 0)), 0.0)
b_exp_ft = lambda w: 1 / (0.5 + 1j * w)
S_ou = lambda w: 2 * 0.5 / (0.5**2 + w**2)

col2 = triangle_lag(b_exp, t_vals)
col3 = sampled_lag(b_exp_ft, S_ou, t_vals)
col5 = band_limited_lag(b_exp_ft, t_vals)
print_lag_table(t_vals, b_exp(t_vals), col2, col3, col5)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Continuous-time lag and its sampled projections
    name: fig-ae-sampled-lag
---
t_fine = np.linspace(-4, 8, 600)
fig, ax = plt.subplots()
ax.plot(t_fine, b_exp(t_fine), "k", lw=1.5, label="continuous-time $b(t)$")
ax.plot(t_vals, col3, "o-", label=r"$B(t)$, autocovariance $e^{-0.5|t|}$")
ax.plot(t_vals, col5, "s--", label=r"$B(t)$, $S_x$ zero outside $(-\pi, \pi)$")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("$t$")
ax.set_ylabel("lag coefficient")
ax.legend()
plt.show()
```

滞后零处的系数 $B(0)$ 在 $0.42$ 到 $0.45$ 之间，而 $b(0) = 1$，这是因为抽样后的回归把 $b$ 在邻近日期上的一个平均值归到了日期零上。

当 $x$ 在频率 $\pi$ 以上没有功率时，单侧滞后 $b$ 会投影到一个双侧离散滞后上，其在负 $t$ 处的系数呈交替符号。

如果一位计量经济学家检验离散时间滞后分布是否为单侧，或者把 $B(0)$ 解读为瞬时响应，那么即便从一个正确估计的 $B$ 出发，也会对 $b$ 得出错误的结论。

{ref}`ae_ex_sampling` 推导了这种符号模式。

### 回归量中的季节性噪声

{cite:t}`Sims1974` 研究了当两个变量都含有季节性噪声时的回归 {eq}`ae_lag`。

他设 $y = x * b + u$ 和 $z = w * c + v$，其中 $*$ 表示卷积，$v$ 与 $w$ 不相关，季节性噪声 $z$ 和 $w$ 均与 $x$ 和 $u$ 不相关，并且计量经济学家观测到的是 $y' = y + z$ 和 $x' = x + w$。

对 $y'$ 关于 $x'$ 所有超前项和滞后项做最小二乘回归，得到的无约束滞后分布 $b'$ 满足

$$
\tilde b' = \tilde b \, \frac{S_x}{S_x + S_w} + \tilde c \, \frac{S_w}{S_x + S_w} ,
$$ (ae_sims_bias)

即 $\tilde b$ 和 $\tilde c$ 的加权平均，其权重随频率而变化。

当 $y$ 中的噪声与 $x$ 中的噪声无关，即 $c = 0$ 时，$\tilde b'$ 除在季节频率处之外都接近 $\tilde b$；在季节频率处，$S_w$ 很大，$\tilde b'$ 会趋向于零。

{cite:t}`Sims1974` 在其方程 (2.7) 中陈述了受约束最小二乘的大样本性质，并归功于 {cite:t}`Sims1972JASA`：当我们"受先验约束对滞后分布做最小二乘估计时，估计量 $\hat b$ 会在约束下最小化以下量"：

$$
\int_{-\pi}^{\pi} S_{x'}(\omega) \left| \tilde{\hat b}(\omega) - \tilde b'(\omega) \right|^2 d\omega ,
$$

这除了因子 $1/2\pi$ 和一个可加常数之外，正是 {eq}`ae_sims_criterion` 中用 $S_{x'} = S_x + S_w$ 代替 $S_x$，用 $\tilde b'$ 代替 $\tilde b$，用 $\hat b$ 代替 $d$ 的结果。

该准则对在季节频率处拟合 $\tilde b'$ 赋予了很大的权重，而 $\tilde b'$ 恰恰在这些频率处偏差最大。

西姆斯由此得出结论：

> 由于在准则函数 (2.7) 中，季节频率处的 $|\tilde{\hat b} - \tilde b'|^2$ 被赋予了特别强的权重，那些阻止 $\tilde{\hat b}$ 在季节频率处出现*尖锐*峰值或凹陷的约束，只会导致估计值在季节频率处出现*宽阔*的峰值或凹陷，即以在远离季节频率处产生相对更多误差为代价，换取在季节频率附近非常精确的拟合。

西姆斯考虑了两种真实滞后分布：恒等滞后（$b_0 = 1$，$b_s = 0$ 对 $s \neq 0$）以及科伊克滞后 $b_s = 0.9^s$（$s \geq 0$）。

他将 $x$ 的自协方差函数设为 $0.9^{|t|}$。

季节性噪声 $w$ 是白噪声的一个移动平均，其权重由西姆斯在脚注 6 中给出。

这些权重是以下两个滤波器的卷积：一个在相隔四个季度的五个值上取平均的滤波器，以及一个从每个值中减去周围十六个季度平均值的滤波器。

西姆斯将 $w$ 标度为具有 $x$ 方差四分之一的方差。

该噪声在季节频率处有峰值，而在频率零处没有功率。

```{code-cell} ipython3
# moving average weights of w, lags -16, ..., 16 (Sims 1974, footnote 6)
h1 = np.zeros(17)
h1[8 + 4 * np.arange(-2, 3)] = 1 / 5
h2 = -np.ones(17) / 16
h2[8] = 1.0
h_w = np.convolve(h1, h2)
h_w_ft = sum(h * np.exp(-1j * ω * k) for k, h in zip(range(-16, 17), h_w))

S_x74 = (1 - 0.9**2) * ar_spectrum(0.9)          # unit variance
S_w74 = np.abs(h_w_ft)**2
S_w74 = S_w74 * 0.25 / np.mean(S_w74)            # one-fourth the variance of x
S_xp74 = S_x74 + S_w74


def lag_from_ft(B_ft, t_vals):
    return np.array([np.mean(B_ft * np.exp(1j * ω * t)).real for t in t_vals])


t74 = np.arange(-8, 13)
b_identity_ft = np.ones(n_grid)
b_koyck_ft = 1 / (1 - 0.9 * z)
bp_identity = lag_from_ft(b_identity_ft * S_x74 / S_xp74, t74)
bp_koyck = lag_from_ft(b_koyck_ft * S_x74 / S_xp74, t74)
print(f"unconstrained coefficient at lag zero, identity lag: "
      f"{bp_identity[t74 == 0][0]:.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Seasonal noise and estimated lag distributions
    name: fig-ae-sims-1974
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(ω[half], np.log(S_x74[half]), label="$S_x$")
ax.plot(ω[half], np.log(S_xp74[half]), label="$S_{x'} = S_x + S_w$")
ax.set_xticks(*π_ticks)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("log spectral density")
ax.legend()

ax = axes[1]
ax.plot([0], [1.0], "kx", ms=8, label="true identity lag at $t = 0$")
ax.plot(t74, bp_identity, "o-", label="estimated, identity lag")
koyck_t = t74[t74 >= 0]
ax.plot(koyck_t, 0.9**koyck_t, "k+", ms=10, label="true Koyck lag $0.9^t$")
ax.plot(t74, bp_koyck, "s-", label="estimated, Koyck lag")
ax.set_xticks(np.arange(-8, 13, 4))
ax.set_xlabel("$t$")
ax.set_ylabel("lag coefficient")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9,
          frameon=False)
plt.tight_layout()
plt.show()
```

季节性噪声不会使系数之和产生偏差，因为该噪声在频率零处没有功率。

它将恒等滞后在滞后零处的系数降低到大约 $0.63$，并且会给两个滞后分布都添加季节性波动，包括对 $x'$ 未来值的系数。

{doc}`sargent_surico` 采用了同样的论证：一个在频率零处没有功率的成分不会改变滞后系数之和 $h(0) = S_{yz}(0)/S_z(0)$。

下面的函数在频域中求解对 $x'$ 的一组滞后值做 $y'$ 回归的大样本正规方程。

其可选参数 `weight` 将 $x'$ 的谱密度与 $y'$ 和 $x'$ 的互谱乘以 $|\tilde k(\omega)|^2$，这等价于对 $y'$ 和 $x'$ 都用滤波器 $k$ 滤波的效果；{ref}`ae_ex_filter` 用到了这个参数。

```{code-cell} ipython3
def constrained_lag(b_ft, lags, weight=None):
    """Large-sample least squares coefficients on x'(t - j), j in lags,
    for the true lag with Fourier transform b_ft and c = 0."""
    wt = np.ones(n_grid) if weight is None else weight
    M = np.array([[np.mean(wt * S_xp74 * np.cos(ω * (j - k))) for k in lags]
                  for j in lags])
    m = np.array([np.mean(wt * S_x74 * b_ft * np.exp(1j * ω * j)).real
                  for j in lags])
    return np.linalg.solve(M, m)


print("identity lag, x'(t) only:         ",
      np.round(constrained_lag(b_identity_ft, [0]), 3))
print("identity lag, x'(t) and x'(t - 1):",
      np.round(constrained_lag(b_identity_ft, [0, 1]), 3))
```

仅使用 $x'(t)$ 作为回归量时，该系数趋向于 $0.80$，即经典变量误差模型中的衰减因子 $1/(1 + 1/4)$。

该系数的偏差小于滞后零处的无约束系数 $0.63$，但无约束回归能无偏估计出的系数之和，则从 $1$ 降为 $0.80$。

加入 $x'(t-1)$ 后，系数趋向于 $0.58$ 和 $0.34$，尽管真实的滞后分布将全部权重都放在日期零上。

{cite:t}`Sims1974` 报告的是 $0.80$，以及 $0.58$ 和 $0.31$。

我们按照他的脚注 6 和公式 (2.6) 所做的计算，第二个系数得到 $0.34$，而我们尝试过的其他变体给出的结果也都在 $0.33$ 到 $0.34$ 之间。

他写道：

> 如果这些是对某个短期劳动力需求函数的估计，那么支持"短期规模报酬递增"的证据看起来会非常强；而由于把滞后分布保持得很短，季节性偏误存在的证据就被掩盖了。

他提出的解决办法是用同一个滤波器 $k$（他写作 $a$）对两个变量同时滤波，该滤波器能去除季节频率附近的功率。

对两个变量同时滤波，会使无约束滞后分布 $b'$ 保持不变，而在受约束准则中，权重 $S_{x'}$ 会被替换为 $|\tilde k|^2 S_{x'}$。

此后，受约束最小二乘会在非季节性频率处拟合 $\tilde b'$，而在那里 $\tilde b'$ 接近 $\tilde b$。

{doc}`sargent_surico` 描述了相反的滤波器选择。

卢卡斯采用参数为 $\beta$ 的双侧移动平均，使权重 $|\tilde k|^2 S_x$ 集中在频率零附近。

用单一系数对一个滤波后序列回归另一个滤波后序列，会收敛到以 $|\tilde k|^2 S_x$ 加权的 $\tilde b$ 的平均值，当 $\beta \to 1$ 时，该平均值趋近于滞后系数之和 $\tilde b(0)$。

(ae_seasonal_section)=
## 动态模型中的季节调整

### 动态系统中的滤波

{cite:t}`Sims1993` 指出，一个完整的动态系统不具备 {ref}`ae_lag_section` 中回归所具有的那种不变性。

如果 $y(t) = A * \varepsilon(t)$ 是一个向量过程的移动平均表示，而计量经济学家使用滤波后的数据 $c * y$，那么移动平均算子就变为 $c * A$：

> 因此，与回归情形中的结果（滤波不影响 $b$）相反，对于一个完整的动态系统，滤波会改变基本的估计参数 $A$。

在回归中，计量经济学家只对给定 $x$ 的 $y$ 的条件分布建模。

用同一滤波器对 $y$ 和 $x$ 同时滤波，会将互谱和 $x$ 的谱都乘以 $|\tilde c|^2$，因此 $\tilde b = S_{yx}/S_x$ 不会改变。

而在动态系统中，计量经济学家对所有变量的联合分布建模，其谱密度矩阵为 $\tilde A \tilde A^*$，滤波会将 $\tilde A$ 替换为 $\tilde c \tilde A$。

理性预期模型的跨方程约束，将主体决策法则的系数与主体所预测过程的参数联系起来，从而把偏好和技术的参数与整个算子 $A$ 联系起来。

因此，对滤波后的数据拟合一个未滤波的模型，会使得这些参数的估计值发生偏移。

如 {ref}`ae_ex_invariance` 所示，如果同时对模型进行滤波，并且滤波器的频率响应只在孤立频率处为零，那么就能恢复 {ref}`ae_ar_section` 所描述的那种不变性。

西姆斯笔下的计量经济学家则是把滤波后的数据当作未经调整的数据来处理。

{cite:t}`HansenSargent1993` 用他们的准则 {eq}`ae_hs_criterion` 比较了 {ref}`ae_ar_section` 中的几种操作。

在他们归之于西姆斯的表述方式中，计量经济学家对数据滤波但不对模型滤波，因此滤波后数据的谱密度 $F^a(\omega) = c(e^{-i\omega}) F(\omega) c(e^{i\omega})^\top$ 取代了 $F(\omega)$，而 $G(\omega, \delta)$ 保持原样。

在理想季节调整 {eq}`ae_hs_ideal` 中，计量经济学家删除了季节性频段。

汉森和萨金特用 {eq}`ae_hs_criterion` 的第二项来阐述西姆斯支持使用调整后数据的论证：

> 论证是，项 $A_2(\delta) = (1/2\pi)\int_{-\pi}^{\pi} \operatorname{trace}[G(\omega, \delta)^{-1} F^a(\omega)] \, d\omega$ 对 $F^a(\omega)$（若使用未调整数据，则为 $F(\omega)$）较大的那些频率赋予了最大的权重。由于 $F(\omega)$ 恰好在 $G(\omega, \delta)$ 设定误差最严重的那些频率处较大，用 $F^a(\omega)$ 而非 $F(\omega)$ 加权，可以减轻设定误差对偏好和技术参数估计的影响。

他们在脚注中补充说，季节调整有效率上的代价：理性预期模型跨方程、跨频率施加的过度识别约束，在被调整所抑制的高功率季节频率处信息量最大。

{cite:t}`Sims1993` 描述了对滤波后数据拟合自回归模型的极限情形。

极限系数使模型的一步向前预测滤波器与滤波后数据的预测滤波器之间的二次距离最小化，权重为滤波后数据的谱密度。

假设该参数化方式能够在季节性频段之外很好地匹配滤波后数据的预测滤波器，但无法在短至频段宽度的区间内急剧变化，且这些频段较窄。

那么，随着调整将滤波后数据在频段内的谱密度推向零，拟合最优的滤波器会在频段之外拟合良好，而在频段内进行插值。

西姆斯证明，用一个在频段内任意设定的滤波器去拟合在该处功率接近零的数据，会使一步向前预测误差膨胀"一加上季节性频段总宽度与 $2\pi$ 之比"的倍数。

### 汉森与萨金特的实验

{cite:t}`HansenSargent1993` 研究了一个线性二次版本的随机增长模型，其中计划者选择消费、投资和资本。

在真实模型中，生产率参数 $\gamma_{s(t)}$ 随季度 $s(t)$ 周期性变化，取值为 $0.13$、$0.1$、$0.1$ 和 $0.08$；调整成本参数为 $\phi_1 = 0.3$；禀赋冲击 $d_t$ 服从系数为 $0.4$ 的一阶自回归。

计量经济学家拟合一个除了具有恒定生产率参数 $\gamma$ 这一点之外、在其他各方面都正确的模型。

计量经济学家固定所有其他参数的真实值，估计 $\gamma$、$\phi_1$，以及滞后为 $1$、$4$、$5$ 期的禀赋冲击自回归系数 $a_1$、$a_4$、$a_5$。

汉森和萨金特将 {eq}`ae_hs_criterion` 应用于把每年四个观测值堆叠起来的过程，因此均值 $\nu$ 包含四个季节均值。

一个具有恒定 $\gamma$ 的模型隐含每个季度都有相同的均值。

他们的方程 (4.11) 将均值项写成在频率 $0$、$\pi/2$、$\pi$ 和 $3\pi/2$ 处季节均值误差的傅里叶分量之和，每一项都除以模型在该频率处 $y_t$ 的谱密度。

汉森和萨金特比较了三种数据：

* 未调整数据，其准则为 {eq}`ae_hs_criterion`；
* 计量经济学家通过对季节虚拟变量做回归去除了季节均值后的数据，其准则去掉了 {eq}`ae_hs_criterion` 的均值项；以及
* 理想调整后的数据，其准则从前两项中删除了季节性频段，并去掉了均值项。

他们的表 1 报告了三种准则的最小化元：

| 参数 | 季节未调整 | 已去除季节均值 | 季节已调整 | 真值 |
|---|---|---|---|---|
| $\gamma$ | 0.0996 | 0.1030 | 0.1025 | $\gamma_{s(t)}$ |
| $\phi_1$ | 0.4132 | 0.2854 | 0.3007 | 0.3000 |
| $a_1$ | 0.931 | 0.3250 | 0.3987 | 0.4000 |
| $a_4$ | 0.4383 | 0.1598 | −0.0003 | 0.000 |
| $a_5$ | −0.4298 | −0.1261 | −0.0005 | 0.000 |

在未调整数据下，$a_1$、$a_4$ 和 $a_5$ 的估计值将周期性生产率所产生的季节性归入了禀赋过程中，而 $\phi_1$ 的估计值高估了调整成本。

在去除季节均值的数据下，$a_4$ 和 $a_5$ 的估计值仍然把一些季节性归入禀赋过程，而 $\phi_1$ 和 $a_1$ 的估计值低于真值。

汉森和萨金特对去除季节均值数据下的估计值作出解释，其中 $c_t$ 和 $i_t$ 分别为消费和投资：

> 为了减小在季节性频段处的误差，估计值以模型所允许的唯一方式——即通过禀赋过程——将一些季节性加入了 $\{c_t, i_t\}$。这样做有助于模型在季节频率处"匹配"数据，代价是在非季节频率处拟合得更差。

汉森和萨金特将未调整数据相对于去除季节均值数据产生更大误差的原因，归结于均值项。

该模型只能通过提高其在季节频率处的谱密度来缩小这一项，而更大的 $a_4$ 和 $a_5$ 估计值正是这样做的。

在理想调整后的数据下，$\phi_1$ 和禀赋过程的估计值接近真值，而 $\gamma$ 的估计值 $0.1025$ 恰好等于四个季度生产率参数的平均值 $(0.13 + 0.1 + 0.1 + 0.08)/4$。

### 西姆斯的永久收入示例

{cite:t}`Sims1993` 使用一个永久收入模型给出了一个调整后数据减小误差的例子，以及一个调整后数据增大误差的例子。

一个具有二次效用函数和贴现因子 $\beta$ 的消费者面对毛利率 $\beta^{-1}$，因此消费 $C_t$ 是一个鞅，满足

$$
\Delta C_t = (1 - \beta) \sum_{j=0}^{\infty} \beta^j (E_t - E_{t-1}) Y_{t+j} ,
$$

其中 $Y_t$ 为收入。

收入是一个非季节性成分

$$
Y^N_t = 1.8 \, Y^N_{t-1} - 0.81 \, Y^N_{t-2} + \varepsilon_t ,
$$

（标准化为单位方差）与一个重复 $\sqrt{2}, 0, -\sqrt{2}, 0$ 模式的确定性季节性成分之和 $Y_t = Y^N_t + Y^S_t$。

该确定性季节成分是完全可预测的，因此它永远不会修正预期收入，也不影响 $\Delta C_t$。

如果 $Y^N_t = \psi(L) \varepsilon_t$，则 $\Delta C_t = (1 - \beta)\psi(\beta) \varepsilon_t$，于是对收入新息 $\varepsilon_t$ 关于 $\Delta C_t$ 做回归的系数为 $1/[(1 - \beta)\psi(\beta)]$。

{doc}`lq_permanent_income` 推导出了这个公式 {eq}`eq:sprob13`，其中用 $d$ 代替 $\psi$。

西姆斯没有给出 $\beta$ 的值；我们采用 $\beta = 0.95$，这是他下面第二个例子中系数 $20$ 所隐含的值。

当 $\psi(L) = (1 - 0.9L)^{-2}$ 时，该系数为 $(1 - 0.9\beta)^2 / (1 - \beta) = 0.4205$，与西姆斯报告的值相符。

一位对收入拟合自回归 $a(L) Y_t = e_t$、并相信永久收入模型的计量经济学家，会计算出系数 $a(\beta)/(1 - \beta)$。

西姆斯首先假设计量经济学家对未调整数据（不含季节虚拟变量）拟合一个九阶自回归。

该大样本自回归是收入过程在九阶自回归上的 m-投影，因此它满足以自协方差 $R^N(s) + R^S(s)$ 构成的尤尔-沃克方程，其中 $R^N$ 是 $Y^N$ 的自协方差函数，$R^S(s) = \cos(\pi s/2)$ 是季节性模式的自协方差函数。

```{code-cell} ipython3
β = 0.95
fN_ar2 = ar_spectrum([1.8, -0.81])
fN_ar2 = fN_ar2 / np.mean(fN_ar2)                    # unit variance


def seasonal_autocov(n_lags, gain2=1.0):
    """R^S(s) = cos(π s / 2), times the squared gain of a filter at π/2."""
    return gain2 * np.cos(np.pi * np.arange(n_lags + 1) / 2)


def implied_coefficient(a):
    """Coefficient of ε on ΔC implied by the autoregression a(L) Y = e."""
    a_β = 1 - sum(a_j * β**(j + 1) for j, a_j in enumerate(a))
    return a_β / (1 - β)


a9 = yule_walker(autocov(fN_ar2, 9) + seasonal_autocov(9), 9)
print("AR(9) coefficients:", np.round(a9, 4))
print(f"true coefficient {(1 - 0.9 * β)**2 / (1 - β):.4f}, "
      f"implied by the AR(9): {implied_coefficient(a9):.4f}")
```

我们的投影给出的系数与西姆斯方程 (17) 中的系数相符，除了 $Y_{t-8}$ 上的系数之外：我们得到的是 $-0.2084$，而西姆斯印出的是 $+0.2084$。

只有负号才能重现他给出的隐含系数 $0.4925$，即偏差因子 $1.17$。

九阶自回归预测了大部分确定性季节成分，但它把收入新息的一部分归因于随机季节成分，于是永久收入模型就隐含消费对收入新息的反应，要弱于实际情况。

西姆斯指出，将该自回归预测误差对数据中的 $\Delta C_t$ 做回归，仍然能恢复出 $0.4205$。

该自回归的预测误差等于真实新息加上一个与之正交的成分，而 $\Delta C_t$ 正比于真实新息。

因此，使用九阶自回归的计量经济学家会错误地得出结论：消费对收入的反应比理论所隐含的更强。

接下来，西姆斯通过减去滤波器 $\kappa(L)\kappa(L^{-1})$ 所提取的季节成分，对 $C$ 和 $Y$ 同时进行调整，其中

$$
\kappa(L) = 0.25 \, \frac{1 - L^{16}}{1 - L^4} - 0.0625 \, \frac{1 - L^{16}}{1 - L}
$$

是他写作 $F(L)$ 的滤波器，并把调整后的序列当作未调整数据来处理。

调整后数据的谱密度是未调整数据谱密度的 $(1 - |\kappa(e^{-i\omega})|^2)^2$ 倍。

西姆斯报告说，对调整后收入拟合一个二阶自回归只得到 $1.0137$ 的偏差因子。

按我们对该滤波器的理解来计算，这一步得到的偏差因子是 $0.893$，我们尝试过的其他理解方式也都未能重现 $1.0137$；同一种理解方式在下面第二个例子中给出 $17.08$，与他的 $17.06$ 相差在 $0.02$ 之内。

我们再补充一个西姆斯未曾报告的计算：从准则中删除季节性频段并拟合二阶自回归，可以恢复真值，因为 $Y^N$ 本身就是一个二阶自回归，而确定性季节成分的全部功率都集中在频率 $\pi/2$ 处。

```{code-cell} ipython3
κ_ft = (0.25 * sum(z**(4 * k) for k in range(4))
        - 0.0625 * sum(z**k for k in range(16)))
adjust2 = (1 - np.abs(κ_ft)**2)**2                # squared gain of 1 - κκ*
z_q = np.exp(-1j * np.pi / 2)                     # the seasonal frequency π/2
κ_q = (0.25 * sum(z_q**(4 * k) for k in range(4))
       - 0.0625 * sum(z_q**k for k in range(16)))
gain2_q = (1 - np.abs(κ_q)**2)**2                 # zero: the seasonal is removed

a2_adj = yule_walker(autocov(fN_ar2 * adjust2, 2)
                     + seasonal_autocov(2, gain2_q), 2)
print(f"AR(2), adjusted data: implied coefficient "
      f"{implied_coefficient(a2_adj):.4f}, bias factor "
      f"{implied_coefficient(a2_adj) / 0.4205:.3f}")

keep_B = seasonal_keep(np.pi / 8)
a2_B, _ = ar_projection(fN_ar2, 2, keep_B)       # the seasonal lies in a deleted band
print("AR(2), seasonal bands deleted:", np.round(a2_B, 4),
      f" implied coefficient {implied_coefficient(a2_B):.4f}")
```

在西姆斯的第二个例子中，使用调整后数据的计量经济学家表现反而不如使用未调整数据的计量经济学家。

西姆斯将 $Y^N$ 替换为单位方差的白噪声，此时真实系数为 $1/(1 - \beta) = 20$，他报告说对未调整数据拟合九阶自回归隐含的系数为 $20.62$，而调整后数据隐含的系数为 $17.06$。

西姆斯并未说明在这个例子中对调整后数据所拟合自回归的阶数；我们像第一个例子那样拟合一个二阶自回归，同时也报告九阶自回归的结果。

```{code-cell} ipython3
f_white = np.ones(n_grid)
a9_w = yule_walker(autocov(f_white, 9) + seasonal_autocov(9), 9)
a2_adj_w = yule_walker(autocov(f_white * adjust2, 2)
                       + seasonal_autocov(2, gain2_q), 2)
a9_adj_w = yule_walker(autocov(f_white * adjust2, 9)
                       + seasonal_autocov(9, gain2_q), 9)
a2_band_w, _ = ar_projection(f_white, 2, keep_B)
print(f"true coefficient:                   {1 / (1 - β):.2f}")
print(f"AR(9), unadjusted data:             {implied_coefficient(a9_w):.2f}")
print(f"AR(2), data adjusted with κκ*:      {implied_coefficient(a2_adj_w):.2f}")
print(f"AR(9), data adjusted with κκ*:      {implied_coefficient(a9_adj_w):.2f}")
print(f"AR(2), seasonal bands deleted:      {implied_coefficient(a2_band_w):.2f}")
γ_adj = autocov(f_white * adjust2, 4)
print("autocorrelations of adjusted white noise, lags 1 to 4:",
      np.round(γ_adj[1:] / γ_adj[0], 3))
print("AR(2) coefficients, adjusted data:", np.round(a2_adj_w, 3))
```

我们的计算对未调整数据得到 $20.64$，与西姆斯的 $20.62$ 很接近。

西姆斯的调整滤波器去除了季节成分，但它同时也降低了数据在季节频率附近的谱密度，因此计量经济学家实际上是对一个谱密度在这些地方出现凹陷（而白噪声没有这种凹陷）的过程拟合二阶自回归。

调整后的白噪声具有正的自相关，在滞后 1 到 3 之间约为 $0.08$ 到 $0.10$，因此二阶自回归的系数为正，即 $0.076$ 和 $0.082$，且 $a(\beta) = 0.854$。

计量经济学家由此将持续性归因于本不具有持续性的收入新息，从而推断消费对收入新息的反应比实际更强。

从准则中删除季节性频段，其他频率保持不变，而在频段之外拟合的二阶自回归系数为零，这隐含真实系数 $20$。

在 {ref}`ae_ex_invariance` 中，计量经济学家对模型和数据都进行了滤波。

西姆斯总结道：

> 使用未调整数据并配合正确设定的季节性变化模型，始终是最佳选择。但若对季节性采取随意处理，将其视为模型中的一个"讨厌成分"，则可能导致比使用调整后数据更糟糕的误差。

(ae_garbling_section)=
## 弱化与数据处理不等式

{doc}`blackwell_kihlstrom` 把一个实验称为另一个实验的**弱化（garbling）**，如果它的信号可以通过一个马尔可夫核从另一个实验的信号生成。

{ref}`数据处理不等式 <bk_data_processing>` 表明，对两个分布应用同一个马尔可夫核不会增大它们之间的任何 f-散度，其中包括 KL 散度；并且，当该核是由这对分布的一个充分统计量所诱导时，散度保持不变。

滤波、删除频段、去除季节均值以及抽样都是弱化操作，而该不等式整理出了本讲座关于它们的各项发现。

### 滤波器、频段与数据处理不等式

当计量经济学家用 $c(L)$ 对数据滤波，并以相同方式对模型滤波时，在真实过程和模型下，滤波后的序列都是未滤波序列的同一个函数。

因此，滤波对两个过程应用的是同一个（确定性的）马尔可夫核，数据处理不等式表明滤波不会增大它们之间的散度。

当频率响应 $c(e^{-i\omega})$ 至多只在孤立频率处为零时，滤波将 $f$ 和 $g$ 都乘以 $|c(e^{-i\omega})|^2$，在其他每个频率处比值 $f/g$ 保持不变，因此散度率 {eq}`ae_kl_rate` 不发生改变。

*对数据和模型同时应用频率响应至多在孤立频率处为零的滤波器，无论它在那些频率附近使谱密度收缩得多么厉害，都会使散度率和 m-投影保持不变。*

此外，如果数据的谱测度在那些频率处不含质量，则滤波后的序列可以在均方意义下确定未滤波的序列，因此该滤波器是这对分布的充分统计量，此时 {ref}`ae_ar_section` 所述的不变性正是数据处理不等式的等号情形。

西姆斯的滤波器 $1 - \kappa(L)\kappa(L^{-1})$ 在 $\pm\pi/2$、$\pi$ 以及 $\pm 0.485\pi$ 附近的两个频率处为零。

它去除了确定性季节成分（其谱测度的全部质量都集中在 $\pm\pi/2$ 处），而对于余下的部分，{ref}`ae_ex_invariance` 表明等号成立。

这种不变性是 KL 散度所独有的，因为 KL 散度通过比值 $f / g$ 逐频率地比较两个过程。

而一个度量概率质量需要移动多远的距离，则不具备这种性质。

{doc}`exponential_families` 中 $N(0, \Sigma_1)$ 与 $N(0, \Sigma_2)$ 之间平方 2-瓦瑟斯坦距离为 $\operatorname{tr}\left(\Sigma_1 + \Sigma_2 - 2 (\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2}\right)$ {cite}`Gelbrich1990`。

当 $T$ 较大时，两个平稳过程中 $T$ 个连续观测值的协方差矩阵几乎都被同一个傅里叶基所对角化，因此对于两个均值为零的高斯过程，该距离除以 $T$ 后收敛到

$$
\frac{1}{2\pi} \int_{-\pi}^{\pi} \left( \sqrt{f(\omega)} - \sqrt{g(\omega)} \right)^2 d\omega .
$$ (ae_w2_rate)

用 $c(L)$ 对两个过程同时滤波会使被积函数乘以 $\lvert c(e^{-i\omega}) \rvert^2$，因此即使散度率保持不变，滤波也会改变该距离。

我们对 $\rho = 0.8$ 的一阶自回归和方差为 $2$ 的白噪声验证 {eq}`ae_w2_rate`，并用 $1 - 0.9 L$ 对两者同时滤波。

```{code-cell} ipython3
def w2_squared_gaussian(S1, S2):
    """Squared 2-Wasserstein distance between N(0, S1) and N(0, S2)."""
    root = sqrtm(S1)
    return np.trace(S1 + S2 - 2 * np.real(sqrtm(root @ S2 @ root)))


def w2_rate(f, g):
    """(1/2π) ∫ (√f - √g)² dω on the grid."""
    return np.mean((np.sqrt(f) - np.sqrt(g))**2)


f_ar, g_wn = ar_spectrum(0.8), ar_spectrum(0.0, 2.0)
for T_w in (100, 400):
    S_f, S_g = toeplitz(autocov(f_ar, T_w - 1)), toeplitz(autocov(g_wn, T_w - 1))
    print(f"T = {T_w}: W2² / T = {w2_squared_gaussian(S_f, S_g) / T_w:.4f}")

gain_sq = np.abs(1 - 0.9 * z)**2
print(f"formula (ae_w2_rate): {w2_rate(f_ar, g_wn):.4f}")
print(f"filtered by 1 - 0.9L: W2 rate {w2_rate(gain_sq * f_ar, gain_sq * g_wn):.4f}, "
      f"KL rate {kl_rate(f_ar, g_wn):.4f} before and "
      f"{kl_rate(gain_sq * f_ar, gain_sq * g_wn):.4f} after")
```

滤波使 KL 散度率保持在原来的值，而瓦瑟斯坦率却发生了变化。

{doc}`exponential_families` 针对单个随机变量的变换描述了同样的对比。

一个滤波器无法把一段频段从似然函数中删除。

一个频率响应在某频段上为零的滤波器会使滤波后的过程变为线性确定性的，此时假设谱密度远离零的公式 {eq}`ae_kl_rate` 不再适用。

例如，如果真实过程和模型都是方差分别为 $1$ 和 $2$ 的白噪声，则滤波后的模型是滤波后真实过程的 $\sqrt{2}$ 倍，于是对于每个 $T$，$T$ 个滤波后观测值分布之间的散度都等于 $T$ 个未滤波观测值分布之间的散度。

汉森和萨金特转而在频域中定义理想季节调整，方法是从频域似然中省略季节频率。

在频率 $\omega_j = 2\pi j / T$（$j = 0, \ldots, T - 1$）处的有限傅里叶变换 $w_j$，是样本的一个可逆线性变换，因此在每个样本量下，丢弃频段内的变换都是一种弱化。

除以 $T$ 后，剩余变换之间的散度收敛到限制在频段之外频率集合 $K$ 上的散度率：

$$
D_K(f \| g) = \frac{1}{4\pi} \int_K \left[ \frac{f(\omega)}{g(\omega)} - 1 - \log \frac{f(\omega)}{g(\omega)} \right] d\omega \leq D(f \| g) .
$$ (ae_dpi_bands)

$D_K$ 的两倍是 {eq}`ae_hs_ideal` 第一项减去不依赖于 $\delta$ 的项，因此 {eq}`ae_hs_ideal` 之后所说的受限散度率至多等于完整散度率，正是数据处理不等式。

当 $f = g$ 在频段内几乎处处成立时，等号成立：此时，在大样本下，频段内的变换在两个过程下具有相同的分布，并且与其余变换相互独立，因此丢弃它们并不会丢失任何能够区分这两个过程的信息。

像西姆斯笔下的计量经济学家那样仅对数据滤波而不对模型滤波，相当于只对两个分布中的一个应用该核，数据处理不等式对此没有给出任何说明。

西姆斯的永久收入示例朝着相反的方向发展，尽管每一种比较也都改变了自回归的阶数。

在第一个例子中，调整后数据使偏差因子更接近一：对未调整数据拟合九阶自回归得到的是 $1.17$，而对调整后数据拟合二阶自回归，按我们的计算得到 $0.893$，按西姆斯的计算得到 $1.0137$。

在第二个例子中，调整后数据使隐含系数从 $20.64$（二阶自回归）变为 $17.08$，或用九阶自回归变为 $16.82$，距真值 $20$ 更远。

### 布莱克韦尔链

汉森和萨金特的三种数据按弱化关系排序。

去除季节均值后的数据，是未调整数据对季节虚拟变量做回归所得到的残差，因此它们是未调整数据的一个函数。

理想调整后的数据是去除季节均值数据的一个函数。

当 $T$ 是 $4$ 的倍数时，季节虚拟变量所张成的空间，恰与频率 $0$、$\pi/2$、$\pi$ 和 $3\pi/2$ 处的有限傅里叶向量所张成的空间相同，因此去除季节均值相当于丢弃这四个频率处的变换，而理想调整还额外丢弃 $\pi/2$、$\pi$ 和 $3\pi/2$ 周围频段内的变换。

这三种数据构成了一系列嵌套的傅里叶变换集合，在每个样本量下，每一种都是前一种的弱化。

对于每个 $\delta$，从三个准则中各自减去不依赖于 $\delta$ 的项，剩下的恰好是相应弱化之后真实过程与模型在 $\delta$ 处散度率的两倍。

因此在每个 $\delta$ 处，从第一种数据到第二种数据、再从第二种到第三种，散度都是弱单调递减的。

最小化后的散度也会递减：如果 $\delta^*$ 是较细数据对应散度的最小化元，那么较粗数据对应的最小值至多等于较粗散度在 $\delta^*$ 处的值，而后者又至多等于较细散度在 $\delta^*$ 处的值。

在汉森和萨金特的表 1 中，估计量概率极限的误差也在每一步都递减。

我们用季度生产率参数的平均值 $0.1025$ 来衡量 $\gamma$ 的误差。

```{code-cell} ipython3
table_1 = {"γ": (0.0996, 0.1030, 0.1025), "φ_1": (0.4132, 0.2854, 0.3007),
           "a_1": (0.931, 0.3250, 0.3987), "a_4": (0.4383, 0.1598, -0.0003),
           "a_5": (-0.4298, -0.1261, -0.0005)}
true_values = {"γ": np.mean([0.13, 0.1, 0.1, 0.08]), "φ_1": 0.3, "a_1": 0.4,
               "a_4": 0.0, "a_5": 0.0}
print("absolute errors   unadjusted   means removed   ideally adjusted")
for name, estimates in table_1.items():
    errors = np.abs(np.array(estimates) - true_values[name])
    print(f"{name:>15}" + "".join(f"{e:15.4f}" for e in errors))
```

布莱克韦尔定理表明，一个了解某实验似然的决策者，在任何决策问题中，使用该实验都至少不会比使用它的任何弱化差。

西姆斯的结论"使用未调整数据并配合正确设定的季节性变化模型，始终是最佳选择"，正是该定理在估计问题上的对应版本：对于一个正确设定的模型，删除频率不可能降低最大似然估计量的渐近方差。

汉森和萨金特的表 1 则颠倒了布莱克韦尔的排序。

他们笔下的计量经济学家是用一个设定有误的模型的似然来评价这三种数据的，因此布莱克韦尔定理并不适用，而偏好、技术和禀赋过程参数估计的概率极限，对于最粗糙的数据反而最接近真值。

汉森和萨金特在引言中描述了他们这一颠倒所回应的论点：

> 其他作者曾断言，理应使用季节未调整数据来估计理性预期模型，理由是被建模的主体本身被假定是用季节未调整数据来求解预测和信号提取问题的，因此他们行为所隐含的跨方程约束也适用于季节未调整的序列。西姆斯回应说，这种论证假设模型是设定正确的，而他断言，一旦承认存在近似误差的可能性，情况就会发生改变。

这里提到的其他作者是指 {cite:t}`Sargent1978Seasonal` 和 {cite:t}`Singleton1988`。

模型内部的主体了解真实的似然，因此布莱克韦尔定理适用于他们，他们没有理由去弱化自己的数据。

西姆斯的回应是，该定理并不适用于一个模型设定有误的计量经济学家。

{doc}`information_transfer` 描述了决策者偏好经过弱化的实验的另一个原因。

### 两对分布

数据处理不等式适用于任意两个分布。

将其应用于真实过程与 $\delta$ 处的模型，它表明弱化不可能增大设定误差；汉森和萨金特的理想调整，即他们对西姆斯论证的表述版本，用的就是这一面。

将其应用于模型在两个不同参数值 $\delta$ 和 $\delta'$ 处的情形，它表明弱化不可能让它们变得更容易区分。

如果某种弱化把它们映射到同一个分布，它们就变得在观测上等价，这正是汉森和萨金特所说的省略频率"可能通过引入额外的最小化元而妨碍识别"的含义。

对于相近的参数值，散度是费舍尔信息矩阵中一个二次型的一半，正如 {doc}`exponential_families` 中所述。

对于在集合 $K$ 内的频率上拟合、谱密度为 $g(\omega; \delta)$ 的标量模型，其频域似然的费舍尔信息矩阵为

$$
\mathcal{I}_K(\delta) = \frac{1}{4\pi} \int_K \nabla_\delta \log g(\omega; \delta) \, \nabla_\delta \log g(\omega; \delta)^\top \, d\omega ,
$$ (ae_info_K)

因此删除频率会在矩阵序意义下降低它：$\mathcal{I}_K(\delta) \preceq \mathcal{I}(\delta)$，其中 $\mathcal{I}$ 使用全部频率。

这正是 {doc}`blackwell_kihlstrom` 中数据处理不等式的费舍尔信息形式。

求逆会反转矩阵序，因此对于一个设定正确的模型，$\sqrt{T}$ 乘以估计量误差的渐近协方差矩阵为 $\mathcal{I}_K(\delta)^{-1} \succeq \mathcal{I}(\delta)^{-1}$，这正是汉森和萨金特脚注 11 中所说的效率代价。

当 $\nabla_\delta \log g = 0$ 在被删除频率上几乎处处成立时，两个矩阵相等，此时这些频率不携带关于 $\delta$ 的任何信息。

汉森和萨金特那些设定正确的例子，即使删去季节频率仍然保持可识别性，尽管这些参数"对季节频率的影响最为强烈"：

> 它们以某种方式影响所有频率，这种方式足以在删去季节频率之后仍然保持可识别性。

在那里，删除频段降低了关于参数的信息量，但不会使任何其他参数值在观测上与真值等价。

汉森和萨金特在结论部分将这个不等式的两个方面相互对照：

> 其次，当估计一个设定正确的模型时，使用季节调整数据会带来相当大的效率损失，而且这种损失会随着原始数据中季节性重要程度的提高而增大。实践中，估计量的方差与偏差之间可能存在权衡关系，而我们在本文中并未探讨这一维度。

他们的脚注 17 补充道："基于导出 (4.2) 的逻辑，关于设定有误模型估计量的渐近分布，是可以说出一些结论的。我们计划在后续工作中探讨这一点。"

当数据来自一个谱密度 $f$ 与 $g(\cdot\,; \delta^*_K)$（$\delta^*_K$ 是在 $K$ 内频率上的 m-投影）不同的高斯过程时，$\sqrt{T}(\hat \delta_K - \delta^*_K)$ 的渐近协方差矩阵为 $\mathcal{H}_K^{-1} \mathcal{J}_K \mathcal{H}_K^{-1}$，其中

$$
\mathcal{H}_K = \frac{1}{4\pi} \int_K \left[ \nabla \log g \, \nabla \log g^\top \, \frac{f}{g} + \nabla^2 \log g \left( 1 - \frac{f}{g} \right) \right] d\omega ,
\qquad
\mathcal{J}_K = \frac{1}{4\pi} \int_K \nabla \log g \, \nabla \log g^\top \left( \frac{f}{g} \right)^2 d\omega ,
$$ (ae_sandwich)

其中 $g$ 及其导数均在 $\delta^*_K$ 处取值。

$\mathcal{H}_K$ 是限制在 $K$ 上的散度率的海森矩阵，$\mathcal{J}_K$ 是 {eq}`ae_whittle` 得分除以 $\sqrt{T}$ 后的渐近方差。

因子 $(f/g)^2$ 来自周期图的方差，该方差始终接近 $f^2$。

当 $f = g(\cdot\,; \delta^*_K)$ 时，两个矩阵都等于 $\mathcal{I}_K$，协方差矩阵就化简为 $\mathcal{I}_K^{-1}$。

下面的代码针对季节性示例，计算在删除几种不同宽度频段后的 AR(1) 投影、最小化后的散度，以及在两种情形下 $\hat\rho$ 的渐近方差：一种是数据仅来自 $x_t$（此时 AR(1) 设定正确），另一种是数据来自 $y_t = x_t + s_t$。

```{code-cell} ipython3
def ar1_derivatives(ρ, σ2):
    """∇ log g and ∇² log g on the grid for g = σ²/|1 - ρe^{-iω}|², θ = (ρ, σ²)."""
    d1 = np.vstack([2 * np.real(z / (1 - ρ * z)), np.ones(n_grid) / σ2])
    d2 = np.zeros((2, 2, n_grid))
    d2[0, 0] = 2 * np.real(z**2 / (1 - ρ * z)**2)
    d2[1, 1] = -1 / σ2**2
    return d1, d2


def ar1_covariance(f, ρ, σ2, keep):
    """Asymptotic covariance matrix of √T(θ̂ - θ*), formula (ae_sandwich)."""
    d1, d2 = ar1_derivatives(ρ, σ2)
    r = np.where(keep, f / ar_spectrum(ρ, σ2), 0.0)
    H = 0.5 * ((d1 * r) @ d1.T
               + (d2 * np.where(keep, 1 - r, 0.0)).sum(axis=2)) / n_grid
    J = 0.5 * ((d1 * r**2) @ d1.T) / n_grid
    H_inv = np.linalg.inv(H)
    return H_inv @ J @ H_inv


print("band width     ρ*    divergence   Var, x alone   Var, y = x + s")
for Δ in (0, np.pi / 16, np.pi / 8, np.pi / 4, np.pi / 2):
    keep = seasonal_keep(Δ)
    a_K, σ2_K = ar_projection(f_y, 1, keep)
    V_x = ar1_covariance(f_x, ρ_x, 1.0, keep)[0, 0]
    V_y = ar1_covariance(f_y, a_K[0], σ2_K, keep)[0, 0]
    print(f"{Δ / np.pi:9.4f}π {a_K[0]:7.4f} "
          f"{kl_rate(f_y, ar_spectrum(a_K, σ2_K), keep):11.5f} "
          f"{V_x:13.4f} {V_y:15.4f}")
```

当频段宽度为 $\pi/2$ 时，只剩下低于 $\pi/4$ 的频率。

当 AR(1) 设定正确时，删除宽度为 $\pi/8$ 的频段，会使 $\hat\rho$ 的渐近方差从 $0.190$ 升至 $0.194$，约增加 2%，这是因为关于 $\rho$ 的信息集中在低频处。

汉森和萨金特的脚注 11 对理性预期模型则有不同的描述：其跨方程约束把大部分信息放在高功率的季节频率处，在那里删除频段的代价更大。

在季节性示例中，使用全部频率的估计量渐近方差为 $4.57$，是 AR(1) 设定正确时方差的 24 倍。

在季节峰值处（即 $f/g$ 较大之处）的周期图纵坐标，会以权重 $(f/g)^2$ 进入 $\mathcal{J}_K$。

删除宽度为 $\pi/8$ 的频段去掉了这些纵坐标，使方差降至 $0.31$，所以在这里，删除频段同时降低了偏差和方差。

*删除频率所带来的效率代价，只有在模型设定正确时才能得到保证。*

{ref}`ae_ex_tradeoff` 求出了当季节性成分较小时能权衡偏差与方差的频段宽度。

{doc}`specification_tests` 构造了一个检验，用以比较删除频段的估计量与使用全部频段的估计量，并利用 {eq}`ae_sandwich` 计算在模型设定有误时二者差异的方差。

### 散度，而非估计值

数据处理不等式限定的是散度；它并不说明最小化元会落在何处。

删除一段低频频段而非季节性频段，同样也是一种弱化。

```{code-cell} ipython3
dist_0 = np.minimum(ω, 2 * np.pi - ω)
print(f"all frequencies:                 ρ* = {a_full[0]:.4f},  "
      f"divergence {kl_rate(f_y, g_full):.4f}")
for Δ in (np.pi / 8, np.pi / 4):
    keep_low = dist_0 >= Δ / 2              # delete a band of width Δ around zero
    a_low, σ2_low = ar_projection(f_y, 1, keep_low)
    print(f"low band of width {Δ / np.pi:.3f}π deleted: ρ* = {a_low[0]:.4f},  "
          f"divergence {kl_rate(f_y, ar_spectrum(a_low, σ2_low), keep_low):.4f}")
```

最小化后的散度从 $0.429$ 降到 $0.366$ 和 $0.330$，这符合该不等式的要求，但 $\rho^*$ 却从 $0.62$ 降到 $0.25$ 和 $0.02$，离非季节性成分的系数 $0.9$ 更远了。

这种弱化丢弃的恰恰是季节性成分几乎没有功率、而关于 $\rho$ 的大部分信息所在的那些频率：零附近宽度为 $\pi/8$ 的频段，容纳了 $s_t$ 方差的 0.03%，却容纳了 AR(1) 设定正确时 $\mathcal{I}(\rho)$ 的 91%。

当一种弱化恰好丢弃模型出错的那些频率、而保留携带关于相关参数信息的频率时，它可以改善计量经济学家所关心参数的估计。

汉森和萨金特在结论部分的第三点限定条件正考虑到了这一点："设定有误模型估计的某些具体例子，无疑有可能与西姆斯的建议相悖。"

他们提出的理想调整得以奏效的条件，正好描述了一种恰好丢弃设定误差的弱化。

如果对于某个 $\delta_0$，模型在季节性频段之外与 $F$ 相匹配，且均值也匹配，那么这种弱化会把真实过程和 $\delta_0$ 处的模型映射到同一个分布，因为两个均值和谱密度都相同的高斯过程就是同一个过程。

此时在 $\delta_0$ 处弱化后的散度为零，这是其可能取到的最小值。

汉森和萨金特关于测量误差的脚注 12 也符合同样的理解：

> 严格来说，要援用这样的论证——通过忽略季节频率就能保持一致性——就必须假设测量误差*纯粹*是季节性的。这样的测量误差过程将是线性确定性的，也就是说，完全可以由其自身的过去线性预测出来。

要使删除频段完全去除某个测量误差，该误差的谱密度在频段之外必须为零。

这样一来，该谱密度对数的积分就是 $-\infty$，而 {ref}`ae_ar_section` 中的柯尔莫哥洛夫公式给出的一步向前预测误差方差为 $\sigma^2_\infty = 0$。

### 低频滤波器

汉森和萨金特在论文结尾将该论证推广到其他滤波器：

> 形式上，支持使用季节调整数据的那套论证，同样适用于那些被滤波以消除（比如说）低频成分的数据。例如，霍德里克和普雷斯科特（1980）构造了一种本质上是高通滤波器的方法，它能从时间序列中消除极低频成分，同时保留商业周期及更高频率的成分。普雷斯科特和许多其他研究者都曾尝试将用霍德里克-普雷斯科特滤波器滤波过的数据，与他们专门为捕捉商业周期频率而设计、但不信任其低频性质的理论模型相匹配。这样的实证策略大概可以通过适当运用我们的近似准则 (4.2) 加以合理化。

{cite:t}`HodrickPrescott1997` 的滤波器中周期性成分的频率响应为

$$
h(\omega) = \frac{4 \lambda (1 - \cos \omega)^2}{1 + 4 \lambda (1 - \cos \omega)^2} ,
$$

它仅在 $\omega = 0$ 处为零。

由于 $h(0) = 0$，该滤波器去除了均值，于是 {eq}`ae_kl_rate_vector` 的均值项消失；又由于 $h$ 在其他频率处都不为零，对数据和模型同时滤波，散度率的其余部分保持不变。

因此，对于本讲座所讨论的平稳过程，对霍德里克-普雷斯科特滤波后的数据拟合模型（并以同样方式对模型滤波）所得到的高斯最大似然估计，与对未滤波数据（去掉均值项）估计所收敛到的 m-投影是相同的。

在大样本下，滤波器可以通过三种方式改变计量经济学家所估计的内容：

* 可以只应用于数据而不应用于模型，如西姆斯的表述方式；
* 可以用从频域似然中删除一段低频频段来代替，这种弱化在模型主要在低频处设定有误时能够改善估计，这正是汉森和萨金特所描述的情形；或者
* 可以与一个并非似然函数的准则一起使用，例如滤波后数据与滤波后模型若干方差和协方差之间的距离，在这种情形下滤波器的频率响应对各个频率进行加权。

下面的代码使用了季度数据常用的平滑参数 $\lambda = 1600$。

```{code-cell} ipython3
λ = 1600
h_hp = 4 * λ * (1 - np.cos(ω))**2 / (1 + 4 * λ * (1 - np.cos(ω))**2)
print(f"D(f || g) = {kl_rate(f_y, g_full):.6f},  "
      f"D(h²f || h²g) = {kl_rate(h_hp**2 * f_y, h_hp**2 * g_full):.6f}")
a_hp, _ = ar_projection(h_hp**2 * f_y, 1)
print(f"AR(1) projection of filtered data, unfiltered model: ρ* = {a_hp[0]:.4f}")
```

在季节性示例中，对滤波后数据拟合未滤波的 AR(1) 得到 $\rho^* = 0.16$，这是因为该滤波器去除了把投影拉向 $0.9$ 的低频功率。

### 抽样

对一个连续时间过程在整数日期上抽样，同样也是一种弱化：在真实过程和模型下，抽样序列都是连续时间路径的同一个函数。

西姆斯在其表 I 第 (4) 列中所用的、对单位区间取平均的做法，也是如此。

{eq}`ae_folding` 的分母——即抽样后回归量的谱密度——在频域中展示了这种弱化：抽样把频率 $\omega + 2\pi k$ 的贡献都加到了频率 $\omega$ 上。

将数据处理不等式应用于两个连续时间模型，它表明抽样不可能让它们变得更容易区分。

两个谱密度折叠后得到相同离散时间谱密度的连续时间过程，从抽样数据来看是观测上等价的；在西姆斯的设定中，不同的连续时间滞后分布可能对应相同的离散时间滞后分布 $B$。

一位把 $B$ 解读为 $b$ 的计量经济学家会面临这种识别问题。

{cite:t}`HansenSargent1983` 对此作了描述：

> 连续时间协方差平稳随机过程理论中有一个众所周知的结果，即一般而言，无法从等间距的离散时间观测中推断出连续时间谱密度函数。这种识别问题通常被称为混叠现象。

他们证明，将连续时间谱密度矩阵限制为有理函数，"在解决混叠问题方面所能达到的程度，远超以往所认识到的"。

在具有有理谱密度的模型这一类别内，一般只有有限个连续时间模型在观测上等价；在参数空间的某些区域中，"可能根本不存在识别问题"；而对于一阶向量马尔可夫过程，"通常存在一个足够精细的离散抽样间隔，使连续时间模型可以被识别"。

抽样在所有连续时间过程构成的集合上并非一一映射，但在一个受限的族内，它可以是一一映射的。

汉森和萨金特补充说，经济理论所施加的约束，同样也可以用来识别连续时间模型。

## 结语

一个设定有误的时间序列模型的最大似然估计量，收敛于真实过程在该模型上的 m-投影。

KL 散度率 {eq}`ae_kl_rate` 按照数据谱密度与模型谱密度之比，对每个频率计费；在一个分布滞后回归中，回归量的谱密度对滞后分布在每个频率处的误差进行加权，如 {eq}`ae_sims_criterion` 所示。

对一个非季节性成分系数为 $0.9$、季节性成分方差为非季节性成分四分之一的序列拟合一阶自回归，得到 $\rho^* = 0.62$；删除宽度为 $\pi/8$ 的季节性频段，使 $\rho^*$ 升至 $0.87$。

用同一个可逆滤波器对数据和模型同时滤波，投影保持不变；删除频段，或仅对数据滤波，都会改变投影。

在 {cite:t}`Sims1971` 的例子中，对抽样数据做最小二乘，在滞后零处放置了约 $0.43$ 的系数，而连续时间滞后分布在该处的值为 $1$。

在 {cite:t}`Sims1974` 的例子中，回归量中的季节性噪声，连同一个使估计滞后分布保持较短的约束，共同产生了一组系数，若将其解读为一个短期劳动力需求函数，会显示出短期规模报酬递增的迹象。

在 {cite:t}`Sims1993` 的第一个例子中，一个针对未调整数据设定有误的季节性模型，把消费对收入的反应低估了 $1.17$ 倍；在他的第二个例子中，调整后数据产生的误差大于未调整数据。

{doc}`phillips_misspecified` 描述了这样一种均衡：在该均衡中，主体的信念是他们自己信念所生成过程在 {eq}`ae_hs_criterion` 意义下的 m-投影；{cite:t}`EspondaPouzo2016` 把这类均衡称为伯克-纳什均衡。

{doc}`specification_tests` 描述了比较两个投影的检验方法。

滤波、删除频段、去除季节均值以及抽样都是弱化操作，数据处理不等式为它们所留下的散度排出了次序；但它并不为估计值排序。

汉森和萨金特的三种数据构成了一条弱化链，而他们那位设定有误的计量经济学家却从最粗糙的数据中获得了最好的估计，这颠倒了布莱克韦尔定理给出的、面向了解真实似然的决策者的那种排序。

对于一个设定正确的模型，删除频率不可能降低估计量的渐近方差；而在这个设定有误的季节性示例中，删除季节性频段同时降低了偏差和方差。

## 练习

```{exercise}
:label: ae_ex_sampling

{cite:t}`Sims1971` 的表 III 给出了对应于驼峰状连续时间滞后分布 $b(t) = 4 t e^{-2t}$（$t \geq 0$）的离散时间滞后分布 $B$。

1. 证明 $\tilde b(\omega) = 4/(2 + i\omega)^2$。
2. 使用函数 `triangle_lag`、`sampled_lag` 和 `band_limited_lag`，对 $t = -4, \ldots, 8$ 计算该表的第 (2)、(3)、(5) 列，并将 $B(0)$ 和 $B(1)$ 与 $b(0)$ 和 $b(1)$ 进行比较。
3. 在第 (5) 列中，$B(t) = \int b(s) \frac{\sin \pi (t - s)}{\pi (t - s)} \, ds$。证明在整数 $t$ 处，$B(t) = (-1)^t \int b(s) \frac{\sin \pi s}{\pi (s - t)} \, ds$，并用此公式预测负 $t$ 处 $B(t)$ 的符号。
```

```{solution-start} ae_ex_sampling
:class: dropdown
```

以下是一种解法。

对于第 1 部分，$\tilde b(\omega) = 4 \int_0^\infty t e^{-(2 + i\omega) t} \, dt = 4 / (2 + i\omega)^2$。

```{code-cell} ipython3
b_hump = lambda t: np.where(t >= 0, 4 * t * np.exp(-2 * t), 0.0)
b_hump_ft = lambda w: 4 / (2 + 1j * w)**2

print_lag_table(t_vals, b_hump(t_vals), triangle_lag(b_hump, t_vals),
                sampled_lag(b_hump_ft, S_ou, t_vals),
                band_limited_lag(b_hump_ft, t_vals))
```

计算出的各列与西姆斯表 III 一致，尽管其标题将滞后分布印为 $\exp(-2t)$，而其图形和第 (1) 列对应的是 $4t\exp(-2t)$。

该连续时间滞后在 $t = 0$ 处为零，在 $t = 0.5$（两个抽样日期之间）处达到峰值。

在第 (2) 和 (3) 列中，尽管 $b(0) = 0$，但 $B(0) \approx 0.27$；而 $B(1) \approx 0.5$ 接近 $b(1) = 0.54$；在第 (5) 列中，$B(0) = 0.29$，$B(1) = 0.60$。

对抽样数据做最小二乘，把日期 $0$ 到 $1$ 之间发生的部分响应归到了日期 $0$ 上。

对于第 3 部分，在整数 $t$ 处，$\sin \pi (t - s) = (-1)^{t+1} \sin \pi s$，因此 $B(t) = (-1)^t \int b(s) \frac{\sin \pi s}{\pi (s - t)} \, ds$。

当 $t < 0$ 时，对每个 $s \geq 0$，因子 $1/(s - t)$ 都为正。

当 $|t|$ 较大时，该积分近似为 $\frac{1}{\pi |t|} \int b(s) \sin \pi s \, ds = \frac{\operatorname{Im} \tilde b(-\pi)}{\pi |t|}$，而 $\operatorname{Im} \tilde b(-\pi) = \operatorname{Im} \, 4/(2 - i\pi)^2 > 0$。

因此，在负 $t$ 处，$B(t)$ 的符号与 $(-1)^t$ 相同：在 $t = -2$ 和 $-4$ 处为正，在 $t = -1$ 和 $-3$ 处为负，与表中所示一致。

```{code-cell} ipython3
print(f"Im b(-π) = {b_hump_ft(-np.pi).imag:.3f}")
exact = integrate.quad(lambda s: b_hump(s) * np.sin(np.pi * s) / (np.pi * (s + 1)),
                       0, 40, limit=200)[0]
print(f"t = -1: integral {exact:.3f}, "
      f"large-|t| approximation {b_hump_ft(-np.pi).imag / np.pi:.3f}")
```

在 $t = -1$ 处该近似较为粗略，但符号同样是正确的。

```{solution-end}
```

```{exercise}
:label: ae_ex_filter

回到西姆斯 1974 年恒等滞后分布且 $c = 0$ 的例子。

假设计量经济学家用一个理想季节调整滤波器 $k$ 对 $y'$ 和 $x'$ 同时滤波，该滤波器的频率响应在 $\pi/2$ 和 $\pi$（以及 $-\pi/2$ 附近对称的频段）周围宽度为 $\Delta$ 的频段内为零，在其他地方为一。设 $y'' = k * y'$ 和 $x'' = k * x'$ 为滤波后的序列。

1. 解释为什么对 $y''$ 关于 $x''$ 滞后项做回归的大样本最小二乘系数满足 `constrained_lag` 在参数 `weight` 为 $|\tilde k|^2$ 时所求解的正规方程。
2. 对于 $\Delta \in \{\pi/16, \pi/8, \pi/4\}$，分别计算仅对 $x''(t)$ 的大样本系数，以及对 $x''(t)$ 和 $x''(t-1)$ 的系数。将它们与未滤波数据下的系数 $0.80$ 和 $(0.58, 0.34)$，以及真值 $1$ 和 $(1, 0)$ 进行比较。
```

```{solution-start} ae_ex_filter
:class: dropdown
```

以下是一种解法。

对于第 1 部分，用 $k$ 对两个变量同时滤波，会将 $x'$ 的谱密度以及 $y'$ 与 $x'$ 的互谱都乘以 $|\tilde k|^2$。

因此，对 $y''$ 关于 $x''$ 滞后项回归所得的大样本矩矩阵，就是未滤波回归中两个谱都加权 $|\tilde k|^2$ 之后得到的矩矩阵。

等价地说，受约束最小二乘最小化的是 {eq}`ae_sims_criterion`，其中用 $\tilde b'$ 代替 $\tilde b$，用权重 $|\tilde k|^2 S_{x'}$ 代替 $S_x$。

```{code-cell} ipython3
for Δ in (np.pi / 16, np.pi / 8, np.pi / 4):
    k2 = seasonal_keep(Δ).astype(float)
    one = constrained_lag(b_identity_ft, [0], k2)
    two = constrained_lag(b_identity_ft, [0, 1], k2)
    share_out = np.mean(S_w74 * seasonal_keep(Δ)) / np.mean(S_w74)
    print(f"Δ = {Δ / np.pi:.4f}π:  x''(t) alone {one.round(3)}, "
          f"with x''(t-1) {two.round(3)};  share of Var(w) outside the bands "
          f"{share_out:.2f}")
```

滤波使系数朝真值靠拢，频段越宽，靠拢的程度越大。

当 $\Delta = \pi/8$（西姆斯建议用于季度数据的频段宽度）时，仅对 $x''(t)$ 的系数为 $0.95$，两个系数为 $(0.80, 0.18)$，相比之下未滤波数据对应的是 $0.80$ 和 $(0.58, 0.34)$。

该滤波器去除了季节性噪声 $w$ 功率最集中的那些频率，于是 $x''$ 中剩余的变动主要是 $x$ 本身的变动，变量误差偏差因此减小。

但该偏差并未完全消失，因为西姆斯例子中的噪声 $w$ 在季节性频段之外仍有功率：大约 19% 的方差落在宽度为 $\pi/8$ 的频段之外。

西姆斯脚注 6 将 $w$ 的季节峰值宽度定在约 $\pi/5$ 左右，他建议所用频段应比 $x'$ 谱密度中最宽的季节峰值更宽。

当 $\Delta = \pi/4$ 时，$w$ 方差的 12% 落在频段之外。

```{solution-end}
```

```{exercise}
:label: ae_ex_invariance

在上文计算的西姆斯第二个永久收入例子版本中，计量经济学家用 $1 - \kappa(L)\kappa(L^{-1})$ 对数据滤波，但拟合二阶自回归时却把数据当作未调整数据处理。

1. 假设计量经济学家改为对模型也应用同一滤波器，使得模型的谱密度为 $(1 - |\kappa|^2)^2 \sigma^2/|a(e^{-i\omega})|^2$。写出滤波后数据与滤波后模型之间的散度率，并解释为什么它在 $(a, \sigma^2)$ 上的最小化元，就是对不含季节成分的未滤波数据而言的最小化元。
2. 通过数值方法验证该最小化元满足 $a = 0$，并隐含系数 $20$。
```

```{solution-start} ae_ex_invariance
:class: dropdown
```

以下是一种解法。

对于第 1 部分，该滤波器去除了确定性季节成分，因此滤波后数据的谱密度为 $(1 - |\kappa|^2)^2 f$，其中 $f = 1$。

设 $g = \sigma^2/|a(e^{-i\omega})|^2$，则滤波后数据相对于滤波后模型的散度率为

$$
\frac{1}{4\pi} \int_{-\pi}^{\pi} \left[ \frac{(1 - |\kappa|^2)^2 f}{(1 - |\kappa|^2)^2 g} - 1 - \log \frac{(1 - |\kappa|^2)^2 f}{(1 - |\kappa|^2)^2 g} \right] d\omega .
$$

在每个满足 $1 - |\kappa|^2 \neq 0$ 的频率处，方括号内的比值都是 $f/g$，正如 {ref}`ae_ar_section` 中的不变性论证所示。

平方增益 $(1 - |\kappa|^2)^2$ 仅在有限个频率处为零：季节频率 $\pm\pi/2$ 和 $\pi$，以及 $|\kappa|$ 穿越一的 $\pm 0.485\pi$ 附近的两个频率。

一个有限集合的测度为零，因此该散度率等于白噪声 $f$ 对应的 $D(f \| g)$，其在 $a = 0$、$\sigma^2 = 1$ 处为零。

```{code-cell} ipython3
def filtered_criterion(θ, f, gain2):
    """Divergence rate of the filtered data from the filtered AR(2) model."""
    a, σ2 = pacf_to_ar(np.tanh(θ[:2])), np.exp(θ[2])
    return kl_rate(gain2 * f, gain2 * ar_spectrum(a, σ2))


res = minimize(filtered_criterion, [0.3, -0.2, 0.0], args=(f_white, adjust2),
               method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14})
a_fm = pacf_to_ar(np.tanh(res.x[:2]))
print("AR(2) coefficients:", np.round(a_fm, 6) + 0.0,
      f" implied coefficient {implied_coefficient(a_fm):.3f}")
```

对模型和数据同时滤波，消除了仅对数据滤波所产生的偏差。

```{solution-end}
```

```{exercise}
:label: ae_ex_tradeoff

汉森和萨金特在结论部分提到了"估计量的方差与偏差之间的一种权衡，而我们在本文中并未探讨这一维度"。

在季节性示例中，设季节性成分 $s_t$ 的方差是 $x_t$ 方差的一个比例 $q$；讲座中的例子取 $q = 1/4$。

一位计量经济学家对 $T$ 个观测值，在删除宽度为 $\Delta$ 的季节性频段后，用频域最大似然法拟合一个 AR(1)，并想要估计非季节性成分的系数 $0.9$。

用渐近均方根误差 $\sqrt{(\rho^*_K - 0.9)^2 + V_K / T}$ 来度量该估计量的误差，其中 $\rho^*_K$ 是在频段之外频率上的 AR(1) 投影，$V_K$ 是协方差矩阵 {eq}`ae_sandwich` 的 $(1, 1)$ 元素。

1. 对于 $q \in \{0, 0.0005, 0.002, 0.005, 0.01, 0.25\}$ 和 $\Delta \in \{0, \pi/32, \pi/16, \pi/8, \pi/4, \pi/2\}$，计算 $\rho^*_K - 0.9$、$V_K$，以及 $T = 400$ 时的渐近均方根误差，并找出使误差最小的频段宽度。在较窄的几种宽度之间作选择有多重要？
2. 解释为什么当 $q = 0$ 时 $\Delta = 0$ 最优，以及为什么最优频段宽度会随 $q$ 增大而增大。
3. 求出 $T = 1600$ 时的最优频段宽度，并解释为什么它们至少不窄于 $T = 400$ 时的最优宽度。
```

```{solution-start} ae_ex_tradeoff
:class: dropdown
```

以下是一种解法。

```{code-cell} ipython3
f_s_unit = f_s / np.mean(f_s)                     # seasonal shape, unit variance
widths = np.pi * np.array([0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2])
header = "".join(f"{Δ / np.pi:9.4f}π" for Δ in widths)

bias_variance = {}
for q in (0, 0.0005, 0.002, 0.005, 0.01, 0.25):
    f = f_x + q * np.mean(f_x) * f_s_unit
    bias, V = [], []
    for Δ in widths:
        keep = seasonal_keep(Δ)
        a_K, σ2_K = ar_projection(f, 1, keep)
        bias.append(a_K[0] - ρ_x)
        V.append(ar1_covariance(f, a_K[0], σ2_K, keep)[0, 0])
    bias_variance[q] = np.array(bias), np.array(V)

print("     q       " + header)
for q, (bias, V) in bias_variance.items():
    print(f"{q:6.4f} ρ*-0.9" + "".join(f"{np.round(b, 4) + 0.0:10.4f}" for b in bias))
    print(f"{'':6s} V_K   " + "".join(f"{v:10.4f}" for v in V))

for T in (400, 1600):
    print(f"\nT = {T}: asymptotic root mean squared error")
    print("     q" + header + "     best")
    for q, (bias, V) in bias_variance.items():
        rmse = np.sqrt(bias**2 + V / T)
        print(f"{q:6.4f}" + "".join(f"{r:10.4f}" for r in rmse)
              + f"   {widths[rmse.argmin()] / np.pi:.4f}π")
```

当 $q = 0$ 时，AR(1) 设定正确，因此对任何频段宽度，投影都等于 $0.9$，而根据费舍尔信息形式的数据处理不等式，$V_K = \mathcal{I}_K^{-1}$ 会随 $\Delta$ 增大而上升。

此时使用全部频率最优。

当 $q > 0$ 时，使用全部频率的估计量存在偏差，其方差以权重 $(f/g)^2$ 包含了季节峰值处的周期图纵坐标。

删除一个窄频段能去除大部分这类偏差和方差。

进一步拓宽频段能继续降低偏差；对于较小的 $q$，这样做会提高方差，因为被加入的频率携带了关于 $\rho$ 的信息，而设定误差很小。

对于 $T = 400$，最优频段宽度从 $q = 0.0005$ 时的 $\pi/32$ 增大到 $q = 0.002$ 和 $0.005$ 时的 $\pi/16$，再到 $q = 0.01$ 时的 $\pi/8$，而在 $q = 1/4$ 时达到网格中最宽的宽度。

对于 $q \leq 0.005$，均方根误差在各窄频段宽度之间几乎是平的：最优宽度比次优宽度的优势大约在 0.2% 以内，甚至更小。

删除某个频段比频段的具体宽度更重要：在 $q = 0.01$、$T = 400$ 时，使用全部频率的误差为 $0.0287$，而使用最优宽度的误差为 $0.0223$。

对于第 3 部分，偏差平方随频段加宽而减小，且不依赖于 $T$；而方差项 $V_K / T$ 则随 $T$ 增大而缩小。

更大的样本会相对更看重偏差平方，因此对于 $T = 400$ 而言是最优的那个较窄频段，对 $T = 1600$ 而言就不可能是最优的了。

在 $q = 0.005$ 时，最优宽度从 $\pi/16$ 升至 $\pi/8$，差距约为 0.1%；在其他 $q$ 值下则没有变化。

使用全部频率时，误差下降得比使用最优频段时更慢，因为其偏差平方不会缩小：在 $q = 0.01$ 时，$T = 1600$ 下该误差为 $0.0187$，而使用最优宽度为 $0.0112$。

```{solution-end}
```