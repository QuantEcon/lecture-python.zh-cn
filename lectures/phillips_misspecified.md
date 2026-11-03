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
translation:
  title: 最优错误设定信念
  headings:
    Overview: 概述
    An experiment in Bray's lab: 布雷实验室中的一个实验
    An experiment in Bray's lab::Constant-gain adaptive expectations: 常增益适应性预期
    An experiment in Bray's lab::The actual law of motion: 实际运动规律
    Optimal misspecification: 最优错误设定
    Optimal misspecification::The perceived model and its root $\rho$: 感知模型及其根 $\rho$
    Optimal misspecification::Forecast errors: 预测误差
    Optimal misspecification::What kind of equilibrium is this?: 这是哪一种均衡？
    Comparing the true and forecasting models: 比较真实模型与预测模型
    Comparing the true and forecasting models::Impulse responses: 脉冲响应
    How the equilibrium depends on $\rho$: 均衡如何依赖于 $\rho$
    Lessons: 结论
    Exercises: 练习
---

(phillips_misspecified)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 最优错误设定信念

```{index} single: Phillips Curve; Optimal Misspecified Beliefs
```

```{contents} Contents
:depth: 2
```

## 概述

本讲座继续研究菲利普斯曲线的权衡关系。

内容遵循 {cite:t}`Sargent1999` 的第 6 章。

在 {doc}`phillips_adaptive` 中，公众使用一个固定的适应性规则来预测通货膨胀，其中的参数 $\lambda$ 是我们直接选定的。

从某种意义上说，这种做法并不令人满意，而 {doc}`phillips_two_stories` 中的*印证*故事是无法容忍这一点的：描述预期的自由参数正是理性预期本应消除的东西。

在这里，我们朝着"赢得"这个参数迈出了第一步——让代理人自己选择该参数，以拟合由他们自身信念所产生的数据。

我们描述了贯穿本系列讲座的三个概念性问题：

1. 如何构建这样一种均衡——其中代理人共享一个共同的*错误设定的*最小二乘预测模型，
2. 预期如何能够在均衡中贡献出独立的动态，以及
3. 经典的适应性预期方案如何利用二阶矩来近似一阶矩。

为了揭示这些问题，我们暂时搁置菲利普斯曲线，转而使用 {cite:t}`Bray1982` 关于单一商品价格的简单模型——这是研究有限理性的一个常用工具。

{ref}`br_bray` 将布雷的模型呈现为增益为 $1/t$ 的松弛算法，而 {doc}`ls_learning` 则研究了该模型中的最小二乘学习。

我们对布雷的模型加以修改，以说明一种以新的方式融合了理性预期与适应性预期两方面特征的均衡概念，并将在 {doc}`phillips_self_confirming` 中把它应用于菲利普斯曲线。

本讲座的重点是**具有最优但错误设定预测的市场均衡**。

* *最优*意味着预测方案的自由参数是通过（非线性）最小二乘法选择的。
* *错误设定*意味着预测模型在函数形式上是错误的，或者使用了过小的信息集。

一个显著的特征是：真实模型*取决于*代理人的模型是如何被错误设定的——代理人的信念会影响其行为，而这种行为又塑造了他们随后用于拟合的数据。

我们还将看到，均衡取决于代理人模型的一个细节，即他们设定的一个略小于 1 的根 $\rho$，而由此产生的预测增益会随着 $\rho \to 1$ 而消失。

我们将在频域中展开分析，因此让我们先导入所需的库：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
```

## 布雷实验室中的一个实验

遵循 {cite:t}`Bray1982`，假设

```{math}
:label: pm_bray

p_t = a + b \, p_{t+1}^e + u_t ,
```

其中 $u_t$ 是独立同分布的，均值为零，方差为 $\sigma_u^2$，$a > 0$，$b \in (0, 1)$，$p_t$ 是市场价格，$p_{t+1}^e$ 是市场对下一期价格的预期。

按照 {ref}`br_bray` 中的记号，布雷的 $b$ 就是反馈系数 $h'$。

理性预期均衡满足 $p_{t+1}^e = \frac{a}{1-b}$ 且 $p_t = \frac{a}{1-b} + u_t$。

布雷假设 $p_{t+1}^e$ 是过去价格的经验平均值，并证明了当 $0 < b < 1$ 时，这种递减增益方案几乎必然收敛到理性预期值 $\frac{a}{1-b}$。

在过渡期间，状态变量 $p_t^e$ 贡献了动态性，使价格呈现序列相关性，但这些动态是暂时性的：在理性预期均衡处，$p_t$ 是一个常数加上一个序列不相关的冲击。

### 常增益适应性预期

为了让预期能够赋予*持续性*的序列相关性，我们偏离布雷的设定，假设市场具有**常增益**适应性预期：

```{math}
:label: pm_bray2

p_{t+1}^e = C p_t + (1 - C) p_t^e, \qquad |C| < 1 .
```

布雷的方案用 $\frac{1}{t}$ 代替 $C$，从而使 $p_{t+1}^e$ 成为一个样本平均值。

而固定 $C$ 则会对过去的观测值进行*贴现*，这会阻止收敛到理性预期，也会阻止 $p_{t+1}^e$ 收敛到某个常数。

写成分布滞后形式：

```{math}
:label: pm_bray3

p_{t+1}^e = \frac{C}{1 - (1 - C) L} p_t ,
```

其中 $L$ 是滞后算子。

如果价格遵循如下的整合移动平均过程，那么方程 {eq}`pm_bray2` 就是线性最小二乘预测：

```{math}
:label: pm_brayper

p_t = p_{t-1} + \epsilon_t - (1 - C)\epsilon_{t-1} ,
```

因此市场将价格感知为由纯粹的永久性成分和暂时性成分构成。

这正是 {cite:t}`Muth1960` 在 {doc}`bounded_rationality` 中提出的逆最优预测问题的答案，其中默斯的移动平均系数等于 $1 - C$。

等价地，该感知模型是一个*局部水平*模型，即带有白噪声观测的随机游走。

对此，{eq}`pm_bray2` 正是 {ref}`kalman_convergence` 中的稳态卡尔曼滤波，增益 $C$ 通过 $q/r = C^2/(1 - C)$（见 {ref}`pmis_ex4`）与随机游走创新方差和噪声方差之比 $q/r$ 相关联。

默斯问的是：对于什么样的过程，给定的平滑规则才是最优的。

而在这里，过程是由规则生成的，代理人则选择与该过程相匹配的规则。

### 实际运动规律

将信念 {eq}`pm_bray3` 代入 {eq}`pm_bray`，可以看出：当市场以这种方式进行预测时，其行动使价格的*实际*运动规律变为

```{math}
:label: pm_bray4

p_t = \frac{a}{1 - b} + \frac{1}{1 - bC}
      \left[ \frac{1 - (1 - C)L}{1 - \frac{1 - C}{1 - bC} L} \right] u_t
    = \nu + f(L) u_t ,
```

其中 $\nu = \frac{a}{1-b}$，$f(L)$ 的定义与之匹配。

该价格的均值为 $\nu$，谱密度为

$$
F(\omega) = f(e^{i\omega}) f(e^{-i\omega}) \, \sigma_u^2, \qquad \omega \in [-\pi, \pi] .
$$

注意 $F$ 通过 $f$ 依赖于 $C$。

让我们对真实过程进行编码。

```{code-cell} ipython3
class BrayModel:
    """
    Bray's price model with constant-gain adaptive expectations.

    ρ is the root of the agents' perceived ARMA(1,1) model, which stands in
    for the unit root of the integrated moving-average model (see below).
    """

    def __init__(self, a=1.0, b=0.5, σ_u=1.0, ρ=0.995, N=1024):
        self.a, self.b, self.σ_u, self.ρ, self.N = a, b, σ_u, ρ, N
        self.ν = a / (1 - b)
        ω = 2 * np.pi * np.arange(N) / N
        self.ω = ω
        self.z = np.exp(1j * ω)

    def true_spectrum(self, C):
        "Spectral density F(ω) of the actual price process, given belief C."
        b, z = self.b, self.z
        φ = (1 - C) / (1 - b * C)
        scale = 1 / (1 - b * C)
        f = scale * (1 - (1 - C) * z) / (1 - φ * z)
        return np.abs(f)**2 * self.σ_u**2

    def approx_spectrum(self, c, σ_ε2=1.0):
        "Spectral density G(ω) of the agent's approximating model."
        g = (1 - (1 - c) * self.z) / (1 - self.ρ * self.z)
        return np.abs(g)**2 * σ_ε2
```

## 最优错误设定

关于实际运动规律 {eq}`pm_bray4` 的两个事实促使我们对 $C$ 施加均衡约束：

1. 假设价格遵循 {eq}`pm_bray4`，真正的线性最小二乘单步预测规则*并不是*像 {eq}`pm_bray2` 那样的几何分布滞后形式。
2. 即使将预测限定为 {eq}`pm_bray2` 的形式，*最佳的*这类规则也应使 $C$ 求解一个预测误差最小化问题，因此 $C$ 应当是一个结果，而非自由参数。

理性预期均衡本可以修正这两个特征。

{cite:t}`Sargent1999` 第 6 章弱化了均衡概念，其做法"本着精神（但非字面）"遵循了 {cite:t}`LucasPrescott1971` 和 {cite:t}`Brock1972`：保留特征 1 不变（代理人保留错误的函数形式），同时修正特征 2（他们在该形式下选择最优参数）。

设想将单个个体置于一个市场中，市场中所有其他人（"代表性代理人"）都使用 $C$，因此价格遵循 {eq}`pm_bray4`。

这种"大 $C$、小 $c$"的手法，正是 {doc}`rational_expectations` 中"大 $Y$、小 $y$"技巧应用于预测规则（而非决策规则）的版本。

该个体选择 $c$ 以拟合以下形式的最佳模型：

```{math}
:label: pm_bray6

p_t = \frac{1 - (1 - c) L}{1 - L}\epsilon_t ,
```

方法是最小化单步预测误差方差。

### 感知模型及其根 $\rho$

具有 {eq}`pm_bray6` 中精确单位根的模型无法拟合具有均值的数据，其谱密度在零频率处为无穷大。

按照 {cite:t}`Sargent1999`（第 6 章，脚注 4 和脚注 5）的做法，代理人的感知模型用一个略小于一的根 $\rho$ 来替代单位根：

```{math}
:label: pm_rhomodel

p_t = \frac{1 - (1 - c) L}{1 - \rho L}\epsilon_t = g(L)\epsilon_t ,
\qquad 0 < \rho < 1 .
```

这是一个平稳的、均值为零的 ARMA(1,1) 模型。

由于它没有常数项，因此必须利用 $1/(1 - \rho L)$ 接近单位根所带来的持续性，来拟合价格的非零均值 $\nu$。

我们将 $\rho$ 视为感知模型的一个原始要素，是代理人所信奉内容的一部分，而非仅仅是数值上的权宜之计：均衡取决于它，正如 {ref}`pm_rho` 所示。

```{prf:definition} 最优估计映射
:label: pm_bmap

给定 $C$ 及其导致的价格过程 {eq}`pm_bray4`，个体的最优预测参数 $c = B(C)$ 是 {eq}`pm_rhomodel` 中 $c$ 的非线性最小二乘估计量，其中数据由 {eq}`pm_bray4` 生成。
```

对 {eq}`pm_rhomodel` 求逆，可得到感知模型的单步预测误差 $\epsilon_t = g(L)^{-1} p_t$。

将其应用于实际价格，该误差的均值为 $\nu / g(1) = \nu (1 - \rho)/c$，其方差可由谱密度 $F$ 计算得到。

按照汉森和萨金特 {cite}`HansenSargent1993` 的频域方法，最佳近似 $(c, \sigma_\epsilon^2)$ 使以下式子最小化：

```{math}
:label: pm_criterion

A(c, \sigma_\epsilon^2) = \frac{1}{N}\sum_{j=0}^{N-1}
\left\{ \log G(\omega_j, c) + \frac{F(\omega_j)}{G(\omega_j, c)} \right\}
+ \frac{\nu^2}{G(0)} ,
```

其中 $\omega_j = \frac{2\pi j}{N}$，且 $G(\omega, c) = |g(e^{i\omega})|^2 \sigma_\epsilon^2$ 是感知模型的谱密度。

项 $\frac{\nu^2}{G(0)} = \frac{\nu^2 (1 - \rho)^2}{c^2 \sigma_\epsilon^2}$ 对感知模型预测误差均值的平方进行了惩罚；这正是使该模型利用其接近单位根的特性来拟合均值的方式。

将 $\sigma_\epsilon^2$ 集中消去后，令其等于均方预测误差：

```{math}
:label: pm_sigma2

\sigma_\epsilon^2(c) = \frac{1}{N}\sum_{j=0}^{N-1} \frac{F(\omega_j)}{|g(e^{i\omega_j})|^2}
+ \frac{\nu^2 (1 - \rho)^2}{c^2} .
```

根据科尔莫哥洛夫关于满足 $g(0) = 1$ 的可逆滤波器的公式，$\log |g(e^{i\omega})|^2$ 在各频率上的平均值为零。

因此，在 $c$ 上最小化 {eq}`pm_criterion` 等价于最小化 $\sigma_\epsilon^2(c)$：代理人选择能够在其模型 {eq}`pm_rhomodel` 下使均方单步预测误差最小的 $c$。

均值项表明了 $\rho$ 的作用所在：更大的 $c$ 会缩小均值的平方 $\nu^2 (1 - \rho)^2/c^2$，因此 $\rho$ 越接近 1，对较大 $c$ 的拉动作用就越弱。

```{prf:definition} 预测错误设定下的均衡
:label: pm_equilibrium

预测错误设定下的均衡是一个不动点 $C = B(C)$。
```

在这样一个不动点处，代表性代理人是名副其实的："代表性"——单个个体的最优参数与所有人都使用的参数相等。

```{code-cell} ipython3
def forecast_mse(F, ν, c, ρ, z):
    """
    Mean square one-step forecast error σ_ε^2(c) of the perceived model
    g(L) = (1 - (1-c)L)/(1 - ρL), when the true process has mean ν and
    spectral density F on the grid z = exp(iω).  ρ = 1 is allowed.
    """
    W = np.abs((1 - ρ * z) / (1 - (1 - c) * z))**2     # |1/g(e^{iω})|^2
    return np.mean(F * W) + ν**2 * W[0]                # variance + squared mean

def best_estimate(model, C):
    "The best-estimate map c = B(C)."
    F = model.true_spectrum(C)
    res = minimize_scalar(lambda c: forecast_mse(F, model.ν, c, model.ρ, model.z),
                          bounds=(1e-4, 0.99), method='bounded',
                          options={'xatol': 1e-10})
    return res.x

def solve_equilibrium(model, C0=0.3, tol=1e-10, maxit=500):
    "Iterate the best-estimate map to a fixed point."
    C = C0
    for _ in range(maxit):
        C_new = best_estimate(model, C)
        if abs(C_new - C) < tol:
            break
        C = C_new
    return C_new
```

现在我们来求解基准模型的均衡。

```{code-cell} ipython3
bray = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=0.995)
C_star = solve_equilibrium(bray)
print(f"equilibrium belief   C = {C_star:.4f}")
```

对于这些参数，均衡信念约为 $C \approx 0.08$，这再现了 {cite:t}`Sargent1999` 第 6 章中报告的数值。

### 预测误差

三个单步预测误差的标准差共同刻画了这一均衡：

* $\bar\sigma_\epsilon$，即感知模型 {eq}`pm_rhomodel` 在 $c = C$ 处的均方根预测误差，包含其误差均值在内：这是 {eq}`pm_sigma2` 最小化值的平方根。
* 代理人实际使用的规则 {eq}`pm_bray2` 的均方根误差。该规则的权重之和为一，因此其误差均值为零；它是在 $\rho = 1$ 时计算得到的 $\sigma_\epsilon$。
* $\sigma_u / (1 - bC)$，即真实过程 {eq}`pm_bray4` 中新息的标准差，这是任何知晓真实运动规律的预测者所能达到的最小单步误差。

```{code-cell} ipython3
F_star = bray.true_spectrum(C_star)
c_star = best_estimate(bray, C_star)

σ_bar = np.sqrt(forecast_mse(F_star, bray.ν, c_star, bray.ρ, bray.z))
σ_rule = np.sqrt(forecast_mse(F_star, bray.ν, C_star, 1.0, bray.z))
σ_true = bray.σ_u / (1 - bray.b * C_star)

print(f"perceived model (ρ = {bray.ρ}):    {σ_bar:.4f}")
print(f"rule actually used (ρ = 1):     {σ_rule:.4f}")
print(f"true innovation σ_u/(1 - bC):   {σ_true:.4f}")
```

第一个数值就是 {cite:t}`Sargent1999` 第 6 章所报告的 $\bar\sigma_\epsilon$。

书中给出最后一个数值为 $1.0208$，它正是此处所打印数值的平方根。

代理人实际做出的预测仅比可达到的最佳预测略差一些。

(pm_concept)=
### 这是哪一种均衡？

不动点 $C = B(C)$ 在后来的文献中有多个名称，各自强调了其不同的特征。

* 它是一种**受限感知均衡**（{cite:t}`EvansHonkapohja2001`，第 13 章）：预测在一个不包含真相的受限模型类中是最优的。
* 对于给定的 $C$，最优估计 $c$ 是 {cite:t}`White1982` 意义上的**伪真值**，即错误设定模型中最接近真相的参数。与怀特所设定的情形不同，这里的真相本身通过 $C$ 依赖于代理人的参数。
* 在忽略常数项的意义上，准则 {eq}`pm_criterion` 是感知模型相对于真实过程之库尔贝克-莱布勒散度的频域（惠特尔）形式。因此，这一不动点同时也是 {cite:t}`EspondaPouzo2016` 意义上的**伯克-纳什均衡**，预测规则即为代理人的行动。

{doc}`learning_approximation` 针对限定为多项式族的储蓄规则，计算了同类的对象，称之为近似均衡。

在那里，代理人的拟合是对某个基的最小二乘投影；而在这里，是对整个频谱上 ARMA 模型的非线性拟合。

由于 $B$ 的作用类似于最优反应映射，我们可以追问：若代理人将其增益逐步向最优估计调整，他们能否找到这个均衡。

与之相关的微分方程 $dC/d\tau = B(C) - C$ 正是 {ref}`br_one_recursion` 中的 E 稳定性条件：当 $B'(C^*) < 1$ 时，该不动点是 E 稳定的。

```{code-cell} ipython3
h = 1e-4
B_prime = (best_estimate(bray, C_star + h) - best_estimate(bray, C_star - h)) / (2 * h)
print(f"B'(C*) = {B_prime:.3f}")
```

此处 $0 < B'(C^*) < 1$，因此该均衡是 E 稳定的，迭代 $B$ 会单调收敛到它，这正是 `solve_equilibrium` 找到它的方式。

为什么代理人最终没有发现自己的模型是错误的呢？

{cite:t}`KalaiLehrer1993Nash` 运用了 {cite:t}`blackwell1962` 的一个合并定理，确立了预测最终必须与理性预期相合并的条件。

本模型从一开始就关闭了这一机制：代理人的模型对那些在真相下具有正概率的事件赋予了零概率，这违反了合并所要求的绝对连续性条件。

若有足够多的数据，恰当的统计检验最终会告诉代理人应如何改进其预测。

{cite:t}`Kreps1998` 认为，理性的代理人最终会放弃一个被数据所拒绝的错误设定模型；{ref}`pmis_ex3` 估计了在本例中需要多少数据才能做到这一点。

答案取决于真实谱密度与近似谱密度之间相差多远，这正是我们接下来要研究的内容。

## 比较真实模型与预测模型

对于均衡 $C$，我们绘制真实模型与近似模型的均衡谱密度。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 真实价格过程与代理人近似模型的谱密度
    name: fig-mis-spectra
---
G = bray.approx_spectrum(c_star, σ_bar**2)

half = bray.N // 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bray.ω[:half], np.log(F_star[:half]), 'C0', label='true model', lw=2)
ax.plot(bray.ω[:half], np.log(G[:half]), 'C1--', label='forecasting model', lw=2)
ax.set_xlabel(r'angular frequency $\omega$')
ax.set_ylabel('log spectral density')
ax.legend()
plt.show()
```

在最小化 {eq}`pm_criterion` 的过程中，近似模型利用其接近单位根的特性来拟合均值。

低频处两个谱密度之间的巨大差距，反映了近似模型是如何用*二阶*矩的特征（零频率处谱密度的一个尖峰）来拟合*一阶*矩（均值 $\nu$）的。

真实谱密度随频率急剧下降——这正是格兰杰所说的"典型谱形状"（{cite}`Granger1966`）——这揭示了价格中存在相当大的正序列相关性，因为代理人认为价格受永久性冲击影响的信念，使得冲击具有持续性。

### 脉冲响应

我们通过向每个移动平均表示输入一个单位冲击，来比较两个模型的脉冲响应函数。

真实过程 {eq}`pm_bray4` 和感知模型 {eq}`pm_rhomodel` 是由不同的冲击 $u_t$ 和 $\epsilon_t$ 驱动的。

为了进行同类比较，我们用真实响应的冲击效应 $f(0) = 1/(1 - bC)$ 对其进行归一化，使两个模型都对单位的一步预测误差做出响应。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 真实模型与近似模型对单位一步预测误差的脉冲响应
    name: fig-mis-irf
---
def impulse_response(num, den, T=25):
    "IRF of (1 - num L)/(1 - den L)."
    h = np.empty(T)
    h[0] = 1.0
    for k in range(1, T):
        h[k] = den * h[k - 1]
    h[1:] -= num * h[:-1]           # apply the numerator (1 - num L)
    return h

φ = (1 - C_star) / (1 - bray.b * C_star)
irf_true = impulse_response(1 - C_star, φ)              # f(L) / f(0)
irf_approx = impulse_response(1 - c_star, bray.ρ)       # g(L)

lags = np.arange(len(irf_true))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, start in zip(axes, [0, 1]):
    ax.plot(lags[start:], irf_true[start:], 'C0o-', ms=4, label='true model', lw=2)
    ax.plot(lags[start:], irf_approx[start:], 'C1s--', ms=4,
            label='approximating model', lw=2)
    ax.set_xlabel('lag')
    ax.set_ylabel('response')
axes[0].set_title('all lags')
axes[1].set_title('lags 1 and beyond')
axes[0].legend()
plt.tight_layout()
plt.show()

print(f"response at lag 1:  true {irf_true[1]:.3f},  approximating {irf_approx[1]:.3f}")
```

真实模型的脉冲响应确认了价格中的序列相关性。

从滞后一期开始，真实响应就低于近似响应：近似模型在每个时点上都高估了冲击的影响，并且由于其接近单位根的特性，这些影响的消退速度要慢得多——其消退速率为 $\rho$，而非 $\phi = (1 - C)/(1 - bC)$。

{cite:t}`Sargent1999` 第六章指出，近似模型低估了短期影响；但该论断是在未经归一化、直接比较冲击效应的基础上得出的，因而其所反映的仅仅是两种冲击之间的不同尺度。

(pm_rho)=
## 均衡如何依赖于 $\rho$

到目前为止的每一个数值都是用感知根 $\rho = 0.995$ 计算得到的。

下一个代码单元针对更接近于一的 $\rho$ 值重新计算均衡，然后针对规则 {eq}`pm_bray2` 自身的精确单位根 $\rho = 1$ 再次计算。

我们使用了更精细的频率网格，因为感知谱密度在零频率处有一个宽度约为 $1 - \rho$ 的尖峰。

```{code-cell} ipython3
print(f"{'ρ':>7} {'C*':>8} {'weight sum':>11} {'RMSE of rule':>13}")
for ρ in [0.98, 0.99, 0.995, 0.999, 0.9995]:
    m = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=ρ, N=2**14)
    C = solve_equilibrium(m, C0=0.1)
    rmse = np.sqrt(forecast_mse(m.true_spectrum(C), m.ν, C, 1.0, m.z))
    print(f"{ρ:>7} {C:>8.4f} {1 - (1 - ρ) / C:>11.3f} {rmse:>13.4f}")

m1 = BrayModel(a=1.0, b=0.5, σ_u=1.0, ρ=1.0, N=2**14)
C_grid = np.linspace(0.005, 0.4, 40)
ratio = max(best_estimate(m1, C) / C for C in C_grid)
print(f"\nρ = 1: largest value of B(C)/C on the grid = {ratio:.3f}")
print(f"ρ = 1: best c given C = {C_star:.4f} is {best_estimate(m1, C_star):.4f}")
```

随着 $\rho$ 升高趋近于一，均衡增益 $C^*$ 趋于零。

当单位根精确等于一时，对于网格上的每一个 $C$ 都有 $B(C) < C$，因此对 $B$ 进行迭代会使 $C$ 趋于零，唯一的不动点是 $C = 0$。

这正是理性预期均衡：当 $C = 0$ 时预期不再变动，价格为 $p_t = \nu + u_t$。

相应地，最后一列中该规则的均方根误差随着 $\rho \to 1$ 而趋于 $\sigma_u = 1$。

其作用机制在于 {eq}`pm_sigma2` 中的均值项。

一个感知模型若 $\rho < 1$，其预测权重之和为 $1 - (1 - \rho)/c < 1$（即"权重和"一列），因此其误差的均值为 $\nu(1 - \rho)/c$。

较大的 $c$ 会缩小这一偏差，而这种趋向较大 $c$ 的拉力正是使 $C^*$ 远离零的原因；当 $\rho \to 1$ 时，该偏差及其拉力都会消失。

这个构造中存在一种张力。

价格 {eq}`pm_bray4` 是由权重和为一的规则 {eq}`pm_bray2` 生成的，但增益却是通过拟合 $\rho$ 模型来选取的。

如果一个主体选择 $c$ 以最小化其所用规则的实际均方误差，他实际上等于设定了 $\rho = 1$，而输出的最后一行表明，他会选取一个远低于 $C^*$ 的增益（{ref}`pmis_ex2` 通过模拟证实了这一点）。

因此，具有 $C^* > 0$ 的均衡及其伴随的持续序列相关性，对于每一个固定的 $\rho < 1$ 都是存在的，但当 $\rho \to 1$ 时则会消失。

这一问题可追溯到 {cite:t}`Sargent1999` 第六章的脚注4和脚注5，其中引入 $\rho < 1$ 是为了使谱密度具有良好的定义。

这一点在 {doc}`phillips_self_confirming` 中再次变得重要，因为该文献中具有误设信念的均衡所产生的优于纳什均衡的结果正是建立在这一点之上的。

## 结论

这个模型中的代理人具有**有限理性**，这一说法的含义见 {doc}`bounded_rationality`：*理性*描述的是他们使用最小二乘法这一事实，而*有限*描述的则是他们的模型错误设定。

在理性预期下，只有一个模型在起作用。

而在有限理性下，则必须至少有两个模型：有限理性代理人所使用的模型，以及真实的模型。

这两者相互影响——有限理性代理人利用自己的模型去近似真实模型，而真实模型又反映了代理人的决策——并且两者都不同于理性预期模型。

适应性预期模型利用单位根来模仿常数这一独特方式，预示了在 {doc}`phillips_self_confirming` 中发展的菲利普斯曲线模型的一个版本，该版本有助于为计量经济学政策评估正名，且其良好结果同样依赖于 $\rho < 1$。

同样的技巧——用单位根来近似常数——驱动了 {doc}`phillips_learning` 中的*逃逸动态*，其中一个正在学习的政府所估计的分布滞后菲利普斯曲线会逐渐趋向归纳假设，并在相信该假设后，将通货膨胀率下调至拉姆齐水平。

而在 {doc}`phillips_escaping_nash` 的静态模型中，一次逃逸则会使感知到的菲利普斯曲线变为垂直，这自然而然地满足了归纳性质。

## 练习

```{exercise-start}
:label: pmis_ex1
```

均衡信念 $C$ 取决于 {eq}`pm_bray` 中的反馈参数 $b$。

1. 在保持 $a = 1$ 和 $\sigma_u = 1$ 固定的情况下，计算并绘制均衡 $C$ 作为 $b \in \{0.1, 0.2, \ldots, 0.8\}$ 的函数。
2. 在 $a$ 固定的情况下，$b$ 的变化也会改变感知模型必须拟合的均值 $\nu = a/(1 - b)$。通过设定 $a = 2(1 - b)$，重复该计算，同时保持 $\nu = 2$ 固定。
3. $b$ 的影响有多大程度是通过均值起作用的？实际价格的持续性 $\phi = (1 - C)/(1 - bC)$ 如何随 $b$ 变化？

```{exercise-end}
```

```{solution-start} pmis_ex1
:class: dropdown
```

```{code-cell} ipython3
b_grid = np.arange(0.1, 0.85, 0.1)
C_a1 = np.array([solve_equilibrium(BrayModel(a=1.0, b=b)) for b in b_grid])
C_ν2 = np.array([solve_equilibrium(BrayModel(a=2 * (1 - b), b=b)) for b in b_grid])
φ_a1 = (1 - C_a1) / (1 - b_grid * C_a1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(b_grid, C_a1, 'o-', lw=2, label=r'$a = 1$ (mean $\nu$ rises with $b$)')
ax.plot(b_grid, C_ν2, 's--', lw=2, label=r'$\nu = 2$ held fixed')
ax.set_xlabel('feedback parameter $b$')
ax.set_ylabel('equilibrium belief $C$')
ax.legend()
plt.show()

print(f"a = 1:   C from {C_a1[0]:.3f} to {C_a1[-1]:.3f},  "
      f"ν from {1 / (1 - b_grid[0]):.2f} to {1 / (1 - b_grid[-1]):.2f}")
print(f"ν = 2:   C from {C_ν2[0]:.3f} to {C_ν2[-1]:.3f}")
print(f"persistence φ ranges from {φ_a1.min():.3f} to {φ_a1.max():.3f}")
```

在 $a = 1$ 的情况下，更强的反馈会大幅提高均衡增益，但它同时也提高了感知模型必须拟合的均值 $\nu$。

在保持 $\nu$ 固定的情况下，增益的提升幅度要小得多。

因此，$b$ 的大部分影响是通过均值起作用的：更大的均值使感知模型的偏差 $\nu(1 - \rho)/c$ 代价更高，因此代理人会选择更大的增益。

实际价格的持续性 $\phi$ 几乎没有变化，因此更强的反馈并不会使价格过程的持续性明显减弱。

```{solution-end}
```

```{exercise-start}
:label: pmis_ex2
```

最优估计映射是在频域中，基于感知模型 {eq}`pm_rhomodel` 计算得到的。

请检验该均衡在时域中所隐含的含义。

以 $C = C^*$ 模拟市场 {eq}`pm_bray`–{eq}`pm_bray2` 一百万期。

对于一组 $c$ 的取值，计算预测 $p^e_{t+1}(c) = c\, p_t + (1 - c) p^e_t(c)$ 对 $p_{t+1}$ 的均方误差。

哪个 $c$ 能使其最小化？

将其与 $C^*$ 以及 {ref}`pm_rho` 中给出的精确单位根最优估计进行比较。

```{exercise-end}
```

```{solution-start} pmis_ex2
:class: dropdown
```

将 {eq}`pm_bray2` 代入 {eq}`pm_bray`，可得 $p_t = [a + b(1 - C) p^e_t + u_t]/(1 - bC)$，进而 $p^e_{t+1} = \phi\, p^e_t + C(a + u_t)/(1 - bC)$，其中 $\phi = (1 - C)/(1 - bC)$。

```{code-cell} ipython3
rng = np.random.default_rng(0)
T, burn = 1_000_000, 10_000
a, b, C = bray.a, bray.b, C_star
u = rng.standard_normal(T)

φ = (1 - C) / (1 - b * C)
pe_next = lfilter([C / (1 - b * C)], [1, -φ], a + u, zi=[φ * bray.ν])[0]
pe = np.concatenate([[bray.ν], pe_next[:-1]])                 # p^e_t
p = (a + b * (1 - C) * pe + u) / (1 - b * C)

def rmse(c):
    "Root mean square error of the forecast c p_t + (1-c) p^e_t of p_{t+1}."
    f = lfilter([c], [1, -(1 - c)], p, zi=[(1 - c) * bray.ν])[0]
    return np.sqrt(np.mean((p[burn + 1:] - f[burn:-1])**2))

c_grid = np.linspace(0.005, 0.15, 59)
errors = np.array([rmse(c) for c in c_grid])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(c_grid, errors, lw=2)
ax.axvline(C_star, color='k', ls='--', lw=1, label='$C^*$')
ax.set_xlabel('gain $c$')
ax.set_ylabel('RMSE of the forecast')
ax.legend()
plt.show()

print(f"RMSE-minimizing c = {c_grid[errors.argmin()]:.3f},  C* = {C_star:.4f}")
```

使实际预测误差最小化的增益远低于 $C^*$，且接近 {ref}`pm_rho` 中的精确单位根最优估计。

如果一个人以实际预测误差来评判规则 {eq}`pm_bray2`，他不会选择 $C^*$。

$C^*$ 仅在感知模型 {eq}`pm_rhomodel` 内部才是最优反应，而该模型中的均值惩罚项并不适用于代理人实际使用的规则，因为该规则的误差均值为零。

```{solution-end}
```

```{exercise-start}
:label: pmis_ex3
```

代理人需要多少数据才能察觉到他们的模型是错误的？

对于高斯过程，真实模型 {eq}`pm_bray4` 相对于感知模型 {eq}`pm_rhomodel`（在 $c = B(C^*)$ 和拟合的 $\bar\sigma_\epsilon^2$ 处）的每个观测值的期望对数似然比，即为库尔贝克-莱布勒散度率。

1. 利用集中化准则和科尔莫戈罗夫公式，证明该速率等于 $\log(\bar\sigma_\epsilon / \sigma_{\rm true})$，其中 $\sigma_{\rm true} = \sigma_u/(1 - bC^*)$。
2. 计算该速率，以及期望对数似然比达到 3（即似然比约为 20）所需的观测次数。
3. 通过模拟对数似然比来确认该速率。

```{exercise-end}
```

```{solution-start} pmis_ex3
:class: dropdown
```

均值为零、谱密度为 $G$ 的高斯模型相对于均值为 $\nu$、谱密度为 $F$ 的高斯过程的库尔贝克-莱布勒散度率为

$$
\frac{1}{2}\left\{ \frac{1}{N}\sum_j \left[\frac{F(\omega_j)}{G(\omega_j)} - 1 - \log\frac{F(\omega_j)}{G(\omega_j)}\right] + \frac{\nu^2}{G(0)} \right\} .
$$

在拟合的 $\bar\sigma_\epsilon^2$ 处，各项 $\frac{1}{N}\sum_j F/G + \nu^2/G(0)$ 之和为一。

根据科尔莫戈罗夫公式，$\log G$ 的平均值为 $\log \bar\sigma_\epsilon^2$，而 $\log F$ 的平均值为 $\log \sigma_{\rm true}^2$，即真实过程的对数一步预测方差。

因此该速率为 $\frac{1}{2}\log(\bar\sigma_\epsilon^2/\sigma_{\rm true}^2) = \log(\bar\sigma_\epsilon/\sigma_{\rm true})$。

```{code-cell} ipython3
kl = np.log(σ_bar / σ_true)
print(f"KL rate = {kl:.4f} nats per observation")
print(f"observations for an expected log-likelihood ratio of 3: {3 / kl:.0f}")

# simulate: true one-step errors are u_t/(1-bC); the perceived model's are g(L)^{-1} p_t
rng = np.random.default_rng(1)
T, burn = 200_000, 10_000
a, b, C = bray.a, bray.b, C_star
u = rng.standard_normal(T)
φ = (1 - C) / (1 - b * C)
pe_next = lfilter([C / (1 - b * C)], [1, -φ], a + u, zi=[φ * bray.ν])[0]
pe = np.concatenate([[bray.ν], pe_next[:-1]])
p = (a + b * (1 - C) * pe + u) / (1 - b * C)

e_true = u / (1 - b * C)
e_model = lfilter([1, -bray.ρ], [1, -(1 - c_star)], p)
llr = (-np.log(σ_true) - e_true**2 / (2 * σ_true**2)) \
    - (-np.log(σ_bar) - e_model**2 / (2 * σ_bar**2))
print(f"simulated mean log-likelihood ratio = {llr[burn:].mean():.4f}")
```

该散度大约为每个观测值 $0.015$ 纳特（nats），因此大约需要两百个观测值，数据才能以约 20 的似然比支持真实模型，而这还是针对一个已经知道应检验哪个备择假设的统计学家而言。

```{solution-end}
```

```{exercise-start}
:label: pmis_ex4
```

{cite:t}`Muth1960` 证明了规则 {eq}`pm_bray2` 对于带噪声观测的随机游走而言是最优的。

考虑局部水平模型 $p_t = \mu_t + \eta_t$，$\mu_t = \mu_{t-1} + w_t$，其中 $\mathbb{V}[w_t] = q$ 且 $\mathbb{V}[\eta_t] = r$。

1. 证明该模型的稳态卡尔曼增益 $K$ 满足 $q/r = K^2/(1 - K)$，并且此时卡尔曼滤波对 $p_{t+1}$ 的预测具有 {eq}`pm_bray2` 的形式，其中 $C = K$。
2. 设定 $q/r = C^{*2}/(1 - C^*)$，对 {ref}`kalman_convergence` 中的里卡蒂方程进行迭代直至其不动点，并确认 $K = C^*$。
3. 这一隐含的信噪比说明了代理人是如何感知价格的？

```{exercise-end}
```

```{solution-start} pmis_ex4
:class: dropdown
```

取 $A = G = 1$，$Q = q$，$R = r$，{doc}`kalman` 中关于 $\mu_t$ 的一步前瞻预测方差 $\Sigma$ 的里卡蒂方程为 $\Sigma_{t+1} = \Sigma_t - \Sigma_t^2/(\Sigma_t + r) + q$。

在其不动点处，$\Sigma r/(\Sigma + r) = \Sigma - q$，且增益为 $K = \Sigma/(\Sigma + r)$。

因此 $(1 - K)\Sigma = \Sigma - q$，于是 $\Sigma = q/K$，代入 $K = \Sigma/(\Sigma + r)$ 得到 $q/r = K^2/(1 - K)$。

对 $p_{t+1}$ 的预测即是对 $\mu_{t+1}$ 的预测，它等于对 $\mu_t$ 的滤波估计：$\hat\mu_{t+1} = \hat\mu_t + K(p_t - \hat\mu_t) = K p_t + (1 - K)\hat\mu_t$，这正是 {eq}`pm_bray2`，其中 $C = K$。

```{code-cell} ipython3
r = 1.0
q = r * C_star**2 / (1 - C_star)
Σ = 1.0
for _ in range(10_000):
    Σ = Σ - Σ**2 / (Σ + r) + q
K = Σ / (Σ + r)
print(f"q/r = {q / r:.5f},  steady-state Kalman gain K = {K:.4f},  C* = {C_star:.4f}")
```

均衡信念隐含的信噪比低于百分之一。

代理人将价格感知为几乎完全是围绕一个缓慢漂移的水平所产生的暂时性噪声，而这一缓慢漂移的水平正是他们的模型用来模拟恒定均值 $\nu$ 的方式。

```{solution-end}
```
