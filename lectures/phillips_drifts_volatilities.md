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
  title: 漂移与波动率
  headings:
    Overview: 概览
    Bad policy or bad luck?: 政策不当还是运气不好？
    A VAR with drifting coefficients and stochastic volatility: 一个系数漂移且波动率随机变化的 VAR
    The data: 数据
    Priors: 先验
    Simulating the posterior: 模拟后验分布
    Simulating the posterior::Imposing stability: 施加稳定性约束
    What the data say: 数据揭示了什么
    What the data say::The rate and structure of drift: 漂移的速率与结构
    What the data say::The evolution of volatility: 波动率的演变
    What the data say::Core inflation and the natural rate: 核心通货膨胀与自然失业率
    What the data say::Inflation persistence: 通货膨胀的持续性
    What the data say::Monetary policy activism: 货币政策的积极程度
    What the data say::The disinflation and the calm: 反通胀与平静期
    Why classical tests missed the drift: 为什么经典检验未能发现漂移
    Bad policy or bad luck? A verdict: 政策不当还是运气不好？一个结论
    'Appendix: the sampler in detail': 附录：采样器详解
    'Appendix: the sampler in detail::The coefficient block': 系数区块
    'Appendix: the sampler in detail::The volatility block': 波动率区块
    'Appendix: the sampler in detail::Monte Carlo error': 蒙特卡洛误差
    Exercises: 练习
---

(phillips_drifts_volatilities)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img
                    style="width:250px;display:inline;"
                    width="250px"
                    src="https://assets.quantecon.org/img/qe-menubar-logo.svg"
                    alt="QuantEcon"
                >
        </a>
</div>
```

# 漂移与波动率

```{index} single: Phillips Curve; Drifts and Volatilities
```

```{contents} Contents
:depth: 2
```

## 概览

本节的讲座讲述了一个关于政府对菲利普斯曲线的*模型*、以及由此引发的*政策*如何随时间*漂移*的故事。

在 {doc}`phillips_learning` 和 {doc}`phillips_escaping_nash` 中，一个通过常增益最小二乘法不断重新拟合近似菲利普斯曲线的政府，反复从一个糟糕的 {doc}`自我实现均衡 <phillips_self_confirming>` 中逃逸，随后又被拉回该均衡。

{doc}`phillips_priors` 展示了政府对漂移系数的先验信念如何塑造这些动态，而 {doc}`phillips_policy_evaluation` 则通过最大似然法将这些适应性模型与美国战后数据进行了比对。

那次比对是*结构性*的：它估计了一个关于学习型政府的特定模型。

本讲座则退回到一个*简化形式*的问题，而上述每一个模型都必须回答这个问题。

它研究了 {cite:t}`CogleySargent2005` 一文，该文提出了一个看似简单的问题：

> 当我们审视美国战后关于通货膨胀、失业和利率的时间序列时，是否能看到动态发生了*漂移*的证据？

蒂姆·科格利（Tim Cogley）和托马斯·萨金特开始这项工作，是作为《征服》一书 {cite}`Sargent1999` 以及 {cite:t}`ChoWilliamsSargent2002` 的实证配套研究；后者所刻画的反复逃逸过程会产生非线性现象，计量经济学家会将其视为系数的漂移。

本讲座回应了 {cite:t}`Sims2001comment` 和 {cite:t}`Stock2001comment` 对早期论文 {cite}`CogleySargent2001` 所作的深入评论，并直面了来自 {cite:t}`Sims1999` 和 {cite:t}`BernankeMihov1998` 的证据，这些证据表明向量自回归的系数*并未*发生变化。

这一证据关系到贯穿本节的核心问题：

*20世纪70年代的大通胀及其在80年代的平息，究竟是政策不当的故事，还是运气不好的故事？*

要让数据回答这个问题，我们需要一个足够灵活、能同时容纳*两种*答案的统计模型。

这个模型是一个*贝叶斯向量自回归模型，其系数按随机游走漂移，其冲击方差按随机波动率演化*。

这正是 {cite:t}`Sims1988` 和 {cite:t}`Chung1990` 赋予其模型政府的设定：对被认为按随机游走漂移的回归系数应用卡尔曼滤波（{ref}`pbr_compliment`）。

在那里，政府运行该滤波器以学习菲利普斯曲线；而在这里，计量经济学家运行它来了解政府。

拟合该模型需要一种马尔可夫链蒙特卡洛算法，该算法结合了 {doc}`卡尔曼滤波 <kalman>`、{cite:t}`CarterKohn1994` 的前向滤波/后向抽样平滑器，以及 {cite:t}`Jacquier1994` 的随机波动率抽样器。

伴随形式的向量自回归在 {doc}`linear_models` 和 {doc}`var_subsets` 中有所讨论，卡尔曼滤波则见于 {doc}`kalman`。

QuantEcon 的其他讲座均未涉及卡尔曼*平滑器*，因此本讲座在抽样器中引入了它，{ref}`csdv2_ex1` 说明了它在此处的重要性。

我们将模型拟合到科格利和萨金特所用的样本数据，该样本截止于2000年；{doc}`phillips_drifts_volatilities_2` 则使用又一个四分之一世纪的新数据对其进行了重新估计。

让我们从一些导入语句和数据的 URL 开始。

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import linalg
from scipy.special import expit
from threadpoolctl import threadpool_limits

# 本讲座中的矩阵规模较小，因此额外的 BLAS 线程只会增加开销
threadpool_limits(1)

data_url = 'https://github.com/QuantEcon/data-lectures/raw/main/lectures/NEWQDATA.csv'
```

## 政策不当还是运气不好？

有两种颇为可信的观点相互竞争，用以解释美国的大通胀。

**政策不当**的观点是本节乃至《征服》一书 {cite}`Sargent1999` 中反复演绎的观点。

亚瑟·伯恩斯（Arthur Burns）对经济的某种*模型*、他的*耐心*，或他无法*承诺*一条更好的规则，导致美联储以一种造成美国和平时期历史上最严重通胀的方式来实施货币政策；而一个改进的模型、更多的耐心，或更强的纪律性，则使保罗·沃尔克（Paul Volcker）得以将其平息 {cite}`DeLong1997,Taylor1997comment`。

{doc}`phillips_two_stories` 开篇提出的“胜利”与“昭雪”这两种故事，都是这一观点的变体：它们的分歧在于美联储的模型是如何改进的，而非政策是否发生了变化。

按照这种观点，20世纪70年代到80年代之间发生变化的，是政策的*系统性部分*，即美联储的利率制定对通货膨胀和失业作出反应的方式。

**运气不好**的观点否定了上述两种故事。

区分伯恩斯时代和沃尔克时代的，并非他们的模型或政策，而是*冲击到*经济体上的各种*扰动*。

按照这种观点，经济简化形式描述的*系数*基本保持不变，改变的是扰动的*大小*，即*波动率*。

经典检验支持了第二种观点。

{cite:t}`Sims1999` 和 {cite:t}`BernankeMihov1998` 均未能拒绝 VAR 系数随时间不变的假设，同时发现创新方差确实发生了变化。

{cite:t}`SimsZha2006` 以不同的方法得出了类似的结论。

他们通过比较马尔可夫转换模型的后验拟合度发现，拟合效果最好的模型只允许扰动的方差发生转换。

当系统性系数也被允许转换时，拟合效果最好的模型只改变了货币政策规则，而且估计出的政策机制之间的差异太小，不足以解释通胀的起落 {cite}`Sargent2008`。

马尔可夫转换和下文所用的随机游走波动率，以不同的方式描述波动率的变化。

隐马尔可夫链允许*跳跃*，但若没有吸收状态，它会在同样几种配置之间永远转换；若有吸收状态，它则只允许发生一次永久性转变。

随机游走波动率必须连续调整，但它允许出现*反复出现的、永久性*的转变，例如大通胀之后紧接着大缓和。

{cite:t}`Primiceri2005` 拟合了一个与下文非常接近的时变 VAR 模型，其中系数漂移并带有随机波动率，并采用了偏向缓慢漂移的先验。

他发现系统性货币政策确实发生了变化，但这些变化对经济其余部分的影响很小，这使他的结论更偏向运气不好一方 {cite}`Sargent2008`。

我们该如何加以区分？

一个系数恒定、波动率恒定的 VAR 可以生成异常大的实现冲击，但它无法表示其方差随时间发生的系统性变化。

一个系数漂移但波动率恒定的模型，也可能把波动率的变化误认为系数漂移，这正是 {cite:t}`Stock2001comment` 对2001年那篇论文的质疑所在。

因此，科格利和萨金特构建了一个能同时容纳*这两种*渠道的模型，并让贝叶斯后验来判定数据究竟需要多少这两种成分。

(csdv-model)=
## 一个系数漂移且波动率随机变化的 VAR

设变量按名义利率、变换后的失业率、通货膨胀的顺序排列，

$$
y_t = \begin{bmatrix} i_t & u_t & \pi_t \end{bmatrix}^\top .
$$

这里的 $u_t$ 并非原始的失业率，而是其 logit 变换，我们将在下面的数据部分定义这一变换。

量测方程是一个滞后两期、系数随日期变化的 VAR，

```{math}
:label: csdv_measurement
y_t = X_t^\top \theta_t + \varepsilon_t,
\qquad
X_t^\top = I_3 \otimes \begin{bmatrix} 1 & y_{t-1}^\top & y_{t-2}^\top \end{bmatrix}.
```

每个方程都有一个截距项和六个滞后系数，因此 $\theta_t$ 包含
$3(1+2\times 3)=21$ 个元素。

一个滞后两期的 VAR 可以通过把 $y_t$ 和
$y_{t-1}$ 堆叠成一个向量，改写为滞后一期的系统；与这个堆叠向量相乘的矩阵称为**伴随矩阵**，改写后的系统称为**伴随形式**的 VAR（参见 {doc}`linear_models`、{doc}`var_subsets`）。

下面的函数根据堆叠的系数构造伴随矩阵。

```{code-cell} ipython3
n_variables = 3
n_lags = 2
n_regressors = 1 + n_variables * n_lags
n_coefficients = n_variables * n_regressors


def companion_matrix(θ):
    """Return the intercept and companion matrix for one coefficient vector."""
    rows = np.asarray(θ, dtype=float).reshape(n_variables, n_regressors)
    intercept = np.r_[rows[:, 0], np.zeros(n_variables)]
    companion = np.zeros((n_variables * n_lags, n_variables * n_lags))
    companion[:n_variables] = rows[:, 1:]
    companion[n_variables:, :n_variables] = np.eye(n_variables)
    return intercept, companion
```

系数向量遵循一个无漂移的随机游走，

```{math}
:label: csdv_transition
\theta_t = \theta_{t-1} + v_t,
\qquad
v_t \sim N(0,Q).
```

漂移协方差 $Q$ 与 {doc}`phillips_priors` 中的漂移协方差 $V$ 是近亲。

在那里，$V$ 是*政府*对菲利普斯曲线漂移速度的先验，它决定了政府卡尔曼滤波器的增益；而在这里，$Q$ 是*计量经济学家*模型中需要被估计的一个参数。

对于给定的漂移协方差，卡尔曼增益不会消失，因此滤波器的行为类似于 {doc}`phillips_learning` 中的恒定增益最小二乘法，这是一种属于 {ref}`br_one_recursion` 所研究范畴的随机逼近算法。

{ref}`olg_ex7` 探讨了这样的增益能从数据中被估计得多好，而 {ref}`csdv2_ex2` 则计算出了这一模型的估计结果所隐含的增益。

当 {cite:t}`SargentWilliamsZha2006` 通过要求一个学习型政府的政策拟合美国通货膨胀数据来估计 $V$ 时，他们需要一个较大的 $V$，即一个"严重贴现过去数据"的政府 {cite}`Sargent2008`；{cite:t}`CarboniEllison2009` 则通过要求政府的信念与美联储工作人员的绿皮书预测相匹配，将其缩小了。

当伴随矩阵的每一个特征值都严格位于单位圆内时，伴随系统是稳定的。

对于 $AR(1)$ 过程而言，这就是 $|\rho|<1$；等价地说，就是 $1-\rho z$ 的零点位于单位圆之外。

科格利和萨金特使用如下截断先验来排除爆炸性路径

```{math}
:label: csdv_stability
p(\theta^T,Q) \propto I(\theta^T) f(\theta^T \mid Q) f(Q),
```

其中，若伴随矩阵在每个日期都稳定，则 $I(\theta^T)=1$，否则 $I(\theta^T)=0$。

他们这样做的理由是经济学上的。

一个有目的地选择其规则的美联储，若其损失函数会对通货膨胀的方差进行惩罚，就绝不会选择一个使通货膨胀中留有单位根的规则，因为那将带来无穷大的损失。

VAR 的根并非仅由菲利普斯曲线决定；它们还取决于政策规则。

在任意规则下，这些根可能位于单位圆内、圆上或圆外，但在最优或接近最优的规则下，它们是稳定的：当某个冲击使通货膨胀偏离其目标时，选择不当的规则可能会任由其漂移，而选择得当的规则则会将其拉回。

{doc}`phillips_adaptive` 中的费尔普斯问题就是一个例子：即使菲利普斯曲线本身具有索洛-托宾单位和形式，最优规则也会将通货膨胀引导至一个静止点。

这一限制也使 $Q$ 的边际先验向那些不太可能生成爆炸性系数路径的取值倾斜。

下面的代码对一整条轨迹进行稳定性检验。

```{code-cell} ipython3
def companion_roots(θ_path):
    """Return the companion roots at every date of a (21, T) path."""
    rows = np.asarray(θ_path, dtype=float).reshape(n_variables, n_regressors, -1)
    companions = np.zeros((rows.shape[2], 6, 6))
    companions[:, :n_variables] = rows[:, 1:].transpose(2, 0, 1)
    companions[:, n_variables:, :n_variables] = np.eye(n_variables)
    return np.linalg.eigvals(companions)


def is_stable(θ_path):
    """Test whether every companion root lies strictly inside the unit circle."""
    return bool(np.max(np.abs(companion_roots(θ_path))) < 1)
```

简化形式的创新（新息）协方差按下式随时间变化，

```{math}
:label: csdv_covariance
\varepsilon_t = R_t^{1/2}\xi_t,
\qquad
\xi_t \sim N(0,I_3),
\qquad
R_t = B^{-1} H_t B^{-1\prime},
```

其中

$$
B =
\begin{bmatrix}
1 & 0 & 0 \\
\beta_{21} & 1 & 0 \\
\beta_{31} & \beta_{32} & 1
\end{bmatrix},
\qquad
H_t = \operatorname{diag}(h_{1t},h_{2t},h_{3t}).
$$

对角元素 $h_{it}$ 使每个正交化冲击的大小能够随时间而消长。

矩阵 $B$ 对 $\varepsilon_t$ 进行正交化，但它并非一种识别方案。

接下来的两个函数分别构造 $B$ 和 $R_t$。

```{code-cell} ipython3
def b_matrix(β):
    """Construct B from β_21, β_31, and β_32."""
    matrix = np.eye(n_variables)
    matrix[1, 0], matrix[2, 0], matrix[2, 1] = β
    return matrix


def innovation_covariance(h, β):
    """Construct R_t = B^{-1} H_t B^{-1}' from one vector of variances."""
    inverse = np.linalg.inv(b_matrix(β))
    return inverse @ np.diag(h) @ inverse.T
```

每个对角波动率都是一个几何随机游走，

```{math}
:label: csdv_volatility
\log h_{it} = \log h_{i,t-1} + \sigma_i \eta_{it},
\qquad
\eta_{it} \sim N(0,1).
```

标准化的量测创新、系数创新和波动率创新是相互独立的。

设定 $Q=0$ 会产生系数恒定、波动率漂移的模型，而固定 $H_t$ 则会产生系数漂移、波动率恒定的模型。

后验包含完整路径 $\theta^T$ 和 $H^T$，以及 $Q$、
$\beta=(\beta_{21},\beta_{31},\beta_{32})$ 和 $\sigma=(\sigma_1,\sigma_2,\sigma_3)$。

## 数据

我们使用 {cite:t}`CogleySargent2005` 的截至2000年第四季度的美国季度数据集。

通货膨胀是经季节调整的全体城市消费者 CPI 的对数差分，采样点为每季度的第三个月。

失业率是经季节调整的民用失业率的季度平均值，以 $0.01\log[u/(1-u)]$ 的形式进入 VAR，这是一种将有界的比率映射为无约束变量的 logit 变换。

名义利率的起点是三个月期国库券利率 $i$（以年化百分比表示，并在每季度的第一个月对每日观测值取平均），其进入模型的形式为 $i/(400 - i)$，即以面值的 $i/400$ 作为贴现率购买的票据的季度回报率。

数据始于1948年第二季度，因为第一个通货膨胀观测值本身已是差分数据，所以两期 VAR 滞后使得1948年第四季度成为第一个可用的回归日期。

截至1958年第四季度的观测数据用于校准先验，而1959年第一季度至2000年第四季度的数据构成估计样本。

```{code-cell} ipython3
def prepare_data(source):
    """Transform a quarterly table and construct the VAR(2) data."""
    table = pd.read_csv(source) if isinstance(source, str) else source
    ur = table['ur'].to_numpy(dtype=float)
    raw_y = np.column_stack([
        table['y3'].to_numpy(dtype=float),       # nominal interest
        0.01 * np.log(ur / (1 - ur)),            # logit unemployment
        table['dp'].to_numpy(dtype=float),       # inflation
    ])
    regressors = np.ones((len(raw_y) - n_lags, n_regressors))
    for lag in range(1, n_lags + 1):
        left = 1 + n_variables * (lag - 1)
        regressors[:, left:left + n_variables] = raw_y[n_lags - lag:-lag]
    targets = raw_y[n_lags:]
    dates = table['date'].to_numpy(dtype=float)[n_lags:]
    n_training = 4 * 11 - n_lags - 1          # 1948Q4--1958Q4
    return {
        'raw_dates': table['date'].to_numpy(dtype=float),
        'raw_y': raw_y,
        'prior_y': targets[:n_training],
        'prior_x': regressors[:n_training],
        'dates': dates[n_training:],
        'y': targets[n_training:],
        'x': regressors[n_training:],
    }


data = prepare_data(data_url)
```

让我们用熟悉的经济学单位来查看数据。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Observed $\pi_t$, $u_t$, and $i_t$, 1948Q2--2000Q4
    name: fig-csdv-historical-data
---
dates_raw = data['raw_dates']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(dates_raw, 400 * data['raw_y'][:, 2], lw=2)
axes[0].set_ylabel('inflation (annual %)')
axes[1].plot(dates_raw, 100 * expit(100 * data['raw_y'][:, 1]), lw=2)
axes[1].set_ylabel('unemployment (%)')
interest = data['raw_y'][:, 0]
axes[2].plot(dates_raw, 400 * interest / (1 + interest), lw=2)
axes[2].set_ylabel('interest (annual %)')
axes[2].set_xlabel('year')
plt.tight_layout()
plt.show()
```

通货膨胀和名义利率在20世纪70年代一同上升，并在1980年前后达到峰值；失业率则在反通胀开始之后才达到峰值；而这三个序列在90年代都变得更为平稳。

这些观察结果定位了这一历史事件，但无法说明究竟是系统性动态发生了变化，还是出现了异常大的冲击才导致了这一现象——而这正是本模型所要考察的区别。

## 先验

各参数的先验彼此独立，且被有意设定得较为宽松，用科格利和萨金特的话说，就是要“让数据自由发声”。

它们是根据一个拟合于训练样本的时不变 VAR 校准得到的。

初始系数先验是一个稳定的、截断的高斯分布，

$$
p(\theta_0) \propto I(\theta_0)N(\bar\theta,\bar P),
$$

其中 $\bar\theta$ 和 $\bar P$ 分别是一个拟合于1948—1958年的常系数似不相关回归（SUR）的点估计和渐近协方差。

由于三个方程具有完全相同的回归变量，SUR 估计等于逐方程的OLS估计。

系数漂移的协方差具有如下逆威沙特先验

```{math}
:label: csdv_q_prior
Q \sim IW_{21}\left(T_0,T_0\bar Q\right),
\qquad
T_0 = 22,
\qquad
\bar Q = \gamma^2 \bar P,
\qquad
\gamma^2 = 3.5\times 10^{-4},
```

其中第一个参数是自由度，第二个参数是尺度矩阵。

这里的 $\gamma^2$ 是科格利和萨金特的先验尺度参数，并非学习增益；在有限理性讲座中，$\gamma$ 表示增益（见 {ref}`br_notation`）。

由于 $T_0$ 仅比 $\theta_t$ 的维数大一，这个先验虽然合式（proper），但没有有限均值。

因此矩阵 $\bar Q$ 是一个保守的尺度校准值，意味着漂移较小，而非 $Q$ 的期望，而且这个先验的宽松性使得数据得以大幅偏离它。

其余的先验分布为

$$
\begin{aligned}
\log h_{i0} &\sim N(\log \bar R_{ii},10), \\
\beta &\sim N(0,10000\, I_3), \\
\sigma_i^2 &\sim IG\left(\frac{1}{2},\frac{0.01^2}{2}\right),
\end{aligned}
$$

其中 $\bar R$ 是来自训练样本回归的残差协方差。

下面的函数汇总了这些超参数。

```{code-cell} ipython3
def calibrate_prior(model_data, γ_squared=3.5e-4):
    """Calibrate every prior from a constant-coefficient training-sample VAR."""
    y, x = model_data['prior_y'], model_data['prior_x']
    xx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xx_inverse @ x.T @ y          # SUR = equation-by-equation OLS
    residuals = y - x @ coefficients
    r_bar = np.cov(residuals, rowvar=False, ddof=1)
    p_bar = np.kron(r_bar, xx_inverse)
    T0 = n_coefficients + 1
    return {
        'θ_mean': coefficients.T.reshape(-1),
        'θ_covariance': p_bar,
        'q_center': γ_squared * p_bar,          # Q bar
        'q_scale': T0 * γ_squared * p_bar,      # T0 Q bar
        'q_degrees_freedom': T0,
        'log_h_mean': np.log(np.diag(r_bar)),
        'log_h_variance': 10.0,
        'β_variance': 10_000.0,
        'σ_degrees_freedom': 1.0,
        'σ_scale': 0.01**2,
        'γ_squared': γ_squared,
    }


prior = calibrate_prior(data)
trace_q_bar = np.trace(prior['q_center'])
assert is_stable(prior['θ_mean'])

pd.Series({
    'T0': prior['q_degrees_freedom'],
    'γ squared': prior['γ_squared'],
    'tr(Q bar)': trace_q_bar,
}, name='value').to_frame()
```

训练样本估计值 $\bar\theta$ 本身是稳定的，因此马尔可夫链可以从该处开始。

(csdv-sampler)=
## 模拟后验分布

我们通过循环遍历 {cite:t}`CogleySargent2005` 使用的五个区块来模拟后验分布；遍历全部五个区块一次称为一次*扫描*（sweep）。

1. **系数路径** $\theta_0,\ldots,\theta_T$，给定 $Q$ 和协方差 $R^T$：先正向运行卡尔曼滤波，然后由卡特-科恩（Carter--Kohn）后向过程依次抽取 $\theta_T$ 以及每个在 $\theta_{t+1}$ 抽样条件下的 $\theta_t$。
   这正是将卡尔曼*平滑器*用作采样器。
2. **漂移协方差** $Q$，给定路径：进行逆威沙特（inverse-Wishart）抽样，其尺度参数由 $T_0\bar Q$ 加上 $T$ 个增量外积 $v_tv_t^\top$ 之和构成。
3. **波动率尺度** $\sigma_i^2$，给定波动率路径：进行逆伽马（inverse-gamma）抽样。
4. **协方差参数** $\beta$，给定 VAR 残差和 $H^T$：两次高斯回归。
5. **波动率路径** $H^T$：使用 {cite:t}`Jacquier1994` 的单点 Metropolis 步骤，逐个日期进行。

任何固定顺序的有效条件更新都会使后验分布保持不变。

只有在那些用混合指示变量来表示波动率似然的采样器中，顺序才需要特别注意，例如 {cite:t}`KimShephardChib1998` 中的方法；{cite:t}`DelNegroPrimiceri2015` 指出 {cite:t}`Primiceri2005` 在循环中的错误位置抽取了这些指示变量。

雅基耶-波尔森-罗西（Jacquier--Polson--Rossi）步骤没有这类指示变量。

### 施加稳定性约束

受限后验分布会将零密度赋予在任何日期（包括 $\theta_0$）爆炸的路径。

科格利和萨金特模拟*无限制*后验分布，仅保留系数路径稳定的那些扫描。

这是有效的，因为受限后验分布正是在稳定性条件下的无限制后验分布；但如果 $a$ 是无限制扫描中稳定的比例，则大约需要 $1/a$ 次完整扫描才能产生一次被保留的结果。

我们改为在区块 1 内部施加稳定性，即反复重新抽取后向过程，直到得到稳定的路径为止。

一次被接受的抽样来自在稳定性条件下的高斯卡特-科恩分布，这恰好就是受限的完全条件分布，因此该步骤是精确的吉布斯更新。

由于正向滤波在重新抽样之间不发生变化，每次重新抽样仅需一次后向过程的开销。

在极少数扫描中，条件值使得稳定抽样极不可能发生；如果 1000 次尝试全部失败，该步骤将保留当前路径。

该事件发生的概率仅取决于其他区块，而与当前路径无关，因此该步骤是精确抽样与保持不变的混合，仍然使受限条件分布保持不变。

这两种方案之间通过一个恒等式相联系。

将其他区块记为 $\Omega=(Q,H^T,\beta,\sigma)$，并令 $p(\Omega)$ 表示一次无限制卡特-科恩抽样是稳定的概率。

$\Omega$ 的受限后验分布就是无限制后验分布 $\pi_U(\Omega)$ 按 $p(\Omega)$ 重新加权的结果，并且一次扫描中后向过程的次数服从均值为 $1/p(\Omega)$ 的几何分布，因此在平稳状态下，每次扫描的平均后向过程次数为

$$
\int \frac{1}{p(\Omega)}\,\frac{p(\Omega)\,\pi_U(\Omega)}{a}\,d\Omega = \frac{1}{a}.
$$

因此，我们的采样器以廉价的后向过程来支付代价，而丢弃方案则以完整扫描来支付代价；平均的后向过程次数可以估计 $1/a$，其中 $a$ 是科格利和萨金特方案中稳定扫描所占的比例。

由于罕见的扫描需要非常多次的尝试，该平均值的噪声较大，而设定 1000 次的上限会使其产生向下的偏差，因此所隐含的 $a$ 是一个上界。

另一个精确方法是 {cite:t}`MurrayAdamsMacKay2010` 提出的椭圆切片采样器（elliptical slice sampler），它沿着一个由当前路径和一次新的卡特-科恩抽样所确定的椭圆移动，并向当前稳定路径收缩。

其新路径与当前路径保持接近，而重新抽样则与当前路径相互独立；但在我们的实验中，两种采样器的缓慢混合主要来自 $Q$ 与系数路径之间的依赖关系，而这两种步骤均未能解决这一问题。

{ref}`csdv-appendix` 给出了详细的递推公式。

由于代码较长，此处将其折叠；展开相应单元格即可查看。

第一个单元格实现了区块 1。

```{code-cell} ipython3
:tags: [hide-input]

def stacked_cholesky(matrices):
    """Cholesky factors of a stack of covariance matrices, with tiny jitter."""
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    scale = np.abs(np.diagonal(matrices, axis1=-2, axis2=-1)).max(axis=-1)
    jitter = 1e-12 * np.maximum(1.0, scale)[..., None, None]
    return np.linalg.cholesky(matrices + jitter * np.eye(matrices.shape[-1]))


def kalman_forward(y, x, q, h, β, prior):
    """Kalman filter for θ_t: filtered means and covariances, predicted covariances."""
    T, k = len(y), n_coefficients
    b_inverse = np.linalg.inv(b_matrix(β))
    r = np.einsum('ij,tj,kj->tik', b_inverse, h, b_inverse)    # R_1, ..., R_T
    X = np.einsum('ij,tk->tijk', np.eye(n_variables), x)
    X = X.reshape(T, n_variables, k)                            # X_t'
    means = np.empty((T + 1, k))
    covariances = np.empty((T + 1, k, k))
    predicted = np.empty((T + 1, k, k))
    means[0], covariances[0] = prior['θ_mean'], prior['θ_covariance']
    for t in range(T):
        p = covariances[t] + q                                  # P_{t+1|t}
        px = p @ X[t].T
        gain = np.linalg.solve(X[t] @ px + r[t], px.T).T
        means[t + 1] = means[t] + gain @ (y[t] - X[t] @ means[t])
        c = p - gain @ px.T
        covariances[t + 1] = 0.5 * (c + c.T)
        predicted[t + 1] = p
    return means, covariances, predicted


def backward_factors(covariances, predicted):
    """Gains J_t and Cholesky roots for the Carter-Kohn backward pass."""
    gains = np.swapaxes(
        np.linalg.solve(predicted[1:], covariances[:-1]), 1, 2
    )
    conditional = covariances[:-1] - covariances[:-1] @ np.swapaxes(gains, 1, 2)
    roots = stacked_cholesky(np.concatenate((conditional, covariances[-1:])))
    return gains, roots


def stable_path(θ_path):
    """Fast version of is_stable: a Schur-Cohn test at every date at once.

    Faddeev-LeVerrier gives the characteristic polynomial of each companion
    matrix, and the Schur-Cohn recursion checks that all its roots lie inside
    the unit circle without computing them.
    """
    rows = np.asarray(θ_path, dtype=float).reshape(n_variables, n_regressors, -1)
    A = np.zeros((rows.shape[2], 6, 6))
    A[:, :n_variables] = rows[:, 1:].transpose(2, 0, 1)
    A[:, n_variables:, :n_variables] = np.eye(n_variables)
    p = np.ones((len(A), 7))            # det(λI - A) = λ^6 + p_1 λ^5 + ... + p_6
    M = np.zeros_like(A)
    for k in range(1, 7):               # Faddeev-LeVerrier recursion
        AM = A @ (M + p[:, k - 1, None, None] * np.eye(6))
        p[:, k] = -np.einsum('tii->t', AM) / k
        M = AM
    for m in range(6, 0, -1):           # Schur-Cohn reduction
        reflection = p[:, m] / p[:, 0]
        if np.any(np.abs(reflection) >= 1):
            return False
        p = p[:, :m] - reflection[:, None] * p[:, m:0:-1]
    return True


def backward_pass(means, gains, roots, rng):
    """Draw θ_T, θ_{T-1}, ..., θ_0, each conditioned on the draw after it."""
    T = len(means) - 1
    noise = np.einsum('tij,tj->ti', roots, rng.standard_normal(means.shape))
    path = np.empty_like(means)
    path[T] = means[T] + noise[T]
    for t in range(T - 1, -1, -1):
        path[t] = means[t] + gains[t] @ (path[t + 1] - means[t]) + noise[t]
    return path.T


def draw_coefficient_path(current, y, x, q, h, β, prior, rng,
                          max_passes=1_000):
    """Redraw the Carter-Kohn path until it is stable.

    If max_passes draws all fail, keep the current (stable) path; the step
    still leaves the restricted conditional invariant.
    """
    means, covariances, predicted = kalman_forward(y, x, q, h, β, prior)
    gains, roots = backward_factors(covariances, predicted)
    for passes in range(1, max_passes + 1):
        path = backward_pass(means, gains, roots, rng)
        if stable_path(path):
            return path, passes
    return current, max_passes + 1
```

第二个单元格实现了区块 2 至区块 5。

```{code-cell} ipython3
:tags: [hide-input]

def draw_q(θ_path, prior, rng):
    """Draw Q from its inverse-Wishart conditional given the T increments.

    Bartlett decomposition: if A is lower triangular with A_ii^2 ~ chi2(df - i)
    and standard normals below the diagonal, then A A' ~ Wishart(df, I), and
    Q = C (A A')^{-1} C' ~ IW(df, C C').
    """
    increments = np.diff(θ_path, axis=1)
    scale = prior['q_scale'] + increments @ increments.T
    degrees_freedom = prior['q_degrees_freedom'] + increments.shape[1]
    k = len(scale)
    A = np.tril(rng.standard_normal((k, k)), -1)
    A[np.diag_indices(k)] = np.sqrt(rng.chisquare(degrees_freedom - np.arange(k)))
    M = np.linalg.solve(A, np.linalg.cholesky(scale).T).T     # C A'^{-1}
    return M @ M.T


def var_residuals(y, x, θ_path):
    """VAR residuals at dates 1, ..., T for a path θ_0, ..., θ_T."""
    coefficients = θ_path[:, 1:].T.reshape(len(y), n_variables, n_regressors)
    return y - np.einsum('tk,tnk->tn', x, coefficients)


def draw_σ(h, prior, rng):
    """Draw the log-volatility innovation standard deviations."""
    increments = np.diff(np.log(h), axis=0)
    shape = (prior['σ_degrees_freedom'] + len(increments)) / 2
    scale = (prior['σ_scale'] + np.sum(increments**2, axis=0)) / 2
    return np.sqrt(scale / rng.gamma(shape, 1.0, size=n_variables))


def draw_β(residuals, h, prior, rng):
    """Draw the free elements of B from two Gaussian regressions."""
    β = np.empty(3)
    for equation, slots in ((1, slice(0, 1)), (2, slice(1, 3))):
        standardized = residuals / np.sqrt(h[1:, equation])[:, None]
        regressors = -standardized[:, :equation]
        precision = (np.eye(equation) / prior['β_variance']
                     + regressors.T @ regressors)
        covariance = np.linalg.inv(precision)
        mean = covariance @ regressors.T @ standardized[:, equation]
        β[slots] = mean + np.linalg.cholesky(covariance) @ (
            rng.standard_normal(equation))
    return β


def draw_volatility_path(h, residuals, β, σ, prior, rng):
    """Jacquier-Polson-Rossi single-site updates of log h_0, ..., log h_T."""
    T = len(residuals)
    squared = (residuals @ b_matrix(β).T)**2         # orthogonalized residuals
    old, new = np.log(h), np.empty_like(h)
    variance = σ**2
    v0 = (prior['log_h_variance'] * variance
          / (variance + prior['log_h_variance']))
    m0 = v0 * (prior['log_h_mean'] / prior['log_h_variance']
               + old[1] / variance)
    new[0] = m0 + np.sqrt(v0) * rng.standard_normal(n_variables)
    z = rng.standard_normal((T, n_variables))
    log_u = np.log(rng.random((T, n_variables)))
    for t in range(1, T + 1):
        if t < T:   # neighbors t-1 (already updated) and t+1 set the proposal
            proposal = (0.5 * (new[t - 1] + old[t + 1])
                        + np.sqrt(variance / 2) * z[t - 1])
        else:       # the last date has only one neighbor
            proposal = new[t - 1] + σ * z[t - 1]
        log_ratio = 0.5 * (old[t] - proposal) + 0.5 * squared[t - 1] * (
            np.exp(-old[t]) - np.exp(-proposal)
        )
        new[t] = np.where(log_u[t - 1] <= log_ratio, proposal, old[t])
    return np.exp(new)
```

最后一个单元格将这五个区块组合在一起，其前面有一个预热（warm-up）阶段，在该阶段中系数固定为 $\bar\theta$，使波动率得以稳定下来。

传入 `fixed_q` 参数会将 $Q$ 固定为该值而不对其进行抽样；零矩阵对应于常系数模型这一特例。

```{code-cell} ipython3
:tags: [hide-input]

def run_sampler(y, x, prior, n_sweeps, burn, thin, seed,
                warmup=200, fixed_q=None):
    """Simulate the stability-restricted posterior.

    Keeps every thin-th sweep after burn, aligned with dates 1, ..., T, and
    records tr(Q) and the number of backward passes at every sweep
    (max_passes + 1 marks a sweep that kept the current path).
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    T = len(y)
    θ = np.repeat(prior['θ_mean'][:, None], T + 1, axis=1)
    residuals = var_residuals(y, x, θ)
    changes = np.diff(y, axis=0)
    h = np.exp(np.vstack((
        np.tile(prior['log_h_mean'], (2, 1)),
        np.log(np.maximum((changes - changes.mean(axis=0))**2, 1e-300)),
    )))
    β = np.zeros(3)
    for _ in range(warmup):
        σ = draw_σ(h, prior, rng)
        β = draw_β(residuals, h, prior, rng)
        h = draw_volatility_path(h, residuals, β, σ, prior, rng)
    q = prior['q_center'] if fixed_q is None else fixed_q
    n_keep = len(range(burn, n_sweeps, thin))
    draws = {
        'θ': np.empty((n_keep, n_coefficients, T)),
        'h': np.empty((n_keep, T, n_variables)),
        'β': np.empty((n_keep, 3)),
        'Q': np.empty((n_keep, n_coefficients, n_coefficients)),
    }
    trace_q = np.empty(n_sweeps)
    passes = np.empty(n_sweeps, dtype=int)
    for sweep in range(n_sweeps):
        θ, passes[sweep] = draw_coefficient_path(θ, y, x, q, h, β, prior, rng)
        if fixed_q is None:
            q = draw_q(θ, prior, rng)
        residuals = var_residuals(y, x, θ)
        σ = draw_σ(h, prior, rng)
        β = draw_β(residuals, h, prior, rng)
        h = draw_volatility_path(h, residuals, β, σ, prior, rng)
        trace_q[sweep] = np.trace(q)
        if sweep >= burn and (sweep - burn) % thin == 0:
            j = (sweep - burn) // thin
            draws['θ'][j], draws['h'][j] = θ[:, 1:], h[1:]
            draws['β'][j], draws['Q'][j] = β, q
    draws.update(trace_q=trace_q, passes=passes,
                 minutes=(time.perf_counter() - started) / 60)
    return draws
```

科格利和萨金特运行了 100000 次扫描，丢弃前一半，并在其余部分中每隔十次保留一次。

我们运行 40000 次扫描，丢弃前 4000 次，并在其余部分中每隔 20 次保留一次用于后验统计摘要，同时在每次扫描时记录 $\operatorname{tr}(Q)$ 以及后向过程的次数。

```{code-cell} ipython3
burn = 4_000
posterior = run_sampler(
    data['y'], data['x'], prior,
    n_sweeps=40_000, burn=burn, thin=20, seed=42,
)
passes = posterior['passes'][burn:]

pd.Series({
    'run time (minutes)': posterior['minutes'],
    'retained draws': len(posterior['β']),
    'mean backward passes per sweep': passes.mean(),
    'share of sweeps stable at the first pass': np.mean(passes == 1),
    'implied share a of stable unrestricted sweeps': 1 / passes.mean(),
    'sweeps that kept the current path': np.sum(passes > 1_000),
}, name='value').to_frame().round(3)
```

所隐含的比例 $a$ 远低于二分之一，因此丢弃方案将会舍弃其大部分扫描结果，而每次扫描均需耗费一次正向滤波加上其他四个区块的计算，相比之下我们的采样器仅需为每次失败付出一次后向过程的代价。

连续的扫描之间仍然保持较强的相关性，尤其是在 $Q$ 的方向上，因此 {ref}`csdv-mc-error` 测量了下文所引用数值中的蒙特卡洛误差。

(csdv-results)=
## 数据揭示了什么

我们用后验均值系数路径 $\mathbb{E}(\theta_t\mid T)$ 和后验均值协方差路径
$\mathbb{E}(R_t\mid T)$ 来总结后验，然后在我们所提出问题的背景下对其加以解释。

### 漂移的速率与结构

$Q$ 的迹衡量的是系数漂移的总体速率，$\operatorname{tr}(Q)=0$ 对应于系数恒定不变的情形。

直方图展示了预烧期之后每一次迭代的 $\operatorname{tr}(Q)$，以及先验尺度 $\operatorname{tr}(\bar Q)$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior $\operatorname{tr}(Q)$ and prior $\operatorname{tr}(\bar Q)$
    name: fig-csdv-drift-rate
---
trace_q = posterior['trace_q'][burn:]
fig, ax = plt.subplots()
ax.hist(trace_q, bins=50, histtype='step', lw=2)
ax.axvline(trace_q_bar, color='C1', lw=2, label=r'prior $\mathrm{tr}(\bar Q)$')
ax.set_xlabel(r'$\mathrm{tr}(Q)$')
ax.set_ylabel('sweeps')
ax.legend()
plt.show()
```

下表将 $Q$ 的若干总结统计量与科格利和萨金特所报告的数字（其表1和第272页）进行了比较。

```{code-cell} ipython3
q_mean = posterior['Q'].mean(axis=0)
q_eigenvalues = np.linalg.eigvalsh(q_mean)[::-1]
retained_trace_q = np.trace(posterior['Q'], axis1=1, axis2=2)

pd.DataFrame({
    'this lecture': [
        trace_q.mean(),
        q_eigenvalues[0],
        retained_trace_q.min() / trace_q_bar,
    ],
    'Cogley and Sargent': [0.047, 0.023, 3.0],
}, index=[
    'posterior mean of tr(Q)',
    'largest eigenvalue of posterior-mean Q',
    'smallest retained tr(Q) / tr(Q bar)',
]).round(3)
```

整个后验分布都远远位于保守的先验尺度右侧，因此在该模型内部，数据所要求的系数漂移远超先验预期。

$\operatorname{tr}(Q)$ 的后验均值和最大特征值都大于科格利和萨金特的结果。

$\operatorname{tr}(Q)$ 的后验分布具有较长的右尾，马尔可夫链对其探索较慢，因此其均值存在不可忽视的蒙特卡洛误差（{ref}`csdv-mc-error`），但在我们用其他随机种子所做的每一次运行中，它都高于科格利和萨金特的数值。

这并非对 $Q=0$ 的正式检验，因为连续先验没有在 $Q=0$ 处赋予任何质量；科格利和萨金特避免使用贝叶斯因子，因为他们对 $Q$ 的先验非常弥散。

归因于 $\theta_t$ 的变动，是简化形式中*系统性*部分的变动；VAR 本身并不能说明是政策还是私人行为导致了这种变动。

科格利和萨金特对三个变量的每一种排序都做了估计，他们的后验均值表明，排序的改变会影响数值大小，但不会消除漂移：

| Ordering | Stable $\operatorname{tr}(Q)$ | Stable $\max(\lambda)$ | Unrestricted $\operatorname{tr}(Q)$ | Unrestricted $\max(\lambda)$ |
|---|---:|---:|---:|---:|
| $(i,\pi,u)$ | 0.055 | 0.025 | 0.056 | 0.027 |
| $(i,u,\pi)$ | 0.047 | 0.023 | 0.059 | 0.031 |
| $(\pi,i,u)$ | 0.064 | 0.031 | 0.082 | 0.044 |
| $(\pi,u,i)$ | 0.062 | 0.031 | 0.088 | 0.051 |
| $(u,i,\pi)$ | 0.057 | 0.026 | 0.051 | 0.028 |
| $(u,\pi,i)$ | 0.055 | 0.024 | 0.072 | 0.035 |

这里 $\max(\lambda)$ 是 $Q$ 的后验均值的最大特征值。

排序之所以重要，是因为前两个变量与其他变量共享随机波动率的来源，而第三个变量则有其自身的来源；遵循原论文，我们采用漂移最小的排序 $(i,u,\pi)$。

对于该排序，去掉稳定性限制会提高后验均值漂移速率，因此该限制并非漂移的制造者。

对 $Q$ 的后验均值进行对角化表明，漂移是低维的。

```{code-cell} ipython3
q_shares = np.cumsum(q_eigenvalues) / q_eigenvalues.sum()
pd.DataFrame({
    'eigenvalue': q_eigenvalues[:3],
    'cumulative share': q_shares[:3],
    'Cogley and Sargent': [0.485, 0.832, 0.945],
}, index=pd.Index([1, 2, 3], name='principal component')).round(3)
```

正如科格利和萨金特所发现的那样，在21个系数中，只有少数几个线性组合存在漂移：第一主成分占总漂移的五分之二以上，前三个主成分则占据了大多数。

后验均值系数路径显示了漂移具体发生在何处。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior mean VAR coefficients $E(\theta_t\mid T)$
    name: fig-csdv-coefficient-paths
---
θ_mean = posterior['θ'].mean(axis=0)
dates = data['dates']
coefficient_labels = ('constant', r'$i_{t-1}$', r'$u_{t-1}$', r'$\pi_{t-1}$',
                      r'$i_{t-2}$', r'$u_{t-2}$', r'$\pi_{t-2}$')
equation_labels = ('interest', 'unemployment', 'inflation')

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
for equation, ax in enumerate(axes):
    rows = slice(7 * equation, 7 * equation + 7)
    lines = ax.plot(dates, θ_mean[rows].T, lw=2)
    ax.axhline(0, color='0.65', lw=1)
    ax.set_xlabel('year')
    ax.set_title(f'{equation_labels[equation]} equation')
fig.legend(lines, coefficient_labels, loc='lower center', ncol=4)
plt.tight_layout(rect=(0, 0.17, 1, 1))
plt.show()
```

利率方程和失业方程相对稳定，而通货膨胀方程的系数——尤其是滞后利率那一项——在整个70年代剧烈波动，并在1980年前后出现转折。

不应对单个滞后系数做出结构性解释，因为成对的滞后项可能相互抵消；下文的动态总结指标提供的信息更多。

### 波动率的演变

我们首先要问的是，冲击的*大小*是如何变化的。

后验均值 $\mathbb{E}(R_t\mid T)$ 是对所有保留抽样的 $B^{-1}H_tB^{-1\prime}$ 取平均。

```{code-cell} ipython3
def mean_innovation_covariance(h_draws, β_draws):
    """E(R_t | T): the average of B^{-1} H_t B^{-1}' over retained draws."""
    b_inverse = np.linalg.inv(np.stack([b_matrix(β) for β in β_draws]))
    total = np.einsum('dij,dtj,dkj->tik', b_inverse, h_draws, b_inverse,
                      optimize=True)
    return total / len(β_draws)


r_mean = mean_innovation_covariance(posterior['h'], posterior['β'])
```

下图展示了创新标准差（以季度基点表示）和创新相关系数。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Standard deviations and correlations implied by $E(R_t\mid T)$
    name: fig-csdv-volatility-correlation
---
variances = ((0, 'Nominal interest'), (2, 'Inflation'), (1, 'Unemployment'))
correlations = ((0, 1, 'Interest--unemployment'),
                (0, 2, 'Interest--inflation'),
                (2, 1, 'Inflation--unemployment'))
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
for row, (index, label) in enumerate(variances):
    axes[row, 0].plot(dates, 1e4 * np.sqrt(r_mean[:, index, index]), lw=2)
    axes[row, 0].set_title(label)
for row, (left, right, label) in enumerate(correlations):
    scale = np.sqrt(r_mean[:, left, left] * r_mean[:, right, right])
    axes[row, 1].plot(dates, r_mean[:, left, right] / scale, lw=2)
    axes[row, 1].set_title(label)
axes[1, 0].set_ylabel(r'standard deviation $\times 10^4$')
axes[1, 1].set_ylabel('correlation')
axes[-1, 0].set_xlabel('year')
axes[-1, 1].set_xlabel('year')
plt.tight_layout()
plt.show()
```

利率和通货膨胀创新标准差在1980年前后出现飙升，并在随后几年内回落，而失业创新波动率则在整个样本期间呈下降趋势，并伴有逆周期性的起伏。

这三种创新相关系数的绝对值都在沃尔克反通胀期间达到最大，因此简化形式冲击的大小和构成都发生了变化。

这些变动为“运气变化”即“运气不好”的解释赋予了重要地位，尽管利率创新本身并非结构性货币政策冲击。

实际上，它与通货膨胀创新呈正相关、与失业创新呈负相关，这更像是一种政策*反应*，而非政策冲击，因为按照传统观点，政策冲击本应使通货膨胀与失业朝相反方向运动。

遵循 {cite:t}`Whittle1953`，$\mathbb{E}(R_t\mid T)$ 的对数行列式总结了每个日期进入系统的一步总体不确定性。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Generalized innovation variance $\log |E(R_t\mid T)|$
    name: fig-csdv-total-variance
---
years = np.floor(dates + 1e-8).astype(int)
logdet_r = np.linalg.slogdet(r_mean)[1]

fig, ax = plt.subplots()
ax.plot(dates, logdet_r, lw=2)
ax.set_xlabel('year')
ax.set_ylabel(r'$\log |E(R_t\mid T)|$')
plt.show()
```

下表给出了该图背后的数字。

```{code-cell} ipython3
def period_mean(values, first, last):
    """Average a quarterly series over the years first, ..., last."""
    return values[(years >= first) & (years <= last)].mean()


pd.Series({
    'mean, 1960--1964': period_mean(logdet_r, 1960, 1964),
    'peak': logdet_r.max(),
    'date of peak': dates[np.argmax(logdet_r)],
    'mean, 1990--2000': period_mean(logdet_r, 1990, 2000),
}, name='log |E(R_t | T)|').to_frame().round(2)
```

总创新方差分两步上升，第一步从60年代中期到70年代初，第二步在70年代末再度上升，在1980年底达到峰值，而随后的大部分下降发生在80年代上半叶。

其90年代的平均值与或低于60年代初的平均值，因此由 {cite:t}`KimNelson1999` 和 {cite:t}`McConnellPerezQuiros2000` 记录的“大缓和”时期看起来更像是回归到了60年代初的稳定状态。

科格利和萨金特把这理解为一种*回归*稳定状态，并指出此前那段平静期是多么短暂。

这是关于运气变化最清晰的总体证据，但它无法解释接下来要考察的、以系数为基础的通货膨胀动态变化。

### 核心通货膨胀与自然失业率

为了研究系统性动态，我们把 $t$ 期的 VAR 用伴随形式写成

$$
z_t = \mu_{t\mid T} + A_{t\mid T}z_{t-1} +
\begin{bmatrix} \varepsilon_t \\ 0 \end{bmatrix},
$$

其中 $z_t=(y_t^\top, y_{t-1}^\top)^\top$，而 $\mu_{t\mid T}$ 和 $A_{t\mid T}$ 则由
$\mathbb{E}(\theta_t\mid T)$ 构造而来。

在日期 $t$，局部均值 $m_t$ 是这样一个点：如果伴随系统的系数保持在其 $t$ 期取值，且未来的创新为零，系统将收敛到该点：

```{math}
:label: csdv_local_means
m_t = (I-A_{t\mid T})^{-1}\mu_{t\mid T},
\qquad
\bar\pi_t = 4s_\pi m_t,
\qquad
\bar u_t = \frac{\exp(100s_u m_t)}{1+\exp(100s_u m_t)}.
```

这里 $s_\pi$ 和 $s_u$ 从 $m_t$ 中选取通货膨胀和失业率，因子四对通货膨胀进行年化，逆 logit 变换则将失业率还原为某个比率。

核心通货膨胀 $\bar\pi_t$ 是冻结 $t$ 期系数所隐含的长期通货膨胀预测，自然失业率 $\bar u_t$ 则是相应的长期失业率锚点。

它们是被冻结的局部系统的稳态，而非漂移过程的无条件均值。

冻结当前系数并向前推算，是 {doc}`phillips_learning`、{doc}`phillips_escaping_nash` 和 {doc}`phillips_lost_conquest` 中学习型政府所采用的*预期效用*（anticipated-utility）手法 {cite}`Kreps1998`，这些政府的行为就好像自己当前的信念永远不会被修正一样。

```{code-cell} ipython3
def local_means(θ_path):
    """Core inflation (annual rate) and the natural unemployment rate."""
    θ_path = np.asarray(θ_path, dtype=float).reshape(n_coefficients, -1)
    core = np.empty(θ_path.shape[1])
    natural = np.empty(θ_path.shape[1])
    for t, θ in enumerate(θ_path.T):
        intercept, companion = companion_matrix(θ)
        m = np.linalg.solve(np.eye(6) - companion, intercept)
        core[t], natural[t] = 4 * m[2], expit(100 * m[1])
    return core, natural


core_inflation, natural_rate = local_means(θ_mean)
```

我们绘制每年第四季度的取值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Core inflation $\bar\pi_t$ and the natural rate $\bar u_t$
    name: fig-csdv-local-means
---
annual = np.array([np.flatnonzero(years == year)[-1]
                   for year in np.unique(years)])

fig, ax = plt.subplots()
ax.plot(dates[annual], 100 * core_inflation[annual], 'o-', lw=2,
        markersize=3, label='core inflation')
ax.plot(dates[annual], 100 * natural_rate[annual], '+-', lw=2,
        markersize=5, label='natural rate')
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.legend()
plt.show()
```

下表给出了该图背后的数字。

```{code-cell} ipython3
pd.Series({
    'core inflation, 1960--1964 mean': 100 * period_mean(core_inflation, 1960, 1964),
    'core inflation, peak': 100 * core_inflation[annual].max(),
    'year of the core peak': years[annual][np.argmax(core_inflation[annual])],
    'core inflation, 1985--2000 mean': 100 * period_mean(core_inflation, 1985, 2000),
    'natural rate, 1960': 100 * natural_rate[annual][0],
    'natural rate, peak': 100 * natural_rate[annual].max(),
    'year of the natural-rate peak': years[annual][np.argmax(natural_rate[annual])],
    'natural rate, 2000': 100 * natural_rate[-1],
    'correlation of core and natural rate': np.corrcoef(core_inflation, natural_rate)[0, 1],
}, name='estimate').to_frame().round(2)
```

核心通货膨胀从60年代初的约1.5%攀升到70年代末的峰值，然后回落到约3%，这一模式与科格利和萨金特所报告的结果（约1.5、8和2.5到3.5个百分点）一致，不过我们得到的峰值更高。

自然失业率从约5%上升到70年代末的峰值，到2000年时又降至4%以下。

由于这两条曲线都由拟合的系数决定，而非由实现的冲击决定，它们的持续性变化指向了一个不断变化的系统性成分。

这两个锚点之间高度相关，相关程度超过了科格利和萨金特所报告的0.748：模型中的长期通货膨胀率和失业率一同上升、又一同回落，这与当前通货膨胀和当前失业率同向变动并不是一回事。

由于当最大根接近于一时，$(I-A_t)^{-1}$ 会放大微小的系数变化，长期均值本质上比短期预测更为敏感。

(csdv_persistence)=
### 通货膨胀的持续性

我们现在要问的是，除了不断变动的波动率之外，系统性动态本身是否也发生了漂移。

主要的总结指标是通货膨胀持续性，用零频率处通货膨胀的归一化谱来衡量。

日期 $t$ 处通货膨胀的谱密度为

```{math}
:label: csdv_spectrum
f_{\pi\pi}(\omega,t)
=
\frac{1}{2\pi}
s_\pi
(I-A_{t\mid T}e^{-i\omega})^{-1}
\mathcal R_t
(I-A_{t\mid T}^\top e^{i\omega})^{-1}
s_\pi^\top,
```

其中 $\mathcal R_t$ 将 $\mathbb{E}(R_t\mid T)$ 嵌入伴随系统中。

低频功率既取决于自回归系数，也取决于创新协方差，因此科格利和萨金特用日期 $t$ 的通货膨胀方差进行归一化，

```{math}
:label: csdv_normalized_spectrum
g_{\pi\pi}(\omega,t)
=
\frac{f_{\pi\pi}(\omega,t)}
{\int_{-\pi}^{\pi}f_{\pi\pi}(\omega,t)d\omega},
```

它衡量的是自相关而非自协方差。

他们把 $g_{\pi\pi}(0,t)$ 解释为通货膨胀持续性的度量。

零频率处的统计量是长期统计量：{doc}`sargent_surico` 把长期斜率计算为零频率处谱密度之比，{ref}`csdv2_ex3` 则根据本模型计算了通货膨胀对失业率的这样一种斜率。

这一归一化去除了 $R_t$ 的一个公共尺度因子，但它仍可能依赖于 $R_t$ 中的相对方差和协方差。

下面这个函数在以每季度周期数表示的若干频率上计算两种谱。

```{code-cell} ipython3
def inflation_spectrum(θ, covariance, frequencies):
    """Inflation power f(ω) and normalized power g(ω) at one date."""
    _, companion = companion_matrix(θ)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    variance = linalg.solve_discrete_lyapunov(companion, innovation)[2, 2]
    power = np.empty(len(frequencies))
    for j, frequency in enumerate(frequencies):
        transfer = np.linalg.inv(
            np.eye(6) - companion * np.exp(-2j * np.pi * frequency))
        power[j] = np.real(transfer @ innovation @ transfer.conj().T)[2, 2]
    return power / (2 * np.pi), power / (2 * np.pi * variance)


def persistence_path(θ_path, r_path):
    """g_ππ(0, t) at every date."""
    return np.array([inflation_spectrum(θ, r, [0.0])[1][0]
                     for θ, r in zip(θ_path.T, r_path)])


inflation_persistence = persistence_path(θ_mean, r_mean)
white_noise = 1 / (2 * np.pi)
```

对于白噪声而言，谱是平坦的，因此 $g_{\pi\pi}(0)=1/(2\pi)\approx 0.16$；这就是解读该路径时所参照的基准值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normalized zero-frequency spectrum $g_{\pi\pi}(0,t)$
    name: fig-csdv-inflation-persistence
---
fig, ax = plt.subplots()
ax.plot(dates[annual], inflation_persistence[annual], 'o-', lw=2, markersize=3)
ax.axhline(white_noise, color='0.45', ls='--', lw=1,
           label=r'white noise, $1/(2\pi)$')
ax.set_xlabel('year')
ax.set_ylabel(r'$g_{\pi\pi}(0,t)$')
ax.legend()
plt.show()
```

下表给出了该图背后的数字。

```{code-cell} ipython3
pd.Series({
    '1960--1964 mean': period_mean(inflation_persistence, 1960, 1964),
    '1970--1979 mean': period_mean(inflation_persistence, 1970, 1979),
    'peak': inflation_persistence[annual].max(),
    'year of the peak': years[annual][np.argmax(inflation_persistence[annual])],
    '1985--2000 mean': period_mean(inflation_persistence, 1985, 2000),
}, name='g(0, t)').to_frame().round(3)
```

持续性在60年代初接近白噪声水平，在60年代末向上扫升，在整个70年代保持高位、并于1979年达到峰值，随后在1980年之后急剧下降，在80年代和90年代的大部分时间里回落到接近白噪声的水平。

1980年之后的这一下降不能用对所有创新进行一次比例缩放来解释，尽管这一归一化统计量仍可能取决于 $R_t$ 的组成。

作为比较，一个系数为 $\rho$ 的 $AR(1)$ 过程具有
$g(0)=(1+\rho)/[2\pi(1-\rho)]$，因此介于2和10之间的取值对应于 $\rho$ 在约
$0.85$ 到 $0.97$ 之间；{ref}`csdv_ex1` 把整条路径转换为这样一个“等价
$AR(1)$”系数。

零频率路径省略了频率分布的其余部分，下面的热力图展示了这一点。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normalized inflation spectrum $g_{\pi\pi}(\omega,t)$
    name: fig-csdv-inflation-spectra
---
spectrum_frequencies = np.linspace(0, 0.5, 41)
normalized_spectrum = np.column_stack([
    inflation_spectrum(θ_mean[:, t], r_mean[t], spectrum_frequencies)[1]
    for t in range(len(dates))
])

fig, ax = plt.subplots(figsize=(8, 4))
image = ax.pcolormesh(dates, spectrum_frequencies,
                      np.log10(normalized_spectrum), shading='auto')
ax.set_xlabel('year')
ax.set_ylabel('cycles per quarter')
fig.colorbar(image, ax=ax, label=r'$\log_{10} g_{\pi\pi}(\omega,t)$')
plt.show()
```

一条低频功率的“脊”从60年代末延伸到80年代初，随后消退。

由于这条“脊”在归一化后依然存在，大通胀不仅仅是一个大规模冲击的时期：通货膨胀冲击也以更持久的方式传播开来。

点估计并不能揭示数据对这些路径的确定程度有多强，因此我们从每一个保留的抽样中计算这些年度特征。

```{code-cell} ipython3
def posterior_feature_draws(draws, indices):
    """Core inflation, the natural rate, and g(0) at selected dates, by draw."""
    n_draws = len(draws['β'])
    core = np.empty((len(indices), n_draws))
    natural = np.empty_like(core)
    persistence = np.empty_like(core)
    for d in range(n_draws):
        θ_d = draws['θ'][d][:, indices]
        core[:, d], natural[:, d] = local_means(θ_d)
        for row, t in enumerate(indices):
            covariance = innovation_covariance(draws['h'][d][t], draws['β'][d])
            persistence[row, d] = inflation_spectrum(
                θ_d[:, row], covariance, [0.0])[1][0]
    return {'core': 100 * core, 'natural': 100 * natural,
            'persistence': persistence}


feature_draws = posterior_feature_draws(posterior, annual)
```

该图针对每个特征，绘制了跨抽样的后验中位数以及90%区间。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Posterior medians and pointwise 90 percent intervals for $\bar\pi_t$, $\bar u_t$, and $g_{\pi\pi}(0,t)$
    name: fig-csdv-feature-uncertainty
---
feature_specs = (('core', 'core inflation (%)'),
                 ('natural', 'natural rate (%)'),
                 ('persistence', r'$g_{\pi\pi}(0,t)$'))
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (key, ylabel) in zip(axes, feature_specs):
    lower, median, upper = np.quantile(feature_draws[key], (0.05, 0.5, 0.95),
                                       axis=1)
    line, = ax.plot(dates[annual], median, lw=2)
    ax.fill_between(dates[annual], lower, upper, color=line.get_color(),
                    alpha=0.2)
    ax.set_ylabel(ylabel)
axes[-1].set_xlabel('year')
plt.tight_layout()
plt.show()
```

这些区间是逐点的，而非针对整条路径的联合区间。

在70年代，这些区间宽阔且右偏，因此上升走势和1980年后的回落要比其具体幅度更为清晰。

正如科格利和萨金特所指出的，关于核心通货膨胀的不确定性会随着持续性的提高而增大，因为核心通货膨胀是一个长期预测值。

下图比较了核心通货膨胀与持续性的时间走势，类似于科格利和萨金特论文中的图11。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Core inflation $\bar\pi_t$ and persistence $g_{\pi\pi}(0,t)$
    name: fig-csdv-core-persistence
---
fig, ax = plt.subplots()
ax.plot(dates[annual], 100 * core_inflation[annual], 'o-', lw=2,
        markersize=3, label='core inflation (%)')
ax.plot(dates[annual], inflation_persistence[annual], 'x-', lw=2,
        markersize=4, label=r'$g_{\pi\pi}(0,t)$')
ax.set_xlabel('year')
ax.legend()
plt.show()

core_persistence_correlation = np.corrcoef(
    core_inflation, inflation_persistence)[0, 1]
print(f'correlation of core inflation and persistence: '
      f'{core_persistence_correlation:.3f}')
```

核心通货膨胀与持续性在60年代末和70年代同步上升，并在沃尔克反通胀期间一同回落。

打印出的季度相关系数较高，接近科格利和萨金特所报告的0.92。

它总结的是时间上的共性，而非因果关系：这两个度量都是同一组拟合系数的非线性函数。

{ref}`csdv-verdict` 解释了为何这一正相关关系给 {doc}`phillips_escaping_nash` 中的逃逸路径模型带来了困扰。

### 货币政策的积极程度

科格利和萨金特用 {cite:t}`Clarida2000` 提出的带有利率平滑的前瞻性泰勒规则来总结系统性政策，

```{math}
:label: csdv_policy_rule
i_t = \psi_0
+ \psi_1 \mathbb{E}_t\bar\pi_{t,t+h_\pi}
+ \psi_2 \mathbb{E}_t\bar u_{t,t+h_u}
+ \psi_3 i_{t-1}
+ e_t,
```

其中 $\bar\pi_{t,t+h_\pi}$ 和 $\bar u_{t,t+h_u}$ 分别是未来 $h_\pi$ 个季度和
$h_u$ 个季度内通货膨胀和失业率的平均值。

他们把积极程度系数定义为 $\mathcal A_t=\psi_1/(1-\psi_3)$，并把
$\mathcal A_t\geq 1$ 的情形称为“积极”政策，意味着名义利率在长期内对预期通货膨胀的反应超过一对一。

长期反应 $\psi_1(1+\psi_3+\psi_3^2+\cdots)$ 只有在 $|\psi_3|<1$ 时才存在。

在每个日期，局部 VAR 所隐含的总体两阶段最小二乘投影会给出这些系数。

第一阶段把美联储的预测投影到滞后信息上，也就是 VAR 在 $t-1$ 期所做出的预测，因为美联储在制定利率时并不能看到本季度的通货膨胀和失业数据。

第二阶段把 $i_t$ 投影到这些拟合预测值和 $i_{t-1}$ 上。

基准视界为 $h_\pi=4$ 季度和 $h_u=2$ 季度，反映了美联储关于政策滞后效应的民间经验之谈。

```{code-cell} ipython3
def policy_rule_coefficients(θ, covariance, h_pi=4, h_u=2):
    """Population 2SLS coefficients (ψ_1, ψ_2, ψ_3) implied by a local VAR."""
    _, companion = companion_matrix(θ)
    innovation = np.zeros((6, 6))
    innovation[:3, :3] = covariance
    stationary = linalg.solve_discrete_lyapunov(companion, innovation)
    power, π_load, u_load = np.eye(6), np.zeros(6), np.zeros(6)
    for horizon in range(1, max(h_pi, h_u) + 1):
        power = power @ companion
        if horizon <= h_pi:
            π_load += power[2] / h_pi
        if horizon <= h_u:
            u_load += power[1] / h_u
    loadings = np.vstack((π_load, u_load, np.eye(6)[0]))
    return np.linalg.solve(loadings @ stationary @ loadings.T,
                           loadings @ stationary @ companion.T[:, 0])


rules = np.array([policy_rule_coefficients(θ_mean[:, t], r_mean[t])
                  for t in range(len(dates))])
ψ1, ψ3 = rules[:, 0], rules[:, 2]
activism = np.where(np.abs(ψ3) < 1, ψ1 / (1 - ψ3), np.nan)
print(f'share of dates with |ψ_3| < 1: {np.mean(np.abs(ψ3) < 1):.3f}')
```

该图绘制了后验均值处的 $\mathcal A_t$，类似于科格利和萨金特论文中的图12，并对所有 $|\psi_3|\geq 1$ 的日期留空。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Activism coefficient $\mathcal A_t=\psi_{1t}/(1-\psi_{3t})$ at the posterior mean
    name: fig-csdv-policy-activism
---
fig, ax = plt.subplots()
ax.plot(dates, activism, lw=2)
ax.axhline(1, color='0.45', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'$\mathcal{A}_t$')
plt.show()
```

从点估计来看，政策在60年代初是积极的，从70年代初到1980年大部分时间是消极的，然后在沃尔克反通胀开始时转为积极，并一直保持积极状态直至90年代中期。

样本末端的估计极不稳定，原因将在下文说明。

当通货膨胀接近白噪声时，该规则的系数识别较弱，因为此时滞后变量对未来通货膨胀是较差的工具变量，因此基于后验均值输入构建的路径会低估不确定性。

因此，我们从1975年、1985年和1995年的每一个保留抽样中计算 $\mathcal A_t$，并将这些后验概率与科格利和萨金特的结果进行比较。

```{code-cell} ipython3
selected_years = (1975, 1985, 1995)
selected_dates = [np.flatnonzero(years == year)[-1] for year in selected_years]
n_draws = len(posterior['β'])
activism_draws = {year: np.empty(n_draws) for year in selected_years}
inside = {year: np.empty(n_draws, dtype=bool) for year in selected_years}
for d in range(n_draws):
    for year, t in zip(selected_years, selected_dates):
        covariance = innovation_covariance(posterior['h'][d][t],
                                           posterior['β'][d])
        ψ = policy_rule_coefficients(posterior['θ'][d][:, t], covariance)
        activism_draws[year][d] = ψ[0] / (1 - ψ[2])
        inside[year][d] = abs(ψ[2]) < 1

activism_events = (
    activism_draws[1975] > 1,
    activism_draws[1985] > 1,
    activism_draws[1995] > 1,
    activism_draws[1985] > activism_draws[1975],
    activism_draws[1995] > activism_draws[1975],
)
pd.DataFrame({
    'this lecture': [event.mean() for event in activism_events],
    'Cogley and Sargent': [0.208, 0.919, 0.941, 0.923, 0.943],
}, index=['P(A_1975 > 1)', 'P(A_1985 > 1)', 'P(A_1995 > 1)',
          'P(A_1985 > A_1975)', 'P(A_1995 > A_1975)']).round(3)
```

与科格利和萨金特一样，这些概率使用了全部抽样；具有 $\mathcal A_t$ 长期解释所需的 $|\psi_3|<1$ 的抽样占比为

```{code-cell} ipython3
pd.Series({year: inside[year].mean() for year in selected_years},
          name='share of draws with |ψ_3| < 1').to_frame().round(3)
```

后验支持1975年政策为消极、1985年和1995年政策为积极的判断，并且表明积极程度很可能有所上升。

但我们所得到的1985年和1995年政策为积极的概率，以及自1975年以来积极程度上升的概率，在我们所做的每一次运行中都明显低于科格利和萨金特的结果。

这与我们后验中更大的漂移协方差是一致的，后者使得任一日期的系数定位都不够精确。

### 反通胀与平静期

{doc}`phillips_escaping_nash` 在结尾处提出了一个问题：80年代中期的平静期与反通胀是否同时到来，因为在一个“平静有助于促成逃逸”的模型中，理应如此。

本文所拟合的模型将二者区分开来：核心通货膨胀衡量的是反通胀过程，而广义创新方差衡量的是平静程度。

下表给出了二者各自的峰值日期，以及二者从峰值回落到1985年之后平均水平一半所需的季度。

```{code-cell} ipython3
def peak_and_halfway(series, first_after=1985):
    """Date of the peak, and of the first later date halfway to the later mean."""
    peak = np.argmax(series)
    later_mean = series[years >= first_after].mean()
    halfway = 0.5 * (series[peak] + later_mean)
    after = peak + np.flatnonzero(series[peak:] <= halfway)[0]
    return dates[peak], dates[after]


pd.DataFrame(
    [peak_and_halfway(core_inflation), peak_and_halfway(logdet_r)],
    index=['core inflation', 'log |E(R_t | T)|'],
    columns=['peak', 'halfway down'],
).round(2)
```

核心通货膨胀率先下降：在达到峰值一年之后就已经回落到一半水平，而总创新方差则在更晚之后才达到峰值，直到1984年前后才回落到一半水平。

反通胀先于大部分平静期出现，因此这一时间顺序并不支持“平静使得向低通胀的逃逸成为可能”这一观点，尽管简化形式的 VAR 无法确定因果关系问题。

(csdv_tests)=
## 为什么经典检验未能发现漂移

一个漂移程度如此之大的模型，怎么可能与那些未能拒绝时间不变性的经典检验共存？

科格利和萨金特在其第 3.7 节中给出的答案是：这些检验对于其后验分布所描述的那种漂移几乎没有检验力。

他们首先对自己的数据应用了伯南克和米霍夫的检验方法，即针对未知日期单一断点的 {cite:t}`Andrews1993` sup-LM 检验，结果与伯南克和米霍夫一样未能拒绝原假设。

只有当该检验在漂移确实存在的情况下很可能会拒绝原假设时，未能拒绝才能作为反对漂移的证据。

为了衡量这一概率，他们从后验分布中抽取 $(\theta^T, H^T, B)$，根据每次抽取模拟出一个人工样本，应用该检验，并在 10,000 次重复中统计在 5% 水平上的拒绝次数。

他们对 {cite:t}`Nyblom1989` 和 {cite:t}`Hansen1992` 的检验（该检验专为以随机游走方式漂移的参数设计）以及安德鲁斯检验的 sup-Wald 版本也做了同样的分析。

| 后验分布下的拒绝频率 | 名义利率 | 失业率 | 通货膨胀 | VAR |
|---|---:|---:|---:|---:|
| 安德鲁斯 sup-LM（表 4） | 0.136 | 0.172 | 0.112 | 0.252 |
| 尼布洛姆-汉森（表 5） | 0.076 | 0.170 | 0.086 | 0.234 |
| 安德鲁斯 sup-Wald（表 6） | 0.173 | 0.269 | 0.711 | 0.296 |

在实际数据中，sup-LM 检验和尼布洛姆-汉森检验对每个方程以及整个 VAR 都未能拒绝原假设。

然而，即便当 $\theta$ 确实如后验分布所示那样发生漂移，这些检验未能拒绝的频率也大约是拒绝频率的三倍甚至更高。

唯一一个检验力高于五成的情形是针对通货膨胀方程的 sup-Wald 检验，而在实际数据中，该检验在 1% 的水平上拒绝了时间不变性；它在 5% 的水平上也拒绝了整个 VAR 的时间不变性。

针对 {cite:t}`Clarida2000` 政策规则参数或 $\theta_t$ 的第一主成分的检验，同样显示出较低的检验力。

因此，{cite:t}`BernankeMihov1998` 和 {cite:t}`Sims1999` 未能拒绝原假设，正是即便系数确实如本文所示发生如此程度的漂移时也应预期出现的结果。

检验力低并不令人意外。

漂移系数模型在 $Q=0$ 处嵌套了时间不变性：当 $Q$ 接近零时，检验力接近检验水平（size）；当 $Q$ 非常大时，检验几乎必然拒绝原假设；而介于两者之间的，则是一段具有经济意义的漂移区间，而这些检验通常无法捕捉到这一区间内的漂移。

与此相关的是，{cite:t}`StockWatson1998` 指出，当漂移方差较小时，其最大似然估计常常恰好为零，因此"无漂移"的经典估计同样只是薄弱的证据。

科格利和萨金特最后给出了一个更深层次的原因，这一原因在连续时间框架下 {cite}`AndersonHansenSargent2003` 表现得最为清晰。

扩散过程的波动率可以从路径上任意短暂一段的二次变差中恢复出来，但其漂移却只能通过长时间跨度的数据缓慢地习得。

因此，时间序列中*系统性*部分的变动要比其*波动率*的变动难以察觉得多，而这恰恰就是"运气不好"证据两个方面之间的不对称性：方差变化的证据强有力，而反对系数变化的证据却十分薄弱。

{ref}`csdv_ex3` 在一个具有漂移系数的简单自回归模型中计算了尼布洛姆-汉森检验的检验力。

(csdv-verdict)=
## 政策不当还是运气不好？一个结论

贝叶斯 VAR 模型对本讲座开篇提出的问题给出了一个细致入微的答案。

- *波动率发生了漂移：* 冲击的大小发生了巨大的变化，在1980年前后出现了一次飙升，并在80年代中期恢复平静，因此运气不好的说法确实抓住了某些真实的东西。

- *系数也发生了漂移：* 核心通货膨胀和通货膨胀持续性在整个70年代上升，并在80年代下降，而估计的政策规则也从被动转为主动； {ref}`csdv_ex2` 表明，冻结系数无法再现1980年后持续性的下降。

- *经典检验并不构成反证：* 它们对这类漂移的检验力很弱。

简化形式的 VAR 本身无法证明是美联储信念的变化导致了系数漂移，因为私人部门行为和其他未被纳入模型的机制同样能够改变简化形式的动态。

还有一个转折，它把我们带回到本节的理论。

{doc}`phillips_learning` 和
{doc}`phillips_escaping_nash` 中的逃逸路径模型对持续性做出了一个明确的预测。

在高通胀的自我实现均衡处，政府的通货膨胀目标几乎不变，因此通货膨胀几乎是序列不相关的；在静态模型中，它就是白噪声。

只有在从高通胀向低通胀*转变的过程中*，当信念以及随之而来的平均通货膨胀在任何数据窗口内迅速变化时，所测得的持续性才会上升。

因此这些模型预测，持续性应当在反通胀过程中达到峰值，而在通货膨胀高企时则较低——用 {cite:t}`CogleySargentConquest2005`（脚注28）的话来说，这是通货膨胀水平与其持续性之间的一种反向关系。

数据却显示出相反的情况：当核心通货膨胀最高时，持续性也最高，并且两者在1980年后一同下降。

{ref}`csdv_ex4` 在 {doc}`phillips_escaping_nash` 的模型中检验了这一预测。

这种正相关关系促使人们提出了一种不同的机制，即一位*不愿实施反通胀*的决策者。

在 {cite:t}`CogleySargentConquest2005` 中，一位对多个相互竞争的菲利普斯曲线模型进行平均的美联储，在70年代想要降低通货膨胀，但行动却非常缓慢，因为一个警告巨大失业成本的模型仍然具有分量，而这种渐进主义恰恰使得在通货膨胀最高时，对目标的偏离变得具有持续性。

在 {cite:t}`Primiceri2006` 中，低估了自然利率和通货膨胀持续性、同时高估了反通胀成本的决策者，同样在70年代行动迟缓，随后才转变方向。

与西姆斯、扎、伯南克和米霍夫的这场友好辩论，因此所做的远不止裁决一个历史问题：它使贯穿本节的学习与漂移理论模型更为精细。

科格利和萨金特的样本止于2000年。

{doc}`phillips_drifts_volatilities_2` 利用涵盖金融危机、零利率下限、疫情以及2021——2022年通货膨胀飙升的数据重新估计了该模型，随后 {doc}`phillips_lost_conquest` 再次回到一个学习型美联储，以解读2020年代的情况。

(csdv-appendix)=
## 附录：采样器详解

### 系数区块

在给定 $Q$ 和 $R^T$ 的条件下，测量方程 {eq}`csdv_measurement` 与随机游走 {eq}`csdv_transition` 构成一个线性高斯状态空间模型，前向递推即为 {doc}`kalman` 中的卡尔曼滤波，从 $\theta_{0\mid 0}=\bar\theta$ 和 $P_{0\mid 0}=\bar P$ 开始：

$$
\begin{aligned}
P_{t\mid t-1} &= P_{t-1\mid t-1} + Q, \\
K_t &= P_{t\mid t-1}X_t\left(X_t^\top P_{t\mid t-1}X_t + R_t\right)^{-1}, \\
\theta_{t\mid t} &= \theta_{t-1\mid t-1} + K_t\left(y_t - X_t^\top\theta_{t-1\mid t-1}\right), \\
P_{t\mid t} &= P_{t\mid t-1} - K_t X_t^\top P_{t\mid t-1}.
\end{aligned}
$$

后向遍历首先抽取 $\theta_T\sim N(\theta_{T\mid T},P_{T\mid T})$，然后对 $t=T-1,\ldots,0$，有

$$
\theta_t \mid \theta_{t+1} \sim
N\left(\theta_{t\mid t} + J_t(\theta_{t+1}-\theta_{t\mid t}),\;
P_{t\mid t} - J_tP_{t+1\mid t}J_t^\top\right),
\qquad
J_t = P_{t\mid t}P_{t+1\mid t}^{-1}.
$$

将随机抽取项设为零，后向遍历就变成了劳赫-东-斯特里贝尔平滑器，在给定 $Q$ 和 $R^T$ 的条件下，给出 $\mathbb{E}(\theta_t\mid T)$。

增益 $J_t$ 以及条件协方差的乔列斯基因子只依赖于 $Q$ 和 $R^T$，因此 `backward_factors` 在每次遍历中只计算一次，每次重新抽取都复用这些结果。

抽取 $\theta_0$ 为 $Q$ 的更新提供了全部 $T$ 个增量。

由于稳定性检验在每次后向遍历之后都会运行，采样器使用 `stable_path`，该函数对所有伴随矩阵的特征多项式一次性应用舒尔-科恩判据，而不是逐一计算其特征值：当且仅当“反射系数”的递推保持在 $(-1,1)$ 内时，根才位于单位圆内，该检验给出与 `is_stable` 相同的结果，但速度快了数倍。

当 $Q=0$ 时，$P_{t+1\mid t}=P_{t\mid t}$，因此 $J_t=I$ 且条件协方差为零：后向遍历返回一条恒定路径，这正是采样器嵌套常系数模型的方式。

重新抽取步骤是拒绝抽样。

在给定 $\Omega$ 的条件下，后向遍历是对整条路径的高斯分布 $N(\hat z, C)$ 的独立抽取，而落入稳定集 $\mathcal S$ 的第一次抽取具有密度

$$
\frac{N(z;\hat z,C)\,\mathbb{1}_{\mathcal S}(z)}{\mathbb{P}\{z\in\mathcal S\mid\Omega\}},
$$

这就是受限的完全条件分布；归一化概率即为 {ref}`csdv-sampler` 恒等式中的 $p(\Omega)$，而且无需对其进行计算。

### 波动率区块

给定其邻居以及正交化残差 $\varepsilon^*_{it}$（即 $B\varepsilon_t$ 的第 $i$ 个分量），$h_{it}$ 的条件密度正比于

$$
h_{it}^{-1/2}\exp\left(-\frac{(\varepsilon^{*}_{it})^2}{2h_{it}}\right)
\times
\frac{1}{h_{it}}
\exp\left(-\frac{\left(\log h_{it}-\tfrac12(\log h_{i,t-1}+\log h_{i,t+1})\right)^2}{\sigma_i^2}\right).
$$

提议分布从正态因子中抽取 $\log h_{it}$，均值为 $\tfrac12(\log h_{i,t-1}+\log h_{i,t+1})$，方差为 $\sigma_i^2/2$，因此梅特罗波利斯接受比只涉及似然因子。

最后一个日期只有一个邻居，而 $\log h_{i0}$ 具有共轭正态条件分布。

(csdv-mc-error)=
### 蒙特卡洛误差

连续的遍历之间存在相关性，因此链的抽取所携带的信息量低于其数量所暗示的水平。

$\operatorname{tr}(Q)$ 在所有遍历（包括预烧期）中的轨迹，展示了该链在 $Q$ 方向上移动得有多缓慢。

原因在于 $Q$ 与系数路径之间的依赖关系：给定路径，其 $T$ 个增量能将 $Q$ 紧紧地固定住；而给定 $Q$，下一条路径的粗糙程度也随之被固定，因此两者只能逐步地一起移动。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: $\operatorname{tr}(Q)$ at every sweep
    name: fig-csdv-trace
---
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(posterior['trace_q'], lw=0.5)
ax.axvline(burn, color='0.45', ls='--', lw=1, label='end of burn-in')
ax.set_xlabel('sweep')
ax.set_ylabel(r'$\mathrm{tr}(Q)$')
ax.legend()
plt.show()
```

积分自相关时间 $\tau$ 衡量了多少次遍历才相当于一次独立抽取；我们使用索卡尔自适应窗口法对其进行估计。

```{code-cell} ipython3
def integrated_autocorrelation_time(x, c=5):
    """Sokal's adaptive-window estimate of the integrated autocorrelation time."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    τ = 2 * np.cumsum(acf / acf[0]) - 1
    window = np.flatnonzero(np.arange(n) >= c * τ)
    return τ[window[0]] if window.size else τ[-1]   # chain too short for the window


τ_trace_q = integrated_autocorrelation_time(trace_q)
pd.Series({
    'sweeps after burn-in': len(trace_q),
    'autocorrelation time of tr(Q)': τ_trace_q,
    'effective sample size of tr(Q)': len(trace_q) / τ_trace_q,
}, name='value').to_frame().round(1)
```

对于其他重点数值，我们使用批均值法：将保留下来的抽取样本分成十个连续的批次，在每个批次上重新计算各统计量，并用批次间标准差除以 $\sqrt{10}$ 来估计完整运行结果的蒙特卡洛标准误。

```{code-cell} ipython3
def headline_statistics(s):
    """Statistics quoted in the lecture, from the retained draws in slice s."""
    θ_s = posterior['θ'][s].mean(axis=0)
    r_s = mean_innovation_covariance(posterior['h'][s], posterior['β'][s])
    core, natural = local_means(θ_s)
    persistence = persistence_path(θ_s, r_s)
    eigenvalues = np.linalg.eigvalsh(posterior['Q'][s].mean(axis=0))[::-1]
    return np.r_[
        np.trace(posterior['Q'][s], axis1=1, axis2=2).mean(),
        eigenvalues[0] / eigenvalues.sum(),
        100 * core[annual].max(),
        np.corrcoef(core, natural)[0, 1],
        np.corrcoef(core, persistence)[0, 1],
        persistence[annual].max(),
        [event[s].mean() for event in activism_events],
    ]


n_batches = 10
batch = n_draws // n_batches
batch_values = np.array([headline_statistics(slice(b * batch, (b + 1) * batch))
                         for b in range(n_batches)])
pd.DataFrame({
    'estimate': headline_statistics(slice(None)),
    'Monte Carlo s.e.': batch_values.std(axis=0, ddof=1) / np.sqrt(n_batches),
    'Cogley and Sargent': [0.047, 0.485, 8.0, 0.748, 0.92, np.nan,
                           0.208, 0.919, 0.941, 0.923, 0.943],
}, index=['posterior mean of tr(Q)', 'share of first principal component',
          'peak core inflation (%)', 'corr(core, natural rate)',
          'corr(core, persistence)', 'peak g(0)', 'P(A_1975 > 1)',
          'P(A_1985 > 1)', 'P(A_1995 > 1)', 'P(A_1985 > A_1975)',
          'P(A_1995 > A_1975)']).round(3)
```

这些标准误是粗略的，因为每个批次只跨越了 $\operatorname{tr}(Q)$ 的几个自相关时间。

对于 $\operatorname{tr}(Q)$、核心通货膨胀的峰值以及积极性概率而言，使用其他种子得到的运行结果与此次结果之间的差异，比标准误所暗示的要大。

正文中的论述是那些在我们所有运行中都成立的结果：漂移的结构、核心通货膨胀及持续性上升和下降的时间点、二者之间的高度相关性，以及政策从被动转向主动的转变，但并非精确的数值大小。

## 练习

```{exercise}
:label: csdv_ex1

考虑一个平稳的 $AR(1)$ 过程，$\pi_t=\rho\pi_{t-1}+w_t$，其中 $|\rho|<1$。

1. 利用 {eq}`csdv_spectrum`，将伴随矩阵替换为标量 $\rho$，证明
   $g(0)=(1+\rho)/[2\pi(1-\rho)]$，并反转该公式，将 $\rho$ 表示为 $g(0)$ 的函数。
2. 对 `feature_draws['persistence']` 中的持续性抽样逐个应用该反转公式，并绘制所得“等效 $AR(1)$”系数
   $\rho_t$ 的后验中位数及逐点90%区间。
3. 哪个 $\rho$ 值对应于白噪声？哪个对应于 $g(0)=1$？等效的 $\rho_t$ 在20世纪60年代初、70年代以及90年代之间是如何变化的？
```

```{solution-start} csdv_ex1
:class: dropdown
```

对于 $AR(1)$ 过程，$f(0)=\sigma_w^2/[2\pi(1-\rho)^2]$，而方差为
$\sigma_w^2/(1-\rho^2)$，因此

$$
g(0)=\frac{1-\rho^2}{2\pi(1-\rho)^2}=\frac{1+\rho}{2\pi(1-\rho)},
\qquad
\rho=\frac{2\pi g(0)-1}{2\pi g(0)+1}.
$$

白噪声时，$g(0)=1/(2\pi)$，对应 $\rho=0$；而 $g(0)=1$ 时，对应
$\rho=(2\pi-1)/(2\pi+1)\approx 0.73$。

该反转函数关于 $g(0)$ 是递增的，因此它能将 $g_{\pi\pi}(0,t)$ 的后验分位数映射为 $\rho_t$ 的后验分位数。

```{code-cell} ipython3
def equivalent_ar1(g):
    """AR(1) coefficient with the same normalized power at frequency zero."""
    return (2 * np.pi * g - 1) / (2 * np.pi * g + 1)


ρ_draws = equivalent_ar1(feature_draws['persistence'])
lower, median, upper = np.quantile(ρ_draws, (0.05, 0.5, 0.95), axis=1)

fig, ax = plt.subplots()
ax.plot(dates[annual], median, lw=2, label='posterior median')
ax.fill_between(dates[annual], lower, upper, alpha=0.2, label='90 percent band')
ax.axhline(0, color='0.45', lw=1)
ax.set_xlabel('year')
ax.set_ylabel(r'equivalent $AR(1)$ coefficient $\rho_t$')
ax.legend()
plt.show()
```

下表给出了该图背后的数字。

```{code-cell} ipython3
pd.Series({
    period: np.median(ρ_draws[(years[annual] >= first) & (years[annual] <= last)])
    for period, (first, last) in {'1960--1964': (1960, 1964),
                                  '1970--1979': (1970, 1979),
                                  '1990--2000': (1990, 2000)}.items()
}, name='median equivalent ρ').to_frame().round(2)
```

等效系数在20世纪60年代初很小，在70年代约为0.9，而在90年代则低于0.5，尽管70年代的区间较宽。

```{solution-end}
```

```{exercise}
:label: csdv_ex2

要对漂移系数与恒定系数进行正式的比较，需要用 $Q=0$（即 VAR 系数恒定、只有随机波动率 $H_t$ 变动的特殊情形）重新拟合模型。

该抽样器支持这一点：向 `run_sampler` 传入
`fixed_q=np.zeros((n_coefficients, n_coefficients))`，即可将 $Q$ 固定为零。

将这个恒定系数模型拟合到同一样本，并将其 $g_{\pi\pi}(0,t)$ 与 {numref}`fig-csdv-inflation-persistence` 中的漂移系数路径进行对比。

仅靠漂移的波动率能否解释所度量的持续性的上升和下降？
```

```{solution-start} csdv_ex2
:class: dropdown
```

在 $Q=0$ 的情形下，系数路径恒定不变（见 {ref}`csdv-appendix`），因此 $A_{t\mid T}$ 不再变动，
$g_{\pi\pi}(0,t)$ 只能通过 $R_t$ 的构成发生变化。

恒定系数链的混合速度要快得多，因此较短的抽样链即可满足需要。

```{code-cell} ipython3
constant_posterior = run_sampler(
    data['y'], data['x'], prior,
    n_sweeps=4_000, burn=1_000, thin=10, seed=42,
    fixed_q=np.zeros((n_coefficients, n_coefficients)),
)
constant_θ = constant_posterior['θ'].mean(axis=0)
constant_r = mean_innovation_covariance(constant_posterior['h'],
                                        constant_posterior['β'])
constant_persistence = persistence_path(constant_θ, constant_r)

fig, ax = plt.subplots()
ax.plot(dates[annual], inflation_persistence[annual], 'o-', lw=2,
        markersize=3, label='drifting coefficients')
ax.plot(dates[annual], constant_persistence[annual], 's-', lw=2,
        markersize=3, label=r'constant coefficients ($Q=0$)')
ax.axhline(white_noise, color='0.45', ls='--', lw=1, label='white noise')
ax.set_xlabel('year')
ax.set_ylabel(r'$g_{\pi\pi}(0,t)$')
ax.legend(loc='center right')
plt.show()
```

下表给出了该图背后的数字。

```{code-cell} ipython3
after_1981 = years[annual] >= 1982
pd.DataFrame({
    label: {
        '1960--1964 mean': period_mean(path, 1960, 1964),
        '1970--1979 mean': period_mean(path, 1970, 1979),
        'peak': path[annual].max(),
        'year of the peak': years[annual][np.argmax(path[annual])],
        '1990--2000 mean': period_mean(path, 1990, 2000),
        'share of 1982--2000 years below one': np.mean(path[annual][after_1981] < 1),
    }
    for label, path in (('drifting coefficients', inflation_persistence),
                        ('constant coefficients', constant_persistence))
}).round(2)
```

在恒定系数的情形下，所度量的持续性仍会变动，因为即便 $R_t$ 的总体尺度已被归一化去除，其构成仍会发生变化；
恒定系数路径同样在1980年前后达到峰值，此时通货膨胀创新相对于其他创新变得更大。

但它遗漏了这个故事的两端。

它使得20世纪60年代初的持续性远高于白噪声水平，而在1980年之后，它一直维持在一以上，90年代的持续性与70年代的水平相差不远，而漂移系数路径在这些年份中的大多数（甚至全部）都低于一。

仅靠漂移的波动率能够解释所度量持续性中的部分变动，但既无法解释其在60年代初的低水平，也无法解释其在1980年之后的崩溃式下降——这正是为何要回答政策不当还是运气不好这个问题，需要同时考虑这两个渠道。

```{solution-end}
```

```{exercise}
:label: csdv_ex3

对于一个由最小二乘法估计、残差为 $\hat u_t$ 的回归 $z_t=x_t^\top b+u_t$，{cite:t}`Nyblom1989` 与
{cite:t}`Hansen1992` 所提出的、用于检验所有系数恒定性的统计量为

$$
L=\frac{1}{T^2}\sum_{t=1}^T S_t^\top V^{-1} S_t,
\qquad
S_t=\sum_{s\leq t}x_s\hat u_s,
\qquad
V=\frac{1}{T}\sum_{s=1}^T x_sx_s^\top\hat u_s^2.
$$

将其应用于带截距的 $AR(1)$ 过程 $y_t=0.5+\rho_t y_{t-1}+w_t$，$w_t\sim N(0,1)$，取
$T=168$、$\rho_0=0.6$，每种情形重复2,000次。

1. 模拟 $\rho_t$ 恒定时的5%临界值，并将其与汉森关于两个系数的渐近值0.749进行比较。
2. 计算当 $\rho_t$ 为随机游走（限制在 $[-0.97, 0.97]$ 内）、创新方差
   $q\in\{10^{-4}, 4\times 10^{-4}, 10^{-3}, 4\times 10^{-3}\}$ 时的拒绝频率。
3. 计算当 $\rho$ 恒定、但 $w_t$ 的标准差在样本中点变为原来三倍时的拒绝频率。

将你的结果与科格利和萨金特在 {ref}`csdv_tests` 中报告的检验功效联系起来。
```

```{solution-start} csdv_ex3
:class: dropdown
```

我们一次性模拟全部重复次数。

```{code-cell} ipython3
def simulate_ar1_panel(n, T, q, rng, volatility_break=False):
    """n replications of y_t = 0.5 + ρ_t y_{t-1} + s_t w_t."""
    y = np.zeros((n, T + 1))
    ρ = np.full(n, 0.6)
    for t in range(1, T + 1):
        ρ = np.clip(ρ + np.sqrt(q) * rng.standard_normal(n), -0.97, 0.97)
        scale = 3.0 if volatility_break and t > T // 2 else 1.0
        y[:, t] = 0.5 + ρ * y[:, t - 1] + scale * rng.standard_normal(n)
    return y


def nyblom_hansen(y):
    """Nyblom-Hansen L statistic for the intercept and slope, by replication."""
    X = np.stack((np.ones_like(y[:, :-1]), y[:, :-1]), axis=2)
    z = y[:, 1:]
    b = np.linalg.solve(np.einsum('nti,ntj->nij', X, X),
                        np.einsum('nti,nt->ni', X, z)[:, :, None])[:, :, 0]
    scores = X * (z - np.einsum('nti,ni->nt', X, b))[:, :, None]
    S = np.cumsum(scores, axis=1)
    V = np.einsum('nti,ntj->nij', scores, scores) / z.shape[1]
    return np.einsum('nti,nij,ntj->n', S, np.linalg.inv(V), S) / z.shape[1]**2


rng = np.random.default_rng(1234)
n, T = 2_000, 168
critical_value = np.quantile(nyblom_hansen(simulate_ar1_panel(n, T, 0.0, rng)), 0.95)

rows = {}
for q in (1e-4, 4e-4, 1e-3, 4e-3):
    L = nyblom_hansen(simulate_ar1_panel(n, T, q, rng))
    rows[f'random-walk ρ, q = {q:g}'] = {
        'sd of the change in ρ over the sample': np.sqrt(T * q),
        'rejection frequency': np.mean(L > critical_value),
    }
L = nyblom_hansen(simulate_ar1_panel(n, T, 0.0, rng, volatility_break=True))
rows['constant ρ, shock sd triples'] = {
    'sd of the change in ρ over the sample': 0.0,
    'rejection frequency': np.mean(L > critical_value),
}
print(f'simulated 5 percent critical value: {critical_value:.3f}')
pd.DataFrame(rows).T.round(3)
```

模拟得到的临界值与汉森的渐近值接近。

检验功效随 $q$ 的增大而上升，但即使 $\rho$ 在样本期内变化的标准差约为0.8，功效仍低于50%；对于那些使 $\rho$
在样本期内漂移几成的情形，该检验只有五分之一到三分之一的概率能够检测出来，这与科格利和萨金特针对其后验分布所发现的0.08到0.23同一数量级。

在 $\rho$ 恒定、冲击标准差变为原来三倍的情形下，拒绝频率并不高于检验本身5%的名义水平，说明该检验不会将波动率的变化误判为漂移——它只是在面对漂移时功效较弱。

```{solution-end}
```

```{exercise}
:label: csdv_ex4

逃逸路径模型能否预测出我们在 {ref}`csdv_persistence` 中发现的平均通货膨胀与持续性之间的正相关关系？

模拟 {doc}`phillips_escaping_nash` 中的静态模型，取 $\theta=1$、$u=5$、$\sigma_1=\sigma_2=0.3$，在恒定增益最小二乘法下，取增益
$\varepsilon=0.02$，模拟40,000期，从自我确认均衡出发。

在40期的滚动窗口中，对通货膨胀拟合一个 $AR(1)$ 过程，计算其
$g(0)=(1+\hat\rho)/[2\pi(1-\hat\rho)]$，并记录平均通货膨胀。

1. 比较处于纳什均衡（两个半段的平均通货膨胀均高于4）、接近拉姆齐均衡（两个半段均低于1），以及跨越一次下降（前半段高于3.5、后半段低于1.5）的窗口中
   $g(0)$ 的中位数。
2. 计算滚动平均通货膨胀与滚动 $g(0)$ 之间的相关性，并将其与数据中核心通货膨胀与持续性之间的相关性进行比较。
```

```{solution-start} csdv_ex4
:class: dropdown
```

该模拟重复了 {doc}`phillips_escaping_nash` 中的 `simulate_path` 函数，其中关于所感知的菲利普斯曲线斜率与截距的信念为
$\gamma=(\gamma_1,\gamma_{-1})$，即 $U=\gamma_1\pi+\gamma_{-1}$。

```{code-cell} ipython3
def simulate_escapes(ε, n_periods, θ=1.0, u=5.0, σ1=0.3, σ2=0.3, seed=0):
    """Inflation in the static CWS model under constant-gain least squares."""
    rng = np.random.default_rng(seed)
    γ = np.array([-θ, u * (1 + θ**2)])                     # SCE beliefs
    x = -γ[1] * γ[0] / (1 + γ[0]**2)
    R = np.array([[x**2 + σ2**2, x], [x, 1.0]])
    π = np.empty(n_periods)
    for n in range(n_periods):
        x = -γ[1] * γ[0] / (1 + γ[0]**2)                    # Phelps choice
        w1, w2 = rng.standard_normal(2)
        π[n] = x + σ2 * w2
        U = u - θ * σ2 * w2 + σ1 * w1
        Φ = np.array([π[n], 1.0])
        R = R + ε * (np.outer(Φ, Φ) - R)
        γ = γ + ε * np.linalg.solve(R, Φ * (U - Φ @ γ))
    return π


π_sim = simulate_escapes(0.02, 40_000, seed=2)
windows = np.lib.stride_tricks.sliding_window_view(π_sim, 40)[::2]
centered = windows - windows.mean(axis=1, keepdims=True)
ρ_hat = np.clip(np.sum(centered[:, 1:] * centered[:, :-1], axis=1)
                / np.sum(centered[:, :-1]**2, axis=1), -0.99, 0.99)
g0 = (1 + ρ_hat) / (2 * np.pi * (1 - ρ_hat))
first, second = windows[:, :20].mean(axis=1), windows[:, 20:].mean(axis=1)

groups = {
    'at Nash': (first > 4) & (second > 4),
    'near Ramsey': (first < 1) & (second < 1),
    'spanning a fall': (first > 3.5) & (second < 1.5),
}
display(pd.DataFrame({
    'windows': {label: mask.sum() for label, mask in groups.items()},
    'median g(0)': {label: np.median(g0[mask]) for label, mask in groups.items()},
}).round(2))
print(f'correlation of rolling mean inflation and rolling g(0): '
      f'{np.corrcoef(windows.mean(axis=1), g0)[0, 1]:.3f}')
print(f'correlation of core inflation and persistence in the data: '
      f'{core_persistence_correlation:.3f}')
```

无论是处于纳什均衡还是接近拉姆齐均衡，所度量的持续性都接近白噪声值 $1/(2\pi)$；只有在跨越一次下降的窗口中，持续性才较高，此时窗口内部不断变化的均值看起来就像是持续性。

在整个模拟过程中，滚动平均通货膨胀与滚动持续性几乎不相关，与数据中那种强烈的正相关关系相去甚远。

逃逸路径模型将持续性与通货膨胀的*变化*联系在一起，而不是与其*水平*联系在一起：自我确认均衡下的高通货膨胀伴随着白噪声式的通货膨胀，而持续性则在通货膨胀下降时达到峰值，这正是
{cite:t}`CogleySargentConquest2005` 所描述的那种反向关系。

```{solution-end}
```
