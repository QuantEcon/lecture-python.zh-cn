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
  title: 失落的征服：2020 年代的美联储政策
  headings:
    Overview: 概览
    The three elements: 三个要素
    The Fed's drifting-coefficients beliefs: 美联储的漂移系数信念
    The Fed's Phelps problem: 美联储的菲尔普斯问题
    The Fed's Phelps problem::How the rule drifted: 该规则是如何漂移的
    'Why the Fed was slow: a counterfactual': 美联储为何反应缓慢：一个反事实分析
    Real-time data and the delay: 实时数据与延迟
    How beliefs shape the Phelps rule: 信念如何塑造菲尔普斯规则
    From policy to measured beliefs: 从政策到测度信念
    From policy to measured beliefs::Nature's model: 自然的模型
    From policy to measured beliefs::What the Fed measures: 美联储测度了什么
    From policy to measured beliefs::The self-confirming fixed point: 自我确认的不动点
    From policy to measured beliefs::A trap without an escape route: 没有逃脱路径的陷阱
    Was the conquest lost?: 征服是否得而复失？
    Exercises: 练习
---

(phillips_lost_conquest)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 失落的征服：2020 年代的美联储政策

```{index} single: Phillips Curve; Fed Policy in the 2020s
```

```{contents} Contents
:depth: 2
```

## 概览

上一讲 {doc}`phillips_drifts_volatilities_2` 将 {cite:t}`CogleySargent2005` 中的漂移系数证据延伸到了 2000 年之后：它探讨了通货膨胀的简化式动态以及冲击其的波动率在《征服》一书写就后数十年间是如何演变的。

本系列的最后一讲则转而采用政策制定者的视角。

它遵循 {cite:t}`SargentWilliams2025` 的思路，将漂移的信念赋予美联储，并追问这些信念在 2020 年代的通货膨胀中建议了怎样的政策。

他们的人工美联储是由此前讲座中开发的工具构建而成：

* 一个 {doc}`菲尔普斯控制问题 <phillips_adaptive>`；
* {doc}`phillips_learning` 中遇到的 {cite:t}`Kreps1998` 的*预期效用*假设；
* 一个通过*常增益递归最小二乘法*估计的*漂移系数*模型，它是 {doc}`phillips_priors` 中卡尔曼滤波的常增益特例。

这是一个仍在持续的谜题。

在 COVID-19 大流行之后，美国通货膨胀率飙升至上世纪 80 年代初以来的最高水平，这正是 {doc}`phillips_two_stories` 中 1999 年后数据末尾所展示的那段情节。

该讲座绘制的是 CPI 通货膨胀率；而本讲座遵循论文的做法，使用个人消费支出（PCE）价格指数，这是美联储偏好的衡量指标。

然而，联邦储备委员会（美联储）超过一年时间没有作出反应。

它在 2022 年 3 月首次上调政策利率，直到 2022 年第二季度才开始真正大力收紧，而此时通货膨胀已经处于峰值。

美联储为何反应如此迟缓？

{cite:t}`SargentWilliams2025` 构建了一个*人工美联储*，它在每个季度都会：

* 通过常增益递归最小二乘法重新估计一条漂移系数的菲利普斯曲线，并且
* 求解一个线性二次型菲尔普斯问题——将其最新估计值视为将永远成立——以设定其利率工具。

控制工程师将这种做法称为*模型预测控制*。

这是推动 {doc}`phillips_learning` 中学习型政府的*先估计再优化*循环，只是现在的工具变成了政策利率，而信念所涉及的是通货膨胀的*持续性*和菲利普斯曲线的*斜率*。

三个要素塑造了人工美联储的决策：

1. **通货膨胀持续性下降。** 美联储已经了解到通货膨胀冲击消退得比以前更快，因此这次飙升看起来是*暂时性的*。
2. **更平坦的菲利普斯曲线。** 美联储已经了解到通货膨胀对经济松弛的反应很弱，因此抑制通货膨胀看起来*代价高昂*。
3. **实时产出缺口的测量误差。** 在实时情况下，美联储所感知到的松弛程度比现今修订后的数据所显示的要大。

我们从公开数据中重现了这三个要素。

主要发现——也是论文的主要发现——可以用一句话概括：

> 漂移的信念使反应变得*温和*；实时测量误差使反应变得*滞后*。

使用如今修订后的数据，人工美联储对 2021 年的飙升反应温和，但却比实际美联储的反应*更早*。

只有当使用美联储在实时所看到的产出缺口时，其反应的时点才与美联储实际的反应时点大致吻合。

本讲座的最后一部分是我们自己的拓展内容。

它探讨了这些良性信念的来源：过去的强力政策是否能使通货膨胀*看起来*是暂时性的，并使菲利普斯曲线*看起来*趋于平坦？

我们计算了这一循环的自我确认不动点，发现与 {doc}`phillips_escaping_nash` 中的政府不同，这个美联储没有逃脱路线。

```{note}
从有限理性系列讲座而来的读者会发现，$\gamma$ 仍然扮演着其一贯的角色——学习增益。

在那里，信念被称为 $\alpha$（参见 {ref}`br_notation`、{ref}`olg_phillips`）；
而在这里，它们是 $\theta_t$。
```

让我们导入所需的库：

```{code-cell} ipython3
import time
import matplotlib.pyplot as plt
import matplotlib as mpl  # i18n
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"  # i18n
mpl.font_manager.fontManager.addfont(FONTPATH)  # i18n
plt.rcParams['font.family'] = ['Source Han Serif SC']  # i18n
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
from scipy.optimize import brentq
```

## 三个要素

前两个要素是现代宏观经济学中记录最为详实的事实之一。

**持续性下降。** 从 20 世纪 70 年代到 80 年代，通货膨胀具有很强的持续性：一次冲击会使通货膨胀持续数年之久。

{cite:t}`CogleySargent2001` 和 {cite:t}`CogleySargent2005` 记录了沃尔克-格林斯潘征服通胀时期通货膨胀持续性的下降。

{doc}`phillips_drifts_volatilities` 再现了他们的证据（{ref}`csdv_persistence`），{doc}`phillips_drifts_volatilities_2` 则将其延续到 2000 年之后（{ref}`csdv2_persistence`）。

{cite:t}`StockWatson2007` 发现，通货膨胀的永久成分在 20 世纪 80 年代和 90 年代有所缩小。

{cite:t}`BenatiSurico2008` 和 {cite:t}`CarlstromFuerstPaustian2009` 将持续性的下降与更为积极主动的货币政策联系起来，我们将在 {ref}`lc_feedback` 中探讨这一联系。

{cite:t}`PivettaReis2007` 提醒道，这种表面上的下降部分可能反映的是冲击规模变小，而非结构性断裂；这正是 {doc}`phillips_drifts_volatilities` 所区分的系数漂移与波动率漂移之间的差异。

前美联储主席珍妮特·耶伦在 2019 年总结了这一证据：“菲利普斯曲线的斜率……自 20 世纪 60 年代以来已显著下降……其次，由于滞后通货膨胀对当前通货膨胀的影响已大幅减弱，通货膨胀的持续性已大大降低。”

**更平坦的菲利普斯曲线。** 自 20 世纪 90 年代以来，菲利普斯曲线斜率的估计值趋于零；大衰退之后的“消失的反通胀”便是最典型的例子。

政策制定者深知这意味着什么。

{cite:t}`Bernanke2022`（第 220 页）写道：“平坦的菲利普斯曲线意味着通货膨胀作为经济过热的指标可靠性降低了。一旦通货膨胀变得过高，将其重新降至目标水平所需付出的失业代价，可能比过去更高。”

值得注意的是，平坦的菲利普斯曲线在此处与在*征服*案例中所指向的政策方向恰恰相反。

在 {cite:t}`Sargent1999` 所述的卢卡斯式模型中，更平坦的曲线意味着意外的通货膨胀对实体经济活动的影响更小，因此菲尔普斯问题建议采取*更低*的通货膨胀。

{doc}`phillips_adaptive`（{ref}`pa_induction`）中的归纳假设——即通货膨胀不具有长期实际效应——正是其极限情形，而这也正是*征服*一文中适应性政府降低通货膨胀的方式。

在 2020 年代的统计模型中，美联储通过调整产出缺口来控制通货膨胀。

在这种情况下，更平坦的曲线意味着更高的牺牲率（{ref}`pa_sacrifice`）：每降低一个百分点的通货膨胀所需付出的产出代价更高，因此菲尔普斯问题建议保持耐心（{cite:t}`SargentWilliams2025`，脚注 5）。

**实时不确定性。** {cite:t}`Orphanides2001` 强调的第三个要素是，产出缺口在*实时*中会被严重误测，尤其是在商业周期的转折点。

在 2020 年至 2023 年间，实时缺口持续*低于*后来修订后的度量值，因此美联储所感知到的经济松弛程度超过了实际情况。

我们将在 {ref}`lc_realtime` 中对此加以度量。

## 美联储的漂移系数信念

我们赋予人工美联储一条具有漂移系数的后顾性菲利普斯曲线：

```{math}
:label: lc_pc

\pi_t = \alpha_{0,t} + \rho_t\, \pi_{t-1} + \kappa_t\, x_t + \varepsilon^{\pi}_t ,
```

其中 $\pi_t$ 是四季度 PCE 通货膨胀率，$x_t$ 是 CBO 产出缺口，$\rho_t$ 是*感知的持续性*，$\kappa_t$ 是*感知的斜率*。

$\rho_t$ 所扮演的角色，相当于《Conquest》中菲尔普斯问题（{ref}`pa_induction`；{cite:t}`SargentWilliams2025`，脚注 11）里滞后通货膨胀系数之和所扮演的角色。

美联储通过常增益递归最小二乘法更新 $\theta_t = (\alpha_{0,t}, \rho_t, \kappa_t)^\top$：

```{math}
:label: lc_rls

\theta_t = \theta_{t-1} + \gamma\, M_t^{-1} X_t\left(\pi_t - X_t^\top \theta_{t-1}\right),
\qquad
M_t = M_{t-1} + \gamma\left(X_t X_t^\top - M_{t-1}\right),
```

其中回归量为 $X_t = (1, \pi_{t-1}, x_t)^\top$，$M_t$ 为矩矩阵。

估计值 $\theta_t$ 使用截至第 $t$ 季度的数据，正是美联储在第 $t$ 季度作出决策时所依据的信念。

矩矩阵在系数之前更新，这是 {cite:t}`EvansHonkapohja2001` 所采用的时序安排。

原论文的公式在系数更新中使用的是滞后矩矩阵 $M_{t-1}$。

除非回归量远远超出其历史范围，否则这两种时序安排之间的差异只是 $\gamma^2$ 阶的量级。

2022 年正是回归量超出历史范围的情形，此时时序安排的不同会显著改变所得到的建议，因此有必要说明代码所采用的是哪一种时序。

增益 $\gamma = 0.03$ 是原论文针对季度数据所作的选择；它对过去的观测值进行折扣，从而使估计值能够追踪漂移。

常增益最小二乘法是 {doc}`phillips_priors`（另见 {doc}`kalman`）中卡尔曼滤波的一个特例，{cite:t}`SargentWilliams2005` 正是在关于系数如何漂移的特定先验下得到了这一特例。

哪个增益最能够合理化美联储的决策，是乔恩问题的一个版本，我们在 {ref}`lc_ex4` 中提出该问题。

我们直接从 FRED 读取 PCE 价格指数、实际 GDP、CBO 潜在产出和联邦基金利率。

如果 FRED 缺失某个孤立月份的数据——正如 2025 年 10 月联邦政府停摆期间部分序列所发生的那样——代码会通过线性插值填补该缺口，并给出相应提示。

```{code-cell} ipython3
FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='

def fred(series_id):
    "直接从 FRED 读取单个序列。"
    s = pd.read_csv(FRED + series_id, index_col=0, parse_dates=True).iloc[:, 0]
    s.index.name = None
    return pd.to_numeric(s, errors='coerce')      # FRED 用 '.' 标记缺口

def quarterly_mean(monthly):
    "对月度序列计算季度平均值，并填补孤立缺失月份。"
    monthly = monthly.asfreq('MS')
    missing = monthly.index[monthly.isna()]
    if len(missing) > 0:
        print(f"{monthly.name}：正在插值",
              [f"{d:%Y-%m}" for d in missing])
    filled = monthly.interpolate(limit=1, limit_area='inside')
    q = filled.resample('QS')
    return q.mean()[q.count() == 3]                # 仅保留完整季度

def qtr(date):
    "用季度标注日期，例如 2022Q2。"
    return f"{date.year}Q{date.quarter}"

pce = quarterly_mean(fred('PCEPI'))
fedfunds = quarterly_mean(fred('FEDFUNDS'))
gdp, potential = fred('GDPC1'), fred('GDPPOT')     # 实际 GDP，CBO 潜在产出

inflation = 100 * (pce / pce.shift(4) - 1)          # 四季度 PCE 通货膨胀率
gap = 100 * (gdp / potential - 1)                   # 产出缺口，百分比

data = pd.concat([inflation.rename('pi'), gap.rename('x'),
                  fedfunds.rename('i')], axis=1, sort=True).dropna().loc['1960':]
print(f"样本区间：{qtr(data.index[0])} 至 {qtr(data.index[-1])}，"
      f"共 {len(data)} 个季度")

peak = data['pi']['2020':].idxmax()
higher = data['pi'][:peak].iloc[:-1]
last_higher = higher[higher > data['pi'][peak]].index[-1]
print(f"PCE 通货膨胀率在 {qtr(peak)} 达到峰值 {data['pi'][peak]:.2f}%，"
      f"为 {qtr(last_higher)} 以来的最高水平")
```

遵循原论文的做法，我们在 2020 年和 2021 年冻结信念，将增益设为零，因为疫情期间的观测值是极端异常值，否则会使估计值剧烈波动；更新在 2022 年恢复。

这种冻结并非无关紧要。

2020 年各季度才是真正的异常值，但正是 2021 年的冻结决定了下文的结果。

如果美联储从 2021 年的数据中学习，其感知的菲利普斯曲线会变得更陡峭，其建议的利率到 2021 年底会跃升至百分之几（{ref}`lc_ex3`）。

在 2021 年将增益设为零，本身就等于判定 2021 年的通胀飙升是暂时性的。

我们从 $\theta = (0.5, 0.9, 0.05)$ 开始递归，而不是采用原论文基于 1949-1959 年样本前期的回归结果；80 年代中期之后的信念路径几乎不受这一选择的影响。

我们只计算一次信念，并在下文重复使用。

```{code-cell} ipython3
def estimate_beliefs(data, γ=0.03, freeze=(2020, 2021)):
    "对漂移菲利普斯曲线进行常增益递归最小二乘估计：α₀、ρ、κ 的路径。"
    π, x = data['pi'].values, data['x'].values
    θ = np.array([0.5, 0.9, 0.05])            # [截距，持续性，斜率]
    M = np.diag([1.0, 10.0, 5.0])             # 矩矩阵
    rows = []
    for t in range(1, len(data)):
        γ_t = 0.0 if data.index[t].year in freeze else γ
        X = np.array([1.0, π[t - 1], x[t]])
        M = M + γ_t * (np.outer(X, X) - M)                     # 先更新 M_t，
        θ = θ + γ_t * np.linalg.solve(M, X * (π[t] - X @ θ))   # 再更新 θ_t
        rows.append(θ)
    return pd.DataFrame(rows, index=data.index[1:],
                        columns=['alpha0', 'rho', 'kappa'])

beliefs = estimate_beliefs(data)
```

以下是感知的持续性和斜率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 美联储感知的通货膨胀持续性和菲利普斯曲线斜率
    name: fig-lc-beliefs
---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(beliefs['rho'])
axes[0].axhline(1, color='k', lw=0.5, ls=':')
axes[0].set_ylabel(r'持续性 $\rho_t$')

axes[1].plot(beliefs['kappa'], color='C1')
axes[1].axhline(0, color='k', lw=0.5, ls=':')
axes[1].set_ylabel(r'斜率 $\kappa_t$')
axes[1].set_xlabel('年份')

plt.tight_layout()
plt.show()

ρ, κ = beliefs['rho'], beliefs['kappa']
after85 = ρ['1985':]
print(f"1985-2001 年间 ρ_t：均值 {ρ['1985':'2001'].mean():.3f}，"
      f"范围 {ρ['1985':'2001'].min():.3f} 至 {ρ['1985':'2001'].max():.3f}")
print(f"1985 年后首次 ρ_t < 0.96 的季度："
      f"{qtr(after85[after85 < 0.96].index[0])}")
print(f"ρ_t 的谷底：{ρ.min():.3f}，出现在 {qtr(ρ.idxmin())}")
print(f"κ_t：{κ['1980':'1984'].mean():.3f}（1980-84 年均值），"
      f"{κ['2009':'2009'].max():.3f}（2009 年峰值），"
      f"{κ['2019-10-01']:.3f}（2019 年第 4 季度）")
print(f"2019 年第 4 季度：ρ = {ρ['2019-10-01']:.3f}；   2022 年第 2 季度：ρ = {ρ['2022-04-01']:.3f}")
```

感知的**持续性** $\rho_t$ 在 20 世纪 70 年代中期跳升，随后在整整四分之一个世纪内——直到 2001 年——一直保持接近 1 的水平。

它在 2000 年代只是缓慢下降，在 2008 年经济衰退之后则大幅下跌，并在 2011 年触底。

{cite:t}`SargentWilliams2025` 描述了 80 年代中期之后的下降，“在 2008 年衰退之后尤为陡峭”；在本次复现中，这一下降主要体现为 2008 年后的那次陡降。

当信念更新在 2022 年恢复时，$\rho_t$ 又跳回接近 1 的水平，恰好也是美联储放弃“暂时性”表述并开始收紧政策的时候。

感知的**斜率** $\kappa_t$ 从 80 年代初的水平下降，在 2008-2009 年经济衰退期间（当时经济短暂出现通货紧缩）出现飙升，到 2019 年时则更低：菲利普斯曲线趋于平坦。

关于 $\rho_t$ 的量级需要说明一点。

由于 $\pi_t$ 是四季度变化率，相邻观测值共享三个季度的数据。

即便季度通货膨胀是序列不相关的，将 $\pi_t$ 对 $\pi_{t-1}$ 回归也会得到接近 $0.75$ 的系数（{ref}`lc_ex5`）。

相对于这一基准，2019 年 $\rho_t$ 的取值意味着*持续性较低*，而非*暂时性*：

```{code-cell} ipython3
ρ_2019 = ρ['2019-10-01']
print(f"在 ρ = {ρ_2019:.3f} 时，通货膨胀冲击的半衰期为："
      f"{np.log(0.5) / np.log(ρ_2019):.1f} 个季度")
```

若采用不存在重叠的季度年化通货膨胀率，同一算法会发现持续性更早、也下降得更多。

该度量在精神上更接近 {doc}`phillips_drifts_volatilities` 中所用的度量；{ref}`lc_ex5` 对两者进行了协调。

尽管如此，到 2019 年，美联储的模型认为通货膨胀的持续性已不如从前，且与经济松弛的联系较弱。

将这些信念输入到菲尔普斯问题中，会得出建议保持耐心的结论。

## 美联储的菲尔普斯问题

美联储在每一季度通过求解一个线性二次型菲尔普斯问题来设定其政策利率，将其*当前*的估计值视为将永远成立。

这正是我们在 {doc}`phillips_learning` 中遇到的 {cite:t}`Kreps1998` 的预期效用假设。

信念菲利普斯曲线 {eq}`lc_pc` 与一条 IS 曲线相配对：

$$
x_t = b_0 + b_1 x_{t-1} + g\,(i_{t-1} - \pi_{t-1}) + \varepsilon^x_t ,
$$

其系数通过对全样本进行一次性 OLS 估计得到。

两者结合，给出状态 $s_t = (1, \pi_t, x_t, i_{t-1})^\top$ 的线性动态：

$$
s_{t+1} = A_t s_t + B_t\, i_t + C \varepsilon_{t+1},
$$

其矩阵取决于第 $t$ 期的信念。

美联储最小化以下目标：

```{math}
:label: lc_loss

\mathbb{E}_t \sum_{j=t}^{\infty}\beta^{j-t}
\Big[ (\pi_j - \pi^*)^2 + \lambda_x\, x_j^2 + \eta\,(i_j - i_{j-1})^2 \Big],
```

其中 $\pi^* = 2$ 是目标值，最后一项对利率的突然变化施加惩罚。

我们使用该论文中的参数取值：$\beta = 0.95$、$\lambda_x = 0.2$ 以及 $\eta = 0.5$。

该论文指出，这些取值是“经过粗略调整，以较合理地匹配观测到的政策结果”。

将每期损失写成 $s^\top Q s + 2 s^\top N i + R\, i^2$（其中 $R = \eta$），就把问题转化为一个带有交叉项的贴现线性二次型调节器问题（参见 {doc}`lqcontrol`）。

预期贴现损失 $s^\top P_t s$（加上一个常数）的矩阵 $P_t$ 满足如下的贴现里卡提方程：

```{math}
:label: lc_riccati

P = Q + \beta A^\top P A
- (\beta A^\top P B + N)(R + \beta B^\top P B)^{-1}(\beta B^\top P A + N^\top),
```

决策规则为 $i_t = -F_t s_t$，其中 $F_t = (R + \beta B^\top P_t B)^{-1}(\beta B^\top P_t A + N^\top)$。

写出来，该规则是一条*平滑化的泰勒规则*：

$$
i_t = \phi_{0,t} + \phi_{\pi,t}\, \pi_t + \phi_{x,t}\, x_t + \phi_{i,t}\, i_{t-1},
$$

其系数会随着信念的变化而变化。

`scipy.linalg.solve_discrete_are` 求解的是 {eq}`lc_riccati` 的非贴现版本。

传入 $\sqrt\beta A$ 和 $\sqrt\beta B$ 即可吸收贴现因子。

交叉项随后以未缩放的形式，作为 `s=N` 输入，因为 $\beta A^\top P B + N = (\sqrt\beta A)^\top P (\sqrt\beta B) + N$。

按照该论文（{cite:t}`SargentWilliams2025`，第 3.3 节）的做法，我们只是事后施加零利率下限，即 $i^{\rm opt}_t = \max\{-F_t s_t, 0\}$。

美联储的菲尔普斯问题并不会预见到该下限可能会生效。

```{code-cell} ipython3
# IS 曲线，通过对全样本进行一次性 OLS 估计得到
π_d, x_d, i_d = data['pi'].values, data['x'].values, data['i'].values
n = len(data)
Z_is = np.column_stack([np.ones(n - 1), x_d[:-1], i_d[:-1] - π_d[:-1]])
b0, b1, g = np.linalg.lstsq(Z_is, x_d[1:], rcond=None)[0]
print(f"IS 曲线：b0 = {b0:.3f}，b1 = {b1:.3f}，g = {g:.4f}；"
      f"实际利率的长期效应 g/(1-b1) = {g / (1 - b1):.2f}")

β, π_star, λ_x, η = 0.95, 2.0, 0.2, 0.5

def phelps_rule(θ, η=η):
    "在给定信念 θ 的情况下，计算菲尔普斯规则 i = -F s 的系数 (ϕ₀, ϕ_π, ϕ_x, ϕ_i)。"
    α0, ρ, κ = θ
    A = np.array([[1, 0, 0, 0],
                  [α0 + κ * b0, ρ - κ * g, κ * b1, 0],
                  [b0, -g, b1, 0],
                  [0, 0, 0, 0]], float)
    B = np.array([[0], [κ * g], [g], [1]], float)
    c_π = np.array([-π_star, 1, 0, 0.])       # π − π*
    c_x = np.array([0, 0, 1, 0.])             # x
    e_i = np.array([0, 0, 0, 1.])             # i_{t-1}
    Q = np.outer(c_π, c_π) + λ_x * np.outer(c_x, c_x) + η * np.outer(e_i, e_i)
    N = (-η * e_i).reshape(4, 1)
    R = np.array([[η]])
    # √β 吸收了 A 和 B 中的贴现因子；交叉项 N 以未缩放的形式输入
    P = solve_discrete_are(np.sqrt(β) * A, np.sqrt(β) * B, Q, R, s=N)
    F = np.linalg.solve(R + β * B.T @ P @ B, β * B.T @ P @ A + N.T)
    return -F.ravel()

def phelps_rules(beliefs, η=η):
    "给定一系列信念路径，计算每个日期的菲尔普斯规则。"
    return pd.DataFrame([phelps_rule(θ, η) for θ in beliefs.values],
                        index=beliefs.index,
                        columns=['φ_0', 'φ_π', 'φ_x', 'φ_i'])

def recommended_rate(data, rules):
    "建议利率 max(-F_t s_t, 0)：零利率下限在事后施加。"
    π, x, i = data['pi'].values, data['x'].values, data['i'].values
    s = np.column_stack([np.ones(len(data) - 1), π[1:], x[1:], i[:-1]])
    return pd.Series(np.maximum((rules.values * s).sum(axis=1), 0.0),
                     index=rules.index)

rules = phelps_rules(beliefs)
optimal = recommended_rate(data, rules)
```

所估计的利率敏感度 $g$ 很小。

政策内生性（我们在 {ref}`lc_feedback` 中会再次遇到）是回归可能低估利率重要性的一个原因：如果美联储会针对预期中的繁荣采取逆向调控，那么实际利率与随后产出之间的相关性就会被削弱。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 信念驱动的菲尔普斯规则与实际联邦基金利率对比
    name: fig-lc-phelps-rate
---
fig, ax = plt.subplots(figsize=(10, 4.5))
window = slice('1991', None)
ax.plot(optimal[window], 'C0', label="菲尔普斯问题的建议利率")
ax.plot(data['i'][window], 'C3', lw=1, label='实际联邦基金利率')
ax.set_xlabel('年份')
ax.set_ylabel('百分比')
ax.legend()
plt.show()

def fit(rate, start='1991'):
    "相对于实际联邦基金利率的相关性与均方根误差。"
    r = rate[start:]
    actual = data['i'][r.index]
    return np.corrcoef(r, actual)[0, 1], np.sqrt(((r - actual)**2).mean())

no_change = data['i'].shift(1).dropna()               # i_t = i_{t-1}
for name, rate in [('菲尔普斯规则', optimal), ('不变规则', no_change)]:
    c, rmse = fit(rate)
    print(f"{name:>15}：相关性 {c:.3f}，均方根误差 {rmse:.2f} "
          f"（1991-{data.index[-1].year} 年）")
```

在长达三十年的时间跨度里，主观最优利率追踪了实际政策的水平和转折点：20 世纪 90 年代中期的紧缩、2001 年的宽松、2008 年之后的零利率时代，以及始于 2015 年的政策正常化。

它也遗漏了一些时期，在 1998 年和 2002 年都明显低于实际利率。

相关性的高低需要一个基准来衡量。

由于该建议是以实际的滞后利率 $i_{t-1}$ 为条件的，一条简单地设定 $i_t = i_{t-1}$ 的朴素规则在两项指标上都表现*更好*，如输出结果所示。

该论文就一步预测拟合效果也得出了相同的结论。

菲尔普斯规则所提供的，并不是对联邦基金利率更好的预测，而是对该利率的一种解释——以信念和损失函数的方式进行解释。

现在来看 2021 年的情况。

信念是冻结不变的，但状态却不是：随着通货膨胀和产出缺口的上升，该规则建议利率随之上升。

```{code-cell} ipython3
print(pd.DataFrame({'Phelps rule': optimal, 'actual': data['i']})
        .loc['2021':'2022'].rename(index=qtr).round(2).T.to_string())
```

根据今天修订后的数据，这个人造的美联储从 2021 年起就开始提高利率，并在 2022 年初就大幅领先于实际的美联储。

因此，使用当前版本的数据，信念漂移模型实际上*领先*于美联储，而不是追随它，这与该论文在使用修订数据时得出的结论一致（{cite:t}`SargentWilliams2025`，第 3.5 节）。

信念的漂移使得建议的反应力度温和，但并不会使其滞后。

我们将在 {ref}`lc_realtime` 中回到这一滞后问题。

### 该规则是如何漂移的

该论文中的图 8 直接展示了这一机制：菲尔普斯规则的系数，以及政策利率对通货膨胀的长期反应 $\phi_{\pi,t}/(1-\phi_{i,t})$。

我们使用今天的数据来计算这些系数；该论文的图使用的是实时数据。

长期反应大于一即满足泰勒原则。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 菲尔普斯规则的系数及其对通货膨胀的长期反应
    name: fig-lc-rule-coefficients
---
long_run = rules['φ_π'] / (1 - rules['φ_i'])

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(rules['φ_π'], label=r'$\phi_{\pi,t}$')
axes[0].plot(rules['φ_x'], label=r'$\phi_{x,t}$')
axes[0].legend()
axes[0].set_ylabel('响应系数')
axes[1].plot(rules['φ_i'], color='C2')
axes[1].set_ylabel(r'平滑系数 $\phi_{i,t}$')
axes[2].plot(long_run, color='C3')
axes[2].axhline(1, color='k', lw=0.5, ls=':')
axes[2].set_ylabel(r'$\phi_{\pi,t}/(1-\phi_{i,t})$')
axes[2].set_xlabel('年份')
plt.tight_layout()
plt.show()

for date in ['2000-01-01', '2019-10-01']:
    print(f"{qtr(pd.Timestamp(date))} 的长期通货膨胀响应："
          f"{long_run[date]:.2f}")
peak_lr = long_run['2021':'2023'].idxmax()
print(f"2021-2023 年的峰值：{long_run[peak_lr]:.2f}，出现于 {qtr(peak_lr)}")
```

随着持续性和斜率下降，该规则对通货膨胀的长期反应从 2000 年时远高于一，降至 2019 年时仅略高于一，而利率平滑程度则有所上升。

2019 年的规则勉强满足泰勒原则。

当更新于 2022 年恢复，且 $\rho_t$ 跳升时，长期反应也随之跳升，随后随着通货膨胀的回落而有所回落。

因此，{numref}`fig-lc-phelps-rate` 中 2022 年紧缩的一部分来自于规则本身的变化，而不仅仅来自于通货膨胀和产出缺口的上升。

## 美联储为何反应缓慢：一个反事实分析

为了分离出漂移信念所起的作用，我们重新计算菲尔普斯建议利率，这次将信念固定在 2000 年第一季度的取值上。

```{code-cell} ipython3
θ_2000 = beliefs.loc['2000-01-01'].values
print(f"beliefs in 2000Q1: α₀ = {θ_2000[0]:.3f}, ρ = {θ_2000[1]:.3f}, "
      f"κ = {θ_2000[2]:.3f}")
```

2000 年时，美联储的模型存在单位根：$\rho \approx 1$，这意味着通货膨胀冲击被认为是永久性的。

其感知的斜率超过 2019 年取值的两倍以上。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "反事实分析：一个自 2000 年以来未曾更新信念的美联储"
    name: fig-lc-counterfactual
---
rules_2000 = pd.DataFrame(np.tile(phelps_rule(θ_2000), (len(beliefs), 1)),
                          index=beliefs.index, columns=rules.columns)
counterfactual = recommended_rate(data, rules_2000)

fig, ax = plt.subplots(figsize=(10, 4.5))
w = slice('2015', None)
ax.plot(optimal[w], 'C0', label='基准情形（漂移信念）')
ax.plot(counterfactual[w], 'C1--', label='反事实情形（信念冻结于 2000 年）')
ax.plot(data['i'][w], 'C3', lw=1, alpha=0.7, label='实际联邦基金利率')
ax.set_xlabel('年份')
ax.set_ylabel('百分比')
ax.legend()
plt.show()

print(f"counterfactual rate in 2021Q4: {counterfactual['2021-10-01']:.2f}%  "
      f"(actual {data['i']['2021-10-01']:.2f}%)")
```

一个持有 2000 年信念的美联储本应在 2021 年立即而剧烈地收紧政策，到 2021 年底将联邦基金利率推高至超过 5% 的水平，而彼时实际美联储尚未采取任何行动。

两条路径中的目标是相同的，不同之处仅在于信念。

由于将通货膨胀视为持续性较弱，将菲利普斯曲线视为更平坦，美联储自身的菲尔普斯问题告诉它应当温和地作出反应。

这正是 {cite:t}`CogleySargentConquest2005` 对 20 世纪 70 年代所做分析的 21 世纪 20 年代版本。

在那篇文章中，一个贝叶斯式的美联储在三个菲尔普斯问题之间进行竞赛，而最坏情形下的萨缪尔森-索洛模型所预示的高昂反通胀代价，使得即便在数据已经支持自然利率模型之后，政策调整依然是渐进的。

(lc_realtime)=
## 实时数据与延迟

该论文的结论取决于第三个要素。

用论文中的话说，"只有纳入对经济的实时不确定性——这种不确定性在商业周期转折点往往会加剧——我们才能解释所观察到的政策响应延迟"（{cite:t}`SargentWilliams2025`，第 3.5 节）。

我们现在重建美联储在实时情况下所观察到的产出缺口。

ALFRED 是圣路易斯联储的历史数据版本存档库，保存了实际 GDP（`GDPC1`）和 CBO 潜在产出（`GDPPOT`）的每一个历史版本。

对于 2019 年以来的每个季度，我们使用该季度结束后 45 天时的两个序列数据来计算缺口，此时 GDP 初步估计值通常已经公布。

有两类季度需要特别处理。

* 部分 GDP 数据的发布因 2025 年末的联邦政府停摆而延迟。对于在其 45 天版本中缺失的任何季度，我们使用之后包含该季度数据的第一个版本。
* 2023 年 9 月，美国经济分析局将实际 GDP 重新表示为以 2017 年美元计价，而 CBO 的潜在产出直到 2024 年初仍以 2012 年美元计价。由这样的版本计算出的缺口会比较不同单位的序列。我们通过检测 2019 年之前两个序列的比率来发现这一问题，并跳过那些基期不一致的版本。

ALFRED 每次请求最多提供十二个序列，因此代码一次请求两个序列各六个版本的数据，整个历史数据只需几次请求即可获取。

```{code-cell} ipython3
ALFRED = 'https://alfred.stlouisfed.org/graph/alfredgraph.csv'

def alfred_vintages(series, dates, per_request=6, tries=3):
    "Vintages of several series at several dates, a few requests in all."
    frames = []
    for j in range(0, len(dates), per_request):
        vd = [f"{d:%Y-%m-%d}" for d in dates[j:j + per_request]]
        url = (f"{ALFRED}?id={','.join(s for s in series for _ in vd)}"
               f"&vintage_date={','.join(vd * len(series))}")
        for attempt in range(tries):
            try:
                frames.append(pd.read_csv(url, index_col=0, parse_dates=True))
                break
            except Exception as err:
                if attempt == tries - 1:
                    raise RuntimeError(
                        "Could not download data vintages from ALFRED "
                        f"(alfred.stlouisfed.org): {err}. This section "
                        "needs internet access to ALFRED.") from err
                time.sleep(5)
    print(f"{len(frames)} requests to ALFRED")
    return pd.concat(frames, axis=1).apply(pd.to_numeric, errors='coerce')

def real_time_gap(quarters, lag_days=45):
    "Output gap for each quarter from the first suitable vintage after it ends."
    # ALFRED relabels a vintage date it treats as still in the future, as it
    # does with today's date for some hours after midnight in St. Louis
    latest = (pd.Timestamp.now(tz='America/Chicago').tz_localize(None).normalize()
              - pd.Timedelta(days=1))
    vintages = sorted({min(q + pd.offsets.QuarterEnd(0)
                           + pd.Timedelta(days=lag_days), latest)
                       for q in quarters})
    V = alfred_vintages(['GDPC1', 'GDPPOT'], vintages)
    base = slice('2015-01-01', '2018-10-01')        # a pre-pandemic window
    gap, late = {}, []
    for q in quarters:
        target = min(q + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=lag_days),
                     latest)
        for v in vintages:
            if v <= q + pd.offsets.QuarterEnd(0):
                continue
            y, y_pot = V[f'GDPC1_{v:%Y%m%d}'], V[f'GDPPOT_{v:%Y%m%d}']
            if pd.isna(y.get(q)) or pd.isna(y_pot.get(q)):
                continue                                # not yet released
            if abs(np.log(y[base] / y_pot[base]).mean()) > 0.04:
                continue                                # different base years
            gap[q] = 100 * (y[q] / y_pot[q] - 1)
            if v != target:
                late.append(f"{qtr(q)} (vintage {v:%Y-%m-%d})")
            break
    print("quarters taken from a later vintage:", late)
    return pd.Series(gap)

x_real_time = real_time_gap(data.loc['2019':].index)
```

下面将实时缺口与今日数据中的缺口进行对比。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The output gap in real time and in today's data
    name: fig-lc-realtime-gap
---
x_now = data['x'][x_real_time.index]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(x_now, 'C0', label="today's data")
ax.plot(x_real_time, 'C1--', marker='o', ms=3, label='real time (ALFRED)')
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('year')
ax.set_ylabel('percent of potential')
ax.legend()
plt.show()

diff = x_real_time - x_now
below = diff['2020':'2023'] < 0
print(f"real-time gap below today's gap in {below.sum()} of {len(below)} "
      f"quarters of 2020-2023; largest shortfall {-diff.min():.2f} pp "
      f"in {qtr(diff.idxmin())}")
```

从 2020 年到 2023 年，实时缺口在每个季度都低于修订后的缺口，最多相差约三个百分点。

在实时情况下，2021 年的繁荣看起来像是从一场深度衰退中缓慢复苏。

我们现在为这个虚拟的美联储提供 2019 年以来的实时缺口，重新估计其信念，并在保持 IS 曲线不变的情况下重新计算其建议利率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Recommended rates with the revised and the real-time output gap
    name: fig-lc-realtime-rate
---
data_rt = data.copy()
data_rt.loc[x_real_time.index, 'x'] = x_real_time.values
beliefs_rt = estimate_beliefs(data_rt)
optimal_rt = recommended_rate(data_rt, phelps_rules(beliefs_rt))

fig, ax = plt.subplots(figsize=(10, 4.5))
w = slice('2019', None)
ax.plot(optimal[w], 'C0', label="Phelps rule, today's data")
ax.plot(optimal_rt[w], 'C1--', label='Phelps rule, real-time gap')
ax.plot(data['i'][w], 'C3', lw=1, label='actual funds rate')
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.legend()
plt.show()

table = pd.DataFrame({"today's data": optimal, 'real time': optimal_rt,
                      'actual': data['i']}).loc['2021':'2023-06']
print(table.rename(index=qtr).round(2).T.to_string())
print("\nmean rate in 2021:", table.loc['2021'].mean().round(2).to_dict())
for name in table:
    liftoff = table.index[table[name] > 0.5][0]
    print(f"first quarter above 0.5%, {name:>12}: {qtr(liftoff)}")
```

在使用美联储实时所见的缺口数据时，建议利率在整个 2021 年都保持在百分之零点五以下。

它在 2022 年之交开始上升，大致是美联储开始加息的时点。

在 2022 年上半年，它领先于实际利率，而从第三季度开始落后于实际利率，因为美联储当时以较大的步幅收紧政策。

这正是论文中所说的"略微滞后"模式。

相比之下，使用今日数据时，该规则的加息启动时间比美联储实际加息早了大约一年。

将各部分综合起来看：

* 反事实分析表明，**漂移的信念使响应变得温和**：若美联储仍持有 2000 年时的信念，它本会迅速且大幅加息；
* 这一比较表明，**实时数据测量误差使响应变得滞后**：若美联储持有漂移的信念但拥有准确的数据，它本会在 2021 年就开始加息。

这两个因素都是解释美联储实际行为所必需的。

## 信念如何塑造菲尔普斯规则

为什么感知持续性越低、感知斜率越平缓，规则就越不激进？

{cite:t}`SargentWilliams2025` 在其附录 A 中针对简化版的菲尔普斯问题给出了解析答案。

美联储认为：

```{math}
:label: lc_appA

\begin{aligned}
\pi_{t+1} &= \rho\, \pi_t + \kappa\, x_{t+1} + \varepsilon^{\pi}_{t+1}, \\
x_{t+1} &= b_1 x_t + g\,(i_t - \pi_t) + \varepsilon^{x}_{t+1},
\end{aligned}
```

其中 $|b_1| < 1$，$\kappa > 0$，且 $g < 0$。

这些是论文中的方程 (6) 和 (7)，其中的 $a_1$ 和 $k$ 对应于我们这里的 $\rho$ 和 $\kappa$。

美联储最小化 $\mathbb{E} \sum_t \beta^t (\pi_t^2 + \lambda_x x_t^2)$，没有平滑惩罚项、没有截距项，目标值为零。

将实际利率 $r_t = i_t - \pi_t$ 作为控制变量，就得到一个状态为 $(\pi_t, x_t)$ 的调节器问题，其黎卡提方程有闭式解。

最优规则为 $i_t = \phi_\pi(\rho, \kappa)\, \pi_t + \phi_x x_t$，其中

$$
\phi_\pi(\rho, \kappa) = 1 - \frac{\rho}{g\kappa}\,\frac{Z}{Z + \lambda_x},
\qquad
\phi_x = -\frac{b_1}{g},
$$

这里

$$
Z = \frac{-\Delta + \sqrt{\Delta^2 + 4\lambda_x\kappa^2}}{2},
\qquad
\Delta = \lambda_x(1 - \beta\rho^2) - \kappa^2 .
$$

该论文证明了两个比较静态结果。

```{prf:proposition} 感知持续性越低 ⇒ 政策越不激进
:label: lc_prop1

对任意 $\beta \in (0,1)$、$\lambda_x > 0$、$\kappa > 0$ 以及 $g < 0$，通胀系数 $\phi_\pi(\rho, \kappa)$ 关于 $\rho \geq 0$ 严格递增。
```

```{prf:proposition} 感知斜率越平缓 ⇒ 政策越不激进
:label: lc_prop2

只要某个充分条件（论文中的不等式 (11)）成立，就有 $\partial \phi_\pi / \partial \kappa > 0$；该条件对小到中等大小的 $\kappa$ 均成立。
```

让我们用数值方法验证这两个命题，并顺带用 `solve_discrete_are` 检验闭式解。

对于 IS 信念，我们使用示例数值 $b_1 = 0.9$ 和 $g = -0.5$。

$b_1$ 接近上面的估计值，但 $g$ 的绝对值要大得多，这样 $\phi_\pi$ 就处于一个熟悉的量级范围内。

由于 $\phi_\pi - 1$ 与 $1/g$ 成比例，$g$ 的取值不会影响比较静态结果的符号。

```{code-cell} ipython3
def φ_π(ρ, κ, b1=0.9, g=-0.5, λ_x=0.2, β=0.95):
    "Closed-form inflation coefficient of the simplified Phelps rule."
    Δ = λ_x * (1 - β * ρ**2) - κ**2
    Z = (-Δ + np.sqrt(Δ**2 + 4 * λ_x * κ**2)) / 2
    return 1 - (ρ / (g * κ)) * Z / (Z + λ_x)

def φ_π_riccati(ρ, κ, b1=0.9, g=-0.5, λ_x=0.2, β=0.95, η_r=1e-9):
    "The same coefficient from solve_discrete_are, with control cost η_r r²."
    A = np.array([[ρ, κ * b1], [0, b1]])
    B = np.array([[κ * g], [g]])
    Q, R = np.diag([1, λ_x]), np.array([[η_r]])
    P = solve_discrete_are(np.sqrt(β) * A, np.sqrt(β) * B, Q, R)
    F = np.linalg.solve(R + β * B.T @ P @ B, β * B.T @ P @ A)
    return 1 - F[0, 0]

grid = [(ρ_, κ_) for ρ_ in np.linspace(0.05, 0.99, 12)
        for κ_ in np.linspace(0.01, 1.0, 12)]
gap_cf = max(abs(φ_π(*p) - φ_π_riccati(*p)) / φ_π(*p) for p in grid)
print(f"largest relative difference, closed form vs Riccati: {gap_cf:.1e}")
```

闭式解通过了检验。

现在来看两个比较静态结果。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The inflation coefficient of the simplified Phelps rule
    name: fig-lc-appendix-a
---
ρ_grid = np.linspace(0, 0.999, 200)
κ_grid = np.linspace(0.005, 1.0, 400)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for κ_ in [0.04, 0.1, 0.3]:
    axes[0].plot(ρ_grid, φ_π(ρ_grid, κ_), label=rf'$\kappa = {κ_}$')
axes[0].set_xlabel(r'perceived persistence $\rho$')
axes[0].set_ylabel(r'$\phi_\pi$')
axes[0].legend()

for ρ_ in [0.5, 0.86, 0.99]:
    axes[1].plot(κ_grid, φ_π(ρ_, κ_grid), label=rf'$\rho = {ρ_}$')
axes[1].set_xlabel(r'perceived slope $\kappa$')
axes[1].legend()
plt.tight_layout()
plt.show()

increasing = all(np.all(np.diff(φ_π(ρ_grid, κ_)) > 0)
                 for κ_ in [0.01, 0.04, 0.1, 0.3, 1.0, 3.0])
print("ϕ_π increasing in ρ for every κ tried:", increasing)
for ρ_ in [0.5, 0.86, 0.99]:
    κ_peak = κ_grid[np.argmax(φ_π(ρ_, κ_grid))]
    print(f"ρ = {ρ_}: ϕ_π increases in κ up to κ = {κ_peak:.3f}, then falls")
print(f"largest κ_t estimated in 2010-2019: {κ['2010':'2019'].max():.3f}")
```

持续性在任何情况下都至关重要。

$\rho$ 越高，今天的通胀对未来通胀的预示作用就越强，美联储对其施加的抑制力度也就越大。

斜率同样重要：曲线越平缓，每一单位的反通胀所付出的产出代价就越高，因此美联储的反应就越弱。

这一点在斜率较小时成立，包括美联储在 2010 年代估计出的所有斜率值。

而在斜率较大时效应会反转，并且随着 $\rho$ 升高，峰值会移向更小的 $\kappa$（{ref}`lc_ex2`）。

因此 {prf:ref}`lc_prop2` 应被理解为关于小斜率情形的论断。

这两个命题合在一起，解释了 {numref}`fig-lc-rule-coefficients` 中规则漂移的原因：随着 $\rho_t$ 和 $\kappa_t$ 的下降，规则对通胀的反应也随之减弱。

(lc_feedback)=
## 从政策到测度信念

到目前为止，我们是从数据中取得美联储的信念的。

这些信念从何而来？

在他们项目的扩展摘要中，{cite:t}`SargentWilliamsConquestLost` 提出，"通货膨胀低持续性和菲利普斯曲线平坦的认知，是基于美联储过去的政策——该政策曾对冲击做出强烈反应并稳定了通货膨胀"。

他们将这一结果定性为一种自我确认均衡，在这种均衡中，美联储陷入了缺乏试验的陷阱。

**本节是本讲座对该思想的自行拓展；该论文2026年1月版本中并不包含这部分内容。**

我们将一个小型新凯恩斯模型作为*自然*的模型。

我们计算在不同政策下美联储的回归会测度出什么，然后用上一节的菲尔普斯规则来闭合这个循环。

用 {cite:t}`Koopmans1947` 的术语来说，美联储的漂移系数模型是一个描述性的"开普勒阶段"模型。

相比之下，自然的模型是一个"牛顿阶段"模型，其参数不受美联储规则的影响。

### 自然的模型

假设自然运行如下系统：

```{math}
:label: lc_nk

\begin{aligned}
\pi_t &= \beta_f\, \mathbb{E}_t \pi_{t+1} + \omega_b\, \pi_{t-1} + \kappa\, x_t + u_t, \\
x_t &= \mathbb{E}_t x_{t+1} - \sigma\left(i_t - \mathbb{E}_t \pi_{t+1} - r_t\right), \\
i_t &= \phi\, \pi_t ,
\end{aligned}
\qquad
\beta_f = \frac{\beta_n}{1 + \beta_n\omega}, \quad \omega_b = \frac{\omega}{1 + \beta_n\omega},
```

其中 $u_t$ 为成本推动冲击，$r_t$ 为自然利率（需求）冲击。

这两个冲击均为独立同分布，且互不相关，标准差分别为 $\sigma_u$ 和 $\sigma_r$。

第一个方程是混合菲利普斯曲线，当无法重新调整价格的企业以 $\omega \in [0, 1]$ 的比率根据滞后通货膨胀进行指数化时，便会出现这种曲线。

其对预期通货膨胀与滞后通货膨胀的权重之和为 $(\beta_n + \omega)/(1 + \beta_n\omega) \leq 1$。

在此归一化下，泰勒原则 $\phi > 1$ 给出一个确定性的均衡，下面的代码通过计算根的个数来检验这一点（{doc}`re_with_feedback`；{cite}`Blanchard_Khan`）。

自然的贴现因子 $\beta_n = 0.99$ 与美联储的贴现因子 $\beta = 0.95$ 不同。

我们设定 $\omega = 1$，结构斜率 $\kappa = 0.1$，以及 $\sigma = 1$。

在最小状态变量均衡中，$\mathbb{E}_t \pi_{t+1} = \lambda\, \pi_t$，并且

$$
\pi_t = \lambda\, \pi_{t-1} + a\, u_t + b\, r_t,
\qquad
x_t = \psi\, \pi_{t-1} + c\, u_t + d\, r_t .
$$

这里 $\lambda$ 是三次方程的稳定根：

$$
\beta_f \lambda^3 - (1 + \beta_f + \kappa\sigma)\lambda^2 + (1 + \omega_b + \kappa\sigma\phi)\lambda - \omega_b = 0,
$$

$\psi = -\sigma\lambda(\phi - \lambda)/(1 - \lambda)$，而冲击系数 $(a, b, c, d)$ 在代码中计算得出。

$\lambda$ 是单变量 AR(1) 过程的系数：这正是一位将通货膨胀对其自身滞后项进行回归的计量经济学家所称的持续性。

### 美联储测度了什么

美联储的回归式 {eq}`lc_pc` 中也包含 $x_t$。

由于 $x_t = \psi\, \pi_{t-1} + v_t$，其中 $v_t = c\, u_t + d\, r_t$ 与 $\pi_{t-1}$ 不相关，其总体系数为

$$
\hat\kappa = \frac{\operatorname{cov}(\pi_t, v_t)}{\mathbb{V}[v_t]},
\qquad
\hat\rho = \lambda - \hat\kappa\, \psi .
$$

因此美联储测度的持续性 $\hat\rho$ *并不是* $\lambda$。

当只有需求冲击发生时（$\sigma_u = 0$），我们可以将 $\mathbb{E}_t\pi_{t+1} = \lambda\pi_t$ 代入菲利普斯曲线。

这给出了精确关系 $\pi_t = [\omega_b\, \pi_{t-1} + \kappa\, x_t]/(1 - \beta_f\lambda)$，于是

$$
\hat\rho = \frac{\omega_b}{1 - \beta_f\, \lambda(\phi)},
\qquad
\hat\kappa = \frac{\kappa}{1 - \beta_f\, \lambda(\phi)} .
$$

美联储恢复了结构滞后权重和斜率，二者都除以 $1 - \beta_f\lambda(\phi)$，这是一个依赖于政策的因子。

这一行等式正是卢卡斯批判的体现。

更加积极的规则会降低 $\lambda$，同时也会降低 $\hat\rho$ 和 $\hat\kappa$：积极的政策使通货膨胀看起来持续性更低，菲利普斯曲线看起来更平坦。

持续性方面的结果与 {cite:t}`BenatiSurico2008` 和 {cite:t}`CarlstromFuerstPaustian2009` 的思路一致；斜率方面以此形式出现的结果来自 {cite:t}`Bullard2018`。

成本推动冲击会进一步压低 $\hat\kappa$，甚至使其变为负值。

这正是 {cite:t}`McLeayTenreyro2019` 所强调的识别问题：当美联储抵消成本推动冲击时，高通货膨胀会与低产出缺口同时出现。

```{code-cell} ipython3
def nk_weights(ω, β_n=0.99):
    "Weights (β_f, ω_b) of the normalized hybrid Phillips curve."
    return β_n / (1 + β_n * ω), ω / (1 + β_n * ω)

def nk_roots(φ, ω=1.0, κ=0.1, σ=1.0, β_n=0.99):
    "Roots of the cubic whose stable root is λ."
    β_f, ω_b = nk_weights(ω, β_n)
    return np.roots([β_f, -(1 + β_f + κ * σ), 1 + ω_b + κ * σ * φ, -ω_b])

def fed_regression(φ, ω=1.0, κ=0.1, σ=1.0, β_n=0.99, σ_u=0.0, σ_r=1.0):
    "λ, and the Fed's population estimates (ρ̂, κ̂), when nature's rule is φ."
    β_f, ω_b = nk_weights(ω, β_n)
    roots = nk_roots(φ, ω, κ, σ, β_n)
    λ = roots[np.abs(roots) < 1].real.item()        # determinacy: exactly one
    m = -σ * (φ - λ) / (1 - λ)                       # x_t = m π_t + σ r_t
    ψ = m * λ
    den = 1 - β_f * λ - κ * m
    a, b = 1 / den, κ * σ / den                      # π_t on (u_t, r_t)
    c, d = m * a, m * b + σ                          # x_t on (u_t, r_t)
    κ_hat = ((a * c * σ_u**2 + b * d * σ_r**2)
             / (c**2 * σ_u**2 + d**2 * σ_r**2))
    return λ, λ - κ_hat * ψ, κ_hat

φ_grid = np.linspace(1.01, 5.0, 200)
determinate = all(np.sum(np.abs(nk_roots(φ, ω)) < 1) == 1
                  for φ in np.r_[φ_grid, 10.0, 100.0] for ω in [0.25, 1.0])
print("one stable root for every φ > 1 tried:", determinate)
```

下面是 $\lambda$、$\hat\rho$ 和 $\hat\kappa$ 作为自然的泰勒规则系数 $\phi$ 的函数的图像。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: What the Fed's regression measures under different policy rules
    name: fig-lc-measured-beliefs
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
λ_path = [fed_regression(φ)[0] for φ in φ_grid]
axes[0].plot(φ_grid, λ_path, 'k:', label=r'$\lambda$ (AR(1) root)')
for σ_u in [0.0, 0.1]:
    est = np.array([fed_regression(φ, σ_u=σ_u)[1:] for φ in φ_grid])
    axes[0].plot(φ_grid, est[:, 0], label=rf'$\hat\rho$, $\sigma_u/\sigma_r = {σ_u}$')
    axes[1].plot(φ_grid, est[:, 1], label=rf'$\sigma_u/\sigma_r = {σ_u}$')
est = np.array([fed_regression(φ, σ_u=0.3)[1:] for φ in φ_grid])
axes[1].plot(φ_grid, est[:, 1], label=r'$\sigma_u/\sigma_r = 0.3$')
axes[1].axhline(0, color='k', lw=0.5, ls=':')
axes[0].set_xlabel(r"nature's Taylor-rule coefficient $\phi$")
axes[0].set_ylabel('measured persistence')
axes[1].set_xlabel(r"nature's Taylor-rule coefficient $\phi$")
axes[1].set_ylabel(r'measured slope $\hat\kappa$')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()
```

测度的持续性 $\hat\rho$ 明显高于 AR(1) 的根 $\lambda$，并且随着政策越来越积极，三者都会下降。

仅存在需求冲击时，$\hat\kappa$ 也随 $\phi$ 下降，但仍高于结构斜率 $\kappa = 0.1$，正如公式所示。

即使是适度的成本推动冲击，也会大幅将其拉低，甚至降至零以下。

该模型是一个简化的季度模型，因此其数值并不旨在与美联储基于四季度通货膨胀的估计相匹配。

### 自我确认的不动点

现在，将这两个映射复合起来。

一段政策历史 $\phi$ 会产生测度的信念 $(\hat\rho(\phi), \hat\kappa(\phi))$。

将这些信念代入菲尔普斯问题，便得到了建议的通货膨胀系数

$$
T(\phi) = \phi_\pi\big(\hat\rho(\phi), \hat\kappa(\phi)\big).
$$

不动点 $\phi^* = T(\phi^*)$ 是 {doc}`phillips_self_confirming` 意义上的自我确认均衡。

在 $\phi^*$ 处，美联储的规则在其信念给定的条件下求解其菲尔普斯问题，而其信念又满足其规则所产生的数据中最小二乘正交性条件。

我们做了两个简化。

自然的规则仅使用菲尔普斯问题所建议的通货膨胀系数。

我们也将美联储关于IS的信念固定在上一节的示例值，这样该循环仅通过菲利普斯曲线运行。

由于 $\hat\rho$ 和 $\hat\kappa$ 随 $\phi$ 上升而下降，而 $\phi_\pi$ 则随二者上升而上升（对于较小的 $\kappa$），因此 $T$ 是*递减的*：该循环是一种负反馈。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The map from past policy to the policy the Phelps problem recommends
    name: fig-lc-t-map
---
def T(φ, **nature):
    "Inflation coefficient recommended by beliefs measured under the rule φ."
    _, ρ_hat, κ_hat = fed_regression(φ, **nature)
    return φ_π(ρ_hat, κ_hat)

def fixed_point(**nature):
    "φ* = T(φ*) and the slope T′(φ*)."
    φ_star = brentq(lambda φ: T(φ, **nature) - φ, 1.01, 5.0)
    h = 1e-5
    slope = (T(φ_star + h, **nature) - T(φ_star - h, **nature)) / (2 * h)
    return φ_star, slope

fig, ax = plt.subplots(figsize=(7, 4.5))
for σ_u in [0.0, 0.1]:
    ax.plot(φ_grid, [T(φ, σ_u=σ_u) for φ in φ_grid],
            label=rf'$T(\phi)$, $\sigma_u/\sigma_r = {σ_u}$')
ax.plot(φ_grid, φ_grid, 'k:', lw=1, label='45° line')
ax.set_xlabel(r'past policy $\phi$')
ax.set_ylabel(r'recommended $\phi_\pi$')
ax.legend()
plt.show()

cases = {'baseline': {}, 'σ_u/σ_r = 0.1': {'σ_u': 0.1},
         'ω = 0.5': {'ω': 0.5}, 'κ = 0.05': {'κ': 0.05}}
for name, nature in cases.items():
    φ_star, slope = fixed_point(**nature)
    _, ρ_hat, κ_hat = fed_regression(φ_star, **nature)
    print(f"{name:>15}: φ* = {φ_star:.3f}, T′(φ*) = {slope:.3f}, "
          f"beliefs (ρ̂, κ̂) = ({ρ_hat:.3f}, {κ_hat:.3f})")
```

$T$ 与45度线只相交一次，因此自我确认均衡是唯一的。

那么它在学习过程中是否稳定呢？

以较小增益进行学习的美联储会遵循如 $d\phi/d\tau = T(\phi) - \phi$ 之类的均值动态——这正是 {ref}`br_one_recursion` 中的常微分方程。

若 $T'(\phi^*) < 1$，则该静止点是E-稳定的。

在信念空间中也会出现同样的条件。

映射 $(\rho, \kappa) \mapsto (\hat\rho, \hat\kappa)(\phi_\pi(\rho, \kappa))$ 具有秩一的雅可比矩阵，其非零特征值为 $T'(\phi^*)$。

因此E-稳定性的特征值为 $-1$ 和 $T'(\phi^*) - 1$，这正是 {doc}`phillips_self_confirming` 对《征服》一书的模型所做的那类计算（{ref}`sc_estability`）。

由于 $T' < 0$，在上述每种情况下该条件都绰绰有余地成立。

由于 $|T'| < 1$ 也成立，甚至增益为一的"蛛网"迭代 $\phi_{k+1} = T(\phi_k)$ 也会收敛。

它描述的是这样一位美联储：只有在其信念完全适应了每一个政策体制之后，才会重新优化：

```{code-cell} ipython3
for name in ['baseline', 'σ_u/σ_r = 0.1']:
    φ_k, path = 3.0, [3.0]                     # start from an aggressive history
    for _ in range(8):
        φ_k = T(φ_k, **cases[name])
        path.append(φ_k)
    print(f"{name:>15}:", np.round(path, 2))
```

积极的历史之后紧跟着温和的规则，随后是部分修正，围绕 $\phi^*$ 呈阻尼振荡。

这是对"征服的失去与重获"的微弱回响，但它最终会消失。

### 没有逃脱路径的陷阱

该不动点具有与《征服》一书中自我确认均衡相同的结构。

预期效用型的美联储从构造上就从不进行试验（{doc}`phillips_learning`）。

在 $\phi^*$ 处，其信念在均衡路径上是正确的，而在路径之外则是错误的：在一个不那么积极的规则下，测度的持续性和斜率都会上升。

这正是 {doc}`phillips_escaping_nash` 中所述的缺乏试验陷阱（{ref}`en_trap`），也正是扩展摘要所框定的那种含义。

差别在于动态机制。

在 {cite:t}`ChoWilliamsSargent2002` 的模型中，逃脱路径是自我强化的。

一连串使感知的菲利普斯曲线变陡的冲击，会促使政府削减通货膨胀，而这又会产生进一步使曲线变陡的数据。

而在这里，反馈的方向恰恰相反。

一连串提高测度持续性的冲击，会使美联储采取更积极的政策，而更积极的政策又会*降低*测度的持续性。

一个恒定增益的学习者会在 $\phi^*$ 附近徘徊，但均值动态会将其从任一侧拉回：这里不存在逃脱路径。

还有两个事实限制了这一循环对数据的解释力。

首先，在 {numref}`fig-lc-beliefs` 中，$\rho_t$ 的下降发生在2008年之后，当时零利率下限使得政策根本无法变得积极。

其次，2022年 $\rho_t$ 的跳升，与通货膨胀峰值同时出现，这与 {doc}`phillips_drifts_volatilities` 所记录的通货膨胀水平与持续性之间的正相关关系相一致。

逃脱路径模型预测的是*负*相关；{cite:t}`CogleySargentConquest2005`（脚注28）指出，上述那种渐进主义可以产生正相关关系。

因此，2020年代并非《征服》一书中反复出现的那种"逃脱"的重演。

延续下来的是《征服》一书的诊断方式。

美联储将一个系数依赖于政策的简化形式当作结构性模型来对待，恰如 {doc}`phillips_two_stories` 所描述的"平反故事"那样忽视了卢卡斯批判（{cite:t}`SargentWilliams2025`，脚注6）。

但如今的误设问题涉及的是菲利普斯曲线的斜率与持续性，而非自然率假说。

```{note}
{cite:t}`SargentWilliams2025` 也承认了一种截然不同的解读，即2020年代的宽松政策源于财政因素，正如他们所引用的财政理论所述。

这将是对同一政策路径的另一种合理化解释。
```

## 征服是否得而复失？

{doc}`phillips_two_stories` 以追问 1999 年以来的数据对其两个故事意味着什么作为结尾。

一个可信的低通胀均衡是否会在每次冲击后都重新锚定，正如自然率理论的胜利所暗示的那样？

还是说，意外事件仍可能使货币当局的信念发生漂移，正如计量经济学政策评估的证实所暗示的那样？

遵循 {cite:t}`SargentWilliams2025`，本讲的答案是：征服一度得而复失。

原因并非美联储目标的改变。

* 这个人造美联储的损失函数从未改变。
* 在一段长期低而稳定的通胀期间，它的信念发生了漂移，以至于到 2019 年，它的费尔普斯问题所给出的长期通胀响应系数只略高于一；这使得它对通胀激增的反应变得*迟缓*。
* 实时衡量的闲置产能指标高估了经济所拥有的余地，这使得它反应*滞后*。
* 一旦 2022 年的数据到来，感知到的持续性迅速跳回接近一，规则对通胀的响应也随之跳升，征服随即重新获得。

用 {doc}`phillips_two_stories` 的语言来说，这段经历看起来像是证实故事中那种适应性政策制定者，被一个系数依赖于政策本身的简化形式方程所误导。

但这并非该故事中反复出现的逃逸现象的重演。

我们构建的政策—信念循环并无逃逸路径：漂移的信念解释了反应为何迟缓，而数据测量错误则解释了反应为何滞后。

本讲内容也丝毫不与 {doc}`phillips_two_stories` 中记录的胜利故事的解读相矛盾，即长期通胀预期始终保持锚定：这个人造美联储的信念涉及的是通胀的简化形式动态，而非公众的预期。

这段经历为《征服》增添的教训，正是扩展摘要所指出的那一点：过去成功政策的经验教训，可能会在长期平静时期逐渐淡忘，而后又不得不重新习得。

## 练习

```{exercise-start}
:label: lc_ex1
```

在损失函数 {eq}`lc_loss` 中的利率平滑权重 $\eta$，用原论文的话来说，是“最重要的参数之一”。

论文指出，更大的 $\eta$ 能更好地拟合观测到的利率，但会使该规则更难被解释为系统性的稳定政策。

让我们将这一权衡量化。

对于 $\eta \in \{0.1, 0.5, 2.0\}$：

1. 计算建议利率及其相对于 1991 年以来实际联邦基金利率的均方根误差（RMSE），并与无变动规则 $i_t = i_{t-1}$ 进行比较；
1. 计算 2019 年第四季度的规则及其对通货膨胀的长期反应 $\phi_\pi/(1 - \phi_i)$；
1. 将建议利率与 2015 年以来的实际利率进行对比绘图。

你会选择哪个 $\eta$？为什么？

```{exercise-end}
```

```{solution-start} lc_ex1
:class: dropdown
```

信念不依赖于 $\eta$，因此我们可以重复使用它们。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 4.5))
w = slice('2015', None)
print(f"{'η':>5} {'RMSE':>6} {'ϕ_π':>7} {'ϕ_i':>7} {'长期反应':>9}")
for η_val in [0.1, 0.5, 2.0]:
    rules_η = phelps_rules(beliefs, η=η_val)
    rate = recommended_rate(data, rules_η)
    f = rules_η.loc['2019-10-01']
    print(f"{η_val:>5} {fit(rate)[1]:>6.2f} {f.φ_π:>7.3f} {f.φ_i:>7.3f} "
          f"{f.φ_π / (1 - f.φ_i):>9.2f}")
    ax.plot(rate[w], lw=1, label=rf'$\eta = {η_val}$')
print(f"无变动规则的 RMSE：{fit(no_change)[1]:.2f}")
ax.plot(data['i'][w], 'k:', lw=1.5, label='实际利率')
ax.set_xlabel('年份')
ax.set_ylabel('百分比')
ax.legend()
plt.show()
```

较小的 $\eta$ 会产生随着通货膨胀和产出缺口剧烈波动的利率，拟合效果很差。

提高 $\eta$ 会大幅改善拟合效果，但在 $\eta = 2$ 时，RMSE 已经接近无变动规则的水平。

与此同时，2019 年规则对通货膨胀的长期反应降到了 1 以下，这意味着该规则不再满足泰勒原则。

较大的 $\eta$ 是通过把菲尔普斯规则变成无变动规则来换取拟合优度的。

这正是论文所说“可解释性”的具体含义。

$\eta = 0.5$ 既能保持大于 1 的长期反应，又比 $\eta = 0.1$ 拟合得好得多。

```{solution-end}
```

```{exercise-start}
:label: lc_ex2
```

本练习探讨简化的菲尔普斯问题 {eq}`lc_appA`。

1. 证明 $P_{12} = 0$ 且 $P_{22} = \lambda_x$ 满足其里卡蒂方程。（提示：$A$ 的第二列是 $B$ 的倍数。）然后推导出 $P_{11}$ 所满足的二次方程，并用 `solve_discrete_are` 验证这两个结果。
1. 对于 $\rho \in \{0.5, 0.9\}$，绘出 $\phi_\pi$ 相对 $\kappa \in (0, 3]$ 的曲线，并找出 $\phi_\pi$ 达到峰值时的斜率。解释为什么 $\phi_\pi$ 最终会随 $\kappa$ 增大而下降。
1. 在损失函数中加入对实际利率的惩罚项 $\eta_r r_t^2$。对于 $\eta_r \in \{0.01, 0.1\}$，这一反转现象是否依然成立？

```{exercise-end}
```

```{solution-start} lc_ex2
:class: dropdown
```

**第 1 部分。** 以 $r_t$ 作为控制变量，

$$
A = \begin{bmatrix} \rho & \kappa b_1 \\ 0 & b_1 \end{bmatrix},
\qquad
B = g\begin{bmatrix} \kappa \\ 1 \end{bmatrix},
$$

因此 $A$ 的第二列等于 $(b_1/g) B$。

在没有控制成本的情况下，里卡蒂方程为

$$
P = Q + \beta A^\top\big(P - PB(B^\top P B)^{-1}B^\top P\big)A .
$$

括号中的矩阵会消去 $B$，因此 $A^\top(\cdot)A$ 的第二行和第二列为零。

于是 $P$ 在第二行和第二列上与 $Q = \operatorname{diag}(1, \lambda_x)$ 相同：$P_{12} = 0$ 且 $P_{22} = \lambda_x$。

从经济学角度看，美联储可以完全抵消下一季度产出缺口中可预测的部分，因此当期产出缺口不会将任何成本带入未来。

利用 $B^\top P B = g^2(\kappa^2 P_{11} + \lambda_x)$ 以及 $(PB)_1 = g\kappa P_{11}$，$(1,1)$ 元素给出

$$
P_{11} = 1 + \beta\rho^2\,\frac{\lambda_x P_{11}}{\kappa^2 P_{11} + \lambda_x},
\quad\text{即}\quad
\kappa^2 P_{11}^2 + \big(\lambda_x(1 - \beta\rho^2) - \kappa^2\big) P_{11} - \lambda_x = 0,
$$

其正根即为论文中的公式(8)；$Z = \kappa^2 P_{11}$。

```{code-cell} ipython3
def riccati_P(ρ, κ, b1=0.9, g=-0.5, λ_x=0.2, β=0.95, η_r=1e-9):
    A = np.array([[ρ, κ * b1], [0, b1]])
    B = np.array([[κ * g], [g]])
    return solve_discrete_are(np.sqrt(β) * A, np.sqrt(β) * B,
                              np.diag([1, λ_x]), np.array([[η_r]]))

for ρ_, κ_ in [(0.9, 0.05), (0.5, 0.2), (0.95, 1.0)]:
    P = riccati_P(ρ_, κ_)
    Δ = 0.2 * (1 - 0.95 * ρ_**2) - κ_**2
    P11 = (-Δ + np.sqrt(Δ**2 + 4 * 0.2 * κ_**2)) / (2 * κ_**2)
    print(f"ρ = {ρ_}, κ = {κ_}: P12 = {P[0, 1]:.1e}, P22 = {P[1, 1]:.4f}, "
          f"P11 = {P[0, 0]:.4f}（闭式解 {P11:.4f}）")
```

**第 2 和第 3 部分。**

```{code-cell} ipython3
κ_wide = np.linspace(0.005, 3.0, 600)
fig, ax = plt.subplots(figsize=(8, 4.5))
for ρ_ in [0.5, 0.9]:
    curve = φ_π(ρ_, κ_wide)
    ax.plot(κ_wide, curve, label=rf'$\rho = {ρ_}$, $\eta_r = 0$')
    print(f"ρ = {ρ_}, η_r = 0：峰值出现在 κ = {κ_wide[np.argmax(curve)]:.3f}")
for η_r in [0.01, 0.1]:
    curve = [φ_π_riccati(0.9, κ_, η_r=η_r) for κ_ in κ_wide]
    ax.plot(κ_wide, curve, '--', label=rf'$\rho = 0.9$, $\eta_r = {η_r}$')
    print(f"ρ = 0.9, η_r = {η_r}：峰值出现在 κ = {κ_wide[np.argmax(curve)]:.3f}，"
          f"在 κ 上单调：{np.all(np.diff(curve) > 0)}")
ax.set_xlabel(r'感知斜率 $\kappa$')
ax.set_ylabel(r'$\phi_\pi$')
ax.legend()
plt.show()
```

$\phi_\pi$ 相对 $\kappa$ 呈驼峰形。

当曲线非常平坦时，通过产出缺口来调节通货膨胀的代价很高，因此美联储几乎不作为：$Z/(Z + \lambda_x)$ 很小。

当曲线很陡峭时，产出缺口的微小变化就能大幅影响通货膨胀，因此只需微小的利率反应即可：$\phi_\pi \approx 1 + \rho/(|g|\kappa)$ 趋近于 1。

峰值位于两者之间，当 $\rho$ 更大时，峰值出现在更小的 $\kappa$ 处。

论文脚注 19 提出，较小的控制成本会使 $\partial\phi_\pi/\partial\kappa > 0$ 对所有 $\kappa$ 均成立。

但数值结果表明，即使施加对实际利率的惩罚，驼峰形依然存在，因此 {prf:ref}`lc_prop2` 最好被理解为一个关于小斜率情形的论断。

美联储在 2010 年代估计出的斜率恰好落在这一区域内。

```{solution-end}
```

```{exercise-start}
:label: lc_ex3
```

基准设定在 2020 年和 2021 年都将增益设为零。

请重新计算信念和建议利率，分别考虑(a)仅冻结 2020 年和(b)完全不冻结两种情形。

报告 2021-2022 年的 $\rho_t$、$\kappa_t$ 以及建议利率。

冻结的哪一部分决定了这一结果？在 2021 年将增益设为零意味着什么假设？

```{exercise-end}
```

```{solution-start} lc_ex3
:class: dropdown
```

我们对大流行季度的三种处理方式依次进行循环计算。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 4.5))
w = slice('2019', '2023')
for freeze, style in [((2020, 2021), '-'), ((2020,), '--'), ((), ':')]:
    b = estimate_beliefs(data, freeze=freeze)
    rate = recommended_rate(data, phelps_rules(b))
    label = f"冻结 {freeze}" if freeze else '不冻结'
    print(f"\n{label}")
    print(pd.DataFrame({'ρ': b['rho'], 'κ': b['kappa'], 'rate': rate})
            .loc['2021':'2022'].rename(index=qtr).round(3).T.to_string())
    ax.plot(rate[w], style, label=label)
ax.plot(data['i'][w], 'k', lw=1, alpha=0.6, label='实际利率')
ax.set_xlabel('年份')
ax.set_ylabel('百分比')
ax.legend()
plt.show()
```

仅冻结 2020 年得到的结果与完全不冻结几乎相同：极端的 2020 年数据并不是关键所在。

真正起决定作用的是 2021 年。

一个从 2021 年数据中学习的美联储会看到通货膨胀随着正的产出缺口而上升，于是其感知斜率 $\kappa_t$ 会急剧变陡，其感知持续性也会随之上升。

到 2021 年底，其建议利率将达到百分之几的水平。

因此，基准设定中 2021 年零增益的做法内置了这样一种判断：2021 年的数据对通货膨胀动态并无信息价值——而这正是“暂时性”观点本身。

这一冻结是一种建模选择，它帮助模型为一个行动迟缓的美联储提供合理化解释，而非对异常值的中性处理。

```{solution-end}
```

```{exercise-start}
:label: lc_ex4
```

{cite:t}`Chung1990` 将政府学习算法的增益视为一个待估计的自由参数（{ref}`olg_ex7`）。

{doc}`phillips_policy_evaluation` 在 1960-1995 年数据上延续了他的问题。

请为这个虚拟美联储提出钟（Chung）式的问题。

对于 $0.01$ 到 $0.05$ 之间的增益 $\gamma$，重新计算信念和建议利率，以及相对于 1991 年以来实际利率的 RMSE。

哪个增益拟合效果最好？

在 $\gamma = 0.05$ 时，2022 年的信念和建议利率会发生什么？

```{exercise-end}
```

```{solution-start} lc_ex4
:class: dropdown
```

对于每个增益，我们记录 2022-2023 年间感知持续性的最大值以及建议利率的最大值。

```{code-cell} ipython3
gains = np.arange(0.01, 0.0501, 0.005)
rmse, extremes = [], {}
for γ_val in gains:
    b = estimate_beliefs(data, γ=γ_val)
    rate = recommended_rate(data, phelps_rules(b))
    rmse.append(fit(rate)[1])
    extremes[round(γ_val, 3)] = (b['rho']['2022':'2023'].max(),
                                 rate['2022':'2023'].max())

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(gains, rmse, marker='o')
ax.set_xlabel(r'增益 $\gamma$')
ax.set_ylabel('RMSE（1991年以来）')
plt.show()

print(f"网格中的最佳增益：{gains[np.argmin(rmse)]:.3f}")
for γ_val in [0.03, 0.05]:
    ρ_max, rate_max = extremes[γ_val]
    print(f"γ = {γ_val}：2022-23 年 ρ_t 的最大值 = {ρ_max:.3f}，"
          f"建议利率的最大值 = {rate_max:.1f}%")
```

最佳拟合接近论文中的 $\gamma = 0.03$。

较小的增益使信念过于迟缓，无法跟上变化趋势。

较大的增益则使信念追逐噪声。

在 $\gamma = 0.05$ 时，2022 年的意外冲击会将感知持续性推高至 1 以上：美联储的模型变得发散不稳，其菲尔普斯问题会要求实施极其剧烈的紧缩。

增益越大，仅仅几个季度的意外冲击就越能从根本上改变美联储的模型。

这一准则是对政策利率的最小二乘拟合，它将损失函数的参数视为已知，而非钟氏针对整个系统的似然方法。

```{solution-end}
```

```{exercise-start}
:label: lc_ex5
```

本练习将本讲座的持续性度量与 {doc}`phillips_drifts_volatilities`（{ref}`csdv_persistence`）中的度量进行调和。

1. 通过模拟证明，如果季度通货膨胀本身是序列不相关的，将四季度通货膨胀对其一季度滞后值进行回归，所得系数会接近 $0.75$。
1. 用季度年化通货膨胀 $400 \log(P_t/P_{t-1})$ 代替四季度通货膨胀重新运行 `estimate_beliefs`，并比较两条 $\rho_t$ 路径。
1. 在每种度量下，持续性分别在何时下降？

```{exercise-end}
```

```{solution-start} lc_ex5
:class: dropdown
```

我们先从重叠性入手，然后在季度数据上重新计算信念。

```{code-cell} ipython3
rng = np.random.default_rng(0)
quarterly = rng.standard_normal(100_000)
four_quarter = np.convolve(quarterly, np.ones(4) / 4, mode='valid')
coef = np.polyfit(four_quarter[:-1], four_quarter[1:], 1)[0]
print(f"季度通货膨胀为独立同分布情形下的回归系数：{coef:.3f}")

data_q = data.copy()
data_q['pi'] = 400 * np.log(pce / pce.shift(1))[data.index]
beliefs_q = estimate_beliefs(data_q)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(beliefs['rho'], label='四季度通货膨胀（基准）')
ax.plot(beliefs_q['rho'], label='季度年化通货膨胀')
ax.axhline(0.75, color='k', lw=0.5, ls=':')
ax.set_xlabel('年份')
ax.set_ylabel(r'感知持续性 $\rho_t$')
ax.legend()
plt.show()

dates = ['1980-01-01', '1985-01-01', '1995-01-01', '2005-01-01',
         '2015-01-01', '2019-10-01', '2022-04-01']
print(pd.DataFrame({'四季度': beliefs['rho'], '季度': beliefs_q['rho']})
        .loc[dates].rename(index=qtr).round(2).T.to_string())
```

四季度利率的重叠性，即便在季度通货膨胀本身毫无持续性的情况下，也会内置出约 $0.75$ 的系数，因此基准 $\rho_t$ 不可能大幅下降。

采用季度通货膨胀后，持续性在 1980 年之后出现了适度下降，到 2000 年代中期进一步下降，并在 2010 年代降至约 0.4。

它在 2022 年再次上升，但远低于 1 的水平。

因此，重叠性既抬高了基准度量的水平，也掩盖了其下降过程的早期阶段。

季度度量更接近 {doc}`phillips_drifts_volatilities` 的结论，后者发现持续性在 1980 年之后不久便开始下降。

两种度量在本讲座所依赖的主要事实上是一致的：到 2019 年，通货膨胀的持续性已远低于以往。

```{solution-end}
```
