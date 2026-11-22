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
  title: 经济动荡、收入不稳定性与间接推断
  headings:
    Overview: 概述
    Laissez-faire economy: 自由放任经济
    Laissez-faire economy::Workers: 工人
    Laissez-faire economy::Computation: 计算
    Laissez-faire economy::Artificial panels: 人工面板数据
    Gottschalk and Moffitt's decomposition: Gottschalk和Moffitt的分解方法
    Gottschalk and Moffitt's decomposition::Definitions: 定义
    Gottschalk and Moffitt's decomposition::Decomposition as information projection: 作为信息投影的分解方法
    Ljungqvist and Sargent's check: Ljungqvist和Sargent的检验
    Ljungqvist and Sargent's check::Histograms: 直方图
    Ljungqvist and Sargent's check::Binding function: 约束函数
    Layoff rates and turbulence: 裁员率与动荡性
    Covariance-structure model: 协方差结构模型
    Concluding remarks: 结束语
    Exercises: 练习
---

(turbulence_inference)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 经济动荡、收入不稳定性与间接推断

```{contents} Contents
:depth: 2
```

## 概述

{cite:t}`LjungqvistSargent1998` 和 {cite:t}`LjungqvistSargent2008` 将1980年后欧洲持续高企的失业率归因于失业救济金与过去收入挂钩，和他们所称的**经济动荡**（即工作因裁员而终止时技能丧失的风险）增加之间的相互作用。

为了检验动荡性的增加是否与美国收入的证据相符，{cite:t}`LjungqvistSargent1998` 在两个动荡水平下，利用其模型的自由放任版本，生成了人工收入面板数据。

他们对这些面板数据应用了 {cite:t}`GottschalkMoffitt1994` 曾对收入动态研究面板（PSID）1970—78年和1979—87年男性数据所采用的、将收入分解为永久性成分和暂时性成分的分解方法，并将所得直方图与Gottschalk和Moffitt的结果进行比较。

{cite:t}`LjungqvistSargent2008` 在一个更丰富的模型中重复了这一检验，两个窗口期的动荡参数分别为 $0.1$ 和 $0.2$。

{cite:t}`LjungqvistSargent1998` 得出结论："我们的动荡参数的增加使Gottschalk-Moffitt分解的两个成分的分布都朝观测到的方向扩散。"他们补充道：

> 同样值得注意的是，我们图14b中更动荡时期收入变异性的增加，发生在比Gottschalk和Moffitt的研究结果更低的标准差水平上。在这方面，我们对1980年代参数化所代表的经济动荡增加，未能达到美国所记录的变化幅度。

在本讲座中，我们将他们的检验描述为信息投影的比较，并针对 {cite:t}`LjungqvistSargent1998` 的模型将其量化。

* Gottschalk和Moffitt的分解方法，在 {doc}`information_projections` 的意义上，是将一个男性在平衡面板中九年年度对数收入的分布向高斯随机效应模型做m-投影的结果。
* Ljungqvist和Sargent的检验将数据中男性平均值和标准差的直方图与其模型在两个动荡参数值下的直方图进行比较；这些直方图的二阶矩就是投影的参数。在该参数的每个取值处计算模型的投影，就得到了 {cite:t}`GourierouxMonfortRenault1993` 所称的约束函数（binding function），而选择使两个投影一致的参数，就是 {doc}`gallant_tauchen` 所描述的间接推断方法。
* 将1970—78年的动荡参数固定在Ljungqvist和Sargent使用的 $0.02$ 时，无论1979—87年的参数取何值，自由放任模型都无法实现Gottschalk和Moffitt测得的永久性方差增量的五分之一以上，或暂时性方差增量的约45%以上。若两个参数都自由取值，这两个上界分别约为三分之一和二分之一。
* 这些上界取决于技能范围和技能转换的速度。仅将范围加倍，动荡性就能实现永久性方差增量的约46%，以及几乎全部的暂时性方差增量；若同时将转换概率加倍，则永久性份额又回落至约五分之一。
* 裁员率的上升会使Gottschalk和Moffitt的两个统计量朝与动荡性增加相同的方向移动，而在Gottschalk和Moffitt样本规模的数据中，这两个统计量几乎无法区分这两种解释。
* {cite:t}`MoffittGottschalk2011` 的协方差结构模型，作为第二个辅助模型，几乎将动荡性所造成的收入不稳定性全部归因于暂时性成分，而Gottschalk和Moffitt的分解方法则将其中约一半归因于永久性成分。

本讲座的结构如下。

* {ref}`ti_model_section` 描述 {cite:t}`LjungqvistSargent1998` 的自由放任经济，并重现其稳态统计量。
* {ref}`ti_gm_section` 将Gottschalk和Moffitt的分解方法描述为一种信息投影。
* {ref}`ti_check_section` 重现Ljungqvist和Sargent的检验并计算约束函数。
* {ref}`ti_ident_section` 将动荡性与更高的裁员率进行比较。
* {ref}`ti_mg_section` 描述第二个辅助模型。

本讲座建立在 {doc}`information_projections`、{doc}`gallant_tauchen` 和 {doc}`specification_tests` 的基础上。

{doc}`mccall_model` 和 {doc}`mccall_model_with_separation` 描述了更简单的搜寻模型。

我们使用以下导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.stats import norm, chi2, ncx2
from scipy.optimize import minimize
```

(ti_model_section)=
## 自由放任经济

### 工人

{cite:t}`LjungqvistSargent1998` 研究了连续统的风险中性工人。

一个时期为两周。

每个工人在各时期之间以概率 $\alpha$ 死亡，并最大化 $E_t \sum_{i=0}^\infty \beta^i (1 - \alpha)^i y_{t+i}$，其中 $y_t$ 为扣除搜寻负效用后的净收入。

新出生的工人以最低技能水平开始失业。

工人的技能 $h$ 取 $[1, 2]$ 上均匀分布的21个值之一。

受雇工人的收入为 $w h$，其中 $w$ 是该工作中单位技能的工资。

在每个时期开始时，受雇工人以概率 $\lambda$ 被裁员。

未被裁员的工人以概率 $0.1$ 提升一个技能水平，并保持工资 $w$ 直到被裁员、辞职或死亡。

失业工人的技能每期以概率 $0.2$ 下降一个水平，最低水平为吸收态。

失业工人以负效用 $c(s) = 0.5 s$ 选择搜寻强度 $s \in [0, 1]$，并以概率 $\pi(s) = s^{0.3}$ 在下一期收到一份工资报价。

工资报价来自均值为 $0.5$、方差为 $0.1$ 的正态分布，截断至 $[0, 1]$。

Ljungqvist和Sargent对动荡性的定义如下：

> 经济动荡性是根据与裁员相关的技能损失的均值和方差来定义的。我们现在让一个刚被裁员工人的技能分布服从一个正态分布的左半部分，其范围从可能的最低技能水平开始，到该工人裁员前的技能水平结束。

他们通过取以 $0.5$ 为中心、限制在 $[0, 1]$ 内的正态分布的左半部分，将其拉伸到从最低水平到工人裁员前所持有水平的技能水平范围内，并进行归一化，从而获得该分布。

该正态分布的方差度量了动荡程度；我们称之为**动荡参数**。

当动荡参数为零时，被裁员的工人保持裁员前所持有的技能水平。

设 $\mu_e(h, h')$、$\mu_u(h, h')$ 和 $\mu_l(h, h')$ 分别为受雇工人、失业工人以及刚被裁员工人的技能转移概率。

自由放任经济中没有失业救济金，也没有税收。

持有或被提供工资为 $w$ 的工作、技能为 $h$ 的工人的价值 $V(w, h)$，以及失业工人的价值 $V_u(h)$，满足

$$
V(w, h) = \max \left\{ w h + (1 - \alpha)\beta \left[ (1 - \lambda) \sum_{h'} \mu_e(h, h') V(w, h') + \lambda \sum_{h'} \mu_l(h, h') V_u(h') \right], \; V_u(h) \right\}
$$

以及

$$
V_u(h) = \max_{s \in [0, 1]} \left\{ -c(s) + (1 - \alpha)\beta \sum_{h'} \mu_u(h, h') \left[ (1 - \pi(s)) V_u(h') + \pi(s) \int V(w, h') \, dF(w) \right] \right\} .
$$

$V(w, h)$ 中最大化的第二个参数，代表拒绝报价或辞职的选项。

设 $G(h) = (1 - \alpha)\beta \sum_{h'} \mu_u(h, h') [\int V(w, h') \, dF(w) - V_u(h')]$ 为报价的期望收益。

搜寻强度的一阶条件给出 $s(h) = \min\{1, (0.6 \, G(h))^{1/0.7}\}$。

在自由放任经济中，这个界约束有效，即每个技能水平下 $s(h) = 1$，正如Ljungqvist和Sargent所报告的。

Ljungqvist和Sargent设定 $\beta = 0.9985$（对应年利率4%）、$\alpha = 0.0009$（对应预期工作年限42.7年）以及 $\lambda = 0.009$（对应裁员前预期就业年限4.3年）。

### 计算

函数 `create_model` 存储了这些原始设定。

其参数 `h_max`、`up` 和 `down` 设定了技能范围以及技能提升和损失的概率，我们将在一个练习中改变这些参数。

```{code-cell} ipython3
Model = namedtuple("Model", ("β", "α", "λ", "turbulence", "h_grid", "w_grid",
                             "f_w", "μ_e", "μ_u", "μ_l"))


def layoff_matrix(n_h, turbulence):
    """Skill transitions at a layoff: the left half of a normal distribution
    centered on 0.5, stretched over the skill levels up to the current one."""
    μ_l = np.eye(n_h)
    if turbulence > 0:
        for j in range(1, n_h):
            x = 0.5 * np.arange(j + 1) / j
            weights = np.exp(-(x - 0.5)**2 / (2 * turbulence))
            μ_l[j] = 0.0
            μ_l[j, :j + 1] = weights / weights.sum()
    return μ_l


def create_model(turbulence=0.0, β=0.9985, α=0.0009, λ=0.009,
                 h_max=2.0, up=0.1, down=0.2, n_h=21, n_w=101):
    h_grid = np.linspace(1, h_max, n_h)
    w_grid = np.linspace(0, 1, n_w)
    pdf = norm.pdf(w_grid, 0.5, np.sqrt(0.1))
    μ_e = np.eye(n_h) * (1 - up)
    μ_e[np.arange(n_h - 1), np.arange(1, n_h)] = up
    μ_e[-1, -1] = 1.0
    μ_u = np.eye(n_h) * (1 - down)
    μ_u[np.arange(1, n_h), np.arange(n_h - 1)] = down
    μ_u[0, 0] = 1.0
    return Model(β, α, λ, turbulence, h_grid, w_grid, pdf / pdf.sum(),
                 μ_e, μ_u, layoff_matrix(n_h, turbulence))
```

该图展示了对于曾处于最高技能水平的工人，在裁员后技能的分布情况，对应Ljungqvist和Sargent用来代表1970—78年和1979—87年的两个动荡参数值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Skill after a layoff from the top level
    name: fig-ti-layoff
---
fig, ax = plt.subplots()
for v in (0.02, 0.04):
    m = create_model(turbulence=v)
    ax.plot(m.h_grid, m.μ_l[-1], "o-", label=f"turbulence parameter {v}")
ax.set_xlabel("skill after the layoff")
ax.set_ylabel("probability")
ax.legend()
plt.show()
```

我们通过值迭代求解贝尔曼方程，并返回接受决策和搜寻强度。

```{code-cell} ipython3
def solve_model(m, tol=1e-9, max_iter=100_000):
    δ = (1 - m.α) * m.β
    earnings = np.outer(m.w_grid, m.h_grid)
    V = np.zeros_like(earnings)
    V_u = np.zeros(len(m.h_grid))
    for _ in range(max_iter):
        W = earnings + δ * ((1 - m.λ) * V @ m.μ_e.T
                            + m.λ * (m.μ_l @ V_u)[None, :])
        V_new = np.maximum(W, V_u[None, :])
        gain = δ * m.μ_u @ (m.f_w @ V - V_u)
        s = np.clip((0.6 * np.maximum(gain, 0))**(1 / 0.7), 0, 1)
        V_u_new = -0.5 * s + δ * m.μ_u @ V_u + s**0.3 * gain
        error = max(np.abs(V_new - V).max(), np.abs(V_u_new - V_u).max())
        V, V_u = V_new, V_u_new
        if error < tol:
            break
    accept = W >= V_u[None, :]          # accept[w, h]
    return accept, s
```

下一个函数计算工人在就业状态、工资和技能上的稳态分布，新出生的工人替代死亡的工人。

由此，我们计算失业率、平均失业持续时间，以及受雇工人的平均收入，Ljungqvist和Sargent将其称为平均生产率。

```{code-cell} ipython3
def steady_state(m, accept, s, tol=1e-12, max_iter=200_000):
    n_w, n_h = len(m.w_grid), len(m.h_grid)
    E, U = np.zeros((n_w, n_h)), np.zeros(n_h)
    U[0] = 1.0
    survive = 1 - m.α
    for _ in range(max_iter):
        E_kept = survive * (1 - m.λ) * E @ m.μ_e
        quits = np.where(accept, 0.0, E_kept).sum(axis=0)
        laid_off = survive * m.λ * E.sum(axis=0) @ m.μ_l
        offers = survive * (U * s**0.3) @ m.μ_u
        hires = np.outer(m.f_w, offers) * accept
        E_new = np.where(accept, E_kept, 0.0) + hires
        U_new = survive * U @ m.μ_u - hires.sum(axis=0) + quits + laid_off
        U_new[0] += m.α
        error = np.abs(E_new - E).max() + np.abs(U_new - U).max()
        E, U = E_new, U_new
        if error < tol:
            break
    inflow = quits.sum() + laid_off.sum() + m.α
    earnings = np.outer(m.w_grid, m.h_grid)
    return {"unemployment rate": U.sum(),
            "duration (weeks)": 2 * U.sum() / inflow,
            "average productivity": (E * earnings).sum() / E.sum()}
```

{cite:t}`LjungqvistSargent1998` 的表5报告了自由放任经济在四个动荡参数值下的这些统计量：失业率分别为6.28%、5.81%、5.77%和5.73%；平均持续时间分别为11.8周、10.6周、10.6周和10.7周；平均生产率分别为1.659、1.552、1.520和1.496。

```{code-cell} ipython3
print("turbulence   unemployment   duration   productivity   search intensity")
for v in (0.0, 0.02, 0.03, 0.04):
    m = create_model(turbulence=v)
    accept, s = solve_model(m)
    stats = steady_state(m, accept, s)
    print(f"{v:9.2f} {100 * stats['unemployment rate']:12.2f}% "
          f"{stats['duration (weeks)']:10.1f} {stats['average productivity']:13.3f}"
          f"        {s.min():.2f} to {s.max():.2f}")
```

我们的计算结果与他们的表格相符，失业率误差在0.11个百分点以内，持续时间误差在0.2周以内，生产率误差在0.002单位以内，且每个技能水平下的搜寻强度均等于一。

在自由放任经济中，动荡性降低了生产率，因为被裁员的工人会失去技能，但它对失业率的影响很小。

{cite:t}`LjungqvistSargent1998` 中福利国家模型——救济金与过去收入挂钩——的反应则大不相同：随着动荡参数从0升至0.04，其失业率从6.95%升至14.87%。

### 人工面板数据

{cite:t}`LjungqvistSargent1998` 描述了他们如何生成这些面板数据：

> 对于自由放任经济，我们为两个不同的子时期分别生成了长度为9年、包含10,000个个体的人工面板数据：相对平静的经济时期和更动荡的时期，分别对应表5中方差为0.02和0.04的指标。前一个面板数据旨在代表Gottschalk-Moffitt研究中的1970—78年期间，后一个面板数据则对应1979—87年期间。对于这两个面板数据，我们使用已在劳动力市场中工作20年的工人的均衡分布作为初始条件。

我们通过模拟新出生工人20年（以其存活为条件）来生成这样的初始条件。

下面的函数记录两周收入、就业状态、裁员情况和技能水平。

```{code-cell} ipython3
def draw_next(cum_rows, rows, u):
    """Draw the next state for each worker from rows of cumulative probabilities."""
    return (u[:, None] > cum_rows[rows]).sum(axis=1)


def simulate_workers(m, accept, s, N, T, T_burn, seed=0):
    rng = np.random.default_rng(seed)
    cum_e, cum_u, cum_l = (np.cumsum(μ, axis=1) for μ in (m.μ_e, m.μ_u, m.μ_l))
    cum_w = np.cumsum(m.f_w)
    n_w = len(m.w_grid)
    emp, wi, hj = np.zeros(N, bool), np.zeros(N, int), np.zeros(N, int)
    record = {"earn": np.zeros((N, T)), "emp": np.zeros((N, T), bool),
              "laid": np.zeros((N, T), bool), "h": np.zeros((N, T), int)}
    for t in range(T_burn + T):
        u = rng.random((N, 4))
        new_emp, new_wi, new_hj = emp.copy(), wi.copy(), hj.copy()
        laid = emp & (u[:, 0] < m.λ)
        kept = emp & ~laid
        new_hj[laid] = draw_next(cum_l, hj[laid], u[laid, 1])
        new_emp[laid] = False
        new_hj[kept] = draw_next(cum_e, hj[kept], u[kept, 1])
        new_emp[kept & ~accept[wi, new_hj]] = False          # quits
        idle = ~emp
        new_hj[idle] = draw_next(cum_u, hj[idle], u[idle, 1])
        offer = np.minimum((u[:, 2][:, None] > cum_w).sum(axis=1), n_w - 1)
        hired = idle & (u[:, 3] < s[hj]**0.3) & accept[offer, new_hj]
        new_emp[hired], new_wi[hired] = True, offer[hired]
        emp, wi, hj = new_emp, new_wi, new_hj
        if t >= T_burn:
            k = t - T_burn
            record["earn"][:, k] = np.where(emp, m.w_grid[wi] * m.h_grid[hj], 0.0)
            record["emp"][:, k], record["laid"][:, k], record["h"][:, k] = emp, laid, hj
    return record


def log_annual_earnings(record, years=9):
    """Log annual earnings with year means removed, for workers with
    positive earnings in every year."""
    annual = record["earn"].reshape(len(record["earn"]), years, 26).sum(axis=2)
    positive = (annual > 0).all(axis=1)
    y = np.log(annual[positive])
    return y - y.mean(axis=0), positive.mean()


def artificial_panel(turbulence, N=10_000, seed=0, **kwargs):
    m = create_model(turbulence=turbulence, **kwargs)
    record = simulate_workers(m, *solve_model(m), N=N, T=9 * 26,
                              T_burn=20 * 26, seed=seed)
    return log_annual_earnings(record)
```

Gottschalk和Moffitt在分解对数收入前，剔除了一个共同的年龄效应。

人工面板数据中的所有工人都有相同的工作经验，所以我们仅剔除每年对数收入的均值。

全年失业的工人没有对数收入，我们剔除这些工人，约占面板数据的四分之一个百分点。

(ti_gm_section)=
## Gottschalk和Moffitt的分解方法

### 定义

{cite:t}`GottschalkMoffitt1994` 研究了PSID中年龄在20到59岁之间、作为户主且非学生、拥有正工资薪金收入的白人男性。

他们剔除了按年龄-教育-年份分组的收入分布中最高和最低的1%。

他们分别对1970—78年和1979—87年，用年龄的四次多项式对对数实际年收入进行回归，并将男性 $i$ 在第 $t$ 年的残差 $y_{it}$ 分解为 $y_{it} = \mu_i + \nu_{it}$。

他们将男性 $i$ 出现在窗口期中 $T_i$ 年内 $y_{it}$ 的平均值 $\bar y_i$ 作为永久性成分，将 $y_{it} - \bar y_i$ 的偏差作为暂时性成分。

他们的附录给出了公式

$$
\sigma^2_\nu = \frac{1}{N} \sum_{i=1}^N \frac{1}{T_i - 1} \sum_{t=1}^{T_i} (y_{it} - \bar y_i)^2 ,
\qquad
\sigma^2_\mu = \frac{1}{N - 1} \sum_{i=1}^N (\bar y_i - \bar y)^2 - \frac{\sigma^2_\nu}{\bar T} ,
$$ (ti_gm)

其中 $\bar y$ 是所有男性和所有年份上 $y_{it}$ 的均值，$\bar T$ 是 $T_i$ 的平均值。

暂时性方差 $\sigma^2_\nu$ 是男性各自收入围绕其自身平均值的方差的平均值。

永久性方差 $\sigma^2_\mu$ 是男性平均值的方差，减去暂时性成分对该方差的贡献部分。

他们的表1报告，永久性方差从1970—78年的 $0.201$ 升至1979—87年的 $0.284$，暂时性方差从 $0.104$ 升至 $0.148$。

他们写道："收入分布扩大中，整整三分之一来自收入不稳定性的上升。"

函数 `gm_decomposition` 计算公式 {eq}`ti_gm`，函数 `trim` 剔除每年对数收入中最高和最低的1%，这会使面板数据变为非平衡面板。

```{code-cell} ipython3
gm_data = {"1970-78": np.array([0.201, 0.104]),
           "1979-87": np.array([0.284, 0.148])}
gm_change = gm_data["1979-87"] - gm_data["1970-78"]


def gm_decomposition(y, keep=None):
    """Permanent and transitory variances, formulas (ti_gm); keep marks the
    years in which each man appears."""
    if keep is None:
        keep = np.ones(y.shape, bool)
    T_i = keep.sum(axis=1)
    y_bar_i = np.where(keep, y, 0).sum(axis=1) / T_i
    σ2_ν = (np.where(keep, (y - y_bar_i[:, None])**2, 0).sum(axis=1)
            / (T_i - 1)).mean()
    σ2_μ = ((y_bar_i - y[keep].mean())**2).sum() / (len(y) - 1) - σ2_ν / T_i.mean()
    return np.array([σ2_μ, σ2_ν])


def trim(y, q=0.01):
    """Drop the top and bottom fraction q of each year's log earnings, remove
    the year means again, and keep the men who remain in at least two years."""
    lo, hi = np.quantile(y, [q, 1 - q], axis=0)
    keep = (y >= lo) & (y <= hi)
    y = y - np.where(keep, y, 0).sum(axis=0) / keep.sum(axis=0)
    two_years = keep.sum(axis=1) >= 2
    return y[two_years], keep[two_years]
```

### 作为信息投影的分解方法

设 $y_i = (y_{i1}, \ldots, y_{iT})$ 为某男性在 $T$ 年窗口期内对数收入的向量，并假设每个男性在窗口期的每一年都出现，如同在平衡面板数据中那样。

**随机效应模型**认为 $y_{it} = m + a_i + e_{it}$，其中 $a_i \sim N(0, \sigma^2_\mu)$，而 $e_{it} \sim N(0, \sigma^2_\nu)$ 彼此独立且随时间独立。

在此模型下，$y_i$ 服从均值为 $m \mathbf{1}$、协方差矩阵为 $\sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I$ 的正态分布，其中 $\mathbf{1}$ 为全一向量。

该协方差矩阵的逆是 $I$ 和 $\mathbf{1}\mathbf{1}^\top$ 的线性组合，因此随机效应模型是一个指数族。

它的充分统计量是平均值 $\bar y_i$、其平方 $\bar y_i^2$，以及组内平方和 $W_i = \sum_t (y_{it} - \bar y_i)^2$。

按照 {doc}`approximation_errors` 的记号，$T \bar y_i^2$ 是某男性收入在频率零处的周期图值，而根据帕塞瓦尔恒等式，$W_i$ 是周期图在其余 $T - 1$ 个傅里叶频率 $2\pi j / T$ 处值的总和。

该模型的收入分布是离散的，所以它与每个正态分布的KL散度都是无穷大，正如 {doc}`information_projections` 所指出的。

{prf:ref}`ip_thm_m_projection` 的证明可转而应用于交叉熵 $-E[\log q_\theta(y_i)]$，它与KL散度的区别仅在于 $y_i$ 的熵（当 $y_i$ 具有密度时）。

对于均值在每年都相同、协方差矩阵非奇异且其非对角元素的平均值非负的 $y_i$ 的任意分布，使交叉熵最小化的参数会匹配充分统计量的期望值，从而设定 $m = E[\bar y_i]$ 和

$$
E[W_i] = (T - 1) \sigma^2_\nu ,
\qquad
T \operatorname{Var}(\bar y_i) = T \sigma^2_\mu + \sigma^2_\nu .
$$

求解得到 $\sigma^2_\nu = E[W_i]/(T - 1)$ 和 $\sigma^2_\mu = \operatorname{Var}(\bar y_i) - \sigma^2_\nu / T$。

我们将这个最小化器称为 $y_i$ 分布向随机效应模型的m-投影。

该投影将 $\sigma^2_\nu$ 设为非零傅里叶频率下期望周期图值的平均，而对于均值为零的 $y_i$（如我们按年去均值的面板数据），将 $T\sigma^2_\mu + \sigma^2_\nu$ 设为频率零处的期望周期图值。

*对于平衡面板数据，Gottschalk和Moffitt的永久性方差和暂时性方差正是将某男性收入分布向随机效应模型做m-投影所得的参数*：公式 {eq}`ti_gm` 就是以样本男性代替总体的上述表达式，只是平均值方差的除数是 $N - 1$。

无论随机效应模型是否能描述数据，该投影都是良定义的。

Gottschalk和Moffitt报告，连续暂时性成分之间的相关系数为0.14和0.18，并得出结论：这些成分"基本上不相关"。

在随机效应模型下，连续偏差 $y_{it} - \bar y_i$ 之间的相关系数为 $-1/(T - 1)$，即当 $T = 9$ 时为 $-0.125$。

因此，Gottschalk和Moffitt的相关系数比序列不相关的暂时性成分所产生的结果高约 $0.3$。

正态分布的交叉熵仅通过均值和协方差矩阵 $S$ 依赖于 $y_i$ 的分布，因此我们可以通过最小化 $N(0, S)$ 与随机效应模型的KL散度来计算该投影。

该投影的参数是 $S$ 的线性函数：

$$
\sigma^2_\nu = \frac{\operatorname{tr} S - \mathbf{1}^\top S \mathbf{1} / T}{T - 1} ,
\qquad
\sigma^2_\mu = \frac{\mathbf{1}^\top S \mathbf{1}}{T^2} - \frac{\sigma^2_\nu}{T} .
$$

{ref}`ti_ex_projection` 推导了这些公式。

因此，如果对数收入是两个独立成分之和，那么总和的投影就是各投影之和。

随机效应协方差矩阵及其逆矩阵位于同一个由 $I$ 和 $\mathbf{1}\mathbf{1}^\top$ 张成的线性空间中。

充分统计量是具有空间内 $A$ 的二次型 $y_i^\top A y_i$，因此匹配其期望值会使 $S - \Sigma$ 在迹内积意义下与该空间正交。

由于 $\Sigma$ 位于该空间中，m-投影与 $S$ 向该空间的最小二乘投影一致，这使 $\sigma^2_\mu$ 等于 $S$ 的非对角元素平均值，$\sigma^2_\mu + \sigma^2_\nu$ 等于对角元素平均值。

仅仅协方差矩阵构成线性空间本身并不能使这两种投影一致：对称Toeplitz矩阵（包括平稳过程的协方差矩阵）构成了一个线性空间，但不包含它们的逆矩阵，而向该空间的两种投影是不同的。

下面的代码针对一个人工面板数据，检验这两种刻画方式以及周期图恒等式。

```{code-cell} ipython3
def random_effects_cov(θ, T=9):
    σ2_μ, σ2_ν = θ
    return σ2_μ * np.ones((T, T)) + σ2_ν * np.eye(T)


def gaussian_kl(S, Σ):
    """KL divergence of N(0, S) from N(0, Σ)."""
    M = np.linalg.solve(Σ, S)
    sign, logdet = np.linalg.slogdet(M)
    return 0.5 * (np.trace(M) - len(S) - logdet) if sign > 0 else np.inf


def euclidean(S, Σ):
    """Sum of squared differences of the distinct elements of S and Σ."""
    upper = np.triu_indices(len(S))
    return np.sum((S[upper] - Σ[upper])**2)


y02, kept = artificial_panel(0.02)
S = np.cov(y02.T)
for name, distance in (("KL", gaussian_kl), ("least squares", euclidean)):
    res = minimize(lambda p: distance(S, random_effects_cov(np.exp(p))),
                   np.log([0.02, 0.05]), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14})
    print(f"{name:14} projection: {np.round(np.exp(res.x), 5)}")
print(f"formulas (ti_gm):         {np.round(gm_decomposition(y02), 5)}")
print(f"fraction of workers kept: {kept:.4f}")

T = y02.shape[1]
periodogram = np.abs(np.fft.fft(y02, axis=1))**2 / T
W = ((y02 - y02.mean(axis=1, keepdims=True))**2).sum(axis=1)
print("largest Parseval discrepancies:",
      np.abs(periodogram[:, 0] - T * y02.mean(axis=1)**2).max(),
      np.abs(periodogram[:, 1:].sum(axis=1) - W).max())
```

对于谱密度为 $f$（按照 {doc}`approximation_errors` 的约定）的平稳过程，以 $S$ 表示的公式给出

$$
\sigma^2_\mu = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) \, \frac{K_T(\omega) - 1}{T - 1} \, d\omega ,
\qquad
\sigma^2_\nu = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) \, \frac{T - K_T(\omega)}{T - 1} \, d\omega ,
$$ (ti_fejer)

其中 $K_T(\omega) = \sin^2(T\omega/2) / (T \sin^2(\omega/2))$ 为费耶尔核（Fejér kernel），即 $T$ 个一组成的序列在频率 $\omega$ 处的周期图值；频率零处的期望周期图值为 $\frac{1}{2\pi}\int_{-\pi}^{\pi} f(\omega) K_T(\omega) \, d\omega$。

$\sigma^2_\mu$ 中加在 $f$ 上的权重在频率零处等于一，积分为零，所以 $\sigma^2_\mu$ 衡量的是频率零附近谱密度超出其平均值的程度。

当 $T = 9$ 时，费耶尔核约有十分之九的权重位于频率 $|\omega| < 2\pi/T$ 处。

让我们针对两个自回归过程，检验 {eq}`ti_fejer` 与以 $S$ 表示的公式是否一致。

```{code-cell} ipython3
n_grid = 2**14
ω = 2 * np.pi * (np.arange(n_grid) + 0.5) / n_grid - np.pi
K = np.sin(T * ω / 2)**2 / (T * np.sin(ω / 2)**2)
for ρ in (0.7, -0.4):
    f = 1 / np.abs(1 - ρ * np.exp(-1j * ω))**2
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    S_ar = ρ**lags / (1 - ρ**2)
    σ2_ν = (np.trace(S_ar) - S_ar.sum() / T) / (T - 1)
    print(f"ρ = {ρ:4}: Fejér form {np.mean(f * (K - 1)) / (T - 1):.6f}, "
          f"{np.mean(f * (T - K)) / (T - 1):.6f};  "
          f"formulas in S {S_ar.sum() / T**2 - σ2_ν / T:.6f}, {σ2_ν:.6f}")
print(f"share of the Fejér kernel's weight at |ω| < 2π/T: "
      f"{np.mean(K * (np.abs(ω) < 2 * np.pi / T)):.3f}")
```

对于负自相关过程，这些公式给出了负的 $\sigma^2_\mu$，因为它的谱密度在频率零附近低于其平均值；其向随机效应模型的m-投影位于边界 $\sigma^2_\mu = 0$ 上。

该投影在 $S$ 上的线性性，为比较不同窗口期之间的变化提供了依据，即使模型和数据的水平不同。

模型中的工人在进入劳动力市场时是完全相同的，且该模型没有测量误差，所以其永久性方差和暂时性方差远低于Gottschalk和Moffitt的结果。

假设数据的对数收入是模型所描述的成分，加上一个在两个窗口期协方差矩阵相同的独立成分之和。

那么数据投影从1970—78年到1979—87年的变化，等于模型所描述成分的投影的变化。

{doc}`approximation_errors` 中Hansen和Sargent的理想季节调整方法遵循同样的策略：它将省略了数据某一特征的模型，仅在该省略特征消失之处与数据进行比较。

在那里，比较使用季节频带之外的频率；这里则使用窗口期之间的变化。

Gottschalk和Moffitt的证据对暂时性方差的支持程度，比对永久性方差的支持程度更强，因为该假设对前者更为成立。

测量误差会抬高他们的暂时性方差，他们写道："问题在于这种偏差是否随时间发生了变化。我们没有发现有相关证据。"

他们写道："价格上涨'技能'或许是解释永久性工资离散程度变化的一个令人满意的解释"，而Ljungqvist和Sargent模型中的工人事前是完全相同的，因此他们的模型没有在进入时就固定的教育或其他特征，其价格也就无从上升。

(ti_check_section)=
## Ljungqvist和Sargent的检验

### 直方图

{cite:t}`LjungqvistSargent1998` 使用Gottschalk和Moffitt的证据，"部分用来启发我们的模型，部分用来检验我们的模型"。

他们的检验将男性平均值 $\bar y_i$ 的直方图和围绕这些平均值的标准差的直方图进行了比较。

下图将Gottschalk和Moffitt的直方图——我们从他们的图2和图4中近似读取——绘制在我们动荡参数为 $0.02$ 和 $0.04$ 的人工面板数据的直方图之上。

```{code-cell} ipython3
y04, _ = artificial_panel(0.04)

gm_fig2_bins = np.arange(-1.75, 1.01, 0.25)
gm_fig2 = {"1970-78": [0.7, 0.4, 1.1, 2.9, 5.6, 11.3, 16.9, 21.5, 23.2, 10.9, 4.1, 1.3],
           "1979-87": [1.1, 1.05, 1.95, 3.05, 6.2, 10.4, 15.1, 19.7, 18.7, 13.95,
                       6.4, 2.0]}
gm_fig4_bins = np.arange(0.1, 1.21, 0.1)
gm_fig4 = {"1970-78": [42.4, 22.8, 12.6, 8.1, 6.05, 2.55, 1.65, 1.15, 1.0, 0.75,
                       0.25, 0.45],
           "1979-87": [36.4, 22.25, 12.75, 7.8, 6.3, 4.45, 3.05, 2.3, 1.4, 0.7,
                       0.9, 1.15]}


def percent_histogram(x, centers):
    width = centers[1] - centers[0]
    edges = np.append(centers - width / 2, centers[-1] + width / 2)
    return 100 * np.histogram(np.clip(x, edges[0], edges[-1] - 1e-9), edges)[0] / len(x)


model_mean_bins = np.arange(-0.5, 0.51, 0.1)
model_sd_bins = np.arange(0.1, 0.71, 0.1)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Earnings histograms in the PSID and the model
    name: fig-ti-histograms
---
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, bins, data, title in ((axes[0, 0], gm_fig2_bins, gm_fig2, "PSID, average"),
                              (axes[0, 1], gm_fig4_bins, gm_fig4,
                               "PSID, standard deviation")):
    width = bins[1] - bins[0]
    ax.bar(bins - width / 5, data["1970-78"], width=0.4 * width, label="1970–78")
    ax.bar(bins + width / 5, data["1979-87"], width=0.4 * width, label="1979–87")
    ax.set_title(title)
    ax.set_ylabel("percent")
    ax.legend()

for ax, bins, stat, title in (
        (axes[1, 0], model_mean_bins, lambda y: y.mean(axis=1), "model, average"),
        (axes[1, 1], model_sd_bins, lambda y: y.std(axis=1, ddof=1),
         "model, standard deviation")):
    width = bins[1] - bins[0]
    for shift, y, label in ((-1, y02, "turbulence 0.02"), (1, y04, "turbulence 0.04")):
        ax.bar(bins + shift * width / 5, percent_histogram(stat(y), bins),
               width=0.4 * width, label=label)
    ax.set_title(title)
    ax.set_ylabel("percent")
    ax.legend()
plt.tight_layout()
plt.show()
```

{cite:t}`LjungqvistSargent1998` 的图14绘制了他们自己面板数据的同样直方图。

我们近似读取其柱状高度并与我们的结果进行比较。

```{code-cell} ipython3
ls_fig14 = {"average": {0.02: [0.5, 1.7, 3.9, 9.0, 16.2, 27.0, 29.7, 10.8, 1.2, 0, 0],
                        0.04: [0.8, 2.3, 5.3, 10.9, 15.0, 22.2, 25.3, 15.0, 3.4, 0, 0]},
            "standard deviation": {0.02: [44.8, 33.2, 14.5, 4.8, 1.9, 0.4, 0.3],
                                   0.04: [38.4, 35.3, 17.5, 5.3, 2.1, 0.7, 0.3]}}
for v, y in ((0.02, y02), (0.04, y04)):
    gaps = [np.abs(percent_histogram(y.mean(axis=1), model_mean_bins)
                   - ls_fig14["average"][v]).max(),
            np.abs(percent_histogram(y.std(axis=1, ddof=1), model_sd_bins)
                   - ls_fig14["standard deviation"][v]).max()]
    print(f"turbulence {v}: largest gaps from their Figure 14, in percentage points, "
          f"{gaps[0]:.1f} (averages) and {gaps[1]:.1f} (standard deviations)")
```

我们的直方图在每个区间内都与他们的结果相符，误差在约一个百分点以内。

将动荡参数从 $0.02$ 提高到 $0.04$，使两个分布都朝Gottschalk和Moffitt所发现的方向扩散。

Ljungqvist和Sargent指出，模型中平均值分布的跨度比数据的跨度小，这是因为他们模型中的工人事前是完全相同的，而且模型中标准差的增加发生在比数据中更低的标准差水平上。

### 约束函数

将动荡参数为 $v$ 的模型的收入进行Gottschalk和Moffitt分解，得到一对数值 $b(v) = (\sigma^2_\mu(v), \sigma^2_\nu(v))$，即模型收入分布向随机效应模型做m-投影所得的参数。

{cite:t}`GourierouxMonfortRenault1993` 称映射 $v \mapsto b(v)$ 为**约束函数**；{doc}`gallant_tauchen` 将其记为 $\theta^0(\rho)$。

Ljungqvist和Sargent计算了 $0.02$ 和 $0.04$ 处 $\bar y_i$ 和男性标准差的分布，这些分布的二阶矩确定了 $b(0.02)$ 和 $b(0.04)$，并将它们与1970—78年和1979—87年数据的分布进行了比较。

正如 {doc}`gallant_tauchen` 所述，间接推断选择能使模型投影与数据投影一致的参数值。

我们比较变化量 $b(v) - b(0.02)$ 与Gottschalk和Moffitt测得的变化量，因为Ljungqvist和Sargent将动荡参数 $0.02$ 赋予1970—78年，并且由于上述原因，两个投影的水平是不同的。

我们通过在一组动荡参数网格的每一点上，使用相同的随机数，模拟10,000名工人的面板数据来估计约束函数。

```{code-cell} ipython3
v_grid = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.1, 0.2, 1.0])
binding = np.array([gm_decomposition(artificial_panel(v)[0]) for v in v_grid])
b02 = binding[v_grid == 0.02][0]
b_uniform = gm_decomposition(artificial_panel(np.inf)[0])   # uniform skill losses

print("turbulence   σ²_μ     σ²_ν    share of the measured change in σ²_μ, σ²_ν")
for v, b in zip(list(v_grid) + [np.inf], list(binding) + [b_uniform]):
    share = (b - b02) / gm_change
    print(f"{v:9.2f} {b[0]:8.4f} {b[1]:8.4f}        {share[0]:6.0%}  {share[1]:6.0%}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Binding function against the measured changes
    name: fig-ti-binding
---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for k, (ax, name) in enumerate(zip(axes, ("permanent variance", "transitory variance"))):
    ax.semilogx(v_grid[1:], binding[1:, k] - b02[k], "o-", label="model")
    ax.axhline(gm_change[k], color="k", ls="--", label="PSID, 1979–87 minus 1970–78")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("turbulence parameter")
    ax.set_title(f"change in {name}")
    ax.legend()
plt.tight_layout()
plt.show()
```

在Ljungqvist和Sargent赋予1979—87年的动荡参数 $0.04$ 下，模型实现了实测永久性方差增量的约8%和暂时性方差增量的约13%。

在不同的随机数种子下，后一个比例在约9%到13%之间变化，均值约为11%。

约束函数在动荡参数大于 $0.01$ 时随动荡性上升而上升，并随着动荡参数增大而趋于平缓，因为技能损失的分布逐渐趋近于从最低水平到工人当前水平这一区间内的均匀分布。

在我们网格中最大的值处，模型实现了永久性方差增量的约21%和暂时性方差增量的约43%。

在技能损失均匀分布的极限情形下，这两个比例约为21%和44%；在不同的随机数种子下，后一个比例在约44%到49%之间变化。

Gottschalk和Moffitt对其数据进行了剔除处理，而我们尚未对模型的面板数据进行剔除。

以同样的方式剔除模型面板数据中每年的对数收入，在动荡参数为 $0.04$ 时这两个比例大致不变，而在动荡参数最大时这两个比例都会降低。

```{code-cell} ipython3
y_max, _ = artificial_panel(1.0)
b_trim = {v: gm_decomposition(*trim(y)) for v, y in ((0.02, y02), (0.04, y04),
                                                      (1.0, y_max))}
print(f"turbulence 0.02: σ²_ν {b02[1]:.4f} without trimming, "
      f"{b_trim[0.02][1]:.4f} with trimming")
for v in (0.04, 1.0):
    share = (b_trim[v] - b_trim[0.02]) / gm_change
    print(f"turbulence {v}: shares of the measured changes with trimming "
          f"{share[0]:.0%}, {share[1]:.0%}")
```

模型每年收入的1%尾部约占模型暂时性方差的三分之一，剔除处理使最大动荡参数下的暂时性份额在不同随机数种子下降至34%到37%之间。

这些份额还取决于技能的范围（它约束了技能损失的大小）以及技能转换的速度（它决定了损失持续的时间）。

{ref}`ti_ex_robust` 表明，当技能范围为 $[1, 3]$ 且转换概率保持原值时，动荡性能实现实测永久性方差增量的约46%，以及几乎全部的暂时性方差增量。

同时将转换概率加倍，则使永久性份额又回落至约五分之一，因为被裁员的工人恢复技能的速度加快了一倍。

将1970—78年的参数固定在 $0.02$ 这一点同样重要。

如果两个窗口期的动荡参数都可自由取值，模型能实现的最大变化量，就是约束函数最大值与最小值之差。

```{code-cell} ipython3
points = np.vstack([binding, b_uniform])
largest = (points.max(axis=0) - points.min(axis=0)) / gm_change
print(f"largest shares with both windows free: {largest[0]:.0%}, {largest[1]:.0%}")
```

没有任何动荡参数值能使模型投影的变化等于数据投影的变化。

基于这两个统计量的间接推断，会将动荡参数推向技能损失均匀分布的极限情形，但仍然存在差距。

Ljungqvist和Sargent观察到，其模型中变异性的增加发生在比数据更低的标准差水平上；约束函数衡量的正是两种方差增量大小之间的这种相关不足。

他们根据与Gottschalk和Moffitt、以及Jacobson、LaLonde和Sullivan的比较，得出结论：

> 这些模拟结果表明，我们模型中产生高长期失业率的机制，在比美国所观测到的经济动荡程度低得多的水平上就能发挥作用。

我们对约束函数的估计支持了他们的推断，即数据所要求的动荡性超过 $0.04$。

这些估计还表明，在该模型中，无论动荡性多大都无法达到数据中的变化幅度，所以这些变化必定也来自模型所省略的成分。

将1970—78年的参数固定在 $0.02$ 时，两个统计量和一个自由参数留下一个过度识别约束。

{doc}`gallant_tauchen` 以这种方式检验结构模型：询问在最优参数值下投影之间仍存在的距离，是否超出了抽样变异所能允许的范围。

这里的检验是对一个联合假设的检验：模型能描述收入的变化，且模型所省略的收入成分没有发生变化。

Gottschalk和Moffitt提出的、技能价格上涨可能解释永久性工资离散程度扩大这一说法，使人对第二部分假设中关于永久性方差的部分产生怀疑。

(ti_ident_section)=
## 裁员率与动荡性

约束函数将一个标量参数映射为两个数值，而其他结构性变化也可能使这两个数值朝同一方向移动。

我们在动荡参数固定为 $0.02$ 且裁员率 $\lambda$ 更高的情况下，计算模型的投影。

```{code-cell} ipython3
b04 = binding[v_grid == 0.04][0]
b_layoff = {λ: gm_decomposition(artificial_panel(0.02, λ=λ)[0])
            for λ in (0.012, 0.015, 0.018, 0.021)}
print("layoff rate   change in σ²_μ, σ²_ν from the 1970–78 calibration")
for λ, b in b_layoff.items():
    print(f"{λ:11.3f}    {b[0] - b02[0]:+.4f}  {b[1] - b02[1]:+.4f}")
print(f"turbulence 0.04:  {b04[0] - b02[0]:+.4f}  {b04[1] - b02[1]:+.4f}")
```

将裁员率翻倍或略高于翻倍，使永久性方差的增幅大致等于将动荡参数从 $0.02$ 提高到 $0.04$ 所带来的增幅，在此次模拟中暂时性方差的增幅约为 $0.003$，而不是 $0.006$；在不同随机数种子下平均约为 $0.002$，而不是 $0.005$。

这两种解释投影到随机效应模型中相邻的两个点上。

这两种解释投影的相邻性，正是 {doc}`specification_tests` 中结论的识别对应物：投影的比较对使投影保持相等的备择假设没有检验力。

一个样本能否区分这两种解释，取决于数据中公式 {eq}`ti_gm` 的抽样变异性。

Gottschalk和Moffitt的样本包含2,730名男性。

暂时性方差 $\sigma^2_\nu$ 是男性平方标准差 $\hat s_i^2$ 的平均值，所以它在2,730名男性样本中的标准差约为 $\hat s_i^2$ 的标准差除以 $\sqrt{2730}$。

我们根据对Gottschalk和Moffitt图4的读数计算该标准差，并计算一个检验的检验力，该检验将暂时性方差的变化与每种解释所预测的变化进行比较。

```{code-cell} ipython3
sd_ν = {}
for window, heights in gm_fig4.items():
    p = np.array(heights) / np.sum(heights)
    s2 = gm_fig4_bins**2
    sd_ν[window] = np.sqrt((p @ s2**2 - (p @ s2)**2) / 2730)
    print(f"{window}: σ²_ν implied by the figure {p @ s2:.3f}, "
          f"standard deviation {sd_ν[window]:.4f}")
sd_change = np.hypot(*sd_ν.values())
gap = (b04[1] - b02[1]) - (b_layoff[0.018][1] - b02[1])
power = ncx2.sf(chi2.ppf(0.95, 1), 1, (gap / sd_change)**2)
print(f"standard deviation of the change {sd_change:.4f}, gap {gap:.4f}, "
      f"power of a 5 percent test {power:.2f}")
```

我们对图形的读数得到的暂时性方差为 $0.103$ 和 $0.145$，与Gottschalk和Moffitt表1中的 $0.104$ 和 $0.148$ 接近。

两个窗口期之间变化量的标准差约为 $0.006$，所以两种解释在暂时性方差变化量上约 $0.003$ 的差异，约相当于半个标准差。

这一计算将两个窗口期视为2,730名男性的独立样本；Gottschalk和Moffitt的窗口期包含部分相同的男性，这会降低变化量的标准差，而每个窗口期所包含的男性少于2,730人，这又会提高该标准差。

利用 {ref}`st_ex_power` 的非中心性计算方法，半个标准差的分离度使5%水平的检验的检验力约为 $0.08$。

该计算所问的是，由一种解释生成的数据，能否拒绝另一种解释。

两种解释都比实测变化量 $0.044$ 低出六个以上的标准差，所以该计算衡量的是这两个统计量在模型内部能识别出什么，而非任何一种解释的拟合优度。

因此，Gottschalk和Moffitt的这两个统计量几乎无法区分更频繁的裁员和更大的技能损失。

{cite:t}`LjungqvistSargent2008` 还将其模型与 {cite:t}`JacobsonLaLondeSullivan1993` 所测得的被裁员工人的收入损失进行了比较。他们写道：

> 我们承认，自1980年代以来美国被裁员工人所经历的巨大收入损失本身，并不能说明1950—60年代与1980年代之后之间动荡性是否增加，因为这需要来自1950—60年代类似被裁员工人研究的证据，而遗憾的是，这样的研究并不存在。

他们补充道："相反，我们间接地从诸如Gottschalk和Moffitt（1994）之类的时间序列研究中，为我们关于动荡性增加的论述找到了支持依据，Katz和Autor（1999）的研究表明，Gottschalk和Moffitt的研究结论在各类研究和数据集中都具有稳健性。"

每增加一个统计量，就会为辅助模型增加充分统计量，并缩小与数据投影一致的结构参数集合。

只有当两种解释在某统计量上的投影不同，且经济学家以相同方式从模型和数据中计算该统计量时，这个额外的统计量才能区分这两种解释。

{ref}`ti_ex_jls` 为这两种解释计算了被裁员工人的相关统计量。

(ti_mg_section)=
## 协方差结构模型

{cite:t}`MoffittGottschalk2011` 描述了分解方法 {eq}`ti_gm` 的一个局限性：

> 然而，除了需要选择任意的日历时间区间外，如果永久性方差不是在区间端点处骤然变化，而是在区间内平滑上升，该方法就会出问题。在这种情况下，区间前段和后段相对于平均收入的偏差，会被错误地归因于暂时性收入。

用周期图的语言来说，在一个窗口期内发生变化的永久性成分，会在非零傅里叶频率处产生功率，而该投影将其计为暂时性成分。

他们转而拟合了一个描述收入整体协方差结构的模型，其中永久性成分服从随机游走，暂时性成分服从ARMA(1,1)过程：

$$
y_{it} = \mu_{it} + \nu_{it} ,
\qquad
\mu_{it} = \mu_{i,t-1} + \omega_{it} ,
\qquad
\nu_{it} = \rho \nu_{i,t-1} + \xi_{it} + \theta \xi_{i,t-1} ,
$$

其中载荷和方差随时间呈趋势性变化。

当 $\rho = \theta = 0$ 时，暂时性成分为白噪声，该模型即为 {doc}`phillips_misspecified` 和 {doc}`lq_permanent_income` 所使用的 {cite:t}`Muth1960` 带噪声观测的随机游走模型。

他们采用最小距离法，以单位权重矩阵估计该模型，即通过最小二乘法使模型的自协方差与样本自协方差相拟合。

他们报告的卡方统计量在每个设定下都超过了1%的临界值，不过他们判断，具有ARMA(1,1)暂时性成分的随机游走模型"能够充分拟合我们的数据"。

我们将参数恒定的版本，分别用最小二乘法（如Moffitt和Gottschalk所做）以及最小化高斯KL散度的方法，拟合到我们人工面板数据九年期的协方差矩阵上。

其参数包括：窗口期前一年永久性成分的方差 $\sigma^2_0$、其增量的方差 $\sigma^2_\omega$，以及 $\rho$、$\theta$ 和暂时性成分创新项 $\xi_{it}$ 的方差 $\sigma^2_\xi$。

```{code-cell} ipython3
def mg_cov(θ, T=9):
    σ2_0, σ2_ω, ρ, ϑ, σ2_ξ = θ
    t = np.arange(1, T + 1)
    lag = np.abs(np.subtract.outer(t, t))
    γ0 = σ2_ξ * (1 + 2 * ρ * ϑ + ϑ**2) / (1 - ρ**2)
    γ1 = σ2_ξ * (1 + ρ * ϑ) * (ρ + ϑ) / (1 - ρ**2)
    transitory = np.where(lag == 0, γ0, γ1 * ρ**np.maximum(lag - 1, 0))
    return σ2_0 + σ2_ω * np.minimum.outer(t, t) + transitory


def mg_params(p):
    return np.array([np.exp(p[0]), np.exp(p[1]), np.tanh(p[2]), np.tanh(p[3]),
                     np.exp(p[4])])


def mg_projection(S, distance):
    best = None
    for p0 in ([np.log(0.02), np.log(0.002), 0.5, -0.3, np.log(0.02)],
               [np.log(0.01), np.log(0.0005), 1.0, 0.0, np.log(0.04)]):
        objective = lambda p: distance(S, mg_cov(mg_params(p)))
        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"maxiter": 40_000, "xatol": 1e-10, "fatol": 1e-15})
        res = minimize(objective, res.x, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    return mg_params(best.x)


print("turbulence  distance       σ²_0     σ²_ω      ρ       θ      σ²_ξ   "
      "permanent  transitory")
for v in (0.02, 0.04):
    y_v = artificial_panel(v, N=40_000, seed=1)[0]
    S = np.cov(y_v.T)
    for name, distance in (("least sq.", euclidean), ("KL", gaussian_kl)):
        θ = mg_projection(S, distance)
        permanent = θ[0] + 5 * θ[1]                   # middle year of the window
        transitory = mg_cov(θ)[4, 4] - permanent
        print(f"{v:9.2f}  {name:10} {θ[0]:8.4f} {θ[1]:8.5f} {θ[2]:7.3f} {θ[3]:7.3f} "
              f"{θ[4]:8.4f} {permanent:9.4f} {transitory:10.4f}")
    print(f"           random effects (ti_gm):  permanent {gm_decomposition(y_v)[0]:.4f}, "
          f"transitory {gm_decomposition(y_v)[1]:.4f}")
```

协方差结构模型几乎将方差完全归因于暂时性成分，而不是永久性成分。

它将模型的收入不稳定性归因于一个年自回归系数介于 $0.63$ 和 $0.71$ 之间、移动平均系数约为 $-0.26$ 的暂时性成分。

Moffitt和Gottschalk在其表5中基于PSID的估计是 $\rho = 0.641$ 和 $\theta = -0.367$。

协方差结构模型将动荡参数从 $0.02$ 升至 $0.04$ 所带来的增量归因于暂时性成分。

对同一面板数据的随机效应投影 {eq}`ti_gm` 则将增量的约一半归因于永久性成分。

这两个辅助模型之所以意见不一致，是因为裁员时的技能损失较大、持续时间长，而且最终会被逆转。

一名损失了一半累积技能的工人，以平均每十个就业期恢复一个技能水平（即 $0.05$）的速度恢复技能，所以该工人的收入会在数年内保持低位。

在一个九年窗口期的早期发生这样一次损失，会降低该工人的平均值，并提高工人之间平均值的方差，而随机效应投影将此计为永久性成分。

用周期图的语言来说，一种持续数年的损失，会将其大部分功率集中在频率零处。

模型中收入的协方差随滞后期而下降，协方差结构模型用其暂时性成分来拟合这种下降，因此它将同样的损失计为暂时性成分。

Moffitt和Gottschalk也注意到了这个问题："如果暂时性成分存在序列相关性，平均收入的方差将同时受到暂时性成分方差的影响。"

模型中工人相对于其自身平均值的偏差，具有较小的序列相关性，与Gottschalk和Moffitt在PSID中的发现类似。

```{code-cell} ipython3
for v, y in ((0.02, y02), (0.04, y04)):
    d = y - y.mean(axis=1, keepdims=True)
    r = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    print(f"turbulence {v}: correlation of successive deviations {r:.2f}")
```

$0.17$ 和 $0.22$ 的连续偏差相关系数，与自回归系数接近 $0.7$ 的暂时性成分是相容的，所以PSID中较小的相关系数并不能表明其暂时性成分接近序列不相关。

最小二乘投影和KL投影向协方差结构模型的投影在细节上有所不同。

该模型不是一个指数族：它的协方差矩阵构成一个弯曲的集合，使两种投影对随机效应模型一致的那个论证不适用于此。

## 结束语

Ljungqvist和Sargent对直方图的比较，是间接推断的一种非正式版本：他们直方图的二阶矩，正是数据和结构模型向同一辅助模型做m-投影所得的参数。

对于 {cite:t}`LjungqvistSargent1998` 的自由放任经济，Gottschalk和Moffitt分解方法的约束函数在动荡参数大于 $0.01$ 时随动荡性而上升。

将1970—78年的参数固定在他们所使用的 $0.02$，约束函数无法达到Gottschalk和Moffitt测得的变化量，而它能接近到何种程度，取决于技能的范围以及被裁员的工人恢复技能的速度。

同样的两个统计量对裁员率的上升也有类似的反应，因此它们只能较弱地识别动荡性。

关于不稳定性有多少属于永久性的问题，其答案取决于所用的辅助模型：随机效应分解方法和 {cite:t}`MoffittGottschalk2011` 的协方差结构模型，对同一模型生成的不稳定性给出了不同的划分结果。

## 练习

```{exercise}
:label: ti_ex_jls

{cite:t}`JacobsonLaLondeSullivan1993` 将1982年第一季度被裁员、工龄为六年的工人的季度收入，与1986年底仍留在原雇主处工作的工人的收入进行了比较。

{cite:t}`LjungqvistSargent1998` 通过追踪100,000名已在劳动力市场工作20年的工人11年的情况，构建了模型中的对应统计量。他们将第7年的前12周称为1982年第一季度，并绘制了技能损失超过40%的离职者的情况。

1. 在动荡参数为 $0.04$ 的自由放任经济中，模拟100,000名这样的工人。识别出留任者，即每一期都受雇且从未被裁员的工人。识别出离职者，即在前六年中一直受雇且从未被裁员，而在第7年的前六个时期被裁员的工人。
2. 计算每组的平均12周收入，以及离职者与留任者收入在最后两年的比率。分别对所有离职者，以及裁员时技能下降超过40%的离职者进行此计算。
3. 在动荡参数为 $0.02$、裁员率为 $0.019$（这是永久性方差的变化大致匹配将动荡性提高至 $0.04$ 所带来变化的裁员率）的经济中，对所有离职者重复第2部分的计算，并计算在每种经济中，工龄六年的工人在第7年开始时被裁员的比例。
4. 这些统计量中，哪些是经济学家能够从像Jacobson、LaLonde和Sullivan那样的数据中计算出来的？其中哪些能够区分这两种解释？
```

```{solution-start} ti_ex_jls
:class: dropdown
```

下面是一种解法。

```{code-cell} ipython3
def displaced_worker_statistics(turbulence, λ=0.009, N=100_000, seed=3):
    m = create_model(turbulence=turbulence, λ=λ)
    rec = simulate_workers(m, *solve_model(m), N=N, T=11 * 26, T_burn=20 * 26,
                           seed=seed)
    emp, laid, h = rec["emp"], rec["laid"], rec["h"]
    k0 = 6 * 26                                        # first period of year 7
    tenured = emp[:, :k0].all(axis=1) & ~laid[:, :k0].any(axis=1)
    stayers = emp.all(axis=1) & ~laid.any(axis=1)
    separators = tenured & laid[:, k0:k0 + 6].any(axis=1)
    rows = np.where(separators)[0]
    k_lay = k0 + laid[rows, k0:k0 + 6].argmax(axis=1)
    loss = np.round(1 - m.h_grid[h[rows, k_lay]] / m.h_grid[h[rows, k_lay - 1]], 9)
    later = np.arange(laid.shape[1])[None, :] > k_lay[:, None]
    big_loss = np.isin(np.arange(N), rows[loss > 0.4])

    def quarterly(mask):
        return rec["earn"][mask][:, :282].reshape(mask.sum(), 47, 6).sum(axis=2).mean(axis=0)

    stay = quarterly(stayers)
    ratio = lambda mask: quarterly(mask)[-8:].mean() / stay[-8:].mean()
    return {"stayers": stayers.sum(), "separators": separators.sum(),
            "shares of separators with losses above 10, 20, 30, 40%":
                np.round([(loss > c).mean() for c in (0.1, 0.2, 0.3, 0.4)], 3),
            "displacement rate": separators.sum() / tenured.sum(),
            "ratio, all separators": ratio(separators),
            "separators with losses above 40%": (loss > 0.4).sum(),
            "later layoffs per separator": (laid[rows] & later).sum(axis=1).mean(),
            "ratio, losses above 40%": ratio(big_loss) if big_loss.any() else np.nan,
            "paths": (stay, quarterly(separators),
                      quarterly(big_loss) if big_loss.any() else None)}


turbulent = displaced_worker_statistics(0.04)
more_layoffs = displaced_worker_statistics(0.02, λ=0.019)
for name, res in (("turbulence 0.04", turbulent),
                  ("turbulence 0.02, layoff rate 0.019", more_layoffs)):
    print(name)
    for key, value in res.items():
        if key != "paths":
            print(f"   {key}: {value:.3f}" if isinstance(value, float)
                  else f"   {key}: {value}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Earnings of stayers and separators
    name: fig-ti-jls
---
fig, ax = plt.subplots()
stay, sep, big = turbulent["paths"]
quarters = np.arange(len(stay)) * 12 / 52 - 6
ax.plot(quarters, stay, label="stayers")
ax.plot(quarters, sep, label="all separators")
ax.plot(quarters, big, label="separators with skill losses above 40%")
ax.set_xlabel("years from the start of year 7")
ax.set_ylabel("12-week earnings")
ax.legend(loc="lower left", fontsize=9)
plt.show()
```

Ljungqvist和Sargent在其100,000名工人中报告了2,889名留任者和629名离职者。

我们的模拟结果中，留任者数量约为其两倍，离职者数量约多出三分之二。

离职者中技能损失超过10%、20%、30%和40%的比例分别为55%、27%、10%和2.5%，而他们报告的比例分别为47%、25%、11%和3%。

我们尚未找到这一差异的来源。

在动荡参数为 $0.04$ 的情况下，所有离职者的收入在裁员后三到五年比留任者低约17%，而技能下降超过40%的少数离职者的收入则低约37%。

Ljungqvist和Sargent将Jacobson、LaLonde和Sullivan的发现描述为一种恢复，"最终达到比被裁员前工作所挣收入低约25%的水平"。

他们的数据是通过裁员这一事件而非技能损失的大小来识别被裁员工人的，所以数据所能提供的收入统计量，正是针对所有离职者的那个统计量。

他们的记录同时涵盖留任者和离职者，所以类似的数据也能提供离职率。

在裁员率更高的情况下，所有离职者的收入比留任者低约21%，因为在被裁员之后，他们再次被裁员的频率大约是动荡经济中离职者的两倍。

因此，收入损失这一统计量几乎无法区分这两种解释。

裁员率更高的经济中的离职率大约是动荡经济中离职率的两倍。

单从一期来衡量，该比率只有在我们信任模型的离职水平时才能区分这两种解释，而贯穿本讲座，我们比较的是变化量而非水平值。

两期之间离职率的变化可以区分这两种解释：当裁员率翻倍时，该变化也翻倍，而不随动荡性变化。

要测量这种变化，需要两个时期都具备可比的被裁员工人研究数据，而上文引用的 {cite:t}`LjungqvistSargent2008` 那段话正说明这正是所欠缺的。

```{solution-end}
```

```{exercise}
:label: ti_ex_projection

设 $y_i$ 为 $\mathbb{R}^T$ 中均值为零的随机向量，具有正定协方差矩阵 $S$，其非对角元素的平均值非负，并考虑随机效应族 $N(0, \sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I)$。

1. 证明 $\Sigma = \sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I$ 的逆为 $\sigma_\nu^{-2} \left(I - c\,\mathbf{1}\mathbf{1}^\top\right)$，其中 $c = \sigma^2_\mu / (\sigma^2_\nu + T \sigma^2_\mu)$，并证明 $\log p(y_i)$ 是 $W_i$ 和 $T \bar y_i^2$ 的线性函数，加上一个只依赖于参数的函数。
2. 对 $(\sigma^2_\mu, \sigma^2_\nu)$ 最小化 $N(0, S)$ 与 $N(0, \Sigma)$ 的高斯KL散度，并证明最小化器满足 $\sigma^2_\nu = (\operatorname{tr} S - \mathbf{1}^\top S \mathbf{1}/T)/(T - 1)$ 和 $\sigma^2_\mu = \mathbf{1}^\top S \mathbf{1}/T^2 - \sigma^2_\nu / T$。
3. 证明 $S$ 向同一族的最小二乘投影给出相同的结果。
```

```{solution-start} ti_ex_projection
:class: dropdown
```

下面是一种解法。

对于第1部分，将 $\Sigma$ 与 $\sigma_\nu^{-2}(I - c\,\mathbf{1}\mathbf{1}^\top)$ 相乘，并利用 $\mathbf{1}^\top \mathbf{1} = T$：其乘积为 $I + \sigma_\nu^{-2}\left[\sigma^2_\mu - c(\sigma^2_\nu + T\sigma^2_\mu)\right]\mathbf{1}\mathbf{1}^\top = I$。

于是 $y_i^\top \Sigma^{-1} y_i = \sigma_\nu^{-2}\left(y_i^\top y_i - c T^2 \bar y_i^2\right)$，且 $y_i^\top y_i = W_i + T \bar y_i^2$，所以

$$
\log p(y_i) = -\frac{W_i}{2\sigma^2_\nu} - \frac{T \bar y_i^2}{2(\sigma^2_\nu + T\sigma^2_\mu)} - \frac{1}{2}\log\det(2\pi\Sigma) ,
$$

因为 $1 - cT = \sigma^2_\nu / (\sigma^2_\nu + T\sigma^2_\mu)$。

对于第2部分，$N(0, S)$ 与 $N(0, \Sigma)$ 之间的KL散度，等于在 $N(0, S)$ 下 $\log p(y_i)$ 期望的相反数，加上一个常数。

$\Sigma$ 的特征值为 $\sigma^2_\nu + T\sigma^2_\mu$（对应特征向量 $\mathbf{1}$）以及 $\sigma^2_\nu$（重数为 $T - 1$），所以 $\log\det\Sigma = \log(\sigma^2_\nu + T\sigma^2_\mu) + (T - 1)\log\sigma^2_\nu$。

利用 $E[W_i] = \operatorname{tr} S - \mathbf{1}^\top S \mathbf{1}/T$ 和 $E[T\bar y_i^2] = \mathbf{1}^\top S \mathbf{1}/T$，目标函数可以分离为 $a = \sigma^2_\nu$ 的函数和 $d = \sigma^2_\nu + T\sigma^2_\mu$ 的函数：

$$
\frac{E[W_i]}{2a} + \frac{T-1}{2}\log a + \frac{E[T\bar y_i^2]}{2d} + \frac{1}{2}\log d .
$$

最小化给出 $a = E[W_i]/(T-1)$ 和 $d = E[T\bar y_i^2]$，即题目所述的公式。

对于第3部分，最小二乘问题 $\min \sum_{t \le s} (S_{ts} - \Sigma_{ts})^2$ 分离为对角元素（都等于 $\sigma^2_\mu + \sigma^2_\nu$）和非对角元素（都等于 $\sigma^2_\mu$）。

最小化器使 $\sigma^2_\mu + \sigma^2_\nu$ 等于对角元素平均值 $\bar s_d$，使 $\sigma^2_\mu$ 等于非对角元素平均值 $\bar s_o$。

由于 $\operatorname{tr} S = T \bar s_d$ 且 $\mathbf{1}^\top S \mathbf{1} = T\bar s_d + T(T - 1)\bar s_o$，第2部分的公式给出 $\sigma^2_\nu = \bar s_d - \bar s_o$ 和 $\sigma^2_\mu = \bar s_o$。

所以 $\sigma^2_\mu \geq 0$ 当且仅当 $S$ 的非对角元素平均值非负，这正是本题假设该条件成立的原因。

```{code-cell} ipython3
S = np.cov(y04.T)
T = len(S)
s_d, s_o = np.mean(np.diag(S)), (S.sum() - np.trace(S)) / (T * (T - 1))
print("least squares:", np.round([s_o, s_d - s_o], 6))
σ2_ν = (np.trace(S) - S.sum() / T) / (T - 1)
print("KL formulas:  ", np.round([S.sum() / T**2 - σ2_ν / T, σ2_ν], 6))
```

```{solution-end}
```

```{exercise}
:label: ti_ex_robust

{cite:t}`LjungqvistSargent1998` 报告："在早期的模拟中，我们曾将技能范围以及持续受雇两期获得一个技能水平、持续失业两期失去一个技能水平的转换概率同时加倍，得到了类似的结果。"

1. 将技能设定在 $[1, 3]$ 上，保持21个技能水平以及原有的转换概率，重新计算动荡参数为 $0.02$、$0.04$、$0.1$ 和 $1$ 时的约束函数。
2. 同时还将技能提升的概率加倍为 $0.2$，技能下降的概率加倍为 $0.4$，重新计算约束函数。

第2部分是我们对他们这句话的理解；第1部分则将范围的影响与转换概率的影响区分开来。

在每种情形下，动荡性能够实现实测永久性方差和暂时性方差变化的多大比例？这两种情形为何会有所不同？
```

```{solution-start} ti_ex_robust
:class: dropdown
```

下面是一种解法。

```{code-cell} ipython3
grid, ends = (0.02, 0.04, 0.1, 1.0), (0.02, 1.0)
cases = (("wider range", dict(h_max=3.0), grid),
         ("wider range, faster transitions", dict(h_max=3.0, up=0.2, down=0.4), grid),
         ("wider range, faster skill gains only", dict(h_max=3.0, up=0.2), ends),
         ("wider range, faster skill losses only", dict(h_max=3.0, down=0.4), ends))
for case, kwargs, v_wide in cases:
    b_wide = np.array([gm_decomposition(artificial_panel(v, **kwargs)[0])
                       for v in v_wide])
    print(case)
    for v, b in zip(v_wide, b_wide):
        share = (b - b_wide[0]) / gm_change
        print(f"   turbulence {v:5.2f}: σ²_μ {b[0]:.4f}, σ²_ν {b[1]:.4f}, "
              f"shares of the measured changes {share[0]:.0%}, {share[1]:.0%}")
```

更宽的技能范围允许更大的技能损失。

当技能范围为 $[1, 3]$ 且转换概率保持原值时，动荡性能够实现实测永久性方差增量的高达约46%，以及几乎全部的暂时性方差增量。

同时将转换概率加倍，会使被裁员工人恢复所损失技能水平所需的时间减半。

技能损失在九年窗口期中所占的比重因而变小，并使工人的平均值下降得更少，所以随机效应投影将其中较少的部分计为永久性成分。

在这种情形下，动荡性最多只能实现永久性方差增量的约五分之一（与原始经济中一样），以及略高于全部的暂时性方差增量。

恢复速度的加快才是关键因素：仅将技能提升概率加倍，会使动荡参数为 $1$ 时的永久性份额降至约21%，而仅将技能下降概率加倍，则使其保持在约41%。

Ljungqvist和Sargent的那句话涉及的是其定性发现；而约束函数表明，仅改变技能范围时，永久性份额并不具有稳健性。

```{solution-end}
```