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
    Laissez-faire economy::Artificial panels: 人为面板
    Gottschalk and Moffitt's decomposition: 戈特沙尔克和莫菲特的分解
    Gottschalk and Moffitt's decomposition::Definitions: 定义
    Gottschalk and Moffitt's decomposition::Decomposition as information projection: 作为信息投影的分解
    Ljungqvist and Sargent's check: 柳奎斯特和萨金特的检验
    Ljungqvist and Sargent's check::Histograms: 直方图
    Ljungqvist and Sargent's check::Where the distributions move: 分布移动的位置
    Ljungqvist and Sargent's check::Binding function: 约束函数
    Layoff rates and turbulence: 解雇率与动荡
    Covariance-structure model: 协方差结构模型
    Concluding remarks: 结语
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

{cite:t}`LjungqvistSargent1998` 和 {cite:t}`LjungqvistSargent2008` 将1980年后欧洲持续高失业率归因于与过去收入挂钩的失业救济金，与他们所称的**经济动荡**（即失业期间失去技能的风险）增加之间的相互作用。

为了检验动荡增加的说法是否与美国的收入证据相一致，{cite:t}`LjungqvistSargent1998` 在两个动荡水平下，用他们模型的自由放任版本生成了人为的收入面板数据。

他们将 {cite:t}`GottschalkMoffitt1994` 曾应用于1970-78年和1979-87年收入动态面板研究（PSID）中男性数据的永久性和暂时性成分分解方法，应用到这些面板数据上，并将得到的直方图与戈特沙尔克和莫菲特的结果进行了比较。

{cite:t}`LjungqvistSargent2008` 在一个更丰富的模型中重复了这一检验，两个时间窗口的动荡参数分别为 $0.1$ 和 $0.2$。

{cite:t}`LjungqvistSargent1998` 得出结论："我们动荡参数的增加使戈特沙尔克-莫菲特分解的两个成分的分布朝着观察到的方向扩散。"他们补充道：

> 同样值得注意的是，我们图14b中更动荡时期收入变异性的增加发生在比戈特沙尔克和莫菲特的研究结果更低的标准差水平上。在这方面，我们对1980年代经济动荡的参数化不及美国已记录的变化那样大。

在本讲座中，我们将他们的检验描述为信息投影的比较，并针对 {cite:t}`LjungqvistSargent1998` 的模型将其量化。

* 戈特沙尔克和莫菲特的分解是一个均衡面板中一位男性九年年度对数收入分布向高斯随机效应模型的m-投影，在 {doc}`information_projections` 的意义上。
* 柳奎斯特和萨金特的检验将数据中男性平均值和标准差的直方图与其模型在两个动荡参数值下的直方图进行比较；这些直方图的二阶矩就是投影的参数。在该参数的每个取值上计算模型的投影，就得到 {cite:t}`GourierouxMonfortRenault1993` 所称的约束函数，而选择参数使两个投影一致，就是 {doc}`gallant_tauchen` 所描述的间接推断方法。
* 在1970-78年动荡参数固定在柳奎斯特和萨金特所取的值 $0.02$ 的情况下，1979-87年参数的任何取值都无法使自由放任模型给出戈特沙尔克和莫菲特测得的永久性方差增量的五分之一以上，或暂时性方差增量的45%以上。若两个参数都可自由取值，上述界限分别约为三分之一和二分之一。
* 这些界限取决于技能的范围和技能转变的速度。单独将范围加倍可使动荡解释永久性方差增量的约46%和几乎全部暂时性方差增量；同时将转变概率也加倍，则永久性份额又回落到约五分之一。
* 解雇率的提高会使戈特沙尔克和莫菲特的两个统计量朝与动荡增加相同的方向变动，而在戈特沙尔克和莫菲特样本规模的样本中，这两个统计量几乎无法区分这两种解释。
* {cite:t}`MoffittGottschalk2011` 的协方差结构模型，作为第二个辅助模型，几乎将动荡所产生的收入不稳定性全部归因于暂时性成分，而戈特沙尔克和莫菲特的分解则将其中约一半归因于永久性成分。

本讲座的结构如下。

* {ref}`ti_model_section` 描述了 {cite:t}`LjungqvistSargent1998` 的自由放任经济并再现了他们的稳态统计量。
* {ref}`ti_gm_section` 将戈特沙尔克和莫菲特的分解描述为一种信息投影。
* {ref}`ti_check_section` 再现了柳奎斯特和萨金特的检验并计算了约束函数。
* {ref}`ti_ident_section` 将动荡与更高的解雇率进行比较。
* {ref}`ti_mg_section` 描述了第二个辅助模型。

本讲座建立在 {doc}`information_projections`、{doc}`gallant_tauchen` 和 {doc}`specification_tests` 的基础之上。

{doc}`mccall_model` 和 {doc}`mccall_model_with_separation` 描述了更简单的搜索模型。

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

{cite:t}`LjungqvistSargent1998` 研究了一个风险中性工人的连续统。

一个时期是两周。

每个工人在各期之间以概率 $\alpha$ 死亡，并最大化 $E_t \sum_{i=0}^\infty \beta^i (1 - \alpha)^i y_{t+i}$，其中 $y_t$ 是扣除搜索负效用后的净收入。

新出生的工人以最低技能水平开始失业。

工人的技能 $h$ 取在 $[1, 2]$ 上均匀分布的21个值之一。

有工作的工人赚取 $w h$，其中 $w$ 是该工作中每单位技能的工资。

每期开始时，有工作的工人以概率 $\lambda$ 被解雇。

未被解雇的工人以概率 $0.1$ 获得一个技能等级的提升，并一直保持工资 $w$ 直到被解雇、辞职或死亡。

失业工人的技能每期以概率 $0.2$ 下降一个等级，且最低等级是吸收态。

失业工人以负效用 $c(s) = 0.5 s$ 选择搜索强度 $s \in [0, 1]$，并以概率 $\pi(s) = s^{0.3}$ 在下期获得一个工资报价。

工资报价从均值为 $0.5$、方差为 $0.1$ 的正态分布中抽取，并截断在 $[0, 1]$ 上。

柳奎斯特和萨金特如下定义动荡：

> 经济动荡以与解雇相关的技能损失的均值和方差来定义。我们现在让一个刚被解雇的工人的技能按正态分布的左半部分分布，范围从最低可能技能水平开始，到该工人被解雇前的技能水平结束。

他们通过截取以 $0.5$ 为中心并限制在 $[0, 1]$ 上的正态分布的左半部分，将其拉伸到从最低技能等级到工人被解雇前所持有的等级之间，并进行归一化，从而得到该分布。

该正态分布的方差表征动荡程度；我们称之为**动荡参数**。

当动荡参数为零时，被解雇的工人保持解雇前所持有的技能水平。

令 $\mu_e(h, h')$、$\mu_u(h, h')$ 和 $\mu_l(h, h')$ 分别为有工作的工人、失业工人，以及刚被解雇的工人的技能转移概率。

自由放任经济中没有失业救济金，也没有税收。

一个持有或被提供工资为 $w$ 的工作、技能为 $h$ 的工人的值 $V(w, h)$，以及一个失业工人的值 $V_u(h)$，满足

$$
V(w, h) = \max \left\{ w h + (1 - \alpha)\beta \left[ (1 - \lambda) \sum_{h'} \mu_e(h, h') V(w, h') + \lambda \sum_{h'} \mu_l(h, h') V_u(h') \right], \; V_u(h) \right\}
$$

以及

$$
V_u(h) = \max_{s \in [0, 1]} \left\{ -c(s) + (1 - \alpha)\beta \sum_{h'} \mu_u(h, h') \left[ (1 - \pi(s)) V_u(h') + \pi(s) \int V(w, h') \, dF(w) \right] \right\} .
$$

$V(w, h)$ 中max的第二个参数是拒绝报价或辞职的选项。

令 $G(h) = (1 - \alpha)\beta \sum_{h'} \mu_u(h, h') [\int V(w, h') \, dF(w) - V_u(h')]$ 为一个报价的预期收益。

搜索强度的一阶条件给出 $s(h) = \min\{1, (0.6 \, G(h))^{1/0.7}\}$。

在自由放任经济中，该约束是紧的，正如柳奎斯特和萨金特所报告的，$s(h) = 1$ 在每个技能等级上都成立。

柳奎斯特和萨金特设定 $\beta = 0.9985$（年利率4%），$\alpha = 0.0009$（预期工作年限42.7年），以及 $\lambda = 0.009$（解雇前预期就业4.3年）。

### 计算

函数 `create_model` 存储了基本参数。

它的参数 `h_max`、`up` 和 `down` 设定了技能的范围以及技能获得和损失的概率，我们将在一道练习中更改这些参数。

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

该图显示了一个拥有最高技能水平的工人被解雇后的技能分布，针对柳奎斯特和萨金特用来代表1970-78年和1979-87年的两个动荡参数值。

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

我们通过值迭代求解贝尔曼方程，并返回接受决策和搜索强度。

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

下一个函数计算工人在就业状态、工资和技能之间的平稳分布，新出生的工人替代死亡的工人。

由此我们可以计算失业率、平均失业持续时间，以及有工作工人的平均收入（柳奎斯特和萨金特称之为平均生产率）。

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

{cite:t}`LjungqvistSargent1998` 的表5在四个动荡参数值下报告了自由放任经济的这些统计量：失业率分别为6.28、5.81、5.77和5.73百分比；平均持续时间分别为11.8、10.6、10.6和10.7周；平均生产率分别为1.659、1.552、1.520和1.496。

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

我们的计算结果与他们的表格相符，失业率差在0.11个百分点以内，持续时间差在0.2周以内，生产率差在0.002单位以内，且搜索强度在每个技能等级都等于一。

在自由放任经济中，动荡降低了生产率，因为被解雇的工人会失去技能，而动荡几乎不改变失业率。

{cite:t}`LjungqvistSargent1998` 的福利国家模型（其中救济金与过去收入挂钩）的反应则截然不同：随着动荡参数从0升到0.04，其失业率从6.95%升至14.87%。

### 人为面板

{cite:t}`LjungqvistSargent1998` 描述了他们如何生成面板数据：

> 对于自由放任经济，我们生成了长度为9年、有10,000个个体的人为面板，针对两个不同的子时期：相对平静的经济时期和更为动荡的时期，在表5中分别以方差0.02和0.04来表示。前一个面板代表戈特沙尔克-莫菲特研究中的1970-78年时期，后一个面板则对应1979-87年时期。对于两个面板，我们使用已在劳动力市场中工作20年的工人的均衡分布作为初始条件。

我们通过模拟新出生工人20年（以其存活为条件）来绘制此类初始条件。

下面的函数记录了两周收入、就业、解雇和技能。

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

戈特沙尔克和莫菲特在分解对数收入之前先剔除了一个共同的年龄轨迹。

在人为面板中，所有工人都拥有相同的工作经验，因此我们只剔除每年对数收入的均值。

整年失业的工人没有对数收入，我们剔除这些工人，约占面板的四分之一百分点。

(ti_gm_section)=
## 戈特沙尔克和莫菲特的分解

### 定义

{cite:t}`GottschalkMoffitt1994` 研究了PSID中年龄在20至59岁之间、作为户主且非学生、并具有正工资和薪金收入的白人男性。

他们在年龄-教育-年份分组内对收入分布的最高和最低1%进行了截尾处理。

他们分别对1970-78年和1979-87年将实际年收入对数对年龄的四次方项进行回归，并将男性 $i$ 在第 $t$ 年的残差 $y_{it}$ 分解为 $y_{it} = \mu_i + \nu_{it}$。

他们将男性 $i$ 在其出现于某个窗口内的 $T_i$ 年中 $y_{it}$ 的均值 $\bar y_i$ 作为永久性成分，将偏差 $y_{it} - \bar y_i$ 作为暂时性成分。

他们的附录给出了公式

$$
\sigma^2_\nu = \frac{1}{N} \sum_{i=1}^N \frac{1}{T_i - 1} \sum_{t=1}^{T_i} (y_{it} - \bar y_i)^2 ,
\qquad
\sigma^2_\mu = \frac{1}{N - 1} \sum_{i=1}^N (\bar y_i - \bar y)^2 - \frac{\sigma^2_\nu}{\bar T} ,
$$ (ti_gm)

其中 $\bar y$ 是所有男性和所有年份中 $y_{it}$ 的均值，$\bar T$ 是 $T_i$ 的平均值。

暂时性方差 $\sigma^2_\nu$ 是男性各自围绕其均值的收入方差的平均值。

永久性方差 $\sigma^2_\mu$ 是男性均值的方差，减去暂时性成分对该方差的贡献部分。

他们的表1报告称，永久性方差从1970-78年的 $0.201$ 升至1979-87年的 $0.284$，暂时性方差从 $0.104$ 升至 $0.148$。

他们写道："收入分布扩大的整整三分之一是由收入不稳定性的上升所导致的。"

函数 `gm_decomposition` 计算公式 {eq}`ti_gm`，函数 `trim` 剔除每年对数收入的最高和最低1%，这会使面板变得不平衡。

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

### 作为信息投影的分解

令 $y_i = (y_{i1}, \ldots, y_{iT})$ 为一个工人在 $T$ 年窗口内对数收入的向量，并假设每个工人在窗口内的每一年都出现，如平衡面板那样。

**随机效应模型**称 $y_{it} = m + a_i + e_{it}$，其中 $a_i \sim N(0, \sigma^2_\mu)$，而 $e_{it} \sim N(0, \sigma^2_\nu)$ 彼此独立且随时间独立。

在此模型下，$y_i$ 服从正态分布，均值为 $m \mathbf{1}$，协方差矩阵为 $\sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I$，其中 $\mathbf{1}$ 是全1向量。

该协方差矩阵的逆是 $I$ 和 $\mathbf{1}\mathbf{1}^\top$ 的线性组合，所以随机效应模型是一个指数族。

它的充分统计量是均值 $\bar y_i$、其平方 $\bar y_i^2$，以及组内平方和 $W_i = \sum_t (y_{it} - \bar y_i)^2$。

按照 {doc}`approximation_errors` 的记号，$T \bar y_i^2$ 是工人收入在零频率处的周期图，而由帕塞瓦尔恒等式，$W_i$ 是其余 $T - 1$ 个傅里叶频率 $2\pi j / T$ 处周期图纵坐标之和。

该模型的收入分布是离散的，因此它与每个正态分布之间的KL散度都是无穷大，正如 {doc}`information_projections` 所指出的。

{prf:ref}`ip_thm_m_projection` 的证明可应用于交叉熵 $-E[\log q_\theta(y_i)]$ 中，当 $y_i$ 有密度时，它与KL散度相差 $y_i$ 的熵。

对于任何在每年均值相同、且协方差矩阵非奇异、非对角元素平均值非负的 $y_i$ 分布，使交叉熵最小化的参数会匹配充分统计量的期望，这设定 $m = E[\bar y_i]$ 以及

$$
E[W_i] = (T - 1) \sigma^2_\nu ,
\qquad
T \operatorname{Var}(\bar y_i) = T \sigma^2_\mu + \sigma^2_\nu .
$$

求解得到 $\sigma^2_\nu = E[W_i]/(T - 1)$ 和 $\sigma^2_\mu = \operatorname{Var}(\bar y_i) - \sigma^2_\nu / T$。

我们称这个极小化子为 $y_i$ 分布向随机效应模型的m-投影。

该投影将 $\sigma^2_\nu$ 设为非零傅里叶频率处预期周期图的平均值，而对于均值为零的 $y_i$（如同我们经年均化处理的面板），将 $T\sigma^2_\mu + \sigma^2_\nu$ 设为零频率处的预期周期图。

*对于平衡面板，戈特沙尔克和莫菲特的永久性和暂时性方差是工人收入分布向随机效应模型的m-投影的参数*：公式 {eq}`ti_gm` 就是上述表达式用男性样本代替总体，但均值的方差使用除数 $N - 1$。

无论随机效应模型是否描述了数据，该投影都是良定义的。

戈特沙尔克和莫菲特报告称，连续暂时性成分之间的相关性为0.14和0.18，并得出结论说这些成分"基本不相关"。

在随机效应模型下，连续偏差 $y_{it} - \bar y_i$ 之间的相关性为 $-1/(T - 1)$，当 $T = 9$ 时为 $-0.125$。

因此，戈特沙尔克和莫菲特的相关性比序列不相关暂时性成分所产生的相关性高约 $0.3$。

正态分布的交叉熵只通过其均值和协方差矩阵 $S$ 依赖于 $y_i$ 的分布，所以我们可以通过最小化 $N(0, S)$ 与随机效应模型之间的KL散度来计算该投影。

该投影的参数是 $S$ 的线性函数：

$$
\sigma^2_\nu = \frac{\operatorname{tr} S - \mathbf{1}^\top S \mathbf{1} / T}{T - 1} ,
\qquad
\sigma^2_\mu = \frac{\mathbf{1}^\top S \mathbf{1}}{T^2} - \frac{\sigma^2_\nu}{T} .
$$

{ref}`ti_ex_projection` 推导了这些公式。

因此，如果对数收入是两个独立成分之和，那么该和的投影就是两个投影之和。

随机效应模型的协方差矩阵及其逆矩阵位于同一线性空间中，该空间由 $I$ 和 $\mathbf{1}\mathbf{1}^\top$ 张成。

充分统计量是以该空间中的 $A$ 为矩阵的二次型 $y_i^\top A y_i$，所以匹配它们的期望使 $S - \Sigma$ 在迹内积意义下与该空间正交。

由于 $\Sigma$ 位于该空间中，m-投影与 $S$ 向该空间的最小二乘投影一致，后者将 $\sigma^2_\mu$ 设为 $S$ 的非对角元素的平均值，将 $\sigma^2_\mu + \sigma^2_\nu$ 设为对角元素的平均值。

协方差矩阵的线性空间本身并不能使两个投影一致：对称托普利茨矩阵（包含平稳过程的协方差矩阵）形成一个不包含其逆矩阵的线性空间，而两个投影在它上面的结果是不同的。

下面的代码针对一个人为面板，检验了这两种刻画方式以及周期图恒等式。

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

对于具有谱密度 $f$（按照 {doc}`approximation_errors` 的约定）的平稳过程，$S$ 中的公式给出

$$
\sigma^2_\mu = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) \, \frac{K_T(\omega) - 1}{T - 1} \, d\omega ,
\qquad
\sigma^2_\nu = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) \, \frac{T - K_T(\omega)}{T - 1} \, d\omega ,
$$ (ti_fejer)

其中 $K_T(\omega) = \sin^2(T\omega/2) / (T \sin^2(\omega/2))$ 是费耶核，即一列 $T$ 个1在频率 $\omega$ 处的周期图；在零频率处的预期周期图为 $\frac{1}{2\pi}\int_{-\pi}^{\pi} f(\omega) K_T(\omega) \, d\omega$。

$\sigma^2_\mu$ 中给予 $f$ 的权重在零频率处等于一，且积分为零，所以 $\sigma^2_\mu$ 衡量的是谱密度在零频率附近超出其均值的程度。

当 $T = 9$ 时，费耶核约有十分之九的权重位于频率 $|\omega| < 2\pi/T$ 处。

让我们对两个自回归过程，用 $S$ 中的公式来检验 {eq}`ti_fejer`。

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

对于负自相关过程，公式给出了负的 $\sigma^2_\mu$，因为其谱密度在零频率附近低于其均值；它向随机效应模型的m-投影位于边界 $\sigma^2_\mu = 0$ 上。

投影关于 $S$ 的线性性，使得即使模型和数据在水平上有差异，比较两个时间窗之间的变化仍是合理的。

模型中的工人在进入劳动力市场时是完全相同的，并且该模型没有测量误差，因此它的永久性和暂时性方差远低于戈特沙尔克和莫菲特的数值。

假设数据中的对数收入是模型所描述的一个成分，加上一个独立成分，且后者的协方差矩阵在两个窗口中相同。

那么数据投影从1970-78年到1979-87年的变化就等于模型所描述成分的投影变化。

汉森和萨金特在 {doc}`approximation_errors` 中的理想季节调整采用了相同的策略：它将省略数据某个特征的模型，仅在该被省略特征消失的地方与数据进行比较。

在那里，比较使用的是季节带之外的频率；这里使用的是时间窗之间的变化。

戈特沙尔克和莫菲特的证据对于暂时性方差而言，比对于永久性方差而言，更支持这一维持性假设。

测量误差会提高他们的暂时性方差，他们写道："问题在于这种偏差是否随时间变化。我们尚未知道有任何证据表明它发生了变化。"

他们写道："技能'价格'的上涨也许是永久性工资分散程度变化的一个令人满意的解释"，而柳奎斯特和萨金特的工人在事前是完全相同的，所以他们的模型中没有入职时固定、其价格可能上涨的教育或其他特征。

(ti_check_section)=
## 柳奎斯特和萨金特的检验

### 直方图

{cite:t}`LjungqvistSargent1998` 使用戈特沙尔克和莫菲特的证据"一方面作为启发，另一方面作为对我们模型的检验"。

他们的检验比较了男性均值 $\bar y_i$ 及其围绕该均值标准差的直方图。

下图将戈特沙尔克和莫菲特的直方图（我们大致从他们的图2和图4中读取）绘制在我们人为面板在动荡参数为 $0.02$ 和 $0.04$ 下的直方图之上。

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

{cite:t}`LjungqvistSargent1998` 的图14绘制了他们自己面板的相同直方图。

我们大致读取了其柱状条的高度，并与我们自己的结果进行了比较。

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

我们的直方图在每个柱状条上与他们的结果相符，误差在约一个百分点以内。

将动荡参数从 $0.02$ 提高到 $0.04$ 使两个分布都朝着戈特沙尔克和莫菲特发现的方向扩散。

柳奎斯特和萨金特指出，模型中均值分布所跨越的范围比数据中的要小，因为他们的工人在事前是完全相同的，而且模型标准差的增加发生在比数据更低的标准差水平上。

### 分布移动的位置

柳奎斯特和萨金特的第二个评论涉及两个时期之间概率质量移动的位置，而不仅仅是分布扩散的程度。

最优传输描述了这种移动。

对于实数轴上的分布，以及一个作为移动距离凸函数的移动代价，将一个分布转化为另一个分布最便宜的方法是配对分位数，正如 {doc}`exponential_families` 中 {ref}`ef_wasserstein_section` 所述：第一个分布 $u$ 分位数处的质量移动到第二个分布 $u$ 分位数处。

**偏移量** $s(u) = G^{-1}(u) - F^{-1}(u)$（其中 $F$ 和 $G$ 分别是两个时期的累积分布函数）记录了该质量移动的距离。

该偏移量比较的是两个分布，而不是同一批男性在两个时期的情况，所以它描述的是分位数如何移动，而不是任何一个男性的收入如何变化。

我们计算男性标准差分布的偏移量，对于戈特沙尔克和莫菲特的直方图，将每个柱状条中的男性在其内部均匀分布，并对动荡参数为 $0.02$ 和 $0.04$ 的人为面板也作同样计算。

```{code-cell} ipython3
def histogram_quantiles(percents, centers, u):
    """Quantiles of a histogram whose mass is spread uniformly within each bar."""
    width = centers[1] - centers[0]
    edges = np.append(centers - width / 2, centers[-1] + width / 2)
    cdf = np.concatenate([[0.0], np.cumsum(percents) / np.sum(percents)])
    return np.interp(u, cdf, edges)


u_grid = np.linspace(0.0005, 0.9995, 1000)
shift_data = (histogram_quantiles(gm_fig4["1979-87"], gm_fig4_bins, u_grid)
              - histogram_quantiles(gm_fig4["1970-78"], gm_fig4_bins, u_grid))
sd02, sd04 = y02.std(axis=1, ddof=1), y04.std(axis=1, ddof=1)
shift_model = np.quantile(sd04, u_grid) - np.quantile(sd02, u_grid)

print(" quantile   PSID shift   model shift")
for u in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
    k = np.argmin(np.abs(u_grid - u))
    print(f"   {u:4.2f}      {shift_data[k]:+.3f}       {shift_model[k]:+.3f}")
print(f"smallest shift: PSID {shift_data.min():+.4f}, model {shift_model.min():+.4f}")
print(f"average shift: PSID {shift_data.mean():.4f}, model {shift_model.mean():.4f}")
print(f"men's averages in the model at turbulence 0.02 range from "
      f"{y02.mean(axis=1).min():.2f} to {y02.mean(axis=1).max():.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Quantile shifts of men's standard deviations
    name: fig-ti-shifts
---
keep = u_grid <= 0.95
fig, ax = plt.subplots()
ax.plot(u_grid[keep], shift_data[keep], lw=2, label="PSID, 1970–78 to 1979–87")
ax.plot(u_grid[keep], shift_model[keep], lw=2, label="model, turbulence 0.02 to 0.04")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("quantile $u$")
ax.set_ylabel("shift $s(u)$")
ax.legend()
plt.show()
```

该图在第95百分位处停止，因为超过该点之后，PSID的分位数取决于我们如何在最后一些数据稀疏的柱状条中分布男性。

在PSID中，偏移量随分位数增加而增大，从第10百分位的 $0.004$ 增至第90百分位的 $0.126$：标准差分布的上分位数上升幅度远大于下分位数（下分位数变化不大）。

在模型中，第10至第95百分位处的偏移量大致都在 $0.01$ 到 $0.02$ 之间：动荡使标准差分布在每个分位数上提升大致相同的幅度。

在约第37百分位以下，模型的偏移量超过了数据的偏移量，而在第90和第95百分位处，数据的偏移量超过模型偏移量的八倍以上。

这就是柳奎斯特和萨金特评论的一个量化版本。

在各分位数上取平均，该偏移量等于平均标准差的变化量：PSID中为 $0.043$，模型中为 $0.014$。

由于PSID的偏移量处处非负，$0.043$ 也是两个时期PSID分布之间的1-Wasserstein距离；偏移量的形状说明了质量移动的位置。

通过KL散度比较这些直方图在这里会失效。

在我们动荡参数为 $0.02$ 的面板中，男性的均值范围从 $-0.84$ 到 $0.29$，而戈特沙尔克和莫菲特1970-78年的直方图显示约2%的男性低于 $-1.1$，约5%高于 $0.6$。

因此，数据的均值分布与模型的均值分布之间的KL散度是无穷大，而传输距离则保持有限。

戈特沙尔克和莫菲特的分解通过比较二阶矩（即向随机效应模型投影的参数）避开了这个问题。

### 约束函数

动荡参数为 $v$ 的模型收入的戈特沙尔克-莫菲特分解是一对数字 $b(v) = (\sigma^2_\mu(v), \sigma^2_\nu(v))$，即模型收入分布向随机效应模型的m-投影的参数。

{cite:t}`GourierouxMonfortRenault1993` 称映射 $v \mapsto b(v)$ 为**约束函数**；{doc}`gallant_tauchen` 将其写作 $\theta^0(\rho)$。

柳奎斯特和萨金特在动荡参数为 $0.02$ 和 $0.04$ 时，计算了 $\bar y_i$ 及男性标准差的分布（其二阶矩确定了 $b(0.02)$ 和 $b(0.04)$），并将它们与1970-78年和1979-87年数据中的分布进行了比较。

正如 {doc}`gallant_tauchen` 所描述的，间接推断选择能使模型投影与数据投影相一致的参数值。

我们将变化 $b(v) - b(0.02)$ 与戈特沙尔克和莫菲特测得的变化进行比较，因为柳奎斯特和萨金特将动荡参数 $0.02$ 赋给1970-78年，而且上述原因使得两个投影在水平上存在差异。

我们通过在每个动荡参数网格点上模拟10,000个工人的面板（在每一点使用相同的随机数）来估计约束函数。

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

在柳奎斯特和萨金特赋予1979-87年的动荡参数 $0.04$ 下，模型给出了约8%的永久性方差测量增量和13%的暂时性方差测量增量。

在不同随机种子下，第二个份额范围约为9%到13%，平均约为11%。

约束函数在动荡参数高于 $0.01$ 时随动荡增加而上升，并随动荡参数增大而趋于平缓，因为技能损失的分布会逐渐趋近于从最低技能等级到工人当前等级之间的均匀分布。

在我们网格中的最大值处，模型给出了约21%的永久性方差增量和43%的暂时性方差增量。

在技能损失均匀分布的极限情况下，这两个份额约为21%和44%；在不同随机种子下，第二个份额范围约为44%到49%。

戈特沙尔克和莫菲特对数据进行了截尾处理，而我们没有对模型的面板进行截尾。

对模型面板中每年的对数收入以相同方式进行截尾，在动荡参数为 $0.04$ 时两个份额基本保持不变，而在最大动荡参数处两个份额都会降低。

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

模型每年收入的1%尾部约占模型暂时性方差的三分之一，而截尾处理使最大动荡参数处的暂时性份额降至不同随机种子下34%到37%之间。

这些份额还取决于技能的范围（它限定了技能损失的大小）以及技能转变的速度（它决定了损失持续的时间）。

{ref}`ti_ex_robust` 表明，若技能取值范围为 $[1, 3]$ 并保持原来的转移概率，动荡可以解释约46%的永久性方差测量增量和几乎全部的暂时性方差增量。

若同时将转移概率也加倍，永久性份额又回落到约五分之一，因为被解雇的工人恢复技能的速度加快了一倍。

将1970-78年参数固定在 $0.02$ 这一事实也很重要。

如果两个时间窗的动荡参数都可以自由取值，模型所能给出的最大变化就是约束函数最大值点与最小值点之差。

```{code-cell} ipython3
points = np.vstack([binding, b_uniform])
largest = (points.max(axis=0) - points.min(axis=0)) / gm_change
print(f"largest shares with both windows free: {largest[0]:.0%}, {largest[1]:.0%}")
```

没有任何动荡参数的取值能使模型投影的变化等于数据投影的变化。

基于这两个统计量的间接推断将动荡参数推向技能损失均匀分布的极限情形，而仍然留有一个缺口。

柳奎斯特和萨金特观察到，他们模型中变异性增加发生在比数据更低的标准差水平上；约束函数衡量了两个方差增量大小之间的一个相关不足之处。

他们从与戈特沙尔克和莫菲特、以及雅各布森、拉隆德和沙利文的比较中得出结论：

> 这些模拟表明，我们模型中产生高长期失业率的机制，在经济动荡水平远低于美国所观察到的水平时就已经发挥作用。

我们对约束函数的估计支持了他们的推断，即数据所需的动荡程度超过 $0.04$。

这些估计还表明，无论这一模型中的动荡程度有多高，都无法达到数据中的变化幅度，因此这些变化也必定来自模型所省略的成分。

将1970-78年参数固定在 $0.02$ 时，这两个统计量和一个自由参数留下了一个过度识别约束。

{doc}`gallant_tauchen` 以这种方式检验一个结构模型：询问在最佳参数值下，投影之间剩余的距离是否超出了抽样变异所允许的范围。

这里的检验是一个联合假设：即该模型能描述收入的变化，并且该模型所省略的收入成分没有发生变化。

戈特沙尔克和莫菲特关于技能价格上涨可能解释永久性工资分散程度扩大的说法，使得第二部分假设在永久性方差方面的可信度受到了质疑。

(ti_ident_section)=
## 解雇率与动荡

约束函数将一个标量参数映射为两个数字，而其他结构性变化也可能使这两个数字朝同一方向移动。

我们计算了在动荡参数固定为 $0.02$ 并提高解雇率 $\lambda$ 情况下模型的投影。

```{code-cell} ipython3
b04 = binding[v_grid == 0.04][0]
b_layoff = {λ: gm_decomposition(artificial_panel(0.02, λ=λ)[0])
            for λ in (0.012, 0.015, 0.018, 0.021)}
print("layoff rate   change in σ²_μ, σ²_ν from the 1970–78 calibration")
for λ, b in b_layoff.items():
    print(f"{λ:11.3f}    {b[0] - b02[0]:+.4f}  {b[1] - b02[1]:+.4f}")
print(f"turbulence 0.04:  {b04[0] - b02[0]:+.4f}  {b04[1] - b02[1]:+.4f}")
```

将解雇率加倍或略多于加倍，能使永久性方差增加的幅度与将动荡参数从 $0.02$ 提高到 $0.04$ 所产生的增幅大致相同，而它使暂时性方差在这个模拟中约增加 $0.003$（而非 $0.006$），在不同随机种子平均下约增加 $0.002$（而非 $0.005$）。

这两种解释的投影位于随机效应模型中相邻的点上。

这两种解释投影相近，正是 {doc}`specification_tests` 中结果在识别意义上的对应物，该结果表明，如果备择假设使两个投影保持相等，那么投影的比较将不具有检验力。

一个样本能否区分这两种解释，取决于数据中公式 {eq}`ti_gm` 的抽样变异性。

戈特沙尔克和莫菲特的样本包含2,730名男性。

暂时性方差 $\sigma^2_\nu$ 是男性平方标准差 $\hat s_i^2$ 的平均值，所以在2,730名男性的样本中，它的标准差大约是 $\hat s_i^2$ 的标准差除以 $\sqrt{2730}$。

我们从对戈特沙尔克和莫菲特图4的读取中计算它，并计算一个检验的功效，该检验比较了暂时性方差的变化与每种解释所预测的变化。

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

我们对这张图的读取结果显示暂时性方差分别为 $0.103$ 和 $0.145$，接近戈特沙尔克和莫菲特表1中的 $0.104$ 和 $0.148$。

两个时间窗之间变化的标准差约为 $0.006$，所以两种解释的暂时性方差变化之差约为 $0.003$，约为半个标准差。

这一计算将两个时间窗视为两个独立的2,730名男性样本；戈特沙尔克和莫菲特的两个时间窗包含一些相同的男性，这会降低该变化的标准差，而每个时间窗所含男性人数少于2,730人，这又会提高该标准差。

利用 {ref}`st_ex_power` 的非中心性计算，半个标准差的分离度使一个5%水平的检验的功效约为 $0.08$。

该计算所问的是，由其中一种解释生成的数据能否拒绝另一种解释。

这两种解释都比测得的变化量 $0.044$ 低六个标准差以上，所以该计算所衡量的是这两个统计量在该模型内能够识别什么，而不是任一解释的拟合优度如何。

因此，戈特沙尔克和莫菲特的这两个统计量几乎无法区分更频繁的解雇与更大的技能损失。

{cite:t}`LjungqvistSargent2008` 还将他们的模型与 {cite:t}`JacobsonLaLondeSullivan1993` 测得的被解雇工人的收入损失进行了比较。他们写道：

> 我们承认，1980年代以来美国被解雇工人所经历的巨大收入损失本身并不能说明1950-60年代与1980年代之后之间动荡的增加，因为这需要来自1950-60年代类似被解雇工人研究的证据，而遗憾的是这样的研究并不存在。

他们补充道："相反，我们从像戈特沙尔克和莫菲特（1994）这样的时间序列研究中间接获得了对我们关于动荡增加这一说法的支持，而卡茨和奥托尔（1999）表明他们的研究结果在各种研究和数据集中都具有稳健性。"

每增加一个统计量，就会为辅助模型增加充分统计量，并缩小与数据投影一致的结构参数集合。

只有当两种解释在某个附加统计量上的投影不同，并且计量经济学家以相同方式从模型和数据中计算该统计量时，附加统计量才能区分这两种解释。

{ref}`ti_ex_jls` 为这两种解释计算了被解雇工人的统计量。

(ti_mg_section)=
## 协方差结构模型

{cite:t}`MoffittGottschalk2011` 描述了分解 {eq}`ti_gm` 的一个局限性：

> 然而，除了需要选择任意的日历时间区间之外，如果永久性方差并非在区间端点处骤然变化，而是随时间平滑地上升趋势，这种方法就存在问题。在那种情况下，区间早期和晚期阶段相对于平均收入的偏差会被错误地归因于暂时性收入。

用周期图的语言来说，在某个窗口内发生变化的永久性成分，其功率分布在非零傅里叶频率上，而该投影会将其算作暂时性成分。

他们转而拟合了一个描述收入整体协方差结构的模型，其中永久性成分服从随机游走，暂时性成分服从ARMA(1,1)过程，

$$
y_{it} = \mu_{it} + \nu_{it} ,
\qquad
\mu_{it} = \mu_{i,t-1} + \omega_{it} ,
\qquad
\nu_{it} = \rho \nu_{i,t-1} + \xi_{it} + \theta \xi_{i,t-1} ,
$$

其载荷和方差随时间呈趋势变化。

当 $\rho = \theta = 0$ 时，暂时性成分是白噪声，该模型即为 {cite:t}`Muth1960` 的带噪声观测的随机游走模型，{doc}`phillips_misspecified` 和 {doc}`lq_permanent_income` 中使用了该模型。

他们通过最小距离法估计该模型，使用单位权重矩阵，这通过最小二乘法将模型的自协方差拟合到样本自协方差。

他们报告的卡方统计量对每一个设定都超过了1%的临界值，不过他们判断带有ARMA(1,1)暂时性成分的随机游走模型"能较好地拟合我们的数据"。

我们将一个参数恒定的版本拟合到我们人为面板的九年协方差矩阵，既用最小二乘法（如莫菲特和戈特沙尔克所做的那样），也用最小化高斯KL散度的方法。

它的参数包括：窗口前一年永久性成分的方差 $\sigma^2_0$、其增量的方差 $\sigma^2_\omega$，以及 $\rho$、$\theta$ 和暂时性成分的创新项 $\xi_{it}$ 的方差 $\sigma^2_\xi$。

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

协方差结构模型给永久性成分分配的方差几乎为零。

它将模型的收入不稳定性归因于一个暂时性成分，其年度自回归系数介于 $0.63$ 和 $0.71$ 之间，移动平均系数约为 $-0.26$。

莫菲特和戈特沙尔克在他们的表5中给出的PSID估计值为 $\rho = 0.641$ 和 $\theta = -0.367$。

协方差结构模型将动荡从 $0.02$ 到 $0.04$ 所带来的增量归因于暂时性成分。

相同面板的随机效应投影 {eq}`ti_gm` 将增量中的约一半归因于永久性成分。

这两个辅助模型之所以意见不一致，是因为解雇时的技能损失规模大、持续时间长，但最终会逆转。

一个失去一半累积技能的工人，以每10个就业期（即 $0.05$）一个技能等级的平均速度重新获得技能，所以该工人的收入在数年内保持较低水平。

这种在九年窗口早期发生的损失会降低工人的均值，并提高工人间均值的方差，这被随机效应投影算作永久性的。

从周期图的角度看，持续数年的损失将其大部分功率置于零频率处。

模型收入的协方差随滞后期增大而递减，协方差结构模型用其暂时性成分拟合这种递减，所以它将同样的损失算作暂时性的。

莫菲特和戈特沙尔克也注意到了这一问题："如果暂时性成分存在序列相关，平均收入的方差将同时受到暂时性成分方差的影响。"

模型中工人围绕其自身均值的偏差具有较小的序列相关性，与戈特沙尔克和莫菲特在PSID中发现的情形类似。

```{code-cell} ipython3
for v, y in ((0.02, y02), (0.04, y04)):
    d = y - y.mean(axis=1, keepdims=True)
    r = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    print(f"turbulence {v}: correlation of successive deviations {r:.2f}")
```

$0.17$ 和 $0.22$ 的连续偏差相关性与一个自回归系数接近 $0.7$ 的暂时性成分相容，所以PSID中较小的相关性并不能表明其暂时性成分接近序列不相关。

最小二乘投影和KL投影在协方差结构模型上的细节有所不同。

该模型不是一个指数族：其协方差矩阵构成一个弯曲的集合，使两个投影在随机效应模型上一致的论证在此不适用。

## 结语

柳奎斯特和萨金特对直方图的比较是间接推断的一个非正式版本：他们直方图的二阶矩，就是数据和结构模型向同一辅助模型m-投影的参数。

对于 {cite:t}`LjungqvistSargent1998` 的自由放任经济，戈特沙尔克-莫菲特分解的约束函数在动荡参数高于 $0.01$ 时随动荡增加而上升。

在将1970-78年参数固定在他们所取的值 $0.02$ 的情况下，约束函数无法达到戈特沙尔克和莫菲特测得的变化幅度，而它能达到的程度取决于技能的范围以及被解雇工人恢复技能的速度。

解雇率的提高会使同样的两个统计量以类似方式作出反应，所以它们对动荡的识别能力较弱。

多少不稳定性是永久性的，这一问题的答案取决于所用的辅助模型：随机效应分解和 {cite:t}`MoffittGottschalk2011` 的协方差结构模型，对同一个由模型生成的不稳定性，给出了不同的拆分方式。

## 练习

```{exercise}
:label: ti_ex_jls

{cite:t}`JacobsonLaLondeSullivan1993` 比较了在1982年第一季度被解雇、已有六年工龄的工人与一直留任至1986年的工人的季度收入。

{cite:t}`LjungqvistSargent1998` 通过追踪100,000名已在劳动力市场工作20年的工人11年，构建了模型中的对应情形。他们将第7年的前12周称为1982年第一季度，并绘制了技能损失超过40%的离职者。

1. 在动荡参数为 $0.04$ 的自由放任经济中模拟100,000名这样的工人。确定留任者——他们在每期都有工作且从未被解雇。确定离职者——他们在前六年每期都有工作且从未被解雇，而在第7年的前六期被解雇。
2. 计算每组工人的平均12周收入，以及离职者与留任者在最后两年收入的比率。对所有离职者，以及解雇时技能下降超过40%的离职者分别做此计算。
3. 在动荡参数为 $0.02$、解雇率为 $0.019$（该解雇率使永久性方差的变化大致匹配将动荡提高到 $0.04$ 所带来的变化）的经济中，对所有离职者重复第2部分的计算，并计算每个经济中在第7年初被解雇的、拥有六年工龄的工人的比例。
4. 这些统计量中，哪些是计量经济学家能够从雅各布森、拉隆德和沙利文那样的数据中计算出来的？其中哪些能够区分这两种解释？
```

```{solution-start} ti_ex_jls
:class: dropdown
```

下面是一种解答。

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

柳奎斯特和萨金特报告，在他们的100,000名工人中，有2,889名留任者和629名离职者。

我们的模拟中，留任者人数约为其两倍，离职者人数约多出三分之二。

技能损失超过10%、20%、30%和40%的离职者份额分别为55%、27%、10%和2.5%，而他们报告的数字为47%、25%、11%和3%。

我们尚未找出这一差异的来源。

在动荡参数为 $0.04$ 时，所有离职者在被解雇后三到五年的收入比留任者低约17%，而技能下降超过40%的少数离职者收入低约37%。

柳奎斯特和萨金特将雅各布森、拉隆德和沙利文的发现描述为一种恢复趋势，"最终达到比解雇前工作所挣收入低约25%的水平"。

他们的数据通过解雇这一事件而非技能损失的大小来识别被解雇的工人，所以数据所能提供的收入统计量是针对所有离职者的那个。

他们的记录同时涵盖留任者和离职者，所以像他们这样的数据也能提供一个离职率。

在解雇率更高的情形下，所有离职者的收入比留任者低约21%，因为被解雇后，他们再次被解雇的频率约为动荡经济中离职者的两倍。

因此，收入损失统计量几乎无法区分这两种解释。

在解雇率更高的经济中，离职率约为动荡经济中离职率的两倍。

单期测量的离职率只有在我们信任模型所给出的离职水平时，才能区分这两种解释，而本讲座自始至终比较的都是变化而非水平。

两个时期之间离职率的变化能够区分这两种解释：当解雇率翻倍时它也翻倍，而动荡增加时它不会发生变化。

测量这一变化需要来自两个时期的可比被解雇工人研究，而上文引述自 {cite:t}`LjungqvistSargent2008` 的段落指出，这正是所缺失的。

```{solution-end}
```

```{exercise}
:label: ti_ex_projection

设 $y_i$ 为 $\mathbb{R}^T$ 中均值为零的随机向量，其协方差矩阵 $S$ 正定且非对角元素的平均值非负，并考虑随机效应族 $N(0, \sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I)$。

1. 证明 $\Sigma = \sigma^2_\mu \mathbf{1}\mathbf{1}^\top + \sigma^2_\nu I$ 的逆为 $\sigma_\nu^{-2} \left(I - c\,\mathbf{1}\mathbf{1}^\top\right)$（其中 $c = \sigma^2_\mu / (\sigma^2_\nu + T \sigma^2_\mu)$），并证明 $\log p(y_i)$ 是 $W_i$ 和 $T \bar y_i^2$ 的线性函数加上一个仅依赖于参数的函数。
2. 对 $(\sigma^2_\mu, \sigma^2_\nu)$ 最小化 $N(0, S)$ 与 $N(0, \Sigma)$ 之间的高斯KL散度，证明极小化子给出 $\sigma^2_\nu = (\operatorname{tr} S - \mathbf{1}^\top S \mathbf{1}/T)/(T - 1)$ 和 $\sigma^2_\mu = \mathbf{1}^\top S \mathbf{1}/T^2 - \sigma^2_\nu / T$。
3. 证明 $S$ 向同一族的最小二乘投影给出相同的答案。
```

```{solution-start} ti_ex_projection
:class: dropdown
```

下面是一种解答。

对于第1部分，将 $\Sigma$ 乘以 $\sigma_\nu^{-2}(I - c\,\mathbf{1}\mathbf{1}^\top)$，并利用 $\mathbf{1}^\top \mathbf{1} = T$：乘积为 $I + \sigma_\nu^{-2}\left[\sigma^2_\mu - c(\sigma^2_\nu + T\sigma^2_\mu)\right]\mathbf{1}\mathbf{1}^\top = I$。

于是 $y_i^\top \Sigma^{-1} y_i = \sigma_\nu^{-2}\left(y_i^\top y_i - c T^2 \bar y_i^2\right)$，而 $y_i^\top y_i = W_i + T \bar y_i^2$，所以

$$
\log p(y_i) = -\frac{W_i}{2\sigma^2_\nu} - \frac{T \bar y_i^2}{2(\sigma^2_\nu + T\sigma^2_\mu)} - \frac{1}{2}\log\det(2\pi\Sigma) ,
$$

这是因为 $1 - cT = \sigma^2_\nu / (\sigma^2_\nu + T\sigma^2_\mu)$。

对于第2部分，$N(0, S)$ 与 $N(0, \Sigma)$ 之间的KL散度，是 $N(0, S)$ 下 $\log p(y_i)$ 期望的负值再加一个常数。

$\Sigma$ 的特征值为 $\sigma^2_\nu + T\sigma^2_\mu$（对应特征向量 $\mathbf{1}$），以及 $\sigma^2_\nu$（重数为 $T - 1$），所以 $\log\det\Sigma = \log(\sigma^2_\nu + T\sigma^2_\mu) + (T - 1)\log\sigma^2_\nu$。

利用 $E[W_i] = \operatorname{tr} S - \mathbf{1}^\top S \mathbf{1}/T$ 和 $E[T\bar y_i^2] = \mathbf{1}^\top S \mathbf{1}/T$，目标函数可分解为关于 $a = \sigma^2_\nu$ 的函数和关于 $d = \sigma^2_\nu + T\sigma^2_\mu$ 的函数：

$$
\frac{E[W_i]}{2a} + \frac{T-1}{2}\log a + \frac{E[T\bar y_i^2]}{2d} + \frac{1}{2}\log d .
$$

最小化给出 $a = E[W_i]/(T-1)$ 和 $d = E[T\bar y_i^2]$，这就是所述的公式。

对于第3部分，最小二乘问题 $\min \sum_{t \le s} (S_{ts} - \Sigma_{ts})^2$ 分解为对角元素（全部等于 $\sigma^2_\mu + \sigma^2_\nu$）和非对角元素（全部等于 $\sigma^2_\mu$）。

极小化子将 $\sigma^2_\mu + \sigma^2_\nu$ 设为对角元素的平均值 $\bar s_d$，将 $\sigma^2_\mu$ 设为非对角元素的平均值 $\bar s_o$。

由于 $\operatorname{tr} S = T \bar s_d$ 以及 $\mathbf{1}^\top S \mathbf{1} = T\bar s_d + T(T - 1)\bar s_o$，第2部分的公式给出 $\sigma^2_\nu = \bar s_d - \bar s_o$ 和 $\sigma^2_\mu = \bar s_o$。

所以 $\sigma^2_\mu \geq 0$ 当且仅当 $S$ 的非对角元素平均值非负，这正是本练习假设该条件的原因。

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

{cite:t}`LjungqvistSargent1998` 报告称："在我们之前的模拟中获得了类似的结果，在那些模拟中，我们将技能范围加倍，并将连续两期保持就业时获得一个技能等级的转移概率，以及连续两期失业时失去一个技能等级的转移概率也加倍。"

1. 将技能设定在 $[1, 3]$ 上，保持21个技能等级和原始的转移概率，并在动荡参数 $0.02$、$0.04$、$0.1$ 和 $1$ 下重新计算约束函数。
2. 同时将技能提升的概率加倍至 $0.2$，将技能损失的概率加倍至 $0.4$，并重新计算约束函数。

第2部分是我们对他们那句话的解读；第1部分将范围的效应与转移概率的效应分开。

在这两种情况下，动荡能够解释永久性方差和暂时性方差测量变化的多大份额？为什么这两种情况会有所不同？
```

```{solution-start} ti_ex_robust
:class: dropdown
```

下面是一种解答。

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

更宽的技能范围允许出现更大的技能损失。

在技能取值范围为 $[1, 3]$ 并保持原始转移概率的情况下，动荡可以解释约46%的永久性方差测量增量，以及几乎全部的暂时性方差增量。

同时将转移概率也加倍，会使被解雇工人恢复所失技能等级所需时间减半。

此时技能损失在九年窗口中所占的时间比例更小，对工人均值的拉低程度也更小，所以随机效应投影将其中更少的部分算作永久性的。

在这种情况下，动荡所能解释的永久性方差增量至多约为五分之一（与原经济中一样），而暂时性方差增量略超过100%。

更快的恢复才是关键所在：仅将技能提升概率加倍，会使动荡参数为 $1$ 时的永久性份额降至约21%，而仅将技能损失概率加倍，则该份额保持在约41%。

柳奎斯特和萨金特那句话涉及的是他们的定性发现；而约束函数表明，仅改变技能范围时，永久性份额并不具有稳健性。

```{solution-end}
```