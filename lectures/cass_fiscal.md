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

# 带扭曲性税收的Cass-Koopmans模型

## 概述

本讲座研究了在非随机版本的 Cass-Koopmans 增长模型下，预期的财政与技术冲击对竞争均衡价格和数量的影响。该模型的特征在QuantEcon讲座{doc}`cass_koopmans_2`中有所介绍。

该模型在{cite}`Ljungqvist2012`的第11章中有更详细的讨论。

我们将此模型作为一个实验室，用来尝试近似均衡的数值方法，并展示动态模型的结构，在这些模型中，决策者对未来政府的决策拥有完美预期。

遵循Robert E. Hall的经典论文{cite}`hall1971dynamic`,我们在Cass-Koopmans最优增长模型的非随机版本基础上,增加了一个政府部门。该政府购买一系列商品，并通过一系列扭曲的比例税来为其支出融资。

扭曲性税收使竞争均衡配置无法解决规划问题。

因此,为了计算均衡配置和价格体系,我们需要解一个非线性差分方程组。该方程组由决策者的一阶条件和其他均衡条件组成。

我们提出两种近似均衡的方法:

- 第一种是射击算法，类似于我们在{doc}`cass_koopmans_2`中使用的。

- 第二种方法是求根算法，该算法最小化消费者与代表性企业一阶条件残差。



(cs_fs_model)=
## 经济模型


### 技术

可行配置满足

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

其中

- $g_t$ 是t时期的政府购买，
- $x_t$ 是总投资，
- $F(k_t, n_t)$ 是一个线性齐次的生产函数，其中资本$k_t$和劳动$n_t$具有正的且递减的边际产出。

物质资本的演化规律为

$$
k_{t+1} = (1 - \delta)k_t + x_t,
$$

其中 $\delta \in (0, 1)$ 是折旧率。

有时，将 $x_t$ 从{eq}`eq:tech_capital`中消除会更方便，可将其表示为

$$
g_t + c_t + k_{t+1} \leq F(k_t, n_t) + (1 - \delta)k_t.
$$

### 竞争均衡的组成部分

所有交易都发生在0时期。

代表性家庭拥有资本，做出投资决策，并将资本和劳动出租给代表性生产企业。

代表性企业使用资本和劳动生产商品，生产函数为 $F(k_t, n_t)$。

**价格体系**是一个三元序列 $\{q_t, \eta_t, w_t\}_{t=0}^\infty$，其中

- $q_t$ 是在 $0$ 时期下一单位$t$ 时期的投资或消费（$x_t$ 或 $c_t$）的税前价格，
- $\eta_t$ 是家庭在 $t$ 时期从企业租赁资本所获得的税前价格，
- $w_t$ 是家庭在 $t$ 时期向企业出租劳动力所获得的税前价格。

价格 $w_t$ 和 $\eta_t$ 是以 $t$ 时期的商品为单位表示的，而 $q_t$ 则以 $0$ 时期的计价物计价，这与{doc}`cass_koopmans_2`中一致。

政府的存在使得本讲座区别于{doc}`cass_koopmans_2`。

$t$ 时期的政府购买为 $g_t \geq 0$。

政府支出计划是一个序列 $g = \{g_t\}_{t=0}^\infty$。

政府税收计划是一个四元序列 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$，其中：

- $\tau_{ct}$ 是 $t$ 时期的消费税率，
- $\tau_{kt}$ 是 $t$ 时期的资本租赁税率，
- $\tau_{nt}$ 是 $t$ 时期的工资税率，
- $\tau_{ht}$ 是 $t$ 时期对消费者的一次性总额税。

由于可以征收一次性总额税 $\tau_{ht}$，政府实际上不应使用任何扭曲性税收。

尽管如此，我们仍然包含所有这些税收，因为像 {cite}`hall1971dynamic` 一样，它们让我们能够分析各种税收如何扭曲生产和消费决策。

在[实验部分](cf:experiments)，我们将看到政府税收计划的变化如何影响转型路径和均衡。


### 代表性家庭

代表性家庭对单一消费品 $c_t$ 和闲暇 $1-n_t$ 的非负序列具有偏好，其偏好由下式给出：

$$
\sum_{t=0}^{\infty} \beta^t U(c_t, 1-n_t), \quad \beta \in (0, 1),
$$ (eq:utility)

其中
- $U$ 对 $c_t$ 严格递增，二次连续可微，并在 $c_t \geq 0$ 且 $n_t \in [0, 1]$ 时严格凹。

代表性家庭在以下单一预算约束下最大化{eq}`eq:utility`：

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{投资时无税}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \eta_t k_t - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{租金收益税}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)

这里我们假设政府从资本租赁收入 $\eta_t k_t$ 扣除折旧补贴 $\delta k_t$，因此只对 $\tau_{kt} (\eta_t - \delta) k_t$ 征收资本租赁税。

### 政府
政府支出计划 $\{ g_t \}_{t=0}^\infty$ 和税收 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ 必须满足以下预算约束

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)

在给定一个预算可行的政府政策 $\{g_t\}_{t=0}^\infty$ 和 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ （满足式{eq}`eq:gov_budget`）的条件下：

- *家庭*选择 $\{c_t\}_{t=0}^\infty$、$\{n_t\}_{t=0}^\infty$ 和 $\{k_{t+1}\}_{t=0}^\infty$，在预算约束{eq}`eq:house_budget`下最大化效用函数{eq}`eq:utility`，
- *企业*选择 $\{k_t\}_{t=0}^\infty$ 和 $\{n_t\}_{t=0}^\infty$ 以最大化利润

    $$
         \sum_{t=0}^\infty q_t [F(k_t, n_t) - \eta_t k_t - w_t n_t]
    $$ (eq:firm_profit)
- **可行配置**是满足可行性条件{eq}`eq:tech_capital`的序列$\{c_t, x_t, n_t, k_t\}_{t=0}^\infty$。

## 均衡

```{prf:definition}
:label: com_eq_tax

**带扭曲性税收的竞争均衡**是一个**预算可行的政府政策**、一个**可行配置**和一个**价格体系**的组合。在给定价格体系和政府政策的情况下，该配置同时解决家庭问题和企业问题。
```

### 无套利条件

无套利论证意味着对跨期的价格和税率有一个限制条件。

通过重新排列{eq}`eq:house_budget`，并将同一时期的$k_t$项组合在一起，我们可以得到

$$
    \begin{aligned}
    \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t \right] &\leq \sum_{t=0}^\infty q_t(1 - \tau_{nt})w_t n_t - \sum_{t=0}^\infty q_t \tau_{ht} \\
    &+ \sum_{t=1}^\infty\left\{ \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]q_t - q_{t-1}\right\}k_t \\
&+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

家庭继承了一个给定的$k_0$，并将其作为初始条件，同时可以自由选择 $\{ c_t, n_t, k_{t+1} \}_{t=0}^\infty$。

由于资源有限，家庭的预算约束{eq}`eq:house_budget`在均衡状态下必须是有界的。

这对价格和税收序列施加了限制。

具体来说，对于 $t \geq 1$，与 $k_t$ 相乘的项必须等于零。

如果这些项严格为正（负），家庭就可以通过选择一个任意大的正（负）$k_t$ 来任意增加（减少）{eq}`eq:house_budget`的右侧，从而导致无限利润或套利机会：

- 如果这些项严格为正，家庭可以购买大量资本存量 $k_t$，并从资本的租赁服务和未折旧价值中获利。
- 如果这些项严格为负，家庭可以通过“卖空”合成单位资本来获利。两种情况都会导致{eq}`eq:house_budget`无界。

因此，通过令与 $k_t$ 相乘的项设为 $0$，我们得到无套利条件：

$$
\frac{q_t}{q_{t+1}} = \left[(1 - \tau_{kt+1})(\eta_{t+1} - \delta) + 1\right].
$$ (eq:no_arb)

此外，我们有终端条件：

$$
-\lim_{T \to \infty} q_T k_{T+1} = 0.
$$ (eq:terminal)

代表性企业的零利润条件对均衡价格和数量施加了额外的限制。

企业利润的现值为：

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right].
$$

将线性齐次函数的欧拉定理应用于 $F(k, n)$，企业利润的现值为：

$$
\sum_{t=0}^\infty q_t \left[ (F_{kt} - \eta_t)k_t + (F_{nt} - w_t)n_t \right].
$$

无套利（或零利润）条件为：

$$
\eta_t = F_{kt}, \quad w_t = F_{nt}.
$$(eq:no_arb_firms)

## 家庭的一阶条件

家庭在{eq}`eq:house_budget`约束下最大化{eq}`eq:utility`。

令 $U_1 = \frac{\partial U}{\partial c}, U_2 = \frac{\partial U}{\partial (1-n)} = -\frac{\partial U}{\partial n}$，我们可以从拉格朗日函数

$$
\mathcal{L} = \sum_{t=0}^\infty \beta^t U(c_t, 1 - n_t) + \mu \left( \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t - (1 - \tau_{nt})w_t n_t + \ldots \right] \right)
$$

推导出一阶条件

$$
\frac{\partial \mathcal{L}}{\partial c_t} = \beta^t U_{1}(c_t, 1 - n_t) - \mu q_t (1 + \tau_{ct}) = 0
$$ (eq:foc_c_1)

和

$$
\frac{\partial \mathcal{L}}{\partial n_t} = \beta^t \left(-U_{2t}(c_t, 1 - n_t)\right) - \mu q_t (1 - \tau_{nt}) w_t = 0.
$$ (eq:foc_n_1)

对{eq}`eq:foc_c_1`和{eq}`eq:foc_n_1`进行整理，我们得到

$$
\begin{aligned}
\beta^t U_1(c_t, 1 - n_t)  = \beta^t U_{1t} = \mu q_t (1 + \tau_{ct}),
\end{aligned}
$$ (eq:foc_c)

$$
\begin{aligned}
\beta^t U_2(c_t, 1 - n_t) = \beta^t U_{2t} = \mu q_t (1 - \tau_{nt}) w_t.
\end{aligned}
$$ (eq:foc_n)


将{eq}`eq:foc_c`代入{eq}`eq:terminal`并替换 $q_t$,我们得到终端条件

$$
-\lim_{T \to \infty} \beta^T \frac{U_{1T}}{(1 + \tau_{cT})} k_{T+1} = 0.
$$ (eq:terminal_final)

## 计算均衡

为了计算均衡,我们需要寻找一个价格体系 $\{q_t, \eta_t, w_t\}$、一个预算可行的政府政策 $\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$ 以及一个配置 $\{c_t, n_t, k_{t+1}\}$,它们能够解决由以下组成的非线性差分方程系统:

- 可行性条件{eq}`eq:tech_capital`、家庭无套利条件{eq}`eq:no_arb`、企业无套利条件{eq}`eq:no_arb_firms`、家庭的一阶条件{eq}`eq:foc_c`和{eq}`eq:foc_n`，
- 初始条件 $k_0$ 和终端条件{eq}`eq:terminal_final`。



(cass_fiscal_shooting)=
## Python代码

我们需要以下导入

```{code-cell} ipython3
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from collections import namedtuple
from mpmath import mp, mpf
from warnings import warn

# 设置计算精度
mp.dps = 40
mp.pretty = True
```

我们使用`mpmath`库在射击算法中执行高精度运算，以防止由于数值不稳定而导致解发散。

```{note}
在下面的函数中，我们包含了一些处理增长成分的例程（将在{doc}`Exogenous growth` 一节中进一步讨论）。

我们在这里提前加入这些代码是为了避免代码重复。
```


我们设置以下参数

```{code-cell} ipython3
# 创建一个命名元组来存储模型参数
Model = namedtuple("Model", 
            ["β", "γ", "δ", "α", "A"])

def create_model(β=0.95, # 贴现因子
                 γ=2.0,  # 相对风险厌恶系数
                 δ=0.2,  # 折旧率
                 α=0.33, # 资本份额
                 A=1.0   # 全要素生产率
                 ):
    """创建一个模型实例。"""
    return Model(β=β, γ=γ, δ=δ, α=α, A=A)

model = create_model()

# 总期数
S = 100
```

### 非弹性劳动供给

在本讲中，我们考虑一个特殊情形，即 $U(c, 1-n) = u(c)$，$f(k) := F(k, 1)$。

我们用 $f(k) := F(k, 1)$ 将{eq}`eq:tech_capital`重写为

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

```{code-cell} ipython3
def next_k(k_t, g_t, c_t, model, μ_t=1):
    """
    下一期资本：k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    带有可选的调整: k_{t+1} = (f(k_t) + (1 - δ) * k_t - c_t - g_t) / μ_{t+1}
    """
    return (f(k_t, model) + (1 - model.δ) * k_t - g_t - c_t) / μ_t
```

根据线性齐次生产函数的性质，我们有 $F_k(k, n) = f'(k)$ 和 $F_n(k, 1) = f(k, 1) - f'(k)k$。

将{eq}`eq:foc_c`、{eq}`eq:no_arb_firms`和{eq}`eq:feasi_capital`代入{eq}`eq:no_arb`,我们得到欧拉方程

$$
\begin{aligned}
&\frac{u'(f(k_t) + (1 - \delta) k_t - g_t - k_{t+1})}{(1 + \tau_{ct})} \\
&- \beta \frac{u'(f(k_{t+1}) + (1 - \delta) k_{t+1} - g_{t+1} - k_{t+2})}{(1 + \tau_{ct+1})} \\
&\times [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1] = 0.
\end{aligned}
$$(eq:euler_house)

这可以简化为:

$$
\begin{aligned}
u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second)

方程{eq}`eq:diff_second`将在我们的均衡计算算法中发挥重要作用。

### 稳态

税率和政府支出在差分方程{eq}`eq:feasi_capital`和{eq}`eq:diff_second`中起到**强制函数**的作用。

定义 $z_t = [g_t, \tau_{kt}, \tau_{ct}]'$。

将二阶差分方程表示为：

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

我们假设政府政策达到稳态，使得 $\lim_{t \to \infty} z_t = \bar z$，且该稳态在 $t > T$ 时保持。

终端稳态资本存量 $\bar{k}$ 满足：

$$
H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0.
$$

由差分方程{eq}`eq:diff_second`，我们可以推导出稳态的约束条件：

$$
\begin{aligned}
u'(\bar{c}) &= \beta u'(\bar{c}) \frac{(1 + \bar{\tau}_{c})}{(1 + \bar{\tau}_{c})} [(1 - \bar{\tau}_{k})(f'(\bar{k}) - \delta) + 1]. \\
&\implies 1 = \beta[(1 - \bar{\tau}_{k})(f'(\bar{k}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second_steady)

### 其他均衡数量和价格

*价格：*

$$
q_t = \frac{\beta^t u'(c_t)}{u'(c_0)}
$$ (eq:equil_q)

```{code-cell} ipython3
def compute_q_path(c_path, model, S=100, A_path=None):
    """
    计算q路径：q_t = (β^t * u'(c_t)) / u'(c_0)
    在增长模型中可以选择性地传入 A_path。
    """
    A = np.ones_like(c_path) if A_path is None else np.asarray(A_path)
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (model.β ** t * 
                     u_prime(c_path[t], model, A[t])) / u_prime(c_path[0], model, A[0])
    return q_path
```
*资本租赁率*

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100, A_path=None):
    """
    计算η路径：η_t = f'(k_t)
    在增长模型中可以选择性地传入 A_path。
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model, A[t])
    return η_path
```
*劳动力租赁率：*

$$
w_t = f(k_t) - k_t f'(k_t)    
$$

```{code-cell} ipython3
def compute_w_path(k_path, η_path, model, S=100, A_path=None):
    """
    计算w路径：w_t = f(k_t) - k_t * f'(k_t)
    在增长模型中可以选择性地传入 A_path。
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = f(k_path[t], model, A[t]) - k_path[t] * η_path[t]
    return w_path
```
*资本的单期回报率：*

$$
\bar{R}_{t+1} = \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1\right] =  \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} R_{t, t+1}
$$ (eq:gross_rate)

```{code-cell} ipython3
def compute_R_bar(τ_ct, τ_ctp1, τ_ktp1, k_tp1, model):
    """
    资本的单期总回报率：
    R_bar = [(1 + τ_c_t) / (1 + τ_c_{t+1})] 
        * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    return ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(k_tp1, model) - model.δ) + 1) 

def compute_R_bar_path(shocks, k_path, model, S=100):
    """
    计算随时间变化的R̄路径。
    """
    R_bar_path = np.zeros(S + 1)
    for t in range(S):
        R_bar_path[t] = compute_R_bar(
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k_path[t + 1], model)
    R_bar_path[S] = R_bar_path[S - 1]
    return R_bar_path
```

*一期贴现因子：*

$$
R^{-1}_{t, t+1} = \frac{q_{t+1}}{q_{t}} = m_{t, t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)} \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}
$$ (eq:equil_bigR)


*一期净利率：*

$$
r_{t, t+1} \equiv R_{t, t+1} - 1 = (1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta)
$$ (eq:equil_r)

根据{eq}`eq:equil_bigR`和 $r_{t, t+1} = - \ln(\frac{q_{t+1}}{q_t})$，我们有

$$
R_{t, t+s} = e^{s \cdot r_{t, t+s}}.
$$

然后根据{eq}`eq:equil_r`，我们有

$$
\frac{q_{t+s}}{q_t} = e^{-s \cdot r_{t, t+s}}.
$$

重新整理上述方程，我们得到

$$
r_{t, t+s} = -\frac{1}{s} \ln\left(\frac{q_{t+s}}{q_t}\right).
$$

```{code-cell} ipython3
def compute_rts_path(q_path, S, t):
    """
    计算r路径：
    r_t,t+s = - (1/s) * ln(q_{t+s} / q_t)
    """
    s = np.arange(1, S + 1) 
    q_path = np.array([float(q) for q in q_path]) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rts_path = - np.log(q_path[t + s] / q_path[t]) / s
    return rts_path
```
## 一些函数形式

我们假设代表性家庭的效用函数具有以下CRRA（常数相对风险厌恶）形式

$$
u(c) = \frac{c^{1 - \gamma}}{1 - \gamma}
$$

```{code-cell} ipython3
def u_prime(c, model, A_t=1):
    """
    边际效用：u'(c) = c^{-γ}
    带可选的技术调整： u'(cA) = (cA)^{-γ}
    """
    return (c * A_t) ** (-model.γ)
```

将{eq}`eq:gross_rate`代入{eq}`eq:diff_second`，我们得到

$$
c_{t+1} = c_t \left[ \beta \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta) + 1 \right] \right]^{\frac{1}{\gamma}} = c_t \left[ \beta \overline{R}_{t+1} \right]^{\frac{1}{\gamma}}
$$ (eq:consume_R)

```{code-cell} ipython3
def next_c(c_t, R_bar, model, μ_t=1):
    """
    下一期消费：c_{t+1} = c_t * (β * R̄)^{1/γ}
    带可选的增长调整：c_{t+1} = c_t * (β * R_bar)^{1/γ} * μ_{t+1}^{-1}
    """
    return c_t * (model.β * R_bar) ** (1 / model.γ) / μ_t
```

对于生产函数，我们假设其为柯布-道格拉斯形式：

$$
F(k, 1) = A k^\alpha
$$

```{code-cell} ipython3
def f(k, model): 
    """
    生产函数：f(k) = A * k^{α}
    """
    return A * k ** model.α

def f_prime(k, model):
    """
    资本的边际产出：f'(k) = α * A * k^{α - 1}
    """
    return model.α * A * k ** (model.α - 1)
```
## 计算

我们介绍两种计算均衡的方法：

* 射击算法
* 残差最小化方法，主要关注满足欧拉方程{eq}`eq:diff_second`和可行性条件{eq}`eq:feasi_capital`。

### 射击算法

该算法包含以下步骤：

1. 求解方程{eq}`eq:diff_second_steady`，得到与永久政策向量 $\bar{z}$ 相对应的终端稳态资本存量 $\bar{k}$。

2. 选择一个远大于 $T$ 的时间指标 $S \gg T$，猜测一个初始消费率 $c_0$，并利用方程{eq}`eq:feasi_capital`求解 $k_1$。

3. 使用方程{eq}`eq:consume_R`确定 $c_{t+1}$。然后，应用方程{eq}`eq:feasi_capital`计算 $k_{t+2}$。

4. 重复步骤3，计算 $t = 1, \dots, S$ 时的候选值 $\hat{k}_t$。

5. 计算差值 $\hat{k}_S - \bar{k}$。如果对于某个小 $\epsilon$，$\left| \hat{k}_S - \bar{k} \right| > \epsilon$，则调整 $c_0$ 并重复步骤2-5。
6. 通过二分法迭代调整 $c_0$，直到找到一个值使得 $\left| \hat{k}_S - \bar{k} \right| < \epsilon$。

以下代码实现了这些步骤。

```{code-cell} ipython3
# 稳态计算
def steady_states(model, g_ss, τ_k_ss=0.0, μ_ss=None):
    """
    计算资本与消费的稳态值，
    在增长模型中可以选择性地传入 A_path。
    """

    β, δ, α, γ = model.β, model.δ, model.α, model.γ

    A = model.A or 1.0

    # 分子中的增长调整：μ^γ 或 1
    μ_eff = μ_ss**γ if μ_ss is not None else 1.0

    num = δ + (μ_eff/β - 1) / (1 - τ_k_ss)
    k_ss = (num / (α * A)) ** (1 / (α - 1))

    c_ss = (
        A * k_ss**α - δ * k_ss - g_ss
        if μ_ss is None
        else k_ss**α + (1 - δ - μ_ss) * k_ss - g_ss
    )

    return k_ss, c_ss

def shooting_algorithm(
    c0, k0, shocks, S, model, A_path=None):
    """
    给定初始 c0 和 k0 的射击算法，
    在增长模型中可以选择性地传入 A_path。
    """
    # 解包并将 shocks 转为 mpf 格式，如果缺少 μ 就填充为全 1
    g = np.array(list(map(mpf, shocks['g'])), dtype=object)
    τ_c = np.array(list(map(mpf, shocks['τ_c'])), dtype=object)
    τ_k = np.array(list(map(mpf, shocks['τ_k'])), dtype=object)
    μ = (np.array(list(map(mpf, shocks['μ'])), dtype=object)
              if 'μ' in shocks else np.ones_like(g))
    A = np.ones_like(g) if A_path is None else A_path

    k_path = np.empty(S+1, dtype=object)
    c_path = np.empty(S+1, dtype=object)
    k_path[0], c_path[0] = mpf(k0), mpf(c0)

    for t in range(S):
        k_t, c_t = k_path[t], c_path[t]
        k_tp1 = next_k(k_t, g[t], c_t, model, μ[t+1])
        if k_tp1 < 0:
            return None, None
        k_path[t+1] = k_tp1

        R_bar = compute_R_bar(
            τ_c[t], τ_c[t+1], τ_k[t+1], k_tp1, model
        )
        c_tp1 = next_c(c_t, R_bar, model, μ[t+1])
        if c_tp1 < 0:
            return None, None
        c_path[t+1] = c_tp1

    return k_path, c_path


def bisection_c0(
    c0_guess, k0, shocks, S, model, tol=mpf('1e-6'), 
    max_iter=1000, verbose=False, A_path=None):
    """
    二分法寻找初始消费值 c0
    """
    # 稳态使用最后一期的 shocks（如果缺少 μ，则设为 1）
    g_last    = mpf(shocks['g'][-1])
    τ_k_last  = mpf(shocks['τ_k'][-1])
    μ_last    = mpf(shocks['μ'][-1]) if 'μ' in shocks else mpf('1')
    k_ss_fin, _ = steady_states(model, g_last, τ_k_last, μ_last)

    c0_lo, c0_hi = mpf('0'), f(k_ss_fin, model)
    c0 = mpf(c0_guess)

    for i in range(1, max_iter+1):
        k_path, _ = shooting_algorithm(c0, k0, shocks, S, model, A_path)
        if k_path is None:
            if verbose:
                print(f"[{i}] 射击失败，c0={c0}")
            c0_hi = c0
        else:
            err = k_path[-1] - k_ss_fin
            if verbose and i % 100 == 0:
                print(f"[{i}] c0={c0}, 误差={err}")
            if abs(err) < tol:
                if verbose:
                    print(f"在 {i} 次迭代后收敛")
                return c0
            # 单行更新区间
            c0_lo, c0_hi = (c0, c0_hi) if err > 0 else (c0_lo, c0)
        c0 = (c0_lo + c0_hi) / mpf('2')

    warn(f"二分法在 {max_iter} 次迭代后未收敛；返回 c0={c0}")
    return c0


def run_shooting(
    shocks, S, model, A_path=None, 
    c0_finder=bisection_c0, shooter=shooting_algorithm):
    """
    计算初始稳态，寻找 c0，并返回 [k,c] 路径
    在增长模型中可以选择性地传入 A_path。
    """
    # t=0 时的初始稳态（如果缺少 μ，则设为 1）
    g0    = mpf(shocks['g'][0])
    τ_k0  = mpf(shocks['τ_k'][0])
    μ0    = mpf(shocks['μ'][0]) if 'μ' in shocks else mpf('1')
    k0, c0 = steady_states(model, g0, τ_k0, μ0)

    optimal_c0 = c0_finder(c0, k0, shocks, S, model, A_path=A_path)
    print(f"模型: {model}\n最优初始消费 c0 = {mpf(optimal_c0)}")

    k_path, c_path = shooter(optimal_c0, k0, shocks, S, model, A_path)
    return np.column_stack([k_path, c_path])
```
(cf:experiments)=
### 实验

让我们进行一些实验。

1. 可预期的一次性永久冲击：在第 10 期，$g$ 从 0.2 上升到 0.4；
2. 可预期的一次性永久冲击：在第 10 期，$\tau_c$ 从 0.0 上升到 0.2；
3. 可预期的一次性永久冲击：在第 10 期，$\tau_k$ 从 0.0 上升到 0.2;
4. 可预期的一次性暂时冲击：在第 10 期，$g$ 从 0.2 上升到 0.4，之后 $g$ 永久恢复为 0.2。

首先,我们准备用于初始化迭代算法的序列。

我们将从一个初始稳态开始,并在指定时间施加冲击。

```{code-cell} ipython3
def plot_results(
    solution, k_ss, c_ss, shocks, shock_param, axes, model,
    A_path=None, label='', linestyle='-', T=40):
    """
    绘制模拟结果 (k, c, R, η 以及政策冲击)，
    在增长模型中可以选择性地传入 A_path。
    """
    k_path = solution[:, 0]
    c_path = solution[:, 1]
    T = min(T, k_path.size)

    # 处理增长参数
    μ0 = shocks['μ'][0] if 'μ' in shocks else 1.0
    A0 = A_path[0] if A_path is not None else (model.A or 1.0)

    # 稳态参考线
    R_bar_ss = (1 / model.β) * (μ0**model.γ)
    η_ss     = model.α * A0 * k_ss**(model.α - 1)

    # 绘制资本路径 k
    axes[0].plot(k_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(k_ss, linestyle='--', color='black')
    axes[0].set_title('k')

    # 绘制消费路径 c
    axes[1].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[1].axhline(c_ss, linestyle='--', color='black')
    axes[1].set_title('c')

    # 绘制 R̄ 路径
    S_full    = k_path.size - 1
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S_full)
    axes[2].plot(R_bar_path[:T], linestyle=linestyle, label=label)
    axes[2].axhline(R_bar_ss, linestyle='--', color='black')
    axes[2].set_title(r'$\bar{R}$')

    # 绘制 η 路径
    η_path = compute_η_path(k_path, model, S_full)
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')

    # 绘制冲击变量
    shock_series = np.array(shocks[shock_param], dtype=object)
    axes[4].plot(shock_series[:T], linestyle=linestyle, label=label)
    axes[4].axhline(shock_series[0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')

    if label:
        for ax in axes[:5]:
            ax.legend()
```
**实验1：可预期的一次性永久冲击：在第 10 期，$g$ 从 0.2 上升到 0.4**

下图显示了在 $t = T = 10$ 时，一个可预期的政府支出 $g$ 的永久增加所带来的结果。该增加通过提高一次性总额税来融资。

```{code-cell} ipython3
# 将冲击定义为字典
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

print(f"稳态资本: {k_ss_initial:.4f}")
print(f"稳态消费: {c_ss_initial:.4f}")

solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```
上述图形表明，均衡中的**消费平滑**机制正在发挥作用，这一机制源自代表性消费者对平滑消费路径的偏好，而这种偏好来自其单期效用函数的曲率。

- 资本存量的稳态值保持不变：
  - 这是因为在欧拉方程的稳态版本中({eq}`eq:diff_second_steady`)，$g$项消失了。
- 在时间 $T$ 之前，由于政府消费增加，消费开始逐渐下降：
  - 家庭减少消费以抵消政府支出，而这些政府支出通过增加一次性税收来融资。
  - 竞争性经济通过增加一次性税收流向家庭发出减少消费的信号。
  - 家庭关注的是税收的现值而非征收时间，因此消费受到不利的财富效应影响，导致立即做出反应。
- 资本在时间 $0$ 到 $T$ 之间由于储蓄增加而逐渐积累,在时间 $T$ 之后逐渐减少:
    - 这种资本存量的时间变化平滑了消费的时间分布,这是由代表性消费者的消费平滑动机驱动的。

让我们把上述程序整合成一个函数,该函数可以针对给定的实验运行求解器并绘制图表

```{code-cell} ipython3
:tags: [hide-input]

def experiment_model(
    shocks, S, model, A_path=None, solver=run_shooting, 
    plot_func=plot_results, policy_shock='g', T=40):
    """
    运行射击算法并绘制结果。
    """
    # 初始稳态 (如果没有增长，则 μ0=None)
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    μ0   = mpf(shocks['μ'][0]) if 'μ' in shocks else None
    k_ss, c_ss = steady_states(model, g0, τk0, μ0)

    print(f"稳态资本: {float(k_ss):.4f}")
    print(f"稳态消费: {float(c_ss):.4f}")
    print('-'*64)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    sol = solver(shocks, S, model, A_path)
    plot_func(
        sol, k_ss, c_ss, shocks, policy_shock, axes, model,
        A_path=A_path, T=T
    )

    # 删除未使用的子图
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```
下图比较了两个经济体在 $t = 10$ 时对预期的 $g$ 增长的响应:

* 实线表示我们原始的 $\gamma = 2$ 的经济体，
* 虚线表示一个除了 $\gamma = 0.2$ 外其他条件完全相同的经济体。

这个比较之所以有趣，是因为效用曲率参数 $\gamma$ 决定了家庭跨期替代消费的意愿，从而决定了其对消费路径随时间平滑程度的偏好。

```{code-cell} ipython3
# 使用射击算法求解模型
solution = run_shooting(shocks, S, model)

# 计算初始稳态
k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

# 绘制 γ=2 时的解
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

label = fr"$\gamma = {model.γ}$"
plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model, label=label, 
             T=40)

# 求解并绘制 γ=0.2 的结果
model_γ2 = create_model(γ=0.2)
solution = run_shooting(shocks, S, model_γ2)

plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model_γ2, 
             label=fr"$\gamma = {model_γ2.γ}$", 
             linestyle='-.', T=40)

handles, labels = axes[0].get_legend_handles_labels()  
fig.legend(handles, labels, loc='lower right', 
           ncol=3, fontsize=14, bbox_to_anchor=(1, 0.1))  

for ax in axes[5:]:
    fig.delaxes(ax)
    
plt.tight_layout()
plt.show()
```
结果表明降低 $\gamma$ 会同时影响消费和资本存量路径，因为它增加了代表性消费者跨期替代消费的意愿：

- 消费路径：
  - 当 $\gamma = 0.2$ 时，与 $\gamma = 2$ 相比，消费变得不那么平滑。
  - 对于 $\gamma = 0.2$，消费更紧密地跟随政府支出路径，在 $t = 10$ 之前保持较高水平。

- 资本存量路径：
  - 当 $\gamma = 0.2$ 时，资本存量的积累和减少幅度较小。
  - $\bar{R}$ 和 $\eta$ 的波动也较小。

让我们编写另一个函数来运行求解器并为这两个实验绘制图表

```{code-cell} ipython3
:tags: [hide-input]

def experiment_two_models(
    shocks, S, model_1, model_2, solver=run_shooting, plot_func=plot_results, 
    policy_shock='g', legend_label_fun=None, T=40, A_path=None):
    """
    比较并绘制两个模型的射击算法路径。
    """
    is_growth = 'μ' in shocks
    μ0 = mpf(shocks['μ'][0]) if is_growth else None

    # 两个模型的初始稳态
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    k_ss1, c_ss1 = steady_states(model_1, g0, τk0, μ0)
    k_ss2, c_ss2 = steady_states(model_2, g0, τk0, μ0)

    # 打印两个模型的结果   
    print(f"Model 1 (γ={model_1.γ}): 稳态 k={float(k_ss1):.4f}, c={float(c_ss1):.4f}")
    print(f"Model 2 (γ={model_2.γ}): 稳态 k={float(k_ss2):.4f}, c={float(c_ss2):.4f}")
    print('-'*64)

    # 默认图例标签
    if legend_label_fun is None:
        legend_label_fun = lambda m: fr"$\gamma = {m.γ}$"

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    # 遍历 (模型, 稳态, 线型)
    for model, (k_ss, c_ss), ls in [
        (model_1, (k_ss1, c_ss1), '-'),
        (model_2, (k_ss2, c_ss2), '-.')
    ]:
        sol = solver(shocks, S, model, A_path)
        plot_func(sol, k_ss, c_ss, shocks, policy_shock, axes, 
                  model, A_path=A_path, 
                  label=legend_label_fun(model), 
                  linestyle=ls, T=T)

    # 在右下角绘制共享图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower right', ncol=2, 
        fontsize=12, bbox_to_anchor=(1, 0.1))

    # 删除未使用的子图
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```
现在我们绘制其他均衡量：

```{code-cell} ipython3
def plot_prices(solution, c_ss, shock_param, axes,
                model, label='', linestyle='-', T=40):
    """
    比较并绘制价格路径
    """
    α, β, δ, γ, A = model.α, model.β, model.δ, model.γ, model.A
    
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    # 绘制消费路径 c
    axes[0].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(c_ss, linestyle='--', color='black')
    axes[0].set_title('c')
    
    # 绘制 q 路径
    q_path = compute_q_path(c_path, model, S=S)
    axes[1].plot(q_path[:T], linestyle=linestyle, label=label)
    axes[1].plot(β**np.arange(T), linestyle='--', color='black')
    axes[1].set_title('q')
    
    # 绘制 r_{t,t+1}
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)
    axes[2].plot(R_bar_path[:T] - 1, linestyle=linestyle, label=label)
    axes[2].axhline(1 / β - 1, linestyle='--', color='black')
    axes[2].set_title('$r_{t,t+1}$')

    # 绘制 r_{t,t+s}
    for style, s in zip(['-', '-.', '--'], [0, 10, 60]):
        rts_path = compute_rts_path(q_path, T, s)
        axes[3].plot(rts_path, linestyle=style, 
                     color='black' if style == '--' else None,
                     label=f'$t={s}$')
        axes[3].set_xlabel('s')
        axes[3].set_title('$r_{t,t+s}$')

    # 绘制 g（冲击变量）
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(shock_param)
```
对于$\gamma = 2$,下图描述了 $q_t$ 以及利率期限结构对于在 $t = 10$ 时可预见的 $g_t$ 增长的响应

```{code-cell} ipython3
solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```
上方的第二幅图比较了初始稳态下的 $q_t$ 与在 $t = 0$ 时预见到 $g$ 增加后的 $q_t$，而第三幅图比较了隐含的短期利率 $r_t$。

第四幅图展示了在 $t=0$、$t=10$ 和 $t=60$ 时的利率期限结构。

注意，到 $t = 60$ 时，系统已经收敛到新的稳态，利率期限结构变得平坦。

在 $t = 10$ 时，利率期限结构呈上升趋势。

这种上升趋势反映了消费增长率随时间的预期增长，如消费图所示。

在 $t = 0$ 时，利率期限结构呈现"U形"模式：

- 在 $s = 10$ 之前呈下降趋势。
- 在 $s = 10$ 之后，随到期期限的增加而上升。
    
这种模式与前两张图中的消费增长模式相一致：即在 $t = 10$ 之前以递增的速率下降，之后以递减的速率下降。

+++

**实验2：可预期的一次性永久冲击：在第 10 期，$\tau_c$ 从 0.0 上升到 0.2**

在劳动供给缺乏弹性的情况下，欧拉方程{eq}`eq:euler_house`和其他均衡条件表明：
- 固定的消费税不会扭曲决策，但是
- 可预期的消费税变化会造成扭曲。

事实上，{eq}`eq:euler_house`或{eq}`eq:diff_second`表明，可预期的 $\tau_{ct}$ 增加（即 $(1+\tau_{ct})(1+\tau_{ct+1})$ 减少）与 $\tau_{kt}$ 增加的作用相同。

下图展示了对可预期的消费税 $\tau_c$ 增加的响应。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, 
                 solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='τ_c')
```
显然，上图中的所有变量最终都会回到其初始稳态值。

预期的 $\tau_{ct}$ 增加导致消费和资本存量随时间发生变化：

- 在 $t = 0$ 时：
    - 可预期的 $\tau_c$ 增加导致*消费的立即跳升*。
    - 随后出现*消费狂潮*，使资本存量在 $t = T = 10$ 之前持续下降。
- 在 $t = 0$ 和 $t = T = 10$ 之间：
    - 资本存量的下降导致 $\bar{R}$ 随时间上升。
    - 均衡条件要求消费增长率持续上升，直到 $t = T$。
- 在 $t = T = 10$ 时：
    - $\tau_c$ 的跳升使 $\bar{R}$ 降至 1 以下，导致*消费急剧下降*。
- 在 $T = 10$ 之后：
    - 预期扭曲的影响结束，经济逐渐调整到更低的资本存量水平。
    - 资本现在必须增长，这需要*紧缩* —— 在 $t = T$ 之后消费大幅下降，表现为更低的消费水平。
    - 利率逐渐下降，消费以递减的速率增长，直至达到最终稳态。

+++

**实验3：可预期的一次性永久冲击：在第 10 期，$\tau_k$ 从 0.0 上升到 0.2**

对于 $\gamma$ 取值为 2 和 0.2 的两种情况，下图显示了在 $t = T = 10$ 时，可预期的一次性永久性 的$\tau_{kt}$ 跳升所带来的反应。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='τ_k')
```
政府支出路径保持不变
- $\tau_{kt}$ 的增加通过减少一次性税收的现值来抵消，以保持预算平衡。

图表显示：

- 对 $\tau_{kt}$ 增加的预期导致资本存量立即下降，这是由于当前和后续消费的增加。
- $\bar{R}$ 从 $t = 0$ 开始上升，在 $t = 9$ 达到峰值，在 $t = 10$ 时因税收变化而急剧下降。
    - $\bar{R}$ 的变化与 $t = 10$ 时税收增加对跨期消费的影响相一致。
- 转型动态推动 $k_t$（资本存量）向一个新的、更低的稳态水平移动。在新的稳态下：
    - 由于资本存量减少导致产出降低，消费水平更低。
    - $\gamma = 2$ 时的消费路径比 $\gamma = 0.2$ 时的更平滑。

+++

到目前为止，我们已经探讨了可预期的一次性永久性政府政策变动的后果。接下来，我们进行一些实验，其中政策变量仅发生可预期的存在可预期的一次性暂时变化（称为"脉冲"）。

**实验4： 可预期的一次性暂时冲击：在第 10 期，$g$ 从 0.2 上升到 0.4，之后 $g$ 永久恢复为 0.2**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model,
                 solver=run_shooting, 
                 plot_func=plot_results,  
                 policy_shock='g')
```
该图表明：

- 消费：
    - 在政策宣布后立即下降，并随着时间推移持续下降。
    - 在 $t = 10$ 的冲击之后，消费开始恢复，但以递减的速度上升，逐步趋近其稳态值。

- 资本和 $\bar{R}$：
    - 在 $t = 10$ 之前，由于利率变化导致家庭为预期中的政府支出增加做准备，资本开始积累。
    - 在 $t = 10$ 时，由于政府消耗了部分资本，资本存量急剧下降。
    - 由于资本减少，$\bar{R}$ 跃升至其稳态值以上，然后逐渐下降回稳态水平。
+++

### 方法2：残差最小化

第二种方法涉及最小化以下方程的残差（即与等式的偏差）：

- 欧拉方程{eq}`eq:diff_second`：

  $$
1 = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma} \frac{(1+\tau_{ct})}{(1+\tau_{ct+1})} \left[(1 - \tau_{kt+1})(\alpha A k_{t+1}^{\alpha-1} - \delta) + 1 \right]
  $$

- 可行性条件 {eq}`eq:feasi_capital`:

  $$
  k_{t+1} = A k_{t}^{\alpha} + (1 - \delta) k_t - g_t - c_t.
  $$

```{code-cell} ipython3
# 欧拉方程与可行性条件 
def euler_residual(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model, μ_tp1=1):
    """
    计算欧拉方程的残差，
    可选增长模型的参数μ_tp1
    """
    R_bar = compute_R_bar(τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model)
    
    c_expected = next_c(c_t, R_bar, model, μ_tp1)

    return c_expected / c_tp1 - 1.0

def feasi_residual(k_t, k_tm1, c_tm1, g_t, model, μ_t=1):
    """
    计算可行性条件的残差，
    可选增长模型的参数μ_t。
    """
    k_t_expected = next_k(k_tm1, g_t, c_tm1, model, μ_t)
    return k_t_expected - k_t
```
算法步骤如下：

1. 根据 $t=0$ 时的政府计划，找到初始稳态 $k_0$。

2. 初始化一个初始猜测 $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$。

3. 计算残差 $l_{ta}$ 和 $l_{tk}$ （对于 $t = 0, \dots, S$），以及 $t = 0$ 时的 $l_{k_0}$ 和 $t = S$ 时的 $l_{k_S}$：
   - 使用{eq}`eq:diff_second`计算 $t = 0, \dots, S$ 的欧拉方程残差：

     $$
     l_{ta} = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] - 1
     $$

   - 使用{eq}`eq:feasi_capital`计算 $t = 1, \dots, S-1$ 的可行性条件残差：

     $$
     l_{tk} = k_{t+1} - f(k_t) - (1 - \delta)k_t + g_t + c_t
     $$

   - 使用{eq}`eq:diff_second_steady`和初始资本 $k_0$ 计算 $k_0$ 的初始条件残差：

     $$
     l_{k_0} = 1 - \beta \left[ (1 - \tau_{k0}) \left(f'(k_0) - \delta \right) + 1 \right]
     $$
    - 在假设 $c_t = c_{t+1} = c_S$、$k_t = k_{t+1} = k_S$、$\tau_{ct} = \tau_{ct+1} = \tau_{cS}$ 和 $\tau_{kt} = \tau_{kt+1} = \tau_{kS}$ 的条件下，使用{eq}`eq:diff_second`计算终端条件 $t = S$ 的残差：
     
     $$
     l_{k_S} = \beta u'(c_S) \frac{(1 + \tau_{cS})}{(1 + \tau_{cS})} \left[(1 - \tau_{kS})(f'(k_S) - \delta) + 1 \right] - 1
     $$

4. 迭代调整 $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$ 的猜测值，以最小化残差 $l_{k_0}$、$l_{ta}$、$l_{tk}$ 和 $l_{k_S}$（对于 $t = 0, \dots, S$）。

```{code-cell} ipython3
def compute_residuals(vars_flat, k_init, S, shock_paths, model):
    """
    计算欧拉方程与可行性条件的残差。
    """
    g, τ_c, τ_k, μ = (shock_paths[key] for key in ('g','τ_c','τ_k','μ'))
    k, c = vars_flat.reshape((S+1, 2)).T
    res = np.empty(2*S+2, dtype=float)

    # 初始资本的边界条件
    res[0] = k[0] - k_init

    # 内部的欧拉方程与可行性条件
    for t in range(S):
        res[2*t + 1] = euler_residual(
            c[t],    c[t+1],
            τ_c[t],  τ_c[t+1],
            τ_k[t+1],k[t+1],
            model, μ[t+1])
        res[2*t + 2] = feasi_residual(
            k[t+1], k[t], c[t],
            g[t],  model,
            μ[t+1])

    # 终端时点 t = S 的欧拉条件
    res[-1] = euler_residual(
        c[S],   c[S],
        τ_c[S], τ_c[S],
        τ_k[S], k[S],
        model,
        μ[S])

    return res


def run_min(shocks, S, model, A_path=None):
    """
    通过对残差求根来求解完整的 (k, c) 路径。
    """
    shocks['μ'] = shocks['μ'] if 'μ' in shocks else np.ones_like(shocks['g'])

    # 计算稳态：既用作初始资本，也用作初始猜测
    k_ss, c_ss = steady_states(
        model,
        shocks['g'][0],
        shocks['τ_k'][0],
        shocks['μ'][0]  # 若无增长，则 = 1
    )

    # 初始猜测：在稳态处保持常数
    guess = np.column_stack([
        np.full(S+1, k_ss),
        np.full(S+1, c_ss)
    ]).flatten()

    sol = root(
        compute_residuals,
        guess,
        args=(k_ss, S, shocks, model),
        tol=1e-8
    )

    return sol.x.reshape((S+1, 2))
```
我们发现方法2没有遇到数值稳定性问题，所以无需使用 `mp.mpf`。

我们把用第二种方法复现我们的实验作为练习。

```{exercise}
:label: cass_fiscal_ex1

使用第二种残差最小化方法复现我们四个实验的图表：
1. 可预期的一次性永久冲击：在第 10 期，$g$ 从 0.2 上升到 0.4；
2. 可预期的一次性永久冲击：在第 10 期，$\tau_c$ 从 0.0 上升到 0.2；
3. 可预期的一次性永久冲击：在第 10 期，$\tau_k$ 从 0.0 上升到 0.2;
4. 可预期的一次性暂时冲击：在第 10 期，$g$ 从 0.2 上升到 0.4，之后 $g$ 永久恢复为 0.2。
```

```{solution-start} cass_fiscal_ex1
:class: dropdown
```

参考答案：

**实验1：可预期的一次性永久冲击：在第 10 期，$g$ 从 0.2 上升到 0.4**

```{code-cell} ipython3
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```
```{code-cell} ipython3
experiment_two_models(shocks, S, model, model_γ2, 
                run_min, plot_results, 'g')
```
```{code-cell} ipython3
solution = run_min(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```
**实验2：可预期的一次性永久冲击：在第 10 期，$\tau_c$ 从 0.0 上升到 0.2**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='τ_c')
```
**实验3：可预期的一次性永久冲击：在第 10 期，$\tau_k$ 从 0.0 上升到 0.2**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='τ_k')
```
**实验4：可预期的一次性暂时冲击：在第 10 期，$g$ 从 0.2 上升到 0.4，之后 $g$ 永久恢复为 0.2**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

```{solution-end}
```

```{exercise}
:label: cass_fiscal_ex2

设计一个新实验，其中政府支出 $g$ 在第 10 期从 0.2 增加到 0.4，然后在第 20 期永久降至 0.1。
```

```{solution-start} cass_fiscal_ex2
:class: dropdown
```

参考答案：

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10:20] = 0.4
g_path[20:] = 0.1

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, solver=run_min, 
                 plot_func=plot_results,  
                 policy_shock='g')
```

```{solution-end}
```

(growth_model)=
## 外生增长

在上一节中，我们考虑了一个没有外生增长的模型。

我们通过令生产函数中的项 $A_t$ 为常数来消除增长的影响，即设定 $A_t = 1$，$\forall t$.

现在，我们准备引入增长因素。

为了纳入增长，我们将生产函数修改为：

$$
Y_t = F(K_t, A_tn_t)
$$

其中 $Y_t$ 表示总产出，$N_t$ 表示总就业，$A_t$ 表示劳动增强型技术变化，$F(K, AN)$ 仍然是一个线性齐次的生产函数，与之前相同。

我们假设 $A_t$ 遵循以下过程

$$
A_{t+1} = \mu_{t+1}A_t
$$ (eq:growth)

并且假定 $\mu_{t+1}=\bar{\mu}>1$。

```{code-cell} ipython3
# 将常数 A 参数设为 None
model = create_model(A=None)
```

```{code-cell} ipython3
def compute_A_path(A0, shocks, S=100):
    """
    计算 A 的随时间变化的路径
    """
    A_path = np.full(S + 1, A0)
    for t in range(1, S + 1):
        A_path[t] = A_path[t-1] * shocks['μ'][t-1]
    return A_path
```

### 非弹性劳动供给

由于生产函数具有线性齐次性，可将其表示为：

$$
y_t=f(k_t)
$$

其中 $f(k)=F(k,1) = k^\alpha$，$k_t=\frac{K_t}{n_tA_t}$，$y_t=\frac{Y_t}{n_tA_t}$。

$k_t$ 和 $y_t$ 均以“有效劳动单位” $A_tn_t$ 为基准进行度量。

同时，定义 $c_t=\frac{C_t}{A_tn_t}$ 和 $g_t=\frac{G_t}{A_tn_t}$，其中 $C_t$表示总消费，$G_t$ 表示政府总支出。

我们继续考虑非弹性劳动供给的情形。

在此设定下，资源约束（可行性条件）可改写为下式（为方程{eq}`eq:feasi_capital`的修正版本）：

$$
k_{t+1}=\mu_{t+1}^{-1}[f(k_t)+(1-\delta)k_t-g_t-c_t]
$$ (eq:feasi_mod)


同样，根据线性齐次生产函数的性质，我们有

$$ 
\eta_t = F_k(k_t, 1) = f'(k_t), w_t = F_n(k_t, 1) = f(k_t) - f'(k_t)k_t 
$$

由于人均消费现在是 $c_tA_t$，欧拉方程{eq}`eq:diff_second`的对应形式为：

$$
\begin{aligned}
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned} 
$$ (eq:diff_mod)

$\bar{R}_{t+1}$ 的定义与{eq}`eq:gross_rate`一致，但此处的 $k_t$ 表示每单位有效劳动的资本存量。

因此，代入{eq}`eq:gross_rate`，{eq}`eq:diff_mod`变为

$$
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1})\bar{R}_{t+1}
$$

假设家庭的效用函数与之前相同，我们有

$$
(c_tA_t)^{-\gamma} = \beta (c_{t+1}A_{t+1})^{-\gamma} \bar{R}_{t+1}
$$

因此，{eq}`eq:consume_R`的对应形式为：

$$
c_{t+1} = c_t \left[ \beta \bar{R}_{t+1} \right]^{\frac{1}{\gamma}}\mu_{t+1}^{-1}
$$ (eq:consume_r_mod)

### 稳态

在稳态中，$c_{t+1} = c_t$。因此，方程{eq}`eq:diff_mod`可化为：

$$
1=\mu^{-\gamma}\beta[(1-\tau_k)(f'(k)-\delta)+1] 
$$ (eq:diff_mod_st)

由此，我们可以求得每单位有效劳动资本的稳态水平满足：

$$
f'(k)=\delta + (\frac{\frac{1}{\beta}\mu^{\gamma}-1}{1-\tau_k})
$$ (eq:cap_mod_st)  

并且

$$
\bar{R}=\frac{\mu^{\gamma}}{\beta}
$$ (eq:Rbar_mod_st)

利用方程{eq}`eq:feasi_mod`，可求得每单位有效劳动消费的稳态水平：

$$
c = f(k)+(1-\delta-\mu)k-g
$$

由于算法和作图步骤与先前内容相同，我们将在{ref}`cass_fiscal_shooting`一节中同时包含稳态计算与射击算法的实现。

### 射击算法

现在，我们可以应用射击算法来计算均衡：将冲击变量向量扩展以包含 $\mu_t$，然后按照之前的方法继续求解。

### 实验

我们来进行以下实验：

1. 可预期的一次性永久性冲击：在第 10 期，$\mu$ 从 1.02 增加到 1.025；
2. 不可预期的一次性永久性冲击：在第 0 期，$\mu$ 从 1.02 增加至 1.025。

+++

#### 可预期的一次性永久性冲击：在第 10 期，$\mu$ 从 1.02 增加到 1.025

下图显示了生产率增长 $\mu$ 在 $t=10$ 时从 1.02 永久性提高到 1.025 的影响。

变量 $c$ 和 $k$ 均以有效劳动单位为度量。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 10), np.repeat(1.025, S - 9)))
}

A_path = compute_A_path(1.0, shocks, S)

k_ss_initial, c_ss_initial = steady_states(model, 
                                         shocks['g'][0],
                                         shocks['τ_k'][0],
                                         shocks['μ'][0]
                                        )

print(f"稳态资本存量: {k_ss_initial:.4f}")
print(f"稳态消费: {c_ss_initial:.4f}")

# 使用 A_path 参数运行射击算法
solution = run_shooting(shocks, S, model, A_path)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'μ', axes, model, 
             A_path, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```

这些图形中的结果主要由方程{eq}`eq:diff_mod_st`驱动，并表明：当 $\mu$ 永久性提高时，每单位有效劳动的稳态资本存量将会下降。

图表明：

- 随着资本变得更加高效，即使资本存量减少，人均消费水平仍可上升；
- 消费平滑使得经济主体在预期到 $\mu$ 的增加时*立即增加消费*；
- 资本生产率的提高导致总回报 $\bar R$ 的增加；
- 完全预见使资本增长率的提升在冲击发生前就已影响经济行为，使效果在 $t=0$ 时即刻显现。

#### 实验2：不可预期的一次性永久性冲击：在第 0 期，$\mu$ 从 1.02 增加至 1.025

下图显示了当 $\mu$ 在 $t=0$ 时从 1.02 意外跃升至 1.025 的影响。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

A_path = compute_A_path(1.0, shocks, S)

k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0],
                                           shocks['τ_k'][0],
                                           shocks['μ'][0]
                                          )

print(f"稳态资本存量: {k_ss_initial:.4f}")
print(f"稳态消费: {c_ss_initial:.4f}")

# 使用 A_path 参数运行射击算法
solution = run_shooting(shocks, S, model, A_path)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_results(solution, k_ss_initial, 
             c_ss_initial, shocks, 'μ', axes, model, A_path, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```

同样地，我们可以将上文使用的各个步骤整合到一个函数中，使其在给定实验条件下自动运行求解器并绘制结果图。

```{code-cell} ipython3
def experiment_model(shocks, S, model, A_path, solver, plot_func, policy_shock, T=40):
    """
    给定模型运行射击算法并绘制结果。
    """
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0], shocks['μ'][0])
    
    print(f"稳态资本存量: {k0:.4f}")
    print(f"稳态消费: {c0:.4f}")
    print('-'*64)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    solution = solver(shocks, S, model, A_path)
    plot_func(solution, k0, c0, 
              shocks, policy_shock, axes, model, A_path, T=T)

    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

experiment_model(shocks, S, model, A_path, run_shooting, plot_results, 'μ')
```

图表明：

- 由于缺乏前馈效应，所有变量的路径现在都是平滑的。
- 每有效劳动单位的资本逐渐下降到较低的稳态水平。
- 每有效劳动单位的消费立即跳跃，然后平滑地下降到其较低的稳态值。
- 税后总回报 $\bar{R}$ 再次与消费增长率同步变化，验证了欧拉方程{eq}`eq:diff_mod_st`。

```{exercise}
:label: cass_fiscal_ex3

使用第二种残差最小化方法复现前述两个实验的图形结果：
1. 可预期的一次性永久性冲击：在第 10 期，$\mu$ 从 1.02 增加到 1.025；
2. 不可预期的一次性永久性冲击：在第 0 期，$\mu$ 从 1.02 增加至 1.025。
```

```{solution-start} cass_fiscal_ex3
:class: dropdown
```

参考答案：

**实验1：可预期的一次性永久性冲击：在第 10 期，$\mu$ 从 1.02 增加到 1.025**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 10), np.repeat(1.025, S - 9)))
}

A_path = compute_A_path(1.0, shocks, S)

experiment_model(shocks, S, model, A_path, run_min, plot_results, 'μ')
```

**实验2：不可预期的一次性永久性冲击：在第 0 期，$\mu$ 从 1.02 增加至 1.025**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1),
    'μ': np.concatenate((np.repeat(1.02, 1), np.repeat(1.025, S)))
}

experiment_model(shocks, S, model, A_path, run_min, plot_results, 'μ')
```

```{solution-end}
```

在下一讲{doc}`cass_fiscal_2`中，我们研究了我们单国模型的两国版本，该版本与{cite:t}`mendoza1998international`的研究密切相关。
