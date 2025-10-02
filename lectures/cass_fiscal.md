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

本讲座研究在非随机版本的Cass-Koopmans增长模型中，可预见的财政和技术冲击对竞争均衡价格和数量的影响，该模型的特征在QuantEcon讲座{doc}`cass_koopmans_2`中有所描述。

该模型在{cite}`Ljungqvist2012`第11章中有更详细的讨论。

我们将该模型作为实验室，用于试验近似均衡的数值技术，并展示决策者对未来政府决策具有完全预见性的动态模型的结构。

遵循Robert E. Hall {cite}`hall1971dynamic`的经典论文，我们对Cass-Koopmans最优增长模型的非随机版本进行了扩展，加入了一个政府，该政府购买商品流，并通过一系列扭曲性的固定税率来为其购买融资。

扭曲性税收使竞争均衡配置无法解决规划问题。

因此，为了计算均衡配置和价格系统，我们求解一个由决策者的一阶条件和其他均衡条件组成的非线性差分方程系统。

我们提出了两种近似均衡的方法：

- 第一种是类似于我们在{doc}`cass_koopmans_2`中使用的射击算法。

- 第二种方法是一种求根算法，它最小化来自消费者和代表性企业的一阶条件的残差。



(cs_fs_model)=
## 经济体


### 技术

可行配置满足

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

其中

- $g_t$ 是时间$t$的政府购买
- $x_t$ 是总投资，以及
- $F(k_t, n_t)$ 是线性齐次生产函数，具有正的且递减的资本$k_t$和劳动$n_t$的边际产量。

实物资本按以下方式演化

$$
k_{t+1} = (1 - \delta)k_t + x_t,
$$

其中$\delta \in (0, 1)$是折旧率。

有时从{eq}`eq:tech_capital`中消除$x_t$并将其表示为

$$
g_t + c_t + k_{t+1} \leq F(k_t, n_t) + (1 - \delta)k_t.
$$ 

### 竞争均衡的组成部分

所有交易都发生在时间$0$。

代表性家庭拥有资本，做出投资决策，并将资本和劳动租给代表性生产企业。

代表性企业使用资本和劳动通过生产函数$F(k_t, n_t)$生产商品。

**价格系统**是序列三元组$\{q_t, \eta_t, w_t\}_{t=0}^\infty$，其中

- $q_t$ 是时间$t$的一单位投资或消费（$x_t$或$c_t$）在时间$0$的税前价格，
- $\eta_t$ 是家庭在时间$t$从企业租赁资本所获得的税前价格，以及
- $w_t$ 是家庭在时间$t$向企业租赁劳动所获得的税前价格。

价格$w_t$和$\eta_t$以时间$t$的商品表示，而$q_t$以时间$0$的计价物表示，如{doc}`cass_koopmans_2`中所示。

政府的存在使本讲座与{doc}`cass_koopmans_2`有所区别。

时间$t$的政府购买商品为$g_t \geq 0$。

政府支出计划是一个序列$g = \{g_t\}_{t=0}^\infty$。

政府税收计划是序列的四元组$\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$，
其中

- $\tau_{ct}$ 是时间$t$的消费税率，
- $\tau_{kt}$ 是时间$t$的资本租金税率，
- $\tau_{nt}$ 是时间$t$的工资收入税率，以及
- $\tau_{ht}$ 是时间$t$对消费者的一次性总额税。

由于一次性总额税$\tau_{ht}$可用，政府实际上不应该使用任何扭曲性税收。

然而，我们包括所有这些税收是因为，像{cite}`hall1971dynamic`一样，它们允许我们分析各种税收如何扭曲生产和消费决策。

在[实验部分](cf:experiments)中，我们将看到政府税收计划的变化如何影响过渡路径和均衡。


### 代表性家庭

代表性家庭对单一消费品$c_t$和闲暇$1-n_t$的非负流具有偏好，这些偏好按以下方式排序：

$$
\sum_{t=0}^{\infty} \beta^t U(c_t, 1-n_t), \quad \beta \in (0, 1),
$$ (eq:utility)

其中

- $U$在$c_t$中严格递增，二次连续可微，并且在$c_t \geq 0$和$n_t \in [0, 1]$的条件下严格凹。


代表性家庭在单一预算约束下最大化{eq}`eq:utility`：

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{投资时无税}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \eta_t k_t - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{租金回报税}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)

这里我们假设政府从资本的总租金$\eta_t k_t$中给予折旧补贴$\delta k_t$，因此对来自资本的租金征收税款$\tau_{kt} (\eta_t - \delta) k_t$。

### 政府

政府购买计划$\{ g_t \}_{t=0}^\infty$和税收计划$\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$必须遵守预算约束

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)



给定满足{eq}`eq:gov_budget`的预算可行政府政策$\{g_t\}_{t=0}^\infty$和$\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$，

- *家庭*选择$\{c_t\}_{t=0}^\infty$、$\{n_t\}_{t=0}^\infty$和$\{k_{t+1}\}_{t=0}^\infty$，在预算约束{eq}`eq:house_budget`下最大化效用{eq}`eq:utility`，以及
- *企业*选择资本序列$\{k_t\}_{t=0}^\infty$和$\{n_t\}_{t=0}^\infty$以最大化利润

    $$
         \sum_{t=0}^\infty q_t [F(k_t, n_t) - \eta_t k_t - w_t n_t]
    $$ (eq:firm_profit)
  
- **可行配置**是满足可行性条件{eq}`eq:tech_capital`的序列$\{c_t, x_t, n_t, k_t\}_{t=0}^\infty$。

## 均衡

```{prf:definition}
:label: com_eq_tax

**带扭曲性税收的竞争均衡**是一个**预算可行的政府政策**、**可行配置**和**价格系统**，在给定价格系统和政府政策的情况下，该配置解决家庭问题和企业问题。
```

## 无套利条件

无套利论证意味着对跨时间的价格和税率的限制。


通过重新排列{eq}`eq:house_budget`并在相同的$t$处对$k_t$进行分组，我们可以得到

$$
    \begin{aligned}
    \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t \right] &\leq \sum_{t=0}^\infty q_t(1 - \tau_{nt})w_t n_t - \sum_{t=0}^\infty q_t \tau_{ht} \\
    &+ \sum_{t=1}^\infty\left\{ \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]q_t - q_{t-1}\right\}k_t \\
    &+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

家庭继承给定的$k_0$作为初始条件，并可以自由选择$\{ c_t, n_t, k_{t+1} \}_{t=0}^\infty$。

由于资源有限，家庭的预算约束{eq}`eq:house_budget`在均衡中必须是有界的。

这对价格和税收序列施加了限制。

具体来说，对于$t \geq 1$，乘以$k_t$的项必须等于零。

如果它们严格为正（负），家庭可以通过选择任意大的正（负）$k_t$来任意增加（减少）{eq}`eq:house_budget`的右侧，从而导致无限的利润或套利机会：

- 对于严格为正的项，家庭可以购买大量资本存量$k_t$，并从其租赁服务和未折旧价值中获利。

- 对于严格为负的项，家庭可以进行资本合成单位的"卖空"。这两种情况都会使{eq}`eq:house_budget`无界。

因此，通过将乘以$k_t$的项设为$0$，我们得到无套利条件：

$$
\frac{q_t}{q_{t+1}} = \left[(1 - \tau_{kt+1})(\eta_{t+1} - \delta) + 1\right].
$$ (eq:no_arb)

此外，我们有终端条件：

$$
-\lim_{T \to \infty} q_T k_{T+1} = 0.
$$ (eq:terminal)



代表性企业的零利润条件对均衡价格和数量施加了额外的限制。

企业利润的现值为

$$
\sum_{t=0}^\infty q_t \left[ F(k_t, n_t) - w_t n_t - \eta_t k_t \right].
$$

将欧拉定理应用于线性齐次函数$F(k, n)$，企业的现值为：

$$
\sum_{t=0}^\infty q_t \left[ (F_{kt} - \eta_t)k_t + (F_{nt} - w_t)n_t \right].
$$

无套利（或零利润）条件为：

$$
\eta_t = F_{kt}, \quad w_t = F_{nt}.
$$(eq:no_arb_firms)

## 家庭的一阶条件

家庭在{eq}`eq:house_budget`约束下最大化{eq}`eq:utility`。

令$U_1 = \frac{\partial U}{\partial c}, U_2 = \frac{\partial U}{\partial (1-n)} = -\frac{\partial U}{\partial n}.$，我们可以从拉格朗日函数导出一阶条件

$$
\mathcal{L} = \sum_{t=0}^\infty \beta^t U(c_t, 1 - n_t) + \mu \left( \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t - (1 - \tau_{nt})w_t n_t + \ldots \right] \right),
$$

代表性家庭问题的一阶必要条件为

$$
\frac{\partial \mathcal{L}}{\partial c_t} = \beta^t U_{1}(c_t, 1 - n_t) - \mu q_t (1 + \tau_{ct}) = 0
$$ (eq:foc_c_1)

和

$$
\frac{\partial \mathcal{L}}{\partial n_t} = \beta^t \left(-U_{2t}(c_t, 1 - n_t)\right) - \mu q_t (1 - \tau_{nt}) w_t = 0
$$ (eq:foc_n_1)

重新排列{eq}`eq:foc_c_1`和{eq}`eq:foc_n_1`，我们有

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


将{eq}`eq:foc_c`代入{eq}`eq:terminal`并替换$q_t$，我们得到终端条件

$$
-\lim_{T \to \infty} \beta^T \frac{U_{1T}}{(1 + \tau_{cT})} k_{T+1} = 0.
$$ (eq:terminal_final)

## 计算均衡

为了计算均衡，我们寻求一个价格系统$\{q_t, \eta_t, w_t\}$、一个预算可行的政府政策$\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$，以及一个配置$\{c_t, n_t, k_{t+1}\}$，它们求解由以下组成的非线性差分方程系统：

- 可行性条件{eq}`eq:tech_capital`，家庭的无套利条件{eq}`eq:no_arb`和企业的无套利条件{eq}`eq:no_arb_firms`，家庭的一阶条件{eq}`eq:foc_c`和{eq}`eq:foc_n`。
- 初始条件$k_0$和终端条件{eq}`eq:terminal_final`。

(cass_fiscal_shooting)=
## Python代码

我们需要以下导入

```{code-cell} ipython3
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from collections import namedtuple
from mpmath import mp, mpf
from warnings import warn

# Set the precision
mp.dps = 40
mp.pretty = True
```

我们使用`mpmath`库在射击算法中执行高精度算术，以应对由于数值不稳定性导致解发散的情况。

```{note}
在下面的函数中，我们包含了处理增长组件的例程，这将在{ref}`growth_model`部分进一步讨论。

我们在这里包含它们以避免代码重复。
```


我们设置以下参数

```{code-cell} ipython3
# Create a namedtuple to store the model parameters
Model = namedtuple("Model", 
            ["β", "γ", "δ", "α", "A"])

def create_model(β=0.95, # discount factor
                 γ=2.0,  # relative risk aversion coefficient
                 δ=0.2,  # depreciation rate
                 α=0.33, # capital share
                 A=1.0   # TFP
                 ):
    """Create a model instance."""
    return Model(β=β, γ=γ, δ=δ, α=α, A=A)

model = create_model()

# Total number of periods
S = 100
```

### 非弹性劳动供给

在本讲座中，我们考虑$U(c, 1-n) = u(c)$和$f(k) := F(k, 1)$的特殊情况。

我们用$f(k) := F(k, 1)$重写{eq}`eq:tech_capital`，

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

```{code-cell} ipython3
def next_k(k_t, g_t, c_t, model, μ_t=1):
    """
    Capital next period: k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    with optional growth adjustment: k_{t+1} = (f(k_t) + (1 - δ) * k_t - c_t - g_t) / μ_{t+1}
    """
    return (f(k_t, model) + (1 - model.δ) * k_t - g_t - c_t) / μ_t
```

根据线性齐次生产函数的性质，我们有$F_k(k, n) = f'(k)$和$F_n(k, 1) = f(k, 1) - f'(k)k$。

将{eq}`eq:foc_c`、{eq}`eq:no_arb_firms`和{eq}`eq:feasi_capital`代入{eq}`eq:no_arb`，我们得到欧拉方程

$$
\begin{aligned}
&\frac{u'(f(k_t) + (1 - \delta) k_t - g_t - k_{t+1})}{(1 + \tau_{ct})} \\
&- \beta \frac{u'(f(k_{t+1}) + (1 - \delta) k_{t+1} - g_{t+1} - k_{t+2})}{(1 + \tau_{ct+1})} \\
&\times [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1] = 0.
\end{aligned}
$$(eq:euler_house)

这可以简化为：

$$
\begin{aligned}
u'(c_t) = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned}
$$ (eq:diff_second)


方程{eq}`eq:diff_second`将在我们的均衡计算算法中占据重要地位。
 

### 稳态

税率和政府支出作为差分方程{eq}`eq:feasi_capital`和{eq}`eq:diff_second`的**强制函数**。

定义$z_t = [g_t, \tau_{kt}, \tau_{ct}]'$。

将二阶差分方程表示为：

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

我们假设政府政策达到稳态，使得$\lim_{t \to \infty} z_t = \bar z$，并且稳态对于$t > T$成立。

终端稳态资本存量$\bar{k}$满足：

$$
H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0.
$$

从差分方程{eq}`eq:diff_second`，我们可以推断出对稳态的限制：

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
    Compute q path: q_t = (β^t * u'(c_t)) / u'(c_0)
    with optional A_path for growth models.
    """
    A = np.ones_like(c_path) if A_path is None else np.asarray(A_path)
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (model.β ** t * 
                         u_prime(c_path[t], model, A[t])) / u_prime(c_path[0], model, A[0])
    return q_path
```

*资本租金率*

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100, A_path=None):
    """
    Compute η path: η_t = f'(k_t)
    with optional A_path for growth models.
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model, A[t])
    return η_path
```

*劳动租金率：*

$$
w_t = f(k_t) - k_t f'(k_t)    
$$

```{code-cell} ipython3
def compute_w_path(k_path, η_path, model, S=100, A_path=None):
    """
    Compute w path: w_t = f(k_t) - k_t * f'(k_t)
    with optional A_path for growth models.
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = f(k_path[t], model, A[t]) - k_path[t] * η_path[t]
    return w_path
```

*资本的单期总回报率：*

$$
\bar{R}_{t+1} = \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1\right] =  \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} R_{t, t+1}
$$ (eq:gross_rate)

```{code-cell} ipython3
def compute_R_bar(τ_ct, τ_ctp1, τ_ktp1, k_tp1, model):
    """
    Gross one-period return on capital:
    R_bar = [(1 + τ_c_t) / (1 + τ_c_{t+1})] 
        * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    return ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(k_tp1, model) - model.δ) + 1)
```

```{code-cell} ipython3
def compute_R_bar_path(shocks, k_path, model, S=100):
    """
    Compute R_bar path over time.
    """
    R_bar_path = np.zeros(S + 1)
    for t in range(S):
        R_bar_path[t] = compute_R_bar(
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k_path[t + 1], model)
    R_bar_path[S] = R_bar_path[S - 1]
    return R_bar_path
```

*单期贴现因子：*

$$
R^{-1}_{t, t+1} = \frac{q_{t+1}}{q_{t}} = m_{t, t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)} \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})}
$$ (eq:equil_bigR)


*净单期利率：*

$$
r_{t, t+1} \equiv R_{t, t+1} - 1 = (1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta)
$$ (eq:equil_r)

根据{eq}`eq:equil_bigR`和$r_{t, t+1} = - \ln(\frac{q_{t+1}}{q_t})$，我们有

$$
R_{t, t+s} = e^{s \cdot r_{t, t+s}}.
$$

然后根据{eq}`eq:equil_r`，我们有

$$
\frac{q_{t+s}}{q_t} = e^{-s \cdot r_{t, t+s}}.
$$

重新排列上述方程，我们有

$$
r_{t, t+s} = -\frac{1}{s} \ln\left(\frac{q_{t+s}}{q_t}\right).
$$

```{code-cell} ipython3
def compute_rts_path(q_path, S, t):
    """
    Compute r path:
    r_t,t+s = - (1/s) * ln(q_{t+s} / q_t)
    """
    s = np.arange(1, S + 1) 
    q_path = np.array([float(q) for q in q_path]) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rts_path = - np.log(q_path[t + s] / q_path[t]) / s
    return rts_path
```

## 一些函数形式

我们假设代表性家庭的期效用具有以下CRRA（常相对风险厌恶）形式

$$
u(c) = \frac{c^{1 - \gamma}}{1 - \gamma}
$$

```{code-cell} ipython3
def u_prime(c, model, A_t=1):
    """
    Marginal utility: u'(c) = c^{-γ}
    with optional technology adjustment: u'(cA) = (cA)^{-γ}
    """
    return (c * A_t) ** (-model.γ)
```

通过将{eq}`eq:gross_rate`代入{eq}`eq:diff_second`，我们得到

$$
c_{t+1} = c_t \left[ \beta \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta) + 1 \right] \right]^{\frac{1}{\gamma}} = c_t \left[ \beta \overline{R}_{t+1} \right]^{\frac{1}{\gamma}}
$$ (eq:consume_R)

```{code-cell} ipython3
def next_c(c_t, R_bar, model, μ_t=1):
    """
    Consumption next period: c_{t+1} = c_t * (β * R̄)^{1/γ}
    with optional growth adjustment: c_{t+1} = c_t * (β * R_bar)^{1/γ} * μ_{t+1}^{-1}
    """
    return c_t * (model.β * R_bar) ** (1 / model.γ) / μ_t
```

对于生产函数，我们假设Cobb-Douglas形式：

$$
F(k, 1) = A k^\alpha
$$

```{code-cell} ipython3
def f(k, model, A=1): 
    """
    Production function: f(k) = A * k^{α}
    """
    return A * k ** model.α

def f_prime(k, model, A=1):
    """
    Marginal product of capital: f'(k) = α * A * k^{α - 1}
    """
    return model.α * A * k ** (model.α - 1)
```

## 计算

我们描述了计算均衡的两种方法：

 * 射击算法
 * 残差最小化方法，专注于施加欧拉方程{eq}`eq:diff_second`和可行性条件{eq}`eq:feasi_capital`。

### 射击算法

该算法执行以下步骤。

1. 求解方程{eq}`eq:diff_second_steady`以获得对应于永久政策向量$\bar{z}$的终端稳态资本$\bar{k}$。

2. 选择一个大的时间索引$S \gg T$，猜测初始消费率$c_0$，并使用方程{eq}`eq:feasi_capital`求解$k_1$。

3. 使用方程{eq}`eq:consume_R`确定$c_{t+1}$。然后，应用方程{eq}`eq:feasi_capital`计算$k_{t+2}$。

4. 迭代步骤3以计算候选值$\hat{k}_t$，对于$t = 1, \dots, S$。

5. 计算差值$\hat{k}_S - \bar{k}$。如果对于某个小的$\epsilon$，$\left| \hat{k}_S - \bar{k} \right| > \epsilon$，调整$c_0$并重复步骤2-5。

6. 使用二分法迭代调整$c_0$，找到确保$\left| \hat{k}_S - \bar{k} \right| < \epsilon$的值。

以下代码实现了这些步骤。

```{code-cell} ipython3
# Steady-state calculation
def steady_states(model, g_ss, τ_k_ss=0.0, μ_ss=None):
    """
    Calculate steady state values for capital and 
    consumption with optional A_path for growth models.
    """

    β, δ, α, γ = model.β, model.δ, model.α, model.γ

    A = model.A or 1.0

    # growth‐adjustment in the numerator: μ^γ or 1
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
    Shooting algorithm for given initial c0 and k0
    with optional A_path for growth models.
    """
    # unpack & mpf‐ify shocks, fill μ with ones if missing
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
    Bisection method to find initial c0
    """
    # steady‐state uses last shocks (μ=1 if missing)
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
                print(f"[{i}] shoot failed at c0={c0}")
            c0_hi = c0
        else:
            err = k_path[-1] - k_ss_fin
            if verbose and i % 100 == 0:
                print(f"[{i}] c0={c0}, err={err}")
            if abs(err) < tol:
                if verbose:
                    print(f"Converged after {i} iter")
                return c0
            # update bounds in one line
            c0_lo, c0_hi = (c0, c0_hi) if err > 0 else (c0_lo, c0)
        c0 = (c0_lo + c0_hi) / mpf('2')

    warn(f"bisection did not converge after {max_iter} iters; returning c0={c0}")
    return c0


def run_shooting(
    shocks, S, model, A_path=None, 
    c0_finder=bisection_c0, shooter=shooting_algorithm):
    """
    Compute initial SS, find c0, and return [k,c] paths
    with optional A_path for growth models.
    """
    # initial SS at t=0 (μ=1 if missing)
    g0    = mpf(shocks['g'][0])
    τ_k0  = mpf(shocks['τ_k'][0])
    μ0    = mpf(shocks['μ'][0]) if 'μ' in shocks else mpf('1')
    k0, c0 = steady_states(model, g0, τ_k0, μ0)

    optimal_c0 = c0_finder(c0, k0, shocks, S, model, A_path=A_path)
    print(f"Model: {model}\nOptimal initial consumption c0 = {mpf(optimal_c0)}")

    k_path, c_path = shooter(optimal_c0, k0, shocks, S, model, A_path)
    return np.column_stack([k_path, c_path])
```

(cf:experiments)=
### 实验

让我们运行一些实验。

1. 在第10期发生的$g$从0.2到0.4的可预见一次性永久性增加，
2. 在第10期发生的$\tau_c$从0.0到0.2的可预见一次性永久性增加，
3. 在第10期发生的$\tau_k$从0.0到0.2的可预见一次性永久性增加，以及
4. 在第10期发生的$g$从0.2到0.4的可预见一次性增加，之后$g$永久恢复到0.2。

+++

首先，我们准备将用于初始化迭代算法的序列。

我们将从初始稳态开始，并在指定的时间应用冲击。

```{code-cell} ipython3
def plot_results(
    solution, k_ss, c_ss, shocks, shock_param, axes, model,
    A_path=None, label='', linestyle='-', T=40):
    """
    Plot simulation results (k, c, R, η, and a policy shock)
    with optional A_path for growth models.
    """
    k_path = solution[:, 0]
    c_path = solution[:, 1]
    T = min(T, k_path.size)

    # handle growth parameters
    μ0 = shocks['μ'][0] if 'μ' in shocks else 1.0
    A0 = A_path[0] if A_path is not None else (model.A or 1.0)

    # steady‐state lines
    R_bar_ss = (1 / model.β) * (μ0**model.γ)
    η_ss     = model.α * A0 * k_ss**(model.α - 1)

    # plot k
    axes[0].plot(k_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(k_ss, linestyle='--', color='black')
    axes[0].set_title('k')

    # plot c
    axes[1].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[1].axhline(c_ss, linestyle='--', color='black')
    axes[1].set_title('c')

    # plot R bar
    S_full    = k_path.size - 1
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S_full)
    axes[2].plot(R_bar_path[:T], linestyle=linestyle, label=label)
    axes[2].axhline(R_bar_ss, linestyle='--', color='black')
    axes[2].set_title(r'$\bar{R}$')

    # plot η
    η_path = compute_η_path(k_path, model, S_full)
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')

    # plot shock
    shock_series = np.array(shocks[shock_param], dtype=object)
    axes[4].plot(shock_series[:T], linestyle=linestyle, label=label)
    axes[4].axhline(shock_series[0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')

    if label:
        for ax in axes[:5]:
            ax.legend()
```

**实验1：在第10期$g$从0.2到0.4的可预见一次性永久性增加**

下图显示了在$t = T = 10$时可预见的$g$永久性增加的后果，该增加通过一次性总额税的增加来融资

```{code-cell} ipython3
# Define shocks as a dictionary
shocks = {
    'g': np.concatenate(
        (np.repeat(0.2, 10), np.repeat(0.4, S - 9))
    ),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

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

上述图表明，均衡**消费平滑**机制正在发挥作用，这是由代表性消费者对平滑消费路径的偏好驱动的，这种偏好来自其单期效用函数的曲率。

- 资本存量的稳态值保持不变：
  - 这源于$g$从欧拉方程的稳态版本（{eq}`eq:diff_second_steady`）中消失的事实。

- 消费在时间$T$之前由于政府消费增加而开始逐渐下降：
  - 家庭减少消费以抵消政府支出，政府支出通过增加的一次性总额税融资。
  - 竞争经济通过增加一次性总额税流向家庭发出减少消费的信号。
  - 关心现值而非税收时间的家庭经历消费的不利财富效应，导致立即反应。
  
- 资本在时间$0$和$T$之间由于增加的储蓄而逐渐积累，并在时间$T$之后逐渐减少：
    - 资本存量的这种时间变化平滑了随时间的消费，由代表性消费者的消费平滑动机驱动。

让我们将上述使用的程序收集到一个函数中，该函数运行求解器并绘制给定实验的图表

```{code-cell} ipython3
:tags: [hide-input]

def experiment_model(
    shocks, S, model, A_path=None, solver=run_shooting, 
    plot_func=plot_results, policy_shock='g', T=40):
    """
    Run the shooting algorithm and plot results.
    """
    # initial steady state (μ0=None if no growth)
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    μ0   = mpf(shocks['μ'][0]) if 'μ' in shocks else None
    k_ss, c_ss = steady_states(model, g0, τk0, μ0)

    print(f"Steady-state capital: {float(k_ss):.4f}")
    print(f"Steady-state consumption: {float(c_ss):.4f}")
    print('-'*64)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    sol = solver(shocks, S, model, A_path)
    plot_func(
        sol, k_ss, c_ss, shocks, policy_shock, axes, model,
        A_path=A_path, T=T
    )

    # remove unused axes
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

下图比较了两个经济体对在$t = 10$时可预见的$g$增加的反应：
 
 *  我们的原始经济体，$\gamma = 2$，显示为实线，以及
 *  其他方面相同但$\gamma = 0.2$的经济体。

这种比较引起我们的兴趣，因为效用曲率参数$\gamma$控制家庭跨时间替代消费的意愿，从而控制其对随时间消费路径平滑度的偏好。

```{code-cell} ipython3
# Solve the model using shooting
solution = run_shooting(shocks, S, model)

# Compute the initial steady states
k_ss_initial, c_ss_initial = steady_states(model, 
                                           shocks['g'][0], 
                                           shocks['τ_k'][0])

# Plot the solution for γ=2
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

label = fr"$\gamma = {model.γ}$"
plot_results(solution, k_ss_initial, c_ss_initial, 
             shocks, 'g', axes, model, label=label, 
             T=40)

# Solve and plot the result for γ=0.2
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

结果表明，降低$\gamma$会影响消费和资本存量路径，因为它增加了代表性消费者跨时间替代消费的意愿：

- 消费路径：
  - 当$\gamma = 0.2$时，与$\gamma = 2$相比，消费变得不那么平滑。
  - 对于$\gamma = 0.2$，消费更密切地反映政府支出路径，在$t = 10$之前保持更高水平。

- 资本存量路径：
  - 当$\gamma = 0.2$时，资本存量的积累和减少较小。
  - $\bar{R}$和$\eta$的波动也较小。

让我们编写另一个函数来运行求解器并绘制这两个实验的图表

```{code-cell} ipython3
:tags: [hide-input]

def experiment_two_models(
    shocks, S, model_1, model_2, solver=run_shooting, plot_func=plot_results, 
    policy_shock='g', legend_label_fun=None, T=40, A_path=None):
    """
    Compare and plot the shooting algorithm paths for two models.
    """
    is_growth = 'μ' in shocks
    μ0 = mpf(shocks['μ'][0]) if is_growth else None

    # initial steady states for both models
    g0   = mpf(shocks['g'][0])
    τk0  = mpf(shocks['τ_k'][0])
    k_ss1, c_ss1 = steady_states(model_1, g0, τk0, μ0)
    k_ss2, c_ss2 = steady_states(model_2, g0, τk0, μ0)

    # print both    
    print(f"Model 1 (γ={model_1.γ}): steady state k={float(k_ss1):.4f}, c={float(c_ss1):.4f}")
    print(f"Model 2 (γ={model_2.γ}): steady state k={float(k_ss2):.4f}, c={float(c_ss2):.4f}")
    print('-'*64)

    # default legend labels
    if legend_label_fun is None:
        legend_label_fun = lambda m: fr"$\gamma = {m.γ}$"

    # prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    # loop over (model, steady‐state, linestyle)
    for model, (k_ss, c_ss), ls in [
        (model_1, (k_ss1, c_ss1), '-'),
        (model_2, (k_ss2, c_ss2), '-.')
    ]:
        sol = solver(shocks, S, model, A_path)
        plot_func(sol, k_ss, c_ss, shocks, policy_shock, axes, 
                  model, A_path=A_path, 
                  label=legend_label_fun(model), 
                  linestyle=ls, T=T)

    # shared legend in lower‐right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower right', ncol=2, 
        fontsize=12, bbox_to_anchor=(1, 0.1))

    # drop the unused subplot
    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```

现在我们绘制其他均衡数量：

```{code-cell} ipython3
def plot_prices(solution, c_ss, shock_param, axes,
                model, label='', linestyle='-', T=40):
    """
    Compares and plots prices
    """
    α, β, δ, γ, A = model.α, model.β, model.δ, model.γ, model.A
    
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    # Plot for c
    axes[0].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(c_ss, linestyle='--', color='black')
    axes[0].set_title('c')
    
    # Plot for q
    q_path = compute_q_path(c_path, model, S=S)
    axes[1].plot(q_path[:T], linestyle=linestyle, label=label)
    axes[1].plot(β**np.arange(T), linestyle='--', color='black')
    axes[1].set_title('q')
    
    # Plot for r_{t,t+1}
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)
    axes[2].plot(R_bar_path[:T] - 1, linestyle=linestyle, label=label)
    axes[2].axhline(1 / β - 1, linestyle='--', color='black')
    axes[2].set_title('$r_{t,t+1}$')

    # Plot for r_{t,t+s}
    for style, s in zip(['-', '-.', '--'], [0, 10, 60]):
        rts_path = compute_rts_path(q_path, T, s)
        axes[3].plot(rts_path, linestyle=style, 
                     color='black' if style == '--' else None,
                     label=f'$t={s}$')
        axes[3].set_xlabel('s')
        axes[3].set_title('$r_{t,t+s}$')

    # Plot for g
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(shock_param)
```

对于$\gamma = 2$，下图描述了$q_t$和利率期限结构对在$t = 10$时可预见的$g_t$增加的反应

```{code-cell} ipython3
solution = run_shooting(shocks, S, model)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

plot_prices(solution, c_ss_initial, 'g', axes, model, T=40)

for ax in axes[5:]:
    fig.delaxes(ax)

handles, labels = axes[3].get_legend_handles_labels()  
fig.legend(handles, labels, title=r"$r_{t,t+s}$ with ", loc='lower right', 
           ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```

顶部的第二个面板比较了初始稳态的$q_t$与在$t = 0$预见$g$增加后的$q_t$，而第三个面板比较了隐含的短期利率$r_t$。

第四个面板显示了$t=0$、$t=10$和$t=60$时的利率期限结构。

注意，到$t = 60$时，系统已经收敛到新的稳态，利率期限结构变得平坦。

在$t = 10$时，利率期限结构向上倾斜。

这种向上倾斜反映了随时间消费增长率的预期增加，如消费面板所示。

在$t = 0$时，利率期限结构呈现"U形"模式：

- 它下降直到到期日$s = 10$。
- 在$s = 10$之后，对于更长的到期日，它增加。
    
这种模式与前两个图中消费增长的模式一致，消费增长在$t = 10$之前以递增的速度下降，然后以递减的速度下降。

+++

**实验2：在第10期$\tau_c$从0.0到0.2的可预见一次性永久性增加**

在非弹性劳动供给下，欧拉方程{eq}`eq:euler_house`和其他均衡条件表明
- 恒定的消费税不会扭曲决策，但
- 对它们的预期变化会。

实际上，{eq}`eq:euler_house`或{eq}`eq:diff_second`表明，可预见的$\tau_{ct}$增加（即$(1+\tau_{ct})$$(1+\tau_{ct+1})$的减少）的作用类似于$\tau_{kt}$的增加。

下图描绘了对可预见的消费税$\tau_c$增加的反应。

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

显然，上述图中的所有变量最终都回到了它们的初始稳态值。

对$\tau_{ct}$增加的预期导致消费和资本存量跨时间的变化：

- 在$t = 0$时：
    - 对$\tau_c$增加的预期导致*消费立即跳跃*。
    - 随后出现*消费狂欢*，使资本存量下降直到$t = T = 10$。
- 在$t = 0$和$t = T = 10$之间：
    - 资本存量的下降随时间推高$\bar{R}$。
    - 均衡条件要求消费增长率上升直到$t = T$。
- 在$t = T = 10$时：
    - $\tau_c$的跳跃使$\bar{R}$压低到$1$以下，导致*消费急剧下降*。
- 在$T = 10$之后：
    - 预期扭曲的影响结束，经济逐渐适应较低的资本存量。
    - 资本现在必须上升，需要*紧缩* —消费在$t = T$后暴跌，由较低的消费水平表明。
    - 利率逐渐下降，消费沿着通往终端稳态的路径以递减的速度增长。

+++

**实验3：在第10期$\tau_k$从0.0到0.2的可预见一次性永久性增加**

对于两个$\gamma$值2和0.2，下图显示了对在$t = T = 10$时可预见的$\tau_{kt}$永久跳跃的反应。

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

政府支出路径保持固定
- $\tau_{kt}$的增加被一次性总额税现值的减少所抵消，以保持预算平衡。

图表明：

- 对$\tau_{kt}$增加的预期导致资本存量由于当前消费增加和消费流增长而立即下降。
- $\bar{R}$在$t = 0$时开始上升，在$t = 9$时达到峰值，在$t = 10$时，$\bar{R}$由于税收变化而急剧下降。
    - $\bar{R}$的变化与$t = 10$税收增加对跨时间消费的影响一致。
- 过渡动态将$k_t$（资本存量）推向新的较低稳态水平。在新的稳态中：
    - 由于较低资本存量的产出减少，消费较低。
    - 当$\gamma = 2$时，消费路径比$\gamma = 0.2$时更平滑。

+++

到目前为止，我们已经探讨了政府政策可预见的一次性永久性变化的后果。接下来我们描述一些实验，其中政策变量有可预见的一次性变化（"脉冲"）。

**实验4：在第10期$g$从0.2到0.4的可预见一次性增加，之后$g$永久恢复到0.2**

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

图表明：

- 消费：
    - 在政策宣布时立即下降，并随时间继续下降，以预期$g$的一次性激增。
    - 在$t = 10$时的冲击后，消费开始恢复，以递减的速度朝其稳态值上升。
    
- 资本和$\bar{R}$：
    - 在$t = 10$之前，资本积累，因为利率变化诱使家庭为预期的政府支出增加做准备。
    - 在$t = 10$时，由于政府消费了其中的一部分，资本存量急剧减少。
    - 由于资本减少，$\bar{R}$跳到其稳态值以上，然后逐渐下降到其稳态水平。

+++

### 方法2：残差最小化

第二种方法涉及最小化以下方程的残差（即与等式的偏差）：

- 欧拉方程{eq}`eq:diff_second`：

  $$
  1 = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma} \frac{(1+\tau_{ct})}{(1+\tau_{ct+1})} \left[(1 - \tau_{kt+1})(\alpha A k_{t+1}^{\alpha-1} - \delta) + 1 \right]
  $$

- 可行性条件{eq}`eq:feasi_capital`：

  $$
  k_{t+1} = A k_{t}^{\alpha} + (1 - \delta) k_t - g_t - c_t.
  $$

```{code-cell} ipython3
# Euler's equation and feasibility condition 
def euler_residual(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model, μ_tp1=1):
    """
    Computes the residuals for Euler's equation 
    with optional growth model parameters μ_tp1.
    """
    R_bar = compute_R_bar(τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model)
    
    c_expected = next_c(c_t, R_bar, model, μ_tp1)

    return c_expected / c_tp1 - 1.0

def feasi_residual(k_t, k_tm1, c_tm1, g_t, model, μ_t=1):
    """
    Computes the residuals for feasibility condition 
    with optional growth model parameter μ_t.
    """
    k_t_expected = next_k(k_tm1, g_t, c_tm1, model, μ_t)
    return k_t_expected - k_t
```

算法如下进行：

1. 根据$t=0$时的政府计划找到初始稳态$k_0$。

2. 初始化初始猜测序列$\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$。

3. 计算$t = 0, \dots, S$的残差$l_a$和$l_k$，以及$t = 0$的$l_{k_0}$和$t = S$的$l_{k_S}$：
   - 使用{eq}`eq:diff_second`计算$t = 0, \dots, S$的欧拉方程残差：

     $$
     l_{ta} = \beta u'(c_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] - 1
     $$

   - 使用{eq}`eq:feasi_capital`计算$t = 1, \dots, S-1$的可行性条件残差：

     $$
     l_{tk} = k_{t+1} - f(k_t) - (1 - \delta)k_t + g_t + c_t
     $$

   - 使用{eq}`eq:diff_second_steady`和初始资本$k_0$计算$k_0$的初始条件残差：

     $$
     l_{k_0} = 1 - \beta \left[ (1 - \tau_{k0}) \left(f'(k_0) - \delta \right) + 1 \right]
     $$

   - 在假设$c_t = c_{t+1} = c_S$、$k_t = k_{t+1} = k_S$、$\tau_{ct} = \tau_{ct+1} = \tau_{c_s}$和$\tau_{kt} = \tau_{kt+1} = \tau_{k_s}$的情况下，使用{eq}`eq:diff_second`计算$t = S$的终端条件残差：
     
     $$
     l_{k_S} = \beta u'(c_S) \frac{(1 + \tau_{c_s})}{(1 + \tau_{c_s})} \left[(1 - \tau_{k_s})(f'(k_S) - \delta) + 1 \right] - 1
     $$

4. 迭代调整$\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$的猜测，以最小化$t = 0, \dots, S$的残差$l_{k_0}$、$l_{ta}$、$l_{tk}$和$l_{k_S}$。

```{code-cell} ipython3
def compute_residuals(vars_flat, k_init, S, shock_paths, model):
    """
    Compute the residuals for the Euler equation and feasibility condition.
    """
    g, τ_c, τ_k, μ = (shock_paths[key] for key in ('g','τ_c','τ_k','μ'))
    k, c = vars_flat.reshape((S+1, 2)).T
    res = np.empty(2*S+2, dtype=float)

    # boundary condition on initial capital
    res[0] = k[0] - k_init

    # interior Euler and feasibility
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

    # terminal Euler condition at t=S
    res[-1] = euler_residual(
        c[S],   c[S],
        τ_c[S], τ_c[S],
        τ_k[S], k[S],
        model,
        μ[S])

    return res


def run_min(shocks, S, model, A_path=None):
    """
    Solve for the full (k,c) path by root‐finding the residuals.
    """
    shocks['μ'] = shocks['μ'] if 'μ' in shocks else np.ones_like(shocks['g'])

    # compute the steady‐state to serve as both initial capital and guess
    k_ss, c_ss = steady_states(
        model,
        shocks['g'][0],
        shocks['τ_k'][0],
        shocks['μ'][0]  # =1 if no growth
    )

    # initial guess: flat at the steady‐state
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

我们发现方法2没有遇到数值稳定性问题，因此不需要使用`mp.mpf`。

我们将使用第二种方法复制一些实验作为练习。

```{exercise}
:label: cass_fiscal_ex1

使用残差最小化的第二种方法复制我们四个实验的图表：
1. 在第10期发生的$g$从0.2到0.4的可预见一次性永久性增加，
2. 在第10期发生的$\tau_c$从0.0到0.2的可预见一次性永久性增加，
3. 在第10期发生的$\tau_k$从0.0到0.2的可预见一次性永久性增加，以及
4. 在第10期发生的$g$从0.2到0.4的可预见一次性增加，之后$g$永久恢复到0.2，
```

```{solution-start} cass_fiscal_ex1
:class: dropdown
```

这是一个解决方案：

**实验1：在第10期$g$从0.2到0.4的可预见一次性永久性增加**

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

**实验2：在第10期$\tau_c$从0.0到0.2的可预见一次性永久性增加。**

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

**实验3：在第10期$\tau_k$从0.0到0.2的可预见一次性永久性增加。**

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

**实验4：在第10期$g$从0.2到0.4的可预见一次性增加，之后$g$永久恢复到0.2**

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

设计一个新实验，其中政府支出$g$在第$10$期从$0.2$增加到$0.4$，
然后在第$20$期永久减少到$0.1$。
```

```{solution-start} cass_fiscal_ex2
:class: dropdown
```

这是一个解决方案：

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

在前面的部分中，我们考虑了没有外生增长的模型。

我们通过为所有$t$设置$A_t = 1$，将生产函数中的项$A_t$设置为常数。

现在我们准备考虑增长。

为了纳入增长，我们修改生产函数为

$$
Y_t = F(K_t, A_tn_t)
$$

其中$Y_t$是总产出，$N_t$是总就业，$A_t$是劳动增强型技术变化，
$F(K, AN)$是与之前相同的线性齐次生产函数。

我们假设$A_t$遵循过程

$$
A_{t+1} = \mu_{t+1}A_t
$$ (eq:growth)

并且$\mu_{t+1}=\bar{\mu}>1$。

```{code-cell} ipython3
# Set the constant A parameter to None
model = create_model(A=None)
```

```{code-cell} ipython3
def compute_A_path(A0, shocks, S=100):
    """
    Compute A path over time.
    """
    A_path = np.full(S + 1, A0)
    for t in range(1, S + 1):
        A_path[t] = A_path[t-1] * shocks['μ'][t-1]
    return A_path
```

### 非弹性劳动供给

通过线性齐次性，生产函数可以表示为

$$
y_t=f(k_t)
$$

其中$f(k)=F(k,1) = k^\alpha$，$k_t=\frac{K_t}{n_tA_t}$，$y_t=\frac{Y_t}{n_tA_t}$。

$k_t$和$y_t$按"有效劳动"$A_tn_t$的单位度量。

我们还让$c_t=\frac{C_t}{A_tn_t}$和$g_t=\frac{G_t}{A_tn_t}$，其中$C_t$和$G_t$是总消费和总政府支出。

我们继续考虑非弹性劳动供给的情况。

基于此，可行性可以通过方程{eq}`eq:feasi_capital`的以下修改版本来总结：

$$
k_{t+1}=\mu_{t+1}^{-1}[f(k_t)+(1-\delta)k_t-g_t-c_t]
$$ (eq:feasi_mod)


同样，根据线性齐次生产函数的性质，我们有

$$ 
\eta_t = F_k(k_t, 1) = f'(k_t), w_t = F_n(k_t, 1) = f(k_t) - f'(k_t)k_t 
$$

由于人均消费现在是$c_tA_t$，欧拉方程{eq}`eq:diff_second`的对应项是

$$
\begin{aligned}
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1}) \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} [(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1].
\end{aligned} 
$$ (eq:diff_mod)

$\bar{R}_{t+1}$继续由{eq}`eq:gross_rate`定义，除了现在$k_t$是每有效劳动单位的资本。

因此，代入{eq}`eq:gross_rate`，{eq}`eq:diff_mod`变为

$$
u'(c_tA_t) = \beta u'(c_{t+1}A_{t+1})\bar{R}_{t+1}
$$

假设家庭的效用函数与之前相同，我们有

$$
(c_tA_t)^{-\gamma} = \beta (c_{t+1}A_{t+1})^{-\gamma} \bar{R}_{t+1}
$$

因此，{eq}`eq:consume_R`的对应项是

$$
c_{t+1} = c_t \left[ \beta \bar{R}_{t+1} \right]^{\frac{1}{\gamma}}\mu_{t+1}^{-1}
$$ (eq:consume_r_mod)

### 稳态

在稳态中，$c_{t+1} = c_t$。然后{eq}`eq:diff_mod`变为

$$
1=\mu^{-\gamma}\beta[(1-\tau_k)(f'(k)-\delta)+1] 
$$ (eq:diff_mod_st)

由此我们可以计算每有效劳动单位的稳态资本水平满足

$$
f'(k)=\delta + (\frac{\frac{1}{\beta}\mu^{\gamma}-1}{1-\tau_k})
$$ (eq:cap_mod_st)  

以及

$$
\bar{R}=\frac{\mu^{\gamma}}{\beta}
$$ (eq:Rbar_mod_st)

每有效劳动单位的稳态消费水平可以使用{eq}`eq:feasi_mod`找到：

$$
c = f(k)+(1-\delta-\mu)k-g
$$

由于算法和绘图例程与之前相同，我们在{ref}`cass_fiscal_shooting`部分包含稳态计算和射击例程。

### 射击算法

现在我们可以应用射击算法来计算均衡。我们通过包括$\mu_t$来增加冲击变量向量，然后按之前的方式进行。

### 实验

让我们运行一些实验：

1. 在第10期$\mu$从1.02到1.025的可预见一次性永久性增加
2. 在第0期$\mu$到1.025的不可预见一次性永久性增加

+++

#### 实验1：在t=10时$\mu$从1.02到1.025的可预见增加

下图显示了在t=10时生产率增长$\mu$从1.02到1.025的永久增加的影响。

它们现在以有效劳动单位度量$c$和$k$。

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

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

# Run the shooting algorithm with the A_path parameter
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

图中的结果主要由{eq}`eq:diff_mod_st`驱动，
并暗示$\mu$的永久增加将导致每有效劳动单位的资本稳态值下降。

图表明：

- 随着资本变得更有效，即使较少的资本，人均消费也可以提高。
- 消费平滑驱动*消费立即跳跃*，以预期$\mu$的增加。
- 资本生产率的提高导致总回报$\bar R$的增加。
- 完全预见使资本增长增加的影响先于它发生，效果在$t=0$时可见。

#### 实验2：在t=0时$\mu$从1.02到1.025的不可预见增加

下图显示了在t=0时$\mu$立即跳到1.025的影响。

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

print(f"Steady-state capital: {k_ss_initial:.4f}")
print(f"Steady-state consumption: {c_ss_initial:.4f}")

# Run the shooting algorithm with the A_path parameter
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

同样，我们可以将上面使用的程序收集到一个函数中，该函数运行求解器并绘制给定实验的图表。

```{code-cell} ipython3
def experiment_model(shocks, S, model, A_path, solver, plot_func, policy_shock, T=40):
    """
    Run the shooting algorithm given a model and plot the results.
    """
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0], shocks['μ'][0])
    
    print(f"Steady-state capital: {k0:.4f}")
    print(f"Steady-state consumption: {c0:.4f}")
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
- 税后总回报$\bar{R}$再次与消费增长率共同移动，验证了欧拉方程{eq}`eq:diff_mod_st`。

```{exercise}
:label: cass_fiscal_ex3

使用残差最小化的第二种方法复制我们两个实验的图表：
1. 在t=10时$\mu$从1.02到1.025的可预见增加
2. 在t=0时$\mu$从1.02到1.025的不可预见增加
```

```{solution-start} cass_fiscal_ex3
:class: dropdown
```

这是一个解决方案：

**实验1：在$t=10$时$\mu$从1.02到1.025的可预见增加**

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

**实验2：在$t=0$时$\mu$从1.02到1.025的不可预见增加**

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

在这个续集{doc}`cass_fiscal_2`中，我们研究了我们单国模型的双国版本，该版本与{cite:t}`mendoza1998international`密切相关。
