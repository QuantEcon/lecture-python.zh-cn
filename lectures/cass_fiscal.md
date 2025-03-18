---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 带扭曲性税收的Cass-Koopmans模型

## 概述

本讲座研究在Cass-Koopmans增长模型的非随机版本中，可预见的财政和技术冲击对竞争均衡价格和数量的影响。该模型的特征在QuantEcon讲座{doc}`cass_koopmans_2`中有所描述。

我们将该模型作为一个实验室，用来试验近似均衡的数值技术，并展示决策者对未来政府决策具有完全预见性的动态模型的结构。
遵循Robert E. Hall的一篇经典论文{cite}`hall1971dynamic`,我们在Cass-Koopmans最优增长模型的非随机版本基础上,增加了一个政府部门,该政府购买商品流并通过多个扭曲性统一税率的序列为其支出提供资金。

扭曲性税收使竞争均衡配置无法解决规划问题。

因此,为了计算均衡配置和价格体系,我们需要求解由决策者的一阶条件和其他均衡条件组成的非线性差分方程组。

我们提出两种近似均衡的方法:

- 第一种是类似于我们在{doc}`cass_koopmans_2`中使用的射击算法。

- 第二种方法是一个寻根算法,用于最小化消费者和代表性企业一阶条件的残差。


## 经济模型


### 技术

可行配置满足

$$
g_t + c_t + x_t \leq F(k_t, n_t),
$$ (eq:tech_capital)

其中

- $g_t$ 是t时期的政府购买
- $x_t$ 是总投资，且
- $F(k_t, n_t)$ 是一个线性齐次的生产函数，其中资本$k_t$和劳动$n_t$具有正的且递减的边际产出。

物质资本按照以下方式演变：

$$
k_{t+1} = (1 - \delta)k_t + x_t,
$$

其中$\delta \in (0, 1)$是折旧率。

有时将$x_t$从{eq}`eq:tech_capital`中消除会比较方便，可以表示为：

$$
g_t + c_t + k_{t+1} \leq F(k_t, n_t) + (1 - \delta)k_t.
$$

### 竞争均衡的组成部分

所有交易在0时期发生。

代表性家庭拥有资本，做出投资决策，并将资本和劳动出租给代表性生产企业。

代表性企业使用资本和劳动，通过生产函数$F(k_t, n_t)$生产商品。
**价格体系**是一个由序列 $\{q_t, \eta_t, w_t\}_{t=0}^\infty$ 组成的三元组，其中：

- $q_t$ 是在时间 $0$ 时购买时间 $t$ 的一单位投资或消费（$x_t$ 或 $c_t$）的税前价格，
- $\eta_t$ 是家庭在时间 $t$ 从企业租赁资本所获得的税前价格，
- $w_t$ 是家庭在时间 $t$ 向企业出租劳动力所获得的税前价格。

价格 $w_t$ 和 $\eta_t$ 是以时间 $t$ 的商品为单位表示的，而 $q_t$ 是以时间 $0$ 的计价单位表示的，这与{doc}`cass_koopmans_2`中的情况相同。

政府的存在使得本讲座与{doc}`cass_koopmans_2`有所不同。

政府在时间 $t$ 的商品购买量为 $g_t \geq 0$。

政府支出计划是一个序列 $g = \{g_t\}_{t=0}^\infty$。

政府税收计划是一个由4个序列组成的四元组 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$，
其中：

- $\tau_{ct}$ 是时间 $t$ 的消费税率，
- $\tau_{kt}$ 是在时间 $t$ 对资本租赁的税率，
- $\tau_{nt}$ 是在时间 $t$ 对工资收入的税率，
- $\tau_{ht}$ 是在时间 $t$ 对消费者的一次性总付税。

由于可以使用一次性总付税 $\tau_{ht}$，政府实际上不应该使用任何扭曲性税收。

尽管如此，我们仍然包含所有这些税收，因为像 {cite}`hall1971dynamic` 一样，它们让我们能够分析各种税收如何扭曲生产和消费决策。

在[实验部分](cf:experiments)，我们将看到政府税收计划的变化如何影响转型路径和均衡。

### 代表性家庭

代表性家庭对单一消费品 $c_t$ 和闲暇 $1-n_t$ 的非负序列具有偏好，这些偏好由以下方式排序：

$$
\sum_{t=0}^{\infty} \beta^t U(c_t, 1-n_t), \quad \beta \in (0, 1),
$$ (eq:utility)

其中
- $U$ 对 $c_t$ 严格递增，二次连续可微，且严格凹，其中 $c_t \geq 0$ 且 $n_t \in [0, 1]$。

代表性家庭在单一预算约束下最大化{eq}`eq:utility`：

$$
\begin{aligned}
    \sum_{t=0}^\infty& q_t \left\{ (1 + \tau_{ct})c_t + \underbrace{[k_{t+1} - (1 - \delta)k_t]}_{\text{投资时无税}} \right\} \\
    &\leq \sum_{t=0}^\infty q_t \left\{ \eta_t k_t - \underbrace{\tau_{kt}(\eta_t - \delta)k_t}_{\text{租金收益税}} + (1 - \tau_{nt})w_t n_t - \tau_{ht} \right\}.
\end{aligned}
$$ (eq:house_budget)

这里我们假设政府从资本总租金 $\eta_t k_t$ 中给予折旧补贴 $\delta k_t$，因此对资本租金征收 $\tau_{kt} (\eta_t - \delta) k_t$ 的税收。

### 政府
政府计划 $\{ g_t \}_{t=0}^\infty$ 对于政府购买和税收 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$ 必须遵守预算约束

$$
\sum_{t=0}^\infty q_t g_t \leq \sum_{t=0}^\infty q_t \left\{ \tau_{ct}c_t + \tau_{kt}(\eta_t - \delta)k_t + \tau_{nt}w_t n_t + \tau_{ht} \right\}.
$$ (eq:gov_budget)

给定一个满足预算可行的政府政策 $\{g_t\}_{t=0}^\infty$ 和 $\{\tau_{ct}, \tau_{kt}, \tau_{nt}, \tau_{ht}\}_{t=0}^\infty$，需要遵守{eq}`eq:gov_budget`，

- *家庭*选择 $\{c_t\}_{t=0}^\infty$、$\{n_t\}_{t=0}^\infty$ 和 $\{k_{t+1}\}_{t=0}^\infty$ 以最大化效用{eq}`eq:utility`，同时受到预算约束{eq}`eq:house_budget`，以及
- *企业*选择资本序列 $\{k_t\}_{t=0}^\infty$ 和 $\{n_t\}_{t=0}^\infty$ 以最大化利润

    $$
         \sum_{t=0}^\infty q_t [F(k_t, n_t) - \eta_t k_t - w_t n_t]
    $$ (eq:firm_profit)
- **可行配置**是满足可行性条件{eq}`eq:tech_capital`的序列$\{c_t, x_t, n_t, k_t\}_{t=0}^\infty$。

## 均衡

```{prf:definition}
:label: com_eq_tax

**带扭曲性税收的竞争均衡**是指**预算可行的政府政策**、**可行配置**和**价格体系**，在给定价格体系和政府政策的情况下，该配置能够解决家庭问题和企业问题。
```

## 无套利条件

无套利论证意味着对不同时期的价格和税率有所限制。

通过重新排列{eq}`eq:house_budget`并将同一时期$t$的$k_t$项组合在一起，我们可以得到

$$
    \begin{aligned}
    \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t \right] &\leq \sum_{t=0}^\infty q_t(1 - \tau_{nt})w_t n_t - \sum_{t=0}^\infty q_t \tau_{ht} \\
    &+ \sum_{t=1}^\infty\left\{ \left[(1 - \tau_{kt})(\eta_t - \delta) + 1\right]q_t - q_{t-1}\right\}k_t \\
&+ \left[(1 - \tau_{k0})(\eta_0 - \delta) + 1\right]q_0k_0 - \lim_{T \to \infty} q_T k_{T+1}
    \end{aligned}
$$ (eq:constrant_house)

家庭继承了一个给定的$k_0$作为初始条件，并可以自由选择$\{ c_t, n_t, k_{t+1} \}_{t=0}^\infty$。

由于资源有限，家庭预算约束{eq}`eq:house_budget`在均衡状态下必须是有界的。

这对价格和税收序列施加了限制。

具体来说，对于$t \geq 1$，乘以$k_t$的项必须等于零。

如果这些项严格为正（负），家庭可以通过选择任意大的正（负）$k_t$来任意增加（减少）{eq}`eq:house_budget`的右侧，从而导致无界利润或套利机会：

- 对于严格为正的项，家庭可以购买大量资本存量$k_t$并从其租赁服务和未折旧价值中获利。
- 对于严格负项，家庭可以对合成资本单位进行"卖空"。这两种情况都会使{eq}`eq:house_budget`无界。

因此，通过将$k_t$的系数项设为$0$，我们得到无套利条件：

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

将线性齐次函数的欧拉定理应用于$F(k, n)$，企业的现值为：

$$
\sum_{t=0}^\infty q_t \left[ (F_{kt} - \eta_t)k_t + (F_{nt} - w_t)n_t \right].
$$

无套利（或零利润）条件为：

$$
\eta_t = F_{kt}, \quad w_t = F_{nt}.
$$(eq:no_arb_firms)
## 家庭的一阶条件

家庭在{eq}`eq:house_budget`约束下最大化{eq}`eq:utility`。

令$U_1 = \frac{\partial U}{\partial c}, U_2 = \frac{\partial U}{\partial (1-n)} = -\frac{\partial U}{\partial n}$，我们可以从拉格朗日函数推导出一阶条件

$$
\mathcal{L} = \sum_{t=0}^\infty \beta^t U(c_t, 1 - n_t) + \mu \left( \sum_{t=0}^\infty q_t \left[(1 + \tau_{ct})c_t - (1 - \tau_{nt})w_t n_t + \ldots \right] \right),
$$

代表性家庭问题的一阶必要条件是

$$
\frac{\partial \mathcal{L}}{\partial c_t} = \beta^t U_{1}(c_t, 1 - n_t) - \mu q_t (1 + \tau_{ct}) = 0
$$ (eq:foc_c_1)

和

$$
\frac{\partial \mathcal{L}}{\partial n_t} = \beta^t \left(-U_{2t}(c_t, 1 - n_t)\right) - \mu q_t (1 - \tau_{nt}) w_t = 0
$$ (eq:foc_n_1)

重新整理{eq}`eq:foc_c_1`和{eq}`eq:foc_n_1`，我们得到

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


将{eq}`eq:foc_c`代入{eq}`eq:terminal`并替换$q_t$,我们得到终端条件

$$
-\lim_{T \to \infty} \beta^T \frac{U_{1T}}{(1 + \tau_{cT})} k_{T+1} = 0.
$$ (eq:terminal_final)

## 计算均衡

为了计算均衡,我们需要寻找一个价格系统$\{q_t, \eta_t, w_t\}$、一个预算可行的政府政策$\{g_t, \tau_t\} \equiv \{g_t, \tau_{ct}, \tau_{nt}, \tau_{kt}, \tau_{ht}\}$以及一个配置$\{c_t, n_t, k_{t+1}\}$,它们能够解决由以下组成的非线性差分方程系统:

- 可行性条件{eq}`eq:tech_capital`、家庭无套利条件{eq}`eq:no_arb`和企业无套利条件{eq}`eq:no_arb_firms`、家庭的一阶条件{eq}`eq:foc_c`和{eq}`eq:foc_n`。
- 初始条件$k_0$和终端条件{eq}`eq:terminal_final`。


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

# 设置精度
mp.dps = 40
mp.pretty = True
```
在解决方案由于数值不稳定而发散的情况下，我们使用`mpmath`库来执行高精度运算。

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

在本讲中，我们考虑特殊情况，其中$U(c, 1-n) = u(c)$且$f(k) := F(k, 1)$。

我们用$f(k) := F(k, 1)$重写{eq}`eq:tech_capital`，

$$
k_{t+1} = f(k_t) + (1 - \delta) k_t - g_t - c_t.
$$ (eq:feasi_capital)

```{code-cell} ipython3
def next_k(k_t, g_t, c_t, model):
    """
    下一期资本：k_{t+1} = f(k_t) + (1 - δ) * k_t - c_t - g_t
    """
    return f(k_t, model) + (1 - model.δ) * k_t - g_t - c_t
```
根据线性齐次生产函数的性质,我们有 $F_k(k, n) = f'(k)$ 和 $F_n(k, 1) = f(k, 1) - f'(k)k$。

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
税率和政府支出作为**强制函数**作用于差分方程{eq}`eq:feasi_capital`和{eq}`eq:diff_second`。

定义$z_t = [g_t, \tau_{kt}, \tau_{ct}]'$。

将二阶差分方程表示为：

$$
H(k_t, k_{t+1}, k_{t+2}; z_t, z_{t+1}) = 0.
$$ (eq:second_ord_diff)

我们假设政府政策达到稳态，使得$\lim_{t \to \infty} z_t = \bar z$，且该稳态在$t > T$时保持。

终端稳态资本存量$\bar{k}$满足：

$$
H(\bar{k}, \bar{k}, \bar{k}, \bar{z}, \bar{z}) = 0.
$$

从差分方程{eq}`eq:diff_second`，我们可以推导出稳态的约束条件：

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
def compute_q_path(c_path, model, S=100):
    """
    计算q路径：q_t = (β^t * u'(c_t)) / u'(c_0)
    """
    q_path = np.zeros_like(c_path)
    for t in range(S):
        q_path[t] = (model.β ** t * 
                     u_prime(c_path[t], model)) / u_prime(c_path[0], model)
    return q_path
```
*资本租赁率*

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100):
    """
    计算η路径：η_t = f'(k_t)
    """
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model)
    return η_path
```
*劳动力租赁率：*

$$
w_t = f(k_t) - k_t f'(k_t)    
$$

```{code-cell} ipython3
def compute_w_path(k_path, η_path, model, S=100):
    """
    计算w路径：w_t = f(k_t) - k_t * f'(k_t)
    """
    A, α = model.A, model.α, model.δ
    w_path = np.zeros_like(k_path)
    for t in range(S):
        w_path[t] = f(k_path[t], model) - k_path[t] * η_path[t]
    return w_path
```
*资本的单期总回报率：*

$$
\bar{R}_{t+1} = \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1\right] =  \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} R_{t, t+1}
$$ (eq:gross_rate)

```{code-cell} ipython3
def compute_R_bar(τ_ct, τ_ctp1, τ_ktp1, k_tp1, model):
    """
    资本的单期总回报率：
    R̄ = [(1 + τ_c_t) / (1 + τ_c_{t+1})] 
        * { [1 - τ_k_{t+1}] * [f'(k_{t+1}) - δ] + 1 }
    """
    A, α, δ = model.A, model.α, model.δ
    return  ((1 + τ_ct) / (1 + τ_ctp1)) * (
        (1 - τ_ktp1) * (f_prime(k_tp1, model) - δ) + 1) 

def compute_R_bar_path(shocks, k_path, model, S=100):
    """
    计算随时间变化的R̄路径。
    """
    A, α, δ = model.A, model.α, model.δ
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

根据{eq}`eq:equil_bigR`和$r_{t, t+1} = - \ln(\frac{q_{t+1}}{q_t})$，我们有

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
def u_prime(c, model):
    """
    边际效用：u'(c) = c^{-γ}
    """
    return c ** (-model.γ)
```
将{eq}`eq:gross_rate`代入{eq}`eq:diff_second`，我们得到

$$
c_{t+1} = c_t \left[ \beta \frac{(1 + \tau_{ct})}{(1 + \tau_{ct+1})} \left[(1 - \tau_{k, t+1})(f'(k_{t+1}) - \delta) + 1 \right] \right]^{\frac{1}{\gamma}} = c_t \left[ \beta \overline{R}_{t+1} \right]^{\frac{1}{\gamma}}
$$ (eq:consume_R)

```{code-cell} ipython3
def next_c(c_t, R_bar, model):
    """
    下一期消费：c_{t+1} = c_t * (β * R̄)^{1/γ}
    """
    β, γ = model.β, model.γ
    return c_t * (β * R_bar) ** (1 / γ)
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
    A, α = model.A, model.α
    return A * k ** α

def f_prime(k, model):
    """
    资本的边际产出：f'(k) = α * A * k^{α - 1}
    """
    A, α = model.A, model.α
    return α * A * k ** (α - 1)
```
## 计算

我们介绍两种计算均衡的方法：

* 射击算法
* 残差最小化方法，主要关注满足欧拉方程{eq}`eq:diff_second`和可行性条件{eq}`eq:feasi_capital`。

### 射击算法

该算法包含以下步骤：

1. 求解方程{eq}`eq:diff_second_steady`以获得与永久政策向量$\bar{z}$相对应的终端稳态资本$\bar{k}$。

2. 选择一个远大于T的时间索引S（$S \gg T$），猜测一个初始消费率$c_0$，并使用方程{eq}`eq:feasi_capital`求解$k_1$。

3. 使用方程{eq}`eq:consume_R`确定$c_{t+1}$。然后，应用方程{eq}`eq:feasi_capital`计算$k_{t+2}$。

4. 重复步骤3以计算$t = 1, \dots, S$的候选值$\hat{k}_t$。

5. 计算差值$\hat{k}_S - \bar{k}$。如果对于某个小$\epsilon$，$\left| \hat{k}_S - \bar{k} \right| > \epsilon$，则调整$c_0$并重复步骤2-5。
6. 迭代调整 $c_0$ 使用二分法找到一个值，确保 $\left| \hat{k}_S - \bar{k} \right| < \epsilon$。

以下代码实现了这些步骤。

```{code-cell} ipython3
# 稳态计算
def steady_states(model, g_ss, τ_k_ss=0.0):
    """
    稳态值：
    - 资本：(1 - τ_k_ss) * [α * A * k_ss^{α - 1} - δ] = (1 / β) - 1
    - 消费：c_ss = A * k_ss^{α} - δ * k_ss - g_ss
    """
    β, δ, α, A = model.β, model.δ, model.α, model.A
    numerator = δ + (1 / β - 1) / (1 - τ_k_ss)
    denominator = α * A
    k_ss = (numerator / denominator) ** (1 / (α - 1))
    c_ss = A * k_ss ** α - δ * k_ss - g_ss
    return k_ss, c_ss

def shooting_algorithm(c0, k0, shocks, S, model):
    """
    给定初始 c0 和 k0 的射击算法。
    """
    # 将冲击转换为高精度
    g_path, τ_c_path, τ_k_path = (
        list(map(mpf, shocks[key])) for key in ['g', 'τ_c', 'τ_k']
    )

    # 用初始值初始化路径
    c_path = [mpf(c0)] + [mpf(0)] * S
    k_path = [mpf(k0)] + [mpf(0)] * S

    # 生成 k_t 和 c_t 的路径
    for t in range(S):
        k_t, c_t, g_t = k_path[t], c_path[t], g_path[t]

        # 计算下一期的资本
        k_tp1 = next_k(k_t, g_t, c_t, model)
        
        # 资本为负时失败
        if k_tp1 < mpf(0):
            return None, None 
        k_path[t + 1] = k_tp1

        # 计算下一期的消费
        R_bar = compute_R_bar(τ_c_path[t], τ_c_path[t + 1], 
                              τ_k_path[t + 1], k_tp1, model)
        c_tp1 = next_c(c_t, R_bar, model)

        # 消费为负时失败
        if c_tp1 < mpf(0):
            return None, None
        c_path[t + 1] = c_tp1

    return k_path, c_path


def bisection_c0(c0_guess, k0, shocks, S, model, 
                 tol=mpf('1e-6'), max_iter=1000, verbose=False):
    """
    使用二分法找到最优初始消费 c0。
    """
    k_ss_final, _ = steady_states(model, 
                                  mpf(shocks['g'][-1]), 
                                  mpf(shocks['τ_k'][-1]))
    c0_lower, c0_upper = mpf(0), f(k_ss_final, model)

    c0 = c0_guess
    for iter_count in range(max_iter):
        k_path, _ = shooting_algorithm(c0, k0, shocks, S, model)
        
        # 射击失败时调整上界
        if k_path is None:
            if verbose:
                print(f"迭代 {iter_count + 1}：c0 = {c0} 时射击失败")
            c0_upper = c0
        else:
            error = k_path[-1] - k_ss_final
            if verbose and iter_count % 100 == 0:
                print(f"迭代 {iter_count + 1}：c0 = {c0}, 误差 = {error}")

            # 检查收敛
            if abs(error) < tol:
                print(f"在第 {iter_count + 1} 次迭代成功收敛")
                return c0 

            # 根据误差更新边界
            if error > mpf(0):
                c0_lower = c0
            else:
                c0_upper = c0

        # 计算二分法的新中点
        c0 = (c0_lower + c0_upper) / mpf('2')

    # 如果未达到收敛则返回最后计算的 c0
    # 此时发出警告信息
    warn(f"收敛失败。返回最后的 c0 = {c0}", stacklevel=2)
    return c0

def run_shooting(shocks, S, model, c0_func=bisection_c0, shooting_func=shooting_algorithm):
    """
    运行射击算法。
    """
    # 计算初始稳态
    k0, c0 = steady_states(model, mpf(shocks['g'][0]), mpf(shocks['τ_k'][0]))
    
    # 找到最优初始消费
    optimal_c0 = c0_func(c0, k0, shocks, S, model)
    print(f"参数：{model}")
    print(f"最优初始消费 c0：{mp.nstr(optimal_c0, 7)} \n")
    
    # 模拟模型
    k_path, c_path = shooting_func(optimal_c0, k0, shocks, S, model)
    
    # 合并并返回结果
    return np.column_stack([k_path, c_path])
```
(cf:experiments)=
### 实验

让我们进行一些实验。

1. 在第10期发生的一次性永久性$g$增长,从0.2增至0.4,
2. 在第10期发生的一次性永久性$\tau_c$增长,从0.0增至0.2,
3. 在第10期发生的一次性永久性$\tau_k$增长,从0.0增至0.2,以及
4. 在第10期发生的一次性$g$增长,从0.2增至0.4,之后$g$永久性恢复到0.2。

+++

首先,我们准备用于初始化迭代算法的序列。

我们将从初始稳态开始,并在指定时间施加冲击。

```{code-cell} ipython3
def plot_results(solution, k_ss, c_ss, shocks, shock_param, 
                 axes, model, label='', linestyle='-', T=40):
    """
    绘制模拟结果,复制RMT中的图表。
    """
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    axes[0].plot(k_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(k_ss, linestyle='--', color='black')
    axes[0].set_title('k')

    # 绘制c
    axes[1].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[1].axhline(c_ss, linestyle='--', color='black')
    axes[1].set_title('c')

    # 绘制g
    R_bar_path = compute_R_bar_path(shocks, k_path, model, S)

    axes[2].plot(R_bar_path[:T], linestyle=linestyle, label=label)
    axes[2].set_title('$\overline{R}$')
    axes[2].axhline(1 / model.β, linestyle='--', color='black')
    
    η_path = compute_η_path(k_path, model, S=T)
    η_ss = model.α * model.A * k_ss ** (model.α - 1)
    
    axes[3].plot(η_path[:T], linestyle=linestyle, label=label)
    axes[3].axhline(η_ss, linestyle='--', color='black')
    axes[3].set_title(r'$\eta$')
    
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(rf'${shock_param}$')
```
**实验1：在第10期预见到$g$从0.2一次性永久性增加到0.4**

下图显示了在$t = T = 10$时预见到$g$永久性增加（通过增加一次性税收来融资）的后果

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
上述数据表明存在一个均衡的**消费平滑**机制，这是由代表性消费者偏好平滑消费路径所驱动的，这种偏好源自其单期效用函数的曲率。

- 资本存量的稳态值保持不变：
  - 这是因为在欧拉方程的稳态版本中({eq}`eq:diff_second_steady`)，$g$项消失了。

- 在时间$T$之前，由于政府消费增加，消费开始逐渐下降：
  - 家庭减少消费以抵消政府支出，这些支出通过增加一次性税收来融资。
  - 竞争性经济通过增加一次性税收流向家庭发出减少消费的信号。
  - 家庭关注的是税收的现值而非征收时间，因此消费受到不利的财富效应影响，导致立即做出反应。
- 资本在时间 $0$ 到 $T$ 之间由于储蓄增加而逐渐积累,在时间 $T$ 之后逐渐减少:
    - 这种资本存量的时间变化平滑了消费的时间分布,这是由代表性消费者的消费平滑动机驱动的。

让我们把上述程序整合成一个函数,该函数可以针对给定的实验运行求解器并绘制图表

```{code-cell} ipython3
:tags: [hide-input]

def experiment_model(shocks, S, model, solver, plot_func, policy_shock, T=40):
    """
    运行射击算法给定模型并绘制结果。
    """
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    
    print(f"稳态资本: {k0:.4f}")
    print(f"稳态消费: {c0:.4f}")
    print('-'*64)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    solution = solver(shocks, S, model)
    plot_func(solution, k0, c0, 
              shocks, policy_shock, axes, model, T=T)

    for ax in axes[5:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
```
下图比较了两个经济体在 $t = 10$ 时对预期的 $g$ 增长的响应:

* 实线表示我们原始的 $\gamma = 2$ 的经济体，以及
* 虚线表示一个除了 $\gamma = 0.2$ 外其他条件完全相同的经济体。

这个比较对我们很有意义，因为效用曲率参数 $\gamma$ 决定了家庭跨期替代消费的意愿，从而决定了其对消费路径随时间平滑程度的偏好。

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

def experiment_two_models(shocks, S, model_1, model_2, solver, plot_func, 
                          policy_shock, legend_label_fun=None, T=40):
    """
    比较并绘制两个模型的射击算法结果。
    """
    k0, c0 = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    print(f"稳态资本：{k0:.4f}")
    print(f"稳态消费：{c0:.4f}")
    print('-'*64)
    
    # 如果没有提供图例标签函数，则使用默认的
    if legend_label_fun is None:
        legend_label_fun = lambda model: fr"$\gamma = {model.γ}$"

    # 设置图形和坐标轴
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()

    # 为每个模型运行和绘图的函数
    def run_and_plot(model, linestyle='-'):
        solution = solver(shocks, S, model)
        plot_func(solution, k0, c0, shocks, policy_shock, axes, model, 
                  label=legend_label_fun(model), linestyle=linestyle, T=T)

    # 为两个模型绘图
    run_and_plot(model_1)
    run_and_plot(model_2, linestyle='-.')

    # 使用第一个坐标轴的标签设置图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=3, 
               fontsize=14, bbox_to_anchor=(1, 0.1))

    # 移除多余的坐标轴并整理布局
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
    比较并绘制价格
    """
    α, β, δ, γ, A = model.α, model.β, model.δ, model.γ, model.A
    
    k_path = solution[:, 0]
    c_path = solution[:, 1]

    # 绘制 c
    axes[0].plot(c_path[:T], linestyle=linestyle, label=label)
    axes[0].axhline(c_ss, linestyle='--', color='black')
    axes[0].set_title('c')
    
    # 绘制 q
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

    # 绘制 g
    axes[4].plot(shocks[shock_param][:T], linestyle=linestyle, label=label)
    axes[4].axhline(shocks[shock_param][0], linestyle='--', color='black')
    axes[4].set_title(shock_param)
```
对于$\gamma = 2$,下图描述了$q_t$以及利率期限结构对于在$t = 10$时可预见的$g_t$增长的响应

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
顶部的第二个面板比较了初始稳态的$q_t$与在$t = 0$时预见到$g$增加后的$q_t$，而第三个面板则比较了隐含的短期利率$r_t$。

第四个面板显示了在$t=0$、$t=10$和$t=60$时的利率期限结构。

注意，到$t = 60$时，系统已经收敛到新的稳态，利率期限结构变得平坦。

在$t = 10$时，利率期限结构呈上升趋势。

这种上升趋势反映了消费增长率随时间的预期增长，如消费面板所示。

在$t = 0$时，利率期限结构呈现"U形"模式：

- 在到期日$s = 10$之前呈下降趋势。
- 在$s = 10$之后，对更长期限呈上升趋势。
    
这种模式与前两张图中的消费增长模式相一致，
即在$t = 10$之前以递增的速率下降，之后以递减的速率下降。

+++
**实验2：在第10期将消费税率$\tau_c$从0.0一次性提高到0.2（可预见）**

在劳动供给缺乏弹性的情况下，欧拉方程{eq}`eq:euler_house`和其他均衡条件表明：
- 固定的消费税不会扭曲决策，但是
- 可预见的消费税变化会造成扭曲。

事实上，{eq}`eq:euler_house`或{eq}`eq:diff_second`表明，可预见的$\tau_{ct}$增加（即$(1+\tau_{ct})$
$(1+\tau_{ct+1})$的减少）的作用类似于$\tau_{kt}$的增加。

下图展示了对可预见的消费税$\tau_c$增加的响应。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_shooting, plot_results, 'τ_c')
```
显然上图中的所有变量最终都会回到其初始稳态值。

预期的$\tau_{ct}$增加导致消费和资本存量随时间发生变化：

- 在$t = 0$时：
    - 预期$\tau_c$的增加导致*消费立即跳升*。
    - 随后出现*消费狂潮*，使资本存量在$t = T = 10$之前持续下降。
- 在$t = 0$和$t = T = 10$之间：
    - 资本存量的下降导致$\bar{R}$随时间上升。
    - 均衡条件要求消费增长率在$t = T$之前上升。
- 在$t = T = 10$时：
    - $\tau_c$的跳升使$\bar{R}$降至1以下，导致*消费急剧下降*。
- 在$T = 10$之后：
    - 预期扭曲的影响结束，经济逐渐调整到较低的资本存量水平。
- 资本现在必须增长，这需要*紧缩*——在 $t = T$ 之后消费大幅下降，表现为较低的消费水平。
    - 利率逐渐下降，消费以递减的速率增长，直至达到最终稳态。

+++

**实验3：在第10期预见到 $\tau_k$ 从0.0一次性永久性增加到0.2**

对于两个 $\gamma$ 值2和0.2，下图显示了在 $t = T = 10$ 时对 $\tau_{kt}$ 的可预见永久性跳跃的响应。

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                run_shooting, plot_results, 'τ_k')
```
政府支出路径保持不变
- $\tau_{kt}$ 的增加通过减少一次性税收的现值来抵消，以保持预算平衡。

图表显示：

- 对 $\tau_{kt}$ 增加的预期导致资本存量立即下降，这是由于当前消费增加和消费流量增长。
- $\bar{R}$ 从 $t = 0$ 开始上升并在 $t = 9$ 达到峰值，在 $t = 10$ 时，$\bar{R}$ 因税收变化而急剧下降。
    - $\bar{R}$ 的变化与 $t = 10$ 时税收增加对跨期消费的影响相一致。
- 转型动态推动 $k_t$（资本存量）向新的、更低的稳态水平移动。在新的稳态下：
    - 由于资本存量减少导致产出降低，消费水平更低。
    - 当 $\gamma = 2$ 时的消费路径比 $\gamma = 0.2$ 时更平滑。

+++

到目前为止，我们已经探讨了可预见的一次性变化的后果
在政府政策中。接下来我们描述一些实验，其中存在可预见的一次性政策变量变化（一个"脉冲"）。

**实验4：在第10期可预见的一次性将$g$从0.2增加到0.4，之后$g$永远回到0.2**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_shooting, plot_results, 'g')
```
该图表明了以下内容：

- 消费：
    - 在政策宣布后立即下降，并在预期一次性的$g$激增过程中持续下降。
    - 在$t = 10$的冲击之后，消费开始恢复，以递减的速率上升至其稳态值。

- 资本和$\bar{R}$：
    - 在$t = 10$之前，由于利率变化导致家庭为预期中的政府支出增加做准备，资本开始积累。
    - 在$t = 10$时，由于政府消耗了部分资本，资本存量急剧下降。
    - 由于资本减少，$\bar{R}$跃升至高于其稳态值的水平，然后逐渐下降至其稳态水平。
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
# 欧拉方程和可行性条件
def euler_residual(c_t, c_tp1, τ_c_t, τ_c_tp1, τ_k_tp1, k_tp1, model):
    """
    计算欧拉方程的残差。
    """
    β, γ, δ, α, A = model.β, model.γ, model.δ, model.α, model.A
    η_tp1 = α * A * k_tp1 ** (α - 1)
    return β * (c_tp1 / c_t) ** (-γ) * (1 + τ_c_t) / (1 + τ_c_tp1) * (
        (1 - τ_k_tp1) * (η_tp1 - δ) + 1) - 1

def feasi_residual(k_t, k_tm1, c_tm1, g_t, model):
    """
    计算可行性条件的残差。
    """
    α, A, δ = model.α, model.A, model.δ
    return k_t - (A * k_tm1 ** α + (1 - δ) * k_tm1 - c_tm1 - g_t)
```
算法按以下步骤进行：

1. 根据 $t=0$ 时的政府计划找到初始稳态 $k_0$。

2. 初始化一个初始猜测序列 $\{\hat{c}_t, \hat{k}_t\}_{t=0}^{S}$。

3. 计算残差 $l_a$ 和 $l_k$ （对于 $t = 0, \dots, S$），以及 $t = 0$ 时的 $l_{k_0}$ 和 $t = S$ 时的 $l_{k_S}$：
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
# 计算残差作为需要最小化的目标函数
def compute_residuals(vars_flat, k_init, S, shocks, model):
    """
    计算欧拉方程、可行性条件和边界条件下的残差向量。
    """
    k, c = vars_flat.reshape((S + 1, 2)).T
    residuals = np.zeros(2 * S + 2)

    # 资本的初始条件
    residuals[0] = k[0] - k_init

    # 计算每个时间步的残差
    for t in range(S):
        residuals[2 * t + 1] = euler_residual(
            c[t], c[t + 1],
            shocks['τ_c'][t], shocks['τ_c'][t + 1], shocks['τ_k'][t + 1],
            k[t + 1], model
        )
        residuals[2 * t + 2] = feasi_residual(
            k[t + 1], k[t], c[t],
            shocks['g'][t], model
        )

    # 终端条件
    residuals[-1] = euler_residual(
        c[S], c[S],
        shocks['τ_c'][S], shocks['τ_c'][S], shocks['τ_k'][S],
        k[S], model
    )
    
    return residuals

# 用于最小化残差的根查找算法
def run_min(shocks, S, model):
    """
    用于最小化残差向量的根查找算法。
    """
    k_ss, c_ss = steady_states(model, shocks['g'][0], shocks['τ_k'][0])
    
    # 解路径的初始猜测
    initial_guess = np.column_stack(
        (np.full(S + 1, k_ss), np.full(S + 1, c_ss))).flatten()

    # 使用根查找求解系统
    sol = root(compute_residuals, initial_guess, 
               args=(k_ss, S, shocks, model), tol=1e-8)

    # 重塑解以获得k和c的时间路径
    return sol.x.reshape((S + 1, 2))
```
我们发现方法2没有遇到数值稳定性问题，所以使用 `mp.mpf` 并不必要。

我们把用第二种方法复现我们的实验作为练习。

## 练习

```{exercise}
:label: cass_fiscal_ex1

使用第二种残差最小化方法复现我们四个实验的图表：
1. 在第10期发生的可预见的一次性永久性政府支出g从0.2增加到0.4，
2. 在第10期发生的可预见的一次性永久性消费税τ_c从0.0增加到0.2，
3. 在第10期发生的可预见的一次性永久性资本税τ_k从0.0增加到0.2，以及
4. 在第10期发生的可预见的一次性政府支出g从0.2增加到0.4，之后g永久性恢复到0.2。
```

```{solution-start} cass_fiscal_ex1
:class: dropdown
```

这是一个解决方案：

**实验1：在第10期可预见的一次性永久性政府支出g从0.2增加到0.4**

```{code-cell} ipython3
S = 100
shocks = {
    'g': np.concatenate((np.repeat(0.2, 10), np.repeat(0.4, S - 9))),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'g')
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
fig.legend(handles, labels, title=r"$r_{t,t+s}$ 带有 ", loc='lower right', ncol=3, fontsize=10, bbox_to_anchor=(1, 0.1))  
plt.tight_layout()
plt.show()
```
**实验2：在第10期将消费税率$\tau_c$从0.0一次性提高到0.2（事先已知）。**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'τ_c')
```
**实验3：在第10期将资本税率$\tau_k$从0.0一次性提高到0.2（事先已知）。**

```{code-cell} ipython3
shocks = {
    'g': np.repeat(0.2, S + 1),
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.concatenate((np.repeat(0.0, 10), np.repeat(0.2, S - 9))) 
}

experiment_two_models(shocks, S, model, model_γ2, 
                run_min, plot_results, 'τ_k')
```
**实验4：在第10期预见到g从0.2一次性增加到0.4，之后g永久回到0.2**

```{code-cell} ipython3
g_path = np.repeat(0.2, S + 1)
g_path[10] = 0.4

shocks = {
    'g': g_path,
    'τ_c': np.repeat(0.0, S + 1),
    'τ_k': np.repeat(0.0, S + 1)
}

experiment_model(shocks, S, model, run_min, plot_results, 'g')
```

```{solution-end}
```

```{exercise}
:label: cass_fiscal_ex2

设计一个新实验，其中政府支出g在第10期从0.2增加到0.4，然后在第20期永久降至0.1。
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

experiment_model(shocks, S, model, run_min, plot_results, 'g')
```

```{solution-end}
```
