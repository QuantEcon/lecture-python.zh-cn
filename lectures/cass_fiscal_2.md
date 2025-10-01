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

# 带扭曲性税收的双国模型

## 概述

本讲是 QuantEcon 讲座 {doc}`cass_fiscal` 的续篇。在那一讲中，我们研究了可预见的财政和技术冲击对 Cass-Koopmans 增长模型(如 QuantEcon 讲座 {doc}`cass_koopmans_2` 中所述)中竞争均衡价格和数量的影响。该模型是非随机版本。

这里我们研究该模型的双国版本。

我们通过将两个 {doc}`cass_koopmans_2` 经济体背靠背放在一起来构建它，然后开放某些商品的国际贸易，而不是所有商品。

这让我们能够关注 {cite:t}`mendoza1998international` 研究的一些问题。

让我们从一些导入开始：
 
```{code-cell} ipython3
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from collections import namedtuple
from mpmath import mp, mpf
from warnings import warn

# 设置精度
mp.dps = 40
mp.pretty = True
```

## 两国Cass-Koopmans模型

本节描述{ref}`cs_fs_model`基本模型的两国版本。

该模型的结构类似于国际实际商业周期文献中使用的模型，其思路遵循{cite:t}`mendoza1998international`对扭曲性税收的分析。

我们允许两国之间进行商品贸易和未来商品债权交易，但不允许劳动力流动。

两国都拥有生产技术，每个国家的消费者都可以在任一国家持有资本，但需要接受不同的税收待遇。

我们用星号(*)表示第二个国家的变量。

两国的家庭都最大化终身效用：

$$
\sum_{t=0}^{\infty} \beta^t u(c_t) \quad \text{和} \quad \sum_{t=0}^{\infty} \beta^t u(c_t^*),
$$

其中$u(c) = \frac{c^{1-\gamma}}{1-\gamma}$，且$\gamma > 0$。

两国都采用具有相同技术参数的柯布-道格拉斯生产函数。

这个两国经济的世界资源约束是：

$$
(c_t+c_t^*)+(g_t+g_t^*)+(k_{t+1}-(1-\delta)k_t)+(k_{t+1}^*-(1-\delta)k_t^*) = f(k_t)+f(k_t^*)
$$

该约束结合了两国的可行性约束。

在后续计算中，我们将使用这个约束作为全球可行性约束。

为了连接两个国家，我们需要明确资本如何跨境流动以及在不同司法管辖区如何征税。

### 资本流动性和税收

第一国的消费者可以在任一国持有资本，但需要按照外国设定的税率对从外国资本持有中获得的租金缴税。

两国居民都可以在时间$t$以相同的Arrow-Debreu价格$q_t$购买消费品。我们假设资本市场是完整的。

让$B_t^f$表示本国代表性消费者通过向外国代表性消费者发行一期期票筹集的$t$期商品数量。

因此，$B_t^f > 0$表示本国消费者在$t$期从国外借款，而$B_t^f < 0$表示本国消费者在$t$期向国外贷款。

因此，第一国代表性消费者的预算约束为：

$$
\begin{aligned}
\sum_{t=0}^{\infty} q_t \left( c_t + (k_{t+1} - (1-\delta)k_t) + (\tilde{k}_{t+1} - (1-\delta)\tilde{k}_t) + R_{t-1,t}B_{t-1}^f \right) \leq \\
\sum_{t=0}^{\infty} q_t \left( (\eta_t - \tau_{kt}(\eta_t - \delta))k_t + (\eta_t^* - \tau_{kt}^*(\eta_t^* - \delta))\tilde{k}_t + (1 - \tau_{nt})w_t n_t - \tau_{ht} + B_t^f \right).
\end{aligned}
$$

对于$t \geq 1$，$k_t$和$\tilde{k}_t$的无套利条件意味着

$$
\begin{aligned}
q_{t-1} &= [(1 - \tau_{kt})(\eta_t - \delta) + 1] q_t, \\
q_{t-1} &= [(1 - \tau^*_{kt})(\eta^*_t - \delta) + 1] q_t,
\end{aligned}
$$

这两个等式共同表明，两国的税后资本租金率是相等的：

$$
(1 - \tau^*_{kt})(\eta^*_t - \delta) = (1 - \tau_{kt})(\eta_t - \delta).
$$

$B_t^f$ 的无套利条件对于 $t \geq 0$ 是 $q_t = q_{t+1} R_{t+1,t}$，这意味着

$$
q_{t-1} = q_t R_{t-1,t}
$$

对于 $t \geq 1$。

由于国内资本、国外资本和消费贷款具有相同的回报率，投资组合是不确定的。

如果我们允许 $B_t^f$ 为非零，我们可以将每个国家的国外资本持有量设为零。

这种解决投资组合不确定性的方法很方便，因为它减少了我们需要指定的初始条件数量。

因此，我们在允许国际借贷的同时，将两国的国外资本持有量都设为零。

给定从国内到国外的初始债务水平 $B_{-1}^f$，且 $R_{t-1,t} = \frac{q_{t-1}}{q_t}$，国际债务动态满足

$$
B^f_t = R_{t-1,t} B^f_{t-1} + c_t + (k_{t+1} - (1 - \delta)k_t) + g_t - f(k_t)
$$

```{code-cell} ipython3
def Bf_path(k, c, g, model):
    """
    计算 B^{f}_t:
      Bf_t = R_{t-1} Bf_{t-1} + c_t + (k_{t+1}-(1-δ)k_t) + g_t - f(k_t)
    其中 Bf_0 = 0.
    """
    S = len(c) - 1                       
    R = c[:-1]**(-model.γ) / (model.β * c[1:]**(-model.γ))

    Bf = np.zeros(S + 1) 
    for t in range(1, S + 1):
        inv = k[t] - (1 - model.δ) * k[t-1]         
        Bf[t] = (
            R[t-1] * Bf[t-1] + c[t] + inv + g[t-1] 
            - f(k[t-1], model))
    return Bf

def Bf_ss(c_ss, k_ss, g_ss, model):
    """
    计算稳态 B^f
    """
    R_ss   = 1.0 / model.β  
    inv_ss = model.δ * k_ss 
    num    = c_ss + inv_ss + g_ss - f(k_ss, model)
    den    = 1.0 - R_ss
    return num / den
```

且

$$
c^*_t + (k^*_{t+1} - (1 - \delta)k^*_t) + g^*_t - R_{t-1,t} B^f_{t-1} = f(k^*_t) - B^f_t.
$$

两国企业的一阶条件为：

$$
\begin{aligned}
\eta_t &= f'(k_t), \quad w_t = f(k_t) - k_t f'(k_t) \\
\eta^*_t &= f'(k^*_t), \quad w^*_t = f(k^*_t) - k^*_t f'(k^*_t).
\end{aligned}
$$

国际商品贸易建立了：

$$
\frac{q_t}{\beta^t} = \frac{u'(c_t)}{1 + \tau_{ct}} = \mu^* \frac{u'(c^*_t)}{1 + \tau^*_{ct}},
$$

其中$\mu^*$是一个非负数，是国家$*$中消费者预算约束的拉格朗日乘数的函数。

我们已将国内国家预算约束的拉格朗日乘数标准化，将相应的国内国家$\mu$设为1。

```{code-cell} ipython3
def compute_rs(c_t, c_tp1, c_s_t, c_s_tp1, τc_t, 
               τc_tp1, τc_s_t, τc_s_tp1, model):
    """
    计算贸易开始后的国际风险分担。
    """

    return (c_t**(-model.γ)/(1+τc_t)) * ((1+τc_s_t)/c_s_t**(-model.γ)) - (
        c_tp1**(-model.γ)/(1+τc_tp1)) * ((1+τc_s_tp1)/c_s_tp1**(-model.γ))
```

均衡要求以下两个国家欧拉方程在 $t \geq 0$ 时满足：

$$
\begin{aligned}
u'(c_t) &= \beta u'(c_{t+1}) \left[ (1 - \tau_{kt+1})(f'(k_{t+1}) - \delta) + 1 \right] \left[ \frac{1 + \tau_{ct+1}}{1 + \tau_{ct}} \right], \\
u'(c^*_t) &= \beta u'(c^*_{t+1}) \left[ (1 - \tau^*_{kt+1})(f'(k^*_{t+1}) - \delta) + 1 \right] \left[ \frac{1 + \tau^*_{ct+1}}{1 + \tau^*_{ct}} \right].
\end{aligned}
$$

以下代码计算国内和国外的欧拉方程。

由于它们具有相同的形式但使用不同的变量，我们可以编写一个函数来处理这两种情况。

```{code-cell} ipython3
def compute_euler(c_t, c_tp1, τc_t, 
                    τc_tp1, τk_tp1, k_tp1, model):
    """
    计算欧拉方程。
    """
    Rbar = (1 - τk_tp1)*(f_prime(k_tp1, model) - model.δ) + 1
    return model.β * (c_tp1/c_t)**(-model.γ) * (1+τc_t)/(1+τc_tp1) * Rbar - 1
```

### 初始条件和稳态

对于初始条件，我们选择贸易前的资本配置($k_0, k_0^*$)和本国（未标星国家）欠外国（标星国家）的初始国际债务水平$B_{-1}^f$。

### 均衡稳态值

两国模型的稳态由两组方程来表征。

首先，以下方程确定每个国家的稳态资本-劳动比率$\bar k$和$\bar k^*$：

$$
f'(\bar{k}) = \delta + \frac{\rho}{1 - \tau_k}
$$ (eq:steady_k_bar)

$$
f'(\bar{k}^*) = \delta + \frac{\rho}{1 - \tau_k^*}
$$ (eq:steady_k_star)

给定这些稳态资本-劳动比率，本国和外国的消费值$\bar c$和$\bar c^*$由以下方程确定：

$$
(\bar{c} + \bar{c}^*) = f(\bar{k}) + f(\bar{k}^*) - \delta(\bar{k} + \bar{k}^*) - (\bar{g} + \bar{g}^*)
$$ (eq:steady_c_k_bar)

$$
\bar{c} = f(\bar{k}) - \delta\bar{k} - \bar{g} - \rho\bar{B}^f
$$ (eq:steady_c_kB)

方程{eq}`eq:steady_c_k_bar`表示稳态下的可行性，而方程{eq}`eq:steady_c_kB`表示稳态下的贸易平衡，包括利息支付。

本国对外国的稳态债务水平$\bar{B}^f$影响两国之间的消费分配，但不影响世界总资本存量。

我们假设在稳态下$\bar{B}^f = 0$，这使我们得到以下函数来计算资本和消费的稳态值

```{code-cell} ipython3
def compute_steady_state_global(model, g_ss=0.2):
    """
    计算资本、消费和投资的稳态值。
    """
    k_ss = ((1/model.β - (1-model.δ)) / (model.A * model.α)) ** (1/(model.α-1))
    c_ss = f(k_ss, model) - model.δ * k_ss - g_ss
    return k_ss, c_ss
```

现在，我们可以应用残差最小化方法来计算资本和消费的稳态值。

我们再次对欧拉方程、全局资源约束和无套利条件的残差进行最小化。

```{code-cell} ipython3
def compute_residuals_global(z, model, shocks, T, k0_ss, k_star, Bf_star):
    """
    计算两国模型的残差。
    """
    k, c, k_s, c_s = z.reshape(T+1, 4).T
    g, gs = shocks['g'], shocks['g_s']
    τc, τk = shocks['τ_c'], shocks['τ_k']
    τc_s, τk_s = shocks['τ_c_s'], shocks['τ_k_s']
    
    res = [k[0] - k0_ss, k_s[0] - k0_ss]

    for t in range(T):
        e_d = compute_euler(
            c[t], c[t+1], 
            τc[t], τc[t+1], τk[t+1], 
            k[t+1], model)
        
        e_f = compute_euler(
            c_s[t], c_s[t+1], 
            τc_s[t], τc_s[t+1], τk_s[t+1], 
            k_s[t+1], model)
        
        rs = compute_rs(
            c[t], c[t+1], c_s[t], c_s[t+1], 
            τc[t], τc[t+1], τc_s[t], τc_s[t+1], 
            model)
        
        # 全局资源约束
        grc = k[t+1] + k_s[t+1] - (
            f(k[t], model) + f(k_s[t], model) +
            (1-model.δ)*(k[t] + k_s[t]) -
            c[t] - c_s[t] - g[t] - gs[t]
        )
        
        res.extend([e_d, e_f, rs, grc])

    Bf_term = Bf_path(k, c, shocks['g'], model)[-1]
    res.append(k[T] - k_star)
    res.append(Bf_term - Bf_star)
    return np.array(res)
```

现在我们绘制结果

```{code-cell} ipython3
# 绘制全球双国模型结果的函数
def plot_global_results(k, k_s, c, c_s, shocks, model, 
                        k0_ss, c0_ss, g_ss, S, T=40, shock='g',
                        # 存储左下面板序列的字典
                        ll_series='None'):
    """
    绘制双国模型的结果。
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    x = np.arange(T)
    τc, τk = shocks['τ_c'], shocks['τ_k']
    Bf = Bf_path(k, c, shocks['g'], model)
    
    # 计算衍生序列
    R_ratio = c[:-1]**(-model.γ) / (model.β * c[1:]**(-model.γ)) \
    *(1+τc[:-1])/(1+τc[1:])
    inv = k[1:] - (1-model.δ)*k[:-1]
    inv_s = k_s[1:] - (1-model.δ)*k_s[:-1]

    # 将初始条件添加到序列中
    R_ratio = np.append(1/model.β, R_ratio)
    c = np.append(c0_ss, c)
    c_s = np.append(c0_ss, c_s)
    k = np.append(k0_ss, k)
    k_s = np.append(k0_ss, k_s)

    # 资本
    axes[0,0].plot(x, k[:T], '-', lw=1.5)
    axes[0,0].plot(x, np.full(T, k0_ss), 'k-.', lw=1.5)
    axes[0,0].plot(x, k_s[:T], '--', lw=1.5)
    axes[0,0].set_title('k')
    axes[0,0].set_xlim(0, T-1)
    
    # 消费
    axes[0,1].plot(x, c[:T], '-', lw=1.5)
    axes[0,1].plot(x, np.full(T, c0_ss), 'k-.', lw=1.5)
    axes[0,1].plot(x, c_s[:T], '--', lw=1.5)
    axes[0,1].set_title('c')
    axes[0,1].set_xlim(0, T-1)
    
    # 利率
    axes[0,2].plot(x, R_ratio[:T], '-', lw=1.5)
    axes[0,2].plot(x, np.full(T, 1/model.β), 'k-.', lw=1.5)
    axes[0,2].set_title(r'$\bar{R}$')
    axes[0,2].set_xlim(0, T-1)
    
    # 投资
    axes[1,0].plot(x, np.full(T, model.δ * k0_ss), 
    'k-.', lw=1.5)
    axes[1,0].plot(x, np.append(model.δ*k0_ss, inv[:T-1]), 
    '-', lw=1.5)
    axes[1,0].plot(x, np.append(model.δ*k0_ss, inv_s[:T-1]), 
    '--', lw=1.5)
    axes[1,0].set_title('x')
    axes[1,0].set_xlim(0, T-1)
    
    # 冲击
    axes[1,1].plot(x, shocks[shock][:T], '-', lw=1.5)
    axes[1,1].plot(x, np.full(T, shocks[shock][0]), 'k-.', lw=1.5)
    axes[1,1].set_title(f'${shock}$')
    axes[1,1].set_ylim(-0.1, 0.5)
    axes[1,1].set_xlim(0, T-1)
    
    # 资本流动
    axes[1,2].plot(x, np.append(0, Bf[1:T]), lw=1.5)
    axes[1,2].plot(x, np.zeros(T), 'k-.', lw=1.5)
    axes[1,2].set_title(r'$B^{f}$')
    axes[1,2].set_xlim(0, T-1)

    plt.tight_layout()
    return fig, axes
```

如同我们在{doc}`cass_fiscal`中的单一国家模型，我们假设一个柯布-道格拉斯生产函数：

$$
F(k, 1) = A k^\alpha
$$

```{code-cell} ipython3
def f(k, model, A=1): 
    """
    生产函数：f(k) = A * k^{α}
    """
    return A * k ** model.α

def f_prime(k, model, A=1):
    """
    资本的边际产出：f'(k) = α * A * k^{α - 1}
    """
    return model.α * A * k ** (model.α - 1)
```

类似地，我们定义资本租赁率

$$
\eta_t = f'(k_t)  
$$

```{code-cell} ipython3
def compute_η_path(k_path, model, S=100, A_path=None):
    """
    计算η路径：η_t = f'(k_t)
    对于增长模型可选择性地包含A_path。
    """
    A = np.ones_like(k_path) if A_path is None else np.asarray(A_path)
    η_path = np.zeros_like(k_path)
    for t in range(S):
        η_path[t] = f_prime(k_path[t], model, A[t])
    return η_path
```

#### 实验1：在t=10时预见到g从0.2增加到0.4

下图展示了国内经济中g从0.2增加到0.4（提前十个周期宣布）后的转换动态。

我们从两个经济体的稳态开始，初始条件为$B_0^f = 0$。

在下图中，蓝线代表国内经济，橙色虚线代表国外经济。

```{code-cell} ipython3
Model = namedtuple("Model", ["β", "γ", "δ", "α", "A"])
model = Model(β=0.95, γ=2.0, δ=0.2, α=0.33, A=1.0)
S = 100

shocks_global = {
    'g': np.concatenate((np.full(10, 0.2), np.full(S-9, 0.4))),
    'g_s': np.full(S+1, 0.2),
    'τ_c': np.zeros(S+1),
    'τ_k': np.zeros(S+1),
    'τ_c_s': np.zeros(S+1),
    'τ_k_s': np.zeros(S+1)
}
g_ss = 0.2
k0_ss, c0_ss = compute_steady_state_global(model, g_ss)

k_star = k0_ss
Bf_star = Bf_ss(c0_ss, k_star, g_ss, model)

init_glob = np.tile([k0_ss, c0_ss, k0_ss, c0_ss], S+1)
sol_glob = root(
    lambda z: compute_residuals_global(z, model, shocks_global,
                                        S, k0_ss, k_star, Bf_star),
    init_glob, tol=1e-12
)
k, c, k_s, c_s = sol_glob.x.reshape(S+1, 4).T

# Plot global results via function
plot_global_results(k, k_s, c, c_s,
                        shocks_global, model,
                        k0_ss, c0_ss, g_ss,
                        S)
plt.show()
```

在时间1时，政府宣布国内政府支出$g$将在十个周期后上升，这将侵占未来的私人资源。

为了平滑消费，国内家庭立即增加储蓄，以抵消预期中对其未来财富的冲击。

在封闭经济中，他们只能通过积累额外的国内资本来储蓄；而在开放的资本市场中，他们还可以向外国人贷款。

一旦资本流动在时间$1$开放，无套利条件将连接这两种储蓄方式的调整：国内家庭储蓄的增加将降低外国经济中债券和资本的均衡回报率，以防止套利机会。

由于无套利使边际效用比率相等，两个经济体的消费和资本路径将同步变化。

在更高的$g$生效之前，两个国家都继续增加其资本存量。

当政府支出在10个周期后最终上升时，国内家庭开始动用部分资本来缓冲消费。

同样根据无套利条件，当$g$实际增加时，两个国家都降低了其投资率。

国内经济随之开始出现经常账户赤字，部分用于资助$g$的增加。

这意味着外国家庭通过减少其资本存量来开始偿还部分外部债务。


#### 实验2：在t=10时$g$从0.2可预见地增加到0.4

我们现在探讨在t = 1时宣布的国内资本税上调政策在10个周期后的影响。

由于这一变化是可预期的,尽管税收直到第11期才开始生效,两国家庭都会立即做出调整。

```{code-cell} ipython3
shocks_global = {
    'g': np.full(S+1, g_ss),
    'g_s': np.full(S+1, g_ss),
    'τ_c': np.zeros(S+1),
    'τ_k': np.concatenate((np.zeros(10), np.full(S-9, 0.2))),
    'τ_c_s': np.zeros(S+1),
    'τ_k_s': np.zeros(S+1),
}
    
k0_ss, c0_ss = compute_steady_state_global(model, g_ss)
k_star = k0_ss
Bf_star = Bf_ss(c0_ss, k_star, g_ss, model)

init_glob = np.tile([k0_ss, c0_ss, k0_ss, c0_ss], S+1)

sol_glob = root(
    lambda z: compute_residuals_global(z, model, 
            shocks_global, S, k0_ss, k_star, Bf_star),
            init_glob, tol=1e-12)

k, c, k_s, c_s = sol_glob.x.reshape(S+1, 4).T

# plot 
fig, axes = plot_global_results(k, k_s, c, c_s, shocks_global, model, 
                                k0_ss, c0_ss, g_ss, S, shock='τ_k')
plt.tight_layout()
plt.show()
```

在宣布增税后，国内家庭预见到资本的税后回报率降低，因此转向更高的当前消费，并允许国内资本存量下降。

这种世界资本供应的萎缩推动全球实际利率上升，促使外国家庭也提高当前消费。

在实际加税之前，国内经济通过进口资本为部分消费提供资金，产生经常账户赤字。

当$\tau_k$最终上升时，国际套利导致投资者迅速将资本重新配置到未征税的国外市场，压缩了各地的债券收益率。

债券利率下跌反映了国内资本的税后回报率降低，以及国外资本存量增加导致其边际产出下降。

外国家庭通过对外借款为其资本购买提供资金，造成显著的经常账户赤字和外部债务的积累。

政策变更后，两国平稳地向新的稳态过渡，其特点是：

  * 各经济体的消费水平稳定在其宣布前路径之下。
  * 资本存量的差异恰好足以使跨境税后回报率趋于相等。

尽管承担着正的净负债，由于较大的资本存量带来更高的产出，外国能够享有更高的稳态消费。

该案例展示了开放资本市场如何在国际间传递国内税收冲击：资本流动和利率变动共同分担负担，随着时间推移平滑了征税和未征税经济体的消费调整。

+++

```{exercise}
:label: cass_fiscal_ex4

在本练习中，用 $\eta_t$ 替换 ${x_t}$ 的图表，以复制 {cite}`Ljungqvist2012` 中的图形。

比较 ${k_t}$ 和 $\eta_t$ 的图形并讨论其经济含义。
```
```{solution-start} cass_fiscal_ex4
:class: dropdown
```

这是一个解决方案。

```{code-cell} ipython3
fig, axes = plot_global_results(k, k_s, c, c_s, shocks_global, model, 
                                k0_ss, c0_ss, g_ss, S, shock='τ_k')

# Clear the plot for x_t
axes[1,0].cla()

# Plot η_t
axes[1,0].plot(compute_η_path(k, model)[:40])
axes[1,0].plot(compute_η_path(k_s, model)[:40], '--')
axes[1,0].plot(np.full(40, f_prime(k_s, model)[0]), 'k-.', lw=1.5)
axes[1,0].set_title(r'$\eta$')

plt.tight_layout()
plt.show()
```

当税收冲击后国内资本 ${k_t}$ 减少时，该国的租金率 $\eta_t$ 上升。

这是因为当资本变得稀缺时，其边际产出会上升。

```{solution-end}
```

