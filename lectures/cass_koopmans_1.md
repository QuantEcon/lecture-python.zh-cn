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
---

(cass_koopmans_1)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Cass-Koopmans模型

## 概述

本讲座和{doc}`Cass-Koopmans竞争均衡 <cass_koopmans_2>`描述了Tjalling Koopmans {cite}`Koopmans`和David Cass {cite}`Cass`用来分析最优增长的模型。

该模型扩展了[早前讲座](https://python-programming.quantecon.org/python_oop.html)中描述的Robert Solow模型。
它通过将储蓄率作为一个决策变量，而不是一个固定的常数来实现这一点。

（索洛假设储蓄率是模型外部决定的常数。）

我们将描述该模型的两个版本，本讲中的规划问题，以及在{doc}`Cass-Koopmans 竞争均衡 <cass_koopmans_2>`讲中的竞争均衡。

这两节课共同说明了**计划经济**和以**竞争均衡**形式组织的分散经济之间实际上存在的更普遍联系。

本讲重点讨论计划经济版本。

在计划经济中，

- 没有价格
- 没有预算约束

相反，有一个决策者告诉人们

- 生产什么
- 在实物资本中投资什么
- 谁在什么时候消费什么

本讲使用的重要概念包括

- 用于解决规划问题的极小极大问题
- 用于在给定初始和终端条件下求解差分方程的**射击算法**
- 对于长期但有限期经济的最优路径的**收费公路**性质
- **稳定流形**和**相位平面**

除了 Anaconda 中已有的库之外，本讲座还需要以下库：

```{code-cell} ipython
:tags: [hide-output]
!pip install quantecon
```

让我们从一些标准导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from numba import jit, float64
from numba.experimental import jitclass
import numpy as np
from quantecon.optimize import brentq
```
## 模型

时间是离散的，取值为 $t = 0, 1 , \ldots, T$，其中 $T$ 是有限的。

(我们最终会研究 $T = + \infty$ 的极限情况)

单一商品可以被消费或投资于实物资本。

消费品不耐用，如果不立即消费就会完全折旧。

资本品是耐用的，但会折旧。

我们用 $C_t$ 表示在时间 $t$ 的非耐用消费品的总消费量。

用 $K_t$ 表示在时间 $t$ 的实物资本存量。

令 $\vec{C}$ = $\{C_0,\dots, C_T\}$ 且
$\vec{K}$ = $\{K_0,\dots,K_{T+1}\}$。

### 插话：聚合理论

我们使用代表性消费者的概念，可以理解如下。

有一个单位质量的相同消费者，用 $\omega \in [0,1]$ 索引。

消费者 $\omega$ 的消费量为 $c(\omega)$。

总消费量为

$$
C = \int_0^1 c(\omega) d \omega
$$

考虑一个福利问题，选择消费者之间的分配 $\{c(\omega)\}$ 以最大化

$$
 \int_0^1 u(c(\omega)) d \omega
$$

其中 $u(\cdot)$ 是一个凹效用函数，满足 $u' >0, u'' < 0$，且最大化受约束于

$$
C = \int_0^1 c(\omega) d \omega .
$$ (eq:feas200)

构建拉格朗日函数 $L = \int_0^1 u(c(\omega)) d \omega + \lambda [C - \int_0^1 c(\omega) d \omega ] $。

对每个 $\omega$ 在积分号下求导，得到一阶必要条件

$$
u'(c(\omega)) = \lambda.
$$

这些条件意味着 $c(\omega)$ 等于一个与 $\omega$ 无关的常数 $c$。

要找到 $c$，使用可行性约束 {eq}`eq:feas200` 可以得出

$$
c(\omega) = c = C.
$$

这种推导过程揭示了代表性消费者消费量为 $C$ 背后的特殊*聚合理论*。

这在宏观经济学中经常出现。

我们将在这里以及讲座{doc}`Cass-Koopmans 竞争均衡 <cass_koopmans_2>`中使用这个聚合理论。

#### 一个经济体

代表性家庭在每个时期 $t$ 都拥有一单位的劳动力，并且喜欢在每个时期消费商品。

代表性家庭在每个时期 $t$ 非弹性地供应一单位劳动力 $N_t$，因此对所有 $t \in \{0, 1, \ldots,  T\}$, $N_t =1$。

代表性家庭对消费组合的偏好由以下效用函数给出：

```{math}
:label: utility-functional

U(\vec{C}) = \sum_{t=0}^{T} \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}
```

其中 $\beta \in (0,1)$ 是贴现因子，$\gamma >0$
决定单期效用函数的曲率。

较大的 $\gamma$ 意味着更大的曲率。

注意

```{math}
:label: utility-oneperiod

u(C_t) = \frac{C_t^{1-\gamma}}{1-\gamma}
```

满足 $u'>0,u''<0$。

$u' > 0$ 表明消费者偏好更多而不是更少。

$u''< 0$ 表明随着 $C_t$ 的增加，边际效用递减。

我们假设 $K_0 > 0$ 是一个外生的初始资本存量。
存在一个全经济范围的生产函数

```{math}
:label: production-function

F(K_t,N_t) = A K_t^{\alpha}N_t^{1-\alpha}
```

其中 $0 < \alpha<1$，$A > 0$。

一个可行的配置 $\vec{C}, \vec{K}$ 满足

```{math}
:label: allocation

C_t + K_{t+1} \leq F(K_t,N_t) + (1-\delta) K_t \quad \text{对所有 } t \in \{0, 1, \ldots,  T\}
```

其中 $\delta \in (0,1)$ 是资本的折旧率。

## 规划问题

规划者选择配置 $\{\vec{C},\vec{K}\}$ 以
最大化 {eq}`utility-functional`，且受约束于 {eq}`allocation`。

令 $\vec{\mu}=\{\mu_0,\dots,\mu_T\}$ 为一个
非负的**拉格朗日乘数**序列。

为了找到最优配置，构建拉格朗日函数

$$
\mathcal{L}(\vec{C} ,\vec{K} ,\vec{\mu} ) =
\sum_{t=0}^T \beta^t\left\{ u(C_t)+ \mu_t
\left(F(K_t,1) + (1-\delta) K_t- C_t - K_{t+1} \right)\right\}
$$ (eq:Lagrangian201)

并提出以下极小极大问题：

```{math}
:label: min-max-prob
\min_{\vec{\mu}} \max_{\vec{C},\vec{K}} \mathcal{L}(\vec{C},\vec{K},\vec{\mu} )
```

- **极值化**意味着
  对 $\vec{C}, \vec{K}$ 求最大值，
  对 $\vec{\mu}$ 求最小值。
- 我们的问题满足一些条件，这些条件能够保证：在满足我们即将计算的一阶必要条件的配置下，二阶条件也得到满足。

在计算一阶条件之前，我们先介绍一些实用的公式。

### 线性齐次生产函数的有用性质

以下技术细节将对我们有帮助。

注意到

$$
F(K_t,N_t) = A K_t^\alpha N_t^{1-\alpha} = N_t A\left(\frac{K_t}{N_t}\right)^\alpha
$$

定义**人均产出生产函数**

$$
\frac{F(K_t,N_t)}{N_t} \equiv f\left(\frac{K_t}{N_t}\right) = A\left(\frac{K_t}{N_t}\right)^\alpha
$$

其自变量是**人均资本**。

回顾以下计算是很有用的: 资本的边际产出

```{math}
:label: useful-calc1

\begin{aligned}
\frac{\partial F(K_t,N_t)}{\partial K_t}
& =
\frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial K_t}
\\ &=
N_t f'\left(\frac{K_t}{N_t}\right)\frac{1}{N_t} \quad \text{(链式法则)}
\\ &=
f'\left.\left(\frac{K_t}{N_t}\right)\right|_{N_t=1}
\\ &= f'(K_t)
\end{aligned}
```

以及劳动力的边际产出

$$
\begin{aligned}
\frac{\partial F(K_t,N_t)}{\partial N_t}
&=
\frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial N_t} \quad \text{(乘积法则)}
\\ &=
f\left(\frac{K_t}{N_t}\right){+} N_t f'\left(\frac{K_t}{N_t}\right) \frac{-K_t}{N_t^2} \quad \text{(链式法则)}
\\ &=
f\left(\frac{K_t}{N_t}\right){-}\frac{K_t}{N_t}f'\left.\left(\frac{K_t}{N_t}\right)\right|_{N_t=1}
\\ &=
f(K_t) - f'(K_t) K_t
\end{aligned}
$$

(这里我们使用了对所有 $t$ 都有 $N_t = 1$ 的条件，因此 $K_t = \frac{K_t}{N_t}$。)

### 一阶必要条件
我们现在计算拉格朗日函数{eq}`eq:Lagrangian201`的**一阶必要条件**：

```{math}
:label: constraint1

C_t: \qquad u'(C_t)-\mu_t=0 \qquad \text{对所有} \qquad t= 0,1,\dots,T
```

```{math}
:label: constraint2

K_t: \qquad \beta \mu_t\left[(1-\delta)+f'(K_t)\right] - \mu_{t-1}=0 \qquad \text{对所有} \qquad t=1,2,\dots,T
```

```{math}
:label: constraint3

\mu_t:\qquad F(K_t,1)+ (1-\delta) K_t  - C_t - K_{t+1}=0 \qquad \text{对所有} \qquad t=0,1,\dots,T
```

```{math}
:label: constraint4

K_{T+1}: \qquad -\mu_T \leq 0, \ \leq 0 \text{ 如果 } K_{T+1}=0; \ =0 \text{ 如果 } K_{T+1}>0
```

在计算{eq}`constraint2`时，我们注意到$K_t$同时出现在时间$t$和时间$t-1$的可行性约束{eq}`allocation`中。

限制条件{eq}`constraint4`来自对$K_{T+1}$求导，并应用以下**卡鲁什-库恩-塔克条件**（KKT）
(参见[库恩塔克条件](https://baike.baidu.com/item/%E5%BA%93%E6%81%A9%E5%A1%94%E5%85%8B%E6%9D%A1%E4%BB%B6/3828439)):

```{math}
:label: kkt

\mu_T K_{T+1}=0
```

将{eq}`constraint1`和{eq}`constraint2`结合得到

$$
\beta u'\left(C_t\right)\left[(1-\delta)+f'\left(K_t\right)\right]-u'\left(C_{t-1}\right)=0
\quad \text{ 对所有 } t=1,2,\dots, T+1
$$

可以重新整理为

```{math}
:label: l12

\beta u'\left(C_{t+1}\right)\left[(1-\delta)+f'\left(K_{t+1}\right)\right]=
u'\left(C_{t}\right) \quad \text{ 对所有 } t=0,1,\dots, T
```

对上述等式两边应用消费的边际效用的反函数得到

$$
C_{t+1} =u'^{-1}\left(\left(\frac{\beta}{u'(C_t)}[f'(K_{t+1}) +(1-\delta)]\right)^{-1}\right)
$$

代入效用函数{eq}`utility-oneperiod`，这就变成了消费的**欧拉方程**

$$
\begin{aligned} C_{t+1} =\left(\beta C_t^{\gamma}[f'(K_{t+1}) +
(1-\delta)]\right)^{1/\gamma} 
%\notag\\= C_t\left(\beta [f'(K_{t+1}) +
%(1-\delta)]\right)^{1/\gamma} 
\end{aligned}
$$ (eq:consn_euler)

我们可以将其与可行性约束{eq}`allocation`结合得到

$$ 
\begin{aligned}
C_{t+1} & = C_t\left(\beta [f'(F(K_t,1)+ (1-\delta) K_t  - C_t) +
(1-\delta)]\right)^{1/\gamma}  \\
K_{t+1}  & = F(K_t,1)+ (1-\delta) K_t  - C_t .
\end{aligned}
$$ (eq:systemdynamics)

这是一对非线性一阶差分方程，将 $C_t, K_t$ 映射到 $C_{t+1}, K_{t+1}$。最优序列$\vec C , \vec K$ 必须满足这些方程。

它还必须满足初始条件:给定 $K_0$ 且 $K_{T+1} = 0$。

下面我们定义一个`jitclass`来存储定义我们经济的参数和函数。

```{code-cell} ipython3
planning_data = [
    ('γ', float64),    # 相对风险厌恶系数
    ('β', float64),    # 贴现因子
    ('δ', float64),    # 资本折旧率
    ('α', float64),    # 人均资本回报率
    ('A', float64)     # 技术水平
]
```
```{code-cell} ipython3
@jitclass(planning_data)
class PlanningProblem():

    def __init__(self, γ=2, β=0.95, δ=0.02, α=0.33, A=1):

        self.γ, self.β = γ, β
        self.δ, self.α, self.A = δ, α, A

    def u(self, c):
        '''
        效用函数
        注意：如果你有一个难以手动求解的效用函数
        你可以使用自动或符号微分
        参见 https://github.com/HIPS/autograd
        '''
        γ = self.γ

        return c ** (1 - γ) / (1 - γ) if γ!= 1 else np.log(c)

    def u_prime(self, c):
        '效用函数的导数'
        γ = self.γ

        return c ** (-γ)

    def u_prime_inv(self, c):
        '效用函数导数的逆函数'
        γ = self.γ

        return c ** (-1 / γ)

    def f(self, k):
        '生产函数'
        α, A = self.α, self.A

        return A * k ** α

    def f_prime(self, k):
        '生产函数的导数'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)

    def f_prime_inv(self, k):
        '生产函数导数的逆函数'
        α, A = self.α, self.A

        return (k / (A * α)) ** (1 / (α - 1))

    def next_k_c(self, k, c):
        ''''
        给定当前资本Kt和任意可行的
        消费选择Ct，通过状态转移定律计算Kt+1
        并通过欧拉方程计算最优Ct+1。
        '''
        β, δ = self.β, self.δ
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f, f_prime = self.f, self.f_prime

        k_next = f(k) + (1 - δ) * k - c
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next) + (1 - δ))))

        return k_next, c_next
```
我们可以用Python代码构建一个经济模型：

```{code-cell} ipython3
pp = PlanningProblem()
```
## 打靶算法

我们使用**打靶法**来计算最优配置
$\vec{C}, \vec{K}$ 和相关的拉格朗日乘数序列
$\vec{\mu}$。

规划问题的一阶必要条件
{eq}`constraint1`, {eq}`constraint2`, 和
{eq}`constraint3` 构成了一个具有两个边界条件的**差分方程**系统：

- $K_0$ 是资本的给定**初始条件**
- $K_{T+1} =0$ 是资本的**终端条件**，这是我们从
  $K_{T+1}$ 的一阶必要条件KKT条件 {eq}`kkt` 推导出来的

我们没有拉格朗日乘数 $\mu_0$ 的初始条件。

如果有的话，我们的工作就会很简单：

- 给定 $\mu_0$ 和 $k_0$，我们可以从方程 {eq}`constraint1` 计算出 $c_0$，
  然后从方程 {eq}`constraint3` 计算出 $k_1$，从方程
  {eq}`constraint2` 计算出 $\mu_1$。
- 我们可以用这种方式继续计算
  $\vec{C}, \vec{K}, \vec{\mu}$ 的其余元素。

然而，我们无法确保卡鲁什-库恩-塔克条件 {eq}`kkt` 会得到满足。

此外，我们没有 $\mu_0$ 的初始条件。

所以这种方法行不通。

实际上，我们的任务之一就是计算 $\mu_0$ 的**最优**值。

为了计算 $\mu_0$ 和其他我们想要的对象，对上述程序做一个简单的修改就可以了。

这被称为**射击算法**。

这是一个**猜测和验证**算法的实例，包含以下步骤：

- 猜测一个初始拉格朗日乘数 $\mu_0$。
- 应用上述描述的**简单算法**。
- 计算 $K_{T+1}$ 并检查它是否等于零。
- 如果 $K_{T+1} =0$，我们就解决了这个问题。
- 如果 $K_{T+1} > 0$，降低 $\mu_0$ 并重试。
- 如果 $K_{T+1} < 0$，提高 $\mu_0$ 并重试。

以下 Python 代码为规划问题实现了射击算法。

（实际上，在下面的代码中，我们对前面的算法稍作修改：从对 $c_0$ 的猜测开始，而不是从对 $\mu_0$ 的猜测开始。）

```{code-cell} ipython3
@jit
def shooting(pp, c0, k0, T=10):
    '''
    给定资本的初始条件k0和消费的初始猜测值c0，
    使用状态转移方程和欧拉方程计算T期内c和k的完整路径。
    '''
    if c0 > pp.f(k0) + (1 - pp.δ) * k0:
        print("初始消费不可行")

        return None

    # 初始化c和k的向量
    c_vec = np.empty(T+1)
    k_vec = np.empty(T+2)

    c_vec[0] = c0
    k_vec[0] = k0

    for t in range(T):
        k_vec[t+1], c_vec[t+1] = pp.next_k_c(k_vec[t], c_vec[t])

    k_vec[T+1] = pp.f(k_vec[T]) + (1 - pp.δ) * k_vec[T] - c_vec[T]

    return c_vec, k_vec
```
我们先从一个错误的猜测开始。

```{code-cell} ipython3
paths = shooting(pp, 0.2, 0.3, T=10)
```
```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

colors = ['blue', 'red']
titles = ['消费', '资本']
ylabels = ['$c_t$', '$k_t$']

T = paths[0].size - 1
for i in range(2):
    axs[i].plot(paths[i], c=colors[i])
    axs[i].set(xlabel='t', ylabel=ylabels[i], title=titles[i])

axs[1].scatter(T+1, 0, s=80)
axs[1].axvline(T+1, color='k', ls='--', lw=1)

plt.show()
```
显然，我们对 $\mu_0$ 的初始猜测值太高了，所以初始消费太低。

我们知道这一点是因为我们超过了目标 $K_{T+1}=0$。

现在我们用一个搜索合适 $\mu_0$ 的算法来自动化这个过程，当我们达到目标 $K_{t+1} = 0$ 时停止。

我们使用**二分法**。

我们对 $C_0$ 做一个初始猜测（我们可以消除 $\mu_0$，因为 $C_0$ 是 $\mu_0$ 的精确函数）。

我们知道 $C_0$ 的最小值只能是 $0$，最大值是初始产出 $f(K_0)$。

猜测 $C_0$ 并向前推算到 $T+1$。

如果 $K_{T+1}>0$，我们将其作为 $C_0$ 的新**下界**。

如果 $K_{T+1}<0$，我们将其作为新的**上界**。

对 $C_0$ 做一个新的猜测，取新的上下界的中间值。

再次向前推算，重复这些步骤直到收敛。

当 $K_{T+1}$ 足够接近 $0$（即在误差容限范围内）时，我们停止。

```{code-cell} ipython3
@jit
def bisection(pp, c0, k0, T=10, tol=1e-4, max_iter=500, k_ter=0, verbose=True):

    # 设置初始边界
    c0_upper = pp.f(k0)
    c0_lower = 0

    i = 0
    while True:
        c_vec, k_vec = shooting(pp, c0, k0, T)
        error = k_vec[-1] - k_ter

        # 检查终端条件是否得到满足
        if np.abs(error) < tol:
            if verbose:
                print('在第', i+1, '迭代步时成功收敛')
            return c_vec, k_vec

        i += 1
        if i == max_iter:
            if verbose:
                print('收敛失败')
            return c_vec, k_vec

        # 如果迭代继续，更新c0的猜测和边界
        if error > 0:
            c0_lower = c0
        else:
            c0_upper = c0

        c0 = (c0_lower + c0_upper) / 2
```
```{code-cell} ipython3
def plot_paths(pp, c0, k0, T_arr, k_ter=0, k_ss=None, axs=None):

    if axs is None:
        fix, axs = plt.subplots(1, 3, figsize=(16, 4))
    ylabels = ['$c_t$', '$k_t$', '$\mu_t$']
    titles = ['消费', '资本', '拉格朗日乘数']

    c_paths = []
    k_paths = []
    for T in T_arr:
        c_vec, k_vec = bisection(pp, c0, k0, T, k_ter=k_ter, verbose=False)
        c_paths.append(c_vec)
        k_paths.append(k_vec)

        μ_vec = pp.u_prime(c_vec)
        paths = [c_vec, k_vec, μ_vec]

        for i in range(3):
            axs[i].plot(paths[i])
            axs[i].set(xlabel='t', ylabel=ylabels[i], title=titles[i])

        # 绘制资本的稳态值
        if k_ss is not None:
            axs[1].axhline(k_ss, c='k', ls='--', lw=1)

        axs[1].axvline(T+1, c='k', ls='--', lw=1)
        axs[1].scatter(T+1, paths[1][-1], s=80)

    return c_paths, k_paths
```
现在我们可以求解模型并绘制消费、资本和拉格朗日乘数的路径。

```{code-cell} ipython3
plot_paths(pp, 0.3, 0.3, [10]);
```
## 将初始资本设定为稳态资本

当 $T \rightarrow +\infty$ 时，最优配置收敛到
$C_t$ 和 $K_t$ 的稳态值。

将 $K_0$ 设定为 $\lim_{T \rightarrow + \infty } K_t$ 是很有启发性的，我们称之为稳态资本。

在稳态下，对于所有很大的 $t$ 都有 $K_{t+1} = K_t=\bar{K}$。

在 $\bar{K}$ 处评估可行性约束 {eq}`allocation` 得到

```{math}
:label: feasibility-constraint

f(\bar{K})-\delta \bar{K} = \bar{C}
```

将所有 $t$ 的 $K_t = \bar K$ 和 $C_t=\bar C$ 
代入 {eq}`l12` 得到

$$
1=\beta \frac{u'(\bar{C})}{u'(\bar{C})}[f'(\bar{K})+(1-\delta)]
$$

定义 $\beta = \frac{1}{1+\rho}$，得到

$$
1+\rho = 1[f'(\bar{K}) + (1-\delta)]
$$

化简得到

$$
f'(\bar{K}) = \rho +\delta
$$

和

$$
\bar{K} = f'^{-1}(\rho+\delta)
$$

代入生产函数 {eq}`production-function`，这变为

$$
\alpha \bar{K}^{\alpha-1} = \rho + \delta
$$
例如，在设定 $\alpha= .33$，
$\rho = 1/\beta-1 =1/(19/20)-1 = 20/19-19/19 = 1/19$，$\delta = 1/50$ 后，
我们得到

$$
\bar{K} = \left(\frac{\frac{33}{100}}{\frac{1}{50}+\frac{1}{19}}\right)^{\frac{67}{100}} \approx 9.57583
$$

让我们用Python验证这个结果，然后使用这个稳态值
$\bar K$ 作为我们的初始资本存量 $K_0$。

```{code-cell} ipython3
ρ = 1 / pp.β - 1
k_ss = pp.f_prime_inv(ρ+pp.δ)

print(f'资本的稳态值为: {k_ss}')
```
现在我们绘制图形

```{code-cell} ipython3
plot_paths(pp, 0.3, k_ss, [150], k_ss=k_ss);
```
显然，当 $T$ 值较大时，$K_t$ 会一直保持在接近 $K_0$ 的水平，直到 $t$ 接近 $T$ 时才会发生变化。

让我们看看当我们将 $K_0$ 设置为低于 $\bar K$ 时，规划者会做什么。

```{code-cell} ipython3
plot_paths(pp, 0.3, k_ss/3, [150], k_ss=k_ss);
```
注意观察规划者如何将资本推向稳态：在那里停留一段时间，然后当 $t$ 接近 $T$ 时，将 $K_t$ 推向终值 $K_{T+1} =0$。

下面的图表比较了在不同的 $T$ 值下的最优结果。

```{code-cell} ipython3
plot_paths(pp, 0.3, k_ss/3, [150, 75, 50, 25], k_ss=k_ss);
```
## 收费公路性质（Turnpike property）

以下计算表明，当 $T$ 非常大时，最优资本存量在大部分时间里都会保持在接近其稳态值的水平。

```{code-cell} ipython3
plot_paths(pp, 0.3, k_ss/3, [250, 150, 50, 25], k_ss=k_ss);
```
在上图中，不同的颜色对应着不同的规划期限 $T$。

注意，随着规划期限的增加，规划者会让 $K_t$ 在更长时间内保持接近稳态值 $\bar K$。

这种模式反映了稳态的**收费公路**性质。

对规划者来说，一个经验法则是：

- 从 $K_0$ 开始，将 $K_t$ 推向稳态，并在接近时间 $T$ 之前保持在稳态附近。

规划者通过调整储蓄率 $\frac{f(K_t) - C_t}{f(K_t)}$ 来实现这一目标。

```{exercise}
:label: ck1_ex1

收费公路性质在 $T$ 足够大的情况下，与初始条件 $K_0$ 无关。

请扩展 `plot_paths` 函数，使其能够绘制多个初始点的轨迹，初始点取 `k0s = [k_ss*2, k_ss*3, k_ss/3]`。
```

```{solution-start} ck1_ex1
:class: dropdown
```

参考答案

```{code-cell} ipython3
def plot_multiple_paths(pp, c0, k0s, T_arr, k_ter=0, k_ss=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        
    ylabels = ['$c_t$', '$k_t$', r'$\mu_t$']
    titles = ['消费', '资本', '拉格朗日乘数']

    colors = plt.cm.viridis(np.linspace(0, 1, len(k0s)))
    
    all_c_paths = []
    all_k_paths = []
    
    for i, k0 in enumerate(k0s):
        k0_c_paths = []
        k0_k_paths = []
        
        for T in T_arr:
            c_vec, k_vec = bisection(pp, c0, k0, T, k_ter=k_ter, verbose=False)
            k0_c_paths.append(c_vec)
            k0_k_paths.append(k_vec)

            μ_vec = pp.u_prime(c_vec)
            paths = [c_vec, k_vec, μ_vec]

            for j in range(3):
                axs[j].plot(paths[j], color=colors[i], 
                           label=f'$k_0 = {k0:.2f}$' if j == 0 and T == T_arr[0] else "", alpha=0.7)
                axs[j].set(xlabel='t', ylabel=ylabels[j], title=titles[j])

            if k_ss is not None and i == 0 and T == T_arr[0]:
                axs[1].axhline(k_ss, c='k', ls='--', lw=1)

            axs[1].axvline(T+1, c='k', ls='--', lw=1)
            axs[1].scatter(T+1, paths[1][-1], s=80, color=colors[i])
        
        all_c_paths.append(k0_c_paths)
        all_k_paths.append(k0_k_paths)
    
    # 如果有多个初始点，添加图例
    if len(k0s) > 1:
        axs[0].legend()

    return all_c_paths, all_k_paths
```

```{code-cell} ipython3
_ = plot_multiple_paths(pp, 0.3, [k_ss*2, k_ss*3, k_ss/3], [250, 150, 75, 50], k_ss=k_ss)
```

我们看到，对于不同的初始值 $K_0$，收费公路性质都成立。

```{solution-end}
```

让我们计算并绘制储蓄率。

```{code-cell} ipython3
@jit
def saving_rate(pp, c_path, k_path):
    '给定 c 和 k 的路径，计算储蓄率的路径。'
    production = pp.f(k_path[:-1])

    return (production - c_path) / production
```
```{code-cell} ipython3
def plot_saving_rate(pp, c0, k0, T_arr, k_ter=0, k_ss=None, s_ss=None):

    fix, axs = plt.subplots(2, 2, figsize=(12, 9))

    c_paths, k_paths = plot_paths(pp, c0, k0, T_arr, k_ter=k_ter, k_ss=k_ss, axs=axs.flatten())

    for i, T in enumerate(T_arr):
        s_path = saving_rate(pp, c_paths[i], k_paths[i])
        axs[1, 1].plot(s_path)

    axs[1, 1].set(xlabel='t', ylabel='$s_t$', title='储蓄率')

    if s_ss is not None:
        axs[1, 1].hlines(s_ss, 0, np.max(T_arr), linestyle='--')
```
```{code-cell} ipython3
plot_saving_rate(pp, 0.3, k_ss/3, [250, 150, 75, 50], k_ss=k_ss)
```
## 极限无限期经济

我们要设定 $T = +\infty$。

合适的做法是将终端条件{eq}`constraint4`替换为

$$
\lim_{T \rightarrow +\infty} \beta^T u'(C_T) K_{T+1} = 0
$$

收敛到最优稳态的路径将满足以上条件。

我们可以通过从任意初始值 $K_0$ 开始，向一个较大但有限的$T+1$时期的最优稳态$K$推进来近似最优路径。

在下面的代码中，我们对一个较大的$T$进行这样的计算，并绘制消费、资本和储蓄率。

我们知道在稳态时储蓄率是恒定的，且$\bar s= \frac{f(\bar K)-\bar C}{f(\bar K)}$。

根据{eq}`feasibility-constraint`，稳态储蓄率等于

$$
\bar s =\frac{ \delta \bar{K}}{f(\bar K)}
$$

稳态储蓄量$\bar S = \bar s f(\bar K)$是每期用于抵消资本折旧所需的数量。
我们首先研究从稳态水平以下开始的最优资本路径。

```{code-cell} ipython3
# 稳态储蓄率
s_ss = pp.δ * k_ss / pp.f(k_ss)

plot_saving_rate(pp, 0.3, k_ss/3, [130], k_ter=k_ss, k_ss=k_ss, s_ss=s_ss)
```
由于$K_0<\bar K$，所以$f'(K_0)>\rho +\delta$。

规划者选择一个高于稳态储蓄率的正储蓄率。

注意$f''(K)<0$，因此随着$K$的增加，$f'(K)$下降。

规划者逐渐降低储蓄率，直到达到$f'(K)=\rho +\delta$的稳态。

## 稳定流形和相图

现在我们描述一个经典图表，用来描述最优的 $(K_{t+1}, C_t)$ 路径。

图表的纵轴是 $K$，横轴是 $C$。

对于任意固定的 $K$，消费欧拉方程{eq}`eq:consn_euler`的不动点 $C$ 满足

$$
C=C\left(\beta\left[f^{\prime}\left(f\left(K\right)+\left(1-\delta\right)K-C\right)+\left(1-\delta\right)\right]\right)^{1/\gamma}
$$

这意味着

$$
\begin{aligned}
C &=f\left(K\right)+\left(1-\delta\right)K-f^{\prime-1}\left(\frac{1}{\beta}-\left(1-\delta\right)\right)  \\
 &\equiv \tilde{C} \left(K\right)
\end{aligned}
$$ (eq:tildeC)

正不动点 $C = \tilde C(K)$ 仅在 $f\left(K\right)+\left(1-\delta\right)K-f^{\prime-1}\left(\frac{1}{\beta}-\left(1-\delta\right)\right)>0$ 时存在

```{code-cell} ipython3
@jit
def C_tilde(K, pp):

    return pp.f(K) + (1 - pp.δ) * K - pp.f_prime_inv(1 / pp.β - 1 + pp.δ)
```
接下来，注意，给定任意一个时不变的 $C$,可行性条件 {eq}`allocation` 的不动点 $K$ 满足以下方程

$$
    K = f(K) + (1 - \delta K) - C .
$$

上述方程的不动点可以用函数表示为

$$
K = \tilde K(C)
$$ (eq:tildeK)

```{code-cell} ipython3
@jit
def K_diff(K, C, pp):
    return pp.f(K) - pp.δ * K - C

@jit
def K_tilde(C, pp):

    res = brentq(K_diff, 1e-6, 100, args=(C, pp))

    return res.root
```
稳态 $\left(K_s, C_s\right)$ 是满足方程 {eq}`eq:tildeC` 和 {eq}`eq:tildeK` 的一对 $(K,C)$ 值。

它是我们将在下图 {numref}`stable_manifold` 中绘制的两条曲线 $\tilde{C}$ 和 $\tilde{K}$ 的交点。

我们可以通过求解方程 $K_s = \tilde{K}\left(\tilde{C}\left(K_s\right)\right)$ 来计算 $K_s$

```{code-cell} ipython3
@jit
def K_tilde_diff(K, pp):

    K_out = K_tilde(C_tilde(K, pp), pp)

    return K - K_out
```
```{code-cell} ipython3
res = brentq(K_tilde_diff, 8, 10, args=(pp,))

Ks = res.root
Cs = C_tilde(Ks, pp)

Ks, Cs
```
我们可以使用打靶算法来计算趋近于$\left(K_s, C_s\right)$的轨迹。

对于给定的$K$，让我们计算一个较大$T$（例如$=200$）时的$\vec{C}$和$\vec{K}$。

我们通过二分法算法计算$C_0$，确保$K_T=K_s$。

让我们计算两条朝向$\left(K_s, C_s\right)$的轨迹，这两条轨迹从$K_s$的不同侧开始：$\bar{K}_0=1e-3<K_s<\bar{K}_1=15$。

```{code-cell} ipython3
c_vec1, k_vec1 = bisection(pp, 5, 15, T=200, k_ter=Ks)
c_vec2, k_vec2 = bisection(pp, 1e-3, 1e-3, T=200, k_ter=Ks)
```
以下代码生成图 {numref}`stable_manifold`，这是仿照 {cite}`intriligator2002mathematical` 第411页的一个图表制作的。

图 {numref}`stable_manifold` 是一个经典的"相平面"，其中"状态"变量 $K$ 在纵轴上，"协态"变量 $C$ 在横轴上。

图 {numref}`stable_manifold` 绘制了三条曲线：

  * 蓝线表示方程 {eq}`eq:tildeC` 所描述的不动点 $C = \tilde C (K)$ 的图像。
  * 红线表示方程 {eq}`eq:tildeK` 所描述的不动点 $K = \tilde K(C)$ 的图像。
  * 绿线表示从时间0时任意 $K_0$ 开始收敛到稳态的稳定流形。
     * 对于给定的 $K_0$，射击算法将 $C_0$ 设置为绿线上的坐标，以启动一条收敛到最优稳态的路径。
    * 绿线上的箭头显示了动态方程{eq}`eq:systemdynamics`推动连续对 $(K_{t+1}, C_t)$ 的方向。

除了显示三条曲线外，图{numref}`stable_manifold`还绘制了箭头来指示当给定 $K_0$ 时，$C_0$不在绿线所示稳定流形上时，动态方程{eq}`eq:systemdynamics`驱动系统的方向。

  * 如果对给定的$K_0$，$C_0$设置在绿线以下，则积累了过多的资本
  
  * 如果对给定的$K_0$，$C_0$设置在绿线以上，则积累的资本太少

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 稳定流形和相平面
    name: stable_manifold
tags: [hide-input]
---
fig, ax = plt.subplots(figsize=(7, 5))

K_range = np.arange(1e-1, 15, 0.1)
C_range = np.arange(1e-1, 2.3, 0.1)

# C tilde
ax.plot(K_range, [C_tilde(Ks, pp) for Ks in K_range], color='b')
ax.text(11.8, 4, r'$C=\tilde{C}(K)$', color='b')

# K tilde
ax.plot([K_tilde(Cs, pp) for Cs in C_range], C_range, color='r')
ax.text(2, 1.5, r'$K=\tilde{K}(C)$', color='r')

# stable branch
ax.plot(k_vec1[:-1], c_vec1, color='g')
ax.plot(k_vec2[:-1], c_vec2, color='g')
ax.quiver(k_vec1[5], c_vec1[5],
          k_vec1[6]-k_vec1[5], c_vec1[6]-c_vec1[5],
          color='g')
ax.quiver(k_vec2[5], c_vec2[5],
          k_vec2[6]-k_vec2[5], c_vec2[6]-c_vec2[5],
          color='g')
ax.text(12, 2.5, r'稳定分支', color='g')

# (Ks, Cs)
ax.scatter(Ks, Cs)
ax.text(Ks-1.2, Cs+0.2, '$(K_s, C_s)$')

# arrows
K_range = np.linspace(1e-3, 15, 20)
C_range = np.linspace(1e-3, 7.5, 20)
K_mesh, C_mesh = np.meshgrid(K_range, C_range)

next_K, next_C = pp.next_k_c(K_mesh, C_mesh)
ax.quiver(K_range, C_range, next_K-K_mesh, next_C-C_mesh)

# infeasible consumption area
ax.text(0.5, 5, "不可行\n消费区")

ax.set_ylim([0, 7.5])
ax.set_xlim([0, 15])

ax.set_xlabel('$K$')
ax.set_ylabel('$C$')

plt.show()
```
## 结论

在{doc}`Cass-Koopmans 竞争均衡 <cass_koopmans_2>`中，我们研究了一个去中心化的经济版本，其技术和偏好结构与本讲完全相同。

在那一讲中，我们用亚当·斯密的**看不见的手**替代了本讲中的规划者。

取代规划者做出的数量选择的是市场价格，这些价格由模型外部的一个"机械神"(即所谓的看不见的手)设定。

均衡市场价格必须协调由代表性家庭和代表性企业各自独立做出的不同决策。

像本讲所研究的计划经济与{doc}`Cass-Koopmans 竞争均衡 <cass_koopmans_2>`中研究的市场经济之间的关系是一般均衡理论和福利经济学的基础性主题。

### 练习

```{exercise}
:label: ck1_ex2

- 当初始资本水平设为稳态值的1.5倍时，在以 $T = 130$ 为终点向稳态打靶的过程中，绘制最优消费、资本和储蓄的路径。
- 储蓄率为什么会出现这样的反应？
```

```{solution-start} ck1_ex2
:class: dropdown
```

```{code-cell} ipython3
plot_saving_rate(pp, 0.3, k_ss*1.5, [130], k_ter=k_ss, k_ss=k_ss, s_ss=s_ss)
```

```{solution-end}
```
