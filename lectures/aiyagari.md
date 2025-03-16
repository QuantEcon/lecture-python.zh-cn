```---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(aiyagari)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 艾亚加里模型

```{contents} 目录
:depth: 2
```

除了 Anaconda 包含的内容，这一讲还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

## 概述

在本次讲座中，我们将描述一类基于Truman Bewley {cite}`Bewley1977`研究的模型的结构。

我们首先讨论一个由Rao Aiyagari {cite}`Aiyagari1994`提出的Bewley模型的例子。

该模型的特点是

* 异质代理
* 一个用于借贷的外生单一工具
* 对个体代理借款额度的限制

艾亚加里模型已被用于研究许多主题，包括

* 预防性储蓄和流动性约束的影响 {cite}`Aiyagari1994`
* 风险分担和资产定价 {cite}`Heaton1996`
* 财富分布的形状 {cite}`benhabib2015`
* 等等等等

让我们从一些导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
from quantecon.markov import DiscreteDP
from numba import jit
```

### 参考文献

本讲的主要参考文献是 {cite}`Aiyagari1994`。

在 {cite}`Ljungqvist2012` 的第18章中可以找到教科书的解释。

SeHyoun Ahn和Benjamin Moll的连续时间版本模型可以在[这里](https://nbviewer.org/github/QuantEcon/QuantEcon.notebooks/blob/master/aiyagari_continuous_time.ipynb)找到。

## 经济体

### 家庭

无限期存在的家庭/消费者面临着特定的收入冲击。

*事先* 相同的家庭的单位区间面临共同的借款约束。

典型家庭面临的储蓄问题是

$$
\max \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

满足

$$
a_{t+1} + c_t \leq w z_t + (1 + r) a_t
\quad
c_t \geq 0,
\quad \text{and} \quad
a_t \geq -B
$$

其中

* $c_t$ 是当前消费
* $a_t$ 是资产
* $z_t$ 是劳动收入的外生部分，捕捉随机失业风险等
* $w$ 是工资率
* $r$ 是净利率
* $B$ 是代理允许借用的最大金额

外生过程 $\{z_t\}$ 遵循具有给定随机矩阵 $P$ 的有限状态马尔可夫链。

工资和利率在时间上是固定的。

在这个简单版本的模型中，家庭无弹性地供给劳动，因为他们不重视休闲。

## 公司

公司通过雇佣资本和劳动来生产产出。

公司以竞争的方式运作，并面临规模报酬不变。

由于规模报酬不变，公司的数量并不重要。

因此，我们可以考虑单个（但仍然是竞争性的）代表性公司。

公司的产出是

$$
Y_t = A K_t^{\alpha} N^{1 - \alpha}
$$

其中

* $A$ 和 $\alpha$ 是参数，满足 $A > 0$ 且 $\alpha \in (0, 1)$
* $K_t$ 是总资本
* $N$ 是总劳动供给（在这个简单版本的模型中是常数）

公司的问题是

$$
max_{K, N} \left\{ A K_t^{\alpha} N^{1 - \alpha} - (r + \delta) K - w N \right\}
$$
用 $F(K, N) = A K^{\alpha} N^{1 - \alpha}$ 表示产出，公司的最优选择就成为

$$
r + \delta = \partial F / \partial K = A \alpha  \left( \frac{N}{K} \right)^{1 - \alpha}
$$

$$
w = \partial F / \partial N = A  (1 - \alpha)  \left( \frac{K}{N} \right)^{\alpha}
$$

这些方程可以重新排列为

```{math}
:label: aiyagari_r

r = A \alpha  \left( \frac{N}{K} \right)^{1 - \alpha} - \delta
```

和

```{math}
:label: aiyagari_w

w = A  (1 - \alpha)  \left( \frac{K}{N} \right)^{\alpha}
```

因此，这些条件与 {gls}`first_optimal_condition` 和 {gls}`hotellings_rule` 一致。

即：

* 在竞争性均衡中，工资等于 LME 第一个最优条件推导的边际产品
* 净利率等于由这两个条件推导出的“资源减去折旧”的资本边际产品

### 市场清算条件

假设政府不提供任何固定的借款额度，那么市场清算条件就是

$$
K = \int a_i \phi (d a_i, d z_i)
$$

其中 $\phi (d a_i, d z_i)$ 是家庭 $(i = 1, 2,..., n)$ 的${a_i, z_i}$ 分布

## 证明我们的推导

我们将通过对市场清算条件进行近似来验证所做出的推导。
```{code-cell} ipython
import numpy as np
from scipy.optimize import fsolve  # 导入fsolve函数用于非线性方程求解

# 定义参数
alpha = 0.36
beta = 0.96
delta = 0.1
A = 1
N = 1

# 定义对于 J(w) = J(r + delta) 的函数
def equations(p):
    r, K = p
    equation1 = r - (A * alpha * ((N / K) ** (1 - alpha)) - delta)
    equation2 = (beta ** -1 - (1 - delta)) - (A * alpha * ((N / K) ** (1 - alpha)))
    return (equation1, equation2)

r, K =  fsolve(equations, (0.1, 5))  # 初始猜测

print(f'r = {r:.4f}')
print(f'K = {K:.4f}')
```
从上面的模型推导得出 $r$ 和 $K$ 的值

* 这些利用我们早期的均衡条件，即在市场均衡时资本市场清算条件 $J(r + \delta)$ 和家庭的资本供给条件
* 这也符合家庭在市场上的借入限制

## 用动态规划描述家庭问题

为了用动态规划方法描述家庭问题，我们需要定义

* 状态：收入 $z_t$ 和资产 $a_t$ 的状态对 $(z_t, a_t)$ 描述了净财富的初始状态；
* 行动：在资产限额不变的情况下，家庭的行动就是选择他们所拥有的消费 $c_t = (1 + r) a_t + wz_t - a_{t+1} $；
* 转移：在时间 $t+1$，资产从 $a_t$ 转移到 $a_{t+1}$，劳动收入从 $z_t$ 转移到 $z_{t+1}$

然后贝尔曼方程是

$$
v(a, z) = \max_{(a')} \left\{ u((1 + r) a + wz - a') + \beta \mathbb{E}_z' v(a', z') \right\}
$$

定义的约束是 $\underline{a} \le a' \le \overline{a}$。

我们将更详细地讨论这些定义。
（{term}`state space`）将在不同的价格体系中运行。

家庭在此标记上的问题变得简单：

1. 定义状态 $(a, z)$ 和行动 $a'$。
2. 找一个有效的 $\pi$，将约束编码为转移矩阵 $Q$。
3. 通过迭代约束和收益来估计贝尔曼方程的近似解。

我们在前面介绍了一种解决技术，它将 [DiscreteDP](https://python-advanced.quantecon.org/pack/discreteecon.html#DiscreteDP) 和向后归纳方法的数据结构结合了线性程序和动态规划的启发式。有关详细信息，请参见相应的指南。

## 操作标准代码 {cite}`evans2022`

下面给出的代码有时会运行得很慢，具体取决于硬件设备。

我们告诉你如何使用 [Numba](https://numba.pydata.org) （一个简单易用的Python代码优化器/加速器）加速代码。

对想要使用（紧密）有限状态空间的读取者而言，一个好的起始点是[Quantitative Economics with Python](https://python-advanced.quantecon.org)讲座系列。）

在这里，我们把状态定义为 $s_t := (a_t, z_t)$，其中 $a_t$ 是资产而 $z_t$ 是震荡。

行动是选择下一个时期的资产水平 $a_{t+1}$。

我们使用Numba加速循环，这样我们可以在参数变化时高效地更新矩阵。

该类还包括一组默认参数，除非另有说明，我们将采用这些参数。

```{code-cell} python3
class Household:
    """
    本类获取定义家庭资产积累问题的参数，并计算生成DiscreteDP实例所需的相应奖励和转移矩阵R和Q，从而求解最优策略。

    有关索引的注释：我们需要将状态空间S枚举为一个序列
    S = {0, ..., n}。为此， (a_i, z_i) 索引对根据以下规则映射到 s_i 索引：

        s_i = a_i * z_size + z_i

    若要反转此映射，请使用

        a_i = s_i // z_size  (整数除法)
        z_i = s_i % z_size

    """


    def __init__(self,
                r=0.01,                      # 利率
                w=1.0,                       # 工资
                β=0.96,                      # 折扣因子
                a_min=1e-10,
                Π=[[0.9, 0.1], [0.1, 0.9]],  # 马尔可夫链
                z_vals=[0.1, 1.0],           # 外生状态
                a_max=18,
                a_size=200):

        # 存储值，设置 a 和 z 的网格
        self.r, self.w, self.β = r, w, β
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size

        self.Π = np.asarray(Π)
        self.z_vals = np.asarray(z_vals)
        self.z_size = len(z_vals)

        self.a_vals = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size

        # 构建数组 Q
        self.Q = np.zeros((self.n, a_size, self.n))
        self.build_Q()

        # 构建数组 R
        self.R = np.empty((self.n, a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        使用此方法重新设定价格。调用此方法将触发 R 的重新构建。
        """
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.Π)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R,
                self.a_size,
                self.z_size,
                self.a_vals,
                self.z_vals,
                self.r,
                self.w)


# 使用JIT函数进行主要工作

@jit
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z + (1 + r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = np.log(c)  # 效用

@jit
def populate_Q(Q, a_size, z_size, Π):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i*z_size + next_z_i] = Π[z_i, next_z_i]


@jit
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i*z_size + z_i]
    return a_probs
```
一个使用例子如下：

```{code-cell} python3
# 示例价格
r = 0.03
w = 0.956

# 创建Household实例
am = Household(a_max=20, r=r, w=w)

# 使用该实例构建一个离散动态程序
am_ddp = DiscreteDP(am.R, am.Q, am.β)

# 使用策略函数迭代求解
results = am_ddp.solve(method='policy_iteration')

# 简化名称
z_size, a_size = am.z_size, am.a_size
z_vals, a_vals = am.z_vals, am.a_vals
n = a_size * z_size

# 获得在z固定的每一行上的a索引集的所有最优动作
a_star = np.empty((z_size, a_size))
for s_i in range(n):
    a_i = s_i // z_size
    z_i = s_i % z_size
    a_star[z_i, a_i] = a_vals[results.sigma[s_i]]

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(a_vals, a_vals, 'k--')  # 45度
for i in range(z_size):
    lb = f'$z = {z_vals[i]:.2}$'
    ax.plot(a_vals, a_star[i, :], lw=2, alpha=0.6, label=lb)
    ax.set_xlabel('现期资产')
    ax.set_ylabel('下一期资产')
ax.legend(loc='upper left')

plt.show()
```
该图显示了不同外生状态下的资产积累策略。

现在我们希望计算均衡。

让我们首先通过视觉方法来尝试。

下面的代码绘制了供给和需求曲线。

它们的交点给出了均衡利率和资本。

```{code-cell} python3
A = 1.0
N = 1.0
α = 0.33
β = 0.96
δ = 0.05

def r_to_w(r):
    """
    与给定利率r相关的均衡工资。
    """
    return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))

def rd(K):
    """
    资本需求的逆曲线。与给定资本需求量K相关的利率。
    """
    return A * α * (N / K)**(1 - α) - δ

def prices_to_capital_stock(am, r):
    """
    将价格映射到资本存量的诱发水平。

    参数:
    ----------

    am : Household
        一个aiyagari_household.Household的实例
    r : float
        利率
    """
    w = r_to_w(r)
    am.set_prices(r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, β)
    # 计算最优策略
    results = aiyagari_ddp.solve(method='policy_iteration')
    # 计算平稳分布
    stationary_probs = results.mc.stationary_distributions[0]
    # 提取资产的边际分布
    asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)
    # 返回K
    return np.sum(asset_probs * am.a_vals)

# 创建Household实例
am = Household(a_max=20)

# 使用该实例构建一个离散动态程序
am_ddp = DiscreteDP(am.R, am.Q, am.β)

# 创建一个r值网格，在该网格上计算资本的需求和供给
num_points = 20
r_vals = np.linspace(0.005, 0.04, num_points)

# 计算资本供给
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(am, r)

# 与公司对资本的需求进行比较
fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='资本供给')
ax.plot(k_vals, rd(k_vals), lw=2, alpha=0.6, label='资本需求')
ax.grid()
ax.set_xlabel('资本')
ax.set_ylabel('利率')
ax.legend(loc='upper right')

plt.show()
```