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
translation:
  title: 一价和二价拍卖
  headings:
    First-price sealed-bid auction (FPSB): 第一价格密封拍卖(FPSB)
    First-price sealed-bid auction (FPSB)::Characterization of FPSB auction: 一价密封拍卖的特征
    Second-price sealed-bid auction (SPSB): 二价密封拍卖(SPSB)
    Characterization of SPSB auction: 二价密封拍卖的特征
    Uniform distribution of private values: 私人价值的均匀分布
    Setup: 设置
    First price sealed bid auction: 第一价格密封投标拍卖
    Second price sealed bid auction: 第二价格密封拍卖
    Python code: Python代码
    Revenue equivalence theorem: 收入等价定理
    Calculation of  bid price in FPSB: FPSB中出价的计算
    $\chi^2$ Distribution: $\chi^2$ 分布
    Code summary: 代码总结
    Exercises: 练习
    Further reading: 延伸阅读
---

# 一价和二价拍卖

本讲座旨在为后续关于 {doc}`house_auction` 的讲座做铺垫。

在那个讲座中，规划者或拍卖人同时将多个商品分配给一组人。

在本讲座中，我们将讨论如何将单个商品分配给一组人中的一个人。

我们将学习并模拟两种经典的拍卖方式：

* 第一价格密封投标拍卖(FPSB)
* 由William Vickrey创建的第二价格密封投标拍卖(SPSB) {cite}`Vickrey_61`

我们还将学习并应用：

* 收益等价定理

我们建议观看Anders Munk-Nielsen关于第二价格拍卖的视频：

```{youtube} qwWk_Bqtue8
```

以及

```{youtube} eYTGQCGpmXI
```

Anders Munk-Nielsen 将他的代码放在了[GitHub](https://github.com/GamEconCph/Lectures-2021/tree/main/Bayesian%20Games)上。

我们下面的大部分Python代码都基于他的代码。




+++

## 第一价格密封拍卖(FPSB)

+++

**规则：**

* 拍卖一件商品。
* 潜在买家同时提交密封投标。
* 每个投标者只知道自己的投标。
* 商品分配给出价最高的人。
* 中标者支付其投标价格。


**详细设定：**

有$n \geq 2$个潜在买家，编号为$i = 1, 2, \ldots, n$。

买家$i$对被拍卖商品的估值为$v_i$。

买家$i$想要最大化她的预期**剩余价值**，定义为$v_i - p$，其中$p$是她在赢得拍卖的情况下需要支付的价格。

显然，

- 如果$i$的出价恰好是$v_i$，她支付的正是她认为物品值得的价格，不会获得任何剩余价值。
- 买家$i$永远不会想要出价高于$v_i$。
- 如果买家 $i$ 出价 $b < v_i$ 并赢得拍卖，她获得的剩余价值为 $v_i - b > 0$。
- 如果买家 $i$ 出价 $b < v_i$ 而其他人出价高于 $b$，买家 $i$ 就会输掉拍卖且没有剩余价值。
- 要继续进行，买家 $i$ 需要知道她的出价 $v_i$ 作为函数时赢得拍卖的概率
   - 这要求她知道其他潜在买家 $j \neq i$ 的出价 $v_j$ 的概率分布
- 根据她对该概率分布的认知，买家 $i$ 希望设定一个能最大化其剩余价值数学期望的出价。

出价是密封的，所以任何竞标者都不知道其他潜在买家提交的出价。

这意味着竞标者实际上参与的是一个玩家不知道其他玩家**收益**的博弈。

这是一个**贝叶斯博弈**，其纳什均衡被称为**贝叶斯纳什均衡**。

为了完整描述这种情况，我们假设潜在买家的估值是独立且同分布的，其概率分布为所有投标者所知。

投标者会选择低于$v_i$的最优投标价格。

### 一价密封拍卖的特征

我们全文假设：

* 估值在投标者之间是**私有的**且**独立的**
* 投标者是**对称的**：他们的估值都来自一个共同的分布$F$，该分布在其支撑集上连续且严格递增
* 投标者是**风险中性的**

在这些假设下，一价密封拍卖具有唯一的对称、严格递增投标策略的贝叶斯纳什均衡。

由于均衡投标策略是严格递增的，估值最高的投标者会提交最高的出价，从而获胜。

买家$i$的最优投标是

$$
\mathbb{E}[y_{i} | y_{i} < v_{i}]
$$ (eq:optbid1)

其中$v_{i}$是投标者$i$的估值，$y_{i}$是所有其他投标者的最高估值：

$$
y_{i} = \max_{j \neq i} v_{j}
$$ (eq:optbid2)

关于这一结果的推导，请参阅维基百科的 [一价密封拍卖页面](https://en.wikipedia.org/wiki/First-price_sealed-bid_auction)，或 {cite:t}`Krishna2009` 第 2 章。

我们将在下面通过模拟来验证这个公式，{ref}`ta_ex2` 要求你推导一个等价的表达式，使其对任意分布$F$都易于计算。

+++

## 二价密封拍卖(SPSB)

+++

**规则：**在二价密封拍卖(SPSB)中，赢家支付第二高的投标价格。

## 二价密封拍卖的特征

在 SPSB 拍卖中，竞标者最优选择是按其真实价值出价。

形式上，在单一不可分物品的 SPSB 拍卖中，按自己的真实价值出价是一种**弱占优**策略。

之所以说是*弱*占优，是因为无论其他竞标者如何出价，一个按非真实价值出价的竞标者永远不会做得更好，有时甚至会更糟。

请注意，这比 FPSB 的结果强得多：它根本不需要关于其他竞标者估值分布的任何假设，也不需要关于他们如何出价的假设。

关于 Vickrey 拍卖的证明可在[维基百科页面](https://en.wikipedia.org/wiki/Vickrey_auction)找到

+++

## 私人价值的均匀分布

+++

我们假设竞标者 $i$ 的估值 $v_{i}$ 服从分布 $v_{i} \stackrel{\text{IID}}{\sim} U(0,1)$。

在这个假设下，我们可以分析计算 FPSB 和 SPSB 中出价的概率分布。

我们将模拟结果，并通过大数定律验证模拟结果与分析结果一致。

我们可以用我们的模拟来说明**收益等价定理**，该定理断言平均而言，一价和二价密封拍卖为卖家提供相同的收益。

该定理要求我们的两种拍卖都满足以下假设：

* 估值是独立且私人的，竞标者是对称且风险中性的
* 这两种机制都将商品授予估值最高的竞标者
* 估值最低的竞标者预期获得零剩余

在这些假设下，任何满足这些条件的两种机制都会为每个竞标者产生相同的预期支付，从而为卖家带来相同的预期收益。

{ref}`ta_ex4` 展示了当其中一个假设——风险中性——不成立时会发生什么。

要了解收入等价定理，请参阅 [此维基百科页面](https://en.wikipedia.org/wiki/Revenue_equivalence)

+++

## 设置

+++

有 $n$ 个投标人。

每个投标人都知道有 $n-1$ 个其他投标人。

## 第一价格密封投标拍卖

投标人 $i$ 在**第一价格密封投标拍卖**中的最优投标由方程 {eq}`eq:optbid1` 和 {eq}`eq:optbid2` 描述。

当投标是从均匀分布中独立同分布抽取时，$y_{i}$ 的累积分布函数为

$$
\begin{aligned}
\tilde{F}_{n-1}(y) = \mathbb{P}\{y_{i} \leq y\} &= \mathbb{P}\{\max_{j \neq i} v_{j} \leq y\} \\
&= \prod_{j \neq i} \mathbb{P}\{v_{j} \leq y\} \\
&= y^{n-1}
\end{aligned}
$$

且 $y_i$ 的概率密度函数为 $\tilde{f}_{n-1}(y) = (n-1)y^{n-2}$。

那么投标人 $i$ 在**第一价格密封投标拍卖**中的最优投标为：

$$
\begin{aligned}
\mathbb{E}[y_{i} | y_{i} < v_{i}] &= \frac{\int_{0}^{v_{i}} y_{i}\tilde{f}_{n-1}(y_{i})dy_{i}}{\int_{0}^{v_{i}} \tilde{f}_{n-1}(y_{i})dy_{i}} \\
&= \frac{\int_{0}^{v_{i}}(n-1)y_{i}^{n-1}dy_{i}}{\int_{0}^{v_{i}}(n-1)y_{i}^{n-2}dy_{i}} \\
&= \frac{n-1}{n}y_{i}\bigg{|}_{0}^{v_{i}} \\
&= \frac{n-1}{n}v_{i}
\end{aligned}
$$

## 第二价格密封拍卖

在**第二价格密封拍卖**中，对竞价者$i$来说，出价$v_i$是最优选择。

+++

## Python代码

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']
import seaborn as sns
import scipy.stats as stats
import scipy.interpolate as interp

# for plots
plt.rcParams.update({'font.size': 14})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ensure the notebook generates the same randomness
rng = np.random.default_rng(1337)
```

我们重复进行一个有5个投标人的拍卖100,000次。

每个投标人的估值服从均匀分布$U(0,1)$。

```{code-cell} ipython3
N = 5
R = 100_000

v = rng.uniform(0, 1, (N, R))

# BNE in first-price sealed bid

b_star = lambda vi, N: ((N-1)/N) * vi
b = b_star(v,N)
```

我们计算并排序在一价密封拍卖（FPSB）和二价密封拍卖（SPSB）下产生的出价分布。

```{code-cell} ipython3
# Bidders' values are sorted in ascending order in each auction.
# We record the order because we want to apply it to bid price and their id.
idx = np.argsort(v, axis=0)  # 在每次拍卖中，竞买人的估值按升序排列。

# same as np.sort(v, axis=0), except now we retain the idx
v = np.take_along_axis(v, idx, axis=0)  # 与np.sort(v, axis=0)相同，但保留了idx
b = np.take_along_axis(b, idx, axis=0)

# In FPSB and SPSB the winner is the bidder with the highest valuation,
# which after sorting is the last row.
# 在FPSB和SPSB中，赢家是估值最高的竞买人，排序后即为最后一行。

# highest bid
winner_pays_fpsb = b[-1, :]  # 最高出价
# 2nd-highest valuation
winner_pays_spsb = v[-2, :]  # 第二高估值
```

让我们绘制_获胜_出价 $b_{(n)}$（即支付金额）与估值 $v_{(n)}$ 的关系图，分别针对FPSB和SPSB。

注意：

- FPSB：每个估值对应一个唯一的出价
- SPSB：因为支付金额等于第二高出价者的估值，即使固定获胜者的估值，支付金额也会有所变化。所以这里每个估值都对应一个支付金额的频率分布。

```{code-cell} ipython3
# 我们打算计算不同群组投标者的平均支付金额
binned = stats.binned_statistic(v[-1, :], v[-2, :], statistic='mean', bins=20)
xx = binned.bin_edges
xx = [(xx[ii]+xx[ii+1])/2 for ii in range(len(xx)-1)]
yy = binned.statistic

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(xx, yy, label='SPSB平均支付金额')
ax.plot(v[-1, :], b[-1, :], '--', alpha=0.8, label='FPSB解析解')
ax.plot(v[-1, :], v[-2, :], 'o', alpha=0.05, 
                markersize=0.1, label='SPSB：实际出价')

ax.legend(loc='best')
ax.set_xlabel('估值, $v_i$')
ax.set_ylabel('出价, $b_i$')
sns.despine()
```

## 收入等价定理

+++

我们现在从卖方预期获得的收入角度比较第一价格密封拍卖(FPSB)和第二价格密封拍卖(SPSB)。

**FPSB的预期收入：**

估值为$y$的赢家支付$\frac{n-1}{n} y$，其中n是投标者数量。

我们之前计算得出CDF为$F_{n}(y) = y^{n}$，PDF为$f_{n} = ny^{n-1}$。

因此，预期收入为

$$
R = \int_{0}^{1}\frac{n-1}{n}v_{i}\times n v_{i}^{n-1}dv_{i} = \frac{n-1}{n+1}
$$

**SPSB的预期收入：**

预期收入等于n乘以一个投标者的预期支付。

计算得出

$$
\begin{aligned}
\mathrm{TR} &= n\mathbb{E}_{v_i}\left[\mathbb{E}_{y_i}[y_{i}|y_{i} < v_{i}]\mathbb{P}\{y_{i} < v_{i}\} + 0\times\mathbb{P}\{y_{i} > v_{i}\}\right] \\
&= n\mathbb{E}_{v_i}\left[\mathbb{E}_{y_i}[y_{i}|y_{i} < v_{i}]\tilde{F}_{n-1}(v_{i})\right] \\
&= n\mathbb{E}_{v_i}[\frac{n-1}{n} \times v_{i} \times v_{i}^{n-1}] \\
&= (n-1)\mathbb{E}_{v_i}[v_{i}^{n}] \\
&= \frac{n-1}{n+1}
\end{aligned}
$$

+++

因此，虽然两种拍卖方式中的中标价格分布通常不同，但我们推断在FPSB和SPSB中的预期支付是相同的。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))

for payment, label in zip([winner_pays_fpsb, winner_pays_spsb], ['FPSB', 'SPSB']):
    print('The average payment of %s: %.4f. Std.: %.4f. Median: %.4f' % (
        label, payment.mean(), payment.std(), np.median(payment)))
    ax.hist(payment, density=True, alpha=0.6, label=label, bins=100)

ax.axvline(winner_pays_fpsb.mean(), ls='--', c='g', label='Mean')
ax.axvline(winner_pays_spsb.mean(), ls='--', c='r', label='Mean')

ax.legend(loc='best')
ax.set_xlabel('出价')
ax.set_ylabel('密度')
sns.despine()
```

**<center>在$[0,1]$上均匀分布的FPSB和SPSB结果总结</center>**

|    密封投标拍卖    |             一价拍卖              |             二价拍卖              |
| :-----------------------: | :----------------------------------: | :-------------------------------------: |
|          获胜者           |            最高出价者             |             最高出价者              |
|        获胜者支付        |             获胜者出价             |             第二高出价              |
|        失败者支付         |                  0                   |                    0                    |
|     占优策略     |            无占优策略            |          如实出价是占优策略           |
| 贝叶斯纳什均衡 | 投标人$i$出价$\frac{n-1}{n}v_{i}$ |    投标人$i$如实出价$v_{i}$     |
|   拍卖者收益    |          $\frac {n-1}{n+1}$          |           $\frac {n-1}{n+1}$            |

+++

**迂回：计算FPSB的贝叶斯纳什均衡**

收入等价定理让我们可以从SPSB拍卖的结果推导出FPSB拍卖的最优投标策略。

设$b(v_{i})$为FPSB拍卖中的最优出价。

收入等价定理告诉我们，价值为$v_{i}$的投标者在这两种拍卖中平均获得相同的**支付**。

因此，

$$
b(v_{i})\mathbb{P}\{y_{i} < v_{i}\} + 0 \cdot \mathbb{P}\{y_{i} \ge v_{i}\} = \mathbb{E}_{y_{i}}[y_{i} | y_{i} < v_{i}]\mathbb{P}\{y_{i} < v_{i}\} + 0 \cdot \mathbb{P}\{y_{i} \ge v_{i}\}
$$

由此可得，FPSB拍卖中的最优投标策略是$b(v_{i}) = \mathbb{E}_{y_{i}}[y_{i} | y_{i} < v_{i}]$。

+++

## FPSB中出价的计算

+++

在方程{eq}`eq:optbid1`和{eq}`eq:optbid2`中，我们展示了FPSB拍卖中对称贝叶斯纳什均衡的最优出价公式。

$$
\mathbb{E}[y_{i} | y_{i} < v_{i}]
$$

其中
- $v_{i} = $ 投标者$i$的价值
- $y_{i} = $：除了竞标者$i$以外所有竞标者的最大值，即$y_{i} = \max_{j \neq i} v_{j}$

我们之前已经为私人价值呈均匀分布的情况下分析计算出了FPSB拍卖中的最优出价。

对于大多数私人价值的概率分布，解析解并不容易计算。

相反，我们可以根据私人价值的分布情况，以数值方式计算FPSB拍卖中的出价。

```{code-cell} ipython3
def evaluate_largest(v_hat, array, order=1):
    """
    一个用于估算其他竞标者的最大值（或特定顺序的最大值）的方法，
    条件是玩家1赢得拍卖。

    我们估计E[y | y < v_hat]，其中y是除该参考竞标者外
    其他竞标者中的最高估值。我们以竞标者1作为参考竞标者
    （由于估值是独立同分布的，选择哪一个并不重要），
    去掉她所在的行，然后在其他所有竞标者的估值都低于v_hat的
    那些拍卖中，对剩余竞标者中的最高估值取平均值。

    参数：
    ----------
    v_hat：float，参考竞标者的估值。

    array：二维数组，形状为(N,R)的竞标者价值，
          其中N：玩家数量，R：拍卖次数

    order：int。对输家的哪个顺序统计量取平均值。
                order=1给出最高的落败估值，
                order=2给出第二高的估值，以此类推。

    """
    N, R = array.shape

    # 去掉参考竞标者所在的行；条件是其余竞标者都落败
    array_residual = array[1:, :].copy() 

    winning_auctions_mask = (array_residual < v_hat).all(axis=0) 

    num_winning_auctions = np.sum(winning_auctions_mask)

    if num_winning_auctions == 0:
        return np.nan

    array_conditional = array_residual[:, winning_auctions_mask]
    
    array_conditional_sorted = np.sort(array_conditional, axis=0)

    order_largest_bids = array_conditional_sorted[-order, :] 
    
    return np.mean(order_largest_bids)
```

我们可以通过将其与解析解进行比较来检验`evaluate_largest`方法的准确性。

我们发现`evaluate_largest`方法运行良好

```{code-cell} ipython3
v_grid = np.linspace(0.3, 1, 8)
bid_analytical = b_star(v_grid, N)

# 重新抽取估值
v = rng.uniform(0, 1, (N, R))
bid_simulated = [evaluate_largest(ii, v) for ii in v_grid]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(v_grid, bid_analytical, '-', color='k', label='解析解')
ax.plot(v_grid, bid_simulated, '--', color='r', label='模拟值')

ax.legend(loc='best')
ax.set_xlabel('估值, $v_i$')
ax.set_ylabel('出价, $b_i$')
ax.set_title('FPSB的解')
sns.despine()
```

## $\chi^2$ 分布

让我们尝试一个例子，其中私有价值的分布是一个 $\chi^2$ 分布。

我们先通过以下Python代码来了解 $\chi^2$ 分布：

```{code-cell} ipython3
rng = np.random.default_rng(1337)
v = rng.chisquare(df=2, size=(N * R,))

plt.hist(v, bins=50, edgecolor='w')
plt.xlabel('Values: $v$')
plt.show()
```

现在我们让Python构建一个出价函数

```{code-cell} ipython3
rng = np.random.default_rng(1337)
v = rng.chisquare(df=2, size=(N, R))

# 我们计算v的分位数作为我们的网格
pct_quantile = np.linspace(0, 100, 101)[1:-1]
v_grid = np.percentile(v.flatten(), q=pct_quantile)

# 由于缺乏观测值，某些低分位数会返回nan值
EV = [evaluate_largest(ii, v) for ii in v_grid]
```

```{code-cell} ipython3
# 我们在网格和出价函数中插入0作为补充
EV = np.insert(EV, 0, 0)
v_grid = np.insert(v_grid, 0, 0)

b_star_num = interp.interp1d(v_grid, EV, fill_value="extrapolate")
```

我们通过计算和可视化结果来检验我们的出价函数。

```{code-cell} ipython3
pct_quantile_fine = np.linspace(0, 100, 1001)[1:-1]
v_grid_fine = np.percentile(v.flatten(), q=pct_quantile_fine)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(v_grid, EV, 'or', label='网格上的模拟')
ax.plot(v_grid_fine, b_star_num(v_grid_fine), 
                '-', label='插值解')

ax.legend(loc='best')
ax.set_xlabel('估值, $v_i$')
ax.set_ylabel('一价密封拍卖中的最优出价')
sns.despine()
```

现在我们可以使用Python来计算中标者支付价格的概率分布

```{code-cell} ipython3
b = b_star_num(v)

idx = np.argsort(v, axis=0)
# same as np.sort(v, axis=0), except now we retain the idx
v = np.take_along_axis(v, idx, axis=0)
b = np.take_along_axis(b, idx, axis=0)

# highest bid
winner_pays_fpsb = b[-1, :]  # 最高出价
# 2nd-highest valuation
winner_pays_spsb = v[-2, :]  # 第二高估值
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))

for payment, label in zip([winner_pays_fpsb, winner_pays_spsb],
                          ['FPSB', 'SPSB']):
    print('%s的平均支付额：%.4f。标准差：%.4f。中位数：%.4f' % (
        label, payment.mean(), payment.std(), np.median(payment)))
    ax.hist(payment, density=True, alpha=0.6, label=label, bins=100)

ax.axvline(winner_pays_fpsb.mean(), ls='--', c='g', label='均值')
ax.axvline(winner_pays_spsb.mean(), ls='--', c='r', label='均值')

ax.legend(loc='best')
ax.set_xlabel('出价')
ax.set_ylabel('密度')
sns.despine()
```

## 代码总结

+++

我们将使用过的函数整合成一个Python类

```{code-cell} ipython3
class bid_price_solution:

    def __init__(self, array):
        """
        一个可以绘制投标者价值分布、
        计算FPSB中投标者最优投标价格
        并绘制FPSB和SPSB中赢家支付分布的类

        参数:
        ----------

        array: 投标者价值的二维数组，形状为(N, R)，
               其中N: 玩家数量, R: 拍卖次数

        """
        self.value_mat = array.copy()

        return None

    def plot_value_distribution(self):
        plt.hist(self.value_mat.flatten(), bins=50, edgecolor='w')
        plt.xlabel('价值: $v$')
        plt.show()

        return None

    def evaluate_largest(self, v_hat, order=1):
        N, R = self.value_mat.shape

        # 删除第一行，因为我们假设第一行是获胜者的出价
        array_residual = self.value_mat[1:, :].copy() 

        winning_auctions_mask = (array_residual < v_hat).all(axis=0) 

        num_winning_auctions = np.sum(winning_auctions_mask)

        if num_winning_auctions == 0:
            return np.nan

        array_conditional = array_residual[:, winning_auctions_mask]
        array_conditional_sorted = np.sort(array_conditional, axis=0)
        order_largest_bids = array_conditional_sorted[-order, :]

        return np.mean(order_largest_bids)

    def compute_optimal_bid_FPSB(self, plot=True):
        # 我们计算v的分位数作为网格
        pct_quantile = np.linspace(0, 100, 101)[1:-1]
        v_grid = np.percentile(self.value_mat.flatten(), q=pct_quantile)

        # 由于缺乏观察值，某些低分位数会返回nan值
        EV = [self.evaluate_largest(ii) for ii in v_grid]

        # 我们在网格和投标价格函数中插入0作为补充
        EV = np.insert(EV, 0, 0)
        v_grid = np.insert(v_grid, 0, 0)

        self.b_star_num = interp.interp1d(v_grid, EV,
                                           fill_value="extrapolate")

        if not plot:
            return None

        pct_quantile_fine = np.linspace(0, 100, 1001)[1:-1]
        v_grid_fine = np.percentile(self.value_mat.flatten(),
                                    q=pct_quantile_fine)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(v_grid, EV, 'or', label='网格上的模拟')
        ax.plot(v_grid_fine, self.b_star_num(v_grid_fine), 
                            '-', label='插值解')

        ax.legend(loc='best')
        ax.set_xlabel('估值, $v_i$')
        ax.set_ylabel('FPSB中的最优投标')
        sns.despine()

        return None

    def plot_winner_payment_distribution(self):
        if not hasattr(self, 'b_star_num'):     # 出价尚未计算
            self.compute_optimal_bid_FPSB(plot=False)

        self.b = self.b_star_num(self.value_mat)

        idx = np.argsort(self.value_mat, axis=0)
        # same as np.sort(v, axis=0), except now we retain the idx
        self.v = np.take_along_axis(self.value_mat, idx, axis=0)  # 与np.sort(v, axis=0)相同，但保留了idx
        self.b = np.take_along_axis(self.b, idx, axis=0)

        # highest bid
        winner_pays_fpsb = self.b[-1, :]  # 最高投标
        # 2nd-highest valuation
        winner_pays_spsb = self.v[-2, :]  # 第二高估值

        fig, ax = plt.subplots(figsize=(6, 4))

        for payment, label in zip([winner_pays_fpsb, winner_pays_spsb],
                                   ['FPSB', 'SPSB']):
            print('%s的平均支付: %.4f. 标准差: %.4f. 中位数: %.4f' %
                  (label, payment.mean(), payment.std(), np.median(payment)))
            ax.hist(payment, density=True, alpha=0.6, label=label, bins=100)

        ax.axvline(winner_pays_fpsb.mean(), ls='--', c='g', label='均值')
        ax.axvline(winner_pays_spsb.mean(), ls='--', c='r', label='均值')

        ax.legend(loc='best')
        ax.set_xlabel('投标')
        ax.set_ylabel('密度')
        sns.despine()

        return None
```

```{code-cell} ipython3
rng = np.random.default_rng(1337)
v = rng.chisquare(df=2, size=(N, R))

chi_squ_case = bid_price_solution(v)
```

```{code-cell} ipython3
chi_squ_case.plot_value_distribution()
```

```{code-cell} ipython3
chi_squ_case.compute_optimal_bid_FPSB()
```

```{code-cell} ipython3
chi_squ_case.plot_winner_payment_distribution()
```

## 练习

```{exercise}
:label: ta_ex1

通过模拟验证收益等价定理。

对于估值独立地从 $U(0,1)$ 中抽取的 $n = 2, 3, 5, 10$ 个竞拍者，模拟多场拍卖并计算

1. FPSB 拍卖中获胜者的平均支付，其中每个竞拍者出价 $\frac{n-1}{n} v_i$
1. SPSB 拍卖中获胜者的平均支付，其中每个竞拍者出价 $v_i$

将两者与理论预期收益 $\frac{n-1}{n+1}$ 进行比较，并评论卖方收益如何随竞拍者数量变化。
```

```{solution-start} ta_ex1
:class: dropdown
```

```{code-cell} ipython3
R_ex = 200_000
rng_ex = np.random.default_rng(1234)

print(f"{'n':>4}{'FPSB':>12}{'SPSB':>12}{'(n-1)/(n+1)':>14}")
for n in (2, 3, 5, 10):
    v_ex = np.sort(rng_ex.uniform(0, 1, (n, R_ex)), axis=0)
    fpsb = (n - 1)/n * v_ex[-1, :]      # winner's own bid
    spsb = v_ex[-2, :]                  # second highest valuation
    print(f"{n:>4}{fpsb.mean():>12.4f}{spsb.mean():>12.4f}{(n-1)/(n+1):>14.4f}")
```

这两种拍卖产生相同的预期收益，并且随着 $n$ 的增长，二者都收敛于估值可能的最高值。

竞拍者越多，竞争就会将获胜支付推向估值支持集的顶端。

请注意，尽管这两种拍卖平均而言产生相同的收益，但获胜者支付的*分布*是不同的：在 FPSB 拍卖中，支付是获胜者估值的确定性函数，而在 SPSB 拍卖中，支付是次高估值，在给定获胜者估值的情况下这是随机的。

```{solution-end}
```

```{exercise}
:label: ta_ex2

方程 {eq}`eq:optbid1` 表明，FPSB 拍卖中的最优出价为 $\mathbb{E}[y_i \mid y_i < v_i]$。

1. 证明这可以写成

   $$
   b(v) = v - \frac{\int_0^{v} F(x)^{n-1} dx}{F(v)^{n-1}}
   $$

   其中 $F$ 是估值的分布函数。

1. 验证当 $F$ 是 $[0,1]$ 上的均匀分布时，这简化为 $\frac{n-1}{n}v$。

1. 计算估值服从 $\chi^2(2)$ 分布时的公式值，并与讲座中通过模拟计算得到的出价函数进行比较。
```

```{solution-start} ta_ex2
:class: dropdown
```

$y_i = \max_{j \neq i} v_j$ 的分布函数为 $\tilde F_{n-1}(y) = F(y)^{n-1}$。

因此

$$
\mathbb{E}[y \mid y < v] = \frac{1}{F(v)^{n-1}} \int_0^v y \, d\left[F(y)^{n-1}\right] .
$$

分部积分，

$$
\int_0^v y \, d\left[F(y)^{n-1}\right] = v F(v)^{n-1} - \int_0^v F(y)^{n-1} dy ,
$$

由此得到该公式。

对于 $[0,1]$ 上的 $F(x) = x$，我们得到 $b(v) = v - \frac{v^n/n}{v^{n-1}} = \frac{n-1}{n} v$。

这个公式有一个很好的解读：竞拍者将其出价压低到估值以下的幅度，会随着竞争对手数量的增加而缩小。

```{code-cell} ipython3
from scipy.integrate import quad

def b_closed_form(v, F, n):
    "Optimal FPSB bid for a bidder with valuation v when rivals' values ~ F."
    shading = quad(lambda x: F(x)**(n - 1), 0, v)[0] / F(v)**(n - 1)
    return v - shading

# check against the analytical solution for the uniform case
print("uniform check")
for v0 in (0.3, 0.6, 0.9):
    print(f"  v = {v0}:  closed form {b_closed_form(v0, lambda x: x, N):.4f}, "
          f"analytical {b_star(v0, N):.4f}")
```

```{code-cell} ipython3
# now the chi-squared case studied in the lecture
F_chi2 = stats.chi2(df=2).cdf
v_test = np.percentile(v.flatten(), [10, 30, 50, 70, 90])

print(f"{'v':>8}{'closed form':>14}{'simulated':>12}")
for v0 in v_test:
    print(f"{v0:>8.3f}{b_closed_form(v0, F_chi2, N):>14.4f}"
          f"{float(b_star_num(v0)):>12.4f}")
```

闭式解与模拟结果高度一致，这为两者都提供了有用的验证。

```{solution-end}
```

```{exercise}
:label: ta_ex3

本练习要求你理解为什么诚实出价在 SPSB 拍卖中是弱占优策略，而在 FPSB 拍卖中却不是。

固定 $n = 5$，考虑一个估值为 $v = 0.75$ 的竞拍者，其对手的估值服从 $U(0,1)$。

1. 在 SPSB 拍卖中，对手诚实出价。将该竞拍者的预期剩余计算为她自己出价 $b$ 的函数，并绘制图形。
1. 在 FPSB 拍卖中，对手出价 $\frac{n-1}{n}v_j$。计算并绘制她的预期剩余作为 $b$ 的函数。
1. 每条曲线的峰值在哪里？如果她按照自己的估值出价，在 FPSB 拍卖中她能获得多少剩余？
```

```{solution-start} ta_ex3
:class: dropdown
```

在 SPSB 拍卖中，当 $y < b$ 时她获胜，然后支付 $y$，因此她的预期剩余为

$$
\int_0^b (v - y) \, (n-1) y^{n-2} dy .
$$

对 $b$ 求导得到 $(v-b)(n-1)b^{n-2}$，当 $b < v$ 时为正，当 $b > v$ 时为负，因此 $b = v$ 是最优的。

在 FPSB 拍卖中，当每个对手的出价都低于 $b$ 时她获胜，这发生的概率为 $\left(\frac{nb}{n-1}\right)^{n-1}$，然后她支付 $b$。

```{code-cell} ipython3
n_ex, v_own = 5, 0.75
bids = np.linspace(0, 1, 401)

spsb_surplus = [quad(lambda y: (v_own - y)*(n_ex - 1)*y**(n_ex - 2),
                     0, min(bb, 1))[0] for bb in bids]
fpsb_surplus = [(v_own - bb)*min(1, n_ex*bb/(n_ex - 1))**(n_ex - 1)
                for bb in bids]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(bids, spsb_surplus, label='SPSB')
ax.plot(bids, fpsb_surplus, label='FPSB')
ax.axvline(v_own, ls='--', c='k', lw=1, label='own valuation')
ax.axvline((n_ex - 1)/n_ex*v_own, ls=':', c='r', lw=1, label='FPSB optimal bid')
ax.set_xlabel('own bid $b$')
ax.set_ylabel('expected surplus')
ax.legend()
plt.show()

print(f"SPSB surplus is maximized at b = {bids[int(np.argmax(spsb_surplus))]:.3f}")
print(f"FPSB surplus is maximized at b = {bids[int(np.argmax(fpsb_surplus))]:.3f}"
      f"  (theory: {(n_ex - 1)/n_ex*v_own:.3f})")
print(f"FPSB surplus from bidding one's valuation: {(v_own - v_own):.3f}")
```

SPSB 曲线恰好在竞拍者估值处达到峰值。

FPSB 曲线的峰值严格低于该估值，并且在 FPSB 拍卖中按照自己的估值出价所获得的剩余恰好为零：竞拍者获胜的次数更多，但每次获胜时都要支付其全部估值。

```{solution-end}
```

```{exercise}
:label: ta_ex4

收益等价定理要求竞拍者是*风险中性*的。

假设每个竞拍者的效用为 $u(x) = x^\rho$，其中 $0 < \rho \leq 1$，因此 $\rho < 1$ 意味着风险厌恶，且估值服从 $U(0,1)$。

可以证明，FPSB 拍卖中的对称均衡出价变为

$$
b(v) = \frac{n-1}{n-1+\rho} v .
$$

1. 通过数值方法验证这一点：对于 $n = 5$ 且估值 $v = 0.8$ 的竞拍者，在对手使用此规则的情况下，将预期效用计算为她自己出价的函数，并检查其最大值所在位置。
1. 计算 $\rho = 1, 0.6, 0.3$ 时 FPSB 和 SPSB 中卖方的预期收益。
1. 解释其中的直觉。
```

```{solution-start} ta_ex4
:class: dropdown
```

```{code-cell} ipython3
def expected_utility(b, v_own, n, ρ):
    "Expected utility of bidding b when rivals bid (n-1)v/(n-1+ρ)."
    win_prob = np.minimum(1, b*(n - 1 + ρ)/(n - 1))**(n - 1)
    return win_prob * np.maximum(v_own - b, 0)**ρ

n_ex, v_own = 5, 0.8
grid = np.linspace(0.001, v_own, 2001)

print(f"{'ρ':>6}{'theory b*':>12}{'numerical':>12}")
for ρ in (1.0, 0.5, 0.2):
    theory = (n_ex - 1)*v_own/(n_ex - 1 + ρ)
    numerical = grid[int(np.argmax(expected_utility(grid, v_own, n_ex, ρ)))]
    print(f"{ρ:>6}{theory:>12.4f}{numerical:>12.4f}")
```

```{code-cell} ipython3
rng_ra = np.random.default_rng(42)
v_ra = np.sort(rng_ra.uniform(0, 1, (n_ex, 200_000)), axis=0)

print(f"{'ρ':>6}{'FPSB revenue':>15}{'SPSB revenue':>15}")
for ρ in (1.0, 0.6, 0.3):
    fpsb = ((n_ex - 1)/(n_ex - 1 + ρ)) * v_ra[-1, :]
    print(f"{ρ:>6}{fpsb.mean():>15.4f}{v_ra[-2, :].mean():>15.4f}")
```

在风险中性（$\rho = 1$）的情况下，这两种拍卖产生相同的收益，正如定理所述。

在风险厌恶（$\rho < 1$）的情况下，FPSB 拍卖产生*更高*的收益。

其直觉是：在 FPSB 拍卖中，压低出价是一种赌博：它提高了获胜条件下的剩余，但降低了获胜的概率。

风险厌恶的竞拍者不喜欢这种赌博，因此压低出价的幅度会减小，这将收益转移给了卖方。

在 SPSB 拍卖中，获胜者的支付不依赖于她自己的出价，因此风险厌恶不会改变任何东西：按照自己的估值出价仍然是弱占优的，卖方的收益也不受影响。

```{solution-end}
```

## 延伸阅读

第二价格密封投标拍卖由 {cite:t}`Vickrey_61` 提出。

有关本讲座内容的教科书式论述，请参阅 {cite:t}`Krishna2009` 和 {cite:t}`Milgrom2004`。

{cite:t}`Klemperer1999` 对相关文献进行了综述。

上文所述一般形式的收益等价定理源自 {cite:t}`Myerson1981` 和 {cite:t}`RileySamuelson1981`。

本讲座所研究的两种拍卖自然地可以用投标人估值的顺序统计量来描述，这一主题在 {cite:t}`DavidNagaraja2003` 中有详尽论述。
