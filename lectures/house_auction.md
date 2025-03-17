---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 多种商品分配机制

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install prettytable
```

## 概述

本讲介绍两种将 $n$ 个私人物品("房屋")分配给 $m$ 个人("买家")的机制。

我们假设 $m > n$，即潜在买家数量多于房屋数量。

潜在买家将这些房屋视为**替代品**。

买家 $j$ 对房屋 $i$ 的估值为 $v_{ij}$。

这些估值是**私人的**

  * $v_{ij}$ 只有买家 $j$ 知道，除非买家 $j$ 选择告诉他人。

我们要求机制最多将一套房屋分配给一个潜在买家。

我们将描述两种不同的机制

 * 多轮递增出价拍卖

 * Groves-Clarke机制{cite}`Groves_73`, {cite}`Clarke_71`的一个特例，其中有一个善意的社会规划者

```{note}
1994年，斯坦福大学实际使用了多轮递增出价拍卖的方式，将校园内9块地块的租约出售给符合条件的教职员工。
```

我们首先概述这两种机制。

## 多商品递增出价拍卖

拍卖由一名**拍卖师**主持

拍卖师有一个 $n \times 1$ 的向量 $r$，表示 $n$ 套房屋的保留价格。

拍卖师只有在某套房屋的最终出价超过 $r_i$ 时才会出售该房屋

拍卖师**同时**分配所有 $n$ 套房屋

拍卖师不知道竞买人对房屋的私人估值 $v_{ij}$

拍卖包含多个**轮次**

 - 在每轮中，活跃参与者可以对任何 $n$ 套房屋中的一套进行出价

 - 每个竞买人在一轮中只能对一套房屋出价

 - 在上一轮中成为某套房屋最高出价者的人，在下一轮将自动保持对该房屋的相同出价

 - 在轮次之间，未成为最高出价者的竞买人可以改变其选择竞价的房屋

 - 当所有房屋的价格在相邻两轮之间都没有变化时，拍卖结束

 - 所有 $n$ 套房屋在最后一轮后都将被分配

- 如果没有潜在买家出价超过 $r_i$，房屋 $i$ 将由拍卖人保留

在这次拍卖中，个人 $j$ 从不向其他人透露他/她的私人估值 $v_{ij}$




## 仁慈的规划者

这个机制的设计使所有潜在买家自愿向**社会规划者**透露他们的私人估值，规划者利用这些信息构建一个社会最优配置。

在所有可行的配置中，**社会最优配置**使所有潜在买家的私人估值总和最大化。

规划者提前告知每个人他/她将如何根据潜在买家报告的估值矩阵来分配房屋。

该机制为每个潜在买家提供动机，使其向规划者透露自己的私人估值向量。

在规划者收到每个人的私人价值向量后，规划者部署一个**顺序**算法来确定房屋的**分配**以及向获得者收取的一系列**费用**，这些费用是为了补偿他们的存在对其他潜在买家造成的负面**外部性**。

## 分配的等价性

值得注意的是，这两种机制可以产生几乎相同的分配结果。

我们用Python代码实现这两种机制。

我们还会手动或半手动计算一些示例。

接下来，让我们深入了解细节。

## 递增出价拍卖

### 基本设置

我们从更详细的情况描述开始。

* 一个卖家拥有$n$套房屋，他想以最高可能的价格将其出售给$m$个潜在的合格买家。

* 卖家最多只想向每个潜在买家出售一套房屋。

* 有$m$个潜在的合格买家，用$j = [1, 2, \ldots, m]$来标识

* 每个潜在买家最多只能购买一套房屋。

    * 买家 $j$ 愿意为房屋 $i$ 支付的最高价格是 $v_{ij}$。

    * 买家 $j$ 知道 $v_{ij}, i= 1, \ldots , n$，但其他人不知道。

    * 如果买家 $j$ 为房屋 $i$ 支付 $p_i$，他获得的剩余价值为 $v_{ij} - p_i$。

    * 每个买家 $j$ 想要选择能使其剩余价值 $v_{ij} - p_i$ 最大化的房屋 $i$。

    * 卖家想要使 $\sum_i p_i$ 最大化。

卖家进行一个**同步的、多商品的升价拍卖**。

拍卖的结果包括：

  * 一个 $n \times 1$ 的销售价格向量 $p = [p_1, \ldots, p_n]$，表示 $n$ 套房屋的价格。

  * 一个由 $0$ 和 $1$ 组成的 $n \times m$ 矩阵 $Q$，其中 $Q_{ij} = 1$ 当且仅当买家 $j$ 购买了房屋 $i$。

  * 一个 $n \times m$ 的剩余价值矩阵 $S$，除非买家 $j$ 购买了房屋 $i$，否则矩阵元素均为零；如果买家 $j$ 购买了房屋 $i$，则 $S_{ij} = v_{ij} - p_i$

+++

我们用**伪代码**来描述拍卖规则。

伪代码将为编写实现拍卖的Python代码提供路线图。

+++

## 伪代码

这里简要概述了我们Python代码可能的简单结构

**输入：**

- $n, m$
- 一个 $n \times m$ 的非负矩阵 $v$，表示私人估值
- 一个 $n \times 1$ 的向量 $r$，表示卖方指定的保留价格
- 卖方不会接受低于房屋 $i$ 的保留价格 $r_i$ 的价格
- 我们可以将这些保留价格视为第 $m+1$ 个虚拟买家的私人估值，该买家实际上不参与拍卖
- 初始出价可以从 $r$ 开始
- 卖方指定的最小加价幅度标量 $\epsilon$

在拍卖的每一轮中，对房屋的新出价必须至少为目前最高出价**加上** $\epsilon$

**拍卖规则**

- 拍卖包含有限数量的**轮次**
- 在每一轮中，潜在买家只能对一个房屋出价

- 每轮结束后，出价最高的人将暂时获得该房屋
    - 每套房屋的暂时中标价会被公布
    - 这为进入下一轮做好准备
- 进行新一轮竞拍
    - 上一轮暂时中标者的出价仍然保留在他们竞拍的房屋上；上一轮的暂时中标者保持其出价不变
    - 所有其他活跃的潜在买家必须对某套房屋提交新的出价
    - 对某套房屋的新出价必须至少等于上一轮的暂时中标价**加上**$\epsilon$
    - 如果一个人既没有提交新的出价，又不是上一轮的暂时中标者，那么这个人必须永久退出拍卖

- 对于每个房屋,会公布最高出价(无论是新出价还是上一轮的临时中标价),并且出价最高的人将(暂时)获得该房屋以开始下一轮
- 轮次持续进行,直到**所有**房屋的价格相比上一轮都没有变化
- 房屋以最终轮次中标者的出价价格售出

**输出:**
- 一个 $n \times 1$ 的销售价格向量 $p$
- 一个 $n \times m$ 的剩余价值矩阵 $S$,除非买家 $j$ 购买了房屋 $i$,此时 $S_{ij} = v_{ij} - p_i$,否则均为零
- 一个 $n \times (m+1)$ 的由 $0$ 和 $1$ 组成的矩阵 $Q$,用于表示哪个买家购买了哪个房屋。(最后一列用于记录未售出的房屋。)

**建议的买家策略:**

在以下伪代码和实际Python代码中,我们假设所有买家都选择使用以下策略

   * 该策略对每个买家来说都是最优的

每个买家 $j = 1, \ldots, m$ 使用相同的策略。

该策略的形式为：
- 令 $\check p^t$ 为第 $t$ 轮开始时的 $n \times 1$ 最高出价向量
- 令 $\epsilon>0$ 为卖方规定的最小加价幅度
- 对于每个潜在买家 $j$，计算在第 $t$ 轮最适合竞价的房屋索引，即
$\hat i_t = \textrm{argmax}_i\{  [  v_{ij} - \check p^t_i - \epsilon  ]\}$
- 如果 $\max_i\{  [  v_{ij} - \check p^t_i - \epsilon  ]\} $ $\leq$ $0$，则买家 $j$ 在第 $t$ 轮永久退出拍卖
- 如果 $v_{\hat i_t, j} - \check p^t_i - \epsilon>0$，则买家 $j$ 对房屋 $j$ 出价 $\check p^t_i + \epsilon$

**解决歧义**：我们目前描述的协议存在两个可能的歧义来源。

(1) **买家在每轮中的最优出价选择。** 买家可能对多个房屋有相同的剩余价值。Python中的argmax函数总是返回第一个最大值元素。我们更倾向于在这些获胜者中随机选择。因此，我们在下面编写了自己的argmax函数。

(2) **当多个买家出价相同时卖家的获胜者选择。** 为了解决这种模糊性，我们使用下面的np.random.choice函数。

鉴于结果的随机性，相同的输入可能会产生不同的房屋分配。

然而，这种情况只会在出价价格增量$\epsilon$不可忽略时发生。

```{code-cell} ipython3
import numpy as np
import prettytable as pt

np.random.seed(100)
```

```{code-cell} ipython3
np.set_printoptions(precision=3, suppress=True)
```

## 示例

+++

在构建 Python 类之前，让我们先一步一步地"手动"解决问题，以便理解拍卖是如何进行的。

逐步解决的方法也有助于减少错误，特别是当价值矩阵比较特殊时（例如，价值之间的差异可以忽略不计，某列包含相同的值，或多个买家具有相同的估值等）。

幸运的是，我们的拍卖算法对各种特殊矩阵都表现良好且稳健。

我们将在本讲稍后提供一些示例。



```{code-cell} ipython3
v = np.array([[8, 5, 9, 4],
              [4, 11, 7, 4],
              [9, 7, 6, 4]])
n, m = v.shape
r = np.array([2, 1, 0])
ϵ = 1
p = r.copy()
buyer_list = np.arange(m)
house_list = np.arange(n)
```

```{code-cell} ipython3
v
```

请记住，列索引 $j$ 表示买家，行索引 $i$ 表示房屋。

上述价值矩阵 $v$ 的特殊之处在于买家3（从0开始索引）对每套待售房屋都赋予相同的价值 $4$。

也许买家3是一名官僚，他只是按照上级的指示来购买这些房屋。

```{code-cell} ipython3
r
```

```{code-cell} ipython3
def find_argmax_with_randomness(v):
    """
    我们构建自己的argmax函数，当存在多个最大值时，随机返回其中一个最大值的索引。
    这个函数类似于np.argmax(v,axis=0)

    参数：
    ----------
    v: 2维np.array

    """

    n, m = v.shape
    index_array = np.arange(n)
    result=[]

    for ii in range(m):
        max_value = v[:,ii].max()
        result.append(np.random.choice(index_array[v[:,ii] == max_value]))

    return np.array(result)
```

```{code-cell} ipython3
def present_dict(dt):
    """
    一个以表格形式展示信息的函数。

    参数：
    ----------
    dt：字典。

    """

    ymtb = pt.PrettyTable()
    ymtb.field_names = ['房屋编号', *dt.keys()]
    ymtb.add_row(['买家', *dt.values()])
    print(ymtb)
```

**检查启动条件**

```{code-cell} ipython3
def check_kick_off_condition(v, r, ϵ):
    """
    一个检查在给定保留价格和价值矩阵的情况下是否可以启动拍卖的函数。
    为了避免保留价格过高导致没有人愿意在第一轮出价的情况。

    参数：
    ----------
    v：形状为(n,m)的价值矩阵。

    r：保留价格

    ϵ：每轮最小价格增量

    """

    # 我们将价格向量转换为与价值矩阵相同形状的矩阵以便于减法运算
    p_start = (ϵ+r)[:,None] @ np.ones(m)[None,:]

    surplus_value = v - p_start
    buyer_decision = (surplus_value > 0).any(axis = 0)
    return buyer_decision.any()
```

```{code-cell} ipython3
check_kick_off_condition(v, r, ϵ)
```

### 第一轮

+++

**提交出价**

```{code-cell} ipython3
def submit_initial_bid(p_initial, ϵ, v):
    """
    描述第一轮竞价信息的函数。

    参数：
    ----------
    p_initial: 拍卖开始时的价格（或保留价格）

    v: 价值矩阵

    ϵ: 每轮最小加价幅度

    返回：
    ----------
    p: 本轮竞价后的价格数组

    bid_info: 包含竞价信息的字典（房屋编号为键，买家为值）

    """

    p = p_initial.copy()
    p_start_mat = (ϵ + p)[:,None] @ np.ones(m)[None,:]
    surplus_value = v - p_start_mat

    # 我们只关注具有正剩余价值的活跃买家
    active_buyer_diagnosis = (surplus_value > 0).any(axis = 0)
    active_buyer_list = buyer_list[active_buyer_diagnosis]
    active_buyer_surplus_value = surplus_value[:,active_buyer_diagnosis]
    active_buyer_choice = find_argmax_with_randomness(active_buyer_surplus_value)
    # choice表示在当前价格和ϵ下的最喜欢的房屋

    # 我们只保留唯一的房屋索引，因为价格在一轮中只增加一次
    house_bid = list(set(active_buyer_choice))
    p[house_bid] += ϵ

    bid_info = {}
    for house_num in house_bid:
        bid_info[house_num] = active_buyer_list[active_buyer_choice == house_num]

    return p, bid_info
```

```{code-cell} ipython3
p, bid_info = submit_initial_bid(p, ϵ, v)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

**检查终止条件**

+++

注意到两个买家对房屋2（从0开始索引）进行竞价。

由于拍卖协议没有规定这种情况下的选择规则，我们**随机**选择一个赢家。

这是合理的，因为卖家无法区分这些买家，也不知道每个买家的估值。

对他来说，随机选择一个赢家既方便又实用。

买家3有50%的概率被选为房屋2的赢家，尽管他对房屋的估值低于买家0。

在这种情况下，买家0必须以更高的价格再次竞价，从而挤出买家3。

因此，最终价格可能是3或4，这取决于最后一轮的赢家。

```{code-cell} ipython3
def check_terminal_condition(bid_info, p, v):
    """
    检查拍卖是否结束的函数。

    请记住，当失败者对每个房屋都没有正的剩余价值，
    或者没有失败者（每个买家都得到一个房屋）时，拍卖结束。

    参数：
    ----------
    bid_info：包含房屋编号（作为键）和买家（作为值）竞价信息的字典。

    p：np.array。房屋价格数组

    v：价值矩阵

    返回：
    ----------
    allocation：描述竞价房屋如何分配的字典。

    winner_list：赢家列表

    loser_list：失败者列表

    """

    # 可能有几个买家竞价一个房屋，我们随机选择一个赢家
    winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]

    allocation = {house_num:winner for house_num,winner in zip(bid_info.keys(),winner_list)}

    loser_set = set(buyer_list).difference(set(winner_list))
    loser_list = list(loser_set)
    loser_num = len(loser_list)

    if loser_num == 0:
        print('拍卖结束，因为每个买家都得到了一个房屋。')
        return allocation,winner_list,loser_list

    p_mat = (ϵ + p)[:,None] @ np.ones(loser_num)[None,:]
    loser_surplus_value = v[:,loser_list] - p_mat
    loser_decision = (loser_surplus_value > 0).any(axis = 0)

    print(~(loser_decision.any()))

    return allocation,winner_list,loser_list
```

```{code-cell} ipython3
分配,获胜者列表,失败者列表 = 检查终止条件(出价信息, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

```{code-cell} ipython3
获胜者列表
```

```{code-cell} ipython3
失败者列表
```

### 第二轮

+++

从第二轮开始，拍卖的进行方式与第一轮不同。

现在只有活跃的失败者（那些具有正剩余价值的人）才有动机提交出价，以取代上一轮的临时获胜者。

```{code-cell} ipython3
def submit_bid(loser_list, p, ϵ, v, bid_info):
    """
    一个在第一轮之后执行出价操作的函数。
    第一轮之后，只有活跃的失败者会以旧价格加增量作为新的出价。
    通过这样的出价，上一轮的获胜者被活跃的失败者取代。

    参数：
    ----------
    loser_list：包含失败者索引的列表

    p：np.array。房屋价格数组

    ϵ：出价的最小增量

    v：价值矩阵

    bid_info：包含房屋编号（作为键）和买家（作为值）的出价信息字典。

    返回：
    ----------
    p_end：此轮出价后的价格数组

    bid_info：包含更新后出价信息的字典。

    """

    p_end=p.copy()

    loser_num = len(loser_list)
    p_mat = (ϵ + p_end)[:,None] @ np.ones(loser_num)[None,:]
    loser_surplus_value = v[:,loser_list] - p_mat
    loser_decision = (loser_surplus_value > 0).any(axis = 0)

    active_loser_list = np.array(loser_list)[loser_decision]
    active_loser_surplus_value = loser_surplus_value[:,loser_decision]
    active_loser_choice = find_argmax_with_randomness(active_loser_surplus_value)

    # 我们保留唯一的房屋索引并增加相应的出价价格
    house_bid = list(set(active_loser_choice))
    p_end[house_bid] += ϵ

    # 我们记录来自活跃失败者的出价信息
    bid_info_active_loser = {}
    for house_num in house_bid:
        bid_info_active_loser[house_num] = active_loser_list[active_loser_choice == house_num]

    # 我们根据活跃失败者的出价更新出价信息
    for house_num in bid_info_active_loser.keys():
        bid_info[house_num] = bid_info_active_loser[house_num]

    return p_end,bid_info
```

```{code-cell} ipython3
p,bid_info = submit_bid(失败者列表, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

```{code-cell} ipython3
分配,获胜者列表,失败者列表 = 检查终止条件(出价信息, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### 第三轮

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

```{code-cell} ipython3
分配,获胜者列表,失败者列表 = 检查终止条件(出价信息, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### 第四轮

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

注意，买家3现在转而竞标房屋1，因为他意识到房屋2不再是他的最佳选择。

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

### 第5轮

```{code-cell} ipython3
p,bid_info = submit_bid(loser_list, p, ϵ, v, bid_info)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
present_dict(bid_info)
```

现在买家1再次对房屋1出价4，挤出了买家3，标志着拍卖的结束。

```{code-cell} ipython3
allocation,winner_list,loser_list = check_terminal_condition(bid_info, p, v)
```

```{code-cell} ipython3
present_dict(allocation)
```

```{code-cell} ipython3
# 对于未售出的房屋

house_unsold_list = list(set(house_list).difference(set(allocation.keys())))
house_unsold_list
```

```{code-cell} ipython3
total_revenue = p[list(allocation.keys())].sum()
total_revenue
```

## Python类

+++

上面我们逐步模拟了一个递增出价拍卖。

在定义函数时,由于Python函数执行完后会丢失变量,我们反复计算了一些中间对象。

这当然导致了代码中的冗余

将上述所有代码收集到一个记录所有回合信息的类中会更有效率。

```{code-cell} ipython3
class ascending_bid_auction:

    def __init__(self, v, r, ϵ):
        """
        一个模拟房屋递增出价拍卖的类。

        给定买家的价值矩阵、卖家的保留价格和最小出价增量，
        该类可以执行递增出价拍卖并逐轮展示信息直至结束。

        参数:
        ----------
        v: 二维价值矩阵

        r: 保留价格的np.array

        ϵ: 最小出价增量

        """

        self.v = v.copy()
        self.n,self.m = self.v.shape
        self.r = r
        self.ϵ = ϵ
        self.p = r.copy()
        self.buyer_list = np.arange(self.m)
        self.house_list = np.arange(self.n)
        self.bid_info_history = []
        self.allocation_history = []
        self.winner_history = []
        self.loser_history = []


    def find_argmax_with_randomness(self, v):
        n,m = v.shape
        index_array = np.arange(n)
        result=[]

        for ii in range(m):
            max_value = v[:,ii].max()
            result.append(np.random.choice(index_array[v[:,ii] == max_value]))

        return np.array(result)


    def check_kick_off_condition(self):
        # 我们将价格向量转换为与价值矩阵相同形状的矩阵以便于相减
        p_start = (self.ϵ + self.r)[:,None] @ np.ones(self.m)[None,:]
        self.surplus_value = self.v - p_start
        buyer_decision = (self.surplus_value > 0).any(axis = 0)
        return buyer_decision.any()


    def submit_initial_bid(self):
        # 我们打算找到每个买家的最优选择
        p_start_mat = (self.ϵ + self.p)[:,None] @ np.ones(self.m)[None,:]
        self.surplus_value = self.v - p_start_mat

        # 我们只关心有正剩余价值的活跃买家
        active_buyer_diagnosis = (self.surplus_value > 0).any(axis = 0)
        active_buyer_list = self.buyer_list[active_buyer_diagnosis]
        active_buyer_surplus_value = self.surplus_value[:,active_buyer_diagnosis]
        active_buyer_choice = self.find_argmax_with_randomness(active_buyer_surplus_value)

        # 我们只保留唯一的房屋索引因为价格在一轮中只增加一次
        house_bid =  list(set(active_buyer_choice))
        self.p[house_bid] += self.ϵ

        bid_info = {}
        for house_num in house_bid:
            bid_info[house_num] = active_buyer_list[active_buyer_choice == house_num]
        self.bid_info_history.append(bid_info)

        print('出价信息为')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['房屋编号', *bid_info.keys()]
        ymtb.add_row(['买家', *bid_info.values()])
        print(ymtb)

        print('房屋的出价为')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['房屋编号', *self.house_list]
        ymtb.add_row(['价格', *self.p])
        print(ymtb)

        self.winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]
        self.winner_history.append(self.winner_list)

        self.allocation = {house_num:[winner] for house_num,winner in zip(bid_info.keys(),self.winner_list)}
        self.allocation_history.append(self.allocation)

        loser_set = set(self.buyer_list).difference(set(self.winner_list))
        self.loser_list = list(loser_set)
        self.loser_history.append(self.loser_list)

        print('获胜者为')
        print(self.winner_list)

        print('失败者为')
        print(self.loser_list)
        print('\n')


    def check_terminal_condition(self):
        loser_num = len(self.loser_list)

        if loser_num == 0:
            print('拍卖结束因为每个买家都得到了一套房子。')
            print('\n')
            return True

        p_mat = (self.ϵ + self.p)[:,None] @ np.ones(loser_num)[None,:]
        self.loser_surplus_value = self.v[:,self.loser_list] - p_mat
        self.loser_decision = (self.loser_surplus_value > 0).any(axis = 0)

        return ~(self.loser_decision.any())


    def submit_bid(self):
        bid_info = self.allocation_history[-1].copy()  # 我们只记录获胜者的出价信息

        loser_num = len(self.loser_list)
        p_mat = (self.ϵ + self.p)[:,None] @ np.ones(loser_num)[None,:]
        self.loser_surplus_value = self.v[:,self.loser_list] - p_mat
        self.loser_decision = (self.loser_surplus_value > 0).any(axis = 0)

        active_loser_list = np.array(self.loser_list)[self.loser_decision]
        active_loser_surplus_value = self.loser_surplus_value[:,self.loser_decision]
        active_loser_choice = self.find_argmax_with_randomness(active_loser_surplus_value)

        # 我们保留唯一的房屋索引并增加相应的出价
        house_bid = list(set(active_loser_choice))
        self.p[house_bid] += self.ϵ

        # 我们记录来自活跃失败者的出价信息
        bid_info_active_loser = {}
        for house_num in house_bid:
            bid_info_active_loser[house_num] = active_loser_list[active_loser_choice == house_num]

        # 我们根据活跃失败者的出价更新出价信息
        for house_num in bid_info_active_loser.keys():
            bid_info[house_num] = bid_info_active_loser[house_num]
        self.bid_info_history.append(bid_info)

        print('出价信息为')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['房屋编号', *bid_info.keys()]
        ymtb.add_row(['买家', *bid_info.values()])
        print(ymtb)

        print('房屋的出价为')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['房屋编号', *self.house_list]
        ymtb.add_row(['价格', *self.p])
        print(ymtb)

        self.winner_list=[np.random.choice(bid_info[ii]) for ii in bid_info.keys()]
        self.winner_history.append(self.winner_list)

        self.allocation = {house_num:[winner] for house_num,winner in zip(bid_info.keys(),self.winner_list)}
        self.allocation_history.append(self.allocation)

        loser_set = set(self.buyer_list).difference(set(self.winner_list))
        self.loser_list = list(loser_set)
        self.loser_history.append(self.loser_list)

        print('获胜者为')
        print(self.winner_list)

        print('失败者为')
        print(self.loser_list)
        print('\n')


    def start_auction(self):
        print('房屋递增出价拍卖')
        print('\n')

        print('基本信息：%d套房屋，%d位买家'%(self.n, self.m))

        print('价值矩阵如下')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['买家编号', *(np.arange(self.m))]
        for ii in range(self.n):
            ymtb.add_row(['房屋%d'%(ii), *self.v[ii,:]])
        print(ymtb)

        print('房屋的保留价格为')
        ymtb = pt.PrettyTable()
        ymtb.field_names = ['房屋编号', *self.house_list]
        ymtb.add_row(['价格', *self.r])
        print(ymtb)
        print('最小出价增量为%.2f' % self.ϵ)
        print('\n')

        ctr = 1
        if self.check_kick_off_condition():
            print('拍卖成功开始')
            print('\n')
            print('第%d轮'% ctr)

            self.submit_initial_bid()

            while True:
                if self.check_terminal_condition():
                    print('拍卖结束')
                    print('\n')

                    print('最终结果如下')
                    print('\n')
                    print('分配方案为')
                    ymtb = pt.PrettyTable()
                    ymtb.field_names = ['房屋编号', *self.allocation.keys()]
                    ymtb.add_row(['买家', *self.allocation.values()])
                    print(ymtb)

                    print('房屋的出价为')
                    ymtb = pt.PrettyTable()
                    ymtb.field_names = ['房屋编号', *self.house_list]
                    ymtb.add_row(['价格', *self.p])
                    print(ymtb)

                    print('获胜者为')
                    print(self.winner_list)

                    print('失败者为')
                    print(self.loser_list)

                    self.house_unsold_list = list(set(self.house_list).difference(set(self.allocation.keys())))
                    print('未售出的房屋为')
                    print(self.house_unsold_list)

                    self.total_revenue = self.p[list(self.allocation.keys())].sum()
                    print('总收入为%.2f' % self.total_revenue)

                    break

                ctr += 1
                print('第%d轮'% ctr)
                self.submit_bid()

            # 我们计算1.1中要求的剩余矩阵S和数量矩阵X
            self.S = np.zeros((self.n, self.m))
            for ii,jj in zip(self.allocation.keys(),self.allocation.values()):
                self.S[ii,jj] = self.v[ii,jj] - self.p[ii]

            self.Q = np.zeros((self.n, self.m + 1))  # 最后一列记录未售出的房屋
            for ii,jj in zip(self.allocation.keys(),self.allocation.values()):
                self.Q[ii,jj] = 1
            for ii in self.house_unsold_list:
                self.Q[ii,-1] = 1

            # 我们按房屋编号对分配结果进行排序
            house_sold_list = list(self.allocation.keys())
            house_sold_list.sort()

            dict_temp = {}
            for ii in house_sold_list:
                dict_temp[ii] = self.allocation[ii]
            self.allocation = dict_temp

        else:
            print('由于保留价格过高，拍卖无法开始')
```

让我们使用我们的类来进行上述示例中描述的拍卖。

```{code-cell} ipython3
v = np.array([[8,5,9,4],[4,11,7,4],[9,7,6,4]])
r = np.array([2,1,0])
ϵ = 1

auction_1 = ascending_bid_auction(v, r, ϵ)

auction_1.start_auction()
```

```{code-cell} ipython3
# 剩余矩阵 S

auction_1.S
```

```{code-cell} ipython3
# 数量矩阵 X

auction_1.Q
```

## 稳健性检验

让我们通过将代码应用于具有不同私人价值矩阵的拍卖来进行压力测试。

**1. 房屋数量 = 买家数量**

```{code-cell} ipython3
v2 = np.array([[8,5,9],[4,11,7],[9,7,6]])

auction_2 = ascending_bid_auction(v2, r, ϵ)

auction_2.start_auction()
```

**2. 多个超额买家**

```{code-cell} ipython3
v3 = np.array([[8,5,9,4,3],[4,11,7,4,6],[9,7,6,4,2]])

auction_3 = ascending_bid_auction(v3, r, ϵ)

auction_3.start_auction()
```

**3. 房屋数量多于买家数量**

```{code-cell} ipython3
v4 = np.array([[8,5,4],[4,11,7],[9,7,9],[6,4,5],[2,2,2]])
r2 = np.array([2,1,0,1,1])

auction_4 = ascending_bid_auction(v4, r2, ϵ)

auction_4.start_auction()
```

**4. 一些房屋的保留价格极高**

```{code-cell} ipython3
v5 = np.array([[8,5,4],[4,11,7],[9,7,9],[6,4,5],[2,2,2]])
r3 = np.array([10,1,0,1,1])

auction_5 = ascending_bid_auction(v5, r3, ϵ)

auction_5.start_auction()
```

**5. 保留价格太高以至于拍卖无法开始**

```{code-cell} ipython3
r4 = np.array([15,15,15])

auction_6 = ascending_bid_auction(v, r4, ϵ)

auction_6.start_auction()
```

+++

## Groves-Clarke 机制

+++

我们现在描述另一种方式，让社会将 $n$ 套房子分配给 $m$ 个潜在买家，以实现所有潜在买家的总价值最大化。

我们继续假设每个买家最多只能购买一套房子。

这个机制是 Groves-Clarke 机制的一个特例 {cite}`Groves_73`, {cite}`Clarke_71`。

其特殊结构大大简化了用 Python 代码寻找最优分配的过程。

我们的机制是这样运作的：

* 价值 $V_{ij}$ 是个人 $j$ 的私人信息

* 该机制使每个人 $j$ 愿意向社会规划者告知他们对所有房子 $i = 1, \ldots, n$ 的私人估值 $V_{i,j}$

* 社会规划者要求所有潜在竞标者告知他们的私人估值 $V_{ij}$

* 社会规划者不向任何人透露这些信息，而是用它们来分配房屋和设定价格

* 该机制的设计使所有潜在买家都愿意向规划者透露他们的私人估值

   - 对每个潜在买家来说，说真话都是占优策略

* 规划者通过计算找到具有最高私人估值的房屋和买家配对
   $(\tilde i, \tilde j) = \operatorname{argmax} (V_{ij})$

* 规划者将房屋 $\tilde i$ 分配给买家 $\tilde j$

* 规划者向买家 $\tilde j$ 收取价格 $\max_{- \tilde j} V_{\tilde i,  j}$，其中 $- \tilde j$ 表示除 $\tilde j$ 外的所有 $j$

* 规划者通过从 $V$ 中删除行（即房屋）$\tilde i$ 和列（即买家）$\tilde j$ 来创建剩余房屋 $-\tilde i$ 的私人估值矩阵
  - （但在此过程中，规划者会记录买家和房屋的真实名称）

* 规划者返回到原始步骤并重复

* 规划者重复迭代直到所有 $n$ 套房屋都被分配且所有 $n$ 套房屋的价格都被确定

+++

## 手工解决的示例

+++

让我们看看Groves-Clarke算法如何处理以下简单的私人估值矩阵$V$

$$
V =\begin{bmatrix} 10 & 9 & 8 & 7 & 6 \cr
                    9 & 9 & 7 & 6 & 6 \cr
                    8 & 6 & 6 & 9 & 4 \cr
                    7 & 5 & 6 & 4 & 9 \end{bmatrix}
$$

**注意：** 在第一步中，当最高私人估值对应多个房屋-竞价者配对时，我们选择具有最高售价的配对。如果最高售价对应多个具有最高私人估值的配对，我们随机选择其中一个。

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6],  # 记录原始值
                   [9, 9, 7, 6, 6],
                   [8, 6, 6, 9, 4],
                   [7, 5, 6, 4, 9]])
V = np.copy(V_orig)  # 用于迭代
n, m = V.shape
p = np.zeros(n) # 房屋价格
Q = np.zeros((n, m)) # 记录房屋和买家的状态
```

**第一个任务**

首先，我们找出具有最高私人估值的房屋和竞价者配对。

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

所以，房屋0将以9的价格卖给买家0。然后我们更新房屋0的销售价格和状态矩阵Q。

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
p, Q
```

然后我们从矩阵$V$中移除第0行和第0列。为了保持实际房屋和买家的数量，我们将这一行和这一列设为-1，这将产生与移除它们相同的效果，因为$V \geq 0$。

```{code-cell} ipython3
V[i, :] = -1
V[:, j] = -1
V
```

**第二个任务**

我们再次找出具有最高私人价值的房屋和竞标者配对。

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

在这个特殊的例子中，有三对数据(1, 1)、(2, 3)和(3, 4)具有相同的最高私人价值。为了解决这个问题，我们选择具有最高销售价格的那一对。

```{code-cell} ipython3
p_candidate = np.zeros(len(i))
for k in range(len(i)):
    p_candidate[k] = np.max(np.delete(V[i[k], :], j[k]))
k, = np.where(p_candidate==np.max(p_candidate))
i, j = i[k], j[k]
i, j
```

所以，房屋1将以价格7卖给买家1。我们更新矩阵。

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
p, Q, V
```

**第三个作业**

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

在这个特殊例子中，有两对(2, 3)和(3, 4)具有相同的最高私人价值。

为了解决分配问题，我们选择具有最高销售价格的那一对。

```{code-cell} ipython3
p_candidate = np.zeros(len(i))
for k in range(len(i)):
    p_candidate[k] = np.max(np.delete(V[i[k], :], j[k]))
k, = np.where(p_candidate==np.max(p_candidate))
i, j = i[k], j[k]
i, j
```

这两对甚至有相同的销售价格。

我们随机选择一对。

```{code-cell} ipython3
k = np.random.choice(len(i))
i, j = i[k], j[k]
i, j
```

最后，房屋2将卖给买家3。

我们相应地更新矩阵。

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
p, Q, V
```

**第四个作业**

```{code-cell} ipython3
i, j = np.where(V==np.max(V))
i, j
```

房屋3将出售给买家4。

最终结果如下。

```{code-cell} ipython3
p[i] = np.max(np.delete(V[i, :], j))
Q[i, j] = 1
V[i, :] = -1
V[:, j] = -1
S = V_orig*Q - np.diag(p)@Q
p, Q, V, S
```

## 另一个Python类

将我们的计算组装在一个Python类中是很高效的。

```{code-cell} ipython3
class GC_Mechanism:

    def __init__(self, V):
        """
        实现特殊的Groves Clarke房屋拍卖机制。

        参数：
        ----------
        V: 二维私人估值矩阵

        """

        self.V_orig = V.copy()
        self.V = V.copy()
        self.n, self.m = self.V.shape
        self.p = np.zeros(self.n)
        self.Q = np.zeros((self.n, self.m))
        self.S = np.copy(self.Q)

    def find_argmax(self):
        """
        找出具有最高估值的房屋-买家配对。
        当最高私人估值对应多个房屋-买家配对时，
        我们选择具有最高售价的配对。
        此外，如果最高售价对应多个具有最高私人估值的配对，
        我们随机选择其中一个。

        参数：
        ----------
        V: 二维私人估值矩阵，其中-1表示已移除的行和列

        返回：
        ----------
        i: 售出房屋的索引

        j: 买家的索引

        """
        i, j = np.where(self.V==np.max(self.V))

        if (len(i)>1):
            p_candidate = np.zeros(len(i))
            for k in range(len(i)):
                p_candidate[k] = np.max(np.delete(self.V[i[k], :], j[k]))
            k, = np.where(p_candidate==np.max(p_candidate))
            i, j = i[k], j[k]

            if (len(i)>1):
                k = np.random.choice(len(i))
                k = np.array([k])
                i, j = i[k], j[k]
        return i, j

    def update_status(self, i, j):
        self.p[i] = np.max(np.delete(self.V[i, :], j))
        self.Q[i, j] = 1
        self.V[i, :] = -1
        self.V[:, j] = -1

    def calculate_surplus(self):
        self.S = self.V_orig*self.Q - np.diag(self.p)@self.Q

    def start(self):
        while (np.max(self.V)>=0):
            i, j = self.find_argmax()
            self.update_status(i, j)
            print("房屋%i以价格%i卖给了买家%i"%(i[0], self.p[i[0]], j[0]))
            print("\n")
        self.calculate_surplus()
        print("房屋价格：\n", self.p)
        print("\n")
        print("状态矩阵：\n", self.Q)
        print("\n")
        print("剩余价值矩阵：\n", self.S)

```

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6],
                   [9, 9, 7, 6, 6],
                   [8, 6, 6, 9, 4],
                   [7, 5, 6, 4, 9]])
gc_mechanism = GC_Mechanism(V_orig)
gc_mechanism.start()
```

### 详细说明

在此我们使用一些额外的符号，这些符号是为了与VCG文献中的标准符号保持一致。

我们要验证我们的伪代码确实是一个**轴心机制**，也称为**VCG**（Vickrey-Clarke-Groves）机制。

  * 该机制以{cite}`Groves_73`、{cite}`Clarke_71`和{cite}`Vickrey_61`的名字命名。

为了准备验证，我们添加一些符号。

令$X$为在上述协议下可行的房屋分配集合（即每人最多分配一套房屋）。

令$X(v)$为机制针对私人价值矩阵$v$所选择的分配。

该机制将私人价值矩阵$v$映射到$x \in X$。

令$v_j(x)$为个人$j$对分配$x \in X$所赋予的价值。

令$\check t_j(v)$为机制向个人$j$收取的支付。

VCG机制选择的分配为

$$
X(v)  = \operatorname{argmax}_{x \in X} \sum_{j=1}^m v_j(x)
$$ (eq:GC1)

并向个人 $j$ 收取"社会成本"

$$
\check t_j(v) = \max_{x \in  X} \sum_{k \neq j} v_k(x) -  \sum_{k \neq j} v_k(X(v))
$$ (eq:GC2)

在我们的情况下，方程 {eq}`eq:GC1` 表明VCG分配是为了最大化成功购房者的总价值。

在我们的情况下，方程 {eq}`eq:GC2` 表明该机制向人们收取他们在社会中的存在对其他潜在买家造成的外部性费用。

因此，根据方程 {eq}`eq:GC2` 可以注意到：

- 未成功的潜在买家支付 $0$，因为将他们从"社会"中移除不会影响机制选择的分配

- 成功的潜在买家支付的金额是：在没有他们存在的情况下社会可以实现的总价值，与在机制下社会中其他人实际实现的总价值之间的差额。

上述伪代码中描述的广义第二价格拍卖确实满足条件(1)。

我们要计算 $\check t_j$ （$j = 1, \ldots, m$）并与第二价格拍卖中的 $p_j$ 进行比较。

+++

### 社会成本

使用 GC_Mechanism 类，我们可以计算每个买家的社会成本。

让我们看一个更简单的例子，私人价值矩阵为

$$
V =\begin{bmatrix} 10 & 9 & 8 & 7 & 6 \cr
                    9 & 8 & 7 & 6 & 6 \cr
                    8 & 7 & 6 & 5 & 4 \end{bmatrix}
$$

首先，我们实现 GC 机制并查看结果。

```{code-cell} ipython3
np.random.seed(666)

V_orig = np.array([[10, 9, 8, 7, 6],
                   [9, 8, 7, 6, 6],
                   [8, 7, 6, 5, 4]])
gc_mechanism = GC_Mechanism(V_orig)
gc_mechanism.start()
```

我们排除买家0并计算分配。

```{code-cell} ipython3
V_exc_0 = np.copy(V_orig)
V_exc_0[:, 0] = -1
V_exc_0
gc_mechanism_exc_0 = GC_Mechanism(V_exc_0)
gc_mechanism_exc_0.start()
```

计算买家0的社会成本。

```{code-cell} ipython3
print("买家0的社会成本：",
     np.sum(gc_mechanism_exc_0.Q*gc_mechanism_exc_0.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 0, axis=1)))
```

对买家1和买家2重复此过程

```{code-cell} ipython3
V_exc_1 = np.copy(V_orig)
V_exc_1[:, 1] = -1
V_exc_1
gc_mechanism_exc_1 = GC_Mechanism(V_exc_1)
gc_mechanism_exc_1.start()

print("\n买家1的社会成本：",
     np.sum(gc_mechanism_exc_1.Q*gc_mechanism_exc_1.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 1, axis=1)))
```

```{code-cell} ipython3
V_exc_2 = np.copy(V_orig)
V_exc_2[:, 2] = -1
V_exc_2
gc_mechanism_exc_2 = GC_Mechanism(V_exc_2)
gc_mechanism_exc_2.start()

print("\n买家2的社会成本：",
     np.sum(gc_mechanism_exc_2.Q*gc_mechanism_exc_2.V_orig)-np.sum(np.delete(gc_mechanism.Q*gc_mechanism.V_orig, 2, axis=1)))
```

