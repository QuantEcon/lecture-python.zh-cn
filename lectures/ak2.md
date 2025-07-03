---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 重叠世代模型中的转换

除了 Anaconda 中的内容外，本讲还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade quantecon
```
## 简介

本讲介绍了由 Peter Diamond {cite}`diamond1965national` 提出的由两期生存的重叠世代组成的生命周期模型。

我们将介绍 Auerbach 和 Kotlikoff (1987) {cite}`auerbach1987dynamic` 在第二章中分析的版本。

Auerbach 和 Kotlikoff (1987) 使用他们的两期模型作为分析长寿人群的重叠世代模型的热身,这是他们这本书的主要主题。

他们的两期生存重叠世代模型是一个有用的起点,因为

* 它阐述了在给定日期存活的不同世代个体之间相互作用的结构
* 它激活了政府和后续几代人面临的力量和权衡
* 它是研究政府税收和补贴计划与发行和偿还政府债务政策之间联系的良好实验室
* 一些涉及从一个稳态到另一个稳态转变的有趣实验可以手工计算
* 这是一个很好的场景，用于说明求解具有初始和终止条件的非线性差分方程组的**打靶法**

```{note}
Auerbach 和 Kotlikoff 使用计算机代码来计算他们的长寿人群模型的转换路径。
```

我们擅自扩展了 Auerbach 和 Kotlikoff 的第二章模型，以研究一些在不同代际之间重新分配资源的安排

  * 这些安排采取一系列特定年龄的一次性税收和转移支付的形式

我们研究这些安排如何影响资本积累和政府债务

## 设置

时间是离散的，用 $t=0, 1, 2, \ldots$ 表示。

经济永远存在，但其中的人不会。

在每个时间点 $t \geq 0$，都有一个有代表性的老年人和一个有代表性的年轻人存活。
在时间 $t$ 时,一个有代表性的老年人与一个有代表性的年轻人共存,这个年轻人将在时间 $t+1$ 时成为老年人。

我们假设人口规模随时间保持不变。

一个年轻人工作、储蓄和消费。

一个老年人不工作,而是消费储蓄。

政府永远存在,即在 $t=0, 1, 2, \ldots$ 时。

每个时期 $t \geq 0$,政府征税、支出、转移支付和借贷。

在模型外部设定的时间 $t=0$ 的初始条件是

* $K_0$ -- 由一个有代表性的初始老年人在时间 $t=0$ 带入的初始资本存量
* $D_0$ -- 在 $t=0$ 到期并由一个有代表性的老年人在时间 $t=0$ 拥有的政府债务
  
$K_0$ 和 $D_0$ 都以时间 $0$ 的商品单位来衡量。

一个政府**政策**由五个序列 $\{G_t, D_t, \tau_t, \delta_{ot}, \delta_{yt}\}_{t=0}^\infty$ 组成,其组成部分是
* $\tau_t$ -- 在时间 $t$ 对工资、资本收益和政府债券征收的统一税率
 * $D_t$ -- 在时间 $t$ 到期的一期政府债券本金,按人均计算
 * $G_t$ -- 政府在时间 $t$ 的商品购买,按人均计算
 * $\delta_{yt}$ -- 在时间 $t$ 对每个年轻人征收的一次性税收
 * $\delta_{ot}$ -- 在时间 $t$ 对每个老年人征收的一次性税收


  
一个**配置**是一组序列 $\{C_{yt}, C_{ot}, K_{t+1}, L_t,  Y_t, G_t\}_{t=0}^\infty $;序列的组成部分包括

 * $K_t$ -- 人均物质资本
 * $L_t$ -- 人均劳动
 * $Y_t$ -- 人均产出

以及

* $C_{yt}$ -- 时间 $t \geq 0$ 时年轻人的消费
* $C_{ot}$ -- 时间 $t \geq 0$ 时老年人的消费 
* $K_{t+1} - K_t \equiv I_t $ -- 时间 $t \geq 0$ 时对物质资本的投资
* $G_t$ -- 政府购买

国民收入和产品账户由一系列等式组成
* $Y_t = C_{yt} + C_{ot} + (K_{t+1} - K_t) + G_t, \quad t \geq 0$ 

一个**价格系统**是一对序列 $\{W_t, r_t\}_{t=0}^\infty$；价格序列的组成部分包括生产要素的租金率

* $W_t$ -- 在时间 $t \geq 0$ 时劳动力的租金率
* $r_t$ -- 在时间 $t \geq 0$ 时资本的租金率


## 生产

有两种生产要素，物质资本 $K_t$ 和劳动力 $L_t$。  

资本不会折旧。  

初始资本存量 $K_0$ 由有代表性的初始老年人拥有，他在时间 $0$ 时将其出租给企业。

在时间 $t$ 的净投资率 $I_t$ 为

$$
I_t = K_{t+1} - K_t
$$

时间 $t$ 的资本存量来自于累积过去的投资率：

$$
K_t = K_0 + \sum_{s=0}^{t-1} I_s 
$$

柯布-道格拉斯技术将物质资本 $K_t$ 和劳动服务 $L_t$ 转化为产出 $Y_t$

$$
Y_t  = K_t^\alpha L_t^{1-\alpha}, \quad \alpha \in (0,1)
$$ (eq:prodfn)


## 政府
在时间 $t-1$ 时，政府发行一期无风险债务，承诺在时间 $t$ 时支付人均 $D_t$ 单位的时间 $t$ 商品。

时间 $t$ 的年轻人购买在时间 $t+1$ 到期的政府债务 $D_{t+1}$。

在时间 $t$ 发行的政府债务在时间 $t+1$ 的税前净利率为 $r_{t}$。

时间 $t \geq 0$ 时的政府预算约束为

$$
D_{t+1} - D_t = r_t D_t + G_t - T_t
$$

或




$$
D_{t+1} = (1 + r_t)  D_t + G_t - T_t  .
$$ (eq:govbudgetsequence) 

总税收减去转移支付等于 $T_t$，满足


$$
T_t = \tau_t W_t L_t + \tau_t r_t (D_t + K_t) + \delta_{yt} + \delta_{ot}
$$




## 要素市场中的活动

**老年人：** 在每个 $t \geq 0$ 时，一个有代表性的老年人

   * 将 $K_t$ 和 $D_t$ 带入该期，
   * 将资本租给一个有代表性的公司，租金为 $r_{t} K_t$，
   * 为其租金和利息收入 $\tau_t r_t (K_t+ D_t)$ 缴税，
* 向政府支付一次性税款 $\delta_{ot}$，
   * 将 $K_t$ 卖给年轻人。  


  **年轻人：** 在每个 $t \geq 0$ 时期，一个有代表性的年轻人 
   * 以工资 $W_t$ 向有代表性的公司出售一单位劳动服务，
   * 为其劳动收入支付税款 $\tau_t W_t$
   * 向政府支付一次性税款 $\delta_{yt}$， 
   * 将 $C_{yt}$ 用于消费，
   * 获得非负资产 $A_{t+1}$，其中包括实物资本 $K_{t+1}$ 和在 $t+1$ 到期的一期政府债券 $D_{t+1}$ 之和。

```{note}
如果一次性税款为负，意味着政府向个人支付补贴。
``` 


## 有代表性公司的问题 

有代表性的公司以有竞争力的工资率 $W_t$ 从年轻人那里雇佣劳动服务，并以有竞争力的租金率 $r_t$ 从老年人那里雇佣资本。 

资本的租金率 $r_t$ 等于政府一期债券的利率。

租金率的单位是：
* 对于 $W_t$，每单位劳动力在时间 $t$ 的产出  
* 对于 $r_t$，每单位资本在时间 $t$ 的产出 


我们将时间 $t$ 的产出作为*计价单位*，因此时间 $t$ 的产出价格为1。

企业在时间 $t$ 的利润为 

$$
K_t^\alpha L_t^{1-\alpha} - r_t K_t - W_t L_t . 
$$

为了最大化利润，企业将边际产品等同于租金率：

$$
\begin{aligned}
W_t & = (1-\alpha) K_t^\alpha L_t^{-\alpha} \\
r_t & = \alpha K_t^\alpha L_t^{1-\alpha}
\end{aligned}
$$  (eq:firmfonc)

产出可以被老年人或年轻人消费；或出售给用它来增加资本存量的年轻人；或出售给政府用于在模型中不会为人们带来效用的用途（即，"它被扔进了海洋"）。  


因此，企业将产出出售给老年人、年轻人和政府。









## 个人问题

### 初始老年人
在时间 $t=0$ 时,一个有代表性的初始老年人拥有 $(1 + r_0(1 - \tau_0)) A_0$ 的初始资产。

他必须向政府支付一笔一次性税款(如果为正)或从政府获得补贴(如果为负)$\delta_{ot}$。

一个老年人的预算约束为

$$
C_{o0} = (1 + r_0 (1 - \tau_0)) A_0 - \delta_{ot} .
$$ (eq:hbudgetold)

初始老年人的效用函数为 $C_{o0}$,因此这个人的最优消费计划由方程 {eq}`eq:hbudgetold` 给出。

### 年轻人

在每个 $t \geq 0$ 时,一个年轻人非弹性地提供一单位劳动,作为回报获得 $W_t$ 单位产出的税前劳动收入。

一个年轻人的税后转移收入为 $W_t (1 - \tau_t) - \delta_{yt}$。

在每个 $t \geq 0$ 时,一个年轻人选择消费计划 $C_{yt}, C_{ot+1}$ 来最大化柯布-道格拉斯效用函数

$$
U_t  = C_{yt}^\beta C_{o,t+1}^{1-\beta}, \quad \beta \in (0,1)
$$ (eq:utilfn)
根据以下时期 $t$ 和 $t+1$ 的预算约束:

$$
\begin{aligned}
C_{yt} + A_{t+1} & =  W_t (1 - \tau_t) - \delta_{yt} \\
C_{ot+1} & = (1+ r_{t+1} (1 - \tau_{t+1}))A_{t+1} - \delta_{ot}
\end{aligned}
$$ (eq:twobudgetc)


求解 {eq}`eq:twobudgetc` 的第二个方程得到储蓄 $A_{t+1}$,并将其代入第一个方程,可得到现值预算约束

$$
C_{yt} + \frac{C_{ot+1}}{1 + r_{t+1}(1 - \tau_{t+1})} = W_t (1 - \tau_t) - \delta_{yt} - \frac{\delta_{ot}}{1 + r_{t+1}(1 - \tau_{t+1})}
$$ (eq:onebudgetc)

为了求解年轻人的选择问题,构建拉格朗日函数

$$ 
\begin{aligned}
{\mathcal L}  & = C_{yt}^\beta C_{o,t+1}^{1-\beta} \\ &  + \lambda \Bigl[ C_{yt} + \frac{C_{ot+1}}{1 + r_{t+1}(1 - \tau_{t+1})} - W_t (1 - \tau_t) + \delta_{yt} + \frac{\delta_{ot}}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr],
\end{aligned}
$$ (eq:lagC)

其中 $\lambda$ 是跨期预算约束 {eq}`eq:onebudgetc` 的拉格朗日乘子。
经过几行代数运算，跨期预算约束 {eq}`eq:onebudgetc` 和最大化 ${\mathcal L}$ 关于 $C_{yt}, C_{ot+1}$ 的一阶条件意味着最优消费计划满足

$$
\begin{aligned}
C_{yt} & = \beta \Bigl[ W_t (1 - \tau_t) - \delta_{yt} - \frac{\delta_{ot}}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr] \\
\frac{C_{0t+1}}{1 + r_{t+1}(1-\tau_{t+1})  } & = (1-\beta)   \Bigl[ W_t (1 - \tau_t) - \delta_{yt} - \frac{\delta_{ot}}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr] 
\end{aligned}
$$ (eq:optconsplan)

最小化拉格朗日函数 {eq}`eq:lagC` 关于拉格朗日乘子 $\lambda$ 的一阶条件恢复了预算约束 {eq}`eq:onebudgetc`，使用 {eq}`eq:optconsplan` 给出最优储蓄计划

$$
A_{t+1} = (1-\beta) [ (1- \tau_t) W_t - \delta_{yt}] + \beta \frac{\delta_{ot}}{1 + r_{t+1}(1 - \tau_{t+1})} 
$$ (eq:optsavingsplan)


(sec-equilibrium)=
## 均衡
**定义：** 均衡是一种资源配置、政府政策和价格体系，具有以下特性：
* 给定价格体系和政府政策，资源配置解决了
    * 对于 $t \geq 0$ 的代表性企业的问题
    * 对于 $t \geq 0$ 的个人问题
* 给定价格体系和资源配置，政府预算约束在所有 $t \geq 0$ 时都得到满足。


## 后续步骤


为了开始我们对均衡结果的分析，我们将研究 Auerbach 和 Kotlikoff (1987) {cite}`auerbach1987dynamic` 在第2章开始分析的模型的特殊情况。

它可以手工求解。

我们接下来将这样做。

在我们推导出封闭形式解之后，我们将假装我们不知道并将计算均衡结果路径。

我们将通过首先将均衡表示为从一系列要素价格和税率到一系列要素价格和税率的映射的不动点来做到这一点。
我们将通过迭代收敛到那个映射来计算均衡。


## 闭式解

为了得到 Auerbach 和 Kotlikoff (1987) {cite}`auerbach1987dynamic` 第2章的特殊情况，我们将 $\delta_{ot}$ 和 $\delta_{yt}$ 都设为零。

作为 {eq}`eq:optconsplan` 的特殊情况，我们计算代表性年轻人的如下消费-储蓄计划：


$$
\begin{aligned}
C_{yt} & = \beta (1 - \tau_t) W_t \\
A_{t+1} &= (1-\beta) (1- \tau_t) W_t
\end{aligned}
$$

使用 {eq}`eq:firmfonc` 和 $A_t = K_t + D_t$，我们得到资本的如下闭式转移规律：

$$
K_{t+1}=K_{t}^{\alpha}\left(1-\tau_{t}\right)\left(1-\alpha\right)\left(1-\beta\right) - D_{t}\\
$$ (eq:Klawclosed)

### 稳态

从 {eq}`eq:Klawclosed` 和政府预算约束 {eq}`eq:govbudgetsequence`，我们计算**时不变**或**稳态值** $\hat K, \hat D, \hat T$：

$$
\begin{aligned}
\hat{K} &=\hat{K}\left(1-\hat{\tau}\right)\left(1-\alpha\right)\left(1-\beta\right) - \hat{D} \\
\hat{D} &= (1 + \hat{r})  \hat{D} + \hat{G} - \hat{T} \\
\hat{T} &= \hat{\tau} \hat{Y} + \hat{\tau} \hat{r} \hat{D} .
\end{aligned}
$$ (eq:steadystates)

这意味着

$$
\begin{aligned}
\hat{K} &= \left[\left(1-\hat{\tau}\right)\left(1-\alpha\right)\left(1-\beta\right)\right]^{\frac{1}{1-\alpha}} \\
\hat{\tau} &= \frac{\hat{G} + \hat{r} \hat{D}}{\hat{Y} + \hat{r} \hat{D}}
\end{aligned}
$$

让我们举一个例子，其中

1. 初始没有政府债务，$D_t=0$，
2. 政府消费 $G_t$ 等于产出 $Y_t$ 的 $15\%$

我们的稳态值公式告诉我们

$$
\begin{aligned}
\hat{D} &= 0 \\
\hat{G} &= 0.15 \hat{Y} \\
\hat{\tau} &= 0.15 \\
\end{aligned}
$$



### 实现

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from numba import jit
from quantecon.optimize import brent_max
```
对于参数 $\alpha = 0.3$ 和 $\beta = 0.5$，让我们计算 $\hat{K}$：

```{code-cell} ipython3
# parameters
α = 0.3
β = 0.5

# steady states of τ and D
τ_hat = 0.15
D_hat = 0.

# solve for steady state of K
K_hat = ((1 - τ_hat) * (1 - α) * (1 - β)) ** (1 / (1 - α))
K_hat
```
知道 $\hat K$ 后,我们可以计算其他均衡对象。

让我们首先定义一些 Python 辅助函数。

```{code-cell} ipython3
@jit
def K_to_Y(K, α):

    return K ** α

@jit
def K_to_r(K, α):

    return α * K ** (α - 1)

@jit
def K_to_W(K, α):

    return (1 - α) * K ** α

@jit
def K_to_C(K, D, τ, r, α, β):

    # 当 δ=0 时老年人的最优消费
    A = K + D
    Co = A * (1 + r * (1 - τ))

    # 当 δ=0 时年轻人的最优消费
    W = K_to_W(K, α)
    Cy = β * W * (1 - τ)

    return Cy, Co
```
我们可以使用这些辅助函数来获得与稳态值 $\hat{K}$ 和 $\hat{r}$ 相关的稳态值 $\hat{Y}$、$\hat{r}$ 和 $\hat{W}$。

```{code-cell} ipython3
Y_hat, r_hat, W_hat = K_to_Y(K_hat, α), K_to_r(K_hat, α), K_to_W(K_hat, α)
Y_hat, r_hat, W_hat
```
由于稳态政府债务 $\hat{D}$ 为 $0$，所有税收都用于支付政府支出

```{code-cell} ipython3
G_hat = τ_hat * Y_hat
G_hat
```
我们使用最优消费计划来找到年轻人和老年人的稳态消费

```{code-cell} ipython3
Cy_hat, Co_hat = K_to_C(K_hat, D_hat, τ_hat, r_hat, α, β)
Cy_hat, Co_hat
```
让我们使用一个名为 `init_ss` 的数组来存储稳态下的数量和价格

```{code-cell} ipython3
init_ss = np.array([K_hat, Y_hat, Cy_hat, Co_hat,     # 数量
                    W_hat, r_hat,                     # 价格
                    τ_hat, D_hat, G_hat               # 政策
                    ])
```
### 转移

<!--
%<font color='red'>Zejin: I tried to edit the following part to describe the fiscal policy %experiment and the objects we are interested in computing. </font>
-->

我们已经计算了一个稳态，其中政府政策序列在时间上都是常数。

我们将使用这个稳态作为时间 $t=0$ 时另一个经济体的初始条件，在这个经济体中，政府政策序列随时间变化。

为了理解我们的计算，我们将 $t=0$ 视为发生巨大意外冲击的时间，表现为

  * 随时间变化的政府政策序列扰乱了原有的稳态
  * 新的政府政策序列最终在时间上是不变的，意味着在某个日期 $T >0$ 之后，每个序列都是时间不变的。
  * 以从时间 $t=0$ 开始的序列形式突然披露新的政府政策
我们假设在时间 $t=0$ 时，包括老年人在内的每个人都知道新的政府政策序列，并据此做出选择。

随着资本存量和其他总量随时间对财政政策变化的调整，经济将接近一个新的稳态。

我们可以通过在序列空间中使用不动点算法来找到从旧稳态到新稳态的过渡路径。

但在我们具有封闭形式解的特殊情况下，我们有一种更简单、更快速的方法可用。

这里我们定义一个 Python 类 `ClosedFormTrans`，用于计算响应特定财政政策变化的长度为 $T$ 的过渡路径。

我们选择足够大的 $T$，以便在 $T$ 期后我们已经非常接近新的稳态。

该类接受三个关键字参数：`τ_pol`、`D_pol` 和 `G_pol`。

它们分别是税率、政府债务水平和政府购买的序列。
在下面的每个政策实验中,我们将传递三个中的两个作为输入,以描述财政政策。

然后,我们将从政府预算约束中计算剩余的未确定的单个政策变量。

当我们模拟转移路径时,区分时间$t$的**状态变量**,如$K_t,Y_t,D_t,W_t,r_t$和包括$C_{yt},C_{ot},\tau_{t},G_t$的**控制变量**很有用。

```{code-cell} ipython3
class ClosedFormTrans:
    """
    This class simulates length T transitional path of a economy
    in response to a fiscal policy change given its initial steady
    state. The simulation is based on the closed form solution when
    the lump sum taxations are absent.

    """

    def __init__(self, α, β):

        self.α, self.β = α, β

    def simulate(self,
                T,           # length of transitional path to simulate
                init_ss,     # initial steady state
                τ_pol=None,  # sequence of tax rates
                D_pol=None,  # sequence of government debt levels
                G_pol=None): # sequence of government purchases

        α, β = self.α, self.β

        # unpack the steady state variables
        K_hat, Y_hat, Cy_hat, Co_hat = init_ss[:4]
        W_hat, r_hat = init_ss[4:6]
        τ_hat, D_hat, G_hat = init_ss[6:9]

        # initialize array containers
        # K, Y, Cy, Co
        quant_seq = np.empty((T+1, 4))

        # W, r
        price_seq = np.empty((T+1, 2))

        # τ, D, G
        policy_seq = np.empty((T+2, 3))

        # t=0, starting from steady state
        K0, Y0 = K_hat, Y_hat
        W0, r0 = W_hat, r_hat
        D0 = D_hat

        # fiscal policy
        if τ_pol is None:
            D1 = D_pol[1]
            G0 = G_pol[0]
            τ0 = (G0 + (1 + r0) * D0 - D1) / (Y0 + r0 * D0)
        elif D_pol is None:
            τ0 = τ_pol[0]
            G0 = G_pol[0]
            D1 = (1 + r0) * D0 + G0 - τ0 * (Y0 + r0 * D0)
        elif G_pol is None:
            D1 = D_pol[1]
            τ0 = τ_pol[0]
            G0 = τ0 * (Y0 + r0 * D0) + D1 - (1 + r0) * D0

        # optimal consumption plans
        Cy0, Co0 = K_to_C(K0, D0, τ0, r0, α, β)

        # t=0 economy
        quant_seq[0, :] = K0, Y0, Cy0, Co0
        price_seq[0, :] = W0, r0
        policy_seq[0, :] = τ0, D0, G0
        policy_seq[1, 1] = D1

        # starting from t=1 to T
        for t in range(1, T+1):

            # transition of K
            K_old, τ_old = quant_seq[t-1, 0], policy_seq[t-1, 0]
            D = policy_seq[t, 1]
            K = K_old ** α * (1 - τ_old) * (1 - α) * (1 - β) - D

            # output, capital return, wage
            Y, r, W = K_to_Y(K, α), K_to_r(K, α), K_to_W(K, α)

            # to satisfy the government budget constraint
            if τ_pol is None:
                D = D_pol[t]
                D_next = D_pol[t+1]
                G = G_pol[t]
                τ = (G + (1 + r) * D - D_next) / (Y + r * D)
            elif D_pol is None:
                τ = τ_pol[t]
                G = G_pol[t]
                D = policy_seq[t, 1]
                D_next = (1 + r) * D + G - τ * (Y + r * D)
            elif G_pol is None:
                D = D_pol[t]
                D_next = D_pol[t+1]
                τ = τ_pol[t]
                G = τ * (Y + r * D) + D_next - (1 + r) * D

            # optimal consumption plans
            Cy, Co = K_to_C(K, D, τ, r, α, β)

            # store time t economy aggregates
            quant_seq[t, :] = K, Y, Cy, Co
            price_seq[t, :] = W, r
            policy_seq[t, 0] = τ
            policy_seq[t+1, 1] = D_next
            policy_seq[t, 2] = G

        self.quant_seq = quant_seq
        self.price_seq = price_seq
        self.policy_seq = policy_seq

        return quant_seq, price_seq, policy_seq

    def plot(self):

        quant_seq = self.quant_seq
        price_seq = self.price_seq
        policy_seq = self.policy_seq

        fig, axs = plt.subplots(3, 3, figsize=(14, 10))

        # quantities
        for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
            ax = axs[i//3, i%3]
            ax.plot(range(T+1), quant_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')

        # prices
        for i, name in enumerate(['W', 'r']):
            ax = axs[(i+4)//3, (i+4)%3]
            ax.plot(range(T+1), price_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')

        # policies
        for i, name in enumerate(['τ', 'D', 'G']):
            ax = axs[(i+6)//3, (i+6)%3]
            ax.plot(range(T+1), policy_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')
```
我们可以为模型参数 $\{\alpha, \beta\}$ 创建一个实例 `closed`，并将其用于各种财政政策实验。

```{code-cell} ipython3
closed = ClosedFormTrans(α, β)
```
(exp-tax-cut)=
### 实验1:减税

为了说明`ClosedFormTrans`的强大功能,让我们首先尝试以下财政政策变化:

1. 在$t=0$时,政府通过发行政府债务$\bar{D}$,意外宣布一次性减税$\tau_0 =(1-\frac{1}{3}) \hat{\tau}$
2. 从$t=1$开始,政府将保持$D_t=\bar{D}$,并调整$\tau_{t}$以征收税款,用于支付政府消费和债务利息
3. 政府消费$G_t$将固定在$0.15 \hat{Y}$

以下方程完全刻画了源自初始稳态的均衡转移路径

$$
\begin{aligned}
K_{t+1} &= K_{t}^{\alpha}\left(1-\tau_{t}\right)\left(1-\alpha\right)\left(1-\beta\right) - \bar{D} \\
\tau_{0} &= (1-\frac{1}{3}) \hat{\tau} \\
\bar{D} &= \hat{G} - \tau_0\hat{Y} \\
\quad\tau_{t} & =\frac{\hat{G}+r_{t} \bar{D}}{\hat{Y}+r_{t} \bar{D}}
\end{aligned}
$$
我们可以模拟经济在 $20$ 个周期内的转换过程,之后经济将接近新的稳态。

第一步是准备描述财政政策的政策变量序列。

我们必须提前定义政府支出 $\{G_t\}_{t=0}^{T}$ 和债务水平 $\{D_t\}_{t=0}^{T+1}$ 的序列,然后将它们传递给求解器。

```{code-cell} ipython3
T = 20

# 减税
τ0 = τ_hat * (1 - 1/3)

# 政府购买序列
G_seq = τ_hat * Y_hat * np.ones(T+1)

# 政府债务序列
D_bar = G_hat - τ0 * Y_hat
D_seq = np.ones(T+2) * D_bar
D_seq[0] = D_hat
```
让我们使用 `closed` 的 `simulate` 方法来计算动态转换。

请注意，我们将 `τ_pol` 保留为 `None`，因为需要确定税率以满足政府预算约束。

```{code-cell} ipython3
quant_seq1, price_seq1, policy_seq1 = closed.simulate(T, init_ss,
                                                      D_pol=D_seq,
                                                      G_pol=G_seq)
closed.plot()
```
我们也可以尝试更低的减税率，例如 $0.2$。

```{code-cell} ipython3
# 更低的减税率
τ0 = 0.15 * (1 - 0.2)

# 相应的债务序列
D_bar = G_hat - τ0 * Y_hat
D_seq = np.ones(T+2) * D_bar
D_seq[0] = D_hat

quant_seq2, price_seq2, policy_seq2 = closed.simulate(T, init_ss,
                                                      D_pol=D_seq,
                                                      G_pol=G_seq)
```
```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# 数量
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq1[:T+1, i], label=name+', 1/3')
    ax.plot(range(T+1), quant_seq2[:T+1, i], label=name+', 0.2')
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# 价格
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq1[:T+1, i], label=name+', 1/3')
    ax.plot(range(T+1), price_seq2[:T+1, i], label=name+', 0.2')
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# 政策
for i, name in enumerate(['τ', 'D', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq1[:T+1, i], label=name+', 1/3')
    ax.plot(range(T+1), policy_seq2[:T+1, i], label=name+', 0.2')
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')
```
税率削减幅度较小的经济体在 $t=0$ 时具有相同的过渡模式,但扭曲程度较小,并且收敛到一个具有更高物质资本存量的新稳态。

(exp-expen-cut)=
### 实验2:政府资产积累

假设经济最初处于相同的稳态。

现在政府承诺 $\forall t \geq 0$ 将其在服务和商品上的支出减半。

政府的目标是维持相同的税率 $\tau_t=\hat{\tau}$,并随着时间的推移积累资产 $-D_t$。

为了进行这个实验,我们将 `τ_seq` 和 `G_seq` 作为输入,并让 `D_pol` 通过满足政府预算约束来确定路径。

```{code-cell} ipython3
# 政府支出减半
G_seq = τ_hat * 0.5 * Y_hat * np.ones(T+1)

# 目标税率
τ_seq = τ_hat * np.ones(T+1)

closed.simulate(T, init_ss, τ_pol=τ_seq, G_pol=G_seq);
closed.plot()
```
随着政府积累资产并将其用于生产，资本的租金率下降，私人投资下降。

因此，政府资产与用于生产的实物资本之比 $-\frac{D_t}{K_t}$ 将随着时间的推移而增加

```{code-cell} ipython3
plt.plot(range(T+1), -closed.policy_seq[:-1, 1] / closed.quant_seq[:, 0])
plt.xlabel('t')
plt.title('-D/K');
```
我们想知道这项政策实验如何影响个人。

从长远来看,未来的世代将在他们的一生中享受更高的消费,因为他们工作时将获得更高的劳动收入。

然而,在短期内,老年人遭受损失,因为他们的劳动收入增加不足以抵消他们的资本收入损失。

这种长期和短期效应的差异促使我们研究转型路径。

```{note}
尽管新的稳态下消费严格更高,但这是以牺牲更少的公共服务和商品为代价的。
```


### 实验3:暂时性支出削减

现在让我们研究一个情景,政府也将其支出削减一半并积累资产。

但现在让政府仅在 $t=0$ 时削减其支出。

从 $t \geq 1$ 开始,政府支出回到 $\hat{G}$,而 $\tau_t$ 调整以维持资产水平 $-D_t = -D_1$。

```{code-cell} ipython3
# sequence of government purchase
G_seq = τ_hat * Y_hat * np.ones(T+1)
G_seq[0] = 0

# sequence of government debt
D_bar = G_seq[0] - τ_hat * Y_hat
D_seq = D_bar * np.ones(T+2)
D_seq[0] = D_hat

closed.simulate(T, init_ss, D_pol=D_seq, G_pol=G_seq);
closed.plot()
```
经济迅速收敛到一个新的稳态,具有更高的物质资本存量、更低的利率、更高的工资率,以及更高的青年人和老年人消费。

即使政府支出 $G_t$ 从 $t \geq 1$ 开始回到最初的高水平,政府也可以在更低的税率下平衡预算,因为它从临时削减支出期间积累的资产中获得了额外的收入 $-r_t D_t$。

如 {ref}`exp-expen-cut` 中所述,转型初期的老年人受到这一政策冲击的影响。


## 计算策略

通过前面的计算,我们研究了由不同财政政策引发的动态转型。

在所有这些实验中,我们保持了没有一次性税收的假设,即 $\delta_{yt}=0, \delta_{ot}=0$。

在本节中,我们将研究存在一次性税收时的转型动态。
政府将使用一次性税收和转移支付来重新分配跨代资源。

包括一次性税收会扰乱封闭形式解,因为它们使最优消费和储蓄计划依赖于未来的价格和税率。

因此,我们通过寻找从序列到序列的映射的不动点来计算均衡过渡路径。

  * 该不动点确定了一个均衡

为了为我们寻求其不动点的映射的进入设置舞台,我们回到 {ref}`sec-equilibrium` 一节中介绍的概念。


**定义:** 给定参数 $\{\alpha$, $\beta\}$,一个竞争均衡包括

* 最优消费序列 $\{C_{yt}, C_{ot}\}$
* 价格序列 $\{W_t, r_t\}$
* 资本存量和产出序列 $\{K_t, Y_t\}$
* 税率、政府资产(债务)、政府购买序列 $\{\tau_t, D_t, G_t\, \delta_{yt}, \delta_{ot}\}$

具有以下性质
* 给定价格体系和政府财政政策，消费计划是最优的
* 对于所有 $t$，政府预算约束都得到满足

可以通过"猜测和验证"某些内生序列来计算均衡转移路径。

在我们的 {ref}`exp-tax-cut` 示例中，序列 $\{D_t\}_{t=0}^{T}$ 和 $\{G_t\}_{t=0}^{T}$ 是外生的。

此外，我们假设一次性税收 $\{\delta_{yt}, \delta_{ot}\}_{t=0}^{T}$ 是给定的，并且模型内的每个人都知道。

我们可以按照以下步骤求解其他均衡序列

1. 猜测价格 $\{W_t, r_t\}_{t=0}^{T}$ 和税率 $\{\tau_t\}_{t=0}^{T}$
2. 求解最优消费和储蓄计划 $\{C_{yt}, C_{ot}\}_{t=0}^{T}$，将未来价格和税率的猜测视为真实值
3. 求解资本存量的转移 $\{K_t\}_{t=0}^{T}$
4. 用均衡条件隐含的值更新价格和税率的猜测
5. 迭代直至收敛
让我们实现这个"猜测和验证"的方法

我们首先定义柯布-道格拉斯效用函数

```{code-cell} ipython3
@jit
def U(Cy, Co, β):

    return (Cy ** β) * (Co ** (1-β))
```
我们使用 `Cy_val` 来计算给定跨期预算约束下任意消费计划 $C_y$ 的终身价值。

请注意，它需要知道未来的价格 $r_{t+1}$ 和税率 $\tau_{t+1}$。

```{code-cell} ipython3
@jit
def Cy_val(Cy, W, r_next, τ, τ_next, δy, δo_next, β):

    # Co 由预算约束给出
    Co = (W * (1 - τ) - δy - Cy) * (1 + r_next * (1 - τ_next)) - δo_next

    return U(Cy, Co, β)
```
最优消费计划 $C_y^*$ 可以通过最大化 `Cy_val` 来找到。

下面是一个计算稳态下最优消费 $C_y^*=\hat{C}_y$ 的例子，其中 $\delta_{yt}=\delta_{ot}=0$，就像我们之前研究的那样

```{code-cell} ipython3
W, r_next, τ, τ_next = W_hat, r_hat, τ_hat, τ_hat
δy, δo_next = 0, 0

Cy_opt, U_opt, _ = brent_max(Cy_val,            # maximand
                             1e-6,              # lower bound
                             W*(1-τ)-δy-1e-6,   # upper bound
                             args=(W, r_next, τ, τ_next, δy, δo_next, β))

Cy_opt, U_opt
```
让我们定义一个Python类`AK2`，它使用不动点算法计算转移路径。

它可以处理非零一次性税收

```{code-cell} ipython3
class AK2():
    """
    该类模拟一个经济体在给定初始稳态下，对财政政策变化的长度为T的转移路径。
    转移路径通过采用不动点算法来满足均衡条件而得出。

    """

    def __init__(self, α, β):

        self.α, self.β = α, β

    def simulate(self,
                T,           # 模拟的转移路径长度
                init_ss,     # 初始稳态
                δy_seq,      # 年轻人的一次性税收序列
                δo_seq,      # 老年人的一次性税收序列
                τ_pol=None,  # 税率序列
                D_pol=None,  # 政府债务水平序列
                G_pol=None,  # 政府购买序列
                verbose=False,
                max_iter=500,
                tol=1e-5):

        α, β = self.α, self.β

        # 解包稳态变量
        K_hat, Y_hat, Cy_hat, Co_hat = init_ss[:4]
        W_hat, r_hat = init_ss[4:6]
        τ_hat, D_hat, G_hat = init_ss[6:9]

        # K, Y, Cy, Co
        quant_seq = np.empty((T+2, 4))

        # W, r
        price_seq = np.empty((T+2, 2))

        # τ, D, G
        policy_seq = np.empty((T+2, 3))
        policy_seq[:, 1] = D_pol
        policy_seq[:, 2] = G_pol

        # 价格的初始猜测
        price_seq[:, 0] = np.ones(T+2) * W_hat
        price_seq[:, 1] = np.ones(T+2) * r_hat

        # 政策的初始猜测
        policy_seq[:, 0] = np.ones(T+2) * τ_hat

        # t=0, 从稳态开始
        quant_seq[0, :2] = K_hat, Y_hat

        if verbose:
            # 准备绘制迭代直到收敛
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # 检查收敛的容器
        price_seq_old = np.empty_like(price_seq)
        policy_seq_old = np.empty_like(policy_seq)

        # 开始迭代
        i_iter = 0
        while True:

            if verbose:
                # 在第i次迭代时绘制当前价格
                for i, name in enumerate(['W', 'r']):
                    axs[i].plot(range(T+1), price_seq[:T+1, i])
                    axs[i].set_title(name)
                    axs[i].set_xlabel('t')
                axs[2].plot(range(T+1), policy_seq[:T+1, 0],
                            label=f'第{i_iter}次迭代')
                axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axs[2].set_title('τ')
                axs[2].set_xlabel('t')

            # 存储上一次迭代的旧价格
            price_seq_old[:] = price_seq
            policy_seq_old[:] = policy_seq

            # 开始更新数量和价格
            for t in range(T+1):
                K, Y = quant_seq[t, :2]
                W, r = price_seq[t, :]
                r_next = price_seq[t+1, 1]
                τ, D, G = policy_seq[t, :]
                τ_next, D_next, G_next = policy_seq[t+1, :]
                δy, δo = δy_seq[t], δo_seq[t]
                δy_next, δo_next = δy_seq[t+1], δo_seq[t+1]

                # 老年人的消费
                Co = (1 + r * (1 - τ)) * (K + D) - δo

                # 年轻人的最优消费
                out = brent_max(Cy_val, 1e-6, W*(1-τ)-δy-1e-6,
                                args=(W, r_next, τ, τ_next,
                                      δy, δo_next, β))
                Cy = out[0]

                quant_seq[t, 2:] = Cy, Co
                τ_num = ((1 + r) * D + G - D_next - δy - δo)
                τ_denom = (Y + r * D)
                policy_seq[t, 0] = τ_num / τ_denom

                # 年轻人的储蓄
                A_next = W * (1 - τ) - δy - Cy

                # K的转移
                K_next = A_next - D_next
                Y_next = K_to_Y(K_next, α)
                W_next, r_next = K_to_W(K_next, α), K_to_r(K_next, α)

                quant_seq[t+1, :2] = K_next, Y_next
                price_seq[t+1, :] = W_next, r_next

            i_iter += 1

            if (np.max(np.abs(price_seq_old - price_seq)) < tol) & \
               (np.max(np.abs(policy_seq_old - policy_seq)) < tol):
                if verbose:
                    print(f"使用{i_iter}次迭代收敛")
                break

            if i_iter > max_iter:
                if verbose:
                    print(f"使用{i_iter}次迭代未能收敛")
                break
        
        self.quant_seq = quant_seq
        self.price_seq = price_seq
        self.policy_seq = policy_seq

        return quant_seq, price_seq, policy_seq

    def plot(self):

        quant_seq = self.quant_seq
        price_seq = self.price_seq
        policy_seq = self.policy_seq

        fig, axs = plt.subplots(3, 3, figsize=(14, 10))

        # 数量
        for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
            ax = axs[i//3, i%3]
            ax.plot(range(T+1), quant_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')

        # 价格
        for i, name in enumerate(['W', 'r']):
            ax = axs[(i+4)//3, (i+4)%3]
            ax.plot(range(T+1), price_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')

        # 政策
        for i, name in enumerate(['τ', 'D', 'G']):
            ax = axs[(i+6)//3, (i+6)%3]
            ax.plot(range(T+1), policy_seq[:T+1, i], label=name)
            ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
            ax.legend()
            ax.set_xlabel('t')
```
我们可以用模型参数 $\{\alpha, \beta\}$ 初始化一个 `AK2` 类的实例,然后用它来进行财政政策实验。

```{code-cell} ipython3
ak2 = AK2(α, β)
```
我们首先检验"猜测和验证"方法得到的数值结果与我们通过封闭解得到的结果一致，当一次性税收被消除时

```{code-cell} ipython3
δy_seq = np.ones(T+2) * 0.
δo_seq = np.ones(T+2) * 0.

D_pol = np.zeros(T+2)
G_pol = np.ones(T+2) * G_hat

# 减税
τ0 = τ_hat * (1 - 1/3)
D1 = D_hat * (1 + r_hat * (1 - τ0)) + G_hat - τ0 * Y_hat - δy_seq[0] - δo_seq[0]
D_pol[0] = D_hat
D_pol[1:] = D1
```
```{code-cell} ipython3
quant_seq3, price_seq3, policy_seq3 = ak2.simulate(T, init_ss,
                                                   δy_seq, δo_seq,
                                                   D_pol=D_pol, G_pol=G_pol,
                                                   verbose=True)
```
```{code-cell} ipython3
ak2.plot()
```
接下来,我们激活一次性税收。

让我们改变我们的 {ref}`exp-tax-cut` 财政政策实验,假设政府同时增加了对年轻人和老年人的一次性税收 $\delta_{yt}=\delta_{ot}=0.005, t\geq0$。

```{code-cell} ipython3
δy_seq = np.ones(T+2) * 0.005
δo_seq = np.ones(T+2) * 0.005

D1 = D_hat * (1 + r_hat * (1 - τ0)) + G_hat - τ0 * Y_hat - δy_seq[0] - δo_seq[0]
D_pol[1:] = D1

quant_seq4, price_seq4, policy_seq4 = ak2.simulate(T, init_ss,
                                                   δy_seq, δo_seq,
                                                   D_pol=D_pol, G_pol=G_pol)
```
请注意，"挤出效应"已经得到缓解。

```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# quantities
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq3[:T+1, i], label=name+', $\delta$s=0')
    ax.plot(range(T+1), quant_seq4[:T+1, i], label=name+', $\delta$s=0.005')
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# prices
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq3[:T+1, i], label=name+', $\delta$s=0')
    ax.plot(range(T+1), price_seq4[:T+1, i], label=name+', $\delta$s=0.005')
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# policies
for i, name in enumerate(['τ', 'D', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq3[:T+1, i], label=name+', $\delta$s=0')
    ax.plot(range(T+1), policy_seq4[:T+1, i], label=name+', $\delta$s=0.005')
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')
```
与 {ref}`exp-tax-cut` 相比，政府提高一次性税收以支付不断增加的债务利息支出，与提高资本所得税税率相比，这种做法的扭曲性更小。


### 实验4：无基金社会保障体系

在这个实验中，老年人和年轻人的一次性税收数额相等，但符号相反。

负的一次性税收是一种补贴。

因此，在这个实验中，我们对年轻人征税，对老年人进行补贴。

我们从与之前几个实验中假设的相同初始稳态开始经济。

政府从 $t=0$ 开始设置一次性税收 $\delta_{y,t}=-\delta_{o,t}=10\% \hat{C}_{y}$。

它将债务水平和支出保持在稳态水平 $\hat{D}$ 和 $\hat{G}$。

实际上，这个实验相当于启动一个无基金社会保障体系。

我们可以使用代码来计算启动这个系统所引发的转变。

让我们将结果与 {ref}`exp-tax-cut` 进行比较。
```{code-cell} ipython3
δy_seq = np.ones(T+2) * Cy_hat * 0.1
δo_seq = np.ones(T+2) * -Cy_hat * 0.1

D_pol[:] = D_hat

quant_seq5, price_seq5, policy_seq5 = ak2.simulate(T, init_ss,
                                                   δy_seq, δo_seq,
                                                   D_pol=D_pol, G_pol=G_pol)
```
```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# 数量
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq3[:T+1, i], label=name+', 减税')
    ax.plot(range(T+1), quant_seq5[:T+1, i], label=name+', 转移支付')
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# 价格
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq3[:T+1, i], label=name+', 减税')
    ax.plot(range(T+1), price_seq5[:T+1, i], label=name+', 转移支付')
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')

# 政策
for i, name in enumerate(['τ', 'D', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq3[:T+1, i], label=name+', 减税')
    ax.plot(range(T+1), policy_seq5[:T+1, i], label=name+', 转移支付')
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.legend()
    ax.set_xlabel('t')
```
一个最初的老年人在社会保障制度启动时会获得特别的好处,因为他能够获得转移支付,但却不需要为此支付任何费用。

但是从长远来看,年轻人和老年人的消费率都会下降,因为社会保障制度降低了储蓄的激励。

这会降低实物资本存量,从而降低产出。

政府必须提高税率以支付其支出。

资本收入的较高税率会进一步扭曲储蓄的激励。
