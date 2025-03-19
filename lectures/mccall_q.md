---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 求职搜索 VII：McCall工人的Q学习

## 概述

本讲解介绍一种强大的机器学习技术——Q学习。

{cite}`Sutton_2018`介绍了Q学习和其他各种统计学习程序。

Q学习算法结合了以下思想：

* 动态规划

* 最小二乘法的递归版本，即[时间差分学习](https://en.wikipedia.org/wiki/Temporal_difference_learning)。

本讲将Q学习算法应用于McCall工人所面临的情况。

本讲还考虑了McCall工人可以选择辞去当前工作的情况。

相对于我们在 {doc}`quantecon 讲座 <mccall_model>` 中学习的 McCall 工人模型的动态规划方法，Q-学习算法让工人对以下方面的了解更少：

* 生成工资序列的随机过程
* 描述接受或拒绝工作后果的奖励函数

Q-学习算法调用统计学习模型来学习这些内容。

统计学习通常可以归结为某种形式的最小二乘法，在这里也是如此。

每当我们提到**统计学习**时，我们都必须说明正在学习的对象是什么。

对于 Q-学习来说，要学习的对象并不是动态规划所关注的**价值函数**。

但它与价值函数密切相关。

在本讲座研究的有限动作、有限状态的情况下，要统计学习的对象是一个 **Q-表**，它是针对有限集合的 **Q-函数**的一个实例。

有时 Q-函数或 Q-表也被称为质量函数或质量表。

Q-表的行和列对应着智能体可能遇到的状态，以及在每个状态下可以采取的可能行动。

一个类似贝尔曼方程的等式在算法中起着重要作用。

它与我们在{doc}`这个 quantecon 讲座 <mccall_model>`中看到的 McCall 模型的贝尔曼方程不同。

在本讲座中，我们将学习一些关于：

* 与任何马尔可夫决策问题相关的**Q-函数**或**质量函数**，其最优值函数满足贝尔曼方程

* **时序差分学习**，Q-学习算法的一个关键组成部分

像往常一样，让我们先导入一些 Python 模块。

```{code-cell} ipython3
:tags: [hide-output]
!pip install quantecon
```

```{code-cell} ipython3
import numpy as np

from numba import jit, float64, int64
from numba.experimental import jitclass
from quantecon.distributions import BetaBinomial

import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']


np.random.seed(123)
```

## McCall 模型回顾

我们首先回顾在{doc}`这个 quantecon 讲座 <mccall_model>`中描述的 McCall 模型。

我们将计算一个最优值函数和实现该值的政策。

我们最终会将这个最优政策与 Q-learning McCall 工人所学到的进行比较。

McCall 模型的特征由参数 $\beta,c$ 和已知的工资分布 $F$ 来描述。

McCall 工人想要最大化预期的终身收入折现总和

$$
\mathbb{E} \sum_{t=0}^{\infty} \beta^t y_t
$$

工人的收入 $y_t$ 在就业时等于他的工资 $w$，在失业时等于失业补助 $c$。

对于刚收到工资offer $w$ 并正在决定是接受还是拒绝的 McCall 工人来说，最优值 $V\left(w\right)$ 满足贝尔曼方程

$$
V\left(w\right)=\max_{\text{accept, reject}}\;\left\{ \frac{w}{1-\beta},c+\beta\int V\left(w'\right)dF\left(w'\right)\right\}
$$ (eq_mccallbellman)

为了与Q-learning的结果进行比较基准，我们首先近似最优值函数。

在有限离散状态空间中，可能的状态由$\{1,2,...,n\}$索引，我们对值函数$v\in\mathbb{R}^{n}$做一个初始猜测，然后对贝尔曼方程进行迭代：

$$
v^{\prime}(i)=\max \left\{\frac{w(i)}{1-\beta}, c+\beta \sum_{1 \leq j \leq n} v(j) q(j)\right\} \quad \text { for } i=1, \ldots, n
$$

让我们使用{doc}`这个quantecon讲座 <mccall_model>`中的Python代码。

我们使用一个名为`VFI`的Python方法，通过值函数迭代来计算最优值函数。

我们构造一个假设的工资分布，并用以下Python代码绘制：

```{code-cell} ipython3
n, a, b = 10, 200, 100                        # default parameters
q_default = BetaBinomial(n, a, b).pdf()       # default choice of q

w_min, w_max = 10, 60
w_default = np.linspace(w_min, w_max, n+1)

# plot distribution of wage offer
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(w_default, q_default, '-o', label='$q(w(i))$')
ax.set_xlabel('wages')
ax.set_ylabel('probabilities')

plt.show()
```

接下来我们将通过对贝尔曼方程进行迭代收敛来计算工人的最优价值函数。

然后我们将绘制贝尔曼算子的各种迭代结果。

```{code-cell} ipython3
mccall_data = [
    ('c', float64),      # 失业补偿
    ('β', float64),      # 贴现因子
    ('w', float64[:]),   # 工资值数组，w[i] = 状态i下的工资
    ('q', float64[:]),    # 概率数组
]


@jitclass(mccall_data)
class McCallModel:

    def __init__(self, c=25, β=0.99, w=w_default, q=q_default):

        self.c, self.β = c, β
        self.w, self.q = w, q

    def state_action_values(self, i, v):
        """
        状态-行动对的值。
        """
        # 简化名称
        c, β, w, q = self.c, self.β, self.w, self.q
        # 评估每个状态-行动对的值
        # 考虑行动 = 接受或拒绝当前offer
        accept = w[i] / (1 - β)
        reject = c + β * np.sum(v * q)

        return np.array([accept, reject])

    def VFI(self, eps=1e-5, max_iter=500):
        """
        找到最优价值函数。
        """

        n = len(self.w)
        v = self.w / (1 - self.β)
        v_next = np.empty_like(v)
        flag=0

        for i in range(max_iter):
            for j in range(n):
                v_next[j] = np.max(self.state_action_values(j, v))

            if np.max(np.abs(v_next - v))<=eps:
                flag=1
                break
            v[:] = v_next

        return v, flag

def plot_value_function_seq(mcm, ax, num_plots=8):
    """
    绘制一系列价值函数。

        * mcm 是 McCallModel 的一个实例
        * ax 是实现了plot方法的轴对象

    """

    n = len(mcm.w)
    v = mcm.w / (1 - mcm.β)
    v_next = np.empty_like(v)
    for i in range(num_plots):
        ax.plot(mcm.w, v, '-', alpha=0.4, label=f"iterate {i}")
        # 更新猜测值
        for i in range(n):
            v_next[i] = np.max(mcm.state_action_values(i, v))
        v[:] = v_next  # 将内容复制到v中

    ax.legend(loc='lower right')
```

```{code-cell} ipython3
mcm = McCallModel()
valfunc_VFI, flag = mcm.VFI()

fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlabel('工资')
ax.set_ylabel('价值')
plot_value_function_seq(mcm, ax)
plt.show()
```

接下来我们将打印出迭代序列的极限值。

这是通过值函数迭代得到的McCall工人价值函数的近似值。

在我们完成Q学习之后，我们将使用这个值函数作为基准。

```{code-cell} ipython3
print(valfunc_VFI)
```

## 隐含质量函数 $Q$

**质量函数** $Q$ 将状态-动作对映射为最优值。

它们与最优值函数紧密相连。

但值函数仅是状态的函数，而不包含动作。

对于每个给定状态，质量函数给出从该状态开始可以达到的最优值列表，列表的每个组成部分表示可以采取的一种可能动作。

对于我们的McCall工人模型，假设有有限的可能工资集合：

* 状态空间 $\mathcal{W}=\{w_1,w_2,...,w_n\}$ 由整数 $1,2,...,n$ 索引

* 动作空间为 $\mathcal{A}=\{\text{accept}, \text{reject}\}$

令 $a \in \mathcal{A}$ 为两个可能动作之一，即接受或拒绝。

对于我们的McCall工人，最优Q函数 $Q(w,a)$ 等于一个此前失业的工人在手头有offer $w$ 时，如果他采取动作 $a$ 所能获得的最大价值。

$Q(w,a)$ 的这个定义假设工人在随后的时期会采取最优行动。

我们的 McCall 工人的最优 Q-函数满足

$$
\begin{aligned}
Q\left(w,\text{accept}\right) & =\frac{w}{1-\beta} \\
Q\left(w,\text{reject}\right) & =c+\beta\int\max_{\text{accept, reject}}\left\{ \frac{w'}{1-\beta},Q\left(w',\text{reject}\right)\right\} dF\left(w'\right)
\end{aligned}
$$ (eq:impliedq)

注意，系统{eq}`eq:impliedq`的第一个方程假设在代理人接受了一个报价后，他将来不会拒绝同样的报价。

这些方程与我们在{doc}`这个 quantecon 讲座 <mccall_model>`中研究的工人最优值函数的贝尔曼方程是一致的。

显然，在那个讲座中描述的最优值函数 $V(w)$ 与我们的 Q-函数有如下关系：

$$
V(w) = \max_{\textrm{accept},\textrm{reject}} \left\{ Q(w, \text{accept} \right), Q\left(w,\text{reject} \right)\}
$$

如果我们观察系统{eq}`eq:impliedq`中的第二个方程，我们注意到由于工资过程在时间上是独立同分布的，$Q\left(w,\text{reject}\right)$，方程右侧与当前状态$w$无关。

因此我们可以将其表示为一个标量

$$ Q_r := Q\left(w,\text{reject}\right) \quad \forall \, w\in\mathcal{W}.
$$

这一事实为我们提供了一种替代方案，而且事实证明，在这种情况下，这是一种更快的方法来计算McCall工人模型的最优值函数和相关的最优策略。

与我们上面使用的值函数迭代不同，我们可以对系统{eq}`eq:impliedq`中第二个方程的一个版本进行迭代直至收敛，该方程将$Q_r$的估计值映射为改进的估计值$Q_r'$：

$$
Q_{r}^\prime=c+\beta\int\max_{\text{}}\left\{ \frac{w'}{1-\beta},Q_{r}\right\} dF\left(w'\right)
$$

在$Q_r$序列收敛后，我们可以从以下公式恢复McCall工人模型的最优值函数$V(w)$：

$$
V\left(w\right)=\max\left\{ \frac{w}{1-\beta},Q_{r}\right\}
$$

+++

## 从概率到样本

我们之前提到，McCall工人模型的最优Q函数满足以下贝尔曼方程：

$$
\begin{aligned}
         w  & + \beta \max_{\textrm{accept, reject}} \left\{ Q (w, \textrm{accept}), Q(w, \textrm{reject}) \right\} - Q (w, \textrm{accept})   = 0  \cr
         c  & +\beta\int\max_{\text{accept, reject}}\left\{ Q(w', \textrm{accept}),Q\left(w',\text{reject}\right)\right\} dF\left(w'\right) - Q\left(w,\text{reject}\right)  = 0  \cr
\end{aligned}
$$ (eq:probtosample1)

注意第二行中对$F(w')$的积分。

删除积分符号为我们开始思考Q-learning提供了一个不严谨但有启发性的思路。

因此，构建一个保留{eq}`eq:probtosample1`第一个方程的差分方程系统。

但是第二个方程通过去除对$F (w')$的积分来替代：

$$
\begin{aligned}
         w  & + \beta \max_{\textrm{accept, reject}} \left\{ Q (w, \textrm{accept}), Q(w, \textrm{reject}) \right\} - Q (w, \textrm{accept})   = 0  \cr
         c  & +\beta \max_{\text{accept, reject}}\left\{ Q(w', \textrm{accept}),Q\left(w',\text{reject}\right)\right\}  - Q\left(w,\text{reject}\right)  \approx 0  \cr
\end{aligned}
$$(eq:probtosample2)

第二个方程不可能对我们状态空间的笛卡尔积中的所有$w, w'$对都成立。

但是，也许我们可以借助大数定律，希望它能对一个长时间序列中抽取的$w_t, w_{t+1}$对**平均**成立，这里我们把$w_t$看作$w$，把$w_{t+1}$看作$w'$。

Q-learning的基本思想是从$F$中抽取一个长样本序列（虽然我们假设工人不知道$F$，但我们是知道的）并对递归式进行迭代

将日期 $t$ 时的 Q 函数估计值 $\hat Q_t$ 映射到日期 $t+1$ 时的改进估计值 $\hat Q_{t+1}$。

为了建立这样一个算法，我们首先定义一些误差或"差异"

$$
\begin{aligned}
         w  & + \beta \max_{\textrm{accept, reject}} \left\{ \hat Q_t (w_t, \textrm{accept}), \hat Q_t(w_t, \textrm{reject}) \right\} - \hat Q_t(w_t, \textrm{accept})   = \textrm{diff}_{\textrm{accept},t}  \cr
         c  & +\beta \max_{\text{accept, reject}}\left\{ \hat Q_t(w_{t+1}, \textrm{accept}),\hat Q_t\left(w_{t+1},\text{reject}\right)\right\}  - \hat Q_t\left(w_t,\text{reject}\right)  = \textrm{diff}_{\textrm{reject},t}  \cr
\end{aligned}
$$ (eq:old105)

自适应学习方案将是以下形式

$$
\hat Q_{t+1} = \hat Q_t + \alpha \ \textrm{diff}_t
$$ (eq:old106)

其中 $\alpha \in (0,1)$ 是一个小的**增益**参数，用于控制学习速率，而 $\hat Q_t$ 和 $\textrm{diff}_t$ 是对应的 $2 \times 1$ 向量

对应方程组 {eq}`eq:old105` 中的对象。

这个非正式的论述将我们引向 Q-学习的门槛。

## Q-学习

让我们首先精确描述一个 Q-学习算法。

然后我们将实现它。

该算法通过使用蒙特卡洛方法来更新 Q-函数的估计值。

我们从 Q-函数的初始猜测开始。

在本讲中研究的例子里，我们有一个有限的动作空间和有限的状态空间。

这意味着我们可以将 Q-函数表示为矩阵或 Q-表，$\widetilde{Q}(w,a)$。

Q-学习通过更新 Q-函数来进行，决策者在模拟生成的工资序列路径上获得经验。

在学习过程中，我们的 McCall 工人采取行动并体验这些行动带来的奖励。

他同时学习关于环境（在这种情况下是工资分布）和奖励函数的知识。

在这种情况下，失业补偿 $c$ 和工资的现值。

更新算法基于对如下递归的略微修改（稍后将描述）：

$$
\widetilde{Q}^{new}\left(w,a\right)=\widetilde{Q}^{old}\left(w,a\right)+\alpha \widetilde{TD}\left(w,a\right)
$$ (eq:old3)

其中

$$
\begin{aligned}
\widetilde{TD}\left(w,\text{accept}\right) & = \left[ w+\beta\max_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w,a'\right) \right]-\widetilde{Q}^{old}\left(w,\text{accept}\right) \\
\widetilde{TD}\left(w,\text{reject}\right) & = \left[ c+\beta\max_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w',a'\right) \right]-\widetilde{Q}^{old}\left(w,\text{reject}\right),\;w'\sim F
\end{aligned}
$$ (eq:old4)

对于 $a = \left\{\textrm{accept,reject} \right\}$，项 $\widetilde{TD}(w,a)$ 是驱动更新的**时间差分误差**。

因此，这个系统是我们在方程 {eq}`eq:old106` 中非正式描述的自适应系统的一个版本。

方程组{eq}`eq:old4`尚未捕捉到算法的一个方面，即我们通过偶尔随机替换的**实验性**尝试

$$
\textrm{argmax}_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w,a'\right)
$$

替换为

$$
\textrm{argmin}_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w,a'\right)
$$

并且
偶尔替换

$$
\textrm{argmax}_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w',a'\right)
$$

替换为

$$
\textrm{argmin}_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w',a'\right)
$$

在以下McCall工人Q-学习的伪代码的第3步中，我们以概率$\epsilon$激活这种实验：

1. 设置一个任意的初始Q表。

2. 从$F$中抽取初始工资报价$w$。

3. 从Q表的相应行中，使用以下$\epsilon$-贪婪算法选择行动：

    - 以概率$1-\epsilon$选择使价值最大化的行动，并且

- 以概率 $\epsilon$ 选择替代行动。

4. 更新与所选行动相关的状态，并根据{eq}`eq:old4`计算 $\widetilde{TD}$，然后根据{eq}`eq:old3`更新 $\widetilde{Q}$。

5. 如果需要则抽取新的状态 $w'$，否则采用现有工资，并再次根据{eq}`eq:old3`更新Q表。

6. 当新旧Q表足够接近时停止，即对于给定的 $\delta$，满足 $\lVert\tilde{Q}^{new}-\tilde{Q}^{old}\rVert_{\infty}\leq\delta$，或者当工人连续接受 $T$ 期（对于预先规定的 $T$）时停止。

7. 带着更新后的Q表返回步骤2。

重复此程序 $N$ 次回合或直到更新的Q表收敛。

我们将步骤2到7的一次完整过程称为时序差分学习的一个"回合"或"轮次"。

在我们的情境中，每个回合都始于代理人抽取一个初始工资报价，即一个新状态。

智能体根据预设的Q表采取行动，获得奖励，然后进入由本期行动所暗示的新状态。

Q表通过时序差分学习进行更新。

我们重复这个过程直到Q表收敛或达到一个回合的最大长度。

多个回合使智能体能够重新开始，并访问那些从前一个回合的终止状态较难访问到的状态。

例如，一个基于其Q表接受了工资报价的智能体将较少可能从工资分布的其他部分获得新的报价。

通过使用$\epsilon$-贪婪方法并增加回合数，Q学习算法在探索和利用之间取得平衡。

**注意：** 注意在{eq}`eq:old3`中定义的与最优Q表相关的$\widetilde{TD}$自动满足对所有状态-动作对$\widetilde{TD}=0$。我们的Q-learning算法的极限是否收敛到最优Q表，取决于算法是否足够频繁地访问所有状态-动作对。

我们在Python类中实现这个伪代码。

为了简单和方便，我们让`s`表示介于$0$和$n=50$之间的状态索引，且$w_s=w[s]$。

Q表的第一列表示拒绝工资的相关值，第二列表示接受工资的相关值。

我们使用`numba`编译来加速计算。

```{code-cell} ipython3
params=[
    ('c', float64),            # 失业补偿
    ('β', float64),            # 折现因子
    ('w', float64[:]),         # 工资值数组，w[i] = 状态i的工资
    ('q', float64[:]),         # 概率数组
    ('eps', float64),          # epsilon贪婪算法参数
    ('δ', float64),            # Q表阈值
    ('lr', float64),           # 学习率α
    ('T', int64),              # 接受的最大期数
    ('quit_allowed', int64)    # 接受工资后是否允许辞职
]

@jitclass(params)
class Qlearning_McCall:
    def __init__(self, c=25, β=0.99, w=w_default, q=q_default, eps=0.1,
                 δ=1e-5, lr=0.5, T=10000, quit_allowed=0):

        self.c, self.β = c, β
        self.w, self.q = w, q
        self.eps, self.δ, self.lr, self.T = eps, δ, lr, T
        self.quit_allowed = quit_allowed


    def draw_offer_index(self):
        """
        从工资分布中抽取状态索引。
        """

        q = self.q
        return np.searchsorted(np.cumsum(q), np.random.random(), side="right")

    def temp_diff(self, qtable, state, accept):
        """
        计算与状态和动作相关的TD。
        """

        c, β, w = self.c, self.β, self.w

        if accept==0:
            state_next = self.draw_offer_index()
            TD = c + β*np.max(qtable[state_next, :]) - qtable[state, accept]
        else:
            state_next = state
            if self.quit_allowed == 0:
                TD = w[state_next] + β*np.max(qtable[state_next, :]) - qtable[state, accept]
            else:
                TD = w[state_next] + β*qtable[state_next, 1] - qtable[state, accept]

        return TD, state_next

    def run_one_epoch(self, qtable, max_times=20000):
        """
        运行一个"轮次"。
        """

        c, β, w = self.c, self.β, self.w
        eps, δ, lr, T = self.eps, self.δ, self.lr, self.T

        s0 = self.draw_offer_index()
        s = s0
        accept_count = 0

        for t in range(max_times):

            # 选择动作
            accept = np.argmax(qtable[s, :])
            if np.random.random()<=eps:
                accept = 1 - accept

            if accept == 1:
                accept_count += 1
            else:
                accept_count = 0

            TD, s_next = self.temp_diff(qtable, s, accept)

            # 更新qtable
            qtable_new = qtable.copy()
            qtable_new[s, accept] = qtable[s, accept] + lr*TD

            if np.max(np.abs(qtable_new-qtable))<=δ:
                break

            if accept_count == T:
                break

            s, qtable = s_next, qtable_new

        return qtable_new

@jit
def run_epochs(N, qlmc, qtable):
    """
    运行N次轮次，每次使用上一次迭代的qtable。
    """

    for n in range(N):
        if n%(N/10)==0:
            print(f"进度：轮次 = {n}")
        new_qtable = qlmc.run_one_epoch(qtable)
        qtable = new_qtable

    return qtable

def valfunc_from_qtable(qtable):
    return np.max(qtable, axis=1)

def compute_error(valfunc, valfunc_VFI):
    return np.mean(np.abs(valfunc-valfunc_VFI))
```

```{code-cell} ipython3
# 创建一个 Qlearning_McCall 实例
qlmc = Qlearning_McCall()

# 运行
qtable0 = np.zeros((len(w_default), 2))
qtable = run_epochs(20000, qlmc, qtable0)
```

```{code-cell} ipython3
print(qtable)
```

```{code-cell} ipython3
# 检查价值函数
valfunc_qlr = valfunc_from_qtable(qtable)

print(valfunc_qlr)
```

```{code-cell} ipython3
# 绘图
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(w_default, valfunc_VFI, '-o', label='VFI')
ax.plot(w_default, valfunc_qlr, '-o', label='QL')
ax.set_xlabel('工资')
ax.set_ylabel('最优值')
ax.legend()

plt.show()
```

现在，让我们计算一个更大状态空间的情况：$n=30$（而不是$n=10$）。

```{code-cell} ipython3
n, a, b = 30, 200, 100                        # 默认参数
q_new = BetaBinomial(n, a, b).pdf()           # 默认的q选择

w_min, w_max = 10, 60
w_new = np.linspace(w_min, w_max, n+1)


# 绘制工资报价分布
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(w_new, q_new, '-o', label='$q(w(i))$')
ax.set_xlabel('工资')
ax.set_ylabel('概率')

plt.show()

# 值函数迭代
mcm = McCallModel(w=w_new, q=q_new)
valfunc_VFI, flag = mcm.VFI()
```

```{code-cell} ipython3
mcm = McCallModel(w=w_new, q=q_new)
valfunc_VFI, flag = mcm.VFI()
valfunc_VFI
```

```{code-cell} ipython3
def plot_epochs(epochs_to_plot, quit_allowed=1):
    "绘制由不断增加的训练轮数所得到的值函数。"
    qlmc_new = Qlearning_McCall(w=w_new, q=q_new, quit_allowed=quit_allowed)
    qtable = np.zeros((len(w_new),2))
    epochs_to_plot = np.asarray(epochs_to_plot)
    # 绘图
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(w_new, valfunc_VFI, '-o', label='VFI')

    max_epochs = np.max(epochs_to_plot)
    # 迭代训练轮数
    for n in range(max_epochs + 1):
        if n%(max_epochs/10)==0:
            print(f"进度: 训练轮数 = {n}")
        if n in epochs_to_plot:
            valfunc_qlr = valfunc_from_qtable(qtable)
            error = compute_error(valfunc_qlr, valfunc_VFI)

            ax.plot(w_new, valfunc_qlr, '-o', label=f'QL:训练轮数={n}, 平均误差={error}')


        new_qtable = qlmc_new.run_one_epoch(qtable)
        qtable = new_qtable

    ax.set_xlabel('工资')
    ax.set_ylabel('最优值')
    ax.legend(loc='lower right')
    plt.show()
```

```{code-cell} ipython3
plot_epochs(epochs_to_plot=[100, 1000, 10000, 100000, 200000])
```

上述图表表明：

* Q-learning算法在学习那些很少被抽取到的工资水平的Q表时会遇到困难

* 随着训练周期的延长，对通过值函数迭代计算得到的"真实"值函数的近似质量会提高

## 在职工人不能辞职

在方程组{eq}`eq:old4`中描述的前述时序差分Q-learning版本允许在职工人辞职，即拒绝其现有工资，转而在本期领取失业补助并在下期获得新的工作机会。

这是{doc}`这个QuantEcon讲座<mccall_model>`中描述的McCall工人不会选择的选项。

参见{cite}`Ljungqvist2012`第6章关于搜索的证明。

但在Q-learning的背景下，给予工人辞职并在失业期间获得失业补助的选项，实际上通过促进探索而非过早地进行利用，加快了学习过程。

为了说明这一点，我们将修改时间差分公式，以禁止已就业的工人辞去她之前接受的工作。

基于对可选择项的这种理解，我们得到以下时间差分值：

$$
\begin{aligned}
\widetilde{TD}\left(w,\text{accept}\right) & = \left[ w+\beta\widetilde{Q}^{old}\left(w,\text{accept}\right) \right]-\widetilde{Q}^{old}\left(w,\text{accept}\right) \\
\widetilde{TD}\left(w,\text{reject}\right) & = \left[ c+\beta\max_{a'\in\mathcal{A}}\widetilde{Q}^{old}\left(w',a'\right) \right]-\widetilde{Q}^{old}\left(w,\text{reject}\right),\;w'\sim F
\end{aligned}
$$ (eq:temp-diff)

事实证明，公式{eq}`eq:temp-diff`与我们的Q学习递归{eq}`eq:old3`结合使用，可以让我们的智能体最终学习到最优值函数，就像在可以重新抽取选项的情况下一样。

但是学习速度较慢，因为如果代理人过早接受工资报价，就会失去在同一回合中探索新状态和调整该状态相关价值的机会。

当训练轮数/回合数较低时，这可能导致次优结果。

但是如果我们增加训练轮数/回合数，我们可以观察到误差会减小，结果会变得更好。

我们用以下代码和图表来说明这些可能性。

```{code-cell} ipython3
plot_epochs(epochs_to_plot=[100, 1000, 10000, 100000, 200000], quit_allowed=0)
```

## 可能的扩展

要将算法扩展到处理连续状态空间的问题，一个典型的方法是限制Q函数和策略函数采用特定的函数形式。

这就是**深度Q学习**的方法，其核心思想是使用多层神经网络作为良好的函数逼近器。

我们将在后续的quantecon课程中讨论这个主题。

