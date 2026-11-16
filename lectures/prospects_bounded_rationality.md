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
  title: 1993 年对宏观经济学中有限理性前景的展望
  headings:
    Overview: 概览
    The quest, restated: 追问，重述
    'The debit side: how much we must hard-wire': 借方一栏：我们必须预先设定多少内容
    Why the econometricians have not returned the compliment: 为何计量经济学家未曾回礼
    Why the econometricians have not returned the compliment::The nuisance parameter, made concrete: 具体化的滋扰参数
    Why the econometricians have not returned the compliment::When the equilibrium is not unique: 当均衡并非唯一时
    'The credit side: selection, computation, and a returned gift': 贷方一栏：选择、计算，以及一份被回赠的礼物
    Reading the ledger against the quest: 对照追问来解读这份账本
    'Postscript: what became of the program': 后记：这一研究计划后来的走向
    'Postscript: what became of the program::The arbitrariness was disciplined': 任意性被规训了
    'Postscript: what became of the program::The transition dynamics arrived, in a narrower form than hoped': 转型动态理论确实到来了，但形式比人们所期望的要狭窄
    'Postscript: what became of the program::The econometricians did return the compliment': 计量经济学家终究回礼了
    'Postscript: what became of the program::A second retreat, made differently': 第二次以不同方式进行的撤退
    'Postscript: what became of the program::The algorithms kept crossing over': 算法持续跨界流动
    'Postscript: what became of the program::Reading the ledger again': 再度解读这份账本
    Exercises: 练习
---

(prospects_bounded_rationality)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 1993 年对宏观经济学中有限理性前景的展望

```{index} single: Bounded Rationality; Prospects
```

```{contents} Contents
:depth: 2
```

## 概览

本讲是本系列的收官讲座，其性质与其他各讲不同。

它不携带任何新模型。

相反，它汇集了 {cite:t}`Sargent1993` 于 1993 年记录下的判断——他在结束《宏观经济学中的有限理性》一书时所留下的观点、保留意见和期望——并将它们与本书开篇、也是本系列第一讲 {doc}`bounded_rationality` 开篇所提出的追问联系起来。

那个追问有一个目标：一套关于**转型动态**的理论，即 1989 年东欧改革者们在没有地图的情况下不得不应对的非均衡调整过程。

其路径是将理性主体从我们的模型中驱逐出去，代之以行为像**计量经济学家**一样的"人工智能"主体——他们收集数据、形成理论、进行估计并不断适应。

我们现在可以追问：截至 1993 年，这条路径把我们带到了离目标多远的地方。

萨金特自己的答案是一份账本，正反两方均有记录。

在贷方一栏：适应性动态作为在多重均衡中进行**选择**的手段，作为**计算**均衡的工具——演化编程，以及作为计量经济学家新工具的来源。

在借方一栏：最初的奖赏——一套转型动态理论——在很大程度上仍未兑现；主体必须被大量提示引导；而且出现了一个引人注目的不对称现象。

这一研究计划的初衷是让我们模型中的主体表现得更像计量经济学家。

萨金特观察到，计量经济学家们却并未以同样的方式回礼。

本讲将解释这一不对称现象，赋予它一个具体的数值形象，并探讨它在何时失效。

随后，本讲将逐一清算 {doc}`bounded_rationality` 所承诺的四项回报，其中一项——新的动态来源——是本系列确实兑现、但萨金特最终账本中却未曾列入的。

本讲最后附有一篇后记，讲述该研究计划在 1993 年之后的演变。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## 追问，重述

{ref}`br_quest` 阐述了下文一切都要以此为标准来衡量的论证。

理性预期提出了两项要求，个体理性以及认知的相互一致性，而当一个理性预期模型被拿去与数据对照时，它赋予模型内部主体的知识，远超过研究他们的计量经济学家所拥有的知识。

有限理性研究计划试图从另一端来弥合这一差距：让主体像计量经济学家必须做的那样，从数据中学习这些分布。

人们曾希望这能带来对系统*仍处于调整过程中*的描述，即在信念与结果尚未达成相互一致之前的状态：一套转型动态理论。

{ref}`br_payoffs` 列出了这项研究计划可能带来的四项收益：均衡选择、计算、转型动态与体制变迁的分析，以及新的动态来源。

我们依照 {cite:t}`Sargent1993` 自己的记账方式来盘点：先看借方，再看核心的不对称现象，最后看贷方；然后再逐一清算这四项收益。

## 借方一栏：我们必须预先设定多少内容

第一项保留意见关乎**任意性**。

有限理性最容易通过其*反面*——理性预期——来定义，而这种可塑性本身就是一个负担。

一旦我们不再坚持主体知晓均衡，我们就必须逐案决定：他们究竟*知道*什么，又是如何学习的。

他们是知道自己的效用函数和利润函数，还是也必须学习这些？

他们是懂微积分和动态规划，还是仅凭试错？

他们是仅从自身经验中学习，还是也从他人经验中学习？

我们把他们所学限定在哪一类近似函数之中？

本系列中的每一个模型都是通过**预先设定**来回答这些问题的，即对主体进行大量提示，并盯着我们所期望其达到的结果。

本系列沿着提示程度递减的梯度展开，每一步都对其主体少告知一些。

布雷的最小二乘学习者（{ref}`br_bray`；另见 {doc}`ls_learning`）——他们将在下文再次出现——知道正确的供给曲线，只需估计一个条件期望值代入其中即可。

该书投资案例中的马塞特-萨金特主体（见 {doc}`ls_learning`）懂得动态规划，知道自己的回报函数，也知道运动法则的参数形式——他们仅仅缺少其系数，而系数是通过向量自回归来更新的。

{doc}`olg_adaptive_money` 中的鲁宾斯-门罗主体知道自己的效用函数，但不知道他们所面对的分布；{doc}`learning_approximation` 中的主体还必须额外选择一种函数形式；{doc}`exchange_rate_learning` 中的牛顿-拉夫森学习者知道自身收益的梯度。

即便是受到提示最少的主体——{doc}`exchange_rate_learning` 中的遗传算法字符串，以及 {doc}`genetic_classifier` 与 {doc}`marimon_mcgrattan_sargent` 中的分类器系统，它们只能看到一个收益信号，只有在体验到效用时才能识别它——仍然被告知*何时*做出选择、*以何种*信息为条件，而其记账系统与遗传算子的整个装置都是手工设计的，并以基约塔基-赖特均衡为目标。

萨金特点明了这一后果："结果对适应性算法细节的一种相伴敏感性"。

{doc}`marimon_mcgrattan_sargent` 中的经济体 A2 是本系列中最尖锐的案例（{ref}`mms_a2`）。

对于有耐心的主体而言，投机性均衡是该经济体*唯一*的平稳均衡，然而分类器主体却落定于基本面交易模式。

这并非学习在诸均衡之间进行选择，因为根本没有可供选择的对象。

这是收敛到一个并非主体所处经济体之均衡的结果。

{cite:t}`MarimonMcGrattanSargent1990` 曾怀疑，早期的短视行为"将经济导向了低贴现因子的平稳均衡，例如基本面均衡"，而 {ref}`mms_impatience` 揭示了其中的原因：接力传递机制使得主体实际上变得非常没有耐心。

这一结果乃是主体构建方式的产物，该论文本身也报告（其脚注 15）称，马里蒙和米勒研究的一种遗传算法在 30 次运行中有 25 次达到了投机性均衡。

第二项保留意见关乎**简单性**。

我们赋予主体的学习任务，与一门初级计量经济学课程相比都显得微不足道，更不用说现实中的企业和家庭实际上正在隐含求解的那些任务了。

我们要求一个主体学习单一的、时不变的决策规则，或一组固定的条件期望。

我们并未要求它去学习一个联立方程系统的参数，或去推断从政策体制到分布的映射——而这些正是计量经济学困难所在。

甚至连学习什么这一选择本身也是一种提示。

{doc}`learning_approximation` 中被限制于低阶多项式的主体，收敛到的是一个近似均衡——这是受限问题的一个不动点，而非理性预期均衡——建模者对函数族的选择决定了他们与真实均衡的接近程度。

综合来看，这些保留意见直接触及最初的奖赏。

我们让适应性主体置身其中的环境，比我们真正关心的那些转型过程要稳定和友善得多。

收敛速度的结果十分稀少，而可处理性又迫使我们对主体信念的分布施加严格的限制。

因此，萨金特在 1993 年判断，适应性过程的文献远未能为一套实时转型动态理论——正是这场追问所要寻找的东西——提供牢靠的基础。

不过，他并不愿以这一失败作为结语，他所选用的棒球比喻是刻意谦逊的：

> 若以适应性方法迄今未能"打出全垒打"、即未能给出一套良好的转型动态理论这一失败作为本文的结尾，那既不明智，也不公平。转型动态问题由来已久、困难重重。因此，或许可以将这些方法加深了我们对该问题的理解，视为打出了一垒安打，或者至少是一记高飞牺牲打。

## 为何计量经济学家未曾回礼

这便是赋予本讲以主题的不对称现象。

有限理性研究计划归根结底是一场运动，旨在让我们模型中的主体表现得更像那些构建并估计这些模型的计量经济学家。

模仿是最真诚的恭维。

因此我们本可以预期，宏观计量经济学家会争相将这些模型拟合到数据上。

然而并未出现这样的争相追捧。

{cite:t}`Chung1990` 估计了西姆斯政策制定者模型（{ref}`olg_phillips`），在该模型中，政府对一条菲利普斯曲线运行卡尔曼滤波（{doc}`kalman`），并认为其系数会按随机游走的方式漂移。

除了结构参数之外，钟还估计了漂移的方差，以及启动政府滤波器的先验均值。

萨金特称这是"据我所知在计量经济学上唯一一个认真对待有限理性的宏观经济实现"。

为何如此冷淡？

其中的原因值得详细说明，因为这并非品味问题。

应用计量经济学家所遵循的至理名言是卢卡斯的告诫：*要警惕携带自由参数而来的理论家*。

用一个有限理性主体替代一个理性主体，会**增加**参数：描述信念以及信念如何变动的参数。

以最简单的情形——布雷模型为例。

相对于其理性预期版本，适应性版本至少增加了初始信念，以及一个设定增益序列的参数，而人们可能还想用更多参数来描述增益的形态。

这已经足以让卢卡斯的警告有的放矢。

但还有一个更深层的问题，这也正是为何这些新增参数不仅不受欢迎，而且实际上难以估计的原因。

由于布雷的适应性系统会**收敛**到理性预期均衡——在她的模型中该均衡是唯一的——这些额外的参数仅仅影响*瞬态过程*。

数据的渐近分布不包含关于它们的任何信息。

让我们直接看一看。

### 具体化的滋扰参数

布雷的蛛网经济 {cite:p}`Bray1982` 通过下式设定市场价格：

$$
p_t = a + b\, \beta_t + u_t,
$$

其中 $\beta_t$ 是主体预期的价格，通过对过去价格取平均而形成：

$$
\beta_t = \beta_{t-1} + \gamma_t\,(p_{t-1} - \beta_{t-1}),
\qquad
\gamma_t = \frac{1}{t + t_0},
$$

$u_t$ 是一个独立同分布的标准正态冲击。

常数 $t_0$ 是主体赋予其初始信念的权重，以观测值个数计量：当 $t_0 = 50$ 时，他们将 $\beta_0$ 视为总结了此前五十个价格，因此经过 $t$ 期之后，$\beta_0$ 依然带有 $t_0/(t + t_0)$ 的权重。

这样的权重是必要的，否则这个练习将毫无内容可言。

当 $t_0 = 0$ 时，首个增益为 $\gamma_1 = 1$，因此 $\beta_1 = p_0$ 恰好成立，初始信念在一期之后便被抹去——计量经济学家将没有任何瞬态过程可供尝试估计。

当 $b < 1$ 时，无论起始值为何，信念都会收敛到理性预期值 $\beta^\star = a/(1-b)$。

这正是布雷的结果，也正是 {ref}`br_one_recursion` 中的 E-稳定性条件：相应的微分方程 $d\beta/d\tau = T(\beta) - \beta = a + (b - 1)\beta$ 恰好在 $b < 1$ 时稳定。

```{code-cell} ipython3
a, b = 5.0, 0.7
t0 = 50                                       # weight on the initial belief
β_star = a / (1 - b)                          # rational expectations belief

def simulate(β0, u):
    "Bray's cobweb under least squares learning, given a shock path u."
    T = len(u)
    β = np.empty(T)
    p = np.empty(T)
    β[0] = β0
    for t in range(T):
        p[t] = a + b * β[t] + u[t]
        if t + 1 < T:
            β[t + 1] = β[t] + (1 / (t + 1 + t0)) * (p[t] - β[t])
    return p, β

print(f"rational expectations belief β* = {β_star:.3f}")
```

初始信念 $\beta_0$ 就是这个额外的"有限理性"参数。

来看三个初始信念迥异的经济体如何朝 $\beta^\star$ 靠拢。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Beliefs drift toward the rational expectations value, slowly"
    name: fig-pbr-belief
---
rng = np.random.default_rng(0)
u = rng.standard_normal(400)

fig, ax = plt.subplots(figsize=(7.5, 4))
for β0, colour in [(2.0, 'C0'), (16.667, 'C1'), (40.0, 'C2')]:
    _, β = simulate(β0, u)
    ax.plot(β, color=colour, lw=1.3, label=fr"$\beta_0 = {β0}$")
ax.axhline(β_star, color='k', ls='--', lw=0.8, label=r"$\beta^\star$")
ax.set_xlabel("$t$")
ax.set_ylabel(r"belief $\beta_t$")
ax.legend(frameon=False)
plt.show()
```

三者都朝 $\beta^\star$ 靠拢，但速度缓慢。

每一期，学习递归平均会以 $1 - (1 - b)/(t + t_0)$ 的因子缩小差距 $\beta_t - \beta^\star$，因此经过 $t$ 期之后，大约还剩下 $(t_0/(t + t_0))^{1-b}$ 的差距。

```{code-cell} ipython3
for β0 in (2.0, 40.0):
    _, β = simulate(β0, u)
    share = (β[-1] - β_star) / (β0 - β_star)
    print(f"β0 = {β0:4.1f}: share of the initial gap left after {len(u)} periods = {share:.2f}")
print(f"approximation (t0 / (t + t0))^(1 - b) = {(t0 / (len(u) - 1 + t0)) ** (1 - b):.2f}")
```

经过400期之后，初始差距大约还有一半尚未消失。

这比普通估计量趋于稳定所需的 $1/\sqrt{t}$ 速度要慢，正如本书第五章所警示的那样：只有当相应微分方程的斜率（这里是 $b - 1$）低于 $-1/2$ 时，最小二乘学习才会以 $\sqrt{T}$ 的速度收敛。

然而，*经济体*遗忘得慢，并不等同于*计量经济学家*能获得信息。

现在换上计量经济学家的视角。

给定一个价格样本以及结构参数 $(a, b)$，该模型对任何候选初始信念 $\beta_0$ 都蕴含着一个冲击 $\hat u_t = p_t - a - b\,\beta_t(\beta_0)$，因为信念路径由 $\beta_0$ 和观测到的价格共同确定。

隐含冲击的平方和衡量了给定的 $\beta_0$ 与数据的拟合程度。

```{code-cell} ipython3
def belief_path(β0, p):
    "Belief sequence implied by an initial belief and an observed price path."
    T = len(p)
    β = np.empty(T)
    β[0] = β0
    for t in range(1, T):
        β[t] = β[t - 1] + (1 / (t + t0)) * (p[t - 1] - β[t - 1])
    return β

def ssr(β0, p):
    "Sum of squared implied shocks, as a function of the belief parameter β0."
    β = belief_path(β0, p)
    return np.sum((p - a - b * β) ** 2)
```

数据能否将 $\beta_0$ 确定下来，是一个关于*信息*随着样本增大而积累速度的问题。

自然的判读方式是看**超额**残差平方和——一个错误的 $\beta_0$ 相对于最优值拟合得有多差。

对于一个普通的、可良好识别的参数而言，每一个新的观测值都会增加对错误取值的惩罚，因此超额值会随 $T$ 成比例增长，置信区间会以 $1/\sqrt{T}$ 的速度收缩。

由于这些冲击的方差为1，超额值同时也是一个似然比统计量，因此惩罚值超过3.84，即可在5%的显著性水平上拒绝该错误取值。

来看看这里会发生什么。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Information about the belief parameter stops accumulating"
    name: fig-pbr-excess
---
grid = np.linspace(2, 40, 80)

def excess_curve(T, seed=1):
    "SSR(β₀) − min SSR across the β₀ grid, for a sample of length T."
    u = np.random.default_rng(seed).standard_normal(T)
    p, _ = simulate(β_star, u)
    curve = np.array([ssr(b0, p) for b0 in grid])
    return curve - curve.min()

fig, ax = plt.subplots(figsize=(7.5, 4))
for T, colour in zip((50, 500, 5000), ('C0', 'C1', 'C2')):
    ax.plot(grid, excess_curve(T), color=colour, lw=1.5, label=f"$T = {T}$")
ax.axvline(β_star, color='k', ls='--', lw=0.8)
ax.set_xlabel(r"belief parameter $\beta_0$")
ax.set_ylabel("excess sum of squares")
ax.legend(frameon=False)
plt.show()
```

$T = 50$ 时的曲线较低，但超过几百个观测值之后，这些曲线便几乎彼此重叠，而不是变得越来越陡峭。

为了说明这并非所绘范围造成的假象，我们将错误 $\beta_0$ 所受的惩罚，与从同样数据中估计出的错误*斜率* $b$ 所受的惩罚作一比较。

我们计算两种斜率惩罚：一种是将截距 $a$ 固定在真实值上，另一种是将 $a$ 与 $b$ 一起重新估计，并对每种惩罚在20个样本上取平均。

```{code-cell} ipython3
def ssr_a_free(r):
    "Sum of squared residuals after re-estimating the intercept (removing the mean)."
    return np.sum((r - r.mean()) ** 2)

print(f"{'':>6}   {'β₀ = 2':>8}   {'b = 0.75':>10}   {'b = 0.75':>14}   {'b = 0.75, a re-est.':>19}")
print(f"{'T':>6}   {'':>8}   {'a known':>10}   {'a re-estimated':>14}   {'largest of 20':>19}")
for T in (50, 500, 5_000, 50_000):
    pen_β0, pen_b_known, pen_b_free = [], [], []
    for seed in range(20):
        u = np.random.default_rng(seed).standard_normal(T)
        p, _ = simulate(β_star, u)
        β = belief_path(β_star, p)              # belief path at the true β0
        pen_β0.append(ssr(2.0, p) - ssr(β_star, p))
        pen_b_known.append(np.sum((p - a - 0.75 * β) ** 2)
                           - np.sum((p - a - b * β) ** 2))
        pen_b_free.append(ssr_a_free(p - 0.75 * β) - ssr_a_free(p - b * β))
    print(f"{T:>6}   {np.mean(pen_β0):8.1f}   {np.mean(pen_b_known):10.1f}   "
          f"{np.mean(pen_b_free):14.3f}   {np.max(pen_b_free):19.3f}")
```

这几列的表现完全不同。

初始信念取错所受到的惩罚停止了增长。

超过几千个观测值之后，它就被固定在接近一个常数的水平上，此后每增加一次观测都无法提供关于 $\beta_0$ 的信息。

$\beta_0$ 的置信区间永远不会收缩；该参数根本无法被一致地估计出来。

其中的原因在信念递归中清晰可见。

给定观测到的价格，计量经济学家在 $t$ 期重建出的信念仅通过权重 $t_0/(t + t_0)$ 依赖于 $\beta_0$，而这些权重的平方和是有限的。

$\beta_0$ 对经济体所留下的长远阴影，其余部分则由观测到的价格本身所承载。

当 $a$ 固定在真实值时，斜率取错所受到的惩罚随样本量成比例增长：数据量增加一百倍，就会带来对错误取值强一百倍的反证力度，这正是可识别性应有的样子。

但这种可识别性是借来的。

在 $a$ 固定的情况下，错误的 $b$ 会改变预测的均价，而均价 $a/(1-b)$ 本就是理性预期版本的模型也能够识别的量。

学习模型所新增的，是将 $a$ 与 $b$ 区分开来的可能性，而要做到这一点，数据必须能观察到信念的变动。

第三列显示出数据对此观察得何其有限。

在重新估计 $a$ 的情况下，这一惩罚还不到1，并且样本量每增加十倍，它只增加不到十分之一；在20个样本中，即便样本量达到50,000，该惩罚也没有一次接近3.84。

在 $1/t$ 增益之下，信念的变动会逐渐消失，因此学习瞬态过程是将 $a$ 与 $b$ 区分开来的唯一依据：使 $\beta_0$ 无法识别的那种弱点，同样以更温和的形式困扰着斜率。

这正是问题的技术核心所在。

有限理性所增添的参数存在于瞬态过程之中，而在递减增益之下，那些能够将旧参数彼此区分开的信息，也同样存在于瞬态过程之中。

无论我们此后观察经济多长时间，一个瞬态过程所贡献的信息量都是有限的，因此这些参数就变成了**估计上的滋扰**——它们进入似然函数，而数据对它们能够说明的内容很少。

萨金特还以其一贯的"或许"口吻，补充了另一个解释计量经济学家冷淡反应的原因。

许多应用宏观计量经济学家所寻求的方法，恰恰是要*减少*解释数据所需的参数数目。

而有限理性所提供的恰恰不是减少，而是相反。

它提供的是更多参数，其中大多数还识别得很弱。

于是，这份恭维只是单向而行。

理论家们按照计量经济学家的形象重塑了他们的模型主体；而计量经济学家们面对充满额外、识别薄弱参数的模型，且已被卢卡斯明确告诫要警惕正是此类情况，便婉拒了这份礼物。

### 当均衡并非唯一时

上述论证假定学习会收敛到一个唯一的、E-稳定的均衡，正如布雷模型在 $b < 1$ 时的情形。

此时，初始信念和增益的形态只能影响瞬态过程。

而在均衡多重的情况下，这一前提可能并不成立。

在 {doc}`olg_adaptive_money` 中，初始信念决定了经济体最终会落入低通胀稳态，还是走向崩溃（{ref}`olg_reversal`）。

在 {doc}`exchange_rate_learning` 中，在牛顿-拉夫森学习者收敛的那种校准下，初始储蓄率和投资组合份额将永远固定住极限汇率（{ref}`xr_dead_hand`）。

在初始信念从一个连续统的极限中进行选择的情形下（如汇率经济中的情形），其影响是永久性的，而永久性的影响正是每一个新观测值都有助于度量的那种效应。

在那样一个确定性的经济体中，若带有观测误差，数据会以通常的速度度量初始信念的影响，就如同度量一个结构参数一样。

{ref}`pbr_ex2` 说明了这一点，同时也表明，仅有永久性并不总是足够的。

当信念由观测到的价格所驱动时（如布雷模型中的情形），计量经济学家便可以从数据中重建出这些信念，而无论均衡结构如何，初始信念在这种重建中的权重都会逐渐消失。

冲击同样会使汇率经济中的图景变得模糊：一旦引入冲击，即便起点完全相同，也会收敛到不同的汇率（{ref}`xr_ex5`），因此极限不再仅仅反映初始信念本身。

## 贷方一栏：选择、计算，以及一份被回赠的礼物

这份账本并非单方面的。

与前述借方相对，萨金特列出了三项成就，其中最后一项悄然抵消了刚刚描述的那种不对称现象的一部分。

在开篇一章中，他还将模型"成功激发了有用的均衡选择检验"——即实验室实验——计入贷方。

**均衡选择。**

在一个理性预期模型存在多重均衡的地方，一个由适应性主体构成的系统往往会收敛到*某一个特定的*均衡，从而使学习成为在多重均衡中进行选择的手段。

系列中最清晰的例子是 {doc}`olg_adaptive_money` 中的经济体，其中最小二乘学习者选择了低通胀稳态，而理性预期动态却拒绝了这一稳态（{ref}`olg_reversal`），并且 {cite:t}`MarimonSunder1993` 的实验室受试者也走向了同样的方向。

在没有赤字的情形下，同样的学习者消灭了 {doc}`bounded_rationality` 中的泡沫，并选择了其无泡沫均衡（{ref}`olg_ex5`）。

在 {doc}`marimon_mcgrattan_sargent` 中，经济体 B 同时存在一个基本均衡和一个投机均衡，而分类器系统选择了基本均衡；在经济体 A1 中，由于基本均衡是唯一的均衡，分类器系统也找到了它。

{doc}`exchange_rate_learning` 中的汇率只能在较弱的意义上记入贷方：学习过程仅凭记住起点来决定汇率，而其极限也会随着算法的调参——其稳定器 $\kappa$ 及其增益——而变化（{ref}`xr_dead_hand`）。

即便如此薄弱的意义，也仅在牛顿-拉夫森学习者收敛的情形下才成立；在阿里福维奇的校准设定下，使用相同的调参，学习过程根本无法收敛。

该书也承认，这"作为确定汇率的依据，似乎是一根相当脆弱的稻草"。

正如前文所述，经济体 A2 应归入借方。

而且，选择并不等同于一条福利定理。

在该书仿照布罗克精神所举的例子中——{doc}`olg_adaptive_money` 将其作为一个警示而回顾——最小二乘法选择了一个平稳均衡，而每一个非平稳均衡在帕累托意义上都优于它。

萨金特坦率地承认，他对这种用法的偏爱，与他对执行这种选择的动态过程本身的怀疑之间，存在某种张力：

> 我知道，对实时动态过程心存怀疑、却又保留由其所选中的均衡，这在逻辑上是不一致的。我承认，我对第六章所描述的货币模型中所执行的那种选择所抱有的偏爱，部分源于我先验地相信被选中的那些均衡在我看来是合理的。

该书第六章正是关于实验室实验的章节：马里蒙和桑德针对 {doc}`olg_adaptive_money` 中单一货币经济体所做的实验，以及阿里福维奇针对 {doc}`exchange_rate_learning` 中双货币经济体所做的实验。

萨金特所指的这种选择，正是最小二乘学习与马里蒙和桑德的受试者一致认同的那个低通胀均衡。

**演化编程。**

如果一个由适应性主体构成的群体能够可靠地收敛到某个均衡，我们就可以将该群体作为一种*计算*该均衡的方法来运行，尤其是在那些复杂到无法手工求解的模型中（{ref}`gc_evolutionary`）。

这一过程属于"猜测—验证"式的：适应性系统提供猜测，而经济学家则检验其是否满足均衡条件。

这正是 {cite:t}`MarimonMcGrattanSargent1990` 针对五种商品的基约塔基-赖特经济所提出的方法，因为当时并没有现成的解析刻画方法（{doc}`marimon_mcgrattan_sargent`）：他们的分类器经济最终稳定在一组交易策略和持有概率上，暗示着存在某种均衡（"我们大概可以借此验证其是否构成均衡"），尽管论文本身只是报告这些模式"几乎似乎描述了"一个基本均衡。

系列中最清晰的实例是 {doc}`learning_approximation`。

在该讲座中，马塞特的参数化预期方法——一种用于计算均衡的工具——有一个学习层面的对应物：主体随着数据的到来不断更新其预测规则，而二者共享相同的不动点，并且通过相关的微分方程，还共享相同的稳定性条件（{ref}`lae_pea`）。

该书早已指出，马塞特的方法"在许多方面都具有学习或适应算法的特征"。

以这种方式使用时，动态过程本身是否可信已无关紧要。

该书第七章描述了兰德尔·赖特如何在计算一个类型份额内生化的基约塔基-赖特均衡时，直接把理论支付告诉给他那个不断演化的群体——而这些支付原本只有通过极其漫长的历史才能习得：这作为实时动态模型是绝望的，但作为一种计算手段却是无害的。

正是这种劳动分工，使人们得以既怀疑动态过程，又依然用它来进行计算。

萨金特预期这种用法将会被频繁采用。

**为计量经济学家提供的新工具。**

在这里，不对称现象出现了反转。

关于并行算法和遗传算法的文献，为计量经济学家提供了新的计算工具——其中包括遗传算法和随机高斯-牛顿程序——用以求解他们*自身*的估计和优化问题。

例如，埃伦·麦格拉坦运用遗传算法来估计非线性理性预期模型，用它们来搜索极大值的邻近区域，然后再切换到牛顿法，这一用法在 {doc}`genetic_classifier` 中已有提及。

因此，计量经济学家终究还是采纳了这些适应性算法，只不过并非将其作为他们所研究的主体的*模型*，而是作为自己手中的*工具*。

这份恭维终究得到了回赠，只不过是从侧门进入的：算法跨越了界限，而模型没有。

**新的动态来源。**

该书第二章还列出了另一项收益：适应性主体或许能够生成一些理性预期模型难以产生的行为，例如无交易定理所排除的那种交易量。

萨金特在最终的账本总结中并未将其计入，但本系列确实提供了两个实例。

{doc}`exchange_rate_learning` 中阿里福维奇基于遗传算法的经济体所产生的汇率始终无法稳定下来，即便储蓄已接近其均衡水平（{ref}`xr_ga`），而且即便在该经济体中每一个两种货币都被赋予价值的均衡都对应着恒定的汇率。

而 {doc}`olg_adaptive_money` 中采用恒定增益的政府，则产生了一种反复出现的锯齿状波动：从自我确认的通胀率反复逃向零通胀（{ref}`olg_phillips`），这是该模型唯一的自我确认均衡所不具备的动态特征。

## 对照追问来解读这份账本

让我们把这两栏账目与我们最初设定的目标对照一番，并对照 {ref}`br_payoffs` 中的四项成果。

| 成果 | 结论 | 本系列文章中的体现之处 |
|---|---|---|
| 均衡选择 | 贷方，但有保留 | {doc}`olg_adaptive_money`（低通胀情形）；{doc}`marimon_mcgrattan_sargent`（经济体 B）；在 {doc}`exchange_rate_learning` 中仅在较弱的意义上成立；经济体 A2 则是借方 |
| 计算 | 贷方 | {doc}`learning_approximation`；{doc}`genetic_classifier`；{doc}`marimon_mcgrattan_sargent`（五种商品经济体）；为计量经济学家提供的工具 |
| 转型动态与体制变迁 | 1993 年为借方，此后部分偿清 | {doc}`olg_adaptive_money` 中的逃逸动态，在 {doc}`phillips_escaping_nash` 中得到刻画 |
| 新的动态来源 | 贷方，1993 年的账本中尚未出现 | {doc}`exchange_rate_learning`（遗传算法带来的波动性）；{doc}`olg_adaptive_money`（逃逸锯齿形态） |

**借方**一栏保存着奖赏本身。

萨金特曾在第二章中论证道，在不同政策体制之间比较理性预期均衡，这一做法本身就与理性预期不相容，而一套关于适应性主体如何从一种体制转移到另一种体制的理论，本应弥补这一缺陷。

一套关于实时转型动态的理论——即东欧改革者们所缺乏的那张地图——到 1993 年为止，在很大程度上仍未被认领，其阻碍在于任意性、需要大量提示、学习任务过于简单，以及经验支持的匮乏。

**贷方**一栏保存着这段旅程沿途所交付的成果：一种在多重均衡中进行选择、且选择结果不依赖于算法具体细节的方法，一种计算那些难以用解析方法处理的均衡的实用方法，新的动态来源，以及一次计算技术向计量经济学本身的转移。

贯穿全书的组织性意象，正是计量经济学家与模型内部主体之间的鸿沟。

理性预期通过将主体提升到计量经济学家所不具备的知识水平，来弥合这一鸿沟。

有限理性则提议从另一端来弥合它，把主体降低到计量经济学家的水平，让他们去学习。

按萨金特自己的记账方式，到 1993 年，这一研究计划尚未交出促使它诞生的转型动态理论，而它试图去模仿的计量经济学家们，也一直对这些模型保持距离——理由很充分：这些模型增添了数据只能微弱地、且并不稳定地识别的参数，而计量经济学家想要的恰恰是更少的参数。

但它确实加深了问题的深度，选择了均衡，计算了均衡，并把自己的工具借给了那些婉拒了它的模型的计量经济学家。

这就是 1993 年时，宏观经济学中有限理性前景所处的状态。

## 后记：这一研究计划后来的走向

三十年的时间，足以让我们看清哪些 1993 年的担忧是永久性的，哪些则只是关于一个当时尚未站稳脚跟的领域。

### 任意性被规训了

第一项保留意见是，一旦我们不再坚持主体知晓均衡，就没有任何东西能告诉我们该用什么来取代它。

后来成熟起来的规训机制是**预期稳定性**。

它的核心内容在 1993 年那本书中其实已经具备。

该书第五章的一条脚注将布雷的信念极限动态写成微分方程 $d\beta/d\tau = T(\beta) - \beta = a + (b - 1)\beta$，其中 $T$ 把感知到的预测映射为最优预测。

同一条脚注将用 $T$ 来定义预期稳定性的功劳归于 {cite:t}`DeCanio1979` 以及 {cite:t}`Evans1985`、{cite:t}`Evans1989`，并将以下结果归功于 {cite:t}`MarcetSargent1989jet`：他们运用 {cite:t}`Ljung1977` 的微分方程方法证明，在一类模型中，$T(\beta) - \beta$ 支配着最小二乘学习的收敛性。

{cite:t}`Woodford1990` 运用同样的方法研究了太阳黑子均衡的学习问题，该书第五章也以关于相应微分方程的一节作结。

{ref}`br_one_recursion` 为整个系列梳理了这条脉络。

1993 年之后发生的事情，是系统化。

{cite:t}`EvansHonkapohja2001` 将这些结果加以推广，并把它们组织成一套关于可学习性的理论，如今它已成为判断一个理性预期均衡是否是适应性行为的合理结果的标准检验。

对于最小二乘法以及理性预期均衡附近的相关递减增益估计器而言，E-稳定性正是恰当的条件，人们无需进行任何模拟，便可判断哪些均衡满足这一条件。

这就缩小了任意性的范围：最小二乘法所执行的选择，并非建模者恰好写下的那个特定递归式的偶然产物。

这一条件也是有局限的，而这些局限同样得到了厘清。

它并不是一条适用于每一种算法的定理。

随机梯度学习者有可能学不到那些满足 E-稳定的均衡 {cite}`Giannitsarou2005`。

群体算法则完全处在这一理论之外：借鉴阿里弗维奇 1991 年的博士论文，该书报告说，即便在 $b > 1$、最小二乘法无法奏效的情况下，一个遗传算法种群有时也能学到布雷模型的均衡（{cite:t}`Sargent1993`，第五章），而经济体 A2 中的分类器主体则收敛到了一个根本不是均衡的结果。

在这些局限之内，这一条件确实发挥了作用，{doc}`olg_adaptive_money` 中的稳定性反转正是一个例证。

马塞特和萨金特的微分方程早在 1993 年就已经解释了这一点。

用 E-稳定性的语言来说，它表明在低通胀稳态处 $T'(2) = 1/2 < 1$，而在高通胀稳态处 $T'(3) = 2 > 1$（{ref}`olg_reversal`），而这一排序本身是关于该模型的一条定理，而非关于这些特定参数的结论（{ref}`olg_ex6`）。

这正是为何 {cite:t}`BrunoFischer1990` 采用适应性预期而非最小二乘法，却得出了相同的反转结果：那些使信念趋向已实现通胀的规则，继承了同样的稳定性条件。

这一结论甚至不需要小增益：按照 {doc}`olg_adaptive_money` 的时序设定，对于所有小于 2 的常数增益而言，低稳态都是局部稳定的，而高稳态则对任何增益都不稳定（{ref}`olg_reversal`，{ref}`pbr_ex1`）。

{cite:t}`BullardDuffy1993` 则提供了一个警示，说明不应将这一结果过度推广：在主体生存期达四期或更长的经济体中，最小二乘学习可能无法局部收敛到理性预期均衡。

### 转型动态理论确实到来了，但形式比人们所期望的要狭窄

那份奖赏原本是一套关于非均衡调整的理论。

而这一研究计划所交付的，实际上是一套关于*偏离*均衡的理论：逃逸动态。

在 1993 年，西姆斯笔下那个政府的逃逸现象，仅能从模拟中得知，该书将其呈现为关于战后通胀的一个故事与一个反故事的形式化版本。

{ref}`olg_phillips` 中所模拟出的那种反复出现的锯齿形状，来自于 {cite:t}`Sargent1999` 在 1993 年之后所研究的该模型的常数增益版本。

{cite:t}`ChoWilliamsSargent2002` 运用大偏差理论对其逃逸现象进行了解析刻画，计算出了最可能的逃逸路径以及逃逸发生的速率，而 {cite:t}`Williams2019` 又大大扩展了这一刻画。

{cite:t}`SargentWilliams2005` 随后解释了为何西姆斯和钟所模拟的经济体会逃逸，并*停留*在拉姆齐结果附近，而萨金特笔下的经济体却一次又一次地被拉回：二者的差异在于政府赋予其回归误差多大的方差。

QuantEcon 讲座 {doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 详细讲解了这两方面的内容。

这比最初追问所要求的要少。

它描述的是从自我确认均衡出发的反复性偏离，而非一个从未有过市场经济的国家迎来市场经济这一事件本身。

但这确实是一套关于一个不会安定下来的系统的、真正的理论，是推导得出而非模拟得出的，而在 1993 年，这样的理论并不存在。

### 计量经济学家终究回礼了

在这一点上，1993 年的评估在很大程度上被超越了。

我们前面看到的障碍，是新增参数存在于一个逐渐消失的瞬态过程之中。

但那一论证仅适用于增益趋于零的学习方案，在这种方案下，信念会收敛并随后停止移动。

若增益不趋于零，信念便永不安定。

它们的运动本身是数据平稳分布的一部分，而支配这种运动的参数，像普通结构参数一样，可以从整个样本中被识别出来。

1993 年的那本书其实已经预见到了这一例外情形。

书中指出，其瞬态过程来自那些终将安定下来的算法，并补充说："例外是常数增益算法"。

而钟的政府——那个唯一认真的实证案例——所运行的是一个针对其认为会漂移的系数的卡尔曼滤波器：对于给定的漂移方差，其增益并不趋于零，这正是为何那些方差能够从整个样本中被估计出来。

让我们在布雷的市场模型上验证这一论断，并给计量经济学家出一道更难的题目：连同增益一起重新估计 $a$ 和 $b$。

我们将初始信念固定在其真实值上；在常数增益之下，其影响会以几何速度消退。

```{code-cell} ipython3
def simulate_constant_gain(β0, gain, u):
    "Bray's cobweb when agents discount old prices at a fixed rate."
    T = len(u)
    β, p = np.empty(T), np.empty(T)
    β[0] = β0
    for t in range(T):
        p[t] = a + b * β[t] + u[t]
        if t + 1 < T:
            β[t + 1] = β[t] + gain * (p[t] - β[t])
    return p, β

def belief_path_cg(β0, gain, p):
    "Belief sequence implied by a candidate gain and an observed price path."
    T = len(p)
    β = np.empty(T)
    β[0] = β0
    for t in range(1, T):
        β[t] = β[t - 1] + gain * (p[t - 1] - β[t - 1])
    return β

def ssr_ab_free(p, β):
    "Sum of squared residuals after re-estimating a and b by least squares."
    X = np.column_stack([np.ones_like(β), β])
    coef, *_ = np.linalg.lstsq(X, p, rcond=None)
    resid = p - X @ coef
    return resid @ resid

gain_true, gain_wrong = 0.05, 0.08
print(f"{'T':>6}   {'gain 0.08 penalty':>17}   {'share > 3.84':>12}   "
      f"{'b = 0.75 penalty':>16}")
for T in (500, 5_000, 50_000, 200_000):
    pen_gain, pen_b = [], []
    for seed in range(20):                    # average over 20 samples
        u = np.random.default_rng(seed).standard_normal(T)
        p, β = simulate_constant_gain(β_star, gain_true, u)
        pen_gain.append(ssr_ab_free(p, belief_path_cg(β_star, gain_wrong, p))
                        - ssr_ab_free(p, belief_path_cg(β_star, gain_true, p)))
        pen_b.append(ssr_a_free(p - 0.75 * β) - ssr_a_free(p - b * β))
    pen_gain = np.array(pen_gain)
    print(f"{T:>6}   {pen_gain.mean():17.2f}   {np.mean(pen_gain > 3.84):12.0%}   "
          f"{np.mean(pen_b):16.2f}")
```

如今这两种惩罚值都会随样本量持续增长，而在最大的样本中，它们是按比例增长的：在从 50,000 到 200,000 个观测值的区间内，每一个都大致翻了四倍。

将此与在 $1/t$ 增益下重新估计 $a$ 时得到的斜率惩罚值相对照，后者从未超过 1 的一个零头。

持续存在的信念运动，提供了一种趋于零的增益所无法给出的识别变异性。

不过，每个观测值所含的信息量却很小。

在 500 个观测值的情况下，增益 0.08 根本无法与 0.05 区分开来：平均惩罚值大致为零，没有任何一个样本能在 5% 的水平上拒绝那个错误的增益。

在 5,000 个观测值的情况下，多数样本能够拒绝它，而在 50,000 个观测值时，全部样本都能做到。

增益是以通常的速率被识别出来的，但就每个观测值而言，这种识别是微弱的；{ref}`pbr_ex4` 展示了信息量如何随增益大小而缩减，而 {ref}`olg_ex7` 则针对 {doc}`olg_adaptive_money` 中的菲利普斯曲线政府提出了同样的问题。

后续的研究建立在钟的基础之上。

{cite:t}`Sargent1999` 把一个常数增益的政府置于解释战后美国通胀的论述核心，而 {cite:t}`SargentWilliams2005` 则展示了政府关于漂移系数的先验信念——以漂移协方差 $V$ 编码——如何塑造其学习动态（{doc}`phillips_priors`）。

{cite:t}`SargentWilliamsZha2006` 估计了一个模型，其中政府针对漂移系数运行卡尔曼滤波器，即西姆斯-钟设定，利用二战后美国数据来估计决定其非消失增益的漂移协方差 $V$。

{cite:t}`Primiceri2006` 估计了一个模型，其中政策制定者学习自然失业率以及通胀持续性。

这些估计结果复活了 1993 年那份担忧的一个变体。

为了拟合数据，萨金特-威廉姆斯-扎（SWZ）需要一个较大的 $V$，而批评者——包括 {cite:t}`Sargent2008` 本人——指出，由此蕴含的信念波动幅度大得令人难以置信（{doc}`phillips_priors`）。

应对之道是引入关于信念的直接证据。

**调查数据。**
{cite:t}`CarboniEllison2009` 要求模型所产生的信念再现美联储绿皮书中的预测，这使得估计出的 $V$ 缩小了几个数量级，同时仍保留了对通胀起伏历程的解释力。

**世代群组。**
{cite:t}`MalmendierNagel2016` 发现，在调查数据中，人们的通胀预期会赋予他们亲身经历过的通胀以权重，而这种增益会随年龄增长而递减。

在一个由世代交叠群体构成的经济体中，总有人正值年轻，因此总体层面的瞬态过程永远不会终结，原本对单个主体而言属于瞬态的参数，变成了平稳分布的参数（{ref}`pbr_ex3`）。

**实验。**
实验室实验——该书曾将其列为一项可取之处——提供了第三条途径：在那里，实验者了解环境，并能直接观察到选择，有时甲还能直接观察到预测本身，如 {cite:t}`MarimonSunder1993` 和 {cite:t}`Arifovic1996` 所示。

计量经济学家们也径直采纳了这些主体所用的工具。

{cite:t}`CogleySargent2005`——{doc}`phillips_drifts_volatilities` 正是沿袭其思路——拟合了一个系数以随机游走方式漂移的向量自回归模型，这正是西姆斯和钟交给其政府的那种设定，并借此追问大通胀究竟是糟糕的政策还是糟糕的运气。

因此，这份恭维终究是双向而行的。

使这些模型变得可估计的那种学习技术，其实早在钟 1990 年的模型中就已经存在。

1993 年之后发生变化的是：一种永不消失的增益，从例外情形变成了标准设定。

这一转变主要是出于经济学上的考量而采纳的——因为一个怀疑自己所处世界正在变化的主体，理应持续对旧数据打折——而它最终也带来了计量经济学上的红利。

### 第二次以不同方式进行的撤退

本书中的研究计划保留了个体理性，放弃了相互一致性：主体会进行优化，但所依据的信念仍处在他们自己的估计之中。

一支并行的文献沿着另一条不同的轴线撤退。

在汉森和萨金特 {cite:p}`HansenSargent2008` 的**稳健性**研究中，主体与计量经济学家共享同一个可被估计的近似模型，但他们并不信任这个模型。

他们针对那些与该近似模型相近、数据难以轻易将其区分开来的一组邻近模型中的最坏情形进行优化，而这一集合的大小可以通过检测误差概率来加以校准（{doc}`robust_permanent_income`）。

有界理性改变的是主体的信念，而稳健性则保留了主体与计量经济学家共有的那个模型，转而改变主体的偏好，使其趋于审慎。

这两种做法是互补的，而非相互竞争的，二者都是对 {doc}`bounded_rationality` 开篇所提问题的回答：我们该如何处理理性预期所赋予的那份知识？

一种回答是，主体应当学习计量经济学家正在学习的东西。

另一种回答是，他们应当在怀疑自己所学知识的同时，依然表现良好。

还有第三种回答，该书因篇幅所限而未予展开：保留完整的理性，让贝叶斯主体在一个理性预期均衡内部进行学习，正如 {doc}`rational_learning_re` 所做的那样。

### 算法持续跨界流动

1993 年的账本记录了，计量经济学家们即便婉拒了这些模型，却仍采纳了这些适应性算法作为计算工具。

这种交流增多了，并再次反转了方向。

霍兰德的"水桶传递"机制（{ref}`gc_bucket_brigade`）——把你的部分回报向后支付给任何促成了你的因素——是*时序差分学习*的一个先驱，而后者后来成为现代强化学习的一个组织性思想 {cite:p}`Sutton_2018`；{doc}`mccall_q` 将时序差分学习应用于一位麦考尔工人。

这种相似性深刻到足以具有实用价值：{ref}`mms_impatience` 把 {doc}`marimon_mcgrattan_sargent` 中水桶传递机制的强度，解读为以极低贴现因子计算出的价值，这有助于解释经济体 A2 的现象。

事后看来，{doc}`marimon_mcgrattan_sargent` 中的分类器系统，可以被辨认为带有手工构建的函数近似器的强化学习者；而 {doc}`genetic_classifier` 中的感知机，则是 {doc}`back_prop` 中深度网络的先驱，如今这些深度网络正充当着函数近似器的角色。

萨金特在 1993 年曾说，他要从统计学、计量经济学与人工智能所积累的方法宝库中，为其主体选取"大脑"。

事实证明，那些方法正是最后这一领域后来所构建成果的早期版本。

### 再度解读这份账本

1993 年的借方并未被全部偿清。

选择仍然众多，我们赋予主体的学习任务，相较于真实企业所求解的那些任务，依然显得简单，也没有人拿出东欧改革所呼唤的那套转型动态理论。

但这些条目已经发生了变化。

1993 年勾勒出的均衡选择理论得到了系统化，其局限也得到了厘清；非收敛现象获得了解析刻画，而不再仅仅是模拟结果；而计量经济学家们，一旦获得了增益永不消失的模型，再加上来自调查、世代群组与实验室的、关于信念本身的证据，便欣然采纳了这一切。

计量经济学家与模型内部主体之间的鸿沟依然存在。

它变窄了，而我们如今对它的宽度也已经了解了不少。

下一讲 {doc}`phillips_two_stories` 开启了一个建立在《征服美国通胀》{cite}`Sargent1999` 之上的系列讲座，其中一个不断重新估计其菲利普斯曲线的政府——西姆斯与钟笔下那位政策制定者的"成年版本"——被要求解释美国通胀的起起落落。

## 练习

```{exercise-start}
:label: pbr_ex1
```

本练习用数值方法检验 {doc}`olg_adaptive_money` 中稳定性逆转的 E 稳定性解释，并将该讲座中覆盖增益在 0 到 1 之间的线性化（{ref}`olg_reversal`）加以扩展，以求出增益究竟可以多大。

在恒定赤字经济中，预期总通货膨胀为 $\beta$ 的年轻代理人储蓄 $s(\beta) = (w_1 - w_2 \beta)/2$，政府通过印钞为每个年轻人融资实际赤字 $G$。

如果每个人都预期相同的常数 $\beta$，则实现的通货膨胀恒为

$$
T(\beta) = \frac{s(\beta)}{s(\beta) - G} .
$$

使用马尔蒙和孙德尔（Marimon and Sunder）的经济 7C，$w_1 = 6$，$w_2 = 1$，$G = 1$。

1. 求 $T$ 的不动点，并对每个不动点解析地和数值地计算 $T'$。哪些稳态是 E 稳定的？将结果与完全预见映射 $g(\pi) = A_1 - A_2/\pi$（其中 $A_1 = (w_1 + w_2 - 2G)/w_2$，$A_2 = w_1/w_2$）在相同点处的斜率进行比较。
1. 用凯根（Cagan）的适应性预期取代最小二乘法，即 $\beta_{t+1} = \beta_t + \lambda(\pi_t - \beta_t)$，其中实现的通货膨胀为 $\pi_t = s(\beta_{t-1})/(s(\beta_t) - G)$，且 $\beta_{-1} = \beta_0$。从初始信念 $1.5$、$2.5$、$2.9$、$2.99$ 和 $3.01$ 出发，对 $\lambda \in \{0.1, 0.5, 1, 1.5\}$ 进行模拟，若 $s(\beta_t) \le G$（即不存在正的价格水平）则停止。路径会走向何处？
1. 在第 2 部分中围绕每个稳态对系统进行线性化，并解析地求出使其局部稳定的所有 $\lambda > 0$ 的取值。将结果与条件 $|1 + \lambda(T' - 1)| < 1$ 进行比较，该条件是没有实现通货膨胀滞后项的递归版本所给出的条件。

```{exercise-end}
```

```{solution-start} pbr_ex1
:class: dropdown
```

这里 $s(\beta) = (6 - \beta)/2$，所以 $T(\beta) = (6 - \beta)/(4 - \beta)$。

其不动点满足 $\beta^2 - 5\beta + 6 = 0$，因此为 $\beta = 2$ 和 $\beta = 3$，这是两个稳态，而 $T'(\beta) = 2/(4 - \beta)^2$ 给出 $T'(2) = 1/2$ 和 $T'(3) = 2$。

```{code-cell} ipython3
w1, w2, G = 6.0, 1.0, 1.0

def s(β):
    return (w1 - w2 * β) / 2

def T_map(β):
    return s(β) / (s(β) - G)

A1, A2 = (w1 + w2 - 2 * G) / w2, w1 / w2
steady_states = np.sort(np.roots([1, -A1, A2]).real)

h = 1e-6
for β in steady_states:
    T_prime = (T_map(β + h) - T_map(β - h)) / (2 * h)
    g_prime = A2 / β**2
    print(f"β = {β:.0f}:  T(β) = {T_map(β):.4f}   T'(β) = {T_prime:.4f}   "
          f"g'(β) = {g_prime:.4f}   E-stable: {T_prime < 1}")
```

只有低通货膨胀稳态是 E 稳定的。

在完全预见下，排序被逆转：$g'(2) = 3/2 > 1$，而 $g'(3) = 2/3 < 1$。

在两个稳态处，$T' - 1$ 和 $g' - 1$ 的符号相反，这就是逆转现象；{ref}`olg_ex6` 证明了这一结论对所有参数都成立。

现在来看适应性预期版本。

```{code-cell} ipython3
def adaptive_path(β0, lam, n=2000):
    "Cagan adaptive expectations with the timing of olg_adaptive_money."
    β_prev, β = β0, β0
    for t in range(n):
        if s(β) <= G:
            return β, False                          # no positive price level
        π = s(β_prev) / (s(β) - G)                   # realized inflation
        β_prev, β = β, β + lam * (π - β)
    return β, True

starts = (1.5, 2.5, 2.9, 2.99, 3.01)
print("λ      " + "".join(f"β0 = {β0:<9}" for β0 in starts))
for lam in (0.1, 0.5, 1.0, 1.5):
    cells = []
    for β0 in starts:
        β_end, alive = adaptive_path(β0, lam)
        cells.append(f"{β_end:.4f}" if alive else "collapse")
    print(f"{lam:<6} " + "".join(f"{c:<14}" for c in cells))
```

每条从低于 3 的点出发的路径都收敛到低通货膨胀稳态，而刚好从 3 以上出发的路径会崩溃：这一逆转现象在凯根的适应性预期下依然存在，正如 {cite:t}`BrunoFischer1990` 所发现的那样。

对于线性化，令 $\delta_t = \beta_t - \beta^\star$。

实现的通货膨胀对当期和滞后一期的信念的反应系数分别为 $c_0 = -s^\star s'/(s^\star - G)^2$ 和 $c_1 = s'/(s^\star - G)$，其中 $s' = -w_2/2$，$s^\star = s(\beta^\star)$，且 $c_0 + c_1 = T'(\beta^\star)$。

因此 $\delta_{t+1} = (1 + \lambda(c_0 - 1))\,\delta_t + \lambda c_1\, \delta_{t-1}$。

```{code-cell} ipython3
def spectral_radius(β_ss, lam):
    "Largest root modulus of the linearized adaptive-expectations system."
    s_ss, s_prime = s(β_ss), -w2 / 2
    c0 = -s_ss * s_prime / (s_ss - G)**2
    c1 = s_prime / (s_ss - G)
    A = np.array([[1 + lam * (c0 - 1), lam * c1],
                  [1.0, 0.0]])
    return np.max(np.abs(np.linalg.eigvals(A)))

for β_ss in steady_states:
    radii = [spectral_radius(β_ss, lam) for lam in (0.5, 1.0, 1.9, 2.0, 2.1)]
    print(f"β* = {β_ss:.0f}: spectral radius at λ = 0.5, 1, 1.9, 2, 2.1:",
          np.round(radii, 3))
```

在 $\beta^\star = 2$ 处，$c_0 = 1$，$c_1 = -1/2$，因此特征多项式为 $z^2 - z + \lambda/2$；其根的乘积为 $\lambda/2$，系统恰好在 $0 < \lambda < 2$ 时稳定。

在 $\beta^\star = 3$ 处，$c_0 = 3$，$c_1 = -1$，因此多项式为 $z^2 - (1 + 2\lambda) z + \lambda$，在 $z = 1$ 处取负值；总存在一个大于 1 的根，故高稳态对任意 $\lambda > 0$ 都不稳定。

若没有滞后项，乘数将为 $1 + \lambda(T' - 1)$：在低稳态处为 $1 - \lambda/2$，对 $\lambda < 4$ 稳定；在高稳态处为 $1 + \lambda$，永不稳定。

实现的通货膨胀中的滞后项将 $\lambda$ 的可行范围减半，但并未改变排序，排序由 $T' - 1$ 的符号决定。

可行范围包括 $(0, 1]$ 内的每一个增益，因此最小二乘法在低稳态附近起作用的约为 0.75 的有效增益也在其中（{ref}`olg_reversal`）。

对于较小的 $\lambda$，任何逐渐将信念向实现的通货膨胀调整的规则都继承了 E 稳定性条件 $T' < 1$，这正是 {ref}`br_one_recursion` 的逻辑；在这里，该条件在远非小增益的情形下依然成立。

```{solution-end}
```

```{exercise-start}
:label: pbr_ex2
```

本练习询问在多重均衡下，初始信念何时可以被识别。

在布雷的（Bray's）市场中令 $a = 0$，$b = 1$。

此时每个常数信念都是自我实现的：$T(\beta) = \beta$，因此 $T' = 1$，即 {doc}`exchange_rate_learning` 中不确定方向的单位特征值。

保持增益为 $1/(t + t_0)$，其中 $t_0 = 50$。

1. 从 $\beta_0 \in \{5, 10, 15\}$ 出发，对学习经济模拟 20000 期，每个起点使用五条冲击序列，并报告最终信念。对 $a = 5$，$b = 0.7$ 重复相同的操作。
1. 使经济变为确定性的，就像 {doc}`olg_adaptive_money` 和 {doc}`exchange_rate_learning` 中的经济那样：代理人观察到 $p_t = a + b\beta_t$，而计量经济学家知道 $a$、$b$ 和 $t_0$，观察到 $y_t = p_t + e_t$，其中 $e_t$ 为独立同分布的标准正态变量。当真实值为 $\beta_0 = 10$ 时，计算 $\beta_0 = 12$ 的预期罚分，分别取 $T = 10^2, 10^3, 10^4, 10^5$，以及 $b = 0.3$ 和 $b = 0.7$（配合 $a = 5$）和 $b = 1$（配合 $a = 0$）。提示：预期罚分为 $\sum_t d_t^2$，其中 $d_t$ 是两条隐含价格路径之间的差值。解释这三种增长率。
1. 回到布雷的设定，其中冲击 $u_t$ 进入市场并影响信念，计量经济学家观察到 $p_t$。在真实值为 $\beta_0 = 10$ 的情况下，计算 $\beta_0 = 12$ 的平均罚分，取 $b = 0.7$（配合 $a = 5$）和 $b = 1$（配合 $a = 0$），以及 $T = 10^2, 10^3, 10^4$。为什么即使在 $b = 1$（此时 $\beta_0$ 会永久性地改变极限）时罚分仍然是有界的？

```{exercise-end}
```

```{solution-start} pbr_ex2
:class: dropdown
```

我们需要一个以 $(a, b)$ 为参数的经济版本和信念重建版本。

```{code-cell} ipython3
t0_ex = 50

def bray(a_, b_, β0, u):
    "Bray's market with gain 1/(t + t0) and shocks u that move beliefs."
    T = len(u)
    β, p = np.empty(T), np.empty(T)
    β[0] = β0
    for t in range(T):
        p[t] = a_ + b_ * β[t] + u[t]
        if t + 1 < T:
            β[t + 1] = β[t] + (p[t] - β[t]) / (t + 1 + t0_ex)
    return p, β

print("final beliefs after 20,000 periods, five shock sequences each")
for a_, b_ in ((0.0, 1.0), (5.0, 0.7)):
    for β0 in (5.0, 10.0, 15.0):
        finals = [bray(a_, b_, β0, np.random.default_rng(seed).standard_normal(20_000))[1][-1]
                  for seed in range(5)]
        print(f"a = {a_}, b = {b_}, β0 = {β0:4.1f}:", np.round(finals, 2))
```

当 $b = 1$ 时，极限保持在经济初始状态附近，而且在不同冲击序列下也各不相同：结果具有路径依赖性，这与该书关于汇率模型的脚注含义相同，{ref}`xr_ex5` 在该经济中对此进行了演示。

当 $b = 0.7$ 时，无论 $\beta_0$ 为何值，每个经济都会缓慢地朝 $\beta^\star = 16.67$ 收敛。

对于第 2 部分，经济是确定性的，因此可以不通过模拟测量误差而直接计算预期罚分。

```{code-cell} ipython3
def deterministic_prices(a_, b_, β0, T):
    "Prices in Bray's market with no shocks in the market itself."
    β, p = β0, np.empty(T)
    for t in range(T):
        p[t] = a_ + b_ * β
        β = β + (p[t] - β) / (t + 1 + t0_ex)
    return p

Ts = (100, 1_000, 10_000, 100_000)
for a_, b_ in ((5.0, 0.3), (5.0, 0.7), (0.0, 1.0)):
    d = deterministic_prices(a_, b_, 12.0, Ts[-1]) - deterministic_prices(a_, b_, 10.0, Ts[-1])
    expected = [np.sum(d[:T] ** 2) for T in Ts]
    print(f"b = {b_}: expected penalty at T = 1e2, 1e3, 1e4, 1e5:", np.round(expected, 1))
```

在确定性经济中，$\beta_0$ 对 $t$ 时价格的影响以 $(t_0/(t + t_0))^{1-b}$ 的速度衰减，因此预期罚分大致是阶数为 $t^{-2(1-b)}$ 的若干项之和。

当 $b = 0.3$ 时，该和收敛，罚分趋于平稳：影响衰减得太快，以致无法被很好地测量到。

当 $b = 0.7$ 时，影响衰减缓慢，罚分持续增长，但增长速度远低于 $T$（最终像 $T^{2b-1} = T^{0.4}$ 那样增长）。

当 $b = 1$ 时，影响永不衰减：价格将永远停留在 $\beta_0$ 处，罚分恰好为 $(12 - 10)^2 T = 4T$。

在这里，多重均衡使得初始信念永久存在，而数据以通常的速率识别它，就如同识别一个结构参数一样。

现在来看第 3 部分，其中冲击会影响信念。

```{code-cell} ipython3
def reconstructed_beliefs(β0, p):
    "Beliefs the econometrician reconstructs from observed prices and a candidate β0."
    T = len(p)
    β = np.empty(T)
    β[0] = β0
    for t in range(1, T):
        β[t] = β[t - 1] + (p[t - 1] - β[t - 1]) / (t + t0_ex)
    return β

for a_, b_ in ((5.0, 0.7), (0.0, 1.0)):
    rows = []
    for T in (100, 1_000, 10_000):
        pens = []
        for seed in range(20):
            p, _ = bray(a_, b_, 10.0, np.random.default_rng(seed).standard_normal(T))
            resid_wrong = p - a_ - b_ * reconstructed_beliefs(12.0, p)
            resid_true = p - a_ - b_ * reconstructed_beliefs(10.0, p)
            pens.append(np.sum(resid_wrong**2) - np.sum(resid_true**2))
        rows.append(np.mean(pens))
    print(f"b = {b_}: average penalty at T = 1e2, 1e3, 1e4:", np.round(rows, 1))
```

两种情况下罚分都是有界的。

在增益为 $1/(t + t_0)$ 的情况下，重建的信念为 $\beta_t = (t_0 \beta_0 + \sum_{s<t} p_s)/(t + t_0)$，因此错误的 $\beta_0$ 会使其偏移 $(\beta_0' - \beta_0)\, t_0/(t + t_0)$，无论 $b$ 取何值。

因此预期罚分大约为 $b^2 (\beta_0' - \beta_0)^2 t_0$，对于 $b = 0.7$ 该值接近 $0.49 \times 4 \times 50 = 98$，对于 $b = 1$ 接近 $4 \times 50 = 200$。

当 $b = 1$ 时，$\beta_0$ 确实会永久性地改变极限，但早期的冲击也会，观察到的价格告诉计量经济学家经济当前所处的位置，却无法告诉他其中有多少是由于初始信念造成的。

永久性是以通常速率进行识别的必要条件，但不是充分条件：初始信念必须留下计量经济学家无法归因于其所观察到的任何其他因素的印记。

```{solution-end}
```

```{exercise-start}
:label: pbr_ex3
```

{cite:t}`MalmendierNagel2016` 发现，人们根据自己亲身经历过的通货膨胀来形成通货膨胀预期，而增益随年龄增长而递减。

将这样的世代群组（cohorts）放入布雷的市场中，取 $a = 5$，$b = 0.7$。

在每个日期，共有 $L = 40$ 个世代群组存活；年龄为 $k = 0, 1, \ldots, L - 1$ 的世代群组以增益 $1/(k + 1 + t_0)$（$t_0 = 5$）更新其信念；每个新出生的世代群组都以先验信念 $\beta_0 = 10$ 起步；市场预期是所有存活世代群组信念的平均值。

1. 模拟 50000 期，并将前 1000 期之后的平均价格与 $\beta^\star = a/(1 - b)$ 进行比较。为什么该经济永远达不到理性预期均衡？
1. 一位计量经济学家知道 $b$、$\beta_0$ 和世代群组结构，并重新估计 $a$。当真实值为 $t_0 = 5$ 时，计算 $t_0 = 10$ 的罚分，取 $T = 500, 5000, 50000$，并对 8 个样本取平均。对一个寿命无限、增益为 $1/(t + 1 + t_0)$ 的单一学习者、相同的 $\beta_0$ 和相同的 $t_0$ 重复相同操作。
1. 解释两者的差异，并将其与后记相联系。

```{exercise-end}
```

```{solution-start} pbr_ex3
:class: dropdown
```

我们用一个按年龄排序的数组来承载各世代群组的信念；每期最老的世代群组消亡，一个新生的世代群组以先验信念进入。

```{code-cell} ipython3
L = 40

def cohort_beliefs(β0, t0_c, p):
    "Average belief of the living cohorts implied by an observed price path."
    gains = 1 / (np.arange(L) + 1 + t0_c)
    B = np.full(L, β0)
    β_bar = np.empty(len(p))
    for t in range(len(p)):
        β_bar[t] = B.mean()
        B = B + gains * (p[t] - B)
        B[1:] = B[:-1].copy()                   # everyone ages; the oldest dies
        B[0] = β0                               # a newborn enters at the prior
    return β_bar

def cohort_economy(β0, t0_c, u):
    "Bray's market when the expectation is the average over overlapping cohorts."
    gains = 1 / (np.arange(L) + 1 + t0_c)
    B = np.full(L, β0)
    p = np.empty(len(u))
    for t in range(len(u)):
        p[t] = a + b * B.mean() + u[t]
        B = B + gains * (p[t] - B)
        B[1:] = B[:-1].copy()
        B[0] = β0
    return p

def single_beliefs(β0, t0_c, p):
    "A single infinitely lived learner with gain 1/(t + 1 + t0)."
    β = np.empty(len(p))
    β[0] = β0
    for t in range(1, len(p)):
        β[t] = β[t - 1] + (p[t - 1] - β[t - 1]) / (t + t0_c)
    return β

def single_economy(β0, t0_c, u):
    p = np.empty(len(u))
    β = β0
    for t in range(len(u)):
        p[t] = a + b * β + u[t]
        β = β + (p[t] - β) / (t + 1 + t0_c)
    return p

def penalty_a_free(p, β_wrong, β_true, Ts):
    "Excess SSR (intercept re-estimated) on the first T observations, for each T."
    r_w, r_t = p - b * β_wrong, p - b * β_true
    return np.array([ssr_a_free(r_w[:T]) - ssr_a_free(r_t[:T]) for T in Ts])

Ts = (500, 5_000, 20_000, 50_000)
pen_cohort, pen_single, mean_price = [], [], []
for seed in range(8):
    u = np.random.default_rng(seed).standard_normal(Ts[-1])
    p = cohort_economy(10.0, 5.0, u)
    mean_price.append(p[1_000:].mean())
    pen_cohort.append(penalty_a_free(p, cohort_beliefs(10.0, 10.0, p),
                                     cohort_beliefs(10.0, 5.0, p), Ts))
    p = single_economy(10.0, 5.0, u)
    pen_single.append(penalty_a_free(p, single_beliefs(10.0, 10.0, p),
                                     single_beliefs(10.0, 5.0, p), Ts))

print(f"cohort economy: average price {np.mean(mean_price):.2f}, β* = {β_star:.2f}")
print("penalty for t0 = 10, at T = 500, 5000, 20000, 50000")
print("   cohorts:      ", np.round(np.mean(pen_cohort, axis=0), 2))
print("   single agent: ", np.round(np.mean(pen_single, axis=0), 2))
```

世代群组经济的平均价格远低于 $\beta^\star$。

每一期都会有一个相信 $\beta_0 = 10$ 的新生世代群组到来，而年轻人对自己短暂的经历赋予很高的权重，因此平均信念被永久性地拉向先验：在总体层面，这种过渡状态永远不会结束。

对单一学习者而言，错误的 $t_0$ 所带来的罚分是有界的，因为 $t_0$ 仅在信念还“年轻”时才起作用：重建的信念为 $(t_0 \beta_0 + \sum_{s<t} p_s)/(t + t_0)$，因此 $t_0$ 对它的影响会以 $1/t$ 的速度消失。

在世代群组经济中，罚分持续增长，在超过几千个观测值之后大致与 $T$ 成比例，即使重新估计 $a$ 吸收了均值中的永久性偏移也是如此。

信息来自动态过程：$t_0$ 决定了年轻人对近期价格的反应强度，而总是存在正在做出反应的年轻世代群组。

这正是 {cite:t}`MalmendierNagel2016` 利用不同世代群组预期的调查数据所采用的识别路径：在一个世代交叠经济中，对单个代理人而言是过渡性的参数，会变成平稳分布的参数。

```{solution-end}
```

```{exercise-start}
:label: pbr_ex4
```

后记中的恒定增益示例表明，增益能够以通常的速率被识别，但每个观测值所携带的信息很弱。

本练习询问信息量如何依赖于增益的大小。

使用布雷的市场，取 $a = 5$，$b = 0.7$，初始信念为 $\beta^\star$，恒定增益为 $g$。

1. 对 $g \in \{0.005, 0.01, 0.02, 0.05, 0.1, 0.2\}$，在重新估计 $a$ 和 $b$ 的情况下，计算在 $T = 20000$ 时错误增益 $1.6 g$ 的罚分，对 20 个样本取平均，并除以 $T$。每个观测值所含信息量如何依赖于 $g$？
1. 在 $T = 200$（大致相当于季度数据中战后记录的长度）时，有多大比例的样本罚分超过 3.84？
1. 这些结果对 $1/t$ 学习者意味着什么？对于 {cite:t}`SargentWilliamsZha2006` 估计出的较大漂移协方差的争论又意味着什么？

```{exercise-end}
```

```{solution-start} pbr_ex4
:class: dropdown
```

我们重复使用后记中的 `simulate_constant_gain`、`belief_path_cg` 和 `ssr_ab_free`。

```{code-cell} ipython3
print(f"{'g':>6}   {'penalty / T, T = 20000':>22}   {'share > 3.84, T = 20000':>23}   "
      f"{'share > 3.84, T = 200':>21}")
for g in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    results = {}
    for T in (200, 20_000):
        pens = []
        for seed in range(20):
            u = np.random.default_rng(seed).standard_normal(T)
            p, β = simulate_constant_gain(β_star, g, u)
            pens.append(ssr_ab_free(p, belief_path_cg(β_star, 1.6 * g, p))
                        - ssr_ab_free(p, β))
        results[T] = np.array(pens)
    big = results[20_000]
    print(f"{g:>6}   {big.mean() / 20_000:22.5f}   {np.mean(big > 3.84):23.0%}   "
          f"{np.mean(results[200] > 3.84):21.0%}")
```

每个观测值所含信息随着增益的增大而增加，大致与之成比例，甚至略快一些。

用于识别的变异来自信念的运动，对于较小的增益，信念围绕 $\beta^\star$ 的平稳方差大致与增益成比例。

在拥有 20000 个观测值的情况下，最大增益的错误值在每个样本中都会被拒绝，而最小增益的错误值只有四分之一的样本中被拒绝；而在只有 200 个观测值的情况下，数据对任何一个增益几乎都没有提供什么信息。

当 $g \to 0$ 时，恒定增益学习者趋近于 $1/t$ 学习者，其信念运动逐渐消失，其学习参数无法被一致地估计：这两种情形是一个连续统的两端。

在像战后记录那样短的样本中，只有当信念大幅波动时，数据才能提供关于学习的信息。

这正是理解 {cite:t}`SargentWilliamsZha2006` 所估计出的较大漂移协方差的一种方式，也是对隐含信念波动过大这一异议的一种解读。

诸如 {cite:t}`CarboniEllison2009` 所使用的美联储绿皮书预测这样的关于信念的直接证据，提供了那种长度的结果数据所无法提供的信息。

```{solution-end}
```
