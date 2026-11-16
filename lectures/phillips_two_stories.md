---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 美国通货膨胀的兴衰
  headings:
    Overview: 概述
    Facts: 事实
    The Phillips curve in the data: 数据中的菲利普斯曲线
    Two interpretations: 两种解释
    Two interpretations::The triumph of natural-rate theory: 自然率理论的胜利
    Two interpretations::The vindication of econometric policy evaluation: 计量经济学政策评估的平反
    Ignoring the Lucas Critique: 忽略卢卡斯批判
    Ignoring the Lucas Critique::The Critique: 批判
    Ignoring the Lucas Critique::The appeal to drifting coefficients: 对漂移系数的诉求
    Ignoring the Lucas Critique::Parameter drift as a point of departure: 以参数漂移为出发点
    'A premature summary: triumph or vindication?': 一个尚早的总结：胜利还是平反？
    'A premature summary: triumph or vindication?::The road ahead': 前路展望
    'A premature summary: triumph or vindication?::The induction hypothesis: villain and hero': 归纳假说：反派与英雄
    'A premature summary: triumph or vindication?::The reservation': 保留意见
    Data patterns after 1999: 1999 年之后的数据模式
    Data patterns after 1999::The rise and fall, extended: 延伸后的兴与衰
    Data patterns after 1999::Unemployment and inflation to the present: 延续至今的失业率与通货膨胀
    Data patterns after 1999::The Phillips curve across three eras: 三个时代中的菲利普斯曲线
    Data patterns after 1999::What the new data mean for the two stories: 新数据对这两个故事意味着什么
    A note on notation: 关于符号的说明
    Exercises: 练习
---

(phillips_two_stories)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 美国通货膨胀的兴衰

```{index} single: Phillips Curve; Rise and Fall of U.S. Inflation
```

```{contents} Contents
:depth: 2
```

## 概述

{doc}`prospects_bounded_rationality` 结束了有限理性系列讲座，并留下了一个不断重新估计其菲利普斯曲线的政府：西姆斯和钟（Sims and Chung）笔下那位政策制定者，如今已经成长起来。

本讲开启了一个新的系列，要求这个政府来解释美国通货膨胀的历史。

该系列基于托马斯·萨金特《美国通货膨胀的征服》（*The Conquest of American Inflation* {cite}`Sargent1999`），而这第一讲综合了该书第一章和第二章的内容，并预告了其结论章的内容。

从有限理性系列而来的读者，已经接触过该模型的一个精简版本。

{ref}`olg_phillips` 研究了一个采用 {doc}`phillips_escaping_nash` 中静态模型校准、具有恒定增益的政府，{ref}`br_one_recursion` 阐述了我们用来分析此类政府的随机逼近与 E-稳定性工具，而 {ref}`pbr_compliment` 则讲述了计量经济学家们是如何开始对其进行估计的。

这个系列提出了一个关于战后美国宏观经济史的问题：

> 如果我们认为通货膨胀是由美联储控制的，那么我们该如何解释美国通货膨胀在 20 世纪 70 年代的上升，以及在 20 世纪 80 年代初保罗·沃尔克治下的骤然下降？

这篇文章评估了两种解释，两者都基于政策制定者对菲利普斯曲线的*信念*。

在这两个故事中，美联储都是通过经验与*先验*推理相结合的方式，学到了自然失业率理论。

两个故事的区别在于该理论是如何被采纳的：

* **自然率理论的胜利。** 学术经济学家发现了自然率假说，指出任何通货膨胀与失业之间的权衡都是暂时的，并最终说服政策制定者追求低通货膨胀。
* **计量经济学政策评估的平反。** 政策制定者从未放弃罗伯特·卢卡斯在其著名批判中所抨击的方法。
  - 他们反复重新估计菲利普斯曲线并用它来选择目标，正是*数据本身*——一条不断向不利方向漂移的经验菲利普斯曲线——引导他们走向更低的通货膨胀。

本讲座介绍了支持这两个故事的事实，勾勒出这两种解释，并回顾了第二章既援引又修正的卢卡斯批判。

随后，本讲在 {ref}`ts_roadmap` 中预告了其余十二讲如何建立相应的模型，并以把该书的数据延续至今作为结尾。

让我们从一些导入开始：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.bk_filter import bkfilter
```

```{note}
接下来两节中的图表复现了 {cite:t}`Sargent1999` 第一章中的图表，这些图表所用的数据取自 20 世纪 90 年代末。

我们直接从 [FRED](https://fred.stlouisfed.org/) 下载相应的原始数据序列，并将关注范围限定在与原书相同的历史区间内。

之后的 {ref}`phillips_after_1999` 一节会将其中最具启发性的图表延伸到当下，并探讨这额外的四分之一个世纪的数据对这两个故事意味着什么。
```

## 事实

我们从整篇文章要解释的这一个事实开始：二战以来美国通货膨胀呈现的驼峰形走势。

关于价格水平更长时间跨度、跨越数个世纪和多个国家的考察，请参阅 {doc}`intro:inflation_history`。

我们按照本书的方法来衡量通货膨胀，即用消费者价格指数（所有项目，未经季节调整）月度变化的年化值来衡量，并用 13 个月的居中移动平均对其进行平滑，以去除季节性和高频噪声。

下面的函数直接从 FRED 读取数据序列；我们用它来获取价格指数，以及下一节中需要用到的失业率数据。

```{code-cell} ipython3
def fred_series(series_id):
    "Download a monthly series from FRED as a pandas Series."
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    frame = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
    return frame.iloc[:, 0].rename(series_id)

cpi_all = fred_series('CPIAUCNS')      # CPI, all items, not seasonally adjusted
u_all = fred_series('LNS14000028')     # unemployment rate, white men 20 and over

book_window = slice('1948', '1999-01')
cpi = cpi_all[book_window]

# 年化月度通货膨胀率，再取 13 个月居中移动平均
inflation = 1200 * np.log(cpi).diff()
inflation_ma = inflation.rolling(13, center=True).mean()
```

图 1.1 绘制了本书所考察时间窗口内的平滑序列。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Monthly CPI inflation, 13-month centered moving average, 1948-1999
    name: fig-ts-inflation
---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(inflation_ma, lw=1.2)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('year')
ax.set_ylabel('inflation (percent, annualized)')
ax.set_title('Figure 1.1: Monthly inflation, CPI all items, '
             '13-month centered moving average')
plt.show()
```

在 20 世纪 50 年代末和 60 年代初，通货膨胀水平较低，随后在 70 年代急剧上升，接着又在 80 年代初随着沃尔克的稳定化政策而骤然下降。

任何将通货膨胀视为受美联储控制的解释，都必须说明这一上升与下降的过程。

## 数据中的菲利普斯曲线

尽管菲利普斯曲线在一些学术界和政策制定圈中名誉扫地，但它在美国数据中依然顽强存在，简单的方法就能将其检测出来。

为了从数据中提炼出菲利普斯曲线，我们沿用原书的两种做法。

第一，我们使用单一人口群体——20 岁及以上的白人男性——的失业率，而非整体失业率，因为整体失业率会受到人口结构缓慢变化的干扰。

第二，我们关注*商业周期*频率，滤除缓慢变化的成分，以便肉眼能够察觉通货膨胀与失业率之间的反向关系。

失业率序列始于 1954 年 1 月，因此联合样本也从该月开始。

```{code-cell} ipython3
u = u_all[book_window]

data = pd.concat([inflation.rename('inflation'),
                  u.rename('unemployment')], axis=1).dropna()
data.head()
```

图 1.2 将这两个序列绘制在一起，其中通货膨胀像图 1.1 中那样经过了平滑处理。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Monthly unemployment (white men 20+) and inflation (13-month centered moving average)
    name: fig-ts-raw-series
---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(inflation_ma[data.index[0]:], 'C0', lw=1,
        label='inflation (CPI, 13-month centered moving average)')
ax.plot(u, 'C1:', lw=1.2, label='unemployment (white men 20+)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.set_title('Figure 1.2: Monthly unemployment and inflation rates')
ax.legend()
plt.show()
```

为了分离出商业周期层面的关系，我们对月度序列采用巴克斯特（Baxter）和金（King）提出的有限滞后带通滤波器 {cite}`BaxterKing1999`。

按照原书的做法，我们保留周期在 24 到 84 个月之间的波动，并使用 84 个月的超前-滞后截断。

由于该滤波器需要在每个日期两侧各取 84 个月的数据，因此会舍去样本最初和最后各七年的数据，所以滤波后的序列从 1961 年开始。

```{code-cell} ipython3
# Baxter-King bandpass: periods between 24 and 84 months, truncation 84
bk = bkfilter(data, low=24, high=84, K=84)
bk.columns = ['inflation_cycle', 'unemployment_cycle']
print(f"filtered sample: {bk.index[0]:%Y-%m} to {bk.index[-1]:%Y-%m}")
```

图 1.3 绘制了滤波后的成分。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Business-cycle components of inflation and unemployment, Baxter-King bandpass filter
    name: fig-ts-bandpass
---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(bk.index, bk['inflation_cycle'], 'C0', lw=1, label='inflation')
ax.plot(bk.index, bk['unemployment_cycle'], 'C1:', lw=1.2,
        label='unemployment')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('year')
ax.set_ylabel('deviation from trend (percent)')
ax.set_title('Figure 1.3: Business-cycle components '
             '(Baxter-King bandpass filter)')
ax.legend()
plt.show()
```

经过滤波的两个成分往往朝相反的方向变动：这是一条商业周期意义上的菲利普斯曲线。

我们可以在最令我们感兴趣的子时期——1960 年至 1982 年——的散点图中更直观地看到这种权衡关系。

与原书一致，图 1.4 将平滑后的通货膨胀与失业率对照绘制，图 1.5 则展示了商业周期成分，两图均以失业率为横轴。

在图 1.5 中，一条浅淡的线将各个月份依次相连。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Inflation against unemployment, 1960-1982: smoothed series and business-cycle components"
    name: fig-ts-scatter-6082
---
sub = slice('1960', '1982')
cycle_sub = bk.loc[sub]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(u[sub], inflation_ma[sub], s=8, alpha=0.6)
axes[0].set_xlabel('unemployment (white men 20+)')
axes[0].set_ylabel('inflation (13-month centered moving average)')
axes[0].set_title('Figure 1.4: Smoothed series, 1960-1982')

axes[1].plot(cycle_sub['unemployment_cycle'], cycle_sub['inflation_cycle'],
             lw=0.5, alpha=0.4)
axes[1].scatter(cycle_sub['unemployment_cycle'],
                cycle_sub['inflation_cycle'], s=8, alpha=0.6)
axes[1].set_xlabel('unemployment (business-cycle component)')
axes[1].set_ylabel('inflation (business-cycle component)')
axes[1].set_title('Figure 1.5: Business-cycle components, '
                  f'{cycle_sub.index[0].year}-{cycle_sub.index[-1].year}')

plt.tight_layout()
plt.show()
```

聚焦于商业周期成分使得表面上的菲利普斯曲线更加清晰。

图 1.5 揭示了**菲利普斯回路**：按月份依次追踪，商业周期成分描绘出的是*顺时针*方向的回路，而非单一的稳定曲线。

顺时针回路意味着通货膨胀*滞后于*失业率。

当失业率下降时，通货膨胀上升迟缓，因此路径沿回路底部向左运行；而当失业率再度上升时，通货膨胀下降迟缓，因此路径沿回路顶部向右运行。

在这些频率上，当通货膨胀与大约提前六个月的失业率配对时，二者之间的负相关性最强（{ref}`ts_ex1` 要求你验证这一点）。

通货膨胀预期的缓慢调整——这正是自然率理论所强调的、位于故事核心的要素——是造成这种滞后的一个可能来源。

```{note}
原书通过选取单一失业率序列来调整人口结构变化的影响，并在第九章中利用这些相同的数据估计了一个适应性的政策制定模型（ {doc}`phillips_policy_evaluation` ）。

若采用更广义的失业定义，就会引入额外的低频人口结构成分，人们可能会用单位根过程来对其建模。

而本文则从另一个来源——即摆脱布雷顿森林体系约束后货币当局*漂移的信念*——将单位根引入通货膨胀-失业率过程之中。
```

## 两种解释

这两个故事都从相同的初始条件出发——即政策制定者在 1960 年前后所继承的通货膨胀与失业史，以及当时的预期状态——而且都假设数据符合自然率假说，无论政策制定者当时是否意识到这一点。

### 自然率理论的胜利

对金本位制、继而对布雷顿森林体系的坚持，使美国得以维持低通货膨胀和低通货膨胀预期。

1960 年，保罗·萨缪尔森和罗伯特·索洛 {cite}`SamuelsonSolow1960` 在美国数据中发现了一条菲利普斯曲线，并宣称它是*可利用的*——即政策可以通过提高通货膨胀来降低失业率。

这是他们在第 192 页得出的教训；但在第 193 页，他们自己却提醒道："不过，认为我们图 2 所示的可获得价格与失业表现之间的权衡关系在更长的时期内会保持同样的形态，这种想法是错误的。"

在不到十年的时间里，这一建议被广泛接受并付诸实施。

令所有人沮丧的是，菲利普斯曲线随后发生了不利的移动：通货膨胀上升了，但失业率平均而言并未下降。

与此同时，埃德蒙·费尔普斯（Edmund Phelps） {cite}`Phelps1967`、米尔顿·弗里德曼 {cite}`friedman1968role` 以及罗伯特·卢卡斯 {cite}`Lucas1972` 提出并完善了自然失业率的概念，该概念将通货膨胀预期置于确定菲利普斯曲线位置的核心地位。

自然率理论只允许*暂时性*的权衡取舍，并解释了这种不利的转移；其理性预期版本则意味着政策制定者应当忽视这种暂时的权衡，只应努力实现低通货膨胀。

在这个故事里，这些思想从学术界扩散到政策制定者手中，并最终带来了 20 世纪 80 年代和 90 年代较低的通货膨胀。

历史的走向由政策制定者的信念——有些是错误的，有些是正确的——以及这些信念所激发的行动共同塑造。

随后，卢卡斯 {cite}`lucas1976econometric` 将萨缪尔森和索洛那种从统计性菲利普斯曲线推导政策的方法，作为错误方法论的一个例证。

他的论证很快便将这种方法从学术研究中心驱逐出去，但正如该书所言，"这种方法却在美联储体系内得以延续并兴旺发展"——这一事实正启发了第二个故事。

### 计量经济学政策评估的平反

另一种解释则将沃尔克的胜利部分归功于卢卡斯所质疑的那些计量经济学和政策制定程序本身的*成功*。

这一说法此前已由克里斯托弗·西姆斯（Christopher Sims） {cite}`Sims1988` 和希泰克·钟（Heetaik Chung） {cite}`Chung1990` 讲述过。

政策制定者接受了萨缪尔森-索洛菲利普斯曲线，将其视为一种可利用的权衡关系，并采用了他们从数据中学习并据此推导政策的方法。

他们反复重新估计一个分布滞后的菲利普斯曲线，并用它来重新设定目标通货膨胀率与失业率的组合。

如果机械地解读这一过程——不将预期识别为隐藏的状态变量——那么这条不断向不利方向移动的经验菲利普斯曲线最终引导政策制定者走向了更低的通货膨胀。

这个故事是用一种与理性预期相去甚微的*适应性*政策理论来讲述的。

它将该系列后续发展的一系列思想串联起来：漂移系数、自我确认均衡、最小二乘法及其他递归学习算法、最小二乘学习者向自我确认均衡的收敛，以及沿着从这些均衡中*逃逸的路径*所展现的周期性动态。

这一关键思想取自 {cite:t}`Sims1988`，即适应性模型使政府能够从其过去利用菲利普斯曲线的尝试中*学习*，并最终发现某种版本的自然率假说，指导其降低通货膨胀。

在西姆斯这一洞见的基础上，该书进一步阐述了适应性系统的*逃逸路径动态*如何能够自发地引发体制转变。

在这个故事中，通货膨胀政策既不是在真空中制定的，也不是作为一种自然实验而出现的；它是从一个适应性过程中逐渐生成的。

尽管该平反故事在一定程度上偏离了理性预期，但它对政府政策施加的限制*多于*胜利故事——后者偶尔会调整描述政府行为的自由参数，以使理论与事实相符。

## 忽略卢卡斯批判

该书第二章直面对平反故事的一个显而易见的质疑：卢卡斯批判难道不正是禁止使用那种机械的、可利用的菲利普斯曲线吗？而这正是平反故事的立论基础。

本文重新启用了卢卡斯曾坚决批判的计量经济学与政策评估程序 {cite}`lucas1976econometric`，并强调了其批判中一个被忽视的方面：**漂移系数**。

### 批判

一个计量经济学模型是一组随机差分方程的集合，其中一些描述了私人主体的决策规则。

丁伯根-泰尔（Tinbergen-Theil）传统中的计量经济学政策评估，是在政府针对某一目标函数优化自身规则的同时，将这些私人决策规则视为*固定不变*的。

卢卡斯指出，如果私人主体求解跨期最优化问题，那么他们的决策规则将*取决于*政府的规则。

由于忽略了这种依赖关系，丁伯根-泰尔方法错误地将政府对结果的偏好，转译为对决策规则的排序，从而给出不可靠的政策建议。

理性预期均衡将这种依赖关系内置其中，它要求私人主体所感知的运动规律，与其自身决策所共同生成的实际运动规律保持一致（ {doc}`rational_expectations` ）。

{doc}`ls_learning` 展示了通过最小二乘法估计其感知运动规律的主体如何能够找到这样一个不动点，而下文所述的自我确认均衡正是同一类不动点。

### 对漂移系数的诉求

卢卡斯承认凯恩斯主义模型在预测方面表现出色，但他认为，良好的预测表现并不能证明丁伯根-泰尔方法所假设的那种*干预下的不变性*成立。

他强调，预测者们习惯性地调整关键方程中的常数项，并将这些调整解读为一种与库利（Cooley）和普雷斯科特（Prescott） {cite}`CooleyPrescott1973` 思路一致的适应性系数模型。

所估计关系在跨期上的不稳定性——即系数漂移，据推测即便在稳定的政策操作规则下也可能存在——削弱了将其视为对政策规则系统性变化保持不变的这种做法。

但卢卡斯并未将这种漂移解读为*印证*了他的理论要点，即私人决策规则会随政府规则而变化。

他所举的三个例子比较的是由不同政府规则所索引的平稳环境，对漂移系数只字未提；该书指出，许多早期读者误以为他对漂移系数的提及是在支持他的基本论点。

卢卡斯并未对这种漂移作出*解释*，而无论是批判之后建立的宏观经济理论，还是理性预期计量经济学，都没有对此作出说明：二者都聚焦于具有时不变转移函数的环境。

### 以参数漂移为出发点

本文以参数漂移为出发点，将其视为一个*确凿的证据*——这正是表明政府关于经济的信念、进而其通货膨胀政策，随时间演化的关键证据。

它由两个部分构建出一个模型：

1. 一个关于政府决策的丁伯根-泰尔理论——即 {doc}`phillips_adaptive` 中的费尔普斯问题；以及
2. 一个针对政府的漂移系数计量经济学程序，其中包含卢卡斯所描述的那种常数项调整。

它通过假设自然率假说的理性预期版本成立，来约束这项分析，并将其他解释搁置一旁："我不考察异常的供给冲击以及其他对二战后美国通货膨胀的合理解释。"

如果环境的基本面是时不变的，那么计量经济学中一致性证明背后的力量，会使政府的估计与其决策规则趋于收敛 {cite}`MarcetSargent1989jet`。

在极限情形下，可能出现一个**自我确认均衡**（ {doc}`phillips_self_confirming` ），其中政府的估计被其自身行为和私人部门对其行为的反应共同强化。

在自我确认均衡中，卢卡斯批判的部分效力便消失了。

尽管政府的不变性假设是错误的，但它并未在结果中受挫，因为这些结果在统计上与它的信念是一致的。

自我确认均衡同样也对政府的模型及其行为施加了约束：由于政府的规则是从其拟合模型中推导出来的，卢卡斯模型中用以描述政府政策的自由参数便不复存在。

因此，自我确认均衡是一种理性预期均衡，但其自由参数比 {cite:t}`Lucas1972` 和 {cite:t}`lucas1976econometric` 的模型*更少*，而所缺失的参数，恰恰正是我们用来表现制度变迁时所需要调整的那些参数。

{cite:t}`Sims1982` 在论证制度变迁之所以罕见，是因为现行政策近乎最优时，便隐含地诉诸了一种自我确认均衡。

要容纳制度变迁和漂移系数，就必须*抵制*向自我确认均衡的收敛。

本文通过用一种类似库利和普雷斯科特所用的、赋予近期数据更高权重的恒定增益、适应性系数算法，取代政府的最小二乘估计量，从而阻止这种收敛——这使政府具备了一种“环境可能不稳定”的疑虑。

这削弱了向自我确认均衡的牵引力，并维持了沿着某条逃逸路径的动态，制度变迁正是沿此路径发生的。

具有讽刺意味的是，正如我们将在 {doc}`phillips_learning` 中看到的那样，*违反*卢卡斯批判的程序，反而可能带来比遵循它的程序更好的结果。

## 一个尚早的总结：胜利还是平反？

本系列后续讲座将建立各种模型。

在我们开始之前，值得先预览一下这些模型将引向何方，以及它们留下的尚未解决的张力——这是对整个旅程的一个提前总结，改编自 {cite:t}`Sargent1999` 的结论章节。

一切都可以围绕两个*基准*模型来组织。

第一个模型源自 {cite:t}`Phelps1967`，公众以*适应性*方式形成预期，而政府在将公众的规则视为既定的前提下*最优地*选择政策。

第二个模型是理性预期自然率模型，其中公众是*理性的*，而政府的政策则被视为*外生且任意的*。

卢卡斯建议用第二个基准取代第一个。

正视这两个故事，促使我们提出在这两极之间做出各种*折衷*的模型——而后续讲座正是这些折衷方案。

(ts_roadmap)=
### 前路展望

我们首先在*双方*都施加理性的假设。

* {doc}`phillips_credibility`（第 3 章）研究 {cite:t}`KydlandPrescott1977` 的单期模型，其时间一致（纳什）结果给出了一个悲观的预测：高通货膨胀，却一无所获。
* {doc}`phillips_credible_policies`（第 4 章）将该经济无限重复下去。它用 {cite:t}`APS1990` 的递归方法计算出可维持数值的整个集合，发现可维持的结果如此之多，以至于该理论以*不可知论*取代了悲观论。

这种不可知论是我们在宣称自然率理论取得胜利之前应当迟疑的第一个理由。

遵循西姆斯和钟的思路，随后我们从卢卡斯批判处折返，重新从费尔普斯基准出发。

* {doc}`phillips_adaptive`（第 5 章）针对一个任意设定的适应性预期菲利普斯曲线求解费尔普斯问题，并说明了为何*归纳假说*至关重要。

接下来我们作一处改动：政府对私人部门的模型不再是任意的——它是*拟合历史数据*得到的。

改变这一拟合问题的各种细节，便产生了接下来的五篇讲座。

* {doc}`phillips_misspecified`（第 6 章）在布雷的市场中引入了具有最优但错误设定的预测函数的均衡。
* {doc}`phillips_self_confirming`（第 7 章）要求政府的菲利普斯曲线是自我确认的，发现自我确认均衡会重现纳什结果或更糟的结果，最后以一个均衡作结，在该均衡中一个细微错误设定的公众预测比纳什结果表现更好。
* {doc}`phillips_learning`（第 8 章）让政府成为一个实时计量经济学家，一个 {cite:t}`Kreps1998` 意义上的"预期效用"学习者，并发现在恒定增益下出现了自发的、类似沃尔克式的稳定化。
* {doc}`phillips_escaping_nash` 遵循 {cite:t}`ChoWilliamsSargent2002` 的方法对这些逃逸路径进行了解析刻画。
* {doc}`phillips_priors` 遵循 {cite:t}`SargentWilliams2005` 的方法，说明政府关于系数漂移的先验是如何塑造收敛、循环与逃逸的。

接下来的三篇讲座将这些模型用于实际数据。

* {doc}`phillips_policy_evaluation`（第 9 章）遵循钟的方法，用最大似然法在美国数据上估计适应性模型，并探讨这些估计结果是否为计量经济学政策评估提供了支持。
* {doc}`phillips_drifts_volatilities` 遵循 {cite:t}`CogleySargent2005` 的方法，将一个具有漂移系数和随机波动率的向量自回归模型拟合到截至 2000 年的数据，并探讨大通胀究竟是政策失误还是运气不佳所致。
* {doc}`phillips_drifts_volatilities_2` 将该模型更新至 2000 年以来的数据。

最后，{doc}`phillips_lost_conquest` 遵循 {cite:t}`SargentWilliams2025` 的方法，将本系列的工具（包括一个采用预期效用的美联储）运用于 2020 年代的通货膨胀以及美联储对此的迟缓反应，从而回到本讲座对数据的考察所留下的那个问题：这场胜利是否已经失而复得？

符号在各讲之间有所变化，因为每一讲都遵循其各自的原始文献；{ref}`phillips_notation` 记录了这些符号的对应关系以及各讲所使用的增益。

这些适应性模型是对理性预期一种*有节制的*退让，而非对其的彻底抛弃。

除了一个增益参数之外，它们不包含控制预期的自由参数，并且在每一期，它们都施加与理性预期模型相同的跨方程约束。

而且，由于自我确认均衡是其*均值动态*的吸引子，在平静的条件下，若增益最终以 $1/t$ 的速度衰减，它们便会收敛回自我确认均衡，从而在均衡路径上收敛回理性预期，满足了 {cite:t}`Kreps1998` 所提出的一个诉求。

这种收敛性对最小二乘法成立；{doc}`phillips_priors` 展示了在其他先验下自我确认均衡可能是不稳定的。

但为了与数据相符，与西姆斯一样，我们真正感兴趣的是适应性所带来的动态。

这些动态有两种类型。

{cite:t}`Sims1988` 和 {cite:t}`Chung1990` 让他们的政府使用了增益不会消失的、针对漂移系数的 {doc}`卡尔曼滤波器 <kalman>`；西姆斯的政府拟合了一条静态菲利普斯曲线，而钟的政府则拟合了一个分布滞后模型。

在他们的模拟中，政府会从纳什结果中逃逸一次，此后便一直停留在接近拉姆齐最优的结果附近，看上去是无限期地持续下去。

{cite:t}`Sargent1999` 和 {cite:t}`ChoWilliamsSargent2002` 中的政府则使用了恒定增益的递归最小二乘法，其逃逸是*反复出现*的：每一次逃逸之后都会出现向纳什结果的回归，从而形成 {ref}`olg_phillips` 中的锯齿状曲线。

这两类历史为何会有所不同，一直是个谜，直到 {cite:t}`SargentWilliams2005` 将其根源追溯到政府的先验：即其滤波器所假设的漂移协方差与噪声方差。

{doc}`phillips_priors` 详细梳理了他们的论述。

{doc}`phillips_learning` 与 {doc}`phillips_escaping_nash` 中的学习型政府正是萨金特笔下的政府：由于怀疑菲利普斯曲线容易发生漂移，它们使用了一种恒定增益算法，这在系数发生漂移时是明智的选择。

其成果颇为惊人：适应性模型产生了通货膨胀的骤然*稳定化*，这一结果违背了均值动态所指向的那种较差的自我确认结果。

这些制度转变既不是源于政府程序的任何改变，也不是源于巨大的冲击，而是源于*政府自身的计量经济学所引发的信念变化*——即在近似模型空间中逃逸路径的数学机制。

### 归纳假说：反派与英雄

这个故事的核心是**归纳假说**——即预期方程中滞后通货膨胀项的权重之和为一的这一限制，从而使得一个永久性更高的通货膨胀率最终会被完全预期到。

它几乎未经任何评论便被内嵌于 {cite:t}`Friedman1957` 和 {cite:t}`Cagan` 的适应性预期假说之中，并且是索洛与托宾对自然率假说所做早期检验的基础 {cite}`Solow1968,Tobin1968`。

在卢卡斯的批判中，它被塑造为*反派*角色——一种理性预期并不蕴含的天真限制——而在适应性模型中，归纳假说却又重新以*英雄*的身份出现：正是激活这一假说，才使得政府的费尔普斯问题呼唤出接近拉姆齐最优、低通货膨胀的政策。

在 {doc}`phillips_learning` 的分布滞后模型中，逃逸路径恰恰就是政府所估计的模型——通过用单位根来近似一个常数——逐渐误入相信归纳假说的那些路径。

正是与此类近似问题的角力，以及同时存在若干个模型的情形，促使 {cite:t}`Sims1980` 将有限理性称为一片*荒野*，与理性预期那个整洁的单一模型世界截然不同。

### 保留意见

那么，究竟哪个故事才是正确的——胜利还是平反？

这场争论并非理性预期与其替代方案之间的对立，因为*两个*故事都是有选择性地采用或放弃理性预期。

而平反的故事，无论它多么契合数据，都只是一项*实证*经济学的练习，而非*规范*经济学的论断。

人们很容易把书中长期出现的接近拉姆齐最优的通货膨胀，解读为对产生这种结果的适应性政策制定程序的一种肯定——但这种诱惑应当被抵制。

因为正是那使得稳定化得以实现的同一套适应性动态，也同样使这种稳定化只是暂时的：一旦信念接近归纳假说，均值动态便指向*远离*它的方向，回到费尔普斯问题建议重新推高通货膨胀的那个区域。

均值动态将系统推回自我确认均衡可能需要很长时间，但这终究是不可避免的。

模拟结果中既包含了看起来像保罗·沃尔克的长时段，也包含了看起来像亚瑟·伯恩斯的长时段。

{cite:t}`KydlandPrescott1977` 之后的理论研究，例如 {cite:t}`Rogoff1985`，坚持认为持久的低通货膨胀必须依赖于*承诺机制*，使货币当局无法逐期相机抉择——而不能寄希望于一个装备了近似模型的适应性政府，能够碰巧最终学会大致做正确的事情。

因此，本书的结尾留下的是一种期望，而非定论：我们*希望*胜利的故事才是正确的那一个——即政策制定者已经学到了自然率假说的正确理性预期版本，并找到了使自己承诺于低通货膨胀的手段。

因为如果平反的故事更接近真相，那么把信念拉回自我确认均衡的那套均值动态，也始终等待着最终把通货膨胀重新推高。

自 1999 年以来的这四分之一个世纪的数据——一段长期而平静的*大缓和*，中间被 2021-2022 年的骤然飙升所打断——正是对这一期望的一次持续检验，接下来我们将转向对它的探讨。

(phillips_after_1999)=
## 1999 年之后的数据模式

{cite:t}`Sargent1999` 写于始于沃尔克的长期反通货膨胀过程结束之际。

如今我们又有了四分之一个世纪的新数据。

本节将把书中最具启发性的图表延伸到当下，并探讨这些新的观测数据对这两个故事意味着什么。

我们使用完整的美联储经济数据序列，直至最新可获得的月份，并首先检查每个序列在样本内部是否存在缺失月份。

```{code-cell} ipython3
def missing_months(series):
    "Months between the first and last observation that have no value."
    months = pd.date_range(series.first_valid_index(),
                           series.last_valid_index(), freq='MS')
    return months.difference(series.dropna().index)

for series in (cpi_all, u_all):
    gaps = [f'{m:%Y-%m}' for m in missing_months(series)]
    print(f"{series.name:12s} last observation "
          f"{series.last_valid_index():%Y-%m}, missing months: {gaps}")
```

2025 年末联邦政府的停摆导致这两个序列在 2025 年 10 月的观测数据均未被采集。

哪怕只缺失一个月的数据，也会使所有涉及该月份的 13 个月移动平均值和 12 个月变化率变为缺失值，从而在不知不觉中缩短了平滑后的序列。

因此，我们通过插值来填补孤立的缺失月份——对价格指数采用几何插值，对失业率采用线性插值，而对更长的缺口则不作处理。

```{code-cell} ipython3
def fill_isolated(series, log=False):
    "Interpolate single missing months; leave longer gaps missing."
    s = np.log(series) if log else series
    isolated = s.isna() & s.shift(1).notna() & s.shift(-1).notna()
    s = s.where(~isolated, s.interpolate(limit_area='inside'))
    return np.exp(s) if log else s

cpi_full = fill_isolated(cpi_all['1948':], log=True)
u_full = fill_isolated(u_all)
print("still missing:",
      [f'{m:%Y-%m}' for m in missing_months(cpi_full).union(missing_months(u_full))])

# 原书所用的度量方式（月度变化年化值，13 个月居中移动平均）
inflation_full = 1200 * np.log(cpi_full).diff()
inflation_ma_full = inflation_full.rolling(13, center=True).mean()

# 同比通货膨胀率：更为平滑，也是当今通常引用的度量方式
inflation_yoy = 100 * (cpi_full / cpi_full.shift(12) - 1)
print(f"13-month centered average available through "
      f"{inflation_ma_full.last_valid_index():%Y-%m}")
```

一个 13 个月居中移动平均值需要某一日期前后各六个月的数据，因此平滑后的序列会在最新观测值之前六个月处结束。

### 延伸后的兴与衰

图 1.1 展示了通货膨胀在 20 世纪 70 年代的上升以及在沃尔克治下的下降。

将其延伸至今，增添了原书所无法看到的三段历程：从 20 世纪 80 年代中期开始、通货膨胀低而稳定的*大缓和*时期；2008 年金融危机后，长达十年通货膨胀低于目标水平的时期，在 2008-09 年一度降至零以下，并在 2015 年再度短暂跌破零线；以及 2021-2022 年一次骤然飙升至 1981 年以来最高水平、随后又迅速回落的过程。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Inflation extended to the present, with the book's window shaded
    name: fig-ts-inflation-long
---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(inflation_ma_full, lw=1)
ax.axhline(0, color='k', lw=0.5)
ax.axvspan(pd.Timestamp('1948-01-01'), pd.Timestamp('1999-01-01'),
           color='C0', alpha=0.06, label="the book's window")
for date, y, txt in [('1951-01-01', 9.3, 'Korean\nWar'),
                     ('1970-01-01', 12.0, '1970s\nacceleration'),
                     ('1984-06-01', 0.5, 'Volcker'),
                     ('2009-07-01', -1.5, '2009\ndeflation scare'),
                     ('2022-06-01', 8.9, '2021-22\nsurge')]:
    ax.annotate(txt, (pd.Timestamp(date), y), ha='center', fontsize=9,
                color='C3')
ax.set_xlabel('year')
ax.set_ylabel('inflation (percent, annualized)')
ax.set_title('Inflation, CPI all items, 13-month centered moving average, '
             f'1948-{inflation_ma_full.last_valid_index().year}')
ax.legend(loc='upper right')
plt.show()
```

原书所要解释的那个驼峰，如今变成了*两个*驼峰中的第一个。

第二个驼峰把一次快速的加速与一次几乎同样迅速的反通货膨胀，压缩在大约三年之内。

与它最为接近的先例并非 20 世纪 70 年代，而是原书自身样本窗口之内、1950-1951 年朝鲜战争期间的那次飙升：通货膨胀在一年之内跃升了约八个百分点，又在随后一年内回落到接近 2% 的水平，其间并没有出现沃尔克治理之前那种拖延已久的加速过程。

### 延续至今的失业率与通货膨胀

图 1.2 绘制了战后时期这两个序列的对比图。

将其延伸后，展示了新数据中最引人注目的两大宏观经济事件：2020 年新冠疫情导致的失业率骤升——对于 20 岁及以上的白人男性而言，这是自该序列始于 1954 年以来的最高水平——以及随之而来的通货膨胀飙升。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Unemployment and inflation since 1990
    name: fig-ts-recent
---
recent = slice('1990', None)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(inflation_yoy[recent], 'C0', lw=1, label='inflation (CPI, year-over-year)')
ax.plot(u_full[recent], 'C1:', lw=1.2, label='unemployment (white men 20+)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('year')
ax.set_ylabel('percent')
ax.legend()
plt.show()
```

有两个特征十分突出。

从 20 世纪 90 年代中期到 2020 年，即使失业率大幅波动，通货膨胀却出奇地平静：失业率在 2000 年以及 2019 年再度降至 20 世纪 70 年代初以来的最低水平，并在 2008 年经济衰退中大致翻了两番。

而在新冠疫情导致的失业率骤升之后，失业率迅速回落，通货膨胀随之飙升。

失业率下降而通货膨胀上升，是沿着一条向下倾斜的菲利普斯曲线移动的表现，这正是需求冲击的特征标志。

但仅凭这一模式，尚无法区分这究竟是沿着一条陡峭的菲利普斯曲线所作的需求驱动型移动，还是供给冲击所致，二者的混合作用至今仍存在争论。

那么，通货膨胀究竟已回落了多少？

```{code-cell} ipython3
since_peak = inflation_yoy['2022-07':]
at_or_below_2 = inflation_yoy[inflation_yoy <= 2]
print(f"peak year-over-year inflation: {inflation_yoy['2021':'2023'].max():.1f}% "
      f"({inflation_yoy['2021':'2023'].idxmax():%Y-%m})")
print(f"lowest reading since then:     {since_peak.min():.1f}% "
      f"({since_peak.idxmin():%Y-%m})")
print(f"latest reading:                {inflation_yoy.iloc[-1]:.1f}% "
      f"({inflation_yoy.index[-1]:%Y-%m})")
print(f"last month at or below 2%:     {at_or_below_2.index[-1]:%Y-%m}")
```

同比消费者价格指数通货膨胀率自 2022 年触及峰值后迅速回落，但截至本文撰写之时，它自 2021 年初以来尚未再度回到 2% 的水平。

### 三个时代中的菲利普斯曲线

原书从 1960-1982 年的数据中提炼出了一条菲利普斯曲线。

1999 年之后最引人注目的模式，是通货膨胀与失业率的散点图在不同时代之间表现出的*不稳定性*。

我们将样本划分为原书所涉及的加速时期、大缓和时期，以及 2008 年之后的时期，并在每个时期分别绘制通货膨胀对失业率的散点图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The inflation-unemployment scatter across three eras
    name: fig-ts-three-eras
---
scatter_data = pd.concat([inflation_yoy.rename('inflation'),
                          u_full.rename('unemployment')], axis=1).dropna()

last_year = scatter_data.index[-1].year
eras = [('1960', '1983', '1960-1983 (acceleration)'),
        ('1984', '2007', '1984-2007 (Great Moderation)'),
        ('2008', None, f'2008-{last_year} (crisis, COVID, surge)')]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
for ax, (lo, hi, title) in zip(axes, eras):
    era_data = scatter_data.loc[lo:hi]
    ax.scatter(era_data['unemployment'], era_data['inflation'], s=8, alpha=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('unemployment')
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel('inflation (year-over-year)')
plt.tight_layout()
plt.show()
```

下面给出每个时代中通货膨胀对失业率的相关系数与最小二乘斜率。

```{code-cell} ipython3
for lo, hi, title in eras:
    era_data = scatter_data.loc[lo:hi]
    corr = era_data.corr().iloc[0, 1]
    slope = np.polyfit(era_data['unemployment'], era_data['inflation'], 1)[0]
    print(f"{title:38s} correlation {corr:+.2f}   slope {slope:+.2f}")
```

在 1960-1983 年间，数据点散布在一个很宽的通货膨胀率范围内——这正是预期不断变化、菲利普斯曲线呈现*回路*形态的时代；就水平值而言，相关系数呈微弱*正*值，这是因为不利的移动压过了权衡关系本身。

在 1984-2007 年间，它们收缩成一团紧密、低位、近乎平坦的点云——这就是大缓和时期，此时通货膨胀几乎对失业率毫无反应。

自 2008 年以来，散点呈现出一种 L 形：在中等和较高失业率水平下近乎平坦，贯穿金融危机后的缓慢复苏期以及新冠疫情引发的经济衰退；而在低失业率水平下则近乎垂直，2021-2022 年的飙升正落于此处。

颇具悖论意味的是，这恰恰是唯一一个水平值相关系数呈明显负值的时代。

L 形散点既可能与一条凸性的菲利普斯曲线相符——它只在劳动力市场紧张时才变得陡峭——也可能是因为供给冲击恰好在失业率处于低位时到来。

无论菲利普斯曲线究竟是什么，它都不是一种稳定的*简化形式*关系。

这正是一个带有漂移型政府信念的自然率经济所预言的情形：将失业率与*意外*通货膨胀联系起来的结构性曲线可以保持不变，而将失业率与通货膨胀本身联系起来的经验曲线，则会随着预期和政策的变化而漂移。

{ref}`ts_ex3` 在商业周期频率下重复了这一比较，而在那里，答案却有所不同。

### 新数据对这两个故事意味着什么

这额外的四分之一个世纪的数据，既没有干净利落地推翻其中任何一个故事，也没有干净利落地证实它们，但它使两者都变得更加清晰。

**对自然率理论的胜利而言。**
大缓和时期可以被解读为这场胜利的最终完成：随着自然率共识的确立以及央行独立性的确立，通货膨胀在整整一代人的时间里保持低位，预期也保持*锚定*。

用本书的语言来说，政策制定者已经学会了自然率假说的一个正确的理性预期版本，并找到了能使自身承诺保持低通货膨胀的手段——这正是本书所期望达成的结果。

即便是 2021-2022 年的飙升，按照这种解读，也支持了胜利的故事：一旦美联储采取果断行动，通货膨胀便从峰值迅速回落，长期预期从未失去锚定，尽管通货膨胀尚未回到 2021 年之前的水平。

**对计量经济学政策评估的平反而言。**
这次飙升也提醒我们，货币当局的*模型*仍然可能误导它：2021 年广泛流行的"通货膨胀只是暂时的"观点，正是一个被数据证伪了的模型，而政策只是在信念发生转变之后才随之调整。

2020 年之前那十年通货膨胀低于目标水平的时期——那条看似*平坦*的菲利普斯曲线，既没有出现 2009-2013 年"消失的反通货膨胀"，也没有出现 2015-2019 年"消失的通货膨胀"，都不符合一条稳定曲线——恰恰正是本书中适应性政府会实时追踪其斜率和截距不断变化的那种漂移型经验关系。

{ref}`ts_ex2` 要求你像这样一个政府那样去追踪这些系数。

```{note}
本书自身也会坚持提出这样一个警示，对它自己所用的样本和我们所用的样本同样适用：其机制假设*基本面*——即真实的数据生成过程——是稳定的，因此所有的作用都来自政府不断演化的信念。

正如我们在前文所见，本书不打算"考察不寻常的供给冲击"，这既排除了 1973 年和 1979 年的石油冲击，也同样排除了 2021-2022 年的疫情中断与能源价格冲击。

将信念的转变与基本面的转变区分开来，正是使这段历史如此难以捉摸、也如此引人入胜的那个识别问题；{doc}`phillips_drifts_volatilities` 将直接对此展开探讨。
```

本系列后续讲座中所建立的工具——自我确认均衡、漂移系数以及逃逸动态——仍然是探讨新数据所提出的这一问题的一种自然语言：低通货膨胀是否会在每次冲击之后重新锚定，还是说一连串的意外仍可能使信念重新开始漂移，就像 1965 年之后所发生的那样？

借用本书书名中的说法：美国对通货膨胀的征服，是否已经功亏一篑？

{doc}`phillips_drifts_volatilities` 与 {doc}`phillips_drifts_volatilities_2` 直接从数据出发，探讨 2000 年前后，战后美国宏观经济动态的系数究竟是否真的发生了漂移。

收尾一讲 {doc}`phillips_lost_conquest` 把本系列的这些工具应用于 2021-2022 年的飙升，并探讨了为什么美联储的反应如此迟缓。

我们将在 {doc}`phillips_credibility` 中着手构建这些模型。

(phillips_notation)=
## 关于符号的说明

各讲座遵循其所依据来源的符号表示，而这些来源之间并不一致。

与其强行统一一套方案、从读者可能想要查阅的论文中脱离出来，我们宁愿让每一讲都忠实于其原始来源，并在此处记录各讲座之间的符号对应关系。

从有限理性系列过来的读者应注意，在该系列中 $\gamma$ 表示学习增益，政府的信念用 $\alpha$ 表示（{ref}`br_notation`，以及 {ref}`olg_phillips` 中的对照表）；而从 {doc}`phillips_adaptive` 到 {doc}`phillips_priors`，$\gamma$ 则表示政府的信念。

| 对象 | 符号 | 出处 |
|---|---|---|
| 通货膨胀 | $y$ | {doc}`phillips_credibility` 到 {doc}`phillips_learning`（该书的符号表示法） |
| | $\pi$ | {doc}`phillips_escaping_nash`、{doc}`phillips_priors`、{doc}`phillips_drifts_volatilities`、{doc}`phillips_drifts_volatilities_2`、{doc}`phillips_lost_conquest`；{ref}`olg_phillips` |
| 公众的预期通货膨胀 | $x$ | {doc}`phillips_credibility` 到 {doc}`phillips_learning` |
| | $\hat x$ | {doc}`phillips_escaping_nash`、{doc}`phillips_priors` |
| 政府的系统性通货膨胀 | $h(\gamma) X$ | {doc}`phillips_self_confirming`、{doc}`phillips_learning` |
| | $x$ | {doc}`phillips_escaping_nash`、{doc}`phillips_priors` |
| | $g_{t-1}$ | {ref}`olg_phillips` |
| 产出缺口 | $x$ | {doc}`phillips_lost_conquest` |
| 自然失业率 | $U^*$ | {doc}`phillips_credibility` 到 {doc}`phillips_learning`；{ref}`olg_phillips` |
| | $u$ | {doc}`phillips_escaping_nash`、{doc}`phillips_priors` |
| 真实菲利普斯曲线的斜率 | $\theta$ | {doc}`phillips_credibility` 到 {doc}`phillips_escaping_nash`；{ref}`olg_phillips` |
| 感知的菲利普斯曲线的斜率 | $\kappa$ | {doc}`phillips_lost_conquest` |
| 政府的信念 | $\gamma$（古典拟合）、$\beta$（凯恩斯拟合） | {doc}`phillips_adaptive` 到 {doc}`phillips_learning` |
| | $\gamma = (\gamma_1, \gamma_{-1})$，斜率在前 | {doc}`phillips_escaping_nash` |
| | $\gamma = (a, b)$，截距在前 | {doc}`phillips_priors`；{ref}`olg_phillips` 中为 $(\alpha_0, \alpha_1)$ |
| | $\theta_t = (\alpha_{0,t}, \rho_t, \kappa_t)$ | {doc}`phillips_lost_conquest` |
| 计量经济学家的 VAR 系数 | $\theta_t$ | {doc}`phillips_drifts_volatilities`、{doc}`phillips_drifts_volatilities_2` |
| 学习增益 | $1/t$、$a_n$ | {doc}`phillips_credibility`（最小二乘法及其附录） |
| | $1 - \lambda$ | {doc}`phillips_adaptive`（凯根-弗里德曼）、{doc}`phillips_learning`（常数增益 $g_t = 1 - \lambda$） |
| | $C$ | {doc}`phillips_misspecified`、{doc}`phillips_self_confirming`（为拟合数据而选定的增益） |
| | $\varepsilon$ | {doc}`phillips_learning` 中的入门部分；{doc}`phillips_escaping_nash`、{doc}`phillips_priors` |
| | $\gamma$ | {doc}`phillips_lost_conquest`；有限理性系列 |
| 贴现因子 | $\delta$ | {doc}`phillips_credible_policies` 到 {doc}`phillips_priors` |
| | $\beta$ | {doc}`phillips_lost_conquest` |

{doc}`phillips_policy_evaluation` 使用该书第九章中采用的符号表示法——即 {doc}`phillips_learning` 的符号表示法——对该模型进行了估计：通货膨胀为 $y$，信念为 $\gamma$ 和 $\beta$，遗忘因子为 $\lambda$，贴现因子为 $\delta$。

有几个字母改变了用途，如果读者将符号的含义一路带下去，就会在此处被绊住。

* 字母 $x$ 换了立场：在 {doc}`phillips_learning` 之前，它表示 *公众* 的预期，而在 {doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 中，它表示 *政府* 设定的值。两者在均衡状态下是相同的，这正是这种转变容易被忽视的原因。在 {doc}`phillips_lost_conquest` 中，$x$ 表示产出缺口。
* 字母 $\theta$ 在早期讲座中是菲利普斯曲线的斜率，在 {doc}`phillips_lost_conquest` 中是政府的整个信念向量，而在 {doc}`phillips_drifts_volatilities` 和 {doc}`phillips_drifts_volatilities_2` 中则是 *计量经济学家* 的系数，而非政府的信念。
* 字母 $\gamma$ 从 {doc}`phillips_adaptive` 到 {doc}`phillips_priors` 表示政府的信念，但在 {doc}`phillips_lost_conquest` 中（与有限理性系列一致）则表示学习增益。
* 字母 $\beta$ 在 {doc}`phillips_adaptive`、{doc}`phillips_self_confirming` 和 {doc}`phillips_learning` 中是政府的凯恩斯菲利普斯曲线系数，在 {doc}`phillips_lost_conquest` 中是美联储的贴现因子（其中 $\beta_n$ 是其新凯恩斯经济模型的贴现因子），而在 {doc}`phillips_drifts_volatilities` 中则是一组协方差参数。
* 字母 $\lambda$ 身兼数职：在 {doc}`phillips_adaptive` 中是凯根-弗里德曼适应性参数，在 {doc}`phillips_learning` 中是遗忘因子，在 {doc}`phillips_priors` 中是先验收紧参数，在 {doc}`phillips_drifts_volatilities` 中是一组参数，而在 {doc}`phillips_lost_conquest` 中则是一个实测的持续性根，其中 $\lambda_x$ 同时也是美联储损失函数中产出缺口的权重。

{doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 中的静态模型存在两种排序方式，因为每一讲都遵循其各自的原始论文。

{doc}`phillips_escaping_nash` 遵循 {cite:t}`ChoWilliamsSargent2002`，先列出政府回归方程的斜率；而 {doc}`phillips_priors` 和 {ref}`olg_phillips` 遵循 {cite:t}`SargentWilliams2005`，先列出截距。

各讲座中的常数增益也各不相同，这同样是遵循各自来源的结果。

| 增益 | 出处 | 来源 |
|---|---|---|
| 0.025 | {doc}`phillips_learning` | 该书的遗忘因子 $\lambda = .975$（第八章） |
| 0.02 | {ref}`olg_phillips` | 有限理性系列中的简化模型 |
| 一个范围 | {doc}`phillips_escaping_nash` | 逃逸时间随增益变化而变化 |
| 0.05 和 0.02 | {doc}`phillips_priors` | 0.05 为 {cite:t}`SargentWilliams2005` 表格中最大的增益，用于分析；0.02 用于与 {ref}`olg_phillips` 进行常数增益比较 |
| 0.015 | 该书第九章及 {doc}`phillips_policy_evaluation`，季度数据 | 该书固定 $\lambda = .985$ 和 $\delta = .999$，并估计其余参数 |
| 0.03 | {doc}`phillips_lost_conquest` | {cite:t}`SargentWilliams2025`，季度数据 |

由于这些设定中各期的长度不同，这些数值不能直接比较：增益 $g$ 使数据具有约 $1/g$ 期的有效记忆长度。

较小的增益使从自我确认均衡中逃逸变得更为罕见，但在类似 {doc}`phillips_learning` 和 {ref}`olg_phillips` 中所采用的增益下，逃逸则较为频繁；{doc}`phillips_escaping_nash` 测量了随着增益减小，逃逸时间是如何增长的。

## 练习

```{exercise-start}
:label: ts_ex1
```

"聚焦于商业周期成分使得表面上的菲利普斯曲线更加清晰"这一论断，可以被定量地表述出来，并且图 1.5 中的环路也可以得到解释。

请使用 `data` 中 1961-1982 年的月度序列——这是本书窗口期中经过滤波的序列所覆盖的部分。

1. 计算原始月度序列之间，以及图 1.4 所绘制的通货膨胀 13 个月移动平均值与失业率之间的相关系数。
1. 将每条原始序列分解为三个频段——周期长于 84 个月、周期介于 24 至 84 个月之间（即图 1.5 中的巴克斯特-金成分），以及周期短于 24 个月——并计算每个频段内的相关系数。
   提示：`bkfilter(data, low=2, high=84, K=84)` 保留了所有短于 84 个月的周期；用数据减去该结果即为低频段，再用低频段减去商业周期成分即为高频段。
1. 哪个频段承载了菲利普斯曲线？哪个频段承载了不利的移动？为什么原始相关系数接近于零？
1. 对于商业周期成分，计算 $\operatorname{corr}(\pi_t, U_{t-k})$，其中 $k = 0, 1, \ldots, 24$ 个月，$\pi_t$ 为通货膨胀，$U_t$ 为失业率。在哪个滞后期相关系数最为负值？这一滞后期如何解释图 1.5 中环路的方向？

```{exercise-end}
```

```{solution-start} ts_ex1
:class: dropdown
```

我们将数据分解为三个频段并计算相关系数。

```{code-cell} ipython3
window = slice('1961', '1982')

below_84 = bkfilter(data, low=2, high=84, K=84)   # periods under 84 months
below_84.columns = data.columns
cycle = bk.set_axis(data.columns, axis=1)          # periods of 24-84 months
bands = {'over 84 months': data.loc[below_84.index] - below_84,
         '24 to 84 months': cycle,
         'under 24 months': below_84 - cycle}

smoothed = pd.concat([inflation_ma, u], axis=1).dropna()
raw_var = data.loc[window, 'inflation'].var()

print(f"{'raw monthly series':28s} correlation "
      f"{data.loc[window].corr().iloc[0, 1]:+.2f}")
print(f"{'13-month moving average':28s} correlation "
      f"{smoothed.loc[window].corr().iloc[0, 1]:+.2f}")
for name, band in bands.items():
    b = band.loc[window]
    share = b['inflation'].var() / raw_var
    print(f"{'periods ' + name:28s} correlation {b.corr().iloc[0, 1]:+.2f}"
          f"   share of inflation variance {share:.2f}")
```

原始相关系数接近于零，是因为它混合了符号相反的三个频段。

在周期长于七年的频段上，相关系数是*正*的：这正是菲利普斯曲线的不利移动，是自然率理论归因于预期通货膨胀的那种缓慢移动的成分，也是辩护叙事中归因于政府信念漂移的那种成分。

在商业周期频段上，相关系数是强烈负相关的：这个频段承载了菲利普斯曲线。

在周期短于两年的频段上——它占据了原始月度通货膨胀方差中相当大的份额，其中很大一部分是未经调整的指数中的四舍五入和季节性噪声——相关系数较弱。

仅靠平滑处理并无助益：13 个月移动平均去除了大部分高频成分，却保留了正向的低频移动，因此其相关系数甚至比原始相关系数还要略微偏正。

带通滤波之所以有效，是因为它将两者都去除了。

现在来看滞后关系模式。

```{code-cell} ipython3
lags = np.arange(25)
cross = np.array([cycle['inflation'].corr(cycle['unemployment'].shift(k).loc[window])
                  for k in lags])
k_star = lags[np.argmin(cross)]
print(f"corr(π_t, U_t)       = {cross[0]:+.2f}")
print(f"most negative at k = {k_star} months: corr(π_t, U_t-k) = {cross.min():+.2f}")
```

当通货膨胀与大约六个月前的失业率配对时，相关系数最为负值，这说明通货膨胀滞后于失业率。

正是这一滞后，使得图 1.5 中以失业率为横轴的环路呈顺时针方向运行。

从环路右侧的失业率峰值开始。

随着失业率下降，通货膨胀尚未作出反应，因此路径沿底部向左移动；在失业率触底之后大约半年，通货膨胀才达到峰值，因此路径沿左侧向上攀升；随着失业率上升，通货膨胀仍处于高位，因此路径沿顶部向右移动；如此周而复始。

```{solution-end}
```

```{exercise-start}
:label: ts_ex2
```

卢卡斯观察到，预测者们不断调整其方程中的常数项。

让我们来看一位计量经济学家对经验菲利普斯曲线做同样的事情——将失业率对通货膨胀进行回归，正如 {ref}`olg_phillips`、{doc}`phillips_escaping_nash` 和 {doc}`phillips_priors` 中的政府所做的那样。

设 $\pi_t$ 为逐年 CPI 通货膨胀（`inflation_yoy`），它仅使用过去的数据，$U_t$ 为失业率（`u_full`）。

通过递归最小二乘法估计

$$
U_t = a_t + b_t \pi_t + \text{error}
$$

从 1954-1959 年的普通最小二乘估计出发，从 1960 年 1 月起到最新月份逐月更新，使用

$$
\begin{aligned}
R_t &= R_{t-1} + g_t \left(z_t z_t^\top - R_{t-1}\right), \\
\begin{bmatrix} a_t \\ b_t \end{bmatrix} &= \begin{bmatrix} a_{t-1} \\ b_{t-1} \end{bmatrix} + g_t R_t^{-1} z_t \left(U_t - a_{t-1} - b_{t-1} \pi_t\right),
\end{aligned}
$$

其中 $z_t = (1, \pi_t)^\top$。

1. 使用 {ref}`olg_phillips` 中的常数增益 $g_t = 0.02$——对于月度数据，这相当于约四年的有效记忆长度，绘制 $a_t$ 和 $b_t$ 的图像。
1. 用普通最小二乘法中递减的增益 $g_t = 1/t$ 重复上述过程，$t$ 从 1954-1959 年预样本的起始处开始计数，并作比较。
1. 估计出的斜率 $b_t$ 在 1960 年代末、1970 年代以及 2008 年之后分别表现如何？截距 $a_t$ 又发生了什么变化？
1. {ref}`olg_phillips` 中的静态政府会设定目标通货膨胀率为 $-a_t b_t / (1 + b_t^2)$。将该目标与实际通货膨胀作比较。

```{exercise-end}
```

```{solution-start} ts_ex2
:class: dropdown
```

下面是一个可同时处理两种增益方式的递归最小二乘例程。

```{code-cell} ipython3
def rls_phillips(π, U, coef0, R0, gain=None, t0=0):
    """
    Recursive least squares for U_t = a + b π_t, starting from estimates
    coef0 = (a, b) and moment matrix R0 based on t0 observations.  Uses a
    constant gain if `gain` is given and the gain 1/t of ordinary least
    squares otherwise.
    """
    coef, R = coef0.copy(), R0.copy()
    path = np.empty((len(π), 2))
    for t in range(len(π)):
        g = gain if gain is not None else 1 / (t0 + t + 1)
        z = np.array([1.0, π[t]])
        R = R + g * (np.outer(z, z) - R)
        coef = coef + g * np.linalg.solve(R, z * (U[t] - z @ coef))
        path[t] = coef
    return path

rls_data = pd.concat([inflation_yoy.rename('π'), u_full.rename('U')],
                     axis=1).dropna()
pre, est = rls_data.loc[:'1959'], rls_data.loc['1960':]

Z = np.column_stack([np.ones(len(pre)), pre['π']])
coef0 = np.linalg.lstsq(Z, pre['U'], rcond=None)[0]   # OLS on the pre-sample
R0 = Z.T @ Z / len(pre)

paths = {}
for label, gain in [('constant gain 0.02', 0.02), ('least squares, gain 1/t', None)]:
    path = rls_phillips(est['π'].values, est['U'].values, coef0, R0,
                        gain=gain, t0=len(pre))
    paths[label] = pd.DataFrame(path, index=est.index, columns=['a', 'b'])
    paths[label]['target'] = (-paths[label]['a'] * paths[label]['b']
                              / (1 + paths[label]['b']**2))
```

我们绘制出漂移的系数以及静态规则的目标值。

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for label, p in paths.items():
    axes[0].plot(p['a'], lw=1.2, label=label)
    axes[1].plot(p['b'], lw=1.2, label=label)
    axes[2].plot(p['target'], lw=1.2, label=f'static target, {label}')
axes[2].plot(est['π'], 'k', lw=0.6, alpha=0.6, label='actual inflation')
axes[1].axhline(0, color='k', lw=0.5)
axes[0].set_ylabel('intercept $a_t$')
axes[1].set_ylabel('slope $b_t$')
axes[2].set_ylabel('percent')
axes[2].set_xlabel('year')
for ax in axes:
    ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()
```

几个具体数字可以使这一比较更加精确。

```{code-cell} ipython3
cg, ls = paths['constant gain 0.02'], paths['least squares, gain 1/t']
for lo, hi in [('1960', '1971'), ('1972', '2007'), ('2008', '2015'), ('2016', None)]:
    b = cg.loc[lo:hi, 'b']
    print(f"steepest constant-gain slope, {lo}-{b.index[-1].year}: "
          f"{b.min():+.2f} ({b.idxmin():%Y-%m})")
print(f"constant-gain slope, 1974-1981: {cg.loc['1974':'1981', 'b'].min():+.2f} "
      f"to {cg.loc['1974':'1981', 'b'].max():+.2f}")
print(f"constant-gain intercept, range: {cg['a'].min():.1f} to {cg['a'].max():.1f}")
print(f"least squares, after 1975:  intercept {ls.loc['1975':, 'a'].min():.1f} "
      f"to {ls.loc['1975':, 'a'].max():.1f},  slope {ls.loc['1975':, 'b'].min():+.2f} "
      f"to {ls.loc['1975':, 'b'].max():+.2f}")
print(f"constant-gain static target, 1960-1985: "
      f"{cg.loc[:'1985', 'target'].min():.1f} to {cg.loc[:'1985', 'target'].max():.1f};"
      f"  actual inflation peak {est.loc[:'1985', 'π'].max():.1f}")
```

在常数增益下，估计出的斜率在 1960 年代初接近于零，随后在 1960 年代末转为强烈负值：按照当时数据所显示的那样，萨缪尔森和索洛所主张的权衡关系看起来是真实且可被利用的。

在 1970 年代，斜率又回落至零附近，有时甚至转为正值，这是因为通货膨胀上升却未能降低失业率。

这正是不利移动，而在 {doc}`phillips_escaping_nash` 的静态模型中，接近于零的斜率恰恰正是促使政府以零通货膨胀为目标的那种信念。

2008 年之后，斜率两次急剧转为负值，分别对应于 2009 年和 2020 年失业率骤增而通货膨胀下降的时期；这是整个样本期内最陡峭的斜率。

截距也在漂移，在 1980 年代初跳升，当时沃尔克式的反通胀与高失业率同时出现，并在 2008 年之后及 2020 年再次跳升：这些正是卢卡斯所说的被不断调整的常数项。

在递减增益下，估计值自 1970 年代中期以后变化十分缓慢，这是因为最小二乘法对 1955 年的一个观测值赋予的权重与对上个月观测值的权重相同，因而无法跟踪系数的漂移。

在 1985 年之前，静态政府的目标通货膨胀率从未明显超过 2%，远低于 1970 年代的通货膨胀水平。

若以这些数据拟合出一个静态的古典政府模型，它绝不会建议推行大通胀；本书第 9 章发现，其自身的古典模型"甚至未能匹配 1970 年前通货膨胀的加速"，因而更偏好一种分布滞后的凯恩斯主义设定（{doc}`phillips_policy_evaluation`）。

在关于漂移的某种特定先验之下（{doc}`phillips_priors`），常数增益最小二乘法近似于针对以随机游走方式漂移的系数所设计的卡尔曼滤波器；{doc}`phillips_learning` 将这样一位学习者置入模型之中，而 {doc}`phillips_drifts_volatilities` 则让计量经济学家的 VAR 系数以同样的方式漂移。

```{solution-end}
```

```{exercise-start}
:label: ts_ex3
```

三个时代的图比较的是各时代的水平值。

请在商业周期频率上重复这一比较。

1. 将图 1.3 中的巴克斯特-金滤波器应用于整个样本期内的原始月度通货膨胀（`inflation_full`）和失业率（`u_full`）。滤波后的序列在何处终止？为什么？
1. 为了延伸至样本末端，请使用克里斯蒂亚诺-菲茨杰拉德带通滤波器（`statsmodels.tsa.filters.cf_filter` 中的 `cffilter`），采用相同的频段，该滤波器在样本两端附近使用不对称权重，而不是直接丢弃这些数据。检验两种滤波器在都可用的区间内是否一致。
1. 针对 1960-1983 年、1984-2007 年以及 2008 年至样本末端这三个时期，计算商业周期成分之间的相关系数，以及通货膨胀成分对失业率成分的最小二乘斜率。大缓和时代的菲利普斯曲线在商业周期频率上是否平坦？2008 年之后又发生了什么？

```{exercise-end}
```

```{solution-start} ts_ex3
:class: dropdown
```

我们用两种方法对整个样本进行滤波。

```{code-cell} ipython3
from statsmodels.tsa.filters.cf_filter import cffilter

full = pd.concat([inflation_full.rename('inflation'),
                  u_full.rename('unemployment')], axis=1).dropna()

bk_full = bkfilter(full, low=24, high=84, K=84).set_axis(full.columns, axis=1)
cf_full = cffilter(full, low=24, high=84, drift=True)[0].set_axis(full.columns, axis=1)

print(f"data end in {full.index[-1]:%Y-%m}; "
      f"Baxter-King components end in {bk_full.index[-1]:%Y-%m}")
overlap = bk_full.index
for col in full.columns:
    print(f"{col}: correlation of the two filters' components "
          f"{bk_full[col].corr(cf_full.loc[overlap, col]):.2f}")
```

巴克斯特-金滤波器是一种两侧各取 84 个月的双边移动平均，因此会丢失样本末尾七年的数据，包括整个 2021-2022 年的飙升期。

克里斯蒂亚诺-菲茨杰拉德成分在两者皆可用的区间内与巴克斯特-金成分高度吻合；在样本末端附近，它依赖于单侧信息，并将随着新数据的到来而被修正。

为了与 {ref}`ts_ex1` 进行比较，我们还使用相同的滤波器计算了周期长于 84 个月频段的相关系数。

```{code-cell} ipython3
below_84_cf = cffilter(full, low=2, high=84, drift=True)[0].set_axis(full.columns, axis=1)
low_cf = full - below_84_cf                       # periods over 84 months

eras_bc = [('1960', '1983'), ('1984', '2007'), ('2008', None)]
for lo, hi in eras_bc:
    e = cf_full.loc[lo:hi]
    corr = e.corr().iloc[0, 1]
    slope = np.polyfit(e['unemployment'], e['inflation'], 1)[0]
    corr_low = low_cf.loc[lo:hi].corr().iloc[0, 1]
    print(f"{lo}-{e.index[-1].year}: business cycle: correlation {corr:+.2f}, "
          f"slope {slope:+.2f}, sd of inflation {e['inflation'].std():.2f};"
          f"  over 84 months: correlation {corr_low:+.2f}")
b = bk_full.loc['2008':'2019']
print(f"2008-{b.index[-1].year}, Baxter-King business cycle: correlation "
      f"{b.corr().iloc[0, 1]:+.2f}")
```

在商业周期频率上，大缓和时代的菲利普斯曲线与 1960-1983 年的曲线大致同样紧密，呈现出相近的负相关性，只是更为平坦，其波动幅度也更小。

1984 年之后消失的是淹没了水平值层面权衡关系的低频漂移——其相关系数由正转为接近于零——而非商业周期层面的权衡关系本身。

2008 年之后，商业周期相关性急剧减弱，而若使用巴克斯特-金滤波器考察 2008-2019 年（即疫情和飙升之前的时期），这一相关性几乎消失。

因此，与大缓和时代的平坦不同，2010 年代的"平坦菲利普斯曲线"在商业周期频率上同样有所体现。

这一排序与水平值层面的排序恰好相反：在水平值层面，2008 年之后的时期是唯一呈现明显负相关的时期；而那种相关性主要来自于比商业周期更缓慢的变动。

在样本末端附近，这些低频成分依赖于单侧信息，因此上述最后一个数字是其中最不可靠的。

```{solution-end}
```
