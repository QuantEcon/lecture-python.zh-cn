---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Python线性回归

```{contents} 目录
:depth: 2
```

除了Anaconda中已有的库外，本讲座还需要以下库：

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install linearmodels
```

## 概述

线性回归是分析两个或多个变量之间关系的标准工具。

在本讲中，我们将使用Python包`statsmodels`来估计、解释和可视化线性回归模型。

在此过程中，我们将讨论多个主题，包括

- 简单和多元线性回归
- 可视化
- 内生性和遗漏变量偏差
- 两阶段最小二乘法

作为示例，我们将复现Acemoglu、Johnson和Robinson具有开创性意义的论文{cite}`Acemoglu2001`中的结果。

* 您可以在[这里](https://economics.mit.edu/research/publications/colonial-origins-comparative-development-empirical-investigation)下载论文。

在这篇论文中，作者强调了制度在经济发展中的重要性。

该论文的主要贡献是使用殖民者死亡率作为制度差异的*外生*变异来源。

这种变化对于确定是制度导致更大的经济增长，而不是反过来，是必要的。

让我们从一些导入开始：

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #设置默认图形大小
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
import seaborn as sns
sns.set_theme()
```

### 预备知识

本讲座假定您熟悉基础计量经济学。

关于这些主题的入门教材，请参见例如{cite}`Wooldridge2015`。

## 简单线性回归

{cite}`Acemoglu2001`希望确定制度差异是否可以帮助解释观察到的经济结果。

我们如何衡量*制度差异*和*经济结果*？

在这篇论文中，

- 经济结果用1995年经汇率调整的人均GDP对数表示。
- 制度差异用[政治风险服务集团](https://www.prsgroup.com/)构建的1985-95年间平均防止征用指数表示。

这些变量和论文中使用的其他数据可以在Daron Acemoglu的[网页](https://economics.mit.edu/people/faculty/daron-acemoglu/data-archive)上下载。

我们将使用pandas的`.read_stata()`函数将`.dta`文件中的数据读入数据框

```{code-cell} python3
df1 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable1.dta?raw=true')
df1.head()
```

让我们使用散点图来观察人均GDP和防止征用指数之间是否存在明显的关系

```{code-cell} python3
df1.plot(x='avexpr', y='logpgp95', kind='scatter')
plt.show()
```

该图显示了防止征用保护与人均GDP对数之间存在相当强的正相关关系。

具体来说，如果更高的防止征用保护是衡量制度质量的指标，那么更好的制度似乎与更好的经济成果（更高的人均GDP）呈正相关。

根据图表，选择线性模型来描述这种关系似乎是一个合理的假设。

我们可以将模型写作

$$
{logpgp95}_i = \beta_0 + \beta_1 {avexpr}_i + u_i
$$

其中：

- $\beta_0$ 是线性趋势线在y轴上的截距
- $\beta_1$ 是线性趋势线的斜率，表示防止风险保护对人均GDP对数的*边际效应*
- $u_i$ 是随机误差项（由于模型未包含的因素导致观测值偏离线性趋势）

从视觉上看，这个线性模型涉及选择一条最佳的直线

拟合数据，如下图所示（图2，引用自{cite}`Acemoglu2001`）

```{code-cell} python3
# 删除NA值是使用numpy的polyfit所必需的
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])

# 仅使用"基础样本"用于绘图目的
df1_subset = df1_subset[df1_subset['baseco'] == 1]

X = df1_subset['avexpr']
y = df1_subset['logpgp95']
labels = df1_subset['shortnam']

# 用国家标签替换标记点
fig, ax = plt.subplots()
ax.scatter(X, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (X.iloc[i], y.iloc[i]))

# 拟合线性趋势线
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

ax.set_xlim([3.3,10.5])
ax.set_ylim([4,10.5])
ax.set_xlabel('1985-95年平均征收风险')
ax.set_ylabel('1995年人均GDP对数（PPP）')
ax.set_title('图2：征收风险与收入之间的OLS关系')
plt.show()
```

估计线性模型参数（$\beta$值）最常用的技术是普通最小二乘法（OLS）。

顾名思义，OLS模型是通过寻找能使*残差平方和*最小化的参数来求解的，即：

$$
\underset{\hat{\beta}}{\min} \sum^N_{i=1}{\hat{u}^2_i}
$$

其中$\hat{u}_i$是观测值与因变量预测值之间的差异。

为了估计常数项$\beta_0$，我们需要在数据集中添加一列1（考虑如果将$\beta_0$替换为$\beta_0 x_i$且$x_i = 1$时的方程）

```{code-cell} python3
df1['const'] = 1
```

现在我们可以使用OLS函数在`statsmodels`中构建我们的模型。

我们将在`statsmodels`中使用`pandas`数据框，不过标准数组也可以作为参数使用

```{code-cell} python3
reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], \
    missing='drop')
type(reg1)
```

到目前为止，我们只是构建了模型。

我们需要使用`.fit()`来获得参数估计值
$\hat{\beta}_0$ 和 $\hat{\beta}_1$

```{code-cell} python3
results = reg1.fit()
type(results)
```

我们现在已将拟合的回归模型存储在`results`中。

要查看OLS回归结果，我们可以调用`.summary()`方法。

请注意，在原始论文中一个观测值被错误地删除了（参见Acemoglu网页中`maketable2.do`文件中的注释），因此系数略有不同。

```{code-cell} python3
print(results.summary())
```

从我们的结果中，我们看到

- 截距 $\hat{\beta}_0 = 4.63$。
- 斜率 $\hat{\beta}_1 = 0.53$。
- 正的 $\hat{\beta}_1$ 参数估计值表明，
  制度质量对经济结果有正面影响，正如
  我们在图中所看到的。
- $\hat{\beta}_1$ 的p值为0.000，表明
  制度对GDP的影响在统计上显著（使用p < 
  0.05作为拒绝规则）。
- R方值为0.611表明约61%的人均GDP对数
  变异可由防止征收保护来解释。

使用我们的参数估计，我们现在可以将估计关系写为

$$
\widehat{logpgp95}_i = 4.63 + 0.53 \ {avexpr}_i
$$

这个方程描述了最适合我们数据的直线，如图2所示。

我们可以使用这个方程来预测特定征收保护指数值
对应的人均GDP对数水平。

例如，对于一个指数值为7.07的国家（这是

在数据集中），我们发现他们预测的1995年人均GDP对数值为8.38。

```{code-cell} python3
mean_expr = np.mean(df1_subset['avexpr'])
mean_expr
```

```{code-cell} python3
predicted_logpdp95 = 4.63 + 0.53 * 7.07
predicted_logpdp95
```

获得这个结果有一个更简单（也更准确）的方法，就是使用
`.predict()` 并设置 $constant = 1$ 和
${avexpr}_i = mean\_expr$

```{code-cell} python3
results.predict(exog=[1, mean_expr])
```

我们可以通过在结果上调用`.predict()`来获取数据集中每个${avexpr}_i$值对应的预测${logpgp95}_i$数组。

将预测值与${avexpr}_i$绘制在图上显示，预测值都落在我们之前拟合的直线上。

同时也绘制了${logpgp95}_i$的观测值以作比较。

```{code-cell} python3
# 从整个样本中删除缺失观测值

df1_plot = df1.dropna(subset=['logpgp95', 'avexpr'])

# 绘制预测值

fix, ax = plt.subplots()
ax.scatter(df1_plot['avexpr'], results.predict(), alpha=0.5,
        label='predicted')

# 绘制观测值

ax.scatter(df1_plot['avexpr'], df1_plot['logpgp95'], alpha=0.5,
        label='observed')

ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('avexpr')
ax.set_ylabel('logpgp95')
plt.show()
```

## 扩展线性回归模型

到目前为止，我们只考虑了制度对经济表现的影响 - 几乎可以肯定还有许多其他因素影响着GDP，但这些因素尚未包含在我们的模型中。

忽略影响$logpgp95_i$的变量将导致**遗漏变量偏差**，从而产生有偏和不一致的参数估计。

我们可以通过添加其他可能影响$logpgp95_i$的因素，将双变量回归模型扩展为**多变量回归模型**。

{cite}`Acemoglu2001`考虑了其他因素，如：

- 气候对经济结果的影响；使用纬度作为代理变量
- 影响经济表现和制度的其他差异，如文化、历史等；通过使用大陆虚拟变量来控制

让我们使用`maketable2.dta`中的数据来估计论文中考虑的一些扩展模型（表2）

```{code-cell} python3
df2 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable2.dta?raw=true')

# 向数据集添加常数项
df2['const'] = 1

# 创建每个回归要使用的变量列表
X1 = ['const', 'avexpr']
X2 = ['const', 'avexpr', 'lat_abst']
X3 = ['const', 'avexpr', 'lat_abst', 'asia', 'africa', 'other']

# 对每组变量估计OLS回归
reg1 = sm.OLS(df2['logpgp95'], df2[X1], missing='drop').fit()
reg2 = sm.OLS(df2['logpgp95'], df2[X2], missing='drop').fit()
reg3 = sm.OLS(df2['logpgp95'], df2[X3], missing='drop').fit()
```

现在我们已经拟合了模型，我们将使用`summary_col`在一个表格中显示结果（模型编号与论文中的相对应）

```{code-cell} python3
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[reg1,reg2,reg3],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 1',
                                         'Model 3',
                                         'Model 4'],
                            info_dict=info_dict,
                            regressor_order=['const',
                                             'avexpr',
                                             'lat_abst',
                                             'asia',
                                             'africa'])

results_table.add_title('表2 - OLS回归')

print(results_table)
```

## 内生性

正如 {cite}`Acemoglu2001` 所讨论的，OLS模型可能存在**内生性**问题，导致模型估计有偏差且不一致。

具体来说，制度和经济结果之间可能存在双向关系：

- 较富裕的国家可能有能力负担或倾向于选择更好的制度
- 影响收入的变量可能也与制度差异相关
- 指数的构建可能存在偏差；分析师可能倾向于认为收入较高的国家拥有更好的制度

为了解决内生性问题，我们可以使用**两阶段最小二乘法(2SLS回归)**，这是OLS回归的扩展。

这种方法需要用一个变量来替代内生变量${avexpr}_i$，该变量必须：

1. 与${avexpr}_i$相关
1. 与误差项不相关（即不应直接影响因变量，否则由于遗漏变量偏差会与$u_i$相关）

这组新的回归变量被称为**工具变量**，其目的是消除我们在衡量制度差异时的内生性问题。

{cite}`Acemoglu2001`的主要贡献在于使用殖民者死亡率作为制度差异的工具变量。

他们假设殖民者较高的死亡率导致了更具掠夺性质的制度建立（对征收的保护较少），而这些制度一直延续至今。

通过散点图（{cite}`Acemoglu2001`中的图3），我们可以看到防止征收风险与殖民者死亡率呈负相关，这与作者的假设相符，满足了有效工具变量的第一个条件。

```{code-cell} python3
# Dropping NA's is required to use numpy's polyfit
df1_subset2 = df1.dropna(subset=['logem4', 'avexpr'])

X = df1_subset2['logem4']
y = df1_subset2['avexpr']
labels = df1_subset2['shortnam']

# Replace markers with country labels
fig, ax = plt.subplots()
ax.scatter(X, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (X.iloc[i], y.iloc[i]))

# Fit a linear trend line
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

ax.set_xlim([1.8,8.4])
ax.set_ylim([3.3,10.4])
ax.set_xlabel('殖民者死亡率对数')
ax.set_ylabel('1985-95年平均征收风险')
ax.set_title('图3：殖民者死亡率与征收风险之间的一阶关系')
plt.show()
```

如果17至19世纪的殖民者死亡率对当前GDP有直接影响（除了通过制度产生的间接影响外），第二个条件可能就不成立。

例如，殖民者死亡率可能与一个国家当前的疾病环境有关，这可能会影响当前的经济表现。

{cite}`Acemoglu2001`认为这种情况不太可能，因为：

- 大多数殖民者死亡是由疟疾和黄热病引起的，
  对当地人的影响有限。
- 非洲或印度等地区的当地人的疾病负担似乎并不高于平均水平，
  这一点从殖民前这些地区相对较高的人口密度可以得到证实。

由于我们似乎有了一个有效的工具变量，我们可以使用二阶段最小二乘法（2SLS）回归来获得一致且无偏的参数估计。

**第一阶段**

第一阶段包括对内生变量（${avexpr}_i$）进行工具变量回归。

工具变量是我们模型中所有外生变量的集合（而不仅仅是我们替换的变量）。

以模型1为例，我们的工具变量仅包含一个常数项和殖民者死亡率${logem4}_i$。

因此，我们将估计如下的第一阶段回归：

$$
{avexpr}_i = \delta_0 + \delta_1 {logem4}_i + v_i
$$

估计这个方程所需的数据位于`maketable4.dta`文件中（仅使用完整数据进行估计，由`baseco = 1`标识）

```{code-cell} python3
# 导入并选择数据
df4 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable4.dta?raw=true')
df4 = df4[df4['baseco'] == 1]

# 添加常数变量
df4['const'] = 1

# 拟合第一阶段回归并打印摘要
results_fs = sm.OLS(df4['avexpr'],
                    df4[['const', 'logem4']],
                    missing='drop').fit()
print(results_fs.summary())
```

**第二阶段**

我们需要使用`.predict()`来获取${avexpr}_i$的预测值。

然后在原始线性模型中，用预测值$\widehat{avexpr}_i$替换内生变量${avexpr}_i$。

因此，我们的第二阶段回归为

$$
{logpgp95}_i = \beta_0 + \beta_1 \widehat{avexpr}_i + u_i
$$

```{code-cell} python3
df4['predicted_avexpr'] = results_fs.predict()

results_ss = sm.OLS(df4['logpgp95'],
                    df4[['const', 'predicted_avexpr']]).fit()
print(results_ss.summary())
```

第二阶段回归结果为我们提供了制度对经济结果影响的无偏且一致的估计。

结果显示出比OLS结果更强的正相关关系。

请注意，虽然我们的参数估计是正确的，但我们的标准误差并不准确，因此不建议"手动"（通过分阶段OLS）计算2SLS。

我们可以使用[linearmodels](https://github.com/bashtage/linearmodels)包（statsmodels的扩展）在一步中正确估计2SLS回归。

注意，在使用`IV2SLS`时，外生变量和工具变量在函数参数中是分开的（而之前工具变量包含了外生变量）

```{code-cell} python3
iv = IV2SLS(dependent=df4['logpgp95'],
            exog=df4['const'],
            endog=df4['avexpr'],
            instruments=df4['logem4']).fit(cov_type='unadjusted')

print(iv.summary)
```

鉴于我们现在已经获得了一致且无偏的估计，我们可以从所估计的模型中推断出，制度差异（源于殖民时期建立的制度）可以帮助解释当今各国之间的收入水平差异。

{cite}`Acemoglu2001`使用0.94的边际效应来计算，智利和尼日利亚之间的指数差异（即制度质量）意味着收入可能相差高达7倍，这强调了制度在经济发展中的重要性。

## 总结

我们已经演示了在`statsmodels`和`linearmodels`中的基本OLS和2SLS回归。

如果你熟悉R语言，你可能想要使用`statsmodels`的[公式接口](https://www.statsmodels.org/dev/example_formulas.html)，或考虑使用[r2py](https://rpy2.github.io/)在Python中调用R。

## 练习

```{exercise}
:label: ols_ex1

在讲座中，我们认为原始模型存在内生性问题

由于收入可能对制度发展产生影响而导致的偏差。

虽然内生性最好通过思考数据和模型来识别，但我们可以使用**豪斯曼检验**来正式检验内生性。

我们要检验内生变量$avexpr_i$和误差项$u_i$之间是否存在相关性

$$
\begin{aligned}
 H_0 : Cov(avexpr_i, u_i) = 0  \quad (无内生性) \\
 H_1 : Cov(avexpr_i, u_i) \neq 0 \quad (存在内生性)
 \end{aligned}
$$

这个检验分两个阶段进行。

首先，我们对工具变量$logem4_i$回归$avexpr_i$

$$
avexpr_i = \pi_0 + \pi_1 logem4_i + \upsilon_i
$$

其次，我们获取残差$\hat{\upsilon}_i$并将其纳入原方程

$$
logpgp95_i = \beta_0 + \beta_1 avexpr_i + \alpha \hat{\upsilon}_i + u_i
$$

如果$\alpha$在统计上显著（p值<0.05），那么我们就拒绝原假设，得出$avexpr_i$是内生的结论。

使用上述信息，估算豪斯曼检验并解释你的结果。

```{solution-start} ols_ex1
:class: dropdown
```

```{code-cell} python3
# 加载数据
df4 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable4.dta?raw=true')

# 添加常数项
df4['const'] = 1

# 估算第一阶段回归
reg1 = sm.OLS(endog=df4['avexpr'],
              exog=df4[['const', 'logem4']],
              missing='drop').fit()

# 获取残差
df4['resid'] = reg1.resid

# 估算第二阶段残差
reg2 = sm.OLS(endog=df4['logpgp95'],
              exog=df4[['const', 'avexpr', 'resid']],
              missing='drop').fit()

print(reg2.summary())
```

输出结果显示残差系数具有统计显著性，表明 $avexpr_i$ 是内生的。

```{solution-end}
```

```{exercise}
:label: ols_ex2

OLS参数 $\beta$ 也可以使用矩阵代数和 `numpy` 来估计（你可能需要复习
[numpy](https://python-programming.quantecon.org/numpy.html) 课程来
完成这个练习）。

我们要估计的线性方程（用矩阵形式表示）是

$$
y = X\beta + u
$$

为了求解未知参数 $\beta$，我们要最小化
残差平方和

$$
\underset{\hat{\beta}}{\min} \hat{u}'\hat{u}
$$

重新整理第一个方程并代入第二个
方程，我们可以写成

$$
\underset{\hat{\beta}}{\min} \ (Y - X\hat{\beta})' (Y - X\hat{\beta})
$$

解这个优化问题得到 $\hat{\beta}$ 系数的解为

$$
\hat{\beta} = (X'X)^{-1}X'y
$$

使用上述信息，计算模型1中的 $\hat{\beta}$

使用 `numpy` - 你的结果应该与讲座前面 `statsmodels` 的输出结果相同。

```{solution-start} ols_ex2
:class: dropdown
```

```{code-cell} python3
# 加载数据
df1 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable1.dta?raw=true')
df1 = df1.dropna(subset=['logpgp95', 'avexpr'])

# 添加常数项
df1['const'] = 1

# 定义 X 和 y 变量
y = np.asarray(df1['logpgp95'])
X = np.asarray(df1[['const', 'avexpr']])

# 计算 β_hat
β_hat = np.linalg.solve(X.T @ X, X.T @ y)

# 打印 2 x 1 向量 β_hat 的结果
print(f'β_0 = {β_hat[0]:.2}')
print(f'β_1 = {β_hat[1]:.2}')
```

也可以使用 `np.linalg.inv(X.T @ X) @ X.T @ y` 来求解 $\beta$，但是推荐使用 `.solve()`，因为它涉及的计算更少。

```{solution-end}
```

