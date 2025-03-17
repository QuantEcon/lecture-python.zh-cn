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

(ppd)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`面板数据的Pandas使用 <single: Pandas for Panel Data>`

```{index} single: Python; Pandas
```

```{contents} 目录
:depth: 2
```

## 概述

在[之前关于pandas的讲座](https://python-programming.quantecon.org/pandas.html)中，我们学习了如何处理简单的数据集。

计量经济学家经常需要处理更复杂的数据集，比如面板数据。

常见任务包括：

* 导入数据、清理数据以及在多个轴上重塑数据。

* 从面板数据中选择时间序列或横截面数据。
* 对数据进行分组和汇总。

`pandas`（源自'panel'和'data'）包含强大且易用的工具，专门用于解决这类问题。

在接下来的内容中，我们将使用来自OECD的实际最低工资面板数据集来创建：

* 数据多个维度的汇总统计
* 数据集中各国平均最低工资的时间序列
* 按大洲划分的工资核密度估计

我们将首先从CSV文件中读取长格式面板数据，并使用`pivot_table`重塑生成的`DataFrame`来构建`MultiIndex`。

使用pandas的`merge`函数将为我们的`DataFrame`添加额外的详细信息，并使用`groupby`函数对数据进行汇总。

## 切片和重塑数据

我们将读取来自OECD的32个国家的实际最低工资数据集，并将其赋值给`realwage`。

数据集可通过以下链接访问：

```{code-cell} python3
url1 = 'https://raw.githubusercontent.com/QuantEcon/lecture-python/master/source/_static/lecture_specific/pandas_panel/realwage.csv'
```

```{code-cell} python3
import pandas as pd

# 为了查看目的显示6列
pd.set_option('display.max_columns', 6)

# 将小数点位数减少到2位
pd.options.display.float_format = '{:,.2f}'.format

realwage = pd.read_csv(url1)
```

让我们看看我们有什么可以使用的数据

```{code-cell} python3
realwage.head()  # 显示前5行
```

数据目前是长格式的，当数据有多个维度时这种格式难以分析。

我们将使用`pivot_table`创建宽格式面板，并使用`MultiIndex`来处理高维数据。

`pivot_table`的参数需要指定数据（values）、索引和我们想要在结果数据框中的列。

通过在columns参数中传入一个列表，我们可以在列轴上创建一个`MultiIndex`

```{code-cell} python3
realwage = realwage.pivot_table(values='value',
                                index='Time',
                                columns=['Country', 'Series', 'Pay period'])
realwage.head()
```

为了更容易地过滤我们的时间序列数据，接下来我们将把索引转换为`DateTimeIndex`

```{code-cell} python3
realwage.index = pd.to_datetime(realwage.index)
type(realwage.index)
```

这些列包含多层级索引，称为`MultiIndex`，各层级按层次结构排序（国家 > 系列 > 支付周期）。

`MultiIndex`是在pandas中管理面板数据最简单和最灵活的方式。

```{code-cell} python3
type(realwage.columns)
```

```{code-cell} python3
realwage.columns.names
```

和之前一样，我们可以选择国家（我们的`MultiIndex`的最高层级）

```{code-cell} python3
realwage['United States'].head()
```

在本讲中，我们将经常使用`MultiIndex`的堆叠和取消堆叠来将数据框重塑成所需的格式。

`.stack()`将列`MultiIndex`的最低层级旋转到行索引（`.unstack()`的作用方向相反 - 你可以试试看）

```{code-cell} python3
realwage.stack().head()
```

我们也可以传入一个参数来选择我们想要堆叠的层级

```{code-cell} python3
realwage.stack(level='Country').head()
```

使用`DatetimeIndex`可以轻松选择特定的时间段。

选择一年并堆叠`MultiIndex`的两个较低层级，可以创建我们面板数据的横截面

```{code-cell} python3
realwage.loc['2015'].stack(level=(1, 2)).transpose().head()
```

在本讲座剩余部分，我们将使用一个数据框，其中包含不同国家和时间段的每小时实际最低工资数据，以2015年美元计价。

要创建我们的筛选数据框（`realwage_f`），我们可以使用`xs`方法在保持更高层级（本例中为国家）的同时，选择多重索引中较低层级的值。

```{code-cell} python3
realwage_f = realwage.xs(('Hourly', 'In 2015 constant prices at 2015 USD exchange rates'),
                         level=('Pay period', 'Series'), axis=1)
realwage_f.head()
```

## 合并数据框和填充空值

与SQL等关系型数据库类似，pandas内置了合并数据集的方法。

使用来自[WorldData.info](https://www.worlddata.info/downloads/)的国家信息，我们将使用`merge`函数将每个国家所属的大洲添加到`realwage_f`中。

可以通过以下链接访问数据集：

```{code-cell} python3
url2 = 'https://raw.githubusercontent.com/QuantEcon/lecture-python/master/source/_static/lecture_specific/pandas_panel/countries.csv'
```

```{code-cell} python3
worlddata = pd.read_csv(url2, sep=';')
worlddata.head()
```

首先，我们将从`worlddata`中只选择国家和大洲变量，并将列名重命名为'Country'

```{code-cell} python3
worlddata = worlddata[['Country (en)', 'Continent']]
worlddata = worlddata.rename(columns={'Country (en)': 'Country'})
worlddata.head()
```

我们想要将新的数据框`worlddata`与`realwage_f`合并。

pandas的`merge`函数允许通过行将数据框连接在一起。

我们的数据框将使用国家名称进行合并，这需要我们使用`realwage_f`的转置，以便两个数据框中的行都对应于国家名称。

```{code-cell} python3
realwage_f.transpose().head()
```

我们可以使用左连接、右连接、内连接或外连接来合并我们的数据集：

* 左连接只包含左侧数据集中的国家
* 右连接只包含右侧数据集中的国家
* 外连接包含左侧和右侧数据集中的任一国家
* 内连接只包含左右数据集共有的国家

默认情况下，`merge`将使用内连接。

在这里，我们将传入`how='left'`以保留`realwage_f`中的所有国家，但丢弃在`worlddata`中没有对应数据项`realwage_f`的国家。

这在下图中用红色阴影部分表示

```{figure} /_static/lecture_specific/pandas_panel/venn_diag.png

```

我们还需要指定每个数据框中国家名称的位置，这将作为合并数据框的"键"。

我们的"左"数据框（`realwage_f.transpose()`）在索引中包含国家，所以我们设置`left_index=True`。

我们的'right'数据框（`worlddata`）在'Country'列中包含国家名称，所以我们设置`right_on='Country'`

```{code-cell} python3
merged = pd.merge(realwage_f.transpose(), worlddata,
                  how='left', left_index=True, right_on='Country')
merged.head()
```

在 `realwage_f` 中出现但在 `worlddata` 中未出现的国家，其 Continent 列将显示 `NaN`。

要检查是否发生这种情况，我们可以在 continent 列上使用 `.isnull()` 并过滤合并后的数据框

```{code-cell} python3
merged[merged['Continent'].isnull()]
```

我们有三个缺失值！

处理 NaN 值的一个选项是创建一个包含这些国家及其各自大洲的字典。

`.map()` 将会把 `merged['Country']` 中的国家与字典中的大洲进行匹配。

注意那些不在我们字典中的国家是如何被映射为 `NaN` 的

```{code-cell} python3
missing_continents = {'Korea': 'Asia',
                      'Russian Federation': 'Europe',
                      'Slovak Republic': 'Europe'}

merged['Country'].map(missing_continents)
```

我们不想用这个映射覆盖整个系列。

`.fillna()` 只会用映射填充 `merged['Continent']` 中的 `NaN` 值，而保持列中的其他值不变

```{code-cell} python3
merged['Continent'] = merged['Continent'].fillna(merged['Country'].map(missing_continents))

# 检查大洲是否正确映射

merged[merged['Country'] == 'Korea']
```

我们还要把美洲合并成一个大洲 - 这样可以让我们后面的可视化效果更好看。

为此，我们将使用`.replace()`并遍历一个包含我们想要替换的大洲值的列表

```{code-cell} python3
replace = ['Central America', 'North America', 'South America']

for country in replace:
    merged['Continent'].replace(to_replace=country,
                                value='America',
                                inplace=True)
```

现在我们已经将所有想要的数据都放在一个`DataFrame`中，我们将把它重新整形成带有`MultiIndex`的面板形式。

我们还应该使用`.sort_index()`来确保对索引进行排序，这样我们之后可以高效地筛选数据框。

默认情况下，层级将按照从上到下的顺序排序

```{code-cell} python3
merged = merged.set_index(['Continent', 'Country']).sort_index()
merged.head()
```

在合并过程中，我们丢失了`DatetimeIndex`，因为我们合并的列不是日期时间格式的

```{code-cell} python3
merged.columns
```

现在我们已经将合并的列设置为索引，我们可以使用`.to_datetime()`重新创建一个`DatetimeIndex`

```{code-cell} python3
merged.columns = pd.to_datetime(merged.columns)
merged.columns = merged.columns.rename('Time')
merged.columns
```

`DatetimeIndex`在行轴上运行更加顺畅，所以我们将对`merged`进行转置

```{code-cell} python3
merged = merged.transpose()
merged.head()
```

## 数据分组和汇总

对于理解大型面板数据集来说，数据分组和汇总特别有用。

一种简单的数据汇总方法是在数据框上调用[聚合方法](https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/06_calculate_statistics.html)，比如`.mean()`或`.max()`。

例如，我们可以计算2006年至2016年期间每个国家的平均实际最低工资（默认是按行聚合）

```{code-cell} python3
merged.mean().head(10)
```

使用这个数据系列，我们可以绘制数据集中每个国家过去十年的平均实际最低工资

```{code-cell} ipython
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
```

```{code-cell} ipython
merged.mean().sort_values(ascending=False).plot(kind='bar',
                                                title="2006-2016年平均实际最低工资")

# 设置国家标签
country_labels = merged.mean().sort_values(ascending=False).index.get_level_values('Country').tolist()
plt.xticks(range(0, len(country_labels)), country_labels)
plt.xlabel('国家')

plt.show()
```

通过向`.mean()`传入`axis=1`参数可以对列进行聚合（得到所有国家随时间变化的平均最低工资）

```{code-cell} python3
merged.mean(axis=1).head()
```

我们可以将这个时间序列绘制成折线图

```{code-cell} python3
merged.mean(axis=1).plot()
plt.title('2006 - 2016年平均实际最低工资')
plt.ylabel('2015年美元')
plt.xlabel('年份')
plt.show()
```

我们也可以指定`MultiIndex`的一个层级（在列轴上）来进行聚合

```{code-cell} python3
merged.groupby(level='Continent', axis=1).mean().head()
```

我们可以将每个大洲的平均最低工资绘制成时间序列图

```{code-cell} python3
merged.groupby(level='Continent', axis=1).mean().plot()
plt.title('平均实际最低工资')
plt.ylabel('2015年美元')
plt.xlabel('年份')
plt.show()
```

为了绘图目的，我们将去掉澳大利亚这个大洲

```{code-cell} python3
merged = merged.drop('Australia', level='Continent', axis=1)
merged.groupby(level='Continent', axis=1).mean().plot()
plt.title('平均实际最低工资')
plt.ylabel('2015年美元')
plt.xlabel('年份')
plt.show()
```

`.describe()` 可以快速获取一些常见的统计摘要数据

```{code-cell} python3
merged.stack().describe()
```

这是使用 `groupby` 的简化方法。

使用 `groupby` 通常遵循"拆分-应用-合并"的过程：

* 拆分：数据根据一个或多个键进行分组
* 应用：在每个组上独立调用函数
* 合并：函数调用的结果被合并到新的数据结构中

`groupby` 方法实现了这个过程的第一步，创建一个新的 `DataFrameGroupBy` 对象，将数据拆分成组。

让我们再次按大洲拆分 `merged`，这次使用 `groupby` 函数，并将结果对象命名为 `grouped`

```{code-cell} python3
grouped = merged.groupby(level='Continent', axis=1)
grouped
```

在对象上调用聚合方法会将函数应用于每个组，其结果会被合并到一个新的数据结构中。

例如，我们可以使用`.size()`返回数据集中每个大洲的国家数量。

在这种情况下，我们的新数据结构是一个`Series`

```{code-cell} python3
grouped.size()
```

通过调用 `.get_group()` 来返回单个组中的国家，我们可以为每个大洲创建2016年实际最低工资分布的核密度估计。

`grouped.groups.keys()` 将返回 `groupby` 对象中的键

```{code-cell} python3
continents = grouped.groups.keys()

for continent in continents:
    sns.kdeplot(grouped.get_group(continent).loc['2015'].unstack(), label=continent, fill=True)

plt.title('Real minimum wages in 2015')
plt.xlabel('US dollars')
plt.legend()
plt.show()
```

## 总结

本讲座介绍了pandas的一些高级特性，包括多重索引、合并、分组和绘图。

在面板数据分析中可能有用的其他工具包括[xarray](https://docs.xarray.dev/en/stable/)，这是一个将pandas扩展到N维数据结构的Python包。

## 练习

```{exercise-start}
:label: pp_ex1
```

在这些练习中，你将使用来自[Eurostat](https://ec.europa.eu/eurostat/data/database)的欧洲按年龄和性别划分的就业率数据集。

可以通过以下链接访问数据集：

```{code-cell} python3
url3 = 'https://raw.githubusercontent.com/QuantEcon/lecture-python/master/source/_static/lecture_specific/pandas_panel/employ.csv'
```

读取 CSV 文件会返回一个长格式的面板数据集。使用 `.pivot_table()` 构建一个带有 `MultiIndex` 列的宽格式数据框。

首先探索数据框和 `MultiIndex` 层级中可用的变量。

编写一个程序，快速返回 `MultiIndex` 中的所有值。

```{exercise-end}
```

```{solution-start} pp_ex1
:class: dropdown
```

```{code-cell} python3
employ = pd.read_csv(url3)
employ = employ.pivot_table(values='Value',
                            index=['DATE'],
                            columns=['UNIT','AGE', 'SEX', 'INDIC_EM', 'GEO'])
employ.index = pd.to_datetime(employ.index) # 确保日期为 datetime 格式
employ.head()
```

由于这是一个大型数据集，因此探索可用的层级和变量很有用

```{code-cell} python3
employ.columns.names
```

可以通过循环快速获取层级中的变量

```{code-cell} python3
for name in employ.columns.names:
    print(name, employ.columns.get_level_values(name).unique())
```

```{solution-end}
```

```{exercise-start}
:label: pp_ex2
```

筛选上述数据框，仅包含以'活动人口'百分比表示的就业数据。

使用`seaborn`创建一个按年龄组和性别分组的2015年就业率箱线图。

```{hint}
:class: dropdown

`GEO`包含地区和国家。
```

```{exercise-end}
```

```{solution-start} pp_ex2
:class: dropdown
```

为了方便按国家筛选，将`GEO`调整到最上层并对`MultiIndex`进行排序

```{code-cell} python3
employ.columns = employ.columns.swaplevel(0,-1)
employ = employ.sort_index(axis=1)
```

我们需要删除`GEO`中一些不是国家的项目。

一个快速去除欧盟地区的方法是使用列表推导式来查找`GEO`中以'Euro'开头的层级值。

```{code-cell} python3
geo_list = employ.columns.get_level_values('GEO').unique().tolist()
countries = [x for x in geo_list if not x.startswith('Euro')]
employ = employ[countries]
employ.columns.get_level_values('GEO').unique()
```

从数据框中仅选择活动人口中的就业百分比

```{code-cell} python3
employ_f = employ.xs(('Percentage of total population', 'Active population'),
                     level=('UNIT', 'INDIC_EM'),
                     axis=1)
employ_f.head()
```

在创建分组箱形图之前删除"总计"值

```{code-cell} python3
employ_f = employ_f.drop('Total', level='SEX', axis=1)
```

```{code-cell} python3
box = employ_f.loc['2015'].unstack().reset_index()
sns.boxplot(x="AGE", y=0, hue="SEX", data=box, palette=("husl"), showfliers=False)
plt.xlabel('')
plt.xticks(rotation=35)
plt.ylabel('人口百分比 (%)')
plt.title('欧洲就业情况 (2015)')
plt.legend(bbox_to_anchor=(1,0.5))
plt.show()
```

```{solution-end}
```

