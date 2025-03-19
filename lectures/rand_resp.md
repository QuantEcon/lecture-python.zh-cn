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

# 随机化回应调查

## 概述

社会污名可能会阻止人们承认潜在的令人尴尬的行为或观点。

当人们不愿意参与关于个人敏感问题的抽样调查时，他们可能会拒绝参与，即使参与了，他们也可能会对敏感问题提供不正确的答案。

这些问题会导致**选择**偏差，给调查的解释和设计带来挑战。

为了说明社会科学家如何思考估计此类令人尴尬的行为和观点的普遍程度，本讲座描述了S. L. Warner {cite}`warner1965randomized`的一种经典方法。

Warner使用基础概率论构建了一种方法，在保护调查受访者**个人**隐私的同时，仍能估算出在一个**群体**中具有社会污名特征或从事社会污名活动的人数比例。

Warner的想法是在受访者的答案与调查制作者最终收到的**信号**之间添加**噪声**。

了解噪声的结构可以让受访者确信调查制作者无法观察到他的答案。

噪声注入程序的统计特性为受访者提供了**合理的可否认性**。

相关理念构成了现代**差分隐私**系统的基础。

(参见 https://en.wikipedia.org/wiki/Differential_privacy)


## Warner的策略

像往常一样，让我们导入将要使用的Python模块。


```{code-cell} ipython3
import numpy as np
import pandas as pd
```

假设人群中的每个人要么属于 A 组，要么属于 B 组。

我们想要估计属于 A 组的人口比例 $\pi$，同时保护个人受访者的隐私。

Warner {cite}`warner1965randomized` 提出并分析了以下程序：

- 从人群中有放回地抽取 $n$ 个随机样本，并对每个人进行访谈。
- 从人群中有放回地抽取 $n$ 个随机样本，并对每个人进行访谈。
- 准备一个**随机转盘**，该转盘指向字母 A 的概率为 $p$，指向字母 B 的概率为 $(1-p)$。
- 每个受试者转动随机转盘，看到一个面试官**看不到**的结果（A 或 B）。
- 受试者说明自己是否属于转盘所指向的组。
- 如果转盘指向受试者所属的组，受试者回答"是"；否则回答"否"。
- 受试者如实回答问题。

Warner构建了一个用于估计总体中集合A所占比例的最大似然估计量。

设

- $\pi$ : 总体中A的真实概率
- $p$ : 指针指向A的概率
- $X_{i}=\begin{cases}1,\text{ 如果第}i\text{个受试者回答是}\\0,\text{ 如果第}i\text{个受试者回答否}\end{cases}$

将样本集编号，使得前$n_1$个报告"是"，而后$n-n_1$个报告"否"。

样本集的似然函数为

$$
L=\left[\pi p + (1-\pi)(1-p)\right]^{n_{1}}\left[(1-\pi) p +\pi (1-p)\right]^{n-n_{1}} 
$$ (eq:one)

似然函数的对数为：

$$
\log(L)= n_1 \log \left[\pi p + (1-\pi)(1-p)\right] + (n-n_{1}) \log \left[(1-\pi) p +\pi (1-p)\right]
$$ (eq:two)

关于$\pi$最大化对数似然函数的一阶必要条件是：

$$
\frac{(n-n_1)(2p-1)}{(1-\pi) p +\pi (1-p)}=\frac{n_1 (2p-1)}{\pi p + (1-\pi)(1-p)} 
$$

或

$$

\pi p + (1-\pi)(1-p)=\frac{n_1}{n}
$$ (eq:3)

如果 $p \neq \frac{1}{2}$，则最大似然估计量（MLE）$\pi$ 为：

$$
\hat{\pi}=\frac{p-1}{2p-1}+\frac{n_1}{(2p-1)n}
$$ (eq:four)

我们计算MLE估计量 $\hat \pi$ 的均值和方差为：

$$
\begin{aligned}
\mathbb{E}(\hat{\pi})&= \frac{1}{2 p-1}\left[p-1+\frac{1}{n} \sum_{i=1}^{n} \mathbb{E} X_i \right] \\
&=\frac{1}{2 p-1} \left[ p -1 + \pi p + (1-\pi)(1-p)\right] \\
&=\pi
\end{aligned}
$$ (eq:five)

以及

$$
\begin{aligned}
Var(\hat{\pi})&=\frac{n Var(X_i)}{(2p - 1 )^2 n^2} \\
&= \frac{\left[\pi p + (1-\pi)(1-p)\right]\left[(1-\pi) p +\pi (1-p)\right]}{(2p - 1 )^2 n^2}\\
&=\frac{\frac{1}{4}+(2 p^2 - 2 p +\frac{1}{2})(- 2 \pi^2 + 2 \pi -\frac{1}{2})}{(2p - 1 )^2 n^2}\\
&=\frac{1}{n}\left[\frac{1}{16(p-\frac{1}{2})^2}-(\pi-\frac{1}{2})^2 \right]
\end{aligned}
$$ (eq:six)

方程 {eq}`eq:five` 表明 $\hat{\pi}$ 是 $\pi$ 的**无偏估计量**，而方程 {eq}`eq:six` 告诉我们估计量的方差。

为了计算置信区间，首先将 {eq}`eq:six` 重写为：

$$
Var(\hat{\pi})=\frac{\frac{1}{4}-(\pi-\frac{1}{2})^2}{n}+\frac{\frac{1}{16(p-\frac{1}{2})^2}-\frac{1}{4}}{n}
$$ (eq:seven)

这个方程表明 $\hat{\pi}$ 的方差可以表示为抽样方差和随机设备方差的和。

从上述表达式中我们可以发现：

- 当 $p$ 为 $\frac{1}{2}$ 时，表达式 {eq}`eq:one` 退化为一个常数。

- 当 $p$ 为 $1$ 或 $0$ 时，随机化估计退化为不含随机抽样的估计量。

我们将只讨论 $p \in (\frac{1}{2},1)$ 的情况

（$p \in (0,\frac{1}{2})$ 的情况是对称的）。

从表达式 {eq}`eq:five` 和 {eq}`eq:seven` 我们可以推导出：

- 随着 $p$ 的增加，$\hat{\pi}$ 的均方误差(MSE)减小。

## 比较两种调查设计

让我们比较前面的随机化回答法与一个简化的非随机化回答法。

在我们的非随机化回答法中，我们假设：

- A组成员以概率 $T_a$ 说真话，而B组成员以概率 $T_b$ 说真话
- $Y_i$ 为1或0，取决于样本中第i个成员的报告是否属于A组。

那么我们可以估计 $\pi$ 为：

$$
\hat{\pi}=\frac{\sum_{i=1}^{n}Y_i}{n}
$$ (eq:eight)

我们计算该估计量的期望值、偏差和方差为：

$$
\begin{aligned}
\mathbb{E}(\hat{\pi})&=\pi T_a + \left[ (1-\pi)(1-T_b)\right]\\
\end{aligned}
$$ (eq:nine)

$$
\begin{aligned}
Bias(\hat{\pi})&=\mathbb{E}(\hat{\pi}-\pi)\\
&=\pi [T_a + T_b -2 ] + [1- T_b] \\
\end{aligned}
$$ (eq:ten)

$$
\begin{aligned}

Var(\hat{\pi})&=\frac{ \left[ \pi T_a + (1-\pi)(1-T_b)\right]  \left[1- \pi T_a -(1-\pi)(1-T_b)\right] }{n}
\end{aligned}
$$ (eq:eleven)

定义一个

$$
\text{MSE 比率}=\frac{\text{随机化的均方误差}}{\text{常规的均方误差}}
$$

我们可以计算不同参数值下不同调查设计的MSE比率。

以下Python代码计算了我们想要观察的对象，以便在不同的$\pi_A$和$n$值下进行比较：

```{code-cell} ipython3
class Comparison:
    def __init__(self, A, n):
        self.A = A
        self.n = n
        TaTb = np.array([[0.95,  1], [0.9,   1], [0.7,    1], 
                         [0.5,   1], [1,  0.95], [1,    0.9], 
                         [1,   0.7], [1,   0.5], [0.95, 0.95], 
                         [0.9, 0.9], [0.7, 0.7], [0.5,  0.5]])
        self.p_arr = np.array([0.6, 0.7, 0.8, 0.9])
        self.p_map = dict(zip(self.p_arr, [f"MSE 比率: p = {x}" for x in self.p_arr]))
        self.template = pd.DataFrame(columns=self.p_arr)
        self.template[['T_a','T_b']] = TaTb
        self.template['Bias'] = None
    
    def theoretical(self):
        A = self.A
        n = self.n
        df = self.template.copy()
        df['Bias'] = A * (df['T_a'] + df['T_b'] - 2) + (1 - df['T_b'])
        for p in self.p_arr:
            df[p] = (1 / (16 * (p - 1/2)**2) - (A - 1/2)**2) / n / \
                    (df['Bias']**2 + ((A * df['T_a'] + (1 - A) * (1 - df['T_b'])) * (1 - A * df['T_a'] - (1 - A) * (1 - df['T_b'])) / n))
            df[p] = df[p].round(2)
        df = df.set_index(["T_a", "T_b", "Bias"]).rename(columns=self.p_map)
        return df
        
    def MCsimulation(self, size=1000, seed=123456):
        A = self.A
        n = self.n
        df = self.template.copy()
        np.random.seed(seed)
        sample = np.random.rand(size, self.n) <= A
        random_device = np.random.rand(size, n)
        mse_rd = {}
        for p in self.p_arr:
            spinner = random_device <= p
            rd_answer = sample * spinner + (1 - sample) * (1 - spinner)
            n1 = rd_answer.sum(axis=1)
            pi_hat = (p - 1) / (2 * p - 1) + n1 / n / (2 * p - 1)
            mse_rd[p] = np.sum((pi_hat - A)**2)
        for inum, irow in df.iterrows():
            truth_a = np.random.rand(size, self.n) <= irow.T_a
            truth_b = np.random.rand(size, self.n) <= irow.T_b
            trad_answer = sample * truth_a + (1 - sample) * (1 - truth_b)
            pi_trad = trad_answer.sum(axis=1) / n
            df.loc[inum, 'Bias'] = pi_trad.mean() - A
            mse_trad = np.sum((pi_trad - A)**2)
            for p in self.p_arr:
                df.loc[inum, p] = (mse_rd[p] / mse_trad).round(2)
        df = df.set_index(["T_a", "T_b", "Bias"]).rename(columns=self.p_map)
        return df
```

让我们使用以下参数值来运行代码

- $\pi_A=0.6$
- $n=1000$

我们可以使用上述公式理论上生成MSE比率。

我们也可以对MSE比率进行蒙特卡洛模拟。

```{code-cell} ipython3
cp1 = Comparison(0.6, 1000)
df1_theoretical = cp1.theoretical()
df1_theoretical
```

```{code-cell} ipython3
df1_mc = cp1.MCsimulation()
df1_mc
```

理论计算很好地预测了蒙特卡洛结果。

我们看到在许多情况下，特别是当偏差不小时，随机抽样方法的均方误差比非随机抽样方法要小。

随着$p$的增加，这些差异变得更大。

通过调整参数$\pi_A$和$n$，我们可以研究不同情况下的结果。

例如，对于Warner {cite}`warner1965randomized`描述的另一种情况：

- $\pi_A=0.5$
- $n=1000$

我们可以使用以下代码

```{code-cell} ipython3
cp2 = Comparison(0.5, 1000)
df2_theoretical = cp2.theoretical()
df2_theoretical
```

```{code-cell} ipython3
df2_mc = cp2.MCsimulation()
df2_mc
```

我们还可以重新审视Warner {cite}`warner1965randomized`结论部分的一个计算，其中

- $\pi_A=0.6$
- $n=2000$

我们使用以下代码

```{code-cell} ipython3
cp3 = Comparison(0.6, 2000)
df3_theoretical = cp3.theoretical()
df3_theoretical
```

```{code-cell} ipython3
df3_mc = cp3.MCsimulation()
df3_mc
```

显然，随着$n$的增加，随机化回应法在更多情况下表现更好。

## 结束语

{doc}`这个QuantEcon讲座<util_rand_resp>`描述了一些其他的随机化回应调查方法。

该讲座介绍了Lars Ljungqvist {cite}`ljungqvist1993unified`对这些替代方案进行的功利主义分析。

