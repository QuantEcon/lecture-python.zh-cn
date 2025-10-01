---
jupytext:
  text_representation:
    extension: .myst
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 一些概率分布

本讲座是{doc}`这个关于矩阵统计的讲座 <prob_matrix>`的补充内容。

它描述了一些常见的分布，并使用Python从这些分布中进行采样。

它还描述了一种通过转换均匀概率分布的样本来从你自己设计的任意概率分布中采样的方法。

除了Anaconda中已有的库外，本讲座还需要以下库：

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install prettytable
```

像往常一样，我们先导入一些库

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import prettytable as pt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')
```

## 一些离散概率分布

让我们编写一些Python代码来计算单变量随机变量的均值和方差。

我们将使用代码来

- 从概率分布计算总体均值和方差
- 生成N个独立同分布的样本并计算样本均值和方差
- 比较总体和样本的均值和方差

## 几何分布

离散几何分布的概率质量函数为

$$
\textrm{Prob}(X=k)=(1-p)^{k-1}p,k=1,2, \ldots,  \quad p \in (0,1)
$$

其中$k = 1, 2, \ldots$是第一次成功之前的试验次数。

这个单参数概率分布的均值和方差为

$$
\begin{aligned}
\mathbb{E}(X) & =\frac{1}{p}\\\mathbb{Var}(X) & =\frac{1-p}{p^2}
\end{aligned}
$$

让我们使用Python从该分布中抽取观测值，并将样本均值和方差与理论结果进行比较。

```{code-cell} ipython3
# 指定参数
p, n = 0.3, 1_000_000

# 从分布中抽取观测值
x = np.random.geometric(p, n)

# 计算样本均值和方差
μ_hat = np.mean(x)
σ2_hat = np.var(x)

print("样本均值为：", μ_hat, "\n样本方差为：", σ2_hat)

# 与理论结果比较
print("\n总体均值为：", 1/p)
print("总体方差为：", (1-p)/(p**2))
```

## 帕斯卡（负二项）分布

考虑一个独立伯努利试验序列。

设 $p$ 为成功的概率。

设 $X$ 为在获得 $r$ 次成功之前失败的次数的随机变量。

其分布为

$$
\begin{aligned}
X  & \sim NB(r,p) \\
\textrm{Prob}(X=k;r,p) & = \begin{bmatrix}k+r-1 \\ r-1 \end{bmatrix}p^r(1-p)^{k}
\end{aligned}
$$

这里，我们从 $k+r-1$ 个可能的结果中选择，因为最后一次抽取根据定义必须是成功的。

我们计算得到均值和方差为

$$
\begin{aligned}
\mathbb{E}(X) & = \frac{k(1-p)}{p} \\
\mathbb{V}(X) & = \frac{k(1-p)}{p^2}
\end{aligned}
$$

```{code-cell} ipython3
# specify parameters
r, p, n = 10, 0.3, 1_000_000

# draw observations from the distribution
x = np.random.negative_binomial(r, p, n)

# compute sample mean and variance
μ_hat = np.mean(x)
σ2_hat = np.var(x)

print("The sample mean is: ", μ_hat, "\nThe sample variance is: ", σ2_hat)
print("\nThe population mean is: ", r*(1-p)/p)
print("The population variance is: ", r*(1-p)/p**2)
```

## 纽科姆-本福特分布

**纽科姆-本福特定律**适用于许多数据集，例如向税务机关报告的收入，其中首位数字更可能是小数而不是大数。

参见 <https://en.wikipedia.org/wiki/Benford%27s_law>

本福特概率分布为

$$
\textrm{Prob}\{X=d\}=\log _{10}(d+1)-\log _{10}(d)=\log _{10}\left(1+\frac{1}{d}\right)
$$

其中 $d\in\{1,2,\cdots,9\}$ 可以被视为数字序列中的**首位数字**。

这是一个定义明确的离散分布，因为我们可以验证概率是非负的且和为 $1$。

$$
\log_{10}\left(1+\frac{1}{d}\right)\geq0,\quad\sum_{d=1}^{9}\log_{10}\left(1+\frac{1}{d}\right)=1
$$

本福特分布的均值和方差为

$$
\begin{aligned}
\mathbb{E}\left[X\right]	 &=\sum_{d=1}^{9}d\log_{10}\left(1+\frac{1}{d}\right)\simeq3.4402 \\
\mathbb{V}\left[X\right]	 & =\sum_{d=1}^{9}\left(d-\mathbb{E}\left[X\right]\right)^{2}\log_{10}\left(1+\frac{1}{d}\right)\simeq6.0565
\end{aligned}
$$

我们使用`numpy`来验证上述结果并计算均值和方差。

```{code-cell} ipython3
Benford_pmf = np.array([np.log10(1+1/d) for d in range(1,10)])
k = np.arange(1, 10)

# mean
mean = k @ Benford_pmf

# variance
var = ((k - mean) ** 2) @ Benford_pmf

# verify sum to 1
print(np.sum(Benford_pmf))
print(mean)
print(var)
```

```{code-cell} ipython3
# 绘制分布图
plt.plot(range(1,10), Benford_pmf, 'o')
plt.title('本福特分布')
plt.show()
```

现在让我们来看看一些连续型随机变量。

## 一元高斯分布

我们用

$$
X \sim N(\mu,\sigma^2)
$$

来表示概率分布

$$f(x|u,\sigma^2)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{[-\frac{1}{2\sigma^2}(x-u)^2]} $$

在下面的例子中，我们设定 $\mu = 0, \sigma = 0.1$。

```{code-cell} ipython3
# 指定参数
μ, σ = 0, 0.1

# 指定抽样次数
n = 1_000_000

# 从分布中抽取观测值
x = np.random.normal(μ, σ, n)

# 计算样本均值和方差
μ_hat = np.mean(x)
σ_hat = np.std(x)

print("样本均值为：", μ_hat)
print("样本标准差为：", σ_hat)
```

```{code-cell} ipython3
# 比较
print(μ-μ_hat < 1e-3)
print(σ-σ_hat < 1e-3)
```

## 均匀分布

$$
\begin{aligned}
X & \sim U[a,b] \\
f(x)& = \begin{cases} \frac{1}{b-a}, & a \leq x \leq b \\ \quad0, & \text{其他}  \end{cases}
\end{aligned}
$$

总体均值和方差为

$$
\begin{aligned}
\mathbb{E}(X) & = \frac{a+b}{2} \\
\mathbb{V}(X) & = \frac{(b-a)^2}{12}
\end{aligned}
$$

```{code-cell} ipython3
# 指定参数
a, b = 10, 20

# 指定抽样次数
n = 1_000_000

# 从分布中抽取观测值
x = a + (b-a)*np.random.rand(n)

# 计算样本均值和方差
μ_hat = np.mean(x)
σ2_hat = np.var(x)

print("样本均值为：", μ_hat, "\n样本方差为：", σ2_hat)
print("\n总体均值为：", (a+b)/2)
print("总体方差为：", (b-a)**2/12)
```

##  混合离散-连续分布

让我们用一个小故事来说明这个例子。

假设你去参加一个工作面试,你要么通过要么失败。

你有5%的机会通过面试,而且你知道如果通过的话,你的日薪会在300~400之间均匀分布。

我们可以用以下概率来描述你的日薪这个离散-连续变量:

$$
P(X=0)=0.95
$$

$$
P(300\le X \le 400)=\int_{300}^{400} f(x)\, dx=0.05
$$

$$
f(x) = 0.0005
$$

让我们先生成一个随机样本并计算样本矩。

```{code-cell} ipython3
x = np.random.rand(1_000_000)
# x[x > 0.95] = 100*x[x > 0.95]+300
x[x > 0.95] = 100*np.random.rand(len(x[x > 0.95]))+300
x[x <= 0.95] = 0

μ_hat = np.mean(x)
σ2_hat = np.var(x)

print("样本均值是: ", μ_hat, "\n样本方差是: ", σ2_hat)
```

可以计算解析均值和方差：

$$
\begin{aligned}
\mu &= \int_{300}^{400}xf(x)dx \\
&= 0.0005\int_{300}^{400}xdx \\
&= 0.0005 \times \frac{1}{2}x^2\bigg|_{300}^{400}
\end{aligned}
$$

$$
\begin{aligned}
\sigma^2 &= 0.95\times(0-17.5)^2+\int_{300}^{400}(x-17.5)^2f(x)dx \\
&= 0.95\times17.5^2+0.0005\int_{300}^{400}(x-17.5)^2dx \\
&= 0.95\times17.5^2+0.0005 \times \frac{1}{3}(x-17.5)^3 \bigg|_{300}^{400}
\end{aligned}
$$

```{code-cell} ipython3
mean = 0.0005*0.5*(400**2 - 300**2)
var = 0.95*17.5**2+0.0005/3*((400-17.5)**3-(300-17.5)**3)
print("mean: ", mean)
print("variance: ", var)
```

## 从特定分布中抽取随机数

假设我们有一个可以生成均匀随机变量的伪随机数，即具有如下概率分布：

$$
\textrm{Prob}\{\tilde{X}=i\}=\frac{1}{I},\quad i=0,\ldots,I-1
$$

我们如何将$\tilde{X}$转换为随机变量$X$，使得$\textrm{Prob}\{X=i\}=f_i,\quad i=0,\ldots,I-1$，
其中$f_i$是在$i=0,1,\dots,I-1$上的任意离散概率分布？

关键工具是累积分布函数(CDF)的逆函数。

注意，分布的CDF是单调且非递减的，取值在$0$和$1$之间。

我们可以按以下方式抽取具有已知CDF的随机变量$X$的样本：

- 从$[0,1]$上的均匀分布中抽取随机变量$u$
- 将$u$的样本值代入目标CDF的**"逆函数"**得到$X$
- $X$具有目标CDF

因此，知道分布的**"逆"**CDF就足以从这个分布中进行模拟。

```{note}
这个方法要求"逆"CDF必须存在。
```

逆CDF定义为：

$$
F^{-1}(u)\equiv\inf \{x\in \mathbb{R}: F(x) \geq u\} \quad(0<u<1)
$$

这里我们使用下确界是因为CDF是非递减且右连续的函数。

因此，假设：

- $U$是一个均匀随机变量$U\in[0,1]$
- 我们想要采样CDF为$F$的随机变量$X$

事实证明，如果我们抽取均匀随机数$U$，然后通过以下方式计算$X$：

$$
X=F^{-1}(U),
$$

那么$X$是一个随机变量，其累积分布函数为$F_X(x)=F(x)=\textrm{Prob}\{X\le x\}$。

我们将在$F$连续且双射的特殊情况下验证这一点，因此其反函数存在，可以表示为$F^{-1}$。

注意到

$$
\begin{aligned}
F_{X}\left(x\right)	& =\textrm{Prob}\left\{ X\leq x\right\} \\
	& =\textrm{Prob}\left\{ F^{-1}\left(U\right)\leq x\right\} \\
	& =\textrm{Prob}\left\{ U\leq F\left(x\right)\right\} \\
	& =F\left(x\right)
\end{aligned}
$$

其中最后一个等式成立是因为$U$在$[0,1]$上均匀分布，而给定$x$时，$F(x)$是一个也位于$[0,1]$上的常数。

让我们用`numpy`来计算一些例子。

**例子：连续几何（指数）分布**

设$X$服从几何分布，参数为$\lambda>0$。

其密度函数为

$$
\quad f(x)=\lambda e^{-\lambda x}
$$

其累积分布函数为

$$
F(x)=\int_{0}^{\infty}\lambda e^{-\lambda x}=1-e^{-\lambda x}
$$

设$U$服从$[0,1]$上的均匀分布。

$X$是一个随机变量，满足$U=F(X)$。

可以从以下推导得出$X$的分布：

$$
\begin{aligned}
U& =F(X)=1-e^{-\lambda X}\qquad\\
\implies & \quad -U=e^{-\lambda X}\\
\implies&  \quad \log(1-U)=-\lambda X\\
\implies & \quad X=\frac{(1-U)}{-\lambda}
\end{aligned}
$$

让我们从$U[0,1]$中抽取$u$并计算$x=\frac{log(1-U)}{-\lambda}$。

我们将检验$X$是否似乎服从**连续几何**（指数）分布。

让我们用`numpy`来检验。

```{code-cell} ipython3
n, λ = 1_000_000, 0.3

# 生成均匀分布随机数
u = np.random.rand(n)

# 转换
x = -np.log(1-u)/λ

# 生成指数分布
x_g = np.random.exponential(1 / λ, n)

# 绘图并比较
plt.hist(x, bins=100, density=True)
plt.show()
```

```{code-cell} ipython3
plt.hist(x_g, bins=100, density=True, alpha=0.6)
plt.show()
```

**几何分布**

设 $X$ 服从几何分布，即

$$
\begin{aligned}
\textrm{Prob}(X=i) & =(1-\lambda)\lambda^i,\quad\lambda\in(0,1), \quad  i=0,1,\dots \\
 & \sum_{i=0}^{\infty}\textrm{Prob}(X=i)=1\longleftrightarrow(1- \lambda)\sum_{i=0}^{\infty}\lambda^i=\frac{1-\lambda}{1-\lambda}=1
\end{aligned}
$$

其累积分布函数为

$$
\begin{aligned}
\textrm{Prob}(X\le i)& =(1-\lambda)\sum_{j=0}^{i}\lambda^i\\
& =(1-\lambda)[\frac{1-\lambda^{i+1}}{1-\lambda}]\\
& =1-\lambda^{i+1}\\
& =F(X)=F_i \quad
\end{aligned}
$$

再次，设 $\tilde{U}$ 服从均匀分布，我们要找到满足 $F(X)=\tilde{U}$ 的 $X$。

让我们从以下推导 $X$ 的分布：

$$
\begin{aligned}
\tilde{U} & =F(X)=1-\lambda^{x+1}\\
1-\tilde{U} & =\lambda^{x+1}\\
\log(1-\tilde{U})& =(x+1)\log\lambda\\
\frac{\log(1-\tilde{U})}{\log\lambda}& =x+1\\
\frac{\log(1-\tilde{U})}{\log\lambda}-1 &=x
\end{aligned}
$$

然而，对于任何 $x\geq0$，$\tilde{U}=F^{-1}(X)$ 可能不是整数。

所以令

$$
x=\lceil\frac{\log(1-\tilde{U})}{\log\lambda}-1\rceil
$$

其中 $\lceil . \rceil$ 是向上取整函数。

因此 $x$ 是使离散几何累积分布函数大于或等于 $\tilde{U}$ 的最小整数。

我们可以通过以下 `numpy` 程序验证 $x$ 确实服从几何分布。

```{note}
指数分布是几何分布的连续模拟。
```

```{code-cell} ipython3
n, λ = 1_000_000, 0.8

# 生成均匀分布随机数
u = np.random.rand(n)

# 转换
x = np.ceil(np.log(1-u)/np.log(λ) - 1)

# 生成几何分布
x_g = np.random.geometric(1-λ, n)

# 绘图并比较
plt.hist(x, bins=150, density=True)
plt.show()
```


```{code-cell} ipython3
np.random.geometric(1-λ, n).max()
```

```{code-cell} ipython3
np.log(0.4)/np.log(0.3)
```

```{code-cell} ipython3
plt.hist(x_g, bins=150, density=True, alpha=0.6)
plt.show()
```

