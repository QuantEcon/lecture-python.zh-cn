---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 非共轭先验

本讲是{doc}`quantecon讲座 <prob_meaning>`的续篇。

那节课在似然函数和参数先验分布恰好形成**共轭**对的情况下，提供了概率的贝叶斯解释，其中：

- 应用贝叶斯法则产生的后验分布与先验具有相同的函数形式

具有共轭关系的似然和先验可以简化后验的计算，有助于进行解析或近似解析计算。

但在许多情况下，似然和先验不需要形成共轭对。

- 毕竟，一个人的先验是他或她自己的事情，只有在极小的巧合下才会采取与似然共轭的形式
在这些情况下，计算后验概率会变得非常具有挑战性。

在本讲中，我们将说明现代贝叶斯学者如何通过使用蒙特卡洛技术来处理非共轭先验，这涉及到：

- 首先巧妙地构建一个马尔可夫链，其不变分布就是我们想要的后验分布
- 模拟该马尔可夫链直到其收敛，然后从不变分布中采样以近似后验分布

我们将通过使用两个强大的Python模块来说明这种方法，这些模块实现了这种方法以及下面将要描述的另一种密切相关的方法。

这两个Python模块是：

- `numpyro`
- `pymc4`

像往常一样，我们首先导入一些Python代码。

```{code-cell} ipython3
:tags: [hide-output]

# install dependencies
!pip install numpyro pyro-ppl torch jax
```
```{code-cell} ipython3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from scipy.stats import binom
import scipy.stats as st
import torch

# jax
import jax.numpy as jnp
from jax import lax, random

# pyro
import pyro
from pyro import distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import MCMC, NUTS, SVI, ELBO, Trace_ELBO
from pyro.optim import Adam

# numpyro
import numpyro
from numpyro import distributions as ndist
import numpyro.distributions.constraints as nconstraints
from numpyro.infer import MCMC as nMCMC
from numpyro.infer import NUTS as nNUTS
from numpyro.infer import SVI as nSVI
from numpyro.infer import ELBO as nELBO
from numpyro.infer import Trace_ELBO as nTrace_ELBO
from numpyro.optim import Adam as nAdam
```
## 在二项分布似然上释放MCMC

本讲座从{doc}`quantecon讲座<prob_meaning>`中的二项分布示例开始。

该讲座通过以下方式计算后验分布：

- 通过选择共轭先验进行解析计算

本讲座则通过以下方式计算后验分布：

- 通过MCMC方法对后验分布进行数值采样，以及
- 使用变分推断(VI)近似

我们使用`pyro`和`numpyro`包，并借助`jax`来近似后验分布

我们使用几种不同的先验分布

我们将计算得到的后验分布与{doc}`quantecon讲座<prob_meaning>`中描述的共轭先验相关的后验分布进行比较


### 解析后验分布

假设随机变量$X\sim Binom\left(n,\theta\right)$。

这定义了一个似然函数

$$
L\left(Y\vert\theta\right) = \textrm{Prob}(X =  k | \theta) =
\left(\frac{n!}{k! (n-k)!} \right) \theta^k (1-\theta)^{n-k}
$$
其中 $Y=k$ 是一个观测数据点。

我们将 $\theta$ 视为一个随机变量，为其指定一个具有密度 $f(\theta)$ 的先验分布。

我们稍后会尝试其他先验分布，但现在，假设先验分布为 $\theta\sim Beta\left(\alpha,\beta\right)$，即：

$$
f(\theta) = \textrm{Prob}(\theta) = \frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}
$$

我们现在选择这个作为先验分布，是因为我们知道二项分布似然函数的共轭先验是贝塔分布。

在 $N$ 个样本观测中观察到 $k$ 次成功后，$\theta$ 的后验概率分布为：

$$
\textrm{Prob}(\theta|k) = \frac{\textrm{Prob}(\theta,k)}{\textrm{Prob}(k)}=\frac{\textrm{Prob}(k|\theta)\textrm{Prob}(\theta)}{\textrm{Prob}(k)}=\frac{\textrm{Prob}(k|\theta) \textrm{Prob}(\theta)}{\int_0^1 \textrm{Prob}(k|\theta)\textrm{Prob}(\theta) d\theta}
$$
=\frac{{N \choose k} (1 - \theta)^{N-k} \theta^k \frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}}{\int_0^1 {N \choose k} (1 - \theta)^{N-k} \theta^k\frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{B(\alpha, \beta)} d\theta}
$$

$$
=\frac{(1 -\theta)^{\beta+N-k-1} \theta^{\alpha+k-1}}{\int_0^1 (1 - \theta)^{\beta+N-k-1} \theta^{\alpha+k-1} d\theta} .
$$

因此，

$$
\textrm{Prob}(\theta|k) \sim {Beta}(\alpha + k, \beta+N-k)
$$

以下Python代码实现了给定共轭beta先验的解析后验。

```{code-cell} ipython3
def simulate_draw(theta, n):
    """
    生成一个大小为n的伯努利样本，其中P(Y=1) = theta
    """
    rand_draw = np.random.rand(n)
    draw = (rand_draw < theta).astype(int)
    return draw


def analytical_beta_posterior(data, alpha0, beta0):
    """
    给定观测数据，用参数(alpha, beta)的beta先验分布
    解析计算后验分布

    参数
    ---------
    num : int.
        计算后验时的观测数量
    alpha0, beta0 : float.
        beta先验分布的参数

    返回值
    ---------
    后验beta分布
    """
    num = len(data)
    up_num = data.sum()
    down_num = num - up_num
    return st.beta(alpha0 + up_num, beta0 + down_num)
```
### 近似后验分布的两种方法

假设我们没有共轭先验。

那么我们就无法解析地计算后验分布。

相反，我们使用计算工具来近似一组替代先验分布的后验分布，这需要用到Python中的`Pyro`和`Numpyro`包。

我们首先使用**马尔可夫链蒙特卡洛**（MCMC）算法。

我们实现NUTS采样器来从后验分布中采样。

通过这种方式，我们构建一个近似后验分布的采样分布。

在此之后，我们部署另一个称为**变分推断**（VI）的程序。

特别是，我们在`Pyro`和`Numpyro`中都实现了随机变分推断（SVI）机制。

MCMC算法据说能产生更准确的近似，因为原则上它直接从后验分布中采样。

但是它在计算上可能很昂贵，尤其是当维度很大时。
VI方法可能更便宜，但很可能会产生较差的后验近似，原因很简单，因为它需要猜测一个用于近似后验的参数化**指导函数形式**。

这个指导函数充其量也只能是一个不完美的近似。

通过限制假定后验具有受限函数形式所付出的代价，后验近似问题被转化为一个明确的优化问题，该问题寻求假定后验的参数，以最小化真实后验和假定后验分布之间的Kullback-Leibler (KL)散度。

  - 最小化KL散度等价于最大化一个称为**证据下界**（ELBO）的标准，我们很快就会验证这一点。

## 先验分布

为了能够应用MCMC采样或VI，`Pyro`和`Numpyro`要求先验分布满足特殊性质：
- 我们必须能够从中进行采样；
- 我们必须能够逐点计算对数概率密度函数；
- 概率密度函数必须对参数可微。

我们需要定义一个分布`class`。

我们将使用以下先验：

- 在区间$[\underline \theta, \overline \theta]$上的均匀分布，其中$0 \leq \underline \theta < \overline \theta \leq 1$。

- 支撑在$[0,1]$上的截断对数正态分布，参数为$(\mu,\sigma)$。

    - 要实现这一点，令$Z\sim Normal(\mu,\sigma)$且$\tilde{Z}$为支撑在$[\log(0),\log(1)]$上的截断正态分布，则$\exp(Z)$具有支撑在$[0,1]$上的对数正态分布。这很容易编码，因为`Numpyro`内置了截断正态分布，而`Torch`提供了包含指数变换的`TransformedDistribution`类。
- 另外，我们可以使用拒绝采样策略，将界限外的概率率设为$0$，并通过原始分布的CDF计算的总概率来重新缩放被接受的样本（即在界限内的实现值）。这可以通过使用`pyro`的`dist.Rejector`类来定义截断分布类来实现。

    - 我们在下面的部分实现这两种方法，并验证它们产生相同的结果。

- 一个支撑限制在$[0,1]$区间内的偏移冯·米塞斯分布，其参数为$(\mu,\kappa)$。

    - 设$X\sim vonMises(0,\kappa)$。我们知道$X$的支撑范围是$[-\pi, \pi]$。我们可以定义一个偏移的冯·米塞斯随机变量$\tilde{X}=a+bX$，其中$a=0.5, b=1/(2 \pi)$，这样$\tilde{X}$的支撑范围就在$[0,1]$上。

    - 这可以使用`Torch`的`TransformedDistribution`类及其`AffineTransform`方法来实现。
- 如果我们想要先验服从冯·米塞斯分布(von-Mises)且中心为$\mu=0.5$,我们可以选择一个较高的集中度参数$\kappa$,使得大部分概率质量位于$0$和$1$之间。然后我们可以使用上述策略进行截断。这可以通过`pyro`的`dist.Rejector`类来实现。在这种情况下,我们选择$\kappa > 40$。

- 一个截断的拉普拉斯分布。

    - 我们还考虑了截断的拉普拉斯分布,因为它的密度函数呈现分段非光滑的形式,并具有独特的尖峰形状。

    - 可以使用`Numpyro`的`TruncatedDistribution`类创建截断的拉普拉斯分布。

```{code-cell} ipython3
# 由Numpyro使用
def TruncatedLogNormal_trans(loc, scale):
    """
    使用numpyro的TruncatedNormal和ExpTransform获取截断对数正态分布
    """
    base_dist = ndist.TruncatedNormal(low=jnp.log(0), high=jnp.log(1), loc=loc, scale=scale)
    return ndist.TransformedDistribution(
        base_dist,ndist.transforms.ExpTransform()
        )

def ShiftedVonMises(kappa):
    """
    使用AffineTransform获取平移的冯·米塞斯分布
    """
    base_dist = ndist.VonMises(0, kappa)
    return ndist.TransformedDistribution(
        base_dist, ndist.transforms.AffineTransform(loc=0.5, scale=1/(2*jnp.pi))
        )

def TruncatedLaplace(loc, scale):
    """
    获取区间[0,1]上的截断拉普拉斯分布
    """
    base_dist = ndist.Laplace(loc, scale)
    return ndist.TruncatedDistribution(
        base_dist, low=0.0, high=1.0
    )

# 由Pyro使用
class TruncatedLogNormal(dist.Rejector):
    """
    通过Pyro中的拒绝采样定义截断对数正态分布
    """
    def __init__(self, loc, scale_0, upp=1):
        self.upp = upp
        propose = dist.LogNormal(loc, scale_0)

        def log_prob_accept(x):
            return (x < upp).type_as(x).log()

        log_scale = dist.LogNormal(loc, scale_0).cdf(torch.as_tensor(upp)).log()
        super(TruncatedLogNormal, self).__init__(propose, log_prob_accept, log_scale)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(0, self.upp)


class TruncatedvonMises(dist.Rejector):
    """
    通过Pyro中的拒绝采样定义截断冯·米塞斯分布
    """
    def __init__(self, kappa, mu=0.5, low=0.0, upp=1.0):
        self.low, self.upp = low, upp
        propose = dist.VonMises(mu, kappa)

        def log_prob_accept(x):
            return ((x > low) & (x < upp)).type_as(x).log()

        log_scale = torch.log(
            torch.tensor(
                st.vonmises(kappa=kappa, loc=mu).cdf(upp)
                - st.vonmises(kappa=kappa, loc=mu).cdf(low))
        )
        super(TruncatedvonMises, self).__init__(propose, log_prob_accept, log_scale)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.upp)
```
### 变分推断

变分推断方法不直接从后验分布中采样，而是用一族可处理的分布/密度来近似未知的后验分布。

然后，它寻求最小化近似分布与真实后验分布之间的统计差异度量。

因此，变分推断(VI)通过求解最小化问题来近似后验分布。

设我们要推断的潜在参数/变量为$\theta$。

设先验分布为$p(\theta)$，似然函数为$p\left(Y\vert\theta\right)$。

我们想要求得$p\left(\theta\vert Y\right)$。

根据贝叶斯法则：

$$
p\left(\theta\vert Y\right)=\frac{p\left(Y,\theta\right)}{p\left(Y\right)}=\frac{p\left(Y\vert\theta\right)p\left(\theta\right)}{p\left(Y\right)}
$$

其中

$$
p\left(Y\right)=\int d\theta p\left(Y\mid\theta\right)p\left(Y\right).
$$ (eq:intchallenge)

{eq}`eq:intchallenge`右侧的积分通常很难计算。
考虑一个由参数$\phi$参数化的**引导分布**$q_{\phi}(\theta)$，我们将用它来近似后验分布。

我们选择引导分布的参数$\phi$，以最小化近似后验分布$q_{\phi}(\theta)$与后验分布之间的Kullback-Leibler (KL)散度：

$$
 D_{KL}(q(\theta;\phi)\;\|\;p(\theta\mid Y)) \equiv -\int d\theta q(\theta;\phi)\log\frac{p(\theta\mid Y)}{q(\theta;\phi)}
$$

因此，我们需要一个能解决以下问题的**变分分布**$q$：

$$
\min_{\phi}\quad D_{KL}(q(\theta;\phi)\;\|\;p(\theta\mid Y))
$$

注意到：

$$
\begin{aligned}D_{KL}(q(\theta;\phi)\;\|\;p(\theta\mid Y)) & =-\int d\theta q(\theta;\phi)\log\frac{P(\theta\mid Y)}{q(\theta;\phi)}\\
 & =-\int d\theta q(\theta)\log\frac{\frac{p(\theta,Y)}{p(Y)}}{q(\theta)}\\
 & =-\int d\theta q(\theta)\log\frac{p(\theta,Y)}{p(\theta)q(Y)}\\
 & =-\int d\theta q(\theta)\left[\log\frac{p(\theta,Y)}{q(\theta)}-\log p(Y)\right]\\
$$
& =-\int d\theta q(\theta)\log\frac{p(\theta,Y)}{q(\theta)}+\int d\theta q(\theta)\log p(Y)\\
 & =-\int d\theta q(\theta)\log\frac{p(\theta,Y)}{q(\theta)}+\log p(Y)\\
\log p(Y)&=D_{KL}(q(\theta;\phi)\;\|\;p(\theta\mid Y))+\int d\theta q_{\phi}(\theta)\log\frac{p(\theta,Y)}{q_{\phi}(\theta)}
\end{aligned}
$$

对于观测数据$Y$，$p(\theta,Y)$是一个常数，所以最小化KL散度等价于最大化

$$
ELBO\equiv\int d\theta q_{\phi}(\theta)\log\frac{p(\theta,Y)}{q_{\phi}(\theta)}=\mathbb{E}_{q_{\phi}(\theta)}\left[\log p(\theta,Y)-\log q_{\phi}(\theta)\right]
$$ (eq:ELBO)

公式{eq}`eq:ELBO`被称为证据下界(ELBO)。

可以使用标准优化程序来搜索我们参数化分布$q_{\phi}(\theta)$中的最优$\phi$。

参数化分布$q_{\phi}(\theta)$被称为**变分分布**。
我们可以在Pyro和Numpyro中使用`Adam`梯度下降算法来实现随机变分推断(SVI)以近似后验分布。

我们使用两组变分分布：Beta分布和支撑在$[0,1]$上的截断正态分布

  - Beta分布的可学习参数是(alpha, beta)，两者都是正数。
  - 截断正态分布的可学习参数是(loc, scale)。

<u>我们将截断正态分布的'loc'参数限制在区间$[0,1]$内</u>。

## 实现

我们构建了一个Python类`BaysianInference`，初始化时需要以下参数：

- `param`：依赖于分布类型的参数元组/标量
- `name_dist`：指定分布名称的字符串

(`param`, `name_dist`)配对包括：
- ('beta', alpha, beta)

- ('uniform', upper_bound, lower_bound)

- ('lognormal', loc, scale)
   - 注意：这是截断的对数正态分布。
- ('vonMises', kappa)，其中kappa表示集中参数，中心位置设为$0.5$。
   - 注意：在使用`Pyro`时，这是原始vonMises分布的截断版本；
   - 注意：在使用`Numpyro`时，这是**平移后**的分布。

- ('laplace', loc, scale)
   - 注意：这是截断的拉普拉斯分布

类`BaysianInference`有几个关键方法：
- `sample_prior`:
   - 可用于从给定的先验分布中抽取单个样本。

- `show_prior`:
   - 通过重复抽样并拟合核密度曲线来绘制近似的先验分布。

- `MCMC_sampling`:
   - 输入：(data, num_samples, num_warmup=1000)
   - 接收一个`np.array`数据并生成大小为`num_samples`的后验MCMC采样。

- `SVI_run`:
  - 输入：(data, guide_dist, n_steps=10000)
  - guide_dist = 'normal' - 使用**截断的**正态分布作为参数化的guide
- guide_dist = 'beta' - 使用beta分布作为参数化的指导分布
  - 返回值: (params, losses) - 以`dict`形式存储的学习参数和每一步的损失向量。

```{code-cell} ipython3
class BayesianInference:
    def __init__(self, param, name_dist, solver):
        """
        参数
        ---------
        param : tuple.
            包含分布所有相关参数的元组对象
        dist : str.
            分布的名称 - 'beta', 'uniform', 'lognormal', 'vonMises', 'tent'
        solver : str.
            pyro或numpyro
        """
        self.param = param
        self.name_dist = name_dist
        self.solver = solver

        # jax需要显式传入PRNG状态
        self.rng_key = random.PRNGKey(0)


    def sample_prior(self):
        """
        定义在Pyro/Numpyro模型中用于采样的先验分布。
        """
        if self.name_dist=='beta':
            # 解包参数
            alpha0, beta0 = self.param
            if self.solver=='pyro':
                sample = pyro.sample('theta', dist.Beta(alpha0, beta0))
            else:
                sample = numpyro.sample('theta', ndist.Beta(alpha0, beta0), rng_key=self.rng_key)

        elif self.name_dist=='uniform':
            # 解包参数
            lb, ub = self.param
            if self.solver=='pyro':
                sample = pyro.sample('theta', dist.Uniform(lb, ub))
            else:
                sample = numpyro.sample('theta', ndist.Uniform(lb, ub), rng_key=self.rng_key)

        elif self.name_dist=='lognormal':
            # 解包参数
            loc, scale = self.param
            if self.solver=='pyro':
                sample = pyro.sample('theta', TruncatedLogNormal(loc, scale))
            else:
                sample = numpyro.sample('theta', TruncatedLogNormal_trans(loc, scale), rng_key=self.rng_key)

        elif self.name_dist=='vonMises':
            # 解包参数
            kappa = self.param
            if self.solver=='pyro':
                sample = pyro.sample('theta', TruncatedvonMises(kappa))
            else:
                sample = numpyro.sample('theta', ShiftedVonMises(kappa), rng_key=self.rng_key)

        elif self.name_dist=='laplace':
            # 解包参数
            loc, scale = self.param
            if self.solver=='pyro':
                print("警告：请使用Numpyro进行截断拉普拉斯分布。")
                sample = None
            else:
                sample = numpyro.sample('theta', TruncatedLaplace(loc, scale), rng_key=self.rng_key)

        return sample


    def show_prior(self, size=1e5, bins=20, disp_plot=1):
        """
        通过从先验分布采样并绘制近似采样分布来可视化先验分布
        """
        self.bins = bins

        if self.solver=='pyro':
            with pyro.plate('show_prior', size=size):
                sample = self.sample_prior()
            # 转换为numpy
            sample_array = sample.numpy()

        elif self.solver=='numpyro':
            with numpyro.plate('show_prior', size=size):
                sample = self.sample_prior()
            # 转换为numpy
            sample_array=jnp.asarray(sample)

        # 绘制直方图和核密度估计
        if disp_plot==1:
            sns.displot(sample_array, kde=True, stat='density', bins=bins, height=5, aspect=1.5)
            plt.xlim(0, 1)
            plt.show()
        else:
            return sample_array


    def model(self, data):
        """
        通过指定先验分布、条件似然和数据条件来定义概率模型
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        # 设置先验
        theta = self.sample_prior()

        # 从条件似然中采样
        if self.solver=='pyro':
            output = pyro.sample('obs', dist.Binomial(len(data), theta), obs=torch.sum(data))
        else:
            # 注意：numpyro.sample()要求obs=np.ndarray
            output = numpyro.sample('obs', ndist.Binomial(len(data), theta), obs=torch.sum(data).numpy())
        return output


    def MCMC_sampling(self, data, num_samples, num_warmup=1000):
        """
        使用MCMC数值计算给定数据下的后验分布，先验为由(alpha0, beta0)参数化的beta分布
        """
        # 使用pyro
        if self.solver=='pyro':
            # 张量化
            data = torch.tensor(data)
            nuts_kernel = NUTS(self.model)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_warmup, disable_progbar=True)
            mcmc.run(data)

        # 使用numpyro
        elif self.solver=='numpyro':
            data = np.array(data, dtype=float)
            nuts_kernel = nNUTS(self.model)
            mcmc = nMCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=False)
            mcmc.run(self.rng_key, data=data)

        # 收集样本
        samples = mcmc.get_samples()['theta']
        return samples


    def beta_guide(self, data):
        """
        定义用于在Pyro/Numpyro中近似后验的候选参数化变分分布
        这里我们使用参数化beta分布
        """
        if self.solver=='pyro':
            alpha_q = pyro.param('alpha_q', torch.tensor(0.5),
                            constraint=constraints.positive)
            beta_q = pyro.param('beta_q', torch.tensor(0.5),
                            constraint=constraints.positive)
            pyro.sample('theta', dist.Beta(alpha_q, beta_q))

        else:
            alpha_q = numpyro.param('alpha_q', 10,
                            constraint=nconstraints.positive)
            beta_q = numpyro.param('beta_q', 10,
                            constraint=nconstraints.positive)

            numpyro.sample('theta', ndist.Beta(alpha_q, beta_q))


    def truncnormal_guide(self, data):
        """
        定义用于在Pyro/Numpyro中近似后验的候选参数化变分分布
        这里我们使用[0,1]上的截断正态分布
        """
        loc = numpyro.param('loc', 0.5,
                        constraint=nconstraints.interval(0.0, 1.0))
        scale = numpyro.param('scale', 1,
                        constraint=nconstraints.positive)
        numpyro.sample('theta', ndist.TruncatedNormal(loc, scale, low=0.0, high=1.0))


    def SVI_init(self, guide_dist, lr=0.0005):
        """
        使用Adam优化器初始化SVI训练模式
        注意：truncnormal_guide只能与numpyro求解器一起使用
        """
        adam_params = {"lr": lr}

        if guide_dist=='beta':
            if self.solver=='pyro':
                optimizer = Adam(adam_params)
                svi = SVI(self.model, self.beta_guide, optimizer, loss=Trace_ELBO())

            elif self.solver=='numpyro':
                optimizer = nAdam(step_size=lr)
                svi = nSVI(self.model, self.beta_guide, optimizer, loss=nTrace_ELBO())

        elif guide_dist=='normal':
            # 仅允许numpyro
            if self.solver=='pyro':
                print("警告：请使用Numpyro和TruncatedNormal指导")
                svi = None

            elif self.solver=='numpyro':
                optimizer = nAdam(step_size=lr)
                svi = nSVI(self.model, self.truncnormal_guide, optimizer, loss=nTrace_ELBO())
        else:
            print("警告：请输入'beta'或'normal'")
            svi = None

        return svi

    def SVI_run(self, data, guide_dist, n_steps=10000):
        """
        运行SVI并返回优化后的参数和损失

        返回值
        --------
        params : 指导分布的学习参数
        losses : 每一步的损失向量
        """

        # 初始化SVI
        svi = self.SVI_init(guide_dist=guide_dist)

        # 执行梯度步骤
        if self.solver=='pyro':
             # 张量化数据
            if not torch.is_tensor(data):
                data = torch.tensor(data)
            # 存储损失向量
            losses = np.zeros(n_steps)
            for step in range(n_steps):
                losses[step] = svi.step(data)

            # pyro仅支持beta VI分布
            params = {
                'alpha_q': pyro.param('alpha_q').item(),
                'beta_q': pyro.param('beta_q').item()
                }

        elif self.solver=='numpyro':
            data = np.array(data, dtype=float)
            result = svi.run(self.rng_key, n_steps, data, progress_bar=False)
            params = dict(
                (key, np.asarray(value)) for key, value in result.params.items()
                )
            losses = np.asarray(result.losses)

        return params, losses
```
