---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 卡尔曼滤波器进阶
  headings:
    A worker's output: 劳动者的产出
    A firm's wage-setting policy: 公司的工资设定政策
    A state-space representation: 状态空间表示
    An innovations representation: 创新表示
    Some computational experiments: 一些计算实验
    Future extensions: 未来扩展
---

(kalman_2)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 卡尔曼滤波器进阶

```{index} single: Kalman Filter 2
```

```{contents} Contents
:depth: 2
```

在 {doc}`kalman` 中，我们使用卡尔曼滤波器来估计火箭的位置。

在本讲座中，我们将使用卡尔曼滤波器来推断劳动者的人力资本，以及劳动者投入人力资本积累的努力程度，这两个变量都是公司无法直接观察到的。

本讲座是对 {doc}`kalman` 中介绍的滤波与预测递归的一个应用。

公司只能通过观察劳动者历史产出，以及理解这些产出如何依赖于劳动者的人力资本，以及人力资本如何作为劳动者努力程度的函数来演化，来了解上述变量。

我们将设定一个规则，说明公司如何根据每期获得的信息来支付劳动者工资。

{doc}`intermediate:kalman_filter_var` 这一讲座使用相同的递归来构建新息（innovations）、似然函数和向量自回归模型。

除了Anaconda中的内容外，本讲座还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

为了进行模拟，我们引入以下函数库，与 {doc}`kalman` 相同：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

import numpy as np
from quantecon import Kalman, LinearStateSpace
from collections import namedtuple
from scipy.stats import multivariate_normal
```

## 劳动者的产出

一个代表性劳动者永久受雇于一家公司。

劳动者的产出由以下动态过程描述：

```{math}
:label: worker_model

\begin{aligned}
h_{t+1} &= \alpha h_t + \beta u_t + c \epsilon_{t+1}, \quad \epsilon_{t+1} \sim N(0,1) \\
u_{t+1} & = u_t \\
y_t & = g h_t + v_t , \quad v_t \sim N(0, R)
\end{aligned}
```

其中：

* $h_t$ 是时间 $t$ 时的人力资本对数
* $u_t$ 是时间 $t$ 时劳动者投入人力资本积累的努力程度的对数
* $y_t$ 是时间 $t$ 时劳动者产出的对数
* $\epsilon_{t+1}$ 是人力资本上一个独立同分布的标准正态冲击
* $h_0 \sim N(\hat h_0, \sigma_{h,0}^2)$
* $u_0 \sim N(\hat u_0, \sigma_{u,0}^2)$

模型的参数是 $\alpha, \beta, c, R, g, \hat h_0, \hat u_0, \sigma_{h,0}, \sigma_{u,0}$，其中 $\sigma_{h,0}$ 和 $\sigma_{u,0}$ 是公司关于 $h_0$ 和 $u_0$ 的初始信念的标准差。

我们假设 $h_0$、$u_0$、$\{\epsilon_t\}$ 和 $\{v_t\}$ 是相互独立的。

在时间 $0$，公司雇佣了劳动者。

劳动者永久依附于公司，因此在所有时间 $t =0, 1, 2, \ldots$ 都为同一家公司工作。

在时间 $0$ 开始时，公司既无法观察到劳动者天生的初始人力资本 $h_0$，也无法观察到其固有的永久努力水平 $u_0$。

公司认为特定劳动者的 $u_0$ 服从高斯概率分布，因此由 $u_0 \sim N(\hat u_0, \sigma_{u,0}^2)$ 描述。

劳动者"类型"中的 $h_t$ 部分随时间变化，而方程 $u_{t+1} = u_t$ 意味着对所有 $t$ 都有 $u_t = u_0$。

因此，从公司的角度来看，努力程度是劳动者类型中一个固定的、不可观测的组成部分，必须从产出观测值中加以推断。

在时间 $t \geq 1$ 开始时，在设定工资 $w_t$ 之前，公司已经观察到历史记录 $y^{t-1} = [y_{t-1}, y_{t-2}, \ldots, y_0]$。

公司无法观察劳动者的"类型" $(h_0, u_0)$。

在第 $t$ 期生产之后，公司观察到劳动者的产出 $y_t$，然后用它来更新进入第 $t+1$ 期的信念。

## 公司的工资设定政策

在时间 $t \geq 1$，在观察到当期产出 $y_t$ 之前，公司使用过去的产出历史 $y^{t-1}$ 来设定劳动者的对数工资：

$$
w_t = g \mathbb{E}[h_t | y^{t-1}], \quad t \geq 1
$$

而在时间 $0$，公司支付给劳动者的对数工资等于 $y_0$ 的无条件均值：

$$
w_0 = g \hat h_0
$$

在使用这一支付规则时，公司考虑到劳动者当期的对数产出部分源于纯粹由运气决定的随机成分 $v_t$，且假设该成分与 $h_t$ 和 $u_t$ 相互独立。

## 状态空间表示

将系统 [](worker_model) 写成状态空间形式：

```{math}
\begin{aligned}
\begin{bmatrix} h_{t+1} \cr u_{t+1} \end{bmatrix} &= \begin{bmatrix} \alpha & \beta \cr 0 & 1 \end{bmatrix}\begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + \begin{bmatrix} c \cr 0 \end{bmatrix} \epsilon_{t+1} \cr
y_t & = \begin{bmatrix} g & 0 \end{bmatrix} \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + v_t
\end{aligned}
```

这等价于：

```{math}
:label: ssrepresent
\begin{aligned} 
x_{t+1} & = A x_t + C \epsilon_{t+1} \cr
y_t & = G x_t + v_t \cr
x_0 & \sim N(\hat x_0, \Sigma_0) 
\end{aligned}
```

其中：

```{math}
x_t  = \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} , \quad
\hat x_0  = \begin{bmatrix} \hat h_0 \cr \hat u_0 \end{bmatrix} , \quad
\Sigma_0  = \begin{bmatrix} \sigma_{h,0}^2 & 0 \cr
                     0 & \sigma_{u,0}^2 \end{bmatrix}
```

为了计算公司的工资设定政策，我们首先创建一个 `namedtuple` 来存储模型的参数：

```{code-cell} ipython3
WorkerModel = namedtuple("WorkerModel", 
                ('A', 'C', 'G', 'R', 'xhat_0', 'Σ_0'))

def create_worker(α=.8, β=.2, c=.2,
                  R=.5, g=1.0, hhat_0=4, uhat_0=4, 
                  σ_h=2, σ_u=2):
    
    A = np.array([[α, β], 
                  [0, 1]])
    C = np.array([[c], 
                  [0]])
    G = np.array([g, 0])

    # 定义初始状态和协方差矩阵
    xhat_0 = np.array([[hhat_0], 
                       [uhat_0]])
    
    # σ_h 和 σ_u 是标准差，因此 Σ_0 保存的是它们的平方
    Σ_0 = np.array([[σ_h**2, 0],
                    [0, σ_u**2]])
    
    return WorkerModel(A=A, C=C, G=G, R=R, xhat_0=xhat_0, Σ_0=Σ_0)
```

请注意 `WorkerModel` namedtuple 是如何创建计算相应状态空间表示 {eq}`ssrepresent` 所需的所有对象的。

这十分方便，因为为了模拟劳动者的历史 $\{y_t, h_t\}$，我们需要使用 [`LinearStateSpace`](https://quanteconpy.readthedocs.io/en/latest/tools/lss.html) 类为其构建状态空间系统。

```{code-cell} ipython3
# 定义 A, C, G, R, xhat_0, Σ_0
worker = create_worker()
A, C, G, R = worker.A, worker.C, worker.G, worker.R
xhat_0, Σ_0 = worker.xhat_0, worker.Σ_0

# 创建 LinearStateSpace 对象
ss = LinearStateSpace(A, C, G, np.sqrt(R), 
        mu_0=xhat_0, Sigma_0=np.zeros((2,2)))

T = 100
x, y = ss.simulate(T)
y = y.flatten()

h_0, u_0 = x[0, 0], x[1, 0]
```

我们设定 `Sigma_0=np.zeros((2,2))`，使得模拟过程固定某个特定劳动者的初始状态 $(h_0, u_0)$，而公司在进入第 $0$ 期时，仍然持有驱动其卡尔曼滤波器的非退化先验信念 $\hat x_0$ 和 $\Sigma_0$。

接下来，为了计算公司基于其获得的关于劳动者的信息来设定对数工资的政策，我们使用本量化经济学讲座 {doc}`kalman` 中描述的卡尔曼滤波器。

特别是，我们想要计算"创新表示"中的所有对象。

## 创新表示

我们已经掌握了形成劳动者产出过程 $\{y_t\}_{t=0}^{T-1}$ 的创新表示所需的所有对象。

让我们现在编写代码：

```{math}
\begin{aligned}
\hat x_{t+1} & = A \hat x_t + K_t a_t \cr
y_{t} & = G \hat x_t + a_t
\end{aligned}
```
其中 $\hat x_t = \mathbb{E}[x_t | y^{t-1}]$ 是公司在观察到 $y_t$ 之前形成的对状态的预测，而 $K_t$ 是时间 $t$ 的卡尔曼增益矩阵。

这里 $a_t = y_t - G \hat x_t$ 是时间 $t$ 的**新息（innovation）**，即公司根据历史记录 $y^{t-1}$ 对产出 $y_t$ 进行一步预测所产生的误差。

由于 $\hat x_t$ 是以 $y^{t-1}$（而非 $y^t$）为条件的，增益 $K_t$ 将利用当期观测值 $y_t$ 的滤波更新，与将状态推进到 $t+1$ 期的一步预测，两者合而为一。

记 $\Sigma_t = \mathbb{E}[(x_t - \hat x_t)(x_t - \hat x_t)' | y^{t-1}]$ 为状态的条件协方差，则增益为

```{math}
K_t = A \Sigma_t G' (G \Sigma_t G' + R)^{-1} = A L_t ,
```

其中 $L_t = \Sigma_t G' (G \Sigma_t G' + R)^{-1}$ 是滤波增益，用于在观察到 $y_t$ 之后更新公司关于 $x_t$ 的信念。

我们使用 [`Kalman`](https://quanteconpy.readthedocs.io/en/latest/tools/kalman.html) 类在以下代码中完成这个任务：

```{code-cell} ipython3
kalman = Kalman(ss, xhat_0, Σ_0)
Σ_t = np.zeros((*Σ_0.shape, T))
y_hat_t = np.zeros(T)
x_hat_t = np.zeros((2, T))

for t in range(T):
    # 记录公司在看到 y_t 之前，基于 y^{t-1} 对 x_t 的信念
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t[:, :, t] = Σ
    x_hat_t[:, t] = x_hat.reshape(-1)
    y_hat_t[t] = (worker.G @ x_hat).item()
    
    # 然后纳入观测值 y_t，并将滤波器推进到 t+1 期
    kalman.update(y[t])

u_hat_t = x_hat_t[1, :]
```

对于这个固定的劳动者初始状态，我们绘制 $\mathbb{E}[y_t | y^{t-1}] = G \hat x_t$，其中 $\hat x_t = \mathbb{E}[x_t | y^{t-1}]$。

我们还绘制 $\mathbb{E}[u_0 | y^{t-1}]$，这是公司基于其在第 $t$ 期获得的关于该劳动者的信息 $y^{t-1}$，对劳动者固有的"工作伦理" $u_0$ 所做出的推断。

我们可以观察随着更多产出观测值的到来，公司如何更新它对劳动者工作伦理的推断 $\mathbb{E}[u_0 | y^{t-1}]$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 公司的产出预测与随时间推移推断出的工作伦理
    name: fig-kalman2-inference
---
fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$\mathbb{E}[y_t| y^{t-1}]$')
ax[0].set_xlabel('时间')
ax[0].set_ylabel(r'$\mathbb{E}[y_t| y^{t-1}]$')
ax[0].set_title(r'$\mathbb{E}[y_t| y^{t-1}]$ 随时间变化')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$\mathbb{E}[u_0|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', 
            linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('时间')
ax[1].set_ylabel(r'$\mathbb{E}[u_0|y^{t-1}]$')
ax[1].set_title('推断的工作伦理随时间变化')
ax[1].legend()

fig.tight_layout()
plt.show()
```

## 一些计算实验

让我们看看 $\Sigma_0$ 和 $\Sigma_{T-1}$，以观察在我们设定的时间范围内，公司对隐藏状态了解了多少。

```{code-cell} ipython3
print(Σ_t[:, :, 0])
```

```{code-cell} ipython3
print(Σ_t[:, :, -1])
```

显然，条件方差随时间变小。

通过在不同时间 $t$ 绘制给定 $y^{t-1}$ 条件下 $x_t$ 的条件二元正态密度的等高线，我们可以形象地展示公司的条件信念是如何演化的。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 三个时间点上公司关于 $x_t$ 的信念密度等高线
    name: fig-kalman2-contours
---
# 创建用于等高线绘制的点网格
h_range = np.linspace(x_hat_t[0, :].min()-0.5*Σ_t[0, 0, 1], 
                      x_hat_t[0, :].max()+0.5*Σ_t[0, 0, 1], 100)
u_range = np.linspace(x_hat_t[1, :].min()-0.5*Σ_t[1, 1, 1], 
                      x_hat_t[1, :].max()+0.5*Σ_t[1, 1, 1], 100)
h, u = np.meshgrid(h_range, u_range)

# 为每个时间步创建子图
fig, axs = plt.subplots(1, 3, figsize=(12, 7))

# 遍历每个时间步
for i, t in enumerate(np.linspace(0, T-1, 3, dtype=int)):
    # 创建具有时间步 t 的 x_hat 和 Σ 的多变量正态分布
    μ = x_hat_t[:, t]
    cov = Σ_t[:, :, t]
    mvn = multivariate_normal(mean=μ, cov=cov)
    
    # 在网格上评估多变量正态 PDF
    pdf_values = mvn.pdf(np.dstack((h, u)))
    
    # 创建 PDF 的等高线图
    con = axs[i].contour(h, u, pdf_values, cmap='viridis')
    axs[i].clabel(con, inline=1, fontsize=10)
    axs[i].set_title('时间步'+f' {t}')
    axs[i].set_xlabel(r'$h_{{{}}}$'.format(str(t)))
    axs[i].set_ylabel(r'$u_{{{}}}$'.format(str(t)))
    
    cov_latex = (
        r'$\Sigma_{{{}}}= \begin{{bmatrix}} {:.2f} & {:.2f} \\ '
        r'{:.2f} & {:.2f} \end{{bmatrix}}$'
    ).format(t, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1])
    axs[i].text(0.33, -0.15, cov_latex, transform=axs[i].transAxes)

    
plt.tight_layout()
plt.show()
```

注意，随着样本量 $t$ 的增长，证据 $y^{t-1}$ 的积累是如何影响密度等高线的形状的。

现在让我们使用我们的代码将隐藏状态 $x_0$ 设置为特定的向量，以观察公司如何从我们感兴趣的某个 $x_0$ 开始学习。

例如，让我们设 $h_0 = 0$ 和 $u_0 = 4$。

这是实现这个例子的一种方式：

```{code-cell} ipython3
# 例如，我们可能想要 h_0 = 0 和 u_0 = 4
μ_0 = np.array([[0.0],
                [4.0]])

# 创建一个 LinearStateSpace 对象，其中 Sigma_0 为零矩阵
ss_example = LinearStateSpace(A, C, G, np.sqrt(R), mu_0=μ_0, 
                              # 这行强制 h_0=0 和 u_0=4
                              Sigma_0=np.zeros((2, 2))
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# 现在 h_0=0 和 u_0=4
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)
```

实现相同目标的另一种方式是使用以下代码：

```{code-cell} ipython3
# 如果我们想要设置初始 
# h_0 = hhat_0 = 0.0 和 u_0 = uhat_0 = 4.0:
worker_example = create_worker(hhat_0=0.0, uhat_0=4.0)

# 公司的先验仍然保持在原来的 xhat_0 和 Σ_0
ss_example = LinearStateSpace(A, C, G, np.sqrt(R), 
                              # 这行取 h_0=hhat_0 和 u_0=uhat_0
                              mu_0=worker_example.xhat_0,
                              # 这行强制 h_0=hhat_0 和 u_0=uhat_0
                              Sigma_0=np.zeros((2, 2))
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# 现在 h_0 和 u_0 将精确等于 hhat_0 和 uhat_0
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)
```

对于这个劳动者，让我们生成一个类似上面的图：

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "$u_0=4$ 的劳动者的产出预测与推断出的工作伦理"
    name: fig-kalman2-worker
---
# 公司使用其先验信念 xhat_0 和 Σ_0 对 ss_example 的产出进行滤波
kalman = Kalman(ss_example, xhat_0, Σ_0)
Σ_t = []
y_hat_t = np.zeros(T)
u_hat_t = np.zeros(T)

# 然后我们使用基于上述线性状态模型的观测值 y 
# 迭代更新卡尔曼滤波器类：
for t in range(T):
    # 记录公司在看到 y_t 之前，基于 y^{t-1} 对 x_t 的信念
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t.append(Σ)
    y_hat_t[t] = (G @ x_hat).item()
    u_hat_t[t] = x_hat[1].item()

    # 然后纳入观测值 y_t，并将滤波器推进到 t+1 期
    kalman.update(y[t])


# 生成 y_hat_t 和 u_hat_t 的图
fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$\mathbb{E}[y_t| y^{t-1}]$')
ax[0].set_xlabel('时间')
ax[0].set_ylabel(r'$\mathbb{E}[y_t| y^{t-1}]$')
ax[0].set_title(r'$\mathbb{E}[y_t| y^{t-1}]$ 随时间变化')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$\mathbb{E}[u_0|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', 
            linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('时间')
ax[1].set_ylabel(r'$\mathbb{E}[u_0|y^{t-1}]$')
ax[1].set_title('推断的工作伦理随时间变化')
ax[1].legend()

fig.tight_layout()
plt.show()
```

更一般地，我们可以在 `create_worker` namedtuple 中更改定义劳动者的部分或全部参数。

这是一个例子：

```{code-cell} ipython3
# 我们可以在创建劳动者时设置这些参数 -- 就像类一样！
hard_working_worker = create_worker(α=.4, β=.8, 
                        hhat_0=7.0, uhat_0=100, σ_h=2.5, σ_u=3.2)

print(hard_working_worker)
```

我们还可以为不同的劳动者模拟这个系统 $T = 100$ 期。

当努力通过人力资本影响产出时，即 $g \neq 0$ 且 $\beta \neq 0$ 时，公司对 $u_0$ 的不确定性会随时间下降，推断出的工作伦理会收敛到真实的 $u_0$。

如果 $\beta = 0$，努力永远不会影响 $h_t$；如果 $g = 0$，产出不携带关于 $h_t$ 的任何信息，那么在这两种情况下，公司都无法仅从产出中学习到 $u_0$。

这说明，在这些可观测性条件下，滤波器会逐渐让公司了解到劳动者的努力程度。

```{code-cell} ipython3
:tags: [hide-input]

def simulate_workers(worker, T, ax, μ_sim_0=None, Σ_sim_0=None, 
                    diff=True, name=None, random_state=None):
    A, C, G, R = worker.A, worker.C, worker.G, worker.R
    xhat_0, Σ_prior = worker.xhat_0, worker.Σ_0
    
    # μ_sim_0 和 Σ_sim_0 设定被模拟劳动者的初始状态，而
    # xhat_0 和 Σ_prior 是滤波器中公司的先验信念
    if μ_sim_0 is None:
        μ_sim_0 = xhat_0
    if Σ_sim_0 is None:
        Σ_sim_0 = Σ_prior
        
    ss = LinearStateSpace(A, C, G, np.sqrt(R), 
                        mu_0=μ_sim_0, Sigma_0=Σ_sim_0)

    x, y = ss.simulate(T, random_state=random_state)
    y = y.flatten()

    u_0 = x[1, 0]
    
    # 计算卡尔曼滤波器
    kalman = Kalman(ss, xhat_0, Σ_prior)
    Σ_t = []
    
    y_hat_t = np.zeros(T)
    u_hat_t = np.zeros(T)

    for i in range(T):
        # 记录公司在看到 y_i 之前，基于 y^{i-1} 对 x_i 的信念
        x_hat, Σ = kalman.x_hat, kalman.Sigma
        Σ_t.append(Σ)
        y_hat_t[i] = (worker.G @ x_hat).item()
        u_hat_t[i] = x_hat[1].item()

        # 然后纳入观测值 y_i，并推进滤波器
        kalman.update(y[i])

    if diff :
        ax.plot(u_hat_t - u_0, alpha=.5, label=name)
        ax.axhline(y=0, color='grey', linestyle='dashed')
        ax.set_xlabel('时间')
        ax.set_ylabel(r'$\mathbb{E}[u_0|y^{t-1}] - u_0$')
        
    else:
        label_line = (r'$\mathbb{E}[u_0|y^{t-1}]$' if name is None 
                      else name)
        
        u_hat_plot = ax.plot(u_hat_t, label=label_line)
        ax.axhline(y=u_0, color=u_hat_plot[0].get_color(), 
                    linestyle='dashed', alpha=0.5)
        ax.set_xlabel('时间')
        ax.set_ylabel(r'$\mathbb{E}[u_0|y^{t-1}]$')
```

对于三位劳动者，我们首先绘制公司推断出的工作伦理与真实 $u_0$ 之间的差距随时间的变化。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 推断出的工作伦理与真实工作伦理之间的差异随时间的变化
    name: fig-kalman2-diff
---
num_workers = 3
T = 100
fig, ax = plt.subplots(figsize=(7, 7))

for i in range(num_workers):
    worker = create_worker(uhat_0=4+2*i)
    simulate_workers(worker, T, ax, name=fr'$\hat u_0 = {4+2*i}$',
                     random_state=2 + i)
ax.set_ylim(ymin=-2, ymax=2)
ax.legend()
plt.show()
```

在这个模拟中，公司推断出的工作伦理逐渐趋向真实的 $u_0$。

在正确设定的可观测线性-高斯模型下，$u_0$ 的后验均值是一致的，因此随着产出历史的增长，这一差距会逐渐缩小。

通过设置 `diff=False`，我们转而绘制每位劳动者推断出的工作伦理水平 $\mathbb{E}[u_0|y^{t-1}]$，并附上一条表示真实 $u_0$ 的虚线。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 三位劳动者随时间推断出的工作伦理
    name: fig-kalman2-three
---
fig, ax = plt.subplots(figsize=(7, 7))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.2, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax, diff=False, 
                     name=r'$u_{{{}, 0}}$'.format(i),
                     random_state=3 + i)

ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

这三位劳动者不仅 $\hat u_0$ 不同，$\alpha$ 和 $\beta$ 也不同，他们学习的速度也大相径庭。

$\beta$ 最大的劳动者（此处为 $\beta = 0.9$ 的 $u_{1,0}$）几乎立即收敛到其真实值的虚线上，而 $\beta$ 最小的劳动者（此处为 $\beta = 0.2$ 的 $u_{0,0}$）收敛得更慢。

原因在于努力仅通过人力资本影响产出，因此在这些 $|\alpha| < 1$ 的稳定例子中，其对产出的稳态影响由 $g \beta / (1 - \alpha)$ 决定，而较小的 $\beta$ 使公司在此时间范围内获得的信号太少，不足以准确确定 $u_0$。

学习速度也反映了测量噪声 $R$、冲击尺度 $c$，以及公司的先验方差。

我们还可以通过向 `simulate_workers` 传入一个固定的 `μ_sim_0` 和一个零矩阵 `Σ_sim_0`，让每位劳动者拥有相同的真实初始状态，这里设为 $h_0=2$ 和 $u_0=1$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 当每位劳动者都从 $h_0=2$ 和 $u_0=1$ 开始时推断出的工作伦理
    name: fig-kalman2-exact
---
fig, ax = plt.subplots(figsize=(7, 7))

μ_sim_0 = np.array([[2.0],
                    [1.0]])
Σ_sim_0 = np.zeros((2,2))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.2, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax, μ_sim_0=μ_sim_0, Σ_sim_0=Σ_sim_0, 
                     diff=False, name=r'$u_{{{}, 0}}$'.format(i))
    
# 这控制图的边界
ax.set_ylim(ymin=-3, ymax=3)
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

尽管公司从不同的先验均值 $\hat u_0$ 出发，但这三位劳动者都拥有相同的真实工作伦理 $u_0 = 1$，推断出的路径以各自反映 $\beta$ 的速度收敛到那条共同的虚线上。

最后，我们追踪同一类型的劳动者在两种不同真实努力水平下的表现，比较一位 $u_0=100$ 的勤奋劳动者与一位 $u_0=30$ 的普通劳动者。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 一位勤奋的劳动者和一位不太勤奋的劳动者
    name: fig-kalman2-two
---
T = 50
fig, ax = plt.subplots(figsize=(7, 7))

μ_sim_0_1 = np.array([[1],
                      [100]])
μ_sim_0_2 = np.array([[1],
                      [30]])
Σ_sim_0 = np.zeros((2, 2))

worker = create_worker(uhat_0=1, α=0.5, β=0.3)
simulate_workers(worker, T, ax, μ_sim_0=μ_sim_0_1, Σ_sim_0=Σ_sim_0, 
                 diff=False, name='勤奋的劳动者')
simulate_workers(worker, T, ax, μ_sim_0=μ_sim_0_2, Σ_sim_0=Σ_sim_0, 
                 diff=False, name='普通劳动者')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

两条推断路径都从公司共同的先验 $\hat u_0 = 1$ 出发，逐渐朝各自不同的真实值攀升，这表明随着证据的积累，滤波器会不断修正先验与真实值之间的差距。

## 未来扩展

我们可以通过创建新类型的劳动者，并让公司仅通过观察他们的产出历史来学习他们（公司未曾观察到的）隐藏状态，从而进行许多有启发性的实验。
