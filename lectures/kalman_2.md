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

(kalman)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 再看卡尔曼滤波

```{index} single: Kalman Filter 2
```

```{contents} 目录
:depth: 2
```

在这个 quantecon 讲座 {doc}`卡尔曼滤波初探 <kalman>` 中，我们使用卡尔曼滤波来估计火箭的位置。

在本讲座中，我们将使用卡尔曼滤波来推断一个工人的人力资本以及该工人投入到积累中的努力。

人力资本，这两者公司都无法直接观察到。

公司只能通过观察员工为公司创造的产出历史，以及理解这些产出如何依赖于员工的人力资本，以及人力资本如何随着员工的努力而演变来了解这些情况。

我们将提出一个规则，表达公司每期支付给员工的工资如何取决于公司每期掌握的信息。

除了Anaconda中已有的库外，本讲还需要以下库：

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

为了进行模拟，我们导入这些包，如同在{doc}`卡尔曼滤波器初探 <kalman>`中一样。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from quantecon import Kalman, LinearStateSpace
from collections import namedtuple
from scipy.stats import multivariate_normal
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
```

## 工人的产出

一名代表性工人在一家公司长期就业。

工人的产出由以下动态过程描述：

```{math}
:label: worker_model

\begin{aligned}
h_{t+1} &= \alpha h_t + \beta u_t + c w_{t+1}, \quad c_{t+1} \sim {\mathcal N}(0,1) \\
u_{t+1} & = u_t \\
y_t & = g h_t + v_t , \quad v_t \sim {\mathcal N} (0, R)
\end{aligned}
```

其中

* $h_t$ 是t时刻人力资本的对数
* $u_t$ 是t时刻工人积累人力资本所付出努力的对数
* $y_t$ 是t时刻工人产出的对数
* $h_0 \sim {\mathcal N}(\hat h_0, \sigma_{h,0})$
* $u_0 \sim {\mathcal N}(\hat u_0, \sigma_{u,0})$

模型的参数包括 $\alpha, \beta, c, R, g, \hat h_0, \hat u_0, \sigma_h, \sigma_u$。

在0时刻，公司雇佣了这名工人。

该工人与公司建立永久性雇佣关系，因此在所有时间点 $t =0, 1, 2, \ldots$ 都在同一家公司工作。

在时间 $0$ 开始时，公司既不知道工人的先天初始人力资本 $h_0$，也不知道其固有的永久性努力水平 $u_0$。

公司认为特定工人的 $u_0$ 是从高斯概率分布中抽取的，因此可以表示为 $u_0 \sim {\mathcal N}(\hat u_0, \sigma_{u,0})$。

工人"类型"中的 $h_t$ 部分会随时间变化，但工人类型中的努力成分为 $u_t = u_0$。

这意味着从公司的角度来看，工人的努力实际上是一个未知的固定"参数"。

在时间 $t\geq 1$ 时，对于特定工人，公司观察到 $y^{t-1} = [y_{t-1}, y_{t-2}, \ldots, y_0]$。

公司无法观察到工人的"类型" $(h_0, u_0)$。

但公司可以观察到工人在时间 $t$ 的产出 $y_t$，并记住工人过去的产出 $y^{t-1}$。

## 公司的工资制定政策

基于公司在时间 $t \geq 1$ 时掌握的关于工人的信息，公司支付给工人的对数工资为

$$
w_t = g  E [ h_t | y^{t-1} ], \quad t \geq 1
$$

在时间 $0$ 时，公司支付给工人的对数工资等于 $y_0$ 的无条件均值：

$$
w_0 = g \hat h_0
$$

在使用这个支付规则时，公司考虑到工人今天的对数产出部分来自于完全由运气决定的随机成分 $v_t$，且假设该成分与 $h_t$ 和 $u_t$ 相互独立。

## 状态空间表示

将系统 [](worker_model) 写成状态空间形式

```{math}
\begin{aligned}
\begin{bmatrix} h_{t+1} \cr u_{t+1} \end{bmatrix} &= \begin{bmatrix} \alpha & \beta \cr 0 & 1 \end{bmatrix}\begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + \begin{bmatrix} c \cr 0 \end{bmatrix} w_{t+1} \cr
y_t & = \begin{bmatrix} g & 0 \end{bmatrix} \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} + v_t
\end{aligned}
```

这等价于

```{math}
:label: ssrepresent

\begin{aligned} 
x_{t+1} & = A x_t + C w_{t+1} \cr
y_t & = G x_t + v_t \cr
x_0 & \sim {\mathcal N}(\hat x_0, \Sigma_0) 
\end{aligned}
```

其中

```{math}
x_t  = \begin{bmatrix} h_{t} \cr u_{t} \end{bmatrix} , \quad
\hat x_0  = \begin{bmatrix} \hat h_0 \cr \hat u_0 \end{bmatrix} , \quad
\Sigma_0  = \begin{bmatrix} \sigma_{h,0} & 0 \cr
                     0 & \sigma_{u,0} \end{bmatrix}
```

为了计算公司的工资设定政策，我们首先创建一个`namedtuple`来存储模型的参数

```{code-cell} ipython3
WorkerModel = namedtuple("WorkerModel", 
                ('A', 'C', 'G', 'R', 'xhat_0', 'Σ_0'))

def create_worker(α=.8, β=.2, c=.2,
                  R=.5, g=1.0, hhat_0=4, uhat_0=4, 
                  σ_h=4, σ_u=4):
    
    A = np.array([[α, β], 
                  [0, 1]])
    C = np.array([[c], 
                  [0]])
    G = np.array([g, 1])

    # 定义初始状态和协方差矩阵
    xhat_0 = np.array([[hhat_0], 
                       [uhat_0]])
    
    Σ_0 = np.array([[σ_h, 0],
                    [0, σ_u]])
    
    return WorkerModel(A=A, C=C, G=G, R=R, xhat_0=xhat_0, Σ_0=Σ_0)
```

请注意`WorkerModel`命名元组如何创建计算相关状态空间表示{eq}`ssrepresent`所需的所有对象。

这很方便，因为为了模拟工人的历史序列$\{y_t, h_t\}$，我们需要使用[`LinearStateSpace`](https://quanteconpy.readthedocs.io/en/latest/tools/lss.html)类为他/她构建状态空间系统。

```{code-cell} ipython3
# 定义 A, C, G, R, xhat_0, Σ_0
worker = create_worker()
A, C, G, R = worker.A, worker.C, worker.G, worker.R
xhat_0, Σ_0 = worker.xhat_0, worker.Σ_0

# 创建一个LinearStateSpace对象
ss = LinearStateSpace(A, C, G, np.sqrt(R), 
        mu_0=xhat_0, Sigma_0=np.zeros((2,2)))

T = 100
x, y = ss.simulate(T)
y = y.flatten()

h_0, u_0 = x[0, 0], x[1, 0]
```

接下来，为了根据公司掌握的工人信息来计算其设定对数工资的政策，我们使用在这个 quantecon 讲座 {doc}`A First Look at the Kalman filter <kalman>` 中描述的卡尔曼滤波。

特别地，我们要计算"创新表示"中的所有对象。

## 创新表示

我们已经掌握了所有必要的对象，可以为工人的输出过程 $\{y_t\}_{t=0}^T$ 形成创新表示。

让我们现在把它编码出来。

```{math}
\begin{aligned}
\hat x_{t+1} & = A \hat x_t + K_t a_t \cr
y_{t} & = G \hat x_t + a_t
\end{aligned}
```
其中 $K_t$ 是时间 t 的卡尔曼增益矩阵。

我们使用 [`Kalman`](https://quanteconpy.readthedocs.io/en/latest/tools/kalman.html) 类在以下代码中实现这一点。

```{code-cell} ipython3
kalman = Kalman(ss, xhat_0, Σ_0)
Σ_t = np.zeros((*Σ_0.shape, T-1))
y_hat_t = np.zeros(T-1)
x_hat_t = np.zeros((2, T-1))

for t in range(1, T):
    kalman.update(y[t])
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t[:, :, t-1] = Σ
    x_hat_t[:, t-1] = x_hat.reshape(-1)
    y_hat_t[t-1] = worker.G @ x_hat

x_hat_t = np.concatenate((x[:, 1][:, np.newaxis], 
                    x_hat_t), axis=1)
Σ_t = np.concatenate((worker.Σ_0[:, :, np.newaxis], 
                    Σ_t), axis=2)
u_hat_t = x_hat_t[1, :]
```

对于给定的$h_0, u_0$，我们绘制$E y_t = G \hat x_t$，其中$\hat x_t = E [x_t | y^{t-1}]$。

我们还绘制$E [u_0 | y^{t-1}]$，这是公司对员工固有"工作道德"$u_0$的推断，基于进入第$t$期时掌握的关于该员工的信息$y^{t-1}$。

我们可以观察到公司对员工工作道德的推断$E [u_0 | y^{t-1}]$如何逐渐收敛到隐藏的$u_0$值，而$u_0$是公司无法直接观察到的。

```{code-cell} ipython3

fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$E[y_t| y^{t-1}]$')
ax[0].set_xlabel('时间')
ax[0].set_ylabel(r'$E[y_t]$')
ax[0].set_title(r'随时间变化的$E[y_t]$')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', 
            linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('时间')
ax[1].set_ylabel(r'$E[u_t|y^{t-1}]$')
ax[1].set_title('随时间变化的推断工作道德')
ax[1].legend()

fig.tight_layout()
plt.show()
```

## 一些计算实验

让我们来看看 $\Sigma_0$ 和 $\Sigma_T$，以了解在我们设定的时间范围内，公司对隐藏状态的了解程度。

```{code-cell} ipython3
print(Σ_t[:, :, 0])
```

```{code-cell} ipython3
print(Σ_t[:, :, -1])
```

显然，条件协方差矩阵的元素随时间变小。

通过在不同时间点 $t$ 绘制围绕 $E [x_t |y^{t-1}] $ 的置信椭圆，可以直观地展示条件协方差矩阵 $\Sigma_t$ 是如何演变的。

```{code-cell} ipython3

# 创建用于等值线绘图的网格点
h_range = np.linspace(x_hat_t[0, :].min()-0.5*Σ_t[0, 0, 1], 
                      x_hat_t[0, :].max()+0.5*Σ_t[0, 0, 1], 100)
u_range = np.linspace(x_hat_t[1, :].min()-0.5*Σ_t[1, 1, 1], 
                      x_hat_t[1, :].max()+0.5*Σ_t[1, 1, 1], 100)
h, u = np.meshgrid(h_range, u_range)

# 创建包含多个子图的图形
fig, axs = plt.subplots(1, 3, figsize=(12, 7))

# 遍历每个时间步
for i, t in enumerate(np.linspace(0, T-1, 3, dtype=int)):
    # 用时间步t的x_hat和Σ创建多元正态分布
    mu = x_hat_t[:, t]
    cov = Σ_t[:, :, t]
    mvn = multivariate_normal(mean=mu, cov=cov)
    
    # 在网格上计算多元正态PDF的值
    pdf_values = mvn.pdf(np.dstack((h, u)))
    
    # 为PDF创建等值线图
    con = axs[i].contour(h, u, pdf_values, cmap='viridis')
    axs[i].clabel(con, inline=1, fontsize=10)
    axs[i].set_title(f'时间步 {t+1}')
    axs[i].set_xlabel(r'$h_{{{}}}$'.format(str(t+1)))
    axs[i].set_ylabel(r'$u_{{{}}}$'.format(str(t+1)))
    
    cov_latex = r'$\Sigma_{{{}}}= \begin{{bmatrix}} {:.2f} & {:.2f} \\ {:.2f} & {:.2f} \end{{bmatrix}}$'.format(
        t+1, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
    )
    axs[i].text(0.33, -0.15, cov_latex, transform=axs[i].transAxes)

    
plt.tight_layout()
plt.show()
```

注意观察随着样本量 $t$ 的增长，证据 $y^t$ 的累积如何影响置信椭圆的形状。

现在让我们使用代码将隐藏状态 $x_0$ 设置为特定向量，以观察公司如何从我们感兴趣的某个 $x_0$ 开始学习。

例如，假设 $h_0 = 0$ 且 $u_0 = 4$。

这里是实现这一点的一种方法。

```{code-cell} ipython3

# 例如，我们可能想要 h_0 = 0 且 u_0 = 4
mu_0 = np.array([0.0, 4.0])

# 创建一个LinearStateSpace对象，将Sigma_0设为零矩阵
ss_example = LinearStateSpace(A, C, G, np.sqrt(R), mu_0=mu_0, 
                              # 这行代码强制设定 h_0=0 且 u_0=4
                              Sigma_0=np.zeros((2, 2))
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# 现在 h_0=0 且 u_0=4
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)
```

实现相同目标的另一种方法是使用以下代码。

```{code-cell} ipython3

# 如果我们想设置初始值
# h_0 = hhat_0 = 0 和 u_0 = uhhat_0 = 4.0:
worker = create_worker(hhat_0=0.0, uhat_0=4.0)

ss_example = LinearStateSpace(A, C, G, np.sqrt(R), 
                              # 这行代码设置 h_0=hhat_0 和 u_0=uhhat_0
                              mu_0=worker.xhat_0,
                              # 这行代码强制精确设置 h_0=hhat_0 和 u_0=uhhat_0
                              Sigma_0=np.zeros((2, 2))
                             )

T = 100
x, y = ss_example.simulate(T)
y = y.flatten()

# 现在 h_0 和 u_0 将精确等于 hhat_0
h_0, u_0 = x[0, 0], x[1, 0]
print('h_0 =', h_0)
print('u_0 =', u_0)
```

让我们为这个工人生成一个类似上面的图。

```{code-cell} ipython3
# 首先我们用初始值 xhat_0 和 Σ_0 计算卡尔曼滤波
kalman = Kalman(ss, xhat_0, Σ_0)
Σ_t = []
y_hat_t = np.zeros(T-1)
u_hat_t = np.zeros(T-1)

# 然后我们基于上面的线性状态模型，
# 使用观测值 y 迭代更新卡尔曼滤波类：
for t in range(1, T):
    kalman.update(y[t])
    x_hat, Σ = kalman.x_hat, kalman.Sigma
    Σ_t.append(Σ)
    y_hat_t[t-1] = worker.G @ x_hat
    u_hat_t[t-1] = x_hat[1]


# 为 y_hat_t 和 u_hat_t 生成图表
fig, ax = plt.subplots(1, 2)

ax[0].plot(y_hat_t, label=r'$E[y_t| y^{t-1}]$')
ax[0].set_xlabel('时间')
ax[0].set_ylabel(r'$E[y_t]$')
ax[0].set_title(r'随时间变化的 $E[y_t]$')
ax[0].legend()

ax[1].plot(u_hat_t, label=r'$E[u_t|y^{t-1}]$')
ax[1].axhline(y=u_0, color='grey', 
            linestyle='dashed', label=fr'$u_0={u_0:.2f}$')
ax[1].set_xlabel('时间')
ax[1].set_ylabel(r'$E[u_t|y^{t-1}]$')
ax[1].set_title('推断的工作态度随时间变化')
ax[1].legend()

fig.tight_layout()
plt.show()
```

更广泛地说，我们可以在`create_worker`命名元组中更改部分或全部定义工人的参数。

这里是一个例子。

```{code-cell} ipython3

# 在创建工人时我们可以设置这些参数 -- 就像类一样！
hard_working_worker =  create_worker(α=.4, β=.8, 
                        hhat_0=7.0, uhat_0=100, σ_h=2.5, σ_u=3.2)

print(hard_working_worker)
```

我们也可以对不同的工人模拟 $T = 50$ 个时期的系统。

推断的工作道德和真实工作道德之间的差异随时间趋近于 $0$。

这表明过滤器正在逐步帮助工人和公司了解工人的努力程度。

```{code-cell} ipython3
:tags: [hide-input]

def simulate_workers(worker, T, ax, mu_0=None, Sigma_0=None, 
                    diff=True, name=None, title=None):
    A, C, G, R = worker.A, worker.C, worker.G, worker.R
    xhat_0, Σ_0 = worker.xhat_0, worker.Σ_0
    
    if isinstance(mu_0, type(None)):
        mu_0 = xhat_0
    if isinstance(Sigma_0, type(None)):
        Sigma_0 = worker.Σ_0
        
    ss = LinearStateSpace(A, C, G, np.sqrt(R), 
                        mu_0=mu_0, Sigma_0=Sigma_0)

    x, y = ss.simulate(T)
    y = y.flatten()

    u_0 = x[1, 0]
    
    # Compute Kalman filter
    kalman = Kalman(ss, xhat_0, Σ_0)
    Σ_t = []
    
    y_hat_t = np.zeros(T)
    u_hat_t = np.zeros(T)

    for i in range(T):
        kalman.update(y[i])
        x_hat, Σ = kalman.x_hat, kalman.Sigma
        Σ_t.append(Σ)
        y_hat_t[i] = worker.G @ x_hat
        u_hat_t[i] = x_hat[1]

    if diff == True:
        title = ('Difference between inferred and true work ethic over time' 
                 if title == None else title)
        
        ax.plot(u_hat_t - u_0, alpha=.5)
        ax.axhline(y=0, color='grey', linestyle='dashed')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$E[u_t|y^{t-1}] - u_0$')
        ax.set_title(title)
        
    else:
        label_line = (r'$E[u_t|y^{t-1}]$' if name == None 
                      else name)
        title = ('Inferred work ethic over time' 
                if title == None else title)
        
        u_hat_plot = ax.plot(u_hat_t, label=label_line)
        ax.axhline(y=u_0, color=u_hat_plot[0].get_color(), 
                    linestyle='dashed', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$E[u_t|y^{t-1}]$')
        ax.set_title(title)
```

```{code-cell} ipython3

num_workers = 3
T = 50
fig, ax = plt.subplots(figsize=(7, 7))

for i in range(num_workers):
    worker = create_worker(uhat_0=4+2*i)
    simulate_workers(worker, T, ax)
ax.set_ylim(ymin=-2, ymax=2)
plt.show()
```

```{code-cell} ipython3

# 我们也可以生成 u_t 的图：

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.1, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax,
                    # 通过设置 diff=False，将显示 u_t
                    diff=False, name=r'$u_{{{}, t}}$'.format(i))
    
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

```{code-cell} ipython3

# 我们也可以为所有工人使用精确的 u_0=1 和 h_0=2

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

# 这两行为所有工人设置 u_0=1 和 h_0=2
mu_0 = np.array([[1],
                 [2]])
Sigma_0 = np.zeros((2,2))

uhat_0s = [2, -2, 1]
αs = [0.2, 0.3, 0.5]
βs = [0.1, 0.9, 0.3]

for i, (uhat_0, α, β) in enumerate(zip(uhat_0s, αs, βs)):
    worker = create_worker(uhat_0=uhat_0, α=α, β=β)
    simulate_workers(worker, T, ax, mu_0=mu_0, Sigma_0=Sigma_0, 
                     diff=False, name=r'$u_{{{}, t}}$'.format(i))
    
# 这控制着图的边界
ax.set_ylim(ymin=-3, ymax=3)
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```

```{code-cell} ipython3

# 我们可以为其中一个工人生成图表：

T = 50
fig, ax = plt.subplots(figsize=(7, 7))

mu_0_1 = np.array([[1],
                 [100]])
mu_0_2 = np.array([[1],
                 [30]])
Sigma_0 = np.zeros((2,2))

uhat_0s = 100
αs = 0.5
βs = 0.3

worker = create_worker(uhat_0=uhat_0, α=α, β=β)
simulate_workers(worker, T, ax, mu_0=mu_0_1, Sigma_0=Sigma_0, 
                 diff=False, name=r'勤奋的工人')
simulate_workers(worker, T, ax, mu_0=mu_0_2, Sigma_0=Sigma_0, 
                 diff=False, 
                 title='一个勤奋的工人和一个较不勤奋的工人',
                 name=r'普通工人')
ax.axhline(y=u_0, xmin=0, xmax=0, color='grey', 
           linestyle='dashed', label=r'$u_{i, 0}$')
ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()
```


## 未来扩展

通过创建新类型的工人，让企业仅通过观察他们的产出历史来了解其（对企业）隐藏的状态，我们可以进行许多富有启发性的实验。

