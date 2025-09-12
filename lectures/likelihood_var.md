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
---

(var_likelihood)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# VAR模型的似然过程

```{contents} 目录
:depth: 2
```

## 概述

本讲座将我们对似然比过程的分析扩展到向量自回归(VAR)模型。

我们将：

* 构建VAR模型的似然函数
* 形成用于比较两个VAR模型的似然比过程
* 可视化似然比随时间的演变
* 将VAR似然比与萨缪尔森乘数-加速器模型联系起来

我们的分析建立在以下概念之上：
- {doc}`likelihood_ratio_process`
- {doc}`linear_models`
- {doc}`samuelson`

让我们首先导入有用的库：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from scipy import linalg
from scipy.stats import multivariate_normal as mvn
from quantecon import LinearStateSpace
import quantecon as qe
from numba import jit
from typing import NamedTuple, Optional, Tuple
from collections import namedtuple
```

## VAR模型设置

考虑以下形式的VAR模型：

$$
\begin{aligned} 
x_{t+1} & = A x_t + C w_{t+1} \\
x_0 & \sim \mathcal{N}(\mu_0, \Sigma_0) 
\end{aligned}
$$

其中：
- $x_t$ 是一个 $n \times 1$ 状态向量
- $w_{t+1} \sim \mathcal{N}(0, I)$ 是一个 $m \times 1$ 的冲击向量
- $A$ 是一个 $n \times n$ 转移矩阵
- $C$ 是一个 $n \times m$ 波动率矩阵

让我们为VAR模型定义必要的数据结构

```{code-cell} ipython3
VARModel = namedtuple('VARModel', ['A', 'C', 'μ_0', 'Σ_0',        
                                    'CC', 'CC_inv', 'log_det_CC', 
                                    'Σ_0_inv', 'log_det_Σ_0'])
def compute_stationary_var(A, C):
    """
    计算VAR模型的平稳均值和协方差
    """
    n = A.shape[0]
    
    # 检查稳定性
    eigenvalues = np.linalg.eigvals(A)
    if np.max(np.abs(eigenvalues)) >= 1:
        raise ValueError("VAR不是平稳的")
    
    μ_0 = np.zeros(n)
    
    # 平稳协方差：求解离散Lyapunov方程
    # Σ_0 = A @ Σ_0 @ A.T + C @ C.T
    CC = C @ C.T
    Σ_0 = linalg.solve_discrete_lyapunov(A, CC)
    
    return μ_0, Σ_0

def create_var_model(A, C, μ_0=None, Σ_0=None, stationary=True):
    """
    创建带有参数和预计算矩阵的VAR模型
    """
    A = np.asarray(A)
    C = np.asarray(C)
    n = A.shape[0]
    CC = C @ C.T
    
    if stationary:
        μ_0_comp, Σ_0_comp = compute_stationary_var(A, C)
    else:
        μ_0_comp = μ_0 if μ_0 is not None else np.zeros(n)
        Σ_0_comp = Σ_0 if Σ_0 is not None else np.eye(n)
    
    # 检查CC是否奇异
    det_CC = np.linalg.det(CC)
    if np.abs(det_CC) < 1e-10:
        # 对奇异情况使用伪逆
        CC_inv = np.linalg.pinv(CC)
        CC_reg = CC + 1e-10 * np.eye(CC.shape[0])
        log_det_CC = np.log(np.linalg.det(CC_reg))
    else:
        CC_inv = np.linalg.inv(CC)
        log_det_CC = np.log(det_CC)
    
    # 对Σ_0进行相同的检查
    det_Σ_0 = np.linalg.det(Σ_0_comp)
    if np.abs(det_Σ_0) < 1e-10:
        Σ_0_inv = np.linalg.pinv(Σ_0_comp)
        Σ_0_reg = Σ_0_comp + 1e-10 * np.eye(Σ_0_comp.shape[0])
        log_det_Σ_0 = np.log(np.linalg.det(Σ_0_reg))
    else:
        Σ_0_inv = np.linalg.inv(Σ_0_comp)
        log_det_Σ_0 = np.log(det_Σ_0)
    
    return VARModel(A=A, C=C, μ_0=μ_0_comp, Σ_0=Σ_0_comp,
                    CC=CC, CC_inv=CC_inv, log_det_CC=log_det_CC,
                    Σ_0_inv=Σ_0_inv, log_det_Σ_0=log_det_Σ_0)
```

### 联合分布

联合概率分布 $f(x_T, x_{T-1}, \ldots, x_0)$ 可以分解为：

$$
f(x_T, \ldots, x_0) = f(x_T | x_{T-1}) f(x_{T-1} | x_{T-2}) \cdots f(x_1 | x_0) f(x_0)
$$

由于VAR是马尔可夫的，$f(x_{t+1} | x_t, \ldots, x_0) = f(x_{t+1} | x_t)$。

### 条件密度

基于高斯结构，条件分布 $f(x_{t+1} | x_t)$ 是高斯分布，其：
- 均值：$A x_t$
- 协方差：$CC'$

对数条件密度为

$$
\log f(x_{t+1} | x_t) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log \det(CC') - \frac{1}{2} (x_{t+1} - A x_t)' (CC')^{-1} (x_{t+1} - A x_t)
$$ (eq:cond_den)

```{code-cell} ipython3
def log_likelihood_transition(x_next, x_curr, model):
    """
    计算从x_curr到x_next的转移对数似然
    """
    x_next = np.atleast_1d(x_next)
    x_curr = np.atleast_1d(x_curr)
    n = len(x_next)
    diff = x_next - model.A @ x_curr
    return -0.5 * (n * np.log(2 * np.pi) + model.log_det_CC + 
                  diff @ model.CC_inv @ diff)
```

初始状态的对数密度为：

$$
\log f(x_0) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log \det(\Sigma_0) - \frac{1}{2} (x_0 - \mu_0)' \Sigma_0^{-1} (x_0 - \mu_0)
$$

```{code-cell} ipython3
def log_likelihood_initial(x_0, model):
    """
    计算初始状态的对数似然
    """
    x_0 = np.atleast_1d(x_0)
    n = len(x_0)
    diff = x_0 - model.μ_0
    return -0.5 * (n * np.log(2 * np.pi) + model.log_det_Σ_0 + 
                  diff @ model.Σ_0_inv @ diff)
```

现在让我们把似然计算组合成一个函数，用于计算整个路径的对数似然

```{code-cell} ipython3
def log_likelihood_path(X, model):
    """
    计算整个路径的对数似然
    """

    T = X.shape[0] - 1
    log_L = log_likelihood_initial(X[0], model)
    
    for t in range(T):
        log_L += log_likelihood_transition(X[t+1], X[t], model)
        
    return log_L

def simulate_var(model, T, N_paths=1):
    """
    从VAR模型中模拟路径
    """
    n = model.A.shape[0]
    m = model.C.shape[1]
    paths = np.zeros((N_paths, T+1, n))
    
    for i in range(N_paths):
        # 生成初始状态
        x = mvn.rvs(mean=model.μ_0, cov=model.Σ_0)
        x = np.atleast_1d(x)
        paths[i, 0] = x
        
        # 向前模拟
        for t in range(T):
            w = np.random.randn(m)
            x = model.A @ x + model.C @ w
            paths[i, t+1] = x
            
    return paths if N_paths > 1 else paths[0]
```

## 似然比过程

现在让我们计算两个VAR模型的似然比过程。

对于具有状态向量$x_t$的VAR模型，在时间$t$的对数似然比为

$$
\ell_t = \log \frac{p_f(x_t | x_{t-1})}{p_g(x_t | x_{t-1})}
$$

其中$p_f$和$p_g$分别是模型$f$和$g$下的条件密度。

累积对数似然比过程为

$$
L_t = \sum_{s=1}^{t} \ell_s = \sum_{s=1}^{t} \log \frac{p_f(x_s | x_{s-1})}{p_g(x_s | x_{s-1})}
$$

其中$p_f(x_t | x_{t-1})$和$p_g(x_t | x_{t-1})$由{eq}`eq:cond_den`中定义的各自条件密度给出。

让我们用Python编写这些方程

```{code-cell} ipython3
def compute_likelihood_ratio_var(paths, model_f, model_g):
    """
    计算VAR模型的似然比过程
    """
    if paths.ndim == 2:
        paths = paths[np.newaxis, :]
    
    N_paths, T_plus_1, n = paths.shape
    T = T_plus_1 - 1
    log_L_ratios = np.zeros((N_paths, T+1))
    
    for i in range(N_paths):
        X = paths[i]
        
        # 初始对数似然比
        log_L_f_0 = log_likelihood_initial(X[0], model_f)
        log_L_g_0 = log_likelihood_initial(X[0], model_g)
        log_L_ratios[i, 0] = log_L_f_0 - log_L_g_0
        
        # 递归计算
        for t in range(1, T+1):
            log_L_f_t = log_likelihood_transition(X[t], X[t-1], model_f)
            log_L_g_t = log_likelihood_transition(X[t], X[t-1], model_g)
            
            # 更新对数似然比
            log_diff = log_L_f_t - log_L_g_t
            
            log_L_prev = log_L_ratios[i, t-1]
            log_L_new = log_L_prev + log_diff
            log_L_ratios[i, t] = log_L_new

    return log_L_ratios if N_paths > 1 else log_L_ratios[0]
```

## 示例1：两个AR(1)过程

让我们从一个简单的例子开始，比较两个单变量AR(1)过程，其中$A_f = 0.8$，$A_g = 0.5$，以及$C_f = 0.3$，$C_g = 0.4$

```{code-cell} ipython3
# 模型f：AR(1)，持续性系数 ρ = 0.8
A_f = np.array([[0.8]])
C_f = np.array([[0.3]])

# 模型g：AR(1)，持续性系数 ρ = 0.5
A_g = np.array([[0.5]])
C_g = np.array([[0.4]])

# 创建VAR模型
model_f = create_var_model(A_f, C_f)
model_g = create_var_model(A_g, C_g)
```

让我们从模型 $f$ 生成100条长度为200的路径，并计算似然比过程

```{code-cell} ipython3
# 从模型f进行模拟
T = 200
N_paths = 100
paths_from_f = simulate_var(model_f, T, N_paths)

L_ratios_f = compute_likelihood_ratio_var(paths_from_f, model_f, model_g)

fig, ax = plt.subplots()

for i in range(min(20, N_paths)):
    ax.plot(L_ratios_f[i], alpha=0.3, color='C0', lw=2)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel(r'$\log L_t$')
ax.set_title('对数似然比过程(本质 = f)')

plt.tight_layout()
plt.show()
```

正如我们预期的那样，似然比过程随着T的增加趋向于$+\infty$，表明我们的算法正确选择了模型$f$。

## 示例2：二元VAR模型

现在让我们考虑一个二元VAR模型的例子，其中

$$
A_f & = \begin{bmatrix} 0.7 & 0.2 \\ 0.1 & 0.6 \end{bmatrix}, \quad C_f = \begin{bmatrix} 0.3 & 0.1 \\ 0.1 & 0.3 \end{bmatrix} 
$$

和

$$
A_g & = \begin{bmatrix} 0.5 & 0.3 \\ 0.2 & 0.5 \end{bmatrix}, \quad C_g = \begin{bmatrix} 0.4 & 0.0 \\ 0.0 & 0.4 \end{bmatrix}
$$

```{code-cell} ipython3
A_f = np.array([[0.7, 0.2],
                 [0.1, 0.6]])

C_f = np.array([[0.3, 0.1],
                 [0.1, 0.3]])

A_g = np.array([[0.5, 0.3],
                 [0.2, 0.5]])

C_g = np.array([[0.4, 0.0],
                 [0.0, 0.4]])

# 创建VAR模型
model2_f = create_var_model(A_f, C_f)
model2_g = create_var_model(A_g, C_g)

# 检查平稳性
print("模型f的特征值:", np.linalg.eigvals(A_f))
print("模型g的特征值:", np.linalg.eigvals(A_g))
```

让我们从两个模型中各生成50条长度为50的路径，并计算似然比过程

```{code-cell} ipython3
# 从两个模型中模拟
T = 50
N_paths = 50

paths_from_f = simulate_var(model2_f, T, N_paths)
paths_from_g = simulate_var(model2_g, T, N_paths)

# 计算似然比
L_ratios_ff = compute_likelihood_ratio_var(paths_from_f, model2_f, model2_g)
L_ratios_gf = compute_likelihood_ratio_var(paths_from_g, model2_f, model2_g)
```

我们可以看到，对于从模型 $f$ 生成的路径，似然比过程趋向于 $+\infty$，而对于从模型 $g$ 生成的路径，则趋向于 $-\infty$。

```{code-cell} ipython3
# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i in range(min(20, N_paths)):
    ax.plot(L_ratios_ff[i], alpha=0.5, color='C0', lw=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=2)
ax.set_title(r'$\log L_t$ (nature = f)')
ax.set_ylabel(r'$\log L_t$')

ax = axes[1]
for i in range(min(20, N_paths)):
    ax.plot(L_ratios_gf[i], alpha=0.5, color='C1', lw=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=2)
ax.set_title(r'$\log L_t$ (nature = g)')
plt.tight_layout()
plt.show()
```

让我们应用{doc}`likelihood_ratio_process`中描述的Neyman-Pearson频率主义决策规则，当$\log L_T \geq 0$时选择模型$f$，当$\log L_T < 0$时选择模型$g$

```{code-cell} ipython3
fig, ax = plt.subplots()
T_values = np.arange(0, T+1)
accuracy_f = np.zeros(len(T_values))
accuracy_g = np.zeros(len(T_values))

for i, t in enumerate(T_values):
    # 当数据来自f时的正确选择
    accuracy_f[i] = np.mean(L_ratios_ff[:, t] > 0)
    # 当数据来自g时的正确选择
    accuracy_g[i] = np.mean(L_ratios_gf[:, t] < 0)

ax.plot(T_values, accuracy_f, 'C0', linewidth=2, label='accuracy (nature = f)')
ax.plot(T_values, accuracy_g, 'C1', linewidth=2, label='accuracy (nature = g)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('T')
ax.set_ylabel('accuracy')
ax.legend()

plt.tight_layout()
plt.show()
```

显然，随着 $T$ 的增加，准确率趋近于 $1$，而且这个过程非常快。

让我们也检查一下类型 I 和类型 II 错误作为 $T$ 的函数的变化

```{code-cell} ipython3
def model_selection_analysis(T_values, model_f, model_g, N_sim=500):
    """
    分析不同样本大小的模型选择性能
    """
    errors_f = []  # 类型 I 错误
    errors_g = []  # 类型 II 错误
    
    for T in T_values:
        # 从模型 f 模拟
        paths_f = simulate_var(model_f, T, N_sim//2)
        L_ratios_f = compute_likelihood_ratio_var(paths_f, model_f, model_g)
        
        # 从模型 g 模拟
        paths_g = simulate_var(model_g, T, N_sim//2)
        L_ratios_g = compute_likelihood_ratio_var(paths_g, model_f, model_g)
        
        # 决策规则：如果 log L_T >= 0 则选择 f
        errors_f.append(np.mean(L_ratios_f[:, -1] < 0))
        errors_g.append(np.mean(L_ratios_g[:, -1] >= 0))
    
    return np.array(errors_f), np.array(errors_g)

T_values = np.arange(1, 50, 1)
errors_f, errors_g = model_selection_analysis(T_values, model2_f, model2_g, N_sim=400)

fig, ax = plt.subplots()

ax.plot(T_values, errors_f, 'C0', linewidth=2, label='类型 I 错误')
ax.plot(T_values, errors_g, 'C1', linewidth=2, label='类型 II 错误')
ax.plot(T_values, 0.5 * (errors_f + errors_g), 'g--', 
linewidth=2, label='平均错误')
ax.set_xlabel('$T$')
ax.set_ylabel('错误概率')
ax.set_title('模型选择错误')
plt.tight_layout()
plt.show()
```

## 应用：萨缪尔森乘数-加速器模型

现在让我们来看萨缪尔森乘数-加速器模型。

该模型包括：

- 消费：$C_t = \gamma + a Y_{t-1}$ 其中 $a \in (0,1)$ 是边际消费倾向
- 投资：$I_t = b(Y_{t-1} - Y_{t-2})$ 其中 $b > 0$ 是加速系数
- 政府支出：$G_t = G$ (常数)

我们有国民收入恒等式

$$
Y_t = C_t + I_t + G_t
$$

这些方程得出二阶差分方程：

$$
Y_t = (\gamma + G) + (a + b)Y_{t-1} - b Y_{t-2} + \sigma \epsilon_t
$$

令 $\rho_1 = a + b$ 且 $\rho_2 = -b$，我们得到：

$$
Y_t = (\gamma + G) + \rho_1 Y_{t-1} + \rho_2 Y_{t-2} + \sigma \epsilon_t
$$

为了符合我们的讨论，我们将其写成状态空间表示。

为了正确处理常数项，我们使用增广状态向量 $\mathbf{x}_t = [1, Y_t, Y_{t-1}]'$：

$$
\mathbf{x}_{t+1} = \begin{bmatrix} 
1 \\ 
Y_{t+1} \\ 
Y_t 
\end{bmatrix} = \begin{bmatrix} 
1 & 0 & 0 \\
\gamma + G & \rho_1 & \rho_2 \\
0 & 1 & 0 
\end{bmatrix} \begin{bmatrix} 
1 \\ 
Y_t \\ 
Y_{t-1} 
\end{bmatrix} + \begin{bmatrix} 
0 \\ 
\sigma \\ 
0 
\end{bmatrix} \epsilon_{t+1}
$$

观测方程提取经济变量：

$$
\mathbf{y}_t = \begin{bmatrix} 
Y_t \\ 
C_t \\ 
I_t 
\end{bmatrix} = \begin{bmatrix} 
\gamma + G & \rho_1 & \rho_2 \\
\gamma & a & 0 \\
0 & b & -b 
\end{bmatrix} \begin{bmatrix} 
1 \\ 
Y_t \\ 
Y_{t-1} 
\end{bmatrix}
$$

这给出了：

- $Y_t = (\gamma + G) \cdot 1 + \rho_1 Y_{t-1} + \rho_2 Y_{t-2}$ (总产出)
- $C_t = \gamma \cdot 1 + a Y_{t-1}$ (消费)
- $I_t = b(Y_{t-1} - Y_{t-2})$ (投资)

```{code-cell} ipython3
def samuelson_to_var(a, b, γ, G, σ):
    """
    将萨缪尔森模型参数转换为带扩展状态的VAR形式
    
    萨缪尔森模型:
    - Y_t = C_t + I_t + G
    - C_t = γ + a*Y_{t-1}
    - I_t = b*(Y_{t-1} - Y_{t-2})
    
    简化形式: Y_t = (γ+G) + (a+b)*Y_{t-1} - b*Y_{t-2} + σ*ε_t
    
    状态向量为 [1, Y_t, Y_{t-1}]'
    """
    ρ_1 = a + b
    ρ_2 = -b
    
    # 扩展状态的状态转移矩阵
    A = np.array([[1,      0,     0],
                  [γ + G,  ρ_1,   ρ_2],
                  [0,      1,     0]])
    
    # 冲击载荷矩阵
    C = np.array([[0],
                  [σ],
                  [0]])
    
    # 观测矩阵(提取Y_t, C_t, I_t)
    G_obs = np.array([[γ + G,  ρ_1,  ρ_2],   # Y_t
                      [γ,      a,    0],     # C_t
                      [0,      b,   -b]])    # I_t
    
    return A, C, G_obs
```

我们在下面的代码单元中定义函数来获取初始条件并检查稳定性

```{code-cell} ipython3
:tags: [hide-input]

def get_samuelson_initial_conditions(a, b, γ, G, y_0=None, y_m1=None, 
                                    stationary_init=False):
    """
    获取萨缪尔森模型的初始条件
    """
    # 计算稳态
    y_ss = (γ + G) / (1 - a - b)
    
    if y_0 is None:
        y_0 = y_ss
    if y_m1 is None:
        y_m1 = y_ss if stationary_init else y_0 * 0.95
    
    # 初始均值
    μ_0 = np.array([1.0, y_0, y_m1])
    
    if stationary_init:
        Σ_0 = np.array([[0,  0,    0],
                        [0,  1,    0.5],
                        [0,  0.5,  1]])
    else:
        Σ_0 = np.array([[0,  0,    0],
                        [0,  25,   15],
                        [0,  15,   25]])
    
    return μ_0, Σ_0

def check_samuelson_stability(a, b):
    """
    检查萨缪尔森模型的稳定性并返回特征根
    """
    ρ_1 = a + b
    ρ_2 = -b

    roots = np.roots([1, -ρ_1, -ρ_2])
    max_abs_root = np.max(np.abs(roots))
    is_stable = max_abs_root < 1
    
    # 确定动态类型
    if np.iscomplex(roots[0]):
        if max_abs_root < 1:
            dynamics = "阻尼振荡"
        else:
            dynamics = "爆炸性振荡"
    else:
        if max_abs_root < 1:
            dynamics = "平滑收敛"
        else:
            if np.max(roots) > 1:
                dynamics = "爆炸性增长"
            else:
                dynamics = "爆炸性振荡(实根)"
    
    return is_stable, roots, max_abs_root, dynamics
```

让我们实现并检查由两个具有不同参数的萨缪尔森模型产生的似然比过程。

```{code-cell} ipython3
def create_samuelson_var_model(a, b, γ, G, σ, stationary_init=False,
                               y_0=None, y_m1=None):
    """
    从萨缪尔森参数创建VAR模型
    """
    A, C, G_obs = samuelson_to_var(a, b, γ, G, σ)
    
    μ_0, Σ_0 = get_samuelson_initial_conditions(
        a, b, γ, G, y_0, y_m1, stationary_init
    )
    
    # 创建VAR模型
    model = create_var_model(A, C, μ_0, Σ_0, stationary=False)
    is_stable, roots, max_root, dynamics = check_samuelson_stability(a, b)
    info = {
        'a': a, 'b': b, 'γ': γ, 'G': G, 'σ': σ,
        'ρ_1': a + b, 'ρ_2': -b,
        'steady_state': (γ + G) / (1 - a - b),
        'is_stable': is_stable,
        'roots': roots,
        'max_abs_root': max_root,
        'dynamics': dynamics
    }
    
    return model, G_obs, info

def simulate_samuelson(model, G_obs, T, N_paths=1):
    """
    模拟萨缪尔森模型
    """
    # 模拟状态路径
    states = simulate_var(model, T, N_paths)
    
    # 使用G矩阵提取可观测值
    if N_paths == 1:
        # 单一路径: states是(T+1, 3)
        observables = (G_obs @ states.T).T
    else:
        # 多个路径: states是(N_paths, T+1, 3)
        observables = np.zeros((N_paths, T+1, 3))
        for i in range(N_paths):
            observables[i] = (G_obs @ states[i].T).T
    
    return states, observables
```

现在让我们模拟两个具有不同加速系数的萨缪尔森模型并绘制它们的样本路径

```{code-cell} ipython3
# 模型 f: 较高的加速系数
a_f, b_f = 0.98, 0.9
γ_f, G_f, σ_f = 10, 10, 0.5

# 模型 g: 较低的加速系数
a_g, b_g = 0.98, 0.85
γ_g, G_g, σ_g = 10, 10, 0.5


model_sam_f, G_obs_f, info_f = create_samuelson_var_model(
    a_f, b_f, γ_f, G_f, σ_f, 
    stationary_init=False, 
    y_0=100, y_m1=95
)

model_sam_g, G_obs_g, info_g = create_samuelson_var_model(
    a_g, b_g, γ_g, G_g, σ_g,
    stationary_init=False,
    y_0=100, y_m1=95
)

T = 50
N_paths = 50

# 获取状态和观测值
states_f, obs_f = simulate_samuelson(model_sam_f, G_obs_f, T, N_paths)
states_g, obs_g = simulate_samuelson(model_sam_g, G_obs_g, T, N_paths)

output_paths_f = obs_f[:, :, 0] 
output_paths_g = obs_g[:, :, 0]
    
print("模型 f:")
print(f"  ρ_1 = a + b = {info_f['ρ_1']:.2f}")
print(f"  ρ_2 = -b = {info_f['ρ_2']:.2f}")
print(f"  特征根: {info_f['roots']}")
print(f"  动态特征: {info_f['dynamics']}")

print("\n模型 g:")
print(f"  ρ_1 = a + b = {info_g['ρ_1']:.2f}")
print(f"  ρ_2 = -b = {info_g['ρ_2']:.2f}")
print(f"  特征根: {info_g['roots']}")
print(f"  动态特征: {info_g['dynamics']}")


fig, ax = plt.subplots(1, 1)

for i in range(min(20, N_paths)):
    ax.plot(output_paths_f[i], alpha=0.6, color='C0', linewidth=0.8)
    ax.plot(output_paths_g[i], alpha=0.6, color='C1', linewidth=0.8)
ax.set_xlabel('$t$')
ax.set_ylabel('$Y_t$')
ax.legend(['模型 f', '模型 g'], loc='upper left')
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# 计算似然比
L_ratios_ff = compute_likelihood_ratio_var(states_f, model_sam_f, model_sam_g)
L_ratios_gf = compute_likelihood_ratio_var(states_g, model_sam_f, model_sam_g) 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i in range(min(20, N_paths)):
    ax.plot(L_ratios_ff[i], alpha=0.5, color='C0', lw=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title(r'$\log L_t$ (真实模型 = f)')
ax.set_ylabel(r'$\log L_t$')

ax = axes[1]
for i in range(min(20, N_paths)):
    ax.plot(L_ratios_gf[i], alpha=0.5, color='C1', lw=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title(r'$\log L_t$ (真实模型 = g)')
plt.show()
```

在左图中,数据由$f$生成,似然比趋向正无穷。

在右图中,数据由$g$生成,似然比趋向负无穷。

在这两种情况下,为了数值稳定性,我们对对数似然比过程设置了上下限阈值,因为它们会很快增长到无界。

在这两种情况下,似然比过程最终都能帮助我们选择正确的模型。

