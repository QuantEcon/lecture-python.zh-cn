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
  title: VAR模型的似然过程
  headings:
    Overview: 概述
    VAR model setup: VAR模型设置
    VAR model setup::Joint distribution: 联合分布
    VAR model setup::Conditional densities: 条件密度
    Likelihood ratio process: 似然比过程
    'Example 1: two AR(1) processes': 示例1：两个AR(1)过程
    'Example 2: bivariate VAR models': 示例2：二元VAR模型
    KL divergence rate: KL 散度率
    'Application: Samuelson multiplier-accelerator': 应用：萨缪尔森乘数-加速器模型
    Exercises: 练习
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
* 计算控制对数似然比漂移的库尔贝克-莱布勒散度率
* 将VAR似然比与萨缪尔森乘数-加速器模型联系起来

我们的分析建立在以下概念之上：
- {doc}`likelihood_ratio_process`
- {doc}`divergence_measures`
- {doc}`linear_models`
- {doc}`samuelson`

让我们首先导入有用的库：

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
import matplotlib as mpl
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

from scipy import linalg
from scipy.stats import multivariate_normal as mvn
from collections import namedtuple

rng = np.random.default_rng(1234)
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

联合概率密度 $p(x_T, x_{T-1}, \ldots, x_0)$ 可以分解为：

$$
p(x_T, \ldots, x_0) = p(x_T | x_{T-1}) p(x_{T-1} | x_{T-2}) \cdots p(x_1 | x_0) p(x_0)
$$

由于VAR是马尔可夫的，$p(x_{t+1} | x_t, \ldots, x_0) = p(x_{t+1} | x_t)$。

我们用 $p$ 表示一般的密度，并保留 $f$ 和 $g$ 作为两个模型的名称。

### 条件密度

基于高斯结构，条件分布 $p(x_{t+1} | x_t)$ 是高斯分布，其：
- 均值：$A x_t$
- 协方差：$CC^\top$

对数条件密度为

$$
\log p(x_{t+1} | x_t) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log \det(CC^\top) - \frac{1}{2} (x_{t+1} - A x_t)^\top (CC^\top)^{-1} (x_{t+1} - A x_t)
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
\log p(x_0) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log \det(\Sigma_0) - \frac{1}{2} (x_0 - \mu_0)^\top \Sigma_0^{-1} (x_0 - \mu_0)
$$ (eq:init_den)

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

下面的函数从VAR模型中模拟路径

```{code-cell} ipython3
def simulate_var(model, T, rng, N_paths=1):
    """
    从VAR模型中模拟路径
    """
    n = model.A.shape[0]
    m = model.C.shape[1]
    paths = np.zeros((N_paths, T+1, n))
    
    for i in range(N_paths):
        # 生成初始状态
        x = mvn.rvs(mean=model.μ_0, cov=model.Σ_0, random_state=rng)
        x = np.atleast_1d(x)
        paths[i, 0] = x
        
        # 向前模拟
        for t in range(T):
            w = rng.standard_normal(m)
            x = model.A @ x + model.C @ w
            paths[i, t+1] = x
            
    return paths if N_paths > 1 else paths[0]
```

## 似然比过程

现在让我们计算用于比较两个VAR模型$f$和$g$的似然比过程。

设$p_f$和$p_g$分别表示模型$f$和$g$下的密度。

正如 {doc}`likelihood_ratio_process` 中那样，$L_t$表示样本$x_0, x_1, \ldots, x_t$的似然比：

$$
L_t = \frac{p_f(x_t, \ldots, x_0)}{p_g(x_t, \ldots, x_0)}
    = \frac{p_f(x_0)}{p_g(x_0)} \prod_{s=1}^{t} \frac{p_f(x_s | x_{s-1})}{p_g(x_s | x_{s-1})}
$$

其中条件密度由 {eq}`eq:cond_den` 给出，$x_0$的密度由 {eq}`eq:init_den` 给出。

对数似然比过程为

$$
\log L_t = \log \frac{p_f(x_0)}{p_g(x_0)} + \sum_{s=1}^{t} \ell_s, 
\qquad 
\ell_s = \log \frac{p_f(x_s | x_{s-1})}{p_g(x_s | x_{s-1})}
$$ (eq:var_llr)

它满足递归式$\log L_t = \log L_{t-1} + \ell_t$，初始条件为$\log L_0 = \log p_f(x_0) - \log p_g(x_0)$。

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
paths_from_f = simulate_var(model_f, T, rng, N_paths)

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

正如我们预期的那样，对数似然比过程随着$t$的增加趋向于$+\infty$，表明我们的算法正确选择了模型$f$。

## 示例2：二元VAR模型

现在让我们考虑一个二元VAR模型的例子，其中

$$
A_f = \begin{bmatrix} 0.7 & 0.2 \\ 0.1 & 0.6 \end{bmatrix}, \quad C_f = \begin{bmatrix} 0.3 & 0.1 \\ 0.1 & 0.3 \end{bmatrix} 
$$

和

$$
A_g = \begin{bmatrix} 0.5 & 0.3 \\ 0.2 & 0.5 \end{bmatrix}, \quad C_g = \begin{bmatrix} 0.4 & 0.0 \\ 0.0 & 0.4 \end{bmatrix}
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

paths_from_f = simulate_var(model2_f, T, rng, N_paths)
paths_from_g = simulate_var(model2_g, T, rng, N_paths)

# 计算似然比
L_ratios_ff = compute_likelihood_ratio_var(paths_from_f, model2_f, model2_g)
L_ratios_gf = compute_likelihood_ratio_var(paths_from_g, model2_f, model2_g)
```

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

对于从模型 $f$ 生成的路径，对数似然比过程趋向于 $+\infty$，而对于从模型 $g$ 生成的路径，则趋向于 $-\infty$。

在 $g$ 下路径下降的速度比在 $f$ 下上升的速度更快；{ref}`var_kl_rate` 一节将解释其中的原因。

让我们应用 {doc}`likelihood_ratio_process` 中 {ref}`模型选择错误概率 <lrp_model_selection>` 一节所描述的决策规则：当 $\log L_T \geq 0$ 时选择模型 $f$，当 $\log L_T < 0$ 时选择模型 $g$。

该规则是一个阈值为 $L_T = 1$ 的似然比检验。

假设某位贝叶斯主义者对每个模型赋予先验概率 $1/2$。

那么模型 $f$ 的后验概率为 $L_T/(1 + L_T)$，正如 {doc}`likelihood_bayes` 中所述。

该规则选择后验概率较高的模型，因此它使选错模型的概率 $\frac{1}{2}(\alpha_T + \beta_T)$ 最小化，其中 $\alpha_T = {\rm Prob}(\log L_T < 0 | f)$，$\beta_T = {\rm Prob}(\log L_T \geq 0 | g)$。

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

让我们也检查一下类型 I 和类型 II 错误 $\alpha_T$ 和 $\beta_T$ 作为 $T$ 的函数的变化

```{code-cell} ipython3
def model_selection_analysis(T_values, model_f, model_g, N_sim=500):
    """
    分析不同样本大小的模型选择性能
    """
    errors_f = []  # 类型 I 错误
    errors_g = []  # 类型 II 错误
    
    for T in T_values:
        # 从模型 f 模拟
        paths_f = simulate_var(model_f, T, rng, N_sim//2)
        L_ratios_f = compute_likelihood_ratio_var(paths_f, model_f, model_g)
        
        # 从模型 g 模拟
        paths_g = simulate_var(model_g, T, rng, N_sim//2)
        L_ratios_g = compute_likelihood_ratio_var(paths_g, model_f, model_g)
        
        # 决策规则：如果 log L_T >= 0 则选择 f
        errors_f.append(np.mean(L_ratios_f[:, -1] < 0))
        errors_g.append(np.mean(L_ratios_g[:, -1] >= 0))
    
    return np.array(errors_f), np.array(errors_g)

T_values = np.arange(1, 50, 1)
errors_f, errors_g = model_selection_analysis(T_values, model2_f, model2_g, N_sim=400)

fig, ax = plt.subplots()

ax.plot(T_values, errors_f, 'C0', linewidth=2, label=r'type I error $\alpha_T$')
ax.plot(T_values, errors_g, 'C1', linewidth=2, label=r'type II error $\beta_T$')
ax.plot(T_values, 0.5 * (errors_f + errors_g), 'g--', 
linewidth=2, label='average error')
ax.set_xlabel('$T$')
ax.set_ylabel('error probability')
ax.set_title('model selection errors')
ax.legend()
plt.tight_layout()
plt.show()
```

上面的决策规则是预先固定样本大小 $T$ 的。

{doc}`wald_friedman` 中描述的瓦尔德序贯概率比检验则不同，它在 $\log L_t$ 首次离开某个区间时才停止抽样。

该讲座的练习 {ref}`wald_friedman_ex2` 将此方法应用于两个VAR模型。

(var_kl_rate)=
## KL 散度率

上面各图中对数似然比路径的平均斜率就是库尔贝克-莱布勒散度率。

它们是 {doc}`likelihood_ratio_process` 中 {ref}`马尔可夫链的 KL 散度率 <lrp_markov>` 的向量自回归（VAR）对应版本。

{doc}`divergence_measures` 描述了库尔贝克-莱布勒散度。

设 $\Sigma_f = C_f C_f^\top$ 且 $\Sigma_g = C_g C_g^\top$，并设 $V_f$ 求解以下离散李雅普诺夫方程

$$
V_f = A_f V_f A_f^\top + \Sigma_f
$$

从而 $\mathcal{N}(0, V_f)$ 是模型 $f$ 下 $x_t$ 的平稳分布。

假设模型 $f$ 生成数据，并且 $x_0$ 服从 $\mathcal{N}(0, V_f)$ 分布，如示例 1 和示例 2 中所述。

那么 {eq}`eq:var_llr` 中的每个增量 $\ell_t$ 都具有相同的期望值

$$
\begin{aligned}
h_{KL}(f, g) & = E_f \left[ \log \frac{p_f(x_t | x_{t-1})}{p_g(x_t | x_{t-1})} \right] \\
& = \frac{1}{2} \left[ \operatorname{tr}(\Sigma_g^{-1} \Sigma_f) - n 
     + \log \frac{\det \Sigma_g}{\det \Sigma_f} 
     + \operatorname{tr}\left( \Sigma_g^{-1} (A_f - A_g) V_f (A_f - A_g)^\top \right) \right]
\end{aligned}
$$ (eq:var_kl_rate)

括号中前三项衡量冲击协方差矩阵 $\Sigma_f$ 和 $\Sigma_g$ 之间的差异。

最后一项衡量条件均值 $A_f x_{t-1}$ 和 $A_g x_{t-1}$ 之间的差异，该差异是在 $x_{t-1}$ 的平稳分布上取平均得到的。

由于 $\{x_t\}$ 在 $f$ 下是遍历的，

$$
\frac{1}{t} \log L_t \to h_{KL}(f, g) \quad \text{当 } t \to \infty \text{ 时几乎必然成立}
$$

当模型 $g$ 生成数据时，$\frac{1}{t} \log L_t \to -h_{KL}(g, f)$，其中 $h_{KL}(g, f)$ 是将 {eq}`eq:var_kl_rate` 中 $f$ 和 $g$ 的角色互换后得到的表达式。

练习 {ref}`lv_ex1` 要求你推导 {eq}`eq:var_kl_rate`，并通过模拟验证这些极限。

下面的函数计算 {eq}`eq:var_kl_rate` 的两个部分。

它从 `model_f.Σ_0` 中获取 $V_f$，因此 `model_f` 必须在创建时设置 `stationary=True`。

```{code-cell} ipython3
def kl_rate_terms(model_f, model_g):
    """
    Covariance and conditional-mean terms of the KL divergence rate
    h_KL(f, g), with x_t drawn from the stationary distribution of model f
    """
    n = model_f.A.shape[0]
    D = model_f.A - model_g.A
    V_f = model_f.Σ_0
    cov_term = 0.5 * (np.trace(model_g.CC_inv @ model_f.CC) - n
                      + model_g.log_det_CC - model_f.log_det_CC)
    mean_term = 0.5 * np.trace(model_g.CC_inv @ D @ V_f @ D.T)
    return cov_term, mean_term

def kl_rate_var(model_f, model_g):
    """
    KL divergence rate h_KL(f, g) between two stationary Gaussian VARs
    """
    return sum(kl_rate_terms(model_f, model_g))

examples = [("Example 1", model_f, model_g),
            ("Example 2", model2_f, model2_g)]

for name, m_f, m_g in examples:
    for label, m1, m2 in [("h_KL(f, g)", m_f, m_g), ("h_KL(g, f)", m_g, m_f)]:
        cov_term, mean_term = kl_rate_terms(m1, m2)
        print(f"{name}: {label} = {cov_term + mean_term:.4f}  "
              f"(covariance terms {cov_term:.4f}, mean term {mean_term:.4f})")
```

对于示例 2，$h_{KL}(g, f) = 0.961$ 几乎是 $h_{KL}(f, g) = 0.344$ 的三倍。

这就是为什么当 $g$ 生成数据时 $\log L_t$ 下降的速度比 $f$ 生成数据时 $\log L_t$ 上升的速度更快。

协方差项在两个速率中都占据主要部分。

两个模型都为 $(x_{1t} + x_{2t})/\sqrt{2}$ 的冲击赋予了方差 $0.16$。

模型 $g$ 为 $(x_{1t} - x_{2t})/\sqrt{2}$ 的冲击赋予了方差 $0.16$；模型 $f$ 则赋予其方差 $0.04$。

设 $r$ 为生成数据的模型下 $(x_{1t} - x_{2t})/\sqrt{2}$ 冲击的方差除以另一模型下该冲击方差的比值。

由于两个模型在另一方向上是一致的，协方差项等于 $\frac{1}{2}(r - 1 - \log r)$。

当 $r = 4$（自然 $= g$）时，该表达式等于 $0.807$；当 $r = 1/4$（自然 $= f$）时，该表达式等于 $0.318$。

库尔贝克-莱布勒散度对低估方差的模型的惩罚力度，要比对高估方差的模型更重。

在示例 1 中，两部分的贡献大致相等，且 $h_{KL}(g, f) = 0.208$ 超过了 $h_{KL}(f, g) = 0.139$。

让我们将示例 2 的理论速率与上面模拟得到的各路径中 $\frac{1}{T} \log L_T$ 的样本均值进行比较

```{code-cell} ipython3
T2 = L_ratios_ff.shape[1] - 1

print(f"nature = f: mean of log L_T / T = {L_ratios_ff[:, -1].mean() / T2:.4f}, "
      f"h_KL(f, g) = {kl_rate_var(model2_f, model2_g):.4f}")
print(f"nature = g: mean of log L_T / T = {L_ratios_gf[:, -1].mean() / T2:.4f}, "
      f"-h_KL(g, f) = {-kl_rate_var(model2_g, model2_f):.4f}")
```

样本均值包含了初始条件项 $\log L_0$ 除以 $T = 50$ 的结果，因此它们不一定与理论速率完全相等。

KL 速率描述的是增量 $\ell_t$ 的均值。

小样本 $T$ 下的误差概率取决于 $\log L_T$ 的整个分布。

```{code-cell} ipython3
for nature, L_ratios in [('f', L_ratios_ff), ('g', L_ratios_gf)]:
    ℓ = np.diff(L_ratios, axis=1).ravel()
    print(f"nature = {nature}: increments ℓ_t have mean {ℓ.mean():.2f}, "
          f"median {np.median(ℓ):.2f}, standard deviation {ℓ.std():.2f}")
```

在 $g$ 下，增量 $\ell_t$ 的均值接近 $-1$，中位数接近 $0$，标准差较大。

在 $f$ 下，其中位数超过均值，且标准差较小。

这就是为什么在上面的误差概率图中，尽管 $\log L_t$ 在 $g$ 下下降的速度比在 $f$ 下上升的速度更快，但对于较小的 $T$，$\beta_T$ 仍然超过 $\alpha_T$。

## 应用：萨缪尔森乘数-加速器模型

现在让我们来看萨缪尔森乘数-加速器模型。

该模型包括：

- 消费：$C_t = \gamma + a Y_{t-1}$ 其中 $a \in (0,1)$ 是边际消费倾向
- 投资：$I_t = b(Y_{t-1} - Y_{t-2}) + \sigma \epsilon_t$ 其中 $b > 0$ 是加速系数，且 $\{\epsilon_t\}$ 是独立同分布的标准正态随机变量序列
- 政府支出：$G_t = G$ (常数)

我们有国民收入恒等式

$$
Y_t = C_t + I_t + G_t
$$

{doc}`samuelson` 用 $\alpha$ 和 $\beta$ 来表示 $a$ 和 $b$。

那篇讲座将冲击 $\sigma \epsilon_t$ 直接加到 $Y_t$ 的二阶差分方程中。

我们将冲击置于投资方程中。

这样得到与 $Y_t$ 相同的差分方程，同时也确定了 $C_t$ 和 $I_t$ 对 $\epsilon_t$ 的反应方式。

将消费方程和投资方程代入国民收入恒等式，得出二阶差分方程：

$$
Y_t = (\gamma + G) + (a + b)Y_{t-1} - b Y_{t-2} + \sigma \epsilon_t
$$

令 $\rho_1 = a + b$ 且 $\rho_2 = -b$，我们得到：

$$
Y_t = (\gamma + G) + \rho_1 Y_{t-1} + \rho_2 Y_{t-2} + \sigma \epsilon_t
$$

当 $z^2 - \rho_1 z - \rho_2$ 的根位于单位圆内时，$Y_t$ 具有平稳分布，其均值为

$$
Y^* = \frac{\gamma + G}{1 - \rho_1 - \rho_2} = \frac{\gamma + G}{1 - a}
$$

当 $\sigma = 0$ 时，$Y_t$ 收敛到稳态 $Y^*$，此时投资为零，且 $Y^* = \gamma + a Y^* + G$。

为了符合我们的讨论，我们将其写成状态空间表示。

为了正确处理常数项，我们使用增广状态向量 $x_t = [1, Y_t, Y_{t-1}]^\top$：

$$
x_{t+1} = \begin{bmatrix} 
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
y_t = \begin{bmatrix} 
Y_t \\ 
C_t \\ 
I_t 
\end{bmatrix} = \begin{bmatrix} 
0 & 1 & 0 \\
\gamma & 0 & a \\
-(\gamma + G) & 1 & -a 
\end{bmatrix} \begin{bmatrix} 
1 \\ 
Y_t \\ 
Y_{t-1} 
\end{bmatrix}
$$

这给出了：

- $Y_t$ (总产出)
- $C_t = \gamma \cdot 1 + a Y_{t-1}$ (消费)
- $I_t = Y_t - C_t - G = b(Y_{t-1} - Y_{t-2}) + \sigma \epsilon_t$ (投资)

第三行利用国民收入恒等式，从 $Y_t$ 和 $Y_{t-1}$ 中恢复出 $I_t$，因此状态向量无需包含 $Y_{t-2}$。

矩阵 $CC^\top$ 的秩为一：给定 $x_t$，$x_{t+1}$ 的第一和第三个分量是已知的，只有 $Y_{t+1}$ 是随机的。

因此 {eq}`eq:cond_den` 不是关于 $\mathbb{R}^3$ 上勒贝格测度的密度。

`create_var_model` 用伪逆替代 $(CC^\top)^{-1}$，并在取对数行列式之前为 $CC^\top$ 加上 $10^{-10} I$。

这些正则化项在 $\ell_{t+1}$ 中相互抵消，$\ell_{t+1}$ 等于在给定 $x_t$ 的条件下，两个模型中 $Y_{t+1}$ 条件密度的对数比。

同样的方法也用于处理 $x_0$ 的奇异协方差矩阵 $\Sigma_0$，其第一个分量恒为 $1$。

一个观测到 $Y_t$ 和 $Y_{t-1}$ 的计量经济学家，相当于观测到了状态 $x_t$。

当计量经济学家只观测到状态向量的部分分量，或者带有误差地观测它们时，卡尔曼滤波器会产生用于计算似然的新息，正如 {doc}`kalman` 和 {doc}`kalman_filter_var` 中所述。

```{code-cell} ipython3
def samuelson_to_var(a, b, γ, G, σ):
    """
    将萨缪尔森模型参数转换为带扩展状态的VAR形式
    
    萨缪尔森模型:
    - Y_t = C_t + I_t + G
    - C_t = γ + a*Y_{t-1}
    - I_t = b*(Y_{t-1} - Y_{t-2}) + σ*ε_t
    
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
    
    # 观测矩阵(从[1, Y_t, Y_{t-1}]'中提取Y_t, C_t, I_t)
    G_obs = np.array([[0,         1,   0],    # Y_t
                      [γ,         0,   a],    # C_t = γ + a Y_{t-1}
                      [-(γ + G),  1,  -a]])   # I_t = Y_t - C_t - G
    
    return A, C, G_obs
```

我们在下面的代码单元中定义函数来获取初始条件并检查稳定性

当 `stationary_init=True` 时，初始状态 $[1, Y_0, Y_{-1}]^\top$ 具有平稳分布：$(Y_0, Y_{-1})$ 的均值为 $(Y^*, Y^*)$，协方差矩阵通过求解离散李雅普诺夫方程得到。

```{code-cell} ipython3
:tags: [hide-input]

def get_samuelson_initial_conditions(a, b, γ, G, σ, y_0=None, y_m1=None, 
                                    stationary_init=False):
    """
    获取萨缪尔森模型初始状态[1, Y_0, Y_{-1}]'的均值和协方差
    """
    # 稳态(投资为零): Y* = γ + a Y* + G
    y_ss = (γ + G) / (1 - a)
    
    if stationary_init:
        # (Y_t, Y_{t-1})的平稳分布
        ρ = np.array([[a + b, -b],
                      [1,      0]])
        if np.max(np.abs(np.linalg.eigvals(ρ))) >= 1:
            raise ValueError("萨缪尔森模型不是平稳的")
        V = linalg.solve_discrete_lyapunov(ρ, np.diag([σ**2, 0]))
        μ_0 = np.array([1.0, y_ss, y_ss])
        Σ_0 = np.zeros((3, 3))
        Σ_0[1:, 1:] = V
        return μ_0, Σ_0
    
    if y_0 is None:
        y_0 = y_ss
    if y_m1 is None:
        y_m1 = y_0 * 0.95
    
    # 初始均值
    μ_0 = np.array([1.0, y_0, y_m1])
    
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
        a, b, γ, G, σ, y_0, y_m1, stationary_init
    )
    
    # 创建VAR模型
    model = create_var_model(A, C, μ_0, Σ_0, stationary=False)
    is_stable, roots, max_root, dynamics = check_samuelson_stability(a, b)
    info = {
        'a': a, 'b': b, 'γ': γ, 'G': G, 'σ': σ,
        'ρ_1': a + b, 'ρ_2': -b,
        'steady_state': (γ + G) / (1 - a),
        'is_stable': is_stable,
        'roots': roots,
        'max_abs_root': max_root,
        'dynamics': dynamics
    }
    
    return model, G_obs, info

def simulate_samuelson(model, G_obs, T, rng, N_paths=1):
    """
    模拟萨缪尔森模型
    """
    # 模拟状态路径
    states = simulate_var(model, T, rng, N_paths)
    
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

两个模型具有相同的稳态 $Y^* = 20/(1 - 0.98) = 1000$。

两者都从 $Y_0 \sim \mathcal{N}(100, 25)$ 和 $Y_{-1} \sim \mathcal{N}(95, 25)$ 开始，远低于 $Y^*$。

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
states_f, obs_f = simulate_samuelson(model_sam_f, G_obs_f, T, rng, N_paths)
states_g, obs_g = simulate_samuelson(model_sam_g, G_obs_g, T, rng, N_paths)

output_paths_f = obs_f[:, :, 0] 
output_paths_g = obs_g[:, :, 0]
    
print("模型 f:")
print(f"  ρ_1 = a + b = {info_f['ρ_1']:.2f}")
print(f"  ρ_2 = -b = {info_f['ρ_2']:.2f}")
print(f"  特征根: {info_f['roots']}")
print(f"  动态特征: {info_f['dynamics']}")
print(f"  稳态: {info_f['steady_state']:.1f}")

print("\n模型 g:")
print(f"  ρ_1 = a + b = {info_g['ρ_1']:.2f}")
print(f"  ρ_2 = -b = {info_g['ρ_2']:.2f}")
print(f"  特征根: {info_g['roots']}")
print(f"  动态特征: {info_g['dynamics']}")
print(f"  稳态: {info_g['steady_state']:.1f}")


fig, ax = plt.subplots(1, 1)

for i in range(min(20, N_paths)):
    ax.plot(output_paths_f[i], alpha=0.6, color='C0', linewidth=0.8)
    ax.plot(output_paths_g[i], alpha=0.6, color='C1', linewidth=0.8)
ax.axhline(info_f['steady_state'], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('$t$')
ax.set_ylabel('$Y_t$')
ax.legend(['模型 f', '模型 g'], loc='lower right')
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

在左图中，数据由 $f$ 生成，对数似然比趋向正无穷。

在右图中，数据由 $g$ 生成，对数似然比趋向负无穷。

在这两种情况下，似然比过程最终都能帮助我们选择正确的模型。

区分这两个模型的信息几乎全部来自于使 $Y_t$ 从 $Y_0 \approx 100$ 向 $Y^* = 1000$ 过渡的瞬态过程。

这两个模型共享相同的 $a$、$\gamma$、$G$、$\sigma$ 和 $Y^*$；它们仅在加速系数 $b$ 上有所不同。

它们对 $Y_{t+1}$ 的预测相差 $(b_f - b_g)(Y_t - Y_{t-1}) = 0.05 \, \Delta Y_t$。

由于两个模型共享相同的 $\sigma$，在模型 $f$ 下 $\log L_t$ 的预期增量为

$$
E_f [\ell_{t+1} | x_t] = \frac{(b_f - b_g)^2 (\Delta Y_t)^2}{2 \sigma^2}
$$

而 $E_g [-\ell_{t+1} | x_t]$ 由同样的公式给出。

在瞬态过程中，相对于 $\sigma = 0.5$ 而言，$|\Delta Y_t|$ 较大。

在平稳分布中，$\Delta Y_t$ 较小，且KL散度率为 $h_{KL}(f, g) = (b_f - b_g)^2 E_f[(\Delta Y_t)^2] / (2\sigma^2)$。

下面的函数根据使用 `stationary_init=True` 创建的模型的 $x_t$ 平稳分布，计算该散度率。

```{code-cell} ipython3
def samuelson_kl_rate(model_f, model_g):
    """
    针对两个使用stationary_init=True创建的萨缪尔森模型的KL散度率
    h_KL(f, g)；x_{t+1}中只有Y分量是随机的
    """
    σ2_f, σ2_g = model_f.CC[1, 1], model_g.CC[1, 1]
    d = model_f.A[1] - model_g.A[1]      # Y_{t+1}预测的差异
    M_f = model_f.Σ_0 + np.outer(model_f.μ_0, model_f.μ_0)   # E_f[x_t x_t']
    return 0.5 * (σ2_f / σ2_g - 1 + np.log(σ2_g / σ2_f) 
                  + d @ M_f @ d / σ2_g)

model_stat_f, _, _ = create_samuelson_var_model(
    a_f, b_f, γ_f, G_f, σ_f, stationary_init=True)
model_stat_g, _, _ = create_samuelson_var_model(
    a_g, b_g, γ_g, G_g, σ_g, stationary_init=True)

h_fg = samuelson_kl_rate(model_stat_f, model_stat_g)
h_gf = samuelson_kl_rate(model_stat_g, model_stat_f)

print(f"median log L_{T}: nature = f: {np.median(L_ratios_ff[:, -1]):.1f}, "
      f"nature = g: {np.median(L_ratios_gf[:, -1]):.1f}")
print(f"stationary sd of ΔY_t under f: "
      f"{np.sqrt(2 * (model_stat_f.Σ_0[1, 1] - model_stat_f.Σ_0[1, 2])):.2f}")
print(f"h_KL(f, g) = {h_fg:.4f}, h_KL(g, f) = {h_gf:.4f}")
print(f"{T} × h_KL(f, g) = {T * h_fg:.2f}, {T} × h_KL(g, f) = {T * h_gf:.2f}")

# 沿模拟路径计算预期增量 E[ℓ_t | x_{t-1}]
d = model_sam_f.A[1] - model_sam_g.A[1]
σ2 = σ_f**2
inc_f = ((states_f[:, :-1, :] @ d)**2 / (2 * σ2)).mean(axis=0)
inc_g = ((states_g[:, :-1, :] @ d)**2 / (2 * σ2)).mean(axis=0)
t_grid = np.arange(1, T + 1)

print(f"largest mean expected increment: "
      f"nature = f: {inc_f.max():.1f} at t = {t_grid[inc_f.argmax()]}, "
      f"nature = g: {inc_g.max():.1f} at t = {t_grid[inc_g.argmax()]}")

fig, ax = plt.subplots()
ax.semilogy(t_grid, inc_f, color='C0', lw=2, 
            label=r'$E_f[\ell_t | x_{t-1}]$, mean over paths')
ax.semilogy(t_grid, inc_g, color='C1', lw=2, 
            label=r'$E_g[-\ell_t | x_{t-1}]$, mean over paths')
ax.axhline(h_fg, color='C0', ls='--', lw=2, label=r'$h_{KL}(f, g)$')
ax.axhline(h_gf, color='C1', ls='--', lw=2, label=r'$h_{KL}(g, f)$')
ax.set_xlabel('$t$')
ax.set_ylabel('expected increment')
ax.legend()
plt.show()
```

在 $t = 10$ 附近，即 $Y_t$ 上升最快时，模型 $f$ 下的预期增量达到约 $34$，模型 $g$ 下达到约 $24$。

只有在振荡的转折点附近，即 $\Delta Y_t$ 接近零时，它们才接近平稳速率 $h_{KL}(f, g)$ 和 $h_{KL}(g, f)$。

模型 $f$ 下 $\Delta Y_t$ 的平稳标准差为 $1.15$。

从平稳分布来看，$50$ 期只能使 $E_f [\log L_t]$ 增加约 $0.33$，而瞬态过程则能使 $\log L_{50}$ 的中位数提高到约 $400$。

练习 {ref}`lv_ex2` 让两个模型都从各自的平稳分布开始，研究这两个模型变得难以区分的程度。

## 练习

```{exercise}
:label: lv_ex1

推导公式 {eq}`eq:var_kl_rate`，即 KL 散度率 $h_{KL}(f, g)$ 的公式。

使用两个多元正态分布的 Kullback-Leibler 散度公式：

$$
KL\bigl(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)\bigr)
= \frac{1}{2} \left[ \operatorname{tr}(\Sigma_2^{-1} \Sigma_1) - n 
  + (\mu_1 - \mu_2)^\top \Sigma_2^{-1} (\mu_1 - \mu_2) 
  + \log \frac{\det \Sigma_2}{\det \Sigma_1} \right]
$$

然后通过模拟验证：当 $f$ 生成数据时，$\frac{1}{t} \log L_t \to h_{KL}(f, g)$；当 $g$ 生成数据时，$\frac{1}{t} \log L_t \to -h_{KL}(g, f)$。

针对示例 1 和示例 2，取 $T = 2000$ 完成此验证。
```

```{solution-start} lv_ex1
:class: dropdown
```

在模型 $f$ 下，$x_t | x_{t-1} \sim \mathcal{N}(A_f x_{t-1}, \Sigma_f)$；在模型 $g$ 下，$x_t | x_{t-1} \sim \mathcal{N}(A_g x_{t-1}, \Sigma_g)$。

因此 $E_f [\ell_t | x_{t-1}] = KL\bigl(\mathcal{N}(A_f x_{t-1}, \Sigma_f), \mathcal{N}(A_g x_{t-1}, \Sigma_g)\bigr)$。

在上述公式中令 $\mu_1 - \mu_2 = (A_f - A_g) x_{t-1}$，得到

$$
E_f [\ell_t | x_{t-1}] = \frac{1}{2} \left[ \operatorname{tr}(\Sigma_g^{-1} \Sigma_f) - n 
  + \log \frac{\det \Sigma_g}{\det \Sigma_f}
  + x_{t-1}^\top (A_f - A_g)^\top \Sigma_g^{-1} (A_f - A_g) x_{t-1} \right]
$$

在 $f$ 下，$x_{t-1} \sim \mathcal{N}(0, V_f)$。

对任意矩阵 $M$，$E [x_{t-1}^\top M x_{t-1}] = \operatorname{tr}(M V_f)$。

令 $M = (A_f - A_g)^\top \Sigma_g^{-1} (A_f - A_g)$，并利用迹的循环性质，有

$$
E_f [x_{t-1}^\top M x_{t-1}] = \operatorname{tr}\left( \Sigma_g^{-1} (A_f - A_g) V_f (A_f - A_g)^\top \right)
$$

对 $E_f [\ell_t | x_{t-1}]$ 关于 $x_{t-1}$ 取期望，即得 {eq}`eq:var_kl_rate`。

由于 $x_{t-1}$ 在每个 $t$ 都服从相同的分布 $\mathcal{N}(0, V_f)$，所以每个增量 $\ell_t$ 的均值均为 $h_{KL}(f, g)$，且

$$
E_f [\log L_t] = E_f \left[ \log \frac{p_f(x_0)}{p_g(x_0)} \right] + t \, h_{KL}(f, g)
$$

遍历定理表明 $\frac{1}{t} \sum_{s=1}^t \ell_s \to h_{KL}(f, g)$ 几乎必然成立，且 $\frac{1}{t} \log \frac{p_f(x_0)}{p_g(x_0)} \to 0$。

以下是模拟过程。

```{code-cell} ipython3
T_long = 2000
N_long = 20
t_grid = np.arange(1, T_long + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, m_f, m_g) in zip(axes, examples):
    for nature, m_true, rate, color in [
            ('f', m_f, kl_rate_var(m_f, m_g), 'C0'),
            ('g', m_g, -kl_rate_var(m_g, m_f), 'C1')]:
        paths = simulate_var(m_true, T_long, rng, N_long)
        log_L = compute_likelihood_ratio_var(paths, m_f, m_g)
        avg = log_L[:, 1:] / t_grid
        ax.plot(t_grid, avg.T, color=color, alpha=0.2, lw=1)
        ax.axhline(rate, color=color, ls='--', lw=2, 
                   label=f'theory (nature = {nature})')
        print(f"{name}, nature = {nature}: mean of log L_T / T = "
              f"{avg[:, -1].mean():.4f}, theory = {rate:.4f}")
    ax.set_xscale('log')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\log L_t / t$')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
```

在全部四种情形中，$\frac{1}{t} \log L_t$ 都稳定在理论比率附近。

```{solution-end}
```

```{exercise}
:label: lv_ex2

在萨缪尔森示例中，两个模型都从 $Y_0 \approx 100$ 开始，远低于共同的稳态 $Y^* = 1000$。

现在改为利用通过 `stationary_init=True` 创建的模型 `model_stat_f` 和 `model_stat_g`，让两个模型都从各自的平稳分布开始。

1. 对每个模型分别模拟 500 条长度为 $T = 50$ 的路径。报告 $\log L_{50}$ 的中位数以及误差概率 $\alpha_{50}$、$\beta_{50}$，以及 $\frac{1}{2}(\alpha_{50} + \beta_{50})$。将其与从 $Y_0 \approx 100$ 开始的路径结果进行比较。
1. 利用 `samuelson_kl_rate` 以及 $(Y_0, Y_{-1})$ 两个平稳分布之间的 Kullback-Leibler 散度，计算使得 $E_f [\log L_t] = 5$ 成立的周期数 $t$。
1. 模拟该长度的路径，并绘制 $\alpha_t$ 和 $\beta_t$ 的图像。
```

```{solution-start} lv_ex2
:class: dropdown
```

以下是两种初始条件下，在 $T = 50$ 时的误差概率。

```{code-cell} ipython3
T_s, N_s = 50, 500

paths_stat_f = simulate_var(model_stat_f, T_s, rng, N_s)
paths_stat_g = simulate_var(model_stat_g, T_s, rng, N_s)
log_L_stat_f = compute_likelihood_ratio_var(paths_stat_f, 
                                            model_stat_f, model_stat_g)
log_L_stat_g = compute_likelihood_ratio_var(paths_stat_g, 
                                            model_stat_f, model_stat_g)

for label, log_L_f, log_L_g in [
        ("stationary start", log_L_stat_f, log_L_stat_g),
        ("Y_0 ≈ 100", L_ratios_ff, L_ratios_gf)]:
    α = np.mean(log_L_f[:, -1] < 0)
    β = np.mean(log_L_g[:, -1] >= 0)
    print(f"{label}:")
    print(f"  median log L_50: nature = f: {np.median(log_L_f[:, -1]):.3f}, "
          f"nature = g: {np.median(log_L_g[:, -1]):.3f}")
    print(f"  α_50 = {α:.3f}, β_50 = {β:.3f}, "
          f"average error = {0.5 * (α + β):.3f}")
```

从平稳分布出发，$\log L_{50}$ 的中位数都在零附近 $0.5$ 的范围内，$\alpha_{50}$ 大约为 $0.4$，并且似然比检验大约有三分之一的样本选错了模型。

而从 $Y_0 \approx 100$ 出发，每个模型的全部 $50$ 个样本都能正确选出模型。

在 $f$ 下，$E_f [\log L_t] = KL_0 + t \, h_{KL}(f, g)$，其中 $KL_0$ 是在 $f$ 与在 $g$ 下 $(Y_0, Y_{-1})$ 平稳分布之间的 Kullback-Leibler 散度。

```{code-cell} ipython3
def kl_gauss(μ_1, Σ_1, μ_2, Σ_2):
    """
    KL divergence of N(μ_2, Σ_2) from N(μ_1, Σ_1)
    """
    n = len(μ_1)
    Σ_2_inv = np.linalg.inv(Σ_2)
    d = μ_1 - μ_2
    return 0.5 * (np.trace(Σ_2_inv @ Σ_1) - n + d @ Σ_2_inv @ d
                  + np.log(np.linalg.det(Σ_2) / np.linalg.det(Σ_1)))

KL_0 = kl_gauss(model_stat_f.μ_0[1:], model_stat_f.Σ_0[1:, 1:],
                model_stat_g.μ_0[1:], model_stat_g.Σ_0[1:, 1:])
t_star = int(np.ceil((5 - KL_0) / h_fg))

print(f"KL_0 = {KL_0:.4f}, h_KL(f, g) = {h_fg:.5f}")
print(f"E_f[log L_50]: theory = {KL_0 + 50 * h_fg:.3f}, "
      f"simulation = {log_L_stat_f[:, -1].mean():.3f}")
print(f"stationary start: E_f[log L_t] = 5 at t = {t_star}")

# Transient from Y_0 ≈ 100: cumulative mean expected increments
t_transient = np.argmax(np.cumsum(inc_f) >= 5) + 1
print(f"Y_0 ≈ 100: cumulative expected increments reach 5 at t = {t_transient}")
```

平均而言，大约 $750$ 期的平稳数据所含有关于 $b$ 的信息量，相当于从 $Y_0 \approx 100$ 开始的过渡过程中前三期所含的信息量。

```{code-cell} ipython3
N_star = 200

paths_f = simulate_var(model_stat_f, t_star, rng, N_star)
paths_g = simulate_var(model_stat_g, t_star, rng, N_star)
log_L_f = compute_likelihood_ratio_var(paths_f, model_stat_f, model_stat_g)
log_L_g = compute_likelihood_ratio_var(paths_g, model_stat_f, model_stat_g)

α_t = np.mean(log_L_f < 0, axis=0)
β_t = np.mean(log_L_g >= 0, axis=0)

fig, ax = plt.subplots()
ax.plot(α_t, color='C0', lw=2, label=r'$\alpha_t$')
ax.plot(β_t, color='C1', lw=2, label=r'$\beta_t$')
ax.plot(0.5 * (α_t + β_t), 'g--', lw=2, label='average error')
ax.set_xlabel('$t$')
ax.set_ylabel('error probability')
ax.legend()
plt.show()

print(f"at t = {t_star}: α = {α_t[-1]:.3f}, β = {β_t[-1]:.3f}")
```

即使经过大约 $750$ 期，两个误差概率仍大致保持在 $0.1$ 左右。

```{solution-end}
```

```{exercise}
:label: lv_ex3

假设示例 2 中的 $f$ 和 $g$ 都不是数据的生成模型。

而是由第三个矩阵为 $(A_h, C_h)$ 的 VAR 模型 $h$ 来生成数据。

证明

$$
\frac{1}{t} \log L_t \to h_{KL}(h, g) - h_{KL}(h, f)
$$

几乎必然成立，其中 $h_{KL}(h, \cdot)$ 是用 $h$ 替代 $f$ 后的 {eq}`eq:var_kl_rate`。

这是 {doc}`likelihood_ratio_process` 中 {ref}`KL_link` 公式的 VAR 对应形式。

针对 $(A_h, C_h) = (A_f, C_g)$ 和 $(A_h, C_h) = (A_g, C_f)$ 两种情形，通过模拟验证该极限。

在每种情形下，似然比会选择哪个模型？
```

```{solution-start} lv_ex3
:class: dropdown
```

写出

$$
\ell_t = \log \frac{p_h(x_t | x_{t-1})}{p_g(x_t | x_{t-1})} - \log \frac{p_h(x_t | x_{t-1})}{p_f(x_t | x_{t-1})}
$$

在 $h$ 下取期望，其中 $x_{t-1}$ 取自 $h$ 的平稳分布，得到 $E_h [\ell_t] = h_{KL}(h, g) - h_{KL}(h, f)$。

随后利用 $h$ 下的遍历定理即可得到该几乎必然极限。

```{code-cell} ipython3
model_h1 = create_var_model(model2_f.A, model2_g.C)
model_h2 = create_var_model(model2_g.A, model2_f.C)

T_h, N_h = 1000, 20
t_grid = np.arange(1, T_h + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cases = [("(A_f, C_g)", r'$(A_h, C_h) = (A_f, C_g)$', model_h1),
         ("(A_g, C_f)", r'$(A_h, C_h) = (A_g, C_f)$', model_h2)]

for ax, (name, title, m_h) in zip(axes, cases):
    K_f = kl_rate_var(m_h, model2_f)
    K_g = kl_rate_var(m_h, model2_g)
    paths = simulate_var(m_h, T_h, rng, N_h)
    log_L = compute_likelihood_ratio_var(paths, model2_f, model2_g)
    avg = log_L[:, 1:] / t_grid
    ax.plot(t_grid, avg.T, color='C2', alpha=0.3, lw=1)
    ax.axhline(K_g - K_f, color='k', ls='--', lw=2,
               label=r'$h_{KL}(h, g) - h_{KL}(h, f)$')
    ax.set_xscale('log')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\log L_t / t$')
    ax.set_title(title)
    ax.legend()
    print(f"(A_h, C_h) = {name}: h_KL(h, f) = {K_f:.4f}, "
          f"h_KL(h, g) = {K_g:.4f}, difference = {K_g - K_f:.4f}, "
          f"mean of log L_T / T = {avg[:, -1].mean():.4f}")

plt.tight_layout()
plt.show()
```

当 $(A_h, C_h) = (A_f, C_g)$ 时，$\log L_t \to -\infty$，似然比选择了 $g$，尽管 $f$ 具有正确的转移矩阵。

当 $(A_h, C_h) = (A_g, C_f)$ 时，$\log L_t \to +\infty$，似然比选择了 $f$，尽管 $g$ 具有正确的转移矩阵。

在这两种情形下，似然比都选择了具有正确冲击协方差矩阵的模型。

当 $(A_h, C_h) = (A_f, C_g)$ 时，$h_{KL}(h, f)$ 仅由协方差项构成，而 $h_{KL}(h, g)$ 仅由条件均值项构成。

当 $(A_h, C_h) = (A_g, C_f)$ 时，$f$ 与 $g$ 的角色互换。

在这两种情形下，协方差项都更大，这与上文针对示例 2 的分解结果一致。

```{solution-end}
```
