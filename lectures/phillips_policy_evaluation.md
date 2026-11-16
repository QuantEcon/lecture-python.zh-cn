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
  title: 计量经济学政策评估
  headings:
    Overview: 概述
    The model: 模型
    The model::The actual Phillips curve: 实际菲利普斯曲线
    The model::The government: 政府
    The likelihood function: 似然函数
    Data: 数据
    Beliefs and recommendations: 信念与建议
    Estimates: 估计结果
    Recommendations and misses: 建议与误差
    Recommendations and misses::The induction hypothesis emerges: 归纳假设逐渐显现
    Misses favor vindication: 误差支持平反说
    Misses favor vindication::How robust are the misses?: 这些误差有多稳健？
    Misses favor vindication::A positive, not a normative, conclusion: 这是实证结论，而非规范性结论
    What came later: 后续发展
    Exercises: 练习
---

(phillips_policy_evaluation)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 计量经济学政策评估

```{index} single: Phillips Curve; Econometric Policy Evaluation
```

```{contents} Contents
:depth: 2
```

## 概述

本讲座遵循 {cite:t}`Sargent1999` 第 9 章《计量经济学政策评估》的内容。

前面几讲构建了该书第 8 章中的适应性政府，并通过模拟对其进行了研究（{doc}`phillips_learning`、{doc}`phillips_escaping_nash`、{doc}`phillips_priors`）。

第 9 章将同样的模型应用于美国数据，按照 {cite:t}`Chung1990` 的做法，在古典和凯恩斯两种识别方案下通过最大似然法对其进行估计。

这是该书唯一一次用数据对 {doc}`phillips_two_stories` 中的"平反"故事进行结构性检验。

问题是反事实的：如果 20 世纪 60 年代和 70 年代的政策顾问拟合了一条分布滞后的菲利普斯曲线，对旧数据打折扣，并将估计值代入 {doc}`phillips_adaptive` 中的费尔普斯问题，他会给出怎样的建议？

估计出的模型能够直接回答这个问题，因为在模型中，通货膨胀的拟合值*就是*政府的建议。

一旦增益和贴现因子固定下来，似然函数中就没有描述信念的自由参数了：我们像政府本身那样，仅从数据中计算出政府的信念和建议。

我们重现了该书的主要发现：

* 两个模型对通货膨胀的拟合都不好，古典模型拟合得更差；
* 误差是系统性的，整个 20 世纪 70 年代，建议的通货膨胀率都低于实际通货膨胀率，这正是"估计模型的误差模式支持平反说"这一说法的含义；
* 到 20 世纪 70 年代中期，凯恩斯政府估计的菲利普斯曲线已满足归纳假设，其计划在沃尔克上任前数年就已要求低通货膨胀；
* 古典政府则一直建议低通货膨胀，这正是其建议"本会更加及时"的原因。

该书没有报告其滞后阶数，也没有说明如何将数据转换为季度频率，我们会指出这些选择在何处产生了影响。

让我们先导入一些库。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2
```

## 模型

该模型将 {doc}`phillips_self_confirming` 中的实际菲利普斯曲线与 {doc}`phillips_learning` 中的适应性政府结合起来；所用符号见 {ref}`phillips_notation`。

### 实际菲利普斯曲线

设 $U_t$ 为失业率，$y_t$ 为通货膨胀率，$x_t$ 为公众对通货膨胀的预期，经济遵循一条自然率菲利普斯曲线，其中意外通货膨胀 $v_{2t} = y_t - x_t$ 具有几何分布滞后，并伴有序列相关的扰动：

```{math}
:label: ppe_actual

U_t = U^* - \frac{\theta}{1 - \rho_2 L}\, v_{2t} + \frac{1}{1 - \rho_1 L}\, v_{1t},
```

其中 $L$ 是滞后算子，$|\rho_1| < 1$ 且 $|\rho_2| < 1$。

两边乘以 $(1-\rho_1 L)(1-\rho_2 L)$，得到该书中编号为（48）的形式：

```{math}
:label: ppe_actual48

U_t = U^*(1-\rho_1)(1-\rho_2) + (\rho_1+\rho_2) U_{t-1} - \rho_1\rho_2 U_{t-2}
      + (1-\rho_2 L) v_{1t} - \theta (1-\rho_1 L) v_{2t} .
```

冲击向量 $v_t = (v_{1t}, v_{2t})^\top$ 是高斯白噪声，具有不受约束的协方差矩阵 $V_v$。

该书第 7 章和第 8 章为了阐明理论要点，设定 $\rho_1 = \rho_2 = 0$，并使 $V_v$ 为对角矩阵；第 9 章则将两者都放开了。

### 政府

设 $X_{t-1} = \begin{bmatrix} U_{t-1} & \cdots & U_{t-m_U} & y_{t-1} & \cdots & y_{t-m_y} & 1 \end{bmatrix}^\top$ 为截至 $t-1$ 及更早期的信息集合。

与 {doc}`phillips_adaptive` 中一样，政府拟合两种简化型菲利普斯曲线之一，二者仅拟合方向不同：

```{math}
:label: ppe_fits

\text{古典：}\quad U_t = \gamma^\top X_{C,t} + \varepsilon_{C,t},
\qquad
\text{凯恩斯：}\quad y_t = \beta^\top X_{K,t} + \varepsilon_{K,t},
```

其中 $X_{C,t} = \begin{bmatrix} y_t & X_{t-1}^\top \end{bmatrix}^\top$，$X_{K,t} = \begin{bmatrix} U_t & X_{t-1}^\top \end{bmatrix}^\top$。

政府每期通过常增益递归最小二乘法重新估计其曲线，古典方案的更新公式为

```{math}
:label: ppe_rls

\begin{aligned}
\gamma_t &= \gamma_{t-1} + g\, R_{XC,t}^{-1} X_{C,t}\left(U_t - \gamma_{t-1}^\top X_{C,t}\right), \\
R_{XC,t} &= R_{XC,t-1} + g \left(X_{C,t} X_{C,t}^\top - R_{XC,t-1}\right),
\end{aligned}
```

增益为常数 $g = 1 - \lambda$；凯恩斯方案以同样的方式更新 $\beta_t$ 和 $R_{XK,t}$。

凯恩斯政府通过对菲利普斯曲线求逆，将其估计值转换为古典形式：如果 $\beta_1$ 是 $U_t$ 的系数，$\beta_{-1}$ 收集其余系数，则 $\gamma(\beta) = \begin{bmatrix} \beta_1^{-1} & -\beta_{-1}^\top/\beta_1 \end{bmatrix}^\top$。

每期政府都表现得好像其当前估计值将永远支配菲利普斯曲线，并通过求解费尔普斯问题来设定通货膨胀率，即在其估计曲线的约束下使 $\mathbb{E}\sum_t \delta^t (U_t^2 + y_t^2)$ 最小化，这就得到了规则

```{math}
:label: ppe_rule

y_t = h(\gamma_{t-1}) X_{t-1} + v_{2t},
```

其中 $\gamma_{t-1}$ 是古典估计值或 $\gamma(\beta_{t-1})$，$v_{2t}$ 是控制误差。

遵循 {cite:t}`Sims1988` 和 {cite:t}`Chung1990` 的做法，公众知道这一规则，因此 $x_t = h(\gamma_{t-1}) X_{t-1}$，而控制误差 $v_{2t}$ 也就是式 {eq}`ppe_actual` 中的通货膨胀意外。

(ppe_estimation)=
## 似然函数

要模拟该模型，我们需要抽取冲击 $\{v_t\}$，并计算出 $(y_t, U_t)$ 以及政府信念的历史路径。

要进行估计，我们则反过来操作：从初始信念出发，利用数据 $\{y_t, U_t\}$ 求解能够产生这些数据的冲击，

```{math}
:label: ppe_inverse

\begin{aligned}
v_{2t} &= y_t - h(\gamma_{t-1}) X_{t-1}, \\
v_{1t} - \theta v_{2t} &= U_t - (\rho_1+\rho_2) U_{t-1} + \rho_1\rho_2 U_{t-2}
       + \rho_2 v_{1,t-1} - \rho_1 \theta v_{2,t-1} - U^*(1-\rho_1)(1-\rho_2),
\end{aligned}
```

同时按照式 {eq}`ppe_rls` 更新 $\gamma_t$（或 $\beta_t$）；这些就是该书中的公式（105）。

给定过去的信息，从 $v_t$ 到 $(U_t, y_t)$ 的映射是具有单位对角元的三角形映射，因此其雅可比行列式为 1，长度为 $T$ 的样本的高斯对数似然为

```{math}
:label: ppe_loglik

\log L = -T \log 2\pi - \frac{T}{2}\log\det V_v - \frac{1}{2}\sum_{t=1}^T v_t^\top V_v^{-1} v_t .
```

正如该书所建议的，我们将 $V_v$ 消去：其极大值为 $\hat V_v = T^{-1}\sum_t v_t v_t^\top$，于是 $\log L = -T\log 2\pi - \frac{T}{2}\log\det \hat V_v - T$。

```{note}
该书第 9 章附录中把若干密度之和的对数写成了对数密度之和的意思，并将常数项给成 $-\frac{T}{2}\log 2\pi$，这对标量观测值是正确的；但二元密度的常数项应为 $-T\log 2\pi$。

这两处笔误都不影响估计结果，但第二处会使 $\log L$ 的水平发生偏移。
```

该书估计了 $(\theta, U^*, \rho_1, \rho_2, V_v)$，并将贴现因子固定为 $\delta = .999$，遗忘因子固定为 $\lambda = .985$，因此增益为 $g = .015$。

请注意，列表中*没有*任何描述信念的参数。

在 $\lambda$、$\delta$ 和初始信念固定的情况下，无论待估参数取何值，政府的信念 $\gamma_t$、其建议 $h(\gamma_{t-1})X_{t-1}$ 以及因此产生的误差 $v_{2t}$ 都完全由数据本身计算得出。

这带来两个后果。

1. 由于 $\det \hat V_v$ 等于 $v_{2t}$ 的均方乘以 $v_{1t}$ 中未被 $v_{2t}$ 解释部分的均方，参数 $(\theta, U^*, \rho_1, \rho_2)$ 是在把政府的误差视为给定的情况下，从失业方程中估计出来的。
1. 两种识别方案之间的比较主要是其误差大小的比较，也就是建议与实际发生情况之间的差距大小的比较。

这是理性预期计量经济学中常见的一种跨方程约束：政府的规则是由其计量经济学估计和费尔普斯问题共同确定的，而不是自由设定的。

## 数据

第 9 章使用了第 1 章的度量指标：消费者价格指数（所有项目）和 20 岁及以上白人男性的失业率，这些序列绘制在 {doc}`phillips_two_stories` 中。

尽管第 1 章绘制的是月度数据，但第 9 章的估计似乎是季度数据：

* 样本期以季度标注（印刷为"1995I–1997I"，显然是 1955I–1997I 的误印，因为图形从 1955 年开始）；
* 图 9.2b 提到"1960 年第一季度"；
* 季度数据能很好地重现该书的估计结果和图形，而月度数据则不能，原因我们将在{ref}`下文 <ppe_misses>`中解释。

因此，我们通过取每个季度的最后一个月将月度序列转换为季度数据，并将通货膨胀率度量为 CPI 的年化百分比变化。

由于我们每季度抽取一个月的数据，所以我们使用经季节调整的指数，而不是 {doc}`phillips_two_stories` 中绘制的未经调整的指数。

```{code-cell} ipython3
def read_fred(series_id):
    "Read a monthly series from FRED."
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    s = pd.read_csv(url, index_col=0, parse_dates=True)[series_id]
    return pd.to_numeric(s, errors='coerce')

cpi_monthly = read_fred('CPIAUCSL')     # CPI, all items, seasonally adjusted
u_monthly = read_fred('LNS14000028')    # unemployment rate, white men 20+

def make_data(how='last', end='1997-03'):
    "Inflation (annualized percent) and unemployment at quarterly or monthly frequency."
    if how == 'monthly':
        cpi, u, scale = cpi_monthly, u_monthly, 1200
    elif how == 'last':                         # last month of each quarter
        cpi, u, scale = (cpi_monthly.resample('QS').last(),
                         u_monthly.resample('QS').last(), 400)
    else:                                       # quarterly averages
        cpi, u, scale = (cpi_monthly.resample('QS').mean(),
                         u_monthly.resample('QS').mean(), 400)
    y = scale * np.log(cpi).diff()
    data = pd.concat([y.rename('y'), u.rename('U')], axis=1).dropna()
    return data.loc[:end]

data = make_data()
print(f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}: {len(data)} quarters")
```

失业率序列从 1954 年开始，而政府的回归需要几年的数据才能使其估计值有意义。

因此，我们将 1954—1957 年作为政府信念的训练样本，并从 1958 年第一季度到 1997 年第一季度对似然函数进行评估。

该书没有报告其滞后阶数 $m_U$ 和 $m_y$；其第 8 章的模拟使用了各两个滞后期。

我们使用一个滞后期的失业率和两个滞后期的通货膨胀率，在我们尝试过的滞后阶数中，这一设置使我们的估计结果最接近该书的结果；下文{ref}`稳健性检验表 <ppe_misses>`展示了各使用两个滞后期时结果的变化。

## 信念与建议

费尔普斯问题是 {doc}`phillips_learning` 中带贴现的 LQ 问题，推广到任意滞后阶数的情形。

将所认为的菲利普斯曲线写为 $U_t = \gamma_0 y_t + c^\top s_t$，状态为 $s_t = X_{t-1}$，于是每期损失为 $s_t^\top c c^\top s_t + (\gamma_0^2 + 1) y_t^2 + 2 \gamma_0 y_t c^\top s_t$。

我们使用与 {doc}`phillips_learning` 相同的求解器：在带贴现的问题中，只有 $A$ 和 $B$ 乘以 $\sqrt{\delta}$，交叉项 $N$ 未经缩放直接传入黎卡提求解器。

```{code-cell} ipython3
def lq_matrices(γ, mU, my):
    """
    Believed law of motion s_{t+1} = A s_t + B y_t for the state
    s_t = [U_{t-1},...,U_{t-mU}, y_{t-1},...,y_{t-my}, 1]
    when the believed Phillips curve is U_t = γ[0] y_t + γ[1:] @ s_t.
    """
    γ0, c = γ[0], γ[1:]
    n = mU + my + 1
    A, B = np.zeros((n, n)), np.zeros((n, 1))
    if mU > 0:
        A[0, :], B[0, 0] = c, γ0                 # U_t
        for j in range(1, mU):
            A[j, j - 1] = 1.0
    if my > 0:
        B[mU, 0] = 1.0                           # y_t
        for j in range(mU + 1, mU + my):
            A[j, j - 1] = 1.0
    A[-1, -1] = 1.0                              # constant
    return A, B

def phelps_h(γ, δ, mU, my):
    "Phelps rule y_t = h @ s_t for the believed Phillips curve γ."
    γ0, c = γ[0], γ[1:]
    A, B = lq_matrices(γ, mU, my)
    R = np.outer(c, c)
    Q = np.array([[γ0**2 + 1.0]])
    N = (γ0 * c).reshape(1, -1)
    Ad, Bd = np.sqrt(δ) * A, np.sqrt(δ) * B
    P = solve_discrete_are(Ad, Bd, R, Q, s=N.T)
    F = np.linalg.solve(Q + Bd.T @ P @ Bd, Bd.T @ P @ Ad + N)
    return -F.ravel()

def γ_of_β(β):
    "Invert a Keynesian Phillips curve into classical form."
    return np.concatenate([[1 / β[0]], -β[1:] / β[0]])
```

在弥散初始条件 $R_{XC,-1} = 0$ 下，递归式 {eq}`ppe_rls` 给出的是*折扣最小二乘法*：$\gamma_t$ 使 $\sum_{s \le t} \lambda^{t-s} (U_s - \gamma^\top X_{C,s})^2$ 最小化，因此一个 $k$ 个季度前的观测值权重为 $\lambda^k$。

下面的函数就是以这种方式从数据的第一个季度开始计算信念，并逐季度重建政府本应给出的建议 $h(\gamma_{t-1}) X_{t-1}$。

```{code-cell} ipython3
def government(data, scheme, λ=0.985, δ=0.999, mU=1, my=2, n_init=8):
    """
    Beliefs and recommendations of a government that fits the classical ('C')
    or Keynesian ('K') Phillips curve by discounted least squares and solves
    the Phelps problem, reconstructed from data on (y, U).
    """
    U, y = data['U'].to_numpy(), data['y'].to_numpy()
    T, k, m = len(U), mU + my + 2, max(mU, my)
    S, b, zz, w = np.zeros((k, k)), np.zeros(k), 0.0, 0.0
    coef = np.full((T, k), np.nan)
    S_path, σ2 = np.full((T, k, k), np.nan), np.full(T, np.nan)
    states, yhat = np.full((T, k - 1), np.nan), np.full(T, np.nan)
    for t in range(m, T):
        s = np.concatenate([U[t - mU:t][::-1], y[t - my:t][::-1], [1.0]])
        states[t] = s
        if t - m > n_init:                       # recommendation from γ_{t-1}
            β = coef[t - 1]
            γ = β if scheme == 'C' else γ_of_β(β)
            yhat[t] = phelps_h(γ, δ, mU, my) @ s
        x = np.r_[y[t], s] if scheme == 'C' else np.r_[U[t], s]
        z = U[t] if scheme == 'C' else y[t]
        S, b = λ * S + np.outer(x, x), λ * b + x * z
        zz, w = λ * zz + z**2, λ * w + 1
        if t - m >= n_init:
            coef[t] = np.linalg.solve(S, b)
            S_path[t] = S
            σ2[t] = (zz - b @ coef[t]) / (w - k)  # discounted residual variance
    lags_y = coef[:, 1 + mU:1 + mU + my].sum(axis=1)
    out = data.copy()
    out['yhat'] = yhat
    out['miss'] = out['y'] - out['yhat']           # v_2t
    out['sum_y'] = lags_y + coef[:, 0] if scheme == 'C' else lags_y
    return dict(frame=out, coef=coef, S=S_path, σ2=σ2, states=states,
                scheme=scheme, λ=λ, δ=δ, mU=mU, my=my)

gov = {s: government(data, s) for s in 'CK'}
```

## 估计结果

函数 `loglik` 在 $V_v$ 被消去的情况下计算式 {eq}`ppe_loglik`。

它使用似然样本之前两个季度的数据来提供 $U_{t-1}$ 和 $U_{t-2}$，并像该书一样，将样本前的冲击设为零。

```{code-cell} ipython3
def residuals(params, U, v2):
    "Solve the actual Phillips curve for v_1 given the misses v_2."
    θ, U_star, ρ1, ρ2 = params
    U0, U1, U2 = U[2:], U[1:-1], U[:-2]
    v2 = v2[2:]
    v2_lag = np.r_[0.0, v2[:-1]]
    rhs = (θ * (v2 - ρ1 * v2_lag) + U0 - (ρ1 + ρ2) * U1 + ρ1 * ρ2 * U2
           - U_star * (1 - ρ1) * (1 - ρ2))
    v1 = lfilter([1.0], [1.0, -ρ2], rhs)           # v1_t = ρ2 v1_{t-1} + rhs_t
    return v1, v2

def loglik(params, U, v2):
    "Gaussian log likelihood with V_v concentrated out."
    v = np.vstack(residuals(params, U, v2))
    T = v.shape[1]
    V = v @ v.T / T
    return -T * np.log(2 * np.pi) - T / 2 * np.linalg.slogdet(V)[1] - T, V

def estimate(g, start='1958'):
    "Maximum likelihood estimates of (θ, U*, ρ1, ρ2) for a government g."
    f = g['frame']
    i0 = f.index.get_loc(pd.Timestamp(start)) - 2
    U, v2 = f['U'].to_numpy()[i0:], f['miss'].to_numpy()[i0:]

    def neg(p):
        return -loglik(p, U, v2)[0] if max(abs(p[2]), abs(p[3])) < 1 else 1e10

    starts = [(0.2, 4.0, ρ1, ρ2) for ρ1 in (0.8, 0.95) for ρ2 in (0.0, 0.5, 0.9)]
    fits = [minimize(neg, x0, method='Nelder-Mead',
                     options=dict(xatol=1e-8, fatol=1e-9, maxiter=20_000))
            for x0 in starts]
    best = min(fits, key=lambda r: r.fun)
    return dict(params=best.x, loglik=-best.fun, U=U, v2=v2,
                V=loglik(best.x, U, v2)[1], T=len(U) - 2)

def std_errors(est, h=1e-4):
    "Standard errors from the numerical Hessian of the concentrated likelihood."
    x, k = est['params'], len(est['params'])
    f = lambda p: loglik(p, est['U'], est['v2'])[0]
    E, H = h * np.eye(k), np.empty((k, k))
    for i in range(k):
        for j in range(k):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                       - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h**2)
    return np.sqrt(np.diag(np.linalg.inv(-H)))

est = {s: estimate(gov[s]) for s in 'CK'}
se = {s: std_errors(est[s]) for s in 'CK'}
```

下面将我们的估计结果与该书表 3 和表 4 并列展示。

```{code-cell} ipython3
book = {'C': [0.1192, 1.6483, 0.9145, 0.5754, -560],
        'K': [0.163, 2.745, 0.932, 0.667, -333]}
names = ['θ', 'U*', 'ρ1', 'ρ2']
rows = []
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    for i, name in enumerate(names):
        p, e = est[s]['params'][i], se[s][i]
        rows.append([label, name, book[s][i], p, e, abs(p - book[s][i]) / e])
    rows.append([label, 'log L', book[s][4], est[s]['loglik'], np.nan, np.nan])
table = pd.DataFrame(rows, columns=['scheme', 'parameter', 'book', 'ours',
                                    'std. error', '|ours - book| / s.e.'])
print(f"likelihood sample: {est['C']['T']} quarters, 1958Q1-1997Q1")
table.set_index(['scheme', 'parameter']).round(3)
```

在两种方案下，该书对 $\theta$ 和 $\rho_1$ 的估计值都与我们的估计值相差不到一个标准误；其 $U^*$ 和 $\rho_2$ 的值与我们的标准误相差大约一个半标准误。

通货膨胀意外的影响效应 $\theta$ 很小，每一个百分点的年化季度通货膨胀率只会使失业率变化十分之一到五分之一个百分点。

菲利普斯曲线扰动具有持续性（$\rho_1$ 高于 .9），而一次意外冲击的影响消退得更快（$\rho_2$ 接近 .7）。

有两个参数确定得较弱：如果似然样本的起点推迟两个季度，即 1958 年衰退之后，$\rho_2$ 的估计值会明显变化，

```{code-cell} ipython3
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    p = estimate(gov[s], start='1958-07')['params']
    print(f"{label:10s} from 1958Q3: θ = {p[0]:.3f}, U* = {p[1]:.2f}, "
          f"ρ1 = {p[2]:.3f}, ρ2 = {p[3]:.3f}")
```

且自然率 $U^*$ 的标准误很大。

凯恩斯方案的似然值要高得多。

我们所得到的两方案间的差距小于该书的差距；由于该常数项的误印以及数据版本不同，$\log L$ 的具体水平不具可比性，但差距的符号是相同的。

残差显示了这一差距的来源，也说明了为何 $U^*$ 难以确定：

```{code-cell} ipython3
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    v1, v2 = residuals(est[s]['params'], est[s]['U'], est[s]['v2'])
    rms = lambda v: np.sqrt(np.mean(v**2))
    print(f"{label:10s}: misses v2: root mean square {rms(v2):.2f}, mean {v2.mean():.2f};  "
          f"v1: root mean square {rms(v1):.2f}, mean {v1.mean():.2f};  "
          f"corr(v1, v2) = {np.corrcoef(v1, v2)[0, 1]:.2f}")
print(f"average unemployment over the sample: {est['C']['U'][2:].mean():.2f}")
```

就均方根而言，古典政府的误差超过了凯恩斯政府误差的两倍。

两组误差的均值都为正，因为在很长一段时间内，建议都低于实际通货膨胀率。

由于 $v_1$ 与 $v_2$ 相关，$v_1$ 的均值也可能为正，而较低的 $U^*$ 加上 $v_1$ 均值为正的组合，几乎与较高的 $U^*$ 拟合得一样好；这大概就是为什么该书和我们都发现 $U^*$ 的值低于样本期内的平均失业率。

## 建议与误差

该模型中通货膨胀的拟合值并非外部计量经济学家所做的预测：它就是政府自身程序根据其已看到的数据所给出的建议通货膨胀率。

下面两幅图重现了该书的图 9.1 和图 9.2a：每幅图都把建议值与实际通货膨胀率对照展示，并展示政府估计的菲利普斯曲线中通货膨胀系数之和。

当该和在古典回归中为 $0$，在凯恩斯回归中为 $1$ 时，归纳假设（{ref}`pa_induction`）成立。

```{code-cell} ipython3
def plot_government(g, ref, title, sum_label, start='1958'):
    f = g['frame'].loc[start:]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(f.index, f['y'], 'C0--', lw=1, label='actual')
    axes[0].plot(f.index, f['yhat'], 'C1', lw=1.8, label='recommended')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('inflation (percent)')
    axes[0].legend(frameon=False)
    axes[1].plot(f.index, f['sum_y'], 'C2', lw=1.5)
    axes[1].axhline(ref, color='k', ls=':', lw=1, label='induction hypothesis')
    axes[1].set_ylabel(sum_label)
    axes[1].legend(frameon=False)
    axes[0].set_title(title)
    plt.tight_layout()
    plt.show()
```

首先来看古典政府。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Classical government: recommended and actual inflation, and the sum of coefficients on current and lagged inflation (book Figure 9.1)"
    name: fig-ppe-classical
---
plot_government(gov['C'], 0, 'Classical identification',
                'sum on current and lagged $y$')
```

古典政府的建议通货膨胀率几乎始终低于零，仅在 1970 年前后短暂升到零以上。

正如该书所说："古典模型甚至无法反映出 1970 年之前通货膨胀的加速过程。"

其对当期和滞后通货膨胀所做的失业率回归，得到的通货膨胀系数之和很小，从 20 世纪 70 年代初开始接近于零。

这样的政府几乎看不到任何可利用的权衡关系，因此费尔普斯问题会要求它将目标定为低通货膨胀。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Keynesian government: recommended and actual inflation, and the sum of coefficients on lagged inflation (book Figure 9.2a)"
    name: fig-ppe-keynesian
---
plot_government(gov['K'], 1, 'Keynesian identification', 'sum on lagged $y$')
```

凯恩斯政府的建议更好地反映了通货膨胀的变化模式，部分地匹配上了 20 世纪 60 年代末的加速过程。

但从 20 世纪 70 年代初开始，这些建议大大低于实际通货膨胀率，并在之后近二十年的大部分时间里都低于实际值。

下方的面板说明了其中的原因。

滞后通货膨胀的权重之和从 20 世纪 60 年代远低于 1 的水平，攀升到 70 年代中期的大约 1，之后一直保持在这一水平附近。

到那时，凯恩斯政府已经学会了索洛—托宾版本的自然率假说，费尔普斯问题也因此要求它降低通货膨胀率。

### 归纳假设逐渐显现

该书的图 9.2b 展示了数据如何逐渐消除政府的疑虑。

我们用 $\hat\sigma_t^2 \left(\sum_{s\le t}\lambda^{t-s} X_{K,s}X_{K,s}^\top\right)^{-1}$ 近似表示政府对 $\beta_t$ 的不确定性，其中 $\hat\sigma_t^2$ 是折扣残差方差。

这正是实施常增益最小二乘法的卡尔曼滤波器会赋予其估计值的协方差矩阵。

该图绘制了 1960、1965、……、1985 年第一季度，滞后通货膨胀权重之和与常数项的 95% 和 99% 置信椭圆。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Keynesian government: 95% and 99% confidence ellipses for the sum of weights on lagged inflation and the constant (book Figure 9.2b)"
    name: fig-ppe-ellipses
---
def ellipse(center, cov, level, n=200):
    "Points on a confidence ellipse for a bivariate normal estimate."
    w, Q = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(a), np.sin(a)])
    return center[:, None] + Q @ (np.sqrt(chi2.ppf(level, 2) * w)[:, None] * circle)

g = gov['K']
k = g['coef'].shape[1]
D = np.zeros((2, k))
D[0, 1 + g['mU']:1 + g['mU'] + g['my']] = 1.0     # sum of weights on lagged y
D[1, -1] = 1.0                                     # constant

fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharex=True, sharey=True)
for ax, year in zip(axes.ravel(), range(1960, 1990, 5)):
    t = g['frame'].index.get_loc(pd.Timestamp(f'{year}-01-01'))
    center = D @ g['coef'][t]
    cov = g['σ2'][t] * D @ np.linalg.inv(g['S'][t]) @ D.T
    for level, ls in [(0.95, '-'), (0.99, '--')]:
        e = ellipse(center, cov, level)
        ax.plot(e[0], e[1], 'C0', ls=ls, lw=1)
    ax.plot(*center, 'C0o', ms=3)
    ax.axvline(1, color='k', ls=':', lw=1)
    ax.set_title(f'{year}Q1')
for ax in axes[1]:
    ax.set_xlabel('sum on lagged $y$')
for ax in axes[:, 0]:
    ax.set_ylabel('constant')
plt.tight_layout()
plt.show()
```

这一模式与该书的图形以及其第 8 章模拟出的置信椭圆类似。

20 世纪 60 年代，椭圆远离归纳假设（虚线所示），但其尺寸仍大到留有疑虑的空间。

进入 70 年代后，椭圆中心移动到归纳假设线上并一直停留在那里，椭圆也变得比 60 年代初更小，不过没有该书中的那么小。

用该书的话说，点估计发生了移动，"使得椭球体的中心发生偏移，疑虑逐渐消散。"

(ppe_misses)=
## 误差支持平反说

从逐期拟合效果来看，这些适应性模型令人失望：即使是凯恩斯模型，20 世纪 70 年代也存在一长串对通货膨胀的低估。

萨金特对此有不同的解读：

> 但这样的误差未必不能证明计量经济学政策评估说的合理性。恰恰相反，这些估计模型的误差模式支持这一平反说。只要记住，适应性模型的拟合值就是随着时间推移政府所给出的建议，就能从这些计量经济学上的不足中得出平反的结论。

这些误差表明，建议值系统性地*低于*实际发生的通货膨胀率：政府的程序一直在告诉它应该降低通货膨胀，而实际政策却没有这样做。

一步建议 $h(\gamma_{t-1})X_{t-1}$ 混合了两样东西：政府希望通货膨胀走向何方，以及它计划以多快的速度从所继承的通货膨胀水平到达那里。

为了将二者分开，我们计算政府计划的稳态，即如果当前信念正确，费尔普斯规则将引导经济趋向的长期通货膨胀率。

```{code-cell} ipython3
def planned_long_run(γ, δ, mU, my):
    "Steady-state inflation of the government's plan under its own beliefs."
    h = phelps_h(γ, δ, mU, my)
    A, B = lq_matrices(γ, mU, my)
    M = A + B @ h[None, :]                         # believed closed loop
    n = M.shape[0] - 1                             # last state is the constant
    s_bar = np.linalg.solve(np.eye(n) - M[:n, :n], M[:n, -1])
    return h @ np.r_[s_bar, 1.0]

def long_run_path(g):
    path = np.full(len(g['frame']), np.nan)
    for t in range(1, len(path)):
        β = g['coef'][t - 1]
        if not np.isnan(β).any():
            γ = β if g['scheme'] == 'C' else γ_of_β(β)
            path[t] = planned_long_run(γ, g['δ'], g['mU'], g['my'])
    return pd.Series(path, index=g['frame'].index)

for s in 'CK':
    gov[s]['frame']['long_run'] = long_run_path(gov[s])
```

下图将凯恩斯政府的计划与其一步建议、古典政府的建议以及实际通货膨胀率并列展示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Actual inflation, the Keynesian government's recommendation, and the long-run inflation rate its plan aimed for"
    name: fig-ppe-plans
---
fK, fC = gov['K']['frame'].loc['1958':], gov['C']['frame'].loc['1958':]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(fK.index, fK['y'], 'C0--', lw=1, label='actual inflation')
ax.plot(fK.index, fK['yhat'], 'C1', lw=1.5, label='Keynesian recommendation')
ax.plot(fK.index, fK['long_run'], 'C3', lw=2, label='Keynesian planned long-run inflation')
ax.plot(fC.index, fC['yhat'], 'C2', lw=1, alpha=0.8, label='classical recommendation')
ax.axvline(pd.Timestamp('1979-08-01'), color='k', ls=':', lw=1)
ax.text(pd.Timestamp('1980-06-01'), -4, 'Volcker', fontsize=9)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('percent')
ax.legend(frameon=False, fontsize=9, loc='upper left')
plt.show()
```

五个时代的平均值概括了该图的内容。

```{code-cell} ipython3
eras = [('1958', '1965'), ('1966', '1972'), ('1973', '1979'),
        ('1980', '1984'), ('1985', '1997')]
summary = pd.DataFrame(
    {f'{a}-{b}': [fK['y'].loc[a:b].mean(),
                  fC['yhat'].loc[a:b].mean(),
                  fK['yhat'].loc[a:b].mean(),
                  fK['long_run'].loc[a:b].median()]
     for a, b in eras},
    index=['actual inflation', 'classical recommendation',
           'Keynesian recommendation', 'Keynesian long-run plan (median)'])
summary.round(2)
```

从 1973 年到 1990 年，建议值大多低于实际通货膨胀率：

```{code-cell} ipython3
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    m = gov[s]['frame']['miss'].loc['1973':'1990']
    print(f"{label:10s}: mean miss 1973-1990 = {m.mean():5.2f}, "
          f"share of quarters with recommendation below actual = {(m > 0).mean():.0%}")
```

这一计算证实了该书的解读：

> 计量经济学估计告诉我们，那些未经改造的凯恩斯菲利普斯曲线拟合者，本会察觉到经验菲利普斯曲线出现的不利变化，并通过归纳假设结合费尔普斯问题，本会建议降低通货膨胀率。这些量化的政策评估者本不会认同 20 世纪 70 年代末盛行的、论证松散的观点，即预期的长滞后使得紧缩代价过高。我们的结果表明，在古典识别方案下，建议本会更加及时。

整个 20 世纪 60 年代，凯恩斯政府的目标是将长期通货膨胀率定在百分之二到三。

一旦其估计的滞后通货膨胀权重之和接近 1，其目标便降到约百分之一或更低，并在整个 80 年代初一直维持在这一水平，而实际通货膨胀率却远高于此。

70 年代末，其一步建议大约为百分之四，要求将通货膨胀降到远低于所继承的水平。

沃尔克紧缩之后，权重之和略有下降，计划的长期通货膨胀率回到百分之二到三，接近实际通货膨胀率。

古典政府的建议即使在 60 年代末的加速期也接近于零，之后更是低于零：这正是其建议"本会更加及时"这一说法的含义所在。

该书第 1 章得出结论："这些计量经济学估计提出了一个谜团：为什么稳定措施被推迟了。"

### 这些误差有多稳健？

由于该书没有报告其滞后阶数，也没有说明如何转换为季度数据，下表针对滞后阶数、季度取样方法以及数据频率进行了变化检验。

月度数据列使用 $\lambda^{1/3}$ 和 $\delta^{1/3}$，以保持相同的记忆长度和相同的日历时间贴现率。

```{code-cell} ipython3
specs = [('end of quarter', 'last', 1, 2), ('end of quarter', 'last', 2, 2),
         ('quarterly average', 'avg', 1, 2), ('quarterly average', 'avg', 2, 2),
         ('monthly', 'monthly', 1, 2), ('monthly', 'monthly', 2, 2)]
rows = []
for label, how, mU, my in specs:
    d = make_data(how)
    λ, δ = (0.985**(1/3), 0.999**(1/3)) if how == 'monthly' else (0.985, 0.999)
    g = {s: government(d, s, λ=λ, δ=δ, mU=mU, my=my) for s in 'CK'}
    e = {s: estimate(g[s]) for s in 'CK'}
    fk = g['K']['frame']
    fk['long_run'] = long_run_path(g['K'])
    β_U = pd.Series(g['K']['coef'][:, 0], index=fk.index)
    m = fk['miss'].loc['1973':'1990']
    rows.append([label, f'({mU}, {my})',
                 e['K']['loglik'] - e['C']['loglik'],
                 e['K']['params'][0],
                 g['C']['frame']['yhat'].loc['1973':'1979'].mean(),
                 β_U.loc['1973':'1979'].mean(),
                 fk['yhat'].loc['1973':'1979'].mean(),
                 (m > 0).mean(),
                 fk['sum_y'].loc['1975':'1984'].mean(),
                 fk['long_run'].loc['1973':'1984'].median()])
robust = pd.DataFrame(rows, columns=[
    'data', '(m_U, m_y)', 'log L: K minus C', 'K estimate of θ',
    'C recommendation 1973-79', 'K coefficient on U_t, 1973-79',
    'K recommendation 1973-79', 'K share of misses > 0, 1973-90',
    'K sum on lagged y, 1975-84', 'K long-run plan 1973-84'])
robust.set_index(['data', '(m_U, m_y)']).round(2).T
```

有些发现在每一列中都成立：凯恩斯方案拟合得更好，古典政府在 70 年代建议的通货膨胀率都低于零。

无论滞后阶数或取样方法如何，只要使用季度数据，凯恩斯政府的滞后通货膨胀权重之和在 1975 年后都接近 1，其长期计划在 1973—1984 年间要求的通货膨胀率都在大约百分之一或更低。

*一步*凯恩斯建议则较为脆弱，$\theta$ 的估计值也是如此。

原因在于凯恩斯回归中当期失业率的系数 $\beta_1$，政府对其求逆以得到短期权衡关系 $1/\beta_1$。

使用一个失业率滞后期时，1973—1979 年间该系数明显为负；使用两个滞后期时，在以季末取样的情况下该系数大约只有一半大小，而在以季度平均取样的情况下该系数大约为零。

对曲线求逆就等于除以一个接近零的数，于是所认为的失业率动态变得不稳定，政府对所继承通货膨胀的短期反应也随之剧烈波动；这就是这些列中一步误差变小的原因。

月度数据出于更根本的原因而失效。

两个月度通货膨胀滞后期只跨越两个月，其系数之和远小于 1，因此月度凯恩斯政府从未激活归纳假设，其长期计划反而随通货膨胀上升而非下降。

足以捕捉归纳假设的分布滞后需要跨越若干季度，而不是若干个月。

### 这是实证结论，而非规范性结论

萨金特并未将这些估计结果转化为政策背书。

{doc}`phillips_learning` 中的模拟表明，均值动态最终会将适应性政府拉回自我确认均衡，因此一段时期内的良好建议并不能保证长久的良好政策。

该书第 10 章总结道：

> 最后，尽管我们的模拟和计量经济学证据都支持计量经济学政策评估的平反说，但我们希望这是一个错误的故事。相反，我们希望决策者以某种方式学会了自然率假说的正确理性预期版本，并致力于实现低通货膨胀。

## 后续发展

第 9 章将增益固定不变。

{ref}`olg_ex7` 提出了钟（Chung）的问题，即在有限理性系列模型模拟出的数据上估计增益本身，其中增益被称为 $\gamma$，信念被称为 $\alpha$；下文的 {ref}`ppe_ex1` 在本讲座的数据上探讨了同样的问题。

{ref}`pbr_compliment` 解释了为什么不会消失的增益能够使这类学习参数可估计。

{cite:t}`SargentWilliamsZha2006` 是本章方法的现代对应版本。

他们的政府对漂移的菲利普斯曲线系数运行卡尔曼滤波器（{doc}`kalman`），即西姆斯—钟（Sims–Chung）设定，并且与第 9 章一样，通货膨胀的系统性部分根据政府当前的估计求解费尔普斯问题。

与第 9 章不同的是，他们使用贝叶斯马尔可夫链蒙特卡洛算法，与真实模型的参数一起联合估计了决定政府增益的漂移协方差。

{doc}`phillips_priors` 讨论了他们的估计结果，以及 {cite:t}`CarboniEllison2009` 的改进方案，后者利用美联储的绿皮书预测来约束政府的信念。

{doc}`phillips_lost_conquest` 则针对 21 世纪 20 年代的美联储提出了钟的问题。

在本讲座中，政府的信念在似然函数中没有留下任何自由参数。

下一讲 {doc}`phillips_drifts_volatilities` 采取了相反的做法：它拟合一个简化型向量自回归模型，其系数随时间漂移，其冲击具有随机波动率。

它要探讨的是：平反说将其置于战后通货膨胀核心地位的漂移信念，是否真的在数据中留下了该说法所预测的痕迹。

## 练习

```{exercise-start}
:label: ppe_ex1
```

该书将 $\lambda = .985$ 固定不变。

计算增益的剖面似然：对 $.95$ 到 $1$ 之间网格上的 $\lambda$ 值，重新构建政府的信念和建议，重新估计 $(\theta, U^*, \rho_1, \rho_2)$，并针对两种识别方案绘制最大化后的 $\log L$ 对 $\lambda$ 的图形。

注意 $\lambda = 1$ 即为最小二乘法，其增益按 $1/t$ 的速率递减。

数据更偏好哪些 $\lambda$ 值？这一答案对该书"为什么稳定措施被推迟"这一谜团有何启示？

```{exercise-end}
```

```{solution-start} ppe_ex1
:class: dropdown
```

```{code-cell} ipython3
λ_grid = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
profile = {s: [] for s in 'CK'}
mean_K = []
for λ in λ_grid:
    for s in 'CK':
        g = government(data, s, λ=λ)
        profile[s].append(estimate(g)['loglik'])
        if s == 'K':
            mean_K.append(g['frame']['yhat'].loc['1973':'1979'].mean())

fig, ax = plt.subplots(figsize=(8, 4.5))
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    ax.plot(λ_grid, np.array(profile[s]) - max(profile[s]), 'o-', label=label)
ax.axvline(0.985, color='k', ls=':', lw=1)
ax.set_xlabel(r'forgetting factor $\lambda$')
ax.set_ylabel(r'$\log L - \max_\lambda \log L$')
ax.legend(frameon=False)
plt.show()

print(" λ      log L (C)  log L (K)  K recommendation 1973-79")
for λ, lc, lk, m in zip(λ_grid, profile['C'], profile['K'], mean_K):
    print(f"{λ:5.3f}  {lc:9.1f}  {lk:9.1f}  {m:8.2f}")
```

古典似然在 $\lambda$ 高于 .97 时几乎是平坦的，因为无论记忆长度如何，古典政府都会建议低通货膨胀。

随着 $\lambda$ 趋近于 1，凯恩斯似然稳步上升：数据更偏好对过去打折最少的政府。

打印出的平均值说明了原因。

记忆更长的政府会给予 20 世纪 50 年代和 60 年代更大的权重，而那时其回归显示存在长期权衡关系，因此它会更晚地激活归纳假设，并在 70 年代建议更高的通货膨胀率，更接近实际发生的通货膨胀率。

由于似然函数只能通过使建议更接近实际通货膨胀率来奖励某个增益值，它更偏好学习得最慢的那个。

这是该书之谜的另一种表述方式：实际政策的表现仿佛比该书中常增益政府更晚才学会自然率假说。

在 {ref}`olg_ex7` 的模拟经济中，政府确实遵循了自己的建议，似然函数能恢复出真实的增益；而在这里，对缓慢学习的偏好在一定程度上反映了实际政策相对于该程序的滞后程度。

```{solution-end}
```

```{exercise-start}
:label: ppe_ex2
```

{cite:t}`Sims1988` 为其政府设定了一条*静态*菲利普斯曲线，$U_t = \gamma_0 y_t + \gamma_c + \varepsilon_t$，不含滞后项。

在两种识别方案下估计该静态模型（$m_U = m_y = 0$），并与本讲座中的分布滞后模型进行比较。

对于这四个模型中的每一个，报告最大化后的 $\log L$、1973—1979 年间的平均建议值，以及建议通货膨胀率与实际通货膨胀率之间的相关性。

静态古典模型是否失败？滞后结构和拟合方向，哪个更重要？

```{exercise-end}
```

```{solution-start} ppe_ex2
:class: dropdown
```

```{code-cell} ipython3
rows = []
for mU, my, lag_label in [(0, 0, 'static'), (1, 2, 'distributed lag')]:
    for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
        g = government(data, s, mU=mU, my=my)
        f = g['frame'].loc['1958':]
        rows.append([lag_label, label, estimate(g)['loglik'],
                     f['yhat'].loc['1973':'1979'].mean(),
                     f[['y', 'yhat']].corr().iloc[0, 1]])
pd.DataFrame(rows, columns=['lags', 'scheme', 'log L', 'recommendation 1973-79',
                            'corr(recommended, actual)']
             ).set_index(['lags', 'scheme']).round(2)
```

静态古典模型确实失败了：它在 70 年代建议的通货膨胀率接近零，而且其建议值与实际通货膨胀率呈负相关。

但加入滞后项并不能挽救古典方案：含分布滞后的古典政府建议的通货膨胀率甚至更低，其似然值也低于静态模型。

拟合方向比滞后结构更重要，两个凯恩斯模型的拟合效果都远好于两个古典模型。

在将失业率对通货膨胀率回归时，季度通货膨胀率中的噪声会使通货膨胀系数被压缩，因此古典政府只能看到微弱的权衡关系；而将通货膨胀率对失业率回归并求逆，则会得到一个陡峭的权衡关系。

对于凯恩斯方案，滞后结构同样重要：分布滞后模型的似然值更高，其建议值与实际通货膨胀率的吻合程度也高得多。

只有分布滞后结构才能让政府发现归纳假设；静态凯恩斯政府的模型中不含滞后通货膨胀项，因此无法以索洛—托宾的形式学会自然率假说。

该书从 {cite:t}`KingWatson1994` 处借用了这两种识别方案及其名称，并指出，正是凯恩斯方向拟合效果更好这一点，促使美国菲利普斯曲线文献采纳了这一方向。

```{solution-end}
```

```{exercise-start}
:label: ppe_ex3
```

取凯恩斯政府在 1979 年第二季度末、保罗·沃尔克就任美联储主席前夕的信念。

计算它为未来二十个季度所*计划*的通货膨胀和失业路径：即由其费尔普斯规则和自身估计的菲利普斯曲线所隐含的路径，假设未来所有冲击都为零，从 1979 年第三季度的状态开始。

对古典政府做同样的计算，并将两者的计划与实际发生的情况一并绘图比较。

凯恩斯计划与实际的紧缩过程相比如何？每个政府各自预期为此要付出多大的失业率上升代价？

```{exercise-end}
```

```{solution-start} ppe_ex3
:class: dropdown
```

```{code-cell} ipython3
def planned_paths(g, date, H=20):
    "Inflation and unemployment the government planned from `date` on."
    t = g['frame'].index.get_loc(pd.Timestamp(date))
    β = g['coef'][t - 1]
    γ = β if g['scheme'] == 'C' else γ_of_β(β)
    h = phelps_h(γ, g['δ'], g['mU'], g['my'])
    A, B = lq_matrices(γ, g['mU'], g['my'])
    s = g['states'][t].copy()
    y_plan, U_plan = [], []
    for _ in range(H):
        y = h @ s
        y_plan.append(y)
        U_plan.append(γ[0] * y + γ[1:] @ s)
        s = A @ s + B[:, 0] * y
    index = g['frame'].index[t:t + H]
    return pd.DataFrame({'y': y_plan, 'U': U_plan}, index=index[:H])

date = '1979-07-01'
plans = {s: planned_paths(gov[s], date) for s in 'CK'}
actual = data.loc[date:].iloc[:20]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, var, ylabel in [(axes[0], 'y', 'inflation'), (axes[1], 'U', 'unemployment')]:
    ax.plot(actual.index, actual[var], 'k--', lw=1.2, label='actual')
    ax.plot(plans['K'].index, plans['K'][var], 'C1', lw=2, label='Keynesian plan')
    ax.plot(plans['C'].index, plans['C'][var], 'C2', lw=2, label='classical plan')
    ax.set_ylabel(f'{ylabel} (percent)')
    ax.tick_params(axis='x', rotation=30)
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

print(f"1979Q2: inflation {data.loc['1979-04-01', 'y']:.1f}, "
      f"unemployment {data.loc['1979-04-01', 'U']:.1f}")
paths = {'Keynesian plan': plans['K'], 'classical plan': plans['C'], 'actual': actual}
print(f"{'':16s} {'y, 1st qtr':>10s} {'y, 4th qtr':>10s} {'y, 20th qtr':>11s} {'peak U':>7s}")
for label, p in paths.items():
    print(f"{label:16s} {p['y'].iloc[0]:10.1f} {p['y'].iloc[3]:10.1f} "
          f"{p['y'].iloc[-1]:11.1f} {p['U'].max():7.1f}")
```

在沃尔克任命前夕，凯恩斯政府计划立即大幅削减通货膨胀，从两位数降到几个百分点，然后缓慢地将其进一步降至零。

它预期这次削减代价不菲：计划中的失业率跳升到远高于 1979 年的水平，然后缓慢下降。

这正是该书第 5 章（{doc}`phillips_adaptive`）中费尔普斯问题的模式，即政府"制造了一场重大衰退，并立即将通货膨胀率降低到其最终极限值一半以上的水平。"

古典政府则计划一次温和的紧缩，并预期其代价很小，因为其估计曲线几乎没有显示出通货膨胀对失业率的短期影响。

实际政策的推进比两个计划都要慢：通货膨胀率在高位维持了一年多，直到 1982—1983 年才达到 80 年代的水平，而白人男性的失业率直到 1982 年底才达到峰值。

两个政府的计量经济学估计都表明应立即紧缩，凯恩斯政府甚至在预期紧缩代价高昂的情况下仍然建议这样做。

两者都不会接受 20 世纪 70 年代末盛行的那种观点，即预期的长滞后使得紧缩代价过高以至于不值得尝试。

```{solution-end}
```