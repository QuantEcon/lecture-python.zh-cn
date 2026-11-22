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
  title: 计量经济政策评估
  headings:
    Overview: 概览
    The model: 模型
    The model::The actual Phillips curve: 实际菲利普斯曲线
    The model::The government: 政府
    The likelihood function: 似然函数
    Data: 数据
    Beliefs and recommendations: 信念与建议
    Estimates: 估计结果
    Recommendations and misses: 建议与误差
    Recommendations and misses::The induction hypothesis emerges: 归纳假设的形成
    Misses favor vindication: 误差支持平反
    Misses favor vindication::How robust are the misses?: 这些误差有多稳健？
    Misses favor vindication::A positive, not a normative, conclusion: 这是一个实证结论，而非规范性结论
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

# 计量经济政策评估

```{index} single: Phillips Curve; Econometric Policy Evaluation
```

```{contents} Contents
:depth: 2
```

## 概览

本讲座遵循 {cite:t}`Sargent1999` 第 9 章《计量经济政策评估》的内容。

前面的讲座构建了该书第 8 章中的适应性政府，并通过模拟对其进行了研究（{doc}`phillips_learning`、{doc}`phillips_escaping_nash`、{doc}`phillips_priors`）。

第 9 章将同样的模型应用于美国数据，遵循 {cite:t}`Chung1990` 的方法，在经典和凯恩斯两种识别方案下通过最大似然法对其进行估计。

这是该书唯一一次将 {doc}`phillips_two_stories` 的平反故事与数据进行结构性对照。

问题是反事实的：如果 20 世纪 60 年代和 70 年代的政策顾问拟合了一条分布滞后菲利普斯曲线，对旧数据进行贴现，并将估计结果输入 {doc}`phillips_adaptive` 中的费尔普斯问题，他会给出怎样的建议？

估计的模型直接给出了答案，因为在该模型中，通货膨胀的拟合值*就是*政府的建议。

一旦增益和贴现因子固定下来，似然函数中就没有描述信念的自由参数了：我们像政府本身那样，从数据中计算出政府的信念和建议。

我们重现了该书的主要发现：

* 两个模型对通货膨胀的拟合都很差，而经典模型的拟合效果要差得多；
* 误差是系统性的，整个 20 世纪 70 年代建议的通货膨胀率都低于实际通货膨胀率，这正是"估计模型的误差模式支持平反"这一说法的含义；
* 凯恩斯政府估计的菲利普斯曲线到 20 世纪 70 年代中期已满足归纳假设，其计划在沃尔克上任前多年就已呼吁实行低通货膨胀；
* 经典政府自始至终都建议实行低通货膨胀，这正是其建议"本会更加及时"的原因。

该书没有报告其滞后阶数，也没有说明如何将数据转换为季度频率，我们在这些选择产生影响的地方会加以说明。

让我们先导入一些包。

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

该模型将 {doc}`phillips_self_confirming` 中的实际菲利普斯曲线与 {doc}`phillips_learning` 中的适应性政府结合在一起；所用符号与 {ref}`phillips_notation` 中相同。

### 实际菲利普斯曲线

设 $U_t$ 为失业率，$y_t$ 为通货膨胀率，$x_t$ 为公众对通货膨胀的预期，该经济体服从一条自然率菲利普斯曲线，其中意外通货膨胀 $v_{2t} = y_t - x_t$ 具有几何分布滞后结构，且存在序列相关扰动：

```{math}
:label: ppe_actual

U_t = U^* - \frac{\theta}{1 - \rho_2 L}\, v_{2t} + \frac{1}{1 - \rho_1 L}\, v_{1t},
```

其中 $L$ 为滞后算子，$|\rho_1| < 1$ 且 $|\rho_2| < 1$。

两边同乘 $(1-\rho_1 L)(1-\rho_2 L)$，得到该书中编号为（48）的形式：

```{math}
:label: ppe_actual48

U_t = U^*(1-\rho_1)(1-\rho_2) + (\rho_1+\rho_2) U_{t-1} - \rho_1\rho_2 U_{t-2}
      + (1-\rho_2 L) v_{1t} - \theta (1-\rho_1 L) v_{2t} .
```

冲击向量 $v_t = (v_{1t}, v_{2t})^\top$ 是具有不受限协方差矩阵 $V_v$ 的高斯白噪声。

该书第 7 章和第 8 章设 $\rho_1 = \rho_2 = 0$，并令 $V_v$ 为对角矩阵以阐明理论要点；第 9 章则将两者都放开。

### 政府

设 $X_{t-1} = \begin{bmatrix} U_{t-1} & \cdots & U_{t-m_U} & y_{t-1} & \cdots & y_{t-m_y} & 1 \end{bmatrix}^\top$ 为收集了 $t-1$ 期及更早时期信息的向量。

与 {doc}`phillips_adaptive` 中一样，政府拟合以下两条简约式菲利普斯曲线中的一条，二者仅在拟合方向上有所不同：

```{math}
:label: ppe_fits

\text{经典：}\quad U_t = \gamma^\top X_{C,t} + \varepsilon_{C,t},
\qquad
\text{凯恩斯：}\quad y_t = \beta^\top X_{K,t} + \varepsilon_{K,t},
```

其中 $X_{C,t} = \begin{bmatrix} y_t & X_{t-1}^\top \end{bmatrix}^\top$，$X_{K,t} = \begin{bmatrix} U_t & X_{t-1}^\top \end{bmatrix}^\top$。

它每期通过常增益递归最小二乘法重新估计曲线，对于经典方案而言：

```{math}
:label: ppe_rls

\begin{aligned}
\gamma_t &= \gamma_{t-1} + g\, R_{XC,t}^{-1} X_{C,t}\left(U_t - \gamma_{t-1}^\top X_{C,t}\right), \\
R_{XC,t} &= R_{XC,t-1} + g \left(X_{C,t} X_{C,t}^\top - R_{XC,t-1}\right),
\end{aligned}
```

增益 $g = 1 - \lambda$ 为常数；凯恩斯方案以相同方式更新 $\beta_t$ 和 $R_{XK,t}$。

凯恩斯政府通过反转菲利普斯曲线，将其估计值转换为经典形式：若 $\beta_1$ 是 $U_t$ 的系数，$\beta_{-1}$ 收集其余系数，则 $\gamma(\beta) = \begin{bmatrix} \beta_1^{-1} & -\beta_{-1}^\top/\beta_1 \end{bmatrix}^\top$。

每一期，政府都表现得好像其当前的估计将永远支配菲利普斯曲线一样，并通过求解费尔普斯问题来设定通货膨胀率，即在其估计曲线的约束下最小化 $\mathbb{E}\sum_t \delta^t (U_t^2 + y_t^2)$，这产生了以下规则：

```{math}
:label: ppe_rule

y_t = h(\gamma_{t-1}) X_{t-1} + v_{2t},
```

其中 $\gamma_{t-1}$ 是经典估计值或 $\gamma(\beta_{t-1})$，$v_{2t}$ 是控制误差。

按照 {cite:t}`Sims1988` 和 {cite:t}`Chung1990` 的方法，公众知道该规则，因此 $x_t = h(\gamma_{t-1}) X_{t-1}$，而控制误差 $v_{2t}$ 也即 {eq}`ppe_actual` 中的意外通货膨胀。

(ppe_estimation)=
## 似然函数

要模拟该模型，我们需要抽取冲击 $\{v_t\}$，并计算出 $(y_t, U_t)$ 及政府信念的历史路径。

要估计该模型，我们则反过来：从初始信念出发，利用数据 $\{y_t, U_t\}$ 反解出能够产生这些数据的冲击，

```{math}
:label: ppe_inverse

\begin{aligned}
v_{2t} &= y_t - h(\gamma_{t-1}) X_{t-1}, \\
v_{1t} - \theta v_{2t} &= U_t - (\rho_1+\rho_2) U_{t-1} + \rho_1\rho_2 U_{t-2}
       + \rho_2 v_{1,t-1} - \rho_1 \theta v_{2,t-1} - U^*(1-\rho_1)(1-\rho_2),
\end{aligned}
```

同时按 {eq}`ppe_rls` 更新 $\gamma_t$（或 $\beta_t$）；这些即为该书中编号为（105）的方程。

给定过去的信息，从 $v_t$ 到 $(U_t, y_t)$ 的映射是单位对角的三角形映射，因此其雅可比行列式为 1，长度为 $T$ 的样本的高斯对数似然为：

```{math}
:label: ppe_loglik

\log L = -T \log 2\pi - \frac{T}{2}\log\det V_v - \frac{1}{2}\sum_{t=1}^T v_t^\top V_v^{-1} v_t .
```

正如该书所建议的，我们将 $V_v$ 消去：其最大化值为 $\hat V_v = T^{-1}\sum_t v_t v_t^\top$，于是 $\log L = -T\log 2\pi - \frac{T}{2}\log\det \hat V_v - T$。

```{note}
该书第 9 章附录将多个密度之和的对数写成了多个密度对数之和的形式，并给出常数为 $-\frac{T}{2}\log 2\pi$，这对标量观测值而言是正确的；但二元密度应为 $-T\log 2\pi$。

两处笔误都不影响估计结果，但第二处会改变 $\log L$ 的水平值。
```

该书对 $(\theta, U^*, \rho_1, \rho_2, V_v)$ 进行估计，并将贴现因子固定为 $\delta = .999$，遗忘因子固定为 $\lambda = .985$，因此增益为 $g = .015$。

请注意，列表中*没有*任何描述信念的内容。

在 $\lambda$、$\delta$ 和初始信念固定的情况下，政府的信念 $\gamma_t$、其建议 $h(\gamma_{t-1})X_{t-1}$，以及由此产生的误差 $v_{2t}$，无论待估参数取何值，都完全由数据本身计算得出。

这带来两个后果。

1. 由于 $\det \hat V_v$ 等于 $v_{2t}$ 均方值与 $v_{1t}$ 中无法被 $v_{2t}$ 解释的部分的均方值之积，参数 $(\theta, U^*, \rho_1, \rho_2)$ 是以政府的误差为给定条件，从失业方程中估计出来的。
1. 两种识别方案之间的比较，主要是对其误差大小的比较，即建议与实际发生情况的偏离程度。

这是理性预期计量经济学中常见的一种跨方程约束：政府的规则是由其计量经济学和费尔普斯问题共同确定的，而不是自由的。

## 数据

第 9 章采用的是第 1 章的度量指标：消费者价格指数（所有项目）以及 20 岁及以上白人男性的失业率，即 {doc}`phillips_two_stories` 中绘制的序列。

尽管第 1 章绘制的是月度数据，第 9 章的估计结果似乎是季度性的：

* 样本期是以季度标注的（印刷为"1995I–1997I"，显然是 1955I–1997I 的笔误，因为图表是从 1955 年开始的）；
* 图 9.2b 提到"1960 年第一季度"；
* 季度数据能够较好地重现该书的估计值和图表，而月度数据则不能，原因我们将在{ref}`下文 <ppe_misses>`中解释。

因此我们通过取每季度最后一个月的数据，将月度序列转换为季度序列，并将通货膨胀率衡量为 CPI 的年化百分比变化。

因为我们是每季度抽样一个月份，所以使用经季节调整的指数，而非 {doc}`phillips_two_stories` 中绘制的未经调整的指数。

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

失业数据序列始于 1954 年，政府的回归需要几年的数据之后，其估计结果才有意义。

因此，我们将 1954–1957 年作为政府信念的训练样本，并从 1958 年第一季度到 1997 年第一季度对似然进行评估。

该书没有报告其滞后阶数 $m_U$ 和 $m_y$；其第 8 章的模拟使用了两期滞后。

我们使用一期失业率滞后和两期通货膨胀滞后，在我们尝试过的滞后阶数中，这一选择使我们的估计结果最接近该书的结果；下文的稳健性表（{ref}`下文 <ppe_misses>`）展示了使用两期滞后时会发生怎样的变化。

## 信念与建议

费尔普斯问题是 {doc}`phillips_learning` 中的贴现 LQ 问题，推广到任意滞后阶数。

将所信奉的菲利普斯曲线写为 $U_t = \gamma_0 y_t + c^\top s_t$，状态为 $s_t = X_{t-1}$，于是单期损失为 $s_t^\top c c^\top s_t + (\gamma_0^2 + 1) y_t^2 + 2 \gamma_0 y_t c^\top s_t$。

我们使用与 {doc}`phillips_learning` 相同的求解器：在贴现问题中，只有 $A$ 和 $B$ 乘以 $\sqrt{\delta}$，交叉项 $N$ 以未缩放形式传入黎卡提求解器。

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

在弥散的初始条件下，即 $R_{XC,-1} = 0$，递归式 {eq}`ppe_rls` 给出的是*贴现最小二乘法*：$\gamma_t$ 最小化 $\sum_{s \le t} \lambda^{t-s} (U_s - \gamma^\top X_{C,s})^2$，因此一个 $k$ 季度之前的观测值获得的权重为 $\lambda^k$。

下面的函数以这种方式从数据的第一季度开始计算信念，并逐季度重建政府本会做出的建议 $h(\gamma_{t-1}) X_{t-1}$。

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

函数 `loglik` 在消去 $V_v$ 的情况下计算 {eq}`ppe_loglik`。

它使用似然样本之前两个季度的数据来提供 $U_{t-1}$ 和 $U_{t-2}$，并与该书一样，将样本前的冲击设为零。

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

下面是我们的估计结果与该书表 3 和表 4 的对比。

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

在两种方案下，该书对 $\theta$ 和 $\rho_1$ 的估计值都落在我们估计值一个标准误差的范围之内；其 $U^*$ 和 $\rho_2$ 的值与我们的估计值相差约一个半标准误差。

通货膨胀意外冲击对 $\theta$ 的影响较小，每一个百分点的年化季度通货膨胀率对应的失业率变化在十分之一到五分之一个百分点之间。

菲利普斯曲线扰动具有持续性（$\rho_1$ 高于 .9），而冲击的影响衰减得更快（$\rho_2$ 接近 .7）。

有两个参数的确定性较弱：如果似然样本晚两个季度开始（即在 1958 年衰退之后），$\rho_2$ 的估计值会出现明显变化，

```{code-cell} ipython3
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    p = estimate(gov[s], start='1958-07')['params']
    print(f"{label:10s} from 1958Q3: θ = {p[0]:.3f}, U* = {p[1]:.2f}, "
          f"ρ1 = {p[2]:.3f}, ρ2 = {p[3]:.3f}")
```

而自然率 $U^*$ 的标准误差较大。

凯恩斯方案的似然值要高得多。

将两种方案最大化对数似然之间的差距除以样本量，即估计出数据分布与两种方案的 m-投影之间 KL 散度率的差异（{doc}`approximation_errors`）；该差距对两个设定有误的模型进行排序，而不对任何一个模型进行检验。

我们两种方案之间的差距小于该书的结果；由于常数项的笔误以及数据版本的不同，$\log L$ 的水平值不具可比性，但差距的符号是相同的。

残差显示了差距的来源，以及为什么 $U^*$ 难以精确确定：

```{code-cell} ipython3
for s, label in [('C', 'classical'), ('K', 'Keynesian')]:
    v1, v2 = residuals(est[s]['params'], est[s]['U'], est[s]['v2'])
    rms = lambda v: np.sqrt(np.mean(v**2))
    print(f"{label:10s}: misses v2: root mean square {rms(v2):.2f}, mean {v2.mean():.2f};  "
          f"v1: root mean square {rms(v1):.2f}, mean {v1.mean():.2f};  "
          f"corr(v1, v2) = {np.corrcoef(v1, v2)[0, 1]:.2f}")
print(f"average unemployment over the sample: {est['C']['U'][2:].mean():.2f}")
```

就均方根而言，经典政府的误差是凯恩斯政府误差的两倍多。

两组误差的均值都为正，因为在很长的时间段内建议值都低于实际通货膨胀率。

由于 $v_1$ 与 $v_2$ 相关，$v_1$ 的均值也可能为正，而较低的 $U^*$ 与 $v_1$ 的正均值相结合，拟合效果几乎与较高的 $U^*$ 一样好；这大概就是为什么我们和该书都发现 $U^*$ 的估计值低于样本期平均失业率的原因。

## 建议与误差

该模型中通货膨胀的拟合值并非外部计量经济学家做出的预测：它正是政府自身的程序根据其所见数据给出的建议通货膨胀率。

接下来的两幅图重现了该书的图 9.1 和图 9.2a：每幅图都展示了建议值与实际通货膨胀率的对比，以及政府估计的菲利普斯曲线中通货膨胀系数之和。

当该系数之和在经典回归中为 $0$，在凯恩斯回归中为 $1$ 时，归纳假设（{ref}`pa_induction`）成立。

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

首先来看经典政府。

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

经典政府几乎始终建议低于零的通货膨胀率，仅在 1970 年前后短暂升至零以上。

正如该书所言："经典模型甚至未能匹配 1970 年之前通货膨胀的加速过程。"

其以当期及滞后通货膨胀对失业率所做的回归发现，通货膨胀系数之和很小，并且从 20 世纪 70 年代初开始接近于零。

这样的政府几乎看不到任何可利用的权衡，因此费尔普斯问题指示它追求低通货膨胀。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Keynesian government: recommended and actual inflation, and the sum of coefficients on lagged inflation (book Figure 9.2a)"
    name: fig-ppe-keynesian
---
plot_government(gov['K'], 1, 'Keynesian identification', 'sum on lagged $y$')
```

凯恩斯政府的建议更好地反映了通货膨胀模式，部分匹配了 20 世纪 60 年代末的加速过程。

然而，从 20 世纪 70 年代初开始，这些建议明显低于实际通货膨胀率，并且在此后约二十年的大部分时间里都低于实际值。

下方的面板解释了原因。

滞后通货膨胀的权重之和从 20 世纪 60 年代远低于 1 的水平，攀升至 20 世纪 70 年代中期的约 1，并在此后保持在这一水平附近。

到那时，凯恩斯政府已经学会了索洛-托宾版本的自然率假说，费尔普斯问题指示它降低通货膨胀。

### 归纳假设的形成

该书的图 9.2b 展示了数据如何逐渐消除政府的疑虑。

我们用 $\hat\sigma_t^2 \left(\sum_{s\le t}\lambda^{t-s} X_{K,s}X_{K,s}^\top\right)^{-1}$ 来近似政府对 $\beta_t$ 的不确定性，其中 $\hat\sigma_t^2$ 为贴现残差方差。

这正是实现常增益最小二乘法的卡尔曼滤波会赋予其估计值的协方差矩阵。

该图绘制了 1960 年、1965 年……1985 年各年第一季度，滞后通货膨胀权重之和与常数项的 95% 和 99% 置信椭圆。

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

这一模式与该书的图形以及第 8 章模拟中的置信椭圆相似。

20 世纪 60 年代，椭圆远离归纳假设（点线），但足够大，留有怀疑的余地。

20 世纪 70 年代期间，中心点移动到归纳假设线上并保持在那里，椭圆也比 20 世纪 60 年代初的更小，尽管不如该书中的那样小。

用该书的话说，点估计发生了移动，"使得椭球体的中心发生偏移，疑虑逐渐消散。"

(ppe_misses)=
## 误差支持平反

若按逐期拟合效果来评判，这些适应性模型是令人失望的：即便对凯恩斯模型而言，20 世纪 70 年代也存在一长串低估通货膨胀的情况。

萨金特给出了不同的解读：

> 但这样的误差不一定不能为计量经济政策评估正名。恰恰相反，估计模型的误差模式支持平反。记住适应性模型的拟合值就是政府随时间推移给出的建议，这就从计量经济学上的缺陷中提炼出了平反的含义。

这些误差表明，建议值系统性地*低于*实际发生的通货膨胀率：政府的程序一直在告诉它要实行反通胀政策，而实际政策并未这样做。

一步建议 $h(\gamma_{t-1})X_{t-1}$ 混合了两样东西：政府希望通货膨胀走向何方，以及它计划以多快的速度从继承而来的通货膨胀水平到达那里。

为了将两者分开，我们计算政府计划的稳态，即如果当前信念正确，费尔普斯规则会将经济引导至的长期通货膨胀率。

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

下图将凯恩斯政府的计划与其一步建议、经典建议以及实际通货膨胀率并列呈现。

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

五个时期的平均值对该图做了总结。

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

这一计算印证了该书的解读：

> 计量经济学估计告诉我们，未经改造的凯恩斯菲利普斯曲线拟合者本会发现经验菲利普斯曲线出现了不利的偏移，并通过归纳假设结合费尔普斯问题本会建议降低通货膨胀率。这些量化政策评估者本不会认同 20 世纪 70 年代末流行的那种论证松散的说法，即预期中的长滞后使得反通胀的代价过高。我们的结果表明，经典识别方案下的建议本会更加及时。

整个 20 世纪 60 年代，凯恩斯政府的目标是将长期通货膨胀率定在百分之二到三之间。

一旦其估计的滞后通货膨胀权重之和接近 1，其目标就降到了约百分之一或更低，并在整个 20 世纪 80 年代初一直保持在这一水平，而实际通货膨胀率则远高于此。

它在 20 世纪 70 年代末的一步建议约为百分之四，呼吁将通货膨胀大幅降至继承而来的水平之下。

沃尔克反通胀之后，权重之和略有下降，计划的长期通货膨胀率回到了百分之二到三，接近实际通货膨胀率。

即使在 20 世纪 60 年代末的加速期，经典政府的建议也接近于零，此后更是低于零：这正是其建议"本会更加及时"的含义所在。

该书第 1 章得出了这样的结论："计量经济学估计提出了一个谜团：为什么稳定政策被推迟了。"

### 这些误差有多稳健？

由于该书没有报告其滞后阶数或转换为季度数据的方法，下表对滞后阶数、季度取样方式以及频率进行了变化处理。

月度数据栏使用 $\lambda^{1/3}$ 和 $\delta^{1/3}$，以保持相同的记忆长度和相同的日历时间贴现率。

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

一些发现在每一列中都成立：凯恩斯方案拟合得更好，经典政府在 20 世纪 70 年代建议低于零的通货膨胀率。

无论滞后阶数或取样方式如何，只要使用季度数据，凯恩斯政府滞后通货膨胀权重之和在 1975 年后都接近 1，其长期计划在 1973–1984 年间呼吁实行约百分之一或更低的通货膨胀率。

*一步*凯恩斯建议则更为脆弱，对 $\theta$ 的估计也是如此。

原因在于凯恩斯回归中当期失业率的系数 $\beta_1$，政府通过对其求逆得到其短期权衡 $1/\beta_1$。

使用一期失业率滞后时，1973–1979 年间该系数明显为负；使用两期滞后时，季末取样下该系数约为原来的一半，而季度平均取样下则约为零。

{cite:t}`Sims1971`（在 {doc}`approximation_errors` 中有所描述）表明，点取样和平均取样会将同一个连续时间关系投影到不同的离散时间滞后分布上。

对曲线求逆时就要除以一个接近零的数，于是所信奉的失业率动态变得不稳定，政府对继承的通货膨胀的短期反应也随之剧烈波动；这就是这些列中一步误差变小的原因。

月度数据失败的原因更为根本。

两期月度通货膨胀滞后仅跨越两个月，其系数之和远达不到 1，因此月度凯恩斯政府从未激活归纳假设，其长期计划随通货膨胀上升而非下降。

一个足以捕捉归纳假设的分布滞后，需要跨越若干季度，而不是若干个月。

### 这是一个实证结论，而非规范性结论

萨金特并未将这些估计结果当作背书。

{doc}`phillips_learning` 中的模拟表明，均值动态最终会将适应性政府拉回自我确认均衡，因此一段时期的良好建议并不能保证政策长期良好。

该书第 10 章结尾写道：

> 最后，虽然我们的模拟和计量经济学证据支持计量经济政策评估得到平反的故事，但我们希望这是一个错误的故事。我们反而希望政策制定者以某种方式已经学会了自然率假说的正确的理性预期版本，并致力于实行低通货膨胀。

## 后续发展

第 9 章固定了增益。

{ref}`olg_ex7` 提出了钟（Chung）的问题，即在有界理性系列模型的模拟数据上估计增益本身，其中增益被称为 $\gamma$，信念被称为 $\alpha$；下文的 {ref}`ppe_ex1` 则在本讲座的数据上探讨了同样的问题。

{ref}`pbr_compliment` 解释了为何一个不消失的增益使得这类学习参数是可估计的。

{cite:t}`SargentWilliamsZha2006` 是本章内容的现代版本。

他们的政府在漂移的菲利普斯曲线系数上运行卡尔曼滤波（{doc}`kalman`），即 Sims–Chung 设定，并且与第 9 章一样，通货膨胀的系统性部分在政府当前的估计下求解费尔普斯问题。

与第 9 章不同，他们使用贝叶斯马尔可夫链蒙特卡洛算法，将设定政府增益的漂移协方差与真实模型的参数一起联合估计。

{doc}`phillips_priors` 讨论了他们的估计以及 {cite:t}`CarboniEllison2009` 的改进方法，后者利用美联储的绿皮书预测来约束政府的信念。

{doc}`phillips_lost_conquest` 就 21 世纪 20 年代的美联储提出了钟的问题。

在本讲座中，政府的信念在似然函数中没有留下任何自由参数。

下一讲，{doc}`phillips_drifts_volatilities`，采取了相反的方法：它拟合了一个简约式向量自回归，其系数会漂移，其冲击具有随机波动率。

它探讨了平反故事所认为处于战后通货膨胀核心的漂移信念，是否在数据中留下了该故事所预测的痕迹。

## 练习

```{exercise-start}
:label: ppe_ex1
```

该书将 $\lambda = .985$ 固定不变。

计算增益的廓线似然：对于从 $.95$ 到 $1$ 的 $\lambda$ 网格，重建政府的信念和建议，重新估计 $(\theta, U^*, \rho_1, \rho_2)$，并绘制两种识别方案下最大化的 $\log L$ 对 $\lambda$ 的图。

注意 $\lambda = 1$ 即为最小二乘法，其增益以 $1/t$ 的速率衰减。

数据偏好哪些 $\lambda$ 值？这一答案对该书中"为什么稳定政策被推迟"这一谜团有何启示？

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

经典似然函数在 $\lambda$ 高于 .97 时几乎是平坦的，因为无论记忆长度如何，经典政府都建议实行低通货膨胀。

随着 $\lambda$ 趋近于 1，凯恩斯似然函数稳步上升：数据更偏好记忆衰减最慢的政府。

打印出来的平均值解释了原因。

记忆更长的政府会赋予 20 世纪 50 年代和 60 年代更大的权重，而当时的回归显示存在长期权衡，因此该政府更晚激活归纳假设，在 20 世纪 70 年代建议更高的通货膨胀率，更接近实际发生的通货膨胀率。

由于似然函数只有在建议值更接近实际通货膨胀率时才能对增益给予奖励，因此它偏好学习速度最慢的政府。

这是陈述该书谜团的另一种方式：实际政策的表现仿佛比该书中常增益政府更晚才学会自然率假说。

在 {ref}`olg_ex7` 的模拟经济中，政府确实遵循其建议，似然函数能恢复出真实的增益；而在这里，对缓慢学习的偏好在一定程度上衡量了实际政策相对于该程序滞后了多少。

```{solution-end}
```

```{exercise-start}
:label: ppe_ex2
```

{cite:t}`Sims1988` 给其政府设定了一条*静态*菲利普斯曲线，$U_t = \gamma_0 y_t + \gamma_c + \varepsilon_t$，不含滞后项。

在两种识别方案下估计该静态模型（$m_U = m_y = 0$），并与本讲座中的分布滞后模型进行比较。

对这四个模型中的每一个，报告最大化的 $\log L$、1973–1979 年间的平均建议值，以及建议通货膨胀率与实际通货膨胀率之间的相关性。

静态经典模型是否失败？滞后结构与拟合方向，哪一个因素更为重要？

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

静态经典模型确实失败了：它在 20 世纪 70 年代建议接近零的通货膨胀率，其建议与实际通货膨胀率呈负相关。

但增加滞后项并不能挽救经典方案：分布滞后经典政府的建议甚至更低，其似然值也低于静态模型。

拟合方向比滞后结构更为重要，并且两个凯恩斯模型的拟合效果都远优于两个经典模型。

在以通货膨胀对失业率所做的回归中，季度通货膨胀中的噪声会使通货膨胀系数产生衰减偏误，因此经典政府只能看到微弱的权衡；而对以失业率对通货膨胀所做的回归求逆，则会得到一个陡峭的权衡关系。

对于凯恩斯方案，滞后结构也很重要：分布滞后模型具有更高的似然值，其建议对实际通货膨胀率的追踪也要紧密得多。

只有分布滞后结构才能让政府发现归纳假设；静态凯恩斯政府的模型中不含滞后通货膨胀，因此它无法以索洛-托宾形式学习自然率假说。

该书从 {cite:t}`KingWatson1994` 那里采用了这两种识别方案及其名称，并指出凯恩斯方向更好的拟合效果正是美国菲利普斯曲线文献采纳该方向的原因。

```{solution-end}
```

```{exercise-start}
:label: ppe_ex3
```

取凯恩斯政府在 1979 年第二季度末、即保罗·沃尔克就任美联储主席之前的信念。

计算该政府为接下来二十个季度*计划*的通货膨胀率和失业率路径：即从 1979 年第三季度的状态出发，将所有未来冲击设为零，根据其费尔普斯规则和自己估计的菲利普斯曲线所隐含的路径。

对经典政府做同样的计算，并将两种计划与实际发生的情况进行对比绘图。

凯恩斯计划与实际的反通胀过程相比如何？两个政府分别预期为此要付出多大的失业率上升代价？

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

在沃尔克获任命的前夕，凯恩斯政府计划立即削减通货膨胀率，从两位数降至几个百分点，然后缓慢将其进一步降至零。

它预期这一削减会付出代价：在其计划中，失业率跳升至远高于 1979 年水平的程度，之后再缓慢下降。

这正是该书第 5 章（{doc}`phillips_adaptive`）中费尔普斯问题的模式，即政府"人为制造了一场重大衰退，并立即将通货膨胀降低到超过其最终极限值一半以上的水平。"

经典政府计划实行温和的通货紧缩，并预期代价很小，因为它估计的曲线几乎未显示出通货膨胀对失业率的短期影响。

实际政策的推进比这两个计划都要缓慢：通货膨胀率在一年多的时间里一直居高不下，直到 1982–1983 年才降至 20 世纪 80 年代的水平，而白人男性失业率则在 1982 年底达到峰值。

两个政府的计量经济学都主张立即反通胀，而凯恩斯政府即使预期反通胀代价高昂，也仍然建议实行反通胀。

两者都不会接受 20 世纪 70 年代末流行的那种论点，即预期中的长滞后使得反通胀的尝试代价过高。

```{solution-end}
```