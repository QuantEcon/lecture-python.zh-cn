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
translation:
  title: 设定检验
  headings:
    Overview: 概览
    Wu's statistic: 吴氏统计量
    Hausman's lemma: 豪斯曼引理
    Testing independence of irrelevant alternatives: 检验无关备择方案的独立性
    Hansen and Sargent's test of exogeneity: 汉森和萨金特的外生性检验
    Comparing frequency bands: 比较频段
    Concluding remarks: 结语
    Exercises: 练习
---

(specification_tests)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 设定检验

```{contents} Contents
:depth: 2
```

## 概览

本讲座中的设定检验比较了两个估计量。

当模型设定正确时，两个估计量都会收敛到真实参数。

当模型设定错误时，这两个估计量会收敛到 {doc}`information_projections` 意义上的两个 m-投影：即真实分布在两个不同模型上的投影，或者是真实分布衍生出的两个分布在同一模型上的投影。

在大样本情形下，只有当设定误差使得这两个投影的参数产生差异时，检验才对该设定误差具有检验力；对于接近原假设的备择假设，其检验力随着这两个投影之间的距离增大而增大，该距离用我们下面定义的度量 $V(\hat q)^{-1}$ 来衡量。

本讲座即围绕这一思路组织展开。

{doc}`approximation_errors` 计算了设定错误的时间序列模型的伪真值；本讲座描述了比较两个伪真值的检验方法。

* {cite:t}`Wu1973` 比较了普通最小二乘法与工具变量法。在方差保持固定的情况下，他的统计量是一个直角三角形两条直角边长度平方之比，这正是 {doc}`information_projections` 中毕达哥拉斯定理意义下的比值。
* {cite:t}`Hausman1978` 比较了一个有效估计量与一个在备择假设下仍保持一致性的估计量。他的引理指出，有效估计量与两估计量之差不相关，这使得该差的方差在渐近协方差所定义的内积下构成一个毕达哥拉斯定理。
* {cite:t}`HausmanMcFadden1984` 比较了多项逻辑模型的两个估计量。在他们的三选项示例中，似然比统计量是 KL 散度的某个倍数，而拉格朗日乘数统计量和豪斯曼统计量则是该散度在不同坐标系下二次近似的相同倍数。
* {cite:t}`HanSar1980` 通过比较两个拟合模型，提出了一个检验理性预期模型中变量严格外生性的检验方法。
* 我们构造了一个检验，用于比较使用全部频率的频域最大似然估计量与删去季节性频段后的估计量。

{doc}`turbulence_inference` 将投影比较应用于一个经过校准的搜寻模型。

我们使用以下导入项。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.stats import chi2, ncx2, binom
from scipy.stats import f as f_dist
```

(st_wu_section)=
## 吴氏统计量

{cite:t}`Wu1973` 研究了回归方程 $y = Y_2 \beta + Z_1 \gamma + u$，其中 $Y_2$ 中的 $G$ 个回归量可能与扰动项 $u$ 相关，而 $Z_1$ 中的 $K_1$ 个回归量是外生的。

这些回归量满足 $Y_{i2} = \mu_i + V_i$，其中 $\mu_i$ 是未知的非随机向量，$(u_i, V_i)$ 服从联合正态分布；在联立方程系统中，$\mu_i$ 是一组包含 $Z_{i1}$ 的工具变量向量 $Z_i$ 的线性函数。

原假设是 $Y_2$ 与 $u$ 相互独立，即 $E[V_i u_i] = 0$。

令 $\sigma_{11} = \operatorname{Var}(u_i)$，$\delta = E[V_i u_i]$，$\Sigma_{22} = \operatorname{Var}(V_i)$。

在备择假设下，联合正态性使得 $E[u_i \mid V_i] = V_i^\top \alpha$，其中 $\alpha = \Sigma_{22}^{-1} \delta$，因此扰动项 $u_i - V_i^\top \alpha$ 独立于 $V_i$ 以及 $Y_{i2}$。

{cite:t}`Hausman1978` 在他的方程(2.18)–(2.23)中，将吴氏的比较表示为一个回归。

他将 $Y_2$ 拆分为对工具变量回归所得的拟合值（在两种假设下均渐近地与 $u$ 不相关）以及该回归的残差矩阵 $\hat V$。

然后他将 $y$ 对 $Y_2$、$Z_1$ 和 $\hat V$ 进行回归，并检验 $\hat V$ 的系数 $\alpha$ 是否为零；该回归中 $Y_2$ 的系数即为工具变量估计量。

他给出的一般版本(2.23)用拟合值代替了 $\hat V$；这两个回归具有相同的拟合值和相同的 $\alpha = 0$ 检验。

{doc}`ols` 将这种回归应用于阿西莫格鲁、约翰逊和鲁滨逊的数据中，其中有一个内生回归量。

考虑给定回归量下 $y$ 的条件正态分布的两个族，二者方差均固定为 $\sigma_{11}$：

* 原假设模型，均值为 $Y_2 \beta + Z_1 \gamma$；以及
* 更大的模型，均值为 $Y_2 \beta + Z_1 \gamma + \hat V \alpha$。

在备择假设下，更大模型的扰动项 $u_i - V_i^\top \alpha$ 的方差小于 $\sigma_{11}$；将两个方差都固定为 $\sigma_{11}$ 使得这两个族具有共同方差且相互嵌套。

这是两个嵌套的指数族，而原假设模型正是 $\alpha = 0$ 时的指数子族。

普通最小二乘法以及 $y$ 对 $Y_2$、$Z_1$ 和 $\hat V$ 的最小二乘回归，分别是这两个族中的最大似然估计量，因此它们是数据在这两个族上的 m-投影。

{doc}`information_projections` 中的毕达哥拉斯定理 {eq}`ip_pythagoras_m`，以适用于连续结果的对数似然形式 {eq}`ip_lr_kl` 表示，因此适用，并且在此高斯情形下，它就是最小二乘法的毕达哥拉斯定理：

$$
Q_4 = Q_2 + Q^* .
$$ (st_wu)

我们沿用吴氏的下标记号。

这里，$Q_4$ 是普通最小二乘法的残差平方和，$Q_2$ 是更大回归的残差平方和，$Q^*$ 是两个拟合值向量之间的平方距离。

除以 $\sigma_{11}$ 后，$Q_4 - Q_2$ 即为两种拟合的对数似然比的两倍，而 $Q^*$ 是各观测值上两个拟合条件分布之间 KL 散度之和的两倍；{eq}`ip_lr_kl` 使二者相等。

吴氏从普通最小二乘估计量 $b_1$ 与 $\beta$ 的工具变量估计量 $b_2$ 之差计算出 $Q^*$：

$$
Q^* = (b_1 - b_2)^\top \left[ (Y_2^\top A_2 Y_2)^{-1} - (Y_2^\top A_1 Y_2)^{-1} \right]^{-1} (b_1 - b_2) ,
$$

其中 $Z = [Z_1, Z_2]$ 是包含全部 $K_1 + K_2$ 个工具变量的 $N \times (K_1 + K_2)$ 矩阵，$A_1 = I - Z_1(Z_1^\top Z_1)^{-1}Z_1^\top$，$A_2 = Z(Z^\top Z)^{-1}Z^\top - Z_1(Z_1^\top Z_1)^{-1}Z_1^\top$。

吴氏将 $Q_2 = Q_4 - Q^*$ 定义为他的方程(2.19)，并证明在原假设下且以 $Y_2$ 为条件时，$Q^*/\sigma_{11}$ 和 $Q_2/\sigma_{11}$ 是相互独立的卡方变量。

$Q_2$ 同时也是豪斯曼更大回归的残差平方和。

在剔除 $Z_1$ 的影响后，回归量 $A_1 Y_2$ 被拆分为正交的两部分 $A_2 Y_2$ 和 $\hat V$，而豪斯曼的方程(2.21)表明，对 $\alpha = 0$ 的检验恰恰重现了 $Q^*$。

吴氏统计量

$$
T_2 = \frac{Q^* / G}{Q_2 / (N - K_1 - 2G)}
$$

是直角三角形 {eq}`st_wu` 两条直角边长度平方（各自除以自由度）之比，其中 $N$ 为观测数。

吴氏证明，在原假设下，只要 $N - K_1 > 2G$，$T_2$ 就服从自由度为 $G$ 和 $N - K_1 - 2G$ 的 $F$ 分布。

他的统计量 $T_4$ 和 $T_3$ 分别用最小二乘法和工具变量法对 $\sigma_{11}$ 的估计值去除 $Q^*$。

他证明，比值 $G T_2 / T_4$ 和 $T_4 / T_3$ 依概率收敛到大于等于1的常数，且仅当 $\delta = 0$ 时等于1。

因此，在他对检验力函数的大样本近似中，在相同的 $\chi^2_G$ 临界值下，$G T_2$ 的拒绝频率至少与 $T_4$ 一样高，而 $T_4$ 的拒绝频率至少与 $T_3$ 一样高。

吴氏全文都假设工具变量数多于内生回归量数，即 $K_2 > G$；只有他的统计量 $T_1$（分母自由度为 $K_2 - G$）需要这一假设。

下面的代码检验了 {eq}`st_wu` 式以及一个模型精确的 $F$ 分布，该模型 $Y_2$ 中仅有一个回归量，$Z_1$ 中无回归量，仅有一个工具变量，且 $N = 25$ 个观测值。

```{code-cell} ipython3
N, β_true, Π = 25, 1.0, 0.6
rng = np.random.default_rng(0)
z_inst = rng.normal(size=N)                      # nonstochastic instrument


def wu_statistics(y, x, z_inst):
    x_hat = z_inst * (z_inst @ x) / (z_inst @ z_inst)
    b1, b2 = (x @ y) / (x @ x), (x_hat @ y) / (x_hat @ x)
    Q_star = (b1 - b2)**2 / (1 / (x_hat @ x_hat) - 1 / (x @ x))
    X = np.column_stack([x, x - x_hat])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    fit0, fit1 = b1 * x, X @ coef
    Q4, Q2 = np.sum((y - fit0)**2), np.sum((y - fit1)**2)
    return Q_star, Q4, Q2, np.sum((fit1 - fit0)**2)


u = rng.normal(size=N)
x = Π * z_inst + rng.normal(size=N)
Q_star, Q4, Q2, dist2 = wu_statistics(β_true * x + u, x, z_inst)
print(f"Q4 = {Q4:.6f},  Q2 + Q* = {Q2 + Q_star:.6f},  Q* = {Q_star:.6f},  "
      f"squared distance between fits = {dist2:.6f}")

R = 20_000
T2 = np.empty(R)
for r in range(R):
    u = rng.normal(size=N)
    x = Π * z_inst + rng.normal(size=N)
    Q_star, _, Q2, _ = wu_statistics(β_true * x + u, x, z_inst)
    T2[r] = Q_star / (Q2 / (N - 2))
crit = f_dist.ppf(0.95, 1, N - 2)
print(f"fraction of T2 above the 5% critical value of F(1, {N - 2}): "
      f"{np.mean(T2 > crit):.4f}")
```

(st_hausman_section)=
## 豪斯曼引理

{cite:t}`Hausman1978` 将此比较推广到任意一对估计量 $\hat \beta_0$ 和 $\hat \beta_1$，只要在原假设下两者都是一致且渐近正态的，且 $\hat \beta_0$ 是渐近有效的，而在备择假设下只有 $\hat \beta_1$ 保持一致性。

他的引理2.1陈述如下：

> 考虑两个都一致且渐近正态分布的估计量 $\hat\beta_0$、$\hat\beta_1$，其中 $\hat\beta_0$ 达到渐近克拉美-罗下界，从而 $\sqrt{T}(\hat\beta_0 - \beta) \overset{A}{\sim} N(0, V_0)$，而 $\sqrt{T}(\hat\beta_1 - \beta) \overset{A}{\sim} N(0, V_1)$，其中 $V_0$ 是费舍尔信息矩阵的逆。考虑 $\hat q = \hat\beta_1 - \hat\beta_0$。那么 $\sqrt{T}(\hat\beta_0 - \beta)$ 与 $\sqrt{T}\hat q$ 的极限分布具有零协方差，$C(\hat\beta_0, \hat q) = 0$，即一个零矩阵。

豪斯曼给出了理由：“若非如此，便可以取 $\hat\beta_0$ 与 $\hat q$ 的某个线性组合，得到一个一致估计量 $\hat\beta_*$，其渐近方差会小于被假定为渐近有效的 $\hat\beta_0$。”

由此可推出豪斯曼的推论2.6：$\hat\beta_1 = \hat\beta_0 + \hat q$ 中的两项不相关，因此它们的方差相加，

$$
V_1 = V_0 + V(\hat q) .
$$ (st_hausman_pythagoras)

以渐近协方差作为内积，{eq}`st_hausman_pythagoras` 就是一个毕达哥拉斯定理：有效估计量与两估计量之差正交。

当 $\hat\beta_1$ 是嵌套了原假设模型的某个模型的最大似然估计量时（如吴氏情形中那样），{eq}`st_hausman_pythagoras` 即为 {doc}`information_projections` 中毕达哥拉斯定理的局部形式。

设更大的模型具有参数 $(\beta, \psi)$，原假设模型施加 $\psi = \psi_0$ 的约束。

{doc}`exponential_families` 表明，在真实分布附近，KL 散度约等于费舍尔信息矩阵下的一个二次型的一半，因此到原假设模型的 m-投影变成费舍尔内积下的正交投影。

受约束估计量 $\hat\theta_0$ 于是近似地等于无约束估计量 $\hat\theta_1$ 到原假设模型上的正交投影，而该直角三角形的直角边 $\hat\theta_1 - \hat\theta_0$ 与原假设模型正交；其 $\beta$-坐标即为 $\hat q$。

向量 $\Delta = \sqrt{T}(\hat\theta_1 - \theta)$ 的协方差矩阵为 $\mathcal{I}^{-1}$，所以对任意向量 $a$ 和 $b$，有 $\operatorname{Cov}(a^\top \mathcal{I}\Delta, \, b^\top \mathcal{I}\Delta) = a^\top \mathcal{I} b$：$\Delta$ 中在费舍尔内积 $\langle a, b \rangle = a^\top \mathcal{I} b$ 下正交的分量不相关。

沿原假设模型方向及与其正交方向上的分量，其 $\beta$-坐标分别为 $\sqrt{T}(\hat\beta_0 - \beta)$ 和 $\sqrt{T}\hat q$。

对于一般的一致估计量 $\hat\beta_1$，该引理在两估计量展开式的主导线性项上也表达了相同的结论：有效估计量所对应的项与两项之差不相关。

豪斯曼在其方程(2.7)中称之为 $m$ 的统计量为 $H = T \hat q^\top \tilde V^{-1} \hat q$，其中 $\tilde V$ 是 $V(\hat q) = V_1 - V_0$ 的一致估计。

他的定理2.1指出，在原假设下 $H$ 渐近地服从 $\chi^2_K$ 分布，其中 $K$ 是 $\beta$ 中的参数个数；他的脚注12指出，在其工具变量示例中，只有当工具变量足够多时，$V(\hat q)$ 才是非奇异的。

在吴氏模型中，$\hat \beta_0$ 是普通最小二乘，$\hat \beta_1$ 是工具变量法，以最小二乘法估计 $u$ 的方差计算出的 $H$ 等于 $Q^*$ 除以该估计值。

豪斯曼在其脚注11中指出，这一选择给出了该检验的拉格朗日乘数形式，而用工具变量法估计方差则给出沃尔德形式。

这两种形式正是吴氏的 $T_4$ 和 $T_3$。

下面的仿真检验了原假设下的 {eq}`st_hausman_pythagoras` 式，并展示了当原假设为假时这两个估计量各自收敛到的值。

恰好识别的工具变量估计量没有有限矩，因此我们将仿真得到的方差解读为对渐近方差的近似；在一阶段系数为 $0.6$、$n = 400$ 个观测值的情形下，这些近似是准确的。

```{code-cell} ipython3
def ols_iv(σ_uv, rng, n=400, R=20_000, chunk=5_000):
    b_ols, b_iv = [], []
    for start in range(0, R, chunk):            # chunks limit memory use
        size = min(chunk, R - start)
        z_ = rng.normal(size=(size, n))
        u_ = rng.normal(size=(size, n))
        v_ = σ_uv * u_ + np.sqrt(1 - σ_uv**2) * rng.normal(size=(size, n))   # V
        x_ = Π * z_ + v_
        y_ = β_true * x_ + u_
        b_ols.append((x_ * y_).sum(axis=1) / (x_ * x_).sum(axis=1))
        b_iv.append((z_ * y_).sum(axis=1) / (z_ * x_).sum(axis=1))
    return np.concatenate(b_ols), np.concatenate(b_iv)


rng_h = np.random.default_rng(1)
b_ols, b_iv = ols_iv(0.0, rng_h)
q = b_iv - b_ols
print(f"Var(IV) = {b_iv.var():.3e},  "
      f"Var(OLS) + Var(IV - OLS) = {b_ols.var() + q.var():.3e}")
print(f"correlation of OLS with IV - OLS: {np.corrcoef(b_ols, q)[0, 1]:.4f}")

b_ols, b_iv = ols_iv(0.3, rng_h)
print(f"\nwith Cov(u, V) = 0.3: mean OLS = {b_ols.mean():.4f} "
      f"(pseudo-true value {β_true + 0.3 / (Π**2 + 1):.4f}), "
      f"mean IV = {b_iv.mean():.4f}")
```

当 $u$ 与 $V$ 相关时，普通最小二乘法收敛到伪真值 $\beta + \operatorname{Cov}(u, x)/\operatorname{Var}(x)$，即真实分布投影到假定独立模型上的 m-投影系数，而工具变量法则收敛到 $\beta$。

在一系列局部备择假设下，$H$ 渐近服从非中心 $\chi^2$ 分布，其非中心参数等于 $T$ 乘以两估计量概率极限之间在度量 $V(\hat q)^{-1}$ 下的平方距离。

豪斯曼检验所检测的是，感兴趣的参数是否取决于真实分布被投影到哪个模型上，而非检测模型本身是否为真；若某种设定误差使得两个概率极限相等，该检验对此没有检验力。

豪斯曼的脚注13给出了一个例子：在没有滞后因变量的回归中，序列相关的扰动项不会影响普通最小二乘法和自回归估计量的一致性，因此 $\hat q$ 的概率极限为零。

(st_iia_section)=
## 检验无关备择方案的独立性

{doc}`exponential_families` 中的多项逻辑模型给出了决策者从选择集 $C$ 中选择方案 $i$ 的概率为

$$
P(i \mid z, C, \beta) = \frac{\exp(z_i^\top \beta)}{\sum_{j \in C} \exp(z_j^\top \beta)} ,
$$

其中 $z_j$ 是方案 $j$ 的属性向量。

这些概率构成了一个以 $\beta$ 为参数、以 $z_i$ 为充分统计量的指数族。

逻辑模型具有**无关备择方案独立性**的性质：

$$
P(i \mid z, C, \beta) = P(i \mid z, A, \beta) \cdot P(A \mid z, C, \beta)
$$

对于 $C$ 的子集 $A$ 中的选择 $i$，其中 $P(i \mid z, A, \beta)$ 是仅应用于 $A$ 中方案的逻辑公式。

{cite:t}`HausmanMcFadden1984` 比较了由全部选择计算得到的最大似然估计量 $\hat\beta_C$ 与仅由落入 $A$ 的选择计算得到的条件最大似然估计量 $\hat\beta_A$。

这两个估计量分别是两个分布的 m-投影：全部选择的分布在逻辑模型上的投影，以及 $A$ 内选择的条件分布在限制于 $A$ 的逻辑模型上的投影。

当逻辑模型设定正确时，两个估计量都收敛到真实的 $\beta$。

当真实模型是嵌套逻辑模型（即同一分支中方案的未观测效用相关）时，这两个估计量收敛到不同的值。

豪斯曼和麦克法登给出了一个三选项示例，其中方案2和方案3具有相同的属性，因此在逻辑模型下它们具有相同的选择概率。

设 $n_1, n_2, n_3$ 分别为各方案被选中的次数，$m = n_2 + n_3$，$s = n_2 / m$。

豪斯曼和麦克法登推导出一个似然比统计量、一个拉格朗日乘数统计量，以及一个通过删去方案3而计算得到的豪斯曼统计量：

$$
LR = 2 n_2 \log \frac{2 n_2}{n_2 + n_3} + 2 n_3 \log \frac{2 n_3}{n_2 + n_3} ,
\quad
LM = \frac{(n_3 - n_2)^2}{n_2 + n_3} ,
\quad
H_3 = \left( \log \frac{2 n_2}{n_2 + n_3} \right)^2 (n_2 + n_3) .
$$

若改为删去方案2，则得到 $H_2 = \left( \log \frac{2 n_3}{n_2 + n_3} \right)^2 (n_2 + n_3)$。

以 $m$ 为条件，$n_2$ 服从以 $m$ 为试验次数、以 $\pi_2 = p_2/(p_2 + p_3)$ 为成功概率的二项分布，$s$ 是 $\pi_2$ 的最大似然估计量。

原假设设定 $\pi_2 = \frac{1}{2}$。

由 {doc}`information_projections` 中的 {eq}`ip_lr_kl` 式可知，似然比统计量等于 $2m$ 乘以参数为 $s$ 的伯努利分布相对于参数为 $\frac{1}{2}$ 的伯努利分布的 KL 散度。

拉格朗日乘数统计量是该散度在期望参数坐标 $s$ 下二次近似的 $2m$ 倍。

豪斯曼统计量 $H_3$ 是该散度在坐标 $\log s$（既非自然参数 $\log(s/(1 - s))$ 也非期望参数 $s$）下二次近似的 $2m$ 倍，而 $H_2$ 则是该散度在坐标 $\log(1 - s)$ 下二次近似的 $2m$ 倍。

$\pi_2 = \frac{1}{2}$ 处关于 $\pi_2$ 的费舍尔信息量为 $1/(\pi_2(1-\pi_2)) = 4$，而关于 $\log \pi_2$ 的费舍尔信息量为 $\pi_2/(1 - \pi_2) = 1$，二者分别给出二次近似 $2(s - \frac{1}{2})^2$ 和 $\frac{1}{2}(\log 2s)^2$。

```{code-cell} ipython3
def bernoulli_kl(s, p=0.5):
    return s * np.log(s / p) + (1 - s) * np.log((1 - s) / (1 - p))


for n2, n3 in ((230, 270), (180, 320)):
    m, s = n2 + n3, n2 / (n2 + n3)
    LR = 2 * n2 * np.log(2 * n2 / m) + 2 * n3 * np.log(2 * n3 / m)
    LM = (n3 - n2)**2 / m
    H3 = np.log(2 * n2 / m)**2 * m
    print(f"n2, n3 = {n2}, {n3}:  LR = {LR:.4f} = 2m KL = "
          f"{2 * m * bernoulli_kl(s):.4f};  LM = {LM:.4f} = "
          f"{2 * m * 2 * (s - 0.5)**2:.4f};  H3 = {H3:.4f} = "
          f"{2 * m * 0.5 * np.log(2 * s)**2:.4f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: KL divergence and its quadratic approximations
    name: fig-st-iia
---
s_vals = np.linspace(0.2, 0.8, 300)
fig, ax = plt.subplots()
ax.plot(s_vals, bernoulli_kl(s_vals), lw=3, label="KL divergence (LR)")
ax.plot(s_vals, 2 * (s_vals - 0.5)**2, "--", label=r"quadratic in $s$ (LM)")
ax.plot(s_vals, 0.5 * np.log(2 * s_vals)**2, ":",
        label=r"quadratic in $\log s$ ($H_3$)")
ax.plot(s_vals, 0.5 * np.log(2 * (1 - s_vals))**2, "-.",
        label=r"quadratic in $\log(1 - s)$ ($H_2$)")
ax.set_xlabel("$s = n_2 / (n_2 + n_3)$")
ax.set_ylabel("divergence per choice of alternative 2 or 3")
ax.set_ylim(0, 0.3)
ax.legend(loc="upper center", fontsize=9)
plt.show()
```

在原假设以及局部备择假设下，这四个统计量具有相同的渐近分布，但在有限样本中彼此不同。

豪斯曼和麦克法登列表给出了这些检验以及嵌套逻辑模型中参数为 $\lambda$ 的 $\lambda = 1$ 沃尔德检验的精确拒绝概率。

他们的备择模型是 $\lambda < 1$ 的嵌套逻辑模型，其中方案1和方案2属于同一分支，因此 $p_2 < p_3$。

在他们的表III.2和表III.3中，沃尔德检验和 $H_3$ 检验力最强，其次是似然比检验和拉格朗日乘数检验，而 $H_2$ 检验力最弱。

他们的正文中有四处将表中显示为 $H_3$ 的行标注为 $H_2$，又有一处将显示为 $H_2$ 的行标注为 $H_3$；他们在第1233页陈述的排序 WALD > H3 > LR > LM > H2 与表格一致。

他们报告了两个“相当出人意料的结果”：

> (i) 所谓的“神圣三位一体”渐近检验中的三种检验，即使在名义显著性水平相同的情况下，它们的操作特性也存在明显差异。(ii) 沃尔德检验和豪斯曼类型检验的检验力大致相当。其次是似然比检验和拉格朗日乘数检验。这两个结果很可能是因为我们考虑的是非局部的备择假设。

他们补充道：“样本量可能不足以可靠地应用渐近理论。”

{doc}`exponential_families` 比较了泊松族中在自然坐标和期望坐标下对 KL 散度的二次近似。

{ref}`st_ex_iia` 计算了精确的显著性水平和检验力。

(st_hs_section)=
## 汉森和萨金特的外生性检验

{cite:t}`HanSar1980` 推导出企业劳动力需求的理性预期模型对就业人数 $n_t$ 与企业视为给定的变量向量 $x_t$ 的联合过程所施加的跨方程约束。

企业观测到一个计量经济学家观测不到的冲击 $a_t$。

在就业不格兰杰引起 $x_t$ 的前提下，当且仅当一个回归系数向量 $v$ 为零时，向量 $x_t$ 在企业决策规则中才是严格外生的，这等价于要求 $x_t$ 中的新息与 $a_t$ 中的新息不相关。

他们写道：

> 这后一种检验在精神上与吴氏（1973）在经典联立方程设定中提出的外生性检验相似。

并补充道，他们的设定与吴氏的设定不同，“因为我们允许扰动项存在序列相关，并且我们是通过理性预期假设所隐含的非线性跨方程约束来实现参数识别的。”

他们提议用准最大似然法对该模型估计两次，一次令 $v$ 自由，另一次令 $v = 0$，并用一个在原假设下渐近服从 $\chi^2$ 分布的似然比统计量来比较两个似然值。

第二种方法是用 $\hat v$ 的估计协方差矩阵检验 $v = 0$。

他们还提议通过将该模型嵌套于一个不施加这些约束、设定更为宽松的模型中，来对跨方程约束本身进行检验。

{doc}`hansen_singleton_1983` 将这样一种检验应用于消费欧拉方程对向量自回归所施加的跨方程约束。

利用 {doc}`approximation_errors` 中的频域似然 {eq}`ae_whittle` 式，每个对数似然（在不依赖于模型的某一项之外）等于各频率上 $N(0, I(\omega_j))$ 相对于 $N(0, G(\omega_j, \delta))$ 的交叉熵之和的负值。

因此每个似然比统计量都是各频率上取值之和。

一般而言，无约束模型和受约束模型都不是指数族，而跨方程约束使得受约束模型成为无约束模型中的一个曲面子集。

因此，毕达哥拉斯分解式 {eq}`ip_pythagoras_m` 仅在二阶意义上成立，而这恰恰是渐近 $\chi^2$ 分布所使用的阶数。

(st_band_section)=
## 比较频段

我们构造了一个豪斯曼检验，用以比较使用全部频率的频域最大似然估计量与删去季节性频段后的估计量。

{doc}`approximation_errors` 表明，如 {eq}`ae_hs_ideal` 式所示，从汉森-萨金特的判据中删去某些频段，相当于将过程中落在这些频段之外的部分投影到模型中落在这些频段之外的部分上。

当模型设定正确时，两个估计量收敛到相同的参数，且使用全部频率的估计量是有效的，因此豪斯曼引理适用。

当模型在季节性频率处设定错误时，两个估计量收敛到不同的投影。

在 {doc}`approximation_errors` 的季节性示例中，季节性成分的方差为非季节性成分方差的四分之一，AR(1)投影在使用全部频率时为 $\rho^* = 0.62$，而在删去宽度为 $\pi/8$ 的频段后为 $\rho^* = 0.87$。

在下面使用1%占比的情形下，{ref}`st_ex_power` 的解答计算出的值分别为 $0.885$ 和 $0.899$。

对于一个高斯过程以及一个谱密度为 $g(\omega; \theta)$ 的设定正确的标量模型，使用集合 $K$ 中频率的频域最大似然估计量误差乘以 $\sqrt{T}$，具有渐近协方差矩阵 $\mathcal{I}_K(\theta)^{-1}$，其中

$$
\mathcal{I}_K(\theta) = \frac{1}{4\pi} \int_{K} \nabla_\theta \log g(\omega; \theta) \, \nabla_\theta \log g(\omega; \theta)^\top \, d\omega .
$$

如 {doc}`exponential_families` 所述，$\mathcal{I}_K(\theta)$ 是某个 KL 散度的曲率：它是散度率 {eq}`ae_kl_rate` 式（限制在 $K$ 上）在 $\theta' = \theta$ 处关于 $g(\cdot\,; \theta')$ 对 $g(\cdot\,; \theta)$ 的黑塞矩阵。

由 {eq}`st_hausman_pythagoras` 式可知，删去频段后的估计量与全频段估计量之差具有渐近协方差矩阵 $\mathcal{I}_K(\theta)^{-1} - \mathcal{I}(\theta)^{-1}$，其中 $\mathcal{I}$ 使用全部频率。

删去这些频段是 {doc}`blackwell_kihlstrom` 意义下的一种信息扰乱，因此由费舍尔信息形式的数据处理不等式可知 $\mathcal{I}_K(\theta) \preceq \mathcal{I}(\theta)$；求逆会使矩阵序反转，因此该协方差矩阵是半正定的。

我们将该检验应用于一个参数为 $\theta = (\rho, \sigma^2)$、拟合了 $T = 400$ 个季度观测值的一阶自回归模型，删去 $\pi/2$ 和 $\pi$ 附近宽度为 $\pi/8$ 的频段，并仅检验 $\rho$，利用这些矩阵的 $(1, 1)$ 元素。

每个估计量都使用傅里叶频率 $\omega_j = 2\pi j / T$，$j = 1, \ldots, T - 1$，因为样本均值被去除，所以省略 $j = 0$；我们也在相同的这些频率上计算 $\mathcal{I}_K$，使得 $\frac{1}{4\pi} \int_K$ 变为 $\frac{1}{2T} \sum_{j \in K}$。

省略 $j = 0$ 会在 $T = 400$ 时去掉关于 $\rho$ 约10%的信息，这是因为 $x_t$ 的谱密度在零频率处有一个狭窄的峰值；$\mathcal{I}^{-1}$ 的 $(1, 1)$ 元素是 $0.21$ 而非 $1 - \rho^2 = 0.19$。

在每个样本中，我们在全频段估计的 $\rho$ 处对两个信息矩阵求值。

在原假设下，数据来自 $x_t = 0.9 x_{t-1} + e_t$。

在备择假设下，我们加入 {doc}`approximation_errors` 中的季节性成分，此处称为 $w_t$（因为 $s$ 已用来表示豪斯曼-麦克法登的份额）：$(1 + \varphi L + \varphi^2 L^2 + \varphi^3 L^3) w_t = \eta_t$，其中 $\eta_t$ 为白噪声，$\varphi = 0.98$。

我们对 $\eta_t$ 进行缩放，使得 $w_t$ 的方差分别为 $x_t$ 方差的0.5%、1%或2%。

```{code-cell} ipython3
T_obs, ρ_x, φ = 400, 0.9, 0.98
ω_T = 2 * np.pi * np.arange(1, T_obs) / T_obs
z_T = np.exp(-1j * ω_T)


def seasonal_keep(ω_set, Δ):
    """True outside bands of width Δ centered on π/2, π, and 3π/2."""
    def dist(c):
        return np.minimum(np.abs(ω_set - c), 2 * np.pi - np.abs(ω_set - c))
    inside = ((dist(np.pi / 2) < Δ / 2) | (dist(np.pi) < Δ / 2)
              | (dist(3 * np.pi / 2) < Δ / 2))
    return ~inside


keep_all = np.ones(T_obs - 1, bool)
keep_band = seasonal_keep(ω_T, np.pi / 8)
var_x = 1 / (1 - ρ_x**2)
ω_fine = 2 * np.pi * (np.arange(2**14) + 0.5) / 2**14
z_fine = np.exp(-1j * ω_fine)
var_w_unit = np.mean(1 / np.abs(1 + φ * z_fine + φ**2 * z_fine**2
                                + φ**3 * z_fine**3)**2)


def whittle_ar1(I, keep):
    """Frequency-domain ML estimate of ρ in an AR(1), over the frequencies in keep."""
    def criterion(ρ):
        a2 = np.abs(1 - ρ * z_T[keep])**2
        return np.log(np.mean(I[keep] * a2)) - np.mean(np.log(a2))
    return minimize_scalar(criterion, bounds=(-0.999, 0.999), method="bounded",
                           options={"xatol": 1e-8}).x


def ar1_information(ρ, keep):
    """(1/4π) ∫_K ∇log g ∇log g' dω at the Fourier frequencies, θ = (ρ, σ²), σ² = 1."""
    d = np.vstack([2 * (np.cos(ω_T) - ρ) / np.abs(1 - ρ * z_T)**2,
                   np.ones(T_obs - 1)])
    return 0.5 * (d[:, keep] @ d[:, keep].T) / T_obs


def band_test(share, seed, R=1000, burn=200):
    rng = np.random.default_rng(seed)
    stats, ρ_full, ρ_band = np.empty(R), np.empty(R), np.empty(R)
    scale = np.sqrt(share * var_x / var_w_unit)
    for r in range(R):
        x_ = lfilter([1], [1, -ρ_x], rng.normal(size=T_obs + burn))
        w_ = lfilter([1], [1, φ, φ**2, φ**3], rng.normal(size=T_obs + burn))
        y_ = (x_ + scale * w_)[burn:]
        I = np.abs(np.fft.fft(y_ - y_.mean()))[1:]**2 / T_obs
        ρ_full[r] = whittle_ar1(I, keep_all)
        ρ_band[r] = whittle_ar1(I, keep_band)
        V_full = np.linalg.inv(ar1_information(ρ_full[r], keep_all))[0, 0]
        V_band = np.linalg.inv(ar1_information(ρ_full[r], keep_band))[0, 0]
        stats[r] = T_obs * (ρ_band[r] - ρ_full[r])**2 / (V_band - V_full)
    return stats, ρ_full, ρ_band


crit_1 = chi2.ppf(0.95, 1)
stats, ρ_full, ρ_band = band_test(0.0, seed=2)
print(f"null: Var(band) = {ρ_band.var():.3e},  Var(full) + Var(band - full) = "
      f"{ρ_full.var() + (ρ_band - ρ_full).var():.3e}")
print(f"null: rejection frequency at 5%: {np.mean(stats > crit_1):.3f}")
V_full_0 = np.linalg.inv(ar1_information(ρ_x, keep_all))[0, 0]
V_band_0 = np.linalg.inv(ar1_information(ρ_x, keep_band))[0, 0]
print(f"V_band - V_full = {V_band_0 - V_full_0:.4f},  V_full = {V_full_0:.4f}")

# the same seed for every share gives common random numbers
simulated_power, band_draws = {}, {}
for share in (0.005, 0.01, 0.02):
    stats, ρ_full, ρ_band = band_test(share, seed=3)
    simulated_power[share] = np.mean(stats > crit_1)
    band_draws[share] = (ρ_full, ρ_band)
    print(f"seasonal variance {share:.3f} of Var(x): mean ρ_full = "
          f"{ρ_full.mean():.3f}, mean ρ_band = {ρ_band.mean():.3f}, "
          f"rejection frequency {simulated_power[share]:.3f}")
```

在原假设下，仿真方差满足豪斯曼恒等式 {eq}`st_hausman_pythagoras`，且该检验的拒绝频率接近其名义水平。

方差为 $x_t$ 方差1%或2%的季节性成分，足以将全频段估计值拉低于删去频段后的估计值，使得该检验在大多数样本中都能拒绝原假设；而当方差占比为0.5%时，该检验在不到一半的样本中拒绝原假设。

该检验之所以检验力强，是因为对该模型而言，删去季节性频段对关于 $\rho$ 的信息损失很小，而 $\rho$ 的信息主要集中在低频：$\mathcal{I}_K(\theta)^{-1} - \mathcal{I}(\theta)^{-1}$ 的 $(1, 1)$ 元素约为 $\mathcal{I}(\theta)^{-1}$ 的 $(1, 1)$ 元素的2%。

用 {doc}`approximation_errors` 中 {ref}`ae_garbling_section` 的语言来说，该统计量的分子 $T(\hat\rho_{\text{band}} - \hat\rho_{\text{full}})^2$ 等于 $T$ 乘以删去频段所引起的 m-投影偏移量的平方，再加上抽样噪声，而分母则衡量了删去这些频段所损失的信息，因此当偏移量相对于损失的信息较大时，该检验具有检验力。

{ref}`st_ex_power` 根据两个投影之间的距离计算了该检验的渐近检验力，而 {doc}`turbulence_inference` 将同样的计算应用于对收入不稳定性上升的两种解释。

## 结语

本讲座中的每个设定检验都比较了两个估计量，当模型设定正确时，这两个估计量的概率极限重合；而对于使两个概率极限保持相等的某种设定误差，每个检验都没有检验力。

在方差保持固定的情况下，吴氏统计量是一个直角三角形两条直角边长度平方之比，而该三角形的毕达哥拉斯定理正是嵌套指数族上 m-投影的毕达哥拉斯定理。

豪斯曼引理是渐近协方差所定义的内积下的一个毕达哥拉斯定理，并且当那个非有效估计量是嵌套了原假设模型的某个模型的最大似然估计量时，它正是 m-投影毕达哥拉斯定理的局部形式。

在豪斯曼和麦克法登的三选项示例中，似然比、拉格朗日乘数和豪斯曼统计量分别是 $2m$ 乘以某个 KL 散度及其在坐标 $s$、$\log s$ 和 $\log(1 - s)$ 下二次近似的结果。

在我们的例子中，当 $T = 400$ 个季度观测值、非季节性 AR(1) 系数为 $0.9$、季节性频段宽度为 $\pi/8$ 时，比较全频率估计量与季节性频段之外频率估计量的检验，在季节性成分方差至少为非季节性成分方差1%时，在大多数样本中都能拒绝原假设。

{doc}`turbulence_inference` 比较了一个经过校准的搜寻模型以及数据在一个辅助统计模型上的投影，而 {doc}`gallant_tauchen` 将此类比较描述为间接推断。

## 练习

```{exercise}
:label: st_ex_iia

{cite:t}`HausmanMcFadden1984` 通过枚举一个选择样本的所有可能结果，计算出其检验的精确拒绝概率。

在三选项示例中，设原假设下的选择概率为 $p_1 = 0.5$、$p_2 = p_3 = 0.25$，样本量为 $N = 100$。

1. 以 $m = n_2 + n_3$ 为条件，$n_2$ 的分布是什么？利用它计算 $LR$、$LM$、$H_3$ 和 $H_2$ 统计量超过 $\chi^2_1$ 分布5%临界值的精确概率。将无穷大的统计量视为拒绝，并忽略 $m = 0$ 的样本（其概率为 $0.5^{100}$）。
2. 豪斯曼和麦克法登所用的备择假设是方案1和方案2共享同一分支的嵌套逻辑模型。设参数为 $\lambda$、$\beta = \log 2$，令 $a = \beta/\lambda$、$\Pi_1 = e^a/(e^a + 1)$、$p = (e^a + 1)^\lambda / (1 + (e^a + 1)^\lambda)$；则 $p_1 = \Pi_1 p$、$p_2 = (1 - \Pi_1) p$、$p_3 = 1 - p$，而 $\lambda = 1$ 时给出原假设下的各概率。对 $\lambda = 0.8$ 重复该计算，并将你的结果与他们的表III.3进行比较。
3. 将这四个检验之间的差异与 {numref}`fig-st-iia` 联系起来。
```

```{solution-start} st_ex_iia
:class: dropdown
```

下面给出一种解答。

给定 $m$，$n_2$ 服从以 $m$ 为试验次数、以 $p_2/(p_2 + p_3)$ 为成功概率的二项分布，而 $m$ 服从以 $N$ 为试验次数、以 $p_2 + p_3$ 为成功概率的二项分布。

这四个统计量都只依赖于 $m$ 和 $n_2$。

```{code-cell} ipython3
def iia_statistics(n2, m):
    """LR, LM, H3, and H2 for an array of values of n2, given m = n2 + n3."""
    n3 = m - n2
    with np.errstate(divide="ignore", invalid="ignore"):
        LR = 2 * (np.where(n2 > 0, n2 * np.log(2 * n2 / m), 0)
                  + np.where(n3 > 0, n3 * np.log(2 * n3 / m), 0))
        H3 = np.where(n2 > 0, np.log(2 * n2 / m)**2 * m, np.inf)
        H2 = np.where(n3 > 0, np.log(2 * n3 / m)**2 * m, np.inf)
    LM = (n3 - n2)**2 / m
    return np.array([LR, LM, H3, H2])


def rejection_probabilities(p2, p3, N=100, level=0.05):
    crit = chi2.ppf(1 - level, 1)
    probs = np.zeros(4)
    for m in range(1, N + 1):
        n2 = np.arange(m + 1)
        weights = binom.pmf(m, N, p2 + p3) * binom.pmf(n2, m, p2 / (p2 + p3))
        probs += (iia_statistics(n2, m) > crit) @ weights
    return probs


def nested_logit(λ, β=np.log(2)):
    """Choice probabilities with alternatives 1 and 2 in one nest."""
    a = β / λ
    Π_1 = np.exp(a) / (np.exp(a) + 1)
    p = (np.exp(a) + 1)**λ / (1 + (np.exp(a) + 1)**λ)
    return Π_1 * p, (1 - Π_1) * p, 1 - p


for λ in (1.0, 0.8):
    p1, p2, p3 = nested_logit(λ)
    print(f"λ = {λ}: p = ({p1:.4f}, {p2:.4f}, {p3:.4f});  LR, LM, H3, H2 reject "
          f"with probabilities {np.round(rejection_probabilities(p2, p3), 4)}")
```

在原假设下，精确显著性水平为：似然比检验为 $0.050$，拉格朗日乘数检验为 $0.049$，两个豪斯曼检验均为 $0.055$；$H_3$ 与 $H_2$ 显著性水平相同，是因为在原假设下 $n_2$ 和 $n_3$ 具有相同的分布。

豪斯曼和麦克法登的表III.3在 $\lambda = 1$ 时报告两个豪斯曼检验的值均为 $0.054$；精确枚举给出的值为 $0.055$，而他们在5%水平下关于 $\lambda < 1$ 的条目与精确枚举结果在五位小数内一致。

当 $\lambda = 0.8$ 时，检验力分别为 $0.135$、$0.134$、$0.193$ 和 $0.084$，与豪斯曼和麦克法登表III.3中的数值一致。

这一排序可以从 {numref}`fig-st-iia` 中看出。

在此备择假设下 $p_2 < p_3$，因此大多数样本满足 $s < \frac{1}{2}$。

当 $s < \frac{1}{2}$ 时，坐标 $\log s$ 下的二次近似位于 KL 散度上方，而坐标 $\log(1 - s)$ 下的二次近似位于其下方，因此 $H_3$ 大于 $LR$，而 $H_2$ 较小；坐标 $\log(1 - s)$ 下的二次近似也位于坐标 $s$ 下的二次近似之下，因此 $H_2$ 小于 $LM$。

在每个样本中都有 $LR \ge LM$，这是因为由平斯克不等式 $D(\text{Bern}(s) \| \text{Bern}(\frac{1}{2})) \ge 2(s - \frac{1}{2})^2$ 可知：坐标 $s$ 下的二次近似位于 KL 散度下方。

```{solution-end}
```

```{exercise}
:label: st_ex_power

频段比较检验的检验力取决于两个投影之间的距离。

1. 对于季节性方差占比 $0.005$、$0.01$ 和 $0.02$，计算伪真值 $\rho^*_{\text{full}}$ 和 $\rho^*_{\text{band}}$，即过程 $y_t = x_t + w_t$ 在使用全部频率以及删去季节性频段后的 AR(1) 投影。
2. 计算非中心参数 $\nu = T (\rho^*_{\text{band}} - \rho^*_{\text{full}})^2 / (V_{\text{band}} - V_{\text{full}})$，其中方差在 $\rho^*_{\text{full}}$ 处求值，以及渐近检验力 $P(\chi^2_1(\nu) > 3.84)$。
3. 将渐近检验力与仿真拒绝频率进行比较，并解释二者的差异。
```

```{solution-start} st_ex_power
:class: dropdown
```

下面给出一种解答。

我们在样本量为 $T$ 的傅里叶频率上计算这些投影，用 $y_t$ 的谱密度代替周期图。

```{code-cell} ipython3
def spectral_density_y(share):
    f_x = 1 / np.abs(1 - ρ_x * z_T)**2
    f_w = 1 / np.abs(1 + φ * z_T + φ**2 * z_T**2 + φ**3 * z_T**3)**2
    return f_x + share * var_x / var_w_unit * f_w


projections = {}
print("share   ρ*_full  ρ*_band     ν    asymptotic power   simulated power")
for share in (0.005, 0.01, 0.02):
    f_y = spectral_density_y(share)
    ρ_f, ρ_b = whittle_ar1(f_y, keep_all), whittle_ar1(f_y, keep_band)
    V_full = np.linalg.inv(ar1_information(ρ_f, keep_all))[0, 0]
    V_band = np.linalg.inv(ar1_information(ρ_f, keep_band))[0, 0]
    projections[share] = (ρ_f, ρ_b, V_full, V_band)
    nc = T_obs * (ρ_b - ρ_f)**2 / (V_band - V_full)
    power = ncx2.sf(crit_1, 1, nc)
    print(f"{share:5.3f} {ρ_f:8.3f} {ρ_b:8.3f} {nc:7.1f} {power:12.2f} "
          f"{simulated_power[share]:17.2f}")
```

渐近检验力随着两个投影之间平方距离的增大而增大。

在最小和最大占比处，它与仿真检验力接近，而在中间占比处则偏高。

非中心 $\chi^2$ 近似所使用的是原假设下两个估计量之差的方差。

下面的代码将该差的仿真均值和标准差，与两个投影之间的距离以及由原假设下方差所隐含的标准差进行了比较。

```{code-cell} ipython3
print("share   mean difference   distance   SD   null SD   ratio   "
      "corr(ρ_full, difference)")
for share in (0.005, 0.01, 0.02):
    ρ_f, ρ_b, V_full, V_band = projections[share]
    ρ_full, ρ_band = band_draws[share]
    difference = ρ_band - ρ_full
    sd_null = np.sqrt((V_band - V_full) / T_obs)
    print(f"{share:5.3f} {difference.mean():13.4f} {ρ_b - ρ_f:12.4f} "
          f"{difference.std():8.4f} {sd_null:7.4f} {difference.std() / sd_null:7.2f} "
          f"{np.corrcoef(ρ_full, difference)[0, 1]:12.2f}")
```

在每个占比下，仿真之差的均值都接近两个投影之间的距离，但其标准差约为原假设下方差所隐含值的1.5、2和3倍。

在这些备择假设下，AR(1) 模型是设定错误的，豪斯曼引理不再适用：$\hat\rho_{\text{full}}$ 与该差之间的仿真相关系数分别为 $-0.34$、$-0.50$ 和 $-0.67$。

全频段估计量使用了季节性峰值处的周期图纵坐标值，其方差与该处谱密度的平方成正比；而删去频段后的估计量则不使用这些值。

这些纵坐标值提高了 $\hat\rho_{\text{full}}$ 以及该差的方差，因此 $V_{\text{band}} - V_{\text{full}}$ 低估了该差的真实方差。

于是该检验统计量近似地等于某个非中心 $\chi^2$ 变量的某个倍数，其非中心参数比近似所用的值要小。

在占比最大时，两种检验力都接近1；而在占比最小时，较大的离散程度与较小的非中心参数大致相互抵消。

在中间占比处，较小的非中心参数占主导，近似高估了检验力。

豪斯曼和麦克法登的示例则显示出相反的误差：“当 $\lambda$ 小于.9时，所有检验的精确检验力都大于渐近近似所给出的预测值。”

一旦备择假设是非局部的，该近似可能向任一方向出错。

{doc}`approximation_errors` 中的公式 {eq}`ae_sandwich` 给出了每个估计量在备择假设下的渐近协方差矩阵，这使上述解释得以量化。

这两个估计量在频段之外共用相同的周期图纵坐标值，因此 $\sqrt{T}$ 乘以两个估计量误差之间的渐近协方差矩阵为 $\mathcal{H}^{-1} \mathcal{J}_{\text{cross}} \mathcal{H}_K^{-1}$，其中

$$
\mathcal{J}_{\text{cross}} = \frac{1}{4\pi} \int_K \nabla \log g_{\text{full}} \, \nabla \log g_{\text{band}}^\top \, \frac{f^2}{g_{\text{full}} \, g_{\text{band}}} \, d\omega ,
$$

$g_{\text{full}}$ 和 $g_{\text{band}}$ 分别是两个投影的谱密度，而 $\mathcal{H}$ 和 $\mathcal{H}_K$ 分别是 {eq}`ae_sandwich` 式中对应全部频率和频段之外频率的矩阵 $\mathcal{H}_K$，各自在其自身投影处求值。

下面的代码从这些矩阵出发，计算了该差的标准差与原假设标准差之比、$\hat\rho_{\text{full}}$ 与该差之间的相关系数，以及当该统计量由 $V_{\text{diff}} / (V_{\text{band}} - V_{\text{full}})$ 乘以一个非中心参数为 $T (\rho^*_{\text{band}} - \rho^*_{\text{full}})^2 / V_{\text{diff}}$ 的非中心 $\chi^2_1$ 变量来近似时所得的检验力，其中 $V_{\text{diff}}$ 是该差在备择假设下的渐近方差。

```{code-cell} ipython3
def ar1_derivatives(ρ, σ2):
    """∇ log g and ∇² log g at the Fourier frequencies, θ = (ρ, σ²)."""
    d1 = np.vstack([2 * (np.cos(ω_T) - ρ) / np.abs(1 - ρ * z_T)**2,
                    np.ones(T_obs - 1) / σ2])
    d2 = np.zeros((2, 2, T_obs - 1))
    d2[0, 0] = 2 * np.real(z_T**2 / (1 - ρ * z_T)**2)
    d2[1, 1] = -1 / σ2**2
    return d1, d2


def projection_parts(f, keep):
    """AR(1) projection over keep, the inverse of H, ∇ log g, and f/g on keep."""
    ρ = whittle_ar1(f, keep)
    σ2 = np.mean((f * np.abs(1 - ρ * z_T)**2)[keep])
    d1, d2 = ar1_derivatives(ρ, σ2)
    r = np.where(keep, f * np.abs(1 - ρ * z_T)**2 / σ2, 0.0)
    H = 0.5 * ((d1 * r) @ d1.T
               + (d2 * np.where(keep, 1 - r, 0.0)).sum(axis=2)) / T_obs
    return ρ, np.linalg.inv(H), d1, r


def score_covariance(d_a, r_a, d_b, r_b):
    return 0.5 * ((d_a * r_a * r_b) @ d_b.T) / T_obs


print("share   SD ratio   corr(ρ_full, difference)   power   simulated power")
for share in (0.005, 0.01, 0.02):
    f_y = spectral_density_y(share)
    ρ_f, H_f, d_f, r_f = projection_parts(f_y, keep_all)
    ρ_b, H_b, d_b, r_b = projection_parts(f_y, keep_band)
    V_f = (H_f @ score_covariance(d_f, r_f, d_f, r_f) @ H_f)[0, 0]
    V_b = (H_b @ score_covariance(d_b, r_b, d_b, r_b) @ H_b)[0, 0]
    C_fb = (H_f @ score_covariance(d_f, r_f, d_b, r_b) @ H_b)[0, 0]
    V_diff = V_f + V_b - 2 * C_fb
    _, _, V_full, V_band = projections[share]
    V_null = V_band - V_full
    power = ncx2.sf(crit_1 * V_null / V_diff, 1, T_obs * (ρ_b - ρ_f)**2 / V_diff)
    print(f"{share:5.3f} {np.sqrt(V_diff / V_null):9.2f} "
          f"{(C_fb - V_f) / np.sqrt(V_f * V_diff):19.2f} {power:15.2f} "
          f"{simulated_power[share]:12.2f}")
```

备择假设下的协方差矩阵给出的比值为 $1.42$、$1.96$ 和 $2.96$，相关系数为 $-0.34$、$-0.51$ 和 $-0.68$，与仿真得到的 $1.48$、$2.02$、$3.01$ 以及 $-0.34$、$-0.50$、$-0.67$ 接近。

利用这些值所得的检验力近似给出 $0.47$、$0.77$ 和 $0.92$，而仿真检验力分别为 $0.43$、$0.78$ 和 $0.98$：它纠正了中间占比处的误差，在最小占比处比原假设近似稍微偏离仿真结果更远，而在最大占比处则低估了检验力。

```{solution-end}
```