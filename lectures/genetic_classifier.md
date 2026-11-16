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
  title: 网络、遗传算法与分类器系统
  headings:
    Overview: 概览
    'The perceptron: an agent as a discriminant function': 感知机：作为判别函数的主体
    'Associative memory: the Hopfield network': 联想记忆：霍普菲尔德网络
    The genetic algorithm: 遗传算法
    The genetic algorithm::Axelrod's iterated prisoner's dilemma: 阿克塞尔罗德的重复囚徒困境
    Classifier systems: 分类器系统
    Classifier systems::Matching, specificity, and bids: 匹配、特异性与出价
    Classifier systems::The auction: 拍卖
    Classifier systems::A two-armed bandit: 双臂老虎机
    Classifier systems::The bucket brigade: 桶链
    Evolutionary programming: 演化编程
    Concluding remarks: 结束语
    Exercises: 练习
---

(genetic_classifier)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 网络、遗传算法与分类器系统

```{index} single: Bounded Rationality; Genetic Algorithms
```

```{contents} Contents
:depth: 2
```

## 概览

到目前为止的讲座给适应性主体赋予了相当计量经济学化的大脑。

在 {doc}`olg_adaptive_money` 中，它们运行随机近似和递归最小二乘法；在 {doc}`learning_approximation` 中，它们拟合条件期望；在 {doc}`exchange_rate_learning` 中，它们针对实现效用采取了牛顿步骤。

这些学习者中的每一个都被告知要估计什么（一个储蓄率、一个条件期望、一个投资组合），并通过 {ref}`br_one_recursion` 中描述的那种递归来更新其估计。

{doc}`exchange_rate_learning` 还初步展示了另一类学习者：阿里福维奇（Arifovic）的遗传算法经济体（{ref}`xr_ga`），其汇率从未趋于稳定。

本讲座将系统阐述这一算法及其相关方法。

它们构成了 {cite:t}`Sargent1993` 所汲取的工具箱，来源于约翰·霍兰德（John Holland）关于适应性系统的研究 {cite}`Holland1975,HollandHolyoakNisbettThagard1986`，以及神经网络的连接主义文献。

萨金特的论述框架是，有限理性研究纲领要求我们让模型中充满"表现得像计量经济学家"的主体，而人工智能文献则是这些主体候选大脑的目录：

> 正是从这些文献所积累的方法储备中，我们将挑选出赋予有限理性主体的"大脑"。

我们将巡览其中四种：

1. **感知机**（perceptron），最简单的神经网络，它其实是一个线性判别函数，与"主体作为计量经济学家"这一解读直接相关；
1. **霍普菲尔德网络**（Hopfield network），一种联想记忆机制，通过在能量函数上向下滚动，从受损输入中恢复存储的模式；
1. **遗传算法**（genetic algorithm），霍兰德基于种群的搜索方法，我们将其应用于阿克塞尔罗德的重复囚徒困境博弈；
1. **分类器系统**（classifier system），霍兰德提出的"大脑即竞争性经济体"的if-then规则集合，其最简单的实例——一个双臂老虎机——已经暴露出一个微妙的局限性，而其**桶链算法**（bucket brigade）则沿着决策链向后传递信用。

前两者是与此前内容的衔接桥梁。

感知机是计量经济学家一眼就能认出的参数化规则，而霍普菲尔德网络与其说是规则，不如说是一种记忆。

后两者则带来了新的东西：它们不是调整固定规则的系数，而是**发现规则**本身，仅凭经验作为指引，在一个庞大的可能性空间中进行搜索。

我们以**演化编程**（evolutionary programming）作为结尾：即一个适应性主体种群，在缓慢趋向均衡的过程中，可被用来*计算*我们无法直接求解的均衡。

这正是 {doc}`marimon_mcgrattan_sargent` 完整实现的思路，本讲座构建了它所使用的机制基础。

让我们从一些导入开始。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## 感知机：作为判别函数的主体

最简单的神经网络是单个**感知机**：$k$ 个输入 $x_i$、权重 $w_i$，以及一个输出

$$
y = S\!\left(\sum_{i=1}^k w_i x_i\right) = S(w^\top x),
$$

其中 $S$ 是一个将 $\mathbb{R}$ 映射到 $[0, 1]$ 的"压缩函数"：可以是阶跃函数，也可以是S型函数
$S(z) = 1/(1 + e^{-z})$，或任何累积分布函数。

对于固定的权重，感知机是一个**分类器**：当 $w^\top x > 0$ 时它被激活（$y = 1$），否则保持静默，因此边界 $w^\top x = 0$ 是一个分隔两个类别的超平面。

训练感知机，即选择 $w$ 以最小化 $\sum_t (y_t - S(w^\top x_t))^2$，是一个非线性最小二乘问题，通过随机梯度递归求解：

$$
w_t = w_{t-1} + \gamma_t \nabla S(w_{t-1}, x_t)\,(y_t - S(w_{t-1}^\top x_t)),
$$

其中 $\nabla S(w, x)$ 是 $S(w^\top x)$ 关于 $w$ 的梯度，对于S型函数而言等于
$S(w^\top x)(1 - S(w^\top x))\,x$。

这具有 $\theta_t = \theta_{t-1} + \gamma_t Q(\theta_{t-1}, x_t)$ 的形式，这是本系列中每个学习者共享的形式（{ref}`br_one_recursion`）。

书中提出了一个简单的方案：将 $\gamma_t$ 设为一个小常数，并反复让样本通过该递归，直到权重稳定下来。

以书中的例子为例：根据体重和薪水这两个特征（任意单位的合成数据）区分足球运动员和经济学家。

```{code-cell} ipython3
rng = np.random.default_rng(0)
n = 100
economists = rng.multivariate_normal([-1.0, -0.3], [[0.5, 0.1], [0.1, 0.5]], n)
players    = rng.multivariate_normal([1.2,  0.8], [[0.6, 0.0], [0.0, 0.6]], n)
X = np.vstack([economists, players])
y = np.r_[np.zeros(n), np.ones(n)]
X_aug = np.column_stack([np.ones(len(X)), X])          # prepend an intercept

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_perceptron(X_aug, y, γ=0.1, passes=200, nls=True, seed=1):
    "Constant-gain stochastic gradient, run repeatedly through the sample."
    rng = np.random.default_rng(seed)
    w = np.zeros(X_aug.shape[1])
    for _ in range(passes):
        for t in rng.permutation(len(y)):
            S = sigmoid(X_aug[t] @ w)
            grad = S * (1 - S) * X_aug[t] if nls else X_aug[t]
            w += γ * grad * (y[t] - S)
    return w

w = train_perceptron(X_aug, y)                         # the recursion displayed above
print(f"training accuracy: {np.mean((sigmoid(X_aug @ w) > 0.5) == y):.3f}")
```

`nls=False` 分支去掉了因子 $S(1 - S)$；我们稍后会用到它。

{cite:t}`Sargent1993` 强调的要点是，这对计量经济学家来说并非什么异类事物：感知机的决策边界本质上是一个**线性判别函数**。

将感知机的边界法向量与费舍尔线性判别方向进行比较。

```{code-cell} ipython3
def unit(v):
    return v / np.linalg.norm(v)

def angle(u, v):
    "Angle in degrees between two directions, ignoring sign."
    return np.degrees(np.arccos(min(1.0, abs(unit(u) @ unit(v)))))

m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
S_within = np.cov(X[y == 0].T) * n + np.cov(X[y == 1].T) * n
lda_direction = unit(np.linalg.solve(S_within, m1 - m0))
perceptron_direction = unit(w[1:])

print(f"perceptron boundary normal   : {perceptron_direction.round(3)}")
print(f"Fisher discriminant direction: {lda_direction.round(3)}")
print(f"cosine similarity: {abs(perceptron_direction @ lda_direction):.4f}, "
      f"angle {angle(w[1:], lda_direction):.1f} degrees")
```

从递归中去掉因子 $S(1 - S)$，会将其变为逻辑斯蒂对数似然函数上的随机梯度上升，也就是**逻辑斯蒂回归**。

```{code-cell} ipython3
w_logit = train_perceptron(X_aug, y, nls=False)
print(f"logistic regression: cosine {abs(unit(w_logit[1:]) @ lda_direction):.4f}, "
      f"angle {angle(w_logit[1:], lda_direction):.1f} degrees")
```

下面是感知机的边界。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Perceptron boundary separating the two groups"
    name: fig-gc-perceptron
---
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(*economists.T, s=18, c='C0', label="economists ($y=0$)")
ax.scatter(*players.T, s=18, c='C1', label="football players ($y=1$)")
xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
ax.plot(xs, -(w[0] + w[1]*xs) / w[2], 'k-', lw=1.5, label="perceptron boundary")
ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
ax.set_xlabel("weight")
ax.set_ylabel("salary")
ax.legend(frameon=False)
plt.show()
```

这两条边界都非常接近费舍尔方向：感知机的方向大约偏离 11.5 度，而逻辑斯蒂回归的方向大约偏离 4 度。

这种接近并非偶然。

如果这两个类别是具有共同协方差矩阵 $\Sigma$ 的高斯分布，那么 $\Pr(y = 1 \mid x)$ 恰好是 $x$ 的一个S型函数，其斜率为 $\Sigma^{-1}(\mu_1 - \mu_0)$，即费舍尔方向，因此两种递归都指向该方向。

这里两个协方差矩阵略有不同，且样本是有限的，而最小二乘准则对观测值的加权方式与似然函数不同，因此两个方向有所差异。

明斯基和帕佩特著名的批评 {cite:p}`MinskyPapert1969` 恰恰指出，单个感知机*只能*表示线性判别函数，因此无法分离一条直线无法分开的类别。

该领域的复兴源于人们认识到，**分层**感知机，即将其输出馈送到进一步的感知机中，可以逼近任意非线性判别函数，这正是 {doc}`back_prop` 所讨论的主题。

就我们的目的而言，感知机是这个目录中与前几讲已经做过的事情最接近的条目：一个参数化规则，通过随机梯度递归训练，计量经济学家一眼就能认出来。

霍普菲尔德网络是另一种桥梁，而最后两种大脑则更加陌生。

## 联想记忆：霍普菲尔德网络

第二种装置存储模式并从受损碎片中回忆它们。

将一个模式表示为长度为 $N$ 的 $\pm 1$ 向量。

我们希望存储 $p$ 个模式 $\sigma^1, \ldots, \sigma^p$，即一个 $N \times p$ 矩阵 $\sigma$ 的各列，使每个模式都是某个动态系统的不动点，从而输入一个受损版本能迅速收敛到最近的存储模式。

**霍普菲尔德网络**（Hopfield network） {cite}`Hopfield1982` 通过以下动态实现这一点：

$$
s(t+1) = \operatorname{sgn}(w\, s(t)),
$$

以及一个由模式本身构建的权重矩阵。

当模式正交时，赫布规则（Hebb's rule）$w = \tfrac{1}{N}\sigma\sigma^\top$ 就足够了；对于仅仅线性独立（相关的）模式，则使用投影规则 $w = \tfrac{1}{N}\sigma V^{-1}\sigma^\top$，其中 $V = \tfrac{1}{N}\sigma^\top \sigma$。

两者都能使每个存储的模式成为精确的不动点。

投影规则具有简单的几何意义：$w = \sigma(\sigma^\top\sigma)^{-1}\sigma^\top$ 是到存储模式张成子空间上的正交投影。

因此能量 $E(s) = -\tfrac{1}{2}s^\top w s$ 等于 $-\tfrac{1}{2}\lVert w s\rVert^2$，其值至少为 $-\tfrac{1}{2}\lVert s\rVert^2 = -N/2$，当且仅当 $s$ 位于该子空间中时等号成立。

因此，存储的模式是能量的*全局*最小值。

我们在 $5\times5$ 像素网格上存储五个字母，由于字母共享许多像素，它们是相关的，因此投影规则才是正确的选择。

（代码将模式存储为 `σ` 的行，即公式中矩阵的转置。）

```{code-cell} ipython3
PATTERNS = {
    'A': ["01110", "10001", "11111", "10001", "10001"],
    'E': ["11111", "10000", "11110", "10000", "11111"],
    'I': ["11111", "00100", "00100", "00100", "11111"],
    'O': ["01110", "10001", "10001", "10001", "01110"],
    'T': ["11111", "00100", "00100", "00100", "00100"],
}

def to_vector(rows):
    return np.array([1 if c == '1' else -1 for r in rows for c in r])

letters = list(PATTERNS)
σ = np.array([to_vector(PATTERNS[c]) for c in letters])      # (p, N): one pattern per row

def projection_rule(patterns):
    "Weight matrix making each (correlated) pattern an exact fixed point."
    N = patterns.shape[1]
    Σ = patterns.T                                          # (N, p)
    V = Σ.T @ Σ / N
    return Σ @ np.linalg.inv(V) @ Σ.T / N

def energy(w, s):
    "Hopfield energy; stored patterns are global minima."
    return -0.5 * s @ w @ s

def recall(w, s0, max_iter=30):
    "Iterate sgn(w s) to a fixed point."
    s = s0.copy()
    for _ in range(max_iter):
        s_new = np.sign(w @ s)
        s_new[s_new == 0] = 1
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s

w_hop = projection_rule(σ)
fixed = all(np.array_equal(np.sign(w_hop @ σ[i]), σ[i]) for i in range(len(letters)))
print(f"all stored letters are fixed points: {fixed}")
print(f"energy of each stored letter: {[round(energy(w_hop, σ[i]), 1) for i in range(len(letters))]}")
```

每个存储的字母都处于相同的能量 $-N/2$ 处，且都是不动点。

现在破坏几个像素，让网络回落到最近的记忆。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Hopfield recall of letters from corrupted inputs"
    name: fig-gc-hopfield
---
rng = np.random.default_rng(3)
show = ['E', 'O', 'T']

fig, axes = plt.subplots(len(show), 3, figsize=(6, 6))
for row, letter in enumerate(show):
    i = letters.index(letter)
    corrupt = σ[i].copy()
    corrupt[rng.choice(25, 4, replace=False)] *= -1             # flip 4 pixels
    recovered = recall(w_hop, corrupt)
    for col, (img, title) in enumerate([(σ[i], "stored"),
                                        (corrupt, "corrupted"),
                                        (recovered, "recovered")]):
        axes[row, col].imshow(img.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(title)
plt.tight_layout()
plt.show()
```

受损的字母恢复到了它们的原始状态。

恢复的可靠程度取决于损坏了多少内容以及模式之间的相关程度。

一个受损的字母偶尔会落入*另一个*字母的吸引域，或落入设计者从未存储过的虚假混合状态中。

```{code-cell} ipython3
for n_flip in (3, 5, 7):
    correct = 0
    for i in range(len(letters)):
        for _ in range(40):
            corrupt = σ[i].copy()
            corrupt[rng.choice(25, n_flip, replace=False)] *= -1
            if np.array_equal(recall(w_hop, corrupt), σ[i]):
                correct += 1
    print(f"{n_flip} pixels flipped: recovered exactly {correct}/{len(letters)*40}")
```

在 25 个像素上存储五个模式是一个较重的负荷（$p/N = 0.2$），且字母之间重叠很多：I 和 T 仅在 25 个像素中的 4 个像素上不同。

这就是为什么即使是 3 个像素的损坏，也大约有十分之一的情况会失败。

书中对回忆的分析假设模式数量远少于像素数量，即 $p \ll N$。

这个机制值得点名，因为它在整个研究方案中反复出现。

回忆是**在能量函数上的下降**，其最小值包括存储的模式。

输入一个受损的模式会将系统置于能量表面上靠近正确最小值的位置，而动态过程会将其滚动下降。

这正是**模拟退火**（simulated annealing）背后的几何原理，书中提到这一方法用于逃离*不需要的*局部最小值：加入一定量可控的随机"扰动"，一开始较大，随时间递减，从而使系统能够跳出浅层虚假吸引域，最终落入一个深层吸引域。

退火法和随机牛顿法对牛顿法进行了修改，以改善其全局行为，同时不放弃其良好的局部行为。

早期多做实验、后期少做实验的同一原理，支配着 {doc}`marimon_mcgrattan_sargent` 中的分类器系统——其遗传算法以按 $1/\sqrt{t}$ 递减的概率触发——也支配着 {ref}`gc_ex3` 中的 softmax 拍卖，其"温度"$1/\beta$ 扮演着扰动的角色。

(gc_ga)=
## 遗传算法

第三种大脑与牛顿法的背离更为根本：它根本不在光滑表面上下降。

霍兰德的**遗传算法**（genetic algorithm）{cite}`Holland1975,Goldberg1989` 通过演化一个编码为二进制字符串的候选解**种群**（population），在缺乏牛顿法所需光滑性的"崎岖景观"中进行搜索。

这两种方法是可以结合使用的。

埃伦·麦格拉坦（Ellen McGrattan）曾将遗传算法用于估计非线性理性预期模型：她先用遗传算法搜索最大值附近的区域，然后切换到牛顿法 {cite:p}`Sargent1993`。

该算法被赋予两样东西：一个待最大化的非负适应度函数 $f$，以及一个将每个长度为 $S$ 的二进制字符串映射为候选解 $x$ 的*编码*。

给定一个由 $N$ 个字符串组成的种群，每一代应用四个算子：

1. **评估**：计算每个字符串的适应度 $f(x_i)$。
1. **繁殖**：将字符串以概率 $f(x_i)/\sum_j f(x_j)$ 复制到下一代，字符串 $i$ 的复制概率与适应度成正比（"有偏轮盘赌"）。
1. **交叉**：将字符串配对，并在随机切割点交换它们的尾部，形成重组了父代片段的子代。
1. **变异**：以较小的概率 $p_m$ 独立地翻转每个位。

该算法的参数是 $N$、$S$ 和 $p_m$。

下面的代码使用了一种常见的繁殖变体，称为*窗口化*（windowing）：它在转动轮盘之前先减去该代中最低的适应度。

当所有适应度都彼此接近时（下文情形正是如此），朴素的比例选择几乎无法区分各字符串；窗口化能使其区分度更强，并使最差的字符串几乎没有繁殖的机会。

繁殖将种群集中于已经奏效的方案；交叉和变异注入新的候选方案。

霍兰德对交叉为何有效的解释是用**模式**（schemata）来说明的。

一个模式 $H$ 是 $\{0, 1, *\}^S$ 中的一个模板，其中 $*$ 可以匹配任一位：`1**0*` 表示长度为5、第一位为1、第四位为0的一组字符串。

它的*阶*（order）$o(H)$ 是固定位置的数量，它的*定义长度*（defining length）$\delta(H)$ 是其中第一个和最后一个固定位置之间的距离。

霍兰德的**模式定理**（schema theorem）指出，在按适应度比例进行繁殖、对每一对字符串应用单点交叉、并以速率 $p_m$ 进行变异的情况下，模式 $H$ 中字符串的期望数量满足

$$
\mathbb{E}\, m(H, t+1) \ \ge\ m(H, t)\,\frac{f(H)}{\bar f}
\left[1 - \frac{\delta(H)}{S - 1} - o(H)\, p_m\right],
$$

其中 $f(H)$ 是 $H$ 中字符串的平均适应度，$\bar f$ 是整个种群的平均适应度（若采用窗口化，则 $f$ 应理解为窗口化后的适应度）。

因此，适应度高于平均水平、长度短且阶数低的模式往往会增殖。

该界限只针对一代而言。

"构建块假说"（building-block hypothesis）——即该算法通过将短小、适应度高的模式组装成更优字符串来发挥作用——是一种启发式说法，而非定理。

它是否适用，用该书中的话说，"恰恰取决于给定问题的编码方式"：取决于协同作用良好的位是否彼此相邻。

模式中的 $*$ 在下文中以分类器条件中的 $\#$ 重新出现。

关键的是，个体字符串本身不学习：每个只存活一代便消亡。

只有*社会*，即种群的序列，才在学习。

（在 {doc}`exchange_rate_learning` 中，阿里福维奇的选择算子使这一点有所缓和：除非子代会表现更好，否则父代便得以存活，因此一个优良字符串可以存活很多代。）

这使得遗传算法作为单个大脑的模型显得别扭，而更适合作为种群或市场的模型——这一点在我们讨论到 {doc}`marimon_mcgrattan_sargent` 时至关重要。

### 阿克塞尔罗德的重复囚徒困境

{cite:t}`Axelrod1987` 使用遗传算法演化**重复囚徒困境**（iterated prisoner's dilemma）博弈的策略，该博弈的循环赛计算机锦标赛著名地由针锋相对（tit-for-tat）策略获胜。

他将策略针对参加其第二届锦标赛的提交策略中具有代表性的八个策略进行培育；而我们在此使用一个更小的对手小组。

两个玩家各自选择合作（Cooperate）或背叛（Defect）；收益结构奖励对合作者的背叛，但惩罚双方互相背叛。

```{code-cell} ipython3
# 1 = Cooperate, 0 = Defect;  T > R > P > S is the prisoner's dilemma ranking
T_pay, R_pay, P_pay, S_pay = 5, 3, 1, 0
PAYOFF = np.array([[P_pay, T_pay],       # row: my move (D, C); column: opponent's (D, C)
                   [S_pay, R_pay]])
```

按照阿克塞尔罗德的方法，策略是一个**70位字符串**：一个策略基于最后三轮的结果作出条件反应。

每轮都是四种结果之一（我的行动、对手的行动），因此三轮会产生 $4^3 = 64$ 种可能的历史，64位指定了每种历史下的行动。

剩余的6位编码了一个假定的博弈前历史，以启动最初的行动。

我们的 `play` 函数可以一次性对整个字符串种群进行博弈，每行对应一个字符串。

```{code-cell} ipython3
POW = 2 ** np.arange(5, -1, -1)          # reads 6 history bits as an integer 0..63

def play(pop, opponent, n_rounds=150):
    "Play every 70-bit string in pop against opponent; return average payoffs."
    pop = np.atleast_2d(pop)
    N = len(pop)
    action = pop[:, :64]
    idx = pop[:, 64:] @ POW                       # the presumed pre-game history
    mine = np.zeros((N, n_rounds), int)
    theirs = np.zeros((N, n_rounds), int)
    my_total, opp_total = np.zeros(N), np.zeros(N)
    for t in range(n_rounds):
        a = action[np.arange(N), idx]             # my move from the lookup table
        b = opponent(mine[:, :t], theirs[:, :t], t)
        my_total += PAYOFF[a, b]
        opp_total += PAYOFF[b, a]
        mine[:, t], theirs[:, t] = a, b
        idx = 32 * a + 16 * b + (idx >> 2)        # newest round in, oldest round out
    return my_total / n_rounds, opp_total / n_rounds
```

该策略经过培育以便在固定的对手小组中表现出色。

每个小组成员都是历史*就该成员所见*的函数：其第一个参数是对手所打的历史，第二个参数是自己所打的历史。

Random（随机策略）成员每一轮都抛一枚均匀硬币。

为了保证比较的公平性，每一代都抽取一个硬币序列，种群中的每个字符串都面对同一个序列。

这种手法称为**公共随机数**（common random numbers），它使得同一代中适应度的差异反映的是策略本身的差异，而非抽签运气的差异。

```{code-cell} ipython3
def all_cooperate(opp_moves, own_moves, t): return np.ones(len(opp_moves), int)
def all_defect(opp_moves, own_moves, t):    return np.zeros(len(opp_moves), int)

def tit_for_tat(opp_moves, own_moves, t):
    return opp_moves[:, -1] if t > 0 else np.ones(len(opp_moves), int)

def grudger(opp_moves, own_moves, t):
    return opp_moves.all(axis=1).astype(int)     # cooperate until the first defection

def random_opponent(flips):
    "An opponent who plays coin flips drawn in advance: row k of flips in game k."
    return lambda opp_moves, own_moves, t: flips[:, t]

PANEL = {'AllC': all_cooperate, 'AllD': all_defect,
         'TFT': tit_for_tat, 'Grudger': grudger}

def fitness(pop, flips):
    "Average payoff of each string against the panel, Random included."
    opponents = list(PANEL.values()) + [random_opponent(flips)]
    return np.mean([play(pop, opp)[0] for opp in opponents], axis=0)
```

针锋相对策略复制其对手最后一次的行动，而"记仇者"（grudger）一旦对手曾经背叛过一次，便永远背叛下去。

这两者本身也都是70位字符串。

针锋相对策略的表格复制对手最近一次的行动。

"记仇者"的表格仅在连续三轮相互合作之后才合作，而从一个合作的博弈前历史出发，这等价于从未见过背叛。

我们将这两个字符串作为基准，并通过每个字符串对每个小组成员的收益来对其评分，针对Random（随机策略）则在400场新的随机对局中取平均值（用于比较的每个字符串都使用同样的400场对局）。

```{code-cell} ipython3
TFT_GENE = np.r_[(np.arange(64) >> 4) & 1, np.ones(6, int)]
GRUDGER_GENE = np.r_[(np.arange(64) == 63).astype(int), np.ones(6, int)]

def panel_scores(gene, reps=400, seed=1):
    "Payoff against each panel member; Random is averaged over reps fresh games."
    flips = np.random.default_rng(seed).integers(0, 2, (reps, 150))
    scores = {name: play(gene, opp)[0][0] for name, opp in PANEL.items()}
    scores['Random'] = play(np.tile(gene, (reps, 1)), random_opponent(flips))[0].mean()
    return scores

def panel_fitness(gene):
    return np.mean(list(panel_scores(gene).values()))

def first_move(gene):
    return 'C' if gene[:64][gene[64:] @ POW] else 'D'
```

现在开始演化。

我们展示种子为16的一次运行，之所以选择该种子，是因为其胜出者的适应度接近于下文所考察的二十次运行的中位数。

```{code-cell} ipython3
def evolve(N=60, generations=80, p_mut=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (N, 70))
    best_hist, mean_hist = [], []
    for _ in range(generations):
        flips = rng.integers(0, 2, (1, 150))           # one coin sequence for everyone
        fit = fitness(pop, flips)
        best_hist.append(fit.max())
        mean_hist.append(fit.mean())
        weight = fit - fit.min() + 1e-6                # windowing
        cum = np.cumsum(weight)
        parents = pop[np.searchsorted(cum, rng.random(N) * cum[-1])]
        mom, dad = parents[0::2], parents[1::2]
        left = np.arange(70) < rng.integers(1, 70, (N // 2, 1))   # single-point cuts
        pop = np.vstack([np.where(left, mom, dad), np.where(left, dad, mom)])
        pop[rng.random(pop.shape) < p_mut] ^= 1        # mutation
    fit = fitness(pop, rng.integers(0, 2, (1, 150)))
    return pop[fit.argmax()], np.array(best_hist), np.array(mean_hist)


champion, best_hist, mean_hist = evolve(seed=16)
print(f"generation 0:  best {best_hist[0]:.3f}, mean {mean_hist[0]:.3f}")
print(f"generation {len(best_hist)-1}: best {best_hist[-1]:.3f}, mean {mean_hist[-1]:.3f}")
```

下图追踪了每一代内部的最佳适应度和平均适应度。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Population fitness across generations"
    name: fig-gc-fitness
---
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(best_hist, label="best in population", lw=2)
ax.plot(mean_hist, label="population mean", lw=2)
ax.set_xlabel("generation")
ax.set_ylabel("panel fitness")
ax.legend(frameon=False)
plt.show()
```

种群的平均适应度在大约前十几代内迅速攀升，随后在2.4附近波动，这是因为交叉和变异不断产生较弱的字符串，从而将其压低。

最佳字符串的适应度则波动不定，并缓慢上升，到最后约达到2.6。

它之所以波动，是因为每一代都面对一个来自Random的全新硬币序列，也因为并没有任何机制能保证最佳字符串在繁殖、交叉与变异之后得以存活。

演化究竟发现了什么样的策略？

```{code-cell} ipython3
strategies = {'champion': champion, 'TFT': TFT_GENE, 'Grudger': GRUDGER_GENE}
table = {name: panel_scores(gene) for name, gene in strategies.items()}

print(f"{'payoff vs':>10}" + "".join(f"{name:>10}" for name in strategies))
for opp in table['champion']:
    print(f"{opp:>10}" + "".join(f"{table[name][opp]:>10.3f}" for name in strategies))
print(f"{'panel avg':>10}" + "".join(f"{np.mean(list(table[name].values())):>10.3f}"
                                     for name in strategies))
print(f"champion's first move: {first_move(champion)}")
```

用阿克塞尔罗德的话说，演化得到的策略是*友善*的：对于AllC（全合作者）、TFT（针锋相对）或"记仇者"，它从不率先背叛，因此对这三者都能获得3分的相互合作收益。

在最初两轮之后，它拒绝被AllD（全背叛者）欺骗，并且对Random（随机策略）的背叛频率足以使其获得远高于针锋相对策略的收益。

在这个对手小组面前，它胜过针锋相对策略，而且差距几乎完全来自对Random的表现。

但它并未胜过"记仇者"——这也是对手小组的一员——后者表现得还略好一些。

这次运行是否具有代表性？

下一个单元格用二十个不同的种子重复了这个实验。

```{code-cell} ipython3
results = []
for seed in range(20):
    sc = panel_scores(evolve(seed=seed)[0])
    nice = all(sc[k] == 3.0 for k in ('AllC', 'TFT', 'Grudger'))
    results.append((np.mean(list(sc.values())), nice))
fits = np.array([r[0] for r in results])
nice = np.array([r[1] for r in results])
tft_fit, grudger_fit = panel_fitness(TFT_GENE), panel_fitness(GRUDGER_GENE)

print(f"nice champions: {nice.sum()} of 20")
print(f"champion fitness: median {np.median(fits):.3f}, "
      f"range [{fits.min():.3f}, {fits.max():.3f}]")
print(f"  nice ones: mean {fits[nice].mean():.3f};  others: mean {fits[~nice].mean():.3f}")
print(f"beat TFT ({tft_fit:.3f}): {np.sum(fits > tft_fit)} of 20")
print(f"within 0.01 of Grudger ({grudger_fit:.3f}): {np.sum(fits > grudger_fit - 0.01)} of 20;"
      f"  above it: {np.sum(fits > grudger_fit)}")
```

大多数胜出者是友善的，但并非全部如此。

其余的则是*试探并利用*（probe-and-exploit）型策略：它们在某一时刻会无端背叛，这样做每轮能从AllC身上赢得5分，但会使其与"记仇者"之间陷入永久的相互背叛。

这个对手小组使这两类策略几乎势均力敌。

AllC和"记仇者"在有人背叛之前表现完全相同，因此若不进行试探，一个策略无法分辨两者。

一次试探能从AllC身上每轮多赚2分，却会在"记仇者"身上每轮损失2分，因此这两类策略在面对这对对手时打成平手。

该算法最终落入哪一类，取决于种子：这是一个崎岖景观的小型实例，其中有两座高度几乎相同的峰。

该对手小组本身也存在一个上限。

任何策略对AllD的平均得分都不可能超过1，对Random的平均得分都不可能超过3（面对一枚硬币，无论其结果如何，背叛总是更优），对针锋相对策略的平均得分都不可能超过约3，对AllC和"记仇者"二者合计的得分都不可能超过约6。

因此小组整体适应度不可能超过约 $13/5 = 2.6$，而"记仇者"本身已经与这一上限相差不到0.01。

每一个胜出者都胜过针锋相对策略，但没有一个能以超出舍入误差的幅度胜过"记仇者"。

正如 {cite:t}`Sargent1993` 所总结的那样，阿克塞尔罗德自己的遗传算法"产生了一个能够赢得该锦标赛的策略，尤其是能够胜过赢得该锦标赛的'针锋相对'策略"。

我们这个简化版本再现了这句话的后半部分，并揭示了其优势究竟来自何处。

这并非来自额外的记忆能力："记仇者"仅需一个比特的记忆，表现却不逊于任何胜出者。

真正的关键在于拒绝宽恕：面对Random时，针锋相对策略不断回归合作，结果被反复欺骗，而"记仇者"与最优胜出者则会转而坚持背叛。

而且，正如萨金特所指出的那样，这样的比较其实有利于遗传算法，因为它"得以事先针对"其对手进行练习，而提交参加锦标赛的策略却只能猜测自己将面对什么样的对手。

(gc_classifier)=
## 分类器系统

遗传算法演化的是一个其中没有个体学习的种群。

霍兰德的**分类器系统**（classifier system）将同样的演化机制置于*单个主体内部*，作为单一大脑的模型 {cite}`HollandHolyoakNisbettThagard1986`。

萨金特将其描述为霍兰德关于心智作为**竞争性经济体**（competitive economy）的构想：

> 各条语句相互竞争决策的机会。分类器系统以霍兰德称之为竞争性经济体的方式，将遗传算法的要素与其他方面结合起来代表一个大脑。

分类器系统包括五个部分：

* **分类器**（classifiers），即以三元字母表 $\{0, 1, \#\}$ 编码的if-then规则，其条件部分与状态相匹配，动作部分规定一个行动。
* **解码器**（decoder），给定当前状态，找出哪些分类器的条件是匹配的。
* **拍卖**（auction），选择一个匹配的分类器来执行行动。
* **计账系统**（accounting system），根据规则的决策所带来的收益来更新每个分类器的**强度**（strength），并在序贯问题中，沿着产生这些收益的规则链将收益*反向*传递。
* **创造新分类器的装置**，使系统的词汇本身也在演化：
  * 当没有分类器匹配某个状态时，就创建一个条件与之匹配、动作随机的分类器；
  * 新规则不时会取代弱规则，以便未尝试过的动作得以尝试；
  * 规则会被*泛化*（将0和1替换为 $\#$）或*特化*（将 $\#$ 替换为0和1），这常常通过遗传算子完成。

我们先讨论解码器和拍卖，再通过两个例子——老虎机和桶链——来讨论计账系统。

### 匹配、特异性与出价

如果一个状态与某个条件在所有非 $\#$（即"我不在乎"符号）的位置上都一致，则该状态与该条件相匹配。

条件 `1#0` 匹配状态 `100` 和 `110`，不匹配其他任何状态。

含有许多 $\#$ 的规则是*通用的*（general）：其条件经常被满足。

含有较少 $\#$ 的规则是*特定的*（specific）。

{cite:t}`MarimonMcGrattanSargent1990` 用 $\sigma_e = 1/(1 + e \text{ 中 } \#\text{ 的个数})$ 来衡量分类器 $e$ 的特异性。

```{code-cell} ipython3
def matches(condition, state):
    "A state matches if it agrees with every position that is not '#'."
    return all(c in ('#', s) for c, s in zip(condition, state))

def specificity(condition):
    return 1 / (1 + condition.count('#'))

state = '110'
for condition in ('110', '1#0', '##0', '###', '0##'):
    print(f"condition {condition}: matches {state}? {str(matches(condition, state)):5}  "
          f"specificity {specificity(condition):.2f}")
```

通常会有多条规则同时匹配，它们对该做什么可能意见不一。

每条匹配的规则 $e$ 出价为其强度 $S_e$ 的一个比例 $b(e)$，该比例随特异性而上升：

$$
B_e = b(e)\, S_e, \qquad b(e) = b_{11} + b_{12}\, \sigma_e ,
$$

其中，按照 {cite:t}`MarimonMcGrattanSargent1990` 的记号，$b_{11}$ 和 $b_{12}$ 是和小于一的正常数。

当一条通用规则和一条特定规则强度相等时，特定规则的出价更高，因此基于出价的拍卖会青睐它。

这正是分类器系统处理书中所称的*通用性与判别性*（generality versus discrimination）之间权衡的方式。

通用规则因其条件经常被满足而频繁地被检验，但它们在许多状态下规定的是同一个动作。

特定规则可以使动作与状态相匹配，但它们需要很长的历史积累，其强度才有意义。

这种权衡类似于参数估计与非参数估计之间的权衡（{ref}`lae_kernel`）。

由于存在特异性溢价，一条特定规则能够为一条通用规则开出一个例外，而通用性与特异性之间的平衡本身也会随着规则的泛化与特化而演化。

### 拍卖

拍卖将出价转化为决策。

书中列出了两条原则：出价最高者获胜，或每条匹配的规则以与其相对强度相等的概率获胜。

这两者都属于同一类：**softmax**：

$$
\Pr(e \text{ 获胜}) = \frac{\exp(\beta B_e)}{\sum_{e' \in M} \exp(\beta B_{e'})},
$$

其中 $M$ 是匹配规则的集合。

当 $\beta \to \infty$ 时，softmax 将全部权重放在最高出价上，因此**argmax拍卖**是其极限情形。

将其应用于强度的对数并令 $\beta = 1$，便得到相对强度规则，因为 $\exp(\ln S_e) / \sum_{e'} \exp(\ln S_{e'}) = S_e / \sum_{e'} S_{e'}$。

参数 $\beta$ 设定了系统进行试验的程度。

argmax拍卖永远不会让一条看起来较差的规则行动，因此它永远无法发现该规则其实更好；而在 $\beta$ 有限时，系统总是有可能发现这一点。

{doc}`marimon_mcgrattan_sargent` 使用的是对强度而非出价施加的argmax拍卖：在那里，最强的匹配规则采取行动，出价仅用来决定桶链中的支付，所有的试验探索都交由创造规则的算子来完成。

(gc_bandit)=
### 双臂老虎机

最简单的分类器系统是布莱恩·亚瑟（Brian Arthur）和卡尔·西蒙（Carl Simon）提出的一个例子，他们于1989年在圣塔菲研究所分别独立提出了这一想法（另见 {cite:t}`Arthur1991`）。

臂 $i$ 支付一个均值为 $\mu_i > 0$ 的随机奖励，且 $\mu_1 > \mu_2$，但玩家对任何一个分布都一无所知。

该分类器系统持有两条规则，`#1`（"无论历史如何，拉动臂1"）和 `#0`（"拉动臂2"），它们的条件总是被满足。

亚瑟的计账方法将规则 $i$ 的强度设为该臂所提供收益的平均值：

$$
S_{i,\tau_i} = S_{i,\tau_i - 1} + \frac{1}{\tau_i}\left(x_{i,\tau_i} - S_{i,\tau_i - 1}\right),
$$

其中 $\tau_i$ 记录臂 $i$ 被拉动的次数，拍卖以概率 $S_i / (S_1 + S_2)$ 选中规则 $i$。

这两条规则一开始拥有**相等**的强度：系统对任何一个臂都一无所知，必须从其所经历的奖励中建立自己的估计。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "The classifier bandit converges to probability matching"
    name: fig-gc-bandit
---
def two_armed_bandit(μ, noise_sd=0.5, T=20_000, seed=0):
    "Strength = running average of an arm's payoff; pull ∝ strength."
    rng = np.random.default_rng(seed)
    S = np.ones(2)                             # equal strengths: no prior knowledge
    τ = np.array([1, 1])                       # pull counters
    pulls = np.empty(T, int)
    for t in range(T):
        w = np.maximum(S, 1e-9)                # guard against a dip below zero
        i = 0 if rng.random() < w[0] / w.sum() else 1
        reward = μ[i] + noise_sd * rng.standard_normal()
        τ[i] += 1
        S[i] += (reward - S[i]) / τ[i]         # running average
        pulls[t] = i
    return pulls

pulls = two_armed_bandit([3.0, 1.0])
frac_best = np.cumsum(pulls == 0) / np.arange(1, len(pulls) + 1)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(frac_best, lw=1)
ax.axhline(0.75, color='k', ls='--', label="probability match  $\\mu_1/(\\mu_1+\\mu_2)$")
ax.axhline(1.0, color='C3', ls=':', label="optimal (always best arm)")
ax.set_xlabel("$t$")
ax.set_ylabel("fraction of pulls on the best arm")
ax.set_ylim(0.5, 1.05)
ax.legend(frameon=False)
plt.show()
```

拉动较好那个臂的比例会收敛，但**不会收敛到一**。

它收敛到 $\mu_1/(\mu_1 + \mu_2)$，即该臂占总预期奖励的份额。

```{code-cell} ipython3
for μ in ([1.0, 0.5], [2.0, 1.0], [3.0, 1.0]):
    frac = np.mean(two_armed_bandit(μ)[-5000:] == 0)
    print(f"μ = {μ}: fraction on best arm = {frac:.3f}, "
          f"probability match = {μ[0]/(μ[0]+μ[1]):.3f}  (optimal = 1.0)")
```

亚瑟运用随机逼近（stochastic approximation）的方法证明了：强度比例分类器会进行**概率匹配**（probability-matches）。

论证过程很简短。

只要强度保持为正——这需要 $\mu_i > 0$——每个臂被拉动的概率就始终为正，因此会被无限次拉动。

每个强度都是该臂收益的样本均值，因此根据强大数定律，它会收敛到 $\mu_i$。

拍卖的概率因而收敛到 $\mu_1/(\mu_1 + \mu_2)$，拉动臂1的比例也是如此。

该分类器按照各臂预期奖励的比例来拉动，而不是专注于最好的那个，因此它永远都在把奖励留在桌面上。

当 $\mu = (3, 1)$ 时，它每次拉动的收益为 $0.75 \times 3 + 0.25 \times 1 = 2.5$，而始终拉动臂1则可获得3。

该比例规则也只关心比率 $\mu_1/\mu_2$：$\mu = (1, 0.5)$ 和 $\mu = (2, 1)$ 产生相同的份额。

请注意问题出在哪里。

计账是正确的：每个强度都收敛到正确的数值，即其对应臂的平均收益。

问题出在**拍卖**上，它将正确的强度转化为与之成比例的选择概率。

{ref}`gc_ex3` 仅通过改变拍卖就治愈了概率匹配问题。

这将两项工作区分开来。

拍卖掌管着探索与利用之间的平衡。

计账则掌管*功劳分配*（credit assignment），这在老虎机问题中是平凡的，因为每份收益都归属于产生它的那个臂，但在序贯问题中却很困难。

(gc_bucket_brigade)=
### 桶链

在序贯问题中，一条规则的决策可能要经过一连串后续决策，才在很久以后才获得回报。

计账方式此时必须为仅仅*建立起*一个未来有利可图的决策的规则给予奖励。

霍兰德为此设计的装置是**桶链**（bucket brigade）：每条赢得拍卖的规则都将其出价支付给在它之前刚刚采取行动的那条规则，即将系统带入当前规则得以行动的状态的那条规则。

一条规则的收入是其从环境中获得的收益，加上其后继规则支付给它的出价。

考虑最短的链条。

*设立*规则 $s$ 先行动，将系统带入一个状态，在该状态下*收益*规则 $p$ 行动并获得 $R$。

规则 $p$ 将其出价 $b S_p$ 支付给 $s$；为简单起见，每条规则出价都是其强度的相同比例 $b$。

霍兰德将强度保持为**累积余额**（cumulative balances）：一条行动的规则支出其出价并获得收入，就像一个银行账户，即 $S \leftarrow S - bS + \text{收入}$。

而亚瑟（在老虎机问题中）以及 {cite:t}`MarimonMcGrattanSargent1990`（在下一讲的经济体中）则改用**平均**（average）方式：

$$
S_{\tau} = S_{\tau-1} + \frac{1}{\tau}\left[\text{收入}_\tau - (1 + b)\, S_{\tau-1}\right],
$$

其中 $\tau$ 记录该规则行动的次数，因此强度是扣除出价后收入的移动平均值。

当 $b = 0$ 时，这就是老虎机问题中的移动平均。

下面的代码在这条链上分别运行两种计账方式，强度均从零开始，$R = 1$，$b = 0.1$。

```{code-cell} ipython3
def bucket_brigade(R=1.0, b=0.1, passes=10_000, accounting='average'):
    "Two-link chain: the setup rule acts, then the payoff rule acts and earns R."
    S = np.zeros(2)                                # strengths of [setup, payoff]
    path = []
    for τ in range(1, passes + 1):
        receipts = np.array([b * S[1], R])         # the payoff rule pays its bid back
        if accounting == 'average':                # Arthur; Marimon-McGrattan-Sargent
            S = S + (receipts - (1 + b) * S) / τ
        else:                                      # Holland's cumulative balance
            S = S - b * S + receipts
        path.append(S.copy())
    return np.array(path)

for accounting in ('balance', 'average'):
    path = bucket_brigade(accounting=accounting)
    print(f"{accounting}: [setup, payoff] after passes 1, 2, 3: "
          f"{path[0].round(3)}, {path[1].round(3)}, {path[2].round(3)};  "
          f"limit {path[-1].round(3)}")
```

有两点值得注意。

第一，功劳**每一轮只向后传递一环**。

在第一轮中，收益规则获得了 $R$，但设立规则什么也没有得到，因为收益规则的强度、也就是它的出价，在它支付时仍然为零。

在一条含有 $k$ 个环节的链中，收益需要经过 $k - 1$ 轮才能到达第一条规则。

第二，两种计账方式对设立规则价值的判断不同。

累积余额在收入等于出价处趋于静止，即 $b S_p = R$ 且 $b S_s = b S_p$，因此 $S_s = S_p = R/b$：设立规则最终的强度恰好与收益规则相等。

平均方式则在 $(1 + b)S_p = R$ 且 $(1 + b)S_s = b S_p$ 处趋于静止，因此

$$
S_p = \frac{R}{1+b}, \qquad S_s = \frac{b}{1+b}\, S_p .
$$

沿着链条向后每一环，强度都按 $b/(1+b)$ 的比例缩小，当 $b = 0.1$ 时约为0.09。

平均计账方式对设立规则的估值，仿佛它对未来进行了非常严重的贴现，这一事实在 {doc}`marimon_mcgrattan_sargent`（{ref}`mms_impatience`）中会产生重大影响。

{ref}`gc_ex5` 探讨了设立规则在什么情况下能够胜过立即获得收益的规则。

那么为什么要用平均方式呢？

霍兰德基于与强度成比例出价的余额方式可以写成 $S \leftarrow S + b\,(\text{收入}/b - S)$，这是对收入$/b$的*固定增益*（constant-gain）平均，只要收入是随机的，它就会持续波动。

$1/\tau$ 增益则将其转化为 {ref}`br_one_recursion` 中所研究的那种递减增益随机逼近。

其极限是均值动态 $dS/d\tau = \mathbb{E}[\text{收入}] - (1 + b) S$ 的一个静止点，正是这一点使强度得以收敛，并使下一讲能够刻画其极限。

桶链也是**时序差分学习**（temporal-difference learning） {cite}`Sutton_2018` 的先驱。

一条规则的强度被推向其当前所获得的收益，再加上其将系统交给的那条规则的强度的一部分。

{doc}`mccall_q` 和 {doc}`inventory_q` 中的Q学习，以同样的方式将一个状态-动作对的价值推向当前收益加上下一状态的贴现价值，其中后继者的出价扮演了该贴现价值的角色。

设计这种反向流动是构建分类器系统的核心技艺，也正是 {doc}`marimon_mcgrattan_sargent` 必须做对的地方，以使主体学会今天为了明天的交易而接受货币。

(gc_evolutionary)=
## 演化编程

我们已经巡览了四种大脑。

最后一个想法关乎如何*运用*它们。

本系列中的许多适应性主体系统，无论多么迟缓，都会找到通往均衡的路径。

{doc}`olg_adaptive_money` 中的最小二乘学习者收敛到一个平稳均衡，而 {doc}`learning_approximation` 中的学习者则收敛到其函数形式所能达到的最佳近似均衡。

在萨金特的校准下，{doc}`exchange_rate_learning` 中的牛顿学习者收敛到由历史所挑选出的连续统均衡中的一个。

但并非所有情形都如此。

在阿里福维奇的校准下，同样的牛顿学习者未能收敛。

而 {doc}`exchange_rate_learning` 中的遗传算法经济体（{ref}`xr_ga`）则从未稳定下来：其储蓄率趋近于均衡值附近，但变异不断注入新的字符串，使得其汇率持续以不会消退的波动率波动。

**演化编程**将这种收敛倾向转化为一种工具。

如果一个适应性主体种群能够可靠地收敛到某个均衡，我们就可以将该种群作为*计算*该均衡的*方法*来运行，尤其是在模型过于复杂以至于无法手工求解的情况下。

遗传算法经济体提醒我们，这种可靠性必须经过设计才能获得：如果要让种群最终稳定下来，实验性的探索就必须像退火过程中的扰动那样逐渐消退。

萨金特谨慎地说明了这里正在发生和没有发生的事情：

> 适应性主体在"教导"经济学家，正如任何用于求解非线性方程的数值算法都在"教导"数学家一样。当这些主体能够"教导"我们某些东西时，那是因为我们设计它们如此。

这与贯穿 {doc}`learning_approximation` 的对偶性相同：一个学习型经济体是一个去中心化的均衡计算过程，而一个均衡计算过程是一个中心化的学习算法。

遗传算法和分类器系统只是比递归最小二乘更丰富的计算引擎，能够搜索崎岖的景观，发现一个好规则的结构，而不仅仅是调整固定规则的系数。

其代表性应用是 {cite:t}`KiyotakiWright1989` 的货币搜索理论模型，其中交换媒介并非被假定，而必须从主体选择如何交易的过程中**涌现**出来。

均衡是一组交易策略和匹配概率，而对于该模型的丰富版本，很难通过解析方法进行刻画。

{cite:t}`MarimonMcGrattanSargent1990` 将霍兰德分类器系统的种群放入该环境中，观察它们收敛到均衡，然后构建了一个没有已知解析解的五种商品版本，让分类器系统提示该均衡的样貌。

## 结束语

这份目录中的四种大脑在其所假设的既定条件上有所不同。

感知机被赋予了一个函数形式，只被要求提供其系数，这就是为什么计量经济学家一眼就能认出它。

霍普菲尔德网络被赋予了模式本身，只被要求回忆它们。

遗传算法被赋予了一种编码和一个适应度函数，必须在一个无法计算梯度的空间中进行搜索。

分类器系统被赋予了一套可以书写规则的词汇，必须发现哪些规则值得保留。

沿着这份清单往下走，我们赋予主体的东西越来越少，而要求它发现的东西越来越多，这正是有限理性研究纲领推动我们前进的方向。

在此过程中出现了三个警示，三者都会在下一讲中再次出现。

遗传算法的个体不学习：只有种群在学习，这使得它更像是一个社会的模型，而非一个心智的模型，而且正如 {doc}`exchange_rate_learning` 中所示，一个从未停止试验的种群不一定会稳定下来。

老虎机的例子表明，一个分类器系统的行为是由其拍卖机制及其计账方式决定的，而不是仅仅因为拥有分类器这一事实本身：拍卖机制决定它试验的程度，而计账方式决定它如何分配信用。

而桶链算法表明，这种计账方式内含了对未来的一种估值：在强度取平均的情况下，一个比收获收益的规则早一环设立该收益的规则，其价值只相当于后者的 $b/(1+b)$。

{doc}`marimon_mcgrattan_sargent` 将这套机制组装成必须从零开始学习该用哪种商品作为货币的主体。

它们的规则在 $\{0, 1, \#\}$ 上设定条件，并按强度比例出价，附加特异性溢价；一个基于强度的argmax拍卖机制在它们之中做出选择；一个桶链算法将每一次交易与它所促成的消费联系起来；而一个活跃度随时间衰减的遗传算法则不断对它们进行更新。

这三个警示都直接关系到这些主体最终能够学到什么。

## 练习

```{exercise-start}
:label: gc_ex1
```

霍普菲尔德网络的回忆过程是在能量 $E(s) = -\tfrac{1}{2}s^\top ws$ 上的下降过程。

直接验证这一下降过程。

取一个存储的字母，破坏若干像素，并在回忆动态的每一步记录能量。

确认能量从不增加，并且一次停止于存储字母的回忆最终会停止在能量 $-N/2$ 处，这是任何状态所能具有的最低能量。

然后证明，当 $w$ 为投影规则矩阵时，在同步更新 $s' = \operatorname{sgn}(ws)$ 下能量不可能增加。

最后，更严重地破坏模式，并对失败情况进行分类：回忆是落在一个*不同*的存储字母上，还是落在一个从未被存储过的*虚假*状态上？

比较各种情况下的能量，并用它来解释为什么会出现错误。

```{exercise-end}
```

```{solution-start} gc_ex1
:class: dropdown
```

下面是一个在每一步都记录能量的 `recall` 版本。

```{code-cell} ipython3
def recall_with_energy(w, s0, max_iter=30):
    s = s0.copy()
    trace = [energy(w, s)]
    for _ in range(max_iter):
        s_new = np.sign(w @ s)
        s_new[s_new == 0] = 1
        trace.append(energy(w, s_new))
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s, trace

rng = np.random.default_rng(1)
i = letters.index('E')
corrupt = σ[i].copy()
corrupt[rng.choice(25, 5, replace=False)] *= -1
final, trace = recall_with_energy(w_hop, corrupt)

print(f"energy along the recall path: {[round(e, 2) for e in trace]}")
print(f"monotonically non-increasing: {all(np.diff(trace) <= 1e-9)}")
print(f"recovered the intended letter 'E': {np.array_equal(final, σ[i])}")
```

能量从其被破坏后的值降至 $-N/2 = -12.5$，并保持在那里。

下面说明为什么能量不可能上升。

设 $s' = \operatorname{sgn}(ws)$，$d = s' - s$。

展开这个二次型，并利用 $w$ 的对称性，可得：

$$
E(s') - E(s) = -d^\top w s - \tfrac{1}{2}\, d^\top w d .
$$

$d$ 中每个非零项为 $d_i = 2 s'_i = 2\operatorname{sgn}((ws)_i)$，所以第一项是
$-2\sum_i |(ws)_i|$，对所有被翻转的像素求和，这一项永远不为正。

第二项也永远不为正，因为投影矩阵 $w$ 是半正定的。

所以能量从不上升，并且只要某个像素相对一个非零的场而翻转，能量就会严格下降。

由于状态数是有限的，回忆过程不可能循环，必然会停止在某个不动点上。

（如代码所示，平局时偏向 $+1$，因此一个像素只有在场严格为负时才会从 $+1$ 翻转到 $-1$，而这种严格的下降排除了循环的可能。）

这一论证用到了半正定性，而这正是投影规则所具有的性质。

通常对角线置零的赫布规则并不具有这一性质，在那样的权重下，同时更新所有像素可能会在两个状态之间循环。

霍普菲尔德本人的能量论证 {cite}`Hopfield1982` 针对的是一次更新一个像素的情形。

现在更严重地破坏模式，并将结果分为三类：预期的字母、一个*不同的*存储字母，以及一个虽是不动点但从未被教授过的虚假状态。

下面的代码还会在全部 1000 条回忆路径上检验单调性，并查看处于最低可能能量的虚假状态。

```{code-cell} ipython3
def classify_outcome(final, i):
    if np.array_equal(final, σ[i]):
        return "intended"
    if any(np.array_equal(final, σ[k]) for k in range(len(letters))):
        return "wrong letter"
    return "spurious"

counts = {"intended": 0, "wrong letter": 0, "spurious": 0}
wrong_energy, spurious_energy, deepest = [], [], []
always_downhill = True
for i in range(len(letters)):
    for seed in range(200):
        r = np.random.default_rng(1000*i + seed)
        corrupt = σ[i].copy()
        corrupt[r.choice(25, 7, replace=False)] *= -1
        final, trace = recall_with_energy(w_hop, corrupt)
        always_downhill &= all(np.diff(trace) <= 1e-9)
        kind = classify_outcome(final, i)
        counts[kind] += 1
        if kind == "wrong letter":
            wrong_energy.append(energy(w_hop, final))
        elif kind == "spurious":
            spurious_energy.append(energy(w_hop, final))
            if np.isclose(energy(w_hop, final), -12.5):
                deepest.append(final)

negatives = sum(any(np.array_equal(s, -σ[k]) for k in range(len(letters))) for s in deepest)
print(f"7-pixel corruptions ({sum(counts.values())} trials): {counts}")
print(f"energy non-increasing on every path: {always_downhill}")
print(f"stored patterns all have energy {energy(w_hop, σ[0]):.1f}")
print(f"wrong-letter results: energy in [{min(wrong_energy):.1f}, {max(wrong_energy):.1f}]")
print(f"spurious results:     energy in [{min(spurious_energy):.1f}, {max(spurious_energy):.1f}]")
print(f"spurious results at energy -N/2: {len(deepest)}, "
      f"of which sign-reversed letters: {negatives}")
```

两种失败模式都会出现，而且在这种破坏程度下，虚假状态的数量比错误字母多出近九倍。

一个**错误字母**的结果恰好处于存储的能量 $-N/2$ 上：这种破坏将起点推过了一个盆地边界，进入了另一个同样深的存储记忆的吸引域。

能量下降会收敛到*某个*最小值，但当相关模式的吸引域相互交错时，无法保证收敛到*最近*的那个。

一个**虚假**的结果是设计者从未打算得到的不动点，作为存储规则的一种副产品被创造出来。

大多数虚假结果都比存储的模式浅：它们是存在于字母张成空间之外的不动点，其中大多数是局部最小值。

有少数恰好处于 $-N/2$，这要求它们恰好位于该张成空间之内。

其中一些是字母的"底片"：因为 $\operatorname{sgn}(w(-s)) = -\operatorname{sgn}(ws)$，只要 $\sigma$ 是不动点，向量 $-\sigma$ 就也是不动点。

大多数则是恰好取值为 $\pm 1$ 的字母的带符号混合，例如 $E - I + T$ 和 $A - E + I$。

因此，仅凭深度并不能判断某个记忆是否是我们所要求的那种。

由于所有这些都是更新规则下的不动点，动态过程无法从任何一个中逃脱。

这正是为什么**模拟退火**很重要：加入递减的随机扰动，能让系统在稳定下来之前跳出一个浅层虚假吸引域，或跨越一个盆地边界，而纯粹的能量下降永远做不到这一点。

```{solution-end}
```

```{exercise-start}
:label: gc_ex2
```

遗传算法的探索来自两个算子：交叉和变异。

书中指出，交叉"是该算法的核心"，而单独的变异"是注入多样性的一种糟糕机制"。

在阿克塞尔罗德的博弈中检验这一论断。

编写一个仅使用变异的变体（每个子代都是从一个被选中的父代经变异得到的副本，不进行交叉），并将其达到的适应度与完整算法进行比较。

用每次运行的冠军字符串的 `panel_fitness` 来给每次运行打分，这样每个策略都会面对相同的随机对局，并使用多个随机种子。

书中谨慎地指出了*何时*单独的变异是弱势的："当变异率设定为非常低的值时，单独的变异是一种注入多样性的糟糕机制。"

因此首先在**低**变异率 $p_m = 0.002$ 下运行比较，此时交叉必须承担探索的重任，然后再在较高的变异率 $p_m = 0.02$ 下运行一次。

```{exercise-end}
```

```{solution-start} gc_ex2
:class: dropdown
```

这个仅使用变异的变体保留了带窗口的轮盘赌选择，但去掉了交叉。

```{code-cell} ipython3
def evolve_mutation_only(N=60, generations=60, p_mut=0.002, seed=0):
    "Each child is a mutated copy of one roulette-selected parent."
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (N, 70))
    for _ in range(generations):
        fit = fitness(pop, rng.integers(0, 2, (1, 150)))
        weight = fit - fit.min() + 1e-6
        cum = np.cumsum(weight)
        pop = pop[np.searchsorted(cum, rng.random(N) * cum[-1])]
        pop[rng.random(pop.shape) < p_mut] ^= 1
    fit = fitness(pop, rng.integers(0, 2, (1, 150)))
    return pop[fit.argmax()]

n_seeds = 12
for p_mut in (0.002, 0.02):
    gaps = []
    for seed in range(n_seeds):
        with_x = panel_fitness(evolve(generations=60, p_mut=p_mut, seed=seed)[0])
        without_x = panel_fitness(evolve_mutation_only(p_mut=p_mut, seed=seed))
        gaps.append(with_x - without_x)
    gaps = np.array(gaps)
    print(f"p_mut = {p_mut}: crossover minus mutation-only = {gaps.mean():.3f} "
          f"(standard error {gaps.std(ddof=1) / np.sqrt(n_seeds):.3f}); "
          f"crossover ahead in {np.sum(gaps > 0)} of {n_seeds} seeds")
```

在低变异率下，交叉的优势很明显：其冠军字符串在几乎每个随机种子下的适应度都更高。

在较高的变异率下，这一优势几乎消失：在这些随机种子中，平均差距位于零的一个标准误差之内，而且交叉只在大约一半的种子中领先。

原因正是书中所给出的。

在变异很少的情况下，仅使用变异的种群只能每次通过一次罕见的位翻转来缓慢前进，并且随着选择过程不断复制其少数最优字符串，多样性很快就会丧失。

而交叉则重组了那些已经证明有用的字符串片段。

在这种编码方式下，连续16个表项构成的一个区块对应于共享最近一轮结果的一组历史记录，因此交叉是在把一个字符串对某些近期结果的应对，拼接到另一个字符串对其他近期结果的应对上。

这就是模式理论（schema story）的说法，不过模式定理只保证短小而适应度高的区块往往能够存续下来，而并不保证拼接它们一定能改善字符串本身。

当变异本身就能产生大量多样性时，交叉的贡献就不再是关键性的，这正是书中所作的限定：只有在变异率非常低时，单独的变异才是弱势的。

```{solution-end}
```

```{exercise-start}
:label: gc_ex3
```

双臂老虎机分类器进行概率匹配，这是次优的：它会永远以固定的比例继续拉动较差的那个臂。

一个自然的修正方法是使拍卖变得更加"贪婪"。

用一个softmax函数替代与强度成比例的选择规则：

$$
\pi_1 = \frac{e^{\beta S_1}}{e^{\beta S_1} + e^{\beta S_2}},
$$

其中 $\beta$ 控制贪婪程度（$\beta \to \infty$ 时总是选择更强的那个臂）。

1. 实现它，并展示在 $\mu = (3, 1)$ 的情形下，长期而言拉动最佳臂的比例如何依赖于 $\beta$。
1. $\beta$ 需要多大，softmax 才能胜过概率匹配？
1. 现在让两个臂变得难以区分：$\mu = (1.2, 1.0)$，奖励噪声的标准差为2。在40个随机种子下，比较 $\beta = 2$ 与非常贪婪的 $\beta = 32$。贪婪会带来什么代价？

```{exercise-end}
```

```{solution-start} gc_ex3
:class: dropdown
```

softmax老虎机只需改动选择臂的那一行代码。

```{code-cell} ipython3
def bandit_softmax(μ, β, noise_sd=0.5, T=20_000, seed=0):
    "Share of the last 5000 pulls on arm 1 under a softmax auction."
    rng = np.random.default_rng(seed)
    S = np.ones(2)                             # equal strengths, as before
    τ = np.array([1, 1])
    pulls = np.empty(T, int)
    for t in range(T):
        p1 = 1 / (1 + np.exp(-β * (S[0] - S[1])))
        i = 0 if rng.random() < p1 else 1
        reward = μ[i] + noise_sd * rng.standard_normal()
        τ[i] += 1
        S[i] += (reward - S[i]) / τ[i]
        pulls[t] = i
    return np.mean(pulls[-5000:] == 0)

μ = [3.0, 1.0]
β_star = np.log(μ[0] / μ[1]) / (μ[0] - μ[1])
print(f"probability match target = {μ[0]/(μ[0]+μ[1]):.3f},  optimal = 1.000,  "
      f"break-even β = {β_star:.3f}\n")
for β in (0.5, 1.0, 2.0, 4.0, 8.0):
    theory = 1 / (1 + np.exp(-β * (μ[0] - μ[1])))
    print(f"β = {β:>4}: fraction on best arm = {bandit_softmax(μ, β):.3f}  "
          f"(limit {theory:.3f})")
```

softmax使得每个臂被选中的概率都有界地远离零，因此每个臂都会被无限次拉动，其强度会收敛到 $\mu_i$，与比例规则的情形一样。

因此，长期而言落在臂1上的比例为 $1/(1 + e^{-\beta(\mu_1 - \mu_2)})$，也就是"limit"那一列，模拟结果与此相符。

它超过概率匹配比例 $\mu_1/(\mu_1+\mu_2)$ 的条件恰好是 $e^{-\beta(\mu_1-\mu_2)} < \mu_2/\mu_1$，即

$$
\beta > \frac{\ln(\mu_1/\mu_2)}{\mu_1 - \mu_2} = \frac{\ln 3}{2} \approx 0.55 .
$$

所以 $\beta = 0.5$ 的表现略*差*于概率匹配，而从 $\beta = 1$ 开始，softmax的表现就更好了，并随着 $\beta$ 的增大逐渐逼近最优策略。

与比例规则不同，softmax依赖于差值 $\mu_1 - \mu_2$ 而非比值，所以这一阈值取决于收益所采用的单位。

当两个臂难以区分时，贪婪是有代价的。

```{code-cell} ipython3
for β in (2.0, 32.0):
    shares = np.array([bandit_softmax([1.2, 1.0], β, noise_sd=2.0, seed=s)
                       for s in range(40)])
    print(f"β = {β:>4}: mean share on the better arm {shares.mean():.3f}; "
          f"runs ending mostly on the worse arm: {np.sum(shares < 0.5)} of 40")
```

在 $\beta = 32$ 时，拍卖很快就会集中，并且在这40次运行中，这使得落在较好臂上的平均比例高于 $\beta = 2$ 时的情形。

不过这种优势并不稳健：在400个随机种子上，两者的平均值分别为 $0.63$ 和 $0.60$，相差约一个标准误差。

而且在大约三分之一的运行中，拍卖反而集中在了*较差*的那个臂上。

少数几次不走运的早期回报会让较好的臂看起来更差，此后它几乎再也不会被拉动，因此它的强度也几乎不再被修正：系统因此**锁死**了。

在 $\beta = 2$ 时，系统永远不会锁死在较差的臂上，但也永远不会真正集中。

softmax正是统计力学中温度为 $1/\beta$ 的玻尔兹曼分布。

要想同时获得两方面的好处，需要先"热启动"再缓慢降温，让 $\beta_t$ 随时间趋于无穷，这正是模拟退火的做法。

{doc}`marimon_mcgrattan_sargent` 中的argmax拍卖正是 $\beta = \infty$ 的极限情形，一旦出错就会锁死；这正是为什么其分类器系统需要专门的算子来进行试验，而这种试验活动会被设计为随时间递减。

因此这个练习具体说明了为什么*拍卖*和*计账*——而不仅仅是拥有分类器这一事实——决定了一个分类器系统的表现好坏。

```{solution-end}
```

```{exercise-start}
:label: gc_ex4
```

回到布雷的市场模型（{ref}`br_bray`，{doc}`ls_learning`），其中价格为

$$
p_t = a + b\, \bar\beta_t + u_t ,
$$

其中 $\bar\beta_t$ 是对 $p_t$ 的平均预测，$u_t$ 是独立同分布的 $N(0, \sigma_u^2)$。

当 $b < 1$ 时，最小二乘学习者会收敛到理性预期价格 $a/(1-b)$；当 $b > 1$ 时则不会 {cite}`Bray1982`。

将布雷的代表性最小二乘学习者替换为一个由 $N = 60$ 个预测者组成的群体，每个预测者都是一个16位字符串，被解码为 $[0, 20]$ 区间内的一个预测值。

在每一期：

* 价格由群体的平均预测决定；
* 每个预测者的适应度为其预测误差平方的负值；
* 群体通过本讲中的遗传算法（带窗口的轮盘赌选择、单点交叉、以0.01的概率变异）来繁育下一期的预测者，随后施加 {doc}`exchange_rate_learning` 中的选举算子：只有当子代预测本期价格的能力优于其父代时，子代才会取代父代。

设 $a = (1 - b)p^*$，使得无论 $b$ 取何值，理性预期价格都为 $p^* = 10$，并取 $\sigma_u = 0.1$。

1. 在 $b = 0.7$ 时，展示在20个随机种子下，平均预测会在300期之内稳定在理性预期价格处。
1. 尝试 $b = 1.05$，略高于布雷的临界值。会发生什么，为什么？
1. 萨金特的书中提到，贾斯米娜·阿里福维奇在其1991年的博士论文中发现，这样一个群体"有时即使违反了布雷的必要条件 $b < 1$，也能学会收敛到理性预期均衡" {cite}`Sargent1993`。你的实验在何种意义上与此相符？（{cite:t}`Arifovic1994` 研究了蛛网模型中的遗传算法学习。）

```{exercise-end}
```

```{solution-start} gc_ex4
:class: dropdown
```

下面是预测者的群体。

```{code-cell} ipython3
def decode(pop, lo=0.0, hi=20.0):
    "Read each row of bits as a binary number and map it into [lo, hi]."
    L = pop.shape[1]
    return lo + (hi - lo) * (pop @ 2.0 ** np.arange(L - 1, -1, -1)) / (2**L - 1)

def ga_forecasters(b, p_star=10.0, σ_u=0.1, N=60, L=16, T=300, p_mut=0.01, seed=0):
    "Path of the average forecast in Bray's market with a GA population."
    rng = np.random.default_rng(seed)
    a = (1 - b) * p_star
    pop = rng.integers(0, 2, (N, L))
    path = np.empty(T)
    for t in range(T):
        β = decode(pop)
        path[t] = β.mean()
        p = a + b * β.mean() + σ_u * rng.standard_normal()
        fit = -(p - β)**2
        weight = fit - fit.min() + 1e-6
        cum = np.cumsum(weight)
        parents = pop[np.searchsorted(cum, rng.random(N) * cum[-1])]
        mom, dad = parents[0::2], parents[1::2]
        left = np.arange(L) < rng.integers(1, L, (N // 2, 1))
        kids = np.vstack([np.where(left, mom, dad), np.where(left, dad, mom)])
        kids[rng.random(kids.shape) < p_mut] ^= 1
        parents = np.vstack([mom, dad])
        better = (p - decode(kids))**2 < (p - decode(parents))**2    # election
        pop = np.where(better[:, None], kids, parents)
    return path

paths = {b: np.array([ga_forecasters(b, seed=s) for s in range(20)]) for b in (0.7, 1.05)}
for b, P in paths.items():
    gap = np.abs(P - 10)
    print(f"b = {b}: median |average forecast - p*| at t = 10, 100, 299: "
          f"{np.median(gap[:, 10]):.3f}, {np.median(gap[:, 100]):.3f}, "
          f"{np.median(gap[:, -1]):.3f};  runs within 0.25 of p* at the end: "
          f"{np.sum(gap[:, -1] < 0.25)} of 20")
```

下图展示了全部四十条路径。

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (b, P) in zip(axes, paths.items()):
    ax.plot(P.T, lw=1, alpha=0.6)
    ax.axhline(10, color='k', ls='--', lw=1, label="$p^* = 10$")
    ax.set_title(f"$b = {b}$")
    ax.set_xlabel("period")
axes[0].set_ylabel("average forecast")
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()
```

在 $b = 0.7$ 时，这些运行中的每一条都稳定在了理性预期价格。

在 $b = 1.05$ 时，只有少数几条运行最终接近这一价格。

其余的运行在最初几期（此时群体仍具有多样性）就被推离了这一价格。

此后，除了偶尔出现的跳跃外，它们都停滞在一个每一期都朝同一方向偏离的错误预测上。

{ref}`br_one_recursion` 中的T映射解释了这一方向。

当群体的平均预测为 $\bar\beta$ 时，平均而言会导致价格 $T(\bar\beta) = a + b\bar\beta$，而选择和选举操作都会偏向于更接近实现价格的预测。

由于 $T(\bar\beta) - \bar\beta = (b - 1)(\bar\beta - p^*)$，当 $b < 1$ 时被偏好的预测位于 $p^*$ 的同一侧，而当 $b > 1$ 时则位于另一侧。

遗传算法超出了随机近似理论的范畴，但在这里它的移动方向与该理论的均值动态所预测的方向一致。

为什么它会停滞，而不是一路跑到 $[0, 20]$ 的边界？

因为 $b$ 接近1，这种推力很弱：一个与 $p^*$ 相差1.5的群体，每期产生的系统性预测误差只有 $0.05 \times 1.5 = 0.075$，小于噪声的大小。

而且一旦字符串几乎完全相同，交叉就不会创造任何新的东西了，并且预测值的许多微小移动需要同时翻转多个比特位（二进制编码的"汉明悬崖"），因此群体可能会被冻结住。

所以这个实验在一种弱的意义上与书中的说法相符。

当 $b > 1$ 时，群体只有在恰好停滞于理性预期价格附近的那些运行中，才会收敛到该价格附近。

均衡本身并不吸引它，正如它也不吸引最小二乘学习一样，后者在 $b > 1$ 时会发散，不过当 $b$ 如此接近1时，发散的速度会非常缓慢（{doc}`ls_learning`）。

即使在 $b = 0.7$ 时，选举算子也很重要。

如果去掉它，群体无论子代预测得多差都会保留下来，并且收敛到 $p^*$ 的紧密程度会大大降低：正如 {doc}`exchange_rate_learning` 中所示，正是选举操作通过使群体趋于同质化，才使得一个遗传群体能够学习。

```{solution-end}
```

```{exercise-start}
:label: gc_ex5
```

扩展 {ref}`gc_bucket_brigade` 中的两环节链条，在第一阶段给系统增加一个选择。

一条*短视*规则 $m$ 立即收取 $r_m = 1$ 并结束该回合。

设置规则 $s$ 不收取任何回报，但会将系统推进到第二阶段，在那里支付规则 $p$ 会收取 $R$。

为了将计账与拍卖区分开来，在每个回合中以相等概率随机选择 $m$ 或 $s$，并使用 $b = 0.1$ 的平均化计账方式。

1. 假设没有向后支付：支付规则的出价不付给任何人。设置规则的强度会发生什么？
1. 如果支付规则将其出价支付给设置规则，求 $m$ 和 $s$ 的平稳强度，以及argmax拍卖会选择设置规则所需的最小比率 $R/r_m$。通过模拟检验你的答案。
1. 用霍兰德的累积余额方式重复第2部分。哪种计账方式更有耐心？

```{exercise-end}
```

```{solution-start} gc_ex5
:class: dropdown
```

下面是在第一阶段带有选择的链条。

```{code-cell} ipython3
def two_stage(R, r_m=1.0, b=0.1, T=20_000, accounting='average', backward=True, seed=0):
    "Strengths of [myopic, setup, payoff] when the first-stage rule is chosen at random."
    rng = np.random.default_rng(seed)
    S, τ = np.zeros(3), np.zeros(3)
    for _ in range(T):
        if rng.random() < 0.5:
            acts = [(0, r_m)]                                  # myopic: paid at once
        else:
            acts = [(1, b * S[2] if backward else 0.0),        # setup: paid the next bid
                    (2, R)]                                    # payoff: paid by the world
        for k, receipts in acts:
            τ[k] += 1
            if accounting == 'average':
                S[k] += (receipts - (1 + b) * S[k]) / τ[k]
            else:
                S[k] += receipts - b * S[k]
    return S

b = 0.1
print(f"predicted threshold R / r_m = (1 + b) / b = {(1 + b) / b:.1f}\n")
for R in (5.0, 10.0, 12.0, 20.0):
    rows = {'no backward payment': two_stage(R, backward=False),
            'average': two_stage(R),
            'balance': two_stage(R, accounting='balance')}
    for name, S in rows.items():
        winner = 'setup' if S[1] > S[0] else 'myopic'
        print(f"R = {R:>4}, {name:>19}: S_m = {S[0]:7.3f}, S_s = {S[1]:7.3f}  -> {winner}")
    print()
```

**第1部分。** 没有向后支付时，设置规则的收入始终为零，所以无论 $R$ 多大，其强度都不会升到零以上。

任何基于强度的拍卖都不会选择它。

**第2部分。** 在平均化计账下，平稳强度满足 $(1+b)S_m = r_m$，$(1+b)S_p = R$ 以及 $(1+b)S_s = b S_p$，因此

$$
S_m = \frac{r_m}{1+b}, \qquad S_s = \frac{b}{(1+b)^2}\, R .
$$

设置规则恰好在 $R / r_m > (1 + b)/b = 11$ 时获胜。

模拟证实了这一点：在 $R = 10$ 时设置规则落败，而在 $R = 12$ 时则获胜。

**第3部分。** 在累积余额计账下，$S_m = r_m/b$，$S_s = S_p = R/b$，因此只要 $R > r_m$，设置规则就会获胜。

霍兰德的计账方式更有耐心；而平均化计账方式则非常急躁。

一个以 $b = 0.1$ 对强度进行平均化的分类器系统，会偏好现在的1而不是一步之后的10。

{ref}`mms_impatience` 展示了同样的算术，当应用到将交易与消费相连接的链条上时，是如何使 {doc}`marimon_mcgrattan_sargent` 中的主体实际上变得非常急躁的。

```{solution-end}
```
