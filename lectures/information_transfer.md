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
  title: 信息传递的激励
  headings:
    Overview: 概览
    A statistical decision problem: 一个统计决策问题
    A statistical decision problem::States, signals, and actions: 状态、信号与行动
    A planner and an agent: 规划者与代理人
    A planner and an agent::Action plans and reports: 行动计划与报告
    A planner and an agent::The revelation principle: 显示原理
    A planner and an agent::Solving the linear program: 求解线性规划
    Economies with two actions: 只有两个行动的经济
    Dominated actions as threats: 用被占优的行动作为威胁
    Truthful but coarse reports: 如实但粗略的报告
    Comparing experiments: 比较实验
    Comparing experiments::Blackwell's order through posteriors: 通过后验表达的布莱克威尔序
    Comparing experiments::A transport problem: 一个运输问题
    Better information can hurt the planner: 更好的信息可能损害规划者
    Better information can hurt the planner::Green's third example: 格林的第三个例子
    Better information can hurt the planner::A family of refinements: 一个精细化族
    Better information can hurt the planner::Experiments with two signals: 只有两个信号的实验
    What the discussants added: 评论者补充的内容
    Relation to other lectures: 与其他讲座的联系
    Relation to other lectures::Later work on communication and delegation: 关于沟通与委托的后续研究
    Concluding remarks: 结语
    Exercises: 练习
---

(information_transfer)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 信息传递的激励

```{contents} Contents
:depth: 2
```

## 概览

在许多组织中，收集信息的人并不是依据信息采取行动的人。

股票经纪人研究一家公司，而客户决定是否购买其股票。

医生检查病人，病人选择治疗方案。

部门经理了解当地情况，而总部则在各部门之间分配资金。

当信息提供者与决策者的偏好不一致时，信息提供者可能会谎报他所看到的情况。

{cite:t}`Green1982` 研究了一位决策者——**规划者**，她必须从一位**代理人**那里获取观测信息，而该代理人对规划者行动的偏好与她不同。

规划者先行动，并承诺一个**行动计划**，这是一条将每个可能的报告映射到行动的概率分布上的规则。

然后代理人观察从某统计实验中抽取的信号，形成后验信念，并发送该行动计划给予最大回报的报告。

格林得到了以下结果。

1. 规划者将自己局限于那些使代理人愿意如实报告的计划不会有任何损失，因此规划者的问题是一个单一的线性规划问题。
2. 若只有两个行动，规划者最多只能做到忽略代理人或听任他行事；若有两个行动和两个信号，她要么达到她的最优结果，要么从代理人那里得不到任何好处。
3. 若行动多于两个，最优计划可以随机化，并且可以把正概率放在一个规划者若亲自看到信号绝不会采取的行动上。
4. 如实报告并不意味着所有信息都被利用：最优计划可以把一些信号合并处理，同时区分另一些信号。
5. 按照 {doc}`blackwell_kihlstrom` 的意义给代理人一个信息量更大的实验，可能会降低规划者的期望效用。
6. 当两个实验都只有两个信号时，这种情况不会发生。

结果5和6把格林的模型与布莱克威尔定理联系起来。

一个亲自看到信号的决策者永远不会偏好信息量较小的实验。

一个必须从别人那里获取信号的规划者则有时会偏好，因为更清晰的信号可能会暴露出一种利益冲突，而较粗略的信号则掩盖了这种冲突。

为了比较实验，格林使用了 {cite:t}`MarschakMiyasawa1968` 提出的关于后验概率的布莱克威尔序的一种刻画。

我们将看到，他们的这种刻画把"一个实验是否是另一个实验的扰动"这个问题转化为 {doc}`opt_transport` 中研究的那类运输问题，再加上一组额外的约束。

该论文发表时附有 {cite:t}`Postlewaite1982` 和 {cite:t}`Riley1982` 的评论。

练习将梳理他们的例子，以及来自 {cite:t}`MarschakMiyasawa1968` 的信息条件比较。

本讲座的内容安排如下。

1. 我们用格林的记号建立一个统计决策问题。
2. 我们加入一个代理人，陈述格林的显示原理，并求解规划者的线性规划问题。
3. 我们研究只有两个行动的经济情形，以及格林关于随机化和信号合并的例子。
4. 我们使用马尔沙克-宫泽的刻画来比较实验，说明更好的信息如何会损害规划者，并证明当两个实验都只有两个信号时不会出现这种情况。
5. 我们汇报评论者补充的内容，并将格林的模型与其他讲座以及后续关于沟通与委托的研究联系起来。

我们先导入一些包。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import product

np.set_printoptions(precision=4, suppress=True)
```

## 一个统计决策问题

### 状态、信号与行动

存在 $M$ 个自然状态 $\theta_1, \ldots, \theta_M$，先验概率为 $\pi = (\pi_1, \ldots, \pi_M)$。

在行动之前，某人会观察到 $N$ 个信号中的一个，记为 $y_1, \ldots, y_N$。

$M \times N$ 的**似然矩阵** $\Lambda = [\lambda_{mn}]$（我们也称之为**实验**）的元素为

$$
\lambda_{mn} = \Pr(y_n \mid \theta_m),
$$

因此 $\Lambda$ 的每一行都是关于信号的一个概率分布。

信号概率和各状态的后验概率为

$$
q = \Lambda^\top \pi,
\qquad
p_{mn} = \Pr(\theta_m \mid y_n) = \frac{\lambda_{mn} \pi_m}{q_n} .
$$ (it_bayes)

我们将这些后验值收集到 $M \times N$ 矩阵 $P = [p_{mn}]$ 中，它的第 $n$ 列 $p_n$ 是收到信号 $y_n$ 后的后验。

存在 $K$ 个行动 $a_1, \ldots, a_K$。

$K \times M$ 的效用矩阵 $U = [u_{km}]$ 给出了行动 $a_k$ 在状态 $\theta_m$ 下的效用 $u_{km}$。

$K \times N$ 矩阵 $UP$ 的第 $(k, n)$ 个元素是收到信号 $y_n$ 后采取行动 $a_k$ 的期望效用。

一个亲自观察到信号的决策者在 $UP$ 的每一列中都会选择取值最大的那个行动。

我们称这条规则为她的**最优计划**，并称她的期望效用

$$
v(\Lambda) = \sum_{n=1}^N q_n \max_k \, (UP)_{kn}
$$ (it_bayes_value)

为她的**最优价值**。

如果没有信号，她选择使 $(U\pi)_k$ 最大的行动，她的期望效用是 $v_0 = \max_k (U\pi)_k$。

以下函数计算后验以及两个基准价值 $v(\Lambda)$ 和 $v_0$。

```{code-cell} ipython3
def posteriors(Λ, π):
    """Signal probabilities q and the M x N matrix P of posteriors."""
    joint = π[:, None] * Λ
    q = joint.sum(axis=0)
    if np.any(q <= 0):
        raise ValueError("every signal must have positive probability")
    return q, joint / q


def bayes_value(U, Λ, π):
    """Expected utility of a decision maker who sees the signal herself."""
    q, P = posteriors(Λ, π)
    return q @ (U @ P).max(axis=0)


def no_info_value(U, π):
    """Expected utility of the best action taken without a signal."""
    return (U @ π).max()
```

## 规划者与代理人

### 行动计划与报告

现在假设信号不是由决策者观察到的，而是由一名代理人观察到的。

从这里开始，我们把决策者称为规划者。

代理人在意规划者的行动以及状态，他的效用是 $K \times M$ 矩阵 $W = [w_{km}]$ 中的元素。

格林用 $U'$ 表示代理人的矩阵；我们用 $W$，并保留撇号用于表示其他实验。

规划者和代理人拥有共同的先验 $\pi$，并且双方都知道 $\Lambda$、$U$ 和 $W$。

事件的发展过程如下。

1. 规划者宣布一个行动计划，即一个 $N \times K$ 的马尔可夫矩阵 $Z = [z_{nk}]$，也就是说，一个各行之和为一的非负矩阵；$z_{nk}$ 是当代理人报告 $y_n$ 时规划者采取行动 $a_k$ 的概率。
2. 代理人观察到一个信号 $y_n$，他的后验为 $p_n$。
3. 代理人报告某个信号 $y_r$，该信号不必等于 $y_n$。
4. 规划者从 $Z$ 的第 $r$ 行中抽取一个行动。

记

$$
V = Z W P .
$$

$N \times N$ 矩阵 $V$ 的第 $(r, n)$ 个元素是代理人在观察到 $y_n$ 而报告 $y_r$ 时的期望效用。

因此，代理人的最优报告是每一列中最大的元素所对应的报告。

规划者是有承诺约束的：一旦代理人说了话，即便在知道报告之后另一个行动在她看来更好，她也要执行已宣布的随机化。

格林强调，这一点也包括抽签本身。

在收到报告 $y_r$ 之后，她必须采取第 $r$ 行的抽签所产生的任何行动；她不能在该行所允许的行动中替换成自己最喜欢的那个，并声称是抽签选中了它。

从这个意义上说，规划者是一个斯塔克尔伯格领导者，正如 {doc}`Stackelberg plans <advanced:dyn_stack>` 中那样。

承诺使她能够做出威胁——比如下面格林的第一个例子中那个被弱占优的行动——这些威胁如果她能在听到报告后重新选择，就不会真的执行。

能够承诺的价值也是 {doc}`phillips_credibility` 的主题，该讲座研究了货币政策的基德兰-普雷斯科特模型。

如果没有承诺，规划者会在每次报告之后根据她听到报告后的信念采取她的最优行动，那样我们就会进入 {cite:t}`CrawfordSobel1982` 所研究的廉价磋商（cheap-talk）世界。

### 显示原理

一个计划 $Z$ 会导出一个**应答规则** $\rho$：在观察到 $y_n$ 之后，代理人报告 $y_{\rho(n)}$。

按照格林的做法，应答规则是从观测到报告的一个确定性映射。

格林的第一个结果说，规划者可以将自己局限于应答规则为恒等映射的计划。

```{prf:theorem} 显示原理（格林定理1）
:label: it_thm_revelation

当代理人对每个计划都用对他而言最优的应答规则来回应时，规划者在所有行动计划中所能获得的最高期望效用，等于以下线性规划问题的值：

$$
\max_{Z} \; \sum_{n=1}^N q_n \sum_{k=1}^K z_{nk} \, (UP)_{kn}
$$

满足约束 $z_{nk} \geq 0$，对所有 $n$ 有 $\sum_k z_{nk} = 1$，以及**如实报告约束** $V_{nn} \geq V_{rn}$（对所有 $n$ 和 $r$），即

$$
\sum_{k=1}^K z_{nk} \, (WP)_{kn}
\;\geq\;
\sum_{k=1}^K z_{rk} \, (WP)_{kn}
\quad \text{对所有 } n \text{ 和 } r .
$$
```

该定理是机制设计中显示原理的一个版本；{cite:t}`Myerson1979` 是格林认为与他的分析最接近的参考文献。

论证如下。

假设计划 $Z$ 导出了应答规则 $\rho$。

构造一个新的计划 $\hat Z$，其第 $n$ 行等于 $Z$ 的第 $\rho(n)$ 行。

$\hat Z$ 的各行都在 $Z$ 的各行之中，因此代理人没有任何新的选择。

在观察到 $y_n$ 之后，他此前选择的是 $Z$ 的第 $\rho(n)$ 行，所以他愿意选择 $\hat Z$ 的第 $n$ 行，这与之前是同一个抽签。

在 $\hat Z$ 和如实报告之下，规划者在每次观测之后的行动分布与在 $Z$ 和原始报告之下相同，所以她的期望效用也相同。

格林将如实报告约束写成弱不等式，所以一个无差异的代理人被假定会如实报告。

我们用 $\hat v(\Lambda)$ 表示该线性规划的值。

一个忽略代理人报告的计划满足如实报告约束，而任何计划都不会优于亲自看到信号的规划者，因此

$$
v_0 \;\leq\; \hat v(\Lambda) \;\leq\; v(\Lambda) .
$$

{cite:t}`Riley1982` 指出，若做适当的解释，显示原理本质上近乎是定义性的：代理人只是从 $Z$ 的各行所构成的菜单中选取一个抽签。

这样的理解使格林的规划者成为一个菜单的设计者，由一位信息更充分的下属从中选取，这也是本讲座末尾讨论的委托问题的一个主题。

如实报告并不意味着所有信息都能传递过去：如果 $Z$ 的两行相等，规划者就把对应的两个信号一视同仁，格林的第二个例子将说明这一点。

### 求解线性规划

我们将 $Z$ 的元素按行堆叠为一个长度为 $NK$ 的向量，并将问题交给 SciPy 的 `linprog`。

$Z$ 每行之和为一的约束便采取克罗内克积的形式 `np.kron(np.eye(N), np.ones(K))`。

{doc}`opt_transport` 按列堆叠而不是按行，所以在那里同样的形式编码的是列之和。

由于 `linprog` 执行的是最小化，我们传入规划者目标函数的负值。

```{code-cell} ipython3
def solve_planner(U, W, Λ, π, ties=None):
    """
    Solve the planner's linear program of Green's revelation principle.

    U, W : K x M utility matrices of the planner and the agent
    Λ    : M x N likelihood matrix
    π    : prior over the M states
    ties : None, "agent_best", or "agent_worst"; when the planner has
           several optimal plans, return the one that the agent likes
           best or least, instead of the one the solver happens to find

    Returns the N x K action plan Z, the planner's optimal expected
    utility, and the agent's expected utility under Z when he reports
    truthfully.
    """
    q, P = posteriors(Λ, π)
    K, N = U.shape[0], Λ.shape[1]
    UP, WP = U @ P, W @ P

    # Entry n * K + k of the decision vector is z_{nk}
    planner_obj = (UP * q).T.ravel()
    agent_obj = (WP * q).T.ravel()

    # For every true signal n and every false report r,
    # sum_k (z_{rk} - z_{nk}) (WP)_{kn} <= 0
    rows = []
    for n in range(N):
        for r in range(N):
            if r != n:
                row = np.zeros((N, K))
                row[r] += WP[:, n]
                row[n] -= WP[:, n]
                rows.append(row.ravel())
    A_ub = np.array(rows).reshape(-1, N * K)
    b_ub = np.zeros(len(A_ub))

    # Each row of Z sums to one
    A_eq = np.kron(np.eye(N), np.ones(K))
    b_eq = np.ones(N)

    def solve(c, A_ub, b_ub):
        # HiGHS's presolve step occasionally misreports these small
        # problems as infeasible, so we switch it off
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method="highs",
                      options={"presolve": False})
        if res.status != 0:
            raise RuntimeError(res.message)
        return res.x

    x = solve(-planner_obj, A_ub, b_ub)
    planner = planner_obj @ x
    if ties is not None:
        # Among plans worth the planner's optimum up to a small tolerance,
        # make the agent as well off (or as badly off) as possible; the
        # plan returned is then optimal for the planner up to that tolerance
        sign = {"agent_best": -1, "agent_worst": 1}[ties]
        tol = 1e-8 * max(1, abs(planner))
        x = solve(sign * agent_obj, np.vstack([A_ub, -planner_obj]),
                  np.append(b_ub, -planner + tol))
    # Remove round-off below zero, and turn -0.0 into 0.0
    Z = np.maximum(x, 0.0).reshape(N, K) + 0.0
    return Z, planner, agent_obj @ x
```

当规划者有多个最优计划时，代理人的期望效用在各个计划之间可能不同。

在 `ties=None` 的情况下，`solve_planner` 会报告求解器所找到的任意一个最优计划下代理人的效用。

在 `ties="agent_best"` 或 `ties="agent_worst"` 的情况下，它会求解第二个线性规划，使规划者保持在她的最优值，同时使代理人尽可能好或尽可能差。

## 只有两个行动的经济

当只有两个行动时，正如格林接下来的两个结果所示，规划者的问题会变得简单。

格林针对两个状态证明了第一个结果，但下面的论证对任意数量的状态和信号都适用。

```{prf:theorem} 两个行动（格林定理3）
:label: it_thm_two_actions

若 $K = 2$，规划者存在一个最优计划，要么忽略代理人的报告，要么在每次信号之后都采取代理人若身为决策者本人会选择的那个行动。
```

在第二种情况下，规划者实施的是代理人的最优计划，而不是她自己的最优计划。

当只有两个行动时，$Z$ 的一行由 $z_{n1}$ 决定，即收到报告 $y_n$ 后采取 $a_1$ 的概率，在本节中我们简记为 $z_n$。

在观察到 $y_n$ 之后，代理人对报告 $y_r$ 的估值为

$$
(WP)_{2n} + z_r \left[(WP)_{1n} - (WP)_{2n}\right],
$$

因此无论他看到了什么信号，代理人仅凭 $z_r$ 来对 $Z$ 的各行进行排序。

当他在自己的后验下偏好 $a_1$ 时，他报告具有最大 $z_r$ 的信号；当他偏好 $a_2$ 时，他报告具有最小 $z_r$ 的信号。

因此，只要代理人从未恰好无差异，如实报告就会迫使 $z_n$ 在代理人偏好 $a_1$ 的信号上取某个共同值 $z_{\rm hi}$，而在其他信号上取另一个共同值 $z_{\rm lo} \leq z_{\rm hi}$。

如果代理人在每个信号后都偏好同一个行动，所有的 $z_n$ 都相等，计划就忽略了报告。

否则，规划者的期望效用在三角形 $0 \leq z_{\rm lo} \leq z_{\rm hi} \leq 1$ 上关于 $(z_{\rm lo}, z_{\rm hi})$ 是线性的，所以它在某个顶点处取得最大值。

其中两个顶点 $(0, 0)$ 和 $(1, 1)$ 忽略了报告，而第三个顶点 $(0, 1)$ 则在每次信号之后都给出代理人偏好的行动。

因此 $\hat v(\Lambda) = \max\{v_0, v_A(\Lambda)\}$，其中 $v_A(\Lambda)$ 是规划者在每次信号之后都采取代理人偏好的行动时的期望效用；如果代理人有时无差异，他的平局应以有利于规划者的方式打破。

该定理说的是*某个*最优计划采取这两种形式之一。

当规划者的目标函数沿三角形的某条边保持不变时，介于两者之间的计划也是最优的，而且它们可能给代理人带来不同的收益。

格林指出解"必须"是这两种类型之一，但他自己的第三个例子（下面将研究）却存在既非这两种类型的最优计划。

当信号也只有两个时，规划者要么得到全部，要么一无所获。

```{prf:corollary} 两个行动和两个信号（格林定理2）
:label: it_thm_two_signals

若 $K = N = 2$，则要么 $\hat v(\Lambda) = v(\Lambda)$（规划者达到她的最优结果），要么 $\hat v(\Lambda) = v_0$（代理人的信息对她而言毫无价值）。
```

如果两位参与者的最优计划一致，规划者就宣布她自己的计划，代理人如实报告，于是 $\hat v(\Lambda) = v(\Lambda)$。

现在假设这两个计划不同，并且为了使论证简短，假设任何一方都从不处于无差异状态。

根据 {prf:ref}`it_thm_two_actions`，$\hat v(\Lambda) = \max\{v_0, v_A(\Lambda)\}$，因此只需证明 $v_A(\Lambda) \leq v_0$。

* 若代理人在两个信号之后都偏好同一个行动，他的最优计划只采取单一行动，这对规划者而言至多值 $v_0$。
* 若代理人在两个信号上的偏好行动不同，而规划者的偏好行动相同，则 $v_A(\Lambda) \leq v(\Lambda) = v_0$。
* 若双方在两个信号上的偏好行动都不同，那么它们是相反的，因此代理人的最优计划在两个信号之后都会让规划者得到她不太喜欢的那个行动，而在两个信号之后都采取其中任一行动的表现都更好。

每种情况下都有 $\hat v(\Lambda) = v_0$。

这就证明了该推论，并使其更加精确：当有两个行动和两个信号时，除非双方拥有共同的最优计划，否则代理人的信息对规划者而言毫无价值。

为了用数值方式验证这两个结果，我们抽取一些随机经济情形，其中双方的最优计划在至少一个信号之后不同，否则规划者会显然地达到她的最优结果。

两位参与者的效用是正相关的，因此他们经常达成一致，但并非总是如此。

```{code-cell} ipython3
def random_economy(M, K, N, rng, corr=0.8):
    """Random prior, likelihood matrix, and utilities with correlation corr."""
    π = rng.dirichlet(np.ones(M))
    Λ = rng.dirichlet(np.ones(N), size=M)
    U = rng.normal(size=(K, M))
    W = corr * U + np.sqrt(1 - corr**2) * rng.normal(size=(K, M))
    return U, W, Λ, π


def plans_differ(U, W, Λ, π):
    """True if the players' preferred actions differ after some signal."""
    _, P = posteriors(Λ, π)
    return np.any((U @ P).argmax(axis=0) != (W @ P).argmax(axis=0))


def agent_plan_value(U, W, Λ, π):
    """Planner's expected utility when the agent's preferred action is taken."""
    q, P = posteriors(Λ, π)
    k = (W @ P).argmax(axis=0)
    return q @ (U @ P)[k, np.arange(len(q))]


rng = np.random.default_rng(1234)

gaps, agent_wins = [], 0
while len(gaps) < 500:
    M, N = rng.integers(2, 5), rng.integers(2, 6)
    U, W, Λ, π = random_economy(M, 2, N, rng)
    if not plans_differ(U, W, Λ, π):
        continue
    v_hat = solve_planner(U, W, Λ, π)[1]
    v_0, v_A = no_info_value(U, π), agent_plan_value(U, W, Λ, π)
    gaps.append(abs(v_hat - max(v_0, v_A)))
    agent_wins += v_A > v_0 + 1e-10
print(f"K = 2: largest gap {max(gaps):.1e}; letting the agent have his way "
      f"beats ignoring him in {agent_wins} of {len(gaps)} draws")

gaps = []
while len(gaps) < 500:
    U, W, Λ, π = random_economy(rng.integers(2, 5), 2, 2, rng)
    if not plans_differ(U, W, Λ, π):
        continue
    gaps.append(abs(solve_planner(U, W, Λ, π)[1] - no_info_value(U, π)))
print(f"K = N = 2: largest gap between v_hat and v_0 {max(gaps):.1e}")
```

在每次抽样中，线性规划的值都等于该定理所预测的 $\max\{v_0, v_A(\Lambda)\}$（在舍入误差范围内）。

当有两个行动和两个信号时，这个值总是等于 $v_0$，正如 {prf:ref}`it_thm_two_signals` 的精确化版本所说。

当有更多信号时，听任代理人行事有时会优于忽略他，但在大多数抽样中并非如此。

## 用被占优的行动作为威胁

当只有两个行动时，一个抽签只是 $a_1$ 的一个概率，因此代理人对抽签的排序由他偏好哪个行动所固定，规划者最多只能做到忽略他或顺从他。

当有三个行动时，即使代理人对行动本身的排序不变，他对抽签的排序也会随着他的后验而变化，这使得规划者能够比这两种做法都做得更好。

格林的第一个例子有两个状态、两个信号和三个行动。

先验是均匀的，并且

$$
\Lambda = \begin{bmatrix} 3/4 & 1/4 \\ 1/4 & 3/4 \end{bmatrix},
\qquad
U = \begin{bmatrix} -2 & 3 \\ 0 & 0 \\ -1 & -1 \end{bmatrix},
\qquad
W = \begin{bmatrix} 0 & -1 \\ 2 & 0 \\ 0 & -4 \end{bmatrix} .
$$

$\theta_1$ 的后验概率在 $y_1$ 之后为 $3/4$，在 $y_2$ 之后为 $1/4$。

如果规划者亲自看到信号，她会在 $y_1$ 之后采取 $a_2$，在 $y_2$ 之后采取 $a_1$。

代理人在任一信号之后都偏好 $a_2$，所以如果规划者宣布她的最优计划，一个看到 $y_2$ 的代理人就会报告 $y_1$。

行动 $a_3$ 在两个状态下都给规划者带来 $-1$，而 $a_2$ 带来 $0$，所以一个亲自看到信号的规划者绝不会采取 $a_3$。

让我们求解规划者的线性规划。

```{code-cell} ipython3
π_ex1 = np.array([1/2, 1/2])
Λ_ex1 = np.array([[3/4, 1/4],
                  [1/4, 3/4]])
U_ex1 = np.array([[-2.0, 3.0],
                  [0.0, 0.0],
                  [-1.0, -1.0]])
W_ex1 = np.array([[0.0, -1.0],
                  [2.0, 0.0],
                  [0.0, -4.0]])

Z_ex1, v_ex1, _ = solve_planner(U_ex1, W_ex1, Λ_ex1, π_ex1)
print("optimal plan Z =\n", Z_ex1)
print(f"planner: {v_ex1:.4f}"
      f"   first best: {bayes_value(U_ex1, Λ_ex1, π_ex1):.4f}"
      f"   no information: {no_info_value(U_ex1, π_ex1):.4f}")
```

最优计划在报告 $y_2$ 之后采取 $a_1$；在报告 $y_1$ 之后以 $9/14$ 的概率采取 $a_2$，以 $5/14$ 的概率采取被占优的行动 $a_3$。

规划者的期望效用为 $\hat v(\Lambda) = 39/56 \approx 0.696$，介于她忽略代理人所得的 $v_0 = 0.5$ 和她亲自看到信号所得的 $v(\Lambda) = 0.875$ 之间。

格林给出的数值是 $0.696$ 以及概率 $0.643$ 和 $0.357$。

为什么被占优的行动会有帮助？

该计划必须阻止一个已经看到 $y_2$ 的代理人报告 $y_1$，因此它要对报告 $y_1$ 附带一个这种代理人厌恶的行动。

将每个候选威胁与 $a_2$（规划者在 $y_1$ 之后想要采取的行动）进行比较。

在真实后验 $3/4$ 下，行动 $a_3$ 使规划者付出 $1$ 的代价，而在撒谎代理人的后验 $1/4$ 下，它使他付出 $1/2 - (-3) = 7/2$ 的代价。

行动 $a_1$ 只会使规划者付出 $3/4$ 的代价，但只会使撒谎的代理人付出 $1/2 - (-3/4) = 5/4$ 的代价。

所以 $a_3$ 相对于使规划者付出的每单位代价，能使撒谎者承受 $7/2$ 的损失，而 $a_1$ 只能是 $5/3$。

这是一种更便宜的威胁，因为它在代理人认为可能性较大的状态 $\theta_2$ 中对代理人伤害最大，而一个看到 $y_2$ 的代理人恰好认为 $\theta_2$ 的可能性较大。

一个看到 $y_2$ 的代理人若如实报告能获得 $-3/4$，所以威胁必须使他声称看到 $y_1$ 所获得的效用降低 $1/2 - (-3/4) = 5/4$。

用 $a_3$ 需要的概率为 $(5/4)/(7/2) = 5/14$，正好使他在如实报告和撒谎之间无差异；用 $a_1$ 则需要概率为一。

随机化和被占优的行动需要同时发挥作用。

确定性计划行不通：无论收到哪个信号，代理人对行动的排序都相同，依次为 $a_2$、然后 $a_1$、然后 $a_3$，所以任何在两个报告之后采取不同行动的计划都会诱使一个只看到两个信号之一的代理人撒谎，而在两个报告之后都采取相同行动的最优计划只值 $v_0$。

如果没有 $a_3$，就只剩下两个行动，代理人在两个后验下都偏好 $a_2$，因此按照 {prf:ref}`it_thm_two_actions` 背后的论证，如实报告迫使该计划对两个报告一视同仁；这样的最优计划在两个报告之后都采取 $a_1$，只值 $v_0$。

```{code-cell} ipython3
v_no_threat = solve_planner(U_ex1[:2], W_ex1[:2], Λ_ex1, π_ex1)[1]
print(f"planner's value without a_3: {v_no_threat:.4f}")
```

下图将每位参与者从每个行动得到的期望效用，相对于 $\theta_1$ 的后验概率作图。

虚线表示该计划对报告 $y_1$ 所附加的抽签的期望效用，点状竖线标出了两个后验值的位置。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Expected utilities in Green's first example
    name: fig-it-dominated
---
p_grid = np.linspace(0, 1, 200)
lottery = Z_ex1[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, who, loc in zip(axes, (U_ex1, W_ex1), ("planner", "agent"),
                             ("upper right", "lower right")):
    for k in range(3):
        ax.plot(p_grid, mat[k, 0] * p_grid + mat[k, 1] * (1 - p_grid),
                lw=2, label=f"$a_{k+1}$")
    mix = lottery @ mat
    ax.plot(p_grid, mix[0] * p_grid + mix[1] * (1 - p_grid),
            "k--", lw=2, label="plan after report $y_1$")
    for p_n in (1/4, 3/4):
        ax.axvline(p_n, color="0.6", lw=1, ls=":")
    ax.set_xlabel(r"posterior probability of $\theta_1$")
    ax.set_ylabel(f"expected utility of the {who}")
    ax.legend(loc=loc)

plt.tight_layout()
plt.show()
```

在右侧面板中，虚线恰好在后验 $1/4$ 处与代理人关于 $a_1$ 的直线相交，而正是在这个后验上，看到 $y_2$ 的代理人感到无差异。

在后验 $3/4$ 处，虚线明显高于 $a_1$ 的直线，因此一个看到 $y_1$ 的代理人会严格偏好如实报告。

在左侧面板中，虚线位于 $a_2$ 的直线之下。

在后验 $3/4$ 处的差距为 $5/14$，这是规划者在报告 $y_1$ 之后所受的损失。

由于 $y_1$ 出现的概率为 $1/2$，她获取真相的期望代价为 $5/28 = v(\Lambda) - \hat v(\Lambda)$。

## 如实但粗略的报告

格林的第二个例子展示了激励使规划者损失信息的另一种方式。

存在三个等概率的信号，格林直接给出了期望效用矩阵：

$$
WP = \begin{bmatrix} 1 & 3/2 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix},
\qquad
UP = \begin{bmatrix} 5 & -1 & -1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} .
$$

我们可以用三个等概率状态、均匀先验，以及完全信息实验 $\Lambda = I$（此时 $P = I$）来表示这些内容。

```{code-cell} ipython3
π_ex2 = np.full(3, 1/3)
Λ_ex2 = np.eye(3)
U_ex2 = np.array([[5.0, -1.0, -1.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
W_ex2 = np.array([[1.0, 1.5, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

Z_ex2, v_ex2, _ = solve_planner(U_ex2, W_ex2, Λ_ex2, π_ex2)
print("optimal plan Z =\n", Z_ex2)
print(f"planner: {v_ex2:.4f}"
      f"   first best: {bayes_value(U_ex2, Λ_ex2, π_ex2):.4f}"
      f"   no information: {no_info_value(U_ex2, π_ex2):.4f}")
```

规划者只想在 $y_1$ 之后采取 $a_1$，但一个看到 $y_2$ 的代理人也喜欢 $a_1$，对他而言 $a_1$ 的价值是 $3/2$，比其他任何行动都高。

假设在报告 $y_1$ 之后该计划以概率 $\alpha$ 采取 $a_1$，其余的概率都放在 $a_3$ 上（这是看到 $y_2$ 的代理人最不喜欢的行动），而在报告 $y_2$ 之后它以概率 $\beta$ 采取 $a_1$，否则采取 $a_2$。

一个看到 $y_2$ 的代理人如实报告能获得 $1 + \beta/2$，而声称看到 $y_1$ 能获得 $3\alpha/2$，因此真实性要求 $\beta \geq 3\alpha - 2$。

所以，在报告 $y_2$ 之后绝不采取 $a_1$ 的最优计划，在报告 $y_1$ 之后以概率 $2/3$ 采取 $a_1$，这对规划者而言值 $10/9$。

超过 $\alpha = 2/3$ 之后，每增加一单位在报告 $y_1$ 后对 $a_1$ 的概率，规划者获得 $5$，但也会迫使三单位的概率放在报告 $y_2$ 后的 $a_1$ 上，每单位使她付出 $1$ 的代价，所以她会把 $\alpha$ 一直提高到 $1$。

因此最优计划在报告 $y_1$ 和 $y_2$ 之后给出相同的抽签，即肯定采取 $a_1$，线性规划也证实了这个族以外没有任何计划能做得更好。

代理人如实报告，但规划者把前两个信号一视同仁，获得 $4/3$，介于忽略代理人所得的 $1$ 和最优结果所得的 $5/3$ 之间。

$Z$ 的第三行并不唯一确定：在报告 $y_3$ 之后规划者在 $a_2$ 和 $a_3$ 之间无差异，代理人的期望效用取决于她选取哪一个。

```{code-cell} ipython3
for ties in ("agent_worst", "agent_best"):
    agent = solve_planner(U_ex2, W_ex2, Λ_ex2, π_ex2, ties=ties)[2]
    print(f"{ties}: agent {agent:.4f}")
```

在规划者的各个最优计划之间，代理人的期望效用从 $5/6$ 到 $7/6$ 不等。

## 比较实验

### 通过后验表达的布莱克威尔序

为了探讨规划者是否希望她的代理人获得更好的信息，格林需要一种比较实验的方法。

设 $\Lambda'$ 为另一个实验，具有 $N'$ 个信号 $y'_1, \ldots, y'_{N'}$，并让撇号标注由它计算出来的一切，比如 $q'$ 和 $P'$。

根据 {prf:ref}`thm-blackwell`，对于任意效用矩阵 $U$ 和任意先验 $\pi$，$v(\Lambda) \geq v(\Lambda')$ 当且仅当存在某个 $N \times N'$ 马尔可夫矩阵 $B$，使得 $\Lambda' = \Lambda B$。

当该条件成立时，称 $\Lambda$（弱）**信息量更大**，而 $\Lambda'$ 是 $\Lambda$ 的一个**扰动**。

```{note}
在 {doc}`blackwell_kihlstrom` 中，实验用 $\mu$ 而非 $\Lambda$ 表示，扰动矩阵是随机变换 $Q$ 而非 $B$；在那里 $B(\mu, A, u)$ 表示可达期望效用向量所构成的集合。

该讲座还用 $N$ 来计数状态、用 $M$ 来计数信号，将先验写作 $p$，并用 $U$ 表示不确定性函数而非效用矩阵。
```

正如格林所指出的，布莱克威尔条件可以通过求解关于 $B$ 的线性不等式来检验，这比对所有效用矩阵比较决策者要容易得多。

不过，为了格林的目的，一种以后验表达的版本更有用，因为代理人的激励取决于他的后验。

{cite:t}`MarschakMiyasawa1968` 证明，对于一个具有完全支撑的固定先验，$\Lambda$ 比 $\Lambda'$ 信息量更大当且仅当存在两个信号的联合分布 $J = [J_{nn'}]$，其边缘为 $q$ 和 $q'$，满足

$$
\sum_{n=1}^N p_{mn} \, J_{nn'} = p'_{mn'} \, q'_{n'}
\quad \text{对所有 } m \text{ 和 } n' .
$$ (it_barycenter)

于是信息量较小的实验的每个后验都是信息量较大的实验诸后验的一个平均值，权重为 $J_{nn'}/q'_{n'}$，即给定 $y'_{n'}$ 时 $y$ 的条件分布。

我们称 {eq}`it_barycenter` 为**平均化条件**。

这正是 {doc}`blackwell_kihlstrom` 中基尔斯特罗姆方法所强调的性质：在 $\Lambda$ 下的后验分布是在 $\Lambda'$ 下的后验分布的保均值展开。

格林用这些权重所构成的矩阵 $C = J \operatorname{diag}(q')^{-1}$ 来表述该条件，其各列都是概率向量：

$$
P' = P C
\qquad \text{以及} \qquad
q = C q' .
$$ (it_mm)

这是格林的方程(12)，他取自马尔沙克和宫泽。

给定 $J$，矩阵 $B = \operatorname{diag}(q)^{-1} J$（即给定 $y$ 时 $y'$ 的条件分布）就是布莱克威尔的扰动矩阵。

这是格林的方程(13)，即马尔沙克和宫泽的方程(8.23)。

为了验证 $\Lambda' = \Lambda B$，利用 {eq}`it_bayes` 中的 $\lambda_{mn}/q_n = p_{mn}/\pi_m$，再利用 {eq}`it_barycenter`：

$$
(\Lambda B)_{mn'} = \frac{1}{\pi_m} \sum_n p_{mn} J_{nn'} = \frac{p'_{mn'} \, q'_{n'}}{\pi_m} = \lambda'_{mn'} .
$$

### 一个运输问题

寻找 $J$ 是一个线性可行性问题，我们用 {doc}`线性规划 <intro:lp_intro>` 的工具来求解。

如果没有平均化条件，$J$ 可以是 {doc}`opt_transport` 中任意一个具有边缘 $q$ 和 $q'$ 的联合分布所构成的运输方案，独立耦合 $J = q q'^\top$ 总是可行的。

平均化条件是一组额外的线性约束，它使得该运输方案成为这两个后验分布的一个**鞅耦合**。

这里没有需要最小化的成本，所以我们给 `linprog` 传入零目标函数。

列和约束由 {eq}`it_barycenter` 蕴含，因为 $P$ 的每一列中的后验之和为一，所以我们只施加行和约束以及 {eq}`it_barycenter`。

```{code-cell} ipython3
def garbling(Λ, Λ_prime, π):
    """
    Look for a joint distribution J of the signals of Λ and Λ_prime under
    which every posterior of Λ_prime is the J-weighted average of
    posteriors of Λ.

    Returns Blackwell's Markov matrix B, with Λ_prime = Λ B, or None if
    the solver reports that no such J exists.
    """
    if np.any(π <= 0):
        raise ValueError("the prior must give every state positive probability")
    q, P = posteriors(Λ, π)
    q_prime, P_prime = posteriors(Λ_prime, π)
    N, N_prime = len(q), len(q_prime)

    # Entry n * N_prime + j of the decision vector is J[n, j]
    A_eq = np.vstack([np.kron(np.eye(N), np.ones(N_prime)),  # row sums
                      np.kron(P, np.eye(N_prime))])           # averaging
    b_eq = np.concatenate([q, (P_prime * q_prime).ravel()])
    # As in solve_planner, we switch off presolve, so that a spurious
    # infeasibility report cannot pass for a missing garbling
    res = linprog(np.zeros(N * N_prime), A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs",
                  options={"presolve": False})
    if res.status == 2:                  # infeasible
        return None
    if res.status != 0:
        raise RuntimeError(res.message)
    J = np.maximum(res.x, 0.0).reshape(N, N_prime) + 0.0
    return J / q[:, None]
```

与 {doc}`blackwell_kihlstrom` 中的最小二乘搜索不同，这个函数要么找到一个扰动矩阵（精确到求解器的数值容差），要么报告不存在这样的扰动矩阵。

## 更好的信息可能损害规划者

### 格林的第三个例子

当代理人的信息改善时，两件事会发生变化。

规划者可以凭借更好的信号做出更好的决策，但更好的信号也会改变代理人愿意如实作出哪些报告。

格林的第三个例子表明第二种效应可能占主导地位。

存在两个具有均匀先验的状态、两个行动，以及

$$
U = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix},
\qquad
W = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix} .
$$

当 $\theta_1$ 的后验概率超过 $2/3$ 时，规划者偏好 $a_1$；当其超过 $1/3$ 时，代理人偏好 $a_1$。

粗略实验 $\Lambda'$ 有两个信号，对 $\theta_1$ 的后验分别为 $3/4$ 和 $1/4$，各自概率为 $1/2$。

精细实验 $\Lambda$ 有三个信号，后验分别为 $3/4$、$1/2$ 和 $0$，概率分别为 $1/2$、$1/4$ 和 $1/4$。

下面的函数根据给定的 $\theta_1$ 后验列表及其概率构造一个两状态的似然矩阵。

这些概率之和必须为一，且后验的加权平均必须等于 $\theta_1$ 的先验概率；否则 $\Lambda$ 的各行之和不为一。

```{code-cell} ipython3
def experiment(post, probs, π):
    """Two-state likelihood matrix with the given posteriors of θ_1."""
    post, probs = np.asarray(post), np.asarray(probs)
    if not (np.isclose(probs.sum(), 1) and np.isclose(probs @ post, π[0])):
        raise ValueError("the probabilities must sum to one and the "
                         "posteriors must average to the prior")
    P = np.array([post, 1 - post])
    return P * probs / π[:, None]


π_ex3 = np.array([1/2, 1/2])
U_ex3 = np.array([[2.0, 0.0],
                  [0.0, 4.0]])
W_ex3 = np.array([[4.0, 0.0],
                  [0.0, 2.0]])
Λ_coarse = experiment([3/4, 1/4], [1/2, 1/2], π_ex3)
Λ_fine = experiment([3/4, 1/2, 0], [1/2, 1/4, 1/4], π_ex3)

print("B =\n", garbling(Λ_fine, Λ_coarse, π_ex3))
print("reverse garbling:", garbling(Λ_coarse, Λ_fine, π_ex3))
```

扰动矩阵 $B$ 把精细实验的第二个和第三个信号合并起来，因此精细实验信息量更大，而没有任何扰动矩阵可以反方向进行。

让我们比较规划者的价值。

```{code-cell} ipython3
for name, Λ in (("coarse", Λ_coarse), ("fine", Λ_fine)):
    v_hat = solve_planner(U_ex3, W_ex3, Λ, π_ex3)[1]
    print(f"{name}: planner {v_hat:.4f}, "
          f"first best {bayes_value(U_ex3, Λ, π_ex3):.4f}")
print(f"no information: {no_info_value(U_ex3, π_ex3):.4f}")
```

在粗略实验中，两位参与者在两个信号之后都达成一致，因为 $3/4$ 同时超过两个临界值，而 $1/4$ 同时低于两个临界值。

所以规划者宣布她自己的最优计划，代理人如实报告，双方都获得 $9/4$。

精细实验把后验为 $1/4$ 的信号拆分为后验为 $1/2$ 和 $0$ 的两个信号。

在后验 $1/2$ 处，双方意见相左：规划者想要 $a_2$，代理人想要 $a_1$。

根据 {prf:ref}`it_thm_two_actions`，规划者要么忽略代理人，要么听任他行事。

对规划者而言，每种选择都值 $2$，少于 $9/4$，所以规划者宁愿她的代理人知道得更少一些。

事实上，介于这两者之间的任何计划也都是最优的，代理人的收益在这些计划之间有所不同。

```{code-cell} ipython3
for ties in ("agent_worst", "agent_best"):
    Z, v_hat, agent = solve_planner(U_ex3, W_ex3, Λ_fine, π_ex3, ties=ties)
    print(f"{ties}: planner {v_hat:.4f}, agent {agent:.4f}, plan\n{Z}")
```

每一个在后验为 $0$ 的信号之后采取 $a_2$、而在另外两个信号之后以共同概率 $z \in [0, 1]$ 采取 $a_1$ 的计划，对规划者而言都值 $2$。

代理人在这些计划下的期望效用为 $1 + 3z/2$，从规划者忽略他时的 $1$ 一直到她听任他行事时的 $5/2$。

代理人也有理由秘密地去获取这种更精细的信息。

假设规划者相信代理人拥有粗略实验，并为此宣布了她的最优计划，而代理人实际上观察到的是精细信号。

在后验为 $1/2$ 的信号之后，代理人偏好 $a_1$，所以他报告后验为 $3/4$ 的粗略信号。

他的期望效用从 $9/4$ 上升到 $5/2$，而规划者的期望效用则从 $9/4$ 下降到 $2$。

这说明了一个一般性要点：对于一个固定的计划而言，代理人是从 $Z$ 各行所构成的菜单中进行选择的一位单一决策者，因此按照布莱克威尔定理，更好的信息永远不会损害他。

当更好的信息确实损害了代理人时，那是因为规划者在知道代理人所知道的情况后，选择了一个不同的计划。

{cite:t}`Postlewaite1982` 就他自己的例子提出了关于秘密获取信息的这一要点，并提出了他所称的"关于信息的信息"问题：只有当规划者知道代理人没有秘密地改善信息时，她才能依赖那个粗略实验。

### 一个精细化族

我们可以通过把后验为 $1/4$ 的粗略信号拆分为后验为 $s$ 和 $0$ 的两个信号（其中 $s$ 介于 $1/4$ 和 $1$ 之间），更全面地看清这种权衡。

后验为 $s$ 的信号的概率为 $1/(8s)$，这保持了平均后验等于先验。

提高 $s$ 会使该实验在布莱克威尔意义下信息量更大，下面的代码沿网格检验了这一点。

网格从刚好高于 $1/4$ 的地方开始，因为在 $s = 1/4$ 时，后验为 $0$ 的信号概率为零，该精细化就是粗略实验本身。

```{code-cell} ipython3
def refined(s):
    """Green's example with the posterior 1/4 split into s and 0."""
    w = 1 / (8 * s)
    return experiment([3/4, s, 0], [1/2, w, 1/2 - w], π_ex3)


s_grid = np.linspace(0.26, 1, 150)
nested = all(garbling(refined(s_hi), refined(s_lo), π_ex3) is not None
             for s_lo, s_hi in zip(s_grid[:-1], s_grid[1:]))
print("each experiment is more informative than the previous one:", nested)

v_hat_grid = np.array([solve_planner(U_ex3, W_ex3, refined(s), π_ex3)[1]
                       for s in s_grid])
v_grid = np.array([bayes_value(U_ex3, refined(s), π_ex3) for s in s_grid])
```

该图把规划者从线性规划中得到的价值 $\hat v$ 以及她的最优价值 $v$ 相对于 $s$ 作图。

点状竖线标出了代理人的临界值 $1/3$ 和规划者的临界值 $2/3$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Planner's value as information improves
    name: fig-it-refine
---
fig, ax = plt.subplots()
ax.plot(s_grid, v_grid, lw=5, alpha=0.4, label="planner observes the signal")
# v_hat jumps at s = 1/3, so we draw the two pieces separately
for piece, label in ((s_grid < 1/3, "agent observes the signal"),
                     (s_grid > 1/3, None)):
    ax.plot(s_grid[piece], v_hat_grid[piece], color="C1", lw=2, label=label)
ax.axhline(no_info_value(U_ex3, π_ex3), color="0.6", lw=1, ls="--",
           label="no information")
for cutoff in (1/3, 2/3):
    ax.axvline(cutoff, color="0.6", lw=1, ls=":")
ax.set_xlabel("posterior $s$ of the refined signal")
ax.set_ylabel("planner's expected utility")
ax.legend()
plt.show()
```

随着 $s$ 的提高，最优价值 $v$ 从不下降，正如布莱克威尔定理所要求的那样，但规划者的价值 $\hat v$ 却会下降。

当 $s < 1/3$ 时，无论哪个精细化信号出现，双方都想要 $a_2$，因此该精细化既无害也无用。

当 $1/3 < s < 2/3$ 时，双方在后验为 $s$ 的信号之后意见相左。

此时规划者要么忽略代理人（值 $2$），要么听任他行事（值 $3 - 1/(2s)$）。

所以 $\hat v$ 在 $s = 1/3$ 处从 $9/4$ 跳降到 $2$，一直停留在 $2$ 直到 $s = 1/2$，然后随着 $s$ 趋近 $2/3$ 又重新攀升回 $9/4$ 附近。

只有当 $s > 2/3$ 时，规划者和代理人再次达成一致，这种精细化才能改善规划者的决策。

### 只有两个信号的实验

在格林的例子中，损害来自于把一个信号拆分为两个的精细化。

他的最后一个结果说，当两个实验都是**二项信道**（格林用这个术语表示只有两个信号的实验）时，这种情况不会发生。

```{prf:theorem} 二项信道（格林定理4）
:label: it_thm_binary

若 $\Lambda$ 比 $\Lambda'$ 信息量更大，且二者都只有两个信号，则对任意数量的状态和行动都有 $\hat v(\Lambda) \geq \hat v(\Lambda')$。
```

以下是一个证明。

设 $p_1$ 和 $p_2$ 为 $\Lambda$ 的后验，而 $p'_1$ 和 $p'_2$ 为 $\Lambda'$ 的后验。

根据 {eq}`it_barycenter`，每个 $p'_{n'}$ 都是 $p_1$ 和 $p_2$ 的一个平均值，所以它们都位于连接 $p_1$ 和 $p_2$ 的线段上。

若 $p'_1 = p'_2$，则 $\Lambda'$ 是无信息的，$\hat v(\Lambda') = v_0$，不需要再证明什么。

否则，按 $p'_1$ 比 $p'_2$ 更接近 $p_1$ 的方式给 $\Lambda'$ 的信号标号。

令 $Z$ 为 $\Lambda'$ 下的某个最优计划，$\ell_1$ 和 $\ell_2$ 为其各行，即收到报告 $y'_1$ 和 $y'_2$ 之后的抽签。

对 $\Lambda$ 使用同一个计划，即在收到报告 $y_n$ 之后采取抽签 $\ell_n$。

在后验 $p$ 处，代理人从抽签 $\ell_1$ 相对于抽签 $\ell_2$ 所得的收益为 $\phi(p) = (\ell_1 - \ell_2) W p$，规划者的收益为 $\psi(p) = (\ell_1 - \ell_2) U p$。

二者都是关于 $p$ 线性的。

1. $\Lambda'$ 下的如实报告意味着 $\phi(p'_1) \geq 0 \geq \phi(p'_2)$。四个后验按 $p_1, p'_1, p'_2, p_2$ 的顺序位于该线段上，所以在 $p'_1$ 和 $p'_2$ 处取这些符号的线性函数在 $p_1$ 和 $p_2$ 处也保持这些符号。因此 $\phi(p_1) \geq 0 \geq \phi(p_2)$，该计划使得在 $\Lambda$ 下如实报告也是最优的。
2. $Z$ 在 $\Lambda'$ 下的最优性蕴含 $\psi(p'_1) \geq 0 \geq \psi(p'_2)$。因为若 $\psi(p'_1) < 0$，规划者本可以在两个报告之后都使用抽签 $\ell_2$ 来做得更好，这是一个忽略代理人的计划，因而满足如实报告约束；在 $\psi(p'_2) > 0$ 的情形下类似。根据第1步的论证，$\psi(p_1) \geq 0 \geq \psi(p_2)$。
3. 设 $J$ 为 {eq}`it_barycenter` 中两个信号的联合分布。由于 $q'_{n'} p'_{n'} = \sum_n J_{nn'} \, p_n$ 且 $q_n = \sum_{n'} J_{nn'}$，规划者在 $\Lambda$ 下从该计划所得的期望效用减去在 $\Lambda'$ 下所得的期望效用为

   $$
   \sum_n q_n \ell_n U p_n - \sum_{n'} q'_{n'} \ell_{n'} U p'_{n'}
   = \sum_{n, n'} J_{nn'} (\ell_n - \ell_{n'}) U p_n
   = J_{12} \, \psi(p_1) - J_{21} \, \psi(p_2) \;\geq\; 0 .
   $$

在 $\Lambda$ 下，规划者的表现至少能和这个计划一样好，所以 $\hat v(\Lambda) \geq \hat v(\Lambda')$。

```{note}
格林的证明走的是同一条路线，但有两处漏洞。

首先，它假定对信号的标号方式使得扰动矩阵 $B$ 的每一行的最大元素都在对角线上。

这并非总是可能的：对于 $B = \begin{bmatrix} 0.9 & 0.1 \\ 0.6 & 0.4 \end{bmatrix}$，无论如何标号，两行的最大元素都在同一列上。

论证所需的标号方式正是上面所用的那种，它等价于 $b_{11} \geq b_{21}$，即 $\det B \geq 0$。

其次，格林的最后一步把只对代理人成立的符号性质（由他前面的引理证明）用在了规划者的收益上；上面的第2步则由 $Z$ 的最优性补充了这一点。
```

为了用数值方式验证该定理，我们随机抽取最多四个状态、三个或四个行动的二项信道。

我们只保留那些该定理起到实际作用的抽样。

如果 $\hat v(\Lambda') = v_0$ 或 $\hat v(\Lambda) = v(\Lambda)$，那么结论仅由 $v_0 \leq \hat v \leq v$ 和布莱克威尔定理即可得出，因此我们跳过这些抽样。

当只有两个行动时，根据 {prf:ref}`it_thm_two_signals`，$\hat v$ 要么是 $v_0$ 要么是 $v$，所以若定理成立，每次抽样都会被跳过；这就是我们使用三个或四个行动的原因。

```{code-cell} ipython3
rng = np.random.default_rng(4321)

gains, tries = [], 0
while len(gains) < 100:
    tries += 1
    M, K = rng.integers(2, 5), rng.integers(3, 5)
    U, W, Λ, π = random_economy(M, K, 2, rng)
    if not plans_differ(U, W, Λ, π):         # then v_hat(Λ) = v(Λ)
        continue
    Λ_prime = Λ @ rng.dirichlet(np.ones(2), size=2)
    v_hat_prime = solve_planner(U, W, Λ_prime, π)[1]
    if v_hat_prime < no_info_value(U, π) + 1e-8:
        continue
    v_hat = solve_planner(U, W, Λ, π)[1]
    if v_hat > bayes_value(U, Λ, π) - 1e-8:
        continue
    gains.append(v_hat - v_hat_prime)
print(f"{len(gains)} informative draws out of {tries}")
print(f"smallest gain from better information: {min(gains):.4f}")
```

在这些抽样的每一次中，规划者都从更好的信息中获益。

当更精细的实验有三个或更多信号时，该定理不再适用，但损失仍然要求双方在精细实验的某个后验处意见相左，正如在 $1/3 < s < 2/3$ 时的 $s$ 处那样；否则规划者就能达到 $\hat v(\Lambda) = v(\Lambda) \geq v(\Lambda') \geq \hat v(\Lambda')$。

## 评论者补充的内容

{cite:t}`Postlewaite1982` 构建了四状态的例子，即他的情形I到情形IV，其中一个信息量更大的实验要么使双方都受益，要么使双方都受损，要么使代理人受益而规划者受损，要么使规划者受益而代理人受损。

{ref}`it_ex_postlewaite` 用线性规划对这些情形做了检验。

在我们的随机抽样中，情形I和情形IV总是如他所说那样出现。

在情形II中，正如波斯尔思韦特所假设的那样取 $h$ 接近 $j$ 时，代理人在每次抽样中都受损，但规划者只在大约一半的抽样中受损。

在情形III中，代理人在每次抽样中都是受损而非受益，而规划者则在某些抽样中受益、在另一些抽样中受损。

波斯尔思韦特问为什么代理人不能干脆忽略更好的信息，他的回答是：规划者知道代理人是知情的，可以设计她的计划，使得一个忽略信息的代理人处境会更差。

正如我们在格林第三个例子之后所看到的，在一个固定的计划下，更好的信息不可能损害代理人，所以"处境更差"总是弱意义上成立的。

在他的例子中，这一点只是弱意义上成立的：{ref}`it_ex_postlewaite` 研究了一族计划，该族在情形II和情形III的例子中都包含规划者的最优计划，并说明在该族中她所偏好的计划下，一个忽略更精细信息的代理人所得到的恰好等于他利用该信息所得到的。

代理人之所以受损，仅仅是因为规划者知道他信息更充分而选择了不同的计划；这就是为什么波斯尔思韦特补充说，如果代理人能说服规划者相信自己没有获得更好的信息，他就会愿意承诺使用信息量较小的实验。

波斯尔思韦特还问是谁选择了实验，以及如果不是规划者而是代理人先行承诺会发生什么。

他断言很容易构造出双方在领导者-跟随者角色互换后都变得更好的例子；{ref}`it_ex_reverse` 在代理人承诺使用一个确定性应答规则（如格林所定义的那样）时证实了这一点。

{cite:t}`Riley1982` 用期望效用相对于后验的图形（类似于 {numref}`fig-it-dominated`）解释了格林的例子，并给出了一个两状态的例子，在该例子中解开一个被扰动的消息会使双方都变得更差。

他总结道，"除非以某种方式精细化这个模型，否则我们不能指望得到的不仅仅是大量的特殊情形"，并提出了两种精细化方式。

一种是向代理人支付"取决于最终自然状态的费用"。

另一种是考虑到"信息交换的重复性质"，因为经纪人与客户、医生与病人之间的长期关系让双方可以了解对方的利益所在。

重复也可以让声誉在一定程度上替代承诺，不过 {doc}`phillips_credible_policies` 表明声誉可以支撑许多结果，不只是承诺所能带来的那一个。

{ref}`it_ex_riley` 说明，当效用关于货币是线性的时候，仅取决于状态的费用无法改变代理人的报告，而同时取决于报告和状态的报酬会减少（但在赖利的例子中不会消除）更好信息所带来的损害。

## 与其他讲座的联系

{doc}`pricing_information` 研究了一位设计并定价布莱克威尔实验的卖家，她通过一个实验菜单来筛选先验信念为私人信息的买家。

格林的规划者也用菜单来进行筛选，但该菜单提供的是对规划者自身行动的抽签，而私人信息是代理人的信号。

在 {doc}`information_market_equilibrium` 中，外部观察者可以从均衡价格中推断出内部人的私人信息，因此信息是通过市场而不是通过报告来传递的。

在 {doc}`util_rand_resp` 中，随机应答调查的设计者有意使每个答案信息量变小，以使受访者愿意如实回答。

设计者接受一个被扰动但如实的答案，而不是一个更清晰但不如实的答案，这与格林第三个例子中规划者偏好拥有较粗略实验的代理人颇为相似。

格林的规划者无法向代理人支付报酬，所以她是通过扭曲自己的行动来购买如实报告的。

当可以进行转移支付时，就像 {doc}`house_auction` 中的格罗夫斯-克拉克机制以及 {cite:t}`Groves_73` 那样，支付可以承担这项任务；{ref}`it_ex_riley` 将支付加入了格林的模型中。

### 关于沟通与委托的后续研究

格林的模型属于一个关于策略性沟通的模型家族，这些模型在谁对什么做出承诺这一点上有所不同。

在这类文献的语言中，代理人是**发送者**，规划者是**接收者**。

* 在格林的模型中，接收者承诺一个根据报告采取行动的计划。
* 在 {cite:t}`CrawfordSobel1982` 的廉价磋商模型中，没有人做出承诺，接收者会根据她所听到的内容采取最优行动。
* 在 {cite:t}`KamenicaGentzkow2011` 的说服模型中，发送者承诺的是产生信号的那个实验。
* 在波斯尔思韦特的角色互换（在 {ref}`it_ex_reverse` 中研究）中，发送者承诺一个确定性规则，用来报告他所观察到的内容，这相当于把说服限制在 $\Lambda$ 的信号划分范围内；如果该规则可以随机化，那就是把说服限制在 $\Lambda$ 的扰动范围内。

赖利把格林的计划解读为一个菜单，这把它与 {cite:t}`Holmstrom1984` 的委托问题联系起来，在该问题中，委托人选择一组行动，然后让代理人从中挑选一个；而格林的菜单所包含的是对行动的抽签，而非行动本身。

{cite:t}`MelumadShibano1991` 研究了在排除转移支付的情况下，有承诺和无承诺时的沟通问题，{cite:t}`AlonsoMatouschek2008` 则刻画了最优的委托行动集合。

{cite:t}`KovacMylovanov2009` 允许像格林那样的随机化计划，并说明在二次支付函数和一个正则性条件下，最优计划不会随机化。

格林关于规划者可能偏好信息较少的代理人这一发现也有后续研究。

{cite:t}`FischerStocken2001` 说明在廉价磋商中，一个信息不太精确的发送者反而可能使接收者获得更充分的信息，而 {cite:t}`Ivanov2010` 研究了一位在代理人沟通之前限制其信息精度的委托人。

## 结语

格林的模型将看到信号的人与依据信号采取行动的人区分开来。

承诺使规划者能够把代理人的激励问题转化为一个线性规划问题，该问题的解可以随机化、使用被占优的行动作为威胁、并对信号进行合并。

由于激励约束取决于后验，实验之间的比较也要通过后验来进行，而马尔沙克-宫泽的刻画把布莱克威尔序转化为一个运输问题。

核心的教训关乎信息的价值。

布莱克威尔定理为一个亲自看到信号的决策者给实验排序，但一旦必须从别人那里获取信号，一个信息更充分的代理人对规划者而言可能价值更低——除非两个实验都只有两个信号。

## 练习

```{exercise}
:label: it_ex_riley

{cite:t}`Riley1982` 用一个两状态、两行动的例子（在该例子的图C3.4中）展示了更好的信息如何能使双方都受损，他把信号称为"消息"。

以下数字再现了他的图中的模式。

赖利正文中给这两个行动命名的方式与图中相反；我们的标注遵循图中的方式。

先验是均匀的，以 $p$ 表示 $\theta_1$ 的后验概率，规划者从 $a_1$ 得到的期望效用为 $p$，从 $a_2$ 得到 $0.7$，而代理人从 $a_1$ 得到的期望效用为 $p$，从 $a_2$ 得到 $(1 - p)/2$。

在粗略实验中，代理人的后验为 $0.8$ 或 $0.2$，各自概率为 $1/2$。

在精细实验中，后验为 $0.2$ 的信号被解析为后验为 $0.4$ 和 $0$ 的两个信号，各自概率为 $1/4$。

1. 用 `garbling` 确认精细实验比粗略实验信息量更大。然后在两个实验下分别求解规划者的线性规划，比较双方的期望效用，并用 {prf:ref}`it_thm_two_actions` 解释这一结果。
2. 赖利建议向代理人支付"取决于最终自然状态的费用"。假设双方的效用都关于货币是线性的，这样一笔支付会加到代理人的效用上，并从规划者的效用中扣除。证明仅取决于状态的支付无法改变代理人的报告。
3. 现在让规划者在代理人报告 $y_r$ 而状态最终为 $\theta$ 时支付 $t(r, \theta) \geq 0$。把线性规划扩展为同时选择计划和支付方案，并对精细实验求解。当支付只能取决于报告时也做同样的事。支付能否消除更好信息所带来的损害？
```

```{solution-start} it_ex_riley
:class: dropdown
```

以下是一个解答。

对于第1部分，我们构建这两个实验，检验扰动矩阵，并在每种情形下求解线性规划。

```{code-cell} ipython3
π_ri = np.array([1/2, 1/2])
U_ri = np.array([[1.0, 0.0],
                 [0.7, 0.7]])
W_ri = np.array([[1.0, 0.0],
                 [0.0, 0.5]])
Λ_ri_coarse = experiment([0.8, 0.2], [1/2, 1/2], π_ri)
Λ_ri_fine = experiment([0.8, 0.4, 0.0], [1/2, 1/4, 1/4], π_ri)

print("B =\n", garbling(Λ_ri_fine, Λ_ri_coarse, π_ri))
print("reverse garbling:", garbling(Λ_ri_coarse, Λ_ri_fine, π_ri))

for name, Λ in (("coarse", Λ_ri_coarse), ("fine", Λ_ri_fine)):
    Z, v_hat, agent = solve_planner(U_ri, W_ri, Λ, π_ri)
    print(f"{name}: planner {v_hat:.4f}, agent {agent:.4f}, plan\n{Z}")

print(f"fine, agent's first-best plan: "
      f"planner {agent_plan_value(U_ri, W_ri, Λ_ri_fine, π_ri):.4f}, "
      f"agent {bayes_value(W_ri, Λ_ri_fine, π_ri):.4f}")
```

在粗略实验下，双方在两个信号之后都达成一致：在 $0.8$ 时都想要 $a_1$，在 $0.2$ 时都想要 $a_2$。

所以规划者得到 $0.75$，代理人得到 $0.6$。

在精细实验下，双方在后验 $0.4$ 处意见相左，规划者想要 $a_2$，而代理人想要 $a_1$。

根据 {prf:ref}`it_thm_two_actions`，规划者要么听任代理人行事（对她而言值 $0.675$，对他而言值 $0.625$），要么忽略他（值 $0.7$）。

她选择忽略他，并始终采取 $a_2$，这只给代理人留下 $0.25$。

解析该信号使双方都变得更差，正如赖利图中所示。

下面的图具有与赖利图相同的结构。

每个面板中的实线是某一方从这两个行动得到的期望效用，点状竖线标出了先验 $1/2$。

在粗略实验下，规划者在 $0.2$ 处采取 $a_2$，在 $0.8$ 处采取 $a_1$，因此每一方的期望效用是连接该方在这两个后验处效用的虚线弦（实心点）在先验处的高度。

在精细实验下，规划者总是采取 $a_2$，因此每一方的期望效用是该方关于 $a_2$ 的直线在先验处的高度（空心点）。

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, who in zip(axes, (U_ri, W_ri), ("planner", "agent")):
    for k in range(2):
        ax.plot(p_grid, mat[k, 0] * p_grid + mat[k, 1] * (1 - p_grid),
                lw=2, label=f"$a_{k+1}$")
    # coarse experiment: a_2 at the posterior 0.2 and a_1 at the posterior 0.8
    ends = [mat[1] @ [0.2, 0.8], mat[0] @ [0.8, 0.2]]
    ax.plot([0.2, 0.8], ends, "k--", lw=1)
    ax.plot(0.5, np.mean(ends), "ko", label="coarse experiment")
    # fine experiment: a_2 after every signal
    ax.plot(0.5, mat[1] @ [0.5, 0.5], "o", mfc="white", mec="k",
            label="fine experiment")
    ax.axvline(0.5, color="0.6", lw=1, ls=":")
    ax.set_xlabel(r"posterior probability of $\theta_1$")
    ax.set_ylabel(f"expected utility of the {who}")
    ax.legend()
plt.tight_layout()
plt.show()
```

在两个面板中，空心点都位于实心点之下：规划者方面为 $0.70$ 对 $0.75$，代理人方面为 $0.25$ 对 $0.6$。

对于第2部分，一个取决于状态 $\theta_m$ 的支付 $t(\theta_m)$ 会加到 $W$ 第 $m$ 列的每个元素上。

在观察到 $y_n$ 之后，代理人在任何报告下都会多获得 $\sum_m p_{mn} t(\theta_m)$，所以该支付不会改变哪种报告是最优的。

更根本的是，它不会改变代理人在任何状态下对行动的排序，因此它无法减少他与规划者之间的冲突。

要改变代理人的激励，支付必须取决于他所能控制的东西，比如他的报告。

对于第3部分，我们把支付加入决策向量。

取决于报告 $r$ 和状态的支付会在信号 $y_n$ 之后对报告 $y_r$ 的期望效用加上 $\sum_m p_{mn} t(r, \theta_m)$；仅取决于报告的支付会加上 $t(r)$。

所以如实报告约束以及规划者的期望支付仍然是线性的。

有些支付，比如在后验为 $0$ 的报告下且状态为 $\theta_1$ 时的那笔支付，当代理人如实报告时永远不会被用到，因此它们对规划者而言没有代价，也未被唯一确定；对每笔支付施加一个微小的代价就能打破平局，使其倾向于零。

```{code-cell} ipython3
def solve_with_pay(U, W, Λ, π, on_state):
    """
    Green's linear program when the planner can also pay the agent.

    If on_state is True the planner pays t[r, m] >= 0 when the agent
    reports y_r and the state is θ_m; otherwise she pays t[r] >= 0 after
    report y_r. Returns the plan, the payments, and the planner's
    expected utility net of expected payments.
    """
    q, P = posteriors(Λ, π)
    K, N = U.shape[0], Λ.shape[1]
    UP, WP = U @ P, W @ P
    # S[l, n] is the probability of payment contingency l after signal n
    S = P if on_state else np.ones((1, N))
    L = S.shape[0]

    # z_{nk} sits at n * K + k, then t[r, l] at N * K + r * L + l
    expected_pay = (S * q).T.ravel()
    c = np.concatenate([-(UP * q).T.ravel(), expected_pay + 1e-9])
    rows = []
    for n in range(N):
        for r in range(N):
            if r != n:
                row_z, row_t = np.zeros((N, K)), np.zeros((N, L))
                row_z[r] += WP[:, n]
                row_z[n] -= WP[:, n]
                row_t[r] += S[:, n]
                row_t[n] -= S[:, n]
                rows.append(np.concatenate([row_z.ravel(), row_t.ravel()]))
    A_ub = np.array(rows).reshape(-1, N * (K + L))
    A_eq = np.hstack([np.kron(np.eye(N), np.ones(K)), np.zeros((N, N * L))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(A_ub)),
                  A_eq=A_eq, b_eq=np.ones(N), bounds=(0, None),
                  method="highs", options={"presolve": False})
    if res.status != 0:
        raise RuntimeError(res.message)
    x = np.maximum(res.x, 0.0) + 0.0
    Z = x[:N * K].reshape(N, K)
    t = x[N * K:].reshape(N, L)
    planner = (UP * q).T.ravel() @ x[:N * K] - expected_pay @ x[N * K:]
    return Z, t, planner


for on_state, name in ((True, "report and state"), (False, "report only")):
    Z, t, v_net = solve_with_pay(U_ri, W_ri, Λ_ri_fine, π_ri, on_state)
    print(f"pay on the {name}: planner {v_net:.4f}\nplan\n{Z}\npayments\n{t}")
```

当支付取决于报告和状态时，规划者宣布她的最优计划，并在代理人报告后验 $0.4$ 且状态最终为 $\theta_1$ 时支付 $0.25$。

一个看到 $0.4$ 的代理人从该支付中期望得到 $0.4 \times 0.25 = 0.1$，恰好等于他声称看到 $0.8$ 时所获得的收益 $0.4 - 0.3$。

这笔支付之所以便宜，是因为它是针对他而设计的：一个看到 $0$ 的代理人对它没有任何期望，而一个看到 $0.8$ 的代理人也不会被诱惑。

期望支付为 $(1/4)(0.4)(0.25) = 0.025$，所以规划者得到 $0.725$。

当支付只取决于报告时，规划者在报告 $0.4$ 之后支付 $1/12$，并同时以概率 $1/6$ 在那里采取 $a_1$，净值为 $43/60 \approx 0.717$。

无论哪种方式，支付都能缩小更好信息所带来的损害，但不能消除它：这两个值都介于规划者无支付时得到的 $0.70$ 和她在粗略实验下得到的 $0.75$ 之间，而在粗略实验下根本不需要支付。

这些支付有两个特点值得注意。

取决于状态的支付要求之后能够观察到该状态，就像能观察到股票的回报那样。

而且支付必须是非负的：格林的模型没有参与约束，所以若没有这个限制，规划者本可以向代理人收取任意大的费用，线性规划就会变得无界。

```{solution-end}
```

```{exercise}
:label: it_ex_postlewaite

{cite:t}`Postlewaite1982` 研究了四个等概率的状态 $\theta_1, \ldots, \theta_4$ 和四个行动。

用规划者的正参数 $b, c, d, e$ 以及代理人的正参数 $f, g, h, j$，支付矩阵（行是行动，列是状态）为

$$
U = 2 \begin{bmatrix} b & 0 & c & 0 \\ 0 & c & 0 & b \\ d & 0 & e & 0 \\ 0 & e & 0 & d \end{bmatrix},
\qquad
W = 2 \begin{bmatrix} 0 & f & 0 & g \\ g & 0 & f & 0 \\ h & 0 & j & 0 \\ 0 & j & 0 & h \end{bmatrix} .
$$

在精细实验中，代理人能观察到状态。

在粗略实验中，他只能得知状态是在 $\{\theta_1, \theta_2\}$ 中还是在 $\{\theta_3, \theta_4\}$ 中。

1. 波斯尔思韦特的情形II假设 $b > d + e > (b + c)/2 > c$ 且 $f > h + j > (f + g)/2 > g$，他得出结论："如果上述不等式成立，且 $h$ 足够接近 $j$"，精细实验会使双方都变差。取 $(b, c, d, e) = (10, 4, 5, 3)$ 和 $(f, g, h, j) = (10, 4.5, 3.9, 4)$。用 `garbling` 确认精细实验比粗略实验信息量更大。在两个实验下求解规划者的线性规划，比较双方的期望效用。在精细实验下，最优计划把多少概率放在对规划者而言被占优的行动 $a_3$ 和 $a_4$ 上（当 $b > d$ 且 $c > e$ 时它们被占优）？
2. 考虑精细实验下这样一族计划：在报告 $\theta_1$ 和 $\theta_3$ 之后以概率 $x$ 采取 $a_1$、否则采取 $a_3$，在报告 $\theta_2$ 和 $\theta_4$ 之后以概率 $x$ 采取 $a_2$、否则采取 $a_4$。证明如实报告是最优的当且仅当 $x \leq x^* = \min\{h/(h + g), \, j/(j + f)\}$，并求出双方作为 $x$ 的函数的期望效用。波斯尔思韦特写道，可以放在 $a_1$ 和 $a_2$ 上的概率"当 $h$ 任意接近 $j$ 时会变得任意小"。这是对的吗？
3. 波斯尔思韦特的情形III假设 $b > c > d > e$、$b > d + e$、$f > g > j > h$ 且 $f < j + h$，他得出结论：当 $h$ 非常接近 $j$ 时，在精细实验下"代理人变得更好，而委托人变得更差"。对 $(b, c, d, e) = (10, 6, 4, 3)$ 和 $(f, g, h, j) = (10, 8, 5.9, 6)$ 求解线性规划。然后证明，在情形III的不等式下，规划者在第2部分那族计划中所偏好的计划给代理人的收益少于他在粗略实验下所得到的。
4. 波斯尔思韦特的情形I假设 $e > d > b > c$ 且 $j > h > f > g$，他得出结论：更好的信息能使双方都受益。他的情形IV假设 $b > c > d > e$、$b < d + e$、$f > g > j > h$ 且 $f > j + h$，他得出结论：当 $h$ 接近 $j$ 时，它使规划者受益、使代理人受损。对每一种情形，从 $[0, 10]^8$ 中依照该情形的不等式均匀抽取大量参数向量（情形II到情形IV中 $h$ 与 $j$ 相差在 $0.1$ 以内），并记录每一方受益或受损的频率。由于规划者可能有多个最优计划，请对所有这些计划比较代理人的收益。波斯尔思韦特的四个结论中，哪些在每次抽样中都成立？
```

```{solution-start} it_ex_postlewaite
:class: dropdown
```

以下是一个解答。

我们构建支付矩阵以及两个实验。

在粗略实验下，似然矩阵把状态 $\theta_1$ 和 $\theta_2$ 都映射到第一个信号，把状态 $\theta_3$ 和 $\theta_4$ 都映射到第二个信号，因此它本身也是扰动矩阵。

```{code-cell} ipython3
def postlewaite(b, c, d, e, f, g, h, j):
    """Payoffs in Postlewaite's example (rows: actions, columns: states)."""
    U = 2 * np.array([[b, 0, c, 0],
                      [0, c, 0, b],
                      [d, 0, e, 0],
                      [0, e, 0, d]], dtype=float)
    W = 2 * np.array([[0, f, 0, g],
                      [g, 0, f, 0],
                      [h, 0, j, 0],
                      [0, j, 0, h]], dtype=float)
    return U, W


π_pw = np.full(4, 1/4)
Λ_pw_fine = np.eye(4)
Λ_pw_coarse = np.array([[1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.0]])

print("B =\n", garbling(Λ_pw_fine, Λ_pw_coarse, π_pw))
```

下面的函数报告规划者的价值以及代理人期望效用在规划者各个最优计划之间的变化范围，`compare` 则对两个实验分别打印出这些信息。

```{code-cell} ipython3
def payoffs(U, W, Λ, π):
    """
    Planner's value, the agent's lowest and highest payoffs across the
    planner's optimal plans (up to the tolerance in solve_planner), and a
    plan that gives the highest.
    """
    _, v_hat, agent_lo = solve_planner(U, W, Λ, π, ties="agent_worst")
    Z, _, agent_hi = solve_planner(U, W, Λ, π, ties="agent_best")
    return v_hat, agent_lo, agent_hi, Z


def compare(params):
    """Print both players' payoffs; return the plan for the fine experiment."""
    U, W = postlewaite(**params)
    for name, Λ in (("coarse", Λ_pw_coarse), ("fine", Λ_pw_fine)):
        v_hat, lo, hi, Z = payoffs(U, W, Λ, π_pw)
        agent = f"{lo:.4f}" if np.isclose(lo, hi) else f"{lo:.4f} to {hi:.4f}"
        print(f"{name}: planner {v_hat:.4f}, agent {agent}")
    return Z


case_2 = dict(b=10, c=4, d=5, e=3, f=10, g=4.5, h=3.9, j=4)
Z_fine = compare(case_2)
print("optimal plan under the fine experiment:\n", Z_fine)
```

在粗略实验下，"第一个信号之后采取 $a_1$，第二个信号之后采取 $a_2$"这个计划给规划者带来 $b = 10$，给代理人带来 $f = 10$，这是在该实验下双方各自能得到的最大期望收益。

在精细实验下，规划者得到约 $9.71$，代理人得到约 $5.64$，所以双方都受损。

精细实验下的最优计划把约 $5/7 \approx 0.71$ 的概率放在被占优的行动 $a_3$ 和 $a_4$ 上，波斯尔思韦特将此描述为"把大部分概率放在 $a_3$ 和 $a_4$ 上"。

对于第2部分，一个看到 $\theta_1$ 的代理人如实报告能获得 $2h(1 - x)$，而报告 $\theta_2$ 或 $\theta_4$ 能获得 $2gx$。

报告 $\theta_3$ 所得到的抽签与如实报告相同。

所以他如实报告当且仅当 $h(1 - x) \geq g x$，即 $x \leq h/(h + g)$。

类似地，在状态 $\theta_4$ 下，真实性要求 $x \leq h/(h + g)$，而在状态 $\theta_2$ 和 $\theta_3$ 下，它要求 $j(1 - x) \geq f x$，即 $x \leq j/(j + f)$。

规划者的期望效用为 $(b + c) x + (d + e)(1 - x)$，当 $b + c > d + e$ 时随 $x$ 递增，所以她设定 $x = x^*$。

代理人的期望效用为 $(h + j)(1 - x)$。

```{code-cell} ipython3
def family_values(b, c, d, e, f, g, h, j):
    """x*, and both players' payoffs at x*, in the family of part 2."""
    x_star = min(h / (h + g), j / (j + f))
    planner = (b + c) * x_star + (d + e) * (1 - x_star)
    return x_star, planner, (h + j) * (1 - x_star)


x_star, v_family, a_family = family_values(**case_2)
print(f"x* = {x_star:.4f}, planner {v_family:.4f}, agent {a_family:.4f}")
```

这些结果与线性规划的结果相匹配。

当一个看到 $\theta_1$ 的代理人撒谎时，他得到的是对 $a_2$ 和 $a_4$ 的一个抽签，这两个行动在该状态下分别给他带来 $2g$ 和 $0$，因此他的如实报告约束比较的是 $h$ 和 $g$，而 $j$ 并不出现在其中。

所以边界 $x^*$ 取决于 $h$ 与 $g$ 的比较以及 $j$ 与 $f$ 的比较，而不取决于 $h$ 与 $j$ 的比较。

所以波斯尔思韦特的断言是不对的：随着 $h$ 趋近 $j$，放在 $a_1$ 和 $a_2$ 上的权重仍保持远离零，在这个例子中它是 $2/7$。

一个只知道哪一对状态发生了的代理人，在这样的计划下获得 $\max\{xf + (1-x)h, \, xg + (1-x)j\}$。

在 $x = x^*$ 处，这等于 $(1 - x^*)(h + j)$，恰好等于知情的代理人所得到的。

若 $x^* = j/(j + f)$（如本例所示），第一项等于 $(1 - x^*)(h + j)$，而第二项不会更大，因为 $x^* \leq h/(h + g)$；若 $x^* = h/(h + g)$，两项的角色则互换。

所以在该族中规划者所偏好的计划下，忽略更精细的信息会使代理人恰好同样好，而不是更差。

对于第3部分，我们求解情形III并与第2部分的族进行比较。

```{code-cell} ipython3
case_3 = dict(b=10, c=6, d=4, e=3, f=10, g=8, h=5.9, j=6)
compare(case_3)
x_star, v_family, a_family = family_values(**case_3)
print(f"family of part 2: x* = {x_star:.4f}, planner {v_family:.4f}, "
      f"agent {a_family:.4f}")
```

在粗略实验下双方都得到 $10$，而在精细实验下规划者得到 $10.375$，代理人得到 $7.4375$：规划者获益，代理人受损，这与波斯尔思韦特的结论恰好相反。

事实上，只要情形III的不等式成立，代理人在第2部分那族计划中规划者所偏好的计划下总是受损的。

由于 $b > d + e$，规划者设定 $x = x^*$。

在粗略实验下，代理人得到 $f$，而在该计划下他得到 $(h + j)(1 - x^*)$。

$x$ 较小的计划会给他更多，在 $x = 0$ 时最多可达 $h + j > f$，但规划者不会选择它们。

若 $x^* = h/(h + g)$，他的收益为 $(h + j) g/(h + g)$，这小于 $f$，因为 $g(j - f) < 0 < h(f - g)$。

若 $x^* = j/(j + f)$，他的收益为 $(h + j) f/(j + f)$，这小于 $f$，因为 $h < f$。

波斯尔思韦特的论证之所以失效，是因为他把 $a_1$ 和 $a_2$ 上的权重当成可忽略不计，这相当于在 $x = 0$ 附近评估代理人的收益。

对于第4部分，我们为每种情形抽取参数。

对于情形I、III和IV，对均匀抽样排序可以得到在它们有序的那个区域内的均匀抽样，这比拒绝无序抽样要快得多。

在情形II到情形IV中，我们抽取 $h$ 使其与 $j$ 相差在 $0.1$ 以内。

```{code-cell} ipython3
def draw_case(case, rng, gap=0.1):
    """
    Uniform draw from [0, 10]^8 subject to the inequalities of a case and,
    in Cases II to IV, to h being within gap of j.
    """
    def decreasing(n):
        return np.sort(rng.uniform(0, 10, n))[::-1]
    while True:
        if case == "I":
            e, d, b, c = decreasing(4)
            j, h, f, g = decreasing(4)
            ok = True
        elif case == "II":
            b, c, d, e, f, g, j = rng.uniform(0, 10, 7)
            h = j + rng.uniform(-gap, gap)
            ok = (0 < h < 10 and b > d + e > (b + c) / 2 > c
                  and f > h + j > (f + g) / 2 > g)
        else:
            b, c, d, e = decreasing(4)
            f, g, j = decreasing(3)
            h = j - rng.uniform(0, gap)
            if case == "III":
                ok = h > 0 and b > d + e and f < j + h
            else:
                ok = h > 0 and b < d + e and f > j + h
        if ok:
            return dict(b=b, c=c, d=d, e=e, f=f, g=g, h=h, j=j)


tol, n_draws = 1e-6, 300
for case in ("I", "II", "III", "IV"):
    rng = np.random.default_rng(0)
    counts = dict(planner_gains=0, planner_loses=0,
                  agent_gains=0, agent_loses=0)
    for _ in range(n_draws):
        U, W = postlewaite(**draw_case(case, rng))
        v_c, lo_c, hi_c, _ = payoffs(U, W, Λ_pw_coarse, π_pw)
        v_f, lo_f, hi_f, _ = payoffs(U, W, Λ_pw_fine, π_pw)
        counts["planner_gains"] += v_f > v_c + tol
        counts["planner_loses"] += v_f < v_c - tol
        # the agent gains or loses whichever optimal plans are used
        counts["agent_gains"] += lo_f > hi_c + tol
        counts["agent_loses"] += hi_f < lo_c - tol
    shares = ", ".join(f"{k} {v / n_draws:.0%}" for k, v in counts.items())
    print(f"Case {case}: {shares}")
```

情形I和情形IV在每次抽样中都如波斯尔思韦特所说那样出现：在情形I中，双方都获益；在情形IV中，规划者获益而代理人受损。

他针对情形IV的论证重复了"$h$ 接近 $j$ 使得 $a_1$ 和 $a_2$ 上的权重可忽略不计"这一断言，而第2部分表明这是不对的。

他关于规划者的结论并不需要那个断言：在情形IV的不等式下，第2部分那族计划中的每一个计划对她而言都值 $(b + c)x + (d + e)(1 - x) > b$，因为 $b + c$ 和 $d + e$ 都超过 $b$。

该族计划也给代理人 $(h + j)(1 - x) < f$，但规划者的最优计划有时位于该族之外，所以代理人的损失要依赖具体抽样。

在情形II中，代理人在每次抽样中都受损，但规划者在大约一半的抽样中获益，所以"双方都更差"只在部分参数下成立。

以下是一个满足情形II不等式、$h$ 接近 $j$、但规划者获益的例子。

```{code-cell} ipython3
case_2_gain = dict(b=10, c=9, d=5, e=4.8, f=10, g=4.5, h=3.9, j=4)
_ = compare(case_2_gain)
```

规划者从 $10$ 上升到约 $12.43$，而代理人从 $10$ 下降到约 $5.64$。

波斯尔思韦特关于 $h$ 要接近 $j$ 的条件对情形II中代理人一方来说很重要：如果没有这个条件，代理人是可能获益的。

```{code-cell} ipython3
case_2_far = dict(b=9, c=3, d=2.5, e=4, f=8, g=7.8, h=1, j=6.95)
_ = compare(case_2_far)
```

这些参数满足情形II的不等式，且 $h$ 远离 $j$，代理人从 $8$ 上升到 $10.85$，而规划者从 $9$ 下降到 $8.5$，这正是波斯尔思韦特归因于情形III的那种模式。

在情形III中，无论使用规划者哪个最优计划，代理人在每次抽样中都受损，而规划者在超过一半的抽样中获益。

第3部分只针对该族中规划者所偏好的计划证明了代理人的损失；这些抽样表明这一点对线性规划的最优计划似乎也成立，其中一些计划位于该族之外。

这些比例取决于参数是如何被抽取的，所以它们描述的是每个区域上的均匀分布，而不是这些情形本身。

波斯尔思韦特更广泛的要点依然成立：更好的信息可能使双方都变差，或者使一方受益而另一方受损。

```{solution-end}
```

```{exercise}
:label: it_ex_reverse

{cite:t}`Postlewaite1982` 问如果角色互换会发生什么：代理人先承诺一个应答规则，然后规划者根据所听到的内容采取她最喜欢的行动。

波斯尔思韦特使用了格林的术语，而格林的应答规则是确定性的；第1部分也允许代理人随机化。

1. 论证代理人推荐规划者愿意遵从的行动不会有任何损失。把代理人的问题（当他也可以随机化时）写成一个关于 $N \times K$ 马尔可夫矩阵 $\Gamma = [\gamma_{nk}]$ 的线性规划问题，其中 $\gamma_{nk}$ 是代理人在观察到 $y_n$ 之后推荐 $a_k$ 的概率，受限于对规划者而言的**服从约束**。假设一个无差异的规划者会遵从推荐。
2. 对 {ref}`it_ex_riley` 中赖利例子的精细实验、格林的第一个例子，以及 {ref}`it_ex_postlewaite` 中情形II的例子求解该线性规划。然后通过检验所有确定性规则（即每个 $\gamma_{nk}$ 都是 $0$ 或 $1$ 的规则）来找出代理人最好的确定性规则。比较两种情形下双方的期望效用：规划者领先时，以及代理人领先时。
3. 波斯尔思韦特说"构造出领导者-跟随者角色互换后委托人和代理人都变得更好的例子是很容易的"。这些例子支持他的说法吗？
```

```{solution-start} it_ex_reverse
:class: dropdown
```

以下是一个解答。

对于第1部分，任何发送消息的规则都会使规划者在每条消息之后采取某个行动。

把所有导致她采取同一行动 $a_k$ 的消息合并为单一消息"采取 $a_k$"。

她在合并后的消息之后的后验是她在原始消息之后各个后验的一个平均值，而她在每一个后验下都偏好 $a_k$，所以根据线性性，她在合并后的后验下依然偏好 $a_k$。

合并后的规则产生了相同的信号与行动联合分布，所以双方的期望效用都不变，而确定性规则仍保持确定性。

规划者遵从推荐 $a_k$ 当且仅当，对每个其他行动 $a_{k'}$，

$$
\sum_{n=1}^N q_n \, \gamma_{nk} \left[(UP)_{kn} - (UP)_{k'n}\right] \;\geq\; 0 ,
$$

而代理人在这些约束下最大化 $\sum_n q_n \sum_k \gamma_{nk} (WP)_{kn}$。

这正是 {cite:t}`KamenicaGentzkow2011` 的说服问题，不同之处在于代理人只能对自己的信号进行扰动，而不能随心所欲地设计任何他喜欢的实验。

对于第2部分，`solve_agent_leads` 求解该线性规划，`best_deterministic_rule` 检验所有确定性规则；`obedience_rows` 和 `leader_payoffs` 是二者共用的辅助函数。

```{code-cell} ipython3
def obedience_rows(U, Λ, π):
    """Rows of the obedience constraints, A @ vec(Γ) <= 0."""
    q, P = posteriors(Λ, π)
    UP = U @ P
    K, N = UP.shape
    # Entry n * K + k of vec(Γ) is γ_{nk}; for each recommended action k
    # and alternative j: sum_n q_n γ_{nk} [(UP)_{jn} - (UP)_{kn}] <= 0
    rows = []
    for k in range(K):
        for j in range(K):
            if j != k:
                row = np.zeros((N, K))
                row[:, k] = q * (UP[j] - UP[k])
                rows.append(row.ravel())
    return np.array(rows)


def leader_payoffs(Γ, U, W, Λ, π):
    """Expected utilities of the planner and the agent under Γ."""
    q, P = posteriors(Λ, π)
    return q @ (Γ * (U @ P).T).sum(axis=1), q @ (Γ * (W @ P).T).sum(axis=1)


def solve_agent_leads(U, W, Λ, π):
    """
    The agent's best randomized recommendation rule.

    Returns the N x K rule Γ and the expected utilities of the planner and
    of the agent under it.
    """
    q, P = posteriors(Λ, π)
    K, N = U.shape[0], Λ.shape[1]
    A_ub = obedience_rows(U, Λ, π)
    res = linprog(-((W @ P) * q).T.ravel(), A_ub=A_ub,
                  b_ub=np.zeros(len(A_ub)),
                  A_eq=np.kron(np.eye(N), np.ones(K)), b_eq=np.ones(N),
                  bounds=(0, None), method="highs",
                  options={"presolve": False})
    if res.status != 0:
        raise RuntimeError(res.message)
    Γ = np.maximum(res.x, 0.0).reshape(N, K) + 0.0
    return (Γ, *leader_payoffs(Γ, U, W, Λ, π))


def best_deterministic_rule(U, W, Λ, π):
    """
    The agent's best deterministic recommendation rule, by enumeration;
    among rules that are equally good for him, the planner's favorite.

    Returns the rule Γ and the expected utilities of the planner and of
    the agent under it.
    """
    K, N = U.shape[0], Λ.shape[1]
    A_ub = obedience_rows(U, Λ, π)
    best = None
    for actions in product(range(K), repeat=N):
        Γ = np.eye(K)[list(actions)]
        if np.any(A_ub @ Γ.ravel() > 1e-12):
            continue
        planner, agent = leader_payoffs(Γ, U, W, Λ, π)
        if (best is None or agent > best[2] + 1e-12
                or (agent > best[2] - 1e-12 and planner > best[1])):
            best = (Γ, planner, agent)
    return best


U_c2, W_c2 = postlewaite(**case_2)
examples = (("Riley", U_ri, W_ri, Λ_ri_fine, π_ri),
            ("Green's first", U_ex1, W_ex1, Λ_ex1, π_ex1),
            ("Case II", U_c2, W_c2, Λ_pw_fine, π_pw))
for name, U, W, Λ, π in examples:
    _, v_lead, a_lead = solve_planner(U, W, Λ, π)
    _, v_rand, a_rand = solve_agent_leads(U, W, Λ, π)
    _, v_det, a_det = best_deterministic_rule(U, W, Λ, π)
    print(f"{name}: planner leads ({v_lead:.4f}, {a_lead:.4f}); "
          f"agent leads, deterministic ({v_det:.4f}, {a_det:.4f}), "
          f"randomized ({v_rand:.4f}, {a_rand:.4f})")
```

每一对数值中，第一个是规划者的期望效用，第二个是代理人的期望效用。

在使用确定性规则时，代理人在赖利的例子和格林第一个例子中的最优选择是如实告知。

承诺说出真相使规划者能够根据她所知道的一切采取行动，这给规划者带来了她的最优价值，而且对代理人而言，这比规划者领先时他所得到的更多。

在情形II中，他最好的确定性规则只报告哪一对状态发生了，这相当于让他承诺使用粗略实验，正如波斯尔思韦特所说他愿意做的那样；此时双方都得到 $10$。

在这三个例子中，一个随机化的规则能让代理人做得更好，但代价是规划者受损。

在赖利的例子中，他以 $2/3$ 的概率在后验 $0.4$ 之后推荐 $a_1$，恰好足以使一个被告知采取 $a_1$ 的规划者（她此后把 $0.7$ 的概率放在 $\theta_1$ 上）愿意遵从。

在格林的第一个例子中，规划者下降到 $v_0 = 0.5$，在情形II中则下降到 $7$。

对于第3部分，波斯尔思韦特对他所描述的角色互换是正确的：在使用确定性应答规则时，角色互换会使这三个例子中的双方都严格变得更好。

在这些例子中，一个还能随机化的代理人会把规划者的收益据为己有，而规划者则不会变得更好，有时甚至比她领先时更糟。

但这并非一般规律。

服从约束保证了规划者至少能得到 $v_0$，因为无论收到什么推荐，她总是可以采取某个固定行动，所以当她自己的计划只能给她 $v_0$ 时，让代理人领先（即使是随机化规则）也不会使她受损。

在赖利的例子中她只是打平，但在下一个例子中双方都严格获益。

以下是一个有两个状态、均匀先验、完全信息实验的例子。

```{code-cell} ipython3
U_rv = np.array([[0.0, 2.0],
                 [2.0, 1.0],
                 [3.0, 0.0]])
W_rv = np.array([[2.0, 2.0],
                 [0.0, 0.0],
                 [1.0, 2.0]])
Λ_rv, π_rv = np.eye(2), np.array([1/2, 1/2])

v_lead = solve_planner(U_rv, W_rv, Λ_rv, π_rv)[1]
a_best = solve_planner(U_rv, W_rv, Λ_rv, π_rv, ties="agent_best")[2]
_, v_rand, a_rand = solve_agent_leads(U_rv, W_rv, Λ_rv, π_rv)
print(f"planner leads: planner {v_lead:.4f}, agent at most {a_best:.4f}")
print(f"agent leads, randomized: planner {v_rand:.4f}, agent {a_rand:.4f}")
```

当规划者领先时，她只得到 $v_0 = 1.5$，代理人最多得到 $1.5$；当代理人以随机化规则领先时，双方都得到 $1.75$。

```{solution-end}
```

```{exercise}
:label: it_ex_mm

{cite:t}`MarschakMiyasawa1968` 比较了关于后验概率的几个条件。

其中之一，我们称之为**凸包条件**，要求 $\Lambda'$ 的每个后验都位于 $\Lambda$ 的后验所构成的凸包之内。

另一个是存在一个两个信号的联合分布 $J$（边缘为 $q$ 和 $q'$），满足 {eq}`it_barycenter`；它等价于布莱克威尔序。

取两个具有均匀先验的状态。

实验 $\Lambda$ 对 $\theta_1$ 的后验为 $0.9$、$0.5$ 和 $0.1$，概率分别为 $0.05$、$0.9$ 和 $0.05$。

实验 $\Lambda'$ 的后验为 $0.7$ 和 $0.3$，各自概率为 $1/2$。

1. 证明凸包条件对 $\Lambda$ 相对 $\Lambda'$ 成立，但 $\Lambda$ 并不比 $\Lambda'$ 信息量更大。用后验的方差解释原因。$\Lambda'$ 比 $\Lambda$ 信息量更大吗？
2. 一个实验的**模糊度**是后验的期望熵，$-\sum_n q_n \sum_m p_{mn} \log p_{mn}$。计算这两个实验的模糊度。马尔沙克和宫泽的定理12.3说，较低的模糊度是信息量更大的必要条件，但不是充分条件。通过找出两位对 $\Lambda$ 和 $\Lambda'$ 排序相反的决策者来说明这一点。与 {doc}`blackwell_kihlstrom` 中对熵的处理进行比较。
3. 马尔沙克和宫泽的定理10.1说，当 $\Lambda$ 只有两个信号时，凸包条件是充分的。解释原因，并用两个状态做数值检验。这与 {prf:ref}`it_thm_binary` 的证明有何关联？
```

```{solution-start} it_ex_mm
:class: dropdown
```

以下是一个解答。

对于第1部分，$\Lambda'$ 的后验 $0.7$ 和 $0.3$ 位于 $\Lambda$ 的后验所张成的区间 $[0.1, 0.9]$ 内，因此凸包条件成立。

```{code-cell} ipython3
π_mm = np.array([1/2, 1/2])
Λ_mm = experiment([0.9, 0.5, 0.1], [0.05, 0.9, 0.05], π_mm)
Λ_mm_prime = experiment([0.7, 0.3], [1/2, 1/2], π_mm)

print("Λ more informative than Λ':",
      garbling(Λ_mm, Λ_mm_prime, π_mm) is not None)
print("Λ' more informative than Λ:",
      garbling(Λ_mm_prime, Λ_mm, π_mm) is not None)

for name, Λ in (("Λ", Λ_mm), ("Λ'", Λ_mm_prime)):
    q, P = posteriors(Λ, π_mm)
    print(f"variance of the posterior under {name}: {q @ (P[0] - 0.5)**2:.4f}")
```

$\Lambda$ 并不比 $\Lambda'$ 信息量更大。

凸包条件只比较后验的位置，而忽略它们各自的可能性大小。

在 $\Lambda$ 下，后验通常是 $0.5$，所以其方差为 $0.016$，而在 $\Lambda'$ 下方差为 $0.04$。

保均值展开会提高方差，所以 $\Lambda$ 下的后验分布不可能是 $\Lambda'$ 下后验分布的保均值展开。

排除这种可能性的是 $J$ 的边缘必须为 $q$ 这一要求。

要把后验 $0.7$ 写成 $0.9$、$0.5$、$0.1$ 的一个平均值，至少一半的权重必须放在 $0.9$ 上，所以 $J$ 必须在后验 $0.9$ 上放置至少 $1/4$ 的概率，而 $\Lambda$ 只以概率 $0.05$ 产生这个后验。

$\Lambda'$ 也不比 $\Lambda$ 信息量更大：$\Lambda$ 的后验 $0.9$ 和 $0.1$ 位于 $\Lambda'$ 后验所张成的凸包 $[0.3, 0.7]$ 之外。

凸包条件是布莱克威尔序的必要条件，因为信息量较小的实验的每个后验都是信息量较大的实验诸后验的一个平均值，所以这里不满足该条件，而 `garbling` 也证实了这一点。

这两个实验是不可比较的。

对于第2部分，我们计算每个实验的模糊度。

```{code-cell} ipython3
def equivocation(Λ, π):
    """Expected entropy of the posterior."""
    q, P = posteriors(Λ, π)
    logs = np.log(P, where=P > 0, out=np.zeros_like(P))
    return -q @ (P * logs).sum(axis=0)


for name, Λ in (("Λ", Λ_mm), ("Λ'", Λ_mm_prime)):
    print(f"equivocation of {name}: {equivocation(Λ, π_mm):.4f}")
```

$\Lambda'$ 的模糊度更低，但它并不比 $\Lambda$ 信息量更大。

考虑这样的决策者：他们在一个在状态 $\theta_1$ 下支付 $1$、在状态 $\theta_2$ 下损失 $x$ 的赌局，与不下注之间进行选择。

当后验超过 $x/(1 + x)$ 时，这个赌局是值得一下的。

```{code-cell} ipython3
def bet_utility(x):
    """Bet paying 1 in θ_1 and losing x in θ_2, against no bet."""
    return np.array([[1.0, -x],
                     [0.0, 0.0]])


for x in (4.0, 1.5):
    U_bet = bet_utility(x)
    print(f"x = {x}: threshold {x / (1 + x):.2f}, "
          f"value with Λ {bayes_value(U_bet, Λ_mm, π_mm):.4f}, "
          f"with Λ' {bayes_value(U_bet, Λ_mm_prime, π_mm):.4f}")
```

一个只在后验超过 $0.8$ 时才下注的谨慎赌徒看重 $\Lambda$，因为它有时会产生后验 $0.9$，而从 $\Lambda'$ 那里他一无所获。

一个临界值为 $0.6$ 的更大胆的赌徒则偏好 $\Lambda'$。

这正是马尔沙克和宫泽所指出的：模糊度不同于信息量，它能给所有的实验排序，因此无法体现信息价值对使用者支付函数的依赖性。

{doc}`blackwell_kihlstrom` 用德格鲁特的术语说明了同样的要点：布莱克威尔序要求不确定性的减少对每一个凹的不确定性函数都更大，而不仅仅是对熵更大。

对于第3部分，假设 $\Lambda$ 有两个信号，其后验 $p_1$ 和 $p_2$ 互不相同，并假设 $\Lambda'$ 的每个后验都位于连接它们的线段上。

把 $\Lambda'$ 的每个后验 $p'_{n'}$ 写成 $p_1$ 和 $p_2$ 的一个混合，并按照该混合权重拆分信号 $y'_{n'}$。

这给出了一个联合分布 $J$，其各列之和为 $q'$，并满足 {eq}`it_barycenter`。

其各行之和把概率放在 $p_1$ 和 $p_2$ 上，这些概率的平均值为 $\sum_{n'} q'_{n'} p'_{n'} = \pi$。

在两个互不相同的点上，只有一个分布使均值等于 $\pi$，所以各行之和就是 $q$，于是 $\Lambda$ 比 $\Lambda'$ 信息量更大。

$J$ 的行之和等于 $q$ 这一要求（在第1部分中失败）在这里自动成立，对任意数量的状态都是如此。

我们通过随机抽取实验 $\Lambda'$ 以及一个后验能夹住 $\Lambda'$ 后验的两信号实验 $\Lambda$ 来验证这一点。

```{code-cell} ipython3
rng = np.random.default_rng(2468)

found = []
for _ in range(500):
    N_prime = rng.integers(2, 6)
    post = rng.uniform(0.05, 0.95, N_prime)
    probs = rng.dirichlet(np.ones(N_prime))
    π = np.array([probs @ post, 1 - probs @ post])
    lo = rng.uniform(0, post.min())
    hi = rng.uniform(post.max(), 1)
    w_hi = (π[0] - lo) / (hi - lo)
    Λ_binary = experiment([hi, lo], [w_hi, 1 - w_hi], π)
    Λ_other = experiment(post, probs, π)
    found.append(garbling(Λ_binary, Λ_other, π) is not None)
print("garbling found in every draw:", all(found))
```

{prf:ref}`it_thm_binary` 的证明通过两个事实来利用信息序：$\Lambda'$ 的后验位于连接 $\Lambda$ 后验的线段上，并且存在满足 {eq}`it_barycenter` 的联合分布 $J$。

当 $\Lambda$ 只有两个信号时，上面的论证表明第一个事实能推出第二个事实，所以此时布莱克威尔序就变成了一个关于后验位置的条件，而不是关于它们各自可能性大小的条件。

```{solution-end}
```