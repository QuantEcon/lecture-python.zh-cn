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
    Overview: 概述
    A statistical decision problem: 一个统计决策问题
    A statistical decision problem::States, signals, and actions: 状态、信号与行动
    A planner and an agent: 规划者与代理人
    A planner and an agent::Action plans and reports: 行动计划与报告
    A planner and an agent::The revelation principle: 显示原理
    A planner and an agent::Solving the linear program: 求解线性规划
    Economies with two actions: 具有两种行动的经济体
    Dominated actions as threats: 被支配行动作为威胁
    Truthful but coarse reports: 如实但粗略的报告
    Comparing experiments: 比较各个实验
    Comparing experiments::Blackwell's order through posteriors: 通过后验看布莱克韦尔序
    Comparing experiments::A transport problem: 一个运输问题
    Better information can hurt the planner: 更好的信息可能损害规划者
    Better information can hurt the planner::Green's third example: 格林的第三个例子
    Better information can hurt the planner::A family of refinements: 一族精细化方案
    Better information can hurt the planner::Experiments with two signals: 只有两种信号的实验
    What the discussants added: 评论者补充的内容
    Relation to other lectures: 与其他讲座的关系
    Relation to other lectures::Later work on communication and delegation: 关于沟通与授权的后续研究
    Concluding remarks: 结束语
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

## 概述

在许多组织中，收集信息的人并不是根据信息采取行动的人。

股票经纪人研究一家公司，而客户决定是否购买其股票。

医生检查病人，而病人选择治疗方案。

部门经理了解当地情况，而总部则在各部门之间分配资本。

当信息提供者和决策者想要不同的东西时，信息提供者可能会歪曲他所看到的情况。

{cite:t}`Green1982` 研究了一位决策者——**规划者**——她必须从一位**代理人**那里获取观察结果，而该代理人对规划者行动的偏好与她不同。

规划者先行动，并承诺一个**行动计划**，即一条将每个可能的报告映射为行动上的概率分布的规则。

代理人随后观察到从某个统计实验中抽取的信号，形成后验，并发送该行动计划最能奖励的报告。

格林得出了以下结果。

1. 规划者将自己限制在使代理人更愿意如实报告的计划中并不会有所损失，因此规划者的问题是一个单一的线性规划问题。
2. 在两种行动的情况下，规划者最多只能做到忽略代理人或听任代理人行事；在两种行动和两种信号的情况下，她要么达到她的第一优选结果，要么从代理人那里一无所获。
3. 当行动数超过两种时，最优计划可以随机化，并且可以将正概率赋予一个规划者如果自己看到信号就绝不会采取的行动。
4. 如实报告并不意味着所有信息都被利用：最优计划可以在区分某些信号的同时，将另一些信号合并处理。
5. 按照 {doc}`blackwell_kihlstrom` 的含义，给代理人提供信息量更大的实验，可能会降低规划者的预期效用。
6. 当两个实验都只有两种信号时，这种情况不会发生。

结果5和6将格林的模型与布莱克韦尔定理联系起来。

亲自看到信号的决策者永远不会偏好信息量较小的实验。

而必须从他人那里获取信号的规划者有时确实会如此，因为更清晰的信号可能会暴露出一种被较粗糙信号所掩盖的利益冲突。

为了比较实验，格林采用了 {cite:t}`MarschakMiyasawa1968` 提出的、用后验概率来刻画布莱克韦尔序的方法。

我们将看到，他们的刻画方式将"一个实验是否是另一个实验的粗化"这一问题，转化为了 {doc}`opt_transport` 中所研究的那种运输问题，并附加了一组额外的约束。

该论文发表时附有 {cite:t}`Postlewaite1982` 和 {cite:t}`Riley1982` 的评论。

练习题将逐步讲解他们的例子，以及 {cite:t}`MarschakMiyasawa1968` 对信息条件的比较。

本讲座的内容安排如下。

1. 我们用格林的符号建立一个统计决策问题。
2. 我们引入一个代理人，陈述格林的显示原理，并求解规划者的线性规划问题。
3. 我们研究具有两种行动的经济体，以及格林关于随机化和合并信号的例子。
4. 我们使用马尔沙克-宫泽刻画来比较实验，展示更优的信息如何会损害规划者，并证明当两个实验都有两种信号时这种情况不会发生。
5. 我们报告评论者补充了什么内容，并将格林的模型与其他讲座以及后续关于沟通和授权的研究联系起来。

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

有 $M$ 个自然状态 $\theta_1, \ldots, \theta_M$，其先验概率为 $\pi = (\pi_1, \ldots, \pi_M)$。

在行动之前，某人会观察到 $N$ 个信号 $y_1, \ldots, y_N$ 中的一个。

$M \times N$ 的**似然矩阵** $\Lambda = [\lambda_{mn}]$（我们也称之为**实验**）的元素为

$$
\lambda_{mn} = \Pr(y_n \mid \theta_m),
$$

因此 $\Lambda$ 的每一行都是关于信号的一个概率分布。

信号概率和各状态的后验概率分别为

$$
q = \Lambda^\top \pi,
\qquad
p_{mn} = \Pr(\theta_m \mid y_n) = \frac{\lambda_{mn} \pi_m}{q_n} .
$$ (it_bayes)

我们将后验概率收集到 $M \times N$ 矩阵 $P = [p_{mn}]$ 中，其第 $n$ 列 $p_n$ 为信号 $y_n$ 之后的后验概率。

有 $K$ 种行动 $a_1, \ldots, a_K$。

一个 $K \times M$ 的效用矩阵 $U = [u_{km}]$ 给出了行动 $a_k$ 在状态 $\theta_m$ 下的效用 $u_{km}$。

$K \times N$ 矩阵 $UP$ 的第 $(k, n)$ 个元素是信号 $y_n$ 之后行动 $a_k$ 的预期效用。

一个亲自观察到信号的决策者，会在 $UP$ 的每一列中选取数值最大的那个行动。

我们称这一规则为她的**第一优选计划**，而她的预期效用

$$
v(\Lambda) = \sum_{n=1}^N q_n \max_k \, (UP)_{kn}
$$ (it_bayes_value)

称为她的**第一优选值**。

如果没有信号，她会选择使 $(U\pi)_k$ 最大化的行动，其预期效用为 $v_0 = \max_k (U\pi)_k$。

下面的函数计算后验概率以及两个基准值 $v(\Lambda)$ 和 $v_0$。

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

现在假设信号不是由决策者观察到的，而是由一个代理人观察到的。

从现在起，我们称决策者为规划者。

代理人在意规划者的行动以及状态，他的效用是一个 $K \times M$ 矩阵 $W = [w_{km}]$ 的元素。

格林用 $U'$ 表示代理人的矩阵；我们使用 $W$，并将撇号保留给可替代的实验。

规划者和代理人共享先验 $\pi$，且两者都知道 $\Lambda$、$U$ 和 $W$。

事件按如下顺序展开。

1. 规划者宣布一个行动计划，即一个 $N \times K$ 的马尔可夫矩阵 $Z = [z_{nk}]$，也就是一个各行之和为1的非负矩阵；$z_{nk}$ 是代理人报告 $y_n$ 时规划者采取行动 $a_k$ 的概率。
2. 代理人观察到信号 $y_n$，其后验为 $p_n$。
3. 代理人报告某个信号 $y_r$，该信号不必等于 $y_n$。
4. 规划者从 $Z$ 的第 $r$ 行中抽取一个行动。

记

$$
V = Z W P .
$$

$N \times N$ 矩阵 $V$ 的第 $(r, n)$ 个元素是代理人在看到 $y_n$ 但报告 $y_r$ 时的预期效用。

因此代理人的最佳报告就是每一列中数值最大的那些元素。

规划者是有承诺约束的：一旦代理人发言，她就要执行所宣布的随机化方案，即使根据该报告，在她看来另一个行动可能更好。

格林强调，这包括抽签本身。

在报告 $y_r$ 之后，她必须采取第 $r$ 行的彩票所产生的任何行动；她不能用该行允许的行动中她最喜欢的那个来替代，并声称彩票选择了它。

从这个意义上说，规划者是一个斯塔克尔伯格领导者，就像 {doc}`Stackelberg plans <advanced:dyn_stack>` 中那样。

承诺使她能够做出威胁——比如下文格林第一个例子中的被支配行动——而如果她能够在听到报告后重新进行最优化，她就不会真正执行这些威胁。

承诺的价值也是 {doc}`phillips_credibility` 的主题，该讲座研究了基德兰德-普雷斯科特的货币政策模型。

如果没有承诺，规划者会在每次报告之后根据她的信念采取最优行动，我们就会处于 {cite:t}`CrawfordSobel1982` 的廉价交谈世界中。

### 显示原理

一个计划 $Z$ 诱导出一个**应答规则** $\rho$：在观察到 $y_n$ 之后，代理人报告 $y_{\rho(n)}$。

遵循格林的做法，应答规则是从观察到报告的确定性映射。

格林的第一个结果表明，规划者可以将自己限制在应答规则为恒等映射的计划中。

```{prf:theorem} 显示原理（格林的定理1）
:label: it_thm_revelation

当代理人对每一个计划都以对他最优的应答规则作出反应时，规划者在所有行动计划中所能获得的最高预期效用，等于以下线性规划问题的值：

$$
\max_{Z} \; \sum_{n=1}^N q_n \sum_{k=1}^K z_{nk} \, (UP)_{kn}
$$

约束条件为 $z_{nk} \geq 0$，对每个 $n$ 都有 $\sum_k z_{nk} = 1$，以及**如实报告约束**：对所有 $n$ 和 $r$ 都有 $V_{nn} \geq V_{rn}$，即

$$
\sum_{k=1}^K z_{nk} \, (WP)_{kn}
\;\geq\;
\sum_{k=1}^K z_{rk} \, (WP)_{kn}
\quad \text{for all } n \text{ and } r .
$$
```

这个定理是机制设计中显示原理的一个版本；格林引用的与他的分析最接近的文献是 {cite:t}`Myerson1979`。

下面是论证过程。

假设计划 $Z$ 诱导出应答规则 $\rho$。

构造一个新计划 $\hat Z$，其第 $n$ 行为 $Z$ 的第 $\rho(n)$ 行。

$\hat Z$ 的各行都在 $Z$ 的各行之中，所以代理人没有新的选项。

在观察到 $y_n$ 之后，他此前选择了 $Z$ 的第 $\rho(n)$ 行，因此他愿意选择 $\hat Z$ 的第 $n$ 行，这与之前是同一个彩票。

在 $\hat Z$ 下如实报告时，规划者在每次观察之后的行动分布与在 $Z$ 和原始报告下的分布相同，所以她的预期效用是相同的。

格林将如实报告约束写成弱不等式，所以一个无差异的代理人被假定会如实报告。

我们用 $\hat v(\Lambda)$ 表示该线性规划的值。

一个忽略代理人报告的计划满足如实报告约束，而且没有任何计划能比一个亲自看到信号的规划者做得更好，因此

$$
v_0 \;\leq\; \hat v(\Lambda) \;\leq\; v(\Lambda) .
$$

{cite:t}`Riley1982` 指出，经过恰当解释，显示原理本质上是定义性的：代理人只是从 $Z$ 的各行所构成的菜单中选择一个彩票。

这种解读使得格林的规划者成为一个菜单设计者，供一个信息更灵通的下属从中选择，这正是本讲座末尾所讨论的授权问题的主题。

如实报告并不意味着所有信息都能传递：如果 $Z$ 的两行相等，规划者就会对两个对应的信号一视同仁，这一点格林的第二个例子将会说明。

### 求解线性规划

我们将 $Z$ 的元素按行堆叠成一个长度为 $NK$ 的向量，并将问题交给 SciPy 的 `linprog`。

每行 $Z$ 之和为1的约束随后采用克罗内克积的形式 `np.kron(np.eye(N), np.ones(K))`。

{doc}`opt_transport` 是按列而不是按行堆叠的，所以在那里相同的形式表示的是列和约束。

由于 `linprog` 执行最小化，我们传入规划者目标的负值。

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

当规划者有多个最优计划时，代理人的预期效用在不同计划之间可能不同。

当 `ties=None` 时，`solve_planner` 报告求解器所找到的那个最优计划下代理人的效用。

当 `ties="agent_best"` 或 `ties="agent_worst"` 时，它会求解第二个线性规划，使规划者保持在其最优值，同时使代理人尽可能地好或尽可能地差。

## 具有两种行动的经济体

在只有两种行动的情况下，规划者的问题变得简单，正如格林接下来的两个结果所示。

格林是针对两种状态证明了第一个结果，但下面的论证对任意数量的状态和信号都成立。

```{prf:theorem} 两种行动（格林的定理3）
:label: it_thm_two_actions

如果 $K = 2$，规划者存在一个最优计划，它要么忽略代理人的报告，要么在每个信号之后，都采取代理人若身为决策者会选择的行动。
```

在第二种情况下，规划者执行的是代理人的第一优选计划，而非她自己的。

在两种行动的情况下，$Z$ 的一行由 $z_{n1}$ 决定，即报告 $y_n$ 之后采取 $a_1$ 的概率，在本节中我们简记为 $z_n$。

在观察到 $y_n$ 之后，代理人对报告 $y_r$ 的评价为

$$
(WP)_{2n} + z_r \left[(WP)_{1n} - (WP)_{2n}\right],
$$

所以无论他看到了什么信号，代理人仅凭 $z_r$ 对 $Z$ 的各行进行排序。

当他在后验下偏好 $a_1$ 时，他会报告具有最大 $z_r$ 的信号；当他偏好 $a_2$ 时，他会报告具有最小 $z_r$ 的信号。

因此，如实报告迫使 $z_n$ 在代理人偏好 $a_1$ 的那些信号上取同一个公共值 $z_{\rm hi}$，而在其余信号上取另一个公共值 $z_{\rm lo} \leq z_{\rm hi}$，至少在代理人从不恰好无差异的情况下如此。

如果代理人在每一个信号之后都偏好同一个行动，则所有 $z_n$ 都相等，计划忽略了报告。

否则，规划者的预期效用在三角形 $0 \leq z_{\rm lo} \leq z_{\rm hi} \leq 1$ 上关于 $(z_{\rm lo}, z_{\rm hi})$ 是线性的，所以它在某个顶点处达到最大值。

两个顶点 $(0, 0)$ 和 $(1, 1)$ 忽略了报告，而第三个顶点 $(0, 1)$ 则使代理人在每一个信号之后都得到他偏好的行动。

所以 $\hat v(\Lambda) = \max\{v_0, v_A(\Lambda)\}$，其中 $v_A(\Lambda)$ 是规划者在每个信号之后都采取代理人所偏好行动时的预期效用；如果代理人有时处于无差异状态，应当以有利于规划者的方式打破这种平局。

该定理说明*某个*最优计划采取这两种形式之一。

当规划者的目标在三角形的某条边上保持不变时，介于两端之间的计划也是最优的，而且它们可能给代理人带来不同的收益。

格林指出解"必须"是这两种类型之一，但他自己的第三个例子（下文将研究）却有既非这两种类型的最优计划。

当同时具有两种行动和两种信号时，规划者要么全得要么全无。

```{prf:corollary} 两种行动和两种信号（格林的定理2）
:label: it_thm_two_signals

如果 $K = N = 2$，那么要么 $\hat v(\Lambda) = v(\Lambda)$，即规划者达到她的第一优选结果；要么 $\hat v(\Lambda) = v_0$，即代理人的信息对她来说毫无价值。
```

如果两位参与者的第一优选计划是一致的，规划者就宣布她自己的计划，代理人如实报告，且 $\hat v(\Lambda) = v(\Lambda)$。

相反，假设两者的计划不同，并且为了使论证简短，假设两位参与者都从不处于无差异状态。

根据 {prf:ref}`it_thm_two_actions`，$\hat v(\Lambda) = \max\{v_0, v_A(\Lambda)\}$，所以只需证明 $v_A(\Lambda) \leq v_0$ 即可。

* 如果代理人在两个信号之后都偏好同一行动，他的第一优选计划采取一个单一的行动，对规划者来说价值最多为 $v_0$。
* 如果代理人偏好的行动在两个信号之间不同，而规划者的偏好不变，那么 $v_A(\Lambda) \leq v(\Lambda) = v_0$。
* 如果两位参与者偏好的行动在两个信号之间都不同，那么它们是相反的，所以代理人的第一优选计划会在两个信号之后都给规划者带来她较不喜欢的行动，而在两个信号之后都采取其中任一行动都会更好。

在每种情况下，$\hat v(\Lambda) = v_0$。

这证明了该推论，并使其更加精确：在具有两种行动和两种信号的情况下，除非两位参与者有一个共同的第一优选计划，否则代理人的信息对规划者而言毫无价值。

为了用数值方法验证这两个结果，我们随机抽取一些经济体，使两位参与者的第一优选计划至少在一个信号之后有所不同，因为否则规划者会平凡地达到她的第一优选结果。

两位参与者的效用是正相关的，所以他们经常但并非总是意见一致。

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

在每一次抽取中，线性规划的值都等于定理所预测的值 $\max\{v_0, v_A(\Lambda)\}$，误差仅来自舍入。

在具有两种行动和两种信号的情况下，它总是等于 $v_0$，正如 {prf:ref}`it_thm_two_signals` 的加强版本所说的那样。

在信号更多的情况下，听任代理人行事有时会优于忽略他，但在大多数抽取中并非如此。

## 被支配行动作为威胁

在只有两种行动的情况下，一个彩票就是采取 $a_1$ 的某个概率，所以代理人对彩票的排序是由他对各行动的偏好所固定的，而规划者最多只能忽略他或听任他行事。

在有三种行动的情况下，即使代理人对各行动本身的排序没有改变，他对彩票的排序也可能随着后验的变化而变化，这使得规划者能够做得比这两种方案都更好。

格林的第一个例子有两种状态、两种信号和三种行动。

先验是均匀的，且

$$
\Lambda = \begin{bmatrix} 3/4 & 1/4 \\ 1/4 & 3/4 \end{bmatrix},
\qquad
U = \begin{bmatrix} -2 & 3 \\ 0 & 0 \\ -1 & -1 \end{bmatrix},
\qquad
W = \begin{bmatrix} 0 & -1 \\ 2 & 0 \\ 0 & -4 \end{bmatrix} .
$$

$\theta_1$ 的后验概率在 $y_1$ 之后为 $3/4$，在 $y_2$ 之后为 $1/4$。

如果规划者亲自看到信号，她会在 $y_1$ 之后采取 $a_2$，在 $y_2$ 之后采取 $a_1$。

代理人无论哪种信号都偏好 $a_2$，因此如果规划者宣布她自己的第一优选计划，一个看到 $y_2$ 的代理人会报告 $y_1$。

行动 $a_3$ 在两种状态下都给规划者带来 $-1$ 的收益，而 $a_2$ 带来 $0$，所以一个亲自看到信号的规划者永远不会采取 $a_3$。

让我们来求解规划者的线性规划。

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

最优计划在报告为 $y_2$ 之后采取 $a_1$；在报告为 $y_1$ 之后，以 $9/14$ 的概率采取 $a_2$，以 $5/14$ 的概率采取被支配行动 $a_3$。

规划者的预期效用为 $\hat v(\Lambda) = 39/56 \approx 0.696$，介于她忽略代理人所得的 $v_0 = 0.5$ 与她亲自看到信号所得的 $v(\Lambda) = 0.875$ 之间。

格林报告的数值是 $0.696$，概率分别为 $0.643$ 和 $0.357$。

为什么被支配的行动会有所帮助呢？

该计划必须阻止一个看到 $y_2$ 的代理人报告 $y_1$，所以它在报告 $y_1$ 上附加了一个这个代理人所厌恶的行动。

按照每个候选威胁对规划者希望在 $y_1$ 之后采取的行动 $a_2$ 所造成的损失来衡量它们。

在如实的后验 $3/4$ 处，行动 $a_3$ 使规划者付出代价 $1$，而在一个说谎代理人的后验 $1/4$ 处，它使他付出代价 $1/2 - (-3) = 7/2$。

行动 $a_1$ 只会使规划者付出代价 $3/4$，但只会使说谎的代理人付出代价 $1/2 - (-3/4) = 5/4$。

所以 $a_3$ 相对于规划者每单位代价对说谎者施加的代价为 $7/2$，而 $a_1$ 为 $5/3$。

它是更便宜的威胁，因为它在状态 $\theta_2$ 中对代理人伤害最大，而看到 $y_2$ 的代理人认为该状态是可能发生的。

一个看到 $y_2$ 的代理人如实报告能得到 $-3/4$，所以这个威胁必须将他谎称看到 $y_1$ 所得到的收益降低 $1/2 - (-3/4) = 5/4$。

用 $a_3$ 时概率为 $(5/4)/(7/2) = 5/14$，这恰好使他在说实话和撒谎之间无差异；用 $a_1$ 时则需要概率为1。

随机化和被支配行动必须同时使用。

确定性计划会失败：无论哪种信号之后，代理人对各行动的排序都是相同的，即 $a_2$、然后是 $a_1$、再然后是 $a_3$，所以任何在两种报告之后采取不同行动的计划都会诱使看到某一信号的代理人撒谎，而在两种报告之后都采取同一行动的最佳计划只值 $v_0$。

如果没有 $a_3$，就只有两种行动，而代理人在两种后验下都偏好 $a_2$，所以根据 {prf:ref}`it_thm_two_actions` 背后的论证，如实报告迫使计划对两种报告一视同仁；满足这一条件的最佳计划在两种报告之后都采取 $a_1$，价值只有 $v_0$。

```{code-cell} ipython3
v_no_threat = solve_planner(U_ex1[:2], W_ex1[:2], Λ_ex1, π_ex1)[1]
print(f"planner's value without a_3: {v_no_threat:.4f}")
```

下图绘制了每位参与者在每种行动下的预期效用对 $\theta_1$ 后验概率的函数。

虚线是该计划对报告 $y_1$ 所附加彩票的预期效用，点状竖线标出了两个后验值。

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

在右侧面板中，虚线恰好在后验 $1/4$ 处与代理人关于 $a_1$ 的直线相交，而正是在这个后验上，看到 $y_2$ 的代理人处于无差异状态。

在后验 $3/4$ 处，虚线远高于 $a_1$ 的直线，所以看到 $y_1$ 的代理人严格地偏好如实报告。

在左侧面板中，虚线位于 $a_2$ 的直线之下。

在后验 $3/4$ 处的差距 $5/14$，就是规划者在报告 $y_1$ 之后的损失。

由于 $y_1$ 的概率为 $1/2$，她为了获取真实信息所付出的预期成本为 $5/28 = v(\Lambda) - \hat v(\Lambda)$。

## 如实但粗略的报告

格林的第二个例子展示了激励如何以另一种方式使规划者付出信息代价。

有三种概率相等的信号，格林直接给出了预期效用矩阵：

$$
WP = \begin{bmatrix} 1 & 3/2 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix},
\qquad
UP = \begin{bmatrix} 5 & -1 & -1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} .
$$

我们可以通过三种等可能的状态、均匀先验和一个完全信息的实验 $\Lambda = I$（对应 $P = I$）来表示这些内容。

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

规划者只希望在 $y_1$ 之后采取 $a_1$，但一个看到 $y_2$ 的代理人也喜欢 $a_1$，它对他的价值为 $3/2$，超过其他任何一种行动。

假设在报告 $y_1$ 之后，该计划以概率 $\alpha$ 采取 $a_1$，其余部分采取 $a_3$（即看到 $y_2$ 的代理人最不喜欢的行动），而在报告 $y_2$ 之后，它以概率 $\beta$ 采取 $a_1$，否则采取 $a_2$。

一个看到 $y_2$ 的代理人如实报告能得到 $1 + \beta/2$，谎称 $y_1$ 能得到 $3\alpha/2$，所以如实报告要求 $\beta \geq 3\alpha - 2$。

所以，在报告 $y_2$ 之后绝不采取 $a_1$ 的最佳计划，在报告 $y_1$ 之后以概率 $2/3$ 采取 $a_1$，这对规划者价值为 $10/9$。

超过 $\alpha = 2/3$ 之后，报告 $y_1$ 之后每增加一单位的 $a_1$ 概率能为规划者带来 $5$ 的收益，但会迫使报告 $y_2$ 之后三单位的概率落在 $a_1$ 上，每单位都使她付出代价 $1$，所以她将 $\alpha$ 一直提高到 $1$。

因此最优计划在报告 $y_1$ 和 $y_2$ 之后给出相同的彩票——确定采取 $a_1$，而线性规划证实在这个家族之外没有任何计划能做得更好。

代理人如实报告，但规划者对前两种信号一视同仁，获得 $4/3$，介于忽略代理人的 $1$ 与第一优选的 $5/3$ 之间。

$Z$ 的第三行并没有被确定：在报告 $y_3$ 之后，规划者在 $a_2$ 和 $a_3$ 之间是无差异的，而代理人的预期效用取决于她采取哪一个。

```{code-cell} ipython3
for ties in ("agent_worst", "agent_best"):
    agent = solve_planner(U_ex2, W_ex2, Λ_ex2, π_ex2, ties=ties)[2]
    print(f"{ties}: agent {agent:.4f}")
```

在规划者的各个最优计划之间，代理人的预期效用在 $5/6$ 到 $7/6$ 之间变化。

## 比较各个实验

### 通过后验看布莱克韦尔序

为了探讨规划者是否希望她的代理人信息更灵通，格林需要一种对实验进行排序的方法。

设 $\Lambda'$ 是另一个实验，具有 $N'$ 个信号 $y'_1, \ldots, y'_{N'}$，用撇号标记从中计算出的所有量，比如 $q'$ 和 $P'$。

根据 {prf:ref}`thm-blackwell`，当且仅当存在一个 $N \times N'$ 的马尔可夫矩阵 $B$ 使得 $\Lambda' = \Lambda B$ 时，对于每一个效用矩阵 $U$ 和每一个先验 $\pi$，都有 $v(\Lambda) \geq v(\Lambda')$。

当这成立时，称 $\Lambda$（弱）**信息量更大**，而 $\Lambda'$ 是 $\Lambda$ 的**粗化**。

```{note}
在 {doc}`blackwell_kihlstrom` 中，实验写作 $\mu$ 而非 $\Lambda$，粗化矩阵是随机变换 $Q$ 而非 $B$；该讲座中 $B(\mu, A, u)$ 表示一组可达到的预期效用向量。

该讲座还用 $N$ 来计数状态、用 $M$ 来计数信号，将先验写作 $p$，并用 $U$ 表示不确定性函数而非效用矩阵。
```

正如格林所指出的，布莱克韦尔条件可以通过求解关于 $B$ 的线性不等式来检验，这比对所有效用矩阵比较决策者要容易得多。

不过，为了格林的目的，一个用后验表示的版本更为有用，因为代理人的激励取决于他的后验。

{cite:t}`MarschakMiyasawa1968` 表明，在固定具有完全支持的先验的情况下，当且仅当存在两个信号的联合分布 $J = [J_{nn'}]$（边际分布为 $q$ 和 $q'$）满足

$$
\sum_{n=1}^N p_{mn} \, J_{nn'} = p'_{mn'} \, q'_{n'}
\quad \text{for all } m \text{ and } n' .
$$ (it_barycenter)

时，$\Lambda$ 的信息量才大于 $\Lambda'$。

于是，信息量较小的实验的每一个后验，都是信息量较大的实验的后验的一种加权平均，权重为 $J_{nn'}/q'_{n'}$，即给定 $y'_{n'}$ 时 $y$ 的条件分布。

我们称 {eq}`it_barycenter` 为**平均化条件**。

这正是基尔斯特罗姆在 {doc}`blackwell_kihlstrom` 中所强调的性质：在 $\Lambda$ 之下后验的分布是在 $\Lambda'$ 之下后验分布的一种保均值展形。

格林用这些权重的矩阵 $C = J \operatorname{diag}(q')^{-1}$（其各列均为概率向量）来表述这个条件：

$$
P' = P C
\qquad \text{and} \qquad
q = C q' .
$$ (it_mm)

这是格林的方程（12），取自马尔沙克和宫泽。

给定 $J$，矩阵 $B = \operatorname{diag}(q)^{-1} J$（给定 $y$ 时 $y'$ 的条件分布）即为布莱克韦尔的粗化。

这是格林的方程（13），对应于马尔沙克和宫泽的方程（8.23）。

要验证 $\Lambda' = \Lambda B$，可利用 {eq}`it_bayes` 中的 $\lambda_{mn}/q_n = p_{mn}/\pi_m$，再结合 {eq}`it_barycenter`：

$$
(\Lambda B)_{mn'} = \frac{1}{\pi_m} \sum_n p_{mn} J_{nn'} = \frac{p'_{mn'} \, q'_{n'}}{\pi_m} = \lambda'_{mn'} .
$$

### 一个运输问题

求 $J$ 是一个线性可行性问题，我们使用 {doc}`linear programming <intro:lp_intro>` 中的工具来求解。

若没有平均化条件，$J$ 可以是 {doc}`opt_transport` 中的任何一个运输方案，即边际分布为 $q$ 和 $q'$ 的任意联合分布，而独立耦合 $J = q q'^\top$ 总是可行的。

平均化条件是一组额外的线性约束，它使得该运输方案成为两个后验分布之间的**鞅耦合**。

这里没有需要最小化的成本，所以我们给 `linprog` 传入一个零目标函数。

列和约束是由 {eq}`it_barycenter` 蕴含的，因为 $P$ 每一列中的后验之和为1，所以我们只施加行和约束以及 {eq}`it_barycenter`。

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

与 {doc}`blackwell_kihlstrom` 中的最小二乘搜索不同，这个函数要么（在求解器的数值容差范围内）精确地找到一个粗化，要么报告不存在这样的粗化。

## 更好的信息可能损害规划者

### 格林的第三个例子

当代理人的信息改善时，有两件事会发生变化。

规划者可以凭借更好的信号做出更好的决策，但更好的信号也会改变代理人愿意如实作出哪些报告。

格林的第三个例子表明，第二种效应可能占据主导地位。

有两种状态，先验为均匀分布，两种行动，且

$$
U = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix},
\qquad
W = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix} .
$$

当 $\theta_1$ 的后验概率超过 $2/3$ 时，规划者偏好 $a_1$，而当其超过 $1/3$ 时，代理人偏好 $a_1$。

粗糙实验 $\Lambda'$ 具有两种信号，$\theta_1$ 的后验分别为 $3/4$ 和 $1/4$，每种概率为 $1/2$。

精细实验 $\Lambda$ 具有三种信号，后验分别为 $3/4$、$1/2$ 和 $0$，概率分别为 $1/2$、$1/4$ 和 $1/4$。

下面的函数根据给定的 $\theta_1$ 的后验列表及其概率，构造一个两状态似然矩阵。

这些概率之和必须为1，且各后验的加权平均值必须等于 $\theta_1$ 的先验概率；否则 $\Lambda$ 的各行之和不为1。

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

粗化矩阵 $B$ 将精细实验的第二个和第三个信号合并，所以精细实验信息量更大，而反过来的粗化并不存在。

让我们来比较规划者的值。

```{code-cell} ipython3
for name, Λ in (("coarse", Λ_coarse), ("fine", Λ_fine)):
    v_hat = solve_planner(U_ex3, W_ex3, Λ, π_ex3)[1]
    print(f"{name}: planner {v_hat:.4f}, "
          f"first best {bayes_value(U_ex3, Λ, π_ex3):.4f}")
print(f"no information: {no_info_value(U_ex3, π_ex3):.4f}")
```

在粗糙实验下，两位参与者在两种信号之后都意见一致，因为 $3/4$ 超过两个临界值，而 $1/4$ 低于两个临界值。

所以规划者宣布她自己的第一优选计划，代理人如实报告，两位参与者各获得 $9/4$。

精细实验将后验为 $1/4$ 的信号拆分为后验分别为 $1/2$ 和 $0$ 的信号。

在后验 $1/2$ 处，两位参与者意见不一致：规划者想要 $a_2$，而代理人想要 $a_1$。

根据 {prf:ref}`it_thm_two_actions`，规划者要么忽略代理人，要么听任他行事。

这两种选择对规划者的价值都是 $2$，低于 $9/4$，所以规划者宁愿她的代理人知道得更少一些。

事实上，介于这两者之间的每一个计划也都是最优的，代理人的收益沿着这些计划变化。

```{code-cell} ipython3
for ties in ("agent_worst", "agent_best"):
    Z, v_hat, agent = solve_planner(U_ex3, W_ex3, Λ_fine, π_ex3, ties=ties)
    print(f"{ties}: planner {v_hat:.4f}, agent {agent:.4f}, plan\n{Z}")
```

在后验为 $0$ 的信号之后采取 $a_2$，而在另外两个信号之后以共同的概率 $z \in [0, 1]$ 采取 $a_1$，这样的每一个计划对规划者都值 $2$。

沿着这些计划，代理人的预期效用为 $1 + 3z/2$，当规划者忽略他时为 $1$，当她听任他行事时为 $5/2$。

代理人也有理由秘密获取这种更精细的信息。

假设规划者相信代理人拥有粗糙实验，并为此宣布她自己的第一优选计划，而代理人实际上观察到的是精细信号。

在后验为 $1/2$ 的信号之后，代理人偏好 $a_1$，所以他报告后验为 $3/4$ 的粗糙信号。

他的预期效用从 $9/4$ 上升到 $5/2$，而规划者的预期效用从 $9/4$ 下降到 $2$。

这说明了一个普遍的道理：对于一个固定的计划，代理人是一个从 $Z$ 的各行所构成的菜单中进行选择的单一决策者，所以根据布莱克韦尔定理，更好的信息永远不会损害他。

当更好的信息确实损害了代理人时，那是因为规划者知道代理人知道什么，从而选择了不同的计划。

{cite:t}`Postlewaite1982` 针对他自己的例子提出了关于秘密获取信息的这一点，并提出了他所称的"关于信息的信息"问题：只有当规划者知道代理人并没有秘密改进他的信息时，她才能依赖粗糙实验。

### 一族精细化方案

我们可以通过将后验为 $1/4$ 的粗糙信号拆分为后验分别为 $s$ 和 $0$ 的信号（其中 $s$ 介于 $1/4$ 和 $1$ 之间），来更全面地看出这种权衡。

后验为 $s$ 的信号的概率为 $1/(8s)$，这使得平均后验保持等于先验。

提高 $s$ 会使得该实验在布莱克韦尔意义下信息量更大，下面的代码沿着网格对此进行了验证。

该网格从刚好高于 $1/4$ 处开始，因为在 $s = 1/4$ 时，后验为 $0$ 的信号概率为零，此时的精细化方案就是粗糙实验本身。

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

下图绘制了规划者由线性规划得到的值 $\hat v$ 以及她的第一优选值 $v$ 关于 $s$ 的函数。

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

第一优选值 $v$ 随着 $s$ 的上升永不下降，这是布莱克韦尔定理所要求的，但规划者的值 $\hat v$ 却下降了。

对于 $s < 1/3$，两位参与者在任一精细信号之后都想要 $a_2$，所以这种精细化方案是无害的，也是无用的。

对于 $1/3 < s < 2/3$，两位参与者在后验为 $s$ 的信号之后意见不一致。

此时规划者要么忽略代理人，价值为 $2$，要么听任他行事，价值为 $3 - 1/(2s)$。

所以 $\hat v$ 在 $s = 1/3$ 处从 $9/4$ 跳跌到 $2$，并一直保持在 $2$，直到 $s = 1/2$，然后随着 $s$ 接近 $2/3$ 而重新攀升到 $9/4$ 附近。

只有当 $s > 2/3$ 时——此时规划者和代理人再次意见一致——这种精细化方案才改善了规划者的决策。

### 只有两种信号的实验

在格林的例子中，损害源于一种把一个信号拆分为两个的精细化方案。

他的最后一个结果表明，当两个实验都是**二项信道**（格林对只有两种信号的实验所用的术语）时，这种情况不会发生。

```{prf:theorem} 二项信道（格林的定理4）
:label: it_thm_binary

如果 $\Lambda$ 比 $\Lambda'$ 信息量更大，且两者都具有两种信号，那么无论状态和行动的数量如何，都有 $\hat v(\Lambda) \geq \hat v(\Lambda')$。
```

下面是一个证明。

设 $p_1$ 和 $p_2$ 是 $\Lambda$ 的后验，$p'_1$ 和 $p'_2$ 是 $\Lambda'$ 的后验。

根据 {eq}`it_barycenter`，每一个 $p'_{n'}$ 都是 $p_1$ 和 $p_2$ 的一种平均，所以两者都位于连接 $p_1$ 和 $p_2$ 的线段上。

如果 $p'_1 = p'_2$，那么 $\Lambda'$ 是无信息量的，$\hat v(\Lambda') = v_0$，无需再证。

否则，为 $\Lambda'$ 的信号标注标签，使得 $p'_1$ 比 $p'_2$ 更靠近 $p_1$。

设 $Z$ 是 $\Lambda'$ 下的一个最优计划，$\ell_1$ 和 $\ell_2$ 分别是其对应于报告 $y'_1$ 和 $y'_2$ 的行所表示的彩票。

在 $\Lambda$ 之下采用同样的计划，在报告 $y_n$ 之后采取彩票 $\ell_n$。

在后验 $p$ 处，代理人从彩票 $\ell_1$ 相对于彩票 $\ell_2$ 所获得的收益为 $\phi(p) = (\ell_1 - \ell_2) W p$，规划者的收益为 $\psi(p) = (\ell_1 - \ell_2) U p$。

两者都是关于 $p$ 的线性函数。

1. 在 $\Lambda'$ 下如实报告意味着 $\phi(p'_1) \geq 0 \geq \phi(p'_2)$。这四个后验以 $p_1, p'_1, p'_2, p_2$ 的顺序位于该线段上，所以一个在 $p'_1$ 和 $p'_2$ 处具有这些符号的线性函数，在 $p_1$ 和 $p_2$ 处也保持这些符号。因此 $\phi(p_1) \geq 0 \geq \phi(p_2)$，该计划使得在 $\Lambda$ 下如实报告也是最优的。
2. $Z$ 在 $\Lambda'$ 下的最优性意味着 $\psi(p'_1) \geq 0 \geq \psi(p'_2)$。因为如果 $\psi(p'_1) < 0$，规划者本可以在两种报告之后都使用彩票 $\ell_2$ 做得更好，这是一个忽略代理人因而满足如实报告约束的计划；若 $\psi(p'_2) > 0$，情况类似。根据第1步的论证，$\psi(p_1) \geq 0 \geq \psi(p_2)$。
3. 设 $J$ 是 {eq}`it_barycenter` 中两个信号的联合分布。由于 $q'_{n'} p'_{n'} = \sum_n J_{nn'} \, p_n$ 且 $q_n = \sum_{n'} J_{nn'}$，规划者在 $\Lambda$ 下采用该计划所得的预期效用减去在 $\Lambda'$ 下所得的预期效用为

   $$
   \sum_n q_n \ell_n U p_n - \sum_{n'} q'_{n'} \ell_{n'} U p'_{n'}
   = \sum_{n, n'} J_{nn'} (\ell_n - \ell_{n'}) U p_n
   = J_{12} \, \psi(p_1) - J_{21} \, \psi(p_2) \;\geq\; 0 .
   $$

规划者在 $\Lambda$ 下至少能做得和这个计划一样好，所以 $\hat v(\Lambda) \geq \hat v(\Lambda')$。

```{note}
格林的证明走的是同一条路径，但有两个漏洞。

首先，它为信号标注标签，使得粗化矩阵 $B$ 的每一行都在对角线上取得最大值。

这并非总是可能的：对于 $B = \begin{bmatrix} 0.9 & 0.1 \\ 0.6 & 0.4 \end{bmatrix}$，无论如何标注标签，两行的最大元素都在同一列上。

论证所需要的那种标注方式正是上面所使用的方式，它等价于 $b_{11} \geq b_{21}$，即 $\det B \geq 0$。

其次，格林最后一步将一个仅为代理人证明的符号性质用在了规划者的收益上；上面的第2步则从 $Z$ 的最优性中补足了这一点。
```

为了用数值方法验证这个定理，我们随机抽取最多具有四种状态、三种或四种行动的二项信道。

我们只保留那些定理确实有所作用的抽样结果。

如果 $\hat v(\Lambda') = v_0$ 或 $\hat v(\Lambda) = v(\Lambda)$，那么结论仅凭 $v_0 \leq \hat v \leq v$ 以及布莱克韦尔定理即可得出，所以我们跳过这些抽样结果。

在只有两种行动的情况下，根据 {prf:ref}`it_thm_two_signals`，$\hat v$ 要么是 $v_0$ 要么是 $v$，所以如果该定理成立，每一次抽样都会被跳过；这就是为什么我们使用三种或四种行动。

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

在这些抽样结果中，规划者在每一次抽样中都从更好的信息中获益。

当精细实验具有三种或更多信号时，该定理并不适用，但一个损失的产生仍然需要两位参与者在精细实验的某个后验处意见不一致，就像在 $1/3 < s < 2/3$ 时的 $s$ 那样；否则规划者就能达到 $\hat v(\Lambda) = v(\Lambda) \geq v(\Lambda') \geq \hat v(\Lambda')$。

## 评论者补充的内容

{cite:t}`Postlewaite1982` 构造了四状态的例子，即他的情形I到情形IV，其中信息量更大的实验可能使两位参与者都受益、都受损、使代理人受益而规划者受损、或使规划者受益而代理人受损。

{ref}`it_ex_postlewaite` 用线性规划对这些情形进行了检验。

在我们的随机抽样中，情形I和情形IV总是如他所说的那样。

在情形II中，当 $h$ 接近 $j$（正如波斯特怀特所假设的那样）时，代理人在每一次抽样中都受损，但规划者只在大约一半的抽样中受损。

在情形III中，代理人在每一次抽样中都受损而非受益，而规划者则在部分抽样中受益、在部分抽样中受损。

波斯特怀特问道，为什么代理人不能简单地忽略更好的信息，他的回答是：规划者知道代理人是知情的，可以设计她的计划，使得一个忽略了信息的代理人处境更糟。

正如我们在格林的第三个例子之后所看到的，在一个固定的计划下，更好的信息不可能损害代理人，所以"处境更糟"总是以弱的方式成立。

在他的例子中，这种关系仅以弱的方式成立：{ref}`it_ex_postlewaite` 研究了一族计划，它在情形II和情形III的例子中都包含了规划者的最优计划，并表明在该族计划中她所偏好的计划之下，一个忽略了精细信息的代理人所得到的收益，恰好等于他利用该信息所得到的收益。

代理人之所以受损，仅仅是因为规划者知道他信息更灵通，从而选择了不同的计划；这就是为什么波斯特怀特补充说，如果代理人能够说服规划者相信他已经这样做了，他就愿意承诺采用信息量较小的实验。

波斯特怀特还问道，是谁来选择实验，以及如果是代理人而不是规划者先行承诺，会发生什么。

他断言，构造出两位参与者在角色互换后都变得更好的例子是容易的；{ref}`it_ex_reverse` 在代理人按照格林所定义的方式承诺于一种确定性应答规则时，证实了这一点。

{cite:t}`Riley1982` 用关于预期效用对后验的图表（类似于 {numref}`fig-it-dominated`）解释了格林的例子，并给出了一个两状态的例子，在该例中，将一个被粗化的信息解析清楚反而使两位参与者都变得更糟。

他总结道，"除非以某种方式对该模型加以精细化，否则我们不能指望得到的仅仅是一堆特殊情形"，并提出了两种精细化方式。

一种是向代理人支付"视最终自然状态而定的报酬"。

另一种是考虑"信息交换的重复性质"，因为长期关系——比如经纪人与客户之间，或医生与病人之间的关系——能让双方了解到彼此的利益所在。

重复博弈也能让声誉替代承诺的作用，尽管 {doc}`phillips_credible_policies` 表明，声誉可以支撑许多结果，而不仅仅是承诺所能带来的那一个。

{ref}`it_ex_riley` 表明，当效用关于货币是线性的时，仅依赖状态的报酬无法改变代理人的报告，而依赖报告和状态两者的报酬则会减少（但在莱利的例子中并不能消除）更好信息所带来的损害。

## 与其他讲座的关系

{doc}`pricing_information` 研究了一个设计并定价布莱克韦尔实验的卖方，她用一个实验菜单来甄别先验信念是私人信息的买方。

格林的规划者同样用菜单进行甄别，但该菜单提供的是对规划者自身行动的彩票，而私人信息则是代理人的信号。

在 {doc}`information_market_equilibrium` 中，局外的观察者可以从均衡价格中推断出内部人的私人信息，所以信息是通过市场而非通过报告来传递的。

在 {doc}`util_rand_resp` 中，随机应答调查的设计者有意使每个答案的信息量都变小，从而使受访者愿意如实作答。

设计者宁愿接受一个被粗化但如实的答案，而不要一个更清晰但不如实的答案，这与格林第三个例子中规划者更喜欢一个拥有较粗糙实验的代理人颇为相似。

格林的规划者无法向代理人支付报酬，所以她是用自身行动上的扭曲来换取如实的报告。

当可以转移支付时，就像 {doc}`house_auction` 和 {cite:t}`Groves_73` 中的格罗夫斯-克拉克机制那样，支付可以替代这一作用；{ref}`it_ex_riley` 将支付加入了格林的模型中。

### 关于沟通与授权的后续研究

格林的模型属于一个策略性沟通模型的大家族，其成员的区别在于谁对什么做出承诺。

在这一文献的语言中，代理人是**发送方**，规划者是**接收方**。

* 在格林的模型中，接收方承诺了一个根据报告采取行动的计划。
* 在 {cite:t}`CrawfordSobel1982` 的廉价交谈模型中，没有人做出承诺，接收方根据所听到的内容采取最优行动。
* 在 {cite:t}`KamenicaGentzkow2011` 的说服模型中，发送方承诺产生信号的那个实验。
* 在波斯特怀特的角色互换（于 {ref}`it_ex_reverse` 中研究）中，发送方承诺一个报告他所观察到内容的确定性规则，这相当于将说服限制在 $\Lambda$ 信号的分割之上；如果该规则可以随机化，则相当于将说服限制在 $\Lambda$ 的粗化之上。

莱利将格林的计划解读为一个菜单，这将它与 {cite:t}`Holmstrom1984` 的授权问题联系了起来，在该问题中，委托人选择一组行动，让代理人从中挑选一个；而格林的菜单所包含的则是对行动的彩票。

{cite:t}`MelumadShibano1991` 研究了在排除转移支付时有承诺和无承诺的沟通情形，{cite:t}`AlonsoMatouschek2008` 则刻画了最优的授权行动集合。

{cite:t}`KovacMylovanov2009` 允许像格林那样的随机化计划，并表明在二次支付函数和某个正则性条件下，最优计划不会随机化。

格林关于规划者可能偏好一个信息较少的代理人这一发现也有后续研究的延伸。

{cite:t}`FischerStocken2001` 表明，在廉价交谈中，一个信息较不精确的发送方反而可能使接收方获得更充分的信息，{cite:t}`Ivanov2010` 研究了一个在代理人进行沟通之前就限制其信息精度的委托人。

## 结束语

格林的模型将看到信号的人与根据信号采取行动的人区分开来。

承诺使规划者能够将代理人的激励问题转化为一个线性规划问题，其解可以随机化、使用被支配的行动作为威胁，并将信号合并处理。

由于激励约束取决于后验，实验之间的比较同样要通过后验来进行，而马尔沙克-宫泽的刻画方式将布莱克韦尔序转化为一个运输问题。

核心的教训关乎信息的价值。

布莱克韦尔定理对一个亲自看到信号的决策者来说为各实验排了序，但一旦信号必须从他人处获取，一个信息更灵通的代理人对规划者而言其价值反而可能更低——除非两个实验都只有两种信号。

## 练习

```{exercise}
:label: it_ex_riley

{cite:t}`Riley1982` 用一个两状态、两行动例子的图示，说明了更好的信息如何能同时损害两位参与者，他将信号称为"信息"。

下面这些数据再现了他图C3.4中的模式。

莱利的正文中将这两种行动与他图中的标注方式相反；我们的标注遵循图中的方式。

先验是均匀的，以 $p$ 表示 $\theta_1$ 的后验概率，规划者从 $a_1$ 得到的预期效用为 $p$，从 $a_2$ 得到的为 $0.7$，而代理人从 $a_1$ 得到的为 $p$，从 $a_2$ 得到的为 $(1 - p)/2$。

在粗糙实验中，代理人的后验为 $0.8$ 或 $0.2$，每种概率均为 $1/2$。

在精细实验中，后验为 $0.2$ 的信号被解析为后验分别为 $0.4$ 和 $0$ 的信号，每种概率均为 $1/4$。

1. 使用 `garbling` 来验证精细实验比粗糙实验信息量更大。然后在两种实验下分别求解规划者的线性规划，比较两位参与者的预期效用，并利用 {prf:ref}`it_thm_two_actions` 解释这一结果。
2. 莱利建议向代理人支付"视最终自然状态而定的报酬"。假设两位参与者的效用都关于货币是线性的，所以支付会增加代理人的效用，并减少规划者的效用。证明仅依赖状态的支付不能改变代理人的报告。
3. 现在让规划者在代理人报告 $y_r$ 且状态最终为 $\theta$ 时支付 $t(r, \theta) \geq 0$。将线性规划扩展为同时选择计划和支付方案，并对精细实验求解。当支付只能依赖于报告时也这样做。支付是否消除了更好信息所带来的损害？
```

```{solution-start} it_ex_riley
:class: dropdown
```

下面给出一种解法。

对于第1部分，我们构造两个实验，检验粗化关系，并在每种实验下求解线性规划。

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

在粗糙实验下，两位参与者在两种信号之后都意见一致：在 $0.8$ 处都想要 $a_1$，在 $0.2$ 处都想要 $a_2$。

所以规划者得到 $0.75$，代理人得到 $0.6$。

在精细实验下，两者在后验 $0.4$ 处意见不一致，此时规划者想要 $a_2$，而代理人想要 $a_1$。

根据 {prf:ref}`it_thm_two_actions`，规划者要么听任代理人行事（这对她价值 $0.675$，对他价值 $0.625$），要么忽略他（这对她价值 $0.7$）。

她选择忽略他，总是采取 $a_2$，这只给代理人留下 $0.25$。

将信号解析清楚使得两位参与者都变得更糟，正如莱利图中所示。

下图的结构与莱利的图相同。

在每个面板中，实线是一位参与者从两种行动中获得的预期效用，点状竖线标出了先验 $1/2$。

在粗糙实验下，规划者在 $0.2$ 处采取 $a_2$，在 $0.8$ 处采取 $a_1$，所以每位参与者的预期效用，就是连接该参与者在这两个后验处效用的虚线弦（实心点）在先验处的高度。

在精细实验下，规划者总是采取 $a_2$，所以每位参与者的预期效用，就是该参与者关于 $a_2$ 的直线在先验处的高度（空心点）。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Riley's example, with expected utilities read off at the prior
    name: fig-it-riley
---
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

在两个面板中，空心点都位于实心点下方：对规划者而言是 $0.70$ 对 $0.75$，对代理人而言是 $0.25$ 对 $0.6$。

对于第2部分，一个在状态 $\theta_m$ 下支付 $t(\theta_m)$ 的方案，会使 $W$ 的第 $m$ 列中每一个元素都增加 $t(\theta_m)$。

在观察到 $y_n$ 之后，代理人在每一种报告中都会额外得到 $\sum_m p_{mn} t(\theta_m)$，所以这种支付方案不会改变哪种报告是最好的。

更根本的是，它不会改变代理人在任何状态下对行动的排序，所以它无法减少他与规划者之间的冲突。

要改变代理人的激励，支付方案必须依赖于他所能控制的某些东西，比如他的报告。

对于第3部分，我们将支付加入决策向量。

依赖于报告 $r$ 和状态的支付方案，会使代理人在信号 $y_n$ 之后对报告 $y_r$ 的预期效用增加 $\sum_m p_{mn} t(r, \theta_m)$；仅依赖于报告的支付方案则增加 $t(r)$。

所以如实报告约束以及规划者的预期支付仍然是线性的。

有些支付——比如在后验为 $0$ 的报告下且状态为 $\theta_1$ 时的支付——在代理人如实报告时永远不会被执行，所以它们不会使规划者付出任何代价，也不会被确定下来；给每一笔支付都加上一个微小的代价，就能将这种平局打破为零。

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

在依赖报告和状态两者的支付方案下，规划者宣布她的第一优选计划，并在代理人报告后验 $0.4$ 且状态为 $\theta_1$ 时支付 $0.25$。

一个看到 $0.4$ 的代理人期望从这笔支付中获得 $0.4 \times 0.25 = 0.1$，这恰好等于他谎称看到 $0.8$ 所能获得的收益 $0.4 - 0.3$。

这种支付之所以便宜，是因为它是针对他而设计的：一个看到 $0$ 的代理人对此没有任何期望，而一个看到 $0.8$ 的代理人也不会受到诱惑。

预期支付为 $(1/4)(0.4)(0.25) = 0.025$，所以规划者得到 $0.725$。

仅依赖报告的支付方案中，规划者在报告 $0.4$ 之后支付 $1/12$，同时也在那里以 $1/6$ 的概率采取 $a_1$，净值为 $43/60 \approx 0.717$。

无论哪种方式，支付都缩小了更好信息所带来的损害，但并未将其消除：两个值都介于规划者在无支付方案下所得的 $0.70$，与她在无需支付的粗糙实验下所得的 $0.75$ 之间。

这些支付方案有两个值得关注的特点。

依赖状态的支付要求事后能够观察到该状态，就像股票的回报那样。

而且这些支付必须是非负的：格林的模型没有参与约束，所以如果没有这一限制，规划者可以向代理人收取任意大的费用，线性规划就会变得无界。

```{solution-end}
```

```{exercise}
:label: it_ex_postlewaite

{cite:t}`Postlewaite1982` 研究了四种等可能的状态 $\theta_1, \ldots, \theta_4$ 以及四种行动。

用规划者的正参数 $b, c, d, e$ 和代理人的正参数 $f, g, h, j$，收益矩阵（行是行动，列是状态）为

$$
U = 2 \begin{bmatrix} b & 0 & c & 0 \\ 0 & c & 0 & b \\ d & 0 & e & 0 \\ 0 & e & 0 & d \end{bmatrix},
\qquad
W = 2 \begin{bmatrix} 0 & f & 0 & g \\ g & 0 & f & 0 \\ h & 0 & j & 0 \\ 0 & j & 0 & h \end{bmatrix} .
$$

在精细实验中，代理人能观察到状态本身。

在粗糙实验中，他只能得知状态是在 $\{\theta_1, \theta_2\}$ 中还是在 $\{\theta_3, \theta_4\}$ 中。

1. 波斯特怀特的情形II假设 $b > d + e > (b + c)/2 > c$ 且 $f > h + j > (f + g)/2 > g$，他得出结论：如果上述不等式成立，并且如果"$h$ 充分接近 $j$"，那么精细实验会使两位参与者都变得更糟。取 $(b, c, d, e) = (10, 4, 5, 3)$ 和 $(f, g, h, j) = (10, 4.5, 3.9, 4)$。使用 `garbling` 验证精细实验比粗糙实验信息量更大。在两种实验下分别求解规划者的线性规划，并比较两位参与者的预期效用。精细实验下的最优计划将多少概率赋予了行动 $a_3$ 和 $a_4$（当 $b > d$ 且 $c > e$ 时，这两个行动对规划者而言是被支配的）？
2. 考虑精细实验下这样一族计划：在报告 $\theta_1$ 和 $\theta_3$ 之后，以概率 $x$ 采取 $a_1$，否则采取 $a_3$；在报告 $\theta_2$ 和 $\theta_4$ 之后，以概率 $x$ 采取 $a_2$，否则采取 $a_4$。证明如实报告是最优的当且仅当 $x \leq x^* = \min\{h/(h + g), \, j/(j + f)\}$，并求出两位参与者的预期效用作为 $x$ 的函数。波斯特怀特写道，能够赋予 $a_1$ 和 $a_2$ 的概率"随着 $h$ 任意接近于 $j$ 而变得任意小"。这是正确的吗？
3. 波斯特怀特的情形III假设 $b > c > d > e$、$b > d + e$、$f > g > j > h$ 且 $f < j + h$，他得出结论：当 $h$ 非常接近 $j$ 时，在精细实验下"代理人变得更好，而委托人变得更糟"。对 $(b, c, d, e) = (10, 6, 4, 3)$ 和 $(f, g, h, j) = (10, 8, 5.9, 6)$ 求解线性规划。然后证明，在情形III的不等式之下，规划者在第2部分那族计划中所偏好的计划，给予代理人的收益要少于他在粗糙实验下所得的收益。
4. 波斯特怀特的情形I假设 $e > d > b > c$ 且 $j > h > f > g$，他得出结论：更好的信息使两位参与者都受益。他的情形IV假设 $b > c > d > e$、$b < d + e$、$f > g > j > h$ 且 $f > j + h$，他得出结论：当 $h$ 接近 $j$ 时，这使规划者受益而使代理人受损。对于这四种情形中的每一种，从 $[0, 10]^8$ 中均匀地抽取大量满足该情形不等式约束的参数向量（在情形II到情形IV中，使 $h$ 落在 $j$ 的 $0.1$ 范围之内），并记录每位参与者受益或受损的频率。由于规划者可能有多个最优计划，请比较代理人在所有这些计划下的收益。波斯特怀特的四个结论中，哪些在每一次抽样中都成立？
```

```{solution-start} it_ex_postlewaite
:class: dropdown
```

下面给出一种解法。

我们构造收益矩阵以及两个实验。

在粗糙实验下，似然矩阵将状态 $\theta_1$ 和 $\theta_2$ 都映射到第一个信号，将状态 $\theta_3$ 和 $\theta_4$ 都映射到第二个信号，所以它同时也是该粗化方案。

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

下面的函数报告规划者的值，以及代理人预期效用在规划者所有最优计划上的范围，`compare` 为两种实验都打印出这些结果。

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

在粗糙实验下，"第一个信号之后采取 $a_1$，第二个信号之后采取 $a_2$"这一计划给规划者带来 $b = 10$，给代理人带来 $f = 10$，这是两者在该实验下所能获得的最大预期收益。

在精细实验下，规划者获得约 $9.71$，代理人获得约 $5.64$，所以两者都受损。

精细实验下的最优计划将约 $5/7 \approx 0.71$ 的概率赋予了被支配的行动 $a_3$ 和 $a_4$，波斯特怀特将其描述为"将大部分概率放在 $a_3$ 和 $a_4$ 上"。

对于第2部分，一个看到 $\theta_1$ 的代理人如实报告能得到 $2h(1 - x)$，而报告 $\theta_2$ 或 $\theta_4$ 能得到 $2gx$。

报告 $\theta_3$ 给出的彩票与如实报告相同。

所以他如实报告当且仅当 $h(1 - x) \geq g x$，即 $x \leq h/(h + g)$。

类似地，在状态 $\theta_4$ 中，如实报告要求 $x \leq h/(h + g)$，而在状态 $\theta_2$ 和 $\theta_3$ 中则要求 $j(1 - x) \geq f x$，即 $x \leq j/(j + f)$。

规划者的预期效用为 $(b + c) x + (d + e)(1 - x)$，当 $b + c > d + e$ 时随 $x$ 上升而增大，所以她将 $x$ 设为 $x^*$。

代理人的预期效用为 $(h + j)(1 - x)$。

```{code-cell} ipython3
def family_values(b, c, d, e, f, g, h, j):
    """x*, and both players' payoffs at x*, in the family of part 2."""
    x_star = min(h / (h + g), j / (j + f))
    planner = (b + c) * x_star + (d + e) * (1 - x_star)
    return x_star, planner, (h + j) * (1 - x_star)


x_star, v_family, a_family = family_values(**case_2)
print(f"x* = {x_star:.4f}, planner {v_family:.4f}, agent {a_family:.4f}")
```

这些结果与线性规划的解相符。

当一个看到 $\theta_1$ 的代理人撒谎时，他得到的是在 $a_2$ 和 $a_4$ 上的彩票，而这两种行动在该状态下分别使他获得 $2g$ 和 $0$，所以他的如实报告约束比较的是 $h$ 和 $g$，$j$ 并不出现在这一约束中。

所以边界 $x^*$ 取决于 $h$ 与 $g$ 的比较以及 $j$ 与 $f$ 的比较，而不取决于 $h$ 与 $j$ 的比较。

所以波斯特怀特的断言并不正确：当 $h$ 接近 $j$ 时，赋予 $a_1$ 和 $a_2$ 的权重仍然远离零，在这个例子中它是 $2/7$。

一个只知道哪一对状态发生了的代理人，在这样的计划下能得到 $\max\{xf + (1-x)h, \, xg + (1-x)j\}$。

在 $x = x^*$ 处，这等于 $(1 - x^*)(h + j)$，恰好等于知情代理人所得到的。

如果 $x^* = j/(j + f)$（正如这个例子中那样），第一项等于 $(1 - x^*)(h + j)$，第二项不会更大，因为 $x^* \leq h/(h + g)$；如果 $x^* = h/(h + g)$，两项的角色则相反。

所以，在规划者于这族计划中所偏好的计划之下，忽略更精细的信息会让代理人恰好处境相同，而不是更糟。

对于第3部分，我们求解情形III，并与第2部分的那族计划相比较。

```{code-cell} ipython3
case_3 = dict(b=10, c=6, d=4, e=3, f=10, g=8, h=5.9, j=6)
compare(case_3)
x_star, v_family, a_family = family_values(**case_3)
print(f"family of part 2: x* = {x_star:.4f}, planner {v_family:.4f}, "
      f"agent {a_family:.4f}")
```

在粗糙实验下，两位参与者都得到 $10$，而在精细实验下，规划者得到 $10.375$，代理人得到 $7.4375$：规划者受益而代理人受损，这与波斯特怀特的结论恰好相反。

事实上，每当情形III的不等式成立时，在规划者所偏好的第2部分那族计划下，代理人都会受损。

由于 $b > d + e$，规划者将 $x$ 设为 $x^*$。

在粗糙实验下，代理人得到 $f$，而在这个计划下他得到 $(h + j)(1 - x^*)$。

更小的 $x$ 的计划原本能给他更多，在 $x = 0$ 时最多能达到 $h + j > f$，但规划者并没有选择这些计划。

如果 $x^* = h/(h + g)$，他的收益为 $(h + j) g/(h + g)$，这小于 $f$，因为 $g(j - f) < 0 < h(f - g)$。

如果 $x^* = j/(j + f)$，他的收益为 $(h + j) f/(j + f)$，这小于 $f$，因为 $h < f$。

波斯特怀特的论证之所以站不住脚，是因为他将 $a_1$ 和 $a_2$ 上的权重视为可忽略不计，这相当于在 $x = 0$ 附近评估代理人的收益。

对于第4部分，我们为每种情形抽取参数。

对于情形I、III和IV，将均匀抽取的数值排序能得到该情形中这些值有序区域上的一个均匀抽样，这比拒绝无序抽样要快得多。

在情形II到情形IV中，我们在 $j$ 的 $0.1$ 范围内抽取 $h$。

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

情形I和情形IV在每一次抽样中都如波斯特怀特所说：在情形I中两位参与者都受益，在情形IV中规划者受益而代理人受损。

他对情形IV的论证重复了"$h$ 接近 $j$ 会使 $a_1$ 和 $a_2$ 上的权重可忽略不计"这一说法，而第2部分已表明这是错误的。

他关于规划者的结论并不需要这一说法：在情形IV的不等式下，第2部分那族计划中的每一个计划对她的价值都是 $(b + c)x + (d + e)(1 - x) > b$，因为 $b + c$ 和 $d + e$ 都超过 $b$。

族计划也给代理人带来 $(h + j)(1 - x) < f$，但规划者的最优计划有时会落在这个家族之外，所以代理人的损失取决于具体的抽样。

在情形II中，代理人在每一次抽样中都受损，但规划者在大约一半的抽样中受益，所以"两者都变得更糟"只在部分参数下成立。

下面是一个满足情形II不等式且 $h$ 接近 $j$ 的例子，在其中规划者受益。

```{code-cell} ipython3
case_2_gain = dict(b=10, c=9, d=5, e=4.8, f=10, g=4.5, h=3.9, j=4)
_ = compare(case_2_gain)
```

规划者从 $10$ 上升到约 $12.43$，而代理人从 $10$ 下降到约 $5.64$。

波斯特怀特关于 $h$ 接近 $j$ 的条件对情形II中代理人那一侧很重要：如果没有这个条件，代理人是可能受益的。

```{code-cell} ipython3
case_2_far = dict(b=9, c=3, d=2.5, e=4, f=8, g=7.8, h=1, j=6.95)
_ = compare(case_2_far)
```

这些参数满足情形II的不等式，且 $h$ 远离 $j$，代理人从 $8$ 上升到 $10.85$，而规划者从 $9$ 下降到 $8.5$，这正是波斯特怀特归属于情形III的模式。

在情形III中，代理人在每一次抽样中都受损，无论使用规划者的哪一个最优计划，而规划者在超过一半的抽样中受益。

第2部分仅为族计划中规划者所偏好的那个计划证明了代理人的损失；抽样结果表明这对线性规划的最优计划也同样成立，其中有一些落在该族计划之外。

这些比例取决于参数的抽取方式，所以它们描述的是每个区域上的均匀分布，而不是这些情形本身。

波斯特怀特更广泛的观点依然成立：更好的信息可能使两位参与者都变得更糟，或者使一方受益而另一方受损。

```{solution-end}
```

```{exercise}
:label: it_ex_reverse

{cite:t}`Postlewaite1982` 探讨了角色互换会发生什么：代理人先承诺一个应答规则，然后规划者根据她所听到的内容采取她认为最好的行动。

波斯特怀特沿用了格林的术语，格林的应答规则是确定性的；第1部分也允许代理人随机化。

1. 论证代理人仅推荐规划者愿意遵从的行动并无损失。将代理人的问题（他也可以随机化）写成关于一个 $N \times K$ 马尔可夫矩阵 $\Gamma = [\gamma_{nk}]$ 的线性规划，其中 $\gamma_{nk}$ 是代理人在观察到 $y_n$ 之后推荐 $a_k$ 的概率，约束条件为规划者的**服从约束**。假设一个处于无差异状态的规划者会遵从该推荐。
2. 对 {ref}`it_ex_riley` 中莱利例子的精细实验、格林的第一个例子，以及 {ref}`it_ex_postlewaite` 中情形II的例子，求解该线性规划。然后通过检验所有确定性规则（其中每一个 $\gamma_{nk}$ 都是 $0$ 或 $1$），找出代理人最好的确定性规则。将两位参与者在这两种情形下（规划者领先时和代理人领先时）的预期效用加以比较。
3. 波斯特怀特说，"构造出领导者-跟随者角色互换后委托人和代理人都变得更好的例子是容易的"。这些例子支持他的说法吗？
```

```{solution-start} it_ex_reverse
:class: dropdown
```

下面给出一种解法。

对于第1部分，任何发送信息的规则都会使规划者在每条信息之后采取某个行动。

将所有导致她采取同一行动 $a_k$ 的信息合并为单个信息"采取 $a_k$"。

她在合并后的信息之后的后验，是她在原始信息之后各个后验的一种平均，而她在每一个原始后验处都偏好 $a_k$，所以根据线性性，她在合并后的后验处仍然偏好 $a_k$。

合并后的规则会产生相同的信号与行动联合分布，所以两位参与者的预期效用都不变，一个确定性规则在合并后仍是确定性的。

如果对每一个其他行动 $a_{k'}$，都有

$$
\sum_{n=1}^N q_n \, \gamma_{nk} \left[(UP)_{kn} - (UP)_{k'n}\right] \;\geq\; 0 ,
$$

规划者就会遵从推荐 $a_k$，而代理人在这些约束下最大化 $\sum_n q_n \sum_k \gamma_{nk} (WP)_{kn}$。

这就是 {cite:t}`KamenicaGentzkow2011` 的说服问题，不同之处在于代理人只能粗化他自己的信号，而不能设计他喜欢的任何实验。

对于第2部分，`solve_agent_leads` 求解线性规划，`best_deterministic_rule` 检验每一个确定性规则；`obedience_rows` 和 `leader_payoffs` 是两者都要用到的辅助函数。

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

每一对数值中，第一个是规划者的预期效用，第二个是代理人的预期效用。

对于一个确定性规则，在莱利的例子和格林的第一个例子中，代理人的最佳选择都是如实报告。

承诺如实报告让规划者能够利用他所知道的一切信息来行动，这给了规划者她的第一优选值，并且给代理人带来的收益也超过了规划者领先时他所得到的收益。

在情形II中，他最好的确定性规则只报告哪一对状态发生了，这就相当于承诺采用粗糙实验，正如波斯特怀特所说他愿意做的那样；此时两位参与者都得到 $10$。

在这三个例子中，一个随机化规则能让代理人做得更好，但这是以牺牲规划者的利益为代价的。

在莱利的例子中，他以 $2/3$ 的概率在后验 $0.4$ 之后推荐 $a_1$，这个频率恰好足以使被告知采取 $a_1$ 的规划者（她随后将 $0.7$ 的概率赋予 $\theta_1$）愿意遵从。

在格林的第一个例子中，规划者下降到 $v_0 = 0.5$，在情形II中则下降到 $7$。

对于第3部分，对于他所描述的那种角色互换，波斯特怀特是正确的：在使用确定性应答规则的情况下，角色互换使这三个例子中的两位参与者都严格地变得更好。

在这些例子中，一个还可以随机化的代理人将规划者的收益据为己有，而规划者并不会变得更好，有时甚至比她领先时还要糟糕得多。

但这并非一般性的规律。

服从约束保证规划者至少能得到 $v_0$，因为她总可以不管收到何种推荐都采取同一个行动，所以当她自己的计划只能给她 $v_0$ 时，即使采用随机化规则，她让代理人领先也不会受损。

在莱利的例子中她只是打平，但在下面这个例子中，两位参与者都严格受益。

下面是一个具有两种状态、均匀先验和一个完全信息实验的例子。

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

当规划者领先时，她只能得到 $v_0 = 1.5$，代理人最多能得到 $1.5$；而当代理人用随机化规则领先时，两者都能得到 $1.75$。

```{solution-end}
```

```{exercise}
:label: it_ex_mm

{cite:t}`MarschakMiyasawa1968` 比较了关于后验概率的几个条件。

其中一个，我们称之为**凸包条件**，要求 $\Lambda'$ 的每一个后验都位于 $\Lambda$ 后验的凸包之内。

另一个条件是存在两个信号的联合分布 $J$（边际分布为 $q$ 和 $q'$）满足 {eq}`it_barycenter`；它等价于布莱克韦尔序。

取两种状态，先验为均匀分布。

实验 $\Lambda$ 的 $\theta_1$ 后验为 $0.9$、$0.5$、$0.1$，概率分别为 $0.05$、$0.9$、$0.05$。

实验 $\Lambda'$ 的后验为 $0.7$ 和 $0.3$，各自概率为 $1/2$。

1. 证明凸包条件对于 $\Lambda$ 相对于 $\Lambda'$ 成立，但 $\Lambda$ 的信息量并不比 $\Lambda'$ 更大。利用后验的方差解释这一现象。$\Lambda'$ 的信息量是否比 $\Lambda$ 更大？
2. 一个实验的**歧义度**是后验的期望熵，$-\sum_n q_n \sum_m p_{mn} \log p_{mn}$。分别计算两个实验的歧义度。马尔沙克和宫泽的定理12.3表明，更低的歧义度是信息量更大的必要但非充分条件。通过找出对 $\Lambda$ 和 $\Lambda'$ 做出相反排序的两个决策者来说明这一点。并与 {doc}`blackwell_kihlstrom` 中对熵的处理进行比较。
3. 马尔沙克和宫泽的定理10.1表明，当 $\Lambda$ 只有两种信号时，凸包条件是充分的。解释为什么，并用两种状态在数值上进行验证。这与 {prf:ref}`it_thm_binary` 的证明有何关系？
```

```{solution-start} it_ex_mm
:class: dropdown
```

下面给出一种解法。

对于第1部分，$\Lambda'$ 的后验 $0.7$ 和 $0.3$ 位于 $\Lambda$ 后验所张成的区间 $[0.1, 0.9]$ 之内，所以凸包条件成立。

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

$\Lambda$ 的信息量并不比 $\Lambda'$ 更大。

凸包条件只比较后验所处的位置，而忽略了它们各自发生的可能性。

在 $\Lambda$ 之下，后验通常是 $0.5$，所以其方差为 $0.016$，而在 $\Lambda'$ 之下方差为 $0.04$。

保均值展形会提高方差，所以 $\Lambda$ 下后验的分布不可能是 $\Lambda'$ 下后验分布的一种保均值展形。

真正排除这种可能性的，是 $J$ 必须具有边际分布 $q$ 这一要求。

要将后验 $0.7$ 表示为 $0.9$、$0.5$、$0.1$ 的一种平均，至少一半的权重必须放在 $0.9$ 上，所以 $J$ 就必须将至少 $1/4$ 的概率赋予后验 $0.9$，而 $\Lambda$ 产生这一后验的概率只有 $0.05$。

$\Lambda'$ 的信息量也不比 $\Lambda$ 更大：$\Lambda$ 的后验 $0.9$ 和 $0.1$ 位于 $\Lambda'$ 后验的凸包 $[0.3, 0.7]$ 之外。

凸包条件是布莱克韦尔序的必要条件，因为信息量较小的实验的每一个后验，都是信息量较大的实验的后验的一种平均，所以此处该条件不成立，`garbling` 也证实了这一点。

这两个实验是不可比较的。

对于第2部分，我们计算每个实验的歧义度。

```{code-cell} ipython3
def equivocation(Λ, π):
    """Expected entropy of the posterior."""
    q, P = posteriors(Λ, π)
    logs = np.log(P, where=P > 0, out=np.zeros_like(P))
    return -q @ (P * logs).sum(axis=0)


for name, Λ in (("Λ", Λ_mm), ("Λ'", Λ_mm_prime)):
    print(f"equivocation of {name}: {equivocation(Λ, π_mm):.4f}")
```

$\Lambda'$ 的歧义度更低，但它的信息量并不比 $\Lambda$ 更大。

考虑在两种赌局之间做出选择的决策者：一种赌局是在状态 $\theta_1$ 下支付 $1$，在状态 $\theta_2$ 下损失 $x$，另一种是不下注。

当后验超过 $x/(1 + x)$ 时，值得下这个赌注。

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

一个只在后验超过 $0.8$ 时才下注的谨慎押注者，更看重 $\Lambda$（它有时会给出后验 $0.9$），而从 $\Lambda'$ 中一无所获。

一个临界值为 $0.6$ 的更大胆的押注者则偏好 $\Lambda'$。

这正是马尔沙克和宫泽所指出的一点：歧义度（不同于信息量）能够为所有实验排序，所以它无法体现信息价值对使用者收益结构的依赖性。

{doc}`blackwell_kihlstrom` 用德格鲁特的术语表达了同样的观点：布莱克韦尔序要求不确定性的减少对于每一个凹的不确定性函数都更大，而不仅仅是对熵而言。

对于第3部分，假设 $\Lambda$ 具有两种后验 $p_1$ 和 $p_2$ 不同的信号，且 $\Lambda'$ 的每一个后验都位于连接二者的线段上。

将 $\Lambda'$ 的每一个后验 $p'_{n'}$ 写成 $p_1$ 和 $p_2$ 的一种混合，并根据混合权重来拆分信号 $y'_{n'}$。

这就给出了一个列和为 $q'$ 并满足 {eq}`it_barycenter` 的联合分布 $J$。

它的行和将概率赋予 $p_1$ 和 $p_2$，二者的平均为 $\sum_{n'} q'_{n'} p'_{n'} = \pi$。

在两个不同的点上，只有一个分布的均值为 $\pi$，所以这些行和就是 $q$，因此 $\Lambda$ 的信息量比 $\Lambda'$ 更大。

要求 $J$ 的行和等于 $q$——这一条件在第1部分中未能成立——对于任意数量的状态都是自动成立的。

我们通过随机抽取 $\Lambda'$ 以及一个后验能够包夹住 $\Lambda'$ 后验的两信号 $\Lambda$ 来检验这一点。

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

{prf:ref}`it_thm_binary` 的证明通过两个事实用到了信息序：$\Lambda'$ 的后验位于连接 $\Lambda$ 后验的线段上，以及存在一个满足 {eq}`it_barycenter` 的联合分布 $J$。

当 $\Lambda$ 具有两种信号时，上面的论证表明第一个事实蕴含了第二个事实，所以此时布莱克韦尔序就成了一个关于后验所处位置的条件，而非关于它们发生可能性的条件。

```{solution-end}
```