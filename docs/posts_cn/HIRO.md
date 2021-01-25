## 前言

一般的强化学习算法主要从最基本的层次(atomic scale)来推导动作 — 从环境状态 $s_t$ 来得出动作 $a_t$, 然后执行得到下一个环境状态 $s_{t+1}$, 然后循环. 这种最基本层次的推演让他们很难扩展到复杂的任务中. 对于复杂的任务, 更佳的策略往往是把大任务简化成一个个小的任务, 然后在依次完成各个小的任务. 分层强化学习(Hierarchical Reinforcement Learning, HRL)正是模拟这种思路. 

在本文中, 我们讨论一个由Google Brain的Ofir Nachum等人在NIPS 2018上提出的分层学习算法. 这个算法**HI**erarchical **R**einforcement learning with **O**ff-policy correction(HIRO)主要针对目标导向(goal-directed)的任务. 在这类任务中, 我们的 agent 目标是去接近某个目标状态.

## HIRO

我们先介绍关于分层强化学习的三个问题

1. 如何训练低层策略(low-level policy)来得到语义上不同的行为(semantically distinct behavior)?
2. 如何定义高层策略(high-level policy)的动作?
3. 如何在尽可能高效(sample-efficient)的情况下里训练多层策略?

我们可以基于上述三个问题来很好地解释HIRO (这里我们假设读者知道基本的off-policy算法, 诸如DDPG, TD3等):

1. 除乐环境状态, 我们可以把subgoal(子目标, 由高一层的策略生成)作为输入传给policy net(策略网络). 这样我们就能对于每个subgoal的来生成不同的策略. 为了更好的引导低层策略, 我们定义基于subgoal的reward function(奖励函数)
   $$
   r(s_t,g_t,a_t,s_{t+1})=-\Vert s_t+g_t-s_{t+1}\Vert_2\tag {1}
   $$

2. 高层策略的动作被定义为下层策略的subgoal. 在HIRO中, 低层策略接收到的subgoal是会随时间变化的. 每隔 $c$ 步骤, 高层策略生成一个 subgoal 传递给低层策略. 在 $t\equiv0\ (\mathrm{mod}\ c)$ 时, 这个subgoal会直接被用来作为低层策略的输入, 在其他时间里, 我们将要通过一个转换函数(goal transition function)来重新计算subgoal. 这个函数定义为
   $$
   \begin{align}
   g_t&=\begin{cases}\mu^{high}(s_t)&t\equiv 0\mod c\\
   h(s_{t-1}, g_{t-1}, s_t)&\mathrm{otherwise}\end{cases}\tag {2}\\
   where\ h(s_{t},g_{t},s_{t+1})&=s_t+g_t-s_{t+1}
   \end{align}
   $$

3. 为提高sample efficiency. 在每一层, 我们都使用off-policy算法(例如, TD3). 特别的, 对于一个两层的HRL agent, 我们用 transition tuples, $(s_t, g_t, a_t, r_t, s_{t+1}, g_{t+1})$, 来训练低层策略(其中 $r$ 和 $g$ 是根据等式1和等式2计算得来). 对于高层策略, 我们采用 transition tuples $(s_t, \tilde g_t, \sum R_{t:t+c-1}, s_{t+c})$, 其中 $\tilde g$ 是重新标记的 subgoal, $R$ 是由环境提供的reward.