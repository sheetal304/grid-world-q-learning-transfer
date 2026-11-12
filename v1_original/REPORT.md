# Transfer of a Q-Table in a Small Grid-World: A Tabular Q-Learning Prototype

## Abstract
This report describes a small independent prototype inspired by the research theme of Project 6 of the CaCTüS Internship
(schema learning in meta-reinforcement learning for spatial navigation). It is not a reproduction of that project.
I studied whether a tabular Q-learning agent that has learned one 5x5 grid layout learns faster in a modified layout than
an agent that starts from scratch. Across 20 runs per condition, reusing the Q-table sped up early learning when only the walls
changed, and slowed it down when the goal moved.

## 1. Background and scope
The original project investigates whether recurrent meta-RL agents can discover schema-learning mechanisms for fast
adaptation in new environments. My prototype implements only a much simpler idea: transfer of learned values between two
similar navigation tasks using tabular Q-learning. It has no neural network, no meta-learning, no paired-association task
and no place-cell representation. Its purpose was to build a reproducible experimental pipeline and a baseline for later work.

## 2. Research question
Does initialising a Q-learning agent with a Q-table learned on a source layout (Layout A) speed up or slow down learning
on a modified target layout, compared with learning from scratch?

## 3. Methods
**Environment.** 5x5 grid, start (0,0), four actions. Reward +10 at the goal, -1 per step; walls and borders block movement.
Episodes end at the goal or after 100 steps.

**Layouts.** A (source): walls (1,1),(2,2),(3,3), goal (4,4). B: walls (0,2),(1,2),(3,1),(3,2),(3,3), goal (4,4)
(walls change). C: walls as in A, goal (0,4) (goal changes). Shortest paths: 8 steps in B, 4 steps in C.

**Agent.** Tabular Q-learning, alpha = 0.1, gamma = 0.9, epsilon = 0.1, no bootstrapping from terminal states, random
tie-breaking between equal Q-values.

**Conditions.** Scratch: 300 episodes on the target layout from a zero Q-table. Transfer: 500 episodes on Layout A, then
300 episodes on the target layout with the same Q-table.

**Evaluation.** 20 runs per condition with fixed seeds 0-19. Metrics: mean reward over the first 50 and last 50 target
episodes (a measure of learning speed and final performance), and the number of steps of one greedy (epsilon = 0) rollout
after training, plus the count of runs that never reached the goal.

## 4. Results
| | A -> B (walls change) | A -> C (goal changes) |
|---|---|---|
| Mean reward, first 50 episodes (scratch / transfer) | -14.62 / -0.21 | -3.31 / -15.84 |
| Mean reward, last 50 episodes (scratch / transfer) | 2.10 / 2.00 | 6.45 / 4.99 |
| Mean greedy steps to goal (scratch / transfer) | 8.00 / 8.00 | 4.00 / 5.20 |
| Greedy runs that never reached the goal | 0 / 0 | 0 / 0 |

**A -> B.** Transfer improved early learning substantially (-0.21 vs. -14.62), and both conditions ended with similar
performance and optimal 8-step greedy paths.

**A -> C.** Transfer slowed early learning (-15.84 vs. -3.31), a case of negative transfer. After 300 episodes the
transfer agents still earned less reward (4.99 vs. 6.45) and used longer greedy paths on average (5.20 vs. 4.00 steps,
where 4 is optimal). The example transfer path from the last run, (0,0),(0,1),(0,2),(0,3),(1,3),(1,4),(0,4), contains a detour.

## 5. Discussion
A plausible explanation, which I did not test, is that when the goal stays fixed, most of the source Q-values remain
valid, while moving the goal makes many source values point toward the old goal and requires them to be overwritten.
The persistent gap in A -> C suggests that, with epsilon = 0.1 and 300 episodes, some transferred agents settle on a
suboptimal path. Verifying this would require, for example, inspecting the learned Q-values or training longer.

## 6. Limitations
- Very small, deterministic environment; one source layout and two target layouts.
- No statistical tests or confidence intervals; results are means over 20 seeds.
- A single greedy rollout per run is used to measure path length.
- Hyperparameters and episode counts were not tuned or varied.
- Tabular Q-learning cannot form reusable abstractions, so these results do not speak to schema learning.

## 7. Future work
More target layouts and goal positions with statistical comparison; sensitivity to epsilon and training length; a recurrent
(LSTM) agent trained across many layouts; place-cell-like state representations with shuffled remapping between
environments, which is closer to the original research question.