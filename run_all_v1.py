import random
import numpy as np
import matplotlib.pyplot as plt
from layouts import make_layout_a, make_layout_b, make_layout_c
from q_agent import choose_action, update_q_table

NUM_RUNS = 20


def train(env, q_table, num_episodes=300, max_steps=100,
          alpha=0.1, gamma=0.9, epsilon=0.1):
    # Train the agent and return the reward of every episode
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step_number in range(max_steps):
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            update_q_table(
    q_table, state, action, reward, next_state, done, alpha, gamma
)
            state = next_state
            total_reward = total_reward + reward
            if done:
                break
        rewards.append(total_reward)
    return rewards


def greedy_path(env, q_table, max_steps=100):
    # Follow the learned Q-table with no random moves (epsilon = 0)
    state = env.reset()
    path = [state]
    for step_number in range(max_steps):
        action = choose_action(q_table, state, 0)
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            return path, True
    return path, False


def run_experiment(make_target):
    scratch_rewards, transfer_rewards = [], []
    scratch_steps, transfer_steps = [], []
    scratch_fail, transfer_fail = 0, 0
    last_path = None

    for run in range(NUM_RUNS):
        random.seed(run)
        np.random.seed(run)

        # Condition 1: scratch (empty Q-table, train directly on target)
        env_t = make_target()
        q_scratch = np.zeros((5, 5, 4))
        scratch_rewards.append(train(env_t, q_scratch))
        path, ok = greedy_path(env_t, q_scratch)
        scratch_steps.append(len(path) - 1)
        if not ok:
            scratch_fail += 1

        # Condition 2: transfer (train on Layout A first, then on target)
        env_a = make_layout_a()
        q_transfer = np.zeros((5, 5, 4))
        train(env_a, q_transfer, num_episodes=500)
        env_t = make_target()
        transfer_rewards.append(train(env_t, q_transfer))
        path, ok = greedy_path(env_t, q_transfer)
        transfer_steps.append(len(path) - 1)
        if not ok:
            transfer_fail += 1
        last_path = path

    return (np.array(scratch_rewards), np.array(transfer_rewards),
            scratch_steps, transfer_steps, scratch_fail, transfer_fail, last_path)


lines = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

experiments = [("A -> B (walls change)", make_layout_b),
               ("A -> C (goal changes)", make_layout_c)]

for ax, (name, maker) in zip(axes, experiments):
    s_r, t_r, s_st, t_st, s_f, t_f, path = run_experiment(maker)

    lines.append("Experiment " + name)
    lines.append("  Runs: " + str(NUM_RUNS) + ", training episodes on target: 300")
    lines.append("  First 50 episodes, mean reward: scratch %.2f, transfer %.2f"
                 % (s_r[:, :50].mean(), t_r[:, :50].mean()))
    lines.append("  Last 50 episodes, mean reward: scratch %.2f, transfer %.2f"
                 % (s_r[:, -50:].mean(), t_r[:, -50:].mean()))
    lines.append("  Greedy steps to goal (mean): scratch %.2f, transfer %.2f"
                 % (np.mean(s_st), np.mean(t_st)))
    lines.append("  Greedy runs that never reached the goal: scratch %d, transfer %d"
                 % (s_f, t_f))
    lines.append("  Example transfer path (last run): " + str(path))
    lines.append("")

    ax.plot(s_r.mean(axis=0), label="Scratch")
    ax.plot(t_r.mean(axis=0), label="Transfer (from A)")
    ax.set_title(name)
    ax.set_xlabel("Episode on target layout")
    ax.set_ylabel("Mean reward over " + str(NUM_RUNS) + " runs")
    ax.legend()

plt.tight_layout()
plt.savefig("results_transfer.png")

with open("results.txt", "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("Saved: results_transfer.png and results.txt")