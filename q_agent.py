import random
import numpy as np


def choose_action(q_table, state, epsilon, rng=random):
    # rng is a random number generator. By default it is Python's global
    # "random" module (this keeps the old run_all_v1.py giving the same results).
    # run_v2.py passes its own generator: random.Random(seed).
    row, col = state

    if rng.random() < epsilon:
        return rng.randint(0, 3)

    values = q_table[row, col]
    best_actions = np.flatnonzero(values == np.max(values))

    return int(rng.choice(best_actions))


def update_q_table(
    q_table, state, action, reward, next_state, done, alpha, gamma
):
    row, col = state
    next_row, next_col = next_state

    old_value = q_table[row, col, action]

    if done:
        target = reward
    else:
        best_next = np.max(q_table[next_row, next_col])
        target = reward + gamma * best_next

    q_table[row, col, action] = (
        old_value + alpha * (target - old_value)
    )