"""
run_v2.py - improved experiment (version 2).

Compares "scratch" learning with "transfer" learning (pretrained on Layout A)
on six target layouts. Results are saved in the folder results_v2/ so the
original results (results.txt, results_transfer.png) are never overwritten.

Run with:  python run_v2.py
"""
import os
import csv
import json
import math
import random
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")          # draw graphs without opening a window
import matplotlib.pyplot as plt
from collections import deque

from layouts import make_layout_a, TARGETS
from q_agent import choose_action, update_q_table

# ------------------------- settings -------------------------
NUM_RUNS = 20            # independent runs (seeds 0..19) per condition
SOURCE_EPISODES = 500    # pretraining episodes on Layout A (transfer only)
TARGET_EPISODES = 300    # training episodes on the target layout
MAX_STEPS = 100          # step limit per episode
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
EVAL_EPISODES = 20       # greedy evaluation episodes per trained agent
WINDOW = 50              # "early" = first 50 episodes, "final" = last 50
PERM_SAMPLES = 10000     # shuffles used in the permutation test
OUT_DIR = "results_v2"
# ------------------------------------------------------------


def make_rng(*parts):
    # A separate, reproducible random generator for each purpose.
    # The same text always gives the same random numbers, and different
    # text gives an independent stream.
    return random.Random("-".join(str(p) for p in parts))


def train(env, q_table, num_episodes, rng):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step_number in range(MAX_STEPS):
            action = choose_action(q_table, state, EPSILON, rng)
            next_state, reward, done = env.step(action)
            update_q_table(q_table, state, action, reward, next_state,
                           done, ALPHA, GAMMA)
            state = next_state
            total_reward = total_reward + reward
            if done:
                break
        rewards.append(total_reward)
    return rewards


def greedy_evaluation(env, q_table, rng):
    # Run EVAL_EPISODES greedy episodes (epsilon = 0, no learning).
    # The environment is deterministic, so episodes can differ only through
    # random tie-breaking between equal Q-values.
    path_lengths = []   # steps used in successful episodes
    failures = 0
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        reached = False
        for step_number in range(MAX_STEPS):
            action = choose_action(q_table, state, 0, rng)
            state, reward, done = env.step(action)
            if done:
                path_lengths.append(step_number + 1)
                reached = True
                break
        if not reached:
            failures += 1
    return path_lengths, failures


def shortest_path_length(env):
    # Breadth-first search: the true shortest number of steps start -> goal
    queue = deque([(env.start, 0)])
    seen = {env.start}
    while queue:
        (row, col), dist = queue.popleft()
        if (row, col) == env.goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < env.size and 0 <= nc < env.size \
                    and (nr, nc) not in env.walls and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def permutation_test(a, b, seed=0):
    # Two-sided permutation test for a difference in means.
    # Idea: if scratch and transfer made no difference, shuffling the labels
    # would give differences as big as the real one fairly often.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    gen = np.random.default_rng(seed)
    count = 0
    for i in range(PERM_SAMPLES):
        gen.shuffle(pooled)
        diff = abs(pooled[:len(a)].mean() - pooled[len(a):].mean())
        if diff >= observed - 1e-12:
            count += 1
    return (count + 1) / (PERM_SAMPLES + 1)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(values.mean()), float(sd), int(len(values))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Step 1: pretrain one source Q-table per run on Layout A ----
    # The same pretrained table (run i) is copied for every target layout.
    source_tables = []
    for run in range(NUM_RUNS):
        q = np.zeros((5, 5, 4))
        train(make_layout_a(), q, SOURCE_EPISODES, make_rng("pretrain", run))
        source_tables.append(q)

    results = {}          # results[target][condition] = dict of arrays
    per_run_rows = []

    for name, change, maker in TARGETS:
        optimal = shortest_path_length(maker())
        results[name] = {"change": change, "optimal_steps": optimal}

        for condition in ["scratch", "transfer"]:
            curves, early, final = [], [], []
            success_rates, mean_paths, failures = [], [], []

            for run in range(NUM_RUNS):
                env = maker()
                if condition == "scratch":
                    q = np.zeros((5, 5, 4))
                else:
                    q = source_tables[run].copy()

                # separate random stream for every (condition, target, run)
                rewards = train(env, q, TARGET_EPISODES,
                                make_rng(condition, "train", name, run))
                paths, fails = greedy_evaluation(
                    env, q, make_rng(condition, "eval", name, run))

                curves.append(rewards)
                early.append(np.mean(rewards[:WINDOW]))
                final.append(np.mean(rewards[-WINDOW:]))
                success_rates.append(len(paths) / EVAL_EPISODES)
                mean_paths.append(np.mean(paths) if paths else float("nan"))
                failures.append(fails)

                per_run_rows.append([name, change, condition, run,
                                     round(early[-1], 4), round(final[-1], 4),
                                     success_rates[-1],
                                     "" if not paths else round(mean_paths[-1], 4),
                                     fails])

            results[name][condition] = {
                "curves": np.array(curves),
                "early": early, "final": final,
                "success_rate": success_rates,
                "mean_path": mean_paths,
                "failures": failures,
            }

        # statistical tests (exploratory)
        for metric in ["early", "final"]:
            p = permutation_test(results[name]["scratch"][metric],
                                 results[name]["transfer"][metric])
            results[name]["p_" + metric] = p

    # ---- Step 2: summary table (mean +- SD across the 20 runs) ----
    summary = []
    n_tests = 2 * len(TARGETS)    # Bonferroni: 2 metrics x 6 targets
    for name, change, maker in TARGETS:
        r = results[name]
        for condition in ["scratch", "transfer"]:
            c = r[condition]
            e_m, e_s, _ = mean_sd(c["early"])
            f_m, f_s, _ = mean_sd(c["final"])
            sr_m, sr_s, _ = mean_sd(c["success_rate"])
            p_m, p_s, p_n = mean_sd(c["mean_path"])
            summary.append({
                "target": name, "change": change, "condition": condition,
                "optimal_steps": r["optimal_steps"],
                "early_reward_mean": e_m, "early_reward_sd": e_s,
                "final_reward_mean": f_m, "final_reward_sd": f_s,
                "success_rate_mean": sr_m, "success_rate_sd": sr_s,
                "successful_path_mean": p_m, "successful_path_sd": p_s,
                "agents_with_a_success": p_n,
                "failed_eval_episodes": int(sum(c["failures"])),
                "total_eval_episodes": NUM_RUNS * EVAL_EPISODES,
                "p_early": r["p_early"],
                "p_early_bonferroni": min(1.0, r["p_early"] * n_tests),
                "p_final": r["p_final"],
                "p_final_bonferroni": min(1.0, r["p_final"] * n_tests),
            })

    # ---- Step 3: save everything ----
    with open(os.path.join(OUT_DIR, "per_run_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "change", "condition", "run", "early_reward",
                    "final_reward", "greedy_success_rate",
                    "mean_successful_path", "greedy_failures"])
        w.writerows(per_run_rows)

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    settings = {
        "num_runs": NUM_RUNS, "source_episodes": SOURCE_EPISODES,
        "target_episodes": TARGET_EPISODES, "max_steps": MAX_STEPS,
        "alpha": ALPHA, "gamma": GAMMA, "epsilon": EPSILON,
        "eval_episodes": EVAL_EPISODES, "window": WINDOW,
        "permutation_samples": PERM_SAMPLES, "bonferroni_tests": n_tests,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "date": datetime.date.today().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "results_v2.json"), "w") as f:
        json.dump({"settings": settings, "summary": summary}, f, indent=2)

    # ---- Step 4: learning-curve figure with +-1 SD bands ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    for ax, (name, change, maker) in zip(axes.flat, TARGETS):
        for condition, color in [("scratch", "tab:blue"), ("transfer", "tab:orange")]:
            curves = results[name][condition]["curves"]
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1)
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, label=condition.capitalize())
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.set_title("A -> %s (%s)" % (name, change))
        ax.set_xlabel("Episode on target layout")
        ax.set_ylabel("Reward (mean +- SD, %d runs)" % NUM_RUNS)
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curves_v2.png"), dpi=150)

    # ---- Step 5: readable text summary ----
    lines = ["Results v2  (mean +- SD across %d runs)" % NUM_RUNS, ""]
    for name, change, maker in TARGETS:
        r = results[name]
        lines.append("A -> %s (%s), shortest path = %d steps"
                     % (name, change, r["optimal_steps"]))
        for s in summary:
            if s["target"] != name:
                continue
            lines.append(
                "  %-8s early %.2f +- %.2f | final %.2f +- %.2f | "
                "success %.3f +- %.3f | path %.2f +- %.2f (n=%d) | failed %d/%d"
                % (s["condition"], s["early_reward_mean"], s["early_reward_sd"],
                   s["final_reward_mean"], s["final_reward_sd"],
                   s["success_rate_mean"], s["success_rate_sd"],
                   s["successful_path_mean"], s["successful_path_sd"],
                   s["agents_with_a_success"],
                   s["failed_eval_episodes"], s["total_eval_episodes"]))
        lines.append("  permutation p (early): %.4f   (Bonferroni x%d: %.4f)"
                     % (r["p_early"], n_tests, min(1.0, r["p_early"] * n_tests)))
        lines.append("  permutation p (final): %.4f   (Bonferroni x%d: %.4f)"
                     % (r["p_final"], n_tests, min(1.0, r["p_final"] * n_tests)))
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "results_v2.txt"), "w") as f:
        f.write(text)
    print(text)
    print("Saved all files in the folder:", OUT_DIR)


if __name__ == "__main__":
    main()