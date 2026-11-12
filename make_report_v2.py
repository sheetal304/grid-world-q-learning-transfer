"""
make_report_v2.py - builds REPORT_v2.md from results_v2/results_v2.json.

All numbers in the report are copied from the results file, so the report
can only contain results that run_v2.py actually produced.

Run with:  python make_report_v2.py     (after python run_v2.py)
"""
import json

with open("results_v2/results_v2.json") as f:
    data = json.load(f)

S = data["settings"]
rows = data["summary"]
targets = []
for r in rows:
    if r["target"] not in [t[0] for t in targets]:
        targets.append((r["target"], r["change"]))


def get(target, condition):
    for r in rows:
        if r["target"] == target and r["condition"] == condition:
            return r


def ms(m, s, d=2):
    return ("%." + str(d) + "f +- %." + str(d) + "f") % (m, s)


def pfmt(p):
    return "<0.001" if p < 0.001 else "%.3f" % p


L = []
add = L.append

add("# Transfer of a Q-Table Across Grid-World Layouts: A Tabular Q-Learning Prototype (Version 2)")
add("")
add("## Abstract")
add("This report describes a small independent prototype inspired by the research theme of Project 6 of the CaCTüS "
    "Internship (schema learning in meta-reinforcement learning for spatial navigation). It is **not** a reproduction of that "
    "project. I tested whether a tabular Q-learning agent pretrained on one 5x5 grid layout (Layout A) learns faster or slower "
    "in modified layouts than an agent trained from scratch. Six target layouts were used: three where only the walls change and "
    "three where only the goal moves. Each condition was run %d times with independent random streams, and every trained agent "
    "was evaluated over %d greedy episodes. All numbers below were produced by `run_v2.py`." % (S["num_runs"], S["eval_episodes"]))
add("")
add("## 1. Background and scope")
add("The original project asks whether recurrent meta-RL agents can discover schema-learning mechanisms for fast adaptation in new "
    "environments. This prototype implements only a much simpler idea: reuse of learned values (a Q-table) between similar navigation "
    "tasks. It has no neural network, no meta-learning, no paired-association task and no place-cell representation, so it provides "
    "no evidence about schema learning itself. Its purpose is a reproducible experimental pipeline and a baseline for future work.")
add("")
add("## 2. Research question")
add("Does initialising a Q-learning agent with a Q-table learned on Layout A speed up or slow down learning on a modified target "
    "layout, compared with learning from scratch, and does the answer depend on whether the walls or the goal change?")
add("")
add("## 3. Methods")
add("**Environment.** 5x5 grid, start (0,0), actions up/down/left/right. Reward +10 at the goal and -1 for every other step. Walls and "
    "borders block movement. An episode ends at the goal or after %d steps." % S["max_steps"])
add("")
add("**Layouts.** Source A: walls (1,1),(2,2),(3,3), goal (4,4). Targets where only the walls change (goal (4,4)): "
    "B walls (0,2),(1,2),(3,1),(3,2),(3,3); D walls (1,0),(1,1),(1,2),(3,2),(3,3),(3,4); E walls (2,0),(2,1),(2,3),(2,4). "
    "Targets where only the goal changes (walls as in A): C goal (0,4); F goal (4,0); G goal (2,4). "
    "Shortest path lengths (found by breadth-first search): " +
    ", ".join("%s = %d" % (t, get(t, "scratch")["optimal_steps"]) for t, c in targets) + " steps.")
add("")
add("**Agent.** Tabular Q-learning with alpha = %s, gamma = %s, epsilon = %s. No bootstrapping from terminal states; ties between equal "
    "Q-values are broken randomly. Hyperparameters were not tuned." % (S["alpha"], S["gamma"], S["epsilon"]))
add("")
add("**Conditions.** *Scratch*: %d episodes on the target layout from a zero Q-table. *Transfer*: %d episodes on Layout A, then %d "
    "episodes on the target layout with the same Q-table." % (S["target_episodes"], S["source_episodes"], S["target_episodes"]))
add("")
add("**Random seeds.** Every random purpose uses its own generator (`random.Random`) created from a text seed such as "
    "`scratch-train-C-7`. Therefore the scratch and transfer conditions use **independent** random streams (they are not matched "
    "run by run), and re-running the script gives identical results. For transfer, pretraining on Layout A uses the stream "
    "`pretrain-<run>`; the same pretrained Q-table of run *i* is copied to every target layout (so the transfer runs of different "
    "targets share their source table, but the comparison scratch vs transfer within a target uses independent samples). "
    "The earlier version 1 (see Section 7) drew scratch and transfer from one shared stream.")
add("")
add("**Evaluation.** %d runs per condition and target. Metrics per run: mean reward over the first %d target episodes (early learning) "
    "and over the last %d (final performance, includes epsilon exploration). After training, each agent is evaluated for %d greedy "
    "(epsilon = 0, no learning) episodes. The environment is deterministic, so these episodes differ only through random "
    "tie-breaking between equal Q-values. From them I report the success rate, the mean path length of successful episodes, and the "
    "number of failed episodes (goal not reached within %d steps). Mean and standard deviation (SD, sample SD with n-1) are taken "
    "across the %d runs." % (S["num_runs"], S["window"], S["window"], S["eval_episodes"], S["max_steps"], S["num_runs"]))
add("")
add("**Statistical testing.** For early and final reward I ran a two-sided permutation test (%d shuffles) comparing scratch and transfer "
    "per target. Because %d tests were run, I also report Bonferroni-adjusted p-values (p x %d, capped at 1). These tests are exploratory: "
    "they use only %d runs per group, assume runs are exchangeable, and the choice of metrics and layouts was made by me. "
    "I do not make claims of statistical significance beyond reporting these values." %
    (S["permutation_samples"], S["bonferroni_tests"], S["bonferroni_tests"], S["num_runs"]))
add("")

add("## 4. Results")
add("### 4.1 Learning speed and final reward (mean +- SD over %d runs)" % S["num_runs"])
add("")
add("| Target | Change | Early reward scratch | Early reward transfer | adj. p (early) | Final reward scratch | Final reward transfer | adj. p (final) |")
add("|---|---|---|---|---|---|---|---|")
for t, c in targets:
    a, b = get(t, "scratch"), get(t, "transfer")
    add("| A -> %s | %s | %s | %s | %s | %s | %s | %s |" % (
        t, c, ms(a["early_reward_mean"], a["early_reward_sd"]), ms(b["early_reward_mean"], b["early_reward_sd"]),
        pfmt(b["p_early_bonferroni"]), ms(a["final_reward_mean"], a["final_reward_sd"]),
        ms(b["final_reward_mean"], b["final_reward_sd"]), pfmt(b["p_final_bonferroni"])))
add("")
add("### 4.2 Greedy evaluation (%d episodes per agent)" % S["eval_episodes"])
add("")
add("| Target | Shortest path | Condition | Success rate (mean +- SD) | Mean successful path (mean +- SD over agents) | Agents counted | Failed episodes |")
add("|---|---|---|---|---|---|---|")
for t, c in targets:
    for cond in ["scratch", "transfer"]:
        r = get(t, cond)
        add("| A -> %s | %d | %s | %s | %s | %d/%d | %d/%d |" % (
            t, r["optimal_steps"], cond, ms(r["success_rate_mean"], r["success_rate_sd"], 3),
            ms(r["successful_path_mean"], r["successful_path_sd"]), r["agents_with_a_success"], S["num_runs"],
            r["failed_eval_episodes"], r["total_eval_episodes"]))
add("")
add("'Agents counted' is the number of agents with at least one successful greedy episode (only those enter the path statistic).")
add("")
add("![Learning curves](results_v2/learning_curves_v2.png)")
add("")
add("### 4.3 Summary of findings (generated from the table above)")
add("")
groups = {}
for t, c in targets:
    groups.setdefault(c, []).append(t)

for t, c in targets:
    a, b = get(t, "scratch"), get(t, "transfer")
    d_early = b["early_reward_mean"] - a["early_reward_mean"]
    d_final = b["final_reward_mean"] - a["final_reward_mean"]
    d_path = b["successful_path_mean"] - a["successful_path_mean"]
    add("- **A -> %s (%s):** early reward %s with transfer (difference %+.2f, adjusted p = %s); final reward %s (difference %+.2f, adjusted p = %s); "
        "mean successful greedy path %.2f (transfer) vs %.2f (scratch), optimal %d." % (
            t, c, "higher" if d_early > 0 else "lower", d_early, pfmt(b["p_early_bonferroni"]),
            "higher" if d_final > 0 else "lower", d_final, pfmt(b["p_final_bonferroni"]),
            b["successful_path_mean"], a["successful_path_mean"], a["optimal_steps"]))
add("")
for c, ts in groups.items():
    n = len(ts)
    early_up = sum(1 for t in ts if get(t, "transfer")["early_reward_mean"] > get(t, "scratch")["early_reward_mean"])
    final_down = sum(1 for t in ts if get(t, "transfer")["final_reward_mean"] < get(t, "scratch")["final_reward_mean"])
    longer = sum(1 for t in ts if get(t, "transfer")["successful_path_mean"] > get(t, "scratch")["successful_path_mean"] + 1e-9)
    add("- Across the %d layouts where the **%s**: transfer had higher early reward in %d of %d, lower final reward in %d of %d, "
        "and a longer mean greedy path in %d of %d." % (n, c, early_up, n, final_down, n, longer, n))
total_fail = sum(r["failed_eval_episodes"] for r in rows)
total_eval = sum(r["total_eval_episodes"] for r in rows)
add("- Failed greedy episodes over all agents and layouts: %d of %d." % (total_fail, total_eval))
add("")
add("## 5. Discussion")
walls_ts = groups.get("walls change", [])
goal_ts = groups.get("goal changes", [])
w_all = all(get(t, "transfer")["early_reward_mean"] > get(t, "scratch")["early_reward_mean"] for t in walls_ts)
g_all = all(get(t, "transfer")["early_reward_mean"] < get(t, "scratch")["early_reward_mean"] for t in goal_ts)
if w_all and g_all:
    add("The early-reward pattern of the first experiments (transfer helps when walls change, hurts when the goal moves) held in all six layouts tested.")
else:
    add("The early-reward pattern of the first two experiments (transfer helps when walls change, hurts when the goal moves) did **not** hold in "
        "every layout; see the per-layout lines in Section 4.3. The simple rule 'wall change helps, goal change hurts' is therefore not "
        "a general conclusion from these data.")
add("")
add("A plausible but **untested** explanation is that when the goal stays fixed most source Q-values remain useful, while moving the goal makes "
    "many source values point toward the old goal, which must then be overwritten. How harmful this is probably depends on how far the new goal "
    "is from the old one and on the wall layout. Testing this would need, for example, inspecting the learned Q-values or varying the goal "
    "position systematically.")
add("")
add("## 6. Limitations")
add("- Tiny deterministic 5x5 environment; one source layout and six hand-chosen target layouts.")
add("- %d runs per condition; permutation tests are exploratory and p-values depend on the chosen metrics and layouts." % S["num_runs"])
add("- Greedy evaluation episodes differ only through tie-breaking, so they are not independent samples of a stochastic environment.")
add("- Source pretraining length (%d episodes) and target training length (%d) were fixed and not varied; hyperparameters were not tuned." %
    (S["source_episodes"], S["target_episodes"]))
add("- Final-episode reward includes epsilon exploration, so it is below the optimal reward.")
add("- Tabular Q-learning cannot form reusable abstractions; these results say nothing about schema learning.")
add("")
add("## 7. Earlier version (v1, preserved)")
add("Version 1 (`run_all_v1.py`, outputs `results.txt` and `results_transfer.png`) used 20 runs on layouts B and C, one greedy rollout per agent "
    "and one shared random stream per run. Its results are kept unchanged. Because version 2 uses different random-number handling, the numbers "
    "differ slightly between versions; they should not be mixed in one table.")
add("")
add("## 8. Future work")
add("More systematic goal and wall variations, longer or shorter pretraining, more runs, a recurrent (LSTM) agent trained across many layouts, "
    "and place-cell-like state representations with shuffled remapping, which is closer to the original research question.")
add("")
add("## 9. Reproducibility")
add("Python %s, NumPy %s, run on %s. Commands: `python run_v2.py` then `python make_report_v2.py`. Outputs: `results_v2/summary.csv`, "
    "`results_v2/per_run_results.csv`, `results_v2/results_v2.json`, `results_v2/results_v2.txt`, `results_v2/learning_curves_v2.png`." %
    (S["python"], S["numpy"], S["date"]))

with open("REPORT_v2.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("Saved: REPORT_v2.md")