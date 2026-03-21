"""Compare RL960 vs GLM-5 on the 200 overlapping prompts using all metrics."""
import json, re, statistics, math
from collections import defaultdict
from scipy.stats import spearmanr

glm5_res = json.load(open("out/bench/glm5/healthbench_rubric_quality.json"))
base_res = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl_res = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))

glm5_pp = {r["prompt_id"]: r for r in glm5_res["per_prompt"]}
base_pp = {r["prompt_id"]: r for r in base_res["per_prompt"]}
rl_pp = {r["prompt_id"]: r for r in rl_res["per_prompt"]}

with open("data/healthbench_splits/benchmark_meta_eval.jsonl") as f:
    grouped = defaultdict(list)
    for line in f:
        row = json.loads(line)
        grouped[row["prompt_id"]].append(row)

label_var = {}
for pid, rows in grouped.items():
    ls = [sum(1 for x in r["binary_labels"] if x)/max(len(r["binary_labels"]),1) for r in rows[:8]]
    label_var[pid] = statistics.variance(ls) if len(ls)>=2 else 0

# Common + discriminative
common = set(glm5_pp) & set(base_pp) & set(rl_pp)
disc = [pid for pid in common if label_var.get(pid,0) > 0.01]
print(f"Common prompts: {len(common)}, Discriminative: {len(disc)}")

# 1. Pairwise acc
print("\n=== PAIRWISE ACC (discriminative, n={}) ===".format(len(disc)))
for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    pw = statistics.mean([pp[pid]["pairwise_acc"] for pid in disc])
    print(f"  {name}: {pw:.4f}")

# 2. Confident pair accuracy
print("\n=== CONFIDENT PAIR ACC (|label_gap|>0.2) ===")
for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    correct = total = 0
    for pid in disc:
        ps = pp[pid].get("pred_scores", [])
        ls = pp[pid].get("label_scores", [])
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                if abs(ls[i]-ls[j]) < 0.2: continue
                total += 1
                if (ls[i]>ls[j]) == (ps[i]>ps[j]): correct += 1
                elif ps[i]==ps[j]: correct += 0.5
    print(f"  {name}: {correct/total:.4f} (n={total})")

# 3. Win rate RL vs GLM5
rl_w = glm_w = tie = 0
for pid in disc:
    r = rl_pp[pid]["pairwise_acc"]
    g = glm5_pp[pid]["pairwise_acc"]
    if r > g + 0.001: rl_w += 1
    elif g > r + 0.001: glm_w += 1
    else: tie += 1
print(f"\n=== WIN RATE: RL vs GLM5 (disc) ===")
print(f"  RL: {rl_w} ({100*rl_w/len(disc):.1f}%)")
print(f"  GLM5: {glm_w} ({100*glm_w/len(disc):.1f}%)")
print(f"  Tied: {tie} ({100*tie/len(disc):.1f}%)")

# 4. Hard-prompt subset (base < 0.5)
hard = [pid for pid in disc if base_pp[pid]["pairwise_acc"] < 0.5]
print(f"\n=== HARD PROMPTS (n={len(hard)}) ===")
for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    pw = statistics.mean([pp[pid]["pairwise_acc"] for pid in hard])
    print(f"  {name}: {pw:.4f}")

# 5. Spearman
print(f"\n=== SPEARMAN (disc) ===")
for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    vals = []
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            rho,_ = spearmanr(ps,ls)
            if not math.isnan(rho): vals.append(rho)
    print(f"  {name}: {statistics.mean(vals):.4f} (n={len(vals)})")
