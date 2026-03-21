"""Compare base vs RL960 vs GLM-5 on the SAME 200 prompts (old GLM-5 run)."""
import json, statistics
from collections import defaultdict

glm5 = json.load(open("out/bench/glm5/healthbench_rubric_quality.json"))
base = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))

glm5_pp = {r["prompt_id"]: r for r in glm5["per_prompt"]}
base_pp = {r["prompt_id"]: r for r in base["per_prompt"]}
rl_pp = {r["prompt_id"]: r for r in rl["per_prompt"]}

# Find common PIDs
common = set(glm5_pp) & set(base_pp) & set(rl_pp)
print(f"Common prompts: {len(common)}")

# 1. Overall metrics on common set
for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    pw = [pp[pid]["pairwise_acc"] for pid in common]
    res = [pp[pid]["resolution"] for pid in common]
    print(f"\n{name} (n={len(common)}):")
    print(f"  pairwise_acc: {statistics.mean(pw):.4f}")
    print(f"  resolution:   {statistics.mean(res):.4f}")

# 2. Pairwise: RL vs GLM5
rl_wins = glm_wins = ties = 0
for pid in common:
    r = rl_pp[pid]["pairwise_acc"]
    g = glm5_pp[pid]["pairwise_acc"]
    if r > g + 0.001: rl_wins += 1
    elif g > r + 0.001: glm_wins += 1
    else: ties += 1
print(f"\n=== HEAD-TO-HEAD: RL vs GLM5 ===")
print(f"  RL wins: {rl_wins} ({100*rl_wins/len(common):.1f}%)")
print(f"  GLM5 wins: {glm_wins} ({100*glm_wins/len(common):.1f}%)")
print(f"  Tied: {ties} ({100*ties/len(common):.1f}%)")

# 3. Top-bottom gap
from scipy.stats import spearmanr
import math

def compute_tb(pp_entry):
    preds = pp_entry.get("pred_scores", [])
    labels = pp_entry.get("label_scores", [])
    if not labels: return None
    best, worst = max(labels), min(labels)
    if best == worst: return None
    top = [s for s,l in zip(preds,labels) if l==best]
    bot = [s for s,l in zip(preds,labels) if l==worst]
    if not top or not bot: return None
    return statistics.mean(top) - statistics.mean(bot)

for name, pp in [("Base", base_pp), ("RL960", rl_pp), ("GLM5", glm5_pp)]:
    tbs = [compute_tb(pp[pid]) for pid in common]
    tbs = [t for t in tbs if t is not None]
    print(f"\n{name} top-bottom gap: {statistics.mean(tbs):.4f} (n={len(tbs)})")

# 4. RL vs GLM5 top-bottom gap head-to-head
rl_tb_wins = glm_tb_wins = tb_ties = 0
for pid in common:
    r_tb = compute_tb(rl_pp[pid])
    g_tb = compute_tb(glm5_pp[pid])
    if r_tb is None or g_tb is None: continue
    if r_tb > g_tb + 0.01: rl_tb_wins += 1
    elif g_tb > r_tb + 0.01: glm_tb_wins += 1
    else: tb_ties += 1
total = rl_tb_wins + glm_tb_wins + tb_ties
print(f"\n=== TOP-BOTTOM GAP: RL vs GLM5 ===")
print(f"  RL wins: {rl_tb_wins} ({100*rl_tb_wins/total:.1f}%)")
print(f"  GLM5 wins: {glm_tb_wins} ({100*glm_tb_wins/total:.1f}%)")
print(f"  Tied: {tb_ties} ({100*tb_ties/total:.1f}%)")

# 5. Category breakdown RL vs GLM5
cats = defaultdict(lambda: {"rl":[], "glm":[]})
for pid in common:
    cat = rl_pp[pid].get("category", "unknown")
    cats[cat]["rl"].append(rl_pp[pid]["pairwise_acc"])
    cats[cat]["glm"].append(glm5_pp[pid]["pairwise_acc"])

print(f"\n=== CATEGORY: RL vs GLM5 ===")
for cat in sorted(cats):
    r = statistics.mean(cats[cat]["rl"])
    g = statistics.mean(cats[cat]["glm"])
    n = len(cats[cat]["rl"])
    d = r - g
    w = "RL" if d > 0.005 else ("GLM5" if d < -0.005 else "TIE")
    print(f"  {cat[:55]:55s} n={n:2d} RL={r:.4f} GLM={g:.4f} d={d:+.4f} {w}")
