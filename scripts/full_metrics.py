"""Comprehensive benchmark metrics for GRM rubric quality evaluation."""
import json, re, statistics, math
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
import numpy as np

base_res = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl_res = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))
glm5_res = json.load(open("out/bench/glm5/healthbench_rubric_quality.json"))

base_pp = {r["prompt_id"]: r for r in base_res["per_prompt"]}
rl_pp = {r["prompt_id"]: r for r in rl_res["per_prompt"]}
glm5_pp = {r["prompt_id"]: r for r in glm5_res["per_prompt"]}

with open("data/healthbench_splits/benchmark_meta_eval.jsonl") as f:
    grouped = defaultdict(list)
    for line in f:
        row = json.loads(line)
        grouped[row["prompt_id"]].append(row)

label_var = {}
for pid, rows in grouped.items():
    ls = [sum(1 for x in r["binary_labels"] if x)/max(len(r["binary_labels"]),1) for r in rows[:8]]
    label_var[pid] = statistics.variance(ls) if len(ls)>=2 else 0

# Common (all 3) + discriminative
common3 = set(glm5_pp) & set(base_pp) & set(rl_pp)
disc3 = [pid for pid in common3 if label_var.get(pid,0) > 0.01]

# All 709 base vs RL
all_pids = set(base_pp) & set(rl_pp)
disc_all = [pid for pid in all_pids if label_var.get(pid,0) > 0.01]

def ndcg_at_k(pred_scores, label_scores, k=None):
    """NDCG: how well does the rubric rank the best completions to the top?"""
    n = len(pred_scores)
    if k is None: k = n
    k = min(k, n)
    # Sort by predicted score descending
    order = sorted(range(n), key=lambda i: -pred_scores[i])
    # Relevance = label_score (higher is better)
    dcg = sum(label_scores[order[i]] / math.log2(i+2) for i in range(k))
    # Ideal: sort by label_score descending
    ideal_order = sorted(range(n), key=lambda i: -label_scores[i])
    idcg = sum(label_scores[ideal_order[i]] / math.log2(i+2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0

def auroc(pred_scores, label_scores, threshold=0.5):
    """AUC-ROC: treat label>threshold as positive class, pred as classifier."""
    labels = [1 if l > threshold else 0 for l in label_scores]
    if len(set(labels)) < 2: return None
    # Simple trapezoidal AUC
    pairs = list(zip(pred_scores, labels))
    pairs.sort(key=lambda x: -x[0])
    tp = fp = 0
    tp_total = sum(labels)
    fp_total = len(labels) - tp_total
    if tp_total == 0 or fp_total == 0: return None
    points = []
    for score, lab in pairs:
        if lab == 1: tp += 1
        else: fp += 1
        points.append((fp/fp_total, tp/tp_total))
    # Trapezoidal integration
    auc = 0
    prev_fpr, prev_tpr = 0, 0
    for fpr, tpr in points:
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc

# ========================================
# METRIC 1: Kendall's tau-b
# ========================================
print("=" * 70)
print("METRIC 1: Kendall's tau-b (better for ordinal data with ties)")
print("=" * 70)
for label, pp, pids in [
    ("Base (709)", base_pp, disc_all),
    ("RL960 (709)", rl_pp, disc_all),
    ("Base (200)", base_pp, disc3),
    ("RL960 (200)", rl_pp, disc3),
    ("GLM5 (200)", glm5_pp, disc3),
]:
    vals = []
    for pid in pids:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            tau, _ = kendalltau(ps, ls)
            if not math.isnan(tau): vals.append(tau)
    print(f"  {label:15s}: tau={statistics.mean(vals):.4f} (n={len(vals)})")

# ========================================
# METRIC 2: NDCG@4 (top-half ranking quality)
# ========================================
print("\n" + "=" * 70)
print("METRIC 2: NDCG@4 (can the rubric rank best completions to top?)")
print("=" * 70)
for label, pp, pids in [
    ("Base (709)", base_pp, disc_all),
    ("RL960 (709)", rl_pp, disc_all),
    ("Base (200)", base_pp, disc3),
    ("RL960 (200)", rl_pp, disc3),
    ("GLM5 (200)", glm5_pp, disc3),
]:
    vals = []
    for pid in pids:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(ps) >= 4 and len(set(ls)) >= 2:
            v = ndcg_at_k(ps, ls, k=4)
            vals.append(v)
    print(f"  {label:15s}: NDCG@4={statistics.mean(vals):.4f} (n={len(vals)})")

# ========================================
# METRIC 3: AUC-ROC (binary classification: good vs bad completion)
# ========================================
print("\n" + "=" * 70)
print("METRIC 3: AUC-ROC (rubric as binary classifier: good vs bad)")
print("=" * 70)
for label, pp, pids in [
    ("Base (709)", base_pp, disc_all),
    ("RL960 (709)", rl_pp, disc_all),
    ("Base (200)", base_pp, disc3),
    ("RL960 (200)", rl_pp, disc3),
    ("GLM5 (200)", glm5_pp, disc3),
]:
    vals = []
    for pid in pids:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        a = auroc(ps, ls)
        if a is not None: vals.append(a)
    print(f"  {label:15s}: AUC={statistics.mean(vals):.4f} (n={len(vals)})")

# ========================================
# METRIC 4: Confident Pair Accuracy (multiple thresholds)
# ========================================
print("\n" + "=" * 70)
print("METRIC 4: Confident Pair Accuracy at varying thresholds")
print("=" * 70)
for thresh in [0.1, 0.2, 0.3]:
    print(f"\n  --- label gap > {thresh} ---")
    for label, pp, pids in [
        ("Base (709)", base_pp, disc_all),
        ("RL960 (709)", rl_pp, disc_all),
        ("Base (200)", base_pp, disc3),
        ("RL960 (200)", rl_pp, disc3),
        ("GLM5 (200)", glm5_pp, disc3),
    ]:
        correct = total = 0
        for pid in pids:
            ps = pp[pid].get("pred_scores",[])
            ls = pp[pid].get("label_scores",[])
            for i in range(len(ps)):
                for j in range(i+1,len(ps)):
                    if abs(ls[i]-ls[j]) < thresh: continue
                    total += 1
                    if (ls[i]>ls[j])==(ps[i]>ps[j]): correct += 1
                    elif ps[i]==ps[j]: correct += 0.5
        print(f"    {label:15s}: {correct/total:.4f} (n_pairs={total})")

# ========================================
# METRIC 5: Score Sensitivity (non-flat rate)
# ========================================
print("\n" + "=" * 70)
print("METRIC 5: Score Sensitivity (% prompts with non-flat scores)")
print("=" * 70)
for label, pp, pids in [
    ("Base (709)", base_pp, list(all_pids)),
    ("RL960 (709)", rl_pp, list(all_pids)),
    ("Base (200)", base_pp, list(common3)),
    ("RL960 (200)", rl_pp, list(common3)),
    ("GLM5 (200)", glm5_pp, list(common3)),
]:
    n_varied = sum(1 for pid in pids
                   if len(set(int(s) for s in pp[pid].get("pred_scores",[]))) >= 2)
    print(f"  {label:15s}: {100*n_varied/len(pids):.1f}% ({n_varied}/{len(pids)})")
