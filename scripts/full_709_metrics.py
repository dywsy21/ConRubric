"""Full 709-prompt metrics: Base vs GRM-RL (no GLM-5 dependency)."""
import json, statistics, math
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
import numpy as np

base_res = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl_res = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))

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

all_pids = sorted(set(base_pp) & set(rl_pp))
disc = [pid for pid in all_pids if label_var.get(pid,0) > 0.01]
non_disc = [pid for pid in all_pids if label_var.get(pid,0) <= 0.01]

print(f"Total prompts: {len(all_pids)}")
print(f"Discriminative (label variance > 0.01): {len(disc)}")
print(f"Non-discriminative: {len(non_disc)}")

def ndcg_at_k(pred_scores, label_scores, k=None):
    n = len(pred_scores)
    if k is None: k = n
    k = min(k, n)
    order = sorted(range(n), key=lambda i: -pred_scores[i])
    dcg = sum(label_scores[order[i]] / math.log2(i+2) for i in range(k))
    ideal_order = sorted(range(n), key=lambda i: -label_scores[i])
    idcg = sum(label_scores[ideal_order[i]] / math.log2(i+2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0

def auroc(pred_scores, label_scores, threshold=0.5):
    labels = [1 if l > threshold else 0 for l in label_scores]
    if len(set(labels)) < 2: return None
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
    auc = 0
    prev_fpr, prev_tpr = 0, 0
    for fpr, tpr in points:
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc

models = [("Base Qwen3-8B", base_pp), ("GRM-RL Step960", rl_pp)]

# ============================================================
print("\n" + "=" * 70)
print("1. PAIRWISE ACCURACY")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts (n={len(pids)})]")
    for name, pp in models:
        vals = [pp[pid]["pairwise_acc"] for pid in pids if pid in pp]
        print(f"    {name:20s}: {statistics.mean(vals):.4f}")

# ============================================================
print("\n" + "=" * 70)
print("2. SPEARMAN rho")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts]")
    for name, pp in models:
        vals = []
        for pid in pids:
            ps = pp[pid].get("pred_scores",[])
            ls = pp[pid].get("label_scores",[])
            if len(set(ls))>=2 and len(ps)>=2:
                rho,_ = spearmanr(ps,ls)
                if not math.isnan(rho): vals.append(rho)
        print(f"    {name:20s}: {statistics.mean(vals):.4f} (n={len(vals)})")

# ============================================================
print("\n" + "=" * 70)
print("3. KENDALL TAU-B")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts]")
    for name, pp in models:
        vals = []
        for pid in pids:
            ps = pp[pid].get("pred_scores",[])
            ls = pp[pid].get("label_scores",[])
            if len(set(ls))>=2 and len(ps)>=2:
                tau,_ = kendalltau(ps,ls)
                if not math.isnan(tau): vals.append(tau)
        print(f"    {name:20s}: {statistics.mean(vals):.4f} (n={len(vals)})")

# ============================================================
print("\n" + "=" * 70)
print("4. NDCG@4")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts]")
    for name, pp in models:
        vals = []
        for pid in pids:
            ps = pp[pid].get("pred_scores",[])
            ls = pp[pid].get("label_scores",[])
            if len(ps)>=4 and len(set(ls))>=2:
                vals.append(ndcg_at_k(ps, ls, k=4))
        print(f"    {name:20s}: {statistics.mean(vals):.4f} (n={len(vals)})")

# ============================================================
print("\n" + "=" * 70)
print("5. AUC-ROC")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts]")
    for name, pp in models:
        vals = []
        for pid in pids:
            ps = pp[pid].get("pred_scores",[])
            ls = pp[pid].get("label_scores",[])
            a = auroc(ps, ls)
            if a is not None: vals.append(a)
        print(f"    {name:20s}: {statistics.mean(vals):.4f} (n={len(vals)})")

# ============================================================
print("\n" + "=" * 70)
print("6. CONFIDENT PAIR ACCURACY (multiple thresholds)")
print("=" * 70)
for thresh in [0.1, 0.2, 0.3]:
    print(f"\n  --- label gap > {thresh} ---")
    for subset_name, pids in [("All", all_pids), ("Disc", disc)]:
        for name, pp in models:
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
            print(f"    {name:20s} [{subset_name}]: {correct/total:.4f} (n_pairs={total})")

# ============================================================
print("\n" + "=" * 70)
print("7. SCORE SENSITIVITY (non-flat rate)")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    print(f"\n  [{subset_name} prompts]")
    for name, pp in models:
        n_varied = sum(1 for pid in pids
                       if len(set(int(s) for s in pp[pid].get("pred_scores",[]))) >= 2)
        print(f"    {name:20s}: {100*n_varied/len(pids):.1f}% ({n_varied}/{len(pids)})")

# ============================================================
print("\n" + "=" * 70)
print("8. HEAD-TO-HEAD WIN RATE (RL vs Base)")
print("=" * 70)
for subset_name, pids in [("All 709", all_pids), ("Discriminative", disc)]:
    rl_w = base_w = tie = 0
    for pid in pids:
        r = rl_pp[pid]["pairwise_acc"]
        b = base_pp[pid]["pairwise_acc"]
        if r > b + 0.001: rl_w += 1
        elif b > r + 0.001: base_w += 1
        else: tie += 1
    print(f"\n  [{subset_name} prompts (n={len(pids)})]")
    print(f"    GRM-RL wins: {rl_w} ({100*rl_w/len(pids):.1f}%)")
    print(f"    Base wins:   {base_w} ({100*base_w/len(pids):.1f}%)")
    print(f"    Tied:        {tie} ({100*tie/len(pids):.1f}%)")

# ============================================================
print("\n" + "=" * 70)
print("9. HARD PROMPTS (base pairwise < 0.5)")
print("=" * 70)
hard_all = [pid for pid in all_pids if base_pp[pid]["pairwise_acc"] < 0.5]
hard_disc = [pid for pid in disc if base_pp[pid]["pairwise_acc"] < 0.5]
for subset_name, pids in [("All hard", hard_all), ("Disc hard", hard_disc)]:
    print(f"\n  [{subset_name} (n={len(pids)})]")
    for name, pp in models:
        pw = statistics.mean([pp[pid]["pairwise_acc"] for pid in pids])
        print(f"    {name:20s}: PW={pw:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("10. EASY PROMPTS (base pairwise >= 0.8)")
print("=" * 70)
easy = [pid for pid in disc if base_pp[pid]["pairwise_acc"] >= 0.8]
print(f"\n  [Easy-disc (n={len(easy)})]")
for name, pp in models:
    pw = statistics.mean([pp[pid]["pairwise_acc"] for pid in easy])
    print(f"    {name:20s}: PW={pw:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE (for paper)")
print("=" * 70)
# Compute all key metrics for summary
summary = {}
for name, pp in models:
    d = {}
    # PW all disc
    d["pw_disc"] = statistics.mean([pp[pid]["pairwise_acc"] for pid in disc])
    # Kendall
    vals = []
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            tau,_ = kendalltau(ps,ls)
            if not math.isnan(tau): vals.append(tau)
    d["kendall"] = statistics.mean(vals)
    # Spearman
    vals = []
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            rho,_ = spearmanr(ps,ls)
            if not math.isnan(rho): vals.append(rho)
    d["spearman"] = statistics.mean(vals)
    # NDCG
    vals = []
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        if len(ps)>=4 and len(set(ls))>=2:
            vals.append(ndcg_at_k(ps,ls,k=4))
    d["ndcg4"] = statistics.mean(vals)
    # AUC
    vals = []
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        a = auroc(ps, ls)
        if a is not None: vals.append(a)
    d["auc"] = statistics.mean(vals)
    # Confident pair (>0.2)
    correct = total = 0
    for pid in disc:
        ps = pp[pid].get("pred_scores",[])
        ls = pp[pid].get("label_scores",[])
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ls[i]-ls[j]) < 0.2: continue
                total += 1
                if (ls[i]>ls[j])==(ps[i]>ps[j]): correct += 1
                elif ps[i]==ps[j]: correct += 0.5
    d["conf_pair"] = correct/total
    # Hard prompt PW
    hard = [pid for pid in disc if base_pp[pid]["pairwise_acc"] < 0.5]
    d["hard_pw"] = statistics.mean([pp[pid]["pairwise_acc"] for pid in hard])
    # Sensitivity
    d["sensitivity"] = 100*sum(1 for pid in disc
        if len(set(int(s) for s in pp[pid].get("pred_scores",[])))>=2)/len(disc)
    summary[name] = d

print()
print("%-25s | %14s | %14s | %10s" % ("Metric", "Base Qwen3-8B", "GRM-RL 8B", "Delta"))
print("-"*70)
for key, label in [
    ("pw_disc", "Pairwise Acc"),
    ("kendall", "Kendall τ-b"),
    ("spearman", "Spearman ρ"),
    ("ndcg4", "NDCG@4"),
    ("auc", "AUC-ROC"),
    ("conf_pair", "Conf.Pair(>0.2)"),
    ("hard_pw", "Hard PW(<0.5)"),
    ("sensitivity", "Sensitivity %"),
]:
    b = summary["Base Qwen3-8B"][key]
    r = summary["GRM-RL Step960"][key]
    delta = r - b
    fmt = ".4f" if key != "sensitivity" else ".1f"
    sign = "+" if delta > 0 else ""
    winner = " ✓" if delta > 0 else ""
    print(f"  {label:<23s} | {b:>14{fmt}} | {r:>14{fmt}} | {sign}{delta:>9{fmt}}{winner}")

print(f"\n  (All metrics on {len(disc)} discriminative prompts out of {len(all_pids)} total)")
