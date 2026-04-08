"""Compare all model benchmark results on HealthBench."""
import json, statistics, math
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau

base_res = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl700_res = json.load(open("out/bench_full/rl700/healthbench_rubric_quality.json"))
rl960_res = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))
glm5_res = json.load(open("out/bench_full/glm5/healthbench_rubric_quality.json"))

base_pp = {r["prompt_id"]: r for r in base_res["per_prompt"]}
rl700_pp = {r["prompt_id"]: r for r in rl700_res["per_prompt"]}
rl960_pp = {r["prompt_id"]: r for r in rl960_res["per_prompt"]}
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

def ndcg_at_k(pred_scores, label_scores, k=None):
    n = len(pred_scores)
    if k is None: k = n
    k = min(k, n)
    order = sorted(range(n), key=lambda i: -pred_scores[i])
    dcg = sum(label_scores[order[i]] / math.log2(i+2) for i in range(k))
    ideal = sorted(range(n), key=lambda i: -label_scores[i])
    idcg = sum(label_scores[ideal[i]] / math.log2(i+2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0

def auroc(pred_scores, label_scores, threshold=0.5):
    labels = [1 if l > threshold else 0 for l in label_scores]
    if len(set(labels)) < 2: return None
    pairs = sorted(zip(pred_scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    tp_total = sum(labels)
    fp_total = len(labels) - tp_total
    if tp_total == 0 or fp_total == 0: return None
    points = []
    for score, lab in pairs:
        if lab == 1: tp += 1
        else: fp += 1
        points.append((fp/fp_total, tp/tp_total))
    auc = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for fpr, tpr in points:
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc

def compute_metrics(pp, pids):
    d = {}
    disc = [p for p in pids if label_var.get(p,0) > 0.01]
    hard = [p for p in disc if base_pp[p]["pairwise_acc"] < 0.5]
    easy = [p for p in disc if base_pp[p]["pairwise_acc"] >= 0.8]
    d["pw"] = statistics.mean([pp[p]["pairwise_acc"] for p in disc])
    vals = []
    for p in disc:
        ps, ls = pp[p].get("pred_scores",[]), pp[p].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            tau,_ = kendalltau(ps,ls)
            if not math.isnan(tau): vals.append(tau)
    d["kendall"] = statistics.mean(vals) if vals else 0
    vals = []
    for p in disc:
        ps, ls = pp[p].get("pred_scores",[]), pp[p].get("label_scores",[])
        if len(set(ls))>=2 and len(ps)>=2:
            rho,_ = spearmanr(ps,ls)
            if not math.isnan(rho): vals.append(rho)
    d["spearman"] = statistics.mean(vals) if vals else 0
    vals = []
    for p in disc:
        ps, ls = pp[p].get("pred_scores",[]), pp[p].get("label_scores",[])
        if len(ps)>=4 and len(set(ls))>=2:
            vals.append(ndcg_at_k(ps, ls, k=4))
    d["ndcg4"] = statistics.mean(vals) if vals else 0
    vals = []
    for p in disc:
        ps, ls = pp[p].get("pred_scores",[]), pp[p].get("label_scores",[])
        a = auroc(ps, ls)
        if a is not None: vals.append(a)
    d["auc"] = statistics.mean(vals) if vals else 0
    correct = total = 0
    for p in disc:
        ps, ls = pp[p].get("pred_scores",[]), pp[p].get("label_scores",[])
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ls[i]-ls[j]) < 0.2: continue
                total += 1
                if (ls[i]>ls[j])==(ps[i]>ps[j]): correct += 1
                elif ps[i]==ps[j]: correct += 0.5
    d["conf_pair"] = correct/total if total > 0 else 0
    d["hard_pw"] = statistics.mean([pp[p]["pairwise_acc"] for p in hard]) if hard else 0
    d["easy_pw"] = statistics.mean([pp[p]["pairwise_acc"] for p in easy]) if easy else 0
    d["sensitivity"] = 100*sum(1 for p in disc if len(set(int(s) for s in pp[p].get("pred_scores",[])))>=2)/len(disc) if disc else 0
    d["n_disc"] = len(disc)
    d["n_hard"] = len(hard)
    return d

# TABLE 1: Full 709 prompts
full_pids = sorted(set(base_pp) & set(rl700_pp) & set(rl960_pp))
disc_full = [p for p in full_pids if label_var.get(p,0)>0.01]

print("=" * 90)
print("TABLE 1: Full 709 Prompts (Base vs RL-Step700 vs RL-Step960)")
print("=" * 90)
mf = {}
for name, pp in [("base", base_pp), ("rl700", rl700_pp), ("rl960", rl960_pp)]:
    mf[name] = compute_metrics(pp, full_pids)

print(f"\n{'Metric':<23s} | {'Base 8B':>12s} | {'RL-700':>12s} | {'RL-960':>12s} | {'700vBase':>10s}")
print("-" * 72)
for key, label in [
    ("pw","Pairwise Acc"),("kendall","Kendall t-b"),("spearman","Spearman rho"),
    ("ndcg4","NDCG@4"),("auc","AUC-ROC"),("conf_pair","Conf.Pair(>0.2)"),
    ("hard_pw","Hard PW(<0.5)"),("easy_pw","Easy PW(>=0.8)"),("sensitivity","Sensitivity %"),
]:
    b,r7,r9 = mf["base"][key], mf["rl700"][key], mf["rl960"][key]
    d = r7 - b
    fmt = ".4f" if key != "sensitivity" else ".1f"
    s = "+" if d > 0 else ""
    print(f"  {label:<21s} | {b:>12{fmt}} | {r7:>12{fmt}} | {r9:>12{fmt}} | {s}{d:>9{fmt}}")
print(f"\n  ({len(disc_full)} disc, {mf['base']['n_hard']} hard prompts)")

rl7_w = base_w = tie = 0
for pid in disc_full:
    r,b = rl700_pp[pid]["pairwise_acc"], base_pp[pid]["pairwise_acc"]
    if r>b+0.001: rl7_w+=1
    elif b>r+0.001: base_w+=1
    else: tie+=1
print(f"\n  H2H RL700 vs Base: RL {rl7_w} ({100*rl7_w/len(disc_full):.1f}%) | Base {base_w} ({100*base_w/len(disc_full):.1f}%) | Tie {tie}")

# TABLE 2: 4-model comparison on common prompts
common4 = sorted(set(base_pp) & set(rl700_pp) & set(rl960_pp) & set(glm5_pp))
disc_4 = [p for p in common4 if label_var.get(p,0)>0.01]

print("\n" + "=" * 95)
print("TABLE 2: Common 559 Prompts (All 4 Models)")
print("=" * 95)
m4 = {}
for name, pp in [("base",base_pp),("rl700",rl700_pp),("rl960",rl960_pp),("glm5",glm5_pp)]:
    m4[name] = compute_metrics(pp, common4)

print(f"\n{'Metric':<23s} | {'Base 8B':>10s} | {'RL-700':>10s} | {'RL-960':>10s} | {'GLM-5':>10s} | {'700vGLM5':>9s}")
print("-" * 82)
for key, label in [
    ("pw","Pairwise Acc"),("kendall","Kendall t-b"),("spearman","Spearman rho"),
    ("ndcg4","NDCG@4"),("auc","AUC-ROC"),("conf_pair","Conf.Pair(>0.2)"),
    ("hard_pw","Hard PW(<0.5)"),("easy_pw","Easy PW(>=0.8)"),("sensitivity","Sensitivity %"),
]:
    b,r7,r9,g5 = m4["base"][key],m4["rl700"][key],m4["rl960"][key],m4["glm5"][key]
    d = r7 - g5
    fmt = ".4f" if key != "sensitivity" else ".1f"
    s = "+" if d > 0 else ""
    print(f"  {label:<21s} | {b:>10{fmt}} | {r7:>10{fmt}} | {r9:>10{fmt}} | {g5:>10{fmt}} | {s}{d:>8{fmt}}")
print(f"\n  ({len(disc_4)} disc, {m4['base']['n_hard']} hard prompts)")

rl7_w = g5_w = tie = 0
for pid in disc_4:
    r,g = rl700_pp[pid]["pairwise_acc"], glm5_pp[pid]["pairwise_acc"]
    if r>g+0.001: rl7_w+=1
    elif g>r+0.001: g5_w+=1
    else: tie+=1
print(f"\n  H2H RL700 vs GLM5: RL {rl7_w} ({100*rl7_w/len(disc_4):.1f}%) | GLM5 {g5_w} ({100*g5_w/len(disc_4):.1f}%) | Tie {tie}")

rl7_w = r9_w = tie = 0
for pid in disc_full:
    r7,r9 = rl700_pp[pid]["pairwise_acc"], rl960_pp[pid]["pairwise_acc"]
    if r7>r9+0.001: rl7_w+=1
    elif r9>r7+0.001: r9_w+=1
    else: tie+=1
print(f"  H2H RL700 vs RL960: 700 wins {rl7_w} ({100*rl7_w/len(disc_full):.1f}%) | 960 wins {r9_w} ({100*r9_w/len(disc_full):.1f}%) | Tie {tie}")
