"""Compare holistic vs per-criterion judge mode results."""
import json, statistics, math
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score, roc_auc_score

models = ['base_qwen3_8b', 'rl_step_960', 'glm5']
labels = ['Base Qwen3-8B', 'GRM-RL 8B', 'GLM-5']

holistic = {}
for m in models:
    try:
        holistic[m] = json.load(open(f'out/bench_full/{m}/healthbench_rubric_quality.json'))
    except: pass

per_crit = {}
for m in models:
    try:
        per_crit[m] = json.load(open(f'out/bench_full/{m}/healthbench_rubric_quality_per_criterion.json'))
    except: pass

with open('data/healthbench_splits/benchmark_meta_eval.jsonl') as f:
    grouped = defaultdict(list)
    for line in f:
        row = json.loads(line)
        grouped[row['prompt_id']].append(row)

disc_pids = set()
for pid, rows in grouped.items():
    ls = [sum(1 for x in r['binary_labels'] if x)/max(len(r['binary_labels']),1) for r in rows[:8]]
    if len(ls) >= 2 and statistics.variance(ls) > 0.01:
        disc_pids.add(pid)

def calc(data, pids):
    pp = [r for r in data['per_prompt'] if r['prompt_id'] in pids]
    n = len(pp)
    if n == 0: return None
    avg_pa = sum(r['pairwise_acc'] for r in pp) / n
    avg_res = sum(r['resolution'] for r in pp) / n
    n_sens = sum(1 for r in pp if len(set(int(s) for s in r['pred_scores'])) >= 2)
    sp, kt, nd, au, cp = [], [], [], [], []
    for r in pp:
        p2, la = r['pred_scores'], r['label_scores']
        if len(set(la)) < 2: continue
        rho, _ = spearmanr(p2, la)
        if not math.isnan(rho): sp.append(rho)
        tau, _ = kendalltau(p2, la)
        if not math.isnan(tau): kt.append(tau)
        try: nd.append(ndcg_score([la], [p2], k=4))
        except: pass
        try:
            bl = [1 if l > 0.5 else 0 for l in la]
            if len(set(bl)) >= 2: au.append(roc_auc_score(bl, p2))
        except: pass
        cc, ct = 0.0, 0
        for i in range(len(p2)):
            for j in range(i+1, len(p2)):
                if abs(la[i]-la[j]) <= 0.2: continue
                ct += 1
                if p2[i] == p2[j]: cc += 0.5
                elif (la[i]>la[j]) == (p2[i]>p2[j]): cc += 1.0
        if ct > 0: cp.append(cc/ct)
    M = lambda xs: sum(xs)/len(xs) if xs else float('nan')
    return dict(n=n, pw=avg_pa, kt=M(kt), sp=M(sp), nd=M(nd), au=M(au), cp=M(cp), se=n_sens/n, re=avg_res)

h = {m: calc(holistic[m], disc_pids) for m in models if m in holistic}
p = {}
for m in models:
    if m in per_crit:
        s = per_crit[m]['summary']
        p[m] = dict(n=s['num_prompts'], pw=s['avg_pairwise_acc'], kt=s['kendall_tau'],
                     sp=s['discrimination']['avg_spearman'], nd=s['ndcg_at_4'], au=s['auc_roc'],
                     cp=s['confident_pair_acc'], se=s['score_sensitivity'], re=s['avg_resolution'])

ks = ['pw','kt','sp','nd','au','cp','se','re']
kl = {'pw':'Pairwise Acc','kt':'Kendall tau-b','sp':'Spearman rho','nd':'NDCG@4',
      'au':'AUC-ROC','cp':'ConfPair(>0.2)','se':'Sensitivity','re':'Resolution'}

print("="*95)
print("HOLISTIC vs PER-CRITERION Comparison (disc. prompts, same judge: qwen3.5-35b-a3b)")
print("="*95)
for i, m in enumerate(models):
    if m not in h or m not in p: continue
    hm, pm = h[m], p[m]
    print(f"\n--- {labels[i]} (holistic N={hm['n']}, per-crit N={pm['n']}) ---")
    print(f"{'Metric':>16s} | {'Holistic':>10s} | {'Per-Crit':>10s} | {'Delta':>10s}")
    print("-"*55)
    for k in ks:
        hv, pv = hm[k], pm[k]
        d = pv - hv
        if k == 'se':
            print(f"{kl[k]:>16s} | {hv:>9.1%} | {pv:>9.1%} | {d:>+9.1%}")
        else:
            print(f"{kl[k]:>16s} | {hv:>10.4f} | {pv:>10.4f} | {d:>+10.4f}")

print(f"\n{'='*95}")
print("PER-CRITERION: Cross-Model Comparison")
print(f"{'='*95}")
hdr = f"{'Metric':>16s} |"
for i, m in enumerate(models):
    if m in p: hdr += f" {labels[i]:>14s} |"
print(hdr)
print("-"*70)
for k in ks:
    row = f"{kl[k]:>16s} |"
    for m in models:
        if m in p:
            v = p[m][k]
            if k == 'se': row += f" {v:>13.1%} |"
            else: row += f" {v:>14.4f} |"
    print(row)

# Also print holistic cross-model for side-by-side
print(f"\n{'='*95}")
print("HOLISTIC: Cross-Model Comparison (same 570 disc. prompts)")
print(f"{'='*95}")
hdr = f"{'Metric':>16s} |"
for i, m in enumerate(models):
    if m in h: hdr += f" {labels[i]:>14s} |"
print(hdr)
print("-"*70)
for k in ks:
    row = f"{kl[k]:>16s} |"
    for m in models:
        if m in h:
            v = h[m][k]
            if k == 'se': row += f" {v:>13.1%} |"
            else: row += f" {v:>14.4f} |"
    print(row)
