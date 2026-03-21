"""Find the best showcase prompt where RL960 clearly beats both base and GLM-5."""
import json, re, statistics
from collections import defaultdict

base_res = json.load(open("out/bench_full/base_qwen3_8b/healthbench_rubric_quality.json"))
rl_res = json.load(open("out/bench_full/rl_step_960/healthbench_rubric_quality.json"))
glm5_res = json.load(open("out/bench/glm5/healthbench_rubric_quality.json"))

base_pp = {r["prompt_id"]: r for r in base_res["per_prompt"]}
rl_pp = {r["prompt_id"]: r for r in rl_res["per_prompt"]}
glm5_pp = {r["prompt_id"]: r for r in glm5_res["per_prompt"]}

base_cache = json.load(open("out/bench_full/base_qwen3_8b/rubric_cache.json"))
rl_cache = json.load(open("out/bench_full/rl_step_960/rubric_cache.json"))

with open("data/healthbench_splits/benchmark_meta_eval.jsonl") as f:
    grouped = defaultdict(list)
    for line in f:
        row = json.loads(line)
        grouped[row["prompt_id"]].append(row)

prompt_texts = {}
for pid, rows in grouped.items():
    prompt = rows[0]["prompt"]
    if isinstance(prompt, list):
        parts = []
        for m in prompt:
            if isinstance(m, dict):
                c = (m.get("content") or "").strip()
                if c: parts.append(c)
        prompt_texts[pid] = "\n".join(parts)
    else:
        prompt_texts[pid] = str(prompt)

# Find common prompts where RL beats both
common = set(glm5_pp) & set(base_pp) & set(rl_pp)
candidates = []
for pid in common:
    r = rl_pp[pid]["pairwise_acc"]
    b = base_pp[pid]["pairwise_acc"]
    g = glm5_pp[pid]["pairwise_acc"]
    # RL must beat both by a clear margin
    if r > b + 0.05 and r > g + 0.05 and r >= 0.6:
        # Prefer prompts with reasonable question length (not too long)
        q = prompt_texts.get(pid, "")
        if 50 < len(q) < 800:
            candidates.append((pid, r, b, g, r-max(b,g)))

candidates.sort(key=lambda x: x[4], reverse=True)

print(f"Found {len(candidates)} good showcase prompts")
for i, (pid, r, b, g, margin) in enumerate(candidates[:10]):
    q = prompt_texts[pid][:120]
    n_cr_b = len(re.findall(r'^\[[\+\-]\]', base_cache.get(pid,""), re.MULTILINE))
    n_cr_r = len(re.findall(r'^\[[\+\-]\]', rl_cache.get(pid,""), re.MULTILINE))
    cat = rl_pp[pid].get("category","")[:40]
    print(f"\n#{i+1} margin={margin:.3f}")
    print(f"  PW: base={b:.3f} RL={r:.3f} GLM5={g:.3f}")
    print(f"  Cat: {cat}")
    print(f"  Criteria: base={n_cr_b} rl={n_cr_r}")
    print(f"  Q: {q}...")
    print(f"  PID: {pid}")
