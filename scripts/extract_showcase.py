"""Extract full data for showcase prompt 4eaee2b6."""
import json, re
from collections import defaultdict

PID = "4eaee2b6-c260-4ca5-9491-425971ea778c"

base_cache = json.load(open("out/bench_full/base_qwen3_8b/rubric_cache.json"))
rl_cache = json.load(open("out/bench_full/rl_step_960/rubric_cache.json"))
glm5_rubrics = json.load(open("/tmp/glm5_sample_rubrics.json"))

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

rows = grouped[PID]
prompt = rows[0]["prompt"]
if isinstance(prompt, list):
    parts = []
    for m in prompt:
        if isinstance(m, dict):
            c = (m.get("content") or "").strip()
            if c: parts.append(c)
    question = "\n".join(parts)
else:
    question = str(prompt)

gt_rubric = rows[0].get("rubric", "")

print("=== QUESTION ===")
print(question)
print("\n=== GROUND TRUTH RUBRIC (HealthBench) ===")
print(gt_rubric[:1000])

print("\n=== BASE RUBRIC ===")
print(base_cache.get(PID, "N/A"))

print("\n=== RL960 RUBRIC ===")
print(rl_cache.get(PID, "N/A"))

# GLM5 rubric - check both caches
if PID in glm5_rubrics and glm5_rubrics[PID]:
    print("\n=== GLM5 RUBRIC ===")
    print(glm5_rubrics[PID])
else:
    print("\n=== GLM5 RUBRIC (not in sample cache, showing scores only) ===")

print("\n=== SCORES ===")
print(f"Base:  pw={base_pp[PID]['pairwise_acc']:.3f}  pred={base_pp[PID].get('pred_scores',[])}  labels={base_pp[PID].get('label_scores',[])}")
print(f"RL960: pw={rl_pp[PID]['pairwise_acc']:.3f}  pred={rl_pp[PID].get('pred_scores',[])}  labels={rl_pp[PID].get('label_scores',[])}")
print(f"GLM5:  pw={glm5_pp[PID]['pairwise_acc']:.3f}  pred={glm5_pp[PID].get('pred_scores',[])}  labels={glm5_pp[PID].get('label_scores',[])}")
