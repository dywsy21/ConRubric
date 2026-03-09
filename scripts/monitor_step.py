#!/usr/bin/env python3
"""Monitor a single RL step's rollout log for empathy template and generic rubrics."""
import json, re, os, sys

log_dir = "out/rl/rollout_logs"
empathy_re = re.compile(r"(?i)models?\s+answers?\s+with")
generic_indicators = re.compile(
    r"(?i)(?:empathy|compassion|supportive tone|actionable advice|clarity|accuracy)"
)

step = int(sys.argv[1]) if len(sys.argv) > 1 else None

if step is None:
    # Auto-detect latest step
    files = sorted(os.listdir(log_dir))
    if not files:
        print("No rollout logs found")
        sys.exit(1)
    step = int(files[-1].split("_")[1].split(".")[0])

fname = os.path.join(log_dir, f"step_{step:04d}.jsonl")
if not os.path.exists(fname):
    print(f"No rollout log for step {step}")
    sys.exit(1)

with open(fname) as f:
    items = [json.loads(l) for l in f]

print(f"===== STEP {step} ROLLOUT ANALYSIS =====\n")

for qi, item in enumerate(items[:4]):
    q = item["question"]
    rollouts = item.get("rollouts", [])
    rollouts_sorted = sorted(rollouts, key=lambda r: r.get("reward", 0), reverse=True)

    print(f"Q{qi}: {q[:90]}")

    for ri, r in enumerate(rollouts_sorted[:3]):
        rubric = r.get("rubric", "")
        reward = r.get("reward", 0)
        has_empathy = bool(empathy_re.search(rubric))
        has_generic = bool(generic_indicators.search(rubric))

        flags = []
        if has_empathy:
            flags.append("EMPATHY")
        if has_generic:
            flags.append("GENERIC")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""

        # Show first 2 criteria
        criteria = [l.strip() for l in rubric.split("\n") if l.strip().startswith("- [")]
        preview = " / ".join(c[:80] for c in criteria[:2])
        print(f"  #{ri+1} reward={reward:.2f}{flag_str}: {preview}")
    print()

# Count all rubrics for stats
total_all = 0
emp_all = 0
gen_all = 0
for item in items:
    for r in item.get("rollouts", []):
        rubric = r.get("rubric", "")
        total_all += 1
        if empathy_re.search(rubric):
            emp_all += 1
        if generic_indicators.search(rubric):
            gen_all += 1

print(f"--- OVERALL STATS (step {step}) ---")
print(f"Empathy template: {emp_all}/{total_all} ({100*emp_all/total_all:.1f}%)")
print(f"Generic indicators: {gen_all}/{total_all} ({100*gen_all/total_all:.1f}%)")
clean = total_all - max(emp_all, gen_all)
print(f"Clean rubrics: {clean}/{total_all} ({100*clean/total_all:.1f}%)")
