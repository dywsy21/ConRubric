#!/usr/bin/env python3
"""Analyze rubric specificity across training steps."""
import json, re, os

ROLLOUT_DIR = "out/rl/rollout_logs"

STOP = {"about", "would", "could", "should", "their", "these", "there",
        "where", "which", "while", "after", "being", "other", "first",
        "years", "based", "might", "every", "right", "since", "still",
        "under", "using", "without", "asked", "heard", "night",
        "think", "really", "super", "small", "trouble"}

for step in [321, 340, 370, 380, 420, 460, 500, 540]:
    fname = f"step_{step:04d}.jsonl"
    fpath = os.path.join(ROLLOUT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Step {step}: NOT FOUND")
        continue
    with open(fpath) as f:
        items = [json.loads(l) for l in f]

    total_q = len(items)
    q_with_specific_best = 0
    total_criteria = 0
    criteria_with_topic = 0

    for item in items:
        q = item["question"]
        rollouts = sorted(item["rollouts"], key=lambda r: r.get("reward", 0), reverse=True)
        best = rollouts[0]["rubric"]
        q_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{5,}", q)} - STOP
        criteria = [l.strip() for l in best.split("\n") if l.strip().startswith("- [")]
        total_criteria += len(criteria)
        
        found_any = False
        for c in criteria:
            c_lower = c.lower()
            if any(t in c_lower for t in q_terms):
                criteria_with_topic += 1
                found_any = True
        if found_any:
            q_with_specific_best += 1

    pct_q = 100 * q_with_specific_best / total_q if total_q else 0
    pct_c = 100 * criteria_with_topic / total_criteria if total_criteria else 0
    print(f"Step {step}: {q_with_specific_best}/{total_q} ({pct_q:.0f}%) questions have topic words in best rubric | "
          f"{criteria_with_topic}/{total_criteria} ({pct_c:.0f}%) criteria mention topic words")
