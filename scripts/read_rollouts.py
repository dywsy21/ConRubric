#!/usr/bin/env python3
"""Read and display rollout content for manual quality review."""
import json
import sys

step = int(sys.argv[1]) if len(sys.argv) > 1 else 542
n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 5

path = f"out/rl/rollout_logs/step_{step:04d}.jsonl"
with open(path) as f:
    questions = [json.loads(l) for l in f]

print(f"=== STEP {step}: {len(questions)} questions, each has rollouts ===\n")

for qi in range(min(n_questions, len(questions))):
    q = questions[qi]
    print("=" * 80)
    print(f"QUESTION {qi+1}: {q['question'][:300]}")
    print("=" * 80)
    for ri, r in enumerate(q["rollouts"]):
        rubric_len = len(r["rubric"])
        truncated = rubric_len > 2900
        tag = " [TRUNCATED]" if truncated else ""
        print(f"\n  --- Rollout {ri+1}/{q['n_rollouts']} | reward={r['reward']:.2f}"
              f" | cons={r['consensus']:.1f} | disc={r['disc']:.2f}"
              f" | gold_disc={r['gold_disc']:.2f} | cal={r['calibration']:.2f}"
              f" | #crit={r['n_criteria_total']} | len={rubric_len}{tag} ---")
        if not truncated:
            print(r["rubric"])
        else:
            print("  [Skipping truncated rollout]")
    print()
