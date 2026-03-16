"""Analyze rubric quality trends across training steps.

Looks for signs of quality collapse:
- Criteria getting excessively long (parroting/bloating)
- Criteria count changes
- Token-level bloat patterns

Usage: python scripts/analyze_collapse_point.py
"""

import json
import re
import os


def analyze_step(step):
    fname = f"out/rl/rollout_logs/step_{step:04d}.jsonl"
    try:
        with open(fname) as f:
            lines = [json.loads(l) for l in f]
    except FileNotFoundError:
        return None

    total_criteria = 0
    long_criteria = 0  # criteria with > 200 chars
    very_long_criteria = 0  # > 500 chars
    max_crit_len = 0
    all_crit_lens = []
    avg_crit_count = []
    rubric_lens = []

    for record in lines:
        for rollout in record["rollouts"]:
            rubric = rollout["rubric"]
            rubric_lens.append(len(rubric))
            n_crit = rollout.get("n_criteria_total", 0)
            avg_crit_count.append(n_crit)
            for line in rubric.split("\n"):
                line = line.strip()
                if line.startswith("[+]") or line.startswith("[-]"):
                    total_criteria += 1
                    clen = len(line)
                    all_crit_lens.append(clen)
                    if clen > 200:
                        long_criteria += 1
                    if clen > 500:
                        very_long_criteria += 1
                    max_crit_len = max(max_crit_len, clen)

    n_rollouts = sum(len(r["rollouts"]) for r in lines)
    avg_n = sum(avg_crit_count) / len(avg_crit_count) if avg_crit_count else 0
    pct_long = (long_criteria / total_criteria * 100) if total_criteria > 0 else 0
    pct_vlong = (very_long_criteria / total_criteria * 100) if total_criteria > 0 else 0
    med_crit_len = sorted(all_crit_lens)[len(all_crit_lens) // 2] if all_crit_lens else 0
    avg_rubric_len = sum(rubric_lens) / len(rubric_lens) if rubric_lens else 0

    return {
        "step": step,
        "n_rollouts": n_rollouts,
        "total_criteria": total_criteria,
        "avg_criteria": avg_n,
        "med_crit_len": med_crit_len,
        "max_crit_len": max_crit_len,
        "pct_long": pct_long,
        "pct_vlong": pct_vlong,
        "avg_rubric_len": avg_rubric_len,
    }


def show_sample_rubric(step, q_idx=0, r_idx=0):
    """Show a sample rubric from a specific step."""
    fname = f"out/rl/rollout_logs/step_{step:04d}.jsonl"
    try:
        with open(fname) as f:
            lines = [json.loads(l) for l in f]
    except FileNotFoundError:
        return

    if q_idx < len(lines) and r_idx < len(lines[q_idx]["rollouts"]):
        record = lines[q_idx]
        rollout = record["rollouts"][r_idx]
        q = record["question"]
        rubric = rollout["rubric"]
        print(f"\n{'='*70}")
        print(f"STEP {step} — Q{q_idx} R{r_idx}")
        print(f"Question: {q[:100]}")
        print(f"{'─'*70}")
        # Show first 600 chars
        display = rubric[:600]
        if len(rubric) > 600:
            display += f"\n... ({len(rubric)} chars total)"
        print(display)


def main():
    header = (
        f"{'Step':>5} | {'Roll':>4} | {'Crit':>5} | {'Avg#':>4} | "
        f"{'MedLen':>6} | {'MaxLen':>6} | {'%>200c':>6} | {'%>500c':>6} | {'AvgRubLen':>9}"
    )
    print(header)
    print("-" * len(header))

    steps = list(range(1, 21)) + list(range(20, 501, 20))
    steps = sorted(set(steps))

    results = []
    for step in steps:
        r = analyze_step(step)
        if r:
            results.append(r)
            print(
                f"{r['step']:>5} | {r['n_rollouts']:>4} | {r['total_criteria']:>5} | "
                f"{r['avg_criteria']:>4.1f} | {r['med_crit_len']:>6} | {r['max_crit_len']:>6} | "
                f"{r['pct_long']:>5.1f}% | {r['pct_vlong']:>5.1f}% | {r['avg_rubric_len']:>9.0f}"
            )

    # Find collapse point: where %>500c first exceeds 30%
    print("\n\nCollapse Analysis:")
    print("-" * 50)
    prev_r = None
    for r in results:
        if r["pct_vlong"] > 30 and (prev_r is None or prev_r["pct_vlong"] <= 30):
            print(f"  %Very-long criteria (>500c) first exceeds 30% at step {r['step']}")
            if prev_r:
                print(f"  Previous step {prev_r['step']}: {prev_r['pct_vlong']:.1f}%")
        if r["med_crit_len"] > 300 and (prev_r is None or prev_r["med_crit_len"] <= 300):
            print(f"  Median criterion length first exceeds 300 at step {r['step']}")
            if prev_r:
                print(f"  Previous step {prev_r['step']}: {prev_r['med_crit_len']}")
        prev_r = r

    # Show sample rubrics at key steps
    print("\n\nSample Rubrics at Key Steps:")
    # Early, mid, late
    sample_steps = [1, 20, 100, 200, 300, 380]
    for s in sample_steps:
        if os.path.exists(f"out/rl/rollout_logs/step_{s:04d}.jsonl"):
            show_sample_rubric(s)


if __name__ == "__main__":
    main()
