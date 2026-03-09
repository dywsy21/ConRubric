#!/usr/bin/env python3
"""Comprehensive RL training monitor - metrics + generic % + rubric samples."""
import json, re, os, sys

LOG_FILE = "out/rl_v3.log"
ROLLOUT_DIR = "out/rl/rollout_logs"

empathy_re = re.compile(r"(?i)models?\s+answers?\s+with")
generic_re = re.compile(
    r"(?i)(?:empathy|compassion|supportive tone|actionable advice|clarity|accuracy)"
)

# ── 1. Extract metrics from training log ──
print("=" * 70)
print("TRAINING METRICS (last 5 steps)")
print("=" * 70)

step_lines = []
with open(LOG_FILE) as f:
    for line in f:
        if "step:" in line and "actor/entropy:" in line:
            step_lines.append(line)

for line in step_lines[-5:]:
    step = re.search(r"step:(\d+)", line)
    entropy = re.search(r"actor/entropy:([0-9.]+)", line)
    clipfrac = re.search(r"actor/pg_clipfrac:([0-9.]+)", line)
    grad = re.search(r"actor/grad_norm:([0-9.]+)", line)
    score = re.search(r"critic/score/mean:([0-9.-]+)", line)
    resplen = re.search(r"response_length/mean:([0-9.]+)", line)
    kl = re.search(r"actor/kl_loss:([0-9.]+)", line)
    ent_coeff = re.search(r"entropy_coeff_effective:([0-9.-]+)", line)
    step_time = re.search(r"timing_s/step:([0-9.]+)", line)

    s = step.group(1) if step else "?"
    e = float(entropy.group(1)) if entropy else 0
    c = float(clipfrac.group(1)) if clipfrac else 0
    g = float(grad.group(1)) if grad else 0
    sc = float(score.group(1)) if score else 0
    rl = float(resplen.group(1)) if resplen else 0
    k = float(kl.group(1)) if kl else 0
    ec = float(ent_coeff.group(1)) if ent_coeff else 0
    st = float(step_time.group(1)) if step_time else 0

    print(f"  step {s:>4s} | entropy={e:6.3f} | clip={c:.4f} | grad={g:6.3f} | "
          f"score={sc:5.2f} | resp_len={rl:6.1f} | kl={k:5.3f} | ent_c={ec:7.4f} | {st:.0f}s")

# ── 2. Analyze latest rollout logs ──
print()
print("=" * 70)
print("ROLLOUT LOG ANALYSIS")
print("=" * 70)

rollout_files = sorted(os.listdir(ROLLOUT_DIR))
if not rollout_files:
    print("No rollout logs yet")
    sys.exit(0)

# Get last 3 steps
recent = rollout_files[-3:]
for fname in recent:
    step_num = int(fname.split("_")[1].split(".")[0])
    fpath = os.path.join(ROLLOUT_DIR, fname)
    with open(fpath) as f:
        items = [json.loads(l) for l in f]

    total = 0
    emp = 0
    gen = 0
    for item in items:
        for r in item.get("rollouts", []):
            rubric = r.get("rubric", "")
            total += 1
            if empathy_re.search(rubric):
                emp += 1
            if generic_re.search(rubric):
                gen += 1

    clean = total - max(emp, gen)
    print(f"  step {step_num}: empathy={emp}/{total} ({100*emp/total:.0f}%) | "
          f"generic_words={gen}/{total} ({100*gen/total:.0f}%) | "
          f"clean={clean}/{total} ({100*clean/total:.0f}%)")

# ── 3. Sample rubrics from latest step ──
latest_step = int(rollout_files[-1].split("_")[1].split(".")[0])
fpath = os.path.join(ROLLOUT_DIR, rollout_files[-1])
with open(fpath) as f:
    items = [json.loads(l) for l in f]

print(f"\n--- SAMPLE RUBRICS (step {latest_step}, best per question) ---")
for qi, item in enumerate(items[:3]):
    q = item["question"]
    rollouts = sorted(item.get("rollouts", []), key=lambda r: r.get("reward", 0), reverse=True)
    if not rollouts:
        continue
    best = rollouts[0]
    rubric = best.get("rubric", "")
    reward = best.get("reward", 0)
    criteria = [l.strip() for l in rubric.split("\n") if l.strip().startswith("- [")]

    flags = []
    if empathy_re.search(rubric):
        flags.append("EMPATHY")
    if generic_re.search(rubric):
        flags.append("GENERIC")
    flag_str = " [" + ",".join(flags) + "]" if flags else ""

    print(f"\n  Q{qi}: {q[:80]}")
    print(f"  Best reward={reward:.2f}{flag_str}")
    for c in criteria[:4]:
        print(f"    {c[:100]}")
    if len(criteria) > 4:
        print(f"    ... ({len(criteria)} total criteria)")

# ── 4. Generic flagging stats from meta_reward logs ──
print(f"\n--- META-REWARD GENERIC FLAGS (from training log) ---")
generic_flags = 0
total_questions = 0
with open(LOG_FILE) as f:
    for line in f:
        if "rubrics flagged generic" in line:
            m = re.search(r"(\d+)/(\d+) rubrics flagged generic", line)
            if m:
                generic_flags += int(m.group(1))
                total_questions += int(m.group(2))

if total_questions > 0:
    print(f"  Programmatic generic flags: {generic_flags}/{total_questions} "
          f"({100*generic_flags/total_questions:.1f}%)")
else:
    print("  No generic flag data yet")

# ETA
if step_lines:
    last_step = int(re.search(r"step:(\d+)", step_lines[-1]).group(1))
    remaining = 1000 - last_step
    avg_time = 0
    if len(step_lines) >= 2:
        times = []
        for l in step_lines[-5:]:
            t = re.search(r"timing_s/step:([0-9.]+)", l)
            if t:
                times.append(float(t.group(1)))
        if times:
            avg_time = sum(times) / len(times)
    eta_h = remaining * avg_time / 3600 if avg_time > 0 else 0
    print(f"\n  Current: step {last_step}/1000 | ~{avg_time:.0f}s/step | ETA: {eta_h:.1f}h")
