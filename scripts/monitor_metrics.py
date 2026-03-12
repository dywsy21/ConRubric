#!/usr/bin/env python3
"""Parse RL training worker logs and display vital metrics per step."""
import re
import sys

def main():
    logfile = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ray/session_latest/logs/worker-*.out"
    
    # If glob pattern, find the right file
    if "*" in logfile:
        import glob
        candidates = glob.glob(logfile)
        # Find the one with MetaReward
        for c in candidates:
            with open(c) as f:
                head = f.read(10000)
                if "MetaReward" in head:
                    logfile = c
                    break
    
    with open(logfile) as f:
        lines = f.read()
    
    step_log = re.compile(
        r"step:(\d+).*?actor/entropy:([0-9.e+-]+).*?"
        r"actor/ppo_kl:([0-9.e+-]+).*?"
        r"actor/grad_norm:([0-9.e+-]+).*?"
        r"response_length/mean:([0-9.e+-]+).*?"
        r"response_length/clip_ratio:([0-9.e+-]+)"
    )
    reward_re = re.compile(r"\[MetaReward\] All rewards computed, mean=([0-9.]+)")
    quality_re = re.compile(
        r"quality_adj_mean=([0-9.e+-]+), "
        r"length_penalty_mean=([0-9.e+-]+), "
        r"token_len_penalty_mean=([0-9.e+-]+)"
    )
    step_pattern = re.compile(r"\[RubricSample\] Step (\d+)")
    uniq_pattern = re.compile(r"unique_criteria=\[([^\]]+)\]")
    dup_re = re.compile(r"dup penalty applied, eff_weight=([0-9.]+), (\d+)/(\d+) dup")
    
    # Parse rewards
    rewards = [float(m.group(1)) for m in reward_re.finditer(lines)]
    qualities = [
        (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        for m in quality_re.finditer(lines)
    ]
    
    # Dedup stats per step
    current_step = None
    step_data = {}
    for line in lines.split("\n"):
        m = step_pattern.search(line)
        if m:
            current_step = int(m.group(1))
            if current_step not in step_data:
                step_data[current_step] = []
        m2 = uniq_pattern.search(line)
        if m2 and current_step is not None:
            vals = [int(x.strip().strip("'")) for x in m2.group(1).split(",")]
            step_data[current_step].extend(vals)
    
    # Dup weights per step - group by 64 rollouts
    all_dup_weights = [float(m.group(1)) for m in dup_re.finditer(lines)]
    dup_step_weights = {}
    
    # Parse step metrics
    step_metrics = {}
    for m in step_log.finditer(lines):
        s = int(m.group(1))
        step_metrics[s] = {
            "ent": float(m.group(2)),
            "kl": float(m.group(3)),
            "gn": float(m.group(4)),
            "rl": float(m.group(5)),
            "cr": float(m.group(6)),
        }
    
    # Assign dup weights to steps (64 per step)
    sorted_steps = sorted(step_metrics.keys())
    idx = 0
    for step in sorted_steps:
        chunk = all_dup_weights[idx : idx + 64]
        if chunk:
            dup_step_weights[step] = chunk
            idx += 64
    
    # Print header
    print("=== METRICS PER STEP ===")
    hdr = (
        f"{'Step':>5} | {'Entropy':>8} | {'KL':>10} | {'GradNorm':>9} | "
        f"{'RespLen':>8} | {'ClipR':>6} | {'Reward':>7} | {'QualAdj':>8} | "
        f"{'TokLenP':>8} | {'AvgUniq':>8} | {'Collaps':>8} | {'DupWt':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    
    for i, step in enumerate(sorted_steps):
        sm = step_metrics[step]
        rw = rewards[i] if i < len(rewards) else 0
        qa = qualities[i] if i < len(qualities) else (0, 0, 0)
        vals = step_data.get(step, [])
        avg_u = sum(vals) / len(vals) if vals else 0
        collapse = sum(1 for v in vals if v <= 2) / len(vals) * 100 if vals else 0
        dw = dup_step_weights.get(step, [])
        avg_dw = sum(dw) / len(dw) if dw else 1.0
        print(
            f"{step:5d} | {sm['ent']:8.3f} | {sm['kl']:10.6f} | "
            f"{sm['gn']:9.3f} | {sm['rl']:8.0f} | {sm['cr']:5.1%} | "
            f"{rw:7.3f} | {qa[0]:8.3f} | {qa[2]:8.3f} | "
            f"{avg_u:8.1f} | {collapse:6.0f}%  | {avg_dw:6.3f}"
        )


if __name__ == "__main__":
    main()
