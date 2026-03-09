#!/usr/bin/env python3
"""Generate training metrics plots for the RL report.

Produces a multi-panel figure with:
  1. Entropy over steps
  2. Score (mean) over steps
  3. KL divergence over steps
  4. Grad norm over steps
  5. Response length (token-level, from log) over steps
  6. Pg clipfrac over steps
  7. "Respond with" template percentage (from rollout logs)
  8. Programmatic generic flag percentage (from training log)

Usage:
    python3 scripts/plot_training_metrics.py
    # Outputs: docs/training_metrics.png
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "out/rl_v3.log"
ROLLOUT_DIR = "out/rl/rollout_logs"
OUTPUT = "docs/training_metrics.png"

# ── 1. Parse training log metrics ──────────────────────────────────────

steps, entropies, scores, kls, grads, resplens, clips = [], [], [], [], [], [], []

with open(LOG_FILE) as f:
    for line in f:
        if "step:" in line and "actor/entropy:" in line:
            s = re.search(r"step:(\d+)", line)
            e = re.search(r"actor/entropy:([0-9.]+)", line)
            sc = re.search(r"critic/score/mean:([0-9.-]+)", line)
            k = re.search(r"actor/kl_loss:([0-9.]+)", line)
            g = re.search(r"actor/grad_norm:([0-9.]+)", line)
            rl = re.search(r"response_length/mean:([0-9.]+)", line)
            c = re.search(r"actor/pg_clipfrac:([0-9.]+)", line)
            if s and e:
                steps.append(int(s.group(1)))
                entropies.append(float(e.group(1)))
                scores.append(float(sc.group(1)) if sc else 0)
                kls.append(float(k.group(1)) if k else 0)
                grads.append(float(g.group(1)) if g else 0)
                resplens.append(float(rl.group(1)) if rl else 0)
                clips.append(float(c.group(1)) if c else 0)

# ── 2. Parse rollout logs for template / generic percentages ───────────

respond_with_re = re.compile(
    r"(?i)respond\s+with\s+(empathy|compassion|clarity|accurate|appropriate|understanding|concern)"
)
generic_word_re = re.compile(
    r"(?i)(?:empathy|compassion|supportive tone|actionable advice|clarity|accuracy)"
)

rollout_steps, respond_with_pcts, generic_pcts = [], [], []
rubric_char_lens_mean, rubric_char_lens_min, rubric_char_lens_max = [], [], []
n_criteria_means = []

if os.path.isdir(ROLLOUT_DIR):
    for fname in sorted(os.listdir(ROLLOUT_DIR)):
        step_num = int(fname.split("_")[1].split(".")[0])
        fpath = os.path.join(ROLLOUT_DIR, fname)
        with open(fpath) as fh:
            items = [json.loads(l) for l in fh]

        total = 0
        rw = 0
        gen = 0
        char_lens = []
        crit_counts = []
        for item in items:
            for r in item.get("rollouts", []):
                rubric = r.get("rubric", "")
                total += 1
                if respond_with_re.search(rubric):
                    rw += 1
                if generic_word_re.search(rubric):
                    gen += 1
                char_lens.append(len(rubric))
                crit_counts.append(rubric.count("- ["))

        if total > 0:
            rollout_steps.append(step_num)
            respond_with_pcts.append(100 * rw / total)
            generic_pcts.append(100 * gen / total)
            rubric_char_lens_mean.append(np.mean(char_lens))
            rubric_char_lens_min.append(np.min(char_lens))
            rubric_char_lens_max.append(np.max(char_lens))
            n_criteria_means.append(np.mean(crit_counts))

# ── 3. Parse programmatic generic flag rates from training log ─────────

prog_generic_steps = []
prog_generic_pcts = []

with open(LOG_FILE) as f:
    lines = f.readlines()

current_step = None
step_generic = defaultdict(lambda: [0, 0])

for line in lines:
    sm = re.search(r"step:(\d+)", line)
    if sm and "actor/entropy" in line:
        current_step = int(sm.group(1))

    mg = re.search(r"Q\d+:\s+(\d+)/(\d+)\s+rubrics flagged generic", line)
    if mg and current_step:
        step_generic[current_step][0] += int(mg.group(1))
        step_generic[current_step][1] += int(mg.group(2))

for s in sorted(step_generic.keys()):
    flagged, total = step_generic[s]
    if total > 0:
        prog_generic_steps.append(s)
        prog_generic_pcts.append(100 * flagged / total)

# ── 4. Plot ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle("GRM RL Training Metrics (Steps 361–540)", fontsize=16, y=0.98)

# 4.1 Entropy
ax = axes[0, 0]
ax.plot(steps, entropies, "b-", alpha=0.6, linewidth=0.8)
# Smoothed
if len(entropies) > 10:
    window = 10
    smoothed = np.convolve(entropies, np.ones(window)/window, mode="valid")
    ax.plot(steps[window-1:], smoothed, "b-", linewidth=2, label=f"MA-{window}")
ax.set_ylabel("Entropy")
ax.set_title("Actor Entropy")
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="ent=1.0")
ax.legend()
ax.grid(True, alpha=0.3)

# 4.2 Score
ax = axes[0, 1]
ax.plot(steps, scores, "g-", alpha=0.6, linewidth=0.8)
if len(scores) > 10:
    smoothed = np.convolve(scores, np.ones(10)/10, mode="valid")
    ax.plot(steps[9:], smoothed, "g-", linewidth=2, label="MA-10")
ax.set_ylabel("Score")
ax.set_title("Critic Score (mean)")
ax.legend()
ax.grid(True, alpha=0.3)

# 4.3 KL
ax = axes[1, 0]
ax.plot(steps, kls, "r-", alpha=0.6, linewidth=0.8)
if len(kls) > 10:
    smoothed = np.convolve(kls, np.ones(10)/10, mode="valid")
    ax.plot(steps[9:], smoothed, "r-", linewidth=2, label="MA-10")
ax.set_ylabel("KL Loss")
ax.set_title("KL Divergence")
ax.legend()
ax.grid(True, alpha=0.3)

# 4.4 Grad Norm
ax = axes[1, 1]
ax.plot(steps, grads, "m-", alpha=0.6, linewidth=0.8)
if len(grads) > 10:
    smoothed = np.convolve(grads, np.ones(10)/10, mode="valid")
    ax.plot(steps[9:], smoothed, "m-", linewidth=2, label="MA-10")
ax.set_ylabel("Grad Norm")
ax.set_title("Gradient Norm")
ax.legend()
ax.grid(True, alpha=0.3)

# 4.5 "Respond with" template %
ax = axes[2, 0]
if rollout_steps:
    ax.plot(rollout_steps, respond_with_pcts, "r-o", markersize=2, alpha=0.7, label="'Respond with' template %")
    ax.fill_between(rollout_steps, respond_with_pcts, alpha=0.2, color="red")
ax.set_ylabel("Percentage (%)")
ax.set_title("'Respond with empathy...' Template %")
ax.set_ylim(-5, 105)
ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# 4.6 Programmatic Generic Flag %
ax = axes[2, 1]
if prog_generic_steps:
    ax.plot(prog_generic_steps, prog_generic_pcts, "orange", alpha=0.6, linewidth=0.8)
    if len(prog_generic_pcts) > 10:
        smoothed = np.convolve(prog_generic_pcts, np.ones(10)/10, mode="valid")
        ax.plot(prog_generic_steps[9:], smoothed, "orange", linewidth=2, label="MA-10")
ax.set_ylabel("Percentage (%)")
ax.set_title("Programmatic Generic Flag %")
ax.set_ylim(-5, 105)
ax.legend()
ax.grid(True, alpha=0.3)

# 4.7 Response Length (token)
ax = axes[3, 0]
ax.plot(steps, resplens, "c-", alpha=0.6, linewidth=0.8)
if len(resplens) > 10:
    smoothed = np.convolve(resplens, np.ones(10)/10, mode="valid")
    ax.plot(steps[9:], smoothed, "c-", linewidth=2, label="MA-10")
ax.set_ylabel("Response Length (tokens)")
ax.set_title("Mean Response Length")
ax.legend()
ax.grid(True, alpha=0.3)

# 4.8 Pg Clipfrac
ax = axes[3, 1]
ax.plot(steps, clips, "k-", alpha=0.6, linewidth=0.8)
if len(clips) > 10:
    smoothed = np.convolve(clips, np.ones(10)/10, mode="valid")
    ax.plot(steps[9:], smoothed, "k-", linewidth=2, label="MA-10")
ax.set_ylabel("Clip Fraction")
ax.set_title("PG Clip Fraction")
ax.legend()
ax.grid(True, alpha=0.3)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel("Step")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUTPUT}")
