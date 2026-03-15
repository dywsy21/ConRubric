#!/usr/bin/env python3
"""Plot RL training metrics from rollout logs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rl_data.csv'
out_dir = sys.argv[2] if len(sys.argv) > 2 else 'out'

steps, rewards, reward_mins, reward_maxs = [], [], [], []
n_crits, n_uniqs, unique_ratios = [], [], []

with open(csv_path) as f:
    for row in csv.reader(f):
        steps.append(int(row[0]))
        rewards.append(float(row[1]))
        reward_mins.append(float(row[2]))
        reward_maxs.append(float(row[3]))
        n_crits.append(float(row[4]))
        n_uniqs.append(float(row[5]))
        unique_ratios.append(float(row[6]))

steps = np.array(steps)
rewards = np.array(rewards)
unique_ratios = np.array(unique_ratios)
n_crits = np.array(n_crits)
n_uniqs = np.array(n_uniqs)

def smooth(arr, w=10):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='valid')

sw = 10
s_steps = steps[sw-1:]
n_steps = len(steps)

# === Plot 1: 3-panel overview ===
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'GRM RL Training Metrics (Qwen3-8B, Steps 1-{n_steps})', fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.fill_between(steps, reward_mins, reward_maxs, alpha=0.15, color='blue', label='Min-Max range')
ax1.plot(steps, rewards, 'b-', alpha=0.3, linewidth=0.5)
ax1.plot(s_steps, smooth(rewards, sw), 'b-', linewidth=2, label=f'Mean reward (MA-{sw})')
ax1.set_ylabel('Reward', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_title('Consensus Reward')

ax2 = axes[1]
ax2.plot(steps, unique_ratios, 'g-', alpha=0.3, linewidth=0.5)
ax2.plot(s_steps, smooth(unique_ratios, sw), 'g-', linewidth=2, label=f'Unique ratio (MA-{sw})')
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
ax2.set_ylabel('Unique Ratio', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_title('Rubric Diversity (Spearman Unique Ratio)')
ax2.set_ylim(0.3, 0.85)

ax3 = axes[2]
ax3.plot(steps, n_crits, 'r-', alpha=0.3, linewidth=0.5)
ax3.plot(s_steps, smooth(n_crits, sw), 'r-', linewidth=2, label=f'Total criteria (MA-{sw})')
ax3.plot(steps, n_uniqs, 'orange', alpha=0.3, linewidth=0.5)
ax3.plot(s_steps, smooth(n_uniqs, sw), 'orange', linewidth=2, label=f'Unique criteria (MA-{sw})')
ax3.set_ylabel('Count', fontsize=12)
ax3.set_xlabel('Training Step', fontsize=12)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_title('Criteria Counts')

plt.tight_layout()
path1 = f'{out_dir}/rl_training_metrics.png'
plt.savefig(path1, dpi=150, bbox_inches='tight')
print(f'Plot 1 saved: {path1}')

# === Plot 2: Reward vs Unique Ratio dual-axis ===
fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
color1 = 'tab:blue'
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Reward (mean)', color=color1, fontsize=12)
ax.plot(s_steps, smooth(rewards, sw), color=color1, linewidth=2, label='Reward')
ax.tick_params(axis='y', labelcolor=color1)

ax2r = ax.twinx()
color2 = 'tab:green'
ax2r.set_ylabel('Unique Ratio', color=color2, fontsize=12)
ax2r.plot(s_steps, smooth(unique_ratios, sw), color=color2, linewidth=2, label='Unique Ratio')
ax2r.tick_params(axis='y', labelcolor=color2)
ax2r.set_ylim(0.35, 0.75)

fig2.suptitle(f'Reward vs Unique Ratio (MA-{sw})', fontsize=14, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
path2 = f'{out_dir}/rl_reward_vs_diversity.png'
plt.savefig(path2, dpi=150, bbox_inches='tight')
print(f'Plot 2 saved: {path2}')
