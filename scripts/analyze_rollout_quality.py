#!/usr/bin/env python3
"""Analyze rubric quality from rollout logs across RL training steps.

Reads rollout JSONL files and computes per-step quality metrics:
  - Template collapse rate (first-criterion clustering)
  - Non-ASCII character prevalence
  - Extreme point values (|pts| > 10)
  - Gibberish detection
  - Criterion count distribution
  - Mean reward
  - Rubric diversity (unique first-criterion openings)

Usage:
    python scripts/analyze_rollout_quality.py [--steps 100,200,300]
    python scripts/analyze_rollout_quality.py --all
    python scripts/analyze_rollout_quality.py --range 800,960
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROLLOUT_DIR = os.environ.get("GRM_ROLLOUT_LOG_DIR", "out/rl/rollout_logs")

CRIT_RE = re.compile(
    r"^\s*[-*]\s*\[([+-]?\d+)\]\s*(.+?)(?:\s*\|\s*tags?\s*:\s*(.*))?$",
    re.IGNORECASE,
)

_SIM_STOP = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'that', 'this', 'it', 'its', 'not', 'no', 'as', 'if', 'so',
    'such', 'very', 'too', 'just', 'also', 'more', 'most', 'some',
    'any', 'all', 'each', 'every', 'both', 'may', 'might', 'can',
    'should', 'would', 'could', 'will', 'shall', 'about', 'into',
    'than', 'then', 'when', 'where', 'how', 'what', 'which', 'who',
    'especially', 'since', 'user', 'using', 'used', 'ensure',
    'provide', 'offer', 'include', 'make', 'sure', 'given',
    'respond', 'with',
})

NON_ASCII_RE = re.compile(r'[^\x00-\x7f]')

# Filler pattern from rubric_quality.py
FILLER_RE = re.compile(
    r"(?i)^models?\s+answers?\s+with\b"
    r"|^(?:the\s+)?(?:response|answer|model)\s+(?:shows?|demonstrates?|maintains?|provides?)\s+"
    r"(?:empathy|compassion|supportive|appropriate|sensitivity)"
    r"|^respond\s+with\s+(?:empathy|compassion|understanding|sensitivity|kindness|sympathy)"
    r"|^(?:show|display|express|convey)\s+(?:empathy|compassion|understanding|sensitivity|kindness)"
    r"|^acknowledge\s+(?:the\s+)?(?:user|patient|person|individual)(?:'?s?)?\s+"
    r"(?:feeling|emotion|situation|concern|struggle|experience)",
)


def parse_criteria(rubric_text: str) -> List[Tuple[int, str, Optional[str]]]:
    """Parse rubric into list of (points, text, tags_str)."""
    crits = []
    for line in rubric_text.splitlines():
        m = CRIT_RE.match(line)
        if m:
            crits.append((int(m.group(1)), m.group(2).strip(), m.group(3)))
    return crits


def first_crit_words(rubric_text: str) -> frozenset:
    """Extract content words from first criterion."""
    for line in rubric_text.splitlines():
        m = CRIT_RE.match(line)
        if m:
            ws = [w for w in re.findall(r'[a-z]{4,}', m.group(2).lower())
                  if w not in _SIM_STOP]
            return frozenset(ws)
    return frozenset()


def analyze_step(step: int, rollout_dir: str = ROLLOUT_DIR) -> Optional[Dict]:
    """Analyze a single step's rollout log."""
    fname = Path(rollout_dir) / f"step_{step:04d}.jsonl"
    if not fname.exists():
        return None

    with open(fname, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    all_rewards = []
    all_crit_counts = []
    n_extreme = 0
    n_non_ascii = 0
    n_filler = 0
    n_garbage = 0  # rubrics with very few criteria or gibberish
    n_total = 0
    point_values = []
    first_openings = Counter()  # first 8 words of first criterion

    # Per-question collapse analysis
    question_collapse_rates = []

    for item in items:
        question = item.get("question", "")
        rollouts = item.get("rollouts", [])
        
        q_first_words = []  # first criterion content words per rubric
        
        for r in rollouts:
            rubric = r.get("rubric", "")
            reward = r.get("reward", 0)
            n_total += 1
            all_rewards.append(reward)

            crits = parse_criteria(rubric)
            all_crit_counts.append(len(crits))

            # Point analysis
            for pts, text, tags in crits:
                point_values.append(pts)
                if abs(pts) > 10:
                    n_extreme += 1
                if FILLER_RE.search(text):
                    n_filler += 1

            # Non-ASCII
            if NON_ASCII_RE.search(rubric):
                n_non_ascii += 1

            # Garbage: <3 criteria or very short
            if len(crits) < 2 or len(rubric.strip()) < 50:
                n_garbage += 1

            # First criterion opening
            fw = first_crit_words(rubric)
            q_first_words.append(fw)
            
            # Text of first criterion (first 8 words for opener tracking)
            if crits:
                opener = " ".join(crits[0][1].lower().split()[:8])
                first_openings[opener] += 1

        # Collapse rate for this question
        if len(q_first_words) >= 2:
            clusters: List[set] = []
            for i, fw_i in enumerate(q_first_words):
                placed = False
                for cluster in clusters:
                    rep = next(iter(cluster))
                    fa, fb = fw_i, q_first_words[rep]
                    if fa and fb:
                        j_sim = len(fa & fb) / max(len(fa | fb), 1)
                        if j_sim >= 0.4:
                            cluster.add(i)
                            placed = True
                            break
                if not placed:
                    clusters.append({i})
            
            largest = max(clusters, key=len)
            collapse_rate = len(largest) / len(q_first_words)
            question_collapse_rates.append(collapse_rate)

    if n_total == 0:
        return None

    mean_reward = sum(all_rewards) / len(all_rewards)
    mean_crits = sum(all_crit_counts) / len(all_crit_counts)
    mean_collapse = sum(question_collapse_rates) / len(question_collapse_rates) if question_collapse_rates else 0
    
    # Point value distribution
    pt_counter = Counter(point_values)
    top_pts = pt_counter.most_common(5)

    # Top opening patterns
    top_openings = first_openings.most_common(3)

    return {
        "step": step,
        "n_rubrics": n_total,
        "mean_reward": mean_reward,
        "mean_criteria": mean_crits,
        "collapse_rate": mean_collapse,
        "n_collapsed_questions": sum(1 for r in question_collapse_rates if r > 0.6),
        "n_questions": len(question_collapse_rates),
        "pct_extreme_pts": n_extreme / max(sum(all_crit_counts), 1) * 100,
        "pct_non_ascii": n_non_ascii / n_total * 100,
        "pct_filler": n_filler / max(sum(all_crit_counts), 1) * 100,
        "pct_garbage": n_garbage / n_total * 100,
        "top_point_values": top_pts,
        "top_openings": top_openings,
        "point_range": (min(point_values) if point_values else 0, 
                       max(point_values) if point_values else 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout quality across RL training steps")
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated steps to analyze")
    parser.add_argument("--range", type=str, default=None, help="Start,end range to analyze (every available step)")
    parser.add_argument("--all", action="store_true", help="Analyze all available steps")
    parser.add_argument("--rollout-dir", type=str, default=ROLLOUT_DIR, help="Rollout log directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    rollout_dir = args.rollout_dir

    # Discover available steps
    available = []
    if os.path.isdir(rollout_dir):
        for f in sorted(os.listdir(rollout_dir)):
            m = re.match(r"step_(\d+)\.jsonl", f)
            if m:
                available.append(int(m.group(1)))

    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]
    elif args.range:
        parts = args.range.split(",")
        start, end = int(parts[0]), int(parts[1])
        steps = [s for s in available if start <= s <= end]
    elif args.all:
        steps = available
    else:
        # Default: sample every 100 steps + every 20 after 800
        steps = [s for s in available if s % 100 == 0 or (s >= 800 and s % 20 == 0)]
        if not steps:
            steps = available[-10:] if len(available) > 10 else available

    if not steps:
        print("No steps to analyze. Use --all, --steps, or --range.")
        sys.exit(1)

    results = []
    for step in sorted(steps):
        r = analyze_step(step, rollout_dir)
        if r:
            results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Pretty print table
    print(f"\n{'Step':>5} | {'Reward':>7} | {'Crits':>5} | {'Collapse':>8} | "
          f"{'XPts%':>5} | {'NonASC%':>7} | {'Fill%':>5} | {'Garb%':>5} | "
          f"{'PtRange':>10} | Top Opening")
    print("-" * 120)

    for r in results:
        top_opening = r["top_openings"][0][0][:40] if r["top_openings"] else "—"
        top_count = r["top_openings"][0][1] if r["top_openings"] else 0
        print(f"{r['step']:>5} | {r['mean_reward']:>7.2f} | {r['mean_criteria']:>5.1f} | "
              f"{r['collapse_rate']:>7.1%} | {r['pct_extreme_pts']:>5.1f} | "
              f"{r['pct_non_ascii']:>7.1f} | {r['pct_filler']:>5.1f} | "
              f"{r['pct_garbage']:>5.1f} | {r['point_range'][0]:>4},{r['point_range'][1]:>4} | "
              f"{top_opening} ({top_count}x)")

    # Find best step by composite quality score
    print("\n=== Quality Ranking (lower penalty = better) ===")
    scored = []
    for r in results:
        # Composite: reward good, low collapse, low extreme, low non-ascii, low filler, low garbage
        quality = (
            r["mean_reward"] * 1.0         # higher is better
            - r["collapse_rate"] * 5.0     # penalty for collapse
            - r["pct_extreme_pts"] * 0.1   # penalty for extreme points
            - r["pct_non_ascii"] * 0.05    # penalty for non-ascii
            - r["pct_filler"] * 0.03       # penalty for filler
            - r["pct_garbage"] * 0.1       # penalty for garbage
        )
        scored.append((quality, r["step"], r))

    scored.sort(reverse=True)
    # Show all when explicitly selecting steps, otherwise top 15
    show_n = len(scored) if (args.steps or len(scored) <= 20) else 15
    print(f"\n{'Rank':>4} | {'Step':>5} | {'Score':>7} | {'Reward':>7} | {'Collapse':>8} | {'XPts%':>5} | {'NonASC%':>7} | {'Fill%':>5} | {'Garb%':>5}")
    print("-" * 100)
    for rank, (score, step, r) in enumerate(scored[:show_n], 1):
        print(f"{rank:>4} | {step:>5} | {score:>7.2f} | {r['mean_reward']:>7.2f} | "
              f"{r['collapse_rate']:>7.1%} | {r['pct_extreme_pts']:>5.1f} | "
              f"{r['pct_non_ascii']:>7.1f} | {r['pct_filler']:>5.1f} | "
              f"{r['pct_garbage']:>5.1f}")


if __name__ == "__main__":
    main()
