"""Plot cost-performance chart comparing frontier API models vs MetaRM-GRM.

The plotted performance metric is Kendall tau-b on the discriminative
HealthBench subset used in the paper. This is the headline metric because the
benchmark is an ordinal ranking task and model scores are coarse integers with
many ties.

Usage:
    python scripts/plot_price_score.py
    python scripts/plot_price_score.py --out out/bench/api_models/price_score.pdf
"""

import argparse
import os

import matplotlib.pyplot as plt
MODELS = [
    # display_name, cost_usd_per_709, kendall_tau_b, group, marker, size
    ("Base Qwen3-8B", 0.05, 0.206, "Local 8B", "^", 86),
    ("GRM", 0.05, 0.267, "Local 8B", "o", 94),
    ("GRM-KS", 0.05, 0.284, "Local 8B", "*", 190),
    ("DeepSeek-v3.1", 4.92, 0.233, "Frontier API", "D", 80),
    ("Grok-4", 8.90, 0.199, "Frontier API", "D", 80),
    ("GLM-5", 13.50, 0.224, "Frontier API", "D", 80),
    ("Gemini 3.1-Pro", 29.58, 0.206, "Frontier API", "D", 80),
    ("Claude Opus 4.7", 45.49, 0.227, "Frontier API", "D", 80),
]

COLORS = {
    "Local 8B": "#D94E3B",
    "Frontier API": "#276D8C",
    "Frontier Line": "#A83E32",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/cost_performance.pdf")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(7.15, 3.0))

    ax.axvspan(0.035, 0.07, color="#F7D6CC", alpha=0.55, linewidth=0)

    for name, cost, tau, group, marker, size in MODELS:
        ax.scatter(cost, tau, s=size, marker=marker, color=COLORS[group],
                   edgecolors="white", linewidths=0.9,
                   zorder=6 if group == "Local 8B" else 5)

    labels = {
        "Base Qwen3-8B": (8, -8),
        "GRM": (8, 0),
        "GRM-KS": (8, 7),
        "DeepSeek-v3.1": (8, 3),
        "Grok-4": (8, -10),
        "GLM-5": (8, -3),
        "Gemini 3.1-Pro": (8, -9),
        "Claude Opus 4.7": (8, 3),
    }
    for name, cost, tau, group, _, _ in MODELS:
        weight = "bold" if name == "GRM-KS" else "normal"
        ax.annotate(name, (cost, tau), xytext=labels[name],
                    textcoords="offset points", color=COLORS[group],
                    fontsize=8.2 if group == "Local 8B" else 7.8,
                    fontweight=weight)

    ax.set_xscale("log")
    ax.set_xlim(0.02, 75)
    ax.set_ylim(0.188, 0.296)
    ax.set_xlabel("Rubric generation cost for 709 prompts (USD, log scale)")
    ax.set_ylabel(r"Performance: Kendall $\tau$-b")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(0.284, color=COLORS["Local 8B"], linewidth=0.8, linestyle=":", alpha=0.55)

    plt.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Also save PNG
    png_out = args.out.replace(".pdf", ".png")
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_out}")
    # plt.show()  # disabled for headless environments


if __name__ == "__main__":
    main()
