"""Plot price-score chart comparing frontier API models vs MetaRM-GRM.

Reads results from out/bench/api_models/all_results_summary.json and
combines with known MetaRM-GRM/baseline numbers from the paper.

Usage:
    python scripts/plot_price_score.py
    python scripts/plot_price_score.py --out out/bench/api_models/price_score.pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Known results from the paper (559 common prompts, Table 1)
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_MODELS = {
    # name: (kendall_tau_b, pairwise_acc, cost_usd_per_709, marker, color, label)
    "MetaRM-GRM\n(Ours)": {
        "kendall_tau_b": 0.270,
        "pairwise_acc":  0.611,
        "cost_usd":      0.0,   # local GPU, marginal cost ≈ 0 (amortized)
        "marker": "*",
        "color":  "#e63946",
        "label":  "MetaRM-GRM (Ours, 8B)",
        "zorder": 10,
    },
    "Base Qwen3-8B": {
        "kendall_tau_b": 0.203,
        "pairwise_acc":  0.599,
        "cost_usd":      0.0,
        "marker": "^",
        "color":  "#457b9d",
        "label":  "Base Qwen3-8B",
        "zorder": 5,
    },
}

# Approximate per-1M pricing for API models
PRICE_PER_1M = {
    "gpt-5":                 (15.0,  60.0),
    "deepseek-v3.1":         (0.27,   1.1),
    "gemini-3.1-pro-preview":(7.0,   21.0),
    "claude-opus-4-7":       (15.0,  75.0),
    "grok-4":                (3.0,   15.0),
}

MODEL_DISPLAY = {
    "gpt-5":                 "GPT-5",
    "deepseek-v3.1":         "DeepSeek-V3.1",
    "gemini-3.1-pro-preview":"Gemini 3.1 Pro",
    "claude-opus-4-7":       "Claude Opus 4.7",
    "grok-4":                "Grok-4",
}

# Colors for API models
API_COLORS = [
    "#f4a261",  # GPT-5
    "#264653",  # DeepSeek
    "#1d3557",  # Gemini
    "#8338ec",  # Claude
    "#fb5607",  # Grok
]


def load_api_results(summary_file: str):
    if not os.path.exists(summary_file):
        print(f"WARNING: {summary_file} not found; API model points will be omitted.")
        return {}
    with open(summary_file) as f:
        data = json.load(f)
    return data


def compute_cost(usage: dict, model: str) -> float:
    """Estimate cost in USD from token usage."""
    p_in, p_out = PRICE_PER_1M.get(model, (0, 0))
    return (
        usage.get("input_tokens", 0) / 1e6 * p_in +
        usage.get("output_tokens", 0) / 1e6 * p_out
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="out/bench/api_models/all_results_combined.json")
    parser.add_argument("--out", default="out/bench/api_models/price_score.pdf")
    parser.add_argument("--metric", default="kendall_tau_b",
                        choices=["kendall_tau_b", "pairwise_acc", "avg_spearman"])
    args = parser.parse_args()

    api_data = load_api_results(args.results)

    fig, ax = plt.subplots(figsize=(9, 6))

    legend_handles = []

    # ── Plot API models ──
    for i, (model, color) in enumerate(zip(PRICE_PER_1M.keys(), API_COLORS)):
        if model not in api_data:
            continue
        r = api_data[model]
        metrics = r.get("metrics", {})
        usage   = r.get("usage", {})

        metric_key = f"avg_{args.metric}" if not args.metric.startswith("avg_") else args.metric
        y = metrics.get(metric_key, float("nan"))
        if args.metric == "kendall_tau_b":
            y = metrics.get("avg_kendall_tau_b", float("nan"))
        elif args.metric == "pairwise_acc":
            y = metrics.get("avg_pairwise_acc", float("nan"))
        elif args.metric == "avg_spearman":
            y = metrics.get("avg_spearman", float("nan"))

        cost = r.get("cost_usd_estimate") or compute_cost(usage, model)

        if np.isnan(y):
            continue

        display = MODEL_DISPLAY.get(model, model)
        ax.scatter(cost, y, s=120, color=color, marker="D", zorder=6,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(display, (cost, y), textcoords="offset points",
                    xytext=(-60, 4) if cost > 50 else (8, 4), fontsize=9, color=color)
        legend_handles.append(
            mpatches.Patch(color=color, label=display)
        )

    # ── Plot known models ──
    for name, info in KNOWN_MODELS.items():
        y_key = args.metric if not args.metric.startswith("avg_") else args.metric[4:]
        y = info.get(y_key, info.get("kendall_tau_b", float("nan")))
        cost = info["cost_usd"]
        color = info["color"]
        marker = info["marker"]
        zorder = info["zorder"]

        ax.scatter(cost, y, s=200 if marker == "*" else 120,
                   color=color, marker=marker, zorder=zorder,
                   edgecolors="white", linewidths=0.8)

        offset = (-60, 8) if name.startswith("MetaRM") else (8, 4)
        ax.annotate(info["label"], (cost, y), textcoords="offset points",
                    xytext=offset, fontsize=9, color=color,
                    fontweight="bold" if marker == "*" else "normal")
        legend_handles.append(
            mpatches.Patch(color=color, label=info["label"])
        )

    # ── Formatting ──
    metric_labels = {
        "kendall_tau_b": "Kendall τ-b",
        "pairwise_acc":  "Pairwise Accuracy",
        "avg_spearman":  "Spearman ρ",
    }
    ax.set_xlabel("Estimated Cost per 709 Prompts (USD)", fontsize=11)
    ax.set_ylabel(metric_labels.get(args.metric, args.metric), fontsize=11)
    ax.set_title("Rubric Generation Quality vs. Cost\n(HealthBench, 709 benchmark prompts)",
                 fontsize=12)

    # Log scale x-axis if range spans >2 orders of magnitude
    xvals = [ax.get_xlim()[0]]
    for line in ax.get_lines():
        xvals.extend(line.get_xdata())
    try:
        all_costs = [info["cost_usd"] for info in KNOWN_MODELS.values()] + \
                    [api_data[m].get("cost_usd_estimate", 0) for m in api_data]
        pos_costs = [c for c in all_costs if c > 0]
        if pos_costs and max(pos_costs) / max(min(pos_costs), 0.001) > 100:
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_xlabel("Estimated Cost per 709 Prompts (USD, symlog scale)", fontsize=11)
    except Exception:
        pass

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.9)

    # Highlight MetaRM-GRM threshold
    grm_y = KNOWN_MODELS["MetaRM-GRM\n(Ours)"].get(args.metric, KNOWN_MODELS["MetaRM-GRM\n(Ours)"]["kendall_tau_b"])
    ax.axhline(y=grm_y, color="#e63946", linestyle=":", alpha=0.4, linewidth=1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Also save PNG
    png_out = args.out.replace(".pdf", ".png")
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_out}")
    # plt.show()  # disabled for headless environments


if __name__ == "__main__":
    main()
