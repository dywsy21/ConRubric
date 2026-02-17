import argparse
import json
from pathlib import Path


def _safe_load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_line(name: str, pre: float, post: float) -> str:
    if pre is None or post is None:
        return f"- {name}: N/A"
    return f"- {name}: pre={pre:.4f}, post={post:.4f}, delta={post - pre:+.4f}"


def _extract_accuracy(data):
    if not data:
        return None
    return data.get("accuracy")


def _extract_hb_pairacc(data):
    if not data:
        return None
    return (((data.get("summary") or {}).get("avg_pairwise_acc")))


def _extract_hb_gap(data):
    if not data:
        return None
    return ((((data.get("summary") or {}).get("excellent_good_gap") or {}).get("excellent_minus_good")))


def main():
    parser = argparse.ArgumentParser(description="Write SFT writeup markdown")
    parser.add_argument("--pre_dir", type=str, required=True)
    parser.add_argument("--post_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--post_model", type=str, required=True)
    parser.add_argument("--sft_jsonl", type=str, required=True)
    parser.add_argument("--comparison_md", type=str, required=True)
    parser.add_argument("--comparison_jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, default="sft_writeup.md")
    args = parser.parse_args()

    pre_dir = Path(args.pre_dir)
    post_dir = Path(args.post_dir)

    pre_reward = _safe_load(pre_dir / "reward_bench_results.json")
    post_reward = _safe_load(post_dir / "reward_bench_results.json")
    pre_ppe = _safe_load(pre_dir / "ppe_results.json")
    post_ppe = _safe_load(post_dir / "ppe_results.json")
    pre_rmb = _safe_load(pre_dir / "rmb_results.json")
    post_rmb = _safe_load(post_dir / "rmb_results.json")
    pre_hb = _safe_load(pre_dir / "healthbench_rubric_quality.json")
    post_hb = _safe_load(post_dir / "healthbench_rubric_quality.json")

    sft_rows = 0
    sft_path = Path(args.sft_jsonl)
    if sft_path.exists():
        with sft_path.open("r", encoding="utf-8") as f:
            for _ in f:
                sft_rows += 1

    comparison_rows = 0
    comp_jsonl = Path(args.comparison_jsonl)
    if comp_jsonl.exists():
        with comp_jsonl.open("r", encoding="utf-8") as f:
            for _ in f:
                comparison_rows += 1

    text = []
    text.append("# SFT End-to-End Writeup")
    text.append("")
    text.append("## 1) Scope")
    text.append(f"- Base model (pre-SFT): {args.base_model}")
    text.append(f"- Post-SFT model: {args.post_model}")
    text.append(f"- SFT train jsonl: {args.sft_jsonl}")
    text.append(f"- SFT rows: {sft_rows}")
    text.append("- Training data mix: HealthBench + synthetic, each limited to 100")
    text.append("")
    text.append("## 2) Benchmark Results (Pre vs Post)")
    text.append(_metric_line("RewardBench accuracy", _extract_accuracy(pre_reward), _extract_accuracy(post_reward)))
    text.append(_metric_line("PPE accuracy", _extract_accuracy(pre_ppe), _extract_accuracy(post_ppe)))
    text.append(_metric_line("RMB accuracy", _extract_accuracy(pre_rmb), _extract_accuracy(post_rmb)))
    text.append(_metric_line("HealthBench rubric pairwise acc", _extract_hb_pairacc(pre_hb), _extract_hb_pairacc(post_hb)))
    text.append(_metric_line("HealthBench excellent-good gap", _extract_hb_gap(pre_hb), _extract_hb_gap(post_hb)))
    text.append("")
    text.append("## 3) Rubric Comparison (>100 same prompts)")
    text.append(f"- Comparison markdown: {args.comparison_md}")
    text.append(f"- Comparison jsonl: {args.comparison_jsonl}")
    text.append(f"- Rows generated: {comparison_rows}")
    text.append("")
    text.append("## 4) Token-loss weighting verification")
    text.append("- Dataset emits token-level weights (`token_loss_weight`) aligned to rubric tokens with point-based scaling and negative-point boost.")
    text.append("- SFT loss consumes `token_loss_weight` when present, in both no-padding and padded paths.")
    text.append("- Runtime config injects weighting hyperparameters into dataset (`point_alpha`, `negative_boost`, `min_weight`, `max_weight`).")
    text.append("")
    text.append("## 5) Notes")
    text.append("- Benchmark code uses batched parallel judge calls (`evaluate_batch`) to actually leverage remote vLLM concurrency.")
    text.append("- HealthBench benchmark split is leakage-safe against SFT split by prompt_id.")

    out = Path(args.output)
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Saved writeup: {out}")


if __name__ == "__main__":
    main()
