#!/usr/bin/env python3
"""Scrape xty.app usage logs and compute per-model cost statistics.
quota: 1 unit = $2e-6 USD (verified: 87006 units = $0.174012)
"""
import argparse, json, math, os, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import requests

API_KEY = "sk-Dk3063EgG7Lezr9BJ3nkpgtGsv89KR4CKrODFLB0lgqr0E4d"
BASE_URL = "https://cxapi-all.xty.app"
RAW_FILE = "out/xty_usage_raw.json"
CST = timezone(timedelta(hours=8))
BENCH_START_TS = datetime(2026, 5, 5, 18, 42, 0, tzinfo=CST).timestamp()
QUOTA_TO_USD = 2e-6
GPT5_MIN_OUTPUT = 2048
PRICING = {
    "gpt-5":                  {"in": 15.0,  "out": 60.0},
    "deepseek-v3.1":          {"in": 0.27,  "out": 1.1},
    "gemini-3.1-pro-preview": {"in": 7.0,   "out": 21.0},
    "claude-opus-4-7":        {"in": 15.0,  "out": 75.0},
    "grok-4":                 {"in": 3.0,   "out": 15.0},
}
COMPLETED = {
    "gpt-5": 424, "deepseek-v3.1": 709, "gemini-3.1-pro-preview": 709,
    "claude-opus-4-7": 700, "grok-4": 709,
}
TOTAL = 709


def fetch_all():
    rows = 100
    data = requests.get(f"{BASE_URL}/log/showList?page=1&rows={rows}&apiKey={API_KEY}", timeout=30).json()["content"]
    total_pages = math.ceil(data["total"] / rows)
    all_recs = list(data["records"])
    print(f"Total: {data['total']} records, {total_pages} pages @ {rows}/page")
    for page in range(2, total_pages + 1):
        for attempt in range(3):
            try:
                recs = requests.get(
                    f"{BASE_URL}/log/showList?page={page}&rows={rows}&apiKey={API_KEY}", timeout=30
                ).json()["content"]["records"]
                all_recs.extend(recs)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ERROR p{page}: {e}")
                else:
                    time.sleep(1)
        if page % 10 == 0 or page == total_pages:
            print(f"  Page {page}/{total_pages}: {len(all_recs)} fetched")
        time.sleep(0.05)
    return all_recs


def analyze(records):
    print(f"\nBench start unix ts: {BENCH_START_TS:.0f}")
    bench = [r for r in records if r["createdAt"] >= BENCH_START_TS]
    print(f"Records total: {len(records)}, since bench start: {len(bench)}")
    by_model = defaultdict(list)
    for r in bench:
        by_model[r["modelName"]].append(r)
    print(f"Models in bench window: {sorted(by_model.keys())}\n")

    # Summary table
    print("=" * 90)
    print(f"{'Model':<35} {'Calls':>6} {'Input':>12} {'Output':>12} {'Cost USD':>12}")
    print("=" * 90)
    for m in sorted(by_model.keys()):
        recs = by_model[m]
        ti = sum(r["promptTokens"] for r in recs)
        to = sum(r["completionTokens"] for r in recs)
        tc = sum(r["quota"] for r in recs) * QUOTA_TO_USD
        print(f"{m:<35} {len(recs):>6} {ti:>12,} {to:>12,} ${tc:>11.4f}")
    print("=" * 90)

    # Detailed per model
    results = {}
    print("\n" + "=" * 90)
    print("DETAILED ANALYSIS")
    print("=" * 90)
    for mk in ["gpt-5", "deepseek-v3.1", "gemini-3.1-pro-preview", "claude-opus-4-7", "grok-4"]:
        recs = by_model.get(mk, [])
        n_done = COMPLETED.get(mk, len(recs))
        if not recs:
            print(f"\n{mk}: NOT FOUND")
            results[mk] = {}
            continue
        ti = sum(r["promptTokens"] for r in recs)
        to = sum(r["completionTokens"] for r in recs)
        tc = sum(r["quota"] for r in recs) * QUOTA_TO_USD
        print(f"\n{mk}: {len(recs)} calls, {ti:,} in / {to:,} out tokens, ${tc:.4f} total")
        if mk == "gpt-5":
            full = [r for r in recs if r["completionTokens"] > GPT5_MIN_OUTPUT]
            short = [r for r in recs if r["completionTokens"] <= GPT5_MIN_OUTPUT]
            fc = sum(r["quota"] for r in full) * QUOTA_TO_USD
            sc = sum(r["quota"] for r in short) * QUOTA_TO_USD
            print(f"  Full calls (output>{GPT5_MIN_OUTPUT} tokens): {len(full)}, cost=${fc:.4f}")
            print(f"  Short/retry calls: {len(short)}, cost=${sc:.4f}")
            analysis_recs, analysis_cost, analysis_n = full, fc, len(full)
        else:
            analysis_recs, analysis_cost, analysis_n = recs, tc, len(recs)

        if analysis_n > 0 and n_done > 0:
            cpp = analysis_cost / n_done
            ext = cpp * TOTAL
            avg_in = sum(r["promptTokens"] for r in analysis_recs) / analysis_n
            avg_out = sum(r["completionTokens"] for r in analysis_recs) / analysis_n
            print(f"  n_done={n_done}, cost/prompt=${cpp:.4f}, extrap to 709=${ext:.4f}")
            print(f"  Avg tokens: {avg_in:.0f}in / {avg_out:.0f}out")
            p = PRICING.get(mk)
            if p:
                theory = (avg_in * p["in"] + avg_out * p["out"]) / 1e6 * TOTAL
                print(f"  Theory from pricing: ${theory:.4f} for 709 prompts")
            results[mk] = {
                "calls": analysis_n, "n_done": n_done,
                "input_tokens": ti, "output_tokens": to,
                "cost_actual": analysis_cost,
                "cost_per_prompt": cpp,
                "cost_extrapolated_709": ext,
            }
        else:
            results[mk] = {}

    # Summary
    print("\n" + "=" * 90)
    print("COST SUMMARY")
    print("=" * 90)
    print(f"{'Model':<35} {'N done':>8} {'Actual cost':>14} {'Extrap 709':>14}")
    print("-" * 90)
    ta, te = 0.0, 0.0
    for mk in ["gpt-5", "deepseek-v3.1", "gemini-3.1-pro-preview", "claude-opus-4-7", "grok-4"]:
        r = results.get(mk, {})
        n = r.get("n_done", 0)
        ac = r.get("cost_actual", 0)
        ex = r.get("cost_extrapolated_709", ac)
        ta += ac
        te += ex
        print(f"{mk:<35} {n:>8} ${ac:>12.4f} ${ex:>12.4f}")
    print("-" * 90)
    print(f"{'TOTAL':<44} ${ta:>12.4f} ${te:>12.4f}")
    print("=" * 90)

    os.makedirs("out", exist_ok=True)
    with open("out/xty_cost_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to out/xty_cost_analysis.json")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true")
    args = p.parse_args()
    os.makedirs("out", exist_ok=True)
    if not args.no_fetch:
        recs = fetch_all()
        with open(RAW_FILE, "w") as f:
            json.dump(recs, f, ensure_ascii=False)
        print(f"Saved {len(recs)} records to {RAW_FILE}")
    else:
        with open(RAW_FILE) as f:
            recs = json.load(f)
        print(f"Loaded {len(recs)} records")
    analyze(recs)


if __name__ == "__main__":
    main()
