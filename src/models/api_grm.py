"""API-based RubricGenerator for benchmarking external LLMs (e.g., GLM-5).

Implements the same generate_rubric / generate_batch interface as
RubricGenerator so it can be used as a drop-in replacement in
run_benchmark.py.
"""

import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from src.utils.prompts import RUBRIC_GENERATION_PROMPT

# ── Dedup helpers (same as grm.py) ────────────────────────────────────
_CRITERION_LINE_RE = re.compile(r"^\s*[-*]\s*\[")
_DEDUP_THRESHOLD = 0.55


def _char_ngrams(s: str, n: int = 3) -> Counter:
    s = s.lower()
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def _clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _deduplicate_rubric(text: str, threshold: float = _DEDUP_THRESHOLD) -> str:
    lines = text.split("\n")
    kept: List[str] = []
    kept_ngrams: List[Counter] = []
    for line in lines:
        stripped = line.strip()
        if not _CRITERION_LINE_RE.match(stripped):
            kept.append(line)
            continue
        ng = _char_ngrams(stripped)
        is_dup = False
        for prev_ng in kept_ngrams:
            if _jaccard(ng, prev_ng) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(line)
            kept_ngrams.append(ng)
    return "\n".join(kept)


class APIRubricGenerator:
    """Drop-in replacement for RubricGenerator that calls an external API."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_workers: int = 4,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        thinking: bool = True,
        max_retries: int = 6,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.max_retries = max_retries
        print(f"API Rubric Generator: model={model}, api_base={self.api_base}, "
              f"thinking={thinking}, max_workers={max_workers}")

    def _call_api(self, messages: List[dict]) -> str:
        """Single API call with retry logic and rate-limit backoff."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.thinking:
            body["thinking"] = {"type": "enabled"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=180)
                if resp.status_code == 429:
                    wait = min(5 * (2 ** attempt), 120)
                    print(f"Rate limited (429), waiting {wait}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait = min(5 * (2 ** attempt), 120)
                    print(f"Rate limited (429), waiting {wait}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(wait)
                    continue
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"API error (attempt {attempt}/{self.max_retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"API error (final attempt): {e}")
                    return ""
            except Exception as e:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"API error (attempt {attempt}/{self.max_retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"API error (final attempt): {e}")
                    return ""
        return ""

    def generate_rubric(self, question: str) -> str:
        prompt = RUBRIC_GENERATION_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        result = self._call_api(messages)
        result = _clean_output(result)
        if result:
            result = _deduplicate_rubric(result)
        # Delay between calls to avoid rate limits
        time.sleep(3)
        return result

    def generate_batch(self, questions: List[str], batch_size: int = 8) -> List[str]:
        """Generate rubrics in parallel via API calls."""
        results: List[str] = [""] * len(questions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {}
            for i, q in enumerate(questions):
                fut = pool.submit(self.generate_rubric, q)
                future_to_idx[fut] = i

            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"Error generating rubric {idx}: {e}")
                    results[idx] = ""
                done += 1
                if done % 20 == 0 or done == len(questions):
                    print(f"  API batch progress: {done}/{len(questions)}")

        return results
