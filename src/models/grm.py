import os
import re
import torch
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from typing import List, Optional, Set

from src.utils.prompts import RUBRIC_GENERATION_PROMPT

# Default generation hyperparams (can be overridden via env vars)
_MAX_NEW_TOKENS = int(os.getenv("GRM_MAX_NEW_TOKENS", "512"))
_REPETITION_PENALTY = float(os.getenv("GRM_REPETITION_PENALTY", "1.2"))
_NO_REPEAT_NGRAM = int(os.getenv("GRM_NO_REPEAT_NGRAM", "0"))
_MAX_RETRIES = int(os.getenv("GRM_RETRY_EMPTY", "3"))

# Repetition-stopper / dedup settings
_SIM_STOP_THRESHOLD = float(os.getenv("GRM_SIM_STOP_THRESHOLD", "0.65"))
_MAX_CRITERIA = int(os.getenv("GRM_MAX_CRITERIA", "15"))
_DEDUP_THRESHOLD = float(os.getenv("GRM_DEDUP_THRESHOLD", "0.55"))


# ──────────────────────────────────────────────────────────────────────────
# Similarity helpers (character 3-gram Jaccard — fast & tokenizer-agnostic)
# ──────────────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> Counter:
    """Return a Counter of character n-grams for *text*."""
    t = text.lower().strip()
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def _jaccard(a: Counter, b: Counter) -> float:
    """Jaccard similarity between two n-gram Counters."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Custom StoppingCriteria: halt when rubric becomes repetitive
# ──────────────────────────────────────────────────────────────────────────

# Regex for extracting complete criterion lines from partial output
_CRITERION_LINE_RE = re.compile(
    r"^-\s*\[?\s*[+-]?\d+\s*\]?\s*.+",
    re.MULTILINE,
)


class RubricRepetitionStopper(StoppingCriteria):
    """Stop generation when:
    1. A newly generated criterion is too similar to any previous one, OR
    2. The total number of criteria reaches ``max_criteria``.

    Works with batched generation (checks each sequence independently).
    """

    def __init__(
        self,
        tokenizer,
        prompt_lengths: List[int],
        sim_threshold: float = _SIM_STOP_THRESHOLD,
        max_criteria: int = _MAX_CRITERIA,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths
        self.sim_threshold = sim_threshold
        self.max_criteria = max_criteria
        # Track per-sequence state
        self._prev_criteria_count: List[int] = [0] * len(prompt_lengths)
        self._prev_ngrams: List[List[Counter]] = [[] for _ in prompt_lengths]
        self._stopped: List[bool] = [False] * len(prompt_lengths)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check each sequence in the batch
        all_stopped = True
        for seq_idx in range(input_ids.shape[0]):
            if self._stopped[seq_idx]:
                continue  # already flagged

            gen_ids = input_ids[seq_idx, self.prompt_lengths[seq_idx]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            lines = _CRITERION_LINE_RE.findall(text)
            n_criteria = len(lines)

            # Hard cap
            if n_criteria >= self.max_criteria:
                self._stopped[seq_idx] = True
                continue

            # Check if a new criterion appeared since last call
            if n_criteria > self._prev_criteria_count[seq_idx]:
                new_line = lines[-1]
                new_ng = _char_ngrams(new_line)
                # Compare to all prior criteria
                for prev_ng in self._prev_ngrams[seq_idx]:
                    if _jaccard(new_ng, prev_ng) >= self.sim_threshold:
                        self._stopped[seq_idx] = True
                        break
                if not self._stopped[seq_idx]:
                    self._prev_ngrams[seq_idx].append(new_ng)
                self._prev_criteria_count[seq_idx] = n_criteria

            if not self._stopped[seq_idx]:
                all_stopped = False

        return all_stopped


class RubricGenerator:
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        if device == "auto":
            from model_worker import best_device
            device = best_device()
        self.device = device
        print(f"Loading Rubric Generator (GRM) from: {model_name_or_path} on {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"Error loading GRM model {model_name_or_path}: {e}")
            raise e

        # Pre-check whether the tokenizer supports enable_thinking kwarg
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                enable_thinking=False, add_generation_prompt=True, return_tensors="pt",
            )
            self._supports_thinking = True
        except TypeError:
            self._supports_thinking = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages_list: List[List[dict]]) -> torch.Tensor:
        """Tokenise one or more conversations, left-pad, return (input_ids, attention_mask)."""
        all_ids = []
        tpl_kwargs = {"add_generation_prompt": True}
        if self._supports_thinking:
            tpl_kwargs["enable_thinking"] = False

        for msgs in messages_list:
            ids = self.tokenizer.apply_chat_template(msgs, **tpl_kwargs)
            all_ids.append(ids)

        # Left-pad so we can batch
        max_len = max(len(ids) for ids in all_ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded, masks = [], []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip <think> blocks and trailing noise."""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _deduplicate_rubric(text: str, threshold: float = _DEDUP_THRESHOLD) -> str:
        """Remove near-duplicate criterion lines from a generated rubric.

        Keeps the *first* occurrence of each semantically-unique criterion
        (by char-3-gram Jaccard similarity).  Non-criterion lines (headers,
        blank lines, etc.) are passed through unchanged.
        """
        lines = text.split("\n")
        kept: List[str] = []
        kept_ngrams: List[Counter] = []

        for line in lines:
            stripped = line.strip()
            # If not a criterion line, keep it as-is
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

    # ------------------------------------------------------------------
    # Single generation (with retry for empty output)
    # ------------------------------------------------------------------

    def generate_rubric(self, question: str) -> str:
        """Generate a rubric, retrying up to _MAX_RETRIES on empty output."""
        prompt = RUBRIC_GENERATION_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, _MAX_RETRIES + 1):
            input_ids, attention_mask = self._apply_chat_template([messages])
            prompt_len = input_ids.shape[-1]

            # Build stopping criteria
            stopper = RubricRepetitionStopper(
                self.tokenizer,
                prompt_lengths=[prompt_len],
                sim_threshold=_SIM_STOP_THRESHOLD,
                max_criteria=_MAX_CRITERIA,
            )

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=_REPETITION_PENALTY,
                stopping_criteria=[stopper],
            )
            if _NO_REPEAT_NGRAM > 0:
                gen_kwargs["no_repeat_ngram_size"] = _NO_REPEAT_NGRAM

            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)

            generated_ids = outputs[0][prompt_len:]
            result = self._clean_output(
                self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            # Post-generation dedup (inference-time safety net)
            if result:
                result = self._deduplicate_rubric(result)
            if result:
                return result
            print(f"Warning: empty rubric (attempt {attempt}/{_MAX_RETRIES}) for: {question[:80]}...")

        # All retries exhausted — return empty string
        return ""

    # ------------------------------------------------------------------
    # Batched generation (left-padded, true GPU parallelism)
    # ------------------------------------------------------------------

    def generate_batch(self, questions: List[str], batch_size: int = 8) -> List[str]:
        """Generate rubrics in GPU-parallel batches."""
        all_messages = [
            [{"role": "user", "content": RUBRIC_GENERATION_PROMPT.format(question=q)}]
            for q in questions
        ]

        results: List[str] = [""] * len(questions)
        retry_indices: List[int] = list(range(len(questions)))

        for attempt in range(1, _MAX_RETRIES + 1):
            if not retry_indices:
                break

            # Process in sub-batches
            for start in range(0, len(retry_indices), batch_size):
                batch_idx = retry_indices[start : start + batch_size]
                batch_msgs = [all_messages[i] for i in batch_idx]

                input_ids, attention_mask = self._apply_chat_template(batch_msgs)

                # Per-sequence prompt lengths (after left-padding, all start at same col)
                prompt_len = input_ids.shape[-1]
                prompt_lengths = [prompt_len] * len(batch_idx)

                stopper = RubricRepetitionStopper(
                    self.tokenizer,
                    prompt_lengths=prompt_lengths,
                    sim_threshold=_SIM_STOP_THRESHOLD,
                    max_criteria=_MAX_CRITERIA,
                )

                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=_REPETITION_PENALTY,
                    stopping_criteria=[stopper],
                )
                if _NO_REPEAT_NGRAM > 0:
                    gen_kwargs["no_repeat_ngram_size"] = _NO_REPEAT_NGRAM

                with torch.no_grad():
                    outputs = self.model.generate(**gen_kwargs)

                for local_j, global_i in enumerate(batch_idx):
                    gen_ids = outputs[local_j][prompt_len:]
                    text = self._clean_output(
                        self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    )
                    # Post-generation dedup
                    if text:
                        text = self._deduplicate_rubric(text)
                    results[global_i] = text

            # Collect indices that are still empty for retry
            still_empty = [i for i in retry_indices if not results[i]]
            if still_empty:
                print(f"Retry {attempt}/{_MAX_RETRIES}: {len(still_empty)} empty rubrics")
            retry_indices = still_empty

        return results
