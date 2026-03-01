import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

from src.utils.prompts import RUBRIC_GENERATION_PROMPT

# Default generation hyperparams (can be overridden via env vars)
_MAX_NEW_TOKENS = int(os.getenv("GRM_MAX_NEW_TOKENS", "512"))
_REPETITION_PENALTY = float(os.getenv("GRM_REPETITION_PENALTY", "1.2"))
_MAX_RETRIES = int(os.getenv("GRM_RETRY_EMPTY", "3"))


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

    # ------------------------------------------------------------------
    # Single generation (with retry for empty output)
    # ------------------------------------------------------------------

    def generate_rubric(self, question: str) -> str:
        """Generate a rubric, retrying up to _MAX_RETRIES on empty output."""
        prompt = RUBRIC_GENERATION_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, _MAX_RETRIES + 1):
            input_ids, attention_mask = self._apply_chat_template([messages])
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=_REPETITION_PENALTY,
                )
            generated_ids = outputs[0][input_ids.shape[-1]:]
            result = self._clean_output(
                self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
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
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=_MAX_NEW_TOKENS,
                        temperature=0.7,
                        do_sample=True,
                        repetition_penalty=_REPETITION_PENALTY,
                    )

                for local_j, global_i in enumerate(batch_idx):
                    gen_ids = outputs[local_j][input_ids.shape[-1]:]
                    text = self._clean_output(
                        self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    )
                    results[global_i] = text

            # Collect indices that are still empty for retry
            still_empty = [i for i in retry_indices if not results[i]]
            if still_empty:
                print(f"Retry {attempt}/{_MAX_RETRIES}: {len(still_empty)} empty rubrics")
            retry_indices = still_empty

        return results
