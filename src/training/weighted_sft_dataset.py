import json
from typing import Any

import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

from src.utils.prompts import RUBRIC_GENERATION_PROMPT


class WeightedRubricSFTDataset(Dataset):
    """
    verl-compatible custom SFT dataset with criterion-level token weights.

    Input files are JSONL rows in one of these schemas:
    1) {"question": str, "rubric": [str | {criterion, points}]}
    2) {"prompt": list[{role, content}] | str, "rubrics": [{criterion, points}]}

    Returned sample adds `token_loss_weight` (same length as `input_ids`).
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, processor=None, max_samples: int = -1):
        del processor  # unused

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.files = [copy_to_local(p, verbose=True, use_shm=config.get("use_shm", False)) for p in parquet_files]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_length = int(config.get("max_length", 1024))
        self.truncation = config.get("truncation", "left")
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.sft_instruction_template = config.get(
            "sft_instruction_template",
            RUBRIC_GENERATION_PROMPT,
        )

        # weighting knobs
        self.point_alpha = float(config.get("point_alpha", 0.08))
        self.negative_boost = float(config.get("negative_boost", 1.3))
        self.min_weight = float(config.get("min_weight", 0.2))
        self.max_weight = float(config.get("max_weight", 3.0))

        self.records = []
        for p in self.files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.records.append(json.loads(line))

        if max_samples > 0:
            self.records = self.records[:max_samples]

        print(f"weighted_sft dataset len: {len(self.records)}")

    @staticmethod
    def _prompt_to_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt.strip()
        if isinstance(prompt, list):
            lines = []
            for m in prompt:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                if content:
                    lines.append(f"{role.capitalize()}: {content}")
            return "\n".join(lines).strip()
        return str(prompt).strip()

    def _normalize_rubrics(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        rubrics = row.get("rubrics", row.get("rubric", []))
        out: list[dict[str, Any]] = []
        if not isinstance(rubrics, list):
            return out

        for r in rubrics:
            if isinstance(r, dict):
                c = str(r.get("criterion", "")).strip()
                if not c:
                    continue
                p = int(r.get("points", 1))
                out.append({"criterion": c, "points": p})
            else:
                c = str(r).strip()
                if c:
                    out.append({"criterion": c, "points": 1})
        return out

    def _point_to_weight(self, points: int) -> float:
        w = 1.0 + self.point_alpha * abs(float(points))
        if points < 0:
            w *= self.negative_boost
        return float(max(self.min_weight, min(self.max_weight, w)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]

        question = (row.get("question") or "").strip()
        if not question:
            question = self._prompt_to_text(row.get("prompt"))

        if "{question}" in self.sft_instruction_template:
            instructed_question = self.sft_instruction_template.format(question=question)
        else:
            instructed_question = f"{self.sft_instruction_template}\n\nQuestion:\n{question}"

        rubrics = self._normalize_rubrics(row)

        prompt_chat = [{"role": "user", "content": instructed_question}]
        prompt_chat_str = self.tokenizer.apply_chat_template(
            prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        prompt_ids = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        prompt_attention = torch.ones_like(prompt_ids)

        response_parts: list[torch.Tensor] = []
        response_weights: list[torch.Tensor] = []
        for r in rubrics:
            pts = int(r["points"])
            criterion = r["criterion"]
            line = f"- [{pts:+d}] {criterion}\n"
            line_ids = self.tokenizer(line, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            w = self._point_to_weight(pts)
            response_parts.append(line_ids)
            response_weights.append(torch.full_like(line_ids, fill_value=w, dtype=torch.float32))

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos = torch.tensor([eos_id], dtype=torch.long)
            response_parts.append(eos)
            response_weights.append(torch.ones(1, dtype=torch.float32))

        if response_parts:
            response_ids = torch.cat(response_parts, dim=-1)
            response_token_weight = torch.cat(response_weights, dim=-1)
        else:
            response_ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            response_token_weight = torch.ones(1, dtype=torch.float32)

        response_attention = torch.ones_like(response_ids)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention, response_attention), dim=-1)
        token_loss_weight = torch.cat((torch.zeros_like(prompt_ids, dtype=torch.float32), response_token_weight), dim=-1)

        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                token_loss_weight = token_loss_weight[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                token_loss_weight = token_loss_weight[: self.max_length]
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")
        elif seq_len < self.max_length:
            pad_n = self.max_length - seq_len
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.cat((input_ids, torch.full((pad_n,), pad_id, dtype=input_ids.dtype)), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.zeros((pad_n,), dtype=attention_mask.dtype)), dim=-1)
            token_loss_weight = torch.cat((token_loss_weight, torch.zeros((pad_n,), dtype=torch.float32)), dim=-1)

        position_ids = compute_position_id_with_mask(attention_mask)

        # base mask like SFTDataset: supervise completion tokens only
        loss_mask = attention_mask.clone().float()
        prompt_length = int(prompt_ids.shape[0])
        response_length = int(response_ids.shape[0])
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "token_loss_weight": token_loss_weight,
        }
