import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

from src.utils.prompts import RUBRIC_GENERATION_PROMPT

class RubricGenerator:
    def __init__(self, model_name_or_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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

    def generate_rubric(self, question: str) -> str:
        """
        Generates a rubric for the given question.
        """
        # Use the same prompt template as SFT/RL training
        prompt = RUBRIC_GENERATION_PROMPT.format(question=question)

        # Wrap in chat template so the model treats it as an instruction.
        # Disable thinking mode (Qwen3) to get direct rubric output.
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        try:
            # Qwen3 supports enable_thinking; other models will ignore / raise
            input_ids = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **chat_kwargs
            ).to(self.device)
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages, **chat_kwargs
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_batch(self, questions: List[str]) -> List[str]:
        return [self.generate_rubric(q) for q in questions]
