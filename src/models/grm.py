import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

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
        # Simple prompt for now, should match training format
        prompt = f"""User: Create a detailed scoring rubric (Principles) for the following question.
Question: {question}

Assistant: Here is the scoring rubric:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the rubric part
        if "Assistant: Here is the scoring rubric:" in generated_text:
            return generated_text.split("Assistant: Here is the scoring rubric:")[-1].strip()
        return generated_text.strip()

    def generate_batch(self, questions: List[str]) -> List[str]:
        return [self.generate_rubric(q) for q in questions]
