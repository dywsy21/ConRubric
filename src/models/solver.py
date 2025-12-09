import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

class Solver:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"Loading Solver model: {model_name} on {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback or raise
            raise e

    def generate_answer(self, question: str, rubric: str) -> str:
        """
        Generates an answer for the question, guided by the rubric.
        """
        # Construct prompt
        # This is a simple prompt strategy. Can be improved.
        prompt = f"""You are a helpful assistant. Please answer the following question.
        
Question:
{question}

Please ensure your answer follows these principles:
{rubric}

Answer:
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
        
        # Extract the answer part (naive split)
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()
        return generated_text.strip()

    def generate_batch(self, questions: List[str], rubrics: List[str]) -> List[str]:
        return [self.generate_answer(q, r) for q, r in zip(questions, rubrics)]
