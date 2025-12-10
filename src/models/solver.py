import os
import hashlib
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from openai import OpenAI
import httpx

# Global cache directory
CACHE_DIR = Path(os.environ.get("GRM_CACHE_DIR", "./cache/api_responses"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_key(model: str, question: str, rubric: str = "") -> str:
    """Generate a unique cache key for the request."""
    content = f"{model}|{question}|{rubric}"
    return hashlib.md5(content.encode()).hexdigest()

def _load_from_cache(cache_key: str, cache_type: str = "solver") -> Optional[str]:
    """Load response from cache if exists."""
    cache_file = CACHE_DIR / cache_type / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('response')
        except Exception:
            pass
    return None

def _save_to_cache(cache_key: str, response: str, cache_type: str = "solver", metadata: dict = None):
    """Save response to cache."""
    cache_subdir = CACHE_DIR / cache_type
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_subdir / f"{cache_key}.json"
    try:
        data = {'response': response}
        if metadata:
            data['metadata'] = metadata
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

class Solver:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 is_remote: bool = False, api_key: str = None, api_base: str = None):
        self.model_name = model_name
        self.is_remote = is_remote
        
        if self.is_remote:
            print(f"Initializing Remote Solver: {model_name}")
            # Create HTTP client with no timeout
            http_client = httpx.Client(timeout=None)
            self.client = OpenAI(
                api_key=api_key or os.environ.get("SOLVER_API_KEY"),
                base_url=api_base or os.environ.get("SOLVER_API_BASE"),
                http_client=http_client
            )
        else:
            self.device = device
            print(f"Loading Local Solver model: {model_name} on {device}")
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
                raise e

    def generate_answer(self, question: str, rubric: str) -> str:
        """
        Generates an answer for the question, guided by the rubric.
        """
        # Check cache first
        cache_key = _get_cache_key(self.model_name, question, rubric)
        cached_response = _load_from_cache(cache_key, "solver")
        if cached_response is not None:
            return cached_response
        
        # Construct prompt
        prompt = f"""You are a helpful assistant. Please answer the following question.
        
Question:
{question}

Please ensure your answer follows these principles:
{rubric}

Answer:
"""
        if self.is_remote:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512
                )
                answer = response.choices[0].message.content
                # Save to cache
                _save_to_cache(cache_key, answer, "solver", 
                              {'model': self.model_name, 'question': question[:100], 'rubric': rubric[:100]})
                return answer
            except Exception as e:
                print(f"Error calling Remote Solver API: {e}")
                return ""
        else:
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
