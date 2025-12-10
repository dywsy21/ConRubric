import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import httpx
from src.utils.prompts import REVERSE_ENGINEER_RUBRIC_PROMPT, ANCHOR_EVALUATION_PROMPT, DYNAMIC_RUBRIC_EVALUATION_PROMPT

# Global cache directory
CACHE_DIR = Path(os.environ.get("GRM_CACHE_DIR", "./cache/api_responses"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_key(model: str, prompt: str) -> str:
    """Generate a unique cache key for the request."""
    content = f"{model}|{prompt}"
    return hashlib.md5(content.encode()).hexdigest()

def _load_from_cache(cache_key: str, cache_type: str = "judge") -> Optional[str]:
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

def _save_to_cache(cache_key: str, response: str, cache_type: str = "judge"):
    """Save response to cache."""
    cache_subdir = CACHE_DIR / cache_type
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_subdir / f"{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump({'response': response}, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

class Judge:
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None, api_base: str = None):
        self.model_name = model_name
        # Create HTTP client with no timeout
        http_client = httpx.Client(timeout=None)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or os.environ.get("OPENAI_BASE_URL"),
            http_client=http_client
        )

    def _call_api(self, prompt: str, temperature: float = 0.7, use_cache: bool = True) -> str:
        # Check cache first (only for deterministic calls with temp=0)
        cache_key = None
        if use_cache and temperature == 0.0:
            cache_key = _get_cache_key(self.model_name, prompt)
            cached_response = _load_from_cache(cache_key, "judge")
            if cached_response is not None:
                return cached_response
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            result = response.choices[0].message.content
            
            # Save to cache for deterministic calls
            if cache_key:
                _save_to_cache(cache_key, result, "judge")
            
            return result
        except Exception as e:
            print(f"Error calling Judge API: {e}")
            return ""

    def reverse_engineer_rubric(self, question: str, gold_answer: str) -> List[str]:
        prompt = REVERSE_ENGINEER_RUBRIC_PROMPT.format(question=question, gold_answer=gold_answer)
        response_text = self._call_api(prompt, temperature=0.7)
        
        # Simple parsing logic - assumes the model follows instructions to output a JSON list or similar
        # In a real scenario, we'd want more robust parsing
        try:
            # Attempt to find JSON-like structure
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                rubric = json.loads(json_str)
                if isinstance(rubric, list):
                    return rubric
            
            # Fallback: split by newlines if it looks like a list
            lines = response_text.split('\n')
            rubric = [line.strip('- ').strip() for line in lines if line.strip().startswith('-') or line.strip().startswith('*') or line[0].isdigit()]
            return rubric
        except Exception as e:
            print(f"Error parsing rubric: {e}")
            return []

    def evaluate_answer(self, question: str, answer: str, rubric: str = None) -> float:
        """
        Evaluates an answer and returns a scalar score (0-10).
        """
        if rubric:
            prompt = DYNAMIC_RUBRIC_EVALUATION_PROMPT.format(question=question, answer=answer, rubric=rubric)
        else:
            prompt = ANCHOR_EVALUATION_PROMPT.format(question=question, answer=answer)
            
        response_text = self._call_api(prompt, temperature=0.0)
        
        try:
            # Attempt to find JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                
                # Handle both formats
                if "score" in result:
                    return float(result["score"])
                elif "overall" in result:
                    return float(result["overall"])
                else:
                    print("No score found in JSON")
                    return 0.0
            else:
                print("Could not find JSON in evaluation response")
                return 0.0
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return 0.0

    def check_correctness(self, question: str, answer: str, gold: str) -> bool:
        """
        Checks if the answer is correct relative to the gold solution.
        """
        prompt = f"""You are a math grader.
Problem: {question}
Gold Solution: {gold}

Student Answer: {answer}

Is the Student Answer correct according to the Gold Solution? 
Respond with only "YES" or "NO".
"""
        response = self._call_api(prompt, temperature=0.0).strip().upper()
        return "YES" in response

# Alias for backward compatibility if needed, though we changed the class name
Oracle = Judge
