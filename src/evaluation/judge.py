import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
import httpx

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Default parallel settings
DEFAULT_MAX_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 8))
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

# Default timeout settings
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3

class Judge:
    def __init__(self, model_name: str = None, api_key: str = None, api_base: str = None,
                 timeout: float = DEFAULT_TIMEOUT):
        # Priority: explicit args > ORACLE_* env vars > OPENAI_* env vars
        self.model_name = model_name or os.environ.get("ORACLE_MODEL_NAME") or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.base_timeout = timeout
        self.api_key = api_key or os.environ.get("ORACLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("ORACLE_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("No API key found. Set ORACLE_API_KEY or OPENAI_API_KEY in .env")
        
        print(f"Judge initialized: model={self.model_name}, base_url={self.api_base}")
        self._init_client(timeout)
    
    def _init_client(self, timeout: float):
        """Initialize OpenAI client with specified timeout."""
        http_client = httpx.Client(timeout=httpx.Timeout(timeout, connect=60.0))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client
        )
        self.current_timeout = timeout

    def _call_api(self, prompt: str, temperature: float = 0.7, use_cache: bool = True) -> str:
        # Check cache first (only for deterministic calls with temp=0)
        cache_key = None
        if use_cache and temperature == 0.0:
            cache_key = _get_cache_key(self.model_name, prompt)
            cached_response = _load_from_cache(cache_key, "judge")
            if cached_response is not None:
                return cached_response
        
        result = self._call_with_retry(prompt, temperature)
        
        # Save to cache for deterministic calls
        if cache_key and result:
            _save_to_cache(cache_key, result, "judge")
        
        return result
    
    def _call_with_retry(self, prompt: str, temperature: float, max_retries: int = MAX_RETRIES) -> str:
        """Call API with exponential backoff on timeout."""
        current_timeout = self.base_timeout
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Reinitialize client with current timeout
                if attempt > 0:
                    print(f"Judge retry {attempt}/{max_retries} with timeout={current_timeout}s")
                    self._init_client(current_timeout)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except (httpx.TimeoutException, httpx.ReadTimeout, Exception) as e:
                last_error = e
                error_str = str(e).lower()
                if 'timeout' in error_str or 'timed out' in error_str or isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout)):
                    print(f"Judge timeout on attempt {attempt + 1}/{max_retries}: {e}")
                    # Exponential backoff: double timeout each retry
                    current_timeout = current_timeout * 2
                else:
                    # Non-timeout error, don't retry
                    print(f"Judge non-timeout error: {e}")
                    break
        
        print(f"Judge all retries failed. Last error: {last_error}")
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

    def evaluate_answer(self, question: str, answer: str, rubric: str = None, max_retries: int = 3) -> float:
        """
        Evaluates an answer and returns a scalar score (0-10).
        Retries on parsing errors.
        """
        if rubric:
            prompt = DYNAMIC_RUBRIC_EVALUATION_PROMPT.format(question=question, answer=answer, rubric=rubric)
        else:
            prompt = ANCHOR_EVALUATION_PROMPT.format(question=question, answer=answer)
        
        cache_key = _get_cache_key(self.model_name, prompt)
        
        for attempt in range(max_retries):
            # On retry, skip cache to get fresh response
            use_cache = (attempt == 0)
            response_text = self._call_api(prompt, temperature=0.0, use_cache=use_cache)
            
            score = self._parse_evaluation_response(response_text)
            if score is not None:
                # Valid response - update cache with good response
                if attempt > 0:
                    _save_to_cache(cache_key, response_text, "judge")
                return score
            
            if attempt < max_retries - 1:
                print(f"Retrying evaluation (attempt {attempt + 2}/{max_retries})...")
        
        print("All evaluation retries failed, returning 0.0")
        return 0.0
    
    def _parse_evaluation_response(self, response_text: str) -> Optional[float]:
        """Parse evaluation response and return score, or None if parsing fails."""
        if not response_text:
            return None
            
        try:
            # Attempt to find JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                
                # Fix common JSON escape issues
                # Replace problematic escape sequences
                import re
                # Fix invalid escapes like \_ \* etc by escaping the backslash
                json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try with strict=False for more lenient parsing
                    try:
                        result = json.loads(json_str, strict=False)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing evaluation: {e}")
                        return None
                
                # Handle both formats
                if "score" in result:
                    return float(result["score"])
                elif "overall" in result:
                    return float(result["overall"])
                else:
                    print("No score found in JSON")
                    return None
            else:
                # Try to extract score using regex as fallback
                import re
                score_match = re.search(r'"?(?:score|overall)"?\s*[:=]\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
                if score_match:
                    return float(score_match.group(1))
                print("Could not find JSON in evaluation response")
                return None
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return None

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

    def evaluate_batch(self, questions: List[str], answers: List[str], 
                       rubrics: List[Optional[str]] = None,
                       show_progress: bool = False, 
                       max_workers: int = None) -> List[float]:
        """Evaluate multiple answers in parallel."""
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS
        
        n = len(questions)
        if rubrics is None:
            rubrics = [None] * n
        
        results = [0.0] * n
        
        def evaluate_single(args: Tuple[int, str, str, Optional[str]]) -> Tuple[int, float]:
            idx, q, a, r = args
            return idx, self.evaluate_answer(q, a, rubric=r)
        
        tasks = list(zip(range(n), questions, answers, rubrics))
        
        if max_workers <= 1:
            # Sequential execution
            iterator = tasks
            if show_progress:
                iterator = tqdm(iterator, desc="  Evaluating", leave=False)
            for idx, q, a, r in iterator:
                results[idx] = self.evaluate_answer(q, a, rubric=r)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(evaluate_single, task): task[0] for task in tasks}
                
                if show_progress:
                    pbar = tqdm(total=n, desc="  Evaluating", leave=False)
                
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
        
        return results

# Alias for backward compatibility if needed, though we changed the class name
Oracle = Judge
