import os
import hashlib
import json
import logging
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
import httpx

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Default parallel settings
DEFAULT_MAX_WORKERS = int(os.environ.get("GRM_SOLVER_WORKERS", 8))

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

# Default timeout settings
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3

class Solver:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 is_remote: bool = False, api_key: str = None, api_base: str = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.model_name = model_name
        self.is_remote = is_remote
        self.base_timeout = timeout
        self.api_key = api_key or os.environ.get("SOLVER_API_KEY")
        self.api_base = api_base or os.environ.get("SOLVER_API_BASE")
        
        if self.is_remote:
            print(f"Initializing Remote Solver: {model_name}")
            self._init_client(timeout)
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
    
    def _init_client(self, timeout: float):
        """Initialize OpenAI client with specified timeout."""
        http_client = httpx.Client(timeout=httpx.Timeout(timeout, connect=60.0))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client
        )
        self.current_timeout = timeout
    
    def _call_with_retry(self, prompt: str, max_retries: int = MAX_RETRIES) -> str:
        """Call API with exponential backoff on timeout."""
        current_timeout = self.base_timeout
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Reinitialize client with current timeout
                if attempt > 0:
                    print(f"Solver retry {attempt}/{max_retries} with timeout={current_timeout}s")
                    self._init_client(current_timeout)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except (httpx.TimeoutException, httpx.ReadTimeout, Exception) as e:
                last_error = e
                error_str = str(e).lower()
                if 'timeout' in error_str or 'timed out' in error_str or isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout)):
                    print(f"Solver timeout on attempt {attempt + 1}/{max_retries}: {e}")
                    # Exponential backoff: double timeout each retry
                    current_timeout = current_timeout * 2
                else:
                    # Non-timeout error, don't retry
                    print(f"Solver non-timeout error: {e}")
                    break
        
        print(f"Solver all retries failed. Last error: {last_error}")
        return ""

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
            answer = self._call_with_retry(prompt)
            if answer:
                # Save to cache
                _save_to_cache(cache_key, answer, "solver", 
                              {'model': self.model_name, 'question': question[:100], 'rubric': rubric[:100]})
            return answer
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

    def generate_batch(self, questions: List[str], rubrics: List[str], 
                       show_progress: bool = False, max_workers: int = None) -> List[str]:
        """Generate answers in parallel using thread pool."""
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS if self.is_remote else 1
        
        n = len(questions)
        results = [""] * n
        
        def generate_single(args: Tuple[int, str, str]) -> Tuple[int, str]:
            idx, q, r = args
            return idx, self.generate_answer(q, r)
        
        if max_workers <= 1 or not self.is_remote:
            # Sequential execution for local models
            iterator = zip(range(n), questions, rubrics)
            if show_progress:
                iterator = tqdm(iterator, total=n, desc="  Generating", leave=False)
            for idx, q, r in iterator:
                results[idx] = self.generate_answer(q, r)
        else:
            # Parallel execution for remote API
            tasks = list(zip(range(n), questions, rubrics))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(generate_single, task): task[0] for task in tasks}
                
                if show_progress:
                    pbar = tqdm(total=n, desc="  Generating", leave=False)
                
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
        
        return results
