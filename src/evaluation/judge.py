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
DEFAULT_MAX_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 4))
from src.utils.prompts import REVERSE_ENGINEER_RUBRIC_PROMPT, DYNAMIC_RUBRIC_EVALUATION_PROMPT, BATCH_RUBRIC_EVALUATION_PROMPT

# Global cache directory
CACHE_DIR = Path(os.environ.get("GRM_CACHE_DIR", "./out/cache"))
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
    
    def _call_with_retry(self, prompt: str, temperature: float, max_retries: int = MAX_RETRIES,
                         extra_body: dict = None, max_tokens: int = None) -> str:
        """Call API with exponential backoff on timeout."""
        current_timeout = self.base_timeout
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Reinitialize client with current timeout
                if attempt > 0:
                    print(f"Judge retry {attempt}/{max_retries} with timeout={current_timeout}s")
                    self._init_client(current_timeout)
                
                kwargs = dict(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                
                response = self.client.chat.completions.create(**kwargs)
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

    def reverse_engineer_rubric(self, question: str, gold_answer: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Reverse-engineer rubric with signed criterion points.
        Returns list of {criterion, points, tags} where points in [-10, 10], points != 0,
        and at least one positive + one negative item whenever possible.
        """
        base_prompt = REVERSE_ENGINEER_RUBRIC_PROMPT.format(question=question, gold_answer=gold_answer)
        prompt = base_prompt

        for attempt in range(max_retries):
            response_text = self._call_api(prompt, temperature=0.4, use_cache=False)
            rubric = self._parse_reverse_rubric_response(response_text)
            if rubric:
                return rubric

            # Retry with stricter instruction if model failed formatting/sign-balance
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: You must return a JSON array with BOTH positive and negative items. "
                + "Use integer points in [-10, 10], excluding 0."
            )

        return rubric if rubric else []

    def _parse_reverse_rubric_response(self, response_text: str) -> List[Dict[str, Any]]:
        if not response_text:
            return []

        try:
            # Strip <think>...</think> blocks (Qwen3-style chain-of-thought)
            cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            # Strip markdown fences
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

            # Attempt JSON array extraction
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            parsed = None
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = None

            out: List[Dict[str, Any]] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        criterion = str(item.get("criterion", "")).strip()
                        if not criterion:
                            continue
                        try:
                            points = int(item.get("points", 0))
                        except Exception:
                            continue
                        points = max(-10, min(10, points))
                        if points == 0:
                            continue
                        tags = item.get("tags", [])
                        if not isinstance(tags, list):
                            tags = []
                        out.append({"criterion": criterion, "points": points, "tags": tags})
                    elif isinstance(item, str):
                        # Fallback for old string-only format: assign +1
                        criterion = item.strip()
                        if criterion:
                            out.append({"criterion": criterion, "points": 1, "tags": []})

            if out:
                return out

            # Text fallback: parse lines like "- [+3] ..." / "- [-4] ..."
            lines = cleaned.split('\n')
            for line in lines:
                line_strip = line.strip()
                if not line_strip:
                    continue
                m = re.search(r"\[\s*([+-]?\d+)\s*\]", line_strip)
                if not m:
                    continue
                points = int(m.group(1))
                points = max(-10, min(10, points))
                if points == 0:
                    continue
                criterion = re.sub(r"^[-*\d\.\s]*", "", line_strip)
                criterion = re.sub(r"\[\s*[+-]?\d+\s*\]", "", criterion).strip(" -:")
                if criterion:
                    out.append({"criterion": criterion, "points": points, "tags": []})

            return out
        except Exception as e:
            print(f"Error parsing rubric: {e}")
            return []

    def evaluate_answer(self, question: str, answer: str, rubric: str = None, max_retries: int = 3) -> float:
        """
        Evaluates an answer and returns a scalar score (0-10).
        Uses vLLM guided_choice + disable thinking for near-zero parse failures.
        Falls back to regex parsing for non-vLLM backends.
        """
        if not rubric or not rubric.strip():
            print("Warning: empty rubric passed to evaluate_answer, returning 0.0")
            return 0.0

        prompt = DYNAMIC_RUBRIC_EVALUATION_PROMPT.format(question=question, answer=answer, rubric=rubric)
        
        cache_key = _get_cache_key(self.model_name, prompt)
        
        # vLLM guided output: disable thinking + constrain to 0-10 integer
        guided_extra = {
            "chat_template_kwargs": {"enable_thinking": False},
            "guided_choice": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        }
        
        for attempt in range(max_retries):
            use_cache = (attempt == 0)
            
            # First attempt: try guided output (vLLM)
            # Later attempts: fall back to unguided if guided fails
            extra = guided_extra if attempt == 0 else {"chat_template_kwargs": {"enable_thinking": False}}
            max_tok = 8 if attempt == 0 else 256
            
            try:
                response_text = self._call_api_direct(
                    prompt, temperature=0.0, use_cache=use_cache,
                    extra_body=extra, max_tokens=max_tok,
                )
            except Exception:
                # guided_choice not supported — fall back to plain call
                response_text = self._call_api(prompt, temperature=0.0, use_cache=use_cache)
            
            score = self._parse_evaluation_response(response_text)
            if score is not None:
                if attempt > 0:
                    _save_to_cache(cache_key, response_text, "judge")
                return score
            
            if attempt < max_retries - 1:
                print(f"Retrying evaluation (attempt {attempt + 2}/{max_retries})...")
        
        print("All evaluation retries failed, returning 0.0")
        return 0.0

    def _call_api_direct(self, prompt: str, temperature: float = 0.0,
                         use_cache: bool = True, extra_body: dict = None,
                         max_tokens: int = None) -> str:
        """Call API with extra_body support (for vLLM guided decoding)."""
        cache_key = None
        if use_cache and temperature == 0.0:
            cache_key = _get_cache_key(self.model_name, prompt)
            cached_response = _load_from_cache(cache_key, "judge")
            if cached_response is not None:
                return cached_response

        result = self._call_with_retry(
            prompt, temperature, extra_body=extra_body, max_tokens=max_tokens,
        )

        if cache_key and result:
            _save_to_cache(cache_key, result, "judge")

        return result
    
    def _parse_evaluation_response(self, response_text: str) -> Optional[float]:
        """Parse evaluation response and return score, or None if parsing fails.
        
        Handles multiple formats (in priority order):
        1. Bare integer from guided_choice: "7"
        2. "Score: N" pattern
        3. JSON {"score": N}
        4. "N/10" pattern
        5. Last bare integer 0-10
        """
        if not response_text:
            return None
            
        try:
            # Strip <think>...</think> blocks (Qwen3-style chain-of-thought)
            cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = response_text.strip()

            # 1. Bare integer (from guided_choice)
            if cleaned in ("0","1","2","3","4","5","6","7","8","9","10"):
                return float(cleaned)

            # 2. "Score: N" pattern (take LAST match — model may discuss scores before final)
            score_matches = re.findall(r'[Ss]core\s*[:=]\s*(\d+(?:\.\d+)?)', cleaned)
            if score_matches:
                val = float(score_matches[-1])
                return max(0.0, min(10.0, val))

            # 3. JSON {"score": N} or {"overall": N}
            start = cleaned.find('{')
            if start != -1:
                depth = 0
                end = start
                for i in range(start, len(cleaned)):
                    if cleaned[i] == '{':
                        depth += 1
                    elif cleaned[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = cleaned[start:end]
                json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
                try:
                    result = json.loads(json_str)
                    if "score" in result:
                        return max(0.0, min(10.0, float(result["score"])))
                    elif "overall" in result:
                        return max(0.0, min(10.0, float(result["overall"])))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            # 4. "N/10" pattern
            slash_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', cleaned)
            if slash_match:
                return max(0.0, min(10.0, float(slash_match.group(1))))

            # 5. Last bare integer 0-10 in text
            bare_matches = re.findall(r'\b(\d+)\b', cleaned)
            for candidate in reversed(bare_matches):
                val = int(candidate)
                if 0 <= val <= 10:
                    return float(val)

            print("Could not extract score from evaluation response")
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
        """Evaluate multiple answers in parallel (each scored independently)."""
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
            iterator = tasks
            if show_progress:
                iterator = tqdm(iterator, desc="  Evaluating", leave=False)
            for idx, q, a, r in iterator:
                results[idx] = self.evaluate_answer(q, a, rubric=r)
        else:
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

    def evaluate_answers_by_rubric(
        self,
        question: str,
        answers: List[str],
        rubric: str,
        max_retries: int = 3,
    ) -> List[float]:
        """Score N answers against ONE rubric in a single LLM call.

        The judge sees all answers side-by-side, producing *relative* scores
        that are calibrated against each other.  This is used by the RL
        meta-reward so that the cross-rubric scoring matrix carries genuine
        ranking information.

        Returns a list of N floats in [0, 10].
        """
        if not rubric or not rubric.strip():
            print("Warning: empty rubric in evaluate_answers_by_rubric, returning zeros")
            return [0.0] * len(answers)

        n = len(answers)
        if n == 0:
            return []
        if n == 1:
            return [self.evaluate_answer(question, answers[0], rubric=rubric)]

        answers_block = "\n\n".join(
            f"--- Answer {i+1} ---\n{a}" for i, a in enumerate(answers)
        )
        # Generate a dynamic example that always has exactly n elements
        _eg = [8, 3, 6, 5, 7, 2, 9, 4][:n]
        example_scores = "[" + ", ".join(str(x) for x in _eg) + "]"
        prompt = BATCH_RUBRIC_EVALUATION_PROMPT.format(
            n=n, question=question, rubric=rubric, answers_block=answers_block,
            example_scores=example_scores,
        )

        # vLLM guided JSON: array of n integers 0-10
        guided_extra = {
            "chat_template_kwargs": {"enable_thinking": False},
            "guided_json": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 10},
                "minItems": n,
                "maxItems": n,
            },
        }

        for attempt in range(max_retries):
            extra = guided_extra if attempt == 0 else {"chat_template_kwargs": {"enable_thinking": False}}
            max_tok = 4 * n + 16 if attempt == 0 else 256
            
            try:
                response_text = self._call_api_direct(
                    prompt, temperature=0.0, use_cache=(attempt == 0),
                    extra_body=extra, max_tokens=max_tok,
                )
            except Exception:
                response_text = self._call_api(prompt, temperature=0.0, use_cache=(attempt == 0))
            
            scores = self._parse_batch_scores(response_text, n)
            if scores is not None:
                return scores
            if attempt < max_retries - 1:
                print(f"Retrying batch evaluation (attempt {attempt + 2}/{max_retries})...")

        print(f"All batch-eval retries failed (n={n}), falling back to individual eval")
        # Fallback: score each answer individually
        return [self.evaluate_answer(question, a, rubric=rubric) for a in answers]

    def _parse_batch_scores(self, response_text: str, expected_n: int) -> Optional[List[float]]:
        """Parse a JSON array of N scores from batch evaluation response."""
        if not response_text:
            return None
        try:
            cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = response_text.strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            if start == -1 or end <= start:
                # Fallback: try to find N integers in the text
                nums = re.findall(r'\b(\d+)\b', cleaned)
                valid = [int(x) for x in nums if 0 <= int(x) <= 10]
                if len(valid) == expected_n:
                    return [float(v) for v in valid]
                return None

            arr = json.loads(cleaned[start:end])
            if not isinstance(arr, list) or len(arr) != expected_n:
                print(f"Batch scores length mismatch: got {len(arr) if isinstance(arr, list) else 'non-list'}, expected {expected_n}")
                return None

            scores = []
            for v in arr:
                s = float(v)
                scores.append(max(0.0, min(10.0, s)))
            return scores
        except Exception as e:
            print(f"Error parsing batch scores: {e}")
            return None

# Alias for backward compatibility if needed, though we changed the class name
Oracle = Judge
