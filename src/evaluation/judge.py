import os
import json
import time
from typing import List, Dict, Any
from openai import OpenAI
from src.utils.prompts import REVERSE_ENGINEER_RUBRIC_PROMPT, ANCHOR_EVALUATION_PROMPT

class Oracle:
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None, api_base: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or os.environ.get("OPENAI_BASE_URL")
        )

    def _call_api(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Oracle API: {e}")
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

    def evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        prompt = ANCHOR_EVALUATION_PROMPT.format(question=question, answer=answer)
        response_text = self._call_api(prompt, temperature=0.0)
        
        try:
            # Attempt to find JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                print("Could not find JSON in evaluation response")
                return {}
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return {}
