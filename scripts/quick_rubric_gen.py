#!/usr/bin/env python3
"""Quick rubric generation for checkpoint evaluation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "out/rl/global_step_50/actor/huggingface"
print(f"Loading model from {model_path}...")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

questions = [
    "What are the main causes of climate change?",
    "Write a Python function that sorts a list using merge sort.",
    "Explain the difference between TCP and UDP protocols.",
    "How should a patient with type 2 diabetes manage their diet?",
]

prompt_tpl = (
    "Generate an evaluation rubric of 6-12 criteria for the following question. "
    "Include BOTH positive criteria (what the answer should do) AND negative criteria "
    "(common mistakes or harmful behaviors to penalize). Do not repeat or paraphrase "
    "the same idea across multiple criteria.\n\n"
    "Output each criterion on its own line in this format:\n"
    "- [+/-points] criterion | tags: ...\n\n"
    "Question:\n{question}"
)

sep = "=" * 60
for i, q in enumerate(questions):
    msgs = [{"role": "user", "content": prompt_tpl.format(question=q)}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True)
    resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{sep}\nQ{i+1}: {q}\n{sep}")
    print(resp[:2000])
    print()
