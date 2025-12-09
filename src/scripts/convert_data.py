import pandas as pd
import json
import os

def convert():
    input_path = "data/synthetic_rubrics.jsonl"
    output_path = "data/synthetic_rubrics.parquet"
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    data = []
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            question = item['question']
            rubric_list = item['rubric']
            
            # Format prompt as chat messages
            prompt_messages = [
                {"role": "user", "content": f"Generate a detailed rubric for evaluating the following instruction:\n\n{question}"}
            ]
            
            # Format response
            # Join rubric items
            rubric_text = "\n".join([f"- {r}" for r in rubric_list])
            
            data.append({
                "prompt": prompt_messages,
                "response": rubric_text,
                "question": question,
                "gold_answer": item['gold_answer']
            })
            
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Converted {len(df)} items to {output_path}")

if __name__ == "__main__":
    convert()
