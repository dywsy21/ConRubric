import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src.config import ProjectConfig

class RubricDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        rubric_list = item['rubric']
        rubric_text = "\n".join([f"- {r}" for r in rubric_list])
        
        # Format: User: <q> \n Assistant: <rubric>
        # This is a simplified format. In reality, use the model's chat template.
        prompt = f"User: {question}\n\nPlease generate a rubric for this question.\n\nAssistant: "
        full_text = prompt + rubric_text
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # Mask out the prompt in labels so we don't train on it
        # This is a naive implementation; robust implementation would find the prompt length
        # For now, we just train on everything or assume the collator handles it if we used DataCollatorForCompletionOnlyLM
        
        return {
            "input_ids": input_ids,
            "attention_mask": encodings['attention_mask'].squeeze(),
            "labels": labels
        }

def train_sft():
    config = ProjectConfig()
    
    print(f"Loading model: {config.grm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.grm_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config.grm_model_name, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    data_path = os.path.join(config.data_dir, "synthetic_rubrics.jsonl")
    dataset = RubricDataset(data_path, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "sft_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        save_steps=500,
        logging_steps=100,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )
    
    print("Starting SFT...")
    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, "sft_final"))

if __name__ == "__main__":
    train_sft()
