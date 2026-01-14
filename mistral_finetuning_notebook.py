"""
Fine-tune Mistral LLM using Hugging Face Transformers
For Azure AI Studio
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
import torch

# =======================
# 1. Configuration
# =======================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./mistral_finetuned"
TRAIN_FILE = "./data/training_data.json"  # Update with your file path
VAL_SPLIT = 0.1  # 10% for validation
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

print("✓ Configuration loaded")

# =======================
# 2. Load and Prepare Dataset
# =======================

def load_json_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_csv_dataset(file_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(file_path)
    data = df.to_dict('records')
    return data

# Load your dataset (adjust based on your file format)
print("Loading dataset...")
try:
    # For JSON files with list of objects
    dataset = load_json_dataset(TRAIN_FILE)
except FileNotFoundError:
    print(f"File not found: {TRAIN_FILE}")
    print("Creating sample dataset for demonstration...")
    dataset = [
        {"instruction": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
        {"instruction": "Explain deep learning", "output": "Deep learning uses neural networks with multiple layers to process complex patterns in data."},
        {"instruction": "What is NLP?", "output": "Natural Language Processing is the field of AI focused on understanding and processing human language."},
    ]

# =======================
# 3. Format Dataset
# =======================

def format_prompt(example):
    """Format instruction-output pairs into a single text"""
    if 'instruction' in example and 'output' in example:
        text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
    elif 'text' in example:
        text = example['text']
    else:
        text = str(example)
    return {"text": text}

# Convert to dataset and format
formatted_dataset = [format_prompt(item) for item in dataset]
dataset_df = pd.DataFrame(formatted_dataset)

print(f"Total samples: {len(dataset_df)}")
print(f"Sample text:\n{dataset_df['text'].iloc[0]}\n")

# =======================
# 4. Split into Train and Validation
# =======================

train_df, val_df = train_test_split(
    dataset_df,
    test_size=VAL_SPLIT,
    random_state=42
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset = Dataset.from_pandas(val_df[['text']])

# =======================
# 5. Load Tokenizer and Model
# =======================

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model and tokenizer loaded")

# =======================
# 6. Tokenize Dataset
# =======================

def tokenize_function(examples):
    """Tokenize the input texts"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=True
    )

print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)
val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

print("✓ Tokenization complete")

# =======================
# 7. Set Training Arguments
# =======================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    gradient_accumulation_steps=2,
    fp16=True,
    seed=42
)

print("✓ Training arguments configured")

# =======================
# 8. Create Trainer
# =======================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

print("✓ Trainer initialized")

# =======================
# 9. Fine-tune Model
# =======================

print("\nStarting fine-tuning...")
trainer.train()

print("✓ Fine-tuning complete!")

# =======================
# 10. Save Model
# =======================

print(f"\nSaving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved successfully")

# =======================
# 11. Evaluate
# =======================

print("\nEvaluating on validation set...")
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"Evaluation results: {eval_results}")

# =======================
# 12. Test Inference
# =======================

print("\n" + "="*50)
print("Testing inference on fine-tuned model")
print("="*50)

def generate_response(prompt, max_length=100):
    """Generate text from fine-tuned model"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test prompts
test_prompts = [
    "Instruction: What is artificial intelligence?",
    "Instruction: Explain neural networks"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = generate_response(prompt)
    print(f"Response: {response}\n")

print("\n✓ Fine-tuning pipeline complete!")
