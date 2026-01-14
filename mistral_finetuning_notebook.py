"""
Fine-tune Mistral LLM using Hugging Face Transformers
Optimized for Azure AI Studio - Standard_NC4as_T4_v3 (Single T4 GPU with 16GB VRAM)
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

# ⚠️ OPTIMIZED FOR SINGLE T4 GPU (16GB VRAM)
BATCH_SIZE = 1  # Reduced for single T4 GPU
GRADIENT_ACCUMULATION_STEPS = 8  # Compensate for small batch size
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
USE_8BIT = True  # Enable 8-bit quantization to save VRAM
USE_LORA = True  # Enable LoRA for efficient fine-tuning

print("="*60)
print("Mistral-7B Fine-tuning on Standard_NC4as_T4_v3")
print("="*60)
print(f"GPU: T4 (16GB VRAM)")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"8-bit Quantization: {USE_8BIT}")
print(f"LoRA: {USE_LORA}")
print("="*60)

# =======================
# 2. Install Dependencies (if needed)
# =======================

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Install additional dependencies for T4 optimization
print("\nInstalling optimized dependencies...")
install_if_missing('bitsandbytes')
install_if_missing('peft')
print("✓ Dependencies ready")

# =======================
# 3. Load and Prepare Dataset
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
print("\nLoading dataset...")
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
        {"instruction": "Define neural networks", "output": "Neural networks are computational models inspired by biological brains, consisting of interconnected nodes that process information."},
        {"instruction": "What is transfer learning?", "output": "Transfer learning applies knowledge from one task to improve learning on a related but different task."},
    ]

# =======================
# 4. Format Dataset
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
print(f"\nSample text:")
print(f"{dataset_df['text'].iloc[0]}\n")

# =======================
# 5. Split into Train and Validation
# =======================

train_df, val_df = train_test_split(
    dataset_df,
    test_size=VAL_SPLIT,
    random_state=42
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}\n")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset = Dataset.from_pandas(val_df[['text']])

# =======================
# 6. Load Tokenizer and Model
# =======================

print(f"Loading model: {MODEL_NAME}")
print("This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=USE_8BIT,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("✓ Model and tokenizer loaded")
print(f"Model dtype: {model.dtype}")

# =======================
# 7. Setup LoRA for Efficient Fine-tuning
# =======================

if USE_LORA:
    from peft import get_peft_model, LoraConfig, TaskType
    
    print("\nSetting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Target Mistral attention modules
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("✓ LoRA enabled")

# =======================
# 8. Tokenize Dataset
# =======================

def tokenize_function(examples):
    """Tokenize the input texts"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=True
    )

print("\nTokenizing datasets...")
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
# 9. Set Training Arguments (Optimized for T4)
# =======================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    # Batch and gradient settings
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # Training schedule
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    
    # Logging and evaluation
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    
    # Memory and precision optimization for T4
    fp16=True,  # Mixed precision training
    tf32=False,  # Disable TF32 on T4
    gradient_checkpointing=True,  # Save memory at cost of speed
    
    # Remove old checkpoints to save space
    save_total_limit=2,
    
    seed=42,
    report_to="none"  # Disable wandb logging
)

print("✓ Training arguments configured for T4 GPU")

# =======================
# 10. Create Trainer
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

print("✓ Trainer initialized\n")

# =======================
# 11. Fine-tune Model
# =======================

print("="*60)
print("STARTING FINE-TUNING")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("✓ FINE-TUNING COMPLETE!")
print("="*60)

# =======================
# 12. Save Model
# =======================

print(f"\nSaving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved successfully")

# =======================
# 13. Evaluate
# =======================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"\nValidation Loss: {eval_results['eval_loss']:.4f}")

# =======================
# 14. Test Inference
# =======================

print("\n" + "="*60)
print("INFERENCE TEST")
print("="*60)

def generate_response(prompt, max_length=100):
    """Generate text from fine-tuned model"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
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
    "Instruction: What is machine learning?",
    "Instruction: Explain transfer learning"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = generate_response(prompt)
    print(f"Response: {response}\n")

print("="*60)
print("✓ FINE-TUNING PIPELINE COMPLETE!")
print("="*60)
print("\nYour fine-tuned model is ready for deployment!")
print(f"Location: {OUTPUT_DIR}")
