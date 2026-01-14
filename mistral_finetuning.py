import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================

# Option A: Load from CSV/JSON
df = pd.read_csv("your_dataset.csv")  # Should have 'prompt' and 'response' columns
# Or: df = pd.read_json("your_dataset.jsonl", lines=True)

# Combine prompt and response for training
df['text'] = df['prompt'] + " " + df['response']

# Split into train and validation (e.g., 80-20 split)
train_texts, val_texts = train_test_split(
    df['text'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ============================================
# 2. LOAD MODEL AND TOKENIZER
# ============================================

model_name = "mistralai/Mistral-7B-v0.1"  # or other Mistral variants
# For instruction-tuned version: "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map="auto"  # Automatically distribute across available GPUs
)

# ============================================
# 3. TOKENIZE DATASETS
# ============================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512  # Adjust based on your needs
    )

# Tokenize datasets
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# ============================================
# 4. SETUP TRAINING ARGUMENTS
# ============================================

training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Simulate larger batch size
    save_steps=100,
    save_total_limit=2,  # Keep only 2 checkpoints
    eval_steps=100,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,  # Set to True if pushing to HuggingFace Hub
    # push_to_hub_model_id="your-username/mistral-finetuned",
    seed=42
)

# ============================================
# 5. DATA COLLATOR
# ============================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal language modeling (left-to-right)
)

# ============================================
# 6. INITIALIZE TRAINER
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# ============================================
# 7. START FINE-TUNING
# ============================================

print("Starting fine-tuning...")
trainer.train()

# ============================================
# 8. SAVE MODEL
# ============================================

model.save_pretrained("./mistral-finetuned-final")
tokenizer.save_pretrained("./mistral-finetuned-final")
print("Model and tokenizer saved!")

# ============================================
# 9. TEST INFERENCE (Optional)
# ============================================

# Load the fine-tuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(
    "./mistral-finetuned-final",
    torch_dtype=torch.float16,
    device_map="auto"
)
finetuned_tokenizer = AutoTokenizer.from_pretrained("./mistral-finetuned-final")

# Generate text
prompt = "Your test prompt here:"
inputs = finetuned_tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
outputs = finetuned_model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)
generated_text = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")