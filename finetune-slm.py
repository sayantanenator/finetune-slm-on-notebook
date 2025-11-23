"""
Fine-Tuning an Open-Source Small Language Model (SLM)
using Unsloth + LoRA on Google Colab or Local Machine.

Author: Your Name
"""

# ============================================================
# 0. (Optional) Install Dependencies – uncomment for Colab
# ============================================================
# !pip install unsloth accelerate transformers datasets bitsandbytes safetensors -q

import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, pipeline


# ============================================================
# 1. Load Base SLM in 4-bit
# ============================================================
BASE_MODEL = "unsloth/tinyllama-1.1b-chat"

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=2048,
)

# Enable LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

print("Model loaded with LoRA enabled.")


# ============================================================
# 2. Load & Preprocess Dataset
# ============================================================
DATA_FILE = "data.json"   # your instruction dataset

dataset = load_dataset("json", data_files=DATA_FILE)


def format_sample(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    }


formatted_dataset = dataset.map(format_sample)


def tokenize(x):
    return tokenizer(
        x["text"],
        truncation=True,
        max_length=2048,
    )


tokenized = formatted_dataset["train"].map(tokenize)
print("Dataset tokenized.")


# ============================================================
# 3. Training Configuration
# ============================================================
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=2,
    logging_steps=10,
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_available(),
)

# ============================================================
# 4. Train the Model
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
print("Fine-tuning completed.")


# ============================================================
# 5. Save Fine-Tuned Model
# ============================================================
SAVE_DIR = "finetuned-slm"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved at: {SAVE_DIR}")


# ============================================================
# 6. Export LoRA-Only Weights
# ============================================================
LORA_DIR = "lora-only"
model.save_pretrained(LORA_DIR)
print(f"LoRA adapters saved at: {LORA_DIR}")


# ============================================================
# 7. Convert LoRA Weights to Safetensors
# ============================================================
bin_path = os.path.join(LORA_DIR, "adapter_model.bin")
sf_path = os.path.join(LORA_DIR, "adapter_model.safetensors")

if os.path.exists(bin_path):
    state_dict = torch.load(bin_path)
    save_file(state_dict, sf_path)
    os.remove(bin_path)
    print(f"LoRA adapter converted to: {sf_path}")
else:
    print("No adapter_model.bin found; skipping safetensor conversion.")


# ============================================================
# 8. Inference Test (Single Prompt)
# ============================================================
print("\n================ INFERENCE TEST ================")

ft_model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR,
    torch_dtype="auto",
    device_map="auto",
)

ft_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

pipe = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=ft_tokenizer,
    max_new_tokens=150,
)

test_prompt = """### Instruction:
How does a smart meter detect abnormal spikes in energy usage?

### Response:
"""

out = pipe(test_prompt)[0]["generated_text"]
print(out)


# ============================================================
# 9. Batch Evaluation
# ============================================================
print("\n================ BATCH EVALUATION ================")

eval_prompts = [
    "Explain load imbalance detection in smart meters.",
    "How do smart meters handle voltage fluctuations?",
    "Describe methods to detect reverse energy flow."
]

for p in eval_prompts:
    formatted = f"### Instruction:\n{p}\n\n### Response:\n"
    answer = pipe(formatted)[0]["generated_text"]
    print("\nPrompt:", p)
    print("Answer:", answer)
    print("-" * 60)

print("All tests completed.")
