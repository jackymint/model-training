import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# ---------- Load dataset ----------
with open("data/tool_use_train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

dataset = Dataset.from_list(data)

# ---------- Load tokenizer ----------
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ---------- Preprocessing ----------
def preprocess_function(examples):
    prompts = examples["prompt"]
    responses = examples["response"]
    max_seq_length = 512

    # Combine prompt + response
    inputs = [p + "\n" + r for p, r in zip(prompts, responses)]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

    # Make labels = input_ids, with padding tokens as -100
    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in input_ids]
        for input_ids in model_inputs["input_ids"]
    ]

    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["prompt", "response"]
)
# ---------- Load model ----------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# ---------- LoRA Config ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- Training ----------
training_args = TrainingArguments(
    output_dir="./qwen3-0.6b-lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ---------- Start training ----------
trainer.train()

# ---------- Save LoRA only ----------
model.save_pretrained("./qwen3-0.6b-lora")
tokenizer.save_pretrained("./qwen3-0.6b-lora")

# Load base Qwen3 model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

# Apply LoRA weights
lora_model = PeftModel.from_pretrained(base, "./qwen3-0.6b-lora")

# # Merge LoRA into base
# merged = lora_model.merge_and_unload()

# # Save merged model to a new folder
# merged.save_pretrained("./qwen3-0.6b-lora")
