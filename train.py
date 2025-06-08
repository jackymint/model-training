from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import json

# โหลดไฟล์ finetune-data.txt (เป็น JSON list แบบ [{"instruction": ..., "input": ..., "output": ...}, ...])
with open("finetune-data.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

# สร้าง Dataset จาก list ของ dict
dataset = Dataset.from_list(data)

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    prompt = [
        (inst + " " + inp).strip() if inp else inst
        for inst, inp in zip(examples["instruction"], examples.get("input", [""] * len(examples["instruction"])))
    ]
    max_seq_length = 128  # กำหนดขนาดเท่ากันทั้ง input และ output

    model_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_seq_length)

    labels = tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

    # แปลงค่า pad token เป็น -100 เพื่อไม่ให้คำนวณ loss กับ padding token
    labels_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs

# map preprocess function
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# โหลด model
model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./Qwen/Qwen2.5-0.5B-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    # fp16=True,  # ถ้าใช้ GPU CUDA เท่านั้น (Mac/CPU ปิดไว้ก่อน)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# เริ่มเทรน
trainer.train()

# บันทึก model ที่เทรนแล้ว
model.save_pretrained("./Qwen/Qwen2.5-0.5B-finetuned")
