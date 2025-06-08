# 🔧 Fine-Tune Qwen3-0.6B with LoRA & Export to GGUF

This repository demonstrates how to fine-tune a Qwen3-0.6B (or similar) model using **LoRA**, then convert it to **GGUF** format for use with `llama.cpp`.

---

## 📁 Project Structure

.
├── data/
│   └── tool_use_train.jsonl     # Your prompt-response training data
├── train.py                     # LoRA fine-tuning script
├── requirements.txt             # Python dependencies for training
├── llama.cpp/
│   ├── convert_hf_to_gguf.py    # Script from llama.cpp
│   └── requirements.txt         # Separate env for llama.cpp
└── qwen3-0.6b-lora/         # Output folder after training & GGUF conversion

---

## ⚙️ 1. Environment Setup & Training

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3 train.py

✅ This will fine-tune the model using your data/tool_use_train.jsonl file
✅ Outputs will be saved to train/qwen3-0.6b-lora/

⸻

🔄 2. Convert to GGUF (for llama.cpp)

cd llama.cpp

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3.11 convert_hf_to_gguf.py \
  ../train/qwen3-0.6b-lora \
  --outtype f16 \
  --outfile ../train/qwen3-0.6b-lora/model.gguf

🎉 You’ll get a model.gguf file ready to use in llama.cpp, koboldcpp, llamafile, etc.

⸻

📌 Notes
	•	Make sure you have access to the base model (Qwen/Qwen1.5-0.5B, mistralai/Mistral-7B, etc.) on Hugging Face
	•	If the tokenizer throws an error about pad_token, you can fix it in code with:

tokenizer.pad_token = tokenizer.eos_token



⸻

📦 Example requirements.txt

transformers
datasets
peft
accelerate
sentencepiece
protobuf


⸻

🧠 Reference
	•	PEFT - LoRA Training
	•	llama.cpp GGUF Format

---