import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Login to Hugging Face
login()  # Will prompt for token if not already logged in

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="mps" if torch.backends.mps.is_available() else "auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration optimized for Mistral
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Load and preprocess dataset
dataset = load_dataset("wikisql", split="train[:1%]")


def preprocess_function(examples):
    model_inputs = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    bf16=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
