import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

DATASET_PATH = "processed_dataset_new_prompt"

# Adapter/output folder
OUTPUT_DIR = "qwen_code_review_lora"

# IMPORTANT:
# This is only a temporary subset used for CPU demonstration.
# The original dataset on disk is NOT modified.
TRAIN_SAMPLES = 20

# Keep sequence length small for CPU
MAX_LENGTH = 512

SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("LIGHTWEIGHT LoRA TRAINING")
print("=" * 70)
print(f"Device: {device}")

if device == "cpu":
    print("CPU mode enabled.")
    print("Using a very small training subset for demonstration.")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

dataset = load_from_disk(DATASET_PATH)

print(dataset)
print()

print(f"Train:      {len(dataset['train'])}")
print(f"Validation: {len(dataset['validation'])}")
print(f"Test:       {len(dataset['test'])}")


# ============================================================
# TEMPORARY TRAINING SUBSET
# ============================================================
#
# We DO NOT modify the saved dataset.
#
# We select only 20 examples from the existing train split.
#
# Prefer examples whose original prompt length is <= 400
# so that the review has a better chance of fitting into
# the 512-token sequence.
# ============================================================

print()
print("=" * 70)
print("CREATING TEMPORARY CPU TRAINING SUBSET")
print("=" * 70)

train_data = dataset["train"]

# Prefer shorter prompts for CPU training
short_train = train_data.filter(
    lambda x: x["prompt_length"] <= 400
)

if len(short_train) >= TRAIN_SAMPLES:
    train_subset = short_train.shuffle(seed=SEED).select(
        range(TRAIN_SAMPLES)
    )
else:
    print("Not enough short examples. Using regular training data.")
    train_subset = train_data.shuffle(seed=SEED).select(
        range(min(TRAIN_SAMPLES, len(train_data)))
    )

print(f"Temporary training examples: {len(train_subset)}")
print("Original dataset remains unchanged.")


# ============================================================
# TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# Qwen does not always have a pad token configured
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


# ============================================================
# FORMAT DATA
# ============================================================

print()
print("Formatting dataset...")


def format_example(example):
    """
    Convert the messages stored in the dataset into
    a single text sequence using the Qwen chat template.
    """

    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }


formatted_train = train_subset.map(
    format_example,
    remove_columns=train_subset.column_names
)

print(f"Formatted examples: {len(formatted_train)}")


# ============================================================
# TOKENIZATION
# ============================================================

print()
print("Tokenizing dataset...")


def tokenize_function(example):

    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    # Standard causal language-model training:
    # labels are the same as input IDs.
    result["labels"] = result["input_ids"].copy()

    return result


tokenized_train = formatted_train.map(
    tokenize_function,
    batched=False,
    remove_columns=["text"]
)

print(f"Tokenized examples: {len(tokenized_train)}")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    trust_remote_code=True
)

model.to(device)

# Required when using gradient checkpointing / training
model.config.use_cache = False


# ============================================================
# APPLY LoRA
# ============================================================

print()
print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    # Lightweight target modules
    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

training_args = TrainingArguments(

    # Output
    output_dir=OUTPUT_DIR,

    # CPU-friendly batch size
    per_device_train_batch_size=1,

    # No accumulation for the tiny demo
    gradient_accumulation_steps=1,

    # One pass through the 20 examples
    num_train_epochs=1,

    # LoRA learning rate
    learning_rate=2e-4,

    # No warmup
    warmup_steps=0,

    # Logging
    logging_steps=1,

    # Save at end
    save_strategy="epoch",

    # CPU
    fp16=False,
    bf16=False,

    # Evaluation disabled for this quick CPU demo
    eval_strategy="no",

    # No external logging
    report_to="none",

    # Windows CPU
    dataloader_num_workers=0,
    dataloader_pin_memory=False,

    # Keep unused columns disabled
    remove_unused_columns=False,

    # Reproducibility
    seed=SEED
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Training now...")
print()

train_result = trainer.train()


# ============================================================
# TRAINING RESULTS
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(train_result)


# ============================================================
# SAVE LoRA ADAPTER
# ============================================================

print()
print("Saving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"LoRA adapter: {OUTPUT_DIR}")
print()


# ============================================================
# FINAL INFORMATION
# ============================================================

print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"Base model:       {MODEL_NAME}")
print(f"Training samples: {len(train_subset)}")
print(f"Max length:       {MAX_LENGTH}")
print(f"Epochs:           1")
print(f"Device:           {device}")
print(f"LoRA output:      {OUTPUT_DIR}")

print()
print("Next step:")
print("Run inference using the trained LoRA adapter.")
print("=" * 70)