#!/usr/bin/env python
# perspectief_train_llama3.py  –  works on Unsloth 2025.5

# python perspectief_training.py --output_dir checkpoints/perspectief-llama3

from __future__ import annotations
# Unsloth must be first
from unsloth import FastLanguageModel
import argparse, torch
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_PATH  = "Scrape-website/perspectief_dataset-1.jsonl"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
SPECIALS   = ["<|user|>", "<|assistant|>", "<|end|>"]

# ── CLI ────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--output_dir", required=True)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch",  type=int, default=2)
ap.add_argument("--lr",     type=float, default=2e-5)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
bnb   = BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=dtype)

# ── load model & tokenizer with FA-2 disabled (uses SDPA/xFormers) ─
model, tok = FastLanguageModel.from_pretrained(
    model_name  = BASE_MODEL,
    dtype       = dtype,
    quantization_config = bnb,
    device_map  = "auto",
)

tok.add_special_tokens({"additional_special_tokens": SPECIALS})
tok.pad_token = tok.eos_token
model.resize_token_embeddings(len(tok))

# ── add LoRA adapters — pass ints, not LoraConfig ─
model = FastLanguageModel.get_peft_model(
    model,
    r             = 64,
    lora_alpha    = 16,
    lora_dropout  = 0.05,
    target_modules= ["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
)
model.print_trainable_parameters()

# ── dataset ───────────────────────────────────────────────────────
def to_prompt(ex):
    return {"text": f"<|user|>\n{ex['Question']}\n\nContext:\n{ex['DetailedContext']}"
                    f"\n<|assistant|>\n{ex['Answer']}<|end|>"}

raw   = load_dataset("json", data_files=DATA_PATH, split="train")
assert len(raw) > 0, f"No rows found at {DATA_PATH}"
ds    = raw.map(to_prompt, remove_columns=raw.column_names)
split = ds.train_test_split(test_size=0.1, seed=42)

def tok_fn(batch):
    t = tok(batch["text"], truncation=True, padding="max_length", max_length=768)
    t["labels"] = t["input_ids"].copy(); return t

train_ds = split["train"].map(tok_fn, batched=True, remove_columns=["text"])
valid_ds = split["test"].map(tok_fn,  batched=True, remove_columns=["text"])

# ── training args (OOM-safe) ──────────────────────────────────────
ta = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=4,     # eff batch 8
    learning_rate=args.lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=(dtype==torch.bfloat16),
    logging_steps=25,
    eval_strategy="no",
    save_strategy="no",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model, tokenizer=tok, args=ta,
    train_dataset=train_ds, eval_dataset=valid_ds)
trainer.train()
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)
print(f"✅ Training complete — checkpoint saved to {args.output_dir}")
