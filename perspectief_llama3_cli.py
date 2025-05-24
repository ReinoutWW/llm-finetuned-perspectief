#!/usr/bin/env python
# perspectief_train.py  –  OOM-safe version
from __future__ import annotations
import argparse, torch, numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

DATA_PATH  = "Scrape-website/perspectief_dataset-1.jsonl"     # path as you have it
BASE_MODEL = "facebook/opt-1.3b"
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--output_dir", required=True)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch",  type=int, default=2)   # per-device
ap.add_argument("--lr",     type=float, default=1e-4)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
bnb   = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=dtype)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, torch_dtype=dtype,
    device_map="auto", use_safetensors=True)
model.resize_token_embeddings(len(tok))
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        task_type="CAUSAL_LM"))

# -------- dataset ---------
def to_prompt(ex):
    return {"text": f"<|user|>\n{ex['Question']}\n\nContext:\n{ex['DetailedContext']}"
                    f"\n<|assistant|>\n{ex['Answer']}<|end|>"}

raw = load_dataset("json", data_files=DATA_PATH, split="train")
assert len(raw) > 0, f"No rows found at {DATA_PATH}"
ds   = raw.map(to_prompt, remove_columns=raw.column_names)
split = ds.train_test_split(test_size=0.1, seed=42)

def tok_fn(b):
    t = tok(b["text"], truncation=True, padding="max_length", max_length=512)
    t["labels"] = t["input_ids"].copy(); return t

train_ds = split["train"].map(tok_fn, batched=True, remove_columns=["text"])
valid_ds = split["test"].map(tok_fn,  batched=True, remove_columns=["text"])

# ------- training args ----
ta = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,        # lower peak RAM
    learning_rate=args.lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=(dtype==torch.bfloat16),
    logging_steps=25,
    save_strategy="no",                   # <-- no mid-epoch checkpoints
    eval_strategy="no",                   # <-- avoid eval RAM spike
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
print(f"✅ Training complete — checkpoint in {args.output_dir}")
