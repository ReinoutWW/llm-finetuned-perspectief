#!/usr/bin/env python3
# demo_perspectief_adapter.py
"""
Run a Dutch question against a PEFT-fine-tuned OPT checkpoint,
handling vocab / embedding size mismatches automatically.
"""

from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig

# ────────────────────────────────────────────────────────────────
ADAPTER_DIR = Path("checkpoints/perspectief-llama3")    # ← your LoRA / PEFT dir
QUESTION     = "Hoe wordt de workshop aangeboden en wat is de duur?"
# ────────────────────────────────────────────────────────────────

# 0. Runtime device & dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = (
    torch.bfloat16
    if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
    else torch.float16
)
torch.backends.cuda.matmul.allow_tf32 = True

# 1. Read adapter config to get the *base* model ID
peft_cfg   = PeftConfig.from_pretrained(ADAPTER_DIR)
BASE_MODEL = peft_cfg.base_model_name_or_path

# 2. Tokeniser (can live in adapter dir, otherwise fall back to base)
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR if (ADAPTER_DIR / "tokenizer.json").exists() else BASE_MODEL,
    use_fast=True,
)
# ── OPTIONAL: make sure chat special tokens exist
special = {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]}
num_added = tokenizer.add_special_tokens(special)

# 3. Load *base* model only  ➜ allow mismatches because we’ll fix them next
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="auto",
    use_safetensors=True,
    ignore_mismatched_sizes=True,
)

# 4. Resize embeddings **before** attaching adapter
if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
    print(
        f"Resizing embeddings {base_model.get_input_embeddings().weight.size(0)} "
        f"→ {len(tokenizer)} (added {len(tokenizer) - base_model.get_input_embeddings().weight.size(0)} rows)"
    )
    base_model.resize_token_embeddings(len(tokenizer))

# 5. Now load the PEFT adapter on top
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    torch_dtype=dtype,
    device_map="auto",
    # The adapter contains deltas only, so no shape clash after resize
)
model.eval()

# 6. Build prompt & generate
prompt  = f"<|user|>\n{QUESTION}\n<|assistant|>\n"
inputs  = tokenizer(prompt, return_tensors="pt").to(device)

gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=500,
    repetition_penalty=1.1,
)
with torch.no_grad():
    output_ids = model.generate(**inputs, **gen_cfg.to_dict())

# 7. Decode answer
full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
answer    = full_text.split("<|assistant|>")[-1].split("<|end|>")[0].strip()

print(f"\nQ: {QUESTION}\nA: {answer}")
