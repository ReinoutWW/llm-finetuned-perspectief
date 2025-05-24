#!/usr/bin/env python3
"""
rag_chat.py
-----------
RAG wrapper around your PEFT-tuned Llama-3 adapter.

$ python rag_chat.py --question "Hoe begon Perspectief?"
"""
import argparse, pickle, torch, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig

# ───────────────────────────────────────────────────────────────
ADAPTER_DIR = Path("../checkpoints/perspectief-llama3")   # <-- change to yours
KB_DIR      = Path("kb")                               # created by build_kb.py
EMBEDDER    = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 6
# ───────────────────────────────────────────────────────────────

def load_vector_store():
    index = faiss.read_index(str(KB_DIR / "faiss.index"))
    meta  = pickle.loads(Path(KB_DIR / "meta.pkl").read_bytes())
    return index, meta

def embed_text(text, model):
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.reshape(1, -1)

def build_prompt(question, contexts):
    ctx_block = "\n---\n".join(contexts)
    return f"<|user|>\n{question}\n\nContext:\n{ctx_block}\n<|assistant|>\n"

def load_llm():
    peft_cfg   = PeftConfig.from_pretrained(ADAPTER_DIR)
    base_id    = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR if (ADAPTER_DIR / "tokenizer.json").exists() else base_id,
        use_fast=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]}
    )

    dtype = torch.bfloat16 if (torch.cuda.is_available() and
                               torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="auto",
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    )
    base.resize_token_embeddings(len(tokenizer))  # in case we added specials
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, torch_dtype=dtype,
                                      device_map="auto")
    model.eval()
    return tokenizer, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="User question (Dutch)")
    ap.add_argument("--top_k", type=int, default=TOP_K)
    args = ap.parse_args()

    # 1. Retrieve
    index, meta   = load_vector_store()
    st_model      = SentenceTransformer(EMBEDDER)
    q_emb         = embed_text(args.question, st_model)
    D, I          = index.search(q_emb, args.top_k)

    contexts      = [meta[i]["DetailedContext"].strip() for i in I[0]]
    sources       = [meta[i].get("URL") for i in I[0]]  # optional for logging

    # 2. Prompt
    prompt = build_prompt(args.question, contexts)

    # 3. LLM inference
    tokenizer, model = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(
        do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=512, repetition_penalty=1.1
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_cfg.to_dict())

    answer  = tokenizer.decode(out_ids[0], skip_special_tokens=True)        \
                      .split("<|assistant|>")[-1].split("<|end|>")[0].strip()

    print(f"\nQ: {args.question}\n")
    print("Context passages used:")
    for c, url in zip(contexts, sources):
        print("-" * 80)
        print(c if len(c) < 600 else c[:600] + " …")
        if url:
            print(f"[source] {url}")
    print("-" * 80)
    print(f"\nA: {answer}\n")

if __name__ == "__main__":
    main()
