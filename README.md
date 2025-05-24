# Perspectief LLM + RAG Project

End‑to‑end pipeline that turns a public website into a question‑answering assistant powered by a fine‑tuned **Llama‑3.1 8B Instruct** model, a FAISS knowledge base and a lightweight RAG wrapper.

---

## 🗺️ Repository structure

```
.
├── Scrape-website/                  # website crawler → JSONL dataset
│   └── scraper.py
├── training/                 # fine‑tuning assets
│   ├── perspectief_training.py
├── perspectief-llama3-rag-website/  # retrieval‑augmented chat
│   ├── build_kb.py
│   └── rag_chat.py
└── README.md
```

---

## 0 . Prerequisites

| Tool                                            | Version tested                          |
| ----------------------------------------------- | --------------------------------------- |
| Python                                          |  3.10 – 3.12                            |
| CUDA                                            |  12.x (optional, CPU works but is slow) |
| PyTorch                                         |  2.2+                                   |
| [Unsloth](https://github.com/unslothai/unsloth) |  2025.5                                 |
| Transformers                                    |  4.41                                   |
| Sentence‑Transformers                           |  2.6                                    |
| FAISS                                           |  1.7                                    |

Install everything in one go:

```bash
pip install -r requirements.txt     # provided
```

> **GPU note:** the training script uses 4‑bit QLoRA and runs comfortably on a single RTX 4090 or A6000 (≈ 24 GB vRAM).

---

## 1 . Scrape the website

```bash
python scraper/crawl_perspectief.py \
       --base https://perspectief.eu/ \
       --out  data/perspectief_raw.json
```

The crawler:

* follows internal links only
* extracts **URL**, **title**, **main content**
* skips duplicates & junk pages

---

## 2 . Convert raw pages → fine‑tune dataset

The helper `scraper/dataset-processor.py` reads the raw pages and heuristically derives **Question / Answer** pairs (H2 → Q, paragraph → A). The resulting JSON‑Lines file uses the schema your model expects:

```jsonc
{
  "URL": "…",
  "Question": "What is the total cost for the 4‑hour plenary session, and is there a discount for colleagues?",
  "Answer":   "The plenary session costs €595 (excluding VAT)…",
  "DetailedContext": "The €595 fee applies…"
}
```

Run it:

```bash
python scraper/dataset-processor.py \
       --in  data/perspectief_raw.json \
       --out data/perspectief_dataset-1.jsonl
```

💡 **Manual pass recommended:** skim the JSONL and fix typos—garbage in, garbage out!

---

## 3 . Fine‑tune Llama‑3 with Unsloth

The training script below (already in `training/`) applies 4‑bit QLoRA to **Meta‑Llama‑3.1‑8B‑Instruct**. It formats every row as:

```
<|user|>
{Question}

Context:
{DetailedContext}
<|assistant|>
{Answer}<|end|>
```

### 3.1  Launch training

```bash
python training/perspectief_training.py \
  --output_dir training/checkpoints/perspectief-llama3 \
  --epochs 6 \
  --batch  2 \
  --lr     2e-5
```

* Early stopping isn’t enabled; monitor loss and abort if it plateaus.
* The script automatically splits 10 % for validation.

### 3.2  Result

```
training/checkpoints/perspectief-llama3/
 ├─ adapter_model.safetensors   # LoRA weights
 ├─ adapter_config.json
 ├─ tokenizer.json / tokenizer_config.json
 └─ …
```

---

## 4 . Build the knowledge base (FAISS)

```bash
python rag/build_kb.py \
       --data      data/perspectief_dataset-1.jsonl \
       --index_dir kb
```

* Uses **all‑MiniLM‑L6‑v2** embeddings (swapable via `--embedder`).
* Normalises embeddings so `IndexFlatIP` ≈ cosine similarity.

Output:

```
kb/
 ├─ faiss.index   # vectors
 └─ meta.pkl      # original rows, aligned with vectors
```

---

## 5 . Run Retrieval‑Augmented Generation

```bash
python rag/rag_chat.py \
       --question "" \
       --top_k    3
```

Under the hood:

1. **Embed** the question → cosine search in FAISS.
2. **Assemble prompt** in the exact fine‑tune template (user/context/assistant).
3. **Generate** with the adapter on top of the base model.

Typical output:

```
Context passages used:
--------------------------------------------------------------------------------
The €595 fee applies to each primary participant…
[source] https://...
--------------------------------------------------------------------------------
A: De plenaire sessie kost €595 (excl. btw) en een tweede collega krijgt 40 % korting…
```

---

## 6 . Evaluation & QA checks

| Check                 | Command                                       |
| --------------------- | --------------------------------------------- |
| **Retriever sanity**  | `python rag/quick_test.py --question "…"`     |
| **Exact‑match score** | see `evaluation/simple_em.py`                 |
| **Live chat**         | integrate `rag_chat.py` as a FastAPI endpoint |

---

## 7 . Troubleshooting

### FAISS `AttributeError: numpy_array_to_float_array`

You’re on FAISS ≥ 1.7. The fixed `build_kb.py` already passes a plain `np.ndarray`—make sure you pulled latest.

### CUDA OOM during training

* Reduce `--batch` or enable gradient‑checkpointing (already on).
* Fit can be resumed from the last checkpoint.

### Model ignores retrieved context

* Increase `--top_k` in `rag_chat.py`.
* Split very long `DetailedContext` strings (< 512 tokens ideal).
