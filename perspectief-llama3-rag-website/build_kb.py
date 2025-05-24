#!/usr/bin/env python3
"""
build_kb.py
-----------
Create a FAISS vector store from your Perspectief JSON-Lines dataset.

* Every row **must** have at least the keys
    - "DetailedContext"  (string you want to retrieve over)
    - "Question", "Answer", … (kept as metadata; optional but handy)

Usage
-----
$ python build_kb.py \
      --data ../Scrape-website/perspectief_dataset-1.jsonl \
      --index_dir kb

This will write:

kb/
 ├─ faiss.index   – vectors (inner-product / cosine)
 └─ meta.pkl      – list of original JSON rows (same order as vectors)

Dependencies
------------
pip install sentence-transformers faiss-cpu tqdm
"""
from pathlib import Path
import argparse, json, pickle, numpy as np, faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────────────────────
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, multilingual
# ───────────────────────────────────────────────────────────────


def build_index(examples, embedder_name: str):
    """
    Return (faiss.IndexFlatIP, metadata_list).
    Vectors are L2-normalised so inner-product == cosine.
    """
    model = SentenceTransformer(embedder_name)
    dim   = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    meta, vecs = [], []
    for ex in tqdm(examples, desc="Embedding contexts"):
        ctx = ex["DetailedContext"].strip()
        emb = model.encode(ctx,
                           convert_to_numpy=True,
                           normalize_embeddings=True)        # unit length
        vecs.append(emb.astype("float32"))
        meta.append(ex)

    mat = np.vstack(vecs)                                   # (N, dim)
    index.add(mat)
    return index, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      required=True,
                    help="Path to JSONL file with training rows")
    ap.add_argument("--index_dir", default="kb",
                    help="Folder to write faiss.index + meta.pkl")
    ap.add_argument("--embedder",  default=EMBEDDER_NAME,
                    help="Sentence-Transformers model for embeddings")
    args = ap.parse_args()

    # 1. Load JSONL rows -------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"❌ No such file: {data_path}")

    rows = [json.loads(l) for l in data_path.read_text().splitlines()]
    if not rows:
        raise SystemExit("❌ Input file is empty")

    # 2. Build vector store ----------------------------------------------------
    index, meta = build_index(rows, args.embedder)

    # 3. Persist ---------------------------------------------------------------
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "meta.pkl", "wb") as fh:
        pickle.dump(meta, fh)

    print(f"✅ Saved {len(meta)} contexts to {out_dir}")


if __name__ == "__main__":
    main()
