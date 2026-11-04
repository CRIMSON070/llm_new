"""
scripts/build_grounded_stage3_indexes.py
========================================
Builds isolated BM25 and FAISS BGE-M3 dense vector indexes from the newly generated
offline-grounded Stage 3 corpus (Experiment L).

- Input:  data/heldout_stage3/processed/heldout_grounded_chunks.jsonl (29,335 chunks)
- Output BM25:  data/heldout_stage3/indexes_grounded_exp_l/bm25/
- Output FAISS: data/heldout_stage3/indexes_grounded_exp_l/faiss_bge_m3/
- Manifest:     data/heldout_stage3/indexes_grounded_exp_l/index_manifest.json

Strict Isolation Guarantees:
- Zero writes or modifications to data/heldout_stage3/indexes/ (Baseline).
- Zero writes to data/financebench/.
- Uses exact BGE-M3 (fp16 / normalized) and bm25s English tokenizer settings matching baseline.
"""

import os
import sys
import json
import time
import hashlib
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.bm25Retriever import load_from_chroma_and_save

INPUT_CHUNKS = os.path.join(
    BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl"
)
OUTPUT_INDEX_DIR = os.path.join(
    BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l"
)
BM25_DIR = os.path.join(OUTPUT_INDEX_DIR, "bm25")
FAISS_DIR = os.path.join(OUTPUT_INDEX_DIR, "faiss_bge_m3")
MANIFEST_PATH = os.path.join(OUTPUT_INDEX_DIR, "index_manifest.json")


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_grounded_indexes():
    print("=" * 80)
    print("EXPERIMENT L: BUILDING ISOLATED GROUNDED BM25 AND FAISS INDEXES")
    print("=" * 80)

    if not os.path.exists(INPUT_CHUNKS):
        print(f"[ERROR] Input grounded chunks file missing: {INPUT_CHUNKS}")
        sys.exit(1)

    os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)
    os.makedirs(BM25_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    print(f"Loading grounded chunks from: {INPUT_CHUNKS}...")
    t0 = time.time()
    docs = []

    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = {
                "doc_name": rec["doc_name"],
                "page_number": rec["page_number"],
                "doc_id": rec["chunk_id"],
                "chunk_id": rec["chunk_id"]
            }
            # Use grounded_text for indexing
            docs.append(Document(page_content=rec["grounded_text"], metadata=meta))

    print(f"Loaded {len(docs)} grounded chunks in {time.time() - t0:.2f}s.")
    assert len(docs) == 29335, f"Expected 29335 documents, got {len(docs)}"

    # -------------------------------------------------------------------------
    # 1. Build Isolated BM25 Index
    # -------------------------------------------------------------------------
    print("\n[Step 1/2] Building isolated BM25 index in:", BM25_DIR)
    t_bm25 = time.time()
    load_from_chroma_and_save(docs, BM25_DIR)
    bm25_duration = time.time() - t_bm25
    print(f"[SUCCESS] Grounded BM25 index built in {bm25_duration:.2f}s.")

    # -------------------------------------------------------------------------
    # 2. Build Isolated BGE-M3 FAISS Index
    # -------------------------------------------------------------------------
    print("\n[Step 2/2] Building isolated FAISS BGE-M3 vector index in:", FAISS_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    t_faiss = time.time()
    batch_size = 1000
    vectorstore = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        t_b0 = time.time()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        print(f"  Indexed [{min(i + batch_size, len(docs))}/{len(docs)}] chunks ({time.time() - t_b0:.1f}s)")

    print(f"Saving grounded FAISS index to {FAISS_DIR}...")
    vectorstore.save_local(FAISS_DIR)
    faiss_duration = time.time() - t_faiss
    print(f"[SUCCESS] FAISS index saved in {faiss_duration:.2f}s.")

    # -------------------------------------------------------------------------
    # 3. Create Manifest
    # -------------------------------------------------------------------------
    bm25_files = {f: compute_sha256(os.path.join(BM25_DIR, f)) for f in sorted(os.listdir(BM25_DIR))}
    faiss_files = {f: compute_sha256(os.path.join(FAISS_DIR, f)) for f in sorted(os.listdir(FAISS_DIR))}

    manifest_data = {
        "experiment": "Experiment L: Offline Entity Grounding",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(docs),
        "source_corpus": INPUT_CHUNKS,
        "source_corpus_sha256": compute_sha256(INPUT_CHUNKS),
        "embedding_model": model_name,
        "embedding_device": device,
        "bm25_build_duration_sec": round(bm25_duration, 2),
        "faiss_build_duration_sec": round(faiss_duration, 2),
        "index_paths": {
            "bm25_dir": BM25_DIR,
            "faiss_dir": FAISS_DIR
        },
        "file_hashes": {
            "bm25": bm25_files,
            "faiss": faiss_files
        }
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nSaved Grounded Index Manifest to: {MANIFEST_PATH}")
    print("=" * 80)
    print("INDEX BUILD COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    build_grounded_indexes()
