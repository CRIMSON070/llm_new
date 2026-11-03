"""
scripts/build_heldout_stage3_indexes.py
======================================
Builds fresh, isolated BM25 and BGE-M3 FAISS vector indexes exclusively from the Stage 3 corpus.
- Input: data/heldout_stage3/processed/heldout_enriched_chunks.jsonl (29,335 chunks)
- Output BM25: data/heldout_stage3/indexes/bm25/
- Output Dense: data/heldout_stage3/indexes/faiss_bge_m3/
Zero interaction with data/financebench/indexes/.
Runs integrity checks upon completion.
"""

import os
import sys
import json
import time
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.bm25Retriever import load_from_chroma_and_save

INPUT_CHUNKS = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
BM25_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25")
FAISS_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3")

def build_indexes():
    print("=" * 80)
    print("STAGE 3: BUILDING ISOLATED BM25 AND FAISS DENSE INDEXES")
    print("=" * 80)
    
    if not os.path.exists(INPUT_CHUNKS):
        print(f"[ERROR] Input chunks file missing: {INPUT_CHUNKS}")
        sys.exit(1)
        
    print(f"Loading chunks from: {INPUT_CHUNKS}...")
    docs = []
    t0 = time.time()
    
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = {
                "doc_name": rec["doc_name"],
                "page_number": rec["page_number"],
                "doc_id": rec["chunk_id"],
                "chunk_id": rec["chunk_id"]
            }
            docs.append(Document(page_content=rec["text"], metadata=meta))
            
    print(f"Loaded {len(docs)} chunks in {time.time() - t0:.2f}s.")
    
    # -------------------------------------------------------------------------
    # 1. Build Isolated BM25 Index
    # -------------------------------------------------------------------------
    print("\n[Step 1/2] Building isolated BM25 index in:", BM25_DIR)
    os.makedirs(BM25_DIR, exist_ok=True)
    t_bm25 = time.time()
    load_from_chroma_and_save(docs, BM25_DIR)
    print(f"[SUCCESS] BM25 index built in {time.time() - t_bm25:.2f}s.")
    
    # -------------------------------------------------------------------------
    # 2. Build Isolated BGE-M3 FAISS Index
    # -------------------------------------------------------------------------
    print("\n[Step 2/2] Building isolated FAISS BGE-M3 vector index in:", FAISS_DIR)
    os.makedirs(FAISS_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")
    
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    t_faiss = time.time()
    print("Encoding chunks and constructing FAISS vectorstore (batch processing)...")
    
    # Batch add to vectorstore to track progress
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
        
    print(f"Saving FAISS index to {FAISS_DIR}...")
    vectorstore.save_local(FAISS_DIR)
    print(f"[SUCCESS] FAISS index saved in {time.time() - t_faiss:.2f}s.")
    
    # -------------------------------------------------------------------------
    # 3. Integrity Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3 INDEX INTEGRITY VERIFICATION")
    print("=" * 80)
    
    # Verify BM25 index files
    bm25_files = os.listdir(BM25_DIR)
    print(f"BM25 index files: {bm25_files}")
    assert any("bm25" in f.lower() or "index" in f.lower() or "doc" in f.lower() for f in bm25_files), "BM25 index files missing!"
    
    # Verify FAISS index files
    faiss_files = os.listdir(FAISS_DIR)
    print(f"FAISS index files: {faiss_files}")
    assert "index.faiss" in faiss_files and "index.pkl" in faiss_files, "FAISS index files missing!"
    
    # Test sample retrieval
    print("\nTesting sample retrieval on held-out index...")
    sample_query = "Apple total net sales 2024"
    results = vectorstore.similarity_search(sample_query, k=3)
    print(f"Sample query: '{sample_query}'")
    for r_idx, r in enumerate(results):
        print(f"  Hit {r_idx+1}: doc={r.metadata.get('doc_name')} p.{r.metadata.get('page_number')} | {r.page_content[:100]}...")
        
    print("\n[VERIFIED] Stage 3 Isolated Indexing Completed Successfully!")

if __name__ == "__main__":
    build_indexes()
