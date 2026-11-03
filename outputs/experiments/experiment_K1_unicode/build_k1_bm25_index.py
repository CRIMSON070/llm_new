"""
outputs/experiments/experiment_K1_unicode/build_k1_bm25_index.py
================================================================
Builds an isolated BM25 index on the Unicode NFKC normalized Stage 3 held-out corpus.
- Input: data/heldout_stage3/experiment_k1/processed/heldout_enriched_chunks_nfkc.jsonl (29,335 chunks)
- Output: data/heldout_stage3/experiment_k1/indexes/bm25/
- Uses exact tokenizer and indexing settings from src.utils.bm25Retriever (bm25s, stopwords="english", Stemmer("english")).
- Zero modification to active production indexes or baseline directories.
"""

import os
import sys
import json
import time
import bm25s
import Stemmer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IN_CHUNKS = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "processed", "heldout_enriched_chunks_nfkc.jsonl")
OUT_BM25_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "indexes", "bm25")

def build_bm25_index():
    print("=" * 80)
    print("BUILDING ISOLATED BM25 INDEX ON NFKC NORMALIZED CORPUS")
    print("=" * 80)
    print(f"Input chunks: {IN_CHUNKS}")
    print(f"Output index: {OUT_BM25_DIR}")
    
    os.makedirs(OUT_BM25_DIR, exist_ok=True)
    
    t0 = time.time()
    corpus = []
    doc_ids = []
    
    with open(IN_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus.append(rec["text"])
            doc_ids.append(rec["chunk_id"])
            
    print(f"Loaded {len(corpus):,} chunks in {time.time() - t0:.2f}s.")
    assert len(corpus) == 29335, f"Expected 29,335 chunks, got {len(corpus)}"
    
    print("\nTokenizing corpus with bm25s (stopwords='english', stemmer='english')...")
    t_tok = time.time()
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="english", stemmer=stemmer)
    print(f"Tokenization completed in {time.time() - t_tok:.2f}s.")
    
    print("\nIndexing and saving BM25 index...")
    t_idx = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(OUT_BM25_DIR, corpus=doc_ids)
    print(f"Index built and saved to {OUT_BM25_DIR} in {time.time() - t_idx:.2f}s.")
    
    # Verify index files
    bm25_files = os.listdir(OUT_BM25_DIR)
    print(f"\nGenerated index files: {bm25_files}")
    assert any("params" in f for f in bm25_files), "params.index.json missing!"
    assert any("data" in f for f in bm25_files), "data.csc.index.npy missing!"
    
    # Test sample retrieval
    print("\nTesting sample query retrieval on K1 BM25 index...")
    sample_query = "Apple fiscal 2024 total net sales financial statements"
    query_tokens = bm25s.tokenize([sample_query], stopwords="english", stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=3)
    print(f"Sample Query: '{sample_query}'")
    for r_idx, (doc_id, score) in enumerate(zip(results[0], scores[0])):
        print(f"  Hit {r_idx+1}: {doc_id} | score: {score:.4f}")
        
    print("\n[SUCCESS] Experiment K1 isolated BM25 index built and verified perfectly.")

if __name__ == "__main__":
    build_bm25_index()
