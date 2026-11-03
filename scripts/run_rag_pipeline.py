import os
import json
import time
import torch
import requests
import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker

warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.bm25Retriever import BM25Retriever

def generate_hyde(query: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Please write a factual and detailed financial paragraph that answers the following question. Write it as if it were extracted directly from an official 10-K financial report. Question: {query}\n\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[HyDE Error] {e}")
    return query

def generate_answer(query: str, context: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Answer the financial question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Generation Error] {e}")
    return "Generation failed."

def main():
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    
    # 1. Load raw chunks into a dictionary for fast lookup
    print("1. Loading raw document chunks...")
    chunk_store = {}
    with open(os.path.join(data_dir, "processed", "parsed_chunks.jsonl"), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            # Maintain index mapping for BM25
            chunk_store[i] = record
            chunk_store[record['chunk_id']] = record
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load FAISS and BM25
    print("2. Loading FAISS and BM25 Indexes...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    faiss_index = FAISS.load_local(os.path.join(data_dir, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever(os.path.join(data_dir, "bm25_index"), load_corpus=True)
    
    # 3. Load Reranker
    print("3. Loading Reranker (bge-reranker-base)...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=(device=='cuda'))
    
    # Test Query
    test_query = "What is the FY2018 operating margin for 3M?"
    print(f"\n==================================================")
    print(f"QUERY: {test_query}")
    print(f"==================================================")
    
    # PHASE 7: HyDE + Retrieval
    print("\n[Phase 7] Generating HyDE and Retrieving...")
    t0 = time.time()
    hypothetical_doc = generate_hyde(test_query)
    
    # Dense Retrieval (K=10 for Orig, K=10 for HyDE)
    dense_results_orig = faiss_index.similarity_search_with_score(test_query, k=10)
    dense_results_hyde = faiss_index.similarity_search_with_score(hypothetical_doc, k=10)
    
    candidate_chunks = {}
    
    for doc, score in dense_results_orig + dense_results_hyde:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id and chunk_id not in candidate_chunks:
            candidate_chunks[chunk_id] = chunk_store[chunk_id]['text']
            
    # Sparse Retrieval (K=20)
    bm25_ids, bm25_scores = bm25_retriever.invoke(test_query, k=20)
    for int_id in bm25_ids:
        if int_id in chunk_store:
            chunk_id = chunk_store[int_id]['chunk_id']
            if chunk_id not in candidate_chunks:
                candidate_chunks[chunk_id] = chunk_store[int_id]['text']
                
    print(f" -> Retrieved {len(candidate_chunks)} unique candidate chunks. ({(time.time()-t0):.2f}s)")
    
    # PHASE 8: Reranking
    print("\n[Phase 8] Reranking Candidates...")
    t1 = time.time()
    candidates_list = list(candidate_chunks.items())
    
    # FlagReranker expects list of [query, passage] pairs
    pairs = [[test_query, text] for _, text in candidates_list]
    scores = reranker.compute_score(pairs)
    
    # Sort by score descending
    scored_candidates = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
    
    top_3_chunks = scored_candidates[:3]
    context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{c[0][1]}" for i, c in enumerate(top_3_chunks)])
    print(f" -> Reranking complete. ({(time.time()-t1):.2f}s)")
    print(f" -> Top 1 Score: {top_3_chunks[0][1]:.4f}")
    
    # PHASE 8: Generation
    print("\n[Phase 8] Generating Final Answer...")
    t2 = time.time()
    final_answer = generate_answer(test_query, context_text)
    print(f" -> Generation complete. ({(time.time()-t2):.2f}s)")
    
    print("\n==================================================")
    print(f"FINAL ANSWER:\n{final_answer}")
    print(f"==================================================")
    
    print(f"\nTotal Pipeline Runtime: {(time.time()-start_time):.2f}s")

if __name__ == "__main__":
    main()
