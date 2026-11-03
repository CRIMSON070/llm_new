import os
import json
import time
import torch
import requests
import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.bm25Retriever import BM25Retriever

def generate_answer(query: str, context: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Answer the financial question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    print(f"      [generate_answer] Prompt length: {len(prompt)} chars")
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=60
        )
        elapsed = time.time() - start
        print(f"      [generate_answer] Completed in {elapsed:.2f}s, status {response.status_code}")
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        elapsed = time.time() - start
        print(f"      [generate_answer] Exception after {elapsed:.2f}s: {e}")
    return "Generation failed."

def generate_hyde(query: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Please write a factual and detailed financial paragraph that answers the following question. Write it as if it were extracted directly from an official 10-K financial report. Question: {query}\n\nAnswer:"
    print(f"      [generate_hyde] Prompt length: {len(prompt)} chars")
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=60
        )
        elapsed = time.time() - start
        print(f"      [generate_hyde] Completed in {elapsed:.2f}s, status {response.status_code}")
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        elapsed = time.time() - start
        print(f"      [generate_hyde] Exception after {elapsed:.2f}s: {e}")
    return query

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embeddings on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Loading reranker...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=(device=='cuda'))
    
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_dir = os.path.join(data_dir, "bm25_index")
    
    print("Loading FAISS...")
    faiss_index = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    print("Loading BM25...")
    bm25_retriever = BM25Retriever(bm25_dir, load_corpus=True)
    
    # Load chunk store
    orig_chunks_file = os.path.join(data_dir, "processed", "parsed_chunks.jsonl")
    baseline_chunk_store = {}
    with open(orig_chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            baseline_chunk_store[record["chunk_id"]] = record
            baseline_chunk_store[i] = record 

    query = "What drove the increase in Ulta Beauty's merchandise inventories balance at end of FY2023?"
    print(f"\n--- Diagnosing Q4 ---")
    print(f"Q: {query}")
    
    # Step: HyDE
    print(f"\n[1] Starting HyDE Generation...")
    t0 = time.time()
    hypothetical_doc = generate_hyde(query)
    print(f"    HyDE finished in {time.time() - t0:.2f}s")
    
    # Step: FAISS
    print(f"\n[2] Starting FAISS Dense Retrieval...")
    t0 = time.time()
    dense_results_orig = faiss_index.similarity_search_with_score(query, k=15)
    dense_results_hyde = faiss_index.similarity_search_with_score(hypothetical_doc, k=15)
    print(f"    FAISS finished in {time.time() - t0:.2f}s")
    
    candidate_chunks = {}
    for doc, score in dense_results_orig + dense_results_hyde:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id and chunk_id not in candidate_chunks:
            text = baseline_chunk_store.get(chunk_id, {}).get("text", "")
            if not text:
                text = doc.page_content
            candidate_chunks[chunk_id] = {"text": text, "doc_name": doc.metadata.get("doc_name"), "page": doc.metadata.get("page_number")}
            
    # Step: BM25
    print(f"\n[3] Starting BM25 Sparse Retrieval...")
    t0 = time.time()
    bm25_ids, bm25_scores = bm25_retriever.invoke(query, k=20)
    for int_id in bm25_ids:
        if int_id in baseline_chunk_store:
            chunk_id = baseline_chunk_store[int_id]['chunk_id']
            if chunk_id not in candidate_chunks:
                candidate_chunks[chunk_id] = {"text": baseline_chunk_store[int_id]['text'], "doc_name": baseline_chunk_store[int_id].get("doc_name"), "page": baseline_chunk_store[int_id].get("page_number")}
    print(f"    BM25 finished in {time.time() - t0:.2f}s")
    
    candidates_list = list(candidate_chunks.items())
    pairs = [[query, data["text"]] for _, data in candidates_list]
    
    # Step: Reranker
    print(f"\n[4] Starting Reranker for {len(pairs)} pairs...")
    t0 = time.time()
    if pairs:
        scores = reranker.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
        scored_candidates = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
        top_3 = scored_candidates[:3]
    else:
        top_3 = []
    print(f"    Reranker finished in {time.time() - t0:.2f}s")
    
    context_text = "\n\n".join([f"--- Chunk {c[0][0]} (Page {c[0][1]['page']}) ---\n{c[0][1]['text']}" for c in top_3])
    
    # Step: Answer Generation
    print(f"\n[5] Starting Final Answer Generation...")
    print(f"    Context size: {len(context_text)} chars")
    t0 = time.time()
    final_answer = generate_answer(query, context_text)
    print(f"    Answer Generation finished in {time.time() - t0:.2f}s")
    
    print(f"\n--- Diagnostic Results ---")
    print(f"Top 3 Chunks: {[c[0][0] for c in top_3]}")
    print(f"Source PDFs: {[c[0][1]['doc_name'] for c in top_3]}")
    print(f"Answer: {final_answer}")

if __name__ == '__main__':
    main()
