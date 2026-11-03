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
from src.utils.bm25Retriever import BM25Retriever, load_from_chroma_and_save

def generate_hyde(query: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Please write a factual and detailed financial paragraph that answers the following question. Write it as if it were extracted directly from an official 10-K financial report. Question: {query}\n\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass
    return query

def generate_answer(query: str, context: str, model: str = "llama3.2:latest") -> str:
    prompt = f"Answer the financial question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass
    return "Generation failed."

def run_evaluation(name, data_dir, faiss_index, bm25_retriever, chunk_store, reranker, test_questions):
    print(f"\n--- Evaluating {name} Pipeline ---", flush=True)
    
    results = []
    for idx, q_data in enumerate(test_questions):
        query = q_data["question"]
        doc_name = q_data["doc_name"]
        print(f"\n[{idx+1}/{len(test_questions)}] Q: {query}", flush=True)
        
        hypothetical_doc = generate_hyde(query)
        
        dense_results_orig = faiss_index.similarity_search_with_score(query, k=15)
        dense_results_hyde = faiss_index.similarity_search_with_score(hypothetical_doc, k=15)
        
        candidate_chunks = {}
        for doc, score in dense_results_orig + dense_results_hyde:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in candidate_chunks:
                text = chunk_store.get(chunk_id, {}).get("text", "")
                if not text:
                    text = doc.page_content
                candidate_chunks[chunk_id] = {"text": text, "doc_name": doc.metadata.get("doc_name"), "page": doc.metadata.get("page_number")}
                
        bm25_ids, bm25_scores = bm25_retriever.invoke(query, k=20)
        for int_id in bm25_ids:
            if int_id in chunk_store:
                chunk_id = chunk_store[int_id]['chunk_id']
                if chunk_id not in candidate_chunks:
                    candidate_chunks[chunk_id] = {"text": chunk_store[int_id]['text'], "doc_name": chunk_store[int_id].get("doc_name"), "page": chunk_store[int_id].get("page_number")}
                    
        candidates_list = list(candidate_chunks.items())
        pairs = [[query, data["text"]] for _, data in candidates_list]
        
        if pairs:
            scores = reranker.compute_score(pairs, batch_size=4)
            if not isinstance(scores, list):
                scores = [scores]
            scored_candidates = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
            top_3 = scored_candidates[:3]
        else:
            top_3 = []
        
        retrieved_correct = any(c[0][1]["doc_name"] == doc_name for c in top_3)
        context_text = "\n\n".join([f"--- Chunk {c[0][0]} (Page {c[0][1]['page']}) ---\n{c[0][1]['text']}" for c in top_3])
        final_answer = generate_answer(query, context_text)
        
        print(f" -> Top 3 Chunks: {[c[0][0] for c in top_3]}", flush=True)
        print(f" -> Correct Doc Retrieved: {retrieved_correct}", flush=True)
        print(f" -> Generated Answer: {final_answer}", flush=True)
        print(f" -> Expected: {q_data['answer']}", flush=True)
        
        result_obj = {
            "query": query,
            "top_3_chunks": [c[0][0] for c in top_3],
            "correct_doc_retrieved": retrieved_correct,
            "generated_answer": final_answer,
            "expected_answer": q_data['answer'],
            "context_text": context_text
        }
        results.append(result_obj)
        
        out_file = os.path.join(data_dir, f"{name.lower()}_results.jsonl")
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_obj) + '\n')
            
    return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    
    baseline_faiss_dir = os.path.join(data_dir, "faiss_index")
    baseline_bm25_dir = os.path.join(data_dir, "bm25_index")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embeddings on {device}...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Loading reranker...", flush=True)
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=(device=='cuda'))
    
    questions_file = os.path.join(data_dir, "questions.jsonl")
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
            
    test_pdfs = ["3M_2018_10K", "ULTABEAUTY_2023Q4_EARNINGS"]
    test_questions = [q for q in questions if q["doc_name"] in test_pdfs][:4]
    print(f"Selected {len(test_questions)} questions for evaluation.", flush=True)
    
    print("\n[Step 1] Loading Baseline Index...", flush=True)
    baseline_faiss = FAISS.load_local(baseline_faiss_dir, embeddings, allow_dangerous_deserialization=True)
    baseline_bm25 = BM25Retriever(baseline_bm25_dir, load_corpus=True)
    
    baseline_chunk_store = {}
    with open(orig_chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            baseline_chunk_store[record["chunk_id"]] = record
            baseline_chunk_store[i] = record 
            
    run_evaluation("BASELINE", data_dir, baseline_faiss, baseline_bm25, baseline_chunk_store, reranker, test_questions)
    
    print("\n--- Preparing UPGRADED Pipeline ---", flush=True)
    print("Loading fresh copy of Baseline FAISS index...", flush=True)
    upgraded_faiss = FAISS.load_local(baseline_faiss_dir, embeddings, allow_dangerous_deserialization=True)
    
    upgraded_chunk_store = dict(baseline_chunk_store)
    new_docs = []
    added_count = 0
    next_bm25_id = max([k for k in upgraded_chunk_store.keys() if isinstance(k, int)]) + 1
    
    if os.path.exists(new_chunks_file):
        with open(new_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                upgraded_chunk_store[record["chunk_id"]] = record
                upgraded_chunk_store[next_bm25_id] = record
                next_bm25_id += 1
                
                metadata = {
                    "doc_name": record["doc_name"],
                    "page_number": record.get("page_number", 0),
                    "chunk_id": record["chunk_id"]
                }
                new_docs.append(Document(page_content=record["text"], metadata=metadata))
                added_count += 1
                
        if new_docs:
            print(f"Embedding {added_count} new multimodal chunks into FAISS...", flush=True)
            upgraded_faiss.add_documents(new_docs)
            print("Finished embedding new chunks.", flush=True)
    
    # We will reuse the baseline BM25 index to avoid recompiling it for 54k documents. 
    # FAISS will retrieve the new chunks, and Reranker will score them.
    print("Reusing Baseline BM25 index (FAISS will supply the new multimodal chunks to the reranker).", flush=True)
    
    run_evaluation("UPGRADED", data_dir, upgraded_faiss, baseline_bm25, upgraded_chunk_store, reranker, test_questions)

if __name__ == "__main__":
    main()
