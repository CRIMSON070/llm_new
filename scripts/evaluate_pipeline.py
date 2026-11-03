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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    questions_file = os.path.join(data_dir, "questions.jsonl")
    
    print("1. Loading raw document chunks...")
    chunk_store = {}
    with open(os.path.join(data_dir, "processed", "parsed_chunks.jsonl"), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            chunk_store[i] = record
            chunk_store[record['chunk_id']] = record
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("2. Loading FAISS and BM25 Indexes...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    faiss_index = FAISS.load_local(os.path.join(data_dir, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever(os.path.join(data_dir, "bm25_index"), load_corpus=True)
    
    print("3. Loading Reranker (bge-reranker-base)...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=(device=='cuda'))
    
    # Load FinanceBench Questions
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
            
    # Take a subset to evaluate quickly (e.g., first 5)
    test_questions = questions[:5]
    print(f"\nStarting Evaluation on {len(test_questions)} questions...\n")
    
    metrics = {
        "total": 0,
        "runtime_secs": 0,
        "errors": 0
    }
    
    start_eval = time.time()
    
    for idx, q_data in enumerate(test_questions):
        metrics["total"] += 1
        query = q_data["question"]
        doc_name = q_data["doc_name"]
        expected_answer = q_data["answer"]
        print(f"\n[{idx+1}/{len(test_questions)}] QUESTION: {query}")
        
        t0 = time.time()
        
        try:
            # HyDE
            hypothetical_doc = generate_hyde(query)
            
            # Retrieval
            dense_results_orig = faiss_index.similarity_search_with_score(query, k=10)
            dense_results_hyde = faiss_index.similarity_search_with_score(hypothetical_doc, k=10)
            
            candidate_chunks = {}
            for doc, score in dense_results_orig + dense_results_hyde:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id and chunk_id not in candidate_chunks:
                    candidate_chunks[chunk_id] = {"text": chunk_store[chunk_id]['text'], "doc_name": doc.metadata.get("doc_name")}
                    
            bm25_ids, bm25_scores = bm25_retriever.invoke(query, k=20)
            for int_id in bm25_ids:
                if int_id in chunk_store:
                    chunk_id = chunk_store[int_id]['chunk_id']
                    if chunk_id not in candidate_chunks:
                        candidate_chunks[chunk_id] = {"text": chunk_store[int_id]['text'], "doc_name": chunk_store[int_id].get("doc_name")}
                        
            # Reranking
            candidates_list = list(candidate_chunks.items())
            pairs = [[query, data["text"]] for _, data in candidates_list]
            
            if pairs:
                scores = reranker.compute_score(pairs)
                if not isinstance(scores, list):
                    scores = [scores] # compute_score returns a single float if len(pairs) == 1
                scored_candidates = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
                top_3 = scored_candidates[:3]
            else:
                top_3 = []
            
            # Check if correct document was retrieved in top 3
            retrieved_correct_doc = any(c[0][1]["doc_name"] == doc_name for c in top_3)
            print(f" -> Top 3 Retrieved from correct doc: {retrieved_correct_doc}")
            
            # Generation
            context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{c[0][1]['text']}" for i, c in enumerate(top_3)])
            final_answer = generate_answer(query, context_text)
            
            t1 = time.time()
            print(f" -> Answer: {final_answer}")
            print(f" -> Expected: {expected_answer}")
            print(f" -> Runtime: {(t1-t0):.2f}s")
            
        except Exception as e:
            print(f" -> ERROR processing question: {e}")
            metrics["errors"] += 1
            
    metrics["runtime_secs"] = time.time() - start_eval
    
    print("\n==================================================")
    print("EVALUATION RESULTS")
    print(f"Total Evaluated: {metrics['total']}")
    print(f"Total Errors: {metrics['errors']}")
    print(f"Total Runtime: {metrics['runtime_secs']:.2f}s")
    print(f"Average Time / Query: {metrics['runtime_secs']/metrics['total']:.2f}s")
    print("==================================================")

if __name__ == "__main__":
    main()
