import os
import json
import gc
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

def safe_calculate(operation: str, variables: dict):
    try:
        if operation == "percentage_change":
            old = float(variables.get("Old", 0))
            new = float(variables.get("New", 0))
            return round(((new - old) / old) * 100, 4)
        elif operation == "difference":
            v1 = float(variables.get("Value1", 0))
            v2 = float(variables.get("Value2", 0))
            return round(v1 - v2, 4)
        elif operation == "sum":
            v1 = float(variables.get("Value1", 0))
            v2 = float(variables.get("Value2", 0))
            return round(v1 + v2, 4)
        elif operation == "extract":
            return round(float(variables.get("Value", 0)), 4)
        elif operation == "none":
            return "No calculation needed."
        else:
            return f"Unknown operation: {operation}"
    except Exception as e:
        return f"Error computing {operation}: {e}"

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

def extract_and_calculate(query: str, context: str, model: str) -> dict:
    prompt = f"""You are a strict financial extraction AI.
Answer the user's financial question based ONLY on the provided context.
You must extract the exact numerical values required, and select one predefined calculation operation.

Available operations and required variables:
- 'percentage_change': requires variables 'Old' and 'New'.
- 'difference': requires variables 'Value1' and 'Value2'.
- 'sum': requires variables 'Value1' and 'Value2'.
- 'extract': requires variable 'Value' (use this if the answer is just a single number found in the text).
- 'none': use this if the question is purely text-based and requires no numerical calculation.

Context:
{context}

Question: {query}

Output valid JSON ONLY in the exact following format, with no other text:
{{
  "reasoning": "Step by step explanation of what you are extracting.",
  "operation": "percentage_change",
  "variables": {{"Old": 100.5, "New": 110.0}},
  "units": "USD millions"
}}
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}, "format": "json"},
            timeout=90
        )
        if response.status_code == 200:
            result_str = response.json().get("response", "").strip()
            if result_str.startswith("```json"): result_str = result_str[7:]
            if result_str.startswith("```"): result_str = result_str[3:]
            if result_str.endswith("```"): result_str = result_str[:-3]
            return json.loads(result_str.strip())
    except Exception as e:
        print(f"[{model}] Error parsing JSON: {e}")
    return {"operation": "error", "reasoning": "Failed to generate valid response.", "variables": {}, "units": ""}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    baseline_faiss_dir = os.path.join(data_dir, "faiss_index")
    baseline_bm25_dir = os.path.join(data_dir, "bm25_index")
    
    questions_file = os.path.join(data_dir, "questions.jsonl")
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
            
    test_pdfs = ["3M_2018_10K", "ULTABEAUTY_2023Q4_EARNINGS"]
    test_questions = [q for q in questions if q["doc_name"] in test_pdfs][:4]
    
    test_questions.append({
        "question": "Based on the statements of income, what is the absolute difference in net sales for 3M from 2017 to 2018? Also calculate the percentage change as ((2018 - 2017) / 2017) * 100.",
        "doc_name": "3M_2018_10K",
        "answer": "Net sales in 2018 - Net sales in 2017. Approx 3%."
    })
    test_questions.append({
        "question": "What is the difference between Ulta Beauty's merchandise inventories at the end of FY2023 and the end of FY2022?",
        "doc_name": "ULTABEAUTY_2023_10K",
        "answer": "$1,603,451 - $1,499,218 = $104,233"
    })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading retrieval models on {device}...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=(device=='cuda'))
    
    upgraded_faiss = FAISS.load_local(baseline_faiss_dir, embeddings, allow_dangerous_deserialization=True)
    baseline_bm25 = BM25Retriever(baseline_bm25_dir, load_corpus=True)
    
    chunk_store = {}
    with open(orig_chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            chunk_store[record["chunk_id"]] = record
            chunk_store[i] = record 
            
    new_docs = []
    if os.path.exists(new_chunks_file):
        with open(new_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                chunk_store[record["chunk_id"]] = record
                new_docs.append(Document(page_content=record["text"], metadata={"doc_name": record["doc_name"], "page_number": record.get("page_number", 0), "chunk_id": record["chunk_id"]}))
        if new_docs:
            upgraded_faiss.add_documents(new_docs)

    print("\n--- PHASE 1: RETRIEVAL ---", flush=True)
    retrieved_contexts = []
    for idx, q_data in enumerate(test_questions):
        query = q_data["question"]
        hypothetical_doc = generate_hyde(query)
        dense_results_orig = upgraded_faiss.similarity_search_with_score(query, k=15)
        dense_results_hyde = upgraded_faiss.similarity_search_with_score(hypothetical_doc, k=15)
        
        candidate_chunks = {}
        for doc, score in dense_results_orig + dense_results_hyde:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in candidate_chunks:
                text = chunk_store.get(chunk_id, {}).get("text", doc.page_content)
                candidate_chunks[chunk_id] = {"text": text, "doc_name": doc.metadata.get("doc_name"), "page": doc.metadata.get("page_number")}
                
        bm25_ids, bm25_scores = baseline_bm25.invoke(query, k=20)
        for int_id in bm25_ids:
            if int_id in chunk_store:
                chunk_id = chunk_store[int_id]['chunk_id']
                if chunk_id not in candidate_chunks:
                    candidate_chunks[chunk_id] = {"text": chunk_store[int_id]['text'], "doc_name": chunk_store[int_id].get("doc_name"), "page": chunk_store[int_id].get("page_number")}
                    
        candidates_list = list(candidate_chunks.items())
        pairs = [[query, data["text"]] for _, data in candidates_list]
        top_3 = []
        if pairs:
            scores = reranker.compute_score(pairs, batch_size=4)
            if not isinstance(scores, list): scores = [scores]
            scored_candidates = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
            top_3 = scored_candidates[:3]
            
        context_text = "\n\n".join([f"--- Chunk {c[0][0]} (Page {c[0][1]['page']}) ---\n{c[0][1]['text']}" for c in top_3])
        retrieved_correct = any(c[0][1]["doc_name"] == q_data["doc_name"] for c in top_3)
        retrieved_contexts.append((query, context_text, retrieved_correct, [c[0][0] for c in top_3]))
        print(f"Prepared retrieval for Q{idx+1}", flush=True)

    print("\n--- PHASE 2: UNLOADING MODELS TO FREE VRAM ---", flush=True)
    del upgraded_faiss
    del baseline_bm25
    del embeddings
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM cleared.", flush=True)

    models_to_test = ["llama3.2:latest", "qwen2.5:7b"]
    
    print("\n--- PHASE 3: LLM EVALUATION ---", flush=True)
    out_file = os.path.join(data_dir, "qwen_comparison_results.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)

    for model in models_to_test:
        print(f"\n================ EVALUATING MODEL: {model} ================", flush=True)
        # Unload Ollama's previous model manually if needed by calling an empty request to the other or just letting Ollama handle it.
        # Ollama automatically unloads when a different model is requested and VRAM is tight, but to be safe:
        requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0})
        
        for idx, q_data in enumerate(test_questions):
            query, context_text, retrieved_correct, chunk_ids = retrieved_contexts[idx]
            print(f"\n[{model}] Q{idx+1}: {query}", flush=True)
            
            ext = extract_and_calculate(query, context_text, model)
            
            operation = ext.get("operation", "none")
            calc_result = safe_calculate(operation, ext.get("variables", {}))
            final_answer = f"Operation: {operation}\nVariables: {ext.get('variables')}\nUnits: {ext.get('units')}\nComputed Answer: {calc_result}\nReasoning: {ext.get('reasoning')}"
            
            print(f" -> Computed Answer: {calc_result}", flush=True)
            print(f" -> Expected: {q_data['answer']}", flush=True)
            
            result_obj = {
                "model": model,
                "query": query,
                "top_3_chunks": chunk_ids,
                "correct_doc_retrieved": retrieved_correct,
                "generated_answer": final_answer,
                "expected_answer": q_data['answer'],
                "extraction": ext
            }
            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_obj) + '\n')

if __name__ == "__main__":
    main()
