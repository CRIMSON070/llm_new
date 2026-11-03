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

def safe_eval(formula: str, variables: dict):
    try:
        # A simple mathematical evaluator
        allowed_names = {k: float(v) for k, v in variables.items()}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        # evaluate safely
        result = eval(formula, {"__builtins__": {}}, allowed_names)
        return round(result, 4)
    except Exception as e:
        return f"Error computing formula: {e}"

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

def extract_and_calculate(query: str, context: str, model: str = "llama3.2:latest") -> dict:
    prompt = f"""You are a financial extraction and math logic AI.
Your task is to answer the user's financial question based ONLY on the provided context.
Instead of doing the math yourself, extract the necessary numerical values and provide a mathematical formula. 
If the context does not contain the required information to answer the question, set 'can_calculate' to false.
The 'variables' should be a dictionary mapping variable names (like 'A', 'B', 'sales_2017') to numerical values. Do not include currency symbols or commas in the values.
The 'formula' should be a valid Python math expression using the variable names.

Context:
{context}

Question: {query}

Output valid JSON ONLY in the following format, with no other text or explanation:
{{
  "reasoning": "Explain step by step what numbers you are extracting and why",
  "can_calculate": true,
  "variables": {{"A": 100.5, "B": 50.0}},
  "formula": "A - B"
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
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            if result_str.startswith("```"):
                result_str = result_str[3:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]
            result_str = result_str.strip()
            return json.loads(result_str)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}")
        pass
    return {"can_calculate": False, "reasoning": "Failed to generate valid response.", "variables": {}, "formula": ""}

def run_evaluation(data_dir, faiss_index, bm25_retriever, chunk_store, reranker, test_questions):
    print(f"\n--- Evaluating Phase 9A Pipeline (Upgraded + Python Calculator) ---", flush=True)
    
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
        
        # New Phase 9A generation
        ext = extract_and_calculate(query, context_text)
        if ext.get("can_calculate", False):
            calc_result = safe_eval(ext.get("formula", ""), ext.get("variables", {}))
            final_answer = f"Reasoning: {ext.get('reasoning')}\nFormula: {ext.get('formula')}\nVariables: {ext.get('variables')}\nComputed Answer: {calc_result}"
        else:
            final_answer = f"Cannot calculate deterministically. Reasoning: {ext.get('reasoning')}"
        
        print(f" -> Top 3 Chunks: {[c[0][0] for c in top_3]}", flush=True)
        print(f" -> Correct Doc Retrieved: {retrieved_correct}", flush=True)
        print(f" -> Generated Answer:\n{final_answer}", flush=True)
        print(f" -> Expected: {q_data['answer']}", flush=True)
        
        result_obj = {
            "query": query,
            "top_3_chunks": [c[0][0] for c in top_3],
            "correct_doc_retrieved": retrieved_correct,
            "generated_answer": final_answer,
            "expected_answer": q_data['answer'],
            "context_text": context_text,
            "extraction": ext
        }
        results.append(result_obj)
        
        out_file = os.path.join(data_dir, "phase9a_results.jsonl")
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
    
    # Append custom multiple-figures / percentage change questions
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
    print(f"Selected {len(test_questions)} questions for evaluation.", flush=True)
    
    print("\n[Step 1] Loading Upgraded Index...", flush=True)
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
                metadata = {
                    "doc_name": record["doc_name"],
                    "page_number": record.get("page_number", 0),
                    "chunk_id": record["chunk_id"]
                }
                new_docs.append(Document(page_content=record["text"], metadata=metadata))
        if new_docs:
            print(f"Embedding {len(new_docs)} new multimodal chunks into FAISS...", flush=True)
            upgraded_faiss.add_documents(new_docs)
            print("Finished embedding new chunks.", flush=True)
    
    # Remove existing phase9a file to start fresh
    out_file = os.path.join(data_dir, "phase9a_results.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)
        
    run_evaluation(data_dir, upgraded_faiss, baseline_bm25, chunk_store, reranker, test_questions)

if __name__ == "__main__":
    main()
