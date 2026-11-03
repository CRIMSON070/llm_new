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

def safe_calculate(operation: str, old_val, new_val):
    if operation == "percentage_change":
        if old_val is None or new_val is None or old_val == 0:
            return "Error: Missing or zero values for percentage change."
        return round(((new_val - old_val) / old_val) * 100, 4)
    elif operation == "absolute_difference":
        if old_val is None or new_val is None:
            return "Error: Missing values for difference."
        return round(new_val - old_val, 4)
    elif operation == "extract_single_value":
        if new_val is not None:
            return new_val
        elif old_val is not None:
            return old_val
        else:
            return "Error: No value extracted."
    return "Error: Unknown operation."

def is_value_in_context(val, context_text):
    if val is None:
        return True
    
    v_str = str(val)
    if v_str.endswith(".0"):
        v_str = v_str[:-2]
        
    ctx = context_text.replace(",", "")
    if v_str in ctx:
        return True
    # Check if they wrote it with commas
    v_str_commas = f"{val:,}" if isinstance(val, int) else f"{int(val):,}" if val.is_integer() else v_str
    if v_str_commas in context_text:
        return True
        
    return False

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

def answer_text_question(query: str, context: str, model: str) -> str:
    prompt = f"Answer the following question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
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
    return "Failed to generate text answer."

def extract_and_calculate(query: str, context: str, model: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["none", "extract_single_value", "absolute_difference", "percentage_change"]
            },
            "old_value": {"type": ["number", "null"]},
            "new_value": {"type": ["number", "null"]},
            "units": {"type": ["string", "null"]},
            "reporting_periods": {"type": ["string", "null"]},
            "evidence_chunk_ids": {
                "type": "array",
                "items": {"type": "string"}
            },
            "explanation": {"type": "string"}
        },
        "required": ["operation", "explanation", "evidence_chunk_ids"]
    }
    
    prompt = f"""You are a strict financial extraction AI.
Answer the user's financial question based ONLY on the provided context.
You must extract the exact numerical values AS WRITTEN in the text (do not add zeros for millions/billions).
Select one predefined operation: 'none', 'extract_single_value', 'absolute_difference', 'percentage_change'.
If the question asks for textual reasoning and not math, select 'none'.

Context:
{context}

Question: {query}

Do NOT provide concrete numerical examples in your head. Read the context and output valid JSON matching the schema.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model, 
                "prompt": prompt, 
                "stream": False, 
                "options": {"temperature": 0.0}, 
                "format": schema
            },
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
    return {"operation": "error", "explanation": "Failed to generate valid response.", "evidence_chunk_ids": []}

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
            
    test_pdfs = ["3M_2018_10K", "ULTABEAUTY_2023Q4_EARNINGS", "ULTABEAUTY_2023_10K"]
    test_questions = [q for q in questions if q["doc_name"] in test_pdfs][:4]
    
    test_questions.append({
        "question": "Based on the statements of income, what is the absolute difference in net sales for 3M from 2017 to 2018? Also calculate the percentage change.",
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
    
    print("\n--- PHASE 3: STRICT SCHEMA EVALUATION ---", flush=True)
    out_file = os.path.join(data_dir, "phase9b1_results.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)

    for model in models_to_test:
        print(f"\n================ EVALUATING MODEL: {model} ================", flush=True)
        requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0})
        
        for idx, q_data in enumerate(test_questions):
            query, context_text, retrieved_correct, chunk_ids = retrieved_contexts[idx]
            print(f"\n[{model}] Q{idx+1}: {query}", flush=True)
            
            ext = extract_and_calculate(query, context_text, model)
            
            operation = ext.get("operation", "none")
            old_val = ext.get("old_value")
            new_val = ext.get("new_value")
            
            # Validation Step
            valid = True
            validation_errors = []
            
            if operation not in ["none", "extract_single_value", "absolute_difference", "percentage_change"]:
                valid = False
                validation_errors.append(f"Invalid operation: {operation}")
            
            if operation in ["absolute_difference", "percentage_change"] and (old_val is None or new_val is None):
                valid = False
                validation_errors.append("Missing required old_value or new_value")
                
            if not is_value_in_context(old_val, context_text):
                valid = False
                validation_errors.append(f"old_value {old_val} not found in context (Hallucination detected)")
                
            if not is_value_in_context(new_val, context_text):
                valid = False
                validation_errors.append(f"new_value {new_val} not found in context (Hallucination detected)")
                
            if not valid:
                final_answer = f"REJECTED. Validation Errors: {validation_errors}\nExtraction: {ext}"
            else:
                if operation == "none":
                    text_ans = answer_text_question(query, context_text, model)
                    final_answer = f"Operation: none (Text Answer Generated)\nExplanation: {ext.get('explanation')}\nAnswer: {text_ans}"
                else:
                    calc_result = safe_calculate(operation, old_val, new_val)
                    final_answer = f"Operation: {operation}\nOld: {old_val}, New: {new_val}\nUnits: {ext.get('units')}\nComputed Answer: {calc_result}\nExplanation: {ext.get('explanation')}"
            
            print(f" -> Result: {final_answer}", flush=True)
            print(f" -> Expected: {q_data['answer']}", flush=True)
            
            result_obj = {
                "model": model,
                "query": query,
                "top_3_chunks": chunk_ids,
                "correct_doc_retrieved": retrieved_correct,
                "generated_answer": final_answer,
                "expected_answer": q_data['answer'],
                "extraction": ext,
                "validation_errors": validation_errors
            }
            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_obj) + '\n')

if __name__ == "__main__":
    main()
