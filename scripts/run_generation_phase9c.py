"""
Phase 9C: Step 2 - Downstream Extraction & Generation Evaluation
Loads Qwen2.5-7B-Instruct with the 3-epoch DPO adapter in 4-bit.
Evaluates Top-3 vs Top-5 retrieved contexts from the Hybrid Expanded + Statement-Aware Reranked path.
Generates comprehensive JSON and JSONL outputs and configuration.
"""

import os
import json
import time
import re
import torch
import warnings
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def is_value_in_context(val, context_text):
    if val is None:
        return True
    v_str = str(val).strip()
    if v_str in context_text or v_str.lower() in context_text.lower():
        return True
    if v_str.endswith(".0"):
        v_str = v_str[:-2]
    ctx = context_text.replace(",", "")
    if v_str in ctx or v_str.lower() in ctx.lower():
        return True
    try:
        if isinstance(val, (int, float)):
            v_str_commas = f"{val:,}" if isinstance(val, int) else f"{int(val):,}" if val.is_integer() else str(val)
            if v_str_commas in context_text:
                return True
    except Exception:
        pass
    return False

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

def extract_first_valid_json(text):
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
        escape = False
    return None

def evaluate_generation(model, tokenizer, query, context_text):
    prompt = f"Answer the user's financial question based ONLY on the provided context.\nSelect one predefined operation: 'none', 'extract_single_value', 'absolute_difference', 'percentage_change'.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nJSON Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - t0
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    parsed = extract_first_valid_json(gen_text)
    valid = True
    errors = []
    op = None
    old_val = None
    new_val = None
    calc = None
    
    if parsed is None:
        valid = False
        errors.append("No valid JSON parsed")
    else:
        op = parsed.get("operation", "none")
        old_val = parsed.get("old_value")
        new_val = parsed.get("new_value")
        if old_val is None and new_val is None:
            if "value" in parsed:
                new_val = parsed["value"]
            elif "values" in parsed and isinstance(parsed["values"], list):
                if len(parsed["values"]) > 1:
                    old_val, new_val = parsed["values"][0], parsed["values"][1]
                elif len(parsed["values"]) == 1:
                    new_val = parsed["values"][0]
                    
        if op not in ["none", "extract_single_value", "absolute_difference", "percentage_change"]:
            valid = False
            errors.append(f"Invalid operation: {op}")
            
        if op in ["absolute_difference", "percentage_change"] and (old_val is None or new_val is None):
            valid = False
            errors.append("Missing required old_value or new_value")
            
        if op == "extract_single_value" and (old_val is None and new_val is None):
            valid = False
            errors.append("extract_single_value requires at least one value")
            
        if old_val is not None and not is_value_in_context(old_val, context_text):
            valid = False
            errors.append(f"old_value '{old_val}' not in context")
            
        if new_val is not None and not is_value_in_context(new_val, context_text):
            valid = False
            errors.append(f"new_value '{new_val}' not in context")
            
        if valid:
            calc = safe_calculate(op, old_val, new_val) if op != "none" else None
            
    return {
        "raw_output": gen_text,
        "parsed_json": parsed,
        "operation": op,
        "old_value": old_val,
        "new_value": new_val,
        "calculation": calc,
        "valid": valid,
        "validation_errors": errors,
        "latency_seconds": round(latency, 3)
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    out_dir = os.path.join(base_dir, "outputs", "phase9c")
    os.makedirs(out_dir, exist_ok=True)
    
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    retrieval_data_file = os.path.join(out_dir, "retrieval_ablation_data.json")
    phase2_adapter_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_phase2")
    
    if not os.path.exists(retrieval_data_file):
        raise FileNotFoundError(f"Missing {retrieval_data_file}. Please run scripts/run_retrieval_phase9c.py first.")

    with open(retrieval_data_file, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    test_cases = retrieval_data["test_cases"]
    ablation_results = retrieval_data["ablation_results"]

    print("Loading chunk text store...")
    chunk_store = {}
    with open(orig_chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            chunk_store[record["chunk_id"]] = record
            chunk_store[i] = record
            
    if os.path.exists(new_chunks_file):
        with open(new_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                chunk_store[record["chunk_id"]] = record

    print("Loading 3-Epoch DPO Model for Downstream Extraction...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, phase2_adapter_dir)
    model.eval()

    generation_comparisons = []

    for i, tc in enumerate(test_cases):
        qid = tc["id"]
        q = tc["query"]
        p4_data = ablation_results["hybrid_expanded_statement_reranked"][i]
        
        # Build Top-3 Context
        ctx_top3_parts = []
        for cid in p4_data["final_top3"]:
            c = chunk_store.get(cid, {})
            ctx_top3_parts.append(f"--- Chunk {cid} (Page {c.get('page_number', 0)}) ---\n{c.get('text', '')}")
        ctx_top3 = "\n\n".join(ctx_top3_parts)
        
        # Build Top-5 Context
        ctx_top5_parts = []
        for cid in p4_data["final_top5"]:
            c = chunk_store.get(cid, {})
            ctx_top5_parts.append(f"--- Chunk {cid} (Page {c.get('page_number', 0)}) ---\n{c.get('text', '')}")
        ctx_top5 = "\n\n".join(ctx_top5_parts)

        print(f"\n=======================================================")
        print(f"Downstream Inference on {qid}: {tc['financial_metric']}")
        print(f"=======================================================")
        
        print(f"Running Top-3 Context ({len(ctx_top3)} chars)...")
        gen_top3 = evaluate_generation(model, tokenizer, q, ctx_top3)
        print(f"Top-3 Result: Op={gen_top3['operation']}, NewVal={gen_top3['new_value']}, Calc={gen_top3['calculation']}, Valid={gen_top3['valid']}")

        print(f"Running Top-5 Context ({len(ctx_top5)} chars)...")
        gen_top5 = evaluate_generation(model, tokenizer, q, ctx_top5)
        print(f"Top-5 Result: Op={gen_top5['operation']}, NewVal={gen_top5['new_value']}, Calc={gen_top5['calculation']}, Valid={gen_top5['valid']}")

        generation_comparisons.append({
            "qid": qid,
            "query": q,
            "expected_answer": tc["gold_answer"],
            "top3_chunks": p4_data["final_top3"],
            "top5_chunks": p4_data["final_top5"],
            "gen_top3": gen_top3,
            "gen_top5": gen_top5
        })

    # Save config to outputs/phase9c
    config_dict = {
        "hybrid_retrieval": {
            "dense_model": "BAAI/bge-m3",
            "sparse_model": "BM25Okapi",
            "k_dense": 20,
            "k_sparse": 20,
            "k_pool": 40,
            "k_rrf": 60
        },
        "financial_query_expansion": {
            "enabled": True,
            "mappings": ["CapEx / PP&E Purchases", "Net PP&E", "Debt Securities vs Obligations", "Net Sales / Revenue"],
            "intent_detection": True,
            "statement_detection": True
        },
        "statement_aware_reranking": {
            "reranker_model": "BAAI/bge-reranker-base",
            "use_fp16": True,
            "statement_boost_weight": 2.0,
            "company_boost_weight": 1.5,
            "period_boost_weight": 1.5,
            "table_boost_weight": 0.5,
            "min_confidence_for_boost": 0.5,
            "k_final": 5
        },
        "downstream_llm": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "adapter": "outputs/phase9b_dpo_phase2",
            "quantization": "4bit-nf4"
        }
    }
    with open(os.path.join(out_dir, "phase9c_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Save all results to outputs/phase9c
    out_results_file = os.path.join(out_dir, "phase9c_ablation_results.json")
    with open(out_results_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": config_dict,
            "retrieval_ablations": ablation_results,
            "generation_comparisons": generation_comparisons
        }, f, indent=2)

    # Save per-question JSONL results
    out_jsonl_file = os.path.join(out_dir, "phase9c_results.jsonl")
    with open(out_jsonl_file, "w", encoding="utf-8") as f:
        for i, tc in enumerate(test_cases):
            qid = tc["id"]
            rec = {
                "question_id": qid,
                "query": tc["query"],
                "gold_answer": tc["gold_answer"],
                "gold_chunks": tc["gold_chunks"],
                "reporting_period": tc["reporting_period"],
                "dense_only": ablation_results["dense_only"][i],
                "dense_plus_bm25": ablation_results["dense_plus_bm25"][i],
                "hybrid_expanded": ablation_results["hybrid_expanded"][i],
                "hybrid_expanded_statement_reranked": ablation_results["hybrid_expanded_statement_reranked"][i],
                "generation": generation_comparisons[i]
            }
            f.write(json.dumps(rec) + "\n")

    print(f"\nAll Phase 9C evaluations successfully saved to {out_results_file} and {out_jsonl_file}")

if __name__ == "__main__":
    main()
