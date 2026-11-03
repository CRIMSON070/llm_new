import os
import json
import time
import gc
import re
import torch
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

def clean_json_str(raw_str):
    text = raw_str.strip()
    if "```json" in text:
        content_after = text.split("```json")[1].strip()
        # if the code block has a closing ```
        if "```" in content_after:
            return content_after.split("```")[0].strip()
    elif "```" in text:
        content_after = text.split("```")[1].strip()
        if "```" in content_after:
            return content_after.split("```")[0].strip()

    start = text.find('{')
    if start == -1:
        return text

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
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        pass
        escape = False
    
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def run_evaluation_for_model(model_name, model, tokenizer, test_cases, chunk_store):
    print(f"\n=======================================================")
    print(f"Running evaluation for: {model_name}")
    print(f"=======================================================")
    
    results = []
    total_time = 0.0
    torch.cuda.reset_peak_memory_stats(0)
    
    for idx, tc in enumerate(test_cases):
        query = tc["query"]
        top_chunks = tc["top_3_chunks"]
        expected_ans = tc["expected_answer"]
        
        # Build context
        context_parts = []
        for cid in top_chunks:
            if cid in chunk_store:
                c_data = chunk_store[cid]
                context_parts.append(f"--- Chunk {cid} (Page {c_data.get('page_number', 0)}) ---\n{c_data.get('text', '')}")
            else:
                context_parts.append(f"--- Chunk {cid} ---\n[Text not found]")
        context_text = "\n\n".join(context_parts)
        
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
        dt = time.time() - t0
        total_time += dt
        
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        parsed_json = None
        validation_errors = []
        valid = True
        try:
            cleaned = clean_json_str(gen_text)
            parsed_json = json.loads(cleaned)
        except Exception as e:
            valid = False
            validation_errors.append(f"JSON Parse Error: {e}")
            parsed_json = {"operation": "error", "raw_output": gen_text}
            
        op = None
        old_val = None
        new_val = None
        calc = None
        strict_schema = True
        
        if valid and isinstance(parsed_json, dict):
            op = parsed_json.get("operation", "none")
            old_val = parsed_json.get("old_value")
            new_val = parsed_json.get("new_value")
            
            # Check for non-strict schema fallbacks
            if old_val is None and new_val is None:
                if "value" in parsed_json:
                    new_val = parsed_json["value"]
                    strict_schema = False
                elif "values" in parsed_json and isinstance(parsed_json["values"], list):
                    strict_schema = False
                    if len(parsed_json["values"]) > 1:
                        old_val, new_val = parsed_json["values"][0], parsed_json["values"][1]
                    elif len(parsed_json["values"]) == 1:
                        new_val = parsed_json["values"][0]
            
            if op not in ["none", "extract_single_value", "absolute_difference", "percentage_change"]:
                valid = False
                validation_errors.append(f"Invalid operation: {op}")
                
            if op in ["absolute_difference", "percentage_change"] and (old_val is None or new_val is None):
                valid = False
                validation_errors.append("Missing required old_value or new_value")
                
            if op == "extract_single_value" and (old_val is None and new_val is None):
                valid = False
                validation_errors.append("extract_single_value requires at least one of old_value or new_value")
                
            # Context grounding check
            if old_val is not None and not is_value_in_context(old_val, context_text):
                valid = False
                validation_errors.append(f"old_value '{old_val}' not grounded in context")
                
            if new_val is not None and not is_value_in_context(new_val, context_text):
                valid = False
                validation_errors.append(f"new_value '{new_val}' not grounded in context")
                
            if valid:
                if op == "none":
                    final_answer = f"Operation: none\nExplanation: {parsed_json.get('explanation') or parsed_json.get('answer')}"
                else:
                    calc = safe_calculate(op, old_val, new_val)
                    final_answer = f"Operation: {op}\nOld: {old_val}, New: {new_val}\nUnits: {parsed_json.get('units') or parsed_json.get('unit')}\nComputed Answer: {calc}\nExplanation: {parsed_json.get('explanation')}"
            else:
                final_answer = f"REJECTED. Validation Errors: {validation_errors}\nExtraction: {parsed_json}"
        else:
            final_answer = f"REJECTED. Validation Errors: {validation_errors}\nRaw: {gen_text}"
            
        peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
        
        record = {
            "model": model_name,
            "q_idx": idx + 1,
            "query": query,
            "top_3_chunks": top_chunks,
            "expected_answer": expected_ans,
            "raw_output": gen_text,
            "parsed_json": parsed_json,
            "operation": op,
            "old_value": old_val,
            "new_value": new_val,
            "units": parsed_json.get("units") or parsed_json.get("unit") if isinstance(parsed_json, dict) else None,
            "reporting_periods": parsed_json.get("reporting_periods") if isinstance(parsed_json, dict) else None,
            "strict_schema_adherence": strict_schema and valid,
            "validation_errors": validation_errors,
            "python_calculation": calc,
            "final_answer": final_answer,
            "valid": valid,
            "latency_seconds": round(dt, 3),
            "peak_vram_gb": round(peak_vram, 3)
        }
        results.append(record)
        print(f"[{model_name}] Q{idx+1} ({dt:.2f}s) Valid={valid} Op={op} Final:\n{final_answer}\n")
        
    return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    baseline_results_file = os.path.join(data_dir, "phase9b1_results.jsonl")
    
    pilot_adapter_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_pilot")
    phase2_adapter_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_phase2")
    out_dir = phase2_adapter_dir
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase9b4_comparison_results.jsonl")

    # Load chunks
    chunk_store = {}
    for path in [orig_chunks_file, new_chunks_file]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    chunk_store[rec["chunk_id"]] = rec

    # Load Q1-Q6 test cases
    test_cases = []
    with open(baseline_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec["model"] == "qwen2.5:7b":
                test_cases.append(rec)
            if len(test_cases) == 6:
                break
    print(f"Loaded {len(test_cases)} evaluation cases.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_comparison_results = {}

    # 1. Base Model Qwen2.5-7B-Instruct Evaluation
    print("Loading Base Model Qwen2.5-7B-Instruct...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config,
        device_map="auto"
    )
    base_model.eval()
    all_comparison_results["base_qwen"] = run_evaluation_for_model(
        "Base Qwen2.5-7B-Instruct", base_model, tokenizer, test_cases, chunk_store
    )
    
    # 2. 1-Epoch Adapter Evaluation
    print(f"Loading 1-Epoch Adapter from {pilot_adapter_dir}...")
    pilot_model = PeftModel.from_pretrained(base_model, pilot_adapter_dir)
    pilot_model.eval()
    all_comparison_results["one_epoch_adapter"] = run_evaluation_for_model(
        "1-Epoch Pilot Adapter", pilot_model, tokenizer, test_cases, chunk_store
    )
    
    # Clean up and reload for 3-Epoch Adapter
    del pilot_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 3-Epoch Adapter Evaluation
    print(f"Loading fresh Base Model for 3-Epoch Adapter from {phase2_adapter_dir}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config,
        device_map="auto"
    )
    phase2_model = PeftModel.from_pretrained(base_model, phase2_adapter_dir)
    phase2_model.eval()
    all_comparison_results["three_epoch_adapter"] = run_evaluation_for_model(
        "3-Epoch Phase2 Adapter", phase2_model, tokenizer, test_cases, chunk_store
    )
    
    del phase2_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save all results to jsonl
    with open(out_file, "w", encoding="utf-8") as f:
        for m_key, records in all_comparison_results.items():
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"\nAll comparison evaluations saved to {out_file}")

if __name__ == "__main__":
    main()
