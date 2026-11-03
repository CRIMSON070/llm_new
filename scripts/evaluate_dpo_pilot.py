import os
import json
import gc
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def is_value_in_context(val, context_text):
    if val is None:
        return True
    
    v_str = str(val)
    if v_str.endswith(".0"):
        v_str = v_str[:-2]
        
    ctx = context_text.replace(",", "")
    if v_str in ctx:
        return True
    v_str_commas = f"{val:,}" if isinstance(val, int) else f"{int(val):,}" if isinstance(val, float) and val.is_integer() else v_str
    if v_str_commas in context_text:
        return True
        
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
    raw_str = raw_str.strip()
    # Remove markdown fences if present
    if "```json" in raw_str:
        raw_str = raw_str.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_str:
        raw_str = raw_str.split("```")[1].split("```")[0].strip()
        
    # Extract first JSON object
    match = re.search(r'\{.*\}', raw_str, re.DOTALL)
    if match:
        return match.group(0)
    return raw_str

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench")
    processed_dir = os.path.join(data_dir, "processed")
    orig_chunks_file = os.path.join(processed_dir, "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(processed_dir, "pymupdf_chunks.jsonl")
    baseline_results_file = os.path.join(data_dir, "phase9b1_results.jsonl")
    adapter_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_pilot")
    out_file = os.path.join(data_dir, "phase9b", "phase9b3_dpo_evaluation.jsonl")
    
    # Load chunk stores
    chunk_store = {}
    if os.path.exists(orig_chunks_file):
        with open(orig_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                chunk_store[rec["chunk_id"]] = rec
    if os.path.exists(new_chunks_file):
        with open(new_chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                chunk_store[rec["chunk_id"]] = rec

    # Load baseline test queries (first 6 from phase9b1_results.jsonl)
    test_cases = []
    with open(baseline_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec["model"] == "qwen2.5:7b":
                test_cases.append(rec)
            if len(test_cases) == 6:
                break
                
    print(f"Loaded {len(test_cases)} evaluation cases.")

    # Load model and adapter
    print("Loading base model Qwen/Qwen2.5-7B-Instruct in 4-bit NF4...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print(f"Loading LoRA DPO adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print("Model ready for inference. Peak VRAM:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")

    eval_results = []
    
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
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n=================== EVALUATION Q{idx+1} ===================")
        print(f"Query: {query}")
        print(f"Raw Output:\n{gen_text}")
        
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
            
        if valid and parsed_json:
            op = parsed_json.get("operation", "none")
            old_val = parsed_json.get("old_value")
            new_val = parsed_json.get("new_value")
            
            if op not in ["none", "extract_single_value", "absolute_difference", "percentage_change"]:
                valid = False
                validation_errors.append(f"Invalid operation: {op}")
                
            if op in ["absolute_difference", "percentage_change"] and (old_val is None or new_val is None):
                valid = False
                validation_errors.append("Missing required old_value or new_value")
                
            if op == "extract_single_value" and (old_val is None and new_val is None):
                valid = False
                validation_errors.append("extract_single_value requires at least one of old_value or new_value")
                
            if not is_value_in_context(old_val, context_text):
                valid = False
                validation_errors.append(f"old_value {old_val} not in context")
                
            if not is_value_in_context(new_val, context_text):
                valid = False
                validation_errors.append(f"new_value {new_val} not in context")
                
            if valid:
                if op == "none":
                    final_answer = f"Operation: none\nExplanation: {parsed_json.get('explanation')}"
                else:
                    calc = safe_calculate(op, old_val, new_val)
                    final_answer = f"Operation: {op}\nOld: {old_val}, New: {new_val}\nUnits: {parsed_json.get('units')}\nComputed Answer: {calc}\nExplanation: {parsed_json.get('explanation')}"
            else:
                final_answer = f"REJECTED. Validation Errors: {validation_errors}\nExtraction: {parsed_json}"
        else:
            final_answer = f"REJECTED. Validation Errors: {validation_errors}"
            
        print(f"Final Output: {final_answer}")
        print(f"Expected: {expected_ans}")
        
        eval_record = {
            "model": "qwen2.5-7b-dpo-pilot",
            "query": query,
            "top_3_chunks": top_chunks,
            "expected_answer": expected_ans,
            "extraction": parsed_json,
            "validation_errors": validation_errors,
            "final_answer": final_answer,
            "valid": valid
        }
        eval_results.append(eval_record)
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in eval_results:
            f.write(json.dumps(r) + '\n')
            
    print(f"\nEvaluation complete! Saved results to {out_file}")

if __name__ == "__main__":
    main()
