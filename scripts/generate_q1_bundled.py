"""
Subprocess to run Qwen2.5-7B generation on Q1 bundled context in a fresh process.
"""

import os
import sys
import json
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def parse_json_from_text(text: str):
    pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            continue
    return None

def build_generator_prompt(query: str, context_chunks: list) -> str:
    ctx_parts = []
    for c in context_chunks:
        cid = c.get("chunk_id", "unknown")
        pg = c.get("page_number", 0)
        txt = c.get("text", "")
        ctx_parts.append(f"--- Chunk {cid} (Page {pg}) ---\n{txt}")
    context_str = "\n\n".join(ctx_parts)
    
    system_msg = (
        "You are an expert financial analyst. Answer the user's financial question based ONLY on the provided context. "
        "Select one predefined operation from: ['none', 'extract_single_value', 'absolute_difference', 'percentage_change']. "
        "Return your answer as a JSON object with 'operation' and relevant numeric keys ('value', 'values', 'explanation')."
    )
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

def main():
    ctx_file = os.path.join(BASE_DIR, "outputs", "phase1_chunk_bundling", "bundled_contexts.json")
    if not os.path.exists(ctx_file):
        raise FileNotFoundError(f"Missing {ctx_file}")
        
    with open(ctx_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    q1_query = data["q1_query"]
    ctx_70 = data["ctx_70"]
    
    print("Loading Generator (Qwen2.5-7B-Instruct + DPO Phase 2) in clean process...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    gen_model_name = "Qwen/Qwen2.5-7B-Instruct"
    gen_checkpoint_dir = os.path.join(BASE_DIR, "outputs", "phase9b_dpo_phase2")
    
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_base = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    gen_model = PeftModel.from_pretrained(gen_base, gen_checkpoint_dir)
    gen_model.eval()
    
    prompt = build_generator_prompt(q1_query, ctx_70)
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    
    t0 = time.time()
    with torch.no_grad():
        out_ids = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=gen_tokenizer.eos_token_id
        )
    gen_latency = time.time() - t0
    raw_gen_text = gen_tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    parsed_json = parse_json_from_text(raw_gen_text)
    
    out_res = {
        "raw_output": raw_gen_text,
        "parsed_json": parsed_json,
        "latency_seconds": round(gen_latency, 3)
    }
    
    out_file = os.path.join(BASE_DIR, "outputs", "phase1_chunk_bundling", "q1_generation_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_res, f, indent=2)
        
    print("Generation complete and saved to:", out_file)
    print("Raw output:", raw_gen_text)

if __name__ == "__main__":
    main()
