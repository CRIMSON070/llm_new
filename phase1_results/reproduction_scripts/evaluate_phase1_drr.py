"""
Phase 1: Domain-Specialized Document Re-Ranker (DRR) Evaluation Script.
Evaluates the trained BAAI/bge-reranker-v2-gemma adapter against:
1. Base BAAI/bge-reranker-base (zero heuristic boosts)
2. Base BAAI/bge-reranker-v2-gemma (adapter disabled, zero boosts) [Ablation]
3. Trained BAAI/bge-reranker-v2-gemma + Phase 1 DRR Adapter (zero heuristic boosts)
4. Historical Phase 9C Statement-Aware Reranker (with original GAAP heuristic boosts)

Outputs:
- outputs/phase1_drr_evaluation/per_question_results.json
- outputs/phase1_drr_evaluation/summary_metrics.json
- outputs/phase1_drr_evaluation/drr_evaluation_report.md
"""

import os
import sys
import json
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from FlagEmbedding import FlagReranker
from src.retrieval.statement_reranker import StatementAwareReranker

PROMPT_INSTRUCTION = (
    "Given a query A and a passage B, determine whether the passage contains an "
    "answer to the query by providing a prediction of either 'Yes' or 'No'."
)

def format_cross_encoder_text(query: str, passage: str) -> str:
    return f"A: {query}\nB: {passage}\n{PROMPT_INSTRUCTION}"

def compute_passage_score(model, tokenizer, query: str, passage: str, yes_token_id: int, max_length: int = 512) -> torch.Tensor:
    text = format_cross_encoder_text(query, passage)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False
    ).to(model.device)
    
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_backbone = model.base_model.model.model
        lm_head = model.base_model.model.lm_head
    elif hasattr(model, "model") and hasattr(model, "lm_head"):
        base_backbone = model.model
        lm_head = model.lm_head
    else:
        outputs = model(**inputs)
        return outputs.logits[:, -1, yes_token_id]
        
    outputs = base_backbone(**inputs)
    last_hidden = outputs.last_hidden_state[:, -1, :]
    score = lm_head(last_hidden)[:, yes_token_id]
    return score

def calculate_mrr(ranks: List[Optional[int]]) -> float:
    rr_list = [1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks]
    return sum(rr_list) / max(len(rr_list), 1)

def calculate_recall_at_k(ranks: List[Optional[int]], k: int) -> float:
    hits = [1.0 if (r is not None and 1 <= r <= k) else 0.0 for r in ranks]
    return sum(hits) / max(len(hits), 1)

def main():
    print("=" * 70)
    print("FinSage Phase 1: DRR Adapter Evaluation on Held-Out Questions Q1–Q6")
    print("=" * 70)
    
    out_dir = os.path.join(BASE_DIR, "outputs", "phase1_drr_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    ablation_path = os.path.join(BASE_DIR, "outputs", "phase9c", "retrieval_ablation_data.json")
    adapter_dir = os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter")
    
    if not os.path.exists(ablation_path):
        raise FileNotFoundError(f"Missing {ablation_path}")
        
    ablation_data = json.load(open(ablation_path, encoding="utf-8"))
    test_cases = ablation_data["test_cases"]
    historical_p4_results = ablation_data["ablation_results"]["hybrid_expanded_statement_reranked"]
    
    # 1. Load chunks store
    orig_chunks_file = os.path.join(data_dir, "processed", "parsed_chunks.jsonl")
    new_chunks_file = os.path.join(data_dir, "processed", "pymupdf_chunks.jsonl")
    
    print("Loading chunks store...")
    chunk_store = {}
    for path in [orig_chunks_file, new_chunks_file]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    c = json.loads(line)
                    chunk_store[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store)} chunks.")
    
    # Verify candidate pools across Q1-Q6
    queries_data = []
    for i, tc in enumerate(test_cases):
        qid = tc["id"]
        q = tc["query"]
        gold_chunks = set(tc["gold_chunks"])
        candidates_cids = ablation_data["ablation_results"]["hybrid_expanded"][i]["candidates"]
        cand_objs = [chunk_store[cid] for cid in candidates_cids if cid in chunk_store]
        queries_data.append({
            "qid": qid,
            "query": q,
            "gold_chunks": list(gold_chunks),
            "candidates_cids": candidates_cids,
            "candidates_objs": cand_objs
        })
    print(f"Prepared {len(queries_data)} test cases with exact candidate pools.")
    
    # -------------------------------------------------------------
    # Config 1: Base BAAI/bge-reranker-base (zero boosts)
    # -------------------------------------------------------------
    print("\n[Config 1/3] Evaluating Base BAAI/bge-reranker-base (Zero Boosts)...")
    torch.cuda.reset_peak_memory_stats()
    t0_base = time.time()
    
    base_reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    c1_results = {}
    
    for qd in queries_data:
        qid = qd["qid"]
        q = qd["query"]
        cand_objs = qd["candidates_objs"]
        gold_set = set(qd["gold_chunks"])
        
        pairs = [[q, c["text"]] for c in cand_objs]
        scores = base_reranker.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
            
        ranked = []
        for c, s in zip(cand_objs, scores):
            ranked.append({
                "chunk_id": c["chunk_id"],
                "score": round(float(s), 4),
                "is_gold": c["chunk_id"] in gold_set
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        gold_ranks = [idx + 1 for idx, item in enumerate(ranked) if item["is_gold"]]
        min_gold_rank = min(gold_ranks) if gold_ranks else None
        
        c1_results[qid] = {
            "ranked_chunks": ranked,
            "top3": [r["chunk_id"] for r in ranked[:3]],
            "top5": [r["chunk_id"] for r in ranked[:5]],
            "gold_rank": min_gold_rank,
            "gold_in_top3": min_gold_rank is not None and min_gold_rank <= 3,
            "gold_in_top5": min_gold_rank is not None and min_gold_rank <= 5,
            "gold_in_top10": min_gold_rank is not None and min_gold_rank <= 10,
            "gold_in_pool": min_gold_rank is not None
        }
    
    t_base_total = time.time() - t0_base
    vram_base = torch.cuda.max_memory_allocated() / 1e9
    del base_reranker
    torch.cuda.empty_cache()
    print(f"Base Reranker Complete. Time: {t_base_total:.2f}s | Peak VRAM: {vram_base:.2f} GB")
    
    # -------------------------------------------------------------
    # Config 2: BAAI/bge-reranker-v2-gemma with Phase 1 DRR Adapter (zero boosts)
    # (Also records Base Gemma ablation with adapter disabled)
    # -------------------------------------------------------------
    print("\n[Config 2/3] Evaluating BAAI/bge-reranker-v2-gemma + Phase 1 DRR Adapter (Zero Boosts)...")
    torch.cuda.reset_peak_memory_stats()
    t0_drr = time.time()
    
    model_name = "BAAI/bge-reranker-v2-gemma"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    drr_model = PeftModel.from_pretrained(base_model, adapter_dir)
    drr_model.eval()
    
    # Adapter Verification
    test_q = queries_data[0]["query"]
    test_text = queries_data[0]["candidates_objs"][0]["text"]
    with torch.no_grad():
        score_drr_check = compute_passage_score(drr_model, tokenizer, test_q, test_text, yes_token_id).item()
        with drr_model.disable_adapter():
            score_base_check = compute_passage_score(drr_model, tokenizer, test_q, test_text, yes_token_id).item()
            
    adapter_verified = (score_drr_check != score_base_check)
    print(f"Adapter Loaded & Verified: {adapter_verified} (DRR Score: {score_drr_check:.4f} vs Base Gemma: {score_base_check:.4f})")
    
    c2_drr_results = {}
    c2_base_gemma_results = {}
    
    for qd in queries_data:
        qid = qd["qid"]
        q = qd["query"]
        cand_objs = qd["candidates_objs"]
        gold_set = set(qd["gold_chunks"])
        
        drr_ranked = []
        base_gemma_ranked = []
        
        with torch.no_grad():
            for c in cand_objs:
                cid = c["chunk_id"]
                txt = c["text"]
                
                # DRR Adapted Score
                s_drr = compute_passage_score(drr_model, tokenizer, q, txt, yes_token_id).item()
                drr_ranked.append({
                    "chunk_id": cid,
                    "score": round(float(s_drr), 4),
                    "is_gold": cid in gold_set
                })
                
                # Base Gemma Score (Ablation)
                with drr_model.disable_adapter():
                    s_base_g = compute_passage_score(drr_model, tokenizer, q, txt, yes_token_id).item()
                base_gemma_ranked.append({
                    "chunk_id": cid,
                    "score": round(float(s_base_g), 4),
                    "is_gold": cid in gold_set
                })
                
        drr_ranked.sort(key=lambda x: x["score"], reverse=True)
        base_gemma_ranked.sort(key=lambda x: x["score"], reverse=True)
        
        drr_gold_ranks = [idx + 1 for idx, item in enumerate(drr_ranked) if item["is_gold"]]
        min_drr_gold_rank = min(drr_gold_ranks) if drr_gold_ranks else None
        
        base_g_gold_ranks = [idx + 1 for idx, item in enumerate(base_gemma_ranked) if item["is_gold"]]
        min_base_g_gold_rank = min(base_g_gold_ranks) if base_g_gold_ranks else None
        
        c2_drr_results[qid] = {
            "ranked_chunks": drr_ranked,
            "top3": [r["chunk_id"] for r in drr_ranked[:3]],
            "top5": [r["chunk_id"] for r in drr_ranked[:5]],
            "gold_rank": min_drr_gold_rank,
            "gold_in_top3": min_drr_gold_rank is not None and min_drr_gold_rank <= 3,
            "gold_in_top5": min_drr_gold_rank is not None and min_drr_gold_rank <= 5,
            "gold_in_top10": min_drr_gold_rank is not None and min_drr_gold_rank <= 10,
            "gold_in_pool": min_drr_gold_rank is not None
        }
        
        c2_base_gemma_results[qid] = {
            "ranked_chunks": base_gemma_ranked,
            "top3": [r["chunk_id"] for r in base_gemma_ranked[:3]],
            "top5": [r["chunk_id"] for r in base_gemma_ranked[:5]],
            "gold_rank": min_base_g_gold_rank,
            "gold_in_top3": min_base_g_gold_rank is not None and min_base_g_gold_rank <= 3,
            "gold_in_top5": min_base_g_gold_rank is not None and min_base_g_gold_rank <= 5,
            "gold_in_top10": min_base_g_gold_rank is not None and min_base_g_gold_rank <= 10,
            "gold_in_pool": min_base_g_gold_rank is not None
        }
        
    t_drr_total = time.time() - t0_drr
    vram_drr = torch.cuda.max_memory_allocated() / 1e9
    del drr_model, base_model
    torch.cuda.empty_cache()
    print(f"DRR Evaluation Complete. Time: {t_drr_total:.2f}s | Peak VRAM: {vram_drr:.2f} GB")
    
    # -------------------------------------------------------------
    # Config 3: Historical Phase 9C Reranking (With GAAP Heuristics)
    # -------------------------------------------------------------
    print("\n[Config 3/3] Processing Historical Phase 9C Reranking Results (With GAAP Boosts)...")
    c3_historical_results = {}
    
    for i, qd in enumerate(queries_data):
        qid = qd["qid"]
        gold_set = set(qd["gold_chunks"])
        p4_item = historical_p4_results[i]
        
        # Extract preserved reranked order from Phase 9C
        p4_top3 = p4_item["final_top3"]
        p4_top5 = p4_item["final_top5"]
        
        # Rank of gold chunk in Phase 9C
        # In Phase 9C, if gold was in top3 or top5:
        gold_in_top3 = any(g in gold_set for g in p4_top3)
        gold_in_top5 = any(g in gold_set for g in p4_top5)
        
        # Find exact rank in Phase 9C candidate order
        # Check if details list exists in p4_item
        p4_details = p4_item.get("reranked_details", [])
        p4_ranked_cids = [d["chunk_id"] for d in p4_details]
        
        # Complete pool rank
        gold_ranks_p4 = [idx + 1 for idx, cid in enumerate(p4_ranked_cids) if cid in gold_set]
        min_p4_gold_rank = min(gold_ranks_p4) if gold_ranks_p4 else None
        if min_p4_gold_rank is None and gold_in_top5:
            # check top5
            for rank_idx, cid in enumerate(p4_top5, 1):
                if cid in gold_set:
                    min_p4_gold_rank = rank_idx
                    break
                    
        c3_historical_results[qid] = {
            "top3": p4_top3,
            "top5": p4_top5,
            "gold_rank": min_p4_gold_rank,
            "gold_in_top3": gold_in_top3,
            "gold_in_top5": gold_in_top5,
            "gold_in_top10": p4_item.get("recall_at_10", False),
            "gold_in_pool": p4_item.get("recall_at_40", True),
            "reranked_details": p4_details,
            "latency": p4_item.get("latency", 0.3)
        }
        
    # -------------------------------------------------------------
    # Compute Comparative Metrics
    # -------------------------------------------------------------
    configs = [
        ("Base BGE-Reranker-base (Zero Boosts)", c1_results),
        ("Base BGE-Reranker-v2-Gemma (Zero Boosts)", c2_base_gemma_results),
        ("Trained Phase 1 DRR Adapter (Zero Boosts)", c2_drr_results),
        ("Historical Phase 9C (With GAAP Boosts)", c3_historical_results)
    ]
    
    summary_metrics = {}
    for cfg_name, res_dict in configs:
        ranks = [res_dict[f"Q{i}"]["gold_rank"] for i in range(1, 7)]
        mrr = calculate_mrr(ranks)
        r3 = calculate_recall_at_k(ranks, 3)
        r5 = calculate_recall_at_k(ranks, 5)
        r10 = calculate_recall_at_k(ranks, 10)
        r40 = sum(1.0 for r in ranks if r is not None) / len(ranks)
        
        summary_metrics[cfg_name] = {
            "MRR": round(mrr, 4),
            "Recall@3": round(r3, 4),
            "Recall@5": round(r5, 4),
            "Recall@10": round(r10, 4),
            "Recall@40": round(r40, 4),
            "ranks_per_question": {f"Q{i}": res_dict[f"Q{i}"]["gold_rank"] for i in range(1, 7)}
        }
        
    summary_metrics["hardware_telemetry"] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "base_reranker_peak_vram_gb": round(vram_base, 2),
        "drr_peak_vram_gb": round(vram_drr, 2),
        "base_reranker_total_time_s": round(t_base_total, 2),
        "drr_total_time_s": round(t_drr_total, 2),
        "adapter_active_verified": adapter_verified
    }
    
    # -------------------------------------------------------------
    # Per-Question Detailed Table
    # -------------------------------------------------------------
    per_question_table = []
    for i in range(1, 7):
        qid = f"Q{i}"
        qd = queries_data[i - 1]
        
        r_base = c1_results[qid]["gold_rank"]
        r_base_g = c2_base_gemma_results[qid]["gold_rank"]
        r_drr = c2_drr_results[qid]["gold_rank"]
        r_p9c = c3_historical_results[qid]["gold_rank"]
        
        delta_vs_base = (r_base - r_drr) if (r_base is not None and r_drr is not None) else None
        
        per_question_table.append({
            "qid": qid,
            "query": qd["query"],
            "gold_chunks": qd["gold_chunks"],
            "gold_rank_base": r_base,
            "gold_rank_base_gemma": r_base_g,
            "gold_rank_drr": r_drr,
            "gold_rank_p9c_historical": r_p9c,
            "delta_rank_drr_vs_base": delta_vs_base,
            "in_top3_base": c1_results[qid]["gold_in_top3"],
            "in_top3_drr": c2_drr_results[qid]["gold_in_top3"],
            "in_top3_p9c": c3_historical_results[qid]["gold_in_top3"],
            "drr_top3_chunks": c2_drr_results[qid]["top3"],
            "base_top3_chunks": c1_results[qid]["top3"],
            "p9c_top3_chunks": c3_historical_results[qid]["top3"]
        })
        
    # Save JSON artifacts
    per_q_file = os.path.join(out_dir, "per_question_results.json")
    with open(per_q_file, 'w', encoding='utf-8') as f:
        json.dump(per_question_table, f, indent=2)
        
    summary_file = os.path.join(out_dir, "summary_metrics.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
        
    # Generate Comprehensive Markdown Report
    report_file = os.path.join(out_dir, "drr_evaluation_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Domain-Specialized Document Re-Ranker (DRR) Diagnostic Evaluation Report\n\n")
        f.write("> **Diagnostic Scope Notice:** Held-out test set size is $N=6$ (Q1–Q6). This represents a small diagnostic benchmark to evaluate ranking behavior, not a statistically conclusive large-scale evaluation.\n\n")
        
        f.write("## 1. Summary Metrics Comparison\n\n")
        f.write("| Configuration | MRR | Recall@3 | Recall@5 | Recall@10 | Recall@40 | Status / Paradigm |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|---|\n")
        f.write(f"| **1. Base BGE-Reranker-base** | {summary_metrics['Base BGE-Reranker-base (Zero Boosts)']['MRR']:.4f} | {summary_metrics['Base BGE-Reranker-base (Zero Boosts)']['Recall@3']*100:.1f}% | {summary_metrics['Base BGE-Reranker-base (Zero Boosts)']['Recall@5']*100:.1f}% | {summary_metrics['Base BGE-Reranker-base (Zero Boosts)']['Recall@10']*100:.1f}% | {summary_metrics['Base BGE-Reranker-base (Zero Boosts)']['Recall@40']*100:.1f}% | Baseline (Zero Boosts) |\n")
        f.write(f"| **2. Base Gemma Reranker (Ablation)** | {summary_metrics['Base BGE-Reranker-v2-Gemma (Zero Boosts)']['MRR']:.4f} | {summary_metrics['Base BGE-Reranker-v2-Gemma (Zero Boosts)']['Recall@3']*100:.1f}% | {summary_metrics['Base BGE-Reranker-v2-Gemma (Zero Boosts)']['Recall@5']*100:.1f}% | {summary_metrics['Base BGE-Reranker-v2-Gemma (Zero Boosts)']['Recall@10']*100:.1f}% | {summary_metrics['Base BGE-Reranker-v2-Gemma (Zero Boosts)']['Recall@40']*100:.1f}% | Unadapted Gemma (Zero Boosts) |\n")
        f.write(f"| **3. Trained Phase 1 DRR Adapter** | **{summary_metrics['Trained Phase 1 DRR Adapter (Zero Boosts)']['MRR']:.4f}** | **{summary_metrics['Trained Phase 1 DRR Adapter (Zero Boosts)']['Recall@3']*100:.1f}%** | **{summary_metrics['Trained Phase 1 DRR Adapter (Zero Boosts)']['Recall@5']*100:.1f}%** | **{summary_metrics['Trained Phase 1 DRR Adapter (Zero Boosts)']['Recall@10']*100:.1f}%** | {summary_metrics['Trained Phase 1 DRR Adapter (Zero Boosts)']['Recall@40']*100:.1f}% | **Paper-Faithful DPO (Zero Boosts)** |\n")
        f.write(f"| **4. Historical Phase 9C** | {summary_metrics['Historical Phase 9C (With GAAP Boosts)']['MRR']:.4f} | {summary_metrics['Historical Phase 9C (With GAAP Boosts)']['Recall@3']*100:.1f}% | {summary_metrics['Historical Phase 9C (With GAAP Boosts)']['Recall@5']*100:.1f}% | {summary_metrics['Historical Phase 9C (With GAAP Boosts)']['Recall@10']*100:.1f}% | {summary_metrics['Historical Phase 9C (With GAAP Boosts)']['Recall@40']*100:.1f}% | Historical (GAAP Heuristic Boosts) |\n\n")
        
        f.write("## 2. Per-Question Diagnostic Ranking Table (Q1–Q6)\n\n")
        f.write("| Q# | Financial Focus | Gold Chunks | Base Rank | Base Gemma | Trained DRR Rank | Phase 9C Rank | $\\Delta$ Rank (DRR vs Base) | DRR in Top-3? |\n")
        f.write("|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for row in per_question_table:
            d_str = f"+{row['delta_rank_drr_vs_base']}" if (row['delta_rank_drr_vs_base'] is not None and row['delta_rank_drr_vs_base'] > 0) else str(row['delta_rank_drr_vs_base'])
            top3_icon = "YES" if row['in_top3_drr'] else "NO"
            f.write(f"| **{row['qid']}** | {row['query'][:40]}... | `{', '.join(row['gold_chunks'])}` | #{row['gold_rank_base']} | #{row['gold_rank_base_gemma']} | **#{row['gold_rank_drr']}** | #{row['gold_rank_p9c_historical']} | {d_str} | {top3_icon} |\n")
            
        f.write("\n## 3. Top-3 Retrieved Chunks per Configuration\n\n")
        for row in per_question_table:
            f.write(f"### {row['qid']}: {row['query']}\n\n")
            f.write(f"- **Target Gold Evidence**: `{row['gold_chunks']}`\n")
            f.write(f"- **Base BGE-Reranker Top-3**: `{row['base_top3_chunks']}` (Gold Rank: #{row['gold_rank_base']})\n")
            f.write(f"- **Trained Phase 1 DRR Top-3**: `{row['drr_top3_chunks']}` (Gold Rank: **#{row['gold_rank_drr']}**)\n")
            f.write(f"- **Phase 9C Historical Top-3**: `{row['p9c_top3_chunks']}` (Gold Rank: #{row['gold_rank_p9c_historical']})\n\n")
            
        f.write("## 4. Hardware & Latency Telemetry\n\n")
        f.write(f"- **Device**: {summary_metrics['hardware_telemetry']['device']}\n")
        f.write(f"- **Adapter Active Verification**: {summary_metrics['hardware_telemetry']['adapter_active_verified']}\n")
        f.write(f"- **Base Reranker Time**: {summary_metrics['hardware_telemetry']['base_reranker_total_time_s']}s (Peak VRAM: {summary_metrics['hardware_telemetry']['base_reranker_peak_vram_gb']} GB)\n")
        f.write(f"- **DRR Gemma Time**: {summary_metrics['hardware_telemetry']['drr_total_time_s']}s (Peak VRAM: {summary_metrics['hardware_telemetry']['drr_peak_vram_gb']} GB)\n")
        
    print(f"\nEvaluation Completed Successfully!")
    print(f"Per-question results: {per_q_file}")
    print(f"Summary metrics: {summary_file}")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()
