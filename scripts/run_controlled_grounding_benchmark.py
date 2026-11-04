"""
scripts/run_controlled_grounding_benchmark.py
=============================================
Controlled Stage 3 A/B Retrieval Benchmark:
Arm A (Frozen Baseline) vs Arm B (Experiment L: Offline Entity Grounding).

Configurations:
- Arm A (Frozen Baseline):
  * Corpus:  data/heldout_stage3/processed/heldout_enriched_chunks.jsonl
  * Indexes: data/heldout_stage3/indexes/ (BM25 + FAISS BGE-M3)
- Arm B (Experiment L Grounded):
  * Corpus:  data/heldout_stage3/processed/heldout_grounded_chunks.jsonl
  * Indexes: data/heldout_stage3/indexes_grounded_exp_l/ (BM25 + FAISS BGE-M3)

Strict Experimental Controls:
- Same 100 questions from Stage 3 held-out validation suite.
- Same candidate pool depth: K_pool = 50.
- Same query-time coreference configuration (FinancialQueryExpander with enable_coreference=True).
- Same retrieval paths: BM25 (bm25s) + Dense (BGE-M3) with RRF (k=60).
- Same reranker: BAAI/bge-reranker-base (FP16, max_length=512, top_k=10).
- Same gold evidence criteria and 1-indexed physical PDF page matching.
- Zero overwrite of baseline indexes, corpus, or prior benchmark reports.
- Output directory: outputs/experiments/experiment_l_grounding_benchmark/
"""

import os
import sys
import time
import json
import csv
import hashlib
import torch
from collections import defaultdict
from typing import Dict, Any, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.financial_expander import FinancialQueryExpander
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.phase2_reranker import Phase2Reranker

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l_grounding_benchmark")


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_hits(reranked_chunks: List[Dict[str, Any]], gold_doc: str, target_pages: List[int]) -> int:
    """Finds best 1-indexed rank of a valid gold evidence chunk."""
    for rank, c in enumerate(reranked_chunks, start=1):
        c_doc = c.get("doc_name")
        c_page = c.get("page", 0)
        if c_doc == gold_doc and c_page in target_pages:
            return rank
    return 999


def compute_metrics(ranks: List[int], pool_hits: List[int], lats: List[float], peak_vram: float) -> Dict[str, Any]:
    n = len(ranks)
    r1 = sum(1 for r in ranks if r <= 1) / n * 100
    r3 = sum(1 for r in ranks if r <= 3) / n * 100
    r5 = sum(1 for r in ranks if r <= 5) / n * 100
    r10 = sum(1 for r in ranks if r <= 10) / n * 100
    p5 = sum(1 for r in ranks if r <= 5) / (n * 5) * 100
    p10 = sum(1 for r in ranks if r <= 10) / (n * 10) * 100
    pool_rec = sum(pool_hits) / n * 100
    mrr = sum(1.0 / r for r in ranks if r <= 10) / n
    avg_lat = sum(lats) / n if n > 0 else 0.0

    return {
        "n_questions": n,
        "pool_recall": round(pool_rec, 2),
        "mrr": round(mrr, 4),
        "recall_at_1": round(r1, 2),
        "recall_at_3": round(r3, 2),
        "recall_at_5": round(r5, 2),
        "recall_at_10": round(r10, 2),
        "precision_at_5": round(p5, 2),
        "precision_at_10": round(p10, 2),
        "avg_latency_sec": round(avg_lat, 4),
        "peak_vram_mb": round(peak_vram, 2),
        "total_hits_top10": sum(1 for r in ranks if r <= 10),
        "total_pool_hits": sum(pool_hits)
    }


def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("EXPERIMENT L: STAGE 3 CONTROLLED A/B RETRIEVAL BENCHMARK")
    print("Arm A (Frozen Baseline) vs Arm B (Experiment L: Offline Entity Grounding)")
    print("=" * 80)

    # 1. Device and Shared Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Verify Baseline Immutability
    baseline_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    baseline_corpus_hash = compute_sha256(baseline_corpus_path)
    expected_baseline_hash = "ea4264f5d7e889d30fc4ee9147e7286b67ffa76300967fee3651002974062845"
    assert baseline_corpus_hash == expected_baseline_hash, "Baseline corpus hash mismatch!"
    print(f"Verified Baseline Corpus SHA256: {baseline_corpus_hash} (Immutable)")

    # 3. Load Arm A (Baseline Corpus & Indexes)
    print("\n[Loading Arm A: Frozen Baseline]...")
    arm_a_chunks = {}
    with open(baseline_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            arm_a_chunks[c["chunk_id"]] = c

    arm_a_faiss_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3")
    arm_a_bm25_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25")
    arm_a_faiss = FAISS.load_local(arm_a_faiss_path, embeddings, allow_dangerous_deserialization=True)
    arm_a_bm25 = BM25Retriever(arm_a_bm25_path, load_corpus=True)

    # 4. Load Arm B (Grounded Corpus & Indexes)
    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    grounded_corpus_hash = compute_sha256(grounded_corpus_path)
    print(f"\n[Loading Arm B: Grounded Exp L]...")
    print(f"Grounded Corpus SHA256: {grounded_corpus_hash}")
    arm_b_chunks = {}
    with open(grounded_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            arm_b_chunks[c["chunk_id"]] = c

    arm_b_faiss_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "faiss_bge_m3")
    arm_b_bm25_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "bm25")
    arm_b_faiss = FAISS.load_local(arm_b_faiss_path, embeddings, allow_dangerous_deserialization=True)
    arm_b_bm25 = BM25Retriever(arm_b_bm25_path, load_corpus=True)

    # 5. Build Identical Retrievers and Rerankers
    # Both arms use identical query-time coreference configuration
    expander = FinancialQueryExpander(enable_coreference=True)

    mpr_arm_a = MultiPathRetriever(
        faiss_index=arm_a_faiss, bm25_retriever=arm_a_bm25, chunk_store=arm_a_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=expander
    )
    reranker_arm_a = Phase2Reranker(
        model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True
    )

    mpr_arm_b = MultiPathRetriever(
        faiss_index=arm_b_faiss, bm25_retriever=arm_b_bm25, chunk_store=arm_b_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=expander
    )
    reranker_arm_b = Phase2Reranker(
        model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True
    )

    # 6. Load Questions
    questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "heldout_questions.jsonl")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"\nLoaded {len(questions)} Stage 3 held-out evaluation questions.")

    # -------------------------------------------------------------------------
    # Execution: Arm A (Frozen Baseline)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 1: EVALUATING ARM A (FROZEN BASELINE INDEXES)")
    print("-" * 80)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_a = []
    pool_hits_a = []
    lats_a = []

    for idx, q_data in enumerate(questions, start=1):
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_arm_a.retrieve_multi_path(
            query=q_text, k_pool=50, enable_bm25=True, enable_dense=True, enable_metadata=False, enable_hyde=False
        )
        cands = ret_pack["candidates"]

        # Ensure no duplicate chunk IDs
        cid_set = set(c["chunk_id"] for c in cands)
        assert len(cid_set) == len(cands), f"Duplicate candidates detected in Arm A for {q_data.get('question_id')}"

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_a.append(1 if pool_hit else 0)

        reranked = reranker_arm_a.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_a.append(time.time() - t_q)

        rank_a = evaluate_hits(reranked, doc_name, target_pages)
        ranks_a.append(rank_a)

    peak_vram_a = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_a = compute_metrics(ranks_a, pool_hits_a, lats_a, peak_vram_a)
    print(f"  Arm A Metrics: MRR={metrics_a['mrr']}, R@10={metrics_a['recall_at_10']}%, Pool={metrics_a['pool_recall']}%, Lat={metrics_a['avg_latency_sec']}s")

    # -------------------------------------------------------------------------
    # Execution: Arm B (Experiment L: Grounded Indexes)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 2: EVALUATING ARM B (EXPERIMENT L GROUNDED INDEXES)")
    print("-" * 80)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_b = []
    pool_hits_b = []
    lats_b = []
    question_comparisons = []

    for idx, q_data in enumerate(questions, start=1):
        qid = q_data.get("question_id", f"stage3_heldout_{idx:04d}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_arm_b.retrieve_multi_path(
            query=q_text, k_pool=50, enable_bm25=True, enable_dense=True, enable_metadata=False, enable_hyde=False
        )
        cands = ret_pack["candidates"]
        exp_res = ret_pack["expansion_result"]

        # Ensure no duplicate chunk IDs
        cid_set = set(c["chunk_id"] for c in cands)
        assert len(cid_set) == len(cands), f"Duplicate candidates detected in Arm B for {qid}"

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_b.append(1 if pool_hit else 0)

        reranked = reranker_arm_b.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_b.append(time.time() - t_q)

        rank_b = evaluate_hits(reranked, doc_name, target_pages)
        ranks_b.append(rank_b)

        rank_a = ranks_a[idx - 1]
        pool_a = pool_hits_a[idx - 1]

        # Shift classification
        move = "TIE"
        if rank_b < rank_a:
            move = f"+{rank_a - rank_b}" if rank_a != 999 else f"NEW_HIT_#{rank_b}"
        elif rank_b > rank_a:
            move = f"-{rank_b - rank_a}" if rank_b != 999 else f"LOST_HIT_#{rank_a}"

        pool_move = "SAME"
        if pool_hit and not pool_a:
            pool_move = "RECOVERED_POOL"
        elif pool_a and not pool_hit:
            pool_move = "LOST_POOL"

        question_comparisons.append({
            "index": idx,
            "id": qid,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "target_pages": str(target_pages),
            "pool_arm_a": bool(pool_a),
            "pool_arm_b": bool(pool_hit),
            "pool_shift": pool_move,
            "rank_arm_a": rank_a if rank_a <= 10 else "NO_HIT",
            "rank_arm_b": rank_b if rank_b <= 10 else "NO_HIT",
            "rr_arm_a": round(1.0 / rank_a, 4) if rank_a <= 10 else 0.0,
            "rr_arm_b": round(1.0 / rank_b, 4) if rank_b <= 10 else 0.0,
            "movement": move,
            "is_coref_active": exp_res.is_coreference_resolved,
            "coref_term": exp_res.coreference_term or "",
            "detected_company": exp_res.detected_company or "",
            "detected_ticker": exp_res.detected_ticker or "",
            "question": q_text
        })

    peak_vram_b = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_b = compute_metrics(ranks_b, pool_hits_b, lats_b, peak_vram_b)
    print(f"  Arm B Metrics: MRR={metrics_b['mrr']}, R@10={metrics_b['recall_at_10']}%, Pool={metrics_b['pool_recall']}%, Lat={metrics_b['avg_latency_sec']}s")

    # -------------------------------------------------------------------------
    # Analysis & Save Outputs
    # -------------------------------------------------------------------------
    # Movement aggregations
    improved = [q for q in question_comparisons if q["movement"].startswith("+") or "NEW_HIT" in q["movement"]]
    regressed = [q for q in question_comparisons if q["movement"].startswith("-") or "LOST_HIT" in q["movement"]]
    ties = [q for q in question_comparisons if q["movement"] == "TIE"]

    new_pool_recoveries = [q for q in question_comparisons if q["pool_shift"] == "RECOVERED_POOL"]
    lost_pool = [q for q in question_comparisons if q["pool_shift"] == "LOST_POOL"]

    # 1. Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "stage3_grounding_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(question_comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(question_comparisons)
    print(f"\nSaved question comparison CSV to: {csv_path}")

    # 2. Save Scorecard JSON
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
        },
        "stage3_heldout_n100": {
            "arm_a_frozen_baseline": metrics_a,
            "arm_b_experiment_l": metrics_b,
            "delta": {
                "pool_recall": round(metrics_b["pool_recall"] - metrics_a["pool_recall"], 2),
                "mrr": round(metrics_b["mrr"] - metrics_a["mrr"], 4),
                "recall_at_1": round(metrics_b["recall_at_1"] - metrics_a["recall_at_1"], 2),
                "recall_at_3": round(metrics_b["recall_at_3"] - metrics_a["recall_at_3"], 2),
                "recall_at_5": round(metrics_b["recall_at_5"] - metrics_a["recall_at_5"], 2),
                "recall_at_10": round(metrics_b["recall_at_10"] - metrics_a["recall_at_10"], 2),
                "precision_at_5": round(metrics_b["precision_at_5"] - metrics_a["precision_at_5"], 2),
                "precision_at_10": round(metrics_b["precision_at_10"] - metrics_a["precision_at_10"], 2),
                "latency_sec": round(metrics_b["avg_latency_sec"] - metrics_a["avg_latency_sec"], 4),
                "peak_vram_mb": round(metrics_b["peak_vram_mb"] - metrics_a["peak_vram_mb"], 2),
            },
            "movements": {
                "improved_count": len(improved),
                "regressed_count": len(regressed),
                "tie_count": len(ties),
                "new_pool_recoveries_count": len(new_pool_recoveries),
                "lost_pool_count": len(lost_pool)
            }
        }
    }

    scorecard_path = os.path.join(OUTPUT_DIR, "grounding_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved Scorecard JSON to: {scorecard_path}")

    # Inspect Key Cases
    q75 = next((q for q in question_comparisons if q["id"] == "stage3_heldout_0075"), None)
    q72 = next((q for q in question_comparisons if q["id"] == "stage3_heldout_0072"), None)

    print("\n" + "=" * 80)
    print("KEY CASES FORENSIC STATUS")
    print("=" * 80)
    if q75:
        print(f"stage3_heldout_0075 (Eli Lilly / tirzepatide): Arm A = {q75['rank_arm_a']}, Arm B = {q75['rank_arm_b']} ({q75['movement']})")
    if q72:
        print(f"stage3_heldout_0072 (Mounjaro revenue):        Arm A = {q72['rank_arm_a']}, Arm B = {q72['rank_arm_b']} ({q72['movement']})")

    print("\n" + "=" * 80)
    print("EXPERIMENT L BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Candidate Pool Recall: Arm A = {metrics_a['pool_recall']}% -> Arm B = {metrics_b['pool_recall']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['pool_recall']:+g}%)")
    print(f"MRR:                   Arm A = {metrics_a['mrr']} -> Arm B = {metrics_b['mrr']} (Delta: {scorecard['stage3_heldout_n100']['delta']['mrr']:+g})")
    print(f"Recall@1:              Arm A = {metrics_a['recall_at_1']}% -> Arm B = {metrics_b['recall_at_1']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['recall_at_1']:+g}%)")
    print(f"Recall@5:              Arm A = {metrics_a['recall_at_5']}% -> Arm B = {metrics_b['recall_at_5']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['recall_at_5']:+g}%)")
    print(f"Recall@10:             Arm A = {metrics_a['recall_at_10']}% -> Arm B = {metrics_b['recall_at_10']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['recall_at_10']:+g}%)")
    print(f"Precision@5:           Arm A = {metrics_a['precision_at_5']}% -> Arm B = {metrics_b['precision_at_5']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['precision_at_5']:+g}%)")
    print(f"Precision@10:          Arm A = {metrics_a['precision_at_10']}% -> Arm B = {metrics_b['precision_at_10']}% (Delta: {scorecard['stage3_heldout_n100']['delta']['precision_at_10']:+g}%)")
    print(f"Latency:               Arm A = {metrics_a['avg_latency_sec']}s -> Arm B = {metrics_b['avg_latency_sec']}s")
    print(f"Question Movements:    {len(improved)} Improved, {len(regressed)} Regressed, {len(ties)} Unchanged")
    print(f"New Pool Recoveries:   {len(new_pool_recoveries)} queries entered pool")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
