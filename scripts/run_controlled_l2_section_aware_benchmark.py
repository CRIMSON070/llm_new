"""
scripts/run_controlled_l2_section_aware_benchmark.py
===================================================
Controlled Stage 3 Tri-Arm Retrieval Benchmark:
- Arm A (Frozen Baseline): Baseline Indexes, K_pool = 50 (Frozen Benchmark Reference)
- Arm B (Experiment L.1 Champion): Grounded Indexes, K_pool = 65, Standard Multi-Path Retrieval (Control)
- Arm C (Experiment L.2 Section-Aware): Grounded Indexes, K_pool = 65, Query-Dependent Intent Routing (Approach B)

Strict Experimental Controls:
- Same 100 questions from Stage 3 held-out validation suite.
- Same ground-truth evidence criteria and 1-indexed physical PDF page matching.
- Same query-time coreference configuration in all arms.
- Same BM25 and dense retrieval configuration, fusion parameters, and reranker.
- Zero modifications to baseline indexes or Experiment L grounded indexes.
- Zero rebuild of FAISS or BM25.
- Zero modification to embeddings, reranker, or evaluation logic.
- Output directory: outputs/experiments/experiment_l2_section_aware/
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
from outputs.experiments.experiment_l2_section_aware.section_aware_retriever import (
    SectionAwareMultiPathRetriever
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l2_section_aware")


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

    # Calculate P95 latency
    sorted_lats = sorted(lats)
    p95_idx = int(0.95 * len(sorted_lats))
    p95_lat = sorted_lats[p95_idx] if sorted_lats else 0.0

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
        "p95_latency_sec": round(p95_lat, 4),
        "peak_vram_mb": round(peak_vram, 2),
        "total_hits_top10": sum(1 for r in ranks if r <= 10),
        "total_pool_hits": sum(pool_hits),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("EXPERIMENT L.2: STAGE 3 CONTROLLED SECTION-AWARE BENCHMARK")
    print("Arm A (Base K=50) vs Arm B (Grounded L.1 K=65 Control) vs Arm C (Exp L.2 Approach B K=65)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 1. Verify Immutability of Frozen Baseline Corpus
    baseline_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    baseline_corpus_hash = compute_sha256(baseline_corpus_path)
    expected_baseline_hash = "ea4264f5d7e889d30fc4ee9147e7286b67ffa76300967fee3651002974062845"
    assert baseline_corpus_hash == expected_baseline_hash, "Baseline corpus hash mismatch!"
    print(f"Verified Baseline Corpus SHA256: {baseline_corpus_hash} (Immutable)")

    # 2. Verify Immutability of Frozen Grounded Corpus
    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    grounded_corpus_hash = compute_sha256(grounded_corpus_path)
    expected_grounded_hash = "ea3b2ae3f424b4db463748d00438ff3f6518b06bcbd052948d48366839912ca4"
    assert grounded_corpus_hash == expected_grounded_hash, "Grounded corpus hash mismatch!"
    print(f"Verified Grounded Corpus SHA256: {grounded_corpus_hash} (Immutable)")

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

    # 4. Load Arm B & C (Grounded Corpus & Indexes)
    print("\n[Loading Grounded Corpus & Indexes for Arms B & C]...")
    arm_b_chunks = {}
    with open(grounded_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            arm_b_chunks[c["chunk_id"]] = c

    arm_b_faiss_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "faiss_bge_m3")
    arm_b_bm25_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "bm25")
    arm_b_faiss = FAISS.load_local(arm_b_faiss_path, embeddings, allow_dangerous_deserialization=True)
    arm_b_bm25 = BM25Retriever(arm_b_bm25_path, load_corpus=True)

    # 5. Shared Query Expander and Reranker
    expander = FinancialQueryExpander(enable_coreference=True)
    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True
    )

    # Arm A Retriever (Standard MultiPathRetriever, baseline indexes)
    mpr_arm_a = MultiPathRetriever(
        faiss_index=arm_a_faiss, bm25_retriever=arm_a_bm25, chunk_store=arm_a_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=expander
    )

    # Arm B Retriever (Standard MultiPathRetriever control, grounded indexes)
    mpr_arm_b = SectionAwareMultiPathRetriever(
        faiss_index=arm_b_faiss, bm25_retriever=arm_b_bm25, chunk_store=arm_b_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=expander,
        narrative_table_cap=4, item1_boost=0.05
    )

    # 6. Load Benchmark Questions
    questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "heldout_questions.jsonl")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"\nLoaded {len(questions)} Stage 3 held-out evaluation questions.")

    # -------------------------------------------------------------------------
    # Execution: Arm A (Frozen Baseline, K=50)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 1: EVALUATING ARM A (FROZEN BASELINE, K=50)")
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

        cid_set = set(c["chunk_id"] for c in cands)
        assert len(cid_set) == len(cands), f"Duplicate candidates in Arm A for {q_data.get('question_id')}"

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_a.append(1 if pool_hit else 0)

        reranked = reranker.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_a.append(time.time() - t_q)

        rank_a = evaluate_hits(reranked, doc_name, target_pages)
        ranks_a.append(rank_a)

    peak_vram_a = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_a = compute_metrics(ranks_a, pool_hits_a, lats_a, peak_vram_a)
    print(f"  Arm A Metrics: MRR={metrics_a['mrr']}, R@10={metrics_a['recall_at_10']}%, Pool={metrics_a['pool_recall']}%, Lat={metrics_a['avg_latency_sec']}s (P95={metrics_a['p95_latency_sec']}s)")

    # -------------------------------------------------------------------------
    # Execution: Arm B (Experiment L.1 Champion Control, K=65)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 2: EVALUATING ARM B (EXPERIMENT L.1 CONTROL, K=65)")
    print("-" * 80)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_b = []
    pool_hits_b = []
    lats_b = []

    for idx, q_data in enumerate(questions, start=1):
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_arm_b.retrieve_multi_path(
            query=q_text, k_pool=65, enable_bm25=True, enable_dense=True, arm_mode="control"
        )
        cands = ret_pack["candidates"]

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_b.append(1 if pool_hit else 0)

        reranked = reranker.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_b.append(time.time() - t_q)

        rank_b = evaluate_hits(reranked, doc_name, target_pages)
        ranks_b.append(rank_b)

    peak_vram_b = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_b = compute_metrics(ranks_b, pool_hits_b, lats_b, peak_vram_b)
    print(f"  Arm B Metrics: MRR={metrics_b['mrr']}, R@10={metrics_b['recall_at_10']}%, Pool={metrics_b['pool_recall']}%, Lat={metrics_b['avg_latency_sec']}s (P95={metrics_b['p95_latency_sec']}s)")

    # -------------------------------------------------------------------------
    # Execution: Arm C (Experiment L.2 Section-Aware Approach B, K=65)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 3: EVALUATING ARM C (EXPERIMENT L.2 SECTION-AWARE APPROACH B, K=65)")
    print("-" * 80)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_c = []
    pool_hits_c = []
    lats_c = []
    intent_counts = defaultdict(int)
    comparison_rows = []

    for idx, q_data in enumerate(questions, start=1):
        qid = q_data.get("question_id", f"stage3_heldout_{idx:04d}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        qtype = q_data.get("question_type", "")
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_arm_b.retrieve_multi_path(
            query=q_text,
            k_pool=65,
            enable_bm25=True,
            enable_dense=True,
            question_type=qtype,
            arm_mode="section_aware"
        )
        cands = ret_pack["candidates"]
        intent = ret_pack.get("intent", "UNKNOWN")
        intent_counts[intent] += 1

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_c.append(1 if pool_hit else 0)

        reranked = reranker.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_c.append(time.time() - t_q)

        rank_c = evaluate_hits(reranked, doc_name, target_pages)
        ranks_c.append(rank_c)

        # Movement tracking relative to Arm B (L.1 Champion)
        rank_b_val = ranks_b[idx - 1]
        pool_b_val = pool_hits_b[idx - 1]

        if rank_b_val == rank_c:
            movement = "TIE"
        elif rank_b_val == 999 and rank_c <= 10:
            movement = f"NEW_HIT_#{rank_c}"
        elif rank_b_val <= 10 and rank_c == 999:
            movement = f"LOST_HIT_#{rank_b_val}"
        elif rank_c < rank_b_val:
            movement = f"+{rank_b_val - rank_c}"
        else:
            movement = f"-{rank_c - rank_b_val}"

        if pool_b_val == 0 and pool_hit:
            pool_shift = "RECOVERED_TO_POOL"
        elif pool_b_val == 1 and not pool_hit:
            pool_shift = "LOST_FROM_POOL"
        else:
            pool_shift = "SAME"

        comparison_rows.append({
            "index": idx,
            "id": qid,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "question_type": qtype,
            "classified_intent": intent,
            "target_pages": str(target_pages),
            "pool_arm_a": bool(pool_hits_a[idx - 1]),
            "pool_arm_b": bool(pool_b_val),
            "pool_arm_c": bool(pool_hit),
            "pool_shift": pool_shift,
            "rank_arm_a": rank_a if rank_a <= 10 else "NO_HIT",
            "rank_arm_b": rank_b_val if rank_b_val <= 10 else "NO_HIT",
            "rank_arm_c": rank_c if rank_c <= 10 else "NO_HIT",
            "movement_c_vs_b": movement,
            "question": q_text,
        })

    peak_vram_c = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_c = compute_metrics(ranks_c, pool_hits_c, lats_c, peak_vram_c)
    print(f"  Arm C Metrics: MRR={metrics_c['mrr']}, R@10={metrics_c['recall_at_10']}%, Pool={metrics_c['pool_recall']}%, Lat={metrics_c['avg_latency_sec']}s (P95={metrics_c['p95_latency_sec']}s)")

    # -------------------------------------------------------------------------
    # Write Artifacts
    # -------------------------------------------------------------------------
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if device == "cuda" else 0.0
        },
        "intent_distribution": dict(intent_counts),
        "arm_a_frozen_baseline_k50": metrics_a,
        "arm_b_grounded_exp_l1_k65": metrics_b,
        "arm_c_section_aware_exp_l2_k65": metrics_c,
        "delta_c_vs_b": {
            "pool_recall_delta": round(metrics_c["pool_recall"] - metrics_b["pool_recall"], 2),
            "recall_at_10_delta": round(metrics_c["recall_at_10"] - metrics_b["recall_at_10"], 2),
            "recall_at_5_delta": round(metrics_c["recall_at_5"] - metrics_b["recall_at_5"], 2),
            "mrr_delta": round(metrics_c["mrr"] - metrics_b["mrr"], 4),
            "latency_delta_sec": round(metrics_c["avg_latency_sec"] - metrics_b["avg_latency_sec"], 4),
        },
        "delta_c_vs_a": {
            "pool_recall_delta": round(metrics_c["pool_recall"] - metrics_a["pool_recall"], 2),
            "recall_at_10_delta": round(metrics_c["recall_at_10"] - metrics_a["recall_at_10"], 2),
            "recall_at_5_delta": round(metrics_c["recall_at_5"] - metrics_a["recall_at_5"], 2),
            "mrr_delta": round(metrics_c["mrr"] - metrics_a["mrr"], 4),
        }
    }

    scorecard_path = os.path.join(OUTPUT_DIR, "l2_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nSaved scorecard to: {scorecard_path}")

    csv_path = os.path.join(OUTPUT_DIR, "l2_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved question-level comparison CSV to: {csv_path}")

    print("\nBenchmark Execution Complete!")


if __name__ == "__main__":
    main()
