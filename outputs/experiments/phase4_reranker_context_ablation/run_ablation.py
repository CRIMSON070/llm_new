"""
Phase 4 - Step 3: Controlled Reranker Ablation Benchmark
========================================================
Compares:
Config A (Baseline): BAAI/bge-reranker-base with character slicing c["text"][:512]
Config B (Updated):  BAAI/bge-reranker-base with full text and max_length=512

Evaluates all N=150 FinanceBench questions with identical candidate pools.
"""

import os
import sys
import time
import json
import csv
import torch
from collections import defaultdict
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.phase2_reranker import Phase2Reranker


def rerank_baseline_slice512(reranker: Phase2Reranker, query: str, candidates: List[Dict[str, Any]], top_k: int = 10, apply_boosts: bool = True):
    """Exact pre-fix Phase 2 baseline scoring with c['text'][:512]."""
    if not candidates:
        return []

    q_target_stmt = reranker.detect_query_statement_intent(query)
    q_company = reranker.detect_query_company(query)
    q_periods = reranker.detect_query_periods(query)

    # Pre-fix baseline: string character slice [:512]
    pairs = [[query, c["text"][:512]] for c in candidates]
    raw_scores = reranker.reranker.compute_score(pairs, batch_size=8)
    if not isinstance(raw_scores, list):
        raw_scores = [raw_scores]

    scored_candidates = []
    for c, raw_score in zip(candidates, raw_scores):
        chunk_text = c.get("text", "")
        chunk_stmt = reranker.detect_chunk_statement_type(c)
        has_table = reranker.has_table_structure(chunk_text)
        is_distraction = reranker.is_narrative_distraction(chunk_text)
        c_company, c_period = reranker.extract_chunk_company_and_period(c)

        stmt_boost = 0.0
        table_boost = 0.0
        company_boost = 0.0
        period_boost = 0.0
        penalty = 0.0

        if apply_boosts:
            if q_company and c_company and (q_company.upper() in c_company.upper() or c_company.upper() in q_company.upper()):
                company_boost = reranker.company_boost_weight
            if q_periods and c_period and c_period in q_periods:
                period_boost = reranker.period_boost_weight
            if q_target_stmt and chunk_stmt == q_target_stmt:
                stmt_boost = reranker.statement_boost_weight
            if has_table and q_target_stmt in ["cash_flow_statement", "balance_sheet", "income_statement"]:
                table_boost = reranker.table_boost_weight
            if q_target_stmt and (is_distraction or (not has_table and chunk_stmt != q_target_stmt)):
                penalty = reranker.negative_disclosure_penalty

        final_score = float(raw_score) + company_boost + period_boost + stmt_boost + table_boost + penalty

        item = dict(c)
        item["raw_rerank_score"] = round(float(raw_score), 4)
        item["final_score"] = round(final_score, 4)
        scored_candidates.append(item)

    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_candidates[:top_k]


def evaluate_hits(reranked: List[Dict[str, Any]], doc_name: str, target_pages: List[int]) -> int:
    for rank, c in enumerate(reranked, start=1):
        if c.get("doc_name") == doc_name and c.get("page", 0) in target_pages:
            return rank
    return 999


def main():
    print("=" * 80)
    print("PHASE 4 - STEP 3: CONTROLLED RERANKER ABLATION BENCHMARK (N=150)")
    print("=" * 80)

    out_dir = os.path.join(BASE_DIR, "outputs", "experiments", "phase4_reranker_context_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Chunks
    enriched_chunks_file = os.path.join(BASE_DIR, "data", "financebench", "processed", "phase2_enriched_chunks.jsonl")
    print(f"Loading chunks from {enriched_chunks_file}...")
    chunk_store = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store)} chunks.")

    # 2. Load FAISS & Metadata
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    faiss_dir = os.path.join(BASE_DIR, "data", "financebench", "faiss_index")
    meta_dir = os.path.join(BASE_DIR, "data", "financebench", "metadata_faiss_index")
    dense_faiss = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    metadata_retriever = MetadataRetriever(meta_dir, embeddings=embeddings)

    # 3. Load BM25
    bm25_dir = os.path.join(BASE_DIR, "data", "financebench", "bm25_index_phase2")
    bm25_retriever = BM25Retriever(bm25_dir, load_corpus=True)

    # 4. Multi-Path Retriever
    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        metadata_retriever=metadata_retriever,
        hyde_retriever=None,
        chunk_bundler=None,
        rrf_k=60
    )

    # 5. Initialize Production Phase2Reranker (max_length=512 configured)
    print("Initializing Phase 2 Reranker (BAAI/bge-reranker-base, FP16, max_length=512)...")
    shared_reranker = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True)

    # 6. Load Benchmark Questions
    questions_file = os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Evaluating {len(questions)} FinanceBench questions across identical candidate pools...\n")

    results_comparison = []
    baseline_ranks = []
    updated_ranks = []
    pool_hits = []

    baseline_latencies = []
    updated_latencies = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_bench_start = time.time()
    for idx, q_data in enumerate(questions, start=1):
        f_id = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company")
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence", [])
        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        # Candidate generation: identical for both arms (k=30, k_pool=30)
        sparse_res = mpr.retrieve_sparse(q_text, k=30)
        dense_res = mpr.retrieve_dense(q_text, k=30)
        meta_res, _ = mpr.retrieve_metadata(q_text, k_sections=3)
        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res,
            "path3_metadata": meta_res
        }
        candidates = mpr.reciprocal_rank_fusion(path_dict, k_pool=30)

        # Check candidate pool hit
        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in candidates)
        pool_hits.append(1 if pool_hit else 0)

        # Config A (Baseline Reranker: [:512] char slice)
        t_a0 = time.time()
        reranked_a = rerank_baseline_slice512(shared_reranker, query=q_text, candidates=candidates, top_k=10, apply_boosts=True)
        t_a = time.time() - t_a0
        baseline_latencies.append(t_a)
        rank_a = evaluate_hits(reranked_a, doc_name, target_pages)
        baseline_ranks.append(rank_a)

        # Config B (Updated Reranker: full chunk text to 512 tokens)
        t_b0 = time.time()
        reranked_b = shared_reranker.rerank(query=q_text, candidates=candidates, top_k=10, apply_boosts=True)
        t_b = time.time() - t_b0
        updated_latencies.append(t_b)
        rank_b = evaluate_hits(reranked_b, doc_name, target_pages)
        updated_ranks.append(rank_b)

        rr_a = 1.0 / rank_a if rank_a <= 10 else 0.0
        rr_b = 1.0 / rank_b if rank_b <= 10 else 0.0

        movement = "TIE"
        if rank_b < rank_a:
            movement = f"+{rank_a - rank_b}" if rank_a != 999 else f"NEW_HIT_#{rank_b}"
        elif rank_b > rank_a:
            movement = f"-{rank_b - rank_a}" if rank_b != 999 else f"LOST_HIT_#{rank_a}"

        results_comparison.append({
            "index": idx,
            "financebench_id": f_id,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "target_pages": ";".join(str(p) for p in target_pages),
            "pool_hit": pool_hit,
            "baseline_rank": rank_a if rank_a <= 10 else "NO_HIT",
            "baseline_rr": round(rr_a, 4),
            "updated_rank": rank_b if rank_b <= 10 else "NO_HIT",
            "updated_rr": round(rr_b, 4),
            "movement": movement,
            "baseline_top3": [f"{c['chunk_id']}_p{c['page']}" for c in reranked_a[:3]],
            "updated_top3": [f"{c['chunk_id']}_p{c['page']}" for c in reranked_b[:3]],
            "question": q_text[:80]
        })

        if idx % 25 == 0 or idx == len(questions):
            mrr_a = sum(1.0 / r if r <= 10 else 0.0 for r in baseline_ranks) / len(baseline_ranks)
            mrr_b = sum(1.0 / r if r <= 10 else 0.0 for r in updated_ranks) / len(updated_ranks)
            r10_a = sum(1 for r in baseline_ranks if r <= 10) / len(baseline_ranks) * 100
            r10_b = sum(1 for r in updated_ranks if r <= 10) / len(updated_ranks) * 100
            print(f"  Processed [{idx:3d}/{len(questions)}] | Base MRR: {mrr_a:.4f} (R@10: {r10_a:.1f}%) | Upd MRR: {mrr_b:.4f} (R@10: {r10_b:.1f}%)")

    total_bench_time = round(time.time() - t_bench_start, 2)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Aggregate Metrics Computation
    def calc_metrics(ranks):
        n = len(ranks)
        mrr = sum(1.0 / r if r <= 10 else 0.0 for r in ranks) / n
        r1 = sum(1 for r in ranks if r == 1) / n * 100
        r3 = sum(1 for r in ranks if r <= 3) / n * 100
        r5 = sum(1 for r in ranks if r <= 5) / n * 100
        r10 = sum(1 for r in ranks if r <= 10) / n * 100
        return {
            "mrr": round(mrr, 4),
            "recall_at_1": round(r1, 2),
            "recall_at_3": round(r3, 2),
            "recall_at_5": round(r5, 2),
            "recall_at_10": round(r10, 2),
            "hits_at_1": sum(1 for r in ranks if r == 1),
            "hits_at_3": sum(1 for r in ranks if r <= 3),
            "hits_at_5": sum(1 for r in ranks if r <= 5),
            "hits_at_10": sum(1 for r in ranks if r <= 10),
        }

    metrics_a = calc_metrics(baseline_ranks)
    metrics_b = calc_metrics(updated_ranks)
    metrics_a["avg_rerank_latency_s"] = round(sum(baseline_latencies) / len(baseline_latencies), 4)
    metrics_b["avg_rerank_latency_s"] = round(sum(updated_latencies) / len(updated_latencies), 4)
    metrics_a["candidate_pool_recall"] = round(sum(pool_hits) / len(pool_hits) * 100, 2)
    metrics_b["candidate_pool_recall"] = round(sum(pool_hits) / len(pool_hits) * 100, 2)

    improved = [r for r in results_comparison if (r["updated_rr"] > r["baseline_rr"])]
    regressed = [r for r in results_comparison if (r["updated_rr"] < r["baseline_rr"])]
    unchanged = [r for r in results_comparison if (r["updated_rr"] == r["baseline_rr"])]

    summary = {
        "benchmark": "FinanceBench N=150",
        "total_runtime_seconds": total_bench_time,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "config_a_baseline": metrics_a,
        "config_b_updated": metrics_b,
        "delta": {
            "mrr": round(metrics_b["mrr"] - metrics_a["mrr"], 4),
            "recall_at_1": round(metrics_b["recall_at_1"] - metrics_a["recall_at_1"], 2),
            "recall_at_3": round(metrics_b["recall_at_3"] - metrics_a["recall_at_3"], 2),
            "recall_at_5": round(metrics_b["recall_at_5"] - metrics_a["recall_at_5"], 2),
            "recall_at_10": round(metrics_b["recall_at_10"] - metrics_a["recall_at_10"], 2),
        },
        "movements": {
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "unchanged_count": len(unchanged)
        }
    }

    # Save Checkpoints
    with open(os.path.join(out_dir, "baseline_checkpoint.jsonl"), "w", encoding="utf-8") as f:
        for r in results_comparison:
            f.write(json.dumps({
                "index": r["index"],
                "financebench_id": r["financebench_id"],
                "best_rank": r["baseline_rank"],
                "rr": r["baseline_rr"],
                "top3": r["baseline_top3"]
            }) + "\n")

    with open(os.path.join(out_dir, "updated_checkpoint.jsonl"), "w", encoding="utf-8") as f:
        for r in results_comparison:
            f.write(json.dumps({
                "index": r["index"],
                "financebench_id": r["financebench_id"],
                "best_rank": r["updated_rank"],
                "rr": r["updated_rr"],
                "top3": r["updated_top3"]
            }) + "\n")

    # Save Comparison CSV
    with open(os.path.join(out_dir, "question_level_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "financebench_id", "company", "doc_name", "sector", "target_pages",
            "pool_hit", "baseline_rank", "baseline_rr", "updated_rank", "updated_rr",
            "movement", "baseline_top3", "updated_top3", "question"
        ])
        writer.writeheader()
        writer.writerows(results_comparison)

    # Save Metrics JSON
    with open(os.path.join(out_dir, "ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ABLATION BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Baseline MRR: {metrics_a['mrr']:.4f}  -->  Updated MRR: {metrics_b['mrr']:.4f} (Delta: {summary['delta']['mrr']:+.4f})")
    print(f"Recall@1:     {metrics_a['recall_at_1']}%  -->  {metrics_b['recall_at_1']}% (Delta: {summary['delta']['recall_at_1']:+.2f} pp)")
    print(f"Recall@5:     {metrics_a['recall_at_5']}%  -->  {metrics_b['recall_at_5']}% (Delta: {summary['delta']['recall_at_5']:+.2f} pp)")
    print(f"Recall@10:    {metrics_a['recall_at_10']}%  -->  {metrics_b['recall_at_10']}% (Delta: {summary['delta']['recall_at_10']:+.2f} pp)")
    print(f"Movements:    {len(improved)} Improved, {len(regressed)} Regressed, {len(unchanged)} Unchanged")
    print(f"Peak VRAM:    {summary['peak_vram_mb']} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
