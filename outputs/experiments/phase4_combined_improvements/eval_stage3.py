"""
FinSage Phase 4: Benchmark Evaluation B (Unseen Stage 3 Held-Out N=100)
=======================================================================
Strict Isolated Evaluation on:
- 100 unseen held-out questions from 20 corporate entities across 40 unseen 10-K filings.
- Isolated corpus: data/heldout_stage3/processed/heldout_enriched_chunks.jsonl (29,335 chunks).
- Isolated indexes: data/heldout_stage3/indexes/faiss_bge_m3 and bm25.
- ZERO access to development corpus or indexes.
- NO tuning using Stage 3 gold labels.

Compares head-to-head against:
- Frozen Experiment F / Candidate H2.1 baseline: MRR 0.1050, Recall@10 20.00%, Pool Recall 30.00%.
"""

import os
import sys
import time
import json
import csv
import torch
from collections import defaultdict
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.financial_expander import FinancialQueryExpander, normalize_unicode
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.phase2_reranker import Phase2Reranker


def evaluate_hits(reranked_chunks: List[Dict[str, Any]], gold_doc: str, target_pages: List[int]) -> int:
    """Finds the best 1-indexed rank of a valid gold evidence chunk."""
    for rank, c in enumerate(reranked_chunks, start=1):
        c_doc = c.get("doc_name")
        c_page = c.get("page", 0)
        if c_doc == gold_doc and c_page in target_pages:
            return rank
    return 999


def main():
    print("=" * 80)
    print("FINSAGE PHASE 4: BENCHMARK EVALUATION B — STAGE 3 HELDOUT (N=100)")
    print("=" * 80)

    out_dir = os.path.join(BASE_DIR, "outputs", "experiments", "phase4_combined_improvements", "stage3")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Isolated Stage 3 Chunks (29,335 chunks)
    stage3_chunks_file = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    print(f"Loading isolated Stage 3 chunks from {stage3_chunks_file}...")
    chunk_store = {}
    with open(stage3_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
    print(f"Loaded {len(chunk_store)} isolated Stage 3 chunks.")

    # 2. Load Isolated Stage 3 FAISS Dense Index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    stage3_faiss_dir = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3")
    dense_faiss = FAISS.load_local(stage3_faiss_dir, embeddings, allow_dangerous_deserialization=True)

    # 3. Load Isolated Stage 3 BM25 Index
    stage3_bm25_dir = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25")
    bm25_retriever = BM25Retriever(stage3_bm25_dir, load_corpus=True)

    # 4. Multi-Path Retriever with Phase 4 Intelligence
    expander = FinancialQueryExpander()
    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        metadata_retriever=None,  # Stage 3 uses 2-path (Dense FAISS + BM25)
        hyde_retriever=None,
        chunk_bundler=None,
        rrf_k=60,
        expander=expander,
    )

    # 5. Phase 4 Production Reranker
    reranker = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=True,
        expander=expander,
    )

    # 6. Load Historical Stage 3 Baseline Data (Exp F and H2.1)
    stage3_baseline_checkpoint = os.path.join(
        BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "phase2_retrieval_checkpoint_reference_exp_f.jsonl"
    )
    hist_ranks_base = {}
    if os.path.exists(stage3_baseline_checkpoint):
        with open(stage3_baseline_checkpoint, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("question_id") or rec.get("qid") or rec.get("financebench_id")
                hist_ranks_base[qid] = rec.get("best_rank", 999)

    # 7. Load Stage 3 Held-Out Questions (N=100)
    questions_file = os.path.join(
        BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "heldout_questions.jsonl"
    )
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Evaluating {len(questions)} Stage 3 held-out questions with Phase 4 retrieval improvements...\n")

    results_records = []
    ranks_p4 = []
    pool_hits_40 = []
    pool_hits_50 = []
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_bench_start = time.time()
    for idx, q_data in enumerate(questions, start=1):
        q_id = q_data.get("question_id") or q_data.get("qid", f"S3_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company")
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence_sources", [])
        target_pages = [e.get("evidence_page_num") for e in evidence if e.get("evidence_page_num") is not None]

        t_q0 = time.time()

        # Step 1: Execute Phase 4 Multi-Path Retrieval (K_pool=50 for direct comparison with Stage 3 baseline)
        retrieval_pack = mpr.retrieve_multi_path(
            query=q_text,
            k_pool=50,
            enable_bm25=True,
            enable_dense=True,
            enable_metadata=False,
            enable_hyde=False
        )
        candidates = retrieval_pack["candidates"]

        # Step 2: Check Candidate Pool Recall@40 and Recall@50
        pool_hit_40 = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in candidates[:40])
        pool_hit_50 = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in candidates)
        pool_hits_40.append(1 if pool_hit_40 else 0)
        pool_hits_50.append(1 if pool_hit_50 else 0)

        # Step 3: Phase 4 Statement-Aware & Multi-Statement Balanced Reranking (top_k=10)
        reranked = reranker.rerank(
            query=q_text,
            candidates=candidates,
            top_k=10,
            apply_boosts=True
        )

        t_query = time.time() - t_q0
        latencies.append(t_query)

        rank_p4 = evaluate_hits(reranked, doc_name, target_pages)
        ranks_p4.append(rank_p4)

        rr_p4 = 1.0 / rank_p4 if rank_p4 <= 10 else 0.0

        # Movement tracking vs Historical Exp F Baseline
        prev_base_rank = hist_ranks_base.get(q_id, 999)
        move_vs_base = "TIE"
        if rank_p4 < prev_base_rank:
            move_vs_base = f"+{prev_base_rank - rank_p4}" if prev_base_rank != 999 else f"NEW_HIT_#{rank_p4}"
        elif rank_p4 > prev_base_rank:
            move_vs_base = f"-{rank_p4 - prev_base_rank}" if rank_p4 != 999 else f"LOST_HIT_#{prev_base_rank}"

        results_records.append({
            "index": idx,
            "qid": q_id,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "target_pages": ";".join(str(p) for p in target_pages),
            "pool_hit_40": pool_hit_40,
            "pool_hit_50": pool_hit_50,
            "rank_phase4": rank_p4 if rank_p4 <= 10 else "NO_HIT",
            "rr_phase4": round(rr_p4, 4),
            "prev_exp_f_rank": prev_base_rank if prev_base_rank <= 10 else "NO_HIT",
            "movement_vs_exp_f": move_vs_base,
            "latency_seconds": round(t_query, 4),
            "reranked_top3": [f"{c['chunk_id']}_p{c['page']}" for c in reranked[:3]],
            "question": q_text[:80]
        })

        if idx % 25 == 0 or idx == len(questions):
            curr_mrr = sum(1.0 / r if r <= 10 else 0.0 for r in ranks_p4) / len(ranks_p4)
            curr_r10 = sum(1 for r in ranks_p4 if r <= 10) / len(ranks_p4) * 100
            curr_pool40 = sum(pool_hits_40) / len(pool_hits_40) * 100
            print(f"  Processed [{idx:3d}/{len(questions)}] | Pool@40: {curr_pool40:.1f}% | Top-10 MRR: {curr_mrr:.4f} (Recall@10: {curr_r10:.1f}%)")

    total_bench_time = round(time.time() - t_bench_start, 2)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    n_q = len(ranks_p4)
    mrr_p4 = sum(1.0 / r if r <= 10 else 0.0 for r in ranks_p4) / n_q
    r1_p4 = sum(1 for r in ranks_p4 if r == 1) / n_q * 100
    r5_p4 = sum(1 for r in ranks_p4 if r <= 5) / n_q * 100
    r10_p4 = sum(1 for r in ranks_p4 if r <= 10) / n_q * 100
    pool_rec_40 = sum(pool_hits_40) / n_q * 100
    pool_rec_50 = sum(pool_hits_50) / n_q * 100
    avg_latency = sum(latencies) / n_q

    # Movement Analysis Counts
    improved_vs_base = [r for r in results_records if r["movement_vs_exp_f"].startswith("+") or r["movement_vs_exp_f"].startswith("NEW_HIT")]
    regressed_vs_base = [r for r in results_records if r["movement_vs_exp_f"].startswith("-") or r["movement_vs_exp_f"].startswith("LOST_HIT")]
    unchanged_vs_base = [r for r in results_records if r["movement_vs_exp_f"] == "TIE"]

    scorecard = {
        "benchmark": "Stage 3 Independent Held-Out Evaluation (N=100)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_phase4": {
            "candidate_pool_recall_40": round(pool_rec_40, 2),
            "candidate_pool_recall_50": round(pool_rec_50, 2),
            "mrr": round(mrr_p4, 4),
            "recall_at_1": round(r1_p4, 2),
            "recall_at_5": round(r5_p4, 2),
            "recall_at_10": round(r10_p4, 2),
            "hits_at_1": sum(1 for r in ranks_p4 if r == 1),
            "hits_at_5": sum(1 for r in ranks_p4 if r <= 5),
            "hits_at_10": sum(1 for r in ranks_p4 if r <= 10),
            "pool_hits_at_40": sum(pool_hits_40),
            "pool_hits_at_50": sum(pool_hits_50),
            "total_questions": n_q,
            "avg_latency_seconds": round(avg_latency, 4),
            "total_time_seconds": total_bench_time,
            "peak_vram_mb": round(peak_vram_mb, 2)
        },
        "comparison_vs_exp_f_baseline": {
            "mrr_baseline": 0.1050,
            "mrr_delta": round(mrr_p4 - 0.1050, 4),
            "recall_10_baseline": 20.00,
            "recall_10_delta": round(r10_p4 - 20.00, 2),
            "candidate_pool_baseline": 30.00,
            "candidate_pool_delta": round(pool_rec_50 - 30.00, 2),
            "improved_count": len(improved_vs_base),
            "regressed_count": len(regressed_vs_base),
            "unchanged_count": len(unchanged_vs_base),
        }
    }

    # Save Scorecard JSON
    scorecard_path = os.path.join(out_dir, "stage3_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    # Save CSV Question Details
    csv_path = os.path.join(out_dir, "stage3_question_results.csv")
    fieldnames = [
        "index", "qid", "company", "doc_name", "sector", "target_pages",
        "pool_hit_40", "pool_hit_50", "rank_phase4", "rr_phase4", "prev_exp_f_rank",
        "movement_vs_exp_f", "latency_seconds", "question"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_records)

    print("\n" + "=" * 80)
    print("FINSAGE PHASE 4 — STAGE 3 HELDOUT BENCHMARK RESULTS SUMMARY (N=100)")
    print("=" * 80)
    print(f"Candidate Pool Recall@40: {pool_rec_40:.2f}% ({sum(pool_hits_40)}/{n_q})")
    print(f"Candidate Pool Recall@50: {pool_rec_50:.2f}% ({sum(pool_hits_50)}/{n_q})")
    print(f"Top-10 MRR:               {mrr_p4:.4f}")
    print(f"Recall@1:                 {r1_p4:.2f}% ({sum(1 for r in ranks_p4 if r == 1)}/{n_q})")
    print(f"Recall@5:                 {r5_p4:.2f}% ({sum(1 for r in ranks_p4 if r <= 5)}/{n_q})")
    print(f"Recall@10:                {r10_p4:.2f}% ({sum(1 for r in ranks_p4 if r <= 10)}/{n_q})")
    print(f"Average Latency / Query:  {avg_latency:.4f} s ({avg_latency*1000:.1f} ms)")
    print(f"Peak GPU VRAM:            {peak_vram_mb:.1f} MB")
    print(f"Total Benchmark Time:     {total_bench_time:.2f} s")
    print("-" * 80)
    print(f"Versus Stage 3 Exp F Baseline (MRR 0.1050, R@10 20.00%, Pool 30.00%):")
    print(f"  MRR Delta:         {mrr_p4 - 0.1050:+.4f}")
    print(f"  Recall@10 Delta:   {r10_p4 - 20.00:+.2f} pp")
    print(f"  Pool Recall Delta: {pool_rec_50 - 30.00:+.2f} pp")
    print(f"  Questions:         +{len(improved_vs_base)} improved, -{len(regressed_vs_base)} regressed, {len(unchanged_vs_base)} unchanged")
    print("=" * 80)


if __name__ == "__main__":
    main()
