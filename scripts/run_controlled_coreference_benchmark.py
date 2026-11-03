"""
scripts/run_controlled_coreference_benchmark.py
================================================
Controlled Coreference ON vs OFF Benchmark for FinSage Phase 4.

Evaluates:
1. Dataset 1: FinanceBench (N = 150 questions)
2. Dataset 2: Stage 3 Held-Out (N = 100 questions)

Configurations:
- Config A: Frozen Phase 4 Baseline (Coreference OFF: enable_coreference=False)
- Config B: Current Implementation (Coreference ON: enable_coreference=True)

Strict Controls:
- Same dataset questions & gold evidence criteria
- Same indexed candidate chunks & embeddings
- Same retrieval parameters:
  - FinanceBench: K_pool=40, BM25 + Dense + Metadata (RRF k=60)
  - Stage 3: K_pool=50, BM25 + Dense (RRF k=60)
- Same reranker: BAAI/bge-reranker-base, FP16, max_length=512, top_k=10
- Measures actual pool recall, MRR, Recall@1, @5, @10, latency, and peak GPU VRAM
- Tracks coreference activations, brand entity mappings, and question-level rank shifts
"""

import os
import sys
import time
import json
import csv
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
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.financial_expander import FinancialQueryExpander
from src.retrieval.multi_path_retriever import MultiPathRetriever
from src.retrieval.phase2_reranker import Phase2Reranker


def evaluate_hits(reranked_chunks: List[Dict[str, Any]], gold_doc: str, target_pages: List[int]) -> int:
    """Finds best 1-indexed rank of a valid gold evidence chunk."""
    for rank, c in enumerate(reranked_chunks, start=1):
        c_doc = c.get("doc_name")
        c_page = c.get("page", 0)
        if c_doc == gold_doc and c_page in target_pages:
            return rank
    return 999


def compute_metrics(ranks: List[int], pool_hits: List[int], latencies: List[float], peak_vram_mb: float) -> Dict[str, Any]:
    n = len(ranks)
    rec1 = sum(1 for r in ranks if r == 1) / n * 100.0
    rec5 = sum(1 for r in ranks if r <= 5) / n * 100.0
    rec10 = sum(1 for r in ranks if r <= 10) / n * 100.0
    mrr = sum(1.0 / r if r <= 10 else 0.0 for r in ranks) / n
    pool_rec = sum(pool_hits) / n * 100.0
    avg_lat = sum(latencies) / n if latencies else 0.0

    return {
        "n_questions": n,
        "pool_recall": round(pool_rec, 2),
        "mrr": round(mrr, 4),
        "recall_at_1": round(rec1, 2),
        "recall_at_5": round(rec5, 2),
        "recall_at_10": round(rec10, 2),
        "avg_latency_sec": round(avg_lat, 4),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "total_hits_top10": sum(1 for r in ranks if r <= 10),
        "total_pool_hits": sum(pool_hits)
    }


def run_financebench_benchmark(
    questions: List[Dict[str, Any]],
    mpr_off: MultiPathRetriever,
    reranker_off: Phase2Reranker,
    mpr_on: MultiPathRetriever,
    reranker_on: Phase2Reranker,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    print("\n" + "=" * 80)
    print("RUNNING FINANCEBENCH BENCHMARK (N=150): CONFIG A (OFF) vs CONFIG B (ON)")
    print("=" * 80)

    # 1. Evaluate Config A (Coreference OFF)
    print("[1/2] Evaluating Config A: Coreference OFF...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    ranks_off = []
    pool_hits_off = []
    lats_off = []

    for idx, q_data in enumerate(questions, start=1):
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        evidence = q_data.get("evidence", [])
        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_off.retrieve_multi_path(
            query=q_text, k_pool=40, enable_bm25=True, enable_dense=True, enable_metadata=True, enable_hyde=False
        )
        cands = ret_pack["candidates"]
        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_off.append(1 if pool_hit else 0)

        reranked = reranker_off.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_off.append(time.time() - t_q)

        rank = evaluate_hits(reranked, doc_name, target_pages)
        ranks_off.append(rank)

    peak_vram_off = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_off = compute_metrics(ranks_off, pool_hits_off, lats_off, peak_vram_off)
    print(f"  Config A Completed: MRR={metrics_off['mrr']}, Recall@10={metrics_off['recall_at_10']}%, Pool={metrics_off['pool_recall']}%")

    # 2. Evaluate Config B (Coreference ON)
    print("[2/2] Evaluating Config B: Coreference ON...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_on = []
    pool_hits_on = []
    lats_on = []
    question_comparisons = []
    coref_activations = 0
    brand_mappings = []

    for idx, q_data in enumerate(questions, start=1):
        fid = q_data.get("financebench_id", f"FB_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        evidence = q_data.get("evidence", [])
        target_pages = [e.get("evidence_page_num") + 1 for e in evidence if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_on.retrieve_multi_path(
            query=q_text, k_pool=40, enable_bm25=True, enable_dense=True, enable_metadata=True, enable_hyde=False
        )
        cands = ret_pack["candidates"]
        exp_res = ret_pack["expansion_result"]

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_on.append(1 if pool_hit else 0)

        reranked = reranker_on.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_on.append(time.time() - t_q)

        rank_on = evaluate_hits(reranked, doc_name, target_pages)
        ranks_on.append(rank_on)

        rank_off = ranks_off[idx - 1]
        pool_off = pool_hits_off[idx - 1]

        # Coreference tracking
        is_coref = exp_res.is_coreference_resolved
        coref_term = exp_res.coreference_term
        if is_coref:
            coref_activations += 1
            brand_mappings.append({
                "id": fid,
                "term": coref_term,
                "detected_company": exp_res.detected_company,
                "detected_ticker": exp_res.detected_ticker,
                "type": exp_res.coreference_type
            })

        # Movement tracking
        move = "TIE"
        if rank_on < rank_off:
            move = f"+{rank_off - rank_on}" if rank_off != 999 else f"NEW_HIT_#{rank_on}"
        elif rank_on > rank_off:
            move = f"-{rank_on - rank_off}" if rank_on != 999 else f"LOST_HIT_#{rank_off}"

        question_comparisons.append({
            "index": idx,
            "id": fid,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "target_pages": str(target_pages),
            "pool_off": bool(pool_off),
            "pool_on": bool(pool_hit),
            "rank_off": rank_off if rank_off <= 10 else "NO_HIT",
            "rank_on": rank_on if rank_on <= 10 else "NO_HIT",
            "rr_off": round(1.0 / rank_off, 4) if rank_off <= 10 else 0.0,
            "rr_on": round(1.0 / rank_on, 4) if rank_on <= 10 else 0.0,
            "movement": move,
            "is_coref_active": is_coref,
            "coref_term": coref_term or "",
            "detected_company": exp_res.detected_company or "",
            "detected_ticker": exp_res.detected_ticker or "",
            "question": q_text
        })

    peak_vram_on = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_on = compute_metrics(ranks_on, pool_hits_on, lats_on, peak_vram_on)
    metrics_on["coref_activations"] = coref_activations
    metrics_on["brand_mappings"] = brand_mappings

    print(f"  Config B Completed: MRR={metrics_on['mrr']}, Recall@10={metrics_on['recall_at_10']}%, Pool={metrics_on['pool_recall']}%")
    return metrics_off, metrics_on, question_comparisons


def run_stage3_benchmark(
    questions: List[Dict[str, Any]],
    mpr_off: MultiPathRetriever,
    reranker_off: Phase2Reranker,
    mpr_on: MultiPathRetriever,
    reranker_on: Phase2Reranker,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    print("\n" + "=" * 80)
    print("RUNNING STAGE 3 HELDOUT BENCHMARK (N=100): CONFIG A (OFF) vs CONFIG B (ON)")
    print("=" * 80)

    # 1. Evaluate Config A (Coreference OFF)
    print("[1/2] Evaluating Stage 3 Config A: Coreference OFF...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_off = []
    pool_hits_off = []
    lats_off = []

    for idx, q_data in enumerate(questions, start=1):
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_off.retrieve_multi_path(
            query=q_text, k_pool=50, enable_bm25=True, enable_dense=True, enable_metadata=False, enable_hyde=False
        )
        cands = ret_pack["candidates"]
        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_off.append(1 if pool_hit else 0)

        reranked = reranker_off.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_off.append(time.time() - t_q)

        rank = evaluate_hits(reranked, doc_name, target_pages)
        ranks_off.append(rank)

    peak_vram_off = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_off = compute_metrics(ranks_off, pool_hits_off, lats_off, peak_vram_off)
    print(f"  Stage 3 Config A: MRR={metrics_off['mrr']}, Recall@10={metrics_off['recall_at_10']}%, Pool={metrics_off['pool_recall']}%")

    # 2. Evaluate Config B (Coreference ON)
    print("[2/2] Evaluating Stage 3 Config B: Coreference ON...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ranks_on = []
    pool_hits_on = []
    lats_on = []
    question_comparisons = []
    coref_activations = 0
    brand_mappings = []

    for idx, q_data in enumerate(questions, start=1):
        qid = q_data.get("question_id", f"s3_{idx}")
        q_text = q_data["question"]
        doc_name = q_data["doc_name"]
        company = q_data.get("company", "")
        sector = q_data.get("gics_sector", "General")
        target_pages = [e.get("evidence_page_num") for e in q_data.get("evidence_sources", []) if e.get("evidence_page_num") is not None]

        t_q = time.time()
        ret_pack = mpr_on.retrieve_multi_path(
            query=q_text, k_pool=50, enable_bm25=True, enable_dense=True, enable_metadata=False, enable_hyde=False
        )
        cands = ret_pack["candidates"]
        exp_res = ret_pack["expansion_result"]

        pool_hit = any(c.get("doc_name") == doc_name and c.get("page", 0) in target_pages for c in cands)
        pool_hits_on.append(1 if pool_hit else 0)

        reranked = reranker_on.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
        lats_on.append(time.time() - t_q)

        rank_on = evaluate_hits(reranked, doc_name, target_pages)
        ranks_on.append(rank_on)

        rank_off = ranks_off[idx - 1]
        pool_off = pool_hits_off[idx - 1]

        # Coreference tracking
        is_coref = exp_res.is_coreference_resolved
        coref_term = exp_res.coreference_term
        if is_coref:
            coref_activations += 1
            brand_mappings.append({
                "id": qid,
                "term": coref_term,
                "detected_company": exp_res.detected_company,
                "detected_ticker": exp_res.detected_ticker,
                "type": exp_res.coreference_type
            })

        # Movement tracking
        move = "TIE"
        if rank_on < rank_off:
            move = f"+{rank_off - rank_on}" if rank_off != 999 else f"NEW_HIT_#{rank_on}"
        elif rank_on > rank_off:
            move = f"-{rank_on - rank_off}" if rank_on != 999 else f"LOST_HIT_#{rank_off}"

        question_comparisons.append({
            "index": idx,
            "id": qid,
            "company": company,
            "doc_name": doc_name,
            "sector": sector,
            "target_pages": str(target_pages),
            "pool_off": bool(pool_off),
            "pool_on": bool(pool_hit),
            "rank_off": rank_off if rank_off <= 10 else "NO_HIT",
            "rank_on": rank_on if rank_on <= 10 else "NO_HIT",
            "rr_off": round(1.0 / rank_off, 4) if rank_off <= 10 else 0.0,
            "rr_on": round(1.0 / rank_on, 4) if rank_on <= 10 else 0.0,
            "movement": move,
            "is_coref_active": is_coref,
            "coref_term": coref_term or "",
            "detected_company": exp_res.detected_company or "",
            "detected_ticker": exp_res.detected_ticker or "",
            "question": q_text
        })

    peak_vram_on = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics_on = compute_metrics(ranks_on, pool_hits_on, lats_on, peak_vram_on)
    metrics_on["coref_activations"] = coref_activations
    metrics_on["brand_mappings"] = brand_mappings

    print(f"  Stage 3 Config B: MRR={metrics_on['mrr']}, Recall@10={metrics_on['recall_at_10']}%, Pool={metrics_on['pool_recall']}%")
    return metrics_off, metrics_on, question_comparisons


def main():
    out_dir = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_coreference_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Controlled Coreference Benchmark on Device: {device}")

    # Shared Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # -------------------------------------------------------------------------
    # PART 1: FinanceBench (N=150)
    # -------------------------------------------------------------------------
    fb_chunks_path = os.path.join(BASE_DIR, "data", "financebench", "processed", "phase2_enriched_chunks.jsonl")
    print(f"Loading FinanceBench chunks from {fb_chunks_path}...")
    fb_chunks = {}
    with open(fb_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            fb_chunks[c["chunk_id"]] = c

    fb_faiss = FAISS.load_local(os.path.join(BASE_DIR, "data", "financebench", "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    fb_meta = MetadataRetriever(os.path.join(BASE_DIR, "data", "financebench", "metadata_faiss_index"), embeddings=embeddings)
    fb_bm25 = BM25Retriever(os.path.join(BASE_DIR, "data", "financebench", "bm25_index_phase2"), load_corpus=True)

    # Instantiate Config A (OFF) and Config B (ON) for FinanceBench
    exp_fb_off = FinancialQueryExpander(enable_coreference=False)
    mpr_fb_off = MultiPathRetriever(
        faiss_index=fb_faiss, bm25_retriever=fb_bm25, chunk_store=fb_chunks,
        metadata_retriever=fb_meta, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=exp_fb_off
    )
    reranker_fb_off = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=exp_fb_off, enable_coreference=False)

    exp_fb_on = FinancialQueryExpander(enable_coreference=True)
    mpr_fb_on = MultiPathRetriever(
        faiss_index=fb_faiss, bm25_retriever=fb_bm25, chunk_store=fb_chunks,
        metadata_retriever=fb_meta, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=exp_fb_on
    )
    reranker_fb_on = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=exp_fb_on, enable_coreference=True)

    with open(os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl"), "r", encoding="utf-8") as f:
        fb_questions = [json.loads(line) for line in f]

    fb_off_metrics, fb_on_metrics, fb_comparisons = run_financebench_benchmark(
        fb_questions, mpr_fb_off, reranker_fb_off, mpr_fb_on, reranker_fb_on
    )

    # Save FinanceBench CSV
    fb_csv_path = os.path.join(out_dir, "financebench_coref_comparison.csv")
    with open(fb_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fb_comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(fb_comparisons)
    print(f"Saved FinanceBench comparison to: {fb_csv_path}")

    # -------------------------------------------------------------------------
    # PART 2: Stage 3 Held-Out (N=100)
    # -------------------------------------------------------------------------
    s3_chunks_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    print(f"\nLoading Stage 3 chunks from {s3_chunks_path}...")
    s3_chunks = {}
    with open(s3_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            s3_chunks[c["chunk_id"]] = c

    s3_faiss = FAISS.load_local(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3"), embeddings, allow_dangerous_deserialization=True)
    s3_bm25 = BM25Retriever(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25"), load_corpus=True)

    # Instantiate Config A (OFF) and Config B (ON) for Stage 3
    exp_s3_off = FinancialQueryExpander(enable_coreference=False)
    mpr_s3_off = MultiPathRetriever(
        faiss_index=s3_faiss, bm25_retriever=s3_bm25, chunk_store=s3_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=exp_s3_off
    )
    reranker_s3_off = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=exp_s3_off, enable_coreference=False)

    exp_s3_on = FinancialQueryExpander(enable_coreference=True)
    mpr_s3_on = MultiPathRetriever(
        faiss_index=s3_faiss, bm25_retriever=s3_bm25, chunk_store=s3_chunks,
        metadata_retriever=None, hyde_retriever=None, chunk_bundler=None, rrf_k=60, expander=exp_s3_on
    )
    reranker_s3_on = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=exp_s3_on, enable_coreference=True)

    with open(os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "heldout_questions.jsonl"), "r", encoding="utf-8") as f:
        s3_questions = [json.loads(line) for line in f]

    s3_off_metrics, s3_on_metrics, s3_comparisons = run_stage3_benchmark(
        s3_questions, mpr_s3_off, reranker_s3_off, mpr_s3_on, reranker_s3_on
    )

    # Save Stage 3 CSV
    s3_csv_path = os.path.join(out_dir, "stage3_coref_comparison.csv")
    with open(s3_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(s3_comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(s3_comparisons)
    print(f"Saved Stage 3 comparison to: {s3_csv_path}")

    # -------------------------------------------------------------------------
    # PART 3: Save Summary Scorecard JSON
    # -------------------------------------------------------------------------
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
        },
        "financebench_n150": {
            "config_a_off": fb_off_metrics,
            "config_b_on": fb_on_metrics,
            "delta": {
                "mrr": round(fb_on_metrics["mrr"] - fb_off_metrics["mrr"], 4),
                "recall_at_1": round(fb_on_metrics["recall_at_1"] - fb_off_metrics["recall_at_1"], 2),
                "recall_at_5": round(fb_on_metrics["recall_at_5"] - fb_off_metrics["recall_at_5"], 2),
                "recall_at_10": round(fb_on_metrics["recall_at_10"] - fb_off_metrics["recall_at_10"], 2),
                "pool_recall": round(fb_on_metrics["pool_recall"] - fb_off_metrics["pool_recall"], 2),
                "latency_sec": round(fb_on_metrics["avg_latency_sec"] - fb_off_metrics["avg_latency_sec"], 4),
            }
        },
        "stage3_heldout_n100": {
            "config_a_off": s3_off_metrics,
            "config_b_on": s3_on_metrics,
            "delta": {
                "mrr": round(s3_on_metrics["mrr"] - s3_off_metrics["mrr"], 4),
                "recall_at_1": round(s3_on_metrics["recall_at_1"] - s3_off_metrics["recall_at_1"], 2),
                "recall_at_5": round(s3_on_metrics["recall_at_5"] - s3_off_metrics["recall_at_5"], 2),
                "recall_at_10": round(s3_on_metrics["recall_at_10"] - s3_off_metrics["recall_at_10"], 2),
                "pool_recall": round(s3_on_metrics["pool_recall"] - s3_off_metrics["pool_recall"], 2),
                "latency_sec": round(s3_on_metrics["avg_latency_sec"] - s3_off_metrics["avg_latency_sec"], 4),
            }
        }
    }

    scorecard_path = os.path.join(out_dir, "coreference_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nSaved Scorecard JSON to: {scorecard_path}")

    # Movement breakdown
    fb_improved = [r for r in fb_comparisons if r["movement"].startswith("+") or "NEW_HIT" in r["movement"]]
    fb_regressed = [r for r in fb_comparisons if r["movement"].startswith("-") or "LOST_HIT" in r["movement"]]
    s3_improved = [r for r in s3_comparisons if r["movement"].startswith("+") or "NEW_HIT" in r["movement"]]
    s3_regressed = [r for r in s3_comparisons if r["movement"].startswith("-") or "LOST_HIT" in r["movement"]]

    print("\n" + "=" * 80)
    print("CONTROLLED COREFERENCE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"FinanceBench (N=150):")
    print(f"  Config A (OFF): MRR={fb_off_metrics['mrr']}, R@10={fb_off_metrics['recall_at_10']}%, Pool={fb_off_metrics['pool_recall']}%")
    print(f"  Config B (ON) : MRR={fb_on_metrics['mrr']}, R@10={fb_on_metrics['recall_at_10']}%, Pool={fb_on_metrics['pool_recall']}%")
    print(f"  Movements     : +{len(fb_improved)} Improved, -{len(fb_regressed)} Regressed, {150 - len(fb_improved) - len(fb_regressed)} Unchanged")
    print(f"  Activations   : {fb_on_metrics['coref_activations']} coref queries")

    print(f"\nStage 3 Held-Out (N=100):")
    print(f"  Config A (OFF): MRR={s3_off_metrics['mrr']}, R@10={s3_off_metrics['recall_at_10']}%, Pool={s3_off_metrics['pool_recall']}%")
    print(f"  Config B (ON) : MRR={s3_on_metrics['mrr']}, R@10={s3_on_metrics['recall_at_10']}%, Pool={s3_on_metrics['pool_recall']}%")
    print(f"  Movements     : +{len(s3_improved)} Improved, -{len(s3_regressed)} Regressed, {100 - len(s3_improved) - len(s3_regressed)} Unchanged")
    print(f"  Activations   : {s3_on_metrics['coref_activations']} coref queries")
    print("=" * 80)


if __name__ == "__main__":
    main()
