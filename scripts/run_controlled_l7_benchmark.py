"""
Experiment L.7: Controlled Context Window Stitching Runner
==========================================================

Controlled A/B Evaluation:
- Arm A: Frozen L.2 Configuration without Stitching (Section-Aware Grounded K65)
- Arm B: Identical L.2 Configuration WITH Context Window Stitching Enabled

Evaluated against Benchmark V2:
- Primary Cohort: 65 Confidently Verified Questions
- Secondary Cohort: 35 Ambiguous Questions (isolated)
- Strict Direct Evidence & Comprehensive Evidence standards

Key Invariant Checks:
1. Candidate pools in Arm A and Arm B must be 100% identical in chunk_ids, scores, and ordering.
2. Candidate IDs are strictly preserved as evaluation units.
3. No cross-filing stitching.
"""

import os
import sys
import time
import json
import csv
import hashlib
import torch
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
from src.retrieval.phase2_reranker import Phase2Reranker
from outputs.experiments.experiment_l2_section_aware.section_aware_retriever import (
    SectionAwareMultiPathRetriever
)
from outputs.experiments.experiment_l7_context_window_stitching.context_window_stitcher import (
    ContextWindowStitcher
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l7_context_window_stitching")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_candidate_hits(
    ranked_chunks: List[Dict[str, Any]],
    gold_doc: str,
    valid_pages: List[int],
    valid_cids: List[str]
) -> Tuple[bool, int]:
    """
    Checks if any candidate matches the target filing and (target page OR chunk ID).
    Returns (pool_hit: bool, best_rank: int). Rank is 1-indexed (999 if NO_HIT).
    """
    for rank, c in enumerate(ranked_chunks, start=1):
        c_doc = c.get("doc_name")
        c_page = c.get("page_number", c.get("page", 0))
        c_id = c.get("chunk_id", "")
        if c_doc == gold_doc:
            if c_page in valid_pages or c_id in valid_cids:
                return True, rank
    return False, 999


def compute_cohort_metrics(
    ranks: List[int], pool_hits: List[int], lats: List[float], peak_vram: float
) -> Dict[str, Any]:
    n = len(ranks)
    if n == 0:
        return {}
    r1 = sum(1 for r in ranks if r <= 1) / n * 100
    r3 = sum(1 for r in ranks if r <= 3) / n * 100
    r5 = sum(1 for r in ranks if r <= 5) / n * 100
    r10 = sum(1 for r in ranks if r <= 10) / n * 100
    p5 = sum(1 for r in ranks if r <= 5) / (n * 5) * 100
    p10 = sum(1 for r in ranks if r <= 10) / (n * 10) * 100
    pool_rec = sum(pool_hits) / n * 100
    mrr = sum(1.0 / r for r in ranks if r <= 10) / n
    avg_lat = sum(lats) / n if n > 0 else 0.0

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
    print("=" * 80)
    print("EXPERIMENT L.7: CONTROLLED CONTEXT WINDOW STITCHING EVALUATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Verify Frozen Artifacts
    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    v2_questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l5_ground_truth_v2", "stage3_heldout_questions_v2.jsonl")

    assert compute_sha256(grounded_corpus_path) == "ea3b2ae3f424b4db463748d00438ff3f6518b06bcbd052948d48366839912ca4"
    assert compute_sha256(v2_questions_path) == "d26aac2fe7a7e269f8260dd2ceeae2d147eabba97018dd1e2453a2b0e1598530"
    print("Corpus and Benchmark V2 checksums verified.")

    # 2. Load Models and Indexes (Frozen L.2 setup)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    chunk_store = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(grounded_corpus_path, "r", encoding="utf-8")}
    faiss_index = FAISS.load_local(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "faiss_bge_m3"), embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "bm25"), load_corpus=True)

    expander = FinancialQueryExpander(enable_coreference=True)
    reranker = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True)

    # Section-Aware MultiPathRetriever (Frozen Experiment L.2)
    mpr = SectionAwareMultiPathRetriever(
        faiss_index=faiss_index,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        rrf_k=60,
        expander=expander,
        narrative_table_cap=4,
        item1_boost=0.05
    )

    # Initialize Context Window Stitcher (Budget = 1,200 chars)
    stitcher = ContextWindowStitcher(
        chunk_store=chunk_store,
        max_chars=1200,
        enable_preceding=True,
        enable_succeeding=True
    )
    print(f"Initialized ContextWindowStitcher with max_chars={stitcher.max_chars}.")

    # Load Benchmark V2 Questions
    with open(v2_questions_path, "r", encoding="utf-8") as f:
        questions_v2 = [json.loads(line) for line in f]
    print(f"Loaded {len(questions_v2)} Benchmark V2 questions.")

    # Execute Controlled Retrieval & Candidate Generation (K_pool = 65)
    print("\nExecuting candidate retrieval for all 100 questions (Frozen L.2)...")
    arm_a_outputs = []
    arm_b_outputs = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stitched_stats = {
        "total_candidates": 0,
        "stitched_candidates": 0,
        "questions_with_at_least_one_stitched": 0,
    }

    t0_all = time.time()
    for idx, q_data in enumerate(questions_v2, start=1):
        q_text = q_data["question"]
        qtype = q_data.get("question_type")

        # 1. Candidate Generation (Frozen L.2 Section-Aware Grounded K65)
        t_start_a = time.time()
        ret_pack = mpr.retrieve_multi_path(
            query=q_text,
            k_pool=65,
            enable_bm25=True,
            enable_dense=True,
            question_type=qtype,
            arm_mode="section_aware"
        )
        cands_raw = ret_pack["candidates"]

        # 2. Arm A: Rerank original candidates without stitching
        reranked_a = reranker.rerank(query=q_text, candidates=cands_raw, top_k=10, apply_boosts=True)
        lat_a = time.time() - t_start_a

        # 3. Arm B: Stitch candidates prior to reranking
        t_start_b = time.time()
        cands_stitched = []
        q_has_stitched = False
        for c in cands_raw:
            stitched_c = stitcher.stitch_candidate(c)
            # Create a modified dict for reranking with text replaced by stitched_text
            c_for_rerank = dict(stitched_c)
            c_for_rerank["text"] = stitched_c["stitched_text"]
            cands_stitched.append(c_for_rerank)

            stitched_stats["total_candidates"] += 1
            if stitched_c["is_stitched"]:
                stitched_stats["stitched_candidates"] += 1
                q_has_stitched = True

        if q_has_stitched:
            stitched_stats["questions_with_at_least_one_stitched"] += 1

        reranked_b = reranker.rerank(query=q_text, candidates=cands_stitched, top_k=10, apply_boosts=True)
        lat_b = (time.time() - t_start_b) + ret_pack.get("retrieval_latency", 0.0)

        # Invariant Verification: Candidate identities and ordering must match 100%
        cids_a = [c["chunk_id"] for c in cands_raw]
        cids_b = [c["chunk_id"] for c in cands_stitched]
        assert cids_a == cids_b, f"Invariant violation: Candidate pool IDs mismatch at question {idx}!"

        arm_a_outputs.append({
            "candidates": cands_raw,
            "reranked": reranked_a,
            "latency": lat_a,
        })

        arm_b_outputs.append({
            "candidates": cands_stitched,
            "reranked": reranked_b,
            "latency": lat_b,
        })

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    print(f"Retrieval and Reranking complete in {time.time() - t0_all:.2f}s. Peak VRAM: {peak_vram:.2f} MB")
    print(f"Stitching Statistics across 100 questions:")
    print(f"  Total Candidates Processed: {stitched_stats['total_candidates']}")
    print(f"  Candidates Stitched with Neighbors: {stitched_stats['stitched_candidates']} ({stitched_stats['stitched_candidates']/stitched_stats['total_candidates']*100:.2f}%)")
    print(f"  Questions with >=1 Stitched Candidate: {stitched_stats['questions_with_at_least_one_stitched']}/100")

    # Dual-Standard Evaluation Across Cohorts
    cohorts = {
        "cohort_primary_verified_n65": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") == "VERIFIED",
        "cohort_secondary_ambiguous_n35": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") != "VERIFIED",
        "cohort_full_benchmark_n100": lambda q: True,
    }

    standards = {
        "strict_direct": lambda ev: ev.get("evidence_type") == "direct_evidence",
        "comprehensive": lambda ev: ev.get("evidence_type") in ("direct_evidence", "contextual_table_header", "historical_annotated_reference"),
    }

    per_question_results = []

    for idx, q_data in enumerate(questions_v2):
        qid = q_data["question_id"]
        doc_name = q_data["doc_name"]
        status = q_data.get("verification_metadata", {}).get("benchmark_v2_status", "UNKNOWN")
        orig_class = q_data.get("verification_metadata", {}).get("original_annotation_classification", "UNKNOWN")
        ev_list = q_data.get("evidence_sources", [])

        # Strict direct evidence target pages & cids
        strict_pages = [e["evidence_page_num"] for e in ev_list if standards["strict_direct"](e)]
        strict_cids = [cid for e in ev_list if standards["strict_direct"](e) for cid in e.get("matching_chunk_ids", [])]

        # Comprehensive target pages & cids
        comp_pages = [e["evidence_page_num"] for e in ev_list if standards["comprehensive"](e)]
        comp_cids = [cid for e in ev_list if standards["comprehensive"](e) for cid in e.get("matching_chunk_ids", [])]

        out_a = arm_a_outputs[idx]
        out_b = arm_b_outputs[idx]

        # In case of ambiguous cohort, targets are empty []
        if status != "VERIFIED":
            strict_pages, strict_cids = [], []
            comp_pages, comp_cids = [], []

        # Evaluate Arm A
        pool_str_a, rank_str_a = evaluate_candidate_hits(out_a["candidates"], doc_name, strict_pages, strict_cids)
        top10_str_a, rank_top_str_a = evaluate_candidate_hits(out_a["reranked"], doc_name, strict_pages, strict_cids)

        pool_cmp_a, rank_cmp_a = evaluate_candidate_hits(out_a["candidates"], doc_name, comp_pages, comp_cids)
        top10_cmp_a, rank_top_cmp_a = evaluate_candidate_hits(out_a["reranked"], doc_name, comp_pages, comp_cids)

        # Evaluate Arm B
        pool_str_b, rank_str_b = evaluate_candidate_hits(out_b["candidates"], doc_name, strict_pages, strict_cids)
        top10_str_b, rank_top_str_b = evaluate_candidate_hits(out_b["reranked"], doc_name, strict_pages, strict_cids)

        pool_cmp_b, rank_cmp_b = evaluate_candidate_hits(out_b["candidates"], doc_name, comp_pages, comp_cids)
        top10_cmp_b, rank_top_cmp_b = evaluate_candidate_hits(out_b["reranked"], doc_name, comp_pages, comp_cids)

        # Count stitched candidates for this query
        num_stitched = sum(1 for c in out_b["candidates"] if c.get("is_stitched"))

        per_question_results.append({
            "index": idx + 1,
            "question_id": qid,
            "doc_name": doc_name,
            "v2_status": status,
            "original_classification": orig_class,
            "strict_direct_pages": strict_pages,
            "comprehensive_pages": comp_pages,
            "num_candidates_stitched": num_stitched,
            # Arm A (Without Stitching)
            "pool_strict_arm_a": pool_str_a,
            "rank_strict_arm_a": rank_top_str_a if top10_str_a else "NO_HIT",
            "pool_comp_arm_a": pool_cmp_a,
            "rank_comp_arm_a": rank_top_cmp_a if top10_cmp_a else "NO_HIT",
            "lat_arm_a": round(out_a["latency"], 4),
            # Arm B (With Stitching)
            "pool_strict_arm_b": pool_str_b,
            "rank_strict_arm_b": rank_top_str_b if top10_str_b else "NO_HIT",
            "pool_comp_arm_b": pool_cmp_b,
            "rank_comp_arm_b": rank_top_cmp_b if top10_cmp_b else "NO_HIT",
            "lat_arm_b": round(out_b["latency"], 4),
        })

    # Save Question-Level Results CSV
    csv_file = os.path.join(OUTPUT_DIR, "l7_question_level_results.csv")
    fieldnames = list(per_question_results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question_results)
    print(f"Saved question-level results to {csv_file}")

    # Compute Metrics for Scorecard and Comparison CSV
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "stitching_config": {
            "max_chars": stitcher.max_chars,
            "enable_preceding": stitcher.enable_preceding,
            "enable_succeeding": stitcher.enable_succeeding,
            "stitched_stats": stitched_stats,
        },
        "results": {}
    }

    comparison_rows = []

    for c_name, c_filter in cohorts.items():
        scorecard["results"][c_name] = {}
        c_indices = [i for i, q in enumerate(questions_v2) if c_filter(q)]

        for s_name in ["strict_direct", "comprehensive"]:
            scorecard["results"][c_name][s_name] = {}
            k_prefix = "strict" if s_name == "strict_direct" else "comp"

            # Gather Arm A
            ranks_a = []
            pool_hits_a = []
            lats_a = []
            for i in c_indices:
                row = per_question_results[i]
                r = row[f"rank_{k_prefix}_arm_a"]
                try:
                    r_int = int(r)
                except (ValueError, TypeError):
                    r_int = 999
                ranks_a.append(r_int)
                pool_hits_a.append(1 if row[f"pool_{k_prefix}_arm_a"] in (True, "True") else 0)
                lats_a.append(float(row["lat_arm_a"]))

            m_a = compute_cohort_metrics(ranks_a, pool_hits_a, lats_a, peak_vram)
            scorecard["results"][c_name][s_name]["arm_a"] = m_a

            # Gather Arm B
            ranks_b = []
            pool_hits_b = []
            lats_b = []
            for i in c_indices:
                row = per_question_results[i]
                r = row[f"rank_{k_prefix}_arm_b"]
                try:
                    r_int = int(r)
                except (ValueError, TypeError):
                    r_int = 999
                ranks_b.append(r_int)
                pool_hits_b.append(1 if row[f"pool_{k_prefix}_arm_b"] in (True, "True") else 0)
                lats_b.append(float(row["lat_arm_b"]))

            m_b = compute_cohort_metrics(ranks_b, pool_hits_b, lats_b, peak_vram)
            scorecard["results"][c_name][s_name]["arm_b"] = m_b

            # Comparison Row
            delta_pool = round(m_b["pool_recall"] - m_a["pool_recall"], 2)
            delta_r10 = round(m_b["recall_at_10"] - m_a["recall_at_10"], 2)
            delta_mrr = round(m_b["mrr"] - m_a["mrr"], 4)

            comparison_rows.append({
                "cohort": c_name,
                "standard": s_name,
                "n_questions": len(c_indices),
                "pool_arm_a": m_a["pool_recall"],
                "pool_arm_b": m_b["pool_recall"],
                "delta_pool_b_vs_a": delta_pool,
                "r10_arm_a": m_a["recall_at_10"],
                "r10_arm_b": m_b["recall_at_10"],
                "delta_r10_b_vs_a": delta_r10,
                "mrr_arm_a": m_a["mrr"],
                "mrr_arm_b": m_b["mrr"],
                "delta_mrr_b_vs_a": delta_mrr,
                "lat_avg_arm_a": m_a["avg_latency_sec"],
                "lat_avg_arm_b": m_b["avg_latency_sec"],
                "lat_p95_arm_a": m_a["p95_latency_sec"],
                "lat_p95_arm_b": m_b["p95_latency_sec"],
            })

    # Save Scorecard JSON
    scorecard_path = os.path.join(OUTPUT_DIR, "l7_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved scorecard to {scorecard_path}")

    # Save Comparison CSV
    comp_path = os.path.join(OUTPUT_DIR, "l7_comparison.csv")
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved comparison CSV to {comp_path}")

    # Save Execution Manifest
    manifest = {
        "benchmark_v2_input": {"path": v2_questions_path, "sha256": compute_sha256(v2_questions_path)},
        "grounded_corpus": {"path": grounded_corpus_path, "sha256": compute_sha256(grounded_corpus_path)},
        "context_window_stitcher": {"path": os.path.join(OUTPUT_DIR, "context_window_stitcher.py"), "sha256": compute_sha256(os.path.join(OUTPUT_DIR, "context_window_stitcher.py"))},
        "scorecard": {"path": scorecard_path, "sha256": compute_sha256(scorecard_path)},
        "question_results": {"path": csv_file, "sha256": compute_sha256(csv_file)},
        "comparison_summary": {"path": comp_path, "sha256": compute_sha256(comp_path)},
    }
    manifest_path = os.path.join(OUTPUT_DIR, "execution_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    print("\nBenchmark L.7 Completed Successfully.")


if __name__ == "__main__":
    main()
