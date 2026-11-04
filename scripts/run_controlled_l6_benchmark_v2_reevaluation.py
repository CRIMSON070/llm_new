"""
scripts/run_controlled_l6_benchmark_v2_reevaluation.py
=====================================================
Experiment L.6: Controlled Re-evaluation Using Stage 3 Benchmark V2.

Re-evaluates the three frozen retrieval configurations against the independently
verified ground-truth evidence annotations:
- Arm A: Original Baseline Indexes (K=50)
- Arm B: Experiment L.1 Grounded Indexes (K=65, Standard Multi-Path Control)
- Arm C: Experiment L.2 Section-Aware Grounded Indexes (K=65, Approach B)

Cohorts Evaluated:
1. Primary Cohort: 65 CONFIDENTLY VERIFIED questions
2. Secondary Cohort: 35 PARTIALLY VERIFIED / AMBIGUOUS questions
3. Combined Cohort: All 100 questions

Evaluation Standards:
1. Strict Direct Evidence Recall (evidence_type == "direct_evidence")
2. Comprehensive Evidence Recall ("direct_evidence" + "contextual_table_header")

Strict Constraints:
- Identical question text and filing IDs.
- Zero modification to indexes or models.
- Generates:
  * experiment_l6_report.md
  * l6_scorecard.json
  * l6_question_level_results.csv
  * l6_comparison.csv
  * execution_manifest.json
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

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l6_benchmark_v2_reevaluation")
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
        c_page = c.get("page", 0)
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
    print("EXPERIMENT L.6: CONTROLLED RE-EVALUATION USING STAGE 3 BENCHMARK V2")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Verify Immutability of Corpora and Benchmark V2
    baseline_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    v2_questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l5_ground_truth_v2", "stage3_heldout_questions_v2.jsonl")

    assert compute_sha256(baseline_corpus_path) == "ea4264f5d7e889d30fc4ee9147e7286b67ffa76300967fee3651002974062845"
    assert compute_sha256(grounded_corpus_path) == "ea3b2ae3f424b4db463748d00438ff3f6518b06bcbd052948d48366839912ca4"
    print("Corpora checksums verified.")

    # 2. Load Models and Indexes
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    arm_a_chunks = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(baseline_corpus_path, "r", encoding="utf-8")}
    arm_b_chunks = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(grounded_corpus_path, "r", encoding="utf-8")}

    arm_a_faiss = FAISS.load_local(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3"), embeddings, allow_dangerous_deserialization=True)
    arm_a_bm25 = BM25Retriever(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25"), load_corpus=True)

    arm_b_faiss = FAISS.load_local(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "faiss_bge_m3"), embeddings, allow_dangerous_deserialization=True)
    arm_b_bm25 = BM25Retriever(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "bm25"), load_corpus=True)

    expander = FinancialQueryExpander(enable_coreference=True)
    reranker = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=True, expander=expander, enable_coreference=True)

    mpr_arm_a = MultiPathRetriever(faiss_index=arm_a_faiss, bm25_retriever=arm_a_bm25, chunk_store=arm_a_chunks, rrf_k=60, expander=expander)
    mpr_arm_b_c = SectionAwareMultiPathRetriever(faiss_index=arm_b_faiss, bm25_retriever=arm_b_bm25, chunk_store=arm_b_chunks, rrf_k=60, expander=expander, narrative_table_cap=4, item1_boost=0.05)

    # 3. Load Benchmark V2 Questions
    with open(v2_questions_path, "r", encoding="utf-8") as f:
        questions_v2 = [json.loads(line) for line in f]
    print(f"Loaded {len(questions_v2)} Benchmark V2 questions.")

    # 4. Execute Benchmark Arms and Record Candidate Outputs
    # We execute all 3 arms under strictly controlled conditions and evaluate against V2 evidence
    arm_runs = {
        "arm_a": {"name": "Arm A (Baseline K=50)", "mpr": mpr_arm_a, "k_pool": 50, "mode": "baseline"},
        "arm_b": {"name": "Arm B (Exp L.1 Grounded K=65)", "mpr": mpr_arm_b_c, "k_pool": 65, "mode": "control"},
        "arm_c": {"name": "Arm C (Exp L.2 Section-Aware K=65)", "mpr": mpr_arm_b_c, "k_pool": 65, "mode": "section_aware"},
    }

    raw_arm_outputs = {}

    for arm_key, cfg in arm_runs.items():
        print(f"\nExecuting {cfg['name']}...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        outputs = []
        for idx, q_data in enumerate(questions_v2, start=1):
            q_text = q_data["question"]
            qtype = q_data.get("question_type")

            t0 = time.time()
            if cfg["mode"] == "baseline":
                ret_pack = cfg["mpr"].retrieve_multi_path(query=q_text, k_pool=cfg["k_pool"], enable_bm25=True, enable_dense=True)
            elif cfg["mode"] == "control":
                ret_pack = cfg["mpr"].retrieve_multi_path(query=q_text, k_pool=cfg["k_pool"], enable_bm25=True, enable_dense=True, arm_mode="control")
            else:
                ret_pack = cfg["mpr"].retrieve_multi_path(query=q_text, k_pool=cfg["k_pool"], enable_bm25=True, enable_dense=True, question_type=qtype, arm_mode="section_aware")

            cands = ret_pack["candidates"]
            reranked = reranker.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
            lat = time.time() - t0

            outputs.append({
                "candidates": cands,
                "reranked": reranked,
                "latency": lat,
            })

        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        raw_arm_outputs[arm_key] = {"outputs": outputs, "peak_vram": peak_vram}

    # 5. Dual-Standard Evaluation Across Cohorts
    cohorts = {
        "cohort_primary_verified_n65": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") == "VERIFIED",
        "cohort_secondary_ambiguous_n35": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") != "VERIFIED",
        "cohort_full_benchmark_n100": lambda q: True,
    }

    standards = {
        "strict_direct": lambda ev: ev.get("evidence_type") == "direct_evidence",
        "comprehensive": lambda ev: ev.get("evidence_type") in ("direct_evidence", "contextual_table_header", "historical_annotated_reference"),
    }

    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "results": {}
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

        row = {
            "index": idx + 1,
            "question_id": qid,
            "doc_name": doc_name,
            "v2_status": status,
            "original_classification": orig_class,
            "strict_direct_pages": str(strict_pages),
            "comprehensive_pages": str(comp_pages),
        }

        for arm_key in ["arm_a", "arm_b", "arm_c"]:
            out = raw_arm_outputs[arm_key]["outputs"][idx]
            cands = out["candidates"]
            reranked = out["reranked"]

            # Strict evaluation
            p_hit_s, p_rank_s = evaluate_candidate_hits(cands, doc_name, strict_pages, strict_cids)
            r_hit_s, r_rank_s = evaluate_candidate_hits(reranked, doc_name, strict_pages, strict_cids)

            # Comprehensive evaluation
            p_hit_c, p_rank_c = evaluate_candidate_hits(cands, doc_name, comp_pages, comp_cids)
            r_hit_c, r_rank_c = evaluate_candidate_hits(reranked, doc_name, comp_pages, comp_cids)

            row[f"pool_strict_{arm_key}"] = p_hit_s
            row[f"rank_strict_{arm_key}"] = r_rank_s if r_rank_s <= 10 else "NO_HIT"
            row[f"pool_comp_{arm_key}"] = p_hit_c
            row[f"rank_comp_{arm_key}"] = r_rank_c if r_rank_c <= 10 else "NO_HIT"
            row[f"lat_{arm_key}"] = round(out["latency"], 4)

        per_question_results.append(row)

    # Compute metrics for each cohort and standard
    for c_name, c_filter in cohorts.items():
        scorecard["results"][c_name] = {}
        cohort_indices = [i for i, q in enumerate(questions_v2) if c_filter(q)]

        for std_name, std_filter in standards.items():
            scorecard["results"][c_name][std_name] = {}

            for arm_key in ["arm_a", "arm_b", "arm_c"]:
                arm_out = raw_arm_outputs[arm_key]
                ranks = []
                pool_hits = []
                lats = []

                for idx in cohort_indices:
                    q_data = questions_v2[idx]
                    doc_name = q_data["doc_name"]
                    ev_list = q_data.get("evidence_sources", [])
                    pages = [e["evidence_page_num"] for e in ev_list if std_filter(e)]
                    cids = [cid for e in ev_list if std_filter(e) for cid in e.get("matching_chunk_ids", [])]

                    cand_out = arm_out["outputs"][idx]
                    p_hit, _ = evaluate_candidate_hits(cand_out["candidates"], doc_name, pages, cids)
                    _, r_rank = evaluate_candidate_hits(cand_out["reranked"], doc_name, pages, cids)

                    pool_hits.append(1 if p_hit else 0)
                    ranks.append(r_rank)
                    lats.append(cand_out["latency"])

                m = compute_cohort_metrics(ranks, pool_hits, lats, arm_out["peak_vram"])
                scorecard["results"][c_name][std_name][arm_key] = m

    # Save Scorecard
    scorecard_path = os.path.join(OUTPUT_DIR, "l6_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nSaved Scorecard to: {scorecard_path}")

    # Save Question-Level Results
    results_csv_path = os.path.join(OUTPUT_DIR, "l6_question_level_results.csv")
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_question_results)
    print(f"Saved Question-Level CSV to: {results_csv_path}")

    # Save Summary Comparison CSV
    comp_rows = []
    for c_name in cohorts:
        for std_name in standards:
            m_a = scorecard["results"][c_name][std_name]["arm_a"]
            m_b = scorecard["results"][c_name][std_name]["arm_b"]
            m_c = scorecard["results"][c_name][std_name]["arm_c"]
            comp_rows.append({
                "cohort": c_name,
                "standard": std_name,
                "n_questions": m_a["n_questions"],
                "pool_arm_a": m_a["pool_recall"],
                "pool_arm_b": m_b["pool_recall"],
                "pool_arm_c": m_c["pool_recall"],
                "delta_pool_c_vs_b": round(m_c["pool_recall"] - m_b["pool_recall"], 2),
                "delta_pool_c_vs_a": round(m_c["pool_recall"] - m_a["pool_recall"], 2),
                "r10_arm_a": m_a["recall_at_10"],
                "r10_arm_b": m_b["recall_at_10"],
                "r10_arm_c": m_c["recall_at_10"],
                "delta_r10_c_vs_b": round(m_c["recall_at_10"] - m_b["recall_at_10"], 2),
                "delta_r10_c_vs_a": round(m_c["recall_at_10"] - m_a["recall_at_10"], 2),
                "mrr_arm_a": m_a["mrr"],
                "mrr_arm_b": m_b["mrr"],
                "mrr_arm_c": m_c["mrr"],
                "delta_mrr_c_vs_b": round(m_c["mrr"] - m_b["mrr"], 4),
                "lat_avg_arm_c": m_c["avg_latency_sec"],
                "lat_p95_arm_c": m_c["p95_latency_sec"],
            })

    comp_csv_path = os.path.join(OUTPUT_DIR, "l6_comparison.csv")
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"Saved Summary Comparison CSV to: {comp_csv_path}")

    # Save Execution Manifest
    manifest = {
        "benchmark_v2_input": {
            "path": v2_questions_path,
            "sha256": compute_sha256(v2_questions_path)
        },
        "scorecard": {
            "path": scorecard_path,
            "sha256": compute_sha256(scorecard_path)
        },
        "question_results": {
            "path": results_csv_path,
            "sha256": compute_sha256(results_csv_path)
        },
        "comparison_summary": {
            "path": comp_csv_path,
            "sha256": compute_sha256(comp_csv_path)
        }
    }
    manifest_path = os.path.join(OUTPUT_DIR, "execution_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Execution Manifest to: {manifest_path}")

    print("\nBenchmark V2 Re-evaluation Complete!")


if __name__ == "__main__":
    main()
