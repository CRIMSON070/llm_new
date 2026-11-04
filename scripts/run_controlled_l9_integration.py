"""
Experiment L.9: Forward Stitching Integration & Reproducibility Validation Runner
=================================================================================

Controlled Validation:
- Arm A (Production Integration - No Stitching):
  Retriever + Phase2Reranker(context_stitcher=None)
- Arm D (Production Integration - Forward Stitching):
  Retriever + Phase2Reranker(context_stitcher=ContextWindowStitcher(mode='forward', max_chars=1200))

Evaluated against Stage 3 Benchmark V2:
- Primary Verified Cohort (N=65)
- Secondary Ambiguous Cohort (N=35) (isolated)
- Strict Direct Evidence & Comprehensive Evidence standards

Validation Goal:
Confirm that integrated Arm D reproduces the measured L.8 results (R@10=46.15%, MRR=0.2348, R@1=15.38%).
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
from src.retrieval.context_stitcher import ContextWindowStitcher
from outputs.experiments.experiment_l2_section_aware.section_aware_retriever import (
    SectionAwareMultiPathRetriever
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l9_integration_validation")
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
    for rank, c in enumerate(ranked_chunks, start=1):
        c_doc = c.get("doc_name")
        c_page = c.get("page_number", c.get("page", 0))
        c_id = c.get("chunk_id", "")
        if c_doc == gold_doc:
            if c_page in valid_pages or c_id in valid_cids:
                return True, rank
    return False, 999


def compute_cohort_metrics(
    ranks: List[int], pool_hits: List[int], lats: List[float], peak_vram: float = 0.0
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
    print("EXPERIMENT L.9: FORWARD STITCHING INTEGRATION & REPRODUCIBILITY VALIDATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    v2_questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l5_ground_truth_v2", "stage3_heldout_questions_v2.jsonl")

    corpus_sha = compute_sha256(grounded_corpus_path)
    v2_sha = compute_sha256(v2_questions_path)
    assert corpus_sha == "ea3b2ae3f424b4db463748d00438ff3f6518b06bcbd052948d48366839912ca4"
    assert v2_sha == "d26aac2fe7a7e269f8260dd2ceeae2d147eabba97018dd1e2453a2b0e1598530"
    print("Corpus and Benchmark V2 checksums strictly verified.")

    # Load Models and Index
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    chunk_store = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(grounded_corpus_path, "r", encoding="utf-8")}
    faiss_index = FAISS.load_local(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "faiss_bge_m3"), embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever(os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes_grounded_exp_l", "bm25"), load_corpus=True)

    expander = FinancialQueryExpander(enable_coreference=True)

    # Instantiate production stitcher (forward mode, max 1200 chars)
    forward_stitcher = ContextWindowStitcher(chunk_store=chunk_store, mode="forward", max_chars=1200)

    # Reranker Arm A: Integrated Phase2Reranker with context_stitcher=None
    reranker_arm_a = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=(device == "cuda"),
        expander=expander,
        enable_coreference=True,
        context_stitcher=None
    )

    # Reranker Arm D: Integrated Phase2Reranker with context_stitcher=forward_stitcher
    reranker_arm_d = Phase2Reranker(
        model_name="BAAI/bge-reranker-base",
        use_fp16=(device == "cuda"),
        expander=expander,
        enable_coreference=True,
        context_stitcher=forward_stitcher
    )

    # Retriever (Frozen Section-Aware Grounded K=65)
    mpr = SectionAwareMultiPathRetriever(
        faiss_index=faiss_index,
        bm25_retriever=bm25_retriever,
        chunk_store=chunk_store,
        rrf_k=60,
        expander=expander,
        narrative_table_cap=4,
        item1_boost=0.05
    )

    with open(v2_questions_path, "r", encoding="utf-8") as f:
        questions_v2 = [json.loads(line) for line in f]
    print(f"Loaded {len(questions_v2)} Benchmark V2 questions.")

    # Phase 1: Candidate Pool Generation
    print("\nPhase 1: Retrieving candidates (Frozen K_pool=65)...")
    candidate_pools = []
    pool_latencies = []
    for idx, q_data in enumerate(questions_v2):
        q_text = q_data["question"]
        qtype = q_data.get("question_type")

        t0 = time.time()
        ret_pack = mpr.retrieve_multi_path(
            query=q_text,
            k_pool=65,
            enable_bm25=True,
            enable_dense=True,
            question_type=qtype,
            arm_mode="section_aware"
        )
        lat = time.time() - t0
        candidate_pools.append(ret_pack["candidates"])
        pool_latencies.append(lat)

    print(f"Candidate pools generated. Avg retrieval latency: {sum(pool_latencies)/len(pool_latencies):.4f}s")

    # Phase 2: Integrated Reranking for Arm A and Arm D
    integrated_arms = {
        "arm_a_integrated": {
            "name": "Integrated Arm A: Frozen Baseline (No Stitching)",
            "reranker": reranker_arm_a,
        },
        "arm_d_integrated": {
            "name": "Integrated Arm D: Production Forward Stitching (1200 chars)",
            "reranker": reranker_arm_d,
        },
    }

    arm_outputs = {k: [] for k in integrated_arms}
    arm_peak_vram = {}

    for arm_key, arm_cfg in integrated_arms.items():
        print(f"\nPhase 2: Executing Reranking for {arm_cfg['name']}...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        r = arm_cfg["reranker"]

        for idx, q_data in enumerate(questions_v2):
            q_text = q_data["question"]
            cands = candidate_pools[idx]
            ret_lat = pool_latencies[idx]

            t0 = time.time()
            reranked = r.rerank(query=q_text, candidates=cands, top_k=10, apply_boosts=True)
            rerank_lat = time.time() - t0
            total_lat = ret_lat + rerank_lat

            # Invariant check: candidate identities must match exactly
            orig_cids = [c["chunk_id"] for c in cands]
            rerank_cids = [c["chunk_id"] for c in reranked]
            for rk_cid in rerank_cids:
                assert rk_cid in orig_cids, f"Candidate identity violation in {arm_key}!"

            arm_outputs[arm_key].append({
                "candidates": cands,
                "reranked": reranked,
                "latency": total_lat,
            })

        total_time = time.time() - t_start
        vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        arm_peak_vram[arm_key] = vram
        print(f"  Finished {arm_key} in {total_time:.2f}s. Peak VRAM: {vram:.2f} MB")

    # Evaluation Across Cohorts
    cohorts = {
        "cohort_primary_verified_n65": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") == "VERIFIED",
        "cohort_secondary_ambiguous_n35": lambda q: q.get("verification_metadata", {}).get("benchmark_v2_status") != "VERIFIED",
        "cohort_full_benchmark_n100": lambda q: True,
    }

    standards = {
        "strict_direct": lambda ev: ev.get("evidence_type") == "direct_evidence",
        "comprehensive": lambda ev: ev.get("evidence_type") in ("direct_evidence", "contextual_table_header", "historical_annotated_reference"),
    }

    per_question_rows = []

    for idx, q_data in enumerate(questions_v2):
        qid = q_data["question_id"]
        doc_name = q_data["doc_name"]
        status = q_data.get("verification_metadata", {}).get("benchmark_v2_status", "UNKNOWN")
        orig_class = q_data.get("verification_metadata", {}).get("original_annotation_classification", "UNKNOWN")
        ev_list = q_data.get("evidence_sources", [])

        strict_pages = [e["evidence_page_num"] for e in ev_list if standards["strict_direct"](e)]
        strict_cids = [cid for e in ev_list if standards["strict_direct"](e) for cid in e.get("matching_chunk_ids", [])]

        comp_pages = [e["evidence_page_num"] for e in ev_list if standards["comprehensive"](e)]
        comp_cids = [cid for e in ev_list if standards["comprehensive"](e) for cid in e.get("matching_chunk_ids", [])]

        if status != "VERIFIED":
            strict_pages, strict_cids = [], []
            comp_pages, comp_cids = [], []

        row = {
            "index": idx + 1,
            "question_id": qid,
            "doc_name": doc_name,
            "v2_status": status,
            "original_classification": orig_class,
            "strict_direct_pages": strict_pages,
            "comprehensive_pages": comp_pages,
        }

        for arm_key in integrated_arms:
            out = arm_outputs[arm_key][idx]
            p_str, r_str = evaluate_candidate_hits(out["candidates"], doc_name, strict_pages, strict_cids)
            t_str, rk_str = evaluate_candidate_hits(out["reranked"], doc_name, strict_pages, strict_cids)

            p_cmp, r_cmp = evaluate_candidate_hits(out["candidates"], doc_name, comp_pages, comp_cids)
            t_cmp, rk_cmp = evaluate_candidate_hits(out["reranked"], doc_name, comp_pages, comp_cids)

            row[f"pool_strict_{arm_key}"] = p_str
            row[f"rank_strict_{arm_key}"] = rk_str if t_str else "NO_HIT"
            row[f"pool_comp_{arm_key}"] = p_cmp
            row[f"rank_comp_{arm_key}"] = rk_cmp if t_cmp else "NO_HIT"
            row[f"lat_{arm_key}"] = round(out["latency"], 4)

        per_question_rows.append(row)

    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, "l9_question_level_results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_question_rows)
    print(f"\nSaved question-level results to {csv_file}")

    # Build Scorecard and Comparison
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "integrated_arms_configuration": {
            k: {
                "name": v["name"],
                "peak_vram_mb": arm_peak_vram[k],
            }
            for k, v in integrated_arms.items()
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

            arm_metrics = {}
            for arm_key in integrated_arms:
                ranks, pool_hits, lats = [], [], []
                for i in c_indices:
                    r_val = per_question_rows[i][f"rank_{k_prefix}_{arm_key}"]
                    try:
                        r_int = int(r_val)
                    except (ValueError, TypeError):
                        r_int = 999
                    ranks.append(r_int)
                    pool_hits.append(1 if per_question_rows[i][f"pool_{k_prefix}_{arm_key}"] in (True, "True") else 0)
                    lats.append(float(per_question_rows[i][f"lat_{arm_key}"]))

                m = compute_cohort_metrics(ranks, pool_hits, lats, arm_peak_vram[arm_key])
                scorecard["results"][c_name][s_name][arm_key] = m
                arm_metrics[arm_key] = m

            base_m = arm_metrics["arm_a_integrated"]
            for arm_key in integrated_arms:
                curr_m = arm_metrics[arm_key]
                comparison_rows.append({
                    "cohort": c_name,
                    "standard": s_name,
                    "arm": arm_key,
                    "arm_name": integrated_arms[arm_key]["name"],
                    "n_questions": len(c_indices),
                    "pool_recall": curr_m["pool_recall"],
                    "delta_pool_vs_base": round(curr_m["pool_recall"] - base_m["pool_recall"], 2),
                    "recall_at_1": curr_m["recall_at_1"],
                    "delta_r1_vs_base": round(curr_m["recall_at_1"] - base_m["recall_at_1"], 2),
                    "recall_at_3": curr_m["recall_at_3"],
                    "recall_at_5": curr_m["recall_at_5"],
                    "recall_at_10": curr_m["recall_at_10"],
                    "delta_r10_vs_base": round(curr_m["recall_at_10"] - base_m["recall_at_10"], 2),
                    "precision_at_5": curr_m["precision_at_5"],
                    "precision_at_10": curr_m["precision_at_10"],
                    "mrr": curr_m["mrr"],
                    "delta_mrr_vs_base": round(curr_m["mrr"] - base_m["mrr"], 4),
                    "avg_latency_sec": curr_m["avg_latency_sec"],
                    "p95_latency_sec": curr_m["p95_latency_sec"],
                    "peak_vram_mb": curr_m["peak_vram_mb"],
                })

    scorecard_path = os.path.join(OUTPUT_DIR, "l9_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved scorecard to {scorecard_path}")

    comp_path = os.path.join(OUTPUT_DIR, "l9_comparison.csv")
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved comparison summary to {comp_path}")

    manifest = {
        "benchmark_v2_input": {"path": v2_questions_path, "sha256": v2_sha},
        "grounded_corpus": {"path": grounded_corpus_path, "sha256": corpus_sha},
        "integrated_stitcher_module": {"path": os.path.join(BASE_DIR, "src", "retrieval", "context_stitcher.py"), "sha256": compute_sha256(os.path.join(BASE_DIR, "src", "retrieval", "context_stitcher.py"))},
        "integrated_reranker_module": {"path": os.path.join(BASE_DIR, "src", "retrieval", "phase2_reranker.py"), "sha256": compute_sha256(os.path.join(BASE_DIR, "src", "retrieval", "phase2_reranker.py"))},
        "scorecard": {"path": scorecard_path, "sha256": compute_sha256(scorecard_path)},
        "question_results": {"path": csv_file, "sha256": compute_sha256(csv_file)},
        "comparison_summary": {"path": comp_path, "sha256": compute_sha256(comp_path)},
    }
    manifest_path = os.path.join(OUTPUT_DIR, "execution_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    print("\nExperiment L.9 Integration Validation Complete.")


if __name__ == "__main__":
    main()
