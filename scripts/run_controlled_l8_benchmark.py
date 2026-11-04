"""
Experiment L.8: Context Window Stitching Optimization and Ablation Runner
=========================================================================

Evaluates 5 controlled arms on frozen candidate pools from Experiment L.2:
- Arm A: No stitching (frozen L.2 baseline)
- Arm B: Prev + Candidate + Next, 1,200 chars (reproducing L.7)
- Arm C: Prev + Candidate only, 1,200 chars (asymmetric backward)
- Arm D: Candidate + Next only, 1,200 chars (asymmetric forward)
- Arm E: Prev + Candidate + Next, 800 chars (compact symmetric)

Strict Requirements:
1. Candidate pools are generated once and frozen across all arms.
2. Invariant verification: pool chunk_ids and ranks are identical across all arms.
3. Candidate identities are preserved.
4. Evaluation across Primary Verified Cohort (N=65) and Full Benchmark (N=100)
   using Strict Direct and Comprehensive Evidence standards.
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
from outputs.experiments.experiment_l8_stitching_optimization.optimized_context_stitcher import (
    OptimizedContextWindowStitcher
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l8_stitching_optimization")
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
    print("EXPERIMENT L.8: CONTEXT WINDOW STITCHING OPTIMIZATION & ABLATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    grounded_corpus_path = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl")
    v2_questions_path = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_l5_ground_truth_v2", "stage3_heldout_questions_v2.jsonl")

    assert compute_sha256(grounded_corpus_path) == "ea3b2ae3f424b4db463748d00438ff3f6518b06bcbd052948d48366839912ca4"
    assert compute_sha256(v2_questions_path) == "d26aac2fe7a7e269f8260dd2ceeae2d147eabba97018dd1e2453a2b0e1598530"
    print("Corpus and Benchmark V2 checksums verified.")

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
    reranker = Phase2Reranker(model_name="BAAI/bge-reranker-base", use_fp16=(device == "cuda"), expander=expander, enable_coreference=True)

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

    # 1. Generate Frozen Candidate Pools for all 100 questions (Frozen L.2)
    print("\nPhase 1: Generating and freezing candidate pools (K_pool=65)...")
    frozen_candidate_pools = []
    pool_latencies = []
    for idx, q_data in enumerate(questions_v2, start=1):
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
        pool_lat = time.time() - t0
        cands = ret_pack["candidates"]
        frozen_candidate_pools.append(cands)
        pool_latencies.append(pool_lat)

    print(f"Candidate pools generated for all 100 questions. Avg retrieval latency: {sum(pool_latencies)/len(pool_latencies):.4f}s")

    # Define the 5 Experimental Arms
    # Arm A: No stitching
    # Arm B: Prev + Cand + Next, 1200 chars (L.7)
    # Arm C: Prev + Cand only, 1200 chars
    # Arm D: Cand + Next only, 1200 chars
    # Arm E: Prev + Cand + Next, 800 chars
    arms_config = {
        "arm_a": {
            "name": "Arm A: Frozen L.2 Baseline (No Stitching)",
            "stitcher": None,
            "max_chars": 0,
        },
        "arm_b": {
            "name": "Arm B: Prev + Cand + Next (1200 chars, L.7 Reproduction)",
            "stitcher": OptimizedContextWindowStitcher(chunk_store, max_chars=1200, enable_preceding=True, enable_succeeding=True),
            "max_chars": 1200,
        },
        "arm_c": {
            "name": "Arm C: Prev + Cand Only (1200 chars, Backward Asymmetric)",
            "stitcher": OptimizedContextWindowStitcher(chunk_store, max_chars=1200, enable_preceding=True, enable_succeeding=False),
            "max_chars": 1200,
        },
        "arm_d": {
            "name": "Arm D: Cand + Next Only (1200 chars, Forward Asymmetric)",
            "stitcher": OptimizedContextWindowStitcher(chunk_store, max_chars=1200, enable_preceding=False, enable_succeeding=True),
            "max_chars": 1200,
        },
        "arm_e": {
            "name": "Arm E: Prev + Cand + Next (800 chars, Compact Symmetric)",
            "stitcher": OptimizedContextWindowStitcher(chunk_store, max_chars=800, enable_preceding=True, enable_succeeding=True),
            "max_chars": 800,
        },
    }

    arm_outputs = {k: [] for k in arms_config}
    arm_stitched_counts = {k: 0 for k in arms_config}
    arm_peak_vram = {}

    for arm_key, arm_cfg in arms_config.items():
        print(f"\nPhase 2: Executing Reranking for {arm_cfg['name']}...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_arm_start = time.time()
        for idx, q_data in enumerate(questions_v2):
            q_text = q_data["question"]
            cands_raw = frozen_candidate_pools[idx]
            ret_lat = pool_latencies[idx]

            t0 = time.time()
            if arm_cfg["stitcher"] is None:
                # Arm A: Direct reranking
                cands_to_rerank = cands_raw
            else:
                # Arms B, C, D, E: Stitch candidates
                stitcher = arm_cfg["stitcher"]
                cands_to_rerank = []
                for c in cands_raw:
                    sc = stitcher.stitch_candidate(c)
                    c_dict = dict(sc)
                    c_dict["text"] = sc["stitched_text"]
                    cands_to_rerank.append(c_dict)
                    if sc.get("is_stitched"):
                        arm_stitched_counts[arm_key] += 1

            reranked = reranker.rerank(query=q_text, candidates=cands_to_rerank, top_k=10, apply_boosts=True)
            rerank_lat = time.time() - t0
            total_lat = ret_lat + rerank_lat

            # Invariant check: candidate identity preservation
            orig_cids = [c["chunk_id"] for c in cands_raw]
            rerank_pool_cids = [c["chunk_id"] for c in cands_to_rerank]
            assert orig_cids == rerank_pool_cids, f"Candidate identity mismatch in {arm_key} at question {idx}!"

            arm_outputs[arm_key].append({
                "candidates": cands_to_rerank,
                "reranked": reranked,
                "latency": total_lat,
            })

        arm_time = time.time() - t_arm_start
        vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        arm_peak_vram[arm_key] = vram
        print(f"  Finished {arm_key} in {arm_time:.2f}s. Stitched chunks: {arm_stitched_counts[arm_key]}. Peak VRAM: {vram:.2f} MB")

    # Evaluation Across Cohorts and Standards
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

        for arm_key in arms_config:
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
    csv_file = os.path.join(OUTPUT_DIR, "l8_question_level_results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_question_rows)
    print(f"\nSaved question-level results to {csv_file}")

    # Build Scorecard and Comparison CSV
    scorecard = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "arms_configuration": {
            k: {
                "name": v["name"],
                "max_chars": v["max_chars"],
                "total_stitched_candidates": arm_stitched_counts[k],
                "peak_vram_mb": arm_peak_vram[k],
            }
            for k, v in arms_config.items()
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
            for arm_key in arms_config:
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

            # Baseline Arm A for comparison deltas
            base_m = arm_metrics["arm_a"]
            for arm_key in arms_config:
                curr_m = arm_metrics[arm_key]
                comparison_rows.append({
                    "cohort": c_name,
                    "standard": s_name,
                    "arm": arm_key,
                    "arm_name": arms_config[arm_key]["name"],
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
                    "total_stitched": arm_stitched_counts[arm_key],
                })

    scorecard_path = os.path.join(OUTPUT_DIR, "l8_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved scorecard to {scorecard_path}")

    comp_path = os.path.join(OUTPUT_DIR, "l8_comparison.csv")
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved comparison summary to {comp_path}")

    manifest = {
        "benchmark_v2_input": {"path": v2_questions_path, "sha256": compute_sha256(v2_questions_path)},
        "grounded_corpus": {"path": grounded_corpus_path, "sha256": compute_sha256(grounded_corpus_path)},
        "optimized_stitcher": {"path": os.path.join(OUTPUT_DIR, "optimized_context_stitcher.py"), "sha256": compute_sha256(os.path.join(OUTPUT_DIR, "optimized_context_stitcher.py"))},
        "scorecard": {"path": scorecard_path, "sha256": compute_sha256(scorecard_path)},
        "question_results": {"path": csv_file, "sha256": compute_sha256(csv_file)},
        "comparison_summary": {"path": comp_path, "sha256": compute_sha256(comp_path)},
    }
    manifest_path = os.path.join(OUTPUT_DIR, "execution_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    print("\nExperiment L.8 Execution Complete.")


if __name__ == "__main__":
    main()
