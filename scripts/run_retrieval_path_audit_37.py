"""
FinSage Instrumented Retrieval Audit for Stage 1 Candidate-Pool Failures
========================================================================
Audits the exact 37 questions that failed Candidate Pool Recall@40 in the
Stage 1 reproduction benchmark (outputs/reproduction_stage1_faithful/).

Captures full pre-fusion ranked lists and raw scores for:
- Path 1: BM25 Sparse Lexical Retrieval (k=40)
- Path 2: BGE-M3 Dense FAISS Retrieval (k=40)
- Path 3: Metadata Heading-Summary FAISS Retrieval (k_sections=4)
- Path 4: Generative HyDE FAISS Retrieval (k_per_hypo=20, cached Qwen2.5 passages)
Followed by complete Reciprocal Rank Fusion (RRF, k=60, pool=40) logging.

All outputs are isolated in: outputs/retrieval_path_audit_37/
ZERO modification to original Stage 1 outputs, indexes, or source code.
"""

import os
import sys
import gc
import json
import time
import math
import hashlib
import argparse
import csv
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever


# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
AUDIT_LABEL = "Instrumented Retrieval Audit of 37 Stage 1 Candidate-Pool Failures"

QUESTIONS_FILE = os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "financebench", "processed", "parsed_chunks.jsonl")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "faiss_index")
BM25_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "bm25_index")
META_INDEX_DIR = os.path.join(BASE_DIR, "data", "financebench", "metadata_faiss_index")

DPO_TRAIN_FILE = os.path.join(BASE_DIR, "data", "financebench", "drr", "train_drr.jsonl")
DPO_VAL_FILE = os.path.join(BASE_DIR, "data", "financebench", "drr", "val_drr.jsonl")

STAGE1_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reproduction_stage1_faithful")
STAGE1_CHECKPOINT_FILE = os.path.join(STAGE1_OUTPUT_DIR, "stage1_full_retrieval_checkpoint.jsonl")
HYDE_CACHE_FILE = os.path.join(STAGE1_OUTPUT_DIR, "hyde_generated_passages_cache.json")

AUDIT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "retrieval_path_audit_37")

# Exact original Stage 1 retrieval & fusion parameters
RRF_K = 60
K_POOL = 40
BM25_K = 40
DENSE_K = 40
METADATA_K_SECTIONS = 4
HYDE_K_PER_HYPO = 20

# -----------------------------------------------------------------------------
# Integrity & Protected Hashes Audit
# -----------------------------------------------------------------------------
PROTECTED_TARGETS = [
    # Checkpoints, adapters, configs
    os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter", "adapter_model.safetensors"),
    os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter", "adapter_config.json"),
    
    # Datasets, corpus, manifests
    os.path.join(BASE_DIR, "data", "financebench", "questions.jsonl"),
    os.path.join(BASE_DIR, "data", "financebench", "drr", "train_drr.jsonl"),
    os.path.join(BASE_DIR, "data", "financebench", "drr", "val_drr.jsonl"),
    os.path.join(BASE_DIR, "data", "financebench", "processed", "parsed_chunks.jsonl"),
    os.path.join(BASE_DIR, "data", "financebench", "manifest.json"),
    
    # Stage 1 Faithful Benchmark Outputs
    os.path.join(STAGE1_OUTPUT_DIR, "stage1_full_retrieval_checkpoint.jsonl"),
    os.path.join(STAGE1_OUTPUT_DIR, "scorecard_stage1_full.json"),
    os.path.join(STAGE1_OUTPUT_DIR, "hyde_generated_passages_cache.json"),
    os.path.join(STAGE1_OUTPUT_DIR, "stage1_reproduction_report.md"),
    
    # FAISS Dense Index Files
    os.path.join(FAISS_INDEX_DIR, "index.faiss"),
    os.path.join(FAISS_INDEX_DIR, "index.pkl"),
    
    # BM25S Lexical Index Files
    os.path.join(BM25_INDEX_DIR, "corpus.jsonl"),
    os.path.join(BM25_INDEX_DIR, "corpus.mmindex.json"),
    os.path.join(BM25_INDEX_DIR, "data.csc.index.npy"),
    os.path.join(BM25_INDEX_DIR, "indices.csc.index.npy"),
    os.path.join(BM25_INDEX_DIR, "indptr.csc.index.npy"),
    os.path.join(BM25_INDEX_DIR, "params.index.json"),
    os.path.join(BM25_INDEX_DIR, "vocab.index.json"),
    
    # Metadata Heading-Summary Index Files
    os.path.join(META_INDEX_DIR, "index.faiss"),
    os.path.join(META_INDEX_DIR, "index.pkl"),
    os.path.join(META_INDEX_DIR, "sections_catalog.json"),
]

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.isfile(filepath):
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def record_protected_hashes() -> Dict[str, str]:
    """Capture hashes of protected artifacts."""
    return {p: compute_file_sha256(p) for p in PROTECTED_TARGETS}

def verify_protected_hashes(baseline_hashes: Dict[str, str]):
    """Verify no protected file was altered during the audit run."""
    current_hashes = record_protected_hashes()
    tampered = []
    for path, expected in baseline_hashes.items():
        curr = current_hashes.get(path)
        if curr != expected:
            tampered.append((path, expected, curr))
    if tampered:
        raise RuntimeError(f"CRITICAL: Protected files modified during audit! {tampered}")
    print("[Integrity] Verified all protected SHA-256 hashes remain identical.")


# -----------------------------------------------------------------------------
# Metadata Parsing Utilities
# -----------------------------------------------------------------------------
def get_doc_metadata(doc_name: str) -> Dict[str, Any]:
    """Extract company, year, and filing type from doc_name."""
    if not doc_name:
        return {"company": "UNKNOWN", "year": "UNKNOWN", "filing_type": "UNKNOWN"}
    parts = doc_name.split("_")
    company_tokens = []
    for p in parts:
        if any(char.isdigit() for char in p):
            break
        company_tokens.append(p)
    company = "_".join(company_tokens) if company_tokens else parts[0]
    
    filing_type = "UNKNOWN"
    if "10K" in doc_name:
        filing_type = "10-K"
    elif "10Q" in doc_name:
        filing_type = "10-Q"
    elif "EARNINGS" in doc_name:
        filing_type = "Earnings"
    elif "8K" in doc_name:
        filing_type = "8-K"
        
    year = "UNKNOWN"
    for p in parts:
        for y in range(2015, 2030):
            if str(y) in p:
                year = str(y)
                break
        if year != "UNKNOWN":
            break

    return {"company": company, "year": year, "filing_type": filing_type}


# -----------------------------------------------------------------------------
# Instrumented Multi-Path Retriever Class
# -----------------------------------------------------------------------------
class InstrumentedMultiPathRetriever(MultiPathRetriever):
    """
    Subclass of MultiPathRetriever that intercepts and logs full, untruncated
    path results and per-path score contributions before and after RRF.
    Uses exact original retrieval methods and parameters.
    """
    def retrieve_instrumented(
        self,
        query: str,
        k_pool: int = 40
    ) -> Dict[str, Any]:
        path_results = {}
        telemetry = {}

        # 1. Path 1: BM25 Sparse Lexical
        sparse_res = self.retrieve_sparse(query, k=BM25_K)
        path_results["path1_bm25"] = sparse_res
        telemetry["path1_bm25_count"] = len(sparse_res)

        # 2. Path 2: BGE-M3 Dense Semantic
        dense_res = self.retrieve_dense(query, k=DENSE_K)
        path_results["path2_dense"] = dense_res
        telemetry["path2_dense_count"] = len(dense_res)

        # 3. Path 3: Metadata Heading-Summary
        matched_sections = []
        if self.metadata_retriever is not None:
            meta_res, matched_sections = self.retrieve_metadata(query, k_sections=METADATA_K_SECTIONS)
            path_results["path3_metadata"] = meta_res
            telemetry["path3_metadata_count"] = len(meta_res)
            telemetry["matched_sections"] = matched_sections

        # 4. Path 4: Generative HyDE
        hyde_passages = []
        if self.hyde_retriever is not None:
            hyde_res, hyde_passages = self.retrieve_hyde(query, k_per_hypo=HYDE_K_PER_HYPO)
            path_results["path4_hyde"] = hyde_res
            telemetry["path4_hyde_count"] = len(hyde_res)
            telemetry["hyde_passages"] = hyde_passages

        # Reciprocal Rank Fusion (verbatim implementation)
        candidates = self.reciprocal_rank_fusion(
            path_results=path_results,
            weights=None,
            k_pool=k_pool
        )

        return {
            "query": query,
            "full_path_results": path_results,
            "candidates": candidates,
            "telemetry": telemetry,
            "matched_sections": matched_sections,
            "hyde_passages": hyde_passages
        }


# -----------------------------------------------------------------------------
# Main Audit Execution Flow
# -----------------------------------------------------------------------------
def run_retrieval_path_audit():
    parser = argparse.ArgumentParser(description=AUDIT_LABEL)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Execution device for BGE-M3 embeddings (cuda or cpu)")
    args = parser.parse_args()

    print("=" * 80)
    print(f" {AUDIT_LABEL}")
    print("=" * 80)
    print(f"[Device] Running on: {args.device}")

    # 1. Baseline Integrity Hash Capture
    print("\n[Integrity] Recording baseline SHA-256 hashes of protected artifacts...")
    baseline_hashes = record_protected_hashes()
    for path, h in baseline_hashes.items():
        print(f"  - {os.path.basename(path)}: {h[:12]}...")

    start_time = time.time()
    try:
        # 2. Identify the exact 37 failed questions from Stage 1 checkpoint
        print(f"\n[Checkpoint Audit] Reading Stage 1 checkpoint: {STAGE1_CHECKPOINT_FILE}")
        checkpoint_records = {}
        stage1_miss_records = []
        with open(STAGE1_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["financebench_id"]
                checkpoint_records[qid] = rec
                if rec.get("initial_rrf", {}).get("raw_hit_rank") is None:
                    stage1_miss_records.append(rec)

        target_miss_ids = [r["financebench_id"] for r in stage1_miss_records]
        assert len(stage1_miss_records) == 37, f"Expected 37 misses in checkpoint, found {len(stage1_miss_records)}"
        print(f"[Identified] Successfully loaded exactly {len(target_miss_ids)} candidate-pool failure IDs.")

        # 3. Load question text and gold evidence metadata
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            all_questions = [json.loads(line) for line in f if line.strip()]
        
        target_questions = []
        for q in all_questions:
            qid = q.get("financebench_id")
            if qid in target_miss_ids:
                q_copy = dict(q)
                cp_entry = checkpoint_records[qid]
                q_copy["dpo_leakage_split"] = cp_entry["dpo_leakage_split"]
                q_copy["gold_target_doc"] = cp_entry["gold_doc"]
                q_copy["gold_target_pages"] = cp_entry["gold_target_pages"]
                q_copy["original_checkpoint_top5"] = cp_entry["initial_rrf"]["top5_cids"]
                target_questions.append(q_copy)

        assert len(target_questions) == 37, f"Expected 37 target questions, got {len(target_questions)}"
        print(f"[Questions Prepared] 37 target questions matched with gold doc and pages.")

        # 4. Load Chunk Store
        print(f"\n[Loading] Loading chunk store from {CHUNKS_FILE}...")
        chunk_store = {}
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunk_store[c["chunk_id"]] = c
        print(f"[Loaded] Chunk store contains {len(chunk_store)} chunks.")

        # 5. Load Cached HyDE Passages
        print(f"\n[Loading] Loading HyDE passage cache from {HYDE_CACHE_FILE}...")
        with open(HYDE_CACHE_FILE, "r", encoding="utf-8") as f:
            hyde_cache = json.load(f)
        for qid in target_miss_ids:
            assert qid in hyde_cache, f"Missing cached HyDE passages for {qid}"
            assert len(hyde_cache[qid]) == 3, f"Expected 3 passages for {qid}, got {len(hyde_cache[qid])}"
        print(f"[Loaded] Validated cached passages for all 37 target questions.")

        # 6. Load Retrieval Indexes
        print("\n[Loading] Initializing retrieval indexes...")
        # BGE-M3 Embeddings (for dense and metadata queries)
        bge_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": args.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # BM25 Retriever
        bm25 = BM25Retriever(BM25_INDEX_DIR)

        # FAISS Dense Index
        faiss_index = FAISS.load_local(
            FAISS_INDEX_DIR,
            bge_embeddings,
            allow_dangerous_deserialization=True
        )

        # Metadata Heading-Summary Retriever
        metadata_retriever = MetadataRetriever(
            metadata_index_dir=META_INDEX_DIR,
            embeddings=bge_embeddings
        )

        # Cached HyDE Retriever
        class CachedHyDEGenerator:
            def __init__(self, cache: Dict[str, List[str]]):
                self.cache = cache
                self.current_qid = None
            def generate_hypothetical_passages(self, query: str, **kwargs) -> List[str]:
                if self.current_qid and self.current_qid in self.cache:
                    return self.cache[self.current_qid]
                raise KeyError(f"CRITICAL: Missing cached HyDE passages for question ID {self.current_qid}!")

        cached_generator = CachedHyDEGenerator(hyde_cache)
        hyde_retriever = HyDERetriever(
            faiss_index=faiss_index,
            generator=cached_generator
        )

        # Instrumented MPR Coordinator
        impr = InstrumentedMultiPathRetriever(
            faiss_index=faiss_index,
            bm25_retriever=bm25,
            chunk_store=chunk_store,
            metadata_retriever=metadata_retriever,
            hyde_retriever=hyde_retriever,
            rrf_k=RRF_K
        )

        # 7. Execute Instrumented Audit Across All 37 Questions
        print("\n" + "=" * 70)
        print("[Executing] Running instrumented retrieval across 37 failure questions...")
        print("=" * 70)

        all_path_records = []
        all_rrf_pool_records = []
        hit_analysis_rows = []

        path_names = ["path1_bm25", "path2_dense", "path3_metadata", "path4_hyde"]

        # Aggregate hit counters
        path_hit_counts = defaultdict(int)
        path_first_ranks = defaultdict(list)
        union_hit_count = 0
        rrf_hit_count = 0

        for idx, q in enumerate(target_questions, start=1):
            qid = q["financebench_id"]
            query_text = q["question"]
            gold_doc = q["gold_target_doc"]
            gold_pages = set(q["gold_target_pages"])
            orig_top5 = q["original_checkpoint_top5"]

            cached_generator.current_qid = qid

            # Execute instrumented multi-path retrieval
            audit_res = impr.retrieve_instrumented(query=query_text, k_pool=K_POOL)
            full_paths = audit_res["full_path_results"]
            fused_candidates = audit_res["candidates"]

            # Top-5 verification against checkpoint
            reproduced_top5 = [c["chunk_id"] for c in fused_candidates[:5]]
            assert reproduced_top5 == orig_top5, (
                f"DIVERGENCE for {qid}!\nCheckpoint Top 5: {orig_top5}\nReproduced Top 5: {reproduced_top5}"
            )

            # Analyze individual paths
            per_path_first_hit = {}
            for p_name in path_names:
                items = full_paths.get(p_name, [])
                first_hit_rank = None
                for rank, (cid, raw_score) in enumerate(items, start=1):
                    c_data = chunk_store.get(cid, {})
                    c_doc = c_data.get("doc_name")
                    c_page = c_data.get("page_number")
                    doc_meta = get_doc_metadata(c_doc)
                    is_gold_hit = (c_doc == gold_doc and c_page in gold_pages)

                    if is_gold_hit and first_hit_rank is None:
                        first_hit_rank = rank

                    # Log detailed path candidate
                    all_path_records.append({
                        "financebench_id": qid,
                        "retrieval_path": p_name,
                        "path_rank": rank,
                        "chunk_id": cid,
                        "doc_name": c_doc,
                        "company": doc_meta["company"],
                        "filing_year": doc_meta["year"],
                        "filing_type": doc_meta["filing_type"],
                        "page_number": c_page,
                        "raw_score": round(float(raw_score), 6),
                        "is_gold_hit": is_gold_hit
                    })

                per_path_first_hit[p_name] = first_hit_rank
                if first_hit_rank is not None:
                    path_hit_counts[p_name] += 1
                    path_first_ranks[p_name].append(first_hit_rank)

            # Analyze fused RRF candidates
            rrf_first_hit_rank = None
            for rrf_rank, c in enumerate(fused_candidates, start=1):
                cid = c["chunk_id"]
                c_data = chunk_store.get(cid, {})
                c_doc = c.get("doc_name")
                c_page = c.get("page")
                doc_meta = get_doc_metadata(c_doc)
                is_gold_hit = (c_doc == gold_doc and c_page in gold_pages)

                if is_gold_hit and rrf_first_hit_rank is None:
                    rrf_first_hit_rank = rrf_rank

                contribs = c.get("path_contributions", {})
                all_rrf_pool_records.append({
                    "financebench_id": qid,
                    "rrf_rank": rrf_rank,
                    "chunk_id": cid,
                    "doc_name": c_doc,
                    "company": doc_meta["company"],
                    "filing_year": doc_meta["year"],
                    "filing_type": doc_meta["filing_type"],
                    "page_number": c_page,
                    "is_gold_hit": is_gold_hit,
                    "rrf_score": c.get("rrf_score"),
                    "path_presence": {p: p in contribs for p in path_names},
                    "bm25_rank": contribs.get("path1_bm25", {}).get("rank"),
                    "bm25_raw_score": contribs.get("path1_bm25", {}).get("raw_score"),
                    "bm25_rrf_contribution": contribs.get("path1_bm25", {}).get("rrf_contribution"),
                    "dense_rank": contribs.get("path2_dense", {}).get("rank"),
                    "dense_raw_score": contribs.get("path2_dense", {}).get("raw_score"),
                    "dense_rrf_contribution": contribs.get("path2_dense", {}).get("rrf_contribution"),
                    "metadata_rank": contribs.get("path3_metadata", {}).get("rank"),
                    "metadata_raw_score": contribs.get("path3_metadata", {}).get("raw_score"),
                    "metadata_rrf_contribution": contribs.get("path3_metadata", {}).get("rrf_contribution"),
                    "hyde_rank": contribs.get("path4_hyde", {}).get("rank"),
                    "hyde_raw_score": contribs.get("path4_hyde", {}).get("raw_score"),
                    "hyde_rrf_contribution": contribs.get("path4_hyde", {}).get("rrf_contribution"),
                })

            union_has_hit = any(v is not None for v in per_path_first_hit.values())
            if union_has_hit:
                union_hit_count += 1
            if rrf_first_hit_rank is not None:
                rrf_hit_count += 1

            hit_analysis_rows.append({
                "financebench_id": qid,
                "gold_target_doc": gold_doc,
                "gold_target_pages": ";".join(str(p) for p in sorted(list(gold_pages))),
                "dpo_split": q["dpo_leakage_split"],
                "bm25_first_hit_rank": per_path_first_hit["path1_bm25"] or "None",
                "dense_first_hit_rank": per_path_first_hit["path2_dense"] or "None",
                "metadata_first_hit_rank": per_path_first_hit["path3_metadata"] or "None",
                "hyde_first_hit_rank": per_path_first_hit["path4_hyde"] or "None",
                "union_has_gold_hit": union_has_hit,
                "rrf_top40_first_hit_rank": rrf_first_hit_rank or "None",
                "rrf_top40_has_gold_hit": rrf_first_hit_rank is not None
            })

            print(f"[{idx:02d}/37] Audited {qid} | Verified Top-5 | Union Hit: {union_has_hit} | RRF Top-40 Hit: {rrf_first_hit_rank is not None}")

        # 8. Save All Output Files in Isolated Directory
        os.makedirs(AUDIT_OUTPUT_DIR, exist_ok=True)
        print(f"\n[Saving] Writing audit outputs to {AUDIT_OUTPUT_DIR}...")

        # A. retrieval_paths_37.jsonl
        paths_file = os.path.join(AUDIT_OUTPUT_DIR, "retrieval_paths_37.jsonl")
        with open(paths_file, "w", encoding="utf-8") as f:
            for r in all_path_records:
                f.write(json.dumps(r) + "\n")
        print(f"  - Saved all path items: {paths_file} ({len(all_path_records)} entries)")

        # B. rrf_top40_37.jsonl
        rrf_file = os.path.join(AUDIT_OUTPUT_DIR, "rrf_top40_37.jsonl")
        with open(rrf_file, "w", encoding="utf-8") as f:
            for r in all_rrf_pool_records:
                f.write(json.dumps(r) + "\n")
        print(f"  - Saved fused top-40 items: {rrf_file} ({len(all_rrf_pool_records)} entries)")

        # C. gold_hit_analysis_37.csv
        csv_file = os.path.join(AUDIT_OUTPUT_DIR, "gold_hit_analysis_37.csv")
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "financebench_id", "gold_target_doc", "gold_target_pages", "dpo_split",
                "bm25_first_hit_rank", "dense_first_hit_rank", "metadata_first_hit_rank",
                "hyde_first_hit_rank", "union_has_gold_hit", "rrf_top40_first_hit_rank",
                "rrf_top40_has_gold_hit"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hit_analysis_rows)
        print(f"  - Saved gold hit analysis CSV: {csv_file}")

        # D. retrieval_path_summary.json
        summary_stats = {
            "total_questions_audited": 37,
            "path_hit_counts": dict(path_hit_counts),
            "path_hit_rates_pct": {
                p: round((path_hit_counts[p] / 37.0) * 100.0, 2) for p in path_names
            },
            "path_mean_first_hit_ranks": {
                p: round(sum(path_first_ranks[p]) / len(path_first_ranks[p]), 2) if path_first_ranks[p] else None
                for p in path_names
            },
            "path_median_first_hit_ranks": {
                p: sorted(path_first_ranks[p])[len(path_first_ranks[p]) // 2] if path_first_ranks[p] else None
                for p in path_names
            },
            "union_hit_count": union_hit_count,
            "union_hit_rate_pct": round((union_hit_count / 37.0) * 100.0, 2),
            "rrf_top40_hit_count": rrf_hit_count,
            "rrf_top40_hit_rate_pct": round((rrf_hit_count / 37.0) * 100.0, 2),
            "fusion_displaced_count": union_hit_count - rrf_hit_count,
            "zero_path_hit_count": 37 - union_hit_count
        }

        summary_file = os.path.join(AUDIT_OUTPUT_DIR, "retrieval_path_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"  - Saved aggregate summary statistics: {summary_file}")

        # E. failure_diagnosis_37.md
        diag_file = os.path.join(AUDIT_OUTPUT_DIR, "failure_diagnosis_37.md")
        with open(diag_file, "w", encoding="utf-8") as f:
            f.write(f"# {AUDIT_LABEL}\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
            f.write(f"**Audited Questions:** 37 Stage 1 candidate-pool failures.\n\n")
            
            f.write("## 1. Aggregate Path Recall & Fusion Summary\n\n")
            f.write("| Retrieval Path | Questions with Gold Hit | Hit Rate (%) | Mean Hit Rank | Median Hit Rank |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for p in path_names:
                count = path_hit_counts[p]
                rate = summary_stats["path_hit_rates_pct"][p]
                m_rank = summary_stats["path_mean_first_hit_ranks"][p]
                med_rank = summary_stats["path_median_first_hit_ranks"][p]
                f.write(f"| `{p}` | {count}/37 | {rate}% | {m_rank if m_rank else 'N/A'} | {med_rank if med_rank else 'N/A'} |\n")
            f.write(f"| **Union of All 4 Paths** | **{union_hit_count}/37** | **{summary_stats['union_hit_rate_pct']}%** | - | - |\n")
            f.write(f"| **Fused RRF Top-40** | **{rrf_hit_count}/37** | **{summary_stats['rrf_top40_hit_rate_pct']}%** | - | - |\n\n")

            f.write("## 2. Key Forensic Insights\n\n")
            f.write(f"- **Zero-Path Retrieval Failures:** In {summary_stats['zero_path_hit_count']} of 37 cases, none of the 4 paths retrieved gold evidence within their configured retrieval depth.\n")
            f.write(f"- **Fusion Displacement:** In {summary_stats['fusion_displaced_count']} of 37 cases, gold evidence was retrieved by at least one individual path but was displaced outside the top-40 by RRF.\n\n")

            f.write("## 3. Question-Level Retrieval Matrix\n\n")
            f.write("| Question ID | Target Doc | Pages | BM25 Hit | Dense Hit | Meta Hit | HyDE Hit | Union Hit | RRF Hit |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for row in hit_analysis_rows:
                f.write(f"| `{row['financebench_id']}` | `{row['gold_target_doc']}` | {row['gold_target_pages']} | {row['bm25_first_hit_rank']} | {row['dense_first_hit_rank']} | {row['metadata_first_hit_rank']} | {row['hyde_first_hit_rank']} | {row['union_has_gold_hit']} | {row['rrf_top40_has_gold_hit']} |\n")

        print(f"  - Saved Markdown diagnosis report: {diag_file}")

        # F. integrity_manifest.json
        manifest_file = os.path.join(AUDIT_OUTPUT_DIR, "integrity_manifest.json")
        manifest_data = {
            "audit_label": AUDIT_LABEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - start_time, 2),
            "configuration": {
                "rrf_k": RRF_K,
                "k_pool": K_POOL,
                "bm25_k": BM25_K,
                "dense_k": DENSE_K,
                "metadata_k_sections": METADATA_K_SECTIONS,
                "hyde_k_per_hypo": HYDE_K_PER_HYPO,
                "target_question_count": 37
            },
            "protected_hashes_baseline": baseline_hashes,
            "protected_hashes_post": record_protected_hashes()
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"  - Saved integrity manifest: {manifest_file}")

        print("\n" + "=" * 70)
        print(f"[Complete] Instrumented audit finished in {time.time() - start_time:.2f} seconds.")
        print("=" * 70)

    finally:
        # Final Verification of Protected Hashes
        print("\n[Final Integrity Audit] Verifying zero protected files were modified...")
        verify_protected_hashes(baseline_hashes)


if __name__ == "__main__":
    run_retrieval_path_audit()
