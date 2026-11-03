"""
scripts/diagnose_70_heldout_failures.py
=======================================
Deep empirical analysis of the 70 questions that missed the top-50 candidate pool
in the Stage 3 held-out evaluation.
"""

import os
import sys
import json
import csv
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
BM25_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "bm25")
FAISS_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "indexes", "faiss_bge_m3")
QUESTIONS_FILE = os.path.join(OUT_DIR, "heldout_questions.jsonl")
COMP_CSV = os.path.join(OUT_DIR, "heldout_question_level_comparison.csv")
REF_CHK = os.path.join(OUT_DIR, "phase2_retrieval_checkpoint_reference_exp_f.jsonl")

def main():
    print("=" * 80)
    print("DIAGNOSTIC AUDIT OF 70 CANDIDATE-POOL FAILURES")
    print("=" * 80)

    # 1. Load questions and identify the 70 failures
    with open(COMP_CSV, "r", encoding="utf-8") as f:
        comps = list(csv.DictReader(f))

    failed_qids = [c["question_id"] for c in comps if c["exp_f_pool_hit"] == "0"]
    print(f"Total candidate-pool failures: {len(failed_qids)} / {len(comps)}")
    assert len(failed_qids) == 70

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    with open(REF_CHK, "r", encoding="utf-8") as f:
        ref_records = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    # 2. Load chunk store
    chunk_store = {}
    doc_page_to_cids = defaultdict(list)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c
            doc_page_to_cids[(c["doc_name"], c["page_number"])].append(c["chunk_id"])

    print(f"Loaded {len(chunk_store)} chunks.")

    # 3. Load retrieval indexes for deeper probe (check k=100 or k=200 ranks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading indexes on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    dense_faiss = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    bm25 = BM25Retriever(BM25_DIR, load_corpus=True)

    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25,
        chunk_store=chunk_store,
        metadata_retriever=None,
        rrf_k=60
    )

    # 4. Analyze each failure
    failure_details = []
    
    # Classify evidence section based on target_statement_or_section and relevant_passage
    def classify_section(sec_name: str, passage: str, qtype: str) -> str:
        s_lower = sec_name.lower()
        p_lower = passage.lower()
        
        if any(k in s_lower for k in ["consolidated balance sheet", "consolidated statement of operations", "consolidated statement of income", "consolidated statement of cash flow", "statements of operations", "balance sheet", "statements of cash flow"]):
            # Check if it's primary or footnote
            if "note" in s_lower or "footnote" in s_lower:
                return "footnote_disclosure"
            return "primary_financial_statement"
        elif "note" in s_lower or "footnote" in s_lower or "notes to" in s_lower:
            if any(k in s_lower for k in ["debt", "credit facility", "borrowing"]):
                return "debt_and_borrowing_footnote"
            elif any(k in s_lower for k in ["lease"]):
                return "lease_disclosure"
            elif any(k in s_lower for k in ["segment", "geographic", "product revenue"]):
                return "segment_or_product_disclosure"
            elif any(k in s_lower for k in ["income tax", "tax"]):
                return "tax_disclosure"
            elif any(k in s_lower for k in ["inventory", "inventories"]):
                return "inventory_disclosure"
            return "financial_statement_footnote"
        elif any(k in s_lower for k in ["item 7", "management's discussion", "md&a"]):
            if any(k in s_lower for k in ["segment", "product", "sales by"]):
                return "mda_segment_table"
            return "mda_narrative_and_tables"
        elif any(k in s_lower for k in ["item 1a", "risk factor"]):
            return "risk_factors_item_1a"
        elif any(k in s_lower for k in ["item 1.", "item 1 -", "business"]):
            return "business_description_item_1"
        else:
            return "other_or_uncertain"

    for idx, qid in enumerate(failed_qids, start=1):
        q = questions[qid]
        q_text = q["question"]
        doc_name = q["doc_name"]
        ev = q["evidence_sources"][0]
        page = ev["evidence_page_num"]
        sec = q.get("target_statement_or_section", "")
        passage = ev.get("relevant_passage", "")
        qtype = q.get("question_type", "")
        comp = q.get("company", "")
        sector = q.get("gics_sector", "")

        sec_category = classify_section(sec, passage, qtype)

        gold_cids = doc_page_to_cids.get((doc_name, page), [])

        # Probe retrieval depth: k=100
        sparse_res = mpr.retrieve_sparse(q_text, k=100)
        dense_res = mpr.retrieve_dense(q_text, k=100)

        bm25_rank = 999
        for r, (cid, sc) in enumerate(sparse_res, start=1):
            if cid in gold_cids:
                bm25_rank = r
                break

        dense_rank = 999
        for r, (cid, sc) in enumerate(dense_res, start=1):
            if cid in gold_cids:
                dense_rank = r
                break

        # Check root cause
        if bm25_rank > 100 and dense_rank > 100:
            root_cause = "GENERATION_FAILURE (Gold chunk not retrieved by BM25 or Dense in top 100)"
        elif (bm25_rank <= 50 or dense_rank <= 50):
            root_cause = "FUSION_CUTOFF (Present in top-50 of individual path, but displaced by cross-company/prioritization in RRF)"
        elif (bm25_rank <= 100 or dense_rank <= 100):
            root_cause = "RETRIEVAL_DEPTH_INSUFFICIENT (Present at rank 51-100 in BM25 or Dense; missed by k=50)"
        else:
            root_cause = "OTHER_FAILURE"

        failure_details.append({
            "question_id": qid,
            "company": comp,
            "sector": sector,
            "question_type": qtype,
            "doc_name": doc_name,
            "gold_page": page,
            "target_statement_or_section": sec,
            "evidence_category": sec_category,
            "gold_chunk_ids": gold_cids,
            "bm25_rank_in_100": bm25_rank,
            "dense_rank_in_100": dense_rank,
            "root_cause": root_cause,
            "question": q_text[:80]
        })

    print("\nEvidence Category Taxonomy across 70 Failures:")
    cat_counts = Counter(f["evidence_category"] for f in failure_details)
    for cat, cnt in cat_counts.most_common():
        print(f"  • {cat:<32}: {cnt:>2} ({cnt/70*100:5.1f}%)")

    print("\nRoot Cause Breakdown across 70 Failures:")
    cause_counts = Counter(f["root_cause"] for f in failure_details)
    for cause, cnt in cause_counts.most_common():
        print(f"  • {cause}: {cnt:>2} ({cnt/70*100:5.1f}%)")

    print("\nSector Breakdown across 70 Failures:")
    sec_counts = Counter(f["sector"] for f in failure_details)
    for sec, cnt in sec_counts.most_common():
        print(f"  • {sec:<24}: {cnt:>2} ({cnt/70*100:5.1f}%)")

    # Save detailed CSV
    diag_csv = os.path.join(OUT_DIR, "failure_70_questions_diagnostic.csv")
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(failure_details[0].keys()))
        writer.writeheader()
        writer.writerows(failure_details)
    print(f"\nSaved full diagnostic log to: {diag_csv}")

if __name__ == "__main__":
    main()
