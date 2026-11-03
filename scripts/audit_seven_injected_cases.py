"""
scripts/audit_seven_injected_cases.py
=====================================
Detailed post-evaluation correctness audit of the 7 statement-injection cases.
Compares phase2_retrieval_checkpoint_reference_exp_f.jsonl vs
phase2_retrieval_checkpoint_candidate_h2_1.jsonl.
"""

import os
import json
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation")

REF_CHK = os.path.join(EXP_DIR, "phase2_retrieval_checkpoint_reference_exp_f.jsonl")
H21_CHK = os.path.join(EXP_DIR, "phase2_retrieval_checkpoint_candidate_h2_1.jsonl")
STMT_AUDIT = os.path.join(EXP_DIR, "statement_selection_audit.csv")
FILING_AUDIT = os.path.join(EXP_DIR, "filing_selection_audit.csv")
QUESTIONS_FILE = os.path.join(EXP_DIR, "heldout_questions.jsonl")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")

def main():
    with open(REF_CHK, "r", encoding="utf-8") as f:
        ref_records = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    with open(H21_CHK, "r", encoding="utf-8") as f:
        h21_records = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    with open(STMT_AUDIT, "r", encoding="utf-8") as f:
        stmt_audit = {r["question_id"]: r for r in csv.DictReader(f)}

    with open(FILING_AUDIT, "r", encoding="utf-8") as f:
        filing_audit = {r["question_id"]: r for r in csv.DictReader(f)}

    chunk_store = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c

    injected_qids = [
        qid for qid, r in stmt_audit.items() if r["injected_chunk_ids"] != "NONE"
    ]
    print(f"Total Injected Questions: {len(injected_qids)}")
    print(f"Injected QIDs: {injected_qids}\n")

    audit_results = []

    for qid in injected_qids:
        r_f = ref_records[qid]
        r_h = h21_records[qid]
        q_meta = questions[qid]
        s_meta = stmt_audit[qid]
        f_meta = filing_audit[qid]

        pool_f = r_f["pool_chunk_ids"]
        pool_h = r_h["pool_chunk_ids"]

        set_f = set(pool_f)
        set_h = set(pool_h)

        intersection = set_f.intersection(set_h)
        union = set_f.union(set_h)
        jaccard = len(intersection) / len(union) if union else 1.0
        overlap_count = len(intersection)

        injected_chunks = r_h["injected_chunk_ids"]
        displaced_chunks = r_h["displaced_chunk_ids"]

        # Verify whether injected chunks actually entered pool_h
        injected_in_pool_h = [cid in set_h for cid in injected_chunks]
        all_injected_in_h = all(injected_in_pool_h)

        # Check gold target pages
        target_pages = r_f["target_pages"]
        gold_doc = r_f["doc_name"]

        # Check where gold chunk was
        gold_in_pool_f = r_f["best_rank_in_pool"]
        gold_in_pool_h = r_h["best_rank_in_pool"]
        gold_rank_f = r_f["best_rank"]
        gold_rank_h = r_h["best_rank"]

        # Top 10 chunks
        top10_f = [c["chunk_id"] for c in r_f.get("top10_retrieved", [])] # wait, checkpoint has top3_retrieved
        # Let's check what fields checkpoint has
        # Let's inspect fields of r_f

        res = {
            "question_id": qid,
            "company": r_f["company"],
            "question": q_meta["question"],
            "target_statement_or_section": q_meta.get("target_statement_or_section"),
            "detected_intent": s_meta["detected_intent"],
            "selected_filing": f_meta["selected_filing"],
            "target_filing": f_meta["target_filing"],
            "primary_statement_pages": s_meta["primary_pages_selected"],
            "gold_evidence_pages": target_pages,
            "injected_chunk_ids": injected_chunks,
            "displaced_chunk_ids": displaced_chunks,
            "all_injected_in_h21_pool": all_injected_in_h,
            "pool_f_size": len(pool_f),
            "pool_h_size": len(pool_h),
            "pool_overlap_count": overlap_count,
            "jaccard_similarity": round(jaccard, 4),
            "gold_doc": gold_doc,
            "gold_in_pool_f": gold_in_pool_f,
            "gold_in_pool_h": gold_in_pool_h,
            "gold_final_rank_f": gold_rank_f,
            "gold_final_rank_h": gold_rank_h,
            "rr_f": r_f["rr"],
            "rr_h": r_h["rr"],
            "top3_f": r_f.get("top3_retrieved"),
            "top3_h": r_h.get("top3_retrieved"),
            "injected_chunk_details": [
                {
                    "cid": cid,
                    "doc": chunk_store.get(cid, {}).get("doc_name"),
                    "page": chunk_store.get(cid, {}).get("page_number"),
                    "text_preview": chunk_store.get(cid, {}).get("text", "")[:120].replace("\n", " ")
                }
                for cid in injected_chunks
            ]
        }
        audit_results.append(res)

    print("Detailed Audit of 7 Injected Cases:\n")
    for r in audit_results:
        print(f"[{r['question_id']}] {r['company']}")
        print(f"  Question:            {r['question']}")
        print(f"  Target Concept Sec:  {r['target_statement_or_section']}")
        print(f"  Detected Intent:     {r['detected_intent']}")
        print(f"  Selected Filing:     {r['selected_filing']} (Target: {r['gold_doc']})")
        print(f"  Selected Stmt Page:  {r['primary_statement_pages']} vs Gold Target Page: {r['gold_evidence_pages']}")
        print(f"  Injected Chunks:     {r['injected_chunk_ids']}")
        print(f"  Injected In H2.1:    {r['all_injected_in_h21_pool']} (Pool size: F={r['pool_f_size']}, H={r['pool_h_size']})")
        print(f"  Pool Overlap:        {r['pool_overlap_count']}/50 | Jaccard: {r['jaccard_similarity']}")
        print(f"  Displaced Chunks:    {r['displaced_chunk_ids']}")
        print(f"  Gold Rank In Pool:   Exp F={r['gold_in_pool_f']} | H2.1={r['gold_in_pool_h']}")
        print(f"  Gold Final Rank:     Exp F={r['gold_final_rank_f']} | H2.1={r['gold_final_rank_h']}")
        print(f"  Top 3 Chunks F:      {r['top3_f']}")
        print(f"  Top 3 Chunks H:      {r['top3_h']}")
        for cd in r["injected_chunk_details"]:
            print(f"    Chunk detail: {cd['cid']} (page {cd['page']}): {cd['text_preview']}...")
        print("-" * 80)

if __name__ == "__main__":
    main()
