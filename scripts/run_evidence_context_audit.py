import json
from pathlib import Path
from collections import defaultdict

v2_path = Path("outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl")
audit_path = Path("outputs/experiments/experiment_l5_ground_truth_v2/ground_truth_v2_audit.json")

# Load Grounded Corpus to check chunks and adjacent chunks
grounded_corpus_path = Path("data/heldout_stage3/processed/heldout_grounded_chunks.jsonl")

print("Loading Grounded Corpus chunks...")
chunks_by_doc = defaultdict(dict)
all_chunks = {}
with open(grounded_corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        cid = c["chunk_id"]
        doc = c["doc_name"]
        page = c.get("page", 0)
        chunks_by_doc[doc][cid] = c
        all_chunks[cid] = c

print(f"Loaded {len(all_chunks)} chunks across {len(chunks_by_doc)} documents.")

# Load V2 Questions
v2_questions = []
with open(v2_path, "r", encoding="utf-8") as f:
    for line in f:
        v2_questions.append(json.loads(line))

verified_questions = [q for q in v2_questions if q.get("verification_metadata", {}).get("benchmark_v2_status") == "VERIFIED"]
print(f"Verified questions count: {len(verified_questions)}")

audit_results = []

for q in verified_questions:
    qid = q["question_id"]
    doc = q["doc_name"]
    qtype = q.get("question_type")
    orig_class = q.get("verification_metadata", {}).get("original_annotation_classification")
    ev_sources = q.get("evidence_sources", [])
    
    direct_ev = [e for e in ev_sources if e.get("evidence_type") == "direct_evidence"]
    context_ev = [e for e in ev_sources if e.get("evidence_type") == "contextual_table_header"]
    hist_ev = [e for e in ev_sources if e.get("evidence_type") == "historical_annotated_reference"]
    
    direct_pages = sorted(list(set(e.get("evidence_page_num") for e in direct_ev)))
    all_pages = sorted(list(set(e.get("evidence_page_num") for e in ev_sources)))
    direct_cids = [cid for e in direct_ev for cid in e.get("matching_chunk_ids", [])]
    
    # Check if direct evidence chunks are adjacent
    # Extract chunk indices from chunk_ids, e.g., AAPL_2024_10K_p51_c0 -> page 51, c0
    chunks_info = []
    for cid in direct_cids:
        chunk = all_chunks.get(cid)
        if chunk:
            chunks_info.append({
                "chunk_id": cid,
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
                "is_table": chunk.get("is_table", False),
                "section": chunk.get("section_heading", ""),
                "text_len": len(chunk.get("text", "")),
            })
    
    # Analyze adjacency and context needs
    # Case A: Multi-chunk evidence on the same page (e.g. c0 and c1)
    # Case B: Multi-page evidence (e.g. p93 and p94)
    # Case C: Contextual header separated from table body (e.g. p10 header and p11 body)
    has_same_page_multi_chunk = len(direct_cids) > 1 and len(direct_pages) == 1
    has_multi_page = len(direct_pages) > 1
    has_separated_header = len(context_ev) > 0
    
    audit_results.append({
        "question_id": qid,
        "doc_name": doc,
        "question": q["question"],
        "question_type": qtype,
        "orig_classification": orig_class,
        "direct_pages": direct_pages,
        "all_pages": all_pages,
        "direct_cids": direct_cids,
        "has_same_page_multi_chunk": has_same_page_multi_chunk,
        "has_multi_page": has_multi_page,
        "has_separated_header": has_separated_header,
        "chunks_info": chunks_info,
    })

# Print summary breakdown
same_page_multi = sum(1 for r in audit_results if r["has_same_page_multi_chunk"])
multi_page = sum(1 for r in audit_results if r["has_multi_page"])
sep_header = sum(1 for r in audit_results if r["has_separated_header"])
single_chunk = sum(1 for r in audit_results if len(r["direct_cids"]) == 1 and not r["has_separated_header"])

print("\n--- EVIDENCE-CONTEXT AUDIT BREAKDOWN (N=65) ---")
print(f"Total Verified Questions: {len(audit_results)}")
print(f"1. Single standalone chunk (isolated evidence): {single_chunk} ({single_chunk/65*100:.1f}%)")
print(f"2. Same-page multi-chunk evidence (adjacent chunks on same page): {same_page_multi} ({same_page_multi/65*100:.1f}%)")
print(f"3. Multi-page direct evidence (spans page boundary): {multi_page} ({multi_page/65*100:.1f}%)")
print(f"4. Separated contextual header + body (header on prior page/chunk): {sep_header} ({sep_header/65*100:.1f}%)")

print("\n--- SAMPLE QUESTIONS IDENTIFIED AS CANDIDATES FOR STITCHING ---")
for r in audit_results:
    if r["has_same_page_multi_chunk"] or r["has_multi_page"] or r["has_separated_header"]:
        print(f"QID: {r['question_id']} | Doc: {r['doc_name']} | Type: {r['question_type']} | Class: {r['orig_classification']}")
        print(f"  Pages: direct={r['direct_pages']}, all={r['all_pages']} | Direct chunks: {r['direct_cids']}")
        if r["has_separated_header"]:
            print("  -> Has separated table/context header")
        if r["has_same_page_multi_chunk"]:
            print("  -> Multi-chunk on same page")
        if r["has_multi_page"]:
            print("  -> Multi-page cross-boundary evidence")
