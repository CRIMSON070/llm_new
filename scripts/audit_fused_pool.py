import os
import sys
import json
import torch
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.bm25Retriever import BM25Retriever
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.multi_path_retriever import MultiPathRetriever

def main():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")
    meta_faiss_dir = os.path.join(data_dir, "metadata_faiss_index")
    enriched_chunks_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")

    chunk_store = {}
    with open(enriched_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_store[c["chunk_id"]] = c

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    dense_faiss = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    meta_retriever = MetadataRetriever(meta_faiss_dir, embeddings=embeddings)
    bm25 = BM25Retriever(bm25_phase2_dir, load_corpus=True)

    mpr = MultiPathRetriever(
        faiss_index=dense_faiss,
        bm25_retriever=bm25,
        chunk_store=chunk_store,
        metadata_retriever=meta_retriever,
        rrf_k=60
    )

    with open(os.path.join(data_dir, "questions.jsonl"), "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # Load Category C questions from Experiment D
    import csv
    cat_c_qids = set()
    with open("outputs/experiments/experiment_D_failure_analysis/question_failure_analysis.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Category C" in row["failure_category"]:
                cat_c_qids.add(row["question_id"])

    print(f"Total Category C questions to audit: {len(cat_c_qids)}")

    # For each Category C question, run MPR paths (k=50) and check all fused candidates before [:50] cutoff
    in_all_fused = 0
    in_sparse_50 = 0
    in_dense_50 = 0
    in_meta = 0

    results_table = []

    for q in questions:
        qid = q.get("financebench_id")
        if qid not in cat_c_qids:
            continue

        q_text = q["question"]
        gold_doc = q["doc_name"]
        target_pages = [e.get("evidence_page_num") + 1 for e in q.get("evidence", []) if e.get("evidence_page_num") is not None]

        sparse_res = mpr.retrieve_sparse(q_text, k=50)
        dense_res = mpr.retrieve_dense(q_text, k=50)
        meta_res, _ = mpr.retrieve_metadata(q_text, k_sections=3)

        path_dict = {
            "path1_bm25": sparse_res,
            "path2_dense": dense_res,
            "path3_metadata": meta_res
        }

        # Check in sparse
        s_hit = any(chunk_store.get(cid, {}).get("doc_name") == gold_doc and chunk_store.get(cid, {}).get("page_number") in target_pages for cid, score in sparse_res)
        d_hit = any(chunk_store.get(cid, {}).get("doc_name") == gold_doc and chunk_store.get(cid, {}).get("page_number") in target_pages for cid, score in dense_res)
        m_hit = any(chunk_store.get(cid, {}).get("doc_name") == gold_doc and chunk_store.get(cid, {}).get("page_number") in target_pages for cid, score in meta_res)

        if s_hit: in_sparse_50 += 1
        if d_hit: in_dense_50 += 1
        if m_hit: in_meta += 1

        # Calculate RRF across all items without [:50] truncation
        rrf_scores = defaultdict(float)
        for p_name, items in path_dict.items():
            for rank, (cid, score) in enumerate(items, start=1):
                rrf_scores[cid] += 1.0 / (60 + rank)

        sorted_all = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        gold_fused_rank = None
        for r, (cid, score) in enumerate(sorted_all, start=1):
            c = chunk_store.get(cid, {})
            if c.get("doc_name") == gold_doc and c.get("page_number") in target_pages:
                gold_fused_rank = r
                break

        if gold_fused_rank is not None:
            in_all_fused += 1

        results_table.append({
            "qid": qid,
            "gold_doc": gold_doc,
            "target_pages": target_pages,
            "sparse_hit": s_hit,
            "dense_hit": d_hit,
            "meta_hit": m_hit,
            "gold_fused_rank": gold_fused_rank,
            "total_fused_unique": len(sorted_all)
        })

    print(f"\n--- AUDIT OF 47 CATEGORY C QUESTIONS AT RETRIEVAL DEPTH k=50 ---")
    print(f"Total Category C: {len(cat_c_qids)}")
    print(f"Hits in sparse_50: {in_sparse_50}")
    print(f"Hits in dense_50:  {in_dense_50}")
    print(f"Hits in meta:      {in_meta}")
    print(f"Hits in full fused pool (before [:50] cutoff): {in_all_fused}")

    print("\nQuestions present in full fused pool (>50):")
    for r in results_table:
        if r["gold_fused_rank"] is not None:
            print(f"  {r['qid']} | Doc: {r['gold_doc']} | Fused Rank: {r['gold_fused_rank']} / {r['total_fused_unique']} | S={r['sparse_hit']}, D={r['dense_hit']}")

if __name__ == "__main__":
    main()
