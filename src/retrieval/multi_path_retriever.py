"""
FinSage Multi-Path Retriever (MPR)
==================================
Paper Specifications: FinSage (arXiv:2504.14493, CIKM 2025) Section 3.2 & Section 4.2.1.
Phase 4 Enhancements:
- Path 1: BM25 Sparse Lexical Retrieval (Unicode NFKC normalized)
- Path 2: BGE-M3 Dense FAISS Retrieval on Chunk Embeddings
- Path 3: Heading-Summary Metadata FAISS Retrieval
- Path 4: Generative HyDE
- Reciprocal Rank Fusion (RRF, K_pool=40)
- Multi-Statement Evidence Retrieval: independent component retrieval for ratio / multi-metric questions
"""

import os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.chunk_bundler import ChunkBundler
from src.retrieval.financial_expander import FinancialQueryExpander, normalize_unicode


class MultiPathRetriever:
    """
    Paper-faithful 4-Path Multi-Path Retriever (MPR) for FinSage with Phase 4 Intelligence.
    """

    def __init__(
        self,
        faiss_index,
        bm25_retriever,
        chunk_store: Dict[str, Dict[str, Any]],
        metadata_retriever: Optional[MetadataRetriever] = None,
        hyde_retriever: Optional[HyDERetriever] = None,
        chunk_bundler: Optional[ChunkBundler] = None,
        rrf_k: int = 60,
        expander: Optional[FinancialQueryExpander] = None,
    ):
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.chunk_store = chunk_store
        self.metadata_retriever = metadata_retriever
        self.hyde_retriever = hyde_retriever
        self.chunk_bundler = chunk_bundler
        self.rrf_k = rrf_k
        self.expander = expander or FinancialQueryExpander()

    def retrieve_sparse(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Path 1: BM25 Sparse Retrieval with NFKC normalization."""
        try:
            norm_q = normalize_unicode(query)
            ids, scores = self.bm25_retriever.invoke(norm_q, k=k)
            scored = []
            for item_id, score in zip(ids, scores):
                c_data = self.chunk_store.get(item_id)
                if c_data and "chunk_id" in c_data:
                    scored.append((c_data["chunk_id"], float(score)))
                elif isinstance(item_id, str):
                    scored.append((item_id, float(score)))
            return scored
        except Exception as e:
            print(f"[MPR] Error in BM25 retrieval: {e}")
            return []

    def retrieve_dense(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Path 2: Dense FAISS Retrieval with NFKC normalization."""
        try:
            norm_q = normalize_unicode(query)
            results = self.faiss_index.similarity_search_with_score(norm_q, k=k)
            scored = []
            for doc, score in results:
                cid = doc.metadata.get("chunk_id")
                if cid:
                    scored.append((cid, float(score)))
            return scored
        except Exception as e:
            print(f"[MPR] Error in Dense retrieval: {e}")
            return []

    def retrieve_metadata(self, query: str, k_sections: int = 4) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
        """Path 3: Metadata Heading-Summary Retrieval with NFKC normalization."""
        if self.metadata_retriever is None:
            return [], []
        try:
            norm_q = normalize_unicode(query)
            return self.metadata_retriever.retrieve_metadata_candidates(norm_q, k_sections=k_sections)
        except Exception as e:
            print(f"[MPR] Error in Metadata retrieval: {e}")
            return [], []

    def retrieve_hyde(self, query: str, k_per_hypo: int = 20) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Path 4: Generative HyDE Retrieval with NFKC normalization."""
        if self.hyde_retriever is None:
            return [], []
        try:
            norm_q = normalize_unicode(query)
            return self.hyde_retriever.retrieve_hyde_candidates(norm_q, k_per_hypo=k_per_hypo)
        except Exception as e:
            print(f"[MPR] Error in HyDE retrieval: {e}")
            return [], []

    def reciprocal_rank_fusion(
        self,
        path_results: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None,
        k_pool: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Merges ranked lists from retrieval paths using Reciprocal Rank Fusion (RRF).
        RRF Score = Sum( w_p / (k_rrf + rank_p) )
        """
        if weights is None:
            weights = {p: 1.0 for p in path_results}

        rrf_scores = defaultdict(float)
        path_contributions = defaultdict(dict)

        for path_name, items in path_results.items():
            w = weights.get(path_name, 1.0)
            for rank, (cid, raw_score) in enumerate(items, start=1):
                inc = w / (self.rrf_k + rank)
                rrf_scores[cid] += inc
                path_contributions[cid][path_name] = {
                    "rank": rank,
                    "raw_score": round(raw_score, 4),
                    "rrf_contribution": round(inc, 6)
                }

        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_pool]

        candidates = []
        for cid, rrf_score in sorted_cids:
            c_data = self.chunk_store.get(cid, {})
            candidates.append({
                "chunk_id": cid,
                "text": c_data.get("text", ""),
                "doc_name": c_data.get("doc_name"),
                "page": c_data.get("page_number", 0),
                "rrf_score": round(rrf_score, 6),
                "path_contributions": path_contributions[cid],
            })
        return candidates

    def retrieve_multi_path(
        self,
        query: str,
        k_pool: int = 40,
        enable_bm25: bool = True,
        enable_dense: bool = True,
        enable_metadata: bool = True,
        enable_hyde: bool = True,
        path_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Multi-Path Retrieval across active paths and fuses candidates with RRF.
        Applies Unicode NFKC normalization and multi-statement evidence component retrieval.
        """
        norm_query = normalize_unicode(query)
        exp_res = self.expander.expand(norm_query)

        path_results = {}
        telemetry = {}

        # 1. Path 1: BM25 Sparse
        if enable_bm25:
            bm25_target = exp_res.bm25_query if exp_res.bm25_query else norm_query
            sparse_res = self.retrieve_sparse(bm25_target, k=k_pool)
            path_results["path1_bm25"] = sparse_res
            telemetry["path1_bm25_count"] = len(sparse_res)

        # 2. Path 2: BGE-M3 Dense
        if enable_dense:
            dense_target = exp_res.dense_query if exp_res.dense_query else norm_query
            dense_res = self.retrieve_dense(dense_target, k=k_pool)
            path_results["path2_dense"] = dense_res
            telemetry["path2_dense_count"] = len(dense_res)

        # 3. Path 3: Metadata Heading-Summary
        matched_sections = []
        if enable_metadata and self.metadata_retriever is not None:
            meta_res, matched_sections = self.retrieve_metadata(norm_query, k_sections=4)
            path_results["path3_metadata"] = meta_res
            telemetry["path3_metadata_count"] = len(meta_res)
            telemetry["matched_sections"] = matched_sections

        # 4. Path 4: Generative HyDE
        hyde_passages = []
        if enable_hyde and self.hyde_retriever is not None:
            hyde_res, hyde_passages = self.retrieve_hyde(norm_query, k_per_hypo=20)
            path_results["path4_hyde"] = hyde_res
            telemetry["path4_hyde_count"] = len(hyde_res)
            telemetry["hyde_passages"] = hyde_passages

        # Improvement 4: Multi-Statement Evidence Retrieval
        if exp_res.is_multi_statement and exp_res.multi_statement_components:
            comp_count = 0
            for c_idx, comp in enumerate(exp_res.multi_statement_components, start=1):
                stmt_type = comp["statement"]
                stmt_cue = comp["cue"]
                comp_query = f"{exp_res.detected_company or ''} {stmt_cue} {' '.join(exp_res.detected_periods)}".strip()
                if enable_bm25:
                    comp_sparse = self.retrieve_sparse(comp_query, k=k_pool // 2)
                    path_results[f"path_multi_{stmt_type}_bm25"] = comp_sparse
                    comp_count += len(comp_sparse)
                if enable_dense:
                    comp_dense = self.retrieve_dense(comp_query, k=k_pool // 2)
                    path_results[f"path_multi_{stmt_type}_dense"] = comp_dense
                    comp_count += len(comp_dense)
            telemetry["multi_statement_candidates_added"] = comp_count

        # Reciprocal Rank Fusion
        candidates = self.reciprocal_rank_fusion(
            path_results=path_results,
            weights=path_weights,
            k_pool=k_pool
        )

        return {
            "query": query,
            "normalized_query": norm_query,
            "expansion_result": exp_res,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "telemetry": telemetry,
            "path_results": {p: [c[0] for c in items[:10]] for p, items in path_results.items()}
        }
