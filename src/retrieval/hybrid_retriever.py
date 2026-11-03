"""
Hybrid Retriever with Reciprocal Rank Fusion & Statement-Aware Pipeline.

Combines BGE-M3 Dense FAISS Retrieval, BM25 Sparse Retrieval,
Financial Query Expansion, and Statement-Aware Cross-Encoder Reranking.
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.retrieval.financial_expander import FinancialQueryExpander, FinancialExpansionResult
from src.retrieval.statement_reranker import StatementAwareReranker

class HybridRetriever:
    """
    Orchestrates dense, sparse, and expanded retrieval paths with RRF fusion
    and statement-aware reranking.
    Supports both baseline execution and Phase 3 financial retrieval intelligence.
    """

    def __init__(
        self,
        faiss_index,
        bm25_retriever,
        chunk_store: Dict[Any, Dict[str, Any]],
        expander: Optional[FinancialQueryExpander] = None,
        reranker: Optional[StatementAwareReranker] = None,
        rrf_k: int = 60,
        diagnostics_dir: Optional[str] = None,
    ):
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.chunk_store = chunk_store
        self.expander = expander or FinancialQueryExpander()
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.diagnostics_dir = diagnostics_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs",
            "phase3_retrieval_diagnostics"
        )

    def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.chunk_store.get(chunk_id)

    def retrieve_dense(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Perform dense retrieval on FAISS index."""
        try:
            results = self.faiss_index.similarity_search_with_score(query, k=k)
            scored = []
            for doc, score in results:
                cid = doc.metadata.get("chunk_id")
                if cid:
                    scored.append((cid, float(score)))
            return scored
        except Exception as e:
            print(f"[HybridRetriever] Error in dense retrieval: {e}")
            return []

    def retrieve_sparse(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Perform sparse retrieval on BM25 index."""
        try:
            ids, scores = self.bm25_retriever.invoke(query, k=k)
            scored = []
            for int_id, score in zip(ids, scores):
                c_data = self.chunk_store.get(int_id)
                if c_data and "chunk_id" in c_data:
                    scored.append((c_data["chunk_id"], float(score)))
            return scored
        except Exception as e:
            print(f"[HybridRetriever] Error in sparse retrieval: {e}")
            return []

    def filter_candidates_by_metadata(
        self,
        candidates: List[Dict[str, Any]],
        detected_company: Optional[str] = None,
        detected_periods: Optional[List[str]] = None,
        min_candidates_threshold: int = 5,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Safely filter candidates using explicit query metadata (Task 2).
        Rules:
        1. Filter only when attribute is explicit.
        2. Retain candidate if metadata is missing or ambiguous.
        3. Never silently remove all candidates: if filtered pool has < min_candidates_threshold,
           fall back to unfiltered pool to avoid false drops.
        4. Record exact trace of decisions.
        """
        trace = {
            "applied": False,
            "detected_company": detected_company,
            "detected_periods": detected_periods,
            "unfiltered_count": len(candidates),
            "filtered_count": len(candidates),
            "fallback_triggered": False,
            "reason": "no_explicit_attribute"
        }

        if not detected_company and not detected_periods:
            return candidates, trace

        filtered = []
        for c in candidates:
            doc_name = (c.get("doc_name") or "").upper()
            keep = True

            # Company metadata filtering
            if detected_company:
                comp_clean = detected_company.upper().replace(" ", "")
                # Substring match against doc_name (e.g. '3M_2018_10K', 'APPLE_2022_10K')
                if comp_clean not in doc_name and detected_company.upper() not in doc_name:
                    keep = False

            if keep:
                filtered.append(c)

        trace["filtered_count"] = len(filtered)

        # Safety fallback check
        if detected_company:
            if len(filtered) >= min_candidates_threshold:
                trace["applied"] = True
                trace["reason"] = f"retained {len(filtered)} candidates matching company '{detected_company}'"
                return filtered, trace
            else:
                trace["fallback_triggered"] = True
                trace["reason"] = (
                    f"fallback to unfiltered pool: company filter for '{detected_company}' "
                    f"yielded only {len(filtered)} candidates (< {min_candidates_threshold} threshold)"
                )
                return candidates, trace

        return candidates, trace

    def reciprocal_rank_fusion(
        self,
        path_results: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None,
        k_pool: int = 40,
        rrf_k: Optional[int] = None,
        return_meta: bool = False,
    ):
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF) (Task 4).
        RRF Score = Sum( weight / (k_rrf + rank) )
        Handles deduplication cleanly and attributes source paths.
        """
        k_constant = rrf_k if rrf_k is not None else self.rrf_k
        if weights is None:
            weights = {p: 1.0 for p in path_results}

        rrf_scores = defaultdict(float)
        path_contributions = defaultdict(dict)
        seen_in_paths = defaultdict(list)
        total_raw_candidates = sum(len(items) for items in path_results.values())

        for path_name, items in path_results.items():
            w = weights.get(path_name, 1.0)
            for rank, (cid, raw_score) in enumerate(items, start=1):
                inc = w / (k_constant + rank)
                rrf_scores[cid] += inc
                path_contributions[cid][path_name] = {
                    "rank": rank,
                    "raw_score": round(raw_score, 4),
                    "rrf_contribution": round(inc, 6)
                }
                seen_in_paths[cid].append(path_name)

        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_pool]

        candidates = []
        for rank_idx, (cid, rrf_score) in enumerate(sorted_cids, 1):
            c_data = self._get_chunk_data(cid)
            if c_data:
                c_dict = dict(c_data)
                c_dict["rrf_rank"] = rank_idx
                c_dict["rrf_score"] = round(rrf_score, 6)
                c_dict["path_contributions"] = path_contributions[cid]
                c_dict["contributing_paths"] = seen_in_paths[cid]
                candidates.append(c_dict)

        fusion_meta = {
            "k_constant": k_constant,
            "paths_evaluated": list(path_results.keys()),
            "total_raw_candidates": total_raw_candidates,
            "unique_candidates_pooled": len(rrf_scores),
            "final_pool_size": len(candidates),
            "deduplicated_count": total_raw_candidates - len(rrf_scores),
        }

        if return_meta:
            return candidates, fusion_meta
        return candidates

    def _log_phase3_diagnostics(
        self,
        query: str,
        expansion_res: FinancialExpansionResult,
        paths: Dict[str, Any],
        fusion_meta: Dict[str, Any],
        filter_trace: Dict[str, Any],
        final_candidates: List[Dict[str, Any]]
    ):
        """Record structured retrieval diagnostic log (Task 5). Never overwrites benchmark results."""
        try:
            os.makedirs(self.diagnostics_dir, exist_ok=True)
            q_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.diagnostics_dir, f"phase3_diag_{timestamp}_{q_hash}.json")

            paths_summary = {}
            for p_name, p_res in paths.items():
                paths_summary[p_name] = {
                    "count": len(p_res),
                    "top_3_chunks": [cid for cid, _ in p_res[:3]] if p_res else []
                }

            top_candidates_summary = []
            for c in final_candidates[:10]:
                top_candidates_summary.append({
                    "chunk_id": c.get("chunk_id"),
                    "doc_name": c.get("doc_name"),
                    "page": c.get("page_number", 0),
                    "rrf_score": c.get("rrf_score"),
                    "contributing_paths": c.get("contributing_paths", []),
                    "path_contributions": c.get("path_contributions", {}),
                    "is_table_continuation": c.get("is_table_continuation", False),
                })

            diag_data = {
                "timestamp": timestamp,
                "original_query": query,
                "extracted_intelligence": {
                    "detected_company": expansion_res.detected_company,
                    "detected_metric": expansion_res.detected_metric,
                    "detected_statement": expansion_res.detected_statement,
                    "detected_filing_type": expansion_res.detected_filing_type,
                    "detected_periods": expansion_res.detected_periods,
                    "detected_operation": expansion_res.detected_operation,
                },
                "generated_search_queries": {
                    "dense_query": expansion_res.dense_query,
                    "bm25_query": expansion_res.bm25_query,
                    "targeted_period_queries": expansion_res.targeted_period_queries,
                },
                "path_results_summary": paths_summary,
                "fusion_metadata": fusion_meta,
                "metadata_filtering_trace": filter_trace,
                "final_candidate_count": len(final_candidates),
                "top_10_candidates": top_candidates_summary
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=2)

        except Exception as e:
            print(f"[HybridRetriever] Warning: Could not write Phase 3 diagnostic log: {e}")

    def get_candidate_pool(
        self,
        query: str,
        k_pool: int = 40,
        use_expansion: bool = True,
        dense_k: int = 20,
        sparse_k: int = 20,
        mode: str = "baseline",
        rrf_weights: Optional[Dict[str, float]] = None,
        log_diagnostics: bool = True,
    ) -> Tuple[List[Dict[str, Any]], FinancialExpansionResult]:
        """
        Generate candidate pool using hybrid dense/sparse and expanded paths.
        Supports:
          - mode="baseline": Original Phase 1/2 Multi-Path RRF (default).
          - mode="phase3": Enhanced Financial Retrieval Intelligence with safe metadata filtering,
            targeted period retrieval, clean deduplication, and structured observability.
        """
        # Baseline Mode (100% backward compatible)
        if mode == "baseline":
            expansion_res = self.expander.expand(query)
            paths = {}
            weights = {}

            # 1. Base Dense
            paths["dense_orig"] = self.retrieve_dense(query, k=dense_k)
            weights["dense_orig"] = 1.0

            # 2. Base Sparse
            paths["sparse_orig"] = self.retrieve_sparse(query, k=sparse_k)
            weights["sparse_orig"] = 1.0

            # 3. Expanded Paths
            if use_expansion and expansion_res.expansion_terms:
                paths["dense_expanded"] = self.retrieve_dense(expansion_res.dense_query, k=dense_k)
                weights["dense_expanded"] = 1.2

                paths["sparse_expanded"] = self.retrieve_sparse(expansion_res.bm25_query, k=sparse_k)
                weights["sparse_expanded"] = 1.2

            if rrf_weights is not None:
                weights.update(rrf_weights)

            candidates = self.reciprocal_rank_fusion(paths, weights=weights, k_pool=k_pool)
            return candidates, expansion_res

        # Phase 3 Experimental Mode
        expansion_res = self.expander.expand(query)
        paths = {}
        weights = {}

        # 1. Base Dense & Sparse
        paths["dense_orig"] = self.retrieve_dense(query, k=dense_k)
        weights["dense_orig"] = 1.0

        paths["sparse_orig"] = self.retrieve_sparse(query, k=sparse_k)
        weights["sparse_orig"] = 1.0

        # 2. GAAP Expanded Paths
        if use_expansion and expansion_res.expansion_terms:
            paths["dense_expanded"] = self.retrieve_dense(expansion_res.dense_query, k=dense_k)
            weights["dense_expanded"] = 1.1

            paths["sparse_expanded"] = self.retrieve_sparse(expansion_res.bm25_query, k=sparse_k)
            weights["sparse_expanded"] = 1.1

        # 3. Targeted Period Retrieval Paths (Task 1 & Task 4)
        if expansion_res.targeted_period_queries:
            for idx, p_query in enumerate(expansion_res.targeted_period_queries):
                p_label = f"period_{idx+1}"
                paths[f"dense_{p_label}"] = self.retrieve_dense(p_query, k=15)
                weights[f"dense_{p_label}"] = 1.0

                paths[f"sparse_{p_label}"] = self.retrieve_sparse(p_query, k=15)
                weights[f"sparse_{p_label}"] = 1.0

        if rrf_weights is not None:
            weights.update(rrf_weights)

        # 4. Configurable RRF Fusion with Deduplication (Task 4)
        candidates, fusion_meta = self.reciprocal_rank_fusion(
            paths, weights=weights, k_pool=k_pool, return_meta=True
        )

        # 5. Safe Metadata-Aware Filtering (Task 2)
        filtered_candidates, filter_trace = self.filter_candidates_by_metadata(
            candidates,
            detected_company=expansion_res.detected_company,
            detected_periods=expansion_res.detected_periods,
            min_candidates_threshold=5,
        )

        # 6. Structured Observability Logging (Task 5)
        if log_diagnostics:
            self._log_phase3_diagnostics(
                query=query,
                expansion_res=expansion_res,
                paths=paths,
                fusion_meta=fusion_meta,
                filter_trace=filter_trace,
                final_candidates=filtered_candidates
            )

        return filtered_candidates, expansion_res

    def retrieve(
        self,
        query: str,
        k_pool: int = 40,
        k_final: int = 5,
        use_expansion: bool = True,
        use_statement_reranking: bool = True,
        mode: str = "baseline",
        log_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end retrieval with ablations and Phase 3 support.
        """
        candidates, expansion_res = self.get_candidate_pool(
            query=query,
            k_pool=k_pool,
            use_expansion=use_expansion,
            mode=mode,
            log_diagnostics=log_diagnostics
        )

        if use_statement_reranking and self.reranker is not None:
            final_chunks = self.reranker.rerank(
                query=query,
                candidates=candidates,
                target_statement=expansion_res.detected_statement,
                statement_confidence=expansion_res.statement_confidence,
                top_k=k_final,
                apply_boost=True
            )
        elif self.reranker is not None:
            final_chunks = self.reranker.rerank(
                query=query,
                candidates=candidates,
                target_statement=None,
                statement_confidence=0.0,
                top_k=k_final,
                apply_boost=False
            )
        else:
            final_chunks = candidates[:k_final]

        return {
            "query": query,
            "mode": mode,
            "expansion": expansion_res,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "final_chunks": final_chunks,
        }
