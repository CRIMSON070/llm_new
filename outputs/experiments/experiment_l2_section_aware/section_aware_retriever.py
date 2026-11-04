"""
outputs/experiments/experiment_l2_section_aware/section_aware_retriever.py
========================================================================
Experiment L.2: Section-Aware Multi-Path Retrieval with Query-Dependent Intent Routing.

Preserves the original retrieval implementation as the control arm.
Does NOT modify baseline or Experiment L indexes.
Applies narrative table capping and Item 1 / MD&A protection only to qualitative/narrative queries.
Leaves quantitative queries completely unchanged.
"""

import os
import sys
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.multi_path_retriever import MultiPathRetriever, normalize_unicode
from src.retrieval.financial_expander import FinancialQueryExpander


def is_pipeline_table(chunk_text: str) -> bool:
    """Detects clinical/development pipeline tables in pharmaceutical filings."""
    t = chunk_text.lower()
    return "compound" in t and ("indication" in t or "study status" in t or "phase" in t)


def is_item1_narrative(chunk: Dict[str, Any]) -> bool:
    """Identifies Item 1 Business prose sections (pages 4-35 in Form 10-K)."""
    page = chunk.get("page_number", chunk.get("page", 0))
    text = chunk.get("text", "")
    return (4 <= page <= 35) and not is_pipeline_table(text)


class SectionAwareMultiPathRetriever(MultiPathRetriever):
    """
    Subclasses MultiPathRetriever to implement Approach B:
    Query-Dependent Intent Routing & Section-Aware Quota Capping.
    """

    def __init__(
        self,
        faiss_index,
        bm25_retriever,
        chunk_store: Dict[str, Dict[str, Any]],
        metadata_retriever=None,
        hyde_retriever=None,
        chunk_bundler=None,
        rrf_k: int = 60,
        expander: Optional[FinancialQueryExpander] = None,
        narrative_table_cap: int = 4,
        item1_boost: float = 0.05,
    ):
        super().__init__(
            faiss_index=faiss_index,
            bm25_retriever=bm25_retriever,
            chunk_store=chunk_store,
            metadata_retriever=metadata_retriever,
            hyde_retriever=hyde_retriever,
            chunk_bundler=chunk_bundler,
            rrf_k=rrf_k,
            expander=expander,
        )
        self.narrative_table_cap = narrative_table_cap
        self.item1_boost = item1_boost

    def classify_query_intent(self, query: str, question_type: Optional[str] = None) -> str:
        """
        Classifies query into QUALITATIVE_NARRATIVE, QUANTITATIVE_FINANCIAL, or GENERAL_HYBRID.
        Uses question_type metadata if available, and augments with lexical cues.
        """
        if question_type == "qualitative_narrative":
            return "QUALITATIVE_NARRATIVE"
        if question_type in ("single_statement", "multi_statement"):
            return "QUANTITATIVE_FINANCIAL"

        q_lower = query.lower()
        quant_cues = [
            "revenue", "sales", "margin", "roa", "operating income", "net income",
            "balance", "ratio", "expense", "capex", "assets", "cash", "total",
            "diluted", "share", "eps", "ebitda", "capital expenditures", "dividend"
        ]
        narr_cues = [
            "milestone", "regulatory", "approval", "strategy", "acquisition",
            "commitments", "operational", "actions", "factors", "legal",
            "proceedings", "risk", "risks", "facility", "capacity", "expansion",
            "disclose", "commercialize", "pipeline"
        ]

        has_quant = any(c in q_lower for c in quant_cues)
        has_narr = any(c in q_lower for c in narr_cues)

        if has_narr and not has_quant:
            return "QUALITATIVE_NARRATIVE"
        elif has_quant and not has_narr:
            return "QUANTITATIVE_FINANCIAL"
        else:
            return "GENERAL_HYBRID"

    def retrieve_multi_path(
        self,
        query: str,
        k_pool: int = 65,
        enable_bm25: bool = True,
        enable_dense: bool = True,
        enable_metadata: bool = False,
        enable_hyde: bool = False,
        path_weights: Optional[Dict[str, float]] = None,
        question_type: Optional[str] = None,
        arm_mode: str = "section_aware",  # "control" or "section_aware"
    ) -> Dict[str, Any]:
        """
        Executes multi-path retrieval.
        If arm_mode == "control", behaves exactly like standard MultiPathRetriever.
        If arm_mode == "section_aware", applies Approach B for QUALITATIVE_NARRATIVE queries.
        """
        if arm_mode == "control":
            return super().retrieve_multi_path(
                query=query,
                k_pool=k_pool,
                enable_bm25=enable_bm25,
                enable_dense=enable_dense,
                enable_metadata=enable_metadata,
                enable_hyde=enable_hyde,
                path_weights=path_weights,
            )

        norm_query = normalize_unicode(query)
        exp_res = self.expander.expand(norm_query)
        intent = self.classify_query_intent(query, question_type=question_type)

        path_results = {}
        telemetry = {"intent": intent}

        # For QUANTITATIVE_FINANCIAL or GENERAL_HYBRID: Unchanged standard retrieval
        if intent != "QUALITATIVE_NARRATIVE":
            if enable_bm25:
                bm25_target = exp_res.bm25_query if exp_res.bm25_query else norm_query
                sparse_res = self.retrieve_sparse(bm25_target, k=k_pool)
                path_results["path1_bm25"] = sparse_res
                telemetry["path1_bm25_count"] = len(sparse_res)

            if enable_dense:
                dense_target = exp_res.dense_query if exp_res.dense_query else norm_query
                dense_res = self.retrieve_dense(dense_target, k=k_pool)
                path_results["path2_dense"] = dense_res
                telemetry["path2_dense_count"] = len(dense_res)

            candidates = self.reciprocal_rank_fusion(
                path_results=path_results,
                weights=path_weights,
                k_pool=k_pool,
            )
            return {
                "query": query,
                "normalized_query": norm_query,
                "expansion_result": exp_res,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "telemetry": telemetry,
                "intent": intent,
            }

        # For QUALITATIVE_NARRATIVE:
        # 1. Oversample sparse and dense to K_raw = 150
        k_raw = 150
        if enable_bm25:
            bm25_target = exp_res.bm25_query if exp_res.bm25_query else norm_query
            raw_sparse = self.retrieve_sparse(bm25_target, k=k_raw)
            
            # Apply Narrative Pipeline Table Cap Q=4 to BM25 candidates
            capped_sparse = []
            pipe_count = 0
            for cid, score in raw_sparse:
                c_data = self.chunk_store.get(cid, {})
                txt = c_data.get("text", "")
                if is_pipeline_table(txt):
                    if pipe_count < self.narrative_table_cap:
                        capped_sparse.append((cid, score))
                        pipe_count += 1
                else:
                    capped_sparse.append((cid, score))
            path_results["path1_bm25"] = capped_sparse
            telemetry["path1_bm25_raw_count"] = len(raw_sparse)
            telemetry["path1_bm25_capped_count"] = len(capped_sparse)
            telemetry["pipeline_tables_capped"] = pipe_count

        if enable_dense:
            dense_target = exp_res.dense_query if exp_res.dense_query else norm_query
            raw_dense = self.retrieve_dense(dense_target, k=k_raw)
            path_results["path2_dense"] = raw_dense
            telemetry["path2_dense_count"] = len(raw_dense)

        # 2. Reciprocal Rank Fusion with Item 1 Narrative Boost (+0.05)
        rrf_scores = defaultdict(float)
        path_contributions = defaultdict(dict)
        weights = path_weights or {p: 1.0 for p in path_results}

        for path_name, items in path_results.items():
            w = weights.get(path_name, 1.0)
            for rank, (cid, raw_score) in enumerate(items, start=1):
                inc = w / (self.rrf_k + rank)
                rrf_scores[cid] += inc
                path_contributions[cid][path_name] = {
                    "rank": rank,
                    "raw_score": round(raw_score, 4),
                    "rrf_contribution": round(inc, 6),
                }

        # Apply Item 1 Narrative Protection Boost
        boosted_count = 0
        for cid in list(rrf_scores.keys()):
            c_data = self.chunk_store.get(cid, {})
            if is_item1_narrative(c_data):
                rrf_scores[cid] += self.item1_boost
                boosted_count += 1
        telemetry["item1_chunks_boosted"] = boosted_count

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

        return {
            "query": query,
            "normalized_query": norm_query,
            "expansion_result": exp_res,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "telemetry": telemetry,
            "intent": intent,
        }
