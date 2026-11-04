"""
FinSage Phase 4: Production Statement-Aware Reranker
====================================================
Paper Reference: FinSage (arXiv:2504.14493, CIKM 2025)

Combined Retrieval Intelligence Improvements:
1. Improvement 1: Full 32 FinanceBench + 20 Stage 3 Company Detection, Ticker Mapping, and Filing Alignment.
2. Improvement 2: Fiscal Year and Filing-Type (10-K, 10-Q, 8-K, EARNINGS) Period-Aware Scoring with Fallbacks.
3. Improvement 3: Unicode NFKC Normalization and Terminology Alignment.
4. Improvement 4: Multi-Statement Balanced Evidence Reranking for ratio and multi-metric queries.
5. Calibrated Statement Boosts (+4.5) & Targeted Negative Narrative Penalties (-3.0).
6. Full candidate text evaluation with FlagReranker max_length=512, FP16, batch_size=8.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from FlagEmbedding import FlagReranker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.financial_expander import (
    FinancialQueryExpander,
    normalize_unicode,
    parse_doc_metadata,
)
from src.retrieval.coreference_resolver import CoreferenceResolver


class Phase2Reranker:
    """
    Enhanced production reranker combining neural cross-encoder scores with
    calibrated statement routing, multi-statement balance, table structure detection,
    and company / period / filing alignment.
    """

    STATEMENT_PATTERNS = {
        "cash_flow_statement": [
            r"consolidated\s*statements?\s*of\s*cash\s*flows?",
            r"statements?\s*of\s*cash\s*flows?",
            r"cash\s*flows?\s*from\s*operating\s*activities",
            r"cash\s*flows?\s*from\s*investing\s*activities",
            r"purchases\s*of\s*property[,\s]*plant\s*and\s*equipment",
        ],
        "balance_sheet": [
            r"consolidated\s*balance\s*shee?\s*ts?",
            r"consolidated\s*statements?\s*of\s*financial\s*position",
            r"balance\s*shee?\s*ts?",
            r"property[,\s]*plant\s*and\s*equipment\s*[\u2010\u2013\u2014\-]?\s*net",
            r"total\s*assets\s*liabilities\s*and\s*equity",
            r"merchandise\s*inventories",
        ],
        "income_statement": [
            r"consolidated\s*statements?\s*of\s*incom\s*e?",
            r"consolidated\s*statements?\s*of\s*operations?",
            r"statements?\s*of\s*incom\s*e?",
            r"statements?\s*of\s*operations?",
            r"statements?\s*of\s*earnings",
            r"net\s*sales",
            r"total\s*revenues?",
        ],
        "cover_page": [
            r"securities\s*registered\s*pursuant\s*to\s*section\s*12\(b\)",
            r"title\s*of\s*each\s*class\s*trading\s*symbol",
            r"form\s*10-k\s*annual\s*report",
            r"exact\s*name\s*of\s*registrant",
            r"trading\s*symbol\(s\)",
        ],
    }

    NARRATIVE_DISTRACTION_PATTERNS = [
        r"environmental\s*(?:projects|protection|matters|obligations|liabilities|remediation|cleanup|costs|claims)",
        r"protecting\s+the\s+environment",
        r"item\s+1\.\s*business",
        r"note\s*(?:1[0-9]|2[0-9]|[3-9])[\s\.\:\-]+(?:commitments|contingencies|debt|subsequent|financing)",
        r"legal\s*proceedings",
        r"pfas\s*(?:related|contamination|liabilities)",
        r"superfund\s*(?:sites|liabilities)",
        r"forward-looking\s*statements?",
        r"cautionary\s*note\s*regarding",
        r"commercial\s*paper\s*issuances",
        r"maturation\s*of\s*\$\d+\s*million",
    ]

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
        statement_boost_weight: float = 4.5,
        negative_disclosure_penalty: float = -3.0,
        company_boost_weight: float = 1.5,
        period_boost_weight: float = 1.5,
        filing_boost_weight: float = 0.5,
        table_boost_weight: float = 0.5,
        expander: Optional[FinancialQueryExpander] = None,
        coref_resolver: Optional[CoreferenceResolver] = None,
        enable_coreference: bool = True,
        context_stitcher: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16, max_length=512)
        self.statement_boost_weight = statement_boost_weight
        self.negative_disclosure_penalty = negative_disclosure_penalty
        self.company_boost_weight = company_boost_weight
        self.period_boost_weight = period_boost_weight
        self.filing_boost_weight = filing_boost_weight
        self.table_boost_weight = table_boost_weight
        self.expander = expander or FinancialQueryExpander(enable_coreference=enable_coreference)
        self.enable_coreference = enable_coreference
        self.coref_resolver = (coref_resolver or CoreferenceResolver()) if enable_coreference else None
        self.context_stitcher = context_stitcher

    def detect_query_statement_intent(self, query: str) -> Optional[str]:
        """Identifies if the user query explicitly requests a financial statement."""
        stmt, conf = self.expander.detect_statement(query)
        if conf >= 0.5:
            return stmt
        return None

    def detect_chunk_statement_type(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Detects if chunk text belongs to an official financial statement."""
        prop = chunk.get("propagated_statement")
        if prop:
            prop_l = prop.lower()
            if "cash flows" in prop_l or "cash flow" in prop_l:
                return "cash_flow_statement"
            elif "balance sheets" in prop_l or "balance sheet" in prop_l or "financial position" in prop_l:
                return "balance_sheet"
            elif "income" in prop_l or "operations" in prop_l or "earnings" in prop_l:
                return "income_statement"

        text = chunk.get("text", "")
        t_norm = normalize_unicode(text)
        t_lead = t_norm[:1500].lower()

        if "consolidated statements of cash flows" in t_lead or "consolidated statement of cash flow" in t_lead:
            return "cash_flow_statement"
        elif "consolidated balance sheet" in t_lead or "consolidated statements of financial position" in t_lead:
            return "balance_sheet"
        elif "consolidated statements of income" in t_lead or "consolidated statement of income" in t_lead or "consolidated statements of operations" in t_lead:
            return "income_statement"
        elif "securities registered pursuant to section 12(b)" in t_lead:
            return "cover_page"

        for stmt, patterns in self.STATEMENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, t_lead):
                    return stmt
        return None

    def is_narrative_distraction(self, text: str) -> bool:
        """Detects if chunk is an environmental liability note, debt footnote, or narrative MD&A prose."""
        t_norm = normalize_unicode(text)
        t_lower = t_norm.lower()
        for p in self.NARRATIVE_DISTRACTION_PATTERNS:
            if re.search(p, t_lower):
                return True
        return False

    def has_table_structure(self, text: str) -> bool:
        """Detects if chunk contains structured tables."""
        if "|---" in text or "|:---" in text:
            return True
        num_lines = text.count("\n")
        dollar_signs = text.count("$")
        num_matches = len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text))
        if num_lines >= 4 and (dollar_signs >= 2 or num_matches >= 4):
            return True
        return False

    def detect_query_company(self, query: str) -> Optional[str]:
        """Detect company name across all 32 benchmark companies + 20 Stage 3 companies."""
        return self.expander.detect_company(query)

    def detect_query_periods(self, query: str) -> List[str]:
        """Detect query periods and fiscal years."""
        return self.expander.detect_periods(query)

    def detect_query_filing_type(self, query: str) -> Optional[str]:
        """Detect requested filing type (10-K, 10-Q, 8-K, EARNINGS)."""
        return self.expander.detect_filing_type(query)

    def extract_chunk_company_and_period(self, chunk: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extracts company prefix, year, and filing type from chunk doc_name using reliable metadata parser.
        Returns: (company_name, year_str, filing_type)
        """
        doc_name = chunk.get("doc_name") or ""
        meta = parse_doc_metadata(doc_name)
        if meta:
            return meta["company"], str(meta["year"]), meta["filing_type"]
        return None, None, None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        apply_boosts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Cross-Encoder combined with calibrated statement routing,
        multi-statement balanced representation, and company / period / filing alignment.
        """
        # Optional Context Window Stitching (Experiment L.8/L.9)
        # If context_stitcher is active, enrich candidate text while strictly preserving
        # original candidate chunk_id, metadata, and pool identity.
        if self.context_stitcher is not None:
            cands_for_rerank = []
            for c in candidates:
                sc = self.context_stitcher.stitch_candidate(c)
                c_copy = dict(sc)
                c_copy["text"] = sc["stitched_text"]
                cands_for_rerank.append(c_copy)
        else:
            cands_for_rerank = candidates

        norm_query = normalize_unicode(query)

        # Intelligence Extraction
        q_company = self.detect_query_company(norm_query)
        q_periods = self.detect_query_periods(norm_query)
        q_filing = self.detect_query_filing_type(norm_query)

        # Multi-Statement Intent Detection
        is_multi, multi_rule, multi_components = self.expander.detect_multi_statement(norm_query)
        target_statements: List[str] = []
        if is_multi:
            target_statements = [c["statement"] for c in multi_components]
        else:
            single_stmt = self.detect_query_statement_intent(norm_query)
            if single_stmt:
                target_statements = [single_stmt]

        # Prepare normalized candidate texts for Cross-Encoder with Chunk-Level Coreference Resolution
        # (Section 3.1 & Appendix A.6: resolves generic pronouns to explicit corporate antecedents)
        if self.coref_resolver is not None:
            # Avoid redundant chunk antecedent expansion when parent company is already explicitly in the query.
            # When the parent company is explicit, injecting artificial company headers into filing chunks distorts
            # cross-encoder scoring against narrative evidence chunks.
            is_company_explicit = bool(q_company and self.coref_resolver.is_company_explicit(norm_query, q_company))
            if not is_company_explicit:
                pairs = [
                    [
                        norm_query,
                        normalize_unicode(
                            self.coref_resolver.resolve_chunk_antecedents(
                                chunk_text=c.get("text", ""),
                                doc_name=c.get("doc_name", ""),
                                section_heading=c.get("section_heading")
                            )
                        )
                    ]
                    for c in cands_for_rerank
                ]
            else:
                pairs = [[norm_query, normalize_unicode(c.get("text", ""))] for c in cands_for_rerank]
        else:
            pairs = [[norm_query, normalize_unicode(c.get("text", ""))] for c in cands_for_rerank]

        raw_scores = self.reranker.compute_score(pairs, batch_size=8)
        if not isinstance(raw_scores, list):
            raw_scores = [raw_scores]

        scored_candidates = []
        for c, raw_score in zip(cands_for_rerank, raw_scores):
            raw_text = c.get("text", "")
            chunk_stmt = self.detect_chunk_statement_type(c)
            has_table = self.has_table_structure(raw_text)
            is_distraction = self.is_narrative_distraction(raw_text)
            doc_name = c.get("doc_name", "")
            c_company, c_period, c_filing = self.extract_chunk_company_and_period(c)

            stmt_boost = 0.0
            table_boost = 0.0
            company_boost = 0.0
            period_boost = 0.0
            filing_boost = 0.0
            penalty = 0.0

            if apply_boosts:
                # 1. Company Match (with reliable taxonomy and ticker aliases)
                if q_company and doc_name:
                    if self.expander.match_company(q_company, doc_name):
                        company_boost = self.company_boost_weight

                # 2. Period Match (with explicit year / fiscal year detection)
                if q_periods and c_period and c_period in q_periods:
                    period_boost = self.period_boost_weight

                # 3. Filing Type Match (10-K, 10-Q, 8-K, EARNINGS)
                if q_filing and c_filing and (q_filing in c_filing or c_filing in q_filing):
                    filing_boost = self.filing_boost_weight

                # 4. Statement Routing (Supports single and multi-statement queries)
                if target_statements:
                    if chunk_stmt in target_statements:
                        stmt_boost = self.statement_boost_weight

                    # Table Structure Boost for financial statements
                    if has_table and any(s in ["cash_flow_statement", "balance_sheet", "income_statement"] for s in target_statements):
                        table_boost = self.table_boost_weight

                    # Targeted Negative Disclosure Penalty
                    if is_distraction or (not has_table and chunk_stmt not in target_statements):
                        penalty = self.negative_disclosure_penalty

            final_score = float(raw_score) + company_boost + period_boost + filing_boost + stmt_boost + table_boost + penalty

            item = dict(c)
            item["raw_rerank_score"] = round(float(raw_score), 4)
            item["chunk_statement_type"] = chunk_stmt
            item["has_table"] = has_table
            item["is_narrative_distraction"] = is_distraction
            item["company_boost"] = round(company_boost, 4)
            item["period_boost"] = round(period_boost, 4)
            item["filing_boost"] = round(filing_boost, 4)
            item["statement_boost"] = round(stmt_boost, 4)
            item["table_boost"] = round(table_boost, 4)
            item["negative_penalty"] = round(penalty, 4)
            item["final_score"] = round(final_score, 4)

            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Improvement 4: Balanced Multi-Statement Top-K Selection
        if is_multi and len(target_statements) >= 2:
            stmt_a = target_statements[0]
            stmt_b = target_statements[1]

            quota_per_stmt = max(2, min(3, top_k // 3))
            selected_cids = set()
            balanced_top_k: List[Dict[str, Any]] = []

            # 1. Best candidates for Statement A
            cands_a = [c for c in scored_candidates if c.get("chunk_statement_type") == stmt_a]
            for c in cands_a[:quota_per_stmt]:
                if c["chunk_id"] not in selected_cids:
                    selected_cids.add(c["chunk_id"])
                    balanced_top_k.append(c)

            # 2. Best candidates for Statement B
            cands_b = [c for c in scored_candidates if c.get("chunk_statement_type") == stmt_b]
            for c in cands_b[:quota_per_stmt]:
                if c["chunk_id"] not in selected_cids:
                    selected_cids.add(c["chunk_id"])
                    balanced_top_k.append(c)

            # 3. Fill remaining slots with the highest overall scoring candidates
            for c in scored_candidates:
                if len(balanced_top_k) >= top_k:
                    break
                if c["chunk_id"] not in selected_cids:
                    selected_cids.add(c["chunk_id"])
                    balanced_top_k.append(c)

            # Sort balanced selection by final_score descending
            balanced_top_k.sort(key=lambda x: x["final_score"], reverse=True)
            return balanced_top_k[:top_k]

        # Standard top_k selection for single-statement and ordinary queries
        return scored_candidates[:top_k]
