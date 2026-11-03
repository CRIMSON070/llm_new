"""
Statement-Aware Reranker for FinSage Phase 9C.

Combines semantic cross-encoder reranking (FlagReranker BAAI/bge-reranker-base)
with transparent, configurable metadata-based adjustments for financial statements and table structures.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from FlagEmbedding import FlagReranker

class StatementAwareReranker:
    """
    Reranks candidate chunks using a cross-encoder model combined with
    statement-type detection and structural table boosts.
    """

    STATEMENT_SIGNATURES = {
        "cash_flow_statement": [
            r"consolidated\s*statements?\s*of\s*cash\s*flows?",
            r"statement\s*of\s*cash\s*flow\s*s?",
            r"cash\s*flows?\s*from\s*operating\s*activities",
            r"cash\s*flows?\s*from\s*investing\s*activities",
            r"cash\s*flows?\s*from\s*financing\s*activities",
            r"purchases\s*of\s*property[,\s]*plant\s*and\s*equipment",
            r"free\s*cash\s*flow\s*\(non-gaap",
        ],
        "balance_sheet": [
            r"consolidated\s*balance\s*shee?\s*ts?",
            r"consolidated\s*statements?\s*of\s*financial\s*position",
            r"balance\s*shee?\s*ts?",
            r"property[,\s]*plant\s*and\s*equipment\s*[\u2010\u2013\u2014\-]?\s*net",
            r"current\s*assets\s*cash\s*and\s*cash\s*equivalents",
            r"total\s*assets\s*liabilities\s*and\s*equity",
        ],
        "income_statement": [
            r"consolidated\s*statements?\s*of\s*incom\s*e?",
            r"consolidated\s*statements?\s*of\s*operations?",
            r"statements?\s*of\s*incom\s*e?",
            r"statements?\s*of\s*operations?",
            r"statements?\s*of\s*earnings",
            r"cost\s*of\s*sales\s*selling[,\s]*general",
        ],
        "cover_page": [
            r"securities\s*registered\s*pursuant\s*to\s*section\s*12\(b\)",
            r"title\s*of\s*each\s*class\s*trading\s*symbol",
            r"form\s*10-k\s*annual\s*report",
            r"exact\s*name\s*of\s*registrant",
            r"trading\s*symbol\(s\)",
        ],
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
        statement_boost_weight: float = 2.0,
        company_boost_weight: float = 1.5,
        period_boost_weight: float = 1.5,
        table_boost_weight: float = 0.5,
        min_confidence_for_boost: float = 0.5,
    ):
        self.model_name = model_name
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        self.statement_boost_weight = statement_boost_weight
        self.company_boost_weight = company_boost_weight
        self.period_boost_weight = period_boost_weight
        self.table_boost_weight = table_boost_weight
        self.min_confidence_for_boost = min_confidence_for_boost

    def detect_chunk_statement_type(self, text: str) -> Optional[str]:
        """Detect if the chunk belongs to an official financial statement."""
        t_lower = text[:1500].lower()  # focus on top of chunk / headers
        for stmt, patterns in self.STATEMENT_SIGNATURES.items():
            for p in patterns:
                if re.search(p, t_lower):
                    return stmt
        return None

    def has_table_structure(self, text: str) -> bool:
        """Detect whether the chunk contains structured tables."""
        if "|---" in text or "|:---" in text:
            return True
        num_lines = text.count("\n")
        dollar_signs = text.count("$")
        if num_lines > 5 and dollar_signs >= 3:
            return True
        return False

    def detect_query_company(self, query: str) -> Optional[str]:
        q_lower = query.lower()
        if re.search(r"\b3m\b", q_lower):
            return "3M"
        elif re.search(r"\bulta(\s*beauty)?\b", q_lower):
            return "ULTABEAUTY"
        return None

    def detect_query_period(self, query: str) -> List[str]:
        # Extract years like 2018, 2022, 2023, FY18, FY2023
        found = set(re.findall(r"\b(201\d|202\d)\b", query))
        fy_found = re.findall(r"\bfy\s*(20\d\d|\d\d)\b", query.lower())
        for fy in fy_found:
            if len(fy) == 2:
                found.add(f"20{fy}")
            else:
                found.add(fy)
        return list(found)

    def extract_chunk_company_and_period(self, chunk: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        doc_name = chunk.get("doc_name") or ""
        doc_upper = doc_name.upper()
        comp = None
        if "3M" in doc_upper:
            comp = "3M"
        elif "ULTA" in doc_upper:
            comp = "ULTABEAUTY"
            
        period = None
        m = re.search(r"(201\d|202\d)", doc_upper)
        if m:
            period = m.group(1)
        return comp, period

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        target_statement: Optional[str] = None,
        statement_confidence: float = 0.0,
        top_k: int = 5,
        apply_boost: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks with cross-encoder and statement-aware adjustments.
        """
        if not candidates:
            return []

        q_company = self.detect_query_company(query)
        q_periods = self.detect_query_period(query)

        # Form pairs for BGE Reranker
        pairs = [[query, c["text"]] for c in candidates]
        raw_scores = self.reranker.compute_score(pairs, batch_size=8)
        if not isinstance(raw_scores, list):
            raw_scores = [raw_scores]

        scored_candidates = []
        for c, raw_score in zip(candidates, raw_scores):
            chunk_text = c.get("text", "")
            chunk_stmt = self.detect_chunk_statement_type(chunk_text)
            has_table = self.has_table_structure(chunk_text)
            c_company, c_period = self.extract_chunk_company_and_period(c)

            stmt_boost = 0.0
            table_boost = 0.0
            company_boost = 0.0
            period_boost = 0.0

            if apply_boost:
                # 1. Company Boost
                if q_company and c_company == q_company:
                    company_boost = self.company_boost_weight

                # 2. Period Boost
                if q_periods and c_period and c_period in q_periods:
                    period_boost = self.period_boost_weight

                # 3. Statement Boost (only if confident)
                if (
                    target_statement is not None
                    and statement_confidence >= self.min_confidence_for_boost
                    and chunk_stmt == target_statement
                ):
                    stmt_boost = self.statement_boost_weight

                # 4. Table Boost
                if has_table and target_statement in ["cash_flow_statement", "balance_sheet", "income_statement"]:
                    table_boost = self.table_boost_weight

            final_score = float(raw_score) + company_boost + period_boost + stmt_boost + table_boost

            item = dict(c)
            item["raw_rerank_score"] = round(float(raw_score), 4)
            item["chunk_statement_type"] = chunk_stmt
            item["has_table"] = has_table
            item["chunk_company"] = c_company
            item["chunk_period"] = c_period
            item["company_boost"] = round(company_boost, 4)
            item["period_boost"] = round(period_boost, 4)
            item["statement_boost"] = round(stmt_boost, 4)
            item["table_boost"] = round(table_boost, 4)
            item["final_score"] = round(final_score, 4)

            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_candidates[:top_k]
