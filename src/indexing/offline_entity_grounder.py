"""
src/indexing/offline_entity_grounder.py
=======================================
Implements offline corporate entity grounding for FinSage (Experiment L).
Prepends deterministic, compact corporate antecedent headers to raw chunk text
prior to FAISS dense embedding and BM25 tokenization.

Design & Safety Rules:
1. Resolves canonical corporate identity (name, ticker, filing metadata) deterministically
   from the filing manifest or document prefix.
2. Identifies product brands or subsidiaries that *actually appear* in the chunk text,
   strictly using the verified COREFERENCE_TAXONOMY.
3. Does NOT rewrite narrative sentences or perform hallucinated text replacements.
4. Preserves all existing table continuation headers ([TABLE CONTINUATION | ...]).
5. Idempotent: safe to run multiple times without duplicate headers.
"""

import os
import re
import csv
from typing import Dict, Any, List, Optional, Tuple

from src.retrieval.coreference_resolver import (
    COMPANY_CANONICAL_INFO,
    COREFERENCE_TAXONOMY,
    COMPILED_BRAND_PATTERNS
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_CSV = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "downloaded_filings_manifest.csv"
)


class OfflineEntityGrounder:
    """
    Grounds raw filing chunks with structured corporate entity metadata
    before index generation.
    """

    def __init__(self, manifest_path: Optional[str] = None):
        self.canonical_info = COMPANY_CANONICAL_INFO
        self.taxonomy = COREFERENCE_TAXONOMY
        self.brand_patterns = COMPILED_BRAND_PATTERNS
        self.filing_metadata: Dict[str, Dict[str, str]] = {}
        
        # Load manifest if available for exact filing date / period metadata
        path = manifest_path or MANIFEST_CSV
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    self.filing_metadata[r["doc_name"]] = {
                        "company_name": r.get("company_name", ""),
                        "ticker": r.get("ticker", ""),
                        "fiscal_year": r.get("fiscal_year", ""),
                    }

    def resolve_filing_identity(self, doc_name: str) -> Dict[str, str]:
        """
        Deterministically resolves corporate identity from manifest or doc_name prefix.
        e.g. LLY_2023_10K -> Entity: Eli Lilly and Company, Ticker: LLY, Year: 2023
        """
        # 1. Check filing manifest
        if doc_name in self.filing_metadata:
            meta = self.filing_metadata[doc_name]
            return {
                "company_name": meta["company_name"],
                "ticker": meta["ticker"],
                "fiscal_year": meta["fiscal_year"],
                "doc_name": doc_name
            }

        # 2. Derive from doc_name prefix (e.g. LLY_2023_10K)
        parts = doc_name.split("_")
        pfx = parts[0].upper()
        year = parts[1] if len(parts) > 1 and parts[1].isdigit() else ""

        for c_id, info in self.canonical_info.items():
            if (
                info.get("doc_prefix", "").upper() == pfx
                or info.get("ticker", "").upper() == pfx
                or c_id == pfx
            ):
                return {
                    "company_name": info.get("name", pfx),
                    "ticker": info.get("ticker", pfx),
                    "fiscal_year": year,
                    "doc_name": doc_name
                }

        # Fallback
        return {
            "company_name": pfx,
            "ticker": pfx,
            "fiscal_year": year,
            "doc_name": doc_name
        }

    def detect_chunk_brands(self, text: str, canonical_company: Optional[str] = None) -> List[str]:
        """
        Scans chunk text for verified brand/product names associated with this company.
        Never associates a brand of Company A with Company B.
        """
        if not text:
            return []

        matched_brands = set()
        for canon, pat, pat_str in self.brand_patterns:
            if canonical_company and canon != canonical_company:
                continue
            m = pat.search(text)
            if m:
                # Store the actual matched product name (cleaned)
                matched_brands.add(m.group(0).strip())

        return sorted(list(matched_brands))

    def get_canonical_company_id(self, ticker: str, doc_name: str) -> Optional[str]:
        """Finds canonical key (e.g. ELILILLY) for a ticker or doc_name prefix."""
        pfx = doc_name.split("_")[0].upper()
        ticker_up = ticker.upper()
        for c_id, info in self.canonical_info.items():
            if (
                info.get("ticker", "").upper() == ticker_up
                or info.get("doc_prefix", "").upper() == pfx
                or c_id == pfx
            ):
                return c_id
        return None

    def build_corporate_header(self, doc_name: str, raw_text: str) -> str:
        """
        Generates a concise corporate context header:
        [CORPORATE CONTEXT | Entity: Eli Lilly and Company | Ticker: LLY | Filing: LLY_2023_10K | Product: Mounjaro]
        """
        identity = self.resolve_filing_identity(doc_name)
        c_name = identity["company_name"]
        ticker = identity["ticker"]
        year = identity["fiscal_year"]
        
        canon_id = self.get_canonical_company_id(ticker, doc_name)
        detected_brands = self.detect_chunk_brands(raw_text, canonical_company=canon_id)

        header_parts = [f"Entity: {c_name}"]
        if ticker:
            header_parts.append(f"Ticker: {ticker}")
        header_parts.append(f"Filing: {doc_name}")
        if year:
            header_parts.append(f"Fiscal Year: {year}")
        if detected_brands:
            header_parts.append(f"Products/Brands: {', '.join(detected_brands)}")

        return f"[CORPORATE CONTEXT | {' | '.join(header_parts)}]\n"

    def ground_chunk(self, doc_name: str, text: str) -> str:
        """
        Applies offline entity grounding to a chunk.
        - Idempotent: avoids duplicate headers.
        - Preserves existing [TABLE CONTINUATION | ...] if present.
        """
        if not text:
            return text

        # Idempotence check
        if text.startswith("[CORPORATE CONTEXT |"):
            return text

        header = self.build_corporate_header(doc_name, text)

        # Prepend header to existing content (preserving any [TABLE CONTINUATION | ...] headers)
        return header + text
