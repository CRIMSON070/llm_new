"""
FinSage Phase 2: Table Header Propagation and Ingestion Layer Enrichment (WP1)
=============================================================================
Detects table bisections in financial statements and prepends document metadata,
statement titles, reporting units, and fiscal periods to continuation chunks.
Builds the Phase 2 BM25 index in data/financebench/bm25_index_phase2.
"""

import os
import sys
import json
import re
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_core.documents import Document
from src.utils.bm25Retriever import load_from_chroma_and_save, BM25Retriever

STATEMENT_PATTERNS = {
    "Consolidated Statements of Cash Flows": re.compile(
        r"(?:^|\n)\s*(?:[A-Za-z0-9\s,\.\-]+\n\s*)?(?:Consolidated\s+)?Statements?\s+of\s+Cash\s+Flow\s*s?",
        re.IGNORECASE
    ),
    "Consolidated Balance Sheets": re.compile(
        r"(?:^|\n)\s*(?:[A-Za-z0-9\s,\.\-]+\n\s*)?(?:Consolidated\s+)?Balance\s+Shee?\s*ts?",
        re.IGNORECASE
    ),
    "Consolidated Statements of Income": re.compile(
        r"(?:^|\n)\s*(?:[A-Za-z0-9\s,\.\-]+\n\s*)?(?:Consolidated\s+)?Statements?\s+of\s+(?:Income|Operations|Earnings)",
        re.IGNORECASE
    )
}

UNITS_PATTERN = re.compile(
    r"(\(dollars\s+in\s+[a-z]+\)|\(in\s+[a-z]+\)|\([a-z]+\s+in\s+[a-z]+\)|\(millions\)|\(thousands\)|\(billions\)|\bin\s+usd\s+(?:millions|billions|thousands)\b)",
    re.IGNORECASE
)
YEARS_PATTERN = re.compile(r"\b(201\d|202\d)\b")


def format_company_name(doc_name: str) -> str:
    comp = doc_name.split("_")[0]
    if comp.upper() == "ULTABEAUTY":
        return "Ulta Beauty"
    elif comp.upper() == "3M":
        return "3M Company"
    elif comp.upper() == "ADOBE":
        return "Adobe"
    elif comp.upper() == "AMAZON":
        return "Amazon"
    elif comp.upper() == "MICROSOFT":
        return "Microsoft"
    elif comp.upper() == "JPMORGAN":
        return "JPMorgan Chase"
    return comp


def enrich_chunks():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    input_file = os.path.join(data_dir, "processed", "parsed_chunks.jsonl")
    output_file = os.path.join(data_dir, "processed", "phase2_enriched_chunks.jsonl")
    bm25_phase2_dir = os.path.join(data_dir, "bm25_index_phase2")

    print("=" * 80)
    print("FinSage Phase 2: Table Header Propagation & Chunk Enrichment")
    print("=" * 80)
    print(f"Reading chunks from: {input_file}")

    enriched_records = []
    continuation_count = 0

    current_doc = None
    current_stmt = None
    current_units = None
    current_years = []
    current_page = None

    t0 = time.time()
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            doc = c["doc_name"]
            page = c["page_number"]
            text = c["text"]
            cid = c["chunk_id"]

            if doc != current_doc:
                current_doc = doc
                current_stmt = None
                current_units = None
                current_years = []
                current_page = None

            # Skip table of contents and cover pages (< page 5) from propagating statements
            found_stmt = None
            if page >= 5:
                text_lead = text[:500]
                for sname, pat in STATEMENT_PATTERNS.items():
                    if pat.search(text_lead):
                        found_stmt = sname
                        break

            is_continuation = False
            enrichment_header = ""

            if found_stmt:
                current_stmt = found_stmt
                current_page = page
                m_u = UNITS_PATTERN.search(text[:500])
                current_units = m_u.group(1).strip() if m_u else None
                y_found = list(dict.fromkeys(YEARS_PATTERN.findall(text[:500])))
                current_years = y_found if y_found else current_years
            elif current_stmt and (page == current_page or page == current_page + 1):
                # Continuation chunk on same page or immediate next page of statement table
                num_count = len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text))
                if num_count >= 3 or "|" in text or "$" in text:
                    is_continuation = True
                    continuation_count += 1
                    comp_name = format_company_name(doc)
                    years_str = ", ".join(current_years) if current_years else ""
                    units_str = current_units if current_units else "USD"
                    
                    header_parts = [f"Filing: {doc} ({comp_name})", f"Statement: {current_stmt}"]
                    if units_str:
                        header_parts.append(f"Units: {units_str}")
                    if years_str:
                        header_parts.append(f"Periods: {years_str}")
                        
                    enrichment_header = f"[{' | '.join(header_parts)}]\n"
            else:
                current_stmt = None

            new_text = enrichment_header + text if is_continuation else text
            enriched_record = dict(c)
            enriched_record["text"] = new_text
            enriched_record["is_table_continuation"] = is_continuation
            if is_continuation:
                enriched_record["propagated_statement"] = current_stmt
                enriched_record["propagated_header"] = enrichment_header.strip()

            enriched_records.append(enriched_record)

    t_read = round(time.time() - t0, 2)
    print(f"Processed {len(enriched_records)} chunks in {t_read}s.")
    print(f"Identified and enriched {continuation_count} bisected table continuation chunks.")

    # Write enriched chunks to output file
    print(f"Writing enriched chunks to: {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out:
        for r in enriched_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Enriched chunks written successfully.")

    # Build Phase 2 BM25 Index
    print(f"\nBuilding Phase 2 BM25 Index in: {bm25_phase2_dir}...")
    t_bm0 = time.time()
    docs = []
    for r in enriched_records:
        metadata = {
            "doc_name": r["doc_name"],
            "page_number": r["page_number"],
            "doc_id": r["chunk_id"],
            "is_continuation": r.get("is_table_continuation", False)
        }
        docs.append(Document(page_content=r["text"], metadata=metadata))

    os.makedirs(bm25_phase2_dir, exist_ok=True)
    load_from_chroma_and_save(docs, bm25_phase2_dir)
    t_bm = round(time.time() - t_bm0, 2)
    print(f"Phase 2 BM25 Index built in {t_bm}s.")


if __name__ == "__main__":
    enrich_chunks()
