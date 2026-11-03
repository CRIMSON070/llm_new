"""
Build Heading-Summary Metadata Index (Path 3 of Multi-Path Retrieval)
=====================================================================
Paper Specifications: FinSage (arXiv:2504.14493, CIKM 2025) Section 3.1 (Appendix A.7) & Section 3.2.

Extracts document structure and financial statements across FinanceBench filings,
generates descriptive heading summaries, and builds a dedicated BGE-M3 FAISS index
where each metadata entry maps to its constituent chunk IDs.
"""

import os
import sys
import re
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

SECTION_RULES = [
    # Primary Financial Statements
    ("Statement of Cash Flows", r"(consolidated\s+statement[s]?\s+of\s+cash\s+flows?|condensed\s+consolidated\s+statement[s]?\s+of\s+cash\s+flows?)", "Consolidated Statement of Cash Flows detailing operating activities, investing activities, capital expenditures (PP&E), and financing activities."),
    ("Balance Sheet", r"(consolidated\s+balance\s+sheets?|condensed\s+consolidated\s+balance\s+sheets?)", "Consolidated Balance Sheets detailing assets (cash, inventories, property plant and equipment PP&E), liabilities, and shareholders equity."),
    ("Income Statement", r"(consolidated\s+statement[s]?\s+of\s+income|consolidated\s+statement[s]?\s+of\s+operations?|consolidated\s+statement[s]?\s+of\s+earnings)", "Consolidated Statements of Income and Operations detailing net sales, revenues, operating expenses, operating income, and net earnings."),
    ("Statement of Stockholders Equity", r"(consolidated\s+statement[s]?\s+of\s+(changes\s+in\s+)?equity|consolidated\s+statement[s]?\s+of\s+stockholders['\s]+equity)", "Consolidated Statement of Stockholders' Equity detailing common stock, retained earnings, and equity rollforward."),
    ("Statement of Comprehensive Income", r"(consolidated\s+statement[s]?\s+of\s+comprehensive\s+income)", "Consolidated Statement of Comprehensive Income detailing net income and other comprehensive income items."),

    # Key SEC Items
    ("Item 7: MD&A", r"(item\s+7[\.\:\s]+management['\s]+s\s+discussion\s+and\s+analysis)", "Item 7 Management's Discussion and Analysis of Financial Condition and Results of Operations, liquidity, capital resources, and non-GAAP measures."),
    ("Item 8: Financial Statements", r"(item\s+8[\.\:\s]+financial\s+statements\s+and\s+supplementary\s+data)", "Item 8 Financial Statements and Supplementary Data index and auditor report."),
    ("Item 1: Business", r"(item\s+1[\.\:\s]+business)", "Item 1 Business description, operational segments, and principal products."),
    ("Item 1A: Risk Factors", r"(item\s+1a[\.\:\s]+risk\s+factors)", "Item 1A Risk Factors impacting business operations and financial condition."),
    ("Cover Page: Registered Securities", r"(securities\s+registered\s+pursuant\s+to\s+section\s+12\(b\)|title\s+of\s+each\s+class)", "Cover page Form 10-K registered securities, common stock, trading symbol, and exchange registration."),

    # Key Notes
    ("Note 1: Accounting Policies", r"(note\s+1[\.\:\s]+significant\s+accounting\s+policies|note\s+1[\.\:\s]+summary\s+of\s+significant)", "Note 1 Summary of Significant Accounting Policies, basis of presentation, and revenue recognition."),
    ("Note 2: Revenues / Segments", r"(note\s+2[\.\:\s]+revenue|note\s+2[\.\:\s]+segment)", "Note 2 Disaggregated revenues, commercial terms, and reporting categories."),
    ("Note 3: Acquisitions & Divestitures", r"(note\s+3[\.\:\s]+acquisition|note\s+3[\.\:\s]+business\s+combination)", "Note 3 Acquisitions, business combinations, mergers, purchase price allocation, and divestitures."),
    ("Note 4: Goodwill & Intangibles", r"(note\s+4[\.\:\s]+goodwill|note\s+4[\.\:\s]+intangible)", "Note 4 Goodwill and intangible assets, impairments, and amortizable lives."),
    ("Note 5: Inventories / Restructuring", r"(note\s+5[\.\:\s]+inventor|note\s+5[\.\:\s]+restructuring)", "Note 5 Merchandise inventories, inventory valuation, and restructuring exit actions."),
    ("Note 12: Debt & Borrowings", r"(note\s+12[\.\:\s]+long[\s\-]term\s+debt|note\s+12[\.\:\s]+debt)", "Note 12 Long-term debt, credit facilities, short-term borrowings, maturities, and interest rates."),
    ("Note 13: Pension & Benefits", r"(note\s+13[\.\:\s]+pension|note\s+13[\.\:\s]+postretirement)", "Note 13 Pension and postretirement benefit plans, actuarial assumptions, and funding status."),
    ("Note 16: Commitments & Contingencies", r"(note\s+16[\.\:\s]+commitment|note\s+16[\.\:\s]+contingenc)", "Note 16 Commitments, legal proceedings, guarantees, and environmental contingent liabilities.")
]


def extract_filing_sections(parsed_chunks_file: str) -> List[Dict[str, Any]]:
    """
    Groups chunks into logical sections and statements across all filings.
    """
    print(f"Reading chunks from {parsed_chunks_file}...")
    chunks_by_doc_page = defaultdict(lambda: defaultdict(list))
    
    with open(parsed_chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            doc = c["doc_name"]
            page = c["page_number"]
            chunks_by_doc_page[doc][page].append(c)

    sections = []
    
    for doc_name, pages in chunks_by_doc_page.items():
        # First page is always Cover Page
        if 1 in pages:
            cover_cids = [c["chunk_id"] for c in pages[1]]
            sections.append({
                "section_id": f"{doc_name}_cover",
                "doc_name": doc_name,
                "section_title": "Cover Page and Registered Securities",
                "summary": f"{doc_name} Cover Page Form 10-K/10-Q filing details, commission file number, registered securities under Section 12(b), common stock, and market exchanges.",
                "page_start": 1,
                "page_end": 1,
                "chunk_ids": cover_cids
            })

        # Scan each page for section headers
        for page_num in sorted(pages.keys()):
            page_chunks = pages[page_num]
            full_page_text = " ".join([c.get("text", "") for c in page_chunks])
            
            for title, pattern, desc in SECTION_RULES:
                if re.search(pattern, full_page_text, re.IGNORECASE):
                    cids = [c["chunk_id"] for c in page_chunks]
                    # Also include next page if it continues the statement (multi-page table)
                    next_page = page_num + 1
                    if next_page in pages:
                        next_page_text = " ".join([c.get("text", "") for c in pages[next_page]])
                        # If next page doesn't start a brand new major section, incorporate as continuation
                        starts_new = any(re.search(pat, next_page_text[:300], re.IGNORECASE) for _, pat, _ in SECTION_RULES)
                        if not starts_new:
                            cids.extend([c["chunk_id"] for c in pages[next_page]])

                    sections.append({
                        "section_id": f"{doc_name}_p{page_num}_{title.replace(' ', '_').lower()}",
                        "doc_name": doc_name,
                        "section_title": f"{doc_name} {title}",
                        "summary": f"{doc_name} {title}: {desc}",
                        "page_start": page_num,
                        "page_end": page_num + (1 if len(cids) > len(page_chunks) else 0),
                        "chunk_ids": list(dict.fromkeys(cids))
                    })
                    break  # Avoid double tagging on the same page

    print(f"Extracted {len(sections)} structural sections across filings.")
    return sections


def build_and_save_index(sections: List[Dict[str, Any]], output_dir: str):
    """
    Embeds heading summaries with BGE-M3 and creates a FAISS index.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nInitializing BGE-M3 Embeddings for Heading Summaries...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    docs = []
    for s in sections:
        content = f"{s['section_title']}\n{s['summary']}"
        meta = {
            "section_id": s["section_id"],
            "doc_name": s["doc_name"],
            "section_title": s["section_title"],
            "summary": s["summary"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "chunk_ids": s["chunk_ids"],
            "chunk_count": len(s["chunk_ids"])
        }
        docs.append(Document(page_content=content, metadata=meta))

    print(f"Building FAISS index over {len(docs)} heading summaries...")
    faiss_index = FAISS.from_documents(docs, embeddings)
    
    print(f"Saving FAISS metadata index to {output_dir}...")
    faiss_index.save_local(output_dir)
    
    # Save a json catalog for quick inspect
    catalog_path = os.path.join(output_dir, "sections_catalog.json")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2)
        
    print(f"Catalog saved to {catalog_path}. Build complete.")


def main():
    data_dir = os.path.join(BASE_DIR, "data", "financebench")
    parsed_chunks_file = os.path.join(data_dir, "processed", "parsed_chunks.jsonl")
    output_dir = os.path.join(data_dir, "metadata_faiss_index")
    
    sections = extract_filing_sections(parsed_chunks_file)
    build_and_save_index(sections, output_dir)


if __name__ == "__main__":
    main()
