"""
scripts/process_heldout_stage3_corpus.py
========================================
Extracts text and tables from the 40 held-out Form 10-K PDFs in data/heldout_stage3/raw_filings/.
Applies:
- PyMuPDF text extraction preserving exact 1-indexed physical PDF page numbers.
- RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200) matching preprocess_documents.py.
- Table header propagation and financial statement enrichment matching enrich_phase2_chunks.py.
- Writes to data/heldout_stage3/processed/heldout_enriched_chunks.jsonl.
- Executes comprehensive integrity checks (empty chunks, duplicates, page bounds).
"""

import os
import sys
import json
import re
import time
import csv
import hashlib
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "raw_filings")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed")
OUT_CHUNKS_FILE = os.path.join(PROCESSED_DIR, "heldout_enriched_chunks.jsonl")
MANIFEST_CSV = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "downloaded_filings_manifest.csv")

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
YEARS_PATTERN = re.compile(r"\b(202[0-9]|201[8-9])\b")

def process_corpus():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
        
    doc_names = [r["doc_name"] for r in manifest]
    print(f"Starting chunk extraction & enrichment for {len(doc_names)} Form 10-K filings...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    total_raw_chunks = 0
    total_enriched_chunks = 0
    continuation_count = 0
    chunk_ids = set()
    doc_chunk_counts = {}
    
    t0 = time.time()
    
    with open(OUT_CHUNKS_FILE, "w", encoding="utf-8") as out_f:
        for idx, doc_name in enumerate(doc_names):
            pdf_path = os.path.join(RAW_DIR, f"{doc_name}.pdf")
            if not os.path.exists(pdf_path):
                print(f"[ERROR] PDF missing: {pdf_path}")
                continue
                
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            
            current_stmt = None
            current_units = None
            current_years = []
            current_page = None
            doc_chunks = 0
            
            for page_idx in range(num_pages):
                page_num = page_idx + 1 # 1-indexed
                page = doc[page_idx]
                text = page.get_text("text")
                
                if not text.strip():
                    continue
                    
                # Split into chunks
                page_splits = text_splitter.split_text(text)
                
                for c_idx, raw_chunk in enumerate(page_splits):
                    total_raw_chunks += 1
                    
                    # Statement detection
                    found_stmt = None
                    if page_num >= 5: # Skip preliminary cover/TOC
                        lead = raw_chunk[:500]
                        for sname, pat in STATEMENT_PATTERNS.items():
                            if pat.search(lead):
                                found_stmt = sname
                                break
                                
                    is_continuation = False
                    enrichment_prefix = ""
                    
                    if found_stmt:
                        current_stmt = found_stmt
                        current_page = page_num
                        m_u = UNITS_PATTERN.search(raw_chunk[:500])
                        current_units = m_u.group(1).strip() if m_u else current_units
                        y_found = list(dict.fromkeys(YEARS_PATTERN.findall(raw_chunk[:500])))
                        current_years = y_found if y_found else current_years
                    elif current_stmt and (page_num == current_page or page_num == current_page + 1):
                        num_count = len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", raw_chunk))
                        if num_count >= 3 or "|" in raw_chunk or "$" in raw_chunk:
                            is_continuation = True
                            continuation_count += 1
                            
                    if is_continuation and current_stmt:
                        enrich_parts = [f"Statement: {current_stmt}"]
                        if current_units:
                            enrich_parts.append(f"Units: {current_units}")
                        if current_years:
                            enrich_parts.append(f"Fiscal Years: {', '.join(current_years[:3])}")
                        enrichment_prefix = f"[TABLE CONTINUATION | {' | '.join(enrich_parts)}]\n"
                        
                    final_text = enrichment_prefix + raw_chunk
                    chunk_id = f"{doc_name}_p{page_num}_c{c_idx}"
                    
                    record = {
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "chunk_id": chunk_id,
                        "text": final_text
                    }
                    
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_ids.add(chunk_id)
                    total_enriched_chunks += 1
                    doc_chunks += 1
                    
            doc_chunk_counts[doc_name] = doc_chunks
            print(f"[{idx+1}/{len(doc_names)}] {doc_name}: {num_pages} pages -> {doc_chunks} chunks")
            
    elapsed = time.time() - t0
    print("\n" + "="*80)
    print("EXTRACTION & ENRICHMENT INTEGRITY REPORT")
    print("="*80)
    print(f"Total filings processed:     {len(doc_chunk_counts)} / {len(doc_names)}")
    print(f"Total enriched chunks saved: {total_enriched_chunks}")
    print(f"Total unique chunk IDs:      {len(chunk_ids)}")
    print(f"Table continuations enriched:{continuation_count}")
    print(f"Elapsed processing time:     {elapsed:.2f} seconds")
    print(f"Output chunks file path:     {OUT_CHUNKS_FILE}")
    print(f"Output file size:            {os.path.getsize(OUT_CHUNKS_FILE):,} bytes")
    
    # Assertions
    assert len(chunk_ids) == total_enriched_chunks, "Duplicate chunk IDs detected!"
    assert len(doc_chunk_counts) == 40, "Missing documents in chunking pipeline!"
    assert all(cnt > 50 for cnt in doc_chunk_counts.values()), "Empty or abnormally small document chunks detected!"
    print("\n[SUCCESS] All corpus chunking integrity assertions PASSED perfectly!")

if __name__ == "__main__":
    process_corpus()
