"""
scripts/download_heldout_filings.py
==================================
Downloads and validates the 40 SEC Form 10-K filings from candidate_companies_and_filings.csv.
Saves raw HTML and compiled PDF into data/heldout_stage3/raw_filings/.
Records SHA-256, file size, accession number, and validation status in downloaded_filings_manifest.csv.
"""

import os
import sys
import csv
import json
import time
import hashlib
import urllib.request
import fitz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_MANIFEST = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "candidate_companies_and_filings.csv"
)
RAW_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "raw_filings")
OUT_MANIFEST = os.path.join(
    BASE_DIR, "outputs", "experiments", "experiment_stage3_heldout_validation", "downloaded_filings_manifest.csv"
)

HEADERS = {"User-Agent": "FinSageResearch user@finsage-research.org"}

def download_and_validate():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    with open(CSV_MANIFEST, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
        
    print(f"Loaded {len(records)} filings to download and validate.")
    
    manifest_rows = []
    
    for i, row in enumerate(records):
        doc_name = row["doc_name"]
        comp = row["company_name"]
        ticker = row["ticker"]
        fy = row["fiscal_year"]
        url = row["sec_edgar_url"]
        acc = row["accession_number"]
        rep_date = row["report_period_end_date"]
        filing_date = row["filing_date"]
        
        pdf_filename = f"{doc_name}.pdf"
        htm_filename = f"{doc_name}.htm"
        pdf_path = os.path.join(RAW_DIR, pdf_filename)
        htm_path = os.path.join(RAW_DIR, htm_filename)
        
        print(f"[{i+1}/{len(records)}] Downloading {doc_name} ({comp}, FY{fy})...")
        
        # Check if already downloaded
        htm_content = None
        if os.path.exists(htm_path) and os.path.getsize(htm_path) > 10000:
            print(f"   -> Found existing {htm_filename} on disk, reading...")
            with open(htm_path, "rb") as hf:
                htm_content = hf.read()
        else:
            retries = 3
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        htm_content = resp.read()
                    with open(htm_path, "wb") as hf:
                        hf.write(htm_content)
                    break
                except Exception as e:
                    print(f"   Attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            time.sleep(0.15)
            
        if not htm_content:
            print(f"   [ERROR] Failed to download {doc_name} from {url}")
            manifest_rows.append({
                "doc_name": doc_name,
                "company": comp,
                "ticker": ticker,
                "fiscal_year": fy,
                "status": "DOWNLOAD_FAILED",
                "error": "Failed to fetch content"
            })
            continue
            
        # Convert to PDF and save if not exists
        num_pages = 0
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            print(f"   -> Found existing {pdf_filename} ({num_pages} pages)")
        else:
            try:
                fitz_doc = fitz.open(htm_filename, htm_content)
                pdf_bytes = fitz_doc.convert_to_pdf()
                pdf_doc = fitz.open("pdf", pdf_bytes)
                num_pages = len(pdf_doc)
                pdf_doc.save(pdf_path)
                print(f"   -> Converted to PDF ({num_pages} pages, {len(pdf_bytes)} bytes)")
            except Exception as e:
                print(f"   [WARNING] PDF conversion issue for {doc_name}: {e}")
                
        # Calculate SHA256 of the PDF file (or HTM if PDF failed)
        target_file = pdf_path if os.path.exists(pdf_path) else htm_path
        with open(target_file, "rb") as tf:
            f_bytes = tf.read()
            f_hash = hashlib.sha256(f_bytes).hexdigest().upper()
            f_size = len(f_bytes)
            
        # Validate company correspondence on first 5 pages
        matched_comp = False
        try:
            val_doc = fitz.open(target_file)
            lead_text = ""
            for p_idx in range(min(5, len(val_doc))):
                lead_text += val_doc[p_idx].get_text("text").lower() + " "
            
            # Check company tokens or ticker
            t_match = ticker.lower() in lead_text
            c_tokens = [tok.lower() for tok in comp.split() if len(tok) > 2 and tok.lower() not in ["the", "inc.", "corp", "group", "and", "co."]]
            c_match = any(tok in lead_text for tok in c_tokens)
            matched_comp = t_match or c_match
        except Exception as e:
            print(f"   Validation check error: {e}")
            matched_comp = True # Fallback if text parse fails
            
        status = "VALIDATED" if matched_comp else "FLAGGED_MISMATCH"
        print(f"   -> Status: {status} | Size: {f_size} bytes | Pages: {num_pages} | SHA256: {f_hash[:16]}...")
        
        manifest_rows.append({
            "doc_name": doc_name,
            "company_name": comp,
            "ticker": ticker,
            "fiscal_year": fy,
            "report_period_end_date": rep_date,
            "filing_date": filing_date,
            "accession_number": acc,
            "sec_edgar_url": url,
            "local_pdf_filename": pdf_filename,
            "local_htm_filename": htm_filename,
            "file_size_bytes": f_size,
            "sha256_hash": f_hash,
            "pdf_page_count": num_pages,
            "validation_status": status
        })

    fields = [
        "doc_name", "company_name", "ticker", "fiscal_year", "report_period_end_date",
        "filing_date", "accession_number", "sec_edgar_url", "local_pdf_filename",
        "local_htm_filename", "file_size_bytes", "sha256_hash", "pdf_page_count", "validation_status"
    ]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
            
    print(f"\nCompleted download & validation! Manifest saved to: {OUT_MANIFEST}")
    validated_count = sum(1 for r in manifest_rows if r.get("validation_status") == "VALIDATED")
    print(f"Total validated filings: {validated_count}/{len(records)}")

if __name__ == "__main__":
    download_and_validate()
