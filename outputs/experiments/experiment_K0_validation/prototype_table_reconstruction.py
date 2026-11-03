"""
outputs/experiments/experiment_K0_validation/prototype_table_reconstruction.py
==============================================================================
Prototype table reconstruction module for Stage 3 held-out filings:
1. Reconstructs multi-column financial tables from sequential text streams.
2. Preserves exact numeric values, dates, line item labels, currency symbols, and signs.
3. Strictly adheres to conservative reconstruction:
   - Does NOT infer missing cells or invent labels.
   - Requires exact matching of column counts across all reconstructed rows.
   - Retains original extracted text when reliable reconstruction is impossible.
4. Generates before-and-after audit samples with PDF page references.
"""

import os
import sys
import re
import json
import unicodedata
from typing import Dict, List, Tuple, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(BASE_DIR, "data", "heldout_stage3", "raw_filings")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_K0_validation")
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")

YEAR_PATTERN = re.compile(r"^(?:202[0-9]|201[8-9])$")
DATE_HEADER_PATTERN = re.compile(
    r"^(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+)?(202[0-9]|201[8-9])$",
    re.IGNORECASE
)

NUMERIC_VALUE_PATTERN = re.compile(
    r"^(?:\$|€|£|¥)?\s*(?:\([0-9,]+(?:\.[0-9]+)?\)|[0-9,]+(?:\.[0-9]+)?|-|—|\bzero\b)\s*%?$",
    re.IGNORECASE
)

def is_numeric_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    if t in ["$", "€", "£", "¥", "%", "-", "—", "—%"]:
        return True
    return bool(NUMERIC_VALUE_PATTERN.match(t))

def clean_token(tok: str) -> str:
    return unicodedata.normalize('NFKC', tok).strip()

def parse_financial_table(lines: List[str]) -> Tuple[Optional[str], str]:
    """
    Attempts to reconstruct a financial table from extracted text lines.
    Returns (reconstructed_markdown, status_reason).
    If reconstruction is unsafe or unsupported, returns (None, reason).
    """
    clean_lines = [clean_token(l) for l in lines if clean_token(l)]
    if len(clean_lines) < 8:
        return None, "INSUFFICIENT_LINES"

    # 1. Locate column headers (e.g. 2, 3, 4, 5 consecutive dates or years)
    header_start_idx = -1
    col_headers = []
    
    for i in range(min(15, len(clean_lines))):
        line = clean_lines[i]
        # Check if line contains a year or date header
        if DATE_HEADER_PATTERN.search(line) or YEAR_PATTERN.search(line) or line in ["Change", "% Change"]:
            # Check if this line or next lines form a group of 2-5 headers
            potential_headers = []
            j = i
            # Look backwards 1 line if date was split across two lines (e.g., 'September 28,' then '2024')
            if i > 0 and any(m in clean_lines[i-1] for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
                combined_first = f"{clean_lines[i-1]} {clean_lines[i]}"
                potential_headers.append(combined_first)
                header_start_idx = i - 1
                j = i + 1
            else:
                header_start_idx = i
                
            while j < min(i + 12, len(clean_lines)):
                h_line = clean_lines[j]
                if DATE_HEADER_PATTERN.search(h_line) or YEAR_PATTERN.search(h_line) or h_line in ["Change", "% Change", "%"]:
                    potential_headers.append(h_line)
                    j += 1
                # Check for two-line date: Month Day, then Year
                elif j + 1 < len(clean_lines) and any(m in h_line for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]) and (YEAR_PATTERN.search(clean_lines[j+1]) or DATE_HEADER_PATTERN.search(clean_lines[j+1])):
                    potential_headers.append(f"{h_line} {clean_lines[j+1]}")
                    j += 2
                else:
                    break

                    
            if len(potential_headers) in [2, 3, 4, 5]:
                col_headers = potential_headers
                header_end_idx = j
                break

    if not col_headers:
        return None, "NO_VALID_COLUMN_HEADERS_FOUND"

    num_cols = len(col_headers)
    
    # 2. Parse body rows following the headers
    body_lines = clean_lines[header_end_idx:]
    rows = []
    
    curr_label = []
    curr_values = []
    
    idx = 0
    while idx < len(body_lines):
        line = body_lines[idx]
        
        # Check if line is a standalone currency symbol or sign
        if line in ["$", "€", "£", "¥"]:
            # Next line should be the number
            if idx + 1 < len(body_lines) and is_numeric_token(body_lines[idx+1]):
                val = f"{line}{body_lines[idx+1]}"
                curr_values.append(val)
                idx += 2
                if len(curr_values) == num_cols and curr_label:
                    rows.append((" ".join(curr_label), list(curr_values)))
                    curr_label = []
                    curr_values = []
                continue
            else:
                # Standalone symbol without number -> abort or label
                curr_label.append(line)
                idx += 1
                continue
                
        # Check if line is a percentage sign attaching to previous
        if line == "%" and curr_values:
            curr_values[-1] = f"{curr_values[-1]}%"
            idx += 1
            if len(curr_values) == num_cols and curr_label:
                rows.append((" ".join(curr_label), list(curr_values)))
                curr_label = []
                curr_values = []
            continue

        # Check if line is numeric value
        if is_numeric_token(line):
            # Check if followed by %
            if idx + 1 < len(body_lines) and body_lines[idx+1] == "%":
                line = f"{line}%"
                idx += 1
            curr_values.append(line)
            idx += 1
            if len(curr_values) == num_cols and curr_label:
                rows.append((" ".join(curr_label), list(curr_values)))
                curr_label = []
                curr_values = []
            continue
            
        # It's a text token / row label
        if curr_values and len(curr_values) < num_cols:
            # Partial row with fewer values than columns!
            # Conservative rule: if an incomplete value list is encountered, cannot assume missing cells!
            return None, f"PARTIAL_ROW_MISMATCH: label='{' '.join(curr_label)}' had {len(curr_values)} values (expected {num_cols})"
            
        curr_label.append(line)
        idx += 1

    # Check remaining
    if curr_values:
        if len(curr_values) == num_cols and curr_label:
            rows.append((" ".join(curr_label), list(curr_values)))
        else:
            return None, f"TRAILING_UNMATCHED_VALUES: {len(curr_values)} values for label '{' '.join(curr_label)}'"

    if len(rows) < 2:
        return None, f"TOO_FEW_ROWS_EXTRACTED: {len(rows)}"

    # 3. Assemble Markdown Table
    md_lines = []
    # Header row
    md_lines.append("| Line Item / Metric | " + " | ".join(col_headers) + " |")
    md_lines.append("| :--- | " + " | ".join([":---:" for _ in col_headers]) + " |")
    for label, vals in rows:
        md_lines.append(f"| {label} | " + " | ".join(vals) + " |")

    return "\n".join(md_lines), "SUCCESS_RECONSTRUCTED"

def run_table_audit_and_prototype():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    import fitz
    
    test_cases = [
        {
            "id": "sample_1_aapl_operations",
            "doc_name": "AAPL_2024_10K",
            "page_num": 68,
            "table_type": "Primary Statement (Operations)",
            "description": "Apple Inc. Consolidated Statements of Operations (FY24, FY23, FY22)"
        },
        {
            "id": "sample_2_aapl_balance_sheet",
            "doc_name": "AAPL_2024_10K",
            "page_num": 74,
            "table_type": "Primary Statement (Balance Sheet)",
            "description": "Apple Inc. Consolidated Balance Sheets (September 28, 2024 & September 30, 2023)"
        },
        {
            "id": "sample_3_aapl_product_performance",
            "doc_name": "AAPL_2024_10K",
            "page_num": 53,
            "table_type": "MD&A / Segment Disclosure",
            "description": "Apple Inc. Net Sales by Category (iPhone, Mac, iPad, Wearables, Services)"
        },
        {
            "id": "sample_4_nvda_income_statement",
            "doc_name": "NVDA_2024_10K",
            "page_num": 113,
            "table_type": "Primary Statement (Income Statement)",
            "description": "NVIDIA Consolidated Statements of Income (Jan 28, 2024, Jan 29, 2023, Jan 30, 2022)"
        },
        {
            "id": "sample_5_nvda_pct_revenue",
            "doc_name": "NVDA_2024_10K",
            "page_num": 84,
            "table_type": "MD&A Tabular Breakdown",
            "description": "NVIDIA Statements of Income as % of Revenue (FY24, FY23)"
        },
        {
            "id": "sample_6_googl_income_statement",
            "doc_name": "GOOGL_2024_10K",
            "page_num": 144,
            "table_type": "Primary Statement (Income Statement)",
            "description": "Alphabet Inc. Consolidated Statements of Income (FY22, FY23, FY24)"
        },
        {
            "id": "sample_7_unsupported_footnote",
            "doc_name": "AAPL_2024_10K",
            "page_num": 80,
            "table_type": "Complex Unstructured Footnote Prose",
            "description": "Apple Inc. Note 1 Summary of Significant Accounting Policies (Prose text with sporadic numbers)"
        }
    ]
    
    results = []
    
    print("=" * 80)
    print("RUNNING PROTOTYPE TABLE RECONSTRUCTION ON REPRESENTATIVE AUDIT SAMPLES")
    print("=" * 80)
    
    for tc in test_cases:
        doc_path = os.path.join(RAW_DIR, f"{tc['doc_name']}.pdf")
        doc = fitz.open(doc_path)
        page = doc[tc["page_num"] - 1]
        raw_text = page.get_text("text")
        lines = raw_text.split("\n")
        
        md_table, status = parse_financial_table(lines)
        
        tc_res = {
            "id": tc["id"],
            "doc_name": tc["doc_name"],
            "page_num": tc["page_num"],
            "table_type": tc["table_type"],
            "description": tc["description"],
            "status": status,
            "raw_line_count": len(lines),
            "reconstructed_table": md_table,
            "raw_text_sample": "\n".join(lines[:30])
        }
        results.append(tc_res)
        
        print(f"\n--- {tc['id'].upper()} ({tc['doc_name']} Page {tc['page_num']}) ---")
        print(f"Type:   {tc['table_type']}")
        print(f"Status: {status}")
        if md_table:
            row_count = len(md_table.split("\n")) - 2
            print(f"Table Reconstructed: {row_count} rows")
            print("Preview:\n" + "\n".join(md_table.split("\n")[:8]))
        else:
            print(f"Safe fallback applied: retained original raw text. (Reason: {status})")
            
        # Save sample markdown file
        sample_file = os.path.join(SAMPLES_DIR, f"{tc['id']}.md")
        with open(sample_file, "w", encoding="utf-8") as sf:
            sf.write(f"# Table Audit Sample: {tc['id']}\n\n")
            sf.write(f"- **Document:** {tc['doc_name']}.pdf\n")
            sf.write(f"- **Physical PDF Page:** {tc['page_num']}\n")
            sf.write(f"- **Type:** {tc['table_type']}\n")
            sf.write(f"- **Description:** {tc['description']}\n")
            sf.write(f"- **Reconstruction Status:** `{status}`\n\n")
            if md_table:
                sf.write("## Reconstructed Markdown Table\n\n")
                sf.write(md_table + "\n\n")
            else:
                sf.write("## Reconstructed Markdown Table\n\n")
                sf.write(f"*Reconstruction safely bypassed: {status}. Original text retained.*\n\n")
            sf.write("## Original Raw Extracted Text (First 35 Lines)\n\n```\n")
            sf.write("\n".join(lines[:35]) + "\n```\n")
            
    summary_path = os.path.join(OUT_DIR, "table_extraction_audit.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[DONE] Table audit results and samples saved to {summary_path} and {SAMPLES_DIR}")
    return results

if __name__ == "__main__":
    run_table_audit_and_prototype()
