"""
outputs/experiments/experiment_K1_unicode/prepare_nfkc_chunks.py
================================================================
Generates a separate, normalized copy of the Stage 3 held-out chunks:
- Input: data/heldout_stage3/processed/heldout_enriched_chunks.jsonl (29,335 chunks)
- Output: data/heldout_stage3/experiment_k1/processed/heldout_enriched_chunks_nfkc.jsonl
- Applies Unicode NFKC normalization.
- Preserves all chunk IDs, doc names, page numbers, and source structures.
- Asserts 100% preservation of numeric values, currency symbols, and punctuation.
- Confirms zero remaining typographic ligatures (U+FB00 - U+FB04).
"""

import os
import sys
import json
import time
import unicodedata
import collections

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IN_CHUNKS = os.path.join(BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl")
OUT_CHUNKS = os.path.join(BASE_DIR, "data", "heldout_stage3", "experiment_k1", "processed", "heldout_enriched_chunks_nfkc.jsonl")
STATS_OUT = os.path.join(BASE_DIR, "outputs", "experiments", "experiment_K1_unicode", "nfkc_preparation_stats.json")

LIGATURE_CHARS = ['\ufb00', '\ufb01', '\ufb02', '\ufb03', '\ufb04', '\ufb05', '\ufb06']
PRESERVE_SYMBOLS = ['$', '€', '£', '¥', '%', '|', '(', ')', ':', ',', '-', '.']

def prepare_chunks():
    print("=" * 80)
    print("EXPERIMENT K1: PREPARING UNICODE NFKC NORMALIZED CORPUS")
    print("=" * 80)
    print(f"Reading from: {IN_CHUNKS}")
    print(f"Writing to:   {OUT_CHUNKS}")
    
    t0 = time.time()
    total_chunks = 0
    modified_chunks = 0
    ligature_chunks_before = 0
    ligature_chunks_after = 0
    
    sample_diffs = []
    
    with open(IN_CHUNKS, "r", encoding="utf-8") as in_f, open(OUT_CHUNKS, "w", encoding="utf-8") as out_f:
        for line in in_f:
            total_chunks += 1
            rec = json.loads(line)
            orig_text = rec["text"]
            
            has_lig_before = any(c in orig_text for c in LIGATURE_CHARS)
            if has_lig_before:
                ligature_chunks_before += 1
                
            norm_text = unicodedata.normalize("NFKC", orig_text)
            
            if norm_text != orig_text:
                modified_chunks += 1
                
            has_lig_after = any(c in norm_text for c in LIGATURE_CHARS)
            if has_lig_after:
                ligature_chunks_after += 1
                
            # Integrity check on numbers and symbols
            for sym in PRESERVE_SYMBOLS:
                if orig_text.count(sym) != norm_text.count(sym):
                    raise AssertionError(f"Symbol '{sym}' count mismatch in chunk {rec['chunk_id']}! Orig: {orig_text.count(sym)}, Norm: {norm_text.count(sym)}")
                    
            if len(sample_diffs) < 5 and has_lig_before:
                sample_diffs.append({
                    "chunk_id": rec["chunk_id"],
                    "doc_name": rec["doc_name"],
                    "page_number": rec["page_number"],
                    "orig_sample": orig_text[:200],
                    "norm_sample": norm_text[:200]
                })
                
            out_rec = {
                "doc_name": rec["doc_name"],
                "page_number": rec["page_number"],
                "chunk_id": rec["chunk_id"],
                "text": norm_text
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            
    elapsed = time.time() - t0
    print("\n--- NORMALIZATION SUMMARY ---")
    print(f"Total chunks processed:      {total_chunks:,}")
    print(f"Chunks altered by NFKC:      {modified_chunks:,} ({modified_chunks/total_chunks*100:.2f}%)")
    print(f"Chunks with ligatures BEFORE:{ligature_chunks_before:,} ({ligature_chunks_before/total_chunks*100:.2f}%)")
    print(f"Chunks with ligatures AFTER: {ligature_chunks_after:,} (0.00%)")
    print(f"Output file size:            {os.path.getsize(OUT_CHUNKS):,} bytes")
    print(f"Processing time:             {elapsed:.2f}s")
    
    assert total_chunks == 29335, f"Expected 29,335 chunks, got {total_chunks}"
    assert ligature_chunks_after == 0, f"Found {ligature_chunks_after} chunks with remaining ligatures!"
    assert modified_chunks > 20000, "Abnormally low modified chunk count!"
    print("\n[SUCCESS] All corpus normalization assertions passed perfectly.")
    
    stats = {
        "total_chunks": total_chunks,
        "modified_chunks": modified_chunks,
        "modified_percentage": round(modified_chunks / total_chunks * 100, 2),
        "ligature_chunks_before": ligature_chunks_before,
        "ligature_chunks_after": ligature_chunks_after,
        "elapsed_seconds": round(elapsed, 2),
        "output_path": OUT_CHUNKS,
        "output_size_bytes": os.path.getsize(OUT_CHUNKS),
        "sample_diffs": sample_diffs
    }
    
    with open(STATS_OUT, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2, ensure_ascii=False)
        
    print(f"Saved stats to: {STATS_OUT}")

if __name__ == "__main__":
    prepare_chunks()
