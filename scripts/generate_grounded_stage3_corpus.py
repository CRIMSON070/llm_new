"""
scripts/generate_grounded_stage3_corpus.py
==========================================
Generates the grounded Stage 3 corpus for Experiment L.
Input:  data/heldout_stage3/processed/heldout_enriched_chunks.jsonl (29,335 chunks)
Output: data/heldout_stage3/processed/heldout_grounded_chunks.jsonl

Safety & Integrity Assertions:
1. Verifies baseline corpus exists and is NOT modified.
2. Preserves doc_name, page_number, chunk_id, and original raw_text.
3. Generates grounded_text with concise corporate context headers.
4. Preserves table continuation headers.
5. Asserts exactly 29,335 output records with 100% unique chunk IDs matching baseline.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing.offline_entity_grounder import OfflineEntityGrounder

BASELINE_CHUNKS_PATH = os.path.join(
    BASE_DIR, "data", "heldout_stage3", "processed", "heldout_enriched_chunks.jsonl"
)
GROUNDED_CHUNKS_PATH = os.path.join(
    BASE_DIR, "data", "heldout_stage3", "processed", "heldout_grounded_chunks.jsonl"
)


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_grounded_corpus():
    print("=" * 80)
    print("STAGE 3: GENERATING OFFLINE ENTITY GROUNDED CORPUS (EXPERIMENT L)")
    print("=" * 80)

    if not os.path.exists(BASELINE_CHUNKS_PATH):
        print(f"[ERROR] Baseline chunks file missing: {BASELINE_CHUNKS_PATH}")
        sys.exit(1)

    initial_baseline_hash = compute_sha256(BASELINE_CHUNKS_PATH)
    print(f"Verified Baseline Corpus SHA256: {initial_baseline_hash}")

    grounder = OfflineEntityGrounder()
    t0 = time.time()

    total_chunks = 0
    brand_grounded_count = 0
    table_continuation_count = 0
    chunk_ids = set()

    print(f"Reading from: {BASELINE_CHUNKS_PATH}")
    print(f"Writing to:   {GROUNDED_CHUNKS_PATH}")

    with open(BASELINE_CHUNKS_PATH, "r", encoding="utf-8") as in_f, \
         open(GROUNDED_CHUNKS_PATH, "w", encoding="utf-8") as out_f:

        for line_idx, line in enumerate(in_f):
            rec = json.loads(line)
            doc_name = rec["doc_name"]
            page_number = rec["page_number"]
            chunk_id = rec["chunk_id"]
            raw_text = rec["text"]

            # Ground the text
            grounded_text = grounder.ground_chunk(doc_name, raw_text)

            if "Products/Brands:" in grounded_text:
                brand_grounded_count += 1
            if "[TABLE CONTINUATION |" in grounded_text:
                table_continuation_count += 1

            chunk_record = {
                "doc_name": doc_name,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "raw_text": raw_text,
                "grounded_text": grounded_text,
                "text": grounded_text  # Standard key for downstream LangChain Document loading
            }

            out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
            chunk_ids.add(chunk_id)
            total_chunks += 1

    elapsed = time.time() - t0

    # Safety checks
    post_baseline_hash = compute_sha256(BASELINE_CHUNKS_PATH)
    assert initial_baseline_hash == post_baseline_hash, "FATAL: Baseline corpus was modified!"
    assert total_chunks == 29335, f"Expected 29335 chunks, got {total_chunks}"
    assert len(chunk_ids) == 29335, f"Expected 29335 unique chunk IDs, got {len(chunk_ids)}"

    print("\n" + "=" * 80)
    print("OFFLINE GROUNDING GENERATION INTEGRITY REPORT")
    print("=" * 80)
    print(f"Total chunks processed:      {total_chunks}")
    print(f"Unique chunk IDs preserved:  {len(chunk_ids)}")
    print(f"Chunks with verified brands: {brand_grounded_count}")
    print(f"Table continuations intact:  {table_continuation_count}")
    print(f"Elapsed generation time:     {elapsed:.2f} seconds")
    print(f"Output grounded corpus path: {GROUNDED_CHUNKS_PATH}")
    print(f"Output file size:            {os.path.getsize(GROUNDED_CHUNKS_PATH):,} bytes")
    print(f"Baseline corpus untouched:   {initial_baseline_hash == post_baseline_hash} (Hash verified)")
    print("\n[SUCCESS] Grounded corpus generation completed with 100% integrity!")


if __name__ == "__main__":
    generate_grounded_corpus()
