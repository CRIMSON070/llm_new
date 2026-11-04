import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import json
from pathlib import Path
from outputs.experiments.experiment_l7_context_window_stitching.context_window_stitcher import ContextWindowStitcher

targets = ['stage3_heldout_0075', 'stage3_heldout_0074', 'stage3_heldout_0099', 'stage3_heldout_0100', 'stage3_heldout_0014', 'stage3_heldout_0025']

grounded_corpus_path = "data/heldout_stage3/processed/heldout_grounded_chunks.jsonl"
chunk_store = {json.loads(line)["chunk_id"]: json.loads(line) for line in open(grounded_corpus_path, "r", encoding="utf-8")}

stitcher = ContextWindowStitcher(chunk_store=chunk_store, max_chars=1200)

v2_path = "outputs/experiments/experiment_l5_ground_truth_v2/stage3_heldout_questions_v2.jsonl"
v2_questions = {json.loads(line)["question_id"]: json.loads(line) for line in open(v2_path, "r", encoding="utf-8") if json.loads(line)["question_id"] in targets}

print("Loaded target questions.")

for qid in targets:
    q = v2_questions[qid]
    print(f"\n=======================================================")
    print(f"QUESTION ID: {qid}")
    print(f"Question: {q['question']}")
    ev_sources = q.get("evidence_sources", [])
    direct_cids = [cid for e in ev_sources if e.get("evidence_type") == "direct_evidence" for cid in e.get("matching_chunk_ids", [])]
    print(f"Direct Gold Chunk IDs: {direct_cids}")
    
    for cid in direct_cids[:2]:
        chunk = chunk_store.get(cid)
        if not chunk:
            continue
        prev_cid, next_cid = stitcher.get_neighboring_chunk_ids(cid)
        stitched = stitcher.stitch_candidate(chunk)
        print(f"\n  Chunk: {cid}")
        print(f"    Preceding neighbor: {prev_cid}")
        print(f"    Succeeding neighbor: {next_cid}")
        print(f"    Neighbors included: {stitched.get('stitched_neighbors')}")
        print(f"    Orig len: {len(chunk.get('text', ''))} -> Stitched len: {len(stitched.get('stitched_text', ''))}")
        print(f"    Stitched preview: {repr(stitched.get('stitched_text', '')[:180])} ...")
