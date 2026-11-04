"""
Context Window Stitcher for FinSage Experiment L.8
==================================================

Ablation & Optimization Configurations:
- Arm A: No stitching (frozen L.2 baseline)
- Arm B: Previous + Candidate + Next, max 1,200 chars (reproducing L.7)
- Arm C: Previous + Candidate only, max 1,200 chars (asymmetric backward)
- Arm D: Candidate + Next only, max 1,200 chars (asymmetric forward)
- Arm E: Previous + Candidate + Next, max 800 chars (compact symmetric)

Design Invariants:
1. Strict filing boundary: Predecessor and successor must belong to the exact same doc_name.
2. Candidate identity preservation: Candidate chunk_id and evaluation identity remain strictly preserved.
3. Candidate text priority: Original candidate text is never truncated to accommodate neighbors.
4. Overlap & header hygiene: Leading grounded company headers from neighbors are stripped to prevent redundant token noise.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class OptimizedContextWindowStitcher:
    """
    Configurable context window stitcher supporting symmetric and asymmetric windowing
    with explicit character budgets.
    """

    def __init__(
        self,
        chunk_store: Dict[str, Dict[str, Any]],
        max_chars: int = 1200,
        enable_preceding: bool = True,
        enable_succeeding: bool = True,
    ):
        """
        Args:
            chunk_store: Mapping of chunk_id -> chunk dict.
            max_chars: Total character budget for stitched text.
            enable_preceding: Whether to stitch the immediate preceding chunk.
            enable_succeeding: Whether to stitch the immediate succeeding chunk.
        """
        self.chunk_store = chunk_store
        self.max_chars = max_chars
        self.enable_preceding = enable_preceding
        self.enable_succeeding = enable_succeeding

        self.doc_to_chunks: Dict[str, List[str]] = {}
        self.chunk_id_to_index: Dict[str, Tuple[str, int]] = {}

        self._build_index()

    def _build_index(self):
        doc_lists: Dict[str, List[Tuple[int, int, str]]] = {}

        for cid, chunk in self.chunk_store.items():
            doc = chunk.get("doc_name", "")
            page = chunk.get("page_number", chunk.get("page", 0))

            m = re.search(r"_p(\d+)_c(\d+)$", cid)
            if m:
                p_num = int(m.group(1))
                c_num = int(m.group(2))
            else:
                p_num = int(page) if page else 0
                c_num = 0

            if doc not in doc_lists:
                doc_lists[doc] = []
            doc_lists[doc].append((p_num, c_num, cid))

        for doc, tuples in doc_lists.items():
            tuples.sort(key=lambda t: (t[0], t[1]))
            ordered_cids = [t[2] for t in tuples]
            self.doc_to_chunks[doc] = ordered_cids
            for idx, cid in enumerate(ordered_cids):
                self.chunk_id_to_index[cid] = (doc, idx)

    def get_neighboring_chunk_ids(self, chunk_id: str) -> Tuple[Optional[str], Optional[str]]:
        if chunk_id not in self.chunk_id_to_index:
            return None, None

        doc, idx = self.chunk_id_to_index[chunk_id]
        doc_cids = self.doc_to_chunks[doc]

        prev_cid = doc_cids[idx - 1] if (self.enable_preceding and idx > 0) else None
        next_cid = doc_cids[idx + 1] if (self.enable_succeeding and idx < len(doc_cids) - 1) else None

        return prev_cid, next_cid

    def stitch_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        cid = candidate.get("chunk_id", "")
        orig_text = candidate.get("text", "")
        if not orig_text and "raw_text" in candidate:
            orig_text = candidate.get("raw_text", "")

        stitched_candidate = dict(candidate)
        stitched_candidate["original_text"] = orig_text

        prev_cid, next_cid = self.get_neighboring_chunk_ids(cid)
        prev_chunk = self.chunk_store.get(prev_cid) if prev_cid else None
        next_chunk = self.chunk_store.get(next_cid) if next_cid else None

        prev_text = (prev_chunk.get("text") or prev_chunk.get("raw_text", "")) if prev_chunk else ""
        next_text = (next_chunk.get("text") or next_chunk.get("raw_text", "")) if next_chunk else ""

        budget_remaining = self.max_chars - len(orig_text)
        neighbors_included = []

        if budget_remaining <= 0:
            stitched_candidate["stitched_text"] = orig_text[:self.max_chars]
            stitched_candidate["stitched_neighbors"] = []
            stitched_candidate["is_stitched"] = False
            return stitched_candidate

        def clean_neighbor_text(text: str) -> str:
            return re.sub(r"^\[Company:[^\]]+\]\s*", "", text).strip()

        cleaned_prev = clean_neighbor_text(prev_text) if self.enable_preceding else ""
        cleaned_next = clean_neighbor_text(next_text) if self.enable_succeeding else ""

        # Budget Allocation:
        # If both preceding and succeeding are enabled: allocate 50% to prev, rest to next
        # If only preceding enabled: 100% of remaining budget to prev
        # If only succeeding enabled: 100% of remaining budget to next
        if self.enable_preceding and self.enable_succeeding:
            prev_budget = budget_remaining // 2
        elif self.enable_preceding:
            prev_budget = budget_remaining
        else:
            prev_budget = 0

        pre_snippet = ""
        if cleaned_prev and prev_budget > 20:
            if len(cleaned_prev) > prev_budget:
                pre_snippet = cleaned_prev[-prev_budget:].strip()
                space_idx = pre_snippet.find(" ")
                if space_idx != -1 and space_idx < 30:
                    pre_snippet = pre_snippet[space_idx + 1:].strip()
            else:
                pre_snippet = cleaned_prev
            if pre_snippet:
                neighbors_included.append(prev_cid)

        # Remaining budget for successor
        post_budget = self.max_chars - len(orig_text) - len(pre_snippet) - 4
        post_snippet = ""
        if cleaned_next and post_budget > 20:
            if len(cleaned_next) > post_budget:
                post_snippet = cleaned_next[:post_budget].strip()
                rspace_idx = post_snippet.rfind(" ")
                if rspace_idx != -1 and rspace_idx > len(post_snippet) - 30:
                    post_snippet = post_snippet[:rspace_idx].strip()
            else:
                post_snippet = cleaned_next
            if post_snippet:
                neighbors_included.append(next_cid)

        # Assemble stitched text
        parts = []
        if pre_snippet:
            parts.append(pre_snippet)
        parts.append(orig_text)
        if post_snippet:
            parts.append(post_snippet)

        final_stitched = " ...\n".join(parts) if (pre_snippet or post_snippet) else orig_text

        stitched_candidate["stitched_text"] = final_stitched
        stitched_candidate["stitched_neighbors"] = neighbors_included
        stitched_candidate["is_stitched"] = len(neighbors_included) > 0

        return stitched_candidate

    def stitch_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.stitch_candidate(c) for c in candidates]
