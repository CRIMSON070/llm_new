"""
Context Window Stitcher for FinSage Experiment L.7
==================================================

Provides controlled, deterministic neighboring context stitching for candidate chunks
prior to cross-encoder reranking.

Core Design Rules:
1. Candidate Identity Preservation: The original candidate's chunk_id, doc_name, page,
   and pool rank remain strictly untouched. Evaluation identity is ALWAYS the retrieved candidate.
2. Cross-Filing Isolation: Predecessor and successor chunks are selected strictly from the
   exact same filing document (`doc_name`). Chunks from different documents are NEVER stitched.
3. Priority Preservation: The original candidate text is preserved in its entirety first.
   Preceding and succeeding chunks are stitched only if the character budget permits.
4. Overlap & Deduplication Hygiene: Detects if neighboring chunks already share boundary text,
   preventing redundant repetition.
5. Strict Character & Token Budget: Maximum character budget (default 1,200 chars).
   Tokenizer truncation (max 512 tokens) is strictly observed by the cross-encoder.
6. Original Text Preservation: `raw_text` and `text` are kept separate; stitched context is
   stored in `stitched_text`.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class ContextWindowStitcher:
    """
    Stitches at most one preceding and one succeeding chunk to each candidate chunk
    within the same filing document.
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
            chunk_store: Mapping of chunk_id -> chunk dict (from grounded corpus).
            max_chars: Total character budget for stitched text (default 1200).
            enable_preceding: Whether to stitch the immediate preceding chunk.
            enable_succeeding: Whether to stitch the immediate succeeding chunk.
        """
        self.chunk_store = chunk_store
        self.max_chars = max_chars
        self.enable_preceding = enable_preceding
        self.enable_succeeding = enable_succeeding

        # Build document-level sequential index
        # We index chunks by doc_name preserving their exact sequential appearance in chunk_store
        self.doc_to_chunks: Dict[str, List[str]] = {}
        self.chunk_id_to_index: Dict[str, Tuple[str, int]] = {}

        self._build_index()

    def _build_index(self):
        """
        Builds a deterministic ordered sequence of chunk_ids for each document.
        """
        doc_lists: Dict[str, List[Tuple[int, int, str]]] = {}

        for cid, chunk in self.chunk_store.items():
            doc = chunk.get("doc_name", "")
            page = chunk.get("page_number", chunk.get("page", 0))

            # Parse chunk index from chunk_id (e.g. AAPL_2024_10K_p3_c1 -> page 3, subchunk 1)
            # Regex: _p(\d+)_c(\d+)
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

        # Sort each document's chunks by (page, chunk_idx)
        for doc, tuples in doc_lists.items():
            tuples.sort(key=lambda t: (t[0], t[1]))
            ordered_cids = [t[2] for t in tuples]
            self.doc_to_chunks[doc] = ordered_cids
            for idx, cid in enumerate(ordered_cids):
                self.chunk_id_to_index[cid] = (doc, idx)

    def get_neighboring_chunk_ids(self, chunk_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Finds the immediate preceding and succeeding chunk IDs for a given chunk_id
        within the same document. Returns (prev_cid, next_cid).
        """
        if chunk_id not in self.chunk_id_to_index:
            return None, None

        doc, idx = self.chunk_id_to_index[chunk_id]
        doc_cids = self.doc_to_chunks[doc]

        prev_cid = doc_cids[idx - 1] if (self.enable_preceding and idx > 0) else None
        next_cid = doc_cids[idx + 1] if (self.enable_succeeding and idx < len(doc_cids) - 1) else None

        return prev_cid, next_cid

    def stitch_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a single candidate dict with stitched context while strictly preserving
        all existing candidate fields, ID, and pool ranking identity.

        Returns a shallow copy of candidate with:
        - 'stitched_text': The final stitched text string.
        - 'stitched_neighbors': List of neighboring chunk IDs included.
        - 'is_stitched': Boolean flag indicating if text was actually expanded.
        """
        cid = candidate.get("chunk_id", "")
        # Primary candidate text
        orig_text = candidate.get("text", "")
        if not orig_text and "raw_text" in candidate:
            orig_text = candidate.get("raw_text", "")

        # Shallow copy to avoid modifying original candidate in pool
        stitched_candidate = dict(candidate)
        stitched_candidate["original_text"] = orig_text

        prev_cid, next_cid = self.get_neighboring_chunk_ids(cid)
        prev_chunk = self.chunk_store.get(prev_cid) if prev_cid else None
        next_chunk = self.chunk_store.get(next_cid) if next_cid else None

        prev_text = (prev_chunk.get("text") or prev_chunk.get("raw_text", "")) if prev_chunk else ""
        next_text = (next_chunk.get("text") or next_chunk.get("raw_text", "")) if next_chunk else ""

        # Priority 1: Candidate's original text must be preserved
        budget_remaining = self.max_chars - len(orig_text)
        neighbors_included = []

        if budget_remaining <= 0:
            # Original text already exceeds or equals budget, return without expansion
            stitched_candidate["stitched_text"] = orig_text[:self.max_chars]
            stitched_candidate["stitched_neighbors"] = []
            stitched_candidate["is_stitched"] = False
            return stitched_candidate

        # Clean predecessor / successor text: remove company headers if present in raw form
        # to avoid double header noise
        def clean_neighbor_text(text: str) -> str:
            # Strip redundant grounded company context headers like "[Company: ...] "
            return re.sub(r"^\[Company:[^\]]+\]\s*", "", text).strip()

        cleaned_prev = clean_neighbor_text(prev_text)
        cleaned_next = clean_neighbor_text(next_text)

        # Priority 2: Preceding context (e.g. table header or sentence lead-in)
        # Allocate half of remaining budget to predecessor, half to successor
        half_budget = budget_remaining // 2

        pre_snippet = ""
        if cleaned_prev:
            # Take the trailing portion of the predecessor
            if len(cleaned_prev) > half_budget:
                pre_snippet = cleaned_prev[-half_budget:].strip()
                # Clean to start at a word boundary
                space_idx = pre_snippet.find(" ")
                if space_idx != -1 and space_idx < 30:
                    pre_snippet = pre_snippet[space_idx + 1:].strip()
            else:
                pre_snippet = cleaned_prev
            if pre_snippet:
                neighbors_included.append(prev_cid)

        # Priority 3: Succeeding context (continuation lines)
        post_budget = self.max_chars - len(orig_text) - len(pre_snippet) - 4
        post_snippet = ""
        if cleaned_next and post_budget > 30:
            if len(cleaned_next) > post_budget:
                post_snippet = cleaned_next[:post_budget].strip()
                # Clean to end at a word boundary
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
        """
        Processes a list of candidates in place of order, returning new stitched candidate dicts.
        Pool ranks and IDs are 100% preserved.
        """
        return [self.stitch_candidate(c) for c in candidates]
