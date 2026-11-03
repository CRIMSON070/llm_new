"""
Phase 1: Chunk Bundling (Bundle Expansion) Module.
Faithful reproduction of FinSage (arXiv:2504.14493) Section 3.2 and Section 4.2.1.

Iteratively expands candidate chunks with adjacent chunks (s_{i-1}, s_{i+1})
from the same document if the cosine similarity of their dense embeddings
meets or exceeds tau_exp (paper specified: 0.85), up to bundle limit K = 5.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn.functional as F
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class ChunkBundler:
    """
    Manages document chunk adjacency and semantic bundle expansion according to
    FinSage Section 3.2 and Section 4.2.1.
    """

    def __init__(
        self,
        chunks_file: str,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        tau_exp: float = 0.85,
        max_bundle_size: int = 5,
        device: Optional[str] = None
    ):
        self.chunks_file = chunks_file
        self.tau_exp = tau_exp
        self.max_bundle_size = max_bundle_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embeddings = embeddings
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )

        self.chunk_store: Dict[str, Dict[str, Any]] = {}
        self.doc_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.chunk_pos_in_doc: Dict[str, int] = {}
        self.embedding_cache: Dict[str, torch.Tensor] = {}

        self._load_and_index_chunks()

    def _load_and_index_chunks(self):
        """Load and order chunks by document to establish sequential adjacency."""
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                cid = c["chunk_id"]
                doc = c["doc_name"]
                self.chunk_store[cid] = c
                self.chunk_pos_in_doc[cid] = len(self.doc_chunks[doc])
                self.doc_chunks[doc].append(c)

        logger.info(
            f"[ChunkBundler] Loaded {len(self.chunk_store)} chunks across "
            f"{len(self.doc_chunks)} documents."
        )

    def get_chunk_embedding(self, chunk_id: str) -> torch.Tensor:
        """Retrieve or compute normalized dense embedding for a chunk."""
        if chunk_id in self.embedding_cache:
            return self.embedding_cache[chunk_id]

        chunk_data = self.chunk_store.get(chunk_id)
        if not chunk_data:
            raise KeyError(f"Chunk ID {chunk_id} not found in store")

        emb = torch.tensor(
            self.embeddings.embed_query(chunk_data["text"]),
            device=self.device
        )
        self.embedding_cache[chunk_id] = emb
        return emb

    def compute_similarity(self, cid_a: str, cid_b: str) -> float:
        """Compute cosine similarity between two chunks."""
        emb_a = self.get_chunk_embedding(cid_a)
        emb_b = self.get_chunk_embedding(cid_b)
        return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    def is_table_boundary_compatible(self, chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> bool:
        """
        Ensure adjacent chunks do not cross unrelated tables or jump across distant pages.
        Maintains table and financial statement evidence integrity.
        """
        # 1. Reject page jumps greater than 1 page
        page_a = chunk_a.get("page_number", 0)
        page_b = chunk_b.get("page_number", 0)
        if abs(page_a - page_b) > 1:
            return False

        # 2. Reject combination across conflicting explicit section headings
        sec_a = chunk_a.get("section_heading")
        sec_b = chunk_b.get("section_heading")
        if sec_a and sec_b and sec_a.strip().lower() != sec_b.strip().lower():
            return False

        # 3. Reject combination across conflicting financial statement types
        stmt_a = chunk_a.get("propagated_statement")
        stmt_b = chunk_b.get("propagated_statement")
        if stmt_a and stmt_b and stmt_a.strip().lower() != stmt_b.strip().lower():
            return False

        return True

    def bundle_single_chunk(
        self,
        lead_cid: str,
        tau_exp: Optional[float] = None,
        max_bundle_size: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Expand a single lead chunk with adjacent chunks within the same document.
        Returns:
            (bundle_chunk_ids, trace_log)
        """
        threshold = tau_exp if tau_exp is not None else self.tau_exp
        max_size = max_bundle_size if max_bundle_size is not None else self.max_bundle_size

        if lead_cid not in self.chunk_store:
            return [lead_cid], {"error": f"Unknown chunk ID {lead_cid}"}

        lead_c = self.chunk_store[lead_cid]
        doc = lead_c["doc_name"]
        pos = self.chunk_pos_in_doc[lead_cid]
        doc_seq = self.doc_chunks[doc]

        bundle_cids = [lead_cid]
        expansion_trace = []

        # 1. Forward Expansion (s_{i+1}, s_{i+2}, ...)
        next_pos = pos + 1
        curr_cid = lead_cid
        curr_c = lead_c
        while len(bundle_cids) < max_size and next_pos < len(doc_seq):
            next_cand = doc_seq[next_pos]
            next_cid = next_cand["chunk_id"]
            
            # Check table and section boundary compatibility
            if not self.is_table_boundary_compatible(curr_c, next_cand):
                expansion_trace.append({
                    "direction": "forward",
                    "from": curr_cid,
                    "to": next_cid,
                    "similarity": 0.0,
                    "threshold": threshold,
                    "accepted": False,
                    "reason": "boundary_incompatible"
                })
                break

            sim = self.compute_similarity(curr_cid, next_cid)
            expansion_trace.append({
                "direction": "forward",
                "from": curr_cid,
                "to": next_cid,
                "similarity": round(sim, 4),
                "threshold": threshold,
                "accepted": sim >= threshold
            })
            if sim >= threshold:
                bundle_cids.append(next_cid)
                curr_cid = next_cid
                curr_c = next_cand
                next_pos += 1
            else:
                break

        # 2. Backward Expansion (s_{i-1}, s_{i-2}, ...)
        prev_pos = pos - 1
        curr_cid = lead_cid
        curr_c = lead_c
        while len(bundle_cids) < max_size and prev_pos >= 0:
            prev_cand = doc_seq[prev_pos]
            prev_cid = prev_cand["chunk_id"]

            # Check table and section boundary compatibility
            if not self.is_table_boundary_compatible(curr_c, prev_cand):
                expansion_trace.append({
                    "direction": "backward",
                    "from": curr_cid,
                    "to": prev_cid,
                    "similarity": 0.0,
                    "threshold": threshold,
                    "accepted": False,
                    "reason": "boundary_incompatible"
                })
                break

            sim = self.compute_similarity(curr_cid, prev_cid)
            expansion_trace.append({
                "direction": "backward",
                "from": curr_cid,
                "to": prev_cid,
                "similarity": round(sim, 4),
                "threshold": threshold,
                "accepted": sim >= threshold
            })
            if sim >= threshold:
                bundle_cids.insert(0, prev_cid)
                curr_cid = prev_cid
                curr_c = prev_cand
                prev_pos -= 1
            else:
                break

        return bundle_cids, {
            "lead_chunk_id": lead_cid,
            "bundle_size": len(bundle_cids),
            "bundle_chunk_ids": bundle_cids,
            "threshold_used": threshold,
            "trace": expansion_trace
        }

    def bundle_candidate_pool(
        self,
        candidate_cids: List[str],
        tau_exp: Optional[float] = None,
        max_bundle_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply chunk bundling to a full list of candidate chunk IDs.
        Returns bundle mappings and expanded metadata.
        """
        threshold = tau_exp if tau_exp is not None else self.tau_exp
        max_size = max_bundle_size if max_bundle_size is not None else self.max_bundle_size

        lead_to_bundle: Dict[str, List[str]] = {}
        bundle_traces: Dict[str, Any] = {}
        all_bundled_cids: Set[str] = set()

        for cid in candidate_cids:
            bundle, trace = self.bundle_single_chunk(
                cid,
                tau_exp=threshold,
                max_bundle_size=max_size
            )
            lead_to_bundle[cid] = bundle
            bundle_traces[cid] = trace
            all_bundled_cids.update(bundle)

        return {
            "candidate_cids": candidate_cids,
            "lead_to_bundle": lead_to_bundle,
            "bundle_traces": bundle_traces,
            "total_unique_chunks": len(all_bundled_cids),
            "tau_exp": threshold,
            "max_bundle_size": max_size
        }

    def assemble_bundled_context(
        self,
        top_lead_cids: List[str],
        lead_to_bundle: Dict[str, List[str]],
        max_total_chunks: int = 5,
        preserve_table_headers: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Assemble final context documents from top ranked lead chunks,
        incorporating their bundled adjacent chunks in proper document order
        while respecting context limits and avoiding duplicate chunks.

        In Phase 3:
        Preserves table integrity by ensuring continuation chunks retain their
        propagated column headers, units, and reporting periods so that table values
        remain interpretable by the downstream generator and reranker.
        """
        selected_cids: List[str] = []
        seen: Set[str] = set()

        for lead_cid in top_lead_cids:
            bundle = lead_to_bundle.get(lead_cid, [lead_cid])
            for cid in bundle:
                if cid not in seen and len(selected_cids) < max_total_chunks:
                    seen.add(cid)
                    selected_cids.append(cid)
            if len(selected_cids) >= max_total_chunks:
                break

        # Return full chunk dicts in order with table header integrity
        context_chunks = []
        for cid in selected_cids:
            if cid in self.chunk_store:
                raw_chunk = self.chunk_store[cid]
                if preserve_table_headers and raw_chunk.get("is_table_continuation"):
                    # Check if propagated header needs to be made explicit
                    prop_header = raw_chunk.get("propagated_header", "")
                    text = raw_chunk.get("text", "")
                    if prop_header and prop_header not in text:
                        c_dict = dict(raw_chunk)
                        c_dict["text"] = f"{prop_header}\n{text}"
                        c_dict["table_header_preserved"] = True
                        context_chunks.append(c_dict)
                        continue
                context_chunks.append(dict(raw_chunk))

        return context_chunks
