"""
FinSage Metadata Retriever (Path 3 of Multi-Path Retrieval)
===========================================================
Paper Specifications: FinSage (arXiv:2504.14493, CIKM 2025) Section 3.1 (Appendix A.7) & Section 3.2.

Queries the Heading-Summary FAISS Index:
- Computes cosine similarity between query embedding and section heading-summary embeddings
- Retrieves all constituent chunks associated with matching section/statement blocks
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class MetadataRetriever:
    """
    Retrieves candidate chunks by matching query against section heading-summaries (Path 3).
    """

    def __init__(
        self,
        metadata_index_dir: str,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ):
        self.metadata_index_dir = metadata_index_dir
        self.embeddings = embeddings
        self.index = None
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.metadata_index_dir):
            raise FileNotFoundError(f"Metadata index not found at {self.metadata_index_dir}")

        if self.embeddings is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )

        self.index = FAISS.load_local(
            self.metadata_index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve_metadata_candidates(
        self,
        query: str,
        k_sections: int = 5,
    ) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
        """
        Queries heading summaries and returns all chunks within the top matching sections.
        
        Returns:
            Tuple of:
            - list of (chunk_id, score)
            - list of matched section metadata dicts
        """
        results = self.index.similarity_search_with_score(query, k=k_sections)
        
        matched_sections = []
        chunk_scores: Dict[str, float] = {}

        for rank, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata
            section_title = meta.get("section_title", "Unknown Section")
            cids = meta.get("chunk_ids", [])
            fscore = float(score)

            matched_sections.append({
                "rank": rank,
                "section_id": meta.get("section_id"),
                "section_title": section_title,
                "doc_name": meta.get("doc_name"),
                "score": fscore,
                "chunk_count": len(cids),
                "chunk_ids": cids
            })

            # Distribute section relevance score to all associated chunks
            # In FAISS L2/normalized space, lower distance = better
            # Adjust score slightly by chunk position so first chunk has a small primacy
            for idx, cid in enumerate(cids):
                adjusted_score = fscore + (idx * 0.001)
                if cid not in chunk_scores or adjusted_score < chunk_scores[cid]:
                    chunk_scores[cid] = adjusted_score

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1])
        return sorted_chunks, matched_sections
