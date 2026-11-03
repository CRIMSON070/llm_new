"""
FinSage HyDE Retriever (Path 4 of Multi-Path Retrieval)
======================================================
Paper Specifications: FinSage (arXiv:2504.14493, CIKM 2025) Section 3.2 & 4.2.1.

Coordinates hypothetical passage generation and dense vector retrieval:
- Prompts Qwen2.5-7B-Instruct to generate 3 hypothetical document passages
- Embeds each passage using BGE-M3 dense embeddings
- Searches FAISS dense index to return candidate chunks for Path 4
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.hyde_generator import GenerativeHyDE


class HyDERetriever:
    """
    Retrieves candidate chunks by embedding LLM-generated hypothetical documents (Path 4).
    """

    def __init__(
        self,
        faiss_index,
        generator: Optional[GenerativeHyDE] = None,
        num_hypo: int = 3,
    ):
        self.faiss_index = faiss_index
        self.generator = generator
        self.num_hypo = num_hypo

    def retrieve_hyde_candidates(
        self,
        query: str,
        k_per_hypo: int = 20,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        Generates 3 hypothetical document passages, queries FAISS for each,
        and aggregates candidate chunk IDs with cosine similarity scores.
        
        Returns:
            Tuple of:
            - list of (chunk_id, raw_score)
            - list of generated hypothetical passages (for auditing/telemetry)
        """
        if self.generator is None:
            self.generator = GenerativeHyDE()

        passages = self.generator.generate_hypothetical_passages(query, num_hypo=self.num_hypo)
        
        chunk_best_scores: Dict[str, float] = {}
        
        for p in passages:
            try:
                results = self.faiss_index.similarity_search_with_score(p, k=k_per_hypo)
                for doc, score in results:
                    cid = doc.metadata.get("chunk_id")
                    if cid:
                        # FAISS returns L2 distance or normalized similarity depending on index
                        # Keep the best (lowest distance or highest similarity)
                        fscore = float(score)
                        if cid not in chunk_best_scores or fscore < chunk_best_scores[cid]:
                            chunk_best_scores[cid] = fscore
            except Exception as e:
                print(f"[HyDERetriever] Error querying FAISS with passage: {e}")

        # Sort candidates
        sorted_candidates = sorted(chunk_best_scores.items(), key=lambda x: x[1])
        return sorted_candidates, passages
