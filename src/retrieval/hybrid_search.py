"""
Hybrid search combining vector similarity with keyword/BM25 scoring.
"""

from typing import Dict, List, Optional

from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger


class HybridSearch:
    """Performs hybrid search combining vector and keyword matching."""

    def __init__(self, vector_store: VectorStore, keyword_weight: float = 0.3):
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Perform hybrid search with reranking."""
        # Get vector search results
        vector_results = self.vector_store.query(
            collection_name=collection_name,
            query_text=query_text,
            top_k=top_k * 2,
            where=where,
        )

        if not vector_results:
            return []

        # Get keyword scores
        keyword_scores = self._keyword_score(query_text, vector_results)

        # Combine scores
        for i, result in enumerate(vector_results):
            vector_score = 1.0 - result.get("score", 0) if result.get("score") else 0.5
            kw_score = keyword_scores.get(i, 0)
            result["hybrid_score"] = (
                (1 - self.keyword_weight) * vector_score
                + self.keyword_weight * kw_score
            )

        # Sort by hybrid score
        vector_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        return vector_results[:top_k]

    def _keyword_score(self, query: str, documents: List[Dict]) -> Dict[int, float]:
        """Calculate keyword match scores for each document."""
        query_terms = set(query.lower().split())
        scores = {}

        for i, doc in enumerate(documents):
            text = doc.get("text", "").lower()
            text_terms = set(text.split())

            if not query_terms or not text_terms:
                scores[i] = 0
                continue

            intersection = query_terms & text_terms
            scores[i] = len(intersection) / max(len(query_terms), 1)

        return scores
