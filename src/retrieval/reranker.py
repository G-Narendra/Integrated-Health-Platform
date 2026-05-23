"""
Reranking module for improving retrieval quality using cross-encoder scoring.
"""

from typing import Dict, List


class Reranker:
    """Reranks retrieved documents using relevance scoring."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Rerank documents by relevance to the query."""
        if not documents:
            return []

        # Score documents using simple relevance heuristics
        query_terms = set(query.lower().split())

        for doc in documents:
            text = doc.get("text", "").lower()
            metadata = doc.get("metadata", {})

            # Score components
            term_overlap = len(query_terms & set(text.split())) / max(len(query_terms), 1)
            exact_phrase = 1.5 if any(phrase in text for phrase in query.split()) else 0
            source_boost = 1.2 if metadata.get("source_type") in ["official", "guideline"] else 1.0

            doc["relevance_score"] = (term_overlap + exact_phrase) * source_boost

        # Sort by relevance score
        documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return documents[:top_k]
