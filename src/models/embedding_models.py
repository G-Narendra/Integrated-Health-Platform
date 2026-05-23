"""
Embedding model manager for the healthcare platform.
Manages multiple embedding providers with fallback logic.
"""

from typing import List, Optional

from src.retrieval.embedding import GeminiEmbedder
from src.utils.logger import logger


class EmbeddingModelManager:
    """Manages embedding models with provider fallback."""

    def __init__(self, primary_provider: str = "gemini"):
        self.primary_provider = primary_provider
        self._gemini_embedder = None

    @property
    def gemini_embedder(self) -> GeminiEmbedder:
        if self._gemini_embedder is None:
            self._gemini_embedder = GeminiEmbedder()
        return self._gemini_embedder

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the primary provider."""
        try:
            return self.gemini_embedder.embed_query(text)
        except Exception as e:
            logger.error(f"Primary embedding failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the primary provider."""
        try:
            return self.gemini_embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"Primary document embedding failed: {e}")
            raise
