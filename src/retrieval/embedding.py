"""
Embedding functions for converting text to vectors.
Uses Google Gemini Embedding API as primary, with fallback options.
"""

import os
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.logger import logger


class GeminiEmbedder:
    """Wrapper around Google Gemini embedding API."""

    def __init__(self, model: str = "models/gemini-embedding-2"):
        self.model_name = model
        self._embeddings = None

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(model=self.model_name)
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding query failed: {e}")
            return [0.0] * 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding documents failed: {e}")
            return [[0.0] * 768 for _ in texts]
