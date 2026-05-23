"""
Base class for embedding model integrations.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbedderBase(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text into a vector."""
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        ...
