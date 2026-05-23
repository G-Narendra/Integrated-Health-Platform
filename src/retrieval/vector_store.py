"""
ChromDB-based vector store for medical knowledge management.
"""

import os
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.retrieval.embedding import GeminiEmbedder
from src.utils.logger import logger


class VectorStore:
    """Manages ChromaDB collections for different medical knowledge types."""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedder = GeminiEmbedder()

        # Collection names
        self.COLLECTIONS = {
            "clinical_guidelines": "clinical_guidelines",
            "drug_database": "drug_database",
            "moh_regulations": "moh_regulations",
            "medical_literature": "medical_literature",
        }

    def get_or_create_collection(self, name: str):
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name=name)
        except ValueError:
            try:
                return self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error(f"Failed to create collection '{name}': {e}")
                return self.client.create_collection(name=name)

    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, str]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """Add documents to a collection."""
        collection = self.get_or_create_collection(collection_name)
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        try:
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Added {len(documents)} documents to '{collection_name}'")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add documents to '{collection_name}': {e}")
            return 0

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Query a collection for similar documents."""
        try:
            collection = self.get_or_create_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}': {e}")
            return []

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, 20),
                where=where,
            )

            docs = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    docs.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": results["distances"][0][i] if results["distances"] else 0,
                    })
            return docs
        except Exception as e:
            logger.error(f"Query failed on '{collection_name}': {e}")
            return []

    def count(self, collection_name: str) -> int:
        """Get document count in a collection."""
        try:
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def list_collections(self) -> List[str]:
        """List all collections."""
        return [c.name for c in self.client.list_collections()]
