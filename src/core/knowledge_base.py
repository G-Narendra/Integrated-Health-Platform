"""
Shared knowledge base accessible by all healthcare subsystems.
Manages connections to ChromaDB collections for different medical knowledge types.
"""

from typing import Dict, List, Optional

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.utils.logger import logger


class SharedKnowledgeBase:
    """Centralized medical knowledge accessible by all subsystems."""

    def __init__(self, chroma_path: str = "./data/chroma_db"):
        self.vector_store = VectorStore(persist_directory=chroma_path)
        self.hybrid_search = HybridSearch(self.vector_store)
        self.reranker = Reranker()

    def query_guidelines(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query clinical guidelines."""
        results = self.vector_store.query(
            collection_name="clinical_guidelines",
            query_text=query_text,
            top_k=top_k,
        )
        return self.reranker.rerank(query_text, results, top_k)

    def query_drugs(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query drug database."""
        results = self.vector_store.query(
            collection_name="drug_database",
            query_text=query_text,
            top_k=top_k,
        )
        return self.reranker.rerank(query_text, results, top_k)

    def query_regulations(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query UAE MOH regulations."""
        results = self.vector_store.query(
            collection_name="moh_regulations",
            query_text=query_text,
            top_k=top_k,
        )
        return self.reranker.rerank(query_text, results, top_k)

    def query_literature(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query medical literature."""
        results = self.vector_store.query(
            collection_name="medical_literature",
            query_text=query_text,
            top_k=top_k,
        )
        return self.reranker.rerank(query_text, results, top_k)

    def get_drug_info(self, drug_name: str) -> Dict:
        """Quick lookup for drug information."""
        results = self.query_drugs(drug_name, top_k=1)
        if results:
            return results[0]
        return {"text": "", "metadata": {}, "score": 0}

    def check_moh_compliance(self, procedure: str) -> Dict:
        """Check UAE MOH compliance for a procedure."""
        results = self.query_regulations(procedure, top_k=3)
        return {
            "compliant": len(results) > 0,
            "relevant_regulations": results,
        }

    def search_all(self, query_text: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """Search across all knowledge bases simultaneously."""
        return {
            "clinical_guidelines": self.query_guidelines(query_text, top_k),
            "drug_database": self.query_drugs(query_text, top_k),
            "moh_regulations": self.query_regulations(query_text, top_k),
            "medical_literature": self.query_literature(query_text, top_k),
        }

    def status(self) -> Dict[str, int]:
        """Get status of all knowledge base collections."""
        return {
            "clinical_guidelines": self.vector_store.count("clinical_guidelines"),
            "drug_database": self.vector_store.count("drug_database"),
            "moh_regulations": self.vector_store.count("moh_regulations"),
            "medical_literature": self.vector_store.count("medical_literature"),
        }
