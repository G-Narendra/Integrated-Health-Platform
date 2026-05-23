"""
RAGAS (RAG Assessment) evaluation integration.
Provides automated evaluation of RAG pipeline quality.
"""

from typing import Dict, List, Optional

from src.utils.logger import logger


class RagasEvaluator:
    """Wrapper for RAGAS evaluation metrics.
    Requires ragas package for full functionality."""

    def __init__(self):
        self._available = False
        try:
            import ragas
            self._available = True
        except ImportError:
            logger.warning("ragas package not installed. Some evaluation features disabled.")

    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
    ) -> Dict:
        """Evaluate retrieval quality."""
        if not self._available:
            return {"error": "ragas not installed", "fallback": True}

        try:
            from ragas.metrics import context_precision, context_recall

            scores = {
                "context_precision": context_precision.score(
                    queries, retrieved_docs, relevant_docs
                ),
                "context_recall": context_recall.score(
                    queries, retrieved_docs, relevant_docs
                ),
            }
            return scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_response(
        self,
        queries: List[str],
        responses: List[str],
        retrieved_docs: List[List[str]],
        ground_truth: List[str],
    ) -> Dict:
        """Evaluate end-to-end RAG response quality."""
        if not self._available:
            return {"error": "ragas not installed", "fallback": True}

        try:
            from ragas.metrics import faithfulness, answer_relevancy

            scores = {
                "faithfulness": faithfulness.score(queries, responses, retrieved_docs),
                "answer_relevancy": answer_relevancy.score(queries, responses, retrieved_docs),
            }
            return scores
        except Exception as e:
            logger.error(f"RAGAS response evaluation failed: {e}")
            return {"error": str(e)}
