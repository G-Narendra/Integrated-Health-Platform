"""
System metrics for evaluating retrieval and response quality.
"""

from typing import Dict, List, Optional


class SystemMetrics:
    """Compute standard metrics for evaluating system performance."""

    @staticmethod
    def precision_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
        """Compute Precision@K."""
        if not retrieved[:k]:
            return 0.0
        relevant_set = set(relevant)
        retrieved_set = set(retrieved[:k])
        if not retrieved_set:
            return 0.0
        return len(relevant_set & retrieved_set) / len(retrieved_set)

    @staticmethod
    def recall_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
        """Compute Recall@K."""
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        retrieved_set = set(retrieved[:k])
        return len(relevant_set & retrieved_set) / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
        """Compute Mean Reciprocal Rank."""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def response_latency(timings_ms: List[float]) -> Dict:
        """Compute latency statistics."""
        if not timings_ms:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0}

        sorted_timings = sorted(timings_ms)
        n = len(sorted_timings)

        return {
            "avg": round(sum(sorted_timings) / n, 2),
            "min": round(sorted_timings[0], 2),
            "max": round(sorted_timings[-1], 2),
            "p95": round(sorted_timings[int(n * 0.95)], 2) if n >= 20 else sorted_timings[-1],
        }

    @staticmethod
    def compute_subsystem_score(metrics_dict: Dict) -> float:
        """Compute overall subsystem score from individual metrics."""
        weights = {
            "precision": 0.3,
            "recall": 0.3,
            "latency_score": 0.2,
            "accuracy": 0.2,
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics_dict:
                score += metrics_dict[metric] * weight

        return round(score, 2)
