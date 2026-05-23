"""
Monitoring and metrics tracking for system health and performance.
"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class MetricsCollector:
    """Simple metrics collector for tracking subsystem performance."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def record_latency(self, subsystem: str, duration_ms: float) -> None:
        self._metrics[f"{subsystem}_latency"].append(duration_ms)
        self._request_counts[subsystem] += 1

    def record_error(self, subsystem: str) -> None:
        self._error_counts[subsystem] += 1

    def get_subsystem_stats(self, subsystem: str) -> Dict:
        latencies = self._metrics.get(f"{subsystem}_latency", [])
        if not latencies:
            return {"avg_latency_ms": 0, "requests": 0, "errors": 0, "error_rate": 0}

        avg_latency = sum(latencies) / len(latencies)
        requests = self._request_counts.get(subsystem, 0)
        errors = self._error_counts.get(subsystem, 0)

        return {
            "avg_latency_ms": round(avg_latency, 2),
            "requests": requests,
            "errors": errors,
            "error_rate": round(errors / requests * 100, 2) if requests > 0 else 0,
        }

    def get_all_stats(self) -> Dict:
        subsystems = set(
            [k.replace("_latency", "") for k in self._metrics.keys() if k.endswith("_latency")]
        )
        return {s: self.get_subsystem_stats(s) for s in sorted(subsystems)}

    def get_uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def reset(self) -> None:
        self._metrics.clear()
        self._error_counts.clear()
        self._request_counts.clear()
        self._start_time = time.time()


# Global metrics collector
metrics = MetricsCollector()
