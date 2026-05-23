"""
Benchmark the system's performance under load.
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.healthcare_orchestrator import HealthcareOrchestrator
from src.utils.logger import logger


# Test queries for benchmarking
BENCHMARK_QUERIES = [
    {"query": "Patient with chest pain and elevated troponin", "type": "clinical_diagnosis"},
    {"query": "Check if Amlodipine 10mg and Metformin 500mg are safe together", "type": "prescription_verify"},
    {"query": "Summarize the patient's medical history for the last year", "type": "medical_record"},
    {"query": "Book an appointment with a cardiologist next Monday", "type": "appointment"},
    {"query": "Latest treatments for diabetic nephropathy in UAE", "type": "research"},
]


def run_single_test(orchestrator, query_data, request_id: int) -> dict:
    """Run a single benchmark test."""
    start = time.time()
    try:
        result = orchestrator.handle_request(query_data)
        duration_ms = (time.time() - start) * 1000
        return {
            "id": request_id,
            "query": query_data["query"],
            "type": query_data.get("type", "unknown"),
            "success": result.get("success", False),
            "duration_ms": round(duration_ms, 2),
            "subsystem": result.get("subsystem_used", "unknown"),
        }
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        return {
            "id": request_id,
            "query": query_data["query"],
            "type": query_data.get("type", "unknown"),
            "success": False,
            "duration_ms": round(duration_ms, 2),
            "error": str(e),
        }


def run_benchmark(concurrent_users: int = 5, iterations: int = 2):
    """Run benchmark with specified concurrency."""
    print("=" * 60)
    print(f"BENCHMARK: {concurrent_users} concurrent users, {iterations} iterations")
    print("=" * 60)

    orchestrator = HealthcareOrchestrator()
    results = []

    total_requests = len(BENCHMARK_QUERIES) * iterations
    print(f"Total requests: {total_requests}")

    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = {}
            for i, query_data in enumerate(BENCHMARK_QUERIES):
                req_id = iteration * len(BENCHMARK_QUERIES) + i
                future = executor.submit(run_single_test, orchestrator, query_data, req_id)
                futures[future] = req_id

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "✅" if result["success"] else "❌"
                print(f"  {status} Req {result['id']:03d} | {result['type']:25s} | {result['duration_ms']:8.1f}ms")

    # Compute statistics
    durations = [r["duration_ms"] for r in results]
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    avg_duration = sum(durations) / max(len(durations), 1)
    sorted_durations = sorted(durations)
    p95 = sorted_durations[int(len(sorted_durations) * 0.95)]
    p99 = sorted_durations[int(len(sorted_durations) * 0.99)]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total requests:     {len(results)}")
    print(f"  Successful:         {len(successful)}")
    print(f"  Failed:             {len(failed)}")
    print(f"  Avg response time:  {avg_duration:.1f}ms")
    print(f"  P95 response time:  {p95:.1f}ms")
    print(f"  P99 response time:  {p99:.1f}ms")

    # Per-subsystem breakdown
    subsystem_stats = {}
    for r in results:
        sub = r.get("subsystem", "unknown")
        if sub not in subsystem_stats:
            subsystem_stats[sub] = {"count": 0, "total_duration": 0, "successful": 0}
        subsystem_stats[sub]["count"] += 1
        subsystem_stats[sub]["total_duration"] += r["duration_ms"]
        if r["success"]:
            subsystem_stats[sub]["successful"] += 1

    print("\n  Per-Subsystem:")
    for sub, stats in sorted(subsystem_stats.items()):
        avg = stats["total_duration"] / stats["count"]
        print(f"    {sub:25s} | Avg: {avg:7.1f}ms | Success: {stats['successful']}/{stats['count']}")

    # Save results
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": {
            "concurrent_users": concurrent_users,
            "iterations": iterations,
        },
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "avg_response_ms": round(avg_duration, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
        },
        "subsystem_stats": subsystem_stats,
        "results": results,
    }

    output_path = Path("data/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to data/benchmark_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the healthcare platform")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent users")
    parser.add_argument("--iterations", type=int, default=2, help="Number of test iterations")

    args = parser.parse_args()
    run_benchmark(concurrent_users=args.concurrent, iterations=args.iterations)
