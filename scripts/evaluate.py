"""
Evaluate the Integrated Healthcare Platform's performance against the golden dataset.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.healthcare_orchestrator import HealthcareOrchestrator
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import SystemMetrics


def load_golden_dataset() -> list:
    """Load the golden test dataset."""
    dataset_path = Path("data/golden_dataset.json")
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def evaluate_system():
    """Run evaluation against golden dataset."""
    print("=" * 60)
    print("EVALUATING INTEGRATED HEALTHCARE PLATFORM")
    print("=" * 60)

    test_cases = load_golden_dataset()
    if not test_cases:
        print("\n⚠️  No golden dataset found. Running build_vector_db.py first...")
        os.system(f"{sys.executable} scripts/build_vector_db.py")
        test_cases = load_golden_dataset()

    if not test_cases:
        print("❌ Could not load test cases.")
        return

    orchestrator = HealthcareOrchestrator()
    judge = LLMJudge()
    metrics = SystemMetrics()

    user_context = {
        "user_id": "EVAL-001",
        "role": "evaluator",
        "facility": "Test Environment",
    }

    results = []
    latencies = []

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing: {case['query'][:60]}...")

        request = {
            "query": case["query"],
            "request_type": case.get("subsystem", "clinical_diagnosis"),
        }

        start = time.time()
        response = orchestrator.handle_request(request, user_context)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        result_text = (
            str(response.get("result", {}).get("final_report", ""))
            or str(response.get("result", {}).get("research_report", ""))
            or str(response.get("result", {}).get("summary", ""))
            or str(response.get("result", {}))
        )

        eval_result = judge.evaluate_response(case["query"], result_text[:1000], {
            "expected_response": case.get("expected_response", ""),
        })

        results.append({
            "test_case": case["query"],
            "latency_ms": round(latency_ms, 2),
            "success": response.get("success", False),
            "subsystem": response.get("subsystem_used", "unknown"),
            "evaluation": eval_result,
        })

        print(f"  Latency: {latency_ms:.0f}ms | Success: {response.get('success')} | Score: {eval_result.get('overall_score', 'N/A')}")

    # Summary
    avg_latency = sum(latencies) / max(len(latencies), 1)
    success_rate = sum(1 for r in results if r["success"]) / max(len(results), 1) * 100
    avg_score = sum(
        r["evaluation"].get("overall_score", 0) for r in results
    ) / max(len(results), 1)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test Cases:     {len(results)}")
    print(f"  Success Rate:   {success_rate:.1f}%")
    print(f"  Avg Latency:    {avg_latency:.0f}ms")
    print(f"  Avg Quality Score: {avg_score:.1f}/5")
    print(f"  Subsystems Used: {set(r['subsystem'] for r in results)}")

    # Save results
    output_path = Path("data/evaluation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "summary": {
                "total_cases": len(results),
                "success_rate": round(success_rate, 1),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_quality_score": round(avg_score, 2),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to data/evaluation_results.json")


if __name__ == "__main__":
    evaluate_system()
