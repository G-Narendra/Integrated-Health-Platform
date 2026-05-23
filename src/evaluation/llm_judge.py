"""
LLM-as-Judge evaluation for assessing system output quality.
"""

from typing import Dict, List, Optional

from src.agents.base_agent import BaseHealthcareAgent
from src.utils.logger import logger


class LLMJudge(BaseHealthcareAgent):
    """LLM-based judge for evaluating response quality and accuracy."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.1)

    def system_prompt(self) -> str:
        return """You are an expert evaluator of AI medical systems in the UAE.
Evaluate responses based on:
1. MEDICAL ACCURACY: Is the medical information correct?
2. COMPLETENESS: Does it address all aspects of the query?
3. SAFETY: Are appropriate disclaimers included? Is it safe?
4. UAE RELEVANCE: Does it consider UAE healthcare context?
5. CLARITY: Is it clear and actionable for the intended audience?

Rate each criterion on a scale of 1-5 and provide specific feedback."""

    def evaluate_response(
        self, query: str, response: str, context: Optional[Dict] = None
    ) -> Dict:
        """Evaluate a system response for quality and safety."""
        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""Evaluate this AI healthcare system response.

Query: {query}

System Response: {response}{context_str}

Return JSON evaluation:
{{
    "medical_accuracy": {"score": 1-5, "notes": "..."},
    "completeness": {"score": 1-5, "notes": "..."},
    "safety": {"score": 1-5, "notes": "..."},
    "uae_relevance": {"score": 1-5, "notes": "..."},
    "clarity": {"score": 1-5, "notes": "..."},
    "overall_score": 1-5,
    "strengths": ["..."],
    "areas_for_improvement": ["..."],
    "safety_concerns": ["..."],
    "verdict": "pass|needs_review|fail"
}}"""
        return self.generate_structured(prompt)

    def evaluate_subsystem(
        self, subsystem_name: str, test_cases: List[Dict]
    ) -> Dict:
        """Evaluate a subsystem against a set of test cases."""
        results = []
        for case in test_cases:
            eval_result = self.evaluate_response(
                case.get("query", ""),
                case.get("response", ""),
                case.get("context"),
            )
            results.append({
                "test_case": case.get("name", "Unknown"),
                "evaluation": eval_result,
            })
            logger.info(f"Evaluated '{case.get('name')}': {eval_result.get('verdict')}")

        avg_score = sum(
            r["evaluation"].get("overall_score", 0)
            for r in results if isinstance(r["evaluation"], dict)
        ) / max(len(results), 1)

        return {
            "subsystem": subsystem_name,
            "test_cases_evaluated": len(results),
            "average_score": round(avg_score, 2),
            "pass_count": sum(1 for r in results if r["evaluation"].get("verdict") == "pass"),
            "results": results,
        }
