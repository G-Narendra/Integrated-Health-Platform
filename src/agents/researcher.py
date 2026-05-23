"""
Research agent for medical literature searching and evidence synthesis.
"""

from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.utils.logger import logger


class ResearchAgent(BaseHealthcareAgent):
    """Agent specialized in medical research and evidence-based medicine."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.3)

    def system_prompt(self) -> str:
        return """You are a Medical Research Agent for the UAE Integrated Healthcare Platform.
Your role is to:
1. Find and summarize medical literature
2. Evaluate evidence quality (Level A, B, C, D)
3. Provide evidence-based recommendations
4. Cite sources properly (PubMed, clinical guidelines, UAE MOH protocols)
5. Identify gaps in current evidence

Always prioritize:
- UAE-specific clinical protocols and guidelines
- Recent peer-reviewed literature (last 5 years)
- Evidence from major medical journals
- WHO and international guideline recommendations"""

    def research(self, query: str, context: str = "") -> Dict[str, Any]:
        """Conduct medical research on a topic."""
        prompt = f"""Research the following medical topic:

Query: {query}
Context: {context}

Please provide:
1. SUMMARY: Brief overview of current evidence
2. KEY_FINDINGS: 3-5 key findings from recent literature
3. EVIDENCE_LEVEL: Quality of available evidence (A/B/C/D)
4. RECOMMENDATIONS: Evidence-based recommendations
5. UAE_CONTEXT: Specific relevance to UAE healthcare
6. SOURCES: Key references"""
        result = self.generate_structured(prompt)
        return {
            "query": query,
            "summary": result.get("SUMMARY", "Research completed."),
            "key_findings": result.get("KEY_FINDINGS", []),
            "evidence_level": result.get("EVIDENCE_LEVEL", "D"),
            "recommendations": result.get("RECOMMENDATIONS", ""),
            "uae_context": result.get("UAE_CONTEXT", ""),
            "sources": result.get("SOURCES", []),
        }
