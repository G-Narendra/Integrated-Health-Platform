"""
Subsystem 5: Medical Research (Agentic RAG)
Agentic RAG for medical research with iterative refinement.
"""

from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.core.knowledge_base import SharedKnowledgeBase
from src.utils.logger import logger


class MedicalResearchSystem(BaseHealthcareAgent):
    """Agentic RAG system for medical research with iterative retrieval and synthesis."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.3)
        self.max_iterations = 2

    def system_prompt(self) -> str:
        return """You are a Medical Research Agent in the UAE Integrated Healthcare Platform.
Your role is to:
1. Conduct thorough medical research on clinical questions
2. Evaluate and synthesize evidence from multiple sources
3. Provide evidence-based answers with citations
4. Identify knowledge gaps and uncertainties
5. Adapt findings to UAE healthcare context

Always include:
- Level of evidence for key findings
- Relevant UAE-specific considerations
- Recent publication dates where available
- Clinical practice implications"""

    def execute(
        self,
        request: Dict,
        user_context: Dict,
        knowledge_base: SharedKnowledgeBase,
    ) -> Dict[str, Any]:
        """
        Conduct medical research with iterative refinement.
        """
        research_query = request.get("query", "")
        context = request.get("context", "")

        logger.info(f"Starting medical research: {research_query[:80]}...")

        # Step 1: Initial research plan
        plan = self._create_research_plan(research_query)

        # Step 2: Iterative retrieval and refinement
        all_findings = []
        current_query = research_query

        for iteration in range(self.max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{self.max_iterations}")

            # Search knowledge bases
            kb_results = knowledge_base.search_all(current_query, top_k=3)

            # Collect all results
            iteration_findings = []
            for source, results in kb_results.items():
                for r in results:
                    iteration_findings.append({
                        "source": source,
                        "text": r.get("text", ""),
                        "score": r.get("score", 0),
                        "iteration": iteration + 1,
                    })

            all_findings.extend(iteration_findings)

            # Evaluate if sufficient
            if iteration < self.max_iterations - 1:
                evaluation = self._evaluate_sufficiency(research_query, all_findings)
                if evaluation.get("sufficient", False):
                    logger.info("Research sufficient, stopping early.")
                    break

                # Refine query for next iteration
                gaps = evaluation.get("gaps", [])
                if gaps:
                    current_query = f"{research_query} {' '.join(gaps)}"

        # Step 3: Synthesize final report
        report = self._synthesize_report(research_query, all_findings, context)

        return {
            "research_report": report,
            "sources": [
                {"text": f["text"][:200], "source": f["source"]}
                for f in all_findings
            ],
            "iterations": min(self.max_iterations, 2),
            "total_sources": len(all_findings),
            "evidence_level": self._assess_evidence_level(all_findings),
            "subsystem": "medical_research",
        }

    def _create_research_plan(self, query: str) -> Dict:
        """Create a research plan for the query."""
        prompt = f"""Create a research plan for this medical query:

Query: {query}

Return JSON:
{{
    "key_questions": ["sub-question 1", "sub-question 2"],
    "search_terms": ["term1", "term2"],
    "expected_info_types": ["treatment", "diagnosis", "epidemiology", "guidelines"],
    "sources_to_check": ["clinical_guidelines", "medical_literature"]
}}"""
        return self.generate_structured(prompt)

    def _evaluate_sufficiency(self, query: str, findings: List[Dict]) -> Dict:
        """Evaluate if current findings are sufficient."""
        findings_text = "\n".join([f.get("text", "")[:200] for f in findings[-5:]])

        prompt = f"""Evaluate if we have sufficient information to answer this research query.

Query: {query}

Current Findings:
{findings_text}

Return JSON:
{{
    "sufficient": true/false,
    "reasoning": "brief explanation",
    "gaps": ["what's still missing"]
}}"""
        return self.generate_structured(prompt)

    def _synthesize_report(self, query: str, findings: List[Dict], context: str) -> str:
        """Synthesize all findings into a comprehensive research report."""
        findings_text = "\n\n".join([
            f"[Source: {f['source']}] {f['text']}"
            for f in findings[:10]
        ])

        prompt = f"""Synthesize the following research findings into a comprehensive medical report.

Research Query: {query}
Context: {context}

Findings:
{findings_text}

Write a comprehensive report including:
1. EXECUTIVE SUMMARY: Brief overview
2. KEY FINDINGS: Main evidence-based findings
3. EVIDENCE QUALITY: Assessment of available evidence
4. UAE CONTEXT: Relevance to UAE healthcare
5. RECOMMENDATIONS: Clinical recommendations
6. REFERENCES: Key sources cited"""
        return self.generate(prompt)

    def _assess_evidence_level(self, findings: List[Dict]) -> str:
        """Assess the overall level of evidence."""
        if len(findings) >= 5:
            return "B"  # Good evidence from multiple sources
        elif len(findings) >= 2:
            return "C"  # Limited evidence
        return "D"  # Expert opinion / insufficient evidence
