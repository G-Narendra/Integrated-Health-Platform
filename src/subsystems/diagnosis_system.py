"""
Subsystem 1: Clinical Diagnosis (Multi-Agent)
Provides multi-agent diagnostic support using LangGraph-style workflow with Gemini.
"""

import json
import re
from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.core.knowledge_base import SharedKnowledgeBase
from src.utils.logger import logger


class SpecialistAgent(BaseHealthcareAgent):
    """Base class for specialist agents in the diagnostic workflow."""

    def __init__(self, specialty: str, system_prompt_text: str):
        super().__init__(model="gemini-2.5-flash-lite", temperature=0.2)
        self.specialty = specialty
        self._system_prompt_text = system_prompt_text

    def system_prompt(self) -> str:
        return self._system_prompt_text


class ClinicalDiagnosisSystem:
    """Multi-agent diagnostic support system."""

    def __init__(self):
        self.specialists = self._create_specialists()

    def _create_specialists(self) -> Dict[str, SpecialistAgent]:
        """Create the specialist agents."""
        cardiology_prompt = """You are a Cardiologist specialist in the UAE Integrated Healthcare Platform.
Review the patient case and provide:
1. Cardiac assessment and differential diagnoses
2. ECG interpretation
3. Cardiac enzyme analysis
4. Risk stratification
5. Recommendations

Format your response as JSON with keys: assessment, differential_diagnoses, ecg_findings, lab_analysis, risk_level, recommendations, red_flags"""

        radiology_prompt = """You are a Radiologist specialist in the UAE Integrated Healthcare Platform.
Review the patient case and provide:
1. Imaging findings and interpretation
2. Differential diagnoses based on imaging
3. Recommended additional imaging
4. Critical findings (if any)

Format your response as JSON with keys: findings, differential_diagnoses, recommended_imaging, critical_findings, confidence"""

        lab_prompt = """You are a Laboratory Medicine specialist in the UAE Integrated Healthcare Platform.
Review the patient case and provide:
1. Lab result analysis and interpretation
2. Critical values identification
3. Recommended additional tests
4. Correlation with clinical presentation

Format your response as JSON with keys: lab_analysis, critical_values, recommended_tests, clinical_correlation, confidence"""

        em_prompt = """You are an Emergency Medicine specialist in the UAE Integrated Healthcare Platform.
Review the patient case and provide:
1. ABCDE assessment
2. Triage level and acuity
3. Immediate management priorities
4. Disposition recommendation

Format your response as JSON with keys: abcde_assessment, triage_level, immediate_actions, disposition, red_flags"""

        return {
            "cardiology": SpecialistAgent("Cardiology", cardiology_prompt),
            "radiology": SpecialistAgent("Radiology", radiology_prompt),
            "lab": SpecialistAgent("Laboratory Medicine", lab_prompt),
            "emergency": SpecialistAgent("Emergency Medicine", em_prompt),
        }

    def execute(
        self,
        request: Dict,
        user_context: Dict,
        knowledge_base: SharedKnowledgeBase,
    ) -> Dict[str, Any]:
        """
        Execute multi-agent diagnosis workflow:
        Supervisor → [Cardiology, Radiology, Lab, EM] (parallel) → Critic → Synthesis
        """
        patient_case = request.get("query", "")
        patient_id = request.get("patient_id", "Unknown")

        logger.info(f"Starting multi-agent diagnosis for patient {patient_id}")

        # Retrieve relevant guidelines
        guidelines = knowledge_base.query_guidelines(patient_case, top_k=3)

        # Build enriched case with knowledge base context
        enriched_case = f"""Patient ID: {patient_id}
Case: {patient_case}

Relevant Clinical Guidelines:
{chr(10).join([g.get('text', '') for g in guidelines])}
"""

        # Run specialists in parallel
        findings = {}
        for specialty, agent in self.specialists.items():
            try:
                logger.info(f"Consulting {specialty} specialist...")
                result = agent.generate_structured(enriched_case)
                findings[specialty] = result
            except Exception as e:
                logger.error(f"{specialty} specialist failed: {e}")
                findings[specialty] = {"error": str(e), "specialty": specialty}

        # Critic review - cross-validate findings
        critic = self._critic_review(patient_case, findings)

        # Synthesis - generate final report
        final_report = self._synthesize(patient_case, findings, critic)

        return {
            "diagnosis": final_report,
            "findings": [
                {
                    "specialty": k.replace("_", " ").title(),
                    "findings": v.get("assessment", v.get("findings", v.get("lab_analysis", v.get("abcde_assessment", str(v))))),
                    "confidence": v.get("confidence", 0.7),
                    "red_flags": v.get("red_flags", v.get("critical_values", v.get("critical_findings", []))),
                    "recommendations": v.get("recommendations", v.get("recommended_imaging", v.get("recommended_tests", v.get("immediate_actions", [])))),
                }
                for k, v in findings.items()
            ],
            "subsystem": "multi_agent_diagnosis",
            "critic_review": critic,
            "final_report": final_report.get("report", "Report generated."),
            "primary_diagnosis": final_report.get("primary_diagnosis", "See report"),
            "confidence": final_report.get("confidence", 0.7),
            "requires_physician_review": True,
        }

    def _critic_review(self, case: str, findings: Dict) -> str:
        """Cross-validate findings from all specialists."""
        findings_text = json.dumps(findings, indent=2)
        prompt = f"""As a Senior Consultant (Critic), review these specialist findings:

Patient Case: {case[:500]}

Specialist Findings:
{findings_text}

Evaluate:
1. Are there any contradictions between specialists?
2. What is the most likely diagnosis?
3. What additional information would help?
4. Rate the consensus level (high/medium/low)"""
        critic = self.specialists["emergency"]
        return critic.generate(prompt)

    def _synthesize(self, case: str, findings: Dict, critic: str) -> Dict:
        """Synthesize all findings into a final diagnostic report."""
        findings_text = json.dumps(findings, indent=2)
        prompt = f"""Synthesize the following diagnostic findings into a final report.

Patient Case: {case[:500]}

Specialist Findings:
{findings_text}

Critic Review: {critic}

Return JSON with:
- primary_diagnosis: The most likely diagnosis
- differential_diagnoses: List of alternative diagnoses
- confidence: Confidence level (0.0-1.0)
- recommended_actions: List of next steps
- report: A comprehensive narrative report"""
        synthesizer = self.specialists["cardiology"]
        return synthesizer.generate_structured(prompt)
