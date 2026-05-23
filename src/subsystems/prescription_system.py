"""
Subsystem 2: Prescription Verification (RAG + Human-in-Loop)
Automated drug safety checking with pharmacist review integration.
"""

import json
import re
from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.core.knowledge_base import SharedKnowledgeBase
from src.utils.logger import logger


class PrescriptionVerificationSystem(BaseHealthcareAgent):
    """Automated prescription verification with RAG and human-in-loop."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash-lite", temperature=0.1)

    def system_prompt(self) -> str:
        return """You are an expert Clinical Pharmacist in the UAE Integrated Healthcare Platform.
Your role is to verify prescriptions for safety, interactions, and UAE MOH compliance.

Analyze for:
1. Drug-Drug Interactions (critical)
2. Drug-Allergy/Contraindications
3. Dosage appropriateness
4. Duplicate therapy
5. Pregnancy/lactation safety
6. UAE MOH regulatory compliance

Return structured JSON with findings and recommendations."""

    def execute(
        self,
        request: Dict,
        user_context: Dict,
        knowledge_base: SharedKnowledgeBase,
    ) -> Dict[str, Any]:
        """
        Verify prescription with RAG and determine if human review is needed.
        """
        prescription_text = request.get("query", "")
        patient_profile = request.get("patient_profile", {})

        logger.info("Starting prescription verification")

        # Step 1: Extract drugs from prescription
        drugs = self._extract_drugs(prescription_text)

        # Step 2: Retrieve drug knowledge from databases
        drug_knowledge = []
        for drug in drugs:
            info = knowledge_base.get_drug_info(drug)
            if info:
                drug_knowledge.append(info)

        # Step 3: Run AI safety analysis
        analysis = self._analyze_safety(prescription_text, patient_profile, drugs, drug_knowledge)

        # Step 4: Determine review requirements
        requires_review = (
            analysis.get("risk_level") == "HIGH"
            or len(analysis.get("interactions_found", [])) > 0
            or len(analysis.get("contraindications_found", [])) > 0
        )

        return {
            "verification": analysis,
            "identified_drugs": drugs,
            "knowledge_sources_used": len(drug_knowledge),
            "status": "pending_pharmacist_review" if requires_review else "auto_approved",
            "auto_dispense": not requires_review,
            "pharmacist_action_required": requires_review,
            "subsystem": "prescription_verify",
        }

    def _extract_drugs(self, text: str) -> List[str]:
        """Extract drug names from prescription text using LLM."""
        prompt = f"""Extract all medication/drug names from this prescription text.
Return ONLY a JSON array of drug names. If no drugs found, return [].

Prescription: {text}"""
        result = self.generate_structured(prompt)
        if isinstance(result, dict):
            drugs = result.get("data", [])
            if isinstance(drugs, list):
                return drugs
        return []

    def _analyze_safety(
        self,
        prescription: str,
        patient: Dict,
        drugs: List[str],
        knowledge: List[Dict],
    ) -> Dict[str, Any]:
        """Analyze prescription safety using LLM."""
        knowledge_text = "\n".join([k.get("text", "") for k in knowledge])
        patient_text = json.dumps(patient, indent=2) if patient else "No patient profile provided"

        prompt = f"""Analyze this prescription for safety:

Prescription Details:
{prescription}

Patient Profile:
{patient_text}

Identified Drugs: {', '.join(drugs) if drugs else 'None'}

Drug Knowledge Base:
{knowledge_text}

Return JSON:
{{
    "risk_level": "LOW|MEDIUM|HIGH",
    "is_safe_to_dispense": true/false,
    "interactions_found": ["interaction details or empty list"],
    "contraindications_found": ["contraindication details or empty list"],
    "dosage_warnings": ["dosage warnings or empty list"],
    "pharmacist_recommendation": "Clear instruction for the pharmacist",
    "moh_compliance": "Compliant|Non-compliant|Not specified"
}}"""
        return self.generate_structured(prompt)

    def extract_drugs_from_text(self, prescription_text: str) -> List[str]:
        """Public method for external use."""
        return self._extract_drugs(prescription_text)
