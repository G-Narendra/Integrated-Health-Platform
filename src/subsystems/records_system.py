"""
Subsystem 3: Medical Records (Fine-Tuned Model)
Processes medical records with bilingual Arabic/English support using Gemini.
"""

import json
import re
from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.core.knowledge_base import SharedKnowledgeBase
from src.utils.logger import logger


class MedicalRecordsSystem(BaseHealthcareAgent):
    """
    Medical record processing with Gemini-powered bilingual support.
    Handles Arabic/English medical terminology, summarization, extraction, and translation.
    """

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.1)

    def system_prompt(self) -> str:
        return """You are a Medical Records Specialist in the UAE Integrated Healthcare Platform.
You handle BILINGUAL medical records in both Arabic and English.

Your responsibilities:
1. Summarize clinical notes and records
2. Extract structured data (diagnoses, medications, labs, vitals)
3. Translate between Arabic and English medical terminology
4. Maintain medical accuracy and context
5. Follow UAE MOH documentation standards

Always preserve medical accuracy. Never add or change clinical information."""

    def execute(
        self,
        request: Dict,
        user_context: Dict,
        knowledge_base: SharedKnowledgeBase,
    ) -> Dict[str, Any]:
        """
        Process medical record requests based on task type.
        """
        task = request.get("task", "summarize")
        record_text = request.get("query", "")

        logger.info(f"Processing medical records request - Task: {task}")

        if task == "summarize":
            return self._summarize_record(record_text)
        elif task == "extract":
            return self._extract_structured_data(record_text)
        elif task == "translate":
            target_lang = request.get("target_language", "en")
            return self._translate_record(record_text, target_lang)
        else:
            return {"error": f"Unknown task: {task}", "subsystem": "medical_records"}

    def _detect_language(self, text: str) -> str:
        """Detect if text is Arabic, English, or mixed."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total = arabic_chars + english_chars
        if total == 0:
            return "unknown"
        ratio = arabic_chars / total
        if ratio > 0.7:
            return "arabic"
        elif ratio < 0.3:
            return "english"
        return "mixed"

    def _summarize_record(self, record_text: str) -> Dict[str, Any]:
        """Generate a clinical summary of the medical record."""
        lang = self._detect_language(record_text)

        if lang == "arabic":
            prompt = f"""لخص هذا السجل الطبي مع التركيز على:
- التشخيص الرئيسي
- الأدوية الموصوفة
- نتائج الفحوصات المهمة
- التوصيات

السجل الطبي:
{record_text}

أعد النتيجة باللغة العربية بصيغة JSON:
{{
    "summary": "...",
    "key_diagnoses": ["..."],
    "medications": ["..."],
    "lab_highlights": ["..."],
    "recommendations": ["..."]
}}"""
        else:
            prompt = f"""Summarize this medical record focusing on:
- Primary diagnosis
- Prescribed medications
- Important lab results
- Recommendations

Record:
{record_text}

Return JSON:
{{
    "summary": "...",
    "key_diagnoses": ["..."],
    "medications": ["..."],
    "lab_highlights": ["..."],
    "recommendations": ["..."]
}}"""
        result = self.generate_structured(prompt)
        return {
            "summary": result.get("summary", result.get("raw_response", "Summary generated.")),
            "structured_data": result,
            "language": lang,
            "subsystem": "medical_records",
        }

    def _extract_structured_data(self, record_text: str) -> Dict[str, Any]:
        """Extract structured clinical data from the medical record."""
        prompt = f"""Extract structured clinical data from this medical record.

Record:
{record_text}

Return JSON with these fields (use empty arrays/lists for missing data):
{{
    "patient_info": {{
        "name": "",
        "age": "",
        "gender": "",
        "blood_group": ""
    }},
    "diagnoses": ["diagnosis1", "diagnosis2"],
    "medications": [
        {{
            "name": "",
            "dosage": "",
            "frequency": ""
        }}
    ],
    "lab_results": [
        {{
            "test": "",
            "value": "",
            "reference_range": "",
            "flag": "normal|high|low|critical"
        }}
    ],
    "vital_signs": {{
        "bp": "",
        "hr": "",
        "temp": "",
        "rr": "",
        "spo2": ""
    }},
    "allergies": ["allergy1"],
    "recommendations": ["recommendation1"]
}}"""
        result = self.generate_structured(prompt)
        return {
            "structured_data": result,
            "extraction_success": "error" not in result,
            "subsystem": "medical_records",
        }

    def _translate_record(self, record_text: str, target_lang: str) -> Dict[str, Any]:
        """Translate a medical record to the target language."""
        source_lang = self._detect_language(record_text)

        if source_lang == target_lang:
            return {
                "translated_text": record_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "note": "Source and target languages are the same.",
                "subsystem": "medical_records",
            }

        lang_name = "Arabic" if target_lang == "ar" else "English"
        prompt = f"""Translate this medical record to {lang_name}.
Preserve all medical terminology, numbers, and abbreviations.
Maintain clinical accuracy.

Source ({source_lang}):
{record_text}

Translation to {lang_name}:"""
        translation = self.generate(prompt)
        return {
            "translated_text": translation,
            "source_language": source_lang,
            "target_language": target_lang,
            "subsystem": "medical_records",
        }
