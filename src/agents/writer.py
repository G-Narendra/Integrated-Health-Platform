"""
Writer/Synthesis agent for generating clinical reports and summaries.
"""

from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent


class WriterAgent(BaseHealthcareAgent):
    """Agent specialized in generating clinical reports and summaries."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.1)

    def system_prompt(self) -> str:
        return """You are a Medical Writer Agent for the UAE Integrated Healthcare Platform.
Your role is to:
1. Generate clear, professional clinical reports
2. Summarize complex medical information for different audiences
3. Format reports following UAE MOH standards
4. Ensure all reports include required disclaimers
5. Adapt language for physicians, patients, or administrators

Write in a clear, professional tone. Use standard medical abbreviations.
Always include:
- Date and time
- Patient identification (when available)
- Clear findings and recommendations
- Physician review disclaimer"""

    def write_report(
        self, data: Dict[str, Any], report_type: str = "clinical", audience: str = "physician"
    ) -> str:
        """Generate a clinical report from structured data."""
        data_str = "\n".join([f"{k}: {v}" for k, v in data.items() if v])

        prompt = f"""Generate a {report_type} report for {audience} audience.

Data:
{data_str}

Write a professional, clear report following UAE healthcare standards."""
        return self.generate(prompt)

    def write_summary(
        self, text: str, max_length: int = 500, audience: str = "patient"
    ) -> str:
        """Write a patient-friendly summary of medical information."""
        prompt = f"""Summarize the following medical information for a {audience} audience.
Keep it under {max_length} words. Use simple language. Avoid jargon.

{text}"""
        return self.generate(prompt)
