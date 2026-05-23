"""
Orchestrator agent that coordinates multiple agents for complex tasks.
"""

from typing import Any, Dict, List

from src.agents.base_agent import BaseHealthcareAgent
from src.utils.logger import logger


class AgentOrchestrator(BaseHealthcareAgent):
    """Coordinates multiple specialized agents for complex healthcare workflows."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.2)

    def system_prompt(self) -> str:
        return """You are the Agent Orchestrator for the UAE Integrated Healthcare Platform.
Your role is to:
1. Analyze complex healthcare requests
2. Break them down into subtasks
3. Route subtasks to appropriate specialized agents
4. Synthesize results into a coherent response

Available agents:
- Clinical Diagnosis Agent: For symptom analysis and diagnosis
- Prescription Verification Agent: For drug safety checking
- Medical Records Agent: For patient record management
- Appointment Agent: For scheduling and calendar management
- Medical Research Agent: For literature search and evidence-based medicine"""

    def decompose_request(self, request: str) -> List[Dict]:
        """Decompose a complex request into subtasks."""
        prompt = f"""Analyze this healthcare request and break it into subtasks.

Request: {request}

Return JSON array of subtasks:
[
  {{
    "task_type": "diagnosis|prescription|records|appointment|research",
    "description": "What needs to be done",
    "priority": "high|medium|low"
  }}
]"""
        result = self.generate_structured(prompt)
        tasks = result.get("data", result) if isinstance(result, dict) else []
        if isinstance(tasks, dict) and "error" not in tasks:
            return [tasks]
        return tasks if isinstance(tasks, list) else []

    def synthesize_results(
        self, original_request: str, results: List[Dict]
    ) -> str:
        """Synthesize results from multiple agents into a coherent response."""
        results_text = "\n\n".join([
            f"[{r.get('agent', 'Unknown')}]: {r.get('result', 'No result')}"
            for r in results
        ])

        prompt = f"""Synthesize these agent results into a coherent response for the original request.

Original Request: {original_request}

Agent Results:
{results_text}

Provide a comprehensive, well-organized response that addresses all aspects of the request."""
        return self.generate(prompt)
