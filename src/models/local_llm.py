"""
Local LLM integration for fine-tuned medical models.
Placeholder for future local model deployment.
"""

from typing import Any, Dict, Optional

from src.core.llm_base import LLMBase
from src.utils.logger import logger


class LocalLLM(LLMBase):
    """Local LLM interface for fine-tuned medical models.
    Currently a placeholder - requires downloading a compatible model."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Attempt to load a local model. Placeholder implementation."""
        logger.warning("LocalLLM: No local model is loaded. This feature requires a fine-tuned model.")
        self.model = None
        self.tokenizer = None

    def generate(self, prompt: str, **kwargs) -> str:
        if self.model is None:
            return "Local model not loaded. Please configure a local model path or use the default Gemini provider."
        return "Local model generation (not yet implemented)"

    def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return self.generate(f"{system_prompt}\n\n{user_prompt}")

    def generate_structured(self, system_prompt: str, user_prompt: str, response_format: Dict, **kwargs) -> Dict[str, Any]:
        return {"error": "Local model structured generation not yet implemented"}

    def stream_generate(self, system_prompt: str, user_prompt: str, **kwargs):
        yield "Local model streaming not yet implemented"
