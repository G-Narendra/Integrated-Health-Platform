"""
OpenAI client as a fallback LLM provider for the healthcare platform.
"""

from typing import Any, Dict, Optional

from src.core.llm_base import LLMBase
from src.utils.logger import logger


class OpenAIClient(LLMBase):
    """OpenAI LLM client for fallback scenarios."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"OpenAI generation error: {e}"

    def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", self.temperature),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"OpenAI generation error: {e}"

    def generate_structured(self, system_prompt: str, user_prompt: str, response_format: Dict, **kwargs) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=kwargs.get("temperature", self.temperature),
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

    def stream_generate(self, system_prompt: str, user_prompt: str, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
