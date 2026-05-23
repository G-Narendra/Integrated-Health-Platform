"""
Anthropic client (optional LLM provider).
Currently a placeholder for future integration.
"""

from typing import Any, Dict

from src.core.llm_base import LLMBase
from src.utils.logger import logger


class AnthropicClient(LLMBase):
    """Anthropic LLM client (optional provider).
    Requires anthropic package and API key."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                raise
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            message = self.client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"Anthropic generation error: {e}"

    def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        try:
            message = self.client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"Anthropic generation error: {e}"

    def generate_structured(self, system_prompt: str, user_prompt: str, response_format: Dict, **kwargs) -> Dict[str, Any]:
        text = self.generate_with_system(
            f"{system_prompt}\n\nRespond in valid JSON format.",
            user_prompt,
        )
        import json
        import re
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"raw_response": text}

    def stream_generate(self, system_prompt: str, user_prompt: str, **kwargs):
        with self.client.messages.stream(
            model=kwargs.get("model", self.model),
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
