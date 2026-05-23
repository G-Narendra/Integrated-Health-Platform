"""
Base class for LLM integrations. Provides a unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBase(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    def generate_with_system(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> str:
        """Generate with a system prompt."""
        ...

    @abstractmethod
    def generate_structured(
        self, system_prompt: str, user_prompt: str, response_format: Dict, **kwargs
    ) -> Dict[str, Any]:
        """Generate a structured (JSON) response from the LLM."""
        ...

    @abstractmethod
    def stream_generate(
        self, system_prompt: str, user_prompt: str, **kwargs
    ):
        """Stream tokens from LLM generation. Yields strings."""
        ...
