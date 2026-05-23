"""
Base agent class that all healthcare agents extend.
Provides LLM integration, tool management, and standardized response format.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import config
from src.utils.logger import logger


class BaseHealthcareAgent(ABC):
    """Base class for all healthcare AI agents."""

    def __init__(self, model: str = None, temperature: float = 0.1):
        self.model_name = model or config.LLM_MODEL
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
            )
        return self._llm

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    def generate(self, prompt: str, system_override: str = None) -> str:
        """Generate a response from the LLM."""
        system = system_override or self.system_prompt()
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed for {self.__class__.__name__}: {e}")
            return f"I encountered an error. Please try again. ({str(e)})"

    def generate_structured(self, prompt: str, system_override: str = None) -> Dict:
        """Generate a structured JSON response."""
        system = system_override or self.system_prompt()
        full_prompt = f"{prompt}\n\nReturn your response as valid JSON only."
        messages = [SystemMessage(content=system), HumanMessage(content=full_prompt)]
        try:
            response = self.llm.invoke(messages)
            return self._extract_json(response.content)
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {"error": str(e)}

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response text."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return {"data": json.loads(match.group())}
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return {"raw_response": text}
