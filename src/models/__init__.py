"""
Models package init. Provides LLM client abstractions.
Primary: Gemini (default)
Fallback: OpenAI (optional)
"""

from src.models.embedding_models import EmbeddingModelManager
from src.models.openai_client import OpenAIClient
