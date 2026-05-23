"""
Central configuration for the Integrated Healthcare Platform.
Loads settings from .env and config files.
"""

import os
from pathlib import Path
from typing import Optional


class AppConfig:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        # Project paths
        self.BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.DATA_DIR = self.BASE_DIR / "data"
        self.CHROMA_DB_PATH = str(self.DATA_DIR / "chroma_db")
        self.AUDIT_DB_PATH = str(self.DATA_DIR / "audit.db")

        # Create data directories
        self.DATA_DIR.mkdir(exist_ok=True)
        Path(self.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

        # API Keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

        # Model settings
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.LLM_LITE_MODEL = os.getenv("LLM_LITE_MODEL", "gemini-2.5-flash-lite")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

        # System settings
        self.MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

        # App settings
        self.APP_NAME = "Integrated Healthcare Platform"
        self.APP_VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    @property
    def is_gemini_available(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def is_openai_available(self) -> bool:
        return bool(self.OPENAI_API_KEY)


# Global singleton
config = AppConfig()
