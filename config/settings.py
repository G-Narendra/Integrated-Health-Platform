"""
Application-level settings for the Integrated Healthcare Platform.
"""

from pathlib import Path

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
APP_NAME = "Integrated Healthcare Platform"
APP_VERSION = "1.0.0"
DEBUG = False

# =============================================================================
# DATA PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
AUDIT_DB_PATH = DATA_DIR / "audit.db"

# =============================================================================
# LLM SETTINGS
# =============================================================================
LLM_MODEL = "gemini-2.5-flash"
LLM_LITE_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-2"

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5

# =============================================================================
# COLLECTION NAMES
# =============================================================================
COLLECTIONS = {
    "clinical_guidelines": "clinical_guidelines",
    "drug_database": "drug_database",
    "moh_regulations": "moh_regulations",
    "medical_literature": "medical_literature",
}

# =============================================================================
# COMPLIANCE
# =============================================================================
HIPAA_ENABLED = True
AUDIT_ALL_ACTIONS = True
PHYSICIAN_REVIEW_REQUIRED = True
