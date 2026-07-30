"""
Application configuration for the ONSU Kitchen Assistant.

All configuration is read from environment variables where possible,
with sensible defaults for local development.
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = Path(
    os.getenv(
        "CORPUS_PATH",
        PROJECT_ROOT / "corpus"
    )
)

VECTOR_DB_PATH = Path(
    os.getenv(
        "VECTOR_DB_PATH",
        PROJECT_ROOT / "vector_db" / "chroma"
    )
)

# -----------------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv(
"OPENAI_API_KEY"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gpt-4.1-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

TOP_K = int(
    os.getenv(
        "TOP_K",
        "4"
    )
)

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------

ONSU_PASSCODE = os.getenv("ONSU_PASSCODE")

# -----------------------------------------------------------------------------
# Chroma
# -----------------------------------------------------------------------------

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "onsu_corpus"
)

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set."
    )