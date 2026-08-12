"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Reduce Chroma / PostHog telemetry noise in CLI apps
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")


def _project_path(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath(*parts)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the food RAG system."""

    project_root: Path = _PROJECT_ROOT
    data_path: Path = _project_path("data", "FoodDataSet.json")
    chroma_persist_dir: Path = _project_path("chroma_db")
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chroma collections
    rag_collection_name: str = "enhanced_rag_food_chatbot"
    search_collection_name: str = "food_similarity_search"

    # Retrieval
    default_top_k: int = 3
    max_conversation_history: int = 5

    # IBM watsonx.ai / Granite
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_api_key: str | None = None
    watsonx_project_id: str = "skills-network"
    watsonx_model_id: str = "ibm/granite-4-h-small"
    watsonx_max_new_tokens: int = 400
    watsonx_verify: bool = False

    # Behavior
    allow_llm_fallback: bool = True
    skip_llm_healthcheck: bool = False


def get_settings() -> Settings:
    """Build settings from environment with sensible defaults."""
    data_path = os.getenv("FOOD_DATA_PATH")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR")
    verify_raw = os.getenv("WATSONX_VERIFY", "false").lower()
    skip_health = os.getenv("SKIP_LLM_HEALTHCHECK", "false").lower()

    return Settings(
        data_path=Path(data_path) if data_path else _project_path("data", "FoodDataSet.json"),
        chroma_persist_dir=(
            Path(chroma_dir) if chroma_dir else _project_path("chroma_db")
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        rag_collection_name=os.getenv(
            "RAG_COLLECTION_NAME", "enhanced_rag_food_chatbot"
        ),
        search_collection_name=os.getenv(
            "SEARCH_COLLECTION_NAME", "food_similarity_search"
        ),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "3")),
        max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "5")),
        watsonx_url=os.getenv(
            "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
        ),
        watsonx_api_key=os.getenv("WATSONX_API_KEY") or None,
        watsonx_project_id=os.getenv("WATSONX_PROJECT_ID", "skills-network"),
        watsonx_model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small"),
        watsonx_max_new_tokens=int(os.getenv("WATSONX_MAX_NEW_TOKENS", "400")),
        watsonx_verify=verify_raw in {"1", "true", "yes"},
        allow_llm_fallback=os.getenv("ALLOW_LLM_FALLBACK", "true").lower()
        in {"1", "true", "yes"},
        skip_llm_healthcheck=skip_health in {"1", "true", "yes"},
    )
