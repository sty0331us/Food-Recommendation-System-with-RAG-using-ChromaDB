"""IBM watsonx.ai / Granite LLM client with graceful degradation."""

from __future__ import annotations

from typing import Any

from food_rag.config import Settings, get_settings

_model: Any | None = None
_init_error: str | None = None


def get_llm_model(settings: Settings | None = None):
    """
    Lazily initialize and return the watsonx ModelInference instance.

    Returns None if the SDK is missing or credentials/config fail.
    """
    global _model, _init_error

    if _model is not None:
        return _model
    if _init_error is not None:
        return None

    settings = settings or get_settings()

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
    except ImportError as exc:
        _init_error = (
            "ibm-watsonx-ai is not installed. "
            "Install dependencies or rely on retrieval-only fallback responses."
        )
        print(f"⚠️  {_init_error} ({exc})")
        return None

    credentials: dict[str, str] = {"url": settings.watsonx_url}
    if settings.watsonx_api_key:
        credentials["apikey"] = settings.watsonx_api_key

    try:
        _model = ModelInference(
            model_id=settings.watsonx_model_id,
            credentials=credentials,
            params={"max_new_tokens": settings.watsonx_max_new_tokens},
            project_id=settings.watsonx_project_id,
            space_id=None,
            verify=settings.watsonx_verify,
        )
        return _model
    except Exception as exc:
        _init_error = str(exc)
        print(f"⚠️  Failed to initialize LLM: {exc}")
        return None


def healthcheck_llm(settings: Settings | None = None) -> bool:
    """Probe the LLM with a tiny prompt. Returns True on success."""
    settings = settings or get_settings()
    if settings.skip_llm_healthcheck:
        print("⏭️  Skipping LLM health check (SKIP_LLM_HEALTHCHECK=true)")
        return get_llm_model(settings) is not None

    model = get_llm_model(settings)
    if model is None:
        return False

    try:
        response = model.generate(prompt="Hello", params=None)
        return bool(response and "results" in response)
    except Exception as exc:
        print(f"⚠️  LLM health check failed: {exc}")
        return False


def generate_text(prompt: str, settings: Settings | None = None) -> str | None:
    """Generate text from the configured LLM. Returns None on failure."""
    model = get_llm_model(settings)
    if model is None:
        return None

    try:
        response = model.generate(prompt=prompt, params=None)
        if response and "results" in response:
            text = response["results"][0].get("generated_text", "")
            return text.strip() if text else None
    except Exception as exc:
        print(f"❌ LLM Error: {exc}")
    return None
