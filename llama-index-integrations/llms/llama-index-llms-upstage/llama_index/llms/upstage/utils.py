import logging
from typing import Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_UPSTAGE_API_BASE = "https://api.upstage.ai/v1/solar"
CHAT_MODELS = {
    "solar-1-mini-chat": 32768,
}

ALL_AVAILABLE_MODELS = {**CHAT_MODELS}

logger = logging.getLogger(__name__)


def resolve_upstage_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Resolve Upstage credentials.

    The order of precedence is:
    1. param
    2. env
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "UPSTAGE_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "UPSTAGE_API_BASE", "")

    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_UPSTAGE_API_BASE

    return final_api_key, str(final_api_base)


def is_chat_model(model: str) -> bool:
    return model in CHAT_MODELS


def is_function_calling_model(model: str) -> bool:
    return is_chat_model(model)


def upstage_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in ALL_AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Upstage model name in: {', '.join(ALL_AVAILABLE_MODELS.keys())}"
        )
    return CHAT_MODELS[modelname]
