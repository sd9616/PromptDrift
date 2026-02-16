# LiteLLM wrapper - unified interface to call Claude, Gemini, Deepseek, Llama.
# Provides generate() with retry logic and consistent settings (temperature=0, max_tokens=4096).

import os
import time

from src.llm.open_source import llama, deepseek
from src.llm.closed_source import claude, gemini

# Registry: friendly_name -> (MODEL_ID, ENV_KEY)
_MODEL_REGISTRY = {
    "Llama": (llama.MODEL_ID, llama.ENV_KEY),
    "Deepseek": (deepseek.MODEL_ID, deepseek.ENV_KEY),
    "Claude": (claude.MODEL_ID, claude.ENV_KEY),
    "Gemini": (gemini.MODEL_ID, gemini.ENV_KEY),
}

# Consistency settings applied to all models
TEMPERATURE = 0
MAX_TOKENS = 4096
TOP_P = 1.0
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3


def _get_model_and_key(model_name: str) -> tuple[str, str]:
    """Resolve friendly name to LiteLLM model string and ensure API key is set."""
    entry = _MODEL_REGISTRY.get(model_name)
    if not entry:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(_MODEL_REGISTRY.keys())}")
    model_id, env_key = entry
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"API key not set. Set {env_key} in .env")
    return model_id, api_key


def generate(prompt_text: str, model: str) -> tuple[str, int]:
    """
    Call the specified LLM with the prompt. Uses retry logic and consistent settings.

    Args:
        prompt_text: The prompt to send.
        model: Friendly name: "Claude", "Gemini", "Deepseek", "Llama".

    Returns:
        (output_text, tokens_used)
    """
    model_id, _ = _get_model_and_key(model)

    try:
        import litellm
    except ImportError:
        raise ImportError("Install litellm: pip install litellm")

    # Claude API does not allow both temperature and top_p
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    if model != "Claude":
        kwargs["top_p"] = TOP_P
    if model == "Gemini":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 128}

    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = litellm.completion(**kwargs)
            output_text = response.choices[0].message.content or ""
            if response.usage and hasattr(response.usage, "total_tokens"):
                tokens_used = response.usage.total_tokens
            elif response.usage:
                tokens_used = getattr(response.usage, "prompt_tokens", 0) + getattr(
                    response.usage, "completion_tokens", 0
                )
            else:
                tokens_used = 0
            return (output_text, tokens_used)
        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    raise last_error or RuntimeError("LLM call failed")
