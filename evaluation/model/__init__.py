from .base import BaseLLMClient
from .litellm import LiteLLMClient
from .vllm import VLLMClient


def create_client(
    model: str,
    timeout: float = 600.0,
    vllm_url: str | None = None,
    gen_kwargs: dict | None = None,
) -> BaseLLMClient:
    """Create an appropriate LLM client based on model_router."""
    if vllm_url:
        return VLLMClient(model, timeout, vllm_url, gen_kwargs)
    return LiteLLMClient(model, timeout, gen_kwargs)


__all__ = [
    "BaseLLMClient",
    "LiteLLMClient",
    "VLLMClient",
    "create_client",
]
