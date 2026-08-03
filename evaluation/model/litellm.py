import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import litellm
from litellm import completion, acompletion

from .base import BaseLLMClient

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

if os.getenv("LITELLM_DEBUG", "false").lower() == "true":
    litellm.set_verbose = True
else:
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


ANTHROPIC_ADAPTIVE_THINKING_MODELS = ("claude-opus-4-8",)


class LiteLLMClient(BaseLLMClient):
    """LLM client via liteLLM (Gemini, OpenAI, etc.)."""

    def __init__(self, model: str, timeout: float = 600.0, gen_kwargs: Optional[Dict] = None):
        super().__init__(model, timeout, gen_kwargs)

    def _uses_adaptive_thinking(self) -> bool:
        return any(m in self.model for m in ANTHROPIC_ADAPTIVE_THINKING_MODELS)

    def _prepare_params(self, messages: List[Dict]) -> Dict:
        params: Dict = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }
        if "temperature" in self.gen_kwargs:
            params["temperature"] = self.gen_kwargs["temperature"]
        if "max_tokens" in self.gen_kwargs:
            params["max_tokens"] = self.gen_kwargs["max_tokens"]
        if "top_p" in self.gen_kwargs:
            params["top_p"] = self.gen_kwargs["top_p"]
        if "top_k" in self.gen_kwargs:
            params["top_k"] = self.gen_kwargs["top_k"]
        if "reasoning_effort" in self.gen_kwargs:
            if self._uses_adaptive_thinking():
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": self.gen_kwargs["reasoning_effort"]}
            else:
                params["reasoning_effort"] = self.gen_kwargs["reasoning_effort"]
        return params

    @staticmethod
    def _finalize_thinking_from_content(content: str, thinking: str) -> Tuple[str, str]:
        """Align with VLLMClient: API reasoning fields first, else <redacted_thinking> in body."""
        if not thinking and content and "<redacted_thinking>" in content:
            m = re.search(r"<redacted_thinking>(.*?)</think>", content, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                content = re.sub(
                    r"<redacted_thinking>.*?</redacted_thinking>",
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip()
        return content, thinking

    @staticmethod
    def _message_content_and_thinking(message: Any) -> Tuple[str, str]:
        """LiteLLM / OpenAI-style message: reasoning_content or reasoning, optional tags in content."""
        content = getattr(message, "content", None)
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        thinking = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        if thinking and not isinstance(thinking, str):
            thinking = str(thinking)

        return LiteLLMClient._finalize_thinking_from_content(content, thinking)

    def _extract_response(self, response, latency_ms: float) -> Tuple[str, Dict]:
        choice = response.choices[0]
        message = choice.message
        content, thinking = self._message_content_and_thinking(message)
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
        finish_reason = getattr(choice, "finish_reason", None) or ""
        return (
            content,
            {
                "latency_ms": latency_ms,
                "tokens": tokens or 0,
                "thinking_content": thinking,
                "finish_reason": str(finish_reason),
            },
        )

    def _handle_error(self, e: Exception, latency_ms: float, is_async: bool = False) -> "Exception":
        prefix = "LLM async generation" if is_async else "LLM generation"
        error_msg = f"{prefix} failed after {latency_ms:.0f}ms: {str(e)}"
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg += "\nHint: Check if your API key is set correctly in .env file"
        logger.error(error_msg)
        return Exception(error_msg)

    def generate(self, messages: List[Dict]) -> Tuple[str, Dict]:
        start = time.time()
        try:
            params = self._prepare_params(messages)
            response = completion(**params)
            return self._extract_response(response, (time.time() - start) * 1000)
        except Exception as e:
            raise self._handle_error(e, (time.time() - start) * 1000, is_async=False)

    async def _async_generate(
        self, messages: List[Dict], max_retries: int = 3
    ) -> Tuple[str, Dict]:
        start = time.time()
        for attempt in range(max_retries):
            try:
                params = self._prepare_params(messages)
                response = await acompletion(**params)
                return self._extract_response(response, (time.time() - start) * 1000)
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                if attempt == max_retries - 1:
                    raise self._handle_error(e, latency_ms, is_async=True)
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    f"LLM async generation failed (attempt {attempt + 1}/{max_retries}): "
                    f"{str(e)[:200]}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
        raise RuntimeError("unreachable: retry loop exited without return or raise")

    def __repr__(self) -> str:
        return f"LiteLLMClient(model={self.model}, gen_kwargs={self.gen_kwargs})"
