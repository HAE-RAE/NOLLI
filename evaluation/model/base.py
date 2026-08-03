import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Common interface for LLM clients."""

    def __init__(self, model: str, timeout: float = 600.0, gen_kwargs: Optional[Dict] = None):
        self.model = model
        self.timeout = timeout
        self.gen_kwargs = gen_kwargs or {}

    @abstractmethod
    def generate(self, messages: List[Dict]) -> Tuple[str, Dict]:
        ...

    @abstractmethod
    async def _async_generate(
        self, messages: List[Dict], max_retries: int = 3
    ) -> Tuple[str, Dict]:
        ...

    async def async_batch_generate(
        self,
        messages_list: List[List[Dict]],
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, Dict]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(messages_list)
        completed = 0
        completed_lock = asyncio.Lock()

        async def _run(messages: List[Dict], index: int):
            nonlocal completed
            try:
                async with semaphore:
                    result = await self._async_generate(messages)
                    async with completed_lock:
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    return result
            except Exception:
                async with completed_lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                raise

        tasks = [_run(m, i) for i, m in enumerate(messages_list)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[Tuple[str, Dict]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error in async request {i + 1}: {result}")
                processed.append(("", {"error": str(result), "tokens": 0}))
            else:
                processed.append(result)
        return processed

    @abstractmethod
    def __repr__(self) -> str:
        ...
