from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Any

class LLMClient(ABC):
    @abstractmethod
    def answer(self, question: str, contexts: Sequence[Mapping[str, Any]]) -> str:
        raise NotImplementedError