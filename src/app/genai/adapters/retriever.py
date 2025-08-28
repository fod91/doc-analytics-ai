from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Mapping, Any

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, k: int = 5) -> list[Mapping[str, Any]]:
        raise NotImplementedError
