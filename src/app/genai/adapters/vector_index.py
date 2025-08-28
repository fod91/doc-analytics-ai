from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Any
import numpy as np

class VectorIndex(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, metadatas: Iterable[Mapping[str, Any]], ids: Iterable[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vectors: np.ndarray, *, k: int = 5) -> list[list[Mapping[str, Any]]]:
        raise NotImplementedError