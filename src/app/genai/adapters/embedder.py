from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
