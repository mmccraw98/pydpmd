from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional, Callable
import numpy as np


# Centralized dtypes
DT_FLOAT = np.float64
DT_INT = np.int32
DT_UINT = np.uint32


class IndexSpace(Enum):
    Particle = auto()
    Vertex = auto()
    System = auto()
    NoneSpace = auto()


class NeighborMethod(Enum):
    Naive = auto()
    Cell = auto()


@dataclass
class FieldSpec:
    name: str
    index_space: IndexSpace
    dtype: np.dtype
    expected_shape: Optional[Tuple[int, ...]] = None  # static expected shape
    expected_shape_fn: Optional[Callable[[], Tuple[int, ...]]] = None  # dynamic expected shape

    def validate(self, arr: np.ndarray) -> None:
        # dtype check (exact match)
        if np.dtype(arr.dtype) != np.dtype(self.dtype):
            raise TypeError(f"{self.name}: expected dtype {np.dtype(self.dtype)}, got {arr.dtype}")
        # shape check
        exp_shape = self.expected_shape_fn() if self.expected_shape_fn is not None else self.expected_shape
        if exp_shape is None:
            return
        if len(arr.shape) != len(exp_shape):
            raise ValueError(f"{self.name}: expected rank {len(exp_shape)}, got {arr.shape}")
        for exp, got in zip(exp_shape, arr.shape):
            if exp != -1 and exp != got:
                raise ValueError(f"{self.name}: expected shape {exp_shape}, got {arr.shape}")


