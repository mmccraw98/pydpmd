from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING, Any
import os
import h5py
import numpy as np
# Avoid importing data classes at module import time to prevent circular imports
if TYPE_CHECKING:
    from .data import BaseParticle


@dataclass
class MetaPaths:
    static: str = "static"
    init: str = "init"
    final: str = "final"
    restart: str = "restart"


@dataclass
class TrajPaths:
    timestep: str = "timestep"


def open_meta(path: str, mode: str = "r") -> h5py.File:
    return h5py.File(path, mode)


def open_traj(path: str, mode: str = "r") -> h5py.File:
    return h5py.File(path, mode)


def read_ds(g: h5py.Group, name: str) -> Optional[np.ndarray]:
    if name in g:
        return g[name][...]
    return None


def write_ds(g: h5py.Group, name: str, arr: np.ndarray) -> None:
    if name in g:
        del g[name]
    g.create_dataset(name, data=arr)


def ensure_unlimited(g: h5py.Group, name: str, shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype) -> h5py.Dataset:
    if name in g:
        return g[name]
    maxshape = (None,) + shape[1:]
    return g.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype)


def append_hyperslab(d: h5py.Dataset, data: np.ndarray) -> None:
    T = d.shape[0]
    newT = T + 1
    d.resize((newT,) + d.shape[1:])
    d[T, ...] = data
