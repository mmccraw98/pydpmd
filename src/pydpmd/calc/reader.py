from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from collections import OrderedDict
import h5py
import numpy as np


@dataclass
class ReaderSpec:
    mode: str  # 'in_memory' or 'h5'
    fields: tuple[str, ...]
    arrays: Optional[Dict[str, np.ndarray]] = None  # for in_memory; arrays[name] shape = (T, ...)
    filename: Optional[str] = None                 # for 'h5'

    def num_frames(self) -> int:
        if self.mode == 'in_memory':
            if not self.arrays:
                return 0
            # assume all share first-dim T
            first = next(iter(self.arrays.values()))
            return int(first.shape[0])
        else:
            assert self.filename is not None
            with h5py.File(self.filename, 'r') as f:
                name0 = self.fields[0]
                return int(f[name0].shape[0])


class FrameReader:
    """Worker-local reader with a tiny LRU cache of frames.
    `get(i)` returns {field: np.ndarray for that time index i}.
    """
    def __init__(self, spec: ReaderSpec, cache_size: int = 4):
        self.spec = spec
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        self._file: Optional[h5py.File] = None
        self._dsets: Optional[Dict[str, h5py.Dataset]] = None
        if spec.mode == 'h5':
            # open lazily on first access in worker
            pass

    def _ensure_open(self):
        if self.spec.mode == 'h5' and self._file is None:
            self._file = h5py.File(self.spec.filename, 'r')  # read-only
            self._dsets = {name: self._file[name] for name in self.spec.fields}

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._dsets = None
                self._cache.clear()

    def get(self, i: int) -> Dict[str, np.ndarray]:
        if self.cache_size > 0 and i in self._cache:
            # move to end (most recent)
            val = self._cache.pop(i)
            self._cache[i] = val
            return val
        if self.spec.mode == 'in_memory':
            assert self.spec.arrays is not None
            out = {name: self.spec.arrays[name][i] for name in self.spec.fields}
        else:
            self._ensure_open()
            out = {name: np.asarray(self._dsets[name][i][...]) for name in self.spec.fields}  # copy into RAM
        if self.cache_size > 0:
            self._cache[i] = out
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)  # evict least-recent
        return out


