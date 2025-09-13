from __future__ import annotations
"""Bin specifications for time-series and lagged analyses.

Examples
--------
Create time bins (no lag) for an entire trajectory:
    >>> from pydpmd.calc.bins import TimeBins
    >>> binspec = TimeBins.from_source(traj)  # bins are time indices [0..T-1]

Time bins for a subrange:
    >>> binspec = TimeBins.from_source(traj, t_min=100, t_max=500)

Linear lag bins (dt = 1..100):
    >>> from pydpmd.calc.bins import LagBinsLinear
    >>> binspec = LagBinsLinear.from_source(traj, dt_min=1, dt_max=100)

Linear lag bins with step and cap (sample at most 10k pairs per dt):
    >>> binspec = LagBinsLinear.from_source(traj, dt_min=1, dt_max=5000, step=5, cap=10_000)

Logarithmic lag bins with 8 per decade:
    >>> from pydpmd.calc.bins import LagBinsLog
    >>> binspec = LagBinsLog.from_source(traj, dt_min=1, dt_max=10_000, num_per_decade=8)

Pseudo-log lag bins (digits 1..9):
    >>> from pydpmd.calc.bins import LagBinsPseudoLog
    >>> binspec = LagBinsPseudoLog.from_source(traj, dt_min=1, dt_max=10_000)
"""

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Any
import h5py
import numpy as np


class BinSpec:
    def __init__(self, T: int):
        self.T = int(T)

    def num_bins(self) -> int:
        raise NotImplementedError

    def bins(self) -> Iterable[int]:
        return range(self.num_bins())

    def value_of_bin(self, b: int) -> Union[int, float, Tuple[Any, ...]]:
        return b

    def values(self) -> np.ndarray:
        B = self.num_bins()
        return np.asarray([self.value_of_bin(b) for b in range(B)])

    def weight(self, b: int) -> int:
        raise NotImplementedError

    def iter_tuples(self, b: int) -> Iterator[List[int]]:
        raise NotImplementedError


def _deterministic_subset(n: int,
                          cap: Optional[int],
                          method: str = 'stride',
                          seed: int = 0,
                          tag: int = 0) -> np.ndarray:
    if cap is None or cap >= n:
        return np.arange(n, dtype=np.int64)
    if cap <= 0:
        return np.empty(0, dtype=np.int64)
    if method == 'stride':
        if cap == 1:
            return np.array([0], dtype=np.int64)
        m = int(cap - 1)
        n1 = int(n - 1)
        ks = np.arange(cap, dtype=np.float64)
        idx = np.floor((ks * n1) / m).astype(np.int64)
        if idx.size > 1:
            idx = np.unique(idx)
        return idx
    if method == 'rng':
        mix = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) ^
               np.uint64(tag)  * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        rng = np.random.default_rng(int(mix))
        return np.sort(rng.choice(n, size=cap, replace=False).astype(np.int64))
    raise ValueError(f"Unknown sampling method: {method!r}")


def _infer_num_frames_from_source(source: Any) -> int:
    if isinstance(source, int):
        return int(source)
    num_frames_fn = getattr(source, 'num_frames', None)
    if callable(num_frames_fn):
        return int(num_frames_fn())
    if isinstance(source, dict):
        if not source:
            raise ValueError("Empty dict provided; cannot infer T")
        first = next(iter(source.values()))
        arr = np.asarray(first)
        if arr.ndim == 0:
            raise ValueError("Arrays must have at least one dimension to infer T")
        return int(arr.shape[0])
    if isinstance(source, str):
        with h5py.File(source, 'r') as f:
            for name in f.keys():
                obj = f.get(name, None)
                if isinstance(obj, h5py.Dataset):
                    return int(obj.shape[0])
        raise ValueError("No datasets found in HDF5 file to infer T")
    arrays = getattr(source, '_arrays', None)
    dsets = getattr(source, '_ds', None)
    if isinstance(arrays, dict) and arrays:
        first = next(iter(arrays.values()))
        return int(np.asarray(first).shape[0])
    if isinstance(dsets, dict) and dsets:
        first = next(iter(dsets.values()))
        return int(first.shape[0])
    raise TypeError("Unsupported source type for inferring number of frames")


class TimeBins(BinSpec):
    def __init__(self, T: int, t_min: Optional[int] = None, t_max: Optional[int] = None):
        super().__init__(T)
        lo = 0 if t_min is None else int(t_min)
        hi = (T - 1) if t_max is None else int(t_max)
        if not (0 <= lo <= hi <= T - 1):
            raise ValueError("Invalid time range for TimeBins")
        self.t_min = lo
        self.t_max = hi
        self._B = self.t_max - self.t_min + 1

    @classmethod
    def from_source(cls, source: Any, t_min: Optional[int] = None, t_max: Optional[int] = None) -> 'TimeBins':
        T = _infer_num_frames_from_source(source)
        return cls(T, t_min=t_min, t_max=t_max)

    def num_bins(self) -> int:
        return self._B

    def value_of_bin(self, b: int) -> int:
        return int(self.t_min + b)

    def weight(self, b: int) -> int:
        return 1

    def iter_tuples(self, b: int):
        yield [self.t_min + b]


class LagBinsExact(BinSpec):
    def __init__(self,
                 T: int,
                 dts: Sequence[int],
                 *,
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0):
        super().__init__(T)
        dts_arr = np.asarray(dts, dtype=np.int64)
        if dts_arr.size == 0:
            raise ValueError("dts must be non-empty")
        if np.any(dts_arr < 1) or np.any(dts_arr > T - 1):
            raise ValueError("dts must be within [1, T-1]")
        dts_arr = np.unique(dts_arr)
        self._dts = dts_arr
        self.cap = None if cap is None else int(cap)
        self.sample = sample
        self.seed = int(seed)
        self._pairs_per_bin = (T - self._dts).astype(np.int64)

    @classmethod
    def from_source(cls,
                    source: Any,
                    dts: Sequence[int],
                    *,
                    cap: Optional[int] = None,
                    sample: str = 'stride',
                    seed: int = 0) -> 'LagBinsExact':
        T = _infer_num_frames_from_source(source)
        return cls(T, dts, cap=cap, sample=sample, seed=seed)

    def num_bins(self) -> int:
        return int(self._dts.size)

    def value_of_bin(self, b: int) -> int:
        return int(self._dts[b])

    def weight(self, b: int) -> int:
        pairs = int(self._pairs_per_bin[b])
        if pairs <= 0:
            return 0
        return pairs if self.cap is None else min(pairs, self.cap)

    def iter_tuples(self, b: int):
        dt = int(self._dts[b])
        n_pairs = int(self._pairs_per_bin[b])
        if n_pairs <= 0:
            return
        sel = _deterministic_subset(n_pairs, self.cap, self.sample, self.seed, tag=dt)
        for i in sel:
            yield [int(i), int(i + dt)]


class LagBinsLinear(LagBinsExact):
    def __init__(self,
                 T: int,
                 dt_min: Optional[int] = None,
                 dt_max: Optional[int] = None,
                 *,
                 num_points: Optional[int] = None,
                 step: int = 1,
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0):
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = T - 1
        if not (1 <= dt_min <= dt_max <= T - 1):
            raise ValueError("Invalid dt range for LagBinsLinear")
        if step <= 0:
            raise ValueError("step must be positive")
        dts = np.arange(int(dt_min), int(dt_max) + 1, int(step), dtype=np.int64)
        if num_points is not None:
            m = int(num_points)
            if m <= 0:
                raise ValueError("num_points must be positive if provided")
            n = int(dts.size)
            if m < n:
                if m == 1:
                    dts = dts[[0]]
                else:
                    ks = np.arange(m, dtype=np.float64)
                    idx = np.floor(ks * (n - 1) / (m - 1)).astype(np.int64)
                    # idx is non-decreasing and within [0, n-1]
                    dts = dts[idx]
        super().__init__(T, dts, cap=cap, sample=sample, seed=seed)

    @classmethod
    def from_source(cls,
                    source: Any,
                    dt_min: Optional[int] = None,
                    dt_max: Optional[int] = None,
                    *,
                    num_points: Optional[int] = None,
                    step: int = 1,
                    cap: Optional[int] = None,
                    sample: str = 'stride',
                    seed: int = 0) -> 'LagBinsLinear':
        T = _infer_num_frames_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, num_points=num_points, step=step, cap=cap, sample=sample, seed=seed)


class LagBinsLog(LagBinsExact):
    def __init__(self,
                 T: int,
                 dt_min: Optional[int] = None,
                 dt_max: Optional[int] = None,
                 *,
                 num_bins: Optional[int] = None,
                 num_per_decade: Optional[int] = None,
                 round_mode: str = 'nearest',
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0):
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = T - 1
        if dt_min < 1:
            dt_min = 1
        if dt_max > T - 1:
            dt_max = T - 1
        if num_bins is None and num_per_decade is None:
            span = max(1, int(np.ceil(10 * np.log10(max(1, dt_max) / max(1, dt_min)))))
            num_bins = span
        if num_bins is not None:
            xs = np.logspace(np.log10(dt_min), np.log10(dt_max), int(num_bins))
        else:
            lo_dec = int(np.floor(np.log10(dt_min)))
            hi_dec = int(np.floor(np.log10(dt_max)))
            xs_list = []
            for k in range(lo_dec, hi_dec + 1):
                left = max(dt_min, int(10 ** k))
                right = min(dt_max, int(10 ** (k + 1)))
                if right < left:
                    continue
                xs_list.append(np.logspace(np.log10(left), np.log10(right), int(num_per_decade), endpoint=False))
            xs = np.concatenate(xs_list) if xs_list else np.array([], dtype=float)
        if round_mode == 'nearest':
            dts = np.rint(xs).astype(np.int64)
        elif round_mode == 'floor':
            dts = np.floor(xs).astype(np.int64)
        elif round_mode == 'ceil':
            dts = np.ceil(xs).astype(np.int64)
        else:
            raise ValueError("round_mode must be one of 'nearest'|'floor'|'ceil'")
        dts = dts[(dts >= 1) & (dts <= dt_max)]
        dts = np.unique(dts)
        if dts.size == 0:
            raise ValueError("No valid integer lags produced; adjust parameters")
        super().__init__(T, dts, cap=cap, sample=sample, seed=seed)

    @classmethod
    def from_source(cls,
                    source: Any,
                    dt_min: Optional[int] = None,
                    dt_max: Optional[int] = None,
                    *,
                    num_bins: Optional[int] = None,
                    num_per_decade: Optional[int] = None,
                    round_mode: str = 'nearest',
                    cap: Optional[int] = None,
                    sample: str = 'stride',
                    seed: int = 0) -> 'LagBinsLog':
        T = _infer_num_frames_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, num_bins=num_bins, num_per_decade=num_per_decade,
                   round_mode=round_mode, cap=cap, sample=sample, seed=seed)


class LagBinsPseudoLog(LagBinsExact):
    def __init__(self,
                 T: int,
                 dt_min: Optional[int] = None,
                 dt_max: Optional[int] = None,
                 *,
                 digits: Sequence[int] = tuple(range(1, 10)),
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0):
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = T - 1
        if dt_min < 1:
            dt_min = 1
        if dt_max > T - 1:
            dt_max = T - 1
        digits = tuple(sorted(set(int(d) for d in digits if d > 0)))
        if not digits:
            raise ValueError("digits must contain at least one positive integer")
        lo_dec = int(np.floor(np.log10(dt_min)))
        hi_dec = int(np.floor(np.log10(dt_max)))
        vals = []
        for k in range(lo_dec, hi_dec + 1):
            base = 10 ** k
            for d in digits:
                dt = d * base
                if dt_min <= dt <= dt_max:
                    vals.append(dt)
        dts = np.array(sorted(set(vals)), dtype=np.int64)
        if dts.size == 0:
            raise ValueError("No pseudo-log lags produced; adjust bounds/digits")
        super().__init__(T, dts, cap=cap, sample=sample, seed=seed)

    @classmethod
    def from_source(cls,
                    source: Any,
                    dt_min: Optional[int] = None,
                    dt_max: Optional[int] = None,
                    *,
                    digits: Sequence[int] = tuple(range(1, 10)),
                    cap: Optional[int] = None,
                    sample: str = 'stride',
                    seed: int = 0) -> 'LagBinsPseudoLog':
        T = _infer_num_frames_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, digits=digits, cap=cap, sample=sample, seed=seed)


