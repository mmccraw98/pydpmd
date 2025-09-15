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
    def __init__(self, T: int, timestep: Optional[np.ndarray] = None):
        self.T = int(T)
        if timestep is None:
            self.timestep = np.arange(self.T, dtype=np.int64)
        else:
            ts = np.asarray(timestep).squeeze()
            if ts.ndim != 1:
                raise ValueError("timestep must be a 1D array")
            if ts.size != self.T:
                raise ValueError("timestep length must equal T")
            self.timestep = ts.astype(np.int64, copy=False)

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


def _infer_timestep_and_T_from_source(source: Any) -> Tuple[np.ndarray, int]:
    """Infer (timestep, T) from various source types.

    Falls back to arange(T) if an explicit timestep dataset is not found.
    """
    # Integer provided -> fabricate 0..T-1
    if isinstance(source, int):
        T = int(source)
        return np.arange(T, dtype=np.int64), T
    # Trajectory-like object with internal stores
    arrays = getattr(source, '_arrays', None)
    dsets = getattr(source, '_ds', None)
    if isinstance(arrays, dict) and arrays:
        if 'timestep' in arrays:
            ts = np.asarray(arrays['timestep']).astype(np.int64, copy=False)
            return ts, int(ts.shape[0])
        first = next(iter(arrays.values()))
        T = int(np.asarray(first).shape[0])
        return np.arange(T, dtype=np.int64), T
    if isinstance(dsets, dict) and dsets:
        if 'timestep' in dsets:
            ts = np.array(dsets['timestep'][...], copy=False).astype(np.int64, copy=False)
            return ts, int(ts.shape[0])
        first = next(iter(dsets.values()))
        T = int(first.shape[0])
        return np.arange(T, dtype=np.int64), T
    # Dict of arrays
    if isinstance(source, dict):
        if 'timestep' in source:
            ts = np.asarray(source['timestep']).astype(np.int64, copy=False)
            return ts, int(ts.shape[0])
        if source:
            first = next(iter(source.values()))
            T = int(np.asarray(first).shape[0])
            return np.arange(T, dtype=np.int64), T
        raise ValueError("Empty dict provided; cannot infer T")
    # HDF5 filename
    if isinstance(source, str):
        with h5py.File(source, 'r') as f:
            if 'timestep' in f:
                ts = np.array(f['timestep'][...], copy=False).astype(np.int64, copy=False)
                return ts, int(ts.shape[0])
            for name in f.keys():
                obj = f.get(name, None)
                if isinstance(obj, h5py.Dataset):
                    T = int(obj.shape[0])
                    return np.arange(T, dtype=np.int64), T
        raise ValueError("No datasets found in HDF5 file to infer T")
    # Trajectory-like object with num_frames() method but no internal stores
    num_frames_fn = getattr(source, 'num_frames', None)
    if callable(num_frames_fn):
        T = int(num_frames_fn())
        return np.arange(T, dtype=np.int64), T
    raise TypeError("Unsupported source type for inferring timestep and T")


class TimeBins(BinSpec):
    def __init__(self, T: int, t_min: Optional[int] = None, t_max: Optional[int] = None, *, timestep: Optional[np.ndarray] = None):
        super().__init__(T, timestep=timestep)
        # Map physical time bounds (ints) to index range; if None, use full range
        ts = self.timestep
        # Build time->index map (unique monotonically increasing ints)
        self._time_to_index = {int(t): i for i, t in enumerate(ts)}
        if t_min is None:
            lo_idx = 0
        else:
            key = int(t_min)
            if key not in self._time_to_index:
                raise ValueError("t_min not found in timestep array")
            lo_idx = int(self._time_to_index[key])
        if t_max is None:
            hi_idx = self.T - 1
        else:
            key = int(t_max)
            if key not in self._time_to_index:
                raise ValueError("t_max not found in timestep array")
            hi_idx = int(self._time_to_index[key])
        if not (0 <= lo_idx <= hi_idx <= self.T - 1):
            raise ValueError("Invalid time range for TimeBins")
        self._indices = np.arange(lo_idx, hi_idx + 1, dtype=np.int64)

    @classmethod
    def from_source(cls, source: Any, t_min: Optional[int] = None, t_max: Optional[int] = None) -> 'TimeBins':
        timestep, T = _infer_timestep_and_T_from_source(source)
        return cls(T, t_min=t_min, t_max=t_max, timestep=timestep)

    def num_bins(self) -> int:
        return int(self._indices.size)

    def value_of_bin(self, b: int) -> int:
        idx = int(self._indices[int(b)])
        return int(self.timestep[idx])

    def weight(self, b: int) -> int:
        return 1

    def iter_tuples(self, b: int):
        idx = int(self._indices[int(b)])
        yield [idx]


class LagBinsExact(BinSpec):
    def __init__(self,
                 T: int,
                 taus: Sequence[int],
                 *,
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0,
                 timestep: Optional[np.ndarray] = None):
        super().__init__(T, timestep=timestep)
        taus_arr = np.asarray(taus, dtype=np.int64)
        if taus_arr.size == 0:
            raise ValueError("taus must be non-empty")
        max_tau = int(self.timestep[-1] - self.timestep[0]) if self.T > 0 else 0
        if max_tau < 1:
            raise ValueError("Insufficient time range for lag bins")
        taus_arr = taus_arr[(taus_arr >= 1) & (taus_arr <= max_tau)]
        taus_arr = np.unique(taus_arr)
        if taus_arr.size == 0:
            raise ValueError("No valid integer time-lags produced; adjust parameters")
        # Precompute pair counts and drop empty taus to ensure bins are never empty
        counts: List[int] = []
        kept: List[int] = []
        for tau in taus_arr:
            c = _count_pairs_for_tau(self.timestep, int(tau))
            if c > 0:
                kept.append(int(tau))
                counts.append(int(c))
        if not kept:
            raise ValueError("All tau bins are empty; adjust parameters or data")
        self._taus = np.asarray(kept, dtype=np.int64)
        self._pairs_per_bin = np.asarray(counts, dtype=np.int64)
        self.cap = None if cap is None else int(cap)
        self.sample = sample
        self.seed = int(seed)

    @classmethod
    def from_source(cls,
                    source: Any,
                    dts: Sequence[int],
                    *,
                    cap: Optional[int] = None,
                    sample: str = 'stride',
                    seed: int = 0) -> 'LagBinsExact':
        timestep, T = _infer_timestep_and_T_from_source(source)
        # Interpret provided dts as physical taus
        return cls(T, dts, cap=cap, sample=sample, seed=seed, timestep=timestep)

    def num_bins(self) -> int:
        return int(self._taus.size)

    def value_of_bin(self, b: int) -> int:
        return int(self._taus[b])

    def weight(self, b: int) -> int:
        pairs = int(self._pairs_per_bin[b])
        if pairs <= 0:
            return 0
        return pairs if self.cap is None else min(pairs, self.cap)

    def iter_tuples(self, b: int):
        tau = int(self._taus[b])
        n_pairs = int(self._pairs_per_bin[b])
        if n_pairs <= 0:
            return
        sel = _deterministic_subset(n_pairs, self.cap, self.sample, self.seed, tag=tau)
        # Enumerate all matching pairs in deterministic order; yield only selected indices
        ts = self.timestep
        i = 0
        j = 0
        k = 0  # index within all matching pairs
        p = 0  # pointer into sel
        sel_size = int(sel.size)
        while i < self.T and j < self.T and p < sel_size:
            # advance j until difference >= tau
            while j < self.T and (ts[j] - ts[i]) < tau:
                j += 1
            if j >= self.T:
                break
            diff = int(ts[j] - ts[i])
            if diff == tau:
                if k == int(sel[p]):
                    yield [int(i), int(j)]
                    p += 1
                k += 1
                i += 1
                if j < i:
                    j = i
            elif diff > tau:
                i += 1
                if j < i:
                    j = i


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
                 seed: int = 0,
                 timestep: Optional[np.ndarray] = None):
        ts = np.arange(T, dtype=np.int64) if timestep is None else np.asarray(timestep, dtype=np.int64)
        max_tau = int(ts[-1] - ts[0]) if T > 0 else 0
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = max_tau
        if not (1 <= dt_min <= dt_max <= max_tau):
            raise ValueError("Invalid tau range for LagBinsLinear")
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
        super().__init__(T, dts, cap=cap, sample=sample, seed=seed, timestep=ts)

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
        timestep, T = _infer_timestep_and_T_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, num_points=num_points, step=step, cap=cap, sample=sample, seed=seed, timestep=timestep)


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
                 seed: int = 0,
                 timestep: Optional[np.ndarray] = None):
        ts = np.arange(T, dtype=np.int64) if timestep is None else np.asarray(timestep, dtype=np.int64)
        max_tau = int(ts[-1] - ts[0]) if T > 0 else 0
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = max_tau
        if dt_min < 1:
            dt_min = 1
        if dt_max > max_tau:
            dt_max = max_tau
        # Determine the base grid unit u = gcd(diff(timestep)) to derive realizable lags
        diffs = np.diff(ts).astype(np.int64)
        u = int(np.gcd.reduce(diffs)) if diffs.size > 0 else 1
        if u <= 0:
            u = 1
        m_min = int(np.ceil(dt_min / u))
        m_max = int(np.floor(dt_max / u))
        if m_max < m_min:
            raise ValueError("No realizable integer lags in the requested range")
        # Generate candidate m values on the integer grid, log-distributed
        if num_bins is None and num_per_decade is None:
            span = max(1, int(np.ceil(10 * np.log10(max(1, m_max) / max(1, m_min)))))
            num_bins = span
        if num_bins is not None:
            xs = np.logspace(np.log10(m_min), np.log10(m_max), int(num_bins))
            m_vals = np.rint(xs).astype(np.int64)
        else:
            lo_dec = int(np.floor(np.log10(m_min)))
            hi_dec = int(np.floor(np.log10(m_max)))
            m_list = []
            for k in range(lo_dec, hi_dec + 1):
                left = max(m_min, int(10 ** k))
                right = min(m_max, int(10 ** (k + 1)))
                if right < left:
                    continue
                xs = np.logspace(np.log10(left), np.log10(right), int(num_per_decade), endpoint=False)
                m_list.append(np.rint(xs).astype(np.int64))
            m_vals = np.concatenate(m_list) if m_list else np.array([], dtype=np.int64)
        m_vals = m_vals[(m_vals >= m_min) & (m_vals <= m_max)]
        m_vals = np.unique(m_vals)
        if m_vals.size == 0:
            raise ValueError("No valid log-spaced m indices produced; adjust parameters")
        taus = (m_vals * u).astype(np.int64)
        super().__init__(T, taus, cap=cap, sample=sample, seed=seed, timestep=ts)

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
        timestep, T = _infer_timestep_and_T_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, num_bins=num_bins, num_per_decade=num_per_decade,
                   round_mode=round_mode, cap=cap, sample=sample, seed=seed, timestep=timestep)


class LagBinsPseudoLog(LagBinsExact):
    def __init__(self,
                 T: int,
                 dt_min: Optional[int] = None,
                 dt_max: Optional[int] = None,
                 *,
                 digits: Sequence[int] = tuple(range(1, 10)),
                 cap: Optional[int] = None,
                 sample: str = 'stride',
                 seed: int = 0,
                 timestep: Optional[np.ndarray] = None):
        ts = np.arange(T, dtype=np.int64) if timestep is None else np.asarray(timestep, dtype=np.int64)
        max_tau = int(ts[-1] - ts[0]) if T > 0 else 0
        if dt_min is None:
            dt_min = 1
        if dt_max is None:
            dt_max = max_tau
        if dt_min < 1:
            dt_min = 1
        if dt_max > max_tau:
            dt_max = max_tau
        digits = tuple(sorted(set(int(d) for d in digits if d > 0)))
        if not digits:
            raise ValueError("digits must contain at least one positive integer")
        # Determine base unit u for realizable lags and work on integer grid m
        diffs = np.diff(ts).astype(np.int64)
        u = int(np.gcd.reduce(diffs)) if diffs.size > 0 else 1
        if u <= 0:
            u = 1
        m_min = int(np.ceil(dt_min / u))
        m_max = int(np.floor(dt_max / u))
        if m_max < m_min:
            raise ValueError("No realizable integer lags in the requested range")
        # Pseudo-log digits applied on m grid
        lo_dec = int(np.floor(np.log10(m_min)))
        hi_dec = int(np.floor(np.log10(m_max)))
        m_vals: List[int] = []
        for k in range(lo_dec, hi_dec + 1):
            base = 10 ** k
            for d in digits:
                m = d * base
                if m_min <= m <= m_max:
                    m_vals.append(m)
        m_arr = np.array(sorted(set(m_vals)), dtype=np.int64)
        if m_arr.size == 0:
            raise ValueError("No pseudo-log m values produced; adjust bounds/digits")
        taus = (m_arr * u).astype(np.int64)
        super().__init__(T, taus, cap=cap, sample=sample, seed=seed, timestep=ts)

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
        timestep, T = _infer_timestep_and_T_from_source(source)
        return cls(T, dt_min=dt_min, dt_max=dt_max, digits=digits, cap=cap, sample=sample, seed=seed, timestep=timestep)


# ---- Helpers for time-based lag enumeration ----
def _count_pairs_for_tau(timestep: np.ndarray, tau: int) -> int:
    ts = timestep
    T = int(ts.shape[0])
    i = 0
    j = 0
    cnt = 0
    while i < T and j < T:
        while j < T and (int(ts[j]) - int(ts[i])) < tau:
            j += 1
        if j >= T:
            break
        diff = int(ts[j]) - int(ts[i])
        if diff == tau:
            cnt += 1
            i += 1
            if j < i:
                j = i
        elif diff > tau:
            i += 1
            if j < i:
                j = i
    return cnt


