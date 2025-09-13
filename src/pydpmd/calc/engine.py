from __future__ import annotations
"""Parallel binned accumulation engine.

End-to-end example (MSD + angular MSD)
--------------------------------------
    >>> import h5py, numpy as np
    >>> from pydpmd.calc.engine import run_binned
    >>> from pydpmd.calc.bins import LagBinsLog
    >>> from pydpmd.calc.kernels import requires_fields

    >>> @requires_fields("pos", "angle")
    ... def fused_msd_kernel(idxs, get_frame, system_id, system_size):
    ...     t0, t1 = idxs
    ...     r0 = get_frame(t0)["pos"]; r1 = get_frame(t1)["pos"]
    ...     dr2 = np.sum((r1 - r0) ** 2, axis=-1)
    ...     th0 = get_frame(t0)["angle"]; th1 = get_frame(t1)["angle"]
    ...     dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))
    ...     dth2 = dth ** 2
    ...     S = int(system_size.shape[0])
    ...     sid = np.asarray(system_id, dtype=np.int64)
    ...     msd = np.bincount(sid, weights=dr2, minlength=S) / system_size
    ...     ang = np.bincount(sid, weights=dth2, minlength=S) / system_size
    ...     return np.column_stack((msd, ang))

    >>> binspec = LagBinsLog.from_source(traj, dt_min=1, dt_max=10_000, num_per_decade=8)
    >>> res = run_binned(fused_msd_kernel, traj, binspec, kernel_kwargs=dict(system_id=sid, system_size=ssz))
    >>> res.mean.shape
    (binspec.num_bins(), 2)
    >>> dt = binspec.values()  # x-axis labels for plotting
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle

from .kernels import KernelFn
from .reader import ReaderSpec, FrameReader
from .bins import BinSpec



@dataclass
class RunResult:
    sums: np.ndarray  # shape (B, *S)
    counts: np.ndarray  # shape (B,)
    mean: np.ndarray    # shape (B, *S)


def _worker_run_range(
    spec: ReaderSpec,
    binspec_blob: bytes,
    b_start: int,
    b_end: int,
    sample_shape: Tuple[int, ...],
    kernel_ser: Tuple[str, bytes],
    kernel_kwargs: Dict[str, Any],
    cache_size: int,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    binspec: BinSpec = pickle.loads(binspec_blob)
    _kernel_name, kernel_blob = kernel_ser
    kernel: KernelFn = pickle.loads(kernel_blob)
    reader = FrameReader(spec, cache_size=cache_size)
    get = reader.get
    Bslice = b_end - b_start
    sums = np.zeros((Bslice,) + sample_shape, dtype=np.float64)
    counts = np.zeros((Bslice,), dtype=np.uint64)
    for local_b, b in enumerate(range(b_start, b_end)):
        acc = np.zeros(sample_shape, dtype=np.float64)
        cnt = 0
        for idxs in binspec.iter_tuples(b):
            val = kernel(idxs, get, **kernel_kwargs)
            if not isinstance(val, np.ndarray):
                val = np.asarray(val)
            if val.shape != sample_shape:
                raise ValueError(f"Kernel returned shape {val.shape}, expected {sample_shape}")
            acc += val
            cnt += 1
        sums[local_b] = acc
        counts[local_b] = cnt
    reader.close()
    return b_start, b_end, sums, counts


def _sample_shape_from_kernel(
    spec: ReaderSpec,
    binspec: BinSpec,
    kernel: KernelFn,
    kernel_kwargs: Dict[str, Any],
) -> Tuple[int, ...]:
    reader = FrameReader(spec, cache_size=2)
    get = reader.get
    sample_val: Optional[np.ndarray] = None
    for b in binspec.bins():
        it = binspec.iter_tuples(b)
        try:
            idxs = next(iter(it))
        except StopIteration:
            continue
        v = kernel(idxs, get, **kernel_kwargs)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        sample_val = v
        break
    reader.close()
    if sample_val is None:
        raise ValueError("All bins are empty; cannot infer sample shape")
    return tuple(sample_val.shape)


def _infer_reader_spec(
    trajectory: Any,
    fields: Sequence[str],
    force_mode: Optional[str] = None,
) -> ReaderSpec:
    if isinstance(trajectory, dict):
        arrays = {k: np.asarray(v) for k, v in trajectory.items() if k in fields}
        return ReaderSpec(mode='in_memory', fields=tuple(fields), arrays=arrays)
    if isinstance(trajectory, str):
        filename = trajectory
        return ReaderSpec(mode='h5', fields=tuple(fields), filename=filename)
    arrays = getattr(trajectory, '_arrays', None)
    file_obj = getattr(trajectory, 'file', None)
    filename = None
    if hasattr(file_obj, 'filename') and file_obj.filename:
        filename = file_obj.filename
    if force_mode == 'h5':
        if filename is None:
            raise ValueError("force_mode='h5' requires trajectory.file.filename to be set")
        return ReaderSpec(mode='h5', fields=tuple(fields), filename=filename)
    if isinstance(arrays, dict) and len(arrays) > 0:
        arrays_sel = {k: arrays[k] for k in fields}
        return ReaderSpec(mode='in_memory', fields=tuple(fields), arrays=arrays_sel)
    if filename is None:
        raise ValueError("Cannot infer data source: provide filename or ensure trajectory.file.filename is set")
    return ReaderSpec(mode='h5', fields=tuple(fields), filename=filename)


def partition_bins_by_weight(weights: Sequence[int], n_parts: int) -> List[Tuple[int, int]]:
    B = len(weights)
    if B == 0:
        return []
    n_parts = max(1, min(n_parts, B))
    total = sum(weights)
    target = total / n_parts if total > 0 else 0
    parts: List[Tuple[int, int]] = []
    start = 0
    accum = 0
    for b in range(B):
        w = weights[b]
        if accum >= target and len(parts) < n_parts - 1:
            parts.append((start, b))
            start = b
            accum = 0
        accum += w
    parts.append((start, B))
    merged: List[Tuple[int, int]] = []
    for s, e in parts:
        if s < e:
            merged.append((s, e))
    if not merged:
        merged = [(0, B)]
    return merged


def run_binned(
    kernel: KernelFn,
    trajectory: Any,
    binspec: BinSpec,
    *,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
    prefer_threads_for_in_memory: bool = True,
    show_progress: bool = False,
    progress_units: str = 'weights',
    cache_size: int = 4,
) -> RunResult:
    kernel_kwargs = kernel_kwargs or {}
    fields = getattr(kernel, 'required_fields', None)
    if not fields:
        raise ValueError("Kernel must declare required fields via @requires_fields('name', ...) or provide non-empty 'required_fields' attribute")
    spec = _infer_reader_spec(trajectory, fields)

    B = binspec.num_bins()
    weights = [max(0, int(binspec.weight(b))) for b in range(B)]
    sample_shape = _sample_shape_from_kernel(spec, binspec, kernel, kernel_kwargs)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))
    use_threads = prefer_threads_for_in_memory and spec.mode == 'in_memory'
    parts = partition_bins_by_weight(weights, max(n_workers, min(B, n_workers * 8)))
    parts = [p for p in parts if p[0] < p[1]]
    if not parts:
        empty_sums = np.zeros((B,) + sample_shape, dtype=np.float64)
        empty_counts = np.zeros((B,), dtype=np.uint64)
        empty_mean = np.full_like(empty_sums, np.nan, dtype=np.float64)
        return RunResult(sums=empty_sums, counts=empty_counts, mean=empty_mean)

    sums_total = np.zeros((B,) + sample_shape, dtype=np.float64)
    counts_total = np.zeros((B,), dtype=np.uint64)
    kernel_blob = pickle.dumps(kernel, protocol=pickle.HIGHEST_PROTOCOL)
    kernel_ser = (getattr(kernel, '__name__', 'kernel'), kernel_blob)
    binspec_blob = pickle.dumps(binspec, protocol=pickle.HIGHEST_PROTOCOL)

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    futures = []
    units_map = {}
    total_units = (sum(weights) if (show_progress and progress_units == 'weights') else (len(parts) if show_progress else 0))
    pbar = tqdm(total=total_units, desc='Binned accumulation') if show_progress else None
    with executor_cls(max_workers=min(n_workers, len(parts))) as pool:
        for (b_start, b_end) in parts:
            fut = pool.submit(
                _worker_run_range,
                spec,
                binspec_blob,
                b_start,
                b_end,
                sample_shape,
                kernel_ser,
                kernel_kwargs,
                cache_size,
            )
            if show_progress:
                units = (sum(weights[b_start:b_end]) if progress_units == 'weights' else (b_end - b_start))
                units_map[fut] = units
            futures.append(fut)
        for fut in as_completed(futures):
            b_start, b_end, sums, counts = fut.result()
            sums_total[b_start:b_end] += sums
            counts_total[b_start:b_end] += counts
            if pbar is not None:
                pbar.update(units_map.get(fut, 0))
    if pbar is not None:
        pbar.close()

    Bn = sums_total.shape[0]
    expand = (1,) * (sums_total.ndim - 1)
    denom = counts_total.astype(np.float64).reshape((Bn,) + expand)
    denom_safe = denom.copy()
    denom_safe[denom_safe == 0] = 1.0
    mean = sums_total / denom_safe
    zero_bins = (counts_total == 0)
    if np.any(zero_bins):
        mean[zero_bins, ...] = np.nan

    return RunResult(sums=sums_total, counts=counts_total, mean=mean)


