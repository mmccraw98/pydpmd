from __future__ import annotations
"""Kernel typing, decorator, and example kernels.

Usage
-----
Declare required fields and implement a kernel:
    >>> from pydpmd.calc.kernels import requires_fields, KernelFn
    >>> @requires_fields("pos")
    ... def my_kernel(idxs, get_frame, factor=1.0):
    ...     f0 = get_frame(idxs[0])["pos"]; f1 = get_frame(idxs[-1])["pos"]
    ...     return (np.sum((f1 - f0) ** 2, axis=-1) * factor)

Run with the engine:
    >>> from pydpmd.calc.engine import run_binned
    >>> from pydpmd.calc.bins import LagBinsLinear
    >>> binspec = LagBinsLinear.from_source(traj, dt_min=1, dt_max=100)
    >>> res = run_binned(my_kernel, traj, binspec)
"""

from typing import Callable, List, Dict
import numpy as np

KernelFn = Callable[[List[int], Callable[[int], Dict[str, np.ndarray]]], np.ndarray]

def requires_fields(*names: str):
    def _decorate(fn: KernelFn) -> KernelFn:
        setattr(fn, "required_fields", tuple(names))
        return fn
    return _decorate

# ------ Additional Kernel Functions ------

@requires_fields("pos")
def msd_kernel(indices: List[int], get_frame: Callable[[int], Dict[str, np.ndarray]], system_id: np.ndarray, system_size: np.ndarray) -> np.ndarray:
    """Return the mean squared displacement between two times.  The average is taken within each system.
    indices: [t0, t1]
    get_frame(t) must return {field: (N, d) array}.
    Output shape: (S,) as ndarray.
    """
    if len(indices) != 2:
        raise ValueError("msd_kernel expects [t0, t1]")
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    dr = r1 - r0
    return np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size

@requires_fields("angle")
def angular_msd_kernel(indices: List[int], get_frame: Callable[[int], Dict[str, np.ndarray]], system_id: np.ndarray, system_size: np.ndarray) -> np.ndarray:
    """Return the mean squared displacement of the angle between two times.  The average is taken within each system.
    indices: [t0, t1]
    get_frame(t) must return {field: (N,) array}.
    Output shape: (S,) as ndarray.
    """
    if len(indices) != 2:
        raise ValueError("angular_msd_kernel expects [t0, t1]")
    t0, t1 = indices
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dtheta = theta1 - theta0
    return np.bincount(system_id, weights=dtheta ** 2) / system_size

@requires_fields("pos", "angle")
def fused_msd_kernel(indices: List[int], get_frame: Callable[[int], Dict[str, np.ndarray]], system_id: np.ndarray, system_size: np.ndarray) -> np.ndarray:
    """Return the mean squared displacement of the position and angle between two times.  The average is taken within each system.
    indices: [t0, t1]
    get_frame(t) must return {field: (N, d) array} and {field: (N,) array}.
    Output shape: (S, 2) as ndarray (msd, angular_msd)
    """
    if len(indices) != 2:
        raise ValueError("fused_msd_kernel expects [t0, t1]")
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dr = r1 - r0
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    dtheta = theta1 - theta0
    angular_msd = np.bincount(system_id, weights=dtheta ** 2) / system_size
    return np.column_stack([msd, angular_msd])