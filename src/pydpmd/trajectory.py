from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Any, List, Mapping, Iterator
import h5py
import numpy as np
from .fields import IndexSpace


@dataclass
class TrajectoryView:
    file: h5py.File
    datasets: Dict[str, h5py.Dataset]

    def __getitem__(self, slc: Any) -> Dict[str, np.ndarray]:
        return {k: d[slc] for k, d in self.datasets.items()}


class Trajectory:
    def __init__(self, file: h5py.File, field_names: Optional[Iterable[str]] = None, load_full: bool = False):
        self.file = file
        self._ds: Dict[str, h5py.Dataset] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        # If explicit field_names provided, respect them; otherwise discover all top-level datasets
        names: Iterable[str]
        if field_names is None:
            names = [name for name in self.file.keys() if isinstance(self.file.get(name, None), h5py.Dataset)]
        else:
            names = field_names
        for name in names:
            if name in self.file and isinstance(self.file.get(name, None), h5py.Dataset):
                if load_full:
                    self._arrays[name] = self.file[name][...]
                else:
                    self._ds[name] = self.file[name]

    def __repr__(self) -> str:
        fname = getattr(self.file, 'filename', None)
        file_str = str(fname) if fname is not None else "<memory>"
        fields = tuple(sorted(self.fields()))
        return f"Trajectory(T={self.num_frames()}, fields={fields}, file={file_str})"

    def fields(self) -> Tuple[str, ...]:
        if self._arrays:
            return tuple(self._arrays.keys())
        return tuple(self._ds.keys())

    def num_frames(self) -> int:
        store = self._arrays if self._arrays else self._ds
        if not store:
            return 0
        first = next(iter(store.values()))
        return first.shape[0]

    def load_all(self) -> Dict[str, np.ndarray]:
        if self._arrays:
            return {k: np.array(v, copy=True) for k, v in self._arrays.items()}
        return {k: np.array(d[...]) for k, d in self._ds.items()}

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        # Return a window view for consistency with __getitem__/slice
        return TrajectoryWindow(self, i)

    def __getitem__(self, slc: Any) -> "TrajectoryWindow":
        # Generic partial-load window with attribute- and dict-style access
        return TrajectoryWindow(self, slc)

    def slice(self, start: int, stop: Optional[int] = None, step: Optional[int] = None) -> "TrajectoryWindow":
        # Window over a time range with attribute- and dict-style access
        return TrajectoryWindow(self, slice(start, stop, step))

    # Internal helpers for composition wrappers
    def _get_frame_field(self, name: str, i: int) -> np.ndarray:
        if self._arrays:
            return np.array(self._arrays[name][i])
        return np.array(self._ds[name][i])

    def _get_full_field(self, name: str) -> np.ndarray:
        if self._arrays:
            return np.array(self._arrays[name], copy=True)
        return np.array(self._ds[name][...])

    # ---------- Attribute-style access ----------
    def __getattribute__(self, name: str) -> Any:
        # Fast path for internals and dunder
        if name.startswith("__") or name in ("_ds", "_arrays", "file", "fields", "num_frames", "load_all", "frame", "__getitem__", "slice", "_get_frame_field", "_get_full_field"):
            return object.__getattribute__(self, name)
        # Determine available dataset names without triggering attribute recursion
        arrays = object.__getattribute__(self, "_arrays")
        dsets = object.__getattribute__(self, "_ds")
        keys = arrays.keys() if arrays else dsets.keys()
        if name in keys:
            # Validate identifier and collision
            if not name.isidentifier():
                raise AttributeError(f"Trajectory field '{name}' is not a valid Python identifier; access via traj['{name}']")
            # Check collision against class attributes/methods
            if hasattr(type(self), name) or name in self.__dict__:
                raise AttributeError(f"Trajectory field '{name}' collides with an attribute; access via traj['{name}']")
            # Return a field accessor
            return FieldAccessor(self, name)
        return object.__getattribute__(self, name)

    def __dir__(self) -> List[str]:
        base = list(super().__dir__())
        arrays = object.__getattribute__(self, "_arrays")
        dsets = object.__getattribute__(self, "_ds")
        keys = arrays.keys() if arrays else dsets.keys()
        for k in keys:
            if k.isidentifier() and not hasattr(type(self), k):
                base.append(k)
        return sorted(set(base))


@dataclass
class FieldAccessor:
    parent: Trajectory
    name: str

    def __repr__(self) -> str:
        p = self.parent
        store = p._arrays if p._arrays else p._ds
        shape = store[self.name].shape
        mode = "full" if p._arrays else "lazy"
        return f"TrajectoryField(name='{self.name}', shape={shape}, mode={mode})"

    def __getitem__(self, slc: Any) -> np.ndarray:
        p = self.parent
        if p._arrays:
            return np.array(p._arrays[self.name][slc])
        return np.array(p._ds[self.name][slc])

    def frame(self, i: int) -> np.ndarray:
        return self.__getitem__(i)

    def load(self) -> np.ndarray:
        return self.parent._get_full_field(self.name)

    # Allow np.array(field) to materialize fully
    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.load()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr


class TrajectoryWindow(Mapping[str, np.ndarray]):
    """Read-only view over a Trajectory for a given time index/slice.

    - Dict-style: window['pos'] -> ndarray for that time selection
    - Attribute-style: window.pos -> ndarray (with validation and collision checks)
    """
    def __init__(self, base: Trajectory, time_sel: Any):
        self._base = base
        self._sel = time_sel

    def __repr__(self) -> str:
        sel = self._sel
        if isinstance(sel, slice):
            s = f"{sel.start}:{sel.stop}:{sel.step}"
        else:
            s = str(sel)
        return f"TrajectoryWindow(sel={s}, fields={self.fields()})"

    # Mapping interface
    def __getitem__(self, key: str) -> np.ndarray:
        arrays = self._base._arrays
        dsets = self._base._ds
        if arrays:
            if key not in arrays:
                raise KeyError(key)
            return np.array(arrays[key][self._sel])
        if key not in dsets:
            raise KeyError(key)
        return np.array(dsets[key][self._sel])

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields())

    def __len__(self) -> int:
        return len(self.fields())

    def fields(self) -> Tuple[str, ...]:
        arrays = self._base._arrays
        dsets = self._base._ds
        keys = arrays.keys() if arrays else dsets.keys()
        return tuple(keys)

    # Attribute-style access
    def __getattr__(self, name: str) -> Any:
        # Only dataset names are supported; validate like Trajectory
        keys = self.fields()
        if name in keys:
            if not name.isidentifier():
                raise AttributeError(f"Trajectory field '{name}' is not a valid Python identifier; access via window['{name}']")
            if hasattr(type(self), name) or name in self.__dict__:
                raise AttributeError(f"Trajectory field '{name}' collides with an attribute; access via window['{name}']")
            return self[name]
        raise AttributeError(name)

    def __dir__(self) -> List[str]:
        base = list(super().__dir__())
        for k in self.fields():
            if k.isidentifier() and not hasattr(type(self), k):
                base.append(k)
        return sorted(set(base))


class ConcatTrajectory:
    """Virtual trajectory that concatenates multiple trajectories along the entity dimension per frame."""
    def __init__(self, trajectories: List[Trajectory]):
        if not trajectories:
            raise ValueError("ConcatTrajectory: empty list")
        self.trajs = trajectories
        # Validate same fields and same num_frames
        fields = self.trajs[0].fields()
        T = self.trajs[0].num_frames()
        for t in self.trajs[1:]:
            if t.fields() != fields:
                raise ValueError("ConcatTrajectory: all trajectories must have identical fields")
            if t.num_frames() != T:
                raise ValueError("ConcatTrajectory: all trajectories must have identical number of frames")
        self._fields = fields
        self._T = T

    def fields(self) -> Tuple[str, ...]:
        return self._fields

    def num_frames(self) -> int:
        return self._T

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in self._fields:
            chunks = [t._ds[name][i] for t in self.trajs]
            # Concatenate along entity axis: after indexing time, entity axis is axis 0
            out[name] = np.concatenate(chunks, axis=0)
        return out

    def load_all(self) -> Dict[str, np.ndarray]:
        # Stack all frames from each trajectory; ensure consistent shape
        res = {name: [] for name in self._fields}
        for i in range(self._T):
            fr = self.frame(i)
            for name in self._fields:
                res[name].append(fr[name])
        for name in self._fields:
            res[name] = np.stack(res[name], axis=0)
        return res


class SliceTrajectory:
    """Virtual trajectory that slices an underlying trajectory along entity dimension per field using per-field ranges.

    ranges: mapping from IndexSpace to (start, stop). For unknown fields, no slicing is applied.
    index_spaces: mapping from dataset name to IndexSpace.
    """
    def __init__(self, base: Trajectory, index_spaces: Dict[str, IndexSpace],
                 ranges: Dict[IndexSpace, Tuple[int, int]]):
        self.base = base
        self.index_spaces = index_spaces
        self.ranges = ranges

    def fields(self) -> Tuple[str, ...]:
        return self.base.fields()

    def num_frames(self) -> int:
        return self.base.num_frames()

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in self.base.fields():
            arr = self.base._ds[name][i]
            space = self.index_spaces.get(name, None)
            if space in self.ranges:
                start, stop = self.ranges[space]
                if arr.ndim == 1:
                    arr = arr[start:stop]
                elif arr.ndim >= 2:
                    arr = arr[start:stop, ...]
            out[name] = arr
        return out

    def load_all(self) -> Dict[str, np.ndarray]:
        res = {name: [] for name in self.base.fields()}
        for i in range(self.num_frames()):
            fr = self.frame(i)
            for name in self.base.fields():
                res[name].append(fr[name])
        for name in self.base.fields():
            res[name] = np.stack(res[name], axis=0)
        return res


