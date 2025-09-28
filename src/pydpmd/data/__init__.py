"""Data subpackage for pydpmd.

Exports particle classes, a centralized class registry, and the high-level
loader for meta/trajectory files.

Quick start
-----------

Create, save, and load a system:

>>> from pydpmd.data import RigidBumpy, load
>>> rb = RigidBumpy()
>>> rb.allocate_systems(1)
>>> rb.allocate_particles(10)
>>> rb.set_ids()
>>> rb.save("/tmp/sim", locations=["init"], save_trajectory=False)
>>> obj = load("/tmp/sim", location="init", load_trajectory=False)

Load with multiple state locations (first applied to top-level):

>>> obj = load("/tmp/sim", location=["restart", "init"], load_trajectory=False)

Attach trajectory (lazy partial I/O by default; set load_full=True to
fully materialize into memory):

>>> obj = load("/tmp/sim", location="init", load_trajectory=True, load_full=False)
>>> # Attribute-style field access
>>> pos_t0 = obj.trajectory[0].pos    # first frame positions (copy)
>>> vel_range = obj.trajectory[10:20].vel  # frames 10..19 velocities (copy)

Registering new particle classes
--------------------------------

Add your class to ``CLASS_MAP`` below so it can be constructed during load:

>>> from pydpmd.data import CLASS_MAP
>>> from pydpmd.data.my_particle import MyParticle
>>> CLASS_MAP["MyParticle"] = MyParticle

When saving, ``BaseParticle.save`` writes the class name as a dataset
``/static/class_name`` (string) for reliable round-trip loading. Older files
may have it under ``/static`` attributes; the loader supports both.
"""

from typing import Optional, List, Union
from .base_particle import BaseParticle
from .base_point_particle import BasePointParticle
from .base_poly_particle import BasePolyParticle
from .disk import Disk
from .rigid_bumpy import RigidBumpy
from . import bumpy_utils
import os
import h5py
import numpy as np
from ..h5io import MetaPaths
from ..trajectory import Trajectory
from ..fields import NeighborMethod

__all__ = [
    "BaseParticle",
    "BasePointParticle",
    "BasePolyParticle",
    "Disk",
    "RigidBumpy",
    "bumpy_utils",
    "CLASS_MAP",
    "load",
]


# Central registry of particle class names for loading
CLASS_MAP = {
    "RigidBumpy": RigidBumpy,
    "Disk": Disk,
}


def load(path: str, location: Optional[Union[str, List[str]]] = None, load_trajectory: bool = False, load_full: bool = True):
    """Load a particle object from a directory path.

    Expected layout:
      - path/meta.h5 (required)
      - path/trajectory.h5 (optional; used when load_trajectory=True)
    """
    meta_path = path if path.endswith(".h5") else f"{path}/meta.h5"
    f = h5py.File(meta_path, "r")
    # Determine class
    cls_name = None
    if MetaPaths.static in f:
        g = f[MetaPaths.static]
        # Prefer dataset under static, fall back to attribute for legacy files
        if "class_name" in g:
            val = g["class_name"][()]
            if isinstance(val, bytes):
                cls_name = val.decode("utf-8")
            elif isinstance(val, np.ndarray):
                # 0-d string dataset may come back as np.ndarray
                cls_name = val.astype(str).item()
            else:
                cls_name = str(val)
        elif "class_name" in g.attrs:
            val = g.attrs["class_name"]
            if isinstance(val, bytes):
                cls_name = val.decode("utf-8")
            else:
                cls_name = str(val)
    if not cls_name or cls_name not in CLASS_MAP:
        f.close()
        raise ValueError("class_name not found in file or unknown class")

    obj = CLASS_MAP[cls_name]()
    # Load static
    if MetaPaths.static in f:
        g = f[MetaPaths.static]
        obj._load_group_to(obj.static, g)
        obj._apply_group_to_top(obj.static)
        if "neighbor_method" in g:
            val = g["neighbor_method"][()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            elif isinstance(val, np.ndarray):
                val = val.tobytes().decode("utf-8") if val.dtype.kind == 'S' else str(val)
            obj.neighbor_method = NeighborMethod[val] if val in NeighborMethod.__members__ else None
        
    # Load state snapshot
    if location:
        if isinstance(location, str):
            location = [location]
        loaded_locs: List[str] = []
        # First, load each requested location group into the object's mirrors
        for loc in location:
            if loc in f:
                g = f[loc]
                if not hasattr(obj, loc):
                    raise ValueError(f"Location '{loc}' not found in {obj.__class__.__name__}")
                obj._load_group_to(getattr(obj, loc), g)
                loaded_locs.append(loc)
        # Then, apply to top-level with fallback semantics: lower priority first, higher last
        for loc in reversed(loaded_locs):
            obj._apply_group_to_top(getattr(obj, loc))
    f.close()
    # Optionally attach trajectory
    if load_trajectory:
        traj_path = path if path.endswith(".h5") else f"{path}/trajectory.h5"
        if os.path.exists(traj_path):
            tf = h5py.File(traj_path, "r")
            obj.trajectory = Trajectory(tf, load_full=load_full)
    obj.fill_in_missing_fields()
    obj.validate()
    return obj

