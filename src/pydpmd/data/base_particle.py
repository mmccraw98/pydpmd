from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from typing import Set
import numpy as np
import h5py
import os

from ..fields import FieldSpec, IndexSpace as I, DT_FLOAT, DT_INT, NeighborMethod
from ..trajectory import Trajectory
# h5io utilities imported where needed to avoid circular imports
from ..calc_utils import assign_lattice_positions

@dataclass
class GroupData:
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def fields(self) -> Tuple[str, ...]:
        return tuple(self.arrays.keys())

    def __getitem__(self, _sel: Any) -> "GroupDataWindow":
        # Stateless window to mirror Trajectory interface; selection is ignored
        return GroupDataWindow(self, _sel)

    def __getattr__(self, name: str) -> Any:
        # Attribute-style direct access to stored arrays
        if name in self.arrays:
            if not name.isidentifier():
                raise AttributeError(
                    f"GroupData field '{name}' is not a valid Python identifier; access via arrays['{name}']"
                )
            return self.arrays[name]
        raise AttributeError(name)

    def __dir__(self) -> List[str]:
        base = list(super().__dir__())
        for k in self.arrays.keys():
            if k.isidentifier():
                base.append(k)
        return sorted(set(base))


@dataclass
class GroupDataWindow:
    base: GroupData
    sel: Any

    def __repr__(self) -> str:
        return f"GroupDataWindow(fields={self.fields()})"

    def fields(self) -> Tuple[str, ...]:
        return self.base.fields()

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.base.arrays:
            raise KeyError(key)
        return self.base.arrays[key]

    def __getattr__(self, name: str) -> Any:
        if name in self.base.arrays:
            if not name.isidentifier():
                raise AttributeError(
                    f"GroupData field '{name}' is not a valid Python identifier; access via window['{name}']"
                )
            return self.base.arrays[name]
        raise AttributeError(name)

    def __dir__(self) -> List[str]:
        base = list(super().__dir__())
        for k in self.base.arrays.keys():
            if k.isidentifier():
                base.append(k)
        return sorted(set(base))


class BaseParticle:
    def __init__(self):
        # Top-level fields mirroring C++ (names only; arrays filled on load)
        self.pos: Optional[np.ndarray] = None  # (N,2)
        self.vel: Optional[np.ndarray] = None  # (N,2)
        self.force: Optional[np.ndarray] = None  # (N,2)
        self.pe: Optional[np.ndarray] = None
        self.ke: Optional[np.ndarray] = None
        self.area: Optional[np.ndarray] = None
        self.perimeter: Optional[np.ndarray] = None
        self.shape_parameter: Optional[np.ndarray] = None

        self.system_id: Optional[np.ndarray] = None
        self.system_size: Optional[np.ndarray] = None
        self.system_offset: Optional[np.ndarray] = None
        self.box_size: Optional[np.ndarray] = None

        self.packing_fraction: Optional[np.ndarray] = None
        self.pressure: Optional[np.ndarray] = None
        self.temperature: Optional[np.ndarray] = None
        self.pe_total: Optional[np.ndarray] = None
        self.ke_total: Optional[np.ndarray] = None

        # HDF5 group mirrors
        self.static = GroupData()
        self.init = GroupData()
        self.final = GroupData()
        self.restart = GroupData()
        self.trajectory: Optional[Trajectory] = None

        # Dynamic FieldSpec via a function to compute expected shape (N, S) at runtime
        base_map = {
            "pos": FieldSpec("pos", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(), 2)),
            "vel": FieldSpec("vel", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(), 2)),
            "force": FieldSpec("force", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(), 2)),
            "pe": FieldSpec("pe", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "ke": FieldSpec("ke", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "area": FieldSpec("area", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "perimeter": FieldSpec("perimeter", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "shape_parameter": FieldSpec("shape_parameter", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "system_id": FieldSpec("system_id", I.Particle, DT_INT, expected_shape_fn=lambda: (self.n_particles(),)),
            "system_size": FieldSpec("system_size", I.System, DT_INT, expected_shape_fn=lambda: (self.n_systems(),)),
            "system_offset": FieldSpec("system_offset", I.System, DT_INT, expected_shape_fn=lambda: (self.n_systems()+1,)),
            "box_size": FieldSpec("box_size", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(), 2)),
            "packing_fraction": FieldSpec("packing_fraction", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
            "pressure": FieldSpec("pressure", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
            "temperature": FieldSpec("temperature", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
            "pe_total": FieldSpec("pe_total", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
            "ke_total": FieldSpec("ke_total", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
        }
        self._spec_fn = lambda m=base_map: m

        # neighbor method (python-side)
        self.neighbor_method: Optional[NeighborMethod] = None

        # Track dynamically added fields that should be saved under static
        self._extra_static_fields: Set[str] = set()

    # ---------- Sizes ----------
    def n_particles(self) -> int:
        return int(self.pos.shape[0]) if self.pos is not None else 0

    def n_systems(self) -> int:
        return int(self.box_size.shape[0]) if self.box_size is not None else 0

    def n_vertices(self) -> int:
        v = getattr(self, "vertex_pos", None)
        return int(v.shape[0]) if v is not None else 0

    def n_dof(self) -> np.ndarray:
        """Get the number of degrees of freedom for each system"""
        raise NotImplementedError("n_dof() needs to be implemented in the derived class")


    # ---------- Allocation ----------
    def allocate_systems(self, S: int) -> None:
        self.system_size = np.empty((S,), dtype=DT_INT)
        self.system_offset = np.empty((S + 1,), dtype=DT_INT)
        self.box_size = np.empty((S, 2), dtype=DT_FLOAT)
        self.packing_fraction = np.zeros((S,), dtype=DT_FLOAT)
        self.pressure = np.zeros((S,), dtype=DT_FLOAT)
        self.temperature = np.zeros((S,), dtype=DT_FLOAT)
        self.pe_total = np.zeros((S,), dtype=DT_FLOAT)
        self.ke_total = np.zeros((S,), dtype=DT_FLOAT)

    def allocate_particles(self, N: int) -> None:
        self.pos = np.empty((N, 2), dtype=DT_FLOAT)
        self.vel = np.zeros((N, 2), dtype=DT_FLOAT)
        self.force = np.zeros((N, 2), dtype=DT_FLOAT)
        self.pe = np.zeros((N,), dtype=DT_FLOAT)
        self.ke = np.zeros((N,), dtype=DT_FLOAT)
        self.area = np.zeros((N,), dtype=DT_FLOAT)
        self.system_id = np.empty((N,), dtype=DT_INT)

    def allocate_vertices(self, Nv: int) -> None:
        # Base: no-op (poly classes override to allocate vertex arrays)
        pass

    def set_ids(self) -> None:
        if self.n_systems() == 1:
            self.system_id.fill(0)
            self.system_size.fill(self.n_particles())
        self.system_offset = np.concatenate([[0], np.cumsum(self.system_size)]).astype(DT_INT)

    def set_neighbor_method(self, neighbor_method: NeighborMethod) -> None:
        self.neighbor_method = neighbor_method

    # ---------- Loading ----------
    def fields(self) -> List[str]:
        # Only report fields currently available on the instance (non-None)
        return sorted([k for k in self._spec_fn().keys() if getattr(self, k, None) is not None])

    def get_static_fields(self) -> List[str]:
        static_fields =  ['system_id', 'system_size', 'system_offset', 'box_size']
        if self.neighbor_method == NeighborMethod.Cell:
            static_fields += ['cell_size', 'cell_dim', 'cell_system_start', 'verlet_skin', 'thresh2']
        # Include any dynamically added arrays requested to be stored under static
        if getattr(self, "_extra_static_fields", None):
            static_fields += sorted(list(self._extra_static_fields))
        return static_fields
    
    def get_state_fields(self) -> List[str]:
        state_fields = ['pe', 'ke', 'packing_fraction', 'pressure', 'temperature', 'pe_total', 'ke_total']
        return state_fields

    # ---------- Saving ----------
    def save(self, path: str, write_static: bool = True, locations: Optional[List[str] | str] = None,
             save_trajectory: bool = False, overwrite_trajectory: bool = True) -> None:
        # Save to directory layout: path/meta.h5 and optionally path/trajectory.h5
        os.makedirs(path, exist_ok=True)
        meta_path = os.path.join(path, "meta.h5")
        meta_file = h5py.File(meta_path, "w")
        if write_static:
            g = meta_file.require_group("static")
            # store class name for round-trip loading as a dataset under /static
            if "class_name" in g:
                del g["class_name"]
            g.create_dataset(
                "class_name",
                data=self.__class__.__name__,
                dtype=h5py.string_dtype(encoding="utf-8")
            )
            for name in self.get_static_fields():
                if name in g:
                    del g[name]
                if getattr(self, name) is not None:
                    g.create_dataset(name, data=getattr(self, name))
            if self.neighbor_method is not None:
                if "neighbor_method" in g: del g["neighbor_method"]
                g.create_dataset(
                    "neighbor_method",
                    data=self.neighbor_method.name,
                    dtype=h5py.string_dtype(encoding="utf-8")
                )
            # Write scalar counts for convenience
            for key, val in (
                ("n_particles", np.asarray(self.n_particles(), dtype=DT_INT)),
                ("n_systems", np.asarray(self.n_systems(), dtype=DT_INT)),
                ("n_vertices", np.asarray(self.n_vertices(), dtype=DT_INT)),
            ):
                if key in g: del g[key]
                g.create_dataset(key, data=val)
        if locations is None:
            locations = ["init"]
        if locations is not None:
            if isinstance(locations, str):
                locations = [locations]
            for location in locations:
                if location in meta_file:
                    del meta_file[location]
                g = meta_file.require_group(location)
                for name in self.get_state_fields():
                    if name in g:
                        del g[name]
                    if getattr(self, name) is not None:
                        g.create_dataset(name, data=getattr(self, name))
        # Optional trajectory save
        if save_trajectory and getattr(self, "trajectory", None) is not None:
            # Only write fields that are present in trajectory
            traj_path = os.path.join(path, "trajectory.h5")
            mode = "w" if overwrite_trajectory or not os.path.exists(traj_path) else "r+"
            traj_target = h5py.File(traj_path, mode)
            fields = self.trajectory.fields()
            for name in fields:
                if name in traj_target:
                    del traj_target[name]
                data = self.trajectory._ds[name][...]
                traj_target.create_dataset(name, data=data)
            traj_target.close()
        meta_file.close()


    # ---------- Validation ----------
    def validate(self) -> None:
        spec_map = self._spec_fn()
        for name, spec in spec_map.items():
            arr = getattr(self, name, None)
            if arr is None:
                continue
            spec.validate(arr)
            if 'offset' in name:
                if arr[0] != 0:
                    raise ValueError(f"offset array {name} must start with 0")
                if np.any(arr[1:] < arr[:-1]):
                    raise ValueError(f"offset array {name} must be monotonically increasing")

    # ---------- Repr ----------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        npart = self.n_particles()
        nsys = self.n_systems()
        nvert = self.n_vertices()
        present = sorted([k for k, v in self._spec_fn().items() if getattr(self, k, None) is not None])
        return f"{cls}(N={npart}, S={nsys}, Nv={nvert}, fields={present})"

    # ---------- Join/Split ----------
    # join/split moved to standalone utilities

    # ---------- Internals ----------
    def _load_group_to(self, container: GroupData, g: h5py.Group) -> None:
        for name in g.keys():
            arr = g[name][...]
            container.arrays[name] = arr

    def _apply_group_to_top(self, container: GroupData) -> None:
        # Load static then override with init/final/restart semantics by caller
        spec_map = self._spec_fn()
        for name, arr in container.arrays.items():
            # Defer strict validation until after all arrays are applied
            # to avoid relying on sizes that are computed from other arrays
            if name in spec_map:
                setattr(self, name, arr)
                continue
            # Auto-register unknown arrays if their leading dimension matches
            # a known index space (System, Particle, or Vertex)
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                leading = int(arr.shape[0])
                idx_space: Optional[I] = None
                # Use current sizes
                n_sys = self.n_systems()
                n_par = self.n_particles()
                n_ver = self.n_vertices()
                if leading == n_sys and n_sys > 0:
                    idx_space = I.System
                elif leading == n_par and n_par > 0:
                    idx_space = I.Particle
                elif leading == n_ver and n_ver > 0:
                    idx_space = I.Vertex
                if idx_space is not None:
                    # Register new FieldSpec with dynamic expected shape
                    tail_shape = arr.shape[1:]
                    def _make_expected_shape_fn(space: I, tail: Tuple[int, ...]):
                        return (lambda s=space, t=tail: ((
                            (self.n_systems() if s == I.System else (
                                self.n_particles() if s == I.Particle else (
                                    self.n_vertices() if s == I.Vertex else 0
                                )
                            )),
                        ) + t))
                    spec_map[name] = FieldSpec(
                        name=name,
                        index_space=idx_space,
                        dtype=arr.dtype,
                        expected_shape_fn=_make_expected_shape_fn(idx_space, tail_shape),
                    )
                    setattr(self, name, arr)
                    # If this came from the static group, ensure we save it back under static
                    if container is getattr(self, 'static', None):
                        self._extra_static_fields.add(name)

    # ---------- Calculations ----------
    def calculate_area(self) -> None:
        raise NotImplementedError("calculate_area() needs to be implemented in the derived class")

    def calculate_perimeter(self) -> None:
        raise NotImplementedError("calculate_perimeter() needs to be implemented in the derived class")

    def calculate_shape_parameter(self) -> None:
        self.shape_parameter = self.perimeter ** 2 / (4 * np.pi * self.area)

    # ---------- Dynamic Arrays ----------
    def add_array(self, arr: np.ndarray, name: str, ignore_missing_index_space: bool = False) -> None:
        """Dynamically attach a new array and register it for validation and save.

        The array must have leading dimension equal to one of:
          - number of systems (System index space)
          - number of particles (Particle index space)
          - number of vertices (Vertex index space)

        The field is added to the spec map with a dynamic expected shape,
        and included in the static field list so it is saved under /static.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("add_array: name must be a non-empty string")
        if not isinstance(arr, np.ndarray):
            raise TypeError("add_array: arr must be a numpy.ndarray")
        if arr.ndim < 1:
            raise ValueError("add_array: array must be at least 1-D with leading dimension matching a known index space")

        # Determine index space by matching the leading dimension
        leading = int(arr.shape[0])
        n_sys = self.n_systems()
        n_par = self.n_particles()
        n_ver = self.n_vertices()
        if leading == n_sys and n_sys > 0:
            idx_space = I.System
            lead_dim_fn = self.n_systems
        elif leading == n_par and n_par > 0:
            idx_space = I.Particle
            lead_dim_fn = self.n_particles
        elif leading == n_ver and n_ver > 0:
            idx_space = I.Vertex
            lead_dim_fn = self.n_vertices
        else:
            if ignore_missing_index_space:
                idx_space = I.NoneSpace
                lead_dim_fn = lambda: 0
            else:
                raise ValueError(
                    f"add_array: cannot infer index space for '{name}' with shape {arr.shape}; "
                    f"leading dimension must match number of systems ({n_sys}), particles ({n_par}), or vertices ({n_ver})"
                )

        # Register FieldSpec with dynamic expected shape (leading dimension varies with system/particle/vertex counts)
        tail_shape = arr.shape[1:]
        expected_shape_fn = lambda t=tail_shape, f=lead_dim_fn: (f(),) + t

        # Update spec map in place so derived classes that captured base_map still see this
        spec_map = self._spec_fn()
        if name in spec_map:
            # If already present, ensure compatibility
            existing = spec_map[name]
            if existing.index_space != idx_space:
                if ignore_missing_index_space:
                    existing.index_space = idx_space
                else:
                    raise ValueError(f"add_array: field '{name}' already exists with different index space")
            # Replace dtype/shape constraints to match provided array precisely
            existing.dtype = arr.dtype
            existing.expected_shape_fn = expected_shape_fn
        else:
            spec_map[name] = FieldSpec(
                name=name,
                index_space=idx_space,
                dtype=arr.dtype,
                expected_shape_fn=expected_shape_fn,
            )

        # Attach data and mark to be saved under static
        setattr(self, name, arr)
        self._extra_static_fields.add(name)

    def calculate_packing_fraction(self) -> None:
        self.calculate_area()
        self.packing_fraction = np.array([
            np.sum(self.area[self.system_offset[i]:self.system_offset[i+1]]) / (self.box_size[i,0] * self.box_size[i,1])
            for i in range(self.n_systems())
        ])

    def scale_positions(self, scale: np.ndarray | float) -> None:
        if isinstance(scale, float):
            scale = np.full((self.n_systems(),), scale)
        self._scale_positions_impl(scale)

    def _scale_positions_impl(self, scale: np.ndarray) -> None:
        raise NotImplementedError("_scale_positions_impl() needs to be implemented in the derived class")
    
    def scale_to_packing_fraction(self, packing_fraction: np.ndarray | float) -> None:
        if isinstance(packing_fraction, float):
            packing_fraction = np.full((self.n_systems(),), packing_fraction)
        self.calculate_packing_fraction()  # this is slow
        scale = np.sqrt(self.packing_fraction / packing_fraction)
        self.box_size *= scale[:, None]
        self.scale_positions(scale)
        self.calculate_packing_fraction()

    def set_positions(self, randomness: int, random_seed: int) -> None:
        if randomness == 0:
            # Square lattice positions
            for i in range(self.n_systems()):
                beg = self.system_offset[i]
                end = self.system_offset[i+1]
                size = self.system_size[i]
                self.pos[beg:end] = assign_lattice_positions(size, self.box_size[i])
        elif randomness == 1:
            # Random uniform positions within each system's box
            for i in range(self.n_systems()):
                np.random.seed(random_seed + i)
                beg = self.system_offset[i]
                end = self.system_offset[i+1]
                size = self.system_size[i]
                self.pos[beg:end, 0] = np.random.uniform(
                    0, self.box_size[i,0], size=size
                )
                self.pos[beg:end, 1] = np.random.uniform(
                    0, self.box_size[i,1], size=size
                )
        self._set_positions_impl(randomness, random_seed)

    def _set_positions_impl(self, randomness: int, random_seed: int) -> None:
        raise NotImplementedError("_set_positions_impl() needs to be implemented in the derived class")

    def calculate_kinetic_energy(self) -> None:
        self._calculate_kinetic_energy_impl()
        self.calculate_total_kinetic_energy()

    def _calculate_kinetic_energy_impl(self) -> None:
        raise NotImplementedError("_calculate_kinetic_energy_impl() needs to be implemented in the derived class")

    def calculate_total_kinetic_energy(self) -> None:
        self.ke_total = np.array([
            np.sum(self.ke[self.system_offset[i]:self.system_offset[i+1]])
            for i in range(self.n_systems())
        ])

    def calculate_temperature(self) -> None:
        self.calculate_kinetic_energy()
        self.temperature = self.ke_total * 2 / self.n_dof()
    
    def set_velocities(self, temperature: np.ndarray | float, random_seed: int) -> None:
        if isinstance(temperature, float):
            temperature = np.full((self.n_systems(),), temperature)
        if np.allclose(temperature, 0):
            self.vel.fill(0)
        np.random.seed(random_seed)
        self.vel = np.concatenate([
            np.random.normal(0, np.sqrt(temperature[i]), size=(self.system_size[i], 2))
            for i in range(self.n_systems())
        ])
        self._set_velocities_impl(temperature, random_seed)
        self.remove_center_of_mass_velocity()
        self.calculate_temperature()
        scale = np.sqrt(temperature / self.temperature)
        self.scale_velocities(scale)

    def _set_velocities_impl(self, temperature: np.ndarray, random_seed: int) -> None:
        raise NotImplementedError("_set_velocities_impl() needs to be implemented in the derived class")

    def scale_velocities(self, scale: np.ndarray | float) -> None:
        if isinstance(scale, float):
            scale = np.full((self.n_systems(),), scale)
        self.vel *= scale[self.system_id, None]
        self._scale_velocities_impl(scale)

    def _scale_velocities_impl(self, scale: np.ndarray) -> None:
        raise NotImplementedError("_scale_velocities_impl() needs to be implemented in the derived class")
    
    def remove_center_of_mass_velocity(self) -> None:
        mean_vel = np.array([
            np.mean(self.vel[self.system_offset[i]:self.system_offset[i+1]], axis=0)
            for i in range(self.n_systems())
        ])
        self.vel -= mean_vel[self.system_id]
        self._remove_center_of_mass_velocity_impl()

    def _remove_center_of_mass_velocity_impl(self) -> None:
        raise NotImplementedError("_remove_center_of_mass_velocity_impl() needs to be implemented in the derived class")

    def fill_in_missing_fields(self) -> None:
        raise NotImplementedError("fill_in_missing_fields() needs to be implemented in the derived class")
