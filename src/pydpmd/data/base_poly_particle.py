from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

from .base_particle import BaseParticle
from ..fields import FieldSpec, IndexSpace as I, DT_FLOAT, DT_INT


class BasePolyParticle(BaseParticle):
    def __init__(self):
        super().__init__()
        # Poly-specific
        self.n_vertices_per_particle = None  # (N,)
        self.particle_offset = None          # (N+1,)
        self.vertex_particle_id = None       # (Nv,)
        self.vertex_system_id = None         # (Nv,)
        self.vertex_system_offset = None     # (S+1,)
        self.vertex_system_size = None       # (S,)
        self.vertex_pos = None               # (Nv,2)
        self.vertex_vel = None               # (Nv,2)
        self.vertex_force = None             # (Nv,2)
        self.vertex_pe = None                # (Nv,)
        self.vertex_mass = None              # (Nv,)
        self.vertex_rad = None               # (Nv,)
        self.moment_inertia = None           # (N,)
        self.mass = None                     # (N,)

        base_map = getattr(self, "_spec_fn")()
        base_map.update({
            "n_vertices_per_particle": FieldSpec("n_vertices_per_particle", I.Particle, DT_INT, expected_shape_fn=lambda: (self.n_particles(),)),
            "particle_offset": FieldSpec("particle_offset", I.NoneSpace, DT_INT, expected_shape_fn=lambda: (self.n_particles()+1,)),
            "vertex_particle_id": FieldSpec("vertex_particle_id", I.Vertex, DT_INT, expected_shape_fn=lambda: (self.n_vertices(),)),
            "vertex_system_id": FieldSpec("vertex_system_id", I.Vertex, DT_INT, expected_shape_fn=lambda: (self.n_vertices(),)),
            "vertex_system_offset": FieldSpec("vertex_system_offset", I.System, DT_INT, expected_shape_fn=lambda: (self.n_systems()+1,)),
            "vertex_system_size": FieldSpec("vertex_system_size", I.System, DT_INT, expected_shape_fn=lambda: (self.n_systems(),)),
            "vertex_pos": FieldSpec("vertex_pos", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),2)),
            "vertex_vel": FieldSpec("vertex_vel", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),2)),
            "vertex_force": FieldSpec("vertex_force", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),2)),
            "vertex_pe": FieldSpec("vertex_pe", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),)),
            "vertex_mass": FieldSpec("vertex_mass", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),)),
            "vertex_rad": FieldSpec("vertex_rad", I.Vertex, DT_FLOAT, expected_shape_fn=lambda: (self.n_vertices(),)),
            "moment_inertia": FieldSpec("moment_inertia", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "mass": FieldSpec("mass", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
        })
        self._spec_fn = lambda m=base_map: m
    
    def allocate_particles(self, N: int) -> None:
        super().allocate_particles(N)
        self.n_vertices_per_particle = np.empty((N,), dtype=DT_INT)
        self.particle_offset = np.empty((N+1,), dtype=DT_INT)
        self.moment_inertia = np.zeros((N,), dtype=DT_FLOAT)

    def allocate_systems(self, S: int) -> None:
        super().allocate_systems(S)
        self.vertex_system_offset = np.empty((S+1,), dtype=DT_INT)
        self.vertex_system_size = np.empty((S,), dtype=DT_INT)

    def allocate_vertices(self, Nv: int) -> None:
        self.vertex_pos = np.empty((Nv, 2), dtype=DT_FLOAT)
        self.vertex_vel = np.zeros((Nv, 2), dtype=DT_FLOAT)
        self.vertex_force = np.zeros((Nv, 2), dtype=DT_FLOAT)
        self.vertex_pe = np.zeros((Nv,), dtype=DT_FLOAT)
        self.vertex_mass = np.empty((Nv,), dtype=DT_FLOAT)
        self.vertex_rad = np.empty((Nv,), dtype=DT_FLOAT)
        self.vertex_particle_id = np.empty((Nv,), dtype=DT_INT)
        self.vertex_system_id = np.empty((Nv,), dtype=DT_INT)

    def get_static_fields(self) -> List[str]:
        static_fields = super().get_static_fields()
        static_fields += ['vertex_mass', 'vertex_rad', 'e_interaction', 'n_vertices_per_particle', 'particle_offset', 'vertex_particle_id', 'vertex_system_id', 'vertex_system_offset', 'vertex_system_size']
        return static_fields
    
    def get_state_fields(self) -> List[str]:
        state_fields = super().get_state_fields()
        state_fields += ['vertex_pos', 'vertex_vel', 'vertex_force', 'vertex_pe']
        return state_fields
    
    def set_ids(self) -> None:
        super().set_ids()
        if self.n_systems() == 1:
            self.vertex_system_id.fill(0)
            self.vertex_system_size.fill(self.n_vertices())
        self.vertex_particle_id = np.concatenate([np.ones(self.n_vertices_per_particle[i], dtype=DT_INT) * i for i in range(self.n_particles())])
        self.vertex_system_offset = np.concatenate([[0], np.cumsum(self.vertex_system_size)]).astype(DT_INT)
        self.particle_offset = np.concatenate([[0], np.cumsum(self.n_vertices_per_particle)]).astype(DT_INT)

    def n_dof(self) -> np.ndarray:
        raise NotImplementedError("n_dof() needs to be implemented in the derived class")
    
    def calculate_area(self) -> None:
        raise NotImplementedError("calculate_area() needs to be implemented in the derived class")

    def _scale_positions_impl(self, scale: np.ndarray) -> None:
        new_pos = self.pos * scale[self.system_id, None]
        displacement = new_pos - self.pos
        self.pos = new_pos
        self.vertex_pos += displacement[self.vertex_particle_id]

    def _set_positions_impl(self, randomness: int, random_seed: int) -> None:
        raise NotImplementedError("_set_positions_impl() needs to be implemented in the derived class")

    def _calculate_kinetic_energy_impl(self) -> None:
        raise NotImplementedError("calculate_kinetic_energy() needs to be implemented in the derived class")

    def _set_velocities_impl(self, temperature: np.ndarray, random_seed: int) -> None:
        raise NotImplementedError("_set_velocities_impl() needs to be implemented in the derived class")

    def _scale_velocities_impl(self, scale: np.ndarray) -> None:
        raise NotImplementedError("_scale_velocities_impl() needs to be implemented in the derived class")

    def _remove_center_of_mass_velocity_impl(self) -> None:
        raise NotImplementedError("_remove_center_of_mass_velocity_impl() needs to be implemented in the derived class")

    def set_vertex_velocities_from_particle_velocities(self) -> None:
        raise NotImplementedError("set_vertex_velocities_from_particle_velocities() needs to be implemented in the derived class")

    def set_particle_velocities_from_vertex_velocities(self) -> None:
        raise NotImplementedError("set_particle_velocities_from_vertex_velocities() needs to be implemented in the derived class")

    def set_particle_positions_from_vertex_positions(self) -> None:
        mean_pos = np.concatenate([
            np.mean(self.vertex_pos[self.particle_offset[i]:self.particle_offset[i+1]], axis=0)
            for i in range(self.n_particles())
        ])
        self.pos = mean_pos[self.vertex_particle_id]

    def system_sum(self, arr: np.ndarray) -> np.ndarray:
        """Sum over the system dimension."""
        return np.add.reduceat(arr, self.system_offset[:-1])  # reduceat assumes to sum to the last element so we need to drop the last element

    def calculate_uniform_vertex_mass(self) -> None:
        self.vertex_mass = (self.mass / self.n_vertices_per_particle)[self.vertex_particle_id]

    def calculate_inertia(self) -> None:
        single_vertex_mask = np.where(self.n_vertices_per_particle == 1)[0]
        self.moment_inertia[single_vertex_mask] = 0
        multi_vertex_mask = np.where(self.n_vertices_per_particle > 1)[0]
        if len(multi_vertex_mask) > 0:
            vertex_ids = np.concatenate([np.arange(self.particle_offset[i], self.particle_offset[i + 1]) for i in multi_vertex_mask])
            particle_ids = self.vertex_particle_id[vertex_ids]
            scaled_distances = self.vertex_mass[vertex_ids] * np.linalg.norm(self.vertex_pos[vertex_ids] - self.pos[particle_ids], axis=1) ** 2
            self.moment_inertia[multi_vertex_mask] = np.array([np.sum(scaled_distances[self.particle_offset[i]:self.particle_offset[i + 1]]) for i in multi_vertex_mask])
