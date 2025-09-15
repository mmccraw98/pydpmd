from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from typing import List

from .base_poly_particle import BasePolyParticle
from ..fields import FieldSpec, IndexSpace as I, DT_FLOAT, DT_INT


class RigidBumpy(BasePolyParticle):
    def __init__(self):
        super().__init__()
        # Rigid bumpy specific
        self.angle = None           # (N,)
        self.torque = None          # (N,)
        self.angular_vel = None     # (N,)
        self.angular_period = None  # (N,)
        self.rad = None             # (N,)
        self.e_interaction = None   # (S,)

        base_map = getattr(self, "_spec_fn")()
        base_map.update({
            "angle": FieldSpec("angle", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "torque": FieldSpec("torque", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "angular_vel": FieldSpec("angular_vel", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "angular_period": FieldSpec("angular_period", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "rad": FieldSpec("rad", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "e_interaction": FieldSpec("e_interaction", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
        })
        self._spec_fn = lambda m=base_map: m

    def allocate_particles(self, N: int) -> None:
        super().allocate_particles(N)
        self.mass = np.empty((N,), dtype=DT_FLOAT)
        self.rad = np.empty((N,), dtype=DT_FLOAT)
        self.angular_period = np.empty((N,), dtype=DT_FLOAT)
        self.angle = np.empty((N,), dtype=DT_FLOAT)
        self.torque = np.empty((N,), dtype=DT_FLOAT)
        self.angular_vel = np.empty((N,), dtype=DT_FLOAT)

    def allocate_systems(self, S: int) -> None:
        super().allocate_systems(S)
        self.e_interaction = np.empty((S,), dtype=DT_FLOAT)

    def get_static_fields(self) -> List[str]:
        static_fields = super().get_static_fields()
        static_fields += ['mass', 'moment_inertia', 'area', 'rad']
        return static_fields
    
    def get_state_fields(self) -> List[str]:
        state_fields = super().get_state_fields()
        state_fields += ['pos', 'angular_period', 'angle', 'torque', 'angular_vel', 'vel', 'force']
        return state_fields
    
    def set_ids(self) -> None:
        super().set_ids()
        self.angular_period = 2 * np.pi / self.n_vertices_per_particle
        self.angular_period[self.n_vertices_per_particle == 1] = 0
    
    def n_dof(self) -> np.ndarray:
        return np.bincount(self.system_id, weights=((self.moment_inertia > 0) + 2)).astype(DT_INT)
    
    def calculate_area(self) -> None:
        qs = 1e4
        single_vertex_mask = np.where(self.n_vertices_per_particle == 1)[0]
        self.area[single_vertex_mask] = np.pi * self.rad[single_vertex_mask] ** 2
        multi_vertex_mask = np.where(self.n_vertices_per_particle > 1)[0]
        if len(multi_vertex_mask) > 0:
            dist = np.linalg.norm(self.vertex_pos[self.particle_offset[multi_vertex_mask]] - self.vertex_pos[self.particle_offset[multi_vertex_mask] + 1], axis=-1)
            nv_rad_dist = np.column_stack([self.n_vertices_per_particle[multi_vertex_mask], self.rad[multi_vertex_mask], dist])
            for nrd in np.unique(nv_rad_dist, axis=0):
                mask = np.all(np.isclose(nv_rad_dist, nrd), axis=1)
                n, _, d = nrd
                po = self.particle_offset[multi_vertex_mask & mask]
                vpos = self.vertex_pos[po[0]:po[0] + int(n)]
                vrad = self.vertex_rad[po[0]:po[0] + int(n)]
                if n == 2:
                    if d > np.sum(vrad):  # two non-overlapping circles
                        self.area[multi_vertex_mask[mask]] = np.pi * np.sum(vrad ** 2)
                    else:  # two overlapping circles
                        self.area[multi_vertex_mask[mask]] = unary_union([Point(_vpos).buffer(_vrad, quad_segs=qs) for _vpos, _vrad in zip(vpos, vrad)]).area
                else:
                    self.area[multi_vertex_mask[mask]] = unary_union([Polygon(vpos)] + [Point(_vpos).buffer(_vrad, quad_segs=qs) for _vpos, _vrad in zip(vpos, vrad)]).area

    def set_positions(self, randomness: int, random_seed: int) -> None:
        pos_old = self.pos.copy()
        angle_old = self.angle.copy()
        super().set_positions(randomness, random_seed)
        delta_pos = self.pos - pos_old
        delta_angle = self.angle - angle_old
        self.vertex_pos += delta_pos[self.vertex_particle_id]
        self._rotate_vertices(delta_angle[self.vertex_particle_id])
    
    def _rotate_vertices(self, delta_angle: np.ndarray) -> None:
        multi_vertex_mask = np.where(self.n_vertices_per_particle > 1)[0]
        if len(multi_vertex_mask) > 0:
            vertex_ids = np.concatenate([np.arange(self.particle_offset[i], self.particle_offset[i + 1]) for i in multi_vertex_mask])
            particle_ids = self.vertex_particle_id[vertex_ids]
            local_vertex_pos = self.vertex_pos[vertex_ids] - self.pos[particle_ids]
            dtheta = delta_angle[vertex_ids]
            cos_dtheta = np.cos(dtheta)
            sin_dtheta = np.sin(dtheta)
            x = local_vertex_pos[:, 0]
            y = local_vertex_pos[:, 1]
            rotated_x = x * cos_dtheta - y * sin_dtheta
            rotated_y = x * sin_dtheta + y * cos_dtheta
            rotated_vertex_pos = np.column_stack([rotated_x, rotated_y])
            self.vertex_pos[vertex_ids] = rotated_vertex_pos + self.pos[particle_ids]

    def _set_positions_impl(self, randomness: int, random_seed: int) -> None:
        np.random.seed(random_seed)
        self.angle = np.random.uniform(0, 1, size=(self.n_particles(),)) * self.angular_period

    def _calculate_kinetic_energy_impl(self) -> None:
        self.ke = 0.5 * self.mass * np.sum(self.vel ** 2, axis=1) + self.moment_inertia * self.angular_vel ** 2

    def _set_velocities_impl(self, temperature: np.ndarray, random_seed: int) -> None:
        np.random.seed(random_seed)
        self.angular_vel = np.concatenate([
            np.random.normal(0, np.sqrt(temperature[i]), size=(self.system_size[i],))
            for i in range(self.n_systems())
        ])
        self.angular_vel *= (self.moment_inertia > 0)[:, None]
        self.set_vertex_velocities_from_particle_velocities()

    def _scale_velocities_impl(self, scale: np.ndarray) -> None:
        self.angular_vel *= scale[self.system_id]
        self.set_vertex_velocities_from_particle_velocities()

    def _remove_center_of_mass_velocity_impl(self) -> None:
        pass  # no-op

    def set_vertex_velocities_from_particle_velocities(self) -> None:
        # translation:
        self.vertex_vel = self.vel[self.vertex_particle_id]
        # rotation
        multi_vertex_mask = np.where(self.n_vertices_per_particle > 1)[0]
        if len(multi_vertex_mask) > 0:
            vertex_ids = np.concatenate([np.arange(self.particle_offset[i], self.particle_offset[i + 1]) for i in multi_vertex_mask])
            particle_ids = self.vertex_particle_id[vertex_ids]
            local_vertex_ids = vertex_ids - self.particle_offset[particle_ids]
            vertex_angles = local_vertex_ids * self.angular_period[particle_ids] + self.angle[particle_ids]
            distances = np.linalg.norm(self.vertex_pos[vertex_ids] - self.pos[particle_ids], axis=1)
            angular_component = np.column_stack([np.sin(vertex_angles) * distances, np.cos(vertex_angles) * distances])
            self.vertex_vel[vertex_ids] += angular_component

    def set_particle_velocities_from_vertex_velocities(self) -> None:
        raise NotImplementedError("set_particle_velocities_from_vertex_velocities() needs to be implemented in the derived class")

    def set_vertices_on_particles_as_disk(self) -> None:
        single_vertex_mask = np.where(self.n_vertices_per_particle == 1)[0]
        self.vertex_pos[self.particle_offset[single_vertex_mask]] = self.pos[single_vertex_mask]
        multi_vertex_mask = np.where(self.n_vertices_per_particle > 1)[0]
        if len(multi_vertex_mask) > 0:
            vertex_ids = np.concatenate([np.arange(self.particle_offset[i], self.particle_offset[i + 1]) for i in multi_vertex_mask])
            particle_ids = self.vertex_particle_id[vertex_ids]
            local_vertex_ids = vertex_ids - self.particle_offset[particle_ids]
            inner_radius = self.rad[particle_ids] - self.vertex_rad[vertex_ids]
            vertex_angles = local_vertex_ids * self.angular_period[particle_ids] + self.angle[particle_ids]
            self.vertex_pos[vertex_ids] = np.column_stack([
                inner_radius * np.cos(vertex_angles),
                inner_radius * np.sin(vertex_angles),
            ]) + self.pos[particle_ids]

    def fill_in_missing_fields(self) -> None:
        if self.angular_period is None:
            self.angular_period = 2 * np.pi / self.n_vertices_per_particle
            self.angular_period[self.n_vertices_per_particle == 1] = 0
            self.angle[self.n_vertices_per_particle == 1] = 0