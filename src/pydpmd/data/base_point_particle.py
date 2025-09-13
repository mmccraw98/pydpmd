from __future__ import annotations
import numpy as np
from typing import List

from .base_particle import BaseParticle
from ..fields import FieldSpec, IndexSpace as I, DT_FLOAT


class BasePointParticle(BaseParticle):
    def __init__(self):
        super().__init__()
        self.e_interaction = None  # (S,)
        self.mass = None           # (N,)
        self.rad = None            # (N,)
        # Extend spec for point fields
        base_map = getattr(self, "_spec_fn")()
        base_map.update({
            "e_interaction": FieldSpec("e_interaction", I.System, DT_FLOAT, expected_shape_fn=lambda: (self.n_systems(),)),
            "mass": FieldSpec("mass", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
            "rad": FieldSpec("rad", I.Particle, DT_FLOAT, expected_shape_fn=lambda: (self.n_particles(),)),
        })
        self._spec_fn = lambda m=base_map: m

    def allocate_particles(self, N: int) -> None:
        super().allocate_particles(N)
        self.mass = np.empty((N,), dtype=DT_FLOAT)
        self.rad = np.empty((N,), dtype=DT_FLOAT)
    
    def allocate_systems(self, S: int) -> None:
        super().allocate_systems(S)
        self.e_interaction = np.empty((S,), dtype=DT_FLOAT)

    def n_dof(self) -> np.ndarray:
        raise NotImplementedError("n_dof() needs to be implemented in the derived class")
    
    def get_static_fields(self) -> List[str]:
        static_fields = super().get_static_fields()
        static_fields += ['e_interaction', 'mass', 'rad', 'area']
        return static_fields
    
    def get_state_fields(self) -> List[str]:
        state_fields = super().get_state_fields()
        state_fields += ['pos', 'vel', 'force']
        return state_fields

    def calculate_area(self) -> None:
        raise NotImplementedError("calculate_area() needs to be implemented in the derived class")

    def _scale_positions_impl(self, scale: np.ndarray) -> None:
        self.pos *= scale[self.system_id, None]

    def _set_positions_impl(self, randomness: int, random_seed: int) -> None:
        raise NotImplementedError("_set_positions_impl() needs to be implemented in the derived class")

    def _calculate_kinetic_energy_impl(self) -> None:
        raise NotImplementedError("_calculate_kinetic_energy_impl() needs to be implemented in the derived class")

    def _set_velocities_impl(self, temperature: np.ndarray, random_seed: int) -> None:
        raise NotImplementedError("_set_velocities_impl() needs to be implemented in the derived class")

    def _scale_velocities_impl(self, scale: np.ndarray) -> None:
        raise NotImplementedError("_scale_velocities_impl() needs to be implemented in the derived class")

    def _remove_center_of_mass_velocity_impl(self) -> None:
        pass  # no-op
