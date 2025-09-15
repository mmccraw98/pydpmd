from __future__ import annotations
import numpy as np

from .base_point_particle import BasePointParticle


class Disk(BasePointParticle):
    def __init__(self):
        super().__init__()

    def n_dof(self) -> np.ndarray:
        return self.system_size.copy() * 2

    def calculate_area(self) -> None:
        self.area = np.pi * self.rad ** 2

    def _set_positions_impl(self, randomness: int, random_seed: int) -> None:
        pass  # no-op

    def _calculate_kinetic_energy_impl(self) -> None:
        self.ke = 0.5 * (self.vel[:, 0]**2 + self.vel[:, 1]**2) * self.mass

    def _set_velocities_impl(self, temperature: np.ndarray, random_seed: int) -> None:
        pass  # no-op

    def _scale_velocities_impl(self, scale: np.ndarray) -> None:
        pass  # no-op

    def fill_in_missing_fields(self) -> None:
        pass  # no-op