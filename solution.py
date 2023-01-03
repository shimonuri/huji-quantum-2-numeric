import dataclasses
import numpy as np
import scipy.special

import constants


@dataclasses.dataclass
class Solution:
    wave_function: np.ndarray
    uwave_function: np.ndarray
    angular_momentum: int
    energy: float
    r_grid: np.ndarray
    steps: int
    level: int

    @property
    def r_max(self):
        return self.r_grid[-1]

    @property
    def error(self):
        return np.abs(1 - self.energy / (-constants.RY / (self.level ** 2)))

    @property
    def at_infinity(self):
        return abs(self.wave_function[-1])


def add_spherical_harmonic(wave_function_no_sph, l, m):
    if l == 0:
        return wave_function_no_sph * scipy.special.sph_harm(0, 0, 0, 0).real
    raise NotImplementedError("l != 0 not implemented")


def normalize(wave_function, r_grid):
    norm = np.sqrt(np.trapz(y=np.square(np.abs(wave_function)), x=r_grid))
    if norm > 0:
        return wave_function / norm

    return wave_function
