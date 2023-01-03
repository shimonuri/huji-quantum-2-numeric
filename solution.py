import dataclasses
import numpy as np
import scipy.special

import constants
import logging


@dataclasses.dataclass
class Solution:
    wave_function: np.ndarray
    uwave_function: np.ndarray
    l_level: int
    m_level: int
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
    def abs_at_infinity(self):
        return abs(self.uwave_function[-1])

    @property
    def at_infinity(self):
        return self.uwave_function[-1]

    @property
    def rms_radius(self):
        return np.sqrt(
            np.sum(self.r_grid ** 2 * self.uwave_function ** 2)
            / np.sum(self.uwave_function ** 2)
        )


def add_spherical_harmonic(wave_function_no_sph, l_level, m_level):
    return wave_function_no_sph * scipy.special.sph_harm(l_level, m_level, 0, 0).real


def normalize(wave_function, r_grid):
    norm = np.sqrt(np.trapz(y=np.square(np.abs(wave_function)), x=r_grid))
    if norm > 0:
        return wave_function / norm

    return wave_function
