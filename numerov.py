import numpy as np
import constants
import scipy.constants


def numerov_wf(
    energy, angular_momentum, potential, r_grid, mass_a, mass_b,
):
    reduced_mass = mass_a * mass_b / (mass_a + mass_b)
    inhomogeneous = (
        lambda r: ((2 * reduced_mass) / (constants.HBARC ** 2))
        * (energy - potential(r))
        - angular_momentum * (angular_momentum + 1) / r ** 2
    )
    r_diff = r_grid[1] - r_grid[0]
    u_wave_function = np.zeros(len(r_grid))
    u_wave_function[0] = 0
    u_wave_function[1] = r_diff ** (angular_momentum + 1)

    for i in range(1, len(r_grid) - 1):
        u_wave_function[i + 1] = (
            u_wave_function[i] * (2 - (5 / 6) * r_diff ** 2 * inhomogeneous(r_grid[i]))
            - u_wave_function[i - 1]
            * (1 + (1 / 12) * r_diff ** 2 * inhomogeneous(r_grid[i - 1]))
        ) / (1 + (1 / 12) * r_diff ** 2 * inhomogeneous(r_grid[i + 1]))

    wave_fucntion = np.zeros(len(r_grid))
    for i in range(len(u_wave_function)):
        wave_fucntion[i] = u_wave_function[i] / r_grid[i]

    return normalize(u_wave_function, r_grid), normalize(wave_fucntion, r_grid)


# Solution to the Klein-Gordon w.f.
def numerov_kgwf(E, l, potential, r_grid):
    work = np.zeros(len(r_grid))
    wave_function = np.zeros(len(r_grid))
    return normalize(wave_function, r_grid)


def normalize(wave_function, r_grid):
    norm = np.sqrt(np.trapz(y=np.square(np.abs(wave_function)), x=r_grid))
    if norm > 0:
        return wave_function / norm

    return wave_function
