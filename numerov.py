import numpy as np
import constants
import solution


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

    wave_function_no_sph = np.zeros(len(r_grid))
    for i in range(len(u_wave_function)):
        wave_function_no_sph[i] = u_wave_function[i] / r_grid[i]

    return solution.Solution(
        uwave_function=solution.normalize(u_wave_function, r_grid),
        wave_function=solution.normalize(
            solution.add_spherical_harmonic(
                wave_function_no_sph, l=angular_momentum, m=0
            ),
            r_grid,
        ),
        angular_momentum=angular_momentum,
        energy=energy,
        r_grid=r_grid,
    )


# Solution to the Klein-Gordon w.f.
def numerov_kgwf(E, l, potential, r_grid):
    work = np.zeros(len(r_grid))
    wave_function = np.zeros(len(r_grid))
    return solution.normalize(wave_function, r_grid)

