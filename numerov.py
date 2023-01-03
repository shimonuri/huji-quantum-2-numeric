import numpy as np
import constants
import solution
import logging
logging.getLogger().setLevel(logging.INFO)

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
        steps=len(r_grid),
        level=1,
    )


def find_bound_state(
    potential,
    r_grid,
    mass_a,
    mass_b,
    angular_momentum,
    min_energy,
    max_energy,
    exit_param=1e-6,
    max_iterations=int(100),
):
    max_energy_solution = numerov_wf(
        max_energy, angular_momentum, potential, r_grid, mass_a, mass_b,
    )
    min_energy_solution = numerov_wf(
        min_energy, angular_momentum, potential, r_grid, mass_a, mass_b,
    )
    solution = min(
        min_energy_solution, max_energy_solution, key=lambda s: s.at_infinity
    )
    previous_energy = np.inf
    i = 0
    while abs(previous_energy - solution.energy) > exit_param:
        i += 1
        if i % 100 == 0:
            logging.info(
                f"iteration {i}, at_infinity {solution.at_infinity}, energy {solution.energy}"
            )
        if i > max_iterations:
            logging.warning("Max iterations reached")
            break

        average_energy = (max_energy + min_energy) / 2
        average_energy_solution = numerov_wf(
            average_energy, angular_momentum, potential, r_grid, mass_a, mass_b,
        )
        previous_energy = solution.energy
        solution = average_energy_solution
        if max_energy_solution.at_infinity < min_energy_solution.at_infinity:
            min_energy = average_energy
            min_energy_solution = average_energy_solution
        else:
            max_energy = average_energy
            max_energy_solution = average_energy_solution

    return solution


# Solution to the Klein-Gordon w.f.
def numerov_kgwf(E, l, potential, r_grid):
    work = np.zeros(len(r_grid))
    wave_function = np.zeros(len(r_grid))
    return solution.normalize(wave_function, r_grid)
