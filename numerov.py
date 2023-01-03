import numpy as np
import constants
import solution
import logging
import scipy.stats

logging.getLogger().setLevel(logging.INFO)


def numerov_wf(
    energy, n_level, l_level, potential, r_grid, mass_a, mass_b,
):
    reduced_mass = mass_a * mass_b / (mass_a + mass_b)
    inhomogeneous = lambda r: ((2 * reduced_mass) / (constants.HBARC ** 2)) * (
        energy - potential(r)
    ) - (l_level * (l_level + 1)) / (r ** 2)
    r_diff = r_grid[1] - r_grid[0]
    u_wave_function = np.zeros(len(r_grid))
    u_wave_function[0] = 0
    u_wave_function[1] = r_diff ** (l_level + 1)

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
                wave_function_no_sph, l_level=l_level, m_level=0
            ),
            r_grid,
        ),
        n_level=n_level,
        l_level=l_level,
        m_level=0,
        energy=energy,
        r_grid=r_grid,
        steps=len(r_grid),
    )


def find_bound_state(
    potential,
    r_grid,
    mass_a,
    mass_b,
    n_level,
    l_level,
    min_energy,
    max_energy,
    exit_param=1e-6,
    max_iterations=int(100),
):
    if l_level > n_level - 1:
        raise ValueError("l_level must be less than n_level - 1")

    max_energy_solution = numerov_wf(
        max_energy, n_level, l_level, potential, r_grid, mass_a, mass_b,
    )
    min_energy_solution = numerov_wf(
        min_energy, n_level, l_level, potential, r_grid, mass_a, mass_b,
    )
    current_solution = min(
        min_energy_solution, max_energy_solution, key=lambda s: s.abs_at_infinity
    )
    if np.sign(min_energy_solution.at_infinity * max_energy_solution.at_infinity) == 1:
        logging.warning("max_energy and min_energy has same sign at infinity")

    previous_energy = np.inf
    i = 0
    while abs(previous_energy - current_solution.energy) > exit_param:
        i += 1
        if i % 100 == 0:
            logging.info(
                f"iteration {i}, at_infinity {current_solution.abs_at_infinity}, energy {current_solution.energy}"
            )
        if i > max_iterations:
            logging.warning("Max iterations reached")
            break

        newton_energy_solution = _get_newton_solution(
            n_level,
            l_level,
            mass_a,
            mass_b,
            max_energy_solution,
            min_energy_solution,
            potential,
            r_grid,
        )
        previous_energy = current_solution.energy
        current_solution = newton_energy_solution

        if (
            np.sign(
                newton_energy_solution.at_infinity * max_energy_solution.at_infinity
            )
            == 1
        ):
            max_energy_solution = newton_energy_solution
        else:
            min_energy_solution = newton_energy_solution

    return current_solution


def _get_newton_solution(
    n_level,
    l_level,
    mass_a,
    mass_b,
    max_energy_solution,
    min_energy_solution,
    potential,
    r_grid,
):
    linear_curve = scipy.stats.linregress(
        x=[min_energy_solution.at_infinity, max_energy_solution.at_infinity],
        y=[min_energy_solution.energy, max_energy_solution.energy],
    )
    newton_energy = linear_curve.intercept
    newton_energy_solution = numerov_wf(
        newton_energy, n_level, l_level, potential, r_grid, mass_a, mass_b,
    )
    return newton_energy_solution


# Solution to the Klein-Gordon w.f.
def numerov_kgwf(E, l, potential, r_grid):
    work = np.zeros(len(r_grid))
    wave_function = np.zeros(len(r_grid))
    return solution.normalize(wave_function, r_grid)
