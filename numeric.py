import numpy as np
import constants
import solution
import logging
import scipy.stats
import enum

logging.getLogger().setLevel(logging.INFO)


class NumerovCase(enum.Enum):
    RELATIVISTIC = 1
    NON_RELATIVISTIC = 2


def numerov_wf(
    energy,
    n_level,
    l_level,
    potential,
    r_grid,
    mass_a,
    mass_b,
    should_find_wave=False,
    numerov_case=NumerovCase.NON_RELATIVISTIC,
):
    reduced_mass = mass_a * mass_b / (mass_a + mass_b)
    if numerov_case == NumerovCase.NON_RELATIVISTIC:
        inhomogeneous = lambda r: ((2 * reduced_mass) / (constants.HBARC ** 2)) * (
            energy - potential(r)
        ) - (l_level * (l_level + 1)) / (r ** 2)
    elif numerov_case == NumerovCase.RELATIVISTIC:
        inhomogeneous = lambda r: (1 / (constants.HBARC ** 2)) * (
            energy - potential(r)
        ) ** 2 + (
            2 * constants.M_RED * (energy - potential(r)) / (constants.HBARC ** 2)
            + (l_level * (l_level + 1)) / r ** 2
        )
    else:
        raise ValueError("Unknown case")

    r_diff = r_grid[1] - r_grid[0]
    uwave_function = np.zeros(len(r_grid))
    uwave_function[0] = 0
    uwave_function[1] = r_diff ** (l_level + 1)

    for i in range(1, len(r_grid) - 1):
        uwave_function[i + 1] = (
            uwave_function[i] * (2 - (5 / 6) * r_diff ** 2 * inhomogeneous(r_grid[i]))
            - uwave_function[i - 1]
            * (1 + (1 / 12) * r_diff ** 2 * inhomogeneous(r_grid[i - 1]))
        ) / (1 + (1 / 12) * r_diff ** 2 * inhomogeneous(r_grid[i + 1]))

    wave_function_no_sph = np.zeros(len(r_grid))
    for i in range(len(uwave_function)):
        wave_function_no_sph[i] = uwave_function[i] / r_grid[i]

    if should_find_wave:
        wave_function = solution.add_spherical_harmonic(
            wave_function_no_sph, l_level=l_level, m_level=0
        )
        wave_norm = solution.get_norm(wave_function, r_grid)
        uwave_function = (1 / wave_norm) * uwave_function
        wave_function = (1 / wave_norm) * wave_function
    else:
        wave_function = None
        uwave_norm = solution.get_norm(uwave_function, r_grid)
        uwave_function = (1 / uwave_norm) * uwave_function

    return solution.Solution(
        uwave_function=uwave_function,
        wave_function=wave_function,
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
    exit_param=1e-15,
    max_iterations=int(50),
    should_find_wave=False,
    numerov_case=NumerovCase.NON_RELATIVISTIC,
):
    if l_level > n_level - 1:
        raise ValueError("l_level must be less than n_level - 1")

    max_energy_solution = numerov_wf(
        max_energy,
        n_level,
        l_level,
        potential,
        r_grid,
        mass_a,
        mass_b,
        should_find_wave,
        numerov_case,
    )
    min_energy_solution = numerov_wf(
        min_energy,
        n_level,
        l_level,
        potential,
        r_grid,
        mass_a,
        mass_b,
        should_find_wave,
        numerov_case,
    )
    current_solution = min(
        min_energy_solution, max_energy_solution, key=lambda s: s.abs_at_infinity
    )
    if np.sign(min_energy_solution.at_infinity * max_energy_solution.at_infinity) == 1:
        logging.warning("max_energy and min_energy has same sign at infinity")
        import ipdb; ipdb.set_trace()

    previous_energy = np.inf
    i = 0
    while (
        abs(current_solution.at_infinity) > exit_param
        and abs(max_energy_solution.energy - min_energy_solution.energy) > exit_param
    ):
        # print(
        #     f"current energy: {current_solution.energy / constants.RY}, at_infinity: {current_solution.at_infinity}"
        # )
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
            should_find_wave,
            numerov_case,
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
    should_find_wave,
    case,
):
    linear_curve = scipy.stats.linregress(
        x=[min_energy_solution.at_infinity, max_energy_solution.at_infinity],
        y=[min_energy_solution.energy, max_energy_solution.energy],
    )
    newton_energy = linear_curve.intercept
    newton_energy_solution = numerov_wf(
        newton_energy,
        n_level,
        l_level,
        potential,
        r_grid,
        mass_a,
        mass_b,
        should_find_wave,
        case,
    )
    return newton_energy_solution


def energy_shift_perturbation(r_grid, basic_solution, perturbation_potential):
    perturbation = np.array([perturbation_potential(r) for r in r_grid])
    return np.trapz(
        y=perturbation * (basic_solution.uwave_function ** 2), x=r_grid
    ) / solution.get_norm(basic_solution.uwave_function, r_grid)
