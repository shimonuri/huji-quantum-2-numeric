from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit, brentq
from scipy.integrate import simps
from math import pi
import itertools
import pathlib
import numpy as np
import constants
import potentials
import numerov
import solution


def energy_shift_perutrbation(r_grid, u):
    ##  COMPLETE  ##
    e_shift = constants.DUMMY
    ##  COMPLETE  ##

    return e_shift


def FindBoundState(potential, l, Emin, Emax, r_grid):
    # locate the exact binding energy in the range (Emin,Emax)
    # return:
    #         E - energy,
    #         u[0:len(r_grid)] - wave function

    ## COMPLETE ##
    ## COMPLETE ##
    ## COMPLETE ##

    return (E, u)


class Task:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.log_file = None
        self.plot_file = None

    def run(self, output_dir):
        raise NotImplementedError("Task.run() is not implemented")

    def _log(self, text):
        if self.log_file is None:
            raise RuntimeError("log_file is not initialized")
        message = f"{self.name}: {text}"
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def _open_output_files(self, output_dir):
        if self.log_file is not None:
            raise RuntimeError("log_file is already initialized")

        self.plot_file = PdfPages(output_dir / f"{self.name}.pdf")
        self.log_file = (output_dir / f"{self.name}.log").open("wt")

    def _close_output_files(self):
        self.plot_file.close()
        self.log_file.close()


class PointNucleus(Task):
    def __init__(self, rmin, rmax, n_grid_points, energies, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.rmax = rmax
        self.n_grid_points = n_grid_points
        self.energies = energies

    def run(self, output_dir):
        self._open_output_files(pathlib.Path(output_dir))
        self._log(f"Start")
        r_grid = self._get_r_grid(
            rmin=self.rmin, rmax=self.rmax, n_grid_points=self.n_grid_points,
        )
        analytic_solution, numeric_solutions = self._solve(
            energies=self.energies, r_grid=r_grid
        )
        self._plot(analytic_solution, numeric_solutions)
        self._close_output_files()

    def _solve(self, energies, r_grid):
        numeric_solutions = self._get_numeric_solutions(r_grid, 0, energies)
        analytic_solution = self._get_analytic_solution(r_grid)
        return analytic_solution, numeric_solutions

    def _plot(self, analytic_solution, numeric_solutions):
        fig, (wave_ax, uwave_ax) = plt.subplots(1, 2, figsize=(12, 6))
        for numeric_solution in numeric_solutions:
            wave_ax.plot(
                numeric_solution.r_grid,
                numeric_solution.wave_function,
                label=f"$E$ = {numeric_solution.energy / constants.RY:6.2f}",
            )
            uwave_ax.plot(
                numeric_solution.r_grid,
                numeric_solution.uwave_function,
                label=f"$E$ = {numeric_solution.energy / constants.RY:6.2f}",
            )
        uwave_ax.plot(
            analytic_solution.r_grid,
            analytic_solution.uwave_function,
            "--",
            c="black",
            label=f"Analytic",
        )
        wave_ax.plot(
            analytic_solution.r_grid,
            analytic_solution.wave_function,
            "--",
            c="black",
            label=f"Analytic",
        )
        for ax in (wave_ax, uwave_ax):
            ax.legend()
            ax.set_xlabel(f"$r$ [fm]")
            ax.set_ylabel(f"$u$")
            ax.set_xlim(0.0, analytic_solution.r_grid[-1])
            ax.legend()
            ax.grid(True)
        self.plot_file.savefig()

    def _get_r_grid(self, rmin, rmax, n_grid_points):
        return np.linspace(rmin, rmax, num=n_grid_points, endpoint=True)

    def _get_numeric_solutions(self, r_grid, angular_momenta, energies):
        solutions = []
        for energy in energies:
            self._log(f"E={energy / constants.RY} Ry")
            solutions.append(
                numerov.numerov_wf(
                    energy,
                    angular_momenta,
                    potentials.get_coulomb_potential(
                        constants.Z * constants.HBARC * constants.ALPHA_FS
                    ),
                    r_grid,
                    mass_a=constants.N_NUCL,
                    mass_b=constants.M_PION,
                )
            )
        return solutions

    def _get_analytic_solution(self, r_grid):
        uwave_exact = (
            lambda r: (constants.Z / constants.A_BHOR) ** (3 / 2)
            * 2
            * np.exp(-constants.Z * r / constants.A_BHOR)
            * r
        )
        wave_exact = (
            lambda r: (constants.Z / constants.A_BHOR) ** (3 / 2)
            * 2
            * np.exp(-constants.Z * r / constants.A_BHOR)
            * r
        )
        return solution.Solution(
            energy=0.0,
            angular_momentum=0,
            wave_function=solution.normalize(
                solution.add_spherical_harmonic(
                    np.array([wave_exact(r) for r in r_grid]), l=0, m=0
                ),
                r_grid,
            ),
            uwave_function=solution.normalize(
                np.array([uwave_exact(r) for r in r_grid]), r_grid
            ),
            r_grid=r_grid,
            steps=len(r_grid),
        )


class PointNucleusFindBoundState(Task):
    def __init__(
        self, rmin, max_radii, energy_min, energy_max, energy_step, angular_momenta, **kwargs
    ):
        super().__init__(**kwargs)
        if energy_min >= energy_max:
            raise ValueError("energy_min must be less than energy_max")
        self.rmin = rmin
        self.max_radii = max_radii
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_step = energy_step
        self.angular_momenta = angular_momenta

    def run(self, output_dir):
        self._open_output_files(pathlib.Path(output_dir))
        self._log(f"Start")
        l = 0
        potential = potentials.get_coulomb_potential(
            constants.Z * constants.HBARC * constants.ALPHA_FS
        )

        numbers_of_steps = [10 ** k for k in range(2, 6)]
        steps_and_max_radius_to_bound_state = self._find_bounded_states(
            potential=potential,
            max_radii=self.max_radii,
            numbers_of_steps=numbers_of_steps,
        )
        self._plot(steps_and_max_radius_to_bound_state, self.max_radii, numbers_of_steps)
        self._close_output_files()

    def _plot(
        self, steps_and_max_radius_to_bound_state, max_radiuses, numbers_of_steps
    ):
        for max_radius in max_radiuses:
            bounded_states = [
                steps_and_max_radius_to_bound_state[(steps, max_radius)]
                for steps in numbers_of_steps
            ]
            plt.loglog(
                [bounded_state.steps for bounded_state in bounded_states],
                [bounded_state.error for bounded_state in bounded_states],
                "-.s",
                label=f"R={max_radius / constants.A_BHOR} $a_B$",
            )
        #
        # plt.xlabel(f"$N$")
        # plt.ylabel(f"$\eta$")
        # plt.xlim(numbers_of_steps[0], numbers_of_steps[-1])
        # plt.ylim(1.0e-8, 0.1)
        # plt.legend()
        # plt.grid(True)
        # self.plot_file.savefig()
        # plt.close()
        # # plot eta vs rmax
        # plt.semilogy(
        #     max_radiuses / constants.A_BHOR, Err_nr[-1, :], "-s", label=f"N=$10^5$"
        # )
        # plt.xlabel(f"$R [a_B]$")
        # plt.ylabel(f"$\eta$")
        # plt.xlim(0.0, max_radiuses[-1] / constants.A_BHOR)
        # plt.legend()
        # plt.grid(True)
        self.plot_file.savefig()
        plt.close()

    def _find_bounded_states(self, max_radii, numbers_of_steps, potential):
        steps_and_max_radius_to_bound_state = {}
        for max_radius, number_of_steps in itertools.product(
            max_radii, numbers_of_steps
        ):
            r_grid = np.linspace(self.rmin, max_radius, num=number_of_steps, endpoint=True)

            solution = numerov.find_bound_state(
                mass_a=constants.N_NUCL,
                mass_b=constants.M_PION,
                energy_min=self.energy_min,
                energy_max=self.energy_max,
                energy_step=self.energy_step,
                angular_momentum=self.angular_momenta,
                potential=potential,
                r_grid=r_grid,
            )
            steps_and_max_radius_to_bound_state[
                (number_of_steps, max_radius)
            ] = solution

            self._log(
                f"Energy level #{1} R={max_radius:6.1f}  N={number_of_steps:7d} E [MeV] = {solution.energy:.6E}"
                + f"   Validation: 1-E/(-Ry/n^2) = {solution.error:.6E}"
            )
        return steps_and_max_radius_to_bound_state


class Task3(Task):
    def run(self, output_dir):
        self._open_output_files(pathlib.Path(output_dir))
        self._log(f"Start")

        potential = potentials.get_coulomb_potential

        nmax = 4
        lmax = 2
        ngrid = 20000
        # n = radial excitation
        # l = orbital mometum
        for l in range(0, lmax + 1):
            Esteps = np.zeros(nmax - l + 1)

            ## COMPLETE ##

            ## define Esteps that bracket the energy roots

            ## COMPLETE ##

            for n in range(1, nmax - l + 1):
                Emin = Esteps[n - 1]
                Emax = Esteps[n]
                rmin = 0
                rmax = (n + l) * 20 * constants.A_BHOR
                r_grid = np.linspace(rmin, rmax, num=ngrid, endpoint=True)

                ## COMPLETE ##
                Ep = 0.0
                wf = 0 * r_grid
                ## Ep, wf = FindBoundState(...) ##

                ## COMPLETE ##

                radius = self._get_rms_radius(r_grid, wf)
                umax = wf[-1]
                Error = np.abs(1 - Ep / (-constants.RY / (n + l) ** 2))
                self._log(
                    f"  n={n:2d} l={l:2d}   E [MeV] = {Ep:.4E}  radius [fm] = {radius:7.3f}"
                    + f"  radius [a_B] = {radius / constants.A_BHOR:7.4f}   u(rmax) = {umax:9.2E}   |1-E/(-Ry/n^2)| = {Error:.3E}"
                )

        self._close_output_files()

    @staticmethod
    def _get_rms_radius(self, r_grid, u):
        ## COMPLETE ##
        radius = 0.0
        ## COMPLETE ##

        return radius


class Task4(Task):
    def run(self, output_dir):
        self._open_output_files(pathlib.Path(output_dir))
        self._log(f"Start")

        potential_p = potentials.get_coulomb_potential
        potential_s = potentials.SmearedCoulomb

        nmax = 4
        lmax = 2
        ngrid = 40000
        # n = radial excitation
        # l = orbital mometum
        self._log(f"\n Units MeV, fm")
        for l in range(0, lmax + 1):
            Esteps = np.zeros(nmax - l + 1)
            ## COMPLETE ##

            ## define Esteps that bracket the energy roots

            ## COMPLETE ##

            for n in range(1, nmax - l + 1):
                Emin = Esteps[n - 1]
                Emax = Esteps[n]
                rmin = 0
                rmax = (n + l) * 20 * constants.A_BHOR
                r_grid = np.linspace(rmin, rmax, num=ngrid, endpoint=True)

                ## COMPLETE ##

                Ep = 1.0
                Es = 2.0
                dE_perturb = 0.0
                wfp = 0 * r_grid

                ## Ep, wfp = FindBoundState(potential_p,...) ##
                ## dE_perturb = energy_shift_perutrbation(r_grid,wfp)
                ## Es, wfs = FindBoundState(potential_s,...) ##

                ## COMPLETE ##

                umax = wfp[-1]
                Error = np.abs(1 - dE_perturb / (Es - Ep))
                self._log(
                    f"  n={n:2d} l={l:2d}  Ep = {Ep:.6E}  Es = {Es:.6e}"
                    + f"  dE_exct ={Es - Ep:9.2e}  dE_prtb ={dE_perturb:9.2e}"
                    + f"  1-dE/E = {(Es - Ep) / Ep:.2e}  |1-dE_prtb/dE_exct| = {Error:.2e}"
                )

        self._close_output_files()


class Task5(Task):
    def run(self, output_dir):
        self._open_output_files(pathlib.Path(output_dir))
        self._log(f"Start")

        potential = potentials.get_coulomb_potential

        nmax = 4
        lmax = 2
        ngrid = 200000
        # n = radial excitation
        # l = orbital mometum
        self._log(f"\n Units MeV, fm")
        for l in range(0, lmax + 1):
            Esteps = np.zeros(nmax - l + 1)
            ## COMPLETE ##

            ## define Esteps that bracket the energy roots

            ## COMPLETE ##

            for n in range(1, nmax - l + 1):
                Emin = Esteps[n - 1]
                Emax = Esteps[n]
                rmin = 0
                rmax = (n + l) * 25 * constants.A_BHOR
                r_grid = np.linspace(rmin, rmax, num=ngrid, endpoint=True)

                ## COMPLETE ##
                Enr = 1
                Ekg = 1
                ## Enr,wfnr = FindBoundState(...)
                ## Ekg,wfkg = FindBoundStateKG(...)
                ## COMPLETE ##

                diff = 1 - Ekg / Enr
                self._log(
                    f"  n={n:2d} l={l:2d}  E_NR = {Enr:.6E}  E_KG = {Ekg:.6e}"
                    + f"  E_NR/Ry = {Enr / constants.RY:.6e}  E_KG/Ry = {Ekg / RY:.6e}  |1-E_KG/E_NR| = {diff:.3e}"
                )

        self._close_output_files()
