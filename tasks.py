from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit, brentq
from scipy.integrate import simps
from math import pi
import numpy as np
import constants
import potentials
import numerov


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
    def __init__(self, name):
        self.name = name

    def run(self):
        raise NotImplementedError("Task.run() is not implemented")


class Task1(Task):
    def run(self):
        pltfile = PdfPages("Task1.pdf")
        sikum = open("Task1.sikum", "w")

        line = f"\n\t Task 1"
        print(line)
        sikum.write(line)

        l = 0
        potential = potentials.PointCoulomb
        n_grid_points = 10001
        rmin = 0
        rmax = 10 * constants.A_BHOR
        r_grid = np.linspace(rmin, rmax, num=n_grid_points, endpoint=True)

        Evals = [-(0.9 + i * 0.05) * constants.RY for i in range(0, 5)]

        for Ep in Evals:
            print(f"\t Task 1: E={Ep / constants.RY} Ry")
            sikum.write(f"\n\t Task 1: E={Ep / constants.RY:6.2f} Ry")
            up = numerov.numerov_wf(Ep, l, potential, r_grid)
            plt.plot(r_grid, up, label=f"$E$ = {Ep / constants.RY:6.2f}")

            ## COMPLETE ##
            f_exact = r_grid
            ## COMPLETE ##
        plt.plot(r_grid, f_exact, "--", c="black", label=f"Analytic")

        plt.xlabel(f"$r$ [fm]")
        plt.ylabel(f"$u$")
        plt.xlim(0.0, r_grid[-1])
        plt.legend()
        plt.grid(True)
        pltfile.savefig()
        plt.close()

        pltfile.close()
        sikum.close()


class Task2(Task):
    def run(self):
        pltfile = PdfPages("Task2.pdf")
        sikum = open("Task2.sikum", "w")

        line = f"\n\t Task 2"
        print(line)
        sikum.write(line)

        l = 0
        potential = potentials.PointCoulomb
        n_grid_points = 10001

        ngrid_list = [10 ** k for k in range(2, 6)]
        rmax_list = np.array([5, 10, 15, 20]) * constants.A_BHOR

        Eval_nr = np.zeros((len(ngrid_list), len(rmax_list)))
        Err_nr = np.zeros((len(ngrid_list), len(rmax_list)))

        Emin = -1.1 * constants.RY
        Emax = -0.9 * constants.RY
        for jr in range(len(rmax_list)):
            for jn in range(len(ngrid_list)):
                rmax = rmax_list[jr]
                ngrid = ngrid_list[jn]

                r_grid = np.linspace(0.0, rmax, num=ngrid, endpoint=True)

                ## COMPLETE ##
                ## Ep,wf = FindBoundState(...) ##
                Ep = 0.0
                Error = 99999.0
                ## COMPLETE ##

                Eval_nr[jn, jr] = Ep
                Err_nr[jn, jr] = Error

                line = (
                    f"Energy level #{1} R={rmax:6.1f}  N={ngrid:7d} E [MeV] = {Ep:.6E}"
                    + f"   Validation: 1-E/(-Ry/n^2) = {Error:.6E}"
                )
                print(line)
                sikum.write("\n" + line)

        # plot eta vs N for different tmax
        for jr in range(len(rmax_list)):
            rmax = rmax_list[jr]
            plt.loglog(
                ngrid_list, Err_nr[:, jr], "-.s", label=f"R={rmax / A_BHOR} $a_B$"
            )
        plt.xlabel(f"$N$")
        plt.ylabel(f"$\eta$")
        plt.xlim(ngrid_list[0], ngrid_list[-1])
        plt.ylim(1.0e-8, 0.1)
        plt.legend()
        plt.grid(True)
        pltfile.savefig()
        plt.close()

        # plot eta vs rmax
        plt.semilogy(
            rmax_list / constants.A_BHOR, Err_nr[-1, :], "-s", label=f"N=$10^5$"
        )
        plt.xlabel(f"$R [a_B]$")
        plt.ylabel(f"$\eta$")
        plt.xlim(0.0, rmax_list[-1] / constants.A_BHOR)
        plt.legend()
        plt.grid(True)
        pltfile.savefig()
        plt.close()

        pltfile.close()
        sikum.close()


class Task3(Task):
    def run(self):
        sikum = open("Task3.sikum", "w")

        line = f"\n\t Task 3"
        print(line)
        sikum.write(line)

        potential = potentials.PointCoulomb

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
                line = (
                    f"  n={n:2d} l={l:2d}   E [MeV] = {Ep:.4E}  radius [fm] = {radius:7.3f}"
                    + f"  radius [a_B] = {radius / constants.A_BHOR:7.4f}   u(rmax) = {umax:9.2E}   |1-E/(-Ry/n^2)| = {Error:.3E}"
                )
                print(line)
                sikum.write("\n" + line)

        sikum.close()

    @staticmethod
    def _get_rms_radius(self, r_grid, u):
        ## COMPLETE ##
        radius = 0.0
        ## COMPLETE ##

        return radius


class Task4(Task):
    def run(self):
        sikum = open("Task4.sikum", "w")

        line = f"\n\t Task 4"
        print(line)
        sikum.write(line)

        potential_p = potentials.PointCoulomb
        potential_s = potentials.SmearedCoulomb

        nmax = 4
        lmax = 2
        ngrid = 40000
        # n = radial excitation
        # l = orbital mometum
        sikum.write(f"\n Units MeV, fm")
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
                line = (
                    f"  n={n:2d} l={l:2d}  Ep = {Ep:.6E}  Es = {Es:.6e}"
                    + f"  dE_exct ={Es - Ep:9.2e}  dE_prtb ={dE_perturb:9.2e}"
                    + f"  1-dE/E = {(Es - Ep) / Ep:.2e}  |1-dE_prtb/dE_exct| = {Error:.2e}"
                )
                print(line)
                sikum.write("\n" + line)

        sikum.close()


class Task5(Task):
    def run(self):
        sikum = open("Task5.sikum", "w")

        line = f"\n\t Task 5"
        print(line)
        sikum.write(line)

        potential = potentials.PointCoulomb

        nmax = 4
        lmax = 2
        ngrid = 200000
        # n = radial excitation
        # l = orbital mometum
        sikum.write(f"\n Units MeV, fm")
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
                line = (
                    f"  n={n:2d} l={l:2d}  E_NR = {Enr:.6E}  E_KG = {Ekg:.6e}"
                    + f"  E_NR/Ry = {Enr / constants.RY:.6e}  E_KG/Ry = {Ekg / RY:.6e}  |1-E_KG/E_NR| = {diff:.3e}"
                )
                print(line)
                sikum.write("\n" + line)

        sikum.close()

