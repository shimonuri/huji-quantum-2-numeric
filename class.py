from scipy import constants as constants
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit, brentq
from scipy.integrate import simps
from math import pi
import numpy as np

# =============================================================================
# Constants
# hbar*c, fine structure constant
hbarc = 197.3269804  # MeV fm
alpha_fs = constants.fine_structure

# Nucleus
A = 12
Z = 6

# Masses in energy units [MeV]
m_nucl = A * 931.49432
m_pion = 139.57039
m_red = m_nucl * m_pion / (m_nucl + m_pion)

# typical length = bhor radius
a_bhor = hbarc / (Z * alpha_fs * m_red)

# Rydberg energy
Ry = 0.5 * Z ** 2 * alpha_fs ** 2 * m_red

# Nuclear Radius [fm]
R_nucl = A ** (1.0 / 3.0) * 1.2  # fm

dummy = 1


# ================================ POTENTIALS ================================================
def PointCoulomb(r):
    ##  COMPLETE  ##
    vv = dummy
    ##  COMPLETE  ##

    return vv


# -----------------------------------------------------------------------------
def SmearedCoulomb(r):
    ##  COMPLETE  ##
    vv = dummy
    ##  COMPLETE  ##

    return vv


# =============================================================================================
def normalization(u, r_grid):
    ##  COMPLETE  ##
    u_norm = dummy * u
    ##  COMPLETE  ##

    return u_norm


# ==============================================================================================
def energy_shift_perutrbation(r_grid, u):
    ##  COMPLETE  ##
    e_shift = dummy
    ##  COMPLETE  ##

    return e_shift


# ==============================================================================================
def NumerovWF(E, l, potential, r_grid):
    W = np.zeros(len(r_grid))
    u = np.zeros(len(r_grid))

    ##  COMPLETE  ##

    ##  COMPLETE  ##

    u_norm = normalization(u, r_grid)

    return u_norm


# ==============================================================================================
# Solution to the Klein-Gordon w.f.
def NumerovKGWF(E, l, potential, r_grid):
    W = np.zeros(len(r_grid))
    u = np.zeros(len(r_grid))

    ##  COMPLETE  ##

    ##  COMPLETE  ##

    u_norm = normalization(u, r_grid)

    return u_norm


# ===============================================================================================
def FindBoundState(potential, l, Emin, Emax, r_grid):
    # locate the exact binding energy in the range (Emin,Emax)
    # return:
    #         E - energy,
    #         u[0:len(r_grid)] - wave function

    ## COMPLETE ##
    ## COMPLETE ##
    ## COMPLETE ##

    return (E, u)


# =============================================================================
def rms_radius(r_grid, u):
    ## COMPLETE ##
    radius = 0.0
    ## COMPLETE ##

    return radius


# =============================================================================
def Task1():
    pltfile = PdfPages("Task1.pdf")
    sikum = open("Task1.sikum", "w")

    line = f"\n\t Task 1"
    print(line)
    sikum.write(line)

    l = 0
    potential = PointCoulomb
    n_grid_points = 10001
    rmin = 0
    rmax = 10 * a_bhor
    r_grid = np.linspace(rmin, rmax, num=n_grid_points, endpoint=True)

    Evals = [-(0.9 + i * 0.05) * Ry for i in range(0, 5)]

    for Ep in Evals:
        print(f"\t Task 1: E={Ep / Ry} Ry")
        sikum.write(f"\n\t Task 1: E={Ep / Ry:6.2f} Ry")
        up = NumerovWF(Ep, l, potential, r_grid)
        plt.plot(r_grid, up, label=f"$E$ = {Ep / Ry:6.2f}")

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


# =============================================================================
def Task2():
    pltfile = PdfPages("Task2.pdf")
    sikum = open("Task2.sikum", "w")

    line = f"\n\t Task 2"
    print(line)
    sikum.write(line)

    l = 0
    potential = PointCoulomb
    n_grid_points = 10001

    ngrid_list = [10 ** k for k in range(2, 6)]
    rmax_list = np.array([5, 10, 15, 20]) * a_bhor

    Eval_nr = np.zeros((len(ngrid_list), len(rmax_list)))
    Err_nr = np.zeros((len(ngrid_list), len(rmax_list)))

    Emin = -1.1 * Ry
    Emax = -0.9 * Ry
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
        plt.loglog(ngrid_list, Err_nr[:, jr], "-.s", label=f"R={rmax / a_bhor} $a_B$")
    plt.xlabel(f"$N$")
    plt.ylabel(f"$\eta$")
    plt.xlim(ngrid_list[0], ngrid_list[-1])
    plt.ylim(1.0e-8, 0.1)
    plt.legend()
    plt.grid(True)
    pltfile.savefig()
    plt.close()

    # plot eta vs rmax
    plt.semilogy(rmax_list / a_bhor, Err_nr[-1, :], "-s", label=f"N=$10^5$")
    plt.xlabel(f"$R [a_B]$")
    plt.ylabel(f"$\eta$")
    plt.xlim(0.0, rmax_list[-1] / a_bhor)
    plt.legend()
    plt.grid(True)
    pltfile.savefig()
    plt.close()

    pltfile.close()
    sikum.close()


# =============================================================================
def Task3():
    sikum = open("Task3.sikum", "w")

    line = f"\n\t Task 3"
    print(line)
    sikum.write(line)

    potential = PointCoulomb

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
            rmax = (n + l) * 20 * a_bhor
            r_grid = np.linspace(rmin, rmax, num=ngrid, endpoint=True)

            ## COMPLETE ##
            Ep = 0.0
            wf = 0 * r_grid
            ## Ep, wf = FindBoundState(...) ##

            ## COMPLETE ##

            radius = rms_radius(r_grid, wf)
            umax = wf[-1]
            Error = np.abs(1 - Ep / (-Ry / (n + l) ** 2))
            line = (
                f"  n={n:2d} l={l:2d}   E [MeV] = {Ep:.4E}  radius [fm] = {radius:7.3f}"
                + f"  radius [a_B] = {radius / a_bhor:7.4f}   u(rmax) = {umax:9.2E}   |1-E/(-Ry/n^2)| = {Error:.3E}"
            )
            print(line)
            sikum.write("\n" + line)

    sikum.close()


# =============================================================================
def Task4():
    sikum = open("Task4.sikum", "w")

    line = f"\n\t Task 4"
    print(line)
    sikum.write(line)

    potential_p = PointCoulomb
    potential_s = SmearedCoulomb

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
            rmax = (n + l) * 20 * a_bhor
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


# =============================================================================
def Task5():
    sikum = open("Task5.sikum", "w")

    line = f"\n\t Task 5"
    print(line)
    sikum.write(line)

    potential = PointCoulomb

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
            rmax = (n + l) * 25 * a_bhor
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
                + f"  E_NR/Ry = {Enr / Ry:.6e}  E_KG/Ry = {Ekg / Ry:.6e}  |1-E_KG/E_NR| = {diff:.3e}"
            )
            print(line)
            sikum.write("\n" + line)

    sikum.close()


# =============================================================================
#####################################################
#                                                   #
# NUMEROV ALGORITHM FOR BOUND STATES                #
#                                                   #
#####################################################

print(f"\n\t Numerov Solver for pi-12C system:")
line = f"\n\t nucleus mass= {m_nucl} MeV \n\t pion    mass= {m_pion} MeV"
line = line + f"\n\t reduced mass= {m_red} MeV\n"
print(line)

Task1()
Task2()
Task3()
Task4()
Task5()
# =============================================================================
