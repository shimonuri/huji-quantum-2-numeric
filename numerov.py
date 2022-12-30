import numpy as np
import constants


def numerov_wf(E, l, potential, r_grid):
    W = np.zeros(len(r_grid))
    u = np.zeros(len(r_grid))

    ##  COMPLETE  ##

    ##  COMPLETE  ##

    u_norm = normalization(u, r_grid)

    return u_norm


# Solution to the Klein-Gordon w.f.
def numerov_kgwf(E, l, potential, r_grid):
    W = np.zeros(len(r_grid))
    u = np.zeros(len(r_grid))

    ##  COMPLETE  ##

    ##  COMPLETE  ##

    u_norm = normalization(u, r_grid)

    return u_norm


def normalization(u, r_grid):
    ##  COMPLETE  ##
    u_norm = constants.DUMMY * u
    ##  COMPLETE  ##

    return u_norm
