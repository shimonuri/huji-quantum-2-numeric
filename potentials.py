import constants
import numpy as np


def get_coulomb_potential(const):
    return lambda r: -const / r


# -----------------------------------------------------------------------------
def get_smeared_coulomb(density):
    return lambda r: -((4 / 3) * np.pi * r ** 3) * density / r
