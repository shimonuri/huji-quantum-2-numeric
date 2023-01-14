import constants
import numpy as np


def get_coulomb_potential(const):
    return lambda r: -const / r


# -----------------------------------------------------------------------------
def get_smeared_coulomb(density):
    def electric_change(r):
        if r < constants.R_NUCL:
            return 4 / 3 * np.pi * density * r ** 3
        else:
            return 4 / 3 * np.pi * density * constants.R_NUCL ** 3

    return lambda r: -electric_change(r) / r
