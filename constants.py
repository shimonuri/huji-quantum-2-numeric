# Constants
import scipy.constants

# hbar*c, fine structure constant
HBARC = 197.3269804  # MeV fm
ALPHA_FS = scipy.constants.fine_structure
# Nucleus
A = 12
Z = 6

# Masses in energy units [MeV]
N_NUCL = A * 931.49432
M_PION = 139.57039
M_RED = N_NUCL * M_PION / (N_NUCL + M_PION)

# typical length = bhor radius
A_BHOR = HBARC / (Z * ALPHA_FS * M_RED)

# Rydberg energy
RY = 0.5 * Z ** 2 * ALPHA_FS ** 2 * M_RED

# Nuclear Radius [fm]
R_NUCL = A ** (1.0 / 3.0) * 1.2  # fm

DUMMY = 1
