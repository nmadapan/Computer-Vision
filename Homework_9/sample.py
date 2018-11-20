import numpy as np
from helpers import *

from scipy.optimize import least_squares

P = np.zeros((3, 4))
P[:, :3] = np.eye(3)

print np.linalg.pinv(P)
