import numpy as np
import os, sys

sys.path.insert(0, os.path.join('..', 'utils'))
from helpers import *

x = np.random.randint(0, 10, (5, 2))

x[:] = True

a1, a2 = np.nonzero(x)
print np.array([a1, a2])


# xpts, _ = get_pts(x.shape)
# print xpts
# boo = np.sum(xpts, axis =1) > 1
# print boo

# xf = x.flatten()
# print xf

# print xf[boo]

# x = [  3.62000000e+02,   1.90300000e+03,  -1.71733800e+06]
# y = [  1.87200000e+03,   1.36175000e+03,   1.00000000e+00]

# x = [  1.33000000e+02,  -1.95900000e+03,   4.01432200e+06]
# y = [  1.87200000e+03,   1.36175000e+03,   1.00000000e+00]

# print x, y
# print np.dot(x, y)
# print np.logical_and(True, False)
