import numpy as np

a = np.zeros((3,3))
b = np.empty((3,1))

print np.append(b, a, axis = 1)
