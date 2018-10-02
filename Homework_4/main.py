import numpy as np

from utils import *

import cv2

# print cv2.getGaussianKernel(5, 1)
# harris('', 1)

x = np.random.randint(0, 10, (3, 3))
y = np.random.randint(0, 10, (3, 3))
print x
print y

rows, cols = np.nonzero(x)
kps = zip(rows.tolist(), cols.tolist())
print np.array(kps)

# gam = 0.1
# val = gam / ( (1+gam)* (1+gam))
# print val

# gam = 0.5
# val = gam / ( (1+gam)* (1+gam))
# print val

# for aidx, idx in enumerate(range(3,10)):
# 	print aidx, idx