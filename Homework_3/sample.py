import numpy as np
import sys, os
import cv2

ref = 255 * np.ones((256, 256, 3), dtype = np.uint8)
cv2.imshow('frame', ref)
cv2.waitKey(1000)

# sys.path.insert(0, os.path.join('..', 'utils'))
# from helpers import *

# x = np.random.randint(0,10,3)
# print x
# print nmlz(x)
