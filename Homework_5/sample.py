import cv2
import numpy as np
import time
from os.path import join, basename, splitext, dirname
import sys

sys.path.insert(0, r'..\utils')
from helpers import *

a = np.array([2,3])
x, y = a
print x, y

# a = np.array([[  9.603e-01,  -2.674e-02,   8.104e-01],
#  [ -2.175e-03,   1.006e+00,   1.032e-01],
#  [ -6.189e-06,   2.771e-05,   1.000e+00]])

# print a

# # ransac([[]]*400)

# a = np.random.randint(0, 10, 10)

# print a

# nz_ids = np.nonzero(a<4)[0]
# print a[nz_ids]

# print eval('3/1.5')

# N = '(h11*{0}+h12*{1}+h13)'
# D = '(h31*{0}+h32*{1}+h33)'

# def senc(value): return '('+str(value)+')'

# def jac_row(x, y, axis = 'x'):
#     d = [0]*9
#     if(axis == 'x'):
#         d[0] = '{0}'+'/'+D
#         d[1] = '{1}'+'/'+D
#         d[2] = '1'+'/'+D
#         d[3] = '0'
#         d[4] = '0'
#         d[5] = '0'
#         d[6] = '(-'+N+'*'+'{0})/' + '(' + D + '**2)'
#         d[7] = '(-'+N+'*'+'{1})/' + '(' + D + '**2)'
#         d[8] = '(-'+N+'*'+'1)/' + '(' + D + '**2)'
#     else:
#         d[0] = '0'
#         d[1] = '0'
#         d[2] = '0'
#         d[3] = '{0}'+'/'+D
#         d[4] = '{1}'+'/'+D
#         d[5] = '1'+'/'+D
#         d[6] = '(-'+N+'*'+'{0})/' + '(' + D + '**2)'
#         d[7] = '(-'+N+'*'+'{1})/' + '(' + D + '**2)'
#         d[8] = '(-'+N+'*'+'1)/' + '(' + D + '**2)'

#     for idx, _ in enumerate(d):
#         d[idx] = d[idx].format(x, y)

#     return d

# print jac_row(senc(1.2), senc(-3.7), axis = 'x')
