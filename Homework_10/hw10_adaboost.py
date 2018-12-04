import cv2
import numpy as np
from os.path import join, basename, dirname, isfile, isdir
import sys, time, copy
from helpers import *
from matplotlib import pyplot as plt

## Training data
X, Y = read_dataset('datasets\\dataset1\\train\\positive')
print X.shape
pr

## Testing data
Xtest, Ytest = read_dataset('datasets\\dataset1\\test\\positive')
