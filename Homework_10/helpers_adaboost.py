import cv2
import numpy as np
from os.path import join, basename, dirname, isfile, isdir
import sys, time, copy
from glob import glob
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def read_dataset(data_path):
	pos_img_paths = glob(join(join(data_path, 'positive'), '*.png'))
	neg_img_paths = glob(join(join(data_path, 'negative'), '*.png'))
	print len(pos_img_paths)
	return None, None


## Training data
X, Y = read_dataset('datasets\\dataset1\\train')
print X.shape
print Y.shape

## Testing data
Xtest, Ytest = read_dataset('datasets\\dataset1\\test\\positive')
