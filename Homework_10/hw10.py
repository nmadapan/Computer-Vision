import cv2
import numpy as np
from os.path import join, basename, dirname, isfile, isdir
import sys, time, copy
from helpers import *
from matplotlib import pyplot as plt

## Training data
X, Y = read_dataset('datasets\\dataset2\\train')

## Testing data
Xtest, Ytest = read_dataset('datasets\\dataset2\\test')

###################
####### PCA #######
###################
pca = PCA(X, K = 10)
pca.compute_eigen()
acc_list = []
st = time.time()
for K in range(30):
	acc = pca.test(X, Y, Xtest, Ytest, K = K)
	acc_list.append(acc)
print 'Time taken: %.04f secs'%(time.time()-st)
plt.plot(acc_list)
plt.title('PCA: Classification Accuracy versus K')
plt.ylabel('Accuracy in %')
plt.xlabel('No. of Eigen vectors (K)')
plt.grid('on')
plt.show()

###################
####### LDA #######
###################
da = LDA(X, Y)
da.compute_eigen()
acc_list = []
st = time.time()
for K in range(29):
	acc = da.test(X, Y, Xtest, Ytest, K = K)
	acc_list.append(acc)
print 'Time taken: %.04f secs'%(time.time()-st)
plt.plot(acc_list)
plt.title('LDA: Classification Accuracy versus K')
plt.ylabel('Accuracy in %')
plt.xlabel('No. of Eigen vectors (K)')
plt.grid('on')
plt.show()
