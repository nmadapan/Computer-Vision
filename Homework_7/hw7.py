import os, sys, time
import numpy as np
import cv2
from os.path import join, basename, dirname, splitext
from copy import deepcopy
from BitVector import BitVector
from glob import glob

import pickle

sys.path.append('..\\utils')
from helpers import *

# binvec = [0, 1, 0, 0, 0, 1, 1, 0]

# bv = BitVector(bitlist = binvec)

# for _ in range(100000):
#     intvals =  [int(bv<<1) for _ in range(len(binvec))]
#     minbv = BitVector(intVal = min(intvals), size = len(binvec))

# # print minbv

# # bvruns = minbv.runs()
# # print len(bvruns)

def create_circ_points(radius = 1, num_neighbors = 8):
    R = radius
    P = num_neighbors

    x_lst = []
    y_lst = []
    for p in range(P):
        du = R*np.cos(2*np.pi*p/P)
        dv = R*np.sin(2*np.pi*p/P)
        if(abs(du) < 1e-4): du = 0.0
        if(abs(dv) < 1e-4): dv = 0.0
        y_lst.append(du)
        x_lst.append(dv)
    return x_lst, y_lst

def bilin_interp(A, B, C, D, dx, dy):
    # print A, B, C, D
    # dx is x-axis distance from A to the point
    # dy is y-axis distance from A to the point
    return (1-dy)*(1-dx)*A + (1-dy)*dx*B + dy*(1-dx)*C + dy*dx*D

def lbp_binvec(A):
    '''
    Input:
        * A: np.ndarray of size 3 x 3
    '''
    v1 = A[2, 1]
    v2 = bilin_interp(A[1,1], A[1,2], A[2,1], A[2,2], 0.707, 0.707)
    v3 = A[1, 2]
    v4 = bilin_interp(A[1,1], A[1,2], A[0,1], A[0,2], 0.707, 0.707)
    v5 = A[0, 1]
    v6 = bilin_interp(A[1,1], A[1,0], A[0,1], A[0,0], 0.707, 0.707)
    v7 = A[1,0]
    v8 = bilin_interp(A[1,1], A[1,0], A[2,1], A[2,0], 0.707, 0.707)
    ret = np.array([v1, v2, v3, v4, v5, v6, v7, v8])

    ret = ret >= A[1,1]

    return ret.astype(int).tolist()

def lbp_value(binvec, P = 8):
    bv = BitVector(bitlist = binvec)
    intvals =  [int(bv<<1) for _ in range(len(binvec))]
    minbv = BitVector(intVal = min(intvals), size = len(binvec))
    bvruns = minbv.runs()
    if(len(bvruns) > 2): return P + 1
    elif(len(bvruns) == 1 and bvruns[0][0] == '1'): return P
    elif(len(bvruns) == 1 and bvruns[0][0] == '0'): return 0
    else: return len(bvruns[1])

def get_lbp_hist(img_path):
    img = cv2.imread(img_path, 0)
    hist = [0]*(P+2)
    for x_idx in range(1, img.shape[1]-1):
        for y_idx in range(1, img.shape[0]-1):
            frame = img[y_idx-1:y_idx+2, x_idx-1:x_idx+2]
            binvec = lbp_binvec(frame)
            pvalue = lbp_value(binvec)
            hist[pvalue] += 1

    hist = np.array(hist).astype(float) / np.sum(hist)
    return hist

def fake_get_lbp_hist(img_path):
    return np.random.randn(P+2)


R = 1
P = 8

# ## Creating training data
# print '=============== Creating Training Data ================='
# training_dir_path = 'Images\\training'
# classnames = os.listdir(training_dir_path)
# features = {cname: [] for cname in classnames}

# for cname in classnames:
#     print '---' + cname + '---'
#     img_dir = os.path.join(training_dir_path, cname)
#     img_paths = glob(os.path.join(img_dir, '*.jpg'))
#     for img_path in img_paths:
#         print os.path.basename(img_path),
#         hist = get_lbp_hist(img_path)
#         features[cname].append(hist)
#     print ''

#     features[cname] = np.array(features[cname])

# with open('train_features.pickle', 'wb') as fp:
#     pickle.dump(features, fp)

# with open('train_features.pickle', 'rb') as fp:
#     features = pickle.load(fp)


## Creating testing data
print '\n============ Creating Testing Data ============='
testing_dir_path = 'Images\\testing'
test_img_paths = glob(os.path.join(testing_dir_path, '*.jpg'))
out_features = {os.path.basename(test_img_path): None for test_img_path in test_img_paths}

for test_img_path in test_img_paths:
    print os.path.basename(test_img_path),
    out_feat_vec = get_lbp_hist(test_img_path)
    out_features[os.path.basename(test_img_path)] = out_feat_vec

with open('test_features.pickle', 'wb') as fp:
    pickle.dump(out_features, fp)

with open('test_features.pickle', 'rb') as fp:
    out_features = pickle.load(fp)


# x_disp, y_disp = create_circ_points()
# x_indices = np.int8(np.round(x_disp))
# y_indices = np.int8(np.round(y_disp))

# print x_indices
# print y_indices

# print x_disp
# print y_disp

# A = [[1, 5, 3], [5, 3, 1], [4, 0, 0]]
# A = np.array(A)

# # binvec = lbp_binvec(A)

# print lbp_value([1,0,0,1,1,1,1,1])
