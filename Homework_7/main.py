import pickle
import numpy as np
import os, sys, time
from sklearn.metrics import confusion_matrix

sys.path.append('..\\utils')
from helpers import *

with open('train_features.pickle', 'rb') as fp:
    train_features = pickle.load(fp)

with open('test_features.pickle', 'rb') as fp:
    test_features = pickle.load(fp)

num_classes = len(train_features.keys())

cnames_to_ids = {cname: idx+1 for idx, cname in enumerate(train_features.keys())}
classnames = [cname for cname in train_features.keys()]

## Creating train data

train_input = None
train_output = None

for cname, data in train_features.items():
    cid = cnames_to_ids[cname]
    if(train_input is None): train_input = data
    else: train_input = np.append(train_input, data, axis = 0)
    out = cid * np.ones(data.shape[0])
    if(train_output is None): train_output = out
    else: train_output = np.append(train_output, out).astype(int)

def knn(train_input, train_output, test_inst, metric = 'euclidean', K = 5):
    # train_input: 2D np.ndarray
    # train_output: 1D np.ndarray. train instance labels.
    # test_inst: 1D np.ndarray. Size of vec is equal to no. of columns in M
    if(metric == 'euclidean'):
        dist = np.linalg.norm(train_input - test_inst, axis = 1)
    elif(metric == 'dot'):
        dist = -1 * np.sum(train_input * test_inst, axis = 1)
    elif(metric == 'cosine'):
        norm_train_input = train_input.transpose() / np.linalg.norm(train_input, axis = 1)
        norm_train_input = norm_train_input.transpose()
        norm_test_inst = test_inst / np.linalg.norm(test_inst)
        dist = 1 - np.sum(norm_train_input * norm_test_inst, axis = 1)

    argmin_ids = np.argsort(dist)[:5]
    c_argmin_ids = train_output[argmin_ids]

    all_class_ids = np.unique(train_output).astype(int)
    freqs = [0]*len(all_class_ids)
    for cid in c_argmin_ids:
        freqs[cid-1] += 1
    return np.argmax(freqs) + 1

test_pred_label = []
test_true_label = []

## Creating test data
for fname, test_inst in test_features.items():
    cname = os.path.splitext(fname)[0].split('_')[0]
    cid = cnames_to_ids[cname]

    ## Euclidean
    pred_label = knn(train_input, train_output, test_inst, metric = 'euclidean', K = 5)

    test_pred_label.append(pred_label)
    test_true_label.append(cid)

print zip(test_true_label, test_pred_label)

conf_mat = confusion_matrix(test_true_label, test_pred_label)

plot_confusion_matrix(conf_mat, classnames, normalize = True)
