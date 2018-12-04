import cv2
import numpy as np
from os.path import join, basename, dirname, isfile, isdir
import sys, time, copy
from glob import glob
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def read_dataset(data_path):
	'''
	Description:
		Read all the images and vectorize them.
		NOTE: All images should be of same size.
		NOTE: Images are read as gray scale images by default.
	Input:
		* data_path: absolute path to a directory containing images
	Return:
		* X: 2D np.ndarray (dimension x num_images)
		* Y: 1D np.ndarray (num_images, )
	'''
	## Find absolute paths to images
	img_paths = glob(join(data_path, '*.png'))
	num_images = len(img_paths)

	## Find the image size or dimension
	frame = cv2.imread(img_paths[0], 0)
	dim = frame.size

	## Find class labels
	# Contains actual class labels
	class_labels = list(set([int(float(basename(img_path).split('_')[0])) for img_path in img_paths]))
	num_classes = len(class_labels)

	## Initialize X and Y
	X = np.zeros((dim, num_images))
	# Class ids start from 0.
	Y = np.zeros(num_images, dtype = np.int16)

	for idx, img_path in enumerate(img_paths):
		## Read grayscale image
		frame = cv2.imread(img_path, 0)
		## Find the class ID
		subj_id = int(float(basename(img_path).split('_')[0]))
		class_id = class_labels.index(subj_id)
		## Vectorize the image and attach a class id
		X[:, idx] = frame.flatten()
		Y[idx] = class_id

	return X, Y

def mean_normalize(X, mean = None, std = None, axis = 0):
	'''
	Description:
		Mean normalize along the rows/columns determined the 'axis'
	Input arguments:
		* X: 2D np.ndarray of size (num_instances x dimension)
	Return:
		* Mean normalized X
	'''
	if(mean is None): mean = np.mean(X, axis = axis)
	if(std is None): std = np.std(X, axis = axis)
	if(axis == 0):
		return (X - mean)/std
	else:
		return ((X.T - mean)/std).T

class PCA:
	def __init__(self, X, K = None):
		self.X = X
		## Thresholding no. of eigen vectors
		self.K = K
		## Eigen vectors
		self.E = None # Updated by call to compute_eigen
		## PCA mean computed on the dataset
		self.mean = None # Updated by call to compute_eigen

	def compute_eigen(self):
		## Compute X^T * X. Eigen values of X * X^T and X^T * X are same.
		## However, Eigen vectors need to mapped using X.
		self.mean = np.mean(self.X, axis = 1)
		X = (self.X.T - self.mean).T
		XtX = np.dot(X.T, X)
		## svd function returs V^T. So take a transpose to get V.
		## Columns of V are eigen vectors
		_, S, Vt = np.linalg.svd(XtX, full_matrices = False)
		V = Vt.T
		## Map eigen vectors of X^T * X to eigen vectors of X * X^T
		self.E = np.dot(self.X, V)
		## Normalize the transformed eigen vectors
		self.E = self.E / np.linalg.norm(self.E, axis = 0)

	def dim_red(self, Z, K = None):
		## Z: dimension x num_instances
		if(Z.ndim == 1): Z = Z.reshape(-1, 1)
		# Mean Normalize each column
		# Z = mean_normalize(Z, axis = 0)
		## Substract PCA mean from each column
		Z = (Z.T - self.mean).T
		if(K is None):
			return np.dot(self.E[:,:self.K].T, Z)
		else:
			return np.dot(self.E[:,:K].T, Z)

	def test(self, Xtr, Ytr, Xts, Yts, K, display = False):
		X_r = self.dim_red(Xtr, K = K)
		Xts_r = self.dim_red(Xts, K = K)

		## Compute confusion matrix
		test_pred_label = []
		test_true_label = []
		## Creating test data
		for idx, test_inst in enumerate(Xts_r.T):
			pred_label = knn(X_r.T, Ytr, test_inst, K = 3)
			test_pred_label.append(pred_label)
			test_true_label.append(Yts[idx])

		acc = np.mean(np.array(test_pred_label) == np.array(test_true_label))
		print 'Accuracy: %.02f'%acc

		if(display):
			conf_mat = confusion_matrix(test_true_label, test_pred_label)
			plot_confusion_matrix(conf_mat, range(30), normalize = True, title = 'PCA Confusion Matrix')

		return acc

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

	argmin_ids = np.argsort(dist)[:K]
	c_argmin_ids = train_output[argmin_ids]

	all_class_ids = np.unique(train_output).astype(int)
	freqs = [0]*len(all_class_ids)
	for cid in c_argmin_ids:
		freqs[cid] += 1
	return np.argmax(freqs)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = 100 * (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
		cm = cm.astype('int')
	else:
		print('Confusion matrix, without normalization')

	# print(cm)
	plt.figure()

	np.set_printoptions(precision=0)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = 'd' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", \
			color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

class LDA:
	def __init__(self, X, Y):
		## Y = [0, 0, ..., i, i, ..., K-1, K-1] for K classes
		## X: dimension x num_instances
		self.X = X
		self.Y = Y
		self.num_classes = np.max(self.Y) + 1
		self.dim = self.X.shape[0]
		self.data = self.reorganize()
		self.mean = None ## Overall mean. Updated by bet_class_scatter
		self.E = None ## LDA Vectors. Updated by within_class_scatter

	def reorganize(self):
		data = {}
		for class_id in range(self.num_classes):
			data[class_id] = self.X[:, self.Y == class_id]
		return data

	def bet_class_scatter(self):
		Sb = []
		means = []
		for class_id in range(self.num_classes):
			Xi = self.data[class_id]
			means.append(np.linalg.norm(Xi, axis = 1))
		overall_mean = np.linalg.norm(self.X, axis = 1)
		self.mean = overall_mean

		means = (np.array(means) - overall_mean).T

		Sb = np.dot(means.T, means)

		_, D, Vt = np.linalg.svd(Sb, full_matrices = False)
		V = Vt.T
		## Map eigen vectors of X^T * X to eigen vectors of X * X^T
		V = np.dot(means, V)
		## Normalize the transformed eigen vectors
		V = V / np.linalg.norm(V, axis = 0)
		return V

	def within_class_scatter(self):
		Z = self.bet_class_scatter()
		NX = np.zeros((self.dim,1))
		for class_id in range(self.num_classes):
			Xi = self.data[class_id]
			mean = np.linalg.norm(Xi, axis = 1)
			Xi = (Xi.T - mean).T
			Xi = Xi / np.sqrt(Xi.shape[1])
			NX = np.append(NX, Xi, axis = 1)
		NX = NX[:, 1:]
		NX = NX / np.sqrt(self.num_classes)

		NXZ = np.dot(Z.T, NX)

		XXt = np.dot(NXZ, NXZ.T)
		_, D, Vt = np.linalg.svd(XXt, full_matrices = False)
		V = Vt.T

		W = np.dot(Z, V)
		W = W / np.linalg.norm(W, 1)
		self.E = W

		return self.E

	def compute_eigen(self):
		self.within_class_scatter()

	def dim_red(self, Z, K = None):
		## Z: dimension x num_instances
		if(Z.ndim == 1): Z = Z.reshape(-1, 1)
		# Mean Normalize each column
		# Z = mean_normalize(Z, axis = 0)
		## Substract PCA mean from each column
		Z = (Z.T - self.mean).T
		if(K is None):
			return np.dot(self.E.T, Z)
		else:
			return np.dot(self.E[:,:K].T, Z)

	def test(self, Xtr, Ytr, Xts, Yts, K, display = False):
		X_r = self.dim_red(Xtr, K = K)
		Xts_r = self.dim_red(Xts, K = K)

		## Compute confusion matrix
		test_pred_label = []
		test_true_label = []
		## Creating test data
		for idx, test_inst in enumerate(Xts_r.T):
			pred_label = knn(X_r.T, Ytr, test_inst, K = 3)
			test_pred_label.append(pred_label)
			test_true_label.append(Yts[idx])

		acc = np.mean(np.array(test_pred_label) == np.array(test_true_label))
		print 'Accuracy: %.02f'%acc

		if(display):
			conf_mat = confusion_matrix(test_true_label, test_pred_label)
			plot_confusion_matrix(conf_mat, range(30), normalize = True, title = 'PCA Confusion Matrix')

		return acc

