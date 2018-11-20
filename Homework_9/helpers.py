import cv2
import numpy as np
import os, time, sys
# from NonlinearLeastSquares import NonlinearLeastSquares as NLS
import matplotlib.pyplot as plt
from os.path import basename, dirname, splitext, join
import itertools
from scipy.optimize import least_squares

def rectify_images(left_img, right_img, F, P_prime, P = None):
	'''
	Description:
	Input arguments:
		* left_img: path or 3D np.ndarray of the left image
		* right_img: path or 3D np.ndarray of the right image
		* F: 3 x 3 np.ndarray
		* P_prime: 3 x 4 np.ndarray
	Return:
	'''
	if(isinstance(left_img, str)): left_img = cv2.imread(left_img)
	if(isinstance(right_img, str)): right_img = cv2.imread(right_img)
	if(P is None):
		P = np.zeros((3, 4))
		P[:, :-3] = np.eye(3)

	l_height, l_width = left_shape[0], left_shape[1]
	r_height, r_width = right_shape[0], right_shape[1]

	T_left = np.array([[1, 0, -1*l_width/2.0],[0, 1, -1*l_height/2.0],[0, 0, 1]])
	T_right = np.array([[1, 0, -1*r_width/2.0],[0, 1, -1*r_height/2.0],[0, 0, 1]])

	###
	## YET TO DO ... INCOMPLETE
	###

	pass

#########################
### Global Variables ####
#########################
x_true = None
x_prime_true = None

def loss_fn(w):
	global x_true, x_prime_true
	P = np.zeros((3, 4))
	P[:,:3] = np.eye(3)

	P_prime = np.reshape(w[:12], (3, 4))

	num_points = len(w[12:])/3

	X = np.reshape(w[12:], (num_points, 3))

	x_hat = homo_to_real(np.dot(P, real_to_homo(X).T).T)
	x_prime_hat = homo_to_real(np.dot(P_prime, real_to_homo(X).T).T)

	left_err = np.linalg.norm(x_true - x_hat, axis = 1)**2
	right_err = np.linalg.norm(x_prime_true - x_prime_hat, axis = 1)**2

	cost = np.sqrt(np.sum(left_err) + np.sum(right_err))

	return cost

def skew(vec):
	try:
		assert len(vec) == 3, 'Error! vec can not have more than three elements.'
	except AssertionError as err:
		print err
		return None
	x, y, z = vec[0], vec[1], vec[2]
	return np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]])

def compute_global_coordinates(F, left_mps, right_mps, left_shape = None, right_shape = None):
	'''
	Description:
	Input arguments:
		* F: 3 x 3 np.ndarray
		* left_mps: 8 x 2 np.ndarray
		* right_mps: 8 x 2 np.ndarray
		* left_shape: shape of the left image
		* right_shape: shape of the right image
	Return:
	'''
	#########################
	### Global Variables ####
	#########################
	global x_true, x_prime_true

	##################
	## Left Camera ###
	##################
	## Compute e_left (epipole of left image)
	U, S, V = np.linalg.svd(F)
	e_left = V.T[:,-1]
	## Left camera matrix
	P = np.zeros((3, 4))
	P[:,:3] = np.eye(3)

	##################
	## Right Camera ##
	##################
	## Compute e_prime or e_right (epipole of right image)
	U, S, V = np.linalg.svd(F.T)
	e_right = V.T[:,-1]
	M = np.dot(skew(e_right), F)
	P_prime = np.zeros((3, 4))
	P_prime[:,:3] = M
	P_prime[:, 3] = e_right

	# print P_prime

	################################
	## Estimate World Coordinates ##
	################################
	G = [] ## Global coordinates
	for idx in range(left_mps.shape[0]):
		xl, yl = left_mps[idx, :]
		xr, yr = right_mps[idx, :]
		A = np.array([xl * P[2,:] - P[0, :],
						yl * P[2, :] - P[1, :],
						xr * P_prime[2,:] - P_prime[0, :],
						yr * P_prime[2, :] - P_prime[1, :]])
		U, S, V = np.linalg.svd(A)
		temp = V.T[:,-1]
		G.append(temp)

	G = np.array(G)
	G = homo_to_real(G)

	# print 'Old P Prime: '
	# print P_prime
	# print 'Old G: '
	# print G

	## Initial values for non linear least squares
	w = np.append(P_prime.flatten(), G.flatten())
	x_true = left_mps
	x_prime_true = right_mps
	w_new = least_squares(loss_fn, w)

	w_new = w_new.x

	new_P_prime = np.reshape(w_new[:12], (3, 4))
	new_G = np.reshape(w_new[12:], (G.shape[0], 3))
	# print 'New P Prime: '
	# print new_P_prime
	# print 'New G: '
	# print new_G

	## Compute modified parameters
	new_e_prime = new_P_prime[:, -1]
	new_F = np.dot(np.dot(skew(new_e_prime), new_P_prime), np.linalg.pinv(P))
	## Compute e_left (epipole of left image)
	U, S, V = np.linalg.svd(new_F)
	new_e_left = homo_to_real(V.T[:,-1])

	return new_F, new_P_prime, new_G

def compute_fund_mat(left_mps, right_mps, left_shape = None, right_shape = None):
	'''
	Description:
	Input arguments:
		* left_mps: 8 x 2 np.ndarray
		* right_mps: 8 x 2 np.ndarray
		* left_shape: shape of the left image
		* right_shape: shape of the right image
	'''
	try:
		assert isinstance(left_mps, np.ndarray), 'left_mps should be numpy array'
		assert isinstance(right_mps, np.ndarray), 'right_mps should be numpy array'
		assert left_mps.shape[0] == right_mps.shape[0], 'Error! No. of rows should be same'
	except AssertionError as err:
		print err
		return None

	if(left_shape is not None):
		l_height, l_width = left_shape[0], left_shape[1]
	else:
		l_height, l_width = 0, 0

	if(right_shape is not None):
		r_height, r_width = right_shape[0], right_shape[1]
	else:
		r_height, r_width = 0, 0

	## Normalization Matrices: Translate origin to center of the image
	T_left = np.array([[1, 0, -1*l_width/2.0],[0, 1, -1*l_height/2.0],[0, 0, 1]])
	T_right = np.array([[1, 0, -1*r_width/2.0],[0, 1, -1*r_height/2.0],[0, 0, 1]])

	## Transform pixel coordinates so that image's center is the origin
	left_mps_t = homo_to_real(np.dot(T_left, real_to_homo(left_mps).T).T)
	right_mps_t = homo_to_real(np.dot(T_right, real_to_homo(right_mps).T).T)

	## Form matrix A to estimate the fundamental matrix F
	A = []
	for idx in range(left_mps.shape[0]):
		xl, yl = left_mps_t[idx, :]
		xr, yr = right_mps_t[idx, :]
		temp = [xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1]
		A.append(temp)
	A = np.array(A)

	## Use SVD to solve linear least squares.
	[U, S, V] = np.linalg.svd(A, full_matrices = True)
	f = V.T[:,-1]
	f = f / f[-1]
	F = np.reshape(f, (3, 3))

	## Condition the F, by making the determinant = 0. Zero the last eigen value.
	[U, S, V] = np.linalg.svd(F, full_matrices = True)
	S[-1] = 0.0
	F = np.dot(np.dot(U, np.diag(S)), V)

	## Denormalization
	F = np.dot(np.dot(T_right.T, F), T_left)

	## Note. x_prime is right and x is the left image.
	# Compute x_prime_transpose * F * x. In theory, it should be equal to zero.
	err_vals = []
	for idx in range(left_mps.shape[0]):
		xl, yl = left_mps[idx, :]
		xr, yr = right_mps[idx, :]
		temp = np.dot(np.dot([xr, yr, 1], F), [xl, yl, 1])
		err_vals.append(temp)

	# print 'Error values: ', err_vals

	return F

def real_to_homo(pts):
	# pts is a 2D numpy array of size _ x 2/3
	# This function converts it into _ x 3/4 by appending 1
	if(pts.ndim == 1):
		return np.append(pts, 1)
	else:
		return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis = 1)

def homo_to_real(pts):
	# pts is a 2D numpy array of size _ x 3/4
	# This function converts it into _ x 2/3 by removing last column
	if(pts.ndim == 1):
		pts = pts / pts[-1]
		return pts[:-1]
	else:
		pts = pts.T
		pts = pts / pts[-1,:]
		return pts[:-1,:].T

def save_mps(event, x, y, flags, param):
	fac, mps = param
	if(event == cv2.EVENT_LBUTTONUP):
		mps.append([int(fac*x), int(fac*y)])
		print(int(fac*x), int(fac*y))

def create_matching_points(img_path, suff = ''):
	npz_path = img_path[:-4]+ suff + '.npz'
	flag = os.path.isfile(npz_path)
	if(not flag):
		img = cv2.imread(img_path)
		fac = max(float(int(img.shape[1]/960)), float(int(img.shape[0]/540)))
		if(fac < 1.0): fac = 1.0
		resz_img = cv2.resize(img, None, fx=1.0/fac, fy=1.0/fac, interpolation = cv2.INTER_CUBIC)
		cv2.namedWindow(img_path)
		mps = []
		cv2.setMouseCallback(img_path, save_mps, param=(fac, mps))
		cv2.imshow(img_path, resz_img)
		cv2.waitKey(0)
		np.savez(npz_path, mps = np.array(mps))
		cv2.destroyAllWindows()
	return np.load(npz_path)

def nmlz(x):
	assert isinstance(x, np.ndarray), 'x should be a numpy array'
	assert x.ndim > 0 and x.ndim < 3, 'dim of x >0 and <3'
	if(x.ndim == 1 and x[-1]!=0): return x/float(x[-1])
	if(x.ndim == 2 and x[-1,-1]!=0): return x/float(x[-1,-1])
	return x

def rem_transl(H):
	assert isinstance(H, np.ndarray), 'H should be a numpy array'
	assert H.ndim == 2, 'H should be a numpy array of two dim'
	assert H.shape[0] == H.shape[1], 'H should be a square matrix'
	H_clone = np.copy(H)
	H_clone[:-1,-1] = 0
	return H_clone

def hinv(H):
	assert isinstance(H, np.ndarray), 'H should be a numpy array'
	assert H.ndim == 2, 'H should be a numpy array of two dim'
	assert H.shape[0] == H.shape[1], 'H should be a square matrix'
	Hinv = np.linalg.inv(H)
	return Hinv / Hinv[-1,-1]
