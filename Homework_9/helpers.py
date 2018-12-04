import cv2
import numpy as np
import os, time, sys
# from NonlinearLeastSquares import NonlinearLeastSquares as NLS
import matplotlib.pyplot as plt
from os.path import basename, dirname, splitext, join
import itertools
from scipy.optimize import least_squares

#########################
### Global Variables ####
#########################
x_true = None
x_prime_true = None

def get_null_vec(A):
	U, S, V = np.linalg.svd(A)
	return V.T[:,-1]

def abc_loss_fn(w):
	temp = np.sum(real_to_homo(x_prime_true) * np.array(w), axis = 1) - x_true[:, 0]
	return np.sum(temp ** 2)

def rectify_images(left, right, F):
	'''
	Description:
	Input arguments:
		* left is a dict
			# 'img': image path or np.ndarray of an image
			# 'mps': matching points. 2D np.ndarray. Rows are points. Columns are x and y coordinates.
			# 'e': 1D np.array. Epi pole
			# 'P': Camera matrix of left image. 2D 3 x 4 np.ndarray
		* right is a dict
			# 'img': image path or np.ndarray of an image
			# 'mps': matching points. 2D np.ndarray. Rows are points. Columns are x and y coordinates.
			# 'e': 1D np.array. Epi pole
			# 'P': Camera matrix of left image. 2D 3 x 4 np.ndarray
		* F: 3 x 3 np.ndarray
	Return:
	'''
	global x_true, x_prime_true
	left_img = left['img']
	left_mps = left['mps']
	left_e = left['e']
	P = left['P']

	right_img = right['img']
	right_mps = right['mps']
	right_e = right['e']
	P_prime = right['P']

	if(isinstance(left_img, str)): left_img = cv2.imread(left_img)
	if(isinstance(right_img, str)): right_img = cv2.imread(right_img)

	l_height, l_width = left_img.shape[0], left_img.shape[1]
	r_height, r_width = right_img.shape[0], right_img.shape[1]

	T_left = np.array([[1, 0, -1*l_width/2.0],[0, 1, -1*l_height/2.0],[0, 0, 1]])
	T_right = np.array([[1, 0, -1*r_width/2.0],[0, 1, -1*r_height/2.0],[0, 0, 1]])

	## Rectifying the right image
	t_right_e = nmlz(np.dot(T_right, right_e))
	angle = np.arctan(-1*t_right_e[1]/t_right_e[0])
	print np.arctan2(t_right_e[1], -1*t_right_e[0])
	f = t_right_e[0] * np.cos(angle) - t_right_e[1] * np.sin(angle)

	G = np.array([[1, 0, 0],
				[0, 1, 0],
				[-1.0/f, 0, 1]])
	R = np.array([[np.cos(angle), -1*np.sin(angle), 0],
				[np.sin(angle), np.cos(angle), 0],
				[0, 0, 1]])
	H2 = np.dot(np.dot(G, R), T_right)
	# print 'new e right', np.dot(H2, right_e)

	r_center_rect = nmlz(np.dot(H2, [r_width/2.0, r_height/2.0, 1]))
	T2 = np.array([[1, 0, r_width/2.0 - r_center_rect[0]],
				[0, 1, r_height/2.0 - r_center_rect[1]],
				[0, 0, 1]])

	H2 = np.dot(T2, H2)

	## Rectifying left image.
	M = np.dot(P_prime, np.linalg.pinv(P))
	H0 = np.dot(H2, M)
	left_mps_hat = homo_to_real(np.dot(H0, real_to_homo(left_mps).T).T)
	right_mps_hat = homo_to_real(np.dot(H2, real_to_homo(right_mps).T).T)
	x_prime_true = left_mps_hat
	x_true = right_mps_hat
	x = [0.0, 0., 0.]
	x = least_squares(abc_loss_fn, x).x
	print 'x', x
	# A = real_to_homo(left_mps_hat)
	# b = right_mps_hat[:, 0]
	# x = np.dot(np.linalg.pinv(A), b)

	HA = np.array([[x[0], x[1], x[2]],
				[0, 1, 0],
				[0, 0, 1]])

	H1 = np.dot(HA, H0)
	H1 = H0
	print 'x', x
	print 'H1', H1

	# l_center_rect = nmlz(np.dot(H1, [l_width/2.0, l_height/2.0, 1]))
	# print 'l_center_rect', l_center_rect
	# print 'l actual center', [l_width/2.0, l_height/2.0, 1]
	# T1 = np.array([[1, 0, l_width/2.0 - l_center_rect[0]],
	# 			[0, 1, l_height/2.0 - l_center_rect[1]],
	# 			[0, 0, 1]])
	# print 'T1', T1
	# H1 = np.dot(T1, H1)

	## Find rectified F
	F_rect = np.dot(np.dot(np.linalg.inv(H2.T), F), np.linalg.inv(H1))

	## Rectified matching points
	left_mps_rect = homo_to_real(np.dot(H1, real_to_homo(left_mps).T).T)
	right_mps_rect = homo_to_real(np.dot(H2, real_to_homo(right_mps).T).T)

	## Rectified epi poles
	left_e_rect = nmlz(get_null_vec(F_rect))
	right_e_rect = nmlz(get_null_vec(F_rect.T))

	left['mps_rect'] =  left_mps_rect
	left['e_rect'] = left_e_rect
	left['H'] = H1

	right['mps_rect'] = right_mps_rect
	right['e_rect'] = right_e_rect
	right['H'] = H2

	return left, right, F_rect

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

def compute_global_coordinates(left, right, F):
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

	left_mps = left['mps']
	left_shape = left['img'].shape

	right_mps = right['mps']
	right_shape = right['img'].shape

	##################
	## Left Camera ###
	##################
	## Compute e_left (epipole of left image)
	U, S, V = np.linalg.svd(F)
	e_left = nmlz(V.T[:,-1])
	## Left camera matrix
	P = np.zeros((3, 4))
	P[:,:3] = np.eye(3)

	##################
	## Right Camera ##
	##################
	## Compute e_prime or e_right (epipole of right image)
	U, S, V = np.linalg.svd(F.T)
	e_right = nmlz(V.T[:,-1])
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
	new_e_prime = nmlz(new_P_prime[:, -1])
	new_F = np.dot(np.dot(skew(new_e_prime), new_P_prime), np.linalg.pinv(P))
	## Compute e_left (epipole of left image)
	U, S, V = np.linalg.svd(new_F)
	new_e_left = nmlz(V.T[:,-1])

	return new_F, new_e_left, new_e_prime, P, new_P_prime, new_G

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

def apply_homography2(img_path, H, num_partitions = 1, suff = ''):
	if(isinstance(img_path, str)): img = cv2.imread(img_path)
	else:
		img = img_path
		img_path = 'sample.jpg'

	img[0,:], img[:,0], img[-1,:], img[:,-1] = 0, 0, 0, 0

	xv, yv = np.meshgrid(range(0, img.shape[1], img.shape[1]-1), range(0, img.shape[0], img.shape[0]-1))
	img_pts = np.array([xv.flatten(), yv.flatten()]).T
	trans_img_pts = np.dot(H, real_to_homo(img_pts).T)
	ttt = homo_to_real(trans_img_pts.T).T
	_w = np.max(ttt[0, :]) - np.min(ttt[0, :])
	_h = np.max(ttt[1, :]) - np.min(ttt[1, :])
	l1, l2 = img.shape[1] / _w, img.shape[0] / _h
	K = np.diag([l1, l2, 1])
	H = np.dot(K, H)

	xv, yv = np.meshgrid(range(0, img.shape[1], img.shape[1]-1), range(0, img.shape[0], img.shape[0]-1))
	img_pts = np.array([xv.flatten(), yv.flatten()]).T
	trans_img_pts = np.dot(H, real_to_homo(img_pts).T)
	trans_img_pts = homo_to_real(trans_img_pts.T).astype(int)

	xmin, ymin = np.min(trans_img_pts[:,0]), np.min(trans_img_pts[:,1])
	xmax, ymax = np.max(trans_img_pts[:,0]), np.max(trans_img_pts[:,1])
	W_new = xmax - xmin
	H_new = ymax - ymin

	img_new = np.zeros((H_new+1, W_new+1, 3), dtype = np.uint8)
	print 'Shape of new image: ', img_new.shape

	x_batch_sz = int(W_new/float(num_partitions))
	y_batch_sz = int(H_new/float(num_partitions))
	for x_part_idx in range(num_partitions):
		for y_part_idx in range(num_partitions):
			x_start, x_end = x_part_idx*x_batch_sz, (x_part_idx+1)*x_batch_sz
			y_start, y_end = y_part_idx*y_batch_sz, (y_part_idx+1)*y_batch_sz
			xv, yv = np.meshgrid(range(x_start, x_end), range(y_start, y_end))
			xv, yv = xv + xmin, yv + ymin
			img_new_pts = np.array([xv.flatten(), yv.flatten()]).T
			trans_img_new_pts = np.dot(hinv(H), real_to_homo(img_new_pts).T)
			trans_img_new_pts = homo_to_real(trans_img_new_pts.T).astype(int)
			trans_img_new_pts[:,0] = np.clip(trans_img_new_pts[:,0], 0, img.shape[1]-1)
			trans_img_new_pts[:,1] = np.clip(trans_img_new_pts[:,1], 0, img.shape[0]-1)
			img_new_pts = img_new_pts - [xmin, ymin]
			# This is the bottle nect step. It takes the most time.
			img_new[img_new_pts[:,1].tolist(), img_new_pts[:,0].tolist(), :] = img[trans_img_new_pts[:,1].tolist(), trans_img_new_pts[:,0].tolist(), :]

	fname, ext = tuple(os.path.basename(img_path).split('.'))
	write_filepath = os.path.join(os.path.dirname(img_path), fname+suff+'.'+ext)
	print write_filepath
	cv2.imwrite(write_filepath, img_new)

def apply_homography(img_path, H, num_partitions = 1, suff = ''):
	if(isinstance(img_path, str)): img = cv2.imread(img_path)
	else:
		img = img_path
		img_path = 'sample.jpg'

	img[0,:], img[:,0], img[-1,:], img[:,-1] = 0, 0, 0, 0

	xv, yv = np.meshgrid(range(0, img.shape[1], img.shape[1]-1), range(0, img.shape[0], img.shape[0]-1))
	img_pts = np.array([xv.flatten(), yv.flatten()]).T
	trans_img_pts = np.dot(H, real_to_homo(img_pts).T)
	trans_img_pts = homo_to_real(trans_img_pts.T).astype(int)

	print 'trans_img_pts'
	print trans_img_pts

	xmin, ymin = np.min(trans_img_pts[:,0]), np.min(trans_img_pts[:,1])
	xmax, ymax = np.max(trans_img_pts[:,0]), np.max(trans_img_pts[:,1])
	W_new = xmax - xmin
	H_new = ymax - ymin

	img_new = np.zeros((H_new+1, W_new+1, 3), dtype = np.uint8)
	print 'Shape of new image: ', img_new.shape

	x_batch_sz = int(W_new/float(num_partitions))
	y_batch_sz = int(H_new/float(num_partitions))
	for x_part_idx in range(num_partitions):
		for y_part_idx in range(num_partitions):
			x_start, x_end = x_part_idx*x_batch_sz, (x_part_idx+1)*x_batch_sz
			y_start, y_end = y_part_idx*y_batch_sz, (y_part_idx+1)*y_batch_sz
			xv, yv = np.meshgrid(range(x_start, x_end), range(y_start, y_end))
			xv, yv = xv + xmin, yv + ymin
			img_new_pts = np.array([xv.flatten(), yv.flatten()]).T
			trans_img_new_pts = np.dot(hinv(H), real_to_homo(img_new_pts).T)
			trans_img_new_pts = homo_to_real(trans_img_new_pts.T).astype(int)
			trans_img_new_pts[:,0] = np.clip(trans_img_new_pts[:,0], 0, img.shape[1]-1)
			trans_img_new_pts[:,1] = np.clip(trans_img_new_pts[:,1], 0, img.shape[0]-1)
			img_new_pts = img_new_pts - [xmin, ymin]
			# This is the bottle nect step. It takes the most time.
			img_new[img_new_pts[:,1].tolist(), img_new_pts[:,0].tolist(), :] = img[trans_img_new_pts[:,1].tolist(), trans_img_new_pts[:,0].tolist(), :]

	fname, ext = tuple(os.path.basename(img_path).split('.'))
	write_filepath = os.path.join(os.path.dirname(img_path), fname+suff+'.'+ext)
	print write_filepath
	cv2.imwrite(write_filepath, img_new)
