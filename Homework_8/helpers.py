import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import sys, os, time
from helpers import *
import glob
from copy import deepcopy
from NonlinearLeastSquares import NonlinearLeastSquares as NLS

def get_v_rep(H, p, q):
	'''
	Description:
		V(H, p, q) is computed. It is a part of Zhang's algorithm for camera callibration.
	Input argumnets:
		* H - 2D square np.ndarray of size 3 x 3
		* p - {0, 1, 2}
		* q - {0, 1, 2}
	Return:
		V(H, p, q)
	'''
	assert isinstance(H, np.ndarray), 'H should be a numpy array'
	assert H.ndim == 2, 'H should be a 2D np.ndarray'
	assert H.shape[0] == H.shape[1], 'H should be a square matrix'
	assert p < 3 and p >= 0, 'p should be 0, 1, or 2'
	assert q < 3 and q >= 0, 'q should be 0, 1, or 2'

	# v0 = H[p, 0] * H[q, 0]
	# v1 = H[p, 0] * H[q, 1] + H[p, 1] * H[q, 0]
	# v2 = H[p, 1] * H[q, 1]
	# v3 = H[p, 2] * H[q, 0] + H[p, 0] * H[q, 2]
	# v4 = H[p, 2] * H[q, 1] + H[p, 1] * H[q, 2]
	# v5 = H[p, 2] * H[q, 2]

	v0 = H[0, p] * H[0, q]
	v1 = H[0, p] * H[1, q] + H[1, p] * H[0, q]
	v2 = H[1, p] * H[1, q]
	v3 = H[2, p] * H[0, q] + H[0, p] * H[2, q]
	v4 = H[2, p] * H[1, q] + H[1, p] * H[2, q]
	v5 = H[2, p] * H[2, q]

	return np.array([v0, v1, v2, v3, v4, v5])

def get_world_coordinates(pattern_size, unit_size, homo = False, display = False):
	'''
	Description:
		Compute the world coordinates of the checkerboard pattern of given size.
		Returns the word coo. in raster scan order i.e. row by row starting from the first row.
	Input arguments:
		* pattern_size = (height, width). For instance (9, 7) ==> (NUM_HORZ_LINES-1, NUM_VERT_LINES-1)
			* width: width of the pattern in terms of no. of blocks along x - axis.
			* height: height of the pattern in terms of no. of blocks along y - axis.
		* unit_size: A floating point value indicating the size of the block.
	Return:
		* mat: A 2D np.ndarray. Each row is the world point (x, y) either in physical or homogeneous coordinates.
			* It looks like [[x1, y1, 1], [x2, y2, 1], ...]
	'''
	height, width = pattern_size
	height = height + 1
	width = width + 1

	mat = []
	for yidx in range(height):
		for xidx in range(width):
			if(homo):
				mat.append([xidx*unit_size, yidx*unit_size, 1])
			else:
				mat.append([xidx*unit_size, yidx*unit_size])

	## Display the world coordinates for debugging purposes.
	if(display):
		scale = 20
		offset_x = 50
		offset_y  = 50
		img_width = int(width*unit_size*scale) + offset_x
		img_height = int(height * unit_size*scale) + offset_y
		img = 255 * np.ones((img_height, img_width, 3), dtype = np.uint8)
		for row in mat:
			x, y = int(row[0]*scale+offset_x), int(row[1]*scale+offset_y)
			cv2.circle(img, (x, y), 5, color = (255, 0, 0))
			cv2.imshow('World points', img)
			cv2.waitKey(0)

	return np.array(mat)

def order_lines(lines, type = 'h'):
	'''
	lines: list of elements. Each element is of form [x1, y1, x2, y2]
	type: 'h' for horizontal lines and 'v' for vertical lines
	'''
	intercept_list = []
	for idx, line in enumerate(lines):
		x1, y1, x2, y2 = tuple(line.tolist())
		line = np.cross([x1, y1, 1], [x2, y2, 1])
		if(type == 'h'):
			intercept = -1 * line[2] / line[1]
		elif(type == 'v'):
			intercept = -1 * line[2] / line[0]
		else:
			raise Exception('type should be "h" or "v"')
		intercept_list.append(intercept)
	argsort = np.argsort(intercept_list)
	return lines[argsort, :]

def plot_lines(img, lines):
	'''
	img: 2D or 3D np.ndarray
	lines: list of elements. Each element is a list: it looks like [x1, y1, x2, y2]
	'''
	img = np.copy(img)
	for line in lines:
		x1, y1, x2, y2 = tuple(line)
		cv2.line(img,(x1,y1),(x2,y2),(0,255,25),2)
	cv2.imshow('Lines', img)
	cv2.waitKey(0)

def filter_white_points(img, points, kernel_sz = 10, thresh = 110, debug = False):
	'''
	points: 2D/3D np.ndarray. Rows look like [x1, y1].
	'''
	img = np.copy(img)
	flags = np.zeros(points.shape[0]).astype(int) == 0
	for idx, point in enumerate(points):
		point = point.astype(int)
		x, y = point[0], point[1]
		if(img.ndim == 2):
			temp = img[y-kernel_sz:y+kernel_sz, x-kernel_sz:x+kernel_sz].flatten()
		else:
			temp = img[y-kernel_sz:y+kernel_sz, x-kernel_sz:x+kernel_sz, :].flatten()
		max_min_diff = np.max(temp) - np.min(temp)
		if(debug):
			cv2.circle(img, (x, y), 10, color = [255, 0, 0])
			print max_min_diff
		if(max_min_diff < thresh):
			flags[idx] = False
		if(debug):
			cv2.imshow('', img)
			cv2.waitKey(0)
	return flags

def intersect_lines(pair1, pair2):
	'''
	pair1: A list [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are start and end points of the line
	pair2: A list [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are start and end points of the line
	'''
	line1 = np.cross([pair1[0], pair1[1], 1], [pair1[2], pair1[3], 1])
	line2 = np.cross([pair2[0], pair2[1], 1], [pair2[2], pair2[3], 1])
	point = np.cross(line1, line2)
	point = point / point[-1]
	return point[:-1].tolist()

def filter(M, thresh):
	'''
	Description:
		* Filters the rows in M. Eliminate the rows in M are
		 very close according to euclidean norm between rows.
	M: 2D np.ndarray. Rows are features
	'''
	M = deepcopy(M)
	if(M.ndim == 1):
		M = M.reshape(-1, 1)
	flags = np.zeros(M.shape[0]).astype(int) == 0
	dist_mat = dist_mat_mat(M, M)
	max_val = 2 * np.max(M.flatten()) # Some big value
	for idx, row in enumerate(dist_mat):
		row[:idx+1] = max_val
		nz_ids = np.nonzero(row < thresh)[0]
		flags[nz_ids] = False
	return flags, M[flags, :]

def rth_to_xy(rth_arr):
	'''
	rth_arr: 2D np.ndarray. Each row has two elements (rho and theta).
	'''
	xy_arr = []
	for line in rth_arr:
		rho, theta = line[0], line[1]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		nline = np.cross([x1, y1, 1],[x2, y2, 1])
		nline = nline / np.max(nline)
		xy_arr.append(nline)
	return np.array(xy_arr)

def find_checkerboard_points(img_path, pattern_size, unit_size, display = False):
	'''
	Description:
		Return checker board edges
	Input arguments:
		img_path: Absolute path to the image of a checker board pattern
		* pattern_size = (height, width). For instance (9, 7) ==> (NUM_HORZ_LINES-1, NUM_VERT_LINES-1)
			* width: width of the pattern in terms of no. of blocks along x - axis.
			* height: height of the pattern in terms of no. of blocks along y - axis.
	Return:
		Return lines of checker board pattern in the form of rho and theta
	'''

	if(not os.path.isfile(img_path)):
		raise IOError('ERROR! ' + img_path + ' does NOT exists !!')

	num_horz_lines = pattern_size[0] + 1
	num_vert_lines = pattern_size[1] + 1

	color_img = cv2.imread(img_path)
	img = cv2.imread(img_path, 0) # Read image as grayscale
	height, width = img.shape
	diag_length = np.max([width, height])

	## Apply Canny edge detector
	edges = cv2.Canny(img, 100, 200)

	## Apply Hough transform to detect the edges
	lines = cv2.HoughLines(edges, 1, np.pi/180, 50) # (_ x 1 x 2)
	lines = np.reshape(lines, (lines.shape[0], lines.shape[2])) # (_ x 2)

	h_lines = []
	v_lines = []

	for line in lines:
		rho, theta = line[0], line[1]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		# if(display): cv2.line(color_img,(x1,y1),(x2,y2),(0,255,25),2)
		if np.abs(np.cos(theta)) > np.abs(np.sin(theta)):
			v_lines.append([x1, y1, x2, y2])
		else:
			h_lines.append([x1, y1, x2, y2])

	h_lines = order_lines(np.array(h_lines), type = 'h')
	v_lines = order_lines(np.array(v_lines), type = 'v')

	# print 'NO. of hlines: ', len(h_lines)
	# print 'NO. of vlines: ', len(v_lines)

	for rep_idx in range(20):
		if(rep_idx == 0):
			hidx = len(h_lines)/2
			vidx = len(v_lines)/2
		else:
			hidx = np.random.randint(0, len(h_lines))
			vidx = np.random.randint(0, len(v_lines))
		## Intersect first vertical line with all horizontal lines
		h_points = np.array([intersect_lines(line, v_lines[vidx, :]) for line in h_lines])
		## Intersect first horizontal line with all vertical lines
		v_points = np.array([intersect_lines(line, h_lines[hidx, :]) for line in v_lines])

		flags, _ = filter(h_points[:, 1], thresh = 10.0)
		h_points = h_points[flags, :]
		h_lines = h_lines[flags, :]
		flags = filter_white_points(color_img, h_points, debug = False)
		new_h_lines = h_lines[flags, :]
		new_h_lines = order_lines(np.array(new_h_lines), type = 'h')
		h_lines = new_h_lines

		flags, _ = filter(v_points[:, 0], thresh = 10.0)
		v_points = v_points[flags, :]
		v_lines = v_lines[flags, :]
		flags = filter_white_points(color_img, v_points, debug = False)
		new_v_lines = v_lines[flags, :]
		new_v_lines = order_lines(np.array(new_v_lines), type = 'v')
		v_lines = new_v_lines

		if(len(h_lines) == num_horz_lines and len(v_lines) == num_vert_lines):
			break
		else:
			print 'Cleaning: ',
			print len(h_lines), len(v_lines)

	if(display):
		plot_lines(color_img, np.concatenate([new_h_lines, new_v_lines], axis = 0))

	## Obtain final list of points in raster scan order.
	final_points = []
	for hidx, hline in enumerate(h_lines):
		for vidx, vline in enumerate(v_lines):
			final_points.append(intersect_lines(hline, vline))
	final_points = np.array(final_points)

	world_points = get_world_coordinates(pattern_size, unit_size)

	assert final_points.shape[0] == world_points.shape[0], \
			'Error! No. of image and world points should be same '

	return final_points, world_points

def get_pts(shap, corners = False, H=None, targ_shap = None):
	#####
	#
	# Description:
	#
	# Input:
	#   shap: tuple (num_rows, num_cols, optional) - for given image
	#   corners: if True, only corner points, if False, all points
	#   H: 3 x 3 ndarray - given image to target image
	#   targ_shap: tuple (num_rows, num_cols, optional) - for target image
	#
	#   if H is not None, apply the homography
	#   if targ_shap is not None, clip the transformed pts accordingly.
	#
	# Return:
	#   trasformed points. In (x, y) format. It depends on if H, targ_shap are None
	#
	#####

	M, N = shap[0], shap[1]
	if(corners):
		pts = np.array([[0, 0],[N-1, 0],[N-1, M-1], [0, M-1]])
	else:
		xv, yv = np.meshgrid(range(N), range(M))
		pts = np.array([xv.flatten(), yv.flatten()]).T
	if H is None: return pts, None

	# else
	t_pts = np.dot(H, real_to_homo(pts).T)
	t_pts = homo_to_real(t_pts.T).astype(int)
	if(targ_shap is None): return pts, t_pts

	#else
	t_pts[:,0] = np.clip(t_pts[:,0], 0, targ_shap[1]-1)
	t_pts[:,1] = np.clip(t_pts[:,1], 0, targ_shap[0]-1)
	return pts, t_pts

def find_homography_2d(pts1, pts2):
	# H: 2 --> 1

	# Assertion
	assert pts1.shape[1] == 2, 'pts1 should have two columns'
	assert pts2.shape[1] == 2, 'pts2 should have two columns'
	assert pts1.shape[0] == pts2.shape[0], 'pts1 and pts2 should have same number of rows'

	# Forming the matrix A (8 x 9)
	A = []
	for (x1, y1), (x2, y2) in zip(pts1, pts2):
		A.append([x2, y2, 1, 0, 0, 0, -1*x1*x2, -1*x1*y2, -1*x1])
		A.append([0, 0, 0, x2, y2, 1, -1*y1*x2, -1*y1*y2, -1*y1])
	A = np.array(A)

	[U, S, V] = np.linalg.svd(A, full_matrices = True)
	h = V.T[:,-1]
	h = h / h[-1]

	# # Finding the homography. H[3,3] is assumed 1.
	# h = np.dot(np.linalg.pinv(A[:,:-1]), -1*A[:,-1])
	# h = np.append(h, 1)

	H = np.reshape(h, (3, 3))
	return H, hinv(H)

hvars = ['h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32']
Nx = '(h11*{0}+h12*{1}+h13)'
Ny = '(h21*{0}+h22*{1}+h23)'
D = '(h31*{0}+h32*{1}+1)'

def senc(value): return '('+str(value)+')'

def fvec_row(x, y, axis = 'x'):
	if(axis == 'x'):
		fvec = Nx + '/' + D
	else:
		fvec = Ny + '/' + D
	return fvec.format(senc(x), senc(y))

def jac_row(x, y, axis = 'x'):
	d = [0]*8
	if(axis == 'x'):
		d[0] = '{0}'+'/'+D
		d[1] = '{1}'+'/'+D
		d[2] = '1'+'/'+D
		d[3] = '0'
		d[4] = '0'
		d[5] = '0'
		d[6] = '(-'+Nx+'*'+'{0})/' + '(' + D + '**2)'
		d[7] = '(-'+Nx+'*'+'{1})/' + '(' + D + '**2)'
		# d[8] = '(-'+Nx+'*'+'1)/' + '(' + D + '**2)'
	else:
		d[0] = '0'
		d[1] = '0'
		d[2] = '0'
		d[3] = '{0}'+'/'+D
		d[4] = '{1}'+'/'+D
		d[5] = '1'+'/'+D
		d[6] = '(-'+Ny+'*'+'{0})/' + '(' + D + '**2)'
		d[7] = '(-'+Ny+'*'+'{1})/' + '(' + D + '**2)'
		# d[8] = '(-'+Ny+'*'+'1)/' + '(' + D + '**2)'

	for idx, _ in enumerate(d):
		d[idx] = d[idx].format(senc(x), senc(y))

	return d

def LM_Minimizer(point_corresps, H_init, max_iter = 200, \
				 delta_for_jacobian = 0.000001, \
				 delta_for_step_size = 0.0001, debug = False):
	'''
		H: 2 --> 1
	'''
	nls =  NLS(max_iterations = max_iter, \
			   delta_for_jacobian = delta_for_jacobian, \
			   delta_for_step_size = delta_for_step_size, debug = debug)

	pts1 = point_corresps[:,:2]
	pts2 = point_corresps[:,2:]

	X = pts1.flatten().reshape(-1, 1) # [x1, y1, x2, y2, ...]

	Jac = []
	Fvec = []
	for x, y in pts2:
		fx = fvec_row(x, y, 'x')
		fy = fvec_row(x, y, 'y')
		dfx = jac_row(x, y, 'x')
		dfy = jac_row(x, y, 'y')
		Jac.append(dfx)
		Jac.append(dfy)
		Fvec.append(fx)
		Fvec.append(fy)

	Fvec = np.array(Fvec).reshape(-1, 1)
	Jac = np.array(Jac)

	nls.set_Fvec(Fvec)
	nls.set_X(X)
	nls.set_jacobian_functionals_array(Jac)
	nls.set_params_ordered_list(hvars)
	nls.set_initial_params(dict(zip(hvars, H_init.flatten().tolist())))

	# print Jac
	# print ''
	# print Fvec

	return nls.leven_marq()

def apply_trans_patch(base_img_path, template_img_path, H, suff = '_fnew'):
	## Read images
	if(isinstance(base_img_path, str)): base_img = cv2.imread(base_img_path)
	else: base_img = np.copy(base_img_path)

	if(isinstance(template_img_path, str)): temp_img = cv2.imread(template_img_path)
	else: temp_img = np.copy(template_img_path)

	## Find corners in base that correspond to corners in template
	temp_cpts, trans_temp_cpts = get_pts(temp_img.shape, corners=True, H=H, targ_shap = base_img.shape)

	_cpts = real_to_homo(trans_temp_cpts) # homo. representation
	_cent_cpts = np.mean(_cpts, axis = 0)# centroid of four points

	# Find the four lines of quadrilateral
	lines = [np.cross(_cpts[0], _cpts[1]), np.cross(_cpts[1], _cpts[2]), np.cross(_cpts[2], _cpts[3]), np.cross(_cpts[3], _cpts[0])]

	## Finding points in the base that are present in the quadrilateral
	base_bool = np.zeros(base_img.shape[:-1]).flatten() == 0 # True -> inside the quadrilateral
	base_all_pts, _ = get_pts(base_img.shape) # get all pts
	for line in lines:
		line = line / line[-1]
		sn = int(np.sign(np.dot(_cent_cpts, line)))
		nsn = np.int8(np.sign(np.dot(real_to_homo(base_all_pts), line)))
		base_bool = np.logical_and(base_bool, nsn==sn)
	base_bool = base_bool
	base_bool = np.reshape(base_bool, (base_img.shape[0], base_img.shape[1]))
	row_ids, col_ids = np.nonzero(base_bool)
	des_base_pts = np.array([col_ids, row_ids])

	# Find corresponding points in the template image
	trans_des_base_pts = homo_to_real(np.dot(hinv(H), real_to_homo(des_base_pts.T).T).T).astype(int).T

	# Clip the points
	trans_des_base_pts[:, 0] = np.clip(trans_des_base_pts[:, 0], 0, temp_img.shape[1]-1)
	trans_des_base_pts[:, 1] = np.clip(trans_des_base_pts[:, 1], 0, temp_img.shape[0]-1)

	base_img[des_base_pts[1].tolist(), des_base_pts[0].tolist(), :] = temp_img[trans_des_base_pts[1].tolist(), trans_des_base_pts[0].tolist(), :]

	# Write the resulting image to a file
	fname, ext = tuple(os.path.basename(base_img_path).split('.'))
	write_filepath = os.path.join(os.path.dirname(base_img_path), fname+suff+'.'+ext)
	print write_filepath
	cv2.imwrite(write_filepath, base_img)


def dist_mat_vec(M, vec):
	# Compute distance between each row of 'M' with 'vec'
	# method: 'ncc', 'dot', 'ssd'
	# M : ndarray ( _ x k); vec: (1 x k)
	# Returns a 1D numpy array of distances.
	return np.linalg.norm(M - vec, axis = 1)

def dist_mat_mat(M1, M2):
	# M1, M2 --> ndarray (y1 x k) and (y2 x k)
	# Returns y1 x y2 ndarray with the distances.
	# If y1 and y2 are huge, it might run into MemoryError
	D = np.zeros((M1.shape[0], M2.shape[0]))
	for idx2 in range(M2.shape[0]):
		D[:, idx2] = dist_mat_vec(M1, M2[idx2, :])
	return D

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

def hinv(H):
	assert isinstance(H, np.ndarray), 'H should be a numpy array'
	assert H.ndim == 2, 'H should be a numpy array of two dim'
	assert H.shape[0] == H.shape[1], 'H should be a square matrix'
	Hinv = np.linalg.inv(H)
	return Hinv / Hinv[-1,-1]

def nmlz(x):
	assert isinstance(x, np.ndarray), 'x should be a numpy array'
	assert x.ndim > 0 and x.ndim < 3, 'dim of x >0 and <3'
	if(x.ndim == 1 and x[-1]!=0): return x/float(x[-1])
	if(x.ndim == 2 and x[-1,-1]!=0): return x/float(x[-1,-1])
	return x
