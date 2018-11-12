import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import sys, os, time
from helpers import *
import glob
from copy import deepcopy

DEBUG = False
NUM_HORZ_LINES = 10
NUM_VERT_LINES = 8
SQUARE_SZ = 2.5

def get_world_coordinates(width = NUM_VERT_LINES, height = NUM_HORZ_LINES, homo = False):
	mat = []
	for yidx in range(height):
		for xidx in range(width):
			if(homo):
				mat.append([xidx*SQUARE_SZ, yidx*SQUARE_SZ, 1])
			else:
				mat.append([xidx*SQUARE_SZ, yidx*SQUARE_SZ])
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

def filter_white_points(img, points, kernel_sz = 10, thresh = 100):
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
		if(DEBUG):
			cv2.circle(img, (x, y), 10, color = [255, 0, 0])
			print max_min_diff
		if(max_min_diff < thresh):
			flags[idx] = False
		if(DEBUG):
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

def find_checkerboard_edges(img_path, display = False):
	'''
	Description:
		Return checker board edges
	Input arguments:
		img_path: Absolute path to the image of a checker board pattern
	Return:
		Return lines of checker board pattern in the form of rho and theta
	'''

	if(not os.path.isfile(img_path)):
		raise IOError('ERROR! ' + img_path + ' does NOT exists !!')

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
		flags = filter_white_points(color_img, h_points)
		new_h_lines = h_lines[flags, :]
		new_h_lines = order_lines(np.array(new_h_lines), type = 'h')
		h_lines = new_h_lines

		flags, _ = filter(v_points[:, 0], thresh = 10.0)
		v_points = v_points[flags, :]
		v_lines = v_lines[flags, :]
		flags = filter_white_points(color_img, v_points)
		new_v_lines = v_lines[flags, :]
		new_v_lines = order_lines(np.array(new_v_lines), type = 'v')
		v_lines = new_v_lines

		if(len(h_lines) == NUM_HORZ_LINES and len(v_lines) == NUM_VERT_LINES):
			break
		else:
			print 'Cleaning: ',
			print len(h_lines), len(v_lines)

	plot_lines(color_img, np.concatenate([new_h_lines, new_v_lines], axis = 0))

	## Obtain final list of points in raster scan order.
	final_points = []
	for hidx, hline in enumerate(h_lines):
		for vidx, vline in enumerate(v_lines):
			final_points.append(intersect_lines(hline, vline))
	final_points = np.array(final_points)

	print final_points.shape
	world_points = get_world_coordinates()
	print world_points.shape

	# if(display):
	# 	cv2.imshow(img_path, color_img)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	return lines

base_dir = r'Data\Dataset1'

img_paths = glob.glob(os.path.join(base_dir, '*.jpg'))

# img_path = img_paths[1]
for img_path in img_paths:
	lines = find_checkerboard_edges(img_path, display = True)
