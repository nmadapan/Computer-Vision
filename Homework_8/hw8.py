import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import sys, os, time
from helpers import *
import glob

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
		nline = nline / np.linalg.norm(nline)
		xy_arr.append(nline)
	return np.array(xy_arr)

def find_checkerboard_edges(img_path, dist_thresh = 0.02, display = False):
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

	## Parameters
	# dist_thresh = 0.02

	img = cv2.imread(img_path, 0) # Read image as grayscale
	height, width = img.shape
	diag_length = np.max([width, height])

	## Apply Canny edge detector
	edges = cv2.Canny(img, 100, 200)

	## Apply Hough transform to detect the edges
	lines = cv2.HoughLines(edges, 1, np.pi/180, 40) # (_ x 1 x 2)
	lines = np.reshape(lines, (lines.shape[0], lines.shape[2])) # (_ x 2)

	## Remove the lines that are very close to each other
	M = dist_mat_mat(lines, lines, weights = [10/diag_length, 10/np.pi])
	flags = np.zeros(lines.shape[0]).astype(int) == 0
	for idx, row in enumerate(M):
		# plt.bar(range(len(row)), row)
		# plt.show()
		row[:idx+1] = 1000
		nz_ids = np.nonzero(row < dist_thresh)[0]
		flags[nz_ids] = False
	lines = lines[flags]

	if(display):
		for line in lines:
			print line
			rho, theta = line[0], line[1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			# print x1, y1, x2, y2
			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
			cv2.imshow(img_path, img)
			cv2.waitKey(0)
		cv2.destroyAllWindows()

	return lines

base_dir = r'Data\Dataset1'

img_paths = glob.glob(os.path.join(base_dir, '*.jpg'))
img_path = img_paths[3]

dist_thresh = 0.05
lines = find_checkerboard_edges(img_path, dist_thresh = dist_thresh, display = True)

