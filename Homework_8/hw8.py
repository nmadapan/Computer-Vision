import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import sys, os, time
from helpers import *
import glob
from copy import deepcopy
import pickle

NUM_HORZ_LINES = 10
NUM_VERT_LINES = 8
SQUARE_SZ = 25

dataset = 'dataset1'
base_dir = os.path.join('.\\Data', dataset)
out_dir = os.path.join('.\\Data\\results', dataset)

img_paths = glob.glob(os.path.join(base_dir, '*.jpg'))

if os.path.isfile('homographies_'+os.path.basename(base_dir)+'.pkl'):
	with open('homographies_'+os.path.basename(base_dir)+'.pkl', 'rb') as fp:
		temp = pickle.load(fp)
	homographies, img_points_list, world_points_list = zip(*temp['homographies'])
else:
	homographies = []
	print 'Processing'
	for img_path in img_paths:
		print img_path
		img_points, world_points = find_checkerboard_points(img_path, \
			(NUM_HORZ_LINES-1, NUM_VERT_LINES-1), unit_size = SQUARE_SZ, display = False)
		## Find homography from 2 --> 1
		H, Hinv = find_homography_2d(img_points, world_points)
		## Uncomment to apply the transformed patch on the original image.
		# template_img = 255 * np.ones((SQUARE_SZ*(NUM_HORZ_LINES-1), SQUARE_SZ*(NUM_VERT_LINES-1), 3), dtype = np.uint8)
		# apply_trans_patch(img_path, template_img, H, suff = '_fnew')

		lmres = LM_Minimizer(np.concatenate([img_points, world_points], axis = 1), H)

		new_H = np.squeeze(np.asarray(lmres['parameter_values']))
		new_H = np.append(new_H, np.array([1])).reshape(3, 3)
		new_H = nmlz(new_H)

		# pred_img_points = homo_to_real(np.dot(new_H, real_to_homo(world_points).T).T)
		# print np.append(img_points, pred_img_points, axis = 1)
		# sys.exit()

		homographies.append((new_H, img_points, world_points))

	with open('homographies_'+os.path.basename(base_dir)+'.pkl', 'wb') as fp:
		pickle.dump({'homographies': homographies}, fp)
	homographies, img_points_list, world_points_list = zip(*homographies)

V = []
for H in homographies:
	V12 = get_v_rep(H, 0, 1)
	V11 = get_v_rep(H, 0, 0)
	V22 = get_v_rep(H, 1, 1)
	V.append(V12)
	V.append(V11 - V22)
V = np.array(V)

[U, S, E] = np.linalg.svd(V, full_matrices = True)
b = E.T[:,-1]

# ww = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
# dd = b[0]*b[2] - b[1]**2
# alph = np.sqrt(ww/(dd*b[0])) # alpha_x
# bet = np.sqrt(b[0]*ww/(dd**2)) # alpha_y
# gam = -1 * b[1] * np.sqrt(ww/((dd**2)*b[0])) # s # I added negative sign.
# uc = (b[1]*b[4] - b[2]*b[3]) / dd # x0
# vc = (b[1]*b[3] - b[0]*b[4]) / dd # y0

# K = np.array([[alph, gam, uc],
# 			[0, bet, vc],
# 			[0, 0, 1]])

W11, W12, W22, W13, W23, W33 = b[0], b[1], b[2], b[3], b[4], b[5]
# Estimate intrinstic parameters
y0 = (W12*W13 - W11*W23)/(W11*W22 - (W12**2))
lambd = W33 - (W13**2 + y0*(W12*W13 - W11*W23))/W11
alpha_x = np.sqrt(lambd/W11)
alpha_y = np.sqrt(lambd*W11/(W11*W22 - W12**2))
s = -((W12*(alpha_x**2)*alpha_y)/lambd)
x0 = s*y0/alpha_y - W13*(alpha_x**2)/lambd

# define K matrix
K = np.array([[alpha_x, s, x0],
			[0, alpha_y, y0],
			[0, 0, 1]], dtype=np.float64)

Kinv = np.linalg.inv(K)
print 'K'
print K

for hidx, H in enumerate(homographies):
	img_path = img_paths[hidx]
	# H = [h1, h2, h3]
	H = homographies[hidx]
	print 'H'
	print H
	img_points = img_points_list[hidx]
	world_points = world_points_list[hidx]
	h1 = H[:, 0]
	h2 = H[:, 1]
	h3 = H[:, 2]
	t = np.dot(Kinv, h3)
	zeta = 1.0 / np.linalg.norm(np.dot(Kinv, h1))
	if(t[2] < 0): zeta = -1 * zeta
	# print zeta

	R = np.zeros((3, 3))
	Z = np.zeros((3, 4))
	r1 = zeta * np.dot(Kinv, h1)
	r2 = zeta * np.dot(Kinv, h2)
	r3 = np.cross(r1, r2)
	t = zeta * t
	R[:, 0] = r1
	R[:, 1] = r2
	R[:, 2] = r3

	[U, S, E] = np.linalg.svd(R, full_matrices = True)
	R = np.dot(U, E)

	Z[0:3,0:3] = R
	Z[:,-1] = t

	# print Z

	image = np.copy(cv2.imread(img_path))

	## Compute the error
	C_mat = np.dot(K, Z)
	for row in img_points.astype(int):
		cv2.circle(image, tuple(row.tolist()), 3, color = [250, 0, 0], thickness = 1)
	world_points = np.concatenate([world_points, np.zeros((world_points.shape[0], 1))], axis = 1)
	world_points = np.concatenate([world_points, np.ones((world_points.shape[0], 1))], axis = 1)
	pred_img_points = homo_to_real(np.dot(C_mat, world_points.T).T)
	for row in pred_img_points.astype('int'):
		cv2.circle(image, tuple(row.tolist()), 3, color = [0, 25, 255], thickness = 1)
	# print np.append(world_points, np.append(img_points, pred_img_points, axis = 1), axis = 1).astype(int)

	out_path = os.path.join(out_dir, os.path.basename(img_path))
	print out_path
	cv2.imwrite(out_path, image)

