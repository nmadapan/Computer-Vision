import cv2
import numpy as np
import time
from os.path import join, basename, splitext, dirname

def harris_corners(img, sigma = 1.414, thresh = 0.08):
	assert img.ndim == 2, 'img is a 2D ndarray (grayscale image)'

	# Size of Gaussian Kernel
	K = int(2 * np.floor(3 * 1.414 * sigma) + 1)
	# img = cv2.GaussianBlur(img, (K, K), 0)

	img = cv2.GaussianBlur(img, None, sigma)
	dx_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=K)
	dy_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=K)

	img_height, img_width = img.shape[:2]

	Nh = int(5 * sigma)
	if(Nh % 2 == 0): Nh += 1

	df_idx = (Nh-1)/2

	H = np.zeros((img_height-2*df_idx, img_width-2*df_idx))

	nms_win_size = 31 # non maximum suppression window size.
	nms_half = (nms_win_size-1)/2

	desc_size = 8 # window size is 31

	kps = []
	features = []

	for zr_idx, row_idx in enumerate(range(df_idx, img_height - df_idx)):
		row_ids = range(row_idx-df_idx, row_idx+df_idx)
		for zc_idx, col_idx in enumerate(range(df_idx, img_width - df_idx)):
			col_ids = range(col_idx-df_idx, col_idx+df_idx)
			M = img[row_ids, col_ids]
			dMx = dx_img[row_ids, col_ids]
			dMy = dy_img[row_ids, col_ids]

			c11 = np.sum(np.sum(dMx**2))
			c22 = np.sum(np.sum(dMy**2))
			c12 = np.sum(np.multiply(dMx, dMy).flatten())
			c21 = c12

			trace = c11 + c22
			det = c11 * c22 - c12 * c21

			if(trace < 1e-5): continue

			beta = det / (trace * trace) # Min = 0, Max = 0.25. Higher the better.

			if(beta > thresh):
				H[zr_idx, zc_idx] = beta

	## Non Maximum supression
	for zr_idx, row_idx in enumerate(range(df_idx, img_height - df_idx)):
		for zc_idx, col_idx in enumerate(range(df_idx, img_width - df_idx)):

			# if did not pass previous threshold, continue
			if(H[zr_idx, zc_idx] == 0): continue

			# Boundary conditions. Ignore points on the boundary.
			if(zr_idx - nms_half < 0 or zc_idx - nms_half < 0): continue
			if(zr_idx + nms_half > img_height-1 or \
				zc_idx + nms_half > img_width-1): continue

			nms_win = H[zr_idx - nms_half:zr_idx + nms_half, \
						zc_idx - nms_half:zc_idx + nms_half]

			# if beta is max in that window, then consider. Ignore otherwise.
			if(np.max(nms_win[:]) == H[zr_idx, zc_idx]):
				kps.append([row_idx, col_idx])
				features.append(img[row_idx-desc_size:row_idx+desc_size, \
					col_idx-desc_size:col_idx+desc_size].flatten().tolist())

	return np.float32(kps), np.float32(features)

def extract_kps(image, ftype = 'sift', sigma = 1.414):
	# ftype - feature type 'sift' or 'surf'
	# image - np.ndarray (_ x _ x 3)
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if(ftype.lower() == 'sift'):
		descriptor = cv2.xfeatures2d.SIFT_create()
		# keypoints (cv2.KeyPoint object) and features (ndarray).
		(kps, features) = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])
	elif ftype.lower() == 'surf':
		descriptor = cv2.xfeatures2d.SURF_create()
		# keypoints (cv2.KeyPoint object) and features (ndarray).
		(kps, features) = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])
	elif ftype.lower() == 'harris':
		start = time.time()
		(kps, features) = harris_corners(gray, sigma = sigma)
		print 'Harris corners: %.02f secs'%(time.time()-start)

	# kps: ndarray (_ x 2); features: ndarray (_ x 128)
	return (kps, features)

def mean_normalize(M, axis = 0):
	# Mean normalize matrix M (_ x k) in rows
	if(axis == 1): M = M.transpose() # (k x _)
	M -= np.mean(M, axis = 0)

	norms = np.linalg.norm(M, axis = 0)
	norms[norms == 0.0] = 1e-10

	M /= norms
	if(axis == 1): M = M.transpose()
	return M

def dist_mat_vec(M, vec, method = 'ncc'):
	# M : ndarray ( _ x k); vec: (1 x k)
	if(method.lower() == 'ssd'):
		return np.linalg.norm(M - vec, axis = 1)
	elif(method.lower() == 'ncc'):
		# Mean Normalizing rows of M
		M = mean_normalize(M, axis = 1)
		# Mean normalizing vec
		vec = vec - np.mean(vec)
		vect = vec / np.linalg.norm(vec)
		return np.dot(M, vec)
	elif(method.lower() == 'dot'):
		return np.dot(M, vec)

def dist_mat_mat(M1, M2, method = 'ncc'):
	# M1, M2 --> ndarray (y1 x k) and (y2 x k)
	D = np.zeros((M1.shape[0], M2.shape[0]))
	if(method.lower() == 'ncc'):
		M1 = mean_normalize(M1, axis = 1)
		M2 = mean_normalize(M2, axis = 1)
		method = 'dot'
	for idx2 in range(M2.shape[0]):
		D[:, idx2] = dist_mat_vec(M1, M2[idx2, :], method = method)
	return D

def filter_kps(kpA, kpB, featuresA, featuresB, method = 'ncc', thresh = 0.97):
	print len(featuresA), len(featuresB)
	if(method.lower() == 'ncc'):
		featuresA = mean_normalize(featuresA, axis = 1)
		featuresB = mean_normalize(featuresB, axis = 1)
		method = 'dot'

	matches = []
	for idxB in range(featuresB.shape[0]):
		temp = dist_mat_vec(featuresA, featuresB[idxB, :], method = method)
		temp[temp<thresh] = 0.0
		if(np.max(temp) == 0.0): continue
		else: matches.append((np.argmax(temp), idxB))

	return matches

def draw_matches(imageA, imageB, kpsA, kpsB, matches):
	# initialize the output visualization image
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB

	# loop over the matches
	for queryIdx, trainIdx in matches:
		# only process the match if the keypoint was successfully
		# matched
		# draw the match
		ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
		ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
		cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

	# return the visualization
	return vis

def run(image1_path, image2_path, ftype = 'sift', method = 'ncc', \
	thresh = 0.97, sigma = 1.414):
	img1 = cv2.imread(image1_path)
	img2 = cv2.imread(image2_path)

	kps1, features1 = extract_kps(img1, ftype = ftype, sigma = sigma)
	kps2, features2 = extract_kps(img2, ftype = ftype, sigma = sigma)

	start = time.time()
	matches = filter_kps(kps1, kps2, features1, features2, method = method, thresh = thresh)
	print 'Filter Kps: %.02f secs'%(time.time()-start)

	vis = draw_matches(img1, img2, kps1, kps2, matches)

	fname = splitext(basename(image1_path))[0] + '_' + splitext(basename(image2_path))[0]
	fname = fname + '_' + str(ftype) + '_' + str(int(sigma*1000)) + '_' + method + '.jpg'

	fname_path = join(dirname(image1_path), fname)
	print 'Writing to: ', fname_path

	cv2.imwrite(fname_path, vis)
	cv2.imshow('Visualization', vis)
	cv2.waitKey(0)

if __name__ == '__main__':
	img1_path = 'pair1/1.jpg'
	img2_path = 'pair1/2.jpg'

	ftype = 'harris'
	method = 'ncc'
	thresh = 0.999 # SURF 0.9999

	vis = run(img1_path, img2_path, ftype = ftype, method=method, thresh = thresh)
