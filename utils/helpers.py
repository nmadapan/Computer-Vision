import cv2
import numpy as np
import os

def find_homography_2d(pts1, pts2):
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

    # Finding the homography. H[3,3] is assumed 1.
    h = np.dot(np.linalg.pinv(A[:,:-1]), -1*A[:,-1])
    h = np.append(h, 1)
    H = np.reshape(h, (3, 3))
    Hinv = np.linalg.inv(H)
    return H, Hinv / Hinv[-1,-1]

def apply_homography(img_path, H):
	H[:,2] = 1.0 # To ensure there is no translation of the image.
	img = cv2.imread(img_path)
	xv, yv = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
	img_pts = np.array([xv.flatten(), yv.flatten()]).T
	trans_img_pts = np.dot(H, real_to_homo(img_pts).T)
	trans_img_pts = homo_to_real(trans_img_pts.T).astype(int)
	W_new, H_new = np.max(trans_img_pts[:,0]), np.max(trans_img_pts[:,1])
	img_new = np.zeros((H_new+1, W_new+1, 3), dtype = np.uint8)

	img_new[trans_img_pts[:,1].tolist(), trans_img_pts[:,0].tolist(), :] = img[img_pts[:,1].tolist(), img_pts[:,0].tolist(), :]

	write_filepath = os.path.join(os.path.dirname(img_path), os.path.basename(img_path).split('.')[0]+'_new2.jpg')
	cv2.imwrite(write_filepath, img_new)

def apply_trans_patch(base_img_path, template_img_path, H):
	## Read images
	base_img = cv2.imread(base_img_path)
	temp_img = cv2.imread(template_img_path)

	xv, yv = np.meshgrid(range(temp_img.shape[1]), range(temp_img.shape[0]))
	temp_img_pts = np.array([xv.flatten(), yv.flatten()]).T
	trans_temp_img_pts = np.dot(H, real_to_homo(temp_img_pts).T)

	trans_temp_img_pts = homo_to_real(trans_temp_img_pts.T).astype(int)
	trans_temp_img_pts[:,0] = np.clip(trans_temp_img_pts[:,0], 0, base_img.shape[1])
	trans_temp_img_pts[:,1] = np.clip(trans_temp_img_pts[:,1], 0, base_img.shape[0])

	base_img[trans_temp_img_pts[:,1].tolist(), trans_temp_img_pts[:,0].tolist(), :] = temp_img[temp_img_pts[:,1].tolist(), temp_img_pts[:,0].tolist(), :]

	write_filepath = os.path.join(os.path.dirname(base_img_path), os.path.basename(base_img_path).split('.')[0]+'_new.jpg')
	cv2.imwrite(write_filepath, base_img)

def real_to_homo(pts):
    # pts is a 2D numpy array of size _ x 2/3
    # This function converts it into _ x 3/4 by appending 1
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis = 1)

def homo_to_real(pts):
    # pts is a 2D numpy array of size _ x 3/4
    # This function converts it into _ x 2/3 by removing last column
    pts = pts.T
    pts = pts / pts[-1,:]
    return pts[:-1,:].T

def save_mps(event, x, y, flags, param):
    fac, mps = param
    if(event == cv2.EVENT_LBUTTONUP):
        mps.append([int(fac*x), int(fac*y)])
        print(int(fac*x), int(fac*y))

def create_matching_points(img_path):
    npz_path = img_path[:-4] + '.npz'
    flag = os.path.isfile(npz_path)
    if(not flag):
        img = cv2.imread(img_path)
        fac = max(float(int(img.shape[1]/960)), float(int(img.shape[0]/540)))
        resz_img = cv2.resize(img, None, fx=1/fac, fy=1/fac, interpolation = cv2.INTER_CUBIC)
        cv2.namedWindow('TempImage')
        mps = []
        cv2.setMouseCallback('TempImage', save_mps, param=(fac, mps))
        cv2.imshow('TempImage', resz_img)
        cv2.waitKey(0)
        np.savez(npz_path, mps = np.array(mps))
    return np.load(npz_path)
