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
    npz_path = img_path[:-4]+'.npz'
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
        np.savez(npz_path, mps = np.array(mps), size_info = {'x_max': img.shape[1], 'y_max': img.shape[0]})
    return np.load(npz_path)
