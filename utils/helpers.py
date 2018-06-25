import cv2
import numpy as np
import os, time

def find_homography_vl(pts):
    ## Find homography resulting from vanishing line approach.
    # pts should contain points either in clockwise or anti clockwise order
    # Assertion
    assert isinstance(pts, np.ndarray), 'pts should be a numpy array'
    assert pts.shape[1] == 2, 'pts should have two columns'
    assert pts.shape[0] == 4, 'pts should have four rows'

    pts = real_to_homo(pts)
    line1 = np.cross(pts[0,:], pts[1,:])
    line2 = np.cross(pts[2,:], pts[3,:])
    point1 = np.cross(line1, line2)

    line3 = np.cross(pts[0,:], pts[3,:])
    line4 = np.cross(pts[1,:], pts[2,:])
    point2 = np.cross(line3, line4)

    van_line = np.cross(point1, point2)
    van_line = van_line / van_line[-1]

    H = np.eye(3)
    H[2, 0] = van_line[0]
    H[2, 1] = van_line[1]

    return H

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

    [U, S, V] = np.linalg.svd(A, full_matrices = True)
    h = V.T[:,-1]
    h = h / h[-1]

    # # Finding the homography. H[3,3] is assumed 1.
    # h = np.dot(np.linalg.pinv(A[:,:-1]), -1*A[:,-1])
    # h = np.append(h, 1)

    H = np.reshape(h, (3, 3))
    Hinv = np.linalg.inv(H)
    return H, Hinv / Hinv[-1,-1]

def apply_homography(img_path, H, num_partitions = 1):
    # H[:2,-1] = 0.0 # To ensure there is no translation of the image.
    img = cv2.imread(img_path)
    img[0,:], img[:,0], img[-1,:], img[:,-1] = 0, 0, 0, 0
    xv, yv = np.meshgrid(range(0, img.shape[1], img.shape[1]-1), range(0, img.shape[0], img.shape[0]-1))
    img_pts = np.array([xv.flatten(), yv.flatten()]).T
    trans_img_pts = np.dot(H, real_to_homo(img_pts).T)
    trans_img_pts = homo_to_real(trans_img_pts.T).astype(int)
    W_new = np.max(trans_img_pts[:,0]) - np.min(trans_img_pts[:,0])
    H_new = np.max(trans_img_pts[:,1]) - np.min(trans_img_pts[:,1])
    xmin, ymin = np.min(trans_img_pts[:,0]), np.min(trans_img_pts[:,1])
    img_new = np.zeros((H_new+1, W_new+1, 3), dtype = np.uint8)

    x_batch_sz = int(W_new/float(num_partitions))
    y_batch_sz = int(H_new/float(num_partitions))
    for x_part_idx in range(num_partitions):
        for y_part_idx in range(num_partitions):
            x_start, x_end = x_part_idx*x_batch_sz, (x_part_idx+1)*x_batch_sz
            y_start, y_end = y_part_idx*y_batch_sz, (y_part_idx+1)*y_batch_sz
            xv, yv = np.meshgrid(range(x_start, x_end), range(y_start, y_end))
            xv, yv = xv + xmin, yv + ymin
            img_new_pts = np.array([xv.flatten(), yv.flatten()]).T
            trans_img_new_pts = np.dot(np.linalg.inv(H), real_to_homo(img_new_pts).T)
            trans_img_new_pts = homo_to_real(trans_img_new_pts.T).astype(int)
            trans_img_new_pts[:,0] = np.clip(trans_img_new_pts[:,0], 0, img.shape[1]-1)
            trans_img_new_pts[:,1] = np.clip(trans_img_new_pts[:,1], 0, img.shape[0]-1)
            img_new_pts = img_new_pts - [xmin, ymin]
            # This is the bottle nect step. It takes the most time.
            img_new[img_new_pts[:,1].tolist(), img_new_pts[:,0].tolist(), :] = img[trans_img_new_pts[:,1].tolist(), trans_img_new_pts[:,0].tolist(), :]

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
    trans_temp_img_pts[:,0] = np.clip(trans_temp_img_pts[:,0], 0, base_img.shape[1]-1)
    trans_temp_img_pts[:,1] = np.clip(trans_temp_img_pts[:,1], 0, base_img.shape[0]-1)

    base_img[trans_temp_img_pts[:,1].tolist(), trans_temp_img_pts[:,0].tolist(), :] = temp_img[temp_img_pts[:,1].tolist(), temp_img_pts[:,0].tolist(), :]

    write_filepath = os.path.join(os.path.dirname(base_img_path), os.path.basename(base_img_path).split('.')[0]+'_new.jpg')
    cv2.imwrite(write_filepath, base_img)

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
