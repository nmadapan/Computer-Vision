import cv2
import numpy as np
import os, time, sys

def find_homography_gh(pts_list):
    ## Find general homography that removes both projective and affine distortion.
    # pts should contain points either in clockwise or anti clockwise order
    # Assertion
    # assert isinstance(pts, np.ndarray), 'pts should be a numpy array'
    # assert pts.shape[1] == 2, 'pts should have two columns'
    # assert pts.shape[0] == 4, 'pts should have four rows'

    ln_pairs = []
    for pts in pts_list:
        pts = real_to_homo(pts)
        ln_pairs.append((nmlz(np.cross(pts[0,:], pts[1,:])), nmlz(np.cross(pts[1,:], pts[2,:]))))
        ln_pairs.append((nmlz(np.cross(pts[1,:], pts[2,:])), nmlz(np.cross(pts[2,:], pts[3,:]))))
        ln_pairs.append((nmlz(np.cross(pts[2,:], pts[3,:])), nmlz(np.cross(pts[3,:], pts[0,:]))))
        ln_pairs.append((nmlz(np.cross(pts[3,:], pts[0,:])), nmlz(np.cross(pts[0,:], pts[1,:]))))
        ln_pairs.append((nmlz(np.cross(pts[0,:], pts[2,:])), nmlz(np.cross(pts[1,:], pts[3,:]))))
    Y = []
    for line1, line2 in ln_pairs:
        r1 = line1[0]*line2[0]
        r2 = line1[0]*line2[1] + line1[1]*line2[0]
        r3 = line1[1]*line2[1]
        r4 = line1[0]*line2[2] + line1[2]*line2[0]
        r5 = line1[1]*line2[2] + line1[2]*line2[1]
        r6 = line1[2]*line2[2]
        Y.append([r1, r2, r3, r4, r5, r6])
    # print Y
    [_, _, V] = np.linalg.svd(Y, full_matrices = True)
    h = V.T[:,-1]
    h = h / h[-1]
    S = np.array([[h[0], h[1], h[3]],[h[1], h[2], h[4]],[h[3], h[4], h[5]]])
    # print S

    # Find 2 x 2
    [U, D2, V] = np.linalg.svd(S[:-1, :-1], full_matrices = True)

    H = np.eye(3)
    H[:-1,:-1] = np.dot(np.dot(V, np.diag(np.sqrt(D2))), V.T)
    vv = np.dot(np.linalg.inv(H[:-1,:-1]), np.array([h[3], h[4]])).T
    vv = vv / np.linalg.norm(vv)
    H[2,:-1] = vv
    return H

def find_homography_af(pts):
    ## Find homography resulting from vanishing line approach.
    # pts should contain points either in clockwise or anti clockwise order
    # Assertion
    assert isinstance(pts, np.ndarray), 'pts should be a numpy array'
    assert pts.shape[1] == 2, 'pts should have two columns'
    assert pts.shape[0] == 4, 'pts should have four rows'

    pts = real_to_homo(pts)
    ln_pairs = []
    ln_pairs.append((np.cross(pts[0,:], pts[1,:]), np.cross(pts[1,:], pts[2,:])))
    ln_pairs.append((np.cross(pts[1,:], pts[2,:]), np.cross(pts[2,:], pts[3,:])))
    ln_pairs.append((np.cross(pts[2,:], pts[3,:]), np.cross(pts[3,:], pts[0,:])))
    ln_pairs.append((np.cross(pts[0,:], pts[2,:]), np.cross(pts[1,:], pts[3,:])))
    A = []
    for line1, line2 in ln_pairs:
        r1 = line1[0]*line2[0]
        r2 = line1[0]*line2[1] + line1[1]*line2[0]
        r3 = line1[1]*line2[1]
        A.append([r1, r2, r3])
    print A
    [_, _, V] = np.linalg.svd(A, full_matrices = True)
    h = V.T[:,-1]
    h = h / h[-1]
    S = np.array([[h[0], h[1]],[h[1], h[2]]])
    print S
    [_, D2, V] = np.linalg.svd(S, full_matrices = True)
    H = np.eye(3)
    H[0:-1,0:-1] = np.dot(np.dot(V, np.diag(np.sqrt(D2))), V.T)
    return H

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
    print 'van_line: ', van_line

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
    return H, hinv(H)

def apply_homography2(img_path, H, num_partitions = 1, suff = ''):
    img = cv2.imread(img_path)
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
    img = cv2.imread(img_path)
    img[0,:], img[:,0], img[-1,:], img[:,-1] = 0, 0, 0, 0

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

def apply_trans_patch(base_img_path, template_img_path, H, suff = '_fnew'):
    ## Read images
    base_img = cv2.imread(base_img_path)
    temp_img = cv2.imread(template_img_path)

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
