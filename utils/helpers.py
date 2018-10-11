import cv2
import numpy as np
import os, time, sys

from NonlinearLeastSquares import NonlinearLeastSquares as NLS

hvars = ['h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32']
Nx = '(h11*{0}+h12*{1}+h13)'
Ny = '(h21*{0}+h22*{1}+h23)'
D = '(h31*{0}+h32*{1}+1)'

def mosaic_two_images(img1, img2, H):
    if(isinstance(img1, str)):
        img1 = cv2.imread(img1)
    if(isinstance(img2, str)):
        img2 = cv2.imread(img2)

    ## H: 2 --> 1
    ## Hinv: 1 --> 2

    ## img1 is assumed to be on the left and img2 on the right.
    Hinv = hinv(H)

    img2_cpts, t_img2_cpts = get_pts(img2.shape, H = H, corners = True)
    img1_cpts, _ = get_pts(img1.shape, H = H, corners = True)
    xmin, ymin = np.min(np.append(t_img2_cpts, img1_cpts, axis = 0), axis = 0)
    xmax, ymax = np.max(np.append(t_img2_cpts, img1_cpts, axis = 0), axis = 0)

    print 't_img2_cpts'
    print t_img2_cpts

    print 'xmin, ymin: ', xmin, ymin
    print 'xmax, ymax: ', xmax, ymax

    if(xmin < 0):
        t_img2_cpts[:, 0] -= xmin
        W_img = xmax - xmin
    else:
        W_img = xmax
        xmin = 0
    if(ymin < 0):
        t_img2_cpts[:, 1] -= ymin
        H_img = ymax - ymin
    else:
        H_img = ymax
        ymin = 0

    final_img = np.zeros((H_img+1, W_img+1, 3), dtype = np.uint8)

    print final_img.shape

    print 't_img2_cpts'
    print t_img2_cpts

    print 'xmin, ymin: ', xmin, ymin
    print 'xmax, ymax: ', xmax, ymax

    _cpts = real_to_homo(t_img2_cpts) # homo. representation
    _cent_cpts = np.mean(_cpts, axis = 0)# centroid of four points

    # Find the four lines of quadrilateral
    lines = [np.cross(_cpts[0], _cpts[1]), np.cross(_cpts[1], _cpts[2]), np.cross(_cpts[2], _cpts[3]), np.cross(_cpts[3], _cpts[0])]

    ## Finding points in the final image that are present in the quadrilateral
    base_bool = np.zeros(final_img.shape[:-1]).flatten() == 0 # True -> inside the quadrilateral
    base_all_pts, _ = get_pts(final_img.shape) # get all pts
    for line in lines:
        sn = int(np.sign(np.dot(_cent_cpts, line)))
        nsn = np.int8(np.sign(np.dot(real_to_homo(base_all_pts), line)))
        base_bool = np.logical_and(base_bool, nsn==sn)
    base_bool = base_bool
    base_bool = np.reshape(base_bool, (final_img.shape[0], final_img.shape[1]))
    row_ids, col_ids = np.nonzero(base_bool)
    des_base_pts = np.array([col_ids, row_ids])
    des_base_pts = des_base_pts.transpose() + [xmin, ymin]
    des_base_pts = des_base_pts.transpose() # 2 x _

    print np.min(des_base_pts, axis = 1)
    print np.max(des_base_pts, axis = 1)

    # Find corresponding points in the template image
    trans_des_base_pts = homo_to_real(np.dot(hinv(H), real_to_homo(des_base_pts.T).T).T).astype(int).T # 2 x _

    # Clip the points
    trans_des_base_pts[0] = np.clip(trans_des_base_pts[0], 0, img2.shape[1]-1)
    trans_des_base_pts[1] = np.clip(trans_des_base_pts[1], 0, img2.shape[0]-1)

    des_base_pts = des_base_pts.transpose() - [xmin, ymin]
    des_base_pts = des_base_pts.transpose()

    # print np.max(trans_des_base_pts, axis = 1)
    # print img2.shape

    final_img[des_base_pts[1], des_base_pts[0], :] = img2[trans_des_base_pts[1], trans_des_base_pts[0], :]

    # img2_pts, t_img2_pts = get_pts(img2.shape, H = H)
    # t_img2_pts -= [xmin, ymin]
    # final_img[t_img2_pts[:,1], t_img2_pts[:, 0]] = img2[img2_pts[:, 1], img2_pts[:, 0]]

    cv2.imshow('Partial image', final_img)
    cv2.waitKey(0)

    img1_pts, _ = get_pts(img1.shape, targ_shap = final_img.shape)
    m_img1_pts = img1_pts - [xmin, ymin]

    # print np.max(m_img1_pts, axis = 0)
    # print final_img.shape

    final_img[m_img1_pts[:, 1], m_img1_pts[:, 0]] = np.uint8(0.5*np.float32(final_img[m_img1_pts[:, 1], m_img1_pts[:, 0]]) + 0.5*np.float32(img1[img1_pts[:, 1], img1_pts[:, 0]]))

    cv2.imshow('Mosaic image', final_img)
    cv2.waitKey(0)

    return final_img

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

def count_inliers(point_corresps, H, delta = 40):
    #####################
    # Input:
    #
    #   point_corresps: np.ndarray of shape _ x 4
    #       Column 0 and 1 correspond to [x_coordinate, y_coordinate] of img1
    #       Column 2 and 3 correspond to [x_coordinate, y_coordinate] of img2
    #       It is point correspondences between two images.
    #
    #   H: Homography (np.ndarray) of shape 3 x 3
    #
    #   delta: decision threshold. Either threshold on SSD or NCC to
    #       determine if a corresp. is an inlier or outlier.
    #       Default value is 40 pixels.
    #
    # Return:
    #
    #   inlier_sz: size of the inlier set. No. of points that are inliers.
    #
    #   inlier_ids: a np array containing indices of points in inlier set
    #
    #####################

    pts1 = point_corresps[:,:2]
    pts2 = point_corresps[:,2:]

    homo_pts2 = real_to_homo(pts2)
    trans_homo_pts2 = np.dot(H, homo_pts2.transpose())
    trans_pts2 = homo_to_real(trans_homo_pts2.transpose())

    err = np.linalg.norm(pts1 - trans_pts2, axis = 1)

    inlier_sz = int(np.sum(err < delta))

    return inlier_sz, np.nonzero(err < delta)[0]

def ransac(point_corresps, param_p = 0.99, eps = 0.1, param_n = 8, delta = 40):
    #####################
    # Input:
    #
    #   point_corresps: np.ndarray of shape _ x 4
    #       Column 0 and 1 correspond to [x_coordinate, y_coordinate] of img1
    #       Column 2 and 3 correspond to [x_coordinate, y_coordinate] of img2
    #       It is point correspondences between two images.
    #
    #   p: prob. that at least one of N trials will be free of outliers.
    #       Default value is 0.99/
    #
    #   eps: prob. that a pt. corresp. is an outlier
    #
    #   n: min. no. of point correspondences needed to estimate the homography
    #
    #   delta: decision threshold. Either threshold on SSD or NCC to
    #       determine if a corresp. is an inlier or outlier.
    #       Default value is 40 pixels.
    #
    # Return:
    #   H: Homography from 2 --> 1. np.ndarray of shape 3 x 3.
    #
    #   new_matches: Point correspondences of points in inlier set.
    #       Datatype is similar to point_corresps but with only inliers.
    #####################

    # Determine num_trials (N). No. of trials or times we need to run RANSAC
    #   so that at least one trial will contain all inliers
    N = np.int16(np.log(1 - param_p) / np.log(1 - (1 - eps)**param_n))

    # thresh_inlier_sz (M)
    #   A minimum size of inlier set that is acceptable.
    n_total = len(point_corresps)
    M = np.int((1 - eps) * n_total)

    print 'Len. of point_corresps: ', len(point_corresps)
    print 'No. of trials (N): ', N
    print 'Min. acceptable size of inlier set (M): ', M

    trial_info = []

    for tr_idx in range(N):
        ## Find 'param_n' point correspondences at random
        tr_match_ids = np.random.randint(0, n_total, param_n)
        # If point correspondences repeat, try until you get unique ids.
        while(len(tr_match_ids) < param_n):
            tr_match_ids = np.random.randint(0, n_total, param_n)

        ## Find homography with the obtained point correspondences
        tr_matches = point_corresps[tr_match_ids, :]
        tr_H, _ = find_homography_2d(tr_matches[:,:2], tr_matches[:,2:])

        ## Find the size of inlier set
        inlier_sz, _ = count_inliers(point_corresps, tr_H, delta = delta)

        # If size of inlier set exceed M, store the trial information.
        if(inlier_sz >= M):
            trial_info.append((inlier_sz, tr_match_ids))

    if len(trial_info) == 0: return None, None
    # Find the inlier set with maximum inlier size
    inlier_sz_list, inlier_pt_ids = zip(*trial_info)
    best_inlier_tr_idx = np.argmax(inlier_sz_list)
    best_inlier_ids = inlier_pt_ids[int(best_inlier_tr_idx)]

    print '% of inliers:', np.max(inlier_sz_list)/float(n_total)

    ## Estimate the homography with best inlier ids (only param_n corresp.)
    temp_matches = point_corresps[best_inlier_ids, :]
    temp_H, _ = find_homography_2d(temp_matches[:,:2], temp_matches[:,2:])

    ## Find all inlier ids and estimate the homography
    _, all_inlier_ids = count_inliers(point_corresps, temp_H, delta = delta)
    new_matches = point_corresps[all_inlier_ids, :]
    H, _ = find_homography_2d(new_matches[:,:2], new_matches[:,2:])

    return new_matches, H

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
