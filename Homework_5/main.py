import cv2
import numpy as np
import time
from os.path import join, basename, splitext, dirname
import sys
from glob import glob

sys.path.insert(0, r'..\utils')
from helpers import *

def extract_kps(image, ftype = 'sift', sigma = 1.414):
    ################
    # Description:
    #   Find interest points (keypoints) and descriptors
    # Input:
    #   image: RGB image. 3D ndarray (H x W x 3).
    #   ftype: 'sift' or 'surf'
    #   sigma: scale applied to the image
    # Output:
    #   A tuple (keypoints, features)
    #   keypoints: ndarray (Z x 2). each row has (row_idx, col_idx)
    #   features:  ndarray (Z x desc_size**2)
    #   Z is no. of interest points
    ################

    ## Assertion
    assert image.ndim == 3, 'img is a 3D ndarray (RGB image: H x W x 3)'

    ## convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if(ftype.lower() == 'sift'):
        descriptor = cv2.xfeatures2d.SIFT_create()
        # kps (cv2.KeyPoint object) and features (ndarray).
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
    elif ftype.lower() == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
        # kps (cv2.KeyPoint object) and features (ndarray).
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
    else:
        raise ValueError('Unknown feature type')

    # kps: ndarray (Z x 2); features: ndarray (Z x 128)
    return (kps, features)

def mean_normalize(M, axis = 0):
    ## First, substract mean and next, normalize the rows/columns.
    # Mean normalize matrix M (_ x k) in rows
    if(axis == 1): M = M.transpose() # (k x _)
    M -= np.mean(M, axis = 0)

    norms = np.linalg.norm(M, axis = 0)
    norms[norms == 0.0] = 1e-10

    M /= norms
    if(axis == 1): M = M.transpose()
    return M

def dist_mat_vec(M, vec, method = 'ncc'):
    # Compute distance between each row of 'M' with 'vec'
    # method: 'ncc', 'dot', 'ssd'
    # M : ndarray ( _ x k); vec: (1 x k)
    # Returns a 1D numpy array of distances.
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
    # Returns y1 x y2 ndarray with the distances.
    # If y1 and y2 are huge, it might run into MemoryError
    D = np.zeros((M1.shape[0], M2.shape[0]))
    if(method.lower() == 'ncc'):
        M1 = mean_normalize(M1, axis = 1)
        M2 = mean_normalize(M2, axis = 1)
        method = 'dot'
    for idx2 in range(M2.shape[0]):
        D[:, idx2] = dist_mat_vec(M1, M2[idx2, :], method = method)
    return D

def filter_kps(kpA, kpB, featuresA, featuresB, method = 'ncc', thresh = 0.97):
    # Filter the keypoints and the descriptors by thresholding.
    # Returns the matches. List of tuples. (a_idx, b_idx). Point correspondences.
    print len(featuresA), len(featuresB)
    if(method.lower() == 'ncc'):
        featuresA = mean_normalize(featuresA, axis = 1)
        featuresB = mean_normalize(featuresB, axis = 1)
        method = 'dot'

    matches = []
    for idxB in range(featuresB.shape[0]):
        temp = dist_mat_vec(featuresA, featuresB[idxB, :], method = method)
        temp[temp<thresh] = 0.0
        ## Append the ones that pass the threshold
        if(np.max(temp) == 0.0): continue
        else: matches.append((np.argmax(temp), idxB))

    return matches

def draw_matches_one_to_one(imageA, imageB, kpsA, kpsB):
    ########
    # kpsA and kpsB should have same no. of rows.
    # There is one to one correspondence between rows of kpsA and kpsB
    ########
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ptA, ptB in zip(kpsA, kpsB):
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]) + wA, int(ptB[1]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis, ptA, ptB, color, 2)

    # return the visualization
    return vis

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
        color = tuple(np.random.randint(127, 255, 3).tolist())
        cv2.line(vis, ptA, ptB, color, 1)

    # return the visualization
    return vis

def run(image1_path, image2_path, ftype = 'sift', method = 'ncc', \
    thresh = 0.97, sigma = 1.414, write_flag = False):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    kps1, features1 = extract_kps(img1, ftype = ftype, sigma = sigma)
    kps2, features2 = extract_kps(img2, ftype = ftype, sigma = sigma)

    # start = time.time()
    matches = filter_kps(kps1, kps2, features1, features2, method = method, thresh = thresh)
    print 'No. of matches: ', len(matches)
    # print 'Filter Kps: %.02f secs'%(time.time()-start)

    vis = draw_matches(img1, img2, kps1, kps2, matches)

    ## Obtain keypoint matches
    matches = np.array(matches)
    kps1, kps2 = np.array(kps1), np.array(kps2)
    ord_kps1 = kps1[matches[:, 0], :]
    ord_kps2 = kps2[matches[:, 1], :]
    # Format of kp_matches: _ x 4 np.ndarray.
    # Columns 0 and 1 for [x1, y1] of image 1
    # Columns 2 and 3 for [x1, y1] of image 2
    kp_matches = np.append(ord_kps1, ord_kps2, axis = 1)

    delta = 0.5
    while True:
        new_kp_matches, H = ransac(kp_matches, delta = delta, eps = 0.20)
        if H is None: delta = delta * 2
        else: break

    new_kps1 = new_kp_matches[:,:2].tolist()
    new_kps2 = new_kp_matches[:,2:].tolist()

    print 'No. matches: ', len(new_kps1)

    print 'Performing LM: '
    lmres = LM_Minimizer(new_kp_matches, H)

    new_H = np.squeeze(np.asarray(lmres['parameter_values']))
    new_H = np.append(new_H, np.array([1])).reshape(3, 3)
    new_H = nmlz(new_H)

    # mosaic_two_images(img1, img2, new_H)
    # mosaic_two_images(img2, img1, hinv(new_H))

    vis = draw_matches_one_to_one(img1, img2, new_kps1, new_kps2)

    #####
    if(write_flag):
        fname = splitext(basename(image1_path))[0] + '_' + splitext(basename(image2_path))[0]
        fname = fname + '_' + str(ftype) + '_' + str(int(sigma*1000)) + '_' + str(int(thresh*10000)) + '_' + method + '.jpg'
        fname_path = join(dirname(image1_path), fname)
        print 'Writing to: ', fname_path
        cv2.imwrite(fname_path, vis)

    return vis, new_H

if __name__ == '__main__':
    base_img_dir = 'pair4'
    img_paths = glob(join(base_img_dir, '*.jpg'))
    if(len(img_paths)%2 == 0): img_paths = img_paths[:-1]

    num_images = len(img_paths)
    mid_id = int(num_images/2)

    ftype = 'sift'
    method = 'ncc'
    thresh = 0.98 # SURF 0.9999
    sigma = 2.00

    # H = [None] * (num_images-1)
    # V = [None] * (num_images-1)
    # for idx in range(num_images-1):
    #     vis, temp_h = run(img_paths[idx], img_paths[idx+1], ftype = ftype, method=method, thresh = thresh, sigma = sigma, write_flag = False)
    #     V[idx] = vis
    #     H[idx] = temp_h

    # np.savez('homo_trans_images.npz', H = H, V = V)

    dic = np.load('homo_trans_images.npz')
    H = dic['H']
    V = dic['V']

    # ## Method 1
    # img_in_1 = mosaic_two_images(img_paths[1], img_paths[0], hinv(H[0]))
    # img_in_2 = mosaic_two_images(img_paths[2], img_in_1, hinv(H[1]))
    # img_in_3 = mosaic_two_images(img_paths[3], img_in_2, hinv(H[2]))

    # img_in_5 = mosaic_two_images(img_paths[5], img_paths[6], H[5])
    # img_in_4 = mosaic_two_images(img_paths[4], img_in_5, H[4])
    # nimg_in_3 = mosaic_two_images(img_in_3, img_in_4, H[3])
    # cv2.imshow('new img_in_3', nimg_in_3)

    ## Method 2
    HM =[None]*num_images
    HM[0] = nmlz(np.dot(np.dot(hinv(H[0]), hinv(H[1])), hinv(H[2]))) # 0 --> 3
    HM[1] = nmlz(np.dot(hinv(H[1]), hinv(H[2]))) # 1 --> 3
    HM[2] = nmlz(hinv(H[2])) # 2 --> 3
    HM[3] = np.eye(3) # 3 --> 3
    HM[4] = H[3] # 4 --> 3
    HM[5] = nmlz(np.dot(H[4], H[3])) # 5
    HM[6] = nmlz(np.dot(np.dot(H[5], H[4]), H[3]))

    print HM[0]
    omg = mosaic_two_images(img_paths[mid_id], img_paths[0], HM[0]) # 0 --> 3
    # omg = mosaic_two_images(omg, img_paths[1], HM[1])
    # omg = mosaic_two_images(omg, img_paths[2], HM[2])
    # omg = mosaic_two_images(omg, img_paths[3], HM[3])
    # omg = mosaic_two_images(omg, img_paths[4], HM[4])
    # omg = mosaic_two_images(omg, img_paths[5], HM[5])
    # omg = mosaic_two_images(omg, img_paths[6], HM[6])
    # vis = run(img1_path, img2_path, ftype = ftype, method=method, thresh = thresh, sigma = sigma, write_flag = False)

    # cv2.imshow('Visualization', vis)
    # cv2.waitKey(0)

