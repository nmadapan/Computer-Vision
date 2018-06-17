import sys, os
import numpy as np
from helpers import *

## Initialization
img_path1 = '.\\images\\1.jpg'
pts1 = np.array([[2108, 423],[3310, 537],[3309, 1357],[2150, 1494]])

img_path2 = '.\\images\\2.jpg'
pts2 = np.array([[1585,800],[2975,773],[3000,1520],[1620,1600]])

img_path3 = '.\\images\\3.jpg'
pts3 = np.array([[990,565], [2425,415], [2400,1480],[1020,1400]])

img_path_ref = '.\\images\\Seinfeld.jpg'
pts_ref = np.array([[0,0],[2560, 0],[2560, 1536],[0, 1536]])

img_path, pts = img_path3, pts3

## Read images
img1 = cv2.imread(img_path)
img2 = cv2.imread(img_path_ref)

param = {}
param['img1'] = {'x_max': img1.shape[1], 'y_max': img1.shape[0]}
param['img2'] = {'x_max': img2.shape[1], 'y_max': img2.shape[0]}

## Find homography from 2 --> 1
H, Hinv = find_homography_2d(pts, pts_ref)

## Transfrom points in image 2 to image 1 and update image 1
xv, yv = np.meshgrid(range(param['img2']['x_max']), list(range(param['img2']['y_max'])))
img2_pts = np.array([xv.flatten(), yv.flatten()]).T
trans_img2_pts = np.dot(H, real_to_homo(img2_pts).T) # Transform using homography
trans_img2_pts = homo_to_real(trans_img2_pts.T).astype(int)
trans_img2_pts[:,0] = np.clip(trans_img2_pts[:,0], 0, param['img1']['x_max'])
trans_img2_pts[:,1] = np.clip(trans_img2_pts[:,1], 0, param['img1']['y_max'])
img1[trans_img2_pts[:,1].tolist(), trans_img2_pts[:,0].tolist(), :] = img2[img2_pts[:,1].tolist(), img2_pts[:,0].tolist(), :]

write_filepath = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)[:-4]+'_new.jpg')
cv2.imwrite(write_filepath, img1)

