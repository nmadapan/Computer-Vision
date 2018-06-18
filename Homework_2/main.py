import sys, os
import numpy as np
from helpers import *

## Initialization
all_img_paths = ['.\\images\\1.jpg', '.\\images\\2.jpg', '.\\images\\3.jpg', '.\\images\\Seinfeld.jpg']

img_path = '.\\images\\n1.jpg'
img_path_ref = '.\\images\\n2.jpg'

param = {}

img_info = create_matching_points(img_path)
pts = img_info['mps']
param['img1'] = img_info['size_info'].item()

img_ref_info = create_matching_points(img_path_ref)
pts_ref = img_ref_info['mps']
param['img2'] = img_ref_info['size_info'].item()

## Find homography from 2 --> 1
H, Hinv = find_homography_2d(pts, pts_ref)

## Transfrom points in image 2 to image 1 and update image 1
xv, yv = np.meshgrid(range(param['img2']['x_max']), range(param['img2']['y_max']))
img2_pts = np.array([xv.flatten(), yv.flatten()]).T
trans_img2_pts = np.dot(H, real_to_homo(img2_pts).T) # Transform using homography

trans_img2_pts = homo_to_real(trans_img2_pts.T).astype(int)
trans_img2_pts[:,0] = np.clip(trans_img2_pts[:,0], 0, param['img1']['x_max'])
trans_img2_pts[:,1] = np.clip(trans_img2_pts[:,1], 0, param['img1']['y_max'])

## Read images
img1 = cv2.imread(img_path)
img2 = cv2.imread(img_path_ref)

img1[trans_img2_pts[:,1].tolist(), trans_img2_pts[:,0].tolist(), :] = img2[img2_pts[:,1].tolist(), img2_pts[:,0].tolist(), :]

write_filepath = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)[:-4]+'_new.jpg')
cv2.imwrite(write_filepath, img1)

