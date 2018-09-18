import sys, os
import numpy as np
from glob import glob

sys.path.insert(0, os.path.join('..', 'utils'))
from helpers import *

import time

## Initialization
base_img_path = os.path.join('.', 'images')
all_img_paths = [os.path.join(base_img_path,'1.jpg'), os.path.join(base_img_path,'2.jpg'), os.path.join(base_img_path,'3.jpg'), os.path.join(base_img_path,'4.jpg'), os.path.join(base_img_path,'5.jpg'), os.path.join(base_img_path,'6.jpg'), os.path.join(base_img_path,'ref.jpg')]

ref_img_size = (10, 10, 3)
clear_npz = True
vp_suff = '_vp'
gh_suff = '_gh'

if clear_npz:
    for _file in glob(os.path.join(base_img_path, '*.npz')):
        os.remove(_file)

M, N, _ = ref_img_size
cv2.imwrite(all_img_paths[-1], 255*np.ones(ref_img_size, dtype=np.uint8))

img_path = all_img_paths[2]

img_path_ref = all_img_paths[2]
pts = create_matching_points(img_path, suff = '_p2p')['mps']
pts_ref = np.array([[0,0],[N-1,0],[N-1,M-1],[0,M-1]])
## Point to point correspondence
H, Hinv = find_homography_2d(pts, pts_ref)
apply_homography2(img_path, Hinv, num_partitions = 4, suff = '_p2p')

# # Vanishing line approach - Homework 3
# # Two step approach
# pts = create_matching_points(img_path, suff = vp_suff)['mps']
# H_vl = find_homography_vl(pts)
# print H_vl

# apply_homography2(img_path, H_vl, num_partitions = 4, suff = vp_suff)
# vp_img_path = os.path.splitext(img_path)[0] + vp_suff + '.jpg'
# pts_vl = create_matching_points(vp_img_path, suff = '_')['mps']
# H_af = find_homography_af(pts_vl)
# apply_homography2(img_path, np.dot(hinv(H_af), H_vl), num_partitions=4, suff = '_out')


# # ## Vanishing line approach - Homework 3
# pts1 = create_matching_points(img_path, suff = gh_suff+'1')['mps']
# # pts2 = create_matching_points(img_path, suff = gh_suff+'2')['mps']
# # pts3 = create_matching_points(img_path, suff = gh_suff+'3')['mps']
# H = find_homography_gh([pts1])
# print H
# print hinv(H)
# apply_homography2(img_path, hinv(H), num_partitions = 4, suff = gh_suff)
